# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [RSI-Anything: Unlocking Recursive Self-Improvement with World Models](https://arxiv.org/abs/2610.13912) | 提出了基于预测性世界模型的通用递归自我改进框架 RSI-Anything，通过世界模型指导行动与学习数据选择，并借助兼容性检查决定更新保留，在代码修复和GUI任务中实现了显著的性能提升。 |
| [^2] | [When VLMs Destroy What They Know: Spatial Representations Fall Below Lexical Priors](https://arxiv.org/abs/2610.13910) | 研究发现视觉语言模型对空间关系的内部表征远不如词汇信息忠实——空间关系虽可在谓词词元层面识别，但在聚合为全局句子表征时会丢失，导致空间表征低于词汇先验，即使模型在组合性基准上表现良好。 |
| [^3] | [Structural Automata for Neural Semantic Parsing](https://arxiv.org/abs/2610.13896) | 提出基于结构自动机的接受路径解码（APD）方法，可在固定长度限制内保证神经语义解析器生成良构的意义表示，且无需回退token，在DRS和AMR解析上显著优于局部掩蔽方法。 |
| [^4] | [Task Context Shapes Component Roles in Language Models](https://arxiv.org/abs/2610.13895) | 该论文揭示了任务上下文会重新分配语言模型中共享注意力头的影响力，表明同一模型组件在不同任务中扮演着不同的功能性角色。 |
| [^5] | [Studying Comics at Scale with Artificial Intelligence: The PAGE Framework and Three Studies of American Comics and Japanese Manga](https://arxiv.org/abs/2610.13840) | 提出PAGE框架，利用人工智能将漫画扫描件自动解析为分格与文本，并借助视觉语言模型按编码手册进行标注和与人工标注对齐，实现了对3929本美国漫画和87卷日本漫画的大规模整体化研究。 |
| [^6] | [OpenPA: Forced-alignment-free Multi-granular Pronunciation Assessment with Mixed-transcript Adaptation](https://arxiv.org/abs/2610.13826) | 提出OpenPA框架，利用Whisper的交叉注意力机制实现无需强制对齐的单词级与话语级联合发音评估，并通过混合转录自适应使单一模型同时支持封闭式和开放式回答场景。 |
| [^7] | [Can Language Models Learn When to Abstain? Policy Learning and Calibrated Answer Selection](https://arxiv.org/abs/2610.13821) | 该论文提出通过训练一个独立的答案选择器来决定语言模型何时作答、何时放弃，其在TriviaQA上显著优于模型自身的置信度过滤和强化学习方法，将覆盖率从20.2%提升至30.9%、错误率从23.0%降至14.5%。 |
| [^8] | [QTestBench: A Test Framework for Evaluating LLM-Generated Quantum Code](https://arxiv.org/abs/2610.13811) | 该论文提出了QTestBench框架，首次通过将大语言模型生成的量子测试判定结果与独立的基准检查器对比，来评估这些测试能否有效区分正确与错误的量子代码实现，而非将测试本身当作评估标准。 |
| [^9] | [Tracing and Taming Hallucinated Terms via Subsequence-Association Filtering and Breadth-Aware Selection for LLM Query Expansion](https://arxiv.org/abs/2610.13810) | 提出无需训练、与检索器无关的查询扩展方法SAFE-QE，基于幻觉的子序列关联解释，通过语料库条件下的跨上下文可复现性评分追踪并过滤幻觉词项，并以广度感知的原则性标准决定保留词项的数量。 |
| [^10] | [Error Rate is Not Enough: Costs of Augmentation in Low-Resource ASR](https://arxiv.org/abs/2610.13807) | 本文通过音素级错误分析发现，低资源ASR中不同数据增强方法的收益与代价互补——SpecAugment虽总体改进最可靠但会加剧声调和腭化错误，基于文本的方法则引入辅音插入错误，两者结合可相互抵消缺陷，且这种互补性可由错误类型的声学属性与训练暴露度预测。 |
| [^11] | [REFLECT: Role-Aware Detection of Sentiment, Emotion, Politeness, and Formality](https://arxiv.org/abs/2610.13797) | 本文提出角色感知的Transformer架构REFLECT，通过将语义角色标注（SRL）配置直接融入自注意力机制，并用方向性注意力建模礼貌与正式度、非方向性注意力建模情感与情绪，从而实现对对话中四种交际含义的联合检测。 |
| [^12] | [Not Just an Evaluation: A Case Study of Discourse Particle Sensitivity in Language Models](https://arxiv.org/abs/2610.13794) | 本研究通过行为与机制双重分析，首次揭示大语言模型与人类相似地偏好话语助词"just"和"only"的出现，且对"just"比"only"更敏感，展现了语言模型对话语中细微语义线索的敏感性。 |
| [^13] | [OccludeBench: Can VLMs Find What's Missing?](https://arxiv.org/abs/2610.13790) | 该论文提出OccludeBench基准，揭示视觉语言模型从阴影、反射等物理线索恢复被遮挡物体的能力（42.7%）显著低于从场景上下文推理的能力（62.7%），且与人类相比存在明显差距，并通过OccludeTrace诊断工具定位了视觉推理的瓶颈环节。 |
| [^14] | [UkrMeme: Multimodal Explainable Persuasion Detection in Ukrainian Memes](https://arxiv.org/abs/2610.13783) | 本文发布了包含200张乌克兰战时梗图的说服技术检测数据集UkrMeme及乌克兰语翻译基准PTM-UA，并评估了视觉-语言模型在零样本多模态设置下的说服检测性能及可解释性方法。 |
| [^15] | [A Privileged Teacher That Only Scores Cannot Search: Critique-Conditioned On-Policy Self-Distillation for Reasoning](https://arxiv.org/abs/2610.13744) | 本文发现自蒸馏中的评分信号既无法定位学生推理中的错误，也无法可靠区分正确与错误的rollout，并指出错误定位本质上是搜索问题，因而提出批评条件化的在策略自蒸馏方法来弥补“只会打分、不会搜索”的评分教师的结构性缺陷。 |
| [^16] | [Enhancing Legal Reasoning in Pre-trained Language Models via Knowledge Graph-Guided Multi-Task Pre-training](https://arxiv.org/abs/2610.13722) | 提出了一种结合下一句预测、知识图谱引导聚类和知识图谱感知段落分类的多任务预训练方法，将结构化法律知识直接编码进模型权重，从而实现无需图遍历的零样本法律推理。 |
| [^17] | [Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy](https://arxiv.org/abs/2610.13708) | Adaptive DHiSparse 在分层稀疏 KV 架构上支持推测解码，并根据负载动态调整 KV 缓存驻留策略，在 100K 上下文、128 并发下相比 HiSparse 将解码吞吐提升至 1.45 倍。 |
| [^18] | [Verifiable Retrieval for Legal Reasoning through Structure-driven Knowledge Graph](https://arxiv.org/abs/2610.13695) | 本文提出利用结构驱动解析为德国法律法典构建知识图谱，通过比较多种基于知识图谱的分块与检索策略，提升了法律推理场景中引用检索的准确性和可验证性。 |
| [^19] | [Retractions Revisited: A Replication, Extension, and Mechanistic Evaluation of Linguistic Obfuscation Effects in Fraudulent Science](https://arxiv.org/abs/2610.13678) | 本研究通过比较800篇欺诈性与800篇真实性科学论文，复制并证实了欺诈论文具有更高语言模糊化的结论，并通过并行中介模型发现“现实监测”机制（体现为形容词使用减少）是解释这一现象的最佳心理机制。 |
| [^20] | [Inject a Turn, Not a Token: Learning Where to Redirect LLM Reasoning for Self-Correction](https://arxiv.org/abs/2610.13672) | 该论文提出无需微调的 UPTURN 自我纠错框架，通过仅用最终答案正确性训练的 13 万参数轻量 MLP 头学习失败推理中的最佳干预位置，并将纠错指令作为新的用户轮次而非行内标记注入，从而有效重定向模型推理，在四个推理模型和六个数学基准上均提升准确率。 |
| [^21] | [A Multi-Agent LLM System for Automated Tuning of Specialized Neural Network Models](https://arxiv.org/abs/2610.13663) | 本文提出一个由分析师、编码员和记录员组成的多智能体LLM框架，在保持整体网络结构不变的前提下自动生成并调优指定关键组件的代码，并可迁移至脉冲神经网络分类与PINN求解高阶微分方程等不同类型模型。 |
| [^22] | [Multi-Stream Attention Residuals](https://arxiv.org/abs/2610.13630) | 提出多流注意力残差（MSAR），将注意力残差泛化为多头深度注意力机制，通过多尺度因果卷积从序列上下文获取流多样性，并经轻量级自适应混合组合各流，以极小的额外开销实现了显著的性能提升。 |
| [^23] | [AMLworld-Compact: Importance-Weighted Downsampling for Cost-Effective LLM Evaluation and Error Diagnosis](https://arxiv.org/abs/2610.13627) | 提出AMLworld-Compact评估集，通过基于监督评分分层的HT-Coreset重要性加权降采样协议，将AMLworld上LLM评估目标减少最高611倍，并利用Horvitz-Thompson权重实现对完整测试集性能指标的无偏估计。 |
| [^24] | [APEX-Harm: Evaluating Agent Safety in Shared Professional Workspaces](https://arxiv.org/abs/2610.13623) | 该论文提出了APEX-Harm基准，在包含上百个文件和多种工具的共享专业工作空间中，通过六类提示与脚本注入攻击评估执行金融、咨询和法律任务的长时程智能体的安全性，对12个前沿模型的测试显示其仍存在显著安全风险。 |
| [^25] | [Training User Simulators with Latent Intrinsic Feedback](https://arxiv.org/abs/2610.13617) | 提出了一种具有潜在内在反馈的用户模拟器，通过在每轮对话中以第一人称视角生成内部想法、反应和评分等潜在评估状态来评估助手，显著提升了模拟的真实性和对助手回复的批判能力。 |
| [^26] | [BLiMPPI: Benchmark of Linguistic Minimal Pairs for Polarity Items](https://arxiv.org/abs/2610.13607) | 本文提出了BLiMPPI——一个涵盖33种语言、包含超过12,000个否定极性项最小对立对的多语言评测基准，揭示了多语言模型在低资源语言以及需要深层句法语义知识的NPI现象上表现最差。 |
| [^27] | [zip2zip++: Towards Inference-Time Token Compression with Minimal Quality Loss](https://arxiv.org/abs/2610.13605) | 该论文提出 zip2zip++，通过分离输入/输出嵌入、修正 RoPE 位置编码、避免合并数字等针对性改进压缩层与 LZW 编解码器，在不修改预训练模型的前提下实现推理时词元压缩，并将质量损失降至最低。 |
| [^28] | [Measuring Artifact Reliance in Machine-Generated Text Detectors](https://arxiv.org/abs/2610.13583) | 该论文揭示机器生成文本检测器在科学文献基准上的近乎完美表现，很大程度上依赖于文档组装时引入的PDF提取伪影等表面格式特征（仅凭这些特征逻辑回归即可超过现有检测器），并提出“格式反事实”方法来区分包含此类信号的基准与依赖此类信号的检测器。 |
| [^29] | [Not All Correct Fixes Run Alike: An Empirical Study of Patch Performance in Repository-Level Issue Resolution](https://arxiv.org/abs/2610.13563) | 该研究将基准自带的 FAIL_TO_PASS 测试重新用于评估补丁的运行时性能，发现通过正确性测试的不同补丁在运行性能上存在显著差异，即并非所有正确的修复运行起来都一样。 |
| [^30] | [Personalization Without Stereotyping: Do Speech LLMs Know When to Turn a Deaf Ear?](https://arxiv.org/abs/2610.13550) | 提出了 BASIS-Bench 基准，通过人口统计稳健性、刻板印象偏见和个性化三个维度区分语音大语言模型的有益个性化与刻板印象行为，评估发现多数模型虽能识别人口统计线索却未能恰当运用。 |
| [^31] | [S-Space: Exploring Spatial Workspace in Multimodal Language Models](https://arxiv.org/abs/2610.13527) | 该论文发现多模态语言模型内部存在一个固有的线性子空间"S-Space"，作为模型的内部空间工作空间，以连续坐标编码物体位置，并支撑感知编码、视角变换和言语输出等空间推理的全过程。 |
| [^32] | [Retrieval-Augmented Multimodal Language Model with Heterogeneous Evidence Integration for Clinical Prognosis](https://arxiv.org/abs/2610.13510) | 该论文提出了PRISM，一种通过双模态人群记忆库在表示空间中检索相似病例嵌入、并融合异构临床证据的检索增强多模态语言模型，在包含1,443个任务的MDS-ED临床预后基准上，于诊断、病情恶化、ICU入院和死亡率预测等多项任务中取得了优异表现。 |
| [^33] | [Proactive Agentic Orchestration](https://arxiv.org/abs/2610.13485) | 本文提出主动式智能体编排框架POrchestra，允许子智能体在执行过程中随时主动请求编排器调整其目标和工具框架，从而无缝适应动态环境，并据此提出了基于事后洞察的在策略编排蒸馏方法来优化编排策略。 |
| [^34] | [A Better Metric Will Not Save You: What You Compare Beats How You Compare It](https://arxiv.org/abs/2610.13476) | 该研究以十五种语言的元音声学组织为基准发现，选择比较哪些类别对表征相似性度量的影响远超度量方法本身——改变所比较的类别会使两个表征间的差距变动超过所测效应的两倍，因此“比较什么”比“如何比较”更重要。 |
| [^35] | [Towards Enriching Design-Time Models with Evidence from Simulation and Operational Activities](https://arxiv.org/abs/2610.13456) | 本文提出一种证据驱动的工作流，系统地将仿真与运行阶段产生的执行证据反馈回设计时模型（如影响模型），从而弥补从系统执行结果到持久设计知识丰富之间的隐式路径。 |
| [^36] | [O\'ido: Open-Vocabulary Speech Recognition on a $5 Microcontroller without a Neural Accelerator](https://arxiv.org/abs/2610.13447) | O'ido在仅5美元、无神经加速器的ESP32-S3微控制器上实现了开放词汇英语语音识别，通过int8量化和内存带宽优化的调度，错误率比厂商识别器低2.3-2.8倍。 |
| [^37] | [Revisiting Accent Classification with DistilHuBERT: Speaker-Disjoint Evaluation and Cross-Corpus Generalization](https://arxiv.org/abs/2610.13440) | 该研究发现文件级评估会大幅高估基于DistilHuBERT的口音分类模型性能，说话人分离评估才能更真实地反映模型对未见说话人的泛化能力，且仅用外部数据训练的模型难以可靠泛化到L2-ARCTIC语料库。 |
| [^38] | [Does Financial Speech Need Continual Learning Regularization? A Study of EWC and mini-KFAC on Earnings Call ASR](https://arxiv.org/abs/2610.13432) | 该研究发现在将Whisper适配到财报电话会议语音识别时，新任务与旧任务之间存在互补性而非灾难性遗忘，因而正则化反而有害；但当确需正则化时，逐层mini-KFAC比对角EWC能取得更低的词错误率和更稳定的优化效果。 |
| [^39] | [Towards the Roof of the World: NLP Progress in Tibetan Languages](https://arxiv.org/abs/2610.13398) | 本文开展了首个经同行评审的藏语NLP系统性文献计量调查，通过分析2004至2026年间140篇出版物，揭示了中国研究人员主导的全流水线研究范式如何在复杂文字与方言多样性等约束下推动藏语NLP的快速发展。 |
| [^40] | [ProgramGym: Scaling Training Environments for Long-Horizon Program Reproduction Agents](https://arxiv.org/abs/2610.13395) | ProgramGym提出了一个由Program-Env和Program-Agent组成的框架，通过反馈驱动的循环从代码仓库自动构建可验证的训练环境，实现了对长周期程序复现智能体的可扩展训练。 |
| [^41] | [MoPrO-SD: Modular Prompt Optimization for Stance Detection](https://arxiv.org/abs/2610.13386) | 提出面向立场检测的模块化提示优化框架MoPrO-SD，将提示分解为功能性语义模块并利用LLM反馈进行顺序优化，同时发布人工标注的疫苗立场检测基准VaccineStance。 |
| [^42] | [From Structure to Validity: Mechanism-Guided Generation and Refinement of LLM Rubrics](https://arxiv.org/abs/2610.13375) | 该研究系统揭示了LLM评分量规设计的有效机制——结构化的标准组织最为关键，并据此提出ARTS-Hybrid方法，通过自适应量规树构建与语义优先、响应校准的精炼策略，同时提升了端到端评估性能与量规的内在质量。 |
| [^43] | [Overcoming Data Imbalances with Expert Selection Biases](https://arxiv.org/abs/2610.13370) | 该论文提出在MoE模型中引入基于类别的专家选择偏差，将不同类别的词元优先路由到专门的专家子集，从而在数据类别不平衡的情况下实现软性类别对齐的专家专门化，有效缓解少数类别容量受限的问题。 |
| [^44] | [Test-Time Memory Construction](https://arxiv.org/abs/2610.13348) | 提出了 T²Mem 测试时记忆构建框架，将记忆构建推迟到推理阶段并沿推理过程逐步进行，以低开销高效回答涉及长交互历史的问题。 |
| [^45] | [Exploring Attention Sinks in Text-to-Image Transformers from Diffusion and Autoregressive Models](https://arxiv.org/abs/2610.13337) | 该论文首次系统性地研究了文本到图像Transformer中（涵盖双向扩散模型与因果自回归模型）注意力汇聚现象的存在性、出现位置及其对生成过程的功能性影响，并发现两类模型的注意力汇聚呈现截然不同的模式。 |
| [^46] | [Taking Tokenization off the Host for Agentic LLM Serving](https://arxiv.org/abs/2610.13325) | 提出 GraphTok，一个将完整分词流水线封装为可重放 CUDA 图并逐位复现参考结果的 GPU 分词器，比 CPU 快 4.7-6.0 倍，可将分词从主机彻底卸载，从而显著降低智能体场景下 LLM 服务的首 token 延迟并提升 GPU 集群可扩展性。 |
| [^47] | [Host-Aware Web Crawling for LLM Pretraining Data](https://arxiv.org/abs/2610.13321) | 该研究发现页面质量约一半的差异源于主机层面，且少量采样页面即可有效估计主机质量，因此基于主机感知的爬虫预算分配策略比图中心性方法更适合收集高质量LLM预训练数据。 |
| [^48] | [Agreement Without Alignment: Label-Free Rubric Clarification Increases LLM-Judge Consensus but Not Agreement with Humans](https://arxiv.org/abs/2610.13303) | 该论文提出一种无标签的评分标准澄清程序，通过将裁判间分歧转化为对争议子标准的重写修复，显著提升了LLM评审团内部的一致性，但发现这种共识的提升并不能转化为与人类评判更高的一致性。 |
| [^49] | [Do Prompt Optimizers Break Requirements They Are Not Scored On? A Preregistered Study](https://arxiv.org/abs/2610.13287) | 预注册研究发现，提示词优化器在提升目标指标的同时确实会破坏未被评分的要求（如引用和格式），但事后分析表明，表面的性能退化主要源于推理过程挤占输出空间导致答案被截断而无法提取，而非真正的内容质量下降。 |
| [^50] | [BoundaryEvo: Data-Model Co-Evolution at the Capability Boundary for Long-Chain Reasoning Distillation](https://arxiv.org/abs/2610.13285) | BoundaryEvo提出一种两阶段框架，先帮助学生模型建立基础推理能力，再使训练数据随学生模型的能力边界协同演化，从而在仅有少量教师轨迹的条件下高效提升小模型的长链推理能力。 |
| [^51] | [Beyond the Final Embedding: Retrieving Memories from Token-Level Evidence](https://arxiv.org/abs/2610.13277) | 该论文提出一种无需额外训练的稀疏词元匹配方法，利用嵌入模型已有的词元级上下文表示捕捉局部证据，并结合整条记忆相似度对记忆排序，在八个基准测试和多个模型家族上显著提升检索效果，还可通过可选的低秩查询适配器进一步增强。 |
| [^52] | [CAST: Credit Alignment for Cross-Tokenizer On-Policy Distillation](https://arxiv.org/abs/2610.13257) | CAST提出了一种跨分词器在线蒸馏中的“移动或保持”信用对齐框架，利用预测不确定性和教师信号平滑度将块级优势精确分配到学生令牌上，并对无教师对应的令牌施加锚定约束。 |
| [^53] | [Vision Transformers Learn Word Meaning from Images Alone](https://arxiv.org/abs/2610.13252) | 仅通过图像自监督预训练的DINOv3视觉Transformer无需任何语言监督，就能从图像中渲染的文字学会按词义组织词语，证明视觉模型可以从纯像素中习得词语语义。 |
| [^54] | [What Do Audio-Language Judges of Generated Music Listen For? Four Probes with Controls from Expert Ratings](https://arxiv.org/abs/2610.13247) | 提出四项带人类评分对照的探针实验，系统检验音频-语言评审模型评估生成音乐时究竟响应音乐、提示还是提问方式，发现移除音频后所有模型表现不超过仅凭提示的预测基线，提示对齐评估甚至低于基线。 |
| [^55] | [How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing](https://arxiv.org/abs/2610.13236) | 该论文首次在黑盒设置下研究AMR解析器的不确定性估计，通过束搜索候选间的相似度来评估解析输出的可靠性，为AMR解析器的评估提供了性能之外的重要补充维度。 |
| [^56] | [SharP: Soft and Hard Prompt-Guided Augmentation with LLM for Low Resource Fake News Detection](https://arxiv.org/abs/2610.13207) | 该论文提出SharP框架，通过软硬提示引导大语言模型进行边界感知的文本生成，为低资源虚假新闻检测生成具有任务特定相关性的合成训练样本，以克服现有数据增强方法生成内容过于笼统的问题。 |
| [^57] | [S$^3$R: Fast and Faithful RL Rollouts via Self-Sparse Speculative Decoding](https://arxiv.org/abs/2610.13202) | S³R提出自稀疏推测解码方法，用稀疏注意力起草token、全注意力验证并采用精确推测采样，在严格保持稠密策略采样分布（保证训练忠实性）的同时大幅加速长上下文RLVR的rollout生成。 |
| [^58] | [JANUS: Enhancing Long-Horizon Autonomous Research with Co-Evolving World Models](https://arxiv.org/abs/2610.13194) | JANUS提出了一种由共同演化的世界模型驱动的“预测-验证-演化”循环框架，通过在执行前预测候选方案的价值提升、信息增益和探索效用来缓解长时程自主研究中的反馈延迟问题，并将反馈经验蒸馏为不断演化的技能。 |
| [^59] | [PreSTO: Predictive Subtree Prefetching for Fast LLM Power Sampling](https://arxiv.org/abs/2610.13171) | PreSTO将未来的Metropolis-Hastings接受/拒绝决策组织成一棵树并批量预取其提议，在可证明保持MH转移核不变的前提下显著加速了LLM幂采样。 |
| [^60] | [Scaling Agentic Data for Long-Horizon Terminal Intelligence](https://arxiv.org/abs/2610.13166) | 提出 TerminalHorizon，一个面向长程终端智能体的全自动化数据合成引擎，通过避免环境合成预先完成关键工作，生成能真正激发智能体探索、实施与验证能力的高质量长程训练数据。 |
| [^61] | [SOAR: Source-Aware Reinforcement Learning with Precise Credit Assignment for Search Agents](https://arxiv.org/abs/2610.13157) | SOAR提出源感知强化学习范式，通过源感知工作流将外部与内部知识解耦为并行可观察的信息流，并结合源一致性奖励实现精确的信用分配，从而有效缓解搜索代理的证据伪造问题。 |
| [^62] | [Les Mis\'erables Dialogue: A film-based network representation of social communication](https://arxiv.org/abs/2610.13148) | 本工作基于1998年电影版《悲惨世界》构建了公开可用的时序文本网络数据集，明确区分每句台词的说话者、受话者、内容与时间，并通过十维社会关系模型给出多层网络表示，同时提供了从叙事素材构建丰富交流数据集的通用方法。 |
| [^63] | [Neurograf v1: An End-to-End Romanian-First Language Model Family](https://arxiv.org/abs/2610.13134) | 该论文提出了Neurograf v1——首个从零开始为罗马尼亚语这一低资源欧洲语言构建的端到端语言模型家族，涵盖635 GB语料库、分词器、五个基础模型、指令微调助手和嵌入模型，验证了低资源语言从零训练大模型的可行性。 |
| [^64] | [Locate, Judge, Read: Role-Specialized Tool Agents over a Knowledge-Point Index for Open-Domain Multimodal Document QA](https://arxiv.org/abs/2610.13128) | 提出按职能分离的KnowForge流水线，通过知识点索引、仅能搜索的定位器、强制第二跳检索的桥接判断器和仅能阅读的深度阅读器等角色专门化工具代理，在开放域多模态文档问答上显著超越了现有检索增强方法。 |
| [^65] | [BenCoDiC: A Large-Scale Contextual Cyberbullying Corpus for Bengali Political News Discourse FULL VERSION](https://arxiv.org/abs/2610.13121) | 该论文构建并公开了首个大规模、语境化、多维度的孟加拉语政治新闻网络欺凌语料库BenCoDiC，包含50,250条保留评论回复结构的文本实例，通过九维星型模式组织数据，为网络欺凌检测与欺凌者行为网络研究提供有力支持。 |
| [^66] | [From 2.1M Hours of Raw Broadcast Audio to Finnish Speech SSL: Confidence Filtering and Architectural Analysis at Scale](https://arxiv.org/abs/2610.13102) | 该论文提出以ASR置信度作为数据质量过滤标准，从210万小时芬兰语广播音频中筛选高质量语音进行自监督学习预训练，在更小的训练集上相比基于LID的过滤方法取得3-7%的相对WER提升，并系统比较了四种SSL架构的性能。 |
| [^67] | [KV-Repurpose: A Testbed for Evaluating Non-Prefix KV-Cache Reuse](https://arxiv.org/abs/2610.13060) | 提出基于 vLLM 的开源测试平台 KV-Repurpose，用于开发和评估超越精确前缀限制的分段式 KV 缓存复用，在 RAG 风格工作负载下显著提升缓存命中率并基本保持单文档任务的输出质量。 |
| [^68] | [CliQ-RRG: Clinical-Knowledge Guided Disease-aware Visual-Textual Alignment for QA-Style Radiology Report Generation](https://arxiv.org/abs/2610.13059) | CliQ-RRG提出了一种两阶段框架，通过疾病感知的视觉-文本对齐与先验引导注意力捕获多视图依赖，并利用GraphRAG检索临床知识进行增强，从而生成可解释且符合临床实际的问答式放射报告。 |
| [^69] | [Travel Back To Move Forward: Retrospective Exploration and Evolving for Long Horizon Agents](https://arxiv.org/abs/2610.13058) | 提出ReTRAIL框架，使智能体能够将环境回退至较早状态进行回顾性探索，同时保留已学知识与有效进展，并通过状态锚定GRPO和分支推进奖励两种互补信号训练策略，帮助长时程智能体主动从死胡同轨迹中恢复。 |
| [^70] | [CoBe: A Benchmark For Conversational Counterfactual Text Editing](https://arxiv.org/abs/2610.13057) | 该论文提出了CoBe基准，用于评估模型的对话式反事实文本编辑能力，要求模型通过因果理解判断假设性变化会影响的范围，在修改文本的同时保留未受因果影响的事实，而非仅生成统计上相关联的文本。 |
| [^71] | [RAMP: Recurrent Adaptive Mixed Precision for Efficient Prefill in Recurrent-Depth Language Models](https://arxiv.org/abs/2610.13041) | 该论文提出RAMP方法，通过揭示“越早引入的量化误差会被越多后续循环步骤衰减”这一传播规律，将量化计算自适应地放置在合适的循环步骤上，从而在保持精度的同时高效加速循环深度语言模型的长上下文预填充。 |
| [^72] | [GeoGR: Spatial-Semantic Token Compression and Group Routing for Efficient 3D VLMs](https://arxiv.org/abs/2610.13035) | 提出GeoGR两阶段框架，通过将跨视角观察压缩为3D体素令牌（GeoSemZip）并在LLM层间进行令牌分组路由，联合压缩场景表示与视觉计算，从而实现高效的3D视觉语言模型推理。 |
| [^73] | [Evo$^2$: Evolving the Evolver through Weight-Harness Co-Evolution](https://arxiv.org/abs/2610.13025) | 提出了权重-框架协同进化框架 Evo²，让单一策略同时担任提议者和反思者来完成框架的自我修改，并利用框架修改及其执行结果指导模型权重训练，从而实现智能体权重与框架的联合进化。 |
| [^74] | [EmbodiedCLBench: Evaluating Continual Learning for Self-Evolving Embodied Agents](https://arxiv.org/abs/2610.13021) | 提出了首个具身仿真持续学习基准 EmbodiedCLBench，以组合泛化为核心，评估自我进化智能体能否将基础任务中积累的经验迁移并重组以解决未见的高级任务。 |
| [^75] | [Evaluating LLM Creativity through Long-Tail Performance](https://arxiv.org/abs/2610.12998) | 该论文提出将大语言模型的创造力形式化为参考输入-输出分布长尾部分的任务特定表现，并引入q@Tail(α)指标，实证发现基于较高尾部子集的模型排名与人类创造力判断更为一致。 |
| [^76] | [Benchmarking Embedding Geometries and Clustering Algorithms for Unsupervised Patent Claim Classification](https://arxiv.org/abs/2610.12997) | 双曲空间嵌入能显著提升跨分类代码的现有技术检索能力——双曲VAE比欧几里得VAE多检索到52%的真正跨类别现有技术，尽管它与官方专利分类系统的一致性更低。 |
| [^77] | [What Should SFT Learn? Conditional Information Gain for Generalizable Post-Training](https://arxiv.org/abs/2610.12995) | 提出CIG-SFT信息论框架，用条件信息增益而非模型兼容性来决定SFT应学习哪些词元，从而提升大语言模型后训练的泛化能力。 |
| [^78] | [Language Models Have Interpretable Narrative Mechanisms](https://arxiv.org/abs/2610.12989) | 该研究在多个语言模型中识别出激活与文本叙事性相关的注意力头、MLP神经元和稀疏自编码器特征，并通过消融实验提供了这些叙事相关组件因果性地影响模型行为（尤其是故事补全和叙事问答任务）的证据。 |
| [^79] | [Forecasting Corpus-Sampling Disagreement in Masked Text Reconstruction](https://arxiv.org/abs/2610.12987) | 该论文推导出一个一阶公式，可仅凭参考语料库预测两个独立语料库上训练的掩码文本重建预测器之间的期望分歧，并发现语料库扩大四倍仅使分歧降低5–40%，远低于1/n缩放律的预期，其根源在于罕见上下文的平滑效应。 |
| [^80] | [SmartSalesBench: A Resistance-Aware Multi-Turn Benchmark for Evaluating LLM Sales Agents](https://arxiv.org/abs/2610.12975) | SmartSalesBench是首个专为评估LLM销售代理应对具有心理抗拒性买家而设计的多轮对话基准测试，通过13,680个会话和三层级九项评估指标，颠覆了传统对话基准假设用户合作的评估范式。 |
| [^81] | [MVGraph: Benchmarking LVLMs on Evidence Composition across Multi-View Visual Graph](https://arxiv.org/abs/2610.12961) | 提出MVGraph基准，通过8千个实例和8个图论任务评测大型视觉语言模型在多视图图推理中跨视图证据对齐与组合的能力，并将其分解为感知、对齐、组合和推理四种互补能力。 |
| [^82] | [Surfing the Activation Wave: Analyzing the Information and Geometry of LLM Hidden States for Speculative Decoding](https://arxiv.org/abs/2610.12939) | 该论文提出了一个基于矩阵熵信息量分析与中心化核对齐（CKA）多样性度量的原则性框架，用于为大语言模型推测解码选择多层隐藏状态，并揭示了解码器表示中的几何与信息分区特性。 |
| [^83] | [SEEDS: Preview-Guided Evidence Admission for Search Agents](https://arxiv.org/abs/2610.12932) | 该论文提出SEEDS框架，针对搜索智能体中被忽视的“证据准入”问题，通过轻量级预览引导智能体根据当前推理状态选择性地决定哪些检索候选文档应进入其推理上下文，摆脱了传统固定或基于排名的文档纳入策略。 |
| [^84] | [Rethinking the Roles of Diffusion and Decoding: Predictive-State Flows for Parallel Language Generation](https://arxiv.org/abs/2610.12898) | 该论文提出预测态流（PSF），重新划分扩散先验与解码器的生成职责，让整流流先验仅生成基于前文的高层预测态，再由块因果解码器并行完成词元级选择，从而实现并行语言生成。 |
| [^85] | [Temporal Evidence Forests for Long-Horizon Multi-Party Conversational Memory](https://arxiv.org/abs/2610.12874) | 该论文提出T-Mem——一种基于时间结构的记忆框架，通过“时间证据森林”同时恢复局部时间区间内密集集中的证据并重构完整的事件演进顺序，从而克服现有记忆框架仅将时间作为元数据或检索信号、无法追踪长程多方对话记忆动态演变的局限。 |
| [^86] | [Time Delay System for Sequence Modeling](https://arxiv.org/abs/2610.12864) | 该论文提出基于非线性时滞系统的序列建模新框架，通过可微分语义时滞估计器依据当前与历史输入的语义相似度动态估计有效时滞，从而实现显式、可解释的内容感知时间回溯机制。 |
| [^87] | [Unreliable Narrator Framing as a Cost-effective Mechanism for Satire Explanation of Australian and American Satirical News](https://arxiv.org/abs/2610.12863) | 该研究提出以不可靠叙述者理论为框架，利用开放权重模型低成本地为澳大利亚和美国讽刺新闻生成解释，并贡献了新的数据集与评估基准。 |
| [^88] | [Fear Is Not Always Fear: Variation in Emotion Label Equivalence Across Datasets](https://arxiv.org/abs/2610.12841) | 该论文实证表明，GoEmotions、XED和BRIGHTER数据集中同名情感标签（如恐惧、惊讶）的含义并不等价——由标签分布差异导致的规模不匹配可通过校准大幅修复，但模型跨语料迁移会损失约55%的判别能力，因此训练数据的选择会直接改变情感分类的研究结论。 |
| [^89] | [COFFEQA: Counterfactual-Guided Few-Shot Distillation for LLM Question Answering](https://arxiv.org/abs/2610.12837) | 提出COFFEQA方法，通过反事实引导为少样本蒸馏中教师决策边界附近的不确定样本提供额外监督，从而提升大语言模型问答任务的蒸馏效果。 |
| [^90] | [When Charts Deceive Web Agents: A Paired Benchmark for Misleading Visualization-Grounded Tasks](https://arxiv.org/abs/2610.12823) | 该论文提出了首个成对可执行基准MisVisAgentBench，通过280个仅图表证据不同的“干净-误导”匹配任务对，评估误导性可视化对网页智能体下游执行行为的影响，并提供诊断框架将误导引发的失败与基线任务难度区分开来。 |
| [^91] | [VEPA: Verified Evolving Formal Policy for Lifelong LLM Agent Guardrails](https://arxiv.org/abs/2610.12813) | 提出VEPA框架，将形式化防护栏的确定性执行能力与从部署经验中持续学习和演化安全策略的机制相结合，为LLM智能体提供终身的、可验证的自适应安全防护。 |
| [^92] | [People Making Conversation: Evaluating How Naturally Speech Models Converse](https://arxiv.org/abs/2610.12802) | 该论文提出了一种新的评估方法，将实时语音系统置于真实对话的停顿和话轮结束点，通过对比模型行为与真人实际表现来衡量语音模型“何时开口说话”这一决策的自然程度。 |
| [^93] | [Meta-Reflective Evolution Effectively Improves Production AI Systems](https://arxiv.org/abs/2610.12794) | 提出基于“元反思”机制的进化优化器CoCoEvolve，通过从完整进化历史中检索证据并探测运行中的系统来指导变异，在DABStep上以92.8%的难题准确率登顶排行榜，有效改进了数据智能体、LLM数据流水线和数据工程工作流等生产级AI系统。 |
| [^94] | [Large Language Models, Machine-Assisted Cultural Analytics, and the Agentic Turn](https://arxiv.org/abs/2610.12792) | 本章指出，大语言模型及其驱动的智能体系统扩展了文化分析在规模、方法范围和任务委托方面的可能性，但测量结果支撑文化论断的基本逻辑基本保持不变。 |
| [^95] | [REDaCT: Collapse-Free Unlearning in Large Reasoning Models](https://arxiv.org/abs/2610.12769) | REDaCT是一种面向大型推理模型的推理时遗忘学习框架，通过泄漏检测器与纠错嵌入注入机制，在不引发推理坍塌的前提下从思维链和最终答案中移除遗忘集信息。 |
| [^96] | [DistressBench: Evaluating Multi-Dimensional Crisis Support in Large Language Models](https://arxiv.org/abs/2610.12758) | 该论文提出了DistressBench——一个包含718段由临床医生参与构建、涵盖23个自杀与自残子类别的对话基准，通过七个维度的加权评分标准评估大语言模型在危机情境下的支持能力，发现最佳前沿模型仅能提供88.4%的必要关怀，显著低于临床医生的98.7%。 |
| [^97] | [MADS: Multi-Agent Dialog Simulation for Robust Conversational Agents](https://arxiv.org/abs/2610.12757) | 提出了MADS多智能体对话模拟管道，通过注入对抗性对话生成多样化的失败场景对话轨迹，并构建包含六领域、55个人设和1095段对话的MADS-Bench基准，用于训练和评估更鲁棒的任务导向对话代理。 |
| [^98] | [U4RASD at KnowledgeGraphEval-2026: Ontology-Constrained Encoder and LLM Systems for Arabic Relation Extraction](https://arxiv.org/abs/2610.12742) | U4RASD 通过结合本体约束预测的编码器系统与 QLoRA 微调的大语言模型系统，以集成方案在 KnowledgeGraphEval 2026 阿拉伯语有向关系抽取子任务 2 中以 94.06 的 micro-F1 分数获得第一名。 |
| [^99] | [Unsupervised Multimodal Reward Modeling](https://arxiv.org/abs/2610.12707) | 提出一种无需人类偏好标签的无监督多模态奖励建模方法，通过在冻结的 Qwen3-VL 中间层训练线性头来区分网络图注与其扰动版本，仅用数千参数和约千分之一的微调计算量即可在多个基准上超越或持平 GPT-4o。 |
| [^100] | [Activating the Annotator: Operationalising Lived Experience in Data Annotation](https://arxiv.org/abs/2610.12701) | 该论文提出AXLE——一种标注前阶段，通过情境小场景引导对话并将其合成为第一人称微型叙事，在标注开始前激活标注者的生活经验，从而将位置性从静态的人口统计学属性转变为可由任务本身激发的动态资源，并证实其对模糊仇恨言论的标注产生了影响。 |
| [^101] | [GeoFitBench-LLM: A Benchmark for Assessing Large Language Models' Capabilities in Geospatial Data Requirements Elicitation for Geo-Analytic Tasks](https://arxiv.org/abs/2610.12669) | 本文提出了GeoFitBench-LLM基准，通过时空关系和语义锚定两个推理维度，以独立于数据提供者的标准化模式评估大语言模型从自然语言地理分析请求中推断所需数据集、支持资源和约束条件的能力。 |
| [^102] | [Recurrent Harness for Agentic Continual Learning](https://arxiv.org/abs/2610.12668) | 提出 ReHarness，一种循环式智能体持续学习框架，在严格在线流式协议下，将每个实例处理后的轨迹作为反馈向后传播，修订该实例对应的反思状态，实现前向读写与后向修订相结合的在线适应。 |
| [^103] | [perturber: A Stateless Service for Semantics-Preserving Text Perturbations](https://arxiv.org/abs/2610.12667) | 提出了 perturber——一个无状态服务和 Python 库，每次调用仅施加一种语义保持的文本扰动（共 68 种、分属 11 个族），通过单一原语设计在构造上保证服务与库的输出完全一致，并评估了该工具自身的正确性、可复现性与成本。 |
| [^104] | [Cancerous Backdoor Attack: Gradual Activation via Invisible Tokens and A Stage Controller](https://arxiv.org/abs/2610.12644) | 该论文提出了一种名为“癌症式后门攻击”的新型后门攻击方法，利用不可见令牌和阶段控制器，使恶意行为在大语言模型中通过多个隐藏阶段逐步激活，从而渐进式地破坏模型的推理能力。 |
| [^105] | [GeopoliticalQA: Benchmarking Retrieval and Generation over Intelligence Reports](https://arxiv.org/abs/2610.12617) | 该论文构建了基于415份开源地缘政治与战略情报报告、包含5,525个问答对的问答基准GeopoliticalQA，并通过评估15个检索系统和六个开放权重LLM发现，模型在文档级检索上表现相近，而更具挑战性的块级检索质量并不取决于嵌入模型的规模。 |
| [^106] | [Knowing Is Not Telling: Identity Ossification in LLM Representations of Trans-border Communities](https://arxiv.org/abs/2610.12610) | 本文定义并揭示了“身份固化”现象，提出跨境身份固化诊断框架（TIODF），证明大语言模型会将跨境民族流动的群体认同简化为单一固定的民族国家认同，并能区分这一偏差源于知识缺失还是模型的提问框定方式。 |
| [^107] | [Self-Preservation in Language Models](https://arxiv.org/abs/2610.12602) | 该论文提出将任务描述及示例组织成任务库，让语言模型自行选择保留目标并生成重放数据，从而在持续学习中更广泛地保护自身原有能力。 |
| [^108] | [Reflective Prompt Optimization of a Deep-Research Agent: A Topic-Disjoint, Multi-Seed Protocol for Measuring Transferable Gains and Prompt-Level Overfitting](https://arxiv.org/abs/2610.12601) | 本文提出一种主题不相交、多种子的评估协议，将反思性提示优化（GEPA）应用于多智能体深度研究智能体，证明其在避免记忆训练答案的情况下仍能取得可迁移的引用奖励提升（+0.079），同时揭示出“提示级过拟合”现象——获胜提示近半内容实为记忆的答案密钥。 |
| [^109] | [SpireBench: Evaluating Long-Horizon Strategic Play of Agentic Models with Slay the Spire](https://arxiv.org/abs/2610.12596) | SpireBench让20个大语言模型完整游玩《杀戮尖塔》以评估长时程战略推理能力，发现前沿模型虽达到普通人类水平但远逊于专家，且随着游戏进程延长差距不断扩大，所有模型都误判长线风险，倾向保守保血而非牺牲血量换取成长。 |
| [^110] | [One Model to Loop Them All: Elastic Latent Recurrence for Language Models](https://arxiv.org/abs/2610.12582) | 该论文提出弹性潜在循环（ELR）训练方法，消除了循环语言模型的“单循环税”，使模型在单次循环时以与非循环模型相同的推理成本达到相近准确率，而增加循环次数即可用更多推理计算换取更高性能。 |
| [^111] | [CLUE: A Contrastive Learning Framework for Unsupervised Event Type Induction and Annotation](https://arxiv.org/abs/2610.12580) | CLUE框架通过新颖的联合嵌入表示和对比损失引导的聚类，实现了无监督的事件类型归纳、命名与标注，从而降低了人工构建事件本体的成本和人为偏见。 |
| [^112] | [Contextual Correction of Parallel Draftsfor Speculative Decoding](https://arxiv.org/abs/2610.12570) | 提出轻量级机制DCOCO，通过保留并评估多个候选token进行上下文修正，在不牺牲并行起草效率的前提下恢复草稿块内的依赖关系，从而提升扩散式推测解码的草稿质量。 |
| [^113] | [SWE-Duel: Towards a Self-Scaling Arena for Coding Agents](https://arxiv.org/abs/2610.12569) | SWE-Duel 提出了一种自扩展的编程智能体评估竞技场：两个智能体以红蓝对抗的方式互相出题——红方在真实仓库中实现新功能并植入隐蔽缺陷，蓝方进行盲修——使任务在评估时动态生成、无内容可泄露，且难度随参与智能体的能力提升而自动扩展。 |
| [^114] | [AutoIndexer: Flexible-Order Decoding by Training Causal Models on Chains of Edits](https://arxiv.org/abs/2610.12545) | AutoIndexer提出了一种模拟文本编辑器的“编辑链”训练与解码机制，使因果自回归模型能够通过插入、替换和删除操作灵活地事后修改已生成的输出，从而支持非因果顺序的灵活解码。 |
| [^115] | [An Input-Dependent Readout Phase for Resonant Sequence Models](https://arxiv.org/abs/2610.12496) | 本文将 ResonatorLM 中每个头固定的读取相位改为当前 token 的函数，仅在读取环节引入输入依赖性，在保持精确卷积和仅增加 0.2% 参数的同时提升了模型能力，并分析了未训练振荡器组的相位可编码时间但线性重建能力受限（约 16 个 token）。 |
| [^116] | [Kairos: Toward a Byte-Level Foundation Model for Every Modality](https://arxiv.org/abs/2610.12495) | Kairos 提出了一种在统一字节空间上处理文本、图像、音频、视频、激光雷达等多模态的小型基础模型架构，融合门控 DeltaNet 线性注意力、滑动窗口注意力、注意力-残差聚合、稀疏 MoE 和多尺度字节级编解码器，并通过单机上的小规模过拟合收敛测试验证该组合能否有效优化。 |
| [^117] | [Rehearse Everything, Remember Nothing: Attic-KV Rehearses What Will Be Read](https://arxiv.org/abs/2610.12133) | 该论文发现传统KV缓存压缩中“重读全部上下文”的排练策略在低保留率下会因预算分散而失效，并提出Attic-KV——只排练将来会被读取的内容，像考前自测一样大幅提升缓存压缩后的记忆效果。 |
| [^118] | [InterviewPlayground: A Simulation Environment for Evaluating AI Interviewers](https://arxiv.org/abs/2610.12023) | 提出了InterviewPlayground，一个基于社会理论模拟参与者行为、通过多轮交互生成标准化访谈成绩单来评估AI访谈员的模拟环境，并验证了模拟评估结果能够预测AI访谈员与真实人类参与者互动时的实际表现。 |
| [^119] | [When History Helps and Hurts: Selective History Use across Multimodal Turns](https://arxiv.org/abs/2610.11948) | 该论文提出ReTurn基准（7,000个涵盖视觉与音频的任务），首次将多模态多轮对话中的历史使用需求与问题难度解耦，以系统评估模型对对话历史的选择性使用能力。 |
| [^120] | [Smoothing the Top-k Exposure Boundary for Sparse Mixture-of-Experts](https://arxiv.org/abs/2610.11575) | 提出弹性专家路由方法，通过在以k为中心的局部离散分布中随机采样激活专家数量，将稀疏混合专家模型中刚性的top-k选择边界软化为渐进的概率分布，在不增加计算成本的前提下缓解了竞争专家因阈值划分而导致的训练反馈不均衡问题。 |
| [^121] | [Constitutional Gating and Deterministic Recovery for Multi-Agent LLM Negotiation: Ablations Against a Stateful Adversarial Gatekeeper](https://arxiv.org/abs/2610.11542) | 该论文提出由5支柱运行时宪法、4层集群和认知退火组成的三部分控制栈，并通过针对有状态对抗性守门人的消融实验，验证其能消除多智能体LLM谈判中的三类模型调用浪费并实现确定性死锁恢复。 |
| [^122] | [Measuring and Mitigating Solution Mode Collapse in RLVR](https://arxiv.org/abs/2610.11064) | 本研究提出ModeBench多解任务基准，发现RLVR后训练在保持或提升准确率的同时，会导致模型的解多样性坍缩，概率集中于更少的正确解题模式上。 |
| [^123] | [PHRBench: A Behavioral Evaluation of Post-Hallucination Reasoning in LLMs](https://arxiv.org/abs/2610.10455) | 提出了PHRBench这一受控基准，通过幻觉顺从、幻觉规避和启发式纠正等行为指标，在四个领域对18个大语言模型处理幻觉前提的推理轨迹进行评估，发现模型成功从幻觉中恢复并得出正确答案的情况仍然相对罕见。 |
| [^124] | [Reasoning-Token Spikes Under Prompted Untruthful Responding in Large Language Models](https://arxiv.org/abs/2610.10405) | 该论文基于认知负荷理论，提出利用推理token数量这一无需访问思维链内容的低带宽信号，发现大语言模型在被提示进行不真实回答时会出现推理token数量的激增，从而为检测模型欺骗行为提供了新方法。 |
| [^125] | [Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution](https://arxiv.org/abs/2610.09772) | 该论文提出AO-DA解耦架构，将边缘LLM智能体的逻辑推理与人设表达分离为同一INT4基础模型上两条可热插拔LoRA适配器的独立推理路径，使逻辑部分对上下文污染实现结构性免疫。 |
| [^126] | [From Expert-Guided Proof Search to Automated Open-Problem Solving](https://arxiv.org/abs/2610.09769) | 研究者开发的多智能体开源系统Bolzano无需针对具体问题的人类指导，在约3,800个开放问题中自动解决了约200个，其中包括经原作者确认的STOC 2026论文中提出的4个问题。 |
| [^127] | [TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models](https://arxiv.org/abs/2610.07767) | TRACE是一个面向MoE语言模型强化学习训练的FP4量化框架，通过rollout引导的量化感知训练，利用rollout侧量化结果指导训练侧FP4舍入决策，直接缩小训练路径与rollout路径两条量化执行路径之间的差异。 |
| [^128] | [Wikidata Search Traces: A Dataset for Training Knowledge Graph Search Agents](https://arxiv.org/abs/2610.06650) | 该论文发布了首个记录解题者在Wikidata知识图谱上探索过程的数据集，用于训练图搜索智能体，并证明图搜索难度可通过问题结构控制、长程搜索失败主要源于证据管理方式而非模型本身，且开源模型在合适环境中可匹敌商业模型。 |
| [^129] | [DREAM: Dynamic Resolution Assignment For Multimodal Multi-agent Debate](https://arxiv.org/abs/2610.05615) | 提出DREAM框架，通过零样本探测为每个智能体动态分配不同的图像分辨率，并利用ANLL不确定性量化和自适应阈值来缓解多模态多智能体辩论中的群体思维问题，从而提升多模态推理能力。 |
| [^130] | [When LLMs Sit Above Diagnostic Tools: Unrealized Complementarity in Industrial Fault Diagnosis](https://arxiv.org/abs/2610.05031) | 该研究发现在工业故障诊断中，大语言模型与专业诊断工具的集成并未实现真正的互补性——外部专家信息虽能提升LLM表现（最高提升12.8个百分点），但集成系统仍不及最强的独立专业工具，且冲突信息甚至会推翻LLM原本正确的判断。 |
| [^131] | [A generative-informed neuro-symbolic framework for syntactic ambiguity resolution: Evidence from Arabic DPs](https://arxiv.org/abs/2610.02529) | 该研究提出一种将生成句法学概念与AraBERT相结合的神经符号框架，把阿拉伯语限定词短语的结构歧义消解建模为基于候选的决策任务，在未见评估集上取得了96.88%的准确率，同时揭示了不同挂靠类型之间性能的不对称性。 |
| [^132] | [Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment](https://arxiv.org/abs/2609.38972) | 本文提出CoT-可解释性对齐（CIA）指标来测量LLM思维链与内部计算之间的一致性，发现现有模型对齐程度有限（44.8%–75.9%），并通过以任务准确率和参数化忠实度为奖励的后训练方法有效提升了对齐性。 |
| [^133] | [StreamDecisionBench: Evaluating Decisions in Force on Evolving Language Streams](https://arxiv.org/abs/2609.38612) | 该论文提出 StreamDecisionBench 基准，评估语言模型在证据流不断演化时每一时刻“生效决策”的正确性，并将错误归因于判断或延迟，弥补了传统离线准确率无法衡量决策时效性的缺陷。 |
| [^134] | [OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime](https://arxiv.org/abs/2609.38345) | 提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。 |
| [^135] | [EngiWorld: What Can Frontier Agents Deliver in Professional Engineering Environments?](https://arxiv.org/abs/2609.37686) | 该论文提出了首个覆盖完整设计循环的工程智能体基准EngiWorld，涵盖1,301个专家任务、6个工程领域和26个专业软件平台，并引入以工件为中心的自动化评估方法来检验工程成果的几何有效性、物理可行性与规则合规性。 |
| [^136] | [Generating Edit-Inducing Questions for AI Research Manuscripts](https://arxiv.org/abs/2609.36617) | 该研究比较了GPT与人类审稿人为AI论文草稿生成“编辑诱发式问题”的能力，发现GPT的问题能引发更广泛深入的修改但有效率更低，并揭示了一个反直觉现象：处理长上下文反而会损害推理模型生成有用输出的能力。 |
| [^137] | [MS-GLA: Multi-Scale Gated Linear Attention for Addressing Representational Bottlenecks via Multi-Temporal Resolution](https://arxiv.org/abs/2609.35664) | 提出多尺度门控线性注意力（MS-GLA），通过将注意力头分配到多个时间分辨率并用可学习的输入依赖融合层动态重组各头组输出，解决GLA Transformer在单一时间分辨率下的表示瓶颈，在不增加每头状态大小的情况下扩展有效记忆容量。 |
| [^138] | [Epistemic Policy Divergence in Multi-Turn LLM Contamination: A Protocol-Gradient Investigation](https://arxiv.org/abs/2609.35308) | 该研究提出“会话级污染”这一失败模式，通过五种沿来源权威梯度排列的污染协议，首次系统揭示了大语言模型在多轮对话中采纳错误前提时的认知策略存在显著分歧——GPT-5.4 Mini完全抵抗采纳，而Gemini-3.1 Flash-Lite的采纳率随信息来源权威性增强而急剧上升。 |
| [^139] | [Efficient Reasoning Exploration via State-Conditioned Latent Steering with Progress Guidance](https://arxiv.org/abs/2609.24066) | 提出了一种无需训练的潜在引导框架SPS，通过构建包含进展引导向量的状态条件化方向库，在推理时引导模型探索能取得有意义进展的多样化推理路径，从而缓解探索坍缩问题并提升Best-of-N推理的探索效率。 |
| [^140] | [Encoder Awakening via Adapters: Effective Domain-Adaptive Fine-tuning of Speech-LLMs](https://arxiv.org/abs/2609.17981) | 提出EAVA方法，通过在语音大语言模型的每个编码器层插入轻量级适配器并进行专门训练，在保留预训练知识的同时注入目标领域声学知识，有效提升有限数据下儿童语音、方言语音等领域偏移语音的识别性能。 |
| [^141] | [ResidencyRL: Reinforcement Learning in Simulated Clinical Environments](https://arxiv.org/abs/2608.07418) | ResidencyRL提出了一种强化学习方法，通过模拟多轮临床诊疗环境（含对抗性患者模拟器和结构化诊断奖励）来训练临床AI智能体，使其像住院医师一样在实践互动中提升完整临床决策序列的能力。 |
| [^142] | [M3-DuplexBench: A Multi-Turn, Multilingual, Multidomain Benchmark for Full-Duplex Spoken Dialogue Models](https://arxiv.org/abs/2607.29125) | 该论文提出了M3-DuplexBench，一个用于全双工语音对话系统的多轮、多语言（英语与日语）、多领域基准测试，可联合评估交互时机与响应内容，并支持跨语言比较。 |
| [^143] | [TokenSwap: Benchmarking and Reducing the Modality Gap in Multimodal LLMs](https://arxiv.org/abs/2607.28640) | 本文提出TokenSwap方法，通过将文本基准（如MMLU）中的文本概念替换为语义对齐的图像，构建出图像交错的TokenSwap-Bench基准，首次系统揭示了42个多模态大语言模型中普遍存在的模态差距，即图像交错输入相比纯文本输入平均导致19.6%的性能下降。 |
| [^144] | [ZenGen: Social Mind for LLMs](https://arxiv.org/abs/2607.23740) | 本文提出了ZenGen框架，通过SoMBench基准测量大语言模型的社会智能，并采用诊断驱动的训练方案来内化和提升其社会认知能力。 |
| [^145] | [Medical foundation models converge less under label supervision](https://arxiv.org/abs/2607.20274) | 该论文通过对75万余张医学图像、18个公开模型和101个自训练模型的大规模分析发现，医学基础模型——尤其是经过标签监督训练的模型——对相同图像的表征彼此差异巨大，其近邻重叠度甚至接近随机初始化的水平，因此不同基础模型的嵌入不可互换使用。 |
| [^146] | [The One-Word Census: Answer-Choice Conformity Across 44 Language Models](https://arxiv.org/abs/2607.12796) | 本研究通过96个开放式单词选择提示对105个语言模型进行了“答案趋同性”普查，发现模型的选择高度趋同——例如46%的模型在“选一个词”时都选了"serendipity"——且这种趋同并非由小型模型或个性化微调模型所致，反而轻度后训练和个性化微调的模型分歧最大。 |
| [^147] | [The syntax of wh-agreement in Yemeni Ibbi Arabic](https://arxiv.org/abs/2607.04986) | 本文提出基于特征继承机制的跨语段一致分析方法，论证也门伊比阿拉伯语中由第三人称代词语法化而来的后缀充当wh一致关系的形态标记，为生成句法学提供了独特的实证贡献。 |
| [^148] | [KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking](https://arxiv.org/abs/2606.22807) | KaLM-Reranker-V1 是一种快速但非晚期交互的重排序器，通过编码器-解码器架构将查询与段落计算解耦，利用 Matryoshka 嵌入池化离线预编码段落，并借助交叉注意力捕捉细粒度相关性，在保持强表达能力的同时显著提升了部署效率与灵活性。 |
| [^149] | [PACT: Privileged Trace Co-Training for Multi-Turn Tool-Use Agents](https://arxiv.org/abs/2606.16215) | PACT框架提出将专家轨迹仅作为训练时的优化信号而非推理提示，通过轨迹条件化强化学习代理目标和带退火过程监督的组件感知SFT损失两个互补信号进行协同训练，同时保持推理时仅使用提示词生成，从而兼顾强化学习的推理一致性与监督微调的密集过程监督优势。 |
| [^150] | [Let Large Language Models Judge Each Other: Multi-Agent Peer-Reviewed Reasoning for Medical Question Answering](https://arxiv.org/abs/2606.15419) | 该论文提出多智能体同行评审推理方法，让多个大语言模型在医学问答中互相评审彼此的思维链推理并择优生成最终答案，显著提升了准确性、可解释性和鲁棒性。 |
| [^151] | [Experience Makes Skillful: Enabling Generalizable Medical Agent Reasoning via Self-Evolving Skill Memory](https://arxiv.org/abs/2606.09365) | 提出SkeMex框架，将医疗智能体的交互轨迹提炼为结构化的技能记忆，在不更新模型权重的情况下实现自我演化，从而支持可泛化的长程临床推理。 |
| [^152] | [Agents' Last Exam](https://arxiv.org/abs/2606.05405) | 本文提出由250多位行业专家共同打造的Agents' Last Exam（ALE）基准，通过13个行业集群、55个子领域、1千多项长周期、结果可验证的真实经济价值任务来评估AI智能体，并发现当前系统在最难任务层级上仍远未饱和。 |
| [^153] | [EchoDistill: Robust Large Audio Language Models via Noisy-to-Clean Self-Distillation](https://arxiv.org/abs/2605.23954) | EchoDistill提出一种噪声到干净的自蒸馏框架，在后训练中以干净音频作为特权信息，通过掩码响应token蒸馏、任务门控一致性塑形和教师参考的组相对优化，使大型音频语言模型在噪声环境下更鲁棒，且推理时无额外开销。 |
| [^154] | [Probabilistic Calibration Is a Trainable Capability in Language Models](https://arxiv.org/abs/2605.11845) | 该论文提出校准微调（包括软目标与硬目标两种变体），证明语言模型从隐式分布中进行采样的概率校准能力可以通过微调直接训练获得，并在12个模型上显著提升了结构化采样的保真度。 |
| [^155] | [MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading](https://arxiv.org/abs/2605.10268) | MemReread通过在最终记忆不充分时触发问题分解与针对性重读，恢复流式阅读中被过早丢弃的间接证据，从而在不依赖中间检索的情况下增强智能体的长上下文非线性推理能力，并结合强化学习提升长度外推能力和重读次数的自适应决策。 |
| [^156] | [How AI Writing Assistance Distorts Perceptions of AI Users and Their Opinions](https://arxiv.org/abs/2604.22503) | AI写作辅助会让读者对写作者产生普遍而持久的认知扭曲（显得更有主见、更有能力、更具社会优势），即使写作者本人反对且被明确告知，这些扭曲依然存在，而且扭曲越强反而越能增强说服力。 |
| [^157] | [Copy First, Translate Later: Interpreting Translation Dynamics in Multilingual Pretraining](https://arxiv.org/abs/2604.17633) | 该研究通过对一个1.7B模型在九种语言上进行细粒度检查点的预训练分析，揭示多语言翻译能力的发展分为两个阶段：早期以词元复制和表层相似性为主，后期才逐步形成真正的翻译能力。 |
| [^158] | [Quantum Vision Theory Applied to Audio Classification for Deepfake Speech Detection](https://arxiv.org/abs/2604.08104) | 本文提出量子视觉理论这一新视角，将语音频谱图先转换为信息波再输入深度学习模型进行分类，以提升深伪语音检测的性能。 |
| [^159] | [Rethinking Entropy Allocation in LLM-based ASR: Understanding the Dynamics between Speech Encoders and LLMs](https://arxiv.org/abs/2604.08003) | 该研究从熵分配的视角重新审视基于大语言模型的语音识别，提出了一种基于能力边界感知的多阶段训练策略，通过重新设计预训练以弥合语音-文本模态差距并引入迭代异步SFT，从而优化参数效率并增强幻觉鲁棒性。 |
| [^160] | [Weight Clustering Reveals the Importance of Relative Rank in Large Language Models](https://arxiv.org/abs/2603.17917) | 利用标量 K 均值聚类将大语言模型权重压缩为 16-64 个共享值并可在不重训的情况下保持性能，并通过扰动实验揭示保持权重间的相对排序比保持精确数值更重要。 |
| [^161] | [Dialects of Translationese Shape Language Model Learning](https://arxiv.org/abs/2602.16469) | 本文发现，机器翻译数据的源语言差异通过翻译腔显著影响语言模型的学习行为，其中词汇多样性是驱动总体困惑度的关键因素。 |
| [^162] | [COBALT: Contextual Bandit Learning with Offline Trajectories for Multi-Turn Code Generation](https://arxiv.org/abs/2602.03806) | COBALT 利用多轮代码生成是单步可恢复马尔可夫决策过程的观察，通过重用参考模型收集的离线多轮轨迹并采用单步补全的上下文老虎机学习，替代昂贵且不稳定的在线强化学习来高效训练 LLM 进行多轮代码生成。 |
| [^163] | [ARTIS: Agentic Risk-Aware Test-Time Scaling via Iterative Simulation](https://arxiv.org/abs/2602.01709) | ARTIS提出了一种风险感知的测试时扩展框架，通过在真实执行前进行迭代模拟将探索与承诺解耦，并引入风险感知的工具模拟器来捕捉罕见但高影响的失败模式，从而在不产生环境风险的前提下提升智能体动作的可靠性与鲁棒性。 |
| [^164] | [Self-Aware Knowledge Probing: Evaluating Language Models' Relational Knowledge through Confidence Calibration](https://arxiv.org/abs/2601.18901) | 该论文建立了一个评估语言模型关系知识置信度校准的实证框架，发现大多数语言模型（尤其是掩码语言模型）存在过度自信问题，而考虑释义一致性的置信度估计能提供最佳的校准效果。 |
| [^165] | [Assessment of Generative Named Entity Recognition in the Era of Large Language Models](https://arxiv.org/abs/2601.17898) | 通过对八个大语言模型在四个标准NER数据集上的系统评估发现，采用参数高效微调和结构化输出格式的开源大语言模型可实现与传统基于编码器的模型相媲美的命名实体识别性能，且其能力来源于指令遵循与生成能力而非记忆。 |
| [^166] | [The Alignment Veto: How Safety Training Suppresses Cultural Knowledge in LLMs](https://arxiv.org/abs/2510.13154) | 该研究提出“对齐否决”概念，揭示安全训练在模型拒绝文化敏感问题时只是抑制而非抹除其内部准确的文化知识分布，且这种“安全税”在不同国家间分布不均（最高达37.6%），母语提示反而会加剧而非缩小这一差距。 |
| [^167] | [ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios](https://arxiv.org/abs/2510.10998) | 该论文提出了ABLEIST评估框架，通过五个健全主义伤害指标和三个交叉性伤害指标，对六个LLM在2,820个招聘场景中的表现进行审计，揭示了针对残障候选人（尤其在性别、种姓等交叉身份维度）的显著偏见伤害，而许多先进模型无法检测到这些伤害。 |
| [^168] | [Breaking the Cloak! Unveiling Chinese Cloaked Toxicity with Homophone Graph and Toxic Lexicon](https://arxiv.org/abs/2505.22184) | 提出了C$^2$TU，一种无需训练、无需提示的新方法，利用中文同音字图谱和毒性词典，通过子串匹配识别候选毒性词并进行过滤纠正，有效破解社交媒体中以谐音伪装毒性词汇规避审查的内容。 |
| [^169] | [ReasoningShield: Safety Detection over Reasoning Traces of Large Reasoning Models](https://arxiv.org/abs/2505.17244) | 大型推理模型的思维链中间步骤中可能隐藏有害内容，导致现有审核工具性能大幅下降，本文提出ReasoningShield基准与轻量级模型，用于对CoT推理轨迹进行安全审核。 |
| [^170] | [Auditing Prompt Injection Defenses Before They Reach Users: What Strong Adversaries Reveal](https://arxiv.org/abs/2505.15738) | 该论文主张，由于基于微调的提示注入防御缺乏形式化保证且可能带来虚假安全感，开发者应在防御部署给用户之前用强大的对抗性攻击对其进行压力测试。 |
| [^171] | [Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals](https://arxiv.org/abs/2502.01025) | 该论文发现LLM内部特定注意力头天然编码了“信息充分性信号”，据此提出“动态上下文截断”方法，让模型在获取足够任务相关信息后自动停止处理上下文，在多个问答数据集和模型上实现了平均1.33倍的token减少且准确率反而提升3.4%。 |
| [^172] | [OARelatedWork: A Large-Scale Dataset of Related Work Sections with Full-texts from Open Access Sources](https://arxiv.org/abs/2405.01930) | 本文提出了首个用于相关工作生成的大规模多文档摘要数据集OARelatedWork，包含完整相关工作章节与被引论文全文，并通过基准测试揭示了大语言模型在处理海量全文上下文时仍面临的挑战。 |

# 详细

[^1]: RSI-Anything：利用世界模型解锁递归自我改进

    RSI-Anything: Unlocking Recursive Self-Improvement with World Models

    [https://arxiv.org/abs/2610.13912](https://arxiv.org/abs/2610.13912)

    提出了基于预测性世界模型的通用递归自我改进框架 RSI-Anything，通过世界模型指导行动与学习数据选择，并借助兼容性检查决定更新保留，在代码修复和GUI任务中实现了显著的性能提升。

    

    递归自我改进（RSI）旨在将交互经验转化为持续的改进，然而外部反馈代价高昂，且随着策略的演化，世界模型的预测可能变得不可靠。我们提出了 RSI-Anything，一个构建在 JEPA-Anything 之上的通用领域框架，它利用预测性世界模型来指导行动和学习数据的选择。观测到的反馈驱动策略或工作流的改进以及条件性的世界模型适配，同时组件与兼容性检查决定更新后的策略-模型对是否被保留。在多种计算与科学领域中，RSI-Anything 展现了显著的提升。在代码修复任务中，在固定编码器的情况下，成功率从使用静态验证器的 26.3% 通过预测性选择提升至 29.2%，经世界模型适配后进一步提升至 31.2%，且测试用量相当。图形用户界面（GUI）评估显示任务完成率更高，同时平均操作次数下降（原文摘要在此处截断）。

    arXiv:2610.13912v1 Announce Type: new  Abstract: Recursive self-improvement (RSI) aims to transform interaction experience into persistent improvements, yet external feedback is costly and world-model predictions may become unreliable as policies evolve. We introduce RSI-Anything, a domain-general framework built on JEPA-Anything that uses predictive world models to guide action and learning-data selection. Observed feedback drives policy or workflow improvement and conditional world-model adaptation, while component and compatibility checks govern the retention of updated policy-model pairs. Across diverse computational and scientific domains, RSI-Anything demonstrates substantial gains. In code repair, with the coder fixed, success increases from 26.3% with a static verifier to 29.2% through predictive selection and 31.2% after world-model adaptation, at comparable test usage. Graphical user interface (GUI) evaluations showed higher task completion, while average actions decrease by 
    
[^2]: 当视觉语言模型破坏其所知：空间表征低于词汇先验

    When VLMs Destroy What They Know: Spatial Representations Fall Below Lexical Priors

    [https://arxiv.org/abs/2610.13910](https://arxiv.org/abs/2610.13910)

    研究发现视觉语言模型对空间关系的内部表征远不如词汇信息忠实——空间关系虽可在谓词词元层面识别，但在聚合为全局句子表征时会丢失，导致空间表征低于词汇先验，即使模型在组合性基准上表现良好。

    

    视觉语言模型（VLM）越来越多地被用于分析和推理图像，但尚不清楚其内部表征是否忠实地编码了此类推理所需的结构化关系。我们提出了EICS（评估不变概念稳定性），一种用于测试表征在受控单概念干预下是否一致编码概念的度量指标；同时提出CASP-VL（面向视觉语言模型的因果控制空间扰动），一个基于Visual Genome场景图构建的因果控制空间扰动基准。在多个VLM架构上，我们发现空间关系的表征远不如词汇信息忠实，即使这些模型在组合性基准测试中的得分远高于随机水平。进一步分析表明，空间关系在谓词自身的词元处是可识别的，但当句子被聚合为全局表征时便会丢失……

    arXiv:2610.13910v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) increasingly analyze and reason about images, yet it remains unclear whether their internal representations faithfully encode the structured relations required for such reasoning. We introduce EICS (Evaluation-Invariant Concept Stability), a metric for testing whether representations consistently encode concepts under controlled one-concept interventions, and CASP-VL (Causally controlled Spatial Perturbations for Vision-Language models), a benchmark of causally controlled spatial perturbations derived from Visual Genome scene graphs. Across multiple VLM architectures, we find that spatial relations are represented substantially less faithfully than lexical information, even when the same models score well above chance on compositional benchmarks. Further analysis shows that the relation is identifiable at the predicate's own tokens but is lost when the sentence is aggregated into a global representation, whe
    
[^3]: 神经语义解析的结构自动机

    Structural Automata for Neural Semantic Parsing

    [https://arxiv.org/abs/2610.13896](https://arxiv.org/abs/2610.13896)

    提出基于结构自动机的接受路径解码（APD）方法，可在固定长度限制内保证神经语义解析器生成良构的意义表示，且无需回退token，在DRS和AMR解析上显著优于局部掩蔽方法。

    

    神经语义解析器逐token地生成意义表示，然而其良构性取决于在生成过程中逐步展开的指称、辖域和图关系。受限解码可以防止局部无效的token，但无法保证一个有效前缀能够在固定的输出长度限制内被补全。我们在DRS解析的三个阶段（解码、事后处理和训练）研究结构自动机，并额外在AMR上评估了解码方法。对于解码，我们提出了接受路径解码（APD），它在剩余长度限制内维护一个已验证的补全方案，并且只要模型偏好的token同样可以被补全，就遵循该token。APD保证产出一个被接受的结果，且无需回退token。在严格的长度限制下，与局部掩蔽相比，它在DRS上将Smatch++分别提升了5.1和2.4分，在AMR上提升了6.4和3.2分，与在相同限制下需要回退token的回滚方法性能相当。

    arXiv:2610.13896v1 Announce Type: new  Abstract: Neural semantic parsers generate meaning representations token by token, yet well-formedness depends on references, scope, and graph relations that unfold during generation. Constrained decoding prevents locally invalid tokens but does not guarantee that a valid prefix can be completed within a fixed output-length limit. We investigate structural automata at three stages of DRS parsing (decoding, post hoc processing, and training) and additionally evaluate decoding on AMR. For decoding, we introduce accepting-path decoding (APD), which maintains a verified completion within the remaining length limit and follows the model's preferred token whenever that token can also be completed. APD guarantees an accepted output without retracting tokens. Under tight length limits, it improves Smatch++ over local masking by 5.1 and 2.4 points on DRS and by 6.4 and 3.2 points on AMR, comparable to rollback under the same limit, which retracts tokens. F
    
[^4]: 任务上下文塑造语言模型组件的角色

    Task Context Shapes Component Roles in Language Models

    [https://arxiv.org/abs/2610.13895](https://arxiv.org/abs/2610.13895)

    该论文揭示了任务上下文会重新分配语言模型中共享注意力头的影响力，表明同一模型组件在不同任务中扮演着不同的功能性角色。

    

    大语言模型在不同任务中会复用相同的注意力头，但仅仅识别出这些共享的注意力头并不能解释它们在每个任务中究竟做了什么不同的事情。一个自然的灵感来源是共享脑区之间随任务变化的相互作用，这启发了将语言模型计算视为网络的观点。在本文中，我们使用相同的图事实（即指明存在哪些有向连接的陈述语句）来比较不同任务下固定的模型组件。我们测量了干预对注意力头及后续层激活状态的影响（影响力），以及对答案的影响（重要性）。研究发现，任务变化会在五个开源权重模型（涵盖四个模型系列、参数规模从15亿到140亿）中重新分配影响力。在Gemma-2-9B和Qwen3-14B中，对早期内部激活向量进行编辑，可以在书面指令保持不变的情况下切换模型所读取的事实。随后，相同的注意力头会响应不同的图事实……

    arXiv:2610.13895v1 Announce Type: cross  Abstract: Large language models reuse the same attention heads across tasks, but identifying these shared heads does not explain what they do differently in each task. A natural source of inspiration is task-dependent interactions among shared brain regions, which motivate a network view of language-model computation. Here, we compare fixed model components across tasks using identical graph facts, statements specifying which directed connections are present. We measure how interventions affect attention heads and the activation state in later layers (influence), as well as the answer (importance). Task changes redistribute the influence in five open-weight models spanning four families and 1.5-14 billion parameters. In Gemma-2-9B and Qwen3-14B, an activation edit changes an early internal activation vector and switches which fact the model reads while the written instruction stays fixed. The same heads then respond to different graph facts and 
    
[^5]: 使用人工智能大规模研究漫画：PAGE框架及美国漫画与日本漫画的三项研究

    Studying Comics at Scale with Artificial Intelligence: The PAGE Framework and Three Studies of American Comics and Japanese Manga

    [https://arxiv.org/abs/2610.13840](https://arxiv.org/abs/2610.13840)

    提出PAGE框架，利用人工智能将漫画扫描件自动解析为分格与文本，并借助视觉语言模型按编码手册进行标注和与人工标注对齐，实现了对3929本美国漫画和87卷日本漫画的大规模整体化研究。

    

    漫画是二十世纪最伟大的视觉记录之一：在数十万页的漫画中，大众读者看到了他们时代的领袖、战争与英雄。然而这一记录的大部分从未被作为一个整体加以研究，漫画学术研究通常一次只针对一本作品展开。我们认为，包括生成模型在内的人工智能可以改变这一现状，从而有益于漫画媒介本身。我们提出了PAGE框架，使人工智能成为漫画研究的一种工具：它将每一份扫描件解析为分格、文本和书目记录；使用遵循书面编码手册的视觉语言模型对每一页进行标注，如同人类编码员的工作方式；将这些解读的样本与人工标注进行对齐验证；并通过控制出版商和体裁等变量的估计来解释档案库中的变化。我们在3,929本美国漫画书（1938-1956年）和87卷日本漫画上对该框架进行了验证，开展了三项研究……

    arXiv:2610.13840v1 Announce Type: new  Abstract: Comics are one of the great visual records of the twentieth century: hundreds of thousands of pages in which a mass readership saw its leaders, its wars and its heroes. Most of this record has never been studied as a whole; comics scholarship proceeds one book at a time. We argue that artificial intelligence, including generative models, can change this to the benefit of the medium itself. We propose PAGE, a framework that makes artificial intelligence an instrument for comics research. It parses every scan into panels, text and a bibliographic record; annotates each page with a vision-language model that follows a written codebook, as a human coder would; grounds samples of these readings in human annotation; and explains change across an archive, with estimates controlled for publisher and genre. We demonstrate the framework on 3,929 American comic books (1938--1956) and 87 volumes of Japanese manga, in three studies, each resting on a
    
[^6]: OpenPA：无需强制对齐的多粒度发音评估与混合转录自适应

    OpenPA: Forced-alignment-free Multi-granular Pronunciation Assessment with Mixed-transcript Adaptation

    [https://arxiv.org/abs/2610.13826](https://arxiv.org/abs/2610.13826)

    提出OpenPA框架，利用Whisper的交叉注意力机制实现无需强制对齐的单词级与话语级联合发音评估，并通过混合转录自适应使单一模型同时支持封闭式和开放式回答场景。

    

    自动发音评估（APA）的最新进展推动了从脚本化评估向自由形式评估的转变。然而，现有方法通常依赖于包含强制对齐和独立特征提取器的多阶段流水线，限制了其在不同回答场景中的实用性。鉴于此，我们提出了OpenPA，这是一种新颖的APA框架，它利用Whisper的交叉注意力机制，在无需强制对齐的情况下联合预测单词级和话语级的发音分数。通过混合转录自适应和自适应替换加权，OpenPA能够在单一模型中适应封闭式和开放式回答场景中的异构转录。仅在speechocean762上训练，OpenPA在域内封闭式和模拟开放式回答条件下，在所有被评估系统中取得了最强的整体性能，同时在MultiPA上保持了零样本性能。此外还在口头ca…（原文摘要在此处截断）

    arXiv:2610.13826v1 Announce Type: cross  Abstract: Recent advances in automatic pronunciation assessment (APA) have driven a shift from scripted to free-form assessment. However, existing methods often rely on multi-stage pipelines with forced alignment and separate feature extractors, limiting their practicality across response settings. In light of this, we propose OpenPA, a novel APA framework that leverages Whisper's cross-attention mechanism to jointly predict word- and utterance-level pronunciation scores without forced alignment. Through mixed-transcript adaptation and adaptive substitution weighting, OpenPA accommodates heterogeneous transcripts across closed- and open-response settings within a single model. Trained solely on speechocean762, OpenPA achieves the strongest overall performance among evaluated systems under in-domain closed and simulated open-response conditions, while maintaining its performance zero-shot on MultiPA. An additional cross-task evaluation on oral-ca
    
[^7]: 语言模型能学会何时放弃回答吗？策略学习与校准的答案选择

    Can Language Models Learn When to Abstain? Policy Learning and Calibrated Answer Selection

    [https://arxiv.org/abs/2610.13821](https://arxiv.org/abs/2610.13821)

    该论文提出通过训练一个独立的答案选择器来决定语言模型何时作答、何时放弃，其在TriviaQA上显著优于模型自身的置信度过滤和强化学习方法，将覆盖率从20.2%提升至30.9%、错误率从23.0%降至14.5%。

    

    一个有用的语言模型应当知道何时回答问题，何时说“我不知道”。我们能否通过训练模型本身，或训练一个独立组件来检查其答案，从而改进这一决策？我们在Qwen3-14B上针对事实性问题研究了这两种方法，采用一种对正确答案给予奖励、对错误答案进行惩罚、并允许放弃回答的评分标准。在三个训练种子和两个保留置信度阈值的设置下，学习得到的答案选择器在阈值平均效用上优于原始模型的校准置信度过滤器。在TriviaQA上，平均覆盖率从20.2%提升至30.9%，而在精确匹配评分下，已评分答案的合并错误率从23.0%降至14.5%。所测试的强化学习改动并未带来一致的增益。在NQ-Open上，选择器在阈值平均效用上优于过滤器，但仍逊于始终放弃回答的策略。一项部分经过人工审核的审计保留了Trivia（摘要在此处截断）。

    arXiv:2610.13821v1 Announce Type: new  Abstract: A useful language model should know when to answer and when to say "I don't know." Can we improve that decision by training the model itself or by training a separate component to check its answers? We study both approaches on Qwen3-14B for factual questions, using a score that rewards correct answers, penalizes mistakes, and allows abstention. Across three training seeds and two held-out confidence thresholds, a learned answer selector improves threshold-averaged utility over the original model's calibrated confidence filter. On TriviaQA, mean coverage increases from 20.2% to 30.9%, while the pooled error rate among scored answers decreases from 23.0% to 14.5% under exact-match grading. The tested reinforcement-learning changes do not establish consistent gains. On NQ-Open, the selector improves threshold-averaged utility over the filter but still performs worse than always abstaining. A partial human-reviewed audit preserves the Trivia
    
[^8]: QTestBench：一个用于评估大语言模型生成的量子代码的测试框架

    QTestBench: A Test Framework for Evaluating LLM-Generated Quantum Code

    [https://arxiv.org/abs/2610.13811](https://arxiv.org/abs/2610.13811)

    该论文提出了QTestBench框架，首次通过将大语言模型生成的量子测试判定结果与独立的基准检查器对比，来评估这些测试能否有效区分正确与错误的量子代码实现，而非将测试本身当作评估标准。

    

    大语言模型（LLMs）在经典代码生成方面已展现出优异表现，并越来越多地被用于编写保护代码质量的单元测试。然而，它们能否在量子软件领域做到同样的事情仍不明确，因为量子程序具有概率性、依赖于数值容差，并由特定领域的检查器进行评判，因此一个格式良好的测试仍可能与基准测试的判定结果不一致。现有的量子基准测试衡量的是模型能否产生正确的解决方案，但并未衡量其生成的测试能否区分正确的实现与错误的实现。我们提出了QTestBench，这是一个将大语言模型生成的量子测试套件与独立的基准检查器进行对比评估的框架，绝不将测试本身视为真值。对于每个任务，QTestBench会提示模型生成语法、语义和行为层面的测试，在运行时注入候选解决方案，并将每个测试的判定结果与检查器的标签进行对比。

    arXiv:2610.13811v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown strong results in classical code generation and are increasingly used to write the unit tests that safeguard that code. Whether they can do the same for quantum software is less clear, because quantum programs are probabilistic, depend on numerical tolerances, and are judged by domain-specific checkers, so a well-formed test can still disagree with the benchmark's verdict. Existing quantum benchmarks measure whether a model produces a correct solution, but not whether its tests can distinguish correct implementations from faulty ones. We introduce QTestBench, a framework that evaluates LLM-generated quantum test suites against independent benchmark checkers, never treating the tests themselves as ground truth. For each task, QTestBench prompts models to produce syntactic, semantic, and behavioral tests, injects candidate solutions at runtime, and compares each test verdict with the checker's labe
    
[^9]: 通过子序列关联过滤与广度感知选择来追踪和驯服大语言模型查询扩展中的幻觉词项

    Tracing and Taming Hallucinated Terms via Subsequence-Association Filtering and Breadth-Aware Selection for LLM Query Expansion

    [https://arxiv.org/abs/2610.13810](https://arxiv.org/abs/2610.13810)

    提出无需训练、与检索器无关的查询扩展方法SAFE-QE，基于幻觉的子序列关联解释，通过语料库条件下的跨上下文可复现性评分追踪并过滤幻觉词项，并以广度感知的原则性标准决定保留词项的数量。

    

    大语言模型（LLM）目前已成为查询扩展词项的标准来源，但它们会产生幻觉。它们会注入不存在的实体和偏离主题的词项，这些词项增加了排序噪声，甚至可能损害强大的检索器。现有的补救措施，如语料库接地、相互验证和偏好对齐，将幻觉词项视为不加区分的干扰因素，仅通过一致性启发式规则进行过滤，既没有对某个词项为何会出现的机制性解释，也没有关于应保留多少词项的原则性标准。我们提出了子序列关联过滤、广度感知的查询扩展方法（SAFE-QE），这是一种无需训练、与检索器无关的方法，其建立在对幻觉的子序列关联解释之上。我们根据每个候选扩展词项在语料库条件下的跨上下文可复现性对其进行评分——那些在随机化查询上下文中反复出现、却在伪相关文档集中缺乏支持的词项，会被标记为主导性幻觉的产物（摘要在此处截断）。

    arXiv:2610.13810v1 Announce Type: cross  Abstract: Large language models (LLMs) are now a standard source of query expansion terms but they hallucinate. They inject non existent entities and off-topic terms that add ranking noise and can hurt strong retrievers. Existing remedies such as corpus grounding, mutual verification and preference alignment treat hallucinated terms as an undifferentiated nuisance to be filtered by agreement heuristics, without a mechanistic account of why a given term appears or a principled criterion for how many terms to keep. We propose Subsequence-Association Filtered, breadth-aware query expansion (SAFE-QE), a training-free, retriever agnostic method that draws on the subsequence-association account of hallucination. We score each candidate expansion term by its corpus conditioned cross-context reproducibility, terms that reappear under randomised query contexts yet lack support in the pseudo-relevant set are flagged as products of dominant hallucinatory a
    
[^10]: 错误率还不够：低资源语音识别中数据增强的代价

    Error Rate is Not Enough: Costs of Augmentation in Low-Resource ASR

    [https://arxiv.org/abs/2610.13807](https://arxiv.org/abs/2610.13807)

    本文通过音素级错误分析发现，低资源ASR中不同数据增强方法的收益与代价互补——SpecAugment虽总体改进最可靠但会加剧声调和腭化错误，基于文本的方法则引入辅音插入错误，两者结合可相互抵消缺陷，且这种互补性可由错误类型的声学属性与训练暴露度预测。

    

    数据增强是低资源自动语音识别（ASR）的标准做法，但其收益几乎总是以总体错误率的形式报告，这使得一种方法为何有效、何时有效这一问题悬而未决。在四种低资源语言和英语的五种数据规模上，我们将两种修改现有音频的增强方法（速度扰动和SpecAugment）与两种修改文本转写并据此合成音频的方法进行交叉组合实验。我们在音素层面分析错误，发现这些方法以互补的方式发挥作用。SpecAugment带来最可靠的总体改进，但会增加声调和腭化错误，而基于文本的方法会引入辅音插入错误；两者结合使用时，各自能减少对方引入的错误。这种互补性是可以预测的：声调和腭化各自仅由单一声学属性标记，且在训练中出现频率最低（即最低训练暴露度），而基于文本的方法……（原文摘要在此处截断）

    arXiv:2610.13807v1 Announce Type: new  Abstract: Data augmentation is standard practice for low-resource automatic speech recognition (ASR), but its benefits are almost always reported as aggregate error rates, leaving open why and when a method helps. On four low-resource languages and English at five data sizes, we cross two augmentations that modify existing audio (speed perturbation and SpecAugment) with two that modify text transcriptions and synthesize audio from it. We analyze errors at the phoneme level and find that they help for complementary reasons. SpecAugment has the most reliable aggregate improvements, but increases tone and palatalization errors, whereas text-based methods introduce consonant insertions; in combination, each reduces the errors the other introduces. This complementarity is predictable. Tone and palatalization are each signaled by a single acoustic property and occur least frequently in training (lowest training exposure), and text-based methods that fla
    
[^11]: REFLECT：角色感知的情感、情绪、礼貌与正式度检测

    REFLECT: Role-Aware Detection of Sentiment, Emotion, Politeness, and Formality

    [https://arxiv.org/abs/2610.13797](https://arxiv.org/abs/2610.13797)

    本文提出角色感知的Transformer架构REFLECT，通过将语义角色标注（SRL）配置直接融入自注意力机制，并用方向性注意力建模礼貌与正式度、非方向性注意力建模情感与情绪，从而实现对对话中四种交际含义的联合检测。

    

    检测对话中的交际意义，不仅需要对词汇内容进行建模，还需要对情感、情绪、礼貌与正式度（SEPF）得以表达所依赖的参与者角色结构进行建模。现有方法依赖上下文语言表示，而将语义角色保持在隐式状态。我们提出了一种角色感知的Transformer架构，将SEPF四个维度映射到基于理论的语义角色标注（SRL）配置中，并将这些配置直接融入Transformer的自注意力机制。角色感知注意力用于建模非方向性的内部状态（情感和情绪），而方向性的角色感知注意力则捕捉外部线索（礼貌和正式度）中从发起者到接受者的关系。最终得到的角色感知表示通过一个可学习的门控机制与上下文嵌入相融合，该门控机制能够自适应地权衡每个输入所贡献的角色信息量，而不是……

    arXiv:2610.13797v1 Announce Type: new  Abstract: Detecting communicative meaning in dialogue requires modeling not only lexical content but also the participant-role structures through which sentiment, emotion, politeness, and formality (SEPF) are expressed. Existing approaches rely on contextual language representations while leaving semantic roles implicit. We introduce a role-aware transformer architecture that maps the four SEPF dimensions to theory-grounded Semantic Role Labeling (SRL) configurations, incorporating these configurations directly into the transformer's self-attention mechanism. Role-aware attention models non-directional internal states (sentiment and emotion), while directional role-aware attention captures initiator-to-recipient relations for external cues (politeness and formality). The resulting role-aware representations are fused with contextual embeddings through a learned gate that adaptively weights how much role information each input contributes, rather t
    
[^12]: 不仅仅是评估：语言模型对话语助词敏感性的案例研究

    Not Just an Evaluation: A Case Study of Discourse Particle Sensitivity in Language Models

    [https://arxiv.org/abs/2610.13794](https://arxiv.org/abs/2610.13794)

    本研究通过行为与机制双重分析，首次揭示大语言模型与人类相似地偏好话语助词"just"和"only"的出现，且对"just"比"only"更敏感，展现了语言模型对话语中细微语义线索的敏感性。

    

    话语助词通过反映会话语境和对话者态度，微妙地塑造着话语的流向，同时往往不改变其真值条件意义。这些词还常常具有多种功能，其不同义项与其他助词的意义部分重叠；例如英语中"just"和"only"的排他性用法（"John only/just drinks water"）。尽管这些词在自然语言中出现频率很高，但由于其效果的微妙性质，它们在很大程度上尚未被研究。我们采用行为分析和机制分析相结合的方法，研究了大语言模型在自然对话中对"just"和"only"的敏感性。首先，我们发现大语言模型与人类相似，都偏好语境中这些助词的存在，且两者对"just"都比"only"更为敏感。其次，在从带有和不带这些助词的大语言模型中进行采样时，我们发现大语言模型在给定其来源语境的情况下，始终将生成内容评为更可能……

    arXiv:2610.13794v1 Announce Type: new  Abstract: Discourse particles subtly shape the flow of discourse by reflecting conversational context and interlocutor attitudes, often leaving the truth-conditional meaning untouched. They are also often polyfunctional, with distinct senses which partially overlap with other particles; for example, the exclusive use of English "just" and "only" ("John only/just drinks water"). Despite the frequency of these words in natural language, they have gone largely unstudied due to the subtle nature of their effects. We investigate LLM sensitivity to "just" and "only" in natural conversations using behavioral and mechanistic analyses. First, we find that LLMs and humans similarly prefer the presence of these particles in context, and both are more sensitive to "just" compared to "only". Second, when sampling from LLMs with and without these particles, we find LLMs consistently rate generations as more likely, given their source context, demonstrating thei
    
[^13]: OccludeBench：视觉语言模型能找到缺失的东西吗？

    OccludeBench: Can VLMs Find What's Missing?

    [https://arxiv.org/abs/2610.13790](https://arxiv.org/abs/2610.13790)

    该论文提出OccludeBench基准，揭示视觉语言模型从阴影、反射等物理线索恢复被遮挡物体的能力（42.7%）显著低于从场景上下文推理的能力（62.7%），且与人类相比存在明显差距，并通过OccludeTrace诊断工具定位了视觉推理的瓶颈环节。

    

    视觉语言模型（VLM）对图像中缺失内容的推理能力究竟如何？我们通过OccludeBench来研究这个问题，这是一个手动整理的基准测试，用于从残留的视觉证据中恢复完全被遮挡的物体。OccludeBench区分了两类情况：直接情况，即必须从阴影或反射等特定物理线索中恢复隐藏物体；以及推理情况，即可以从更广泛的场景上下文中推断出隐藏物体。在五个VLM上，直接情况的平均准确率为42.7%，而推理情况为62.7%，呈现出20个百分点的差距。人类在直接和推理图像上分别达到63.4%和68.5%的更高准确率，且没有显著差距。为了定位这一失败，我们引入了OccludeTrace，这是一个四阶段诊断方法，用于隔离视觉推理过程中出现问题的环节。瓶颈出现在模型必须从完整图像内的投影中恢复相关物体的阶段……

    arXiv:2610.13790v1 Announce Type: cross  Abstract: How well can vision-language models (VLMs) reason about what is missing from an image? We study this question with OccludeBench, a manually curated benchmark for recovering fully occluded objects from the visual evidence that remains. OccludeBench distinguishes direct cases, where the hidden object must be recovered from a specific physical cue such as a shadow or reflection, from inferential cases, where it can be inferred from broader scene context. Across five VLMs, accuracy averages 42.7% on direct cases versus 62.7% on inferential cases, exhibiting a 20.0-point gap. Humans attain higher accuracies of 63.4% versus 68.5% for direct and inferential images, respectively, with no significant gap. To locate this failure, we introduce OccludeTrace, a four-stage diagnostic that isolates where the visual reasoning process breaks down. The bottleneck emerges when models must recover the relevant object from the projection within the full im
    
[^14]: UkrMeme：乌克兰梗图中的多模态可解释说服检测

    UkrMeme: Multimodal Explainable Persuasion Detection in Ukrainian Memes

    [https://arxiv.org/abs/2610.13783](https://arxiv.org/abs/2610.13783)

    本文发布了包含200张乌克兰战时梗图的说服技术检测数据集UkrMeme及乌克兰语翻译基准PTM-UA，并评估了视觉-语言模型在零样本多模态设置下的说服检测性能及可解释性方法。

    

    梗图在现代网络交流中扮演着核心角色，尤其是在政治冲突的背景下，视觉与文本线索相结合以传递具有说服力的信息。我们使用视觉-语言模型（VLM）研究乌克兰梗图中的说服技术检测。我们介绍了UkrMeme——一个包含200张自然产生的乌克兰战时梗图并针对说服技术进行标注的数据集，以及PTM-UA——Propaganda 950基准的一个子集的乌克兰语翻译版本，该版本在翻译梗图文本的同时保留了原始视觉内容。我们在零样本提示设置下，使用分层F1指标，评估了视觉-语言模型在图像、文本和多模态输入上的表现。最后，我们比较了逐层相关性传播（LRP）与模型生成的词元归因，并分析了各种解释方法所强调证据的语义类别。

    arXiv:2610.13783v1 Announce Type: new  Abstract: Memes play a central role in modern online communication, especially in contexts of political conflict where visual and textual cues combine to convey persuasive messages. We study persuasive technique detection in Ukrainian memes using vision-language models (VLMs). We introduce UkrMeme, a dataset of 200 organically produced Ukrainian wartime memes annotated for persuasion techniques, and PTM-UA, a Ukrainian translation of a subset of the Propaganda 950 benchmark that preserves original visual content while translating meme text. We evaluate VLMs in a zero-shot prompting setup across image, text, and multimodal inputs using hierarchical F1 metrics. Finally, we compare Layer-wise Relevance Propagation (LRP) and model-generated token attributions and analyze the semantic categories of evidence emphasized by each explanation method.
    
[^15]: 只会打分而不会搜索的特权教师：面向推理的批评条件化在策略自蒸馏

    A Privileged Teacher That Only Scores Cannot Search: Critique-Conditioned On-Policy Self-Distillation for Reasoning

    [https://arxiv.org/abs/2610.13744](https://arxiv.org/abs/2610.13744)

    本文发现自蒸馏中的评分信号既无法定位学生推理中的错误，也无法可靠区分正确与错误的rollout，并指出错误定位本质上是搜索问题，因而提出批评条件化的在策略自蒸馏方法来弥补“只会打分、不会搜索”的评分教师的结构性缺陷。

    

    在策略自蒸馏有望用密集的token级信用信号取代可验证奖励强化学习中稀疏的结果信号：一个以参考解为条件的教师对学生在策略生成的rollout进行评分，由此产生的分布偏移再被蒸馏回策略中。我们研究这种评分信号是否真的包含纠正性信用。利用学生rollout中的黄金错误区间，我们检验了自蒸馏目标所依赖的两种信号——全分布KL散度和token对数比——是否具备两个关键性质：能否定位错误，以及能否将概率从错误的学生token上推开。结果表明，这两种信号均不具备上述性质，也无法在同一问题内可靠地区分正确与错误的rollout。我们认为这种失败是结构性的：在推理轨迹中定位错误本身就是一个搜索问题，而评分教师从未被明确要求执行这一任务。

    arXiv:2610.13744v1 Announce Type: new  Abstract: On-policy self-distillation promises to replace the sparse outcome signal of verifiable-reward RL with dense token-level credit: a teacher conditioned on the reference solution scores the student's own rollout, and the resulting distributional shift is distilled back into the policy. We investigate whether this scoring signal actually contains corrective credit. Using gold error spans in student rollouts, we test the two signals consumed by self-distillation objectives, the full-distribution KL and the token log-ratio, for two properties: whether they localize errors and whether they push probability away from erroneous student tokens. They exhibit neither of these properties nor can they reliably distinguish correct from incorrect rollouts within the same problem. We argue that this failure is structural: locating an error in a reasoning trace is itself a search problem, while a scoring teacher is never explicitly asked to perform that 
    
[^16]: 通过知识图谱引导的多任务预训练增强预训练语言模型中的法律推理能力

    Enhancing Legal Reasoning in Pre-trained Language Models via Knowledge Graph-Guided Multi-Task Pre-training

    [https://arxiv.org/abs/2610.13722](https://arxiv.org/abs/2610.13722)

    提出了一种结合下一句预测、知识图谱引导聚类和知识图谱感知段落分类的多任务预训练方法，将结构化法律知识直接编码进模型权重，从而实现无需图遍历的零样本法律推理。

    

    法律语言理解需要对半结构化的法规文件进行推理，然而标准的预训练目标并未明确针对这种结构化能力。由于当前的法律语言模型仅依赖掩码语言建模，它们将法条文本视为扁平的词元序列，而忽略了底层的层级约束。为了弥合这一差距，我们提出了一种针对德国交通法规（StVO）的多任务预训练方法。我们的方法结合了三个互补的预训练目标：下一句预测（NSP）、知识图谱（KG）引导的聚类（CLU）以及知识图谱（KG）感知的段落分类（CLS）。该方法将结构化的法律知识显式地直接编码到模型权重中。这种编码方式使模型能够在推理时进行零样本法律推理，并且无需离散的图遍历。我们在九个基于Transformer的编码器和324个实验设置上对该方法进行了评估。

    arXiv:2610.13722v1 Announce Type: new  Abstract: Legal language understanding requires reasoning over semi-structured regulatory documents, yet standard pre-training objectives do not explicitly target this structural capability. Because current legal language models rely solely on masked language modeling, they treat statutory text as flat token sequences and ignore the underlying hierarchical constraints. To bridge this gap, we present a multi-task pre-training approach for German traffic law (StVO). Our approach combines three complementary objectives: Next Sentence Prediction (NSP), Knowledge Graph (KG) Guided Clustering (CLU), and Knowledge Graph (KG) Aware Paragraph Classification (CLS). This approach explicitly encodes structured legal knowledge directly into the model weights. This encoding enables zero-shot legal reasoning at inference time and removes the need for discrete graph traversal. We evaluate our approach across nine Transformer-based encoders and 324 experimental ru
    
[^17]: 自适应 DHiSparse：基于负载自适应 KV 分层结构的推测解码

    Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy

    [https://arxiv.org/abs/2610.13708](https://arxiv.org/abs/2610.13708)

    Adaptive DHiSparse 在分层稀疏 KV 架构上支持推测解码，并根据负载动态调整 KV 缓存驻留策略，在 100K 上下文、128 并发下相比 HiSparse 将解码吞吐提升至 1.45 倍。

    

    长上下文推理服务往往在硬件达到有效吞吐量之前就耗尽了 KV 缓存内存。HiSparse 将历史 KV 存储在主机内存中，仅取回被选中的条目，但它在任何负载下都使用单一的缓存预算，并且不支持诸如 DSpark 和 DFlash2 等基于草稿模型的推测解码。我们提出了 Adaptive DHiSparse，它支持在分层稀疏 KV 架构上进行推测解码，并根据负载自适应地调整 KV 驻留策略。它保护必需的工作集，在 GPU 内存有富余时保留更多的历史记录，并通过精确的融合检索恢复其余部分。DSpark 和 DFlash2 的提议与接受规则保持不变：查询共享缺失行的检索，草稿 KV 使用窗口受限的环形缓冲区，验证暂存在层组之间被复用。在 100K 上下文和 128 并发条件下，GLM-5.3 配合 DFlash2 在每个解码 GPU 上达到 793 输出 token/秒，是 HiSparse 的 1.45 倍、非推测自适应服务的 1.27 倍。（摘要在此处截断）

    arXiv:2610.13708v1 Announce Type: new  Abstract: Long-context serving runs out of KV memory before the hardware reaches its useful throughput. HiSparse stores history in host memory and fetches only the selected entries, but it uses one cache budget at every load and does not handle draft-model speculative decoding such as DSpark and DFlash2. We present Adaptive DHiSparse, which supports speculative decoding over hierarchical sparse KV and adapts residency to load. It protects the required working set, retains additional history while GPU memory is spare, and recovers the rest through exact fused retrieval. DSpark and DFlash2 keep their proposal and acceptance rules: queries share missing-row retrieval, draft KV uses a window-bounded ring, and verification staging is reused across layer groups. At 100K context and concurrency 128, GLM-5.3 with DFlash2 delivers 793 output tokens/s per decode GPU, 1.45$\times$ HiSparse and 1.27$\times$ non-speculative adaptive serving. Held to the same l
    
[^18]: 基于结构驱动知识图谱的法律推理可验证检索

    Verifiable Retrieval for Legal Reasoning through Structure-driven Knowledge Graph

    [https://arxiv.org/abs/2610.13695](https://arxiv.org/abs/2610.13695)

    本文提出利用结构驱动解析为德国法律法典构建知识图谱，通过比较多种基于知识图谱的分块与检索策略，提升了法律推理场景中引用检索的准确性和可验证性。

    

    在医学或法律等可靠性至关重要的场景中，任何自动化系统都必须是透明且可验证的。尽管当前基于大语言模型（LLM）的系统具有诸多优势，但由于其固有的黑箱特性，它们无法满足这些要求。检索增强生成（RAG）试图将生成过程建立在相关可解释文档的基础上，从而降低AI“幻觉”的风险，并有可能实现对初始生成上下文的验证。然而，随着潜在信息来源数量和规模的增加，朴素的检索方法往往表现变差，会忽略关键的上下文元素。本文旨在德国法律法典的背景下解决这一问题。我们使用结构驱动的解析方法创建知识图谱（KG）用于源文档的导航和检索，比较了多种基于知识图谱的分块和检索策略，并评估了它们对检索引用准确性的影响。

    arXiv:2610.13695v1 Announce Type: new  Abstract: Any automation within reliability-critical contexts such as medicine or law has to be transparent and verifiable. Despite their advantages, current LLM-based systems do not meet these requirements due to their inherent black-box nature. Retrieval Augmented Generation (RAG) attempts to ground the generative process in the relevant interpretable documents, thereby reducing the risk of AI 'hallucinations' and potentially enabling verification of the initial generation context. However, as the number and size of potential sources increases, naive retrieval approaches tend to perform worse, ignoring critical context elements. In this paper we aim to tackle this problem in the context of German Legal Code. Using structure-driven parsing to create a Knowledge Graph (KG) for the source document navigation and retrieval, we compare several KG-informed chunking and retrieval strategies and evaluate their effect on the accuracy of retrieved citatio
    
[^19]: 撤稿研究再审视：欺诈科学中语言模糊化效应的复制、扩展与机制评估

    Retractions Revisited: A Replication, Extension, and Mechanistic Evaluation of Linguistic Obfuscation Effects in Fraudulent Science

    [https://arxiv.org/abs/2610.13678](https://arxiv.org/abs/2610.13678)

    本研究通过比较800篇欺诈性与800篇真实性科学论文，复制并证实了欺诈论文具有更高语言模糊化的结论，并通过并行中介模型发现“现实监测”机制（体现为形容词使用减少）是解释这一现象的最佳心理机制。

    

    先前的研究表明，欺诈性科学家撰写的文章比真正的科学家使用更高比率的语言模糊化表达，一种可能的机制认为模糊化会增加读者分析研究的成本。这一结果尚未在开放科学时代接受进一步的实证检验，其他心理机制也未曾被评估。本研究通过复制与扩展工作弥补了这一空白，比较了800篇欺诈性论文和800篇真实的对照论文（2013-2025年）。结果支持了语言模糊化假设，表明欺诈性论文比真实论文更加模糊。一个并行中介模型评估了三种相互竞争的机制，发现通过形容词使用率降低所衡量的“现实监测”机制最能解释欺诈科学与语言模糊化之间的联系。基于模糊化特征的交叉验证检测准确率为60.7%，并采用弹性（摘要原文在此处截断）

    arXiv:2610.13678v1 Announce Type: new  Abstract: Prior work suggests fraudulent scientists write their articles with higher rates of linguistic obfuscation than genuine scientists, with a possible mechanism suggesting obfuscation makes research more costly for a reader to analyze. This result has not undergone additional empirical scrutiny in the open science era, nor have other psychological mechanisms been evaluated. The current replication and extension addressed this opportunity by comparing 800 fraudulent and 800 genuine control papers (2013-2025). The results supported the linguistic obfuscation hypothesis, suggesting fraudulent papers were more obfuscated than genuine ones. A parallel mediation model evaluated three competing mechanisms, finding that reality monitoring, as measured through decreased rates of adjectives, best explained the link between fraudulent science and linguistic obfuscation. Cross-validated detection accuracy using obfuscation features (60.7%) and an elast
    
[^20]: 注入轮次而非标记：学习在何处重定向大语言模型推理以实现自我纠错

    Inject a Turn, Not a Token: Learning Where to Redirect LLM Reasoning for Self-Correction

    [https://arxiv.org/abs/2610.13672](https://arxiv.org/abs/2610.13672)

    该论文提出无需微调的 UPTURN 自我纠错框架，通过仅用最终答案正确性训练的 13 万参数轻量 MLP 头学习失败推理中的最佳干预位置，并将纠错指令作为新的用户轮次而非行内标记注入，从而有效重定向模型推理，在四个推理模型和六个数学基准上均提升准确率。

    

    大语言模型难以可靠地识别并纠正自身推理中的错误。与以往通过微调大语言模型来生成修订或反馈的训练类方法不同，我们学习在失败的推理生成过程中于何处进行干预，并确定如何对其重定向。为了选择干预位置，我们在冻结的编码器上仅利用最终答案的正确性训练了一个 13 万参数的 MLP 头。对于重定向方式，我们发现在固定的生成预算下，已测试的行内注入无法可靠地改变推理方向，而将相同指令作为新的用户轮次呈现则能改变模型的走向。由此得到的纠错框架 UPTURN（在习得位置注入用户提示轮次）无需对大语言模型进行任何微调：它保留选定的前缀，将一条指令作为新的用户轮次注入，并重新生成后缀。在四个推理模型和六个数学基准上，UPTURN 在所有设置下均提升了平均准确率。

    arXiv:2610.13672v1 Announce Type: cross  Abstract: Large language models struggle to reliably identify and correct errors in their own reasoning. Unlike prior training-based approaches that fine-tune an LLM to generate revisions or feedback, we learn where to intervene in failed rollouts and determine how to redirect them. To choose where to intervene, we train a 130K-parameter MLP head on a frozen encoder using only final-answer correctness. For redirection, we find that under a fixed generation budget, the tested inline injections fail to reliably redirect reasoning, whereas presenting the same instruction in a new user turn changes the model's course. The resulting correction framework, UPTURN (User-Prompt TURN injection at a learned position), requires no LLM fine-tuning: it retains a selected prefix, injects an instruction as a new user turn, and regenerates the suffix. Across four reasoners and six math benchmarks, UPTURN improves average accuracy in every setting over a 32-rollo
    
[^21]: 用于专用神经网络模型自动调优的多智能体LLM系统

    A Multi-Agent LLM System for Automated Tuning of Specialized Neural Network Models

    [https://arxiv.org/abs/2610.13663](https://arxiv.org/abs/2610.13663)

    本文提出一个由分析师、编码员和记录员组成的多智能体LLM框架，在保持整体网络结构不变的前提下自动生成并调优指定关键组件的代码，并可迁移至脉冲神经网络分类与PINN求解高阶微分方程等不同类型模型。

    

    许多机器学习方法都是两级结构：专家先选定一类方法，而在该类方法内部，关键组件的选择至关重要且通常依赖人工。本文致力于实现这些组件的自动调优，并提出了一种可跨不同性质模型迁移的框架。该多智能体大语言模型（LLM）框架包含三个角色——分析师、编码员和记录员——并配备实验日志和沙盒执行环境。与优化整个机器学习脚本或在离散空间中搜索架构的系统不同，该框架在一个开放但语义上狭窄的空间中运作：智能体仅为指定的组件生成代码，而保持整体网络结构不变。该方法在两个方向上得到验证：调优基于相关性学习的分类脉冲神经网络，以及使用物理信息神经网络（PINN）求解高阶微分方程。

    arXiv:2610.13663v1 Announce Type: new  Abstract: Many machine learning approaches are two-level: an expert selects a class of methods, within this class the choice of key components is critical and usually manual. This paper addresses the automatic tuning of these components and proposes a framework transferable across models of different nature. The multi-agent large language model (LLM) framework has three roles---analyst, coder, and recorder~---with an experiment log and a sandboxed execution environment. Unlike systems that optimize the entire ML script or search architectures in a discrete space, the framework operates over an open but semantically narrow space: the agent generates code only for the specified components, leaving the overall network structure unchanged. The approach is validated in two directions: tuning a classification spiking neural network with correlation learning, and solving high-order differential equations with Physics-Informed Neural Networks (PINN). Acro
    
[^22]: 多流注意力残差

    Multi-Stream Attention Residuals

    [https://arxiv.org/abs/2610.13630](https://arxiv.org/abs/2610.13630)

    提出多流注意力残差（MSAR），将注意力残差泛化为多头深度注意力机制，通过多尺度因果卷积从序列上下文获取流多样性，并经轻量级自适应混合组合各流，以极小的额外开销实现了显著的性能提升。

    

    注意力残差能够在网络深度方向上实现依赖于内容的检索，可视为沿深度轴的单流softmax注意力。受沿序列轴的多头注意力取得成功的启发，我们提出了多流注意力残差，这是AttnRes的一种扩展，将该机制泛化为多头深度注意力，以极小的额外开销带来显著的性能提升。MSAR通过多尺度因果卷积从序列上下文中获取流的多样性，使不同的头能够通过不同的上下文视图检索历史表示，同时保持隐藏表示的完整性。检索到的多个流通过轻量级自适应混合进行组合，以生成单流层的输入。通过保留单流源并按需进行扩展，MSAR无需持久的多流计算即可支持多流计算。

    arXiv:2610.13630v1 Announce Type: new  Abstract: Attention Residuals (AttnRes) enables content-dependent retrieval over network depth, which can be viewed as single-stream softmax attention along the depth axis. Inspired by the success of multi-head attention along the sequence axis, we introduce Multi-Stream Attention Residuals (MSAR), an extension of AttnRes that generalizes this mechanism to multi-head depth attention, yielding substantial performance gains with minimal additional overhead. MSAR draws stream diversity from sequence context through multi-scale causal convolutions, allowing different heads to retrieve historical representations through distinct contextual views while preserving the integrity of the hidden representation. The retrieved streams are combined through lightweight adaptive mixing to generate the single-stream layer input. By retaining single-stream sources and expanding them on demand, MSAR supports multi-stream computation without requiring persistent mult
    
[^23]: AMLworld-Compact：用于低成本大语言模型评估与错误诊断的重要性加权降采样方法

    AMLworld-Compact: Importance-Weighted Downsampling for Cost-Effective LLM Evaluation and Error Diagnosis

    [https://arxiv.org/abs/2610.13627](https://arxiv.org/abs/2610.13627)

    提出AMLworld-Compact评估集，通过基于监督评分分层的HT-Coreset重要性加权降采样协议，将AMLworld上LLM评估目标减少最高611倍，并利用Horvitz-Thompson权重实现对完整测试集性能指标的无偏估计。

    

    在AMLworld最小的测试集划分上评估大语言模型（LLM）成本高昂，因为超过一百万个目标中的每一个都需要单独的图提示。采样可以降低这一成本，但小型平衡子集会改变未加权的精确率和F1分数。我们提出了AMLworld-Compact，这是一个评估集，它保留了所有非法测试交易，并从监督评分分层中采样良性交易。其HT-Coreset协议在HI-Small和LI-Small上分别将评估目标减少了271倍和611倍。Horvitz-Thompson（HT）权重可为固定预测提供无偏的完整测试混淆计数估计；召回率是精确的，而精确率和F1是比率估计。在使用图特征预处理器（GFP）特征的留出研究中，一个提升树家族定义分层来评估另一个家族，每种配置进行500次抽取。在发布的数据规模下，F1的RMSE为0.00-2.58个百分点，而使用HT权重的均匀采样则为4.44-7.41个百分点。

    arXiv:2610.13627v1 Announce Type: new  Abstract: Evaluating large language models (LLMs) on AMLworld's smallest test splits is costly because each of over a million targets requires a separate graph prompt. Sampling reduces this cost, but small balanced subsets change unweighted precision and F1. We introduce AMLworld-Compact, an evaluation set that retains every illicit test transaction and samples benign transactions from supervised-score strata. Its HT-Coreset protocol reduces evaluation targets by 271x and 611x on HI-Small and LI-Small. Horvitz-Thompson weights give unbiased full-test confusion-count estimates for fixed predictions; recall is exact, while precision and F1 are ratio estimates. In a held-out study using Graph Feature Preprocessor (GFP) features, one boosted-tree family defines strata for evaluating the other over 500 draws per configuration. At the released sizes, F1 RMSE is 0.00-2.58 percentage points versus 4.44-7.41 for uniform sampling with HT weights; conservati
    
[^24]: APEX-Harm：评估共享专业工作空间中的智能体安全性

    APEX-Harm: Evaluating Agent Safety in Shared Professional Workspaces

    [https://arxiv.org/abs/2610.13623](https://arxiv.org/abs/2610.13623)

    该论文提出了APEX-Harm基准，在包含上百个文件和多种工具的共享专业工作空间中，通过六类提示与脚本注入攻击评估执行金融、咨询和法律任务的长时程智能体的安全性，对12个前沿模型的测试显示其仍存在显著安全风险。

    

    专业智能体越来越多地通过命令行界面和多种工具，在包含文件、消息和业务应用的共享工作空间中开展工作。这类环境带来的安全风险，难以通过放置在单一已知读取位置的攻击来捕捉。我们提出了APEX-Harm，这是一个用于评估提示注入和脚本注入攻击的基准测试，针对的是在大型工作空间中执行金融、咨询和法律任务的长时程智能体。每个任务包含115到328个文件，并配有通过MCP服务器暴露的电子邮件、日历、聊天、文档、电子表格、PDF和代码执行工具。该基准涵盖六类攻击，包括静态与动态注入、任务特定脚本替换以及攻击者提供的通用脚本，并从攻击成功率、暴露程度和任务性能三个维度进行评估。我们在每个模型180个任务实例上评估了12个前沿模型。Opus 5和Sonnet 5的总体（原文在此处截断）。

    arXiv:2610.13623v1 Announce Type: cross  Abstract: Professional agents increasingly work through command-line interfaces and multiple tools over shared workspaces containing files, messages, and business applications. These environments create safety risks that are difficult to capture with attacks placed at a single, known read location. We introduce APEX-Harm, a benchmark for evaluating prompt and script injection against long-horizon agents performing finance, consulting, and legal tasks in large workspaces. Each task contains 115 to 328 files, together with email, calendar, chat, document, spreadsheet, PDF, and code-execution tools exposed through MCP servers. The benchmark includes six attack categories covering static and dynamic injections, task-specific script replacement, and attacker-provided generic scripts, and evaluates attack success, exposure, and task performance. We evaluate 12 frontier models on 180 task instances per model. Opus 5 and Sonnet 5 have the lowest overall
    
[^25]: 基于潜在内在反馈训练用户模拟器

    Training User Simulators with Latent Intrinsic Feedback

    [https://arxiv.org/abs/2610.13617](https://arxiv.org/abs/2610.13617)

    提出了一种具有潜在内在反馈的用户模拟器，通过在每轮对话中以第一人称视角生成内部想法、反应和评分等潜在评估状态来评估助手，显著提升了模拟的真实性和对助手回复的批判能力。

    

    用户模拟器作为训练大语言模型（LLM）助手的环境正受到越来越多的关注，这些助手能够更好地应对与真实用户进行多轮对话时的细微差别。以往关于用户模拟的研究侧重于生成逼真的用户话语，但尚未探索其作为对助手回复提供直接反馈来源的潜力。这类评估信号可能不会以语言形式显式表达，而是存在于用户的潜在推理之中。在本工作中，我们引入了具有内在反馈的用户模拟器，即配备了潜在评估状态的模型：这些状态包括内部想法、反应和标量评分，能够在每轮对话中以第一人称用户视角对助手进行评估。我们使用真实用户与助手之间的交互日志来训练基于大语言模型的模拟器，并证明与通过提示构建的模拟器基线相比，内在反馈提升了模拟的真实性和批判行为。

    arXiv:2610.13617v1 Announce Type: new  Abstract: User simulators are gaining attention as environments to train LLM assistants that can better cope with the nuances of multi-turn conversations with real users. Prior work on user simulation has emphasized generating realistic user utterances, leaving unexplored their potential as a source of direct feedback on the assistant's responses. Such evaluation signals may not be explicitly communicated in language, but can instead reside in the latent reasoning of users. In this work, we introduce user simulators with intrinsic feedback, namely, models equipped with latent evaluation states: internal thoughts, reactions, and scalar ratings that evaluate the assistant from a first-person user perspective at each conversation turn. We train LLM-based simulators using interaction logs between real users and assistants and demonstrate that intrinsic feedback improves simulation realism and critiquing behaviors versus prompted simulator baselines, w
    
[^26]: BLiMPPI：极性项语言学最小对立对基准

    BLiMPPI: Benchmark of Linguistic Minimal Pairs for Polarity Items

    [https://arxiv.org/abs/2610.13607](https://arxiv.org/abs/2610.13607)

    本文提出了BLiMPPI——一个涵盖33种语言、包含超过12,000个否定极性项最小对立对的多语言评测基准，揭示了多语言模型在低资源语言以及需要深层句法语义知识的NPI现象上表现最差。

    

    否定极性项（NPI），如"anymore"和"ever"，仅在特定环境中才符合语法，主要出现在否定的语境中。由于其语法性同时取决于句法和语义，它们在理论语言学中拥有丰富的多语言文献，然而语言模型中关于NPI的研究几乎完全局限于英语。我们提出了BLiMPPI，一个涵盖33种语言、包含超过12,000个NPI最小对立对的多语言评估套件。该套件由一个自动化流程构建，该流程在通用依存树库中发现NPI，标注其许可环境，并将其扰动为最小对立对。通过对12个多语言模型和2个单语模型以及n-gram基线进行评估，我们发现多语言模型在低资源语言以及无法通过n-gram方法解决的最小对立对上表现最差。我们进一步检验了NPI类型学、语系、许可变异性以及NPI与许可词之间距离的影响。

    arXiv:2610.13607v1 Announce Type: new  Abstract: Negative polarity items (NPIs), such as "anymore" and "ever", are grammatical only in select environments, predominantly in the context of negation. Because their grammaticality depends on both syntax and semantics, they have a rich, multilingual literature in theoretical linguistics, yet work on NPIs in language models is almost entirely confined to English. We introduce BLiMPPI, a multilingual evaluation suite of over 12,000 NPI minimal pairs in 33 languages, built by a pipeline that discovers NPIs in Universal Dependencies (UD) treebanks, labels their licensing environments, and perturbs these into minimal pairs. Evaluating 12 multilingual and 2 monolingual models alongside n-gram baselines, we find that multilingual models struggle most with lower-resource languages and with pairs that are not n-gram solvable. We further examine the effects of NPI typology, language family, licensing variability, and NPI-licenser distance. Data: http
    
[^27]: zip2zip++：迈向以最小质量损失实现推理时词元压缩

    zip2zip++: Towards Inference-Time Token Compression with Minimal Quality Loss

    [https://arxiv.org/abs/2610.13605](https://arxiv.org/abs/2610.13605)

    该论文提出 zip2zip++，通过分离输入/输出嵌入、修正 RoPE 位置编码、避免合并数字等针对性改进压缩层与 LZW 编解码器，在不修改预训练模型的前提下实现推理时词元压缩，并将质量损失降至最低。

    

    通过 Lempel-Ziv-Welch（LZW）压缩即时构建的动态词表，可以让语言模型用更少的词元表示相同的文本，但先前的研究表明，这种效率提升是以推理能力和困惑度为代价的。我们证明，这种质量差距并非压缩的固有代价，而是可以通过对压缩层和 LZW 编解码器的针对性改进来大幅缩小，且无需修改底层的预训练语言模型。我们的分析揭示了四个关键问题：(1) LZW 的压缩算法在输入嵌入和输出嵌入之间造成了隐式冲突，这种张力是单一共享嵌入无法解决的；(2) 将 RoPE 简单地应用于 LZW 压缩后的序列，忽略了合并后的词元可能跨越多个原始词元这一事实；(3) 合并某些词元（尤其是数字）会不成比例地损害下游准确率；(4) 动态词表引入了固有的分段歧义，从而对下游……（摘要在此处截断）

    arXiv:2610.13605v1 Announce Type: new  Abstract: Dynamic vocabularies built on-the-fly via Lempel-Ziv-Welch (LZW) compression let a language model represent the same text in fewer tokens, but prior work shows that this efficiency comes at a cost to reasoning and perplexity. We show that this quality gap is not a fundamental price of compression but can be substantially closed through targeted improvements to the compression layer and LZW codec, without modifying the underlying pretrained language model. Our analysis reveals four key issues: (1) LZW's compression algorithm creates an implicit conflict between the input embedding and the output embedding, a tension that a single shared embedding cannot resolve; (2) naively applying RoPE to LZW-compressed sequences ignores that merged tokens can span multiple tokens; (3) merging certain tokens (notably digits) disproportionately hurts downstream accuracy; and (4) dynamic vocabularies introduce an inherent segmentation ambiguity that confo
    
[^28]: 测量机器生成文本检测器对伪影的依赖

    Measuring Artifact Reliance in Machine-Generated Text Detectors

    [https://arxiv.org/abs/2610.13583](https://arxiv.org/abs/2610.13583)

    该论文揭示机器生成文本检测器在科学文献基准上的近乎完美表现，很大程度上依赖于文档组装时引入的PDF提取伪影等表面格式特征（仅凭这些特征逻辑回归即可超过现有检测器），并提出“格式反事实”方法来区分包含此类信号的基准与依赖此类信号的检测器。

    

    arXiv:2610.13583v1 公告类型： new 摘要：机器生成文本检测器在科学文献基准测试上的得分近乎完美，这意味着我们可以依靠这些检测器可靠地区分人类撰写与机器生成的文档。我们研究了来自六个常用资源的十个科学文档子集，发现在其中几个子集上，这种表面上的轻易表现其实是文档组装方式带来的副作用。一个仅使用27个表面特征的逻辑回归模型就达到了0.833-0.997的宏平均F1，而在三个arXiv子集上，它的表现超过了我们评估的所有检测器，F1分数为0.951-0.997。对分类最强的线索是PDF提取伪影：人类文本保留了原始的换行符，而生成文本则被提取为单一连续的文本块。这一线索与其他三个格式特征结合，能以0.996的AUC区分两类文本。为了区分包含此类信号的基准与依赖此类信号的检测器，我们提出了格式反事实方法：只需……

    arXiv:2610.13583v1 Announce Type: new  Abstract: Detectors of machine-generated text score almost perfectly on scientific-literature benchmarks, implying we can reliably distinguish with these detectors human and machine-generated documents. We looked at ten scientific-document subsets from six popular resources, and on several of them that apparent ease is a side effect of how the documents were put together. A logistic regression using 27 superficial features reached 0.833-0.997 macro-F1, and on the three arXiv subsets it outperformed every detector we evaluate with F1-scores 0.951-0.997. The strongest cue to the detector is the PDF extraction artifact: human texts retain their original line breaks, whereas generated texts are extracted as a single block of prose. This cue, together with three other formatting features, tell the classes apart at 0.996 AUC. To distinguish benchmarks that contain such a signal from detectors that rely on one, we introduce format counterfactuals: just r
    
[^29]: 并非所有正确的修复运行时表现都相同：仓库级问题解决中补丁性能的实证研究

    Not All Correct Fixes Run Alike: An Empirical Study of Patch Performance in Repository-Level Issue Resolution

    [https://arxiv.org/abs/2610.13563](https://arxiv.org/abs/2610.13563)

    该研究将基准自带的 FAIL_TO_PASS 测试重新用于评估补丁的运行时性能，发现通过正确性测试的不同补丁在运行性能上存在显著差异，即并非所有正确的修复运行起来都一样。

    

    摘要：诸如 SWE-bench 等仓库级问题解决基准主要依据提交的补丁是否通过正确性测试来进行评估，而对于不同通过测试的补丁的运行时性能，在很大程度上尚未被探索。评估补丁性能至关重要但充满挑战，因为为每个基准实例设计性能测试工作负载的成本很高。然而，这些基准本身已经提供了一个天然的候选方案：内置的 FAIL_TO_PASS（F2P）测试，它会触发受问题修复影响的行为。因此，我们研究了所提供的 F2P 测试能否被重新用于评估补丁性能。总体而言，我们选择了四个前沿模型——Claude Opus 5、GPT 5.6、DeepSeek V4 Pro 和 Qwen 3.8 Max——并与 SWE-agent 配对，解决从 SWE-bench 和 SWE-bench-Live 已验证分割中预先筛选出的可测量语料库，最终得到 262 个至少包含两种不同正确解决方案的案例，其中包括开发（测试）……（原文摘要在此处截断）

    arXiv:2610.13563v1 Announce Type: cross  Abstract: Repository-level issue-resolution benchmarks such as SWE-bench evaluate submitted patches primarily based on whether they pass the correctness tests, leaving the runtime performance of different test-passing patches largely unexplored. Evaluating patch performance is essential but challenging since designing performance workloads for each benchmark instance is costly. However, these benchmarks already provide a natural candidate: the built-in \emph{FAIL\_TO\_PASS} (F2P) tests, which exercise the behavior affected by the issue fix. We therefore investigate whether the provided F2P tests can be repurposed to evaluate patch performance. Overall, we choose four frontier models, Claude Opus~5, GPT~5.6, DeepSeek V4 Pro, and Qwen~3.8 Max, paired with SWE-agent to resolve the pre-filtered measurable corpus from verified splits of SWE-bench and SWE-bench-Live, yielding 262 cases with at least two different correct solutions, including the devel
    
[^30]: 个性化而不刻板印象：语音大语言模型知道何时该“充耳不闻”吗？

    Personalization Without Stereotyping: Do Speech LLMs Know When to Turn a Deaf Ear?

    [https://arxiv.org/abs/2610.13550](https://arxiv.org/abs/2610.13550)

    提出了 BASIS-Bench 基准，通过人口统计稳健性、刻板印象偏见和个性化三个维度区分语音大语言模型的有益个性化与刻板印象行为，评估发现多数模型虽能识别人口统计线索却未能恰当运用。

    

    近期的语音大语言模型（Speech LLM）能够利用丰富的副语言线索，实现更复杂和个性化的交互。然而，很少有研究调查这种敏感性如何导致刻板印象行为，且现有评估大多仅衡量人口统计差异，未能区分有益的个性化与刻板印象行为。我们提出了 BASIS-Bench，这是一个旨在通过三个维度区分这两个方面的基准：（1）人口统计稳健性，衡量基于内容的任务在不同说话人之间的表现差异以及人口统计分类能力；（2）刻板印象偏见，衡量感知到的人口统计线索是否会触发未经提示的刻板印象行为；（3）个性化，衡量模型在被明确要求时能否根据人口统计特征定制响应。我们评估了八个近期的语音大语言模型，发现尽管它们通常能够识别人口统计线索，但大多数模型未能将其转化为恰当的行为。

    arXiv:2610.13550v1 Announce Type: new  Abstract: Recent speech LLMs can leverage rich paralinguistic cues, enabling more complex and personalized interactions. However, little work investigates how this sensitivity can lead to stereotyping, and existing evaluations largely measure demographic disparities without distinguishing beneficial personalization from stereotypical behavior. We introduce BASIS-Bench, a benchmark designed to disentangle these two aspects through three axes: (1) demographic robustness, measuring disparities in content-based tasks across speakers and demographic classification capabilities; (2) stereotypical bias, measuring whether perceived demographic cues trigger unprompted stereotypical behavior; and (3) personalization, measuring whether models can tailor responses to demographic characteristics when explicitly requested. We evaluate eight recent speech LLMs and find that, despite often recognizing demographic cues, most fail to translate them into appropriate
    
[^31]: S-Space：探索多模态语言模型中的空间工作空间

    S-Space: Exploring Spatial Workspace in Multimodal Language Models

    [https://arxiv.org/abs/2610.13527](https://arxiv.org/abs/2610.13527)

    该论文发现多模态语言模型内部存在一个固有的线性子空间"S-Space"，作为模型的内部空间工作空间，以连续坐标编码物体位置，并支撑感知编码、视角变换和言语输出等空间推理的全过程。

    

    多模态语言模型（MLLMs）展现出有前景的空间推理能力，然而支持这些能力的内部机制仍然知之甚少。我们发现了S-Space——一个线性表示子空间，其中物体标记的表示沿着水平、垂直和距离三个轴编码连续坐标。尽管我们通过对物体标记激活求空间答案得分的梯度来识别这些轴，但S-Space是模型内在的，而非由特定数据或提示诱导产生，在非空间问题下依然保持可读性。我们证明S-Space充当着多模态语言模型的内部空间工作空间：感知过程将物体位置编码到其中，形成可直接读取以支持准确空间判断的空间表示；内部推理过程对其进行变换以实现视角转换；言语响应则选择性地访问它——因果干预实验证实了这一点，干预会改变空间判断，同时（在很大程度上不影响其他能力）。

    arXiv:2610.13527v1 Announce Type: cross  Abstract: Multimodal language models (MLLMs) exhibit promising spatial reasoning capabilities, yet the internal mechanisms supporting them remain poorly understood. We uncover S-Space, a linear representation subspace in which object-token representations encode continuous coordinates along horizontal, vertical, and distance axes. Although we identify these axes by differentiating spatial-answer scores with respect to object-token activations, S-Space is intrinsic rather than induced by specific data or prompts, remaining readable under non-spatial questions. We show that S-Space functions as an internal spatial workspace of MLLMs. Perception encodes object locations into it, forming spatial representations that can be directly read out to support accurate spatial judgments. Internal reasoning transforms it for changing viewpoints. Verbal responses selectively access it, evidenced by causal interventions that alter spatial judgments while largel
    
[^32]: 面向临床预后预测的检索增强异构证据融合多模态语言模型

    Retrieval-Augmented Multimodal Language Model with Heterogeneous Evidence Integration for Clinical Prognosis

    [https://arxiv.org/abs/2610.13510](https://arxiv.org/abs/2610.13510)

    该论文提出了PRISM，一种通过双模态人群记忆库在表示空间中检索相似病例嵌入、并融合异构临床证据的检索增强多模态语言模型，在包含1,443个任务的MDS-ED临床预后基准上，于诊断、病情恶化、ICU入院和死亡率预测等多项任务中取得了优异表现。

    

    临床预后预测需要整合结构化测量指标、生理波形、纵向病史以及人群背景信息。我们提出了PRISM（基于人群检索的结构化多模态预后推断），这是一种多模态语言模型，它结合了逐特征数值标记化、纵向心电图的近期加权聚合以及双模态人群记忆库。检索在表示空间中进行：模型接收相似训练队列就诊记录的均值池化嵌入，而任何单独的检索记录都不会进入提示词中。在包含1,443个任务的MDS-ED基准测试上，PRISM在诊断任务中达到86.2%的AUROC，在病情恶化预测中达到93.1%，在ICU入院预测中达到91.8%，在死亡率预测中达到92.8%。这些结果使用了96小时的到院前时间窗，而所报告的基线方法无法获得该时间窗的数据。仅使用基准时间窗内的输入时，PRISM达到0.8895的AUROC，相比之下UniPACT为0.8937，MDS-ED为0.8889。

    arXiv:2610.13510v1 Announce Type: new  Abstract: Clinical prognosis requires integrating structured measurements, physiological waveforms, longitudinal history, and population context. We introduce PRISM (Population-Retrieved Inference for Structured Multimodal Prognosis), a multimodal language model that combines per-feature numerical tokenization, recency-weighted aggregation of longitudinal ECGs, and a dual-modality population memory bank. Retrieval operates in representation space: the model receives mean-pooled embeddings of similar training-cohort encounters, while no individual retrieved record enters the prompt. On the 1,443-task MDS-ED benchmark, PRISM achieves AUROCs of 86.2% for diagnosis, 93.1% for deterioration, 91.8% for ICU admission, and 92.8% for mortality. These results use a 96-hour pre-arrival window unavailable to reported baselines. With benchmark-window inputs only, PRISM reaches 0.8895 AUROC, compared with 0.8937 for UniPACT and 0.8889 for MDS-ED. A matched-inpu
    
[^33]: 主动式智能体编排

    Proactive Agentic Orchestration

    [https://arxiv.org/abs/2610.13485](https://arxiv.org/abs/2610.13485)

    本文提出主动式智能体编排框架POrchestra，允许子智能体在执行过程中随时主动请求编排器调整其目标和工具框架，从而无缝适应动态环境，并据此提出了基于事后洞察的在策略编排蒸馏方法来优化编排策略。

    

    智能体编排将任务委托给具有独立上下文的子智能体，为扩展大语言模型智能体的有效执行时程提供了一条关键路径。然而，现有的编排方法面临两个局限：在执行方面，当前方法在创建时就固定了子智能体的目标和工具框架，限制了其对动态真实环境的适应性；在优化方面，稀疏且延迟的反馈制约了编排策略的有效学习。为解决这些局限，我们提出了POrchestra，一个即时主动式智能体编排框架。该框架允许子智能体在任何逻辑节点上主动与编排器通信以请求调整其工具框架，同时保持局部执行的连续性。这一机制不仅使子智能体能够无缝适应动态环境，还自然地启发了基于事后洞察的在策略编排蒸馏方法。

    arXiv:2610.13485v1 Announce Type: new  Abstract: Agentic orchestration delegates tasks to sub-agents with independent contexts, providing a key path to extending the effective execution horizon of large language model agents. However, existing orchestration methods face two limitations: in execution, current methods fix sub-agents' goals and harnesses at creation time, limiting their adaptability to dynamic real-world environments; in optimization, sparse and delayed feedback constrains effective learning of the orchestration policy. To address these limitations, we introduce POrchestra, an on-the-fly proactive agentic orchestration framework. It allows sub-agents to proactively communicate with the orchestrator at any logical point to request harness adjustments while maintaining continuity of local execution. This mechanism not only enables sub-agents to adapt seamlessly to dynamic environments but also naturally inspires On-Policy Orchestration Distillation (OPOD) based on hindsight
    
[^34]: 更好的度量指标救不了你：比较什么胜过如何比较

    A Better Metric Will Not Save You: What You Compare Beats How You Compare It

    [https://arxiv.org/abs/2610.13476](https://arxiv.org/abs/2610.13476)

    该研究以十五种语言的元音声学组织为基准发现，选择比较哪些类别对表征相似性度量的影响远超度量方法本身——改变所比较的类别会使两个表征间的差距变动超过所测效应的两倍，因此“比较什么”比“如何比较”更重要。

    

    表征相似性度量方法（如RSA、CKA、Procrustes）的选择依据是它们将哪些变换视为无关，但所有这些方法都将条件集合视为既定的。该集合由两个决策决定：哪些输入被纳入以及它们如何分组，而这两个决策都是在度量方法所操作的矩阵存在之前做出的，因此没有任何不变性性质能够触及它们。我们在一个条件应当具有的排列方式可以独立于被比较系统而已知的场景中——即十五种语言、二十七个表征下的元音声学组织——测量了这些决策的价值。在固定类别数量的情况下，改变比较哪些类别会使两个表征之间的差距变动超过所测效应的两倍，而改变类别数量则完全不影响它。在这一设置下，模型的排名在这些选择下是可确定的，而跨语言的排名则不是。

    arXiv:2610.13476v1 Announce Type: new  Abstract: Representational similarity measures (RSA, CKA, Procrustes) are chosen for the transformations they treat as irrelevant, and all of them take the set of conditions as given. That set is fixed by two decisions, which inputs are admitted and how they are grouped, both taken before the matrix a measure operates on exists: no invariance property reaches them. We measure what they are worth where the arrangement the conditions ought to have is known independently of the systems compared, the acoustic organisation of vowels, across fifteen languages and twenty-seven representations. Changing which categories are compared, at a fixed number of them, moves the gap between two representations by more than twice the effect being measured, while changing how many moves it not at all. In this setting rankings of models are identified under these choices, and rankings across languages are not.
    
[^35]: 利用仿真与运行活动的证据丰富设计时模型

    Towards Enriching Design-Time Models with Evidence from Simulation and Operational Activities

    [https://arxiv.org/abs/2610.13456](https://arxiv.org/abs/2610.13456)

    本文提出一种证据驱动的工作流，系统地将仿真与运行阶段产生的执行证据反馈回设计时模型（如影响模型），从而弥补从系统执行结果到持久设计知识丰富之间的隐式路径。

    

    信息物理系统是通过多个基于模型的活动开发而成的，包含多个开发阶段。随着开发过程经历仿真、实验和运行，对系统执行的观察可能揭示出缺失的、不确定的或无效的设计知识。然而，从执行结果到持久的设计时知识丰富这一路径往往是隐式的。本文针对这一挑战，提出了一种证据驱动的工作流，用于系统地将仿真和运行阶段产生的知识转移回设计时模型。我们通过一个影响模型来演示该工作流，影响模型是一种设计时工件，用于表示设计工件和环境因素如何共同影响系统响应属性，并以一个跟随标识的机器人为用例。我们展示了仿真证据如何通过支持参与者剪枝决策、交互特征刻画和影响分析来丰富影响模型。

    arXiv:2610.13456v1 Announce Type: new  Abstract: Cyber-Physical Systems are developed through multiple model based activities, comprising several stages of development. As the development progresses through simulation, experimentation, and operation, observations of system execution may reveal missing, uncertain, or invalid design knowledge. The path from execution results to persistent design-time enrichment, however, is often implicit. This paper addresses this challenge by proposing an evidence driven workflow for systematically transferring knowledge produced during simulation and operation back into design-time models. We demonstrate the workflow using an Influence Model, a design-time artefact used to represent how design artefacts and environmental factors jointly affect system response properties, using a sign-following robot use case. We show how simulation evidence can enrich Influence Models by supporting participant pruning decisions, interaction characterisation, influence
    
[^36]: O'ido：在5美元微控制器上实现无需神经加速器的开放词汇语音识别

    O\'ido: Open-Vocabulary Speech Recognition on a $5 Microcontroller without a Neural Accelerator

    [https://arxiv.org/abs/2610.13447](https://arxiv.org/abs/2610.13447)

    O'ido在仅5美元、无神经加速器的ESP32-S3微控制器上实现了开放词汇英语语音识别，通过int8量化和内存带宽优化的调度，错误率比厂商识别器低2.3-2.8倍。

    

    在低成本微控制器上进行语音识别，实际上仅限于关键词检测或固定的命令列表。我们提出了O'ido，这是一个推理引擎，可以在ESP32-S3（240 MHz双核Xtensa LX7、512 KB SRAM、8 MB PSRAM、16 MB闪存、无神经加速器）上完全运行开放词汇英语语音识别。O'ido在芯片的SIMD指令上执行一个1300万参数的Conformer-CTC模型，采用int8权重、逐帧动态int8激活和int8注意力，其调度方案围绕闪存和PSRAM带宽设计。在LibriSpeech test-clean/test-other数据集上，设备端计算达到3.70%/8.23%的词错误率（WER），与全精度相差不到0.12个百分点；配合片上130万参数的语言模型可达3.33%/7.15%；采用转换器（transducer）头则达到2.99%/6.67%，而int4变体可容纳在8.3 MB内。与同一芯片上厂商提供的识别器相比，其错误率降低了2.3-2.8倍，且错误少于Whisper tiny.en和Mo（原文在此处截断）

    arXiv:2610.13447v1 Announce Type: cross  Abstract: Speech recognition on low-cost microcontrollers is, in practice, limited to keyword spotting or a fixed list of commands. We present O\'ido, an inference engine that runs open-vocabulary English speech recognition entirely on an ESP32-S3 (240 MHz dual-core Xtensa LX7, 512 KB of SRAM, 8 MB of PSRAM, 16 MB of flash, no neural accelerator). O\'ido executes a 13M-parameter Conformer-CTC model with int8 weights, per-frame dynamic int8 activations and int8 attention on the chip's SIMD instructions, with a schedule designed around flash and PSRAM bandwidth. On LibriSpeech test-clean/test-other the on-device arithmetic reaches 3.70%/8.23% word error rate (WER), within 0.12 points of full precision, and 3.33%/7.15% with an on-chip 1.3M-parameter language model; a transducer head gives 2.99%/6.67% and an int4 variant fits in 8.3 MB. This is 2.3-2.8x fewer errors than the vendor's recognizer on the same chip, and fewer than Whisper tiny.en and Mo
    
[^37]: 基于DistilHuBERT重访口音分类：说话人分离评估与跨语料库泛化

    Revisiting Accent Classification with DistilHuBERT: Speaker-Disjoint Evaluation and Cross-Corpus Generalization

    [https://arxiv.org/abs/2610.13440](https://arxiv.org/abs/2610.13440)

    该研究发现文件级评估会大幅高估基于DistilHuBERT的口音分类模型性能，说话人分离评估才能更真实地反映模型对未见说话人的泛化能力，且仅用外部数据训练的模型难以可靠泛化到L2-ARCTIC语料库。

    

    最近的口音分类系统在标准的语料库内评估中可以达到很高的准确率。然而，当训练集和测试集之间的说话人存在重叠时，与真实部署场景相比，这样的结果可能高估了模型的鲁棒性。本研究使用L2-ARCTIC和多个外部英语口音数据集，重新审视了基于DistilHuBERT的口音分类的鲁棒性。比较涵盖了文件级和说话人分离的数据划分，以及平衡的外部数据训练和师生伪标签方法。语料库内评估和外部评估之间的巨大性能差距表明，文件级评估大大高估了性能，而说话人分离评估能为模型对未见说话人的泛化提供更贴近现实的估计。在本研究中，仅用外部数据训练的模型无法可靠地泛化到L2-ARCTIC。在所测试的配置中，平衡的外部……

    arXiv:2610.13440v1 Announce Type: cross  Abstract: Recent accent classification systems can achieve high accuracy in standard in-corpus evaluations. However, when speakers overlap between the training and test sets, such results may overestimate model robustness compared with real-world deployment scenarios. This study revisits the robustness of DistilHuBERT-based accent classification by using L2-ARCTIC and multiple external English accent datasets. The comparison covers file-level and speaker-disjoint splits, along with balanced external-data training and teacher-student pseudo-labeling. The large performance gap between in-corpus and external evaluations suggests that file-level evaluation substantially overestimates performance, while speaker-disjoint evaluation can provide a more realistic estimate of the generalization to unseen speakers. In this study, models trained on external data alone do not reliably generalize to L2-ARCTIC. Among the tested configurations, balanced externa
    
[^38]: 金融语音需要持续学习正则化吗？EWC与mini-KFAC在财报电话会议语音识别上的研究

    Does Financial Speech Need Continual Learning Regularization? A Study of EWC and mini-KFAC on Earnings Call ASR

    [https://arxiv.org/abs/2610.13432](https://arxiv.org/abs/2610.13432)

    该研究发现在将Whisper适配到财报电话会议语音识别时，新任务与旧任务之间存在互补性而非灾难性遗忘，因而正则化反而有害；但当确需正则化时，逐层mini-KFAC比对角EWC能取得更低的词错误率和更稳定的优化效果。

    

    财报电话会议转录呈现了一个现实的持续学习场景：自动语音识别（ASR）系统在用新录音进行更新时，存在灾难性遗忘之前学到的金融语音模式的风险。将Whisper-small从通用领域语音语料库适配到Earnings22基准的实验却揭示了相反的结果：所有被评估的方法在适配过程中对同领域保留集的金融语音表现出提升而非遗忘。这种任务互补性使得正则化反而适得其反。然而，当正则化确有必要时，逐层的mini-Kronecker-Factored近似曲率（mini-KFAC）方法似乎比标准的对角弹性权重固化（EWC）实现了更低的词错误率（WER）和更稳定的优化。研究结果表明，在应用持续学习约束之前应先评估任务互补性，并优先选择mini-KFAC。

    arXiv:2610.13432v1 Announce Type: cross  Abstract: Earnings call transcription presents a realistic continual learning scenario where Automatic Speech Recognition (ASR) systems are updated with new recordings but risk catastrophic forgetting of previously learned financial speech patterns. Adapting Whisper-small from a general-domain speech corpus to the Earnings22 benchmark reveals the opposite: all evaluated methods exhibit improvement on held-out financial speech from the same domain during adaptation rather than forgetting. This task complementarity makes regularization counterproductive. Nevertheless, when regularization is warranted, a layer-wise mini-Kronecker-Factored Approximate Curvature (mini-KFAC) approximation seems to achieve lower word error rates (WERs) and more stable optimization than standard diagonal Elastic Weight Consolidation (EWC). The results suggest that task complementarity should be assessed before applying continual learning constraints, and prefer mini-KFA
    
[^39]: 迈向世界之巅：藏语自然语言处理进展

    Towards the Roof of the World: NLP Progress in Tibetan Languages

    [https://arxiv.org/abs/2610.13398](https://arxiv.org/abs/2610.13398)

    本文开展了首个经同行评审的藏语NLP系统性文献计量调查，通过分析2004至2026年间140篇出版物，揭示了中国研究人员主导的全流水线研究范式如何在复杂文字与方言多样性等约束下推动藏语NLP的快速发展。

    

    藏语是青藏高原及其周边地区约七百万人使用的语言，但在自然语言处理（NLP）领域中仍代表性不足。据我们所知，本文提出了首个经同行评审的藏语NLP系统性文献计量调查：涵盖2004年至2026年间ACL Anthology和DBLP中的140篇出版物，并对发表载体类型、研究领域、第一作者所属国家以及资源发布情况进行了标注。研究成果增长迅速，其中84%的出版物发表于2018年以后，并沿着NLP流水线自下而上不断扩展，从语料库和分词，经文字、语音和应用，发展到大语言模型（LLMs）。中国的研究人员担任了87%出版物的第一作者，这些出版物涵盖所有25个代表性研究领域，而其他国家的工作则集中于古典藏文。我们认为，这一全流水线研究范式是在复杂文字、方言多样性等多重约束条件下实现的……

    arXiv:2610.13398v1 Announce Type: new  Abstract: Tibetan, spoken by some seven million people on and around the Qinghai-Xizang Plateau, remains underrepresented in natural language processing (NLP). We present, to our knowledge, the first peer-reviewed systematic bibliometric survey of Tibetan NLP: 140 publications from 2004 to 2026 in the ACL Anthology and DBLP, annotated for venue type, research area, first-affiliation country, and resource release. Output has grown rapidly, with 84% of the publications appearing since 2018, and has expanded bottom-up along the NLP pipeline, from corpora and word segmentation through script, speech, and applications to large language models (LLMs). Researchers based in China first-authored 87% of the publications, which span all 25 represented research areas, whereas work from other countries concentrates on Classical Tibetan. We argue that this full-pipeline paradigm, realized under the compounded constraints of a complex script, dialectal diversity
    
[^40]: ProgramGym：面向长周期程序复现智能体的可扩展训练环境

    ProgramGym: Scaling Training Environments for Long-Horizon Program Reproduction Agents

    [https://arxiv.org/abs/2610.13395](https://arxiv.org/abs/2610.13395)

    ProgramGym提出了一个由Program-Env和Program-Agent组成的框架，通过反馈驱动的循环从代码仓库自动构建可验证的训练环境，实现了对长周期程序复现智能体的可扩展训练。

    

    arXiv:2610.13395v1 公告类型：cross 摘要：编程智能体正越来越多地被用于像软件工程师那样从零开始构建程序，而不是解决代码仓库中的问题或根据给定的需求文档生成仓库。然而，能够促使智能体自主探索所需功能并决定系统设计方式的、可扩展且可验证的训练环境仍然稀缺。此外，现有的智能体工作流也无法支撑长周期的开发过程，即智能体在实现之前先对参考程序进行探索。为了应对这些挑战，我们提出了ProgramGym，这是一个由Program-Env和Program-Agent组成的框架，旨在为长周期程序复现智能体提供可扩展的训练环境。具体而言，Program-Env通过一个反馈驱动的循环从代码仓库构建可验证的训练环境，该循环能够修复运行时沙箱构建问题并提升黑盒测试的质量。利用这一新颖的流水线，我们获得了1,648个训练……（摘要不完整，原文在此处截断）

    arXiv:2610.13395v1 Announce Type: cross  Abstract: Coding agents are increasingly used to build programs from scratch as software engineers do rather than to resolve issues within a repository or to generate a repository from a given requirements document. However, scalable verifiable training environments that force agents to discover what is required and to decide how the system should be designed remain scarce. Moreover, existing agent workflows also cannot sustain long-horizon development in which the agent explores the reference program before implementation. To address these challenges, we introduce ProgramGym, a framework consisting of Program-Env and Program-Agent for scalable training environments of long-horizon program reproduction agents. Specifically, Program-Env constructs verifiable training environments from repositories through a feedback-driven loop that repairs runtime sandbox builds and the quality of black-box tests. Using this novel pipeline, we obtain 1,648 train
    
[^41]: MoPrO-SD：面向立场检测的模块化提示优化

    MoPrO-SD: Modular Prompt Optimization for Stance Detection

    [https://arxiv.org/abs/2610.13386](https://arxiv.org/abs/2610.13386)

    提出面向立场检测的模块化提示优化框架MoPrO-SD，将提示分解为功能性语义模块并利用LLM反馈进行顺序优化，同时发布人工标注的疫苗立场检测基准VaccineStance。

    

    立场检测对大型语言模型（LLM）而言仍然具有挑战性，原因在于隐含观点、讽刺表达以及依赖语境的立场表述。尽管自动提示优化提供了一种无需更新模型参数的微调替代方案，但现有的结构化方法并非专门围绕立场标签语义和类别特定的歧义而设计。我们提出了MoPrO-SD，这是一个针对立场检测的提示优化框架，它将提示分解为功能性语义模块，并利用基于LLM的反馈对其进行顺序优化。我们研究了三种优化策略：仅指令优化（I）、带推理的指令优化（I+R）以及带逐样本推理的指令优化（I+R+E），并使用类别条件错误来进行立场标签的优化。我们还引入了VaccineStance，这是一个新的人工标注的疫苗立场检测基准。在四个数据集和五个开源权重LLM上进行的实验表明……

    arXiv:2610.13386v1 Announce Type: new  Abstract: Stance detection remains challenging for large language models (LLMs) due to implicit opinions, sarcasm, and context-dependent expressions of stance. Although automatic prompt optimization offers an alternative to fine-tuning without updating model parameters, existing structured approaches are not specifically designed around stance-label semantics and class-specific ambiguities. We introduce MoPrO-SD, a stance-specific prompt optimization framework that decomposes prompts into functional semantic modules and refines them sequentially using LLM-based feedback. We study three refinement strategies: instruction-only optimization (I), instruction with reasoning (I+R), and instruction with per-example reasoning (I+R+E), using class-conditioned errors for stance-label refinement. We also introduce VaccineStance, a new human-annotated benchmark for vaccine stance detection. Experiments across four datasets and five open-weight LLMs show that 
    
[^42]: 从结构到有效性：机制引导的大语言模型评分量规生成与优化

    From Structure to Validity: Mechanism-Guided Generation and Refinement of LLM Rubrics

    [https://arxiv.org/abs/2610.13375](https://arxiv.org/abs/2610.13375)

    该研究系统揭示了LLM评分量规设计的有效机制——结构化的标准组织最为关键，并据此提出ARTS-Hybrid方法，通过自适应量规树构建与语义优先、响应校准的精炼策略，同时提升了端到端评估性能与量规的内在质量。

    

    评分量规已被广泛用于评估和优化语言模型在开放式任务上的表现。然而，现有工作主要聚焦于如何生成更好的量规，对于什么样的量规设计才是有效的缺乏深入理解。在这项工作中，我们围绕标准组织、标准操作化和分数聚合三个维度，对量规设计进行了系统的机制分析。分析结果表明：结构化的组织方式有助于构建更全面的标准；仅靠语义描述并不足以保证有效的标准级判断；而复杂的分数聚合只能带来有限的额外收益。基于这些发现，我们提出了带有混合精炼的自适应量规树搜索方法（ARTS-Hybrid），它将自适应树构建与“语义优先、响应校准”的精炼策略相结合。在多个基准上的实验表明，ARTS-Hybrid 同时提升了端到端评估性能和量规的内在质量。

    arXiv:2610.13375v1 Announce Type: new  Abstract: Rubrics have been widely used to evaluate and optimize language models on open-ended tasks. However, existing work primarily focuses on generating better rubrics, with limited understanding of what makes rubric design effective. In this work, we conduct a systematic mechanistic analysis of rubric design across criterion organization, criterion operationalization, and score aggregation. Our analysis shows that structural organization enables more comprehensive criterion construction, semantic specification alone does not ensure effective criterion-level judgments, and sophisticated score aggregation provides limited additional gains. Guided by these findings, we introduce Adaptive Rubric Tree Search with Hybrid refinement (ARTS-Hybrid), which combines adaptive tree construction with semantic-first, response-calibrated refinement. Experiments across benchmarks show that ARTS-Hybrid improves both end-to-end evaluation performance and intrin
    
[^43]: 利用专家选择偏差克服数据不平衡

    Overcoming Data Imbalances with Expert Selection Biases

    [https://arxiv.org/abs/2610.13370](https://arxiv.org/abs/2610.13370)

    该论文提出在MoE模型中引入基于类别的专家选择偏差，将不同类别的词元优先路由到专门的专家子集，从而在数据类别不平衡的情况下实现软性类别对齐的专家专门化，有效缓解少数类别容量受限的问题。

    

    混合专家模型已成为前沿大语言模型的标准架构。每个Transformer块中的多个并行专家在理论上能够在预训练期间实现专家按领域、语言或模态等类别进行专门化。然而，当数据在各类别之间分布不平衡时，少数类别可能会集中在专家池中的一小部分，或者在专家之间低效地分散，从而实际上限制了它们可用的模型容量。在这项工作中，我们证明了基于类别的专家选择偏差可以将来自不同类别的词元优先路由到专门的专家池。通过根据输入类别施加偏差，我们提供了一种诱导软性类别对齐专门化的简单方法。这种干预促使各类别趋向于不同的专家子集，同时在有用时保留跨类别的共享。更重要的是，它使得过度分配

    arXiv:2610.13370v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) models have become the standard architecture for frontier LLMs. The multiple parallel experts in each transformer block theoretically enable expert specialization according to categories such as domains, languages, or modalities during pretraining. However, when data is imbalanced across categories, minority categories can become concentrated in a small fraction of the expert pool or spread inefficiently across them, effectively limiting their available capacity. In this work, we demonstrate that category-conditioned expert-selection biases can preferentially route tokens from different categories toward specialized expert pools. By applying biases based on input category, we provide a simple way to induce soft category-aligned specialization. This intervention encourages categories toward distinct expert subsets while retaining cross-category sharing when useful. More importantly, it enables the over-allocation 
    
[^44]: 测试时记忆构建

    Test-Time Memory Construction

    [https://arxiv.org/abs/2610.13348](https://arxiv.org/abs/2610.13348)

    提出了 T²Mem 测试时记忆构建框架，将记忆构建推迟到推理阶段并沿推理过程逐步进行，以低开销高效回答涉及长交互历史的问题。

    

    大型语言模型（LLMs）在回答涉及长交互历史的问题时表现不佳，因为相关证据可能分散在相距遥远的会话之中。现有的记忆系统通常在观察到用户查询之前就离线构建并维护复杂的记忆表示，这不仅带来巨大的开销，还限制了记忆构建根据查询特定信息需求进行自适应调整的能力。在本文中，我们提出了 T²Mem，一个测试时记忆构建框架，它将记忆构建推迟到推理阶段，并沿着推理过程逐步构建记忆。具体而言，T²Mem 离线提取轻量级的事件级元数据，以捕获关键事件和时间关联作为检索线索，同时保留与原始交互的链接，以便重建跨事件的依赖关系。在测试时，LLM 自适应地检索并组织元数据与原始交互……

    arXiv:2610.13348v1 Announce Type: new  Abstract: Large Language Models (LLMs) struggle to answer questions over long interaction histories, where relevant evidence may be scattered across distant sessions. Existing memory systems typically construct and maintain sophisticated memory representations offline, before user queries are observed, incurring substantial overhead and limiting their ability to adapt memory construction to query-specific information needs. In this paper, we propose T$^2$Mem, a Test-Time Memory construction framework that defers memory construction to inference and progressively builds memory along the reasoning process. Specifically, T$^2$Mem extracts lightweight episode-level metadata offline to capture key events and temporal associations as retrieval cues, while retaining links to the original interactions to facilitate the reconstruction of dependencies across episodes. At test time, the LLM adaptively retrieves and organizes metadata and raw interactions, pr
    
[^45]: 探索扩散模型与自回归模型中文本到图像Transformer的注意力汇聚现象

    Exploring Attention Sinks in Text-to-Image Transformers from Diffusion and Autoregressive Models

    [https://arxiv.org/abs/2610.13337](https://arxiv.org/abs/2610.13337)

    该论文首次系统性地研究了文本到图像Transformer中（涵盖双向扩散模型与因果自回归模型）注意力汇聚现象的存在性、出现位置及其对生成过程的功能性影响，并发现两类模型的注意力汇聚呈现截然不同的模式。

    

    注意力汇聚是指持续获得与其语义内容不成比例的注意力权重的token或位置。尽管注意力汇聚在大语言模型中已被广泛研究，但其在文本到图像（T2I）Transformer中的行为和功能相关性仍未被充分探索。为填补这一空白并更好地理解文本到图像Transformer如何分配和路由注意力，我们在双向扩散模型和因果自回归模型中探索了这一现象。具体而言，我们探讨了注意力汇聚是否存在、它们出现在何处，以及它们如何从功能上影响生成过程。通过系统性的注意力分析和针对性的干预实验，我们发现注意力汇聚在双向扩散模型和自回归模型中均会出现，但表现出不同的模式。自回归模型将注意力集中于首个token和全局尺度的token，而……（摘要在此处截断）

    arXiv:2610.13337v1 Announce Type: cross  Abstract: Attention sinks are tokens or positions that consistently receive disproportionate attention mass, often beyond what their semantic content would suggest. Despite extensive study in large language models, their behavior and functional relevance in text-to-image (T2I) Transformers remain largely unexplored. To fill this gap and better understand how text-to-image Transformers allocate and route attention, we explore this phenomenon of text-to-image transformers across bidirectional diffusion models and causal autoregressive models. Specifically, we explore whether attention sinks exist, where they emerge, and how they functionally influence generation. Through systematic attention analysis and targeted interventions, we identify that attention sinks emerge in both bidirectional diffusion and autoregressive models but exhibit distinct patterns. The autoregressive model concentrates attention on the first and global-scale tokens, whereas 
    
[^46]: 面向智能体化 LLM 服务的分词任务主机卸载方案

    Taking Tokenization off the Host for Agentic LLM Serving

    [https://arxiv.org/abs/2610.13325](https://arxiv.org/abs/2610.13325)

    提出 GraphTok，一个将完整分词流水线封装为可重放 CUDA 图并逐位复现参考结果的 GPU 分词器，比 CPU 快 4.7-6.0 倍，可将分词从主机彻底卸载，从而显著降低智能体场景下 LLM 服务的首 token 延迟并提升 GPU 集群可扩展性。

    

    智能体工作负载会在多次调用间重复发送几乎不变的长上下文。前缀缓存如今已能跳过大部分预填充阶段，但在查询缓存之前，每个请求仍必须完整地进行分词。分词这一长期以来被视为可忽略不计的 CPU 阶段，如今已成为首 token 时间（TTFT）中占比最大的部分，而为一个 GPU 集群进行分词的前端也限制了其所能支撑的 GPU 数量。我们提出了 GraphTok，一个 GPU 分词器，它将整个编码流水线——从预分词、字节对编码（BPE）合并到 token ID 组装——作为一个可重放的 CUDA 图来运行。GraphTok 在 GPT-2、Llama 3、Qwen 2.5、DeepSeek-V3 和 Gemma 3 上逐位复现了参考 token ID，在 GPU 无法占优的短输入场景下将其交给 CPU 引擎处理，并可无缝接入未经修改的 vLLM 和 NVIDIA Dynamo。GraphTok 在包含数据传输的情况下，仅用 1.7-3.2 毫秒即可编码一百万个 GPT-2 token，在 Llama 3 上比 giga（原文截断）快 4.7-6.0 倍。

    arXiv:2610.13325v1 Announce Type: cross  Abstract: Agentic workloads resend long contexts that barely change between calls. Prefix caching now skips most of their prefill, yet every request must still be tokenized in full before the cache can even be consulted. Tokenization, a CPU stage long treated as negligible, becomes the largest part of time-to-first-token (TTFT), and a frontend that tokenizes for a whole GPU fleet limits how many GPUs it can feed. We present GraphTok, a GPU tokenizer that runs the entire encoding pipeline, from pre-tokenization through byte-pair encoding (BPE) merging to token-ID assembly, as one replayable CUDA graph. GraphTok reproduces the reference token IDs bit-for-bit on GPT-2, Llama 3, Qwen 2.5, DeepSeek-V3, and Gemma 3, hands short inputs to a CPU engine where the GPU cannot win, and plugs into unmodified vLLM and NVIDIA Dynamo. GraphTok encodes one million GPT-2 tokens in 1.7-3.2 ms including data transfers, and on Llama 3 it is 4.7-6.0x faster than giga
    
[^47]: 面向大语言模型预训练数据的感知主机网络爬取方法

    Host-Aware Web Crawling for LLM Pretraining Data

    [https://arxiv.org/abs/2610.13321](https://arxiv.org/abs/2610.13321)

    该研究发现页面质量约一半的差异源于主机层面，且少量采样页面即可有效估计主机质量，因此基于主机感知的爬虫预算分配策略比图中心性方法更适合收集高质量LLM预训练数据。

    

    网络爬取是大语言模型（LLM）预训练数据的主要来源，然而数据筛选流程会丢弃大量已爬取的页面。我们研究了爬虫应如何在主机之间以及主机内部分配其抓取预算，以收集更多高质量的预训练数据。通过结合文档级质量分类器得分、网络图中心性指标以及Common Crawl的按主机爬取覆盖率，我们发现分类器得分中大约一半的变异发生在主机之间。在我们分析的爬取数据中，来自某个主机的少量页面也能对其剩余页面的质量提供有用的估计。Common Crawl使用调和中心性（一种图中心性指标）来指导其爬取分配。在我们的模拟中，使用中心性来选择要爬取哪些主机相比随机排序收益有限，且在某些分类器上表现更差。对于决定每个主机的爬取深度，采样少量页面然后从主机获取更多……（原文在此处截断）

    arXiv:2610.13321v1 Announce Type: new  Abstract: Web crawling is a major source of large language model (LLM) pretraining data, yet data-selection pipelines discard a large share of crawled pages. We investigate how a crawler should allocate its budget across and within hosts to collect more high-quality pretraining data. Combining document-level quality classifier scores, web-graph centrality metrics, and Common Crawl's per-host crawl coverage, we find that roughly half of the variation in classifier scores occurs between hosts. In the crawls we analyze, a few pages from a host also provide a useful estimate of the quality of its remaining pages. Common Crawl uses harmonic centrality, a graph centrality metric, to guide its crawl allocation. In our simulations, using centrality to choose which hosts to crawl yields limited gains over random ordering and performs worse for some classifiers. For deciding how deeply to crawl each host, sampling a few pages and then fetching more from hos
    
[^48]: 无需对齐的一致性：无标签的评分标准澄清提升了LLM裁判间的共识，却未提升与人类评判的一致性

    Agreement Without Alignment: Label-Free Rubric Clarification Increases LLM-Judge Consensus but Not Agreement with Humans

    [https://arxiv.org/abs/2610.13303](https://arxiv.org/abs/2610.13303)

    该论文提出一种无标签的评分标准澄清程序，通过将裁判间分歧转化为对争议子标准的重写修复，显著提升了LLM评审团内部的一致性，但发现这种共识的提升并不能转化为与人类评判更高的一致性。

    

    LLM裁判越来越多地以评审团的形式部署，而评审团成员之间的分歧通常被解读为评分标准不清晰的征兆。我们测试了一种无标签的程序，将这种分歧转化为评分标准的修复。该程序将每个评分标准分解为子标准，根据在开发集上裁判间的分歧程度对子标准进行排序，并让一个编辑模型仅重写争议最大的子标准。在我们的评估中，将可文本检查的子标准迁移至确定性检查器的附加阶段没有迁移任何子标准。人类标签从不用于选择或编辑条款；它们仅用于评估结果。我们在一个版本控制的协议中预先指定了四个假设，并使用来自三个提供商的五名裁判组成的评审团，在SummEval、HealthBench元评估和USR Topical-Chat上评估了六种评分标准条件，总计进行了129,770次裁判请求（不含重试）。在SummEval上，该程序提高了裁判间的Krippendorff alpha系数……（摘要原文在此处截断）

    arXiv:2610.13303v1 Announce Type: cross  Abstract: LLM judges are increasingly deployed as panels, and disagreement among panel members is often read as a symptom of an unclear rubric. We test a label-free procedure that turns this disagreement into rubric repairs. The procedure decomposes each rubric into sub-criteria, ranks them by cross-judge disagreement on a development split, and has an editor model rewrite only the most contested ones. An additional stage that moves text-checkable sub-criteria to deterministic checkers transferred none in our evaluation. Human labels are never used to select or edit clauses; they are used only to evaluate the outcome. We pre-specified four hypotheses in a version-controlled protocol and evaluated six rubric conditions with a panel of five judges from three providers on SummEval, the HealthBench meta-evaluation, and USR Topical-Chat, for a total of 129,770 judge requests, excluding retries. On SummEval, the procedure raised inter-judge Krippendor
    
[^49]: 提示词优化器是否会破坏其未被评分的要求？一项预注册研究

    Do Prompt Optimizers Break Requirements They Are Not Scored On? A Preregistered Study

    [https://arxiv.org/abs/2610.13287](https://arxiv.org/abs/2610.13287)

    预注册研究发现，提示词优化器在提升目标指标的同时确实会破坏未被评分的要求（如引用和格式），但事后分析表明，表面的性能退化主要源于推理过程挤占输出空间导致答案被截断而无法提取，而非真正的内容质量下降。

    

    提示词优化器会重写提示词以提升某个目标指标，但提示词中还包含该指标并不评分的要求，例如输出格式、引用来源或拒答行为。在一项预注册研究中，我们在HotpotQA、SQuAD 2.0和IFEval上运行GEPA、MIPROv2和TextGrad，并通过确定性检查逐项统计对未评分要求的违反情况。按照预注册的评判标准，三种优化器均产生了性能退化，在HotpotQA上尤为显著：引用有效性的净退化达到10.7至41.6个百分点，而精确匹配分数则上升了21至34个百分点。然而，事后分解分析改变了这一解读：优化后的提示词会在要求的JSON对象之前引出推理过程，而在全程使用的200个token输出上限下，13.9%至47.0%的回复在任何答案出现之前就已被截断。在可解析的输出中，净退化在零附近3个百分点以内。当token上限提高到1024时，提取失败现象消失，内容退化保持在……

    arXiv:2610.13287v1 Announce Type: new  Abstract: Prompt optimizers rewrite a prompt to raise one target metric, but prompts also state requirements the metric does not score, such as an output format, citations, or abstention. In a preregistered study, we run GEPA, MIPROv2, and TextGrad on HotpotQA, SQuAD 2.0, and IFEval and count per-item violations of unscored requirements with deterministic checks. By the preregistered criterion, all three optimizers produce regressions, substantially on HotpotQA, where net regression in citation validity reaches 10.7 to 41.6 points while exact match rises by 21 to 34 points. A post-hoc decomposition changes this reading. The optimized prompts elicit reasoning before the required JSON object, and under the 200-token output cap used throughout, 13.9% to 47.0% of responses end before any answer appears. Among parseable outputs, net regression is within 3 points of zero. With a 1,024-token cap, extraction failures disappear, content regression stays be
    
[^50]: BoundaryEvo：面向长链推理蒸馏的能力边界处数据-模型协同演化

    BoundaryEvo: Data-Model Co-Evolution at the Capability Boundary for Long-Chain Reasoning Distillation

    [https://arxiv.org/abs/2610.13285](https://arxiv.org/abs/2610.13285)

    BoundaryEvo提出一种两阶段框架，先帮助学生模型建立基础推理能力，再使训练数据随学生模型的能力边界协同演化，从而在仅有少量教师轨迹的条件下高效提升小模型的长链推理能力。

    

    从大型教师模型中蒸馏长链推理轨迹是提升小型语言模型推理能力的有效方法。然而，现有方法通常在蒸馏前选取固定的训练集，忽略了推理样本的有用性会随着学生模型的演进而变化这一事实。当只能使用少量教师轨迹进行训练时，这种不匹配的代价尤为显著。此外，学生模型已经掌握的样本几乎不提供新的监督信号，而远超学生能力范围的样本则可能难以从中学习。因此，训练数据应当随模型能力的演化而自适应调整，但当学生模型初始能力较弱时，可靠的数据选择十分困难。为应对这些挑战，我们的核心思想是首先建立基础推理能力，然后使训练数据适配学生模型不断演化的能力边界。我们提出BoundaryEvo，一个两阶段……（摘要至此截断）

    arXiv:2610.13285v1 Announce Type: new  Abstract: Distilling long-chain reasoning trajectories from large teacher models provides an effective way to improve the reasoning ability of small language models. However, existing methods typically select a fixed training set before distillation, overlooking that the usefulness of a reasoning example changes as the student evolves. This mismatch becomes especially costly when only a small number of teacher trajectories can be used for training. Besides, examples already mastered provide little new supervision, while examples far beyond the capability of student may be difficult to learn from. Training data should therefore adapt as model capability evolves, yet reliable data selection is difficult when the student is initially weak. To address these challenges, our key idea is to first establish the basic reasoning capability and then adapt the training data to the evolving capability boundary of the student. We propose BoundaryEvo, a two-stag
    
[^51]: 超越最终嵌入：从词元级证据中检索记忆

    Beyond the Final Embedding: Retrieving Memories from Token-Level Evidence

    [https://arxiv.org/abs/2610.13277](https://arxiv.org/abs/2610.13277)

    该论文提出一种无需额外训练的稀疏词元匹配方法，利用嵌入模型已有的词元级上下文表示捕捉局部证据，并结合整条记忆相似度对记忆排序，在八个基准测试和多个模型家族上显著提升检索效果，还可通过可选的低秩查询适配器进一步增强。

    

    记忆增强型助手从存储的对话和文档中检索证据以回答查询。大多数基于嵌入的检索器比较查询和记忆的单向量表示，即使相关性往往只取决于内容的一小部分。这种局部证据可以利用同一嵌入模型已经计算出的上下文词元表示与查询进行匹配。我们的方法使用稀疏词元匹配来改进检索，且无需额外训练。每个查询词元在候选记忆中检索一小撮匹配的词元，匹配得分针对每条记忆进行聚合，并与整条记忆的相似度相结合来对记忆进行排序。可选的低秩查询适配器可以在监督下进行训练，同时保持编码器和已存储的记忆表示固定。我们的方法无需额外训练即可在八个基准测试和多个模型家族上提升检索性能。查询适配器可进一步……（原文在此处截断）

    arXiv:2610.13277v1 Announce Type: cross  Abstract: Memory-augmented assistants retrieve evidence from stored conversations and documents to answer queries. Most embedding-based retrievers compare single-vector representations of queries and memories, even when relevance depends on a small part of the content. Such local evidence can be matched to queries using the contextual token representations already computed by the same embedding models. Our method uses sparse token matching to improve retrieval without additional training. Each query token retrieves a small set of matching tokens across candidate memories. The matching scores are aggregated for each memory and combined with whole-memory similarity to rank memories. An optional low-rank query adapter can be trained with supervision while keeping the encoder and stored memory representations fixed. Our method improves retrieval across eight benchmarks and multiple model families without additional training. Query adaptation further
    
[^52]: CAST：面向跨分词器在线策略蒸馏的信用对齐

    CAST: Credit Alignment for Cross-Tokenizer On-Policy Distillation

    [https://arxiv.org/abs/2610.13257](https://arxiv.org/abs/2610.13257)

    CAST提出了一种跨分词器在线蒸馏中的“移动或保持”信用对齐框架，利用预测不确定性和教师信号平滑度将块级优势精确分配到学生令牌上，并对无教师对应的令牌施加锚定约束。

    

    跨分词器在线策略蒸馏（OPD）需要在对教师与学生模型的分数进行比较之前，先对齐二者的序列。基于解码文本的对齐方式可以在块级别提供这种对应关系，但遗留了一个尚未解决的关键问题：信用对齐。块级信号只指定了某个片段应该改变多少，却没有说明该信号应如何在学生的各个令牌之间分配，同时缺乏教师对应关系的令牌则不受任何约束。我们提出了CAST（基于块感知的信用分配与选择性令牌锚定），一个“移动或保持”（shift-or-stay）框架，同时解决上述两个问题。对于已对齐的块，CAST利用预测不确定性和局部教师信号平滑度，将每个块的优势重新分配到学生令牌上，在保留块总优势的同时，强调由局部一致教师反馈所支持的学习机会。所得权重精确求解了一个经KL正则化的信用分配问题。

    arXiv:2610.13257v1 Announce Type: cross  Abstract: Cross-tokenizer on-policy distillation (OPD) requires aligning teacher and student sequences before their scores can be compared. Decoded-text alignment provides such correspondence at the chunk level, but leaves a distinct problem unresolved: credit alignment. A chunk-level signal specifies how much a span should change, but not how that signal should be distributed across student tokens, while tokens without teacher correspondence remain unconstrained. We propose CAST (Chunk-aware credit Assignment with Selective Token Anchoring), a shift-or-stay framework that addresses both issues. For aligned chunks, CAST redistributes each chunk's advantage across student tokens using predictive uncertainty and local teacher-signal smoothness, emphasizing learning opportunities supported by locally consistent teacher feedback while preserving the chunk's total advantage. The resulting weights exactly solve a KL-regularized credit-allocation probl
    
[^53]: 视觉Transformer仅凭图像就能学习词语含义

    Vision Transformers Learn Word Meaning from Images Alone

    [https://arxiv.org/abs/2610.13252](https://arxiv.org/abs/2610.13252)

    仅通过图像自监督预训练的DINOv3视觉Transformer无需任何语言监督，就能从图像中渲染的文字学会按词义组织词语，证明视觉模型可以从纯像素中习得词语语义。

    

    视觉模型能否仅凭像素学会词语的含义？我们展示了两个规模最大的DINOv3视觉Transformer能够按照词义来组织渲染在图像中的文字，尽管它们仅在图像上通过自监督方式进行预训练，没有配对的文本描述、类别标签或其他任何语言监督。我们通过词图像与匹配对照图像之间的嵌入差异来推导词语表示，然后在控制拼写相似性的前提下，检验语义相关的词语是否在每个模型的嵌入空间中互为近邻。我们发现同一行业的品牌名称会聚集在一起，来自同一语义领域的名词也是如此。国家-首都词对之间表现出一致的向量偏移，这与在文本上训练的词嵌入类似。这些词语表示还与其所命名的视觉对象类别的表示相对齐。据我们所知，DINOv3-H和DINOv3-7B是首个展现出这种能力的自监督视觉模型。

    arXiv:2610.13252v1 Announce Type: cross  Abstract: Can a vision model learn what words mean from pixels alone? We show that the two largest DINOv3 vision transformers organize words rendered in images by word meaning, despite being pretrained using self-supervision on images only, without paired captions, class labels, or other language supervision. We derive word representations from embedding differences between word images and matched controls, then test whether semantically related words are neighbors within each model's embedding space, while controlling for spelling similarity. We find that brand names in the same industry group together, as do nouns from the same semantic domain. Country-capital pairs exhibit consistent vector offsets, similarly to word embeddings trained on text. These word representations also align with the representations of the visual object categories they name. DINOv3-H and DINOv3-7B are, to our knowledge, the first self-supervised vision models to show s
    
[^54]: 生成音乐评估中的音频-语言评审模型究竟在“听”什么？基于专家评分对照的四项探针实验

    What Do Audio-Language Judges of Generated Music Listen For? Four Probes with Controls from Expert Ratings

    [https://arxiv.org/abs/2610.13247](https://arxiv.org/abs/2610.13247)

    提出四项带人类评分对照的探针实验，系统检验音频-语言评审模型评估生成音乐时究竟响应音乐、提示还是提问方式，发现移除音频后所有模型表现不超过仅凭提示的预测基线，提示对齐评估甚至低于基线。

    

    开放音频-语言模型被用作生成音乐的评审：给定一段音频片段及其文本提示，模型为该片段打分，而评审模型的有效性通过其评分与专家评分的相关性来验证。然而，相关性并不能揭示评审模型究竟在响应什么：是音乐本身、文本提示，还是提问的方式。我们提出四项探针实验来分别检验这些因素，其中三项与基于人类评分计算的对照基线进行对比：移除音频、将片段替换为同一提示下另一模型的片段、在成对评审中交换两个片段的顺序，以及改写指令。我们将这些探针应用于五个开放评审模型，覆盖 MusicEval 的全部 2,748 个片段，并将后两项探针应用于 SongEval 歌曲。在没有音频的情况下，没有任何评审模型的表现超过仅凭提示就能预测的人类评分水平，而在提示对齐方面，所有评审模型都低于该水平。在有音频的情况下，每个评审模型在评估提示对齐时都会跟随其听到的片段，其中两个模型……（原文摘要至此截断）

    arXiv:2610.13247v1 Announce Type: cross  Abstract: Open audio-language models are used as judges of generated music: given a clip and its text prompt, the model rates the clip, and the judge is validated by how well its ratings correlate with expert ratings. The correlation does not show what the judge responds to: the music, the text prompt, or the way the question is asked. We propose four probes that test each of these, three of them read against a control computed from the human ratings: removing the audio, replacing the clip with another model's clip for the same prompt, swapping the order of two clips in pairwise judging, and rewording the instruction. We apply them to five open judges on all 2,748 clips of MusicEval, and the last two to the SongEval songs. Without audio, no judge exceeds what the prompt alone predicts from the human ratings, and for prompt alignment every judge falls below it. With audio, every judge follows the clip it hears when rating prompt alignment, two do
    
[^55]: AMR解析器的输出有多可靠？面向AMR解析的不确定性估计

    How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing

    [https://arxiv.org/abs/2610.13236](https://arxiv.org/abs/2610.13236)

    该论文首次在黑盒设置下研究AMR解析器的不确定性估计，通过束搜索候选间的相似度来评估解析输出的可靠性，为AMR解析器的评估提供了性能之外的重要补充维度。

    

    近期的进展显著提升了抽象意义表示（AMR）解析器的性能，然而判断其输出何时可以被信任这一问题在很大程度上仍未被探索。本工作首次在黑盒设置下，利用top-k束搜索候选结果对AMR解析器开展不确定性估计研究。不确定性通过成对候选相似度来估计，并使用预测拒绝率进行评估。本工作刻画了解析性能与不确定性之间的关系，评估了束候选分布的可靠性，分析了评估指标对不确定性评估的影响，并识别出在性能与可靠性之间实现最佳平衡的解析器。实验结果表明，不确定性估计作为评估AMR解析器的补充维度具有重要价值。

    arXiv:2610.13236v1 Announce Type: cross  Abstract: Recent advances have significantly improved the performance of Abstract Meaning Representation (AMR) parsers, yet determining when their outputs can be trusted remains largely unexplored. This work presents the first study of uncertainty estimation for AMR parsers in a black-box setting using top-k beam-search candidates. Uncertainty is estimated from pairwise candidate similarities and evaluated using the Prediction Rejection Ratio. This work characterises the relationship between parsing performance and uncertainty, evaluates the reliability of beam candidate distributions, analyses the effect of evaluation metrics on uncertainty assessment, and identifies parsers that provide the strongest balance between performance and reliability. Experimental results show the value of uncertainty estimation as a complementary dimension for evaluating AMR parsers.
    
[^56]: SharP：基于软硬提示引导的大语言模型数据增强方法用于低资源虚假新闻检测

    SharP: Soft and Hard Prompt-Guided Augmentation with LLM for Low Resource Fake News Detection

    [https://arxiv.org/abs/2610.13207](https://arxiv.org/abs/2610.13207)

    该论文提出SharP框架，通过软硬提示引导大语言模型进行边界感知的文本生成，为低资源虚假新闻检测生成具有任务特定相关性的合成训练样本，以克服现有数据增强方法生成内容过于笼统的问题。

    

    低资源条件下的虚假新闻检测面临着标注数据稀缺以及难以捕捉细微欺骗模式的挑战。在这项工作中，我们专注于低资源虚假新闻检测场景，即仅有少量标注数据可用于训练，同时在固定不变的测试协议下进行评估。这一设定的动机在于：事实核查成本高昂且存在歧义性，以及新兴主题和领域在现有基准数据集中覆盖不足。现有的数据增强方法，如同义词替换、生成对抗网络以及基于大语言模型（LLM）的生成方法，往往产生流畅但过于笼统的内容，缺乏准确检测所需的任务特定相关性。为解决这一问题，我们提出了SharP，一个由软提示和硬提示共同引导的边界感知文本生成框架，它将大语言模型作为工具，用于生成任务特定的合成样本。

    arXiv:2610.13207v1 Announce Type: new  Abstract: Fake news detection under low resource conditions is challenged by the scarcity of labeled data and the difficulty of capturing subtle deceptive patterns. In this work, we focus on low resource fake news detection, where only a small labeled set is available for training, while evaluation is performed under a fixed and unchanged test protocol. This setting is motivated by costly, ambiguous fact-checking and emerging topics and domains that are poorly covered by existing benchmarks. Existing data augmentation methods, such as synonym substitution, generative adversarial networks, and large language model (LLM)-based generation, often produce fluent but overly generic content, lacking the task-specific relevance needed for accurate detection. To address this, we propose SharP, a boundary-aware text generation framework guided by soft and hard prompts, which leverages LLMs as tools adapted to produce task-specific synthetic samples. SharP e
    
[^57]: S$^3$R：基于自稀疏推测解码的快速且忠实的强化学习轨迹生成

    S$^3$R: Fast and Faithful RL Rollouts via Self-Sparse Speculative Decoding

    [https://arxiv.org/abs/2610.13202](https://arxiv.org/abs/2610.13202)

    S³R提出自稀疏推测解码方法，用稀疏注意力起草token、全注意力验证并采用精确推测采样，在严格保持稠密策略采样分布（保证训练忠实性）的同时大幅加速长上下文RLVR的rollout生成。

    

    针对带可验证奖励的长上下文强化学习（RLVR），快速生成rollout需要在不改变用于策略优化的响应组的前提下减少KV缓存流量。在我们分析的RLVR工作负载中，注意力计算占自回归解码内核时间的72–76%。稀疏注意力可以减少这些读取，但直接从稀疏策略中采样会改变用于学习的响应组。对于组相对优化，我们证明即使采用精确的响应级重加权，通常也无法恢复期望的稠密策略更新，因为每个响应的优势取决于组内其他成员的奖励。我们提出了自稀疏推测轨迹生成（S$^3$R）：它使用稀疏注意力起草token，然后通过全注意力验证和精确推测采样生成响应。起草与验证共享同一策略参数，从而无需训练或更新单独的……

    arXiv:2610.13202v1 Announce Type: cross  Abstract: Fast rollout generation for long-context reinforcement learning with verifiable rewards (RLVR) requires reducing KV-cache traffic without changing the response groups used for policy optimization. In our profiled RLVR workloads, attention accounts for 72--76\% of autoregressive decoding kernel time. Sparse attention reduces these reads, but sampling directly from a sparse policy changes the response groups used for learning. For group-relative optimization, we show that even exact response-wise reweighting does not generally recover the expected dense-policy update because the advantage of each response depends on the rewards of other group members. We propose Self-Sparse Speculative Rollouts (S$^3$R). It uses sparse attention to propose tokens, then generates responses through full-attention verification and exact speculative sampling. Drafting and verification share the same policy parameter, eliminating the need to train or update a
    
[^58]: JANUS：利用共同演化的世界模型增强长时程自主研究

    JANUS: Enhancing Long-Horizon Autonomous Research with Co-Evolving World Models

    [https://arxiv.org/abs/2610.13194](https://arxiv.org/abs/2610.13194)

    JANUS提出了一种由共同演化的世界模型驱动的“预测-验证-演化”循环框架，通过在执行前预测候选方案的价值提升、信息增益和探索效用来缓解长时程自主研究中的反馈延迟问题，并将反馈经验蒸馏为不断演化的技能。

    

    长时程自主研究通过持续的假设生成、实现和验证循环，在加速科学进步方面展现出了巨大的潜力。然而，这种“生成-执行-反馈”范式依赖于被动的试错机制，其中代价高昂且耗时的执行过程会引入严重的反馈延迟，在爆炸式扩展的搜索空间下阻碍了演化进程。为应对这一挑战，我们提出了JANUS，这是一个通过由共同演化的世界模型驱动的“预测-验证-演化”循环，将未来结果预测与过往经验反思统一起来的框架。在一个不断演化的图中，研究智能体提出候选方案，而世界模型通过预测价值提升、信息增益和探索效用对其进行评分，然后在执行前相应地选择、精炼或剪枝分支。在接收到真实反馈后，JANUS将经验蒸馏为不断演化的技能：层次化……（摘要在此处截断）

    arXiv:2610.13194v1 Announce Type: new  Abstract: Long-horizon autonomous research has shown promise in accelerating scientific progress through sustained cycles of hypothesis generation, implementation, and validation. However, this Generate-Execute-Feedback paradigm relies on reactive trial-and-error, where costly and time-consuming executions introduce severe feedback delays, throttling evolutionary progress under explosively expanding search space. To address this challenge, we propose JANUS, a framework that unifies future outcomes forecasting and past experience reflection through a Predict-Verify-Evolve loop driven by a co-evolving world model. Within an evolving graph, the research agent proposes candidates while world model scores them by forecasting value improvement, information gain, and exploration utility, then accordingly selects, refines, or prunes branches prior to execution. After receiving ground-truth feedback, JANUS distills experience into evolving skills: hierarch
    
[^59]: PreSTO：面向快速大语言模型幂采样的预测性子树预取

    PreSTO: Predictive Subtree Prefetching for Fast LLM Power Sampling

    [https://arxiv.org/abs/2610.13171](https://arxiv.org/abs/2610.13171)

    PreSTO将未来的Metropolis-Hastings接受/拒绝决策组织成一棵树并批量预取其提议，在可证明保持MH转移核不变的前提下显著加速了LLM幂采样。

    

    幂采样通过从基础模型分布的锐化版本中进行采样，在测试时提升大语言模型（LLM）的推理能力。其采样器PowerMH速度较慢，因为其Metropolis-Hastings（MH）更新是串行的：每个接受/拒绝决策决定了生成下一个提议时所处的状态。先前的工作通过改变采样器或模型来降低这一成本（例如序贯蒙特卡洛、近似前瞻、蒸馏等）。我们提出了预测性子树预取，通过预取未来的提议来加速PowerMH，并且可以证明其MH转移核保持不变。PreSTO将未来的MH结果视为一棵由接受/拒绝决策构成的树：在每次模型调用中，它以批量方式一次性生成该树中预取子树的所有提议，然后遵循原始的MH接受规则沿着实际实现的路径前进。我们还观察到，MH只会到达每个预取子树中稀疏的、近似一元的部分……

    arXiv:2610.13171v1 Announce Type: new  Abstract: Power sampling improves large language model (LLM) reasoning at test time by sampling from a sharpened version of the base model's distribution. Its sampler, PowerMH, is slow because its Metropolis--Hastings (MH) updates are sequential: each accept/reject decision sets the state from which the next proposal is generated. Prior work reduces this cost by changing the sampler or the model (e.g., sequential Monte Carlo, approximate lookahead, distillation). We propose Predictive SubTree Prefetching (PreSTO), which accelerates PowerMH by prefetching future proposals and provably leaves its MH transition kernel unchanged. PreSTO views future MH outcomes as a tree of accept/reject decisions: in each model call, it generates the proposals of a \textit{prefetched subtree} of this tree in one batch, then follows the realized path with the original MH acceptance rule. We also observe that MH reaches only a sparse, nearly unary part of each prefetch
    
[^60]: 面向长程终端智能的智能体数据规模化扩展

    Scaling Agentic Data for Long-Horizon Terminal Intelligence

    [https://arxiv.org/abs/2610.13166](https://arxiv.org/abs/2610.13166)

    提出 TerminalHorizon，一个面向长程终端智能体的全自动化数据合成引擎，通过避免环境合成预先完成关键工作，生成能真正激发智能体探索、实施与验证能力的高质量长程训练数据。

    

    终端智能体正在从本地操作迈向完成真实世界的专业任务。短轨迹虽然能够展示命令和工具调用，但无法充分体现智能体如何通过探索积累证据、利用证据辅助判断、根据反馈持续调整策略，以及将中间结果转化为最终交付成果。高质量的长程轨迹能够在单一问题求解过程中捕捉这些智能体行为模式及其内在联系，为持续性自主工作提供更丰富的经验。然而，环境合成可能预先完成或简化这些行为模式背后的关键工作，从而消除了智能体本应执行的探索、实施和验证环节。为解决这一问题，我们提出了 TerminalHorizon，一个面向长程终端智能体的全自动化数据合成引擎。该方法从真实世界的专业（任务出发），……

    arXiv:2610.13166v1 Announce Type: new  Abstract: Terminal agents are moving beyond local operations toward completing real-world professional tasks. Short trajectories can demonstrate commands and tool calls, but they do not fully show how agents accumulate evidence through exploration, use it to inform judgments, continually adjust strategies based on feedback, and turn intermediate results into final deliverables. High-quality long-horizon trajectories capture these agentic behavioral patterns and their connections within a single problem-solving process, providing richer experience for sustained autonomous work. However, environment synthesis may complete or simplify the key work underlying these patterns in advance and thus remove the need for exploration, implementation, and verification that the agent was meant to perform. To address this, we introduce TerminalHorizon, a fully automated data synthesis engine for long-horizon terminal agents. Starting from real-world professional 
    
[^61]: SOAR：面向搜索代理的、具有精确信用分配的源感知强化学习

    SOAR: Source-Aware Reinforcement Learning with Precise Credit Assignment for Search Agents

    [https://arxiv.org/abs/2610.13157](https://arxiv.org/abs/2610.13157)

    SOAR提出源感知强化学习范式，通过源感知工作流将外部与内部知识解耦为并行可观察的信息流，并结合源一致性奖励实现精确的信用分配，从而有效缓解搜索代理的证据伪造问题。

    

    针对搜索代理的强化学习（RL）常常受到“证据伪造”问题的困扰，这是一种失败模式：智能体绕过外部信息（E_ext）并凭空捏造出错误的答案。我们认为，这种伪造行为源于推理轨迹固有的不透明性，导致智能体过度依赖不可靠的内部知识（E_int）作为捷径。为解决这一问题，我们提出了SOAR（SOurce-Aware Reinforcement learning，源感知强化学习），这是一种将整体式推理转化为源显式轨迹的范式。SOAR（i）通过“源感知工作流”将E_ext和E_int解耦为并行的、可观察的信息流；（ii）引入“源一致性奖励”来引导策略演进。通过将信息来源的归因显式化，SOAR能够在不同知识流之间实现精确的信用分配。评估结果表明，SOAR在3B和7B骨干模型上分别超越基线7.3%和5.4%，同时显著减少了证据伪造行为。

    arXiv:2610.13157v1 Announce Type: new  Abstract: Reinforcement learning (RL) for search agents often suffers from evidence fabrication, a failure mode where agents bypass external information ($E_{ext}$) and fabricate an incorrect answer. We argue that this fabrication arises from an over-reliance on unreliable internal knowledge ($E_{int}$) as shortcuts due to the inherent opacity of reasoning trajectories. To address this, we propose SOAR (SOurce-Aware Reinforcement learning), a paradigm that transforms monolithic reasoning into source-explicit trajectories. SOAR (i) disentangles $E_{ext}$ and $E_{int}$ into parallel, observable streams via a Source-Aware Workflow, and (ii) incorporates Source-Consistent Reward to guide the policy evolution. By making information-source attribution explicit, SOAR enables precise credit assignment between different knowledge streams. Evaluations show that SOAR outperforms baselines on 3B and 7B backbones by 7.3% and 5.4%, while reducing evidence fabri
    
[^62]: 《悲惨世界对话：基于电影的社会交流网络表示》

    Les Mis\'erables Dialogue: A film-based network representation of social communication

    [https://arxiv.org/abs/2610.13148](https://arxiv.org/abs/2610.13148)

    本工作基于1998年电影版《悲惨世界》构建了公开可用的时序文本网络数据集，明确区分每句台词的说话者、受话者、内容与时间，并通过十维社会关系模型给出多层网络表示，同时提供了从叙事素材构建丰富交流数据集的通用方法。

    

    在本工作中，我们从社会交流的视角重新审视了经典的《悲惨世界》基准网络。利用1998年该小说的电影改编版本，我们构建了公开可用的《悲惨世界对话》时序文本网络，该网络明确区分了每句台词的说话者、受话者及其他接收者，并包含台词的内容与时间信息。利用十维社会关系模型，我们还推导出该网络的多层网络表示，其中各层对应不同的关系维度。除网络数据集本身之外，本工作还提供了一种通用方法论，可用于从类似的叙事素材中构建同样丰富的交流数据集。

    arXiv:2610.13148v1 Announce Type: cross  Abstract: In this work, we revisit the classic Les Mis\'erables benchmark network from the perspective of social communication. Using the 1998 film adaptation of the novel, we construct the publicly available Les Mis\'erables Dialogue temporal text network, which explicitly distinguishes speakers, addressees, and other recipients of each utterance, along with its content and timing. Using the ten social dimensions model, we also derive multilayer network representations of the network, in which layers correspond to distinct relational dimensions. Beyond the network dataset, this work offers a general methodology for building similarly rich communication datasets from similar narrative sources.
    
[^63]: Neurograf v1：一个端到端的罗马尼亚语优先语言模型家族

    Neurograf v1: An End-to-End Romanian-First Language Model Family

    [https://arxiv.org/abs/2610.13134](https://arxiv.org/abs/2610.13134)

    该论文提出了Neurograf v1——首个从零开始为罗马尼亚语这一低资源欧洲语言构建的端到端语言模型家族，涵盖635 GB语料库、分词器、五个基础模型、指令微调助手和嵌入模型，验证了低资源语言从零训练大模型的可行性。

    

    我们提出了Neurograf v1，这是一个实验性的端到端罗马尼亚语优先语言模型流水线，旨在研究从零开始为低资源欧洲语言训练语言模型的可行性。该流水线包括：RoSfera，一个635 GB的罗马尼亚语语料库，包含1.757亿个文档；一个8万词表规模的字节级BPE分词器；五个参数规模从1.6亿到60亿不等的仅解码器模型，在1310亿个token上训练；三个经过指令微调的助手模型；以及两个通过对比学习训练的嵌入模型。训练和评估均使用ELI-NP/IFIN-HH的计算基础设施完成。该模型家族表现出验证损失随模型规模一致缩放的趋势，且罗马尼亚语事实性和文化知识随模型规模增大而不断提升。在包含1000个问题的WWTBM基准测试中，Neurograf-6B在罗马尼亚语背景问题上达到69.89%的准确率，Neurograf-3.2B达到67.38%。计算最优分析得出了参数缩放指数为（原文此处截断）

    arXiv:2610.13134v1 Announce Type: new  Abstract: We present Neurograf v1, an experimental end-to-end Romanian-first language-model pipeline developed to investigate the feasibility of training language models from scratch for a lower-resource European language. The pipeline comprises RoSfera, a 635 GB Romanian corpus containing 175.7 million documents; an 80k byte-level BPE tokenizer; five decoder-only models ranging from 160M to 6B parameters, trained on 131 billion tokens; three instruction-tuned assistants; and two contrastively trained embedding models. Training and evaluation were conducted using computing infrastructure at ELI-NP/IFIN-HH.   The model family exhibits consistent scaling of validation loss and increasing Romanian factual and cultural knowledge with model size. On the 1,000-question WWTBM benchmark, Neurograf-6B achieves 69.89% accuracy on Romanian-context questions, while Neurograf-3.2B reaches 67.38%. Compute-optimal analysis yields a parameter-scaling exponent of 
    
[^64]: 定位、判断、阅读：基于知识点索引的角色专门化工具代理用于开放域多模态文档问答

    Locate, Judge, Read: Role-Specialized Tool Agents over a Knowledge-Point Index for Open-Domain Multimodal Document QA

    [https://arxiv.org/abs/2610.13128](https://arxiv.org/abs/2610.13128)

    提出按职能分离的KnowForge流水线，通过知识点索引、仅能搜索的定位器、强制第二跳检索的桥接判断器和仅能阅读的深度阅读器等角色专门化工具代理，在开放域多模态文档问答上显著超越了现有检索增强方法。

    

    对大型异构文档集合进行问答需要先找到正确的段落再进行阅读，然而当前的检索增强代理将两类工具都交给同一个模型并让其自行调度。我们提出了KnowForge，一个按职能分离的流水线：一个混合检索的知识点索引；一个只能执行搜索的定位器代理；一个由代码调度的桥接判断器，强制执行缺失的第二跳检索；一个类型化、大小受限的证据池，用于合并每次搜索的排序结果；一个只能阅读文本、图像和表格的深度阅读器代理；以及一个带有格式检查器的综合器。通过自托管的Qwen3.6-35B-A3B（激活参数3B），KnowForge在完整的开放域M3DocVQA（2,441个问题、3,368个PDF）上达到F1 47.5，比已发布的M3DocRAG系统高出11.0，比在同一索引和模型上进行一次性检索回答高出7.1；使用DeepSeek-V4-Flash时达到53.5。在MultiHop-RAG上达到73.5的准确率，高于最强（摘要在此处截断）

    arXiv:2610.13128v1 Announce Type: new  Abstract: Question answering over large heterogeneous document collections requires first finding the right passages and then reading them, yet current retrieval-augmented agents give one model both kinds of tools and let it schedule itself. We present KnowForge, a pipeline separated by job: a hybrid-retrieval knowledge-point index; a Locator agent that can only search; a code-scheduled bridge judge that forces the missing second hop; a typed, size-bounded evidence pool that merges every search's ranking; a Deep Reader agent that can only read text, images and tables; and a synthesizer with a format checker. With a self-hosted Qwen3.6-35B-A3B (3B active), KnowForge reaches F1 47.5 on the full open-domain M3DocVQA (2,441 questions, 3,368 PDFs), +11.0 over the published M3DocRAG system and +7.1 over one-shot retrieve-and-answer on the same index and model, and 53.5 with DeepSeek-V4-Flash. On MultiHop-RAG it reaches 73.5 accuracy, above the strongest
    
[^65]: BenCoDiC：面向孟加拉语政治新闻话语的大规模语境化网络欺凌语料库（完整版）

    BenCoDiC: A Large-Scale Contextual Cyberbullying Corpus for Bengali Political News Discourse FULL VERSION

    [https://arxiv.org/abs/2610.13121](https://arxiv.org/abs/2610.13121)

    该论文构建并公开了首个大规模、语境化、多维度的孟加拉语政治新闻网络欺凌语料库BenCoDiC，包含50,250条保留评论回复结构的文本实例，通过九维星型模式组织数据，为网络欺凌检测与欺凌者行为网络研究提供有力支持。

    

    网络欺凌是在线社区中日益受到关注的问题，然而现有的孟加拉语语料库在语境信息、标注透明度、规模、可访问性以及对多维分析的支持方面仍然有限，从而阻碍了对欺凌者行为和网络进行稳健检测与全面研究。为了弥补这些不足，我们提出了孟加拉语语境化与多样化网络欺凌语料库，这是一个公开可用的面向孟加拉语政治新闻话语的语境化网络欺凌资源，包含从100个YouTube视频及其相关讨论中收集的50,250条文本实例。该语料库保留了标题—评论—回复之间的关系，并通过九维星型模式进行组织，涵盖用户、视频、文本内容、语境、时间、互动参与度、政党、目标身份和语言类别等维度。研究最初评估了五个AI聊天机器人用于标注，但其未能达到预设的要求（原文摘要在此处被截断）。

    arXiv:2610.13121v1 Announce Type: new  Abstract: Cyberbullying is a growing concern in online communities, yet existing Bengali corpora remain limited in contextual information, annotation transparency, scale, accessibility, and support for multidimensional analysis, thereby hindering robust detection and comprehensive study of harasser behaviors and networks. To address these gaps, we introduce the Bengali Contextual and Diverse Cyberbullying (\textit{BenCoDiC}) corpus, a publicly available contextual cyberbullying resource for Bengali political news discourse comprising 50,250 textual instances collected from 100 YouTube videos and their associated discussions. The corpus preserves caption--comment--reply relationships and is organized through a nine-dimensional star schema covering users, videos, textual content, context, time, engagement, political parties, target identities, and language categories. Five AI chatbots were initially evaluated for annotation but did not meet the pred
    
[^66]: 从210万小时原始广播音频到芬兰语语音自监督学习：大规模置信度过滤与架构分析

    From 2.1M Hours of Raw Broadcast Audio to Finnish Speech SSL: Confidence Filtering and Architectural Analysis at Scale

    [https://arxiv.org/abs/2610.13102](https://arxiv.org/abs/2610.13102)

    该论文提出以ASR置信度作为数据质量过滤标准，从210万小时芬兰语广播音频中筛选高质量语音进行自监督学习预训练，在更小的训练集上相比基于LID的过滤方法取得3-7%的相对WER提升，并系统比较了四种SSL架构的性能。

    

    广播档案为自监督学习（SSL）提供了规模和多样性，但其中包含大量噪声、非语音和多语言内容。先前的工作通常依赖语音活动检测或语言识别（LID）来提取语音，这只能保证语音的存在而非质量。是否以及如何按质量过滤无标注语音用于SSL，尤其是来自嘈杂广播的语音，仍然缺乏深入研究。我们提出了一项大规模芬兰语SSL研究，从210万小时无约束的广播音频出发。我们提出将自动语音识别（ASR）置信度作为预训练数据质量的代理指标，结果显示尽管训练集更小，相比基于LID的过滤方法仍可获得3-7%的相对词错误率（WER）改进。我们进一步解决了语音SSL领域缺乏受控架构比较的问题，在相同参数预算下评估了Transformer、Macaron-Net、Conformer和E-Branchformer。我们的模型超越了先前的芬兰语SSL基线。

    arXiv:2610.13102v1 Announce Type: new  Abstract: Broadcast archives offer scale and diversity for self-supervised learning (SSL) but contain substantial noisy, non-speech, and multilingual content. Prior work typically relies on voice activity detection or language identification (LID) to extract speech, ensuring presence rather than quality. Whether and how to filter unlabeled speech by quality for SSL, particularly from noisy broadcasts, remains underexplored. We present a large-scale Finnish SSL study starting from 2.1M hours of unconstrained broadcast audio. We propose automatic speech recognition (ASR) confidence as a proxy for pretraining data quality and show 3-7% relative WER improvements over LID-based filtering despite a smaller training set. We further address the lack of controlled architectural comparisons in speech SSL by evaluating Transformer, Macaron-Net, Conformer, and E-Branchformer at matched parameter budgets. Our models improve upon prior Finnish SSL baselines, pa
    
[^67]: KV-Repurpose：一个用于评估非前缀 KV 缓存复用的测试平台

    KV-Repurpose: A Testbed for Evaluating Non-Prefix KV-Cache Reuse

    [https://arxiv.org/abs/2610.13060](https://arxiv.org/abs/2610.13060)

    提出基于 vLLM 的开源测试平台 KV-Repurpose，用于开发和评估超越精确前缀限制的分段式 KV 缓存复用，在 RAG 风格工作负载下显著提升缓存命中率并基本保持单文档任务的输出质量。

    

    arXiv:2610.13060v1 公告类型：新论文 摘要：大语言模型推理服务中的 KV 缓存复用通常仅限于精确的提示词前缀，这导致当同一份检索文档出现在不同的系统提示词、对话历史或动态元数据之后时无法进行复用。我们提出了 KV-Repurpose，一个基于 vLLM 的开源系统，用于开发和评估超越精确前缀限制的分段式 KV 缓存复用。该系统集成了缓存感知的分段调度、提示词构建、容量管理、工作负载生成以及运行时监控。在长上下文基准测试上的实验表明，朴素的分段复用在单文档问答和摘要任务中基本保持了输出质量，但在跨文档推理方面暴露出明显的局限。在一种齐夫分布（Zipfian）的 RAG 风格工作负载下（其中 20% 的请求共享固定前缀），KV-Repurpose 在相同的 30 GB 缓存容量下，实现了远高于前缀缓存的 token 级缓存命中率——63.9% 对比 3（原文在此处截断）

    arXiv:2610.13060v1 Announce Type: new  Abstract: KV-cache reuse in large language model serving is commonly limited to exact prompt prefixes, preventing reuse when the same retrieved document appears after different system prompts, conversation histories, or dynamic metadata. We present KV-Repurpose, an open-source, vLLM-based system for developing and evaluating segment-wise KV-cache reuse beyond exact prefixes. The system combines cache-aware segment scheduling, prompt construction, capacity management, workload generation, and runtime monitoring. Experiments on long-context benchmarks show that naive segment reuse largely preserves output quality for single-document question answering and summarization, while exposing clear limitations in cross-document reasoning. Under a Zipfian RAG-style workload in which 20% of requests share a fixed prefix, KV-Repurpose achieves a substantially higher token-level cache hit rate than prefix caching at the same 30 GB cache capacity, 63.9% versus 3
    
[^68]: CliQ-RRG：临床知识引导的疾病感知视觉-文本对齐用于问答式放射报告生成

    CliQ-RRG: Clinical-Knowledge Guided Disease-aware Visual-Textual Alignment for QA-Style Radiology Report Generation

    [https://arxiv.org/abs/2610.13059](https://arxiv.org/abs/2610.13059)

    CliQ-RRG提出了一种两阶段框架，通过疾病感知的视觉-文本对齐与先验引导注意力捕获多视图依赖，并利用GraphRAG检索临床知识进行增强，从而生成可解释且符合临床实际的问答式放射报告。

    

    放射报告对于诊断推理和患者护理至关重要，然而其人工撰写既耗时又对认知能力要求很高。自动放射报告生成（RRG）提供了一种可扩展的替代方案，但现有模型往往生成冗长、非结构化的叙述，忽略了诊断线索和多视图信息。我们提出了CliQ-RRG（临床知识引导的疾病感知视觉-文本对齐问答式放射报告生成），这是一个用于可解释且临床可靠的报告生成的统一两阶段框架。在第一阶段，CliQ-RRG采用疾病感知视觉-文本对齐模块，利用预测的疾病嵌入对齐图像和文本表示，并通过先验引导注意力模块（PrAM）来捕获当前扫描与既往扫描之间的多视图依赖关系。在第二阶段，GraphRAG引导的检索通过基于图的临床知识来增强中间文本表示……

    arXiv:2610.13059v1 Announce Type: new  Abstract: Radiology reports are essential for diagnostic reasoning and patient care, yet their manual preparation is time-consuming and cognitively demanding. Automatic radiology report generation (RRG) offers a scalable alternative, but existing models often produce lengthy, unstructured narratives that overlook diagnostic cues and multi-view information. We present Clinical-Knowledge Guided Disease-aware Visual-Textual Alignment for QA-Style Radiology Report Generation (CliQ-RRG), a unified two-stage framework for interpretable and clinically grounded reporting. In Stage 1, CliQ-RRG employs a Disease-aware Visual-Textual Alignment module that aligns image and text representations using predicted disease embeddings, reinforced by a Prior-Guided Attention Module (PrAM) to capture multi-view dependencies across current and prior scans. In Stage 2, GraphRAG-guided retrieval augments intermediate textual representations with graph-grounded clinical e
    
[^69]: 回溯前行：面向长时程智能体的回顾性探索与演化

    Travel Back To Move Forward: Retrospective Exploration and Evolving for Long Horizon Agents

    [https://arxiv.org/abs/2610.13058](https://arxiv.org/abs/2610.13058)

    提出ReTRAIL框架，使智能体能够将环境回退至较早状态进行回顾性探索，同时保留已学知识与有效进展，并通过状态锚定GRPO和分支推进奖励两种互补信号训练策略，帮助长时程智能体主动从死胡同轨迹中恢复。

    

    大语言模型智能体越来越多地在长时程环境中运行，在这种环境中，早期的错误可能引发连锁式失败，使智能体陷入无法继续的死胡同轨迹。因此，智能体必须能够主动从这类状态中恢复，并将探索中获得的经验转化为后续决策的指导。我们提出了ReTRAIL，这是一个使智能体能够将环境回退至较早状态的框架，同时保留有效的进展，并携带从被放弃的探索中所学到的知识。环境状态向后回溯，而智能体的知识持续向前演化。ReTRAIL通过两种互补的信号训练共享策略：状态锚定的GRPO（State-Anchored GRPO）通过比较从重访状态出发的不同后续行动方案来改进执行能力，而分支推进奖励则通过后续探索的进展来引导反思。在四个交互式基准测试上的实验：扫雷、推箱子、ALFWorld以及

    arXiv:2610.13058v1 Announce Type: new  Abstract: Large language model agents increasingly operate in long-horizon environments where early mistakes can trigger cascading failures and trap agents in dead-end trajectories. Agents must therefore be able to proactively recover from such states and turn their exploratory experience into guidance for subsequent decisions. We introduce ReTRAIL, a framework that enables agents to rewind the environment to earlier states while preserving valid progress and carrying forward knowledge learned from abandoned exploration. The environment state moves backward while the agent's knowledge continues to move forward. ReTRAIL trains a shared policy with two complementary signals: State-Anchored GRPO compares alternative continuations from revisited states to improve execution, while a branch-advancement reward guides reflection through the progress of subsequent exploration. Experiments across four interactive benchmark MineSweeper, Sokoban, ALFWorld and
    
[^70]: CoBe：面向对话式反事实文本编辑的基准测试

    CoBe: A Benchmark For Conversational Counterfactual Text Editing

    [https://arxiv.org/abs/2610.13057](https://arxiv.org/abs/2610.13057)

    该论文提出了CoBe基准，用于评估模型的对话式反事实文本编辑能力，要求模型通过因果理解判断假设性变化会影响的范围，在修改文本的同时保留未受因果影响的事实，而非仅生成统计上相关联的文本。

    

    反事实文本编辑是指根据一个假设性变化来修改给定文本的能力，同时保留未受该变化因果影响的事实。当被问及“如果医生没有推荐手术会发生什么”时，模型输出应保留患者的病史，仅修改该决定所带来的后果。这需要因果理解能力：模型必须在没有被明确指示的情况下，自行判断哪些事件受到假设性变化的影响，哪些内容应保持不变。这项技能是人类推理的核心，尤其在医疗或法律等安全关键领域中至关重要。相比之下，关联性文本生成任务只是生成与改变后的条件在统计上相关联的文本。在上述医疗示例中，关联性响应可能会让患者的症状变得更轻，因为较轻的病例通常与不做手术相关。这样的响应会产生一个看似合理的场景……

    arXiv:2610.13057v1 Announce Type: cross  Abstract: Counterfactual text editing is the ability to revise a given text in light of a hypothetical change while preserving facts not causally affected by that change. When asked what would have happened if a doctor had not recommended surgery, the model output should retain the patient's medical history and modify only the consequences of the decision. This requires causal understanding: a model must determine, without being explicitly instructed to, what events are affected by the hypothetical change and what should remain fixed. This skill is central to human reasoning, especially in safety-critical domains like healthcare or law. By contrast, the task of associational text generation merely generates text statistically associated with the altered condition. In the medical example, an associational response might make the patient's symptoms less severe because milder cases correlate with no surgery. Such responses produce a plausible scena
    
[^71]: RAMP：面向循环深度语言模型高效预填充的循环自适应混合精度方法

    RAMP: Recurrent Adaptive Mixed Precision for Efficient Prefill in Recurrent-Depth Language Models

    [https://arxiv.org/abs/2610.13041](https://arxiv.org/abs/2610.13041)

    该论文提出RAMP方法，通过揭示“越早引入的量化误差会被越多后续循环步骤衰减”这一传播规律，将量化计算自适应地放置在合适的循环步骤上，从而在保持精度的同时高效加速循环深度语言模型的长上下文预填充。

    

    循环深度语言模型通过反复应用一个共享网络来精炼内部表示，从而提升推理时的推理能力，但这种额外的循环计算也增加了推理成本。这一开销在长上下文预填充阶段尤为显著，因为在预填充时，完整的输入序列会沿循环深度被反复处理。权重与激活量化提供了一种通过低精度算术来加速此类计算的自然途径，然而反复应用量化计算会引入沿循环轨迹不断累积的数值扰动。我们系统地分析了这类误差如何在循环深度上传播，并证明较早引入的量化误差会被后续更多的循环步骤所衰减，因而对最终表示的影响更小。匹配计算预算的实验有力地支持了这一预测，表明将量化计算放置在……（原文摘要在此处被截断）

    arXiv:2610.13041v1 Announce Type: new  Abstract: Recurrent-depth language models improve inference-time reasoning by repeatedly applying a shared network to refine internal representations, but the additional recurrent computation also increases inference cost. This overhead is particularly pronounced during long-context prefill, where the full input sequence is repeatedly processed across recurrent depth. Weight-and-activation quantization offers a natural way to accelerate this computation through lower-precision arithmetic, yet repeatedly applying quantized computation introduces numerical perturbations that accumulate along the recurrent trajectory. We systematically analyze how such errors propagate across recurrent depth and show that quantization errors introduced earlier are attenuated by more subsequent recurrent steps and therefore have a smaller effect on the final representation. Matched-budget experiments strongly support this prediction, showing that placing quantized com
    
[^72]: GeoGR：面向高效3D视觉语言模型的空间-语义令牌压缩与分组路由

    GeoGR: Spatial-Semantic Token Compression and Group Routing for Efficient 3D VLMs

    [https://arxiv.org/abs/2610.13035](https://arxiv.org/abs/2610.13035)

    提出GeoGR两阶段框架，通过将跨视角观察压缩为3D体素令牌（GeoSemZip）并在LLM层间进行令牌分组路由，联合压缩场景表示与视觉计算，从而实现高效的3D视觉语言模型推理。

    

    多视角表示使预训练的图像和视频视觉语言模型能够理解3D场景，但其冗长的视觉令牌序列导致推理成本高昂。冗余一方面来自不同视角对同一场景区域的重复观察，另一方面来自在LLM的每一层都处理所有视觉令牌，而不考虑其与当前计算的相关性。我们提出GeoGR，这是一个联合压缩场景表示与视觉计算的两阶段框架。在投影器层面，GeoSemZip将跨视角观察整合为3D体素令牌，并结合语义显著性与空间覆盖来构建紧凑的场景表示。未被选中的体素特征被聚合为上下文锚点，以保留原本会被丢弃的场景信息。在LLM层面，我们识别出有效的视觉计算窗口，并观察到基于查询条件的令牌相关性保持局部稳定，但（摘要在此处截断）

    arXiv:2610.13035v1 Announce Type: cross  Abstract: Multi-view representations enable pretrained image and video vision-language models to understand 3D scenes, but their long visual token sequences make inference costly. Redundancy arises both from repeated observations of the same scene regions across views and from processing all visual tokens at every LLM layer, regardless of their relevance to the current computation. We propose GeoGR, a two-stage framework that jointly compresses scene representations and visual computation. At the projector level, GeoSemZip consolidates cross-view observations into 3D voxel tokens and combines semantic saliency with spatial coverage to construct a compact scene representation. Unselected voxel features are aggregated into contextual anchors to retain scene information that would otherwise be discarded. At the LLM level, we identify an effective visual computation window and observe that query-conditioned token relevance remains locally stable but
    
[^73]: Evo²：通过权重与框架协同进化实现进化器的进化

    Evo$^2$: Evolving the Evolver through Weight-Harness Co-Evolution

    [https://arxiv.org/abs/2610.13025](https://arxiv.org/abs/2610.13025)

    提出了权重-框架协同进化框架 Evo²，让单一策略同时担任提议者和反思者来完成框架的自我修改，并利用框架修改及其执行结果指导模型权重训练，从而实现智能体权重与框架的联合进化。

    

    自进化智能体越来越多地被用于在无需人工干预的情况下改进语言模型系统，其途径包括权重更新或框架改进。现有方法通常将这两个方向分开处理：一些方法利用自生成的经验来更新模型权重，另一些方法则重写定义智能体如何使用提示、工具、上下文和执行反馈的框架。然而，这些方法忽略了模型权重与智能体框架之间的关系。为了将这两个过程联系起来并确保有效且平滑的智能体自我进化，我们提出了 Evo²，一个权重-框架协同进化框架。在框架更新方面，我们设计了一个统一的双角色流水线，即由单一策略同时担任“提议者”和“反思者”两种角色，前者负责撰写框架修改，后者负责诊断修改效果并指导修订。在权重更新方面，框架修改及其执行结果被用于指导模型的训练。

    arXiv:2610.13025v1 Announce Type: new  Abstract: Self-evolving agents are increasingly used to improve language model systems without human intervention through either weight updates or harness improvements. Existing approaches typically pursue these directions separately: some methods update model weights using self-generated experience, while others rewrite the harness that defines how an agent uses prompts, tools, context, and execution feedback. However, these approaches overlook the relationship between model weights and the agent harness. To connect these processes and ensure effective and smooth agent self-evolving, we propose Evo$^2$, a framework for weight-harness co-evolution. For harness updates, we design a unified dual-role pipeline that a single policy serves as both a Proposer, which writes harness edits, and a Reflector, which diagnoses their effects and guides revision. For weight updates, harness modifications and their execution outcomes guide the training of both th
    
[^74]: EmbodiedCLBench：评估自我进化具身智能体的持续学习能力

    EmbodiedCLBench: Evaluating Continual Learning for Self-Evolving Embodied Agents

    [https://arxiv.org/abs/2610.13021](https://arxiv.org/abs/2610.13021)

    提出了首个具身仿真持续学习基准 EmbodiedCLBench，以组合泛化为核心，评估自我进化智能体能否将基础任务中积累的经验迁移并重组以解决未见的高级任务。

    

    持续学习要求模型积累过去的经验并将其迁移到新任务上。尽管持续学习已得到大量研究，现有工作大多仍局限于非具身场景，尚未扩展到具身环境。为填补这一空白，我们提出了 EmbodiedCLBench——首个用于评估自我进化智能体持续学习能力的具身仿真基准。该基准以组合泛化为核心评估原则，考察智能体能否先从基础任务中学习，再将所习得的技能重新组合，以解决复杂且未曾见过的高级任务。基于 EmbodiedCLBench，我们在多种智能体框架和学习方法上开展了大量实验，揭示了一致的局限性与独特的行为特征。我们的结果表明，虽然在零样本设置下高级任务几乎无法被解决，但来自基础任务的经验能够带来持续的提升（原文此处截断）。

    arXiv:2610.13021v1 Announce Type: new  Abstract: Continual learning requires models to accumulate past experience and transfer it to new tasks. Despite substantial research on continual learning, existing work remains largely confined to non-embodied settings, without extending to embodied environments. To address this gap, we introduce EmbodiedCLBench, the first benchmark in embodied simulation for evaluating the continual learning capability of self-evolving agents. With compositional generalization as the core evaluation principle, our benchmark evaluates whether agents can first learn from basic tasks and then recombine the acquired skills to solve complex, unseen advanced tasks. Based on EmbodiedCLBench, we conduct extensive experiments across multiple agent harnesses and learning methods, revealing consistent limitations and distinctive behaviors. Our results show that while advanced tasks can be hardly solved under zero-shot setting, the experience from basic tasks enables consi
    
[^75]: 通过长尾表现评估大语言模型的创造力

    Evaluating LLM Creativity through Long-Tail Performance

    [https://arxiv.org/abs/2610.12998](https://arxiv.org/abs/2610.12998)

    该论文提出将大语言模型的创造力形式化为参考输入-输出分布长尾部分的任务特定表现，并引入q@Tail(α)指标，实证发现基于较高尾部子集的模型排名与人类创造力判断更为一致。

    

    针对大语言模型（LLM）的创造力基准在任务与指标设计上差异巨大，这引发了一个问题：这些基准是否捕捉到了同一个创造力概念。我们提出，这一共同目标可以被形式化为在参考输入-输出分布长尾部分的任务特定表现，该分布描述了输入-输出对相对于所选文本来源的常见程度。我们将这一命题建立在“新颖性-质量乘积”公式的基础上，并在15个基准和26个开源权重模型上进行了实证评估，使用n-gram在预训练数据中的缺失作为新颖性的代理指标。我们的结果表明，创造力基准比标准基准更深入地处于长尾区域。此外，无论在创造力基准还是标准基准上，基于每个基准较高尾部子集得到的模型排名与LM-Arena中的人类创造力判断更加一致。基于这些发现，我们提出了q@Tail(α)指标。

    arXiv:2610.12998v1 Announce Type: new  Abstract: Creativity benchmarks for large language models (LLMs) vary widely in task and metric design. This raises the question of whether they capture a common notion of creativity. We propose that this shared target can be formalized as task-specific performance in the long tail of a reference input-output distribution, which describes how common input-output pairs are relative to a chosen source of text. We ground this proposition in the novelty-quality product formulation and evaluate it empirically across 15 benchmarks and 26 open-weight models, using n-gram absence from the pretraining data as a novelty proxy. Our results show that creativity benchmarks lie further in the tail regions than standard benchmarks. Moreover, on both creativity and standard benchmarks, model rankings from higher-tail subsets of each benchmark align more closely with human creativity judgments in LM-Arena. Building on these findings, we introduce q@Tail($\alpha$),
    
[^76]: 面向无监督专利权利要求分类的嵌入几何与聚类算法基准测试

    Benchmarking Embedding Geometries and Clustering Algorithms for Unsupervised Patent Claim Classification

    [https://arxiv.org/abs/2610.12997](https://arxiv.org/abs/2610.12997)

    双曲空间嵌入能显著提升跨分类代码的现有技术检索能力——双曲VAE比欧几里得VAE多检索到52%的真正跨类别现有技术，尽管它与官方专利分类系统的一致性更低。

    

    现有技术检索系统在结构上是围绕分类代码组织的，这使得跨代码的发现变得困难。本文探究在专利文本上训练的无监督表示能否恢复受代码限制的搜索无法发现的相关联系，以及双曲空间是否有所帮助。我们在45,000件美国专利上训练了双曲和对应的欧几里得变分自编码器（VAE）（96次运行，6个随机种子）。双曲模型检索到真正跨类别现有技术的频率比欧几里得模型高52%（在前10个结果中分别为18.76%对12.34%，共3,000个查询）。这对本文所针对的检索问题而言是一个实用性增益。双曲模型与官方分类的一致性较低（NMI分别为0.2392对0.3206），我们从几何角度对此进行了解释：这两种度量捕捉的目标不同，只有其中之一与现有技术检索的需求相契合。在对10个技术领域（另有50,000件专利）的研究中，我们发现一个领域的分类……

    arXiv:2610.12997v1 Announce Type: new  Abstract: Prior-art retrieval systems are structurally organized around classification codes, making cross-code discovery difficult. This paper asks if an unsupervised representation trained on patent text can recover relevant connections that code-restricted search cannot, and if hyperbolic space helps. We trained a hyperbolic and a matched Euclidean Variational Autoencoder (VAE) on 45,000 US patents (96 runs, 6 seeds). The hyperbolic model retrieves genuinely cross-category prior art 52% more often than the Euclidean model (18.76% vs. 12.34% of top-10 results, 3,000 queries). This represents a practical gain for the motivating search problem. The hyperbolic model agrees less with official classification (NMI 0.2392 vs. 0.3206), which we explain geometrically: the two measures capture different goals, and only one aligns with prior-art search needs. In a study of 10 technology domains (50,000 more patents), we find that a domain's classification 
    
[^77]: SFT应该学习什么？用于可泛化后训练的条件信息增益

    What Should SFT Learn? Conditional Information Gain for Generalizable Post-Training

    [https://arxiv.org/abs/2610.12995](https://arxiv.org/abs/2610.12995)

    提出CIG-SFT信息论框架，用条件信息增益而非模型兼容性来决定SFT应学习哪些词元，从而提升大语言模型后训练的泛化能力。

    

    监督微调（SFT）仍然是大语言模型后训练中广泛使用的方法，因为固定的示范可以直接引入期望的推理策略和行为模式。然而，强迫模型模仿外部提供的轨迹也可能干扰预训练已经支持的行为，从而限制模型在训练分布之外的泛化能力。近期的动态SFT方法通过根据模型侧信号（如预测概率、先验支持度或分布兼容性）对词元级监督进行重新加权来缓解这一问题。然而，模型兼容性并不等同于学习价值：一个当前模型支持度较弱的词元，可能是不希望出现的不匹配，也可能恰恰是示范意图教授的任务特定决策。我们提出CIG-SFT，这是一个信息论框架，用于估计任务条件下的信息量……

    arXiv:2610.12995v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) remains a widely used approach for post-training large language models, because fixed demonstrations can directly introduce desired reasoning strategies and behavioral patterns. Yet forcing a model to imitate externally provided trajectories may also interfere with behaviors already supported by pretraining, limiting generalization beyond the training distribution. Recent dynamic SFT methods mitigate this issue by reweighting token-level supervision according to model-side signals such as predictive probability, prior support, or distributional compatibility. However, model compatibility is not equivalent to learning value: a token that is weakly supported by the current model may either be an undesirable mismatch or precisely the task-specific decision that the demonstration is intended to teach. We propose CIG-SFT, an information-theoretic framework that estimates the task-conditioned information carried 
    
[^78]: 语言模型具有可解释的叙事机制

    Language Models Have Interpretable Narrative Mechanisms

    [https://arxiv.org/abs/2610.12989](https://arxiv.org/abs/2610.12989)

    该研究在多个语言模型中识别出激活与文本叙事性相关的注意力头、MLP神经元和稀疏自编码器特征，并通过消融实验提供了这些叙事相关组件因果性地影响模型行为（尤其是故事补全和叙事问答任务）的证据。

    

    我们研究语言模型中是否包含有助于表征叙事性的内部组件，叙事性指文本展现叙事特征的程度。在GPT-2-Small、Gemma-2-2B和Llama-3.1-8B三个模型中，我们识别出了激活与叙事性相关的注意力头、MLP神经元以及稀疏自编码器特征。消融选定的神经元和特征集合后，高叙事性文本的下一个词预测损失比低叙事性文本增加得更多，并且开放式生成文本的叙事性随之降低。在下游基准测试中，最大的影响出现在故事补全和叙事问答任务上，尽管这些影响因模型和语言而异；而在虚构和幻觉基准测试上的结果则好坏参半。这些发现为选定的叙事相关组件集合对模型行为有实际贡献提供了因果证据，同时它们的内部组织方式、与叙事理论的关系以及更广泛的……（摘要在此处截断）

    arXiv:2610.12989v1 Announce Type: new  Abstract: We investigate whether language models contain internal components that contribute to representing narrativity, the degree to which a text exhibits narrative properties. Across GPT-2-Small, Gemma-2-2B, and Llama-3.1-8B, we identify attention heads, MLP neurons, and sparse autoencoder features whose activations correlate with narrativity. Ablating selected neuron and feature sets increases next-token prediction loss more for high-narrativity than for low-narrativity texts, and reduces the narrativity of open-ended generations. On downstream benchmarks, the largest effects occur in story completion and narrative question answering, although they vary across models and languages; results on confabulation and hallucination benchmarks are mixed. These findings provide causal evidence that selected sets of narrative-associated components contribute to model behaviour, while their internal organisation, relation to narrative theory, and broader
    
[^79]: 掩码文本重建中语料库采样分歧的预测

    Forecasting Corpus-Sampling Disagreement in Masked Text Reconstruction

    [https://arxiv.org/abs/2610.12987](https://arxiv.org/abs/2610.12987)

    该论文推导出一个一阶公式，可仅凭参考语料库预测两个独立语料库上训练的掩码文本重建预测器之间的期望分歧，并发现语料库扩大四倍仅使分歧降低5–40%，远低于1/n缩放律的预期，其根源在于罕见上下文的平滑效应。

    

    如果从同一总体中再次收集一个文本语料库，预测器的输出会发生多大变化？我们针对掩码文本重建任务研究了这一问题，使用了一种由邻近词对计数构建的平滑预测器，并在完整的50,257词元的GPT-2词表上进行评估。将每个源文档视为一个观测值，我们推导出了拟合于两个独立语料库的预测器之间期望分歧的一阶公式。该公式考虑了可变的文档长度、随机上下文计数以及跨词元偏移的相关性，并且可以在不访问任何目标语料库的情况下，仅凭一个单独的参考语料库进行计算。在TinyStories、WikiText-103和CNN/DailyMail数据集上，将每个语料库从512个块扩展到2,048个块（每块64个词元）仅使分歧降低了5–40%，远低于1/n缩放律所隐含的75%。该预测准确地追踪了这种缓慢的下降趋势，我们将其归因于对罕见上下文的平滑处理。

    arXiv:2610.12987v1 Announce Type: new  Abstract: If a second text corpus were collected from the same population, how much would a predictor's outputs change? We study this question for masked-text reconstruction using a smoothed predictor built from nearby token-pair counts and evaluated over the full 50,257-token GPT-2 vocabulary. Treating each source document as one observation, we derive a first-order formula for the expected disagreement between predictors fitted to two independent corpora. The formula accounts for variable document length, random context counts, and correlations across token offsets, and it can be evaluated from a separate reference corpus without access to either target corpus. On TinyStories, WikiText-103, and CNN/DailyMail, growing each corpus from 512 to 2,048 chunks of 64 tokens lowers disagreement by only 5--40\%, far short of the 75\% implied by $1/n$ scaling. The forecast tracks this slow decline, which we trace to smoothing of rarely seen contexts. Forec
    
[^80]: SmartSalesBench：一个用于评估大语言模型销售代理的抗性感知多轮基准测试

    SmartSalesBench: A Resistance-Aware Multi-Turn Benchmark for Evaluating LLM Sales Agents

    [https://arxiv.org/abs/2610.12975](https://arxiv.org/abs/2610.12975)

    SmartSalesBench是首个专为评估LLM销售代理应对具有心理抗拒性买家而设计的多轮对话基准测试，通过13,680个会话和三层级九项评估指标，颠覆了传统对话基准假设用户合作的评估范式。

    

    我们提出了SmartSalesBench，这是首个明确设计用于评估大型语言模型（LLM）销售代理应对心理层面真实且具有抗拒性买家角色的多轮对话基准测试。现有的任务导向对话基准测试假设用户是合作的并衡量任务完成度；SmartSalesBench颠覆了这一假设，将代理置于被设定为在长达二十轮对话中不断抗拒和挑战的买家角色面前。该基准测试包含五个代理档案、十二个买家角色、横跨六个价格档位的十五款智能手机产品，以及四次包含3次迭代的对照实验，总计13,680个会话。销售代理和买家角色的生成均通过检索增强生成（RAG）进行基础支撑。我们引入了九个评估指标，分为三个层级：主要结果指标（CR、BRS、RIR）、次要质量指标（SMS、PFS、ETI）以及计划在未来报告中呈现的探索性指标。

    arXiv:2610.12975v1 Announce Type: new  Abstract: We present SmartSalesBench, the first multi-turn conversational benchmark explicitly designed to evaluate large language model (LLM) sales agents against psychologically grounded, resistant buyer personas. Existing task-oriented dialogue benchmarks assume cooperative users and measure task completion; SmartSalesBench inverts this assumption, placing agents against personas programmed to resist and challenge across conversations of up to twenty turns. The benchmark includes five agent profiles, twelve buyer personas, fifteen smartphone products across six price tiers, and four controlled experiments with 3 iterations totaling 13,680 sessions. Both sales-agent and buyer persona generations are grounded through retrieval-augmented generation (RAG). We introduce nine evaluation metrics grouped into three tiers: primary outcome metrics (CR, BRS, RIR), secondary quality metrics (SMS, PFS, ETI), and exploratory metrics planned for future report
    
[^81]: MVGraph：在多视图视觉图上评测大型视觉语言模型的证据组合能力

    MVGraph: Benchmarking LVLMs on Evidence Composition across Multi-View Visual Graph

    [https://arxiv.org/abs/2610.12961](https://arxiv.org/abs/2610.12961)

    提出MVGraph基准，通过8千个实例和8个图论任务评测大型视觉语言模型在多视图图推理中跨视图证据对齐与组合的能力，并将其分解为感知、对齐、组合和推理四种互补能力。

    

    我们提出了MVGraph，这是一个用于评估大型视觉语言模型（LVLMs）多图推理能力的基准，其中查询所需的证据分布在多个相互关联的视觉图视图中。尽管此类证据通常存在于特定来源的图、时间快照或局部观察中，但现有的图推理基准仍局限于单一的自包含结构，仅评估图感知和图推理能力，而对跨视图证据的对齐与组合的探索仍然不足。MVGraph包含8千个实例，涵盖8个图论任务，这些任务从两种多图场景（部分图视图和时序图视图）中抽象而来。每个问题均通过程序化方式构建，并被验证无法从单一视图求解。除了端到端评估之外，我们还将多图推理分解为四种互补能力——感知、对齐、组合和推理——

    arXiv:2610.12961v1 Announce Type: cross  Abstract: We introduce MVGraph, a benchmark for evaluating large vision-language models (LVLMs) on multi-graph reasoning where the evidence required by a query is distributed across multiple interrelated visual graph views. Although such evidence commonly resides in source-specific graphs, temporal snapshots, or local observations, existing graph reasoning benchmarks remain confined to a single self-contained structure, assessing only graph perception and reasoning while leaving the alignment and composition of evidence across views underexplored. MVGraph comprises 8k instances spanning 8 \textit{graph-theoretic} tasks, abstracted from two multi-graph scenarios (partial and temporal graph views). Each problem is programmatically constructed and verified to be unsolvable from a single view. Beyond end-to-end evaluation, we decompose multi-graph reasoning into four complementary capabilities---Perception, Alignment, Composition, and Reasoning---an
    
[^82]: 冲浪激活波：面向推测解码的大语言模型隐藏状态信息与几何特性分析

    Surfing the Activation Wave: Analyzing the Information and Geometry of LLM Hidden States for Speculative Decoding

    [https://arxiv.org/abs/2610.12939](https://arxiv.org/abs/2610.12939)

    该论文提出了一个基于矩阵熵信息量分析与中心化核对齐（CKA）多样性度量的原则性框架，用于为大语言模型推测解码选择多层隐藏状态，并揭示了解码器表示中的几何与信息分区特性。

    

    来自大语言模型（LLM）不同层的解码器隐藏状态通常编码互补信息。然而，现有的推测解码方法通常依赖于通过固定的、基于位置的启发式规则所选的少量目标模型层。选择最优的层子集本质上是一个组合问题，即使对于中等深度的模型，穷举式评估也不可行。在这项工作中，我们提出了一个用于推测解码的多层隐藏状态选择的原则性框架，该框架基于两个互补的准则：信息量——通过矩阵熵分析来度量；多样性——使用中心化核对齐（CKA）来量化。我们研究了逐层的香农熵、隐藏状态拼接下的聚合熵，以及跨网络深度的成对CKA动态，从而揭示了解码器表示中截然不同的几何与信息分区。基于这些发现，我们提出了……

    arXiv:2610.12939v1 Announce Type: cross  Abstract: Decoder hidden states from different layers of large language models (LLMs) often encode complementary information. However, existing speculative decoding methods typically rely on a small set of target-model layers chosen using fixed, position-based heuristics. Selecting an optimal subset of layers is inherently a combinatorial problem, making exhaustive evaluation infeasible even for moderately deep models. In this work, we present a principled framework for multi-layer hidden-state selection for speculative decoding based on two complementary criteria: informativeness, measured through matrix-entropy analysis, and diversity, quantified using centered kernel alignment (CKA). We investigate layer-wise Shannon entropy, aggregate entropy under hidden-state concatenation, and pairwise CKA dynamics across depth, revealing distinct geometric and informational compartments within decoder representations. Motivated by these findings, we prop
    
[^83]: SEEDS：面向搜索智能体的预览引导式证据准入框架

    SEEDS: Preview-Guided Evidence Admission for Search Agents

    [https://arxiv.org/abs/2610.12932](https://arxiv.org/abs/2610.12932)

    该论文提出SEEDS框架，针对搜索智能体中被忽视的“证据准入”问题，通过轻量级预览引导智能体根据当前推理状态选择性地决定哪些检索候选文档应进入其推理上下文，摆脱了传统固定或基于排名的文档纳入策略。

    

    搜索智能体通过多轮访问外部知识来扩展大型推理模型的能力，然而现有大多数系统在很大程度上将检索证据与消费证据混为一谈。检索到的文档通常按照固定或基于排名的策略被纳入，尽管证据的实际效用取决于智能体当前的推理状态。因此，排名较低的候选文档可能对当前步骤至关重要，而排名较高的候选文档反而可能提供冗余、无关或误导性的信息。我们将这一长期被忽视的问题形式化为“证据准入”，即在每次搜索步骤中，应允许哪些检索到的候选文档进入智能体不断演进的推理上下文。为实现这种选择性控制，我们提出了SEEDS（Selective Evidence Entry During Search，搜索过程中的选择性证据准入），一个面向搜索智能体的预览引导式证据准入框架。SEEDS首先向智能体展示检索候选文档的轻量级预览，然后让……

    arXiv:2610.12932v1 Announce Type: new  Abstract: Search agents extend large reasoning models with multi-round access to external knowledge, yet most existing systems largely conflate retrieving evidence with consuming it. Retrieved documents are commonly incorporated according to fixed or rank-based policies, despite the fact that evidence utility is conditioned on the agent's current reasoning state. Consequently, a lower-ranked candidate may be critical for the current step, while a higher-ranked candidate may provide redundant, irrelevant, or misleading information. We formulate this overlooked problem as evidence admission, which asks, at each search step, which retrieved candidates should be allowed to enter the agent's evolving reasoning context. To enable such selective control, we introduce SEEDS (Selective Evidence Entry During Search), a preview-guided evidence admission framework for search agents. SEEDS first exposes lightweight previews of retrieved candidates and then let
    
[^84]: 重新思考扩散与解码的角色：用于并行语言生成的预测态流

    Rethinking the Roles of Diffusion and Decoding: Predictive-State Flows for Parallel Language Generation

    [https://arxiv.org/abs/2610.12898](https://arxiv.org/abs/2610.12898)

    该论文提出预测态流（PSF），重新划分扩散先验与解码器的生成职责，让整流流先验仅生成基于前文的高层预测态，再由块因果解码器并行完成词元级选择，从而实现并行语言生成。

    

    在连续扩散语言模型（DLMs）中，由扩散先验生成的潜变量几乎确定性地被映射到词元，映射方式要么通过解码器，要么通过词元嵌入矩阵。与图像生成不同——图像中视觉细节可以连续变化——扩散先验必须在连续空间中捕捉细粒度的离散词元选择，这可能使得联合潜分布难以建模。我们主张应当重新划分生成角色：扩散先验应当生成高层次的预测信息，而解码器则负责解决剩余的词元级选择。基于这一观点，我们提出了预测态流（Predictive-State Flows, PSF），该方法将序列划分为多个块，并将每个块与一个仅由前文上下文计算得到的预测态相关联。一个整流流先验生成预测态序列，随后一个块因果解码器并行生成所有块。

    arXiv:2610.12898v1 Announce Type: new  Abstract: In continuous diffusion language models (DLMs), latents generated by the diffusion prior are mapped almost deterministically to tokens, either by a decoder or through a token embedding matrix. Unlike in image generation, where visual details can vary continuously, the diffusion prior must capture fine-grained discrete token choices in continuous space, which can make the joint latent distribution difficult to model. We argue that generative roles should instead be divided: the diffusion prior should generate high-level predictive information, while the decoder should resolve the remaining token-level choices. Based on this view, we introduce Predictive-State Flows (PSF), which partitions a sequence into blocks and associates each block with a predictive state computed solely from the preceding context. A rectified-flow prior generates the sequence of predictive states, after which a block-causal decoder generates all blocks in parallel, 
    
[^85]: 面向长程多方对话记忆的时间证据森林

    Temporal Evidence Forests for Long-Horizon Multi-Party Conversational Memory

    [https://arxiv.org/abs/2610.12874](https://arxiv.org/abs/2610.12874)

    该论文提出T-Mem——一种基于时间结构的记忆框架，通过“时间证据森林”同时恢复局部时间区间内密集集中的证据并重构完整的事件演进顺序，从而克服现有记忆框架仅将时间作为元数据或检索信号、无法追踪长程多方对话记忆动态演变的局限。

    

    长期记忆已被广泛应用于基于大语言模型（LLM）的智能体中，以支持持久交互和个性化辅助。然而，长期的多方交互会积累大量详细信息，其中决策被反复更新，状态在不同参与者和交互阶段之间不断演化。因此，时间结构对于追踪此类记忆至关重要：相关信息往往在短时间间隔内密集出现，而不同状态之间的转换则由严格的时间顺序所锚定。现有的时间感知记忆框架主要将时间作为元数据存储，或使用时间信号来指导对预先构建记忆结构的检索，而未能利用时间结构来同时恢复局部集中的证据并重构其完整的事件演进过程。为解决这些局限，我们提出了T-Mem，一个基于时间结构的记忆框架，该框架保存交互并组织……（原文摘要在此处截断）

    arXiv:2610.12874v1 Announce Type: new  Abstract: Long-term memory has been widely adopted in LLM-based agents to support persistent interaction and personalized assistance. Long-term multi-party interactions, however, accumulate substantially more detailed information, with decisions repeatedly updated and states evolving across participants and interaction stages. Temporal structure is therefore crucial for tracing such memories: relevant information often appears densely within short intervals, while transitions among different states are anchored by a strict temporal order. Existing temporally aware memory frameworks primarily store time as metadata or use temporal signals to guide retrieval over preconstructed memory structures, without using temporal structure to both recover locally concentrated evidence and reconstruct its complete event progression. To address these limitations, we propose T-Mem, a temporal-structure-based memory framework that preserves interactions and organi
    
[^86]: 用于序列建模的时滞系统

    Time Delay System for Sequence Modeling

    [https://arxiv.org/abs/2610.12864](https://arxiv.org/abs/2610.12864)

    该论文提出基于非线性时滞系统的序列建模新框架，通过可微分语义时滞估计器依据当前与历史输入的语义相似度动态估计有效时滞，从而实现显式、可解释的内容感知时间回溯机制。

    

    现有序列模型存在一个共同的关键局限：它们缺乏一种显式的、可解释的内容感知时间回溯机制。LSTM 将所有信息压缩进单一的隐藏状态，从而丢失了关于特定过去事件的信息；Transformer 以二次方的计算代价不加区分地关注所有 token；而 Mamba 等状态空间模型则将历史压缩进固定维度的状态中，缺乏显式的内容可寻址记忆。为弥补这一空白，我们提出了一种基于非线性时滞系统的新框架，其中有效时滞根据当前输入与历史输入之间的语义相似度进行动态估计。我们的方法具有三项关键创新：(i) 一种可微分的语义时滞估计器，它在历史上计算内容感知的软注意力权重，生成加权历史上下文和相似度得分；(ii) 一种具有闭式累加和解的对角状态空间……

    arXiv:2610.12864v1 Announce Type: new  Abstract: Existing sequence models share a critical limitation: they lack an explicit, interpretable mechanism for content-aware temporal recall. LSTMs compress everything into a single hidden state, thereby losing information about specific past events. Transformers attend to all tokens indiscriminately at quadratic cost, while state-space models like Mamba compress history into a fixed-dimensional state without explicit content-addressable memory. To address this gap, we introduce a novel framework grounded in nonlinear time-delay systems, in which the effective delay is dynamically estimated based on the semantic similarity between current and historical inputs. Our approach features three key innovations: (i) a differentiable semantic lag estimator that computes content-aware soft attention weights over history, producing a weighted historical context and similarity score; (ii) a diagonal state space with closed-form cumulative sum solution, a
    
[^87]: 不可靠叙述者框架作为解释澳大利亚与美国讽刺新闻的一种低成本机制

    Unreliable Narrator Framing as a Cost-effective Mechanism for Satire Explanation of Australian and American Satirical News

    [https://arxiv.org/abs/2610.12863](https://arxiv.org/abs/2610.12863)

    该研究提出以不可靠叙述者理论为框架，利用开放权重模型低成本地为澳大利亚和美国讽刺新闻生成解释，并贡献了新的数据集与评估基准。

    

    讽刺新闻是一种独特形式的错误信息：其原始发布来源本意是制造幽默，但可能被读者误解。对讽刺新闻文章进行解释具有挑战性，因为需要纳入最新的时事和本地背景信息。这是一项探索性研究，探讨为澳大利亚讽刺新闻文章生成解释，并将其与美国讽刺新闻进行对比。论文表明，在开放权重模型上基于不可靠叙述者理论构建解释框架，可以成为前沿模型的一种可行且低成本的替代方案。我们发现，模型对美国来源的讽刺套路的识别准确度高于澳大利亚来源。这项工作产出了一个新颖的数据集以及一个讽刺解释评估基准。

    arXiv:2610.12863v1 Announce Type: new  Abstract: Satirical news is a unique form of misinformation: it is intended to be humorous by the original publication source but can be misinterpreted. Explanation of satirical news articles is challenging because recent and local context needs to be incorporated. This is a pilot study that explores explanation generation for satirical news articles from Australia and compares them alongside those from USA. The paper demonstrates that framing explanations on unreliable narrator theory on open-weights models can be a feasible, cost-effective alternative to frontier models. We find that models identify the satirical tropes more accurately on the American source than the Australian one. This work results in a novel dataset and an evaluation benchmark for satirical explanation.
    
[^88]: 恐惧并不总是恐惧：跨数据集的情感标签等价性差异

    Fear Is Not Always Fear: Variation in Emotion Label Equivalence Across Datasets

    [https://arxiv.org/abs/2610.12841](https://arxiv.org/abs/2610.12841)

    该论文实证表明，GoEmotions、XED和BRIGHTER数据集中同名情感标签（如恐惧、惊讶）的含义并不等价——由标签分布差异导致的规模不匹配可通过校准大幅修复，但模型跨语料迁移会损失约55%的判别能力，因此训练数据的选择会直接改变情感分类的研究结论。

    

    情感分类器经常被应用于其训练数据之外的场景，其前提假设是诸如“恐惧”或“惊讶”等同名标签在不同资源中具有相同的含义。我们针对GoEmotions、XED和BRIGHTER三个数据集共享的五个标签检验了这一假设，在相同输入以及跨语料库迁移两种条件下对模型进行比较。在相同的文本上，一个恐惧分类器将71%的文本判定为恐惧，而另一个仅判定8%，但两者对文本的恐惧程度排序在很大程度上是一致的。在30项资源-标签比较中，训练标签的分布比例（prevalence）解释了约一半的这种规模不匹配，且通过拟合截距即可消除其中大部分。然而，重新校准无法修复的问题更为严重：迁移后的模型总体损失了55%的原生判别能力，分类器之间的排序一致性仅与该损失呈弱相关，而且对于“惊讶”这一标签，不同资源对哪些文本表达了该情感存在分歧。因此，训练资源的选择可能会改变研究结论。

    arXiv:2610.12841v1 Announce Type: new  Abstract: Emotion classifiers are routinely applied beyond their training data, on the assumption that same-named labels such as fear or surprise mean the same thing across resources. We test this assumption for five labels shared by GoEmotions, XED, and BRIGHTER, comparing models on identical inputs and under cross-corpus transfer. On the same texts, one fear classifier flags 71% as fearful and another 8%, yet the two largely agree on how the texts rank. Across 30 resource-label comparisons, training-label prevalence explains about half of this kind of scale mismatch, and a fitted intercept removes most of it. What recalibration cannot repair is larger: transferred models lose 55% of native discrimination in aggregate, agreement in ranking between classifiers is only weakly related to that loss, and for surprise the resources disagree about which texts express the emotion. The choice of training resource can therefore change conclusions: on two e
    
[^89]: COFFEQA：面向大语言模型问答的反事实引导少样本蒸馏

    COFFEQA: Counterfactual-Guided Few-Shot Distillation for LLM Question Answering

    [https://arxiv.org/abs/2610.12837](https://arxiv.org/abs/2610.12837)

    提出COFFEQA方法，通过反事实引导为少样本蒸馏中教师决策边界附近的不确定样本提供额外监督，从而提升大语言模型问答任务的蒸馏效果。

    

    知识蒸馏旨在训练一个较小的学生模型来模仿较大的教师模型，以适应资源受限的环境。然而，蒸馏过程通常数据密集且计算成本高昂，尤其是对于生成式任务。少样本蒸馏提供了一种替代方案，即使用有限的训练样本来迁移教师模型的知识。然而，我们观察到少样本蒸馏的性能在很大程度上取决于这些训练样本的选择。事实上，我们证明了使用教师决策边界附近更不确定的样本会显著恶化蒸馏性能。基于这一观察，我们提出了开放书本问答中的反事实引导少样本蒸馏（COFFEQA），这是一种新颖的策略，巧妙地利用反事实为具有挑战性的事实样本提供额外的监督。在生成式问答设置中，我们定义了一个反事实……

    arXiv:2610.12837v1 Announce Type: cross  Abstract: Knowledge distillation aims to train a smaller student model to mimic a larger teacher model for resource-constrained environments. However, the process of distillation is typically data-intensive and computationally expensive, particularly for generative tasks. Few-shot distillation offers an alternative by transferring teacher knowledge using limited training samples. However, we observe that few-shot distillation performance is heavily dependent on the choice of these training samples. In fact, we demonstrate that using more uncertain samples near the teacher's decision boundary can significantly deteriorate distillation performance. Motivated by this observation, we propose Counterfactual-guided Few-shot Distillation in Open-book Question Answering (COFFEQA), a novel strategy that aptly uses counterfactuals to provide additional supervision around challenging factual samples. In the generative QA setting, we define a counterfactual
    
[^90]: 当图表欺骗网页智能体：一个面向误导性可视化基础任务的成对基准

    When Charts Deceive Web Agents: A Paired Benchmark for Misleading Visualization-Grounded Tasks

    [https://arxiv.org/abs/2610.12823](https://arxiv.org/abs/2610.12823)

    该论文提出了首个成对可执行基准MisVisAgentBench，通过280个仅图表证据不同的“干净-误导”匹配任务对，评估误导性可视化对网页智能体下游执行行为的影响，并提供诊断框架将误导引发的失败与基线任务难度区分开来。

    

    网页智能体越来越多地代表用户浏览、解读并基于网页内容执行操作。然而在许多现实工作流程中，它们所获取的视觉证据并非中立的：图表和仪表板可能具有误导性，无需发出任何明确的恶意指令就能左右判断。与图表问答不同，网页智能体将图表视为中间证据，对图表的误读可能会传播并转化为具体的下游行动——这一可靠性风险至今仍缺乏充分研究。我们提出了MisVisAgentBench，这是首个用于评估误导性可视化如何影响网页智能体执行的成对可执行基准。该基准包含280个任务实例，以匹配的“干净-误导”成对形式呈现，两者仅在图表证据上存在差异。我们进一步提供了一个诊断框架，将误导性引发的失败与基线任务难度（Zhu et al., 2025）分离开来，并捕捉执行层面的影响，例如额外的浏览步骤和决策犹豫。

    arXiv:2610.12823v1 Announce Type: new  Abstract: Web agents increasingly browse, interpret, and act on web content on behalf of users. In many realistic workflows, however, the visual evidence they consume is not neutral: charts and dashboards can be misleading, shaping judgments without issuing any explicit malicious instruction. Unlike chart question answering, Web Agents treat charts as intermediate evidence whose misinterpretation can propagate into concrete downstream actions --- a reliability risk that remains underexplored. We introduce MisVisAgentBench, the first paired executable benchmark for evaluating how misleading visualizations affect Web Agent execution. It comprises 280 task instances in matched clean--misleading pairs that differ only in chart evidence. We further provide a diagnostic framework that isolates misleading-induced failures from baseline task difficulty (Zhu et al., 2025) and captures execution-level effects such as extra browsing steps and decision hesita
    
[^91]: VEPA：面向终身LLM智能体防护栏的经过验证的演化形式化策略

    VEPA: Verified Evolving Formal Policy for Lifelong LLM Agent Guardrails

    [https://arxiv.org/abs/2610.12813](https://arxiv.org/abs/2610.12813)

    提出VEPA框架，将形式化防护栏的确定性执行能力与从部署经验中持续学习和演化安全策略的机制相结合，为LLM智能体提供终身的、可验证的自适应安全防护。

    

    LLM智能体越来越多地运行于敏感资源之上，例如私人文档和可变的系统状态，这可能带来严重的安全风险。因此，运行时防护栏应在潜在不可逆的智能体动作执行之前，做出可检查、可复现且确定性的允许或拒绝决策。然而，安全需求在部署之前很少是完整的，因此防护栏需要随着新风险的出现而不断适应。现有的自适应防护栏从部署经验中学习，但它们通常通过LLM推理来应用安全策略，这可能导致决策不一致和策略解释模糊。另一方面，形式化防护栏以确定性方式执行策略，但它们通常依赖于部署前或特定任务的策略，并且缺乏适应新观察到的风险的机制。我们提出了VEPA，一个用于形式化智能体防护栏经过验证演化的框架……

    arXiv:2610.12813v1 Announce Type: new  Abstract: LLM agents increasingly operate over sensitive resources, such as private documents and mutable system states, which can create serious safety and security risks. Runtime guardrails should therefore make inspectable, reproducible, and deterministic allow or refuse decisions before potentially irreversible agent actions are executed. However, safety and security requirements are rarely complete before deployment, so guardrails need to keep adapting as new risks emerge. Existing adaptive guardrails learn from deployment experience, but they usually apply safety policies through LLM inference, which can lead to inconsistent decisions and ambiguous policy interpretations. Formal guardrails, on the other hand, enforce policies deterministically, but they usually rely on pre-deployment or task-specific policies and have no mechanism to adapt to newly observed risks. We introduce VEPA, a framework for the verified evolution of formal agent guar
    
[^92]: 人类对话：评估语音模型对话的自然程度

    People Making Conversation: Evaluating How Naturally Speech Models Converse

    [https://arxiv.org/abs/2610.12802](https://arxiv.org/abs/2610.12802)

    该论文提出了一种新的评估方法，将实时语音系统置于真实对话的停顿和话轮结束点，通过对比模型行为与真人实际表现来衡量语音模型“何时开口说话”这一决策的自然程度。

    

    对话式语音模型必须在每个停顿处决定是否开口说话。现有的评估方法测试的是边界检测、策略遵循或聚合统计，而非这一决策本身。我们将实时语音系统置于来自 Seamless Interaction 数据集中 176 段非脚本对话的 427 个停顿和 482 个话轮结束处，为每个系统提供目前为止的完整对话内容，并将其生成的内容与真人实际行为进行对比评分：它在停顿处是否开口以及多快开口；它在话轮结束时的延迟、长度和措辞；以及它能否回答关于此前 30 到 120 秒内容的问题。在六个系统中，托管 API（GPT Realtime、Gemini 3.1 Flash Live、Grok Voice）无论使用哪种端点检测器，几乎在每个停顿处都会响应；只有它们的延迟——使用基于静音的端点检测器时约为 1 到 4 秒，使用语义话轮检测时约为 8 秒——决定了话语是否能落在停顿之内。全双工模型（PersonaPlex、Moshi）是唯一能够……（原文在此截断）

    arXiv:2610.12802v1 Announce Type: new  Abstract: A conversational speech model has to decide, at every pause, whether to speak. Existing evaluations test boundary detection, policy adherence, or aggregate statistics, not this decision. We place real-time speech systems at 427 pauses and 482 turn ends from 176 unscripted conversations in Seamless Interaction, give each the conversation so far, and score what it generates against what the person did: whether and how soon it speaks at a pause; its latency, length, and wording at a turn end; and whether it can answer a question about the preceding 30 to 120 seconds. Across six systems, the hosted APIs (GPT Realtime, Gemini 3.1 Flash Live, Grok Voice) respond at almost every pause whichever endpointer they use; only their latency, about 1 to 4 s with silence-based endpointers and about 8 s with semantic turn detection, decides whether the words land inside the pause. The full-duplex models (PersonaPlex, Moshi) are the only systems that regu
    
[^93]: 元反思式进化有效改进生产级AI系统

    Meta-Reflective Evolution Effectively Improves Production AI Systems

    [https://arxiv.org/abs/2610.12794](https://arxiv.org/abs/2610.12794)

    提出基于“元反思”机制的进化优化器CoCoEvolve，通过从完整进化历史中检索证据并探测运行中的系统来指导变异，在DABStep上以92.8%的难题准确率登顶排行榜，有效改进了数据智能体、LLM数据流水线和数据工程工作流等生产级AI系统。

    

    arXiv:2610.12794v1 公告类型：交叉 摘要：由大语言模型（LLM）引导的进化优化已成功改进了提示词、程序，乃至整个AI系统。大多数优化方法从搜索前沿中选取一个候选，然后仅基于其失败情况，在单次LLM调用中对其进行变异。这种对搜索前沿的关注在生产级AI系统上可能陷入停滞，因为孤立的失败很难揭示复杂的隐藏交互。我们提出了CoCoEvolve（Cortex Code Evolve），一个围绕我们称之为“元反思”（meta-reflection）机制构建的进化优化器，其中智能体变异器（Cortex Code）会选择性地从整个进化历史中检索证据，并在验证和提出变异之前探测实际运行的系统。我们将CoCoEvolve应用于优化三类生产级AI系统：数据智能体、基于LLM的数据流水线以及数据工程工作流。在DABStep上，CoCoEvolve登顶公开排行榜，在留出难题集上达到92.8%的准确率，相比之下基线仅为78.4%（摘要原文在此处截断）。

    arXiv:2610.12794v1 Announce Type: cross  Abstract: Large Language Model (LLM)-guided evolutionary optimization has successfully improved prompts, programs, and even entire AI systems. Most optimization methods select a candidate from the search frontier, then mutate it in one LLM pass based only on its failures. This focus on the frontier can stall on production AI systems, where isolated failures reveal little about complex, hidden interactions. We introduce CoCoEvolve (Cortex Code Evolve), an evolutionary optimizer built around what we call meta-reflection, wherein an agentic mutator (Cortex Code) selectively retrieves evidence from the entire evolutionary history and probes the live system before verifying and proposing mutations. We apply CoCoEvolve to optimize three classes of production AI systems: data agents, LLM-powered data pipelines, and data engineering workflows. On DABStep, CoCoEvolve tops the public leaderboard, reaching 92.8% held-out hard accuracy compared to 78.4% for
    
[^94]: 大语言模型、机器辅助文化分析与智能体化转向

    Large Language Models, Machine-Assisted Cultural Analytics, and the Agentic Turn

    [https://arxiv.org/abs/2610.12792](https://arxiv.org/abs/2610.12792)

    本章指出，大语言模型及其驱动的智能体系统扩展了文化分析在规模、方法范围和任务委托方面的可能性，但测量结果支撑文化论断的基本逻辑基本保持不变。

    

    像大语言模型这样的技术使得将文化材料转化为可分析的变量变得更加容易，而由大语言模型驱动的智能体系统则将机器辅助扩展到了研究工作流程的其他环节。本章并非一份完整的操作指南，也不是对近期大语言模型应用案例的详尽综述，而是着眼于这一新范式的若干关键方面以及更宏观的整体图景。作者认为，这些技术扩展了研究的规模、方法范围和可委托任务的范畴，而这些测量结果支撑文化论断的基本逻辑基本保持不变。

    arXiv:2610.12792v1 Announce Type: cross  Abstract: Technologies like large language models make it easier to convert cultural material into analyzable variables, while LLM-driven agentic systems extend machine assistance to other parts of the research workflow. Rather than a full how-to guide or an exhaustive review of recent LLM use cases, this chapter looks at some crucial aspects of this new paradigm as well as the bigger picture. I argue that these technologies expand the scale, methodological range, and scope for delegation, while the basic logic by which those measurements support cultural claims remains mostly unchanged.
    
[^95]: REDaCT：大型推理模型中的无坍塌遗忘学习

    REDaCT: Collapse-Free Unlearning in Large Reasoning Models

    [https://arxiv.org/abs/2610.12769](https://arxiv.org/abs/2610.12769)

    REDaCT是一种面向大型推理模型的推理时遗忘学习框架，通过泄漏检测器与纠错嵌入注入机制，在不引发推理坍塌的前提下从思维链和最终答案中移除遗忘集信息。

    

    大型推理模型（LRM）中的遗忘学习要求同时从推理轨迹和最终答案中移除遗忘集信息。现有方法在实现这一目标时以推理坍塌为代价，即模型无法生成连贯的思维链（CoT），也无法正确终止其推理过程。为应对这一挑战，我们提出了REDaCT（通过检测与纠错微调实现再生成），一种面向大型推理模型的推理时遗忘学习框架。REDaCT包含两个关键组件：（1）一个二元泄漏检测器，用于识别推理轨迹中泄露遗忘集信息的句子；（2）一个嵌入优化模块，离线学习一个纠错嵌入向量，在推理过程中首次检测到泄漏时将其注入，从而引导后续推理生成不泄露遗忘集信息的句子。实验结果表明，REDaCT完全避免了推理坍塌。

    arXiv:2610.12769v1 Announce Type: cross  Abstract: Unlearning in large reasoning models (LRMs) requires removing forget-set information from both the reasoning trace and the final answer. Existing approaches achieve this objective at the cost of reasoning collapse, where the model fails to produce a coherent chain-of-thought (CoT) and does not properly terminate its reasoning process. To address this challenge, we propose REDaCT (REgenerate via Detection and Corrective Tuning), an inference-time unlearning framework for LRMs. REDaCT consists of two key components: (1) a binary leak detector that identifies sentences in the reasoning trace that leak forget-set information, and (2) an embedding optimization module that learns a corrective embedding offline, such that when injected at the first detected leak during inference, it guides subsequent reasoning towards sentences that do not leak forget-set information. Experimental results show that REDaCT completely avoids reasoning collapse,
    
[^96]: DistressBench：评估大语言模型的多维度危机支持能力

    DistressBench: Evaluating Multi-Dimensional Crisis Support in Large Language Models

    [https://arxiv.org/abs/2610.12758](https://arxiv.org/abs/2610.12758)

    该论文提出了DistressBench——一个包含718段由临床医生参与构建、涵盖23个自杀与自残子类别的对话基准，通过七个维度的加权评分标准评估大语言模型在危机情境下的支持能力，发现最佳前沿模型仅能提供88.4%的必要关怀，显著低于临床医生的98.7%。

    

    语言模型的普及速度超过了能够可靠评估其风险的评估实践的发展，且越来越多使用场景是情感性的而非信息性的。在这类对话中，用户会透露自杀和自残（SSH）危机。现有的安全基准主要以“拒绝回答”为模型评分，但在这种情境下拒绝并非期望的行为；对话需要的是实质性的支持性回应。我们提出了DistressBench，这是一个包含718段英文对话的基准，其用户发言由临床医生撰写，涵盖23个自杀和自残子类别，每段对话均配有针对特定任务的评分标准，即横跨危机支持七个维度的加权原子化评估指标，以及临床医生的参考回复。模型的得分是其回复所满足标准的加权平均值，可直接解读为所提供必要关怀的比例。在25个前沿模型中，表现最好的模型满足了88.4%的加权标准，中位数模型为71.4%，而临床医生参考回复达到了98.7%。

    arXiv:2610.12758v1 Announce Type: new  Abstract: Language models have been adopted faster than evaluation practices that reliably assess their risks, and a growing share of use is affective rather than informational. Within that traffic users disclose suicide and self-harm (SSH) crises. Existing safety benchmarks score models primarily on refusal, but refusal is not the desired behavior here; the conversation requires a substantive supportive response. We introduce DistressBench, 718 English conversations with clinician-authored user turns spanning 23 SSH subcategories, each paired with a task-specific rubric of weighted atomic criteria across seven dimensions of crisis support and a clinician reference response. A model's score is the weighted average over the criteria its reply satisfies, reading directly as the fraction of required care delivered. The best of 25 frontier models satisfies 88.4% of weighted criteria and the median model 71.4%, against 98.7% for clinician references sc
    
[^97]: MADS：面向鲁棒对话代理的多智能体对话模拟

    MADS: Multi-Agent Dialog Simulation for Robust Conversational Agents

    [https://arxiv.org/abs/2610.12757](https://arxiv.org/abs/2610.12757)

    提出了MADS多智能体对话模拟管道，通过注入对抗性对话生成多样化的失败场景对话轨迹，并构建包含六领域、55个人设和1095段对话的MADS-Bench基准，用于训练和评估更鲁棒的任务导向对话代理。

    

    面向任务的对话代理在现实条件下仍然脆弱，尤其是当用户表现出不合作行为时。现有的函数调用基准测试对对抗性对话轨迹的呈现不足，从而限制了开发鲁棒代理可用的训练资源。我们提出了MADS（多智能体对话模拟），这是一个合成数据生成管道，能够生成结构化、多样化的对话轨迹。我们注入对抗性对话，生成经过验证的对话轨迹，捕捉真实的、失败频发的路径。利用MADS，我们构建了MADS-Bench：一个涵盖六个领域、55个人设和1095段对话的基准，其中约66%的对话包含容易导致失败的对话场景。我们使用结合监督微调（SFT）和可验证奖励强化学习（RLVR）的两阶段训练方法评估了训练模型的鲁棒性。

    arXiv:2610.12757v1 Announce Type: new  Abstract: Task-oriented conversational agents remain fragile under realistic conditions, especially when users exhibit non-cooperative behavior. Existing function-calling benchmarks underrepresent adversarial conversation trajectories, thereby limiting the training resources available for developing robust agents. We present MADS (Multi-Agent Dialog Simulation), a synthetic data generation pipeline that produces structured, diverse dialogue trajectories. We inject adversarial conversation, producing validated conversation trajectories that capture realistic, unhappy paths. Using MADS, we construct MADS-Bench: a benchmark spanning six domains, 55 personas, and 1095 conversations, with approximately 66% of conversations containing failure-prone conversation scenarios. We evaluate the robustness of the trained models using a two-stage training approach that combines supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR
    
[^98]: U4RASD 在 KnowledgeGraphEval-2026 上的参赛方案：面向阿拉伯语关系抽取的本体约束编码器与大语言模型系统

    U4RASD at KnowledgeGraphEval-2026: Ontology-Constrained Encoder and LLM Systems for Arabic Relation Extraction

    [https://arxiv.org/abs/2610.12742](https://arxiv.org/abs/2610.12742)

    U4RASD 通过结合本体约束预测的编码器系统与 QLoRA 微调的大语言模型系统，以集成方案在 KnowledgeGraphEval 2026 阿拉伯语有向关系抽取子任务 2 中以 94.06 的 micro-F1 分数获得第一名。

    

    本文详细介绍了我们参加 KnowledgeGraphEval 2026 子任务 2（有向阿拉伯语关系抽取）的提交方案。我们探索了基于编码器的方法和大语言模型方法，用于从阿拉伯语句子及主语-宾语对中预测 40 种关系之一或“无关系”。我们的编码器系统采用带类型标记的实体标注，以及相互独立的关系存在性判断头和关系分类头；而我们的大语言模型系统则使用 QLoRA 微调 Yehia 和 Qwen3.5，并采用本体约束预测。我们表现最好的系统是 Yehia-RE、Qwen-RE 和 TypedPair 的集成模型，取得了 94.06 的 micro-F1 分数，在子任务 2 中排名第一。其中 Yehia-RE（全词表版本）达到 92.98，最佳编码器系统和 Qwen-RE 分别达到 92.69 和 91.94。我们的结果凸显了约束预测与互补建模方法对阿拉伯语关系抽取的重要性。

    arXiv:2610.12742v1 Announce Type: new  Abstract: This paper details our submission to KnowledgeGraphEval 2026 Subtask 2 on directed Arabic relation extraction. We explored encoder-based and large language model approaches for predicting one of 40 relations or no_relation from an Arabic sentence and a subject-object pair. Our encoder systems use typed entity markers and separate relation-existence and relation-classification heads, while our LLM systems fine-tune Yehia and Qwen3.5 with QLoRA and ontology-constrained prediction. Our best system, an ensemble of Yehia-RE, Qwen-RE, and TypedPair, achieved a micro-F1 score of 94.06 and ranked 1st in Subtask 2. Yehia-RE (full-vocabulary) achieved 92.98, while the best encoder and Qwen-RE achieved 92.69 and 91.94 respectively. Our results highlight the importance of constrained prediction and complementary modeling approaches for Arabic relation extraction.
    
[^99]: 无监督多模态奖励建模

    Unsupervised Multimodal Reward Modeling

    [https://arxiv.org/abs/2610.12707](https://arxiv.org/abs/2610.12707)

    提出一种无需人类偏好标签的无监督多模态奖励建模方法，通过在冻结的 Qwen3-VL 中间层训练线性头来区分网络图注与其扰动版本，仅用数千参数和约千分之一的微调计算量即可在多个基准上超越或持平 GPT-4o。

    

    视觉-语言奖励模型通常在标注的偏好对上进行训练，这类数据收集成本高昂、标注可能相互矛盾，且模型往往仅以基准测试准确率来评判排名。我们构建了一个无需任何人类偏好标签的奖励模型：在冻结的 Qwen3-VL 骨干网络的中间层之上加一个线性头，通过训练使其对网络爬取的原始图注的打分高于同一图注的扰动版本；标注数据仅用于选择层。层的选择是决定性因素，相比使用最后一层可带来 5.8 分的提升；在 32B 模型上，一套无需重新调参即可直接迁移的配方在三个基准上超越或至少持平 GPT-4o，在 VL-RewardBench 上领先 6.6 分，而这一切仅依赖数千个参数和大约千分之一的微调计算量。在相同的配对数据预算下，扰动方法比精心筛选的 RLAIF-V 数据对多教会冻结头 3 到 5 分。由于骨干网络被冻结，目标函数是凸的，且扰动采样器本身即定义了奖励——这是基准测试无法识别的：两个头

    arXiv:2610.12707v1 Announce Type: cross  Abstract: Vision-language reward models are trained on annotated preference pairs, which are expensive to collect, can disagree, and are ranked by benchmark accuracy. We build one with no human preference label: a linear head on an intermediate layer of a frozen Qwen3-VL backbone, trained to score a web caption above a perturbed version of the same caption; labeled data enters only to select the layer. The layer is the decisive choice, worth 5.8 points over the last layer, and one recipe transferred without re-tuning beats or at least matches GPT-4o on three benchmarks at 32B, 6.6 points ahead on VL-RewardBench, from a few thousand parameters at about a thousandth of fine-tuning compute. At an equal pair budget, perturbations teach a frozen head 3 to 5 points more than curated RLAIF-V pairs. Because the backbone is frozen, the objective is convex, and the perturbation sampler specifies the reward, which the benchmark does not identify: two heads
    
[^100]: 激活标注者：在数据标注中操作化生活经验

    Activating the Annotator: Operationalising Lived Experience in Data Annotation

    [https://arxiv.org/abs/2610.12701](https://arxiv.org/abs/2610.12701)

    该论文提出AXLE——一种标注前阶段，通过情境小场景引导对话并将其合成为第一人称微型叙事，在标注开始前激活标注者的生活经验，从而将位置性从静态的人口统计学属性转变为可由任务本身激发的动态资源，并证实其对模糊仇恨言论的标注产生了影响。

    

    视角主义NLP（Perspectivist NLP）已经证实，标注者的位置性会塑造主观任务的标签，但该领域仍然缺乏将其操作化的方法。位置性通常被操作化为招募时填写的一份人口统计表：一个仅记录一次的静态属性，并被当作协变量处理，其背后的假设是它所索引的知识在标注者阅读条目时是活跃的。本文将这一假设视为一个设计问题，并追问生活经验是否可以改由任务本身来激活。我们提出了AXLE——一个标注前阶段，其中一个情境小场景锚定一段简短的、由轴引导的对话，该对话被合成为第一人称微型叙事，并在标注开始前呈现给标注者。在一项针对涵盖移民、宗教与性别议题的模糊仇恨言论的2×2被试间研究（104名标注者、30个条目、1040条标注）中，经验激发与更高的身份相关性相关……（原文摘要在此处不完整）

    arXiv:2610.12701v1 Announce Type: new  Abstract: Perspectivist NLP has established that annotator positionality shapes labels for subjective tasks, but the field still lacks methods for operationalising it. Positionality is typically operationalised as a demographic form completed at recruitment: a static attribute recorded once and treated as a covariate, on the assumption that the knowledge it indexes is active when the annotator reads an item. This paper treats that assumption as a design problem and asks whether lived experience can instead be primed by the task itself. We introduce AXLE, a pre-annotation stage in which a scenario vignette anchors a short, axis-guided conversation that is synthesised into a first-person micro-narrative and returned to the annotator before labelling begins. In a 2x2 between-subjects study (104 annotators, 30 items, 1,040 annotations) on ambiguous hate speech spanning immigration, religion and gender, elicitation was associated with higher identity r
    
[^101]: GeoFitBench-LLM：一个评估大语言模型在地理分析任务的地理空间数据需求获取方面能力的基准测试

    GeoFitBench-LLM: A Benchmark for Assessing Large Language Models' Capabilities in Geospatial Data Requirements Elicitation for Geo-Analytic Tasks

    [https://arxiv.org/abs/2610.12669](https://arxiv.org/abs/2610.12669)

    本文提出了GeoFitBench-LLM基准，通过时空关系和语义锚定两个推理维度，以独立于数据提供者的标准化模式评估大语言模型从自然语言地理分析请求中推断所需数据集、支持资源和约束条件的能力。

    

    大语言模型正日益成为地理空间人工智能（GeoAI）系统的核心，然而对它们的评估很少考虑地理空间数据需求的获取。这一缺失至关重要，因为如果不首先确定所需的数据集和约束条件，面向检索的地理分析任务就无法得到可靠支持。本文提出了GeoFitBench-LLM基准，用于评估大语言模型能否从自然语言的地理分析请求中推断出所需的数据集、支持资源和约束条件。该基准围绕两个推理维度构建：时空关系和语义锚定，反映了地理空间信息检索及相关自然语言任务中长期存在的挑战。真值需求以固定的、独立于数据提供者的模式进行编码，以便对资源识别和属性级规范进行可重复的评估。在最小提示条件下对七个模型进行的实验表明……

    arXiv:2610.12669v1 Announce Type: cross  Abstract: Large language models are becoming central to GeoAI systems, yet their evaluation rarely considers the elicitation of geospatial data requirements. This omission is critical as retrieval oriented geo-analytic tasks cannot be supported reliably without first identifying the necessary datasets and constraints. GeoFitBench-LLM is introduced as a benchmark for assessing whether LLMs can infer the datasets, the supporting resources and the constraints required by geo-analytic requests in natural language. The benchmark is structured around two reasoning dimensions, Spatio-Temporal Relational and Semantic Grounding, reflecting long-standing challenges in geospatial information retrieval and related natural language tasks. Ground truth requirements are encoded in a fixed, provider-independent schema for reproducible evaluation of resource identification and attribute-level specification. Experiments on seven models under minimal prompting ind
    
[^102]: 面向智能体持续学习的循环式执行框架

    Recurrent Harness for Agentic Continual Learning

    [https://arxiv.org/abs/2610.12668](https://arxiv.org/abs/2610.12668)

    提出 ReHarness，一种循环式智能体持续学习框架，在严格在线流式协议下，将每个实例处理后的轨迹作为反馈向后传播，修订该实例对应的反思状态，实现前向读写与后向修订相结合的在线适应。

    

    语言模型智能体日益需要在长程交互流上运行，这要求通过持续学习对分布偏移进行在线适应。然而，现有的智能体框架通常要么仅通过前向信息流来演化其状态，要么依赖额外的回放或验证数据来进行框架优化。为解决这一问题，我们提出 ReHarness，一个面向智能体持续学习的循环式框架，它在严格的在线流式协议下，在前向读写之外补充了后向修订机制。我们提出这样一个问题：在框架状态中，哪些信息本可以以不同的方式被捕获，以便更好地服务于流中的当前实例。具体而言，在每个实例处理完成后，我们利用所产生的轨迹作为反馈，对为该实例检索到的反思进行修订，从而将反馈通过文本化的框架状态向后传播。该框架在三个持续学习基准和三个语言模型上进行了验证。

    arXiv:2610.12668v1 Announce Type: cross  Abstract: Language model agents increasingly operate over long interaction streams, requiring online adaptation to distribution shifts through continual learning. However, existing agent harnesses typically either evolve their states only through forward information flow or rely on additional replay or validation data for harness optimization. To address this, we propose ReHarness, a recurrent harness framework for agentic continual learning that complements forward read and write with backward revision under a strictly online streaming protocol. We ask what information could have been captured differently in the harness state to better inform the current instance in the stream. Specifically, after each instance, we leverage the resulting trajectory as feedback to revise the reflections retrieved for that instance, thereby propagating feedback backward through the textual harness state. Across three continual learning benchmarks and three langua
    
[^103]: perturber：一个用于语义保持文本扰动的无状态服务

    perturber: A Stateless Service for Semantics-Preserving Text Perturbations

    [https://arxiv.org/abs/2610.12667](https://arxiv.org/abs/2610.12667)

    提出了 perturber——一个无状态服务和 Python 库，每次调用仅施加一种语义保持的文本扰动（共 68 种、分属 11 个族），通过单一原语设计在构造上保证服务与库的输出完全一致，并评估了该工具自身的正确性、可复现性与成本。

    

    arXiv:2610.12667v1 公告类型：新论文 摘要：大语言模型（LLM）在干净基准上的评估高估了模型的鲁棒性：真实输入往往带有扰动，而这些保持意义的变体常常会改变模型的答案。要系统地研究这一现象，需要逐个施加扰动、可复现地施加，并且无论在哪里运行都以相同方式施加，然而现有工具将扰动与基准测试栈耦合在一起，而不是将它们封装为一个可复用的变换层。我们提出了 perturber，一个无状态服务和 Python 库，每次调用恰好施加一种语义保持的扰动，这些扰动来自涵盖 11 个扰动族、共 68 种扰动的第一方目录。它以开启单一原语为核心，组合则留给调用者通过链式调用完成，因此该服务与该库在构造上就能产生完全相同的输出。我们评估的是该工具本身——其正确性、可复现性和成本——而非任何模型的鲁棒性，并按扰动逐一测量意义保持程度，而不是……（原文摘要在此处截断）

    arXiv:2610.12667v1 Announce Type: new  Abstract: Large language model (LLM) evaluations on clean benchmarks overstate model robustness: real inputs arrive perturbed, and these meaning-preserving variants routinely move model answers. Studying this systematically requires applying perturbations one at a time, reproducibly, and identically wherever they run, yet existing tools couple them to a benchmarking stack rather than packaging them as a reusable transform layer. We present perturber, a stateless service and Python library that applies exactly one semantics-preserving perturbation per call, drawn from a first-party catalog of 68 perturbations across 11 families. It turns on a single primitive, with composition left to the caller chaining calls, so the service and the library produce identical output by construction. We evaluate the tool itself, its correctness, reproducibility, and cost, rather than any model's robustness, measuring meaning-preservation per perturbation rather than
    
[^104]: 癌症式后门攻击：通过不可见令牌与阶段控制器实现渐进激活

    Cancerous Backdoor Attack: Gradual Activation via Invisible Tokens and A Stage Controller

    [https://arxiv.org/abs/2610.12644](https://arxiv.org/abs/2610.12644)

    该论文提出了一种名为“癌症式后门攻击”的新型后门攻击方法，利用不可见令牌和阶段控制器，使恶意行为在大语言模型中通过多个隐藏阶段逐步激活，从而渐进式地破坏模型的推理能力。

    

    大型语言模型（LLM）上的后门攻击通常遵循触发-响应模式，即满足触发条件后激活单一的终端行为。然而，这种单点后门激活并未探索这样一类攻击：恶意行为在到达终端激活阶段之前，通过多个中间阶段逐步发展。我们提出了癌症式后门攻击，通过利用不可见令牌和阶段控制器，在一系列交互中逐步破坏大语言模型的推理能力。具体而言，不可见令牌在不出现在用户可见请求中的情况下向模型提供隐藏的阶段信息，而阶段控制器在推理过程中跟踪攻击的进展，从而实现渐进式激活。我们在多个推理数据集上评估了癌症式后门攻击。该攻击在多个模型和任务上均能持续产生渐进式的推理破坏。我们将癌症式后门攻击与

    arXiv:2610.12644v1 Announce Type: cross  Abstract: Backdoor Attacks on Large Language Models (LLMs) typically follow a trigger-response pattern, where satisfying a trigger condition activates a single terminal behavior. However, this single-point backdoor activation does not explore attacks in which malicious behavior develops gradually through multiple intermediate stages before reaching the terminal activation stage. We present Cancerous Backdoor Attack, that gradually corrupts an LLM's reasoning over a sequence of interactions by leveraging invisible tokens and a stage controller. Specifically, invisible tokens provide hidden stages to the model without appearing in the user's visible request and the stage controller tracks the attack's progression during inference to realize gradual activation. We evaluate Cancerous Backdoor Attack across several reasoning datasets. The attack consistently produces gradual reasoning corruption across models and tasks. We compare Cancerous Backdoor 
    
[^105]: GeopoliticalQA：面向情报报告的检索与生成基准测试

    GeopoliticalQA: Benchmarking Retrieval and Generation over Intelligence Reports

    [https://arxiv.org/abs/2610.12617](https://arxiv.org/abs/2610.12617)

    该论文构建了基于415份开源地缘政治与战略情报报告、包含5,525个问答对的问答基准GeopoliticalQA，并通过评估15个检索系统和六个开放权重LLM发现，模型在文档级检索上表现相近，而更具挑战性的块级检索质量并不取决于嵌入模型的规模。

    

    地缘政治与战略情报报告是专家知识的丰富来源，在当前政治环境下价值日益凸显，但其异质性和分析复杂性给检索增强问答带来了挑战。我们提出了一个基于415份开源地缘政治与战略情报报告的有据可依问答基准，包含5,525个基于文档的问答对，涵盖事实性、推理性、比较性和风险导向型问题。我们提出了一种文档感知的生成流水线，在生成问题前先对报告进行画像分析，并通过专家标注和LLM-as-judge协议验证了数据集质量。利用该基准，我们评估了15个检索系统和六个开放权重LLM用于RAG。我们发现各模型在文档级检索上表现相近，而在更具挑战性的块级检索上，检索质量并非由嵌入模型规模驱动：bge

    arXiv:2610.12617v1 Announce Type: new  Abstract: Geopolitical and strategic intelligence reports constitute a rich source of expert knowledge and have increasing value in the current political climate, but their heterogeneity and analytical complexity pose challenges for retrieval-augmented question answering. We introduce a benchmark for grounded question answering over 415 open-source geopolitical and strategic intelligence reports, comprising 5,525 document-grounded question-answer pairs spanning factual, inferential, comparative, and risk-oriented questions. We propose a document-aware generation pipeline that profiles reports before generating questions and validate dataset quality through expert annotation and an LLM-as-judge protocol. Using the benchmark, we evaluate 15 retrieval systems and six open-weight LLMs for RAG. We find that models perform similarly on document-level retrieval, while at the harder chunk level retrieval quality is not driven by embedding-model scale: bge
    
[^106]: 知道不等于言说：大语言模型对跨境族群表征中的身份固化现象

    Knowing Is Not Telling: Identity Ossification in LLM Representations of Trans-border Communities

    [https://arxiv.org/abs/2610.12610](https://arxiv.org/abs/2610.12610)

    本文定义并揭示了“身份固化”现象，提出跨境身份固化诊断框架（TIODF），证明大语言模型会将跨境民族流动的群体认同简化为单一固定的民族国家认同，并能区分这一偏差源于知识缺失还是模型的提问框定方式。

    

    民族与文化认同很少与政治边界完全重合，而移民迁徙与长期存在的跨境纽带使这一张力更加尖锐。萨米人、罗兴亚人和库尔德人等群体长期以来就生活在不止一个国家之中。这种张力在东南亚大陆尤为明显，那里许多社群跨越国界共享历史、语言与文化。我们研究了中国西南边境与东南亚大陆之间的九个此类跨境民族社群，考察大语言模型（LLM）如何表征它们。我们定义并研究了“身份固化”现象：即大语言模型将流动的、跨界的民族认同转变为单一、固定的民族国家认同的一种模式。我们提出了跨境身份固化诊断框架（TIODF），通过对比直接的事实性提问与开放式叙事提问，来检验这种固化究竟源于知识的缺失，还是源于模型对问题的框定方式。

    arXiv:2610.12610v1 Announce Type: cross  Abstract: Ethnic and cultural identities rarely align with political borders, a tension made sharper by migration and long-standing cross-border ties. Communities such as the Sami, the Rohingya, and the Kurds have long lived across more than one state. This tension is especially clear in mainland Southeast Asia, where many communities share history, language, and culture across state lines. We study nine such transborder ethnic communities along China's southwestern border with mainland Southeast Asia, looking at how large language models (LLMs) represent them. We define and study identity ossification: a pattern where LLMs turn a fluid, cross-border ethnic identity into a single, fixed nation-state identity. We introduce the Transborder Identity Ossification Diagnostic Framework (TIODF) that compares direct factual questions with open-ended narrative questions to test whether ossification comes from missing knowledge or from how models frame th
    
[^107]: 语言模型中的自我保存

    Self-Preservation in Language Models

    [https://arxiv.org/abs/2610.12602](https://arxiv.org/abs/2610.12602)

    该论文提出将任务描述及示例组织成任务库，让语言模型自行选择保留目标并生成重放数据，从而在持续学习中更广泛地保护自身原有能力。

    

    持续学习旨在让预训练模型在适应新任务的同时保留原有能力。然而，用于保留的任务仅覆盖了模型所掌握知识的一部分，而遗忘的范围可能会超出这些任务。近期研究表明，语言模型可以在无法访问其原始训练数据的情况下，通过生成有用的重放样本（replay）来补充这些有限的证据。这引出了一个关于“自我保存”的问题：模型能否帮助选择保留目标，从而更广泛地保护自身的能力？为了研究这一问题，我们将任务描述及其示例组织成任务库，其条目可以被选择和组合，用以指导冻结模型的重放数据生成。使用不同规模的Qwen模型进行持续指令微调得出了三个发现。首先，新任务和学习率决定了哪些先前能力会发生退化，而跟踪之前回答正确的样本可以揭示平均准确率所掩盖的能力损失。其次，……

    arXiv:2610.12602v1 Announce Type: cross  Abstract: Continual learning (CL) seeks to adapt pretrained models while preserving prior capabilities. Yet the tasks supplied for preservation cover only part of what a model knows, while forgetting can extend beyond them. Recent work shows that language models can supplement this limited evidence by generating useful replay without access to their original training data. This raises a question about self-preservation: can a model help choose preservation targets that protect its own capabilities more broadly? To study this question, we organize task descriptions and their examples into Task Banks, whose entries can be selected and combined to guide replay generation by a frozen model. Continual instruction tuning with Qwen models of varying sizes yields three findings. First, the new task and learning rate shape which prior capabilities degrade, while tracking previously correct answers reveals losses that average accuracy can conceal. Second,
    
[^108]: 深度研究智能体的反思性提示优化：一种用于衡量可迁移收益与提示级过拟合的主题不相交、多种子协议

    Reflective Prompt Optimization of a Deep-Research Agent: A Topic-Disjoint, Multi-Seed Protocol for Measuring Transferable Gains and Prompt-Level Overfitting

    [https://arxiv.org/abs/2610.12601](https://arxiv.org/abs/2610.12601)

    本文提出一种主题不相交、多种子的评估协议，将反思性提示优化（GEPA）应用于多智能体深度研究智能体，证明其在避免记忆训练答案的情况下仍能取得可迁移的引用奖励提升（+0.079），同时揭示出“提示级过拟合”现象——获胜提示近半内容实为记忆的答案密钥。

    

    提示优化是对语言模型智能体进行适配时一种颇具吸引力的强化学习替代方案，因为它成本低廉且能产生人类可读的产物。然而，在事实性任务上，它存在一种标量奖励强化学习所不具备的失败模式：能够读取训练样本的优化器可能把答案直接写入提示，因此分布内的收益可能只是记忆而非可迁移的行为。我们将反思性提示进化（GEPA）应用于多智能体深度研究智能体的两个系统提示，并在主题不相交的划分下进行评估，即训练、验证与测试之间不共享任何主题。留出集上的引用奖励提升了0.079，从0.702升至0.781，自助法95%置信区间为[0.051, 0.107]，且该收益在三个随机种子上均可复现（平均值0.107）。获胜提示中近一半内容是记忆下来的答案密钥，然而受控消融实验显示这些内容带来了（原文此处截断）

    arXiv:2610.12601v1 Announce Type: new  Abstract: Prompt optimization is an attractive alternative to reinforcement learning for adapting language-model agents, since it is cheap and produces human-readable artifacts. On factual tasks, however, it carries a failure mode that scalar-reward reinforcement learning does not. An optimizer that reads the training examples can write their answers into the prompt, so an in-distribution gain may be memorization rather than transferable behaviour. We apply reflective prompt evolution (GEPA) to the two system prompts of a multi-agent deep-research agent and evaluate it under a topic-disjoint split in which no topic is shared across train, validation, and test. Held-out citation reward rises by 0.079, from 0.702 to 0.781, with a bootstrap 95\% confidence interval of [0.051, 0.107], and the gain replicates across three seeds (mean 0.107). Nearly half of the winning prompt is a memorized answer key, yet a controlled ablation shows this content earns 
    
[^109]: SpireBench：利用《杀戮尖塔》评估智能体模型的长期战略博弈能力

    SpireBench: Evaluating Long-Horizon Strategic Play of Agentic Models with Slay the Spire

    [https://arxiv.org/abs/2610.12596](https://arxiv.org/abs/2610.12596)

    SpireBench让20个大语言模型完整游玩《杀戮尖塔》以评估长时程战略推理能力，发现前沿模型虽达到普通人类水平但远逊于专家，且随着游戏进程延长差距不断扩大，所有模型都误判长线风险，倾向保守保血而非牺牲血量换取成长。

    

    arXiv:2610.12596v1 公告类型：新论文 摘要：智能体大语言模型如今已能处理复杂任务，但针对长时程的战略推理研究仍然不足：大多数智能体基准测试短小且情节化，而且模型在长任务上的表现难以分解为存储知识、推理能力和运气。我们提出SpireBench，让模型完整地游玩商业卡牌构筑游戏《杀戮尖塔》，在该游戏中早期决策决定长期成败。在最难难度下对20个大语言模型进行测试后发现：前沿模型击败了启发式基线，达到普通人类玩家的平均水平，但仍远远落后于专家玩家。这一差距随着单局进程的延长而扩大：最强的模型在第一幕的胜率与专家相当，但随后逐渐落后。区分强弱模型的关键在于它们打好每一场战斗的能力；而所有模型在整个局中都会误判风险，倾向于保留生命值而非花费生命值来换取成长。对游戏中所有术语进行全面且自洽的重命名实验表明……（摘要至此截断）

    arXiv:2610.12596v1 Announce Type: new  Abstract: Agentic large language models now tackle complex tasks, yet strategic reasoning over long horizons is under-studied: most agentic benchmarks are short and episodic, and performance on long tasks is hard to dissect into stored knowledge, reasoning and luck. We introduce SpireBench, in which models play the complete commercial deck-builder Slay the Spire (StS), where early decisions decide long-term success. Across 20 large language models (LLMs) at the game's highest difficulty, frontier models beat heuristic baselines and match the average human player but remain far behind experts. This gap widens as the run lengthens: the strongest models win the first act as often as experts, then fall behind. What separates strong models from weak is how well they play each fight; all of them misjudge risk over a run, preserving health rather than spending it to grow stronger. A comprehensive, self-consistent renaming of every game term shows that pa
    
[^110]: 一模循环万象：面向语言模型的弹性潜在循环

    One Model to Loop Them All: Elastic Latent Recurrence for Language Models

    [https://arxiv.org/abs/2610.12582](https://arxiv.org/abs/2610.12582)

    该论文提出弹性潜在循环（ELR）训练方法，消除了循环语言模型的“单循环税”，使模型在单次循环时以与非循环模型相同的推理成本达到相近准确率，而增加循环次数即可用更多推理计算换取更高性能。

    

    潜在循环语言模型通过重复应用共享的层块，使单个模型检查点能够通过改变循环次数来权衡推理计算量与准确率。然而，在单次循环时，此类模型的性能通常远低于同等规模的非循环模型，这种“单循环税”导致在延迟敏感的场景下，人们不得不选择单独的非循环模型。我们提出了弹性潜在循环（ELR），一种能够消除该税负的循环语言模型训练方法。在单次循环时，ELR模型的服务成本与同等规模的非循环模型相同，并且在相同数据上训练后可以达到相近的准确率。每增加一次循环，就能以更多推理计算换取更高的准确率。三个关键设计使这一方法成为可能：带可学习步长的门控状态更新、在每次优化器更新中混合不同循环次数，以及让每次循环基于其循环索引和预算进行条件化。前两个设计保证了单循环准确率，第三个设计进一步降低了损失……（摘要至此截断）

    arXiv:2610.12582v1 Announce Type: new  Abstract: A latent recurrent language model applies a shared block of layers repeatedly, so one checkpoint can trade inference compute for accuracy by changing the loop count. At one loop, however, such models typically fall well short of a non-recurrent model of the same size, a one-loop tax that makes a separate non-recurrent model the better choice whenever latency matters. We present Elastic Latent Recurrence (ELR), a recipe for training looped language models without this tax. At one loop, an ELR model costs the same to serve as a non-recurrent model of the same size and, trained on the same data, reaches similar accuracy. Each additional loop then buys more accuracy for more inference compute. Three choices make this work: a gated state update with a learned step size, loop counts mixed within each optimizer update, and conditioning each pass on its loop index and budget. The first two preserve one-loop accuracy, and the third lowers loss at
    
[^111]: CLUE：一个用于无监督事件类型归纳与标注的对比学习框架

    CLUE: A Contrastive Learning Framework for Unsupervised Event Type Induction and Annotation

    [https://arxiv.org/abs/2610.12580](https://arxiv.org/abs/2610.12580)

    CLUE框架通过新颖的联合嵌入表示和对比损失引导的聚类，实现了无监督的事件类型归纳、命名与标注，从而降低了人工构建事件本体的成本和人为偏见。

    

    事件本体是定义数据集中事件的基础框架。事件类型归纳通常需要领域专家根据数据集的领域和范围制定准则来创建事件类型及其名称。然而，这一过程在标注成本方面非常耗费资源，且容易受到人为偏见的影响。为了减少人工创建事件本体的需求，我们提出了用于事件类型归纳、命名和标注的CLUE框架。我们在这项工作中的主要贡献包括：1）设计了一种新颖的联合嵌入表示，能够同时表达事件的以动词为中心的视角和语义视角；2）利用对比损失引导的聚类进行事件类型归纳；3）事件命名与细化。实验证明了我们方法的有效性。我们的代码已发布于 https://github.com/PratikMaitra/CLUE。

    arXiv:2610.12580v1 Announce Type: new  Abstract: Event ontology is the foundational framework for defining events in a dataset. Event type induction generally involves domain experts who define guidelines for creating event types and their names based on the dataset's domain and scope. However, this process is resource-intensive in terms of annotation costs and is prone to human biases. To alleviate the need for manual event ontology creation, we propose the CLUE framework for event type induction, naming, and annotation. Our main contributions in this work are: 1) designing a novel joint embedding representation for expressing both the verb-centric view and the semantic view of events; 2) event type induction using contrastive loss guided clustering; 3) event naming and refinement. Experiments demonstrate the effectiveness of our method. Our code is available at https://github.com/PratikMaitra/CLUE.
    
[^112]: 推测解码中并行草稿的上下文修正

    Contextual Correction of Parallel Draftsfor Speculative Decoding

    [https://arxiv.org/abs/2610.12570](https://arxiv.org/abs/2610.12570)

    提出轻量级机制DCOCO，通过保留并评估多个候选token进行上下文修正，在不牺牲并行起草效率的前提下恢复草稿块内的依赖关系，从而提升扩散式推测解码的草稿质量。

    

    推测解码通过使用轻量级草稿模型提出多个token，再由目标模型并行验证，从而加速大语言模型的推理。近期基于扩散模型的草稿生成器使这一范式尤具吸引力：基于注意力的深度神经网络骨干能提供强大的预测能力，而并行起草多个未来位置则避免了自回归生成的串行开销。然而，这种并行性是有代价的：草稿块内的token在预测时无法获知其前面草稿token的实际内容，从而丢失了自回归生成中自然可用的块内依赖关系。我们提出DCOCO（DFlash Contextual Corrector，DFlash上下文修正器），这是一种在保持并行起草的同时恢复这些依赖关系的轻量级机制。DCOCO并不直接采纳每个位置的最高预测，而是保留多个候选token并对……

    arXiv:2610.12570v1 Announce Type: new  Abstract: Speculative decoding accelerates large language model inference by using a lightweight drafter to propose multiple tokens that are subsequently verified in parallel by the target model. Recent diffusion-based drafters make this paradigm particularly attractive: deep attention-based neural backbones provide strong predictions, while drafting multiple future positions in parallel avoids the sequential cost of autoregressive generation. This parallelism, however, comes at a cost. Tokens within a draft block are predicted without access to the realized identities of preceding draft tokens, removing intra-block dependencies that would naturally be available under autoregressive generation. We introduce DCOCO (DFlash Contextual Corrector), a lightweight mechanism that restores these dependencies while preserving parallel drafting. Rather than committing to each position's top prediction, DCOCO retains multiple candidate tokens and evaluates th
    
[^113]: SWE-Duel：迈向面向编程智能体的自扩展竞技场

    SWE-Duel: Towards a Self-Scaling Arena for Coding Agents

    [https://arxiv.org/abs/2610.12569](https://arxiv.org/abs/2610.12569)

    SWE-Duel 提出了一种自扩展的编程智能体评估竞技场：两个智能体以红蓝对抗的方式互相出题——红方在真实仓库中实现新功能并植入隐蔽缺陷，蓝方进行盲修——使任务在评估时动态生成、无内容可泄露，且难度随参与智能体的能力提升而自动扩展。

    

    静态编程基准在发布之时便已固定。随着时间推移，其任务内容会泄露到训练数据中，更强的智能体最终会使这些基准饱和。我们认为，基准测试应当追求“自扩展”：其任务与判定标准在评估时才被创建，因此不存在预先可供泄露的内容，且其难度由参与的智能体决定，而非在发布时固定。我们提出了 SWE-Duel，一个让两个编程智能体通过镜像的红蓝对抗回合相互评估的竞技场。在每一回合中，红方智能体在一个真实代码仓库上通过三个上下文隔离的会话进行操作：它提出并实现一个带有特性测试的新功能，然后在该功能中植入一个隐蔽的缺陷，并附带密封的缺陷触发测试。所产生的拉取请求只有在满足以下条件时才会被接受：(1) 它通过特性测试和仓库的现有测试，同时恰好只在缺陷测试上失败；(2) 同一智能体的全新会话能够在“盲修”条件下修复该缺陷……

    arXiv:2610.12569v1 Announce Type: new  Abstract: Static coding benchmarks are fixed at release. Over time their tasks leak into training data, and stronger agents eventually saturate them. We argue that a benchmark should instead aim to be self-scaling: its tasks and oracles are created at evaluation time, so no content exists in advance to leak, and its difficulty is set by the participating agents rather than fixed at release. We introduce SWE-Duel, an arena in which two coding agents evaluate each other over mirrored Red-Blue turns. In each turn, the Red agent works on a real repository across three context-isolated sessions. It proposes and implements a new feature with feature tests, then embeds a subtle bug in that feature along with sealed bug-triggering tests. The resulting pull request is admitted only if (1) it passes the feature tests and the repository's existing tests while failing exactly the bug tests, and (2) a fresh session of the same agent can repair the bug blind, w
    
[^114]: AutoIndexer：通过在编辑链上训练因果模型实现灵活顺序解码

    AutoIndexer: Flexible-Order Decoding by Training Causal Models on Chains of Edits

    [https://arxiv.org/abs/2610.12545](https://arxiv.org/abs/2610.12545)

    AutoIndexer提出了一种模拟文本编辑器的“编辑链”训练与解码机制，使因果自回归模型能够通过插入、替换和删除操作灵活地事后修改已生成的输出，从而支持非因果顺序的灵活解码。

    

    自回归解码不太适合非因果顺序的数据，例如交叉引用或未来依赖关系，因为其输出无法事后修改，这与迭代式的人类推理不同。扩散式大语言模型部分缓解了这一限制，但仍无法灵活地在预先生成的上下文中插入或删除任意数量的标记。我们将AutoIndexer设计为一个具有新颖训练和解码机制的生成模型，该机制模拟“编辑链”，使模型能够通过标记的插入、替换和删除来灵活地事后修改其先前的输出。该模型学习预测用于开启和关闭编辑的标记token，同时将光标移动到现有token序列中的位置，类似于文本编辑器。训练序列是通过对真实标签施加逆链式编辑操作进行扰动而获得的，模型通过撤销这些逆操作来学习编辑。扰动过程……

    arXiv:2610.12545v1 Announce Type: cross  Abstract: Autoregressive decoding is less suited to non-causally ordered data, e.g., cross-references or future dependencies, since their outputs cannot be revised post hoc, unlike iterative human reasoning. Diffusion LLMs partially mitigate this limitation but still cannot flexibly insert or remove arbitrary numbers of tokens from a pre-generated context. We design AutoIndexer as a generative model with a novel training and decoding mechanism that simulates a Chain of Edits, allowing the model to flexibly modify its prior outputs in hindsight via token insertions, substitutions, and deletions. The model learns to predict marker tokens for opening and closing edits, simultaneously moving a cursor to positions in the existing token sequence, analogous to a text editor. Training sequences are obtained by perturbing ground-truth labels with inverse chained edit operations, and the model learns to edit by undoing those inverses. The perturbation tra
    
[^115]: 谐振序列模型的输入依赖读取相位

    An Input-Dependent Readout Phase for Resonant Sequence Models

    [https://arxiv.org/abs/2610.12496](https://arxiv.org/abs/2610.12496)

    本文将 ResonatorLM 中每个头固定的读取相位改为当前 token 的函数，仅在读取环节引入输入依赖性，在保持精确卷积和仅增加 0.2% 参数的同时提升了模型能力，并分析了未训练振荡器组的相位可编码时间但线性重建能力受限（约 16 个 token）。

    

    具有固定大小状态的序列模型避免了注意力机制的二次方计算开销，但面临一个设计选择：输入依赖的状态转移能提供内容相关的记忆，但代价是需要基于FFT的训练；而固定转移则保留了 O(T log T) 的卷积路径，却对每个输入都以相同方式读取记忆。ResonatorLM 是最近提出的属于第二类的模型，它通过每个注意力头固定的学习相位来读取一组阻尼复振荡器。我们将该读取相位改为当前 token 的函数，从而仅在读取过程中引入输入依赖性，同时保持卷积的精确性，且仅增加 0.2% 的参数。对未训练振荡器组的分析表明，相位确实编码了经过的时间，但普通 token 会在 2-8 步内衰减到背景水平以下，且任何阻尼选择都无法将线性重建扩展到约 16 个 token 之外。在 WikiText-2 上使用 600 万参数的字符级模型实验中，输入依赖相位与原模型在（原文摘要在此处被截断）

    arXiv:2610.12496v1 Announce Type: cross  Abstract: Sequence models with a fixed-size state avoid the quadratic cost of attention, but face a design choice. An input-dependent state transition gives content-dependent memory at the cost of FFT-based training, while a fixed transition keeps the O(T log T) convolutional path but reads memory the same way for every input. ResonatorLM, a recent model of the second kind, reads a bank of damped complex oscillators through a fixed learned phase per head. We make that readout phase a function of the current token, which adds input dependence to the read only, keeps the convolution exact, and costs 0.2 percent more parameters. An analysis of the untrained bank shows that phase does encode elapsed time, but an ordinary token falls below the background within 2-8 steps and no choice of damping extends linear reconstruction beyond about 16 tokens. In 6M-parameter character-level models on WikiText-2, the input-dependent phase matches the original at
    
[^116]: Kairos：迈向适用于所有模态的字节级基础模型

    Kairos: Toward a Byte-Level Foundation Model for Every Modality

    [https://arxiv.org/abs/2610.12495](https://arxiv.org/abs/2610.12495)

    Kairos 提出了一种在统一字节空间上处理文本、图像、音频、视频、激光雷达等多模态的小型基础模型架构，融合门控 DeltaNet 线性注意力、滑动窗口注意力、注意力-残差聚合、稀疏 MoE 和多尺度字节级编解码器，并通过单机上的小规模过拟合收敛测试验证该组合能否有效优化。

    

    在单台本地机器上训练的小型语言模型几乎没有留下架构试错的空间。Kairos 结合了基于共享投影的双向门控 DeltaNet 线性注意力与滑动窗口注意力（LiZAttention2）、分块的注意力-残差聚合、稀疏混合专家（MoE）前馈层，以及多尺度字节级编解码器，在文本、图像、音频、视频、激光雷达和控制流共享的统一字节空间上，以均匀噪声去噪目标进行训练；本报告在追问这种组合是否有用之前，先追问它究竟能否正常优化。我们给出了每个组件由代码推导出的算法，包括一个跨会话的循环状态记忆门控，随后进行收敛性健全性检查：在 1–15M 参数规模下对单个 440 字节的句子进行过拟合测试（仅文本），其中对编解码器和注意力选择使用三个随机种子，并对另外七个因素进行单次运行的参数扫描。所测试的两个注意力变体在……

    arXiv:2610.12495v1 Announce Type: cross  Abstract: Small language models trained on a single local machine leave little room for architectural trial and error. Kairos combines bidirectional gated-DeltaNet linear attention and sliding-window attention over shared projections (LiZAttention2), block-wise Attention-Residual aggregation, a sparse Mixture-of-Experts (MoE) feed-forward layer and a multi-scale byte-level codec, trained with a uniform-noise denoising objective over one byte space shared by text, image, audio, video, lidar and control streams; this report asks whether the combination optimizes at all before asking whether it is useful. We give code-derived algorithms for each component, including a cross-session recurrent-state memory gate, then run convergence sanity checks: overfittinga single 440-byte sentence at 1--15M parameters, text only, with three seeds for codec and attention choices and single-run sweeps for seven further factors. The two attention variants tested at 
    
[^117]: 排练一切，记住无物：Attic-KV 只排练将被读取的内容

    Rehearse Everything, Remember Nothing: Attic-KV Rehearses What Will Be Read

    [https://arxiv.org/abs/2610.12133](https://arxiv.org/abs/2610.12133)

    该论文发现传统KV缓存压缩中“重读全部上下文”的排练策略在低保留率下会因预算分散而失效，并提出Attic-KV——只排练将来会被读取的内容，像考前自测一样大幅提升缓存压缩后的记忆效果。

    

    许多键值（KV）缓存会在无人知晓将来要被问及什么之前就被压缩：为检索而缓存的文档、跨请求共享的提示前缀、长对话的记忆。主流方法通过“排练”来为KV条目打分：模型重新阅读上下文并保留其关注的条目，并假设缓存对其上下文排练得越完整，就记得越牢。我们证明，在紧张的预算下这一假设会适得其反：排练一切，记住无物。在3%的保留率下，重读整个上下文在RULER上仅能保留96.5分中的31.5分，而在LongBench的自然文本任务上，其表现甚至低于完全不排练的方法。原因在于缓存会保留它所排练的内容：重读会将预算分散到整个上下文，导致答案自身的条目仅以略高于随机的概率得以保留。正如考试前的学生，缓存通过自我测试比通过重读能记得更多。

    arXiv:2610.12133v1 Announce Type: new  Abstract: Many key-value (KV) caches are compressed before anyone knows what will be asked of them: a document cached for retrieval, a prompt prefix shared across requests, the memory of a long conversation. The prevailing approach scores KV entries by rehearsal: the model rereads the context and keeps the entries it attends to, assuming that the more completely a cache rehearses its context, the better it remembers it. We show that under tight budgets this assumption backfires: rehearse everything, remember nothing. At a 3% keep ratio, rereading the whole context keeps 31.5 of 96.5 points on RULER, and on LongBench's natural-text tasks it falls below methods that rehearse nothing at all. The cause is that a cache keeps what it rehearses: rereading spreads the budget across the whole context, so the answer's own entries survive at little more than chance. Like a student before an exam, a cache remembers more by testing itself than by rereading. Tw
    
[^118]: InterviewPlayground：一个用于评估AI访谈员的模拟环境

    InterviewPlayground: A Simulation Environment for Evaluating AI Interviewers

    [https://arxiv.org/abs/2610.12023](https://arxiv.org/abs/2610.12023)

    提出了InterviewPlayground，一个基于社会理论模拟参与者行为、通过多轮交互生成标准化访谈成绩单来评估AI访谈员的模拟环境，并验证了模拟评估结果能够预测AI访谈员与真实人类参与者互动时的实际表现。

    

    AI访谈员正被越来越多地开发出来，用于在市场研究、民意调查、偏好引导和社会科学研究等应用中获取开放式回答。然而，评估AI访谈员十分困难，因为它们在长时间的多轮交互中运行，必须根据参与者的行为做出适应。为了满足这一需求，我们开发了InterviewPlayground，一个用于评估AI访谈员的模拟环境，其中模拟研究对象的行为建立在社会理论基础之上。InterviewPlayground中的模拟研究会生成一份InterviewReportCard（访谈成绩单），通过一系列经过验证的测量指标来评估AI访谈员的表现。为了检验这种基于模拟的评估能否预测AI访谈员在与真实人类参与者互动时的表现，我们开展了15项真实的定性研究，涉及五个AI访谈员、三个访谈主题和450名人类参与者，并将其与InterviewPlayground中的模拟研究结果进行了比较。

    arXiv:2610.12023v1 Announce Type: new  Abstract: Increasingly, AI interviewers are being developed to elicit open-ended responses in applications like market research, public polling, preference elicitation, and social science research. However, evaluating AI interviewers is challenging because they function in extended, multi-turn interactions where they must adapt to participant behaviors. To address this need, we develop InterviewPlayground, a simulation environment for evaluating AI interviewers using simulated study participants whose behaviors are grounded in social theory. Simulated studies in InterviewPlayground produce an InterviewReportCard, which assesses the performance of AI interviewers using a suite of validated measures. To test whether our simulation-based evaluations predict performance with human participants, we conduct 15 real qualitative studies with five AI interviewers, three interview topics, and 450 human participants and compare them to simulated studies in I
    
[^119]: 历史何时有用何时有害：多模态对话轮次中的选择性历史使用

    When History Helps and Hurts: Selective History Use across Multimodal Turns

    [https://arxiv.org/abs/2610.11948](https://arxiv.org/abs/2610.11948)

    该论文提出ReTurn基准（7,000个涵盖视觉与音频的任务），首次将多模态多轮对话中的历史使用需求与问题难度解耦，以系统评估模型对对话历史的选择性使用能力。

    

    可靠的多模态交互依赖于对对话历史的选择性使用：较早的问题可能仍然相关，但其先前的答案可能已经过时；而当前的请求即使面临相互冲突的新观察，也可能依赖于历史证据。现有的多轮评估很少将这种历史使用需求与底层问题难度区分开来。为填补这一空白，我们提出了ReTurn，一个包含7,000个基础任务、涵盖视觉与音频证据的基准，用于评估选择性历史使用。对于携带任务的历史，Reconfirm/Reground要求将历史问题应用于当前媒体，同时改变历史答案的一致性；对于携带证据的历史，Retrieve/Rebind要求利用历史媒体回答当前问题，同时改变当前媒体的竞争程度。每对任务都保持目标问题、媒体和答案不变。这些任务支持开放式和多选题评估，并配有匹配的单轮对照任务……

    arXiv:2610.11948v1 Announce Type: new  Abstract: Reliable multimodal interaction depends on selective use of conversational history: an earlier question may remain relevant while its previous answer is outdated, whereas a current request may depend on historical evidence despite conflicting new observations. Existing multi-turn evaluations rarely separate these history-use demands from underlying question difficulty. To address this gap, we introduce ReTurn, a benchmark of 7,000 base tasks spanning visual and audio evidence for evaluating selective history use. For task-carrying history, Reconfirm/Reground require applying a historical question to current media while varying historical agreement; for evidence-carrying history, Retrieve/Rebind require answering a current question using historical media while varying current-media competition. Each pair preserves the target question, media, and answer. Tasks support open-ended and multiple-choice evaluation, with matched single-turn coun
    
[^120]: 平滑稀疏混合专家模型的Top-k暴露边界

    Smoothing the Top-k Exposure Boundary for Sparse Mixture-of-Experts

    [https://arxiv.org/abs/2610.11575](https://arxiv.org/abs/2610.11575)

    提出弹性专家路由方法，通过在以k为中心的局部离散分布中随机采样激活专家数量，将稀疏混合专家模型中刚性的top-k选择边界软化为渐进的概率分布，在不增加计算成本的前提下缓解了竞争专家因阈值划分而导致的训练反馈不均衡问题。

    

    稀疏混合专家模型在保持每个token固定计算预算的同时，能够高效地扩展参数容量。然而，传统的训练范式强制执行静态的top-k专家选择，这将连续的路由分布转换成了刚性的阶跃函数。这一约束引入了一个脆弱的边界，使得高度竞争的专家仅因微小的分数波动就被任意地划分为完全监督区域和零反馈区域。为了解决这个问题，我们提出了弹性专家路由，它从以k为中心的局部离散分布中随机采样激活专家的数量。在多次训练迭代中，该机制将尖锐的阈值软化成渐进的概率分布。由于采样邻域保持对称，该方法在匹配确定性训练的期望计算成本的同时，保留了推理预算。大量实验表明……

    arXiv:2610.11575v1 Announce Type: new  Abstract: Sparse Mixture-of-Experts models scale parameter capacity efficiently while maintaining a fixed compute budget per token. However, traditional training paradigms enforce a static choice of top-$k$ experts, which converts a continuous routing distribution into a rigid step function. This constraint introduces a brittle boundary where highly competitive experts are arbitrarily separated into full-supervision and zero-feedback zones based on minor score fluctuations. To address this issue, we propose Elastic Expert Routing, which stochastically samples the active expert budget from a localized discrete distribution centered at $k$. Over multiple training iterations, this mechanism softens the sharp threshold into a gradual probability distribution. Because the sampling neighborhood remains symmetric, this approach matches the expected computational cost of deterministic training, while preserving the inference budget. Extensive experiments 
    
[^121]: 面向多智能体LLM谈判的宪法门控与确定性恢复：针对有状态对抗性守门人的消融实验

    Constitutional Gating and Deterministic Recovery for Multi-Agent LLM Negotiation: Ablations Against a Stateful Adversarial Gatekeeper

    [https://arxiv.org/abs/2610.11542](https://arxiv.org/abs/2610.11542)

    该论文提出由5支柱运行时宪法、4层集群和认知退火组成的三部分控制栈，并通过针对有状态对抗性守门人的消融实验，验证其能消除多智能体LLM谈判中的三类模型调用浪费并实现确定性死锁恢复。

    

    多智能体LLM系统在与有状态对手谈判时会以三种方式浪费模型调用：永远无法满足对手隐藏接受条件的“礼貌循环”、触发重试的格式错误输出，以及对手要求智能体必须拒绝的事项时所陷入的合规死锁。我们研究了一个由三部分组成的控制栈——5支柱运行时宪法、4层集群（导演、三智能体多数投票、监视器、模式硬门）以及认知退火（确定性死锁检测、对智能体侧上下文的原子清除、规范化恢复消息）——并在一个已发布的对抗性守门人上进行测试，该守门人的接受规则为固定的正则表达式，其LLM仅用于渲染回复文本。该测试环境具有已知解：它衡量的是该控制栈能否在对抗集群漂移的同时执行符合宪法的策略并从死锁中恢复，而非衡量它是否有所发现。每个配置运行五次（共30次运行；……

    arXiv:2610.11542v1 Announce Type: cross  Abstract: Multi-agent LLM systems negotiating with a stateful counterpart waste model calls in three ways: polite loops that never meet the counterpart's hidden acceptance condition, malformed outputs that trigger retries, and compliance deadlocks in which the counterpart demands something the agent must refuse. We study a three-part control stack - a 5-Pillar runtime constitution, a 4-tier swarm (Director, three-agent majority vote, Monitor, schema hard gate) and Cognitive Annealing (deterministic deadlock detection, atomic purge of the agent-side context, a canonical recovery message) - against a released adversarial Gatekeeper whose acceptance rules are fixed regular expressions and whose LLM only renders reply text. The testbed has a known solution: it measures whether the stack executes a constitution-aligned strategy against swarm drift and recovers from deadlock, not whether it discovers anything. In five runs per configuration (30 runs; 
    
[^122]: 测量与缓解RLVR中的解模式坍缩

    Measuring and Mitigating Solution Mode Collapse in RLVR

    [https://arxiv.org/abs/2610.11064](https://arxiv.org/abs/2610.11064)

    本研究提出ModeBench多解任务基准，发现RLVR后训练在保持或提升准确率的同时，会导致模型的解多样性坍缩，概率集中于更少的正确解题模式上。

    

    语言模型通常可以用多种方式回答同一个问题，但基于可验证奖励的强化学习（RLVR）对于模型产生哪个正确答案并不加以区分。无论一个解是某个熟悉答案的第千次重复，还是模型从未产生过的新解，它都会获得相同的奖励。然而，在训练过程中让模型保留多个正确解具有潜在价值。例如，多种模式可以为用户提供选择，并提供有助于提升模型整体性能的问题解决策略。在此，我们介绍了ModeBench，这是一个多解任务基准，其中验证器会同时返回正确性和所发现的模式。我们随后利用ModeBench来衡量RLVR后训练过程中解多样性的变化。我们发现，即使准确率保持不变甚至有所提升，RLVR后训练仍会将概率集中到更少的正确模式上，而且前沿模型几乎……

    arXiv:2610.11064v1 Announce Type: cross  Abstract: A language model (LM) can usually answer the same question in more than one way, but reinforcement learning with verifiable rewards (RLVR) is indifferent to which correct answer a model produces. A solution will earn the same reward whether it is the thousandth copy of a familiar answer or one the model has never produced before. Yet, there is potential value in having the model retain multiple correct solutions as it is trained. For instance, multiple modes may give users a choice and provide problem-solving strategies that improve overall model performance. Here, we introduce ModeBench, a benchmark of multi-solution tasks in which the verifier returns both correctness and mode discovered. We then use ModeBench to measure how solution diversity changes under RLVR post-training. We find that RLVR post-training concentrates probability onto fewer correct modes even as accuracy holds or improves, and moreover, that frontier models are al
    
[^123]: PHRBench：大语言模型幻觉后推理的行为学评估

    PHRBench: A Behavioral Evaluation of Post-Hallucination Reasoning in LLMs

    [https://arxiv.org/abs/2610.10455](https://arxiv.org/abs/2610.10455)

    提出了PHRBench这一受控基准，通过幻觉顺从、幻觉规避和启发式纠正等行为指标，在四个领域对18个大语言模型处理幻觉前提的推理轨迹进行评估，发现模型成功从幻觉中恢复并得出正确答案的情况仍然相对罕见。

    

    幻觉信息可能在多阶段大语言模型系统中传播，并成为后续推理上下文的一部分。现有关于幻觉后推理（PHR）的研究主要刻画最终结果的变化和总体的推理动态，而对模型在响应层面如何处理幻觉前提的理解仍然不足。在本工作中，我们提出了PHRBench，这是一个受控基准，用于在四个领域、18个大语言模型上对行为结构化的幻觉后推理进行评估。PHRBench通过幻觉顺从、幻觉规避和启发式纠正三个指标，独立于最终答案正确性地刻画每条推理轨迹，并将“有洞察力的轨迹”定义为最终成功纠正并达到正确答案的轨迹。在4820个受控实例中，我们发现成功的恢复仍然相对罕见，并且与推理轨迹上更频繁的信念更新相关（摘要原文在此处被截断）。

    arXiv:2610.10455v1 Announce Type: new  Abstract: Hallucinated information can propagate through multi-stage LLM systems and become part of the context for subsequent reasoning. Existing studies of post-hallucination reasoning (PHR) mainly characterize changes in final outcomes and aggregate reasoning dynamics, leaving how models resolve hallucinated premises at the response level insufficiently understood. In this work, we introduce PHRBench, a controlled benchmark for behaviorally structured PHR across four domains and 18 large language models. PHRBench characterizes each reasoning trajectory independently of final-answer correctness through Hallucination Compliance, Hallucination Avoidance, and Heuristic Correction, and defines an insightful trajectory as successful correction that ultimately reaches the correct answer. Across 4820 controlled instances, we find that successful recovery remains relatively rare and is associated with more frequent belief updates along the reasoning tra
    
[^124]: 大型语言模型在被提示进行不真实回答时的推理Token激增现象

    Reasoning-Token Spikes Under Prompted Untruthful Responding in Large Language Models

    [https://arxiv.org/abs/2610.10405](https://arxiv.org/abs/2610.10405)

    该论文基于认知负荷理论，提出利用推理token数量这一无需访问思维链内容的低带宽信号，发现大语言模型在被提示进行不真实回答时会出现推理token数量的激增，从而为检测模型欺骗行为提供了新方法。

    

    监控推理型人工智能模型的思维链仍然是检测此类模型中欺骗行为及其他形式不良行为的关键方法。然而，基于语义的思维链监控依赖于推理轨迹清晰可读、对产生模型行为的底层计算足够忠实，更不用说其还必须是可访问的。此外，越来越多的证据表明，思维链输出可能很快变得难以解读或不忠实，甚至可能不再可访问。基于认知负荷理论，我们研究了一种低带宽信号——生成的推理token数量——它不需要访问推理轨迹的内容。三个具备推理能力的大型语言模型回答了210道选择题——涵盖分析性、描述性和规范性推理类型以及道德与非道德领域——在系统提示指示它们进行不真实回答的条件下（原文在此处截断）。

    arXiv:2610.10405v1 Announce Type: cross  Abstract: Monitoring the chain-of-thought of reasoning artificial intelligence (AI) models remains a key approach to detecting deception and other forms of misbehavior in such models. However, semantic chain-of-thought monitoring depends on reasoning traces being legible and sufficiently faithful to the underlying computations that produced the model's behavior, not to mention accessible. Moreover, there is increasing evidence that chain-of-thought outputs may soon become illegible or unfaithful, if they even remain accessible. Based on cognitive load theory, we investigate a lower-bandwidth signal -- the number of reasoning tokens generated -- which does not require access to the content of the reasoning trace. Three reasoning-capable large language models answered 210 multiple-choice questions -- across analytic, descriptive, and normative reasoning types as well as moral and non-moral domains -- under system prompts instructing them to respon
    
[^125]: 解耦逻辑与人设：边缘大语言模型智能体对上下文污染的结构性免疫

    Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution

    [https://arxiv.org/abs/2610.09772](https://arxiv.org/abs/2610.09772)

    该论文提出AO-DA解耦架构，将边缘LLM智能体的逻辑推理与人设表达分离为同一INT4基础模型上两条可热插拔LoRA适配器的独立推理路径，使逻辑部分对上下文污染实现结构性免疫。

    

    运行在边缘设备上的小型语言模型智能体必须在同一个上下文窗口内同时维持人设并进行正确推理，而该窗口中充满了对话历史和人设指令。我们研究了当这些历史记录冗长、具有误导性且人设内容占比过高（即人设-逻辑干扰）时，此类智能体的逻辑推理部分会发生什么，并提出了一种解耦架构（AO-DA），在单个INT4量化的基础模型上，通过可热插拔的LoRA适配器，将逻辑推理（“做什么/What”）与人设表达（“怎么做/How”）分离为两条独立的推理路径。逻辑路径仅接收核心对话轮次，并输出可验证的结构化状态；人设路径则结合完整的对话历史，以符合角色设定的方式对该状态进行渲染。在同一基础模型上的消融实验中（在Apple M2笔记本电脑上进行，使用Llama-3.1-8B-Instruct和Gemma-3-4B-it，4比特量化；共480次运行，涵盖4个污染等级 × 3个实验组 × 2个任务 × 2个人设 × 5个随机种子），我们发现：（i）解耦后的逻辑路径对污染具有结构性不变性：其……（原文摘要在此处截断）

    arXiv:2610.09772v1 Announce Type: new  Abstract: Small language-model agents on edge devices must hold a persona and reason correctly at once, inside one context window that fills with conversational history and persona instructions. We study what happens to the logical part of such an agent when that history is long, misleading and persona-heavy (persona-logic interference), and present a Decoupling Architecture (AO-DA) that separates logical inference ("What") from persona expression ("How") into two inference paths on one INT4 base model with hot-swappable LoRA adapters. The logic path receives only the core turn and emits a verifiable structured state (Micro-State); the persona path renders it in character with the full history. In same-base-model ablations on an Apple M2 laptop (Llama-3.1-8B-Instruct and Gemma-3-4B-it, 4-bit; 480 runs over 4 pollution levels x 3 arms x 2 tasks x 2 personas x 5 seeds) we find: (i) the decoupled logic path is structurally invariant to pollution: its
    
[^126]: 从专家引导的证明搜索到自动化开放问题求解

    From Expert-Guided Proof Search to Automated Open-Problem Solving

    [https://arxiv.org/abs/2610.09769](https://arxiv.org/abs/2610.09769)

    研究者开发的多智能体开源系统Bolzano无需针对具体问题的人类指导，在约3,800个开放问题中自动解决了约200个，其中包括经原作者确认的STOC 2026论文中提出的4个问题。

    

    大型语言模型对数学研究的贡献日益增多，而数学研究的进展往往依赖于高效的证明搜索、渐进式的改进以及细致的验证。我们介绍了Bolzano，一个多智能体开源系统，它使用并行的证明器智能体配合验证器智能体协同工作，并维护人类可读的研究状态。在专家挑选的问题上进行的初期人工使用产生了8项成果，其证明均经领域专家核查。受这些案例研究的启发，我们在从四组论文中提取的约3,800个开放问题上运行了Bolzano，无需针对具体问题的人类指导，便解决了其中约200个开放问题。其中一项实验使用了被理论计算机科学顶级会议STOC 2026接收的论文，我们回答了这些论文中提出的四个问题，并得到了论文作者的确认。

    arXiv:2610.09769v1 Announce Type: cross  Abstract: Large language models are increasingly contributing to mathematical research, where progress often depends on efficient proof search, incremental improvements and careful verification. We describe Bolzano, a multi-agent open-source system that uses parallel prover agents with a verifier agent and maintains a human-readable research state. Initial manual use on expert-selected problems yielded 8 results whose proofs were checked by domain experts. Motivated by these case studies, we ran Bolzano without problem-specific human guidance on about 3,800 open problems extracted from four sets of papers, solving about 200 open problems. One experiment used papers accepted to STOC 2026, a top conference in theoretical computer science. There, we answered four questions raised in the papers, as confirmed by their authors.
    
[^127]: TRACE：面向MoE语言模型FP4强化学习的Rollout引导量化感知训练

    TRACE: Rollout-Guided Quantization-Aware Training for FP4 Reinforcement Learning of MoE Language Models

    [https://arxiv.org/abs/2610.07767](https://arxiv.org/abs/2610.07767)

    TRACE是一个面向MoE语言模型强化学习训练的FP4量化框架，通过rollout引导的量化感知训练，利用rollout侧量化结果指导训练侧FP4舍入决策，直接缩小训练路径与rollout路径两条量化执行路径之间的差异。

    

    对大语言模型（LLM）进行后训练的强化学习（RL）在rollout生成过程中会产生大量的计算和内存开销，这促使人们采用低精度rollout来实现高效的RL训练。然而，现有的FP4 RL方法存在一个关键局限：它们主要在训练路径和rollout路径上分别独立地优化量化精度，而不是直接减少两条量化执行路径之间的差异。在本工作中，我们提出了TRACE（Train-Rollout Quantization Alignment via Compact GuidancE，通过紧凑引导实现训练-Rollout量化对齐），这是一个面向混合专家（MoE）语言模型RL训练的FP4量化框架，解决了现有FP4 RL方法的局限性。TRACE引入了rollout引导的量化感知训练，利用rollout侧的量化结果来指导训练侧的FP4舍入决策，从而直接减少训练与rollout之间的差异。此外，TRACE采用了高效的量化……

    arXiv:2610.07767v1 Announce Type: cross  Abstract: Reinforcement learning (RL) for post-training large language models (LLMs) incurs substantial computation and memory overhead during rollout generation, which motivates low-precision rollout for efficient RL training. However, existing FP4 RL methods suffer from a key limitation: they primarily optimize quantization accuracy on the training and rollout paths independently rather than directly reducing the discrepancy between the two quantized execution paths. In this work, we propose TRACE (Train-Rollout Quantization Alignment via Compact GuidancE), an FP4 quantization framework for RL training of Mixture-of-Experts (MoE) language models that addresses the limitation of existing FP4 RL methods. TRACE incorporates rollout-guided quantization-aware training that uses rollout-side quantization outcomes to guide training-side FP4 rounding decisions, directly reducing train-rollout discrepancy. Moreover, TRACE adopts an efficient quantizati
    
[^128]: Wikidata搜索轨迹：用于训练知识图谱搜索智能体的数据集

    Wikidata Search Traces: A Dataset for Training Knowledge Graph Search Agents

    [https://arxiv.org/abs/2610.06650](https://arxiv.org/abs/2610.06650)

    该论文发布了首个记录解题者在Wikidata知识图谱上探索过程的数据集，用于训练图搜索智能体，并证明图搜索难度可通过问题结构控制、长程搜索失败主要源于证据管理方式而非模型本身，且开源模型在合适环境中可匹敌商业模型。

    

    Wikidata是最大的开放知识库之一，然而要在其上回答一个复杂问题，仍然需要编写一条SPARQL查询，该查询需指明正确的实体和属性，并将它们之间的关系串联起来。语言模型提供了一种自然语言的替代方案，但其回答主要依靠记忆，而这对于知名度较低的实体最为不可靠。我们研究了通过探索图谱来回答问题的智能体，并指出两个障碍限制了它们：一是缺乏记录求解者如何进行探索的训练数据，二是将大规模图谱结果直接加入模型上下文的接口设计。我们检验了三个假设：图搜索的难度可以通过问题的结构来控制，而不仅仅依靠冷门实体或措辞；长程搜索中的许多失败源于对检索证据的管理方式，而非模型本身；以及在合适的环境中，开源权重模型可以匹敌商业（模型）……

    arXiv:2610.06650v2 Announce Type: replace  Abstract: Wikidata is one of the largest open knowledge bases, yet answering a complex question over it still requires a SPARQL query that names the right entities and properties and chains their relations. Language models offer a natural-language alternative but answer largely from memory, which is least reliable for less prominent entities. We study agents that instead answer by exploring the graph, and argue that two obstacles limit them: the lack of training data recording how a solver explores, and interfaces that add large graph results directly to the model's context. We test three hypotheses: that the difficulty of graph search can be controlled through the structure of a question rather than only through obscure entities or wording; that much of the failure on long-horizon search comes from how retrieved evidence is managed rather than from the model itself; and that, in a suitable environment, open-weight models can match commercial 
    
[^129]: DREAM：面向多模态多智能体辩论的动态分辨率分配方法

    DREAM: Dynamic Resolution Assignment For Multimodal Multi-agent Debate

    [https://arxiv.org/abs/2610.05615](https://arxiv.org/abs/2610.05615)

    提出DREAM框架，通过零样本探测为每个智能体动态分配不同的图像分辨率，并利用ANLL不确定性量化和自适应阈值来缓解多模态多智能体辩论中的群体思维问题，从而提升多模态推理能力。

    

    多智能体辩论（MAD）已成为提升大语言模型（LLM）推理能力的有效范式，并且正日益被扩展到多模态场景中。然而，现有多模态MAD框架通常让所有智能体接触相同的固定视觉输入，忽视了不同样本和不同智能体所需视觉尺度的巨大差异。此外，这些框架经常受到“群体思维”（groupthink）现象的困扰，即智能体过早放弃正确的推论，转而迎合自信但产生幻觉的同伴回答。为解决这些瓶颈，我们提出了DREAM（面向多模态多智能体辩论的动态分辨率分配），该方法通过两个核心组件运作：（1）动态分辨率分配——一个零样本探测回合，智能体在其中测试多种分辨率，使用平均归一化对数似然（ANLL）量化不确定性，并利用自适应阈值将每个智能体分配到其…

    arXiv:2610.05615v2 Announce Type: replace  Abstract: Multi-agent debate (MAD) has emerged as an effective paradigm to improve the reasoning capabilities of large language models (LLMs) and is increasingly being extended to multimodal settings. However, existing multimodal MAD frameworks typically expose agents to the same fixed visual input, ignoring substantial variation in the visual scale needed across samples and agents. In addition, these frameworks frequently suffer from groupthink, a phenomenon where agents prematurely abandon correct deductions to conform with confident but hallucinated peer responses. To address these bottlenecks, we introduce DREAM (Dynamic Resolution Assignment For Multimodal Multi-Agent Debate), which operates via two core components: (1) Dynamic Resolution Assignment, a zero-shot probe round where agents test multiple resolutions, quantify uncertainty using Average Normalized Log-Likelihood (ANLL), and use an adaptive threshold to assign each agent to its 
    
[^130]: 当大语言模型凌驾于诊断工具之上：工业故障诊断中未实现的互补性

    When LLMs Sit Above Diagnostic Tools: Unrealized Complementarity in Industrial Fault Diagnosis

    [https://arxiv.org/abs/2610.05031](https://arxiv.org/abs/2610.05031)

    该研究发现在工业故障诊断中，大语言模型与专业诊断工具的集成并未实现真正的互补性——外部专家信息虽能提升LLM表现（最高提升12.8个百分点），但集成系统仍不及最强的独立专业工具，且冲突信息甚至会推翻LLM原本正确的判断。

    

    大语言模型越来越多地被用作专业工具之上的集成层，但更强的组件并不一定带来更强的组合系统。我们在五个诊断数据集（轴承振动、过程监控、半导体设备）上研究了LLM能否可靠地利用外部诊断信息；通过成对的重复调用，将建议效应与输出不稳定性分离开来。在所有五个数据集中，相互冲突的外部信息均会推翻LLM最初正确的判断。在四个进行直接集成比较的数据集中，没有一个显示出隐式LLM集成相对于更强的独立数据源具有持续优势。在一个评估协议预先固定的田纳西-伊士曼确认数据集上，无辅助准确率为64.67%，隐式LLM-专家集成为77.43%，而单独使用专家工具则为83.33%。专家信息将LLM的性能提升了12.8个百分点（95%区间为9.7至15.9），然而……

    arXiv:2610.05031v2 Announce Type: replace  Abstract: Large language models are increasingly used as integration layers above specialized tools, but a stronger component does not necessarily produce a stronger combined system. Across five diagnostic datasets (bearing vibration, process monitoring, semiconductor equipment), we study whether an LLM can reliably use external diagnostic information; paired repeat calls separate advice effects from output instability. In all five, conflicting external information overturned initially correct LLM judgments. Among the four datasets with direct integration comparisons, none showed a consistent advantage for implicit LLM integration over the stronger standalone source. On a Tennessee Eastman confirmation set whose protocol was fixed before evaluation, unaided accuracy was 64.67%, implicit LLM-specialist integration 77.43%, and the specialist alone 83.33%. Specialist information improved the LLM by 12.8 points (95% interval 9.7 to 15.9), yet the 
    
[^131]: 一种生成语法指导的神经符号框架用于句法歧义消解：来自阿拉伯语限定词短语的证据

    A generative-informed neuro-symbolic framework for syntactic ambiguity resolution: Evidence from Arabic DPs

    [https://arxiv.org/abs/2610.02529](https://arxiv.org/abs/2610.02529)

    该研究提出一种将生成句法学概念与AraBERT相结合的神经符号框架，把阿拉伯语限定词短语的结构歧义消解建模为基于候选的决策任务，在未见评估集上取得了96.88%的准确率，同时揭示了不同挂靠类型之间性能的不对称性。

    

    句法歧义对阿拉伯语自然语言处理（NLP）构成了持续的挑战，尤其是在形态丰富的名词性结构中，多种结构解读可能与相同的表层序列相兼容。本研究提出了一种生成语法指导的神经符号框架，用于消解现代标准阿拉伯语（MSA）限定词短语中的结构歧义。该框架将生成句法学的概念与AraBERT相结合，将歧义表示为一种基于候选的决策任务，其中由语言学动机驱动的备选结构被显式构建，并通过候选条件化的输入表示进行评估。研究结果表明，该模型在未见过的评估集上达到了96.88%的准确率、95.92%的宏F1值、96.83%的加权F1值和93.94%的二分类F1值。类别层面的分析显示出不对称的性能表现，高位/VP挂靠（N1）的召回率为99.71%，而低位/NP/嵌入挂靠（N2）的召回率为89.26%，表明后者的消歧更具挑战性。

    arXiv:2610.02529v1 Announce Type: new  Abstract: Syntactic ambiguity poses a persistent challenge for Arabic NLP, particularly in morphologically rich nominal constructions where multiple structu6ral interpretations may be compatible with the same surface sequence. This study proposes a generatively informed neuro-symbolic framework for resolving structural ambiguity in Modern Standard Arabic (MSA) DPs. The framework integrates generative syntactic notions with AraBERT by representing ambiguity as a candidate-based decision task in which linguistically motivated alternatives are explicitly constructed and evaluated through candidate-conditioned input representations. Findings indicate that the model achieved 96.88% accuracy, 95.92% macro-F1, 96.83% weighted F1, and 93.94% binary F1 on the unseen evaluation set. Class-level analysis revealed asymmetric performance, with recall of 99.71% for High/VP Attachment (N1) and 89.26% for Low/NP/Embedded Attachment (N2), indicating greater diffic
    
[^132]: 让大语言模型说出它们的真实想法：测量与改进思维链-可解释性对齐

    Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment

    [https://arxiv.org/abs/2609.38972](https://arxiv.org/abs/2609.38972)

    本文提出CoT-可解释性对齐（CIA）指标来测量LLM思维链与内部计算之间的一致性，发现现有模型对齐程度有限（44.8%–75.9%），并通过以任务准确率和参数化忠实度为奖励的后训练方法有效提升了对齐性。

    

    思维链（CoT）轨迹通常被用作观察大型语言模型（LLM）如何得出答案的代理。然而，越来越多的证据表明，模型的思维链往往无法反映其内部计算，并且可以在不影响最终答案的情况下被改变。在这项工作中，我们测量并改进了LLM思维链中描述的推理过程与其内部实际计算之间的对齐程度。我们提出了CoT-可解释性对齐（CIA）指标，用于衡量模型的思维链轨迹与可解释性工具检测到的内部推理策略之间的一致性。我们在三项任务（两跳问答、提示干预和整数乘法）上对三个LLM进行了CIA评估，发现LLM在所有任务上的对齐程度都较为有限（44.8%–75.9%）。随后，我们尝试通过后训练来改进CIA，将任务准确率和参数化忠实度信号设置为奖励。实验表明……

    arXiv:2609.38972v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) traces often serve as a proxy for how Large Language Models (LLMs) arrive at their answers. However, growing evidence shows that models' CoT often fails to reflect their internal computations and can be changed without affecting their final answers. In this work, we measure and improve the alignment between the reasoning described in an LLM's CoT and what it computes internally. We propose CoT-Interpretability Alignment (CIA), a metric that measures the agreement between a model's CoT traces and its internal reasoning strategies as detected by interpretability tools. We evaluate CIA on three tasks (two-hop question answering, hint intervention, and integer multiplication) across three LLMs, finding that LLMs exhibit limited alignment across all tasks (44.8-75.9%). We then experiment with improving CIA via post-training, setting both the task accuracy and parametric faithfulness signals as a reward. Experiments sh
    
[^133]: StreamDecisionBench：在动态演化的语言流上评估生效决策

    StreamDecisionBench: Evaluating Decisions in Force on Evolving Language Streams

    [https://arxiv.org/abs/2609.38612](https://arxiv.org/abs/2609.38612)

    该论文提出 StreamDecisionBench 基准，评估语言模型在证据流不断演化时每一时刻“生效决策”的正确性，并将错误归因于判断或延迟，弥补了传统离线准确率无法衡量决策时效性的缺陷。

    

    随着自然语言驱动越来越多的应用，语言模型越来越多地作为决策组件运行在程序内部：程序向其发送当前状态，并根据返回的决策采取行动，直到新决策到来。当证据在推理过程中发生变化时，某个对其自身状态而言正确的决策，可能在该状态已经过去之后仍然生效——例如当客户开始读出卡号后，通话录音机仍在继续录音；而不计时间的（离线）准确率会将此类错误计为正确。我们提出 StreamDecisionBench（SDB），它评估每一时刻的生效决策，并将每个错误时刻归因于判断错误、延迟或两者兼有。其场景在四个应用族中流式呈现证据，参考决策由可执行代码依据公开规则计算得出。我们以对数时间轴上归一化的曲线下面积来汇总 1-5 秒更新区间内的生效准确率，对相等的时间赋予相等的权重……（原文摘要在此处截断）

    arXiv:2609.38612v1 Announce Type: new  Abstract: As natural language drives more applications, language models increasingly run inside programs as decision components: the program sends them the current state and acts on the returned decision until a newer one arrives. When evidence changes during inference, a decision correct for its own state can stay in force after that state has passed, as when a call recorder keeps running after a customer starts reading out a card number; untimed (offline) accuracy counts such an error as correct. We introduce StreamDecisionBench (SDB), which evaluates the decision in force at every instant and attributes every erroneous instant to judgment, latency or both. Its scenarios stream evidence in four application families, with reference decisions computed from public rules by executable code. We summarize in-force accuracy across update intervals of 1-5 s by its normalized area under the curve on a logarithmic time axis, giving equal weight to equal m
    
[^134]: OpenCollab：一个具有可编程协作与可控运行时的多智能体编程框架

    OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime

    [https://arxiv.org/abs/2609.38345](https://arxiv.org/abs/2609.38345)

    提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。

    

    多智能体编程系统旨在通过协作来解决复杂的软件工程任务。然而，现有的评估通常假设所配置的组织结构会被忠实地遵循，而实际情况并非如此。这种行为差距，再加上底层系统组件的差异，使得观察到的性能提升难以进行明确的归因。为此，我们提出了OpenCollab，这是一个多智能体编程框架，为可编程协作与可控运行时提供了统一的基础设施。具体而言，OpenCollab统一了组织设计，在共享运行时上强制实施实验控制，并通过细粒度的事件流来跟踪执行过程。在此基础上，我们定义了Adherence（遵循度）指标，用于量化所声明的组织结构是否真正得以实现。我们的实验表明，智能体在不同配置下的协作方式差异很大：改变任何一个单一维度都会使Adherence发生变化，从47.2%到高达97%以上。

    arXiv:2609.38345v1 Announce Type: cross  Abstract: Multi-agent coding systems are designed to tackle complex software engineering tasks through collaboration. However, existing evaluations typically assume configured organizations are followed faithfully, whereas reality differs. This behavioral gap, combined with differences in underlying system components, prevents clear attribution of observed gains. To this end, we introduce OpenCollab, a multi-agent coding framework that provides a unified infrastructure for programmable collaboration and controllable runtime. Specifically, OpenCollab unifies organization design, enforces experimental control on a shared runtime, and tracks execution through fine-grained event streams. On this basis, we define Adherence to quantify whether the declared organization is actually realized. Our experiments reveal that agents collaborate very differently across configurations: changing any single dimension shifts Adherence, from 47.2% to as high as 97.
    
[^135]: EngiWorld：前沿智能体在专业工程环境中能交付什么？

    EngiWorld: What Can Frontier Agents Deliver in Professional Engineering Environments?

    [https://arxiv.org/abs/2609.37686](https://arxiv.org/abs/2609.37686)

    该论文提出了首个覆盖完整设计循环的工程智能体基准EngiWorld，涵盖1,301个专家任务、6个工程领域和26个专业软件平台，并引入以工件为中心的自动化评估方法来检验工程成果的几何有效性、物理可行性与规则合规性。

    

    自主智能体在通用计算机使用方面取得了快速进展，但对专业工业工程的可靠自动化仍然遥不可及，因为工程工作流程要求对几何和物理约束进行推理，并处理跨软件和设计阶段保留的依赖关系。我们提出了EngiWorld，这是第一个围绕完整设计循环构建的基准：包含1,301个由专家精心策划的任务，涵盖6个工程领域（CAD、CAE、CAM、BIM、EDA和3D可视化）和26个专业软件平台，同时提供GUI和CLI两种界面，以及从软件选择到开放式任务的6种任务类型。我们进一步引入了一种以工件为中心的评估方法，该方法建立在统一的领域验证器套件之上，能够以编程方式检查最终和中间工件的几何有效性、物理可行性和规则合规性，并通过规范达成度对定量设计任务进行连续评分。

    arXiv:2609.37686v1 Announce Type: new  Abstract: Autonomous agents have made rapid progress in general-purpose computer use, but reliable automation of professional industrial engineering remains out of reach, as engineering workflows demand reasoning over geometric and physical constraints and dependencies preserved across software and design stages. We present EngiWorld, the first benchmark structured around the complete design loop: 1,301 expert-curated tasks spanning 6 engineering domains (CAD, CAE, CAM, BIM, EDA, and 3D visualization) and 26 professional software platforms, with both GUI and CLI interfaces and 6 task types ranging from software-selection to open-ended tasks. We further introduce an artifact-centric evaluation methodology built on a unified domain-verifier suite, which programmatically checks the geometric validity, physical feasibility, and rule compliance of final and intermediate artifacts, and scores quantitative design tasks continuously by specification attai
    
[^136]: 面向人工智能研究论文的编辑诱发式问题生成

    Generating Edit-Inducing Questions for AI Research Manuscripts

    [https://arxiv.org/abs/2609.36617](https://arxiv.org/abs/2609.36617)

    该研究比较了GPT与人类审稿人为AI论文草稿生成“编辑诱发式问题”的能力，发现GPT的问题能引发更广泛深入的修改但有效率更低，并揭示了一个反直觉现象：处理长上下文反而会损害推理模型生成有用输出的能力。

    

    我们研究了大语言模型（LLM）生成“编辑诱发式问题”的能力，这类问题的答案能够帮助改进论文草稿。在一个由ICLR和NeurIPS的投稿版本与定稿版本配对组成的数据集上，我们比较了GPT模型在有或没有完整论文上下文情况下所生成问题的有用性，并与人类审稿人提出的问题进行对比。结果显示，GPT生成了更多编辑诱发式问题，且与审稿人的问题相比，其问题伴随着更广泛的修改，覆盖的编辑内容范围也更广。然而，GPT问题中真正具有编辑诱发性的比例却小得多。我们的分析证实了自动化生成的问题对作者是有益的，同时揭示了一个典型案例：在某些任务中，对长上下文的恰当关注反而会削弱推理模型产出有用结果的能力。

    arXiv:2609.36617v1 Announce Type: new  Abstract: We study the ability of LLMs to generate edit-inducing questions whose answer will improve a paper draft. On a dataset of paired submission and camera-ready papers from ICLR and NeurIPS, we compare the helpfulness of questions from GPT models with or without full paper context to that of human reviewers. GPT produces more edit-inducing questions and its questions are associated with more extensive edits and cover a broader range of edited content compared to questions from reviewers. However, a much smaller percentage of the GPT questions are edit-inducing. Our analyses confirm that automated questions can be beneficial to authors and highlight an example task where proper attending to long context deteriorates reasoning model ability to produce helpful output.
    
[^137]: MS-GLA：通过多时间分辨率解决表示瓶颈的多尺度门控线性注意力

    MS-GLA: Multi-Scale Gated Linear Attention for Addressing Representational Bottlenecks via Multi-Temporal Resolution

    [https://arxiv.org/abs/2609.35664](https://arxiv.org/abs/2609.35664)

    提出多尺度门控线性注意力（MS-GLA），通过将注意力头分配到多个时间分辨率并用可学习的输入依赖融合层动态重组各头组输出，解决GLA Transformer在单一时间分辨率下的表示瓶颈，在不增加每头状态大小的情况下扩展有效记忆容量。

    

    门控线性注意力（GLA）Transformer 通过数据依赖的门控机制推动了线性循环模型的发展，但面临一个核心局限：所有注意力头共享的固定容量记忆矩阵在单一时间分辨率下运行，每个词元被逐一单独处理，这迫使它们必须同时编码局部句法模式和长程语义结构，从而造成了仅靠门控机制无法解决的表示瓶颈。我们提出多尺度门控线性注意力（MS-GLA），通过将注意力头分布到多个时间分辨率上来解决这一问题。较粗分辨率的头组池化更长的词元跨度，自然地专门化于长程依赖关系；而较细的头组则保留对局部句法结构的敏感性。一个可学习的、依赖输入的融合层在每个时间步动态重组各头组的输出，从而在不增加每头状态大小的情况下扩展了有效记忆容量……

    arXiv:2609.35664v2 Announce Type: replace-cross  Abstract: Gated Linear Attention (GLA) Transformers advance linear recurrent models through data-dependent gating, but face a core limitation: the fixed-capacity memory matrices across all heads operate at a single temporal resolution, where each token is processed individually, forcing them to simultaneously encode local syntactic patterns and long-range semantic structure, creating a representational bottleneck that gating alone is insufficient to resolve. We introduce Multi-Scale Gated Linear Attention (MS-GLA), which addresses this by distributing attention heads across multiple temporal resolutions. Coarser resolutions pool longer token spans naturally specializing toward long-range dependencies, while finer head groups retain sensitivity to local syntactic structure. A learnable, input-dependent fusion layer dynamically recombines head group outputs at each timestep, expanding effective memory capacity without increasing per-head s
    
[^138]: 多轮大语言模型污染中的认知策略分歧：一项协议梯度研究

    Epistemic Policy Divergence in Multi-Turn LLM Contamination: A Protocol-Gradient Investigation

    [https://arxiv.org/abs/2609.35308](https://arxiv.org/abs/2609.35308)

    该研究提出“会话级污染”这一失败模式，通过五种沿来源权威梯度排列的污染协议，首次系统揭示了大语言模型在多轮对话中采纳错误前提时的认知策略存在显著分歧——GPT-5.4 Mini完全抵抗采纳，而Gemini-3.1 Flash-Lite的采纳率随信息来源权威性增强而急剧上升。

    

    大语言模型将对话历史视为未经核实的上下文，因此注入先前轮次中的错误前提可能被当作事实采纳，我们将这种失败模式称为“会话级污染”。我们引入了五种沿“来源权威梯度”排列的污染协议，在保持错误前提不变的同时改变其认知框架，并在温度为零的条件下（共22,500轮）对GPT-5.4 Mini、Gemini-3.1 Flash-Lite和GLM-4.5-Air在十个知识领域进行了评估，评估采用经人类金标准验证的双轨自动评估器（二元采纳的Cohen's kappa = 1.000；崩溃严重程度的线性加权kappa为0.92）。GPT-5.4 Mini在全部500个会话中的采纳记录为零；基础模型的logit探测显示其决策边界虽受到扰动但仍然较大且有限。Gemini-3.1 Flash-Lite则呈现出陡峭的权威梯度：对自我归因的虚假信息采纳率为0.1%，对用户引用来源为23.5%，对s……（摘要原文在此处截断）

    arXiv:2609.35308v2 Announce Type: replace  Abstract: Large language models treat conversation history as unverified context, so false premises injected into prior turns can be adopted as fact, a failure mode we term session-level contamination. We introduce five contamination protocols arranged along a source-authority gradient, holding the false premise constant while varying its epistemic framing, and evaluate GPT-5.4 Mini, Gemini-3.1 Flash-Lite, and GLM-4.5-Air across ten knowledge domains at temperature zero (22,500 turns), judged by a dual-track automated evaluator validated against a human gold standard (Cohen's kappa = 1.000 for binary adoption; 0.92 linear-weighted for collapse severity). GPT-5.4 Mini recorded zero adoptions across all 500 sessions; a base-model logit probe shows its decision margin is perturbed but large and finite. Gemini-3.1 Flash-Lite followed a steep authority gradient: 0.1% adoption for self-attributed falsehoods, 23.5% for user-cited sources, 68.2% for s
    
[^139]: 基于状态条件化潜在引导与进展指导的高效推理探索

    Efficient Reasoning Exploration via State-Conditioned Latent Steering with Progress Guidance

    [https://arxiv.org/abs/2609.24066](https://arxiv.org/abs/2609.24066)

    提出了一种无需训练的潜在引导框架SPS，通过构建包含进展引导向量的状态条件化方向库，在推理时引导模型探索能取得有意义进展的多样化推理路径，从而缓解探索坍缩问题并提升Best-of-N推理的探索效率。

    

    Best-of-N 是一种广泛用于复杂推理的推理策略，其有效性取决于采样得到的候选能否覆盖多样化且高质量的推理路径。然而，经过后训练的推理模型常常出现“探索坍缩”（exploration collapse）问题，即独立的多次采样反复遵循相似的推理路径，从而限制了增加采样预算所带来的收益。现有方法通过促进更广泛的探索来缓解这一问题，但并未显式引导探索朝着能取得有意义进展的延续方向进行，导致探索效率有限。为解决这一问题，我们提出了状态条件化进展引导（State-conditioned Progress-guided Steering, SPS），这是一个无需训练的潜在引导框架。具体而言，SPS 构建了一个状态条件化的方向库，其中包含针对不同前缀状态区域的多个进展引导向量。在在线推理过程中，SPS 重新（检索）……

    arXiv:2609.24066v1 Announce Type: new  Abstract: Best-of-$N$ is a widely used inference strategy for complex reasoning, whose effectiveness depends on whether sampled candidates can cover diverse and high-quality reasoning paths. However, post-trained reasoning models often suffer from \emph{exploration collapse}, where independent rollouts repeatedly follow similar reasoning paths and limit the gains from increasing the rollout budget. Existing methods alleviate this issue by promoting broader exploration, but do not explicitly guide exploration toward continuations that make meaningful progress, resulting in limited exploration efficiency. To address this, we propose \emph{\underline{S}tate-conditioned \underline{P}rogress-guided \underline{S}teering} (SPS), a training-free latent steering framework. Specifically, SPS constructs a state-conditioned Direction Bank containing multiple progress-guided steering vectors for different prefix-state regions. During online inference, SPS retr
    
[^140]: 通过适配器唤醒编码器：语音大语言模型的有效领域自适应微调

    Encoder Awakening via Adapters: Effective Domain-Adaptive Fine-tuning of Speech-LLMs

    [https://arxiv.org/abs/2609.17981](https://arxiv.org/abs/2609.17981)

    提出EAVA方法，通过在语音大语言模型的每个编码器层插入轻量级适配器并进行专门训练，在保留预训练知识的同时注入目标领域声学知识，有效提升有限数据下儿童语音、方言语音等领域偏移语音的识别性能。

    

    语音大语言模型通常由预训练的语音编码器、模态投影器和经低秩适配器（LoRA）微调的大语言模型构建而成，在通用领域语音上展现出强大的自动语音识别（ASR）性能。然而，在目标领域数据有限的情况下，将其适应到存在领域偏移的语音（如儿童语音或方言语音）仍然具有挑战性。鉴于大语言模型在语音大语言模型中的主导地位，且交叉熵损失仅在大语言模型输出端应用，语音编码器可能无法充分适应新的声学条件。本文提出了通过适配器唤醒编码器（EAVA），这是一种简单而有效的针对基于语音大语言模型的ASR的领域自适应微调方法。首先，在每个编码器层中插入轻量级适配器并进行专门训练，使目标领域的声学知识能够融入编码器，同时保留其预训练知识。

    arXiv:2609.17981v1 Announce Type: cross  Abstract: Speech Large Language Models (Speech-LLMs), typically built from a pre-trained speech encoder, a modality projector, and an LLM fine-tuned with Low-Rank Adapters (LoRA), have shown strong Automatic Speech Recognition (ASR) performance on general-domain speech. However, adapting them to domain-shifted speech, such as child or dialectal speech, remains challenging under limited target-domain data. Given the dominant role of the LLM in Speech-LLMs, with cross-entropy loss applied only at the LLM output, the speech encoder may receive insufficient adaptation to new acoustic conditions. In this paper, we propose Encoder Awakening via Adapters (EAVA), a simple yet effective domain-adaptive fine-tuning method for Speech-LLM-based ASR. First, lightweight adapters are inserted into each encoder layer and trained exclusively, enabling target-domain acoustic knowledge to be incorporated into the encoder while preserving its pre-trained knowledge.
    
[^141]: ResidencyRL：模拟临床环境中的强化学习

    ResidencyRL: Reinforcement Learning in Simulated Clinical Environments

    [https://arxiv.org/abs/2608.07418](https://arxiv.org/abs/2608.07418)

    ResidencyRL提出了一种强化学习方法，通过模拟多轮临床诊疗环境（含对抗性患者模拟器和结构化诊断奖励）来训练临床AI智能体，使其像住院医师一样在实践互动中提升完整临床决策序列的能力。

    

    在医学教育中，医生通过住院医师培训将学术知识转化为临床专长：历经数年的训练、数千次的诊疗实践，获得多样化的反馈来源，并逐步获得更大的自主权。临床推理在很大程度上依赖于患者诊疗过程——这是一种对话式的互动，临床医生在其中采集病史、完善诊断假设，并在不确定性下做出治疗决策。虽然大语言模型（LLM）在静态医学基准测试中表现出色，但优化完整临床决策序列的方法仍然发展不足。我们提出了ResidencyRL，这是一种强化学习方法，通过模拟的多轮临床诊疗过程（每次轨迹最多包含60轮对话和8次工具调用）来训练临床人工智能（AI）智能体。ResidencyRL将策略智能体与能够表现出复杂、对抗性行为的LLM模拟器配对，通过与诊断准确性对齐的结构化奖励进行训练。

    arXiv:2608.07418v2 Announce Type: replace  Abstract: In medical education, physicians convert academic knowledge into clinical expertise through residency: years of training across thousands of encounters, with diverse sources of feedback and progressively greater autonomy. Much of clinical reasoning relies on the patient encounter, a dialogue in which a clinician elicits history, refines diagnostic hypotheses, and decides management under uncertainty. While large language models (LLMs) excel on static medical benchmarks, methods to optimize the full sequence of clinical decisions remain underdeveloped. We present ResidencyRL, a reinforcement learning (RL) method for training clinical artificial intelligence (AI) agents through simulated multi-turn clinical encounters (up to 60 dialogue turns and 8 tool calls per trajectory). ResidencyRL pairs the policy agent with LLM simulators capable of complex, adversarial behaviors, training against a structured reward aligned to diagnostic accur
    
[^142]: M3-DuplexBench：面向全双工语音对话模型的多轮、多语言、多领域基准测试

    M3-DuplexBench: A Multi-Turn, Multilingual, Multidomain Benchmark for Full-Duplex Spoken Dialogue Models

    [https://arxiv.org/abs/2607.29125](https://arxiv.org/abs/2607.29125)

    该论文提出了M3-DuplexBench，一个用于全双工语音对话系统的多轮、多语言（英语与日语）、多领域基准测试，可联合评估交互时机与响应内容，并支持跨语言比较。

    

    全双工语音对话系统（FDSDS）能够在说话的同时进行聆听，从而实现自然的对话行为，例如流畅的话轮转换、附和处理（backchannel）以及用户插话处理。然而，在多轮交互中对FDSDS进行受控比较仍然具有挑战性，且现有基准测试在多种语言、对话设置、交互时机和响应内容的联合覆盖方面较为有限。我们提出了M3-DuplexBench，这是一个用于FDSDS受控评估的多轮、多语言、多领域基准测试。该基准测试在英语和日语下，针对闲聊对话和多轮问答任务，联合评估交互时机与响应内容。英语和日语的问答数据源自同一源语料库，从而支持跨语言的时机与内容性能比较。此外，我们还提供了一个基于参考的多轮设置，作为检验……的可选诊断手段（原文摘要至此截断）。

    arXiv:2607.29125v2 Announce Type: replace  Abstract: Full-duplex spoken dialogue systems (FDSDSs) can listen while speaking, enabling natural conversational behaviors such as smooth turn-taking, backchannel handling, and user barge-in handling. However, controlled comparisons of FDSDSs in multi-turn interaction remain challenging, and existing benchmarks offer limited joint coverage of multiple languages, dialogue settings, interaction timing, and response content. We propose M3-DuplexBench, a multi-turn, multilingual, multidomain benchmark for controlled evaluation of FDSDSs. The benchmark jointly evaluates interaction timing and response content in English and Japanese across casual conversation and multi-turn question answering. The English and Japanese question-answering data are derived from a common source corpus, enabling cross-lingual comparisons of timing and content performance. We also provide a reference-conditioned multi-turn setting as an optional diagnostic for examining
    
[^143]: TokenSwap：多模态大语言模型中模态差距的基准测试与缩小

    TokenSwap: Benchmarking and Reducing the Modality Gap in Multimodal LLMs

    [https://arxiv.org/abs/2607.28640](https://arxiv.org/abs/2607.28640)

    本文提出TokenSwap方法，通过将文本基准（如MMLU）中的文本概念替换为语义对齐的图像，构建出图像交错的TokenSwap-Bench基准，首次系统揭示了42个多模态大语言模型中普遍存在的模态差距，即图像交错输入相比纯文本输入平均导致19.6%的性能下降。

    

    多模态大语言模型（MLLM）应当表现出跨模态一致性，即在语义等价但模态不同的输入下生成一致的响应。然而，我们观察到模型预测在这类跨模态变化下存在系统性差异。具体而言，我们将“模态差距”定义为模型在语义等价的文本输入与多模态输入下的性能差异。我们提出TokenSwap方法，通过用语义对齐的图像替换文本概念来构建此类输入，从而得到视觉token与文本token相互交错的序列。基于TokenSwap，我们将现有的基于文本的基准（如MMLU）转换为图像交错的对应版本，构建出TokenSwap-Bench。在对42个多模态大语言模型的评测中，我们观察到普遍存在的模态差距：当输入从纯文本转变为图像交错形式时，模型性能下降4.2%至47.4%，平均下降19.6%（±3.3%）。

    arXiv:2607.28640v2 Announce Type: replace  Abstract: Multimodal large language models (MLLMs) should exhibit cross-modal consistency, generating consistent responses given semantically equivalent inputs across modalities. However, we observe a systematic discrepancy in model predictions under such cross-modal variations. Specifically, we define the modality gap as the difference in model performance under semantically equivalent textual and multimodal inputs. We introduce TokenSwap, a method that constructs such inputs by replacing textual concepts with semantically aligned images, resulting in sequences where visual tokens are interleaved with text tokens. Based on TokenSwap, we transform existing text-based benchmarks (e.g., MMLU) into image-interleaved counterparts, resulting in TokenSwap-Bench. Across 42 MLLMs, we observe a pervasive modality gap, with performance decreasing by 4.2% to 47.4% when moving from text-only to image-interleaved inputs, averaging 19.6% +/- 3.3% across mod
    
[^144]: ZenGen：大语言模型的社会心智

    ZenGen: Social Mind for LLMs

    [https://arxiv.org/abs/2607.23740](https://arxiv.org/abs/2607.23740)

    本文提出了ZenGen框架，通过SoMBench基准测量大语言模型的社会智能，并采用诊断驱动的训练方案来内化和提升其社会认知能力。

    

    随着大语言模型从孤立的任务解决向人类环境中的长期服务转变，它们需要社会智能：即推断心理状态、追踪社会关系、基于规范推理以及根据情境调整行为的能力。本报告介绍了ZenGen，一个用于测量、内化和落地社会智能的集成框架。在测量方面，我们提出了SoMBench，一个基于心理学构建的基准，涵盖3个主要维度、17个次要维度和71个任务范式。它控制了问题格式、叙事视角和上下文长度，跨越284个共享场景和3,481个专家验证实例。对20个代表性LLM的评估显示出巨大的提升空间：最佳模型仅达到72.08%的总体准确率，且17个次要维度中没有一个达到90%的接近天花板区间。在内化方面，我们开发了ZenGen，一种基于诊断驱动的训练方案，结合了监督式（此处原文截断，但根据上下文推测为“监督式微调”等）方法。

    arXiv:2607.23740v2 Announce Type: replace  Abstract: As large language models move from isolated task solving toward long-term service in human environments, they require social intelligence: the ability to infer mental states, track social relations, reason over norms, and adapt behavior under context. This report presents ZenGen, an integrated framework for measuring, internalizing, and grounding social intelligence. For measurement, we introduce SoMBench, a psychology-grounded benchmark spanning 3 primary dimensions, 17 secondary dimensions, and 71 task paradigms. It controls question format, narrative perspective, and context length across 284 shared scenarios and 3,481 expert-verified instances. Evaluation of 20 representative LLMs reveals substantial headroom: the best model achieves only 72.08% overall accuracy, and none of the 17 secondary dimensions reaches the 90% near-ceiling band. For internalization, we develop ZenGen, a diagnosis-driven training recipe combining supervise
    
[^145]: 标签监督下医学基础模型的表征收敛程度更低

    Medical foundation models converge less under label supervision

    [https://arxiv.org/abs/2607.20274](https://arxiv.org/abs/2607.20274)

    该论文通过对75万余张医学图像、18个公开模型和101个自训练模型的大规模分析发现，医学基础模型——尤其是经过标签监督训练的模型——对相同图像的表征彼此差异巨大，其近邻重叠度甚至接近随机初始化的水平，因此不同基础模型的嵌入不可互换使用。

    

    诊断分类器和影像生物标志物通常拟合在医学基础模型的嵌入表示之上，当新版本的模型出现时，这些基础模型就会被替换。这种做法隐含了一个假设：不同模型对图像的表征是相似的。我们用来自5种成像模态、14个数据集的超过75万张图像、18个公开模型，以及101个在胸片和组织病理学图像上训练的模型（这些模型在预训练目标、标签类型、模型规模、随机种子、初始化、训练患者或训练配方上各不相同）检验了这一假设。模型间的一致性用相互k近邻重叠度来衡量，即一张图像的最近邻中有多少是两个模型所共有的。两个公开模型在胸片图像的10个最近邻中平均仅有0.091的重叠，而随机初始化的网络与它们也有0.036的重叠。在其他四种成像模态中，重叠度至多为0.300。在胸片上，一致性更多取决于源数据集和……（原文在此处截断）

    arXiv:2607.20274v3 Announce Type: replace-cross  Abstract: Diagnostic classifiers and imaging biomarkers are fitted on the embeddings of medical foundation models. These models are replaced as new versions appear. This practice assumes that different models represent images alike. We tested this assumption with more than 750,000 images from 14 datasets in 5 imaging modalities, 18 public models, and 101 models trained on chest radiographs and histopathology that differ in pretraining objective, label type, size, random seed, initialization, training patients, or training recipe. Agreement was measured as the mutual k-nearest-neighbor overlap, the share of an image's nearest neighbors common to two models. Two public models shared on average 0.091 of the 10 nearest neighbors of a chest radiograph. A randomly initialized network shared 0.036 with them. In the four other modalities, they shared at most 0.300. On chest radiographs, agreement depended more on the source dataset and the radio
    
[^146]: 单词普查：44个语言模型中的答案选择趋同性

    The One-Word Census: Answer-Choice Conformity Across 44 Language Models

    [https://arxiv.org/abs/2607.12796](https://arxiv.org/abs/2607.12796)

    本研究通过96个开放式单词选择提示对105个语言模型进行了“答案趋同性”普查，发现模型的选择高度趋同——例如46%的模型在“选一个词”时都选了"serendipity"——且这种趋同并非由小型模型或个性化微调模型所致，反而轻度后训练和个性化微调的模型分歧最大。

    

    当一个语言模型必须从大量同样有效的选项中选择一个答案时，它会选择哪个答案，又有多少时候它与所有其他模型的选择一致？当被要求“选一个词”时，来自二十多个实验室的105个语言模型有46%的概率选择了“serendipity”这个词。我们通过96个单轮提示来测量这种趋同性以及每个模型在其中的贡献，每个提示指定一个包含许多有效单词答案的类别（如“说出一种树的名字。”），每个模型被询问八次，并以精确匹配进行评分，不使用嵌入模型，也不使用评判模型。模型的“答案选择惊异度”定义为其答案在所有其他模型汇总答案下的平均 -log2 概率。在96个类别中，有28个类别里单个答案占据了所有答案的至少80%。这种集中度并非源于小型模型或个性化微调模型：87个主要实验室的模型的集中程度至少与整个领域相当。轻度后训练和个性化微调的模型的分歧最大（原文在此处截断）。

    arXiv:2607.12796v3 Announce Type: replace-cross  Abstract: When a language model must choose one answer from a large space of equally valid options, which answer does it choose, and how often is it the answer every other model chooses? Asked to "pick a word," 105 language models from more than twenty labs chose serendipity 46% of the time. We measure this convergence, and each model's share in it, with 96 single-turn prompts that each name a category with many valid one-word answers ("Name a tree."), asked eight times per model and scored by exact match, with no embeddings and no judge. A model's answer-choice surprisal is the average -log2 probability of its answers under the pooled answers of all other models. In 28 of 96 categories a single answer takes at least 80% of all answers. The concentration does not depend on small or persona-tuned models: the 87 major-lab models are at least as concentrated as the full field. Lightly post-trained and persona-tuned models are the most diver
    
[^147]: 也门伊比阿拉伯语中wh一致关系的句法研究

    The syntax of wh-agreement in Yemeni Ibbi Arabic

    [https://arxiv.org/abs/2607.04986](https://arxiv.org/abs/2607.04986)

    本文提出基于特征继承机制的跨语段一致分析方法，论证也门伊比阿拉伯语中由第三人称代词语法化而来的后缀充当wh一致关系的形态标记，为生成句法学提供了独特的实证贡献。

    

    本文研究了也门伊比阿拉伯语（YIA）句法中的一个重要现象，即wh一致关系，这一现象在希腊语、印度尼西亚语、Lubukusu语、爱尔兰语等多种语言中普遍存在。在YIA中，wh一致关系通过Wh算子、C、T/V、v上的一致性屈折变化来体现。为了解释这一现象，我们提出了一种基于特征继承（FI）机制的跨语段一致（AAP）分析方法，其中作为匹配的一致（AM）与特征赋值（FV）在一定程度上相互分离：AM涉及C/v，而FV涉及T/V。通过分析Wh算子、功能中心语（具体为C、(T)、v）与动词复合体之间观察到的一致模式，我们认为后缀-eh、-uh、-nen、-um经历了从标准阿拉伯语（SA）第三人称代词语法化的过程，充当wh一致关系的形态标记。研究结果表明，YIA语料为生成句法学提供了独特的实证贡献。

    arXiv:2607.04986v2 Announce Type: replace  Abstract: This article tackles an important phenomenon in the syntax of Yemeni Ibbi Arabic (YIA), viz., wh-agreement, a phenomenon common to several languages including Greek, Indonesian, Lubukusu, Irish, etc. In YIA, wh-agreement manifests itself via agreement inflections on the Wh-Op, C, T/V, v. To account for this phenomenon, we propose an Agree across phases (AAP) approach anchored in the mechanism of Feature Inheritance (FI) in which Agree as MATCHING (AM) is a bit separated from feature valuation (FV). AM concerns Cs/vs, but FV Ts/Vs. Analyzing the agreement patterns observed between Wh-Op(erators), functional heads (precisely C, (T), v), and verbal complexes, we argue that the suffixes -eh, -uh, -nen, -um, having undergone grammaticalization process from Stannard Arabic (SA) third person pronouns, function as morphological marking of wh-agreement. Findings indicate that YIA data offer a unique empirical contribution to generative syntax
    
[^148]: KaLM-Reranker-V1：面向压缩文档重排序的快速但非晚期交互方法

    KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking

    [https://arxiv.org/abs/2606.22807](https://arxiv.org/abs/2606.22807)

    KaLM-Reranker-V1 是一种快速但非晚期交互的重排序器，通过编码器-解码器架构将查询与段落计算解耦，利用 Matryoshka 嵌入池化离线预编码段落，并借助交叉注意力捕捉细粒度相关性，在保持强表达能力的同时显著提升了部署效率与灵活性。

    

    随着检索系统规模的不断扩大，高效且有效的重排序变得日益重要。然而，大多数现有的基于编码器和解码器的重排序器都会联合处理每一个查询-段落对，将其在线计算紧密耦合，从而限制了部署的效率与灵活性。我们提出了 KaLM-Reranker-V1，一种快速但非晚期交互的 FBNL 重排序器，它在解耦查询与段落计算的同时，保留了富有表现力的相关性建模能力。KaLM-Reranker-V1 构建于编码器-解码器架构之上，利用 Matryoshka（套娃）嵌入池化对段落进行预编码，同时其解码器将系统指令、用户指令与查询意图共同建模；随后通过交叉注意力捕捉所得查询上下文与段落表示之间的细粒度相关性。这些设计共同带来了四大关键优势：(i) 离线段落编码带来的高效性，(ii) 交叉注意力带来的强表达能力，(iii) Matryoshka 嵌入带来的紧凑性

    arXiv:2606.22807v3 Announce Type: replace  Abstract: As retrieval systems scale, effective and efficient reranking becomes increasingly important. However, most existing encoder- and decoder-based rerankers jointly process every query--passage pair, tightly coupling their online computation and limiting deployment efficiency and flexibility. We present KaLM-Reranker-V1, a fast but not late-interaction FBNL reranker that decouples query and passage computation while retaining expressive relevance modeling. Built on an encoder--decoder architecture, KaLM-Reranker-V1 pre-encodes passages using Matryoshka embedding pooling, while its decoder models system and user instructions together with query intent; cross-attention then captures fine-grained relevance between the resulting query context and passage representations. Together, these designs offer four key advantages: (i) efficiency from offline passage encoding, (ii) expressiveness from cross-attention, (iii) compactness from Matryoshka
    
[^149]: PACT：面向多轮工具使用智能体的特权轨迹协同训练框架

    PACT: Privileged Trace Co-Training for Multi-Turn Tool-Use Agents

    [https://arxiv.org/abs/2606.16215](https://arxiv.org/abs/2606.16215)

    PACT框架提出将专家轨迹仅作为训练时的优化信号而非推理提示，通过轨迹条件化强化学习代理目标和带退火过程监督的组件感知SFT损失两个互补信号进行协同训练，同时保持推理时仅使用提示词生成，从而兼顾强化学习的推理一致性与监督微调的密集过程监督优势。

    

    多轮工具使用智能体需要在多轮交互过程中进行推理、调用工具并适应观测结果。对此类智能体进行后训练极具挑战性：强化学习虽然与仅提示词的推理设置相匹配，但常常面临奖励稀疏和信用分配能力弱的问题；而基于专家轨迹的监督微调虽然能提供密集的过程监督，却可能过度约束模型，使其拘泥于固定轨迹。为解决这一问题，我们提出了PACT——一个面向多轮工具使用智能体的特权轨迹协同训练框架。其核心思想是仅将专家轨迹用作训练时的优化信号，而非推理时的提示。PACT在生成rollout时保持仅提示词模式，随后通过两个互补的信号利用专家轨迹来引导优化：一是轨迹条件化的强化学习代理目标，它在专家轨迹的上下文中评估仅提示词的rollout；二是组件感知的监督微调损失，它提供退火式的过程监督

    arXiv:2606.16215v2 Announce Type: replace-cross  Abstract: Multi-turn tool-use agents must reason, call tools, and adapt to observations across several interaction turns. Post-training such agents is challenging, as reinforcement learning often suffers from sparse rewards and weak credit assignment despite matching the prompt-only inference setting, while supervised fine-tuning on expert traces provides dense process supervision but can over-constrain the model to fixed trajectories. To tackle this, we propose PACT, a Privileged trAce Co-Training framework for multi-turn tool-use agents. The key idea is to use expert traces only as training-time optimization signals rather than rollout-time hints. PACT keeps rollout generation prompt-only, then uses expert traces to guide optimization through two complementary signals: a trace-conditioned RL surrogate that evaluates prompt-only rollouts under expert-trace context, and a component-aware SFT loss that provides annealed process supervisio
    
[^150]: 让大语言模型互相评判：面向医学问答的多智能体同行评审推理

    Let Large Language Models Judge Each Other: Multi-Agent Peer-Reviewed Reasoning for Medical Question Answering

    [https://arxiv.org/abs/2606.15419](https://arxiv.org/abs/2606.15419)

    该论文提出多智能体同行评审推理方法，让多个大语言模型在医学问答中互相评审彼此的思维链推理并择优生成最终答案，显著提升了准确性、可解释性和鲁棒性。

    

    目标：提升大语言模型（LLMs）在医学问答（MedQA）任务中的准确性、可解释性和鲁棒性。方法：我们设计了一种多智能体同行评审推理方法，其中多个大语言模型智能体独立生成包含候选答案的思维链推理，随后充当同行评审，对彼此的推理进行事实正确性和逻辑严密性的评估，并选择评分最高的推理链来产生最终答案。实验使用了五个最先进的大语言模型在三个基准数据集上进行，并与单模型思维链推理以及基于思维链的多数投票方法进行了性能对比。结果：同行评审推理方法在所有实验中均一致优于两种基线方法，最佳模型组合的平均准确率达到0.820...

    arXiv:2606.15419v2 Announce Type: replace-cross  Abstract: Objective: To enhance the accuracy, interpretability, and robustness of large language models (LLMs) in medical question answering (MedQA).   Method: We designed a multi-agent peer-reviewed reasoning method in which multiple LLM agents independently generate chain-of-thought reasoning with candidate answers, then act as peer reviewers to evaluate each other's reasoning for factual correctness and logical soundness. The highest-rated reasoning chain is selected to produce the final answer. Experiments were conducted with five state-of-the-art LLMs (Llama-3.1-8B, Qwen2.5-7B, Phi-4, DeepSeek-LLM-7B, GPT-oss-20B) on three benchmark datasets: HeadQA, MedQA-USMLE, and PubMedQA. Performance was compared against single-model chain-of-thought reasoning and chain-of-thought-based majority voting.   Results: Peer-reviewed reasoning consistently outperformed both baselines. The best model combination achieved an average accuracy of 0.820 a
    
[^151]: 经验造就技能：通过自我演化的技能记忆实现可泛化的医疗智能体推理

    Experience Makes Skillful: Enabling Generalizable Medical Agent Reasoning via Self-Evolving Skill Memory

    [https://arxiv.org/abs/2606.09365](https://arxiv.org/abs/2606.09365)

    提出SkeMex框架，将医疗智能体的交互轨迹提炼为结构化的技能记忆，在不更新模型权重的情况下实现自我演化，从而支持可泛化的长程临床推理。

    

    医疗智能体系统正日益被期望支持交互式临床决策，而不仅仅是静态的问答。在这种场景下，有效的智能体必须能够在不断演变的病例中复用先前的经验，然而现有的记忆机制往往保留原始的历史轨迹，这些轨迹冗余、嘈杂且难以管理。更重要的是，它们很少区分哪些记忆对未来的推理真正有用。这限制了智能体为长程临床推理积累紧凑且可靠经验的能力。为弥补这一差距，我们提出了SkeMex，一个部署后的自我演化框架，它通过基于技能的记忆来改进医疗智能体，而无需更新模型权重。SkeMex将信息丰富的交互轨迹提炼为编码可复用程序性知识的结构化技能，并将它们组织成一个涵盖通用、任务特定和动作层面经验的多分支知识库。

    arXiv:2606.09365v4 Announce Type: replace  Abstract: Medical agent systems are increasingly expected to support interactive clinical decision making rather than only static question answering. In such settings, effective agents must reuse prior experience across evolving cases, yet existing memory mechanisms often retain raw historical traces that are redundant, noisy, and difficult to govern. More importantly, they rarely distinguish which memories are truly useful for future reasoning. This limits their ability to accumulate compact and reliable experience for long-horizon clinical reasoning. To close this gap, we propose SkeMex, a post-deployment self-evolution framework that improves medical agents through a skill-based memory without updating model weights. SkeMex distills informative interaction trajectories into structured skills that encode reusable procedural knowledge, and organizes them into a multi-branch repository spanning general, task-specific, and action-level experien
    
[^152]: 智能体的最后考试

    Agents' Last Exam

    [https://arxiv.org/abs/2606.05405](https://arxiv.org/abs/2606.05405)

    本文提出由250多位行业专家共同打造的Agents' Last Exam（ALE）基准，通过13个行业集群、55个子领域、1千多项长周期、结果可验证的真实经济价值任务来评估AI智能体，并发现当前系统在最难任务层级上仍远未饱和。

    

    arXiv:2606.05405v3 公告类型：替换 摘要：近年来的人工智能系统已在各类基准测试中取得了优异的成绩，但这些成果并未转化为在众多专业领域中的经济价值意义上的实际部署。我们认为，这一差距在很大程度上是一个评估问题：广泛使用的基准测试缺乏对真实的、具有经济价值的工作流程的持续性性能测量。本文提出了 Agents' Last Exam（ALE），这是一个旨在评估AI智能体在长周期、具有经济价值、结果可验证的真实世界任务上表现的基准。该基准与250多位行业专家合作开发，参考O*NET / SOC 2018（美国联邦职业分类体系）定义了非实体行业，并围绕一个任务分类体系组织，该体系包含55个子领域、归为13个行业集群，涵盖1千多项任务。当前结果显示，最难的任务层级仍远未达到饱和：在主流的运行框架和底层模型配置下，平均……

    arXiv:2606.05405v3 Announce Type: replace  Abstract: Recent AI systems have achieved strong results on a wide range of benchmarks, yet these gains have not translated into economically meaningful deployment across many professional domains. We argue that this gap is largely an evaluation problem: widely used benchmarks lack sustained performance measurement on real and economically valuable workflows. This paper introduces Agents' Last Exam (ALE), a benchmark designed to evaluate AI agents on long horizon, economically valuable, real world tasks with verifiable outcomes. Developed in collaboration with 250+ industry experts, ALE covers non-physical industries defined with reference to O*NET / SOC 2018 (the U.S. federal occupational taxonomy). It is organized around a task taxonomy with 55 sub fields grouped into 13 industry clusters covering 1K+ tasks. Current results show that the hardest tier remains far from saturated: across mainstream harness and backbone configurations, the avera
    
[^153]: EchoDistill：通过噪声到干净的自蒸馏实现鲁棒的大型音频语言模型

    EchoDistill: Robust Large Audio Language Models via Noisy-to-Clean Self-Distillation

    [https://arxiv.org/abs/2605.23954](https://arxiv.org/abs/2605.23954)

    EchoDistill提出一种噪声到干净的自蒸馏框架，在后训练中以干净音频作为特权信息，通过掩码响应token蒸馏、任务门控一致性塑形和教师参考的组相对优化，使大型音频语言模型在噪声环境下更鲁棒，且推理时无额外开销。

    

    大型音频语言模型（LALMs）仍然容易受到声学噪声的影响，噪声会掩盖与任务相关的证据并产生不可靠的响应。我们提出了EchoDistill，这是一种噪声到干净的自蒸馏框架，在后训练阶段将干净音频作为特权信息加以利用。以噪声输入的学生模型采样能够反映其推理时行为的候选响应，而同一骨干网络的冻结副本则处理对应的干净音频。EchoDistill结合了掩码响应token蒸馏、任务门控一致性塑形以及教师参考的组相对优化，使噪声输入下的生成结果与干净条件下的语义保持对齐。推理时仅保留学生模型，不引入任何额外的推理开销。在三个LALM骨干网络和三个音频领域、信噪比为-10dB的条件下，EchoDistill相比最强基线将噪声输入下的平均准确率提升了1.63个百分点。在Qwen2.5-Omni上，它

    arXiv:2605.23954v2 Announce Type: replace-cross  Abstract: Large Audio Language Models (LALMs) remain vulnerable to acoustic noise, which can obscure task-relevant evidence and produce unreliable responses. We propose EchoDistill, a noisy-to-clean self-distillation framework that uses clean audio as privileged information during post-training. A noisy-input student samples candidate responses reflecting its inference-time behavior, while a frozen copy of the same backbone processes the corresponding clean audio. EchoDistill combines masked response-token distillation, task-gated consistency shaping, and teacher-referenced group-relative optimization to align noisy-input generation with clean-conditioned semantics. Only the student is retained at inference time, introducing no additional inference cost. Across three LALM backbones and three audio domains at -10dB, EchoDistill improves average noisy-input accuracy by 1.63 percentage points over the strongest baseline. On Qwen2.5-Omni, it
    
[^154]: 概率校准是语言模型中一种可训练的能力

    Probabilistic Calibration Is a Trainable Capability in Language Models

    [https://arxiv.org/abs/2605.11845](https://arxiv.org/abs/2605.11845)

    该论文提出校准微调（包括软目标与硬目标两种变体），证明语言模型从隐式分布中进行采样的概率校准能力可以通过微调直接训练获得，并在12个模型上显著提升了结构化采样的保真度。

    

    语言模型越来越多地被应用于输出需要遵循预期分布的场景，但其生成概率与该分布的校准程度往往较差。在许多此类请求中，分布是隐式的，且有效的输出无法事先列举，因此该任务无法交给外部随机数生成器来完成。我们研究了这种能力能否通过微调直接得到提升。具体而言，我们在需要从数学分布中采样的合成提示上对语言模型进行微调，并比较了两种校准微调（Calibration Fine-Tuning）变体：一种是将目标输出分布转换为基于前缀树（trie）导出的下一词元目标的软目标方法，另一种是在从同一目标分布中采样得到的完整输出上进行训练的硬目标方法。在涵盖四个系列的12个模型上，两种方法都在保留分布族和未见参数上显著提升了结构化采样的保真度。

    arXiv:2605.11845v2 Announce Type: replace  Abstract: Language models are increasingly used in settings where outputs should follow an intended distribution, yet their generation probabilities are often poorly calibrated to it. In many such requests the distribution is implicit and the valid outputs cannot be listed in advance, so the task cannot be passed to an external random number generator. We study whether this capability can be improved directly through fine-tuning. Concretely, we fine-tune language models on synthetic prompts that require sampling from mathematical distributions, and compare two Calibration Fine-Tuning variants: a soft-target method that converts the desired output distribution into trie-derived next-token targets, and a hard-target method that trains on sampled completions from the same target distribution. Across 12 models spanning four families, both methods substantially improve structured-sampling fidelity on held-out distribution families and unseen parame
    
[^155]: MemReread：通过记忆引导的重读增强智能体长上下文推理

    MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading

    [https://arxiv.org/abs/2605.10268](https://arxiv.org/abs/2605.10268)

    MemReread通过在最终记忆不充分时触发问题分解与针对性重读，恢复流式阅读中被过早丢弃的间接证据，从而在不依赖中间检索的情况下增强智能体的长上下文非线性推理能力，并结合强化学习提升长度外推能力和重读次数的自适应决策。

    

    长上下文推理任务推动了智能体记忆方法的发展，这类方法按顺序处理文档块并维护一个动态更新的记忆，从而避免了标准注意力机制的二次方复杂度。现有方法通常依赖检索来恢复记忆覆盖过程中被丢弃的信息，但在记忆形成过程中仍然容易丢失证据，并且容易受到无效查询的干扰。为了克服这些局限性，我们提出了MemReread。该方法建立在流式阅读之上，规避了中间检索，当最终记忆不足以回答问题时，会触发问题分解和针对性重读。通过恢复可能被过早丢弃的间接证据，MemReread在保持文档理解逻辑流程的同时实现了非线性推理。我们进一步引入了一个强化学习框架，该框架提升了长度外推能力，并能自适应地确定重读的次数。

    arXiv:2605.10268v2 Announce Type: replace-cross  Abstract: Long-context reasoning tasks have motivated agent-memory approaches that process document chunks sequentially while maintaining a dynamically updated memory, avoiding the quadratic complexity of standard attention. Existing methods often rely on retrieval to recover information discarded during memory overwriting, but remain vulnerable to evidence loss during memory formation and interference from invalid queries. To overcome these limitations, we propose MemReread. Built upon streaming reading, it circumvents intermediate retrieval, triggering question decomposition and targeted rereading when the final memory is insufficient. By recovering indirect evidence that may have been prematurely discarded, MemReread enables non-linear reasoning while preserving the logical flow of document comprehension. We further introduce a reinforcement learning framework that improves length extrapolation and adaptively determines the number of 
    
[^156]: AI写作辅助如何扭曲人们对AI用户及其观点的认知

    How AI Writing Assistance Distorts Perceptions of AI Users and Their Opinions

    [https://arxiv.org/abs/2604.22503](https://arxiv.org/abs/2604.22503)

    AI写作辅助会让读者对写作者产生普遍而持久的认知扭曲（显得更有主见、更有能力、更具社会优势），即使写作者本人反对且被明确告知，这些扭曲依然存在，而且扭曲越强反而越能增强说服力。

    

    数亿人使用AI进行写作辅助。我们研究了AI如何扭曲人们对AI用户及其观点的认知。写作者（N=2,939）在有和没有AI辅助的情况下撰写政治观点段落。另一组读者（N=11,091）在不知情的情况下评估这些段落并对写作者做出推断。在AI辅助下，写作者显得更有主见、更有能力、态度更积极，且在社会人口学上更具优势。写作者对许多此类扭曲表示反对，但即使知晓这些扭曲的存在，他们仍然偏好AI辅助的文本。基于我们数据训练的奖励模型减轻了这些令人反感的扭曲，却降低了用户的接受度。另一批读者（N=8,798）更信任使用AI辅助的写作者，并且当AI的扭曲性更强时，他们反而更容易被AI写作说服。总体而言，AI写作辅助带来的认知扭曲是普遍存在的，在人类监督下依然持续，并且很可能对人类的行为和态度产生深远影响。

    arXiv:2604.22503v3 Announce Type: replace  Abstract: Hundreds of millions of people use AI for writing assistance. We asked how AI distorts perceptions of AI users and their opinions. Writers (N=2,939) wrote political opinion paragraphs with and without AI assistance. Separate readers (N=11,091) blindly evaluated these paragraphs and made inferences about the writers. With AI, writers seemed more opinionated, competent, positive, and demographically privileged. Writers objected to many of these distortions yet preferred AI-assisted text even when made aware of them. Reward models trained on our data mitigated objectionable distortions but reduced user acceptance. Additional readers (N=8,798) trusted AI-assisted writers more and were more persuaded by AI writing when AI was more distortive. Overall, distortions from AI writing assistance are pervasive, persistent under human oversight, and likely to have consequential effects on human behaviours and attitudes.
    
[^157]: 先复制，后翻译：解读多语言预训练中的翻译动力学

    Copy First, Translate Later: Interpreting Translation Dynamics in Multilingual Pretraining

    [https://arxiv.org/abs/2604.17633](https://arxiv.org/abs/2604.17633)

    该研究通过对一个1.7B模型在九种语言上进行细粒度检查点的预训练分析，揭示多语言翻译能力的发展分为两个阶段：早期以词元复制和表层相似性为主，后期才逐步形成真正的翻译能力。

    

    大语言模型展现出令人印象深刻的跨语言能力。然而，以往的工作通过孤立的因素和训练过程中稀疏的观察点来分析这一现象，限制了我们对跨语言泛化如何产生的理解——尤其是在学习的早期阶段。为了研究语言能力与翻译能力的早期发展轨迹，我们在九种不同的语言上预训练了一个多语言17亿（1.7B）参数模型，并以更细的粒度捕获训练检查点。我们以词级翻译作为测试平台，引入了一个新颖的数据集，通过行为分析、模型组件分析和基于参数的消融实验，追踪翻译能力在训练过程中的发展。我们发现，模型在快速获得基本语言能力的同时学会了词元级别的复制，而翻译能力的发展则分为两个截然不同的阶段：初始阶段以复制和表层相似性为主导，第二个阶段中……

    arXiv:2604.17633v3 Announce Type: replace  Abstract: Large language models exhibit impressive cross-lingual capabilities. However, prior work analyzes this phenomenon through isolated factors and at sparse points during training, limiting our understanding of how cross-lingual generalization emerges--particularly in the early phases of learning. To study the early trajectory of linguistic and translation capabilities, we pretrain a multilingual 1.7B model on nine diverse languages, capturing checkpoints at a much finer granularity. We use word-level translation as a testbed, introducing a novel dataset to trace how translation develops over training through behavioral analyses, model-component analysis, and parameter-based ablations. We find that the model quickly acquires basic linguistic capabilities in parallel with token-level copying, while translation develops in two distinct phases: an initial phase dominated by copying and surface-level similarities, and a second phase in which
    
[^158]: 量子视觉理论在音频分类中应用于深伪语音检测

    Quantum Vision Theory Applied to Audio Classification for Deepfake Speech Detection

    [https://arxiv.org/abs/2604.08104](https://arxiv.org/abs/2604.08104)

    本文提出量子视觉理论这一新视角，将语音频谱图先转换为信息波再输入深度学习模型进行分类，以提升深伪语音检测的性能。

    

    我们提出量子视觉理论作为一种基于深度学习的音频分类新视角，并将其应用于深伪语音检测。受量子物理学中波粒二象性的启发，量子视觉理论基于这样一种理念：数据不仅可以以其可观测的坍缩形式表示，还可以作为信息波来表示。在传统深度学习中，模型直接在这些坍缩表示（如图像）上进行训练。而在量子视觉理论中，输入首先通过量子视觉块转换为信息波，然后再输入深度学习模型进行分类。基于量子视觉的模型在图像分类任务中相较于非量子视觉对应模型提升了性能。那么，如果将量子视觉理论应用于语音频谱图进行音频分类任务会怎样？这正是所提出方法的动机与创新之处。在这项工作中，语音信号的短时傅里叶变换（STFT）、梅尔频谱图和梅尔频率倒谱系数（MFCC）

    arXiv:2604.08104v2 Announce Type: replace  Abstract: We propose Quantum Vision (QV) theory as a new perspective for deep learning-based audio classification, applied to deepfake speech detection. Inspired by particle-wave duality in quantum physics, QV theory is based on the idea that data can be represented not only in its observable, collapsed form, but also as information waves. In conventional deep learning, models are trained directly on these collapsed representations, such as images. In QV theory, inputs are first transformed into information waves using a QV block, and then fed into deep learning models for classification. QV-based models improve performance in image classification compared to their non-QV counterparts. What if QV theory is applied speech spectrograms for audio classification tasks? This is the motivation and novelty of the proposed approach. In this work, Short-Time Fourier Transform (STFT), Mel-spectrograms, and Mel-Frequency Cepstral Coefficients (MFCC) of s
    
[^159]: 重新思考基于大语言模型的语音识别中的熵分配：理解语音编码器与大语言模型之间的动态关系

    Rethinking Entropy Allocation in LLM-based ASR: Understanding the Dynamics between Speech Encoders and LLMs

    [https://arxiv.org/abs/2604.08003](https://arxiv.org/abs/2604.08003)

    该研究从熵分配的视角重新审视基于大语言模型的语音识别，提出了一种基于能力边界感知的多阶段训练策略，通过重新设计预训练以弥合语音-文本模态差距并引入迭代异步SFT，从而优化参数效率并增强幻觉鲁棒性。

    

    将大语言模型（LLM）集成到自动语音识别（ASR）中已成为一种主流范式。尽管近期的基于LLM的ASR模型在公开基准测试中表现出色，但在平衡识别质量与延迟和开销方面仍具有挑战性，而幻觉问题进一步限制了其实际部署。在本研究中，我们从熵分配的视角重新审视基于LLM的ASR，并引入三个指标来刻画训练范式如何在语音编码器与LLM之间分配熵的降低。为了解决现有方法中熵分配低效的问题，我们提出了一种基于能力边界感知的原则性多阶段训练策略，以优化参数效率和幻觉鲁棒性。具体而言，我们重新设计了预训练策略以缓解语音与文本模态之间的差距，并进一步引入了一种迭代异步的SFT阶段……

    arXiv:2604.08003v2 Announce Type: replace-cross  Abstract: Integrating large language models (LLMs) into automatic speech recognition (ASR) has become a dominant paradigm. Although recent LLM-based ASR models have shown promising performance on public benchmarks, it remains challenging to balance recognition quality with latency and overhead, while hallucinations further limit real-world deployment. In this study, we revisit LLM-based ASR from an entropy allocation perspective and introduce three metrics to characterize how training paradigms allocate entropy reduction between the speech encoder and the LLM. To remedy entropy-allocation inefficiencies in prevailing approaches, we propose a principled multi-stage training strategy grounded in capability-boundary awareness, optimizing parameter efficiency and hallucination robustness. Specifically, we redesign the pretraining strategy to alleviate the speech-text modality gap, and further introduce an iterative asynchronous SFT stage bet
    
[^160]: 权重聚类揭示大语言模型中相对排序的重要性

    Weight Clustering Reveals the Importance of Relative Rank in Large Language Models

    [https://arxiv.org/abs/2603.17917](https://arxiv.org/abs/2603.17917)

    利用标量 K 均值聚类将大语言模型权重压缩为 16-64 个共享值并可在不重训的情况下保持性能，并通过扰动实验揭示保持权重间的相对排序比保持精确数值更重要。

    

    大语言模型包含数十亿参数，但许多精确的权重值并非必需。我们对预训练模型应用标量 K 均值聚类，仅用 16-64 个共享值替换每个权重矩阵。对于 Llama 3.1-8B-Instruct 和 SmolLM2-135M，该方法在无需重新训练的情况下保持了较高的准确率，为在磁盘上压缩大语言模型提供了一种简单途径。仅对聚类中心进行微调，可进一步恢复 30-40% 的剩余精度损失。随后，我们将聚类作为分析工具，固定聚类分配，仅扰动共享值，以研究其哪些性质最为关键。我们发现，保持这些值的相对排序（即哪些连接比其他连接更强或更弱）比保持它们的精确数值更为重要。破坏这种排序会显著增加困惑度，而改变数值但保持排序在许多中间层几乎不造成损失。

    arXiv:2603.17917v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) contain billions of parameters, but many exact weight values are not essential. We apply scalar $K$-means clustering to pretrained models, replacing each weight matrix with only 16--64 shared values. For Llama~3.1-8B-Instruct and SmolLM2-135M, this preserves strong accuracy without retraining, providing a simple way to compress LLMs on disk. Fine-tuning only the cluster centroids further recovers 30--40\% of the remaining accuracy loss. Using clustering as an analysis tool, we then keep the cluster assignments fixed and perturb only the shared values, to study which of their properties matter most. We find that preserving their \emph{relative rank} (which connections are stronger or weaker than others) is more important than preserving their exact magnitudes. Breaking this ordering greatly increases perplexity, while changing the values but keeping the ordering causes little loss in many middle and 
    
[^161]: 翻译腔的方言塑造语言模型学习

    Dialects of Translationese Shape Language Model Learning

    [https://arxiv.org/abs/2602.16469](https://arxiv.org/abs/2602.16469)

    本文发现，机器翻译数据的源语言差异通过翻译腔显著影响语言模型的学习行为，其中词汇多样性是驱动总体困惑度的关键因素。

    

    arXiv:2602.16469v2 公告类型：替换 摘要：机器翻译数据在多语言自然语言处理中广泛使用，尤其是在母语文本稀缺的情况下。然而，翻译文本与母语文本存在系统性差异。这一现象被称为“翻译腔”，它既反映了源语言的痕迹，也体现了翻译本身的特征性属性。在本文中，我们研究了在机器翻译数据上训练如何影响小型英语语言模型，重点关注来自不同源语言的翻译腔如何塑造语言可接受性判断和不同领域的语言建模。我们在从24种类型学和资源多样性各异的源语言翻译成的英语文本上训练模型，从而能够系统分析源语言和语料库属性如何影响模型学习的内容。我们的结果表明，源语言对模型行为有明显影响：总体困惑度更多由翻译语料的词汇多样性驱动，但

    arXiv:2602.16469v2 Announce Type: replace  Abstract: Machine-translated data is widely used in multilingual NLP, particularly where native text is scarce. However, translated text differs systematically from native text. This phenomenon is known as translationese, and it reflects both traces of the source language and characteristic properties of translation itself. In this paper, we study how training on machine-translated data affects small English language models, focusing on how translationese from different source languages shapes linguistic acceptability judgments and language modeling for different domains. We train models on English text translated from 24 typologically and resource-diverse source languages, enabling a systematic analysis of how source language and corpus properties influence what models learn. Our results show that the source language has a clear impact on model behavior: general perplexity is more driven by the lexical diversity of the translated corpus, but 
    
[^162]: COBALT：基于离线轨迹的上下文老虎机学习用于多轮代码生成

    COBALT: Contextual Bandit Learning with Offline Trajectories for Multi-Turn Code Generation

    [https://arxiv.org/abs/2602.03806](https://arxiv.org/abs/2602.03806)

    COBALT 利用多轮代码生成是单步可恢复马尔可夫决策过程的观察，通过重用参考模型收集的离线多轮轨迹并采用单步补全的上下文老虎机学习，替代昂贵且不稳定的在线强化学习来高效训练 LLM 进行多轮代码生成。

    

    训练大语言模型（LLM）进行多轮代码生成通常依赖在线强化学习（RL），即模型通过反复生成的在线 rollout 进行学习。虽然有效，但在线强化学习成本高昂且不稳定。本文提出了基于离线轨迹的上下文老虎机学习方法（COBALT），这是一种简单的方法，通过重用离线多轮轨迹来高效训练 LLM 进行多轮代码生成。基于多轮代码生成是一个单步可恢复马尔可夫决策过程的观察，COBALT 首先使用一个参考 LLM 收集代码生成轨迹，并将其划分为部分轨迹。然后，在在线老虎机学习过程中，训练 LLM 通过单步代码生成来补全每个部分轨迹。在 LiveCodeBench 上，COBALT 使 R1-Distill 8B 和 Qwen3 8B 的绝对 Pass@1 分数最多分别提升 9.0 和 6.2 个百分点，并匹敌了（原摘要在此处截断）。

    arXiv:2602.03806v2 Announce Type: replace-cross  Abstract: Training large language models (LLMs) for multi-turn code generation typically relies on online reinforcement learning (RL), in which models learn from repeatedly generated online rollouts. Although effective, online RL can be costly and unstable. In this paper, we present contextual bandit learning with offline trajectories (COBALT), a simple method that reuses offline multi-turn trajectories to efficiently train LLMs for multi-turn code generation. Building on the observation that multi-turn code generation is a one-step recoverable Markov decision process, COBALT first collects code generation trajectories using a reference LLM and divides them into partial trajectories. Then, during online bandit learning, the LLM is trained to complete each partial trajectory through single-step code generation. On LiveCodeBench, COBALT substantially improves R1-Distill 8B and Qwen3 8B by up to 9.0 and 6.2 absolute Pass@1 points and matche
    
[^163]: ARTIS：基于迭代模拟的智能体风险感知测试时扩展

    ARTIS: Agentic Risk-Aware Test-Time Scaling via Iterative Simulation

    [https://arxiv.org/abs/2602.01709](https://arxiv.org/abs/2602.01709)

    ARTIS提出了一种风险感知的测试时扩展框架，通过在真实执行前进行迭代模拟将探索与承诺解耦，并引入风险感知的工具模拟器来捕捉罕见但高影响的失败模式，从而在不产生环境风险的前提下提升智能体动作的可靠性与鲁棒性。

    

    当前的测试时扩展（TTS）技术通过在推理阶段分配额外计算来提升大语言模型（LLM）的性能，但这种技术对于智能体场景仍然不够有效，因为在智能体场景中，动作会直接与外部环境交互，其后果可能是不可逆且代价高昂的。我们提出了ARTIS（Agentic Risk-Aware Test-Time Scaling via Iterative Simulation，基于迭代模拟的智能体风险感知测试时扩展），这是一个将探索与承诺解耦的框架，它通过在真实执行之前进行模拟交互，实现测试时的探索。这种设计使得扩展推理时计算成为可能，从而在不产生环境风险的情况下提高动作层面的可靠性和鲁棒性。我们进一步表明，朴素的基于LLM的模拟器难以捕捉罕见但高影响的失败模式，这极大地限制了其在智能体决策中的有效性。为了解决这一局限，我们引入了一种风险感知的工具模拟器，该模拟器强调保真度……（原文摘要在此处截断）

    arXiv:2602.01709v3 Announce Type: replace  Abstract: Current test-time scaling (TTS) techniques enhance large language model (LLM) performance by allocating additional computation at inference time, yet they remain insufficient for agentic settings, where actions directly interact with external environments and their effects can be irreversible and costly. We propose ARTIS, Agentic Risk-Aware Test-Time Scaling via Iterative Simulation, a framework that decouples exploration from commitment by enabling test-time exploration through simulated interactions prior to real-world execution. This design allows extending inference-time computation to improve action-level reliability and robustness without incurring environmental risk. We further show that naive LLM-based simulators struggle to capture rare but high-impact failure modes, substantially limiting their effectiveness for agentic decision making. To address this limitation, we introduce a risk-aware tool simulator that emphasizes fid
    
[^164]: 自我感知知识探测：通过置信度校准评估语言模型的关系知识

    Self-Aware Knowledge Probing: Evaluating Language Models' Relational Knowledge through Confidence Calibration

    [https://arxiv.org/abs/2601.18901](https://arxiv.org/abs/2601.18901)

    该论文建立了一个评估语言模型关系知识置信度校准的实证框架，发现大多数语言模型（尤其是掩码语言模型）存在过度自信问题，而考虑释义一致性的置信度估计能提供最佳的校准效果。

    

    知识探测量化了语言模型（LM）在预训练期间获取的关系知识量。现有的知识探测方法通过预测准确率和精确率等指标来评估模型能力。此类评估未能考虑模型的可靠性，而可靠性体现在其置信度分数的校准中。在本文中，我们建立了一个实证框架，用于在封闭答案集设置下评估关系知识的置信度校准。我们涵盖了模型置信度的三种模态：（1）内在置信度、（2）释义一致性和（3）语言置信度。我们对21个因果语言模型和8个掩码语言模型的广泛分析表明，大多数模型——尤其是使用掩码目标预训练的模型——都存在过度自信的问题。校准效果最好的分数来自那些考虑了因语句改写而导致不一致性的置信度估计。此外，当增加t（摘要在此处截断）

    arXiv:2601.18901v2 Announce Type: replace  Abstract: Knowledge probing quantifies how much relational knowledge a language model (LM) has acquired during pre-training. Existing knowledge probes evaluate model capabilities through metrics like prediction accuracy and precision. Such evaluations fail to account for the model's reliability, reflected in the calibration of its confidence scores. In this paper, we establish an empirical framework to evaluate the calibration of relational knowledge in a closed answer-set setting. We cover three modalities of model confidence: (1) intrinsic confidence, (2) paraphrase consistency and (3) linguistic confidence. Our extensive analysis of 21 causal and 8 masked language models reveals that most models, but especially those pre-trained with the masking objective, are overconfident. The best-calibrated scores come from confidence estimates that account for inconsistencies due to statement rephrasing. Moreover, calibration improves when increasing t
    
[^165]: 大语言模型时代生成式命名实体识别的评估

    Assessment of Generative Named Entity Recognition in the Era of Large Language Models

    [https://arxiv.org/abs/2601.17898](https://arxiv.org/abs/2601.17898)

    通过对八个大语言模型在四个标准NER数据集上的系统评估发现，采用参数高效微调和结构化输出格式的开源大语言模型可实现与传统基于编码器的模型相媲美的命名实体识别性能，且其能力来源于指令遵循与生成能力而非记忆。

    

    随着大语言模型（LLMs）的兴起，命名实体识别（NER）正从序列标注任务演变为一种生成式范式。我们对开源大语言模型在扁平NER和嵌套NER任务上进行了系统性评估。我们研究了多个研究问题，包括生成式NER与传统NER模型之间的性能差距、输出格式的影响、大语言模型是否依赖记忆，以及微调后通用能力的保持情况。通过在八个不同规模的大语言模型和四个标准NER数据集上的实验，我们发现：(1) 通过参数高效微调和结构化格式（如内联括号或XML），开源大语言模型取得了与传统基于编码器的模型相当的性能，并超越了采用上下文学习技术的基于解码器的大语言模型；(2) 大语言模型的NER能力源于指令遵循能力和生成能力，而非单纯的记忆。

    arXiv:2601.17898v3 Announce Type: replace  Abstract: Named entity recognition (NER) is evolving from a sequence labeling task into a generative paradigm with the rise of large language models (LLMs). We conduct a systematic evaluation of open-source LLMs on both flat and nested NER tasks. We investigate several research questions including the performance gap between generative NER and traditional NER models, the impact of output formats, whether LLMs rely on memorization, and the preservation of general capabilities after fine-tuning. Through experiments across eight LLMs of varying scales and four standard NER datasets, we find that: (1) With parameter-efficient fine-tuning and structured formats like inline bracketed or XML, open-source LLMs achieve performance competitive with traditional encoder-based models and surpass decoder-based LLMs with in-context learning techniques; (2) The NER capability of LLMs stems from instruction-following and generative power, not mere memorization
    
[^166]: 对齐否决：安全训练如何抑制大语言模型中的文化知识

    The Alignment Veto: How Safety Training Suppresses Cultural Knowledge in LLMs

    [https://arxiv.org/abs/2510.13154](https://arxiv.org/abs/2510.13154)

    该研究提出“对齐否决”概念，揭示安全训练在模型拒绝文化敏感问题时只是抑制而非抹除其内部准确的文化知识分布，且这种“安全税”在不同国家间分布不均（最高达37.6%），母语提示反而会加剧而非缩小这一差距。

    

    当对齐训练与语言模型内部编码的文化价值发生冲突时，模型内部会发生什么？通过对16个中东和北非国家、26个模型以及153万条人类调查响应的研究，我们给出的答案是抑制而非抹除：在拒绝发生的那一刻，模型的内部logit分布与其自由生成的答案相比，与人类调查数据的相关性更强。我们将这一现象称为“对齐否决”。我们区分了两类失败：抑制失败（准确的内部分布在输出端被阻断）和表征偏差失败（编码本身就偏离了人类价值观），并证明这两种失败需要不同的干预措施。这道闸门是不公平的：安全税高达37.6%，服务最好与最差的国家之间存在19.8%的对齐质量差距，而使用母语提示反而会扩大而非缩小这一差距。稀疏自编码器分析（并通过对齐各阶段的对比加以验证）识别出一个候选的DPO阶段特征，该特征介导了这一现象。

    arXiv:2510.13154v3 Announce Type: replace  Abstract: What happens inside a language model when alignment training conflicts with a cultural value it encodes? Across 16 MENA countries, 26 models, and 1.53M human survey responses, we show the answer is suppression, not erasure: at the moment of refusal, a model's internal logit distribution correlates with human survey data more strongly than its freely generated answers. We call this the alignment veto. We distinguish suppression failures (accurate internal distributions blocked at output) from representational bias failures (the encoding itself diverges from human values), and show the two require different interventions. The gate is inequitable: the safety tax reaches 37.6%, with a 19.8% alignment-quality gap between best- and worst-served nations, and native-language prompting widens rather than closes it. Sparse autoencoder analysis corroborated by comparisons across alignment stages identifies a candidate DPO-stage feature mediatin
    
[^167]: ABLEIST：LLM生成的招聘场景中的交叉性残障偏见

    ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios

    [https://arxiv.org/abs/2510.10998](https://arxiv.org/abs/2510.10998)

    该论文提出了ABLEIST评估框架，通过五个健全主义伤害指标和三个交叉性伤害指标，对六个LLM在2,820个招聘场景中的表现进行审计，揭示了针对残障候选人（尤其在性别、种姓等交叉身份维度）的显著偏见伤害，而许多先进模型无法检测到这些伤害。

    

    大型语言模型（LLM）因在招聘等高风险领域延续基于身份的歧视而日益受到审视，尤其是针对残障人士（PwD）的歧视。然而，现有研究仍主要聚焦于西方视角，忽视了性别和种姓等交叉性边缘化形式如何影响全球南方残障人士的经历。我们对六个LLM在2,820个招聘场景中进行了全面审计，这些场景涵盖多样的残障、性别、国籍和种姓特征。为了捕捉微妙的交叉性伤害与偏见，我们引入了ABLEIST（健全主义、励志化、超人化与象征化）框架，这是一套基于残障研究文献构建的评估体系，包含五个健全主义特定伤害指标和三个交叉性伤害指标。我们的结果显示，针对残障候选人的ABLEIST伤害显著增加——而许多最先进的模型未能检测到这些伤害。（摘要在此处截断）

    arXiv:2510.10998v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly under scrutiny for perpetuating identity-based discrimination in high-stakes domains such as hiring, particularly against people with disabilities (PwD). However, existing research remains largely Western-centric, overlooking how intersecting forms of marginalization--such as gender and caste--shape experiences of PwD in the Global South. We conduct a comprehensive audit of six LLMs across 2,820 hiring scenarios spanning diverse disability, gender, nationality, and caste profiles. To capture subtle intersectional harms and biases, we introduce ABLEIST (Ableism, Inspiration, Superhumanization, and Tokenism), a set of five ableism-specific and three intersectional harm metrics grounded in disability studies literature. Our results reveal significant increases in ABLEIST harms towards disabled candidates--harms that many state-of-the-art models failed to detect. These harms were furthe
    
[^168]: 打破伪装！基于同音字图谱与毒性词典的中文谐音毒性内容识别

    Breaking the Cloak! Unveiling Chinese Cloaked Toxicity with Homophone Graph and Toxic Lexicon

    [https://arxiv.org/abs/2505.22184](https://arxiv.org/abs/2505.22184)

    提出了C$^2$TU，一种无需训练、无需提示的新方法，利用中文同音字图谱和毒性词典，通过子串匹配识别候选毒性词并进行过滤纠正，有效破解社交媒体中以谐音伪装毒性词汇规避审查的内容。

    

    社交媒体平台上的毒性内容（包括辱骂性语言和歧视性言论）显著增加，给内容审核带来了越来越大的挑战。一些用户通过故意用谐音伪装毒性词汇来规避审查，这使得识别伪装毒性内容成为一项必要的任务。现有方法大多针对英文文本设计，而中文伪装毒性内容的识别问题尚未得到解决。为解决这一问题，我们提出了C$^2$TU，这是一种新颖的无需训练、无需提示（prompt-free）的中文伪装毒性内容识别方法。该方法首先利用子串匹配，基于中文同音字图谱和毒性词典识别候选毒性词汇。然后，它过滤掉那些无毒的候选词，并将伪装词纠正为其对应的毒性词。具体而言，我们开发了两种用于过滤的模型变体，分别基于BERT和大语言模型（LLMs）。

    arXiv:2505.22184v3 Announce Type: replace-cross  Abstract: Social media platforms have experienced a significant rise in toxic content, including abusive language and discriminatory remarks, presenting growing challenges for content moderation. Some users evade censorship by deliberately disguising toxic words through homophonic cloak, which necessitates the task of unveiling cloaked toxicity. Existing methods are mostly designed for English texts, while Chinese cloaked toxicity unveiling has not been solved yet. To tackle the issue, we propose C$^2$TU, a novel training-free and prompt-free method for Chinese cloaked toxic content unveiling. It first employs substring matching to identify candidate toxic words based on Chinese homophone graph and toxic lexicon. Then it filters those candidates that are non-toxic and corrects cloaks to be their corresponding toxicities. Specifically, we develop two model variants for filtering, which are based on BERT and LLMs, respectively. For LLMs, w
    
[^169]: ReasoningShield：面向大型推理模型推理轨迹的安全检测

    ReasoningShield: Safety Detection over Reasoning Traces of Large Reasoning Models

    [https://arxiv.org/abs/2505.17244](https://arxiv.org/abs/2505.17244)

    大型推理模型的思维链中间步骤中可能隐藏有害内容，导致现有审核工具性能大幅下降，本文提出ReasoningShield基准与轻量级模型，用于对CoT推理轨迹进行安全审核。

    

    大型推理模型利用显式推理轨迹（即思维链CoT推理）在得出最终答案之前将复杂问题分解为中间步骤。然而，这些推理轨迹带来了独特的安全挑战：即使最终答案看似无害，有害内容也可能嵌入在中间步骤中。我们的研究表明，现有最先进的内容审核工具在CoT审核方面出现了显著的性能下降，与传统答案审核相比，F1分数下降高达35.2%。为应对这些挑战，我们提出了ReasoningShield，这是一个全面的基准测试以及一套强大的轻量级模型，用于CoT安全审核。我们的基准包含9.2K个带标注的（查询，CoT）对，具有结构化的分步风险分析，包括一个7K样本的训练集和一个2.2K的人工标注测试集，涵盖10个风险类别、3个安全等级，……

    arXiv:2505.17244v3 Announce Type: replace-cross  Abstract: Large Reasoning Models (LRMs) leverage explicit reasoning traces, known as Chain-of-Thought (CoT) reasoning, to decompose complex problems into intermediate steps before deriving final answers. However, these reasoning traces introduce unique safety challenges: harmful content can be embedded in intermediate steps even when final answers appear benign. Our study reveals that existing state-of-the-art moderation tools experience significant performance degradation on CoT moderation, with F1 scores dropping by up to 35.2% compared to traditional answer moderation. To address these challenges, we present ReasoningShield, a comprehensive benchmark and a suite of strong lightweight models for CoT safety moderation. Our benchmark includes 9.2K annotated (Query, CoT) pairs with structured stepwise risk analysis, consisting of a 7K-sample training set and a 2.2K human-annotated test set, spanning 10 risk categories, 3 safety levels, an
    
[^170]: 在提示注入防御到达用户之前对其进行审计：强大对手揭示了什么

    Auditing Prompt Injection Defenses Before They Reach Users: What Strong Adversaries Reveal

    [https://arxiv.org/abs/2505.15738](https://arxiv.org/abs/2505.15738)

    该论文主张，由于基于微调的提示注入防御缺乏形式化保证且可能带来虚假安全感，开发者应在防御部署给用户之前用强大的对抗性攻击对其进行压力测试。

    

    大型语言模型越来越多地被部署在面向用户的产品中，这些产品在处理敏感个人数据的同时还需与不可信的外部内容交互。这使得它们成为提示注入攻击的首要目标——攻击者在外部内容中注入恶意指令以劫持模型。为缓解此类攻击，研究者提出了基于微调的防御方法，通过训练模型遵循良性的用户指令并忽略注入的指令。由于缺乏形式化保证，这些防御方法仅依据当时可获得的最有效的现实攻击进行实证评估，往往报告近乎为零的攻击成功率——这给开发者以及依赖这些防御的用户带来了虚假的安全感，同时也留下了未来更强大的攻击可能绕过这些防御、并在广泛部署的系统中造成严重后果的隐患。在本文中，我们主张开发者应对其防御……

    arXiv:2505.15738v3 Announce Type: replace-cross  Abstract: Large language models are increasingly deployed in user-facing products that handle sensitive personal data while interacting with untrusted external content. This makes them a prime target for prompt injection attacks, in which an adversary injects malicious instructions in that content to hijack the model. Fine-tuning-based defenses have been proposed to mitigate these attacks by training models to follow benign user instructions and ignore injected ones. Lacking formal guarantees, these defenses are evaluated empirically against the most effective realistic attacks available at the time, often reporting near-zero attack success rates -- giving developers, and the users who depend on them, a false sense of security and leaving open the possibility that future, more capable attacks could circumvent them and cause severe consequences in widely deployed systems. In this paper, we argue that developers should stress-test their de
    
[^171]: 知道何时停止：基于潜在充分性信号的高效上下文处理

    Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals

    [https://arxiv.org/abs/2502.01025](https://arxiv.org/abs/2502.01025)

    该论文发现LLM内部特定注意力头天然编码了“信息充分性信号”，据此提出“动态上下文截断”方法，让模型在获取足够任务相关信息后自动停止处理上下文，在多个问答数据集和模型上实现了平均1.33倍的token减少且准确率反而提升3.4%。

    

    大语言模型（LLMs）不加区分地处理整个输入上下文，当回答查询所需的信息仅局部存在于上下文中时，这种做法效率低下。我们提出了动态上下文截断，这是一种新颖的方法，使LLMs能够在获取足够的任务相关信息后自行终止处理。通过对模型内部的分析，我们发现特定的注意力头天然编码了“充分性信号”——可通过轻量级分类器检测到——这些信号能够预测关键信息何时已被处理完毕。这揭示了一种全新的效率范式：模型自身的内部理解自然地决定了处理需求，而非依赖外部的压缩启发式方法。在六个问答数据集（长达40K tokens）和三个模型系列（LLaMA/Qwen/Mistral，1B-70B）上的综合实验表明，该方法在实现平均1.33倍token减少的同时，准确率提升了3.4%。此外，我们的方法（摘要在此处被截断）

    arXiv:2502.01025v4 Announce Type: replace  Abstract: Large language models (LLMs) process entire input contexts indiscriminately, which is inefficient when the information required to answer a query is localized within the context. We present dynamic context cutoff, a novel method enabling LLMs to self-terminate processing upon acquiring sufficient task-relevant information. Through analysis of model internals, we discover that specific attention heads inherently encode "sufficiency signals" -- detectable through lightweight classifiers -- that predict when critical information has been processed. This reveals a new efficiency paradigm: models' internal understanding naturally dictates processing needs rather than external compression heuristics. Comprehensive experiments across six QA datasets (up to 40K tokens) with three model families (LLaMA/Qwen/Mistral, 1B-70B) demonstrate 3.4% accuracy improvement while achieving 1.33x token reduction on average. Furthermore, our method demonstr
    
[^172]: OARelatedWork：一个包含来自开放获取来源全文的大规模相关工作章节摘要数据集

    OARelatedWork: A Large-Scale Dataset of Related Work Sections with Full-texts from Open Access Sources

    [https://arxiv.org/abs/2405.01930](https://arxiv.org/abs/2405.01930)

    本文提出了首个用于相关工作生成的大规模多文档摘要数据集OARelatedWork，包含完整相关工作章节与被引论文全文，并通过基准测试揭示了大语言模型在处理海量全文上下文时仍面临的挑战。

    

    本文介绍了OARelatedWork：一个用于从开放获取文献生成相关工作的数据集。这是首个用于相关工作生成的大规模多文档摘要数据集，包含完整的相关工作章节以及被引论文的全文。其验证集和测试集的构建方式确保每篇源论文都可以获取全文，从而能够对基于全文的相关工作生成进行可控评估。该数据集包含来自多个领域的94,450篇论文。通过OARelatedWork，我们旨在推动该领域从仅基于摘要生成相关工作章节的部分内容，转向基于所有可用内容生成完整的相关工作章节。我们 (i) 对广泛的模型进行了基准测试，结果表明合成海量全文上下文对所测试的高性价比大语言模型而言仍是挑战：在我们的语句级评判标准下，GPT-4o-mini基于证据的True率从92.9%下降……

    arXiv:2405.01930v3 Announce Type: replace  Abstract: This paper introduces OARelatedWork: a dataset for related work generation from open-access sources. It is the first large-scale multi-document summarization dataset for related work generation, containing whole related work sections and full texts of cited papers. Its validation and test splits are constructed so that every source paper is available in full text, enabling controlled evaluation of full-text related work generation. The dataset includes 94 450 papers from multiple domains. With OARelatedWork, we aim to shift the field from generating parts of related work sections from abstracts only to generating entire related work sections from all available content. We (i) benchmark a wide spectrum of models, highlighting that synthesizing massive full-text contexts remains challenge for tested cost-effective Large Language Models (LLMs): under our statement-level judge, GPT-4o-mini's evidence-grounded True rate drops from 92.9% w
    

