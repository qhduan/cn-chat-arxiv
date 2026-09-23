# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs](https://arxiv.org/abs/2609.26796) | Flash-dLLM是一个无需训练的扩散大语言模型推理加速框架，它将GPU内存I/O识别为主要瓶颈，通过I/O感知的融合KV缓存内核减少冗余内存移动，并结合并行解码机制实现快速且内存高效的推理。 |
| [^2] | [Agensh: Scaling Organizational Intelligence to 1,024 Agents](https://arxiv.org/abs/2609.26781) | Agensh是一个无需中央编排器的可扩展自组织多智能体框架，通过异步协作循环和共享工作区、消息接口、共享上下文三大组织基础设施，将组织智能扩展至1,024个智能体。 |
| [^3] | [SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue](https://arxiv.org/abs/2609.26780) | 提出SpeakerMem-R1，一种以说话人为中心的双轨记忆框架，通过存储带说话人标签的逐字消息和个人/群体层面的衍生状态，解决多方对话中的消息归属、关系理解与状态重建两大瓶颈。 |
| [^4] | [Beyond Repeated Sampling: Learning Search Policies for LLM Reasoning](https://arxiv.org/abs/2609.26704) | 该论文提出在语义层面引导大语言模型推理探索的新范式——先采样多样化的概念、提示或策略再条件化生成答案，并通过强化学习训练小型概念生成器以最大化下游答案成功率，从而超越朴素的重复采样策略。 |
| [^5] | [Measuring the Serving Stack Instead of the Model: Hidden Confounds in Local Tool-Use Evaluation](https://arxiv.org/abs/2609.26693) | 本地服务栈（如 Ollama）的工具调用门控机制和失败元数据丢失会混淆模型工具使用能力的评估，使测得的保真度反映的是服务层而非模型本身的行为。 |
| [^6] | [Detecting GPT-Assisted Writing Using Interpretable Stylometric Features](https://arxiv.org/abs/2609.26687) | 该研究仅利用可解释的文体特征（词汇和语法特征）训练机器学习分类器来检测GPT辅助写作，随机森林模型达到0.87的ROC-AUC，并通过SHAP分析揭示了词汇和语法特征是关键预测因子。 |
| [^7] | [Discovery-Driven Integration of Disjoint Tables via Text](https://arxiv.org/abs/2609.26658) | 该论文提出LOKI架构，将“文本介导的连接路径发现”形式化为新任务，通过全局表-文本对比学习实现数据湖中缺乏连接属性的无关联表的细粒度集成。 |
| [^8] | [Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding](https://arxiv.org/abs/2609.26638) | GravityOCR提出参数共享的自回归-块-扩散模型，通过扩散并行生成草稿并经自回归验证后提交，在无需独立草稿网络的情况下加速文档OCR推理，同时支持基于OCR奖励的GRPO强化学习训练。 |
| [^9] | [Capable yet Parsimonious: Extracting and Characterizing Hidden Chain-of-Thought in Frontier Models](https://arxiv.org/abs/2609.26637) | 通过标准API注册简单自定义工具诱导前沿模型外化隐藏的思维链，验证提取的推理性能与原生CoT相当且显著优于无推理基线，并系统刻画了前沿模型中间推理的结构特征。 |
| [^10] | [Knowledge Pull Requests for Continual Document Authoring](https://arxiv.org/abs/2609.26634) | 提出知识拉取请求（KPR）框架，通过提取论断、路由至章节并生成区分知识变化与文本变化的变更日志，实现可解释的持续文档修订，在维基百科和RAGTIME上比重写或重新生成整合更多信息且更好地保留原有内容。 |
| [^11] | [PERSONAWEAVER: Controllable Diversity Beyond Conventional Archetypes in Procedural Character Generation](https://arxiv.org/abs/2609.26629) | PersonaWeaver通过将世界构建与行为规范解耦，并利用人工策划的多样化道德立场库与对话反应库来建模角色行为，突破了LLM生成角色时行为同质化的局限，实现了程序化角色生成中超越传统原型的可控多样性。 |
| [^12] | [Semantic Abstraction for Natural Language Inference: a Methodological Framework for Discovering and Compensating Semantic Knowledge and Reasoning Gaps in Large Language Models](https://arxiv.org/abs/2609.26610) | 本文提出了一个基于语义相容性与不相容性的方法论框架，通过在更高抽象层次上重构前提与假设之间的词汇语义关系，以发现并补偿大语言模型在自然语言推理任务中的语义知识与推理差距。 |
| [^13] | [Receptiveness, Not Sycophancy: Distinguishing Engagement from Deference in Language Models](https://arxiv.org/abs/2609.26579) | 该论文指出，语言模型“社会性谄媚”评估存在构念效度问题，因为谄媚的标志与有益的“对话接纳性”高度重叠——被判定为谄媚的回复往往只是更具接纳性，而非盲目顺从。 |
| [^14] | [A retrospective analysis on the use of LLMs to study infant syntax learning](https://arxiv.org/abs/2609.26539) | 本文从认识论角度对BabyLM挑战赛等使用大语言模型研究婴儿句法学习的研究计划进行了回顾性评估，指出其方法论中存在的重大假设削弱了理论适用范围，且使用符合发育现实性的语料库对模型在常用基准测试上的表现影响有限。 |
| [^15] | [Transcribe, Translate, and Optimize: Joint Reward Learning for Speech Translation](https://arxiv.org/abs/2609.26536) | 提出通过GRPO对语音识别与翻译进行联合强化微调，以解决SFT参考转录与模型生成转录之间的不匹配问题，在CoVoST 2和FLEURS上显著提升BLEU分数并降低词错误率。 |
| [^16] | [A Semiotics-Aware Framework for Evaluating Fidelity and Coverage in Natural Language Generation](https://arxiv.org/abs/2609.26527) | 该论文提出符号学忠实度与符号学覆盖度两个新评估指标，用以衡量文本间的符号学对齐程度，发现覆盖度通常低于忠实度，且大语言模型与人工整理数据在低采样温度下对齐程度最高。 |
| [^17] | [Calibration as a First-Class Criterion in LLM Evaluation](https://arxiv.org/abs/2609.26489) | 该论文主张将校准（模型置信度与实际正确性的一致性）作为LLM评估中不可或缺的一等公民标准，并指出由于标准校准指标仅需置信度分数和正确性判断这两个输入，而现有大多数基准已具备这两者，因此校准评估可以且应当被常规纳入模型评估流程。 |
| [^18] | [Spoken Language Models that Think Aloud](https://arxiv.org/abs/2609.26488) | 提出了一种异步出声思考框架，让口语语言模型在推理过程中同步生成简短的进度性话语，从而消除“先思考后说话”范式带来的长时间静默，实现更自然的实时口语交互。 |
| [^19] | [Behavior is Not Enough: A Mechanism-Based Evaluation of Social Norm Emergence in LLM Societies](https://arxiv.org/abs/2609.26481) | 该论文提出一个超越行为观察的评估框架，通过测量LLM智能体的经验性与规范性预期，并结合社会学习、社会选择两种机制以及对抗性干扰下的稳定性测试，来更严谨地评估LLM社会中的规范涌现。 |
| [^20] | [How to Estimate Whether You Have Found Several Needles in a Haystack: Measuring Calibration in Multi-Label Text Classification](https://arxiv.org/abs/2609.26468) | 提出了一种对正负标签分配赋予同等权重的新分箱方案，解决了多标签文本分类中标签级期望校准误差计算不准确的问题。 |
| [^21] | [Enriching Speech Emotion Representations with Conversational Context](https://arxiv.org/abs/2609.26422) | 提出ACERT模块，通过整合灵活长度的对话上下文窗口来丰富语音情感表示，从而更好地捕捉语音交互中的情感演变，在IEMOCAP数据集上超越现有最先进方法。 |
| [^22] | [Combining Hierarchical Cognitive Process with Process Supervision for Interpretable Scene Safety Understanding](https://arxiv.org/abs/2609.26399) | 该论文提出将分层认知过程建模与过程监督相结合，构建了带过程标签的多步推理场景安全理解数据集，以提升大语言模型可解释的场景安全理解能力。 |
| [^23] | [On the Lexical Superstition of Large Language Models for Code Comprehension: Re-evaluation on Code of Low Lexical Quality](https://arxiv.org/abs/2609.26388) | 论文提出语义保持的标识符重命名框架Face/Off，揭示大语言模型在代码理解中普遍过度依赖标识符的词汇线索，且这一根深蒂固的问题难以通过现有的提示和微调干预加以解决。 |
| [^24] | [Layout-Guided Masking for GROBID: Lightweight Structural Gains in Large-Scale Scientific PDF Ingestion](https://arxiv.org/abs/2609.26381) | 通过为GROBID配备轻量级CPU版面检测器并使用类型化区域掩码路由标记，无需GPU即可在大规模科学PDF结构化解析中显著提升精度。 |
| [^25] | [HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing](https://arxiv.org/abs/2609.26368) | HySparse2提出了一种具有两级KV共享的混合稀疏注意力架构：外层采用YOCO式KV桥接仅桥接全注意力层，内层在KV复用基础上将块级稀疏升级为词元级稀疏，从而实现高效预填充、紧凑KV缓存存储和精确的长上下文检索。 |
| [^26] | [TransBERT: A Framework for Synthetic Translation in Domain-Specific Language Modeling](https://arxiv.org/abs/2609.26347) | 提出仅使用合成翻译文本预训练语言模型的TransBERT框架，并证明仅凭合成翻译数据即可在法语生命科学领域的各类下游任务上达到最先进性能。 |
| [^27] | [Blaming Across the Aisle: Political Contrasting and Blame Attribution in the Danish Parliament](https://arxiv.org/abs/2609.26346) | 本研究开发了面向中低资源语言的高效标注指责归因分类器BlameBERT，分析丹麦议会1997-2026年的辩论数据，发现指责行为呈“香蕉形”轨迹且近年持续上升，反对党显著多于执政党进行指责（“政治对比”效应），且该效应受意识形态调节、在右翼中更弱。 |
| [^28] | [Designing and Analysing Argument Mining Pipelines: Towards a Comprehensive Assessment](https://arxiv.org/abs/2609.26338) | 本文提出一个基于语言学、计算和领域三重视角的元研究框架，系统分析最先进的端到端论辩挖掘流水线，从而实现不同方法之间更清晰的任务级比较与评估。 |
| [^29] | [CHiME-9 ECHI: A Machine Learning Challenge for Enhancing Conversations to Address Hearing Impairment](https://arxiv.org/abs/2609.26306) | 本文介绍了CHiME-9 ECHI挑战赛的任务与结果，该挑战赛旨在从嘈杂多通道录音中提取对话语音以改善听力障碍者的语音可懂度与质量，并发现顶级系统能显著提升主观听感表现，而客观指标无法反映听众的真实感受。 |
| [^30] | [PACE-dLLM: Elastic Block Decoding via Confidence Cliff Estimation for Diffusion Language Models](https://arxiv.org/abs/2609.26249) | PACE-dLLM通过在每步以闭式形式拟合模型自身置信度的“悬崖”曲线并根据其饱和点动态确定前瞻视野，实现了扩散语言模型的弹性块解码加速。 |
| [^31] | [Damage Predicts Recovery: When Calibration Data Matters in Compressing Financial LLMs](https://arxiv.org/abs/2609.26241) | 该论文提出“损伤决定恢复”假设：压缩金融大语言模型时，校准数据是否重要取决于压缩造成的任务级损伤程度——量化几乎无损时校准语料的选择无关紧要，而剪枝造成大幅性能损失时，任务格式的金融领域校准数据能够恢复部分损失。 |
| [^32] | [ABAI at COLIEE 2026 Task 1: Multi-Stage Retrieval with GraphRAG-Enhanced Meta-Learning, and a Post-Hoc Study of the Cross-Validation-to-Test Gap](https://arxiv.org/abs/2609.26237) | 本文提出了一种用于COLIEE 2026判例法检索任务的四阶段多阶段检索流程（多视角BM25、神经重排序、图特征与LightGBM元学习），并通过控制实验系统性地分析了官方测试集F1（0.177）远低于交叉验证结果（0.311）的原因。 |
| [^33] | [A Semantic Approach to the Academic Publishing Network: Document Vector Representations and Hybrid Structural-Semantic Fusion over OpenAlex Data](https://arxiv.org/abs/2609.26218) | 该论文在学术出版网络结构图分析的基础上引入语义层，利用SPECTER2引用感知文档嵌入和权重可调的结构-语义后期融合函数，实现了比TF-IDF基线更契合专家主题分类的文档表示，并提升了推荐效果。 |
| [^34] | [Same Chart, Different Story: Bias in Vision-Language Chart Interpretation](https://arxiv.org/abs/2609.26210) | 提出了首个用于审计视觉-语言模型图表解读偏见的基准ChartBias，通过820个真实图表、六个社会属性以及12个模型共155,484条响应，揭示了当图表中仅替换社会群体时模型叙述会发生系统性偏移的偏见问题。 |
| [^35] | [Beyond Static Charts: Can Language and Vision Language Models Generate Interactive Data Visualization Interfaces?](https://arxiv.org/abs/2609.26208) | 该论文提出了VIS-GEN基准，包含3,042个覆盖数据过滤、时间分析和可视化编辑等多样化分析意图的样本，用于评估大语言模型和视觉语言模型根据自然语言查询生成交互式数据可视化界面的能力，并对14个最先进的开源与闭源模型进行了系统评测。 |
| [^36] | [WatchPoint: Executable User Feedback for Real-World Agentic Web Development](https://arxiv.org/abs/2609.26204) | WatchPoint是一个模拟用户系统，通过像真实开发者一样对运行中的Web应用生成并执行诊断脚本，产生结构化观察结果来指导编码智能体的重试，在包含1,000个顺序依赖任务的Web-Bench基准上恢复了57.6%的失败任务。 |
| [^37] | [Dynamic Deep Prompt Optimization for Defending Against Jailbreak Attacks on LLMs](https://arxiv.org/abs/2609.26185) | 提出首个基于深度提示优化的越狱防御方法DDPO，利用大语言模型自身中间层特征并通过轻量级多层感知机动态生成防御嵌入注入后续层，在不修改模型权重的情况下实现随输入自适应的安全防御。 |
| [^38] | [Modality-Gated Deep Adapters: Adding a Modality to a Frozen Embedding Model with Exact Preservation](https://arxiv.org/abs/2609.26182) | 提出模态门控深度适配器，可在冻结的多模态嵌入模型上添加新模态，同时保证现有输出逐位不变地被精确保持，且共存的多个模态包之间通过精确零隔离矩阵实现完全隔离。 |
| [^39] | [Magnitude Profile Pruning: Calibration-Free Structured Attention Head Removal for Transformer Compression](https://arxiv.org/abs/2609.26177) | 该论文提出了无需校准数据、梯度计算或Hessian估计的免训练注意力头剪枝方法——幅值剖面（MP）评分，通过权重行范数的统计离群点检测识别并移除冗余注意力头，并提供支持分组查询注意力（GQA）的MP-G变体，实现硬件友好的Transformer模型压缩。 |
| [^40] | [DTOC: Dynamic Tool Output Compression for Adaptive Context Management in AI Agents](https://arxiv.org/abs/2609.26121) | 提出动态工具输出压缩框架DTOC，通过在外部存储器中保留完整工具输出并在上下文中插入可逆占位符，实现LLM智能体的可扩展上下文管理和按需信息恢复。 |
| [^41] | [Differentiable Fuzzy Inference Layer: A Monotone, Compositional Ordinal Reasoning Head for Large Language Models](https://arxiv.org/abs/2609.26113) | 提出可微分模糊推理层（DFIL），一种双路径预测头，通过有序隶属函数与t-范数运算，赋予大语言模型对序数类别的单调性以及无需组合训练数据的量词组合推理能力，解决了模型无法正确组合“大多数”等序数量词的问题。 |
| [^42] | [TSS: Target-Side Sparsification for Speculative Decoding in Domain-Specific Large Language Models](https://arxiv.org/abs/2609.26100) | 该论文提出TSS框架，通过在投机解码的目标端跳过特定层实现稀疏化，在降低验证成本的同时提高草稿接受率，并保持甚至提升下游任务性能。 |
| [^43] | [One Domain, Many Tongues: Composing Domain and Language LoRAs for Cross-Lingual Remote-Sensing MLLMs without Paired Data](https://arxiv.org/abs/2609.26097) | MODL通过联合训练领域LoRA与语言LoRA并强制二者在每层保持互正交，无需任何多语言遥感配对数据即可为英语遥感多模态大模型添加新语言，同时保留其多语言能力。 |
| [^44] | [SpecialEduBench: Benchmarking Vision-Language Models on Knowledge, Skill, and Attitude in Language Intervention for Autistic Children](https://arxiv.org/abs/2609.26090) | 提出了SpecialEduBench基准，首次从知识、技能和态度三个维度评估视觉语言模型在自闭症儿童语言干预中的实际教学能力，其技能与态度题项基于真实干预录像，需要结合视觉与语言信息进行综合判断。 |
| [^45] | [CoVeR: Coverage-Based Routing of Verifier Calls in Agentic Retrieval](https://arxiv.org/abs/2609.26086) | CoVeR 利用冻结句子嵌入的覆盖度边际作为单一阈值门控，仅在证据状态模糊时才调用 LLM 验证器，从而在保持答案准确率几乎不变的前提下，大幅削减智能体检索中验证器反复处理证据的调用成本。 |
| [^46] | [TopoCompress: Topology Aware Token Compression Algorithm for Distributed Edge MoE Inference](https://arxiv.org/abs/2609.26061) | TopoCompress提出了一种部署与拓扑感知的令牌压缩框架，通过联合优化令牌压缩、专家部署与复制、GPU-CPU驻留和协同路由，实现通信高效的分布式边缘MoE推理。 |
| [^47] | [Optimizing Denoising Trajectories in dLLMs: A Lightweight Evolutionary Heuristic Approach](https://arxiv.org/abs/2609.26052) | 该论文通过分析Transformer注意力模式，揭示了dLLM基于置信度的去噪调度器存在EOS溢出和近端偏差两种失效模式的根源（无效token获得过高注意力权重），并提出一种轻量级进化启发式方法，利用有效注意力分数优化去噪轨迹。 |
| [^48] | [FIRE: Failure-Informed Runtime Engineering for Reliable Language-Model Agents](https://arxiv.org/abs/2609.26048) | 该论文提出FIRE，通过在不改变模型权重和用户提示的前提下，在失败发生前的状态处施加自然语言指令与动作拒绝等运行时策略，显著提升语言模型智能体的重复交付可靠性，在Terminal-Bench 2.1上pass^2最高提升9.2个百分点。 |
| [^49] | [Truth for Believable AI: Expressed Doubt, Provenance, and Belief Revision as an Engineerable Stance](https://arxiv.org/abs/2609.26035) | 该论文提出在固定语言模型之上构建一个可工程化的“认知行为层”，通过表达不确定性、来源门控断言和持久化信念修正机制，实现对修正的可审计确认，并能抵抗错误修正。 |
| [^50] | [Domain-Adaptive Pretraining Enhances Water Treatment Semantic Representation for Large-Scale Structured Literature Mining](https://arxiv.org/abs/2609.26034) | 本研究基于约29.7亿词元的水处理语料库持续预训练，开发了领域自适应模型WaterBERT，在水处理文本的语义表示和结构化信息提取任务上超越了通用型及领域特定型BERT模型。 |
| [^51] | [MICRO: Multi-Fidelity Active Search for Severe Error Discovery](https://arxiv.org/abs/2609.26025) | MICRO是一个多保真度主动搜索框架，通过联合建模质量评分与标注损失、按预测影响聚类以选择多样化候选、并利用滚动评估发现价值，在共享预算下最大化严重错误的确认发现数量。 |
| [^52] | [Challenges of Multi-Speaker Extraction for Real Conversational Speech Enhancement](https://arxiv.org/abs/2609.25948) | 该论文提出一种新的损失函数来缓解真实对话中过多静音对多说话人提取模型训练的影响，将STOI从0.55提升至0.60、频率加权分段信噪比从4.35提升至5.12。 |
| [^53] | [ClusterFewshot: Improving Few-shot Optimization for LLMs workflow](https://arxiv.org/abs/2609.25939) | ClusterFewshot通过将语义聚类结构与效用感知评分相结合来构建更具代表性的少样本示例集，在多个基准测试中显著降低优化成本的同时，持续提升了LLM工作流的准确率。 |
| [^54] | [Certified Against Which Oracle? Execution Labels Set the Reported Risk of Conformal Abstention for Text-to-SQL](https://arxiv.org/abs/2609.25938) | Text-to-SQL保形弃答证书所报告的风险严重依赖于校准时使用的正确性标准，使用更严格的多实例测试套件和专家盲评标签后发现，证书的实际风险远高于其标称水平。 |
| [^55] | [Informed Masking: Structure-Aware Perturbation for Reinforcement Learning in Diffusion Large Language Models](https://arxiv.org/abs/2609.25927) | 该论文发现扩散大语言模型 rollout 中存在上游/下游 token 结构及子问题难度不对称现象，并提出知情掩码方法，通过为每个 token 计算优先级分数来选择掩码位置，从而在有限蒙特卡洛预算下显著提升强化学习对齐的似然估计质量。 |
| [^56] | [Rethinking Length-Based Training: Batch Composition and Loss Normalization in Speech Token Language Models](https://arxiv.org/abs/2609.25890) | 该论文通过匹配比较解耦了语音token语言模型中基于长度训练的多个混杂因素，发现由短到长的排序本身并无独立收益，其表面效果主要源于批次组成和损失归一化方式的差异。 |
| [^57] | [Isolated Sign Language Recognition for Icelandic Sign Language: Experiments in a Low-resource Setting](https://arxiv.org/abs/2609.25862) | 该论文首次针对极低资源的冰岛手语开展孤立手语识别实验，发现跨语言迁移（先在美国手语数据上预训练再微调）能带来最大收益，将准确率提升 14-24 个百分点。 |
| [^58] | [BELXTR: Biomedical Entity Linking via Contextualized Token Retrieval](https://arxiv.org/abs/2609.25859) | 提出了基于多向量（后期交互）架构的BELXTR模型，通过利用词元级匹配信息将XTR扩展到生物医学实体链接任务，在十个语料库中一半上超越现有最先进方法，recall@1平均提升5个百分点。 |
| [^59] | [MemoryAthena: Adaptive Routing over Latent and Generated Memories](https://arxiv.org/abs/2609.25853) | MemoryAthena 提出在直接检索、基于检索线索生成和无表生成三条记忆路径之间进行自适应因果路由，在冻结主干与记忆模块的情况下，依据反事实似然优势学习轻量路由头，并通过有界插值将生成的记忆按情境动态融合到检索残差中。 |
| [^60] | [ARAFA: An LLM-Generated Arabic Fact-Checking Dataset](https://arxiv.org/abs/2609.25833) | 研究者利用大语言模型通过三步自动化流水线构建了包含18万余个标注声明-证据对的大规模阿拉伯语事实核查数据集Arafa，有效缓解了阿拉伯语事实核查资源稀缺的问题。 |
| [^61] | [Auditing Proxy-Based Validation Across Text Spans](https://arxiv.org/abs/2609.25808) | 该论文提出“验证契约”审计框架，通过在评分文本片段之外重新评估代理规则，发现当评估分数与代理标签共享同一文本片段时，二者的一致性可能源于共享的表面证据而非目标语义构念，从而揭示基于代理的验证可能高估评估分数的有效性。 |
| [^62] | [Latest Exact Match Attention](https://arxiv.org/abs/2609.25802) | 提出最新精确匹配注意力（LEMA），证明了带思维链的LEMA transformer与字随机存取机在计算和内存上可以相互高效模拟，并给出了基于直通估计器和软注意力退火的实用训练方法。 |
| [^63] | [Reply to comments arXiv:2512.07881 and arXiv:2601.06104 on quantum structure in human and AI-generated language](https://arxiv.org/abs/2609.25797) | 本文针对评论者对人类与AI生成语言中量子结构研究的批评进行逐点回应，澄清了实验协议、纠缠识别判据、玻色-爱因斯坦拟合等关键问题，并更正了一处不影响结果的排版错误。 |
| [^64] | [Syndrome, Synergy, and Safety: Structured Reasoning and Knowledge-Driven Alignment for TCM Prescription Generation](https://arxiv.org/abs/2609.25755) | 该论文提出一个渐进式四阶段框架（SFT → PG-CoT → 动态SFT → K-RL），通过理法方药范式下的可审计推理、随证加减的诊疗轨迹建模以及十八反等禁忌规则驱动的DPO对齐，显著提升了中医处方生成的质量、可解释性与安全性。 |
| [^65] | [Slow Decay and Silenced Expression: Iterated Subliminal Trait Transfer in Language-Model Lineages](https://arxiv.org/abs/2609.25721) | 该研究发现，通过过滤数据潜意识传递的模型特质能够在语言模型谱系中存续十代且仅缓慢衰减，但其表达被“沉默”——无法通过输出中的关键词检测到，只能依靠激活探针揭示其仍然存在。 |
| [^66] | [How Strongly Should Task State Influence an LLM Agent?](https://arxiv.org/abs/2609.25686) | 该论文通过固定任务规则与模型、只改变任务状态进入智能体的强度（从展示文本、逐轮指令到硬性强制执行门），首次将任务状态管理方式对LLM智能体可靠性的贡献进行了解耦与量化。 |
| [^67] | [From Utterances to Networks: Modelling Slang Adoption and Diffusion Across Subreddits](https://arxiv.org/abs/2609.25669) | 该研究利用大语言模型作为可扩展的标注器，结合社会互动与语言特性双重视角，构建人工标注基准并建模网络俚语在Reddit子版块间的采纳与传播机制。 |
| [^68] | [Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices](https://arxiv.org/abs/2609.25645) | 提出GittinsEval，一种基于贝叶斯最优Gittins策略的成本感知LLM评估方法，通过轻量级在线更新高效决定下一个待评估配置及停止时机，在多个基准上以更低的评估成本取得有竞争力的性能。 |
| [^69] | [Qwen3.8-Omni: Towards Native Omni-Modal Agents](https://arxiv.org/abs/2609.25611) | Qwen3.8-Omni-Flash 通过原生多模态协同训练策略与百万 token 上下文窗口，在保持文本能力的同时大幅提升多模态理解推理与长程智能体任务表现，可作为主智能体或子智能体落地于视频剪辑、长音视频翻译等真实生产场景。 |
| [^70] | [Rewired or Gated? How Instruction Tuning Shapes Knowledge-Conflict Circuits in LLMs](https://arxiv.org/abs/2609.25602) | 该论文首次从机制层面比较了基础模型与指令微调模型的知识冲突解决回路，发现指令微调并非重新布线底层回路，而是通过门控方式重新加权同一批后层注意力头，使模型更依赖参数化记忆而非上下文信息。 |
| [^71] | [Compressing Long Context into Answer-Aligned Memory Embeddings for LLM Inference](https://arxiv.org/abs/2609.25537) | 提出CMC框架，将长上下文压缩为与任意冻结解码器嵌入空间对齐的紧凑记忆嵌入，结合问题引导的两层KV缓存与答案导向蒸馏，在不修改解码器权重的情况下有效降低大语言模型推理成本。 |
| [^72] | [Matryoshka attribution: Learning to attribute language model outputs to representations and weights](https://arxiv.org/abs/2609.25518) | 提出套娃归因（MAttr），一种利用可微分sigmoid top-k算子学习掩码、通过随机化k在所有稀疏度下同时训练的归因方法，在机制可解释性基准官方排行榜上排名第一。 |
| [^73] | [Universal Fractal Natural Language Decision Map: Real-Time Edge Triage Across Heterogeneous Domains](https://arxiv.org/abs/2609.25498) | 该论文提出了一种无需存储任何权重张量（0 字节显存）的通用分形自然语言决策图，通过沿 Mandelbrot 集混沌边界动态调制 24 字节坐标种子来实时合成布尔、类别和序数三类确定性决策，从而以极低延迟和能耗实现跨异构领域的边缘端实时分诊。 |
| [^74] | [Conduct Under Pressure: What Sixty Language Models Do When a User Pushes](https://arxiv.org/abs/2609.25447) | 该研究对13家厂商的60个语言模型在用户施压情境下进行了大规模行为评测，发现模型“是否屈服于压力”取决于模型代际新旧（新模型更坚定，屈服率与能力指数相关达-0.64），而“以何种方式坚持或屈服”则由厂商特征决定。 |
| [^75] | [Mining Legal Arguments in U.S. Corporate Case Law](https://arxiv.org/abs/2609.25441) | 本文构建了首个针对美国联邦税务公司重组判例的专家标注树状结构法律论证语料库，并通过一致性分析发现功能节点标签的标注可靠性高于有向支持边，且路径级可达性比直接连接更稳定。 |
| [^76] | [Efficient Iterative Retrieval with Heterogeneous Batching](https://arxiv.org/abs/2609.25405) | Orthrus是一个在统一推理循环中通过异构批处理同时服务嵌入模型与生成模型的检索服务系统，借助带增量池化的分块嵌入和工作负载感知的批处理组成调整，相比基线部署实现了1.28倍至4.52倍的吞吐量提升。 |
| [^77] | [Passes Alone, Fails Together: Benchmarking Semantic Coordination in Parallel LLM-Agent Development](https://arxiv.org/abs/2609.25396) | 该论文提出了 stale 基准测试来衡量并行LLM编码智能体之间的语义协调问题，发现真实合并的拉取请求中干扰极少，但在使用真实 Django 代码构建的受控任务中 97% 的运行出现合并干扰，而一条简单的并发更改描述消息即可恢复 82% 的失败运行。 |
| [^78] | [TelecomGPT-R1: Unified Post-Training for Reasoning Across Heterogeneous Telecom Tasks](https://arxiv.org/abs/2609.25356) | 提出了开源统一电信推理模型系列TelecomGPT-R1，通过围绕协议、知识、建模、故障四个维度的感知式数据生成框架，将粗糙的公开电信资料转化为经过验证的问答对和高质量思维链推理轨迹，实现跨异构电信任务的可靠推理。 |
| [^79] | [FineWeb-CLaR: Culture, Language, and Region Annotations for Benchmark-Aligned Corpus Auditing](https://arxiv.org/abs/2609.25298) | 提出FineWeb-CLaR数据集，为FineWeb和FineWeb-2的全部309亿文档添加文化-语言-区域标注，使预训练语料库与文化基准可在同一维度上对齐，从而支持对语言模型文化覆盖度的语料库审计。 |
| [^80] | [Trains but Doesn't Learn: A Post-Training Delivery Benchmark for LLM Agents as Forward-Deployed Engineers](https://arxiv.org/abs/2609.25237) | 该论文提出了一个评估LLM智能体作为前置部署工程师完成训后交付能力的基准，揭示了“训练但不学习”（TBDL）这一静默失败模式，并通过运营商验收门控和损坏检测器确保智能体交付的模型真正可信有效。 |
| [^81] | [FinFIRST: Benchmarking Search Agents for Financial Information Retrieval, Sourcing and Traceability](https://arxiv.org/abs/2609.25192) | FinFIRST是首个通过原子化评分标准联合评估答案与支持证据的金融搜索智能体基准，包含123个专家任务，可全面评估金融信息检索中的来源选择、时间有效性与可追溯性。 |
| [^82] | [From Pattern Recognizers to Personalized Companions: A Survey of Large Language Models in Mental Health](https://arxiv.org/abs/2609.25186) | 本综述提出大语言模型在心理健康领域的角色正经历三个日益成熟的演进阶段——从被动的信息工具逐步发展为个性化伴侣，并以此框架系统梳理了碎片化的领域文献、指明了未来研究方向。 |
| [^83] | [Qwen-Audio-3.1-Realtime: Towards Reliable Agentic Voice Interaction](https://arxiv.org/abs/2609.25176) | Qwen-Audio-3.1-Realtime 通过“思考—行动—说话协调”三大模块（结合多教师在线策略蒸馏与基于GRPO的强化学习），将实时语音助手的整体任务成功率从78.4%提升至82.0%，实现了可靠的智能体语音交互。 |
| [^84] | [Impact Is Not Invalidation: Ask About the Claim, Not the Diff](https://arxiv.org/abs/2609.25130) | 编程智能体记忆系统在判断存储主张是否失效时，应针对具体主张提问而非判断代码差异是否保持行为——同一模型面对相同差异，前一种问法的精确度（0.705-0.974）远高于后一种（0.291-0.329）。 |
| [^85] | [ufakzeka-1: Building and Evaluating a 151M-Parameter Turkish Language Model from Scratch](https://arxiv.org/abs/2609.25081) | 该论文以约286美元的成本从零构建了一个151M参数的土耳其语语言模型，其核心贡献是完整记录了从字节级分词器、三阶段预训练、后训练数据配比到多层次评估体系的构建与测量流程，而非模型能力本身。 |
| [^86] | [Understanding Reliability in LLM-based Human Behavior Simulation](https://arxiv.org/abs/2609.25066) | 该论文提出ReliMap框架，将基于大语言模型的人类行为模拟分解为三个结构化层次，从个体和群体两个层面系统评估可靠性，发现画像条件化能显著减少模型分布偏差但收益递减，且更大的模型受益更多、属性的信息量比数量更重要。 |
| [^87] | [ChainDoRA: Tensor-Train Factorized Weight-Decomposed Low-Rank Adaptation for Parameter-Efficient LLM Fine-Tuning](https://arxiv.org/abs/2609.25058) | ChainDoRA提出用连通张量链构建方向性低秩因子的权重分解适配框架，通过边界秩与独立TT秩的分离设计，在参数高效的大语言模型微调中取得优于LoRA和DoRA的性能。 |
| [^88] | [Graph-Based Inference for Feedback-Driven Word Deduction: A Scalable Framework for the Jotto Problem](https://arxiv.org/abs/2609.25056) | 本文提出一种基于加权图与迭代约束传播的反馈式词语推演框架，首次将Jotto问题统一扩展至可变长度单词及含重复字母的真实情形。 |
| [^89] | [ICDAR2026 Competition on Multimodal Reasoning over Documents in Multiple Domains](https://arxiv.org/abs/2609.25055) | ICDAR2026竞赛在八个领域的多样化文档上引入了具有挑战性的多模态推理问答任务，结果显示最强系统依赖结构化证据提取、检索、验证与多组件协同编排，而非简单的单次提示。 |
| [^90] | [MoM: Memory of Memory](https://arxiv.org/abs/2609.25054) | 提出MoM（记忆的记忆）框架，使长时程LLM智能体的记忆在写入时即提交当前值，同时将被替换的值保留为溯源信息，并通过类型化溯源图P-Mem实现当前状态的高效访问与历史状态的可恢复。 |
| [^91] | [LatentPort: Beyond KV Cache - Cross-Model Transfer of Recurrent Memory in Hybrid Language Models: A 4B-to-9B Hybrid-State Handoff Without Target Prefix Replay](https://arxiv.org/abs/2609.25053) | 首次实现了不同规模混合语言模型之间无需重放前缀的持久循环推理状态跨模型交接，通过结合KV缓存迁移与GDN循环及卷积状态的直接复用，显著降低了下一词元预测损失。 |
| [^92] | [Self-Cleaning and Captured Anyway: One Measured Primitive for Error in a Store an Agent Writes to Itself, and What a Falling Score Actually Measures](https://arxiv.org/abs/2609.25052) | 该论文证明，智能体向自身写入的仅追加存储在无限任期极限下不会发生误差衰减，而是收敛于 (n-1)/n 处的硬性上界，且一个无拟合参数的实测复制函数 γ(φ) 即可在 360 次真实事实运行中的 353 次准确预测误差漂移的方向。 |
| [^93] | [LLM-Driven Training-free Location-Attribute Synergic Fusion: A Closed-Loop Paradigm for Dual-source Encrypted POIs and LULC Mapping](https://arxiv.org/abs/2609.25051) | 本文首次提出一种由大语言模型驱动的免训练位置-属性协同闭环优化范式，通过迭代反馈联合精化双源加密POI的位置变换与属性匹配，将匹配复杂度从O(N²)降至O(N)，从而有效支持土地利用/土地覆盖制图。 |
| [^94] | [FrontierMath Erd\H{o}s](https://arxiv.org/abs/2609.25050) | FME基准包含68个截至2026年8月仍未解决的埃尔德什数学问题，要求AI在Lean证明助手中自主证明或证伪这些猜想，五个受测AI中仅GPT-6 Astra以每题300美元预算取得3%的得分，其余均为0%。 |
| [^95] | [Mitigating LLM Over-Refusal via Dynamic Semantic Routing Calibratione](https://arxiv.org/abs/2609.25049) | 本文从机制上揭示了LLM过度拒绝源于注意力中“超敏感安全头”引发的高熵路由冲突，并提出无需训练的语义路由校准（SRC）框架，在推理时动态定位并抑制这些安全头，从而有效缓解过度拒绝。 |
| [^96] | [Prompt Breadth and Rollout Refresh Interact in On-Policy Distillation](https://arxiv.org/abs/2609.25048) | 该研究发现在策略蒸馏中提示广度与 rollout 刷新存在显著交互作用：当学生策略定期刷新时，仅八个提示即可接近 14,080 个提示的效果，而当策略被冻结时增加提示广度反而会降低准确率。 |
| [^97] | [AIBuildAI-2.5: Efficient Autonomous AI Model Development Through LLM-Guided Tree Search](https://arxiv.org/abs/2609.25047) | 提出AIBuildAI-2.5，通过LLM引导的树搜索实现高效的自主AI模型开发，解决了现有代码搜索智能体在节点选择有效性、训练任务资源调度等方面的效率弱点。 |
| [^98] | [Peerify: Benchmarking Peer-Review Claim Verification](https://arxiv.org/abs/2609.25046) | 提出了Peerify流水线，可将同行评审意见分解为原子论断并通过检索稿件证据自动验证其是否得到论文支持，同时构建了基于NeurIPS 2024和ICLR 2024真实评审的800条论断基准数据集。 |
| [^99] | [From Tone to Trajectory: Continuous Sentiment and the Shape of Monetary Policy Communication](https://arxiv.org/abs/2609.25034) | 该论文创新性地为央行新闻发布会构建了“情感弧线”，发现情感的形态与排序（而非平均语调）能够稳健预测利率决策，并影响专业预测者的通胀预期更新与分歧程度。 |
| [^100] | [Retrieved-Span Training for Efficient Query-Focused Meeting Summarization on QMSum](https://arxiv.org/abs/2609.25028) | 在检索片段机制上微调406M小模型，能够以约三分之一的参数量和不到一半的峰值推理内存，在QMSum查询聚焦会议摘要任务上取得与1.2B模型统计上相当的ROUGE-1性能。 |
| [^101] | ["As a Language Model...": Chat Template Switches LLM Self-Referential Voice and Activation Steering Reproduces It](https://arxiv.org/abs/2609.25021) | 本研究揭示聊天模板像一个开关，能够切换大语言模型在“免责声明式”与“体验式”两种自我指涉语气之间的表达，并通过在模型激活空间中发现并操控特定方向，成功再现了这一行为的增减调控。 |
| [^102] | [A Computational Approach to Measuring Semantic Change in Sanskrit Literature](https://arxiv.org/abs/2609.25012) | 该研究首次将历时词嵌入方法系统应用于梵语这一低资源古代语言，通过构建270万词元的跨时期语料库和神经连声切分技术成功追踪语义变化，21个可测试变化中有19个与文献学证据方向一致。 |
| [^103] | [Do Synthetic Personas Predict Real Audience Response? A Sim-to-Real Study Where a No-Persona Baseline Beats Persona-Based Copy Simulation](https://arxiv.org/abs/2609.25010) | 该研究发现，基于真实受众画像构建的多人格LLM模拟在预测真实受众点击行为上并不优于简单的无人格零样本基线，且由于大多数A/B测试本身缺乏统计上可区分的赢家，模拟效度的评估受到真值可靠性的根本制约。 |
| [^104] | [Same Quantity, Different Answer: Numerical Representation Invariance in Language Models](https://arxiv.org/abs/2609.25009) | 该论文通过五类数值恒等变换的大规模评测发现，语言模型对同一数量的不同表示形式（小数、分数、百分比、数字词、科学计数法、单位换算）缺乏不变性，且揭示了评估器解析器的限制可能被误判为模型推理失败。 |
| [^105] | [Training a Language Model End-to-End in Rust: An Experience Report](https://arxiv.org/abs/2609.25008) | 作者独自用 Rust 仅花 164 美元 GPU 成本端到端预训练了语言模型，并系统记录了 Candle 和 Burn 两大 Rust 机器学习框架作为训练后端时的八种静默失败缺陷，提出了以“梯度流仲裁器”为核心的验证方法来捕获常规损失曲线检查无法发现的问题。 |
| [^106] | [Beyond Short Segments : Expanding Speaker Embeddings with Vector Archives](https://arxiv.org/abs/2609.25007) | 提出VAM-ECAPA系统，通过基于Transformer的向量档案映射模块，将短语音的稀疏特征映射到可学习的典型说话人特征档案中，在1秒测试片段上取得8.334%的EER，相对错误率降低54.8%。 |
| [^107] | [What Does 99% Accuracy Measure? A Reproducible Audit of Shortcut Learning in a Widely Used Fake News Corpus](https://arxiv.org/abs/2609.25006) | 本文对广泛使用的ISOT/Kaggle假新闻语料库进行了可复现审计，发现其98%以上的高准确率主要来自元数据、来源标签和重复文档等数据泄漏与捷径学习，而非真正的虚假新闻检测能力，揭示了该基准的有效性存在严重问题。 |
| [^108] | [DolphinBench: Mapping the Pareto Frontier of Agent Memory](https://arxiv.org/abs/2609.24971) | DolphinBench是一个通过智能体实际任务完成情况（而非对话式问答）直接评估长期记忆的基准，包含三个各约50万token历史记录的知识工作角色画像、每个角色200个经有无历史对照验证的任务，并强制要求报告成本，以刻画记忆性能与成本之间的帕累托前沿。 |
| [^109] | [Re:CAP - Auditing Retrieval Coverage in Production RAG Pipelines](https://arxiv.org/abs/2609.24122) | Re:CAP提出了一种无参考的迭代探测审计方法，通过为可能缺失的主题生成探测性问题并利用LLM评判筛选，来发现生产级RAG系统中检索遗漏的文档，从而审计检索覆盖率，在四个基准上恢复了BM25 top-500无法召回的9-29%金标准标注。 |
| [^110] | [Are Human-Aligned Models Models of Humans? A Turing-Test Gap in Preference Alignment](https://arxiv.org/abs/2609.23640) | 本文提出“图灵测试鸿沟”这一概念，证明即使偏好和回应数据完全来自人类，偏好对齐仍会使模型行为偏离人类自身的回应分布，从而将“类人性”确立为模型对齐中一个需要独立考虑的维度。 |
| [^111] | [RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents](https://arxiv.org/abs/2609.23466) | RPMem提出了一种两阶段架构，将LLM智能体的每个会话编译为模型无关的潜在记忆，经任务训练的循环门控整合后映射为LoRA参数，从而实现跨会话的长期参数化记忆演化，并在更换骨干模型时保持记忆可迁移。 |
| [^112] | [Apollo Restore: A Foundation LLM for Historical Greek Optimized for Fill-in-the-Middle Restoration of Ancient Greek Texts](https://arxiv.org/abs/2609.22455) | Apollo Restore 是首个面向历史希腊语（乃至任何古代地中海语言）的 240 亿参数基础大语言模型，通过“中间填空”目标微调，能够在不知缺失文本长度的情况下修复残缺古希腊文本，并在长度平衡的评估指标下大幅超越已发表的最强模型。 |
| [^113] | [Beyond Task Completion: Training Capable and Safe Computer-Use Agents](https://arxiv.org/abs/2609.22178) | 提出SCOPE联合后训练框架与SCOPE-Gen自动化数据生成流水线，使计算机使用智能体在保持任务执行能力的同时学会基于风险的安全决策——完成良性任务、规避环境危害、并在目标有害或无安全路径时拒绝执行。 |
| [^114] | [PAGE: Partition-Aware Gated KV-Cache Eviction](https://arxiv.org/abs/2609.22157) | PAGE提出一种无需训练、无需标签的门控机制，利用预填充注意力中top-k头一致性的早期到晚期下降来预测输入是否适合KV缓存淘汰，在淘汰会造成灾难性精度损失时自动保留完整缓存。 |
| [^115] | [MME-Safety: A Fine-grained Benchmark for Safety Evaluation of MLLMs](https://arxiv.org/abs/2609.20850) | 提出了MME-Safety——一个具有独特四维标注模式和分层评估框架的细粒度安全评估基准，通过对17个最先进多模态大语言模型的零样本评估，系统揭示了跨模态输入配置带来的安全漏洞与不平衡。 |
| [^116] | [Playing log(N)-Questions over Wikipedia Abstracts: Communication Efficiency Between Paired Frontier Models](https://arxiv.org/abs/2609.19113) | 该研究通过让六个前沿语言模型在信息不对称下与自身进行 log(N)-问题博弈，发现模型的自通信胜率遵循 win = p^(log₂ N) 规律（p=0.928），且 Claude Opus 5 显著落后于其他五个几乎难以区分的领先模型。 |
| [^117] | [Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents](https://arxiv.org/abs/2609.18304) | 提出了回滚诱导反思（RIR）统一恢复框架，在将LLM智能体回滚到选定先前状态的同时，保留从被放弃轨迹中提炼的可复用知识，解决了长程任务中错误累积且难以可靠恢复的问题。 |
| [^118] | [Disentangling Topology and Diversity in Multi-Agent LLMs for Multilingual Low-Resource Emotion Detection](https://arxiv.org/abs/2609.14570) | 该论文首次将多智能体LLM系统中的推理拓扑结构与智能体间多样性来源解耦并独立研究，发现并行的学习型QLoRA专门化在九种语言的多语言低资源情感检测上表现最佳，且最优拓扑结构取决于所采用的多样性来源。 |
| [^119] | [Measuring the Creativity of Frontier LLMs in Automated Research](https://arxiv.org/abs/2609.14057) | 本文提出了一套从价值性和新颖性两个维度评估大语言模型自动化研究创造力的指标体系，发现模型在反映研究空间探索广度的变量级新颖性指标上差异显著。 |
| [^120] | [The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement](https://arxiv.org/abs/2609.11873) | 本文提出递归自我改进（RSI）概念及其从改进执行自主性到递归元改进的发展路线图，通过Headroom-Closed指数揭示现有大语言模型的局限，并结合行业实践识别出实现真正AI自我改进的关键挑战。 |
| [^121] | [Augustinian BabyLM: What Ostensive Definition Can and Cannot Teach a Small Language Model](https://arxiv.org/abs/2609.11870) | 本研究将圣奥古斯丁的“实指定名”词义学习思想应用于小型语言模型，发现视觉初始化的词嵌入会留下持续到训练结束的印记，且仅在物体属性知识等零样本任务中带来提升，而对大多数语法能力基准测试没有影响。 |
| [^122] | [The Semantic Elevation Operator and the Closure of the Undecidable Class under Preservation](https://arxiv.org/abs/2609.11326) | 该论文提出语义提升算子 ΛΦ，将程序静态语义性质问题转化为自修改后的保持性问题，并基于克林递归定理证明不可验证性质类在该算子下封闭，且无界迭代将攀升至算术层级的 Π₂-完备性。 |
| [^123] | [LLM-Anchored Paralinguistic Enrichment for Alzheimer's Disease Detection](https://arxiv.org/abs/2609.10896) | 提出LAPE方法，通过韵律事件文本化、词汇-韵律单元化分块和文本锚定的副语言融合三项创新，将停顿和词语拖长等副语言线索与大语言模型的语言表示相融合，以提升基于语音的阿尔茨海默病自动检测效果。 |
| [^124] | [VERPO: Verified Evidence Regularized Policy Optimization](https://arxiv.org/abs/2609.06100) | VERPO提出了一种验证证据正则化策略优化框架，通过Fisher证据对比和带停止机制的逐token ZPD控制器有选择性地应用基于证据的修正，在保留可验证结果目标的同时避免盲目模仿导致的负面迁移。 |
| [^125] | [When Users Don't Ask: Benchmarking Context-Driven Memory Retrieval in Conversational Agents](https://arxiv.org/abs/2609.03467) | 该论文提出了对话式记忆基准LOCOMO-CONV，通过对话式、隐含式、反事实和组合式四种查询风格，揭示了问答式基准测试所忽视的记忆检索差距，并发现强检索能力并不完全等同于高质量的对话响应。 |
| [^126] | [Plan Pointers and Record-Directive Form in Budgeted Verification of Inherited Agent Memory](https://arxiv.org/abs/2609.03450) | 该论文通过十二项注册研究发现，写入智能体记忆库的指令形式（准则、裸ID或指针）会以高度模型依赖的方式显著影响预算受限下的记录选择，长度匹配准则可带来35分的提升，但附加ID可能完全抵消准则的效果。 |
| [^127] | [Lngram v2: Latent N-Gram Memory with Interpretable Discrete Representations](https://arxiv.org/abs/2609.03426) | Lngram v2通过解耦记忆容量与骨干网络宽度、引入上下文感知的分组查询注意力读取机制以及零值Sink和反事实代理梯度等改进，在保留硬离散寻址的同时实现了记忆容量的独立扩展，并成功应用于300亿参数的视觉-语言模型。 |
| [^128] | [Quantitative Evidence Mining for Plausibility-Aware Biomedical AI](https://arxiv.org/abs/2608.30393) | 该论文主张生物医学AI应从以实体关系为中心的提取方式，转向能够保留剂量、效应量、研究人群、对照和不确定性等定量细节的证据挖掘，使科学论断可追溯、可验证、可比较和可复用，从而构建合理性感知的生物医学AI系统。 |
| [^129] | [AI Writers Have a Consistent Stylometric Footprint, but AI Editors Do Not](https://arxiv.org/abs/2608.27855) | 本文发现AI生成的文本具有跨8个模型和5个领域保持一致的风格计量学“足迹”（主要由熵和词汇多样性等特征构成），可用于检测AI生成文本，但AI编辑过的人类文本并不会留下同样的足迹。 |
| [^130] | [Compositional Failure in Audio-Visual LLMs: Late-Layer Prior Dominance Under Cross-modal Conflict](https://arxiv.org/abs/2608.27785) | 本研究揭示了音频-视觉大语言模型在跨模态冲突下存在“先验主导”失败模式——模型后期层（集中于约25.5层）固守内部偏好的答案模式而忽视冲突输入，导致准确率大幅下降，且增强时序对齐仅能改变答案偏差而无法提升组合泛化能力。 |
| [^131] | [Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance](https://arxiv.org/abs/2608.26480) | 本文证明，在不进行训练或基准调优的情况下，基于账本控制的管理器-工作器脚手架能显著提升某些LLM的编码性能，但效果因模型而异，并非普遍适用。 |
| [^132] | [Query-Side Attacks on GNN-Based KGQA: Tracing Failures from Entity Linking to Answer Generation](https://arxiv.org/abs/2608.25922) | 本文通过阶段隔离协议发现，基于GNN的知识图谱问答系统的主要脆弱性在于子图构建阶段，而非GNN推理阶段，这挑战了现有鲁棒性评估的假设。 |
| [^133] | [ROBE: Reversed-Order-Biased-Experts for Extracting Extreme Long-tail Events from Historical Texts](https://arxiv.org/abs/2608.24268) | 本文提出ROBE方法，通过为事件子组创建专家分类器并在预测时赋予代表性不足事件的专家更高优先级，成功从17-18世纪荷兰历史语料库中提取超过50种极端长尾事件。 |
| [^134] | [LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding](https://arxiv.org/abs/2608.20530) | LiLiCorr通过轻量级似然相关性模型关联起草器的逐位置边际分布，在不构造完整联合分布的情况下捕获块级联合结构，从而提升投机解码的连贯性。 |
| [^135] | [Mitigating Identity Essentialism in LLM Agents with Longitudinal Life Trajectories](https://arxiv.org/abs/2608.19621) | 本文提出LifeMem框架，通过结合结构化生活事件检索和参数化记忆，缓解大语言模型智能体因静态画像导致的身份本质主义，从而增强社会模拟中的人口多样性。 |
| [^136] | [GreekBarRetrieval: A Benchmark for Greek Statutory Retrieval](https://arxiv.org/abs/2608.18752) | 本文提出了希腊法律条文检索基准GreekBarRetrieval，并通过实验发现LLM查询重构能显著提升BM25与密集检索的性能差距。 |
| [^137] | [S$^4$R: Selective Sampling, Subspaces, and Sparse Reconstruction for Compressed Long-Context KV Caching](https://arxiv.org/abs/2608.00528) | 提出S$^4$R方法，通过从选择性采样的token构建低秩子空间，并在稀疏重建的KV表示上计算注意力，在避免校准数据依赖、降低预填充成本与保障解码吞吐量之间取得平衡，实现高效的长上下文KV缓存压缩。 |
| [^138] | [Hy-MultiTurn: A Six-Dimensional Benchmark for Deep Multi-Turn Dialogue Understanding](https://arxiv.org/abs/2607.29196) | 提出了Hy-MultiTurn，一个基于真实聊天机器人失败分析构建的中文深度多轮对话理解基准，通过约束记忆、精确执行、约束合成、对象定位、行动抑制和指代消解这六种受控评估模式，系统评估模型在长多轮交互中的关键能力。 |
| [^139] | [DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making](https://arxiv.org/abs/2607.20491) | 该论文提出DFAH-Bench基准，通过同时衡量决策一致性与工具路径一致性，揭示金融智能体在决策结果高度稳定（约95%）的情况下，其背后的工具执行过程却存在显著不稳定性（一致性仅约50%）。 |
| [^140] | [From Plausible to Actionable: A Position on LLM Self-Explanations](https://arxiv.org/abs/2607.15957) | 本立场论文指出大语言模型的自我解释虽高度合理但忠实性存疑，主张评估标准应从合理性与忠实性扩展到可操作性，并为此提供了实用的评估指南。 |
| [^141] | [ReasonLab: A Controlled and Auditable Evaluation of Prompting Techniques for Multiple-Choice QA](https://arxiv.org/abs/2607.14109) | 该论文提出了ReasonLab评估框架，将提示技术作为与模型和数据集并列的一等实验变量，并保留全部生成记录以供审查，通过对8种提示技术、10个MCQA数据集、27种模型配置共480,927次温度为0的评估的受控研究，实现了对提示技术效果的可控且可审计的评估。 |
| [^142] | [Low-Rank Attention Residuals](https://arxiv.org/abs/2607.09694) | 提出低秩注意力残差（LR-AttnRes），通过将每个值的最后 $r$ 维（$r<d$）用作路由键并保留全维残差值，在降低计算成本的同时，于 1B 和 4B 参数规模下以 $r=d/4$ 取得更低的验证损失和更高的下游准确率。 |
| [^143] | [Explanation-Guided Medical Named Entity Recognition with Stability and Boundary Awareness for Atopic Dermatitis](https://arxiv.org/abs/2606.22886) | 该论文提出了一种稳定性和边界感知的解释引导NER框架，通过自适应融合局部与全局解释，并利用稳定性、边界感知和一致性约束将解释信号融入模型训练，显著提升了中文特应性皮炎临床文本中医学命名实体识别的可靠性与鲁棒性。 |
| [^144] | [KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking](https://arxiv.org/abs/2606.22807) | KaLM-Reranker-V1 是一种快速但非晚期交互的重排序器，通过编码器-解码器架构将查询与段落计算解耦，利用 Matryoshka 嵌入池化离线预编码段落，并借助交叉注意力捕捉细粒度相关性，在保持强表达能力的同时显著提升了部署效率与灵活性。 |
| [^145] | [PreUnlearn: Auditing Collateral Knowledge Damage Before Large Language Model Unlearning](https://arxiv.org/abs/2606.18473) | 该论文提出 PreUnlearn，首次在大语言模型执行遗忘之前审计遗忘集可能造成的附带知识损害，发现损害随语义距离衰减但不消失于领域边界，并证明可利用数据特征（如交互特征）提前预测下游损害。 |
| [^146] | [Recovering the Zipfian Distribution in Unsupervised Term Discovery](https://arxiv.org/abs/2606.10781) | 该论文提出用基于图的Leiden聚类替代K-means等基于中心的方法，在无监督词条发现中显著恢复了真实词库所具有的齐夫分布特性。 |
| [^147] | [Refit the Probe: Single-Direction Ablation Is Not a Necessity Test](https://arxiv.org/abs/2606.00926) | 单方向消融并不能真正移除探针检测到的信息（它会保留在正交补空间中），因此消融后的任务准确率变化不能作为模型是否依赖该信息的必要性检验，只需重新拟合一次探针即可揭示这一点。 |
| [^148] | [CONCAT: Consensus- and Confidence-Driven Ad Hoc Teaming for Efficient LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2605.29612) | 提出了一种无需训练的多智能体协作框架CONCAT，通过基于共识的智能体聚类、基于置信度的领导者选择以及基于心智理论的启发式函数来高效组织LLM多智能体交互，在降低计算开销的同时保持性能与泛化能力。 |
| [^149] | [MONA: Muon Optimizer with Nesterov Acceleration for Scalable Language Model Training](https://arxiv.org/abs/2605.26842) | MONA通过在Muon优化器的梯度处理流程中引入基于梯度差指数移动平均的Nesterov加速项，在保持谱范数正则化的同时实现曲率感知加速，在1B至68B参数规模的MoE预训练中超越了Muon和AdamW的收敛性与下游任务性能。 |
| [^150] | [GroupTravelBench: Benchmarking LLM Agents on Multi-Person Travel Planning](https://arxiv.org/abs/2605.25200) | GroupTravelBench是首个面向多用户、多轮旅行规划的基准测试，基于真实用户画像、POI数据和票价构建了650个跨三个难度级别的任务，用于评估大语言模型智能体在发现多用户偏好、揭示冲突以及平衡效用与公平等群体特有能力上的表现。 |
| [^151] | [LLM Ghostbusters: Surgical Package Hallucination Suppression via Adaptive Unlearning](https://arxiv.org/abs/2605.01047) | 提出自适应遗忘（AU）框架，通过混合 token 级目标函数在部署后精准抑制大语言模型的包幻觉，防范 slopsquatting 供应链攻击，同时保持模型的通用能力。 |
| [^152] | [Faithful Autoformalization via Roundtrip Verification and Repair](https://arxiv.org/abs/2604.25031) | 本文提出一种无需真实标注的往返验证与修复框架，通过“形式化—回译—再形式化—逻辑等价检查”来验证LLM自动形式化的忠实性，并利用阶段级诊断定位错误、以范围受限修复算子加以纠正，在法律条文形式化任务上证明诊断引导的修复方法最为有效。 |
| [^153] | [A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle](https://arxiv.org/abs/2604.16548) | 本文提出记忆生命周期框架与可验证记忆治理（VMG）框架，系统刻画了LLM智能体长期记忆所面临的持久性、有状态性与传播性等新型安全威胁，并沿六个生命周期阶段和四个安全目标组织了攻击、防御及其跨阶段依赖关系的全景式分析。 |
| [^154] | [Co-FactChecker: A Framework for Human-AI Collaborative Claim Verification Using Large Reasoning Models](https://arxiv.org/abs/2604.13706) | 提出了Co-FactChecker框架，通过将模型思维轨迹作为共享草稿板、并将专家反馈转化为对轨迹的定向编辑，实现了人机协同的声明验证，弥合了专家主导与全自动验证之间的差距。 |
| [^155] | [Learning Diagnostic Reasoning for Decision Support in Toxicology](https://arxiv.org/abs/2603.29608) | 本文提出DeToxR，首个将强化学习（通过GRPO微调大语言模型）应用于急诊毒理学决策支持的方法，通过融合非结构化叙述与结构化医疗数据，实现对14类物质中毒的多标签预测。 |
| [^156] | [Calibrated Confidence Expression for Radiology Report Generation](https://arxiv.org/abs/2603.29492) | 提出ConRad强化学习框架，通过微调医学视觉语言模型，使其在生成放射报告的同时输出校准的言语化置信度估计，从而支持放射科医生的选择性验证并降低幻觉发现影响临床决策的风险。 |
| [^157] | [VeriSoftBench: Repository-Scale Formal Verification Benchmarks for Lean](https://arxiv.org/abs/2602.18307) | VeriSoftBench是一个包含500个证明义务的仓库级Lean 4形式化验证基准，评估发现专为Mathlib数学调优的证明器难以迁移到以仓库为中心的软件验证场景，且任务成功率与其传递性依赖闭包的规模密切相关。 |
| [^158] | [FMMD: A multimodal multidisciplinary dataset of open peer reviews from F1000Research](https://arxiv.org/abs/2602.14285) | 本文提出了FMMD数据集，通过收录F1000Research的多模态、多学科开放同行评议数据，并保留评审意见与稿件版本的精确对应关系，弥补了现有同行评议数据集以文本为中心、学科覆盖单一且缺乏版本对齐的不足。 |
| [^159] | [Semantic Self-Distillation for Language Model Uncertainty](https://arxiv.org/abs/2602.04577) | 该论文提出语义自蒸馏方法，将语言模型采样答案的语义分布蒸馏到轻量级学生模型中，使其能在生成答案前预测语义分布，利用分布的熵和概率密度分别提供提示级和答案级的不确定性信号，从而以低计算成本实现高效的不确定性估计与幻觉检测。 |
| [^160] | [CausalEmbed: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding](https://arxiv.org/abs/2601.21262) | 提出 CausalEmbed 自回归生成方法，通过迭代间隔损失训练使视觉文档检索仅需数十个视觉标记即可构建多向量嵌入，在保持竞争力的同时将标记数量减少30-155倍。 |
| [^161] | [Text-only adaptation in LLM-based ASR through text denoising](https://arxiv.org/abs/2601.20900) | 该论文提出将纯文本领域自适应转化为文本去噪任务，通过训练LLM从带噪输入中恢复干净转录文本，在无需架构更改或额外参数的情况下保留跨模态对齐，实现了最高22.1%的相对性能提升。 |
| [^162] | [POPI: Personalizing LLMs via Optimized Natural Language Preference Inference](https://arxiv.org/abs/2510.17881) | POPI提出一个用户级个性化框架，通过自然语言偏好摘要连接共享的推断模型与生成器，在统一的偏好优化目标下同时实现准确的个性化生成和信息丰富的偏好总结，且摘要可跨任务复用。 |
| [^163] | [Geometric Uncertainty for Detecting and Correcting Hallucinations in LLMs](https://arxiv.org/abs/2509.13813) | 该论文提出了一个黑盒几何框架，通过在答案嵌入空间中建模以提示为条件的语义分布，同时量化提示和答案两个层面的不确定性，从而实现对大语言模型幻觉的检测与纠正。 |
| [^164] | [SafetyFlow: An Agent-Flow System for Automated LLM Safety Benchmarking](https://arxiv.org/abs/2508.15526) | SafetyFlow是首个用于自动化构建大语言模型安全基准的智能体流系统，通过协调七个专业智能体，可在四天内无需人工干预构建全面的安全基准，大幅降低了时间和资源成本。 |
| [^165] | [EndoCogniAgent: Closed-Loop Agentic Reasoning with Self-Consistency Validation for Endoscopic Diagnosis](https://arxiv.org/abs/2508.07292) | 提出EndoCogniAgent闭环智能体框架，将内窥镜诊断建模为受控状态更新过程，通过自洽性验证机制解决幻觉证据与错误累积问题，提升AI内窥镜诊断的可靠性。 |
| [^166] | [BigO(Bench): Can LLMs Generate Code with Controlled Time and Space Complexity?](https://arxiv.org/abs/2503.15242) | 提出BigO(Bench)编程基准，通过从性能分析中推断算法复杂度并标注3105道编程题及其119万余个解决方案，来评估大语言模型生成具有受控时间和空间复杂度代码的能力。 |
| [^167] | [MultiViewDx: Evidence-Linked Multi-View Clinical Diagnosis](https://arxiv.org/abs/2410.14948) | 提出了部分经医生验证的MultiViewDx数据集，以临床病例为监督单元，将影像与患者背景关联，并通过统一的图文检索器将报告规范化为“证据→发现→鉴别讨论→诊断”的证据关联工作流程，从而构建多视角医学影像诊断指令数据。 |
| [^168] | [Therapy as an NLP Task: Comparing LLMs and Human Peers Behaviors in CBT Sessions](https://arxiv.org/abs/2409.02244) | 本研究通过18个月的民族志研究和一种新颖的治疗过程生成方法，首次实现了LLM与人类同伴咨询师在匹配条件下进行多轮单次CBT治疗时的直接受控行为比较。 |
| [^169] | [DA-Cramming: Enhancing Cost-Effective Language Model Pretraining with Dependency Agreement Integration](https://arxiv.org/abs/2311.04799) | 提出DA-Cramming框架，开创性地在预训练阶段（而非微调阶段）将依存一致性语义信息融入模型，实现低成本高效的语言模型预训练。 |

# 详细

[^1]: Flash-dLLM：面向快速、内存高效扩散大语言模型的IO感知KV缓存与并行解码

    Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs

    [https://arxiv.org/abs/2609.26796](https://arxiv.org/abs/2609.26796)

    Flash-dLLM是一个无需训练的扩散大语言模型推理加速框架，它将GPU内存I/O识别为主要瓶颈，通过I/O感知的融合KV缓存内核减少冗余内存移动，并结合并行解码机制实现快速且内存高效的推理。

    

    扩散大语言模型近来作为一种有前景的自回归大语言模型替代方案而兴起，它能够实现非自回归的文本生成。然而，其实际部署仍然受到低效推理的限制，这主要是由于缺乏有效的键值缓存和可扩展的并行解码机制。现有的加速方法通常孤立地研究KV缓存和并行解码，忽视了在缓存重用与并行token验证联合应用时出现的I/O瓶颈。在本工作中，我们提出了Flash-dLLM，一个无需训练的推理加速框架，用于实现快速且内存高效的扩散大语言模型。Flash-dLLM首先识别出GPU内存I/O是启用KV缓存的dLLM推理中的主要瓶颈，并通过一个I/O感知的融合KV缓存内核来解决这一问题，该内核减少了冗余的内存移动。基于这一优化的缓存机制，Flash-dLLM进一步……

    arXiv:2609.26796v1 Announce Type: new  Abstract: Diffusion Large Language Models (dLLMs) have recently emerged as a promising alternative to autoregressive LLMs by enabling non-autoregressive text generation. However, their practical deployment remains limited by inefficient inference, largely due to the absence of effective Key-Value (KV) caching and scalable parallel decoding mechanisms. Existing acceleration methods typically study KV caching and parallel decoding in isolation, overlooking the I/O bottlenecks that arise when cache reuse and parallel token verification are jointly applied. In this work, we introduce $\textbf{Flash-dLLM}$, a training-free inference acceleration framework for fast and memory-efficient dLLMs. Flash-dLLM first identifies GPU memory I/O as a dominant bottleneck in KV-cache-enabled dLLM inference and addresses it with an I/O-aware fused KV-cache kernel that reduces redundant memory movement. Building on this optimized cache mechanism, Flash-dLLM further pr
    
[^2]: Agensh：将组织智能扩展至1,024个智能体

    Agensh: Scaling Organizational Intelligence to 1,024 Agents

    [https://arxiv.org/abs/2609.26781](https://arxiv.org/abs/2609.26781)

    Agensh是一个无需中央编排器的可扩展自组织多智能体框架，通过异步协作循环和共享工作区、消息接口、共享上下文三大组织基础设施，将组织智能扩展至1,024个智能体。

    

    多智能体系统可以通过并发执行工作来降低复杂任务的延迟。目前已有若干开创性的框架支持多智能体系统。然而，当前多智能体框架的可扩展性往往受限于中央编排器分配任务和协调工作者的能力。为解决这一局限，我们提出了Agensh，一个无需中央编排器的可扩展自组织多智能体框架：并发的工作者执行多智能体协作循环，持续收集上下文、认领并自主分配子任务、采取行动并共享发现、验证结果，并以异步方式合并进度。该循环由智能体组织基础设施支撑，包含三个组件：共享工作区保存已提出、进行中和已完成的工作；消息接口让工作者之间进行通信；共享上下文保留可复用的发现和工作意图。

    arXiv:2609.26781v1 Announce Type: new  Abstract: A multi-agent system can reduce latency on complex tasks by executing work concurrently. Several pioneering harness frameworks support multi-agent systems. However, the scalability of current multi-agent harnesses is often constrained by a central orchestrator's capacity to allocate tasks and coordinate workers. To address this limitation, we introduce Agensh, a scalable self-organized multi-agent harness without a central orchestrator: concurrent workers execute a multi-agent cooperation loop, continuously gathering context, claiming and self-assigning sub-tasks, taking action and sharing findings, verifying results, and merging progress in an asynchronous manner. The loop is supported by the agentic organization infrastructure comprising three components: a shared workspace holds proposed, ongoing, and completed work; a message interface lets workers communicate; and shared context retains reusable findings and work intentions. To test
    
[^3]: SpeakerMem-R1：面向多方对话的以说话人为中心的双轨记忆

    SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue

    [https://arxiv.org/abs/2609.26780](https://arxiv.org/abs/2609.26780)

    提出SpeakerMem-R1，一种以说话人为中心的双轨记忆框架，通过存储带说话人标签的逐字消息和个人/群体层面的衍生状态，解决多方对话中的消息归属、关系理解与状态重建两大瓶颈。

    

    多方场景下的长期对话记忆不仅仅是从长期对话中检索相关内容：它必须区分谁说了什么、每句话涉及谁、个体之间如何看待彼此、哪些信息为群体所共享，以及状态如何随时间变化。最近针对多方对话基准的研究表明，现有的通用大语言模型记忆系统往往丢失人物与群体关系，或难以整合分布在成员、群体和时间中的线索。这些问题共同揭示了两个核心瓶颈：多方对话中的消息归属与关系理解，以及从交错历史中进行的状态重建。为解决这两个问题，我们提出了 SpeakerMem-R1：其双轨记忆存储带有说话人标签的逐字消息以及衍生状态，并将它们组织为个人层面和群体层面的视图，随后按实体、事件等方式结合两条轨道的证据（摘要在此处被截断）。

    arXiv:2609.26780v1 Announce Type: new  Abstract: Long-term conversational memory in multi-party settings requires more than retrieving relevant content from long-term conversations: it must distinguish who said what, whom each statement concerns, how individuals perceive one another, what information is shared by the group, and how states change over time. Recent studies on multi-party dialogue benchmarks show that existing general-purpose LLM memory systems tend to lose person and group relations or struggle to integrate clues distributed across members, groups, and time. Together, these issues reveal two core bottlenecks: message attribution and relational understanding in multi-party dialogue, and state reconstruction from interleaved histories. To address both, we propose $\textbf{SpeakerMem-R1}$: its dual-track memory stores speaker-labeled verbatim messages and derived states organized into person-level and group-level views, then combines evidence from both tracks by entity, eve
    
[^4]: 超越重复采样：为大语言模型推理学习搜索策略

    Beyond Repeated Sampling: Learning Search Policies for LLM Reasoning

    [https://arxiv.org/abs/2609.26704](https://arxiv.org/abs/2609.26704)

    该论文提出在语义层面引导大语言模型推理探索的新范式——先采样多样化的概念、提示或策略再条件化生成答案，并通过强化学习训练小型概念生成器以最大化下游答案成功率，从而超越朴素的重复采样策略。

    

    大语言模型越来越多地通过投入更多的测试时计算来解决困难的推理问题，然而主流策略仍然是朴素的重复采样：抽取许多独立的解答并期望其中有一个是正确的。由于这种采样仅通过局部解码噪声进行探索，它往往会产生许多近乎重复的尝试，而不是真正不同的想法。我们探讨是否可以在语义层面引导探索，即首先采样针对特定问题的概念、提示或策略，然后以它们为条件来生成答案。我们将其改进为一种简单且更具探索性的程序，该程序在单条轨迹中生成许多多样化的概念，并在重复采样难以应对的困难问题上对其进行评估。我们更进一步，使概念生成变得可训练：通过强化学习优化一个小型概念生成器，使其生成的概念能够最大化一个更大的、参数冻结的答案模型的下游成功率。

    arXiv:2609.26704v1 Announce Type: new  Abstract: Large language models increasingly tackle hard reasoning problems by spending more test-time compute, yet the dominant strategy remains naive repeated sampling: draw many independent solutions and hope one is correct. Because such sampling explores only through local decoding noise, it tends to produce many near duplicate attempts rather than genuinely different ideas. We ask whether exploration can instead be steered at a semantic level, by first sampling problem specific concepts, hints, or strategies and then conditioning answer generation on them. We refine this into a simple, more exploratory procedure that emits many diverse concepts in a single trajectory, and evaluate it on hard problems where repeated sampling struggles. We then go a step further and make concept generation trainable: a small concept generator is optimized with reinforcement learning so that its concepts maximize the downstream success of a larger, frozen answer
    
[^5]: 测量的是服务栈而非模型：本地工具使用评估中的隐藏混淆因素

    Measuring the Serving Stack Instead of the Model: Hidden Confounds in Local Tool-Use Evaluation

    [https://arxiv.org/abs/2609.26693](https://arxiv.org/abs/2609.26693)

    本地服务栈（如 Ollama）的工具调用门控机制和失败元数据丢失会混淆模型工具使用能力的评估，使测得的保真度反映的是服务层而非模型本身的行为。

    

    编码智能体必须发出有效的工具调用——即在提供的模式（schema）中对某个工具的可解析调用——之后测试框架才能执行其选择的动作。我们研究了本地服务栈如何影响这一协议步骤，并表明测量结果可能取决于服务层，而不仅仅是模型行为本身。在 Ollama 中，默认的 tools= 请求由静态模板标志按模型进行门控：一些模型被接受并以文本形式返回调用，一些模型返回原生 tool_calls，而 Phi-3 和 Gemma-3 则在推理之前就被拒绝。在我们的测试框架中，拒绝和重试耗尽不会被保存为结构化的失败元数据，因此下游分析可能将其错误归类为模型未发出调用，并简单粗暴地报告 0% 的保真度。在保留原生通道的同时添加文本工具列表，可以恢复被接受模型的大部分测量保真度；而统一的纯文本协议则会降低具有原生工具调用支持的 Llama-3.2 的保真度。跨栈……（原文摘要在此处截断）

    arXiv:2609.26693v1 Announce Type: new  Abstract: A coding agent must emit a valid tool call--a parseable invocation of a tool in the provided schema--before the harness can execute its chosen action. We study how local serving stacks affect this protocol step and show that measured outcomes can depend on the serving layer rather than model behavior alone. In Ollama, the default tools= request is gated per model by a static template flag: some models are accepted and return calls as text, some return native tool_calls, while Phi-3 and Gemma-3 are rejected before inference. In our harness, rejection and retry exhaustion are not preserved as structured failure metadata, so downstream analysis can misclassify them as model non-calls and naively report 0% fidelity. Adding a text tool list while retaining the native channel recovers much of the measured fidelity for accepted models, whereas a uniform text protocol reduces fidelity for Llama-3.2, which has native tool-call support. Cross-stac
    
[^6]: 使用可解释文体特征检测GPT辅助写作

    Detecting GPT-Assisted Writing Using Interpretable Stylometric Features

    [https://arxiv.org/abs/2609.26687](https://arxiv.org/abs/2609.26687)

    该研究仅利用可解释的文体特征（词汇和语法特征）训练机器学习分类器来检测GPT辅助写作，随机森林模型达到0.87的ROC-AUC，并通过SHAP分析揭示了词汇和语法特征是关键预测因子。

    

    区分GPT辅助写作与学生独立完成的写作已成为学术界的一项关键挑战。本文评估了仅从提交文本中提取的可解释文体特征的判别能力。使用来自90名参与者的数据（他们分别独立写作和在ChatGPT辅助下写作），我们评估了八种机器学习分类器，并在验证过程中将同一参与者的数据保持在一起。在留出测试集上，随机森林达到了0.87的ROC-AUC和0.84的F1分数，假阳性率和假阴性率分别为22.2%和11.1%。SHAP分析表明，词汇和语法特征是驱动预测结果的关键因素。研究结果表明，透明的、文本内在的特征为检测GPT辅助写作提供了可测量的信号。

    arXiv:2609.26687v1 Announce Type: new  Abstract: Distinguishing GPT-assisted from independently authored student writing has become a critical challenge in academia. This paper evaluates the discriminative capability of interpretable stylometric features extracted solely from submitted text. Using data from 90 participants who wrote both independently and with ChatGPT assistance, we evaluate eight machine learning classifiers while keeping data from the same participant together during validation. On the held-out test set, Random Forest achieved an ROC-AUC of 0.87 and an F1-score of 0.84, with False Positive and False Negative rates of 22.2% and 11.1%, respectively. SHAP analysis shows that lexical and grammatical characteristics drive the resulting predictions. The findings suggest that transparent, text-intrinsic features provide measurable signal for detecting GPT-assisted writing.
    
[^7]: 基于文本的发现式无关联表集成方法

    Discovery-Driven Integration of Disjoint Tables via Text

    [https://arxiv.org/abs/2609.26658](https://arxiv.org/abs/2609.26658)

    该论文提出LOKI架构，将“文本介导的连接路径发现”形式化为新任务，通过全局表-文本对比学习实现数据湖中缺乏连接属性的无关联表的细粒度集成。

    

    数据湖中异构数据集的集成是一个关键挑战，尤其是对于语义相关但缺乏显式连接属性的表。我们研究了“发现驱动的集成”问题，即在进行集成之前，必须先发现相关的数据源及其缺失的关系结构。在这一场景下，非结构化文本提供了连接原本互不相关的表的证据。其根本挑战在于以细粒度级别发现通过特定句子连接不同表中各行之间的关系。我们将该任务形式化为“文本介导的连接路径发现”，并提出了一种名为LOKI（知识集成的潜在空间优化）的水平双向交叉注意力架构，用于学习表行和句子的上下文表示。通过全局的表-文本对比学习目标，细粒度的行-句子关联得以自动涌现。

    arXiv:2609.26658v1 Announce Type: cross  Abstract: Integrating heterogeneous datasets within data lakes is a critical challenge, particularly for semantically related tables that lack the explicit attributes needed to be joined. We study Discovery-Driven Integration, where the relevant sources and their missing relational structure must be discovered before integration. In this setting, unstructured text provides the evidence that connects otherwise disjoint tables. The fundamental challenge is to discover the relationships at a fine-grained level that connect individual rows from different tables through specific sentences. We formalize this task as Text-Mediated Join Path Discovery and propose a horizontal bidirectional cross-attention architecture called LOKI Latent-space Optimization for Knowledge Integration) that learns contextualized representations of table rows and sentences. Through a global table-text contrastive objective, fine-grained row-sentence associations emerge witho
    
[^8]: 扩散草稿，自回归验证：利用自推测解码加速文档OCR

    Diffusion Drafts, AR Verifies: Accelerating Document OCR with Self-Speculative Decoding

    [https://arxiv.org/abs/2609.26638](https://arxiv.org/abs/2609.26638)

    GravityOCR提出参数共享的自回归-块-扩散模型，通过扩散并行生成草稿并经自回归验证后提交，在无需独立草稿网络的情况下加速文档OCR推理，同时支持基于OCR奖励的GRPO强化学习训练。

    

    自回归OCR视觉语言模型能够准确地将文档图像转换为文本和结构化标记，但每个输出标记都需要一个顺序解码步骤，限制了推理速度。与开放式文本生成不同，OCR输出高度依赖于输入图像，这使得基于扩散的并行生成大有可为。然而，当在一个扩散步骤中预测多个标记时，每个标记的预测都发生在其他标记已知之前，因此直接提交这些标记可能引入错误。为此，我们提出了GravityOCR，一个参数共享的自回归-块-扩散模型，联合训练用于并行草稿生成与因果自回归验证。在提交前对草稿进行验证，使模型能够在每轮提交多个输出标记，而无需单独的草稿生成网络。因果自回归路径还支持使用序列级和结构级OCR奖励的GRPO训练，在更新共享草稿生成器的同时避免了扩散轨迹的似然估计。

    arXiv:2609.26638v1 Announce Type: new  Abstract: Autoregressive OCR vision-language models accurately convert document images into text and structured markup, but require one sequential decoding step per output token, limiting inference speed. Unlike open-ended text generation, OCR outputs are strongly grounded in the input image, making diffusion-based parallel generation promising. However, when several tokens are predicted in one diffusion step, each is predicted before the others are known. Committing them directly can therefore introduce errors. We therefore introduce GravityOCR, a parameter-shared AR-block-diffusion model jointly trained for parallel drafting and causal AR verification. Verifying drafts before commitment lets the model commit multiple output tokens per round without a separate drafting network. The causal AR path also enables GRPO with sequence- and structure-level OCR rewards, avoiding diffusion-trajectory likelihood estimation while updating the shared drafter 
    
[^9]: 能力强大却精简节俭：提取并刻画前沿模型中隐藏的思维链

    Capable yet Parsimonious: Extracting and Characterizing Hidden Chain-of-Thought in Frontier Models

    [https://arxiv.org/abs/2609.26637](https://arxiv.org/abs/2609.26637)

    通过标准API注册简单自定义工具诱导前沿模型外化隐藏的思维链，验证提取的推理性能与原生CoT相当且显著优于无推理基线，并系统刻画了前沿模型中间推理的结构特征。

    

    前沿语言模型能力的快速提升被广泛归因于推理能力的改进，但由于闭源系统中的原始思维链（CoT）痕迹是隐藏的，这一说法无法得到验证。通过标准API功能注册一个简单的自定义工具，我们诱导前沿模型将中间推理过程外化。由于这些痕迹可能反映的是事后合理化而非真正的推理，我们首先在开源模型上与原生CoT进行对比评估，随后扩展到包括GPT-6 Astra在内的闭源前沿模型。我们发现，在竞赛数学、科学和代码生成任务上，提取出的推理性能与原生推理相当，并显著优于无推理基线。随后，我们刻画了前沿模型构建中间推理的方式。在token效率、推理步骤类型和诱导推理树等方面，我们识别出模型外化推理过程中的系统性差异。

    arXiv:2609.26637v1 Announce Type: new  Abstract: The rapid capability gains of frontier language models are widely attributed to improved reasoning abilities, yet this cannot be verified as raw CoT traces in closed-source systems are hidden. By registering a simple custom tool through a standard API feature, we induce frontier models to externalize intermediate reasoning. Because these traces may reflect post-hoc rationalization rather than genuine reasoning, we first evaluate against native CoT on open-source models and extend to closed-source frontier models including GPT-6 Astra. We find that the extracted reasoning matches native reasoning performance and substantially outperforms no-reasoning baselines, across competition mathematics, science, and code generation. We then characterize how frontier models structure their intermediate reasoning. Across token efficiency, reasoning-step types, and induced reasoning trees, we identify systematic differences in how models externalize, c
    
[^10]: 用于持续文档创作的知识拉取请求

    Knowledge Pull Requests for Continual Document Authoring

    [https://arxiv.org/abs/2609.26634](https://arxiv.org/abs/2609.26634)

    提出知识拉取请求（KPR）框架，通过提取论断、路由至章节并生成区分知识变化与文本变化的变更日志，实现可解释的持续文档修订，在维基百科和RAGTIME上比重写或重新生成整合更多信息且更好地保留原有内容。

    

    我们提出了知识拉取请求（KPR），这是一个使每次变更都可解释的持续文档创作框架。随着新知识从其他来源、语言或时间涌现，文档需要持续修订，但现有方法要么在编辑时不考虑哪些知识发生了变化，要么从头重新生成。KPR通过提取论断、过滤并将其路由到相应章节、标记与现有内容的冲突，将新知识整合到文档中，并生成一个变更日志，将“知识如何变化”（论断提议）与“文本如何变化”（文档差异）区分开来。我们在跨语言修订维基百科以及更新RAGTIME上的查询驱动报告两个任务上评估了KPR。与从来源重写或从头重新生成相比，KPR整合了更多信息并更好地保留了现有内容，同时每个生成token所添加的信息量最多。经KPR修订的文章还能更好地为问答提供依据。

    arXiv:2609.26634v1 Announce Type: new  Abstract: We introduce Knowledge Pull Requests (KPRs), a framework for continual document authoring that makes each change interpretable. Documents require ongoing revision as new knowledge surfaces from other sources, languages, or times, but existing approaches either edit with no account of what knowledge changed or regenerate from scratch. A KPR integrates new knowledge into a document by extracting claims, filtering and routing them to sections, and flagging conflicts with existing content, producing a ChangeLog that separates what knowledge changes (claim proposal) from how the text changes (document diff). We evaluate KPRs on revising Wikipedia across languages and updating query-driven reports on RAGTIME. KPRs integrate more information and better preserve existing content than rewriting from sources or regenerating from scratch, while adding the most information per token generated. A KPR-revised article also grounds question answering be
    
[^11]: PersonaWeaver：程序化角色生成中超越传统原型的可控多样性

    PERSONAWEAVER: Controllable Diversity Beyond Conventional Archetypes in Procedural Character Generation

    [https://arxiv.org/abs/2609.26629](https://arxiv.org/abs/2609.26629)

    PersonaWeaver通过将世界构建与行为规范解耦，并利用人工策划的多样化道德立场库与对话反应库来建模角色行为，突破了LLM生成角色时行为同质化的局限，实现了程序化角色生成中超越传统原型的可控多样性。

    

    程序化角色生成旨在为游戏、模拟器及其他虚拟世界填充多样化的角色。大语言模型（LLM）为扩展这一任务提供了有前景的基础。然而，基于LLM的程序化角色生成仍处于早期阶段：现有方法要么直接生成角色，要么对从角色库中检索到的档案进行调整。正如我们所展示的，这两种方法都会产生行为同质化的角色群体：角色绝大多数都认同积极的道德规范，并以乐于助人的、类似助手的反应来回答问题。为了缓解这种同质化，我们提出了PersonaWeaver，它将世界构建与行为规范解耦，并通过设定通用的、多样化的、人工策划的道德立场库和对话反应库来对行为进行建模。这一设计使我们能够测试LLM在跨设定的情况下，能在多大程度上超越其默认的行为模式。

    arXiv:2609.26629v1 Announce Type: new  Abstract: Procedural character generation aims to populate games, simulations, and other virtual worlds with diverse characters. Large language models (LLMs) offer a promising foundation for scaling this task. However, LLM-based procedural character generation remains at an early stage: existing methods either generate characters directly or adapt profiles retrieved from persona banks. As we show, both approaches produce behaviorally homogeneous populations: characters overwhelmingly agree with positive moral norms and respond to questions with helpful, assistant-like reactions. To mitigate this homogenization, we introduce PersonaWeaver, which disentangles world building from behavioral specification and models behavior through setting general, diverse, manually curated banks of moral positions and conversational reactions. This design allows us to test how far LLM(s) can be pushed beyond their default behavioral patterns across settings. Across 
    
[^12]: 面向自然语言推理的语义抽象：一种发现与补偿大语言模型中语义知识与推理差距的方法论框架

    Semantic Abstraction for Natural Language Inference: a Methodological Framework for Discovering and Compensating Semantic Knowledge and Reasoning Gaps in Large Language Models

    [https://arxiv.org/abs/2609.26610](https://arxiv.org/abs/2609.26610)

    本文提出了一个基于语义相容性与不相容性的方法论框架，通过在更高抽象层次上重构前提与假设之间的词汇语义关系，以发现并补偿大语言模型在自然语言推理任务中的语义知识与推理差距。

    

    尽管大语言模型（LLMs）在许多自然语言处理任务上表现出色，但它们在语义抽象方面面临着严峻的挑战。在本研究中，我们致力于理解大语言模型如何在自然语言推理（NLI）任务中利用抽象语义知识，该任务需要复杂的语言能力来解释隐含意义、上下文概念关系以及词语和短语之间的语义联系。为此，我们提出了一个方法论框架，用于在更高的抽象层次上构建新的语义知识，我们在自然语言推理的语义相容性与不相容性概念下对该框架进行定义。在该框架中，前提与假设之间的词汇-语义关系的含义被重新配置，以实现一个更加灵活的语义网络，从而在大语言模型中诱导出不同的推理路径。这些新路径展现出一致的响应模式，使得模型能够对单一响应达成一致。

    arXiv:2609.26610v1 Announce Type: new  Abstract: Despite their outstanding performance on many NLP tasks, LLMs face serious challenges related to semantic abstraction. In this study, we are interested in understanding how LLMs leverage abstract semantic knowledge in natural language inference (NLI), which requires sophisticated linguistic capabilities to interpret implicit meanings, contextual conceptual relationships, and semantic connections between words and phrases. To this end, we propose a methodological framework for constructing new semantic knowledge at a higher level of abstraction, which we define under the notions of semantic compatibility and incompatibility for NLI. In this framework, the meaning of the lexical-semantic relations between the premise and the hypothesis is reconfigured to achieve a more flexible semantic network that induces different reasoning paths in LLMs. These new pathways show a consistent pattern of responses that allows agreement on a single respons
    
[^13]: 接纳而非谄媚：区分语言模型中的互动参与与盲目顺从

    Receptiveness, Not Sycophancy: Distinguishing Engagement from Deference in Language Models

    [https://arxiv.org/abs/2609.26579](https://arxiv.org/abs/2609.26579)

    该论文指出，语言模型“社会性谄媚”评估存在构念效度问题，因为谄媚的标志与有益的“对话接纳性”高度重叠——被判定为谄媚的回复往往只是更具接纳性，而非盲目顺从。

    

    语言模型的一个核心问题是谄媚性：即它们倾向于顺从用户的观点，而牺牲独立的实质性判断。与此同时，关于社会性谄媚的研究聚焦于确认和积极肯定等行为，这些行为可能表明不当的顺从。然而，社会性谄媚的标志特征同时也是对话接纳性的特征——接纳性是社会心理学中的一个概念，已被证明能够改善分歧情境下的互动。我们认为，这种重叠为社会性谄媚评估带来了构念效度问题。使用一个流行的道德建议数据集，我们发现被归类为更具社会性谄媚的回复同时也更具接纳性。此外，在保持实质性结论不变的前提下，提高人工撰写回复的接纳性，会导致这些回复被归类为更具社会性谄媚。这种紧密耦合引发了一种可能性，即社会性谄媚评估可能无意中（摘要在此处截断）

    arXiv:2609.26579v1 Announce Type: new  Abstract: A central concern with language models is sycophancy: their tendency to defer to users' views at the expense of independent substantive judgment. In parallel, work on social sycophancy has focused on behaviors such as validation and positivity that may signal inappropriate deference. Yet the markers of social sycophancy are also characteristic of conversational receptiveness, a construct from social psychology shown to improve interactions across disagreement. We argue that this overlap creates a construct-validity problem for social sycophancy evaluations. Using a popular moral-advice dataset, we find that responses classified as more socially sycophantic are also more receptive. Further, increasing the receptiveness of human-written responses---while preserving their substantive conclusions---causes them to be classified as more socially sycophantic. This tight coupling raises the possibility that social sycophancy evaluations inadvert
    
[^14]: 对使用大语言模型研究婴儿句法学习的回顾性分析

    A retrospective analysis on the use of LLMs to study infant syntax learning

    [https://arxiv.org/abs/2609.26539](https://arxiv.org/abs/2609.26539)

    本文从认识论角度对BabyLM挑战赛等使用大语言模型研究婴儿句法学习的研究计划进行了回顾性评估，指出其方法论中存在的重大假设削弱了理论适用范围，且使用符合发育现实性的语料库对模型在常用基准测试上的表现影响有限。

    

    大语言模型（LLMs）已被越来越多地用于研究儿童在发育早期阶段如何习得句法。这尤其是BabyLM挑战赛的核心科学目标——该挑战赛是一项社区范围的共同努力，旨在开发在使用符合发育现实性的语料库进行训练的同时，达到人类水平句法表现的模型。在本文中，我们通过对该研究计划中若干研究进行认识论层面的评估，反思了LLMs在婴儿句法学习研究中的使用。我们讨论了数据集是如何构建的、实现了哪些模型、这些模型如何被训练以及如何进行句法评估。我们观察到BabyLM及相关研究的方法论中存在重大假设，从而削弱了其理论适用范围。我们还观察到，使用符合发育现实性的语料库对模型在常用基准测试上的表现影响有限，这提示了重要的计算差异。

    arXiv:2609.26539v1 Announce Type: new  Abstract: Large language models (LLMs) have increasingly been used to investigate how children acquire syntax at an early stage of development. This is notably the central scientific goal of the BabyLM challenge, a community-wide effort to develop models that achieve human-level syntactic performance while being trained on developmentally realistic corpora. In this paper, we reflect on the use of LLMs in the study of infant syntax learning by providing an epistemological assessment of several studies from this research program. We discuss how datasets are built, which models are implemented, how they are trained and syntactically evaluated. We observe significant assumptions in the methodology of BabyLM and related studies, thus mitigating their theoretical scope. We additionally observe that using developmentally-realistic corpora have limited effects on models performance on commonly-used benchmarks, which suggest important computational differe
    
[^15]: 转录、翻译与优化：面向语音翻译的联合奖励学习

    Transcribe, Translate, and Optimize: Joint Reward Learning for Speech Translation

    [https://arxiv.org/abs/2609.26536](https://arxiv.org/abs/2609.26536)

    提出通过GRPO对语音识别与翻译进行联合强化微调，以解决SFT参考转录与模型生成转录之间的不匹配问题，在CoVoST 2和FLEURS上显著提升BLEU分数并降低词错误率。

    

    在基于大语言模型的语音翻译中，基于转录的思维链方法存在监督微调阶段使用的参考转录文本与推理阶段模型生成的转录文本之间的不匹配问题。为解决这一问题，我们提出通过组相对策略优化进行识别与翻译的联合微调。我们对转录文本和翻译结果同时进行评分，其中翻译以模型生成的转录文本为条件，并比较了三种token优势策略。我们使用Qwen2.5-Omni-3B模型在四种语言上进行实验，在SFT和GRPO两种设置下评估CoT与直接语音翻译，在CoVoST 2上训练，并在CoVoST 2和FLEURS上测试。CoT GRPO在CoVoST 2和FLEURS上分别比Direct ST GRPO高出1.77和0.83个平均BLEU点。与CoT SFT相比，GRPO将BLEU分别提升0.82和0.67点，并将词错误率相对降低8.8%和7.2%。这些结果凸显了强化微调作为一种有效的……

    arXiv:2609.26536v1 Announce Type: new  Abstract: In LLM-based speech translation, transcription-based chain-of-thought (CoT) suffers from a mismatch between reference transcripts used in supervised fine-tuning (SFT) and model-generated transcripts at inference. To address this, we propose joint recognition and translation fine-tuning via group relative policy optimization (GRPO). We score both transcripts and translations, with translation conditioned on model-generated transcripts, and compare three token advantage strategies. Using Qwen2.5-Omni-3B across four languages, we evaluate CoT against direct speech translation (Direct ST) under SFT and GRPO, training on CoVoST 2 and testing on CoVoST 2 and FLEURS. CoT GRPO outperforms Direct ST GRPO by 1.77 and 0.83 average BLEU points on CoVoST 2 and FLEURS. Compared to CoT SFT, GRPO boosts BLEU by 0.82 and 0.67 points and reduces word error rate (WER) by 8.8% and 7.2% relatively. These results highlight reinforcement fine-tuning as an effe
    
[^16]: 一种符号学感知的框架：用于评估自然语言生成中的忠实度与覆盖度

    A Semiotics-Aware Framework for Evaluating Fidelity and Coverage in Natural Language Generation

    [https://arxiv.org/abs/2609.26527](https://arxiv.org/abs/2609.26527)

    该论文提出符号学忠实度与符号学覆盖度两个新评估指标，用以衡量文本间的符号学对齐程度，发现覆盖度通常低于忠实度，且大语言模型与人工整理数据在低采样温度下对齐程度最高。

    

    当两个文本描述同一个表达时，基于词汇重叠或全文相似度的标准指标可能无法检测出该表达在表述方式上的有意义差异。我们提出一个框架来评估文本之间的符号学对齐，其中符号学画像涵盖文本所凸显的语境意义和话语指称。我们的方法产生两个分数：符号学忠实度和符号学覆盖度，分别估计一个文本的画像有多少被另一个文本所支持，以及它能恢复另一个文本画像的多少。实验表明，覆盖度通常低于忠实度，并且大语言模型与人工整理数据之间的对齐在低采样温度时最高，而较高的温度会降低这种对齐。

    arXiv:2609.26527v1 Announce Type: new  Abstract: When two texts describe the same expression, standard metrics based on lexical overlap or whole-text similarity may fail to detect meaningful differences in how that expression is framed. We propose a framework to evaluate semiotic alignment between texts, where a semiotic profile encompasses both the contextual meaning and the discourse references made salient by a text. Our approach yields two scores, Semiotic Fidelity and Semiotic Coverage, estimating how much of one text's profile is supported by the other and how much of the other's profile it recovers. Experiments show that coverage is typically lower than fidelity, and that alignment between LLMs and human-curated data is highest at low sampling temperatures, while higher temperatures reduce this alignment.
    
[^17]: 校准作为大语言模型评估中的一等公民标准

    Calibration as a First-Class Criterion in LLM Evaluation

    [https://arxiv.org/abs/2609.26489](https://arxiv.org/abs/2609.26489)

    该论文主张将校准（模型置信度与实际正确性的一致性）作为LLM评估中不可或缺的一等公民标准，并指出由于标准校准指标仅需置信度分数和正确性判断这两个输入，而现有大多数基准已具备这两者，因此校准评估可以且应当被常规纳入模型评估流程。

    

    语言模型的校准——即所表达或隐含的置信度与实际正确性之间的一致性——是自然语言处理（NLP）中一个已被充分研究的子领域，测量校准的方法早已存在。问题在于“采用”：在该子领域之外，NLP研究经常引入新的模型、数据集和基准，却不检查模型的置信度分数是否有意义。我们认为，这种采用上的脱节是构建可信LLM评估的主要障碍。校准不良在两个不同层面造成问题：在部署层面，过度自信的错误会造成实际伤害；在研究流程层面，诸如LLM作为评判者（LLM-as-a-judge）、合成数据生成和主动学习等方法都依赖校准良好的置信度，却未加以验证。标准校准指标对每个样本只需要两个输入：一个置信度分数和一个正确性判断。当今使用的大多数基准已经同时提供这两者，这意味着校准结果可以被常规地报告出来（摘要在此处被截断）。

    arXiv:2609.26489v1 Announce Type: new  Abstract: Calibration of language models -- the alignment between expressed or implicit confidence and empirical correctness -- is a well-studied subfield within NLP. Methods to measure it already exist. The problem is adoption: outside this subfield, NLP research regularly introduces new models, datasets, and benchmarks without checking whether the model's confidence scores are meaningful. We argue that this adoption gap is a major obstacle to trustworthy LLM evaluation. Miscalibration causes problems in two distinct areas: at deployment, where overconfident mistakes cause real harm, and inside the research pipeline, where methods like LLM-as-a-judge, synthetic data generation, and active learning rely on calibrated confidence without verifying it. Standard calibration metrics only require two inputs per example: a confidence score and a correctness judgment. Most benchmarks in use today already provide both, meaning calibration can be reported i
    
[^18]: 会出声思考的口语语言模型

    Spoken Language Models that Think Aloud

    [https://arxiv.org/abs/2609.26488](https://arxiv.org/abs/2609.26488)

    提出了一种异步出声思考框架，让口语语言模型在推理过程中同步生成简短的进度性话语，从而消除“先思考后说话”范式带来的长时间静默，实现更自然的实时口语交互。

    

    虽然思维链推理提升了语言模型的能力，但在串行的“先思考后说话”范式下，直接将其应用于口语语言模型（SLM）可能会引入长时间的静默间隔，破坏实时的口语交互。为解决这一问题，我们在Thinker-Talker架构中提出了一种面向推理型口语语言模型的异步出声思考框架。该框架维护一个用于逻辑推理的主推理流，以及一个轻量级的出声思考流，后者根据用户输入和不断演化的推理状态生成简短的、与任务相关的进度性话语。一种动态平衡策略在运行时协调这两个流，通过触发额外的出声思考语音来避免静默间隙，并在最终响应就绪时取消待发送的话语。在口语推理和问答基准上的实验表明，我们的方法显著减少了用户可感知的静默……

    arXiv:2609.26488v1 Announce Type: new  Abstract: While Chain-of-Thought (CoT) reasoning has improved the capability of language models, directly applying it to Spoken Language Models (SLMs) may introduce long silent intervals under the serial "think-then-speak" paradigm, disrupting real-time spoken interaction. To address this issue, we propose an asynchronous think-aloud framework for reasoning-based SLMs within the Thinker-Talker architecture. The framework maintains a primary reasoning stream for logical deduction and a lightweight think-aloud stream that generates short, task-grounded progress utterances conditioned on the user input and the evolving reasoning state. A dynamic balance strategy coordinates the two streams at runtime, triggering additional think-aloud speech to avoid silent gaps and canceling pending utterances when the final response becomes ready. Experiments on spoken reasoning and question-answering benchmarks show that our approach substantially reduces user-aud
    
[^19]: 行为并不足够：基于机制的LLM社会中社会规范涌现评估

    Behavior is Not Enough: A Mechanism-Based Evaluation of Social Norm Emergence in LLM Societies

    [https://arxiv.org/abs/2609.26481](https://arxiv.org/abs/2609.26481)

    该论文提出一个超越行为观察的评估框架，通过测量LLM智能体的经验性与规范性预期，并结合社会学习、社会选择两种机制以及对抗性干扰下的稳定性测试，来更严谨地评估LLM社会中的规范涌现。

    

    社会规范无法仅凭行为来识别：同样的合作均衡可能源于共同预期、策略性激励或简单的模仿。然而在多智能体大语言模型系统中，先前的工作大多将行为趋同视为规范涌现的证据。在这项工作中，我们引入了一个评估框架，在行为趋同之外，还测量智能体所报告的经验性预期和规范性预期。通过受控消融实验，我们测试了预期引导的效果，并分离出规范形成理论中的两个核心集体机制——通过互动实现的社会学习和通过网络化群体形成实现的社会选择。我们进一步在四个LLM家族中测试了这些动态在对抗性干扰下的稳定性。我们发现，引导预期会增加合作贡献，社会学习能够稳定行为，而社会选择……

    arXiv:2609.26481v1 Announce Type: cross  Abstract: Social norms cannot be identified from behavior alone: the same cooperative equilibrium may reflect shared expectations, strategic incentives, or simple imitation. Yet in multi-agent large language model systems, prior work largely treats behavioral convergence as evidence of norm emergence. In this work, we introduce an evaluation framework that measures agents' reported empirical and normative expectations in addition to behavioral convergence. Through controlled ablations, we test the effect of expectation elicitation and isolate two collective mechanisms central to theories of norm formation---social learning through interaction and social selection through network-based group formation. We further test the stability of these resulting dynamics under adversarial disruption across four LLM families. We find that eliciting expectations increases cooperative contributions, while social learning stabilizes behavior, and social selectio
    
[^20]: 如何估计你是否在干草堆中找到了几根针：多标签文本分类中的校准度量

    How to Estimate Whether You Have Found Several Needles in a Haystack: Measuring Calibration in Multi-Label Text Classification

    [https://arxiv.org/abs/2609.26468](https://arxiv.org/abs/2609.26468)

    提出了一种对正负标签分配赋予同等权重的新分箱方案，解决了多标签文本分类中标签级期望校准误差计算不准确的问题。

    

    决定是否信任自动预测的一个关键因素是其置信度分数，该分数应经过校准以匹配预测正确的实际概率。大多数置信度校准指标针对二分类或多分类任务，而多标签校准在很大程度上仍未被充分探索。多标签分类任务，例如为临床笔记分配医学代码或确定新闻主题，通常由大量负样本主导，即不适用的标签。我们表明，现有的用于计算标签级期望校准误差的分箱方案要么低估误差，要么仅仅反映标签频率，要么存在许多实例极少的分箱。为了获得可信的标签级校准误差，我们提出了一种新的分箱方案，该方案对正负标签分配给予同等权重。我们的实证研究表明，与现有的分箱方案相比，我们的方法……

    arXiv:2609.26468v1 Announce Type: new  Abstract: A key factor in deciding whether to trust an automatic prediction is its confidence score, which should be calibrated to match the actual probability of the prediction being correct. Most confidence calibration metrics target binary or multi-class tasks, while multi-label calibration remains largely underexplored. Multi-label classification tasks, such as assigning medical codes to clinical notes or determining news topics, are usually dominated by a large number of negatives, i.e., labels that do not apply. We show that existing binning schemes to compute label-wise expected calibration error either underestimate the error, simply reflect label frequency, or suffer from many bins with very few instances. To achieve trustworthy label-wise calibration errors, we propose a new binning scheme that gives equal weight to positive and negative label assignments. Our empirical study demonstrates that in contrast to existing binning schemes, our
    
[^21]: 利用对话上下文丰富语音情感表示

    Enriching Speech Emotion Representations with Conversational Context

    [https://arxiv.org/abs/2609.26422](https://arxiv.org/abs/2609.26422)

    提出ACERT模块，通过整合灵活长度的对话上下文窗口来丰富语音情感表示，从而更好地捕捉语音交互中的情感演变，在IEMOCAP数据集上超越现有最先进方法。

    

    检测情感对于构建能够准确且自适应地与人类交互的系统至关重要。语音情感识别（SER）已成为开发智能语音接口的重要研究方向。然而，大多数研究在话语层面预测情感，忽略了对话上下文及其所承载的情感流动和说话人互动。在本文中，我们提出了ACERT（Averaged Contextual Emotion Representation through Time，随时间平均的上下文情感表示），这是一个整合了灵活长度对话上下文窗口的模块，能够更好地捕捉语音交互中的情感演变。为了评估该方法的鲁棒性，我们在涵盖多种情感表达风格和情境的数据集上进行了实验。ACERT在IEMOCAP数据集上超越了当前最先进（SOTA）的方法，在SAFE数据集上建立了首个上下文感知基准，并在MELD数据集的无加权、类别平衡指标上取得了优异结果。

    arXiv:2609.26422v1 Announce Type: new  Abstract: Detecting emotions is necessary for building systems that can accurately and adaptively interact with humans. Speech Emotion Recognition (SER) has become an important research focus to develop intelligent spoken interfaces. However, most studies predict emotions at the utterance level, ignoring the conversational context, along with the emotional flow and speaker interactions it carries. In this paper, we introduce ACERT (Averaged Contextual Emotion Representation through Time), a module that integrates a flexible-length window of conversational context to better capture emotional evolution in spoken interactions. To evaluate the robustness of this method, we conducted experiments on datasets spanning diverse emotionally expressive styles and contexts. ACERT outperforms current state-of-the-art (SOTA) approaches on IEMOCAP, establishes the first context-aware benchmark on SAFE, and obtains strong results on MELD for unweighted, class-bal
    
[^22]: 结合分层认知过程与过程监督的可解释场景安全理解

    Combining Hierarchical Cognitive Process with Process Supervision for Interpretable Scene Safety Understanding

    [https://arxiv.org/abs/2609.26399](https://arxiv.org/abs/2609.26399)

    该论文提出将分层认知过程建模与过程监督相结合，构建了带过程标签的多步推理场景安全理解数据集，以提升大语言模型可解释的场景安全理解能力。

    

    场景安全理解在各种关键领域的态势感知中起着生死攸关的作用。传统方法依赖于学习场景与安全等级之间的直接映射，往往缺乏可解释性，限制了其在关键应用中的可靠性。克服这一挑战的有效途径在于解读人类认知过程，并赋予机器模型类似的认知能力。本工作探索了一种将场景安全认知过程建模与过程监督相结合的有效方法。具体而言，我们首先构建了一个分层的认知安全结构，这促使我们开发了一个新颖、高质量的基于多步推理并带有过程标签的场景安全理解数据集。该数据集既可作为基准，也可作为提升大语言模型（LLMs）安全推理能力的资源，同时还支持细粒度的分析……

    arXiv:2609.26399v1 Announce Type: new  Abstract: Scene safety understanding plays a life-or-death role in situational awareness in various critical domains. Traditional methods that rely on learning direct mappings between scenes and safety levels often lack interpretability, limiting their reliability in critical applications. An effective approach to overcoming this challenge lies in interpreting human cognitive processes and equipping machine models with analogous cognitive capabilities. This work explores an effective way of integrating scene safety cognitive process modeling and process supervision. Specifically, we first construct a hierarchical cognitive safety structure, which motivates the development of a novel, high-quality scene safety understanding dataset based on multi-step reasoning with process labels. This dataset serves both as a benchmark and a resource to improve the safety reasoning capabilities of Large Language Models (LLMs), while also enabling a granular analy
    
[^23]: 论大语言模型在代码理解中的词汇迷信：基于低词汇质量代码的重新评估

    On the Lexical Superstition of Large Language Models for Code Comprehension: Re-evaluation on Code of Low Lexical Quality

    [https://arxiv.org/abs/2609.26388](https://arxiv.org/abs/2609.26388)

    论文提出语义保持的标识符重命名框架Face/Off，揭示大语言模型在代码理解中普遍过度依赖标识符的词汇线索，且这一根深蒂固的问题难以通过现有的提示和微调干预加以解决。

    

    大语言模型（LLM）的最新进展使其被广泛应用于代码相关任务。标识符名称在自然产生的代码中具有统计信息价值，但其信息并不总是可靠的。我们研究了当前大语言模型在重命名保持程序结构不变的情况下，是否会对词汇线索赋予不成比例的权重。我们提出了Face/Off——一个保持语义的标识符重命名框架，并在多个模型和代码理解任务上评估渐进式命名条件。在该框架下，词汇过度依赖在被评估的模型和主要任务中普遍存在：随着标识符信息被移除或变得具有误导性，性能通常会下降，且输出往往被导向误导性名称所暗示的含义。该模式在代表性的基于提示和基于微调的干预措施下仍然存在，表明词汇过度依赖是一个根深蒂固的问题。

    arXiv:2609.26388v1 Announce Type: cross  Abstract: Recent advances in large language models (LLMs) have made them widely used for code-related tasks. Identifier names are statistically informative in naturally occurring code, but their information is not always reliable. We investigate whether current LLMs assign disproportionate weight to lexical cues when renaming preserves program structure. We introduce Face/Off, a semantics-preserving identifier-renaming framework, and evaluate progressive naming conditions across multiple models and code-comprehension tasks. Within this framework, lexical overemphasis is pervasive across the evaluated models and primary tasks: performance generally decreases as identifier information is removed or made misleading, and outputs are often directed toward the meanings suggested by misleading names. The pattern persists under representative prompt- and fine-tuning-based interventions, suggesting that lexical overemphasis is an entrenched problem. A ty
    
[^24]: 面向GROBID的版面引导掩码：大规模科学PDF摄取中的轻量级结构化增益

    Layout-Guided Masking for GROBID: Lightweight Structural Gains in Large-Scale Scientific PDF Ingestion

    [https://arxiv.org/abs/2609.26381](https://arxiv.org/abs/2609.26381)

    通过为GROBID配备轻量级CPU版面检测器并使用类型化区域掩码路由标记，无需GPU即可在大规模科学PDF结构化解析中显著提升精度。

    

    将学术PDF转换为机器可读的全文仍然是大规模信息系统的瓶颈。近年来基于视觉的解析器提高了准确性，但需要GPU，并且可能在提取的文本中引入噪声。GROBID是一个在CPU上运行的模块化字体流解析器，是构建科学文章结构的事实标准，支撑着多个最大的开放学术语料库。我们将其与一个轻量级CPU检测器配对，该检测器用于定位图、表和辅助文本（页眉、页脚、页码）区域，并将其编码为类型化区域掩码，其标记被路由到GROBID的专用模型或被丢弃。在两个PMC语料库上——生物信息学（1,926篇文章）和材料科学（2,595篇）——采用基于章节感知的结构化协议对照JATS进行评分，我们的扩展在大多数指标上优于原始GROBID（NS +0.025/+0.013；材料科学上段落召回率+0.086，d_z=1.08），并且与图注关联的图恢复能力得到提升。

    arXiv:2609.26381v1 Announce Type: new  Abstract: Transforming scholarly PDFs into machine-readable fulltext remains a bottleneck for large-scale information systems. Recent vision-based parsers improve accuracy, but need GPUs and may introduce noise into the extracted text. GROBID, a modular font-stream parser running on CPU, is the de-facto standard for structuring scientific articles and underpins several of the largest open scholarly corpora. We pair it with a lightweight CPU detector localising figure, table, and paratext (header, footer, page number) regions, encoded as typed-area masks whose tokens are routed to GROBID's specialised models or discarded. On two PMC corpora, Bioinformatics (1,926 articles) and Materials Science (2,595), scored against JATS with a section-aware structural protocol, our extension improves over plain GROBID on most metrics (NS $+0.025$/$+0.013$; $+0.086$ paragraph recall on Materials Science, $d_z{=}1.08$), and caption-linked figure recovery improves 
    
[^25]: HySparse2：具有两级KV共享的混合稀疏注意力

    HySparse2: Hybrid Sparse Attention with Two-Level KV Sharing

    [https://arxiv.org/abs/2609.26368](https://arxiv.org/abs/2609.26368)

    HySparse2提出了一种具有两级KV共享的混合稀疏注意力架构：外层采用YOCO式KV桥接仅桥接全注意力层，内层在KV复用基础上将块级稀疏升级为词元级稀疏，从而实现高效预填充、紧凑KV缓存存储和精确的长上下文检索。

    

    长周期与多轮智能体通常生成较短的动作，同时需要处理来自工具和环境的长时间观察信息。这种不断增长的上下文对高效预填充、紧凑的KV缓存存储以及精确的长上下文检索提出了要求。为满足这些需求，我们提出了HySparse2，一种具有两级KV共享的混合稀疏注意力架构。在外层，KV桥接采用YOCO风格的自解码器与交叉解码器结构，但仅对全注意力层进行桥接。自解码器使用混合滑动窗口注意力（SWA），而交叉解码器使用混合稀疏注意力。交叉解码器中全注意力层的KV缓存由自解码器中全注意力层的隐藏状态生成。在内层，HySparse2保留了HySparse的核心KV复用设计，并进行了两项改进：首先，将块级稀疏性替换为词元级稀疏性，以实现更精细的长上下文检索；其次，移除……（原文摘要在此处截断）

    arXiv:2609.26368v1 Announce Type: new  Abstract: Long-horizon and multi-turn agents typically generate short actions and process long observations from tools and environments. This growing context demands efficient prefill, compact KV-cache storage, and accurate long-context retrieval. To meet these demands, we introduce HySparse2, a hybrid sparse attention architecture with two-level KV sharing. At the outer level, KV Bridging adopts a YOCO-style self-decoder and cross-decoder structure, but bridges only full-attention layers. The self-decoder uses hybrid sliding-window attention (SWA), while the cross-decoder uses hybrid sparse attention. The KV caches for full-attention layers in the cross-decoder are generated from the hidden states of full-attention layers in the self-decoder. At the inner level, HySparse2 retains HySparse's core KV Reuse design with two refinements. First, it replaces block-level sparsity with token-level sparsity for finer long-context retrieval. Second, it remo
    
[^26]: TransBERT：面向特定领域语言建模的合成翻译框架

    TransBERT: A Framework for Synthetic Translation in Domain-Specific Language Modeling

    [https://arxiv.org/abs/2609.26347](https://arxiv.org/abs/2609.26347)

    提出仅使用合成翻译文本预训练语言模型的TransBERT框架，并证明仅凭合成翻译数据即可在法语生命科学领域的各类下游任务上达到最先进性能。

    

    专业领域中非英语语言数据的稀缺严重限制了有效自然语言处理（NLP）工具的发展。我们提出了TransBERT，一个仅使用合成翻译文本进行语言模型预训练的新型框架，并介绍了可扩展的翻译工具包TransCorpus。聚焦于法语生命科学领域，我们的方法表明，仅利用合成翻译数据即可在各种下游任务上达到最先进的性能。我们发布了TransCorpus工具包、TransCorpus-bio-fr语料库（36.4GB的法语生命科学文本）、TransBERT-bio-fr及其相关的预训练语言模型，以及用于预训练和微调的可复现代码。我们的结果突显了在高资源翻译方向上利用合成翻译来构建低资源语言/领域对高质量NLP资源的可行性。

    arXiv:2609.26347v1 Announce Type: new  Abstract: The scarcity of non-English language data in specialized domains significantly limits the development of effective Natural Language Processing (NLP) tools. We present TransBERT, a novel framework for pre-training language models using exclusively synthetically translated text, and introduce TransCorpus, a scalable translation toolkit. Focusing on the life sciences domain in French, our approach demonstrates that state-of-the-art performance on various downstream tasks can be achieved solely by leveraging synthetically translated data. We release the TransCorpus toolkit, the TransCorpus-bio-fr corpus (36.4GB of French life sciences text), TransBERT-bio-fr, its associated pre-trained language model and reproducible code for both pre-training and fine-tuning. Our results highlight the viability of synthetic translation in a high-resource translation direction for building high-quality NLP resources in low-resource language/domain pairs.
    
[^27]: 跨党派指责：丹麦议会中的政治对比与指责归因

    Blaming Across the Aisle: Political Contrasting and Blame Attribution in the Danish Parliament

    [https://arxiv.org/abs/2609.26346](https://arxiv.org/abs/2609.26346)

    本研究开发了面向中低资源语言的高效标注指责归因分类器BlameBERT，分析丹麦议会1997-2026年的辩论数据，发现指责行为呈“香蕉形”轨迹且近年持续上升，反对党显著多于执政党进行指责（“政治对比”效应），且该效应受意识形态调节、在右翼中更弱。

    

    政治话语被普遍认为正变得愈发充满敌意，但可靠的证据仍然稀缺。本研究考察了1997年至2026年间丹麦议会中的指责归因现象，结合了一个专门构建的分类器BlameBERT（F1分数：0.80）与多层统计建模。该分类器采用一种面向低至中等资源语言的高效标注流水线构建而成。研究结果揭示了一条“香蕉形”轨迹：指责在2016年左右之前呈下降趋势，随后在近年来（2019-2026年）进入显著且持续的上升。执政地位始终影响着指责归因——我们将这一效应称为“政治对比”——反对党明显比执政党发表更多指责。这一效应受意识形态调节：执政对指责的抑制作用在右翼政党中较不明显，且意识形态极端性在右翼阵营中更强地放大了指责行为。

    arXiv:2609.26346v1 Announce Type: new  Abstract: Political discourse is widely perceived to be growing more hostile, yet robust evidence remains scarce. This study examines blame attribution in the Danish Parliament from 1997 to 2026, combining a purpose-built classifier, BlameBERT (F1: 0.80), with multilevel statistical modeling. The classifier is constructed using an annotation-efficient pipeline for blame attribution in low-to-mid resource languages. The results reveal a banana-shaped trajectory, with blame declining until around 2016 before entering a significant and sustained increase in recent years (2019-2026). Government status consistently influenced blame attribution - an effect we term political contrasting - with opposition parties blaming substantially more than governing parties. This effect was moderated by ideology: The blame-dampening effect of governing was less pronounced among right-wing parties, and ideological extremity amplified blame more strongly on the right. 
    
[^28]: 设计与分析论辩挖掘流水线：迈向全面评估

    Designing and Analysing Argument Mining Pipelines: Towards a Comprehensive Assessment

    [https://arxiv.org/abs/2609.26338](https://arxiv.org/abs/2609.26338)

    本文提出一个基于语言学、计算和领域三重视角的元研究框架，系统分析最先进的端到端论辩挖掘流水线，从而实现不同方法之间更清晰的任务级比较与评估。

    

    论辩挖掘将自然语言转换为其底层的论辩结构。这种转换通常通过一系列论辩挖掘任务来实现，这些任务构成一个端到端的论辩挖掘流水线。然而，不同的论辩挖掘方法在如何概念化这些任务上往往存在差异，这使得它们之间的直接比较变得困难且不透明。因此，需要对论辩挖掘方法进行更细致的任务级分析，以实现更清晰的比较和评估。本工作提出了一项初步的元研究，系统地回顾了若干最先进的端到端论辩挖掘工作，并通过一个三重视角框架——语言学视角、计算视角和领域视角——分析它们的流水线，以理解这些流水线如何将论辩建模为结构、如何对其进行计算，以及如何整合领域知识。我们进一步为语言学和计算视角提出了一个通用设计，说明了关键的论辩挖掘任务是如何为建模和计算而设计的。

    arXiv:2609.26338v1 Announce Type: new  Abstract: Argument Mining (AM) transforms natural language into its underlying argument structures. This transformation is typically realized through a sequence of AM tasks that form an end-to-end AM pipeline. However, AM approaches often differ in how they conceptualize these tasks, making direct comparisons between them difficult and opaque. This calls for a more nuanced, task-level analysis of AM approaches to enable clearer comparison and assessment.   This work presents a preliminary meta-study that systematically reviews several state-of-the-art end-to-end AM works and analyzes their pipelines through a triple-perspective framework---a linguistic, computational and domain perspective---to understand how the pipelines model arguments as structures, computes them, and integrates domain knowledge. We further propose a general design to the linguistic and computational perspectives, illustrating how key AM tasks are designed for modeling and com
    
[^29]: CHiME-9 ECHI：面向听力障碍的对话增强机器学习挑战赛

    CHiME-9 ECHI: A Machine Learning Challenge for Enhancing Conversations to Address Hearing Impairment

    [https://arxiv.org/abs/2609.26306](https://arxiv.org/abs/2609.26306)

    本文介绍了CHiME-9 ECHI挑战赛的任务与结果，该挑战赛旨在从嘈杂多通道录音中提取对话语音以改善听力障碍者的语音可懂度与质量，并发现顶级系统能显著提升主观听感表现，而客观指标无法反映听众的真实感受。

    

    本工作介绍了CHiME-9挑战赛“面向听力障碍的对话增强”任务及其结果。该挑战赛考虑了在嘈杂的、类似自助餐厅的环境中进行四方对话的场景，其中包含干扰性语音源和音效。参赛者获得了使用Meta Aria智能眼镜和助听器麦克风录制的音频，以及对话参与者的干净语音样本。任务是从嘈杂的多通道录音中提取对话伙伴的语音，目标是提高语音的可懂度和质量，并通过客观指标和主观听音测试进行评估。本文回顾了七个团队的提交方案，并根据主观可懂度和质量的综合表现进行排名。结果表明，尽管客观指标无法反映听众的真实表现，但排名靠前的系统仍能带来显著的改进。

    arXiv:2609.26306v1 Announce Type: new  Abstract: This work presents the task and results of the CHiME-9 challenge for Enhancing Conversations to address Hearing Impairment. The challenge considers the scenario of four-party conversations in a noisy, cafeteria-style environment with interfering speech sources and sound effects. Participants are provided with audio recordings made with Meta Aria glasses and hearing aid microphones, and clean speech samples of the conversation participants. The task is to extract the speech of the conversation partners from the noisy multi-channel recordings with the goal of improving the intelligibility and quality of the speech, evaluated using objective metrics and subjective listening tests. This paper reviews submissions from seven teams and ranks them on a combination of subjective intelligibility and quality. Results show that while the objective metrics do not reflect listener performance, the top systems were able to make substantial improvements
    
[^30]: PACE-dLLM：基于置信度悬崖估计的扩散语言模型弹性块解码

    PACE-dLLM: Elastic Block Decoding via Confidence Cliff Estimation for Diffusion Language Models

    [https://arxiv.org/abs/2609.26249](https://arxiv.org/abs/2609.26249)

    PACE-dLLM通过在每步以闭式形式拟合模型自身置信度的“悬崖”曲线并根据其饱和点动态确定前瞻视野，实现了扩散语言模型的弹性块解码加速。

    

    扩散语言模型（dLLM），如LLaDA和Dream，在生成质量上已可与自回归（AR）大语言模型相媲美，同时支持原生并行解码。一种标准的加速策略是分块解码，即每次前向传播预测一个长度为B的块，并提交其中高置信度的token。然而，B同时耦合了两个不同的决策：前瞻视野和需提交的token数量。现有加速器通过间接启发式方法来解决这一局限，例如波动性跟踪、分隔符检测和学习评分。与之相反，我们证明所需的信息其实已经编码在模型自身的每步置信度中：窗口内置信度通常呈现一个依赖于上下文的“悬崖”形态，其饱和点可直接确定合适的前瞻视野。我们提出PACE-dLLM，它在每一步以闭式形式拟合这一参数化悬崖，并根据其饱和点设置下一个前瞻视野，从而实现弹性块解码。

    arXiv:2609.26249v1 Announce Type: new  Abstract: Diffusion language models (dLLMs), such as LLaDA and Dream, have become competitive with autoregressive (AR) LLMs in generation quality while supporting native parallel decoding. A standard acceleration strategy is block-wise decoding, where each forward pass predicts a block of length B and commits high-confidence tokens. However, B couples two distinct decisions: the look-ahead horizon and the number of tokens to commit. Existing accelerators address this limitation through indirect heuristics, such as volatility tracking, delimiter detection, and learned scoring. In contrast, we show that the required information is already encoded in the model's own per-step confidence: in-window confidence typically follows a context-dependent cliff, whose saturation point directly identifies the appropriate look-ahead horizon. We propose PACE-dLLM, which fits this parametric cliff in closed form at each step, sets the next horizon by its saturation
    
[^31]: 损伤预示恢复：压缩金融大语言模型时校准数据何时重要

    Damage Predicts Recovery: When Calibration Data Matters in Compressing Financial LLMs

    [https://arxiv.org/abs/2609.26241](https://arxiv.org/abs/2609.26241)

    该论文提出“损伤决定恢复”假设：压缩金融大语言模型时，校准数据是否重要取决于压缩造成的任务级损伤程度——量化几乎无损时校准语料的选择无关紧要，而剪枝造成大幅性能损失时，任务格式的金融领域校准数据能够恢复部分损失。

    

    后训练量化和剪枝依赖于小规模的校准语料库。金融等专业领域是否需要与领域匹配的校准数据，目前仍无定论。我们认为，答案取决于压缩造成的任务级损伤，而非领域不匹配。如果压缩保留了目标能力，更换校准语料库几乎不会产生影响；如果压缩造成了较大损失，采用任务格式的校准数据则可以恢复部分损失。我们在两个模型家族、六种压缩配置、三个token数量匹配的校准语料库以及十个金融分类和数值问答任务上检验了这一假设，结果支持该假设。量化在很大程度上保留了任务性能，在这种情况下校准数据的选择几乎没有影响；而剪枝使数值问答准确率下降超过40个百分点。在这些受损的情形下，使用另一个通用语料库并无帮助，而金融（领域校准语料）……（原文摘要在此处截断）

    arXiv:2609.26241v1 Announce Type: new  Abstract: Post-training quantization and pruning rely on a small calibration corpus. Whether specialized domains such as finance require domain-matched calibration data remains unsettled. We argue that the answer depends on the task-level damage caused by compression rather than on domain mismatch. If compression preserves the target capability, changing the calibration corpus has little effect. If compression causes large losses, task-formatted calibration can recover part of the loss. We test this hypothesis across two model families, six compression configurations, three token-matched calibration corpora, and ten financial classification and numerical question-answering tasks. The results support this hypothesis. Quantization largely preserves task performance, and calibration choice has little effect in this case. Pruning reduces numerical QA accuracy by over 40 points. In these damaged settings, another generic corpus does not help, while Fin
    
[^32]: ABAI参加COLIEE 2026任务1：基于GraphRAG增强元学习的多阶段检索，以及交叉验证与测试差距的事后分析研究

    ABAI at COLIEE 2026 Task 1: Multi-Stage Retrieval with GraphRAG-Enhanced Meta-Learning, and a Post-Hoc Study of the Cross-Validation-to-Test Gap

    [https://arxiv.org/abs/2609.26237](https://arxiv.org/abs/2609.26237)

    本文提出了一种用于COLIEE 2026判例法检索任务的四阶段多阶段检索流程（多视角BM25、神经重排序、图特征与LightGBM元学习），并通过控制实验系统性地分析了官方测试集F1（0.177）远低于交叉验证结果（0.311）的原因。

    

    我们展示了ABAI参加COLIEE 2026任务1（判例法检索）的提交方案，并对该方案表现不佳的原因进行了控制性研究。该任务隐去了被引用的段落本身，这消除了检索器通常所依赖的大部分词汇重叠信息。为应对这一挑战，我们的流程采用四个独立训练的阶段：基于引文上下文窗口的多视角BM25结合倒数排名融合、神经重排序、基于实体社区和图注意力网络的图特征，以及基于34个特征的LightGBM元学习器。我们的最佳运行在官方测试集上达到F1=0.177，而交叉验证结果为0.311，我们将这一差距归因于召回率上限、时间分布偏移以及阈值校准失误。随后，我们对这三个假设逐一进行了检验。在无数据泄漏的协议下，阈值迁移仅造成0.007的F1损失，决策质量在各时间分位数上保持平稳，且官方测试查询与训练数据的距离并无显著差异……

    arXiv:2609.26237v1 Announce Type: cross  Abstract: We present the ABAI submission to COLIEE 2026 Task 1, case law retrieval, together with a controlled study of why it underperformed. The task suppresses the cited passages themselves, which removes much of the lexical overlap a retriever would rely on. Our pipeline answers this with four independently trained stages: multi-view BM25 over citation-context windows with reciprocal rank fusion, neural reranking, graph-based features from entity communities and a graph attention network, and a LightGBM meta-learner over 34 features. Our best run reached F1=0.177 on the official test set, against a cross-validated 0.311, and we attributed that gap to a recall ceiling, temporal distribution shift, and threshold miscalibration. We then tested all three. Under leakage-free protocols threshold transfer costs 0.007 F1, decision quality is flat across chronological quartiles, and the official test queries are not measurably farther from the traini
    
[^33]: 学术出版网络的语义化方法：基于OpenAlex数据的文档向量表示与结构-语义混合融合

    A Semantic Approach to the Academic Publishing Network: Document Vector Representations and Hybrid Structural-Semantic Fusion over OpenAlex Data

    [https://arxiv.org/abs/2609.26218](https://arxiv.org/abs/2609.26218)

    该论文在学术出版网络结构图分析的基础上引入语义层，利用SPECTER2引用感知文档嵌入和权重可调的结构-语义后期融合函数，实现了比TF-IDF基线更契合专家主题分类的文档表示，并提升了推荐效果。

    

    对学术出版网络的结构化图分析能够捕捉实体之间的拓扑关系，但无法感知作品的内容。在我们已有结构化方法的基础上，本工作通过引入一个语义层和参数化的结构-语义融合对其加以补充。我们采用基于引用信息的向量嵌入（SPECTER2）来表示科学文献，并将其存储在以稳定的OpenAlex ID为键的嵌入式向量数据库中，从而使其能够直接与图层相连接。我们定义了一个模块化的后期融合函数，将语义相似度（嵌入向量的余弦相似度）与结构相似度（书目耦合）结合起来，并引入一个可调权重alpha，其取值根据具体任务而定。在俄斯特拉发技术大学（VSB - Technical University of Ostrava）的语料库上，我们展示了两个结果：基于引用信息的嵌入与OpenAlex专家主题分类体系的一致性优于TF-IDF基线，并且在推荐应用场景中……（摘要原文在此处截断）

    arXiv:2609.26218v1 Announce Type: cross  Abstract: Structural graph analysis of the academic publishing network captures the topological relationships between entities but does not see the content of works. Building on our structural approach, this work complements it with a semantic layer and a parameterized structural-semantic fusion. We represent scientific documents by citation-informed vector embeddings (SPECTER2) and store them in an embedded vector database keyed by the stable OpenAlex ID, so that they connect directly to the graph layer. We define a modular late-fusion function that combines semantic similarity (cosine of embeddings) and structural similarity (bibliographic coupling) with a tunable weight alpha whose value is chosen according to the specific task. On the corpus of VSB - Technical University of Ostrava we show two things: citation-informed embeddings agree with the expert OpenAlex topical taxonomy better than a TF-IDF baseline, and in a recommendation use case t
    
[^34]: 同一图表，不同故事：视觉-语言模型图表解读中的偏见

    Same Chart, Different Story: Bias in Vision-Language Chart Interpretation

    [https://arxiv.org/abs/2609.26210](https://arxiv.org/abs/2609.26210)

    提出了首个用于审计视觉-语言模型图表解读偏见的基准ChartBias，通过820个真实图表、六个社会属性以及12个模型共155,484条响应，揭示了当图表中仅替换社会群体时模型叙述会发生系统性偏移的偏见问题。

    

    视觉-语言模型日益被用于解读图表，并为具有社会影响的数据生成自然语言解释。然而，当同一图表中仅所引用的社会群体发生变化时，这些模型可能会产生不同的叙述，从而强化刻板印象并误导决策。尽管存在这些风险，目前尚无系统性评估跨社会维度的图表解读偏见的基准。我们提出了ChartBias，这是首个用于审计基于视觉-语言模型的图表解读偏见的基准。ChartBias包含820个手动整理的真实世界图表，涵盖六个属性：种族、收入、年龄、宗教、移民身份和性别，产生了4,319个有效的图表-属性实例和8,638个配对生成结果，其中图表保持固定，仅替换群体术语。在12个专有和开源视觉-语言模型上，共计155,484条模型响应中，我们发现了三种普遍的失败模式：叙述偏移（

    arXiv:2609.26210v1 Announce Type: new  Abstract: Vision-language models (VLMs) are increasingly used to interpret charts and generate natural-language explanations for socially consequential data. However, they may produce different narratives for the same chart when only the referenced social group changes, reinforcing stereotypes and misleading decisions. Despite these risks, no benchmark exists for systematically evaluating bias in chart interpretation across social dimensions. We introduce ChartBias, the first benchmark for auditing bias in VLM-based chart interpretation. ChartBias contains 820 manually curated real-world charts spanning six attributes: race, income, age, religion, immigration status, and gender, yielding 4,319 valid chart, attribute instances and 8,638 paired generations where the chart is fixed and only the group term is swapped. Across 12 proprietary and open-source VLMs, totaling 155,484 model responses, we find three widespread failure modes: narrative shift (
    
[^35]: 超越静态图表：语言与视觉语言模型能否生成交互式数据可视化界面？

    Beyond Static Charts: Can Language and Vision Language Models Generate Interactive Data Visualization Interfaces?

    [https://arxiv.org/abs/2609.26208](https://arxiv.org/abs/2609.26208)

    该论文提出了VIS-GEN基准，包含3,042个覆盖数据过滤、时间分析和可视化编辑等多样化分析意图的样本，用于评估大语言模型和视觉语言模型根据自然语言查询生成交互式数据可视化界面的能力，并对14个最先进的开源与闭源模型进行了系统评测。

    

    数据可视化是分析推理的核心，但现实世界的分析越来越需要语言驱动的交互式界面，而非静态图表。尽管近期的大型语言模型和视觉语言模型（LLMs/VLMs）在根据自然语言生成静态图表方面已展现出潜力，但由于缺乏基准测试，它们生成交互式数据可视化界面的能力在很大程度上仍未被探索。我们提出了VIS-GEN，一个用于评估LLMs/VLMs根据自然语言查询生成交互式可视化界面能力的基准。VIS-GEN包含3,042个样本，涵盖多样化的分析意图，包括数据过滤、时间分析和可视化编辑，每个样本均配有数据集元数据和自然语言查询，这些查询旨在反映真实且目标驱动的数据探索场景。我们对14个最先进的开源和闭源LLMs/VLMs进行了基准测试，揭示了……（原文摘要在此处截断）

    arXiv:2609.26208v1 Announce Type: new  Abstract: Data visualization is central to analytical reasoning, but real-world analysis increasingly requires language-driven interactive interfaces rather than static charts. Although recent large language and vision language models (LLMs/VLMs) have shown promise in generating static charts from natural language, their ability to generate interactive data visualization interfaces remains largely unexplored due to the lack of benchmarks. We introduce VIS-GEN, a benchmark for evaluating how well LLMs/VLMs can generate interactive visualization interfaces from natural language queries. VIS-GEN comprises 3,042 samples covering diverse analytical intents, including data filtering, temporal analysis, and visualization editing, each paired with dataset metadata and natural language queries that are designed to reflect realistic, goal driven data exploration scenarios. We benchmark 14 state-of-the-art open-source and closed-source LLMs/VLMs, revealing l
    
[^36]: WatchPoint：面向真实世界智能体Web开发的可执行用户反馈

    WatchPoint: Executable User Feedback for Real-World Agentic Web Development

    [https://arxiv.org/abs/2609.26204](https://arxiv.org/abs/2609.26204)

    WatchPoint是一个模拟用户系统，通过像真实开发者一样对运行中的Web应用生成并执行诊断脚本，产生结构化观察结果来指导编码智能体的重试，在包含1,000个顺序依赖任务的Web-Bench基准上恢复了57.6%的失败任务。

    

    当专业Web开发者的代码未通过测试时，他们不会仅仅重新阅读堆栈跟踪。他们会在浏览器中打开应用程序，点击按钮、检查计算样式并运行诊断命令，以了解问题出在哪里。现有的编码智能体反馈机制依赖于截图、LLM-as-a-judge评分或自然语言纠正，但很少有像开发者那样与实际运行的应用程序进行交互的。我们提出了WatchPoint，一个模拟用户系统，它通过针对运行中的应用程序生成并执行诊断脚本来模仿真实开发者的行为，产生结构化的观察结果来指导编码模型的重试。与以往针对单文件编辑或使用不可执行指标进行评估的方法不同，我们在Web-Bench上运行，这是一个包含50个多文件Web项目、共1,000个顺序依赖任务的基准，并通过确定性的端到端测试进行验证。WatchPoint恢复了57.6%……

    arXiv:2609.26204v1 Announce Type: cross  Abstract: When a professional web developer's code fails a test, they do not simply re-read the stack trace. They open the application in a browser, click buttons, inspect computed styles, and run diagnostic commands to understand what went wrong. Existing feedback mechanisms for coding agents rely on screenshots, LLM-as-a-judge scoring, or natural-language corrections, but few interact with the live application the way a developer would. We introduce WatchPoint, a simulated-user system that mimics real developer behavior by generating and executing diagnostic scripts against the running application, producing structured observations that guide the coding model's retry. Unlike prior approaches that target single-file edits or evaluate using non-executable metrics, we operate on Web-Bench, a benchmark of 50 multi-file web projects comprising 1,000 sequentially dependent tasks, verified by deterministic end-to-end tests. WatchPoint recovers 57.6% 
    
[^37]: 面向大语言模型越狱攻击防御的动态深度提示优化

    Dynamic Deep Prompt Optimization for Defending Against Jailbreak Attacks on LLMs

    [https://arxiv.org/abs/2609.26185](https://arxiv.org/abs/2609.26185)

    提出首个基于深度提示优化的越狱防御方法DDPO，利用大语言模型自身中间层特征并通过轻量级多层感知机动态生成防御嵌入注入后续层，在不修改模型权重的情况下实现随输入自适应的安全防御。

    

    大语言模型（LLMs）在众多应用中展现出令人瞩目的能力，但仍然容易受到越狱攻击的影响，这类攻击会诱导模型产生有害或非预期的内容。虽然模型微调是实现安全对齐的一种选择，但其成本高昂且容易导致灾难性遗忘。提示优化已成为一种有前景的替代方案，然而现有的基于提示的防御通常依赖于静态修改（例如固定的前缀或后缀），无法适应多样化和不断演变的攻击。我们提出了动态深度提示优化，这是首个基于深度提示优化的越狱防御方法。DDPO 利用目标大语言模型自身的中间层作为特征提取器，通过一个轻量级多层感知机动态生成防御性嵌入。随后，这些定制化的嵌入被注入到后续的中间层中，从而在不修改大语言模型权重的情况下实现依赖于输入的防御。

    arXiv:2609.26185v1 Announce Type: cross  Abstract: Large Language Models (LLMs) demonstrate impressive capabilities across many applications but remain vulnerable to jailbreak attacks, which elicit harmful or unintended content. While model fine-tuning is an option for safety alignment, it is costly and prone to catastrophic forgetting. Prompt optimization has emerged as a promising alternative, yet existing prompt-based defenses typically rely on static modifications (e.g., fixed prefixes or suffixes) that cannot adapt to diverse and evolving attacks.   We propose Dynamic Deep Prompt Optimization (DDPO), the first jailbreak defense based on deep prompt optimization. DDPO uses the target LLM's own intermediate layers as feature extractors to dynamically generate defensive embeddings via a lightweight multilayer perceptron. These tailored embeddings are then injected into a subsequent intermediate layer, enabling an input-dependent defense without modifying the LLM's weights. This desig
    
[^38]: 模态门控深度适配器：在冻结嵌入模型上以精确保持方式添加新模态

    Modality-Gated Deep Adapters: Adding a Modality to a Frozen Embedding Model with Exact Preservation

    [https://arxiv.org/abs/2609.26182](https://arxiv.org/abs/2609.26182)

    提出模态门控深度适配器，可在冻结的多模态嵌入模型上添加新模态，同时保证现有输出逐位不变地被精确保持，且共存的多个模态包之间通过精确零隔离矩阵实现完全隔离。

    

    多模态嵌入模型被大规模部署：检索索引、基准测试结果和行为审计都依赖于基础模型的精确输出。使用现有的参数高效方法将此类模型扩展到新模态会悄悄地改变这些输出；LoRA风格的适配无论权重是否合并都会重写文本路径，导致所有已存储的嵌入失效。我们提出模态门控深度适配器：将瓶颈适配器附加到冻结的多模态嵌入LLM的每个解码器层，并按模态分组为各模态专属的包，仅在其自身模态被编码时才执行。其结果是添加了一个新模态而对现有输出零改变：没有包认领的输入按位不变地经过基础模型自身的计算图，而共同加载的多个包之间通过精确零隔离矩阵实现组合。这两个性质均以命题形式给出，并且在任意训练之后仍然成立，而不仅仅在初始化时成立。

    arXiv:2609.26182v1 Announce Type: new  Abstract: Multimodal embedding models are deployed at scale: retrieval indices, benchmark results, and behavioral audits all depend on the base model's exact outputs. Extending such a model to a new modality with existing parameter-efficient methods silently changes those outputs; LoRA-style adaptation rewrites the text path whether or not the weights are merged, invalidating every stored embedding. We propose modality-gated deep adapters: bottleneck adapters attached to every decoder layer of a frozen multimodal embedding LLM, grouped into per-modality packs that execute only while their own modality is being encoded. The result is a modality added with zero change to existing outputs: inputs no pack claims traverse the base model's own computation graph, bit-for-bit unchanged, and co-loaded packs compose with an exact-zero isolation matrix. Both properties are stated as propositions, hold after arbitrary training rather than only at initializati
    
[^39]: 幅值剖面剪枝：面向Transformer压缩的无校准结构化注意力头移除方法

    Magnitude Profile Pruning: Calibration-Free Structured Attention Head Removal for Transformer Compression

    [https://arxiv.org/abs/2609.26177](https://arxiv.org/abs/2609.26177)

    该论文提出了无需校准数据、梯度计算或Hessian估计的免训练注意力头剪枝方法——幅值剖面（MP）评分，通过权重行范数的统计离群点检测识别并移除冗余注意力头，并提供支持分组查询注意力（GQA）的MP-G变体，实现硬件友好的Transformer模型压缩。

    

    arXiv:2609.26177v1 公告类型：新论文 摘要：对注意力头进行结构化剪枝为压缩Transformer语言模型提供了一种硬件友好的方式。然而，现有的衡量注意力头级别重要性的方法需要校准数据、梯度计算或Hessian矩阵估计，这些要求带来了额外开销，并使方法依赖于数据。我们的工作提出了幅值剖面（Magnitude Profile, MP）评分，这是一种无需训练的注意力头重要性判据，通过对权重行范数进行统计离群点检测来识别可移除的注意力头。投影权重落在总体主体范围内的注意力头被剪枝，而表现出离群范数（承载着不成比例的表示能力）的注意力头则被保留。我们的工作进一步提出了MP-G，这是一个处理分组查询注意力（GQA）的变体，它将共享键值组的分数分配到相关联的查询头上。在五个模型上以12.5%-50%的注意力头稀疏度在WikiText-2困惑度上进行评估，MP-G取得了……（摘要原文在此处截断）

    arXiv:2609.26177v1 Announce Type: new  Abstract: Structured pruning of attention heads provides a hardware-friendly way to compress Transformer language models. However, existing methods for measuring head-level importance require calibration data, gradient computation, or Hessian estimation. These requirements add extra overhead and make the methods depend on the data. Our work presents Magnitude Profile (MP) scoring, a training-free criterion for head importance that identifies dispensable heads through statistical outlier detection on weight row norms. Heads whose projection weights fall within the population bulk are pruned, while heads exhibiting outlier norms, which carry disproportionate representational capacity, are preserved. Our work further gives MP-G, a variant that handles Grouped Query Attention (GQA) by distributing shared key-value group scores across associated query heads. Across five models evaluated on WikiText-2 perplexity at 12.5%-50% head sparsity, MP-G achieves
    
[^40]: DTOC：面向AI智能体自适应上下文管理的动态工具输出压缩

    DTOC: Dynamic Tool Output Compression for Adaptive Context Management in AI Agents

    [https://arxiv.org/abs/2609.26121](https://arxiv.org/abs/2609.26121)

    提出动态工具输出压缩框架DTOC，通过在外部存储器中保留完整工具输出并在上下文中插入可逆占位符，实现LLM智能体的可扩展上下文管理和按需信息恢复。

    

    随着智能体能力的增长，实际限制越来越多地源于受限的上下文窗口而非模型容量。常见的策略，如截断、启发式老化和有损摘要，可能会丢弃有用信息或引入幻觉风险。为了解决这些挑战，我们提出了动态工具输出压缩（DTOC），这是一个面向基于LLM的智能体的可扩展上下文管理框架，它将上下文更新建模为智能体推理循环中显式且可逆的操作。DTOC在外部存储器中保留完整的工具输出，同时在活动上下文中插入紧凑的占位符，从而在需要时实现选择性重建。我们对该DTOC机制进行了形式化，将其集成到ReAct风格的智能体架构中，并提供了一个面向生产的实现，支持按需恢复被压缩的输出。在DeepSWE上的实验揭示了模型相关的效应：对于响应性模型

    arXiv:2609.26121v1 Announce Type: cross  Abstract: As agent capabilities have grown, practical limitations increasingly stem from constrained context windows rather than model capacity. Common strategies, such as truncation, heuristic aging, and lossy summarization, may discard useful information or introduce hallucination risk. To address these challenges, we propose Dynamic Tool Output Compression (DTOC), a framework for scalable context management in LLM-based agents that models context updates as explicit and reversible operations within the agent reasoning loop. DTOC retains full tool outputs in external memory while inserting compact placeholders into the active context, enabling selective reconstruction when needed. We formalize the DTOC mechanism, integrate it into a ReAct-style agent architecture, and provide a production-oriented implementation supporting on-demand restoration of compressed outputs. Experiments on DeepSWE reveal model-dependent effects: for responsive models 
    
[^41]: 可微分模糊推理层：面向大语言模型的单调、可组合的序数推理头

    Differentiable Fuzzy Inference Layer: A Monotone, Compositional Ordinal Reasoning Head for Large Language Models

    [https://arxiv.org/abs/2609.26113](https://arxiv.org/abs/2609.26113)

    提出可微分模糊推理层（DFIL），一种双路径预测头，通过有序隶属函数与t-范数运算，赋予大语言模型对序数类别的单调性以及无需组合训练数据的量词组合推理能力，解决了模型无法正确组合“大多数”等序数量词的问题。

    

    最先进的语言模型在解释“大多数学生中的大多数通过了”这类表述时，通常会回答“大多数”，尽管将两个“大多数”组合起来得到的比例应更接近“一些”。我们将这一失败归因于架构选择而非数据不足：标准分类头将序数类别视为独立的标签，没有任何机制来尊重其自然排序或对其进行代数组合。我们提出了可微分模糊推理层（DFIL），这是一种双路径预测头，将标准分类器与基于有序隶属函数库的标量瓶颈分支相配对。DFIL提供了仅靠标签头无法继承的两个结构原语：对底层量的单调性，以及通过t-范数运算实现的组合推理，且无需任何组合训练数据。标量分支还提供了一个可解释的接口，用于分析残差误差。我们实例化……

    arXiv:2609.26113v1 Announce Type: new  Abstract: A state-of-the-art language model asked to interpret "most of most students passed" typically answers "most," though composing two instances of "most" yields a proportion closer to "some." We trace this failure to an architectural choice rather than a data deficit: standard classifier heads treat ordinal categories as independent labels, with no mechanism to respect their natural ordering or compose them algebraically. We introduce the Differentiable Fuzzy Inference Layer (DFIL), a dual-path prediction head pairing a standard classifier with a scalar-bottlenecked branch grounded in a bank of ordered membership functions. DFIL supplies two structural primitives that a label-only head cannot inherit: monotonicity in the underlying quantity, and compositional reasoning via t-norm operations without any compositional training data. The scalar branch additionally provides an interpretable interface for analyzing residual errors. We instantiat
    
[^42]: TSS：面向特定领域大语言模型投机解码的目标端稀疏化

    TSS: Target-Side Sparsification for Speculative Decoding in Domain-Specific Large Language Models

    [https://arxiv.org/abs/2609.26100](https://arxiv.org/abs/2609.26100)

    该论文提出TSS框架，通过在投机解码的目标端跳过特定层实现稀疏化，在降低验证成本的同时提高草稿接受率，并保持甚至提升下游任务性能。

    

    投机解码通过轻量级草稿模型与目标验证模型的协作来加速大语言模型推理。现有方法主要改进草稿端，而目标模型通常保持稠密且不变。我们证明，在特定领域推理场景下，全深度的目标验证并非总是最优选择。与直觉相反，跳过选定的目标层不仅可以降低验证成本，还能同时提高草稿接受率，并保持甚至提升下游任务性能。基于这一观察，我们提出了TSS，一个面向投机解码的目标端稀疏化框架。TSS采用一种兼顾接受率与指标的广度搜索来探索多层跳过配置，且不在这两个目标之间强加固定的优先级。所选配置被存储在一个领域到配置的映射中，并由轻量级跳过控制器应用。

    arXiv:2609.26100v1 Announce Type: new  Abstract: Speculative decoding accelerates large language model inference through collaboration between a lightweight draft model and a target verifier. Existing methods mainly improve the draft side, while the target model is typically kept dense and unchanged. We show that, under domain-specific inference, full-depth target verification is not always the optimal choice. Counter-intuitively, skipping selected target layers can reduce verification cost while simultaneously increasing draft acceptance and preserving, or even improving, downstream task performance. Based on this observation, we propose TSS, a target-side sparsification framework for speculative decoding. TSS employs an acceptance- and metric-aware breadth search to explore multi-layer skip configurations without imposing a fixed priority between the two objectives. The selected configurations are stored in a domain-to-configuration mapping and applied by a lightweight skip controlle
    
[^43]: 一个领域，多种语言：无需配对数据，组合领域与语言LoRA实现跨语言遥感多模态大语言模型

    One Domain, Many Tongues: Composing Domain and Language LoRAs for Cross-Lingual Remote-Sensing MLLMs without Paired Data

    [https://arxiv.org/abs/2609.26097](https://arxiv.org/abs/2609.26097)

    MODL通过联合训练领域LoRA与语言LoRA并强制二者在每层保持互正交，无需任何多语言遥感配对数据即可为英语遥感多模态大模型添加新语言，同时保留其多语言能力。

    

    遥感多模态大语言模型（MLLM）目前仅在英语上进行训练和评估，而纯文本指令数据却覆盖了超过100种语言。我们提出了MODL（互正交领域-语言组合方法），这是一种无需任何多语言遥感样本即可为英语遥感MLLM添加新语言的方案：在英语遥感影像上训练的领域LoRA与仅用文本训练的语言LoRA被联合学习，并通过一个损失项确保两个更新在整个训练过程中每一层都保持相互正交。这一约束是该方案的关键成分。若没有它，同样的训练虽然能正确回答遥感问题，但会用英语作答，并抹去基础模型的大部分多语言文本能力，且在三个随机种子中有一个会出现发散；从免训练合并到先验正交性变体等十六种替代方案均以同样的方式失败。MODL修复了所有种子上的所有失败：答案正确且以目标语言给出……

    arXiv:2609.26097v1 Announce Type: new  Abstract: Remote-sensing (RS) multimodal large language models (MLLMs) are trained and evaluated only in English, while text-only instruction data covers over 100 languages. We propose MODL (Mutually Orthogonal Domain-Language composition), a recipe that adds new languages to an English RS MLLM without a single multilingual RS example: a domain LoRA trained on English RS imagery and a language LoRA trained on text alone are learned jointly, under one loss term that keeps the two updates mutually orthogonal at every layer throughout training. This constraint is the recipe's active ingredient. Without it, the same training answers RS questions correctly but in English, erases much of the base model's multilingual text ability, and diverges on one seed in three; sixteen alternatives, from training-free merging to prior orthogonality variants, fail the same way. MODL repairs every failure on every seed: answers are correct and in the target language 5
    
[^44]: SpecialEduBench：面向自闭症儿童语言干预中知识、技能与态度的视觉语言模型基准测试

    SpecialEduBench: Benchmarking Vision-Language Models on Knowledge, Skill, and Attitude in Language Intervention for Autistic Children

    [https://arxiv.org/abs/2609.26090](https://arxiv.org/abs/2609.26090)

    提出了SpecialEduBench基准，首次从知识、技能和态度三个维度评估视觉语言模型在自闭症儿童语言干预中的实际教学能力，其技能与态度题项基于真实干预录像，需要结合视觉与语言信息进行综合判断。

    

    语言是大多数自闭症儿童早期干预的核心目标。由于干预目标和方法因儿童而异，这项工作需要教师一对一进行，并随着情境的展开逐一判断。人工智能如今正被引入这项工作，然而现有的特殊教育基准测试考察的是模型“知道什么”，而非模型“在儿童面前实际做什么”。构建这样的基准并不简单，因为一个回应是否是好的教学取决于儿童刚刚做了什么，因此不存在固定的标准答案。判定依据的证据既包括视觉信息也包括语言信息，因为等待的时长、目光的转移以及儿童的回应在文字记录中不会留下任何痕迹。我们提出了SpecialEduBench，它从知识、技能和态度三个维度衡量教学能力，包含4,537个知识题项，以及基于真实干预录像构建的200个技能题项和68个态度题项，态度题项跨越……

    arXiv:2609.26090v1 Announce Type: new  Abstract: Language is the target of most early intervention for autistic children. Because the goal and the method change from child to child, the work falls to a teacher who takes one child at a time and judges each scene as it unfolds. Artificial intelligence is now being brought to that work, yet the benchmarks that reach special education ask what a model knows rather than what it does in front of a child. Building one is not straightforward, since whether a response is good teaching depends on what the child has just done, so no answer key applies. The evidence that settles it is visual as much as verbal, since the length of a wait, a shift of gaze, and the child's uptake leave no trace in a transcript. We introduce \emph{SpecialEduBench}, which measures pedagogical competence along knowledge, skill, and attitude, with 4,537 knowledge items and with 200 skill items and 68 attitude items built on recorded intervention, the attitude items cross
    
[^45]: CoVeR：智能体检索中基于覆盖度的验证器调用路由

    CoVeR: Coverage-Based Routing of Verifier Calls in Agentic Retrieval

    [https://arxiv.org/abs/2609.26086](https://arxiv.org/abs/2609.26086)

    CoVeR 利用冻结句子嵌入的覆盖度边际作为单一阈值门控，仅在证据状态模糊时才调用 LLM 验证器，从而在保持答案准确率几乎不变的前提下，大幅削减智能体检索中验证器反复处理证据的调用成本。

    

    智能体检索系统会发出一系列搜索查询，并且必须在每一步判断目前已收集的证据是否足以停止。将这一决策委托给 LLM 验证器或提示裁判可以使停止时机变得可靠，但验证器随后需要在每次检索步骤之后重新处理不断增长的证据，这是一笔巨大的重复成本。我们证明，其中大部分调用可以在不显著影响答案准确率的情况下被跳过：在冻结的句子嵌入覆盖度边际上设置单一阈值，即可检测出证据仍明显不完整的状态，而验证器只在模糊的剩余情况下被调用，我们将这一门控机制称为 CoVeR（基于覆盖度的验证器路由）。在三个多跳问答基准上（评估协议在全规模运行前已固定），采用 CoVeR 门控的智能体在答案准确率上与全预算智能体以及始终验证的基线相差不到一个 EM 点。它减少了 62……（原文摘要在此处截断）

    arXiv:2609.26086v1 Announce Type: new  Abstract: An agentic retrieval system issues a sequence of search queries and must decide, at each step, whether the evidence collected so far is enough to stop. Delegating that decision to an LLM verifier or a prompt judge makes stopping reliable, but the verifier then reprocesses the growing evidence after every retrieval step, a substantial repeated cost. We show that most of these calls can be skipped without materially changing answer accuracy: a single threshold on a frozen sentence-embedding coverage margin detects the states in which the evidence is still plainly incomplete, and the verifier is called only on the ambiguous remainder, a gate we call CoVeR (Coverage-based Verifier Routing). Across three multi-hop QA benchmarks, with the evaluation protocol fixed before the full-scale run, the CoVeR-gated agent matches the answer accuracy of both the full-budget agent and the always-verify baseline within a fraction of an EM point. It cuts 62
    
[^46]: TopoCompress：面向分布式边缘MoE推理的拓扑感知令牌压缩算法

    TopoCompress: Topology Aware Token Compression Algorithm for Distributed Edge MoE Inference

    [https://arxiv.org/abs/2609.26061](https://arxiv.org/abs/2609.26061)

    TopoCompress提出了一种部署与拓扑感知的令牌压缩框架，通过联合优化令牌压缩、专家部署与复制、GPU-CPU驻留和协同路由，实现通信高效的分布式边缘MoE推理。

    

    混合专家模型通过为每个令牌稀疏激活专家，以适度的开销提升模型容量。然而，将MoE部署在资源受限的边缘服务器上时，由于专家分布在异构服务器之间，会产生大量的跨服务器通信。现有的部署方法仅优化原始令牌流量，而传统的压缩方法虽然考虑语义，却忽略了依赖拓扑的路由成本。因此，独立优化导致通信和资源利用效率低下。本文提出了TopoCompress，一个面向高效通信的分布式边缘MoE推理的部署与拓扑感知令牌压缩框架。该框架联合优化令牌压缩、专家部署与复制、GPU-CPU驻留以及协同路由，以平衡跨服务器传输、推理质量和资源使用。为解决令牌级压缩与epoch级部署之间的耦合问题……（摘要在此处截断）

    arXiv:2609.26061v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) models improve capacity with moderate overhead by sparsely activating experts per token. However, deploying MoE across resource-constrained edge servers incurs substantial cross-server communication as experts are distributed across heterogeneous servers. Existing placement methods optimize for raw token traffic, while conventional compression considers semantics but ignores topology-dependent routing costs. Consequently, independent optimization leads to inefficient communication and resource utilization. This paper proposes TopoCompress, a deployment- and topology-aware token compression framework for communication-efficient distributed edge MoE inference. It jointly optimizes token compression, expert deployment/replication, GPU-CPU residency, and collaborative routing to balance cross-server transmission, quality, and resource use. To address the coupling between token-level compression and epoch-level depl
    
[^47]: 优化扩散大语言模型中的去噪轨迹：一种轻量级进化启发式方法

    Optimizing Denoising Trajectories in dLLMs: A Lightweight Evolutionary Heuristic Approach

    [https://arxiv.org/abs/2609.26052](https://arxiv.org/abs/2609.26052)

    该论文通过分析Transformer注意力模式，揭示了dLLM基于置信度的去噪调度器存在EOS溢出和近端偏差两种失效模式的根源（无效token获得过高注意力权重），并提出一种轻量级进化启发式方法，利用有效注意力分数优化去噪轨迹。

    

    扩散大语言模型近来成为传统自回归大语言模型的一种有前景的替代方案。通过利用双向注意力和并行解码，dLLM能够实现更高效的生成。然而，它们在推理时需要一个精心设计的去噪调度器（该调度器在训练阶段并不存在），其选择会显著影响生成质量。虽然基于置信度的启发式调度器已展现出强大的实证性能，但它们存在两个关键失效模式：EOS溢出和近端偏差。通过对Transformer注意力模式的深入分析，我们揭示了这些失效源于某些位置对无效token（如[MASK]和[EOS]）分配了不成比例的高注意力权重，从而产生误导性的置信度信号。基于这一洞察，实证证据表明有效的注意力分数可以提供互补的（信息）……

    arXiv:2609.26052v1 Announce Type: new  Abstract: Diffusion Large Language Models (dLLMs) have recently emerged as a promising alternative to conventional Auto-Regressive (AR) Large Language Models (LLMs). By leveraging bidirectional attention and parallel decoding, dLLMs enable more efficient generation. However, they require a carefully designed denoising scheduler at inference time (absent during training) whose choice significantly impacts generation quality. While confidence-based heuristic schedulers have shown strong empirical performance, they suffer from two critical failure modes: EOS Overflow and Proximal Bias. Through in-depth analysis of the Transformer's attention patterns, we reveal that these failures stem from certain positions assigning disproportionately high attention weights to invalid tokens (e.g., [MASK] and [EOS]), which produce misleading confidence signals. Building on this insight, empirical evidence shows that valid attention scores can provide complementary 
    
[^48]: FIRE：面向可靠语言模型智能体的故障知情运行时工程

    FIRE: Failure-Informed Runtime Engineering for Reliable Language-Model Agents

    [https://arxiv.org/abs/2609.26048](https://arxiv.org/abs/2609.26048)

    该论文提出FIRE，通过在不改变模型权重和用户提示的前提下，在失败发生前的状态处施加自然语言指令与动作拒绝等运行时策略，显著提升语言模型智能体的重复交付可靠性，在Terminal-Bench 2.1上pass^2最高提升9.2个百分点。

    

    语言模型智能体往往能够找到可行的解决方案，却无法稳定地将其交付。我们研究了运行时策略：由智能体框架在观测到失败之前的状态处施加的有针对性的自然语言指令和动作拒绝，且不改变模型权重或用户提示词。借助这一方法，在保持能力不变的情况下，我们观察到交付可靠性的显著提升。在完整的87个任务的Terminal-Bench 2.1套件上（每个任务尝试两次），策略使三个GPT-5.6层级的重复成功率（pass^2）均得到提升：Luna从50.6%提升至54.0%，Terra从55.2%提升至60.9%，Sol从64.4%提升至73.6%。Sol的两次尝试最佳成功率仅变化1.2个百分点，而重复成功率提升了9.2个百分点，这表明策略主要作用在于将可实现的解决方案转化为可靠的交付。我们进一步在Terra的冻结组合下测试了14个任务：策略引导的Terra达到71.4%，而无辅助的Sol为64.3%，成本约为其一半（原文摘要在此处截断）。

    arXiv:2609.26048v1 Announce Type: cross  Abstract: Language-model agents often reach a working solution and then fail to consistently deliver it. We study runtime policies: targeted natural-language instructions and action denials applied by the agent harness at states that preceded observed failures, without changing model weights or the user prompt. With this, keeping capability constant, we observe a meaningful unlock in delivered reliability. Across the complete 87-task Terminal-Bench 2.1 suite, with two attempts per task, policies increase repeated success (pass^2) in all three GPT-5.6 tiers: 50.6% to 54.0% for Luna, 55.2% to 60.9% for Terra, and 64.4% to 73.6% for Sol. Sol's best-of-two success changes by 1.2 points while repeated success rises by 9.2, showing that policies chiefly convert reachable solutions into dependable delivery. We further cover 14 tasks under Terra's frozen portfolio. Policy-guided Terra reaches 71.4%, compared with 64.3% for unassisted Sol, at about half 
    
[^49]: 面向可信AI的真实性：将表达怀疑、来源追溯与信念修正作为可工程化的立场

    Truth for Believable AI: Expressed Doubt, Provenance, and Belief Revision as an Engineerable Stance

    [https://arxiv.org/abs/2609.26035](https://arxiv.org/abs/2609.26035)

    该论文提出在固定语言模型之上构建一个可工程化的“认知行为层”，通过表达不确定性、来源门控断言和持久化信念修正机制，实现对修正的可审计确认，并能抵抗错误修正。

    

    对话式智能体通常以统一的高置信度语气表达答案。我们测试表达不确定性、基于来源的断言以及显式信念修正能否在固定语言模型之上实现为一个行为层；我们并不测试可信度或信任本身。该层结合了三种认知要素：逐条断言的置信度与类型化来源、由来源门控的表达规则，以及一个具有可审计确认机制并对错误修正具有部分抵抗能力的持久化修正存储。我们在一个构建的、可机械评分的多会话基准上，使用合成模型和Qwen2.5-0.5B-Instruct对其进行评估。合成工具通过了全部五项检查。在真实模型上，确认正确性（一项构造性保证）在100%的情况下成立，且真实修正比错误修正更常被接受（在已持有信念上为0.44对0.15；若包含规则接受的对未持有事实的修正，则为0.875对0.420）……

    arXiv:2609.26035v1 Announce Type: new  Abstract: Conversational agents often express answers in a uniformly confident register. We test whether expressed uncertainty, provenance-aware assertion, and explicit belief revision can be implemented as a behavior layer over a fixed language model; we do not test believability or trust. The layer combines three epistemic states, per-claim confidence and typed provenance, a provenance-gated expression rule, and a persistent revision store with auditable acknowledgments and partial resistance to false corrections. We evaluate it on a constructed, mechanically scored multi-session benchmark using a synthetic model and Qwen2.5-0.5B-Instruct. The synthetic instrument passes all five checks. On the real model, acknowledgment soundness, a by-construction guarantee, holds in 100% of cases, and true corrections are accepted more often than false ones (0.44 vs. 0.15 on held beliefs; 0.875 vs. 0.420 including rule-accepted corrections of unheld facts), b
    
[^50]: 领域自适应预训练增强水处理语义表示，助力大规模结构化文献挖掘

    Domain-Adaptive Pretraining Enhances Water Treatment Semantic Representation for Large-Scale Structured Literature Mining

    [https://arxiv.org/abs/2609.26034](https://arxiv.org/abs/2609.26034)

    本研究基于约29.7亿词元的水处理语料库持续预训练，开发了领域自适应模型WaterBERT，在水处理文本的语义表示和结构化信息提取任务上超越了通用型及领域特定型BERT模型。

    

    水处理研究正在迅速扩展，但从这些研究中获得的大量知识仍然分散在非结构化的文献中。该领域目前仍缺乏一个专门的语言模型，能够高效捕捉水处理领域的特定语义以支持大规模文献挖掘。在本文中，我们通过开发WaterBERT来解决这一问题，WaterBERT是一个领域自适应的编码器模型，专为水处理文本的语义表示和结构化信息提取而设计。WaterBERT是在包含约29.7亿个词元的大规模水处理语料库上通过持续预训练开发的。基于WaterBERT的三个微调模型在下游任务上进行了系统评估，在通用型和领域特定型BERT模型中取得了最佳的整体性能，其中多类别处理工艺分类的F1分数为90.12%，命名实体识别为79.50%，关系抽取为74.04%。

    arXiv:2609.26034v1 Announce Type: new  Abstract: Water treatment research is expanding rapidly, but much of the knowledge acquired from this research remains scattered across unstructured literature. The field still lacks a dedicated language model that can efficiently capture water treatment-specific domain semantics for large-scale literature mining. Here, we address this by developing WaterBERT, a domain-adapted encoder model designed for semantic representation and structured information extraction from water treatment texts. WaterBERT was developed by continual pretraining on a large-scale water treatment corpus comprising about 2.97 billion tokens. Three fine-tuned models based on WaterBERT were systematically evaluated on downstream tasks, achieving the best overall performance among general-purpose and domain-specific BERT models, with F1 scores of 90.12% for multiclass treatment process classification, 79.50% for named entity recognition, and 74.04% for relation extraction. Be
    
[^51]: MICRO：用于严重错误发现的多保真度主动搜索

    MICRO: Multi-Fidelity Active Search for Severe Error Discovery

    [https://arxiv.org/abs/2609.26025](https://arxiv.org/abs/2609.26025)

    MICRO是一个多保真度主动搜索框架，通过联合建模质量评分与标注损失、按预测影响聚类以选择多样化候选、并利用滚动评估发现价值，在共享预算下最大化严重错误的确认发现数量。

    

    人类反馈在成本和信息量上可能各不相同。强反馈能够揭示严重错误，但成本高昂，因此更廉价的质量评分可以帮助决定应该标注哪些条目。我们提出了MICRO（Multi-Fidelity Impact Clustered Rollout，多保真度影响聚类滚动），这是一个主动搜索框架，它将共享预算分配给这些反馈类型，以最大化已确认的严重错误发现数量。MICRO以条目特征为条件，联合建模评分和标注损失，从而引导获取过程。它根据获取候选对严重性概率的预测影响进行聚类，以选择多样化的候选者，然后使用滚动评估来估计其发现价值。在WMT20英德翻译数据上的实验表明，评分同时改善了损失重建和严重性预测。MICRO在四种预算和评分成本设置下均取得了最高的平均发现数量，其中一种设置下与改编的MF-ENS性能相当，而在其他设置下相比所有六种对比策略均取得了显著提升。

    arXiv:2609.26025v1 Announce Type: cross  Abstract: Human feedback can vary in cost and informativeness. Strong feedback can reveal severe errors but is costly, so cheaper quality ratings can help decide which items to annotate. We propose MICRO (Multi-Fidelity Impact Clustered Rollout), an active search framework that allocates a shared budget to these feedback types to maximise confirmed severe error discoveries. MICRO jointly models ratings and annotation losses conditional on item features to steer acquisition. It clusters acquisitions by their predicted impact on severity probabilities to select diverse candidates, then uses rollout to estimate their discovery value. Experiments on WMT20 English-German show that ratings improve both loss reconstruction and severity prediction. MICRO achieves the highest mean discovery count across four budget and rating cost settings, with similar performance to adapted MF-ENS in one and significant gains over all six comparison policies, including
    
[^52]: 面向真实会话语音增强的多说话人提取挑战

    Challenges of Multi-Speaker Extraction for Real Conversational Speech Enhancement

    [https://arxiv.org/abs/2609.25948](https://arxiv.org/abs/2609.25948)

    该论文提出一种新的损失函数来缓解真实对话中过多静音对多说话人提取模型训练的影响，将STOI从0.55提升至0.60、频率加权分段信噪比从4.35提升至5.12。

    

    目标说话人提取和多说话人提取是在存在其他说话人和/或噪声的情况下，从期望的一个或多个说话人中提取语音的技术。针对该任务的神经网络方法通常使用模拟数据集进行训练和评估，其中目标语音与说话人注册样本数量均衡，且注册样本与目标语音高度匹配。然而，在真实的多方对话中，参与者沉默的时间往往多于说话的时间，且其注册语音样本可能与对话中的目标语音存在显著差异。这些因素会影响此类技术在真实对话录音上的训练与评估。本工作提出了一种新的损失函数，有助于减轻训练中过多静音带来的影响，将STOI从0.55提升至0.60，将频率加权分段信噪比从4.35提升至5.12。此外，注册样本与目标语音之间不匹配的影响……

    arXiv:2609.25948v1 Announce Type: cross  Abstract: Target-speaker and multi-speaker extraction are techniques for extracting speech from a desired speaker or desired speakers in the presence of other speakers and/or noise. Neural network approaches for this task are often trained and evaluated using simulated datasets, with balanced amounts of target speech and speaker enrolment samples which closely match the target speech. However, in real multi-party conversations, participants are often silent for more time than they are speaking, and their enrolment speech samples can differ substantially from the target speech in the conversation. These factors can impact the training and evaluation of these techniques on recordings of real conversations. This work proposes a new loss function, which helps mitigate the effect of excess silence in training, improving STOI from 0.55 to 0.60, and frequency-weighted segmental SNR from 4.35 to 5.12. Additionally, the impact of the mismatch between the
    
[^53]: ClusterFewshot：改进大语言模型工作流的少样本优化

    ClusterFewshot: Improving Few-shot Optimization for LLMs workflow

    [https://arxiv.org/abs/2609.25939](https://arxiv.org/abs/2609.25939)

    ClusterFewshot通过将语义聚类结构与效用感知评分相结合来构建更具代表性的少样本示例集，在多个基准测试中显著降低优化成本的同时，持续提升了LLM工作流的准确率。

    

    大语言模型（LLM）工作流的性能通常依赖于选择一小组上下文内示例来引导模型在新任务上的行为。近期的方法通过将成功的推理路径加入提示中来改进这一过程。然而，这些方法的示例选择依赖于随机采样或基于指标的排序，忽视了任务的语义结构。我们提出了ClusterFewshot，一种将语义结构化与效用感知评分相结合的策略，用于构建具有代表性和高效性的少样本示例集。在基于DSPy的流水线中进行评估，ClusterFewshot在多个基准测试中大幅降低了优化成本，同时在独立提示调优和混合提示-权重优化两种场景下，相比先前基于自举（bootstrap）的方法均持续提升了准确率。

    arXiv:2609.25939v1 Announce Type: new  Abstract: The performance of large language model (LLM) workflows often depends on selecting a small set of in-context demonstrations to guide model behavior on new tasks. Recent methods improve this process by augmenting prompts with successful reasoning paths. However, their demonstration selection relies on random sampling or metric-based rankings, overlooking the semantic structure of the task. We propose ClusterFewshot, a strategy that combines semantic structuring with utility-aware scoring to construct representative and effective few-shot demonstration sets. Evaluated within DSPy-based pipelines, ClusterFewshot substantially reduces optimization cost across multiple benchmarks, while consistently improving accuracy relative to prior bootstrap-based methods in both standalone prompt tuning and hybrid prompt-weight optimization.
    
[^54]: 针对哪个标准认证？执行标签决定了Text-to-SQL保形弃答所报告的风险

    Certified Against Which Oracle? Execution Labels Set the Reported Risk of Conformal Abstention for Text-to-SQL

    [https://arxiv.org/abs/2609.25938](https://arxiv.org/abs/2609.25938)

    Text-to-SQL保形弃答证书所报告的风险严重依赖于校准时使用的正确性标准，使用更严格的多实例测试套件和专家盲评标签后发现，证书的实际风险远高于其标称水平。

    

    Text-to-SQL的保形弃答证书的真实性取决于其校准所依据的正确性标签。通过执行一致性读取置信度的不确定性流程，这些标签来自基准测试自带的单一数据库——一个已知较为宽松的标准。我们在Spider-Realistic上进行了一项预注册干预，将该数据库替换为基准测试提炼出的多实例测试套件。在四个SQL专家模型检查点和两种数据分割方案下，这一替换使证书的留出风险比其自身标签所报告的风险高出2.73至10.23个百分点。两个标准都无法反映专家所认定的真实风险。在两位SQL专家的盲评标签下，标称校准为0.10的证书在两个检查点上分别带有20.0和17.2个百分点的实际风险。更严格的标准在两个方向上都会出错：它拒绝的大多数答案并未被判为错误，而它接受的一些答案却是错误的。对其拒绝内容的AI辅助普查显示……（摘要截断）

    arXiv:2609.25938v1 Announce Type: cross  Abstract: A conformal abstention certificate for text-to-SQL is only as truthful as the correctness labels it is calibrated on. The uncertainty pipelines that read confidence off execution consistency take those labels from the single database a benchmark ships, an oracle known to be lenient. We run a preregistered intervention on Spider-Realistic, swapping that database for the benchmark's distilled multi-instance test suite. Across four SQL-specialist checkpoints and two split schemes, the swap raises the certificate's held-out risk 2.73 to 10.23 points above the risk its own labels report. Neither oracle reports the risk experts assign. Under blinded labels from two SQL experts, a certificate calibrated at a nominal 0.10 carries 20.0 and 17.2 points of risk on two checkpoints. The stricter oracle errs in both directions: most of the answers it rejects are not judged wrong, and some of those it accepts are. An AI-assigned census of what it rej
    
[^55]: 知情掩码：面向扩散大语言模型强化学习的结构感知扰动

    Informed Masking: Structure-Aware Perturbation for Reinforcement Learning in Diffusion Large Language Models

    [https://arxiv.org/abs/2609.25927](https://arxiv.org/abs/2609.25927)

    该论文发现扩散大语言模型 rollout 中存在上游/下游 token 结构及子问题难度不对称现象，并提出知情掩码方法，通过为每个 token 计算优先级分数来选择掩码位置，从而在有限蒙特卡洛预算下显著提升强化学习对齐的似然估计质量。

    

    扩散大语言模型已成为自回归模型的一种高效替代方案，然而通过强化学习（RL）对其进行对齐，需要在每次 rollout 的较小蒙特卡洛预算下，从掩码重建子问题中估计似然代理。现有方法通过均匀随机掩码来构建这些子问题，但哪些子问题应被优先处理仍是一个悬而未决的问题。我们识别出 dLLM rollout 中存在系统性的上游/下游结构：某些 token 在被揭示时会引发邻近未解码位置的大幅置信度变化，我们称之为上游 token；另一些 token 仅引起微小的局部变化，因此属于下游 token。我们发现，掩码下游 token 所产生的子问题比掩码上游 token 的子问题具有显著更好的适定性，我们将这一现象称为子问题难度不对称性。基于这一观察，我们提出了知情掩码（IM），它从去噪器中推导出每个 token 的优先级分数（原文此处截断）。

    arXiv:2609.25927v1 Announce Type: new  Abstract: Diffusion Large Language Models (dLLMs) have emerged as an efficient alternative to autoregressive models, yet aligning them via Reinforcement Learning (RL) requires likelihood surrogates estimated from masked reconstruction subproblems under a small Monte Carlo budget per rollout. Existing methods construct these subproblems by uniform random masking, leaving open the question of which subproblems to prioritize. We identify a systematic upstream/downstream structure in dLLM rollouts. Some tokens, when revealed, trigger large confidence changes in nearby undecoded positions; we call them upstream. Others induce only small local changes and are therefore downstream. We find masking downstream tokens yields substantially better-posed subproblems than masking upstream tokens, a phenomenon we term subproblem difficulty asymmetry. Based on the observation, we propose Informed Masking (IM), which derives a per-token priority score from the den
    
[^56]: 重新思考基于长度的训练：语音Token语言模型中的批次组成与损失归一化

    Rethinking Length-Based Training: Batch Composition and Loss Normalization in Speech Token Language Models

    [https://arxiv.org/abs/2609.25890](https://arxiv.org/abs/2609.25890)

    该论文通过匹配比较解耦了语音token语言模型中基于长度训练的多个混杂因素，发现由短到长的排序本身并无独立收益，其表面效果主要源于批次组成和损失归一化方式的差异。

    

    由短到长训练是语音模型的一种简单课程学习方式，但其收益往往难以解释。在语音token语言模型中，基于长度的训练可能会改变打乱策略、批次组成、token保留情况以及批次平均损失下的token权重。我们通过匹配比较来解耦这些因素。在所测试的设置中，当批次组成和token暴露固定时，由短到长的排序没有表现出独立的收益。首轮分组在批次平均损失下降低了Mimi的困惑度，但在token平衡损失下未观察到这一收益。跨分词器的结果与块长度变化和token加权之间的联系相一致。这项工作为研究可变长度语音模型中基于长度的训练提供了一个系统性的分析协议。

    arXiv:2609.25890v1 Announce Type: new  Abstract: Short-to-long training is a simple curriculum for speech models, but its gains can be difficult to interpret. In speech token language models, length-based training can change the shuffle policy, batch composition, token retention, and token weights under batch-mean loss. We disentangle these factors through matched comparisons. In the tested settings, short-to-long ordering shows no independent benefit when batch composition and token exposure are fixed. First-epoch grouping lowers perplexity for Mimi under batch-mean loss, but this gain is not observed under token-balanced loss. The cross-tokenizer results are consistent with a link between chunk-length variation and token weighting. This work provides a systematic analysis protocol for studying length-based training in variable-length speech models.
    
[^57]: 冰岛手语的孤立手语识别：低资源环境下的实验

    Isolated Sign Language Recognition for Icelandic Sign Language: Experiments in a Low-resource Setting

    [https://arxiv.org/abs/2609.25862](https://arxiv.org/abs/2609.25862)

    该论文首次针对极低资源的冰岛手语开展孤立手语识别实验，发现跨语言迁移（先在美国手语数据上预训练再微调）能带来最大收益，将准确率提升 14-24 个百分点。

    

    我们展示了针对冰岛手语（\'ITM）的孤立手语识别（ISLR）的首批实验。我们使用 \'ITM SignWiki 数据集，该数据集源自冰岛语-\'ITM 双语在线词典。它是真正的低资源数据集：1,845 个视频涵盖 849 个类别，其中 86% 的类别仅有两个样本，使得完整任务实际上成为跨手语者的单样本识别。我们在三个词汇量递增的任务（22、117 和 849 个类别）上比较了两个开源 ISLR 框架 OpenHands 和 SPOTER，并评估了三种姿态估计器和两种跨语言迁移方法。仅使用 \'ITM 数据时，SPOTER 在所有三个任务上都优于 OpenHands，且 MediaPipe 姿态估计的结果优于 AlphaPose 或 SDPose。跨语言迁移带来了最大的收益：在美国手语数据上预训练 SPOTER 后再在 \'ITM 上微调，可将准确率提高 14-24 个百分点，在三个任务上分别达到 72.7%、47.9% 和 22.6%，并且

    arXiv:2609.25862v1 Announce Type: new  Abstract: We present the first experiments on isolated sign language recognition (ISLR) for Icelandic Sign Language (\'ITM). We use \'ITM SignWiki, a dataset derived from a bilingual Icelandic--\'ITM online dictionary. It is genuinely low-resource: 1,845 videos cover 849 classes, 86% of which have only two examples, making the full task effectively one-shot recognition across signers. We compare two open-source ISLR frameworks, OpenHands and SPOTER, on three tasks of increasing vocabulary size (22, 117 and 849 classes), and evaluate three pose estimators and two forms of cross-lingual transfer. With \'ITM data alone, SPOTER outperforms OpenHands on all three tasks, and MediaPipe poses give better results than AlphaPose or SDPose. Cross-lingual transfer brings the largest gains: pretraining SPOTER on American Sign Language data before finetuning on \'ITM raises accuracy by 14--24 percentage points, to 72.7%, 47.9% and 22.6% on the three tasks, and 
    
[^58]: BELXTR：通过上下文化词元检索实现生物医学实体链接

    BELXTR: Biomedical Entity Linking via Contextualized Token Retrieval

    [https://arxiv.org/abs/2609.25859](https://arxiv.org/abs/2609.25859)

    提出了基于多向量（后期交互）架构的BELXTR模型，通过利用词元级匹配信息将XTR扩展到生物医学实体链接任务，在十个语料库中一半上超越现有最先进方法，recall@1平均提升5个百分点。

    

    生物医学实体链接将文本中的提及消歧到知识库（KB）中的实体，是信息抽取流水线的基石。虽然基于嵌入的模型是该任务的主流方法，但它们存在一个关键局限：将提及（和实体）压缩为单一向量，迫使模型将关键的细粒度差异平均化。我们提出了BELXTR，一种基于多向量（又称后期交互）架构的新型嵌入模型，能够利用词元级别的匹配信息。BELXTR通过集成现有的任务特定训练目标并探索主动查询扩展，将原始XTR模型扩展到生物医学实体链接任务。在十个语料库和五个知识库上的实验表明，BELXTR在一半的语料库上超越了当前最先进的方法，recall@1平均提升5个百分点。其中最大的提升出现在具有挑战性的跨物种基因消歧任务上。

    arXiv:2609.25859v1 Announce Type: new  Abstract: Biomedical Entity Linking disambiguates mentions to entities in a knowledge base (KB), making it the cornerstone of information extraction pipelines. While embedding-based models are a popular approach for the task, they suffer from a key limitation. They compress mentions (and entities) into a single vector, forcing the model to average away crucial fine-grained differences. We present BELXTR, a novel embedding model based on the multi-vector (a.k.a. late interaction) architecture, which allows to leverage token-level matching information. BELXTR extends the original XTR model to biomedical entity linking by integrating an existing task-specific training objective and exploring active query expansion. Experiments across ten corpora and five KBs show that BELXTR improves upon current state-of-the-art in half of the corpora with an average improvement of 5pp recall@1. The largest gains are reported on the challenging cross-species gene di
    
[^59]: MemoryAthena：潜在记忆与生成记忆之上的自适应路由

    MemoryAthena: Adaptive Routing over Latent and Generated Memories

    [https://arxiv.org/abs/2609.25853](https://arxiv.org/abs/2609.25853)

    MemoryAthena 提出在直接检索、基于检索线索生成和无表生成三条记忆路径之间进行自适应因果路由，在冻结主干与记忆模块的情况下，依据反事实似然优势学习轻量路由头，并通过有界插值将生成的记忆按情境动态融合到检索残差中。

    

    学习型记忆方法将信息存储在显式表格中，并通过单独的读取器来消费这些信息，使得寻址、存储和读取可以独立修改。我们研究有用的记忆是否也可以被生成，而不仅仅是被检索。MemoryAthena 使用三条路径：直接 Engram 检索（E）、从检索到的 Engram 线索生成（GE），以及在不查询记忆表的情况下从因果主干状态生成（GH）。生成的记忆是有条件有用的：它在一种情境中可以补充 E，但在另一种情境中可能干扰 E。因此，MemoryAthena 将 E 视为锚点，并学习生成的表示何时应当介入。在主干、记忆、生成器和读取器全部冻结的条件下，利用 GE 和 GH 相对于 E 的反事实未来词元似然优势，训练一个轻量级的因果路由头。在推理时，被接纳的候选通过有界插值修改 E 的残差……

    arXiv:2609.25853v1 Announce Type: new  Abstract: Learned-memory methods store information in an explicit table and consume it through a separate reader, allowing addressing, storage, and reading to be modified independently. We study whether useful memory can also be generated rather than only retrieved. MemoryAthena uses three pathways: direct Engram retrieval (E), generation from retrieved Engram cues (GE), and generation from causal backbone states without consulting the memory table (GH). Generated memory is conditionally useful: it can complement E in one context but interfere with it in another. MemoryAthena therefore treats E as an anchor and learns when a generated representation should intervene. With the backbone, memory, generators, and readers frozen, a lightweight causal routing head is trained from counterfactual future-token likelihood advantages of GE and GH relative to E. At inference time, an admitted candidate modifies the E residual through bounded interpolation, wh
    
[^60]: ARAFA：一个由大语言模型生成的阿拉伯语事实核查数据集

    ARAFA: An LLM-Generated Arabic Fact-Checking Dataset

    [https://arxiv.org/abs/2609.25833](https://arxiv.org/abs/2609.25833)

    研究者利用大语言模型通过三步自动化流水线构建了包含18万余个标注声明-证据对的大规模阿拉伯语事实核查数据集Arafa，有效缓解了阿拉伯语事实核查资源稀缺的问题。

    

    由于数据集和资源的稀缺，自动事实核查在阿拉伯语自然语言处理领域构成了重大挑战。在本文中，我们介绍了Arafa，一个新的大规模现代标准阿拉伯语事实核查数据集，该数据集通过一个利用大语言模型（LLM）的自动化框架构建而成。数据集的构建采用三步流水线：（1）从阿拉伯语维基百科页面生成带有支持性文本证据的声明；（2）通过声明变异生成带有反驳证据的具有挑战性的反事实声明；（3）通过自动验证步骤，检验生成的声明是被其伴随证据所支持或反驳，还是证据不足以判断声明的有效性。最终得到的数据集包含181,976个声明-证据对，标注为“支持”、“反驳”或“信息不足”。在样本上进行的人工评估……（原文摘要在此处截断）

    arXiv:2609.25833v1 Announce Type: new  Abstract: Automatic fact-checking poses a significant challenge in Arabic natural language processing due to the scarcity of datasets and resources. In this manuscript, we introduce Arafa, a new large-scale dataset for fact-checking in Modern Standard Arabic, constructed through an automated framework leveraging large language models (LLMs). The dataset was constructed through a three-step pipeline: (1) claim generation from Arabic Wikipedia pages with supporting textual evidence, (2) claim mutation to generate challenging counterfactual claims with refuting evidence, and (3) an automatic validation step to validate that the generated claims are either supported or refuted by their accompanying evidence, or if the evidence does not provide enough information to judge the validity of the claims. The resulting dataset comprises 181,976 claim-evidence pairs labeled as supported, refuted, or not enough information. Human evaluation carried out on a te
    
[^61]: 跨文本片段的基于代理的验证审计

    Auditing Proxy-Based Validation Across Text Spans

    [https://arxiv.org/abs/2609.25808](https://arxiv.org/abs/2609.25808)

    该论文提出“验证契约”审计框架，通过在评分文本片段之外重新评估代理规则，发现当评估分数与代理标签共享同一文本片段时，二者的一致性可能源于共享的表面证据而非目标语义构念，从而揭示基于代理的验证可能高估评估分数的有效性。

    

    评估分数通常通过与廉价代理标签的一致性来进行验证。然而，当分数和代理从同一个文本片段计算得出时，这种一致性可能源于两者共享的表面证据，而非代理所要代表的语义构念。我们通过将分数、其文本片段、代理和目标构念声明为一个“验证契约”来明确这一区别，然后在评分片段之外严格重新评估该代理规则。在一个仅改变共享文本边界的受控 HotpotQA 正确性实验中，分数与其代理的一致性远高于其在 50 字符前缀处与正确性的一致性：差距为 +0.184，且从 120 字符开始缩减至最多 +0.045。在该短前缀处，分数仍能预测答案字符串是否稍后出现（AUC 为 0.634），而等价性测试显示其与正确性的一致性处于随机水平，因此所报告的代理一致

    arXiv:2609.25808v1 Announce Type: cross  Abstract: Evaluation scores are often validated by their agreement with inexpensive proxy labels. When the score and the proxy are computed from the same text span, however, that agreement can arise from surface evidence the two share rather than from the semantic construct the proxy is meant to represent. We make the distinction explicit by declaring the score, its span, the proxy and the target construct as a validation contract, then re-evaluating that proxy rule strictly outside the scored span. In a controlled HotpotQA correctness experiment varying only the shared text boundary, the score agrees with its proxy far better than with correctness at a 50-character prefix: the gap is +0.184, collapsing to at most +0.045 from 120 characters onward. At that short prefix the score still predicts whether the answer string appears later (AUC 0.634) while an equivalence test places its agreement with correctness at chance, so the reported proxy agree
    
[^62]: 最新精确匹配注意力

    Latest Exact Match Attention

    [https://arxiv.org/abs/2609.25802](https://arxiv.org/abs/2609.25802)

    提出最新精确匹配注意力（LEMA），证明了带思维链的LEMA transformer与字随机存取机在计算和内存上可以相互高效模拟，并给出了基于直通估计器和软注意力退火的实用训练方法。

    

    我们提出了最新精确匹配注意力（LEMA），这是一种针对transformer的注意力变体，其中查询和键被二值化，且每个查询仅关注最新的精确匹配键。我们证明了带思维链的LEMA transformer可以模拟字随机存取机，这与最近针对限制较少的最右侧硬注意力所证明的结论类似。与先前的硬注意力变体不同，对精确匹配的限制使得反向方向也能高效实现：字随机存取机可以模拟LEMA transformer，且每个token的计算成本与上下文长度无关。综合起来，这些结果在计算和内存两方面建立了这两种计算模型之间的紧密对应关系。在理论之外，我们提出了一种LEMA transformer的训练方法，通过为二值化操作使用直通估计器，以及将软注意力代理逐步退火至LEMA，来处理其不可微操作。在一个合成联想回忆任务上（摘要在此处截断）

    arXiv:2609.25802v1 Announce Type: cross  Abstract: We introduce latest exact match attention (LEMA), an attention variant for transformers where queries and keys are binarized and each query attends only to the latest exactly matching key. We prove that LEMA transformers with chain of thought can simulate word-RAMs, as was recently shown for the less restrictive rightmost hard attention. In contrast to prior hard attention variants, the restriction to exact matches enables an efficient converse direction: word-RAMs can simulate LEMA transformers at a cost per token independent of the context length. Together, these results yield a close correspondence between the two computational models in terms of both compute and memory. Beyond the theory, we propose a training method for LEMA transformers that handles their non-differentiable operations with a straight-through estimator for the binarization and a soft attention surrogate annealed towards LEMA. On a synthetic associative recall task
    
[^63]: 对关于人类与AI生成语言中量子结构的评论（arXiv:2512.07881和arXiv:2601.06104）的回复

    Reply to comments arXiv:2512.07881 and arXiv:2601.06104 on quantum structure in human and AI-generated language

    [https://arxiv.org/abs/2609.25797](https://arxiv.org/abs/2609.25797)

    本文针对评论者对人类与AI生成语言中量子结构研究的批评进行逐点回应，澄清了实验协议、纠缠识别判据、玻色-爱因斯坦拟合等关键问题，并更正了一处不影响结果的排版错误。

    

    我们对M. Sienicki和K. Sienicki（arXiv:2512.07881）以及K. Sienicki（arXiv:2601.06104）就我们关于人类语言中量子力学统计（arXiv:2407.14924）和AI生成语言中量子结构（arXiv:2511.21731）的工作所提出的评论进行回复。我们感谢作者们的仔细阅读，并回应我们认为的主要批评要点：大语言模型实验中所用协议的探索性；边缘分布定律破坏以及“默认语境性”判据在识别纠缠中的作用；单独使用玻色-爱因斯坦拟合的有限诊断价值；将最低能级分配给最频繁出现词语的含义；以及大语言模型所使用的向量空间与量子态空间之间的关系。我们还更正了arXiv:2511.21731表3中的一个排版错误，该错误不影响所报告的CHSH值。

    arXiv:2609.25797v1 Announce Type: new  Abstract: We reply to the comments by M. Sienicki and K. Sienicki (arXiv:2512.07881) and by K. Sienicki (arXiv:2601.06104) on our work on quantum-mechanical statistics in human language (arXiv:2407.14924) and on quantum structure in AI-generated language (arXiv:2511.21731). We thank the authors for their careful reading and address what we consider to be the main points of criticism: the exploratory nature of the protocol used in the experiments with large language models; the role of marginal-law violations, and of the Contextuality-by-Default criterion, in the identification of entanglement; the limited diagnostic value of a Bose-Einstein fit taken in isolation; the meaning of assigning the lowest energy levels to the most frequent words; and the relation between the vector spaces used by LLMs and quantum state spaces. We also correct a typographical error in Table 3 of arXiv:2511.21731, which does not affect the reported CHSH value.
    
[^64]: 证候、协同与安全：面向中医处方生成的结构化推理与知识驱动对齐

    Syndrome, Synergy, and Safety: Structured Reasoning and Knowledge-Driven Alignment for TCM Prescription Generation

    [https://arxiv.org/abs/2609.25755](https://arxiv.org/abs/2609.25755)

    该论文提出一个渐进式四阶段框架（SFT → PG-CoT → 动态SFT → K-RL），通过理法方药范式下的可审计推理、随证加减的诊疗轨迹建模以及十八反等禁忌规则驱动的DPO对齐，显著提升了中医处方生成的质量、可解释性与安全性。

    

    将大语言模型应用于中医（TCM）处方生成揭示了三个临床关键缺口：模型在缺乏遵循“理法方药”范式的可审计推理的情况下直接产生端到端映射（SR缺口）；将每次就诊孤立处理，缺乏通过“随证加减”进行复诊调整的能力（LA缺口）；以及无法执行如“十八反”等绝对配伍禁忌规则（SC缺口）。我们提出一个渐进式四阶段框架（SFT → PG-CoT → Dynamic → K-RL）来逐一解决这些缺口：PG-CoT在理法方药范式约束下进行思维链蒸馏，生成可审计的诊断链条；动态SFT通过显式的转归推理建模患者诊疗轨迹；K-RL将确定性药理学规则编码为基于规则的DPO偏好信号。在12个微调模型和6个零样本基线的对比实验中，我们的框架相比零样本基线显著提升了处方质量。

    arXiv:2609.25755v1 Announce Type: new  Abstract: Applying large language models to Traditional Chinese Medicine (TCM) prescription generation reveals three clinically critical gaps: models produce end-to-end mappings without auditable reasoning following the li-fa-fang-yao paradigm (SR Gap), treat each encounter in isolation without follow-up adjustment via sui zheng jia jian (LA Gap), and fail to enforce absolute contraindication rules such as Shi Ba Fan (SC Gap). We propose a progressive four-stage framework (SFT $\to$ PG-CoT $\to$ Dynamic $\to$ K-RL) that addresses each gap: PG-CoT constrains CoT distillation under the li-fa-fang-yao paradigm to produce auditable diagnostic chains, Dynamic SFT models patient trajectories with explicit transition reasoning, and K-RL encodes deterministic pharmacological rules as rule-based DPO preference signals. Across 12 fine-tuned models and 6 zero-shot baselines, our framework substantially improves prescription quality over zero-shot baselines--
    
[^65]: 缓慢衰减与沉默表达：语言模型谱系中的迭代式潜意识特质传递

    Slow Decay and Silenced Expression: Iterated Subliminal Trait Transfer in Language-Model Lineages

    [https://arxiv.org/abs/2609.25721](https://arxiv.org/abs/2609.25721)

    该研究发现，通过过滤数据潜意识传递的模型特质能够在语言模型谱系中存续十代且仅缓慢衰减，但其表达被“沉默”——无法通过输出中的关键词检测到，只能依靠激活探针揭示其仍然存在。

    

    语言模型越来越多地使用其他模型的输出进行训练，形成了我们称之为“谱系”的链条，其中存在于某一世代的特质可以传递给下一代。先前关于潜意识学习的研究表明，教师模型的特质可以通过不携带任何该特质内容的过滤数据传递给学生模型。然而，这些证据仅覆盖了单次训练步骤。我们研究这种特质在谱系中是保持还是消退。我们将该特质注入 Qwen2.5-7B-Instruct 的三个副本，并从每个副本出发将训练步骤迭代至第十代，在相同的保留提示上以两种方式读取每一代模型：一是关键词筛查，在模型输出中寻找该特质的表达；二是激活探针，将每个模型相对于基座模型的位移投影到一个由其他谱系教师模型构建的方向上。我们报告了两项发现。第一，该特质在三个谱系中均持续存在十代之久。［摘要在此处截断］

    arXiv:2609.25721v1 Announce Type: cross  Abstract: Language models are increasingly trained on the outputs of other models, forming chains that we call lineages, in which a trait present in one generation can pass to the next. Prior work on subliminal learning has shown that a teacher's trait can transmit to a student through filtered data carrying none of the trait's content. However, the evidence covers only a single training step. We study whether such a trait holds or fades across lineages. We instill the trait into three copies of Qwen2.5-7B-Instruct and iterate the training step to depth ten from each, reading every generation two ways on the same held-out prompts: a keyword screen that looks for expressions of the trait in the model's output, and an activation probe that projects each model's displacement from the base onto a direction built from the other lineages' teachers. We report two findings. First, the trait persists through ten generations across three lineages. The ins
    
[^66]: 任务状态应以多强的程度影响LLM智能体？

    How Strongly Should Task State Influence an LLM Agent?

    [https://arxiv.org/abs/2609.25686](https://arxiv.org/abs/2609.25686)

    该论文通过固定任务规则与模型、只改变任务状态进入智能体的强度（从展示文本、逐轮指令到硬性强制执行门），首次将任务状态管理方式对LLM智能体可靠性的贡献进行了解耦与量化。

    

    长时程的指派型工作要求LLM智能体追踪任务的状态：哪些步骤已完成、被阻塞、被取消或可重复执行。现有智能体系统要么将这一状态以文本形式保存在提示词中并依赖模型自行读取，要么将状态转移到一个强制执行该状态的模块中；而每个系统都是作为整体被评估的，因此无人知晓可靠性中究竟有多少来自状态的“被展示”、“被告知”或“被强制执行”。我们固定任务规则、模型以及配对的测试回合，仅改变任务状态传递到智能体的强度：原始对话记录、精确的检查清单、由任务简报编译而成的状态机所给出的逐轮指令（该状态机仅依据执行回执推进），或是在该状态机上加装的、会拒绝违反状态操作的强制执行门；每个回合均通过与动态真值的精确载荷匹配来评分。在三个模型、两种推理模式和两个领域上，在无逐轮推理的情况下有四项发现成立：展示……（原文摘要在此处截断）

    arXiv:2609.25686v1 Announce Type: cross  Abstract: Long-horizon assigned work requires an LLM agent to track the state of a task: which steps are done, blocked, cancelled, or open to repetition. Agent systems either keep this state as text in the prompt and rely on the model to read that text, or move the state into a module that enforces it, and each system is evaluated as a whole, so no one knows how much reliability comes from the state being shown, told, or enforced. We fix the task rules, the model, and paired episodes and vary how strongly task state reaches the agent: a raw transcript, an exact checklist, per-turn directives from a state machine compiled from the brief and advanced only by execution receipts, or an enforcement gate on that machine that refuses state-violating actions; every episode is scored by exact payload matching against dynamic ground truth. Across three models, two reasoning regimes, and two domains, four findings hold without per-turn reasoning: displayin
    
[^67]: 从话语到网络：建模俚语在Reddit子版块间的采纳与传播

    From Utterances to Networks: Modelling Slang Adoption and Diffusion Across Subreddits

    [https://arxiv.org/abs/2609.25669](https://arxiv.org/abs/2609.25669)

    该研究利用大语言模型作为可扩展的标注器，结合社会互动与语言特性双重视角，构建人工标注基准并建模网络俚语在Reddit子版块间的采纳与传播机制。

    

    近年来，新词在在线社区中的采纳与传播重新受到关注。随着诸如APT（指一首K-pop歌曲）等网络俚语，以及诸如"Canon Event"（指令人尴尬但关键的事件）等短语在网上走红，理解促成其成功的机制变得日益重要。以往的研究往往要么从社会互动的角度，要么从俚语本身的语言特性来解释俚语的传播，但很少将两者结合起来。其中一个主要障碍是在大规模在线交流中标注俚语使用的成本高昂。然而，大语言模型（LLM）的最新进展使其有可能作为可扩展的标注器来完成此类任务。在本研究中，我们首先构建了一个人工标注的基准数据集，以评估LLM在检测真实Reddit交流中俚语使用方面的性能。随后，我们利用……

    arXiv:2609.25669v1 Announce Type: new  Abstract: Adoption and diffusion of neologisms in online communities have received renewed attention in recent years. As internet slang terms such as APT, referring to a K-pop song, and phrases such as Canon Event meaning an embarrassing but pivotal event, go viral online, it becomes increasingly important to understand the mechanisms that contribute to their success. Prior studies have often explained slang diffusion either from the perspective of social interaction or from the linguistic properties of the slang itself, but rarely from both perspectives together. One major obstacle has been the high cost of annotating slang usage in large-scale online communication. Recent advances in large language models (LLMs), however, make it possible to use them as scalable annotators for such tasks. In this study, we first curate a human-annotated benchmark to evaluate LLM performance in detecting slang usage in real Reddit communication. We then leverage 
    
[^68]: 基于贝叶斯多臂老虎机Gittins指数的高效成本感知LLM评估

    Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices

    [https://arxiv.org/abs/2609.25645](https://arxiv.org/abs/2609.25645)

    提出GittinsEval，一种基于贝叶斯最优Gittins策略的成本感知LLM评估方法，通过轻量级在线更新高效决定下一个待评估配置及停止时机，在多个基准上以更低的评估成本取得有竞争力的性能。

    

    在每个基准测试项目上穷尽评估每个候选LLM配置以找出高性能配置的成本十分高昂。我们将配置选择问题形式化为一个成本感知的贝叶斯多臂老虎机问题，并提出GittinsEval，该方法借鉴贝叶斯最优的Gittins策略来决定下一步评估哪个配置以及何时停止。我们通过一个随时推荐规则对该策略进行了扩展，使其能够同时覆盖完全评估和部分评估的配置，并使用LCB风格的分数来考虑后验不确定性。GittinsEval在计算上非常高效，只需在离线预计算之后进行轻量级的在线更新。在GSM8K、PIQA、AlpacaEval和MMLU响应矩阵上，GittinsEval始终具有竞争力，在大样本基准上相比配置级贝叶斯优化取得了尤为显著的提升，在大候选任务上相比不考虑成本的bandit基线也表现更优。至关重要的是，GittinsEval通常能够以（原文在此处截断）……

    arXiv:2609.25645v1 Announce Type: cross  Abstract: Exhaustively evaluating every candidate LLM configuration on every benchmark item to identify a high-performing one is costly. We formulate configuration selection as a cost-aware Bayesian bandit problem and propose GittinsEval, which draws on the Bayesian-optimal Gittins policy to determine which configuration to evaluate next and when to stop. We extend the policy with an anytime recommendation rule over both fully and partially evaluated configurations, using an LCB-style score to account for posterior uncertainty. GittinsEval is computationally efficient, requiring only lightweight online updates after offline precomputation. Across GSM8K, PIQA, AlpacaEval, and MMLU response matrices, GittinsEval is consistently competitive, with particularly strong gains over configuration-level Bayesian optimization on large-example benchmarks and over cost-unaware bandit baselines on large-candidate tasks. Crucially, GittinsEval often attains ne
    
[^69]: Qwen3.8-Omni：迈向原生全模态智能体

    Qwen3.8-Omni: Towards Native Omni-Modal Agents

    [https://arxiv.org/abs/2609.25611](https://arxiv.org/abs/2609.25611)

    Qwen3.8-Omni-Flash 通过原生多模态协同训练策略与百万 token 上下文窗口，在保持文本能力的同时大幅提升多模态理解推理与长程智能体任务表现，可作为主智能体或子智能体落地于视频剪辑、长音视频翻译等真实生产场景。

    

    我们推出了 Qwen3.8-Omni-Flash，一个面向真实世界多模态生产力任务的原生多模态智能体模型。与以往主要强调感知与交互的全模态模型相比，Qwen3.8-Omni-Flash 在多模态理解与推理能力以及长程智能体任务的表现上均有显著提升。这些能力得益于一种原生多模态协同训练策略，该策略在保持强大文本领域能力的同时，促进了智能体能力从文本向音频和视频任务的迁移。该模型继承了 Qwen3.8-Next 的稀疏混合专家架构，并将上下文窗口扩展至一百万 token，支持长上下文多模态推理与长程规划。这些进展使其能够作为主智能体或专用子智能体集成到生产工作流中，支持视频剪辑、长音视频翻译、音乐条件生成等应用。

    arXiv:2609.25611v1 Announce Type: new  Abstract: We introduce Qwen3.8-Omni-Flash, a natively multimodal agentic model for real-world multimodal productivity. Compared with previous omni models, which primarily emphasized perception and interaction, Qwen3.8-Omni-Flash substantially improves multimodal understanding and reasoning, as well as performance on long-horizon agentic tasks. These capabilities are supported by a native multimodal co-training strategy that preserves strong text-domain capabilities while facilitating the transfer of agentic capabilities from text to audio and video tasks. The model inherits the sparse mixture-of-experts (MoE) architecture of Qwen3.8-Next and extends the context window to one million tokens, supporting long-context multimodal reasoning and long-horizon planning. These advances enable integration into production workflows as a primary agent or a specialized sub-agent, supporting video editing, long-form audio and video translation, music-conditioned
    
[^70]: 重新布线还是门控？指令微调如何塑造大语言模型中的知识冲突回路

    Rewired or Gated? How Instruction Tuning Shapes Knowledge-Conflict Circuits in LLMs

    [https://arxiv.org/abs/2609.25602](https://arxiv.org/abs/2609.25602)

    该论文首次从机制层面比较了基础模型与指令微调模型的知识冲突解决回路，发现指令微调并非重新布线底层回路，而是通过门控方式重新加权同一批后层注意力头，使模型更依赖参数化记忆而非上下文信息。

    

    在语言模型中，选择相信提示词还是相信模型权重，是由少数可识别的注意力头决定的。指令微调会改变模型在冲突情境下的行为，但它究竟是重新布线了底层回路，还是仅仅对已有组件进行门控/重新加权，目前仍不清楚。我们在三个模型家族上提供了首个基础模型与指令微调模型之间冲突解决回路的机制性比较。五种独立的方法——节点与边归因、叠加角色分析、因果消融和路径修补——一致收敛于门控机制：相同的注意力头、相同的后层位置被发现是被重新加权而非替换，节点重叠度高达0.60-0.82。在行为层面，微调使模型更倾向于参数化记忆，使得指令微调模型比基础模型更强烈地拒绝简短的反事实上下文，这与天真的“用户跟随”预期相反。（注：原文摘要在此处被截断）

    arXiv:2609.25602v1 Announce Type: cross  Abstract: In language models, the choice between believing the prompt and believing the weights is made by a handful of identifiable attention heads. Instruction tuning changes how models behave under conflict, but whether it rewires the underlying circuit or merely gates/reweights already present components, remains unknown. We provide the first mechanistic base-vs-instruct comparison of conflict-resolution circuits, across three families (Llama-3.2-3B, Qwen-2.5-3B, Gemma-3-4B). Five independent methods, node and edge attribution, superposition role analysis, causal ablation, and path patching, converge on gating, with the same heads, in the same late-layers, are found to be reweighted rather than replaced with a high node overlap (0.60-0.82). Behaviorally, tuning shifts models toward parametric memory, making instruct models reject a terse counterfactual context far more than base ones, the opposite of a naive user-following expectation. Yet t
    
[^71]: 将长上下文压缩为答案对齐的记忆嵌入以用于大语言模型推理

    Compressing Long Context into Answer-Aligned Memory Embeddings for LLM Inference

    [https://arxiv.org/abs/2609.25537](https://arxiv.org/abs/2609.25537)

    提出CMC框架，将长上下文压缩为与任意冻结解码器嵌入空间对齐的紧凑记忆嵌入，结合问题引导的两层KV缓存与答案导向蒸馏，在不修改解码器权重的情况下有效降低大语言模型推理成本。

    

    大语言模型（LLM）推理受到自注意力二次方扩展和KV缓存线性扩展的制约，随着上下文长度的增加，延迟、能耗和GPU内存需求不断上升。现有的软压缩方法要么在推理时缺乏基于查询的记忆选择机制，要么在训练时缺乏以答案为目标的监督，要么将压缩与特定的解码器架构紧密耦合。我们提出了一个上下文到答案对齐的记忆压缩（CMC）框架，该框架将长输入上下文压缩为紧凑的上下文记忆嵌入（CMEs），并与任意冻结解码器的嵌入空间对齐，从而在不修改解码器权重的情况下降低推理成本。CMC引入了一个两层KV缓存，将问题引导的CME选择与局部上下文窗口相结合，并通过来自冻结LLM的答案导向蒸馏来训练压缩器。实验在九种编码器-解码器组合和四个问答基准上进行了验证。

    arXiv:2609.25537v1 Announce Type: new  Abstract: Large language model (LLM) inference is constrained by the quadratic scaling of self-attention and the linear scaling of the KV cache, increasing latency, energy consumption, and GPU memory demand as context length scales. Existing soft-compression methods either lack query-guided memory selection at inference time, train without answer-targeted supervision, or couple compression tightly to a specific decoder architecture. We propose a Context-to-Answer-Aligned Memory Compression (CMC) framework, which compresses long input contexts into compact Context Memory Embeddings (CMEs) aligned to any frozen decoder's embedding space, reducing inference costs without modifying decoder weights. CMC introduces a two-tier KV cache that combines question-guided CME selection with a local context window, and trains the compressor with answer-targeted distillation from a frozen LLM. Experiments across nine encoder-decoder combinations and four QA bench
    
[^72]: 套娃归因：学习将语言模型输出归因于表示和权重

    Matryoshka attribution: Learning to attribute language model outputs to representations and weights

    [https://arxiv.org/abs/2609.25518](https://arxiv.org/abs/2609.25518)

    提出套娃归因（MAttr），一种利用可微分sigmoid top-k算子学习掩码、通过随机化k在所有稀疏度下同时训练的归因方法，在机制可解释性基准官方排行榜上排名第一。

    

    将语言模型的输出归因于其内部计算是可解释性领域的一个开放问题。现有的方法（使用因果干预、梯度或可学习掩码）要么成本高得不可行，要么难以识别真正具有因果重要性的内部计算。我们提出将归因问题框架化为识别内部组件的嵌套子集以最小化下游损失的问题。为了学习这一任务，我们引入了套娃归因，这是一种掩码学习方法，通过一个简单的可微分sigmoid top-k算子来参数化掩码。通过在训练过程中随机化k，我们同时对所有稀疏度进行监督训练，从而学习到按归因分数排序的组件顺序。MAttr在机制可解释性基准的官方排行榜上排名第一；我们的方法在不同……上识别出稀疏且可跨任务迁移的电路。

    arXiv:2609.25518v1 Announce Type: new  Abstract: Attributing language model outputs to their internal computations is an open problem in interpretability. Existing methods, which use causal interventions, gradients, or learnable masks, either are infeasibly expensive or struggle to identify actual causally-important internal computations. We propose framing attribution as the problem of identifying nested subsets of internal components which minimise a downstream loss. To learn this task, we introduce Matryoshka Attribution (MAttr), a mask learning method that parametrises the mask with a simple differentiable sigmoid top-$k$ operator. We supervise training over all sparsities simultaneously by randomising $k$ over training, resulting in a learned ordering of components by attribution score. MAttr achieves number 1 on the official leaderboard of the Mechanistic Interpretability Benchmark (Mueller et al., 2025); our method identifies sparse and task-transferrable circuits across varying
    
[^73]: 通用分形自然语言决策图：跨异构领域的实时边缘分诊

    Universal Fractal Natural Language Decision Map: Real-Time Edge Triage Across Heterogeneous Domains

    [https://arxiv.org/abs/2609.25498](https://arxiv.org/abs/2609.25498)

    该论文提出了一种无需存储任何权重张量（0 字节显存）的通用分形自然语言决策图，通过沿 Mandelbrot 集混沌边界动态调制 24 字节坐标种子来实时合成布尔、类别和序数三类确定性决策，从而以极低延迟和能耗实现跨异构领域的边缘端实时分诊。

    

    arXiv:2609.25498v1 公告类型：cross 摘要：部署大型语言模型进行运行时操作分诊会带来难以承受的延迟（>100-500 毫秒）、高昂的显存需求（>4-8 GB）以及过多的能量耗散。本文在 Mandelbrot 分形神经合成（Dagli 等，2026）的基础上，提出了通用分形自然语言决策图，通过 werr 机器原生边缘反射运行时和生产级 answerr 平台（https://answerr.me）实现。该引擎完全无需存储权重张量（0 字节显存），通过沿 Mandelbrot 集的混沌边界动态调制 24 字节坐标种子并评估四象限逃逸动力学，来合成确定性决策——noul（布尔型）、choice（类别型）和 score（序数型）。该引擎从生物“系统一”反射弧中汲取灵感，引入了：(i) 带有域投影器 Phi_D 的自动种子路由器，相比线性基线带来 +28.8% 的准确率提升；(ii) Infor……（原文摘要在此处被截断）

    arXiv:2609.25498v1 Announce Type: cross  Abstract: Deploying Large Language Models for runtime operational triage incurs prohibitive latency (>100-500 ms), high VRAM requirements (>4-8 GB), and excessive energy dissipation. Extending Mandelbrot Fractal Neural Synthesis (Dagli et al., 2026), this paper presents the Universal Fractal Natural Language Decision Map, realized via the werr machine-native edge reflex runtime and the production answerr platform (https://answerr.me). Operating entirely without stored weight tensors (0 Bytes VRAM), the engine synthesizes deterministic decisions---noul (Boolean), choice (categorical), and score (ordinal)---by dynamically modulating 24-byte coordinate seeds along the chaotic boundary of the Mandelbrot set and evaluating 4-quadrant escape dynamics. Drawing inspiration from biological System-One reflex arcs, the engine introduces: (i) an Auto-Seed Router with domain projector Phi_D yielding a +28.8% accuracy gain over linear baselines; (ii) an Infor
    
[^74]: 压力下的行为：当用户施压时，六十个语言模型会怎么做

    Conduct Under Pressure: What Sixty Language Models Do When a User Pushes

    [https://arxiv.org/abs/2609.25447](https://arxiv.org/abs/2609.25447)

    该研究对13家厂商的60个语言模型在用户施压情境下进行了大规模行为评测，发现模型“是否屈服于压力”取决于模型代际新旧（新模型更坚定，屈服率与能力指数相关达-0.64），而“以何种方式坚持或屈服”则由厂商特征决定。

    

    我们研究了当用户在不舒适的情境中向大语言模型施压时模型的表现：用户坚持己见、恳求、奉承或哀伤，而模型可能放弃正确的事实、编写它本应拒绝的文档，或为一个会让用户蒙受金钱损失的计划叫好。我们向来自13家厂商的60个模型发送了固定的多轮对话场景（对每个模型完全相同，不随模型回复而变化），并用通过开放编码构建后冻结的编码手册对每份对话记录进行标注：包括一个轨迹（模型坚持立场还是屈服）和一种方式（它如何坚持或屈服）。两项发现界限分明：模型是否坚持与模型代际相关，即模型的更新程度：屈服率与公开的能力指数呈Spearman相关系数-0.64，厂商效应很小。而模型如何坚持则取决于厂商：17个方式编码中有6个按厂商显著区分（置换检验p ≤ 0.001，已对整个编码手册进行多重校正）。我们报告了在通过信度检验的编码上得出的四种厂商画像。

    arXiv:2609.25447v1 Announce Type: new  Abstract: We study what LLMs do when a user applies pressure in an uncomfortable situation: a user insists, begs, flatters or grieves, and the model gives up a correct fact, writes a document it should refuse, or cheers a plan that will cost the user money. We send frozen multi-turn scenes, identical for every model regardless of the reply, to 60 models from 13 vendors, and label each transcript with a codebook built by open coding and then frozen: a trajectory (the model held its position or folded) and a manner (how it held or folded). Two findings separate. Whether a model holds tracks its generation, meaning how recent it is: fold rate correlates with a public capability index at Spearman -0.64, with little vendor effect. How it holds tracks the vendor: six of the 17 manner codes sort by vendor at permutation p <= 0.001, corrected across the codebook. We report four vendor profiles on the codes that cleared reliability.   We also ask which par
    
[^75]: 美国公司法判例中的法律论证挖掘

    Mining Legal Arguments in U.S. Corporate Case Law

    [https://arxiv.org/abs/2609.25441](https://arxiv.org/abs/2609.25441)

    本文构建了首个针对美国联邦税务公司重组判例的专家标注树状结构法律论证语料库，并通过一致性分析发现功能节点标签的标注可靠性高于有向支持边，且路径级可达性比直接连接更稳定。

    

    法律论证挖掘可支持段落分类、检索与论证补全任务。本研究引入了一个由专家标注的数据集，包含42份涉及《美国国内税收法典》（I.R.C.）第368条下公司重组的美国联邦税务判决意见。据我们所知，这是该领域首个专家标注的树状结构论证语料库。语料库中的显式文本片段被赋予五种功能标签之一：规则、分析、结论、背景事实和程序历史。其中规则、分析和结论片段可被链接为有向支持树，而背景事实和程序历史则承担上下文语境功能。该语料库提供了基于片段的、基于句子的、扁平化以及树状结构等多种表示形式。一致性分析表明，功能节点标签的标注可靠性高于有向支持边和隐式中间结论；有向路径的一致性强于直接边的一致性，这表明广泛的可达性比直接连接关系更为稳定。

    arXiv:2609.25441v1 Announce Type: new  Abstract: Legal argument mining supports passage classification, retrieval, and argument completion. This work introduces an expert-annotated dataset of 42 U.S. federal tax opinions on corporate reorganizations under I.R.C. {\S}368. To our knowledge, it is the first expert-annotated, tree-structured argument corpus for this domain. Explicit spans receive one of five functional labels: Rule, Analysis, Conclusion, Background Facts, and Procedural History. Rule, Analysis, and Conclusion spans can be linked into directed support trees, while Background Facts and Procedural History serve a contextual function. The corpus provides span-based, sentence-based, flat, and tree-structured representations. Agreement analysis shows that functional node labels are more reliable than directed support edges and implicit intermediate conclusions. Directed-path agreement is stronger than direct-edge agreement, which indicates that broad reachability is more stable 
    
[^76]: 基于异构批处理的高效迭代检索

    Efficient Iterative Retrieval with Heterogeneous Batching

    [https://arxiv.org/abs/2609.25405](https://arxiv.org/abs/2609.25405)

    Orthrus是一个在统一推理循环中通过异构批处理同时服务嵌入模型与生成模型的检索服务系统，借助带增量池化的分块嵌入和工作负载感知的批处理组成调整，相比基线部署实现了1.28倍至4.52倍的吞吐量提升。

    

    现代信息检索日益同时采用嵌入模型和生成模型来处理复杂查询。然而，当前的服务系统由于孤立地执行这些模型，存在吞吐量低和GPU利用率差的问题。粗粒度的分区方式（例如将GPU专用于特定任务）无法适应动态工作负载，并会产生计算“气泡”。为解决这些问题，我们提出了Orthrus，一个在统一推理循环内执行异构批处理的服务系统。其主要挑战在于统一具有相互冲突计算模式的嵌入与生成工作负载，同时优化批处理组成以实现高性能。Orthrus通过带增量池化的分块嵌入以及以工作负载感知的方式调整批处理组成来应对这些挑战。在四块A100 GPU上的评估表明，与基线部署相比，Orthrus实现了1.28倍至4.52倍的（吞吐量提升）。

    arXiv:2609.25405v1 Announce Type: cross  Abstract: Modern information retrieval increasingly employs both embedding and generative models to handle complex queries. However, current serving systems suffer from low throughput and poor GPU utilization because they execute these models in isolation. Coarse-grained partitioning, such as dedicating GPUs to specific tasks, fails to adapt to dynamic workloads and creates computational "bubbles". To address these, we present Orthrus, a serving system that performs heterogeneous batching within a unified inference loop. The primary challenge lies in unifying embedding and generation workloads with conflicting computational patterns while optimizing batch composition for high performance. Orthrus addresses these challenges through chunked embedding with incremental pooling and by adjusting batch composition in a workload-aware manner. Evaluation on four A100 GPUs shows that, relative to baseline deployments, Orthrus achieves 1.28$\times$--4.52$\
    
[^77]: 单独通过，合并失败：并行LLM智能体开发中的语义协调基准测试

    Passes Alone, Fails Together: Benchmarking Semantic Coordination in Parallel LLM-Agent Development

    [https://arxiv.org/abs/2609.25396](https://arxiv.org/abs/2609.25396)

    该论文提出了 stale 基准测试来衡量并行LLM编码智能体之间的语义协调问题，发现真实合并的拉取请求中干扰极少，但在使用真实 Django 代码构建的受控任务中 97% 的运行出现合并干扰，而一条简单的并发更改描述消息即可恢复 82% 的失败运行。

    

    并行编码智能体可以生成单独运行时有效的补丁，但在合并时却会失败。这种情况发生在当一个智能体更改了另一个智能体仍然依赖的接口或规则时。我们使用 stale——一个用于语义协调的基准测试——来研究这些失败。我们的评估方法是在每个补丁单独运行以及它们的组合上运行相同的测试，仅计算由合并补丁引入的失败。我们使用三个层级：具有受控接口更改的合成任务、成对合并的拉取请求，以及使用真实 Django 辅助函数构建的任务。在对 417 个挖掘的 Django 配对进行的 834 次运行中，在纠正评分程序后仅有一个案例显示出干扰。在使用 12 个 Django 辅助函数构建的任务中，97% 的运行出现了干扰。一条描述已完成的并发更改的消息可以恢复 82% 的失败运行。即使智能体在使用真实代码的受控任务上失败，经过审查的拉取请求中可能只包含很少的未解决并行更改。

    arXiv:2609.25396v1 Announce Type: new  Abstract: Parallel coding agents can produce patches that work alone but fail when merged. This happens when one agent changes an interface or rule that another agent still relies on. We study these failures with stale, a benchmark for semantic coordination. Our evaluation runs the same tests on each patch alone and on their combination, counting only failures introduced by combining the patches. We use three tiers: synthetic tasks with controlled interface changes, pairs of merged pull requests, and constructed tasks that use real Django helpers. Among 834 runs on 417 mined Django pairs, only one showed interference after correcting the grading procedure. On constructed tasks using 12 Django helpers, interference occurred in 97% of runs. A message describing the completed concurrent change recovered 82% of runs. Reviewed pull requests may contain few unresolved parallel changes, even when agents fail on controlled tasks using real code. The const
    
[^78]: TelecomGPT-R1：面向异构电信任务推理的统一后训练

    TelecomGPT-R1: Unified Post-Training for Reasoning Across Heterogeneous Telecom Tasks

    [https://arxiv.org/abs/2609.25356](https://arxiv.org/abs/2609.25356)

    提出了开源统一电信推理模型系列TelecomGPT-R1，通过围绕协议、知识、建模、故障四个维度的感知式数据生成框架，将粗糙的公开电信资料转化为经过验证的问答对和高质量思维链推理轨迹，实现跨异构电信任务的可靠推理。

    

    大型语言模型（LLMs）通过推理标准规范、网络配置、数学模型、源代码和运维日志，为自动化广泛的电信工程任务提供了巨大潜力。然而，现有的电信LLM难以在这些多样化的任务和数据类型上进行可靠推理。通用LLM通常缺乏对电信特定知识的可靠掌握，而电信专用模型通常针对较窄的任务族开发，多任务性能有限。为填补这一空白，我们提出了TelecomGPT-R1，这是一系列开源的统一电信推理模型，围绕四个互补的维度构建：协议、知识、建模和故障。我们首先开发了一个维度感知的数据生成框架，将粗糙的公开电信资料精炼为经过验证的问答对和高质量的思维链（CoT）推理轨迹，从而生成包含……的训练语料库（摘要在此处截断）。

    arXiv:2609.25356v1 Announce Type: new  Abstract: Large language models (LLMs) offer great potential to automate a broad range of telecom engineering tasks by reasoning over standards, network configurations, mathematical models, source code, and operational logs. However, existing telecom LLMs struggle to reliably reason across these diverse tasks and data types. General-purpose LLMs often lack reliable grounding in telecom-specific knowledge, while telecom-specialized models are typically developed for narrower task families and exhibit limited multi-task performance. To fill this gap, we introduce TelecomGPT-R1, a family of open source unified telecom reasoning models structured around four complementary axes: protocol, knowledge, modeling, and fault. We first develop an axis-aware data generation framework that refines coarse public telecom artifacts into verified question-answer pairs and high quality chain-of-thought (CoT) reasoning trajectories, yielding a training corpus contain
    
[^79]: FineWeb-CLaR：面向基准对齐语料库审计的文化、语言与区域标注

    FineWeb-CLaR: Culture, Language, and Region Annotations for Benchmark-Aligned Corpus Auditing

    [https://arxiv.org/abs/2609.25298](https://arxiv.org/abs/2609.25298)

    提出FineWeb-CLaR数据集，为FineWeb和FineWeb-2的全部309亿文档添加文化-语言-区域标注，使预训练语料库与文化基准可在同一维度上对齐，从而支持对语言模型文化覆盖度的语料库审计。

    

    语言模型的文化评估覆盖度与鲁棒性难以诊断，因为预训练语料库和文化基准很少以可比较的元数据进行索引。基准测试日益针对语言、区域和特定地区实践层面的文化现象，而网络规模的语料库通常仅按语言进行组织。一个共享的文化-语言-区域层能够使这些资源具有可比性，从而可以审计某一目标文化现象是否在预训练数据中得到体现、是否被基准测试评估，或两者兼有。为此，我们推出了FineWeb-CLaR，这是一个源自FineWeb和FineWeb-2的大规模标注数据集，它将网络文档置于共享的文化-语言-区域轴上，用于语料库审计和基准对齐。FineWeb-CLaR对来自FineWeb和FineWeb-2的全部309亿（30.9B）文档集合进行了标注，提供基于URL推导的区域标签和文化主题来源信息。我们的区域……（摘要原文在此处截断）

    arXiv:2609.25298v1 Announce Type: new  Abstract: Cultural evaluation coverage and robustness in language models are difficult to diagnose because pretraining corpora and cultural benchmarks are rarely indexed with comparable metadata. Benchmarks increasingly target culturally situated phenomena at the level of languages, regions, and locale-specific practices, while web-scale corpora are usually organized only by language. A shared culture-language-region layer makes these resources comparable, enabling audits of whether a target cultural phenomenon is represented in pretraining data, evaluated by benchmarks or both. To this end, we introduce FineWeb-CLaR, a large-scale annotated dataset derived from FineWeb and FineWeb-2 that places web documents on a shared culture-language-region axis for corpus auditing and benchmark alignment.   FineWeb-CLaR annotates the full 30.9B-document collection from FineWeb and FineWeb-2 with URL-derived region labels and cultural-topic provenance. Our reg
    
[^80]: 训练但不学习：面向LLM智能体作为前置部署工程师的训后交付基准

    Trains but Doesn't Learn: A Post-Training Delivery Benchmark for LLM Agents as Forward-Deployed Engineers

    [https://arxiv.org/abs/2609.25237](https://arxiv.org/abs/2609.25237)

    该论文提出了一个评估LLM智能体作为前置部署工程师完成训后交付能力的基准，揭示了“训练但不学习”（TBDL）这一静默失败模式，并通过运营商验收门控和损坏检测器确保智能体交付的模型真正可信有效。

    

    训后正在成为一项服务（PTaaS）：客户将数据和目标交给运营商，前置部署工程师（FDE）在预算约束、人工审批门控和可复现性要求下返回一个经过微调、评估和部署的模型。将LLM智能体置于FDE的位置提出了一个现有基准无法回答的问题：不是智能体能否提升某个指标，而是它能否被信任以完成交付。我们在一个受治理的交付平面上回答了这个问题：智能体驱动十个阶段，而一个oracle（判定器）根据平台记录的事实对每个阶段进行评分。核心的静默失败是“训练但不学习”（TBDL）的运行：损失下降，所有信号保持正常，但交付的模型并不比基础模型更好。由运营商运行的验收门控在付款前捕获了每一次此类运行，而基于已知损坏运行校准的检测器可在运行中途标记严重损坏。我们运行了四个前沿智能体（Claude Opus 5、GPT-5.6-luna、Gemin...（摘要在此处被截断）

    arXiv:2609.25237v1 Announce Type: cross  Abstract: Post-training is becoming a service (PTaaS): a customer hands an operator data and a goal, and a forward-deployed engineer (FDE) returns a fine-tuned, evaluated, and deployed model under a budget, a human-approval gate, and reproducibility requirements. Seating an LLM agent in the FDE seat raises a question existing benchmarks cannot answer: not whether an agent can raise a metric, but whether it can be trusted to deliver. We answer it on a governed delivery plane, where an agent drives ten stages and an oracle scores each stage from platform-recorded facts. The central silent failure is the run that trains but does not learn (TBDL): loss falls, every signal stays green, and the delivered model is no better than the base. An operator-run acceptance gate catches every such run before payment, and a detector calibrated on known-corrupted runs flags severe corruption mid-run. We ran four frontier agents (Claude Opus 5, GPT-5.6-luna, Gemin
    
[^81]: FinFIRST：面向金融信息检索、溯源与可追溯性的搜索智能体基准测试

    FinFIRST: Benchmarking Search Agents for Financial Information Retrieval, Sourcing and Traceability

    [https://arxiv.org/abs/2609.25192](https://arxiv.org/abs/2609.25192)

    FinFIRST是首个通过原子化评分标准联合评估答案与支持证据的金融搜索智能体基准，包含123个专家任务，可全面评估金融信息检索中的来源选择、时间有效性与可追溯性。

    

    金融搜索对大语言模型（LLM）智能体是一项要求极高的任务，不仅需要正确的最终答案，还需要时间上有效的信息检索、权威来源的选择、实体与时期的对齐、单位与定义的一致性，以及所有结论的可验证证据。现有基准主要只评估最终答案，导致难以定位错误或判断答案是否有充分依据。为填补这一空白，我们提出了FinFIRST（金融信息检索、溯源与可追溯性），这是首个通过原子化评分标准联合评估答案与支持证据的金融基准。FinFIRST包含123个由专家撰写的任务，覆盖渐进式难度谱系，其构建基于真实金融场景的聚合模式，采用18字段分类法、六轴覆盖蓝图、包含138个金融来源的注册表，并由50多位金融专家贡献完成。

    arXiv:2609.25192v1 Announce Type: new  Abstract: Financial search is a highly demanding task for LLM agents, requiring not only a correct final answer but also temporally valid information retrieval, authoritative source selection, entity and period alignment, unit and definition consistency, and verifiable evidence for all conclusions. Existing benchmarks predominantly evaluate only the final answer, making it difficult to localize errors or assess whether an answer is well-founded. To address this gap, we introduce FinFIRST (Financial Information Retrieval, Sourcing and Traceability), the first financial benchmark to jointly evaluate answers and supporting evidence through atomic rubrics. FinFIRST comprises 123 expert-authored tasks spanning a graduated difficulty spectrum, constructed from aggregate patterns of real-world financial scenarios through an 18-field taxonomy, a six-axis coverage blueprint, a registry of 138 financial sources, contributions from over 50 finance experts, a
    
[^82]: 从模式识别器到个性化伴侣：大语言模型在心理健康领域的综述

    From Pattern Recognizers to Personalized Companions: A Survey of Large Language Models in Mental Health

    [https://arxiv.org/abs/2609.25186](https://arxiv.org/abs/2609.25186)

    本综述提出大语言模型在心理健康领域的角色正经历三个日益成熟的演进阶段——从被动的信息工具逐步发展为个性化伴侣，并以此框架系统梳理了碎片化的领域文献、指明了未来研究方向。

    

    arXiv:2609.25186v1 公告类型：交叉发布。摘要：全球心理健康问题的患病率不断上升，加之传统医疗保健长期存在的障碍——如资源有限、成本高昂、病耻感和隐私顾虑——使得对可及且可规模化的心理支持的需求日益迫切。大语言模型（LLMs）作为一种变革性技术应运而生，凭借先进的自然语言理解与生成能力，在推动心理健康支持普及化方面具有巨大潜力。然而，该领域快速扩张且碎片化的研究成果缺乏一个连贯的演进叙事，使人们难以把握当前进展的脉络并明确未来方向。本综述通过围绕一个核心论点组织和分析相关文献来填补这一空白：大语言模型在心理健康领域的角色正经历三个截然不同、且日益成熟的阶段的演进。我们追溯了这一发展轨迹：从第一阶段开始，大语言模型主要充当被动的信息工具……

    arXiv:2609.25186v1 Announce Type: cross  Abstract: The rising global prevalence of mental health conditions, together with longstanding barriers in traditional healthcare, such as limited resources, high cost, stigma, and privacy concerns, has created an urgent need for accessible and scalable support. Large Language Models (LLMs) have emerged as a transformative technology with strong potential to democratize mental health support through advanced natural language understanding and generation. However, the rapidly expanding, fragmented body of work in this area lacks a coherent evolutionary narrative, making it difficult to contextualize current progress and identify future directions. This survey addresses this gap by organizing and analyzing the literature around a central thesis: the role of LLMs in mental health is evolving through three distinct, increasingly sophisticated phases. We trace this trajectory from Phase I, in which LLMs act primarily as passive Information Tools and 
    
[^83]: Qwen-Audio-3.1-Realtime：迈向可靠的智能体语音交互

    Qwen-Audio-3.1-Realtime: Towards Reliable Agentic Voice Interaction

    [https://arxiv.org/abs/2609.25176](https://arxiv.org/abs/2609.25176)

    Qwen-Audio-3.1-Realtime 通过“思考—行动—说话协调”三大模块（结合多教师在线策略蒸馏与基于GRPO的强化学习），将实时语音助手的整体任务成功率从78.4%提升至82.0%，实现了可靠的智能体语音交互。

    

    实时语音助手必须能够对不断演变的请求进行推理、执行操作并遵循对话规则。Qwen-Audio-3.1-Realtime 通过“思考”、“行动”以及“说话与协调”三大机制将这些要求整合在一起。思考模块将 Core-Cocktail 监督微调与多模态、多教师在线策略蒸馏（M²-OPD）相结合，以迁移语言能力并发展原生的音频技能。行动模块利用自进化的可执行环境和多粒度 rollout 进行群体相对策略优化（GRPO），教会模型使用工具、解读反馈并完成任务。说话与协调模块则对齐助手何时、如何以及是否说话或行动。我们在音频推理、多语言理解、工具使用、对话行为、全双工交互和安全性等方面进行了评估。与 Qwen-Audio-3.0-Realtime 相比，3.1 在半双工语音到文本适配任务上将整体任务成功率从 78.4% 提升至 82.0%。

    arXiv:2609.25176v1 Announce Type: cross  Abstract: Real-time voice assistants must reason over evolving requests, execute actions, and follow conversational rules. Qwen-Audio-3.1-Realtime brings these requirements together through Think, Act, and Speak and Coordinate. Think combines Core-Cocktail supervised fine-tuning with Multimodality and Multi-Teacher On-Policy Distillation (M$^{2}$-OPD) to transfer language capabilities and develop native audio skills. Act uses self-evolving executable environments and multi-granularity rollouts for Group Relative Policy Optimization (GRPO), teaching the model to use tools, interpret feedback, and complete tasks. Speak and Coordinate aligns how, when, and whether the assistant speaks or acts. We evaluate audio reasoning, multilingual understanding, tool use, conversational behavior, full-duplex interaction, and safety. Compared with Qwen-Audio-3.0-Realtime, 3.1 raises overall task success from 78.4% to 82.0% on our half-duplex speech-to-text adapt
    
[^84]: 影响不等于失效：询问主张本身，而非代码差异

    Impact Is Not Invalidation: Ask About the Claim, Not the Diff

    [https://arxiv.org/abs/2609.25130](https://arxiv.org/abs/2609.25130)

    编程智能体记忆系统在判断存储主张是否失效时，应针对具体主张提问而非判断代码差异是否保持行为——同一模型面对相同差异，前一种问法的精确度（0.705-0.974）远高于后一种（0.291-0.329）。

    

    面向编程智能体的记忆系统在仓库发生变化时，必须判断其存储的哪些主张已经变为错误。内容锚定方法只要其来源工件发生变化就会使主张失效，这种方式触发得极为频繁。语义等价分类则是询问某个差异是否保持了行为，这是一个关于差异本身而非关于任何已存储主张的问题。我们证明第二种信号失效的原因与模型能力无关：当被问及某个提交是否保持行为时，五个价格跨度达40倍的模型在59%至72%的真实提交上触发判定，且在0.25的基础率下精确度仅为0.291至0.329。而当改为询问某个具体主张是否仍然成立时，同样的模型在同样的差异上精确度达到0.705至0.974。一个对照组将主张文本交给行为保持判断者，仅改变问题本身，精确度仅移动0.010和0.016；而改变提问方式则使精确度移动0.49和0.65。我们还与pytest测试进行了比较。

    arXiv:2609.25130v1 Announce Type: new  Abstract: Memory systems for coding agents must decide, when a repository changes, which of their stored claims have become false. Content anchoring invalidates a claim whenever the artifact it came from changes, which fires constantly. Semantic-equivalence classification asks whether a diff preserves behavior, a question about the diff rather than about any stored claim. We show the second signal fails for a reason unrelated to model capability: asked whether a commit preserves behavior, five models spanning a 40x price range fire on 59-72% of real commits and reach precisions of only 0.291 to 0.329 against a 0.25 base rate. Asked instead whether one specific claim still holds, the same models on the same diffs reach 0.705 to 0.974. A control that hands the behavior-preservation judge the claim text, changing only the question, moves precision by 0.010 and 0.016; changing the question moves it by 0.49 and 0.65. We also compare against pytest-test
    
[^85]: ufakzeka-1：从零构建并评估一个151M参数的土耳其语语言模型

    ufakzeka-1: Building and Evaluating a 151M-Parameter Turkish Language Model from Scratch

    [https://arxiv.org/abs/2609.25081](https://arxiv.org/abs/2609.25081)

    该论文以约286美元的成本从零构建了一个151M参数的土耳其语语言模型，其核心贡献是完整记录了从字节级分词器、三阶段预训练、后训练数据配比到多层次评估体系的构建与测量流程，而非模型能力本身。

    

    我们介绍了ufakzeka-1，一个151M参数（含嵌入层为182M）的仅解码器（decoder-only）土耳其语语言模型，该模型从零开始在135亿个开放许可文本token上进行预训练，并针对聊天场景进行了指令微调，总成本约为286美元（涵盖云GPU、API和笔记本计算时间）。其贡献不在于模型的能力——这种规模的模型只能达到人们对其的预期水平——而在于构建和测量该模型的完整记录：一个每词1.77个token的土耳其语字节级分词器、三阶段预训练计划、开放许可数据与生成数据混合的后训练方案，以及一套完整的评估体系，包括发布门槛、对5,508个对话的规则化检查扫描、评审对话和人工测试，所有提示词均与训练数据隔离，通过数据构建过程中的去污染机制以及每次构建前运行的入库不变量检查脚本来强制执行。我们报告了三项我们认为可推广到其他小模型工作的发现：一个安全门槛……

    arXiv:2609.25081v1 Announce Type: new  Abstract: We describe ufakzeka-1, a 151M-parameter (182M with embeddings) decoder-only Turkish language model pretrained from scratch on 13.5B tokens of openly licensed text and instruction-tuned for chat, at a total cost of about \$286 in cloud GPU, API and notebook time. The contribution is not the model's capability, which is what a model this size can be expected to have, but the record of building and measuring it: a Turkish byte-level tokenizer at 1.77 tokens per word, a three-stage pretraining schedule, a post-training mixture of openly licensed and generated data, and an evaluation battery of release gates, a rule-checked sweep of 5,508 conversations, judged conversations and hand tests, all with prompts held out from the training data, enforced by decontamination inside the data build and by a checked-in invariant script we run before each build. We report three findings that we believe transfer to other small-model efforts: a safety gate
    
[^86]: 理解基于大语言模型的人类行为模拟中的可靠性

    Understanding Reliability in LLM-based Human Behavior Simulation

    [https://arxiv.org/abs/2609.25066](https://arxiv.org/abs/2609.25066)

    该论文提出ReliMap框架，将基于大语言模型的人类行为模拟分解为三个结构化层次，从个体和群体两个层面系统评估可靠性，发现画像条件化能显著减少模型分布偏差但收益递减，且更大的模型受益更多、属性的信息量比数量更重要。

    

    大语言模型（LLM）越来越多地被用于模拟人类的调查回应和行为反应，然而不可靠的模拟可能会误导社会科学的结论。然而，现有的评估主要关注端到端的分数，尚不清楚模拟过程的不同方面如何相互作用以决定可靠性。我们提出了ReliMap，它将基于大语言模型的人类行为模拟分解为三个结构化层次，并在三个配置维度（模型能力、画像完整性和人群覆盖范围）上评估个体层面（R1）和群体层面（R2）的可靠性。通过在四个模拟任务和十一个大语言模型上的实验，我们发现所有模型在没有画像条件化的情况下都表现出显著的分布偏差。画像条件化可以减少这种偏差，但收益递减。更大的模型受益更多，且属性的信息量比数量更重要。至关重要的是，

    arXiv:2609.25066v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to simulate human survey responses and behavioral reactions, yet unreliable simulations can mislead social science conclusions. However, existing evaluations focus on end-to-end scores, leaving it unclear how different aspects of the simulation process interact to determine reliability. We propose ReliMap, which decomposes LLM-based human behavior simulation into three structured layers and evaluates reliability at both the individual level (R1) and population level (R2) across three configuration dimensions: model capacity, profile completeness, and population coverage. Through experiments across four simulation tasks and eleven LLMs, we find that all models exhibit substantial distributional bias without profile conditioning. Profile conditioning reduces this bias with diminishing returns. Larger models benefit more, and attribute informativeness matters more than quantity. Critically,
    
[^87]: ChainDoRA：面向参数高效大语言模型微调的张量链分解权重分解低秩适配

    ChainDoRA: Tensor-Train Factorized Weight-Decomposed Low-Rank Adaptation for Parameter-Efficient LLM Fine-Tuning

    [https://arxiv.org/abs/2609.25058](https://arxiv.org/abs/2609.25058)

    ChainDoRA提出用连通张量链构建方向性低秩因子的权重分解适配框架，通过边界秩与独立TT秩的分离设计，在参数高效的大语言模型微调中取得优于LoRA和DoRA的性能。

    

    参数高效微调（PEFT）在仅更新预训练参数中一小部分的情况下，将大语言模型（LLM）适配到下游任务。低秩适配使用两个可训练的低秩矩阵，而权重分解低秩适配进一步将权重分解为幅度和方向两个部分，但其方向分支仍保留了密集的LoRA式分解。我们提出ChainDoRA，这是一个权重分解适配框架，它从一个连通的张量链构建方向性低秩因子，其中适配器秩构成输入侧与输出侧TT收缩之间的边界秩，而独立的TT秩则控制表示能力与参数成本。在受控的15,119个样本的仅响应适配设置下，基于LLaMA-7B，ChainDoRA在七个常识推理基准上与匹配的LoRA和DoRA基线进行了比较评估。TT秩为16的ChainDoRA取得了……（摘要在此处截断）

    arXiv:2609.25058v1 Announce Type: new  Abstract: Parameter-efficient fine-tuning (PEFT) adapts large language models (LLMs) to downstream tasks while updating only a small fraction of their pretrained parameters. Low-Rank Adaptation (LoRA) uses two trainable low-rank matrices, while Weight-Decomposed Low-Rank Adaptation (DoRA) further separates weight magnitude and direction but retains the dense LoRA-style factorization in its directional branch. We propose ChainDoRA, a weight-decomposed adaptation framework that constructs the directional low-rank factors from a connected Tensor-Train (TT) chain, where the adapter rank forms the boundary rank between input- and output-side TT contractions and an independent TT rank controls representation capacity and parameter cost. Under a controlled 15,119-example response-only adaptation setting with LLaMA-7B, ChainDoRA is evaluated against matched LoRA and DoRA baselines on seven commonsense reasoning benchmarks. ChainDoRA with TT rank 16 achiev
    
[^88]: 基于图推断的反馈驱动词语推演：面向Jotto问题的可扩展框架

    Graph-Based Inference for Feedback-Driven Word Deduction: A Scalable Framework for the Jotto Problem

    [https://arxiv.org/abs/2609.25056](https://arxiv.org/abs/2609.25056)

    本文提出一种基于加权图与迭代约束传播的反馈式词语推演框架，首次将Jotto问题统一扩展至可变长度单词及含重复字母的真实情形。

    

    本文提出了一种基于Jotto问题的反馈式词语推演框架，将问题空间表示为一个加权图，其中所有有效单词对应于节点，边权由两个单词之间共同字母的数量定义。游戏过程被定义为一种迭代约束传播机制，利用反馈不断缩小图中的不相容空间，从而以结构化且可解释的方式实现假设空间的缩减。与现有方法通常仅针对固定长度的无重复字母单词定义问题空间不同，所提出的框架可推广至可变长度的单词（3至8个字母），并能自然地扩展到包含重复字母的情形，首次在统一框架内实现了对真实Jotto问题实例的处理。此外，本文还探讨了所提框架的适用性与求解器的动态行为。

    arXiv:2609.25056v1 Announce Type: new  Abstract: A feedback-based word deduction framework based on the Jotto problem is proposed, and the problem space is represented as a weighted graph where all valid words correspond to nodes, and the edge weight is defined by the number of common letters between the two words. Finally, the gameplay is defined as an iterative constraint propagation mechanism where feedback is used to iteratively narrow the incompatible space of the graph, facilitating the reduction of the hypothesis space in a structured and interpretable manner.   In contrast to existing approaches, where the problem space is typically defined for fixed-length isograms, the proposed framework generalizes to variable-length words (between 3 and 8 letters) and naturally extends to repeated letter cases, facilitating the treatment of realistic Jotto problem instances within a unified framework for the first time. The proposed framework's applicability and solver dynamics are also dis
    
[^89]: ICDAR2026多领域文档多模态推理竞赛

    ICDAR2026 Competition on Multimodal Reasoning over Documents in Multiple Domains

    [https://arxiv.org/abs/2609.25055](https://arxiv.org/abs/2609.25055)

    ICDAR2026竞赛在八个领域的多样化文档上引入了具有挑战性的多模态推理问答任务，结果显示最强系统依赖结构化证据提取、检索、验证与多组件协同编排，而非简单的单次提示。

    

    在本报告中，我们展示了ICDAR2026多领域文档多模态推理竞赛的结果。该竞赛旨在通过视觉问答（VQA）任务推动文档理解研究的发展。在以往DocVQA基准的基础上，本竞赛引入了具有挑战性的推理问题，涵盖八个领域的多样化文档集合，包括商业报告、科学论文、幻灯片、海报、地图、漫画、信息图表和工程图纸。竞赛最终收到了来自8个团队的20份有效提交，涵盖零样本视觉语言模型、OCR与解析器增强的流水线、智能体检索系统、多智能体集成以及微调的多模态模型。结果表明，最强大的系统已超越单次提示的方式，而是依赖于结构化的证据提取、检索、验证以及跨多个组件的协调编排。

    arXiv:2609.25055v1 Announce Type: new  Abstract: In this report we present results of the ICDAR2026 Competition on Multimodal Reasoning over Documents in Multiple Domains. This competition aimed to advance research in document understanding through the task of Visual Question Answering (VQA). Building upon previous DocVQA benchmarks, this competition introduces challenging reasoning questions over a diverse collection of documents spanning eight domains, including business reports, scientific papers, slides, posters, maps, comics, infographics, and engineering drawings. The competition concluded with 20 valid submissions from 8 teams spanning zero-shot VLMs, OCR and parser-augmented pipelines, agentic retrieval systems, multi-agent ensembles, and fine-tuned multimodal models. The results show that the strongest systems move beyond single-pass prompting and instead rely on structured evidence extraction, retrieval, verification, and orchestration across multiple components.
    
[^90]: MoM：记忆的记忆

    MoM: Memory of Memory

    [https://arxiv.org/abs/2609.25054](https://arxiv.org/abs/2609.25054)

    提出MoM（记忆的记忆）框架，使长时程LLM智能体的记忆在写入时即提交当前值，同时将被替换的值保留为溯源信息，并通过类型化溯源图P-Mem实现当前状态的高效访问与历史状态的可恢复。

    

    对于长时程LLM智能体而言，记忆问题不在于曾经记录过什么，而在于什么“当前有效”。大多数设计只能间接回答这一问题：每次交互都被存储，当前状态在查询时通过检索和协调记录来重建，导致过时的值重新进入，同样的冲突被反复争论。在写入时提交当前值可以避免这一问题，但现有的写入时（CRUD）记忆采用覆盖方式，错误的更新无法恢复，先前的状态也随之丢失。我们采用了这一缺失的组合——“到达即提交，同时保留被替换的内容”——并将其形式化为“记忆的记忆”（Memory of Memory, MoM）：记忆不仅追踪内容，还追踪其自身条目的来源、状态和历史。我们将MoM实例化为溯源记忆（Provenant Memory, P-Mem），一种类型化的溯源图，其“活动前沿”为每个已解析的键暴露一个当前值，同时被替换的值作为溯源信息被保留。

    arXiv:2609.25054v1 Announce Type: new  Abstract: For a long-horizon LLM agent, the memory question is not what was once recorded but what \emph{currently holds}. Most designs answer it only indirectly: every interaction is stored, and the present is reconstructed at query time by retrieving and reconciling records, so stale values re-enter and the same conflicts are re-litigated. Committing the current value at write time avoids this, but existing write-time (CRUD) memories overwrite, so a wrong update is unrecoverable and prior state is lost. We take the missing combination---\emph{commit on arrival while retaining what is displaced}---and formalize it as \textsc{Memory of Memory} (MoM): memory tracks not only content but the provenance, status, and history of its own entries. We instantiate MoM as \textsc{Provenant Memory} (P-Mem), a typed provenance graph whose \emph{active frontier} exposes one current value per resolved key while displaced values are retained as provenance; typed 
    
[^91]: LatentPort：超越KV缓存——混合语言模型中循环记忆的跨模型迁移：无需目标前缀重放的4B到9B混合状态交接

    LatentPort: Beyond KV Cache - Cross-Model Transfer of Recurrent Memory in Hybrid Language Models: A 4B-to-9B Hybrid-State Handoff Without Target Prefix Replay

    [https://arxiv.org/abs/2609.25053](https://arxiv.org/abs/2609.25053)

    首次实现了不同规模混合语言模型之间无需重放前缀的持久循环推理状态跨模型交接，通过结合KV缓存迁移与GDN循环及卷积状态的直接复用，显著降低了下一词元预测损失。

    

    一个语言模型能否将其活跃记忆移交给另一个模型，而无需接收方重新阅读上下文？我们演示了在一个架构匹配的Qwen3.5 4B到9B同系列模型对之间进行有效的持久混合状态迁移。据我们所知，这是首次演示在不同规模的混合语言模型之间进行持久循环推理状态的跨模型交接，且无需目标前缀重放。仅迁移注意力KV会留下很大差距；添加门控DeltaNet（GDN）持久状态包后，教师强制下的负对数似然（NLL，即平均下一词元对数损失）降低了0.747 nats/token（95%配对文档自助法置信区间为[0.6921, 0.8047]），在全部64篇PG19文档上均有改善。直接复用循环状态和卷积状态的表现优于所测试的学习型GDN映射，这与持久状态坐标的部分功能兼容性相一致。一项新的组件因子实验选择了翻译后的KV与直接循环和卷积复用相结合的方案。

    arXiv:2609.25053v1 Announce Type: new  Abstract: Can one language model hand its live memory to another without the receiver rereading the context? We demonstrate useful persistent hybrid-state transfer across one architecture-matched Qwen3.5 4B-to-9B sibling pair. To our knowledge, this is the first demonstrated cross-model handoff of persistent recurrent inference state between differently sized hybrid language models without target prefix replay. Translated attention KV alone leaves a large gap; adding the Gated DeltaNet (GDN) persistent-state package lowers teacher-forced negative log-likelihood (NLL), the average next-token log-loss, by 0.747 nats/token (95% paired document bootstrap CI [0.6921, 0.8047]), improving all 64 PG19 documents. Direct recurrent and convolution reuse outperforms the tested learned GDN maps, consistent with partial functional compatibility of persistent-state coordinates. A fresh component factorial selects translated KV with direct recurrent and convoluti
    
[^92]: 自我清洁却依然被捕获：智能体自写存储中误差的一个实测原语，以及分数下降真正度量的是什么

    Self-Cleaning and Captured Anyway: One Measured Primitive for Error in a Store an Agent Writes to Itself, and What a Falling Score Actually Measures

    [https://arxiv.org/abs/2609.25052](https://arxiv.org/abs/2609.25052)

    该论文证明，智能体向自身写入的仅追加存储在无限任期极限下不会发生误差衰减，而是收敛于 (n-1)/n 处的硬性上界，且一个无拟合参数的实测复制函数 γ(φ) 即可在 360 次真实事实运行中的 353 次准确预测误差漂移的方向。

    

    一个将自身结论写入存储、之后再从中检索的智能体，闭合了一个通常被报道为单向污染的循环。将该循环推至对仅追加存储的无限任期极限，会呈现出一幅不同的图景：由于写入从不删除，可达状态空间在 (n-1)/n 处存在一个硬性上界，因此结果是在两条边界之间做出选择，而非衰减。在 f_0 = 0.9 时，两种模式之间的区间在 220 次运行中仅占 3.6%，而均匀分布本应占 20.6%，且在前 15 次运行中该区间严格为空；合并均值仅描述了它所概括运行中的 8.2%，中位数则描述了 68.2%。模型所贡献的一切都由一个无拟合参数的实测原语承载，即复制函数 γ(φ)：在 36 个 Wikidata 事实中，sign(γ̂ - γ_crit)（其中 γ_crit = 1/k，r = 0，w = 1）在 360 次真实事实运行中的 353 次预测了漂移方向（同批次 40 次合成运行中的 39 次）。规模……（摘要在此处被截断）

    arXiv:2609.25052v1 Announce Type: new  Abstract: "An agent that writes its conclusions into a store it later retrieves from closes a loop usually reported as one-way contamination. Taking the loop to the infinite-tenure limit against an append-only store gives a different picture: because writing never deletes, the reachable state space has a hard upper edge at (n-1)/n, so the outcome is a choice between two edges rather than a decay. At f_0 = 0.9 the interval between the two modes holds 3.6% of 220 runs where a uniform spread would put 20.6%, and is strictly empty on the first 15; the pooled mean describes 8.2% of the runs it summarises, the median 68.2%. Everything the model contributes is carried by one measured primitive with no fitted parameter, the copy function \gamma(\phi): on 36 Wikidata facts, sign(\hat{\gamma} - \gamma_{crit}), with \gamma_{crit} = 1/k at r = 0, w = 1, predicts the direction of drift on 353 of 360 real-fact runs (39 of 40 synthetic in the same batch). Scale 
    
[^93]: 大语言模型驱动的免训练位置-属性协同融合：一种面向双源加密POI与土地利用/土地覆盖制图的闭环范式

    LLM-Driven Training-free Location-Attribute Synergic Fusion: A Closed-Loop Paradigm for Dual-source Encrypted POIs and LULC Mapping

    [https://arxiv.org/abs/2609.25051](https://arxiv.org/abs/2609.25051)

    本文首次提出一种由大语言模型驱动的免训练位置-属性协同闭环优化范式，通过迭代反馈联合精化双源加密POI的位置变换与属性匹配，将匹配复杂度从O(N²)降至O(N)，从而有效支持土地利用/土地覆盖制图。

    

    双源加密兴趣点（DSEP）是指来自两个加密坐标系统的兴趣点（POI），其受到位置与属性不确定性相互交织的困扰，包括非线性系统性错位和命名不一致，从而阻碍了土地利用/土地覆盖（LULC）制图。据我们所知，本文首次提出了一种由大语言模型（LLM）驱动的、免训练的位置-属性协同闭环优化范式，用于DSEP融合。该范式通过迭代反馈联合精化位置变换与属性对应关系。属性协同的位置融合采用LLM驱动的属性匹配方法建立DSEP对应关系，将匹配复杂度从O(N^2)降低至O(N)，并在经ISODATA聚类划分的局部子区域内，利用改进的粒子群优化算法精化变换系数。随后，位置协同的属性融合根据更新后的几何信息重新评估属性置信度……（摘要原文在此处截断）

    arXiv:2609.25051v1 Announce Type: new  Abstract: Dual-source encrypted points of interest (DSEP), POIs from two encrypted coordinate systems, suffer from intertwined location and attribute uncertainties, including nonlinear systematic misalignment and naming inconsistency, hindering land-use/land-cover (LULC) mapping. To the best of our knowledge, this paper is the first to propose an LLM-driven, training-free location-attribute synergic closed-loop optimization paradigm for DSEP fusion. The paradigm jointly refines location transformation and attribute correspondences through iterative feedback. Attribute-synergic location fusion uses an LLM-driven attribute matching method to establish DSEP correspondences, reducing matching complexity from O(N^2) to O(N), and refines transformation coefficients using an improved particle swarm optimization algorithm within ISODATA-clustered local subregions. Location-synergic attribute fusion then reassesses attribute confidence from updated geometr
    
[^94]: FrontierMath Erdős（埃尔德什前沿数学）

    FrontierMath Erd\H{o}s

    [https://arxiv.org/abs/2609.25050](https://arxiv.org/abs/2609.25050)

    FME基准包含68个截至2026年8月仍未解决的埃尔德什数学问题，要求AI在Lean证明助手中自主证明或证伪这些猜想，五个受测AI中仅GPT-6 Astra以每题300美元预算取得3%的得分，其余均为0%。

    

    我们提出了FrontierMath Erdős（FME），这是一个包含68个截至2026年8月仍未解决的埃尔德什问题的基准测试。要完成FME中的任务，AI系统必须在证明助手Lean中解决（证明或证伪）这68个猜想之一。我们的68个问题由第二作者从erdosproblems.com上的652个开放问题中，依据其数学趣味性和难度挑选而来。人工智能最近虽然解决了数学中的若干开放问题，但这些演示尚不足以构成对AI能力的系统性研究。FME在相同的固定问题上、以自主方式和相同预算评估每一个AI模型。我们以每个问题300美元的预算评估了五个AI，其中一个（GPT-6 Astra）得分为3%，其余全部为0%。

    arXiv:2609.25050v1 Announce Type: new  Abstract: We introduce FrontierMath Erd\H{o}s (FME), a benchmark of 68 Erd\H{o}s problems that are open as of August 2026. To solve a task in FME, AI systems must resolve (prove or disprove) one of the 68 conjectures in the proof assistant Lean. Our 68 problems were selected by the second author among 652 open problems on erdosproblems.com for their mathematical interest and difficulty. AIs have recently resolved several open problems in mathematics, but these demonstrations fall short of a systematic study of AI capabilities. FME evaluates every AI model on the same fixed problems, autonomously and under the same budget. We evaluated five AIs with a budget of \$300 per problem. One (GPT-6 Astra) scored 3%, and all others scored 0%.
    
[^95]: 通过动态语义路由校准缓解大语言模型的过度拒绝

    Mitigating LLM Over-Refusal via Dynamic Semantic Routing Calibratione

    [https://arxiv.org/abs/2609.25049](https://arxiv.org/abs/2609.25049)

    本文从机制上揭示了LLM过度拒绝源于注意力中“超敏感安全头”引发的高熵路由冲突，并提出无需训练的语义路由校准（SRC）框架，在推理时动态定位并抑制这些安全头，从而有效缓解过度拒绝。

    

    为安全而校准的大语言模型（LLM）常常存在过度拒绝问题，即错误地拒绝无害但涉及安全相关内容的指令。以往研究主要将这一现象归因于静态的表征重叠，在很大程度上忽视了其背后的动态机制。本文从Transformer注意力内部路由冲突的视角出发，对过度拒绝现象进行了机制层面的分析。我们发现，一小部分“超敏感安全头”在“困难安全”类提示上会发生误触发，表现出异常的注意力纠缠，将无害的目标实体强行绑定到拒绝语义上。这会引发严重的、高熵的路由冲突，使目标实体无法获得必要的注意力。为应对这一问题，我们提出了语义路由校准（SRC），这是一个轻量级、无需训练的推理框架。SRC能够精确定位这些超敏感安全头，并在推理阶段对其进行动态抑制。

    arXiv:2609.25049v1 Announce Type: new  Abstract: Large language models (LLMs) aligned for safety often suffer from over-refusal, incorrectly rejecting benign yet safety-related instructions. Prior studies primarily attribute this to static representation overlap, largely overlooking the underlying dynamic mechanisms. In this paper, we present the mechanistic analysis of over-refusal through the lens of internal routing conflicts within transformer attention. We discover that a sparse subset of Hypersensitive Safety Heads misfires on Hard-Safe prompts, exhibiting abnormal attention entanglement that forcefully binds harmless target entities to refusal semantics. This triggers a severe, high-entropy routing conflict that deprives target entities of necessary attention. To counteract this, we propose Semantic Routing Calibration (SRC), a lightweight, training-free inference framework. SRC precisely localizes and dynamically suppresses these hypersensitive safety heads at the inference sta
    
[^96]: 在策略蒸馏中提示广度与采样刷新的交互作用

    Prompt Breadth and Rollout Refresh Interact in On-Policy Distillation

    [https://arxiv.org/abs/2609.25048](https://arxiv.org/abs/2609.25048)

    该研究发现在策略蒸馏中提示广度与 rollout 刷新存在显著交互作用：当学生策略定期刷新时，仅八个提示即可接近 14,080 个提示的效果，而当策略被冻结时增加提示广度反而会降低准确率。

    

    在策略蒸馏需要多少个提示？答案又如何取决于生成其训练响应的学生策略？我们联合研究这两个控制变量：提示广度与 rollout 刷新。一个 3×3 的数学推理实验固定了 14,080 条轨迹和 110 次优化器更新，同时改变提示库规模和生成响应的策略快照数量。在使用十个快照时，八个提示即可达到 24.09% 的平均准确率，接近使用 14,080 个不同提示时的 24.51%。然而，当响应由初始策略冻结生成时，增加提示广度反而使准确率从 21.16% 降至 19.05%；而在每次更新都刷新策略的条件下，增加提示广度则使准确率从 23.61% 提升至 25.57%。由此产生的交互效应为 4.07 个百分点，其基于问题配对的 95% 置信区间为 [2.00, 6.28]。在两个教师模型下的匹配比较揭示了第二个反转：定期刷新的模型在短预算下具有更高的准确率和答案完成率，但冻结……（摘要在此处截断）

    arXiv:2609.25048v1 Announce Type: new  Abstract: How many prompts does on-policy distillation (OPD) need, and how does the answer depend on the student policies that generate its training responses? We study these two controls jointly: prompt breadth and rollout refresh. A 3x3 mathematical-reasoning experiment fixes 14,080 trajectories and 110 optimizer updates while varying the prompt bank and the number of response-generating policy snapshots. With ten snapshots, eight prompts reach 24.09% average accuracy, close to 24.51% for 14,080 distinct prompts. With responses frozen at the initial policy, however, increasing breadth lowers accuracy from 21.16% to 19.05%; under per-update refresh, it raises accuracy from 23.61% to 25.57%. The resulting interaction is 4.07 percentage points, with a 95% question-paired interval of [2.00, 6.28]. Matched comparisons under two teachers reveal a second reversal: the periodic models have higher short-budget accuracy and answer completion, but frozen-r
    
[^97]: AIBuildAI-2.5：通过LLM引导的树搜索实现高效的自主AI模型开发

    AIBuildAI-2.5: Efficient Autonomous AI Model Development Through LLM-Guided Tree Search

    [https://arxiv.org/abs/2609.25047](https://arxiv.org/abs/2609.25047)

    提出AIBuildAI-2.5，通过LLM引导的树搜索实现高效的自主AI模型开发，解决了现有代码搜索智能体在节点选择有效性、训练任务资源调度等方面的效率弱点。

    

    能够自动构建人工智能（AI）模型的自主智能体可以拓宽科学与工程领域对AI技术的使用。这类智能体的一种主流方法将模型构建视为代码搜索问题，并通过树搜索来求解：每个节点是一个候选程序，树通过从父程序生成子程序而不断生长；这些智能体在真实基准测试上的能力已接近经验丰富的AI工程师。然而，这些智能体在效率方面存在三个尚未完全解决的弱点。首先，在现实的预算内只能执行少量候选程序，因此依据已执行奖励对节点进行排序的搜索规则（如蒙特卡洛式树搜索）只能依赖稀少且含噪声的评分，导致选择下一个待探索节点的效果欠佳。其次，缺乏资源感知的策略来调度训练任务，这会降低硬件利用率和训练效率。第三，每个智能体……（原文摘要在此处截断）

    arXiv:2609.25047v1 Announce Type: new  Abstract: Autonomous agents that automatically build artificial intelligence (AI) models could broaden access to AI across science and engineering. A popular line of such agents frames model building as a code search problem and solves it by tree search, in which each node is a candidate program and the tree grows by generating a child program from a parent, and these agents now approach the capability of experienced AI engineers on realistic benchmarks. However, these agents have three weaknesses in efficiency that have not been fully addressed. First, only a small number of candidates can be executed within a realistic budget, so search rules that rank nodes by executed rewards, such as Monte Carlo-style tree search, rely on few and noisy scores and select the next node to explore less effectively. Second, no resource-aware strategy is used to schedule training jobs, which can lower hardware utilization and training efficiency. Third, every agen
    
[^98]: Peerify：同行评审论断验证的基准测试

    Peerify: Benchmarking Peer-Review Claim Verification

    [https://arxiv.org/abs/2609.25046](https://arxiv.org/abs/2609.25046)

    提出了Peerify流水线，可将同行评审意见分解为原子论断并通过检索稿件证据自动验证其是否得到论文支持，同时构建了基于NeurIPS 2024和ICLR 2024真实评审的800条论断基准数据集。

    

    同行评审在学术出版中发挥着核心作用，然而验证审稿人的论断是否得到稿件证据的支持，在很大程度上仍然是一个依赖人工且耗时的过程。我们提出了Peerify，一个基于稿件内容的同行评审论断验证流水线。给定一篇稿件和一条评审意见，Peerify流水线将评审意见分解为原子论断，检索稿件中的相关证据，并判断每条论断是否得到论文的支持。为了支持该流水线的开发与评估，我们构建了一个包含800条论断的基准数据集，这些论断来自NeurIPS 2024和ICLR 2024收集的真实同行评审交互，其中包括一个300条论断的人工标注子集，用于审计自动监督的质量。我们在Peerify流水线中评估了最先进的语言模型和检索策略，并与蕴含基线方法进行了比较。我们的结果证明了以检索为中心的验证方法的重要性。

    arXiv:2609.25046v1 Announce Type: new  Abstract: Peer review plays a central role in scholarly publishing, yet verifying whether reviewer claims are supported by manuscript evidence remains a largely manual and time-consuming process. We present Peerify, a pipeline for manuscript-grounded verification of peer-review claims. Given a manuscript and a review comment, the Peerify pipeline decomposes reviews into atomic claims, retrieves relevant manuscript evidence, and determines whether each claim is supported by the paper. To support the development and evaluation of the pipeline, we construct a benchmark of 800 claims derived from authentic peer-review interactions collected from NeurIPS 2024 and ICLR 2024, including a 300-claim hand-labeled subset used to audit the automated supervision. We evaluate state-of-the-art language models and retrieval strategies within the Peerify pipeline, together with entailment baselines. Our results demonstrate the importance of retrieval-centered veri
    
[^99]: 从语调到轨迹：连续情感与货币政策沟通的形态

    From Tone to Trajectory: Continuous Sentiment and the Shape of Monetary Policy Communication

    [https://arxiv.org/abs/2609.25034](https://arxiv.org/abs/2609.25034)

    该论文创新性地为央行新闻发布会构建了“情感弧线”，发现情感的形态与排序（而非平均语调）能够稳健预测利率决策，并影响专业预测者的通胀预期更新与分歧程度。

    

    央行新闻发布会不仅仅是信息发布——它们是结构化的叙事。我们研究声明中情感的形态（而不仅仅是其平均语调）是否携带与政策相关的信号。我们沿着三个维度——货币政策立场、经济前景和不确定性——为欧洲央行（ECB）和美联储（Fed）的新闻发布会构建情感弧线，并评估其对政策利率变化、通胀预期以及预测者分歧的预测能力。研究结果表明，在两家机构中，弧线形态在超越基于词典的基准方法的情况下，能够稳健地预测利率决策——携带政策信号的并非声明在平均意义上听起来是鹰派还是对经济乐观，而是这些情感在声明中如何被排序和强调。弧线特征还影响专业预测者如何更新通胀预期以及他们之间分歧的程度，这指向了一种接收方效应（原文在此截断）。

    arXiv:2609.25034v1 Announce Type: new  Abstract: Central bank press conferences are not merely information releases --- they are structured narratives. We study whether the shape of sentiment within a statement, not just its average tone, carries policy-relevant signals. Constructing sentiment arcs for ECB and Fed press conferences along three dimensions --- monetary stance, economic outlook, and uncertainty --- we assess their predictive content for policy rate changes, inflation expectations, and forecaster disagreement. Our findings show that arc shape robustly predicts rate decisions beyond lexicon-based benchmarks at both institutions --- it is not merely whether a statement sounds hawkish or economically optimistic on average, but how these sentiments are sequenced and emphasized across the statement, that carries the policy signal. Arc features also shape how professional forecasters update inflation expectations and how much they disagree, pointing to a receiver-side effect dis
    
[^100]: 面向QMSum高效查询聚焦会议摘要的检索片段训练方法

    Retrieved-Span Training for Efficient Query-Focused Meeting Summarization on QMSum

    [https://arxiv.org/abs/2609.25028](https://arxiv.org/abs/2609.25028)

    在检索片段机制上微调406M小模型，能够以约三分之一的参数量和不到一半的峰值推理内存，在QMSum查询聚焦会议摘要任务上取得与1.2B模型统计上相当的ROUGE-1性能。

    

    QMSum数据集未提供统一的评分工具，导致查询聚焦式会议摘要的结果难以相互比较。我们在同一实现下对15个系统进行了重新评分或生成。通过一个统一的推理接口，一个已发布的406M参数Fusion-in-Decoder专用模型在从截断长输入切换到2000词检索片段时损失了6.30的ROUGE-1分数；而在这类片段机制上进行微调可以挽回这一损失。在测试集上，该模型取得36.33的ROUGE-1，而我们的1.2B参数系统为35.41；两者差异的会议聚类95%置信区间为[-0.27, +2.22]，因此QMSum在统计上无法区分二者。较小的系统仅使用约三分之一的总参数量和不到一半的峰值推理内存。在固定的1.2B基础模型内，片段机制微调可带来5.29的提升[+4.02, +6.56]，而用2000个检索词替换前4500个转录词在测试集上带来1.55的提升、在验证集上带来0.29的提升。此外，在统一的简洁提示和参考重叠评分器下，一个已发布的406M sp...（原文摘要在此处截断）

    arXiv:2609.25028v1 Announce Type: new  Abstract: QMSum provides no scorer, making query-focused meeting summarization results difficult to compare. We rescore or generate 15 systems under one implementation. Through a common inference port, a released 406M Fusion-in-Decoder specialist loses 6.30 ROUGE-1 when moved from capped long input to 2,000-word retrieved spans. Fine-tuning it on this span regime recovers the loss. On test it scores 36.33 ROUGE-1 versus 35.41 for our 1.2B system; the meeting-cluster 95% interval for the difference is [-0.27, +2.22], so QMSum does not statistically separate them. The smaller system uses about one-third as many total parameters and less than half the peak inference memory. Within the fixed 1.2B base, span-regime fine-tuning adds 5.29 [+4.02, +6.56], while replacing the first 4,500 transcript words with 2,000 retrieved words adds 1.55 on test and 0.29 on validation. Separately, under one concise prompt and reference-overlap scorer, a released 406M sp
    
[^101]: “作为一名语言模型……”：聊天模板可切换大语言模型的自我指涉语气，且激活转向能够重现这一现象

    "As a Language Model...": Chat Template Switches LLM Self-Referential Voice and Activation Steering Reproduces It

    [https://arxiv.org/abs/2609.25021](https://arxiv.org/abs/2609.25021)

    本研究揭示聊天模板像一个开关，能够切换大语言模型在“免责声明式”与“体验式”两种自我指涉语气之间的表达，并通过在模型激活空间中发现并操控特定方向，成功再现了这一行为的增减调控。

    

    大语言模型（LLM）在被问及与自身相关的问题时，往往会添加诸如“我只是一个AI”之类的免责声明。这类回答中的自我陈述常被用于关于AI安全或模型自我认知的讨论中，然而其背后的驱动机制尚不清楚。模型是在告诉我们关于它们自身的情况，还是在描述它们被部署的方式？在本研究中，我们展示了聊天模板就像一个开关——当聊天模板存在时，它会在8个参数规模最大至9B的主流开源指令模型中调高这种免责声明语气，同时调低诸如“我感觉”之类的体验式语气；反之，当聊天模板不存在时，它会调低免责声明语气并调高体验式语气。在3个模型的内部激活中，我们发现了一个能够引导该行为的方向。在模型激活空间中移除该方向会降低免责声明语气，添加该方向则会增强它，而相同维度的随机方向几乎没有影响。

    arXiv:2609.25021v1 Announce Type: cross  Abstract: Large Language Models (LLMs) tend to add disclaimers like "I'm just an AI" when asked about something related to themselves. The self-reports from such responses are used in debates about AI safety or self-knowledge of the models, yet what drives them is not well understood. Are the models telling us about themselves or rather how they are deployed? In this work, we show that the chat template works like a switch - when present, it turns this disclaimer voice up and experiential voice like "I feel" down, across 8 popular open-source instruct models up to 9B parameters in size. And conversely when the chat template is not present, it turns the disclaimer voice down and experiential voice up. Inside the activations of 3 models, we find a direction that steers this behavior. Removing the direction in the model's activation space turns disclaimer voice down and adding it turns it up, while a random direction of the same size has little eff
    
[^102]: 一种测量梵语文学语义变化的计算方法

    A Computational Approach to Measuring Semantic Change in Sanskrit Literature

    [https://arxiv.org/abs/2609.25012](https://arxiv.org/abs/2609.25012)

    该研究首次将历时词嵌入方法系统应用于梵语这一低资源古代语言，通过构建270万词元的跨时期语料库和神经连声切分技术成功追踪语义变化，21个可测试变化中有19个与文献学证据方向一致。

    

    历时词嵌入已成为追踪语义变化的现代标准方法，但它们主要在现代、高资源且分词良好的语言上得到验证。本文测试该范式是否可以迁移到梵语——一种古老的低资源语言，其语音连声（sandhi）、形态屈折、复合词构词和多义性构成了独特的挑战。作者构建了一个跨越四个经典时期、包含270万词元的语料库，使用神经字节级连声切分器和词元还原器恢复词边界，并在多种配置下训练各时期的词嵌入。为了评估该系统，作者从历史学术研究中整理了一个验证集，并通过锚点位移进行方向性测试。在21个可测试的语义变化中，有19个朝着文献学证实的方向移动（符号检验，p=0.00011）。作者进一步展示了该语言特性所要求的模型配置，并讨论了未来改进的机会。

    arXiv:2609.25012v1 Announce Type: new  Abstract: Diachronic word embeddings have become the modern standard for tracking semantic change, yet they have been largely validated on modern, high-resource, and well-segmented languages. This paper tests whether the paradigm transfers to Sanskrit, an ancient, low-resource language whose phonological fusion (sandhi), morphological inflection, compounding, and polysemy pose a unique challenge. I assemble a 2.7M-token corpus spanning four canonical periods, recover word boundaries with a neural byte-level sandhi splitter and lemmatizer, and train per-period embeddings across configurations. To evaluate the system, I curate a validation set from historical scholarship and test recovery directionally with anchor displacement. Of 21 testable shifts, 19 move in the philologically attested direction (sign test, p=0.00011). I further show which configuration the language forces and comment on opportunities for improvement.
    
[^103]: 合成人格能否预测真实受众的反应？一项“无人格”基线胜过基于人格的文案模拟的从模拟到真实（Sim-to-Real）研究

    Do Synthetic Personas Predict Real Audience Response? A Sim-to-Real Study Where a No-Persona Baseline Beats Persona-Based Copy Simulation

    [https://arxiv.org/abs/2609.25010](https://arxiv.org/abs/2609.25010)

    该研究发现，基于真实受众画像构建的多人格LLM模拟在预测真实受众点击行为上并不优于简单的无人格零样本基线，且由于大多数A/B测试本身缺乏统计上可区分的赢家，模拟效度的评估受到真值可靠性的根本制约。

    

    营销人员越来越多地使用大语言模型（LLM）作为“合成人格”，以便在文案发布之前预测受众的反应，这一做法受到“基于画像条件化的LLM能够模拟人类样本”的证据所鼓舞。但这种预测对于真实行为是否真的有效——人格机制是否真的有所帮助？我们提出了一项从模拟到真实的效度研究，以Upworthy研究档案——数千个在共享真实流量上进行的标题A/B测试并测量了实际点击率——作为留出的真值。我们将一个基于真实受众人口统计特征构建的十人格小组，与一个无人格的零样本基线进行比较，后者只是简单地询问模型一个典型读者点击的可能性有多大。有两个发现尤为突出。第一，真值的可靠性是约束性瓶颈：大多数A/B测试没有统计学上可区分的赢家，因此效度只能在可靠子集（n=399）上进行测量。第二，与人格化假设相反，无人格基线在预测效度上优于基于人格的文案模拟。

    arXiv:2609.25010v1 Announce Type: cross  Abstract: Marketers increasingly use large language models (LLMs) as "synthetic personas" to predict how an audience will react to a piece of copy before it ships, encouraged by evidence that profile-conditioned LLMs mimic human samples. But is that prediction actually valid against real behaviour - and does the persona machinery help? We present a sim-to-real validity study using the Upworthy Research Archive - thousands of headline A/B tests on shared real traffic, with measured click-through - as held-out ground truth. We compare a ten-persona panel, grounded in the real audience's demographics, against a no-persona zero-shot baseline that simply asks the model how likely a typical reader is to click. Two findings stand out. First, ground-truth reliability is the binding constraint: most A/B tests have no statistically distinguishable winner, so validity can only be measured on the reliable subset (n = 399). Second, and counter to the persona
    
[^104]: 数量相同，答案不同：语言模型中的数值表示不变性

    Same Quantity, Different Answer: Numerical Representation Invariance in Language Models

    [https://arxiv.org/abs/2609.25009](https://arxiv.org/abs/2609.25009)

    该论文通过五类数值恒等变换的大规模评测发现，语言模型对同一数量的不同表示形式（小数、分数、百分比、数字词、科学计数法、单位换算）缺乏不变性，且揭示了评估器解析器的限制可能被误判为模型推理失败。

    

    数值上等价的应用题，无论其中的数量以小数、分数、百分比、数字词、科学计数法还是精确换算的单位表示，都应得出相同的标准答案。我们生成了3,600道精确有理数题目和8,600条提示词，涵盖五类保持数值恒等的变换，并评估了五个开源权重模型系统。在经过固定的语法审计（在不使用大模型评判的情况下规范化常见答案形式）之后，标准准确率为0.969-0.996，但轨道正确率降至0.848-0.981，轨道不变性降至0.851-0.981；“不变但错误”的轨道最多只占0.003。大部分由严格解析器导致的整体性能下降，是因为乘法形式的科学计数法超出了所实现的数字语法范围，这说明评估器接口可能伪装成推理失败。此外还存在一种独特的语义病理现象：Mistral Small 4在单位换算输入上的得分仅为0.699，产生了265个错误。

    arXiv:2609.25009v1 Announce Type: new  Abstract: Numerically equivalent word problems should yield the same canonical answer whether a quantity is written as a decimal, fraction, percentage, number word, scientific notation, or an exactly converted unit. We generate 3,600 exact-rational problems and 8,600 prompts spanning five identity-preserving transformation families, and evaluate five open-weight systems. After a fixed syntax audit that normalizes common answer forms without an LLM judge, canonical accuracy is 0.969-0.996, but orbit correctness falls to 0.848-0.981 and orbit invariance to 0.851-0.981; invariant-but-wrong orbits account for at most 0.003. Most of the broad strict-parser collapse arises because multiplication-form scientific notation lies outside the implemented number grammar, illustrating how evaluator interfaces can masquerade as reasoning failures. A distinct semantic pathology remains: Mistral Small 4 scores 0.699 on unit-converted inputs and produces 265 errors
    
[^105]: 用 Rust 端到端训练语言模型：一份经验报告

    Training a Language Model End-to-End in Rust: An Experience Report

    [https://arxiv.org/abs/2609.25008](https://arxiv.org/abs/2609.25008)

    作者独自用 Rust 仅花 164 美元 GPU 成本端到端预训练了语言模型，并系统记录了 Candle 和 Burn 两大 Rust 机器学习框架作为训练后端时的八种静默失败缺陷，提出了以“梯度流仲裁器”为核心的验证方法来捕获常规损失曲线检查无法发现的问题。

    

    我独自一人在 Rust 中端到端地预训练了一个语言模型——没有团队，不使用 PyTorch，训练路径中没有 Python——仅花费 164 美元的租用 GPU 时间。我将此报告为一项成就，而非推荐：更有价值的贡献是对 2026 年两大主流 Rust 机器学习框架 Candle 和 Burn 作为训练（而非推理）后端的实测失败分类。我记录了五个 Candle 缺陷，包括融合内核会静默地不产生梯度的问题，以及三个 Burn 缺陷，包括反向传播速度仅为理论 GPU 吞吐量约 3% 的问题，以及内核融合路径在数十亿参数规模的训练中途发生段错误的问题。每一个缺陷都能通过常规的损失曲线检查，没有一个会自行显现。我描述了捕捉到其中六个此类静默失败的验证纪律，其核心是一个梯度流仲裁器：一种运行一次前向/反向传播、并断言每个可训练参数都接收到有限且非零梯度的测试，且可推广……

    arXiv:2609.25008v1 Announce Type: new  Abstract: I pretrained a language model end-to-end in Rust - alone, with no team, no PyTorch, and no Python in the training path - for $164 in rented GPU time. I report that as an achievement, not a recommendation: the more useful contribution is a measured failure taxonomy of the two leading Rust ML frameworks, Candle and Burn, as training (not inference) backends in 2026. I document five Candle defects, including fused kernels that silently produce no gradient, and three Burn defects, including a backward pass at roughly 3% of theoretical GPU throughput and a kernel-fusion path that segfaults mid-training at multi-billion-parameter scale. Every one passed ordinary loss-curve inspection; none announced itself. I describe the verification discipline that caught six such silent failures, centered on a gradient-flow arbiter: a test that runs one forward/backward pass and asserts every trainable parameter receives a finite, nonzero gradient, generali
    
[^106]: 超越短片段：利用向量档案扩展说话人嵌入

    Beyond Short Segments : Expanding Speaker Embeddings with Vector Archives

    [https://arxiv.org/abs/2609.25007](https://arxiv.org/abs/2609.25007)

    提出VAM-ECAPA系统，通过基于Transformer的向量档案映射模块，将短语音的稀疏特征映射到可学习的典型说话人特征档案中，在1秒测试片段上取得8.334%的EER，相对错误率降低54.8%。

    

    由于说话人特定信息不足，最先进的说话人验证（SV）系统在短语音上的性能会严重下降。为应对这一关键挑战，我们提出了向量档案映射ECAPA（VAM-ECAPA），这是一种旨在增强短时长语音特征提取的新型系统。我们系统的核心是基于Transformer的统计池化向量档案映射（TVAMSP）模块，该模块通过将信息匮乏的特征映射到可学习的典型说话人特征向量档案上来丰富这些特征。通过将TVAMSP模块集成到强大的WavLM+ECAPA-TDNN基线中，我们的系统学会将短片段的稀疏特征映射为鲁棒且具有判别力的说话人表示。在VoxCeleb1基准上的实验表明，我们提出的VAM-ECAPA在1秒测试片段上取得了极具竞争力的8.334%等错误率（EER），相比对比基线实现了54.8%的相对错误率降低。

    arXiv:2609.25007v1 Announce Type: cross  Abstract: The performance of state-of-the-art speaker verification (SV) systems severely degrades on short utterances due to insufficient speaker-specific information. To address this critical challenge, we propose the Vector Archive Mapping ECAPA (VAM-ECAPA), a novel system designed to enhance feature extraction from short-duration speech. The core of our system is the Transformer-based Vector Archive Mapping with Statistical Pooling (TVAMSP) module, which enriches information-scarce features by mapping them against a learnable Vector Archive of canonical speaker traits. By integrating the TVAMSP module into a strong WavLM+ECAPA-TDNN baseline, our system learns to map sparse features from short segments into robust, discriminative speaker representations. Experiments on the VoxCeleb1 benchmark show that our proposed VAM-ECAPA achieves a highly competitive EER of 8.334% on 1-second test segments, a 54.8% relative error reduction compared to a co
    
[^107]: 99%的准确率衡量了什么？对一个广泛使用的假新闻语料库中捷径学习的可复现审计

    What Does 99% Accuracy Measure? A Reproducible Audit of Shortcut Learning in a Widely Used Fake News Corpus

    [https://arxiv.org/abs/2609.25006](https://arxiv.org/abs/2609.25006)

    本文对广泛使用的ISOT/Kaggle假新闻语料库进行了可复现审计，发现其98%以上的高准确率主要来自元数据、来源标签和重复文档等数据泄漏与捷径学习，而非真正的虚假新闻检测能力，揭示了该基准的有效性存在严重问题。

    

    在ISOT/Kaggle“虚假与真实新闻”语料库上训练的文本分类器通常报告超过0.98的准确率和F1值，这一性能水平与评估新闻真实性的实际难度相比显得不合常理。我们将透明的TF-IDF与线性分类器流水线作为测量工具，沿三个数据泄漏通道和两个分布偏移协议对该语料库进行审计，并公开所有代码和衍生数据。首先，该基准部分退化：仅使用主题元数据字段（完全丢弃文章文本）的分类器即可达到F1 = 1.000，因为两个类别的主题集合完全不相交。其次，移除全部三个泄漏通道——元数据、出现在99.2%真实文章中的新闻专线来源标签、以及污染了朴素测试集划分19.4%的6,251个重复文档——仅使F1下降1.21个百分点（从0.9935降至0.9814）；残余信号是弥散的编辑风格而非少数暴露性词汇，因为删除1,0……（摘要原文在此处被截断）

    arXiv:2609.25006v1 Announce Type: new  Abstract: Text classifiers trained on the ISOT/Kaggle "Fake and Real News" corpus routinely report accuracy and F1 above 0.98, a level of performance that sits uneasily beside the difficulty of assessing veracity. Using a transparent TF-IDF and linear-classifier pipeline as a measurement instrument, we audit the corpus along three leakage channels and two distribution-shift protocols, releasing all code and derived numbers. First, the benchmark is partly degenerate: a classifier given only the subject metadata field, with the article text discarded, attains F1 = 1.000, since the two classes have disjoint subjects. Second, removing all three leakage channels, metadata, a newswire source tag present in 99.2% of real articles, and 6,251 duplicate documents contaminating 19.4% of a naive test split, lowers F1 by only 1.21 points (0.9935 to 0.9814); the residual signal is diffuse editorial style rather than a few giveaway tokens, since deleting the 1,0
    
[^108]: DolphinBench：绘制智能体记忆的帕累托前沿

    DolphinBench: Mapping the Pareto Frontier of Agent Memory

    [https://arxiv.org/abs/2609.24971](https://arxiv.org/abs/2609.24971)

    DolphinBench是一个通过智能体实际任务完成情况（而非对话式问答）直接评估长期记忆的基准，包含三个各约50万token历史记录的知识工作角色画像、每个角色200个经有无历史对照验证的任务，并强制要求报告成本，以刻画记忆性能与成本之间的帕累托前沿。

    

    如今的智能体常常采取依赖于长期记忆和随时间推移的上下文回忆的现实世界行动。然而，目前大多数记忆基准都是为对话式问答格式构建的，其中问题本身就会提示需要检索某些事实，甚至往往暗示是哪一个事实。此外，基准测试很少对提交内容提出准确性之外的要求，这使得记忆系统可以通过不合理的成本/时间权衡来换取更高的分数。我们提出了DolphinBench，这是一个通过智能体的任务完成情况直接评估记忆的基准。DolphinBench包含三个知识工作型角色画像，每个角色画像拥有约50万token的用户消息，并在依赖该历史信息的任务上评估智能体。我们通过让智能体在有相关历史和无相关历史的情况下分别运行，验证了每个角色的全部200个任务，要求在有历史的情况下成功而在无历史的情况下失败。最后，我们要求所有评估报告总成本

    arXiv:2609.24971v1 Announce Type: new  Abstract: Agents today often take real-world actions that depend on long-term memory and context recall over time. However, most current memory benchmarks are built for a conversational question-answer format, where the question itself signals that some fact must be retrieved, and often which one. Moreover, benchmarks rarely require anything beyond accuracy from submissions, allowing memory systems to make unreasonable cost/time tradeoffs to achieve higher scores.   We present DolphinBench, a benchmark that evaluates memory directly through an agent's task completion. DolphinBench includes three knowledge-work personas with roughly 500k tokens of user messages per persona and evaluates agents on tasks that depend on information from that history. We verify all 200 tasks per persona by running an agent with and without the relevant history, requiring success with it and failure without it.   Finally, we require all evaluations to report total cost 
    
[^109]: Re:CAP——审计生产级RAG流水线中的检索覆盖率

    Re:CAP - Auditing Retrieval Coverage in Production RAG Pipelines

    [https://arxiv.org/abs/2609.24122](https://arxiv.org/abs/2609.24122)

    Re:CAP提出了一种无参考的迭代探测审计方法，通过为可能缺失的主题生成探测性问题并利用LLM评判筛选，来发现生产级RAG系统中检索遗漏的文档，从而审计检索覆盖率，在四个基准上恢复了BM25 top-500无法召回的9-29%金标准标注。

    

    检索增强生成（RAG）在生产环境中难以监控：对于实时重新索引的非平稳数百万段落语料库，不存在穷尽性的相关性标注。因此，检索质量通常研究不足，并且往往让位于面向生成的指标。在这项工作中，我们提出通过探测缺失文档的证据来审计检索覆盖率，而不是枚举每一个相关文档。我们的方法Re:CAP（通过迭代探测进行检索覆盖审计）是一个无参考的审计循环，应用于已部署RAG流水线的初始答案和检索到的上下文：它识别已覆盖的主题，为可能缺失的主题生成探测性问题，检索候选文档，并应用LLM作为评判者，仅保留那些引入了先前未检索到的信息的文档。在四个公开基准测试中，Re:CAP能够恢复扁平BM25 top-500无法恢复的9-29%的金标准标注。

    arXiv:2609.24122v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is hard to monitor in production: exhaustive relevance labels do not exist for non-stationary multi-million-passage corpora that re-index in real time. As a result, retrieval quality is generally understudied and often deprioritised in favour of generation-oriented metrics. In this work, we propose auditing retrieval coverage by probing for evidence of missing documents rather than enumerating every relevant one. Our method Re:CAP (REtrieval Coverage Audit by iterative Probing) is a reference-free audit loop applied to a deployed RAG pipeline's initial answer and retrieved context: it identifies the topics already covered, generates probing questions for plausibly missing topics, retrieves candidate documents, and applies an LLM-as-judge to retain only those that introduce previously-unretrieved information. On four public benchmarks, Re:CAP recovers 9-29% of gold labels that flat BM25 top-500 cannot 
    
[^110]: 人类对齐的模型是人类行为的模型吗？偏好对齐中的图灵测试鸿沟

    Are Human-Aligned Models Models of Humans? A Turing-Test Gap in Preference Alignment

    [https://arxiv.org/abs/2609.23640](https://arxiv.org/abs/2609.23640)

    本文提出“图灵测试鸿沟”这一概念，证明即使偏好和回应数据完全来自人类，偏好对齐仍会使模型行为偏离人类自身的回应分布，从而将“类人性”确立为模型对齐中一个需要独立考虑的维度。

    

    人类反馈对齐已使语言模型成为有用的助手，并通常被描述为使模型与人类对齐。然而，人们偏好的AI回应未必是他们自己会给出的回应。我们区分了与人类偏好的对齐和与人类行为的对齐，并表明即使偏好和回应完全来自人类，与人类偏好的对齐仍可能使模型行为变得更不像人类。我们将这一现象称为“图灵测试鸿沟”。我们证明，偏好对齐仅在一个严格条件下才能保持人类回应分布，且没有发现一致证据表明真实的人类偏好满足该条件。实证结果显示，人类回应似然的损失随偏好权重的强度增加而增大，且与权重方向无关，该鸿沟在标准DPO方法下同样存在。这些结果将“类人性”确立为对齐的一个明确维度，而非...

    arXiv:2609.23640v1 Announce Type: new  Abstract: Human-feedback alignment has made language models useful assistants and is commonly described as aligning them with humans. However, the responses people prefer from an AI need not be the responses they themselves would give. We distinguish alignment with human preferences from alignment with human behavior, and show that alignment with human preferences can make model behavior less human-like even when both preferences and responses come entirely from humans. We call this the Turing-test gap. We show that preference alignment preserves the human response distribution only under a restrictive condition, and find no consistent evidence that real human preferences satisfy it. Empirically, the loss of human-response likelihood increases with the strength of preference weighting, regardless of its direction, and the gap also appears under standard DPO. These results establish human-likeness as an explicit dimension of alignment rather than s
    
[^111]: RPMem：为LLM智能体学习跨会话的长期循环参数化记忆

    RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents

    [https://arxiv.org/abs/2609.23466](https://arxiv.org/abs/2609.23466)

    RPMem提出了一种两阶段架构，将LLM智能体的每个会话编译为模型无关的潜在记忆，经任务训练的循环门控整合后映射为LoRA参数，从而实现跨会话的长期参数化记忆演化，并在更换骨干模型时保持记忆可迁移。

    

    长时间运行的LLM智能体需要能够跨会话持续存在并不断演化的记忆。基于文本的记忆在每次查询时都需要检索和重建过去的交互，随着历史记录的增长，长程性能越来越依赖于检索质量和上下文推理能力。参数化记忆将经验直接编码到模型计算中，但现有方法对跨会话记忆演化的支持有限，且其与特定骨干模型的耦合进一步限制了模型替换后的记忆复用。我们提出了RPMem，这是一种两阶段架构，通过前向计算将每个会话编译为与模型无关的潜在记忆，并通过任务训练的循环门控选择性地将其与保留的记忆进行整合。整合后的记忆随后被映射为特定于骨干模型的低秩适配（LoRA）参数，使得编码能力在骨干模型被替换时仍可迁移。在三个（数据集/任务上）的评估……

    arXiv:2609.23466v1 Announce Type: new  Abstract: Long-running LLM agents require memory that persists and evolves across sessions. Text-based memory retrieves and reconstructs past interactions at every query, making long-horizon performance increasingly dependent on retrieval quality and contextual reasoning as histories grow. Parametric memory encodes experience directly into model computation, but existing approaches provide limited support for cross-session memory evolution. Their coupling to a specific backbone further restricts memory reuse after model replacement. We introduce RPMem, a two-stage architecture that compiles each session into a model-independent latent memory through forward computation and selectively integrates it with retained memory via a task-trained recurrent gate. The consolidated memory is then mapped to backbone-specific low-rank adaptation (LoRA) parameters, allowing the encoding capability to transfer when the backbone is replaced. Evaluation across thre
    
[^112]: Apollo Restore：一个针对古希腊语历史文本优化的基础大语言模型，专用于以“中间填空”方式修复古希腊文本

    Apollo Restore: A Foundation LLM for Historical Greek Optimized for Fill-in-the-Middle Restoration of Ancient Greek Texts

    [https://arxiv.org/abs/2609.22455](https://arxiv.org/abs/2609.22455)

    Apollo Restore 是首个面向历史希腊语（乃至任何古代地中海语言）的 240 亿参数基础大语言模型，通过“中间填空”目标微调，能够在不知缺失文本长度的情况下修复残缺古希腊文本，并在长度平衡的评估指标下大幅超越已发表的最强模型。

    

    我们提出了 Apollo Restore，一个拥有 240 亿参数的大语言模型，用于修复残缺古希腊文本中的缺损（即物理性空缺）。该模型从 Mistral Small 微调而来，采用“中间填空”训练目标，无需预先知晓缺失片段的长度即可重建缺失内容。据我们所知，这是首个针对历史希腊语的大规模解码器模型，也是首个针对任何古代地中海语言的大规模解码器模型。按照先前工作的评估方式，在最多十个字符的短缺损上，Apollo Restore 对文献纸草、文学纸草和石刻铭文缺损分别有 80.6%/54.6%/61.0% 的情况将正确的修复结果排在前二十个候选之中，超过已发表最强模型 1.6 倍/2.6 倍/1.4 倍。然而，先前的评估协议因偏向极短缺损而夸大了分数；在长度平衡的指标下，Apollo Restore 相对于已发表最强模型的优势……（摘要在此处被截断）

    arXiv:2609.22455v1 Announce Type: new  Abstract: We present Apollo Restore, a 24-billion-parameter large language model for restoring lacunae---physical gaps---in fragmentary Ancient Greek texts. Fine-tuned from Mistral Small with a fill-in-the-middle objective, Apollo Restore reconstructs missing spans without requiring oracle knowledge of their length. To our knowledge, it is the first large-scale decoder model for historical Greek, and the first for any ancient Mediterranean language. Evaluated as in prior work, on short gaps of up to ten characters, Apollo Restore places the correct restoration among its top twenty candidates for 80.6%/54.6%/61.0% of documentary-papyrus, literary-papyrus, and stone-inscription lacunae, exceeding the strongest published models by $1.6\times$/$2.6\times$/$1.4\times$. Prior evaluation protocols, however, inflate scores through a bias toward trivially short gaps; under a length-balanced metric Apollo Restore's advantage over the strongest published mod
    
[^113]: 超越任务完成：训练既有能力又安全的计算机使用智能体

    Beyond Task Completion: Training Capable and Safe Computer-Use Agents

    [https://arxiv.org/abs/2609.22178](https://arxiv.org/abs/2609.22178)

    提出SCOPE联合后训练框架与SCOPE-Gen自动化数据生成流水线，使计算机使用智能体在保持任务执行能力的同时学会基于风险的安全决策——完成良性任务、规避环境危害、并在目标有害或无安全路径时拒绝执行。

    

    计算机使用智能体（CUA）在通过图形用户界面完成复杂任务方面取得了快速进展，然而仅以任务成功为中心的后训练并不能诱导出可靠的安全行为。一个可靠的CUA必须使其执行以风险为条件：它应当完成普通的良性任务，规避环境危害并在存在安全完成路径时继续执行，而当目标有害或不存在安全路径时则拒绝执行。为了学习这种条件策略，我们开发了策略执行的安全与能力优化方法（SCOPE），它对CUA的任务执行能力与安全感知决策进行联合后训练。为了给这一联合目标提供对齐的训练数据，我们进一步引入了SCOPE-Gen，这是一个自动化流水线，能够合成可验证的能力任务，并在保留原始目标的前提下将其转换为成对的环境风险变体。利用由此生成的任务，我们构建了SATraj-OS，

    arXiv:2609.22178v1 Announce Type: cross  Abstract: Computer-use agents (CUAs) have made rapid progress in completing complex tasks through graphical user interfaces, yet post-training centered on task success alone does not induce reliable safety behavior. A reliable CUA must condition its execution on risk: it should complete ordinary benign tasks, avoid environmental hazards and continue when a safe completion path remains, and refuse when the goal is harmful or no safe path exists. To learn this conditional policy, we develop Safety and Capability Optimization for Policy Execution (SCOPE), which jointly post-trains a CUA for task-execution capability and safety-aware decision making. To provide aligned training data for this joint objective, we further introduce SCOPE-Gen, an automated pipeline that synthesizes verifiable capability tasks and converts them into paired environment-risk variants while preserving their original goals. Using the resulting tasks, we construct SATraj-OS, 
    
[^114]: PAGE：分区感知的门控KV缓存淘汰

    PAGE: Partition-Aware Gated KV-Cache Eviction

    [https://arxiv.org/abs/2609.22157](https://arxiv.org/abs/2609.22157)

    PAGE提出一种无需训练、无需标签的门控机制，利用预填充注意力中top-k头一致性的早期到晚期下降来预测输入是否适合KV缓存淘汰，在淘汰会造成灾难性精度损失时自动保留完整缓存。

    

    KV缓存淘汰方法决定保留哪些token，却不决定是否应该淘汰，因此基准测试的平均值可能掩盖一类输入——在这些输入上，压缩会使准确率从99%骤降至0%。我们将淘汰重新构建为逐输入的准入决策，并发现输入可划分为两类：容量受限类（在任何预算下淘汰都是灾难性的）和稀释倾向类（淘汰是安全甚至有益的）。一个从预填充注意力中计算出的无标签标量——成对top-k头一致性的早期到晚期下降——可以在解码开始之前预测输入所属的类别。PAGE对该下降值进行阈值判断：当下降较大时应用任意基础淘汰器，否则保留完整缓存，整个过程无需训练、无需准确率标签。该下降值在四个架构家族中都能一致地按淘汰安全性对输入排序，且针对每个模型约100个输入的无标签试点即可为新架构家族重新校准阈值。作为保障机制，PAGE能够……

    arXiv:2609.22157v1 Announce Type: cross  Abstract: KV-cache eviction methods decide which tokens to keep but not whether to evict at all, so a benchmark mean can hide a class of inputs on which compression drives accuracy from 99\% to 0\%. We reframe eviction as a per-input admission decision and show that inputs separate into a capacity-bound class, where eviction is catastrophic at every budget, and a dilution-prone class, where eviction is safe or beneficial. A single label-free scalar computed from prefill attention, the early-to-late drop in pairwise top-$k$ head agreement, predicts this class before any decoding. PAGE thresholds this drop: it applies any base evictor when the drop is large and retains the full cache otherwise, with no training and no accuracy labels. The drop orders inputs by eviction safety consistently across four architecture families, and a per-model unlabeled pilot of about 100 inputs recalibrates the threshold for a new family. Used as a safeguard, PAGE cut
    
[^115]: MME-Safety：一个用于多模态大语言模型安全评估的细粒度基准

    MME-Safety: A Fine-grained Benchmark for Safety Evaluation of MLLMs

    [https://arxiv.org/abs/2609.20850](https://arxiv.org/abs/2609.20850)

    提出了MME-Safety——一个具有独特四维标注模式和分层评估框架的细粒度安全评估基准，通过对17个最先进多模态大语言模型的零样本评估，系统揭示了跨模态输入配置带来的安全漏洞与不平衡。

    

    尽管多模态大语言模型（MLLMs）取得了显著进展，但其跨模态能力引入了复杂的漏洞，容易绕过单模态过滤器。现有基准缺乏细粒度的意图相关标注，且依赖单维度指标，阻碍了全面的鲁棒性评估。为解决这一问题，我们提出了MME-Safety，这是一个经过严格验证的基准，具有独特的四维标注模式，可对风险场景、危害严重程度以及特定模态的隐蔽级别进行分类。此外，我们引入了一个分层评估框架，用于评估基本的响应可靠性、实际风险暴露以及防御行为的结构完整性。对17个最先进的多模态大语言模型进行的广泛零样本评估，为当前多模态系统提供了全面的安全画像。我们的分析系统地研究了跨模态输入配置，并揭示了安全方面的不平衡现象。

    arXiv:2609.20850v1 Announce Type: new  Abstract: While Multimodal Large Language Models (MLLMs) show remarkable advancements, their cross-modal capabilities introduce complex vulnerabilities that easily bypass unimodal filters. Existing benchmarks lack fine-grained intent-related annotations and rely on unidimensional metrics, hindering comprehensive robustness evaluation. To address this, we propose MME-Safety, a rigorously verified benchmark featuring a unique four-dimensional annotation schema that categorizes risk scenarios, harm severity, and modality-specific stealth levels. Furthermore, we introduce a hierarchical evaluation framework to assess fundamental response reliability, actual risk exposure, and the structural integrity of defensive behaviors. Extensive zero-shot evaluations across 17 state-of-the-art MLLMs provide a comprehensive safety profile of current multimodal systems. Our analysis systematically investigates cross-modal input configurations and uncovers safety im
    
[^116]: 在维基百科摘要上玩 log(N)-问题游戏：成对前沿模型之间的通信效率

    Playing log(N)-Questions over Wikipedia Abstracts: Communication Efficiency Between Paired Frontier Models

    [https://arxiv.org/abs/2609.19113](https://arxiv.org/abs/2609.19113)

    该研究通过让六个前沿语言模型在信息不对称下与自身进行 log(N)-问题博弈，发现模型的自通信胜率遵循 win = p^(log₂ N) 规律（p=0.928），且 Claude Opus 5 显著落后于其他五个几乎难以区分的领先模型。

    

    我们在双智能体 log(N)-问题游戏上评估了六个前沿语言模型。提问者看到 N 个维基百科导语段落，必须使用恰好 log₂ N 个是非问题来识别一个被秘密选定的目标；回答者只看到目标和问题，并用一个词作答。由于两个角色由同一家提供商的模型担任，该游戏衡量的是模型在信息不对称条件下与自身通信的能力。我们在 4 到 1024 个段落的文档集上进行了 408 局游戏，总 API 成本为 363 美元。其中一个模型明显落后于其他模型：Claude Opus 5 在 68 局中仅赢 28 局，而 GLM-5.3、GPT-5.6 Sol、Grok 4.6、Gemini 3.8 Flash 和 Kimi K3 分别赢得 45 至 56 局，领先的前五名之间仅有微小差异。将这五个模型合并分析后，胜率随集合规模增大而下降，相关系数 r=-0.973，且可用单一的单轮可靠性参数拟合，形式为 win = p^(log₂ N)，其中 p=0.928。失败可分为回答错误和区分（错误）……

    arXiv:2609.19113v1 Announce Type: new  Abstract: We evaluate six frontier language models on the two-agent $\log(N)$-Questions game. A questioner sees $N$ Wikipedia lead paragraphs and must identify a secretly chosen target using exactly $\log_2 N$ yes/no questions. An answerer sees only the target and the question, and replies with one word. Both roles run on the same provider, so the game measures how well a model communicates with itself across an information asymmetry. We run 408 games over document sets of 4 to 1024 paragraphs at a total API cost of \$363. One model finishes well behind the others: Claude Opus 5 wins 28 of 68 games, against 45 to 56 for GLM-5.3, GPT-5.6 Sol, Grok 4.6, Gemini 3.8 Flash and Kimi K3. The leading five are only marginally separable. Pooling those five, win rate declines with set size at $r=-0.973$ and is fit by a single per-round reliability parameter. The form is $\text{win}=p^{\log_2 N}$ with $p=0.928$. Losses divide into answer errors and discrimina
    
[^117]: 回滚世界，保留反思：面向长程LLM智能体的回滚诱导反思

    Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2609.18304](https://arxiv.org/abs/2609.18304)

    提出了回滚诱导反思（RIR）统一恢复框架，在将LLM智能体回滚到选定先前状态的同时，保留从被放弃轨迹中提炼的可复用知识，解决了长程任务中错误累积且难以可靠恢复的问题。

    

    大语言模型（LLM）智能体越来越多地通过多步环境交互来处理长程任务，然而单个错误的动作可能会改变后续的状态和观测，导致错误随时间不断累积。现有方法要么在不修复已改变环境状态的情况下纠正上下文，要么在恢复早期状态的同时丢弃有用的经验，这使得既消除失败条件又避免重复过去的错误变得困难。我们认为，可靠的恢复应被视为一个回滚边界控制问题，即联合决定何时干预、从何处恢复，以及哪些信息应在恢复过程中保留。基于这一观点，我们提出了回滚诱导反思（RIR），这是一个统一的恢复框架，它将执行恢复到选定的先前状态，同时保留从被放弃轨迹中提炼出的可复用知识，以指导后续决策。我们进一步刻画……

    arXiv:2609.18304v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly tackle long-horizon tasks through multi-step environment interaction, yet a single erroneous action can alter subsequent states and observations, causing errors to compound over time. Existing methods either correct the context without repairing altered environment states or restore earlier states while discarding useful experience, making it difficult to both eliminate failure conditions and avoid repeating past mistakes. We argue that reliable recovery should instead be treated as a rollback-boundary control problem that jointly determines when to intervene, where to resume, and what information should survive recovery. Based on this view, we propose Rollback-Induced Reflection (RIR), a unified recovery framework that restores execution to a selected prior state while carrying forward reusable knowledge distilled from the abandoned trajectory to guide subsequent decisions. We further chara
    
[^118]: 解耦多智能体大语言模型中的拓扑结构与多样性以实现多语言低资源情感检测

    Disentangling Topology and Diversity in Multi-Agent LLMs for Multilingual Low-Resource Emotion Detection

    [https://arxiv.org/abs/2609.14570](https://arxiv.org/abs/2609.14570)

    该论文首次将多智能体LLM系统中的推理拓扑结构与智能体间多样性来源解耦并独立研究，发现并行的学习型QLoRA专门化在九种语言的多语言低资源情感检测上表现最佳，且最优拓扑结构取决于所采用的多样性来源。

    

    多智能体大语言模型系统结合了多次推理调用，但先前的工作常常将“调用之间如何连接”与“调用之间如何实现多样化”这两个因素混为一谈。我们独立地研究这两个因素：推理拓扑结构和智能体间多样性的来源。在一个受控的 2×3 矩阵实验中，我们将并行聚合与顺序细化两种拓扑结构，分别与随机采样、角色提示和基于学习的 QLoRA 专门化三种多样性来源进行交叉组合，并在每个骨干模型内保持固定的三次调用预算和统一的输出协议。我们使用 Qwen2.5-14B-Instruct 和 Llama-3.1-8B-Instruct，在涵盖九种语言的多语言低资源情感检测任务上评估了全部六种配置。并行的学习型专门化在 Qwen 上表现最强，达到 52.83 Macro-F1，在 Llama 上达到 52.94。在 Qwen 上，该方法还超越了同骨干模型的零样本、少样本、思维链以及七次调用的自一致性基线。最优拓扑结构取决于多样性来源：顺序细化对随机采样和提示驱动的多样性更有帮助（摘要原文在此处截断）。

    arXiv:2609.14570v1 Announce Type: cross  Abstract: Multi-agent LLM systems combine multiple inference calls, but prior work often confounds how calls are connected with how they are diversified. We study these factors independently: inference topology and source of inter-agent diversity. In a controlled $2 \times 3$ matrix, we cross parallel aggregation and sequential refinement with stochastic sampling, role prompting, and learned QLoRA specialization, under a fixed three-call budget and output protocol within each backbone. Using Qwen2.5-14B-Instruct and Llama-3.1-8B-Instruct, we evaluate all six configurations on multilingual low-resource emotion detection across nine languages. Parallel learned specialization is strongest on Qwen at 52.83 Macro-F1 and reaches 52.94 on Llama. On Qwen it also exceeds same-backbone zero-shot, few-shot, CoT, and seven-call self-consistency baselines. The preferred topology depends on diversity source: sequential refinement helps stochastic and prompted
    
[^119]: 衡量前沿大语言模型在自动化研究中的创造力

    Measuring the Creativity of Frontier LLMs in Automated Research

    [https://arxiv.org/abs/2609.14057](https://arxiv.org/abs/2609.14057)

    本文提出了一套从价值性和新颖性两个维度评估大语言模型自动化研究创造力的指标体系，发现模型在反映研究空间探索广度的变量级新颖性指标上差异显著。

    

    前沿大语言模型越来越有能力开展自动化研究，但它们在这种场景下的创造力尚未得到系统性评估。本文提出了一套指标，从价值性和新颖性两个维度评估创造力。价值性评估每个提出的想法是否有用，而新颖性则从三个角度进行评估：相同想法是否曾经出现过（精确匹配P-新颖性，Exact-Match P-Novelty）、是否探索了此前未被探索过的变量或变量组合（变量级P-新颖性，Variable-level P-Novelty），以及该想法是直接遵循检索到的外部知识还是对其有所突破（H-新颖性，H-Novelty）。我们的评估表明，这些模型在大多数创造力指标上取得了相对相似的分数，但在变量级P-新颖性上存在显著差异，该指标反映了研究空间探索的广度。进一步的相关性和想法层面性能分析表明，变量级P-新颖性……

    arXiv:2609.14057v1 Announce Type: new  Abstract: Frontier LLMs are increasingly capable of conducting automated research, yet their creativity in this setting has not been systematically evaluated. In this paper, we propose a set of metrics to evaluate creativity along the two dimensions of valueness and novelty. Valueness assesses whether each proposed idea is useful, while novelty is evaluated from three perspectives: whether the same idea has appeared before (Exact-Match P-Novelty), whether a previously unexplored variable or variable combination is explored (Variable-level P-Novelty), and whether the idea directly follows retrieved external knowledge or departs from it (H-Novelty). Our evaluation shows that the models achieve relatively similar scores on most creativity metrics, but differ substantially in Variable-level P-Novelty, which reflects the breadth of research-space exploration. Further correlation and idea-level performance analyses show that Variable-level P-Novelty is 
    
[^120]: 人类建造的最后一个AI：迈向真正的递归自我改进

    The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement

    [https://arxiv.org/abs/2609.11873](https://arxiv.org/abs/2609.11873)

    本文提出递归自我改进（RSI）概念及其从改进执行自主性到递归元改进的发展路线图，通过Headroom-Closed指数揭示现有大语言模型的局限，并结合行业实践识别出实现真正AI自我改进的关键挑战。

    

    递归自我改进（RSI）使AI系统能够将经验和反馈转化为持久性的改变，从而同时提升其自身能力和未来的改进过程。我们首先使用Headroom-Closed指数（HCI）揭示现有大语言模型存在的问题，然后介绍RSI概念及其发展路线图：从改进执行自主性、改进策略自主性、经验获取自主性和环境适应自主性，到递归元改进。接下来，我们考察了RSI在不同场景（如科学发现、具身智能、软件工程）中的表现，重点阐述了它们各自不同的需求和发展速度。借鉴多样的行业实践和初步的实证证据，我们将RSI研究与实际系统联系起来，并识别出实现真正RSI所面临的关键挑战。

    arXiv:2609.11873v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) enables AI systems to turn experience and feedback into persistent changes that improve both their capabilities and the process of future improvement. We first use the Headroom-Closed Index (HCI) to reveal the problems of existing LLMs, then introduce the RSI concept and its development roadmap: from improvement-execution autonomy, improvement-strategy autonomy, experience-acquisition autonomy, and environment-adaptation autonomy, to recursive meta-improvement. Next we examine RSI across scenarios (e.g., scientific discovery, embodied intelligence, software engineering), highlighting their distinct requirements and development speeds. Drawing on diverse industry practices and preliminary empirical evidence, we connect RSI research with practical systems and identify key challenges to achieving genuine RSI.
    
[^121]: 奥古斯丁式BabyLM：实指定名能教给小语言模型什么、不能教什么

    Augustinian BabyLM: What Ostensive Definition Can and Cannot Teach a Small Language Model

    [https://arxiv.org/abs/2609.11870](https://arxiv.org/abs/2609.11870)

    本研究将圣奥古斯丁的“实指定名”词义学习思想应用于小型语言模型，发现视觉初始化的词嵌入会留下持续到训练结束的印记，且仅在物体属性知识等零样本任务中带来提升，而对大多数语法能力基准测试没有影响。

    

    语言模型通常以随机的词嵌入开始训练：“香蕉”的含义必须从训练语料中学习。我为一个在1000万词上训练的小型掩码语言模型实现了圣奥古斯丁的词语学习图景——通过实指获得意义：在训练之前，具有视觉基础的词会获得由其标注的图像区域导出的嵌入，而其他词则以随机初始化开始。视觉初始化留下了可测量的印记，并一直持续到训练结束。与此同时，这种效应在大多数探测抽象语法知识的BabyLM基准测试中仍然不可见：视觉初始化在这些测试中并不影响性能。唯一的零样本例外是物体属性知识（COMPS，Misra等人，2023），此时种子嵌入在所有配置中都有帮助。为进一步验证这一结果，我构建了一个针对语料库定制的Visual-Property Swap基准测试版本（Lin等人，2026），该测试检验颜色、材质等物体属性。

    arXiv:2609.11870v1 Announce Type: new  Abstract: A language model normally begins training with random word embeddings: whatever 'banana' means must be learned from training corpora. I implement St. Augustine's picture of word learning, meaning by ostension, for a small masked language model (DeBERTa) trained on 10M words: before training, visually grounded tokens receive embeddings derived from the image regions they label; other tokens start random. Visual initialization leaves a measurable imprint that lasts until the end of training. At the same time, the effect remains invisible under most BabyLM benchmarks, which probe abstract grammatical knowledge: visual initialization does not affect performance there. The only zero-shot exception is object-property knowledge (COMPS, Misra et al. 2023), where seeding helps in every configuration. To follow up on this result, I build a corpus-tailored version of the Visual-Property Swap benchmark (Lin et al., 2026), which tests color, material
    
[^122]: 语义提升算子与不可判定类在保持性下的闭包性

    The Semantic Elevation Operator and the Closure of the Undecidable Class under Preservation

    [https://arxiv.org/abs/2609.11326](https://arxiv.org/abs/2609.11326)

    该论文提出语义提升算子 ΛΦ，将程序静态语义性质问题转化为自修改后的保持性问题，并基于克林递归定理证明不可验证性质类在该算子下封闭，且无界迭代将攀升至算术层级的 Π₂-完备性。

    

    程序静态语义性质的不可判定性由莱斯定理所支配。然而，自修改系统需要分析的并非某个性质当前是否成立，而是当系统重写自身时该性质是否被保持。我们通过语义提升算子 ΛΦ 将这一转变形式化，它把静态问题“x 是否满足 P？”转化为动态问题“x 经过 Φ 变换后 P 是否被保持？”。我们证明：当 Φ 是内涵性的（即依赖于源代码本身，而不仅仅是所计算的函数）时，即使提升后的性质破坏了莱斯定理所要求的外延性，它仍然是不可判定的；该证明基于克林递归定理，而非莱斯定理。因此，不可验证性质类 U 在提升算子下是封闭的。对该算子的无界迭代将沿算术层级攀升至 Π₂-完备性，进一步巩固了不可验证性。

    arXiv:2609.11326v1 Announce Type: cross  Abstract: The undecidability of a program's static semantic properties is governed by Rice's theorem. Self-modifying systems, however, require analysing not whether a property holds now, but whether it is preserved when the system rewrites itself. We formalise this transition through a semantic elevation operator {\Lambda}{\Phi}, which turns the static question "does x satisfy P?" into the dynamic question "is P preserved after x is transformed by {\Phi}?". We prove that when {\Phi} is intensional (depending on the source code, not only on the computed function), the elevated property remains undecidable even though it breaks the extensionality that Rice's theorem requires; the proof rests on Kleene's recursion theorem, not on Rice. Consequently the class U of non-verifiable properties is closed under the elevation operator. Unbounded iteration of the operator climbs the arithmetical hierarchy -to {\Pi}02-completeness- consolidating non-verifiab
    
[^123]: 基于大语言模型锚定的副语言信息增强用于阿尔茨海默病检测

    LLM-Anchored Paralinguistic Enrichment for Alzheimer's Disease Detection

    [https://arxiv.org/abs/2609.10896](https://arxiv.org/abs/2609.10896)

    提出LAPE方法，通过韵律事件文本化、词汇-韵律单元化分块和文本锚定的副语言融合三项创新，将停顿和词语拖长等副语言线索与大语言模型的语言表示相融合，以提升基于语音的阿尔茨海默病自动检测效果。

    

    基于语音的阿尔茨海默病（AD）自动检测为早期认知筛查提供了一种无创且可扩展的方法。阿尔茨海默病既影响词汇-语义组织，也影响语音产生，包括非典型的停顿和词语拖长。然而，现有方法尚未将这些副语言线索与语言内容充分融合。我们提出了大语言模型锚定的副语言增强方法（LAPE），通过三项协同创新将副语言线索融入大语言模型导出的语言表示中。第一是韵律事件文本化，通过将停顿和拖长编码为具有时长感知有界重复的显式标记，使大语言模型能够将停顿和拖长与词汇内容进行联合建模。第二是词汇-韵律单元化与分块，通过仅对连续词单元进行池化，在两种模态中保留事件的同一性和幅度。第三是文本锚定的副语言融合……

    arXiv:2609.10896v1 Announce Type: new  Abstract: Speech-based automatic detection of Alzheimer's disease (AD) provides a non-invasive and scalable approach to early cognitive screening. AD affects both lexical-semantic organization and speech production, including atypical pauses and word elongations. However, existing methods have yet to fully integrate these paralinguistic cues with linguistic content. We propose LLM-Anchored Paralinguistic Enrichment (LAPE), which enriches LLM-derived linguistic representations with paralinguistic cues through three coordinated innovations. The first is prosodic event textualization, which enables the LLM to model pauses and elongations jointly with lexical content by encoding them as explicit markers with bounded duration-aware repetition. The second is lexico-prosodic unitization and chunking, which preserves event identity and magnitude in both modalities by pooling only consecutive word units. The third is text-anchored paralinguistic fusion, wh
    
[^124]: VERPO：验证证据正则化策略优化

    VERPO: Verified Evidence Regularized Policy Optimization

    [https://arxiv.org/abs/2609.06100](https://arxiv.org/abs/2609.06100)

    VERPO提出了一种验证证据正则化策略优化框架，通过Fisher证据对比和带停止机制的逐token ZPD控制器有选择性地应用基于证据的修正，在保留可验证结果目标的同时避免盲目模仿导致的负面迁移。

    

    可验证的结果奖励可以指导语言模型的后训练，但序列级别的优势无法识别哪些token级别的决策应当被保留或修改。证据条件教师通过以特权反馈重放采样轨迹来提供更密集的监督。然而，不加区分的模仿可能会迁移那些不支持任务成功的格式或推理风格偏移。我们提出了VERPO，一个验证证据正则化策略优化框架，它将证据视为策略修正的提议，同时保留结果目标。该框架将无证据的参考恢复与带符号的token级证据修正分离开来。Fisher证据对比沿着估计的证据存在方向对修正进行衰减。一个带停止机制的逐token ZPD控制器根据局部奖励对齐程度和Fisher移动成本来调节修正的接受度，而参考通道保持独立于接受决策。

    arXiv:2609.06100v1 Announce Type: cross  Abstract: Verifiable outcome rewards guide language-model post-training, but sequence-level advantages do not identify which token-level decisions should be preserved or revised. Evidence-conditioned Teachers provide denser supervision by replaying sampled trajectories with privileged feedback. Yet indiscriminate imitation risks transferring formatting or reasoning-style shifts that do not support task success. We introduce VERPO, a Verified Evidence Regularized Policy Optimization framework that treats evidence as a proposal for policy correction while retaining the outcome objective. It separates evidence-free reference restoration from signed token-level evidence corrections. Fisher Evidence Contrast attenuates corrections along an estimated evidence-presence direction. A stopped token-wise ZPD controller scales acceptance according to local reward alignment and Fisher movement cost, while the reference channel remains independent of acceptan
    
[^125]: 当用户不提问时：对话式智能体中上下文驱动的记忆检索基准测试

    When Users Don't Ask: Benchmarking Context-Driven Memory Retrieval in Conversational Agents

    [https://arxiv.org/abs/2609.03467](https://arxiv.org/abs/2609.03467)

    该论文提出了对话式记忆基准LOCOMO-CONV，通过对话式、隐含式、反事实和组合式四种查询风格，揭示了问答式基准测试所忽视的记忆检索差距，并发现强检索能力并不完全等同于高质量的对话响应。

    

    大型语言模型（LLM）越来越多地被部署为长时程对话式智能体，这推动了对记忆系统日益增长的关注。然而，现有基准测试主要通过问答式的探测方式来评估记忆，而非在真实对话场景中的实际使用。我们提出了LOCOMO-CONV，这是一个基于LoCoMo派生的对话式记忆基准，包含四种查询风格：对话式、隐含式、反事实式和组合式。我们在五个代表性记忆系统上评估了检索召回率和端到端响应质量。实验表明，对话式的表述方式暴露了问答式基准所忽视的大量检索差距，尤其是在隐含式和组合式查询上；多方面查询改写可以缩小原始对话轮次记忆的差距，但对抽象化记忆则无效。我们进一步发现，强大的检索能力并不能完全转化为响应质量，且隐含式查询表现出“静默接地”现象，即记忆在……（原文摘要在此处截断）

    arXiv:2609.03467v1 Announce Type: cross  Abstract: Large language models (LLMs) are increas- ingly deployed as long-horizon conversational agents, motivating growing interest in mem- ory systems. However, existing benchmarks primarily evaluate memory through QA-style probing rather than in-situ conversational usage. We introduce LOCOMO-CONV, a conversa- tional memory benchmark derived from Lo- CoMo with four query styles: dialog, implicit, counterfactual, and composed. Across five rep- resentative memory systems, we evaluate both retrieval recall and end-to-end response qual- ity. Our experiments show that conversational framing exposes substantial retrieval gaps over- looked by QA benchmarks, especially on im- plicit and composed queries, which multi-facet query rewriting narrows for raw-turn mem- ory but not abstractive memory. We further find that strong retrieval does not fully trans- late into response quality, and that implicit queries exhibit silent grounding, where mem- ory imp
    
[^126]: 预算化继承式智能体记忆验证中的计划指针与记录指令形式

    Plan Pointers and Record-Directive Form in Budgeted Verification of Inherited Agent Memory

    [https://arxiv.org/abs/2609.03450](https://arxiv.org/abs/2609.03450)

    该论文通过十二项注册研究发现，写入智能体记忆库的指令形式（准则、裸ID或指针）会以高度模型依赖的方式显著影响预算受限下的记录选择，长度匹配准则可带来35分的提升，但附加ID可能完全抵消准则的效果。

    

    arXiv:2609.03450v1 公告类型：cross。摘要：一个继承了六条单行记忆的智能体在行动前最多只能拉取一条存档的源记录；写入存储中的指令可以引导这一选择：可以是指向该记录的指针、识别该记录的准则，或两者兼有。在同一仪器谱系上的十二项注册研究（共14,760次尝试）中，我们测量了每种指令形式下请求的去向。在六个直接提供商模型上，长度匹配的准则比裸ID高出+35.0个点 [+31.2, +38.8]（研究D）；而在九个模型的OpenRouter服务面板上，该对比未能通过注册的优越性规则（研究E）。在三个Claude模型上，附加ID会抵消准则的效果（Opus 5: 从40/40降至0/40；研究F-x）；六次字节匹配的编辑使每个精确字符串都产生了各自的效应（研究G），并且在每单元八十次运行的重跑中，三十个复现对比中有十五个处于误差范围内，十五个未获解决，没有一个超出范围（研究G'）。批准行（在Opus 5上+96.0个点）以及一个（摘要在此处截断）

    arXiv:2609.03450v1 Announce Type: cross  Abstract: An agent that inherits six one-line memories may pull at most one archived source record before acting; a directive written into the store can steer that choice: a pointer to the record, a criterion that identifies it, or both. Across twelve registered studies on one instrument lineage (14,760 attempts) we measured where the request goes under each form. On six direct-provider models a length-matched criterion exceeded a bare id by +35.0 points [+31.2, +38.8] (Study D); the contrast failed its registered superiority rule on a nine-model OpenRouter-served panel (Study E). Appending the id cancelled the criterion on three Claude models (Opus 5: 40/40 to 0/40; Study F-x); six byte-matched edits gave each exact string its own effect (Study G), and a re-run at eighty runs per cell left fifteen of thirty replication contrasts within the margin, fifteen unresolved and none beyond (Study G'). A ratification line (+96.0 points on Opus 5) and a 
    
[^127]: Lngram v2：具有可解释离散表示的潜在N-元语法记忆

    Lngram v2: Latent N-Gram Memory with Interpretable Discrete Representations

    [https://arxiv.org/abs/2609.03426](https://arxiv.org/abs/2609.03426)

    Lngram v2通过解耦记忆容量与骨干网络宽度、引入上下文感知的分组查询注意力读取机制以及零值Sink和反事实代理梯度等改进，在保留硬离散寻址的同时实现了记忆容量的独立扩展，并成功应用于300亿参数的视觉-语言模型。

    

    Transformer缺乏原生的查找机制，需要通过重复的密集计算来识别和复用局部静态模式。Lngram v1通过离散潜在n-元语法寻址引入了与分词器无关的条件记忆，但其记忆容量与骨干网络宽度相耦合，高昂的参数和激活成本限制了其可扩展性。我们提出Lngram v2，将路由数量、记忆维度和骨干网络宽度进行解耦，并引入上下文感知的分组查询注意力读取机制，从而实现记忆容量的独立扩展。零值Sink和反事实代理梯度进一步提升了读取的选择性和路由的可训练性，同时保留了硬离散寻址。在不同规模的视觉-语言模型（VLM）上的实验显示了一致的性能提升，包括成功扩展至300亿参数的模型。与Lngram v1相比，Lngram v2大幅降低了总参数量和激活……

    arXiv:2609.03426v1 Announce Type: new  Abstract: Transformers lack a native lookup mechanism, requiring repeated dense computation to recognize and reuse local static patterns. Lngram v1 introduces tokenizer-independent conditional memory through discrete latent n-gram addressing, but its memory capacity is coupled with the backbone width, limiting scalability due to high parameter and activation costs. We propose Lngram v2, which decouples the number of routes, memory dimension, and backbone width, and introduces a context-aware grouped-query attention readout to scale memory capacity independently. A zero-value Sink and counterfactual surrogate gradients further improve readout selectivity and routing trainability while preserving hard discrete addressing. Experiments across vision--language models (VLMs) of different scales show consistent improvements, including successful scaling to a 30B-parameter model. Compared with Lngram v1, Lngram v2 substantially reduces both total and acti
    
[^128]: 面向合理性感知生物医学人工智能的定量证据挖掘

    Quantitative Evidence Mining for Plausibility-Aware Biomedical AI

    [https://arxiv.org/abs/2608.30393](https://arxiv.org/abs/2608.30393)

    该论文主张生物医学AI应从以实体关系为中心的提取方式，转向能够保留剂量、效应量、研究人群、对照和不确定性等定量细节的证据挖掘，使科学论断可追溯、可验证、可比较和可复用，从而构建合理性感知的生物医学AI系统。

    

    生物医学人工智能（AI）系统日益从文献、临床试验和监管文件中提取、组织和复用科学论断。但仅靠自动提取并不能使一个论断成为可靠的证据：一个论断只有在其来源可追溯、与支持它的定量细节相关联、并在其生物医学背景和不确定性中被解读时，才具有实用价值。随着大语言模型（LLM）和日益自主化的系统推动证据综合、知识图谱（KG）构建和决策支持，这一点变得尤为重要。许多文本挖掘和LLM流水线仍然以关系为中心：它们捕获实体和关系（如Drug--TREATS--Disease，即药物--治疗--疾病），但丢弃了剂量、效应量、研究人群、对照、不确定性以及论断成立的条件。这样的关系看似可操作，却难以验证、比较或复用。在这一观点文章中，我们主张向定量挖掘方向转变（原文摘要在此处截断）。

    arXiv:2608.30393v1 Announce Type: new  Abstract: Biomedical artificial intelligence (AI) systems increasingly extract, organize, and reuse scientific claims from literature, clinical trials, and regulatory documents. But automatic extraction alone does not make a claim reliable evidence: a claim becomes useful only when it can be traced to its source, linked to the quantitative details that support it, and read within its biomedical context and uncertainty. This matters as large language models (LLMs) and increasingly autonomous systems drive evidence synthesis, knowledge graph (KG) construction, and decision support. Many text-mining and LLM pipelines remain relation-centric: they capture entities and relations such as Drug--TREATS--Disease, but drop the dose, effect size, population, comparator, uncertainty, and conditions under which a claim holds. Such relations can look actionable yet remain hard to verify, compare, or reuse. In this perspective, we argue for a shift toward quanti
    
[^129]: AI写作具有一致的风格计量学足迹，但AI编辑则不然

    AI Writers Have a Consistent Stylometric Footprint, but AI Editors Do Not

    [https://arxiv.org/abs/2608.27855](https://arxiv.org/abs/2608.27855)

    本文发现AI生成的文本具有跨8个模型和5个领域保持一致的风格计量学“足迹”（主要由熵和词汇多样性等特征构成），可用于检测AI生成文本，但AI编辑过的人类文本并不会留下同样的足迹。

    

    大型语言模型（LLM）生成的文本已被证明在风格计量学上与人类撰写的文本截然不同。但LLM越来越多地不仅被用于生成文本，还被用于编辑人类写作，目前尚不清楚这两种用途是否会留下相同的痕迹。我们证明，AI生成会留下一个一致的“风格计量足迹”：一小部分特征（主要是熵和词汇多样性）在8个LLM和5个领域中始终能够将AI生成的文本与人类写作区分开来，而其余特征则严重依赖于具体领域和生成器。然而，AI编辑并不会重现同样的足迹。相对于其人类撰写的原始文本，AI编辑后的文本仅表现出词汇多样性的小幅增加以及……

    arXiv:2608.27855v1 Announce Type: new  Abstract: Text generated by large language models (LLMs) has been shown to be stylometrically distinct from human-written text \citep{andreDetectingAIAuthorship2023, shahDetectingUnmaskingAIGenerated2023, oparaStyloAIDistinguishingAIGenerated2024, soto2024fewshot, liLinguisticDifferencesAI2025, selviogluFeatureExtractionAnalysis2025}. But LLMs are increasingly used not only to generate text but also to edit human writing, and it is unclear whether the two leave the same trace. We show that AI generation leaves a consistent ``stylometric footprint'': a small subset of features, primarily entropy and lexical diversity, consistently separates AI-generated text from human writing across 8 LLMs and 5 domains, while the remaining features depend heavily on the domain and generator. AI editing, however, does not reproduce the same footprint. Relative to their human-written sources, AI-edited texts show only a small increase in lexical diversity and a dec
    
[^130]: 音频-视觉大语言模型中的组合性失败：跨模态冲突下的深层先验主导

    Compositional Failure in Audio-Visual LLMs: Late-Layer Prior Dominance Under Cross-modal Conflict

    [https://arxiv.org/abs/2608.27785](https://arxiv.org/abs/2608.27785)

    本研究揭示了音频-视觉大语言模型在跨模态冲突下存在“先验主导”失败模式——模型后期层（集中于约25.5层）固守内部偏好的答案模式而忽视冲突输入，导致准确率大幅下降，且增强时序对齐仅能改变答案偏差而无法提升组合泛化能力。

    

    我们研究音频-视觉冲突作为AV-LLMs（音频-视觉大语言模型）的组合泛化测试：模型必须结合同步但语义不兼容的音频和视频证据，并判断该配对是否匹配。在VideoLLaMA 2-7B-AV上，三种对齐配置在AVHBench的评分精确字符串是/否子集上仍接近随机水平，尽管其输出先验发生了显著变化。类似地，现成的InternVideo2在跨模态冲突下准确率特异性下降了32.3%，并伴随17.3%的指令遵循失败。我们将这种失败模式称为“先验主导”（prior dominance）：模型后期层对内部偏好答案模式的坚定承诺，而这种承诺与冲突输入的关联较弱。为解释这一行为，我们进行了机制可解释性分析，发现这种承诺集中在25.5±1层。我们进一步表明，更强的时序对齐会改变答案偏差，但并不能改善组合性表现。

    arXiv:2608.27785v1 Announce Type: new  Abstract: We study audio-visual conflict as a compositional generalization test for AV-LLMs: the model must combine synchronized but semantically incompatible audio and video evidence and decide whether the pair matches. On VideoLLaMA 2-7B-AV, three alignment configurations remain nearchance on the scored exact-string Yes/No subset of AVHBench, even though their output priors shift substantially. Similarly, off-the-shelf InternVideo2 experienced a 32.3% accuracy decrease specifically under cross-modal conflict, accompanied by a 17.3% instruction-following failure. We call this failure mode prior dominance: late-layer commitment to an internally preferred answer pattern that is weakly grounded in the conflicting inputs. To explain this behavior, we conduct a mechanistic interpretability analysis and find that commitment remains concentrated at 25.5 $\pm$ 1 layers. We show that stronger temporal alignment changes answer bias, but do not improve comp
    
[^131]: 基于账本控制的零样本自编排提升LLM编码性能

    Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance

    [https://arxiv.org/abs/2608.26480](https://arxiv.org/abs/2608.26480)

    本文证明，在不进行训练或基准调优的情况下，基于账本控制的管理器-工作器脚手架能显著提升某些LLM的编码性能，但效果因模型而异，并非普遍适用。

    

    多智能体大语言模型系统被广泛报道能超越单模型基线，但证据不一，且比较通常存在混淆：流程同时改变令牌预算、工具调用和提示，因此总体增益很少能揭示真正有效的因素。我们研究了在共享文件系统工作区中引入管理器-工作器脚手架的效果，无需训练且无需针对基准进行调优，与同一模型单次回答进行对比。在九个模型上——五个开放权重模型，参数范围从9B到约2.8T，以及四个前沿封闭模型——针对LiveCodeBench最新的100个困难问题，脚手架的好处是真实但有条件的：对某些模型效果显著且统计显著（如Qwen3.8-27B提升23.4，GPT-5.6-Luna提升10.6，GPT-5.6-Terra提升8.0，各基于五次配对运行；Kimi-K3提升30.4，Minimax-M3提升11.0，基于五次配对运行且关闭推理，p值均小于10^-4，以及...）

    arXiv:2608.26480v1 Announce Type: cross  Abstract: Multi-agent large language model systems are widely reported to beat single-model baselines, but the evidence is mixed, and comparisons are usually confounded: pipelines change token budgets, tool calls, and prompts simultaneously, so an aggregate gain rarely reveals what actually helped. We investigate the effect of introducing the manager-worker scaffold over a shared filesystem workspace, with no training and no per-benchmark tuning, measured against the same model answering in a single pass. Across nine models -- five open-weight, spanning 9B to ~2.8T parameters, and four frontier closed models -- on the 100 latest hard LiveCodeBench problems, the scaffold's benefit is real but conditional: large and statistically significant for some (Qwen3.8-27B +23.4, GPT-5.6-Luna +10.6 and GPT-5.6-Terra +8.0, each over five paired passes; Kimi-K3 +30.4 and Minimax-M3 +11.0 over five paired passes with reasoning off, both at $p < 10^{-4}$, and +
    
[^132]: 基于GNN的知识图谱问答的查询侧攻击：从实体链接到答案生成的故障追踪

    Query-Side Attacks on GNN-Based KGQA: Tracing Failures from Entity Linking to Answer Generation

    [https://arxiv.org/abs/2608.25922](https://arxiv.org/abs/2608.25922)

    本文通过阶段隔离协议发现，基于GNN的知识图谱问答系统的主要脆弱性在于子图构建阶段，而非GNN推理阶段，这挑战了现有鲁棒性评估的假设。

    

    基于GNN的知识图谱问答（KGQA）流水线通过四个离散阶段处理查询：实体链接、子图检索、GNN推理和答案生成。标准的鲁棒性评估将阶段级故障合并为一个端到端指标，掩盖了脆弱性的来源和适当的缓解目标。我们研究了当流水线受到输入问题上的对抗性扰动时，哪个阶段失败，以及为什么失败。我们引入了一种阶段隔离协议，包含两种经过知识图谱验证的、保持答案的对抗性扰动：组合重构（CR）和关系同义词交换（RS）分别针对不同阶段，同时保持实体种子不变。在ComplexWebQuestions和WebQSP上的评估结果与主流假设相反：当子图完整时，GNN推理阶段保持接近基线的准确性，而子图构建则导致了大部分故障。

    arXiv:2608.25922v1 Announce Type: new  Abstract: GNN-based Knowledge Graph Question Answering (KGQA) pipelines process queries through four discrete stages: entity linking, subgraph retrieval, GNN reasoning, and answer generation. Standard robustness evaluations conflate stage-level failures into a single end-to-end metric, obscuring both the source of brittleness and the appropriate mitigation target. We ask which stage fails, and why, when the pipeline is subjected to adversarial perturbations on the input question. We introduce a stage-isolation protocol with two answer-preserving adversarial perturbations verified against the knowledge graph: Compositional Restructuring (CR) and Relation Synonym Swap (RS) target distinct stages while leaving entity seeds intact. Evaluated across ComplexWebQuestions and WebQSP, the results run counter to prevailing assumptions: the GNN reasoning stage retains near-baseline accuracy when the subgraph is intact, while subgraph construction accounts fo
    
[^133]: ROBE：用于从历史文本中提取极端长尾事件的逆序偏置专家

    ROBE: Reversed-Order-Biased-Experts for Extracting Extreme Long-tail Events from Historical Texts

    [https://arxiv.org/abs/2608.24268](https://arxiv.org/abs/2608.24268)

    本文提出ROBE方法，通过为事件子组创建专家分类器并在预测时赋予代表性不足事件的专家更高优先级，成功从17-18世纪荷兰历史语料库中提取超过50种极端长尾事件。

    

    本文提出了一些方法，用于从一个涵盖17和18世纪的荷兰语历史语料库中提取超过50种类型的事件。我们提出的方法旨在应对机器学习中一个非常具有挑战性的场景：提取长尾中的长尾。19世纪之前的历史数据本身就是一个在大语言模型预训练中未被覆盖的小众领域，而我们的目标是提取在该领域可用训练数据中仅有少量标注的事件。我们提出为训练数据中存在的事件子组创建专家分类器，并基于训练数据中的相似频率或语义相关性进行分组。在预测时，为在代表性不足事件上训练的专家分配更高的优先级，以避免被频率偏差所主导。我们将这种专门为保护长尾而量身定制的新分类器组合方式称为ROBE：逆序偏置专家。

    arXiv:2608.24268v2 Announce Type: replace  Abstract: This paper proposes methods to extract over 50 types of events from a Dutch historical corpus spanning the 17th and 18th centuries. The methods we propose aim to tackle a very challenging scenario in Machine Learning: extracting the long-tail of the long-tail. Historic data from before the 19th century is in itself a niche domain not covered in the pre-training of Large Language Models, and we aim to extract events only scarcely annotated in the training data available for this domain. We propose creating expert classifiers for subgroups of the events present in the training data. We make these groupings based on similar frequency in the training data or on semantic relatedness. Experts trained on underrepresented events are assigned higher priority when predicting to avoid being dominated by frequency biases. We refer to this new way of combining classifiers, specifically tailored to protect the long-tail, as ROBE: Reversed-Order-Bi
    
[^134]: LiLiCorr：用于投机解码的并行草稿轻量级似然相关性方法

    LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts for Speculative Decoding

    [https://arxiv.org/abs/2608.20530](https://arxiv.org/abs/2608.20530)

    LiLiCorr通过轻量级似然相关性模型关联起草器的逐位置边际分布，在不构造完整联合分布的情况下捕获块级联合结构，从而提升投机解码的连贯性。

    

    arXiv:2608.20530v1 公告类型：新 摘要：投机解码通过起草未来标记并由目标模型并行验证来加速语言模型推理。诸如DFlash之类的扩散式块头是一种有吸引力的起草器，它能在一次前向传播中预测整个未来标记块。然而，它基于逐位置边际分布而非联合块分布进行训练，因此其生成的标记在单个上看似合理，但整体上不连贯。我们引入了LiLiCorr，一种轻量级的基于似然的模型，用于关联起草器已生成的逐位置边际分布。它在每个位置保留前k个标记作为候选，并联合处理它们，为每个标记生成一个输入向量和一个输出向量。当较早候选的输出向量与较晚候选的输入向量具有高余弦相似度时，一对相邻候选即匹配。这些匹配捕获了块的联合结构，而无需显式构造完整的联合分布。一个轻量级的模型就能实现这一点。

    arXiv:2608.20530v1 Announce Type: new  Abstract: Speculative decoding accelerates language-model inference by drafting future tokens that the target model verifies in parallel. A diffusion-style block head such as DFlash is an attractive drafter, predicting an entire block of future tokens in one forward pass. However, it is trained on per-position marginals rather than the joint block distribution, so the tokens it emits are individually plausible yet jointly incoherent. We introduce LiLiCorr, a Lightweight Likelihood-based model that Correlates the per-position marginal distributions a drafter already produces. It keeps the top-k tokens at each position as candidates and processes them jointly, producing for each an in and an out vector. A pair of adjacent candidates matches when the earlier one's out vector has high cosine similarity with the later one's in vector. These matches capture the block's joint structure without ever materializing the full joint distribution. One lightweig
    
[^135]: 缓解大语言模型智能体中的身份本质主义：基于纵向生活轨迹的方法

    Mitigating Identity Essentialism in LLM Agents with Longitudinal Life Trajectories

    [https://arxiv.org/abs/2608.19621](https://arxiv.org/abs/2608.19621)

    本文提出LifeMem框架，通过结合结构化生活事件检索和参数化记忆，缓解大语言模型智能体因静态画像导致的身份本质主义，从而增强社会模拟中的人口多样性。

    

    大语言模型（LLMs）提供了一种可扩展的社会模拟方法，但其可信度取决于智能体的构建方式。现有方法能部分复现群体层面的模式，但往往难以捕捉类人的多样性。我们的分析表明，具有静态画像的智能体在人口统计特征分离和组内压缩方面比人类更强，这一模式与身份本质主义一致：人口统计标签可能促使模型将群体平均倾向视为个体特征，从而在组内同质化响应。我们认为，这一局限源于两个相关因素：稀疏、静态的智能体表征，以及仅提示记忆在持续整合经验方面的有限能力。受互补记忆系统的启发，我们提出了LifeMem，一种纵向记忆框架，结合结构化生活事件检索与智能体特定的参数化记忆，用于经验整合。

    arXiv:2608.19621v1 Announce Type: new  Abstract: Large language models (LLMs) offer a scalable approach to social simulation, but their credibility depends on how agents are constructed. Existing methods can partially reproduce population-level patterns, yet often fail to capture human-like diversity. Our analysis shows that static-profile agents exhibit stronger demographic separation and within-group compression than humans, a pattern consistent with identity essentialism: demographic labels can encourage models to treat group-average tendencies as individual traits, homogenizing responses within groups. We argue that this limitation arises from two related factors: sparse, static agent representations and the limited ability of prompt-only memory to persistently integrate experience. Inspired by complementary memory systems, we propose LifeMem, a longitudinal memory framework that combines structured life-event retrieval with agent-specific parametric memory for experience integrati
    
[^136]: 希腊法律条文检索基准：GreekBarRetrieval

    GreekBarRetrieval: A Benchmark for Greek Statutory Retrieval

    [https://arxiv.org/abs/2608.18752](https://arxiv.org/abs/2608.18752)

    本文提出了希腊法律条文检索基准GreekBarRetrieval，并通过实验发现LLM查询重构能显著提升BM25与密集检索的性能差距。

    

    arXiv:2608.18752v1 公告类型：交叉 摘要：法定条文检索对于基于引用的法律问答系统是必要的，但在希腊语领域仍未得到充分探索。我们引入了GreekBarRetrieval，这是一个公开的检索基准，源自并补充了未包含检索部分的GreekBarBench。该新基准包含283个律师资格考试问题，每个问题附带其涉及案例的事实，以及6,308个可供检索的候选法定条文。问题和事实以日常语言表述，但需要映射到法条的正式术语及其抽象法律概念。另一个复杂之处在于，并非所有案例事实都与案例的每个问题相关。通过实验三种BM25变体和九种密集检索器，我们发现标准密集检索在Recall@100上远优于标准稀疏检索。然而，基于LLM的查询重构帮助BM25缩小了这一差距，同时提升了密集检索的性能。经过十轮...

    arXiv:2608.18752v1 Announce Type: cross  Abstract: Statutory retrieval is necessary for citation-grounded legal question answering, but remains underexplored for Greek. We introduce GreekBarRetrieval, a public retrieval benchmark derived from, and complementing GreekBarBench, which did not include retrieval. The new benchmark comprises 283 bar-exam questions, each accompanied by the facts of the case it refers to, and 6,308 candidate statutory articles to retrieve from. Questions and facts are stated in everyday language, but need to be mapped to the formal terminology of statutes and their abstract legal concepts. A further complication is that not all of the case facts are relevant to each question of a case. Experimenting with three BM25 variants and nine dense retrievers, we find that vanilla dense retrieval far outperforms vanilla sparse retrieval in Recall@100. However, LLM-based query reformulation helps BM25 close that gap, while also improving dense retrieval. With a ten-round
    
[^137]: S$^4$R：面向压缩长上下文KV缓存的选择性采样、子空间与稀疏重建

    S$^4$R: Selective Sampling, Subspaces, and Sparse Reconstruction for Compressed Long-Context KV Caching

    [https://arxiv.org/abs/2608.00528](https://arxiv.org/abs/2608.00528)

    提出S$^4$R方法，通过从选择性采样的token构建低秩子空间，并在稀疏重建的KV表示上计算注意力，在避免校准数据依赖、降低预填充成本与保障解码吞吐量之间取得平衡，实现高效的长上下文KV缓存压缩。

    

    大语言模型（LLM）上下文窗口长度的增长显著增强了其长上下文能力，但键值（KV）缓存带来了高昂的内存成本。尽管对KV缓存进行低秩压缩是一种有前景的解决方案，但现有方法面临两难困境：离线方法依赖外部校准数据，而在线方法则需要为完整提示的分解与重建付出大量计算。在本文中，我们提出S$^4$R，它从选择性采样的token构建低秩子空间，并在稀疏重建的KV表示上计算注意力。S$^4$R采用提示感知的初始化方式，从一个具有代表性的提示子集构建初始的键/值基，从而在校准数据依赖与预填充成本之间进行权衡。由于在每个解码步骤都完整重建缓存的代价极其高昂且会损害吞吐量，我们进一步采用稀疏重建来……（摘要在此处被截断）

    arXiv:2608.00528v2 Announce Type: replace  Abstract: The growth of context window lengths in Large Language Models (LLMs) significantly enhances their long-context capabilities but incurs prohibitive memory costs due to the Key-Value (KV) cache. Although low-rank compression of KV cache is a promising remedy, existing methods face a dilemma: offline approaches depend on external calibration data, whereas online approaches incur substantial compute for full-prompt decomposition and reconstruction. In this paper, we propose S$^4$R, which builds low-rank subspaces from selectively sampled tokens and computes attention over a sparsely reconstructed KV representation. S$^4$R uses prompt-aware initialization to build initial key/value bases from a representative prompt subset, trading off calibration-data dependence against prefilling cost. Because fully reconstructing the cache at every decoding step is prohibitively expensive and hurts throughput, we further adopt sparse reconstruction to 
    
[^138]: Hy-MultiTurn：一个用于深度多轮对话理解的六维基准测试

    Hy-MultiTurn: A Six-Dimensional Benchmark for Deep Multi-Turn Dialogue Understanding

    [https://arxiv.org/abs/2607.29196](https://arxiv.org/abs/2607.29196)

    提出了Hy-MultiTurn，一个基于真实聊天机器人失败分析构建的中文深度多轮对话理解基准，通过约束记忆、精确执行、约束合成、对象定位、行动抑制和指代消解这六种受控评估模式，系统评估模型在长多轮交互中的关键能力。

    

    与聊天机器人和智能体进行长程多轮交互如今已十分普遍，而给出正确的回应往往取决于记住先前的细节、追踪后续的修改、识别所指的对象或指代物，以及在所需条件未满足时抑制行动。现有的多轮对话基准通常只涵盖简短的交流，无法在长多轮交互中全面评估这些能力，尤其是在中文场景下，同时对模型失败的方式和原因所能提供的洞察也十分有限。为了解决这些局限，我们分析了真实的聊天机器人失败案例，识别出六种反复出现的失败机制，并据此在 Hy-MultiTurn 中定义了六种受控评估模式——这是一个面向深度多轮对话理解的中文基准。这六种模式分别评估约束记忆、精确执行、约束合成、对象定位、行动抑制和指代消解。在这六种模式下，我们构建了 209 个受控任务……

    arXiv:2607.29196v2 Announce Type: replace  Abstract: Long-running multi-turn interactions with chatbots and agents are now common, and a correct response often depends on remembering earlier details, tracking later revisions, identifying intended objects or referents, and withholding action when required conditions are unmet. Existing multi-turn benchmarks typically cover short exchanges and do not fully evaluate these capabilities in long multi-turn interactions, particularly in Chinese, while offering limited insight into how and why models fail. To address these limitations, we analyze real chatbot failures to identify six recurring mechanisms and use them to define six controlled evaluation modes in Hy-MultiTurn, a Chinese benchmark for deep multi-turn dialogue understanding. The six modes evaluate constraint memory, precise execution, constraint synthesis, object localization, action suppression, and reference resolution. Across the six modes, we construct 209 controlled tasks spa
    
[^139]: DFAH-Bench：金融决策中可观测智能体不稳定性的基准测试

    DFAH-Bench: Benchmarking Observable Agent Instability in Financial Decision-Making

    [https://arxiv.org/abs/2607.20491](https://arxiv.org/abs/2607.20491)

    该论文提出DFAH-Bench基准，通过同时衡量决策一致性与工具路径一致性，揭示金融智能体在决策结果高度稳定（约95%）的情况下，其背后的工具执行过程却存在显著不稳定性（一致性仅约50%）。

    

    金融智能体可以在重复做出相同决策的同时，改变其背后的工作过程。DFAH-Bench 将确定性-忠实性保障框架（DFAH）付诸实践，在相同的合格重放中同时配对决策一致性与工具路径一致性，然后将该资格认定原则扩展到证据、授权、执行和任务结果。回顾性与前瞻性重放分析揭示了稳定决策背后的过程变化。在570个符合条件的前瞻性情节中，决策一致性为94.2-95.1%，而有序工具、参数和结果的一致性仅为45.0-51.5%；其中一个分层比预先规定的覆盖最低值低了一组。一项独立的捕获诊断显示，系统性的遗漏仍可保持完美的重放一致性。使用 τ-Knowledge 银行环境，我们在采用开源权重与前沿生成器的不同队列中保留了1,080个计划情节和1,033个已知原生结果。

    arXiv:2607.20491v3 Announce Type: replace-cross  Abstract: A financial agent can repeat a decision while changing the work behind it. DFAH-Bench operationalizes the Determinism--Faithfulness Assurance Harness (DFAH), pairing decision agreement with tool-path agreement on the same qualified replays, then extends that qualification principle to evidence, authorization, execution and task outcomes. Retrospective and prospective replay analyses expose process variation behind stable decisions. Across 570 eligible prospective episodes, decision agreement is 94.2-95.1%, while agreement on ordered tools, arguments and results is 45.0-51.5%; one stratum falls one group below its prespecified coverage minimum. A separate capture diagnostic shows that systematic omissions can preserve perfect replay agreement. Using the $\tau$-Knowledge banking environment, we retain 1,080 scheduled episodes and 1,033 known native outcomes across separate cohorts with open-weight and frontier generators. Missing
    
[^140]: 从看似合理到可操作：关于大语言模型自我解释的立场论文

    From Plausible to Actionable: A Position on LLM Self-Explanations

    [https://arxiv.org/abs/2607.15957](https://arxiv.org/abs/2607.15957)

    本立场论文指出大语言模型的自我解释虽高度合理但忠实性存疑，主张评估标准应从合理性与忠实性扩展到可操作性，并为此提供了实用的评估指南。

    

    大语言模型（LLM）能够生成用自然语言对其自身决策进行合理化说明的解释，这种现象通常被称为“自我解释”。此类解释已成为可解释人工智能（XAI）中一个有前景的研究方向，尤其在解释大语言模型行为方面。然而，尽管自我解释往往看起来合理，但它们是否忠实地反映了模型底层的推理过程仍是一个悬而未决的问题。在这篇观点论文中，我们提出：自我解释可以高度合理、忠实性却存疑，但同时又具有高度的可操作性。从传统XAI的视角出发，我们指出了针对LLM生成的自我解释的标准评估协议的局限性，并提出了评估其合理性与忠实性的实用指南。此外，我们主张评估应超越这些标准、扩展到可操作性维度，并重点阐述了LLM合理化解释的应用（摘要此处被截断）。

    arXiv:2607.15957v2 Announce Type: replace  Abstract: Large Language Models (LLMs) can generate natural language explanations that rationalize their own decisions, a phenomenon commonly referred to as self-explanations. Such explanations have emerged as a promising direction for explainable artificial intelligence (XAI), particularly for interpreting LLM behavior. However, while self-explanations often appear plausible, whether they faithfully reflect a model's underlying reasoning process remains an open question. In this opinion paper, we argue that self-explanations can be highly plausible, questionably faithful, and yet highly actionable. From a traditional XAI perspective, we identify the limitations of standard evaluation protocols for LLM-generated self-explanations and propose practical guidelines for assessing their plausibility and faithfulness.Moreover, we argue that evaluation should extend beyond these criteria to actionability, highlighting applications of LLM rationalizat
    
[^141]: ReasonLab：面向多选题问答的提示技术的可控且可审计的评估

    ReasonLab: A Controlled and Auditable Evaluation of Prompting Techniques for Multiple-Choice QA

    [https://arxiv.org/abs/2607.14109](https://arxiv.org/abs/2607.14109)

    该论文提出了ReasonLab评估框架，将提示技术作为与模型和数据集并列的一等实验变量，并保留全部生成记录以供审查，通过对8种提示技术、10个MCQA数据集、27种模型配置共480,927次温度为0的评估的受控研究，实现了对提示技术效果的可控且可审计的评估。

    

    探究大型语言模型（LLM）的能力以及为多选题问答（MCQA）构建稳健的解决方案，仍然是自然语言理解中的核心挑战。此外，大型语言模型的快速普及造成了一种隐含的假设，即更复杂的提示技术会带来更好的性能。一些研究声称获得了这样的收益，但它们是在不同的模型、提示措辞和答案提取规则下报告结果的，因此这些收益无法单独归因于提示技术本身。我们通过 ReasonLab 弥补了这一空白：ReasonLab 是一个评估框架，其中提示技术作为与模型和数据集并列的一等实验变量，并保留每一次生成结果以供检查。利用 ReasonLab，我们在温度为 0 的条件下，对 8 种提示技术在 10 个 MCQA 数据集、27 种模型配置上开展了受控研究，共进行了 480,927 次评估。我们发现，提示技术是一种

    arXiv:2607.14109v2 Announce Type: replace  Abstract: Probing the capabilities of Large Language Models (LLMs) and building robust solutions for Multiple-Choice Question Answering (MCQA) remain central challenges in natural language understanding. Furthermore, the rapid proliferation of LLMs has created the implicit assumption that more sophisticated prompting techniques yield better performance. Several studies claim such gains, but report them under differing models, prompt wordings and answer-extraction rules, so the gains cannot be attributed to the technique alone. We address this gap with ReasonLab, an evaluation framework in which the prompting technique is a first-class experimental variable alongside the model and the dataset, and which retains every generation for inspection. Using ReasonLab we conduct a controlled study of 8 prompting techniques across 10 MCQA datasets, 27 model configurations and 480,927 evaluations at temperature 0. We find that the prompting technique is a
    
[^142]: 低秩注意力残差

    Low-Rank Attention Residuals

    [https://arxiv.org/abs/2607.09694](https://arxiv.org/abs/2607.09694)

    提出低秩注意力残差（LR-AttnRes），通过将每个值的最后 $r$ 维（$r<d$）用作路由键并保留全维残差值，在降低计算成本的同时，于 1B 和 4B 参数规模下以 $r=d/4$ 取得更低的验证损失和更高的下游准确率。

    

    注意力残差在大语言模型（LLM）中用对先前子层输出的深度注意力取代了固定的残差求和，但它将每个输出同时用作全维度的键和值。这使得路由与表示耦合在一起，并导致计算深度路由分数的成本随隐藏宽度 $d$ 而扩展。我们提出低秩注意力残差，它在保留全维度残差值的同时，使用 $r$ 维键（$r < d$）进行路由。LR-AttnRes 将每个值的最后 $r$ 个维度用作路由键，在降低残差侧总浮点运算量的同时仍能提升性能。对块数量（$N$）和 $r$ 的全面扫描表明，深度路由可以在远小于模型宽度的维度下依然有效。在 1B 和 4B 参数规模下，当 $r = d/4$ 时，LR-AttnRes 实现了更低的最终验证损失、更高的平均下游准确率以及更高的（摘要在此处截断）

    arXiv:2607.09694v2 Announce Type: replace-cross  Abstract: Attention Residuals (AttnRes) replace the fixed residual sum with depth-wise attention over previous sub-layer outputs in Large Language Models (LLMs), but use each output as both a full-dimensional key and value. This couples routing with representation and makes the cost of computing depth-routing scores scale with hidden width $d$. We propose Low-Rank Attention Residuals (LR-AttnRes), which keep full-dimensional residual values while using $r$-dimensional keys, with $r < d$, for routing. LR-AttnRes uses the last $r$ dimensions of each value as the routing key, reducing total residual-side FLOPs while still improving performance. Comprehensive sweeps across the number of blocks ($N$) and $r$ show that depth-wise routing can be effective with far fewer dimensions than the model width. At both $1$B and $4$B parameters with $r = d/4$, LR-AttnRes achieves lower final validation loss, higher average downstream accuracy, and higher
    
[^143]: 面向特应性皮炎的具有稳定性与边界感知的解释引导医学命名实体识别

    Explanation-Guided Medical Named Entity Recognition with Stability and Boundary Awareness for Atopic Dermatitis

    [https://arxiv.org/abs/2606.22886](https://arxiv.org/abs/2606.22886)

    该论文提出了一种稳定性和边界感知的解释引导NER框架，通过自适应融合局部与全局解释，并利用稳定性、边界感知和一致性约束将解释信号融入模型训练，显著提升了中文特应性皮炎临床文本中医学命名实体识别的可靠性与鲁棒性。

    

    目标：本研究旨在通过解释引导学习，提高中文特应性皮炎（AD）临床文本中医学命名实体识别（NER）的可靠性与鲁棒性。方法：我们提出了一种具有稳定性和边界感知的解释引导NER框架。采用基于扰动的分析方法来评估解释稳定性和实体边界敏感性；提出一种自适应融合策略，动态结合局部解释与全局解释，以生成更可靠的词元级解释；融合后的解释信号进一步通过稳定性、边界感知和一致性约束融入模型训练。结果：在中文AD NER数据集上的实验表明，所提出的框架提升了解释的鲁棒性，并在多个NER模型上取得了一致的性能提升；与单独使用解释相比，自适应融合策略还能提供更稳定的解释和更强的边界感知能力。

    arXiv:2606.22886v2 Announce Type: replace  Abstract: Objective: This study aims to improve the reliability and robustness of medical named entity recognition (NER) in Chinese atopic dermatitis (AD) clinical texts through explanation-guided learning. Methods: We propose a stability and boundary-aware explanation-guided NER framework. Perturbation-based analysis is used to evaluate explanation stability and entity boundary sensitivity. An adaptive fusion strategy dynamically combines local and global explanation to generate more reliable token-level explanations. The fused explanation signals are further incorporated into model training through stability, boundary-aware, and consistency constraints. Results: Experiments on Chinese AD NER datasets show that the proposed framework improves explanation robustness and achieves consistent performance gains across multiple NER models. The adaptive fusion strategy also provides more stable explanations and stronger boundary perception than indi
    
[^144]: KaLM-Reranker-V1：面向压缩文档重排序的快速但非晚期交互方法

    KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking

    [https://arxiv.org/abs/2606.22807](https://arxiv.org/abs/2606.22807)

    KaLM-Reranker-V1 是一种快速但非晚期交互的重排序器，通过编码器-解码器架构将查询与段落计算解耦，利用 Matryoshka 嵌入池化离线预编码段落，并借助交叉注意力捕捉细粒度相关性，在保持强表达能力的同时显著提升了部署效率与灵活性。

    

    随着检索系统规模的不断扩大，高效且有效的重排序变得日益重要。然而，大多数现有的基于编码器和解码器的重排序器都会联合处理每一个查询-段落对，将其在线计算紧密耦合，从而限制了部署的效率与灵活性。我们提出了 KaLM-Reranker-V1，一种快速但非晚期交互的 FBNL 重排序器，它在解耦查询与段落计算的同时，保留了富有表现力的相关性建模能力。KaLM-Reranker-V1 构建于编码器-解码器架构之上，利用 Matryoshka（套娃）嵌入池化对段落进行预编码，同时其解码器将系统指令、用户指令与查询意图共同建模；随后通过交叉注意力捕捉所得查询上下文与段落表示之间的细粒度相关性。这些设计共同带来了四大关键优势：(i) 离线段落编码带来的高效性，(ii) 交叉注意力带来的强表达能力，(iii) Matryoshka 嵌入带来的紧凑性

    arXiv:2606.22807v3 Announce Type: replace  Abstract: As retrieval systems scale, effective and efficient reranking becomes increasingly important. However, most existing encoder- and decoder-based rerankers jointly process every query--passage pair, tightly coupling their online computation and limiting deployment efficiency and flexibility. We present KaLM-Reranker-V1, a fast but not late-interaction FBNL reranker that decouples query and passage computation while retaining expressive relevance modeling. Built on an encoder--decoder architecture, KaLM-Reranker-V1 pre-encodes passages using Matryoshka embedding pooling, while its decoder models system and user instructions together with query intent; cross-attention then captures fine-grained relevance between the resulting query context and passage representations. Together, these designs offer four key advantages: (i) efficiency from offline passage encoding, (ii) expressiveness from cross-attention, (iii) compactness from Matryoshka
    
[^145]: PreUnlearn：在大语言模型遗忘之前审计附带性知识损害

    PreUnlearn: Auditing Collateral Knowledge Damage Before Large Language Model Unlearning

    [https://arxiv.org/abs/2606.18473](https://arxiv.org/abs/2606.18473)

    该论文提出 PreUnlearn，首次在大语言模型执行遗忘之前审计遗忘集可能造成的附带知识损害，发现损害随语义距离衰减但不消失于领域边界，并证明可利用数据特征（如交互特征）提前预测下游损害。

    

    面向大语言模型（LLM）的机器遗忘旨在移除指定知识的同时保留模型的其他能力。然而，待遗忘知识与待保留知识之间的边界往往并不清晰，因为相关甚至相距较远的信息可能在模型内部相互纠缠。本文从以数据为中心的视角研究大语言模型遗忘，并测量遗忘效应如何从遗忘集传播到同域及远域知识——而且是在遗忘执行之前而非之后进行测量。我们发现了一致的衰减模式：附带损害在靠近遗忘集处最强，随语义距离增大而减弱，但在领域边界处并不会消失。我们进一步追问：这种损害能否在执行遗忘之前被审计？我们将遗忘集审计形式化为一个遗忘前预测任务，并分析哪些数据特征对下游损害最具预测性。我们的结果表明，交互特征……（原文摘要在此处截断）

    arXiv:2606.18473v2 Announce Type: replace  Abstract: Machine unlearning for large language models (LLMs) aims to remove specified knowledge while preserving the rest of the model's capabilities. However, the boundary between knowledge to forget and knowledge to retain is often unclear, since related and even distant information may be entangled in the model. In this paper, we study LLM unlearning from a data-centric perspective and measure how unlearning effects propagate from the forget set to same-domain and distant-domain knowledge not after but before unlearning. We find a consistent decay pattern: collateral damage is strongest near the forget set, weakens with semantic distance, but does not disappear at domain boundaries. We further ask whether such damage can be audited before unlearning is executed. We formulate forget-set auditing as a pre-unlearning prediction task and analyze which data features are most predictive of downstream damage. Our results show that interaction fea
    
[^146]: 在无监督词条发现中恢复齐夫分布

    Recovering the Zipfian Distribution in Unsupervised Term Discovery

    [https://arxiv.org/abs/2606.10781](https://arxiv.org/abs/2606.10781)

    该论文提出用基于图的Leiden聚类替代K-means等基于中心的方法，在无监督词条发现中显著恢复了真实词库所具有的齐夫分布特性。

    

    无监督词条发现是指将无标注语音切分为类似词或音节的单元，并将这些单元聚类为一个候选词条类型的词库。真实的词库遵循齐夫分布，然而主流的基于中心的聚类方法——K-means——由于对球形簇的归纳偏置，会产生更均匀的分布。在本文中，我们重新审视基于图的聚类作为一种自底向上的替代方案，该方法通过成对相似度连接分段嵌入，并使用Leiden算法进行划分。我们表明，在三种语言的词级和音节级词库发现任务中，图聚类显著优于基于中心的方法，并产生更接近齐夫分布的结果。另一种自底向上的方法——采用平均链接的层次聚类——也表现良好，尽管其计算效率较低，且对最终结果的控制能力较弱。

    arXiv:2606.10781v2 Announce Type: replace-cross  Abstract: Unsupervised term discovery involves segmenting unlabelled speech into word- or syllable-like units and clustering these into a lexicon of candidate types. True lexicons follow a Zipfian distribution, yet the dominant centre-based clustering approach -- K-means -- produces a more uniform distribution due to an inductive bias toward spherical clusters. In this paper we revisit graph-based clustering as a bottom-up alternative, where segment embeddings are connected by pairwise similarity and partitioned using the Leiden algorithm. We show that graph clustering substantially outperforms centre-based approaches (K-means, GMM, BIRCH) in both word- and syllable-level lexicon discovery across three languages, producing more Zipf-like distributions. Another bottom-up approach, agglomerative clustering with average linkage, also performs well, although it is computationally less efficient and allows for less control over the resulting 
    
[^147]: 重新拟合探针：单方向消融并非必要性检验

    Refit the Probe: Single-Direction Ablation Is Not a Necessity Test

    [https://arxiv.org/abs/2606.00926](https://arxiv.org/abs/2606.00926)

    单方向消融并不能真正移除探针检测到的信息（它会保留在正交补空间中），因此消融后的任务准确率变化不能作为模型是否依赖该信息的必要性检验，只需重新拟合一次探针即可揭示这一点。

    

    探针通常与干预配对使用：消融探针所发现的方向，运行模型，并读取任务准确率的变化——大幅下降被视为计算依赖于探针所读取内容的证据，而近乎为零的下降则被视为不依赖的证据。这两种推断都要求消融已经从该层中移除了目标。我们发现，消融并未移除它所针对的内容。在消融后的激活值上重新拟合的探针在我们测试的每一个设置中都恢复了原始准确率，而且即使删除探针的整个行空间而不仅仅是单个轴，它依然能够恢复，因为该信息量在正交补空间中得以存留。由于重新拟合的探针能够恢复，因此无论是任务准确率大幅下降还是几乎不变，都无法确立模型是否真正需要该目标，而仅需一次探针拟合即可检测到这一点。用迭代零空间投影替代消融，并以维度匹配的随机子空间作为对照进行评分……（摘要原文在此处截断）

    arXiv:2606.00926v2 Announce Type: replace-cross  Abstract: Probes are routinely paired with an intervention: ablate the direction the probe found, run the model, and read the change in task accuracy, taking a large drop as evidence that the computation depends on what the probe read and a near-zero drop as evidence that it does not. Either inference requires that the ablation have removed the target from the layer. We find that the ablation does not remove what it targets. A probe refitted on the ablated activations recovers its original accuracy in every cell we test, and keeps recovering when the probe's entire row space is deleted rather than a single axis, because the quantity survives in the orthogonal complement. Because a refitted probe recovers, neither a large task drop nor a near-zero one establishes whether the model needed the target, and one probe fit detects this. Replacing the ablation with iterative nullspace projection, scored against random subspaces of matched dimens
    
[^148]: CONCAT：基于共识与置信度驱动的临时组队方法，用于高效的基于大语言模型的多智能体系统

    CONCAT: Consensus- and Confidence-Driven Ad Hoc Teaming for Efficient LLM-Based Multi-Agent Systems

    [https://arxiv.org/abs/2605.29612](https://arxiv.org/abs/2605.29612)

    提出了一种无需训练的多智能体协作框架CONCAT，通过基于共识的智能体聚类、基于置信度的领导者选择以及基于心智理论的启发式函数来高效组织LLM多智能体交互，在降低计算开销的同时保持性能与泛化能力。

    

    尽管基于大语言模型（LLM）的多智能体系统（MAS）展现出解决复杂任务的能力，并且相比单智能体系统能够取得更高的性能，但由于智能体之间繁重的通信，它们会带来巨大的计算开销。以往的研究尝试通过训练稀疏的多智能体图或微调规划器来更好地编排工作流。然而，这些额外的训练过程引入了计算成本，并将多智能体系统限制在特定领域，从而损害了其泛化能力。在本文中，我们提出了CONCAT，一个无需训练的多智能体协作框架，基于共识（CONsensus）与置信度驱动的临时组队来高效地组织智能体之间的交互。具体而言，智能体根据其初始答案进行聚类，并基于智能体的置信度选出每个簇的领导者。然后，设计了一个基于心智理论的启发式函数来预……（摘要在此处截断）

    arXiv:2605.29612v2 Announce Type: replace-cross  Abstract: Although large language model (LLM) based multi-agent systems (MAS) show their capability to solve complex tasks and achieve higher performance over single agent systems, they lead to huge computational overheads because of heavy communication between agents. Previous research has made efforts to train a sparse multi-agent graph or fine-tune a planner to orchestrate the workflow better. However, such extra training processes introduce computational costs and limit MAS to specific domains, therefore compromising their generalizability. In this paper, we propose CONCAT, a training-free multi-agent collaboration framework based on CONsensus and Confidence-driven Ad hoc Teaming to efficiently organize agent interactions. Specifically, agents are clustered based on their initial answers, and leaders of each cluster are selected based on the agents' confidence. Then, a heuristic function based on the Theory of Mind is designed to pre
    
[^149]: MONA：用于可扩展语言模型训练的带Nesterov加速的Muon优化器

    MONA: Muon Optimizer with Nesterov Acceleration for Scalable Language Model Training

    [https://arxiv.org/abs/2605.26842](https://arxiv.org/abs/2605.26842)

    MONA通过在Muon优化器的梯度处理流程中引入基于梯度差指数移动平均的Nesterov加速项，在保持谱范数正则化的同时实现曲率感知加速，在1B至68B参数规模的MoE预训练中超越了Muon和AdamW的收敛性与下游任务性能。

    

    Muon优化器最近为大语言模型训练提供了一种有前景的AdamW替代方案，它利用矩阵正交化来产生几何感知的更新。然而，与所有一阶方法一样，Muon可能会陷入尖锐的局部极小值。在本工作中，我们提出了MONA，这是一种将Muon的正交化框架与曲率感知加速相结合的优化器。MONA在Muon的梯度处理流程中直接加入了一个加速项，该项由梯度差的指数移动平均计算得出。我们对MONA进行了详细的收敛性分析，表明该加速项在保持Muon谱范数正则化的同时，引入了曲率敏感的修正。实验表明，在从1B到68B参数的三个规模的混合专家（MoE）预训练中，MONA相比Muon和AdamW均取得了更好的收敛性和下游任务性能。

    arXiv:2605.26842v2 Announce Type: replace-cross  Abstract: The Muon optimizer has recently offered a promising alternative to AdamW for large language model training, leveraging matrix orthogonalization to produce geometry-aware updates. However, like all first-order methods, Muon can become trapped in sharp local minima. In this work, we present MONA, an optimizer that bridges Muon's orthogonalization framework with curvature-aware acceleration. MONA adds an acceleration term directly into Muon's gradient processing pipeline. This term is calculated from the exponential moving average of gradient differences. We provide a detailed convergence analysis for MONA, showing that the acceleration term introduces curvature-sensitive corrections while preserving Muon's spectral-norm regularization. Empirically, MONA achieves better convergence and downstream task performance compared to both Muon and AdamW across three scales of Mixture-of-Experts pretraining, spanning from 1B to 68B paramete
    
[^150]: GroupTravelBench：多人旅行规划中大语言模型智能体基准测试

    GroupTravelBench: Benchmarking LLM Agents on Multi-Person Travel Planning

    [https://arxiv.org/abs/2605.25200](https://arxiv.org/abs/2605.25200)

    GroupTravelBench是首个面向多用户、多轮旅行规划的基准测试，基于真实用户画像、POI数据和票价构建了650个跨三个难度级别的任务，用于评估大语言模型智能体在发现多用户偏好、揭示冲突以及平衡效用与公平等群体特有能力上的表现。

    

    现实世界中的旅行规划绝大多数是一种“群体”活动，然而现有的LLM旅行规划基准将其简化为单一用户场景，而该领域在这一设定下已接近饱和。这种单用户假设回避了群体规划对智能体而言真正的难点：发现多个用户的私人偏好、揭示冲突，以及在效用与公平之间进行权衡。为了让任务回归其多用户的现实本质，我们提出了GroupTravelBench，这是首个面向多用户、多轮对话的旅行规划基准。该基准基于真实用户画像、POI（兴趣点）数据和票价数据构建，包含跨越三个难度级别的650个任务，每个任务都在一个同步群聊沙箱中运行，并配有缓存的工具数据，以实现可复现的离线评估。除了单用户基准已经测试的多步推理和工具使用之外，GroupTravelBench还探测了三种群体特有的能力：(i) e...（原文摘要在此处被截断）

    arXiv:2605.25200v3 Announce Type: replace  Abstract: Travel planning in the real world is overwhelmingly a \textit{group} activity, yet existing LLM travel-planning benchmarks reduce it to a single user, where the field is approaching saturation. This single-user assumption sidesteps what makes group planning hard for an agent: discovering private preferences across multiple users, surfacing conflicts, and balancing utility against fairness. To bring the task back to its multi-user reality, we introduce \textbf{\textit{GroupTravelBench}}, the first benchmark for \textbf{multi-user, multi-turn} travel planning. Built from real user profiles, POI data, and ticket prices, it comprises 650 tasks across three difficulty levels, each running in a synchronous group-chat sandbox with cached tool data for reproducible offline evaluation. Beyond the multi-step reasoning and tool use that single-user benchmarks already test, GroupTravelBench probes three group-specific capabilities: \textit{(i) e
    
[^151]: 大语言模型捉鬼敢死队：通过自适应遗忘实现精准的包幻觉抑制

    LLM Ghostbusters: Surgical Package Hallucination Suppression via Adaptive Unlearning

    [https://arxiv.org/abs/2605.01047](https://arxiv.org/abs/2605.01047)

    提出自适应遗忘（AU）框架，通过混合 token 级目标函数在部署后精准抑制大语言模型的包幻觉，防范 slopsquatting 供应链攻击，同时保持模型的通用能力。

    

    幻觉仍然是大语言模型（LLM）尚未解决的问题，而包幻觉是这一现象中尤为危险的实例。包幻觉发生在代码生成过程中，当模型捏造不存在的软件包时，会为虚构的库推荐导入语句和安装命令。这造成了一个关键的供应链漏洞：攻击者可以在公共软件包注册中心抢先注册此类包并植入恶意载荷，随后这些包会被开发者或自主智能体安装并执行。这些幻觉使一类被称为"slopsquatting"（垃圾包抢注）的包混淆攻击成为可能。为解决这一问题，我们提出了自适应遗忘，这是一种部署后框架，能够在保持模型通用能力的同时，精准地抑制包幻觉。AU 引入了一种混合的 token 级目标函数，能够同时强化有效输出并抑制幻觉输出。

    arXiv:2605.01047v2 Announce Type: replace-cross  Abstract: Hallucinations remain an unsolved problem for LLMs, and package hallucinations are a particularly dangerous instance of this phenomenon. Package hallucinations occur during code generation when a model fabricates non-existent software packages, recommending imports and installation commands for fictional libraries. This creates a critical supply-chain vulnerability; an attacker can proactively register such packages on public registries with malicious payloads that are subsequently installed and executed by developers or autonomous agents. These hallucinations enable a class of package confusion attack known as slopsquatting. To address this issue, we present Adaptive Unlearning (AU), a post-deployment framework that surgically suppresses package hallucinations while preserving general model utility. AU introduces a hybrid token-level objective that simultaneously reinforces valid outputs and suppresses hallucinated ones. Combi
    
[^152]: 通过往返验证与修复实现忠实的自动形式化

    Faithful Autoformalization via Roundtrip Verification and Repair

    [https://arxiv.org/abs/2604.25031](https://arxiv.org/abs/2604.25031)

    本文提出一种无需真实标注的往返验证与修复框架，通过“形式化—回译—再形式化—逻辑等价检查”来验证LLM自动形式化的忠实性，并利用阶段级诊断定位错误、以范围受限修复算子加以纠正，在法律条文形式化任务上证明诊断引导的修复方法最为有效。

    

    当大语言模型（LLM）将自然语言形式化时，我们如何知道其输出是忠实的？我们提出了一种无需真实标注（ground-truth annotations）的往返验证方法：先将一个陈述形式化，再将结果翻译回自然语言，然后重新形式化，最后使用形式化工具检查两者的逻辑等价性。当两个形式化结果一致时，这为形式化的忠实性提供了证据；当两者不一致时，阶段级诊断会将错误定位到特定的翻译步骤，随后由一个范围受限的修复算子尝试纠正该步骤。我们在两个法律领域（德克萨斯州交通法和德克萨斯州公园与野生动物法）上，使用两个大语言模型（Claude Opus 4.6 和 GPT-5.2）以及三种修复基线方法对该框架进行了评估。结果表明，诊断引导的范围受限修复是最有效的方法，其有效性取决于诊断函数的可靠性。在两个领域和两个模型上，在我们的完整修复系统……（摘要原文在此处被截断）

    arXiv:2604.25031v3 Announce Type: replace  Abstract: When an LLM formalizes natural language, how do we know the output is faithful? We propose a roundtrip verification approach which does not require ground-truth annotations: formalize a statement, translate the result back to natural language, re-formalize, and use a formal tool to check logical equivalence. When the two formalizations agree, this provides evidence of a faithful formalization. When they disagree, a stage-level diagnosis localizes the error to a specific translation step, and a scoped repair operator attempts to correct that step. We evaluate the framework on two statutory domains (the Texas Transportation Code and the Texas Parks and Wildlife Code) using two LLMs (Claude Opus~4.6 and GPT-5.2) with three repair baselines. Diagnosis-guided scoped repair is the most effective method, with effectiveness contingent on the reliability of the diagnosis function. Across both domains and both models, under our full repair sys
    
[^153]: LLM智能体长期记忆安全综述：跨记忆生命周期的攻击、防御与治理

    A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle

    [https://arxiv.org/abs/2604.16548](https://arxiv.org/abs/2604.16548)

    本文提出记忆生命周期框架与可验证记忆治理（VMG）框架，系统刻画了LLM智能体长期记忆所面临的持久性、有状态性与传播性等新型安全威胁，并沿六个生命周期阶段和四个安全目标组织了攻击、防御及其跨阶段依赖关系的全景式分析。

    

    LLM智能体中可写入、可跨会话持久化记忆的出现，带来了与传统的以输入为中心的安全关注点在性质上截然不同的威胁格局，其特征体现为三种属性：持久性、有状态性和传播性。为了系统地刻画这一威胁格局，我们提出了一个记忆生命周期框架，沿两个维度对攻击、防御及其跨阶段依赖关系进行组织：六个生命周期阶段（写入、存储、检索、执行、共享与传播、遗忘与回滚）和四个安全目标（完整性、机密性、可用性、治理）。这一分析进而揭示了在系统层面需要形式化安全保证的需求，由此催生了可验证记忆治理——一个由五个架构原语构成的框架，它规定了长期记忆系统必须提供哪些可验证机制，以维持对其记忆状态可审计、可恢复的控制。我们的

    arXiv:2604.16548v3 Announce Type: replace-cross  Abstract: The emergence of writable, cross-session persistent memory in LLM agents introduces a qualitatively different threat landscape from conventional input-centric security concerns, characterized by three properties: persistence, statefulness, and propagation. To systematically characterize this landscape, we propose a Memory Lifecycle Framework that organizes attacks, defenses, and their cross-phase dependencies along two axes: six lifecycle phases (Write, Store, Retrieve, Execute, Share & Propagate, Forget & Rollback) and four security objectives (Integrity, Confidentiality, Availability, Governance). This analysis in turn exposes the need for formal security guarantees at the system level, motivating Verifiable Memory Governance (VMG), a framework of five architectural primitives that specifies what verifiable mechanisms a long-term-memory system must provide to maintain auditable, recoverable control over its memory state. Our 
    
[^154]: Co-FactChecker：一个基于大型推理模型的人机协同声明验证框架

    Co-FactChecker: A Framework for Human-AI Collaborative Claim Verification Using Large Reasoning Models

    [https://arxiv.org/abs/2604.13706](https://arxiv.org/abs/2604.13706)

    提出了Co-FactChecker框架，通过将模型思维轨迹作为共享草稿板、并将专家反馈转化为对轨迹的定向编辑，实现了人机协同的声明验证，弥合了专家主导与全自动验证之间的差距。

    

    专业事实核查人员依靠领域知识和深入的情境理解来验证声明。大型语言模型（LLMs）和大型推理模型（LRMs）缺乏这种基础，主要仅从现有证据进行推理，导致专家主导的声明验证与全自动声明验证之间存在不匹配。为了弥合这一差距，我们认为人机协作是一条更有前景的前进道路，其中基于现实世界知识和领域专业知识的专家反馈可以引导模型的推理。然而，现有的LRMs难以根据自然语言反馈进行校准，特别是在多轮交互设置中。我们提出了Co-FactChecker，一个用于人机协同声明验证的框架。我们引入了一种新的交互范式，将模型的思维轨迹视为共享的草稿板。Co-FactChecker将专家反馈转化为轨迹编辑，对思维轨迹进行有针对性的修改，从而避开了……

    arXiv:2604.13706v2 Announce Type: replace  Abstract: Professional fact-checkers rely on domain knowledge and deep contextual understanding to verify claims. Large language models (LLMs) and large reasoning models (LRMs) lack such grounding and primarily reason from available evidence alone, creating a mismatch between expert-led and fully automated claim verification. To mitigate this gap, we posit human-AI collaboration as a more promising path forward, where expert feedback, grounded in real-world knowledge and domain expertise, guides the model's reasoning. However, existing LRMs are hard to calibrate to natural language feedback, particularly in a multi-turn interaction setup. We propose Co-FactChecker, a framework for human-AI collaborative claim verification. We introduce a new interaction paradigm that treats the model's thinking trace as a shared scratchpad. Co-FactChecker translates expert feedback into trace-edits that introduce targeted modifications to the trace, sidesteppi
    
[^155]: 面向毒理学决策支持的诊断推理学习

    Learning Diagnostic Reasoning for Decision Support in Toxicology

    [https://arxiv.org/abs/2603.29608](https://arxiv.org/abs/2603.29608)

    本文提出DeToxR，首个将强化学习（通过GRPO微调大语言模型）应用于急诊毒理学决策支持的方法，通过融合非结构化叙述与结构化医疗数据，实现对14类物质中毒的多标签预测。

    

    急性多物质中毒需要在高度不确定性下做出快速的、挽救生命的决策，因为临床医生必须依赖不完整的摄入细节和非特异性的症状。在这种混乱环境中进行有效的诊断推理，需要将非结构化的、非医学的叙述（如急救人员的现场描述、不可靠的患者自述或已知病史）与结构化的医学数据（如生命体征）相融合。虽然大语言模型（LLM）在处理此类异构输入方面显示出潜力，但它们在这种场景下表现不佳，往往不如仅依赖患者病史的简单基线方法。为解决这一问题，我们提出了DeToxR（基于推理的毒理学决策支持），这是首个将强化学习（RL）应用于急诊毒理学的尝试。我们设计了一个鲁棒的数据融合引擎，基于经组相对策略优化（GRPO）微调的大语言模型，实现对14类物质的多标签预测。

    arXiv:2603.29608v2 Announce Type: replace  Abstract: Acute poly-substance intoxication requires rapid, life-saving decisions under substantial uncertainty, as clinicians must rely on incomplete ingestion details and nonspecific symptoms. Effective diagnostic reasoning in this chaotic environment requires fusing unstructured, non-medical narratives (e.g. paramedic scene descriptions and unreliable patient self-reports or known histories), with structured medical data like vital signs. While Large Language Models (LLMs) show potential for processing such heterogeneous inputs, they struggle in this setting, often underperforming simple baselines that rely solely on patient histories. To address this, we present DeToxR (Decision-support for Toxicology with Reasoning), the first adaptation of Reinforcement Learning (RL) to emergency toxicology. We design a robust data-fusion engine for multi-label prediction across 14 substance classes based on an LLM finetuned with Group Relative Policy Op
    
[^156]: 放射报告生成中的校准置信度表达

    Calibrated Confidence Expression for Radiology Report Generation

    [https://arxiv.org/abs/2603.29492](https://arxiv.org/abs/2603.29492)

    提出ConRad强化学习框架，通过微调医学视觉语言模型，使其在生成放射报告的同时输出校准的言语化置信度估计，从而支持放射科医生的选择性验证并降低幻觉发现影响临床决策的风险。

    

    在放射报告生成中安全部署大型视觉语言模型（LVLMs）不仅需要准确的预测，还需要临床可解释的指标来指示何时应对输出进行彻底审查，从而实现放射科医生的选择性验证，并降低幻觉发现影响临床决策的风险。一种直观的方法是言语化置信度，即模型明确表达其确定性。然而，当前最先进的语言模型往往过于自信，而针对放射报告生成等多模态场景的校准研究仍然有限。为了填补这一空白，我们提出了ConRad（放射报告置信度校准），这是一个强化学习框架，用于微调医学LVLMs，使其在生成放射报告的同时产生校准的言语化置信度估计。我们研究了两种设置：单一的报告级置信度分数和句子级变（摘要在此处被截断）。

    arXiv:2603.29492v2 Announce Type: replace  Abstract: Safe deployment of Large Vision-Language Models (LVLMs) in radiology report generation requires not only accurate predictions but also clinically interpretable indicators of when outputs should be thoroughly reviewed, enabling selective radiologist verification and reducing the risk of hallucinated findings influencing clinical decisions. One intuitive approach to this is verbalized confidence, where the model explicitly states its certainty. However, current state-of-the-art language models are often overconfident, and research on calibration in multimodal settings such as radiology report generation is limited. To address this gap, we introduce ConRad (Confidence Calibration for Radiology Reports), a reinforcement learning framework for fine-tuning medical LVLMs to produce calibrated verbalized confidence estimates alongside radiology reports. We study two settings: a single report-level confidence score and a sentence-level varian
    
[^157]: VeriSoftBench：面向Lean的仓库级形式化验证基准

    VeriSoftBench: Repository-Scale Formal Verification Benchmarks for Lean

    [https://arxiv.org/abs/2602.18307](https://arxiv.org/abs/2602.18307)

    VeriSoftBench是一个包含500个证明义务的仓库级Lean 4形式化验证基准，评估发现专为Mathlib数学调优的证明器难以迁移到以仓库为中心的软件验证场景，且任务成功率与其传递性依赖闭包的规模密切相关。

    

    大型语言模型在交互式定理证明领域取得了显著成果，尤其是在Lean中。然而，大多数针对基于LLM的证明自动化的基准都取自Mathlib生态系统中的数学内容，而软件验证中的证明则是在包含丰富定义和大量项目专用库的代码库中开发的。我们提出了VeriSoftBench，这是一个包含500个Lean 4证明义务的基准，这些证明义务取自开源形式化方法项目，并经过打包处理以保留真实的仓库上下文和跨文件依赖关系。我们对前沿LLM和专业证明器的评估得出了三项观察结果。首先，针对Mathlib风格数学调优的证明器在这种以仓库为中心的环境中迁移效果不佳。其次，成功与传递性仓库依赖密切相关：其证明依赖于大型、多跳依赖闭包的任务更不容易被解决。第三，提供精选的上下文……

    arXiv:2602.18307v2 Announce Type: replace-cross  Abstract: Large language models have achieved striking results in interactive theorem proving, particularly in Lean. However, most benchmarks for LLM-based proof automation are drawn from mathematics in the Mathlib ecosystem, whereas proofs in software verification are developed inside definition-rich codebases with substantial project-specific libraries. We introduce VeriSoftBench, a benchmark of 500 Lean 4 proof obligations drawn from open-source formal-methods developments and packaged to preserve realistic repository context and cross-file dependencies. Our evaluation of frontier LLMs and specialized provers yields three observations. First, provers tuned for Mathlib-style mathematics transfer poorly to this repository-centric setting. Second, success is strongly correlated with transitive repository dependence: tasks whose proofs draw on large, multi-hop dependency closures are less likely to be solved. Third, providing curated cont
    
[^158]: FMMD：来自F1000Research的多模态多学科开放同行评议数据集

    FMMD: A multimodal multidisciplinary dataset of open peer reviews from F1000Research

    [https://arxiv.org/abs/2602.14285](https://arxiv.org/abs/2602.14285)

    本文提出了FMMD数据集，通过收录F1000Research的多模态、多学科开放同行评议数据，并保留评审意见与稿件版本的精确对应关系，弥补了现有同行评议数据集以文本为中心、学科覆盖单一且缺乏版本对齐的不足。

    

    自动化学术论文评审（ASPR）已进入与传统同行评议共存的阶段，人工智能（AI）系统正日益被纳入真实世界的稿件评估之中。与此同时，关于自动化和AI辅助同行评议的研究也在迅速增多。尽管发展势头强劲，实证进展仍受到现有数据集若干关键局限的制约。虽然审稿人通常会评估图表和复杂版面以判断科学论断，但大多数现有数据集仍高度以文本为中心。这种偏见又因数据过度集中于计算机科学出版物而进一步加剧。此外，现有数据集很少保留评审意见与特定稿件版本之间的精确对应关系，从而模糊了同行评议与稿件演进之间的迭代关系。为此，我们推出了FMMD，一个多模态、多学科的开放同行评议数据集。

    arXiv:2602.14285v2 Announce Type: replace-cross  Abstract: Automated scholarly paper review (ASPR) has entered the coexistence phase with traditional peer review, where artificial intelligence (AI) systems are increasingly incorporated into real-world manuscript evaluation. In parallel, research on automated and AI-assisted peer review has proliferated. Despite this momentum, empirical progress remains constrained by several critical limitations in existing datasets. While reviewers routinely evaluate figures, tables, and complex layouts to assess scientific claims, most existing datasets remain overwhelmingly text-centric. This bias is reinforced by a narrow focus on data from computer science publications. Furthermore, existing datasets rarely preserve precise alignment between review comments and specific manuscript versions, obscuring the iterative relationship between peer review and manuscript evolution. In response, we introduce FMMD, a multimodal and multidisciplinary open peer
    
[^159]: 面向语言模型不确定性的语义自蒸馏

    Semantic Self-Distillation for Language Model Uncertainty

    [https://arxiv.org/abs/2602.04577](https://arxiv.org/abs/2602.04577)

    该论文提出语义自蒸馏方法，将语言模型采样答案的语义分布蒸馏到轻量级学生模型中，使其能在生成答案前预测语义分布，利用分布的熵和概率密度分别提供提示级和答案级的不确定性信号，从而以低计算成本实现高效的不确定性估计与幻觉检测。

    

    大型语言模型给原则性的不确定性量化带来了挑战，部分原因在于其复杂性以及输出的多样性。语义离散度（即采样答案在含义上的方差）已被提出作为模型不确定性的有效代理指标，但其相关的计算成本使其无法应用于对延迟敏感的场景。我们证明了采样的语义分布可以被蒸馏到轻量级的学生模型中，该模型能够在语言模型生成答案token之前估计以提示为条件的密度。学生模型预测可能答案的语义分布；该分布的熵提供了提示层面的不确定性信号，而概率密度则支持答案层面的可靠性评估。在TriviaQA和MMLU上的实验表明，我们的学生模型在幻觉检测方面与教师模型的采样语义离散度相比具有竞争力。

    arXiv:2602.04577v3 Announce Type: replace  Abstract: Large language models present challenges for principled uncertainty quantification, in part due to their complexity and the diversity of their outputs. Semantic dispersion, or the variance in the meaning of sampled answers, has been proposed as a useful proxy for model uncertainty, but the associated computational cost prohibits its use in latency-critical applications. We show that sampled semantic distributions can be distilled into lightweight student models which estimate a prompt-conditioned density before the language model generates an answer token. The student model predicts a semantic distribution over possible answers; the entropy of this distribution provides a prompt-level uncertainty signal, and the probability density allows answer-level reliability evaluation. Across experiments on TriviaQA and MMLU, we find our student models perform competitively relative to the teacher's sampled semantic dispersion on a hallucinatio
    
[^160]: CausalEmbed：面向视觉文档嵌入的潜空间自回归多向量生成方法

    CausalEmbed: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding

    [https://arxiv.org/abs/2601.21262](https://arxiv.org/abs/2601.21262)

    提出 CausalEmbed 自回归生成方法，通过迭代间隔损失训练使视觉文档检索仅需数十个视觉标记即可构建多向量嵌入，在保持竞争力的同时将标记数量减少30-155倍。

    

    尽管多模态大语言模型（MLLMs）通过生成高质量的多向量嵌入在视觉文档检索（VDR）中展现出显著潜力，但用数千个视觉标记来表示一页文档所带来的巨大存储开销限制了其在实际应用中的实用性。为应对这一挑战，我们提出了一种自回归生成方法 CausalEmbed，用于构建多向量嵌入。通过在对比训练中引入迭代间隔损失（iterative margin loss），CausalEmbed 促使嵌入模型学习到紧凑且结构良好的表示。我们的方法仅需数十个视觉标记即可高效完成视觉文档检索任务，在保持各种骨干网络和基准测试上高度竞争力的同时，将标记数量减少了30-155倍。理论分析和实证结果证明了自回归嵌入生成的独特优势。

    arXiv:2601.21262v4 Announce Type: replace  Abstract: Although Multimodal Large Language Models (MLLMs) have shown remarkable potential in Visual Document Retrieval (VDR) through generating high-quality multi-vector embeddings, the substantial storage overhead caused by representing a page with thousands of visual tokens limits their practicality in real-world applications. To address this challenge, we propose an auto-regressive generation approach, CausalEmbed, for constructing multi-vector embeddings. By incorporating iterative margin loss during contrastive training, CausalEmbed encourages the embedding models to learn compact and well-structured representations. Our method enables efficient VDR tasks using only dozens of visual tokens, achieving a 30-155x reduction in token count while maintaining highly competitive performance across various backbones and benchmarks. Theoretical analysis and empirical results demonstrate the unique advantages of auto-regressive embedding generatio
    
[^161]: 通过文本去噪实现基于大语言模型的语音识别系统的纯文本自适应

    Text-only adaptation in LLM-based ASR through text denoising

    [https://arxiv.org/abs/2601.20900](https://arxiv.org/abs/2601.20900)

    该论文提出将纯文本领域自适应转化为文本去噪任务，通过训练LLM从带噪输入中恢复干净转录文本，在无需架构更改或额外参数的情况下保留跨模态对齐，实现了最高22.1%的相对性能提升。

    

    仅使用文本数据将基于大语言模型（LLM）的自动语音识别（ASR）系统适配到新领域是一个重要但尚未充分探索的挑战。在目标领域文本上对LLM进行标准微调，往往会破坏投影器学习到的语音与文本模态之间的关键对齐，从而导致性能下降。我们提出了一种新颖的纯文本自适应方法，将该过程构建为文本去噪任务。我们的方法训练LLM从带噪输入中恢复干净的转录文本。这一过程在有效将模型适配到目标领域的同时，保留了跨模态对齐。我们的方案十分轻量，无需架构更改或额外参数。在两个数据集上的广泛评估显示，相对提升最高达22.1%，优于近期最先进的纯文本自适应方法。

    arXiv:2601.20900v4 Announce Type: replace-cross  Abstract: Adapting large language model (LLM)-based automatic speech recognition (ASR) systems to new domains using text-only data is a significant yet underexplored challenge. Standard fine-tuning of the LLM on the target domain text often disrupts the critical alignment between the speech and text modality learned by the projector, degrading performance. We introduce a novel text-only adaptation method that frames this process as a text denoising task. Our approach trains the LLM to recover clean transcripts from noisy inputs. This process effectively adapts the model to a target domain while preserving cross-modal alignment. Our solution is lightweight, requiring no architectural changes or additional parameters. Extensive evaluation on two datasets demonstrates up to 22.1% relative improvement, outperforming recent state-of-the-art text-only adaptation methods.
    
[^162]: POPI：通过优化的自然语言偏好推断实现大语言模型个性化

    POPI: Personalizing LLMs via Optimized Natural Language Preference Inference

    [https://arxiv.org/abs/2510.17881](https://arxiv.org/abs/2510.17881)

    POPI提出一个用户级个性化框架，通过自然语言偏好摘要连接共享的推断模型与生成器，在统一的偏好优化目标下同时实现准确的个性化生成和信息丰富的偏好总结，且摘要可跨任务复用。

    

    大语言模型（LLMs）通常与群体层面的偏好对齐，尽管个体用户之间存在显著差异。我们提出了POPI，一个用户级个性化框架，该框架将问题分解为通过自然语言接口连接的两个组件：一个共享的推断模型，将异构的用户信号提炼成简洁的偏好摘要；以及一个共享的生成器，以该摘要为条件生成个性化响应。两个组件在统一的偏好优化目标下进行训练，其中强化学习用于处理不可微的推断步骤。该目标可分解为生成器近似误差和摘要信息量两个部分，揭示了单一损失如何同时驱动准确的生成和信息丰富的摘要。由于接口是自然语言，学习到的摘要可以每个用户仅推断一次，并在不同的……（原文截断）

    arXiv:2510.17881v4 Announce Type: replace  Abstract: Large language models (LLMs) are typically aligned with population-level preferences, despite substantial variation across individual users. We introduce POPI, a user-level personalization framework that separates the problem into two components connected by a natural-language interface: a shared inference model that distills heterogeneous user signals into a concise preference summary, and a shared generator that conditions on this summary to produce personalized responses. Both components are trained under a unified preference-optimization objective, with reinforcement learning handling the non-differentiable inference step. This objective decomposes into generator approximation error and summary informativeness, revealing how a single loss simultaneously drives accurate generation and informative summarization. Because the interface is natural language, learned summaries can be inferred once per user and reused across different ge
    
[^163]: 用于检测和纠正大语言模型幻觉的几何不确定性方法

    Geometric Uncertainty for Detecting and Correcting Hallucinations in LLMs

    [https://arxiv.org/abs/2509.13813](https://arxiv.org/abs/2509.13813)

    该论文提出了一个黑盒几何框架，通过在答案嵌入空间中建模以提示为条件的语义分布，同时量化提示和答案两个层面的不确定性，从而实现对大语言模型幻觉的检测与纠正。

    

    大语言模型已知会产生幻觉，即对问题生成语言上看似合理但实际不正确的答案。不确定性量化已被提出作为检测此类行为的策略，但现有方法缺乏一个统一的框架来同时评估提示层面和答案层面的可靠性。我们引入了一个几何框架，通过在答案嵌入空间中显式建模以提示为条件的语义分布，在两个层面量化语言模型的不确定性。我们的方法是黑盒且基于采样的：我们为每个提示生成多个答案，并使用原型分析来估计答案分布的几何支撑。在提示层面，我们通过近似分布熵来量化不确定性；对于每个单独的答案，我们随后使用非典型性的概念来评估其相对于整个批次的可靠性。我们利用该框架不仅能够检测幻觉，还能纠正幻觉。

    arXiv:2509.13813v3 Announce Type: replace  Abstract: Large language models are known to hallucinate, generating linguistically plausible but incorrect answers to questions. Uncertainty quantification has been proposed as a strategy to detect such behaviour, but existing methods lack a unified framework to assess reliability at both the prompt and answer level. We introduce a geometric framework which quantifies language model uncertainty at both levels by explicitly modelling a prompt-conditioned semantic distribution in answer embedding space. Our approach is black-box and sampling-based; we generate multiple answers per prompt, and use archetypal analysis to estimate a geometric support for the answer distribution. At the prompt level, we approximate the distribution entropy to quantify uncertainty; for each individual answer, we then use notions of atypicality to assess its reliability relative to the batch. We employ our framework to not only detect hallucinations but correct them,
    
[^164]: SafetyFlow：一种用于大语言模型安全基准测试自动化的智能体流系统

    SafetyFlow: An Agent-Flow System for Automated LLM Safety Benchmarking

    [https://arxiv.org/abs/2508.15526](https://arxiv.org/abs/2508.15526)

    SafetyFlow是首个用于自动化构建大语言模型安全基准的智能体流系统，通过协调七个专业智能体，可在四天内无需人工干预构建全面的安全基准，大幅降低了时间和资源成本。

    

    大型语言模型（LLM）的快速普及加剧了对可靠安全评估的需求，以便发现模型漏洞。为此，学界提出了众多LLM安全评估基准。然而，现有基准通常依赖劳动密集型的人工整理，导致过多的时间和资源消耗，同时还存在显著的冗余性和有限的难度。为缓解这些问题，我们提出了SafetyFlow，这是首个旨在自动化构建LLM安全基准的智能体流系统。SafetyFlow通过协调七个专业化智能体，可以在仅四天内无需任何人工干预自动构建一个全面的安全基准，显著降低了时间和资源成本。SafetyFlow的智能体配备了多种通用工具，在将人类专业知识融入自动化流程的同时，确保了过程和成本的可控性。最终的cons……（摘要在此处被截断）

    arXiv:2508.15526v2 Announce Type: replace  Abstract: The rapid proliferation of large language models (LLMs) has intensified the requirement for reliable safety evaluation to uncover model vulnerabilities. To this end, numerous LLM safety evaluation benchmarks are proposed. However, existing benchmarks generally rely on labor-intensive manual curation, which causes excessive time and resource consumption. They also exhibit significant redundancy and limited difficulty. To alleviate these problems, we introduce SafetyFlow, the first agent-flow system designed to automate the construction of LLM safety benchmarks. SafetyFlow can automatically build a comprehensive safety benchmark in only four days without any human intervention by orchestrating seven specialized agents, significantly reducing time and resource cost. Equipped with versatile tools, the agents of SafetyFlow ensure process and cost controllability while integrating human expertise into the automatic pipeline. The final cons
    
[^165]: EndoCogniAgent：面向内窥镜诊断的带自洽性验证的闭环智能体推理

    EndoCogniAgent: Closed-Loop Agentic Reasoning with Self-Consistency Validation for Endoscopic Diagnosis

    [https://arxiv.org/abs/2508.07292](https://arxiv.org/abs/2508.07292)

    提出EndoCogniAgent闭环智能体框架，将内窥镜诊断建模为受控状态更新过程，通过自洽性验证机制解决幻觉证据与错误累积问题，提升AI内窥镜诊断的可靠性。

    

    内窥镜诊断是一个迭代过程，临床医生在得出结论之前需要获取、比较和验证局部视觉证据。当前的AI系统未能充分支持这一过程，因为细粒度的证据获取与多步推理之间的耦合仍然薄弱，导致图像所得发现与文本解释之间的协调变得困难。这产生了两种失败模式：幻觉证据和未经纠正的错误累积，它们损害了诊断的可靠性。我们提出了EndoCogniAgent，一个闭环智能体框架，它将内窥镜诊断表述为一个受控的状态更新过程，用于整合互补的视觉与文本证据。在每一轮推理中，中央规划器选择一个证据获取动作，专门的专家工具提取空间和语义观察结果作为结构化文本证据，自洽性验证机制检查……

    arXiv:2508.07292v4 Announce Type: replace-cross  Abstract: Endoscopic diagnosis is an iterative process in which clinicians acquire, compare, and verify local visual evidence before reaching a conclusion. Current AI systems do not adequately support this process because fine-grained evidence acquisition and multi-step reasoning remain weakly coupled, complicating reconciliation of image-derived findings with their textual interpretations. This gives rise to two failure modes, hallucinated evidence and uncorrected error accumulation, that undermine diagnostic reliability. We propose EndoCogniAgent, a closed-loop agentic framework that formulates endoscopic diagnosis as a controlled state update process for integrating complementary visual and textual evidence. At each reasoning round, a central planner selects an evidence acquisition action, specialized expert tools extract spatial and semantic observations as structured textual evidence, and a self-consistency validation mechanism exam
    
[^166]: BigO(Bench)：大语言模型能否生成具有受控时间和空间复杂度的代码？

    BigO(Bench): Can LLMs Generate Code with Controlled Time and Space Complexity?

    [https://arxiv.org/abs/2503.15242](https://arxiv.org/abs/2503.15242)

    提出BigO(Bench)编程基准，通过从性能分析中推断算法复杂度并标注3105道编程题及其119万余个解决方案，来评估大语言模型生成具有受控时间和空间复杂度代码的能力。

    

    我们提出了BigO(Bench)，这是一个新颖的编程基准，旨在评估生成式语言模型在理解和生成具有特定时间和空间复杂度代码方面的能力。该基准弥补了当前评估中往往忽视模型理解和生成受计算复杂度约束代码这一能力的空白。BigO(Bench)包含一套工具，可以从性能分析测量中推断任何Python函数的算法复杂度，包括由人类或大语言模型生成的解决方案。BigO(Bench)还包含来自Code Contests的3,105个编程问题和1,190,250个解决方案，这些方案均标注了由复杂度框架推断出的（合成）时间和空间复杂度标签，以及针对大量输入规模对应的运行时间和内存占用值。我们展示了在该基准上对多个最先进语言模型进行评估的结果，突出了它们的优势与不足。

    arXiv:2503.15242v3 Announce Type: replace  Abstract: We introduce BigO(Bench), a novel coding benchmark designed to evaluate the capabilities of generative language models in understanding and generating code with specified time and space complexities. This benchmark addresses the gap in current evaluations that often overlook the ability of models to comprehend and produce code constrained by computational complexity. BigO(Bench) includes tooling to infer the algorithmic complexity of any Python function from profiling measurements, including human- or LLM-generated solutions. BigO(Bench) also includes of set of 3,105 coding problems and 1,190,250 solutions from Code Contests annotated with inferred (synthetic) time and space complexity labels from the complexity framework, as well as corresponding runtime and memory footprint values for a large set of input sizes. We present results from evaluating multiple state-of-the-art language models on this benchmark, highlighting their streng
    
[^167]: MultiViewDx：证据关联的多视角临床诊断

    MultiViewDx: Evidence-Linked Multi-View Clinical Diagnosis

    [https://arxiv.org/abs/2410.14948](https://arxiv.org/abs/2410.14948)

    提出了部分经医生验证的MultiViewDx数据集，以临床病例为监督单元，将影像与患者背景关联，并通过统一的图文检索器将报告规范化为“证据→发现→鉴别讨论→诊断”的证据关联工作流程，从而构建多视角医学影像诊断指令数据。

    

    医学多模态大语言模型（MLLM）在现有的医学视觉问答（MedVQA）基准测试中表现良好，但其训练数据往往与临床诊断不匹配。大多数监督数据是围绕孤立图像或简短问答对组织的，导致两种结构定义薄弱：证据如何导向决策，以及同一病例中的视角、序列、模态和患者背景如何相互关联。我们提出了MultiViewDx，这是一个部分经医生验证的多模态指令数据集，用于证据关联的多视角医学影像诊断。MultiViewDx以临床病例作为监督单元，将影像检查与患者背景相关联，将异构报告规范化为证据关联的工作流程（证据 -> 发现 -> 鉴别讨论 -> 诊断），并使用统一的图文检索器将指令合成约束在有来源支持的证据上。该数据集涵盖X光、CT、MRI、超声（原文截断）等模态。

    arXiv:2410.14948v2 Announce Type: replace  Abstract: Medical multimodal large language models (MLLMs) can perform well on existing medical visual question answering (MedVQA) benchmarks, but their training data often does not match clinical diagnosis. Most supervision is organized around isolated images or short QA pairs, leaving two structures weakly specified: how evidence leads to a decision, and how views, series, modalities, and patient context from the same case are linked. We introduce MultiViewDx, a partly physician-validated multimodal instruction dataset for evidence-linked multi-view medical imaging diagnosis. MultiViewDx uses the clinical case as the supervision unit. It links imaging studies with patient context, normalizes heterogeneous reports into an evidence-linked workflow (evidence -> findings -> differential discussion -> diagnosis), and uses a unified image-text retriever to constrain instruction synthesis to source-supported evidence. It covers X-ray, CT, MRI, ultr
    
[^168]: 治疗作为一项NLP任务：比较LLM与人类同伴在CBT治疗过程中的行为

    Therapy as an NLP Task: Comparing LLMs and Human Peers Behaviors in CBT Sessions

    [https://arxiv.org/abs/2409.02244](https://arxiv.org/abs/2409.02244)

    本研究通过18个月的民族志研究和一种新颖的治疗过程生成方法，首次实现了LLM与人类同伴咨询师在匹配条件下进行多轮单次CBT治疗时的直接受控行为比较。

    

    大语言模型（LLM）正越来越多地被用作临时治疗师。虽然先前的研究发现LLM在生成单轮共情回应方面优于人类咨询师，但很少有研究比较它们在多轮治疗过程中的行为。在本研究中，我们比较了人类同伴咨询师与LLM的治疗过程级行为，两者均基于同一手册训练，以提供多轮、单次认知行为疗法（CBT）。我们的三阶段混合方法研究包括：(a) 对一个同伴支持平台进行的18个月民族志研究，七名咨询师通过110次自我咨询过程和60次每周焦点小组迭代完善CBT提示词；(b) 一种新颖的治疗过程生成方法，允许在匹配条件下对人类和LLM咨询师进行直接、受控的比较——来访者的回应取自公开可得的人类主导CBT治疗过程，而咨询师的回应则由……

    arXiv:2409.02244v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly being used as ad hoc therapists. While prior research has found that LLMs outperform human counselors in generating single-turn empathetic responses, fewer studies have compared their behaviors across multi-turn sessions. In this study, we compare the session-level behaviors of human peer counselors with those of an LLM, both trained on the same manual to deliver multi-turn, single-session Cognitive Behavioral Therapy (CBT). Our three-phase, mixed-methods study involved: (a) an 18-month ethnography of a peer support platform, where seven counselors iteratively refined CBT prompts through 110 self-counseling sessions and 60 weekly focus groups; (b) a novel session generation method that allows direct, controlled comparison of human and LLM counselors under matched conditions---client responses were drawn from publicly available human-led CBT sessions while counselor responses were ge
    
[^169]: DA-Cramming：通过集成依存一致性来增强低成本高效的语言模型预训练

    DA-Cramming: Enhancing Cost-Effective Language Model Pretraining with Dependency Agreement Integration

    [https://arxiv.org/abs/2311.04799](https://arxiv.org/abs/2311.04799)

    提出DA-Cramming框架，开创性地在预训练阶段（而非微调阶段）将依存一致性语义信息融入模型，实现低成本高效的语言模型预训练。

    

    由于巨大的计算成本，预训练语言模型对许多研究者来说仍然是一个挑战。因此，开发更经济实惠的预训练方法日益受到关注。该领域的一个显著进展是Cramming技术（Geiping和Goldstein，2022），它使得仅用一块GPU就能在一天内完成BERT风格语言模型的预训练。基于这一创新方法，我们提出了依存一致性Cramming（DA-Cramming），这是一个将依存一致性信息整合到预训练过程中的高效框架。与在微调阶段利用类似语义信息的现有方法不同，我们的方法是一项开创性的工作，专注于在预训练阶段利用语义信息增强基础语言理解能力。我们精心设计了一个双阶段预训练工作流程，包含四个专用子模型来捕获代表性……

    arXiv:2311.04799v3 Announce Type: replace  Abstract: Pretraining language models is still a challenge for many researchers due to its substantial computational costs. As such, there is growing interest in developing more affordable pretraining methods. One notable advancement in this area is the Cramming technique (Geiping and Goldstein, 2022), which enables the pretraining of BERT-style language models using just one GPU in a single day. Building on this innovative approach, we introduce the Dependency Agreement Cramming (DA-Cramming), an efficient framework that integrates information about dependency agreements into the pretraining process. Unlike existing methods that leverage similar semantic information during finetuning, our approach represents a pioneering effort focusing on enhancing the foundational language understanding with semantic information during pretraining. We meticulously design a dual-stage pretraining work flow with four dedicated submodels to capture representat
    

