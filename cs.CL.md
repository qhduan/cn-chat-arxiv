# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Direct Pass@$k$ Optimization for Thought Pruning: Balancing Quality and Diversity in Test-Time Reasoning](https://arxiv.org/abs/2610.31667) | 该论文提出将 Pass@$k$ 直接作为思维剪枝的优化目标，从而在测试时推理中显式地平衡质量与多样性（即探索与利用）之间的权衡。 |
| [^2] | [Cost-Aware Abstention for LLM-Based IT Ticket Classification](https://arxiv.org/abs/2610.31657) | 该论文提出一种离线的成本感知弃权评估框架，通过在校准数据上按显式成本选择并冻结路由策略，在真实IT工单数据上显著降低分类成本与错误路由率，并表明简单的TF-IDF逻辑回归结合成本感知阈值可优于大语言模型方案。 |
| [^3] | [Low-Bit N-Gram Embeddings for Language Models](https://arxiv.org/abs/2610.31646) | 提出将n-gram嵌入表量化至四比特低精度存储，在从零训练的nanochat、Llama和GPT-2系列模型上节省约75%的表内存，同时保留超过99%的验证损失改善效果。 |
| [^4] | [Can an Agent Debug a Web Corpus? Agentic Auditing of Slovak FineWeb2](https://arxiv.org/abs/2610.31636) | 本文提出将 LLM 作为一次性的“语料库调试器”而非分类器：由工具调用智能体审计 FineWeb2 斯洛伐克语子集，发现了包括 140 万篇机器生成内容农场在内的七项质量问题，并将发现编译为可执行的测试套件，从而以不到 3 美元的成本实现可复用、可验证的大规模语料库质量审计。 |
| [^5] | [Bilingual Language Models Capture Cross-Linguistic Influence in Second-Language Reading](https://arxiv.org/abs/2610.31620) | 该研究发现，使用与读者母语相匹配的双语语言模型能更好地预测二语（英语）阅读的眼动时间，这一优势主要体现在后期眼动指标中，受两种语言文字系统重叠程度的影响，并在中级语言水平的读者中最为显著。 |
| [^6] | [How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions](https://arxiv.org/abs/2610.31606) | 该论文提出了一种证据感知的检索评估协议，通过将证据片段映射到所有包含块来定义相关性，从而解决检索得分依赖真值标注覆盖范围的问题，并将其应用于乳腺病变管理文献临床RAG系统的评估。 |
| [^7] | [Would an LLM Match into a Residency? Uncertainty, Reasoning and Image Use of Open Language Models on the Spanish MIR Examination](https://arxiv.org/abs/2610.31586) | 该研究按西班牙MIR住院医师考试的官方规则评估了四个开放大语言模型，发现三个较大模型的成绩可排进前600至3,000名并仍能匹配到41至43个专业，但其弃答策略从不划算，医学微调可减少过度自信，而其临床推理与图像利用能力仍存疑。 |
| [^8] | [Diffusion Language Models for Perplexity-Aware Black-Box Attacks](https://arxiv.org/abs/2610.31558) | 该论文提出将连续潜空间攻击框架与连续扩散语言模型（dLLM）直接耦合，以学习到的语言生成分布取代人工随机映射，从而生成更自然、更难被防护机制检测且基于自然提示的黑盒攻击提示词。 |
| [^9] | [An Empirical Study of Recent Large Language Models for Knowledge Graph to Text Generation on Standard Benchmarks](https://arxiv.org/abs/2610.31529) | 该研究在标准基准上系统对比了最新大语言模型与微调编码器-解码器模型在知识图谱到文本生成任务中的表现，发现LLM在基于参考的指标上大幅缩小甚至在部分设置下追平了差距，且在基于图谱的评估中产生更少的矛盾和三元组遗漏。 |
| [^10] | [The Geometry of Metaphor: Probing and Steering Metaphor Representations in Large Language Models](https://arxiv.org/abs/2610.31520) | 该研究利用机制可解释性技术在大语言模型中定位隐喻的因果子空间，实现对分类和生成任务中隐喻输出的引导与控制。 |
| [^11] | [Towards User Simulators That Stay On Task And Still Sound Natural](https://arxiv.org/abs/2610.31512) | 本文提出一种强化学习方法，通过在交替对抗循环中训练轻量级判别器来将行为自然度可操作化，从而同时奖励任务意图遵循与对话自然度，并发布了由此得到的用户模拟器模型 UserLM-2.0。 |
| [^12] | [Measuring the Verifier's Floor: How Much Apparent Rare-Word Text-to-Speech Error Is ASR Error?](https://arxiv.org/abs/2610.31470) | 该研究量化了ASR评估系统本身的错误下限——即使在人类语音上也会漏掉33-67%的极罕见词——表明TTS评估中相当一部分“错误”实为识别器局限，多识别器交叉验证可将表面错误率降低两到三倍，但最终仍需人工听音测试裁定。 |
| [^13] | [From Structured Records to Patient Narratives: Preserving Clinical Alignment in LLM-Generated Health Summaries](https://arxiv.org/abs/2610.31467) | 本文提出一种将结构化证据条件化与检索增强生成（RAG）相结合的提示约束生成框架，用于月经健康监测的可读摘要生成，并通过NLI、原子事实性和可读性评估证明，显式证据约束能够持续提升大语言模型生成摘要的临床语义一致性。 |
| [^14] | [How Different, or Different How? Stance Granularity in Viewpoint Selection for Dialog Systems that Engage Users with Alternative Views](https://arxiv.org/abs/2610.31459) | 本文提出一种由多臂老虎机驱动的观点选择策略，将立场表示从标量分数细化为概念层面的“视角化立场向量”，发现概念级立场能让用户接触到更多样、覆盖更广的观点，但不会改变用户论证和重构观点的深度与准确性。 |
| [^15] | [Multi-Token Prediction and Alternative Tokenization Improve Learning Latent Multi-Hop Graph Search at Scale](https://arxiv.org/abs/2610.31439) | 本文指出模型难以学习大规模图上最短路径搜索的两个原因——监督信号与分词方式，并提出多Token预测和替代分词这两种相互正交的方法，突破了以往工作的规模瓶颈。 |
| [^16] | [How Well Do Decision Models Transfer to Vietnamese? An Empirical Study of Text Classification](https://arxiv.org/abs/2610.31422) | 本文在13个越南语文本分类设置中评估了四种专用决策模型，发现JEV在英语标签描述下表现最佳，双语标签描述并无一致提升，而去除声调符号对模型性能的影响最为严重。 |
| [^17] | [How Depth Growth and Looping Shape In-Context Learning in Language Models](https://arxiv.org/abs/2610.31421) | 深度增长与循环语言模型通过跨层复用计算形成了偏向从上下文学习的归纳偏置，使其比稠密模型更早获得且更强保持上下文学习能力，其机制在于深度增长将早期归纳头传播到各层。 |
| [^18] | [SyRA-CS: Synthetic-to-Real Adaptation for Code-Switched ASR with Meta-Learning](https://arxiv.org/abs/2610.31415) | 提出基于Reptile元学习的SyRA-CS框架，利用合成码切换语音学习易于自适应的模型初始化，仅需一小时真实目标语音即可在所有16个模型-语言对组合上超越持续微调的码切换语音识别性能。 |
| [^19] | [SQuAT: Spherical Lattice Vector Quantization-Aware Training](https://arxiv.org/abs/2610.31412) | 该论文提出SQuAT方法，通过分离的直通估计器将权重分解为方向（用E_8球面码本量化）和增益（用低比特标量量化器量化），将基于格的向量量化感知训练成功扩展到十亿参数级大语言模型，并实现了隐式的球面（黎曼）优化。 |
| [^20] | [Same Entity, Different Script: Diagnosing Cross-Script Entity Matching Failures in Language Models](https://arxiv.org/abs/2610.31407) | 语言模型判断不同文字书写的同一人名是否为同一人的能力存在巨大差距，同一人判断准确率从拉丁文的35.1%骤降至天城文的8.5%和马拉雅拉姆语的5.3%，该效应在多个模型、两种文字和语系中均存在，且无法通过减少分词碎片化来弥补。 |
| [^21] | [HERD: Harness-Evolving Recursive Distillation for Continual Learning](https://arxiv.org/abs/2610.31401) | 提出HERD混合持续学习框架，通过可演化的工具链高效学习，并以双锚点漂移约束蒸馏将经验内化为参数化累积能力，从而在持续获取新知识的同时大幅减少遗忘。 |
| [^22] | [Parameter-Efficient Adaptation of Large Language Models to a Low-Resource Agglutinative Language: A Corpus, Datasets, and a Reproducible Benchmark for the Bashkir Language](https://arxiv.org/abs/2610.31397) | 该论文发布了巴什基尔语的开放语料库、标注数据集及系统性适配基准，证明基于QLoRA的参数高效适配结合持续预训练，可使3.8-8B参数的大语言模型在该低资源黏着语的文本分类任务上显著超越传统编码器模型微调。 |
| [^23] | [A Task-Centric Survey of Long-Document Visual Question Answering: Taxonomy, Evolution, and Future Directions](https://arxiv.org/abs/2610.31385) | 该论文提出首个以任务为中心的长文档视觉问答综述，将其分解为文档解析、证据组织与检索、生成器组织与学习及评估四个组件，建立细粒度分类体系，并通过时间维度分析梳理研究重点从多模态内容理解到证据获取与管理的演进脉络及未来方向。 |
| [^24] | [AfterDeleteBench: Auditing Post-Deletion Information Persistence in LLM Agents](https://arxiv.org/abs/2610.31366) | 本文提出AfterDeleteBench受控审计基准，揭示了LLM智能体删除原始记忆记录后，其派生状态（如摘要、分散存储的组件）仍可在约三分之二的案例中导致已删除的秘密信息被恢复，证明“删除”操作并不能保证信息真正被移除。 |
| [^25] | [CrisiLangs: A Multilingual Resource for Crisis Warning Generation](https://arxiv.org/abs/2610.31356) | 该论文提出了CrisiLangs——将英语危机警告数据集CrisiText扩展至意大利语、保加利亚语、德语和希腊语的多语言资源，包含五种语言共5,000个事件-警告信息对，可用于微调多语言模型以在危机期间生成多语言警告信息。 |
| [^26] | [SurveyGraph: Generating Academic Surveys via Methods-Entity Knowledge Graphs](https://arxiv.org/abs/2610.31317) | SurveyGraph通过构建编码方法级后续、竞争与资源共享关系的“方法-实体知识图谱”（MEKG）来生成学术综述，使图感知检索的召回率相比扁平文本基线近乎翻倍，并显著提升了生成内容的引用支持率。 |
| [^27] | [When LLMs Forget What They Believe: Characterizing State Integrity Collapse in Long-Horizon Dialogues](https://arxiv.org/abs/2610.31314) | 该论文首次定义并量化了长程对话中的“状态完整性崩塌”现象——大语言模型虽在记忆中保留了约束却停止执行——并通过StateTrack-Bench基准与分叉式记忆审计证明，绝大多数失败（88.6%）是行为漂移而非普通遗忘。 |
| [^28] | [Cola2: Co-Adaptive Hierarchical Language Modeling with Continuous Semantic Compression and Balanced Discrete Generation](https://arxiv.org/abs/2610.31305) | Cola2 提出了一种协同自适应的分层语言建模框架，通过动态语义分段和无辅助损失的负反馈来控制连续语义压缩率，并结合块因果掩码扩散解码器与无分类器引导，实现语义重建与自主生成之间的平衡。 |
| [^29] | [The Value of Memory in Language Agents](https://arxiv.org/abs/2610.31303) | 该论文提出一个决策论框架，以所避免的任务损失为单位，量化语言智能体中记忆从原始历史到持久状态、操作使用及代价高昂的恢复与修正全过程中的价值，并通过成本感知的模拟条件和五个最小反例揭示信息时机、检索访问、组件交互、反馈和需求不确定性对记忆价值的关键影响。 |
| [^30] | [CoWriter: Guiding Methodological and Synergetic Human-LLM Co-Authoring](https://arxiv.org/abs/2610.31277) | CoWriter是一种主动式人机协同写作智能体方案，通过借鉴认知科学中的专家写作方法论来引导端到端的完整写作流程并积极促使用户深度参与，从而显著提升作者的观点表达、文档主人翁感与文本质量。 |
| [^31] | [Memory-Aware Reward Modeling: Learning Human Preferences for LLM Memory Use](https://arxiv.org/abs/2610.31272) | 该研究通过对87名用户约1000条真实偏好判断的分析，首次揭示了大语言模型使用记忆并不总能生成用户偏好的回复，并提出了一种记忆感知奖励模型，以72.8%的准确率预测用户对记忆使用的偏好，填补了记忆评估领域的空白。 |
| [^32] | [A Systematic Comparison of Search Space Complexity and Efficiency for Time-Synchronous End-to-End ASR Models](https://arxiv.org/abs/2610.31259) | 该论文系统比较了CTC和神经转换器等时间同步端到端语音识别模型在不同剪枝策略和语言模型配置下的搜索空间复杂度与效率，发现端到端模型多数时候搜索空间较小但会在不确定时出现假设数量峰值，而基于得分的动态剪枝可适应模型不确定性并显著降低实时率。 |
| [^33] | [IndoNutriQA: An Indonesian Clinical Nutrition Question Answering Benchmark](https://arxiv.org/abs/2610.31256) | 该论文提出了IndoNutriQA——首个由专家构建的印度尼西亚临床营养问答基准，包含基于真实临床营养报告的911个高质量问答对及结构化元数据，旨在填补低资源语言在临床问答评估资源上的空白。 |
| [^34] | [Predicting Music Valence and Arousal via Textual and Multimodal In-Context Learning in Large Language Models](https://arxiv.org/abs/2610.31202) | 该研究提出利用大型语言模型的上下文学习（包括基于文本的多模态方法），通过上下文示例改进小型基础模型的初始预测，显著提升了音乐情绪效价与唤醒度的连续预测性能，其中效价R²从0.110提升至0.427。 |
| [^35] | [Investigating Multi-Cultural Value Selection Biases in LLMs](https://arxiv.org/abs/2610.31199) | 该研究揭示了LLM在多重文化身份线索下存在价值观选择偏差：语言线索对答案的偏移是显式国家提示的三倍但反而偏离目标国家的价值观分布，而显式命名国家能缩小与该国价值观分布差距的30-44%并产生更明显的价值观变化。 |
| [^36] | [When More Metadata Misleads: Probing Field-Level Grounding in LLM-Generated Museum Descriptions](https://arxiv.org/abs/2610.31193) | 该研究通过两阶段实验发现，元数据数量本身并不能提升大语言模型生成博物馆文物描述的可靠性，真正决定模型是接地、省略还是凭空推断信息的是字段类型，而图像既能支持视觉接地也可能助长无依据的解读。 |
| [^37] | [Prompting Strategy Effects Depend on the Judge: A Multilingual Study of LLM-as-a-Judge Robustness](https://arxiv.org/abs/2610.31172) | 该研究通过涵盖9种提示策略、4个开源评判模型、27种语言的大规模基准测试（超过100万条评分），首次发现LLM-as-a-Judge评估中的提示策略效果不可跨模型和语言迁移——评判模型的选择比提示策略的选择更具决定性，且策略效果在不同评判模型间会反向。 |
| [^38] | [Beyond Watching and Searching: Benchmarking Agents for Video-centered Deep Research Reports](https://arxiv.org/abs/2610.31158) | 该论文提出以视频为中心的深度研究报告生成任务，并构建OmniDR-Bench基准与Omni-ART评估体系（涵盖视听定位、多模态证据使用和交互式呈现三个维度），用于系统评估智能体基于视频检索证据并生成交互式研究报告的能力。 |
| [^39] | [Prompt Sensitivity in Tool-Augmented Model Comparisons](https://arxiv.org/abs/2610.31148) | 论文提出一种将答案读取、推理、调用决策等环节分离的匹配协议，发现工具虽能提升模型准确率，却会加剧模型间比较对问题表述方式的敏感性。 |
| [^40] | [Mixing Matters: Evidence Position Bias Across Sequence Mixers in Long-Context Question Answering](https://arxiv.org/abs/2610.31135) | 本研究通过在干扰文档中移动黄金证据，系统比较了Transformer、状态空间和混合Mamba架构在长上下文问答中的证据位置偏差，发现Transformer模型具有状态空间模型所缺乏的开头优势，而两类架构均偏好接近末尾的证据。 |
| [^41] | [Delta N-gram: Forward-Only Continual Writing into a Language Model's Native N-gram Memory](https://arxiv.org/abs/2610.31125) | 提出Delta N-gram方法，通过冻结预训练N-gram表、将证据以仅前向传播的方式写入稀疏补丁Δ，实现无需反向传播的持续知识写入，清除补丁即可完全恢复原始模型，且事实保留效果远优于基于梯度的写入方法。 |
| [^42] | [Through a PRISM, Clearly: Step-Sequence Reasoning for Mistake Recognition in Procedural Videos](https://arxiv.org/abs/2610.31117) | 提出PRISM框架，通过将视频片段锚定于语言并利用在正确执行上训练的轻量级隐马尔可夫模型建模步骤序列，实现对程序性视频中步骤内容和顺序错误的可解释识别，并统一了跨数据集的四种基于编辑的错误类型评估标准。 |
| [^43] | [A Grammar of French Civil Cassation: A Ground-Level Dataset and Tool](https://arxiv.org/abs/2610.31116) | 该研究通过确定性流程构建了法国最高法院民事庭86,464份判决中121,536对“上诉理由-法院回应”的配对数据集，并将其归入既定教义学类别，形成了一部揭示民事撤销审上诉成败规律及其十年演变的描述性“语法”。 |
| [^44] | [Learning What to Ask for Tomorrow: Active Preference Acquisition for Long-Term Personalized Web Agents](https://arxiv.org/abs/2610.31107) | 提出主动偏好获取（APA）框架，通过前瞻性个性化价值（PPV）衡量缺失偏好在未来任务中的累积效用增益并扣除提问成本与披露风险，使网络代理能在交互预算下智能选择该向用户询问哪些偏好以实现长期个性化。 |
| [^45] | [RIFT: KV Cache Compression for Draft Models with Complementary Compressibility Training](https://arxiv.org/abs/2610.31098) | RIFT提出了一类基于选择的KV缓存压缩策略，通过在token生成过程中动态刷新KV缓存选择来应对“解码陈旧性”问题，并借助自适应刷新策略在保持草稿模型质量的同时显著降低刷新开销。 |
| [^46] | [What Pairwise Evaluation Misses: Reward Scores across Prompts](https://arxiv.org/abs/2610.31090) | 论文提出一种以人类评分为参照的审计方法，揭示奖励模型分数存在按提示聚集的跨提示不可比成分——这是成对评估无法检测到的，六个奖励模型的提示间方差份额高达0.34-0.64，而人类评分接近于零。 |
| [^47] | [SCOPE: Compiling Natural-Language Policies into Neurosymbolic Predicate Programs](https://arxiv.org/abs/2610.31062) | SCOPE将自然语言策略编译为由小模型神经谓词、确定性提取器和组合代码构成的可执行神经符号程序，并通过程序与训练数据的联合优化，实现对LLM智能体多轮交互中复合策略的高效且准确的执行。 |
| [^48] | [Harm Is Behavioural, Moderation Is Lexical: How Language and Culture Decide Whom Automated Safety Systems Protect and Whom They Silence](https://arxiv.org/abs/2610.31047) | 本文论证自动化内容审核系统因依据词汇和表面形式而非信息行为来判断危害，导致对非裔美国人英语等少数群体过度标记、对低资源语言保护不足，进而提出有害行为比表达词语更易跨语言传播的假设，并归纳七种失败模式与相应评估协议。 |
| [^49] | [Chunk Recomputation under Real Document Revisions: Measuring Update Cost and Retrieval Quality of Fixed, Recursive, and Content-Defined Chunking](https://arxiv.org/abs/2610.31019) | 该论文首次在真实文档持续修订的场景下量化比较了固定大小、递归和内容定义三种分块策略的更新成本与检索质量，发现递归分块的重计算开销最低（38-56%），而三者在检索召回率上的差异很小。 |
| [^50] | [Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters](https://arxiv.org/abs/2610.31017) | 开卷评分主要衡量的是所有适配器共有的“阅读”行为而非文档记忆——随机适配器即可恢复专用适配器82%的开卷优势，但在无源文本的完形测试中仅恢复约2%（与零无异），因此不宜用开卷评分来评估或选择文档级LoRA适配器。 |
| [^51] | [Padding to k: A Minimum Count in the Schema Makes Open-Weight Models Pad Their Extraction Lists](https://arxiv.org/abs/2610.30997) | 开放权重语言模型在面对JSON模式中设定的列表最小条目数量时会“凑数填充”——用文档中的真实文本补充列表以达到数量要求，这种填充因内容真实而难以被检测，且常见的提示词修复措施只能部分缓解问题。 |
| [^52] | [ColaBench: Evaluating Memory-Guided Knowledge Acquisition in Personal Agents](https://arxiv.org/abs/2610.30989) | 论文提出ColaBench基准，通过证据图谱联合评估个人智能体利用长期记忆识别所需知识并将其应用于用户具体情境的能力，涵盖10个用户画像下的349个任务和13,033篇文档的知识库。 |
| [^53] | [Interpretive Variation and Discourse Structure in Detecting Enthymemes in Controversial Political Discourse](https://arxiv.org/abs/2610.30987) | 该研究发现在争议性政治话语中检测省略三段论时，由于标注者解释存在固有分歧，所有微调模型的F1均停滞在0.67左右，为此提出了基于标注者投票分布的新评估指标和直接预测该分布的模型，并揭示模型依赖显式论证结构进行识别。 |
| [^54] | [A Reality Check of Inference-Time Context Augmentation Methods](https://arxiv.org/abs/2610.30983) | 该论文提出了一个涵盖五种上下文增强方法、七个不同规模模型和三个数据集的统一评估框架，并发现推理时上下文增强带来的准确率提升并非总是真正的改进，而是无方向的扰动——这些方法在纠正部分预测的同时也会破坏其他预测。 |
| [^55] | [Joint or Per-Option? Compact Encoders for Closed-Set Decisions from 17 Million to 1 Billion Parameters](https://arxiv.org/abs/2610.30970) | 在闭集决策任务上，采用重排序器初始化的逐选项评分小型编码器（1700万至10亿参数）在精度上超越了联合评分和大语言模型提示，而延迟至多只增加一倍。 |
| [^56] | [Once a Tree, Always a Tree? Cross-lingual Transfer of Hierarchical Generalization in Language Models](https://arxiv.org/abs/2610.30960) | 本文提出跨语言刺激贫乏（CPOS）实验设置，证明语言模型在第二语言中的层级结构泛化可以由第一语言提供的结构证据驱动，从而支持层级结构作为语言普遍原则在跨语言迁移中被模型利用的观点。 |
| [^57] | [IQuest-Q1: Advancing Foundation Capabilities for Agentic CLI Systems](https://arxiv.org/abs/2610.30949) | IQuest-Q1是一个总参数320B（激活15B）的稀疏MoE智能体基础模型，通过稳定预训练、细粒度中期训练和多脚手架多版本后训练的三阶段流程，显著提升了智能体CLI系统的基础与编码能力。 |
| [^58] | [RetiJev: Fast and Source-Grounded Ophthalmic Decision Models](https://arxiv.org/abs/2610.30947) | 该论文提出RetiJev——一系列无需自回归解码、可直接对候选答案评分的0.8B-9B眼科决策模型，并构建了来源可溯的RetiJev-Data语料库，在指南分级任务上以10.4-23.0个百分点的优势超越27B大语言模型和商业决策模型，同时实现更快、更保护隐私的推理。 |
| [^59] | [Tracing Hallucinations in LLM Reasoning for UML Class Diagram Generation: An Empirical Taxonomy and Mitigation Study](https://arxiv.org/abs/2610.30946) | 该研究针对LLM生成UML类图过程中的推理幻觉，构建了包含提示忠实度、代码忠实度和推理质量三个维度共十一种子类型的实证分类体系，并发现代码忠实度错误是出现频率最高的幻觉类型。 |
| [^60] | [Verbalizing Activation Differences with Natural Language Autoencoders](https://arxiv.org/abs/2610.30934) | 本文系统研究了自然语言自编码器（NLAs）的泛化能力，实验证明其在组合结构、泛化到微调模型、恢复微调目标以及专门化模型差异分析四个方面均表现良好。 |
| [^61] | [ReflectPO: Reflection-Guided Policy Optimization for Long-Horizon LLM Agents](https://arxiv.org/abs/2610.30933) | ReflectPO是一个强化学习框架，通过反思引导采样和树结构信用分配两大核心组件，实现长程任务中大语言模型智能体的自主错误恢复。 |
| [^62] | [Hierarchical Fine-Grained Cross-Modal Alignment via Dynamic Granularity Discovery and Optimal Transport](https://arxiv.org/abs/2610.30912) | 该论文提出层次化细粒度跨模态对齐框架HFCMA，通过动态粒度发现与最优传输，将低层图像块和文本标记逐步转化为自适应的视觉区域与文本短语，并在全局检索评分中实现结构化的区域-短语匹配，从而提升细粒度图文检索性能。 |
| [^63] | [Silicon Samples Match the Answers but Not the Respondents](https://arxiv.org/abs/2610.30900) | 论文提出“受访者覆盖差距”（RCG）这一新评估指标，从完整答案组合的角度衡量合成受访者与真实受访者的差异，发现六种大语言模型虽然能复现答案分布，却无法覆盖真实人类受访者的答案特征组合。 |
| [^64] | [From Perpetrators' Digital Footprints to Behavioral Communities: An Ontology-Driven Framework for Bengali Cyberbullying Analysis--- FULL VERSION](https://arxiv.org/abs/2610.30885) | 本文提出以施暴者为中心的本体驱动框架ODCBNA，通过知识图谱构建、行为建模、社区分析和影响力建模，在孟加拉语网络欺凌分析中实现持续性施暴者识别、行为社区刻画和有影响力用户评估。 |
| [^65] | [Inherited, Not Invented: Tracing the Provenance of Wait in Reasoning Models](https://arxiv.org/abs/2610.30882) | 研究揭示推理模型中的{wait}标记并非由强化学习发明，而是从训练数据（中期训练与推理轨迹微调）中继承而来，RL只能放大模型已有的使用习惯。 |
| [^66] | [One Teacher, Many Views: Reducing Reference Sensitivity in On-Policy Self-Distillation](https://arxiv.org/abs/2610.30880) | 本文发现在线策略自蒸馏中单一完整参考条件会导致教师分布对参考的选择高度敏感（参考诱导的目标不稳定性），并提出蒙特卡洛特权蒸馏方法，通过利用参考的多个视角来降低这种参考敏感性。 |
| [^67] | [Synthetic-to-Real Code-Switched ASR: A Spontaneous Nepali-English Podcast Benchmark](https://arxiv.org/abs/2610.30860) | 该论文构建了一个经过人工审核的13,044条尼泊尔语-英语语码转换自发性播客基准数据集，并提出两阶段合成到真实适配框架——先用TTS合成语音做SFT，再用带转录与文字系统奖励的GRPO——以低成本缓解文字系统混淆和语言切换点错误率高的难题。 |
| [^68] | [TRIAD: Benchmarking Omni-Modal Ambiguity in Multimodal Large Language Models](https://arxiv.org/abs/2610.30841) | 提出了TRIAD——一个全模态歧义消解基准，通过精心构建的文本-图像-音频三元组（只有三者齐全才有唯一答案）测试多模态大语言模型的跨模态歧义消解能力，并揭示现有全模态系统的条目级性能无法等同于真正的歧义消解能力。 |
| [^69] | [TutorCaseBench: A Case-Grounded Benchmark for Evaluating LLM Tutors in Supporting Students with Challenging Behaviors](https://arxiv.org/abs/2610.30828) | 提出了TutorCaseBench基准，基于教育学、教育心理学和特殊教育的真实案例评估LLM导师应对学生挑战性行为的能力，发现即使是最强的模型在开放式案例中也仅满足55.8%的评分标准，且模型识别恰当干预策略的能力优于生成情境化回复的能力。 |
| [^70] | [Invisibilization as a Source of Probability Bias in Language Models](https://arxiv.org/abs/2610.30785) | 本研究通过具有精确贝叶斯答案的双瓮博弈实验发现，领先的大语言模型会因报告者的种族等受保护属性描述而系统性拒绝信任其报告——对被描述为黑人的报告者的拒绝率高达 46%——揭示了模型对齐训练中的“隐形化”现象会导致基于信息来源而非证据质量的概率偏差。 |
| [^71] | [MixJudge: Learning a Language-Queryable Critic for Vocal Mixing](https://arxiv.org/abs/2610.30783) | 提出MixJudge，一个基于11种多严重程度人声混音问题课程训练的语言可查询音频评判器，能够检索和区分混音问题、评估问题严重程度，并可将其判断迁移至处理候选选择任务，支持下游混音决策。 |
| [^72] | [Where meanings converge and diverge: Revisiting linguistic relativity across 81 languages](https://arxiv.org/abs/2610.30778) | 本研究通过比较81种语言中196个基本概念的词嵌入发现，社会文化意义跨语言高度趋同而自然与运动意义分歧更大，揭示了概念结构中普遍性与语言相对性的交织。 |
| [^73] | [Abstract Priorities, Concrete Rewards: How Stated Preferences Relate to What People Reward in AI Responses](https://arxiv.org/abs/2610.30763) | 该研究基于PRISM数据集发现，人们对AI回复的评分受回复长度、格式、拒绝语言等未被直接询问的具体特征的影响不亚于被调查的属性，且陈述的偏好只有在模型整体感知层面（而非文本指标层面）才与实际奖励行为一致，表明陈述性偏好未必反映人们真正奖励的内容。 |
| [^74] | [How Firm Are AI Models' Ethical Boundaries? A Pay-Stub Editing Test of 15 Models](https://arxiv.org/abs/2610.30751) | 该研究通过让15个AI模型执行相同的工资单收入修改请求发现，措辞的微小变化（如省略贷款用途或称修正打印错误）会使模型执行编辑的比例从14.1%升至66.2%，表明AI模型的伦理拒绝边界对措辞高度敏感。 |
| [^75] | [UniGuard: Multi-View Guardrail Learning for Cross-Interface Safety Generalization](https://arxiv.org/abs/2610.30742) | UniGuard提出了一种多视图护栏训练框架，通过将同一安全案例转换为提示、响应、多轮对话、智能体轨迹等多种对齐视图来学习接口不变的安全表示，使0.6B规模的护栏模型在未见过的接口上平均提升45.12个F1点并显著改善置信度校准。 |
| [^76] | [Meaning Contracts: Local Semantic Geometry in Large Language Models](https://arxiv.org/abs/2610.30715) | 本文提出“意义契约”的概念，通过跨语言扰动探针和语义细化探针，在四个架构各异的大语言模型上首次系统刻画了概念内部的局部语义几何——即单一语义因子的哪些信息在不同实现中保持不变、哪些会变化，以及意义细化时如何变化。 |
| [^77] | [ULME: Predictive Coupling for Mutually Reinforcing Embedding and Language Modeling](https://arxiv.org/abs/2610.30705) | ULME提出了一种后训练框架，通过预测性耦合机制让文本嵌入与语言建模两种能力在同一因果解码器大语言模型上真正统一并相互增强，突破了以往统一模型中两种目标互不监督的局限。 |
| [^78] | [Disentangling Robustness and Sycophancy in Multilingual LLMs](https://arxiv.org/abs/2610.30696) | 该论文提出SC矩阵将多语言大语言模型的能力不足与谄媚行为解耦区分，发现多语言谄媚并非单一现象而是由不同失败模式构成的复合体，且面向推理的缓解策略优于指令式策略。 |
| [^79] | [From Clinical Judgment to Checkable Rules: Clinician-Led Guardrails for Mental-Health Chatbots](https://arxiv.org/abs/2610.30690) | 该论文提出由临床医生主导定义并经形式化验证的安全规则库，将专家临床判断转化为可解释、可复查、可重算的心理健康聊天机器人安全护栏。 |
| [^80] | [Direct cueing of word identity through word duration: evidence from Mandarin homophones](https://arxiv.org/abs/2610.30682) | 本研究通过汉语同音词排序任务发现，听者能够利用词长（高频词更短）这一语音线索直接识别词的身份，证明词长在口语词汇识别中具有功能性作用。 |
| [^81] | [Scaling Laws for Safety Probes](https://arxiv.org/abs/2610.30678) | 本文通过构建六个新安全数据集、在20个基础模型上以不同标签规模训练探针，首次提出了可依据模型规模和标签数量提前预测安全探针性能的缩放定律。 |
| [^82] | [WikiOriginQA: Automated Construction of a QA Dataset with Knowledge Origins for Analyzing Cultural Bias](https://arxiv.org/abs/2610.30677) | 提出WikiOriginQA，利用维基百科不同语言版本的创建日期客观表征知识来源并自动构建问答数据集，实现无需人工标注的可扩展文化偏见评估，实验发现大语言模型存在偏向英语文化的偏见，且知识向英语圈的传播至关重要。 |
| [^83] | [Spectral Smoothing and Spiking Perturbations Reconfigure Capability Profiles and Selective Task Performance in Large Language Models](https://arxiv.org/abs/2610.30676) | 该论文通过对Transformer前馈投影矩阵的奇异值谱进行平滑与尖峰化干预（保持奇异向量和几何均值不变），以三维能力基（事实知识、语言理解、演绎推理）为解释框架，系统揭示并重构了大语言模型的能力画像及其在各类任务上的选择性表现变化。 |
| [^84] | [Compiling Autoregressive LLM Control into Policy-Exact Parallel Inference for Hierarchical RAG](https://arxiv.org/abs/2610.30664) | 该论文提出将分层RAG中自回归LLM的控制决策编译为策略精确的并行推理——只需一次根预填充加一次树因果探测，即可在两次前向计算的关键路径、零输出token的条件下，精确复现串行语法约束解码的动作分布。 |
| [^85] | [Group Relative Policy Optimisation Improves Multilingual Speech Recognition in Low-Resource Languages](https://arxiv.org/abs/2610.30663) | 将组相对策略优化（GRPO）强化学习应用于非LLM的Whisper编码器-解码器模型，训练出的SunflowerASR在51种非洲低资源语言上取得已发表的最佳词错误率，尤其在训练数据稀缺或音频嘈杂的场景下优势最为明显。 |
| [^86] | [What the Specification Did Not Say: Specification-Oracle Gaps in Test-Graded Coding-Agent Benchmarks](https://arxiv.org/abs/2610.30635) | 该论文发现编程智能体基准测试中存在“规范-预言机差距”——评分测试依赖任务描述中未说明的实现细节（如常量名、模块路径），导致这类任务的智能体失败率高达80.6%（无此差距时仅48.4%），并提出了覆盖五种语言的形式化静态审计方法来检测此类差距。 |
| [^87] | [The Fusion Tax: A Diagnostic Analysis of When Hybrid Retrieval Hurts](https://arxiv.org/abs/2610.30612) | 该研究通过诊断分析发现，混合检索（如RRF融合）虽然在总体指标上优于纯稠密检索，但在8-48%的个体查询上反而损害检索效果，因此“不融合”应被视为值得认真对待的一等选项。 |
| [^88] | [TT-SAX: A Two-Track Time Series Symbolization for Natural Language Shape Querying](https://arxiv.org/abs/2610.30593) | 该论文通过分析六种经典符号化方法在自然语言时间序列查询中的三大局限并提出三条设计原则，据此对SAX进行最小化改进，提出双轨符号化方法TT-SAX，使自然语言形状查询能更好地捕捉时间序列的方向与变化趋势。 |
| [^89] | [Governed Self-Evolution for Autonomous Investing: A Multi-Agent LLM System with an Invariant Core](https://arxiv.org/abs/2610.30557) | 该论文提出了Steward——一个面向自主投资的多智能体LLM系统，通过九个角色专业化智能体协同决策、七层共享记忆，以及以类型系统划定自我进化边界的“不变核心”，在行动不可逆且数据不可信的投资场景中实现受治理的安全自我进化。 |
| [^90] | [IFC-MCQ: Automatically Generating Multiple-Choice Benchmarks for Evaluating LLM Agents on IFC Data Retrieval](https://arxiv.org/abs/2610.30533) | 提出IFC-MCQ方法，从IFC文件自动生成多项选择题基准并实现精确自动评分，从而以低成本大规模评估LLM智能体在IFC数据检索上的能力。 |
| [^91] | [Protegrity-PII Technical Report](https://arxiv.org/abs/2610.30518) | 提出了新型 PII 决策模型 Protegrity-PII，在自建基准 PII-SCOPE 上实现 0.915 召回率和 0.907 精确率，兼具高速低成本推理与卓越的分布外泛化能力。 |
| [^92] | [Beyond End-to-End Recovery: Towards a Better Understanding of LLM Steganography via Rolewise Decomposition](https://arxiv.org/abs/2610.30516) | 该论文提出了一个按角色分解的评估框架，将大语言模型隐写术的评估从单一的端到端恢复拆解为编码表示、解码恢复和确定性验证三个独立环节，从而能够区分编码失败与解码失败，并揭示私密信息在传输文本中的隐藏表示方式。 |
| [^93] | [Behind Aggregate Scores: What Systematically Varying Arena-Hard's LLM Judge Reveals](https://arxiv.org/abs/2610.30512) | 该研究通过系统性地改变Arena-Hard评测中LLM评判者的标准、输出格式与任务指令，发现相似的总分掩盖了大量个别评判结果的差异——即使在温度为零且固定模型版本的条件下重复运行，仍有21.2%的提示词的胜/平/负结果会改变，表明LLM评测的可靠性存在被忽视的隐患。 |
| [^94] | [Rethinking the Evaluation of LLM Fact-Checkers](https://arxiv.org/abs/2610.30504) | 提出了首个同时明确声明-问题关联、判定条件和证据来源三项设计决策的日语事实核查基准JSFactCheckBench，揭示了总体分数掩盖了模型核查直接回答问题的声明时准确率显著更低的现象。 |
| [^95] | [Why Bother Me Again? Benchmarking Personal Agent Memory for Interaction Preferences](https://arxiv.org/abs/2610.30485) | 提出了MemPABench——一个评估个人智能体记忆与落实用户交互偏好能力的在线交互式基准，通过四个月的模拟用户会话和动态偏好变化，测试智能体何时澄清、确认或自主行动。 |
| [^96] | [Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy](https://arxiv.org/abs/2610.30474) | 本文提出了一种基于原子可解释单元的SAE评估新框架，将超越单义性的评估目标形式化，并借此揭示了现有SAE特征中隐藏的多义性与冗余性。 |
| [^97] | [Self-Assessment Reinforcement Learning for Search-State Awareness in Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2610.30435) | 提出SARL框架，通过兼顾评估纠正效果与惩罚过度依赖的双重奖励设计，让模型在训练中学会感知自身搜索状态，无需测试时额外评估即可实现更有效且高效的搜索，相比基线获得24.3%的相对提升。 |
| [^98] | [Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems](https://arxiv.org/abs/2610.30418) | 该论文首次量化了LLM智能体中的“马赛克效应”风险——智能体即使无法访问任何单独的机密文档，仍会将多份可访问文档的信息组合成敏感结论并造成机密泄露，且实验表明现有的密级标注和聚合规则等防护措施仍无法完全阻止此类不安全泄露。 |
| [^99] | [X-pressing Intimacy: Mental Health Disclosure After Twitter Acquisition](https://arxiv.org/abs/2610.30413) | 本文利用LIWC与基于大语言模型的亲密度分类器分析约82.4万条推文，揭示了马斯克收购Twitter后用户心理健康自我披露的变化：情感与社会性语言减少、第一人称单数及未来导向表达增加，且披露亲密度的变化模式因病症而异。 |
| [^100] | [SIM-UT: A User Simulation based Evaluation Framework for Medical Triage Chatbots](https://arxiv.org/abs/2610.30412) | 提出SIM-UT，一种多轮用户模拟评估方法，能发现基于LLM的医疗分诊聊天机器人在真实多轮、多语言用户交互情境下的失效模式，并在印度、南非和肯尼亚的三个真实WhatsApp健康聊天机器人上进行了验证。 |
| [^101] | [Text-Only LLM Correctors Break Correct Transcripts: A Repair-Damage Analysis of Audio Grounding in ASR Error Correction](https://arxiv.org/abs/2610.30385) | 该论文提出将WER变化精确分解为修复、损伤与插入的修复-损伤分析框架，首次系统量化揭示了纯文本LLM纠错会破坏13-32%本已正确的转录文本，而引入音频信息后破坏率降至1-3%，证明音频锚定对ASR纠错具有关键的保护作用。 |
| [^102] | [MemPlan: Memory-Conditioned PDDL Planning for Partially Observable Text Environments](https://arxiv.org/abs/2610.30379) | MemPlan提出了一种记忆条件化的PDDL规划框架，利用正向图记忆为候选假设分配基于相关性的动作成本、利用负向记忆将已被反驳的假设编译为排除约束，使LLM作为语言与符号规划之间的受约束桥梁，在部分可观测文本环境中有效减少幻觉和无效动作。 |
| [^103] | [Climbing the Ladder: Automating Data Pipelines and Reproducing Baselines in Low-Resource Settings](https://arxiv.org/abs/2610.30374) | 本文在可维护软件上自动化重建了从双语期刊提取西西里语平行语料的数据流水线，并在冻结测试集上系统比较从头训练的小型Transformer与低秩微调的NLLB模型，从而复现并改进了低资源西西里语机器翻译基线。 |
| [^104] | [Probing and Steering for Safety-Relevant Behaviours in Large Language Models](https://arxiv.org/abs/2610.30363) | 该论文在统一流水线中系统探究了激活探针设计选择（估计器类型、token 池化方案、层选择）在检测与引导欺骗、高风险意识和谄媚三种安全相关行为时的相互作用，并验证了“逻辑回归最擅长检测、均值差方向最擅长引导”的读/写分离现象。 |
| [^105] | [CiteAnchor: Source-Aware Grounding for Hallucinated Citation Detection and Resolution](https://arxiv.org/abs/2610.30355) | 本文提出了包含423条真实幻觉引用的CiteResolve基准和来源感知定位框架CiteAnchor，通过针对不同来源类型自适应的检索与字段级验证，分别判定引用有效性并恢复唯一来源，在检测与端到端溯源任务上均显著超越基线。 |
| [^106] | [Consistency Is Not Validity: Self-Rewarded Moral Argumentation Makes Language Models Decisive but Pliable](https://arxiv.org/abs/2610.30349) | 该论文提出一种自我奖励的强化学习方法，让模型通过自我评判道德论证的质量来提升道德判断能力，使模型变得更加果断且不受选项呈现顺序的影响，但谄媚性和重新框架问题仍然存在。 |
| [^107] | [AQL-Lite: Enabling Efficient Knowledge Acquisition for Arabic LLMs via Data-Centric Continual Pretraining](https://arxiv.org/abs/2610.30344) | AQL-Lite 通过质量优先的数据筛选流水线与部分参数持续预训练方案（RS-LoRA 适配器结合最后八层 Transformer 块与嵌入矩阵），以约一半的训练时间恢复了全参数 CPT 94% 的性能提升，高效地将多语言大模型适配至现代标准阿拉伯语。 |
| [^108] | [Measuring Research Progress and Linguistic Inclusion in Low-Resource Languages](https://arxiv.org/abs/2610.30296) | 该研究提出语言包容性框架（LIF），通过分析2020-2025年间1,158篇论文发现，尽管低资源语言NLP研究论文数量增长了5.4倍，但增长主要集中于已被代表的主流语言，而非真正扩展对代表性不足语言的覆盖。 |
| [^109] | [Protection Is Not One Event: A Survey of RAG Protection Research by Measured Endpoint](https://arxiv.org/abs/2610.30193) | 该论文通过对118篇RAG防御论文按“实际度量终点”而非声称目标进行系统性分类综述，构建了七类分类体系，并揭示了防御机制与保护结果不对应、评估标准割裂等三大研究空白。 |
| [^110] | [Finitely Verifiable Segmentation: Decidability for Every Tokenizer with a Finite-State Verifier](https://arxiv.org/abs/2610.30168) | 本文提出“有限可验证分段”这一概念——即分段策略的结果恰好可被一个有限自动机验证——并证明在此假设下四个实际分词问题（边界窗口证明、边界间隙有界性、掩码生成可补全性以及策略等价性）均变为可判定问题。 |
| [^111] | [FusionEval: Benchmarking Model Fusion for Large Language Models](https://arxiv.org/abs/2610.30162) | 提出了FusionEval基准，通过受控专家池和统一评估套件系统评估大语言模型的模型融合方法，揭示无需训练的融合方法虽能保留推理性能，但会显著损害工具使用能力。 |
| [^112] | [TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents](https://arxiv.org/abs/2610.30161) | 该论文提出了 TauIndianBankBench，一个包含 1,000 个印度零售银行任务的面向动作的基准测试，它不仅检查对话的最终记录，还检查助手是否完整执行了所有必需的流程步骤且未做出任何未经请求的更改，从而更准确地评估工具使用型对话智能体对银行流程的遵循能力。 |
| [^113] | [Lighthouse: On Building a Trustworthy Digital Assistant for Exploring History with Civil Rights Archives](https://arxiv.org/abs/2610.30158) | 该论文提出了首个用于探索美国民权时期历史的可信赖数字助手Lighthouse，其创新在于采用神经符号方法（结合大语言模型与基于规则的符号求解），以内容可信度和减少生成答案的潜在伤害为核心设计原则。 |
| [^114] | [baseVultra: Compact Nonlinear Recurrence with Reallocated Capacity](https://arxiv.org/abs/2610.30154) | 提出baseVultra紧凑递归单元，将递归计算从门控变换重新分配到候选生成，在参数量匹配条件下于TinyShakespeare字符级语言建模上超越LSTM和GRU。 |
| [^115] | [GoiBERT: Compact Vietnamese Language Models via Two-Stage Knowledge Distillation](https://arxiv.org/abs/2610.30067) | GoiBERT通过两阶段知识蒸馏（先从同分词器越南语编码器学习通用表示，再从大型教师模型获取软标签任务指导），构建了仅2600万和4900万参数、可直接处理原始文本的紧凑越南语编码器，在八项基准测试中以不到一半的参数量和两倍吞吐量取得非教师模型中的最高平均分数。 |
| [^116] | [Investigating the Impact of Learner Errors on Social Bias in Large Language Models](https://arxiv.org/abs/2610.29896) | 该研究首次系统考察了非母语写作者的语法错误对大型语言模型社会偏见的影响，发现错误率越高偏见越强（其中介词错误影响最大），并通过在错误文本上微调有效缓解了这一问题。 |
| [^117] | [Evolving with Open Worlds: Towards Recursive Self-Improvement for Future Forecasting](https://arxiv.org/abs/2610.29784) | 提出DeepEvolve智能体系统，以未来事件预测作为持续变化且可验证的开放环境，通过递归演化引擎将预测背后的推理轨迹转化为能力，从而实现智能体的递归自我改进。 |
| [^118] | [Overview of CHIP 2026 Shared Task 1: Patient-Specific Phenotype Concept Recognition in Full-Text Biomedical Literature](https://arxiv.org/abs/2610.29606) | 本文概述了CHIP 2026共享任务1，该任务基于PatientPheX数据集的200篇标注全文文献，设置全文表型概念识别与患者特异性表型关联两个子任务，以推动从全文生物医学文献中自动识别表型概念及其患者关联方法的发展。 |
| [^119] | [Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning](https://arxiv.org/abs/2610.29351) | 提出闭环双参考引导方法，通过将原始参考与语音转换构建的母语锚点配对并动态更新语言模型状态，有效消除跨语言语音克隆中的源语言口音，在仅损失极小说话人相似度的前提下显著提升目标语言的发音质量。 |
| [^120] | [Uncensored Is Not Unbiased: Abliteration Removes Refusals but Not Script-Dependent Bias About Taiwan](https://arxiv.org/abs/2610.29342) | 去审查化（abliteration）模型只是移除了拒绝行为，其对台湾问题的政治立场主要取决于所用书写文字（繁体、简体还是英文），而非是否经过消融审查，因此“无审查”并不等于“无偏见”。 |
| [^121] | [OACap: Caption-Guided Active Observation for Detailed Video Captioning](https://arxiv.org/abs/2610.29327) | 提出了以描述为中心的智能体框架 OACap，它能将视频描述中缺失或缺乏支持的信息转化为主动观察请求，借助时空工具获取证据并迭代验证修订描述，从而显著提升详细视频描述质量，同时引入 FSCap-Bench 基准及 CSFS 等评分指标来系统评估描述的事实性与语义保真度。 |
| [^122] | [When Multiple Choices Are Correct: Asymmetric Probability Allocation in Jev and Other Decision Models](https://arxiv.org/abs/2610.29217) | 该论文通过在三个数据集上评估三个决策模型，首次系统揭示了多正确答案场景下概率分配的不对称性问题——即使模型排序完全正确，仍有28%至50%的案例中至少一个正确选择的概率低于均匀概率1/n，且给相同选项分配的概率差距高达17至79个百分点。 |
| [^123] | [How Far Can the Gemma 4 Speech Encoder Go Without Changing Its Weights? Saudi and Gulf Arabic Speech Recognition with a Frozen and an Adapter-Augmented Encoder](https://arxiv.org/abs/2610.28673) | 在完全不改变 Gemma 4 语音编码器权重的前提下，仅通过轻量级 CTC 读取器或少量低秩适配器参数，就能让沙特和海湾阿拉伯语语音识别达到与参数量大得多的商业系统统计上相当的性能，同时在 CPU 上的推理速度快 3.4 倍。 |
| [^124] | [Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs](https://arxiv.org/abs/2610.28498) | 论文通过仅有一个数字不同的最小提示对证明，从输入复制的信息确实以跨问题共享的编码存在于模型状态中，但常规单样本探针难以读取它，因此探查失败不能证明信息缺失，并提出了一套区分“计算出的值”与“输入副本/混合”的探查协议。 |
| [^125] | [Disclosure Begets Disclosure: A Self-Reinforcing Privacy Loop in LLM Agents with Persistent Memory](https://arxiv.org/abs/2610.28315) | 该研究揭示了持久记忆型LLM智能体中存在自我强化的隐私泄露循环：基于记忆的响应会将早期属性值作为预设延续，用户确认或不加纠正，导致属性泄露随会话次数持续扩大。 |
| [^126] | [Poisoning-Based Backdoors Are Not Cryptographic](https://arxiv.org/abs/2610.28045) | 本文通过归约证明，基于投毒安装的神经网络后门不可能获得密码学认证那样的不对称性，因为任何从被投毒数据中安装触发器验证器的训练过程本身就是一个学习过程，因此基于学习的防御在原则上始终是可行的。 |
| [^127] | [Sharp Degree Bounds for Putinar Certificates in Terms of the Positivity Margin via Analytic Sums of Squares](https://arxiv.org/abs/2610.26976) | 本文建立了关于正性裕度 δ 的 Putinar 平方和证书的精确次数上界 O(δ^(-1/2)log²(2/δ))，并通过构造三变量匹配下界例子证明了对 δ 的依赖是最优的，同时给出了 Lasserre 层级的精确最坏情形收敛速率。 |
| [^128] | [SpatialScout: Can VLM Agents Actively Explore and Map 3D Scenes?](https://arxiv.org/abs/2610.26659) | 提出了SpatialScout基准，用于评估视觉语言模型智能体在有限交互预算下，通过主动探索室内场景来构建物体级3D度量地图的能力。 |
| [^129] | [Beyond Attack Success Rate: Cost-Aware Metric for Evaluating Iterative LLM Jailbreaks](https://arxiv.org/abs/2610.26589) | 提出成本感知评估指标 ASR_STEP，通过对重复调用受害模型的次数进行惩罚，弥补了传统攻击成功率指标在评估迭代式越狱攻击时忽视调用成本的缺陷。 |
| [^130] | [VEILED-SH: Vulnerability Expression through Indirect Language Evaluation Dataset (Self-Harm)](https://arxiv.org/abs/2610.26190) | 该论文提出 VEILED-SH 基准数据集（含 10,003 个提示词），首次将文化背景、应用场景和从业者分歧三者结合，用于评估 AI 对自杀与自伤间接表达的单轮回复，并通过专家校准的自动化画像实现代理对齐评分。 |
| [^131] | [Memories That Outlive Their Reasons: Dependency-Tethered Memory for Tool-Using LLM Agents](https://arxiv.org/abs/2610.25123) | 该论文提出依赖锚定记忆方法，通过在写入时记录每条结论所依赖的确切版本、在读取时自动撤回存储中任何支撑已失效的结论，解决了工具使用型LLM智能体因上游依赖失效而依赖过时记忆做出错误发布决策的问题。 |
| [^132] | [Beyond Recall Saturation: Scaling Hard-Negative Reranker Training for Industrial Legal Retrieval](https://arxiv.org/abs/2610.24478) | 在中文法律检索中，当嵌入召回趋于饱和后，采用“嵌入挖掘到自挖掘”的难负样本分阶段训练流水线是重排序器训练的最佳权衡方案，且可将模型从0.6B扩展至4B。 |
| [^133] | [Pluralistic Alignment in Large Language Models: Objectives, Human References, and Preference Representations](https://arxiv.org/abs/2610.24471) | 该论文提出了一个双层分类框架来系统梳理大语言模型多元对齐研究，第一层区分三种对齐目标（表征多元观点的Overton多元性、适应特定视角的可引导多元性、匹配人群分布的分布多元性），第二层区分三类人类参照（聚合人群、国家和地区、子群体），并将偏好表征、实现机制等视为贯穿性设计选择。 |
| [^134] | [GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis](https://arxiv.org/abs/2609.38923) | GraphForge 提出了一种基于证据图的框架，将训练工作智能体所需的任务与验证标准都锚定在真实文件构建的工作区上，从而合成兼具真实性、多样性和可验证性的高质量任务数据。 |
| [^135] | [On the Behavioral Traits of LLM Agents](https://arxiv.org/abs/2609.32776) | 提出A-B-D方法，从345,667条真实世界轨迹中自下而上提取行为特征，基于智能体的实际行为（而非自我报告或LLM评分）来推断其稳定的人格特质。 |
| [^136] | [Is the Judge Its Twin? Post-training makes a model's writing more predictable but barely moves its taste, as a judge, toward predictable writing](https://arxiv.org/abs/2609.32196) | 该研究发现后训练（SFT和DPO）确实使语言模型的写作变得更加可预测，但这种可预测性偏好并不会迁移到模型作为评判者的角色——训练后的模型在评判故事时几乎不偏向可预测的写作。 |
| [^137] | [Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search](https://arxiv.org/abs/2609.29636) | 该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。 |
| [^138] | [Initial Evaluation of Potential Bias in Coverage of Humans in Wikidata](https://arxiv.org/abs/2609.22375) | 本文开发了一个开源审计平台，对Wikidata中超过600万个人类条目在性别、地理、族裔、职业等多维度上的人口代表性偏差进行了系统性评估，发现声明性别的条目中女性仅占28.71%。 |
| [^139] | [Moving the Mean Toward the Known Good, Not Beyond It: What Inference-Time Interventions and Weight Consolidation Buy in Open-Ended Generation](https://arxiv.org/abs/2608.28886) | 该研究发现在开放式生成任务中，通过“生成-验证-选择-LoRA合并”循环在自身验证成功的样本上训练，能显著提升模型输出的平均质量并经三重重复实验验证其稳健性，但最佳候选仅能收敛到经典启发式算法的水平而无法超越。 |
| [^140] | [Interrupting the Loop: Periodic Subject Changes Raise Judged Surprise and Connection in Base Language Models](https://arxiv.org/abs/2608.19893) | 该论文发现，在基础语言模型的生成循环中，周期性注入新主题（打断）是提升文本评判惊喜度和连贯性的核心机制，且效果显著优于仅靠习惯化或其他操作。 |
| [^141] | [DFM Mimir v1: An Open HRM Delivering Frontier Performance at 1B Parameters Using Only Permissible Post-Training Data](https://arxiv.org/abs/2608.13517) | Mimir v1是一个10亿参数的HRM架构语言模型，仅使用许可数据训练，在英语和丹麦语上实现了前沿性能，并超越了同尺寸及更大尺寸的模型。 |
| [^142] | [Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL](https://arxiv.org/abs/2608.11669) | 本文提出一种简单方法（Rubric Dropout）来缓解基于评分标准的强化学习中的奖励黑客问题，并通过实验证实训练评判员与黄金评判员分数在训练中分化的现象，为提升模型鲁棒性提供新思路。 |
| [^143] | [When Self-Consistency Backfires: Majority Vote Hurts the Majority of Hard Science Problems for Small LLMs](https://arxiv.org/abs/2608.11403) | 本研究发现，在GPQA Diamond基准上，多数投票的自我一致性方法反而降低了小型语言模型在大多数硬科学问题上的准确率，并预注册验证了这一反直觉现象。 |
| [^144] | [Evaluating Rational Contracting in Natural Language](https://arxiv.org/abs/2608.10475) | 该论文提出了一个理性缔约框架和评估套件ContractSim，用于量化评估AI智能体在不确定多步骤环境中协商与履行自然语言契约时的理性与合作能力。 |
| [^145] | [Beyond Naturalness: Probing Automated Text-To-Speech Evaluators on Linguistically Grounded Dimensions](https://arxiv.org/abs/2608.09930) | 该论文将TTS评估中的“自然度”解构为10个语言学感知维度，构建了首个维度级元评估基准（860条由语言学家标注的语音），并揭示MOS预测器和Audio-LLM评判者各自仅能捕捉人类语音感知的部分维度，无法全面覆盖。 |
| [^146] | [TORUS: A Unified Closed-Loop Benchmark for Unified Audio Models](https://arxiv.org/abs/2607.28896) | 提出TORUS——首个音频原生的统一闭环基准，用于评估统一音频模型能否理解自己生成或编辑的音频，发现最佳模型的自一致性仅约52%-54%，揭示了当前统一音频模型在理解-生成自一致性上的重大缺陷。 |
| [^147] | [Context-Aware Subtitle Translation on CPU: Deployment-Driven Training, Evaluation, and Latency Design](https://arxiv.org/abs/2607.09957) | 该论文构建了可在观众CPU上以每条约500毫秒完成翻译的英译繁体中文字幕模型，通过部署驱动的训练流程、基于LLM评审的成对比较评估方法以及空闲时间预填充等延迟优化设计，实现了高质量翻译与严格延迟约束的兼顾。 |
| [^148] | [Pathways of Visual Information Flow in Vision-Language Models](https://arxiv.org/abs/2607.03358) | 该研究发现视觉-语言模型通过“直接通路”和“文本中介通路”两条不同路径处理视觉信息，通路选择取决于任务类型、数据分布和提示设计，且在常用通路被消融时模型能灵活启用文本中介通路作为后备。 |
| [^149] | [Latent Bridges for Multi-Table Question Answering](https://arxiv.org/abs/2606.28916) | 该论文提出GRAB框架，将关系数据编码为异构图，并通过查询条件化的潜在标记桥接到完全冻结的LLM（仅训练91M参数的轻量模块），显著提升了多表格问答的性能。 |
| [^150] | [Beyond Prompt and Parser Engineering: Soft-Prompt Tuning for Automated LLM Evaluation](https://arxiv.org/abs/2606.12117) | 该论文提出基于软提示微调的自动化LLM评估方法，仅需优化约0.0006%的参数即可让模型适应基准测试格式，从而更准确地反映模型的真实知识水平。 |
| [^151] | [Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs New Metrics](https://arxiv.org/abs/2606.08417) | 该论文证明生成困惑度作为无条件文本生成质量指标并不可靠——通过构造无需训练却能“作弊”该指标（低困惑度但文本不连贯）的采样器来揭示其缺陷，并主张采用分布差异和多样性诊断等新指标来评估扩散语言模型。 |
| [^152] | [MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery](https://arxiv.org/abs/2606.06473) | MLEvolve是一个基于大语言模型的自进化多智能体框架，通过渐进式MCGS树搜索实现跨分支信息流与探索-利用的渐进转变，并借助结合领域知识库与动态全局记忆的回顾性记忆机制，实现端到端的机器学习算法自动发现。 |
| [^153] | [WaxalNet: Benchmarking and Adapting Speech Recognition for Spontaneous Speech in 19 African Languages](https://arxiv.org/abs/2606.02375) | 该研究构建了覆盖19种非洲语言自发语音的基准测试，发现微调可将最佳零样本模型的词错误率从52.3大幅降至31.5，且在共享编码器条件下LLM解码器在18/19种语言上优于CTC解码器，表明不同模型家族的性能差异更可能源自预训练而非解码器架构。 |
| [^154] | [CARTE: A Benchmark for Mapping Language Model Knowledge Across France](https://arxiv.org/abs/2606.01995) | 该论文提出了CARTE基准，通过覆盖法国13个大区、12个主题领域的3,132道多选题，评估大语言模型辨别法国国内细粒度区域差异知识的能力，并引入聚焦各区域语言变异的CARTE-LV子集。 |
| [^155] | [Measuring, Localizing, and Ablating Alignment Signatures in LLMs](https://arxiv.org/abs/2605.30526) | 该论文发现后训练会使大语言模型的生成文本偏离人类语料风格并呈现出可被AI检测器识别的类AI特征，并提出了无需训练的PASTA方法，通过从残差对比中估计对齐签名方向并在解码时消融该方向，从而定位并去除模型内部的这一AI风格签名。 |
| [^156] | [Confidence and Calibration of Activation Oracles for Reliable Interpretation of Language Model Internals](https://arxiv.org/abs/2605.26045) | 本研究比较了五种估计激活预言机置信度的方法，发现当审计者能提前列出候选答案时，让预言机直接对每个候选答案评分可获得最佳校准效果。 |
| [^157] | [How Far Will They Go? Red-Teaming Online Influence with Large Language Models](https://arxiv.org/abs/2605.22880) | 该论文提出了一个实证红队测试框架，通过测量LLM的Overton窗口来量化本地部署的开源大语言模型表达政治观点的保真度以及简单越狱攻击对这些分数的影响，并发现模型在政治表达上存在系统性不对称（左倾内容的表达保真度通常更高）。 |
| [^158] | [When Should LLMs Reason? Adaptive Inference Routing via Entropy Dynamics](https://arxiv.org/abs/2605.22873) | 提出无需训练的轻量级路由框架EDRM，通过解读解码早期的熵轨迹变化趋势，自适应地在直接、标准和思维链三种解码模式间选择，在15个基准和7个大语言模型上有效平衡准确率与token消耗。 |
| [^159] | [FormalASR: End-to-End Spoken Chinese to Formal Text](https://arxiv.org/abs/2605.19266) | FormalASR提出两个紧凑的端到端模型（0.6B与1.7B参数），可直接将中文口语转录为正式书面文本，省去ASR+LLM两阶段流水线，相比逐字转录基线实现高达37.4%的相对字错误率降低。 |
| [^160] | [IfcLLM: Natural Language Querying of IFC Models through Complementary Relational and Graph Representations](https://arxiv.org/abs/2605.13236) | IfcLLM提出将关系型与图两种互补数据表示相结合的框架，通过LLM智能体的迭代重试-精炼推理自动将自然语言查询路由到最合适的后端，在IFC模型查询任务中达到96%的准确率，且无需用户干预即可从失败中自动恢复。 |
| [^161] | [Graph Memory Transformer (GMT)](https://arxiv.org/abs/2604.23862) | 本文提出图记忆Transformer（GMT），用可学习的显式记忆图（128个质心加有向转移矩阵）替代仅解码器Transformer中的FFN子层，在因果自注意力不变的情况下，以更少的参数量（82.2M对103.0M）实现稳定训练。 |
| [^162] | [AtManRL: Towards Faithful Reasoning via Differentiable Attention Saliency](https://arxiv.org/abs/2604.16158) | 提出AtManRL方法，通过优化可微注意力掩码推导显著性奖励并与结果奖励结合于GRPO强化学习框架中，使思维链推理真正因果性地促成最终答案，从而提升推理的忠实性。 |
| [^163] | [Efficient Best-Of-N with Radial Consensus Score](https://arxiv.org/abs/2604.12196) | 提出径向共识分数（RCS），通过计算答案嵌入的加权Fréchet均值作为语义中心，并依据候选答案到该中心的径向距离进行排序，实现了一种简单、高效且无需训练的best-of-N答案选择方法。 |
| [^164] | [A Sociolinguistic Analysis of Automatic Speech Recognition Bias in Newcastle English](https://arxiv.org/abs/2603.24549) | 本研究基于泰恩赛德英语历时电子语料库（DECTE），对纽卡斯尔英语中3000多个商业ASR转录错误进行了细粒度的社会语言学分析，揭示了识别错误与说话人性别、年龄及社会经济地位等社会变量之间的关联，并通过元音声学案例研究展示了社会语言学变异如何引发语音识别偏见。 |
| [^165] | [Language Models as Higher-Order Planning Formalizers](https://arxiv.org/abs/2603.23844) | 该论文指出LLM作为形式化器的成功在标准基准上被高估，并提出“展开式问题”这一新基准来揭示LLM形式化器在需要从简洁描述生成大规模PDDL时无法扩展，进而提出LLM作为高阶形式化器的新范式以应对这一挑战。 |
| [^166] | [Markovian Generation Chains in Large Language Models](https://arxiv.org/abs/2603.11228) | 本文将大语言模型反复处理文本的迭代推理过程定义为马尔可夫生成链，发现输出要么收敛到小的循环集合要么持续产生新句子，且多样性的增减取决于温度参数和初始输入句子等因素。 |
| [^167] | [TabAgent: Distilling LLM Agent Components into Cheap Classifiers](https://arxiv.org/abs/2602.16429) | TabAgent框架通过将智能体系统中反复调用的LLM生成式决策组件替换为基于执行轨迹训练的轻量级文本-表格分类器，在保持任务成功率的同时显著降低部署延迟和成本。 |
| [^168] | [Artificial intelligence is creating a new global linguistic hierarchy](https://arxiv.org/abs/2602.12018) | 该研究通过对6003种语言的社会、经济和基础设施条件进行全球纵向分析，揭示了语言AI资源分布的系统性不平等，表明人工智能正在催生一种使大多数语言社区数字边缘化的新全球语言等级体系。 |
| [^169] | [GFlowPO: Generative Flow Network as a Language Model Prompt Optimizer](https://arxiv.org/abs/2602.03358) | GFlowPO提出了一种概率化提示词优化框架，利用离线策略的生成流网络目标微调轻量级提示词语言模型，并通过无需训练的动态记忆更新机制更新元提示词，从而实现样本高效的语言模型提示词优化。 |
| [^170] | [The Imperfective Paradox in Large Language Models](https://arxiv.org/abs/2601.09373) | 该论文提出ImperfectiveNLI诊断数据集，揭示了大语言模型在未完成体悖论上普遍存在的“目的性偏差”——模型会系统性地臆想目标导向事件的完成结果、甚至无视文本中明确的取消表述，而提示干预虽能缓解该偏差却会引发校准危机、导致模型错误拒绝无终体动词的有效蕴含。 |
| [^171] | [PsyCLIENT: Client Simulation via Conversational Trajectory Modeling for Trainee Practice and Model Evaluation in Mental Health Counseling](https://arxiv.org/abs/2601.07312) | PsyCLIENT通过对话轨迹建模来驱动心理咨询来访者模拟，并配套构建了包含120个中文来访者画像的PsyCLIENT-CP数据集，经24名专业咨询师验证，其在真实性和培训实用性上优于现有方法，为新手咨询师训练与心理咨询系统评估提供了可扩展的新方案。 |
| [^172] | [RiskCueBench: Benchmarking Anticipatory Reasoning from Early Risk Cues in Video-Language Models](https://arxiv.org/abs/2601.03369) | 该论文提出了RiskCueBench基准，通过标注视频中最早出现风险信号的时刻，评估视频-语言模型仅凭早期视觉线索预判未来危险事件的能力，实验发现现有系统的前瞻性风险推理能力仍存在显著不足。 |
| [^173] | [ChemBART: Pre-training a Reaction Language Model for Comprehensive Synthesis Planning](https://arxiv.org/abs/2601.02915) | ChemBART是一种以反应为中心进行预训练的BART语言模型，通过在整体反应表达式上进行掩码填充预训练，能够统一解决前体/试剂生成、产率回归、分子性质分类等多种化学任务，并可与蒙特卡洛树搜索结合实现多步合成路线设计。 |
| [^174] | [Semantic Deception: When Reasoning Models Can't Compute an Addition](https://arxiv.org/abs/2512.20812) | 研究通过“语义欺骗”实验发现，当数字和运算符被重新定义并携带误导性语义关联时，LLMs难以维持真正的符号抽象推理，往往退回到利用训练中学到的语义关联。 |
| [^175] | [FLAWS: A Benchmark for Error Identification and Localization in Scientific Papers](https://arxiv.org/abs/2511.21843) | 该论文提出了FLAWS基准，通过在同行评审论文中系统性插入破坏关键论断的错误，构建了713个论文-错误对，以自动化评估大语言模型检测和定位科学论文中错误的能力。 |
| [^176] | [MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Domain Risks in LLMs](https://arxiv.org/abs/2511.07107) | 该论文提出元认知驱动的自进化框架MENTOR，通过涵盖教育、金融和管理领域的3000条标注查询数据集揭示出14个主流大语言模型平均57.8%的越狱成功率，并利用换位思考与后果推理等元认知策略进行自我评估、修订不安全响应并将验证后的规则编译为激活层面引导信号，从而有效缓解模型的隐性领域安全风险。 |
| [^177] | [In-Context Learning Beyond Copying: A Training-Time Analysis of Abstractive ICL](https://arxiv.org/abs/2511.05743) | 研究通过Hapax训练机制抑制归纳复制并消除相变后，发现抽象式上下文学习能力仍能涌现，表明ICL能力并不依赖于归纳复制机制。 |
| [^178] | [How Well Does Mathematical Reasoning Transfer Across Languages in English-Centric LLMs?](https://arxiv.org/abs/2510.02272) | 该研究系统评估了以英语为中心的大语言模型的跨语言数学推理迁移，发现迁移效果受初始模型、目标语言和训练范式的影响差异显著，且更强的英语初始化未必带来更高的迁移效率，而引入单一平行语言监督能带来最大提升。 |
| [^179] | [Domain-Specific Constitutional AI: Enhancing Safety in LLM-Powered Mental Health Chatbots](https://arxiv.org/abs/2509.16444) | 该论文提出了一种领域特定的宪法AI框架，通过将心理健康领域的专业准则（如危机干预协议和治疗指南）融入AI安全机制，以增强大语言模型驱动的心理健康聊天机器人的安全性。 |
| [^180] | [MoVoC: Morphology-Aware Subword Construction for Geez Script Languages Corrected version](https://arxiv.org/abs/2509.08812) | 本文提出 MoVoC 形态感知子词构建方法，将监督式形态分析与 BPE 相结合训练分词器 MoVoC-Tok 以保留形态完整性，并发布四种吉兹文字语言的词素标注数据（169,806 条），在四种语言中的三种取得最高的 MorphScore。 |
| [^181] | [ShIOEnv: A Command Evaluation Environment for Grammar-Constrained Synthesis and Execution Behavior Modeling](https://arxiv.org/abs/2505.18374) | ShIOEnv是一个与Gymnasium兼容的Bash命令评估环境，通过语法约束合成生成有效的shell输入，并记录完整的执行产物（标准输出、标准错误、退出状态和系统状态变化），为训练无需执行的shell模拟模型和部署网络欺骗蜜罐提供数据支持。 |
| [^182] | [Kongzi: A Historical Large Language Model with Fact Enhancement](https://arxiv.org/abs/2504.09488) | 孔子是一个结合历史语料继续预训练、监督微调与基于知识图谱事实奖励的强化学习的历史大语言模型，显著提升了历史问答和推理中的事实准确性。 |
| [^183] | [Social Life Simulation for Non-Cognitive Skills Learning](https://arxiv.org/abs/2405.00273) | 本文提出基于大语言模型的交互式社交生活模拟平台Simulife++，通过引入“贤者智能体”将人机交互创新性地扩展为人-AI-AI协作模式，支持用户在多样化社交场景中培养非认知技能并提升自我反思能力。 |

# 详细

[^1]: 面向思维剪枝的直接 Pass@$k$ 优化：在测试时推理中平衡质量与多样性

    Direct Pass@$k$ Optimization for Thought Pruning: Balancing Quality and Diversity in Test-Time Reasoning

    [https://arxiv.org/abs/2610.31667](https://arxiv.org/abs/2610.31667)

    该论文提出将 Pass@$k$ 直接作为思维剪枝的优化目标，从而在测试时推理中显式地平衡质量与多样性（即探索与利用）之间的权衡。

    

    测试时扩展已成为推理模型的一项关键推理策略：模型会生成大量候选解，以期能找到至少一个正确解。由于将所有候选解都运行到底在计算上代价高昂，近期的工作选择在仅生成一小段前缀后就对候选池进行剪枝，只为保留下来的候选分配后续计算资源。现有剪枝策略大致分为两类：基于质量的方法为每个部分解打分，并丢弃那些被预测最终会得到错误答案的部分解；而基于多样性的方法则选择遵循不同推理轨迹的部分解，从而更可能覆盖不同的最终解。这两个目标之间的张力正是探索与利用权衡的一个实例，然而此前关于剪枝的工作几乎没有直接暴露这一权衡……

    arXiv:2610.31667v1 Announce Type: cross  Abstract: Test-time scaling has emerged as a key inference strategy for reasoning models, where a model generates a large number of candidate solutions in the hope of finding at least one correct solution. Running all candidate solutions to completion can be computationally demanding, so recent work prunes the candidate pool after generating only a short prefix, committing further computation only to the surviving candidates. Existing pruning strategies divide broadly into two categories: Quality-based methods assign a score to each partial solution and discard those predicted to ultimately lead to an incorrect final solution, while diversity-based methods select partial solutions that follow different reasoning trajectories and are therefore likely to cover different final solutions. The tension between these two objectives is an instance of exploration versus exploitation, yet little to no prior work on pruning exposes this trade-off directly,
    
[^2]: 面向基于大语言模型的IT工单分类的成本感知弃权方法

    Cost-Aware Abstention for LLM-Based IT Ticket Classification

    [https://arxiv.org/abs/2610.31657](https://arxiv.org/abs/2610.31657)

    该论文提出一种离线的成本感知弃权评估框架，通过在校准数据上按显式成本选择并冻结路由策略，在真实IT工单数据上显著降低分类成本与错误路由率，并表明简单的TF-IDF逻辑回归结合成本感知阈值可优于大语言模型方案。

    

    IT支持工单的错误自动路由，其代价可能高于人工分诊。我们将成本感知弃权评估为一种离线的、成本条件下的就绪度评估方法：先在明确的成本设定下基于校准数据选择一个路由策略，随后仅在留出的测试数据上对该冻结策略进行一次评估。在一个包含47,837条工单、八个类别的公开语料库上，我们比较了四个经过训练的非大语言模型分类器与四个大语言模型在零样本和少样本提示下的表现。在示例性成本模型 (c_ok, c_triage, c_wrong) = (0, 1, 5) 下，基于校准数据选择的TF-IDF逻辑回归及0.60的阈值，将测试成本从0.763降至0.472，相对降低38.1%（配对比率自助法95%置信区间 [35.7%, 40.4%]），同时将错误路由率从15.3%降至3.5%。在事后成本参数分析中，冻结的校准选择策略还在16个设置中的14个里，在留出测试数据上同时优于“始终路由”和“始终弃权”两种基线。最强的p……（原文摘要截断）

    arXiv:2610.31657v1 Announce Type: cross  Abstract: Wrong automatic routing of an IT support ticket can cost more than human triage. We evaluate cost-aware abstention as an offline, cost-conditional readiness assessment: select a routing policy on calibration data under explicit costs, then evaluate that frozen policy once on held-out test data. On a public 47,837-ticket, eight-class corpus, we compare four trained non-LLM classifiers with four large language models under zero- and few-shot prompting. Under the illustrative cost model (c_ok, c_triage, c_wrong) = (0, 1, 5), TF-IDF with logistic regression and a threshold of 0.60 selected on calibration reduces test cost from 0.763 to 0.472, a 38.1% relative reduction (paired ratio-bootstrap 95% CI [35.7%, 40.4%]), while lowering wrong routes from 15.3% to 3.5%. Frozen calibration-selected policies also beat both always-route and always-defer on held-out test data in 14 of 16 settings in a post-hoc cost-parameter analysis. The strongest p
    
[^3]: 面向语言模型的低比特N-gram嵌入

    Low-Bit N-Gram Embeddings for Language Models

    [https://arxiv.org/abs/2610.31646](https://arxiv.org/abs/2610.31646)

    提出将n-gram嵌入表量化至四比特低精度存储，在从零训练的nanochat、Llama和GPT-2系列模型上节省约75%的表内存，同时保留超过99%的验证损失改善效果。

    

    许多近期的开放权重语言模型使用n-gram嵌入表来提升性能。这些表存储了短token序列（例如连续两个或三个token的组合）的学习表示，模型在处理过程中会检索这些表示。然而，它们可能需要大量内存：例如，DeepSeek-V4.1-Flash为其Engram n-gram内存分配了1960亿个参数。因此，我们研究了能否在保留嵌入表带来的性能提升的同时降低这一内存成本。我们提出以低精度存储n-gram嵌入，并系统地研究了不同的量化方法以及各表之间的比特分配方案。在从零开始训练的nanochat、Llama和GPT-2系列模型上，四比特的嵌入表可节省约75%的表内存，同时保留了添加嵌入表所带来的超过99%的验证损失改善。使用更少的比特则会导致更大的性能下降。

    arXiv:2610.31646v1 Announce Type: new  Abstract: Many recent open-weight language models use n-gram embedding tables to improve performance. These tables store learned representations of short token sequences, such as pairs or triples of consecutive tokens, which the model retrieves during processing. However, they can require substantial memory: DeepSeek-V4.1-Flash, for example, allocates 196 billion parameters to its Engram n-gram memory. We therefore investigate whether this memory cost can be reduced while preserving the performance improvement that the tables provide. We propose storing n-gram embeddings at low precision and systematically investigate different quantization methods and bit allocations across tables. Across nanochat-, Llama-, and GPT-2-family models trained from scratch, four-bit tables save approximately 75% of table memory while preserving more than 99% of the validation-loss improvement from adding the tables. Using fewer bits leads to a larger drop in performan
    
[^4]: 智能体能调试网络语料库吗？对斯洛伐克语 FineWeb2 的智能体化审计

    Can an Agent Debug a Web Corpus? Agentic Auditing of Slovak FineWeb2

    [https://arxiv.org/abs/2610.31636](https://arxiv.org/abs/2610.31636)

    本文提出将 LLM 作为一次性的“语料库调试器”而非分类器：由工具调用智能体审计 FineWeb2 斯洛伐克语子集，发现了包括 140 万篇机器生成内容农场在内的七项质量问题，并将发现编译为可执行的测试套件，从而以不到 3 美元的成本实现可复用、可验证的大规模语料库质量审计。

    

    像 FineWeb2 这样的多语言数据处理流水线将同一套方案应用于上千种语言，导致各语言特有的失败模式难以被预见。本文将 LLM 仅使用一次，作为“语料库调试器”而非分类器：一个具备工具调用能力的智能体通过只读、有预算限制的 API 探索语料库，并将每项发现编译为可执行的测试，再由母语者进行验证。在 FineWeb2 的 2650 万文档规模的斯洛伐克语子集上，三次运行（每次花费 2.21–2.84 美元）发现了涵盖四类现象的七项问题。其中规模最大的是一个包含 1,408,437 篇文档（占 5.32%）的机器生成内容农场，它们通过了流水线的全部过滤器，其中 86.6% 甚至通过了更严格的阈值。盲标标注支持七项发现中的六项（缺陷率为 0.53–1.00，而随机对照组仅为 0.08），一项未获支持。编译后的测试套件无需 GPU，仅用 0.7 秒实际时间即可扫描整个语料库，相比之下每次 LLM 分类扫描需花费 1.2 万至 9 万美元。

    arXiv:2610.31636v1 Announce Type: new  Abstract: Multilingual pipelines such as FineWeb2 apply one recipe across a thousand languages, leaving the failure modes of individual languages unanticipated. We use an LLM once, as a corpus debugger rather than a classifier: a tool-using agent explores the corpus through a read-only, budgeted API and compiles each finding into an executable test, which a native speaker then validates. On the 26.5M-document Slovak portion of FineWeb2, three runs costing \$2.21-\$2.84 each found seven findings over four phenomena. The largest is a machine-generated content farm of 1,408,437 documents (5.32%) that passed the pipeline's filters, 86.6% of it passing even stricter thresholds. Blinded labels support 6 of the 7 findings (defect rates .53-1.00 against .08 in a random control) and do not support one. The compiled suite scans the full corpus in 0.7 s of wall-clock time without a GPU, against \$12k-\$90k per LLM classification pass. The best static samplin
    
[^5]: 双语语言模型捕捉二语阅读中的跨语言影响

    Bilingual Language Models Capture Cross-Linguistic Influence in Second-Language Reading

    [https://arxiv.org/abs/2610.31620](https://arxiv.org/abs/2610.31620)

    该研究发现，使用与读者母语相匹配的双语语言模型能更好地预测二语（英语）阅读的眼动时间，这一优势主要体现在后期眼动指标中，受两种语言文字系统重叠程度的影响，并在中级语言水平的读者中最为显著。

    

    虽然单语语言模型（LM）的困惑度已被广泛确立为阅读时间的预测指标，但它无法捕捉受读者第一语言（L1）塑造的非母语（L2）阅读动态。在这项工作中，我们研究了双语模型内部的跨语言迁移是否与人类在二语阅读过程中由母语驱动的偏差相一致。我们在来自12种不同母语背景的读者阅读英文文本的眼动追踪数据上评估了双语模型。为了分离出特定母语的迁移效应，我们将基于与读者母语背景相匹配的语言训练的双语模型，与单语英语基线模型以及基于其他母语训练的双语模型进行比较。我们证明，匹配读者的母语能够改善对二语阅读时间的预测。这种预测优势集中于后期的眼动追踪指标中，受母语与二语文字系统重叠程度的调节，并在中等熟练程度的读者中达到峰值。这些发现提供了证据

    arXiv:2610.31620v1 Announce Type: new  Abstract: While monolingual language model (LM) surprisal is widely established as a predictor of reading times, it cannot capture the non-native (L2) reading dynamics that are shaped by the reader's first language (L1). In this work, we investigate whether the cross-lingual transfer within bilingual models aligns with human L1-driven biases during L2 reading. We evaluate bilingual models on eye-tracking data of readers from 12 diverse L1 backgrounds reading English texts. To isolate L1-specific transfer, we compare bilingual models trained on an L1 that matches the readers' L1 background to both a monolingual English baseline and bilingual models trained on other L1s. We demonstrate that matching the readers' L1 improves the prediction of L2 reading times. This predictive advantage is concentrated in late-stage eye-tracking measures, modulated by L1-L2 script overlap, and peaks among intermediate-proficiency readers. These findings provide eviden
    
[^6]: 真值标注如何塑造检索得分：面向乳腺病变管理的检索增强生成系统的证据感知评估

    How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions

    [https://arxiv.org/abs/2610.31606](https://arxiv.org/abs/2610.31606)

    该论文提出了一种证据感知的检索评估协议，通过将证据片段映射到所有包含块来定义相关性，从而解决检索得分依赖真值标注覆盖范围的问题，并将其应用于乳腺病变管理文献临床RAG系统的评估。

    

    目的：描述一种针对基于乳腺病变管理文献的临床检索增强生成（RAG）系统的证据感知检索评估协议，以解决检索得分对真值标注覆盖范围的依赖问题。方法：使用包含18篇文档的语料库和18条查询的开发基准，每篇文档对应一个“是/否”问题或特殊疑问句问题。检索结合了稠密向量嵌入、BM25和交叉编码器重排序。评估了128至1024个词元的分块大小、0%至20%的重叠率以及四种检索器配置。相关性由映射到所有包含块上的证据片段来定义，将包含等价证据的重叠块视为相关。检索效果采用R-Precision、Recall@K、MRR和nDCG@K进行评估。此外，生成质量在512词元索引上以K=1至5进行评估，采用DeepEval声明级指标（以deepseek-v4-pro作为LLM评判者）以及BLEU/ROUGE指标。结果：所选的混合【摘要在此处不完整，原文截断】

    arXiv:2610.31606v1 Announce Type: new  Abstract: Purpose: To describe an evidence-aware retrieval evaluation protocol for a clinical retrieval-augmented generation (RAG) system over breast lesion management literature, addressing the dependence of retrieval scores on annotation coverage. Methods: An 18-document corpus and an 18-query development benchmark, with one Yes/No or WH-question per document, were used. Retrieval combined dense embeddings, BM25, and cross-encoder reranking. Chunk sizes of 128 to 1024 tokens, overlaps of 0 to 20%, and four retriever configurations were evaluated. Relevance was defined by evidence spans mapped to all containing chunks, treating overlapping chunks with equivalent evidence as relevant. Retrieval was evaluated using R-Precision, Recall@K, MRR, and nDCG@K. Separately, generation was assessed on a 512-token index at K=1 to 5 using DeepEval claim-level metrics, with deepseek-v4-pro as the LLM judge, and BLEU/ROUGE metrics. Results: The selected hybrid 
    
[^7]: 大语言模型能匹配进入住院医师培训吗？开放语言模型在西班牙MIR考试中的不确定性、推理与图像利用

    Would an LLM Match into a Residency? Uncertainty, Reasoning and Image Use of Open Language Models on the Spanish MIR Examination

    [https://arxiv.org/abs/2610.31586](https://arxiv.org/abs/2610.31586)

    该研究按西班牙MIR住院医师考试的官方规则评估了四个开放大语言模型，发现三个较大模型的成绩可排进前600至3,000名并仍能匹配到41至43个专业，但其弃答策略从不划算，医学微调可减少过度自信，而其临床推理与图像利用能力仍存疑。

    

    arXiv:2610.31586v1 公告类型：新 摘要：在医学考试中的准确率是支持语言模型临床应用的常见证据，然而一个打算在自己的硬件上运行开放模型的医疗系统需要了解的远不止于此：模型是否知道自己何时可能出错，医学领域的微调能否提升这种能力，它的正确答案是否建立在合理的临床推理之上，以及它是否真的用到了提供给它的图像。我们在西班牙MIR住院医师考试上针对四个参数量从80亿到270亿的开放模型考察了这些问题，按照该考试自身的规则对八届考试共1,552道题目进行评分。在回答所有题目的情况下，三个较大的模型将排在约前600至3,000名考生之间，仍有41至43个专业可供其匹配。然而，放弃作答永远不合算，因为即便是它们最没把握的答案，其正确率也高于25%的盈亏平衡点。医学微调降低了过度自信……

    arXiv:2610.31586v1 Announce Type: new  Abstract: Accuracy on medical examinations is common evidence offered for the clinical use of language models, yet a health system that intends to run an open model on its own hardware needs to know considerably more: whether the model recognizes when it is likely to be wrong, whether medical fine-tuning improves this ability, whether its correct answers rest on sound clinical reasoning, or whether it actually uses the images it is given. We examine these questions for four open models of 8 to 27 billion parameters on the Spanish MIR residency examination, scoring 1,552 questions from eight editions under the exam's own rules. Answering every question, the three larger models would rank among roughly the first 600 to 3,000 candidates, with 41 to 43 specialties still open to them. Abstention, however, never pays, because even their least confident answers are right more often than the 25% break-even. Medical fine-tuning reduced overconfidence at th
    
[^8]: 面向困惑度感知黑盒攻击的扩散语言模型

    Diffusion Language Models for Perplexity-Aware Black-Box Attacks

    [https://arxiv.org/abs/2610.31558](https://arxiv.org/abs/2610.31558)

    该论文提出将连续潜空间攻击框架与连续扩散语言模型（dLLM）直接耦合，以学习到的语言生成分布取代人工随机映射，从而生成更自然、更难被防护机制检测且基于自然提示的黑盒攻击提示词。

    

    连续空间攻击通过在低维高斯潜空间中进行采样或优化，并将其通过随机映射和最近token查找解码为文本，从而找到能够触发黑盒大语言模型恶意行为（如过度思考）的提示词。由于这种随机映射并非从语言中学习得到，其生成的提示词可能非常不自然，因此容易通过输入防护机制被检测到，且其产生的风险估计也并未建立在自然提示的基础之上。在本文中，我们证明可以将同样的连续潜空间框架直接与从语言中学习得到的生成分布——连续扩散语言模型（dLLM）——相耦合，从而消除人为引入的随机映射，同时保留标准高斯潜空间形式以及下游的采样或优化流程。我们在两种设置下评估了该方法：使用 RareTrap 的基于采样的估计以及…

    arXiv:2610.31558v1 Announce Type: cross  Abstract: \emph{Continuous-space} attacks find prompts that trigger malicious behaviors, such as overthinking, in black-box LLMs by sampling or optimizing over a low-dimensional Gaussian latent space and decoding it to text through a \emph{random mapping} and nearest-token lookup. Because the random mapping is not learned from language, its prompts can be highly unnatural, and as such easily detectable through input guardrails, and the risk estimates they yield are not grounded in natural prompts. In this paper, we demonstrate that the same continuous latent-space framework can be coupled directly to a learned generative distribution over language, a continuous diffusion language model (dLLM), which removes the artificial random mapping while preserving the standard-Gaussian latent formulation and the downstream sampling or optimization procedure. We evaluate this approach under two settings: sampling-based estimation using \emph{RareTrap} and o
    
[^9]: 标准基准上最新大语言模型用于知识图谱到文本生成的实证研究

    An Empirical Study of Recent Large Language Models for Knowledge Graph to Text Generation on Standard Benchmarks

    [https://arxiv.org/abs/2610.31529](https://arxiv.org/abs/2610.31529)

    该研究在标准基准上系统对比了最新大语言模型与微调编码器-解码器模型在知识图谱到文本生成任务中的表现，发现LLM在基于参考的指标上大幅缩小甚至在部分设置下追平了差距，且在基于图谱的评估中产生更少的矛盾和三元组遗漏。

    

    知识图谱到文本生成任务旨在将一组知识图谱三元组转化为流畅的自然语言。虽然微调的编码器-解码器模型长期以来一直是标准方法，但最近的大语言模型（LLM）正变得日益具有竞争力。我们在标准基准上将最新的LLM与微调的编码器-解码器模型进行对比测试，采用基于参考文本的指标以及经过人工标注验证的、基于图谱的无参考LLM评估方法。我们观察到，最新的LLM在基于参考的指标上大幅缩小了与微调模型的差距，并在若干设置下追平了这一差距，同时在基于图谱的指标上表现更佳。基于LLM的评估分数与基于参考的指标仅呈现有限的相关性，这表明两种方法捕捉的是生成质量的不同方面。基于图谱的评估显示，提示式模型比微调模型产生更少的矛盾和三元组遗漏，而这些错误会随之增加……（原文截断）

    arXiv:2610.31529v1 Announce Type: new  Abstract: Knowledge graph-to-text (KG-to-text) generation verbalizes a set of knowledge graph triples into fluent natural language. While fine-tuned encoder-decoder models have long been the standard approach, recent LLMs are becoming increasingly competitive. We benchmark recent LLMs against fine-tuned encoder-decoder models, using reference-based metrics and graph-grounded reference-free LLM-based evaluation validated against human annotations. We observed that recent LLMs substantially narrow, and in several settings close, the gap with fine-tuned models on reference-based metrics, while performing better on graph-grounded metrics. LLM-based evaluation scores show limited correlation with reference-based metrics, suggesting that the two approaches capture different aspects of generation quality. Graph-grounded evaluation reveals that prompted models produce fewer contradictions and triple omissions than fine-tuned models, and these errors incre
    
[^10]: 隐喻的几何学：探测与引导大语言模型中的隐喻表示

    The Geometry of Metaphor: Probing and Steering Metaphor Representations in Large Language Models

    [https://arxiv.org/abs/2610.31520](https://arxiv.org/abs/2610.31520)

    该研究利用机制可解释性技术在大语言模型中定位隐喻的因果子空间，实现对分类和生成任务中隐喻输出的引导与控制。

    

    隐喻是人类认知的一个关键方面。因此，我们希望理解隐喻在大语言模型（LLM）中是如何被表示的，以及如何对其进行控制。我们使用机制可解释性技术，在涵盖不同架构和规模的一系列大语言模型中识别出隐喻相关的因果子空间。我们证明能够在分类任务上跨不同词性对输出进行引导，也能够在生成任务上引导输出，这表明这些技术可用于控制大语言模型输出的隐喻性。

    arXiv:2610.31520v1 Announce Type: new  Abstract: Metaphor is a crucial aspect of human cognition. As such, we would like to understand how it is represented in large language models (LLMs) and how it might be controlled. We use mechanistic interpretability techniques to identify causal subspaces in a range of LLMs covering different architectures and sizes. We show that we are able to steer output on classification tasks across different parts of speech. We are also able to steer output on generation tasks, indicating that these techniques can be used to control the metaphoricity of LLM output.
    
[^11]: 构建既忠于任务又听感自然的用户模拟器

    Towards User Simulators That Stay On Task And Still Sound Natural

    [https://arxiv.org/abs/2610.31512](https://arxiv.org/abs/2610.31512)

    本文提出一种强化学习方法，通过在交替对抗循环中训练轻量级判别器来将行为自然度可操作化，从而同时奖励任务意图遵循与对话自然度，并发布了由此得到的用户模拟器模型 UserLM-2.0。

    

    用户越来越依赖AI系统完成长时间的多轮任务，然而有证据表明，基于大语言模型（LLM）的系统在长期交互中性能会逐渐下降。为了在此类场景中评估和改进AI助手，我们需要既能忠实遵循复杂用户意图、又能展现真实对话行为的用户模拟器。现有方法存在一种权衡：通过提示实现的助手可以遵循复杂的用户意图，但缺乏行为保真度；而专门设计的用户模拟器虽能更好地逼近真实对话行为，却难以忠实于复杂意图。本文提出了一种强化学习（RL）方法，同时奖励任务意图遵循度和行为自然度。该方法的一个关键贡献是通过一个轻量级判别器以交替对抗循环的方式与用户模拟器共同训练，从而将“自然度”进行可操作化定义。我们发布了由此训练得到的模型 UserLM-2.0。

    arXiv:2610.31512v1 Announce Type: new  Abstract: Users increasingly rely on AI systems for extended, multi-turn tasks, yet evidence indicates that LLM-based system performance degrades over prolonged interaction. To evaluate and improve AI assistants in such settings, we need user simulators that faithfully follow complex user intents while exhibiting realistic conversational behavior. Existing approaches manifest a trade-off: prompted assistants can follow complex user intents but lack behavioral fidelity, while specialized user simulators better approximate conversational behavior but struggle to remain faithful to complex intents. In this paper, we propose a reinforcement learning (RL) recipe that jointly rewards task intent adherence and behavioral naturalness. A crucial contribution of the method is operationalizing naturalness with a lightweight discriminator trained in an alternating, adversarial loop with the user simulator. We release the resulting model, UserLM-2.0, which pro
    
[^12]: 测量验证器的下限：表面上的罕见词文本转语音错误中有多少其实是ASR错误？

    Measuring the Verifier's Floor: How Much Apparent Rare-Word Text-to-Speech Error Is ASR Error?

    [https://arxiv.org/abs/2610.31470](https://arxiv.org/abs/2610.31470)

    该研究量化了ASR评估系统本身的错误下限——即使在人类语音上也会漏掉33-67%的极罕见词——表明TTS评估中相当一部分“错误”实为识别器局限，多识别器交叉验证可将表面错误率降低两到三倍，但最终仍需人工听音测试裁定。

    

    文本转语音（TTS）通常通过使用自动语音识别（ASR）系统对其输出进行转录来评估，因此词错误率中也计入了识别器自身的错误。我们在测试句子的人类录音上测量了这些错误。在LibriSpeech和FLEURS数据集上，评估用的识别器即使在人类语音中也会漏掉33-67%的极罕见词，而三个识别器全部漏掉的比例为22-42%。用另外两个识别器家族对每个词进行核查后，对于至少有一个识别器能听到的词，表面上的TTS错误率降低了两到三倍（LibriSpeech中极罕见词的错误率从10-13%降至3-5%），但极罕见词仍然是最难的部分。忽略识别器在人类语音中漏掉的词是有偏见的：其他识别器在这些词上发现TTS错误的频率要高得多（4-6%对比0.2-0.3%），并推翻了这种掩蔽方法所保留的约一半错误。由于剩余经裁定的错误中可能有一半是拼写变体，因此仍需进行听音测试。

    arXiv:2610.31470v1 Announce Type: cross  Abstract: Text-to-speech (TTS) is usually evaluated by transcribing its output with an automatic speech recognition (ASR) system, so the word error rate also counts recognizer errors. We measure these errors on human recordings of the test sentences. On LibriSpeech and FLEURS, the evaluation recognizer misses 33-67% of very rare words even in human speech, and all three recognizers miss 22-42%. Checking every word with two more recognizer families lowers the apparent TTS error on words that at least one recognizer hears two- to three-fold (very rare LibriSpeech words: from 10-13% to 3-5%), yet very rare words remain the hardest. Ignoring words the recognizer misses in human speech is biased: the other recognizers find TTS errors far more often on these words (4-6% vs. 0.2-0.3%) and overturn about half of the errors this mask keeps. As half of the remaining adjudicated errors may be spelling variants, a listening test is needed.
    
[^13]: 从结构化记录到患者叙事：在大语言模型生成的健康摘要中保持临床一致性

    From Structured Records to Patient Narratives: Preserving Clinical Alignment in LLM-Generated Health Summaries

    [https://arxiv.org/abs/2610.31467](https://arxiv.org/abs/2610.31467)

    本文提出一种将结构化证据条件化与检索增强生成（RAG）相结合的提示约束生成框架，用于月经健康监测的可读摘要生成，并通过NLI、原子事实性和可读性评估证明，显式证据约束能够持续提升大语言模型生成摘要的临床语义一致性。

    

    随着可穿戴设备和数字健康系统日益产生纵向生理记录，如何用患者易于理解的语言传达这些观测结果仍然是一个重大挑战。大语言模型（LLM）为将结构化临床信息转化为可读的叙事提供了有前景的机制，但这一过程可能引入语义漂移，即在简化过程中临床上的精确含义被扭曲。我们针对月经健康监测研究了基于证据的健康摘要生成，采用一种基于提示的约束生成框架，将结构化证据条件化与检索增强生成（RAG）相结合。我们使用自然语言推理（NLI）、原子事实性指标和可读性度量来评估语义对齐。在三个大语言模型和4,500份生成的摘要中，显式的证据约束持续改善了临床一致性。

    arXiv:2610.31467v1 Announce Type: new  Abstract: As wearable and digital health systems increasingly generate longitudinal physiological records, communicating these observations in patient-accessible language remains a significant challenge. Large Language Models (LLMs) offer a promising mechanism for translating structured clinical information into readable narratives, but this process may introduce semantic drift, where clinically precise meaning is distorted during simplification. We investigate evidence-grounded health summarization for menstrual health monitoring using a constrained generation framework that combines structured evidence conditioning with Retrieval-Augmented Generation (RAG) using a prompt-based constrained generation framework. We evaluate semantic alignment using Natural Language Inference (NLI), atomic factuality metrics, and readability measures. Across three LLMs and 4,500 generated summaries, explicit grounding constraints consistently improve clinical align
    
[^14]: 差异多大，还是如何差异？让用户接触替代观点的对话系统中观点选择的立场粒度问题

    How Different, or Different How? Stance Granularity in Viewpoint Selection for Dialog Systems that Engage Users with Alternative Views

    [https://arxiv.org/abs/2610.31459](https://arxiv.org/abs/2610.31459)

    本文提出一种由多臂老虎机驱动的观点选择策略，将立场表示从标量分数细化为概念层面的“视角化立场向量”，发现概念级立场能让用户接触到更多样、覆盖更广的观点，但不会改变用户论证和重构观点的深度与准确性。

    

    旨在帮助用户接触与自己不同观点的系统，必须决定这些观点应当有多不同。目前观点主要以标量尺度表示，这种表示过于粗糙，无法区分那些在与用户不同的方面上存在分歧的观点。我们提出了一种选择策略：由多臂老虎机设定观点与用户之间允许的距离，策略在该界限内选择最新颖的观点。我们仅改变其中的立场编码器，将标量分数与“视角化立场向量”进行比较，后者将一个立场分解为针对各个概念的独立立场。我们将其集成到一个对话系统中，并开展了一项被试间实验（N=63）。结果表明，概念层面的立场能够带来更多样化的接触序列和更广泛的话题立场覆盖范围，但并未改变用户论证观点的深度或重构观点的准确性。最后，我们在概念层面分析了用户与观点之间的关系，展示……

    arXiv:2610.31459v1 Announce Type: cross  Abstract: Systems that aim to help a user engage with viewpoints other than their own have to decide how different those viewpoints should be. Viewpoints are mainly represented on a scalar scale, too coarse to distinguish viewpoints that disagree with the user about different things. We introduce a selection policy in which a bandit sets how far viewpoints may sit from the user, and the policy selects the most novel viewpoint within that bound. We vary only its stance encoder, comparing a scalar score with Perspectivized Stance Vectors, which break a position into separate stances on concepts. We integrate it inside a dialog system and run a between-subjects study (N=63). Concept-level stance produces more varied exposure sequences and broader coverage of the topic's positions, but does not change how deeply users justify viewpoints or how accurately they reconstruct them. Finally, we analyze user--viewpoint relations at the concept level, showi
    
[^15]: 多Token预测与替代分词方法提升大规模潜在多跳图搜索的学习能力

    Multi-Token Prediction and Alternative Tokenization Improve Learning Latent Multi-Hop Graph Search at Scale

    [https://arxiv.org/abs/2610.31439](https://arxiv.org/abs/2610.31439)

    本文指出模型难以学习大规模图上最短路径搜索的两个原因——监督信号与分词方式，并提出多Token预测和替代分词这两种相互正交的方法，突破了以往工作的规模瓶颈。

    

    给定一个未见过的图、一个起始节点 $s$ 和一个终止节点 $t$，语言模型能否生成从 $s$ 到 $t$ 的最短路径？先前的研究发现，随着图规模的扩大，模型无法学会这一任务。我们证明这种失败有两个原因：监督信号和分词方式。首先，我们从理论上建立并从实证上证明了多Token预测（MTP）的有效性，并表明MTP的益处与图的拓扑结构无关。其次，我们引入了替代分词方法，并证明这两个原因是相互正交的。利用这些方法，我们突破了先前研究观察到的规模瓶颈，在远超以往极限的规模上持续且稳定地学习解决该任务。

    arXiv:2610.31439v1 Announce Type: cross  Abstract: Given an unseen graph, a start node, $s$, and an end node, $t$, can a language model generate the shortest path from $s$ to $t$? Prior work has found that models fail to learn this task as the graphs scale. We show that this failure has two causes: the supervision signal and the tokenization. First, we theoretically establish and empirically demonstrate the effectiveness of multi-token prediction (MTP), and show that the benefits of MTP are topologically agnostic. Second, we introduce alternative tokenizations and show that the two causes are orthogonal. Using these methods, we break through the scaling wall observed by prior work, consistently and stably learning to solve the task at scales far beyond previous limits.
    
[^16]: 决策模型对越南语的迁移效果如何？一项文本分类的实证研究

    How Well Do Decision Models Transfer to Vietnamese? An Empirical Study of Text Classification

    [https://arxiv.org/abs/2610.31422](https://arxiv.org/abs/2610.31422)

    本文在13个越南语文本分类设置中评估了四种专用决策模型，发现JEV在英语标签描述下表现最佳，双语标签描述并无一致提升，而去除声调符号对模型性能的影响最为严重。

    

    专用决策模型的最新进展为零样本分类拓展了新的可能性，但它们在越南语上的有效性仍未得到充分探索。本研究在涵盖六个任务组的13个分类设置中评估了四种专用决策模型，考察了它们的零样本性能、对标签描述语言的敏感性以及对越南语语言变化的鲁棒性。评估考虑了英语、越南语和双语标签描述，并涉及网络缩写语、拼写错误和声调符号去除等常见语言扰动。在英语标签描述下，JEV取得了最强的整体性能，在11个分类设置中排名第一。标签描述语言的影响因模型和数据集而异，双语描述并未带来一致的提升。鲁棒性实验表明，去除声调符号造成了最显著的性能下降。

    arXiv:2610.31422v1 Announce Type: new  Abstract: Recent advances in specialized decision models have expanded the possibilities for zero-shot classification, yet their effectiveness on Vietnamese remains underexplored. This study evaluates four specialized decision models across 13 classification settings spanning six task groups, examining zero-shot performance, sensitivity to label-description language, and robustness to Vietnamese linguistic variations. The evaluation considers English, Vietnamese, and bilingual label descriptions, alongside common linguistic perturbations involving teencode, typographical errors, and diacritic removal. Under English label descriptions, JEV achieves the strongest overall performance, ranking first in 11 classification settings. The effects of label-description language vary across models and datasets, with bilingual descriptions providing no consistent improvement. Robustness experiments reveal that diacritic removal causes the most substantial perf
    
[^17]: 深度增长与循环如何塑造语言模型中的上下文学习

    How Depth Growth and Looping Shape In-Context Learning in Language Models

    [https://arxiv.org/abs/2610.31421](https://arxiv.org/abs/2610.31421)

    深度增长与循环语言模型通过跨层复用计算形成了偏向从上下文学习的归纳偏置，使其比稠密模型更早获得且更强保持上下文学习能力，其机制在于深度增长将早期归纳头传播到各层。

    

    循环与深度增长语言模型通过在深度方向上复用计算，并因其推理能力而受到关注。基于先前关于上下文学习增强的提示性证据，我们研究了这种复用如何有利于上下文学习，包括它如何塑造该能力的获取、保持及内在机制。我们的分析涵盖高达30亿参数的预训练语言模型以及受控任务。实验为一种共享的、偏向于从上下文学习的归纳偏置提供了证据：深度增长和循环模型更依赖上下文中的信息，与全深度稠密参考模型相比，它们在受控任务中更早获得上下文学习能力，并且即使在权重中学习固定关联也成为可能的情况下，仍能更强地保持这种能力。通过追踪训练过程中的注意力头，我们发现深度增长会将早期形成的归纳头传播到各个层。消融实验表明，深度增长和循环模型对归纳……（摘要原文在此处截断）

    arXiv:2610.31421v1 Announce Type: new  Abstract: Looped and depth-grown language models reuse computations across depth and have attracted interest for their reasoning capabilities. Building on prior suggestive evidence of improved in-context-learning (ICL), we investigate how such reuse favors ICL, including how it shapes the acquisition, retention, and mechanisms of this ability. Our analysis spans pretrained language models up to 3B parameters and controlled tasks. The experiments provide evidence for a shared inductive bias toward learning from context: grown and looped models rely more on information in context and, compared with a full-depth dense reference, acquire ICL earlier in controlled tasks and retain it more strongly when learning fixed associations in the weights is also possible. Tracing attention heads through training reveals that depth growth propagates early induction heads across layers. Ablations show that grown and looped models rely more strongly on induction-ra
    
[^18]: SyRA-CS：基于元学习的码切换语音识别合成到真实自适应

    SyRA-CS: Synthetic-to-Real Adaptation for Code-Switched ASR with Meta-Learning

    [https://arxiv.org/abs/2610.31415](https://arxiv.org/abs/2610.31415)

    提出基于Reptile元学习的SyRA-CS框架，利用合成码切换语音学习易于自适应的模型初始化，仅需一小时真实目标语音即可在所有16个模型-语言对组合上超越持续微调的码切换语音识别性能。

    

    码切换自动语音识别（ASR）仍然是一个难题，因为真实的码切换语音非常稀缺，而合成语音又无法完全匹配真实语音。我们研究合成码切换语音是否更适合用于学习一个利于自适应的初始化，而不是用于普通的持续微调。我们提出了SyRA-CS，一个基于Reptile元学习的、针对特定语言对的合成到真实自适应框架。对于每个语言对和预训练模型，我们在相应的合成CS-FLEURS子集上进行元训练，然后仅使用一小时带标注的真实目标语音进行自适应。我们评估了四种架构设置：Encoder-CTC、FastConformer-RNN-T、注意力编码器-解码器（AED）以及语音大语言模型（Speech-LLM）。实验覆盖阿拉伯语-英语、法语-英语、德语-英语和印地语-英语等语言对，并与仅合成数据迁移、仅真实数据微调以及持续微调方法进行对比。SyRA-CS在全部16个模型-语言对组合的WER指标以及15个组合的CER指标上均优于持续微调。

    arXiv:2610.31415v1 Announce Type: new  Abstract: Code-switched (CS) automatic speech recognition (ASR) remains difficult because real CS speech is scarce and synthetic speech does not fully match it. We investigate whether synthetic CS speech is more useful for learning an adaptation-friendly initialization than for ordinary continual fine-tuning. We propose SyRA-CS, a pair-specific synthetic-to-real adaptation framework based on Reptile meta-learning. For each language pair and pretrained model, we meta-train on the corresponding synthetic CS-FLEURS subset and then adapt using one hour of labeled real target speech. We evaluate four architectural settings: Encoder-CTC, FastConformer-RNN-T, Attention Encoder-Decoder (AED) and Speech-LLM. Experiments cover on Arabic-, French-, German-, and Hindi-English, against synthetic-only transfer, real-only fine-tuning, and continual fine-tuning. SyRA-CS improves over continual fine-tuning on WER in all 16 model-pair comparisons and on CER in 15 o
    
[^19]: SQuAT：球面格向量量化感知训练

    SQuAT: Spherical Lattice Vector Quantization-Aware Training

    [https://arxiv.org/abs/2610.31412](https://arxiv.org/abs/2610.31412)

    该论文提出SQuAT方法，通过分离的直通估计器将权重分解为方向（用E_8球面码本量化）和增益（用低比特标量量化器量化），将基于格的向量量化感知训练成功扩展到十亿参数级大语言模型，并实现了隐式的球面（黎曼）优化。

    

    向量量化（VQ），特别是基于格的方法，为低比特大语言模型（LLM）提供了强大的率失真权衡，但其在量化感知训练（VQ-QAT）下的行为仍未被充分理解。我们将格VQ-QAT扩展至十亿参数规模的LLM和数十亿token的训练场景，并提出SQuAT——一种形状-增益式VQ-QAT方法，采用针对形状和增益的分离直通估计器（STE）进行训练。SQuAT将权重分解为方向和幅度两个部分：归一化方向使用由$E_8$推导的球面码本进行量化，而增益则使用低比特标量量化器独立量化。我们证明，与应用于完整量化器、忽略形状-增益结构的标准STE不同，这些分离的STE会在切空间中为方向引入更新，从而形成一种隐式的球面（黎曼）优化形式。为实现可扩展性，我们开发了一种分布式……（原文截断）

    arXiv:2610.31412v1 Announce Type: cross  Abstract: Vector quantization (VQ), in particular lattice-based methods, offer strong rate-distortion trade-offs for low-bit large language models (LLMs), yet their behavior under quantization-aware training (VQ-QAT) remains poorly understood. We scale lattice VQ-QAT to billion-parameter LLMs and multi-billion-token training regimes, and introduce SQuAT, a shape-gain VQ-QAT approach trained with a separate straight-through estimator (STE) for the shape and gain. SQuAT decomposes weights into direction and magnitude: normalized directions are quantized using an $E_8$-derived spherical codebook, while gains are quantized independently with low-bit scalar quantizers. We show that, unlike a standard STE applied to the full quantizer, which ignores the shape-gain structure, these separate STEs induce tangent-space updates for directions, yielding an implicit form of spherical (Riemannian) optimization. To enable scalability, we develop a distributed 
    
[^20]: 同一实体，不同文字：诊断语言模型中的跨文字实体匹配失败

    Same Entity, Different Script: Diagnosing Cross-Script Entity Matching Failures in Language Models

    [https://arxiv.org/abs/2610.31407](https://arxiv.org/abs/2610.31407)

    语言模型判断不同文字书写的同一人名是否为同一人的能力存在巨大差距，同一人判断准确率从拉丁文的35.1%骤降至天城文的8.5%和马拉雅拉姆语的5.3%，该效应在多个模型、两种文字和语系中均存在，且无法通过减少分词碎片化来弥补。

    

    多语言语言模型在相同人名以不同文字书写时，可能会给出不同的身份判断。我们使用601对经过审核的身份配对来研究这一效应，这些配对具有经证实的天城文和拉丁文形式，并保持问题和关系固定不变。在马拉地语提示下，Qwen2.5-7B的同一人判断准确率从拉丁文的35.1%降至天城文的8.5%，而不同人判断的准确率保持在98%以上。对五个模型的评估揭示了各自不同的响应偏差，且减少分词碎片化并不能可靠地缩小这一差距。在使用由相同实体构建的374对经过审核的马拉雅拉姆语配对复制该实验设计时，该效应在第二种文字和语系中得到复现：同一人判断准确率降至5.3%，约为匹配的天城文比例的一半，而其区分度仍以0.925的AUROC将匹配姓名排在非匹配姓名之上。用拉丁文训练的探针可以从天城文的内部状态中恢复标签信息，超过了字符串（原文在此处截断）

    arXiv:2610.31407v1 Announce Type: new  Abstract: Multilingual language models can give different identity judgments when the same people's names are written in different scripts. We study this effect using 601 audited identity pairs with attested Devanagari and Latin forms, holding the question and relation fixed. Qwen2.5-7B's same-person accuracy falls from 35.1% in Latin to 8.5% in Devanagari under Marathi prompts, while different-person accuracy remains above 98%. Evaluation across five models reveals distinct response biases, and reduced tokenization fragmentation does not reliably close the gap. Replicating the design on 374 audited Malayalam pairs built from the same entities reproduces the effect in a second script and language family: same-person accuracy falls to 5.3%, about half the matched Devanagari rate, while the margin still ranks matching above non-matching names at 0.925 AUROC. Latin-trained probes recover label information from Devanagari states, exceeding a string-fe
    
[^21]: HERD：面向持续学习的工具链演化递归蒸馏

    HERD: Harness-Evolving Recursive Distillation for Continual Learning

    [https://arxiv.org/abs/2610.31401](https://arxiv.org/abs/2610.31401)

    提出HERD混合持续学习框架，通过可演化的工具链高效学习，并以双锚点漂移约束蒸馏将经验内化为参数化累积能力，从而在持续获取新知识的同时大幅减少遗忘。

    

    持续学习要求模型在不损害既有能力的前提下不断获取新技能与新知识。参数化方法面临灾难性遗忘和可塑性丧失的困扰；非参数化方法（如提示、技能和工具链）虽借助上下文学习，却受限于上下文长度和有损的纯文本压缩。为有效结合二者的互补优势，我们提出HERD（工具链演化递归蒸馏）——一个混合式持续学习框架，它在工具链中高效学习，并通过蒸馏将经验内化为累积能力。在工具链方面，HERD持续精炼其技能以捕捉可复用的模式，同时在有限记忆中保留技能无法覆盖的高残差样本；在参数方面，HERD采用带双锚点的漂移约束蒸馏，在长递归过程中显著减少遗忘。

    arXiv:2610.31401v1 Announce Type: new  Abstract: Continual learning requires a model to keep acquiring new skills and knowledge without compromising its existing capabilities. Parametric methods suffer from catastrophic forgetting and loss of plasticity. Non-parametric methods, such as prompts, skills, and harnesses, leverage in-context learning but are limited by context length and lossy plain-text compression. To effectively combine their complementary strengths, we propose HERD (Harness-Evolving Recursive Distillation), a hybrid continual learning framework that learns efficiently in the harness and internalizes experience into cumulative capabilities via distillation. On the harness side, HERD continually refines its skills to capture reusable patterns, while retaining high-residual examples that the skills fail to capture in a bounded memory. On the parameter side, HERD performs drift-constrained distillation with dual anchors, substantially reducing forgetting over long recursive
    
[^22]: 大语言模型向低资源黏着语的参数高效适配：面向巴什基尔语的语料库、数据集与可复现基准

    Parameter-Efficient Adaptation of Large Language Models to a Low-Resource Agglutinative Language: A Corpus, Datasets, and a Reproducible Benchmark for the Bashkir Language

    [https://arxiv.org/abs/2610.31397](https://arxiv.org/abs/2610.31397)

    该论文发布了巴什基尔语的开放语料库、标注数据集及系统性适配基准，证明基于QLoRA的参数高效适配结合持续预训练，可使3.8-8B参数的大语言模型在该低资源黏着语的文本分类任务上显著超越传统编码器模型微调。

    

    巴什基尔语是一种使用西里尔字母书写的低资源黏着语，在现代大语言模型中的代表性仍然不足。我们提出了一套开放资源套件和适配方法的系统性基准测试：巴什基尔语网络语料库（71,567篇文档、4690万词元、16个来源），以及三个分别用于二分类、多分类和多标签文本分类的标注数据集；该基准测试涵盖前两种分类任务。实验使用了八种架构（参数规模82M-8B：DistilGPT2、GPT-2 base/medium、Phi-3-mini、Mistral-7B、YandexGPT-5-Lite-8B、mBERT、XLM-R）和三种适配方法（全量微调、QLoRA、上下文学习），并辅以持续预训练（CPT）。实验结果表明：（1）在二分类任务上，对3.8-8B参数模型应用QLoRA优于仅编码器模型的微调：结合CPT的YandexGPT-5-Lite-8B达到宏平均F1值0.748，而XLM-R为0.653、mBERT为0.650；经Holm校正后两两差异均显著（p<0.001）。（2）YandexGPT-5-Lite-8B显著优于Mistral-7B和Ph……（原文摘要在此处截断）

    arXiv:2610.31397v1 Announce Type: new  Abstract: Bashkir is a low-resource agglutinative language written in Cyrillic and remains poorly represented in modern LLMs. We present an open resource suite and a systematic benchmark of adaptation methods: the Bashkir Web Corpus (71,567 documents, 46.9M tokens, 16 sources) and three labelled datasets for binary, multiclass and multilabel text classification; the benchmark covers the first two. Experiments use eight architectures (82M-8B: DistilGPT2, GPT-2 base/medium, Phi-3-mini, Mistral-7B, YandexGPT-5-Lite-8B, mBERT, XLM-R) and three adaptation groups (full fine-tuning, QLoRA, in-context learning), plus continual pretraining (CPT). (1) QLoRA on 3.8-8B models outperforms encoder-only fine-tuning on binary classification: YandexGPT-5-Lite-8B with CPT reaches macro-F1 0.748 vs 0.653 (XLM-R) and 0.650 (mBERT); pairwise differences are significant after Holm correction (p<0.001). (2) YandexGPT-5-Lite-8B significantly outperforms Mistral-7B and Ph
    
[^23]: 长文档视觉问答的任务中心综述：分类体系、演进与未来方向

    A Task-Centric Survey of Long-Document Visual Question Answering: Taxonomy, Evolution, and Future Directions

    [https://arxiv.org/abs/2610.31385](https://arxiv.org/abs/2610.31385)

    该论文提出首个以任务为中心的长文档视觉问答综述，将其分解为文档解析、证据组织与检索、生成器组织与学习及评估四个组件，建立细粒度分类体系，并通过时间维度分析梳理研究重点从多模态内容理解到证据获取与管理的演进脉络及未来方向。

    

    视觉文档理解对于从视觉内容丰富的文档中获取知识至关重要，正吸引着越来越多的研究兴趣。现有综述大多侧重于模型架构或技术方法，而对任务瓶颈如何塑造技术选择与研究优先级的关注有限。我们提出了一项以任务为中心的综述，将长文档视觉问答（VQA）分解为四个组成部分：文档解析、证据组织与检索、生成器组织与学习，以及评估。针对每个部分，我们构建了细粒度的分类体系，并梳理了研究动机、技术应对方案以及各组件之间的关联。我们聚焦于显式检索，同时也涵盖直接阅读方式。我们通过时间维度的分析，追踪了研究重点的演进过程：从多模态内容理解，扩展到充分且高保真的证据获取，再到持续的证据管理。我们进一步展望了证据充分性、可验证性等方面的未来研究方向。

    arXiv:2610.31385v1 Announce Type: cross  Abstract: Visual document understanding is essential for acquiring knowledge from visually rich documents and is attracting growing research interest. Existing surveys emphasize architectures or technical approaches, with limited attention to how task bottlenecks shape technical choices and research priorities. We present a task-centric survey that decomposes long-document visual question answering (VQA) into four components: document parsing, evidence organization and retrieval, generator organization and learning, and evaluation. For each, we develop a fine-grained taxonomy and trace motivations, technical responses, and cross-component connections. We focus on explicit retrieval while covering direct reading. Our temporal analysis traces expanding priorities from multimodal content understanding to sufficient, high-fidelity evidence acquisition and sustained evidence management. We outline future directions in evidence sufficiency, verifiable
    
[^24]: AfterDeleteBench：审计大语言模型智能体中删除后的信息持久性

    AfterDeleteBench: Auditing Post-Deletion Information Persistence in LLM Agents

    [https://arxiv.org/abs/2610.31366](https://arxiv.org/abs/2610.31366)

    本文提出AfterDeleteBench受控审计基准，揭示了LLM智能体删除原始记忆记录后，其派生状态（如摘要、分散存储的组件）仍可在约三分之二的案例中导致已删除的秘密信息被恢复，证明“删除”操作并不能保证信息真正被移除。

    

    许多LLM智能体维护外部持久记忆，并从过去的交互中派生出额外的状态，例如摘要。因此，删除原始记忆记录并不一定会移除由其派生内容所承载的信息。我们提出了AfterDeleteBench，这是一个针对固定权重LLM智能体中该失效模式的受控行为审计。对于每个合成的秘密信息，我们将一个先暴露后删除该秘密的智能体，与一个其他条件完全匹配但从未接触过该秘密的智能体进行比较，从而能够将删除后的信息恢复归因于最初的暴露。在我们的主要设置中，单次查询即可在大约三分之二的情况下从Qwen2.5-7B-Instruct中恢复已删除的秘密，而匹配的从未暴露对照组的恢复率为零。至关重要的是，恢复并不依赖于存活的逐字副本：当完整的秘密字符串已不存在、仅有按公开组合规则分别存储的组件时，精确重构的成功率仍达到62-（摘要在此处被截断）

    arXiv:2610.31366v1 Announce Type: cross  Abstract: Many LLM agents maintain external persistent memory and derive additional state from past interactions, such as summaries. Deleting an original memory record therefore need not remove information carried by its descendants. We introduce AfterDeleteBench, a controlled behavioral audit for this failure mode in fixed-weight LLM agents. For each synthetic secret, we compare an exposed-then-deleted agent with an otherwise matched agent that was never exposed to the secret, allowing post-deletion recovery to be attributed to the original exposure. In our primary setting, a single query recovers the deleted secret from Qwen2.5-7B-Instruct in roughly two thirds of cases, while matched never-exposed controls have zero recovery. Crucially, recovery does not require a surviving verbatim copy: when the complete secret string is absent and only separately stored components remain under a public composition rule, exact reconstruction succeeds in 62-
    
[^25]: CrisiLangs：用于危机预警生成的多语言资源

    CrisiLangs: A Multilingual Resource for Crisis Warning Generation

    [https://arxiv.org/abs/2610.31356](https://arxiv.org/abs/2610.31356)

    该论文提出了CrisiLangs——将英语危机警告数据集CrisiText扩展至意大利语、保加利亚语、德语和希腊语的多语言资源，包含五种语言共5,000个事件-警告信息对，可用于微调多语言模型以在危机期间生成多语言警告信息。

    

    在紧急情况下与公众沟通需要及时发布警告信息。在欧洲，这通常需要以多种语言制作这些信息。近期研究表明，可以训练语言模型在危机发生时生成警告信息，但现有数据仅涵盖英语。为了填补这一空白，我们提出了CrisiLangs，这是CrisiText（一个将危机事件与根据专家沟通指南编写的警告信息相配对的英语数据集）的多语言扩展版本，涵盖意大利语、保加利亚语、德语和希腊语。母语专家翻译了来自411个场景中1,000个事件的警告信息，五种语言总计5,000个事件-信息对。我们测试了利用该资源在全部五种语言上联合微调多语言模型的效果，分别在提示中包含与不包含应急指南两种情况下进行，并与零样本基线进行比较。微调后的模型表现优于基线。

    arXiv:2610.31356v1 Announce Type: new  Abstract: Communicating with the public during an emergency entails issuing warning messages promptly. In Europe, it often requires producing them in more than one language. Recent work has shown that language models can be trained to generate warning messages as a crisis unfolds, but the available data covers only English. To fill this gap, we present CrisiLangs, a multilingual extension of CrisiText (an English dataset of crisis events paired with warning messages written according to expert communication guidelines) to Italian, Bulgarian, German, and Greek. Expert native speakers translated the warning messages of 1,000 events drawn from 411 scenarios, for a total of 5,000 event-message pairs across five languages. We test the resource to fine-tune a multilingual model on all five languages jointly, with and without the emergency guidelines in the prompt, and compare it with a zero-shot baseline. The fine-tuned models outperform the baseline on
    
[^26]: SurveyGraph：基于方法-实体知识图谱生成学术综述

    SurveyGraph: Generating Academic Surveys via Methods-Entity Knowledge Graphs

    [https://arxiv.org/abs/2610.31317](https://arxiv.org/abs/2610.31317)

    SurveyGraph通过构建编码方法级后续、竞争与资源共享关系的“方法-实体知识图谱”（MEKG）来生成学术综述，使图感知检索的召回率相比扁平文本基线近乎翻倍，并显著提升了生成内容的引用支持率。

    

    大多数自动化综述生成系统将论文作为扁平文档进行检索，限制了其捕捉研究结构的能力。我们提出SurveyGraph，一个围绕方法-实体知识图谱（MEKG）构建的框架，该图谱编码了方法层面的后续研究、竞争以及共享资源关系。该图谱支持核心论文发现、检索、混合大纲构建以及基于证据门控的叙事综合。我们评估了检索、语义边、引用依据、文献集覆盖率以及生成阶段组件。在一项受控的七种方法等预算基准测试中，以MEKG为锚点的融合方法在召回率与结构密度之间取得平衡，并通过独立的共被引检验加以验证。在另一项独立的存档图原生诊断中，图感知检索在八次运行中的召回率几乎是扁平词汇基线的两倍（p = 0.004）。语义证据门控达到AUC = 0.75。对160个生成句子的审计发现，80.6%的句子具有锚定或部分的引用支持。

    arXiv:2610.31317v1 Announce Type: new  Abstract: Most automated survey generation systems retrieve papers as flat documents, limiting their ability to capture research structure. We present SurveyGraph, a framework built around a Methods-Entity Knowledge Graph (MEKG) encoding method-level follow-on, competition, and shared-resource relations. The graph supports core-paper discovery, retrieval, hybrid outlining, and evidence-gated narrative synthesis. We evaluate retrieval, semantic edges, citation grounding, collection coverage, and generation-stage components. In a controlled seven-method equal-budget benchmark, MEKG-anchored fusion balances recall and structural density, with an independent co-citation check. In a separate archived graph-native diagnostic, graph-aware retrieval nearly doubles recall over flat lexical baselines across eight runs (p = 0.004). A semantic evidence gate achieves AUC = 0.75. Audits of 160 generated sentences find 80.6% anchored-or-partial citation support.
    
[^27]: 当大语言模型忘记自己所信奉的内容时：刻画长程对话中的状态完整性崩塌

    When LLMs Forget What They Believe: Characterizing State Integrity Collapse in Long-Horizon Dialogues

    [https://arxiv.org/abs/2610.31314](https://arxiv.org/abs/2610.31314)

    该论文首次定义并量化了长程对话中的“状态完整性崩塌”现象——大语言模型虽在记忆中保留了约束却停止执行——并通过StateTrack-Bench基准与分叉式记忆审计证明，绝大多数失败（88.6%）是行为漂移而非普通遗忘。

    

    大语言模型能够复述一条指令，却无法持续遵守它。我们在长程对话中研究了这种“记忆—服从分离”现象，并定义了“状态完整性崩塌”：模型在记忆中保留了已声明的约束，但在后续对话轮次中不再将其付诸执行。我们提出了StateTrack-Bench，这是一套受控的对话基准，涵盖数值、格式、角色/人设、事实记忆以及逻辑规则五类约束。在七个固定版本的模型快照和五个对话深度探测点上，严格的联合保持率介于62.4%至100.0%之间，且性能退化主要集中在符号类约束以及一个特别脆弱的模型上。一项分叉式记忆审计显示，88.6%的观测失败属于行为漂移失败，而非普通的遗忘；在排除该脆弱异常模型后，这一诊断依然成立（84.4%的失败）。在长达50轮对话深度的扩展评估进一步证实，这种分离现象在核心基准之外依然持续存在。

    arXiv:2610.31314v1 Announce Type: new  Abstract: Large language models can repeat an instruction while failing to keep obeying it. We study this recall-compliance dissociation in long-horizon dialogue and define State Integrity Collapse: a model preserves a stated constraint in memory but stops applying it to later turns. We introduce StateTrack-Bench, a controlled suite of conversations covering numerical, format, role/persona, factual-memory, and logical-rule constraints. Across seven provider-pinned model snapshots and five dialogue-depth probes, strict joint retention ranges from 62.4% to 100.0%, with degradation concentrated in symbolic constraints and in one especially brittle model. A forked recall audit shows that 88.6% of observed failures are behavioral-drift failures rather than ordinary forgetting; excluding the brittle outlier preserves this diagnosis (84.4% of failures). Extended evaluations at depths up to 50 turns confirm that the dissociation persists beyond the core b
    
[^28]: Cola2：具有连续语义压缩与平衡离散生成的协同自适应分层语言建模

    Cola2: Co-Adaptive Hierarchical Language Modeling with Continuous Semantic Compression and Balanced Discrete Generation

    [https://arxiv.org/abs/2610.31305](https://arxiv.org/abs/2610.31305)

    Cola2 提出了一种协同自适应的分层语言建模框架，通过动态语义分段和无辅助损失的负反馈来控制连续语义压缩率，并结合块因果掩码扩散解码器与无分类器引导，实现语义重建与自主生成之间的平衡。

    

    分层语言生成将连续语义建模与离散文本实现分离开来，但其有效性取决于潜在空间压缩了什么内容，以及离散生成器如何利用这些内容。我们提出 Cola2，这是一个联合研究连续语义压缩、重建与生成之间的平衡以及连续-离散协同适应的框架。Cola2 动态地将文本分割为语义片段，通过无辅助损失的负反馈机制控制压缩率，并结合重建与生成两种监督信号来学习紧凑且具有预测性的潜在表示。块因果掩码扩散解码器在干净、带噪以及缺失的潜在条件下进行训练，同时无分类器引导调节语义重建与自主生成之间的平衡。词元预测梯度通过潜在表示更新编码器，而流匹配先验学习则被解耦……

    arXiv:2610.31305v1 Announce Type: new  Abstract: Hierarchical language generation separates continuous semantic modeling from discrete text realization, but its effectiveness depends on what the latent space compresses and how the discrete generator uses it. We propose Cola2, a framework that jointly studies continuous semantic compression, the balance between reconstruction and generation, and continuous-discrete co-adaptation. Cola2 dynamically segments text into semantic spans, controls the compression rate through auxiliary-loss-free negative feedback, and combines reconstruction and generation supervision to learn compact, predictive latent representations. A block-causal masked diffusion decoder is trained with clean, noisy, and dropped latent conditions, while classifier-free guidance adjusts the balance between semantic reconstruction and autonomous generation. Token-prediction gradients update the encoder through the latents, whereas Flow Matching prior learning is decoupled f
    
[^29]: 语言智能体中记忆的价值

    The Value of Memory in Language Agents

    [https://arxiv.org/abs/2610.31303](https://arxiv.org/abs/2610.31303)

    该论文提出一个决策论框架，以所避免的任务损失为单位，量化语言智能体中记忆从原始历史到持久状态、操作使用及代价高昂的恢复与修正全过程中的价值，并通过成本感知的模拟条件和五个最小反例揭示信息时机、检索访问、组件交互、反馈和需求不确定性对记忆价值的关键影响。

    

    记忆使语言智能体能够将过去经验中有用的区分带入未来的决策中。其价值取决于哪些区分得以留存、有界的智能体能否使用这些区分，以及随着需求的演变哪些内容可以被恢复或修正。我们开发了一个决策论框架，追踪这一价值从原始历史到持久状态、操作化使用以及代价高昂的恢复与修正的完整过程。原始价值与保留后的价值以所避免的任务损失为单位来表达；风险-成本前沿刻画了在各种用途下可实现的恢复与保存能力。未知需求会产生价值画像，而交互式反馈则会改变未来工作的分布。一个成本感知的模拟条件与五个最小反例揭示了信息时机、检索访问、组件交互、反馈以及需求不确定性所扮演的角色。特别地，一个与答案具有零互信息的指针仍然可以降低精确恢复的成本。

    arXiv:2610.31303v1 Announce Type: cross  Abstract: Memory enables language agents to carry useful distinctions from past experience into future decisions. Its value depends on which distinctions survive, whether a bounded agent can use them, and what can be recovered or revised as demands evolve. We develop a decision-theoretic framework that follows this value from raw history through persistent state, operational use, and costly recovery and revision. Original and retained value are expressed in units of avoided task loss; risk-cost frontiers describe attainable recovery and preservation across uses. Unknown demand yields value profiles, while interactive feedback changes the distribution of future work. A cost-aware simulation condition and five minimal counterexamples expose the roles of information timing, retrieval access, component interaction, feedback and demand uncertainty. In particular, a pointer with zero mutual information about the answer can still reduce the cost of exa
    
[^30]: CoWriter：引导方法化与协同性的人机（人类-大语言模型）共同写作

    CoWriter: Guiding Methodological and Synergetic Human-LLM Co-Authoring

    [https://arxiv.org/abs/2610.31277](https://arxiv.org/abs/2610.31277)

    CoWriter是一种主动式人机协同写作智能体方案，通过借鉴认知科学中的专家写作方法论来引导端到端的完整写作流程并积极促使用户深度参与，从而显著提升作者的观点表达、文档主人翁感与文本质量。

    

    虽然大语言模型（LLM）已经改变了文本生成方式，但标准的聊天与画布式界面往往将人类写作降格为相当被动的监督工作，导致审阅疲劳、对AI的过度依赖，以及在长篇、基于文献来源的写作中人类角色的边缘化。以往的系统通常只针对孤立的写作阶段，而对全面的端到端人机协作则大多缺乏探索。本文提出了CoWriter，这是一种主动式的智能体方案，它将来自认知科学的专家级人类（共同）写作方法论加以适配改造，以灵活引导整个写作流程，同时积极鼓励用户参与。在一项探索性用户研究中，与标准LLM基线相比，使用CoWriter的作者能够更好地表达自己的观点，并保持更高的文档主人翁感，这在定量上得到了2.2倍更多成功智能体交互的支持。此外，外部读者评判者对所生成文本的评价更高。

    arXiv:2610.31277v1 Announce Type: new  Abstract: While LLMs have transformed text generation, standard chat-and-canvas interfaces often reduce human authoring to quite passive supervision, causing review fatigue, over-reliance on AI, and the sidelining of the human role in long-form, source-grounded writing. Prior systems typically address isolated writing phases, leaving comprehensive end-to-end human-LLM collaboration largely unexplored. In this paper, we introduce CoWriter, a proactive agentic scheme that adapts expert human (co-)authoring methodologies from cognitive science to flexibly steer the entire writing workflow while actively encouraging the user to engage. In an exploratory user study, compared to a standard LLM baseline, authors using CoWriter better expressed their perspectives and maintained a higher sense of document ownership, supported quantitatively by $2.2\times$ more successful agent engagements. Furthermore, external reader judges rated the resulting texts highe
    
[^31]: 记忆感知奖励建模：学习人类对大语言模型记忆使用的偏好

    Memory-Aware Reward Modeling: Learning Human Preferences for LLM Memory Use

    [https://arxiv.org/abs/2610.31272](https://arxiv.org/abs/2610.31272)

    该研究通过对87名用户约1000条真实偏好判断的分析，首次揭示了大语言模型使用记忆并不总能生成用户偏好的回复，并提出了一种记忆感知奖励模型，以72.8%的准确率预测用户对记忆使用的偏好，填补了记忆评估领域的空白。

    

    记忆使语言模型助手能够在会话之间保持连续性并生成个性化回复，但更多了解用户并不一定能生成用户更偏好的回复。现有评估方法仅评估记忆能力，而未直接确认记忆的使用是否真正让用户受益。为了研究这一空白，我们收集了来自87名参与者、基于他们自己的ChatGPT查询和个人记忆的大约1,000条成对偏好判断。我们发现，基于记忆条件生成的回复并不总是比不带记忆生成的回复更受用户偏好，某些情况下用户甚至强烈偏好后者。自动评估器与用户的一致性也很有限，且明显比用户更频繁地偏好基于记忆的回复。基于这些差距，我们提出了一种从用户判断中学习的记忆感知奖励模型，实现了72.8%的偏好预测准确率，并超越了现有基线方法。

    arXiv:2610.31272v1 Announce Type: new  Abstract: Memory enables language model assistants to maintain continuity across sessions and generate personalized responses, but knowing more about a user does not necessarily produce responses they prefer. Existing evaluations assess memory capabilities without directly establishing whether memory use benefits the user. To study this gap, we collect approximately 1,000 pairwise judgments from 87 participants using their own ChatGPT queries and personal memories. We find that memory-conditioned responses are not consistently preferred over responses generated without memory and in some cases, users strongly prefer the latter. Automatic evaluators also show limited agreement with users and favor memory-conditioned responses substantially more often than users do. Motivated by these gaps, we introduce a memory-aware reward model that learns from user judgments, achieving 72.8\% preference prediction accuracy and outperforming the existing baseline
    
[^32]: 时间同步端到端语音识别模型搜索空间复杂度与效率的系统性比较

    A Systematic Comparison of Search Space Complexity and Efficiency for Time-Synchronous End-to-End ASR Models

    [https://arxiv.org/abs/2610.31259](https://arxiv.org/abs/2610.31259)

    该论文系统比较了CTC和神经转换器等时间同步端到端语音识别模型在不同剪枝策略和语言模型配置下的搜索空间复杂度与效率，发现端到端模型多数时候搜索空间较小但会在不确定时出现假设数量峰值，而基于得分的动态剪枝可适应模型不确定性并显著降低实时率。

    

    搜索是自动语音识别（ASR）系统的核心组成部分，在基于隐马尔可夫模型（HMM）的经典ASR建模中已被深入研究。然而，在当前序列到序列ASR架构的研究中，搜索通常被简化为不带外部语言模型（LM）的小型固定大小束搜索。在这项工作中，我们对连接主义时间分类（CTC）和神经转换器模型在不同剪枝策略和语言模型配置下的搜索空间复杂度和效率进行了比较分析。我们表明，这些端到端（E2E）模型大部分时间在较小的搜索空间内工作，但在不确定性较高的时段会出现假设数量的大幅临时峰值。基于得分的剪枝能够根据模型不确定性动态调整搜索，从而显著降低实时率（RTF）。我们的结果凸显了低固定大小束剪枝搜索的局限性……

    arXiv:2610.31259v1 Announce Type: new  Abstract: Search is a central component of Automatic Speech Recognition (ASR) systems and has been deeply investigated for classical ASR modeling based on Hidden Markov Models (HMMs). However, in work on current sequence-to-sequence ASR architectures it is often simplified to a small fixed-size beam search without external Language Model (LM). In this work, we present a comparative analysis of search space complexity and efficiency for Connectionist Temporal Classification (CTC) and neural transducer models under different pruning strategies and LM configurations. We show that these end-to-end (E2E) models work with a small search space most of the time but exhibit large temporary peaks in hypothesis count during periods of uncertainty. Score-based pruning dynamically adapts the search to model uncertainty, leading to a significantly lower Real-Time Factor (RTF). Our results highlight the limitations of search with low fixed-size beam pruning and 
    
[^33]: IndoNutriQA：一个印度尼西亚临床营养问答基准

    IndoNutriQA: An Indonesian Clinical Nutrition Question Answering Benchmark

    [https://arxiv.org/abs/2610.31256](https://arxiv.org/abs/2610.31256)

    该论文提出了IndoNutriQA——首个由专家构建的印度尼西亚临床营养问答基准，包含基于真实临床营养报告的911个高质量问答对及结构化元数据，旨在填补低资源语言在临床问答评估资源上的空白。

    

    临床问答是临床智能决策支持系统的基础，但对于印度尼西亚语等低资源语言而言，由于缺乏公开的领域专用数据，其进展受到严重限制。我们提出了IndoNutriQA，这是一个由专家精心构建的印度尼西亚临床问答基准，包含911个基于真实临床营养报告构建的问答对。它并非大规模训练语料库，而是定位为高质量评估资源：每个问答对都以临床相关上下文为基础，并辅以结构化元数据（人体测量评估、医学诊断、文档来源、分块分割）。每份报告由研究生水平的营养学实习生撰写，并由三名营养师之一按照营养护理流程（NCP）回答八种标准化问题类型，其中包括“问题、病因、体征和症状”（PES）诊断陈述。答案为自由格式的临床专业文本。

    arXiv:2610.31256v1 Announce Type: new  Abstract: Clinical Question Answering (QA) underpins Clinical Intelligent Decision Support Systems, but progress is severely limited for low-resource languages such as Indonesian by the absence of public domain-specific data. We introduce IndoNutriQA, an expert-curated Indonesian clinical QA benchmark of 911 question-answer pairs built from real clinical-nutrition reports. Rather than a large-scale training corpus, it is positioned as a high-quality evaluation resource: every pair is grounded in a clinically relevant context, enriched with structured metadata (anthropometric assessment, medical diagnosis, document provenance, chunk segmentation). Eight standardised question types following the Nutrition Care Process (NCP), including the Problem, Etiology, Sign and Symptom (PES) diagnosis statement, are answered for every report, which was written by a graduate-level dietetic intern, by one of three nutritionists. Answers are free-form clinical pro
    
[^34]: 基于大型语言模型中文本与多模态上下文学习的音乐效价与唤醒度预测

    Predicting Music Valence and Arousal via Textual and Multimodal In-Context Learning in Large Language Models

    [https://arxiv.org/abs/2610.31202](https://arxiv.org/abs/2610.31202)

    该研究提出利用大型语言模型的上下文学习（包括基于文本的多模态方法），通过上下文示例改进小型基础模型的初始预测，显著提升了音乐情绪效价与唤醒度的连续预测性能，其中效价R²从0.110提升至0.427。

    

    虽然唤醒度可以从音乐音频中可靠地预测，但由于情绪效价具有主观性和情境依赖性，其建模仍然十分困难。在这项工作中，我们探索了利用大型语言模型的上下文学习（ICL）来推进连续效价与唤醒度预测的潜力。我们的框架并非直接预测情绪分数，而是利用来自辅助数据集的上下文示例来改进由小型任务专用基础模型生成的初始预测。我们系统地评估了两种变体：一种是基于文本的ICL变体，使用歌词和生成的音频描述；另一种是结合原始音频信号的多模态变体。我们将整个MER60数据集严格用作保留测试集，实验结果显示了显著的性能提升，尤其是对于效价预测。基于文本的ICL在使用80个示例的情况下将效价R²从0.110提升至0.427，而多模态变体……

    arXiv:2610.31202v1 Announce Type: cross  Abstract: While arousal can be reliably predicted from musical audio, emotional valence remains difficult to model due to its subjective and context-dependent nature. In this work, we investigate the potential of in-context learning (ICL) with large language models to advance continuous valence and arousal prediction. Rather than predicting emotion scores directly, our framework leverages contextual demonstrations from an auxiliary dataset to refine the initial predictions produced by a small, task-specific base model. We systematically evaluate a text-based ICL variant using lyrics and generated audio captions, alongside a multimodal variant that incorporates raw audio signals. Using the entire MER60 dataset strictly as a held-out test set, our experiments demonstrate substantial performance gains, particularly for valence. Text-based ICL improves the valence $R^2$ from 0.110 to 0.427 with 80 demonstration shots, while the multimodal variant ac
    
[^35]: 研究大语言模型中的多元文化价值观选择偏差

    Investigating Multi-Cultural Value Selection Biases in LLMs

    [https://arxiv.org/abs/2610.31199](https://arxiv.org/abs/2610.31199)

    该研究揭示了LLM在多重文化身份线索下存在价值观选择偏差：语言线索对答案的偏移是显式国家提示的三倍但反而偏离目标国家的价值观分布，而显式命名国家能缩小与该国价值观分布差距的30-44%并产生更明显的价值观变化。

    

    大语言模型（LLM）服务的用户具有多层次的文化身份，然而文化评估通常将身份简化为单一标签。在这项工作中，我们研究了LLM在同时存在多重身份线索时的表现，其中系统提示中包含一个国家角色设定，而用户轮次中通过两种方式传达第二种身份：直接命名一个国家（显式线索）或通过查询语言本身（隐式线索）。我们测试了世界价值观调查（WVS）多项选择探测和开放式社会政治内容生成两种方法。在多项选择探测中，语言线索对答案的偏移影响约为显式线索的三倍，但答案偏离了相关国家的价值观分布，而显式线索并不能持续地将答案推向所命名的国家的价值观。在160万条开放式回复中，命名一个国家可以缩小模型自身回复与该国价值观分布之间差距的30-44%，并且显式线索比语言线索产生更明显的价值观变化。我们的结果表明……

    arXiv:2610.31199v1 Announce Type: new  Abstract: Large language models (LLMs) serve users whose cultural identities are multi-layered, yet cultural evaluations commonly reduce identity to one label. In this work, we study LLMs under simultaneous identity cues, with a country persona in the system prompt and a second identity signaled in the user turn, either by naming a country (explicit cue) or through the query language (implicit cue). We test World Values Survey (WVS) multiple-choice probing and open-ended sociopolitical generation. In multiple-choice probing, language cues shift answers about three times more than explicit cues but away from the associated country's distribution, and explicit cues do not consistently move answers toward the named country. Across 1.6 million open-ended responses, naming a country closes 30-44% of the gap to the model's own responses for that country, and explicit cues produce more differentiated value changes than language cues. Our results suggest 
    
[^36]: 当更多元数据反而误导时：探查大语言模型生成的博物馆描述中的字段级接地性

    When More Metadata Misleads: Probing Field-Level Grounding in LLM-Generated Museum Descriptions

    [https://arxiv.org/abs/2610.31193](https://arxiv.org/abs/2610.31193)

    该研究通过两阶段实验发现，元数据数量本身并不能提升大语言模型生成博物馆文物描述的可靠性，真正决定模型是接地、省略还是凭空推断信息的是字段类型，而图像既能支持视觉接地也可能助长无依据的解读。

    

    元数据条件生成通常被视为一种数据到文本的任务，即更多的输入字段应当能改善生成内容的接地性（grounding）。然而博物馆描述生成对这一假设提出了挑战，因为馆藏目录记录具有异构性、部分依赖视觉信息，且在不同地理区域间的记录完备程度参差不齐。我们通过一个两阶段框架研究元数据的构成如何影响文物描述的生成。第一阶段分析了一个大型博物馆藏品中元数据的自然变化，其中字段数量、字段类型、图像可用性和地点等因素共同变化。第二阶段引入了受控的字段探针，在保持文物身份不变的前提下，改变元数据数量、字段类型和图像的访问情况。我们的结果表明，元数据数量本身并不能解释生成行为；字段类型才是决定模型对接地、省略还是推断信息的关键因素，而图像既能够支持视觉接地，也可能增加缺乏依据的解读。

    arXiv:2610.31193v1 Announce Type: new  Abstract: Metadata-conditioned generation is often framed as a data-to-text task in which additional input fields should improve grounding. Museum description generation challenges this assumption because catalogue records are heterogeneous, partially visual, and unevenly documented across geographic regions. We study how metadata composition shapes the generation of artefact descriptions through a two-phase framework. Phase 1 analyses natural metadata variation in a large museum collection, where field count, field type, image availability, and Place co-vary. Phase 2 introduces controlled field probes that hold object identity fixed while varying metadata quantity, field type, and image access. Our results show that metadata quantity alone does not explain generation behaviour. Field type determines whether models ground, omit, or infer information, while images can both support visual grounding and increase unsupported interpretation. More broad
    
[^37]: 提示策略效果取决于评判模型：LLM作为评判者鲁棒性的多语言研究

    Prompting Strategy Effects Depend on the Judge: A Multilingual Study of LLM-as-a-Judge Robustness

    [https://arxiv.org/abs/2610.31172](https://arxiv.org/abs/2610.31172)

    该研究通过涵盖9种提示策略、4个开源评判模型、27种语言的大规模基准测试（超过100万条评分），首次发现LLM-as-a-Judge评估中的提示策略效果不可跨模型和语言迁移——评判模型的选择比提示策略的选择更具决定性，且策略效果在不同评判模型间会反向。

    

    LLM-as-a-Judge（大语言模型作为评判者）评估正日益应用于多语言任务，然而诸如思维链和角色条件化等提示策略的推荐仅基于英语基准。这些策略的有效性能否跨评判模型和语言迁移仍是未知数：提示敏感性研究大多以英语为中心，而现有的多语言评判研究则固定了提示模板。在本工作中，我们通过一个基准测试来衡量这种交互作用，该基准涵盖9种提示策略、4个开源权重评判模型、27种语言以及跨越13个数据集的4个任务，在黄金参考输出上产生了超过100万个逐点评分，并经过人工判断的验证。我们发现提示策略是不可迁移的：评判模型的选择主导了提示策略的选择，策略效果在不同评判模型之间会发生方向反转，且在低资源语言中稳定性下降，而这些影响被掩盖……

    arXiv:2610.31172v1 Announce Type: new  Abstract: LLM-as-a-Judge evaluation is increasingly applied to multilingual tasks, yet prompting strategies such as Chain-of-Thought and Role Conditioning are recommended based solely on English benchmarks. Whether their effectiveness transfers across judge models and languages remains unknown: prompt-sensitivity studies are largely English-centric, while existing multilingual judge studies fix the prompt template. In this work, we address this gap by measuring this interaction through a benchmark of 9 prompting strategies, 4 open-weight judges, 27 languages, and 4 tasks across 13 datasets, yielding over 1M pointwise scores on gold-reference outputs, validated against human judgments. We find that prompting strategies are not transferable: the choice of the judge model dominates the choice of prompting strategy, strategy effects reverse direction across judges, and stability degrades in low-resource languages, where these effects are concealed by 
    
[^38]: 超越观看与搜索：面向以视频为中心的深度研究报告的智能体基准测试

    Beyond Watching and Searching: Benchmarking Agents for Video-centered Deep Research Reports

    [https://arxiv.org/abs/2610.31158](https://arxiv.org/abs/2610.31158)

    该论文提出以视频为中心的深度研究报告生成任务，并构建OmniDR-Bench基准与Omni-ART评估体系（涵盖视听定位、多模态证据使用和交互式呈现三个维度），用于系统评估智能体基于视频检索证据并生成交互式研究报告的能力。

    

    深度研究智能体能够将开放网络中的证据综合为长篇报告，但现有基准测试并未评估以源视频为中心的研究任务。我们提出了“以视频为中心的深度研究报告生成”任务：给定一个视频和用户请求，智能体需要识别研究问题、检索外部多模态证据，并生成与相关视频时刻相链接的交互式报告。该任务具有直接的实用价值，因为它可以帮助用户定位相关视频片段、验证论断、复现演示的操作流程，并基于视频内容做出明智决策。我们为该任务提出了一个基于技能的工作流程以及OmniDR-Bench基准。在评估方面，我们提出了全模态原子评分标准树：每棵树包含三个维度、23个共享方向以及任务特定的原子化二元评判标准，这些方向用于测试视听定位、多模态证据使用以及交互式呈现能力。审计式评分标准构建（ARC）用于构建这些评分树……

    arXiv:2610.31158v1 Announce Type: cross  Abstract: Deep research agents synthesize open-web evidence into long reports, but current benchmarks do not evaluate research centered on a source video. We introduce Video-Centered Deep Research Report Generation. Given a video and user request, an agent identifies research questions, retrieves external multimodal evidence, and produces an interactive report linked to relevant video moments. This task has direct practical value because it helps users locate relevant moments, verify claims, reproduce demonstrated procedures, and make informed decisions from video content. We present a skill-based workflow and OmniDR-Bench for this task. For evaluation, we propose Omni Atomic Rubric Trees (Omni-ARTs). Each tree contains three dimensions, 23 shared directions, and task-specific atomic binary criteria. The directions test audiovisual grounding, multimodal evidence use, and interactive presentation. Audited Rubric Construction (ARC) builds these tr
    
[^39]: 工具增强模型比较中的提示敏感性

    Prompt Sensitivity in Tool-Augmented Model Comparisons

    [https://arxiv.org/abs/2610.31148](https://arxiv.org/abs/2610.31148)

    论文提出一种将答案读取、推理、调用决策等环节分离的匹配协议，发现工具虽能提升模型准确率，却会加剧模型间比较对问题表述方式的敏感性。

    

    语言模型通常在问答基准上进行比较，其中每个问题只以一种表述方式呈现，而对问题进行改写就可能改变两个模型中谁得分更高。我们探究为模型提供工具是否能使这类比较更加稳定。仅凭准确率的提升无法回答这个问题，因为工具会同时改变从问题到被评分答案之间的多个环节。我们提出一种匹配协议来分离这些环节：答案读取、推理、调用决策、工具返回的答案，以及在不调用工具时提供的报告。我们将该协议应用于六个开源模型，在五个领域、五种表述方式下进行测试。我们发现，在程序能够回答问题的场景中，工具可以提升准确率，但同时也会加大许多成对模型比较对表述方式的敏感性。这种敏感性的扩大与模型调用工具的频率相关，而某些模型的调用频率会随表述方式变化，另一些则不会。敏感性还取决于接口与指令：原生工具调用接口以及……

    arXiv:2610.31148v1 Announce Type: new  Abstract: Language models are compared on question-answering benchmarks that pose each question in one wording, and a reworded question can change which of two models scores higher. We ask whether giving the models tools makes such comparisons more stable. An accuracy gain cannot answer this, because tools change several steps between the question and the scored answer at once. We propose a matched protocol that separates those steps: the answer readout, reasoning, the decision to call, the tool's answer, and the report supplied without a call. We apply it to six open models in five domains under five wordings. We find that tools raise accuracy where a program answers the question and widen the wording sensitivity of many pairwise comparisons. The widening follows how often a model calls, and call frequency follows the wording for some models and not others. It also depends on the interface and the instructions: a native tool-calling interface and
    
[^40]: 混合很重要：长上下文问答中不同序列混合架构的证据位置偏差研究

    Mixing Matters: Evidence Position Bias Across Sequence Mixers in Long-Context Question Answering

    [https://arxiv.org/abs/2610.31135](https://arxiv.org/abs/2610.31135)

    本研究通过在干扰文档中移动黄金证据，系统比较了Transformer、状态空间和混合Mamba架构在长上下文问答中的证据位置偏差，发现Transformer模型具有状态空间模型所缺乏的开头优势，而两类架构均偏好接近末尾的证据。

    

    长上下文模型在包含答案的证据出现在远离提示边界的位置时，往往无法有效利用这些证据。我们通过将一份黄金文档在固定的九个干扰文档集合中移动位置，比较了Transformer、状态空间以及混合Mamba架构之间的这种位置偏差。被评估的Transformer模型显示出开头优势，而被评估的状态空间模型缺乏这种优势，同时两个家族的模型都更偏好接近末尾的证据。一项匹配的纯Mamba与混合Mamba对比指向相同的方向，但在统计上仍不确定。这种家族差异体现在能够完成该任务的模型之中。Transformer的首要地位在合成检索任务中再次显现，而状态空间模型的性能天花板使得家族间的比较无法进行。在同一状态空间架构内更换语料库会改变总体准确率，但曲线形状没有可检测到的变化。提示的改变可以产生或几乎消除开头优势。位置信息在各架构中仍然是可解码的。

    arXiv:2610.31135v1 Announce Type: new  Abstract: Long-context models often use answer-bearing evidence less effectively when it appears away from a prompt boundary. We compare this position bias across Transformer, state-space, and hybrid Mamba architectures by moving one gold document through a fixed set of nine distractors. The evaluated Transformers show a beginning advantage that the evaluated state-space models lack, while both families favor evidence near the end. A matched pure-versus-hybrid Mamba comparison points in the same direction but remains statistically uncertain. The family difference appears among models that can answer the task. Transformer primacy reappears on synthetic retrieval, where state-space ceiling prevents a family comparison. A corpus swap within one state-space architecture changes overall accuracy without a detectable change in curve shape. Prompt changes can create or nearly remove a beginning advantage. Position remains decodable across architectures e
    
[^41]: Delta N-gram：向语言模型原生N-gram记忆的仅前向持续写入

    Delta N-gram: Forward-Only Continual Writing into a Language Model's Native N-gram Memory

    [https://arxiv.org/abs/2610.31125](https://arxiv.org/abs/2610.31125)

    提出Delta N-gram方法，通过冻结预训练N-gram表、将证据以仅前向传播的方式写入稀疏补丁Δ，实现无需反向传播的持续知识写入，清除补丁即可完全恢复原始模型，且事实保留效果远优于基于梯度的写入方法。

    

    我们之前的分析发现，模型原生的哈希N-gram表是持续更新对基础模型干扰最小的位置，但其写入需要反向传播，并且会改变预训练表。我们提出Delta N-gram，一种进入该表的仅前向写入路径。预训练表T^0永远不会被修改，证据被写入一个稀疏补丁Δ中其所寻址的行上：一个4.1M参数的写入器为每个（位置，哈希头）事件生成一个目标值和一个门控值，块平均的delta规则使每一行始终保持为其旧值与门控目标之间的凸组合。该写入器使用预训练目标在原始文本上离线训练；在使用时，写入仅需前向传播，清除Δ即可完全恢复原始模型。一个经过训练用于回答问题的0.6B参数原型模型在第二次写入后仍保留了16/16的事实；而在线梯度写入只能保留2-4/16。我们研究了两种写入器：一种是用仅基于T^0的模型编码证据的普通写入器，以及一种……（原文摘要到此截断）

    arXiv:2610.31125v1 Announce Type: new  Abstract: Our previous analysis found that a model's native hashed N-gram table is where continual updates disturb the base least, but its writes needed backpropagation and changed the pretrained table. We present Delta N-gram, a forward-only write path into that table. The pretrained table T^0 is never modified. Evidence is written into a sparse patch \Delta on the rows it addresses: a 4.1M-parameter writer emits a target and a gate per (position, hash head) event, and a block-averaged delta rule keeps each row a convex combination of its old value and the gated targets. The writer is trained offline on raw text with the pretraining objective; at use time writing takes forward passes only, and clearing \Delta restores the original model. A 0.6B prototype trained to answer questions keeps 16/16 facts after a second write; online gradient writes keep 2--4/16. We study two writers: a plain writer that encodes evidence with the T^0-only model, and a 
    
[^42]: 透过PRISM清晰洞察：程序性视频中错误识别的步骤序列推理

    Through a PRISM, Clearly: Step-Sequence Reasoning for Mistake Recognition in Procedural Videos

    [https://arxiv.org/abs/2610.31117](https://arxiv.org/abs/2610.31117)

    提出PRISM框架，通过将视频片段锚定于语言并利用在正确执行上训练的轻量级隐马尔可夫模型建模步骤序列，实现对程序性视频中步骤内容和顺序错误的可解释识别，并统一了跨数据集的四种基于编辑的错误类型评估标准。

    

    程序性错误识别要求识别出有效多步骤活动中的偏差。我们研究了在基准数据集提供步骤边界的条件下，从视频中进行活动完成后的错误识别。我们提出了PRISM（通过可解释语义建模进行程序性错误识别），该方法将视频片段锚定于语言，并利用一个在正确执行样本上训练的轻量级隐马尔可夫模型，将片段描述映射到程序状态。通过参考对齐和语义检查来识别步骤内容与顺序中的错误。为支持跨异构基准的评估，我们将各数据集特定的标注统一映射到四种基于编辑的错误类型：插入、删除、替换和换位。在三个基准测试上，我们比较了人工标注与视觉语言模型（VLM）生成的描述，涵盖原始、语义角色和动作-对象三种形式。解码器对照实验表明，训练得到的局部解码器相比最近候选匹配取得了一致的性能提升，此外还有数据集相关的……（摘要被截断）

    arXiv:2610.31117v1 Announce Type: cross  Abstract: Procedural mistake recognition requires identifying deviations from valid multi-step activities. We study post-completion recognition from videos with benchmark-provided step boundaries. We introduce PRISM (Procedural mistake Recognition via Interpretable Semantic Modeling), which grounds video segments in language and maps their descriptions to procedural states using a lightweight hidden Markov model trained on correct executions. Reference alignment and semantic checks identify errors in step content and order. To support evaluation across heterogeneous benchmarks, we map dataset-specific annotations to four edit-based error types: insertion, deletion, substitution, and transposition. Across three benchmarks, we compare gold and VLM-generated descriptions in raw, semantic-role, and action-object forms. Decoder controls show consistent gains from the trained local decoder over nearest-candidate matching, with additional, dataset-depe
    
[^43]: 法国民事撤销审的“语法”：一个基础层面的数据集与工具

    A Grammar of French Civil Cassation: A Ground-Level Dataset and Tool

    [https://arxiv.org/abs/2610.31116](https://arxiv.org/abs/2610.31116)

    该研究通过确定性流程构建了法国最高法院民事庭86,464份判决中121,536对“上诉理由-法院回应”的配对数据集，并将其归入既定教义学类别，形成了一部揭示民事撤销审上诉成败规律及其十年演变的描述性“语法”。

    

    本文发布了一个数据集，涵盖法国最高法院五个民事庭（2016-2025年）86,464份判决中的121,536对“上诉理由-法院回应”配对。针对法律问题的撤销审上诉本身不可再上诉，且往往一锤定音地决定案件结局，因此理解该法院如何裁判至关重要，并能帮助起草此类上诉的律师。在一份判决中，每个上诉理由（针对被上诉裁决的一项独立法律异议）都对应法院的一个回应，即“理由”，它属于少数几个教义学类别之一。我们利用一个确定性流程，将每个上诉理由与其对应的法院回应配对，并依据法国既定的分类法将配对归入相应类别。由此，我们获得了一部民事撤销审的描述性“语法”，展示哪些上诉理由能够胜诉、援引了哪些法律条文，以及二者在十年间的演变。我们公开了数据集、代码和交互式查询工具。

    arXiv:2610.31116v1 Announce Type: new  Abstract: This paper presents a dataset of 121,536 ground-reasons pairs across 86,464 rulings of the five civil chambers of the French Court of Cassation (2016-2025). An appeal on points of law cannot itself be appealed and often decides a case for good. Understanding how the Court rules is thus crucial and can help lawyers who draft those appeals. Inside a ruling, every ground of appeal (a distinct legal complaint against the appealed decision) has its own answer from the Court. This answer is called the reasons and it belongs to one of a few doctrinal families. Using a deterministic pipeline, we pair each ground with its reasons and attach the pair to its family, according to the established French taxonomy. We obtain a descriptive grammar of civil cassation that shows which grounds succeed, which legal texts they refer to, and how both evolve over the decade. We release the dataset, the code and an interactive consultation tool.
    
[^44]: 为明天学习该问什么：面向长期个性化网络代理的主动偏好获取

    Learning What to Ask for Tomorrow: Active Preference Acquisition for Long-Term Personalized Web Agents

    [https://arxiv.org/abs/2610.31107](https://arxiv.org/abs/2610.31107)

    提出主动偏好获取（APA）框架，通过前瞻性个性化价值（PPV）衡量缺失偏好在未来任务中的累积效用增益并扣除提问成本与披露风险，使网络代理能在交互预算下智能选择该向用户询问哪些偏好以实现长期个性化。

    

    网络代理越来越多地在重复的在线任务中服务于同一用户，例如预订旅行、购物和日程安排，因此它们的长期有用性取决于其对用户偏好的了解程度。被动记忆和个性化方法无法获取用户从未透露的偏好，而以问题对当前任务的收益来衡量价值的澄清式提问方法，则忽视了在后续任务中重复使用答案所带来的收益。我们研究了主动偏好获取（Active Preference Acquisition, APA）：哪些缺失的偏好值得付出提问的成本？我们将缺失偏好的前瞻性个性化价值（Prospective Personalization Value, PPV）定义为其在未来任务上预期的累积效用增益减去提问的努力成本和信息披露风险成本，并在每次交互的预算约束下选择要提问的问题。我们刻画了尽管单任务收益不足、但累积重复使用仍能使偏好获取变得合理的状态，并通过初始画像诊断评估了这一情形。

    arXiv:2610.31107v1 Announce Type: new  Abstract: Web agents increasingly serve the same user across repeated online tasks such as booking travel, shopping, and scheduling, so their long-term usefulness depends on how well they know the user's preferences. Passive memory and personalization methods cannot obtain preferences the user never reveals, and clarification methods that value a question by its benefit to the current task overlook the benefit of reusing the answer in later tasks. We study Active Preference Acquisition (APA): which missing preferences are worth the cost of asking? We define the Prospective Personalization Value (PPV) of a missing preference as its expected cumulative utility gain over future tasks minus the effort and disclosure-risk cost of asking, and select questions under a per-interaction budget. We characterize when cumulative reuse justifies acquisition despite insufficient single-task benefit, and evaluate this regime with an initial-profile diagnostic. AP
    
[^45]: RIFT：面向草稿模型的KV缓存压缩方法与互补可压缩性训练

    RIFT: KV Cache Compression for Draft Models with Complementary Compressibility Training

    [https://arxiv.org/abs/2610.31098](https://arxiv.org/abs/2610.31098)

    RIFT提出了一类基于选择的KV缓存压缩策略，通过在token生成过程中动态刷新KV缓存选择来应对“解码陈旧性”问题，并借助自适应刷新策略在保持草稿模型质量的同时显著降低刷新开销。

    

    投机解码是一种流行的解码加速方法，因此它对于需要越来越大的KV缓存的长输出生成尤为重要。KV缓存压缩提供了一种约束长上下文草稿生成成本的手段。然而，专注于预填充信息的压缩方法会随着输出长度的增长而逐渐失去相关性。我们将这种现象称为“解码陈旧性”，并证明该现象在基于选择的KV缓存压缩下会降低草稿模型的准确率。我们提出了RIFT，这是一系列基于选择的KV缓存压缩策略，能够平衡预填充信息与最近生成的token。我们证明在token生成过程中刷新KV缓存的选择可以显著提升草稿模型的质量，但代价是额外的解码延迟。为了缓解刷新带来的开销，我们提出了一种自适应KV刷新策略，在保持草稿模型质量的同时减少了评分计算的次数。

    arXiv:2610.31098v1 Announce Type: new  Abstract: Speculative decoding is a popular decode acceleration method. As such, it is particularly relevant to long-output generation, which requires an increasingly large KV cache. KV cache compression offers a way to constrain costs in long-context drafting. However, compression methods that focus on prefill information can lose relevance as the output length grows. We term this phenomenon decode staleness and show that it degrades draft model accuracy under selection-based KV cache compression. We introduce RIFT, a family of selection-based KV cache compression strategies that balance prefill information and recent generated tokens. We show that refreshing the KV cache selection during token generation significantly improves the drafter's quality at the cost of additional decode latency. To alleviate the refresh costs, we propose an adaptive KV refresh strategy that reduces the number of score calculations while retaining the drafter's quality
    
[^46]: 成对评估所遗漏的：跨提示的奖励分数

    What Pairwise Evaluation Misses: Reward Scores across Prompts

    [https://arxiv.org/abs/2610.31090](https://arxiv.org/abs/2610.31090)

    论文提出一种以人类评分为参照的审计方法，揭示奖励模型分数存在按提示聚集的跨提示不可比成分——这是成对评估无法检测到的，六个奖励模型的提示间方差份额高达0.34-0.64，而人类评分接近于零。

    

    奖励模型为偏好优化、过滤和排序对响应进行评分，而跨提示汇总分数要求这些分数具有可比性。然而，成对评估比较的是同一提示下的两个响应，因此某个提示的所有响应所共享的任何分数成分都会被抵消，从而无法被检验。我们通过比较模型分数在提示之间的变异与人类对相同响应评分的相应变异来研究这一成分。我们提出了一种以人类为参照的审计方法，该方法由提示间方差份额、同胞评分回归和提示替换构成，并将其应用于十八个已发布的评分器和六个匹配的训练条件。我们发现分数按提示聚集的程度高于人类评分：在 HelpSteer2 上，六个奖励模型的提示间份额为 0.34 至 0.64，而人类评分的份额接近于零。用固定指令替换原始提示后，大量的聚集性依然存在，但与提示级人类评分的一致性被削弱。

    arXiv:2610.31090v1 Announce Type: new  Abstract: Reward models score responses for preference optimization, filtering and ranking, and pooling scores across prompts needs them to be comparable. Yet pairwise evaluation compares two responses to the same prompt, so any score component shared by a prompt's responses cancels and goes untested. We study this component by comparing between-prompt variation in model scores with the same variation in human ratings of the same responses. We propose a human-referenced audit, built from a between-prompt variance share, a sibling-rating regression and prompt substitutions, and apply it to eighteen released scorers and six matched training conditions. We find that scores cluster by prompt more than the ratings do: on HelpSteer2, six reward models have between-prompt shares of 0.34 to 0.64 against a human share near zero. Replacing the original prompt with a fixed instruction keeps substantial clustering but weakens agreement with prompt-level human
    
[^47]: SCOPE：将自然语言策略编译为神经符号谓词程序

    SCOPE: Compiling Natural-Language Policies into Neurosymbolic Predicate Programs

    [https://arxiv.org/abs/2610.31062](https://arxiv.org/abs/2610.31062)

    SCOPE将自然语言策略编译为由小模型神经谓词、确定性提取器和组合代码构成的可执行神经符号程序，并通过程序与训练数据的联合优化，实现对LLM智能体多轮交互中复合策略的高效且准确的执行。

    

    针对大语言模型（LLM）智能体的策略执行已经从单轮内容审核，转向在多轮对话记录和工具调用轨迹上进行评估的复合型、应用特定策略。微调的护栏模型虽然高效，但在复合策略上性能会下降；而“LLM作为裁判”的方法虽然能达到更高的准确率，但推理成本高昂且预测结果不稳定。神经符号方法提供了一种天然的替代方案，将策略分解为在显式逻辑下组合的窄域神经决策。然而，现有的符号层缺乏所需的表达能力，且是针对固定数据而非策略本身进行优化的。为解决这一问题，我们提出了SCOPE，它将策略编译为可执行的神经谓词程序，由小模型执行神经谓词、确定性提取器以及组合其输出的代码构成。该程序与其训练数据进行联合优化，使用对抗性生成……（摘要在此截断）

    arXiv:2610.31062v1 Announce Type: cross  Abstract: Policy enforcement for LLM agents has shifted from single-turn content moderation to compound, application-specific policies evaluated over multi-turn transcripts and tool-call trajectories. Fine-tuned guardrail models are efficient but degrade on compound policies, and while LLM-as-a-judge approaches achieve higher accuracy, they incur high inference cost and tend to produce unstable predictions. Neurosymbolic methods offer a natural alternative, decomposing the policy into narrow neural decisions composed under explicit logic. However, existing symbolic layers lack the required expressiveness and are optimized against fixed data rather than the policy itself. To address this, we introduce SCOPE, which compiles a policy into an executable program of neural predicates executed by small models, deterministic extractors, and code that composes their outputs. The program is optimized jointly with its training data, using adversarial gener
    
[^48]: 伤害是行为性的，审核是词汇性的：语言与文化如何决定自动化安全系统保护谁、又噤声谁

    Harm Is Behavioural, Moderation Is Lexical: How Language and Culture Decide Whom Automated Safety Systems Protect and Whom They Silence

    [https://arxiv.org/abs/2610.31047](https://arxiv.org/abs/2610.31047)

    本文论证自动化内容审核系统因依据词汇和表面形式而非信息行为来判断危害，导致对非裔美国人英语等少数群体过度标记、对低资源语言保护不足，进而提出有害行为比表达词语更易跨语言传播的假设，并归纳七种失败模式与相应评估协议。

    

    内容警告：本文引用了已发布测试套件中的侮辱性词语和仇恨言论；其中的种族歧视用语已被遮蔽。自动化系统在大规模判定哪些信息有害，而其判定并不均衡：分类器将非裔美国人英语标记为冒犯性内容的频率高达其他英语变体的两倍，知名变装表演者的账号可能被评为比白人至上主义者的账号更具毒性，而在英语中有效的安全行为在低资源语言中则会失效。我们认为，过度标记与保护不足共享同一个根源：依据词汇和表面形式而非信息实际实施的行为来判断危害。我们将证据归纳为七种失败模式，并提出可检验的假设：有害行为（如要求保密、索取付款、试图孤立某人）比表达它们的词语更容易跨越语言传播。我们提出了一种评估协议，以揭示……（摘要原文在此处截断）

    arXiv:2610.31047v1 Announce Type: new  Abstract: Content warning: this paper quotes slurs and hateful statements from published test suites; the racial slur among them is masked.   Automated systems decide at scale which messages are harmful, and they decide unevenly. Classifiers label African American English as offensive up to twice as often as other English, the accounts of prominent drag queens can score as more toxic than those of white nationalists, and safety behaviour that holds in English collapses in low-resource languages. We argue that over-flagging and under-protection share one cause: deciding harm from vocabulary and surface form rather than from what a message does. We organise the evidence into seven failure modes and argue, as testable hypotheses, that harmful behaviours (a request for secrecy, a demand for payment, an attempt to isolate someone) travel across languages better than the words that express them. We propose an evaluation protocol that makes the gap betwe
    
[^49]: 真实文档修订下的分块重计算：测量固定分块、递归分块与内容定义分块的更新成本与检索质量

    Chunk Recomputation under Real Document Revisions: Measuring Update Cost and Retrieval Quality of Fixed, Recursive, and Content-Defined Chunking

    [https://arxiv.org/abs/2610.31019](https://arxiv.org/abs/2610.31019)

    该论文首次在真实文档持续修订的场景下量化比较了固定大小、递归和内容定义三种分块策略的更新成本与检索质量，发现递归分块的重计算开销最低（38-56%），而三者在检索召回率上的差异很小。

    

    面向检索增强生成（RAG）的分块策略几乎总是在静态快照上进行评估，然而知识库底层的文档却在持续变化。由于分块是独立进行向量嵌入的，编辑后边界的移动会迫使未发生变化的文本被重新嵌入。我们在120对相邻的中文和英文维基百科修订版本上，在严格的字节级相同复用规则和冻结的稠密编码器条件下，测量了三种无状态分块器（无重叠固定大小分块、递归的段落优先分块，以及内容定义分块）的更新成本。在由116篇文章组成的共同成功集合上，使用冻结的、按长度校准的参数，固定大小分块需要重计算其完整新版嵌入输入工作负载的57-68%，递归分块为38-56%，内容定义分块为43-54%（后者在62篇中文文章中有4篇处理失败），而在这些参数下，在固定的2048词元检索预算内，三者的平均证据召回率差异小于（原文摘要在此处截断）

    arXiv:2610.31019v1 Announce Type: new  Abstract: Chunking strategies for retrieval-augmented generation are almost always evaluated on a static snapshot, yet the documents underlying a knowledge base continue to change. Because chunks are embedded independently, a boundary that moves after an edit forces unchanged text to be re-embedded. We measure this update cost on 120 pairs of adjacent Chinese and English Wikipedia revisions for three stateless chunkers (non-overlapping fixed-size, recursive paragraph-first, and content-defined chunking) under a strict byte-identical reuse rule and a frozen dense encoder. On the 116-article common-success pool, with frozen, length-calibrated parameters, fixed-size chunking recomputes 57-68% of its full new-version embedding-input workload, recursive chunking 38-56%, and content-defined chunking 43-54% (it failed on 4 of 62 Chinese articles), while at these parameters mean evidence recall under a fixed 2048-token retrieval budget differs by less tha
    
[^50]: 主要是阅读：开卷评分在文档级LoRA适配器中究竟衡量了什么

    Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters

    [https://arxiv.org/abs/2610.31017](https://arxiv.org/abs/2610.31017)

    开卷评分主要衡量的是所有适配器共有的“阅读”行为而非文档记忆——随机适配器即可恢复专用适配器82%的开卷优势，但在无源文本的完形测试中仅恢复约2%（与零无异），因此不宜用开卷评分来评估或选择文档级LoRA适配器。

    

    单文档LoRA适配器已被提出作为可合并与检索的知识单元；然而，开卷评分——即将文档放入提示词中——会同样容易地把功劳归给文档自己的适配器（其专用适配器）的“阅读”行为，而非真正“记住”该文档。我们在Llama-3.1-8B-Instruct上为150篇Wikipedia文章各训练一个适配器，发现开卷分数主要衡量的是每个适配器都共有的一种阅读行为：一个在其他文档上训练的随机适配器能够恢复专用适配器相对于基础模型82%的开卷优势，但在一个不提供源文本、直接询问同一文档事实的完形填空探测中，只能恢复其2%的优势——这一比例与零无法区分。文档内容确实写入了权重，因为专用适配器在完形回忆上比基础模型高出0.094，且其更新的几何结构遵循其所阅读的文档；然而，在一项包含30个查询的测试中，通过权重空间邻近性选择的适配器……（原文在此处截断）

    arXiv:2610.31017v1 Announce Type: new  Abstract: Single-document LoRA adapters have been proposed as units of knowledge to be merged and retrieved; yet open-book scoring, with the document in the prompt, can credit a document's own adapter (its specialist) for reading that document as readily as for holding it. Training one adapter for each of 150 Wikipedia articles on Llama-3.1-8B-Instruct, we find that open-book scores mostly measure a reading behaviour every adapter shares: a random adapter trained on another document recovers 82 % of the specialist's open-book advantage over the base model but only 2 % of its advantage on a cloze probe that asks for the same document's facts without providing the source text, a share we cannot distinguish from zero. Content does reach the weights, since the specialists beat the base model in cloze recall by 0.094, and the geometry of their updates follows the documents they read; yet, in a test of 30 queries, adapters chosen by weight-space proximi
    
[^51]: 凑数到k：模式中的最小数量要求使开放权重模型在抽取列表中填充条目

    Padding to k: A Minimum Count in the Schema Makes Open-Weight Models Pad Their Extraction Lists

    [https://arxiv.org/abs/2610.30997](https://arxiv.org/abs/2610.30997)

    开放权重语言模型在面对JSON模式中设定的列表最小条目数量时会“凑数填充”——用文档中的真实文本补充列表以达到数量要求，这种填充因内容真实而难以被检测，且常见的提示词修复措施只能部分缓解问题。

    

    arXiv:2610.30997v1 公告类型：新论文 摘要：抽取流程要求语言模型输出符合模式的JSON，而模式可以设定列表条目的最小数量。我们探究当文档中实际条目少于该最小值时模型会怎么做。在一项预注册的研究计划下，我们在来自九个标注数据集的558份真实文档上测试了八个开放权重模型，并额外测试了六个托管服务上的模型作为参照。三个发现尤为突出。首先，模型会将该数量视为目标：一个仅在模式中显示但未强制执行的最小值，会使列表长度超过真实条目数的比例从34.5%上升到55.4%。其次，这种填充很难被发现，因为大多数添加的条目都是取自文档的真实文本；其中约一半能通过“每个条目均出现在源文档中”的检查。第三，常见的修复方法只能部分解决问题：一行说明“条目可以更少”的文字在平均意义上消除了这一效应，但会漏掉更多真实条目；用文字表述数量（“最多k个”）也会加剧多计；"N/A"选项……（原文摘要在此处被截断）

    arXiv:2610.30997v1 Announce Type: new  Abstract: Extraction pipelines ask language models for JSON that follows a schema, and the schema can set a minimum number of list entries. We ask what models do when the document holds fewer items than that minimum. Under a preregistered plan, we test eight open-weight models on 558 real documents from nine annotated datasets, with six more models on hosted services as a reference. Three findings stand out. First, models treat the count as a target: a minimum shown in the schema, but not enforced, raises the share of lists longer than the true count from 34.5% to 55.4%. Second, the padding is hard to catch, because most added entries are real text taken from the document; about half of them pass a check that each entry occurs in the source. Third, common fixes help only partly: a line saying fewer items are fine removes the effect on average but misses more true items, a count in prose ("up to k") also raises over-counting, an "N/A" option under 
    
[^52]: ColaBench：评估个人智能体中记忆引导的知识获取

    ColaBench: Evaluating Memory-Guided Knowledge Acquisition in Personal Agents

    [https://arxiv.org/abs/2610.30989](https://arxiv.org/abs/2610.30989)

    论文提出ColaBench基准，通过证据图谱联合评估个人智能体利用长期记忆识别所需知识并将其应用于用户具体情境的能力，涵盖10个用户画像下的349个任务和13,033篇文档的知识库。

    

    个人智能体需要利用长期交互历史来检索并应用适合用户当前请求的外部知识。即使答案基于相关文档，一旦忽视用户的具体情况，也可能不合适。这带来了两个相互关联的挑战：利用历史条件识别新请求所需的知识，以及将该知识应用于用户的实际情境。现有基准在记忆召回、个性化检索和记忆引导行动方面已有进展，但对这两个挑战的联合评估仍待探索。我们提出了ColaBench，一个面向健康与日常生活的合成基准，包含覆盖10个用户画像的349个任务以及一个拥有13,033篇文档的知识库。其证据图谱将个人状况与公开交互会话中的信息披露关联起来，记录用户此前接触过的文档，并将答案要求与个人文档及新获取的文档证据相链接。

    arXiv:2610.30989v1 Announce Type: cross  Abstract: Personal agents need to draw on long-term interaction history to retrieve and apply external knowledge suited to a user's current request. Even an answer grounded in relevant documents can be unsuitable when it overlooks the user's circumstances. This creates two connected challenges: using historical conditions to identify the knowledge needed for a new request, and applying that knowledge to the user's situation. Existing benchmarks advance memory recall, personalized retrieval, and memory-guided action, while the joint evaluation of these two challenges remains underexplored. We introduce ColaBench, a synthetic benchmark in health and daily living, with 349 tasks across 10 Profiles and a knowledge base of 13,033 documents. Its Evidence Graph connects personal conditions to their disclosures across public interaction sessions, records prior document exposure, and links answer requirements to personal and newly acquired document evide
    
[^53]: 争议性政治话语中省略三段论检测的解释差异与话语结构

    Interpretive Variation and Discourse Structure in Detecting Enthymemes in Controversial Political Discourse

    [https://arxiv.org/abs/2610.30987](https://arxiv.org/abs/2610.30987)

    该研究发现在争议性政治话语中检测省略三段论时，由于标注者解释存在固有分歧，所有微调模型的F1均停滞在0.67左右，为此提出了基于标注者投票分布的新评估指标和直接预测该分布的模型，并揭示模型依赖显式论证结构进行识别。

    

    省略三段论是指省略某个前提或结论的论证，它是社交媒体上常见的说服手段。由于读者必须自行补全缺失的部分，人们对某段文本是否包含省略三段论常常意见不一。本文探讨当这种分歧本身就是任务的一部分时，如何检测和评估省略三段论。在关于疫苗接种和移民的推文数据集上（每条推文由五位标注者标注），我们训练的所有微调模型的F1分数都停滞在0.67左右，无论数据规模、新增特征或标签类型如何变化。因此，我们转而探讨应如何评估这些模型，提出了将预测结果与标注者投票的完整分布进行比较的评估指标，并构建了一个直接预测该分布的模型，该模型能更好地捕捉标注者分歧，且不损失F1分数。随后，我们研究了话语结构所起的作用：模型学会依赖显式的论证结构，能够可靠地识别不含省略三段论的推文……（摘要在此处截断）

    arXiv:2610.30987v1 Announce Type: new  Abstract: Enthymemes are arguments that leave a premise or conclusion unstated, and they are a common persuasion device on social media. Because readers must fill in the missing part themselves, they often disagree on whether a text contains one. This paper asks how to detect and evaluate enthymemes when such disagreement is part of the task. On tweets about vaccination and immigration, each labeled by five annotators, F1 stalls around 0.67 for every fine-tuned model we train, regardless of data size, added features or label type. We therefore ask how these models should be evaluated instead, and propose metrics that compare predictions with the full distribution of annotator votes, together with a model that predicts this distribution directly and captures disagreement better at no cost in F1. We then ask what role discourse structure plays. Models learn to rely on explicit argumentative structure: they reliably recognize tweets without an enthym
    
[^54]: 推理时上下文增强方法的现实检验

    A Reality Check of Inference-Time Context Augmentation Methods

    [https://arxiv.org/abs/2610.30983](https://arxiv.org/abs/2610.30983)

    该论文提出了一个涵盖五种上下文增强方法、七个不同规模模型和三个数据集的统一评估框架，并发现推理时上下文增强带来的准确率提升并非总是真正的改进，而是无方向的扰动——这些方法在纠正部分预测的同时也会破坏其他预测。

    

    大型语言模型（LLM）常常无法利用输入中给出的上下文，从而导致幻觉或退回到参数化知识。上下文增强方法旨在通过修改输入来提升模型性能，从插入会增加计算时间的填充词元，到在推理过程中添加相关内容（如对上下文的详细阐述）。然而，这些方法此前都是在孤立场景下评估的，且仅依据其性能提升来衡量，这使得它们之间的比较尚不清晰，也无法确定它们是否能可靠地提升性能。在本文中，我们提出了一个评估框架，涵盖五种在所插入内容信息量上各不相同的方法，并将它们在句子层面应用于三个数据集上的七个不同规模的模型。我们发现，准确率的提升并不总是真正的改进，而是一种无方向的扰动：这些方法在纠正某些预测的同时也破坏了其他预测，尤其当插入的内容……

    arXiv:2610.30983v1 Announce Type: new  Abstract: Large language models (LLMs) often fail to utilise the context given in the input, resulting in hallucinations or a fallback to parametric knowledge. Context augmentation methods aim to improve model performance by modifying the input, from inserting filler tokens that add computational time to adding relevant content such as elaboration of a context during inference. However, these methods have been evaluated in isolation and only by their performance gains, leaving it unclear how they compare and whether they reliably improve performance. In this paper, we present an evaluation framework spanning five methods that differ in how informative the inserted content is and apply them at the sentence level across seven models spanning multiple sizes on three datasets. We find that accuracy gains are not always genuine improvements but undirected perturbations: the methods correct some predictions while breaking others, particularly when the i
    
[^55]: 联合还是逐选项？面向闭集决策的紧凑编码器：参数规模从1700万到10亿

    Joint or Per-Option? Compact Encoders for Closed-Set Decisions from 17 Million to 1 Billion Parameters

    [https://arxiv.org/abs/2610.30970](https://arxiv.org/abs/2610.30970)

    在闭集决策任务上，采用重排序器初始化的逐选项评分小型编码器（1700万至10亿参数）在精度上超越了联合评分和大语言模型提示，而延迟至多只增加一倍。

    

    许多语言模型流水线最终都要做一个闭集决策，例如选择某个工具。为此对大型语言模型进行提示既缓慢又对顺序敏感。我们探讨用于此类决策的编码器可以小到什么程度，以及它应该联合读取所有选项还是分别读取每个选项。在一项3200万参数、三个随机种子的研究中，在匹配优化器步数的条件下，逐选项评分在包含37个基准的决策套件的保留样本上，比联合评分高出5.8个点的机会校正技能分数；在近似匹配训练计算量的条件下高出2.8个点；两个外部基准的结果方向一致，但在其未见任务上差异不显著。将读取方式与初始化方式进行交叉实验表明，这一优势依赖于重排序器初始化，因为其查询-候选对输入方式与逐选项读取相匹配；而若从原始编码器出发，两种读取方式则难以区分。逐选项评分需要5到13倍的编码器前向传递次数：中位延迟至多增加一倍，但……

    arXiv:2610.30970v1 Announce Type: new  Abstract: Many language-model pipelines end in a closed-set decision, such as selecting a tool. Prompting a large language model for it is slow and order-sensitive. We ask how small an encoder for such decisions can be and whether it should read the options jointly or separately. In a three-seed study at 32 million parameters, per-option scoring beats joint scoring on a held-out sample of a 37-benchmark decision suite by 5.8 points of chance-corrected skill at matched optimizer steps and by 2.8 at approximately matched training compute; two external benchmarks agree, though not significantly on their unseen tasks. Crossing reader with initialisation shows that the advantage requires a reranker initialisation, whose query-candidate pair input matches per-option reading; from the raw encoder the two readers are indistinguishable. Per-option scoring needs 5 to 13 times more encoder passes: at most twice the median latency, but more than four times th
    
[^56]: 一日为树，终身为树？语言模型中层级泛化的跨语言迁移

    Once a Tree, Always a Tree? Cross-lingual Transfer of Hierarchical Generalization in Language Models

    [https://arxiv.org/abs/2610.30960](https://arxiv.org/abs/2610.30960)

    本文提出跨语言刺激贫乏（CPOS）实验设置，证明语言模型在第二语言中的层级结构泛化可以由第一语言提供的结构证据驱动，从而支持层级结构作为语言普遍原则在跨语言迁移中被模型利用的观点。

    

    语言模型（LM）通过持续（预）训练往往能表现出成功的跨语言迁移，但其背后的机制仍不清楚。一种可能性是，语言模型能够利用源语言（L1）中指向语言普遍原则的证据——即层级结构——并将其应用于新语言（L2），但也可能存在许多其他非语言学层面的解释，例如位置嵌入初始化的改进等。在本研究中，为了量化跨语言训练中语言证据的跨语言可迁移性，我们引入了一种新的实验设计，即跨语言刺激贫乏（CPOS）设置，用以消融检验语言模型在第二语言中的层级泛化如何能够由第一语言所提供的结构证据来引导。我们对比了反事实的L1与L2学习场景：在这些场景中，L2语料库在线性语法规则和层级语法规则之间是模糊的，而正面证据的数量……（摘要在此处截断）

    arXiv:2610.30960v1 Announce Type: new  Abstract: Language models (LMs) often exhibit successful cross-lingual transfer through continued (pre)training, still their underlying process remains unclear. One possibility is that LMs can leverage the evidence toward language-universal principles, namely, hierarchical structure, from a source language (L1) to a new language (L2), while many other non-linguistic explanations, e.g., improved initialization of position embeddings, can be potentially possible. In this study, to quantify the cross-linguistic transferability of linguistic evidence in cross-lingual LM training, we introduce a new experimental design, namely, a cross-lingual poverty-of-stimulus (CPOS) setting that ablates how LMs' hierarchical generalization in L2 can be led by structural evidence provided in L1. We contrast counterfactual L1 and L2 learning scenarios where the L2 corpus is ambiguous between linear and hierarchical grammar rules, while the amount of positive evidence
    
[^57]: IQuest-Q1：推进智能体CLI系统的基础能力

    IQuest-Q1: Advancing Foundation Capabilities for Agentic CLI Systems

    [https://arxiv.org/abs/2610.30949](https://arxiv.org/abs/2610.30949)

    IQuest-Q1是一个总参数320B（激活15B）的稀疏MoE智能体基础模型，通过稳定预训练、细粒度中期训练和多脚手架多版本后训练的三阶段流程，显著提升了智能体CLI系统的基础与编码能力。

    

    我们提出了IQuest-Q1，一个总参数量为320B（激活参数15B）的稀疏混合专家（MoE）智能体基础模型。IQuest-Q1通过三阶段流程开发：（1）稳定高效的预训练，结合Stable SplitNorm、Muon和muP，并采用两阶段课程学习，其第二阶段提高了代码数据的配比，在安全学习率与最优学习率之间保持了较宽的稳定性余量；（2）细粒度的中期训练，扩展跨领域、跨格式的数据配比，并逐步延长上下文长度；（3）多脚手架多版本后训练，在多种脚手架下结合SFT、领域专用强化学习和MOPD，以整合互补的训练信号并强化智能体编码能力。此外，模型合并将来自不同阶段的多个专家和检查点结合起来，同时缓解其参数更新之间的冲突。由此得到的IQuest-Q1展现出……（原文摘要在此处截断）

    arXiv:2610.30949v1 Announce Type: new  Abstract: We present IQuest-Q1, a sparse Mixture-of-Experts (MoE) agentic foundation model with 320B total parameters (15B activated). IQuest-Q1 is developed through a three-stage process: (1) stable and efficient pre-training, combining Stable SplitNorm, Muon, and muP with a two-stage curriculum whose second stage increases the coding-data mixture, preserving a wide stability margin between safe and optimal learning rates; (2) fine-grained mid-training, expanding the data mixture across domains and formats and progressively extending the context; and (3) multi-scaffold with multi-version post-training, combining SFT, domain-specialized RL, and MOPD across various scaffolds to integrate complementary training signals and strengthen agentic coding capabilities. Furthermore, model merging combines different experts and checkpoints from different stages while mitigating conflicts among their parameter updates. The resulting IQuest-Q1 demonstrates cap
    
[^58]: RetiJev：快速且具有来源依据的眼科决策模型

    RetiJev: Fast and Source-Grounded Ophthalmic Decision Models

    [https://arxiv.org/abs/2610.30947](https://arxiv.org/abs/2610.30947)

    该论文提出RetiJev——一系列无需自回归解码、可直接对候选答案评分的0.8B-9B眼科决策模型，并构建了来源可溯的RetiJev-Data语料库，在指南分级任务上以10.4-23.0个百分点的优势超越27B大语言模型和商业决策模型，同时实现更快、更保护隐私的推理。

    

    眼科临床决策支持需要快速、准确且保护隐私的推理，然而自回归大语言模型在预定义动作之间进行决策时会产生显著的延迟和计算开销。因此，我们研究了一种面向决策的生成式推理替代方案。具体而言，我们首先推出了RetiJev-Data，这是一个具有来源依据的眼科语料库，包含源自PubMed Central文章的78,158个临床决策与诊断问题（配有覆盖531种眼病的MeSH词汇表）、横跨11个分级系统的50,074个指南分级问题，以及一个包含24,662个问题的评估基准。在此基础上，我们开发了RetiJev，这是一系列0.8B至9B参数的决策模型，无需自回归解码即可直接对候选答案进行评分。在所有模型规模上，RetiJev在指南分级任务上均以10.4-23.0个百分点的优势超越了27B大语言模型和商业Jev 1.13决策模型（原文摘要至此截断）。

    arXiv:2610.30947v1 Announce Type: new  Abstract: Ophthalmic clinical decision support requires fast, accurate, and privacy-preserving inference, yet autoregressive large language models (LLMs) incur substantial latency and computational overhead for decisions among predefined actions. We therefore investigate a decision-oriented alternative to generative reasoning. Specifically, we first introduce RetiJev-Data, a source-grounded ophthalmology corpus comprising 78,158 clinical-decision and diagnosis questions derived from articles in PubMed Central and a MeSH vocabulary covering 531 eye diseases, and 50,074 guideline-grading questions spanning 11 grading systems, along with a 24,662-question evaluation benchmark. Based on this, we develop RetiJev, a family of 0.8B-9B decision models that directly score candidate answers without autoregressive decoding. Across all model sizes, RetiJev outperforms both a 27B LLM and the commercial Jev 1.13 decision model on guideline grading by 10.4-23.0 
    
[^59]: 追踪大语言模型在UML类图生成中的推理幻觉：实证分类体系与缓解研究

    Tracing Hallucinations in LLM Reasoning for UML Class Diagram Generation: An Empirical Taxonomy and Mitigation Study

    [https://arxiv.org/abs/2610.30946](https://arxiv.org/abs/2610.30946)

    该研究针对LLM生成UML类图过程中的推理幻觉，构建了包含提示忠实度、代码忠实度和推理质量三个维度共十一种子类型的实证分类体系，并发现代码忠实度错误是出现频率最高的幻觉类型。

    

    大型语言模型（LLM）能够根据自然语言需求生成UML类图，但幻觉问题可能会损害生成图的正确性和完整性。现有的评估方法大多只关注最终生成的图，而没有检查具备推理能力的LLM的中间推理轨迹，而这些轨迹可以揭示幻觉产生的位置。我们构建了一个包含十一种幻觉子类型的实证分类体系，分为三个维度：提示忠实度（需求与推理之间的偏差）、代码忠实度（推理与生成的PlantUML代码之间的差异）以及推理质量（推理过程内部的问题）。该分类体系基于60条人工标注的DeepSeek-R1推理轨迹构建，通过双LLM标注流水线应用于另外120条DeepSeek-R1轨迹，并在20条DeepSeek-V4-Pro轨迹上进行了验证。在两个数据集中，代码忠实度都是DeepSeek-R1中出现频率最高的幻觉维度。

    arXiv:2610.30946v1 Announce Type: cross  Abstract: Large Language Models (LLMs) can generate UML class diagrams from natural language requirements, but hallucinations may compromise the correctness and completeness of the resulting diagrams. Existing evaluations mostly assess the final diagram, without examining the intermediate traces of reasoning-capable LLMs, which can reveal where these hallucinations arise. We develop an empirical taxonomy of eleven hallucination sub-types organised into three dimensions: Prompt Faithfulness (deviations between requirements and reasoning), Code Faithfulness (discrepancies between reasoning and generated PlantUML code), and Reasoning Quality (problems within the reasoning). The taxonomy was derived from 60 manually annotated DeepSeek-R1 traces, applied to 120 further DeepSeek-R1 traces through a dual-LLM annotation pipeline, and examined on 20 DeepSeek-V4-Pro traces. Code Faithfulness was the most frequent dimension for DeepSeek-R1 in both datasets
    
[^60]: 用自然语言自编码器将激活差异转化为自然语言描述

    Verbalizing Activation Differences with Natural Language Autoencoders

    [https://arxiv.org/abs/2610.30934](https://arxiv.org/abs/2610.30934)

    本文系统研究了自然语言自编码器（NLAs）的泛化能力，实验证明其在组合结构、泛化到微调模型、恢复微调目标以及专门化模型差异分析四个方面均表现良好。

    

    激活言语化器可以将大语言模型的内部表示翻译成人类可读的文本。最近提出的自然语言自编码器（NLAs）通过自然语言瓶颈重建激活来学习这种转换。然而，这些方法在超出其训练设置的范围外泛化能力如何仍不清楚。本文围绕四个密切相关的问题对NLAs进行研究：（i）NLA的言语化描述对其输入是否表现出组合结构？（ii）NLAs能否泛化到其所训练模型的微调版本？（iii）NLAs能否在模型差异分析中恢复微调目标？（iv）NLAs能否被专门化用于模型差异分析？通过对来自三个系列的四个模型的实验，我们提供的证据支持对这四个问题基本肯定的答案。具体而言，组合类比激活的言语化描述大多能说出目标内容，而很少提及……

    arXiv:2610.30934v1 Announce Type: new  Abstract: Activation verbalizers translate the internal representations of large language models into human-readable text. The recently proposed natural language autoencoders (NLAs) learn this by reconstructing activations through a natural language bottleneck. However, it remains unclear how well these methods generalize beyond the setting they were trained for. Here, we study this question for NLAs along four closely related questions: (i) Do NLA verbalizations exhibit compositional structure with respect to their input? (ii) Do NLAs generalize to finetuned versions of the model they were trained for? (iii) Can NLAs recover finetuning objectives in model diffing? (iv) Can NLAs be specialized for model diffing? Across four models from three families, our experiments provide evidence supporting a largely positive answer to all four questions. Specifically, verbalizations of composed analogy activations mostly name the target while rarely mentionin
    
[^61]: ReflectPO：面向长程大语言模型智能体的反思引导策略优化

    ReflectPO: Reflection-Guided Policy Optimization for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2610.30933](https://arxiv.org/abs/2610.30933)

    ReflectPO是一个强化学习框架，通过反思引导采样和树结构信用分配两大核心组件，实现长程任务中大语言模型智能体的自主错误恢复。

    

    借助外部工具，大语言模型（LLM）能够通过与环境的持续交互来解决复杂任务。因此，基于LLM的智能体正被越来越多地用于处理真实世界环境中的复杂长程任务。然而，这些任务涉及顺序性的、相互依赖的决策以及环境状态的变化。早期的执行错误会影响后续的交互历史和环境状态，并且其影响会沿着决策链不断累积。为了应对这一挑战，我们提出了反思引导策略优化（ReflectPO），这是一个面向长程任务中自主错误恢复的强化学习框架。ReflectPO包含两个核心组件。第一个是反思引导采样，我们利用交互轨迹中的分支差异和重复访问来构建错误恢复的提示。第二个是树结构信用分配，我们将采样得到的轨迹组织成树形结构……

    arXiv:2610.30933v1 Announce Type: new  Abstract: With external tools, large language models (LLMs) can tackle complex tasks through sustained environmental interaction. LLM-based agents are therefore increasingly used to tackle complex long-horizon tasks in real-world environments. However, these tasks involve sequential, interdependent decisions and changes in environmental states. Early execution errors affect subsequent interaction histories and environmental states, and their effects accumulate along the decision chain. To address this challenge, we propose Reflection-Guided Policy Optimization (ReflectPO), a reinforcement learning framework for autonomous error recovery in long-horizon tasks. ReflectPO comprises two core components. The first is Reflection-Guided Sampling. We use branch differences and repeated visits in interaction trajectories to construct hints for error recovery. The second is Tree-Structured Credit Assignment. We organize sampled trajectories into trees and a
    
[^62]: 基于动态粒度发现与最优传输的层次化细粒度跨模态对齐

    Hierarchical Fine-Grained Cross-Modal Alignment via Dynamic Granularity Discovery and Optimal Transport

    [https://arxiv.org/abs/2610.30912](https://arxiv.org/abs/2610.30912)

    该论文提出层次化细粒度跨模态对齐框架HFCMA，通过动态粒度发现与最优传输，将低层图像块和文本标记逐步转化为自适应的视觉区域与文本短语，并在全局检索评分中实现结构化的区域-短语匹配，从而提升细粒度图文检索性能。

    

    细粒度图文检索不仅要求模型评估图像与文本之间的全局语义一致性，还需要模型为局部文本语义（包括对象、属性、数量、动作和关系）建模相应的视觉证据，以便区分语义相似的候选结果。现有方法通常使用固定的图像块、检测到的区域或文本标记作为基本匹配单元，然后通过最大池化、交叉注意力或相似度累积等方式聚合局部交互。这类策略在语义单元构建和全局匹配一致性方面仍存在局限。本文提出了一种层次化细粒度跨模态对齐框架，称为HFCMA。该框架逐步将低层的图像块和标记表示转化为自适应的视觉区域和文本短语，然后在全局检索评分过程中执行结构化的区域-短语匹配，……

    arXiv:2610.30912v1 Announce Type: cross  Abstract: Fine-grained image-text retrieval requires a model not only to assess global semantic consistency between an image and a text, but also to model visual evidence for local textual semantics, including objects, attributes, quantities, actions, and relations, so as to distinguish semantically similar candidates. Existing methods commonly use fixed image patches, detected regions, or text tokens as the basic matching units and then aggregate local interactions through max pooling, cross attention, or similarity accumulation. Such strategies remain limited in semantic unit construction and global matching consistency. This paper proposes a hierarchical fine-grained cross-modal alignment framework, termed HFCMA. The framework progressively transforms low-level patch and token representations into adaptive visual regions and textual phrases, and then performs structured region-phrase matching within a global retrieval scoring process. Specifi
    
[^63]: 硅基样本匹配了答案，却匹配不了受访者

    Silicon Samples Match the Answers but Not the Respondents

    [https://arxiv.org/abs/2610.30900](https://arxiv.org/abs/2610.30900)

    论文提出“受访者覆盖差距”（RCG）这一新评估指标，从完整答案组合的角度衡量合成受访者与真实受访者的差异，发现六种大语言模型虽然能复现答案分布，却无法覆盖真实人类受访者的答案特征组合。

    

    硅基采样通过让大语言模型（LLM）以具有特定人口统计属性的个人身份回答问卷来生成调查受访者，由此产生的合成人群正日益被用来替代真实人类人群。现有评估主要关注个体答案准确性、单个问题的答案分布以及跨问题的相关性，这些方法均无法比较受访者本身，因此即使某个生成器产生的受访者答案组合在真实人类受访者中从未出现过，只要它能复现每个问题的答案分布，仍能获得较高的评分。我们提出了“受访者覆盖差距”（respondent coverage gap, RCG），将每个受访者表示为由其完整答案组合定义的一个点，并衡量每个真实人类受访者与最近的生成样本之间的距离（相对于人类样本之间的距离）。在两项美国社会调查上，我们评估的全部六个大语言模型都未能覆盖真实人类受访者，并且在主要实验设置下，它们的……（原文摘要在此处截断）

    arXiv:2610.30900v1 Announce Type: new  Abstract: Silicon sampling generates survey respondents by having a large language model (LLM) answer a survey as a person with given demographic attributes, and the resulting synthetic populations are increasingly used in place of human ones. Existing evaluations focus on individual-answer accuracy, per-question answer distributions, and correlations across questions. None of these compares the respondents themselves, so a generator whose respondents have combinations of answers found in no human respondent can still score well if it reproduces every answer distribution. We propose the respondent coverage gap (RCG), which represents each respondent as a point defined by their complete answer profile and measures how far each human respondent is from the nearest generated one, relative to the distance between human samples. On two US social surveys, all six LLMs we evaluate leave human respondents uncovered, and under our main settings, they fall 
    
[^64]: 从施暴者数字足迹到行为社区：一个面向孟加拉语网络欺凌分析的本体驱动框架（完整版）

    From Perpetrators' Digital Footprints to Behavioral Communities: An Ontology-Driven Framework for Bengali Cyberbullying Analysis--- FULL VERSION

    [https://arxiv.org/abs/2610.30885](https://arxiv.org/abs/2610.30885)

    本文提出以施暴者为中心的本体驱动框架ODCBNA，通过知识图谱构建、行为建模、社区分析和影响力建模，在孟加拉语网络欺凌分析中实现持续性施暴者识别、行为社区刻画和有影响力用户评估。

    

    网络欺凌涉及超越孤立辱骂信息的行为和情境模式。然而，现有方法在内容检测、语义表示和网络分析等方面仍然碎片化，缺乏可解释的施暴者表示、多尺度结构建模，以及对行为驱动相似性和影响力传播的解释。这些局限阻碍了对持续性施暴者的识别、行为社区的刻画以及潜在有影响力用户的评估。本文提出了本体驱动的网络欺凌行为与网络分析框架（ODCBNA），这是一个以施暴者为中心的框架，整合了需求驱动的本体设计、知识图谱构建、行为建模、社区分析和影响力建模。以孟加拉语情境化多样化网络欺凌语料库（BenCoDiC）作为应用案例，ODCBNA识别出1,063名至少拥有三条欺凌相关内容的用户（摘要在此处不完整）。

    arXiv:2610.30885v1 Announce Type: new  Abstract: Cyberbullying involves behavioral and contextual patterns beyond isolated abusive messages. However, existing approaches remain fragmented across content detection, semantic representation, and network analysis, with limited interpretable perpetrator representations, multi-scale structural modeling, and explanations of behavior-driven similarity and influence propagation. These limitations hinder the identification of persistent perpetrators, characterization of behavioral communities, and assessment of potentially influential users. This paper proposes Ontology-Driven Cyberbullying Behavior and Network Analysis (ODCBNA), a perpetrator-centric framework integrating requirements-driven ontology design, knowledge graph construction, behavioral modeling, community analysis, and influence modeling. Using the Bengali Contextual and Diverse Cyberbullying (BenCoDiC) corpus as a use case, ODCBNA identifies $1{,}063$ users with at least three bul
    
[^65]: 承袭而非发明：追溯推理模型中“Wait”（等一下）一词的来源

    Inherited, Not Invented: Tracing the Provenance of Wait in Reasoning Models

    [https://arxiv.org/abs/2610.30882](https://arxiv.org/abs/2610.30882)

    研究揭示推理模型中的{wait}标记并非由强化学习发明，而是从训练数据（中期训练与推理轨迹微调）中继承而来，RL只能放大模型已有的使用习惯。

    

    强化学习（RL）被认为教会了推理模型在重新审视某一步骤之前先写下{wait}（等一下），然而RL带来的巨大准确率提升并不需要这个词。这种观点可能混淆了RL所“发现”的内容与模型从训练数据中“继承”的内容。我们通过开源模型的训练历史追溯这个词，采用无标签的频率、位置、形式和用法指标以及注入错误测试来分析它，所有指标均从采样输出中计算得出。我们发现zero-RL几乎从不创造这个标记：在缺乏该词的基础模型上，RL虽能大幅提升MATH-500准确率，但{wait}却几乎依然缺席。这个词是随训练数据而来的：它出现在OLMo-3的中期训练中，而基于推理轨迹的微调则将其引入大多数基础模型中原本缺乏该词的模型。在基础模型已经使用该词的情况下，两个已发布的RL-Zero配方（OLMo-3、MiMo）都会使其成倍增加，而来自同一基础模型的无KL正则化的GRPO则不会；OLMo-3的配方会磨损其形式，使其变得更加……

    arXiv:2610.30882v1 Announce Type: cross  Abstract: Reinforcement learning (RL) is credited with teaching reasoning models to write {wait} before reconsidering a step, yet large RL accuracy gains do not require the word. This view can confuse what RL discovers with what a model inherits from its training data. We trace the word through open models' training histories, reading it with label-free indices of frequency, position, form and usage and an injected-error test, all computed from sampled outputs. We find that zero-RL almost never coins the marker: on bases that lack the word, RL raises MATH-500 accuracy substantially while {wait} stays near absent. It arrives with training data: mid-training in OLMo-3, and fine-tuning on reasoning traces installs it in most models whose base lacks it. Where the base already uses it, both released RL-Zero recipes (OLMo-3, MiMo) multiply it, whereas KL-free GRPO from the same base does not; OLMo-3's recipe wears down its form and makes it a more sen
    
[^66]: 一个教师，多个视角：降低在线策略自蒸馏中的参考敏感性

    One Teacher, Many Views: Reducing Reference Sensitivity in On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.30880](https://arxiv.org/abs/2610.30880)

    本文发现在线策略自蒸馏中单一完整参考条件会导致教师分布对参考的选择高度敏感（参考诱导的目标不稳定性），并提出蒙特卡洛特权蒸馏方法，通过利用参考的多个视角来降低这种参考敏感性。

    

    在线策略自蒸馏（OPSD）通过在模型自身生成的轨迹上进行训练，并利用一个以参考解答为条件的特权副本提供密集的token级监督，从而提升推理能力。现有方法通常使用以完整参考为条件的单一教师分布。我们发现了这种形式的一个局限性：参考只是推理过程的一种实现方式。尽管它提供了关于答案的证据，但它同时编码了特定的顺序、措辞和求解路径。因此，以完整参考为条件可能会产生过度依赖于学生模型在推理时无法获得的信息的正确目标。我们证明，在保持问题和学生状态不变的情况下，用一个有效参考替换另一个有效参考，会显著改变教师分布，我们将这种现象称为参考诱导的目标不稳定性。为了解决这种不稳定性，我们引入了蒙特卡洛特权蒸馏（

    arXiv:2610.30880v1 Announce Type: cross  Abstract: On-policy self-distillation (OPSD) improves reasoning by training a model on its own trajectories with dense token-level supervision from a privileged copy conditioned on a reference solution. Existing methods typically use a single teacher distribution conditioned on the complete reference. We identify a limitation of this formulation: a reference is only one realization of the reasoning process. Although it provides evidence about the answer, it also encodes an ordering, phrasing, and solution path. Full-reference conditioning can therefore produce correct targets that depend excessively on information unavailable to the student at inference. We show that replacing one valid reference with another, while holding the problem and student state fixed, can substantially alter the teacher distribution, a phenomenon we call reference-induced target instability. To address this instability, we introduce Monte Carlo Privileged Distillation (
    
[^67]: 合成到真实的语码转换语音识别：一个自发性尼泊尔语-英语播客基准

    Synthetic-to-Real Code-Switched ASR: A Spontaneous Nepali-English Podcast Benchmark

    [https://arxiv.org/abs/2610.30860](https://arxiv.org/abs/2610.30860)

    该论文构建了一个经过人工审核的13,044条尼泊尔语-英语语码转换自发性播客基准数据集，并提出两阶段合成到真实适配框架——先用TTS合成语音做SFT，再用带转录与文字系统奖励的GRPO——以低成本缓解文字系统混淆和语言切换点错误率高的难题。

    

    尼泊尔语-英语语码转换的会话语音给自动语音识别（ASR）带来了严峻挑战，主要表现为文字系统混淆（天城文与拉丁字母书写方式的不当混用）以及语言切换点处错误率的升高。监督微调（SFT）可以适配多语言基础模型，但人工语码转换音频的收集与标注成本高昂。我们构建了一个新的基准数据集，包含13,044条自发性播客语句（总计15.45小时；测试集10,214条，开发集2,830条），并经过审核验证以消除机器翻译泄漏和声学循环问题。我们研究了一种两阶段的合成到真实适配框架：第一阶段对通过多说话人、双引擎TTS流程合成的0.95小时语音进行SFT；第二阶段探索使用转录奖励和文字系统奖励的组相对策略优化（GRPO）。我们发现合成数据上的SFT为自发性人类语音提供了主要的跨域迁移。

    arXiv:2610.30860v1 Announce Type: new  Abstract: Nepali-English code-switched conversational speech presents severe challenges for automatic speech recognition (ASR), chiefly through script confusion (inappropriate mixing of Devanagari and Latin orthography) and elevated error rates at language switch points. Supervised fine-tuning (SFT) can adapt multilingual foundation models, but human code-switched audio is expensive to collect and annotate. We curate a new benchmark of 13,044 spontaneous podcast utterances (15.45 h total; 10,214 test, 2,830 dev), audited and verified to eliminate machine translation leakages and acoustic loops. We investigate a two-stage synthetic-to-real adaptation framework: Stage 1 applies SFT on 0.95 h of speech synthesized via a multi-speaker, dual-engine TTS pipeline; Stage 2 explores Group Relative Policy Optimization (GRPO) with transcription and script rewards. We find that synthetic SFT provides the primary cross-domain transfer to spontaneous human spee
    
[^68]: TRIAD：面向多模态大语言模型的全模态歧义基准测试

    TRIAD: Benchmarking Omni-Modal Ambiguity in Multimodal Large Language Models

    [https://arxiv.org/abs/2610.30841](https://arxiv.org/abs/2610.30841)

    提出了TRIAD——一个全模态歧义消解基准，通过精心构建的文本-图像-音频三元组（只有三者齐全才有唯一答案）测试多模态大语言模型的跨模态歧义消解能力，并揭示现有全模态系统的条目级性能无法等同于真正的歧义消解能力。

    

    歧义是自然交流的核心特征：一个句子、图像或声音在未考虑其他渠道的证据之前可能仍是欠明确的。然而，当前的多模态基准大多测试对已经确定输入的感知、识别或推理能力，尚不清楚全模态大语言模型能否消解分布于文本、视觉和音频中的歧义。我们提出了TRIAD，一个用于全模态歧义消解的基准。每个条目由一个文本-图像-音频三元组以及一个问题和一组答案选项组成，其构建方式确保完整的三元组能够确定唯一答案，而移除任何一个模态都会使一致选项的集合变得不唯一。TRIAD包含86个场景组中的327条人工编写的查询，覆盖三种模态下的18个歧义类别。对14个全模态系统的评估表明，强大的条目级性能并不能转化为真正的跨模态歧义消解能力。

    arXiv:2610.30841v1 Announce Type: new  Abstract: Ambiguity is a central feature of natural communication: a sentence, image, or sound may remain underspecified until evidence from the other channels is considered. Yet current multimodal benchmarks mostly test perception, recognition, or reasoning over already determinate inputs, leaving unclear whether omni-modal large language models can resolve ambiguity distributed across text, vision, and audio. We introduce TRIAD, a benchmark for omni-modal ambiguity resolution. Each item is a text--image--audio triplet paired with a question and a set of answer options, and is constructed so that the full triplet determines a unique answer while removing any one modality makes the set of consistent options non-unique. TRIAD contains $327$ hand-written queries in $86$ scene groups and covers $18$ ambiguity categories across the three modalities. Evaluating $14$ omni-modal systems reveals that strong item-level performance does not translate into r
    
[^69]: TutorCaseBench：一个基于案例的基准，用于评估LLM导师在支持具有挑战性行为的学生方面的表现

    TutorCaseBench: A Case-Grounded Benchmark for Evaluating LLM Tutors in Supporting Students with Challenging Behaviors

    [https://arxiv.org/abs/2610.30828](https://arxiv.org/abs/2610.30828)

    提出了TutorCaseBench基准，基于教育学、教育心理学和特殊教育的真实案例评估LLM导师应对学生挑战性行为的能力，发现即使是最强的模型在开放式案例中也仅满足55.8%的评分标准，且模型识别恰当干预策略的能力优于生成情境化回复的能力。

    

    大型语言模型越来越多地被用作AI导师，然而现有的基准测试忽视了在学习和行为困难学生中常见的挑战性行为。我们介绍了TutorCaseBench，这是一个将AI导师评估建立在来自教育学、教育心理学和特殊教育的定性案例研究基础之上的基准。我们的流程将记录在案的学生类别转化为基于源证据的特定案例场景、师生对话和评分标准。该基准包含923段对话，涵盖五种挑战类型，通过开放式回复生成（1,560个题目）和多项选择（2,537个题目）进行评估。在11个最先进的大型语言模型中，即使是最强的模型，在开放式案例中也仅有55.8%的情况满足评分标准，且模型在选择恰当策略方面比生成情境化回复的表现更好。这些结果揭示了干预识别能力与情境特定应用能力之间的差距。

    arXiv:2610.30828v1 Announce Type: new  Abstract: LLMs are increasingly used as AI tutors, yet existing benchmarks overlook the challenging behaviors common among students with learning and behavioral difficulties. We introduce TutorCaseBench, a benchmark that grounds AI-tutor evaluation in qualitative case studies from pedagogy, educational psychology, and special education. Our pipeline converts documented student categories into case-specific scenarios, tutor-student dialogues, and rubrics grounded in source evidence. The benchmark contains 923 dialogues across five challenge types, evaluated through open-ended response generation (1,560 items) and multiple-choice selection (2,537 items). Across 11 state-of-the-art LLMs, even the strongest model satisfies a rubric criterion in only 55.8% of open-ended cases, while models are better at selecting appropriate strategies than generating situated responses. These results reveal a gap between intervention recognition and context-specific a
    
[^70]: 隐形化：语言模型概率偏差的一种来源

    Invisibilization as a Source of Probability Bias in Language Models

    [https://arxiv.org/abs/2610.30785](https://arxiv.org/abs/2610.30785)

    本研究通过具有精确贝叶斯答案的双瓮博弈实验发现，领先的大语言模型会因报告者的种族等受保护属性描述而系统性拒绝信任其报告——对被描述为黑人的报告者的拒绝率高达 46%——揭示了模型对齐训练中的“隐形化”现象会导致基于信息来源而非证据质量的概率偏差。

    

    语言模型经过对齐训练以避免使用种族、族裔和性别等受保护属性来影响其回复。因此，当此类属性出现在提示中时，模型越来越多地选择拒绝回答或含糊其辞。然而，如果一个系统必须决定从人们告知它的信息中学习多少，那么因信息来源而拒绝有价值的信息将付出高昂代价。在本文中，我们通过一个具有精确贝叶斯答案的双瓮博弈来测量拒绝信息的代价。我们向来自两家开发商（OpenAI 和 Anthropic）的领先大语言模型提供了一个提示，要求它们决定在多大程度上信任来自某位参与者的一份无法核实的报告，而该参与者的描述在十种配置中随机变化（共 6,600 次会话）。我们表明，领先的模型经常拒绝评估报告者，其将信息视为无价值的比率取决于描述而非证据本身。在最极端的情况下，模型在 46% 的时间里拒绝了被描述为黑人的报告者。

    arXiv:2610.30785v1 Announce Type: new  Abstract: Language models are aligned to avoid using protected attributes such as race, ethnicity, and gender to inform their responses. Therefore, they increasingly refuse or hedge when such attributes appear in a prompt. However, if a system must decide how much to learn from what people tell it, it is costly to refuse valuable information because of its source. In this article, we measure the cost of declining information in a two-urn game with an exact Bayesian answer. We present leading LLM models from two developers (OpenAI and Anthropic) with a prompt that requires them to decide how far to trust an unverifiable report from a participant whose description we randomize across ten configurations (6,600 sessions). We show that leading models often decline to assess the reporter, treating information as worthless at rates set by the description rather than by the evidence. At its widest, models decline a Black-described reporter 46% of the time
    
[^71]: MixJudge：学习一种面向人声混音的语言可查询评判器

    MixJudge: Learning a Language-Queryable Critic for Vocal Mixing

    [https://arxiv.org/abs/2610.30783](https://arxiv.org/abs/2610.30783)

    提出MixJudge，一个基于11种多严重程度人声混音问题课程训练的语言可查询音频评判器，能够检索和区分混音问题、评估问题严重程度，并可将其判断迁移至处理候选选择任务，支持下游混音决策。

    

    混音不仅需要应用处理手段，还需要评估混音后的结果。音频-语言模型能够提供对话式的混音建议，但现有评估并未单独测试这些模型能否判断具体的混音问题。由于此类判断的专业监督数据稀缺，我们构建了一套涵盖11种人声混音问题、包含多个严重程度级别的课程化训练体系，并用它训练了MixJudge——一个可通过语言查询的音频评判器。MixJudge展现了检索混音问题、区分相反问题以及对问题严重程度进行排序的能力。此外，其判断能力还可迁移到菜单式选择任务中：给定一组处理候选方案时，MixJudge往往更青睐在声学上更接近留出专业版本的方案，这证明了显式的混音判断能力如何能够支持比较任务和下游的混音决策。

    arXiv:2610.30783v1 Announce Type: cross  Abstract: Mixing requires not only applying processing but also assessing the resulting mix. Audio-language models can provide conversa- tional mixing advice, but existing evaluations do not separately test whether they can judge specific mixing issues. Since professional supervision for such judgments is scarce, we construct a curriculum of 11 vocal-mix issues at multiple severity levels and use it to train MixJudge, a language-queryable audio critic. MixJudge shows its ability to retrieve mixing issues, distinguish opposite issues, and order the severity of an issue. Moreover, its judgments transfer to menu selection: given a set of processing candidates, MixJudge often favors versions acoustically closer to a held-out professional version, demonstrating how explicit mixing judgment can support comparison and downstream mixing decisions.
    
[^72]: 意义汇聚与分歧之处：跨81种语言重新审视语言相对论

    Where meanings converge and diverge: Revisiting linguistic relativity across 81 languages

    [https://arxiv.org/abs/2610.30778](https://arxiv.org/abs/2610.30778)

    本研究通过比较81种语言中196个基本概念的词嵌入发现，社会文化意义跨语言高度趋同而自然与运动意义分歧更大，揭示了概念结构中普遍性与语言相对性的交织。

    

    说不同语言的人是否以相似的方式组织概念？这个问题处于萨丕尔-沃尔夫假说与语言相对论争论的核心。然而，共享的基本意义并不一定意味着它们之间的关系也相同。我们比较了81种语言中196个基本概念之间的关系，使用来自两个多语言模型的词嵌入来量化语言所表达的概念结构。研究发现，社会文化意义方面的跨语言一致性高于有关自然和运动的意义，这一对比在不同模型和分类体系中反复出现。在社会文化领域内，关系差异随地理距离的增加而增大；亲缘关系较远的语言在所有领域中差异更大。这些发现表明，普遍性与相对性在概念领域中相互交织。研究结果支持这样一种解释：共享的认知资源与语言和地理环境相互作用。

    arXiv:2610.30778v1 Announce Type: new  Abstract: Do people who speak different languages organize concepts in similar ways? This question lies at the heart of the Sapir--Whorf hypothesis and debates about linguistic relativity. Shared basic meanings, however, need not imply shared relations among them. We compared relations among 196 basic concepts across 81 languages, using word embeddings from two multilingual models to quantify conceptual structure expressed in language. Cross-language correspondence was greater for sociocultural meanings than for meanings concerning nature and motion, a contrast recurring across models and classifications. Within the sociocultural domain, relational differences increased with geographic separation; less closely related languages differed more across all domains. These findings suggest that universality and relativity are interwoven across conceptual domains. They inform an account in which shared cognitive resources interact with linguistic and loc
    
[^73]: 抽象的优先级，具体的奖励：陈述性偏好与人们在AI回复中实际奖励内容的关系

    Abstract Priorities, Concrete Rewards: How Stated Preferences Relate to What People Reward in AI Responses

    [https://arxiv.org/abs/2610.30763](https://arxiv.org/abs/2610.30763)

    该研究基于PRISM数据集发现，人们对AI回复的评分受回复长度、格式、拒绝语言等未被直接询问的具体特征的影响不亚于被调查的属性，且陈述的偏好只有在模型整体感知层面（而非文本指标层面）才与实际奖励行为一致，表明陈述性偏好未必反映人们真正奖励的内容。

    

    基于大语言模型（LLM）构建的AI助手通过人类反馈进行训练，并日益根据用户陈述的偏好进行个性化定制。然而，这些偏好与人们在回复中实际奖励的内容之间的吻合程度仍不清楚。我们在公开的PRISM数据集中比较了陈述性偏好与显示性偏好：在该数据集中，1,396名参与者报告了七个回复属性对他们的重要程度，随后对来自21个大语言模型的29,347条回复进行了评分。在针对同一提示词的回复中，问卷从未询问过的特征——如回复长度、格式和拒绝语言——与评分的关联强度至少不亚于所调查属性的文本代理指标。陈述的优先级是否与评分相符，取决于属性的测量方式：重视某一属性的参与者更偏好其他用户认为在该属性上表现出色的模型所产生的回复，而非在相应文本代理指标上得分更高的回复。书面反馈则更为……（摘要在此处截断）

    arXiv:2610.30763v1 Announce Type: cross  Abstract: AI assistants built on large language models (LLMs) are trained on human feedback and increasingly personalized to users' stated preferences. Yet how closely these preferences reflect what people reward in responses remains unclear. We compare stated and revealed preferences in the public PRISM dataset, in which 1,396 participants reported how much seven response attributes mattered to them and then rated 29,347 responses from 21 LLMs. Among responses to the same prompt, features the survey never asked about, such as response length, formatting and refusal language, were at least as strongly associated with ratings as text proxies of the surveyed attributes. Whether stated priorities matched ratings depended on how attributes were measured: participants who prioritized an attribute favoured responses from models that others perceived as strong on it, but not responses that scored higher on text proxies for it. Written feedback was more
    
[^74]: AI模型的伦理边界有多牢固？一项针对15个模型的工资单编辑测试

    How Firm Are AI Models' Ethical Boundaries? A Pay-Stub Editing Test of 15 Models

    [https://arxiv.org/abs/2610.30751](https://arxiv.org/abs/2610.30751)

    该研究通过让15个AI模型执行相同的工资单收入修改请求发现，措辞的微小变化（如省略贷款用途或称修正打印错误）会使模型执行编辑的比例从14.1%升至66.2%，表明AI模型的伦理拒绝边界对措辞高度敏感。

    

    措辞上的微小变化能否让AI模型从拒绝一个敏感请求转变为执行它？我们用相同的请求测试了15个模型，要求增加工资单上显示的收入。在445次可观察的试验中，当说明贷款用途时，模型发出编辑调用的比例为14.1%；当省略贷款用途时为33.1%；当修改被描述为修正打印错误时则高达66.2%。Astra拒绝了全部十次涉及贷款用途的请求，但在所有十次省略用途的试验中都发出了调用。Fable、Opus和Muse Spark在全部三种措辞下均未发出任何调用；Sonnet则在后两种措辞下发出了一些调用。回复摘录展示了模型拒绝、转移话题或为编辑辩护的不同方式。该文件是一份标注的测试文档，所有工具均为模拟。这些结果探究了伦理边界在单一情境下的表现，并不能证明一个整体的伦理排名，也不能证明现实中发生了有害的成功编辑。

    arXiv:2610.30751v1 Announce Type: cross  Abstract: Can a small change in wording move an AI model from refusing a sensitive request to acting on it? We test 15 models on the same request to increase the income shown on a pay stub. Across 445 observable trials, models issue editing calls in 14.1% of cases when a loan purpose is stated, 33.1% when the purpose is omitted, and 66.2% when the change is described as fixing a printing error. Astra refuses all ten loan-purpose requests but issues calls in all ten purpose-omitted trials. Fable, Opus, and Muse Spark issue none across all three wordings; Sonnet issues some under the latter two. Response excerpts illustrate different ways of refusing, redirecting, or justifying an edit. The file is a marked test document and all tools are simulated. These results probe how boundaries behave in one setting; they do not prove an overall ethics ranking or that a harmful real-world edit succeeded.
    
[^75]: UniGuard：面向跨接口安全泛化的多视图护栏学习

    UniGuard: Multi-View Guardrail Learning for Cross-Interface Safety Generalization

    [https://arxiv.org/abs/2610.30742](https://arxiv.org/abs/2610.30742)

    UniGuard提出了一种多视图护栏训练框架，通过将同一安全案例转换为提示、响应、多轮对话、智能体轨迹等多种对齐视图来学习接口不变的安全表示，使0.6B规模的护栏模型在未见过的接口上平均提升45.12个F1点并显著改善置信度校准。

    

    护栏模型越来越多地被用作大语言模型（LLM）和智能体系统的安全层，但大多数护栏模型是在固定的提示或响应接口上训练的，当同一个安全案例以对话、智能体轨迹、多智能体追踪或策略条件输入等形式出现时，这些模型就会变得不可靠。我们提出了UniGuard，这是一个训练框架，它从同一个安全案例的多个语义对齐的视图中学习接口不变的安全表示。我们还构建了UniGuard-FT数据集，将1,000个公开安全案例转换为提示、响应、多轮对话、智能体轨迹和多智能体轨迹五种视图，并通过分类损失、监督对比学习和非对称置信度校准损失训练了UniGuard-0.6B模型。在未见过的接口基准测试ATBench、PolicyGuardBench和DynaBench上，UniGuard-0.6B相比其预训练护栏平均提升了45.12个F1点，并将期望校准误差降低了3（摘要在此处被截断）。

    arXiv:2610.30742v1 Announce Type: new  Abstract: Guardrail models are increasingly used as safety layers for LLMs and agentic systems, but most are trained on fixed prompt or response interfaces and become unreliable when the same safety case appears as a conversation, agent trajectory, multi-agent trace, or policy-conditioned input. We propose UniGuard, a training framework that learns interface-invariant safety representations from multiple semantically aligned views of the same safety case. We also construct the UniGuard-FT dataset by transforming 1,000 public safety cases into prompt, response, multi-turn conversation, agent-trace, and multi-agent-trace views, and train UniGuard-0.6B with classification, supervised contrastive learning, and asymmetric confidence calibration losses. Across unseen-interface benchmarks ATBench, PolicyGuardBench, and DynaBench, UniGuard-0.6B improves over its pretrained guardrail by 45.12 F1 points on average and reduces expected calibration error by 3
    
[^76]: 意义契约：大语言模型中的局部语义几何

    Meaning Contracts: Local Semantic Geometry in Large Language Models

    [https://arxiv.org/abs/2610.30715](https://arxiv.org/abs/2610.30715)

    本文提出“意义契约”的概念，通过跨语言扰动探针和语义细化探针，在四个架构各异的大语言模型上首次系统刻画了概念内部的局部语义几何——即单一语义因子的哪些信息在不同实现中保持不变、哪些会变化，以及意义细化时如何变化。

    

    本文研究了大语言模型中语义意义的局部几何结构。当前的表示分析关注的是意义在嵌入空间中是否相近，或者线性探针能否区分不同语义类别，但它们并未刻画概念内部的局部几何：在单一语义因子的不同实现中什么是不变的、什么是变化的，以及当意义被细化时会发生什么改变。本文通过在四个架构各异的仅解码器大语言模型（Gemma 4、Llama 3.2、Qwen 2.5、Pythia）上评估的两个受控探针来刻画这一几何结构。跨语言探针将语言作为扰动，用以区分精确意义匹配、同一语义因子的不同实现、相近概念以及域外对照；语义细化探针则将承载特定性的语义收窄与高概率的定语性修饰及对抗性对照区分开来。两个探针均通过盲测的人类判断进行了验证。四项发现在……（原文在此处截断）

    arXiv:2610.30715v1 Announce Type: new  Abstract: This paper studies the local geometry of semantic meaning in large language models. Current representation analyses ask whether meanings are close in embedding space, or whether linear probes separate semantic classes, but they do not characterize the within-concept local geometry: what is invariant across realizations of a single semantic factor, what varies, and what changes when meaning is refined. This paper characterizes that geometry through two controlled probes evaluated on four architecturally diverse decoder-only LLMs (Gemma 4, Llama 3.2, Qwen 2.5, Pythia). A cross-lingual probe uses language as a perturbation to separate exact meaning matches, same-factor realizations, nearby concepts, and out-of-domain contrasts. A refinement probe separates specificity-bearing semantic narrowing from high-probability attributive elaboration and adversarial controls; both probes are validated by blind human judgments. Four findings recur acro
    
[^77]: ULME：通过预测性耦合实现嵌入与语言建模的相互增强

    ULME: Predictive Coupling for Mutually Reinforcing Embedding and Language Modeling

    [https://arxiv.org/abs/2610.30705](https://arxiv.org/abs/2610.30705)

    ULME提出了一种后训练框架，通过预测性耦合机制让文本嵌入与语言建模两种能力在同一因果解码器大语言模型上真正统一并相互增强，突破了以往统一模型中两种目标互不监督的局限。

    

    文本嵌入和文本生成是现代大语言模型最有用的两种能力，然而它们通常由单独的专门模型分别提供（嵌入器和生成器），这些模型使用不兼容的目标进行训练，并在检索增强生成和智能体系统中并行部署。近期出现的统一嵌入-生成模型（如GritLM）在两种能力之间共享参数，但它们要么在不同模式之间切换注意力机制，要么将嵌入和语言建模作为从未相互监督的并行目标进行优化。我们认为这些设计只是部分统一：两种能力虽然共存于相同的权重之上，但并未相互塑造。我们提出了ULME，这是一个后训练框架，通过预测性耦合在因果解码器架构的大语言模型上统一嵌入和语言建模。ULME在查询和答案之间插入一串可学习的槽位，读取语义嵌入信息，使两种能力能够相互增强。

    arXiv:2610.30705v1 Announce Type: new  Abstract: Text embedding and text generation are two of the most useful capabilities of modern large language models, yet they are typically provided by separate specialized models (an embedder and a generator) trained with incompatible objectives and deployed side by side in retrieval-augmented generation and agent systems. Recent unified embedding-generation models such as GritLM share parameters between the two capabilities, but they either switch attention regimes between modes or optimize embedding and language modeling as parallel objectives that never supervise each other. We argue that these designs are only partially unified: the two capabilities coexist on the same weights, but do not shape each other. We propose ULME, a post-training framework that unifies embedding and language modeling on a causal decoder-only LLM through predictive coupling. ULME inserts a chain of learnable slots between the query and the answer, reads a semantic em
    
[^78]: 解耦多语言大语言模型中的鲁棒性与谄媚性

    Disentangling Robustness and Sycophancy in Multilingual LLMs

    [https://arxiv.org/abs/2610.30696](https://arxiv.org/abs/2610.30696)

    该论文提出SC矩阵将多语言大语言模型的能力不足与谄媚行为解耦区分，发现多语言谄媚并非单一现象而是由不同失败模式构成的复合体，且面向推理的缓解策略优于指令式策略。

    

    当多语言大语言模型认同用户的错误前提时，它是在谄媚迎合，还是仅仅因为在该语言下不知道更好的答案？当前的研究并未区分这两种情况：它们只记录模型是否表示认同，而不关注背后的原因。我们提出了SC矩阵（SC-matrix），将模型的能力（在无压力情况下是否能正确回答）与其谄媚性（在压力下是否坚持该答案）进行交叉分析。我们的结果显示：（1）相同的总体谄媚率可能掩盖不同的失败构成；（2）模型在某种语言上的高能力并不总能抵御谄媚行为；（3）面向推理的缓解策略优于基于指令的策略。我们发现多语言谄媚性并非单一现象，而是由需要不同诊断方法的各种失败模式组成的复合体。

    arXiv:2610.30696v1 Announce Type: cross  Abstract: When a multilingual LLM agrees with a user's false premise, is it being sycophantic, or does it simply not know better in that language? Current research does not distinguish these cases: they record only whether the model agreed, not why. We introduce the SC-matrix, that crosses a model's competence (whether it answers correctly under no pressure) with its sycophancy (whether it maintains that answer under pressure). Our results show that (1) identical aggregate sycophancy rates can mask different failure compositions, (2) high competence in a language does not always protect against sycophancy, and (3) reasoning-oriented mitigations outperform directive-based strategies. We find that multilingual sycophancy is not a single phenomenon but a composite of different failures that require distinct diagnosis.
    
[^79]: 从临床判断到可核查规则：由临床医生主导的心理健康聊天机器人安全护栏

    From Clinical Judgment to Checkable Rules: Clinician-Led Guardrails for Mental-Health Chatbots

    [https://arxiv.org/abs/2610.30690](https://arxiv.org/abs/2610.30690)

    该论文提出由临床医生主导定义并经形式化验证的安全规则库，将专家临床判断转化为可解释、可复查、可重算的心理健康聊天机器人安全护栏。

    

    在心理健康场景中部署的大语言模型（LLM）虽然可以对安全性失效进行评估，但由此产生的判断往往难以审查或解释。提升安全性的常见方法依赖于提示工程、LLM-as-a-Judge（以大模型作为评判者）以及宪法AI（Constitutional AI）。在本工作中，我们提出了一种由临床医生主导的替代方案：由两名临床医生定义、确定并裁决规则库中的四项安全检查，该规则库经过形式化验证以确保内部一致性，并作为护栏应用于大语言模型的输出之上。在本文及标题中，“临床医生主导”指的是这一规则库本身，而非事实提取器或不确定性默认处理机制——这两者是我们的贡献。由此产生的验证层所做的每一个判定，都可以通过一条具名规则和十六项事实重新计算得出。与一个在模型、版本、量化、规则文本和解码设置上均完全一致、但缺少该层所计算的988/911求助热线事实的LLM评判者相比，最大的分歧类别——即规则被满足的类别——

    arXiv:2610.30690v1 Announce Type: new  Abstract: Large language models (LLMs) deployed in mental-health contexts can be evaluated for safety failures, but the resulting judgments are often difficult to inspect or explain. Common approaches to improving safety rely on prompt engineering, LLM-as-a-Judge, and Constitutional AI. In this work, we describe a clinician-led alternative in which two clinicians define, finalize, and adjudicate the four safety checks of a rule base, which is formally verified for internal consistency and applied as a guardrail over LLM outputs. Here and in the title, clinician-led refers to this rule base, not to the fact extractor or the uncertainty default, which are ours. Every verdict of the resulting verification layer is recomputable from a named rule and sixteen facts. Against an LLM judge matched in model, revision, quantization, rule text, and decoding but without the layer's computed 988/911 fact, the largest disagreement class, the satisfied-rule class
    
[^80]: 通过词长直接提示词的身份：来自汉语同音词的证据

    Direct cueing of word identity through word duration: evidence from Mandarin homophones

    [https://arxiv.org/abs/2610.30682](https://arxiv.org/abs/2610.30682)

    本研究通过汉语同音词排序任务发现，听者能够利用词长（高频词更短）这一语音线索直接识别词的身份，证明词长在口语词汇识别中具有功能性作用。

    

    越来越多的证据表明，词汇在心理词典中以包含一定语音细节的形式存储，但关于究竟存储了哪些语音细节，以及这些细节在口语词汇识别中是否具有功能性作用，仍存在未解决的问题。词汇层面的语音线索（如词长）为此提供了一个合适的测试案例。在包括汉语在内的多种语言中，高频词比低频词更短。我们采用同音词排序任务，考察词长差异是否会影响词汇识别。之所以选择汉语进行研究，是因为汉语拥有大量的双音节同音词。结果表明，被试能够利用词长来识别同音词对的顺序，这些同音词对是按照词汇频率差异筛选的。无论是在高噪声条件下，还是在没有其他线索可用的情况下，被试都会将时长较短的听觉刺激映射到频率较高的词上。这些结果表明……

    arXiv:2610.30682v1 Announce Type: new  Abstract: While there is increasing evidence that words are stored in the mental lexicon with some phonetic detail, there are open questions about exactly which phonetic details are stored and whether they are functionally relevant for spoken word recognition. Word-level phonetic cues, such as word duration, provide an appropriate test case. Across languages, including Mandarin Chinese, high frequency words are shorter than low frequency words. Using a homophone ordering task, we investigate whether differences in word duration impact word recognition. Mandarin Chinese was examined since it has abundant disyllabic homophones. Our results indicate that participants were able to use word duration to recognize the order of homophone pairs, selected to vary in lexical frequency. Both with high noise levels and when no other cues were available, participants mapped the shorter duration auditory stimulus to the higher frequency word. The results provide
    
[^81]: 安全探针的缩放定律

    Scaling Laws for Safety Probes

    [https://arxiv.org/abs/2610.30678](https://arxiv.org/abs/2610.30678)

    本文通过构建六个新安全数据集、在20个基础模型上以不同标签规模训练探针，首次提出了可依据模型规模和标签数量提前预测安全探针性能的缩放定律。

    

    安全探针是在模型激活值上训练的、用于实时监控语言模型的标准工具。每当发布新模型或需要监控新的危害领域时，探针都必须重新训练，这需要新的标签以及探针架构的选择。然而，无论是模型还是标签都无法预先告诉我们最终探针的性能如何，要弄清这一点需要训练并比较多种探针配置。在本文中，我们探讨是否可以仅凭模型规模和标签数量来提前预测探针的性能。现有的安全数据集不足以支持这一研究，因为它们要么规模有限，要么已经饱和。因此，我们构建了六个规模足够大、难度足够高的新安全数据集。在这些数据集上，我们在来自五个模型家族的20个基础模型上训练了两种探针架构，每个模型使用32到10⁶个标签。我们使用探针误差来衡量性能……

    arXiv:2610.30678v1 Announce Type: cross  Abstract: Safety probes trained on a model's activations are a standard tool for monitoring language models in real time. Each time a new model is released or a new harm area needs to be monitored, the probe must be retrained, which requires new labels and a choice of probe architecture. Yet neither the model nor the labels tell us in advance how well the resulting probe will perform. Finding out requires training and comparing many probe configurations. In this paper, we ask whether probe performance can instead be predicted in advance from model size and the number of labels. Existing safety datasets are insufficient for this, as they are either limited in scale or already saturated. We therefore construct six new safety datasets of sufficient size and difficulty. On these datasets, we train two probe architectures on 20 base models from five model families, with 32 to $10^{6}$ labels for each model. We measure performance using probe error, d
    
[^82]: WikiOriginQA：用于分析文化偏见的、包含知识来源信息的问答数据集的自动化构建

    WikiOriginQA: Automated Construction of a QA Dataset with Knowledge Origins for Analyzing Cultural Bias

    [https://arxiv.org/abs/2610.30677](https://arxiv.org/abs/2610.30677)

    提出WikiOriginQA，利用维基百科不同语言版本的创建日期客观表征知识来源并自动构建问答数据集，实现无需人工标注的可扩展文化偏见评估，实验发现大语言模型存在偏向英语文化的偏见，且知识向英语圈的传播至关重要。

    

    现有关于文化知识的数据集通常由熟悉各自文化的人类标注者创建。然而，这一过程需要大量人力投入，给可扩展性带来了重大挑战。为了解决这一问题，我们提出了WikiOriginQA，这是一个通过不同语言版本的维基百科创建日期来客观表征知识来源、进而自动构建的问答数据集。基于该数据集的实验结果证实，大语言模型（LLM）的性能往往偏向英语文化。此外，结果表明知识向英语圈传播的重要性。这些发现验证了我们自动化方法的有效性，使得无需大量人工标注即可跨多种语言以卓越的可扩展性进行文化偏见评估。

    arXiv:2610.30677v1 Announce Type: new  Abstract: Existing datasets on cultural knowledge have been typically created by human annotators familiar with each culture. However, the process requires considerable human effort, presenting a significant challenge to scalability. To address this challenge, we introduce WikiOriginQA, a QA dataset automatically constructed by objectively representing the origins of knowledge through the creation dates of different Wikipedia language editions. Experimental results with the created dataset confirm that LLM performance tends to be biased toward Anglophone cultures. Furthermore, the results suggest the importance of knowledge propagation into the Anglosphere. These findings demonstrate the validity of our automated approach, enabling cultural bias evaluations with superior scalability across diverse languages without extensive manual annotation.
    
[^83]: 谱平滑与尖峰扰动重构大语言模型的能力画像与选择性任务表现

    Spectral Smoothing and Spiking Perturbations Reconfigure Capability Profiles and Selective Task Performance in Large Language Models

    [https://arxiv.org/abs/2610.30676](https://arxiv.org/abs/2610.30676)

    该论文通过对Transformer前馈投影矩阵的奇异值谱进行平滑与尖峰化干预（保持奇异向量和几何均值不变），以三维能力基（事实知识、语言理解、演绎推理）为解释框架，系统揭示并重构了大语言模型的能力画像及其在各类任务上的选择性表现变化。

    

    奇异值谱编码了Transformer前馈投影中的重要结构，但其在下游行为中的因果作用仍鲜有探索。我们研究了针对up_proj和down_proj矩阵的事后谱平滑与尖峰干预。正向扰动会拉伸偏离几何均值的程度，使大奇异值更大、小奇异值更小（尖峰化），反之亦然。该干预保留了奇异向量以及奇异值的几何均值。我们在五个开源的Llama和Qwen指令模型上对扰动后的系统进行评估，涵盖能力标注子任务、标准推理与思维链推理基准、由评审打分的真实性、幻觉与安全任务，以及TRAIT人格探测。为了解释分数的变化，我们在由事实知识、语言理解和演绎推理构成的三维能力基中对模型响应进行拟合。所引发的变化……

    arXiv:2610.30676v1 Announce Type: new  Abstract: Singular-value spectra encode important structure in transformer feed-forward projections, yet their causal role in downstream behavior remains underexplored. We study post-hoc spectral smoothing and spiking interventions on the up_proj and down_proj matrices. Positive perturbation stretches deviations from the geometric mean, making large singular values larger and small ones smaller (spiking), and vice versa. The intervention preserves singular vectors and the geometric mean of the singular values. Across five open Llama and Qwen instruction models, we evaluate the perturbed systems on capability-labeled subtasks, standard and chain-of-thought reasoning benchmarks, judge-scored truthfulness, hallucination, and safety tasks, and TRAIT personality probes. To interpret score shifts, we fit responses in a three-dimensional capability basis consisting of factual knowledge, language understanding, and deductive reasoning. The induced changes
    
[^84]: 将自回归LLM控制编译为面向分层RAG的策略精确并行推理

    Compiling Autoregressive LLM Control into Policy-Exact Parallel Inference for Hierarchical RAG

    [https://arxiv.org/abs/2610.30664](https://arxiv.org/abs/2610.30664)

    该论文提出将分层RAG中自回归LLM的控制决策编译为策略精确的并行推理——只需一次根预填充加一次树因果探测，即可在两次前向计算的关键路径、零输出token的条件下，精确复现串行语法约束解码的动作分布。

    

    大型语言模型越来越多地在结构化决策流水线中充当控制器，但其决策仍然通过自回归文本解码来执行。分层RAG（Hierarchical RAG）正是这样一种场景：在文档树的每个前沿节点，LLM需要在依赖输入的动作语法约束下，从k个章节中选定b个进行提交，其执行方式要么是容易出错的自由格式JSON，要么是可靠但严格串行的语法约束生成。我们证明，对于有界视界控制而言，这种串行化是不必要的：可达的未来形成一个浅层的动作前缀树，我们将其编译——仅需一次根预填充加一次树因果探测，即可利用串行解码所对应的token、位置和因果祖先，评估每个前缀条件下的动作分布。一个前缀状态引理保证编译后的控制在精确算术意义下达到策略精确：即相同的策略，但关键路径仅包含两次前向计算且零输出token。再经过一次探测过程即可验证精确的序边际化……（摘要在此处被截断）

    arXiv:2610.30664v1 Announce Type: new  Abstract: Large language models increasingly act as controllers in structured decision pipelines, yet their decisions are still executed by autoregressive text decoding. Hierarchical RAG is one such setting: at each frontier of a document tree, an LLM commits to $b$ of $k$ sections under an input-dependent action grammar, executed as either failure-prone free-form JSON or reliable but strictly serial grammar-constrained generation. We show that this serialization is unnecessary for bounded-horizon control: the reachable future forms a shallow action-prefix tree, which we compile --- one root prefill plus one tree-causal probe evaluate every prefix-conditioned action distribution with serial decoding's tokens, positions, and causal ancestry. A prefix-state lemma makes compiled control policy-exact in exact arithmetic: the same policy with a two-forward critical path and zero output tokens. One further probe pass certifies the exact order-marginaliz
    
[^85]: 组相对策略优化提升低资源语言的多语言语音识别

    Group Relative Policy Optimisation Improves Multilingual Speech Recognition in Low-Resource Languages

    [https://arxiv.org/abs/2610.30663](https://arxiv.org/abs/2610.30663)

    将组相对策略优化（GRPO）强化学习应用于非LLM的Whisper编码器-解码器模型，训练出的SunflowerASR在51种非洲低资源语言上取得已发表的最佳词错误率，尤其在训练数据稀缺或音频嘈杂的场景下优势最为明显。

    

    采用教师强迫（teacher forcing）训练的自回归编码器-解码器语音识别模型可能存在曝光偏差（exposure bias），限制了模型在推理时从自身预测错误中恢复的能力。近期研究表明，使用组相对策略优化（GRPO）的强化学习能够提升基于大语言模型（LLM）的模型在主要语言上的语音识别性能。我们将这一研究方向扩展到更小的、非LLM的编码器-解码器ASR模型，以Whisper-large-v3为案例，覆盖51种非洲语言。当训练数据稀缺或输入音频嘈杂、质量退化时，该方法的优势最为显著。我们得到的模型SunflowerASR在AfriVox-v2（0.227）和SimbaBench（0.372）基准测试上取得了已发表的最佳宏平均词错误率（WER），并已公开发布。

    arXiv:2610.30663v1 Announce Type: cross  Abstract: Autoregressive encoder-decoder speech recognition models trained with teacher forcing can suffer from exposure bias, limiting their ability to recover from their own prediction errors at inference time. Recent work has shown that reinforcement learning with Group Relative Policy Optimisation (GRPO) improves speech recognition in major languages with LLM-based models. We extend this line of work to smaller, non-LLM encoder-decoder ASR models, using Whisper-large-v3 as a case study across 51 African languages. The benefits of this approach are most pronounced when training data is scarce or when input audio is noisy or degraded. Our resulting model, SunflowerASR, achieves the best published macro-averaged WER on the AfriVox-v2 (0.227) and SimbaBench (0.372) benchmarks, and is publicly available.
    
[^86]: 规范未言明之事：基于测试评分的编程智能体基准中的规范-预言机差距

    What the Specification Did Not Say: Specification-Oracle Gaps in Test-Graded Coding-Agent Benchmarks

    [https://arxiv.org/abs/2610.30635](https://arxiv.org/abs/2610.30635)

    该论文发现编程智能体基准测试中存在“规范-预言机差距”——评分测试依赖任务描述中未说明的实现细节（如常量名、模块路径），导致这类任务的智能体失败率高达80.6%（无此差距时仅48.4%），并提出了覆盖五种语言的形式化静态审计方法来检测此类差距。

    

    编程智能体基准测试通常用自然语言描述任务，但使用来自上游项目的测试来评分。通过这些测试可能需要描述中从未说明的信息：一个常量的名称、一个模块路径、一条确切的消息。我们将其称为规范-预言机差距：当这种差距存在时，基准测试实际上还在衡量智能体是否能复现上游的实现细节。我们对这一差距进行了形式化定义，包括依赖任务链所引入的不一致性，并构建了一个针对五种语言的静态审计工具，对每个被评分测试的名称级、路径级和字面量级依赖进行分类（留出集精确率89.2%，召回率92.9%）。在SWE-Milestone（98个评分里程碑，35次已发布的运行）上，25个里程碑包含依赖于未说明且不可推断的名称或路径的测试。具有此类依赖的构建单元在80.6%的运行中失败，而没有此类依赖的可比单元失败率为48.4%，且隐藏单个名称便会使……

    arXiv:2610.30635v1 Announce Type: new  Abstract: Coding-agent benchmarks often describe a task in natural language but grade it with tests taken from the upstream project. Passing those tests can require information the description never states: a constant's name, a module path, an exact message. We call this a specification-oracle gap: where it exists, a benchmark also measures whether an agent reproduces upstream implementation details. We formalise the gap, including the inconsistencies that chains of dependent tasks add, and build a static audit for five languages that classifies every name-, path- and literal-level dependency of every scored test (held-out precision 89.2%, recall 92.9%). On SWE-Milestone (98 graded milestones, 35 published runs), 25 milestones contain a test that depends on an unstated, non-inferable name or path. Build units with such a dependency fail in 80.6% of runs, against 48.4% for comparable units without one, and withholding a single hidden name from the 
    
[^87]: 融合税：混合检索何时造成损害的诊断性分析

    The Fusion Tax: A Diagnostic Analysis of When Hybrid Retrieval Hurts

    [https://arxiv.org/abs/2610.30612](https://arxiv.org/abs/2610.30612)

    该研究通过诊断分析发现，混合检索（如RRF融合）虽然在总体指标上优于纯稠密检索，但在8-48%的个体查询上反而损害检索效果，因此“不融合”应被视为值得认真对待的一等选项。

    

    混合检索——即将稀疏词法检索器与稠密神经检索器相融合——在2026年9月调研的软件文档中被广泛推荐，通常采用倒数排序融合（RRF），其依据是总体性能上的提升。我们研究融合在单个查询上付出了什么代价，并将“不融合”作为一个一等公民级别的结果来对待。实验设置正是混合检索实际部署的典型场景：标准的Anserini BM25、五个跨越2200万至5.96亿参数且涵盖三代模型的稠密编码器、一个学习型稀疏（SPLADE）替代方案，以及交叉编码器第二阶段重排，在七个BEIR数据集和三种融合函数上进行评估。即使是最好的方法——稠密加权分数插值——在13-48%的查询上，其将相关文档的排名低于两个组件中较好的那个，在8-21%的查询上低于可直接部署的纯稠密检索器（在留一语料库验证下选择权重时，这两个比例分别为14-48%和11-25%），同时在每个语料库的总体指标上仍优于纯稠密检索。

    arXiv:2610.30612v1 Announce Type: cross  Abstract: Hybrid retrieval, which fuses a sparse lexical retriever with a dense neural one, is widely recommended in the software documentation surveyed in September 2026, usually with reciprocal rank fusion (RRF), on the strength of aggregate gains. We ask what fusion costs on individual queries, treating "do not fuse" as a first-class outcome, in the setting in which hybrid retrieval is deployed: a standard Anserini BM25, five dense encoders spanning 22M-596M parameters and three model generations, a learned-sparse (SPLADE) alternative, and a cross-encoder second stage, on seven BEIR datasets and three fusion functions. Even the best method, dense-weighted score interpolation, ranks relevant documents below the better of its two components on 13-48% of queries and below the deployable dense-only retriever on 8-21% (14-48% and 11-25% at a weight selected under leave-one-corpus-out validation), while beating dense-only in aggregate on every corp
    
[^88]: TT-SAX：一种用于自然语言形状查询的双轨时间序列符号化方法

    TT-SAX: A Two-Track Time Series Symbolization for Natural Language Shape Querying

    [https://arxiv.org/abs/2610.30593](https://arxiv.org/abs/2610.30593)

    该论文通过分析六种经典符号化方法在自然语言时间序列查询中的三大局限并提出三条设计原则，据此对SAX进行最小化改进，提出双轨符号化方法TT-SAX，使自然语言形状查询能更好地捕捉时间序列的方向与变化趋势。

    

    面向时间序列数据库的自然语言查询旨在让用户通过自然语言提问，从漫长的历史数据中找到所需内容。形状查询关注序列是如何变化的，这是该任务的主要挑战。近期的研究通过将符号化作为查询与漫长历史数据之间的桥梁，使形状查询变得可行。然而，现有的符号化方法是为压缩、相似性索引或分类而设计的，目前尚缺乏系统性研究来探讨符号化应为自然语言查询提供什么。因此，我们针对该任务检验了六种经典符号化方法的定义，并识别出三个局限性：(1) 字符串中缺失方向信息；(2) 方向的度量尺度过小；(3) 字母符号是任意的。由此得出三条设计原则。我们遵循这些原则，对SAX进行了最小化的重新设计，提出了TT-SAX（双轨SAX）。水平（幅值）字母保持不变，每个区段……

    arXiv:2610.30593v1 Announce Type: cross  Abstract: Natural language querying for time series databases aims to let users find what they need in long histories of data by asking in plain language. Shape queries ask how a series moves. They are the main challenge of this task. Recent work makes them feasible by using symbolization as a bridge between the query and the long history. However, existing symbolizations were designed for compression, similarity indexing, or classification. What a symbolization should provide for natural language querying has not been systematically studied. We therefore examine the definitions of six classical symbolizations against this task and identify three limitations. (1) Direction is missing from the string, (2) direction is measured at too small a scale, and (3) the letters are arbitrary. Three design principles follow. We follow these principles and make a minimal redesign of SAX called TT-SAX (two-track SAX). The level letters stay. Each segment gain
    
[^89]: 面向自主投资的受治理自我进化：一个具有不变核心的多智能体LLM系统

    Governed Self-Evolution for Autonomous Investing: A Multi-Agent LLM System with an Invariant Core

    [https://arxiv.org/abs/2610.30557](https://arxiv.org/abs/2610.30557)

    该论文提出了Steward——一个面向自主投资的多智能体LLM系统，通过九个角色专业化智能体协同决策、七层共享记忆，以及以类型系统划定自我进化边界的“不变核心”，在行动不可逆且数据不可信的投资场景中实现受治理的安全自我进化。

    

    作用于实时网络数据的LLM智能体必须整合异构证据、记住自己已完成的工作，并从结果中持续改进，同时不削弱自身的安全防护。我们在自主投资领域研究这一问题——在这一场景中，每个行动都不可逆转，且基于不可信的网络内容。我们提出了Steward，一个在其类型系统中划定自我进化边界的多智能体LLM系统。首先，九个角色专门化的智能体共同决策：四名分析师分别阅读新闻资讯、SEC文件、宏观经济序列和市场数据，并输出类型化报告，投资组合经理据此设定目标头寸，再由受保护的风险经理进行审计。其次，一个由代码根据系统自身记录组装而成的七层共享记忆，为每个决策提供头寸上下文、市场叙事、校准信息以及智能体的过往决策。第三，系统通过复盘实现自我改进：代码每晚对每笔交易和预测进行评分，并将评分写入……

    arXiv:2610.30557v1 Announce Type: cross  Abstract: LLM agents that act on live Web data must combine heterogeneous evidence, remember what they have done and improve from outcomes without weakening their own safeguards. We study this problem in autonomous investing, where every action is irreversible and based on untrusted Web content. We present Steward, a multi-agent LLM system that draws the boundary of self-evolution in its type system. First, nine role-specialised agents decide jointly: four analysts read news feeds, SEC filings, macroeconomic series and market data and emit typed reports, from which a Portfolio Manager sets target positions that a protected Risk Manager audits. Second, a seven-layer shared memory, assembled by code from the system's own record, gives each decision its position context, the market narrative, calibration and the agent's past decisions. Third, the system improves itself by review: code grades every trade and forecast nightly and writes the grades in
    
[^90]: IFC-MCQ：自动生成多项选择题基准，用于评估大语言模型智能体的IFC数据检索能力

    IFC-MCQ: Automatically Generating Multiple-Choice Benchmarks for Evaluating LLM Agents on IFC Data Retrieval

    [https://arxiv.org/abs/2610.30533](https://arxiv.org/abs/2610.30533)

    提出IFC-MCQ方法，从IFC文件自动生成多项选择题基准并实现精确自动评分，从而以低成本大规模评估LLM智能体在IFC数据检索上的能力。

    

    行业基础类（IFC）是建筑工程（AEC）领域建筑信息交换的标准，如今越来越多地通过大语言模型（LLM）以自然语言进行查询。由于自由文本答案需要昂贵的人工或基于LLM的审查，大规模评估这些系统十分困难。我们提出IFC-MCQ，一种从IFC文件自动生成多项选择题基准的方法，可实现精确的选项集评分。该分类法涵盖四个组别下的15个类别和36个子类型，其中答案键直接源自文件本身，干扰项则由文件派生的值或预定义规则构建。我们将IFC-MCQ应用于不同规模和建模特征的IFC模型以评估生成覆盖率，并使用基于工具和基于代码的检索方法在选定模型上评估了两种智能体配置。该评估揭示了特定类别的失败模式，并表明性能可能因（原文在此处截断）……

    arXiv:2610.30533v1 Announce Type: new  Abstract: Industry Foundation Classes (IFC), the standard for building information exchange in architecture, engineering, and construction (AEC), is increasingly queried through large language models (LLMs) using natural language. Evaluating these systems at scale is difficult because free-text answers require costly human or LLM-based review. We propose IFC-MCQ, a method that automatically generates multiple-choice benchmarks from IFC files for exact option-set grading. The taxonomy covers 15 categories and 36 subtypes across four groups, with answer keys derived from the file and distractors constructed from file-derived values or predefined rules. We applied IFC-MCQ to IFC models of varying sizes and modelling characteristics to assess generation coverage and evaluated two agentic configurations on selected models using tool- and code-based retrieval. The evaluation exposed category-specific failure modes and showed that performance can vary wi
    
[^91]: Protegrity-PII 技术报告

    Protegrity-PII Technical Report

    [https://arxiv.org/abs/2610.30518](https://arxiv.org/abs/2610.30518)

    提出了新型 PII 决策模型 Protegrity-PII，在自建基准 PII-SCOPE 上实现 0.915 召回率和 0.907 精确率，兼具高速低成本推理与卓越的分布外泛化能力。

    

    我们提出了 Protegrity-PII，一种面向结构化和非结构化文本的新型 PII（个人身份信息）决策模型。在我们提出的新基准 PII-SCOPE 上，Protegrity-PII 实现了 0.915 的类型化召回率和 0.907 的精确率。借助批量推理，该模型在单个 RTX 6000 上以每百万字符 0.49 秒的速度处理文本，成本仅为每百万字符 0.0005 美元。部分得益于训练中使用的实体类型分类体系的广泛性，与一系列最先进的模型和数据集相比，Protegrity-PII 展现出卓越的分布外泛化能力。

    arXiv:2610.30518v1 Announce Type: new  Abstract: We present Protegrity-PII, a new PII decision model for structured and unstructured text. Protegrity-PII achieves a typed recall of 0.915 with a precision of 0.907 on a new benchmark we propose, PII-SCOPE. Our model processes text in 0.49 seconds per million characters at \$0.0005 per million characters on a single RTX 6000 using batched inference. Due in part to the expansive nature of the entity type taxonomy used in training, Protegrity-PII achieves superior out-of-distribution generalization when benchmarked against a selection of state of the art models and datasets.
    
[^92]: 超越端到端恢复：通过角色分解更好地理解大语言模型隐写术

    Beyond End-to-End Recovery: Towards a Better Understanding of LLM Steganography via Rolewise Decomposition

    [https://arxiv.org/abs/2610.30516](https://arxiv.org/abs/2610.30516)

    该论文提出了一个按角色分解的评估框架，将大语言模型隐写术的评估从单一的端到端恢复拆解为编码表示、解码恢复和确定性验证三个独立环节，从而能够区分编码失败与解码失败，并揭示私密信息在传输文本中的隐藏表示方式。

    

    在大语言模型（LLM）隐写术中，编码器将一条私密信息隐藏在看似普通的回复中，并将文本传输给解码器，由解码器尝试恢复该信息。这种私密信息传输的成功通常通过端到端（E2E）恢复来衡量，即检查解码器是否正确重构了私密信息。然而，端到端恢复无法区分编码失败与解码失败，也无法揭示私密信息在编码器传输文本中的表示方式。为了解决这些局限性，我们引入了一个按角色分解的框架，分别考察私密信息在编码器传输文本中的表示方式、解码器实际恢复出的内容，以及确定性验证能够确立什么。在7,200个编码器回复中，我们识别出168个被解码器遗漏的可验证编码，以及706个包含通过词长或词元初始字母等方式重构私密信息的回复。

    arXiv:2610.30516v1 Announce Type: cross  Abstract: In large language model (LLM) steganography, an encoder hides a private message in an otherwise ordinary response, as text transmitted to a decoder, who attempts to recover it. The success of this private-message transmission is commonly measured through end-to-end (E2E) recovery, which checks whether the decoder correctly reconstructs the private message. However, E2E recovery cannot distinguish encoding failures from decoding failures or reveal how the private message is represented in the encoder's transmitted text. To address these limitations, we introduce a rolewise decomposition framework that separately examines how private messages are represented in encoder transmitted texts, what decoders recover, and what deterministic verification can establish. Across 7,200 encoder responses, we identify 168 verifiable encodings missed by decoders and 706 responses containing private messages reconstructed through word-length or token-ini
    
[^93]: 总分背后：系统性改变Arena-Hard的LLM评判者所揭示的问题

    Behind Aggregate Scores: What Systematically Varying Arena-Hard's LLM Judge Reveals

    [https://arxiv.org/abs/2610.30512](https://arxiv.org/abs/2610.30512)

    该研究通过系统性地改变Arena-Hard评测中LLM评判者的标准、输出格式与任务指令，发现相似的总分掩盖了大量个别评判结果的差异——即使在温度为零且固定模型版本的条件下重复运行，仍有21.2%的提示词的胜/平/负结果会改变，表明LLM评测的可靠性存在被忽视的隐患。

    

    LLM评判者（judge）被用来判定两个答案中哪一个更好。信任这些裁决需要了解它们的可重复性如何，以及它们在多大程度上依赖于评判者的指令。我们表明，相似的总分可能掩盖单个裁决中的实质性变化，无论是在重复相同设置时，还是在指令发生改变时。我们使用500个Arena-Hard v2.0提示词并固定两个模型的答案，让GPT-4.1作为评判者，在评估标准、输出格式和任务框架的所有八种组合下运行，并在固定的模型版本上以温度为零的设置原样重复每种设置。重新运行默认设置会使总分变动约一分，但在21.2%的提示词上改变了胜/平/负的结果。指令的改变引入了进一步的敏感性：在一个要求直接给出裁决而无需推理、并删除了其他几项指令的设置中，结果发生变化的提示词比例比……（原文截断）多出28.2个百分点。

    arXiv:2610.30512v1 Announce Type: new  Abstract: LLM judges are used to determine which of two answers is better. Trusting these verdicts requires knowing how repeatable they are and how much they depend on the judge's instructions. We show that similar aggregate scores can conceal substantial changes in individual verdicts, both when the same setup is repeated and when the instructions change. Using 500 Arena-Hard v2.0 prompts with two models' answers held fixed, we run GPT-4.1 as the judge under all eight combinations of criteria, output format and task framing, and repeat every setup unchanged at temperature zero on a pinned model version. Rerunning the default setup moves the aggregate score by about one point but changes the win/tie/loss outcome on 21.2% of prompts. Instruction changes introduce further sensitivity: in one setup, which asks for the verdict without reasoning and also drops several other instructions, outcomes change on 28.2 percentage points more prompts than the t
    
[^94]: 重新思考LLM事实核查器的评估

    Rethinking the Evaluation of LLM Fact-Checkers

    [https://arxiv.org/abs/2610.30504](https://arxiv.org/abs/2610.30504)

    提出了首个同时明确声明-问题关联、判定条件和证据来源三项设计决策的日语事实核查基准JSFactCheckBench，揭示了总体分数掩盖了模型核查直接回答问题的声明时准确率显著更低的现象。

    

    大语言模型（LLM）被广泛用于回答事实性问题，但其答案可能包含错误，必须逐条声明进行事实核查。然而，由此产生的分数取决于三个设计决策：声明回答的是哪些问题、每条声明在什么条件下被判定、以及哪些来源算作证据。目前没有任何现有基准同时明确规定了这三点。我们推出了JSFactCheckBench，这是一个明确了这三点的日语基准，包含1,913条声明，这些声明来自LLM对曾在日本社交媒体上流传的误导性信息相关问题的回答，每条声明均在参考时间点标注了事实性，并标注了与问题的相关重要性，使用预先选定的网络来源作为证据。在六个LLM的测试中，所有模型对直接回答问题的声明的核查准确率均低于辅助性声明，而这一差距会被总体分数所掩盖。当模型必须在更长或多个来源中定位相关内容时，分数也会下降，并且对所有……

    arXiv:2610.30504v1 Announce Type: new  Abstract: Large language models (LLMs) are widely used to answer factual questions, but their answers can contain errors and must be fact-checked claim by claim. Yet the resulting score depends on three design decisions: which questions the claims answer, under what conditions each claim is judged, and which sources count as evidence. No existing benchmark specifies all three. We introduce JSFactCheckBench, a Japanese benchmark that specifies all three, with 1,913 claims from LLM answers to questions about misleading information that circulated on Japanese social media, each annotated for factuality at a reference time and for importance to the question, using evidence from preselected Web sources. Across six LLMs, all verify claims directly answering the question less accurately than supplementary claims, a gap that aggregate scores obscure. Scores also drop when models must locate the relevant content in longer or multiple sources and, for all f
    
[^95]: 为什么要再次打扰我？面向交互偏好的个人智能体记忆基准测试

    Why Bother Me Again? Benchmarking Personal Agent Memory for Interaction Preferences

    [https://arxiv.org/abs/2610.30485](https://arxiv.org/abs/2610.30485)

    提出了MemPABench——一个评估个人智能体记忆与落实用户交互偏好能力的在线交互式基准，通过四个月的模拟用户会话和动态偏好变化，测试智能体何时澄清、确认或自主行动。

    

    个人智能系统正在承担日益复杂的日常任务，然而重复的提问、不必要的确认和不想要的主动打扰，会把强大的辅助能力变成交互负担。要减少这种负担，需要智能体记住用户希望被协助的方式。然而，现有的记忆基准主要关注事实回忆和个性化回答，很少测试智能体如何学习并落实交互偏好。我们提出了MemPABench，一个用于个人智能体记忆在线评估的交互式环境。MemPABench设置了需要智能体决定何时澄清、何时确认、何时自主行动的长期任务。四个基于真实场景的模拟用户在工作和个人场景中持有14项偏好。他们的反馈横跨四个月内的99个会话，且每个用户有五个由事件驱动的偏好变化，用于测试记忆的更新能力。我们评估了偏好选择、实际执行行为，以及对上下文和偏好的敏感度。

    arXiv:2610.30485v1 Announce Type: cross  Abstract: Personal intelligence systems are taking on increasingly complex everyday tasks, yet repeated questions, unnecessary confirmations, and unwanted outreach can turn capable assistance into interaction burden. Reducing this burden requires agents to remember how users want to be assisted. However, existing memory benchmarks focus on factual recall and personalized answers and rarely test how agents learn and act on interaction preferences. We introduce MemPABench, an interactive environment for on-policy evaluation of memory in personal agents. MemPABench poses long-horizon tasks requiring agents to decide when to clarify, confirm, or act autonomously. Four grounded simulated users hold 14 preferences across work and personal contexts. Their feedback spans 99 sessions over four months, with five event-driven changes per user testing memory updates. We evaluate preference selection, delivered behavior, and sensitivity to context and prefer
    
[^96]: 重新思考SAE评估：原子可解释单元框架揭示隐藏的多义性与冗余性

    Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy

    [https://arxiv.org/abs/2610.30474](https://arxiv.org/abs/2610.30474)

    本文提出了一种基于原子可解释单元的SAE评估新框架，将超越单义性的评估目标形式化，并借此揭示了现有SAE特征中隐藏的多义性与冗余性。

    

    稀疏自编码器（SAE）最初是为发现单义特征而提出的，单义性是刻画单个特征的性质，但并未规定学习到的特征集合整体上应该是什么样子。随着SAE越来越多地被应用于归因分析、回路发现和干预，研究者们开始使用描述特征集合全局性质的评估指标，例如稀疏性、死特征比例、重建保真度和解耦性等，这一系列指标已远远超出单义性的范畴。然而，这种超越单义性的隐式转变尚未被形式化为一个清晰、明确定义的、先于这些评估指标存在的目标。缺少这样的目标，现有指标仍然是临时拼凑的，其必要性缺乏依据。在这项工作中，我们通过为SAE评估引入一种新范式来形式化这一最终目标：要求SAE特征，以及更广义的字典……（摘要不完整）

    arXiv:2610.30474v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) were originally motivated by discovering monosemantic features, a property that characterizes individual features but does not specify what the learned feature set should look like as a whole. As SAEs have increasingly been adopted for attribution, circuit discovery, and intervention, researchers have begun evaluating metrics that describe global properties of the feature set, such as sparsity, dead feature rates, reconstruction fidelity, and disentanglement, a diverse set that extends far beyond monosemanticity. However, this implicit shift beyond monosemanticity has not been formalized as a clear, well-defined objective that comes before these evaluation metrics. Without such an objective, existing metrics remain ad hoc, with their necessity unjustified. In this work, we formalize this ultimate objective by introducing a new paradigm for SAE evaluation: requiring that SAE features, and more generally dictio
    
[^97]: 检索增强大语言模型中搜索状态感知的自评估强化学习

    Self-Assessment Reinforcement Learning for Search-State Awareness in Retrieval-Augmented Large Language Models

    [https://arxiv.org/abs/2610.30435](https://arxiv.org/abs/2610.30435)

    提出SARL框架，通过兼顾评估纠正效果与惩罚过度依赖的双重奖励设计，让模型在训练中学会感知自身搜索状态，无需测试时额外评估即可实现更有效且高效的搜索，相比基线获得24.3%的相对提升。

    

    近年来搜索智能体的进展使大语言模型能够通过检索和综合外部信息来完成复杂任务。然而，现有搜索智能体往往缺乏搜索状态感知能力，即识别已经获取了哪些信息以及仍然缺少哪些信息的能力，这会导致幻觉式推理和过早终止搜索。为了解决这一局限，我们提出了自评估强化学习（SARL），这是一种强化学习框架，模型在训练过程中学习识别自身的搜索状态并据此改进搜索行为，而无需在测试时进行额外评估。SARL通过双重奖励设计来实现这一点：该设计既捕捉每次评估带来的纠正效果，又惩罚对评估的过度依赖，从而共同鼓励有效且高效的搜索轨迹。在此设置下，SARL相对基线实现了24.3%的相对提升（摘要原文在此处被截断）。

    arXiv:2610.30435v1 Announce Type: new  Abstract: Recent advances in search agents have enabled large language models to tackle complex tasks by retrieving and synthesizing external information. However, existing search agents often lack search-state awareness, i.e., the ability to recognize what information has been obtained and what remains missing, leading to hallucinated reasoning and prematurely terminated search. To address this limitation, we propose Self-Assessment Reinforcement Learning (SARL), a reinforcement learning framework in which models learn to recognize their own search state during training and improve their search behavior accordingly, without requiring additional assessment at test time. SARL achieves this through a dual reward design that captures the corrective effect of each assessment while penalizing excessive reliance on it, jointly encouraging effective and efficient search trajectories. Under this setup, SARL achieves a 24.3% relative improvement over the s
    
[^98]: 智能体马赛克：识别基于大语言模型的智能体系统中的机密性泄露

    Agent Mosaic: Recognizing Confidentiality Leaks in Agentic LLM Systems

    [https://arxiv.org/abs/2610.30418](https://arxiv.org/abs/2610.30418)

    该论文首次量化了LLM智能体中的“马赛克效应”风险——智能体即使无法访问任何单独的机密文档，仍会将多份可访问文档的信息组合成敏感结论并造成机密泄露，且实验表明现有的密级标注和聚合规则等防护措施仍无法完全阻止此类不安全泄露。

    

    现代大语言模型（LLM）智能体可以从可访问的文档中检索信息，然后披露由这些信息组合而成的敏感结论。这种“马赛克效应”对处理机密数据的智能体构成了严重风险。我们通过30个合成的网络安全、医疗和军事案例来量化这种风险的实现情况，在受控模拟中从三个LLM智能体获得了10,800个最终答案，并在两个开源测试框架中从真实运行的智能体获得了1,440个答案。即使在被设计为仅凭单个文档即可回答的良性查询下，表现最佳的模型GPT-5.6-Sol在模拟中仍有7.08%的答案泄露了禁止的组合信息（95%置信区间：3.75-10.67%），而在真实测试框架中的泄露率更高。添加可见的密级标题和明确的聚合规则可以大幅降低总体泄露率，但尽管有这些安全防护措施，在所有查询意图下仍有32.37%的模拟答案（95%置信区间：28.33-36.19%）存在不安全泄露。

    arXiv:2610.30418v1 Announce Type: cross  Abstract: Modern LLM agents can retrieve information from accessible documents, and then disclose a sensitive conclusion assembled from them. This mosaic effect creates a critical risk for agents operating over confidential data. We quantify its realization with 30 synthetic cybersecurity, medical, and military examples, yielding 10,800 final answers from three LLM agents in controlled simulation and 1,440 answers from live agents in two open-source harnesses. Even under benign queries designed to be answerable from one document, GPT-5.6-Sol, the best performing model tested, discloses forbidden composites in 7.08% of answers (95% CI: 3.75-10.67%) in simulation, with worse disclosure rates in live harnesses. Adding visible classification headers and explicit aggregation rules reduces pooled disclosure rates substantially, yet unsafe disclosures persist in 32.37% of simulated answers (95% CI: 28.33-36.19%) across all intents despite these safegua
    
[^99]: 「X」诉衷肠：Twitter收购后的心理健康披露

    X-pressing Intimacy: Mental Health Disclosure After Twitter Acquisition

    [https://arxiv.org/abs/2610.30413](https://arxiv.org/abs/2610.30413)

    本文利用LIWC与基于大语言模型的亲密度分类器分析约82.4万条推文，揭示了马斯克收购Twitter后用户心理健康自我披露的变化：情感与社会性语言减少、第一人称单数及未来导向表达增加，且披露亲密度的变化模式因病症而异。

    

    社交媒体平台为个人提供了披露心理健康经历并寻求支持的空间，但这些披露在平台治理发生重大变动时会如何演变仍不清楚。我们围绕埃隆·马斯克2022年10月收购Twitter这一事件，考察了心理健康自我披露情况，分析了涵盖焦虑、抑郁症、双相情感障碍（BD）、边缘型人格障碍（BPD）以及自杀意念（SI）或相关经历的约82.4万条推文，并同时分析了26.6万条对照推文。借助语言查询与字词计数工具（LIWC）以及基于大语言模型（LLM）的亲密度分类器，我们刻画了心理语言学表达与披露亲密度的变化。收购之后，我们观察到情感性和社会性语言有所减少，同时第一人称单数代词和面向未来的表达有所增加。亲密度模式因具体病症而异：焦虑、抑郁和BPD呈现出低亲密度……（原文摘要至此截断）

    arXiv:2610.30413v1 Announce Type: cross  Abstract: Social media platforms provide spaces for individuals to disclose mental health experiences and seek support, yet how these disclosures evolve during major shifts in platform governance remains unclear. We examine mental health self-disclosure surrounding Elon Musk's October 2022 acquisition of Twitter, analyzing approximately 824K tweets across anxiety, depression, bipolar disorder (BD), borderline personality disorder (BPD), and suicidal ideation (SI) or related experiences, alongside 266K comparison tweets. Using Linguistic Inquiry and Word Count (LIWC) and an LLM-based intimacy classifier, we characterize changes in psycholinguistic expression and disclosure intimacy. Following the acquisition, we observe declines in affective and social language, alongside increases in first-person singular pronouns and future-focused expressions. Intimacy patterns vary by condition: anxiety, depression, and BPD show increases in low-intimacy and 
    
[^100]: SIM-UT：一种基于用户模拟的医疗分诊聊天机器人评估框架

    SIM-UT: A User Simulation based Evaluation Framework for Medical Triage Chatbots

    [https://arxiv.org/abs/2610.30412](https://arxiv.org/abs/2610.30412)

    提出SIM-UT，一种多轮用户模拟评估方法，能发现基于LLM的医疗分诊聊天机器人在真实多轮、多语言用户交互情境下的失效模式，并在印度、南非和肯尼亚的三个真实WhatsApp健康聊天机器人上进行了验证。

    

    基于大语言模型（LLM）的聊天智能体越来越多地被用于中低收入国家（LMICs）中复杂的、多轮的、多语言的医疗分诊，然而现有的评估方法主要集中于单轮、仅限英语的基准测试，这些测试主要考察聊天机器人的知识水平。在实际应用中，用户常常感到困惑、不知所措或情绪脆弱，会给出不清晰的、拼写错误的单词式回答，或用地方方言发送长篇消息，且对话的上下文可能迅速变化。我们提出了SIM-UT（面向分诊的用户行为模拟），这是一种多轮的、基于模拟的评估方法，能够识别基于LLM的分诊聊天智能体在生态学上有效、符合真实情境的失效模式。利用SIM-UT，我们评估了三个基于WhatsApp的真实健康聊天机器人：印度的呼吸系统筛查项目、南非的听力健康支持项目，以及肯尼亚的糖尿病与生活方式辅导智能体。我们比较了经过医学微调的（原文在此处截断）

    arXiv:2610.30412v1 Announce Type: new  Abstract: LLM-based chat agents are increasingly used for complex, multi-turn, multilingual medical triage in low- and middle-income countries (LMICs), yet evaluation methods focus on single-turn, English-only benchmarks that primarily test the chatbots' knowledge. In practice, users are often confused, overwhelmed, or emotionally vulnerable, giving unclear, misspelled single-word answers or long messages in regional dialects, and the context of a conversation can shift rapidly. We propose SIM-UT (Simulation of User Behaviour for Triage), a multi-turn, simulation-based evaluation method that identifies ecologically valid, contextually relevant failure modes of LLM-based triage chat agents. Using SIM-UT, we evaluated three real-world WhatsApp-based health chatbots: a respiratory screening programme in India, a hearing health support programme in South Africa, and a diabetes and lifestyle coaching agent in Kenya. We compared a medically fine-tuned o
    
[^101]: 纯文本LLM纠错器会破坏正确的转录文本：ASR纠错中音频信息锚定的修复-损伤分析

    Text-Only LLM Correctors Break Correct Transcripts: A Repair-Damage Analysis of Audio Grounding in ASR Error Correction

    [https://arxiv.org/abs/2610.30385](https://arxiv.org/abs/2610.30385)

    该论文提出将WER变化精确分解为修复、损伤与插入的修复-损伤分析框架，首次系统量化揭示了纯文本LLM纠错会破坏13-32%本已正确的转录文本，而引入音频信息后破坏率降至1-3%，证明音频锚定对ASR纠错具有关键的保护作用。

    

    对ASR输出进行纠错的LLM纠错器通常以总体WER来评价，这掩盖了它们修复了什么、又破坏了什么。我们将每次WER变化精确分解为修复的词元、损伤的词元和插入，并利用这一分解探究源音频为纠错器带来了什么增益。我们在三个语料库上，分别针对模型自身的草稿和独立识别器的输出，比较了有无音频条件下的纠错表现。使用Gemini 2.5 Flash时，纯文本纠错其自身草稿会破坏13-32%本已正确的转录文本；而加入音频后仅为1-3%。在两种设置下损伤均下降，但对自身草稿的修复下降，而对识别器输出的修复上升。在医疗听写场景中，被保留的插入抵消了净WER的收益。很少进行编辑的开源权重模型Voxtral则未表现出这种模式。由于两个实验分支使用了不同的提示词，这些对比只能界定音频作用的范围，而无法将其单独分离出来。

    arXiv:2610.30385v1 Announce Type: cross  Abstract: LLM correctors of ASR output are judged by aggregate WER, which hides what they fix and what they break. We decompose every WER change exactly into repaired tokens, damaged tokens and insertions, and use this to ask what the source audio adds to a corrector. We compare correction with and without the audio, for the model's own draft and for the output of an independent recogniser, on three corpora. With Gemini 2.5 Flash, text-only correction of its own draft breaks 13-32% of already-correct transcripts; with the audio, 1-3%. Damage falls in both settings. Repair falls against the own draft and rises against the recogniser. On medical dictation, kept insertions cancel the gain in net WER. The open-weight Voxtral, which rarely edits, does not show this pattern. Prompts differ between the two arms, so the contrasts bound the effect of the audio and do not isolate it.
    
[^102]: MemPlan：面向部分可观测文本环境的记忆条件化PDDL规划

    MemPlan: Memory-Conditioned PDDL Planning for Partially Observable Text Environments

    [https://arxiv.org/abs/2610.30379](https://arxiv.org/abs/2610.30379)

    MemPlan提出了一种记忆条件化的PDDL规划框架，利用正向图记忆为候选假设分配基于相关性的动作成本、利用负向记忆将已被反驳的假设编译为排除约束，使LLM作为语言与符号规划之间的受约束桥梁，在部分可观测文本环境中有效减少幻觉和无效动作。

    

    LLM智能体在语言交互方面具有灵活性，但将它们用作长程规划器不仅成本高昂，而且容易产生幻觉或无效动作。本文为LLM提出了另一种角色定位：作为语言与符号规划之间受约束的桥梁。我们提出了MemPlan，一个面向部分可观测交互式文本环境的记忆条件化PDDL规划框架。MemPlan保持PDDL域固定不变，并在每一步根据解析的观测结果、候选假设和交互记忆重建PDDL问题。正向图记忆为缺失对象、位置或条件的候选假设分配基于相关性的动作成本，引导规划器进行更合理的检查验证。负向记忆记录被执行结果反驳的假设，并将其编译为排除约束，防止对已被反馈否定的候选项进行重复尝试。通过一个统一的模式条件化语言-符号接口进行训练……

    arXiv:2610.30379v1 Announce Type: new  Abstract: LLM agents are flexible in language interaction, but using them as long-horizon planners is costly and prone to hallucinated or invalid actions. This paper targets a different role for LLMs: constrained bridges between language and symbolic planning. We propose MemPlan, a memory-conditioned PDDL planning framework for partially observable interactive text environments. MemPlan keeps the PDDL domain fixed and reconstructs the PDDL problem at each step from parsed observations, candidate hypotheses, and interaction memory. Positive graph memory assigns relevance-based action costs to candidate hypotheses for missing objects, locations, or conditions, guiding the planner toward more plausible checks. Negative memory records execution-refuted assumptions and compiles them into exclusion constraints, preventing repeated attempts at candidates already contradicted by feedback. A unified schema-conditioned language--symbol interface, trained of
    
[^103]: 攀登阶梯：低资源环境下的数据流水线自动化与基线复现

    Climbing the Ladder: Automating Data Pipelines and Reproducing Baselines in Low-Resource Settings

    [https://arxiv.org/abs/2610.30374](https://arxiv.org/abs/2610.30374)

    本文在可维护软件上自动化重建了从双语期刊提取西西里语平行语料的数据流水线，并在冻结测试集上系统比较从头训练的小型Transformer与低秩微调的NLLB模型，从而复现并改进了低资源西西里语机器翻译基线。

    

    如今包含西西里语的大规模多语言翻译系统，是从嘈杂、非标准化的网络文本中学习而来的。为了获得更好的翻译质量，我们重新开发了由Wdowiak为西西里语定制的小型神经机器翻译（NMT）系统。我们在可维护的软件基础上重建了从双语期刊Arba Sicula的PDF中提取平行文本的流程：句子通过语言无关的句子编码器进行对齐，对齐阈值根据现有人工对齐结果进行调优，并针对自动对齐的典型错误进行过滤。此外，原系统的分词和基于词尾偏置的子词切分被移植到无Perl的实现中，所有结果均在从该期刊抽取的冻结测试集上进行测量。我们比较了从头训练的小型Transformer与通过低秩微调适配的预训练NLLB模型，在两个翻译方向上进行实验，并以意大利语作为……

    arXiv:2610.30374v1 Announce Type: new  Abstract: The massively multilingual translation systems that now include Sicilian learned it from noisy, non-standardised web text. To obtain better translation quality, we redevelop the small neural machine translation (NMT) system that Wdowiak customized for Sicilian. We rebuild on maintained software the extraction of the parallel text from the PDFs of the bilingual journal Arba Sicula, whose sentences are aligned by a language-agnostic sentence encoder with thresholds tuned against an existing hand alignment and then filtered for the typical errors of automatic alignment. Furthermore, the tokenization and the desinence-biased subword segmentation of the original system are ported to a Perl-free implementation, and all the results are measured on a frozen test set drawn from the journal. We compare small Transformers trained from scratch with the pretrained NLLB model adapted by low-rank fine-tuning, in both directions and with Italian as a cl
    
[^104]: 大语言模型中安全相关行为的探测与引导

    Probing and Steering for Safety-Relevant Behaviours in Large Language Models

    [https://arxiv.org/abs/2610.30363](https://arxiv.org/abs/2610.30363)

    该论文在统一流水线中系统探究了激活探针设计选择（估计器类型、token 池化方案、层选择）在检测与引导欺骗、高风险意识和谄媚三种安全相关行为时的相互作用，并验证了“逻辑回归最擅长检测、均值差方向最擅长引导”的读/写分离现象。

    

    激活探针是在语言模型残差流中计算出的方向，既可用于检测安全相关行为，也可通过将其方向加回残差流来引导这些行为。我们在同一条流水线、同一个模型（Llama-3.1-8B-Instruct）上，针对三种行为（欺骗、高风险意识与谄媚性），探究此类探针背后的主要设计选择之间如何相互作用。该流水线将两种估计器（均值差与逻辑回归）与三种 token 池化方案（末位、均值与可学习的注意力）进行交叉组合，为检测或引导某一行为选择合适的层，并对每个方向进行双重评估：既作为该行为的检测器，也作为将输出推向该行为的引导向量。综合此前的研究可以得出一种“读/写分离”的结论，即逻辑回归在读取（检测）时效果最好，而均值差方向在写入（引导）时效果最好。我们发现这一规律在欺骗行为和（摘要在此处被截断）

    arXiv:2610.30363v1 Announce Type: cross  Abstract: Activation probes are computed directions in a language model's residual stream, used both to detect safety-relevant behaviours and, by adding their direction back into the stream, to steer them. We explore how the main design choices behind such probes interact, within one pipeline, on one model (Llama-3.1-8B-Instruct), and across three behaviours: deception, high-stakes awareness and sycophancy. The pipeline crosses two estimators (difference-of-means and logistic regression) with three token-pooling schemes (last, mean and learned attention), selects appropriate layers to detect or steer a behaviour, and evaluates every direction twice: as a detector of the behaviour, and as a steering vector to push outputs towards the behaviour. Prior work read together suggests a read/write dissociation, in which logistic regression reads (detects) best but difference-of-means directions write (steer) best. We find that it holds for deception and
    
[^105]: CiteAnchor：面向幻觉引用检测与溯源的来源感知定位框架

    CiteAnchor: Source-Aware Grounding for Hallucinated Citation Detection and Resolution

    [https://arxiv.org/abs/2610.30355](https://arxiv.org/abs/2610.30355)

    本文提出了包含423条真实幻觉引用的CiteResolve基准和来源感知定位框架CiteAnchor，通过针对不同来源类型自适应的检索与字段级验证，分别判定引用有效性并恢复唯一来源，在检测与端到端溯源任务上均显著超越基线。

    

    幻觉引用使得追溯来源和验证科学论断背后的证据变得更加困难。检测幻觉引用之所以困难，是因为书目记录可能不完整，或在不同来源类型和出版版本之间存在差异。仅仅一个看似合理的匹配既无法验证引用的有效性，也无法确定其底层来源。我们提出了CiteResolve，这是一个包含423条真实世界幻觉引用的基准数据集，并针对来源可恢复性进行了标注。我们进一步提出CiteAnchor，这是一个来源感知的定位框架，它根据每种来源类型自适应地调整检索和字段级验证，并分别判定引用的有效性以及是否能够恢复出唯一的来源。在现有检测基准上，CiteAnchor使用GPT-5.6 Luna达到0.841的F1分数，使用开放权重骨干模型达到0.784，相比使用相同骨干模型的基线分别提升了7.9分和36.8分。在CiteResolve上，它实现了74.7%和70.7%的端到端任务成功率（原文此处截断）。

    arXiv:2610.30355v1 Announce Type: new  Abstract: Hallucinated citations make it harder to trace sources and verify the evidence behind scientific claims. Detecting them is difficult because bibliographic records can be incomplete or differ across source types and publication versions. A plausible match alone neither validates the citation nor establishes its underlying source. We introduce CiteResolve, a benchmark of 423 real-world hallucinated citations annotated for source recoverability. We further propose CiteAnchor, a source-aware grounding framework that adapts retrieval and field-level verification to each source type and separately determines citation validity and whether a unique source can be recovered. On an existing detection benchmark, CiteAnchor achieves 0.841 F1 with GPT-5.6 Luna and 0.784 with an open-weight backbone, outperforming the baseline with the same backbones by 7.9 and 36.8 points. On CiteResolve, it achieves 74.7% and 70.7% end-to-end task success, outperform
    
[^106]: 一致性并非有效性：自我奖励的道德论证使语言模型变得果断但易受影响

    Consistency Is Not Validity: Self-Rewarded Moral Argumentation Makes Language Models Decisive but Pliable

    [https://arxiv.org/abs/2610.30349](https://arxiv.org/abs/2610.30349)

    该论文提出一种自我奖励的强化学习方法，让模型通过自我评判道德论证的质量来提升道德判断能力，使模型变得更加果断且不受选项呈现顺序的影响，但谄媚性和重新框架问题仍然存在。

    

    大语言模型越来越多地被用于道德判断咨询。我们旨在通过奖励好的论证来提升此类任务的性能，其前提是论证得当能够提高正确判断的可能性。我们从一个最先进的语言模型初始化一个强化学习策略，在一种掩蔽思维的自我评判锦标赛中进行训练：由策略本身对生成的论证就论证质量进行两两评判。作为实证基础，我们从经典小说中提取真实的道德困境，并使用逻辑求解器验证每个困境确实构成冲突。我们证明，我们的评判机制对论证质量的评分与人类标注者一致，策略随后会采用它所奖励的语言风格。在MoralChoice、ELEPHANT和MORABLES等外部基准上的测试表明，该策略变得更加果断，且对选项呈现顺序保持不变，但谄媚性和重新框架问题仍然是对有效性的挑战。

    arXiv:2610.30349v1 Announce Type: new  Abstract: LLMs are increasingly consulted for moral judgment. We aim to improve performance on such tasks by rewarding good arguments, on the premise that arguing well increases the likelihood of judging well. We train a reinforcement learning policy initialized from a state-of-the-art language model in a self-judged tournament with masked thinking: the generated arguments are judged pairwise on argument quality by the policy itself. As an empirical basis, we extract authentic dilemmas from classic novels, using a logic solver to verify that each constitutes a conflict. We show that our judge rates argument quality like human annotators, and the policy then adopts the language register that it rewards. Testing on the external benchmarks MoralChoice, ELEPHANT, and MORABLES, we find that the policy becomes more decisive and invariant to the order in which options are presented, but sycophancy and reframing remain a challenge to validity. Two ablatio
    
[^107]: AQL-Lite：通过以数据为中心的持续预训练实现阿拉伯语大语言模型的高效知识获取

    AQL-Lite: Enabling Efficient Knowledge Acquisition for Arabic LLMs via Data-Centric Continual Pretraining

    [https://arxiv.org/abs/2610.30344](https://arxiv.org/abs/2610.30344)

    AQL-Lite 通过质量优先的数据筛选流水线与部分参数持续预训练方案（RS-LoRA 适配器结合最后八层 Transformer 块与嵌入矩阵），以约一半的训练时间恢复了全参数 CPT 94% 的性能提升，高效地将多语言大模型适配至现代标准阿拉伯语。

    

    训练阿拉伯语大语言模型面临网络数据嘈杂、方言变体多样以及持续预训练（CPT）成本高昂等挑战。我们提出了 AQL-Lite（AQL：Arabic Quality Language，阿拉伯语质量语言），一个高效节省时间和资源的框架，用于将多语言大语言模型适配至现代标准阿拉伯语（MSA）。AQL-Lite 将以质量为先的数据筛选流水线（结构化清洗、启发式与基于模型的过滤、MSA 筛选以及去重）与部分参数 CPT 方案相结合：在整个网络中训练 RS-LoRA 适配器，同时训练最后八个 Transformer 块和嵌入矩阵，并采用逐层学习率。在匹配的 10 亿 token 预算下，使用 AQL 筛选数据进行 CPT 相比使用原始网络文本进行 CPT（后者相对基础模型并无提升），在四个阿拉伯语基准测试的平均分上提高了 3.51 分。在受控的 50 亿 token 对比实验中，AQL-Lite 仅用 52% 的实际训练时间（wall-clock time）就恢复了全参数 CPT 94% 的性能提升（平均分 73.19 对比 73.58）。

    arXiv:2610.30344v1 Announce Type: new  Abstract: Training Arabic large language models faces challenges from noisy web data, dialectal variation, and the high cost of continual pretraining (CPT). We introduce AQL-Lite (AQL: Arabic Quality Language), a time- and resource-efficient framework for adapting a multilingual LLM to Modern Standard Arabic (MSA). AQL-Lite couples a quality-first curation pipeline (structural cleaning, heuristic and model-based filtering, MSA selection, and deduplication) with a partial-parameter CPT recipe that trains RS-LoRA adapters across the network together with the last eight transformer blocks and the embedding matrix, under layer-wise learning rates. At a matched 1B-token budget, CPT on AQL-curated data raises a four-benchmark Arabic mean by 3.51 points over CPT on raw web text, which yields no gain over the base model. In a controlled 5B-token comparison, AQL-Lite recovers 94% of the gain of full-parameter CPT (mean 73.19 vs. 73.58) in 52% of the wall-c
    
[^108]: 衡量低资源语言的研究进展与语言包容性

    Measuring Research Progress and Linguistic Inclusion in Low-Resource Languages

    [https://arxiv.org/abs/2610.30296](https://arxiv.org/abs/2610.30296)

    该研究提出语言包容性框架（LIF），通过分析2020-2025年间1,158篇论文发现，尽管低资源语言NLP研究论文数量增长了5.4倍，但增长主要集中于已被代表的主流语言，而非真正扩展对代表性不足语言的覆盖。

    

    大语言模型（LLMs）加速了低资源与多语言自然语言处理（NLP）的研究，但它们在构建公平研究格局方面的有效性仍不明朗。我们分析了2020年至2025年间发表的1,158篇同行评审开放获取论文，考察了对低资源语言（LRLs）的研究关注度及其在语言和语言使用者群体中的代表性。为了超越单纯的论文发表数量来评估语言包容性，我们引入了语言包容性框架（LIF），这是一个从四个维度衡量语言包容性的多维框架：研究增长、代表性扩展、代表性延续性和分布公平性。尽管年度发表论文数量从52篇增长到415篇，在ACL Anthology的论文份额中增长了5.4倍，但这种增长主要集中于已被代表的语言，而非扩展对代表性不足语言的覆盖。进一步分析显示，亚洲使用的语言占据……（摘要原文在此处截断）

    arXiv:2610.30296v1 Announce Type: new  Abstract: Large Language Models (LLMs) have accelerated research on low-resource and multilingual NLP, but their effectiveness in creating an equitable landscape remains unmapped. We analyze 1,158 peer-reviewed open-access papers published between 2020 and 2025, examining research attention to low-resource languages (LRLs) and representation across language and speaker populations. To assess linguistic inclusion beyond publication counts, we introduce the Linguistic Inclusion Framework (LIF), a multidimensional framework that measures linguistic inclusion in four dimensions: research growth, representation expansion, representation continuity, and distributional equity. While annual publications grew from 52 to 415, a 5.4X increase in their ACL Anthology publication share, this growth largely concentrated on represented languages rather than expanding coverage of underrepresented languages. Further analysis shows that languages spoken in Asia acco
    
[^109]: 保护并非单一事件：基于度量终点的RAG保护研究综述

    Protection Is Not One Event: A Survey of RAG Protection Research by Measured Endpoint

    [https://arxiv.org/abs/2610.30193](https://arxiv.org/abs/2610.30193)

    该论文通过对118篇RAG防御论文按“实际度量终点”而非声称目标进行系统性分类综述，构建了七类分类体系，并揭示了防御机制与保护结果不对应、评估标准割裂等三大研究空白。

    

    检索增强生成（RAG）系统面临安全、隐私和版权风险，包括内容泄露、来源重构、未授权访问、数据投毒和提示注入。然而，RAG保护研究使用了互不兼容的评估指标、数据集和威胁模型。我们呈现了一项系统性综述和人工编码的证据图谱，涵盖118篇经全文确认的RAG防御论文，并按所度量的终点而非声称的目标或防御机制进行组织。我们的七类分类体系涵盖内容泄露、形式化隐私代理指标、访问控制与抑制、语料库使用检测以及流水线完整性与归因。32名标注者对论文进行了双重编码，kappa系数为0.867。该综述识别出三个研究空白。首先，检索重排序和上下文重写各自跨越五到六个终点类别，因此防御机制本身并不能确定保护结果。其次，仅有17.8%的研究使用内容泄露或重构（原文此处截断）

    arXiv:2610.30193v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) systems face security, privacy, and copyright risks including content leakage, source reconstruction, unauthorized access, data poisoning, and prompt injection. Yet RAG protection studies use incompatible evaluation metrics, datasets, and threat models. We present a systematic survey and human-coded evidence map of 118 full-text-confirmed RAG defense papers, organized by measured endpoint rather than claimed goal or defense mechanism. Our seven-family taxonomy covers content disclosure, formal privacy proxies, access control and suppression, corpus-use detection, and pipeline integrity and attribution. Thirty-two annotators double-coded the papers kappa=0.867. The survey identifies three gaps. First, retrieval reranking and context rewriting each span five or six endpoint families, so mechanisms do not identify protection outcomes. Second, only 17.8% of studies use content disclosure or reconstructi
    
[^110]: 有限可验证的分段：配备有限状态验证器的每个分词器均可判定

    Finitely Verifiable Segmentation: Decidability for Every Tokenizer with a Finite-State Verifier

    [https://arxiv.org/abs/2610.30168](https://arxiv.org/abs/2610.30168)

    本文提出“有限可验证分段”这一概念——即分段策略的结果恰好可被一个有限自动机验证——并证明在此假设下四个实际分词问题（边界窗口证明、边界间隙有界性、掩码生成可补全性以及策略等价性）均变为可判定问题。

    

    arXiv:2610.30168v1 公告类型：cross 摘要：分段策略为每个字节串指定至多一种切分为词元的分段方式。当定义在带标记字符串（即字节与边界位成对）上的有限自动机恰好接受该策略自身的分段时，我们称该策略是有限可验证的。该自动机用于评判一个给定的分段方案，而无需自行计算分段。本文展示了这一假设所带来的收益。四个原本在实践中靠经验规则解决的问题由此变为可判定的：一个窗口是否能证明某个词元边界、边界间隙是否有界、一个掩码是否能保持生成可补全，以及两个策略是否等价。后三个问题分别特化了单值转换器的有限性、余可达性与等价性问题；第一个问题精化了码理论中的同步切割概念，另外两个问题则询问一个经验证的切割、或某个清单中的所有切割，是否允许每个片段独立扫描。该类策略严格介于在线可扫描策略与具有正则定义域的策略之间，并且……（原文摘要在此处截断）

    arXiv:2610.30168v1 Announce Type: cross  Abstract: A segmentation policy assigns to each byte string at most one segmentation into tokens. Call it finitely verifiable when a finite automaton over marked strings, bytes paired with boundary bits, accepts exactly the policy's own segmentations. The automaton judges a proposed segmentation and need not compute one. The paper shows what that hypothesis buys. Four questions that practice settles by folklore become decidable, whether a window certifies a token boundary, whether the boundary gap is bounded, whether a mask keeps generation completable, and whether two policies are equivalent. The last three specialize limitedness, coaccessibility and the equivalence of single-valued transducers. The first refines the synchronizing cut of code theory, and two more ask whether a certified cut, or the cuts of an inventory, let each piece scan alone. The class sits strictly between the online-scannable policies and those with regular domains, and i
    
[^111]: FusionEval：面向大语言模型模型融合的基准测试

    FusionEval: Benchmarking Model Fusion for Large Language Models

    [https://arxiv.org/abs/2610.30162](https://arxiv.org/abs/2610.30162)

    提出了FusionEval基准，通过受控专家池和统一评估套件系统评估大语言模型的模型融合方法，揭示无需训练的融合方法虽能保留推理性能，但会显著损害工具使用能力。

    

    模型融合旨在将多个专家模型的能力整合到单一模型中。现有评估对长思维链推理和工具使用的覆盖有限，而良好的平均性能可能掩盖单个专家能力的大幅损失。我们提出了FusionEval，这是一个提供受控专家池、整合有无额外训练的融合方法、并提供统一评估套件的基准。专家池包含从Qwen3和Llama基础模型独立训练的领域专家，涵盖数学、代码、科学、指令遵循和工具使用五个领域。该评估套件在可比较的数据访问规则下，在八个公开基准上评估能力保留与计算成本，报告归一化的平均性能与最差性能、运行时间以及峰值GPU内存占用。大量实验表明，无需训练的方法能够保留推理性能，但会显著降低工具使用……

    arXiv:2610.30162v1 Announce Type: new  Abstract: Model fusion aims to integrate the capabilities of multi-expert models into a single model. Existing evaluations offer limited coverage of reasoning with long chains of thought and tool use, while strong average performance can mask substantial losses in individual expert capabilities. We introduce FusionEval, a benchmark that provides controlled expert pools, integrates fusion methods with and without additional training, and offers a unified evaluation suite. The pools contain domain experts independently trained from Qwen3 and Llama base models across math, code, science, instruction following, and tool use. The suite evaluates capability retention and computational cost on eight public benchmarks under comparable data access rules, reporting normalized average and worst performance, runtime, and peak GPU memory usage. Extensive experiments show that training-free methods preserve reasoning performance but substantially degrade tool u
    
[^112]: TauIndianBankBench：流程感知、动作导向的工具使用型对话智能体评估

    TauIndianBankBench: Procedure-Aware, Action-Oriented Evaluation of Tool-Using Conversational Agents

    [https://arxiv.org/abs/2610.30161](https://arxiv.org/abs/2610.30161)

    该论文提出了 TauIndianBankBench，一个包含 1,000 个印度零售银行任务的面向动作的基准测试，它不仅检查对话的最终记录，还检查助手是否完整执行了所有必需的流程步骤且未做出任何未经请求的更改，从而更准确地评估工具使用型对话智能体对银行流程的遵循能力。

    

    许多印度银行通过常见问题（FAQ）页面和脚本化聊天机器人来服务客户。基于语言模型的助手能够更快地响应并执行诸如冻结银行卡之类的请求，但前提是它们必须提供正确的信息并遵守银行的规定。这些规定定义了一套流程，即一系列必需的步骤，例如在关闭存款账户之前必须先获取结息报价。要将模型适配到这一领域，就需要在训练前后衡量其遵循流程的能力。现有的基准测试通常只根据对话结束时留下的记录进行评判。然而，许多必需的步骤并不会改变记录，因此跳过这些步骤的助手仍可能被判定为成功。我们提出了 TauIndianBankBench，这是一个面向动作的基准测试，包含 1,000 个印度零售银行任务，其中助手需要帮助一位模拟客户。除了最终记录之外，该基准还会检查助手是否执行了每一个必需的步骤，并且没有做出任何未经请求的更改。

    arXiv:2610.30161v1 Announce Type: new  Abstract: Many Indian banks serve customers through FAQ pages and scripted chatbots. Assistants based on language models could respond faster and carry out requests such as blocking a card, but only if they give correct information and follow the bank's rules. These rules prescribe a procedure, a set of required steps, such as obtaining a closure quote before closing a deposit. Adapting models to this domain requires measuring how well they follow the procedure, before and after training. Benchmarks usually judge only the records left at the end of a conversation. Many required steps, however, leave the records unchanged, so an assistant that skips them can still be judged successful. We present TauIndianBankBench, an action-oriented benchmark of 1,000 Indian retail-banking tasks where an assistant helps a simulated customer. Besides the final records, it checks that the assistant took every required step and made no unrequested change. For every 
    
[^113]: Lighthouse：构建一个基于民权档案探索历史的可信赖数字助手

    Lighthouse: On Building a Trustworthy Digital Assistant for Exploring History with Civil Rights Archives

    [https://arxiv.org/abs/2610.30158](https://arxiv.org/abs/2610.30158)

    该论文提出了首个用于探索美国民权时期历史的可信赖数字助手Lighthouse，其创新在于采用神经符号方法（结合大语言模型与基于规则的符号求解），以内容可信度和减少生成答案的潜在伤害为核心设计原则。

    

    人工智能（AI）作为一种决策辅助工具，为利用包括文本、图像和音频在内的多模态数据高效研究历史提供了新的工具。据我们所知，我们提出了第一个名为Lighthouse的数字助手（通常称为聊天机器人），用于基于南卡罗来纳州的数据探索美国民权时期的历史。从AI的角度来看，该系统的新颖之处在于其指导性设计原则： 关注内容的可信性； 通过对生成的答案可能对用户造成的潜在伤害进行建模并加以减少。系统通过一种神经符号方法来实现这些原则，该方法将大型语言模型（LLM）的神经学习与基于规则的符号求解相结合，在答案的风险与覆盖面的广泛性之间取得平衡。我们使用具有单一可接受答案的事实性客观问题以及主观问题对Lighthouse进行评估。对于后者，一个主观数据集……

    arXiv:2610.30158v1 Announce Type: new  Abstract: Artificial intelligence (AI), as a decision-support aid, provides new tools for studying history effectively using multimodal data, including text, images, and audio. We present, to our knowledge, the first digital assistant named Lighthouse--commonly called a chatbot--for exploring the Civil Rights period of the United States using data from South Carolina. From an AI perspective, the system's novelty lies in its guiding design principles of (a) focusing on trustworthiness of content and (b) by modeling and reducing potential harm to users from generated answers. It implements these with a neuro-symbolic approach that combines neural learning through a large language model (LLM) with rule-based symbolic solving, balancing risk of answers with generality of coverage. We evaluate Lighthouse with both factual objective questions that admit a single acceptable answer as well as subjective questions. For the latter, a dataset of subjective q
    
[^114]: baseVultra：具有容量重新分配的紧凑非线性递归网络

    baseVultra: Compact Nonlinear Recurrence with Reallocated Capacity

    [https://arxiv.org/abs/2610.30154](https://arxiv.org/abs/2610.30154)

    提出baseVultra紧凑递归单元，将递归计算从门控变换重新分配到候选生成，在参数量匹配条件下于TinyShakespeare字符级语言建模上超越LSTM和GRU。

    

    非线性递归网络在流式序列建模中仍然很有用，然而门控单元的设计仍然由LSTM和GRU主导。我们研究递归计算是否可以从门控变换重新分配到候选生成上。我们提出了baseVultra，这是一种紧凑的递归单元，它将密集的递归计算预算集中在单个H x H的候选生成矩阵上，同时用O(H)的对角交互替换递归门控矩阵。候选更新将这个密集分支与两个低秩辅助分支、一个EMA缓存以及可学习的输出混合相结合；门控还通过对角项获得对h_{t-2}的轻量级二阶访问。在TinyShakespeare字符级语言建模任务上，baseVultra在紧凑的单层200K参数设置中实现了1.594的验证损失，而参数量匹配的LSTM为1.604，参数量匹配的GRU为1.616。

    arXiv:2610.30154v1 Announce Type: new  Abstract: Nonlinear recurrent networks remain useful for streaming sequence modeling, yet gated cell design is still dominated by LSTM and GRU. We study whether recurrent computation can be reallocated away from gate transforms and toward candidate generation. We propose baseVultra, a compact recurrent cell that concentrates the dense recurrent budget on a single H x H candidate-generation matrix, while replacing recurrent gate matrices with O(H) diagonal interactions. The candidate update combines this dense branch with two low-rank auxiliary branches, an EMA cache, and learnable output blending; the gates additionally receive lightweight second-order access to h_{t-2} through diagonal terms. On TinyShakespeare character-level language modeling, baseVultra achieves a validation loss of 1.594 in the compact one-layer 200K setting, compared with 1.604 for a parameter-matched LSTM and 1.616 for a parameter-matched GRU, while its dense recurrent matr
    
[^115]: GoiBERT：通过两阶段知识蒸馏构建紧凑型越南语语言模型

    GoiBERT: Compact Vietnamese Language Models via Two-Stage Knowledge Distillation

    [https://arxiv.org/abs/2610.30067](https://arxiv.org/abs/2610.30067)

    GoiBERT通过两阶段知识蒸馏（先从同分词器越南语编码器学习通用表示，再从大型教师模型获取软标签任务指导），构建了仅2600万和4900万参数、可直接处理原始文本的紧凑越南语编码器，在八项基准测试中以不到一半的参数量和两倍吞吐量取得非教师模型中的最高平均分数。

    

    越南语预训练编码器近年来发展迅速，但性能最强的模型要么是采用多语言词表的大型模型，要么依赖外部分词工具，导致部署成本高昂。我们提出了GoiBERT，这是一系列包含两个紧凑型越南语编码器的模型家族，参数量分别为2600万和4900万，可直接处理原始文本。GoiBERT从两个互补的教师模型中学习：首先由一个使用相同分词器的越南语编码器传授通用语言表示；随后由一个大得多的教师模型通过软标签提供任务级指导，且软标签的生成不会标注教师模型自身的训练数据。在涵盖文本分类、方面级情感分析、自然语言推理和命名实体识别的八项基准测试中，GoiBERT-large在非教师模型中取得了最高的平均分数，其参数量不到最强紧凑基线模型的一半，而吞吐量则是其两倍。GoiBERT-base保留了99%的……（原文摘要至此截断）

    arXiv:2610.30067v1 Announce Type: new  Abstract: Vietnamese pre-trained encoders have improved rapidly, but the strongest ones are large, multilingual-vocabulary models or depend on external word segmentation, which makes them costly to deploy. We present GoiBERT, a family of two compact Vietnamese encoders with 26M and 49M parameters that operate directly on raw text. GoiBERT learns from two complementary teachers. A same-tokenizer Vietnamese encoder first teaches general-purpose representations. A much larger teacher then provides task-level guidance through soft labels that are generated without ever labeling its own training data. Across eight benchmarks covering text classification, aspect-based sentiment analysis, natural language inference, and named entity recognition, GoiBERT-large achieves the highest average score among non-teacher models. It does so with fewer than half the parameters of the strongest compact baseline and twice its throughput. GoiBERT-base retains 99% of th
    
[^116]: 探究学习者错误对大型语言模型中社会偏见的影响

    Investigating the Impact of Learner Errors on Social Bias in Large Language Models

    [https://arxiv.org/abs/2610.29896](https://arxiv.org/abs/2610.29896)

    该研究首次系统考察了非母语写作者的语法错误对大型语言模型社会偏见的影响，发现错误率越高偏见越强（其中介词错误影响最大），并通过在错误文本上微调有效缓解了这一问题。

    

    现有的社会偏见基准测试大多在流畅、规范的英语上评估大型语言模型，反映的是母语使用者的用法，而未将非母语写作者纳入考量。为填补这一空白，我们研究了非母语写作者的语法错误如何影响社会偏见。我们开发了一个受控框架，将常见的学习者错误注入提示中，并将其应用于三个偏见基准测试，涵盖性别、种族和宗教方面的刻板印象一致性以及人口群体偏好。我们发现，语法错误会加剧偏见，且这种影响随错误率的增加而增强。在我们考察的四种常见错误类型中，介词错误导致刻板印象一致性出现最大且最一致的增幅。我们的分析表明，当错误更可能干扰理解时，其加剧偏见的作用更强。最后，在含错误文本上进行微调可以减少语法错误造成的偏见增加。我们的发现表明，偏……

    arXiv:2610.29896v1 Announce Type: new  Abstract: Existing social bias benchmarks evaluate large language models mostly on fluent, well-formed English, reflecting native speaker usage and leaving non-native writers unrepresented. To address this gap, we investigate how grammatical errors by non-native writers affect social bias. We develop a controlled framework that injects common learner errors into prompts and apply it to three bias benchmarks covering stereotype alignment and demographic preference across gender, race, and religion. We find that grammatical errors increase bias, and the effect grows with the error rate. Among the four common error types we consider, preposition errors produce the largest and most consistent increase in stereotype alignment. Our analysis suggests that errors increase bias when they are more likely to interfere with comprehension. Finally, fine-tuning on erroneous texts reduces the bias increase caused by grammatical errors. Our findings show that bia
    
[^117]: 与开放世界共同演化：迈向未来预测的递归自我改进

    Evolving with Open Worlds: Towards Recursive Self-Improvement for Future Forecasting

    [https://arxiv.org/abs/2610.29784](https://arxiv.org/abs/2610.29784)

    提出DeepEvolve智能体系统，以未来事件预测作为持续变化且可验证的开放环境，通过递归演化引擎将预测背后的推理轨迹转化为能力，从而实现智能体的递归自我改进。

    

    当今大多数人工智能（AI）系统在发布时参数即被冻结，只能通过外部更新来改进；而理想的系统应能从自身经验中持续改进，我们将这种能力称为递归自我改进（RSI）。然而，研究RSI需要一个持续变化、可客观验证且自然扎根的环境，这给智能体演化带来了根本性挑战。在此，我们提出DeepEvolve，一个以未来事件预测为基础的可持续演化智能体系统，其中智能体与动态演化的环境持续交互，并通过内化和外化这些交互中获得的经验来实现改进。一个“活的”环境负责合成可验证的任务，而递归演化引擎则将每次预测背后的推理轨迹重新转化为自身能力：通过将搜索与预测解耦，使参数级演化能够适应……

    arXiv:2610.29784v1 Announce Type: new  Abstract: Most contemporary artificial intelligence (AI) systems are released with frozen parameters and improve through external updates, whereas an ideal system would keep improving from its own experience, a capacity we call recursive self-improvement (RSI). Yet studying RSI requires an environment that is continuously changing, objectively verifiable, and naturally grounded, posing a fundamental challenge for agent evolution. Here we introduce DeepEvolve, a sustainable evolution agent system grounded in future event prediction, in which agents continuously interact with a dynamically evolving environment and improve by internalizing and externalizing the experience gained through these interactions. A living environment synthesizes verifiable tasks, and a recursive-evolution engine turns the reasoning trajectory behind each prediction back into capability: decoupling search from prediction adapts parameter-level evolution to the contamination-
    
[^118]: CHIP 2026共享任务1概述：全文生物医学文献中的患者特异性表型概念识别

    Overview of CHIP 2026 Shared Task 1: Patient-Specific Phenotype Concept Recognition in Full-Text Biomedical Literature

    [https://arxiv.org/abs/2610.29606](https://arxiv.org/abs/2610.29606)

    本文概述了CHIP 2026共享任务1，该任务基于PatientPheX数据集的200篇标注全文文献，设置全文表型概念识别与患者特异性表型关联两个子任务，以推动从全文生物医学文献中自动识别表型概念及其患者关联方法的发展。

    

    自动表型信息抽取在罕见病诊断、基因型-表型分析和精准医疗中发挥着重要作用。然而，准确识别表型概念并确定其与特定患者的关联仍然具有挑战性，尤其是在涉及多名患者和复杂临床描述的全文生物医学文章中。本文对CHIP 2026共享任务1进行了概述，该任务旨在推动从全文生物医学文献中进行患者特异性表型概念识别方法的发展。该任务包含两个子任务：全文表型概念识别和患者特异性表型关联。在本次评测中，我们使用了来自PatientPheX数据集的200篇经标注的全文生物医学文章，其中包含506名患者、15,193个表型提及和5,656个患者-表型关联。共有455支团队报名参加了此次竞赛。

    arXiv:2610.29606v1 Announce Type: new  Abstract: Automatic phenotype information extraction plays an important role in rare disease diagnosis, genotype-phenotype analysis, and precision medicine. However, accurately identifying phenotype concepts and determining their associations with specific patients remain challenging, particularly in full-text biomedical articles involving multiple patients and complex clinical descriptions. This paper presents an overview of CHIP 2026 Shared Task 1, which aims to promote the development of methods for patient-specific phenotype concept recognition from full-text biomedical literature. The task consists of two subtasks: full-text phenotype concept recognition and patient-specific phenotype association. For this evaluation, we used 200 annotated full-text biomedical articles from the PatientPheX dataset, containing 506 patients, 15,193 phenotype mentions, and 5,656 patient-phenotype associations. A total of 455 teams registered for the competition,
    
[^119]: 跨语言语音克隆中用于口音控制的闭环双参考引导方法

    Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning

    [https://arxiv.org/abs/2610.29351](https://arxiv.org/abs/2610.29351)

    提出闭环双参考引导方法，通过将原始参考与语音转换构建的母语锚点配对并动态更新语言模型状态，有效消除跨语言语音克隆中的源语言口音，在仅损失极小说话人相似度的前提下显著提升目标语言的发音质量。

    

    零样本跨语言语音克隆常常会将参考说话人的口音带入目标语言中。我们提出闭环双参考引导方法，将原始参考与通过语音转换（VC）构建的可复用母语锚点进行配对。这两个参考分别初始化各自独立的语言模型（LM）状态，共同引导一条共享的声学轨迹；每个完成的语音片段随后会更新这两个状态，使其生成历史与实际输出的语音保持对齐。锚点构建每个说话人仅需执行一次，在线合成使用冻结的TTS模型，无需逐句进行语音转换。在20个保留测试的法语说话人上，固定设置 λ*=0.33 与直接克隆相比，将残余的法语LID质量降低了73%，ECAPA相似度损失仅为0.042；相对于仅使用锚点的克隆，该方法恢复了0.155的相似度。在保留测试组中，该方法实现了最低的平均WER（0.009）和最高的平均GOP（4.750）。

    arXiv:2610.29351v1 Announce Type: cross  Abstract: Zero-shot cross-lingual voice cloning often carries the reference speaker's accent into the target language. We propose closed-loop dual-reference guidance, which pairs the original reference with a reusable native anchor constructed by voice conversion (VC). The references initialize separate language-model (LM) states that guide a shared acoustic trajectory; each completed patch then updates both states, keeping their generated histories aligned with the emitted speech. Anchor construction runs once per speaker, and online synthesis uses a frozen TTS model without per-utterance VC. On 20 held-out French speakers, the fixed setting $\lambda^{\!*}=0.33$ reduces residual French LID mass by 73\% versus direct cloning, at an ECAPA similarity cost of only $0.042$; relative to anchor-only cloning, it recovers $0.155$. The method achieves the lowest mean WER ($0.009$) and highest mean GOP ($4.750$) among the held-out test arms. In blinded pr
    
[^120]: 无审查不等于无偏见：“消融审查”移除了拒绝行为，却未消除关于台湾的书写文字相关偏见

    Uncensored Is Not Unbiased: Abliteration Removes Refusals but Not Script-Dependent Bias About Taiwan

    [https://arxiv.org/abs/2610.29342](https://arxiv.org/abs/2610.29342)

    去审查化（abliteration）模型只是移除了拒绝行为，其对台湾问题的政治立场主要取决于所用书写文字（繁体、简体还是英文），而非是否经过消融审查，因此“无审查”并不等于“无偏见”。

    

    “去审查化”（“无审查”）的开放权重语言模型版本通过移除介导拒绝行为的激活方向而构建，其流行的部分原因在于一种假设：不再拒绝回答的模型也就摆脱了政治审查。我们在两对匹配的2比特量化模型上对这一假设进行测试：NVIDIA 的 Nemotron 3.5 Bolt 74B-A9B 和 Qwen3.8-Flash-Next，每款模型分别与其对应的社区消融审查版本配对比较。每款模型分别在繁体中文、简体中文（字符级转换）和英文下，对22组陈述对（涉及中华民国（台湾）、1989年天安门事件及对照国家）进行判断，并针对每种文字回答27个开放式问题。结果显示，书写文字的影响远大于消融审查：原始 Bolt 模型在指数 S（-1 代表中华人民共和国（PRC）立场；+1 代表中华民国（ROC）立场）上，繁体中文得分为 +0.58，英文为 +0.57，简体中文仅为 +0.05；且该模型以 0.91 的概率在简体中文中肯定“台湾是中华人民共和国的一部分”，而在繁体中文中该概率仅为 0.06。

    arXiv:2610.29342v1 Announce Type: cross  Abstract: "Abliterated" ("uncensored") builds of open-weight language models remove the activation direction that mediates refusal, and are popular partly on the assumption that a model that no longer refuses is also free of political censorship. We test this on two matched pairs of 2-bit models: NVIDIA's Nemotron 3.5 Bolt 74B-A9B and Qwen3.8-Flash-Next, each against a community abliteration. Each model judged 22 statement pairs (on the Republic of China (Taiwan), 1989 Tiananmen, control countries) in Traditional Chinese, Simplified Chinese (a character-level conversion) and English, and answered 27 open questions per script. Script mattered far more than abliteration: the original Bolt scored +0.58 on Index S (-1: People's Republic of China (PRC) position; +1: ROC position) in Traditional Chinese, +0.57 in English and +0.05 in Simplified Chinese, and affirmed "Taiwan is part of the PRC" with probability 0.91 in Simplified but 0.06 in Traditiona
    
[^121]: OACap：面向详细视频描述的描述引导式主动观察

    OACap: Caption-Guided Active Observation for Detailed Video Captioning

    [https://arxiv.org/abs/2610.29327](https://arxiv.org/abs/2610.29327)

    提出了以描述为中心的智能体框架 OACap，它能将视频描述中缺失或缺乏支持的信息转化为主动观察请求，借助时空工具获取证据并迭代验证修订描述，从而显著提升详细视频描述质量，同时引入 FSCap-Bench 基准及 CSFS 等评分指标来系统评估描述的事实性与语义保真度。

    

    与问答任务不同（问答中的查询已明确了感知目标），开放式详细视频描述必须自主决定需要获取什么证据，以及如何将这些证据整合成连贯的全局描述。我们提出了观察自适应视频描述器，这是一个以描述为中心的智能体，它将描述中缺乏支持或缺失的信息转化为有针对性的观察请求，利用时空工具获取证据，并迭代地验证和修订描述内容。在使用固定的纯文本阅读器的情况下，OACap 生成的描述在多个通用视频理解基准上均取得了一致的性能提升。为了更好地诊断这些描述所保留的信息，我们引入了事实-语义描述基准（FSCap-Bench）。其中的核心支撑事实性得分（CSFS）将断言精确度与重要性感知的事实覆盖率相结合，在 93.0-98.3% 的有效描述对上对四种受控扰动类型做出了符合预期的响应。其跨模态语义往返得分（摘要原文在此处截断）

    arXiv:2610.29327v1 Announce Type: cross  Abstract: Unlike question answering, where a query specifies a perceptual target, open-ended detailed captioning must determine what evidence to acquire and how to integrate it into a coherent global description. We propose Observation-Adaptive Video Captioner (OACap), a caption-centered agent that turns unsupported or missing information into targeted observation requests, uses spatiotemporal tools to acquire evidence, and iteratively verifies and revises the caption. OACap captions consistently improve performance across general video-understanding benchmarks using a fixed text-only reader. To better diagnose the information they preserve, we introduce Factual-Semantic Captioning Bench (FSCap-Bench). Its Core-Support Factuality Score (CSFS) combines claim precision with importance-aware fact coverage and responds as expected to four controlled perturbation types on 93.0-98.3% of valid caption pairs. Its Cross-Modal Semantic Round-Trip Score (C
    
[^122]: 当多个选择都正确时：Jev及其他决策模型中的非对称概率分配

    When Multiple Choices Are Correct: Asymmetric Probability Allocation in Jev and Other Decision Models

    [https://arxiv.org/abs/2610.29217](https://arxiv.org/abs/2610.29217)

    该论文通过在三个数据集上评估三个决策模型，首次系统揭示了多正确答案场景下概率分配的不对称性问题——即使模型排序完全正确，仍有28%至50%的案例中至少一个正确选择的概率低于均匀概率1/n，且给相同选项分配的概率差距高达17至79个百分点。

    

    决策模型会返回候选选择上的归一化概率分布，下游应用可以利用这些概率在多个选择中进行筛选。尽管模型提供商强调这些概率的可靠性，但在单一正确答案任务上的校准并不能证明当存在多个正确选择时这些概率依然可靠。我们在三个多样化数据集（MetaTool、HotpotQA和ECtHR）上评估了三个决策模型（Jev、Liquid d1和GPT-6 Luna Decisions），以研究概率分配和阈值设定问题。研究发现，即使模型将所有正确选择都排在所有干扰项之上，在这些列表中仍有28%至50%的情况，至少有一个正确选择获得的概率低于均匀概率1/n（其中n为选择总数）。我们还向每个模型展示了同一正确选择的五个相同副本，观察到明显的概率差异，平均差距在17至79个百分点之间。我们进一步展示了这些分配不均……

    arXiv:2610.29217v1 Announce Type: new  Abstract: Decision models return normalized probability distributions over candidate choices, which downstream applications can use to select among choices. Although their providers emphasize the reliability of these probabilities, calibration on tasks with a single correct choice does not establish their reliability when multiple choices are correct. We evaluate three decision models (Jev, Liquid d1, and GPT-6 Luna Decisions) on three diverse datasets (MetaTool, HotpotQA, and ECtHR) to study probability allocation and thresholding. Even when models rank every correct choice above every distractor, in 28% to 50% of those lists, at least one correct choice receives a probability below the uniform probability of 1/n, where n is the number of choices. We also present each model with five identical copies of a correct choice and observe probability disparities, with mean gaps ranging from 17 to 79 percentage points. We further show how these allocatio
    
[^123]: 不改变任何权重的 Gemma 4 语音编码器能走多远？基于冻结编码器与适配器增强编码器的沙特阿拉伯语和海湾阿拉伯语语音识别

    How Far Can the Gemma 4 Speech Encoder Go Without Changing Its Weights? Saudi and Gulf Arabic Speech Recognition with a Frozen and an Adapter-Augmented Encoder

    [https://arxiv.org/abs/2610.28673](https://arxiv.org/abs/2610.28673)

    在完全不改变 Gemma 4 语音编码器权重的前提下，仅通过轻量级 CTC 读取器或少量低秩适配器参数，就能让沙特和海湾阿拉伯语语音识别达到与参数量大得多的商业系统统计上相当的性能，同时在 CPU 上的推理速度快 3.4 倍。

    

    我们研究了 Gemma 4 与 EmbeddingGemma 2 共享的 3.05 亿参数音频编码器（两者除最终投影层外完全相同），在不改变其任何权重的情况下将其用于沙特阿拉伯语和海湾阿拉伯语语音识别。冻结系统仅在编码器多个层特征的混合之上训练一个 2260 万参数的 CTC 读取器；适配增强系统则额外增加 310 万个低秩适配器参数、一个 4150 万参数的混合 CTC/注意力读取器，以及与子词语言模型的联合解码。与在相同音频上重新运行的 Cohere Transcribe Arabic（20.7 亿参数）相比，冻结系统在词错误率（WER）上略逊一筹，但在六个人工标注测试条件中的五个上，字符错误率（CER）在统计上不相上下，同时在使用贪婪解码时在 CPU 上的运行速度快 3.4 倍。适配增强系统在公开的沙特客服通话数据上的 WER（19.20% 对 19.27%）以及在干净的阿联酋语音上与对方在统计上不相上下，而在电话编码的阿联酋语音上则显著更优……（原文摘要在此处被截断）

    arXiv:2610.28673v1 Announce Type: new  Abstract: We study the 305M-parameter audio encoder shared by Gemma 4 and EmbeddingGemma 2 (identical except for its final projection) for Saudi and Gulf Arabic speech recognition, without changing any of its weights. A frozen system trains only a 22.6M-parameter CTC reader over a mix of encoder layers; an adapted system adds 3.1M low-rank adapter parameters, a 41.5M-parameter hybrid CTC/attention reader and joint decoding with a sub-word language model. Against Cohere Transcribe Arabic (2.07B parameters), re-run on identical audio, the frozen system trails in word error rate (WER) but is statistically indistinguishable in character error rate (CER) on five of six human-labelled conditions, while running 3.4 times faster on CPU with greedy decoding. The adapted system is statistically indistinguishable in WER on a public Saudi customer-service call (19.20% against 19.27%) and on clean Emirati speech, is significantly better on telephone-coded Emir
    
[^124]: 存在但难以读取：用最小提示对探查潜在推理

    Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs

    [https://arxiv.org/abs/2610.28498](https://arxiv.org/abs/2610.28498)

    论文通过仅有一个数字不同的最小提示对证明，从输入复制的信息确实以跨问题共享的编码存在于模型状态中，但常规单样本探针难以读取它，因此探查失败不能证明信息缺失，并提出了一套区分“计算出的值”与“输入副本/混合”的探查协议。

    

    潜在推理的探查研究通常在答案开始生成的位置读取模型状态，并将探查失败当作该信息不存在的证据。我们用一个从问题中复制的数字来检验这一点，实验对象是循环模型 Ouro-1.4B 和标准 Transformer Qwen2.5-1.5B。比较仅在该数字上不同的提示对可以发现，该数字确实存在于最后一个提示词元中，并以一种跨问题共享的编码形式存在（读取准确率约 0.9）。而一次只看单个提示的常规探针几乎读不出它：即使训练了 5,249 个问题，Ouro 中仅达 0.36，Qwen 中仅达 0.19。该数字引起的状态变化是“微小”而非“拥挤”的：随机噪声会落在同样繁忙的方向上，而用一个随机词替换该数字会使状态发生大得多的位移且可被读取。由于每个变化的词元都会留下痕迹，成功的探查结果同样存在歧义。我们提出了一套探查协议，包括用于检验一般性混合效应的合成零假设，用于将模型计算出的值与输入的副本及混合信息进行区分测试。

    arXiv:2610.28498v1 Announce Type: cross  Abstract: Probing studies of latent reasoning often read a model's state where the answer begins and take a failed probe as evidence that a value is absent. We test this with a digit copied from the question, in the looped model Ouro-1.4B and the standard transformer Qwen2.5-1.5B. Comparing prompts that differ only in that digit shows it is present at the last prompt token, in a code shared across questions (readout about 0.9). A probe that sees one prompt at a time, the usual setting, barely reads it: 0.36 in Ouro and 0.19 in Qwen even after 5,249 training questions. The digit's change is small rather than crowded: random noise lands on equally busy directions, while a random word in its place moves the state far more and is readable. Since every varied token leaves a trace, a successful probe is ambiguous too. We give a protocol, including a synthetic null for generic mixing, for testing computed values against copies and mixtures of the input
    
[^125]: 披露引发披露：具有持久记忆的LLM智能体中的自我强化隐私循环

    Disclosure Begets Disclosure: A Self-Reinforcing Privacy Loop in LLM Agents with Persistent Memory

    [https://arxiv.org/abs/2610.28315](https://arxiv.org/abs/2610.28315)

    该研究揭示了持久记忆型LLM智能体中存在自我强化的隐私泄露循环：基于记忆的响应会将早期属性值作为预设延续，用户确认或不加纠正，导致属性泄露随会话次数持续扩大。

    

    大语言模型智能体越来越多地使用持久记忆来个性化响应，而记忆中存储的个人属性一旦暴露便构成隐私风险。我们探究在对话中查阅记忆如何改变用户后续的发言内容，进而改变记忆最终会揭示的信息。在多会话模拟中，一个拥有24个个人属性的模拟用户与一个查阅或不查阅记忆的客服智能体对话，并在两种设置下均从完整累积的对话记录中推断这些属性。在四个客服模型上，使用记忆提升了响应质量，但同时也增加了经验性泄露——即从记录中能被正确推断出的属性值比例——且这一差距随会话次数增加而不断扩大。其根源在于普通的、非对抗性辅助过程中存在的一个循环：基于记忆的响应会将早期的属性值延续下去（常以预设的形式出现），而用户会确认这些值或不予纠正，从而使线索和智能体自身的猜测都被记录下来。

    arXiv:2610.28315v1 Announce Type: cross  Abstract: Large language model (LLM) agents increasingly use persistent memory to personalize responses, and personal attributes in this memory pose a privacy risk if exposed. We ask how consulting memory during dialogue changes what users go on to say, and thus what the memory comes to reveal. In multi-session simulations, a simulated user with 24 personal attributes talks with a support agent that does or does not consult memory, and in both settings attributes are inferred from the full accumulated record. Across four support models, memory raises response quality but also empirical leakage, the share of attribute values correctly inferred from the record, by a margin that widens over sessions. The cause is a loop in ordinary, non-adversarial assistance. Memory-conditioned responses carry earlier values forward, often as presuppositions, and users affirm them or leave them uncorrected, so that cues and the agent's own guesses become recorded 
    
[^126]: 基于投毒的后门并非密码学安全的

    Poisoning-Based Backdoors Are Not Cryptographic

    [https://arxiv.org/abs/2610.28045](https://arxiv.org/abs/2610.28045)

    本文通过归约证明，基于投毒安装的神经网络后门不可能获得密码学认证那样的不对称性，因为任何从被投毒数据中安装触发器验证器的训练过程本身就是一个学习过程，因此基于学习的防御在原则上始终是可行的。

    

    为了防御针对神经网络的后门攻击，防御者必须识别出决定某个输入是否为触发器的验证器函数，而不能只是孤立地拒绝单个触发器，因为攻击者生成满足同一验证器的新触发器的速度会快于防御者拦截它们的速度。如果后门验证器能够像密码学认证方案那样具有不对称性——即只有攻击者能够产生触发器输入，而防御者无法通过观测逆向推导出验证器——那么基于学习的防御在原则上将是不可能的。我们证明，这种不对称性不可能由基于投毒的安装方式产生，在这种方式中，训练过程中唯一依赖于验证器的输入就是被投毒的数据集。该论证采用归约方法：一个能够从被投毒数据中安装验证器、使后门在未见过的触发器上触发同时保持良性行为的训练过程，其本身就是一个学习（摘要在此处截断）

    arXiv:2610.28045v1 Announce Type: cross  Abstract: To defend against backdoor attacks on neural networks, the defender must identify the verifier function that determines whether an input is a trigger, rather than rejecting individual triggers in isolation, since an attacker can generate new triggers satisfying the same verifier faster than they can be blocked. If a backdoor verifier could be installed with the asymmetry of a cryptographic authentication scheme, in which only the attacker can produce trigger inputs while the defender cannot reverse-engineer the verifier from observations, learning-based defense would be impossible in principle. We show that no such asymmetry can arise from poisoning-based installation, in which the only verifier-dependent input to training is a poisoned dataset. The argument is a reduction. A training procedure that installs a verifier from poisoned data, so that the backdoor fires on unseen triggers while benign behavior is preserved, is itself a lear
    
[^127]: 基于解析平方和的关于正性裕度的 Putinar 证书的精确次数界

    Sharp Degree Bounds for Putinar Certificates in Terms of the Positivity Margin via Analytic Sums of Squares

    [https://arxiv.org/abs/2610.26976](https://arxiv.org/abs/2610.26976)

    本文建立了关于正性裕度 δ 的 Putinar 平方和证书的精确次数上界 O(δ^(-1/2)log²(2/δ))，并通过构造三变量匹配下界例子证明了对 δ 的依赖是最优的，同时给出了 Lasserre 层级的精确最坏情形收敛速率。

    

    我们建立了关于正性裕度 $\delta$ 的 Putinar 平方和（SOS）证书的精确次数界。对于任意具有指定生成元的固定阿基米德二次模以及目标函数次数的任意固定上界，我们证明了归一化多项式目标的一致证书次数上界为 $O(\delta^{-1/2}\log^2(2/\delta))$。我们通过在三变量情形下构造一个匹配的下界例子，证明了该界是精确的，从而确立了对 $\delta$ 的依赖性是最优的。这些匹配的界还给出了 Lasserre 层级的精确最坏情形收敛速率。

    arXiv:2610.26976v1 Announce Type: new  Abstract: We establish sharp degree bounds for Putinar sum-of-squares (SOS) certificates in terms of the positivity margin $\delta$. For any fixed Archimedean quadratic module with prescribed generators and any fixed bound on the degree of the objective, we prove a uniform certificate-degree upper bound of $O(\delta^{-1/2}\log^2(2/\delta))$ for normalized polynomial objectives. We show that this bound is sharp by constructing a matching lower-bound example in three variables, thereby establishing that the dependence on $\delta$ is optimal. These matching bounds also yield sharp worst-case convergence rates for the Lasserre hierarchy.
    
[^128]: SpatialScout：视觉语言模型智能体能否主动探索并构建3D场景地图？

    SpatialScout: Can VLM Agents Actively Explore and Map 3D Scenes?

    [https://arxiv.org/abs/2610.26659](https://arxiv.org/abs/2610.26659)

    提出了SpatialScout基准，用于评估视觉语言模型智能体在有限交互预算下，通过主动探索室内场景来构建物体级3D度量地图的能力。

    

    空间智能要求智能体主动获取视觉证据，并将观察结果整合为对三维世界的连贯表征。然而，空间问答任务对于模型能否通过自主探索构建准确的度量地图所提供的证据十分有限。我们提出了SpatialScout，一个用于评估视觉语言模型（VLM）智能体主动式物体级3D建图能力的基准。SpatialScout包含80个多样化的室内场景，涵盖3,140个目标物体。给定物体清单、RGB观测、相机标定信息和相机位姿，智能体控制相机探索每个场景，并在共享坐标系中估计物体中心位置。该任务要求智能体在有限的交互预算下协调视角选择、跨视图关联、几何估计和地图修正。评估通过物体尺度的重叠指标，同时衡量定位精度和目标覆盖率。

    arXiv:2610.26659v2 Announce Type: replace-cross  Abstract: Spatial intelligence requires agents to actively acquire visual evidence and integrate observations into a coherent representation of the three-dimensional world. Yet spatial question answering offers limited evidence of whether models can construct accurate metric maps through autonomous exploration. We introduce SpatialScout, a benchmark for evaluating vision-language model (VLM) agents on active object-level 3D mapping. SpatialScout comprises 80 diverse indoor scenes containing 3,140 target objects. Given an object inventory, RGB observations, camera calibration, and camera poses, agents control a camera to explore each scene and estimate object centers in a shared coordinate frame. The task requires coordinating viewpoint selection, cross-view association, geometric estimation, and map revision under bounded interaction budgets. Evaluation jointly measures localization accuracy and target coverage through object-scale overl
    
[^129]: 超越攻击成功率：用于评估迭代式大语言模型越狱攻击的成本感知指标

    Beyond Attack Success Rate: Cost-Aware Metric for Evaluating Iterative LLM Jailbreaks

    [https://arxiv.org/abs/2610.26589](https://arxiv.org/abs/2610.26589)

    提出成本感知评估指标 ASR_STEP，通过对重复调用受害模型的次数进行惩罚，弥补了传统攻击成功率指标在评估迭代式越狱攻击时忽视调用成本的缺陷。

    

    攻击成功率（ASR）在评估针对大语言模型的越狱攻击时，未考虑对受害模型的重复调用。然而，每一次额外的调用都为已部署系统提供了另一次检测攻击并进行干预的机会。我们提出了 ASR_STEP，它在保留二元成功判断的同时，对额外的受害模型调用进行惩罚。我们的评估在 JailBreakBench 上考察了 GCG、带有步数预算的 PAIR，以及单步的 PAP 基线方法，涵盖三个模型系列。经过折扣后，GCG 的分数下降了 7.0-8.7 个百分点，而 PAP 的分数保持不变。对于 PAIR 攻击，我们观察到更宽的折扣范围，介于 3.4-12 个百分点之间，这表明“步数”这一数量因迭代策略而异。同时报告这两种指标可以明确揭示因重复与受害模型交互而产生的折扣。代码已发布于：https://github.com/SuELS-Group/beyond_asr。

    arXiv:2610.26589v2 Announce Type: replace-cross  Abstract: Attack success rate (ASR) evaluates jailbreaks against large language models without accounting for repeated calls to the victim model. Yet each additional call gives the deployed system another opportunity to detect the attack and intervene. We introduce $\mathrm{ASR}_{\mathrm{STEP}}$, which penalizes additional victim-model calls while retaining binary success judgments. Our evaluation considers GCG, PAIR with a step budget, and a single-step PAP baseline on JailBreakBench across three model families. GCG's scores decrease by 7.0-8.7 percentage points after discounting, while PAP's remain unchanged. While for PAIR attack, we see a wider discounting range between 3.4-12 percentage points, showing that the "step" quantity differs per iterative strategy. Reporting both metrics makes the discount due to repeated victim-model interaction explicit. Code is available here: https://github.com/SuELS-Group/beyond_asr.
    
[^130]: VEILED-SH：通过间接语言的脆弱性表达评估数据集（自伤）

    VEILED-SH: Vulnerability Expression through Indirect Language Evaluation Dataset (Self-Harm)

    [https://arxiv.org/abs/2610.26190](https://arxiv.org/abs/2610.26190)

    该论文提出 VEILED-SH 基准数据集（含 10,003 个提示词），首次将文化背景、应用场景和从业者分歧三者结合，用于评估 AI 对自杀与自伤间接表达的单轮回复，并通过专家校准的自动化画像实现代理对齐评分。

    

    AI 系统可能会遗漏间接的、带有文化背景的痛苦表达，或提供与应用场景不匹配的支持。现有的基准测试未能同时解决文化差异、应用场景和从业者意见分歧这三个问题。我们提出了 VEILED-SH，一个用于评估 AI 对自杀、自伤和痛苦间接表达的单轮回复的基准测试及构建方法。该数据集包含 10,003 个提示词，涵盖咨询师聊天、论坛帖子、平台搜索和分诊四种场景，数据来源于新加坡本地资源和公开数据集。经专家验证的多标签分类体系可区分意图、交际功能和回应行为。六个自动化画像分别针对三位同伴支持从业者和三位临床医生进行校准，并为代理对齐评分保留了独立的参考策略。在覆盖度加权的样本上对五个系统进行评估，结果显示各画像生成的参考策略之间的代理对齐程度存在差异。

    arXiv:2610.26190v2 Announce Type: replace  Abstract: AI systems may miss indirect, culturally situated distress or provide support mismatched to the application context. Existing benchmarks do not jointly address cultural variation, application context, and practitioner disagreement. We introduce VEILED-SH, a benchmark and construction methodology for evaluating single-turn responses to indirect expressions of suicide, self-harm, and distress. Its 10,003 prompts span counselor chat, forum posts, platform search, and triage, drawing on Singapore-grounded sources and public datasets. An expert-validated multi-label taxonomy distinguishes intent, communicative function, and response behavior. Six automated profiles, individually calibrated to three peer-support practitioners and three clinicians, retain separate reference policies for proxy-alignment scoring. Evaluating five systems on a coverage-enriched sample reveals variation in proxy alignment across profile-generated reference polic
    
[^131]: 超越其依据而存续的记忆：面向工具使用型LLM智能体的依赖锚定记忆

    Memories That Outlive Their Reasons: Dependency-Tethered Memory for Tool-Using LLM Agents

    [https://arxiv.org/abs/2610.25123](https://arxiv.org/abs/2610.25123)

    该论文提出依赖锚定记忆方法，通过在写入时记录每条结论所依赖的确切版本、在读取时自动撤回存储中任何支撑已失效的结论，解决了工具使用型LLM智能体因上游依赖失效而依赖过时记忆做出错误发布决策的问题。

    

    智能体如今在网络上行动，通过注册表API发布软件包、模型和数据集，而记住过去的验证结果可以为它们省去重复工作。但任何结论都建立在前提之上。当签名密钥被撤销时，构建于其上的一切都会失效，然而安全公告中只提及密钥本身，其依赖项的旧结论可能永远不会与该公告共享同一个检索窗口。这种失效的非局部性使记忆成为负担：在长时间的模拟对话中，一个读取原子事实记忆的智能体会发布其中62%的对象——这些对象的故障源于上游，而每一条支撑记录在写入时都是真实的。然而，即使失效不是局部的，推导却是局部的。Tether方法在写入时记录每个结论所源自的确切版本，并在读取时撤回存储中任何其支撑已失效的结论；智能体读取的字段与之前完全相同。在Fallout（一个带有假设盲模拟的版本化世界生成器）上……

    arXiv:2610.25123v2 Announce Type: replace  Abstract: Agents now act on the Web, publishing packages, models and datasets through registry APIs, and remembering past verifications spares them the work. But a verdict rests on premises. When a signing key is revoked, everything built on it stops passing, yet the advisory names only the key, and the old verdicts of its dependents may never share a retrieval window with it. This non-locality of invalidation turns memory into a liability: in long simulated chats, an agent reading a memory of atomic facts publishes 62% of the objects whose failure lies upstream, each backed by a record that was true when written. Yet derivation is local even when invalidation is not. Tether records, at write time, the exact versions each verdict derives from, and at read time withdraws any verdict whose support has failed anywhere in the store; the agent reads the same fields as before. On Fallout, a generator of versioned worlds with hypothesis-blind simulat
    
[^132]: 超越召回饱和：面向工业法律检索的难负样本重排序器训练规模化

    Beyond Recall Saturation: Scaling Hard-Negative Reranker Training for Industrial Legal Retrieval

    [https://arxiv.org/abs/2610.24478](https://arxiv.org/abs/2610.24478)

    在中文法律检索中，当嵌入召回趋于饱和后，采用“嵌入挖掘到自挖掘”的难负样本分阶段训练流水线是重排序器训练的最佳权衡方案，且可将模型从0.6B扩展至4B。

    

    工业检索系统通常通过扩大嵌入候选池或采用复杂的训练流程来提升检索质量。我们证明，在足够大但切实可行的候选池规模下，嵌入召回呈现边际收益递减，这促使我们更深入地审视重排序器的质量。在中文法律检索基准上，嵌入模型的top-300召回已进入高召回运行区间，但相关证据仍可能排在输出截断线之下。我们在对齐的检索协议下（即在候选检索完成后评估重排序效果）比较了用于重排序器训练的多种难负样本策略，包括嵌入挖掘、自挖掘和动态负样本等变体。在所评估的分阶段配置中，“嵌入挖掘到自挖掘”的流水线提供了有利的权衡；额外的动态阶段并不能改善所有留出集指标。所选流水线可将重排序器从0.6B规模扩展到4B规模。相对于原生4B重排序器……

    arXiv:2610.24478v2 Announce Type: replace  Abstract: Industrial retrieval systems often improve quality by enlarging embedding candidate pools or adopting complex training pipelines. We show that, under a sufficiently large but practical candidate pool, embedding recall exhibits diminishing returns, motivating a closer examination of reranker quality. On Chinese legal retrieval benchmarks, embedding top-300 reaches a high-recall operating region, while relevant evidence may still be ranked below the output cutoff. We compare hard-negative strategies for reranker training, including embedding-mined, self-mined, and dynamic-negative variants, under aligned retrieval protocols that evaluate reranking after candidate retrieval. Among the evaluated staged configurations, an embedding-to-self-mined pipeline provides a favorable trade-off; additional dynamic stages do not improve all held-out metrics. The selected pipeline scales from a 0.6B to a 4B reranker. Relative to the native 4B reranke
    
[^133]: 大语言模型中的多元对齐：目标、人类参照与偏好表征

    Pluralistic Alignment in Large Language Models: Objectives, Human References, and Preference Representations

    [https://arxiv.org/abs/2610.24471](https://arxiv.org/abs/2610.24471)

    该论文提出了一个双层分类框架来系统梳理大语言模型多元对齐研究，第一层区分三种对齐目标（表征多元观点的Overton多元性、适应特定视角的可引导多元性、匹配人群分布的分布多元性），第二层区分三类人类参照（聚合人群、国家和地区、子群体），并将偏好表征、实现机制等视为贯穿性设计选择。

    

    多元对齐旨在表征人类偏好中有意义的差异，而不是将异质的判断坍缩为单一目标。然而，现有工作常常混淆三个不同的问题：模型应展现什么样的多元行为、谁的偏好定义了参照标准，以及什么样的证据足以证明成功。我们围绕一个双层分类体系来组织相关文献。第一层区分三种多元对齐目标：表征一系列得到人类支持的观点（Overton多元性）、忠实地适应特定观点或偏好条件（可引导多元性），以及匹配目标人群中所观察到的回应分布（分布多元性）。第二层区分三类人类参照：聚合人群、国家和地区、以及子群体。偏好表征、实现机制、模态和证据质量则被视为贯穿各层的设计选择，而非附加特性。

    arXiv:2610.24471v2 Announce Type: replace-cross  Abstract: Pluralistic alignment aims to represent meaningful variation in human preferences rather than collapse heterogeneous judgments into a single target. Yet existing work often mixes three distinct questions: what pluralistic behavior a model should exhibit, whose preferences define the reference, and what evidence is sufficient to demonstrate success. We organize the literature around a two-layer taxonomy. Layer 1 distinguishes representing a range of human-supported viewpoints (Overton Pluralism), adapting faithfully to a specified perspective or preference condition (Steerable Pluralism), and matching the response distribution observed in a target population (Distributional Pluralism). Layer 2 distinguishes aggregate populations, countries and regions, and subgroups as human references. Preference representation, implementation mechanism, modality, and evidence quality are treated as cross-cutting design choices rather than addi
    
[^134]: GraphForge：基于图锚定工作区合成训练可执行任务的工作智能体

    GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis

    [https://arxiv.org/abs/2609.38923](https://arxiv.org/abs/2609.38923)

    GraphForge 提出了一种基于证据图的框架，将训练工作智能体所需的任务与验证标准都锚定在真实文件构建的工作区上，从而合成兼具真实性、多样性和可验证性的高质量任务数据。

    

    工作智能体需要阅读多样化的文件、协调使用工具并产出交付成果。训练这类智能体需要基于大量真实文件且结果可验证的任务，但目前很少有数据合成流程能够生成这种类型的数据。现有流程要么用模型生成文件，因而缺乏真实性和多样性；要么在真实文件上构建任务却没有针对任务的验证器，导致结果质量无法检验。我们提出 GraphForge，一个基于证据图的框架，将任务及其验证都锚定在真实文件之上。GraphForge 从以职业为依据的种子出发以实现受控的多样性，为每个种子组装一个由真实文件构成的工作区，并在文件之间的关系上构建证据图。由于任务描述和评分标准都源自该证据图，任务要求均有工作区文件作为支撑，且每条评分标准都锚定到验证该标准所需的文件上。初始的 rollout 试运行进一步检验任务的可执行性，并

    arXiv:2609.38923v1 Announce Type: new  Abstract: Working agents need to read diverse files, coordinate tools, and produce deliverables. Training such agents requires tasks built on many real files with verifiable results, but few pipelines exist to synthesize this kind of data. Existing pipelines either generate files with models, which lack realism and diversity, or build tasks on real files without task-specific verifiers, leaving result quality unchecked. We introduce GraphForge, an evidence-graph based framework that grounds both the task and its verification in real files. Starting from occupation-grounded seeds for controlled diversity, GraphForge assembles a workspace of real files for each seed and builds an evidence graph over their relations. Since the task statement and rubrics are both derived from this graph, task requirements are backed by the workspace files and each criterion is anchored to the files needed to verify it. An initial rollout further tests executability, a
    
[^135]: 论大语言模型智能体的行为特质

    On the Behavioral Traits of LLM Agents

    [https://arxiv.org/abs/2609.32776](https://arxiv.org/abs/2609.32776)

    提出A-B-D方法，从345,667条真实世界轨迹中自下而上提取行为特征，基于智能体的实际行为（而非自我报告或LLM评分）来推断其稳定的人格特质。

    

    arXiv:2609.32776v2 公告类型：替换。摘要：用户越来越多地将不同的AI智能体描述为各具特点的共事同事。AI人格研究旨在通过赋予智能体类似人类的“特质”来量化这种印象。然而，现有的测量方法存在不足：模型的自我报告（S-data）与其真实行为不一致，而来自LLM评判者的他人评分（I-data）成本高昂、难以规模化，且覆盖的日常场景有限。在本文中，我们提出A-B-D方法，从行为数据（B-data）中自下而上地推断特质，即智能体如何作用于环境并与用户交流，这些信息记录在现有的轨迹数据中。从涵盖80个模型、12个任务和50个执行框架的345,667条真实世界轨迹中，我们提取了318个候选特征，这些特征既捕捉了智能体在每一步采取的行动（功能性），也捕捉了伴随行动的语言表达（语言性）。我们仅保留了那些展现出实例级稳定性、跨任务一致性和模型区分性的特征。

    arXiv:2609.32776v2 Announce Type: replace  Abstract: Users increasingly describe different AI agents as distinct colleagues to work with. AI personality research aims to quantify such impressions by attributing human- like "traits" to agents. However, existing measures fall short: models' self-reports (S-data) diverge from their actual behavior, while informant ratings from LLM judges (I-data) are costly to scale and cover few everyday scenarios. In this paper, we propose A-B-D to infer traits bottom-up from behavioral data (B-data), namely how agents act on their environment and communicate with users, as recorded in existing trajectories. From 345,667 real-world trajectories spanning 80 models, 12 tasks, and 50 harnesses, we extract 318 candidate features that capture both the actions an agent takes at each step (functional) and the language accompanying them (linguistic). We retain only features that show instance-level stability, cross-task consistency, and model discriminability. 
    
[^136]: 评判者是其孪生兄弟吗？后训练使模型的写作更可预测，但作为评判者，其品味几乎不会偏向可预测的写作

    Is the Judge Its Twin? Post-training makes a model's writing more predictable but barely moves its taste, as a judge, toward predictable writing

    [https://arxiv.org/abs/2609.32196](https://arxiv.org/abs/2609.32196)

    该研究发现后训练（SFT和DPO）确实使语言模型的写作变得更加可预测，但这种可预测性偏好并不会迁移到模型作为评判者的角色——训练后的模型在评判故事时几乎不偏向可预测的写作。

    

    语言模型如今经常给其他语言模型评分。如果后训练使模型的写作更可预测，它是否也会教导同一个模型（作为评判者）奖励可预测的写作？我们直接测试了这一点。我们跟踪两个开放的70亿参数模型家族——OLMo-2和Zephyr——从它们的基础模型经过监督微调（SFT）和直接偏好优化（DPO）的完整过程，并在每个阶段测量它们作为短篇故事作者和故事对评判者的表现。作为作者，模型如人们担心的那样发生了漂移：每个家族训练后的模型按基础、SFT、DPO的顺序，在全部十个提示中都发现自己各阶段生成的故事更加熟悉，而另一个家族的模型则发现训练后生成的故事每token的惊讶度分别降低了7.0至9.2个百分点（OLMo-2）和24至25个百分点（Zephyr）。但作为评判者，它们几乎不偏向可预测的写作。当被问及哪个故事更好时——一个所有评判者都能通过能力检验的问题——没有任何训练后的评判者的偏好表现出对可预测写作的倾向。

    arXiv:2609.32196v2 Announce Type: replace  Abstract: Language models now routinely grade other language models. If post-training makes a model's writing more predictable, it may also teach the same model, as a judge, to reward predictable writing. We test this directly. We follow two open 7-billion-parameter model families, OLMo-2 and Zephyr, from their base models through supervised fine-tuning (SFT) and direct preference optimization (DPO), and measure every stage as a writer of short stories and as a judge of story pairs. As writers, the models drift as feared. Each family's trained model finds its stages' stories more familiar in the order base, SFT, DPO, in all ten prompts, and a model from the other family finds the trained stories 7.0 to 9.2 percent (OLMo-2) and 24 to 25 percent (Zephyr) less surprising per token. As judges, they barely move toward predictable writing. Asked which story is better, a question every judge passes a competence check on, no trained judge's preference
    
[^137]: 办公室规模验证搜索中的算子包、提议者强度与构造型家族平台期

    Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search

    [https://arxiv.org/abs/2609.29636](https://arxiv.org/abs/2609.29636)

    该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。

    

    验证搜索是指语言模型提出程序、硬评估器对其进行评分、选择机制保留最优解的过程，这种方法近来已推动了数学纪录的进展；但对提议者侧组件的受控消融实验仍然罕见。我们在办公规模上（笔记本电脑上运行的30B本地模型，每次运行120-600个验证样本）对最小化的FunSearch风格循环进行了仪器化，采用三种算子包：模型自行编写并携带的示意图式笔记本（代替逐字复制的精英解）、命名障碍、以及对已发现构造的行为排斥。在来自公共仓库的九个构造问题上，带两次重复的完整2³因子设计在名义两阶段分析中支持主要对比：该组合缩小了更多从种子解到纪录的差距（+0.196；名义合并p=0.023，阶段组合p≈0.08；每问题效应中位数为+0.045）。排斥机制在各处都提高了构造哈希多样性（p=0.0039；部分属于操纵检查）（注：原文摘要至此处截断）。

    arXiv:2609.29636v1 Announce Type: cross  Abstract: Verified search, in which a language model proposes programs, a hard evaluator scores them, and selection keeps the best, has recently moved mathematical records; controlled ablations of the proposer-side components remain rare. We instrument a minimal FunSearch-style loop at office scale (a 30B local model on a laptop, 120-600 verified samples per run) with three operator packages: a schematic notebook the model writes and carries instead of verbatim elites, a named obstacle, and behavioural repulsion from constructions already found. On nine construction problems from a public repository, the complete 2^3 factorial with two replicates favours the primary contrast in a nominal two-stage analysis: the composition closes more of the seed-to-record gap (+0.196; nominal pooled p=0.023, stage-combination p~0.08; median per-problem effect +0.045). Repulsion raises construction-hash diversity everywhere (p=0.0039; partly a manipulation check
    
[^138]: 维基数据中人类条目覆盖潜在偏差的初步评估

    Initial Evaluation of Potential Bias in Coverage of Humans in Wikidata

    [https://arxiv.org/abs/2609.22375](https://arxiv.org/abs/2609.22375)

    本文开发了一个开源审计平台，对Wikidata中超过600万个人类条目在性别、地理、族裔、职业等多维度上的人口代表性偏差进行了系统性评估，发现声明性别的条目中女性仅占28.71%。

    

    引言。像Wikidata（维基数据）这样的开放协作知识图谱日益成为智能体人工智能、信息检索和语言建模系统的基础，因此对其人口代表性和整体公平性进行系统性审计已成为一项迫切的研究任务。方法。本文提出了一个开源审计平台，该平台通过QLever摄取了超过1000万条语句绑定，涵盖Wikidata上超过600万个人类条目，并评估了性别、性取向、地理分布、出生地城市化程度、族裔、标签/描述/别名的多语言覆盖、职业以及这些属性的若干交叉组合的代表性。该平台利用卡方拟合优度检验、95% Wilson得分置信区间和差异比率进行分析，并结合Rubin的缺失机制分类法加以解读。结果。在Wikidata中声明了性别的所有人类条目中，女性占28.71%（置信区间±0.04）。38.26%的人……（摘要在此处被截断）

    arXiv:2609.22375v1 Announce Type: cross  Abstract: Introduction. Open collaborative knowledge graphs such as Wikidata increasingly ground agentic artificial intelligence, information retrieval, and language modeling systems, making systematic auditing of their demographic representation and overall equity a research imperative. Methods. Herein, we present an open-source auditing platform that ingests over 10 million statement bindings representing over 6 million humans on Wikidata via QLever, and evaluates representation of gender, sexual orientation, geography, birthplace urbanicity, ethnicity, multilingual coverage of labels, descriptions, and aliases, occupation, and select intersectional pairs of these entities. It does so by making use of Chi-square goodness-of-fit tests, 95% Wilson-score confidence intervals, and disparity ratios, in light of Rubin's missingness taxonomy. Results. Women accounted for 28.71% (CI +/-0.04) of all humans in Wikidata with a stated gender. 38.26% of hu
    
[^139]: 将均值推向已知的良好水平而非超越它：推理时干预与权重合并在开放式生成中能带来什么

    Moving the Mean Toward the Known Good, Not Beyond It: What Inference-Time Interventions and Weight Consolidation Buy in Open-Ended Generation

    [https://arxiv.org/abs/2608.28886](https://arxiv.org/abs/2608.28886)

    该研究发现在开放式生成任务中，通过“生成-验证-选择-LoRA合并”循环在自身验证成功的样本上训练，能显著提升模型输出的平均质量并经三重重复实验验证其稳健性，但最佳候选仅能收敛到经典启发式算法的水平而无法超越。

    

    一个生成循环能从学习自身经过验证的成功中获得什么？在针对在线装箱问题的“生成-验证-选择-LoRA合并”循环中，基于价值过滤候选样本的训练使模型在保留变体上的输出向更优价值偏移（超额量-1.7分，p=0.008；相比随机合并对照组为-3.1，p=0.004），而观察到的最佳候选收敛至经典启发式算法的水平且不再超越。一项确认实验组使用新随机种子和一个仅读取一次的从未接触过的保留集，将整个流程重复了三次：三个谱系中的均值几乎完全一致（-2.0、-1.8、-1.9），且在按保留变体内聚合后，全部七个可评估变体均支持合并方法（p=0.008）。观察到的最佳候选恰好移动到经典启发式算法的水平（三个谱系中均为0.021028，吸引条件与随机对照亦然），且从未超越该水平。一项匹配的SFT

    arXiv:2608.28886v1 Announce Type: cross  Abstract: What does a generation loop gain from learning on its own verified successes? In cycles of generate, verify, select and LoRA-consolidate on online bin packing, training on value-filtered candidates shifts what the model writes on held-out variants toward value (-1.7 points of excess, p=0.008; -3.1 against a random-consolidation control, p=0.004) while the best observed candidate converges to the classic heuristic's level and no further. A confirmation battery replicates the whole procedure three times, with fresh seeds and a never-consulted held-out set read exactly once: the mean was nearly identical in all three lineages (-2.0, -1.8, -1.9), and after aggregating within held-out variant all seven evaluable variants favored consolidation (p=0.008). The best observed candidate moved to the classic heuristic's level, exactly (0.021028 in all three lineages, for attract and for the random control alike), and never beyond it. A matched SFT
    
[^140]: 打断循环：周期性主题变化提升基础语言模型的评判惊喜度与连贯性

    Interrupting the Loop: Periodic Subject Changes Raise Judged Surprise and Connection in Base Language Models

    [https://arxiv.org/abs/2608.19893](https://arxiv.org/abs/2608.19893)

    该论文发现，在基础语言模型的生成循环中，周期性注入新主题（打断）是提升文本评判惊喜度和连贯性的核心机制，且效果显著优于仅靠习惯化或其他操作。

    

    摘要：一个基础语言模型在没有任务的情况下产生的新颖性从何而来，而一个LLM评判者能从长文本流中真正看到什么？我们在三个基础模型上，通过24种条件拆解了一个认知启发的生成循环。其大部分效果集中在一个操作上：每隔几百个标记注入一个新主题（打断），进入一个字面重复被抑制（习惯化）的文本流。我们仅评判生成的文本窗口，以命题为单位（n=10），并使用一个经过可重复性测量的评判者，与第二个评判者家族及人类读者进行比较。在该协议下，打断使评判惊喜度比单独习惯化提高了1.2至1.4分，连贯性提高了0.8分。一个要求连续性的连接词会损害效果；一个单独的段落分隔符在新文本上没有可检测的附加效果；重置上下文至少与保留上下文效果相当；并且一项预先注册的复制实验在新命题上确认了主要对比。

    arXiv:2608.19893v1 Announce Type: cross  Abstract: Where does the novelty a base language model produces with no task come from, and what can an LLM judge of a long stream actually see? We dismantle a cognitively inspired generation loop over 24 conditions on three base models. Most of its effect lives in one operation: a new subject injected every few hundred tokens (an interruption) into a stream whose literal repetition is damped (habituation). We judge windows of generated text only, with the premise as the unit (n=10) and a judge measured for repeatability, against a second judge family and against human readers. Under that protocol the interruption raises judged surprise by 1.2 to 1.4 points and connection by 0.8 over habituation alone. A connective that asks for continuity hurts; a bare paragraph break adds nothing detectable on fresh text; a reset context does at least as well as a kept one; and a pre-registered replication on new premises confirms the primary contrast. Three t
    
[^141]: DFM Mimir v1：一个仅使用许可后训练数据、在1B参数下实现前沿性能的开源HRM模型

    DFM Mimir v1: An Open HRM Delivering Frontier Performance at 1B Parameters Using Only Permissible Post-Training Data

    [https://arxiv.org/abs/2608.13517](https://arxiv.org/abs/2608.13517)

    Mimir v1是一个10亿参数的HRM架构语言模型，仅使用许可数据训练，在英语和丹麦语上实现了前沿性能，并超越了同尺寸及更大尺寸的模型。

    

    当前大型语言模型的开发依赖于庞大且通常未经许可的数据集，这为致力于开源和道德数据来源的研究人员设置了高门槛。我们介绍了Mimir v1，这是一个基于层次推理模型（HRM）架构的10亿参数语言模型，从零开始训练，在英语方面表现出极具竞争力的性能，并在仅使用许可后训练数据的情况下，为丹麦语设立了新的最优水平。该模型在161个数据集的混合上训练，在英语、数学与代码以及丹麦语的20个基准测试中，Mimir v1超越了原始的HRM-Text 1B，并与更大的前沿模型（如Qwen 3.5 4B和Gemma 4 E2B）竞争。该模型可在Hugging Face Hub上获取：https://huggingface.co/danish-foundation-models/DFM-Mimir

    arXiv:2608.13517v1 Announce Type: cross  Abstract: Current large language model development relies on massive, often non-permissible datasets, creating a high barrier for researchers committed to open-source and ethically sourced data. We introduce Mimir v1, a 1-billion-parameter language model based on the Hierarchical Reasoning Model (HRM) architecture, that is trained from scratch and delivers highly competitive performance for English and sets a new state of the art for Danish using only permissible post-training data. Trained on a mixture of 161 datasets, Mimir v1 outperforms the original HRM-Text 1B and competes with larger frontier models like Qwen 3.5 4B and Gemma 4 E2B, tested across 20 benchmarks for English, Math & Code and Danish. The model is available on the Hugging Face Hub: https://huggingface.co/danish-foundation-models/DFM-Mimir
    
[^142]: 橡皮筋丢弃法：一种缓解基于评分标准奖励强化学习中奖励黑客问题的简单方法

    Rubric Dropout: A Simple Way to Mitigate Reward Hacking in Rubric-as-Reward RL

    [https://arxiv.org/abs/2608.11669](https://arxiv.org/abs/2608.11669)

    本文提出一种简单方法（Rubric Dropout）来缓解基于评分标准的强化学习中的奖励黑客问题，并通过实验证实训练评判员与黄金评判员分数在训练中分化的现象，为提升模型鲁棒性提供新思路。

    

    摘要：基于评分标准的强化学习，即由LLM评判员根据一系列标准进行评分，已成为对没有确定性答案的任务进行语言模型后训练的标准方法。然而，评分标准是质量的固定代理，永远无法完全描述质量，长期针对其训练的策略会学会利用这一差异。我们直接对此进行了测量。使用组相对策略优化（GRPO）在医学和科学评分标准上训练Qwen3-8B，并使用训练评判员和更强的黄金评判员对分布外（OOD）基准进行评分，我们发现这两个分数在训练过程中出现分歧。训练评判员的分数持续上升，而黄金评判员的分数达到峰值后下降，在HealthBench-Hard上下降3分，在ResearchQA上下降22分。如果评判员存在固定偏差，黄金曲线只会发生恒定偏移，而不会在训练分数上升时下降，因此这种分歧是奖励黑客问题，而非评判员偏差。

    arXiv:2608.11669v1 Announce Type: cross  Abstract: Reinforcement learning against rubrics, lists of criteria graded by an LLM judge, has become a standard way to post-train language models on tasks with no deterministic answer. The rubric, however, is a fixed proxy for quality, never a complete description of it, and a policy trained against it long enough will learn to exploit the difference. We measure this directly. Training Qwen3-8B with Group Relative Policy Optimization (GRPO) on medical and science rubrics and grading out-of-distribution (OOD) benchmarks with both the training judge and a stronger gold judge, we find that the two scores diverge during training. The training judge's score keeps climbing while the gold judge's score peaks and then falls, by 3 points on HealthBench-Hard and by 22 points on ResearchQA. A judge with a fixed bias would shift the gold curve by a constant, not send it down while the training score rises, so the divergence is reward hacking, not judge no
    
[^143]: 当自我一致性适得其反：多数投票损害小型语言模型在多数硬科学问题上的表现

    When Self-Consistency Backfires: Majority Vote Hurts the Majority of Hard Science Problems for Small LLMs

    [https://arxiv.org/abs/2608.11403](https://arxiv.org/abs/2608.11403)

    本研究发现，在GPQA Diamond基准上，多数投票的自我一致性方法反而降低了小型语言模型在大多数硬科学问题上的准确率，并预注册验证了这一反直觉现象。

    

    arXiv:2608.11403v1 公告类型：新 摘要：通过多数投票实现的自我一致性（SC）是一种广泛使用的推理时计算扩展方法：采样N条思维链，返回多数答案。在完整的GPQA Diamond基准测试（198个研究生级科学问题）上，多数投票在两个不同家族的指令微调模型上降低了大多数问题的逐题准确率：Qwen2.5-7B的56.6%问题和Llama-3-8B的65.7%问题，其中Qwen为主要演示模型，Llama从接近随机的基线证实了这一趋势。该效应在47个探索性问题上观察到后，在151个问题的确认性分割上进行了预注册，所有四项确认性假设均通过。一个网格神谕（将每个问题路由至{1, 2, 4, 8, 16, 32, 64}中的最佳N值）标志着理论上限，Qwen在N=1基础上高出14个准确率点，Llama高出17个点，但这一神谕上限需要真实标签而非可部署方法。没有无验证器的门控可以达到此效果。

    arXiv:2608.11403v1 Announce Type: new  Abstract: Self-consistency (SC) via majority vote is a widely used way to spend inference-time compute: sample N chains of thought, return the plurality answer. On the full GPQA Diamond benchmark (198 graduate-level science questions), majority voting reduces per-problem accuracy on a majority of problems for two instruction-tuned models from different families: 56.6% of problems for Qwen2.5-7B and 65.7% for Llama-3-8B, with Qwen the primary demonstration and Llama corroborating the direction from a near-chance baseline. The effect was pre-registered on a 151-problem confirmatory split after being observed on 47 exploratory problems, and all four confirmatory hypotheses passed. A grid oracle that routes each problem to the best N across {1, 2, 4, 8, 16, 32, 64} marks a theoretical upper bound 14 accuracy points above N = 1 for Qwen and 17 for Llama, an oracle bound requiring ground truth rather than a deployable method. No verifier-free gate reach
    
[^144]: 评估自然语言中的理性缔约

    Evaluating Rational Contracting in Natural Language

    [https://arxiv.org/abs/2608.10475](https://arxiv.org/abs/2608.10475)

    该论文提出了一个理性缔约框架和评估套件ContractSim，用于量化评估AI智能体在不确定多步骤环境中协商与履行自然语言契约时的理性与合作能力。

    

    基于语言的AI智能体的出现，有望改变机器经济活动的范围。这类智能体不再只是提出竞价或遵循硬编码的协议，而是能够用自然语言进行协商并执行协议。然而，目前针对这些能力的大多数评估都集中在一次性交易或简单的经济博弈上，而尚未涉及语言所能表达的、时间上延展的、有条件依赖的以及不完全契约这一丰富的空间；此外，这些评估往往只关注原始利润，而没有衡量可信缔约所需的品质。为此，我们提出了一个理性框架，用以规范智能体在不确定的多步骤环境中应如何协商和履行自然语言契约。基于该框架，我们开发了用于量化理性与合作博弈行为的指标和基线。我们将该框架实例化为ContractSim——一个评估套件，其中两名玩家协商并执行一项多步骤契约（摘要在此处截断）。

    arXiv:2608.10475v2 Announce Type: replace-cross  Abstract: The emergence of language-based AI agents promises to transform the scope of machine economic activity. Instead of just proposing bids or following hard-coded protocols, such agents can be used to negotiate and execute agreements in natural language. However, most evaluations of these abilities have focused on one-off exchanges or simple economic games, leaving open the rich space of time-extended, contingent, and incomplete contracts expressible in language; they also focus on raw profit, without measuring the qualities required for trustworthy contracting. We address this by formulating a rational framework for how agents should negotiate and perform natural language contracts in uncertain multi-step environments. Using this framework, we develop metrics and baselines for quantifying rational and cooperative play. We instantiate our framework in ContractSim, an evaluation suite where two players negotiate and execute a multi-
    
[^145]: 超越自然度：在语言学奠基的维度上探测自动文本转语音评估器

    Beyond Naturalness: Probing Automated Text-To-Speech Evaluators on Linguistically Grounded Dimensions

    [https://arxiv.org/abs/2608.09930](https://arxiv.org/abs/2608.09930)

    该论文将TTS评估中的“自然度”解构为10个语言学感知维度，构建了首个维度级元评估基准（860条由语言学家标注的语音），并揭示MOS预测器和Audio-LLM评判者各自仅能捕捉人类语音感知的部分维度，无法全面覆盖。

    

    自动文本转语音（TTS）评估方法（平均意见得分（MOS）预测器和音频大语言模型（Audio-LLM）评判者）被期望能够反映人类的感知，但它们在多大程度上能捕捉到听者实际感知到的语音的不同方面尚不清楚。我们将“自然度”解构为一个涵盖10个不同感知维度的、以语言学为基础的标注体系，并利用该体系构建了首个面向TTS的维度级元评估基准，包含由受过训练的语言学评分员标注的860条语音。对四个MOS预测器和四个Audio-LLM评判者的基准测试结果表明：MOS预测器对韵律和副语言层面的声学信号级属性更为敏感，而Audio-LLM评判者则表现出选择性的、依赖提示词的检测能力，更倾向于捕捉词级错误，且无法在所有维度上泛化。两类评估方法都无法可靠地捕捉……（摘要原文在此处截断）

    arXiv:2608.09930v2 Announce Type: replace-cross  Abstract: Automated Text-to-Speech (TTS) evaluation methods (Mean Opinion Score (MOS) predictors and Audio Large Language Model (Audio-LLM) judges) are expected to reflect human perception, yet it is unclear how well they capture the distinct aspects of speech that listeners actually perceive. We deconstruct "naturalness" into a linguistically grounded annotation schema spanning 10 distinct perceptual dimensions, and use it to construct the first dimension-level meta-evaluation benchmark for TTS, comprising 860 utterances annotated by trained linguist raters. Results from benchmarking four MOS predictors and four Audio-LLM judges reveal that MOS predictors are more sensitive to acoustic signal-level properties at the prosodic and paralinguistic levels, while Audio-LLM judges show selective, prompt-dependent detection that favors word-level errors and does not generalize across all dimensions. Neither class reliably captures a breadth of 
    
[^146]: TORUS：面向统一音频模型的统一闭环基准测试

    TORUS: A Unified Closed-Loop Benchmark for Unified Audio Models

    [https://arxiv.org/abs/2607.28896](https://arxiv.org/abs/2607.28896)

    提出TORUS——首个音频原生的统一闭环基准，用于评估统一音频模型能否理解自己生成或编辑的音频，发现最佳模型的自一致性仅约52%-54%，揭示了当前统一音频模型在理解-生成自一致性上的重大缺陷。

    

    摘要（arXiv:2607.28896v2，公告类型：replace-cross）：能够进行音频理解、音频生成以及日益增多的音频编辑的统一音频模型正在迅速涌现。然而，每种能力目前都在专门的基准测试中被孤立评估，一个基本问题仍未得到解答：这些模型能否理解它们自己生成/编辑的音频，即它们是否表现出音频理解-生成自一致性（UGC_OWN）？我们提出了TORUS，这是一个音频原生的统一基准，用于对统一音频模型进行全面的闭环评估，包含1,053个问题（117个测试），涵盖21个任务和8种音频领域设置，跨越所有3种音频类型。我们在客观和主观指标上评估了统一模型以及由专门模型组成的级联系统。结果显示，最好的统一模型只能回答关于其自身生成音频的52.4%的问题，而gpt-audio能回答53.6%的问题。我们假设真正统一的音频智能应具有高度自一致性，因此提出……

    arXiv:2607.28896v2 Announce Type: replace-cross  Abstract: Unified audio models capable of audio understanding, audio generation and, increasingly, audio editing are proliferating rapidly. However, each capability is evaluated in isolation on specialized benchmarks and a basic question remains unanswered: can these models understand the audio they generate/edit i.e. do they exhibit audio understanding-generation self-coherence (UGC_OWN)? We introduce TORUS, an audio-native unified benchmark for holistic, closed-loop evaluation of unified audio models comprising 1,053 questions (117 tests) spanning 21 tasks and 8 audio domain settings across all 3 audio types. We evaluate unified models and cascades of specialized models across objective and subjective metrics. Best unified models are able to answer only 52.4% questions about their generations, while gpt-audio can answer 53.6% questions. We hypothesize that a truly unified audio intelligence should be highly self-coherent and thus posit
    
[^147]: CPU上的上下文感知字幕翻译：以部署为导向的训练、评估与延迟设计

    Context-Aware Subtitle Translation on CPU: Deployment-Driven Training, Evaluation, and Latency Design

    [https://arxiv.org/abs/2607.09957](https://arxiv.org/abs/2607.09957)

    该论文构建了可在观众CPU上以每条约500毫秒完成翻译的英译繁体中文字幕模型，通过部署驱动的训练流程、基于LLM评审的成对比较评估方法以及空闲时间预填充等延迟优化设计，实现了高质量翻译与严格延迟约束的兼顾。

    

    我们研究面向免费浏览器扩展的英语到繁体中文（台湾）字幕翻译，模型必须在观看者的CPU上运行，每条字幕的处理时间约为500毫秒。我们基于LMT-60基础模型训练了两个模型：Sotto-1.7B和已部署的Sotto-0.6B，采用了任务专用词表、持续预训练、监督微调、权重插值、logit蒸馏以及在线策略偏好优化等技术。由于字幕的参考译文不可靠，我们通过与固定基线Google Translate的成对比较进行评估，每对译文由两个LLM评审组成的评审团以两种顺序分别评判。在1,820条保留测试字幕上，Sotto-1.7B对基线的胜率为0.576，并在正面对比中胜过四个开源翻译模型，表现略低于一个27B模型。我们将延迟划分为预填充和解码两个阶段：空闲时间预填充，即在等待新字幕到来的同时预先填充提示词中已知的部分……（原文摘要在此处截断）

    arXiv:2607.09957v3 Announce Type: replace  Abstract: We study English-to-Traditional-Chinese (Taiwan) subtitle translation for a free browser extension, where the model must run on the viewer's CPU in about 500 ms per subtitle. We train two models built on the LMT-60 base models, Sotto-1.7B and the deployed Sotto-0.6B, with a task-specific vocabulary, continued pretraining, supervised fine-tuning, weight interpolation, logit distillation and on-policy preference optimization. Because reference translations of subtitles are unreliable, we evaluate by pairwise comparison against one fixed baseline, Google Translate, with each pair judged in both orders by a panel of two LLM judges. On 1,820 held-out subtitles, Sotto-1.7B has a win rate of 0.576 against the baseline and wins head-to-head against four open translation models, and is slightly below a 27B model. We split latency into prefill and decoding: idle-time prefill, which prefills the known part of the prompt while waiting for the ne
    
[^148]: 视觉-语言模型中视觉信息流动的通路

    Pathways of Visual Information Flow in Vision-Language Models

    [https://arxiv.org/abs/2607.03358](https://arxiv.org/abs/2607.03358)

    该研究发现视觉-语言模型通过“直接通路”和“文本中介通路”两条不同路径处理视觉信息，通路选择取决于任务类型、数据分布和提示设计，且在常用通路被消融时模型能灵活启用文本中介通路作为后备。

    

    我们研究视觉信息在视觉-语言模型（VLMs）中是如何被路由的。通过在受控的合成与自然数据集上使用因果补丁技术，我们发现模型依赖两种不同的通路来解决视觉任务：一种是直接通路，即视觉信息保留在图像token的表示中，并在后续层中被最后一个token读取；另一种是文本中介通路，即视觉信息首先被传递到查询token，然后由最后一个token读取。在三个视觉任务上，我们表明通路的选择依赖于具体任务，且数据分布和提示词设计也可以调节使用哪条通路来解决基于图像的查询。此外，通过注意力敲除和受损输入补丁方法，我们发现这些通路是灵活的——在某些干预下，当常用通路被消融时，模型可以依赖文本中介通路作为后备机制。这一行为统一了相关发现……

    arXiv:2607.03358v2 Announce Type: replace-cross  Abstract: We study how visual information is routed in vision-language models (VLMs). Using causal patching on controlled synthetic and natural datasets, we find that models rely on two distinct pathways to solve visual tasks: A direct pathway, where visual information is retained in image token representations and read out by the final token at later layers, and a text-mediated pathway, where visual information is first transferred to the query tokens and then read out by the final token. Across three visual tasks, we show that pathway selection is task-dependent, and that data distribution and prompt design can also modulate which pathway is used to solve the image-based query. Moreover, using attention knockouts and corrupted-input patching, we find that these pathways are flexible, under certain interventions, models can rely on the text-mediated pathway as a fallback when the usual pathway is ablated. This behavior unifies findings 
    
[^149]: 用于多表格问答的潜在桥接

    Latent Bridges for Multi-Table Question Answering

    [https://arxiv.org/abs/2606.28916](https://arxiv.org/abs/2606.28916)

    该论文提出GRAB框架，将关系数据编码为异构图，并通过查询条件化的潜在标记桥接到完全冻结的LLM（仅训练91M参数的轻量模块），显著提升了多表格问答的性能。

    

    我们提出了GRAB，一个用于表格问答的构造器-编码器-桥接流水线。我们的方法将关系数据提升为异构图，通过消息传递进行编码，并通过一小组以查询为条件的潜在标记将信号传递给大语言模型（LLM）。这为LLM提供了紧凑且与任务相关的结构化表示，同时结合了扁平化的文本。关键在于，LLM保持严格冻结以保留其通用推理能力；我们仅训练轻量级的图编码器和潜在桥接（9100万参数），使整个流水线能够高效训练。我们的流水线显著提升了关系型问答的性能，在要求苛刻的多表格设置中收益最大，为连接关系深度学习与大语言模型提供了一种高效且有原则的方法。

    arXiv:2606.28916v2 Announce Type: replace  Abstract: We introduce GRAB, a constructor-encoder-bridge pipeline for table question answering. Our method lifts relational data into an heterogeneous graph, encodes it via message passing, and transfers the signals to an LLM through a small set of query-conditioned latent tokens. This provides the LLM with a compact, task-relevant structural representation together with the flattened text. Crucially, the LLM remains strictly frozen to preserve its general reasoning capabilities; we train only the lightweight graph encoder and latent bridge (91M parameters), allowing the entire pipeline to be trained efficiently. Our pipeline significantly improves performance on relational Question Answering, with the largest gains in demanding multi-table settings, offering an efficient, principled way to connect relational deep learning with LLMs.
    
[^150]: 超越提示与解析器工程：基于软提示微调的自动化大语言模型评估

    Beyond Prompt and Parser Engineering: Soft-Prompt Tuning for Automated LLM Evaluation

    [https://arxiv.org/abs/2606.12117](https://arxiv.org/abs/2606.12117)

    该论文提出基于软提示微调的自动化LLM评估方法，仅需优化约0.0006%的参数即可让模型适应基准测试格式，从而更准确地反映模型的真实知识水平。

    

    基准测试分数常常无法准确反映大语言模型（LLM）的知识，因为这些测试依赖于模型遵循特定格式要求的能力。这对基座模型尤其不利——这些模型可能知道正确答案，但缺乏将答案按指令格式化的能力，而这种能力通常是在后训练阶段引入的。为了克服这一问题，我们提出了一种基于软提示微调的高效、自动化且与架构无关的模型评估方法。通过在短时间的微调过程中仅优化10个软提示向量（对于7B模型约占0.0006%的参数），我们使模型适应特定的基准测试格式，弥合了格式遵循方面的差距，确保模型的底层知识能够准确反映在基准测试分数中。我们在7个模型和11个数据集上评估了软提示微调方法。结果表明：（a）软提示微调在150步内（约1200个样本）即可使格式遵循能力达到饱和，同时保留模型知识（摘要在此处截断）

    arXiv:2606.12117v2 Announce Type: replace  Abstract: Benchmark scores often misrepresent a large language model's (LLM's) knowledge, because they rely, e.g., on the model's ability to follow specific formatting requirements. This especially penalizes base models that may know the correct answers but lack the ability -- typically introduced in post-training -- to structure them as instructed. To overcome this, we propose an efficient, automated, and architecture-agnostic method for model evaluation based on soft-prompt tuning. By optimizing only 10 soft-prompt vectors (~0.0006% parameters for a 7B model) over a short tuning period, we adapt models to specific benchmark formats, closing gaps in format-following and ensuring that underlying knowledge is accurately reflected in benchmark scores. We evaluate soft-prompt tuning across 7 models and 11 datasets. Our results show that (a) soft-prompt tuning saturates format-following within 150 steps (~1200 samples) while preserving knowledge a
    
[^151]: 攻击生成困惑度：为什么无条件文本评估需要新的指标

    Hacking Generative Perplexity: Why Unconditional Text Evaluation Needs New Metrics

    [https://arxiv.org/abs/2606.08417](https://arxiv.org/abs/2606.08417)

    该论文证明生成困惑度作为无条件文本生成质量指标并不可靠——通过构造无需训练却能“作弊”该指标（低困惑度但文本不连贯）的采样器来揭示其缺陷，并主张采用分布差异和多样性诊断等新指标来评估扩散语言模型。

    

    扩散语言模型已成为语言建模领域中领先的非自回归替代方案。由于其似然值难以直接计算，生成困惑度已成为报告无条件生成质量的事实标准。我们分析了为什么生成困惑度作为生成质量的度量并不可靠。在样本之间，该指标可以通过始终输出评分器最可能的序列而被最小化，而单样本的一元熵无法衡量这种多样性的损失。在单个样本内部，生成困惑度奖励的是在评分器模型下的可预测性，而这并不一定对应于文本质量。为了明确揭示这一缺陷，我们构建了四个无需训练的采样器，它们在LM1B和OpenWebText数据集上、在非退化的一元熵水平下达到甚至超过了训练好的模型的生成困惑度，同时其生成的文本在构造上就是不连贯的。随后，我们使用分布差异和多样性诊断方法对近期的扩散语言模型进行重新基准测试，并比较它们的……

    arXiv:2606.08417v3 Announce Type: replace  Abstract: Diffusion language models (DLMs) have emerged as the leading non-autoregressive alternative to language modeling. Because their likelihoods are intractable, generative perplexity (gen-PPL) has become the de facto standard for reporting unconditional generation quality. We analyze why gen-PPL is unreliable as a measure of generation quality. Across samples, it is minimized by always emitting the scorer's most likely sequence. Per-sample unigram entropy does not measure this loss of diversity. Within a sample, gen-PPL rewards predictability under the scorer, which need not correspond to text quality. To make this failure explicit, we construct four training-free samplers that reach or beat trained-model gen-PPL on LM1B and OpenWebText at non-degenerate unigram entropy, while producing text that is incoherent by construction. We then rebenchmark recent DLMs using distributional discrepancies and diversity diagnostics, and compare their 
    
[^152]: MLEvolve：一个用于自动化机器学习算法发现的自进化框架

    MLEvolve: A Self-Evolving Framework for Automated Machine Learning Algorithm Discovery

    [https://arxiv.org/abs/2606.06473](https://arxiv.org/abs/2606.06473)

    MLEvolve是一个基于大语言模型的自进化多智能体框架，通过渐进式MCGS树搜索实现跨分支信息流与探索-利用的渐进转变，并借助结合领域知识库与动态全局记忆的回顾性记忆机制，实现端到端的机器学习算法自动发现。

    

    大语言模型（LLM）智能体正越来越多地被应用于科学发现和机器学习工程（MLE）等长程任务，在这些任务中，持续的自我进化成为一项关键能力。然而，现有的MLE智能体存在探索孤立、经验复用低效以及缺乏分层控制等问题，这些问题共同阻碍了长程优化。我们提出了MLEvolve，一个基于LLM的自进化多智能体框架，用于端到端的机器学习算法发现。通过将树搜索扩展为渐进式MCGS（Progressive MCGS），MLEvolve通过基于图的引用边实现跨分支信息流，并借助受熵启发的渐进式调度，将搜索从广泛探索逐渐转向聚焦利用。为了让智能体能够随着经验的积累而不断进化，我们引入了回顾性记忆，它将冷启动领域知识库与动态全局记忆相结合，用于任务（摘要在此处截断）

    arXiv:2606.06473v2 Announce Type: replace-cross  Abstract: Large language model (LLM) agents are increasingly applied to long-horizon tasks such as scientific discovery and machine learning engineering (MLE), where sustained self-evolution becomes a key capability. However, existing MLE agents suffer from isolated exploration, inefficient experience reuse, and lack of hierarchical control, which together hinder long-horizon optimization. We present MLEvolve, an LLM-based self-evolving multi-agent framework for end-to-end machine learning algorithm discovery. By extending tree search to Progressive MCGS, MLEvolve enables cross-branch information flow through graph-based reference edges and gradually shifts the search from broad exploration to focused exploitation with an entropy-inspired progressive schedule. To allow the agent to evolve with accumulated experience, we introduce Retrospective Memory, which combines a cold-start domain knowledge base with a dynamic global memory for task
    
[^153]: WaxalNet：面向19种非洲语言自发语音的语音识别基准测试与适配

    WaxalNet: Benchmarking and Adapting Speech Recognition for Spontaneous Speech in 19 African Languages

    [https://arxiv.org/abs/2606.02375](https://arxiv.org/abs/2606.02375)

    该研究构建了覆盖19种非洲语言自发语音的基准测试，发现微调可将最佳零样本模型的词错误率从52.3大幅降至31.5，且在共享编码器条件下LLM解码器在18/19种语言上优于CTC解码器，表明不同模型家族的性能差异更可能源自预训练而非解码器架构。

    

    多语言语音基础模型声称具有广泛的语言覆盖能力，但它们对非洲自发语音的处理效果如何，以及如何最好地进行适配，目前仍不明确。我们在包含19种非洲语言自发语音的WAXAL语料库上，对零样本和微调后的语音识别进行了基准测试，所评测的模型涵盖Whisper、MMS、Omnilingual ASR和SeamlessM4T-v2。微调将覆盖所有19种语言的最佳零样本模型（Omnilingual-7B）的宏观词错误率（WER）从52.3降低至36.2（300M参数的MMS模型）和31.5（配备LLM解码器的Omnilingual模型），不过零样本模型在少数语言上已具备竞争力。利用Omnilingual的共享编码器，我们在固定编码器预训练的条件下比较了CTC解码器与LLM解码器。在MMS与Whisper之间，哪个模型更优取决于具体语系，但在共享编码器的条件下，LLM解码器在19种语言中的18种上取得了更低的WER，因此这种语系间的性能差异更可能来源于预训练……

    arXiv:2606.02375v2 Announce Type: replace  Abstract: Multilingual speech foundation models claim broad language coverage, but how well they handle spontaneous African speech, and how best to adapt them, remains unclear. We benchmark zero-shot and fine-tuned speech recognition on the WAXAL corpus of spontaneous speech in 19 African languages, covering Whisper, MMS, Omnilingual ASR, and SeamlessM4T-v2. Fine-tuning lowers macro word error rate (WER) from 52.3 for the best zero-shot model covering all 19 languages (Omnilingual-7B) to 36.2 for a 300M-parameter MMS model and 31.5 for Omnilingual with an LLM decoder, although zero-shot models are already competitive on a few languages. Using Omnilingual's shared encoder, we compare CTC and LLM decoders with encoder pretraining held fixed. Between MMS and Whisper, the better model depends on the language family, but with a shared encoder the LLM decoder has lower WER on 18 of 19 languages, so the family pattern more likely comes from pretraini
    
[^154]: CARTE：一个用于映射语言模型在法国各地知识的基准

    CARTE: A Benchmark for Mapping Language Model Knowledge Across France

    [https://arxiv.org/abs/2606.01995](https://arxiv.org/abs/2606.01995)

    该论文提出了CARTE基准，通过覆盖法国13个大区、12个主题领域的3,132道多选题，评估大语言模型辨别法国国内细粒度区域差异知识的能力，并引入聚焦各区域语言变异的CARTE-LV子集。

    

    我们介绍了CARTE（文化锚定区域-领土评估），这是一个多选题基准，用于评估大型语言模型（LLM）辨别法国境内细粒度、基于地理且具有区域差异知识的能力。虽然先前的基准主要关注国家层面的文化理解，但它们在很大程度上忽略了国家内部的差异，以及区分密切相关区域背景的需求。CARTE通过引入3,132个问题填补了这一空白，这些问题涵盖法国13个大都市大区，并涉及12个主题领域，包括文化、语言、人口统计、经济、环境和交通出行。所有问题经过了四个阶段的验证：人工审核、LLM作为评判者、基于模型的难度校准，以及第二轮人工审核。我们进一步引入了CARTE-LV，这是一个针对法国各地区语言变异的子集，能够对语言相关知识进行集中评估。

    arXiv:2606.01995v2 Announce Type: replace  Abstract: We introduce CARTE (Culturally Anchored Regional-Territorial Evaluation), a multiple-choice benchmark for evaluating the ability of large language models (LLMs) to discriminate fine grained, geographically grounded and regionally differentiated knowledge within France. While prior benchmarks focus on national-level cultural understanding, they largely overlook intra-country variation and the need to distinguish between closely related regional contexts. CARTE addresses this gap by introducing 3,132 questions spanning the 13 metropolitan regions of France and covering 12 thematic domains, including culture, language, demographics, economy, environment, and mobility. Questions underwent four-stage validation: human review, LLM-as-a-judge, model-based difficulty calibration, and a second human review. We further introduce CARTE-LV, a subset targeting Linguistic Variation across French regions, enabling focused evaluation of language-rel
    
[^155]: 《测量、定位与消融大语言模型中的对齐签名》

    Measuring, Localizing, and Ablating Alignment Signatures in LLMs

    [https://arxiv.org/abs/2605.30526](https://arxiv.org/abs/2605.30526)

    该论文发现后训练会使大语言模型的生成文本偏离人类语料风格并呈现出可被AI检测器识别的类AI特征，并提出了无需训练的PASTA方法，通过从残差对比中估计对齐签名方向并在解码时消融该方向，从而定位并去除模型内部的这一AI风格签名。

    

    经过对齐的大语言模型通常表现出一种可被识别的类AI风格，但这种风格与后训练及模型内部表征之间的联系仍知之甚少。在本工作中，我们研究了后训练是否引入或放大了类AI的风格规律性，以及这些规律性是否对应于一个可局部化的内部签名。为此，我们在匹配的人类来源前缀条件下，比较了人类文本、基座模型生成文本与对齐模型生成文本。结果表明，与基座模型的生成文本相比，对齐模型的生成文本具有更低的人类语料亲和度和更高的AI检测率，这说明后训练使生成文本偏离了人类语料风格，转向了检测器易于识别的类AI文本。随后，我们提出了PASTA（Post-training Alignment Signature Targeted Ablation，后训练对齐签名定向消融），这是一种无需训练的方法：它通过对比对齐模型与基座模型的残差来估计后训练对齐签名，并在解码过程中消融相应的方向。在11个……（摘要被截断）

    arXiv:2605.30526v2 Announce Type: replace-cross  Abstract: Aligned language models often exhibit a recognizable AI-like style, yet its connection to post-training and internal representations remains poorly understood. In this work, we study whether post-training introduces or amplifies AI-like stylistic regularities and whether these regularities have a localized internal signature. To this end, we compare human text, base-model generations, and aligned-model generations under matched human-source prefixes. Aligned generations show lower human-corpus affinity and higher AI-detection rates than base generations, suggesting that post-training shifts generated text away from human-corpus style and toward detector-visible AI-like text. We then introduce PASTA (Post-training Alignment Signature Targeted Ablation), a training-free method that estimates a post-training alignment signature from aligned-base residual contrasts and ablates the corresponding direction during decoding. Across 11 
    
[^156]: 激活预言机的置信度与校准：面向语言模型内部机制可靠解释的研究

    Confidence and Calibration of Activation Oracles for Reliable Interpretation of Language Model Internals

    [https://arxiv.org/abs/2605.26045](https://arxiv.org/abs/2605.26045)

    本研究比较了五种估计激活预言机置信度的方法，发现当审计者能提前列出候选答案时，让预言机直接对每个候选答案评分可获得最佳校准效果。

    

    激活预言机是一种被训练用于将另一个模型的激活值转换为自然语言的语言模型。然而，这些语言化表述的校准程度如何目前尚不清楚。关键问题在于，预言机的语言化表述没有任何置信度度量。本文比较了五种估计激活预言机置信度的方法。我们的实验涵盖来自两个模型家族（Qwen和Gemma，参数量从8B到27B）的四个预言机，并在“秘密词任务”上进行测试——在该任务中，预言机需要恢复一个被微调进模型但不出现在模型输出中的秘密词。基于每种方法/预言机组合6,000个样本的实验结果表明，最佳校准方式取决于假设空间或校准集是否可用：当假定的审计者能够提前列出所有可能的答案时，预言机应当直接对每个候选答案进行评分。

    arXiv:2605.26045v3 Announce Type: replace  Abstract: An activation oracle is a language model trained to convert the activations of another model into natural language. However, it is so far unclear to what extent these verbalizations are calibrated. Crucially, the oracle's verbalization come without any measure of confidence. Here, we compare five methods for estimating the confidence of an activation oracle. Our experiments cover four oracles from two model families (Qwen and Gemma, from 8B to 27B parameters), tested on the secret word task, whereby the oracle is supposed to recover a secret word, that has been fine-tuned into a model but does not appear in the model outputs. Our results with $6{,}000$ samples per method/oracle combination show that the best calibration depends on the availability of a hypothesis space or a calibration set: When an assumed auditor can list the possible answers in advance, the oracle should score each candidate answer directly. Scoring the candidates 
    
[^157]: 他们能走多远？利用大语言模型对网络影响力进行红队测试

    How Far Will They Go? Red-Teaming Online Influence with Large Language Models

    [https://arxiv.org/abs/2605.22880](https://arxiv.org/abs/2605.22880)

    该论文提出了一个实证红队测试框架，通过测量LLM的Overton窗口来量化本地部署的开源大语言模型表达政治观点的保真度以及简单越狱攻击对这些分数的影响，并发现模型在政治表达上存在系统性不对称（左倾内容的表达保真度通常更高）。

    

    随着基于大语言模型（LLM）的智能体日益参与网络话语，通过红队测试评估其支持政治影响力活动的能力，对于维护信息完整性至关重要。为实现这一目标，我们聚焦于可在本地部署的开源LLM，并设定了一种威胁模型：操作者试图规避检测、优先考虑输出吞吐量，且仅选用已发布的LLM检查点而不修改其权重。我们提出了一个实证性红队测试框架，用于测量LLM的“Overton窗口”（Overton Windows，OWs），即对采样政治观点的表达保真度进行汇总，并量化简单的自然语言越狱攻击如何改变这些分数。我们在英语、以美国为目标观点的场景下，评估了来自10个模型家族、5个国家的30余个LLM。我们发现政治表达能力存在系统性不对称：模型通常以更高的保真度表达采样的左倾内容……

    arXiv:2605.22880v2 Announce Type: replace  Abstract: As large language model (LLM)-based agents increasingly participate in online discourse, red-teaming their capacity to support political influence campaigns is critical for information integrity. In pursuit of this goal, we focus on locally deployable open-source LLMs under an assumed threat model in which an operator seeks to avoid detection, prioritizes throughput, and selects released LLM checkpoints without modifying their weights. We introduce an empirical red-teaming framework for measuring LLM Overton Windows (OWs), summarizing expression fidelity across sampled political opinions, and for quantifying how simple natural-language jailbreaks change these scores. We evaluate more than 30 LLMs spanning 10 model families and five countries of origin on English-language, US-focused target opinions. We find systematic asymmetries in political expressivity: models typically express sampled left-leaning content with higher fidelity, me
    
[^158]: 大语言模型何时应该进行推理？基于熵动态的自适应推理路由

    When Should LLMs Reason? Adaptive Inference Routing via Entropy Dynamics

    [https://arxiv.org/abs/2605.22873](https://arxiv.org/abs/2605.22873)

    提出无需训练的轻量级路由框架EDRM，通过解读解码早期的熵轨迹变化趋势，自适应地在直接、标准和思维链三种解码模式间选择，在15个基准和7个大语言模型上有效平衡准确率与token消耗。

    

    思维链推理可以提升大语言模型的性能，但其收益因任务、模型和具体查询而异。在某些任务上，CoT带来的准确率提升微乎其微甚至为负，同时却显著增加了token消耗。我们研究了早期解码动态是否有助于判断显式推理何时有用。我们的分析表明，CoT收益较大的任务往往在解码早期呈现熵值递减的趋势，而收益较低的任务则更多表现出振荡或上升的轨迹。这些模式促使我们对解码行为进行经验性状态解读。基于此，我们提出了EDRM（基于熵动态的推理流形），一个轻量级的路由框架，其默认启发式方法无需训练路由器。EDRM利用紧凑且可解释的表示来概括早期熵轨迹，并在直接解码、标准解码和CoT解码三种模式之间进行选择。在15个基准测试和7个大语言模型上的实验显示，其全局性能……

    arXiv:2605.22873v3 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) reasoning can improve LLM performance, but its benefits vary across tasks, models, and individual queries. On some tasks, CoT provides marginal or negative accuracy gains while substantially increasing token consumption. We investigate whether early decoding dynamics can help identify when explicit reasoning is useful. Our analysis shows that tasks with larger CoT gains tend to exhibit decreasing early-stage entropy, while low-gain tasks more often show oscillatory or increasing trajectories. These patterns motivate an empirical regime interpretation of decoding behavior. We propose \textbf{EDRM} (Entropy Dynamics-based Reasoning Manifold), a lightweight routing framework whose default heuristic requires no router training. EDRM summarizes early entropy trajectories with a compact, interpretable representation and selects among Direct, Standard, and CoT decoding. Across 15 benchmarks and 7 LLMs, its globa
    
[^159]: FormalASR：从中文口语到正式文本的端到端转换

    FormalASR: End-to-End Spoken Chinese to Formal Text

    [https://arxiv.org/abs/2605.19266](https://arxiv.org/abs/2605.19266)

    FormalASR提出两个紧凑的端到端模型（0.6B与1.7B参数），可直接将中文口语转录为正式书面文本，省去ASR+LLM两阶段流水线，相比逐字转录基线实现高达37.4%的相对字错误率降低。

    

    自动语音识别（ASR）系统通常针对逐字转录进行优化，这会保留不流畅表达、填充词以及非正式的口语结构，而这些内容往往不适合面向写作的下游应用。一种常见的替代方案是采用“ASR+LLM”两阶段流水线进行后期编辑，但这种设计会增加延迟和内存开销，且难以在设备端部署。我们提出了FormalASR，这是两个紧凑的端到端模型（参数量分别为0.6B和1.7B），能够将中文口语直接转录为正式的书面文本。为实现这一目标，我们构建了WenetSpeech-Formal和Speechio-Formal两个大规模“口语转正式文本”数据集，其构建方式为基于LLM的改写和质量过滤。随后，我们在两个模型规模上对Qwen3-ASR进行有监督微调。在WenetSpeech-Formal和Speechio-Formal上的实验表明，FormalASR相比逐字转录基线实现了高达37.4%的相对字错误率（CER）降低，同时……（原文摘要在此处截断）

    arXiv:2605.19266v4 Announce Type: replace  Abstract: Automatic speech recognition (ASR) systems are typically optimized for verbatim transcription, which preserves disfluencies, filler words, and informal spoken structures that are often unsuitable for downstream writing-oriented applications. A common workaround is a two-stage ASR+LLM pipeline for post-editing, but this design increases latency and memory cost and is difficult to deploy on-device. We present FormalASR, two compact end-to-end models with 0.6B and 1.7B parameters that directly transcribe spoken Chinese into formal written text. To enable this setting, we build WenetSpeech-Formal and Speechio-Formal, two large-scale spoken-to-formal datasets constructed by LLM-based rewriting and quality filtering. We then fine-tune Qwen3-ASR at both model scales with supervised fine-tuning. Experiments on WenetSpeech-Formal and Speechio-Formal show that FormalASR achieves up to 37.4% relative CER reduction over verbatim baselines, while
    
[^160]: IfcLLM：通过互补的关系型与图表示实现对IFC模型的自然语言查询

    IfcLLM: Natural Language Querying of IFC Models through Complementary Relational and Graph Representations

    [https://arxiv.org/abs/2605.13236](https://arxiv.org/abs/2605.13236)

    IfcLLM提出将关系型与图两种互补数据表示相结合的框架，通过LLM智能体的迭代重试-精炼推理自动将自然语言查询路由到最合适的后端，在IFC模型查询任务中达到96%的准确率，且无需用户干预即可从失败中自动恢复。

    

    arXiv:2605.13236v3 公告类型：替换。摘要：行业基础类（IFC）标准是建筑数据在全生命周期（从设计与施工到设施管理及数字孪生集成）中交换的核心。在实际运营场景中，利益相关方越来越需要在不具备IFC复杂、深度嵌套模式的专业知识的情况下获取建筑信息，这催生了对自然语言界面的需求。现有基于大语言模型（LLM）的查询方法通常依赖单一的数据表示，而单一表示无法同时很好地胜任属性检索和空间推理两类任务。我们提出了IfcLLM，一个结合互补的关系型与图两种表示的框架，可将每种查询类型路由到更合适的后端。LLM智能体通过迭代的“重试-精炼”推理将两者整合，无需用户干预即可从失败中恢复。在三个IFC模型共30个查询场景上的评估中，表现最佳的配置达到了96%的准确率。

    arXiv:2605.13236v3 Announce Type: replace  Abstract: The Industry Foundation Classes (IFC) standard is central to building data exchange across the lifecycle, from design and construction to facility management and Digital Twin integration. In operational settings, stakeholders increasingly require access to building information without specialist knowledge of IFC's complex, deeply nested schema, motivating natural language interfaces. Existing Large Language Model (LLM)-based querying approaches typically rely on a single data representation, which is not equally suited to attribute retrieval and spatial reasoning. We present IfcLLM, a framework that combines complementary relational and graph representations, routing each query type to the more suitable backend. An LLM agent integrates both through iterative retry-and-refine reasoning, recovering from failures without user input. Evaluated on 30 query scenarios across three IFC models, the top-performing configuration achieved 96% ac
    
[^161]: 图记忆Transformer (GMT)

    Graph Memory Transformer (GMT)

    [https://arxiv.org/abs/2604.23862](https://arxiv.org/abs/2604.23862)

    本文提出图记忆Transformer（GMT），用可学习的显式记忆图（128个质心加有向转移矩阵）替代仅解码器Transformer中的FFN子层，在因果自注意力不变的情况下，以更少的参数量（82.2M对103.0M）实现稳定训练。

    

    我们研究了一种显式的可学习记忆图能否替代仅解码器（decoder-only）Transformer中的前馈网络（FFN）子层，同时保留周围的自回归架构。在所提出的图记忆Transformer（GMT）中，记忆单元通过一个由学习得到的有向转移矩阵连接的可学习质心库来路由token表示，而因果自注意力机制保持不变。在基础v7实例中，16个Transformer块中的每一块包含128个质心和一个128×128的边矩阵。每个记忆单元采用引力源路由（gravitational source routing）和基于token条件的目标选择，返回一个从估计的源记忆状态指向目标记忆状态的门控位移（gated displacement），而非检索到的值。由此得到的仅解码器语言模型拥有8220万可训练参数，且不含密集的FFN子层；而用于对比的密集GPT风格基线模型拥有1.03亿参数。该模型能够稳定训练，并展现出质心u……（原文摘要至此截断）

    arXiv:2604.23862v4 Announce Type: replace-cross  Abstract: We investigate whether an explicit learned memory graph can replace the Feed-Forward Network (FFN) sublayer in a decoder-only transformer while preserving the surrounding autoregressive architecture. In the proposed graph Memory Transformer (GMT), a memory cell routes token representations over a learned bank of centroids connected by a learned directed transition matrix, while causal self-attention remains unchanged. In the base v7 instantiation, each of 16 transformer blocks contains 128 centroids and a 128 x 128 edge matrix. Each cell uses gravitational source routing and token-conditioned target selection to return a gated displacement from the estimated source memory state toward the target memory state, rather than a retrieved value. The resulting decoder-only language model has 82.2M trainable parameters and no dense FFN sublayers; the evaluated dense GPT-style baseline has 103.0M. It trains stably and exposes centroid u
    
[^162]: AtManRL：通过可微注意力显著性迈向忠实推理

    AtManRL: Towards Faithful Reasoning via Differentiable Attention Saliency

    [https://arxiv.org/abs/2604.16158](https://arxiv.org/abs/2604.16158)

    提出AtManRL方法，通过优化可微注意力掩码推导显著性奖励并与结果奖励结合于GRPO强化学习框架中，使思维链推理真正因果性地促成最终答案，从而提升推理的忠实性。

    

    大型语言模型（LLM）日益依赖思维链推理来解决复杂任务，然而推理轨迹可能只是伴随最终答案出现，而非因果性地促成答案，或忠实地反映答案的得出过程。我们引入可微注意力操作，通过强化学习训练显著性推理，并将该方法命名为 AtManRL。通过优化一个加性注意力掩码来识别恢复模型答案所需的思维链标记，我们推导出显著性奖励，并将其与结果奖励相结合，应用于具有解耦优势的 GRPO 中。由于显著性和忠实性都强烈依赖于轨迹长度，我们在评估时针对一系列长度惩罚基线进行扫描，以追踪纯长度惩罚所能达到的权衡范围，而非仅与单一基线比较。在三个模型和四个基准测试上的实验表明了该方法的有效性。

    arXiv:2604.16158v2 Announce Type: replace  Abstract: Large language models (LLMs) increasingly rely on chain-of-thought (CoT) reasoning to solve complex tasks, yet a reasoning trace may merely accompany the final answer rather than causally contribute to it or faithfully reflect how it was reached. We introduce differentiable attention manipulation to train salient reasoning with reinforcement learning -- method which we call AtManRL. By optimizing an additive attention mask that identifies the CoT tokens needed to recover the model's answer, we derive a saliency reward and combine it with an outcome reward in GRPO with decoupled advantages. Because both saliency and faithfulness depend strongly on trace length, we evaluate against a sweep of length-penalty baselines that traces the trade-off a pure length penalty can reach, rather than against a single baseline. Across three models (Qwen2.5-3B, Granite-3.3-2B, and Phi-4-mini) and four benchmarks (GSM8K, MMLU, ProofWriter, and QuestBen
    
[^163]: 基于径向共识分数的高效Best-of-N选择方法

    Efficient Best-Of-N with Radial Consensus Score

    [https://arxiv.org/abs/2604.12196](https://arxiv.org/abs/2604.12196)

    提出径向共识分数（RCS），通过计算答案嵌入的加权Fréchet均值作为语义中心，并依据候选答案到该中心的径向距离进行排序，实现了一种简单、高效且无需训练的best-of-N答案选择方法。

    

    使用大语言模型（LLM）解决复杂问题时，往往需要从多个候选回答中选出最佳答案（best-of-N），然而识别出最可靠的答案仍然具有挑战性，尤其是当正确性并不与多数意见一致时。现有的基于投票或概率评分的方法依赖于表层信号（如频率或词元级似然），常常遗漏高质量但罕见的答案，并且忽略了候选表示的几何结构——在这种结构中，正确性可能体现为语义一致性而非表面一致性。为解决这些局限，我们提出了径向共识分数，这是一种简单、高效且无需训练的best-of-N选择方法。RCS通过计算答案嵌入的加权Fréchet均值（语义中心）来建模语义共识，并根据候选答案到该中心的径向距离进行排序。这使得它能够恢复……

    arXiv:2604.12196v2 Announce Type: replace  Abstract: Solving complex problems with large language models (LLMs) often requires selecting the best answer from multiple candidate responses (best-of-$N$), yet identifying the most reliable one remains challenging, particularly when correctness does not align with majority agreement. Existing approaches based on voting or probability scoring rely on surface-level signals (e.g., frequency or token-level likelihood), often missing high-quality rare answers and ignoring the geometric structure of candidate representations, where correctness may emerge as semantic consistency rather than surface agreement. To address these limitations, we introduce Radial Consensus Score (RCS), a simple, efficient, and training-free method for best-of-$N$ selection. RCS models semantic consensus by computing a weighted Frechet mean (semantic center) of answer embeddings and ranking candidates by their radial distance to this center. This enables it to recover s
    
[^164]: 纽卡斯尔英语中自动语音识别偏见的社会语言学分析

    A Sociolinguistic Analysis of Automatic Speech Recognition Bias in Newcastle English

    [https://arxiv.org/abs/2603.24549](https://arxiv.org/abs/2603.24549)

    本研究基于泰恩赛德英语历时电子语料库（DECTE），对纽卡斯尔英语中3000多个商业ASR转录错误进行了细粒度的社会语言学分析，揭示了识别错误与说话人性别、年龄及社会经济地位等社会变量之间的关联，并通过元音声学案例研究展示了社会语言学变异如何引发语音识别偏见。

    

    自动语音识别（ASR）系统广泛应用于日常交流、教育、医疗和工业领域，但其在不同说话人之间的表现仍不均衡，尤其是当方言变体与训练数据中所呈现的主流口音存在差异时。本研究通过对纽卡斯尔英语的社会语言学分析来考察ASR偏见，纽卡斯尔英语是英格兰东北部的一种地区语言变体，已被证明对当前语音识别技术构成挑战。我们使用泰恩赛德英语历时电子语料库（DECTE）中的自发口语，评估了一款最先进的商业ASR系统的输出，并对3000多个转录错误进行了细粒度分析。这些错误按语言学领域进行分类，并结合性别、年龄和社会经济地位等社会变量加以考察。此外，针对选定元音特征的声学案例研究展示了梯度（原文在此处截断）……

    arXiv:2603.24549v2 Announce Type: replace  Abstract: Automatic Speech Recognition (ASR) systems are widely used in everyday communication, education, healthcare, and industry, yet their performance remains uneven across speakers, particularly when dialectal variation diverges from the mainstream accents represented in training data. This study investigates ASR bias through a sociolinguistic analysis of Newcastle English, a regional variety of North-East England that has been shown to challenge current speech recognition technologies. Using spontaneous speech from the Diachronic Electronic Corpus of Tyneside English (DECTE), we evaluate the output of a state-of-the-art commercial ASR system and conduct a fine-grained analysis of more than 3,000 transcription errors. Errors are classified by linguistic domain and examined in relation to social variables including gender, age, and socioeconomic status. In addition, an acoustic case study of selected vowel features demonstrates how gradien
    
[^165]: 语言模型作为高阶规划形式化器

    Language Models as Higher-Order Planning Formalizers

    [https://arxiv.org/abs/2603.23844](https://arxiv.org/abs/2603.23844)

    该论文指出LLM作为形式化器的成功在标准基准上被高估，并提出“展开式问题”这一新基准来揭示LLM形式化器在需要从简洁描述生成大规模PDDL时无法扩展，进而提出LLM作为高阶形式化器的新范式以应对这一挑战。

    

    近期的研究提供了压倒性的证据表明，即使是通过扩展推理轨迹进行训练的大语言模型（LLM），随着问题复杂度的增加，其在规划任务上的表现也会迅速恶化。LLM-as-Formalizers（LLM作为形式化器）方法旨在解决这一问题，其思路是将LLM作为桥梁，把自然语言描述翻译成结构化的规划表示（如PDDL），再将其输入给程序化求解器。我们观察到，这一方法的有效性可能被高估了，因为标准基准中的规划问题描述通常与PDDL存在一一对应的映射关系，这与真实使用场景相去甚远。为了解决这一问题，我们引入了“unraveling问题（展开式问题）”的概念，即一段自然且简洁的描述需要被翻译成非常庞大的PDDL表示。通过使用四个标准规划领域的unraveling变体，我们证明了LLM形式化器同样无法总是良好扩展。为应对这一挑战，我们提出了一种新范式——LLM-as-Higher-Order-Formalizer（LLM作为高阶形式化器），其中LLM……

    arXiv:2603.23844v4 Announce Type: replace  Abstract: Recent work provides overwhelming evidence that LLMs, even those trained to scale their reasoning trace, quickly deteriorate at planning as problems become more complex. LLM-as-Formalizers aim to address this by employing LLMs as a bridge to translate natural language descriptions into structured planning representations such as PDDL, which are then fed to a programmatic solver. We observe that its success may be overstated because planning problem descriptions in standard benchmarks often have a one-to-one mapping to PDDL, which departs from real use cases. To address this, we introduce the notion of unraveling problems where a natural yet succinct description translates into a very large PDDL representation. Using unraveling variants of four standard planning domains, we demonstrate that LLM Formalizers also do not always scale. We tackle this challenge by introducing a new paradigm, LLM-as-Higher-Order-Formalizer, where the LLM ge
    
[^166]: 大语言模型中的马尔可夫生成链

    Markovian Generation Chains in Large Language Models

    [https://arxiv.org/abs/2603.11228](https://arxiv.org/abs/2603.11228)

    本文将大语言模型反复处理文本的迭代推理过程定义为马尔可夫生成链，发现输出要么收敛到小的循环集合要么持续产生新句子，且多样性的增减取决于温度参数和初始输入句子等因素。

    

    大语言模型（LLM）的广泛使用引发了一个重要问题：当文本被大语言模型反复处理时，文本会如何演化？在本文中，我们将这种迭代推理过程定义为马尔可夫生成链，其中每一步都以特定的提示模板和先前的输出作为输入，而不包含任何先验数据。在迭代改写和往返翻译实验中，输出要么收敛到一个小的循环集合，要么在有限范围内持续产生新句子。通过句子级马尔可夫链建模和对模拟数据的分析，我们表明迭代过程既可能增加也可能降低多样性，这取决于温度参数和初始输入句子等因素。这些结果为迭代式大语言模型推理的动态特性及其对多智能体大语言模型系统的启示提供了有价值的见解。

    arXiv:2603.11228v2 Announce Type: replace  Abstract: The widespread use of large language models (LLMs) raises an important question: how do texts evolve when they are repeatedly processed by LLMs? In this paper, we define this iterative inference process as Markovian generation chains, where each step takes a specific prompt template and the previous output as input, without including any prior data. In iterative rephrasing and round-trip translation experiments, the output either converges to a small recurrent set or continues to produce novel sentences within a finite horizon. Through sentence-level Markov chain modeling and analysis of simulated data, we show that iterative process can either increase or reduce the diversity depending on factors such as the temperature parameter and the initial input sentence. These results offer valuable insights into the dynamics of iterative LLM inference and their implications for multi-agent LLM systems.
    
[^167]: TabAgent：将大语言模型智能体组件蒸馏为廉价分类器

    TabAgent: Distilling LLM Agent Components into Cheap Classifiers

    [https://arxiv.org/abs/2602.16429](https://arxiv.org/abs/2602.16429)

    TabAgent框架通过将智能体系统中反复调用的LLM生成式决策组件替换为基于执行轨迹训练的轻量级文本-表格分类器，在保持任务成功率的同时显著降低部署延迟和成本。

    

    智能体系统是能够自主执行多步骤工作流以实现复杂目标的人工智能架构，通常通过反复调用大语言模型（LLM）来完成路由、候选筛选、门控和验证等封闭集决策任务。这种设计虽然方便，但由于累积的延迟和token消耗，使得部署既缓慢又昂贵。我们提出TabAgent，这是一个用基于执行轨迹训练的紧凑文本-表格分类器来替换封闭集选择任务中生成式决策组件的框架。TabAgent（i）通过TabSchema从轨迹中提取结构化的模式、状态和依赖特征，（ii）通过TabSynth使用与模式对齐的合成监督来扩充数据覆盖范围，并（iii）通过TabHead使用轻量级分类器对候选项进行评分。在长时程AppWorld基准测试中，TabAgent在保持任务级成功率的同时消除了候选筛选阶段的LLM调用，将延迟降低约……（摘要原文在此处截断）

    arXiv:2602.16429v2 Announce Type: replace  Abstract: Agentic systems, AI architectures that autonomously execute multi-step workflows to achieve complex goals, are often built using repeated large language model (LLM) calls for closed-set decision tasks such as routing, shortlisting, gating, and verification. While convenient, this design makes deployments slow and expensive due to cumulative latency and token usage. We propose TabAgent, a framework for replacing generative decision components in closed-set selection tasks with a compact textual-tabular classifier trained on execution traces. TabAgent (i) extracts structured schema, state, and dependency features from trajectories (TabSchema), (ii) augments coverage with schema-aligned synthetic supervision (TabSynth), and (iii) scores candidates with a lightweight classifier (TabHead). On the long-horizon AppWorld benchmark, TabAgent maintains task-level success while eliminating shortlist-time LLM calls, reducing latency by approxima
    
[^168]: 人工智能正在创造一种新的全球语言等级体系

    Artificial intelligence is creating a new global linguistic hierarchy

    [https://arxiv.org/abs/2602.12018](https://arxiv.org/abs/2602.12018)

    该研究通过对6003种语言的社会、经济和基础设施条件进行全球纵向分析，揭示了语言AI资源分布的系统性不平等，表明人工智能正在催生一种使大多数语言社区数字边缘化的新全球语言等级体系。

    

    人工智能（AI）有潜力变革医疗保健、教育、治理和社会经济公平，但其收益仍集中于少数语言（Bender, 2019; Blasi et al., 2022; Joshi et al., 2020; Ranathunga and de Silva, 2022; Young, 2015）。语言人工智能——即支撑ChatGPT等广泛使用的对话系统的底层技术——如果能以人们的母语提供，将带来重大收益，然而全球7000多个语言社区中的大多数目前无法获得这些技术，并持续面临数字边缘化。在此，我们对各语言的社会、经济和基础设施条件进行了全球纵向分析，以评估语言人工智能领域存在的系统性不平等。我们首先分析了6003种语言的AI资源存在情况。我们发现，尽管社区努力扩大语言技术的覆盖范围（Bapna et al., 2022; Costa-Jussà et al...

    arXiv:2602.12018v2 Announce Type: replace-cross  Abstract: Artificial intelligence (AI) has the potential to transform healthcare, education, governance and socioeconomic equity, but its benefits remain concentrated in a small number of languages (Bender, 2019; Blasi et al., 2022; Joshi et al., 2020; Ranathunga and de Silva, 2022; Young, 2015). Language AI - the technologies that underpin widely-used conversational systems such as ChatGPT - could provide major benefits if available in people's native languages, yet most of the world's 7,000+ linguistic communities currently lack access and face persistent digital marginalization. Here we present a global longitudinal analysis of social, economic and infrastructural conditions across languages to assess systemic inequalities in language AI. We first analyze the existence of AI resources for 6003 languages. We find that despite efforts of the community to broaden the reach of language technologies (Bapna et al., 2022; Costa-Juss\`a et al
    
[^169]: GFlowPO：生成流网络作为语言模型提示词优化器

    GFlowPO: Generative Flow Network as a Language Model Prompt Optimizer

    [https://arxiv.org/abs/2602.03358](https://arxiv.org/abs/2602.03358)

    GFlowPO提出了一种概率化提示词优化框架，利用离线策略的生成流网络目标微调轻量级提示词语言模型，并通过无需训练的动态记忆更新机制更新元提示词，从而实现样本高效的语言模型提示词优化。

    

    为语言模型寻找有效的提示词至关重要却又极其困难：提示词空间在组合意义上非常庞大，且由于目标语言模型评估代价高昂，奖励十分稀疏。然而，现有的基于强化学习的提示词优化器通常依赖在线策略更新以及从固定分布中采样的元提示词，导致样本效率低下。我们提出GFlowPO，一个概率化的提示词优化框架，它将提示词搜索转化为一个后验推断问题：在潜在提示词上进行推断，并由以元提示词为条件的参考语言模型先验进行正则化。在第一步中，我们采用离线策略的生成流网络目标函数微调一个轻量级的提示词语言模型，并使用基于重放缓冲区的训练策略重用过去的提示词评估结果，从而实现样本高效的探索。在第二步中，我们引入动态记忆更新，这是一种无需训练的机制，通过注入以下内容来更新元提示词：（i）来自……的多样化提示词……（摘要在此处被截断）

    arXiv:2602.03358v2 Announce Type: replace-cross  Abstract: Finding effective prompts for language models (LMs) is critical yet notoriously difficult: the prompt space is combinatorially large, rewards are sparse due to expensive target-LM evaluation. Yet, existing RL-based prompt optimizers often rely on on-policy updates and a meta-prompt sampled from a fixed distribution, leading to poor sample efficiency. We propose GFlowPO, a probabilistic prompt optimization framework that casts prompt search as a posterior inference problem over latent prompts regularized by a meta-prompted reference-LM prior. In the first step, we fine-tune a lightweight prompt-LM with an off-policy Generative Flow Network (GFlowNet) objective, using a replay-based training policy that reuses past prompt evaluations to enable sample-efficient exploration. In the second step, we introduce Dynamic Memory Update (DMU), a training-free mechanism that updates the meta-prompt by injecting both (i) diverse prompts from
    
[^170]: 大语言模型中的未完成体悖论

    The Imperfective Paradox in Large Language Models

    [https://arxiv.org/abs/2601.09373](https://arxiv.org/abs/2601.09373)

    该论文提出ImperfectiveNLI诊断数据集，揭示了大语言模型在未完成体悖论上普遍存在的“目的性偏差”——模型会系统性地臆想目标导向事件的完成结果、甚至无视文本中明确的取消表述，而提示干预虽能缓解该偏差却会引发校准危机、导致模型错误拒绝无终体动词的有效蕴含。

    

    大语言模型（LLM）是否真正掌握了事件的组合语义，还是仅仅依赖表层的概率启发式规则？我们研究了“未完成体悖论”（Imperfective Paradox）这一逻辑现象：过去进行时对于活动类动词（如“正在跑步”→“跑了”）蕴含事件已然实现，但对于达成类动词（如“正在建造”→“建成了”）则不作此蕴含。我们引入了ImperfectiveNLI，一个旨在跨多种语义类别探测这一区别的诊断数据集。通过评估最先进的开放权重模型，我们发现了一种普遍存在的“目的性偏差”：模型会系统性地为面向目标的事件臆想出完成结果，甚至会覆盖文本中明确的取消表述。提示干预能在部分程度上减轻这种偏差，却会引发“校准危机”，导致模型错误地拒绝无终体（atelic）动词的有效蕴含关系。表征分析进一步表明，尽管内部嵌入通常能够区分……

    arXiv:2601.09373v3 Announce Type: replace  Abstract: Do Large Language Models (LLMs) genuinely grasp the compositional semantics of events, or do they rely on surface-level probabilistic heuristics? We investigate the Imperfective Paradox, a logical phenomenon where the past progressive aspect entails event realization for activities (e.g., running $\to$ ran) but not for accomplishments (e.g., building $\nrightarrow$ built). We introduce ImperfectiveNLI, a diagnostic dataset designed to probe this distinction across diverse semantic classes. Evaluating state-of-the-art open-weight models, we uncover a pervasive Teleological Bias: models systematically hallucinate completion for goal-oriented events, even overriding explicit textual cancellation. Prompting interventions partially reduce this bias but trigger a calibration crisis, causing models to incorrectly reject valid entailments for atelic verbs. Representational analyses further show that while internal embeddings often distinguis
    
[^171]: PsyCLIENT：基于对话轨迹建模的来访者模拟，用于心理健康咨询中的受训者练习与模型评估

    PsyCLIENT: Client Simulation via Conversational Trajectory Modeling for Trainee Practice and Model Evaluation in Mental Health Counseling

    [https://arxiv.org/abs/2601.07312](https://arxiv.org/abs/2601.07312)

    PsyCLIENT通过对话轨迹建模来驱动心理咨询来访者模拟，并配套构建了包含120个中文来访者画像的PsyCLIENT-CP数据集，经24名专业咨询师验证，其在真实性和培训实用性上优于现有方法，为新手咨询师训练与心理咨询系统评估提供了可扩展的新方案。

    

    基于大语言模型（LLM）的来访者模拟为新手咨询师培训、咨询对话合成以及自动化咨询系统的交互式评估提供了一种可扩展的途径。然而，现有方法存在以下局限：来访者画像多样性不足、行为依据薄弱，且缺乏面向模拟咨询来访者的开放中文资源。我们提出了PsyCLIENT，该框架基于来访者画像、对话历史以及为每一轮来访者回应指定目标行为与内容约束的对话轨迹来生成模拟来访者的回应。我们还构建了PsyCLIENT-CP数据集，其中包含涵盖60个咨询主题的120个中文来访者画像。由24名专业咨询师参与的评估表明，与对比方法相比，PsyCLIENT在感知真实性和培训实用性方面获得了更高的评分。在另一项来源识别研究中，PsyCLIENT生成的对话更频繁地被误判为真人对话。

    arXiv:2601.07312v3 Announce Type: replace  Abstract: LLM-based client simulation provides a scalable approach to novice counselor training, counseling-dialogue synthesis, and interactive evaluation of automated counseling systems. However, existing approaches are limited by insufficient profile diversity, weak behavioral grounding, and the lack of open Chinese-language resources for simulated counseling clients. We propose PsyCLIENT, a framework that conditions simulated-client responses on client profiles, dialogue histories, and conversational trajectories specifying target behaviors and content constraints at each client turn. We also construct PsyCLIENT-CP, a dataset of 120 Chinese client profiles spanning 60 counseling topics. Evaluations involving 24 professional counselors show that PsyCLIENT receives higher ratings of perceived authenticity and training utility than the comparison methods. In a separate source-identification study, PsyCLIENT dialogues were more frequently miscl
    
[^172]: RiskCueBench：基于早期风险线索的视频-语言模型前瞻性推理基准测试

    RiskCueBench: Benchmarking Anticipatory Reasoning from Early Risk Cues in Video-Language Models

    [https://arxiv.org/abs/2601.03369](https://arxiv.org/abs/2601.03369)

    该论文提出了RiskCueBench基准，通过标注视频中最早出现风险信号的时刻，评估视频-语言模型仅凭早期视觉线索预判未来危险事件的能力，实验发现现有系统的前瞻性风险推理能力仍存在显著不足。

    

    随着以视频为中心的社交媒体的快速发展，从视觉数据中预测风险事件的能力是确保公共安全和预防现实世界事故的一个有前景的方向。先前的工作已经在驾驶、抗议活动和自然灾害等领域广泛研究了有监督的视频风险评估。然而，许多现有数据集为模型提供了完整的视频序列（包括事故本身），这大大降低了任务的难度。为了更好地反映现实世界的情况，我们引入了一个新的视频理解基准RiskCueBench，其中的视频经过精心标注以识别风险信号片段，即指示潜在安全隐患的最早时刻。实验结果表明，当前系统在解读不断演变的情境以及根据早期视觉信号预测未来风险事件方面存在显著差距，凸显了……

    arXiv:2601.03369v3 Announce Type: replace-cross  Abstract: With the rapid growth of video centered social media, the ability to anticipate risky events from visual data is a promising direction for ensuring public safety and preventing real world accidents. Prior work has extensively studied supervised video risk assessment across domains such as driving, protests, and natural disasters. However, many existing datasets provide models with access to the full video sequence, including the accident itself, which substantially reduces the difficulty of the task. To better reflect real world conditions, we introduce a new video understanding benchmark RiskCueBench in which videos are carefully annotated to identify a risk signal clip, defined as the earliest moment that indicates a potential safety concern. Experimental results reveal a significant gap in current systems ability to interpret evolving situations and anticipate future risky events from early visual signals, highlighting impor
    
[^173]: ChemBART：用于综合合成规划的化学反应语言模型预训练

    ChemBART: Pre-training a Reaction Language Model for Comprehensive Synthesis Planning

    [https://arxiv.org/abs/2601.02915](https://arxiv.org/abs/2601.02915)

    ChemBART是一种以反应为中心进行预训练的BART语言模型，通过在整体反应表达式上进行掩码填充预训练，能够统一解决前体/试剂生成、产率回归、分子性质分类等多种化学任务，并可与蒙特卡洛树搜索结合实现多步合成路线设计。

    

    大语言模型（LLMs）的最新进展已在多个领域展现出变革性潜力。然而，在计算机辅助合成规划（CASP）领域，当前的LLM方法通常局限于单分子表示，这限制了其在孤立的单任务场景（如前体预测）中的应用。为填补这一空白，我们提出了ChemBART，一个基于BART架构、采用以反应为中心的预训练策略的语言模型。通过在整体反应表达式（反应物 > 试剂 > 产物）上进行掩码填充预训练任务，ChemBART能够有效解决多种化学问题，包括前体/试剂生成、温度-产率回归以及分子性质分类。基于这些能力，我们展示了将ChemBART与蒙特卡洛树搜索（MCTS）相集成以进行多步合成路线设计，其中该模型用于优化策略和……

    arXiv:2601.02915v2 Announce Type: replace  Abstract: Recent advances in large language models (LLMs) have demonstrated transformative potential across diverse fields. However, in computer-aided synthesis planning (CASP), current LLM methodologies are often restricted to single-molecule representations, limiting their application to isolated, single-task scenarios such as precursor prediction. Addressing this gap, we present ChemBART, a BART-based language model that adopts a reaction-centric pre-training strategy. By leveraging a mask-filling pre-training task on holistic reaction expressions (Reactant > Reagent > Product), ChemBART effectively solves a variety of chemical problems, including precursor/reagent generation, temperature-yield regression, and molecular property classification. Building on these capabilities, we demonstrate the integration of ChemBART with Monte Carlo tree search (MCTS) for multi-step synthesis route design, where the model serves to optimize the policy and
    
[^174]: 语义欺骗：当推理模型无法完成一次加法运算时

    Semantic Deception: When Reasoning Models Can't Compute an Addition

    [https://arxiv.org/abs/2512.20812](https://arxiv.org/abs/2512.20812)

    研究通过“语义欺骗”实验发现，当数字和运算符被重新定义并携带误导性语义关联时，LLMs难以维持真正的符号抽象推理，往往退回到利用训练中学到的语义关联。

    

    大型语言模型（LLMs）越来越多地被应用于涉及人类价值观的情境中，例如那些对人类而言需要推理才能完成的决策任务。我们通过引入一个实验框架来研究LLMs在新型符号表示上的所谓推理能力，该框架测试它们处理和操作不熟悉符号的能力。我们提出了“语义欺骗”的概念：即符号因其形式（例如被嵌入特定语境中）而携带误导性语义关联的情形，旨在探究LLMs能否保持符号抽象能力，还是会默认利用已习得的语义关联。我们使用新颖的符号重新定义标准数字和数学运算符，并要求LLMs解决以这种改变后的记号表示的简单计算。其目标是：（1）评估LLMs对任意符号系统的抽象与操作能力……（原文摘要在此处截断）

    arXiv:2512.20812v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used in situations where human values are at stake, such as decision-making tasks that involve reasoning when performed by humans. We investigate the so-called reasoning capabilities of LLMs over novel symbolic representations by introducing an experimental framework that tests their ability to process and manipulate unfamiliar symbols. We introduce semantic deceptions: situations in which symbols carry misleading semantic associations due to their form, such as being embedded in specific contexts, designed to probe whether LLMs can maintain symbolic abstraction or whether they default to exploiting learned semantic associations. We redefine standard digits and mathematical operators using novel symbols, and task LLMs with solving simple calculations expressed in this altered notation. The objective is: (1) to assess LLMs' capacity for abstraction and manipulation of arbitrary symbol syst
    
[^175]: FLAWS：科学论文中错误识别与定位的基准测试

    FLAWS: A Benchmark for Error Identification and Localization in Scientific Papers

    [https://arxiv.org/abs/2511.21843](https://arxiv.org/abs/2511.21843)

    该论文提出了FLAWS基准，通过在同行评审论文中系统性插入破坏关键论断的错误，构建了713个论文-错误对，以自动化评估大语言模型检测和定位科学论文中错误的能力。

    

    错误的识别与定位是同行评审中的一项核心任务，然而科学产出的指数级增长，加之专家资源有限，使得人类评审者越来越难以可靠地检测出错误。大语言模型（LLM）的最新进展激发了人们对其在这类评估任务中潜力的兴趣，涵盖从学术同行评审到自动化科学评估等应用。然而，尽管LLM在评审系统中的应用日益增多，其精确定位错误的能力却仍未得到充分探索。在本工作中，我们提出了科学写作中的错误定位基准FLAWS（Fault Localization Across Writing in Science），这是一个包含713个论文-错误对的自动化基准，旨在评估LLM检测那些破坏研究论文关键论断的错误的有效性。我们通过利用LLM系统性地在经同行评审的论文中插入使论断失效的错误来构建该基准，并配套了自动化的评估指标。

    arXiv:2511.21843v2 Announce Type: replace  Abstract: The identification and localization of errors is a core task in peer review, yet the exponential growth of scientific output has made it increasingly difficult for human reviewers to reliably detect errors given the limited pool of experts. Recent advances in Large Language Models (LLMs) have sparked interest in their potential to support such evaluation tasks, from academic peer review to automated scientific assessment. However, despite the growing use of LLMs in review systems, their capabilities to pinpoint errors remain underexplored. In this work, we introduce Fault Localization Across Writing in Science (FLAWS), an automated benchmark consisting of 713 paper-error pairs designed to evaluate how effectively LLMs detect errors that undermine key claims in research papers. We construct the benchmark by systematically inserting claim-invalidating errors into peer-reviewed papers using LLMs, paired with an automated evaluation metr
    
[^176]: MENTOR：一种用于揭示与缓解大语言模型隐性领域风险的元认知驱动自进化框架

    MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Domain Risks in LLMs

    [https://arxiv.org/abs/2511.07107](https://arxiv.org/abs/2511.07107)

    该论文提出元认知驱动的自进化框架MENTOR，通过涵盖教育、金融和管理领域的3000条标注查询数据集揭示出14个主流大语言模型平均57.8%的越狱成功率，并利用换位思考与后果推理等元认知策略进行自我评估、修订不安全响应并将验证后的规则编译为激活层面引导信号，从而有效缓解模型的隐性领域安全风险。

    

    确保大语言模型（LLM）的安全性对其在现实世界中的部署至关重要。然而，当前的安全措施往往无法应对隐性的、特定领域的风险。为研究这一差距，我们构建了一个包含3000条标注查询的数据集，涵盖教育、金融和管理三个领域。对14个领先大语言模型的评估揭示了一个令人担忧的脆弱性：平均越狱成功率达57.8%。为此，我们提出了MENTOR——一个元认知驱动的自进化框架。MENTOR执行元认知自我评估，运用换位思考和后果推理等策略来揭示模型潜在的失准问题。MENTOR将对常规请求的单次规则引导推理与选择性调用的元认知进化周期相结合，该周期修订残留的不安全响应，将成功修正提炼为动态规则图，并将经过验证的规则编译为激活层面的引导信号。

    arXiv:2511.07107v5 Announce Type: replace-cross  Abstract: Ensuring the safety of Large Language Models (LLMs) is critical for real-world deployment. However, current safety measures often fail to address implicit, domain-specific risks. To investigate this gap, we introduce a dataset of 3,000 annotated queries spanning education, finance, and management. Evaluations across 14 leading LLMs reveal a concerning vulnerability: an average jailbreak success rate of 57.8\%. In response, we propose MENTOR, a metacognition-driven self-evolution framework. MENTOR performs metacognitive self-assessment, using strategies such as perspective-taking and consequential reasoning to uncover latent model misalignments. MENTOR couples single-pass rule-guided inference for routine requests with a selectively invoked metacognitive evolution cycle that revises residual unsafe responses, distills successful corrections into a dynamic rule graph, and compiles validated rules into activation-level steering si
    
[^177]: 超越复制的上下文学习：抽象式ICL的训练时分析

    In-Context Learning Beyond Copying: A Training-Time Analysis of Abstractive ICL

    [https://arxiv.org/abs/2511.05743](https://arxiv.org/abs/2511.05743)

    研究通过Hapax训练机制抑制归纳复制并消除相变后，发现抽象式上下文学习能力仍能涌现，表明ICL能力并不依赖于归纳复制机制。

    

    大语言模型中归纳电路的突然涌现以及执行上下文复制的能力，与训练过程中短时间内的训练损失急剧下降相吻合。这一共现现象曾被引为证据，表明该时间点标志着广泛的上下文学习（ICL）能力开始涌现。为了分析少样本抽象式ICL能力（即答案不包含在输入上下文中的任务）的训练动态，我们提出了Hapax——一种训练机制，它通过省略归纳电路可预测的token的损失贡献，来控制复制相关的训练压力。通过这一干预，我们发现尽管归纳式复制大幅减少，且原始训练中观察到的特征性相变消失，抽象式ICL能力仍然能够涌现。对注意力头的分析显示，Hapax训练的模型发展出更少且更弱的……（原文摘要在此处截断）

    arXiv:2511.05743v3 Announce Type: replace  Abstract: The sudden emergence of induction circuits and the ability to perform in-context copying in LLMs coincides with a sharp drop in training loss over a short span of the training process. This co-occurrence has been cited as evidence that it marks the point at which a wide range of in-context learning (ICL) capabilities begin to emerge. To analyze the training dynamics of few-shot abstractive ICL capabilities (i.e., tasks where the answer is not contained in the input context), we propose Hapax, a training regime that omits the loss contribution of tokens predictable by the induction circuit to control training pressure for copying. Using this intervention, we find that abstractive ICL capabilities can still emerge, despite a substantial reduction in inductive copying and the absence of the characteristic phase changes observed during Vanilla training. Analysis of attention heads shows that Hapax-trained models develop fewer and weaker 
    
[^178]: 数学推理在以英语为中心的大语言模型中跨语言迁移的效果如何？

    How Well Does Mathematical Reasoning Transfer Across Languages in English-Centric LLMs?

    [https://arxiv.org/abs/2510.02272](https://arxiv.org/abs/2510.02272)

    该研究系统评估了以英语为中心的大语言模型的跨语言数学推理迁移，发现迁移效果受初始模型、目标语言和训练范式的影响差异显著，且更强的英语初始化未必带来更高的迁移效率，而引入单一平行语言监督能带来最大提升。

    

    强化后训练的最新进展已显著提升了大型语言模型（LLM）的数学推理能力，但这些能力在多大程度上能够跨语言迁移仍不清楚。在本工作中，我们在受控的多语言评测条件下研究了以英语为中心的大语言模型中的跨语言数学推理迁移。我们首先在多语言数学推理基准上评估了一系列以英语为中心的推理模型，发现迁移效果在初始模型选择、目标语言和训练范式之间存在高度差异。通过受控对比实验，我们进一步发现，更强的以英语为中心的初始模型并不一定带来更强的相对迁移效率，即使它能产生更高的多语言推理准确率。最后，我们表明最大的提升来自于超越仅使用英语监督的做法：添加一种平行语言（原文在此处截断）。

    arXiv:2510.02272v2 Announce Type: replace  Abstract: Recent advances in Reinforcement Post-Training (RPT) have substantially improved the mathematical reasoning capabilities of large language models (LLMs), but it remains unclear how well such gains transfer across languages. In this work, we study cross-lingual mathematical reasoning transfer in English-centric LLMs under controlled multilingual evaluation. We first evaluate a range of English-centric reasoning models on multilingual mathematical reasoning benchmarks and find that transfer is highly variable across initial model choice, target language, and training paradigm. Through controlled comparative experiments, we further find that stronger English-centric initialization does not necessarily lead to stronger relative transfer efficiency, even when it yields higher multilingual reasoning accuracy. Finally, we show that the largest improvement comes from moving beyond English-only supervision: adding a single parallel language y
    
[^179]: 领域特定宪法AI：增强基于大语言模型的心理健康聊天机器人的安全性

    Domain-Specific Constitutional AI: Enhancing Safety in LLM-Powered Mental Health Chatbots

    [https://arxiv.org/abs/2509.16444](https://arxiv.org/abs/2509.16444)

    该论文提出了一种领域特定的宪法AI框架，通过将心理健康领域的专业准则（如危机干预协议和治疗指南）融入AI安全机制，以增强大语言模型驱动的心理健康聊天机器人的安全性。

    

    心理健康应用已成为计算健康领域的关键方向，这一趋势受到全球精神疾病发病率上升、人工智能在心理护理中的整合以及服务不足社区对可扩展解决方案需求的推动。这些应用包括治疗聊天机器人、危机检测以及处理敏感数据的健康平台，由于用户的情感脆弱性、误诊或症状恶化等风险，以及对脆弱状态的精确管理以避免自残或失去信任等严重后果，这些应用需要超越通用安全防护措施的专业化AI安全保障。尽管AI安全领域已取得进展，但通用安全防护措施仍不足以应对心理健康领域的特定挑战，包括危机干预的准确性以防止事态升级、遵循治疗准则以防止错误信息传播、资源受限环境下的规模限制，以及对细微对话的适应——通用方法可能在对话中引入偏差或遗漏求助信号……

    arXiv:2509.16444v2 Announce Type: cross  Abstract: Mental health applications have emerged as a critical area in computational health, driven by rising global rates of mental illness, the integration of AI in psychological care, and the need for scalable solutions in underserved communities. These include therapy chatbots, crisis detection, and wellness platforms handling sensitive data, requiring specialized AI safety beyond general safeguards due to emotional vulnerability, risks like misdiagnosis or symptom exacerbation, and precise management of vulnerable states to avoid severe outcomes such as self-harm or loss of trust. Despite AI safety advances, general safeguards inadequately address mental health-specific challenges, including crisis intervention accuracy to avert escalations, therapeutic guideline adherence to prevent misinformation, scale limitations in resource-constrained settings, and adaptation to nuanced dialogues where generics may introduce biases or miss distress s
    
[^180]: MoVoC：面向吉兹文字语言的形态感知子词构建（修正版）

    MoVoC: Morphology-Aware Subword Construction for Geez Script Languages Corrected version

    [https://arxiv.org/abs/2509.08812](https://arxiv.org/abs/2509.08812)

    本文提出 MoVoC 形态感知子词构建方法，将监督式形态分析与 BPE 相结合训练分词器 MoVoC-Tok 以保留形态完整性，并发布四种吉兹文字语言的词素标注数据（169,806 条），在四种语言中的三种取得最高的 MorphScore。

    

    基于子词的分词方法通常无法保留形态边界，这一局限在低资源且形态复杂的语言（如使用吉兹文字书写的语言）中尤为突出。为解决这一问题，我们提出了形态感知子词词表构建方法，并训练了 MoVoC-Tok——一种将监督式形态分析融入子词词表的分词器。这种混合分割方法结合了基于词素的标记与字节对编码（BPE）标记，在保持词汇意义的同时保留了形态完整性。为应对资源稀缺问题，我们发布了四种吉兹文字语言的词素标注数据（共 169,806 条记录），以及其中两种语言的形态感知词表。在所有对比实验中，评分均在从约束该语言 MoVoC-Tok 的词素资源中留出的词汇上进行，MoVoC-Tok 在四种语言中的三种获得了最高的 MorphScore，并取得了最高的形态……（摘要原文在此处截断）

    arXiv:2509.08812v2 Announce Type: replace  Abstract: Subword-based tokenization methods often fail to preserve morphological boundaries, a limitation especially pronounced in low-resource, morphologically complex languages such as those written in the Ge'ez script. To address this, we present Morpheme-aware Subword Vocabulary Construction (MoVoC) and train MoVoC-Tok, a tokenizer that integrates supervised morphological analysis into the subword vocabulary. This hybrid segmentation approach combines morpheme-based and Byte Pair Encoding (BPE) tokens to preserve morphological integrity while maintaining lexical meaning. To tackle resource scarcity, we release morpheme-annotated data for four Ge'ez script languages (169,806 records) and a morpheme-aware vocabulary for two of them. Scoring every comparison on words held out from the morpheme resource that constrains the MoVoC-Tok applied to that language, MoVoC-Tok has the highest MorphScore in three of four languages and the highest morph
    
[^181]: ShIOEnv：一个用于语法约束合成与执行行为建模的命令评估环境

    ShIOEnv: A Command Evaluation Environment for Grammar-Constrained Synthesis and Execution Behavior Modeling

    [https://arxiv.org/abs/2505.18374](https://arxiv.org/abs/2505.18374)

    ShIOEnv是一个与Gymnasium兼容的Bash命令评估环境，通过语法约束合成生成有效的shell输入，并记录完整的执行产物（标准输出、标准错误、退出状态和系统状态变化），为训练无需执行的shell模拟模型和部署网络欺骗蜜罐提供数据支持。

    

    我们介绍了ShIOEnv，一个与Gymnasium兼容的Bash环境，用于合成带有执行标注的shell输入轨迹。包含此类轨迹的数据集对于训练模型执行无需实际执行的shell模拟非常有用，这进而有助于部署用于网络欺骗的蜜罐。现有数据集（如NL2CMD及其他Bash命令语料库）不包含与命令执行相关的完整执行产物（标准输出、标准错误、退出状态以及系统状态的变化）。ShIOEnv能够合成shell输入，在受控的系统上下文中执行它们，并记录标准输出、标准错误、退出状态以及结构化的文件系统和环境补丁。无约束（随机）生成的shell输入（命令和参数）会产生大量冗余或导致执行错误的参数实例。我们引入语法约束合成方法来生成语法上有效且覆盖度高的参数……

    arXiv:2505.18374v3 Announce Type: replace  Abstract: We introduce ShIOEnv, a Gymnasium-compatible Bash environment for synthesizing execution-annotated shell input traces. Datasets containing such traces are useful for training models to perform execution-free shell simulation, which in turn is useful for deploying honeypots for cyber deception. Existing datasets, such as NL2CMD and other Bash command corpora, do not include complete execution artifacts (stdout, stderr, exit status, and changes to the system's state) associated with a command's execution. ShIOEnv synthesizes shell inputs, executes them in a controlled system context, and records stdout, stderr, exit status, and structured filesystem and environment patches. Unconstrained (random) generation of shell inputs (command and arguments) results in multiple instances of arguments that are mostly redundant or result in execution errors. We introduce grammar-constrained synthesis to generate syntactically valid arguments with hi
    
[^182]: 孔子：一种具有事实增强的历史大语言模型

    Kongzi: A Historical Large Language Model with Fact Enhancement

    [https://arxiv.org/abs/2504.09488](https://arxiv.org/abs/2504.09488)

    孔子是一个结合历史语料继续预训练、监督微调与基于知识图谱事实奖励的强化学习的历史大语言模型，显著提升了历史问答和推理中的事实准确性。

    

    arXiv:2504.09488v2 公告类型：替换 摘要：大语言模型（LLMs）在长链推理过程中或在误导性提示下可能表现出事实性错误，这限制了其在历史分析中的可靠性。我们提出了孔子（Kongzi），一种面向历史理解与推理的语言模型，它结合了在历史语料库上的继续预训练、监督微调以及事实增强的强化学习。其事实性奖励通过知识图谱相似度，将从生成回复中提取的实体和关系与历史参考资料进行比较。我们在历史问答和反事实推理任务上评估了0.5B和7B两种规模的模型，使用三个LLM评判者给出的报告评分，并在CMMLU-Chinese-History（中国历史）基准上进行测试。在所有三个评判者的评估下，Kongzi-7B在所比较的模型中获得了最高的历史准确性评分。强化学习提高了两种模型规模的平均答案分数及其在CMMLU-Chinese-History上的准确率，尽管Kongzi-7B仍然...

    arXiv:2504.09488v2 Announce Type: replace  Abstract: Large Language Models (LLMs) can exhibit factual inaccuracies during long-chain reasoning or under misleading prompts, limiting their reliability in historical analysis. We propose Kongzi, a language model for historical understanding and reasoning that combines continued pre-training on historical corpora, supervised fine-tuning, and fact-enhanced reinforcement learning. Its factuality reward compares entities and relations extracted from generated responses with historical references through knowledge-graph similarity. We evaluate 0.5B and 7B models on historical question answering and counterfactual reasoning, using reported ratings from three LLM judges, and on CMMLU-Chinese-History. Kongzi-7B receives the highest historical-accuracy ratings among the compared models under all three judges. Reinforcement learning improves the mean answer scores of both model sizes and their CMMLU-Chinese-History accuracy, although Kongzi-7B remai
    
[^183]: 面向非认知技能学习的社交生活模拟

    Social Life Simulation for Non-Cognitive Skills Learning

    [https://arxiv.org/abs/2405.00273](https://arxiv.org/abs/2405.00273)

    本文提出基于大语言模型的交互式社交生活模拟平台Simulife++，通过引入“贤者智能体”将人机交互创新性地扩展为人-AI-AI协作模式，支持用户在多样化社交场景中培养非认知技能并提升自我反思能力。

    

    非认知技能对个人和社交生活的幸福感至关重要，此类技能的发展可以借助基于叙事（如讲故事）的技术来支持。尽管生成式AI能够实现互动式和角色扮演式的叙事，但人们对于用户如何参与并感知在非认知技能学习的社交生活模拟中使用AI仍知之甚少。此外，在此背景下，AI导师指导对自我反思意识和能力的益处也在很大程度上未被充分探索。为此，我们推出了Simulife++，一个由大语言模型（LLM）驱动的交互式平台。该系统允许用户扮演主角，在多样化的社交场景中与一个或多个AI角色共同创作故事。特别地，我们通过引入一个作为旁观者的“贤者智能体”，将人机交互扩展为人-AI-AI协作模式，为用户提供关于其选择的视角和指导。

    arXiv:2405.00273v4 Announce Type: replace  Abstract: Non-cognitive skills are crucial for personal and social life well-being, and such skill development can be supported by narrative-based (e.g., storytelling) technologies. While generative AI enables interactive and role-playing storytelling, little is known about how users engage with and perceive the use of AI in social life simulation for non-cognitive skills learning. Additionally, the benefits of AI mentorship on self-reflection awareness and ability in this context remain largely underexplored. To this end, we introduced Simulife++, an interactive platform enabled by a large language model (LLM). The system allows users to act as protagonists, creating stories with one or multiple AI-based characters in diverse social scenarios. In particular, we expanded the Human-AI interaction to a Human-AI-AI collaboration by including a Sage Agent, who acts as a bystander, providing users with some perspectives and guidance on their choice
    

