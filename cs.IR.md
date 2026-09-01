# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Configurable Semantic Chunking for Biomedical Information Extraction in Retrieval-Augmented Generation](https://arxiv.org/abs/2608.31139) | 提出了一种可配置的语义分块框架，通过实体保留窗口、触发词中心分块、命题优先抽取和层次化关系解析等策略替代固定大小分块，在仅替换 BioMedRAG 分块构建阶段的前提下，将 GM-CIHT 生物医学关系抽取的 F1 分数提升了 8.4 个百分点。 |
| [^2] | [InsightToast: Proactive Information Retrieval & Glanceable Visualization in the Side Channel of Data-Rich Meetings](https://arxiv.org/abs/2608.31115) | InsightToast通过多智能体LLM与RAG管道实时监测会议话语并主动检索信息，以临时提示和一目了然图表的形式在会议侧信道中提供有据可依的洞察，从而避免任务切换对会议参与和决策的干扰。 |
| [^3] | [Learning to Evaluate Before Improving: Automatic Rubric Induction for Automatic Research Agents](https://arxiv.org/abs/2608.31076) | AutoSciRub 提出“先评估后改进”的框架，在科研任务执行前自动归纳出具体、可验证的任务评分标准，用以指导智能体的执行、标准级验证与迭代修订，弥补开放式研究任务中成功标准不明确的缺陷。 |
| [^4] | [MULTI3IR: A Benchmark for Multi-perspective Multi-domain Multi-modal Information Retrieval](https://arxiv.org/abs/2608.30949) | 该论文提出Multi³IR基准，用于评估检索器在多领域、多模态场景下对开放式查询的多视角覆盖能力，并提出参数与标签高效的SPIN方法，通过学习噪声向量引导嵌入走向多样化语义方向，显著改善了现有多模态检索器的单一视角偏差问题。 |
| [^5] | [ECGQuest: Benchmarking and Fine-Tuning Language Models for Electrocardiography](https://arxiv.org/abs/2608.30893) | 该论文提出了ECGQuest——一个基于23本心电图参考文献和Computing in Cardiology会议论文构建的、包含21,808个真假判断问答对的数据集，用于评估和微调心电图专用语言模型，填补了现有基准测试缺乏心电图背景知识评估的空白。 |
| [^6] | [Playability-Aware Audio-to-Tablature Guitar Transcription via Diffusion Models](https://arxiv.org/abs/2608.30854) | 提出Noise2Fret扩散模型，通过对离散品位与弦目标的连续潜在表示生成吉他六线谱，并引入五个编码可演奏性约束的辅助损失，从而弥合音高准确性与物理可演奏性之间的差距。 |
| [^7] | [Learning from What You Retrieve: Online RL Fine-Tuning for Semantic Retrieval](https://arxiv.org/abs/2608.30753) | 该论文提出PAO（仅正优势）选择性强化学习优化方法，通过只对具有正优势的检索样本施加梯度更新，在文档索引必须冻结的工业约束下避免破坏预训练语义流形，从而提升大规模电商语义检索的端到端质量。 |
| [^8] | [Generative Retrieval for E-commerce: Jointly Learning Embedding and Codebook with Same Product Cluster](https://arxiv.org/abs/2608.30606) | 提出嵌入模型与码本联合训练的电商生成式检索方法，通过引入查询-商品和商品-商品交互建模，保证同一商品簇分配一致的ID，从而解决级联训练的误差累积问题并提升检索准确性。 |
| [^9] | [Preference Shapes Relevance: Cross-component Hierarchical Semantic Alignment for Personalized Generative Retrieval](https://arxiv.org/abs/2608.30553) | 该论文提出CHAP框架，通过跨组件层级语义对齐模块弥合动态查询意图与静态项目语义标识符之间的语义鸿沟，并结合用户偏好建模与高效解码实现个性化生成式检索。 |
| [^10] | [HF-SID: High-Fidelity Semantic IDs for Generative Retrieval in Location-Based Services](https://arxiv.org/abs/2608.30479) | 提出HF-SID方法，通过在表示阶段恢复地理、数值和结构保真度，解决现有语义ID在位置服务生成式检索中丢失细粒度差异信息的问题。 |
| [^11] | [Hi-Q: Hierarchical Evidence-guided Query Refinement for Multi-Hop Question Answering](https://arxiv.org/abs/2608.30468) | Hi-Q提出了一种以证据为条件的分层查询细化框架，通过在每个查询节点上利用解析算子判断证据是否支持当前查询单元，动态构建查询树，从而解决多跳问答中问题表达粒度与语料证据可检索粒度不匹配的核心瓶颈。 |
| [^12] | [CHASE: How Content Ecosystems Are Reshaped When Ranking Is the Only Target](https://arxiv.org/abs/2608.30466) | 提出CHASE受控仿真框架，模拟创作者反复针对LLM排名信号优化内容的过程，发现六个领域中内容质量与排名的一致性均持续下降，揭示了生成式引擎优化驱动的内容同质化现象。 |
| [^13] | [PRIME: Mitigating Subgroup Optimization Competition in Shared CTR Top Networks with Plug-in Residual Input-Conditioned Mixture of Expert](https://arxiv.org/abs/2608.30449) | 提出PRIME，一种插件式、以Dense为锚的低秩输入条件化残差专家混合结构，通过缓解CTR模型共享顶层网络中异质子群间的梯度竞争来提升模型性能，同时保留原有Dense层的初始函数、共享模式与容量。 |
| [^14] | [Beyond Polarization: The Generative Constraint of Chain-of-Thought in Pointwise Reranking](https://arxiv.org/abs/2608.30398) | 研究发现逐点重排序中链式思维模型表现不佳的根源在于通过离散文本传递连续相关性语义所造成的生成性约束，即使采用强化学习、细粒度监督和架构解耦等干预手段，这一瓶颈依然稳定存在且难以克服。 |
| [^15] | [RSLM: Training-Free Vector Quantization for Approximate Nearest Neighbor Search](https://arxiv.org/abs/2608.30384) | 提出了无需训练的向量量化编解码器RSLM，通过编码残差向量和校正L2范数，将大规模近似最近邻搜索的嵌入压缩至每维1-4比特，在降低内存成本和系统复杂度的同时保持或提升召回率。 |
| [^16] | [Beyond Ranking Accuracy: Evaluating LLM-Cited Feature Rationales for Next Basket Repurchase Recommendation](https://arxiv.org/abs/2608.30333) | 该研究超越传统排序准确率评估，构建了节奏、频率、近因性等可解释的复购行为特征，考察现成大语言模型能否作为下次购物篮推荐的有效评分器，以及其引用的特征理由是否真正携带与推荐结果相关的排序信号。 |
| [^17] | [PEARL: Front-Loading Relational Chains for Multi-Hop Table Retrieval](https://arxiv.org/abs/2608.30291) | PEARL是一个无需训练的多跳表格检索框架，通过在预识别的连接路径上离线生成多跳查询并将相关列重组为垂直分区的子表语料单元，实现了无需查询时LLM推理的高效多表检索，在3跳查询上R@2最高提升30.05%。 |
| [^18] | [CAMIE: Co-Engagement-Aware Multimodal Item Embeddings for Snap Dynamic Product Ads Retrieval](https://arxiv.org/abs/2608.30255) | 提出CAMIE框架，基于LLM/MLLM骨干网络并利用从用户行为中挖掘的协同互动商品对进行微调，生成统一的多模态商品嵌入，显著提升Snap动态商品广告的I2I检索效果。 |
| [^19] | [SetMIR: Multi-Interest Retrieval as Set Prediction](https://arxiv.org/abs/2608.30251) | SetMIR将多兴趣检索建模为集合预测问题，通过transformer编码用户行为历史、K个可学习查询结合匈牙利匹配解码出互不重复的兴趣嵌入，并利用存在分数实现动态检索预算，从而解决兴趣坍塌和静态分发两大问题。 |
| [^20] | [Doc-REFRAG: Rethinking Multimodal Document Retrieval-Augmented Generation](https://arxiv.org/abs/2608.30163) | 提出大规模多图像RAG数据集DocLongRAG和问题引导框架Doc-REFRAG，通过将视觉token压缩为粗粒度块并利用轻量级强化学习选择器选择性展开与问题相关的内容，在多图像文档问答中同时提升了准确率并大幅降低了计算开销。 |
| [^21] | [Understanding before verifying: Claim normalization for automated citation verification](https://arxiv.org/abs/2608.30145) | 该论文提出声明规范化方法，通过在检索与分类之前对原始引用声明应用三种重写策略，解决了范围不匹配、视角不匹配和命题纠缠三大问题，并据此构建了新的三阶段引文验证框架CNCV。 |
| [^22] | [E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval](https://arxiv.org/abs/2608.30130) | E-SENS是一种无需训练的重排序方法，通过为被排除概念提取“陷阱查询”并从检索分数中减去其相似度，有效惩罚与用户排除概念相关的文档，从而提升检索系统对负向约束的遵守能力。 |
| [^23] | [The Language of the Question Selects the Market: Query Language and Exit IP as Separable Factors in Commercial Recommendations from a Generative Search Interface](https://arxiv.org/abs/2608.30052) | 该研究通过对ChatGPT的234次对照实验发现，在生成式搜索界面的商业推荐中，查询语言而非出口IP位置决定了本地供应商是否会出现，且首要推荐结果在相同查询的重复运行中存在系统性的不稳定。 |
| [^24] | [Demand-Side Measurement for Generative Engine Optimization: Constructing and Validating a Million-Persona, Intent-Annotated Buyer Corpus](https://arxiv.org/abs/2608.30023) | 本文构建并验证了 PersonaGen-1M——首个包含超过 103 万个合成买家画像、覆盖 511 个行业和 4 种市场情境、并带有搜索意图标签与首选信息来源字段的买家语料库，为生成式引擎优化（GEO）研究提供了可与供给侧推荐测量相衔接的需求侧数据基础。 |
| [^25] | [Spatial Matryoshka Training for Multi-Granularity Visual Document Retrieval](https://arxiv.org/abs/2608.29951) | 该论文提出ColSNAP训练方法，通过空间嵌套平均池化使单个模型一次编码即可生成多级压缩的文档嵌入，从而在视觉文档检索中实现可在索引阶段灵活配置的精度-存储权衡，大幅降低存储成本。 |
| [^26] | [REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling](https://arxiv.org/abs/2608.29899) | REIGN通过在冻结引导网络生成的块嵌入序列上运行对比训练的双编码器，将词元级处理与文档级推理解耦，使长文档检索的训练成本相比分块Transformer微调降低约四个数量级。 |
| [^27] | [You Know What I Mean: A Benchmark for Agentic Conversational Reference Grounding](https://arxiv.org/abs/2608.29834) | 本文提出并形式化了对话引用定位问题，并构建了基于真实开发者聊天与GitHub工作区条目的RepoRef基准，用于评估智能体结合对话上下文与工具使用来解析间接引用的能力。 |
| [^28] | [ICEGR: An Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search](https://arxiv.org/abs/2608.29652) | 提出ICEGR框架，通过在生成式检索的语义ID构建、监督微调和偏好优化等整个训练流程中一致融入查询意图，解决电商搜索中查询意图不一致的问题，从而提升低曝光商品的检索效果和查询-商品相关性。 |
| [^29] | [LLMs Interpret, Embeddings Organize, Graphs Emerge: Agent-Driven Compilation of Scientific Knowledge](https://arxiv.org/abs/2608.29612) | 该论文提出ASKS系统，让大语言模型解读文献、嵌入几何组织变更、图结构呈现知识演化，将科学知识编译为可溯源、可检查的持久化状态转换过程，并通过编译56篇论文验证了其构建作者研究画像的能力。 |
| [^30] | [SnapBench: Benchmarking Snap-and-Ask Multimodal Retrieval for Mobile Interactions](https://arxiv.org/abs/2608.29607) | 提出了首个针对移动“拍照即问”多模态检索的成对鲁棒性基准SnapBench，通过53种受控损坏条件的大规模评估发现图像损坏显著降低检索性能，且仅用干净图像的检索往往优于图文联合检索。 |
| [^31] | [RePair: Turning Retrieval Failures into Counterfactual Hard Pairs](https://arxiv.org/abs/2608.29604) | RePair将检索中排名靠前的假阳性样本视为反事实支架，通过最小化修正其导致失败的局部残差，构建同模态困难正样本以及跨越决策边界的困难负样本对，产生互补的拉-推监督信号以提升视觉-语言检索性能。 |
| [^32] | [Adaptive Doubly Robust Off-Policy Evaluation for Ranking Policies under Diverse User Behavior](https://arxiv.org/abs/2608.29600) | 本文提出了一种面向排序策略的自适应双重稳健离线策略评估方法，通过自适应边缘化重要性权重在偏差与方差之间取得平衡，从而在多样化且未知的用户行为模型下实现可靠的策略评估。 |
| [^33] | [The Edge Spectrum of Choice-Derived Item Graphs: Strong and Weak Edges Encode Different Relations in Collaborative Filtering](https://arxiv.org/abs/2608.29578) | 该论文发现由选择模型导出的物品图中强边与弱边编码了性质不同的关系——强边连接被点击物品的同一列表内竞争者（正是排序梯度要推开的物品对），并将此形式化为平滑算子与排序梯度之间的符号不匹配，从而解释了为何直接用选择模型导出的图算子替换共同点击图无法提升协同过滤性能。 |
| [^34] | [What Are You Listening to? Temporal Music Grounding for Audio-to-Text Large Language Models](https://arxiv.org/abs/2608.29480) | 本文提出了时序音乐定位新任务及具有精确符号-音频对齐的MusicGroundingBench基准，用于评估音频-语言模型能否将音乐查询定位到具体时间段，发现现有模型在此任务上仍面临挑战，而任务特定训练可带来显著提升。 |
| [^35] | [Content Exploration Beyond the Feed: Creator Supply and the Shared Corpus](https://arxiv.org/abs/2608.29430) | 该论文通过某大型短视频平台的四项实验首次揭示了内容探索的双重价值——生产侧探索可使创作者发帖量提升8.55%，观众侧探索虽增加观看次数但减少观看时长，且探索引发的创作者供给与自然采纳会补充共享内容库，突破传统仅衡量观众侧效果的评估局限。 |
| [^36] | [Agents as Knowledge Integrator and Utilizer in Multimodal Recommendation](https://arxiv.org/abs/2608.29410) | 提出AgentMMRec框架，通过整合者与利用者两个协同智能体，将多模态内容与用户行为联合解释为可复用的知识记忆，并利用该知识优化模态物品图与推荐排序，从而弥合多模态信号与推荐目标之间的语义鸿沟。 |
| [^37] | [Personalized Recommender Systems for Gym Workouts: A Reinforcement Learning Approach](https://arxiv.org/abs/2608.29409) | 本文提出了一个基于强化学习的健身房训练推荐框架，将推荐范围从单纯的动作选择扩展到包含动作、组数、重复次数和负荷的完整训练处方，并能利用用户跳过动作的行为实现在线个性化。 |
| [^38] | [FISICA: A Deployed Service for Plantar-Pressure and Posture Assessment with Ontology-Grounded Recommendation](https://arxiv.org/abs/2608.29336) | FISICA是一个已部署的足底压力与体态评估服务，其核心创新在于用与被测者相同的方法测量3D虚拟形象并求解直至两者一致（基于采样不变的脊柱指标），取代传统的角度增益映射方法，将正常与驼背记录的区分度从0.9度提升到7.2度。 |
| [^39] | [Database-Augmented RAG for Automated Repair of REST API Misuses](https://arxiv.org/abs/2608.29290) | 本研究通过构建11种具有不同数据库结构的RAG配置并与基线方法对比修复率，评估了API规范在RAG数据库中的组织方式对REST API误用自动修复效果的影响。 |
| [^40] | [Cloud and On-Premises Deployment of Uzbek Legal RAG via Targeted Retriever Fine-Tuning](https://arxiv.org/abs/2608.29284) | 本文针对乌兹别克语法律问答这一低资源场景，构建了专家标注的检索基准与端到端评测基准，并通过针对性检索器微调，在云端成本约束和本地硬件与延迟约束两种部署模式下实现了高质量的法律RAG助手。 |
| [^41] | [Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge](https://arxiv.org/abs/2608.29249) | 本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。 |
| [^42] | [TAAL: Mitigating Early Beam Pruning in Generative Recommendation via Temporal Autoregressive Alignment](https://arxiv.org/abs/2608.29179) | 该论文提出TAAL方法，在训练阶段构建联合软目标并用前向KL损失对齐早期前缀分布、在推理阶段用逐点互信息校准候选分数，从而缓解生成式推荐中束搜索早期步骤的不可逆剪枝问题，在三个基准上显著提升检索性能。 |
| [^43] | [Context-Aware Interpretable Representations for Retrieval and Graph Convolutional Network Classification](https://arxiv.org/abs/2608.29004) | 该论文提出了一种将流形学习策略与基于排序的可解释图嵌入相结合的新型无监督框架，在保持低维度和下游任务高效能的同时为视觉表示提供可解释性，从而弥合了几何鸿沟与可解释性鸿沟。 |
| [^44] | [Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data](https://arxiv.org/abs/2608.29001) | 本文提出RaDE方法，利用基于排名的信息和代表性节点子集选择来实现图嵌入的维度可解释性，在降低计算成本的同时改进文本与多媒体数据的检索任务。 |
| [^45] | [MERIT: Mitigating Exposure Bias in Generative XMC for User-Interest Propensity Modeling](https://arxiv.org/abs/2608.28931) | MERIT框架通过基于黄金标签与难负样本混合排列的置换不变多目标损失这一自校正目标，缓解生成式极端多标签分类中的曝光偏差，从而提升电商场景下用户兴趣倾向建模的准确性。 |
| [^46] | [ASTRA - Agentic System for Ticket Resolution and Analysis](https://arxiv.org/abs/2608.28790) | ASTRA提出了一种由中央协调器调度三个专门信息收集智能体（历史案例检索、日志分析、领域知识检索）并通过裁判-协调器优化循环生成可验证、有证据支撑的故障排除报告的智能体系统，解决了现有工单自动化缺乏证据建模与来源追溯的问题。 |
| [^47] | [Weaving Visual Narratives: Agentic Image Bundle Composition Beyond Atomic Visual Matching](https://arxiv.org/abs/2608.28695) | 提出了图像束组合（IBC）这一新检索范式及首个基准数据集 IBCBench，将图像检索从对孤立图像的逐点匹配升级为从海量照片池中动态组合具有结构关联的连贯图像束。 |
| [^48] | [Can Large Language Models Identify Meaningful Touchpoints in Conversion Attribution?](https://arxiv.org/abs/2608.28649) | 该论文通过人工标注揭示了现有基于协同过滤的转化归因触点选择方法与用户语义意图之间的显著语义鸿沟，并系统评估了大语言模型识别隐式相关触点的能力以及不同提示策略和基础模型对性能的影响。 |
| [^49] | [NLP-Driven Knowledge Extraction and Thematic Classification of Translated Ancient Indian Medical Texts](https://arxiv.org/abs/2608.28608) | 本研究利用命名实体识别、BERTopic主题建模和Neo4j知识图谱等NLP技术，对《妙闻集》等古印度医学文献译本进行知识提取与主题分类，实现医学概念的语义化表示、知识检索与数字化保存。 |
| [^50] | [HubMixer: Progressive Latent Hub Mixing for Parameter-Efficient Feature Interaction in Recommendation](https://arxiv.org/abs/2608.27991) | 提出 HubMixer 架构，通过渐进式的潜在枢纽混合机制，避免在异构 token 空间中直接混合的低效问题，实现推荐系统中参数高效的特征交互。 |
| [^51] | [RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature](https://arxiv.org/abs/2608.27394) | RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。 |
| [^52] | [ExecRubrics: Executable Tool-Augmented Rubrics for Verifiable and Efficient Long-Form Evaluation](https://arxiv.org/abs/2608.22559) | ExecRubrics通过将评分标准转化为可执行的Python函数，实现了可验证、高效且能捕捉复杂依赖关系的长篇评估，替代了昂贵的黑盒LLM评判器。 |
| [^53] | [MITRE-SAGE: A Multi-Agent Cybersecurity Question-Answering model](https://arxiv.org/abs/2608.16921) | MITRE-SAGE通过多智能体检索增强生成框架，结合语义与结构化网络安全知识，解决了LLM在网络安全问答中的知识不足和幻觉问题，提升了可靠性和可解释性。 |
| [^54] | [Dense Expands, Sparse Anchors: Channel-Asymmetric Query Expansion for Hybrid Retrieval](https://arxiv.org/abs/2608.15851) | 本文提出DESA方法，通过通道非对称的查询扩展（稠密端正交残差扩展、稀疏端分数乘积锚定），解决了混合检索中固定截断值导致评估结果不稳定的问题。 |
| [^55] | [SPARC: Sequence-aware Progressive Attribute Routing and Compression Framework for Generative Recommendation](https://arxiv.org/abs/2607.25339) | 提出SPARC框架，通过序列感知的渐进式属性路由与压缩机制，解决生成式推荐中异构行为属性全展开导致输入过长、直接压缩又过早丢失上下文信息的矛盾。 |
| [^56] | [Rethinking Fairness in LLM-Based Recommender Systems: A Survey](https://arxiv.org/abs/2606.28340) | 这是首篇专门聚焦于基于大语言模型的推荐系统中公平性问题的综述，通过偏见机制与公平性目标的双维度框架，系统梳理了相关研究、评估方法与缓解策略。 |
| [^57] | [Querit-Reranker: Training Compact Multilingual Rerankers via Efficient Label-Free Distribution Adaptation](https://arxiv.org/abs/2606.19037) | Querit-Reranker提出了一套以数据为中心、无需人工标注的高效适配流水线，通过合成查询挖掘、教师分数软标签蒸馏和检查点合并，训练出紧凑且可部署的多语言重排序器。 |
| [^58] | [Exploring Autonomous Agentic Data Engineering for Model Specialization](https://arxiv.org/abs/2605.30407) | 该论文提出了“自主智能体数据工程”这一新任务，首次验证LLM能够自主执行端到端数据工程流水线来驱动模型专业化，其中GPT-5.2自主构建的训练数据使学生模型性能提升57.29%。 |
| [^59] | [LexPath: A domain-oriented multi-path framework for legal article retrieval](https://arxiv.org/abs/2605.30205) | 提出面向法律领域的多路径框架LexPath，通过结合IRAC引导的稀疏检索、基于法律层级与引用关系的稠密检索以及意图感知重排序，有效区分法律相关法条与仅文本相似的干扰法条，提升法条检索的准确性。 |
| [^60] | [Evidence Absence Is Not Evidence Insufficiency: Diagnosing NEI Construction Artifacts in Fact Verification](https://arxiv.org/abs/2605.26663) | 该论文提出NEI-CAP诊断协议，揭示事实核查基准中NEI标签的构建方式会引入捷径伪影，导致验证器识别“证据不足”的能力无法跨构建方式可靠迁移。 |
| [^61] | [SciAtlas: A Computable Atlas of Science for Knowledge-Grounded AI Research](https://arxiv.org/abs/2605.22878) | SciAtlas提出了一个共享的、机器可操作的跨学科学术知识基础设施，通过在统一模式下整合多层次知识并采用统一的神经符号检索机制，为知识驱动的人工智能科学研究提供了可靠的科学知识支撑。 |
| [^62] | [CASTLE: Contrastive and Seed-Guided Training for Cold-Start Natural Language Search](https://arxiv.org/abs/2605.21812) | CASTLE是一个基于LLM的冷启动框架，通过种子查询引导的提示生成真实合成查询、并利用预订会话构造的对比房源对生成接近零误报的相关性标签，支撑了Airbnb自然语言搜索的完整生命周期。 |
| [^63] | [Answer Bubbles: Information Exposure in AI-Mediated Search](https://arxiv.org/abs/2603.16138) | 该研究通过对五个搜索系统中11,000个真实查询的分析，发现生成式AI搜索在引用来源上存在显著的选择偏差，且搜索功能会使AI摘要中的模糊限定表述减少多达60%，进一步加剧了引用偏差并使表述更加自信。 |
| [^64] | [NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](https://arxiv.org/abs/2603.12824) | NanoVDR通过解耦文档索引与查询编码，使用冻结的2B参数VLM教师模型离线索引文档，并蒸馏出仅69M参数的纯文本学生模型来编码查询，在保持检索质量的同时大幅降低了推理延迟和GPU依赖。 |
| [^65] | [SORT: A Systematically Optimized Ranking Transformer for Industrial-scale Recommenders](https://arxiv.org/abs/2603.03988) | 本文提出了一种系统优化的排名Transformer模型，通过请求中心样本组织、局部注意力、查询剪枝和生成式预训练等创新，有效解决了工业级推荐中高特征稀疏性和低标签密度问题，并提升了硬件利用率。 |
| [^66] | [Building Better Encoder-only Cross-Encoders: A Controlled Study of Training Strategies for Neural Re-ranking](https://arxiv.org/abs/2603.03010) | 该研究通过162次受控训练实验系统比较了不同骨干网络与训练目标在神经重排序中的表现，发现强调相对比较的成对MarginMSE和列表InfoNCE目标始终优于其他方法。 |
| [^67] | [UniFAR: A Unified Facet-Aware Retrieval Framework for Scientific Documents](https://arxiv.org/abs/2602.23766) | 提出了UniFAR，一个统一的分面感知检索框架，通过多粒度表示与聚合模块在共享表示空间中同时支持文档-文档和问答-文档两种科学文献检索范式，从而融合二者的互补优势。 |
| [^68] | [MICE: Minimal Interaction Cross-Encoders for efficient Re-ranking](https://arxiv.org/abs/2602.16299) | 通过深入分析交叉编码器内部机制并移除多余的token交互，提出MICE新架构，在大幅降低计算开销的同时保持域内排序效果，并在域外场景中匹配甚至超越原有交叉编码器的性能。 |
| [^69] | [GeoGR: Enabling Spatio-Temporal Aware Industrial-scale Generative POI Recommendations](https://arxiv.org/abs/2602.10411) | 本文提出面向高德地图等导航型位置服务的地理生成式推荐框架GeoGR，通过地理感知的SID分词流水线解决高质量SID建模不足和大语言模型对齐不佳的问题，实现感知用户上下文变化、时空感知的工业级生成式POI推荐。 |
| [^70] | [HeMix: Scaling Industrial Ranking Models with Heterogeneous Token Mixing](https://arxiv.org/abs/2602.09387) | HeMix通过查询混合兴趣提取模块和HeteroMixer异构Token交互模块，在严格在线延迟约束下同时建模上下文相关与无关的用户兴趣，并实现高效的异构特征交互，从而扩展工业级排序模型。 |
| [^71] | [Multi-Source Retrieval and Reasoning for Legal Sentencing Prediction](https://arxiv.org/abs/2602.04690) | 提出了MSR²框架，将大语言模型的多源检索与推理和强化学习的过程级奖励相结合，显著提升了法律量刑预测的准确性和可解释性。 |
| [^72] | [Structural Anchor Pruning: Training-Free Multi-Vector Compression for Visual Document Retrieval](https://arxiv.org/abs/2601.20107) | 提出了一种免训练、与查询无关的结构化锚点剪枝框架（SAP），通过分数保留诊断、自动剪枝窗口选择和视觉入度中心性评分三项技术，在不依赖查询相关训练的情况下，有效压缩视觉文档检索中多向量索引的存储开销，即使在高压缩率下也能保持检索性能。 |
| [^73] | [CORE-T: COherent REtrieval of Tables for Text-to-SQL](https://arxiv.org/abs/2601.13111) | 提出无需训练、可扩展的CORE-T框架，通过LLM生成的表格用途元数据、预计算的表格兼容性缓存，以及“稠密检索—单次LLM筛选—两步增量调整”的流程，在大规模异构表格集合上实现连贯且可连接的多表检索，突破多表text-to-SQL的检索瓶颈。 |
| [^74] | [Loci Similes: A Benchmark for Extracting Intertextualities in Latin Literature](https://arxiv.org/abs/2601.07533) | 本文提出了Loci Similes，一个用于拉丁文学互文性检测的基准数据集，包含约17.6万个文本片段和1,490个专家验证的平行文本，为利用语言模型捕捉历史文本间超越词汇重叠的语义相似性提供了标准化评估基础。 |
| [^75] | [CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation](https://arxiv.org/abs/2511.22707) | 提出CoFiRec生成式推荐框架，通过从粗到细的分词方式显式建模物品语义的层次结构，从而更好地捕捉Web交互中用户意图的渐进演化过程。 |
| [^76] | [Evaluating Perspectival Biases in Cross-Modal Retrieval](https://arxiv.org/abs/2510.26861) | 本文提出跨文化、跨模态、跨语言的3XCM基准，揭示了多模态检索系统中系统性的视角偏差：图像到文本检索偏向流行语言条目而非语义忠实条目，文本到图像检索中存在语义对齐与文化关联之间的“牵引效应”，且低资源语言的相似性判断更易受文化熟悉视觉模式主导。 |
| [^77] | [LeMat-Synth: a multi-modal toolbox to curate broad synthesis procedure databases from scientific literature](https://arxiv.org/abs/2510.26824) | 本文提出开源多模态工具箱LeMat-Synth Parser，利用大语言模型和视觉语言模型从8.1万篇科学文献的文本与图表中自动提取结构化合成流程，构建了迄今最大、最多样化的无机材料合成数据集LeMat-Synth（含5.8万个合成流程）。 |
| [^78] | [On the Consistency and Performance of the Iterative Bayesian Update](https://arxiv.org/abs/2508.09980) | 实验证明迭代贝叶斯更新（IBU）在度量隐私机制下的分布估计性能显著优于矩阵求逆等方法，并从数学上解释了INV性能欠佳的原因，而在k-RR和RAPPOR等本地差分隐私机制下IBU与INV性能相近。 |
| [^79] | [Modeling Ranking Properties with In-Context Learning](https://arxiv.org/abs/2505.17736) | 提出了一种基于上下文学习的列表式LLM重排序方法，仅通过少量展示目标权衡的示例排序，无需任务特定训练即可实现群体公平性、极性多样性和主题多样性等排序目标。 |
| [^80] | [HintEval: An Open-Source Python Toolkit for Hint Generation and Hint Evaluation](https://arxiv.org/abs/2502.00857) | 本文提出了开源Python工具包HintEval，它统一了提示生成与提示评估流程，通过标准化数据集访问、支持答案感知与答案无关的生成方法以及多种评估指标，解决了该领域数据碎片化、格式不一致和评估工具不可用的问题，并支持可复现的多维度研究。 |
| [^81] | [Learning Personalized Prompts for Healthcare Guidance](https://arxiv.org/abs/2412.15957) | 提出个性化提示学习（PPL）框架，通过结合患者自身信息与临床相似病例的同伴信息构建初始个性化提示，并利用强化学习进行优化，使大语言模型能够生成与医生建议相符的个性化医疗健康指导。 |
| [^82] | [Annotation of Soft Onsets in String Ensemble Recordings](https://arxiv.org/abs/2211.08848) | 该论文通过研究24名参与者的标注者间一致性并扩展确定最一致标注者的算法，为弦乐合奏录音中柔和起音的标注建立了最佳实践，并发现音乐经验与标注质量和检测性能之间存在正相关关系。 |
| [^83] | [Advances and Challenges of Multi-task Learning Method in Recommender System: A Survey.](http://arxiv.org/abs/2305.13843) | 本文综述了多任务学习在推荐系统中的应用，提出了基于多任务学习技术的推荐方法分类，同时探讨了未来发展方向。 |

# 详细

[^1]: 面向检索增强生成中生物医学信息抽取的可配置语义分块

    Configurable Semantic Chunking for Biomedical Information Extraction in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2608.31139](https://arxiv.org/abs/2608.31139)

    提出了一种可配置的语义分块框架，通过实体保留窗口、触发词中心分块、命题优先抽取和层次化关系解析等策略替代固定大小分块，在仅替换 BioMedRAG 分块构建阶段的前提下，将 GM-CIHT 生物医学关系抽取的 F1 分数提升了 8.4 个百分点。

    

    BioMedRAG 为生物医学信息抽取引入了带有可学习分块评分器的检索增强生成方法。然而，该方法依赖于固定大小的分块，这可能会割裂语义证据。我们提出了一个可配置的语义分块框架，通过结合实体保留窗口、以触发词为中心的分块、命题优先抽取、分层触发词优先级排序以及层次化关系解析来解决这一局限。该框架与 BioMedRAG 集成，仅需替换分块构建阶段，同时保留嵌入模型、可学习分块评分器、生成器和评估协议。我们在生物医学关系抽取基准数据集（GM-CIHT、DDI、ChemProt）和不良事件分类（ADE）任务上对该框架进行了评估。在 GM-CIHT 数据集上，完整的混合配置达到了 82.6% 的 F1 分数，在我们的实验设置下比固定大小分块基线（74.2% F1）提高了 8.4 个百分点。跨数据集分析显示（摘要原文在此处截断）

    arXiv:2608.31139v1 Announce Type: new  Abstract: BioMedRAG introduced retrieval-augmented generation with a learned chunk scorer for biomedical information extraction. However, it relies on fixed-size chunking which can fragment semantic evidence. We propose a configurable semantic chunking framework that addresses this limitation by combining entity-preserving windows, trigger-centered chunking, proposition-first extraction, tiered trigger prioritization, and hierarchical relation resolution. The framework integrates with BioMedRAG by replacing only the chunk construction stage while preserving the embedding model, learned chunk scorer, generator, and evaluation protocol. We evaluate the framework on biomedical relation extraction benchmarks (GM-CIHT, DDI, ChemProt) and adverse event classification (ADE). On GM-CIHT, the full hybrid configuration achieves 82.6% F1, improving over the fixed-size baseline (74.2% F1) by 8.4 points under our experimental setup. Cross-dataset analysis show
    
[^2]: InsightToast：数据密集型会议侧信道中的主动信息检索与一目了然的可视化

    InsightToast: Proactive Information Retrieval & Glanceable Visualization in the Side Channel of Data-Rich Meetings

    [https://arxiv.org/abs/2608.31115](https://arxiv.org/abs/2608.31115)

    InsightToast通过多智能体LLM与RAG管道实时监测会议话语并主动检索信息，以临时提示和一目了然图表的形式在会议侧信道中提供有据可依的洞察，从而避免任务切换对会议参与和决策的干扰。

    

    会议中缺失机构背景信息会阻碍有效参与。检索相关信息往往需要耗费大量精力的任务切换——这些信息通常分散在异构的内部与外部来源中——这种切换会破坏个人的专注力和集体的对话流畅性，在决策制定等高认知负荷任务中尤为有害。我们提出了InsightToast，这是一个混合主动性（mixed-initiative）应用，它实时监测会议中的口头话语，在话题和信息需求出现时及时识别，并通过一个集成了检索增强生成（RAG）的多智能体大语言模型（LLM）管道主动检索相关信息，生成有来源依据的洞察，以简洁文本和一目了然的交互式图表形式呈现，并通过外围界面以临时提示（toast）的形式在对话的侧信道中传递。为了展示其产生意外洞见的潜力，我们……

    arXiv:2608.31115v1 Announce Type: cross  Abstract: Missing institutional context during meetings can impede effective participation. Retrieving relevant information, often scattered across heterogeneous internal and external sources, requires costly task-switching that disrupts both individual focus and collective conversational flow, particularly detrimental during cognitively demanding tasks such as decision-making. We introduce InsightToast, a mixed-initiative application that monitors verbal discourse in real time, identifies topics and informational needs as they emerge, and proactively retrieves relevant information through a multi-agent large language model (LLM)-based pipeline integrating retrieval-augmented generation (RAG) to produce source-grounded insights as succinct text and glanceable interactive charts, delivered through a peripheral interface as ephemeral toasts in the conversation's side channel. To demonstrate the potential for yielding serendipitous insights, we sho
    
[^3]: 先学会评估再改进：面向自动科研智能体的自动评分标准归纳

    Learning to Evaluate Before Improving: Automatic Rubric Induction for Automatic Research Agents

    [https://arxiv.org/abs/2608.31076](https://arxiv.org/abs/2608.31076)

    AutoSciRub 提出“先评估后改进”的框架，在科研任务执行前自动归纳出具体、可验证的任务评分标准，用以指导智能体的执行、标准级验证与迭代修订，弥补开放式研究任务中成功标准不明确的缺陷。

    

    自主科研智能体正越来越多地应用于端到端的科学工作流程中，包括文献综述、数据分析、实验以及报告生成。然而，开放式研究任务通常不会明确说明完成任务所需的分析、方法和成功标准。因此，智能体可能会遗漏重要的分析、使用不当的方法，或得出证据支持不足的结论。为解决这一问题，我们提出了 AutoSciRub，一个“评估优先”的框架，它在执行研究之前归纳出针对特定任务的可执行评分标准，并用它来指导执行、标准级别的验证以及迭代修订。AutoSciRub 将一个不明确的指令分解为原子化的科学目标，将其扎根于相关文献和任务可见的数据中，并综合出具体、可操作、可验证的标准。所得的评分标准使隐式的（评估信息显性化……摘要在此处截断）

    arXiv:2608.31076v1 Announce Type: cross  Abstract: Autonomous scientific research agents are increasingly applied to end-to-end scientific workflows, including literature review, data analysis, experimentation, and report generation. However, open-ended research tasks often do not clearly specify the analyses, methods, and success criteria required to complete the task. As a result, agents may miss important analyses, use inappropriate methods, or draw conclusions that are insufficiently supported by evidence. To address the problem, we present AutoSciRub, an evaluation-first framework that induces a task-specific executable rubric before research execution, and uses it to guide execution, criterion-level verification as well as iterative revision. AutoSciRub decomposes an underspecified instruction into atomic scientific goals, grounds them in relevant literature and task-visible data, and synthesizes specific, actionable, and verifiable criteria. The resulting rubric makes implicit e
    
[^4]: MULTI3IR：面向多视角、多领域、多模态信息检索的基准测试

    MULTI3IR: A Benchmark for Multi-perspective Multi-domain Multi-modal Information Retrieval

    [https://arxiv.org/abs/2608.30949](https://arxiv.org/abs/2608.30949)

    该论文提出Multi³IR基准，用于评估检索器在多领域、多模态场景下对开放式查询的多视角覆盖能力，并提出参数与标签高效的SPIN方法，通过学习噪声向量引导嵌入走向多样化语义方向，显著改善了现有多模态检索器的单一视角偏差问题。

    

    信息检索（IR）日益面向允许多元视角的开放式查询。然而，现有的IR基准主要聚焦于封闭式查询，即使是开放式基准，其查询所关联的支持文档也大多局限于单一主题领域和模态。我们提出了Multi³IR，这是一个评估检索器能否在多样化领域和模态下全面覆盖开放式查询多方面视角的基准。该基准包含104.9K个Stack Exchange查询，每个查询都标注了能够捕捉其隐含观点的视角描述。我们进一步提出了SPIN，这是一种参数高效且标签高效的方法，通过学习噪声向量将嵌入引导至多样化且有意义的语义方向。实验表明，现有多模态检索器存在单一视角偏差，而SPIN在Multi³IR上显著提升了视角覆盖率，并展现出良好的泛化能力。

    arXiv:2608.30949v1 Announce Type: new  Abstract: Information retrieval (IR) increasingly targets open-ended queries that admit diverse perspectives. Existing IR benchmarks, however, focus primarily on closed-ended queries, while even open-ended benchmarks largely consist of queries whose supporting documents span a single subject domain and modality. We introduce Multi$^3$IR, a benchmark that evaluates how well retrievers cover the multifaceted perspectives of open-ended queries across diverse domains and modalities. It comprises 104.9K Stack Exchange queries, each annotated with perspective descriptions that capture the query's implicit viewpoints. We further propose SPIN, a parameter- and label-efficient method that learns noise vectors to steer embeddings toward diverse yet meaningful semantic directions. Experiments show that existing multimodal retrievers suffer from single-perspective bias, while SPIN substantially improves perspective coverage on Multi$^3$IR and generalizes well
    
[^5]: ECGQuest：面向心电图学的语言模型基准测试与微调

    ECGQuest: Benchmarking and Fine-Tuning Language Models for Electrocardiography

    [https://arxiv.org/abs/2608.30893](https://arxiv.org/abs/2608.30893)

    该论文提出了ECGQuest——一个基于23本心电图参考文献和Computing in Cardiology会议论文构建的、包含21,808个真假判断问答对的数据集，用于评估和微调心电图专用语言模型，填补了现有基准测试缺乏心电图背景知识评估的空白。

    

    心电图（ECG）解读需要掌握心脏病学、电生理学、临床诊断、心电图波形、信号采集以及仪器设备等多方面的知识。然而，现有的语言模型基准测试主要评估广泛的医学知识或对单个心电图信号和图像的解读，而非心电图解读所需的更广泛的背景知识。我们开发了ECGQuest，这是一个基于文献构建的资源，用于评估和微调心电图专用语言模型。基于GPT-4o的流程从23本心电图参考文献以及2003-2025年Computing in Cardiology会议论文集中生成问题。最终数据集包含10,904个独特的真假判断题，并配有其对应的否定形式（共21,808个问答对）。我们在零样本设置下，在保留测试集上评估了三个商业模型和二十个开源语言模型，并使用低秩适配（Low-Rank Adaptation）方法对五个参数量为7-14B的开源模型进行了微调，其中BERT a（摘要在此处被截断）

    arXiv:2608.30893v1 Announce Type: new  Abstract: Electrocardiogram (ECG) interpretation requires knowledge of cardiology, electrophysiology, clinical diagnosis, ECG waveforms, signal acquisition, and instrumentation. Existing language-model benchmarks, however, primarily assess broad medical knowledge or interpretation of individual ECG signals and images rather than the broader contextual knowledge required for ECG interpretation. We developed ECGQuest, a literature-grounded resource for evaluating and fine-tuning ECG-specific language models. A GPT-4o-based pipeline generated questions from 23 ECG references and Computing in Cardiology proceedings from 2003-2025. The final dataset contains 10,904 unique True/False questions paired with their negated forms (21,808 Q&A pairs). We evaluated three commercial and 20 open-source language models on a held-out test set in a zero-shot setting. Five open-source models with 7-14B parameters were fine-tuned using Low-Rank Adaptation, with BERT a
    
[^6]: 基于扩散模型的考虑可演奏性的音频到吉他六线谱转录

    Playability-Aware Audio-to-Tablature Guitar Transcription via Diffusion Models

    [https://arxiv.org/abs/2608.30854](https://arxiv.org/abs/2608.30854)

    提出Noise2Fret扩散模型，通过对离散品位与弦目标的连续潜在表示生成吉他六线谱，并引入五个编码可演奏性约束的辅助损失，从而弥合音高准确性与物理可演奏性之间的差距。

    

    吉他六线谱转录不仅需要准确的音高检测，还需要将每个音符分配到特定的弦-品位位置，因为同一个音高可以在指板上的多个位置演奏。现有方法将此视为标准的分类问题，忽略了支配可演奏指法序列的音乐与物理约束。我们提出了Noise2Fret，一个用于音频到六线谱转录的扩散模型，它通过对离散品位和弦目标的连续潜在表示来生成六线谱，并以频谱和音频特征作为条件。为了弥合音高准确性与物理可演奏性之间的差距，我们在训练目标中直接引入了五个辅助损失，分别编码音高类距离、位置距离、五度圈距离、弦相似度以及手部跨距可行性。在GuitarSet和GOAT数据集上的实验表明，该模型优于基线方法，同时……

    arXiv:2608.30854v1 Announce Type: cross  Abstract: Guitar tablature transcription requires not only accurate pitch detection but also assigning each note to a specific string-fret position, as the same pitch can be played at multiple fretboard positions. Existing approaches treat this as a standard classification problem, ignoring the musical and physical constraints that govern playable fingering sequences. We propose Noise2Fret, a diffusion model for audio-to-tablature transcription that generates tablature through a continuous latent representation of discrete fret and string targets, conditioned on spectral and audio features. To bridge the gap between pitch accuracy and physical playability, we introduce five auxiliary losses encoding Pitch-Class Distance, Positional Distance, Circle-of-Fifths Distance, String Similarity, and Hand-Span Feasibility directly into the training objective. Experiments on GuitarSet and GOAT datasets demonstrate that the model outperforms baselines while
    
[^7]: 从检索结果中学习：面向语义检索的在线强化学习微调

    Learning from What You Retrieve: Online RL Fine-Tuning for Semantic Retrieval

    [https://arxiv.org/abs/2608.30753](https://arxiv.org/abs/2608.30753)

    该论文提出PAO（仅正优势）选择性强化学习优化方法，通过只对具有正优势的检索样本施加梯度更新，在文档索引必须冻结的工业约束下避免破坏预训练语义流形，从而提升大规模电商语义检索的端到端质量。

    

    在大规模电商检索中，双编码器检索器针对对比相似度进行优化，而下游的重排序器则捕捉更细粒度的相关性偏好；这种目标不匹配限制了端到端的检索质量。强化学习提供了一种利用奖励模型反馈来适配检索器的途径，但我们观察到，标准的策略梯度更新可能会损害嵌入的几何结构，尤其是在由于工业约束而必须保持文档索引冻结的情况下。为了解决这一问题，我们提出了PAO（仅正优势，Positive-Advantage-Only），一种选择性的强化学习优化方法。我们的分析表明，在冻结的高维空间中对负样本进行不加区分的惩罚（推远）会破坏预训练的语义流形。PAO选择性地仅对具有正优势的检索项应用梯度更新，从而有效地将查询嵌入拉向高奖励区域，同时保持全局……

    arXiv:2608.30753v1 Announce Type: cross  Abstract: In large-scale e-commerce retrieval, dual-encoder retrievers are op- timized for contrastive similarity, whereas downstream rerankers capture finer-grained relevance preferences; this objective mis- match limits end-to-end retrieval quality. Reinforcement Learning offers a way to use reward-model feedback for retriever adaptation, but we observe that standard policy-gradient updates can degrade embedding geometry, especially when the document index must remain frozen due to industrial constraints. To address this, we propose PAO (Positive-Advantage-Only), a selective RL optimization method. Our analysis reveals that in- discriminate penalization of negative samples (pushing away) in a frozen high-dimensional space disrupts pre-trained semantic man- ifolds. PAO selectively applies gradient updates only to retrieved items with positive advantages, effectively pulling query embed- dings toward high-reward regions while preserving global t
    
[^8]: 面向电商的生成式检索：基于同簇商品的嵌入与码本联合学习

    Generative Retrieval for E-commerce: Jointly Learning Embedding and Codebook with Same Product Cluster

    [https://arxiv.org/abs/2608.30606](https://arxiv.org/abs/2608.30606)

    提出嵌入模型与码本联合训练的电商生成式检索方法，通过引入查询-商品和商品-商品交互建模，保证同一商品簇分配一致的ID，从而解决级联训练的误差累积问题并提升检索准确性。

    

    随着大语言模型（LLM）的发展，生成式检索在电商场景中变得越来越重要。当前主流方法通常采用两阶段训练策略：先训练商品嵌入模型，再学习将嵌入映射为商品ID的码本。这种级联式方法存在两个主要问题：（1）误差累积——如果第一阶段嵌入模型产生了有偏差的表示，第二阶段的码本无法纠正这些错误，导致最终检索性能下降；（2）码本学习仅依赖商品嵌入，缺乏对查询-商品和商品-商品交互的建模。因此，属于同一簇的商品可能被码本分配到不一致的ID，进一步损害检索准确性。为解决这些问题，我们提出了一种新颖的方法，联合训练嵌入模型与码本，使同一商品簇能够获得一致的ID。

    arXiv:2608.30606v1 Announce Type: cross  Abstract: With the development of large language models (LLMs), generative retrieval is becoming increasingly important in e-commerce scenarios. Current mainstream approaches typically use a two-stage training strategy: first train a product embedding model, and then learn a codebook that maps embeddings to product IDs. This cascaded approach suffers from two major issues: (1) error accumulation-if the embedding model in the first stage produces biased representations, the codebook in the second stage cannot correct these errors, degrading final retrieval performance; and (2) codebook learning relies solely on product embeddings and lacks modeling of query-to-product and product-to-product interactions. As a result, products belonging to the same cluster may be assigned inconsistent IDs by the codebook, further hurting retrieval accuracy. To address these problems, we propose a novel method that jointly trains the embedding model and the codeboo
    
[^9]: 偏好塑造相关性：面向个性化生成式检索的跨组件层级语义对齐

    Preference Shapes Relevance: Cross-component Hierarchical Semantic Alignment for Personalized Generative Retrieval

    [https://arxiv.org/abs/2608.30553](https://arxiv.org/abs/2608.30553)

    该论文提出CHAP框架，通过跨组件层级语义对齐模块弥合动态查询意图与静态项目语义标识符之间的语义鸿沟，并结合用户偏好建模与高效解码实现个性化生成式检索。

    

    生成式检索通过将查询直接映射到具有强大候选项目表示能力的语义标识符，已成为一种有前景的范式。然而，现有仅从项目内容派生的语义标识符造成了语义鸿沟，无法将动态的查询意图与静态的项目表示对齐。此外，当前的生成式范式很少对用户行为序列进行建模，并且始终受到束搜索自回归解码带来的高推理延迟的瓶颈制约。为了应对这些挑战，我们提出了跨组件层级语义对齐的个性化生成式检索框架（CHAP），这是一种从层级视角出发的新型个性化生成式检索框架。首先，我们设计了一个层级语义对齐模块，将查询的潜在空间与项目的量化路径对齐，并同步多粒度语义。其次，我们构建了一个个性化……（摘要不完整，原文在此处截断）

    arXiv:2608.30553v1 Announce Type: cross  Abstract: Generative Retrieval (GR) has emerged as a promising paradigm by mapping queries directly to Semantic IDs (SIDs) with powerful representation capabilities for candidate items. However, existing SIDs derived solely from item content create a semantic gap, failing to align dynamic query intents with static item representations. Furthermore, current generative paradigms rarely model user behavior sequences and are always bottlenecked by the high inference latency of beam-search autoregressive decoding. To address these challenges, we propose $\textbf{C}$ross-component $\textbf{H}$ierarchical semantic $\textbf{A}$lignment for $\textbf{P}$ersonalized generative retrieval ($\textbf{CHAP}$), a novel personalized GR framework from a hierarchical perspective. First, we design a Hierarchical Semantic Alignment module to align query's latent space with item's quantization path and synchronize multi-granular semantics. Second, we construct a perso
    
[^10]: HF-SID：面向位置服务生成式检索的高保真语义ID

    HF-SID: High-Fidelity Semantic IDs for Generative Retrieval in Location-Based Services

    [https://arxiv.org/abs/2608.30479](https://arxiv.org/abs/2608.30479)

    提出HF-SID方法，通过在表示阶段恢复地理、数值和结构保真度，解决现有语义ID在位置服务生成式检索中丢失细粒度差异信息的问题。

    

    生成式检索在位置服务（LBS）领域日益受到关注，其中每个兴趣点（POI）被表示为一个语义ID（SID）。由于SID是POI信息传递给生成式模型的唯一通道，其未能保留的信息在解码时将无法恢复，而LBS检索对现有SID所模糊化的细粒度差异尤为敏感。具体而言：（1）大语言模型对连续坐标的嵌入是不连续的，其数值差异无法反映真实的地理距离；（2）动态数值属性的量纲差异巨大，相同的数值差距对一个属性可能是决定性的，而对另一个属性却可以忽略不计；（3）短文本无法传达层级从属关系，因为文本相似的POI可能属于不同的层级。因此，我们提出HF-SID，在表示阶段（即任何信息进入生成式模型之前）恢复地理、数值和结构上的保真度。

    arXiv:2608.30479v1 Announce Type: new  Abstract: Generative retrieval has attracted increasing attention in Location-Based Services (LBS), where each Point-of-Interest (POI) is represented as a Semantic ID (SID). As the SID is the only channel through which POI information reaches the generative model, whatever it fails to preserve is irrecoverable at decoding time, and LBS retrieval is especially sensitive to the fine-grained differences that existing SIDs blur. Specifically, (1) LLMs embed continuous coordinates discontinuously, so their numeric differences do not reflect true geographic distance; (2) dynamic numerical attributes differ vastly in scale, so an identical gap may be decisive for one attribute yet negligible for another; and (3) short text cannot convey hierarchical affiliation, as text-similar POIs may belong to different hierarchies. We therefore propose HF-SID, which restores geographic, numerical, and structural fidelity at the representation stage, before any inform
    
[^11]: Hi-Q：面向多跳问答的分层证据引导查询细化

    Hi-Q: Hierarchical Evidence-guided Query Refinement for Multi-Hop Question Answering

    [https://arxiv.org/abs/2608.30468](https://arxiv.org/abs/2608.30468)

    Hi-Q提出了一种以证据为条件的分层查询细化框架，通过在每个查询节点上利用解析算子判断证据是否支持当前查询单元，动态构建查询树，从而解决多跳问答中问题表达粒度与语料证据可检索粒度不匹配的核心瓶颈。

    

    多跳问答（QA）的一个核心瓶颈在于，问题所表达的粒度往往与语料库中证据可被检索的粒度不一致。现有方法通过在语料库上强加固定图结构、通过迭代重新表述查询、或通过执行生成的程序来解决这种不匹配，但这些策略并未显式地判断一个查询单元何时已被证据支持、何时应当被进一步细化。我们将这一瓶颈形式化为可检索粒度发现问题，并提出了Hi-Q——一个以证据为条件的分层查询细化框架。在每个查询节点上，解析算子检验检索到的证据是否支持当前查询单元；已解决的节点即终止，而未解决的节点则由一个保持依赖关系的二元算子进行扩展，并由语义覆盖验证器进行检查。因此，Hi-Q 构建出一棵查询树，其顶层……

    arXiv:2608.30468v1 Announce Type: new  Abstract: A central bottleneck in multi-hop Question Answering (QA) is that the granularity at which a question is expressed often differs from the granularity at which corpus evidence is retrievable. Existing methods address this mismatch by imposing fixed graph structures over the corpus, by iteratively reformulating the query, or by executing a generated program over it, but these strategies do not explicitly decide when a query unit is already supported by evidence and when it should be refined. We formulate this bottleneck as retrievable granularity discovery and introduce Hi-Q, an evidence-conditioned framework for hierarchical query refinement. At each query node, a resolution operator tests whether retrieved evidence supports the current query unit; resolved nodes terminate, while unresolved nodes are expanded by a dependency-preserving binary operator and checked by a semantic coverage verifier. Hi-Q therefore grows a query tree whose top
    
[^12]: CHASE：当排名成为唯一目标时，内容生态系统如何被重塑

    CHASE: How Content Ecosystems Are Reshaped When Ranking Is the Only Target

    [https://arxiv.org/abs/2608.30466](https://arxiv.org/abs/2608.30466)

    提出CHASE受控仿真框架，模拟创作者反复针对LLM排名信号优化内容的过程，发现六个领域中内容质量与排名的一致性均持续下降，揭示了生成式引擎优化驱动的内容同质化现象。

    

    生成式引擎优化（GEO）正被越来越多地用于提高内容在基于大语言模型的检索系统中的可见性，然而其在反复优化情形下对整个内容生态的群体层面影响仍鲜为人知。我们提出了排名信号利用下的内容同质化研究框架（CHASE），这是一个受控仿真框架，用于研究当创作者反复根据LLM排名信号调整文档时，内容生态系统会如何被重塑。我们以排名作为来源可见性的代理指标，并通过有依据的生成回复中的引用对该抽象进行验证，在六个领域上获得了0.853 ± 0.093的排名-引用AUC。随后，CHASE在不同领域上对排名、特征判别、重写和评估进行了20轮迭代。结果显示，所有六个领域中内容质量与排名的一致性均出现下降：从第0轮到第20轮，Spearman相关系数的变化范围在-0.107到-0.018之间，平均变化为-0.068，这意味着越接近……

    arXiv:2608.30466v1 Announce Type: new  Abstract: Generative Engine Optimization (GEO) is increasingly used to improve content visibility in LLM-based retrieval systems, yet its population-level effects under repeated optimization remain poorly understood. We introduce Content Homogenization under rAnking Signal Exploitation (CHASE), a controlled simulation framework for studying how content ecosystems are reshaped when creators repeatedly adapt documents to an LLM ranking signal. We use ranking as a proxy for source visibility and validate this abstraction against citations in grounded generated responses, obtaining a rank-citation AUC of 0.853 $\pm$ 0.093 across six domains. CHASE then iterates ranking, feature discrimination, rewriting, and evaluation over 20 rounds across different domains. Quality-ranking alignment decreases in all six domains: from R0 to R20, the change in Spearman's rho ranges from -0.107 to -0.018, with a mean change of -0.068, which means documents closer to th
    
[^13]: PRIME：利用插件式残差输入条件化专家混合缓解共享CTR顶层网络中的子群优化竞争

    PRIME: Mitigating Subgroup Optimization Competition in Shared CTR Top Networks with Plug-in Residual Input-Conditioned Mixture of Expert

    [https://arxiv.org/abs/2608.30449](https://arxiv.org/abs/2608.30449)

    提出PRIME，一种插件式、以Dense为锚的低秩输入条件化残差专家混合结构，通过缓解CTR模型共享顶层网络中异质子群间的梯度竞争来提升模型性能，同时保留原有Dense层的初始函数、共享模式与容量。

    

    点击率（CTR）模型在特征交互设计上各有不同，但其顶层网络通常仍是一个由所有样本共享的单一多层感知机。因此，异质的用户、物品和上下文子群会更新相同的参数；弱对齐的学习信号使得聚合梯度成为相互竞争方向之间的折中。我们在Avazu数据集上使用4个模型和4个语义字段研究了这种竞争现象。在所有架构中，语义子群表现出的Top-NN梯度余弦相似度低于按样本量和标签比例匹配的随机组，降幅为0.23-0.37。这种竞争现象启发了输入条件化专家的设计，但直接替换既有的Dense映射会改变其初始函数、共享模式和容量，从而掩盖了性能提升的真正来源。我们提出了PRIME（插件式残差输入条件化专家混合），一种以Dense为锚的低秩残差专家混合结构。PRIME将原始……

    arXiv:2608.30449v1 Announce Type: new  Abstract: Click-through rate (CTR) models vary in feature-interaction design, yet their top networks usually remain a single multilayer perceptron shared by all examples. Heterogeneous user, item, and context subgroups therefore update the same parameters; weakly aligned learning signals make the aggregate gradient a compromise among competing directions. We study the competition on Avazu with 4 models and 4 semantic fields. Across all architectures, semantic subgroups show lower Top-NN gradient cosine similarity than random groups matched by sample size and label ratio, with reductions of 0.23-0.37.   This competition motivates input-conditioned experts, but directly replacing an established Dense mapping changes its initial function, sharing pattern, and capacity, obscuring the source of gains. We introduce PRIME (Plug-in Residual Input-conditioned Mixture of Experts), a Dense-anchored mixture of low-rank residual experts. PRIME anchors the orig
    
[^14]: 超越极化：链式思维在逐点重排序中的生成性约束

    Beyond Polarization: The Generative Constraint of Chain-of-Thought in Pointwise Reranking

    [https://arxiv.org/abs/2608.30398](https://arxiv.org/abs/2608.30398)

    研究发现逐点重排序中链式思维模型表现不佳的根源在于通过离散文本传递连续相关性语义所造成的生成性约束，即使采用强化学习、细粒度监督和架构解耦等干预手段，这一瓶颈依然稳定存在且难以克服。

    

    在逐点文档重排序任务中，链式思维模型通常表现不如直接评分模型。虽然现有的诊断方法将这一现象归因于分类能力较弱、分数极化或校准失效，但针对性训练能否弥合这一差距仍不清楚。我们的实证研究首先证实了这一差距在高达320亿参数的模型规模上均保持稳定，从而排除了模型和数据容量方面的混淆因素。随后，我们运用强化学习、细粒度监督和架构解耦等手段进行压力测试，以显式修复这些偏差。尽管这些干预措施提高了分类准确率和绝对分数，但相对排序差距依然存在。这些发现表明，在逐点评分范式内，通过离散文本传递连续的相关性语义会限制排序信号的分辨率，揭示了一个在当前标准方法下稳定且难以克服的瓶颈。

    arXiv:2608.30398v1 Announce Type: new  Abstract: In pointwise document reranking, Chain-of-Thought models typically underperform direct scoring models. While existing diagnostics attribute this to inferior classification, score polarization, or calibration breakdown, whether targeted training can bridge this gap remains unclear. Our empirical study first confirms that this gap is stable across scales up to 32B parameters, ruling out model and data capacity confounders. We then apply stress tests utilizing reinforcement learning, fine-grained supervision, and architectural decoupling to explicitly repair these deviations. Although these interventions improve classification accuracy and absolute scores, the relative ranking gap persists. These findings suggest that, within the pointwise scoring paradigm, routing continuous relevance semantics through discrete text constrains ranking signal resolution, revealing a bottleneck that is stable and difficult to overcome under current standard 
    
[^15]: RSLM：用于近似最近邻搜索的无训练向量量化方法

    RSLM: Training-Free Vector Quantization for Approximate Nearest Neighbor Search

    [https://arxiv.org/abs/2608.30384](https://arxiv.org/abs/2608.30384)

    提出了无需训练的向量量化编解码器RSLM，通过编码残差向量和校正L2范数，将大规模近似最近邻搜索的嵌入压缩至每维1-4比特，在降低内存成本和系统复杂度的同时保持或提升召回率。

    

    通过引入RSLM（旋转缩放Lloyd-Max），这是一族无需训练的向量量化编解码器，可将嵌入向量压缩至每维1-4比特，我们降低了典型大规模近似最近邻（ANN）搜索系统的内存成本和内存带宽，同时降低了其复杂度，并在多个基准数据集上保持或提高了召回率。最先进的系统使用粗略分区过滤候选者，通过近似打分来缩小候选集合，然后使用更高精度的表示（通常每维至少8比特）对最佳候选重新打分。我们的相对化编解码器可以将这一精度降低到每维2-4比特。我们利用ANN系统的特性，在近似打分阶段和重新打分阶段都对残差向量而非完整向量进行编码。由于最大内积搜索（MIPS）对向量范数非常敏感，我们校正了量化向量的L2范数。我们的主要创新

    arXiv:2608.30384v1 Announce Type: new  Abstract: By introducing RSLM (Rotated Scaled Lloyd-Max), a family of training-free vector quantization codecs compressing embeddings to 1--4 bits per dimension, we reduce memory cost and memory bandwidth of a typical large-scale Approximate Nearest Neighbor (ANN) search system, while reducing its complexity and keeping or improving recall across multiple benchmark datasets. State-of-the-art systems filter candidates using coarse partitions, approximately score them to narrow the set, and then rescore the best with higher precision representations (often >=8 bits per dimension). Our relativized codecs can bring this down to 2--4 bits per dimension.   We use the properties of the ANN system to encode residual vectors instead of full vectors, both for the approximate scoring phase and the rescoring phase. Since Maximum Inner Product Search (MIPS) is very sensitive to vector norms, we correct the $L_2$ norms of quantized vectors. Our major innovation
    
[^16]: 超越排序准确率：评估大语言模型引用的特征理由用于下次购物篮复购推荐

    Beyond Ranking Accuracy: Evaluating LLM-Cited Feature Rationales for Next Basket Repurchase Recommendation

    [https://arxiv.org/abs/2608.30333](https://arxiv.org/abs/2608.30333)

    该研究超越传统排序准确率评估，构建了节奏、频率、近因性等可解释的复购行为特征，考察现成大语言模型能否作为下次购物篮推荐的有效评分器，以及其引用的特征理由是否真正携带与推荐结果相关的排序信号。

    

    下次购物篮复购推荐通常被表述为一个排序任务：给定客户的购买历史，系统对之前购买过、可能再次需要的商品进行排序。然而在生产环境中，排序准确率只是推荐质量的一个组成部分，客户还可能受益于关于“为什么现在推荐该商品”的简明证据。大语言模型（LLM）提供了一种潜在途径，可以通过基于特征、人类可读的理由来呈现此类证据，这些理由建立在可解释的行为信号之上。我们构建了涵盖购买节奏、频率、时间近因性、用户行为和商品热度的复购特征，并在两个公开的生鲜杂货数据集和一个专有零售数据集上评估大语言模型。我们研究了：（1）现成的大语言模型能否利用这些特征作为下次购物篮的评分器，并与启发式和有监督排序器进行比较；（2）大语言模型所引用的特征是否携带基于真实结果的排序信号。

    arXiv:2608.30333v1 Announce Type: cross  Abstract: Next-basket repurchase recommendation is commonly formulated as a ranking task: given a customer's purchase history, the system ranks previously purchased items that may be needed again. In production settings, however, ranking accuracy is only one component of recommendation quality. Customers may also benefit from concise evidence about why an item is recommended now. Large language models (LLMs) offer a potential way to surface such evidence through feature-based, human-readable rationales grounded in interpretable behavioral signals. We construct repurchase features spanning cadence, frequency, recency, user behavior, and item popularity, and evaluate LLMs on two public grocery datasets and one proprietary retail dataset. We investigate (1) whether off-the-shelf LLMs can use these features as next-basket scorers relative to heuristic and supervised rankers, and (2) whether LLM-cited features carry outcome-grounded ranking signal. F
    
[^17]: PEARL：面向多跳表格检索的关系链前置处理

    PEARL: Front-Loading Relational Chains for Multi-Hop Table Retrieval

    [https://arxiv.org/abs/2608.30291](https://arxiv.org/abs/2608.30291)

    PEARL是一个无需训练的多跳表格检索框架，通过在预识别的连接路径上离线生成多跳查询并将相关列重组为垂直分区的子表语料单元，实现了无需查询时LLM推理的高效多表检索，在3跳查询上R@2最高提升30.05%。

    

    尽管大语言模型（LLMs）在表格推理方面已展现出强大的能力，但由于现实世界数据的碎片化和关系性结构，检索相关表格仍然具有挑战性。现有工作通常依赖于整表表示，忽略了由连接关系引入的跨表语义。我们提出了PEARL，这是一个无需训练的框架，将检索范式转向基于垂直分区的子表编码。PEARL通过在预先识别的连接路径上生成多跳查询，并将相关列重组为垂直分区的语料库单元，从而离线增强检索语料库，无需查询时的LLM推理即可实现有效的多表检索。实验表明，PEARL始终优于现有方法，在3跳查询上R@2最高提升30.05%。源代码可在 https://github.com/SOOB2NHO/PEARL 获取。

    arXiv:2608.30291v1 Announce Type: new  Abstract: While large language models (LLMs) have shown strong capabilities in tabular reasoning, retrieving relevant tables remains challenging due to the fragmented and relational structure of real-world data. Existing work typically relies on whole table representations that overlook cross-table semantics induced by join relationships. We propose PEARL, a training-free framework that shifts the paradigm toward vertical partitioning-based sub-table encoding. PEARL augments the retrieval corpus offline by generating multi-hop queries over pre-identified join paths and reorganizing relevant columns into vertically partitioned corpus units, enabling effective multi-table retrieval without query-time LLM inference. Experiments show that PEARL consistently outperforms existing methods, with up to +30.05% gains in R@2 on 3-hop queries. The source code is available at https://github.com/SOOB2NHO/PEARL.
    
[^18]: CAMIE：面向Snap动态商品广告检索的协同互动感知多模态商品嵌入

    CAMIE: Co-Engagement-Aware Multimodal Item Embeddings for Snap Dynamic Product Ads Retrieval

    [https://arxiv.org/abs/2608.30255](https://arxiv.org/abs/2608.30255)

    提出CAMIE框架，基于LLM/MLLM骨干网络并利用从用户行为中挖掘的协同互动商品对进行微调，生成统一的多模态商品嵌入，显著提升Snap动态商品广告的I2I检索效果。

    

    商品到商品（I2I）检索是大规模推荐和广告系统中的核心基础能力。在生产环境的Snap动态商品广告（DPA）中，I2I检索面临两大挑战：分离的视觉、文本和多模态编码器使检索技术栈碎片化，且仅基于内容的训练无法将嵌入与驱动下游转化的协同互动行为对齐。我们提出了CAMIE，一个面向Snap DPA检索的协同互动感知多模态商品嵌入框架。CAMIE构建于LLM/MLLM骨干网络之上，利用其原生多模态接口在共享嵌入空间中表示商品图像和元数据。随后，它采用对称的批内InfoNCE目标函数，在从用户行为轨迹中挖掘的协同互动商品对上对骨干网络进行微调。离线评估中，CAMIE在Recall@10指标上超越了最强的商用多模态嵌入模型，并能从同一检查点以极小的质量损失支持纯文本检索。在线评估中，CAMIE服务……

    arXiv:2608.30255v1 Announce Type: new  Abstract: Item-to-item (I2I) retrieval is a core primitive in large-scale recommendation and advertising systems. In production Snap Dynamic Product Ads (DPA), I2I retrieval faces two challenges: separate visual, textual, and multimodal encoders fragment the retrieval stack, and content-only training does not align embeddings with the co-engagement behavior that drives downstream conversions. We present CAMIE, a co-engagement-aware multimodal item embedding framework for Snap DPA retrieval. CAMIE builds on LLM/MLLM backbones, using their native multimodal interfaces to represent item images and metadata in a shared embedding space. It then fine-tunes the backbone on co-engaged item pairs mined from user journeys with a symmetric in-batch InfoNCE objective. Offline, CAMIE outperforms the strongest commercial multimodal embedding model on Recall@10 and serves text-only retrieval from the same checkpoint with minimal quality loss. Online, CAMIE serve
    
[^19]: SetMIR：将多兴趣检索建模为集合预测问题

    SetMIR: Multi-Interest Retrieval as Set Prediction

    [https://arxiv.org/abs/2608.30251](https://arxiv.org/abs/2608.30251)

    SetMIR将多兴趣检索建模为集合预测问题，通过transformer编码用户行为历史、K个可学习查询结合匈牙利匹配解码出互不重复的兴趣嵌入，并利用存在分数实现动态检索预算，从而解决兴趣坍塌和静态分发两大问题。

    

    基于嵌入的检索是工业推荐系统的核心，但单一用户嵌入往往过于局限，难以捕捉用户多样化的兴趣。多兴趣检索通过使用多个用户嵌入来解决这一问题，然而现有方法仍存在两个问题：兴趣坍塌（即不同的嵌入学习到相同的兴趣）和静态分发（即无论某些嵌入是否必要，服务时都使用固定的检索预算）。我们提出SetMIR，将多兴趣检索视为一个集合预测问题。SetMIR使用transformer编码用户的行为历史，并利用K个可学习的查询来解码出一组用户兴趣，每个兴趣生成一个检索嵌入和一个存在分数。在训练过程中，匈牙利匹配将目标一一对应地分配给各个查询，使得被匹配的查询能够学习到彼此不同的兴趣，而存在预测头则学习判断哪些查询是活跃的。在服务阶段，SetMIR利用存在分数（原文此处被截断）来动态选择嵌入。

    arXiv:2608.30251v1 Announce Type: new  Abstract: Embedding-based retrieval is at the core of industrial recommender systems, but a single user embedding is often too limited to capture a user's diverse interests. Multi-interest retrieval addresses this by using multiple user embeddings, yet existing methods still suffer from two issues: interest collapse, where different embeddings learn the same interest, and static dispatch, where serving uses a fixed retrieval budget even when some embeddings are unnecessary. We propose SetMIR, which treats multi-interest retrieval as a set prediction problem. SetMIR encodes a user's behavior history with a transformer and uses K learnable queries to decode a set of user interests, each producing a retrieval embedding and a presence score. During training, Hungarian matching assigns targets to queries one-to-one, so matched queries learn distinct interests and the presence head learns which queries are active. At serving time, SetMIR uses presence s
    
[^20]: Doc-REFRAG：重新思考多模态文档检索增强生成

    Doc-REFRAG: Rethinking Multimodal Document Retrieval-Augmented Generation

    [https://arxiv.org/abs/2608.30163](https://arxiv.org/abs/2608.30163)

    提出大规模多图像RAG数据集DocLongRAG和问题引导框架Doc-REFRAG，通过将视觉token压缩为粗粒度块并利用轻量级强化学习选择器选择性展开与问题相关的内容，在多图像文档问答中同时提升了准确率并大幅降低了计算开销。

    

    现实世界的知识存在于多模态文档中，因此需要检索增强生成（RAG）来实现准确的问答。然而，现有多模态RAG模型主要针对单图像或封闭文档场景设计，在真实的多图像场景中准确率有限。此外，处理大量检索到的图像会因无关视觉token而带来巨大的计算开销。为应对这些挑战，我们提出了DocLongRAG，一个包含34.3万问答对的大规模数据集，每个问答对平均关联37.4张检索图像，以反映真实的RAG工作流程。基于该数据集，我们提出Doc-REFRAG，一种问题引导的框架，它将视觉token压缩为粗粒度块，并通过轻量级的基于强化学习的选择器选择性地展开与问题相关的块。在六个基准上的实验表明，Doc-REFRAG优于十一个强大的基线模型，达到了最先进的性能。

    arXiv:2608.30163v1 Announce Type: new  Abstract: Real-world knowledge resides in multimodal documents, necessitating retrieval-augmented generation (RAG) for accurate question answering. However, existing multimodal RAG models are primarily designed for single-image or closed-document settings and exhibit limited accuracy in realistic multi-image scenarios. Moreover, processing numerous retrieved images incurs substantial computational overhead from irrelevant visual tokens. To address these challenges, we introduce DocLongRAG, a large-scale dataset of 343K question--answer pairs, each associated with an average of 37.4 retrieved images to reflect authentic RAG workflows. Building on this dataset, we propose Doc-REFRAG, a question-guided framework that compresses visual tokens into coarse chunks and selectively expands question-relevant ones via a lightweight RL-based selector. Experiments on six benchmarks show that Doc-REFRAG outperforms eleven strong baselines, achieving state-of-th
    
[^21]: 先理解后验证：面向自动化引文验证的声明规范化

    Understanding before verifying: Claim normalization for automated citation verification

    [https://arxiv.org/abs/2608.30145](https://arxiv.org/abs/2608.30145)

    该论文提出声明规范化方法，通过在检索与分类之前对原始引用声明应用三种重写策略，解决了范围不匹配、视角不匹配和命题纠缠三大问题，并据此构建了新的三阶段引文验证框架CNCV。

    

    引文准确性因其对研究可靠性的重要性已被研究了数十年。内容层面的引文验证旨在评估学术声明的可靠性。近期工作采用了继承自事实核查的两阶段“检索-分类”框架。然而，这种设计忽略了原始引用声明的复杂性，并给验证系统带来了三个问题，即范围不匹配、视角不匹配和命题纠缠。这些问题增加了检索和分类的难度，从而限制了模型性能。基于这一研究空白，我们提出声明规范化方法，在检索和分类之前对原始引用声明应用三种重写策略，使每个下游模型只需执行单一的、明确定义的任务。基于该方法，我们开发了声明规范化引文验证，这是一个全新的三阶段框架，由声明规范化……

    arXiv:2608.30145v1 Announce Type: new  Abstract: Citation accuracy has been studied for decades because of its importance to research reliability. Content-level citation verification assesses the reliability of scholarly claims. Recent work adopts a two-stage retrieval-classification framework inherited from fact-checking. However, this design overlooks the complexity of the raw citing claim and introduces three issues into the verification system, namely scope mismatch, perspective mismatch, and proposition entanglement. These issues increase the difficulty of retrieval and classification, thereby limiting model performance. Motivated by this gap, we propose claim normalization, which applies three rewriting strategies to the raw citing claim before retrieval and classification, allowing each downstream model to perform a single, well-defined task. Building on this method, we develop Claim-Normalized Citation Verification (CNCV), a new three-stage framework consisting of claim normali
    
[^22]: E-SENS：面向负约束检索的排斥敏感惩罚方法

    E-SENS: Exclusion-Sensitive Penalization for Negative-Constraint Retrieval

    [https://arxiv.org/abs/2608.30130](https://arxiv.org/abs/2608.30130)

    E-SENS是一种无需训练的重排序方法，通过为被排除概念提取“陷阱查询”并从检索分数中减去其相似度，有效惩罚与用户排除概念相关的文档，从而提升检索系统对负向约束的遵守能力。

    

    检索增强语言模型在检索器提供用户明确排除概念的相关证据时，可能无法遵守负向约束。除了显式否定之外，查询还可能要求答案包含一个概念而排除另一个概念，或者要求实体属于某一类别但与密切相关的实例不同。由于被排除的概念仍然出现在查询文本中，稠密检索器可能会对与该概念相关的文档赋予高相似度，即使用户明确要求避开它。我们提出了E-SENS，一种面向否定敏感检索的无训练重排序方法。E-SENS为被排除的一方提取一个紧凑的“陷阱查询”，并从原始查询的检索分数中减去陷阱查询的相似度。在ExcluIR基准上，E-SENS在四个嵌入模型上展现出清晰的召回率-违规权衡，并在保持召回率的设置下有效减少了陷阱检索。

    arXiv:2608.30130v1 Announce Type: cross  Abstract: Retrieval-augmented language models can fail to respect negative constraints when the retriever supplies evidence about concepts the user explicitly excluded. Beyond explicit negation, queries may ask for answers that include one concept while excluding another, or for entities that belong to a category but differ from a closely related instance. Because the excluded concept still appears in the query text, dense retrievers may assign high similarity to documents about that concept even when the user asks to avoid it. We introduce E-SENS, a training-free reranking method for negation-sensitive retrieval. E-SENS extracts a compact trap query for the excluded side and subtracts trap-query similarity from the original-query retrieval score. On ExcluIR, E-SENS shows a clear recall-violation trade-off across four embedding models and reduces trap retrieval at recall-preserving settings.
    
[^23]: 问题的语言选择市场：查询语言与出口IP作为生成式搜索界面商业推荐中可分离的因素

    The Language of the Question Selects the Market: Query Language and Exit IP as Separable Factors in Commercial Recommendations from a Generative Search Interface

    [https://arxiv.org/abs/2608.30052](https://arxiv.org/abs/2608.30052)

    该研究通过对ChatGPT的234次对照实验发现，在生成式搜索界面的商业推荐中，查询语言而非出口IP位置决定了本地供应商是否会出现，且首要推荐结果在相同查询的重复运行中存在系统性的不稳定。

    

    当生成式搜索界面回答一个商业问题时，它提及哪个市场的产品，在模型对产品进行推理之前就已经决定了。我们报告了一项对照探针实验，共234次运行，针对未登录状态的ChatGPT网页界面和OpenAI API，于2026年8月29日和30日收集，涵盖四个出口国家和六种查询语言，每个单元格进行六次相同的运行。得到三个结果。第一，首要推荐是不稳定的：在六个提示词中的四个上，六次相同运行的结果发生了变化，且该不稳定率在浏览器界面和API中（无论是否启用网络搜索）完全相同，因此不稳定性是系统本身的属性，而非某个交互界面的属性。第二，决定本地供应商是否出现的因素是查询语言，而非位置。当查询语言与所在国家匹配时，全球品牌在24次运行中仅获胜1次；而在相同的网络连接下用英语提问时，本地品牌在爱沙尼亚和土耳其的6次运行中获胜0次。第三，（摘要在此处截断）

    arXiv:2608.30052v1 Announce Type: cross  Abstract: When a generative search interface answers a commercial question, which market's products it names is decided before the model reasons about the products. We report a controlled probe of 234 runs against the logged-out ChatGPT web interface and the OpenAI API, collected on 29 and 30 August 2026 across four exit countries and six query languages, with six identical runs per cell. Three results. First, the top recommendation is unstable: it changed across six identical runs on four of six prompts, and that rate was identical in the browser interface and in the API with web search both enabled and disabled, so instability is a property of the system and not of the surface. Second, query language, and not location, decides whether local suppliers appear at all. Where the query language matched the country, a global brand won 1 of 24 runs; asked in English on the same connections, local brands took 0 of 6 runs in Estonia and Turkiye. Third,
    
[^24]: 面向生成式引擎优化的需求侧测量：构建并验证百万级、带意图标注的买家画像语料库

    Demand-Side Measurement for Generative Engine Optimization: Constructing and Validating a Million-Persona, Intent-Annotated Buyer Corpus

    [https://arxiv.org/abs/2608.30023](https://arxiv.org/abs/2608.30023)

    本文构建并验证了 PersonaGen-1M——首个包含超过 103 万个合成买家画像、覆盖 511 个行业和 4 种市场情境、并带有搜索意图标签与首选信息来源字段的买家语料库，为生成式引擎优化（GEO）研究提供了可与供给侧推荐测量相衔接的需求侧数据基础。

    

    ChatGPT、Gemini 和 Perplexity 等生成式引擎会直接回答买家的问题，并在答案中列出一份简短的品牌清单。要研究品牌如何进入或未能进入这份清单，就需要需求侧数据：某一品类中的买家会提出什么问题、需要什么信息、信任哪些信息来源。现有的大型买家画像语料库是为训练数据多样性而构建的，既没有分阶段的搜索意图标签，也没有首选信息来源字段，因此无法与供给侧的推荐测量数据相衔接。我们构建并验证了 PersonaGen-1M，这是一个包含 1,031,732 个合成买家画像的语料库，覆盖 511 个行业标签和 4 种市场情境，共包含 19,416,821 个结构化行为属性，其中 5,160,046 个为搜索查询。每个买家画像都带有一个覆盖其查询集合的单一 primary_intent 标签（78.3% 为信息型，17.4% 为商业型，4.3% 为交易型），以及一个 preferred_sources 字段，用于标明……（摘要原文在此处截断）

    arXiv:2608.30023v1 Announce Type: cross  Abstract: Generative engines such as ChatGPT, Gemini, and Perplexity answer buyer questions directly and name a shortlist of brands inside the answer. Studying how brands enter or fail to enter that shortlist requires demand-side data: what buyers in a category ask, what information they need, and which sources they trust. Existing large persona corpora are built for training-data diversity and carry neither a staged search-intent label nor a preferred-sources field, so they cannot be joined to supply-side recommendation measurements. We built and validated PersonaGen-1M, a corpus of 1,031,732 synthetic buyer personas spanning 511 industry labels and 4 market contexts, carrying 19,416,821 structured behavioral attributes, 5,160,046 of them search queries. Each persona carries a single primary_intent label covering its query set (78.3% informational, 17.4% commercial, 4.3% transactional) and a preferred_sources field naming the source types that 
    
[^25]: 面向多粒度视觉文档检索的空间套娃训练

    Spatial Matryoshka Training for Multi-Granularity Visual Document Retrieval

    [https://arxiv.org/abs/2608.29951](https://arxiv.org/abs/2608.29951)

    该论文提出ColSNAP训练方法，通过空间嵌套平均池化使单个模型一次编码即可生成多级压缩的文档嵌入，从而在视觉文档检索中实现可在索引阶段灵活配置的精度-存储权衡，大幅降低存储成本。

    

    多模态后期交互检索器通过将每个页面表示为逐块嵌入并在词元级别进行匹配，在视觉丰富的文档上实现了强大的检索性能。然而，这种方法会带来高昂的存储成本。现有的压缩方法通常在索引时固定单一压缩级别，限制了灵活性。我们提出了ColSNAP（空间嵌套平均池化），这是一种训练方法，可直接从骨干网络的块网格生成嵌套的压缩级别层次结构。通过将块嵌入空间池化为逐渐粗化的层级并同时训练所有层级，单个模型无需更改架构即可学会支持多个压缩级别的检索。至关重要的是，单次编码即可生成所有层级，使得精度-存储权衡可以在索引时进行配置以匹配可用的存储预算，而不是在训练期间被固定。我们证明模型……

    arXiv:2608.29951v1 Announce Type: new  Abstract: Multi-modal late-interaction retrievers achieve strong retrieval on visually rich documents by representing each page as per patch embeddings and matching at the token level. However, this approach incurs high storage costs. Existing compression methods typically fix a single compression level at indexing time, limiting flexibility. We present ColSNAP (Spatial Nested Average Pooling)1, a training method that generates a nested hierarchy of compression levels directly from a backbone's patch grid. By spatially pooling patch embeddings into pro- gressively coarser tiers and training all tiers simultaneously, a single model learns to support retrieval at multiple compression levels without architectural changes. Crucially, a single encoding pass yields every tier, enabling the accuracy-storage trade-off to be configured at indexing time to match avail- able storage budgets, rather than being fixed during training. We demonstrate that models
    
[^26]: REIGN：利用集成引导网络的翻新嵌入实现高效的上下文长度扩展

    REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling

    [https://arxiv.org/abs/2608.29899](https://arxiv.org/abs/2608.29899)

    REIGN通过在冻结引导网络生成的块嵌入序列上运行对比训练的双编码器，将词元级处理与文档级推理解耦，使长文档检索的训练成本相比分块Transformer微调降低约四个数量级。

    

    对长文档进行稠密检索的代价高昂。词元级编码器在序列长度上呈二次方扩展，而大多数长上下文嵌入模型只能通过架构上的变通方法或拉长十亿参数级大语言模型才能达到32K词元。我们提出REIGN（Refurbished Embeddings with Integrated Guidance Networks，集成引导网络的翻新嵌入），这是一个经过对比训练的双编码器，它在由冻结的引导网络（GN）生成的上下文化块嵌入序列上运行，而不是在原始词元上运行。REIGN针对多块输入，主要用于文档到文档的检索；单块输入则仍由GN处理。通过将词元级处理与文档级推理解耦，并将GN嵌入缓存到磁盘，相对于分块Transformer微调，每个文档的训练成本降低了大约四个数量级。我们还发布了一个合成的长文档检索基准，用于长上下文长度下的对比训练与评估。

    arXiv:2608.29899v1 Announce Type: cross  Abstract: Dense retrieval over long documents is expensive. Token-level encoders scale quadratically in sequence length, and most long-context embedding models reach 32K tokens only through architectural workarounds or by stretching billion-parameter LLMs. We propose REIGN (Refurbished Embeddings with Integrated Guidance Networks), a contrastively trained bi-encoder that operates on sequences of contextualised chunk embeddings from a frozen Guidance Network (GN) rather than on raw tokens. REIGN targets multi-chunk inputs, primarily for document-to-document retrieval; single-chunk inputs stay with the GN. Decoupling token-level processing from document-level reasoning, and caching the GN embeddings to disk, cuts per-document training cost by roughly four orders of magnitude relative to chunked Transformer fine-tuning. We also release a synthetic long-document retrieval benchmark for contrastive training and evaluation at long context lengths. Acr
    
[^27]: 你知道我的意思：一个面向智能体对话引用定位的基准测试

    You Know What I Mean: A Benchmark for Agentic Conversational Reference Grounding

    [https://arxiv.org/abs/2608.29834](https://arxiv.org/abs/2608.29834)

    本文提出并形式化了对话引用定位问题，并构建了基于真实开发者聊天与GitHub工作区条目的RepoRef基准，用于评估智能体结合对话上下文与工具使用来解析间接引用的能力。

    

    协作性对话中经常包含目标为间接而非明确命名的引用：解析“这看起来像昨天讨论过的修复方案”这类表述，需要将对话上下文与周围工作区（可通过API或用户界面访问）中的证据相结合。我们将这一问题形式化为对话引用定位：使用给定的工具集，将对话中的引用解析为说话者所指的唯一外部条目。CoRG具有挑战性，因为它结合了分布在对话和外部工作区中的词汇、语义和时间线索。智能体必须将这些异构信号转化为有效的工具使用：制定策略、发现可能的候选对象、检查其元数据和内容，并排除相近的替代项。我们通过RepoRef来研究CoRG，这是一个包含400个开发者聊天片段的基准测试，这些片段基于GitHub issues、pull requests等真实工作区条目构建。

    arXiv:2608.29834v1 Announce Type: new  Abstract: Collaborative conversations frequently contain references whose targets are indirect rather than named: resolving "this looks like the fix discussed yesterday" requires combining conversational context with evidence from the surrounding workspace which is accessible through APIs or user interfaces. We formalize this problem as Conversational Reference Grounding (CoRG): using a given set of tools to resolve a reference in conversation to the unique external item intended by the speaker. CoRG is challenging because it combines lexical, semantic, and temporal cues distributed across the conversation and the external workspace. Agents must translate these heterogeneous signals into effective tool use: formulating strategies, discovering plausible candidates, inspecting their metadata and content, and ruling out close alternatives. We study CoRG through RepoRef, a benchmark of 400 developer-chat segments grounded in GitHub issues, pull reques
    
[^28]: ICEGR：面向电商搜索的意图连贯端到端生成式检索框架

    ICEGR: An Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search

    [https://arxiv.org/abs/2608.29652](https://arxiv.org/abs/2608.29652)

    提出ICEGR框架，通过在生成式检索的语义ID构建、监督微调和偏好优化等整个训练流程中一致融入查询意图，解决电商搜索中查询意图不一致的问题，从而提升低曝光商品的检索效果和查询-商品相关性。

    

    生成式检索（GR）在电商搜索中前景广阔，但现有方法难以在整个训练流程中保持查询意图的一致性。首先，基于静态商品信息的语义ID（SID）构建方式限制了SID编码商品-意图关联的能力。其次，尽管监督微调（SFT）能够学习整个商品目录中的商品-SID映射，但由于查询到SID的训练仅依赖在线日志，低曝光商品仍然缺乏真实的查询意图监督，导致这些商品的检索性能不佳。第三，面向业务的偏好优化可能偏向热门或高价值商品，而非最匹配查询意图的商品，从而削弱了查询与商品之间的相关性。为解决这些问题，我们提出了ICEGR，一个面向电商搜索的意图连贯端到端生成式检索框架，它在整个GR训练流程中一致地融合查询意图（原文摘要在此处截断）。

    arXiv:2608.29652v1 Announce Type: new  Abstract: Generative Retrieval (GR) is promising for e-commerce search, yet existing methods struggle to maintain query-intent consistency throughout the training pipeline. First, semantic ID (SID) construction based on static product information limits the ability of SIDs to encode product-intent associations. Second, although supervised fine-tuning (SFT) learns product-SID mappings across the catalog, low-exposure products still lack real query-intent supervision because query-to-SID training relies solely on online logs, resulting in poor retrieval performance for these products. Third, business-oriented preference optimization may favor popular or high-value products over those that best match the query intent, weakening query-product relevance. To address these issues, we propose ICEGR, an Intent-Coherent End-to-End Generative Retrieval Framework for E-commerce Search that integrates query intent consistently throughout the GR training pipeli
    
[^29]: 大语言模型负责解读，嵌入负责组织，图谱自然涌现：智能体驱动的科学知识编译

    LLMs Interpret, Embeddings Organize, Graphs Emerge: Agent-Driven Compilation of Scientific Knowledge

    [https://arxiv.org/abs/2608.29612](https://arxiv.org/abs/2608.29612)

    该论文提出ASKS系统，让大语言模型解读文献、嵌入几何组织变更、图结构呈现知识演化，将科学知识编译为可溯源、可检查的持久化状态转换过程，并通过编译56篇论文验证了其构建作者研究画像的能力。

    

    持续的科学工作需要一个知识基底，它能够跨任务传递解释，并保留通往原始证据的路径。我们将这一过程称为“科学知识编译”，并在ASKS（智能体驱动的科学知识系统）中加以实现。对于每个文献来源，大语言模型会生成一个可读的Wiki视图和面向机器的语义表示。确定性检查将后者转换为文档局部的GraphDelta（图谱增量），嵌入几何与显式图规则共同将提议的变更整合进持久化状态中。每一次知识摄取都是对积累知识的一次可检查的状态转换，编译后的Wiki视图和图谱视图均与保留的原始文献记录相链接。我们通过按时间顺序编译来自同一研究项目的56篇已发表论文来检验这一过程。分支存活率、跨论文支持、谱系、覆盖率和变更率等指标共同生成了一份以张量网络方法为中心的、可溯源至原始文献的作者研究画像。

    arXiv:2608.29612v1 Announce Type: new  Abstract: Sustained scientific work requires a knowledge substrate that carries interpretation across tasks and preserves paths to source evidence. We call this process \emph{scientific knowledge compilation} and implement it in ASKS, the \emph{Agent-Driven Scientific Knowledge System}. For each source, an LLM produces a readable Wiki view and machine-facing semantics. Deterministic checks convert the latter into a document-local GraphDelta, and embedding geometry together with explicit graph rules integrates the proposed changes into persistent state. Each ingest is an inspectable state transition over accumulated knowledge, with compiled Wiki and graph views linked to the preserved source record. We examine this process by chronologically compiling 56 published papers from one research program. Branch survival, cross-paper support, lineage, coverage, and churn yield a source-traceable author research portrait centered on tensor-network methods, 
    
[^30]: SnapBench：面向移动交互的“拍照即问”多模态检索基准测试

    SnapBench: Benchmarking Snap-and-Ask Multimodal Retrieval for Mobile Interactions

    [https://arxiv.org/abs/2608.29607](https://arxiv.org/abs/2608.29607)

    提出了首个针对移动“拍照即问”多模态检索的成对鲁棒性基准SnapBench，通过53种受控损坏条件的大规模评估发现图像损坏显著降低检索性能，且仅用干净图像的检索往往优于图文联合检索。

    

    移动AI如同一个视觉预言机，让用户能够拍摄某物的照片并询问相关信息。“拍照即问”检索如今已成为移动AI最常见的入口之一，然而照片常常是模糊的，文本问题也可能过于简短或存在输入错误。现有基准测试仅在干净输入上进行测试，或未能隔离“拍照即问”检索中的成对鲁棒性。因此，我们提出了SnapBench，这是首个针对鲁棒“拍照即问”多模态检索的成对基准，涵盖1,145个查询、9,085个图库条目，在53种受控损坏条件下构建并带有人工标注。我们评估了16个多模态检索器，涵盖双塔编码器和基于嵌入的视觉语言模型（VLM）。结果表明，图像损坏会显著降低检索性能，而文本损坏主要影响纯文本检索，对联合检索的影响有限。仅使用干净图像的检索往往优于联合检索，表明粗糙的文本会对检索造成拖累……

    arXiv:2608.29607v1 Announce Type: cross  Abstract: Mobile AI acts as a visual oracle, empowering users to snap a picture of something and ask for information. Snap-and-ask retrieval is now one of the most common entry points for mobile AI, yet photos are often blurry, while text questions may be short or mistyped. Existing benchmarks only test on clean inputs or do not isolate paired robustness in snap-and-ask retrieval. Therefore, we introduce SnapBench, the first paired benchmark for robust snap-and-ask multimodal retrieval, spanning 1,145 queries, 9,085 gallery items under 53 controlled corruption conditions with human annotations. We evaluate 16 multimodal retrievers, covering dual-tower encoders and embedding-based VLMs. Results show that image corruptions substantially degrade retrieval, while text corruptions mainly affect text-only retrieval and have limited impact on joint retrieval. Clean image-only retrieval often outperforms joint retrieval, indicating the coarse-text drag 
    
[^31]: RePair：将检索失败转化为反事实困难样本对

    RePair: Turning Retrieval Failures into Counterfactual Hard Pairs

    [https://arxiv.org/abs/2608.29604](https://arxiv.org/abs/2608.29604)

    RePair将检索中排名靠前的假阳性样本视为反事实支架，通过最小化修正其导致失败的局部残差，构建同模态困难正样本以及跨越决策边界的困难负样本对，产生互补的拉-推监督信号以提升视觉-语言检索性能。

    

    基于CLIP风格双编码器的视觉-语言检索取得了强大的跨模态性能，然而实际准确率往往取决于局部语义差异——即排名靠前的近似错误样本与真正匹配样本之间仅相差一个关键细节。困难样本挖掘可以筛选出易混淆的候选样本，但无法构建修正后的对应样本；合成数据增强可以生成新样本，但若不以模型实际失败为条件，则会针对无关的困难维度。我们观察到，排名靠前的假阳性样本是一个反事实支架——它与查询共享大部分语义，仅在导致失败的局部残差上有所不同。对这一残差进行最小化修正，可以在同一模态下生成真实匹配的困难正样本；修正后与未编辑的版本构成一个跨越决策边界的困难负样本对，从而产生互补的拉-推监督信号。我们提出了RePair，该方法由三个（原文在此处截断）……

    arXiv:2608.29604v1 Announce Type: new  Abstract: Vision-language retrieval with CLIP-style dual encoders achieves strong cross-modal performance, yet practical accuracy often hinges on localized semantic distinctions where top-ranked near misses differ from the true match by a single critical detail. Hard-sample mining can select confusable candidates but cannot construct corrected counterparts; synthetic augmentation can generate novel samples but, without conditioning on actual model failures, targets irrelevant dimensions of hardness. We observe that a top-ranked false positive is a counterfactual scaffold---sharing most of the query's semantics while differing in a localized failure-causing residual. Minimally correcting this residual yields a hard positive of the ground truth in the same modality; the corrected and unedited versions form a hard negative pair that straddles the decision boundary, producing complementary pull--push supervision. We introduce RePair, guided by three p
    
[^32]: 面向多样化用户行为的排序策略自适应双重稳健离线策略评估

    Adaptive Doubly Robust Off-Policy Evaluation for Ranking Policies under Diverse User Behavior

    [https://arxiv.org/abs/2608.29600](https://arxiv.org/abs/2608.29600)

    本文提出了一种面向排序策略的自适应双重稳健离线策略评估方法，通过自适应边缘化重要性权重在偏差与方差之间取得平衡，从而在多样化且未知的用户行为模型下实现可靠的策略评估。

    

    排序策略的离线策略评估（OPE）具有挑战性，因为从候选集中选择并排序多个项目会使可能的排序数量随候选数量和排序长度呈组合式增长。因此，逆倾向得分（IPS）方法——其重要性权重是评估策略与记录策略下完整排序概率之比——可能产生过大的方差。独立IPS（IIPS）和奖励交互IPS（RIPS）通过对用户浏览排序的方式施加固定假设来降低方差，但当这些假设与实际行为不匹配时可能引入偏差。自适应逆倾向得分（AIPS）通过自适应地对影响每个位置奖励的动作进行重要性权重的边缘化来应对这一权衡。当真实用户行为模型可被观测时，它在基于IPS的无偏估计器类别中达到最小方差。然而，其估计（摘要在此处截断）

    arXiv:2608.29600v1 Announce Type: new  Abstract: Off-policy evaluation (OPE) of ranking policies is challenging be- cause selecting and ordering multiple items from a candidate set makes the number of possible rankings grow combinatorially with the number of candidates and the ranking length. Consequently, Inverse Propensity Scoring (IPS), whose importance weight is the full-ranking probability ratio under the evaluation and logging policies, can have excessive variance. Independent IPS (IIPS) and Reward Interaction IPS (RIPS) reduce variance by imposing fixed assumptions on how users browse rankings, but may introduce bias when those assumptions mismatch actual behavior. Adaptive Inverse Propensity Scoring (AIPS) addresses this trade-off by adap- tively marginalizing importance weights over the actions that affect each position-wise reward. It attains minimum variance within a class of unbiased IPS-based estimators when the true user be- havior model is observed. However, its estimati
    
[^33]: 基于选择模型的物品图的边谱：强边与弱边在协同过滤中编码不同的关系

    The Edge Spectrum of Choice-Derived Item Graphs: Strong and Weak Edges Encode Different Relations in Collaborative Filtering

    [https://arxiv.org/abs/2608.29578](https://arxiv.org/abs/2608.29578)

    该论文发现由选择模型导出的物品图中强边与弱边编码了性质不同的关系——强边连接被点击物品的同一列表内竞争者（正是排序梯度要推开的物品对），并将此形式化为平滑算子与排序梯度之间的符号不匹配，从而解释了为何直接用选择模型导出的图算子替换共同点击图无法提升协同过滤性能。

    

    图协同过滤依赖于物品-物品图，其边被用于正向平滑，这背后隐含着一个假设：更强的边只是编码了与更弱的边相同关系的更多内容。我们证明该假设对于一类实践中重要的图不成立——即边权重来源于选择模型的图。在这类图上，强边与弱边编码了性质上截然不同的关系，我们称之为“边谱”。具体而言，强边集中于被点击物品在同一候选列表内的竞争者，恰好是列表内排序梯度所推开的那类物品对，而弱边则不然。我们将这一现象形式化为平滑算子与排序梯度之间的符号不匹配，并证明共同点击图在构造上不可能出现同样的错位。这一诊断解释了在MIND和EB-NeRD数据集上的三个实证观察：(i) 直接替换的选择模型导出算子并不能超越共同点击，尽管索引……（摘要在此处截断）

    arXiv:2608.29578v1 Announce Type: new  Abstract: Graph collaborative filtering relies on item--item graphs whose edges are used for positive smoothing, under the implicit assumption that stronger edges encode more of the same relation as weaker ones. We show that this assumption fails for a practically important class of graphs: those whose edge weights come from a choice model. On such graphs, strong and weak edges encode qualitatively different relations, which we call an edge spectrum. Specifically, strong edges concentrate on the in-slate competitors of clicked items, exactly the pairs that the within-slate ranking gradient pushes apart, while weak edges do not. We formalize this as a sign mismatch between the smoothing operator and the ranking gradient, and prove that co-click graphs cannot exhibit the same misalignment by construction. This diagnosis explains three empirical observations on MIND and EB-NeRD: (i) drop-in choice-derived operators do not beat co-click, despite index
    
[^34]: 你在听什么？面向音频到文本大语言模型的时序音乐定位

    What Are You Listening to? Temporal Music Grounding for Audio-to-Text Large Language Models

    [https://arxiv.org/abs/2608.29480](https://arxiv.org/abs/2608.29480)

    本文提出了时序音乐定位新任务及具有精确符号-音频对齐的MusicGroundingBench基准，用于评估音频-语言模型能否将音乐查询定位到具体时间段，发现现有模型在此任务上仍面临挑战，而任务特定训练可带来显著提升。

    

    大型音频-语言模型能够生成流畅且在音乐上看似合理的回答，然而这些回答是否真正基于音频输入往往仍不清楚。我们提出了时序音乐定位这一任务，在该任务中，模型需要返回与所查询的音符、音乐事件或音乐模式相对应的一个或多个时间段。为了评估这一能力，我们构建了MusicGroundingBench，这是一个受控基准套件，通过将算法生成的钢琴MIDI渲染为音频，从而获得精确的符号到音频的对齐。该套件包含两个子集：MGBench-3N，用于评估在最多包含三个音符的片段中的音符级定位能力；MGBench-2B，用于评估两小节片段中的结构化定位能力和短形式音乐理解能力。实验表明，时序音乐定位对当前的音频-语言模型而言仍然具有挑战性，而针对特定任务的训练则带来了显著的性能提升。我们还进一步报告了探索性证据。

    arXiv:2608.29480v1 Announce Type: cross  Abstract: Large audio-language models can produce fluent and musically plausible responses, yet it often remains unclear whether those responses are grounded in the audio input. We introduce temporal music grounding, a task in which a model returns one or more time spans corresponding to a queried musical note, event, or pattern. To evaluate this capability, we present MusicGroundingBench, a controlled benchmark suite built by rendering algorithmically generated piano MIDI to audio, yielding exact symbolic-to-audio alignment. The suite comprises two subsets: MGBench-3N, which evaluates note-level grounding in clips containing up to three notes, and MGBench-2B, which evaluates structured grounding and short-form music understanding in two-bar excerpts. Experiments show that temporal music grounding remains challenging for current audio-language models, whereas task-specific training yields substantial gains. We further report exploratory evidence
    
[^35]: 超越信息流的内容探索：创作者供给与共享内容库

    Content Exploration Beyond the Feed: Creator Supply and the Shared Corpus

    [https://arxiv.org/abs/2608.29430](https://arxiv.org/abs/2608.29430)

    该论文通过某大型短视频平台的四项实验首次揭示了内容探索的双重价值——生产侧探索可使创作者发帖量提升8.55%，观众侧探索虽增加观看次数但减少观看时长，且探索引发的创作者供给与自然采纳会补充共享内容库，突破传统仅衡量观众侧效果的评估局限。

    

    工业级推荐系统通过有预算的探索为新内容提供初始曝光，然后依据早期表现决定后续分发。在许多短视频平台上，探索是新视频触达观众的主要途径。观众侧的测试衡量内容消费，而我们综述的已发表预算目标均忽略了创作者的反应。我们分析了某大型短视频平台上的四项实验。一项为期八个月的创作者侧消融实验发现，相对于最低基线，生产侧探索使每位创作者发布的视频数量提升8.55%，至少发布一次视频的创作者数量提升7.10%。一项预算匹配的重新分配实验提高了创作者参与度，且短期内未检测到观众侧的明显变化。一项为期一年的观众侧消融实验发现，视频观看次数增加1.74%，但观看时长减少2.13%。一次投放的观看既能创造即时的信息流价值，也可能引发有机的自然采纳，还能激励创作者供给。自然采纳与创作者供给会持续补充共享内容库，由此产生两个测量上的局限。

    arXiv:2608.29430v1 Announce Type: cross  Abstract: Industrial recommenders give new content initial views through budgeted exploration, then use early performance to decide further delivery. On many short-video platforms, exploration is the primary way new videos reach viewers. Viewer-side tests measure consumption; the published budget objectives we review omit creator response. We analyze four experiments on a major short-video platform. An eight-month creator ablation finds production exploration raises videos posted per creator by 8.55% and creators posting at least once by 7.10% relative to a minimal floor. A budget-matched reallocation raises creator participation with no detectable short-run viewer-side change. A year-long viewer ablation finds 1.74% more video views but 2.13% less view time. A delivered view creates immediate feed value, can trigger organic take-up, and can induce creator supply. Take-up and supply replenish a shared corpus, creating two measurement limits. Vie
    
[^36]: 智能体作为多模态推荐中的知识整合者与利用者

    Agents as Knowledge Integrator and Utilizer in Multimodal Recommendation

    [https://arxiv.org/abs/2608.29410](https://arxiv.org/abs/2608.29410)

    提出AgentMMRec框架，通过整合者与利用者两个协同智能体，将多模态内容与用户行为联合解释为可复用的知识记忆，并利用该知识优化模态物品图与推荐排序，从而弥合多模态信号与推荐目标之间的语义鸿沟。

    

    在线平台日益依赖多模态推荐系统来对商品、媒体及其他网络内容进行排序。现有方法通常将视觉和文本特征注入物品表示中，或基于模态级相似度构建同构图，但由此产生的信号可能仍与推荐目标不一致。我们从知识整合的视角研究这一语义鸿沟：多模态内容应当与用户行为结合解释之后，再用于构建推荐图或调整排序。我们提出AgentMMRec，一个基于智能体的多模态推荐框架，包含两个协同的角色。整合者智能体从训练交互和物品内容中推断出兼顾行为与多模态信息的用户偏好和物品属性，并将其存储在可复用的知识记忆中。利用者智能体则消费这一记忆，用于优化模态特定的物品-物品图、构建行为……（原文摘要在此处被截断）

    arXiv:2608.29410v1 Announce Type: new  Abstract: Online platforms increasingly rely on multimodal recommender systems to rank products, media, and other Web content. Existing methods usually inject visual and textual features into item representations or build homogeneous graphs from modality-level similarity, but the resulting signals can remain misaligned with the recommendation objective. We study this semantic gap from a knowledge-integration perspective: multimodal content should be interpreted together with user behavior before it is used to construct recommendation graphs or adjust rankings.   We propose AgentMMRec, an agent-based multimodal recommendation framework with two coordinated roles. The Integrator Agent infers behavior- and multimodal-aware user preferences and item properties from training interactions and item content, then stores them in a reusable knowledge memory. The Utilizer Agent consumes this memory to refine modality-specific item-item graphs, construct beha
    
[^37]: 基于强化学习的健身房训练个性化推荐系统

    Personalized Recommender Systems for Gym Workouts: A Reinforcement Learning Approach

    [https://arxiv.org/abs/2608.29409](https://arxiv.org/abs/2608.29409)

    本文提出了一个基于强化学习的健身房训练推荐框架，将推荐范围从单纯的动作选择扩展到包含动作、组数、重复次数和负荷的完整训练处方，并能利用用户跳过动作的行为实现在线个性化。

    

    训练推荐系统旨在帮助健身房用户完成有效且引人入胜的训练课程。然而，仅仅推荐训练动作是不够的，一个实用的系统还必须确定合适的组数、重复次数和训练负荷，同时适应用户跳过训练动作等行为。现有方法通常只考虑这些因素中的一部分，限制了它们在现实场景中的适用性。在本文中，我们将训练推荐从动作选择扩展到完整的训练处方。我们提出了一个基于强化学习（RL）的框架，包含四种环境：仅动作推荐和完整处方设置，每种设置又分别包含和不包含基于跳过行为的交互。完整处方环境推荐训练动作、组数、重复次数和负荷，而启用跳过功能的环境则利用用户的跳过行为进行在线个性化。在合成用户上的实验表明，对完整处方的建模（原文在此处截断）

    arXiv:2608.29409v1 Announce Type: new  Abstract: Workout recommender systems aim to help gym users complete effective and engaging training sessions. However, recommending exercises alone is insufficient, as a practical system must also determine appropriate sets, repetitions, and training loads, while adapting to user behavior such as skipping exercises. Existing approaches typically consider only a subset of these factors, limiting their applicability in real-world settings. In this paper, we extend workout recommendation from exercise selection to full workout prescription. We propose a reinforcement learning (RL)-based framework with four environments: exercise-only and full-prescription settings, each with and without skip-based interaction. The full-prescription environments recommend exercises, sets, repetitions, and load, while the skip-enabled environments use user skipping behavior for online personalization. Experiments with synthetic users show that modeling the full prescr
    
[^38]: FISICA：一个已部署的足底压力与体态评估服务，具备基于本体论的推荐功能

    FISICA: A Deployed Service for Plantar-Pressure and Posture Assessment with Ontology-Grounded Recommendation

    [https://arxiv.org/abs/2608.29336](https://arxiv.org/abs/2608.29336)

    FISICA是一个已部署的足底压力与体态评估服务，其核心创新在于用与被测者相同的方法测量3D虚拟形象并求解直至两者一致（基于采样不变的脊柱指标），取代传统的角度增益映射方法，将正常与驼背记录的区分度从0.9度提升到7.2度。

    

    FISICA是一个正在生产环境中运行的身体评估与推荐服务。仅需一次站立测量配合两张照片，即可返回足部负荷测量数据、体态坐标、一个由数据驱动的3D虚拟形象、可视化报告以及经过排序的鞋类和运动推荐候选。测量来自一个特制的测量装置，该装置在1厘米网格上布置了634个力敏元件以及四个称重传感器；每一条推荐均由基于规则的评估器控制，而语言模型仅用于解释已存储的结果。本论文的方法学贡献在于3D虚拟形象：不同于将测量角度通过调谐增益映射到骨架上的传统做法，我们使用与被测者相同的函数来测量虚拟形象，并求解直到两者一致，采用一个采样不变的脊柱指标，该指标将正常记录与驼背记录区分开达7.2度，而单关节公式仅为0.9度。在生产环境中，通用API的中位响应时间为0.023秒，足底压力分析为0.45秒，推荐……

    arXiv:2608.29336v1 Announce Type: cross  Abstract: FISICA is a body-assessment and recommendation service running in production. One standing session with two photographs returns foot-loading measures, posture coordinates, a driven 3D avatar, a visual report, and ranked shoe and exercise candidates. Measurement comes from a purpose-built scale carrying 634 force-sensitive elements on a 1 cm grid and four load cells, and a rule-based evaluator controls every recommendation while a language model only explains the stored result. The method contribution is the avatar. Instead of mapping a measured angle onto a rig through a tuned gain, we measure the avatar with the same function used on the subject and solve until the two agree, on a sampling-invariant spinal metric that separated a normal from a kyphotic record by 7.2 degrees against 0.9 degrees for a single-joint formulation. In production, general APIs respond at a 0.023 s median, plantar-pressure analysis at 0.45 s, and recommendatio
    
[^39]: 面向REST API误用自动修复的数据库增强RAG方法

    Database-Augmented RAG for Automated Repair of REST API Misuses

    [https://arxiv.org/abs/2608.29290](https://arxiv.org/abs/2608.29290)

    本研究通过构建11种具有不同数据库结构的RAG配置并与基线方法对比修复率，评估了API规范在RAG数据库中的组织方式对REST API误用自动修复效果的影响。

    

    许多物联网（IoT）服务提供表述性状态转移（REST）API，这要求客户端开发者实现符合相应API规范的应用程序。当客户端程序包含API误用时，开发者需要基于错误响应进行调试。然而，这类错误响应通常不足以识别根本原因，需要开发者反复与服务器进行通信。检索增强生成（RAG）是一种为大型语言模型（LLM）提供外部知识的有前景的方法。然而，在REST API误用的自动修复中，API规范应如何存储在RAG数据库中仍不明确。本研究评估了组织API规范的不同配置如何影响基于RAG的REST API误用修复效果。我们构建了11种具有不同数据库结构的RAG配置，并将其修复率与基线方法进行比较。（注：原文摘要在此处被截断）

    arXiv:2608.29290v1 Announce Type: new  Abstract: Many Internet of Things (IoT) services provide Representational State Transfer (REST) APIs, which require client developers to implement applications that conform to the corresponding API specifications. When client programs contain API misuse, developers debug them based on error responses. However, such responses are often insufficient for identifying the root cause, requiring developers to repeatedly communicate with the server. Retrieval-Augmented Generation (RAG) is a promising approach for providing large language models (LLMs) with external knowledge. However, in automated repair of REST API misuses, it remains unclear how specifications should be stored in a RAG database. This study evaluates how different configurations for organizing API specifications affect RAG-based repair of REST API misuse. We constructed 11 RAG configurations with different database structures and compared their repair rates with a baseline method. For ev
    
[^40]: 通过针对性检索器微调实现乌兹别克语法律RAG的云端与本地部署

    Cloud and On-Premises Deployment of Uzbek Legal RAG via Targeted Retriever Fine-Tuning

    [https://arxiv.org/abs/2608.29284](https://arxiv.org/abs/2608.29284)

    本文针对乌兹别克语法律问答这一低资源场景，构建了专家标注的检索基准与端到端评测基准，并通过针对性检索器微调，在云端成本约束和本地硬件与延迟约束两种部署模式下实现了高质量的法律RAG助手。

    

    将大语言模型部署用于法律问答会带来通用排行榜无法捕捉的挑战，尤其是在低资源语言和严格运营约束下。我们报告了构建并运营一个乌兹别克语检索增强（RAG）法律助手的经验，该系统必须在两种模式下运行：一种是在每token成本上限内最大化答案质量的托管云服务；另一种是面向法律数据不允许离开其基础设施的客户的本地部署，这限制我们只能在有限的本地硬件上、在延迟约束下使用开放权重的模型。由于该场景此前不存在评估基准，我们构建了两个领域基准：一个是包含178条专家标注法律查询（附带黄金法条片段）的检索基准；另一个是包含504条专家整理问答对的端到端基准，由LLM评判者评分，且我们将该评判者的评分与人类判断以及独立模型家族的评判者进行了验证。

    arXiv:2608.29284v1 Announce Type: new  Abstract: Deploying large language models for legal question answering raises challenges that general-purpose leaderboards do not capture, particularly for low-resource languages and under hard operational constraints. We report on building and operating a retrieval-augmented (RAG) legal assistant for Uzbek that must run in two regimes: a managed cloud service that maximizes answer quality within a per-token cost ceiling, and an on-premises deployment for clients whose legal data may not leave their infrastructure, restricting us to open-weight models on limited local hardware under latency constraints. Because no evaluation existed for this setting, we build two domain benchmarks: a retrieval benchmark of 178 expert-annotated legal queries with gold provision spans, and an end-to-end benchmark of 504 expert-curated question--answer pairs scored by an LLM judge whose ratings we validate against human judgments and against an independent-family jud
    
[^41]: FKG.in的验证：LLM增强的印度食品知识中的健全性评估

    Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge

    [https://arxiv.org/abs/2608.29249](https://arxiv.org/abs/2608.29249)

    本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。

    

    在线烹饪生态系统中，由大型语言模型（LLM）生成、修改或总结的食谱内容日益增多。虽然这些输出通常看似合理，但可能包含虚构的食材、被误述的用量或文化上不合常理的食材组合，从而限制了其在下游应用和知识图谱构建中的适用性。在本文中，我们提出了一种半自动化的健全性评估工作流程，用于验证由LLM从非正式烹饪来源中提取和增强的结构化食谱数据。该流程作为印度食品知识图谱FKG.in的一部分开发而成，通过结合形式文法、基于词汇的检查、统计启发式方法、基于Set Transformer的连贯性建模以及基于检索的验证等多阶段流程，识别并解决常见的失败模式，包括结构性不一致、语义和逻辑上的不连贯以及与源文本的偏差。

    arXiv:2608.29249v1 Announce Type: new  Abstract: The online culinary ecosystem is increasingly populated by recipe content generated, modified, or summarized by Large Language Models (LLMs). While often plausible, such outputs may contain hallucinated ingredients, misrepresented quantities, or culturally implausible combinations, limiting their suitability for downstream applications and knowledge graph construction. In this paper, we present a semi-automated soundness assessment workflow for validating structured recipe data extracted and augmented by LLMs from informal culinary sources. Developed as part of FKG.in, a knowledge graph of Indian food, the pipeline identifies and addresses common failure modes, including structural inconsistencies, semantic and logical incoherence, and deviations from the source text, through a multi-stage process combining formal grammars, vocabulary-based checks, statistical heuristics, Set Transformer-based coherence modeling, and retrieval-based veri
    
[^42]: TAAL：通过时序自回归对齐缓解生成式推荐中的早期束剪枝问题

    TAAL: Mitigating Early Beam Pruning in Generative Recommendation via Temporal Autoregressive Alignment

    [https://arxiv.org/abs/2608.29179](https://arxiv.org/abs/2608.29179)

    该论文提出TAAL方法，在训练阶段构建联合软目标并用前向KL损失对齐早期前缀分布、在推理阶段用逐点互信息校准候选分数，从而缓解生成式推荐中束搜索早期步骤的不可逆剪枝问题，在三个基准上显著提升检索性能。

    

    生成式推荐将物品编码为层次化语义标识符（SIDs），并通过自回归解码检索下一个物品。然而，标准的下一个词元预测并未显式覆盖交互序列中存在的多模态转移，导致真实的SID容易在束搜索的早期分支处被不可逆地剪枝。在三个公开基准数据集上，我们发现91.9%–96.6%的检索失败发生在前两个解码步骤内。因此，我们提出了时序自回归对齐（TAAL）。在训练阶段，TAAL从历史转移中构建联合$(c_1,c_2)$软目标，并通过前向KL目标对齐早期前缀分布。在推理阶段，它利用逐点互信息（PMI）校准候选分数，以降低全局高频前缀的影响。在Amazon Beauty、Instruments和Yelp数据集上，TAAL相比标准基线提升了NDCG@10……

    arXiv:2608.29179v1 Announce Type: cross  Abstract: Generative recommendation encodes items as hierarchical semantic identifiers (SIDs) and retrieves the next item through autoregressive decoding. Standard next-token prediction, however, does not explicitly cover the multimodal transitions present in interaction sequences, leaving the ground-truth SID vulnerable to irreversible pruning at early beam-search branches. Across three public benchmarks, we find that 91.9\%--96.6\% of retrieval failures occur within the first two decoding steps. We therefore propose Temporal Autoregressive Alignment (TAAL). During training, TAAL constructs a joint $(c_1,c_2)$ soft target from historical transitions and aligns the early-prefix distribution with a forward KL objective. During inference, it calibrates candidate scores with pointwise mutual information (PMI) to reduce the influence of globally frequent prefixes. On Amazon Beauty, Instruments, and Yelp, TAAL improves NDCG@10 over the standard basel
    
[^43]: 面向检索与图卷积网络分类的上下文感知可解释表示

    Context-Aware Interpretable Representations for Retrieval and Graph Convolutional Network Classification

    [https://arxiv.org/abs/2608.29004](https://arxiv.org/abs/2608.29004)

    该论文提出了一种将流形学习策略与基于排序的可解释图嵌入相结合的新型无监督框架，在保持低维度和下游任务高效能的同时为视觉表示提供可解释性，从而弥合了几何鸿沟与可解释性鸿沟。

    

    arXiv:2608.29004v1 公告类型：新论文 摘要：过去几十年间，视觉信息建模与表示取得了显著进展，这主要得益于卷积神经网络、基于Transformer的模型以及基础模型的发展。尽管取得了这些进步，但关于相似性评估本质和模型透明度的关键挑战却一直被忽视。一个主要问题是“几何鸿沟”，即传统的成对度量方法无法捕捉数据集流形的内在几何结构。此外，“可解释性鸿沟”依然存在，因为表示往往缺乏与人类认知的对齐。因此，如何在保持低维度和下游任务高效能的同时为表示提供可解释性，仍然是一个悬而未决的挑战。在本文中，我们提出了一种新颖的无监督框架，将流形学习策略与基于排序的可解释图嵌入相结合，我们的方法通过首先……有效地弥合了这些鸿沟。

    arXiv:2608.29004v1 Announce Type: new  Abstract: The advances in visual information modeling and representation during the last decades are remarkable, mainly supported by Convolutional Neural Networks, Transformer-based, and Foundation Models. Despite this progress, critical challenges regarding the nature of similarity assessment and model transparency have been neglected. A primary concern is the Geometric Gap, where traditional pairwise measures fail to capture the intrinsic geometry of the dataset manifold. Furthermore, the Interpretability Gap persists, as representations often lack alignment with human cognition. Therefore, how to provide interpretability to representations while maintaining low dimensionality and high effectiveness in downstream tasks remains an open challenge. In this paper, we propose a novel unsupervised framework that integrates Manifold Learning strategies with Rank-based Interpretable Graph Embeddings. Our approach effectively bridges these gaps by first 
    
[^44]: 面向文本与多媒体数据的有效图与基于排名的上下文嵌入

    Effective Graph and Rank-based Contextual Embeddings for Textual and Multimedia Data

    [https://arxiv.org/abs/2608.29001](https://arxiv.org/abs/2608.29001)

    本文提出RaDE方法，利用基于排名的信息和代表性节点子集选择来实现图嵌入的维度可解释性，在降低计算成本的同时改进文本与多媒体数据的检索任务。

    

    在数据驱动的世界中，高效地组织和映射对象之间的关系至关重要。图是建模这些连接关系的强大工具，被广泛应用于社交网络、电信和生物学等领域。然而，基于图的方法通常面临较高的计算成本，尤其是在内存和空间使用方面。为解决这一问题，图嵌入技术（也称为网络表示学习）将图信息编码到低维表示中，同时保留结构特性。然而，传统方法缺乏可解释的维度。RaDE（Rank Diffusion Embedding，排名扩散嵌入）引入了一种使用基于排名信息的新方法，其关键步骤是选择一个具有代表性的节点子集，从而为其维度提供可解释性并改进检索任务。尽管潜力巨大，RaDE的原始提案并未充分探索代表性子集选择的有效性。

    arXiv:2608.29001v1 Announce Type: new  Abstract: In a data-driven world, efficiently organizing and mapping relationships between objects is crucial. Graphs are powerful tools for modeling these connections, being widely used in social networks, telecommunications, and biology. However, graph-based methods often face high computational costs, particularly in memory and space usage. To address this, graph embedding techniques, also referred to as Network Representation Learning, encode graph information into lower-dimensional representations while preserving structural aspects. Traditional methods, however, lack interpretable dimensions. RaDE (Rank Diffusion Embedding) introduces a new approach using rank-based information, with a key step being the selection of a representative subset of nodes to provide interpretability for its dimensions and improve retrieval tasks. Despite its potential, RaDE's original proposal did not fully explore the effectiveness of representative subset select
    
[^45]: MERIT：在生成式极端多标签分类中缓解曝光偏差以用于用户兴趣倾向建模

    MERIT: Mitigating Exposure Bias in Generative XMC for User-Interest Propensity Modeling

    [https://arxiv.org/abs/2608.28931](https://arxiv.org/abs/2608.28931)

    MERIT框架通过基于黄金标签与难负样本混合排列的置换不变多目标损失这一自校正目标，缓解生成式极端多标签分类中的曝光偏差，从而提升电商场景下用户兴趣倾向建模的准确性。

    

    在大规模上将用户匹配到兴趣类别是个性化购物的核心，但这项任务在大型电商平台中极具挑战性，因为标签空间不断演变，且用户兴趣信号稀疏且呈长尾分布。自回归语言模型之所以有吸引力，是因为其世界知识以及对描述符的语义先验能够泛化到极端标签空间，并可以容纳多个有效的标签分配。然而，在教师强制的微调方式下，推理时的预测会成为条件上下文的一部分：早期错误会将后续输出引向共现的标签，导致过度生成近似相关的标签，同时遗漏不相关的真实兴趣。我们提出了MERIT，一个用户兴趣倾向建模框架，通过自校正目标来缓解这种曝光偏差。一种在黄金标签与挖掘出的难负样本标签的随机混合排列上定义的置换不变多目标损失，使生成器暴露于错误……

    arXiv:2608.28931v1 Announce Type: cross  Abstract: Matching users to interest categories at scale is central to personalized shopping, but the task is challenging in large e-commerce platforms, where label spaces continually evolve and user-interest signals are sparse and long-tailed. Autoregressive language models are appealing because their world knowledge and semantic priors over descriptors generalize across extreme label spaces and accommodate multiple valid label assignments. Yet under teacher-forced fine-tuning, inference-time predictions become part of the conditioning context: early errors steer later outputs toward co-occurring labels, over-generating near-correlates and missing unrelated true interests. We present MERIT, a framework for user-interest propensity modeling that mitigates this exposure bias through a self-correction objective. A permutation-invariant multi-target loss over shuffled mixtures of gold and mined hard-negative labels exposes the generator to erroneou
    
[^46]: ASTRA——用于工单解决与分析的智能体系统

    ASTRA - Agentic System for Ticket Resolution and Analysis

    [https://arxiv.org/abs/2608.28790](https://arxiv.org/abs/2608.28790)

    ASTRA提出了一种由中央协调器调度三个专门信息收集智能体（历史案例检索、日志分析、领域知识检索）并通过裁判-协调器优化循环生成可验证、有证据支撑的故障排除报告的智能体系统，解决了现有工单自动化缺乏证据建模与来源追溯的问题。

    

    技术运营团队需要通过综合来自工单文本、历史案例、系统日志和技术文档的碎片化证据来解决大量的事件。现有的自动化方案通常依赖于缺乏明确证据建模或来源追溯的整体式生成，导致当关键信号分散在不同来源中时，其输出难以验证。我们提出了ASTRA，一个用于工单解决的智能体系统，其中中央协调器负责协调三个专门的信息收集智能体，并驱动“裁判-协调器”优化循环，以生成有证据支撑的故障排除报告。TicketSimilarityAgent通过密集检索和LLM重排序来检索相关的历史先例；LogAgent利用确定性过滤和受约束的LLM分析，将数十万行日志提炼为结构化的、基于引用的发现；DomainKnowledgeAgent通过M（摘要在此处被截断）

    arXiv:2608.28790v1 Announce Type: cross  Abstract: Technical operations teams resolve large volumes of incidents by synthesizing fragmented evidence from ticket text, historical cases, system logs, and technical documentation. Existing automation often relies on monolithic generation without explicit evidence modeling or provenance, making outputs difficult to verify when critical signals are sparse across sources. We propose ASTRA, an agentic system for ticket resolution in which a central orchestrator coordinates three specialist information-gathering agents and drives a judge-orchestrator refinement loop to produce evidence-backed troubleshooting reports. TicketSimilarityAgent retrieves relevant historical precedents through dense retrieval and LLM reranking; LogAgent distills hundreds of thousands of log lines into structured, quote-grounded findings using deterministic filtering and constrained LLM analysis; and DomainKnowledgeAgent retrieves relevant technical knowledge via the M
    
[^47]: 编织视觉叙事：超越原子化视觉匹配的智能体图像束组合

    Weaving Visual Narratives: Agentic Image Bundle Composition Beyond Atomic Visual Matching

    [https://arxiv.org/abs/2608.28695](https://arxiv.org/abs/2608.28695)

    提出了图像束组合（IBC）这一新检索范式及首个基准数据集 IBCBench，将图像检索从对孤立图像的逐点匹配升级为从海量照片池中动态组合具有结构关联的连贯图像束。

    

    图像检索传统上被表述为逐点匹配问题，即每张候选图像被独立评分。然而，这种原子化范式无法捕捉个人照片收藏中人类搜索意图的复杂性——用户往往寻求的是由结构关系紧密联系的紧凑视觉故事，而非孤立的快照。为解决这一局限，我们提出了图像束组合，这是一种新范式，将目标从对单张图像排序转变为从海量非结构化照片池中动态组合出连贯的图像束。由于目标图像束并非预先定义，IBC 带来了严重的组合爆炸挑战，并要求建模不可分解的联合相关性。为建立这一范式，我们构建了 IBCBench——首个 IBC 基准数据集，包含 109,467 张图像和 667 条经核验的查询，通过半自动化核验流水线构建。此外……（注：原文摘要在此处截断）

    arXiv:2608.28695v1 Announce Type: cross  Abstract: Image retrieval has traditionally been formulated as a point-wise matching problem, where each candidate image is scored in isolation. However, this atomic paradigm fails to capture the complexity of human search intent within personal photo collections, where users often seek compact visual stories bound by structural relations rather than isolated snapshots. To address this limitation, we introduce **Image Bundle Composition (IBC)**, a novel paradigm that shifts the objective from ranking individual images to dynamically composing cohesive image bundles from a massive, unstructured photo pool. Since target bundles are not predefined, IBC presents a severe combinatorial explosion challenge and demands modeling non-decomposable joint relevance. To establish this paradigm, we construct **IBCBench**, the first IBC benchmark dataset containing 109,467 images and 667 verified queries, built via a semi-automated verification pipeline. Furth
    
[^48]: 大语言模型能否识别转化归因中的有意义触点？

    Can Large Language Models Identify Meaningful Touchpoints in Conversion Attribution?

    [https://arxiv.org/abs/2608.28649](https://arxiv.org/abs/2608.28649)

    该论文通过人工标注揭示了现有基于协同过滤的转化归因触点选择方法与用户语义意图之间的显著语义鸿沟，并系统评估了大语言模型识别隐式相关触点的能力以及不同提示策略和基础模型对性能的影响。

    

    转化归因中的触点选择，即识别对转化有贡献的有意义触点，对电商推荐和在线广告至关重要。当前的触点选择方法严重依赖基于协同过滤的启发式规则，无法与用户感知的语义意图保持一致。通过人工标注，我们揭示了一个显著的语义鸿沟：许多隐式相关但语义上相关的触点仍未被现有规则检测到。因此，我们系统地评估了大语言模型（LLMs）识别这些隐藏关联的能力。评估结果表明，尽管大语言模型能够有效发现相当一部分隐式相关的触点，但其选择性能仍有很大的提升空间。此外，我们分析了不同提示策略和基础模型选择对识别性能的影响，提供了有价值的...

    arXiv:2608.28649v1 Announce Type: cross  Abstract: Touchpoint selection in conversion attribution, namely identifying meaningful touchpoints contributing to conversions, is essential for e-commerce recommendation and online advertising. Current selection methods rely heavily on collaborative-filtering-based heuristics, which fail to align with user-perceived semantic intent. Through human annotation, we reveal a significant semantic gap: many implicitly-related, semantically relevant touchpoints remain undetected by existing rules. Therefore, we systematically evaluate the capability of Large Language Models (LLMs) in identifying these hidden associations. Our evaluation shows that while LLMs effectively uncover a substantial portion of implicitly-related touchpoints, significant room for improvement remains in their selection performance. Furthermore, we analyze the impact of different prompting strategies and foundation model choices on identification performance, providing valuable 
    
[^49]: 基于自然语言处理的古印度医学文献译本知识提取与主题分类

    NLP-Driven Knowledge Extraction and Thematic Classification of Translated Ancient Indian Medical Texts

    [https://arxiv.org/abs/2608.28608](https://arxiv.org/abs/2608.28608)

    本研究利用命名实体识别、BERTopic主题建模和Neo4j知识图谱等NLP技术，对《妙闻集》等古印度医学文献译本进行知识提取与主题分类，实现医学概念的语义化表示、知识检索与数字化保存。

    

    《妙闻集》等古印度医学文献包含大量关于疾病、治疗方法和外科技术的信息。然而，其古老的书写形式和复杂晦涩的词汇给文献的可访问性和系统化整理带来了困难。本研究利用命名实体识别（NER）、BERTopic主题建模以及在Neo4j中构建知识图谱等自然语言处理（NLP）方法，基于译本对重要概念进行提取、分类和可视化。基于BERTopic的主题分类能够识别其中蕴含的医学主题，而NER则支持对疾病、治疗方法、研究者和药用植物等实体的结构化识别。基于Neo4j的图网络分析还能对提取实体之间的关系进行语义化表示，从而支持知识检索和数字化保存。研究结果展示了图数据库、主题……

    arXiv:2608.28608v1 Announce Type: cross  Abstract: Ancient Indian medical texts like Sushruta Samhita have extensive information on diseases, treatments, and surgical techniques. Yet, their ancient format and use of intricate vocabulary pose difficulties in accessibility and systematic ordering. The research here utilizes Natural Language Processing (NLP) methods like Named Entity Recognition (NER), BERTopic modeling, and Knowledge Graph development in Neo4j to extract, categorize, and visualize important concepts based on translated versions. Thematic classification with BERTopic allows for the identification of the underlying medical topics, whereas NER supports the structured entity recognition of diseases, treatments, researchers, and medicinal plants. Graphbased network analysis with Neo4j also allows for the semantic representation of relationship among extracted entities, supporting knowledge retrieval and digital preservation. The findings illustrate how graph databases, topic 
    
[^50]: HubMixer：面向推荐系统中参数高效特征交互的渐进式潜在枢纽混合

    HubMixer: Progressive Latent Hub Mixing for Parameter-Efficient Feature Interaction in Recommendation

    [https://arxiv.org/abs/2608.27991](https://arxiv.org/abs/2608.27991)

    提出 HubMixer 架构，通过渐进式的潜在枢纽混合机制，避免在异构 token 空间中直接混合的低效问题，实现推荐系统中参数高效的特征交互。

    

    学习有效的特征交互是工业推荐和广告排序系统的核心。近期的 token 混合架构通过轻量级混合算子简化了自注意力机制，提升了硬件效率并支持大规模部署。然而，推荐系统中的 token 本质上是异构的：用户画像、物品属性、行为序列、上下文特征、统计信号以及业务侧特征处于不同的语义空间中，并以稀疏的、样本特定的模式进行交互。因此，直接在原始异构 token 空间中混合所有 token 可能导致参数效率低下，因为模型必须隐式地发现哪些特征组应该交互以及这些交互应如何路由。在本文中，我们提出了 HubMixer，一种面向推荐特征交互的参数高效的潜在枢纽混合架构。HubMixer 不直接混合原始特征 token，而是……

    arXiv:2608.27991v1 Announce Type: new  Abstract: Learning effective feature interactions is central to industrial recommendation and advertising ranking systems. Recent token-mixing architectures simplify self-attention with lightweight mixing operators, improving hardware efficiency and enabling large-scale deployment. However, recommendation tokens are fundamentally heterogeneous: user profiles, item attributes, behavioral sequences, context features, statistical signals, and business-side features live in different semantic spaces and interact in sparse, sample-specific patterns. Directly mixing all tokens in the raw heterogeneous token space may therefore be parameter-inefficient, as the model must implicitly discover which feature groups should interact and how such interactions should be routed. In the paper, we propose HubMixer, a parameter-efficient latent hub mixing architecture for feature interaction in recommendation. Instead of directly mixing raw feature tokens, HubMixer 
    
[^51]: RATIO：科学文献中跨类型构思操作检索的基准

    RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature

    [https://arxiv.org/abs/2608.27394](https://arxiv.org/abs/2608.27394)

    RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。

    

    arXiv:2608.27394v1 公告类型：新 摘要：检索到的科学文献可以为人与AI科学家提供灵感。灵感可以采取不同形式：先前的工作可能直接建议如何解决问题，或在不同抽象层次上指出方向——放大到更一般的视角或缩小到具体实现。我们引入RATIO（跨类型构思操作检索），这是一个大规模基准，其中相关性由三种操作定义，我们称之为构思动作：Address检索针对所提出问题的潜在方法，Broaden检索更一般的表述，Specify检索具体实例。RATIO是通过一种通用方法从CS文献中数百万篇全文科学论文构建而成，该方法将话语标记远距离监督——先前仅用于分类——扩展到语料库级检索，并结合了广泛的LLM和人工审核。实验表明，操作-

    arXiv:2608.27394v1 Announce Type: new  Abstract: Retrieved scientific literature can serve as inspiration for both human and AI scientists. Inspiration can take different forms: prior work may directly suggest how to address a problem, or surface directions at different levels of abstraction - zooming out to a more general view or zooming in to a concrete realization. We introduce RATIO (Retrieval Across Typed Ideation Operations), a large-scale benchmark in which relevance is defined by three operations which we name ideation moves: Address retrieves potential approaches for stated problems, Broaden retrieves more general formulations, and Specify retrieves concrete instantiations. RATIO is constructed from millions of full-text scientific papers across CS literature via a general recipe that extends discourse-marker distant supervision - previously used only for classification - to corpus-scale retrieval, combined with extensive LLM and human vetting. Experiments show that operation-
    
[^52]: ExecRubrics：可执行工具增强的评分标准，用于可验证且高效的长篇评估

    ExecRubrics: Executable Tool-Augmented Rubrics for Verifiable and Efficient Long-Form Evaluation

    [https://arxiv.org/abs/2608.22559](https://arxiv.org/abs/2608.22559)

    ExecRubrics通过将评分标准转化为可执行的Python函数，实现了可验证、高效且能捕捉复杂依赖关系的长篇评估，替代了昂贵的黑盒LLM评判器。

    

    摘要：arXiv:2608.22559v1 公告类型：新 摘要：评分标准旨在通过将回答质量分解为可解释的准则，使语言模型评估透明化。然而，自然语言评分标准往往含糊不清，需要黑盒LLM评判器，并且通常假设准则通过线性加权和独立聚合，这限制了其捕捉依赖关系、替代方案、惩罚和覆盖条件的能力。我们提出ExecRubrics，一个将评分标准表示为紧凑可执行程序的框架。ExecRubrics将评估逻辑编码为可验证的Python评分函数，赋予自然语言评分标准意图一种操作语义：一个可检查、可执行和可编辑的固定决策程序。在三个长篇回答基准测试——HealthBench、HelpSteer和ArgQuality上，我们展示了ExecRubrics可以替代昂贵的黑盒评判器，在偏好排序中优于或匹配自然语言评分标准基线，具有最佳偏好性能。

    arXiv:2608.22559v1 Announce Type: new  Abstract: Rubrics aim to make language-model evaluation transparent by decomposing response quality into interpretable criteria. However, natural-language rubrics are often ambiguous, require black-box LLM judges, and typically assume criteria aggregate independently through linear weighted sums, limiting their ability to capture dependencies, alternatives, penalties, and override conditions. We propose ExecRubrics, a framework for representing rubrics as compact executable programs. ExecRubrics encodes evaluation logic as verifiable Python scoring functions, giving natural-language rubric intent an operational semantics: a fixed decision procedure that can be inspected, executed, and edited. On three long-form response benchmarks-HealthBench, HelpSteer, and ArgQuality-we show that ExecRubrics can substitute for expensive black-box judges in ranking preferred over dispreferred responses, matching or improving NL rubric baselines with best preferen
    
[^53]: MITRE-SAGE：一种多智能体网络安全问答模型

    MITRE-SAGE: A Multi-Agent Cybersecurity Question-Answering model

    [https://arxiv.org/abs/2608.16921](https://arxiv.org/abs/2608.16921)

    MITRE-SAGE通过多智能体检索增强生成框架，结合语义与结构化网络安全知识，解决了LLM在网络安全问答中的知识不足和幻觉问题，提升了可靠性和可解释性。

    

    有效的网络安全运营需要及时、准确地分析大规模异构安全信息；然而，分析师日益面临信息过载、警报疲劳和时间受限的决策挑战。尽管大型语言模型（LLMs）在问答（QA）方面展现出有前景的能力，但其在网络安全领域的有效性仍受限于领域知识不足、易产生幻觉以及难以捕捉语义和结构关系的缺陷。本工作提出MITRE-SAGE，一种多智能体检索增强生成框架，该框架整合语义和结构化的网络安全知识，以提高基于LLM的问答系统的可靠性和可解释性。通过将复杂任务分解为查询解释、证据检索和答案合成，MITRE-SAGE有效支持漏洞评估、威胁分析等网络安全任务。

    arXiv:2608.16921v1 Announce Type: cross  Abstract: Effective cybersecurity operations require timely and accurate analysis of large-scale heterogeneous security information; however, analysts increasingly struggle with information overload, alert fatigue, and time-constrained decision-making. Although large language models (LLMs) have demonstrated promising capabilities for question answering (QA), their effectiveness in cybersecurity remains limited by insufficient domain knowledge, a tendency to hallucinate, and difficulties in capturing both semantic and structural relationships. This work proposes MITRE-SAGE, a multi-agent retrieval-augmented generation framework that integrates semantic and structural cybersecurity knowledge to improve the reliability and interpretability of LLM-based QA systems. By decomposing complex tasks into query interpretation, evidence retrieval, and answer synthesis, MITRE-SAGE effectively supports cybersecurity tasks such as vulnerability assessment, thr
    
[^54]: 稠密扩展，稀疏锚定：面向混合检索的通道非对称查询扩展

    Dense Expands, Sparse Anchors: Channel-Asymmetric Query Expansion for Hybrid Retrieval

    [https://arxiv.org/abs/2608.15851](https://arxiv.org/abs/2608.15851)

    本文提出DESA方法，通过通道非对称的查询扩展（稠密端正交残差扩展、稀疏端分数乘积锚定），解决了混合检索中固定截断值导致评估结果不稳定的问题。

    

    基于大语言模型的查询扩展通过生成类似文档的段落来提升检索效果。然而，在混合检索中，大多数评估方法融合固定的top-L稠密和稀疏排序。由于截断值同时控制跨通道贡献进入融合的方式以及每个排序被访问的程度，在某个L值下测得的增益在另一个L值下可能发生变化甚至反转。我们通过完整列表融合下的检索效果评估来分离这些影响，并记录策略特定的每通道重放停止深度，在该深度下其有序top-K得到验证。随后，我们提出了DESA（稠密扩展与稀疏锚定），一种通道非对称的查询扩展方法。大语言模型生成互补的参考段落；正交残差扩展将这些段落的新语义方向添加到稠密查询中，而分数乘积锚定则将其词汇线索纳入稀疏检索，同时不扩大原始查询的词汇支持范围。

    arXiv:2608.15851v1 Announce Type: cross  Abstract: LLM-based query expansion improves retrieval by generating document-like passages. In hybrid retrieval, however, most evaluations fuse fixed top-$L$ dense and sparse rankings. Because the cutoff controls both which cross-channel contributions enter fusion and how much of each ranking is accessed, gains measured at one $L$ can change or reverse at another. We separate these effects by evaluating retrieval effectiveness under complete-list fusion and recording the policy-specific per-channel replay stopping depths at which its ordered top-$K$ is certified. We then introduce DESA (Dense Expansion and Sparse Anchoring), a channel-asymmetric query expansion method. An LLM generates complementary reference passages; orthogonal residual expansion adds their new semantic directions to the dense query, while score-product anchoring incorporates their lexical cues into sparse retrieval without broadening the original query's lexical support. Acr
    
[^55]: SPARC：面向生成式推荐的序列感知渐进式属性路由与压缩框架

    SPARC: Sequence-aware Progressive Attribute Routing and Compression Framework for Generative Recommendation

    [https://arxiv.org/abs/2607.25339](https://arxiv.org/abs/2607.25339)

    提出SPARC框架，通过序列感知的渐进式属性路由与压缩机制，解决生成式推荐中异构行为属性全展开导致输入过长、直接压缩又过早丢失上下文信息的矛盾。

    

    生成式推荐将物品标记为离散的语义ID（Semantic IDs, SIDs），并以自回归方式从用户的历史SID序列中生成目标物品。尽管现有的SID融合了多模态和结构化信息，但它们通常是静态分配的，且与当前的交互上下文无关。在工业场景中，每种用户行为还包含异构属性，例如类别、品牌、价格、行为类型和时间戳。完全展开这些特征会大幅增加输入长度，而直接将它们压缩为单一表示则可能过早丢弃与上下文相关的信息。我们提出了SPARC——面向生成式推荐的序列感知渐进式属性路由与压缩框架。SPARC首先对每种字段类型的序列依赖关系进行建模，以获得上下文感知的（表示）……（原文摘要在此处截断）

    arXiv:2607.25339v2 Announce Type: replace  Abstract: Generative recommendation tokenizes items as discrete Semantic IDs (SIDs) and autoregressively generates target items from users' historical SID sequences. Although existing SIDs incorporate multimodal and structured information, they are typically statically assigned and independent of the current interaction context. In industrial scenarios, each behavior also contains heterogeneous attributes, such as category, brand, price, behavior type, and timestamp. Fully expanding these features greatly increases the input length, while directly compressing them into a single representation may prematurely discard context-relevant information.   We propose \textbf{SPARC}, \uline{\textbf{S}}equence-aware \uline{\textbf{P}}rogressive \uline{\textbf{A}}ttribute \uline{\textbf{R}}outing and \uline{\textbf{C}}ompression Framework for Generative recommendation. SPARC first models the sequential dependencies of each field type to obtain context-awa
    
[^56]: 重新思考基于大语言模型的推荐系统中的公平性：一项综述

    Rethinking Fairness in LLM-Based Recommender Systems: A Survey

    [https://arxiv.org/abs/2606.28340](https://arxiv.org/abs/2606.28340)

    这是首篇专门聚焦于基于大语言模型的推荐系统中公平性问题的综述，通过偏见机制与公平性目标的双维度框架，系统梳理了相关研究、评估方法与缓解策略。

    

    大语言模型正在通过实现更加语义化、生成式和交互式的推荐流程来重塑推荐系统。然而，这一转变也带来了新的公平性挑战，因为偏见可能源于预训练知识、提示词、生成的解释、解码策略以及反馈循环。本综述对基于大语言模型的推荐系统（LLM4Rec）中的公平性进行了系统性回顾，通过偏见机制与公平性目标的双维度视角组织现有研究，并对评估方法与缓解策略进行了结构化概述。我们进一步将公平性与更广泛的可信性问题相联系，包括可解释性、隐私、鲁棒性和可控性。据我们所知，这是首个专门聚焦于LLM4Rec公平性的综述，旨在为未来关于全面且可靠的公平性评估的研究提供结构化基础。

    arXiv:2606.28340v2 Announce Type: replace  Abstract: Large Language Models (LLMs) are reshaping recommender systems by enabling more semantic, generative, and interactive recommendation pipelines. However, this shift also introduces new fairness challenges, as biases may arise from pretrained knowledge, prompts, generated explanations, decoding strategies, and feedback loops. This survey provides a systematic review of fairness in LLM-based recommender systems (LLM4Rec), organizing existing studies through a two-dimensional view of bias mechanisms and fairness targets, together with a structured overview of the evaluation landscape and mitigation strategies. We further connect fairness with broader trustworthy concerns, including explainability, privacy, robustness, and controllability. To the best of our knowledge, this is the first survey specifically focused on fairness in LLM4Rec, aiming to provide a structured foundation for future research on comprehensive and reliable fairness e
    
[^57]: Querit-Reranker：通过高效的无标签分布适配训练紧凑的多语言重排序器

    Querit-Reranker: Training Compact Multilingual Rerankers via Efficient Label-Free Distribution Adaptation

    [https://arxiv.org/abs/2606.19037](https://arxiv.org/abs/2606.19037)

    Querit-Reranker提出了一套以数据为中心、无需人工标注的高效适配流水线，通过合成查询挖掘、教师分数软标签蒸馏和检查点合并，训练出紧凑且可部署的多语言重排序器。

    

    一个可部署的多语言重排序器不仅需要在语言、领域和排序任务之间具备泛化能力，还必须保持高效，以便在实际系统中充当第二阶段重排序器。然而，将其适配到新的目标分布通常需要大量任务特定的相关性标注。我们提出了Querit-Reranker，这是一个通过以数据为中心的流水线训练的多语言重排序器家族，实现了标签高效的适配。我们将其具体化为Querit-Reranker-A0.4B（由具有0.4B激活参数的自研MoE骨干网络初始化）和Querit-Reranker-4B（由Qwen3-Embedding-4B初始化）。我们的流水线首先从大规模面向排序的数据中学习通用相关性建模，然后通过合成查询挖掘并以教师分数作为连续软标签来适配目标分布。为了整合互补的任务适配优势，我们进一步通过球面线性插值合并检查点……

    arXiv:2606.19037v2 Announce Type: replace  Abstract: A deployable multilingual reranker must not only generalize across languages, domains, and ranking tasks, but also remain efficient to serve as a second-stage reranker in practical systems. However, adapting it to new target distributions typically requires extensive task-specific relevance annotations. We present Querit-Reranker, a family of multilingual rerankers trained with a data-centric pipeline for label-efficient adaptation. We instantiate it as Querit-Reranker-A0.4B, initialized from an in-house MoE backbone with 0.4B activated parameters, and Querit-Reranker-4B, initialized from Qwen3-Embedding-4B. Our pipeline first learns general relevance modeling from large-scale ranking-oriented data, then adapts to target distributions through synthetic-query mining with teacher scores as continuous soft labels. To consolidate complementary task-adapted strengths, we further merge checkpoints via spherical linear interpolation, obtain
    
[^58]: 探索面向模型专业化的自主智能体数据工程

    Exploring Autonomous Agentic Data Engineering for Model Specialization

    [https://arxiv.org/abs/2605.30407](https://arxiv.org/abs/2605.30407)

    该论文提出了“自主智能体数据工程”这一新任务，首次验证LLM能够自主执行端到端数据工程流水线来驱动模型专业化，其中GPT-5.2自主构建的训练数据使学生模型性能提升57.29%。

    

    大型语言模型（LLMs）在通用任务上已展现出强大性能，但在缺乏高质量领域特定数据的情况下，往往难以适应专业化领域。现有的基于LLM的数据整理方法主要依赖人工设计的工作流程，尚未检验LLM能否自主执行端到端的数据工程流水线以实现模型专业化。我们正式提出了“自主智能体数据工程”这一新任务，旨在评估LLM作为自主数据工程师、通过端到端数据整理来推动模型专业化的能力。我们将数据视为可优化的组件，研究能够跨多个领域规划、生成并迭代优化训练数据的智能体，并以训练后性能提升作为指导信号。实验表明，自主LLM数据工程师带来了显著收益，例如GPT-5.2构建了一套训练课程，使学生模型的性能提升了57.29%，完全……

    arXiv:2605.30407v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have demonstrated strong performance on general tasks, while often struggling to adapt to specialized domains without high-quality domain-specific data. Existing LLM-based data curation methods primarily rely on human-designed workflows, leaving it unexamined whether LLMs can autonomously execute an end-to-end data engineering pipeline for model specialization. We formalize Autonomous Agentic Data Engineering, a novel task designed to evaluate LLMs as autonomous data engineers that drive model specialization through end-to-end data curation. We frame data as an optimizable component and study agents that plan, generate, and iteratively optimize training data across multiple domains, guided by post-training performance improvement. Experiments show that autonomous LLM data engineers yield substantial gains, as GPT-5.2 constructs a training curriculum that improves a student model by 57.29%, entirely 
    
[^59]: LexPath：一种面向法律领域的多路径法条检索框架

    LexPath: A domain-oriented multi-path framework for legal article retrieval

    [https://arxiv.org/abs/2605.30205](https://arxiv.org/abs/2605.30205)

    提出面向法律领域的多路径框架LexPath，通过结合IRAC引导的稀疏检索、基于法律层级与引用关系的稠密检索以及意图感知重排序，有效区分法律相关法条与仅文本相似的干扰法条，提升法条检索的准确性。

    

    法条检索对于构建可追溯、可靠的法律人工智能系统至关重要，此类系统的结论必须基于具体的法律条文。然而，通用检索方法严重依赖词汇或语义相似度，难以区分法律上真正相关的法条与文本相似但法律上不适用的法条，尤其是当它们背后的法律意图存在差异时。为弥补这一不足，我们提出了LexPath，一种面向法律领域的多路径框架，包含多路径检索模块和意图感知重排序模块。检索模块结合两条互补的领域特定路径来收集候选法条：一条是IRAC引导的稀疏路径，利用具有法律信息量的关键词对查询进行扩展；另一条是结构引导的稠密路径，使用基于法律层级和引用关系构建的难负样本进行训练。重排序模块通过融入法律意图信息进一步细化候选法条的排序。

    arXiv:2605.30205v2 Announce Type: replace  Abstract: Legal article retrieval is critical for building traceable and reliable legal AI systems, where conclusions must be grounded in specific legal articles. However, general-purpose retrieval methods rely heavily on lexical or semantic similarity, making it difficult to distinguish legally relevant articles from textually similar but legally inapplicable ones, particularly when they differ in their underlying legal intent. To bridge this gap, we propose LexPath, a domain-oriented multi-path framework comprising a multi-path retrieval module and an intent-aware reranking module. The retrieval module combines two complementary domain-specific paths to collect candidate articles: an IRAC-guided sparse path that expands queries with legally informative keywords, and a structure-guided dense path trained with hard negatives derived from legal hierarchy and citation relations. The reranking module further refines candidate rankings by incorpor
    
[^60]: 证据缺失不等于证据不足：诊断事实核查中的NEI构建伪影

    Evidence Absence Is Not Evidence Insufficiency: Diagnosing NEI Construction Artifacts in Fact Verification

    [https://arxiv.org/abs/2605.26663](https://arxiv.org/abs/2605.26663)

    该论文提出NEI-CAP诊断协议，揭示事实核查基准中NEI标签的构建方式会引入捷径伪影，导致验证器识别“证据不足”的能力无法跨构建方式可靠迁移。

    

    证据缺失并不等于证据不足，但事实核查基准可能使二者在观察上显得相似。“信息不足”标签通常是通过构造的证据条件来实现的，而这一选择在无形中决定了验证器会学到什么。我们提出了NEI-CAP，一个面向证据不足评估的、感知构建方式的诊断协议。每个NEI样本都标注了产生它的构建类别；NEI-CAP审计捷径线索，通过人工裁决验证困难样本，并检验能力能否跨构建方式迁移。我们在SciFact上实例化该协议，并以FEVER和HoVer作为有限的外部对照。在这些设置中，NEI能力无法可靠迁移：在易产生捷径的构建方式上训练的编码器验证器和指令微调解码器，无法识别语义相关的证据不足情形，而混合构建训练虽能缩小差距但（摘要在此处截断）

    arXiv:2605.26663v2 Announce Type: replace  Abstract: Evidence absence is not evidence insufficiency, but fact verification benchmarks can make them observationally similar. The Not Enough Information (NEI) label is often operationalized through constructed evidence conditions, and that choice silently determines what a verifier learns. We introduce NEI-CAP, a construction-aware diagnostic protocol for insufficient-evidence evaluation. Each NEI example carries the construction family that produced it; NEI-CAP audits shortcut cues, validates hard cases through human adjudication, and tests whether competence transfers across constructions. We instantiate the protocol on SciFact, with FEVER and HoVer as bounded external controls. Across these settings, NEI competence does not transfer reliably: encoder verifiers and an instruction-tuned decoder trained on shortcut-prone constructions fail to recognize semantically related insufficient evidence, and mixed-construction training narrows but 
    
[^61]: SciAtlas：面向知识驱动AI研究的可计算科学图谱

    SciAtlas: A Computable Atlas of Science for Knowledge-Grounded AI Research

    [https://arxiv.org/abs/2605.22878](https://arxiv.org/abs/2605.22878)

    SciAtlas提出了一个共享的、机器可操作的跨学科学术知识基础设施，通过在统一模式下整合多层次知识并采用统一的神经符号检索机制，为知识驱动的人工智能科学研究提供了可靠的科学知识支撑。

    

    人工智能正迅速进入科学研究的核心工作流程。然而，可靠的科学推理需要获取具有足够广度、深度和标准化程度的积累科学知识。当前的AI科学家通常通过针对特定工作流程和学科领域的管道来组装科学知识，这种方式覆盖不完整、关系隐含不明确，并导致知识获取路径碎片化。在此，我们提出了SciAtlas，一个共享的、机器可操作的跨学科学术知识基础设施，它在统一模式（schema）下整合了证据层、概念层、学科层、专业知识层和规范层。SciAtlas进一步实现了统一的神经符号检索机制，该机制能够将异构研究对象进行落地关联，在学术拓扑结构中传播相关性，并将产生的相关性场投射到每个科学工作流程所需的上下文中。

    arXiv:2605.22878v2 Announce Type: replace  Abstract: Artificial intelligence is rapidly entering the core workflows of scientific research. Yet reliable scientific reasoning requires access to accumulated scientific knowledge with sufficient breadth, depth, and standardization. Current AI scientists typically assemble scientific knowledge through workflow- and discipline-specific pipelines, which provide incomplete coverage, leave relations implicit, and make knowledge acquisition pathways fragmented. Here we present SciAtlas, a shared, machine-actionable cross-disciplinary scholarly knowledge infrastructure that integrates evidential, conceptual, disciplinary, expertise, and normative layers under a shared schema. SciAtlas further achieves a unified neuro-symbolic retrieval mechanism that grounds heterogeneous research objects, propagates relevance across the scholarly topology, and projects the resulting relevance field into the context required by each scientific workflow. Across th
    
[^62]: CASTLE：面向冷启动自然语言搜索的对比与种子引导训练

    CASTLE: Contrastive and Seed-Guided Training for Cold-Start Natural Language Search

    [https://arxiv.org/abs/2605.21812](https://arxiv.org/abs/2605.21812)

    CASTLE是一个基于LLM的冷启动框架，通过种子查询引导的提示生成真实合成查询、并利用预订会话构造的对比房源对生成接近零误报的相关性标签，支撑了Airbnb自然语言搜索的完整生命周期。

    

    部署自然语言搜索系统面临一个关键的冷启动挑战：没有真实用户查询可供学习语言模式，也没有相关性标签可用于训练排序模型。我们提出了CASTLE（Contrastive And Seed-guided Training for natural Language sEarch，面向自然语言搜索的对比与种子引导训练），这是一个基于大语言模型（LLM）的框架，可从结构化目录数据中生成合成查询和相关性标签，为Airbnb的自然语言搜索在其整个生命周期中提供支持。CASTLE做出了三项贡献。第一，我们通过将结构引导提示与来自用户研究的种子查询相结合来生成真实查询，使用模板、少样本和基于属性的提示变体，并配合显式的多样性机制来防止查询坍缩。第二，我们通过从预订会话中导出的对比房源对，以构造方式生成相关性标签，在不依赖LLM判断的情况下实现了接近零的误报率。第三，CASTLE的结构化输入设计具有灵活性（摘要原文在此处截断）

    arXiv:2605.21812v2 Announce Type: replace  Abstract: Deploying natural language search systems presents a critical cold-start challenge: no real user queries to learn linguistic patterns, and no relevance labels to train ranking models. We present CASTLE (Contrastive And Seed-guided Training for natural Language sEarch), an LLM-based framework for generating synthetic queries and relevance labels from structured catalog data, powering Airbnb's natural language search across its full lifecycle.   CASTLE makes three contributions. First, we generate realistic queries by combining structure-guided prompting with seed queries from user research, using template, few-shot, and attribute-grounded prompt variants together with explicit variety mechanisms to prevent query collapse. Second, we produce relevance labels by construction via contrastive listing pairs derived from booking sessions, achieving near-zero false positives without LLM judgment. Third, CASTLE's structured input design is fl
    
[^63]: 答案气泡：AI中介搜索中的信息暴露

    Answer Bubbles: Information Exposure in AI-Mediated Search

    [https://arxiv.org/abs/2603.16138](https://arxiv.org/abs/2603.16138)

    该研究通过对五个搜索系统中11,000个真实查询的分析，发现生成式AI搜索在引用来源上存在显著的选择偏差，且搜索功能会使AI摘要中的模糊限定表述减少多达60%，进一步加剧了引用偏差并使表述更加自信。

    

    生成式搜索系统正日益用AI生成的摘要取代基于链接的检索，然而，人们对这些系统在来源、语言以及对所引材料的忠实度方面有何差异仍知之甚少。我们在三个层面考察了五个系统——原生GPT、Search GPT、带Grok的Perplexity搜索、Google AI Overviews以及传统Google搜索——对11,000个真实搜索查询的响应：来源多样性、生成摘要的语言特征，以及来源-摘要忠实度。我们发现，生成式搜索系统在其引用中表现出显著的“来源选择”偏差，倾向于偏好某些来源。引入搜索功能还会选择性地削弱认知标记，使AI生成摘要中的模糊限定表述减少多达60%，同时保留自信的语言。与此同时，AI摘要进一步加剧了引用偏差：维基百科和较长的来源受到不成比例的……

    arXiv:2603.16138v2 Announce Type: replace-cross  Abstract: Generative search systems are increasingly replacing link-based retrieval with AI-generated summaries, yet little is known about how these systems differ in sources, language, and fidelity to cited material. We examine responses to 11,000 real search queries across five systems---vanilla GPT, Search GPT, Perplexity Search with Grok, Google AI Overviews, and traditional Google Search---at three levels: source diversity, linguistic characterization of the generated summary, and source-summary fidelity. We find that generative search systems exhibit significant \textit{source-selection} biases in their citations, favoring certain sources over others. Incorporating search also selectively attenuates epistemic markers, reducing hedging by up to 60\% while preserving confidence language in the AI-generated summaries. At the same time, AI summaries further compound the citation biases: Wikipedia and longer sources are disproportionate
    
[^64]: NanoVDR：将20亿参数视觉-语言检索器蒸馏为7000万参数纯文本编码器，用于视觉文档检索

    NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval

    [https://arxiv.org/abs/2603.12824](https://arxiv.org/abs/2603.12824)

    NanoVDR通过解耦文档索引与查询编码，使用冻结的2B参数VLM教师模型离线索引文档，并蒸馏出仅69M参数的纯文本学生模型来编码查询，在保持检索质量的同时大幅降低了推理延迟和GPU依赖。

    

    基于视觉-语言模型（VLM）的检索器已将视觉文档检索（VDR）的质量提升至令人瞩目的水平。然而，这类模型在文档索引和查询编码两个环节都需要同一个数十亿参数规模的编码器，即使是纯文本查询也会带来高延迟和对GPU的依赖。我们观察到这种设计存在不必要的对称性：文档在视觉上十分复杂，需要强大的视觉理解能力，而查询只是简短的文本字符串。NanoVDR利用了查询与文档之间的这种不对称性，将两条编码路径解耦：一个冻结的20亿参数VLM教师模型在离线阶段对文档进行索引，而一个经蒸馏得到的、仅6900万参数的纯文本学生模型则在推理时对查询进行编码。关键的设计选择在于蒸馏目标函数。通过在三个骨干网络和22个ViDoRe基准数据集上对六种蒸馏目标进行系统性比较，我们发现基于查询文本的逐点余弦对齐始终优于基于排序的方法。

    arXiv:2603.12824v3 Announce Type: replace-cross  Abstract: Vision-Language Model (VLM) based retrievers have advanced visual document retrieval (VDR) to impressive quality. They require the same multi-billion parameter encoder for both document indexing and query encoding, incurring high latency and GPU dependence even for plain-text queries. We observe that this design is unnecessarily symmetric: documents are visually complex and demand strong visual understanding, whereas queries are just short text strings. NanoVDR exploits this query--document asymmetry by decoupling the two encoding paths: a frozen 2B VLM teacher indexes documents offline, while a distilled text-only student as small as 69M parameters encodes queries at inference. The key design choice is the distillation objective. Through systematic comparison of six objectives across three backbones and 22 ViDoRe benchmark datasets, we find that pointwise cosine alignment on query text consistently outperforms ranking-based an
    
[^65]: SORT：面向工业级推荐系统的系统优化排名Transformer

    SORT: A Systematically Optimized Ranking Transformer for Industrial-scale Recommenders

    [https://arxiv.org/abs/2603.03988](https://arxiv.org/abs/2603.03988)

    本文提出了一种系统优化的排名Transformer模型，通过请求中心样本组织、局部注意力、查询剪枝和生成式预训练等创新，有效解决了工业级推荐中高特征稀疏性和低标签密度问题，并提升了硬件利用率。

    

    摘要：虽然Transformer通过卓越的可扩展性在大型语言模型中取得了显著成功，但其在工业级排名模型中的应用仍处于初期阶段，受到高特征稀疏性和低标签密度挑战的阻碍。在本文中，我们提出了SORT（系统优化排名Transformer），这是一种可扩展模型，旨在弥合Transformer与工业级排名模型之间的差距。我们通过一系列优化来解决高特征稀疏性和低标签密度挑战，包括请求中心样本组织、局部注意力、查询剪枝和生成式预训练。此外，我们引入了对分词、多头注意力和前馈网络模块的一系列改进，这些改进共同稳定了训练过程并扩大了模型容量。为了最大化硬件效率，我们优化了训练系统，以提高模型FLOPs利用率。

    arXiv:2603.03988v2 Announce Type: replace  Abstract: While Transformers have achieved remarkable success in LLMs through superior scalability, their application in industrial-scale ranking models remains nascent, hindered by the challenges of high feature sparsity and low label density. In this paper, we propose SORT (Systematically Optimized Ranking Transformer), a scalable model designed to bridge the gap between Transformers and industrial-scale ranking models. We address the high feature sparsity and low label density challenges through a series of optimizations, including request-centric sample organization, local attention, query pruning and generative pre-training. Furthermore, we introduce a suite of refinements to the tokenization, multi-head attention (MHA), and feed-forward network (FFN) modules, which collectively stabilize the training process and enlarge the model capacity. To maximize hardware efficiency, we optimize our training system to elevate the model FLOPs utiliza
    
[^66]: 构建更好的仅编码器交叉编码器：神经重排序训练策略的受控研究

    Building Better Encoder-only Cross-Encoders: A Controlled Study of Training Strategies for Neural Re-ranking

    [https://arxiv.org/abs/2603.03010](https://arxiv.org/abs/2603.03010)

    该研究通过162次受控训练实验系统比较了不同骨干网络与训练目标在神经重排序中的表现，发现强调相对比较的成对MarginMSE和列表InfoNCE目标始终优于其他方法。

    

    从Transformer骨干网络微调而来的交叉编码器仍然是第二阶段重排序的标准方法，而近期的知识蒸馏策略已经大幅缩小了其与LLM重排序器之间的差距。然而，这些策略尚未在受控条件下进行过比较。特别是，来自LLM排序器的蒸馏与来自强交叉编码器教师的蒸馏、或与纯监督目标相比孰优孰劣仍不清楚。此外，较新的骨干网络（RoBERTa、ELECTRA、DeBERTaV3、ModernBERT）相比原始BERT能带来多大提升也不明确。我们进行了162次受控训练实验（9个骨干网络 × 6个目标函数 × 3个随机种子），涵盖逐点、成对和列表损失，并同时使用人工标注和两种蒸馏信号，在TREC-DL、MSMARCO dev、BEIR、LoTTE和Robust04上进行评估。我们发现，强调相对比较的目标函数——成对MarginMSE和列表InfoNCE——始终优于其他替代方案。

    arXiv:2603.03010v2 Announce Type: replace  Abstract: Cross-encoders fine-tuned from Transformer backbones remain the standard for second-stage re-ranking, and recent knowledge-distillation strategies have closed much of the gap with LLM re-rankers. However, these strategies have not been compared under controlled conditions. In particular, it remains unclear how distillation from LLM rankers compares to distillation from strong cross-encoder teachers, or to purely supervised objectives. It is also unclear how much newer backbones (RoBERTa, ELECTRA, DeBERTaV3, ModernBERT) contribute compared to the original BERT. We run 162 controlled training runs (9 backbones x 6 objectives x 3 seeds), spanning pointwise, pairwise, and listwise losses with both human labels and two distillation signals, and evaluate on TREC-DL, MSMARCO dev, BEIR, LoTTE, and Robust04. We find that objectives emphasizing relative comparisons - pairwise MarginMSE and listwise InfoNCE - consistently outperform alternative
    
[^67]: UniFAR：一种面向科学文献的统一分面感知检索框架

    UniFAR: A Unified Facet-Aware Retrieval Framework for Scientific Documents

    [https://arxiv.org/abs/2602.23766](https://arxiv.org/abs/2602.23766)

    提出了UniFAR，一个统一的分面感知检索框架，通过多粒度表示与聚合模块在共享表示空间中同时支持文档-文档和问答-文档两种科学文献检索范式，从而融合二者的互补优势。

    

    科学文献检索（SDR）在现代科学研究中发挥着关键作用，支持知识发现和基于证据的推理。它沿着两种范式发展：由文档间对比学习驱动的文档-文档（doc-doc）检索，以及随着大语言模型（LLM）和RAG在自然语言交互中应用而兴起的问答-文档（q-doc）检索。在实践中，科学工作流程同时依赖这两种范式，既需要根据种子文档检索相关论文，也需要根据用户问题识别相关文档。然而，现有方法通常将这两种范式分开处理，阻碍了它们互补优势的发挥。为了解决这一问题，我们提出了UniFAR，一个统一的分面感知检索框架，能够在共享表示空间中同时支持文档-文档检索和问答-文档检索。UniFAR引入了多粒度表示与聚合模块，以统一短问题和（原文摘要在此处截断）

    arXiv:2602.23766v2 Announce Type: replace  Abstract: Scientific document retrieval (SDR) plays a critical role in modern scientific research, supporting knowledge discovery and evidence-based reasoning. It has evolved along two paradigms: document--document (doc-doc) retrieval driven by inter-document contrastive learning, and question--document (q-doc) retrieval emerging from LLMs and RAG for natural-language interaction. In practice, scientific workflows rely on both paradigms, requiring retrieval of related papers given a seed document and identifying relevant documents given a user question. However, existing methods typically treat these paradigms separately, hindering their complementary strengths. To address this, we propose UniFAR, a unified facet-aware retrieval framework that jointly supports doc-doc and q-doc retrieval within a shared representation space. UniFAR introduces a multi-granularity representation and aggregation module to unify the encoding of short questions and
    
[^68]: MICE：用于高效重排序的最小交互交叉编码器

    MICE: Minimal Interaction Cross-Encoders for efficient Re-ranking

    [https://arxiv.org/abs/2602.16299](https://arxiv.org/abs/2602.16299)

    通过深入分析交叉编码器内部机制并移除多余的token交互，提出MICE新架构，在大幅降低计算开销的同时保持域内排序效果，并在域外场景中匹配甚至超越原有交叉编码器的性能。

    

    在信息检索（IR）中，交叉编码器具有最先进的排序效果，但其推理成本很高，限制了它们只能用作第二阶段的重排序器。先前的工作从两个基本独立的方向来解决这一瓶颈：一是通过注意力稀疏化加速交叉编码器的推理，二是通过使用更复杂的模型（如后期交互模型）来提升第一阶段检索的效果，从而减轻对重排序器的需求。在本工作中，我们通过对交叉编码器内部机制的深入分析，将这两个方向联系起来。通过识别并移除多余的交互，我们提出了MICE（最小交互交叉编码器），这是一种新的交叉编码器架构，能够在降低计算开销的同时保持有效性。大量评估表明，MICE在域内保留了其对应交叉编码器的大部分性能，在域外场景中甚至能匹配或超越原有性能，同时显著降低了计算成本。

    arXiv:2602.16299v4 Announce Type: replace  Abstract: In Information Retrieval (IR), cross-encoders deliver state-of-the-art ranking effectiveness but have a high inference cost, limiting their use to second-stage re-rankers. Prior work has addressed this bottleneck from two largely separate directions: accelerating cross-encoder inference through attention sparsification, or improving first-stage retrieval effectiveness to alleviate the need of a re-ranker, using more complex models, e.g. late-interactions. In this work, we bridge these two directions through an in-depth analysis of cross-encoder internal mechanisms. By identifying and removing superfluous interactions, we derive MICE (Minimal Interaction Cross-Encoders), a new cross-encoder architecture that retains effectiveness while reducing computational overhead. Extensive evaluations show MICE retains most of the performances of its cross-encoder counterparts in-domain and matches or even exceeds it in out-of-domain, while reduc
    
[^69]: GeoGR：实现时空感知的工业级生成式兴趣点（POI）推荐

    GeoGR: Enabling Spatio-Temporal Aware Industrial-scale Generative POI Recommendations

    [https://arxiv.org/abs/2602.10411](https://arxiv.org/abs/2602.10411)

    本文提出面向高德地图等导航型位置服务的地理生成式推荐框架GeoGR，通过地理感知的SID分词流水线解决高质量SID建模不足和大语言模型对齐不佳的问题，实现感知用户上下文变化、时空感知的工业级生成式POI推荐。

    

    下一步兴趣点（POI）预测是基于位置服务（LBS）中的一项基础任务，对于像高德地图（AMAP）这样在多样化生活场景中服务数十亿用户的大型导航平台而言尤为关键。尽管近期基于语义标识符（SID）的POI推荐方法取得了可喜的性能，但由于两个关键局限，它们在复杂、稀疏的真实世界环境中表现不佳：（1）对能够捕获跨类别时空协作关系的高质量SID建模不足；（2）大语言模型（LLMs）与POI推荐任务之间的对齐效果较差。为此，我们提出了GeoGR，一个专为高德地图这类导航型LBS量身定制的地理生成式推荐框架，它能够感知用户上下文状态的变化，并实现时空感知的POI推荐。GeoGR采用两阶段设计：（i）一个地理感知的SID分词流水线，显式学习……

    arXiv:2602.10411v2 Announce Type: replace  Abstract: Next Point-of-Interest (POI) prediction is a fundamental task in location-based services (LBS), especially critical for large-scale navigation platforms such as AMAP that serve billions of users in diverse lifestyle scenarios. Although recent POI recommendation approaches based on SIDs have achieved promising performance, they struggle in complex, sparse real-world environments due to two key limitations: (1) inadequate modeling of high-quality SIDs that capture cross-category spatio-temporal collaborative relationships, and (2) poor alignment between large language models (LLMs) and the POI recommendation task. To this end, we propose GeoGR, a geographic generative recommendation framework tailored for navigation-based LBS like AMAP, which perceives changes in users' contextual states and enables spatio-temporal aware POI recommendation. GeoGR features a two-stage design: (i) a geo-aware SID tokenization pipeline that explicitly lea
    
[^70]: HeMix：通过异构Token混合扩展工业级排序模型

    HeMix: Scaling Industrial Ranking Models with Heterogeneous Token Mixing

    [https://arxiv.org/abs/2602.09387](https://arxiv.org/abs/2602.09387)

    HeMix通过查询混合兴趣提取模块和HeteroMixer异构Token交互模块，在严格在线延迟约束下同时建模上下文相关与无关的用户兴趣，并实现高效的异构特征交互，从而扩展工业级排序模型。

    

    arXiv:2602.09387v3 公告类型：replace 摘要：为工业推荐系统扩展排序模型面临两个关键挑战：(C1) 现有的序列token化方法无法从异构行为源中同时捕获上下文相关与上下文无关的用户意图；(C2) 主流的交互机制既计算开销高昂又在语义上同质化，限制了严格在线延迟约束下的预测质量。我们提出HeMix，一个可扩展的排序模型，它将查询混合序列token化与异构特征交互统一起来。为解决(C1)，HeMix引入了查询混合兴趣提取模块，该模块采用动态查询与固定查询，从全局和实时行为序列中同时建模上下文相关与上下文无关的兴趣。为解决(C2)，我们设计了HeteroMixer模块，包含多头Token融合、异构混合Token交互与组对齐重校准（原文摘要至此截断）。

    arXiv:2602.09387v3 Announce Type: replace  Abstract: Scaling up ranking models for industrial recommender systems faces two critical challenges: (C1) existing sequence tokenization fails to jointly capture context-aware and context-invariant user intent from heterogeneous behavior sources, and (C2) prevailing interaction mechanisms are both computationally expensive and semantically homogeneous, limiting prediction quality under strict online latency constraints. We propose \textbf{HeMix}, a scalable ranking model that unifies query-mixed sequence tokenization with heterogeneous feature interaction. To address (C1), HeMix introduces a \textit{Query-Mixed Interest Extraction} module that employs dynamic and fixed queries to simultaneously model context-aware and context-invariant interests from global and real-time behavior sequences. To address (C2), we design the \textit{HeteroMixer} block, comprising Multi-Head Token Fusion, Heterogeneous Mixed-Token Interaction and Group-Aligned Rec
    
[^71]: 面向法律量刑预测的多源检索与推理

    Multi-Source Retrieval and Reasoning for Legal Sentencing Prediction

    [https://arxiv.org/abs/2602.04690](https://arxiv.org/abs/2602.04690)

    提出了MSR²框架，将大语言模型的多源检索与推理和强化学习的过程级奖励相结合，显著提升了法律量刑预测的准确性和可解释性。

    

    法律判决预测（LJP）旨在从案件事实中预测司法结果，通常包括法条预测、罪名预测和量刑预测。尽管近期方法在前两个子任务上表现良好，但法律量刑预测（LSP）仍然困难，因为它既需要细粒度的客观知识，又需要灵活的主观推理。为了解决这些局限性，我们提出了MSR²，一个将大语言模型中的多源检索与推理同强化学习相结合的框架。MSR²使大语言模型能够根据推理需求执行多源检索，并应用过程级奖励来引导中间的主观推理步骤。在两个真实世界数据集上的实验表明，MSR²提升了法律量刑预测的准确性和可解释性，为迈向实用的法律AI迈出了有希望的一步。我们的代码可在 https://github.com/cjj826/MSR2 获取。

    arXiv:2602.04690v2 Announce Type: replace  Abstract: Legal judgment prediction (LJP) aims to predict judicial outcomes from case facts and typically includes law article, charge, and sentencing prediction. While recent methods perform well on the first two subtasks, legal sentencing prediction (LSP) remains difficult due to its need for fine-grained objective knowledge and flexible subjective reasoning. To address these limitations, we propose $MSR^2$, a framework that integrates multi-source retrieval and reasoning in LLMs with reinforcement learning. $MSR^2$ enables LLMs to perform multi-source retrieval based on reasoning needs and applies a process-level reward to guide intermediate subjective reasoning steps. Experiments on two real-world datasets show that $MSR^2$ improves both accuracy and interpretability in LSP, providing a promising step toward practical legal AI. Our code is available at https://github.com/cjj826/MSR2.
    
[^72]: 结构化锚点剪枝：面向视觉文档检索的免训练多向量压缩

    Structural Anchor Pruning: Training-Free Multi-Vector Compression for Visual Document Retrieval

    [https://arxiv.org/abs/2601.20107](https://arxiv.org/abs/2601.20107)

    提出了一种免训练、与查询无关的结构化锚点剪枝框架（SAP），通过分数保留诊断、自动剪枝窗口选择和视觉入度中心性评分三项技术，在不依赖查询相关训练的情况下，有效压缩视觉文档检索中多向量索引的存储开销，即使在高压缩率下也能保持检索性能。

    

    arXiv:2601.20107v3 公告类型： replace-cross 摘要：近期的视觉-语言模型（例如 ColPali）能够实现细粒度的视觉文档检索（VDR），但会带来高昂的多向量索引存储开销。现有的免训练剪枝方法要么依赖启发式的层选择，要么在激进压缩下性能急剧下降，这导致先前的研究认为，有效的高压缩剪枝需要依赖查询相关的训练。我们通过结构化锚点剪枝（SAP）挑战了这一观点。SAP 是一种自校准、免训练、与查询无关的索引时框架，它结合了：(i) 分数保留（SR），一种白盒的逐层压缩诊断方法；(ii) SR 引导的窗口选择，可自动定位任何骨干网络的结构化剪枝区域，无需针对每个模型调整超参数；(iii) 一种视觉入度中心性评分器，用于识别该窗口内的锚点图像块。在涵盖 18、28 和 36 层骨干网络的三种架构上的 ViDoRe v1/v2 基准测试中，SAP 保留了 93--（原文摘要在此处截断）

    arXiv:2601.20107v3 Announce Type: replace-cross  Abstract: Recent Vision-Language Models (e.g., ColPali) enable fine-grained Visual Document Retrieval (VDR) but incur prohibitive multi-vector index storage overhead. Existing training-free pruning methods either rely on heuristic layer choices or degrade sharply under aggressive compression, leading prior work to argue that effective high-compression pruning requires query-dependent training. We challenge this view with Structural Anchor Pruning (SAP), a self-calibrating, training-free, query-agnostic index-time framework combining (i) Score Retention (SR), a white-box per-layer compression diagnostic; (ii) SR-guided window selection, which automatically locates the structural pruning region of any backbone with no per-model hyperparameters; and (iii) a visual in-degree centrality scorer that identifies anchor patches within that window. On ViDoRe v1/v2 across three architectures spanning 18, 28, and 36 backbone layers, SAP retains 93--
    
[^73]: CORE-T：面向文本到SQL的连贯表格检索

    CORE-T: COherent REtrieval of Tables for Text-to-SQL

    [https://arxiv.org/abs/2601.13111](https://arxiv.org/abs/2601.13111)

    提出无需训练、可扩展的CORE-T框架，通过LLM生成的表格用途元数据、预计算的表格兼容性缓存，以及“稠密检索—单次LLM筛选—两步增量调整”的流程，在大规模异构表格集合上实现连贯且可连接的多表检索，突破多表text-to-SQL的检索瓶颈。

    

    现实中的文本到SQL（text-to-SQL）工作流程通常需要连接多张表，因此准确检索相关的表格集合成为端到端性能的关键瓶颈。我们研究了一种“开卷”场景：查询需要在从多个来源汇集的大型异构表格集合上作答，且缺乏诸如数据库标识符等清晰的范围界定信号。在该场景下，稠密检索（DR）虽然召回率高，但会返回大量干扰项；而感知表间连接关系的替代方案往往依赖额外假设且/或带来高昂的推理开销。我们提出了CORE-T——一个可扩展、无需训练的框架：它利用LLM生成的表格用途元数据来丰富表格信息，并预先计算一个轻量级的表格兼容性缓存。在推理阶段，稠密检索返回top-K候选表；通过单次LLM调用选出连贯且可连接的表格子集；随后由两步增量调整阶段恢复强兼容性的表格。我们在Bird、Spider、MMQA和Beav……（原文在此处截断）等数据集上进行了评估。

    arXiv:2601.13111v3 Announce Type: replace-cross  Abstract: Realistic text-to-SQL workflows often require joining multiple tables. As a result, accurately retrieving the relevant set of tables becomes a key bottleneck for end-to-end performance. We study an open-book setting where queries must be answered over large, heterogeneous table collections pooled from many sources, without clean scoping signals such as database identifiers. Here, dense retrieval (DR) achieves high recall but returns many distractors, while join-aware alternatives often rely on extra assumptions and/or incur high inference overhead. We propose CORE-T, a scalable, training-free framework that enriches tables with LLM-generated purpose metadata and pre-computes a lightweight table-compatibility cache. At inference time, DR returns top-K candidates; a single LLM call selects a coherent, joinable subset, and a two-step additive adjustment stage restores strongly compatible tables. Across Bird, Spider, MMQA, and Beav
    
[^74]: Loci Similes：拉丁文学互文性提取基准

    Loci Similes: A Benchmark for Extracting Intertextualities in Latin Literature

    [https://arxiv.org/abs/2601.07533](https://arxiv.org/abs/2601.07533)

    本文提出了Loci Similes，一个用于拉丁文学互文性检测的基准数据集，包含约17.6万个文本片段和1,490个专家验证的平行文本，为利用语言模型捕捉历史文本间超越词汇重叠的语义相似性提供了标准化评估基础。

    

    追踪历史文本之间的联系是互文性研究的重要组成部分，它使学者能够重构作家的“虚拟图书馆”，并识别影响其创作过程的文献来源。这些互文性联系以多种形式呈现，从直接的逐字引用，到被词形变化所掩盖的微妙典故和转述。语言模型因其能够捕捉超越词汇重叠的语义相似性，为这一任务提供了一条有前景的路径。然而，标准化基准和易用数据集的匮乏阻碍了该任务新方法的开发。我们通过引入 Loci Similes 来填补这一空白，这是一个用于拉丁语互文性检测的基准，包含一个由约17.6万个文本片段组成的精选数据集以及1,490个经专家验证的平行文本，其中包括来自现有数据集的945个带标签的引用。利用这些数据，我们为检索任务建立了基线。

    arXiv:2601.07533v3 Announce Type: replace-cross  Abstract: Tracing connections between historical texts is an important part of intertextual research, enabling scholars to reconstruct the virtual library of a writer and identify the sources influencing their creative process. These intertextual links manifest in diverse forms, ranging from direct verbatim quotations to subtle allusions and paraphrases disguised by morphological variation. Language models offer a promising path forward due to their capability of capturing semantic similarity beyond lexical overlap. However, the development of new methods for this task is held back by the scarcity of standardized benchmarks and easy-to-use datasets. We address this gap by introducing Loci Similes, a benchmark for Latin intertextuality detection comprising a curated dataset of ~176k text segments and 1,490 expert-verified parallels, including 945 labeled references from an existing dataset. Using this data, we establish baselines for retr
    
[^75]: CoFiRec：面向生成式推荐的从粗到细分词方法

    CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation

    [https://arxiv.org/abs/2511.22707](https://arxiv.org/abs/2511.22707)

    提出CoFiRec生成式推荐框架，通过从粗到细的分词方式显式建模物品语义的层次结构，从而更好地捕捉Web交互中用户意图的渐进演化过程。

    

    在Web环境中，用户偏好通常会随着用户从浏览宽泛类别到探索具体物品而逐步细化。然而，现有的生成式推荐器忽视了这一自然的细化过程。生成式推荐将下一物品预测表述为对分词后用户历史的自回归生成，其中每个物品被表示为离散token的序列。先前的模型通常在量化之前将ID、类别、标题和描述等异构属性融合为单一嵌入，这抹平了物品固有的语义层次结构，无法捕捉Web交互过程中用户意图的渐进演化。为解决这一局限，我们提出了CoFiRec，这是一种新颖的生成式推荐框架，它将物品语义的从粗到细特性显式地融入分词过程中。它不再将所有属性压缩为单一的……

    arXiv:2511.22707v2 Announce Type: replace-cross  Abstract: In web environments, user preferences are often refined progressively as users move from browsing broad categories to exploring specific items. However, existing generative recommenders overlook this natural refinement process. Generative recommendation formulates next-item prediction as autoregressive generation over tokenized user histories, where each item is represented as a sequence of discrete tokens. Prior models typically fuse heterogeneous attributes such as ID, category, title, and description into a single embedding before quantization, which flattens the inherent semantic hierarchy of items and fails to capture the gradual evolution of user intent during web interactions. To address this limitation, we propose CoFiRec, a novel generative recommendation framework that explicitly incorporates the Coarse-to-Fine nature of item semantics into the tokenization process. Instead of compressing all attributes into a single 
    
[^76]: 评估跨模态检索中的视角偏差

    Evaluating Perspectival Biases in Cross-Modal Retrieval

    [https://arxiv.org/abs/2510.26861](https://arxiv.org/abs/2510.26861)

    本文提出跨文化、跨模态、跨语言的3XCM基准，揭示了多模态检索系统中系统性的视角偏差：图像到文本检索偏向流行语言条目而非语义忠实条目，文本到图像检索中存在语义对齐与文化关联之间的“牵引效应”，且低资源语言的相似性判断更易受文化熟悉视觉模式主导。

    

    多模态检索系统被期望在与查询的语言或文化起源无关的语义空间中运行。然而在实践中，检索结果却系统性地反映出视角偏差：即由语言流行度和文化关联所塑造的偏离。我们引入了跨文化、跨模态、跨语言多模态（3XCM）基准来隔离这些效应。我们的研究结果表明，在图像到文本的检索中，模型往往倾向于选择来自流行语言的条目，而非语义上更忠实的条目。在文本到图像的检索中，我们观察到在联合嵌入空间中语义对齐与语言条件化的文化关联之间存在持续的“牵引效应”。当语义表征未能得到充分解析时，尤其是在低资源语言中，相似性越来越多地由文化上熟悉的视觉模式所主导，从而导致系统性的关联偏差。

    arXiv:2510.26861v4 Announce Type: replace-cross  Abstract: Multimodal retrieval systems are expected to operate in a semantic space, agnostic to the language or cultural origin of the query. In practice, however, retrieval outcomes systematically reflect perspectival biases: deviations shaped by linguistic prevalence and cultural associations. We introduce the Cross-Cultural, Cross-Modal, Cross-lingual Multimodal (3XCM) benchmark to isolate these effects. Results from our studies indicate that, for image-to-text retrieval, models tend to favor entries from prevalent languages over those that are semantically faithful. For text-to-image retrieval, we observe a consistent "tugging effect" in the joint embedding space between semantic alignment and language-conditioned cultural association. When semantic representations are insufficiently resolved, particularly in low-resource languages, similarity is increasingly governed by culturally familiar visual patterns, leading to systematic asso
    
[^77]: LeMat-Synth：一个从科学文献中整理广泛合成流程数据库的多模态工具箱

    LeMat-Synth: a multi-modal toolbox to curate broad synthesis procedure databases from scientific literature

    [https://arxiv.org/abs/2510.26824](https://arxiv.org/abs/2510.26824)

    本文提出开源多模态工具箱LeMat-Synth Parser，利用大语言模型和视觉语言模型从8.1万篇科学文献的文本与图表中自动提取结构化合成流程，构建了迄今最大、最多样化的无机材料合成数据集LeMat-Synth（含5.8万个合成流程）。

    

    材料科学中先进实验方法的广泛普及催生了海量的程序性知识，这些知识分散在数十年的科学文献中，并以难以系统性分析的非结构化格式记录。在这项工作中，我们提出了LeMat-Synth Parser，一个模块化、开源的多模态提取工具箱，它利用大语言模型（LLMs）和视觉语言模型（VLMs）自动将从出版物文本和图表中提取的合成方案与性能指标结构化。将LeMat-Synth Parser应用于8.1万篇开放获取出版物，我们整理构建了LeMat-Synth，一个包含5.8万个合成流程的庞大数据集，据我们所知是迄今为止规模最大、最多样化的结构化无机材料合成数据集，基于领域特定本体涵盖了35种合成方法和16种材料类别。我们验证了提取质量……

    arXiv:2510.26824v2 Announce Type: replace-cross  Abstract: Wide access to advanced experimental methods in materials science has given rise to an abundance of procedural knowledge, which is scattered across decades of scientific literature and recorded in unstructured formats that are challenging to analyze systematically. In this work, we present LeMat-Synth Parser, a modular, open-source, and multi-modal extraction toolbox that utilizes large language models (LLMs) and vision language models (VLMs) to automatically structure synthesis protocols and performance metrics extracted from both text and figures of publications. Applying LeMat-Synth Parser to 81K open-access publications, we curate LeMat-Synth, an extensive dataset of 58K synthesis procedures and to our knowledge the largest and most diverse structured inorganic materials synthesis dataset to date, covering 35 synthesis methods and 16 material classes based on a domain-specific ontology. We validate extraction quality agains
    
[^78]: 论迭代贝叶斯更新的相合性与性能

    On the Consistency and Performance of the Iterative Bayesian Update

    [https://arxiv.org/abs/2508.09980](https://arxiv.org/abs/2508.09980)

    实验证明迭代贝叶斯更新（IBU）在度量隐私机制下的分布估计性能显著优于矩阵求逆等方法，并从数学上解释了INV性能欠佳的原因，而在k-RR和RAPPOR等本地差分隐私机制下IBU与INV性能相近。

    

    在许多情况下，估计用户数据中某些属性的分布非常重要。为了在保护用户隐私的同时实现这种估计，通常采用本地隐私模型，其中每个用户对其原始数据应用本地保护机制，向数据收集者发布带噪声的数据版本。随后使用矩阵求逆（INV）、RAPPOR估计器和迭代贝叶斯更新（IBU）等方法来估计原始分布。在本文中，我们通过实验证明，当用户数据通过度量隐私机制进行保护时，IBU显著优于其他方法。我们还从数学上解释了INV在这些度量隐私机制下性能欠佳的原因。相反，在典型的本地差分隐私机制下，特别是k-RR和RAPPOR，IBU表现出与INV相似的性能。此外，我们还研究……

    arXiv:2508.09980v2 Announce Type: replace-cross  Abstract: In many situations, estimating the distribution of users' data concerning certain attributes is important. To facilitate this estimation while safeguarding users' privacy, the local privacy model is commonly employed, in which each user applies a local protection mechanism to release a noisy version of their original data to the data collector. The original distribution is then estimated using methods such as Matrix Inversion (INV), RAPPOR's estimator, and iterative Bayesian update (IBU). In this article, we experimentally demonstrate that IBU significantly outperforms the other methods when user data is protected through metric privacy mechanisms. We also explain the mathematical reason for the suboptimal performance of INV under those metric privacy mechanisms. Conversely, IBU exhibits performance similar to INV under typical mechanisms of local differential privacy, specifically the k-RR and RAPPOR. In addition, we investiga
    
[^79]: 使用上下文学习建模排序属性

    Modeling Ranking Properties with In-Context Learning

    [https://arxiv.org/abs/2505.17736](https://arxiv.org/abs/2505.17736)

    提出了一种基于上下文学习的列表式LLM重排序方法，仅通过少量展示目标权衡的示例排序，无需任务特定训练即可实现群体公平性、极性多样性和主题多样性等排序目标。

    

    虽然标准的信息检索（IR）模型主要设计用于优化相关性，但现实世界的搜索通常需要平衡多样性、公平性等额外目标。这些目标依赖于文档之间的交互，通常通过事后启发式方法或监督学习方法来解决，而这需要对每个排序场景和数据集进行任务特定的训练。在本工作中，我们提出了一种面向列表式LLM重排序器的上下文学习（ICL）方法，从而消除了此类训练的需求。相反，我们的方法依赖于少量示例排序，这些示例展示了与当前输入相似的过去查询在各个目标之间所期望的权衡。我们在常见的IR测试集合上评估了我们的方法，以研究多个辅助目标：群体公平性、极性多样性和主题多样性。我们通过实验验证了我们的方法……

    arXiv:2505.17736v2 Announce Type: replace  Abstract: While standard IR models are primarily designed to optimize relevance, real-world search often needs to balance additional objectives such as diversity and fairness. These objectives depend on inter-document interactions and are commonly addressed using post-hoc heuristics or supervised learning methods, which require task-specific training for each ranking scenario and dataset. In this work, we propose an in-context learning (ICL) approach for listwise LLM rerankers that eliminates the need for such training. Instead, our method relies on a small number of example rankings that demonstrate the desired trade-offs between objectives for past queries similar to the current input. We evaluate our approach on common IR test collections to investigate multiple auxiliary objectives: group fairness (TREC Fairness), polarity diversity (Touch\'e), and topical diversity (TREC Deep Learning 2019/2020). We empirically validate that our method en
    
[^80]: HintEval：一个用于提示生成与提示评估的开源Python工具包

    HintEval: An Open-Source Python Toolkit for Hint Generation and Hint Evaluation

    [https://arxiv.org/abs/2502.00857](https://arxiv.org/abs/2502.00857)

    本文提出了开源Python工具包HintEval，它统一了提示生成与提示评估流程，通过标准化数据集访问、支持答案感知与答案无关的生成方法以及多种评估指标，解决了该领域数据碎片化、格式不一致和评估工具不可用的问题，并支持可复现的多维度研究。

    

    大语言模型日益直接给出用户问题的答案，这引发了人们对批判性思维与问题解决参与度下降的担忧。提示生成提供了一种替代方案，通过引导用户逐步逼近答案而不直接揭示答案；而提示评估则用于衡量这类引导的质量。然而，该领域的研究受到数据集碎片化、标注格式不一致以及评估工具往往局限于特定数据集或难以获取等因素的阻碍。为应对这些挑战，我们提出了HintEval，一个用于统一提示生成与评估的开源Python库。HintEval对多样化提示数据集的访问进行了标准化，支持答案感知与答案无关两类生成方法，并在共享数据模型中实现了多种评估指标。该工具包能够以极少的工程投入实现可复现的实验、跨数据集分析以及多维度评估。我们进一步

    arXiv:2502.00857v2 Announce Type: replace  Abstract: Large Language Models (LLMs) increasingly provide direct answers to user questions, raising concerns about reduced engagement in critical thinking and problem-solving. Hint generation offers an alternative by guiding users toward answers without revealing them, while hint evaluation assesses the quality of such guidance. Research in this area is hindered by fragmented datasets, inconsistent annotation formats, and evaluation tools that are often dataset-specific or unavailable. To address these challenges, we introduce HintEval, an open-source Python library for unified hint generation and evaluation. HintEval standardizes access to diverse hint datasets, supports answer-aware and answer-agnostic generation methods, and implements multiple evaluation metrics within a shared data model. The toolkit enables reproducible experimentation, cross-dataset analysis, and multi-dimensional evaluation with minimal engineering effort. We further
    
[^81]: 面向医疗健康指导的个性化提示学习

    Learning Personalized Prompts for Healthcare Guidance

    [https://arxiv.org/abs/2412.15957](https://arxiv.org/abs/2412.15957)

    提出个性化提示学习（PPL）框架，通过结合患者自身信息与临床相似病例的同伴信息构建初始个性化提示，并利用强化学习进行优化，使大语言模型能够生成与医生建议相符的个性化医疗健康指导。

    

    大型语言模型的快速发展已经改变了许多行业，包括医疗健康领域。在实践中，医院和患者越来越多地寻求能够解读个人健康记录并提供医疗健康指导的基于大语言模型的系统。然而，现有方法主要依赖于通用医学知识，往往无法考虑个体差异，限制了其提供个性化指导的能力。为解决这一问题，我们提出了个性化提示学习（PPL），这是一个通过学习个性化提示来引导大语言模型生成个性化医疗健康建议的框架。PPL利用患者自身信息以及从临床相似病例中提取的同伴信息来构建初始个性化提示。随后，通过强化学习（RL）对这些提示进行优化，使生成的回复更好地与为每位患者撰写的医生建议保持一致。

    arXiv:2412.15957v2 Announce Type: replace-cross  Abstract: The rapid development of large language models (LLMs) has transformed many industries, including healthcare. In practice, hospitals and patients increasingly seek LLM-based systems capable of interpreting personal health records and providing healthcare guidance. However, existing approaches mainly rely on general medical knowledge and often fail to account for individual variability, limiting their ability to provide personalized guidance. To address this, we propose personalized prompt learning (PPL), a framework that learns individualized prompts to guide LLMs in generating personalized healthcare recommendations. PPL constructs initial personalized prompts by leveraging both self-informed patient information and peer-informed signals derived from clinically similar cases. These prompts are then refined using reinforcement learning (RL) to better align the generated responses with physician recommendations written for each p
    
[^82]: 弦乐合奏录音中柔和起音的标注

    Annotation of Soft Onsets in String Ensemble Recordings

    [https://arxiv.org/abs/2211.08848](https://arxiv.org/abs/2211.08848)

    该论文通过研究24名参与者的标注者间一致性并扩展确定最一致标注者的算法，为弦乐合奏录音中柔和起音的标注建立了最佳实践，并发现音乐经验与标注质量和检测性能之间存在正相关关系。

    

    起音检测是指在音频录音中识别音符事件起始点的过程。虽然打击乐起音的检测通常被认为是一个已经解决的问题，但对于最先进的算法而言，弦乐器录音中的柔和起音仍然构成重大挑战。由于缺乏包含专家标注的数据以及关于弦乐器柔和起音标注最佳实践的相关研究，这一问题进一步加剧。为此，我们研究了24名参与者之间的标注者间一致性，扩展了一种用于确定最一致标注者的算法，并比较了人类标注者与最先进起音检测算法的性能。实验结果表明，音乐经验与标注者间一致性以及与自动系统相比的性能之间存在正相关趋势。此外，由变化产生的起音……

    arXiv:2211.08848v2 Announce Type: replace-cross  Abstract: Onset detection is the process of identifying the start points of musical note events within an audio recording. While the detection of percussive onsets is often considered a solved problem, soft onsets-as found in string instrument recordings-still pose a significant challenge for state-of-the-art algorithms. The problem is further exacerbated by a paucity of data containing expert annotations and research related to best practices for curating soft onset annotations for string instruments. To this end, we investigate inter-annotator agreement between 24 participants, extend an algorithm for determining the most consistent annotator, and compare the performance of human annotators and state-of-the-art onset detection algorithms. Experimental results reveal a positive trend between musical experience and both inter-annotator agreement and performance in comparison with automated systems. Additionally, onsets produced by change
    
[^83]: 推荐系统中的多任务学习方法的进展与挑战：综述

    Advances and Challenges of Multi-task Learning Method in Recommender System: A Survey. (arXiv:2305.13843v1 [cs.IR])

    [http://arxiv.org/abs/2305.13843](http://arxiv.org/abs/2305.13843)

    本文综述了多任务学习在推荐系统中的应用，提出了基于多任务学习技术的推荐方法分类，同时探讨了未来发展方向。

    

    多任务学习已在计算机视觉、自然语言处理等领域广泛应用，并取得了良好的性能。近年来，关于多任务学习推荐系统的研究已经涌现，但目前还没有文献总结这些成果。为了弥补这一空白，我们提供了一篇系统文献综述，旨在帮助研究人员和从业者快速了解这个方向的当前进展。在本综述中，我们首先介绍了多任务学习推荐系统的背景和动机。然后，我们根据多任务学习技术的不同阶段，包括任务关系发现、模型架构和优化策略，提供了多任务学习推荐方法的分类。最后，我们就这一领域的应用和未来方向展开讨论。

    Multi-task learning has been widely applied in computational vision, natural language processing and other fields, which has achieved well performance. In recent years, a lot of work about multi-task learning recommender system has been yielded, but there is no previous literature to summarize these works. To bridge this gap, we provide a systematic literature survey about multi-task recommender systems, aiming to help researchers and practitioners quickly understand the current progress in this direction. In this survey, we first introduce the background and the motivation of the multi-task learning-based recommender systems. Then we provide a taxonomy of multi-task learning-based recommendation methods according to the different stages of multi-task learning techniques, which including task relationship discovery, model architecture and optimization strategy. Finally, we raise discussions on the application and promising future directions in this area.
    

