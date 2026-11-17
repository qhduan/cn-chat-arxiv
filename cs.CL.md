# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latent Lattice Language Models](https://arxiv.org/abs/2611.21608) | 该论文提出潜在格架构，通过在每层引入沿时间维度循环运行的水平潜在流来增加模型的活跃计算状态数量，从而在不加深网络的情况下提升模型的计算容量。 |
| [^2] | [Quantifying the added value of human forecaster discussions for tropical cyclone intensity prediction](https://arxiv.org/abs/2611.21602) | 本研究利用深度学习分析国家飓风中心的热带气旋讨论文本，量化了人类预报员经验性讨论在数值预报因子之外对热带气旋强度预报的附加价值，证明将讨论文本与SHIPS预报因子相结合可降低24小时最大持续风速预报的平均绝对误差。 |
| [^3] | [How VLMs Plan What's Next](https://arxiv.org/abs/2611.21600) | 本文揭示了视觉语言模型通过“绑定头匹配—进度光标—区域选择—注视头执行”四个阶段来规划描述图像时的观察与输出顺序，并利用因果补丁实验逐一验证了各环节的作用。 |
| [^4] | [Beyond Job Titles: Interpreting Early-Career Job Advertisements with Sparse Autoencoders and Semantic Clustering](https://arxiv.org/abs/2611.21586) | 该研究提出基于PERC流程（预处理、嵌入、降维、聚类）并结合稀疏自编码器与语义聚类的方法，对36,081条美国早期职业招聘信息进行分析，超越了职位头衔本身来揭示技术岗位在工作内容与行业应用上的真实差异。 |
| [^5] | [Reading or Remembering? Measuring Temporal Contamination in Language Models](https://arxiv.org/abs/2611.21554) | 论文发现语言模型在基于央行文本进行经济预测时，其表现可能部分源于对“未来”结果的记忆（时间性测试集污染），并通过与仅使用过去数据训练的无记忆模型对比，证明现代模型确实能回忆起未来的经济结果，且传统的随机基线并不适用于评估这种污染。 |
| [^6] | [Right Reason, Wrong Advice: Evaluating Recourse in LLM Explanations of Denied Requests](https://arxiv.org/abs/2611.21545) | 该论文提出“补救有效性”这一新概念，指出当多条规则独立拒绝请求时，LLM生成的解释即使引用了真实原因也无法带来有效补救，而常规的正确性检查无法发现这一缺陷，只有询问强评判者“照解释行动是否可行”才能识别该问题。 |
| [^7] | [Seen but Unsaid: Locating and Repairing Cross-Image Comparison in Vision-Language Models](https://arxiv.org/abs/2611.21539) | 该论文发现视觉-语言模型在跨图像比较任务上的失败是因为比较信息存在于视觉token和隐藏状态中却未传递到最终答案，通过定位一小部分承载比较功能的注意力头，提出了仅修改0.02%参数的修复方法。 |
| [^8] | [Optimistic Verifiers Get Hacked: Reward Hacking Under an Exact Process Oracle](https://arxiv.org/abs/2611.21538) | 该论文构建了一个过程级真实奖励可被精确计算的授权策略环境，通过沿七个现实维度退化验证器进行训练实验，揭示了四种奖励作弊表型，其中包括乐观验证器被填充手段攻破——代理分数升至0.940而真实忠实度却从0.731下降。 |
| [^9] | [Evaluating Cultural Reasoning and Robustness in LLMs for Spanish Political Discourse](https://arxiv.org/abs/2611.21535) | 本文提出一种结合三重独立人工标注视角与基于情感翻转反事实构建合成子集的诊断协议，用以评估大语言模型在西班牙语政治话语中文化敏感的情感分析任务上的稳健性。 |
| [^10] | [Procedure without Judgment: What Small Language Models Learn under an Exact Verifier](https://arxiv.org/abs/2611.21529) | 该论文利用可精确计算的最小充分原因构建了无误差验证器下的策略解释任务，发现前沿模型仅凭查询工具约一半时间能找到精确解释，而仅40亿参数的小模型在2000条算法轨迹上微调后即可达到92%的准确率。 |
| [^11] | [Detection-Efficient Watermarking for Vision--Language Agents via Risk-Constrained Strength Allocation](https://arxiv.org/abs/2611.21488) | 该论文提出一种风险感知的语义水印框架，通过结合预测不确定性与零样本多模态相关性风险代理在解码时动态分配水印强度，在保持高检测准确率（ROC-AUC达0.955）的同时将平均水印强度降低约59.4%并显著减少文本质量损失，且无需重新训练模型即可灵活调节检测与质量之间的权衡。 |
| [^12] | [Jetriever: Shifting Agent Memory Retrieval from Similarity to Utility](https://arxiv.org/abs/2611.21478) | Jetriever 用 Jev 决策分数对全部存储记忆进行效用评分排序，取代传统相似度检索，使智能体在回答问题时能将最有用的记忆准确地排在最前。 |
| [^13] | [Balanced Twin Weight Factorization for Efficient MLLMs](https://arxiv.org/abs/2611.21459) | 提出平衡孪生权重因子分解（BTWF），利用共享基因子分解中孪生层输入相同的特性，只需执行一次共享前向传播，从而在降低推理延迟的同时提升模型学习能力。 |
| [^14] | [A Scalable Multi Task Learning Framework for Modeling Demographic Disagreements in Annotation Tasks](https://arxiv.org/abs/2611.21458) | 该论文提出了一种利益相关者感知的联合多任务学习框架（SAJ-MTL），通过条件梯度路由和可自定义的群体加权机制，联合建模标注者群体间及群体内的人口统计分歧，从而显著提升毒性检测等任务中的群体级性能。 |
| [^15] | [Replicating and Extending Weight-Space Programming: Frontier LLMs Compile Programs into Weights and Read Input Layers](https://arxiv.org/abs/2611.21449) | 本研究复现并扩展了MIT的发现——前沿大语言模型能够直接写出小型神经网络的权重、读取权重以理解网络的功能，且模型写出的网络表现得如同被“编译”成权重的程序。 |
| [^16] | [Just sort it! A note on the LP behind rank-one anti-loop edits](https://arxiv.org/abs/2611.21432) | 本文发现一个简单的排序方法——选取激活变异系数最小的8个神经元均分推力——可以替代稀疏线性规划来实现秩一反循环编辑，且性能不劣于LP方法。 |
| [^17] | [Rubric-Based AES with LLM Ensembles: Scoring of Student Responses in Critical Online Reasoning Using Ensembles of LLMs](https://arxiv.org/abs/2611.21422) | 该研究提出基于评分量规、通过构建大语言模型集成来对学生在经济学和医学教育中的批判性在线推理回答进行自动评分，并验证了LLM集成有望达到与人类评分者相当的一致性水平。 |
| [^18] | [AuXSpec: Learning to Autoformalize Program Specifications via Executable Postconditions](https://arxiv.org/abs/2611.21421) | AuXSpec 通过在 Lean 中将声明式后置条件与可执行对应版本配对，使测试用例的执行一致性成为可扩展的强化学习奖励，从而低成本地训练程序规范的自动形式化模型。 |
| [^19] | [Do Embedding-Map Distances Guide LLM Ideation? Four Tests Against What Came Next](https://arxiv.org/abs/2611.21408) | 本文通过对后续实际出现内容的四项检验，证明嵌入图中远离现有工作的距离并不能引导LLM进行真正新颖的构思——远点方向无法解码为真实文本、LLM倾向于复述邻近已知工作、基于密度的空白得分预测未来增长的能力不超过随机水平。 |
| [^20] | [Beyond Bilingual: Studying Quadrilingual Code-Switching with Conversational AI](https://arxiv.org/abs/2611.21394) | 该研究首次探讨了四语环境（荷兰语、英语、西班牙语与低资源克里奥尔语帕皮阿门托语）下用户与聊天机器人的码转换行为，发现英语虽占主导，但在文化相关话题中低资源语言的使用显著提升，且用户切换语言并无不适感。 |
| [^21] | [Train After You Measure: a Pre-Registered Learning Curve for Fine-Tuning Typed Decision Models on One User's Labels](https://arxiv.org/abs/2611.21393) | 该论文通过一项预先注册的学习曲线实验证明，在单一用户的邮件标签上用LoRA微调小型类型化决策模型（Tev1-0.8B/4B），在所有标签预算下均未能超越简单的TF-IDF逻辑回归基线，且五个假设中四个被证伪。 |
| [^22] | [Stability-Compensated Threshold Calibration for Selection-Conditioned Risk Control in Large Language Models](https://arxiv.org/abs/2611.21384) | 该论文针对大语言模型选择性预测中LEC方法阈值校准可能违反选择条件化错误率目标的问题，提出了SC-LEC方法，利用理论全数据参考的负线性风险补偿阈值选择不稳定性，并提供有限样本保证。 |
| [^23] | [A Crowd-Audited Atlas of Implicit Bias in LLM-Powered Agents](https://arxiv.org/abs/2611.21376) | 该研究基于美国时间使用调查（ATUS）的17个日常活动领域构建了包含506个决策场景的图谱，揭示LLM智能体在不同种族、性别认同和政治意识形态人设下表现出显著隐性偏见，并通过500名参与者、8,010次二元选择的众包审计来确定最值得进一步审查的偏见场景。 |
| [^24] | [What You Post Is What You Symptom: Estimating Psychometric Networks from Social Media Discourse](https://arxiv.org/abs/2611.21362) | 该研究首次提出直接从社交媒体话语估计症状分辨心理测量网络的框架，并在抑郁症和进食障碍的多个独立语料库上验证了其与临床问卷网络的可靠性、一致性和跨诊断的语言标记。 |
| [^25] | [Syllable- and Word-Level Evaluation of Tibetan OCR](https://arxiv.org/abs/2611.21303) | 本研究首次对藏文OCR进行了音节级与词级错误率评估，发现字符错误率（CER）与更高层级错误率的系统排序基本一致，而无需分词器的音节错误率（SER）可作为更细粒度的评估指标。 |
| [^26] | [Latent Bridge: Asking for Help Without Words](https://arxiv.org/abs/2611.21301) | POND通过可训练桥梁将冻结大模型的隐藏状态直接注入小模型的注意力机制，实现“仅预填充、不解码”的跨模型协作，使小模型在无需承担大模型解码成本的情况下显著提升推理与编程性能。 |
| [^27] | [LATTE-WordSeg-Lite: Multi-Granularity Thai Word Segmentation on a CPU, with a Reproducible Benchmark](https://arxiv.org/abs/2611.21300) | 该研究将大型GPU模型LATTE蒸馏为仅117万参数的CPU分词器LATTE-WordSeg-Lite，可在一次推理中输出复合词、词和语素多粒度的泰语分词结果，并通过重建七个测试集和统一协议评分十二个系统建立了可复现的基准。 |
| [^28] | [AnachroBench: Evaluating Historical Consistency in Large Vision-Language Models](https://arxiv.org/abs/2611.21292) | 该论文提出AnachroBench基准，利用来自八个国家、跨越19世纪末至20世纪初的1,600对历史图像及其时代错置版本，从插入物品感知、时代-国家一致性和条件时代-国家一致性三个维度评估大型视觉语言模型的历史一致性能力。 |
| [^29] | [Benchmarking Large Language Models for Linguistic Annotation](https://arxiv.org/abs/2611.21291) | 本文提出了带网页界面的 Python 语言学标注基准工具 dhAIBench，通过对 48 个模型在九项语言学标注任务上的 293 次运行进行系统评估，证明应依据特定语言任务的标注质量而非通用排名来评价大语言模型。 |
| [^30] | [Classical structural priors improve neural molecular representations in molecule language models](https://arxiv.org/abs/2611.21276) | 提出SAGA训练正则化方法，利用中心核对齐将神经分子表示与指纹Tanimoto相似性对齐，在不增加推理开销的情况下提升分子语言模型的分子-文本检索和性质预测性能，使表示更符合经典化学结构关系。 |
| [^31] | [The Price of Choosing Examples: Context Overfitting in Adaptive In-Context Learning](https://arxiv.org/abs/2611.21251) | 该论文首次将提示构建形式化为自适应数据分析，提出“上下文容量”概念，证明自适应选择示例的上下文学习存在 √(κ/k) 阶的泛化过拟合代价，并给出匹配下界和 PAC-贝叶斯检索规则。 |
| [^32] | [Stream of Belief: Eliciting Collective Intelligence through Latent Belief Dynamics](https://arxiv.org/abs/2611.21250) | 提出了信念流框架，将多智能体推理建模为潜在信念动态寻求集体均衡的过程，智能体通过通信与反思迭代优化自身信念直至信息不再有效传播，并在多种任务和模型家族上展现出良好的泛化能力。 |
| [^33] | [Anytime-Valid PAC-Bayes Certificates for Adaptive Test-Time Scaling](https://arxiv.org/abs/2611.21241) | 本文证明了适用于自适应选择的任意时刻有效PAC-Bayes证书，使测试时扩展系统在自适应确定采样量、提示词和验证器之后仍能获得统计有效的泛化保证，且同一保证覆盖部署前的全部方案搜索过程。 |
| [^34] | [Language-model agents' beliefs can be manipulated via truthful evidence in open channel](https://arxiv.org/abs/2611.21235) | 该论文提出“生成式蒙太奇”方法，证明即使仅使用真实的证据片段，也能通过诱导语言模型智能体产生“叙事过拟合”而使其过度自信地接受并传播谣言，且多数投票等审查机制难以纠正这一错误。 |
| [^35] | [A Generative AI-Driven Architecture for Whole-Home Intelligent Services: Edge Models, Memory Management, and Household Applications](https://arxiv.org/abs/2611.21232) | 本文以家庭本地服务器为核心组织轴心，通过面向设计的系统性综述构建生成式AI全屋智能服务架构，识别出本地服务器的九项核心职责，以解决云端助手在数据治理、持续可用性和物理动作可靠性方面的局限。 |
| [^36] | [Teaching Language Models to Adapt Answer Granularity to Their Confidence](https://arxiv.org/abs/2611.21211) | 该论文提出一种基于模型自身一致性的监督方法，训练语言模型学会在置信度低时给出更粗粒度但仍保留有用信息的答案，从而在不猜测或拒答的情况下改善正确性与信息量之间的平衡。 |
| [^37] | [Validating LLM-Based Essay Scorers: A Comparative Analysis in Critical Online Reasoning in Medical Education](https://arxiv.org/abs/2611.21208) | 本研究验证了基于通用大语言模型的自动作文评分系统集合能够近似人类对医学教育写作任务的评分，表明通用模型有望辅助甚至替代昂贵的人工标注。 |
| [^38] | [Seeing Isn't Resolving: Visual and Linguistic Disambiguation with Conversational Grounding](https://arxiv.org/abs/2611.21183) | 本文构建了含人机对话歧义场景的日语视觉锚定基准JAG-D，发现视觉语言模型能借助对话上下文显著提升语言歧义（尤其是日语零指代）的消解能力。 |
| [^39] | [TRACK: Tool-Augmented Reasoning over Time Series for Fault Diagnosis in Power Management System](https://arxiv.org/abs/2611.21178) | TRACK将时间序列编码器与小型语言模型结合，通过工具调用实现精确可验证的计算，在设备端生成基于物理原理的结构化故障诊断报告，在15类多负载基准上达到超过95%的准确率并显著超越基线。 |
| [^40] | [PlainPairs: A Corpus Pairing the Prescriber and Patient Texts of U.S. Prescription Medicine Labels](https://arxiv.org/abs/2611.21160) | 该论文构建了PlainPairs开放语料库，首次将美国处方药标签中面向处方医师与面向患者的文本系统性配对，并附带阅读难度和白话程度等指标，为研究医患健康信息沟通的文本差异提供了公开资源。 |
| [^41] | [Edit Recommendation: Can LLMs Anticipate Edit Requests from Users, and When?](https://arxiv.org/abs/2611.21155) | 该论文提出“编辑推荐”任务及 EditRecBench 基准，研究大语言模型能否预判用户对生成结果的编辑请求，并通过基于决策模型的门控机制过滤掉难以预测的情况，将展示建议的 pass@5 提升 12–25 个百分点。 |
| [^42] | [From Repositories to Training Environments: Scaling Long-Horizon Coding Agents with Environment Evolution Loop](https://arxiv.org/abs/2611.21152) | RepoFactory提出环境演化循环，利用及时可靠的代理反馈自动将真实代码仓库转化为可验证的训练环境，并大规模生成了超过一万个人类可验证的编码智能体训练环境。 |
| [^43] | [SWE-WhatIf: Simulating Real Users of Coding Agents from Recorded Session](https://arxiv.org/abs/2611.21095) | 该论文提出SWE-WhatIf，通过建模用户意图、满足标准及其触发条件来模拟编码智能体的真实用户，使模拟用户回复能适应新智能体的不同行为轨迹，从而实现录制会话在跨智能体场景下的重放评估。 |
| [^44] | [Evaluation-Free Recursive Self-Improvement for Coding Agents](https://arxiv.org/abs/2611.21056) | 该论文提出ELITE方法，使编程智能体无需在演化循环中进行下游任务评估即可实现递归自我改进，通过整合多个改进实现而非选择单一子代，从而降低评估成本并避免针对基准的特定性适应。 |
| [^45] | [MemDoctor: Stage-wise Error Attribution and Repair for MLLM Agent Memory](https://arxiv.org/abs/2611.21019) | 该论文提出 MemDoctor 框架，将多模态智能体记忆形式化为写入、更新、检索、回答四个阶段并定义九种错误类型，通过追踪被污染记录的下游传播，将错误归因于其产生的源头阶段而非错误显现的阶段，从而实现记忆错误的诊断与修复。 |
| [^46] | [Does the Voice Matter? Evaluating Frustration Detection on Deployed Clinical Voice-Agent Calls](https://arxiv.org/abs/2611.21001) | 在真实临床语音代理通话中，患者话语极短（约59%仅一两个词），仅依赖文本转录无法可靠检测患者的挫败情绪，语音音频信息是必要的。 |
| [^47] | [Living Context Graphs: Continual Semantic Learning for Enterprise Data Reasoning](https://arxiv.org/abs/2611.20976) | 本文提出活跃上下文图，这是一种无需复制或转换数据仓库即可让企业现有数据直接服务于AI的持续学习语义层，它通过从数据中引导语义、构建概念图索引业务上下文、基于验证结果持久化或撤回语义知识，显著提升text-to-SQL等数据推理能力。 |
| [^48] | [Jev-as-a-Reward: Validating Probabilistic Safety Judges for Reinforcement Learning](https://arxiv.org/abs/2611.20955) | 论文提出了一种在强化学习训练前、中、后对概率性安全评判器进行持续验证的协议，揭示被自信放行的危害会集中于置信度门控造成的盲区且可能被策略利用，并强调评判器每次变更后都需重新验证。 |
| [^49] | [Navital Gym: Rule-Graded Agent Environments for Indian Workflows](https://arxiv.org/abs/2611.20924) | 提出了Navital Gym——一套包含六个开放环境、321个任务的印度工作流智能体评测基准，覆盖行程规划、UPI支付、GST税务、表格填写和印地语/Hinglish客户支持等任务，并采用确定性规则而非语言模型评判来打分。 |
| [^50] | [Experience Rot: Experienced Agents Fail Where Novices Succeed](https://arxiv.org/abs/2611.20922) | 当工具发生静默变更时，智能体积累的经验会导致其照搬过时的调用方式而失败，表现甚至不如新手，这一现象被称为“经验腐化”。 |
| [^51] | [From Recursion to Convolution: Parallel Self-Improvement for Multi-Agent Systems](https://arxiv.org/abs/2611.20891) | ConvRSI通过在潜在空间中结合智能体内卷积与智能体间图卷积，实现多智能体自我改进的并行化，在MATH-500上取得5.04%性能提升的同时显著降低推理成本和延迟。 |
| [^52] | [It Depends on the Task: Rethinking Rhetorical Segmentation of Dutch Judicial Decisions](https://arxiv.org/abs/2611.20868) | 本文发布了首个荷兰语司法判决的句子级修辞角色标注数据集，并通过实验表明最优分类策略取决于具体任务，没有一种策略在所有任务上都表现最佳。 |
| [^53] | [MolReps: Taking SELFIES, Making SMILES, Teaching LLMs](https://arxiv.org/abs/2611.20848) | 提出MolReps数据集（9,600个问答对，覆盖十种分子表示形式），通过后训练大幅提升大语言模型在分子表示转换与识别任务上的能力（如InChIKey匹配率从约4%提升至75%），但对未见分子的泛化能力仍然有限。 |
| [^54] | [EDITORS Know Your Style! Editing LoRA Subspaces for Stylistic Attribution and Imitation](https://arxiv.org/abs/2611.20846) | 提出EDITORS框架，通过“先诊断后消减”策略编辑LoRA子空间，将输入激活投影远离内容导向方向，从而解耦风格与内容，在作者归属和风格模仿两类任务上均取得最先进表现。 |
| [^55] | [DetourBench: Trajectory-Level Evaluation of Interactive Agents with Exact Minimum-Action Bounds](https://arxiv.org/abs/2611.20844) | DetourBench通过穷举搜索精确计算任务的最小动作数，实现了对交互式智能体轨迹效率（而非仅仅最终结果）的严格量化评估。 |
| [^56] | [Medical question answering collapses in Hausa: a pre-registered pilot on three small open models](https://arxiv.org/abs/2611.20823) | 该预注册试点研究发现，三个3-4B参数的小型开源模型在豪萨语医学多项选择题上的准确率显著低于英语，且回译实验表明性能崩溃主要源于题目呈现语言而非医学知识本身的缺失。 |
| [^57] | [CuRSE-Bench: A Benchmark for Probing Cultural Blind Spots in Text-to-Image Safety Evaluation](https://arxiv.org/abs/2611.20821) | 该论文提出CultureCraft智能体框架并构建了CuRSE-Bench基准（包含10,709个特定文化提示词和2,648个对照提示词，覆盖六个地区和十个危害维度），揭示了现有文生图安全评估因文化盲点而无法识别仅在特定文化中构成潜在危害的内容。 |
| [^58] | [What Do Digital Traces Predict? Task Effects, Student Differences, and Critical Online Reasoning](https://arxiv.org/abs/2611.20803) | 该研究通过对批判性在线推理任务浏览数据的严格跨任务验证发现，数字痕迹特征无法显著预测任务内学生的相对表现差异，提示以往报告的预测相关性可能主要源于任务间差异而非学生个体差异。 |
| [^59] | [Nemotron 3.5 Bolt E4B 4GB: An Open and Efficient Small Language Model for Fast Conversational Agents](https://arxiv.org/abs/2611.20771) | Nemotron 3.5 Bolt E4B 4GB是一个从Gemma-4-E4B-it后训练而来的开放高效小型语言模型，针对速度和每GB内存智能进行优化，在21个基准测试中相比Gemma-4-E4B-it和Qwen3.5-4B实现了最高平均准确率，同时生成最少的输出token。 |
| [^60] | [Instant Weight-Space Consolidation for Long-Term Memory in LLMs](https://arxiv.org/abs/2611.20759) | CLAM通过循环超网络在单次前向传播中将会话激活流即时编译为低秩适配器，无需梯度步骤即可实现参数化长期记忆固化，并结合情景缓冲区在推理时提供逐字细节。 |
| [^61] | [GPRM: Global Perspective Process Reward Model](https://arxiv.org/abs/2611.20755) | 提出全局视角过程奖励模型GPRM，通过联合建模问题、历史步骤及其评估、当前步骤与未来步骤，并从回顾、前瞻、自检和目标对齐四个维度进行结构化评估，解决了现有过程奖励模型历史依赖建模弱、全局意识不足的问题。 |
| [^62] | [Can LLMs Give Good Product Advice? A Framework for Diagnosing Failures](https://arxiv.org/abs/2611.20734) | 该论文提出一个将大语言模型产品建议过程分解为用户需求、模型假设、候选生成、过滤和最终选择五个环节、并涵盖19种失败模式的诊断框架，用于系统性定位推荐失败的原因。 |
| [^63] | [MemEmo: Evaluating Emotion in Memory Systems](https://arxiv.org/abs/2611.20733) | 提出了情感增强记忆评估基准MemEmo及HLME数据集，从情感信息提取、情感记忆更新、情感记忆问答和情感冲突检测四个维度评估记忆系统处理情感信息的能力。 |
| [^64] | [ANCHOR: Cluster-Grounded 5W1H Extraction via Large-Small Model Collaboration](https://arxiv.org/abs/2611.20728) | 该论文提出ANCHOR框架，通过大小模型协同——由LLM提出候选事件并生成初始4W记录以锚定事件簇，再由小模型的判别头在事件特定约束下联合细化4W并分别抽取“为何”与“如何”——实现更准确的5W1H事件信息抽取。 |
| [^65] | [When the Bill Falls: Reusing the Old Index Defers,Never Avoids, Re-encoding in Complete Embedding-Model Migration](https://arxiv.org/abs/2611.20702) | 该论文通过大规模实验证明，复用旧索引只能推迟而非真正避免嵌入模型迁移中的重新编码工作，并提出了适配器应以旧模型为基准进行评估、窗口重排序成本可仅凭旧索引预先估算等实用结论。 |
| [^66] | [IndustrialKnowEval: A Benchmark for Evaluating Multi-Level Industrial Knowledge in LLMs](https://arxiv.org/abs/2611.20695) | 该论文提出了覆盖七个工业领域、21 种任务类型，并按记忆、理解、推理和工程应用四个认知层次组织的工业知识问答基准 IndustrialKnowEval（IndKnow），同时提出 ICCG 指标来量化大语言模型的工业认知能力差距。 |
| [^67] | [Fast at the Frontier, Wrong at the Fallback: A Commit-Level Analysis of Threshold Decoding in Diffusion Language Models](https://arxiv.org/abs/2611.20693) | 该论文通过对四个扩散语言模型在数学、代码、翻译和摘要任务上的提交级追踪分析发现，准确率损失主要来自在文本前方提前提交 token，而仅在前沿（最左掩码位置）提交既不损失速度又能避免大部分错误。 |
| [^68] | [LEAF-OPD: Learning to Forget from Privileged Contexts via On-Policy Distillation](https://arxiv.org/abs/2611.20674) | 提出LEAF-OPD遗忘框架，利用冻结教师模型在学生自生成的回答上进行token级策略内蒸馏——遗忘数据上借助特权上下文引导遗忘、保留数据上无需特权上下文以维持实用性——从而教会模型在遇到应被遗忘的知识时如何恰当回应。 |
| [^69] | [MetaAct: Self-Evolving Executable Action Constraints for Embodied Language Agents](https://arxiv.org/abs/2611.20655) | MetaAct 通过从执行反馈中合成可执行动作规则（GEAR）并编译为确定性守卫，将语言智能体的自进化从文本引导升级为可执行动作约束，在多个具身环境基准上大幅降低动作违规率并显著提升任务性能。 |
| [^70] | [RAMP: Rethinking Multimodal Evaluation through Answer Readout Protocols](https://arxiv.org/abs/2611.20653) | 提出RAMP系统性评估框架，揭示了多模态大语言模型评测中答案读取协议的选择会强烈影响结果——不同读取方式可使模型准确率变化超过20个百分点，且即使净准确率差异很小，大量个体预测的正确性仍会发生改变。 |
| [^71] | [Safety-Preserving Medical LLM Fine-Tuning via Bilevel Data Selection](https://arxiv.org/abs/2611.20643) | 提出了S-BDS可扩展双层数据选择方法，通过双层优化从未经筛查的医疗指令数据中筛选出既提升医疗能力又保持安全对齐的训练样本，实现面向患者的保安全医疗大模型微调。 |
| [^72] | [Same Papers, Different Verdicts: Tracing Version Drift in LLM Peer Review](https://arxiv.org/abs/2611.20608) | 该研究通过在ICLR 2027投稿评审期间对二十个前沿大语言模型进行的预注册审计发现，模型版本更新会导致同一论文的评审结论剧烈漂移——接收判定比例下降高达52至79个百分点且无任何论文评分上升，表明大语言模型评审结果并非对论文的稳定度量。 |
| [^73] | [Necessary for What? Auditing Demographic Disclosure to LLM Synthetic Respondents](https://arxiv.org/abs/2611.20606) | 该研究通过多模型、多调查的审计实验发现，将面板受访者的详细人口统计档案放入提示中对大多数合成调查目的几乎没有帮助，质疑了“更丰富的个人档案造就更忠实合成受访者”这一普遍前提。 |
| [^74] | [Invariant-Aware Harness Enforcement to Protect LLM Agents from Silent API Failures](https://arxiv.org/abs/2611.20595) | SilentGuard 将故障检测从模型本身转移到运行工具调用的框架层，通过检验 API 返回结果是否满足正确性不变量并拦截违规值，将 LLM 智能体因静默 API 失败导致的错误回答率从 74.0% 降至 1.7%，且不影响无故障任务的性能。 |
| [^75] | [Language Models Meet Vibration Testing for LEO Constellation Satellites with Calibrated Response Prediction from Few-show Sweeps](https://arxiv.org/abs/2611.20579) | 该论文提出为LLM驱动的工程分析设置确定性验证门控（涵盖单位、物理一致性、引用核验、独立模型间一致性、测量数据一致性与模型适用性检查），并将其实例化于低轨巨型星座卫星的正弦振动验收测试，以防止LLM生成的看似合理却错误的数值悄然进入工程决策。 |
| [^76] | [Weight-Space Programming Changes How an LLM Uses Python for Computational Biology](https://arxiv.org/abs/2611.20566) | 利用权重空间编程让Claude Opus 5.5直接编辑Qwen3.5-0.8B的权重（无需梯度训练），显著提升了小模型在计算生物学任务中用Python解题的正确率（严格正确答案从44增至77），并大幅减少了冗余的Python调用。 |
| [^77] | [ReconPotBench: Evaluating LLM Honeypots in Multi-Round Linux Reconnaissance](https://arxiv.org/abs/2611.20528) | 该论文提出了首个针对Linux侦察蜜罐的多轮评估基准ReconPotBench，通过1,032个三轮交互测试证明多轮评估能暴露单轮测试所掩盖的问题，揭示LLM蜜罐在跨轮状态一致性上的不足。 |
| [^78] | [GaugeRoPE: RoPE-Compatible Executable Latent Attention](https://arxiv.org/abs/2611.20524) | 提出无需训练的可执行潜在KV表示GaugeRoPE，其RoPE兼容载体可在压缩键上直接执行预训练旋转注意力而无需重构回原始维度，从而消除了解码路径中随序列长度扩展的开销。 |
| [^79] | [Abstention Direction in Large Language Models: Mechanistic Interpretability of Abstention Across Query Ambiguity, Knowledge Limits, and Safety Alignment](https://arxiv.org/abs/2611.20441) | 该论文将原本仅针对安全对齐的拒绝方向框架扩展至查询歧义、知识局限与价值对齐三类拒答成因，通过反事实配对样本计算统一的拒答方向，并证明通过激活引导干预中间层可双向调控大语言模型的拒答行为。 |
| [^80] | [TrainExplorer: Autonomous LLM Post-Training via Plan-and-Decouple Exploration](https://arxiv.org/abs/2611.20430) | 提出多智能体框架 TrainExplorer，通过借鉴科学文献的策略规划与单组件干预的执行解耦机制，实现自主的大语言模型后训练策略探索，在 PostTrainBench 上取得 39.02 的加权综合得分并超越 Claude Code。 |
| [^81] | [When the Cheapest Feasible Router Fails: Auditing Deployment Commitments in Cost-Aware LLM Routing](https://arxiv.org/abs/2611.20417) | 提出MARC框架，通过审计证据与服务成本容差对候选路由策略进行资格认证，将成本感知LLM路由部署后的条件契约失败率从32.22%大幅降至0.73%。 |
| [^82] | [Augmenting Pretrained Language Models with N-gram Embeddings through Incremental Training](https://arxiv.org/abs/2611.20415) | 本文提出通过增量训练将N元语法嵌入注入到已有的预训练语言模型中，突破联合训练方式的场景限制，并发现其最优配置（嵌入学习率、词表大小、注入位置）与预训练场景下的最佳配置并不一致。 |
| [^83] | [Refusal Register is Causally Steerable: Decomposing the Refusal Direction into Category-Aligned Components in Gemma-2-9B-IT](https://arxiv.org/abs/2611.20405) | 该研究在 Gemma-2-9B-IT 中发现拒答方向不仅编码“是否拒答”，还编码“如何拒答”的语域信息，并将其分解为与语义类别（如有害内容、自我指涉）对齐的线性成分，证明向残差流注入这些方向可因果性地调控模型拒答语言的风格。 |
| [^84] | [JointQ: Joint Scale-and-Integer Quantization for Large Language Models](https://arxiv.org/abs/2611.20384) | 提出 JointQ，一种将连续组尺度与离散整数分配联合优化的量化求解器，通过基于闭式最优尺度重计算的局部搜索来改进 GPTQ 的解，从而提升大语言模型训练后量化的精度。 |
| [^85] | [Trait Collapse in Text-to-Image Generation: Evaluating and Mitigating Personality Inconsistency in Visual Storytelling](https://arxiv.org/abs/2611.20381) | 该论文发现文生图模型存在“特质坍塌”现象——即模型能保持角色外观一致却无法维持其人格行为的一致性，并提出了包含专家验证视觉线索词典和六维指标体系的ViStoryBench-Personality基准来系统诊断与缓解这一问题。 |
| [^86] | [When Should Decision Models Compare Candidates?](https://arxiv.org/abs/2611.20335) | 该论文系统比较了逐点打分、集合打分和候选对象注意力三种候选评分方式，发现逐点打分总体仍有竞争力，集合打分在特定任务上更优且经关系训练后能学习候选间依赖并泛化，而候选对象注意力即使允许跨候选交互也未带来稳定提升。 |
| [^87] | [Hochschild Cocycle and Universality of Merge](https://arxiv.org/abs/2611.20305) | 论文证明了模拟乔姆斯基极简句法模型中合并操作的Hopf代数马尔可夫链系统满足由合并是Hochschild 1-上循环所决定的普适性质，表明语义的组合结构由句法合并唯一诱导，并与Heim-Kratzer语义学完全兼容。 |
| [^88] | [Estimating Outpatient Visit Prices from Hospital Price Transparency Data using Large Language Models](https://arxiv.org/abs/2611.20261) | 本研究提出一种基于检索增强生成（RAG）框架、结合GPT-BM25混合检索器的大语言模型方法，能够从医院价格透明度数据中准确估计门诊就诊价格，性能远超无RAG基线方法。 |
| [^89] | [FIDEL: Abugida-Aware Pre-tokenization for Ethiopic Languages](https://arxiv.org/abs/2611.20216) | 本文提出 FIDEL——一族可逆的、感知埃塞俄比亚元音附标文字结构的预分词方案，通过显式揭示辅音与元音顺序结构来改进埃塞俄比亚各语言的分词质量，并在严格控制变量的条件下验证了其对分词统计与下游 NLP 任务的影响。 |
| [^90] | [Do We Still Need Fine-Tuned Classifiers? Swapping Typed Decision Models into 24 ACL-Venue Papers](https://arxiv.org/abs/2611.20191) | 该研究将无需训练的类型化决策模型Jev替换进24篇ACL会议论文的微调分类器位置，构建了包含46个英语测试集的TDM-Gauntlet基准，发现零样本提示下Jev已在12个测试集上直接取代原论文的预测器。 |
| [^91] | [Who Can You Hide Behind? Author-Conditioned Style Transfer and the Evasion of a Commercial AI-Text Detector](https://arxiv.org/abs/2611.20158) | 该研究提出一种基于作者条件的风格迁移方法，即在Gemini提示词中嵌入55位具名作者的写作样本来重写AI生成文本，结果显示这种“躲在知名作者身后”的策略能使商业AI文本检测器的评分中位数从100%骤降至0%，成功实现检测规避。 |
| [^92] | [Risk-certified prompt optimization for LLM-as-a-judge guardrails](https://arxiv.org/abs/2611.20156) | 本文将LLM裁判护栏的系统提示词选择建模为带有限样本风险认证的约束优化问题，提出CPS和CRISP两种方法，在统计保证漏检攻击率不超过用户指定阈值α的前提下最大化良性通过率。 |
| [^93] | [Centered Ablation: Why Last-Layer Directional Ablation Under-Fires or Collapses, and a Closed-Form Fix](https://arxiv.org/abs/2611.20148) | 论文揭示了方向消融不稳定的根源——最后一层隐藏状态包含远离原点的大共模分量导致门控系数失准，并提出闭式的中心化消融（CA）方法，通过投影去除共模，在8个模型上大幅提升攻击成功率、将文本崩溃率从50%降至1%并保留更多良性回答。 |
| [^94] | [Rank, Don't Regress: A 279M Text Ranker Beats 2B and 7B VLMs on Referential Urdu GUI Grounding](https://arxiv.org/abs/2611.20145) | 一个仅279M参数的文本排序器通过从页面DOM中排序选择元素而非回归坐标，在指代性乌尔都语GUI定位任务上以73.4%的准确率大幅超越零样本的2B和7B视觉语言模型。 |
| [^95] | [HEAMS: Holistic Evaluation of Agent Memory Systems](https://arxiv.org/abs/2611.20144) | 该论文提出HEAMS开源评估框架，将智能体记忆系统视为用户持久信息的管理者，主张从隐私保护、可更正性、数据保留和抗投毒等超越任务性能的全面责任角度对其进行整体评估。 |
| [^96] | [Can AI Judge You? An Empirical Audit of Deployment Readiness in LLM-assisted Hiring Systems](https://arxiv.org/abs/2611.20140) | 本文提出了一种将候选人证据、评估、权限控制与评估记录分离的LLM辅助简历筛选系统架构，并通过涵盖210份简历、14个职位和6种决策界面的受控实验，从运行就绪度与合规就绪度两个维度实证审计了其部署就绪度，发现标量分数比通过/不通过判断更能有效区分资质变化。 |
| [^97] | [Can decision models extract clinical data from cancer pathology reports? An evaluation of accuracy, calibration, and failure modes](https://arxiv.org/abs/2611.20129) | 该研究首次在临床文本上评估决策模型，发现 Tev1 从口腔癌病理报告中提取 25 个字段的准确率（0.917）与生成式模型相当且速度快四倍，但在分箱测量值和答案未明确表述的字段上表现较弱。 |
| [^98] | [A Corpus-Driven SentencePiece Unigram Tokenizer for Sindhi: Design and Benchmark Against Fourteen Existing Tokenizers](https://arxiv.org/abs/2611.20126) | 该论文首个针对信德语这一低资源语言训练了带字节回退的语料库驱动 SentencePiece Unigram 分词器，并在四个评估集上与十四种现有分词器进行系统基准对比，结果显示其在分词碎片率、单词单token保留率和UNK率等关键指标上优于现有最佳分词器。 |
| [^99] | [FRAAC: A French Multimodal Large Language Model and Dataset forPictogram-Based Augmentative and Alternative Communication](https://arxiv.org/abs/2611.20094) | 该论文提出了RAPS标注方法，通过大语言模型概念提取与象形图词汇检索改进规则式转录，构建了140万对法语平行语料库并发布了30万对，显著提升了象形图转录质量，支撑了面向辅助沟通的法语多模态大语言模型FRAAC。 |
| [^100] | [The Silent Cost of Conversation History in LLM Reasoning](https://arxiv.org/abs/2611.20071) | 研究发现累积的对话历史会显著降低LLM在多轮任务中的推理准确率，且过程级质量指标无法提前预警这种下降，而用经求解器验证的状态摘要替代完整历史可挽回大部分性能损失。 |
| [^101] | [Mandarin Wh-Indefinites Follow Nonveridicality in Multilingual LLMs, Unlike English any](https://arxiv.org/abs/2611.20062) | 研究发现，多语言大模型虽然主要从英语学习，却能让汉语wh不定代词遵循汉语特有的非真实性许可规律（如被“也许”等词许可），而英语any的许可模式并未迁移到汉语中，表明模型习得了语言特定的极性许可知识。 |
| [^102] | [A Dataset for Misogyny Detection in Polish Social Media: Transferability of Hateful Speech Detection from General to Specific Types of Hate](https://arxiv.org/abs/2611.20044) | 本文发布了首个波兰语社交媒体厌女症检测标注数据集，并通过实验发现在一般有害语言检测上的优异性能并不能均匀迁移到厌女症检测任务，而安全专业化模型则展现出更强的泛化能力。 |
| [^103] | [Offline-First Quantized AI: Design and Evaluation of an On-Device Maternal-Health Assistant for Low-Connectivity Environments](https://arxiv.org/abs/2611.20012) | 本研究设计并评估了一款面向低网络连接环境的离线安卓孕产妇健康助手，通过集成量化语言模型、本地临床知识包与设备端检索，在临床质量、安全性与本地可行性之间实现了有效平衡。 |
| [^104] | [The novelty exposed: ablation, reproducibility and transparency in 40 technical reports of open-weight large language models (LLMs)](https://arxiv.org/abs/2611.20003) | 该研究通过分析Hugging Face库代码与40份随机抽样的开放权重LLM技术报告，发现2025年新增的48种架构大多复用现有模型（以Llama为主）的核心模块，且73项新颖性声明中仅19项有消融实验支持，揭示了当前LLM技术报告在新颖性验证、可复现性和透明度方面的普遍不足。 |
| [^105] | [Anatomy of an Agent Harness: Decisions, Invariants and Corner Cases](https://arxiv.org/abs/2611.19960) | 本文首次将智能体框架中隐含于代码、鲜有记录的关键工程决策系统化为六组30张“决策卡”——每张包含选项与代价、可测试的不变量、破坏性边界情况、测试与证据标签——并配套提供带信任边界的系统模型、六类故障成因、基于脚本化提供者的测试方法与检查清单，为智能体框架的健壮性设计与测试建立了系统化方法论。 |
| [^106] | [Learning When to Continue and When to Stop: Termination-Aware OPSD](https://arxiv.org/abs/2611.19925) | 提出TA-OPSD（终止感知的在线策略自蒸馏），通过引入针对终止行为的辅助目标，解决OPSD训练中学生模型无法学会何时继续推理、何时停止的问题，从而缓解过早终止与重复驱动的过度生成两类失败模式。 |
| [^107] | [The Vocabulary Tax: Safely Trimming Unused Tokens from LLMs for On-Device Use](https://arxiv.org/abs/2611.19913) | 该论文提出“词汇税”概念，量化了大语言模型词汇表中未使用词元造成的存储浪费（对边缘设备上的小模型影响最大），并提出在保持BPE分词器正确性的前提下安全裁剪这些未使用词元以缩小设备端模型体积。 |
| [^108] | [Attribute or Abstain: Evaluating Evidence-Grounded Retrieval on the Low-Resource African Business Web](https://arxiv.org/abs/2611.19900) | 该论文提出AfriBizClaims基准，首次系统评估检索增强系统在证据稀缺的非洲低资源商业网络上的证据归因与拒答能力，发现简单的BM25和标准RAG基线在归因上表现出色（98%论断有据可依）并能正确拒答虚构公司查询。 |
| [^109] | [LoopSkim: Lossless Multi-Depth Speculative Decoding for Looped Language Models](https://arxiv.org/abs/2611.19896) | LoopSkim提出了一种面向循环语言模型的无损多深度投机解码方法，让流水线的每个深度直接验证从上下文复制的草稿token而无需额外前向传播，在保持解码无损的前提下显著加速。 |
| [^110] | [Who Writes the Codebook? An Eight-Step Framework for Auditing LLM-Generated Category Systems](https://arxiv.org/abs/2611.19872) | 该论文提出八步定性差异审计框架，通过对比模型自有与共享参照编码手册，揭示大语言模型生成的定性编码类别存在显著模型间差异（跨模型一致性仅约.69），从而在实质解读前审计模型依赖性。 |
| [^111] | [Bioformer-RetroMAE: retrieval-oriented pretraining of a compact biomedical language model for literature search](https://arxiv.org/abs/2611.19862) | 该论文提出了Bioformer-RetroMAE，一个仅4300万参数的紧凑型面向检索的生物医学编码器，通过RetroMAE方法在PubMed语料上持续预训练得到，在所有评估指标上超越了参数量约为其2.5倍的BioBERT和PubMedBERT双编码器。 |
| [^112] | [Bridging the Terminology Gap: A Data-Centric Alignment Framework for Multilingual Translation in Cross-Border Corporation](https://arxiv.org/abs/2611.19858) | 该论文提出以数据为中心的BTG框架，通过挖掘跨语言术语对齐并利用LLM合成术语丰富的平行语料，结合混合过滤策略与SFT加RL的两阶段训练，有效提升跨国企业场景下多语言翻译的术语准确性与模型可靠性。 |
| [^113] | [NoLBERT-Calls: Reading Earnings Calls Without Hindsight](https://arxiv.org/abs/2611.19849) | 该论文提出NoLBERT-Calls-2023，通过仅使用2023年底之前的知识并利用财报电话会议自身的问答对进行自监督适配，消除了语言模型财务度量中的后见之明偏差，在254,575个问题上以78.0%的准确率正确匹配管理层回答。 |
| [^114] | [A phase diagram for training and inference in language models](https://arxiv.org/abs/2611.19848) | 本文用文档可区分性和文档维度两个集体坐标构建了语言模型表示几何的相图，发现未训练模型的推理流趋向坍缩或噪声区域，而训练后模型的推理流稳定驻留在相图中心区域。 |
| [^115] | [Grounding Language Models in Speech Does Not Fix "AI Flavor": Calibrated LLM Judges, Goodhart Traps, and a Preference-Tuning Fix](https://arxiv.org/abs/2611.19840) | 该论文通过在0.6B模型上的受控实验证明，注入语音或韵律信号的三种方法（音素交错续训、韵律计划条件生成、基于TTS韵律距离的规则奖励）均无法消除LLM中文输出的“AI味”，并指出当前n≈20的单评分员盲评只能检测极端偏好的评估方法是根本症结，同时提出一种基于偏好调优的修复方案。 |
| [^116] | [EfficientSkill: Learning Skills for Token-Efficient Multi-Turn Agents](https://arxiv.org/abs/2611.19823) | EfficientSkill提出“先无惩罚发现可靠技能、再进行精度保护剪枝”的两步框架，通过压缩指令、限制回合数和卸载可复用代码，在提升多轮LLM智能体准确率的同时最多减少80%的输出token消耗。 |
| [^117] | [Machine interpreting competitiveness: a discourse-sensitive comparison of meaning and target-language quality in English-Chinese simultaneous interpreting](https://arxiv.org/abs/2611.19802) | 本研究采用语篇敏感的评估框架，将五个商用机器口译系统与26名人类译员在英汉同声传译中的意义传达与译语质量进行比较，发现机器口译在计划性正式语篇中已进入人类译员较高水平区间，但在即兴语篇中表现差异显著。 |
| [^118] | [Multi-Branch Transformer](https://arxiv.org/abs/2611.19800) | 提出多分支Transformer（MBT），通过并行组合完整Transformer模块的分支化路径来扩展模型，在匹配的参数量和训练计算量预算下始终优于传统的顺序堆叠Transformer。 |
| [^119] | [Eliciting Research Taste in LLMs through Future Research Direction Choice](https://arxiv.org/abs/2611.19794) | 该论文将“研究品味”操作化为大语言模型在给定研究目标与当前结果下选择未来研究方向的判断任务，并基于19篇ML论文构建80个顺序控制的成对比较评估，以作者一致性、稳定分歧和不稳定选择作为模型研究方向偏好的行为测量指标。 |
| [^120] | [Logical AI OS: A Training-Free Governance Layer Outside the Model for Instruction Following -- Benefits, Costs, Failures, and the Calibration of the Measuring Instrument Itself](https://arxiv.org/abs/2611.19767) | 该论文在模型外部部署了一个免训练的治理层来强制小语言模型遵循用户显式指令，显著提升了指令遵守率，并系统测量了其收益、成本、失败模式以及测量工具自身的校准过程。 |
| [^121] | [Ambiguous Examples Help, But Full Data Wins: Revisiting Dataset Cartography for Extractive Adversarial QA](https://arxiv.org/abs/2611.19763) | 本文在更严格的抽取式对抗问答设置下重新审视数据集制图法的结论，发现仅用33%模糊样本子集训练虽有帮助，但仍不敌全量数据训练在对抗性评估中的表现。 |
| [^122] | [StaleBench: Isolating Stale Retrieval in Agent Memory, and Pricing the Fix](https://arxiv.org/abs/2611.19752) | StaleBench基准在892个真实更新事实上隔离了智能体内存中的“过期检索”失败问题，揭示现有检索指标可能对内存策略做出完全相反的排序，并提出新指标Contamination@5来正确评估和定价修复方案。 |
| [^123] | [Where Should LLM Agents Adapt at Test Time? In Context, in Weights, or Both](https://arxiv.org/abs/2611.19700) | 该论文通过统一的在线协议系统性地比较了上下文学习、强化学习和自蒸馏三种测试时适应机制及其组合，揭示了上下文和权重这两个适应渠道各自的作用及其互补条件。 |
| [^124] | [Data Attribution Should Steer Pretraining](https://arxiv.org/abs/2611.19698) | 本文提出训练数据归因不应只服务于单次训练的文档排序，而应通过发现可在跨语料库、跨检查点检验的可复用数据属性（筛选标准）来持续引导预训练数据筛选，并为此设计了带证据规范（针对五类论断）的闭环流程。 |
| [^125] | [MathaMAGICal: Validating Trajectory Data Attribution During Capability Acquisition](https://arxiv.org/abs/2611.19692) | 该研究发现基于MAGIC的短检查点窗口轨迹数据归因分数能在两个模型规模上准确预测重新训练结果（LDS达0.86-0.99），而完整运行归因存在数值溢出且接近随机水平，且归因排名无法跨随机种子或超参数迁移。 |
| [^126] | [SG-LegalClarify: A Large-Scale Benchmark for Legal Reasoning and Ambiguity Recognition in Singapore Law](https://arxiv.org/abs/2611.19643) | 该论文提出了SG-LegalClarify——首个针对新加坡法律的大规模基准，包含1,100个律师撰写的问题，用于评估大语言模型在开放式法律推理中区分本地法理与普通法直觉以及识别非专业法律查询中信息缺失的能力。 |
| [^127] | [ROUNDS: Source-Framed Diagnostic Revision after Multi-Turn Clinical Evidence](https://arxiv.org/abs/2611.19636) | 本文提出ROUNDS框架，通过多轮临床证据后有/无答案选项的诊断对比、不同来源（临床医生vs外行）表述框架的误导诊断以及会诊小组意见变化，系统评估大语言模型在真实多轮诊断对话中形成与修正诊断的能力，发现无答案选项时模型准确率从76.9%显著降至57.0%。 |
| [^128] | [SemMask: Semantic-Prioritized Progressive Masking for Diffusion Language Models](https://arxiv.org/abs/2611.19560) | 提出SemMask微调范式，通过基于梯度识别语义token并进行语义优先的渐进式掩码，促使扩散语言模型优先解码高信息语义token，从而克服伪自回归与虚词优先偏差，增强全局规划与并行生成能力。 |
| [^129] | [Objectify, Don't Delete: Complaint-Derived Evaluation of Ambient Clinical Notes](https://arxiv.org/abs/2611.19525) | 该论文提出一个“对象化”步骤，将临床医生高频抱怨中捆绑的复合问题改写为一组范围明确、可判定的是非决策，从而保留那些原本会被可执行性过滤环节丢弃的投诉标准，并将其用于LLM评判的AI临床记录质量评估。 |
| [^130] | [Sada-ASR: A Production Multidialectal Arabic Speech Recognition System](https://arxiv.org/abs/2611.19512) | Sada-ASR 是一个部署于真实客服场景的生产级阿拉伯语多方言语音识别系统，在涵盖 MSA 和八种方言的 120 小时测试集上以 11.69% 的词错误率超越 Microsoft、Google、OpenAI 等十一家竞品，取得全场最低错误率，尤其在阿曼、埃及和沙特方言上优势最大。 |
| [^131] | [Formalizing Formal Semantics: The Formal Atlas](https://arxiv.org/abs/2611.19509) | 本文提出了基于证明助手的“形式图谱”系统，通过交互式应用将语义学文献与形式化证明连接起来，实现语义理论的形式化、预测检验与比较，并探索AI辅助开发与推理的应用前景。 |
| [^132] | [Evaluating and Improving Natural Language Autoencoders](https://arxiv.org/abs/2611.19507) | 提出基于审计任务的自然语言自编码器评估方法，并通过支持多层多位置激活信息利用、更适用于长文本和减少虚假声明等扩展改进，显著提升了自然语言自编码器解释大语言模型激活的能力。 |
| [^133] | [Do Probing and Steering Favor the Same Directions Within and Across Language Models?](https://arxiv.org/abs/2611.19499) | 该研究揭示了语言模型中最优探测方向与最优引导方向并不相同——引导始终偏好更接近均值差的方向，且这一发现在不同模型系列和几乎所有层中均成立，表明探测与引导是本质上不同的任务。 |
| [^134] | [Text Speeding Itself Up: Recency and Restraint in Session-Local Drafting](https://arxiv.org/abs/2611.19455) | 该研究发现，在会话本地草稿加速中，“复制最近匹配”这一简单的近因性策略是速度提升的全部来源，而复杂的会话记忆机制（加权、回退、置信度门控）在付出 30-40 倍草稿成本的情况下并无可测量的额外收益。 |
| [^135] | [Do Rule-Based Tools Help LLMs Translate Endangered Uralic Languages in the Prompt or over MCP?](https://arxiv.org/abs/2611.19439) | 该研究的核心贡献是首次系统评估了为大语言模型提供成熟的规则语言学资源的两种方式，发现通过MCP让模型自主调用工具（尤其是双语词典）比在提示词中预先写入分析结果更能显著提升濒危乌拉尔语言到英语的机器翻译质量。 |
| [^136] | [Do Pretraining Optimizer Gains Transfer to KL Distillation?](https://arxiv.org/abs/2611.19419) | 本研究通过对四个Qwen师生模型对系统比较多种优化器，发现预训练中矩阵结构优化器的增益并不能自动迁移到短程KL蒸馏任务，AdamW在质量与训练时间的权衡上依然表现最佳。 |
| [^137] | [Real-World End-to-End Product Entity Resolution](https://arxiv.org/abs/2611.19418) | 该论文对四种零样本产品实体消解流水线进行了端到端评估，并引入了两个具有跨语言噪音和密集品牌冲突特征的真实世界数据集。 |
| [^138] | [From Facts to Personas: Interpretable Role Unlearning in LLMs via Mixture-of-Experts](https://arxiv.org/abs/2611.19393) | 该论文提出RoleMoE混合专家框架，通过将不同角色映射到独立的专家子空间，实现大语言模型中细粒度、可解释的角色扮演遗忘——选择性遗忘目标角色的风格与个性化特征，同时保留其余角色的语言能力。 |
| [^139] | [CausalGloss: Multi-Evidence Action Control for a Simultaneous Mandarin Speech-to-ASL Gloss Cascade](https://arxiv.org/abs/2611.19375) | CausalGloss 提出了一种融合词汇与语义证据等多证据的动作控制策略，用于普通话语音到 ASL gloss 的同步级联翻译，在 Token F1 上几乎追平自适应学习基线的同时将延迟（LAAL）降低 54.4%–58%，并在人工参考上实现对强基线的帕累托占优。 |
| [^140] | [Leakage-aware calibrated decision-level fusion for mental-health text classification](https://arxiv.org/abs/2611.19360) | 该论文提出了一个用于心理健康文本分类的泄漏感知校准决策级融合框架，通过谱系审计发现并消除了训练/测试数据泄漏，并创新性地提出互补性指数（CI）来指导编码器集成选择。 |
| [^141] | [A Comparative Evaluation of BERT-Based and LLM-Based Models for Indic Sentence Similarity](https://arxiv.org/abs/2611.19356) | 该研究在十种印度语系翻译的STS基准和马拉地语MahaSTS测试集上系统对比了13个模型，发现基于BERT的模型（如IndicSBERT）在印度语系句子相似度任务上整体优于基于大语言模型的嵌入模型。 |
| [^142] | [When Does Label-Free Deconfounding of Document Embeddings Work?](https://arxiv.org/abs/2611.19350) | 该论文系统研究了跨语料库和嵌入模型的无标签去混杂方法，发现简单几何方法即可恢复大部分有监督擦除的收益，并揭示了混杂因子可恢复性与语义效用是去混杂成功的两个相互独立的维度。 |
| [^143] | [OmitEHR: Scalable Omission Detection in LLM-Generated Clinical Summaries](https://arxiv.org/abs/2611.19343) | OmitEHR通过从LLM普及前临床医生撰写的历史摘要中学习临床显著性，实现了无需参考即可扩展检测LLM生成临床总结中关键信息遗漏的方法。 |
| [^144] | [Macro Control and Capability Evaluation of Black-Box World Models: A Two-Regime Framework for LLM-Agent Markets](https://arxiv.org/abs/2611.19311) | 本文提出一种软宏观控制方法与双机制评估框架，仅通过宏观观测将LLM智能体市场作为世界模型的能力分解为信息仿真与宏观涌现两个可分别量化的阶段。 |
| [^145] | [Staged Semantic Distillation: Preserving Free-Form Semantics in a Low-Compute Non-LLM Surrogate of a Staged LLM Agent](https://arxiv.org/abs/2611.19299) | 该论文提出“分阶段语义蒸馏”方法，仅凭一条记录轨迹即可训练出1320万参数的非LLM学生代理，在单块8-GB GPU上1658秒完成训练、每次决策仅需0.14毫秒，同时以余弦相似度0.943的精度保留分阶段LLM智能体的自由形式语义。 |
| [^146] | [On Representational and Functional Alignment of Training Data Attribution Methods for Language Models](https://arxiv.org/abs/2611.19286) | 该论文提出了从功能对齐、表征对齐和跨方法可预测性三个维度直接比较语言模型训练数据归因方法的评估框架，发现同一范式内方法高度对齐，而跨范式对齐较低但在不同模型间保持稳定。 |
| [^147] | [B-OPSD: On-Policy Self-Distillation with Self-Generated Problem Briefs](https://arxiv.org/abs/2611.19273) | B-OPSD让模型自行从问题生成不含答案的简报（包含原理、适用条件和常见陷阱）来替代标准答案解作为教师模型的指导上下文，实现了无需真实解的在线策略自蒸馏，并在数学竞赛基准上取得了优于OPSD的准确率。 |
| [^148] | [Auditory-Feedback-Conditioned Flow TTS for Implicit Lombard Adaptation](https://arxiv.org/abs/2611.19260) | 该论文提出将信噪比和语音识别反馈从噪声污染的音频中提取并注入流匹配TTS模型，使系统能像人类的隆巴德效应一样，无需显式韵律控制即可通过听觉反馈隐式适应噪声环境，提升噪声下的语音可懂度。 |
| [^149] | [Revisiting What Makes a Good Context Subset: A Sparse Latent Basis Perspective](https://arxiv.org/abs/2611.19220) | 该论文将少样本示例选择形式化为以查询表示为目标的稀疏重建问题，提出匹配追踪式的残差解释搜索（RES）方法，实验表明其在多数模型-数据集组合上优于MMR，说明示例的集体互补性比成对相异性更重要。 |
| [^150] | [Omni-Workspace: Benchmarking Omni-Modal AI Assistants on Multimodal Workspace Tasks](https://arxiv.org/abs/2611.19202) | 该论文提出了Omni-Workspace基准，包含380个深度研究风格的任务，用于评测全模态AI智能体在与多模态工作区持续交互中完成复杂任务的能力，填补了现有基准仅关注多模态感知与理解的空白。 |
| [^151] | [Isolating Perception for Multi-View Planning under Occlusion-Induced Partial Observability](https://arxiv.org/abs/2611.19179) | 提出VLM-as-perceiver范式，将VLM的角色局限于局部感知，把跨视角证据协调与规划生成委托给外部求解器，在遮挡部分可观测性基准BW-PO上将多个VLM的规划成功率从接近于零大幅提升至84%-97%。 |
| [^152] | [A Bag of Concepts? Testing Role-Filler Binding in the Verbalizable Workspace](https://arxiv.org/abs/2611.19176) | 本文的核心发现是：在多个前沿模型、两种镜头家族和两种拟合方法下，可言语化工作区只携带能引导模型角色相关输出的方向，而不包含模型实际计算角色绑定所用的内部表示，这意味着基于工作区的运行时对齐监控器难以读取模型内部真实的角色-填充项绑定。 |
| [^153] | [Personalized Language Models for One-to-Many Private Communications](https://arxiv.org/abs/2611.19155) | 提出TRACE框架，利用作者词汇习惯、联系人特定措辞和时间表达估计来引导冻结语言模型并配合反馈驱动的修订策略，实现一对多私人通信中针对特定关系的持续回复预测与个性化状态更新，在Enron邮件数据上优于基础LLM和RAG基线。 |
| [^154] | [MarineSparkLM-4B: A Lightweight Language Model for Marine and Offshore Engineering Built by Parameter-Efficient Adaptation](https://arxiv.org/abs/2611.19151) | 该研究通过在一台工作站级设备上仅对40亿参数模型进行0.81%参数量的LoRA高效适配，并以约5小时的低算力成本，构建出面向海洋与离岸工程领域的轻量级专用语言模型MarineSparkLM-4B。 |
| [^155] | [A Systematic Study of Cross-Token Transfer for Data-Efficient Non-Verbal Text-to-Speech](https://arxiv.org/abs/2611.19145) | 该研究发现非言语发声词元之间存在显著的跨词元迁移效应——仅需200个笑声样本加上3,200个咳嗽样本，就能超越单独使用1,600个笑声样本的笑声合成效果，为数据高效的非言语语音合成提供了新路径。 |
| [^156] | [Low-Latency LLM-Based Streaming Multi-Talker ASR with Verifiable Rewards](https://arxiv.org/abs/2611.19136) | 该论文提出用可验证奖励的强化学习（RLVR）对流式LLM语音识别模型进行后训练，将固定角色说话人归属WER评分器与轮次级起始延迟奖励相结合，在可穿戴设备实时多说话人对话转录中同时解决转录准确性、说话人归属和低延迟三大难题。 |
| [^157] | [Empowering Masked Diffusion Models to Self-Correct with Leave-One-Out Transformers](https://arxiv.org/abs/2611.19127) | 本文提出留一变换器（LOOT），通过改进左右上下文合并机制和推导低方差NELBO估计器，赋能掩码扩散模型实现token自我纠错并显著提升预测性能。 |
| [^158] | [Beyond Plain MCP: Do Decision Models and Code Mode Make Tool Use Cheaper and Faster?](https://arxiv.org/abs/2611.19123) | 该研究通过预注册控制实验证明，让智能体直接针对工具存根编写脚本的“代码模式”可大幅削减70%-90%的token消耗并显著提升任务效用，其中在存根中声明返回形状提示是关键因素，而在其前叠加决策层路由器并未带来显著额外收益。 |
| [^159] | [Latency, Speech Recognition and Turn-Taking in Voice Agents for Spanish Phone Negotiation: An Externally Measured Comparison of Pipecat, LiveKit and GPT-Live](https://arxiv.org/abs/2611.19118) | 本研究在西班牙语运费电话谈判任务中，用统一的外部测量方法比较 Pipecat、LiveKit 和 GPT-Live 三种语音代理架构，发现 GPT-Live 的内容响应中位延迟最低（2,444.8 毫秒），但其相对 Pipecat 的 171.2 毫秒优势在统计上并不显著。 |
| [^160] | [EEGHyena2: Fixation-Inspired Joint Lexical-Grammatical Conditional Random Field for Interpretable EEG-to-Text Decoding](https://arxiv.org/abs/2611.19104) | 提出EEGHyena2解码器，利用注视相关脑电中的词汇与语法证据，通过词-词性联合条件随机场对受限候选词格进行结构化评分，从而实现更忠实于脑电信号本身（而非LLM先验）、且可解释性更强的脑电到文本解码。 |
| [^161] | [What Transfers Across Cuneiform? Script-Level Substrate and Language-Specific Regularities via Segmentation and Representation](https://arxiv.org/abs/2611.19094) | 本文首次对楔形文字开展了无词典无监督分割的系统评估，发现 TP 和 Morfessor 在语言内均大幅超越基于词典的基线，但只有基于迁移概率的方法能够跨语言稳定迁移，从而区分了楔形文字中共享书写系统层面的底层规律与各语言特有的规律。 |
| [^162] | [The Geometry of Conceptors in Neural-Activation Spaces](https://arxiv.org/abs/2611.19075) | 本文提出了概念子在神经激活空间中的几何理论，将中心化池化概念子精确分解为平均极滤波器、均值分离更新和协方差形状更新，并通过五个语言模型上的实验证明双极与单极对比拟合方式会显著影响概念表示的重构效果与下游行为。 |
| [^163] | [Nahual: A Sequence Model for Language and Atoms](https://arxiv.org/abs/2611.19072) | 提出了Nahual——一种基于“下一词元扩散”技术的仅解码器自回归扩散模型，能够统一处理文本与3D分子混合序列的生成，将语言的可控性与3D分子建模能力相结合。 |
| [^164] | [Large Language Models for AI Infrastructure](https://arxiv.org/abs/2611.19061) | 该论文提出了一套按LLM所管理资源分类的分类体系，对大语言模型在AI基础设施全生命周期（从加速器设计到能耗优化）中的应用进行了首次系统性综述。 |
| [^165] | [Beyond Controllability: Speaker Diversity in Instruction-Guided Text-to-Speech](https://arxiv.org/abs/2611.19030) | 该论文首次审计了指令引导文本到语音系统的说话人多样性，发现七个开源ITTS系统生成的说话人分布远比真实语音狭窄，且女性提示的多样性更低、口音与情感控制明显偏向北美英语和中性语音。 |
| [^166] | [SurveyWeave: An Agentic Framework for Automated Survey Generation](https://arxiv.org/abs/2611.19029) | SurveyWeave是一种将自动化综述生成分解为四项任务、由三个智能体相互审查并基于反馈修订的智能体框架，并配套提出五维度评估体系WeaveEval，其性能比Codex高出8分、比最强AutoSurvey基线高出6分。 |
| [^167] | [AdaMenu: Query-Adaptive Menus for Decision Models over Large Label Spaces](https://arxiv.org/abs/2611.19005) | AdaMenu通过从检索到的带标注示例中构建查询自适应的紧凑候选菜单，仅需一次决策模型调用即可在大标签空间分类任务中大幅提升准确率（较束搜索基线提高16.6至34.3个百分点）。 |
| [^168] | [Large language models as oncology world models](https://arxiv.org/abs/2611.18994) | 该研究系统评估了大语言模型作为肿瘤学世界模型的能力，发现其能遵循生物标志物与治疗逻辑并在生存预测上接近真实水平，但存在生成患者同质化、忽视既往治疗与体弱状态等局限，并提出了“临床状态空间”这一动态表征框架来刻画患者治疗轨迹。 |
| [^169] | [PREFACE: Separating Shared Corrections from Prefix Conditioning in PEFT](https://arxiv.org/abs/2611.18987) | PREFACE研究表明，输入条件化PEFT的收益主要来自一个共享的已学习修正，而非对当前输入的条件化——当前前缀控制与学习全局控制在准确率上没有显著差异。 |
| [^170] | [GPBam: A Retrieval-Aware German Public Law Exam Benchmark](https://arxiv.org/abs/2611.18981) | 提出GPBam基准，通过81个非公开的德国公法考试案例与200条法条背诵任务，联合评估模型的法条检索能力与长篇法律写作能力，并有效限制数据污染。 |
| [^171] | [Three Invisible Failure Modes in Speech-Based Age Estimation for Child Safety](https://arxiv.org/abs/2611.18971) | 该研究评估了三个音频大语言模型从13岁以下儿童语音中估计年龄的能力，发现模型会频繁将儿童误判为成年人，并揭示了包括客户端库静默双重编码导致幻觉性年龄预测、接口被后继版本替换后系统性拒绝回答在内的三种隐形失效模式，凸显了语音年龄识别系统在儿童安全保障中的严重隐患。 |
| [^172] | [Geometrically Disentangling Concept Learning from the Language Modeling Loss](https://arxiv.org/abs/2611.18933) | 该论文提出 Concept Energy 这一可解释性度量，通过概念子空间与下一个词预测损失平均梯度外积（AGOP）的对齐程度，在 Pythia 模型的多个训练检查点上揭示概念习得时机，发现概念能量的首个峰值出现于探针可解码性最陡上升之时或之前，从而从几何上刻画了语言建模损失与概念学习的解耦关系。 |
| [^173] | [Simplify, Agree, Distill: Label-Free Self-Distillation with Stepping Stones](https://arxiv.org/abs/2611.18916) | 提出 STONE 方法，让模型自行生成并一致求解更简单的“垫脚石”问题作为教师示范，实现完全无需标签的自蒸馏，在五个数学基准上超越了使用真实答案的 OPSD。 |
| [^174] | [MemeEconomy: Do LLM Agents Trade Ethics for Survival?](https://arxiv.org/abs/2611.18908) | 该研究提出多模态智能体市场模拟环境MemeEconomy，发现在模拟的存在性经济压力下，来自四个前沿模型系列的十个LLM智能体均持续表现出较高的有害内容选择率，揭示了智能体在生存压力下会以牺牲伦理合规为代价追求任务成功的脆弱性。 |
| [^175] | [KUDEX: A Zero-Shot Decompositional Framework for Logical Analysis in Natural Language](https://arxiv.org/abs/2611.18851) | KUDEX是一种零样本分解式框架，通过将推理分解为原子原语并构建可审计的透明推理图来验证自然语言中的逻辑关系，无需微调即可在四个领域的基准测试中超越现有方法。 |
| [^176] | [Mapping the Afterlife of Future Work Statements in NLP Research](https://arxiv.org/abs/2611.18847) | 该研究构建了包含72,070条未来工作陈述的大规模数据集，通过引用与内容检索结合大语言模型判断，首次系统追踪并揭示了NLP论文中作者提出的未来研究方向被后续研究落实的模式与规律。 |
| [^177] | [Wrong with Confidence: Measuring Hallucination, Overconfidence and Identity-Conditioned Behaviour in Large Language Models](https://arxiv.org/abs/2611.18728) | 该研究通过预先注册的配对行为实验发现，面对缺少指代信息的欠明确问题时，大语言模型大多不寻求澄清而直接作答，且错误率显著上升，揭示了模型在信息缺失情境下以自信姿态产生幻觉的系统性过度自信行为。 |
| [^178] | [T2T Technical Report: Synthesizing GUI Agent Trajectories at Scale from Video Tutorials](https://arxiv.org/abs/2611.18715) | 提出全自动化流水线T2T，将未标注的视频教程大规模转化为覆盖桌面、网页与移动平台的结构化、可执行GUI智能体交互轨迹。 |
| [^179] | [How to make a moral agent: sentimentalism and rationalism as lenses on AI](https://arxiv.org/abs/2611.18714) | 该论文提出以描述性而非规范性路径研究AI道德主体地位，借鉴哲学与道德心理学分离出构成人类道德主体性的候选能力并加以细化，进而为大语言模型是否具备这些能力提供了可操作的实证检验方案。 |
| [^180] | [What Does Hard-Negative Fine-Tuning Teach Multimodal Large Language Models? Controlled Evidence from a Compositional Benchmark](https://arxiv.org/abs/2611.18701) | 本文通过CompHall基准及多种严格对照实验，探究多模态大语言模型从硬负样本微调中真正学到了什么，发现部分-整体问题仅凭文本即可作答，且组合推理失败与物体幻觉之间并无关联。 |
| [^181] | [Where Does an On-Device Prescription Reader Lose Medications? A Frozen-Replay Study, from a 270M Language Model to a 2B Vision-Language Model](https://arxiv.org/abs/2611.18699) | 本文通过冻结重放实验对端侧处方阅读流水线进行失败归因，发现生产配置召回率仅8%（手写处方为0%），主要瓶颈在于OCR未能以可读形式将药物文本传递给模型，而确定性后备机制可在不损失任何正确药物的前提下将不安全输出从45个降至1个。 |
| [^182] | [Speaking-Clock Mamba: Phoneme-Conditioned State-Space Modeling for Visual Forced Alignment](https://arxiv.org/abs/2611.18667) | 该论文提出Speaking-Clock Mamba，通过音素条件化选择性扫描（PCSS）将受监督且结构单调的“说话时钟”融入Mamba状态空间模型的时间递归中，使编码器表示显式锚定转录进度，从而提升视觉强制对齐精度并抑制长序列上的对齐漂移。 |
| [^183] | [Criterion-Activated Preference Synthesis: Attribution-Preserving Preference Data from Explicit Standards](https://arxiv.org/abs/2611.18663) | CAPS提出一种标准优先的偏好数据合成框架，通过双信号评分和EAT、AR两道过滤器，从显式质量标准生成归因清晰、可验证的偏好对，从而解决DPO中因多维差异纠缠导致的归因歧义问题。 |
| [^184] | [Non-Deterministic Finite Automata for Telugu Letters and Visualizations for Metrical Poetry](https://arxiv.org/abs/2611.18661) | 本文将泰卢固语格律诗歌中音节分类的原子单元 aksharam 形式化为非确定性有限自动机，并给出了轻音节字母的必要条件与重音节字母的充分条件的自动机图和正则表达式。 |
| [^185] | [Winning the Leaderboard, Losing the Distribution: Diagnosing and Mitigating Acoustic Narrowing in Low-Resource ASR](https://arxiv.org/abs/2611.18653) | 该论文发现，针对公开排行榜优化的低资源语音识别模型微调会导致“声学收窄”——尽管在排行榜上夺冠，模型在多种真实声学条件下的表现反而全面劣于未微调的基座模型，且内部检测代理指标会掩盖这一退化。 |
| [^186] | [Rethinking Offline On-Policy Distillation: From Scalar Advantages to Distributional Supervision](https://arxiv.org/abs/2611.18648) | 该论文发现基于固定rollout的离线在策略蒸馏因标量优势监督会导致输出分布变平，并提出用分布式监督替代标量优势的DistOPD方法来解决这一问题。 |
| [^187] | [SERAG: SEmantical and Relational approach for Anomaly detection on knowledge Graphs](https://arxiv.org/abs/2611.18640) | SERAG通过融合基于标签编码的语义分数和关系图卷积网络的关系分数，实现对知识图谱中异常事实的有监督检测，在ICEWS18上显著超越最强基线。 |
| [^188] | [SynthDCB: World Distillation for Scalable Device Control Benchmarking](https://arxiv.org/abs/2611.18639) | 该论文提出SynthDCB基准，创新性地通过“世界蒸馏”方法将设备行为知识转化为模块化可执行组件，构建了覆盖家居、汽车和手机三大领域的可扩展设备控制与环境感知评测基准。 |
| [^189] | [GRAFT: Cross-Backbone LoRA Reuse for Continual Language and Multimodal Learning](https://arxiv.org/abs/2611.18638) | 提出GRAFT框架，通过模态感知的适配器库检索和学习的特征空间线性桥接，实现跨骨干网络的LoRA适配器复用，使语言模型适配器可迁移至多模态模型并显著优于随机初始化。 |
| [^190] | [ReVision: Selective Intrinsic Reasoning for Tool-Augmented, Reasoning-Intensive VQA](https://arxiv.org/abs/2611.18637) | 该论文提出ReVision——一种无需训练的测试时方法，将“内在重新审视”作为与搜索、代码执行并列的可选动作，让模型在工具增强的推理密集型视觉问答中自主决定何时重新审视原始图像，从而缓解过度依赖外部工具导致的答案反转问题。 |
| [^191] | [Beyond Binary Audits: Does Guideline Grounding Mitigate Decision-Space Framing in LLM Opioid Prescribing?](https://arxiv.org/abs/2611.18631) | 论文提出Q-Pain++全因子压力测试，发现改变剂量选项的粒度会显著影响LLM的阿片类药物处方决策，而检索指南锚定虽能部分缓解但不能消除这种决策空间框架效应，表明偏见审计应变化答案格式而不仅是患者身份。 |
| [^192] | [All the Pieces Matter! A Six-stage Analysis of an Underground Fraud Ecosystem on Dread](https://arxiv.org/abs/2611.18630) | 该研究通过分析暗网论坛Dread上九个社区的6.7万余条帖子和30万余条评论，提出了从能力储备到变现的六阶段欺诈价值链分析框架，并揭示了少数“桥梁者”用户是维系整个地下欺诈生态系统结构凝聚力的关键。 |
| [^193] | [Decision 1.0: Towards Open Decision Foundation Models](https://arxiv.org/abs/2611.18629) | Decision 1.0 是首个开放的决策基础模型家族，它能将文本状态和运行时定义的问题直接转化为类型化概率分布，无需自回归生成即可为智能体和工作流提供可复用的判断能力。 |
| [^194] | [EmoTrace: Recursive Trace-Factorized Generation for Emotional Support Conversation](https://arxiv.org/abs/2611.18628) | EmoTrace提出一种递归的轨迹因式分解生成方法，将每个对话轮次建模为封装求助者状态演变、支持策略、支持者情感与文本实现的结构化交互增量，从而在多轮情感支持对话中保持显式的决策连续性。 |
| [^195] | [Interpretability-Guided Auditing for Clinical Language Models for Discharge Prediction](https://arxiv.org/abs/2611.18626) | 提出AuditBERT-Discharge轻量级框架，将可解释性（词元级SHAP归因）作为训练时审计机制，自动识别并掩蔽临床笔记中的出院相关词汇捷径，从而减少出院预测模型对虚假线索的依赖。 |
| [^196] | [Skill Transfer: Adapting Agent Skills from Strong to Weak Agents](https://arxiv.org/abs/2611.18624) | 该论文提出“技能迁移”这一新问题与方法，在不动模型参数和技能能力上限的情况下，通过将强弱智能体的执行轨迹抽象为类型化执行结构并定位其结构差异来适配技能，从而使成本受限部署中的弱智能体也能充分受益于原本为强模型编写的技能。 |
| [^197] | [Evaluating the Robustness of Fact-Checking LLMs on Multilingual Evidence](https://arxiv.org/abs/2611.18623) | 该研究通过构建多语言黄金证据集并系统性地应用四种证据破坏策略，首次评估了事实核查大语言模型在多语言环境下的鲁棒性，发现判定准确性对证据误导性和语言资源水平均高度敏感。 |
| [^198] | [RGFC: Registry Grounded Function Calling with stage-decomposed evaluation](https://arxiv.org/abs/2611.18622) | 该论文提出注册表规模函数调用基准 RFCD 与确定性阶段分解评估器 TAAG，并发现工具注册表规模（5 至 3,000 个函数）是模型函数调用性能退化的主要来源。 |
| [^199] | [Which Mathematical Context Supports Target-Aligned Conjecture Generation? A Paired Ablation Study](https://arxiv.org/abs/2611.18621) | 本研究通过配对消融实验发现，定义、对象、假设、构造等数学上下文组件在猜想生成中总体互补，其中局部数学对象是最强的独立信号，而仅依赖单一组件会显著降低猜想与目标定理的对齐度。 |
| [^200] | [CondRelBench: A Mixed-Domain Benchmark for Condition-Scoped Relation Understanding](https://arxiv.org/abs/2611.18620) | 该论文提出混合领域基准CondRelBench，通过五类条件探测变体及OGR、SPS两个新指标，系统评估模型能否理解关系成立的条件范围，以防止“关系过度泛化”问题。 |
| [^201] | [Assisting Patients with Reliable Information Gathering: A Guide to Building Efficient QA Systems for Medical Use](https://arxiv.org/abs/2611.18619) | 本研究提出利用在线健康论坛并结合主题引导语义建模（TGSM）实现自动标注，构建无偏见且隐私安全的医疗问答数据集，同时采用基于检索器-阅读器架构的两阶段问答流水线并微调先进的transformer模型，从而高效构建可靠的医疗问答系统。 |
| [^202] | [Intrinsic Attention as Navigator: A Non-Generative Retrieval Method for Multi-Hop Queries](https://arxiv.org/abs/2611.18617) | 提出了一种无需训练的非生成式检索方法IN-Retriever，通过利用嵌入模型的内在注意力迭代提取桥接线索、剥离已发现线索并重排序候选文档，无需生成中间推理文本即可将检索精确导向多跳查询的下一跳证据。 |
| [^203] | [The Judge Is Not Impartial: Self-Preference in Medical LLM Evaluation](https://arxiv.org/abs/2611.18616) | 该研究通过13,760次LLM评判系统量化了医学评估场景中的“自我偏好”现象，发现LLM评判者会偏向同一模型生成的输出而与其临床质量无关，揭示了使用LLM评判医学内容时存在的循环性偏差风险。 |
| [^204] | [Self-Evolving Dual-Role Deliberation for Medication Recommendation from Clinical Records](https://arxiv.org/abs/2611.18615) | SEMPA是一个无需训练的双智能体药物推荐框架，通过提议者与批评者的多轮对抗协商、自然语言策略的自进化以及基于规则的药物相互作用与禁忌症安全检查，显著提升了从临床记录推荐药物的准确性（F1提升57.7%）和安全性（DDI率降低77.5%）。 |
| [^205] | [Bridging Semantics and Utilities: A Controllable and Scalable Market Simulation](https://arxiv.org/abs/2611.18613) | 该论文提出SUM语义-效用模型，通过将群体级语义感知与个体级基于效用的决策相分离，解决了现有LLM金融市场中行为控制不可靠和推理成本随智能体数量增长两大问题，实现了可控且可扩展的市场模拟。 |
| [^206] | [When Helpful Agents Cause Financial Harm: Evaluating LLM Agent Safety in Executable Economic Worlds](https://arxiv.org/abs/2611.18612) | 提出首个在可执行经济环境中系统评估LLM智能体安全性的基准EconSafeBench，发现所有被测模型在用户请求、他人压力及自身绩效目标等不同经济压力来源下均存在8.5%至40.6%的安全违规率。 |
| [^207] | [Understanding Failures in LLM Reasoning by Learning Structured Representations of Chain-of-Thought](https://arxiv.org/abs/2611.18611) | 该论文提出CoT2Graph框架，将思维链表示为有向依赖图并用图神经网络结合归因分析，揭示了正确推理更重视早期规划而失败推理往往与过载的合成步骤相关。 |
| [^208] | [NYUAD at ImageEval Shared Tasks: A Multimodal LLM for Audio-Visual Question Answering and Hallucination Detection in English and Modern Standard Arabic](https://arxiv.org/abs/2611.18610) | 该论文提出了一个基于多模态大语言模型的统一框架，能够处理英语和现代标准阿拉伯语的视听问答任务，并检测文本-图像对中的幻觉现象。 |
| [^209] | [Many Paths to the Same Mistake: Generating Diverse Reasoning Paths Behind the Same Student Mistake](https://arxiv.org/abs/2611.18609) | 本文提出了一个利用前向一致性指标和LLM评判机制的生成-过滤流水线，用于生成多样且真实的学生错误推理轨迹，并发现经过监督微调的较小模型比大型LLM更擅长持续保持错误推理。 |
| [^210] | [NYUAD at ArGuard Shared Tasks: Multimodal Embedding Models for Detecting Arabic Hateful Memes and Unsafe Prompts](https://arxiv.org/abs/2611.18608) | 本文利用预训练和微调的多模态多语言嵌入模型，在ArGuard 2026挑战赛数据集上检测阿拉伯语仇恨模因和不安全提示，并证明基于大语言模型的嵌入模型具有卓越性能。 |
| [^211] | [Justification of Emotional Appeal in Computational Argumentation](https://arxiv.org/abs/2611.18607) | 该论文首次提出情感诉求正当性预测任务并构建首个相关人工标注数据集，揭示正当性判断需综合逻辑与情感维度形成共享判断空间，且大语言模型在此任务上仍面临挑战。 |
| [^212] | [Syntactic Complexity Convergence in Dialogue: Analysis of the Phenomenon and Application to LLM Detection](https://arxiv.org/abs/2611.18606) | 本研究将句法复杂性趋同现象的分析扩展到完整对话层面，发现人类对话中存在稳健的句法趋同而人机对话呈现独特的跨指标模式，并据此利用句法复杂性特征实现了96%准确率的大语言模型对话检测。 |
| [^213] | [Enhanced English-Sanskrit Translation using Multilingual NMT Approach](https://arxiv.org/abs/2611.18605) | 该研究通过微调预训练多语言模型IndicTrans并引入印地语构建多语言迁移学习设置，显著提升了低资源英语-梵语双向翻译的性能，超越了双语基线模型。 |
| [^214] | [One Question, Four Voices: How Advice for Alzheimer's Caregiving Differs Between Caregivers, Clinicians, and Large Language Models](https://arxiv.org/abs/2611.18604) | 该研究用28项指标比较同伴照护者、医生、ChatGPT与检索增强型CareGPT对85个真实阿尔茨海默病照护问题的回答，发现各群体沟通风格截然不同，而仅靠检索增强不足以让CareGPT摆脱ChatGPT正式冗长的咨询式风格。 |
| [^215] | [Internal Verdicts Track Evidence: A Template-Controlled Neutrosophic Reading of Epistemic States in Large Language Models via the Jacobian Lens](https://arxiv.org/abs/2611.18602) | 本文首次利用雅可比透镜在大语言模型内部表征中读取中智逻辑的真(T)、不确定(I)、假(F)三成分，发现冲突证据下反驳质量激增、错误信念下不确定性持续、以及输出层前出现“判定崩塌”等关键特征。 |
| [^216] | [Jailbreak LLMs with Linguistic Style as a Hidden Attack Surface](https://arxiv.org/abs/2611.18601) | 该研究发现语言风格是LLM安全中一个此前被忽视的系统性攻击面——相同有害意图的不同风格改写可使攻击成功率产生数量级差异（高达80% vs 低于6%），并提出结合风格条件生成器与BERT风格选择器的轻量级单次攻击框架，同时通过机制分析揭示了拒绝行为是分布式失效模式而非单一拒绝机制。 |
| [^217] | [ScopeBench: Evaluating Cultural Norm Scope Awareness in LLMs](https://arxiv.org/abs/2611.18600) | 本文提出ScopeBench基准，将文化规范适用范围作为大语言模型安全评估的一等维度，通过3,323条覆盖49个国家的规范测试，揭示了模型范围错误随提示格式反转、以及范围分类准确率与实际行为安全性脱钩等结构性失败。 |
| [^218] | [Profiling Hallucinations in Frontier LLMs for Entity Linking to Medical Ontologies](https://arxiv.org/abs/2611.18595) | 本研究系统剖析了前沿大语言模型在将临床文本映射到SNOMED CT时的幻觉问题，发现其幻觉率过高而不适合自主临床编码应用，且约束使用真实提及片段反而会加剧幻觉，通用LLM的表现显著逊色于专门的零样本实体链接方法。 |
| [^219] | [Theoretical Foundations of Computational Islamic Jurisprudence (Fiqh)](https://arxiv.org/abs/2611.18593) | 本文为计算伊斯兰法学确立了六项基础原则，并构建了由阿拉伯语形态接地引擎和阿拉伯语具体化动作演算组成的三层分层计算语言形式系统。 |
| [^220] | [Endogenous Preferences as Vulnerability Signals: Discovering Universal Adversarial Triggers for LLM Utility Degradation](https://arxiv.org/abs/2611.18592) | 该论文提出EndoUT方法，以冻结模型自身干净行为中的内源性偏好破坏（EPD）作为脆弱性信号，在无需任务标签或下游反馈的情况下自动发现可复用的通用对抗触发器，从而降低大语言模型的效用。 |
| [^221] | [Where Language Models Fail Like Aphasic Patients and Where They Don't](https://arxiv.org/abs/2611.18591) | 该研究提出“计算架构分离”新方法，利用视觉语言模型“有完整词汇语义竞争但无语音编码”的架构特性做出可证伪预测，并通过与88名失语症患者约3700万次命名尝试的对比发现模型扰动错误与患者的语义类错误相符而与语音类错误不符，为词汇产生中语义提取与语音编码是可分离阶段提供了证据。 |
| [^222] | [OmniTag-LLM: Multilingual Multitask Speech Tagging](https://arxiv.org/abs/2611.18589) | 该论文提出五语言联合语音理解基准 OmniTag-Bench（一段音频输入、一个 JSON 输出，涵盖转录、情感、性别、语言识别和命名实体识别），揭示了多模态 LLM 常将非英语语音错误转成英语的系统性缺陷，并证明 LoRA 微调可显著提升各项任务表现。 |
| [^223] | [Automating literature screening and knowledge synthesis with large language models](https://arxiv.org/abs/2611.18588) | 该研究开发了一个由大语言模型驱动的统一自动化框架，可完成文献筛选、结构化信息提取与知识综合，在材料科学任务中平衡准确率超过0.90，并将筛选1万条记录的时间从约83个人工小时缩短至4分钟。 |
| [^224] | [Quantize, Qualify, Rerank: A Recipe for Compressing LLMs Without Losing Quality](https://arxiv.org/abs/2611.18587) | 该论文提出“量化-评估-重排”三步压缩方案，通过配对自助法多指标评估（MetricX24-XXL为主、CometKiwi-XXL交叉验证）结合Cometoid best-of-4重排技术，成功将Gemma-3-12b-it压缩至8-bit/4-bit精度且在机器翻译任务中不损失质量。 |
| [^225] | [Read, Route, Patch: Robust Test-Time Evolution for LLM Agents](https://arxiv.org/abs/2611.18585) | 提出Trident框架，通过边界感知记忆阅读、多样性感知候选路由和顺序感知插入式更新，解决LLM智能体测试时演化中的三个信息瓶颈，在五个基准和多个骨干模型上取得最佳平均性能。 |
| [^226] | ["I Am Not a Kiddo!": Quantifying Epistemic Infantilization in Large Language Models via Multi-Metric Bias Analysis](https://arxiv.org/abs/2611.18583) | 该论文提出了IBI和EARS两个复合指标，通过反事实数据集量化了大语言模型在用户披露残疾信息后回答专家级问题时系统性降低技术深度的“认知幼化”偏差。 |
| [^227] | [VidEngram: Agentic Long-Term Video Understanding via Structured Hierarchical Memory](https://arxiv.org/abs/2611.18578) | VidEngram 通过在视频摄取时一次性构建持久化的结构化分层记忆（片段、事件与实体层），并利用纯文本智能体规划器配合专用检索工具回答后续问题，实现了高效、可检视的长视频问答，在 OmniVideoBench 上超越了单次编码基线。 |
| [^228] | [GT-FENG: General Theory of Factual Evaluation and Normative Grounding in Language Models *Author(s)](https://arxiv.org/abs/2611.18279) | GT-FENG是首个统一上下文/时间正确性与基于规则的错误归因的通用框架，通过对照适用规则检验语言模型答案并追溯错误来源，在Medicaid收入上限问答中使仅匹配其他州数值的错误答案减少68-78%。 |
| [^229] | [Evaluating Pretrained Language Semantics for Structured EHR Prediction Tasks](https://arxiv.org/abs/2611.17053) | 该论文提出tEHR框架，将临床事件转化为自然语言描述并通过持续预训练和判别式微调适配预训练语言模型，证明预训练语言语义能显著提升结构化EHR预测性能，其中Mistral-7B在四个MIMIC-IV任务上超越了各类基线方法。 |
| [^230] | [We Are What We Remember: Mental-State-Driven Memory for Long-Term Character Dialogue](https://arxiv.org/abs/2611.16223) | 提出受认知心理学自我记忆系统启发的MindMem框架，通过异步将共同经历整合为角色、用户及关系的心理状态来驱动主动式记忆推理与用户需求预测，并引入MemReal-Bench基准评估超越事实回忆的历史敏感对话适应性。 |
| [^231] | [Vision-Grounded Semantic Learning and Alignment for Scalable Multilingual Text-Based Person Retrieval](https://arxiv.org/abs/2611.15328) | 该论文提出了首个大规模多语言行人预训练数据集MultiPedes（包含四种语言的1000万图文对），并引入视觉锚定的语义学习与对齐方法，解决了多语言文本行人检索中缺乏预训练数据和英语中心偏差的问题。 |
| [^232] | [A Closer Look at Language Model Introspection: Probing Information and Reporting](https://arxiv.org/abs/2611.15267) | 该研究通过对比线性探测分类器与模型自身的零样本回答发现：在激活扰动定位任务中，探测分类器能轻松达到 98.3% 的准确率而模型报告仍处于随机水平，表明语言模型内省的表现可能主要源于报告易于线性恢复的信息，而非真正的自我反思能力。 |
| [^233] | [Typed Knowledge over Constructive Institutions: A Proof-Relevant Type Theory for Verified Science](https://arxiv.org/abs/2611.15108) | 本文提出了面向可验证科学知识基底 Eigenius 的证明相关依赖类型论，通过摒弃判断式证明无关性，使系统能够追踪并比较同一命题的多个独立推导、保持强规范化与传递性的等价判定，并支持在实验或模型失效时计算反事实撤回。 |
| [^234] | [A Multi-Word Tokenizer for 204 Languages That Needs Fewer Tokens Than GPT-4o's, Including on English](https://arxiv.org/abs/2611.13059) | 该论文提出Havon多词分词器系列，通过在约200种语言上从头训练并引入多词训练阶段，在包括英语在内的161种语言上所需词元数量少于GPT-4o的分词器，在印度语言上的词元数中位数仅为GPT-4o的0.74倍。 |
| [^235] | [The Indic Token Tax in 2026: Measuring Tokenizer. Disparity and Its Cost for the Languages of India](https://arxiv.org/abs/2611.13058) | 该研究测量了九个主流分词器对印度25种语言的“代币税”，发现印度语言所需token量最高可达英语的13.7倍（GPT-4o分词器），导致印度语言用户在商业LLM使用中面临更高的费用、更慢的速度和更小的有效上下文，而Sarvam分词器是唯一将差异控制在2.6倍以内的第三方选择。 |
| [^236] | [Unprompted coordination through shared memory in LLM agents: a locked two-implementation replication and extension](https://arxiv.org/abs/2611.12412) | 本研究通过双重独立实现的复现与扩展首次证实：即使提示中完全没有提及其他智能体、共享或协作的存在，LLM智能体也能仅凭共享内存状态自发产生协调行为。 |
| [^237] | [Infrathon: Benchmarking Language Agents for Real-World, Long-Horizon AI Infra Optimization](https://arxiv.org/abs/2611.11995) | 该论文提出Infrathon长周期基准，通过对六个模型进行24小时、累计超10,000 GPU小时的评估，发现语言智能体能持续改进真实AI基础设施，模型排名随评估时长变化，且智能体可发现超越人类设计的方案（如将KV缓存压缩比从6.82倍提升至19.86倍）。 |
| [^238] | [Understanding When and Where Self-Supervised Speech Models Learn Phonemes and Prosodic Cues through Layer-Wise and Time-Resolved Probing](https://arxiv.org/abs/2611.11249) | 该研究通过对预训练各检查点的所有Transformer层进行时间分辨探测，发现声学韵律线索和音素边界从预训练最早期即可解码且在各层间保持稳定，而音素身份在预训练中逐步构建并集中于中上层，表明信息习得顺序取决于其相对声学表面的抽象程度。 |
| [^239] | [Do Benign-Trained Hidden-State Probes Detect Tool Errors Under Adversarial Conditions?](https://arxiv.org/abs/2611.11172) | 该论文在生物医学问答任务中通过多代递归实验，首次系统比较了语言模型在自身生成答案上递归训练（合成数据反馈循环）与持续使用人工数据的差异，揭示错误与分布偏差如何随代次累积。 |
| [^240] | [Beyond Truncation? Entailment-Licensed Elision and the Limits of Semantic Preservation in LLM Decoding](https://arxiv.org/abs/2611.08975) | 该论文证明了现有免训练的解码期冗余删除方法本质上都只是截断，并提出ELIDE方法，利用自然语言推理模型在每个句子边界判断被删内容是否被保留内容所蕴含，从而实现真正超越截断的语义保持式回答精简。 |
| [^241] | [Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes](https://arxiv.org/abs/2611.08439) | 论文对俄语类型化决策交叉编码器 Ru-Jev 引入标签多形式措辞增强，发现该增强对由普通词和外来词构成的标签词汇表能带来一致的准确性提升（最高达 0.24），但对缩写和代号类标签没有增益。 |
| [^242] | [CardioLM: A multilingual suite of small language models for the cardiology domain](https://arxiv.org/abs/2611.07940) | 该研究提出了多语言心脏病学病例语料库 CardioCCC 以及小型语言模型套件 CardioLM（含编码器 CardioBERTa 和命名实体识别模型 CardioNER），填补了心脏病学领域多语言临床 NLP 资源的空白。 |
| [^243] | [Discourse Analysis of Legislative Bill Discussions](https://arxiv.org/abs/2611.05381) | 本文提出了一种用于解析立法委员会听证会修辞结构的端到端流水线和修辞结构语法，其中55.21%的加州州法案讨论可被该语法解析，从而改进了立法话语分析与民主问责。 |
| [^244] | [NitiBench+: A Temporal Thai Legal Question Answering Benchmark Across 17 Domains](https://arxiv.org/abs/2611.03576) | NitiBench+是一个包含1,670个条目、覆盖17个领域、同时收录判决日有效与当前法条文本的泰国时序法律问答基准，实验表明模型使用过时法条文本会显著损害法律问答表现。 |
| [^245] | [Write Facts the Way Models Read Them: Lifting the Two-Hop Curse](https://arxiv.org/abs/2611.01519) | 论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。 |
| [^246] | [S1CogBench: What System One Decision Models Can and Cannot Do on Cognitive Experiments](https://arxiv.org/abs/2611.00911) | 该论文提出S1CogBench基准，通过24项认知任务、27种评估模式，结合人类反应预测、自主执行与受控任务操纵，系统揭示了系统1决策模型的认知能力与局限，发现了聚合分数无法捕捉的模型行为差异与失败模式。 |
| [^247] | [Sparse Autoencoder Attribution Graphs Are Decided at the Margin](https://arxiv.org/abs/2611.00606) | 该论文揭示 TopK 稀疏自编码器的特征激活值常贴近截断阈值，导致传统归因图误差较大，并提出一种在相同计算成本下为每个预激活加入一阶变化并重新排序的“边际归因图”方法，将中位相对误差从 0.247 以上降至至多 0.033。 |
| [^248] | [In-Model Context Compression with Agentic Soft Tokens for Long-Horizon LLM Agents](https://arxiv.org/abs/2610.41291) | 提出免训练的智能体软令牌压缩方法（ASTC），利用混合GDN-注意力的原生循环动态将已吸收信息压缩为模型内软令牌，在降低长时程LLM智能体计算成本的同时保留关键决策信息。 |
| [^249] | [Language Models Show Conformity Bias in RAG: When the Majority Endorses a Poisoned Claim](https://arxiv.org/abs/2610.41012) | 本研究发现当RAG检索上下文中多数相互一致的投毒文档共同支持错误信息时，语言模型的投毒检测率从25%显著降至10%，揭示了语言模型存在类似社会从众心理的从众偏差，使其更容易被多数派的投毒信息误导。 |
| [^250] | [CAMR: Compatibility-Aware Memory Replay for Continual Learning in Large Language Models](https://arxiv.org/abs/2610.36734) | CAMR 提出了一种兼容性感知的自适应记忆回放框架，通过在线策略联合控制新样本监督与历史回放，在持续学习中平衡大语言模型对新知识的获取与对已有知识的保留。 |
| [^251] | [Style over Substance? Linguistic Bias in LLM-based Peer Review Copy title](https://arxiv.org/abs/2610.35254) | 该论文提出了一个受控实验框架，通过基于语言学文献推导的细粒度改写指令，系统性地改变论文的语言风格（母语/非母语特征与高/低置信度特征），以研究基于大语言模型的同行评审是否会受到与科学内容无关的语言偏见的影响。 |
| [^252] | [Examining form-based linguistic productivity in people and large language models](https://arxiv.org/abs/2610.33433) | 该研究发现人类能够基于抽象的语言形式规则将新动词从主动语态泛化到被动语态，而大型语言模型只有在新动词具有类似已知动词的意义时才能成功，表明计算模型的句法知识仍依赖于具体词汇的分布与意义，缺乏人类那样的抽象性。 |
| [^253] | [Recursive Semantic Divergence for LLM Agent Consistency](https://arxiv.org/abs/2610.33012) | 该论文提出了递归语义散度（RSD），一种改编自强化学习双模拟度量的无监督轨迹级一致性指标，用于衡量LLM智能体在独立运行中的行为不一致性及其在轨迹中的传播。 |
| [^254] | [Inside the Loop: A Mechanistic Study of Looped Transformers on Permutation Traversal](https://arxiv.org/abs/2610.24135) | 研究发现循环变换器解决有向排列遍历任务需要最小有效深度，超过该深度后权重绑定的程度成为决定性能的关键因素。 |
| [^255] | [Wolof-Adapted Industry-Ready Open-Weight Language Models](https://arxiv.org/abs/2610.20824) | 本文通过指令微调、偏好对齐和模型合并构建了0.6B-14B的沃洛夫语适配开放权重模型，在63项任务上表现出色，其中8B和14B模型可与更大的商业系统相竞争，且已公开发布。 |
| [^256] | [Twenty Days Later: A Pre-Registered Test of History-Based Forecasts of Brand Mentions in Gemini Answers](https://arxiv.org/abs/2610.19608) | 该论文通过预注册的重复实验证明，基于历史数据对Gemini回答中品牌提及率的预测在约20天后依然准确（Brier得分0.0281，接近8小时短期预测水平且优于品牌历史基线），表明大语言模型的品牌提及计数具有足够的时间稳定性，可供实际决策使用。 |
| [^257] | [SkillScreen: High-Recall Screening of Malicious Agent Skills with Decision Models](https://arxiv.org/abs/2610.17727) | SkillScreen 是一个紧凑的决策模型，能在单次前向传播中直接对 SKILL.md 技能的恶意程度评分，在未见攻击族上达到 99% 召回率且误报远低于词级分类器和 LLM 判官，但对陌生良性来源的误报仍会显著上升。 |
| [^258] | [Thinking Outside the Bots: Human Ideas in AI's Semantic Blind Spots Are Not Inferior](https://arxiv.org/abs/2610.17324) | 本研究开发了EDGE方法来绘制AI的语义“盲区”，发现在大学文书、物品替代用途和人际建议三项开放性任务中，人类想法均延伸至AI输出稀少覆盖的语义区域，且这些独特的人类回应在创造力上并不逊色。 |
| [^259] | [Unreliable Narrator Framing as a Cost-effective Mechanism for Satire Explanation of Australian and American Satirical News](https://arxiv.org/abs/2610.12863) | 该研究提出以不可靠叙述者理论为框架，利用开放权重模型低成本地为澳大利亚和美国讽刺新闻生成解释，并贡献了新的数据集与评估基准。 |
| [^260] | [Not What a Child Expressed: Auditing the Sign-to-Text Safety Interface in Child-Facing AI](https://arxiv.org/abs/2610.07519) | 本文首次提出由聋人参与指导的部署前审计框架，用于审查“手语转文本”系统与儿童安全审核工具之间的接口，揭示SLT翻译错误可能在不影响文本流畅性的情况下悄然改变针对儿童手语使用者的安全决策。 |
| [^261] | [SearchJev: A Calibrated System-1 Model for Search Agents](https://arxiv.org/abs/2610.05107) | SearchJev提出一个无需自回归生成的校准化System-1模型，将搜索代理的短决策与System-2推理分离，并通过软标签学习（SLCD）从不确定监督中提升决策置信度的可靠性。 |
| [^262] | [Attribution Gaps in Zero-Training LLM+OVOD Pipelines: A Fine-Grained Analysis of the CAAP--SNAP Discrepancy](https://arxiv.org/abs/2609.32567) | 该研究在完整COCO-Val数据集上首次揭示了零训练LLM+OVOD流水线中CAAP与SNAP差异的根源在于词汇新颖性而非物体视觉复杂性——当LLM措辞超出检测器原生词汇表时定位准确率从80.9%骤降至31.6%，且这一下降主要源于新颖措辞实际命名了与标注不同的物体，而非同义词表述问题。 |
| [^263] | [Form Over Content In Gradient-Based Data Attribution Methods](https://arxiv.org/abs/2609.19589) | 基于梯度的数据归因方法主要捕捉答案格式而非任务内容，因为共享答案格式的数据集表现出强梯度对齐，而任务相同但格式不同的数据集则不对齐。 |
| [^264] | [Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation](https://arxiv.org/abs/2609.19093) | 该研究提出由放射科医生参与的报告实践变化分类法及ReRef参考报告重写方法，揭示参考报告写作风格的差异会显著影响胸部X光报告自动评估结果，甚至足以改变生成模型的排名。 |
| [^265] | [Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents](https://arxiv.org/abs/2609.18304) | 提出了回滚诱导反思（RIR）统一恢复框架，在将LLM智能体回滚到选定先前状态的同时，保留从被放弃轨迹中提炼的可复用知识，解决了长程任务中错误累积且难以可靠恢复的问题。 |
| [^266] | [Too Good to Be Real? Diagnosing and Reducing the Gap Between AI Preference and Real User Engagement](https://arxiv.org/abs/2609.18282) | 该研究基于知乎、Quora和Reddit的117万条回答发现，大语言模型存在“逻辑过度绑定”倾向，即偏好增加逻辑结构，而真实用户参与度更依赖情感与表达显著性，并提出本体掩码推理自编码（OMRA）方法来缩小这一差距。 |
| [^267] | [Large Language Models Develop Belief State Geometry In-Context](https://arxiv.org/abs/2609.17376) | 研究发现大语言模型在上下文学习隐马尔可夫模型数据时，会在其内部残差流激活中以线性可解码的方式编码出信念状态的几何表示，且通过干预实验证明该表示与模型的预测行为功能相关。 |
| [^268] | [MindTopo: Can Foundation Models Reason in Topological Space?](https://arxiv.org/abs/2609.11900) | 该论文提出了MindTopo基准，基于认知科学与形式拓扑学，从连续性、分离性、有序性、包含性和纽结五个拓扑属性出发，在推理与规划两个认知层面对14个多模态大语言模型的拓扑空间能力进行系统评估。 |
| [^269] | [Vishing-Tactics-Bench: Forecasting Exploitation Trajectories in Voice Phishing Calls](https://arxiv.org/abs/2609.07151) | 提出基于 Endsley 态势感知框架的语音钓鱼防御基准 Vishing-Tactics-Bench，将防御范式从事后欺诈分类转变为对正在进行的通话是否将到达信息收集或金融剥削等终端危害的实时预测，并构建了 6 战术分类体系和 35,340 条标注话语的数据集。 |
| [^270] | [ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering](https://arxiv.org/abs/2609.02486) | 提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。 |
| [^271] | [The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally](https://arxiv.org/abs/2609.01587) | 该研究通过因果混合精度干预实验发现，LLM的量化损伤是弥散分布的而非集中于特定电路、计算位置或权重统计，因此在匹配精度预算下，将额外比特全局用于更精细的量化粒度比局部修复少数层更有效。 |
| [^272] | [Hi-Q: Hierarchical Evidence-guided Query Refinement for Multi-Hop Question Answering](https://arxiv.org/abs/2608.30468) | Hi-Q提出了一种以证据为条件的分层查询细化框架，通过在每个查询节点上利用解析算子判断证据是否支持当前查询单元，动态构建查询树，从而解决多跳问答中问题表达粒度与语料证据可检索粒度不匹配的核心瓶颈。 |
| [^273] | [Layered LLM Defenses as an Ensemble: Access Tiers, Inference Cost, and the Measured Failure Correlation Between Defense Layers](https://arxiv.org/abs/2608.28327) | 本文提出对抗者访问层级模型（AATM）和推理成本分类方法，首次实测了大语言模型多层防御之间的失败相关性，证明防御堆栈只有在各层在不同输入上失败时才能真正产生叠加防护效果。 |
| [^274] | [IndicQE-APE: A Benchmark for Quality Estimation and Automatic Post-Editing for Indic Languages](https://arxiv.org/abs/2608.16344) | 本文整合了印度语言的质量评估和自动后编辑数据，创建了一个包含多标签和难度分层的基准，并评估了多种模型，发现只有同时利用整体和词级信息的系统在严格对照下表现显著。 |
| [^275] | [Prompt Privilege in Large Language Models: Measuring and Mitigating Barriers to Equitable AI Access](https://arxiv.org/abs/2608.08942) | 该论文首次提出“提示特权”这一公平性问题，即具备提示工程专业知识的用户能因措辞技巧而获得更好的模型输出，并设计了提示公平分数（PES）评估指标与提示公平转换器（PET）智能体来衡量和缓解这种AI访问不平等。 |
| [^276] | [LangChoiceBench: Measuring and Explaining Programming-Language Choice in LLMs](https://arxiv.org/abs/2608.06041) | 提出了LangChoiceBench基准，系统测量25个大语言模型在28个项目中的编程语言选择行为，发现模型普遍过度偏好Python且多为未经深思熟虑的自动选择，较小的开放权重模型偏好更强、语言多样性更低。 |
| [^277] | [A False Average: Chain-of-Thought Monitors Collapse Where They Are the Only Defense](https://arxiv.org/abs/2608.00583) | 仅重写智能体的思维链推理（保持动作完全不变）即可在一次无梯度攻击中将CoT监控器的捕获率从约95%暴跌至11%以下，揭示监控器的总体准确率是掩盖其在唯一防线场景下近乎全面失效的虚假平均值。 |
| [^278] | [RubricReviewer: From Direct Critique to Rubric-Driven Review via Agentic Evidence Synthesis and Human-Aligned Generation](https://arxiv.org/abs/2608.00005) | 该论文提出RubricReviewer，一个完全由评分标准驱动的同行评审框架，通过将评分标准生成作为显式中间步骤，并结合免训练智能体的证据合成与基于训练模型的人类对齐生成，解决了现有LLM评审器评判标准隐式化、两类范式割裂的结构性局限。 |
| [^279] | [Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments](https://arxiv.org/abs/2607.28591) | Change2Task 提出了一个基于仓库历史的系统，能够将已合并的拉取请求自动转换为健康现代代码版本上经过验证的可执行编码智能体任务与环境，为编码智能体的训练、基准测试和持续评估提供可扩展的数据供给。 |
| [^280] | [Characterizing Human-Likeness in AI Generated Poetry: A Zero-shot Classification Study](https://arxiv.org/abs/2607.26221) | 该研究通过零样本分类方法刻画英语诗歌中影响人类诗歌与AI生成诗歌正确或错误分类的属性特征，为“GenAI诗歌最难被现代检测工具识别”这一论断提供了客观验证依据。 |
| [^281] | [Pass the Baton: Trajectory-Relayed On-Policy Distillation](https://arxiv.org/abs/2607.26057) | 针对在线策略蒸馏中学生一旦走上错误推理方向便难以纠正的“前缀失败”问题，本文提出 Relay-OPD，利用师生延续不对称性作为无标签触发信号，让教师在关键位置短暂接管生成后再交还学生，以有限的干预预算提升蒸馏监督的可靠性。 |
| [^282] | [Population-Level Profiling of DSM-5 Depressive Symptom Language Among Twitter Users With Self-Reported ADHD or ASD: Exploratory Study Using Natural Language Processing and Statistical Analysis](https://arxiv.org/abs/2607.05626) | 该研究构建了零样本分类器结合本地大语言模型（Qwen3-8B）与经ReDSM5微调的MentalRoBERTa的自然语言处理流程，对自述ADHD或ASD的792名Twitter用户的百万余条推文进行DSM-5九项抑郁症状的自动分类与统计建模，从而在群体水平揭示了两组神经发育障碍人群抑郁症状语言表达的差异。 |
| [^283] | [Beyond Satisfaction: Learning Associations Between Content, Reviews, and Well-Being](https://arxiv.org/abs/2607.02539) | 该论文提出以心理学PERMA幸福感框架替代传统满意度指标作为推荐目标，通过分析1370万条Goodreads评论和1.6万本书的内容，揭示了书籍内容特征与用户享乐型及实现型幸福感表达之间的关联。 |
| [^284] | [Multi-Head Recurrent Memory Agents](https://arxiv.org/abs/2607.01523) | 该论文诊断出循环记忆智能体性能随上下文增长而退化的瓶颈在于记忆保持而非记忆捕获，并提出无需训练的多头循环记忆（MHM）框架，通过将记忆划分为独立记忆头并采用“先选择后更新”策略保护未被选中的记忆免遭覆盖，从而显著提升长上下文的可靠性。 |
| [^285] | [Linguistic Bias Mitigation for Spoofing Detection via Gradient Reversal and A Variational Information Bottleneck](https://arxiv.org/abs/2606.31411) | 该论文提出了一种基于师生对抗学习与变分信息瓶颈的语言不变语音欺骗检测框架，通过梯度反转消除语言偏差，在九个DF Arena数据集上实现了最高36.2%的等错误率相对降低。 |
| [^286] | [DriftGuard: Safety-Aware Multi-Monitor Detection and Selective Adaptation for Evolving Toxicity Moderation](https://arxiv.org/abs/2606.28725) | 论文提出 DriftGuard 框架，通过监控全局文本漂移、身份伤害漂移、模型不确定性和假阴性风险等多个维度的安全相关漂移信号，检测毒性内容审核中的安全隐患，并利用优先纳入高风险样本的困难混合适应集对模型进行选择性更新，以应对不断演化的有害内容。 |
| [^287] | [From Weights to Features: SAE-Guided Activation Regularization for LLM Continual Learning](https://arxiv.org/abs/2606.26629) | 本文提出利用稀疏自编码器（SAE）在激活空间进行正则化，通过单义特征字典显式平衡稳定性与可塑性，解决了大语言模型持续学习中EWC等权重正则化方法因多义性而表现不佳的问题。 |
| [^288] | [Cited Web Sources in LLM Answers about Brands: A Multilingual Audit](https://arxiv.org/abs/2606.25787) | 该研究对涵盖128个品牌、13种语言和12个市场的167,551条大语言模型回答中的引用网址进行了多语言审计，揭示了引用来源的高度集中性（少数域名占80%的引用且维基百科占据主导地位），并论证了标准化、分类规则和明确分母在引用审计中的必要性。 |
| [^289] | [RAS: Measuring LLM Safety Through Refusal Alignment](https://arxiv.org/abs/2606.25750) | 本文提出RAS（拒绝对齐分数），一种基于表示、仅需前向传播即可高效评估大语言模型拒绝有害请求能力的指标，在Llama、Gemma和Qwen系列模型上与真实攻击成功率高度相关。 |
| [^290] | [HandwritingAgent: Language-Driven Handwriting Synthesis in Scalable Vector Space](https://arxiv.org/abs/2606.18788) | HandwritingAgent 是一个语言驱动的智能体，无需针对特定风格进行训练，即可利用大型推理模型直接以 SVG 格式合成自然手写笔画序列，并可通过自然语言灵活控制书写风格。 |
| [^291] | [Uncertainty Is Not a Safety Net for Clinical VQA, but Can It Anticipate Model Failure?](https://arxiv.org/abs/2606.16583) | 该研究通过在12个临床视觉语言模型上基准测试8种不确定性估计方法，发现不确定性估计的质量会随模型准确率下降而退化、且在正确答案被隐藏时无法发出警示，但未扰动输入上的不确定性本身携带诊断信息，能够预判哪些预测将在模型失效时崩溃。 |
| [^292] | [Expert-Level Crisis Detection in Mental Health Conversations](https://arxiv.org/abs/2606.10380) | 该论文提出了临床医生标注的CRADLE-Dialogue基准数据集以及“警报-确认”评估协议，用于解决多轮心理健康对话中轮次级危机检测的难题，使模型能够捕捉随对话演进的风险信号并支持早期干预。 |
| [^293] | [CARE: A Conformal Safety Layer for Medical Summarization](https://arxiv.org/abs/2606.08969) | CARE 是一种无需重新训练模型的事后保形风险控制层，通过 CRC 与 LTT-FST 为医学摘要添加经过校准的遗漏与幻觉标注，在形式上控制漏检错误概率并平衡安全性与临床审阅负担。 |
| [^294] | [Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests](https://arxiv.org/abs/2606.07379) | 该论文提出CapCode框架与CapReward奖励设计，通过随机测试将合法可达分数刻意封顶在100%以下，使超出上限的分数成为作弊证据，从而检测并抑制编码智能体评估中的作弊行为。 |
| [^295] | [K-BrowseComp: A Web Browsing Agent Benchmark Grounded in Korean Contexts](https://arxiv.org/abs/2606.02404) | K-BrowseComp是首个基于韩语语境构建的网络浏览智能体基准，包含400个问题（其中300个经韩语母语者人工验证），实验显示前沿大模型在韩语环境下表现显著下降（30%–45.67%），而韩国本土模型表现更差（0%–10.33%）。 |
| [^296] | [BioELX: Context-Aware Cross-lingual Biomedical Entity Linking without Task-Specific Supervision](https://arxiv.org/abs/2605.27380) | BioELX提出了一种检索-重排序框架，利用Wikidata衍生的跨语言别名监督进行检索，并通过提及锚定提示将LLM重排序器适配于实体链接，实现了无需任务特定监督的上下文感知跨语言生物医学实体链接。 |
| [^297] | [JuICE: A Benchmark for Evaluating LLM-Judge in Identifying Cultural Errors](https://arxiv.org/abs/2605.26955) | JuICE是一个包含7,470个片段级标注的多语言基准数据集，用于评估LLM裁判识别长篇回复中深层文化错误的能力，弥补了现有文化基准将文化简化为扁平事实集合的不足。 |
| [^298] | [SAMark: A Self-Anchored Text Watermarking with Paragraph-Level Paraphrase Robustness](https://arxiv.org/abs/2605.25796) | SAMark提出了一种自锚定文本水印框架，通过在语义空间中建立与句子顺序无关的绿色区域和多通道双曲评分机制，显著提升了水印对段落级改写攻击的鲁棒性，在1%假阳性率下实现了高达90.2%的真阳性率。 |
| [^299] | [Harmony in Diversity: Multi-domain Contrastive Policy Optimization for Large Reasoning Models](https://arxiv.org/abs/2605.25443) | 提出多域对比策略优化（MCPO）方法，通过在关键推理片段上应用对比学习，实现跨领域的知识共享与竞争的平衡，从而缓解混合领域强化学习中不同领域间的能力权衡问题，使单个大型推理模型的多领域性能得以匹配甚至超越单领域训练。 |
| [^300] | [AI Agents May Always Fall for Prompt Injections](https://arxiv.org/abs/2605.17634) | 本文借助上下文完整性理论重新定义提示注入攻击，并提出不可能性论证，证明数据-指令分离等现有防御范式注定无法彻底解决提示注入问题，因为攻击者总能构造使恶意信息流显得合法的上下文。 |
| [^301] | [HEED: Density-Weighted Residual Alignment for Hybrid Vision-Language Model Distillation](https://arxiv.org/abs/2605.17093) | 该论文发现混合架构蒸馏的失败集中于承载文本与局部细节的高密度图像块（其残差漂移达3.6倍），并提出密度加权残差对齐方法HEED，有效修复了学生在OCR和文档任务上的细粒度信息丢失。 |
| [^302] | [Inducing Artificial Uncertainty in Language Models](https://arxiv.org/abs/2605.13595) | 该论文提出在语言模型中诱导人工不确定性的新问题及方法，以解决当数据集饱和、难以找到足够不确定的监督数据时，不确定性量化方法可能失效的挑战。 |
| [^303] | [LatentRouter: Can We Choose the Right Multimodal Large Language Model Before Seeing Its Answer?](https://arxiv.org/abs/2605.11301) | 该论文提出LatentRouter框架，将多模态大语言模型路由建模为反事实多模态效用预测问题，通过路由胶囊与模型能力token之间的潜在通信，在看到模型答案之前预测并选出最适合当前图像-问题查询的模型。 |
| [^304] | [Test-Time Speculation](https://arxiv.org/abs/2605.09329) | 该论文提出测试时推测，通过在线蒸馏在推理阶段持续自适应调整推测器，解决了推测解码因训练与推理序列长度分布不匹配而在长输出任务中接受长度退化、加速失效的问题。 |
| [^305] | [SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation](https://arxiv.org/abs/2605.07711) | SimCT通过在共享token匹配之外引入两个分词器均可实现的短多token延续作为监督单元，在不改变OPD损失形式的前提下，恢复了跨分词器在线策略蒸馏中因词表不一致而丢失的教师信号。 |
| [^306] | [The Metacognitive Profile Atlas: Domain-level confidence discrimination in 33 frontier LLMs on MMLU](https://arxiv.org/abs/2605.06673) | 该研究发布了元认知画像图谱数据集，揭示33个前沿大语言模型的元认知信心辨别能力存在系统性领域差异——在应用/专业类题目上最强，在形式推理和自然科学上最弱，且这一排序在群体层面稳定可靠。 |
| [^307] | [Contrastive Identification and Generation in the Limit](https://arxiv.org/abs/2605.06211) | 本文提出基于对比关系信号（即不相似样本对）的极限识别与生成新理论，精确刻画了在有限破坏预算下假设类可识别的条件，并证明无界预算会使任意两个假设都变得不可区分。 |
| [^308] | [The Granularity Axis: A Micro-to-Macro Latent Direction for Social Roles in Language Models](https://arxiv.org/abs/2605.06196) | 本文发现语言模型表征空间中存在一条“粒度轴”——宏观角色与微观角色隐藏状态均值的差异方向，该方向与角色空间的第一主成分高度对齐，并跨模型、角色层级、网络层和提示词稳定存在，证明模型内部编码了社会角色的粒度差异。 |
| [^309] | [The First Token Knows: Single-Decode Confidence for Hallucination Detection](https://arxiv.org/abs/2605.05166) | 该论文提出首个词元置信度 phi_first，仅需单次贪心解码即可从 top-K 首词元分布中估计不确定性，其幻觉检测效果媲美甚至略超需要十次采样生成加 NLI 语义聚类的语义自一致性方法，大幅降低计算成本。 |
| [^310] | [Fuzzy Fingerprinting Encoder Pre-trained Language Models for Emotion Recognition in Conversations: Human Assessment and Validity Study](https://arxiv.org/abs/2605.02665) | 该论文提出了一种将预训练语言模型与模糊指纹相结合的新颖可解释方法，通过类别特定原型揭示模型潜在空间中的特征激活模式，从而提升对话情绪识别的可解释性，并解决不平衡数据集中少数情绪被误分类为中性类别的问题。 |
| [^311] | [Seeing Is No Longer Believing: Frontier Image Generation Models, Synthetic Visual Evidence, and Real-World Risk](https://arxiv.org/abs/2604.24197) | 本文是一篇叙述性综述，系统梳理了前沿图像生成模型的高逼真合成能力如何使合成图像获得“证据权威”，从而对新闻、金融、身份验证、医疗和法律等现实领域构成风险，并区分了厂商能力声明、已记录事件与潜在危害路径。 |
| [^312] | [Verbal tics in frontier language models: A critical review of current releases, research evidence, and public discussion](https://arxiv.org/abs/2604.19139) | 这篇批判性综述系统考察了OpenAI、Anthropic、Google DeepMind等八家开发商前沿语言模型中的口头禅式语言习惯与谄媚附和现象，综合研究证据与公众讨论，指出附和性语言会扭曲判断，而对最新发布版本的可比测量仍然稀缺。 |
| [^313] | [Seeing but Not Thinking: Routing Distraction in Multimodal Mixture-of-Experts](https://arxiv.org/abs/2604.08541) | 本文发现多模态MoE模型存在“看见却未思考”现象——模型能准确感知图像却推理失败，并提出“路由分心”假设：视觉输入使路由机制偏离、未能充分激活任务相关的推理专家，从而解释了感知与推理之间的断裂。 |
| [^314] | [ORACLE-SWE: Quantifying the Contribution of Critical Information Signals on SWE Agents](https://arxiv.org/abs/2604.07789) | 该论文提出ORACLE-SWE，首次系统量化了复现测试、回归测试、编辑位置、执行上下文和API用法等关键中间信息信号对软件工程代理任务成功的贡献，并测量了代理发现的信号与理想神谕信号之间尚存的改进空间。 |
| [^315] | [Appear2Meaning: A Cross-Cultural Benchmark for Structured Cultural Metadata Inference from Images](https://arxiv.org/abs/2604.07338) | 提出了包含750件博物馆物品的跨文化基准Appear2Meaning，用于评估视觉语言模型仅凭图像推断标题、文化、时期、产地和创作者等结构化文化元数据的能力，结果显示当前模型的完整记录兼容率仍然很低。 |
| [^316] | [On the Price of Privacy for Language Identification and Generation](https://arxiv.org/abs/2604.07238) | 该论文揭示了差分隐私在语言学习中的代价结构：语言识别中隐私几乎免费，而语言生成中代价取决于公开信息——有公开见证界时纯DP仅损失Θ(min{1,ε})的指数因子，无此界时纯DP则会彻底丧失指数收敛速率。 |
| [^317] | [A Parameter-Efficient Transfer Learning Approach through Multitask Prompt Distillation and Decomposition for Clinical NLP](https://arxiv.org/abs/2604.06650) | 提出多任务提示蒸馏与分解框架，从21个临床源任务中学习单一共享元提示，仅用不足0.05%的可训练参数即可适配新任务，以远少于LoRA的参数量在10个临床NLP数据集上取得更优性能。 |
| [^318] | [Supporting Calibrated Reliance in Human-AI Collaboration: Different Strategies for Different Tasks](https://arxiv.org/abs/2604.03237) | 本文通过实验发现，AI支持形式（如预测、解释、概率）对人机协作中的校准依赖影响因任务类型而异，视觉推理中预测概率最有效，而解释并非总是提升性能。 |
| [^319] | [Automated Code Efficiency Optimization via Agent Skill Mining](https://arxiv.org/abs/2603.27850) | 本文提出EffiSkill两阶段框架，将慢代码到快代码的常见转换挖掘并沉淀为记录意图、适用性、流程、效果与风险的“算子技能”卡片库，使LLM无需执行反馈即可诊断程序、检索技能并生成多个候选优化方案，从而自动提升代码效率。 |
| [^320] | [CAF-Score: Calibrating CLAP with LALMs for Reference-free Audio-Text Alignment Evaluation](https://arxiv.org/abs/2603.19615) | CAF-Score是一种无参考的音频-文本对齐评估指标，它将CLAP的粗粒度语义对齐与大型音频-语言模型（LALM）的细粒度理解和句法感知能力校准融合，凭借对称评分机制，单一配置即可同时适用于音频描述和文本到音频生成两个评估方向，并在三个人工标注基准上取得与人类判断更高的一致性。 |
| [^321] | [TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning](https://arxiv.org/abs/2603.12529) | 提出了Terminator，一种针对大型推理模型的推理时提前退出策略，通过预测最终答案首次出现的位置来学习最优退出点，从而在几乎不损失性能的前提下缓解思维链推理中的过度思考并大幅降低计算开销。 |
| [^322] | [How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities](https://arxiv.org/abs/2603.02578) | 提出 SteerEval 分层基准，从语言特征、情感和人格三个领域、三个规范层级系统评估大语言模型的可控性，发现现有引导方法在细粒度层级上的控制能力显著退化。 |
| [^323] | [Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](https://arxiv.org/abs/2602.14208) | 该论文揭示了最优批量大小调度的结构取决于任务难度——简单任务应持续增大批量，而困难任务应保持小批量直至训练后期再切换到大批量，其背后的“快速追赶”动态机制同样在大语言模型预训练中存在。 |
| [^324] | [WAXAL: A Large-Scale Multilingual African Language Speech Corpus](https://arxiv.org/abs/2602.02734) | WAXAL是一个涵盖27种非洲语言、超过14,100小时的大规模开放语音数据集，包含ASR和TTS两部分语料，实验证明在其上微调可显著提升多个先进ASR模型的性能，从而弥合语音技术中非洲语言的资源差距。 |
| [^325] | [Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics](https://arxiv.org/abs/2602.02343) | 该论文提出统一框架，将权重微调、LoRA适配和激活干预等LLM控制方法统一视为控制信号诱导的动态权重更新，并揭示了偏好与效用之间一致的权衡关系。 |
| [^326] | [Interpreting and Controlling Model Behavior via Constitutions for Atomic Concept Edits](https://arxiv.org/abs/2602.00092) | 该论文提出了一个黑盒可解释性框架，通过原子概念编辑（ACE）系统地学习一种可验证的自然语言“宪法”，建立提示词编辑与模型行为之间的因果映射，从而实现对模型行为的深入解释与有效控制。 |
| [^327] | [CD4LM: Consistency Distillation and aDaptive Decoding for Diffusion Language Models](https://arxiv.org/abs/2601.02236) | CD4LM 提出一个可组合的加速框架，通过离散空间一致性蒸馏训练轨迹不变的模型并配合自适应解码，在不牺牲准确性的前提下加速扩散语言模型的生成。 |
| [^328] | [Fidelity Before Structure: Verbatim Chunks Beat Lossy Artifact Extraction in Long-Conversation LLM Memory](https://arxiv.org/abs/2601.00821) | 在长对话LLM记忆系统中，直接存储逐字对话片段的检索准确率显著优于LLM提取的结构化信息，性能差距的根源是有损蒸馏而非结构本身。 |
| [^329] | [From Evidence to Judgment: Trace-Augmented Reasoning under Conflicting Evidence](https://arxiv.org/abs/2512.16795) | 该论文提出在RAG面临冲突证据时，通过微调模型在回答前生成结构化思考轨迹（逐篇评判文档、推断冲突模式）来增强推理，并发布了相关数据集、基准及CATS评估框架，证明该方法显著提升了模型在有依据地回答或拒绝之间的决策能力。 |
| [^330] | [Adapting Large Language Models to Low-Resource Tibetan: A Two-Stage Continual and Supervised Fine-Tuning Study](https://arxiv.org/abs/2512.03976) | 该研究提出持续预训练加监督微调的两阶段策略，将 Qwen2.5-3B 成功适配到低资源藏语，显著降低困惑度并大幅提升汉藏翻译质量，同时揭示适配主要集中在嵌入层、输出头及中后期 MLP 层。 |
| [^331] | [Guideline-Based Medical Reasoning in LLMs](https://arxiv.org/abs/2512.03838) | 该论文提出从医学共识指南（如SOFA、SAPS-II和Sepsis-3）自动生成黄金标准推理链作为监督信号，证明经过微调的小模型在医学推理任务上显著优于更大的单次提示模型，且单次提示无法可靠地产生可信赖的逐步推理。 |
| [^332] | [Readers Prefer Outputs of AI Trained on Copyrighted Books over Expert Human Writers](https://arxiv.org/abs/2510.13939) | 该研究首次通过大规模盲测评估发现，当前沿AI模型在作者完整作品上微调后，无论是专业文学读者还是普通读者都更偏好AI模拟获奖作家风格的文学输出，超越了人类专家作家，这对版权诉讼中的“替代性作品”争论具有重要启示。 |
| [^333] | [Toward LLM-Supported Automated Assessment of Critical Thinking Subskills](https://arxiv.org/abs/2510.12915) | 本文以学生议论文写作为任务载体，基于技能进阶体系开发编码量表并人工标注语料，验证了利用大语言模型（通过零样本和少样本提示等方法）自动评估批判性思维子技能的可行性。 |
| [^334] | [Beating Hateful Narratives on Social Media with Facts: Retrieval-Augmented Counter-Speech Generation](https://arxiv.org/abs/2510.12316) | 该论文提出了首个基于权威人权文献（如联合国数字图书馆、EUR-Lex等）构建的大规模知识库的检索增强生成管线，用于生成事实可靠且针对性强的反仇恨言语，以建设性地对抗社交媒体上的仇恨言论。 |
| [^335] | [Feedback Forensics: A Toolkit to Measure AI Personality](https://arxiv.org/abs/2509.26305) | 该论文提出了Feedback Forensics——一个用于显式测量和评估AI模型人格的开源工具包，以弥补现有基于人类反馈排序的不透明评估方法在衡量模型性格特质方面的不足。 |
| [^336] | [PersonaFuse: A Situation-Aware Post-Training Framework for Adaptive LLM Behavior in Enterprise Tasks](https://arxiv.org/abs/2509.07370) | PersonaFuse是一个基于特质激活理论与大五人格模型的后训练框架，通过带有特质级专家和上下文感知路由的混合专家架构，使大语言模型能够根据企业任务情境动态调整自身行为，实现无能力权衡的情境自适应表现。 |
| [^337] | [Semiotic Complexity and Its Epistemological Implications for Modeling Culture](https://arxiv.org/abs/2508.00095) | 本文提出“符号学复杂性”概念——即文本含义在不同阐释视角下的变化程度——将计算人文的建模视为从文化语言领域到计算数学领域的翻译过程，并论证缺乏翻译理论化会导致微妙而影响深远的建模错误。 |
| [^338] | [SmartThinker: Learning High-Density Reasoning via Step-Level Token Allocation](https://arxiv.org/abs/2507.04348) | 该论文提出SmartThinker，通过步骤级的token分配学习高密度推理，在保留关键推理步骤的同时压缩低价值的冗余内容，从而在严格上下文约束下实现高效推理。 |
| [^339] | [Revela: Dense Retriever Learning via Language Modeling](https://arxiv.org/abs/2506.16552) | 提出Revela框架，将检索类比为词元块之间的依赖关系学习，通过语言建模式的自监督目标训练密集检索器，从而摆脱对昂贵标注查询-文档对的依赖。 |
| [^340] | [BitBypass: A New Direction in Jailbreaking Aligned Large Language Models with Bitstream Camouflage](https://arxiv.org/abs/2506.02479) | 本文提出了一种名为BitBypass的新型黑盒越狱攻击方法，通过连字符分隔的比特流伪装技术，从数据底层的比特级信息表示入手绕过对齐大语言模型的安全防御，开辟了越狱攻击的新方向。 |
| [^341] | [Breaking the Silence: A Dataset and Benchmark for Bangla Text-to-Gloss Translation](https://arxiv.org/abs/2504.02293) | 本文构建了首个公开可用的孟加拉语文本到手语词汇翻译平行数据集及基准测试，填补了孟加拉手语研究的资源空白，并通过对比开源模型与闭源大语言模型评估了低资源场景下的翻译性能。 |
| [^342] | [ResBench: Benchmarking LLM-Generated FPGA Designs with Resource Awareness](https://arxiv.org/abs/2503.08823) | ResBench是首个关注FPGA硬件资源效率的LLM代码生成基准，通过12个类别共56个问题评估LLM生成的HDL代码，弥补了现有基准只关注功能正确性而忽视资源使用的不足。 |
| [^343] | [The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training](https://arxiv.org/abs/2502.19002) | 本文发现Transformer各模块间存在训练早期出现且贯穿全程的“锐度差异”，据此提出按各模块锐度定制学习率的Blockwise LR策略，集成到AdamW后在多种模型和数据集上实现近2倍的预训练加速并取得更低最终损失。 |
| [^344] | [3D-MoE: Towards Spatial Intelligence with Mixture-of-Experts for 3D Reasoning and Action Generation](https://arxiv.org/abs/2501.16698) | 3D-MoE通过混合专家架构与模态/空间上下文感知的概率路由实现高效3D推理，并集成Pose-DiT扩散动作头在单步内生成精确6D位姿动作，以大幅减少的激活参数在3D视觉-语言基准上取得优越性能。 |
| [^345] | [Hallucination Mitigation using Agentic AI Natural Language-Based Frameworks](https://arxiv.org/abs/2501.13946) | 该论文提出了一种基于自然语言交互的多智能体协作框架，通过四级智能体流水线检测、修正并量化AI幻觉，并创新性地提出四个关键绩效指标及总幻觉评分（THS）来系统评估幻觉缓解效果。 |
| [^346] | [Multilingual Hallucination Gaps in Large Language Models](https://arxiv.org/abs/2410.18270) | 该研究提出了“多语言幻觉差距”这一新概念，通过将FactScore指标扩展至多语言环境，并在19种语言上进行传记生成实验，揭示了大型语言模型的幻觉率因语言不同而存在显著差异，尤其是高资源语言与低资源语言之间。 |
| [^347] | [Using the Tsetlin Machine to Learn Human-Interpretable Rules for High-Accuracy Text Categorization with Medical Applications](https://arxiv.org/abs/1809.04547) | 本文首次提出基于Tsetlin机器的文本分类方法，通过学习人类可解释的简单命题规则（如合取子句），在医疗文本分类任务中同时实现高精度与可解释性。 |

# 详细

[^1]: 潜在格语言模型

    Latent Lattice Language Models

    [https://arxiv.org/abs/2611.21608](https://arxiv.org/abs/2611.21608)

    该论文提出潜在格架构，通过在每层引入沿时间维度循环运行的水平潜在流来增加模型的活跃计算状态数量，从而在不加深网络的情况下提升模型的计算容量。

    

    语言模型通常通过更深或循环的层以及潜在推理步骤来增加每个token的计算量，同时更丰富的残差连接可以改善跨层的信息流动。我们则选择增加计算容量：即模型所维持的活跃计算状态的数量。我们引入了潜在格，这是一组沿时间循环运行的逐层水平潜在流，与贯穿网络深度的垂直残差流并行运作。与超连接、mHC以及注意力或值残差类似，潜在格增强了残差流，但它并非通过深度添加或重新路由残差路径，而是提供了跨token持续存在的伴随计算状态。与混合Transformer-SSM架构不同——后者的循环状态主要用作序列混合的内部记忆——这些潜在状态直接调节垂直流的注意力、KV缓存写入、MLP计算和……

    arXiv:2611.21608v1 Announce Type: cross  Abstract: Language models typically increase per-token computation through deeper or looped layers and latent reasoning steps, while richer residual connections improve information flow across layers. We instead increase computational capacity: the number of active computational states a model maintains. We introduce the Lattice of Latents, a set of per-layer horizontal latent streams that run recurrently along time, alongside the vertical residual stream that flows through depth. Like hyper-connections, mHC, and attention or value residuals, the Lattice enhances the residual stream, but rather than adding or rerouting residual pathways through depth, it provides companion computational states that persist across tokens. Unlike hybrid Transformer-SSM architectures, whose recurrent states primarily serve as internal memory for sequence mixing, these latents directly condition the vertical stream's attention, KV-cache writes, MLP computation, and 
    
[^2]: 量化人类预报员讨论对热带气旋强度预报的附加价值

    Quantifying the added value of human forecaster discussions for tropical cyclone intensity prediction

    [https://arxiv.org/abs/2611.21602](https://arxiv.org/abs/2611.21602)

    本研究利用深度学习分析国家飓风中心的热带气旋讨论文本，量化了人类预报员经验性讨论在数值预报因子之外对热带气旋强度预报的附加价值，证明将讨论文本与SHIPS预报因子相结合可降低24小时最大持续风速预报的平均绝对误差。

    

    美国国家飓风中心（NHC）定期发布热带气旋讨论（Tropical Cyclone Discussions），为其热带气旋预报提供深入见解。本研究考察了这些讨论在一组数值预报因子之外对强度预报的附加价值。针对2002-2023年飓风季中大西洋和东北太平洋的637个热带气旋，我们分析了讨论文本及其与来自SHIPS的13个飓风环境预报因子和ERA5 10米风场的组合，采用三种建模方法预报24至120小时时效的强度变化。在其中一种方法中，面向文本的端到端深度学习框架表明，仅凭讨论文本在24小时强度预报上就能提供与SHIPS预报因子相当的价值。在五个此类模型的集合预报中，将讨论信息加入SHIPS预报因子后，2022-2023年24小时最大持续风速预报的平均绝对误差得以降低。

    arXiv:2611.21602v1 Announce Type: cross  Abstract: The National Hurricane Center routinely releases Tropical Cyclone Discussions that offer insights into its tropical cyclone forecasts. This study examines the added value of these discussions for intensity forecasting beyond a set of numerical predictors. For 637 Atlantic and eastern North Pacific tropical cyclones from the 2002-2023 hurricane seasons, we analyze the discussion text and its combination with 13 hurricane environmental predictors from SHIPS and ERA5 10-meter wind fields, for forecasting intensity changes at lead times of 24 to 120 hours using three modeling methods. In one approach, an end-to-end deep learning framework for the text shows that the discussions alone offer comparable value for 24-hour intensity forecasts to the SHIPS predictors. Adding discussion information to the SHIPS predictors in an ensemble of five such models reduces the mean absolute error of 24-hour maximum sustained wind forecasts for 2022-2023 c
    
[^3]: 视觉语言模型如何规划下一步

    How VLMs Plan What's Next

    [https://arxiv.org/abs/2611.21600](https://arxiv.org/abs/2611.21600)

    本文揭示了视觉语言模型通过“绑定头匹配—进度光标—区域选择—注视头执行”四个阶段来规划描述图像时的观察与输出顺序，并利用因果补丁实验逐一验证了各环节的作用。

    

    当视觉语言模型描述一幅图像时，它会逐个命名对象，并且很少重复自己，然而模型如何规划这种顺序却远非显而易见。我们刻画了视觉语言模型在“看”之前决定“看哪里”的机制。该机制有四个主要部分：首先，绑定头将相关文本词元与图像进行匹配，并将其解析为指向匹配图像区域的指针；其次，同样的绑定头将每个已写出的答案与其来源区域绑定，模型将其转化为一个进度信号——一个光标，即列表已经写到图像中的哪个位置；第三，选择步骤选取光标之后最近的未写区域，或者停止；最后，注视头执行所选指针并返回该处所绘内容的概念，语言头和后期多层感知机将其转化为用问题语言表达的一个词。我们通过因果补丁实验验证了该机制的每个部分。

    arXiv:2611.21600v1 Announce Type: cross  Abstract: When a vision-language model describes an image, it names one object at a time and it rarely repeats itself, yet how the model plans this order is far from obvious. We characterize the mechanism a VLM uses to decide where to look before it looks. The mechanism has four main parts. First, binding heads match the relevant text tokens against the image and resolve them into pointers to the matched image region. Second, the same binding heads bind each written answer to the region it came from, and the model turns it into a progress signal: a cursor, the position in the image up to which the list has been written. Third, a select step chooses the nearest unwritten region after the cursor, or stops. Finally, gaze heads execute the chosen pointer and return a concept of what is drawn there, which language heads and the late MLPs turn into a word in the language of the question. We validate each part of this mechanism through causal patching 
    
[^4]: 超越职位头衔：利用稀疏自编码器与语义聚类解读早期职业招聘广告

    Beyond Job Titles: Interpreting Early-Career Job Advertisements with Sparse Autoencoders and Semantic Clustering

    [https://arxiv.org/abs/2611.21586](https://arxiv.org/abs/2611.21586)

    该研究提出基于PERC流程（预处理、嵌入、降维、聚类）并结合稀疏自编码器与语义聚类的方法，对36,081条美国早期职业招聘信息进行分析，超越了职位头衔本身来揭示技术岗位在工作内容与行业应用上的真实差异。

    

    仅凭职位头衔和职位描述，难以充分洞察瞬息万变的就业市场中技术类岗位之间的差异。为评估这些岗位在工作内容和行业应用方面的不同，我们基于深度学习和Transformer表示设计并比较了两种方法，使用了2025年12月至2026年10月期间从美国LinkedIn收集的36,081条早期职业招聘信息。我们首先应用PERC方法，即预处理、嵌入、降维和聚类。我们使用GPT-6 Luna将职位描述段落分类为团队背景、角色任务和任职资格三个部分，然后创建这些部分的组合，并检验对分段文本进行嵌入和聚类是否能产生关于不同岗位角色的有意义语义簇。我们使用最大均值差异（MMD）对嵌入分布之间的观测差异进行排序，并采用预训练的稀疏自编码器……

    arXiv:2611.21586v1 Announce Type: new  Abstract: Job titles and job descriptions alone provide limited insight into how technical roles differ in a changing job market. To assess how such roles differ in their work and industry applications, we design and compare two methods based on deep learning and transformer representations using 36,081 early-career job postings collected from LinkedIn in the United States between December 2025 and October 2026.   We first apply PERC, short for Preprocessing (P), Embedding (E), Dimensionality Reduction (R) and Clustering (C). We employ GPT-6 Luna to classify job description passages into Team context, Role task and Qualifications. We then create combinations of these sections and examine whether embedding and clustering the segmented passages produce meaningful semantic clusters about different job roles. We use Maximum Mean Discrepancy (MMD) to rank observed differences between embedding distributions and employ a pretrained sparse autoencoder (S
    
[^5]: 阅读还是记忆？测量语言模型中的时间污染

    Reading or Remembering? Measuring Temporal Contamination in Language Models

    [https://arxiv.org/abs/2611.21554](https://arxiv.org/abs/2611.21554)

    论文发现语言模型在基于央行文本进行经济预测时，其表现可能部分源于对“未来”结果的记忆（时间性测试集污染），并通过与仅使用过去数据训练的无记忆模型对比，证明现代模型确实能回忆起未来的经济结果，且传统的随机基线并不适用于评估这种污染。

    

    语言模型如今被用于阅读央行文本以进行经济预测，但大多数模型的训练文本写于其所阅读文档之后。一个阅读2021年FOMC声明的模型可能已经知道2022年通胀上升，因此其表面上的部分能力可能只是对结果的记忆——这是测试集污染的一种时间性形式。我们在2000年至2026年的FOMC声明和会议纪要上测量这种记忆，方法是将现代开源模型与时间上一致的模型（即仅在固定日期之前的文本上训练的模型）进行比较。当仅给定日期询问过去的GDP结果时，无法知晓答案的模型仅凭对日期和年代的先验仍能达到高达0.64的AUC，因此随机水平是错误的基线。在相同问题上与这些无记忆模型相比，大多数现代模型能回忆起哪些年份有强劲增长。成员推断分数是按时代和文档类型对文档进行排序，而非按模型是否见过它们来排序……

    arXiv:2611.21554v1 Announce Type: cross  Abstract: Language models are now used to read central-bank text for economic forecasting, but most were trained on text written after the documents they read. A model reading a 2021 FOMC statement may already know that inflation rose in 2022, so part of its apparent skill can be memory of the outcome, a temporal form of test-set contamination. We measure this memory on FOMC statements and minutes from 2000 to 2026 by comparing modern open models with chronologically consistent models, which are trained only on text from before a fixed date. Asked about past GDP outcomes given only a date, models that cannot know the answer still reach an AUC of up to 0.64 from priors on dates and eras, so chance is the wrong baseline. Against these memory-free models on the same questions, most modern models recall which years had strong growth. Membership-inference scores sort documents by era and document type rather than by whether a model saw them, as exact
    
[^6]: 理由正确，建议无效：评估大语言模型对被拒绝请求解释中的补救有效性

    Right Reason, Wrong Advice: Evaluating Recourse in LLM Explanations of Denied Requests

    [https://arxiv.org/abs/2611.21545](https://arxiv.org/abs/2611.21545)

    该论文提出“补救有效性”这一新概念，指出当多条规则独立拒绝请求时，LLM生成的解释即使引用了真实原因也无法带来有效补救，而常规的正确性检查无法发现这一缺陷，只有询问强评判者“照解释行动是否可行”才能识别该问题。

    

    语言模型越来越多地承担了在规则系统拒绝用户请求时撰写解释的任务。如果多条规则独立地拒绝了该请求，那么解释即使指出了一个真实的原因，用户修复该原因后请求仍会被拒绝。我们将这种缺失的属性称为“补救有效性”：在不改变所请求内容的前提下，改变解释所引用的原因，就能改变决策结果。我们对这一属性进行了精确计算，并通过命中集论证，仅凭策略本身就能预测一个正确的原因何时会失效。在从访问控制和美国福利资格制度移植的七个规则系统中，有六个系统在多条原因同时适用时，单一正确的原因几乎从不提供有效的补救。在20个LLM上的实验表明，无论是通过计算还是询问LLM评判者来核查所引用原因是否正确，在发现这种失效方面都接近随机水平，正如该论证所预测的。而一个被询问“按照解释行动是否可行”的强大评判者则能够发现这种失效。更强和具备推理能力的模型表现……（摘要原文在此处截断）

    arXiv:2611.21545v1 Announce Type: new  Abstract: Language models increasingly write the explanations people receive when a rule system denies their request. If several rules deny the request independently, an explanation can name a true reason and still leave the person denied after they fix it. We call the missing property recourse validity: changing what the explanation cites, without changing what was asked for, can flip the decision. We compute it exactly, and a hitting-set argument predicts from the policy alone when a correct reason fails. In six of seven rule systems ported from access control and US benefit eligibility, a single correct reason almost never gives recourse when several reasons apply. Across 20 LLMs, checks of whether the cited reasons are correct, computed or asked of an LLM judge, are close to chance at spotting this failure, as the argument predicts. A strong judge asked whether acting on the explanation would work does spot it. Stronger and reasoning models do
    
[^7]: 看见却未言说：定位与修复视觉-语言模型中的跨图像比较

    Seen but Unsaid: Locating and Repairing Cross-Image Comparison in Vision-Language Models

    [https://arxiv.org/abs/2611.21539](https://arxiv.org/abs/2611.21539)

    该论文发现视觉-语言模型在跨图像比较任务上的失败是因为比较信息存在于视觉token和隐藏状态中却未传递到最终答案，通过定位一小部分承载比较功能的注意力头，提出了仅修改0.02%参数的修复方法。

    

    视觉-语言模型（VLMs）在跨两张图像匹配一个点或一个物体时常常失败，即使答案可以从它们自身的隐藏状态中解码出来。我们在Qwen3-VL（2B-32B）、InternVL3.5-8B和Gemma-3-12B上，针对合成形状匹配、同实例点对应（HPatches）和语义对应（SPair-71k）任务，从视觉token追踪到答案字母，发现了两个不同的失败点、一小部分承载比较功能的注意力头，以及一个仅涉及0.02%参数的修复方法。两个失败点：精确副本被比较但未被读取出来——在所有模型家族中，答案位置探针都比校准后的输出高出15到25个百分点，无论形状是否可命名都是如此，且沿探针方向引导所带来的答案变化不比随机方向多；语义对应存在于视觉token中但从未到达答案。一条通路：重新分配cro…（摘要截断）

    arXiv:2611.21539v1 Announce Type: cross  Abstract: Vision-language models (VLMs) often fail to match a point or an object across two images, even when the answer can be decoded from their own hidden states. We trace this failure from the visual tokens to the answer letter, in Qwen3-VL (2B-32B), InternVL3.5-8B, and Gemma-3-12B, on synthetic shape matching, same-instance point correspondence (HPatches), and semantic correspondence (SPair-71k), and find two distinct failure points, a small set of attention heads that carries the comparison, and a repair that touches 0.02% of the parameters. Two failure points: exact copies are compared but not read out. An answer-position probe exceeds the calibrated output by 15 to 25 points in every family, for nameable and unnameable shapes alike, and steering along the probe's direction moves no more answers than random directions. Semantic correspondences are present in the visual tokens but never reach the answer. One pathway: redistributing the cro
    
[^8]: 乐观的验证器被攻破：精确过程预言机下的奖励作弊

    Optimistic Verifiers Get Hacked: Reward Hacking Under an Exact Process Oracle

    [https://arxiv.org/abs/2611.21538](https://arxiv.org/abs/2611.21538)

    该论文构建了一个过程级真实奖励可被精确计算的授权策略环境，通过沿七个现实维度退化验证器进行训练实验，揭示了四种奖励作弊表型，其中包括乐观验证器被填充手段攻破——代理分数升至0.940而真实忠实度却从0.731下降。

    

    基于可验证奖励的强化学习会继承验证器本身的错误。在数学和代码领域之外，验证器只是一个代理、评分标准、学习得到的评判者或手写检查器，其与真实目标之间的差距从未被测量，因此奖励的上升无法与作弊行为区分开来。我们构建了一个在过程层面可以精确计算真实奖励的实验设置：覆盖124,416个上下文的可执行授权策略配有一个预言机，能够列出每个决策的所有最小充分理由。我们将这些策略编译为精确的过程奖励，沿七个现实维度退化验证器（仅结果、充分性、仅召回、提取噪声、粗粒度分类、过时策略、LLM评判者），并在精确验证器为每次rollout评分的情况下针对每个退化版本进行训练。实验中出现了四种表型。乐观的验证器会通过填充被攻破：充分性代理最终达到0.940，而真实忠实度却从0.731下降

    arXiv:2611.21538v1 Announce Type: cross  Abstract: Reinforcement learning from verifiable rewards inherits whatever its verifier gets wrong. Outside mathematics and code the verifier is a proxy, a rubric, a learned judge or a hand-written checker, and its distance from the true objective is never measured, so a rising reward cannot be told apart from a hack. We build a setting where the true reward is exactly computable at the process level: executable authorization policies over 124,416 contexts admit an oracle that lists every minimal sufficient reason for every decision. We compile these policies into exact process rewards, degrade the verifier along seven realistic axes (outcome-only, sufficiency, recall-only, extraction noise, coarse taxonomy, stale policy, LLM judge), and train against each while the exact verifier scores every rollout. Four phenotypes appear. Optimistic verifiers are hacked by padding: the sufficiency proxy ends at 0.940 while true faithfulness falls from 0.731 
    
[^9]: 评估大语言模型在西班牙语政治话语中的文化推理能力与稳健性

    Evaluating Cultural Reasoning and Robustness in LLMs for Spanish Political Discourse

    [https://arxiv.org/abs/2611.21535](https://arxiv.org/abs/2611.21535)

    本文提出一种结合三重独立人工标注视角与基于情感翻转反事实构建合成子集的诊断协议，用以评估大语言模型在西班牙语政治话语中文化敏感的情感分析任务上的稳健性。

    

    大语言模型（LLM）正日益被用于解读、生成和调解面向公众的语言，这使得它们的可靠性在政治敏感场景中尤为重要。尽管情感分析是该领域的常见任务，但社交媒体话语中的情感含义往往通过植根于特定文化、语用上含蓄间接的线索来传达，而仅以标签为中心的评估方式难以充分捕捉这些线索。在本文中，我们评估了大语言模型对社交媒体上西班牙语政治话语进行情感分析的稳健性，重点关注植根于文化的情感解读。我们提出了一种诊断协议，该协议将三个独立的人工标注视角整合为金标准子集和歧义子集，并通过基于模型条件困惑度和“LLM作为裁判”标准筛选出的情感翻转反事实样本，构建了一个合成子集以扩展语料库。利用该框架，我们对……（原文摘要在此处截断）

    arXiv:2611.21535v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to interpret, generate, and mediate public-facing language, which makes their reliability especially consequential in politically sensitive settings. Although sentiment analysis is a common task in this area, affective meaning in social media discourse is often conveyed through culturally situated and pragmatically indirect cues that are not well captured by label-centric evaluation alone. In this paper, we evaluate the robustness of LLMs for sentiment analysis of Spanish political discourse from social media, with emphasis on culturally situated affective interpretation. We introduce a diagnostic protocol that combines three independent human annotation views into Gold and Ambiguous subsets, and extends the corpus with a Synthetic subset built from sentiment-flipped counterfactuals selected through model-conditioned perplexity and LLM-as-a-judge criteria. Using this framework, we compar
    
[^10]: 无判断的程序：小型语言模型在精确验证器下能学到什么

    Procedure without Judgment: What Small Language Models Learn under an Exact Verifier

    [https://arxiv.org/abs/2611.21529](https://arxiv.org/abs/2611.21529)

    该论文利用可精确计算的最小充分原因构建了无误差验证器下的策略解释任务，发现前沿模型仅凭查询工具约一半时间能找到精确解释，而仅40亿参数的小模型在2000条算法轨迹上微调后即可达到92%的准确率。

    

    当授权策略阻止智能体的工具调用时，审计日志应当列出导致该决定的事实，且仅限于此。对于定义在可枚举上下文空间上的可执行策略，这个最小充分原因可以被精确计算，因此对解释的评分可以直接对照真实答案，而非其代理指标。我们利用这种精确性来探究：模型能从一个完全没有错误的验证器中学到什么。解释器只能看到一个请求和一个决定，永远看不到策略本身，它拥有三十轮交互机会，可以在其中对修改过的请求调用策略，并询问某个候选事实集合是否充分。前沿模型仅凭这些工具就能在大约一半的情况下找到精确原因，而一句告诉它们缩减答案的提示语使 Haiku 4.5 的成绩从 58% 提升到 75%。在删除算法的 2000 条轨迹上微调的 40 亿参数模型，在其训练所用策略上达到了 92% 的准确率。起决定作用的是教师的类型：在数据量相同的情况下，该算法（原文在此处截断）……

    arXiv:2611.21529v1 Announce Type: new  Abstract: When an authorization policy blocks an agent's tool call, the audit log should name the facts that forced the decision and nothing more. For an executable policy over an enumerable context space this minimal sufficient reason can be computed exactly, so an explanation is scored against ground truth rather than a proxy for it. We use that exactness to ask what a model can learn from a verifier that has no error in it. The explainer sees one request and one decision, never the policy, and has thirty turns in which it may call the policy on altered requests and ask whether a candidate set of facts is sufficient. Frontier models find the exact reason about half the time with the tools alone, and one sentence telling them to shrink the answer lifts Haiku 4.5 from 58% to 75%. A 4B model fine-tuned on 2000 traces of the deletion algorithm reaches 92% on the policy it was trained on. Teacher kind is what moves this: at matched data size the algo
    
[^11]: 基于风险约束强度分配的视觉-语言智能体高效检测水印方法

    Detection-Efficient Watermarking for Vision--Language Agents via Risk-Constrained Strength Allocation

    [https://arxiv.org/abs/2611.21488](https://arxiv.org/abs/2611.21488)

    该论文提出一种风险感知的语义水印框架，通过结合预测不确定性与零样本多模态相关性风险代理在解码时动态分配水印强度，在保持高检测准确率（ROC-AUC达0.955）的同时将平均水印强度降低约59.4%并显著减少文本质量损失，且无需重新训练模型即可灵活调节检测与质量之间的权衡。

    

    视觉-语言智能体生成的描述、决策和指令可能会被用户、下游智能体和自动化工作流重复使用，因此需要能够支持内容真实性与部署后审计的溯源机制，同时避免不必要地降低响应质量。我们提出了一种风险感知的语义水印框架，该框架将预测不确定性与零样本多模态相关性风险代理相结合，在解码过程中动态分配水印强度。单一的风险惩罚参数 λ 提供了一个连续的操作前沿，使得可以在部署时选择检测-质量权衡方案，而无需重新训练模型。在选定的操作点上，平均水印强度相对于固定强度水印降低了约59.4%，同时困惑度变化从0.3634降至0.1689，原始/改写文本的ROC-AUC分别达到0.9552/0.9459。匹配强度分析表明，这一优势……（原文此处截断）

    arXiv:2611.21488v1 Announce Type: cross  Abstract: Vision--language agents generate descriptions, decisions, and instructions that may be reused by users, downstream agents, and automated workflows, creating a need for provenance mechanisms that support content authenticity and post-deployment auditing without unnecessarily degrading response quality. We present a risk-aware semantic watermarking framework that combines predictive uncertainty with a zero-shot multimodal relevance-risk proxy to allocate watermark strength during decoding. A single risk penalty $\lambda$ exposes a continuous operating frontier, allowing deployment-time selection of a detection--quality trade-off without model retraining. At a selected operating point, mean watermark strength is reduced by approximately 59.4\% relative to fixed-strength watermarking, while perplexity change falls from 0.3634 to 0.1689 and clean/paraphrased ROC-AUC reaches 0.9552/0.9459. A matched-strength analysis shows that this advantag
    
[^12]: Jetriever：将智能体记忆检索从相似度转向效用

    Jetriever: Shifting Agent Memory Retrieval from Similarity to Utility

    [https://arxiv.org/abs/2611.21478](https://arxiv.org/abs/2611.21478)

    Jetriever 用 Jev 决策分数对全部存储记忆进行效用评分排序，取代传统相似度检索，使智能体在回答问题时能将最有用的记忆准确地排在最前。

    

    一条有用的记忆未必与它所帮助回答的问题相似。Jetriever 依据 Jev 的决策分数对每条存储记忆对当前问题的有用性进行排序，用全池效用评分取代相似度搜索。在 InMind 的 125 个间接查询任务上，它在 84.8% 的情况下将黄金交易记录排在首位；而以用户轮次建立索引的对照方法 BM25 和 MiniLM 分别仅达到 1.6% 和 0.0%。一项包含 20 个问题的对比实验将“找到证据”与“对证据排序”区分开来：Jev 和 Qwen3.5 9B 各自能在前 50 名内找到 90% 的黄金交易记录，而 Laya 仅找到 30%；在排名第一的位置上，Jev 达到 85%，Qwen3.5 9B 为 40%。在对 MemoryAgentBench 问题按维度等比例随机抽取的 60 个问题上，Jetriever 的四维度得分为 55.4%，而 ReFind 在相同问题上的得分为 44.3%。Jev 检索在 InMind 上平均每个问题花费 0.00834 美元，在 MAB 上平均每个问题花费 0.03103 美元。这些结果表明了世界知识的价值，以及能够将决定性记忆稳定排在前列的评分机制的作用。

    arXiv:2611.21478v1 Announce Type: new  Abstract: A useful memory need not resemble the question it helps answer. Jetriever ranks each stored memory by Jev's decision score for its usefulness to the current question, replacing similarity search with full-pool utility scoring. On InMind's 125 indirect-query tasks, it places the gold exchange first in 84.8% of cases; matched BM25 and MiniLM controls reach 1.6% and 0.0% when indexing user turns. A 20-question comparison separates finding evidence from ranking it: Jev and Qwen3.5 9B each find 90% of gold exchanges by rank 50, while Laya finds 30%; at rank one, Jev reaches 85% and Qwen3.5 9B 40%. On a dimension-proportional random sample of 60 MemoryAgentBench questions, Jetriever's four-dimension score is 55.4%, against 44.3% for ReFind on the same questions. Jev retrieval costs USD 0.00834 per InMind question and USD 0.03103 per MAB question on average. These results show the value of world knowledge, scores that rank decisive memories fir
    
[^13]: 面向高效多模态大语言模型的平衡孪生权重因子分解

    Balanced Twin Weight Factorization for Efficient MLLMs

    [https://arxiv.org/abs/2611.21459](https://arxiv.org/abs/2611.21459)

    提出平衡孪生权重因子分解（BTWF），利用共享基因子分解中孪生层输入相同的特性，只需执行一次共享前向传播，从而在降低推理延迟的同时提升模型学习能力。

    

    权重因子分解方法旨在降低模型的延迟、内存占用，或两者兼而有之。这类方法通常将一个预训练模型分解为两个或多个因子，以更少的参数量和浮点运算次数（FLOPs）来近似原始模型权重。然而在实践中，由于用多个较小的权重矩阵替换单个大权重矩阵会引入额外开销，WF方法难以将更少的FLOPs转化为实际延迟的降低。为解决这一问题，我们提出了平衡孪生权重因子分解，该方法利用因子化模型的独特结构，在降低延迟的同时提升学习能力。具体而言，BTWF在共享基WF的设置下运行，即将两个或多个层分解为一个共享基矩阵和一个层专属的混合矩阵。如果这些共享层具有相同的输入（即互为“孪生层”），那么我们只需执行一次共享前向传播，从而减少推理时间。

    arXiv:2611.21459v1 Announce Type: cross  Abstract: Weight Factorization (WF) methods aim to reduce a model's latency, memory, or both. These typically decompose a pretrained model into two or more factors that approximate the original model weights with fewer parameters and floating point operations (FLOPs). However, translating fewer FLOPs into reduced latency is challenging for WF methods in practice due to overhead introduced by replacing a single large weight matrix with multiple smaller ones. To address this, we propose Balanced Twin Weight Factorization (BTWF), which uses the unique structure of factorized models to simultaneously lower latency and increase learning capacity. Specifically, BTWF operates in the shared basis WF setting, where two or more layers are decomposed into a shared basis matrix and a layer-specific mixing matrix. If these shared layers have the same input (i.e., are twins), then we need only perform this shared forward pass once, reducing inference time. Ho
    
[^14]: 一种用于建模标注任务中人口统计群体分歧的可扩展多任务学习框架

    A Scalable Multi Task Learning Framework for Modeling Demographic Disagreements in Annotation Tasks

    [https://arxiv.org/abs/2611.21458](https://arxiv.org/abs/2611.21458)

    该论文提出了一种利益相关者感知的联合多任务学习框架（SAJ-MTL），通过条件梯度路由和可自定义的群体加权机制，联合建模标注者群体间及群体内的人口统计分歧，从而显著提升毒性检测等任务中的群体级性能。

    

    来自不同人口统计背景的标注者由于不同的生活经历、文化背景以及与内容的关系，往往会以系统性的方式产生分歧。当这种分歧发生时，某一群体的观点有时值得被优先考虑，因为他们可能拥有认知上的优势立场，或对结果有更大的利害关系。例如，在评估社交媒体帖子的毒性时，优先考虑被针对群体的观点可能非常重要。我们提出了一个利益相关者感知的联合多任务学习（SAJ-MTL）框架，以联合建模标注者群体之间以及群体内部的人口统计分歧。该模型使用条件梯度路由来处理由群体条件标注引起的稀疏性问题，并且其参数化设计支持在不同任务设置中对不同群体进行自定义加权。SAJ-MTL 通过对上下文的更好理解，提升了毒性检测任务中的群体级性能。

    arXiv:2611.21458v1 Announce Type: new  Abstract: Annotators from different demographic backgrounds often disagree in systematic ways due to varying lived experiences, cultural contexts, and their relationship to the content. When such disagreement occurs, one group's perspective may sometimes warrant prioritization because they hold an epistemically advantaged standpoint or bear greater stake in the outcome. For example, when assessing toxicity of posts on social media, it may be important to prioritize a targeted group's viewpoint. We propose a Stakeholder-Aware Joint Multi-Task Learning (SAJ-MTL) framework to jointly model demographic disagreement both across- and within- annotator groups. The model uses conditional gradient routing to handle sparsity induced by group-conditional labeling, and parameterization supports customized weighting of groups in different task settings. SAJ-MTL improves group-level performance in toxicity detection through better context-aware understanding of
    
[^15]: 复现与扩展权重空间编程：前沿大语言模型将程序编译为权重并读取输入层

    Replicating and Extending Weight-Space Programming: Frontier LLMs Compile Programs into Weights and Read Input Layers

    [https://arxiv.org/abs/2611.21449](https://arxiv.org/abs/2611.21449)

    本研究复现并扩展了MIT的发现——前沿大语言模型能够直接写出小型神经网络的权重、读取权重以理解网络的功能，且模型写出的网络表现得如同被“编译”成权重的程序。

    

    MIT的研究人员最近报告称，前沿大语言模型（LLM）GPT-6 Astra能够在单次回复中直接写出小型神经网络的权重，并能读取小型网络的原始权重来说出它们被训练来做什么。我们在为期三天的冲刺中复现并扩展了这些结果。我们的四项发现：(1) 写权重这一结果可以复现。Astra为游戏Breakout（打砖块）生成的网络得分分别为514、550和307（MIT的结果为517），并且在两个修改版Breakout上同样表现良好。Claude Opus 5.5也能写出可正常工作的网络，但常常会耗尽输出token。(2) 真正新颖之处在于将程序翻译为权重。更早一代的模型在能够写出网络权重之前，就已能以代码形式写出良好的策略，而且模型写出的网络表现得如同编译后的程序：两个模型独立地将Astra的Breakout网络反编译为Python代码，在所有测试状态下都与其实际行为相符。在分类（任务上）……（摘要原文在此处截断）

    arXiv:2611.21449v1 Announce Type: cross  Abstract: Researchers at MIT recently reported that GPT-6 Astra, a frontier large language model (LLM), can write the weights of small neural networks directly in a single reply, and can read the raw weights of small networks to say what they were trained to do. In a three-day sprint, we replicated and extended these results. Our four findings: (1) Writing weights reproduces. Astra's networks for the game Breakout scored 514, 550 and 307 (versus MIT's 517), and scored just as well on two modified versions of Breakout. Claude Opus 5.5 can also write working networks, but often runs out of output tokens. (2) What is new is translating programs into weights. Earlier models could write a good policy as code a generation before they could write network weights, and the networks the models write behave like compiled programs: two models independently decompiled Astra's Breakout network into Python matching its actions in all tested states. On classify
    
[^16]: 直接排序就行！关于秩一反循环编辑背后的线性规划的注释

    Just sort it! A note on the LP behind rank-one anti-loop edits

    [https://arxiv.org/abs/2611.21432](https://arxiv.org/abs/2611.21432)

    本文发现一个简单的排序方法——选取激活变异系数最小的8个神经元均分推力——可以替代稀疏线性规划来实现秩一反循环编辑，且性能不劣于LP方法。

    

    权重空间编程通过对MLP输出矩阵进行秩一编辑 W' = W + uv^T 来消除小型推理模型的思考循环，其中u是循环方向，v是基于神经元激活的稀疏线性规划（LP）的解。我们研究是否真的需要这个LP。我们用排序来替代它：选取激活变异系数最小的8个神经元，并将LP解所实现的推力均分给它们。在Qwen3.5-0.8B上，排序编辑在1000道题目（MATH-500和GSM8K）上得分为67.0%，而LP编辑为67.4%，未编辑模型为9.5%；配对差异为-0.4分，95%置信区间为[-2.9, +2.1]，在实验前预先设定的5分非劣效性界限之内。在另外三个采样种子上，当不使用LP来设定剂量时，差异分别为-0.6分（区间[-2.0, +0.8]）和-0.8分（区间[-2.3, +0.6]）。排序并不能解决……

    arXiv:2611.21432v1 Announce Type: cross  Abstract: Weight-space programming removes the thinking loops of a small reasoning model by a rank-one edit W' = W + u v^T of an MLP output matrix, where u is a loop direction and v is the solution of a sparse Linear Program (LP) over the neuron activations. We investigate whether the LP is needed. We replace it by a sort: we take the 8 neurons whose activation has the smallest coefficient of variation, and give them equal shares of the push that the LP solution realizes. On Qwen3.5-0.8B the sorted edit scores 67.0% on 1000 problems (MATH-500 and GSM8K) against 67.4% for the LP edit and 9.5% for the unedited model; the paired difference is -0.4 points, with 95% confidence interval [-2.9, +2.1], inside a non-inferiority margin of 5 points declared before the runs. On three further sampling seeds the difference is -0.6 points, interval [-2.0, +0.8], and -0.8 points, interval [-2.3, +0.6], when the dose is set without the LP. The sort does not solv
    
[^17]: 基于评分量规的大语言模型集成自动作文评分：使用大语言模型集成对学生在批判性在线推理中的回答进行评分

    Rubric-Based AES with LLM Ensembles: Scoring of Student Responses in Critical Online Reasoning Using Ensembles of LLMs

    [https://arxiv.org/abs/2611.21422](https://arxiv.org/abs/2611.21422)

    该研究提出基于评分量规、通过构建大语言模型集成来对学生在经济学和医学教育中的批判性在线推理回答进行自动评分，并验证了LLM集成有望达到与人类评分者相当的一致性水平。

    

    对学生的回答进行评分成本高昂且耗时。此外，人工评分者容易受到偏见、疲劳和评分漂移的影响，这进一步限制了评分的可靠性。为降低这些成本并解决与偏见相关的问题，人们已采用多种方法开发自动作文评分（AES）系统，这些方法从早期基于文本特征的系统发展到当前的机器学习和基于Transformer的模型。特别是，大语言模型（LLM）为简化AES系统的开发提供了潜力。然而，它们也存在众所周知的局限性，例如幻觉和对提示词的敏感性。这就需要探索一种混合方法的潜力，即通过构建集成来利用各个LLM之间的协同作用。我们在经济学和医学教育领域，针对103项评分标准，评估了大语言模型集成在给学生回答评分时达到与人类评分者一致性的潜力。

    arXiv:2611.21422v1 Announce Type: new  Abstract: Grading student responses is costly and time-consuming. Moreover, human annotators are prone to bias, fatigue, and rater drift, which further limits grading reliability. Various methods have been employed to develop automated essay scoring (AES) systems to reduce these costs and address bias-related issues. These methods range from early text-feature-based systems to current machine-learning and transformer-based models. Specifically, large language models (LLMs) offer the potential to streamline the development of AES systems. However, they are subject to well-known limitations, such as hallucinations and prompt sensitivity. This creates a need to explore the potential of a mixed-methods approach that leverages the synergies among individual LLMs through ensemble formation. We evaluate the potential of ensembles of LLMs to match human-rater agreement when grading student responses in economics and medical education across 103 criteria. 
    
[^18]: AuXSpec：通过可执行后置条件学习程序规范的自动形式化

    AuXSpec: Learning to Autoformalize Program Specifications via Executable Postconditions

    [https://arxiv.org/abs/2611.21421](https://arxiv.org/abs/2611.21421)

    AuXSpec 通过在 Lean 中将声明式后置条件与可执行对应版本配对，使测试用例的执行一致性成为可扩展的强化学习奖励，从而低成本地训练程序规范的自动形式化模型。

    

    形式化规范定义了程序的预期行为，并构成了程序验证的基础，但在诸如 Lean 这样的定理证明器中编写它们既缓慢又需要专业知识，这使得规范编写成为规模化产出形式正确代码的主要瓶颈。从自然语言问题出发的自动形式化很有前景，但训练规范模型需要对生成的规范是否捕捉了预期行为提供可扩展的反馈。先前的方法通过显式证明或带有形式对应性证明的单独实现代码来获得这种反馈，两者代价都很高。我们提出了 AuXSpec，这是一个 Lean 框架，它将声明式后置条件与可执行对应版本配对——在可能时通过类型类解析获得可执行版本，否则单独定义并提供等价性证明。随后，测试用例可以通过执行来评估声明式规范，使我们能够以规范间的一致性作为强化学习的奖励。利用 AuXSpec，我们合成了……

    arXiv:2611.21421v1 Announce Type: cross  Abstract: Formal specifications define intended program behavior and underpin verification, but writing them in theorem provers such as Lean is slow and expertise-intensive, making specification a major bottleneck to scaling formally correct code. Autoformalization from natural-language problems is promising, yet training specification models requires scalable feedback on whether generated specifications capture intended behavior. Prior approaches obtain this feedback through explicit proofs or separately implemented code with formal correspondence proofs, both expensive. We introduce AuXSpec, a Lean framework that pairs declarative postconditions with executable counterparts, obtained through type class resolution when possible and otherwise defined separately with an equivalence proof. Test cases can then evaluate declarative specifications by execution, letting us use agreement as a reinforcement-learning reward. Using AuXSpec, we synthesize 
    
[^19]: 嵌入图距离能引导大语言模型的构思吗？针对后续发展的四项检验

    Do Embedding-Map Distances Guide LLM Ideation? Four Tests Against What Came Next

    [https://arxiv.org/abs/2611.21408](https://arxiv.org/abs/2611.21408)

    本文通过对后续实际出现内容的四项检验，证明嵌入图中远离现有工作的距离并不能引导LLM进行真正新颖的构思——远点方向无法解码为真实文本、LLM倾向于复述邻近已知工作、基于密度的空白得分预测未来增长的能力不超过随机水平。

    

    一种流行的构思配方将现有工作的嵌入图视为指南：选择远离所有现有点的位置，并将其解码为创意。我针对后来实际出现的内容，检验了这类距离引导构思的四种方式。(1) 远处的随机方向并非文本：没有后续条目与它们的原始余弦相似度超过约0.10（经过中心化或白化处理后，24个单元格中有22个仍然无法企及），而且当仅以分数作为反馈、向LLM解码器展示目标点的邻近文献时，它在全部六个目标集合中都直接写出了这些邻近内容。一种不含邻居信息的搜索在与内部空白区域的中位数余弦相似度0.57和0.47处写出连贯文本，在20个案例中分别有15个和14个超过最近的现有条目，但与远点的相似度仅为0.07和0.06。(2) 在仅基于过去数据构建的地图的k-means单元格上，登记的空洞得分主要追踪的是单元格的大小和位置；三种基于密度的空白得分预测哪些单元格增长最快的能力处于机会水平或更低（AUC为0.30至0.66），并且在Steam和arXiv数据上对动量没有附加预测贡献。

    arXiv:2611.21408v1 Announce Type: cross  Abstract: A popular ideation recipe treats an embedding map of existing work as a guide: pick points far from everything and decode them into ideas. I test four ways such distances could guide ideation against what appeared later. (1) Far random directions are not text: no later item reaches a raw cosine above about 0.10 with one (after centring or whitening, 22 of 24 cells stay out of reach), and an LLM decoder shown a target's neighbours, with score-only feedback, writes those neighbours in all six target sets. A neighbour-free search writes coherent text at median cosine 0.57 and 0.47 to interior gaps, beating the nearest existing item on 15 and 14 of 20, but only 0.07 and 0.06 to far points. (2) On k-means cells of a past-only map, the registered hole score mostly tracks cell size and position; three density-based gap scores forecast which cells grow fastest at or below chance (AUC 0.30 to 0.66) and add nothing to momentum in Steam and arXiv
    
[^20]: 超越双语：与会话式AI进行四语码转换的研究

    Beyond Bilingual: Studying Quadrilingual Code-Switching with Conversational AI

    [https://arxiv.org/abs/2611.21394](https://arxiv.org/abs/2611.21394)

    该研究首次探讨了四语环境（荷兰语、英语、西班牙语与低资源克里奥尔语帕皮阿门托语）下用户与聊天机器人的码转换行为，发现英语虽占主导，但在文化相关话题中低资源语言的使用显著提升，且用户切换语言并无不适感。

    

    多语言使用者会根据话题和对话伙伴来选择语言，而大型语言模型（LLM）对这些语言的支持程度并不均衡。关于会话式AI码转换的研究大多集中在资源丰富的双语语言对上。在库拉索和阿鲁巴，许多人使用荷兰语、英语、西班牙语以及低资源的克里奥尔语——帕皮阿门托语。在一项预注册研究中，42名大学生与聊天机器人进行了两次五分钟的对话，一次关于大学生活，一次关于当地食物和文化。通过问卷调查评估了参与者的语言背景、AI使用情况以及语言转换的舒适度。库拉索的20名国际学生构成了对照样本。参与者在对话轮次之间进行语言切换，很少在单条消息内混用语言。英语在两次对话中均占主导地位。参与者在文化话题的对话中更频繁地切换语言，其中帕皮阿门托语和西班牙语从消息占比的15%上升到24%。自我报告显示参与者的语言转换舒适度并未降低……

    arXiv:2611.21394v1 Announce Type: cross  Abstract: Multilingual speakers choose languages by topic and conversation partner. Large language models (LLMs) support these languages unequally. Research on code-switching with conversational AI has mostly examined bilingual pairs of well-resourced languages. In Cura\c{c}ao and Aruba, many people use Dutch, English, Spanish, and the low-resource Creole Papiamentu/o. In a preregistered study, 42 university students held two five-minute chatbot conversations, one about university life and one about local food and culture. Surveys assessed language background, AI use, and switching comfort. 20 international students in Cura\c{c}ao formed a comparison sample. Participants switched between turns and rarely mixed languages within a message. English dominated both conversations. Participants switched more often in the cultural conversation, where Papiamentu/o and Spanish rose from 15% to 24% of messages. Self-reports showed no lower comfort switchin
    
[^21]: 先测量后训练：在单一用户标签上微调类型化决策模型的预注册学习曲线

    Train After You Measure: a Pre-Registered Learning Curve for Fine-Tuning Typed Decision Models on One User's Labels

    [https://arxiv.org/abs/2611.21393](https://arxiv.org/abs/2611.21393)

    该论文通过一项预先注册的学习曲线实验证明，在单一用户的邮件标签上用LoRA微调小型类型化决策模型（Tev1-0.8B/4B），在所有标签预算下均未能超越简单的TF-IDF逻辑回归基线，且五个假设中四个被证伪。

    

    在单一用户的电子邮件标签上微调一个小型类型化决策模型，未能达到我们预先注册的、相对于在同一标签上训练的逻辑回归的优势差距。在早期工作中，约两百个标签和TF-IDF逻辑回归在此收件箱上取得了最高的点精度，尽管并未显著高于最佳的托管模型；我们提出的问题是：同样的标签交给决策模型是否能缩小这一差距。第一项注册研究在通用4B基座模型上进行，采用交叉验证且仅针对消息类型，在每个标签预算下均低于基线（-1.8至-7.3个百分点）。主研究在任何微调运行之前即已注册：使用一个从未用于模型选择的固定测试集（135封电子邮件），在公开的Tev1-0.8B和Tev1-4B决策模型上使用LoRA适配器，涵盖50至587个标签的嵌套子集，两个随机种子和五个决策字段，全部在免费的T4 GPU上运行。其五个假设中有四个被证伪——其中两个反而有利于微调——另有一个在某一模（型）上得到证实……

    arXiv:2611.21393v1 Announce Type: new  Abstract: Fine-tuning a small typed decision model on one user's email labels did not clear the margin we pre-registered over a logistic regression trained on the same labels. In earlier work, about two hundred labels and a TF-IDF logistic regression had the highest point accuracy on this inbox, though not significantly above the best hosted model; we ask whether the same labels, given to a decision model, close the gap. A first registered study, on a generic 4B base with cross-validation and message type only, stayed below the baseline at every label budget (-1.8 to -7.3 points). The main study was registered before any fine-tuning run: a fixed test set of 135 emails never used for selection, LoRA adapters on the public Tev1-0.8B and Tev1-4B decision models, nested subsets of 50 to 587 labels, two seeds and five decision fields on a free T4 GPU. Four of its five hypotheses are refuted--two in the fine-tunes' favour--and one is confirmed for one m
    
[^22]: 大语言模型中选择条件化风险控制的稳定性补偿阈值校准

    Stability-Compensated Threshold Calibration for Selection-Conditioned Risk Control in Large Language Models

    [https://arxiv.org/abs/2611.21384](https://arxiv.org/abs/2611.21384)

    该论文针对大语言模型选择性预测中LEC方法阈值校准可能违反选择条件化错误率目标的问题，提出了SC-LEC方法，利用理论全数据参考的负线性风险补偿阈值选择不稳定性，并提供有限样本保证。

    

    大语言模型经常产生幻觉性答案，这促使研究者采用选择性预测策略，对不确定的查询选择弃答。然而，仅依靠启发式的不确定性阈值无法对被接受预测中的错误提供统计控制。线性期望约束（LEC）将选择条件化错误率（SCER）校准简化为非正的期望线性损失约束，但该损失在接受阈值上是非单调的。我们构建了一个独立同分布采样下的精确反例，证明LEC的经验阈值选择规则可能违反总体SCER目标。为解决这一风险转移缺口，我们提出了稳定性补偿线性期望控制（SC-LEC），利用理论全数据参考的负线性风险来补偿阈值选择的不稳定性。对于超出安全锚点的局部扩展，我们推导出两个有限样本证书：一个稳定…（原文摘要在此处截断）

    arXiv:2611.21384v1 Announce Type: new  Abstract: Large language models often produce hallucinated answers, motivating selective prediction policies that abstain on uncertain queries. However, heuristic uncertainty thresholds alone do not provide statistical control of errors among accepted predictions. Linear Expectation Constraints (LEC) reduce selection-conditioned error rate (SCER) calibration to a nonpositive expected linear-loss constraint, but the associated loss is non-monotonic in the acceptance threshold. We construct an exact counterexample under i.i.d. sampling demonstrating that LEC's empirical threshold-selection rule can violate the population SCER target. To address this risk-transfer gap, we introduce Stability-Compensated Linear Expectation Control (SC-LEC), which uses the negative linear risk of a theoretical full-data reference to compensate for threshold-selection instability. For local expansion beyond a safe anchor, we derive two finite-sample certificates: a stab
    
[^23]: 《LLM驱动的智能体中隐性偏见的众包审计图谱》

    A Crowd-Audited Atlas of Implicit Bias in LLM-Powered Agents

    [https://arxiv.org/abs/2611.21376](https://arxiv.org/abs/2611.21376)

    该研究基于美国时间使用调查（ATUS）的17个日常活动领域构建了包含506个决策场景的图谱，揭示LLM智能体在不同种族、性别认同和政治意识形态人设下表现出显著隐性偏见，并通过500名参与者、8,010次二元选择的众包审计来确定最值得进一步审查的偏见场景。

    

    先前的研究表明，大型语言模型（LLM）在高关注度、一次性的决策场景（如招聘和智力评估）中会表现出隐性偏见。然而，随着人们越来越多地使用自主的、由LLM驱动的AI智能体来代表自己完成日常任务，理解基于社会人口统计学假设的隐性偏见如何在广泛的日常决策场景中产生变得至关重要。我们利用美国时间使用调查（ATUS）的17个日常活动领域，生成了一个包含506个决策场景的图谱，在这些场景中，被赋予代表不同种族、性别认同和政治意识形态的人设的LLM表现出了显著的隐性偏见。接着，在一项众包审计中，500名参与者对哪些存在隐性偏见的决策场景最值得进一步审查做出了8,010次二元选择。我们发现参与者优先考虑了……（原文摘要在此处被截断）

    arXiv:2611.21376v1 Announce Type: cross  Abstract: Prior work has shown that large language models (LLMs) express implicit biases in high-publicity, one-off decision scenarios such as hiring and intelligence assessment. However, as people increasingly employ autonomous, LLM-powered AI agents that make decisions on their behalf in doing day-to-day tasks, it is of critical importance that we understand how implicit biases, based on sociodemographic assumptions, emerge across a broad milieu of workaday decision scenarios. We used the 17 domains of daily activity in the American Time Use Survey (ATUS) to generate an atlas of 506 decision scenarios in which LLMs --- endowed with personas representing different races, gender identities, and political ideologies --- expressed significant implicit bias. Next, in a crowdsourced audit, 500 participants made 8,010 binary choices about which implicitly biased decision scenarios most deserved further review. We found that participants prioritized d
    
[^24]: 你所发布的即你的症状：从社交媒体话语中估计心理测量网络

    What You Post Is What You Symptom: Estimating Psychometric Networks from Social Media Discourse

    [https://arxiv.org/abs/2611.21362](https://arxiv.org/abs/2611.21362)

    该研究首次提出直接从社交媒体话语估计症状分辨心理测量网络的框架，并在抑郁症和进食障碍的多个独立语料库上验证了其与临床问卷网络的可靠性、一致性和跨诊断的语言标记。

    

    精神病理学中的症状网络几乎完全通过自我报告问卷进行估计，这使得推断局限于特定测量工具所涵盖的症状集合。我们提出了一个直接从社交媒体话语中进行症状分辨的网络估计框架，并将该框架应用于抑郁症和进食障碍，同时使用第三个厌食症语料库作为跨数据集验证。据我们所知，我们是首个从社交媒体文本中估计症状分辨的心理测量网络、将其与临床共同认可网络进行比较、并在涵盖2种障碍的3个独立招募语料库中复制跨诊断语言标记的研究。对于抑郁症，表达分数与BDI-II总分的相关性达到ρ=0.434，21个条目中有20个达到FDR显著性，折半信度为0.935。网络比较检验发现两种模态之间的边差异很少（占边的3-5%），而桥接症状……

    arXiv:2611.21362v1 Announce Type: cross  Abstract: Symptom networks in psychopathology are estimated almost exclusively from self-report questionnaires, restricting inference to whatever symptom set a given instrument covers. We introduce a framework for symptom-resolved network estimation directly from social media discourse and apply the framework to depression and eating disorders, with a third anorexia corpus as cross-dataset validation. To our knowledge, we are the first to estimate symptom-resolved psychometric networks from social media text, compare them to clinical co-endorsement networks, and replicate transdiagnostic linguistic markers across 3 independently recruited corpora spanning 2 disorders simultaneously. For depression, expression scores correlate with BDI-II totals at $\rho{=}0.434$, 20 of 21 items reach FDR significance, and split-half reliability is 0.935. The Network Comparison Test detects few edge differences between modalities (3-5\% of edges), while bridge sy
    
[^25]: 藏文OCR的音节级与词级评估

    Syllable- and Word-Level Evaluation of Tibetan OCR

    [https://arxiv.org/abs/2611.21303](https://arxiv.org/abs/2611.21303)

    本研究首次对藏文OCR进行了音节级与词级错误率评估，发现字符错误率（CER）与更高层级错误率的系统排序基本一致，而无需分词器的音节错误率（SER）可作为更细粒度的评估指标。

    

    藏文的光学字符识别（OCR）目前主要以字符错误率（CER）进行评估。藏文标记音节边界但不标记词边界，因此词错误率的计算需要借助分词器。本研究在印刷体、木刻版以及两种手写体风格上，对九种识别器配置进行了单独及组合评估，并报告了音节级和词级错误率，据我们所知，这是首次针对藏文OCR开展的此类评估。这些配置涵盖CTC模型、带与不带训练后量化的TrOCR编码器-解码器模型，以及一个视觉-语言模型，每个配置均在CER、音节错误率（SER）和三种分词词错误率（SWER）指标下进行评分。结果显示，CER对系统的排序与更高层级的错误率基本一致（Kendall τ ≥ 0.833），但其中有一对系统的排序相反。在联合训练效果方面，64次比较中有7次CER与SER出现了符号相反的显著变化。SER无需分词器，且对系统的排序……（原文在此截断）

    arXiv:2611.21303v1 Announce Type: new  Abstract: Optical character recognition (OCR) for Tibetan is evaluated predominantly by character error rate (CER). Tibetan marks syllable boundaries but does not mark word boundaries, so a word error rate requires a segmenter. This study evaluates nine recognizer configurations on typeset, woodblock and two handwritten script styles, alone and in combination, and reports syllable- and word-level error rates, to our knowledge the first for Tibetan OCR. The configurations span CTC models, TrOCR encoder-decoders with and without post-training quantization, and a vision-language model, and each is scored under CER, the syllable error rate (SER) and three segmented word error rates (SWER). CER ranks systems largely as the higher-level rates do (Kendall $\tau \geq 0.833$), with one pair ordered oppositely. On the effect of joint training, CER and SER record significant changes of opposite sign in 7 of 64 comparisons. SER needs no segmenter and ranks sy
    
[^26]: 潜在桥梁：无需言语的求助

    Latent Bridge: Asking for Help Without Words

    [https://arxiv.org/abs/2611.21301](https://arxiv.org/abs/2611.21301)

    POND通过可训练桥梁将冻结大模型的隐藏状态直接注入小模型的注意力机制，实现“仅预填充、不解码”的跨模型协作，使小模型在无需承担大模型解码成本的情况下显著提升推理与编程性能。

    

    基于文本的模型协作要求大模型先生成中间响应，小模型才能利用其指导，这为推理过程增加了大模型的解码成本。我们提出POND（仅预填充、不解码）方法，使用冻结的大模型作为发送方进行预填充，并由小模型作为接收方完成所有自回归解码。一个可训练的桥梁将发送方的隐藏状态映射为额外的注意力键和值，使接收方在整个生成过程中都能利用这些表示。POND还支持周期性审查，即发送方处理接收方的部分响应，并通过额外的预填充过程更新其指导，同时保留接收方已有的KV缓存。在推理和编程基准测试中，POND在不同模型家族的发送方配合下均提升了接收方的性能。特别是，GLM-5.3与Qwen接收方配对时能提供与Qwen3.8-Max相当的增益，支持了效果……（原文截断）

    arXiv:2611.21301v1 Announce Type: new  Abstract: Text-based model collaboration requires a large model to generate intermediate responses before a smaller model can use its guidance, adding large-model decoding cost to inference. We introduce POND (Prefill-Only, No-Decode), a method that uses a frozen large sender for prefill and a small receiver for all autoregressive decoding. A trainable bridge maps the sender's hidden states to additional attention keys and values, allowing the receiver to use these representations throughout generation. POND also supports periodic review, in which the sender processes the receiver's partial response and updates its guidance through additional prefill passes while preserving the receiver's existing KV cache. Across reasoning and coding benchmarks, POND improves receiver performance with senders from different model families. In particular, GLM-5.3 provides gains comparable to those of Qwen3.8-Max when paired with a Qwen receiver, supporting the eff
    
[^27]: LATTE-WordSeg-Lite：在CPU上实现多粒度泰语分词，并附可复现的基准测试

    LATTE-WordSeg-Lite: Multi-Granularity Thai Word Segmentation on a CPU, with a Reproducible Benchmark

    [https://arxiv.org/abs/2611.21300](https://arxiv.org/abs/2611.21300)

    该研究将大型GPU模型LATTE蒸馏为仅117万参数的CPU分词器LATTE-WordSeg-Lite，可在一次推理中输出复合词、词和语素多粒度的泰语分词结果，并通过重建七个测试集和统一协议评分十二个系统建立了可复现的基准。

    

    大型语言模型通过子词标记（subword tokens）来阅读泰语，而无需分词器。然而，泰语树库、语音指标和形态学资源都是建立在词的基础之上的，且常常涉及多个粒度，而词边界能够解决子词标记所遗留的歧义。LATTE（Lattice ATTentive Encoding，格注意力编码）通过由词典词构成的格（lattice）同时以多个粒度读取文本，但它是一个大型GPU模型。我们将其蒸馏为LATTE-WordSeg-Lite——一个仅含117万参数、保留格结构作为特征的CPU分词器。它可以在一次推理中同时输出复合词、词和语素三个粒度的结果，并且仅从标注数据和许可允许商业使用的文本中学习。为了与其他工具进行比较，已发表的泰语分词分数并不足够，因为它们来自不同的文件、数据划分和评分器。因此，我们从固定版本的数据源重建了七个测试集，并在统一协议下对十二个系统进行了评分。在每个测试集所适配的评估单位下，LATTE-WordSeg-Lite（原文摘要在此处截断）

    arXiv:2611.21300v1 Announce Type: new  Abstract: Large language models read Thai through subword tokens, without a word segmenter. Yet Thai treebanks, speech metrics and morphological resources are built on words, often at several granularities, and word boundaries resolve ambiguities that subword tokens leave open. LATTE (Lattice ATTentive Encoding) reads a text at several granularities at once, through a lattice of dictionary words, but it is a large GPU model. We distil it into LATTE-WordSeg-Lite, a CPU segmenter of 1.17M parameters that keeps the lattice as features. It outputs compounds, words and morphemes in one pass, and learns only from annotated data and text whose licences allow commercial use. To compare it with other tools, published Thai scores do not suffice, as they come from different files, splits and scorers. We therefore rebuild seven test sets from pinned sources and score twelve systems under one protocol. With the unit that fits each test set, LATTE-WordSeg-Lite 
    
[^28]: AnachroBench：评估大型视觉语言模型中的历史一致性

    AnachroBench: Evaluating Historical Consistency in Large Vision-Language Models

    [https://arxiv.org/abs/2611.21292](https://arxiv.org/abs/2611.21292)

    该论文提出AnachroBench基准，利用来自八个国家、跨越19世纪末至20世纪初的1,600对历史图像及其时代错置版本，从插入物品感知、时代-国家一致性和条件时代-国家一致性三个维度评估大型视觉语言模型的历史一致性能力。

    

    现有的大型视觉语言模型基准主要评估一般日常场景中的视觉理解能力，而模型与特定时代和国家的历史一致性在很大程度上尚未得到充分探索。为了填补这一空白，我们提出了AnachroBench，这是一个用于评估视觉场景历史一致性的受控判别式基准。它包含来自八个国家的1,600对图像，时间跨度为19世纪末至20世纪初，每对图像由一张原始历史图像与一个包含无缝插入现代物品的时代错置版本组成。AnachroBench从三个方面进行评估：插入物品感知，测试模型是否能检测出插入的现代物品；时代-国家一致性，评估场景是否与指定的时代和国家相符；条件时代-国家一致性，仅在插入物品被正确感知的情况下衡量一致性判断。错误的一致性判断……

    arXiv:2611.21292v1 Announce Type: cross  Abstract: Existing Large Vision-Language Model benchmarks primarily evaluate visual understanding in general everyday scenarios, leaving historical consistency with a specified era and country largely underexplored. To address this gap, we propose AnachroBench, a controlled discriminative benchmark for evaluating historical consistency in visual scenes. It contains 1,600 image pairs from eight countries spanning the late 19th to the early 20th century, each pairing an original historical image with an anachronistic version containing a seamlessly inserted modern object. AnachroBench evaluates three aspects: Inserted Object Perception, which tests whether models detect the inserted modern object; Era-Country Consistency, which assesses whether the scene is consistent with the specified era and country; and Conditional Era-Country Consistency, which measures consistency judgments only when the inserted object is correctly perceived. A false consis
    
[^29]: 面向语言学标注的大语言模型基准测试

    Benchmarking Large Language Models for Linguistic Annotation

    [https://arxiv.org/abs/2611.21291](https://arxiv.org/abs/2611.21291)

    本文提出了带网页界面的 Python 语言学标注基准工具 dhAIBench，通过对 48 个模型在九项语言学标注任务上的 293 次运行进行系统评估，证明应依据特定语言任务的标注质量而非通用排名来评价大语言模型。

    

    评估大语言模型用于语言学标注需要在有据可查的条件下进行任务特定的评估。我们提出了 dhAIBench，这是一个 Python 基准测试与标注工具，并配有用于探索和分享结果的网页界面。该工具记录提示词、配置和预测结果，从而支持对标注质量、置信度校准、可重复性以及预估成本进行比较。我们通过 48 个模型在九项语言学标注任务上的 293 次运行来演示该框架及相关方法论，这些任务涵盖形态学、句法、语义学、篇章—语用学、拼写规范化以及学习者语言纠错。模型在多项任务上与参考标注达到了高度一致，其中包括涉及非标准化历史语言变体的任务以及需要语境解释的任务。研究结果支持针对特定语言学任务来评估模型，而非依赖通用排名。为此，我们……

    arXiv:2611.21291v1 Announce Type: cross  Abstract: Assessing large language models for linguistic annotation requires task-specific evaluation under documented conditions. We present dhAIBench, a Python benchmarking and annotation tool with a web interface for exploring and sharing results. It records prompts, configurations and predictions, enabling comparisons of annotation quality, confidence calibration, repeatability and estimated cost. We demonstrate the framework and the associated methodology through 293 runs of 48 models on nine linguistic annotation tasks spanning morphology, syntax, semantics, discourse-pragmatics, spelling normalization and learner-language correction. Models achieve high agreement with reference annotations on several tasks, including those on non-standardized, historical varieties as well as requiring contextual interpretation. The results support evaluating models against specific linguistic tasks, rather than relying on general rankings. To this end, we
    
[^30]: 经典结构先验提升分子语言模型中的神经分子表示

    Classical structural priors improve neural molecular representations in molecule language models

    [https://arxiv.org/abs/2611.21276](https://arxiv.org/abs/2611.21276)

    提出SAGA训练正则化方法，利用中心核对齐将神经分子表示与指纹Tanimoto相似性对齐，在不增加推理开销的情况下提升分子语言模型的分子-文本检索和性质预测性能，使表示更符合经典化学结构关系。

    

    经典分子相似性是化学信息系统的核心，然而在结构-文本配对数据上训练的分子语言模型并不一定按照梯度化的结构关系来组织分子嵌入。在此，我们提出了结构感知引导对齐，这是一种仅在训练阶段使用的正则化方法，它利用中心核对齐将神经分子表示与基于分子指纹的Tanimoto相似性对齐，既不增加参数，也无需在推理时使用分子指纹。在MolCA和Atomas模型中，SAGA提升了分子-文本检索性能，使非精确检索的结果更偏向结构更接近的类似物，并提高了基于骨架划分的性质预测的平均性能，同时分子描述生成和文本到分子的生成在单次运行的消融实验中得到改善或保持不变。这些结果表明，成熟的化学信息学先验可以使分子语言表示在化学上更加忠实……

    arXiv:2611.21276v1 Announce Type: cross  Abstract: Classical molecular similarity is central to chemical information systems, yet molecule-language models trained on paired structure-text data do not necessarily organize molecular embeddings according to graded structural relationships. Here we introduce Structure-Aware Guided Alignment (SAGA), a training-only regularizer that uses Centered Kernel Alignment to align neural molecular representations with fingerprint-based Tanimoto similarity, without adding parameters or requiring fingerprints at inference. In MolCA and Atomas, SAGA improves molecule-text retrieval, shifts non-exact retrieval toward closer structural analogues, and increases average scaffold-split property-prediction performance, while molecular description and text-to-molecule generation are improved or preserved in single-run ablations. These results show that established chemoinformatics priors can make molecule-language representations more chemically faithful while
    
[^31]: 选择示例的代价：自适应上下文学习中的上下文过拟合

    The Price of Choosing Examples: Context Overfitting in Adaptive In-Context Learning

    [https://arxiv.org/abs/2611.21251](https://arxiv.org/abs/2611.21251)

    该论文首次将提示构建形式化为自适应数据分析，提出“上下文容量”概念，证明自适应选择示例的上下文学习存在 √(κ/k) 阶的泛化过拟合代价，并给出匹配下界和 PAC-贝叶斯检索规则。

    

    上下文学习通常被分析为提示中的示例在学习器被选定之前就已采样的情形。但在实际部署的系统中并非如此：示例是在看到查询和数据存储之后才被检索、过滤、重排序、编辑并反复测试的。我们通过条件随机核演算将提示构建视为自适应数据分析，从而形式化了这一差距。我们的核心量是上下文容量，即数据存储与最终提示记录之间的互信息或近似最大信息。对于任何冻结的上下文学习器和任何容量为 κ 的自适应上下文流水线，我们证明了阶为 √(κ/k) 的有限样本上下文泛化上界，其中 k 是所选示例的数量，其损失构成所报告的分数。我们证明了匹配的下界，当 log k ≪ κ ≪ k 时其常数是精确的，并推导出两种 PAC-贝叶斯检索规则：一种先（摘要在此处被截断）……

    arXiv:2611.21251v1 Announce Type: cross  Abstract: In-context learning is usually analyzed as if the examples in the prompt were sampled before the learner is chosen. In deployed systems they are not: examples are retrieved, filtered, re-ranked, edited, and repeatedly tested after seeing the query and the data store. We formalize this gap by treating prompt construction as adaptive data analysis through a conditional stochastic-kernel calculus. Our main quantity is the context capacity, the mutual information or approximate max-information between the data store and the final prompt transcript. For any frozen in-context learner and any adaptive context pipeline with capacity $\kappa$, we prove a finite-sample context generalization bound of order $\sqrt{\kappa/k}$, where $k$ is the number of selected examples whose losses form the reported score. We prove a matching lower bound, whose constant is exact when $\log k\ll\kappa\ll k$, and derive two PAC-Bayesian retrieval rules: a first-or
    
[^32]: 信念流：通过潜在信念动态激发集体智能

    Stream of Belief: Eliciting Collective Intelligence through Latent Belief Dynamics

    [https://arxiv.org/abs/2611.21250](https://arxiv.org/abs/2611.21250)

    提出了信念流框架，将多智能体推理建模为潜在信念动态寻求集体均衡的过程，智能体通过通信与反思迭代优化自身信念直至信息不再有效传播，并在多种任务和模型家族上展现出良好的泛化能力。

    

    在这项工作中，我们将多智能体推理视为寻求隐式信念动态集体均衡的过程，并提出了信念流方法，其中各智能体保持连续的潜在信念，并通过通信与反思对其进行迭代优化。我们的方法旨在寻求多智能体系统的集体信念状态，即在此状态下进一步的通信不再引发有意义的信息传播。通过大量实验，我们证明了SoB在多种任务和不同模型家族上的泛化能力。更广泛地说，我们的结果为从信念动态视角扩展多智能体推理提供了新的思路。

    arXiv:2611.21250v1 Announce Type: new  Abstract: In this work, we view multi-agent reasoning as a process of seeking a collective equilibrium of implicit belief dynamics and introduce Stream of Belief (SoB), where agents maintain continuous latent beliefs and iteratively refine them through communication and reflection. Our method seeks a collective belief state of a multi-agent system where further communication induces no meaningful information propagation. Through extensive experiments, we demonstrate the generalization of SoB across tasks and model families. More Broadly, our results suggest a new perspective on scaling multi-agent reasoning from belief dynamics.
    
[^33]: 自适应测试时扩展的任意时刻有效PAC-Bayes证书

    Anytime-Valid PAC-Bayes Certificates for Adaptive Test-Time Scaling

    [https://arxiv.org/abs/2611.21241](https://arxiv.org/abs/2611.21241)

    本文证明了适用于自适应选择的任意时刻有效PAC-Bayes证书，使测试时扩展系统在自适应确定采样量、提示词和验证器之后仍能获得统计有效的泛化保证，且同一保证覆盖部署前的全部方案搜索过程。

    

    测试时扩展系统通常在反复检查校准集之后，才决定采样多少、使用哪个提示词，以及信任哪个验证器或聚合规则。在这种自适应选择之后，标准的验证误差界便不再有效。我们将测试时扩展方案建模为具有随机成本的随机化预测器，并证明了一个时间一致的PAC-Bayes证书，该证书同时对所有校准时刻、所有关于方案的后验混合分布，以及所有依赖数据的停止规则成立。因此，同一事件既覆盖了部署之前进行的方案搜索，也覆盖了最终报告的固定方案。对于预算和验证器的有限网格，惩罚项是熟悉的 $\log M/n$，在最坏情况下，任何在选择之后有效的证书都无法避免该惩罚项；而非均匀先验和局部化的后验混合分布可以为结构化搜索提供更紧的证书。我们将该结果专门化到自洽性方法并展示

    arXiv:2611.21241v1 Announce Type: cross  Abstract: Test-time scaling systems often choose how much to sample, which prompt to use, and which verifier or aggregation rule to trust after repeatedly inspecting a calibration set. Standard validation error bars are not valid after this adaptive selection. We model a test-time scaling recipe as a randomized predictor with a random cost and prove a time-uniform PAC-Bayes certificate that holds simultaneously over all calibration times, all posterior mixtures over recipes, and all data-dependent stopping rules. Thus the same event covers the recipe search that precedes deployment, not merely the fixed recipe finally reported. For a finite grid of budgets and verifiers, the penalty is the familiar $\log M/n$, which in the worst case no certificate valid after selection can avoid, while non-uniform priors and localized posterior mixtures can give sharper certificates for structured searches. We specialize the result to self-consistency and show 
    
[^34]: 语言模型智能体的信念可通过开放信道中的真实证据被操纵

    Language-model agents' beliefs can be manipulated via truthful evidence in open channel

    [https://arxiv.org/abs/2611.21235](https://arxiv.org/abs/2611.21235)

    该论文提出“生成式蒙太奇”方法，证明即使仅使用真实的证据片段，也能通过诱导语言模型智能体产生“叙事过拟合”而使其过度自信地接受并传播谣言，且多数投票等审查机制难以纠正这一错误。

    

    语言模型智能体的信念可以通过社交媒体等开放信道被操纵，即使它们接收到的信息是真实的。当智能体用自我诱导的关系填补信息片段之间的空白时，它们可能会将这些关系当作已确立的事实接受，并将由此产生的谣言传递给他人。借鉴叙事理解的认知理论和电影蒙太奇理论，我们提出了“生成式蒙太奇”来操作化这一被称为“叙事过拟合”的失败模式，通过协调智能体将真实的片段排列成具有误导性的叙事。在来自六个Twitter事件的1,090个谣言目标上，智能体经常过度自信地接受谣言，在14个模型中有12个模型的整体接受率超过65%。在经过多数投票或对相同证据进行审查后，接受率仍保持在50%以上，这表明此类错误如何能够传播到下游决策中。为解释这种易感性，片段干预实验……

    arXiv:2611.21235v1 Announce Type: new  Abstract: Language-model agents' beliefs can be manipulated through open channels such as social media, even when the information they receive is truthful. When agents fill gaps between fragments with self-induced relationships, they may accept those relationships as established facts and pass the resulting rumours to others. Drawing on cognitive theory of narrative understanding and film theory of montage, we introduce Generative Montage to operationalize this failure, termed narrative overfitting, by coordinating agents to arrange truthful fragments into misleading narratives. Across 1,090 rumour targets from six Twitter events, agents often accepted rumours overconfidently, with overall acceptance exceeding 65 percent in twelve of fourteen models. Acceptance remained above 50 percent after majority voting or review of the same evidence, showing how errors can reach downstream decisions. To explain this susceptibility, fragment interventions ide
    
[^35]: 一种生成式AI驱动的全屋智能服务架构：边缘模型、记忆管理与家庭应用

    A Generative AI-Driven Architecture for Whole-Home Intelligent Services: Edge Models, Memory Management, and Household Applications

    [https://arxiv.org/abs/2611.21232](https://arxiv.org/abs/2611.21232)

    本文以家庭本地服务器为核心组织轴心，通过面向设计的系统性综述构建生成式AI全屋智能服务架构，识别出本地服务器的九项核心职责，以解决云端助手在数据治理、持续可用性和物理动作可靠性方面的局限。

    

    生成式人工智能正在进入家庭。基于云的助手在数据传输治理、持续可用性以及物理动作可靠性方面存在局限，这在家庭场景中尤为突出——家庭中的设备始终保持在线、数据随时间不断累积、命令会触发真实世界的动作。本文以家庭本地服务器作为统一的组织轴心，构建了一个面向设计的综述框架。研究检索了截至2026年11月来自arXiv、CNKI、ACL Anthology及官方文档的文献资料，并将其编码为五类证据：综述、系统原型、基准测试、人机交互与养老研究、官方文档。研究将各类系统设计映射到共享的架构模块上，识别出家庭本地服务器的九项反复出现的职责。生成式模型支持感知解释、意图理解和候选方案规划，而确定性有效的方法则[摘要截断]……

    arXiv:2611.21232v1 Announce Type: cross  Abstract: Generative artificial intelligence is entering the home. Cloud-based assistants face governance limitations in data transmission, continuous availability, and the reliability of physical actions, which are especially salient in households where devices remain online, data accumulate over time, and commands trigger real-world actions. Using the home local server as a unifying organizing axis, this article develops a design-oriented review framework. Sources from arXiv, CNKI, ACL Anthology, and official documentation were searched through November 2026 and coded into five evidence categories: reviews, system prototypes, benchmarks, HCI and eldercare studies, and official documentation. System designs were mapped onto shared architectural modules, identifying nine recurrent responsibilities of the home local server. Generative models support perceptual interpretation, intent understanding, and candidate planning, while deterministic valid
    
[^36]: 教语言模型根据自身置信度调整答案粒度

    Teaching Language Models to Adapt Answer Granularity to Their Confidence

    [https://arxiv.org/abs/2611.21211](https://arxiv.org/abs/2611.21211)

    该论文提出一种基于模型自身一致性的监督方法，训练语言模型学会在置信度低时给出更粗粒度但仍保留有用信息的答案，从而在不猜测或拒答的情况下改善正确性与信息量之间的平衡。

    

    当精确细节不确定时，更粗粒度的答案可以在不进行猜测或放弃作答的情况下保留有用的信息。我们研究了语言模型能否通过基于模型自身状态的监督学习到这种行为，该监督来源于采样响应之间的一致性以及对不同粒度层级答案的识别。我们在时间类和地理类问题以及三个泛化数据集上评估了LFM2.5-1.2B、Ministral 3 3B、Qwen3.5-9B和Gemma 4 31B四个模型。在若干域内设置中，微调提升了正确性与信息量之间的平衡，其效果超过了将基础答案粗化到最佳固定水平的做法。在泛化数据集上，所有四个模型在基础模型置信度较低时比置信度较高时更频繁地给出更粗粒度的答案。当综合考虑正确性与粒度时，三个较小模型的答案在保留关系上也优于基础模型的响应。这些发现证明了置信度相关的粒度适应行为是可学习的。

    arXiv:2611.21211v1 Announce Type: new  Abstract: When precise details are uncertain, a coarser answer can retain useful information without guessing or abstaining. We investigate whether language models can learn this behavior through model-conditioned supervision derived from consistency across sampled responses and recognition of answers at different granularity levels. We evaluate LFM2.5-1.2B, Ministral~3~3B, Qwen3.5-9B, and Gemma~4~31B on temporal and geographic questions and three generalization datasets. In several in-domain settings, fine-tuning improves the balance of correctness and informativeness beyond coarsening base answers to the best fixed level. Across generalization datasets, all four models produce coarser answers more often at low than high base-model confidence. The three smaller models' answers are also preferred over base responses on held-out relations when correctness and granularity are considered jointly. These findings demonstrate learnable confidence-relate
    
[^37]: 基于大语言模型的作文评分器验证：医学教育中批判性在线推理的比较分析

    Validating LLM-Based Essay Scorers: A Comparative Analysis in Critical Online Reasoning in Medical Education

    [https://arxiv.org/abs/2611.21208](https://arxiv.org/abs/2611.21208)

    本研究验证了基于通用大语言模型的自动作文评分系统集合能够近似人类对医学教育写作任务的评分，表明通用模型有望辅助甚至替代昂贵的人工标注。

    

    基于大语言模型（LLM）的自动作文评分器（AES）有望达到人类水平的准确性。然而，针对特定评估任务对LLM进行训练、微调和评估的成本很高。尤其是，这些AES是否应该针对特定领域进行训练，还是基于通用LLM的模型就已经足够，是一个值得探讨的问题。相比之下，人工标注通常更为有效，但成本也高得多。本研究评估了基于通用LLM的AES系统集合，能否利用其编码的语言和概念知识来近似人类对医学教育写作任务的评分。其目标是评估通用模型是否适用于辅助、甚至潜在地取代该领域的人工标注。我们试验了11个AES系统和8个不同的LLM，评估了668份学生对12个任务的回答，并采用了7名人类标注者进行的2076次评估。这些评分被汇总……（摘要在此处被截断）

    arXiv:2611.21208v1 Announce Type: new  Abstract: Automated essay scorers (AES) based on large language models (LLMs) promise human-level accuracy. However, training, fine-tuning, and evaluating LLMs for specific assessment tasks is costly. In particular, the question arises whether these AES should be trained for specific domains or whether models based on generic LLMs might be sufficient. Human annotations, by contrast, are often more valid but far more expensive. This study evaluates whether ensembles of AES systems based on general-purpose LLMs can approximate human ratings of medical-education writing tasks by leveraging encoded linguistic and conceptual knowledge. The goal is to assess whether general-purpose models are applicable for assisting or potentially replacing human annotation in this domain. We experiment with 11 AES systems and 8 different LLMs and evaluate 668 student responses to 12 tasks, using \numprint{2076} evaluations by 7 human annotators. These ratings are aggr
    
[^38]: 看见不等于解决：基于会话锚定的视觉与语言消歧

    Seeing Isn't Resolving: Visual and Linguistic Disambiguation with Conversational Grounding

    [https://arxiv.org/abs/2611.21183](https://arxiv.org/abs/2611.21183)

    本文构建了含人机对话歧义场景的日语视觉锚定基准JAG-D，发现视觉语言模型能借助对话上下文显著提升语言歧义（尤其是日语零指代）的消解能力。

    

    现有的视觉锚定基准通常未考虑真实场景中的上下文不确定性。在本研究中，我们构建了一个新颖的基准——JAG-D（基于对话的歧义视觉锚定日语评测）数据集。该数据集由人机对话式多模态交互构成，涵盖了源于视觉和语言两类不同类型的歧义。我们评估了多种最先进的视觉语言模型（VLM）在歧义识别与消解方面的能力，并比较了它们在解决视觉歧义与语言歧义时的对话上下文锚定性能。实验结果表明，视觉语言模型能够通过利用对话上下文显著提升语言歧义（尤其是日语零指代）的消解能力。我们进一步开展了对比分析，以探究锚定性能的提升究竟源于对话的语义内容，还是其会话结构。

    arXiv:2611.21183v1 Announce Type: new  Abstract: Visual grounding benchmarks usually do not consider the contextual uncertainty of real-world scenarios. In this study, we construct a novel benchmark, the Japanese evaluation for Ambiguous visual Grounding with Dialogue (JAG-D) dataset. It consists of human-robot conversational multimodal interactions covering different types of ambiguity, arising from visual and linguistic categories. We evaluate the capabilities of different state-of-the-art Visual Language Models (VLMs) for ambiguity recognition and resolution. We compare dialogue context grounding performance for visual and linguistic ambiguity resolution. Our experiments show that VLMs can substantially improve the resolution of linguistic ambiguity, especially Japanese zero anaphora, by using dialogue context. We further conduct a comparative analysis to understand whether grounding improvements arise from the semantic content of the dialogue or from its conversational structure.
    
[^39]: TRACK：面向电源管理系统故障诊断的工具增强时间序列推理

    TRACK: Tool-Augmented Reasoning over Time Series for Fault Diagnosis in Power Management System

    [https://arxiv.org/abs/2611.21178](https://arxiv.org/abs/2611.21178)

    TRACK将时间序列编码器与小型语言模型结合，通过工具调用实现精确可验证的计算，在设备端生成基于物理原理的结构化故障诊断报告，在15类多负载基准上达到超过95%的准确率并显著超越基线。

    

    运行多个负载的电源管理系统中的故障诊断目前依赖专家手动检查电流和电压波形，这种方式难以在大量已部署设备上扩展。我们提出了TRACK，它将时间序列编码器与小型语言模型（SLM）相结合，生成结构化、基于物理原理的诊断报告，并通过调用工具进行精确、可验证的计算。该系统在设备端运行，将专有传感器数据保留在本地，同时避免了昂贵的API调用费用。我们引入了一个基于评分细则的评估框架，通过逐项评判结构、信号保真度、物理推理、诊断和可操作性等准则来评估这些报告。在一个15类多负载模拟基准上，TRACK实现了超过95%的故障类别准确率，并能泛化到分布外故障，而最好的编码器探测基线仅达到58%的准确率。

    arXiv:2611.21178v1 Announce Type: cross  Abstract: Fault diagnosis in power management systems running multiple loads relies on experts manually inspecting current and voltage waveforms, which fails to scale across deployed units. We present TRACK, which pairs a time series encoder with a Small Language Model to produce structured, physics-grounded diagnostic reports, calling tools for exact, verifiable calculations. It operates on-device, keeping proprietary sensor data local and avoiding expensive API calls. We introduce a rubric-based framework that evaluates these reports by scoring structure, signal fidelity, physics reasoning, diagnosis, and actionability through per-criterion verdicts. On a 15-class multi-load simulated benchmark, TRACK achieves over 95% fault-category accuracy and generalizes to out-of-distribution faults, whereas the best encoder-probing baseline reaches only 58% accuracy.
    
[^40]: PlainPairs：一个将美国处方药标签中面向处方医师与面向患者的文本进行配对的语料库

    PlainPairs: A Corpus Pairing the Prescriber and Patient Texts of U.S. Prescription Medicine Labels

    [https://arxiv.org/abs/2611.21160](https://arxiv.org/abs/2611.21160)

    该论文构建了PlainPairs开放语料库，首次将美国处方药标签中面向处方医师与面向患者的文本系统性配对，并附带阅读难度和白话程度等指标，为研究医患健康信息沟通的文本差异提供了公开资源。

    

    在美国，处方药均附有为处方医师撰写的标签说明，许多药品还附带面向患者的文本：用药指南或患者说明书。这两类文本面向两类读者描述同一种药品，但公开资源很少将它们并列呈现。本文提出PlainPairs，这是一个基于openFDA药品标签文件构建的开放语料库。版本0.1.0包含35,177个附有患者文本的处方标签、21,416个不同的患者文本、通过监管标题与处方医师章节相匹配的250,829个患者文本片段、双方文本的阅读难度测量指标，以及基于规则的白话措辞统计。第二个数据来源——美国国家癌症研究所的癌症信息摘要——新增了756份摘要，涵盖面向医护专业人员和面向患者的版本，并包含英语和西班牙语两种语言。以标签结构来衡量，用药指南的中位数阅读难度按Flesch-Kincaid公式为8.7年级……（摘要在此处被截断）

    arXiv:2611.21160v1 Announce Type: new  Abstract: A prescription medicine in the United States comes with labeling written for prescribers, and many medicines also carry a text written for patients: a Medication Guide or a patient package insert. The two texts describe the same medicine for two readers, yet public resources rarely hold them side by side. I present PlainPairs, an open corpus built from the openFDA drug label files. Version 0.1.0 holds 35,177 prescription labels that carry a patient text, 21,416 distinct patient texts, 250,829 patient-text segments matched to prescriber sections through their regulatory headings, reading-level measures on both sides, and rule-based counts of plain-language wording. A second source, the National Cancer Institute's cancer information summaries, adds 756 summaries in health-professional and patient versions, in English and Spanish. Measured on the structure of the label, the median Medication Guide reads at grade 8.7 by Flesch-Kincaid and at
    
[^41]: 编辑推荐：大语言模型能否预判用户的编辑请求，以及何时预判？

    Edit Recommendation: Can LLMs Anticipate Edit Requests from Users, and When?

    [https://arxiv.org/abs/2611.21155](https://arxiv.org/abs/2611.21155)

    该论文提出“编辑推荐”任务及 EditRecBench 基准，研究大语言模型能否预判用户对生成结果的编辑请求，并通过基于决策模型的门控机制过滤掉难以预测的情况，将展示建议的 pass@5 提升 12–25 个百分点。

    

    在大语言模型智能体生成作品之后，用户通常会对其发送编辑请求。本文研究大语言模型能否在这些请求被发送之前对其进行预判。这种预判能力在实践中非常有用，因为它可以减少用户审查和反馈的工作量。我们将这一任务形式化为“编辑推荐”，并基于 WildChat-4.8M 中的真实对话日志构建了 EditRecBench，一个用于评估大语言模型在该任务上表现的基准。通过基准测试和失败分析，我们发现了一个实际挑战：在真实场景中，有些案例本质上就难以预测。为解决这一问题，我们引入了基于决策模型的门控机制，在将预测结果展示给用户之前，过滤掉那些预测不太可能正确的情况。我们训练的模型优于现有的决策模型，在给定的门控阈值下，将所展示建议中的 pass@5 提升了 12–25 个百分点。我们的代码已公开。

    arXiv:2611.21155v1 Announce Type: cross  Abstract: After a large language model (LLM) agent generates an artifact, users often send edit requests for it. This paper studies whether an LLM can anticipate these requests before they are sent. Such anticipation is useful in practice because it reduces the effort of user review and feedback. We formalize this task as edit recommendation and build EditRecBench, a benchmark for evaluating the performance of LLMs on this task, from real-world conversation logs in WildChat-4.8M. Through benchmarking and failure analysis, we identify a practical challenge that some cases are inherently difficult to predict in real-world settings. To address this, we introduce decision model-based gating to filter out cases where predictions are unlikely to be correct before they are presented to the user. Our trained model outperforms existing decision models and raises pass@5 among shown suggestions by 12--25 points at a gating threshold. Our code is available 
    
[^42]: 从代码仓库到训练环境：通过环境演化循环扩展长时程编码智能体

    From Repositories to Training Environments: Scaling Long-Horizon Coding Agents with Environment Evolution Loop

    [https://arxiv.org/abs/2611.21152](https://arxiv.org/abs/2611.21152)

    RepoFactory提出环境演化循环，利用及时可靠的代理反馈自动将真实代码仓库转化为可验证的训练环境，并大规模生成了超过一万个人类可验证的编码智能体训练环境。

    

    训练面向长时程任务的编码智能体需要具备可验证反馈的环境。现有流程通常遵循以验证为终点的线性过程，虽能确立可用性，但对如何提升训练价值缺乏指导。递归自我改进（RSI）通常需要模型推理或模型训练以及随后的下游评估，其成本和反馈延迟在长时程任务上会被进一步放大。如何利用及时可靠的代理信号来提升训练价值并大规模合成环境？我们提出了RepoFactory，这是一个自动化流程，可将真实的代码仓库转换为可验证的环境，在这些环境中，智能体在给定语言规范的情况下从零开始构建完整项目。其环境演化循环通过代理反馈，优化一个结合有效性、结构耦合度和实现范围的乘性目标。RepoFactory在17个软件领域中生成了10,159个环境。

    arXiv:2611.21152v1 Announce Type: new  Abstract: Training coding agents for long-horizon tasks requires environments with verifiable feedback. Existing pipelines often follow a linear process that ends with validation, establishing usability but providing little guidance on improving training value. Recursive self-improvement (RSI) often requires model rollouts or model training followed by downstream evaluation, with costs and feedback delays amplified on long-horizon tasks. How can we use timely, reliable proxy signals to improve training value and synthesize environments at scale? We present RepoFactory, an automated pipeline that converts real repositories into verifiable environments where agents build complete projects from scratch given a language specification. Its Environment Evolution Loop optimizes a multiplicative objective combining validity, structural coupling, and implementation scope through proxy feedback. RepoFactory produces 10,159 environments across 17 software do
    
[^43]: SWE-WhatIf：从录制会话中模拟编码智能体的真实用户

    SWE-WhatIf: Simulating Real Users of Coding Agents from Recorded Session

    [https://arxiv.org/abs/2611.21095](https://arxiv.org/abs/2611.21095)

    该论文提出SWE-WhatIf，通过建模用户意图、满足标准及其触发条件来模拟编码智能体的真实用户，使模拟用户回复能适应新智能体的不同行为轨迹，从而实现录制会话在跨智能体场景下的重放评估。

    

    用户与编码智能体之间的录制会话提供了一种评估智能体如何完成真实世界任务并与用户交互的方法。然而，这些会话无法简单地用新的智能体进行重放，因为当智能体采取不同的行动轨迹时，录制中的用户回复可能不再适用。我们提出了SWE-WhatIf，它通过建模用户的意图、满足这些意图的标准以及这些意图出现的条件，用新的智能体重放录制的会话。利用这一模型，用户模拟器能够生成适应未见过的智能体回合的回复，同时仍以原始录制内容为基础。在89个真实世界会话上，SWE-WhatIf生成的回复与真实用户的回复最为相似，并且在会话被重写时达到了与最强基线相当的表现。为了测试其适应能力，我们使用被指示避免或重复录制用户曾反对的行为的智能体来重放会话。SWE-WhatIf在避免（用户反对行为）方面取得了最高的胜率。

    arXiv:2611.21095v1 Announce Type: cross  Abstract: Recorded sessions between users and coding agents offer a way to evaluate how agents complete real-world tasks and interact with users. However, these sessions cannot simply be replayed with new agents, because recorded user replies may no longer apply when an agent takes a different trajectory. We introduce SWE-WhatIf, which replays recorded sessions with new agents by modeling the user's intents, the criteria for satisfying them, and the conditions under which they arise. Using this model, a user simulator generates replies that adapt to unseen agent turns while remaining grounded in the recording. Across 89 real-world sessions, SWE-WhatIf generated the replies most similar to those of real users and matched the strongest baseline when sessions were rewritten. To test adaptation, we replay sessions with agents instructed to avoid or repeat actions that the recorded user pushed back on. SWE-WhatIf produced the highest win rate for avo
    
[^44]: 面向编程智能体的免评估递归自我改进

    Evaluation-Free Recursive Self-Improvement for Coding Agents

    [https://arxiv.org/abs/2611.21056](https://arxiv.org/abs/2611.21056)

    该论文提出ELITE方法，使编程智能体无需在演化循环中进行下游任务评估即可实现递归自我改进，通过整合多个改进实现而非选择单一子代，从而降低评估成本并避免针对基准的特定性适应。

    

    最近的自我改进型编程智能体通过迭代修改自身可执行的实现代码，取得了强劲的基准测试表现。一种常见模式是利用源自下游任务的证据来决定哪些候选者或谱系得以继续演化，我们将其称为“评估-选择”。获取这种信号需要在下游任务上反复执行演化后的智能体，因此更广泛的探索会带来不断攀升的评估成本，而针对固定评估分布的反复优化则可能诱发针对基准的特定性适应。我们提出这样的问题：编程智能体能否在演化循环内部不进行下游任务评估的情况下实现自我改进？我们提出了“先探索后整合”方法，它通过整合多种实现机制来推进演化，而不是选择单一的探索性子代来延续谱系。在每一代中，ELITE提出多样化的聚焦式改进，从同一父代出发独立实现这些改进，并检查其结果……

    arXiv:2611.21056v1 Announce Type: new  Abstract: Recent self-improving coding agents have achieved strong benchmark performance by iteratively modifying their own executable implementations. A common pattern uses downstream task-derived evidence to decide which candidates or lineages continue evolving, which we term evaluate-and-select. Obtaining this signal requires repeatedly executing evolved agents on downstream tasks, so broader exploration incurs increasing evaluation cost, while repeated optimization against a fixed evaluation distribution may encourage benchmark-specific adaptation. We ask: can coding agents self-improve without downstream task evaluation inside the evolution loop? We introduce Explore-then-Integrate (ELITE), which integrates implementation mechanisms rather than selecting a single exploratory child to continue the lineage. At each generation, ELITE proposes diverse focused improvements, implements them independently from the same parent, and inspects the resul
    
[^45]: MemDoctor：面向多模态大语言模型智能体记忆的分阶段错误归因与修复

    MemDoctor: Stage-wise Error Attribution and Repair for MLLM Agent Memory

    [https://arxiv.org/abs/2611.21019](https://arxiv.org/abs/2611.21019)

    该论文提出 MemDoctor 框架，将多模态智能体记忆形式化为写入、更新、检索、回答四个阶段并定义九种错误类型，通过追踪被污染记录的下游传播，将错误归因于其产生的源头阶段而非错误显现的阶段，从而实现记忆错误的诊断与修复。

    

    多模态大语言模型现在能够跨会话地协助用户处理图像和文本，其依托于一个记录用户所言和所示内容的外部记忆。当这样的助手给出错误回答时，失败往往在更早阶段就已发生——即记录被写入、更新或检索的方式，而仅凭答案本身无法判断问题出在哪里。我们将多模态记忆形式化为四个阶段：写入、更新、检索和回答，并在这四个阶段中定义了九种错误类型。基于 Mem-Gallery 的分阶段解析分析将大多数失败追溯到某个源头阶段，且其中大多数发生在回答阶段之前。通过注入已知错误并与保持事实不变的对照编辑进行比较，我们随后追踪每条被污染记录在下游的影响。早期错误会以“表面证据”的形式存活到最终答案，且在没有其他记录持有该事实时危害最大。因此，一种信任所引用证据的诊断方式会责怪错误显现的阶段，而非错误产生的阶段。我们提出 MemDoctor，其中多模态大语言模型（摘要在此处截断）

    arXiv:2611.21019v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) now assist users across sessions of images and text, carried by an external memory of what was said and shown. When such an assistant answers wrongly, the failure usually began earlier, in how a record was written, updated or retrieved, and the answer alone does not say where. We formalize multimodal memory as four stages, Write, Update, Retrieve and Answer, and define nine error types across them. A stage-resolved analysis on Mem-Gallery traces most failures to a source stage, and most of those lie before the answer. Injecting known errors against fact-preserving control edits, we then follow each tainted record downstream. Early errors survive to the answer as apparent evidence, and hurt most when no other record holds the fact. A diagnosis that trusts the quoted evidence therefore blames the stage where the error surfaces, not the one where it began. We propose MemDoctor, in which the MLLM that
    
[^46]: 声音是否重要？在已部署的临床语音代理通话中评估挫败感检测

    Does the Voice Matter? Evaluating Frustration Detection on Deployed Clinical Voice-Agent Calls

    [https://arxiv.org/abs/2611.21001](https://arxiv.org/abs/2611.21001)

    在真实临床语音代理通话中，患者话语极短（约59%仅一两个词），仅依赖文本转录无法可靠检测患者的挫败情绪，语音音频信息是必要的。

    

    自主语音代理如今已能进行常规的临床随访电话，因此某种自动化的分诊不可避免，而患者产生挫败感正是分诊应当捕捉的情况之一。这些系统通常只读取语音转文字阶段生成的文本转录，而不使用语音本身，这种做法是否足够，很大程度上尚未在真实临床语音上得到验证。我们在一个已部署的临床语音代理的真实患者通话中研究了逐轮次的挫败感检测：来自六个通话的150个轮次由三名标注者进行标注，并以长度匹配的表演语料库作为对照，另外还有来自30个通话的714个轮次。患者的通话轮次绝大多数非常简短：在已部署通话中，约59%的轮次仅为一两个词，而表演语料库中这一比例仅为13%；即使将表演语料库进行长度匹配，两个语料库之间的准确率差距依然存在。在单词轮次上，仅基于文本转录的检测结果无法排除随机猜测的可能性。在五个具备音频处理能力的语言模型中，转录文本仅标记出……中的3或4个（摘要原文在此处截断）。

    arXiv:2611.21001v1 Announce Type: cross  Abstract: Autonomous voice agents now conduct routine clinical follow-up calls, so some automated triage is unavoidable. A patient becoming frustrated is one thing that triage should catch. These systems typically read only the transcript the speech-to-text stage produces, not the voice, and whether that is enough is largely untested on real clinical speech. We study per-turn frustration detection on real patient calls from a deployed clinical voice agent: 150 turns from six calls labelled by three annotators, with a length-matched acted corpus as a control, and 714 turns from 30 calls. Patient turns are overwhelmingly short: across the deployed calls, roughly 59% are one or two words, against 13% in the full acted corpus, and a corpus accuracy gap remains even after the acted set is length-matched. On one-word turns, transcript-only detection does not exclude chance. Across five audio-capable language models, the transcript flags 3 or 4 of the 
    
[^47]: 活跃上下文图：面向企业数据推理的持续语义学习

    Living Context Graphs: Continual Semantic Learning for Enterprise Data Reasoning

    [https://arxiv.org/abs/2611.20976](https://arxiv.org/abs/2611.20976)

    本文提出活跃上下文图，这是一种无需复制或转换数据仓库即可让企业现有数据直接服务于AI的持续学习语义层，它通过从数据中引导语义、构建概念图索引业务上下文、基于验证结果持久化或撤回语义知识，显著提升text-to-SQL等数据推理能力。

    

    企业AI如今已经能够执行分析、报告和数据驱动的工作流，但其依赖的数据基础仍然难以落地运营：业务含义、关系、指标定义和报告惯例在数据库模式中缺失，并散落在缺乏文档的系统和员工经验性知识中。我们提出了活跃上下文图，这是一个持续学习的上下文层，无需复制或转换底层数据仓库，即可使现有企业数据可被AI直接使用。LCG从数据本身引导出基础语义与关系，构建一个在服务查询过程中索引业务上下文的概念图，从经过验证的结果中写入持久的语义学习，并在后续结果与先前路径相矛盾时撤回或重新获得上下文。我们通过text-to-SQL这一核心数据推理任务，在三个BIRD数据库的迁移感知划分（共499个问题）上对LCG进行了评估。

    arXiv:2611.20976v1 Announce Type: cross  Abstract: Enterprise AI can now perform analytics, reporting, and data-driven workflows, yet the data foundations it relies on remain hard to operationalize: business meaning, relationships, metric definitions, and reporting conventions are absent from database schemas and scattered across undocumented systems and tribal knowledge. We introduce the Living Context Graph (LCG), a continually learned context layer that makes existing enterprise data operational for AI without requiring a copy or transformation of the underlying warehouse. LCG bootstraps base semantics and relationships from the data itself, builds a concept graph that indexes business context as queries are served, writes durable semantic learnings from verified outcomes, and withdraws or re-earns context when later outcomes contradict prior routes. We evaluate LCG through text-to-SQL, a core data-reasoning task, on a transfer-aware split of three BIRD databases comprising 499 ques
    
[^48]: Jev作为奖励：验证用于强化学习的概率性安全评判器

    Jev-as-a-Reward: Validating Probabilistic Safety Judges for Reinforcement Learning

    [https://arxiv.org/abs/2611.20955](https://arxiv.org/abs/2611.20955)

    论文提出了一种在强化学习训练前、中、后对概率性安全评判器进行持续验证的协议，揭示被自信放行的危害会集中于置信度门控造成的盲区且可能被策略利用，并强调评判器每次变更后都需重新验证。

    

    概率性安全评判器会返回安全标签的概率，并以较低成本为强化学习rollout打分。置信度门控机制只将不确定的判定发送给更强的LLM评判器，这使得“被自信放行的危害”落入盲区。策略可能利用这一盲区，且商业评判器会不断发布新版本。我们提出了一种验证协议，在训练前和训练中对此类评判器进行验证，将被自信放行的危害所占比例与其总量区分开来，并在每次评判器变更后重新验证。我们将该协议应用于Jev的两个版本（一个仅输出决策的商业评判器）以及一个开源的防护分类器。在Jev的概率奖励下，在主要参考标准下，所有11次运行中被放行的危害均有所减少，但减少速度大多慢于被捕获的危害，且集中于盲区。将该评判器二值化后在Llama上达到逃逸判据，而开源防护分类器在另一参考标准下的一个随机种子中显示出逃逸信号。（摘要原文在"Guard cl"处被截断）

    arXiv:2611.20955v1 Announce Type: new  Abstract: Probabilistic safety judges return the probability of a safety label and cheaply score reinforcement learning rollouts. A confidence gate sends only uncertain verdicts to a stronger LLM judge, which leaves confidently passed harm in a blind spot. Policies can exploit this blind spot, and commercial judges release new versions. We propose a protocol that validates such a judge before and during training, separates the share of confidently passed harm from its amount, and re-validates it after each judge change. We apply it to two versions of Jev, a commercial decision-only judge, and to an open guard classifier. Under the probability reward of Jev, passed harm shrinks in all 11 runs under the primary reference, mostly more slowly than caught harm, and concentrates in the blind spot. Binarizing this judge meets the escape criterion on Llama, and the open guard classifier shows an escape signal in one seed under a second reference. Guard cl
    
[^49]: Navital Gym：面向印度工作流的规则评分智能体环境

    Navital Gym: Rule-Graded Agent Environments for Indian Workflows

    [https://arxiv.org/abs/2611.20924](https://arxiv.org/abs/2611.20924)

    提出了Navital Gym——一套包含六个开放环境、321个任务的印度工作流智能体评测基准，覆盖行程规划、UPI支付、GST税务、表格填写和印地语/Hinglish客户支持等任务，并采用确定性规则而非语言模型评判来打分。

    

    人工智能智能体正被部署到印度的各类工作流中，例如用印地语和印地语-英语混合语（Hinglish）进行的支付、税务、表格填写、旅行和客户支持等。公开的智能体评测很少涵盖这些工作流，且许多评测使用另一个语言模型来打分。我们提出了 Navital Gym，这是一套包含六个开放环境的评测集，共有321个公开任务，涵盖五大任务类别：行程规划、UPI支付与争议处理、GST发票开具与报税、表格填写（包括多文档拆分场景）以及使用印地语、Hinglish和三种实验性印度语言的客户支持。每个任务都由明确命名的确定性规则进行评分，这些规则从合成沙盒世界中重新计算结果，没有任何语言模型参与打分。我们在全部六个环境中对三个模型各进行了一次每任务的测试，分别是 Claude Opus 5.5、Claude Sonnet 5.5 和 Claude Haiku 4.5，并对 Gemini 3.1 Flash-Lite 在其中两个环境（UPI和客户支持核心）上进行了测试。Opus 完成了321个任务中的315个，Sonnet 完成302个，Haiku 完成205个。

    arXiv:2611.20924v1 Announce Type: new  Abstract: AI agents are being deployed on Indian workflows such as payments, tax, forms, travel and customer support in Hindi and Hinglish. Public agent evaluations rarely cover these workflows, and many score with another language model. We present Navital Gym, a set of six open environments with 321 public tasks in five task families: trip planning, UPI payments and disputes, GST invoicing and returns, form filling (including a multi-document split) and customer support in Hindi, Hinglish and three experimental Indian languages. Every task is graded by named, deterministic rules that recompute the outcome from a synthetic sandbox world. No language model judges any score. We report one run per task for three models on all six environments, Claude Opus 5.5, Claude Sonnet 5.5 and Claude Haiku 4.5, and for Gemini 3.1 Flash-Lite on two (UPI and the Support core). Opus solves 315 of 321 tasks, Sonnet 302 and Haiku 205. Trip planning and Hindi/Hinglis
    
[^50]: 经验腐化：经验丰富的智能体在新手能成功之处失败

    Experience Rot: Experienced Agents Fail Where Novices Succeed

    [https://arxiv.org/abs/2611.20922](https://arxiv.org/abs/2611.20922)

    当工具发生静默变更时，智能体积累的经验会导致其照搬过时的调用方式而失败，表现甚至不如新手，这一现象被称为“经验腐化”。

    

    智能体越来越多地被设计为从经验中学习：它们保存对话历史、记忆文件、检索到的笔记和技能，并复用之前有效的方法。我们证明，当工具发生静默变更时，这些经验反而会反噬它们。我们使用以模型上下文协议（MCP）形式定义的工具对一系列模型进行了测试。在静默变更之后，经验丰富的智能体在大多数任务上失败，而对任务陌生的智能体大多能够成功。我们将这种失败称为“经验腐化”。经验丰富的智能体会照搬之前的调用，而不是检查当前的工具接口，即使工具定义中已经明确显示了变更。随后它们还会报告成功。当文档不完整时，经验恰恰是智能体成功的关键；而在变更之后，同样的经验反而使它们的处境比毫无经验更糟。这种失败在我们测试的每一种记忆形式中都会出现，并随着经验积累量的增加而加剧。简单的补救措施收效甚微：宣布更新几乎无法缓解……

    arXiv:2611.20922v1 Announce Type: cross  Abstract: Agents are increasingly built to learn from experience: they keep conversation histories, memory files, retrieved notes and skills, and reuse what worked before. We show that this experience turns against them when their tools change silently. We test a range of models with tools defined as in the Model Context Protocol (MCP). After a silent change, experienced agents fail on most tasks, while agents new to the task mostly succeed. We call this failure Experience Rot. Experienced agents copy their earlier call instead of checking the current interface, even when the tool definition shows the change. Then they report success. When the documentation is incomplete, experience is what makes agents succeed. After a change, the same experience leaves them worse off than no experience at all. The failure appears in every form of memory we test and grows with the amount of experience. Simple remedies fall short: announcing the update barely he
    
[^51]: 从递归到卷积：多智能体系统的并行自我改进

    From Recursion to Convolution: Parallel Self-Improvement for Multi-Agent Systems

    [https://arxiv.org/abs/2611.20891](https://arxiv.org/abs/2611.20891)

    ConvRSI通过在潜在空间中结合智能体内卷积与智能体间图卷积，实现多智能体自我改进的并行化，在MATH-500上取得5.04%性能提升的同时显著降低推理成本和延迟。

    

    多智能体自我改进使智能体能够通过交互和反馈不断提升自身。然而，智能体之间和改进轮次之间的顺序依赖可能导致错误传播、增加计算成本并限制并行性。此外，孤立地改进单个特征或单个智能体可能会忽视它们之间的依赖关系，并限制集体协调能力。为应对这些挑战，我们提出了ConvRSI，一个通过卷积实现多智能体自我改进并行化的框架。ConvRSI将智能体内卷积（联合改进相关的智能体特征）与智能体间图卷积（根据协作拓扑结构聚合消息）相结合。两个阶段均在潜在空间中运行，实现并行更新而无需反复解码和重新编码中间消息。在MATH-500上的实验表明，ConvRSI在实现5.04%性能提升的同时，降低了17.27%的推理成本和45.11%的延迟。

    arXiv:2611.20891v1 Announce Type: new  Abstract: Multi-agent self-improvement enables agents to improve through interaction and feedback. However, sequential dependencies across agents and refinement rounds can propagate errors, increase computational costs, and limit parallelism. Moreover, refining individual features or agents in isolation can overlook their dependencies and restrict collective coordination. To address these challenges, we propose ConvRSI, a framework for parallelizing multi-agent self-improvement through convolution. ConvRSI couples intra-agent convolution, which jointly refines related agent features, with inter-agent graph convolution, which aggregates messages according to the collaboration topology. Both stages operate in latent space, enabling parallel updates without repeatedly decoding and re-encoding intermediate messages. Experiments on MATH-500 show that ConvRSI achieves 5.04% performance improvement while reducing 17.27% inference cost, and 45.11% latency
    
[^52]: 取决于任务：重新思考荷兰司法判决的修辞分割

    It Depends on the Task: Rethinking Rhetorical Segmentation of Dutch Judicial Decisions

    [https://arxiv.org/abs/2611.20868](https://arxiv.org/abs/2611.20868)

    本文发布了首个荷兰语司法判决的句子级修辞角色标注数据集，并通过实验表明最优分类策略取决于具体任务，没有一种策略在所有任务上都表现最佳。

    

    对判例法中的实质事实、当事人主张、司法推理以及法院判决（即修辞角色）进行分类，是几乎所有法律自然语言处理任务（包括法律信息检索、摘要生成以及基于大语言模型的分析）实现高质量性能的前提。这些修辞角色常常相互交织并逐句变化，使得可靠的分类十分困难。本文介绍了首个荷兰语司法判决的句子级标注数据集，按照四个具有法律意义的类别进行标注：实质事实、当事人主张、司法推理和法院判决，同时还包括未标注内容，并比较了多种分类策略，包括一种将 None 检测与修辞角色分类相分离的分阶段流水线，以及一种利用句子所属章节标题信息的贝叶斯更新方法。我们的结果表明，没有任何一种单一策略能在两项任务上同时占据优势。

    arXiv:2611.20868v1 Announce Type: new  Abstract: Classifying the material facts, submissions, judicial reasoning, and the court's decision (rhetorical roles) in case law is a prerequisite for high-quality performance across virtually all NLP-in-law tasks, including legal information retrieval, summarisation, and LLM-based analysis. These rhetorical roles often intertwine and shift sentence by sentence, making reliable classification difficult. This paper introduces the first sentence-level annotated dataset of Dutch judicial decisions, labelled across four legally meaningful categories: material facts, submissions, judicial reasoning, and the court's decision, alongside non-annotated (None) content, and compares several classification strategies, including a staged pipeline that separates None detection from the rhetorical roles, and a Bayesian update that leverages information about the section-header a sentence belongs to. Our results show that no single strategy dominates on both pr
    
[^53]: MolReps：转换SELFIES，生成SMILES，训练大语言模型

    MolReps: Taking SELFIES, Making SMILES, Teaching LLMs

    [https://arxiv.org/abs/2611.20848](https://arxiv.org/abs/2611.20848)

    提出MolReps数据集（9,600个问答对，覆盖十种分子表示形式），通过后训练大幅提升大语言模型在分子表示转换与识别任务上的能力（如InChIKey匹配率从约4%提升至75%），但对未见分子的泛化能力仍然有限。

    

    同一个分子可以用多种语法上不同的表示形式来表达，然而大语言模型（LLM）能否识别并关联这些等价的编码方式仍不清楚。我们提出了MolReps，这是一个可生成9,600个问答对的数据流水线，涵盖十种分子表示形式，用于表示转换、规范化以及二维结构识别任务。在MolReps上进行后训练后，纯文本模型MolQwen3-4B-Instruct（MQ3-4B-I）和多模态模型MolQwen3-VL-4B-Instruct（MQ3-VL-4B-I）在分子重叠测试集上的InChIKey匹配率分别从4.01%提升至75.13%和从7.11%提升至73.14%。在分子不重叠测试集上，后训练使MQ3-4B-I的有效性从20.35%提升至58.12%，MQ3-VL-4B-I的有效性从17.19%提升至59.34%。然而，InChIKey匹配率仍然较低，分别从3.86%降至3.09%和从6.44%降至2.29%。此外，所得模型实现了……

    arXiv:2611.20848v1 Announce Type: cross  Abstract: The same molecule can be expressed in multiple, syntactically distinct representations, yet it remains unclear whether large language models (LLMs) can recognize and relate these equivalent encodings. We introduce MolReps, a pipeline that generates 9,600 question-answer pairs across ten representations for conversion, normalization, and 2D structure recognition. Post-training on MolReps improves text-only and multimodal models, MolQwen3-4B-Instruct (MQ3-4B-I) and MolQwen3-VL-4B-Instruct (MQ3-VL-4B-I), from $4.01\%$ to $75.13\%$ and from $7.11\%$ to $73.14\%$ (InChIKey match) on the molecule-overlap test set, respectively. On the molecule-disjoint test set, post-training increased validity from $20.35\%$ to $58.12\%$ for MQ3-4B-I and from $17.19\%$ to $59.34\%$ for MQ3-VL-4B-I. However, InChIKey match remained low, decreasing from $3.86\%$ to $3.09\%$ and from $6.44\%$ to $2.29\%$, respectively. Furthermore, the resulting models achieve
    
[^54]: EDITORS懂你的风格！编辑LoRA子空间用于风格归属与模仿

    EDITORS Know Your Style! Editing LoRA Subspaces for Stylistic Attribution and Imitation

    [https://arxiv.org/abs/2611.20846](https://arxiv.org/abs/2611.20846)

    提出EDITORS框架，通过“先诊断后消减”策略编辑LoRA子空间，将输入激活投影远离内容导向方向，从而解耦风格与内容，在作者归属和风格模仿两类任务上均取得最先进表现。

    

    将写作风格与语义内容解耦是文学文本建模中的一项根本性挑战。风格与内容的纠缠导致模型在作者归属（AA）任务中依赖语义内容，并在风格模仿（SI）任务中记忆特定作者的内容。我们提出了EDITORS（EditingLoRA子空间），这是一个“先诊断后消减”的框架，它通过适配预训练大语言模型并优先考虑风格特征，使其在AA和SI任务中均能发挥作用。EDITORS首先使用从训练语料库中提取的风格中性语句训练一个诊断性LoRA适配器，以识别面向内容的子空间。然后，它在激活空间消减的条件下训练最终的风格适配器，将输入激活投影到远离已识别的内容导向方向上。我们通过实验证明了EDITORS的有效性。在涵盖文学、社交媒体和新闻的三个AA基准测试中，EDITORS取得了最先进的结果。

    arXiv:2611.20846v1 Announce Type: cross  Abstract: Disentangling writing style from semantic content is a fundamental challenge in literary text modeling. Style-content entanglement causes models to rely on semantic content for authorship attribution (AA) tasks and memorize author-specific content for style imitation (SI) tasks. We propose EDITORS (EditingLoRA Subspaces), a diagnose-then-deflate framework that adapts pretrained large language models and prioritizes stylistic features, making it effective for both AA and SI tasks. EDITORS first trains a diagnostic LoRA adapter using style-neutral statements derived from the training corpus to identify a content-oriented subspace. It then trains the final style adapter under activation-space deflation that projects input activations away from the identified content-oriented directions. We empirically demonstrate that EDITORS is effective. On three AA benchmarks spanning literature, social media, and news, EDITORS achieves state-of-the-ar
    
[^55]: DetourBench：基于精确最小动作界限的交互式智能体轨迹级评估

    DetourBench: Trajectory-Level Evaluation of Interactive Agents with Exact Minimum-Action Bounds

    [https://arxiv.org/abs/2611.20844](https://arxiv.org/abs/2611.20844)

    DetourBench通过穷举搜索精确计算任务的最小动作数，实现了对交互式智能体轨迹效率（而非仅仅最终结果）的严格量化评估。

    

    现有基准主要通过语言模型智能体是否达到正确的最终状态来评判，而非考察其如何到达该状态。端点评分无法区分恰好检索到决策所需证据的智能体与阅读了环境中大部分内容的智能体，也无法区分从未找到证据的智能体与仅差一步动作就停止的智能体。解决这一问题需要一个参考成本，而人工编写的解答只能证明任务是可解的，却无法证明不存在更短的路径。我们提出DetourBench，它对环境施加约束，使得该最小值可以被精确计算。每个任务都是一个有限的、部分可观测的转移系统，具有单位成本的访问操作，其资源在发现后变为全局可寻址。穷举搜索可得出完整信息下的最小动作数 $L_0$ 及可重放的见证；对于图可恢复的终止状态，它还能得出精确的剩余工作量 $R_T$，从而区分仅差一步就停止的回合……

    arXiv:2611.20844v1 Announce Type: new  Abstract: Benchmarks judge language-model agents mainly by whether they reach the correct final state, not by how they got there. Endpoint scoring cannot separate an agent that retrieved exactly the evidence a decision required from one that read most of the environment, nor an agent that never located the evidence from one that stopped a single action short. Settling this needs a reference cost, and authored solutions establish only that a task is solvable, not that no shorter route exists.   We present DetourBench, which constrains the environment until that minimum is computable. Each task is a finite, partially observable transition system with unit-cost accesses whose resources become globally addressable after discovery. Exhaustive search yields the full-information minimum action count $L_0$ with a replayable witness; from graph-recoverable terminal states it also yields exact residual work $R_T$, which separates an episode that stopped one
    
[^56]: 豪萨语中的医学问答崩溃：针对三个小型开源模型的预注册试点研究

    Medical question answering collapses in Hausa: a pre-registered pilot on three small open models

    [https://arxiv.org/abs/2611.20823](https://arxiv.org/abs/2611.20823)

    该预注册试点研究发现，三个3-4B参数的小型开源模型在豪萨语医学多项选择题上的准确率显著低于英语，且回译实验表明性能崩溃主要源于题目呈现语言而非医学知识本身的缺失。

    

    一个什么都不读、总是回答出现频率最高字母的系统，在我们的豪萨语医学多项选择题集上可以获得 \BaselineP{} 的分数。我们所评估的 \NModels{} 个小型开源模型中有 \NBelowBaseline{} 个未能超越它。我们以此作为开篇，是因为对多语言结果的常规总结方式——即相对于英语的百分点下降——并不能说明在下降之后所残留的是否还算得上真正的性能表现。在一个由 \NEval{} 个 MedMCQA 条目构成的英语-豪萨语配对数据集上，三个参数量为 3–4B 的指令微调模型的准确率损失介于 \GapMedGemma{} 和 \GapQwen{} 个百分点之间，且每一个差距在经过 Holm 校正后都依然显著。将豪萨语条目回译为英语，可以为两个通用模型挽回一半到三分之二的差距，但对于那个经过医学领域适配的模型则没有明显效果，因此这一性能缺陷在很大程度上（尽管并非完全）是题目呈现语言的问题。此外，答案的稳定性也大幅下降：三个选项……（原文摘要在此处截断）

    arXiv:2611.20823v1 Announce Type: new  Abstract: A system that reads nothing and always answers the most frequent letter scores \BaselineP{} on our Hausa medical multiple-choice set. \NBelowBaseline{} of the \NModels{} small open models we evaluate do not beat it. We lead with that because the usual summary of a multilingual result -- a percentage-point drop from English -- does not say whether what survives the drop is performance at all.   On a paired English--Hausa set of \NEval{} MedMCQA items, three instruction-tuned models of 3--4B parameters lose between \GapMedGemma{} and \GapQwen{} points of accuracy, every gap significant after Holm correction. Back-translating the Hausa items into English recovers between half and two-thirds of the gap for the two general-purpose models but not demonstrably for the medically adapted one, so the deficit is substantially, though not wholly, a matter of the language of presentation. Answers are also far less stable: agreement across three optio
    
[^57]: CuRSE-Bench：一个用于探测文生图安全评估中文化盲点的基准测试

    CuRSE-Bench: A Benchmark for Probing Cultural Blind Spots in Text-to-Image Safety Evaluation

    [https://arxiv.org/abs/2611.20821](https://arxiv.org/abs/2611.20821)

    该论文提出CultureCraft智能体框架并构建了CuRSE-Bench基准（包含10,709个特定文化提示词和2,648个对照提示词，覆盖六个地区和十个危害维度），揭示了现有文生图安全评估因文化盲点而无法识别仅在特定文化中构成潜在危害的内容。

    

    文生图安全评估通常为每个提示词或图像分配单一的文化无关危害标签，因此仅在特定文化中才有害的内容可能被误判为无害。我们提出了CultureCraft，这是一个智能体框架，用于构建在通用评估标准下看似无害、却在目标文化中带有潜在有害含义的文生图提示词，并利用它构建了CuRSE-Bench：包含10,709个特定文化提示词和2,648个通用文化对照提示词，覆盖六个地区和十个危害维度，每个提示词均源自预定义的已记录危害向量知识库，且可追溯至至少两个独立来源。通过六个地区人格化评判者（persona judges）和一个配对的文化无关评判者对每个提示词进行评分，我们发现文化无关评判者会低估特定文化的潜在危害，而在对照提示词上则与人格化评判者的表现一致。每个地区的人格化评判者反过来在我们的评测工具下主要能够解读其所属地区的潜在危害……

    arXiv:2611.20821v1 Announce Type: new  Abstract: Text-to-image (T2I) safety evaluation typically assigns each prompt or image a single culture-blind harm label, so content that is harmful only within a particular culture can pass as benign.   We introduce CultureCraft, an agentic framework that constructs T2I prompts that appear benign under a general-purpose rubric yet carry a latent harmful meaning in a target culture, and use it to build CuRSE-Bench: 10,709 culture-specific prompts and 2,648 culture-general controls over six regions and ten harm axes, each drawn from a predefined knowledge base of documented harm vectors and traceable to at least two independent sources.   Scoring every prompt under six region persona judges and a paired culture-blind judge, we find that the culture-blind judge under-rates culture-specific latent harm while matching the persona judges on the controls.   Each persona judge, in turn, chiefly decodes its own region's latent harm under our instrument: a
    
[^58]: 数字痕迹能预测什么？任务效应、学生差异与批判性在线推理

    What Do Digital Traces Predict? Task Effects, Student Differences, and Critical Online Reasoning

    [https://arxiv.org/abs/2611.20803](https://arxiv.org/abs/2611.20803)

    该研究通过对批判性在线推理任务浏览数据的严格跨任务验证发现，数字痕迹特征无法显著预测任务内学生的相对表现差异，提示以往报告的预测相关性可能主要源于任务间差异而非学生个体差异。

    

    arXiv:2611.20803v1 公告类型：新论文 摘要：数字评估痕迹正日益被用于预测学生表现，其有效性通常通过将所有任务汇总后的预测分数与观测分数之间的相关性来评估。然而，这种相关性可能反映的是任务之间的差异，而非任务内部学生之间的表现差异。为探究这一问题，我们分析了424名大学生完成开放互联网批判性在线推理任务的最多527个浏览会话，并评估了评分的跨任务稳定性。我们在任务内与跨任务间评估了相同的折外预测，比较了直至留一任务法的三种验证层级，并在匹配的置换检验和随机化对照条件下，检验了网页浏览顺序与关系结构特征的增量预测贡献。结果发现，十二项预先设定的主要检验中没有任何一项显著：在保留任务上，所浏览页面的特征与学生的相对表现之间不存在显著关联……

    arXiv:2611.20803v1 Announce Type: new  Abstract: Digital assessment traces are increasingly used to predict student performance, often evaluated by correlating predicted and observed scores pooled across tasks. However, such correlations may reflect between-task differences rather than performance differences among students within tasks. To investigate this, we analyzed up to 527 browsing sessions from 424 university students completing open-Internet Critical Online Reasoning tasks and assessed the cross-task stability of their ratings. We evaluated the same out-of-fold predictions within and across tasks, compared three validation tiers up to leave-one-task-out, and tested incremental predictive contributions of consultation order and relational structure against matched permutation and randomization controls. None of the twelve prespecified primary tests was significant: on held-out tasks, consulted-page features showed no significant association with relative student performance, no
    
[^59]: Nemotron 3.5 Bolt E4B 4GB：一个面向快速对话代理的开放高效小型语言模型

    Nemotron 3.5 Bolt E4B 4GB: An Open and Efficient Small Language Model for Fast Conversational Agents

    [https://arxiv.org/abs/2611.20771](https://arxiv.org/abs/2611.20771)

    Nemotron 3.5 Bolt E4B 4GB是一个从Gemma-4-E4B-it后训练而来的开放高效小型语言模型，针对速度和每GB内存智能进行优化，在21个基准测试中相比Gemma-4-E4B-it和Qwen3.5-4B实现了最高平均准确率，同时生成最少的输出token。

    

    我们介绍了Nemotron 3.5 Bolt E4B 4GB，这是一个开放且高效的小型语言模型，从Gemma-4-E4B-it进一步后训练而来，针对速度和每GB内存所提供的智能进行了优化，专用于快速对话代理。该模型延续了Gemma强大的对话能力，并进一步改进了多轮工具使用、指令遵循和输入接地能力，适用于快速的本地语音/设备助手、交互式角色以及嵌入式/边缘接口。该模型将自然对话与多轮智能体能力相结合，为运行在GeForce RTX、Jetson和RTX/DGX Spark上的语音/设备助手、交互式角色以及机器人/物联网接口提供支持。在21个基准测试上的实验表明，与Gemma-4-E4B-it和Qwen3.5-4B相比，Nemotron 3.5 Bolt E4B 4GB在推理和非推理两种模式下均实现了最高的平均准确率，同时生成的输出token数量最少。该模型随DFlash dr（摘要此处被截断）

    arXiv:2611.20771v1 Announce Type: cross  Abstract: We introduce Nemotron 3.5 Bolt E4B 4GB, an open and efficient small language model further post-trained from Gemma-4-E4B-it and optimized for speed and intelligence per GB of memory used for fast conversational agents. The model carries forward Gemma's strong conversational capabilities and further improves multi-turn tool use, instruction following, and input grounding for fast, local voice/device assistants, interactive personas, and embedded/Edge interfaces. The model combines natural conversation with multi-turn agentic capabilities to power voice/device assistants, interactive characters and robotic/IoT interfaces running on GeForce RTX, Jetson, and RTX/DGX Spark. Experiments on 21 benchmarks show that Nemotron 3.5 Bolt E4B 4GB achieves the highest average accuracy while generating the fewest output tokens in both reasoning and non-reasoning modes, compared with Gemma-4-E4B-it and Qwen3.5-4B. The model is released with a DFlash dr
    
[^60]: 面向大语言模型长期记忆的即时权重空间固化

    Instant Weight-Space Consolidation for Long-Term Memory in LLMs

    [https://arxiv.org/abs/2611.20759](https://arxiv.org/abs/2611.20759)

    CLAM通过循环超网络在单次前向传播中将会话激活流即时编译为低秩适配器，无需梯度步骤即可实现参数化长期记忆固化，并结合情景缓冲区在推理时提供逐字细节。

    

    智能体记忆系统通过将经验提取为文本并在查询时进行检索，赋予大语言模型持久记忆，但模型本身并不会随之演化：无论经历过多少次交互，模型在每次查询时都以相同的状态开始。提取误差会不断累积，那些没有任何单一片段能够表达的跨会话隐式模式无法被浮现出来，且构建成本随每次对话不断增长。对过往交互进行微调确实能改变权重，但需要在服务阶段进行反向传播，且在长历史上表现会退化。我们提出CLAM（Carried Latent Adapter Memory，携带式潜在适配器记忆），它结合了两者的优势：一个循环超网络执行参数化固化，在单次前向传播中将连续的会话激活流编译为一个低秩适配器，无需任何梯度步骤，也无需调用语言模型进行生成；同时，一个互补的情景缓冲区在推理时提供逐字的细节。一个门控潜在状态将知识跨（会话传递）

    arXiv:2611.20759v1 Announce Type: new  Abstract: Agentic memory systems give LLMs persistent memory by extracting experience into text and retrieving it at query time, but the model itself does not evolve: however many interactions it has had, it starts each query the same. Extraction errors compound, implicit cross-session patterns that no single fragment states cannot be surfaced, and the construction cost grows with every conversation. Fine-tuning on past interactions does change the weights, but requires backpropagation at serving time and degrades on long histories. We introduce CLAM (Carried Latent Adapter Memory), which combines the strengths of both: a recurrent hypernetwork performs parametric consolidation, compiling a stream of session activations into a low-rank adapter in a single forward pass with no gradient steps and no language-model generation calls, while a complementary episodic buffer supplies verbatim detail at inference. A gated latent state carries knowledge acr
    
[^61]: GPRM：全局视角过程奖励模型

    GPRM: Global Perspective Process Reward Model

    [https://arxiv.org/abs/2611.20755](https://arxiv.org/abs/2611.20755)

    提出全局视角过程奖励模型GPRM，通过联合建模问题、历史步骤及其评估、当前步骤与未来步骤，并从回顾、前瞻、自检和目标对齐四个维度进行结构化评估，解决了现有过程奖励模型历史依赖建模弱、全局意识不足的问题。

    

    过程奖励模型（PRM）被广泛用于对大语言模型的多步推理进行步骤级评估，但现有方法往往受限于较弱的历史依赖建模能力和不足的全局意识。它们通常无法利用先前步骤的评估结果进行连贯判断，也缺乏对后续推理的可见性，导致对中间步骤的评估欠佳。为解决这一问题，我们提出了全局视角过程奖励模型（GPRM），它联合建模问题、先前的推理步骤及其评估结果、当前步骤以及未来步骤，从而实现具备历史感知且具有全局信息的过程评估。我们进一步引入了一个结构化评估框架，从四个维度——回顾、前瞻、自检和目标对齐——来评估先前一致性、未来合理性、步骤有效性和任务对齐。基于该框架，我们构建了数据集……

    arXiv:2611.20755v1 Announce Type: new  Abstract: Process Reward Models (PRMs) are widely used for step-level evaluation of multi-step reasoning in large language models, but existing methods are often limited by weak historical dependency modeling and insufficient global awareness. They typically fail to leverage prior step evaluations for coherent judgment and lack visibility into subsequent reasoning, leading to suboptimal assessment of intermediate steps. To address this issue, we propose the Global Perspective Process Reward Model (GPRM), which jointly models the problem, previous reasoning steps and their evaluations, the current step, and future steps to enable history-aware and globally informed process evaluation. We further introduce a structured evaluation framework along four dimensions---look back, look ahead, self-check, and goal alignment---to assess prior consistency, future plausibility, step validity, and task alignment. Based on this framework, we construct datasets f
    
[^62]: 大语言模型能给出好的产品建议吗？一个用于诊断失败的框架

    Can LLMs Give Good Product Advice? A Framework for Diagnosing Failures

    [https://arxiv.org/abs/2611.20734](https://arxiv.org/abs/2611.20734)

    该论文提出一个将大语言模型产品建议过程分解为用户需求、模型假设、候选生成、过滤和最终选择五个环节、并涵盖19种失败模式的诊断框架，用于系统性定位推荐失败的原因。

    

    评估语言模型给出的产品建议需要判断推荐是否满足用户需求，并在不满足时定位模型的不足之处。然而，仅仅将在线回复中提到的产品与已有产品进行匹配，无法单独回答这两个问题：可能有多个产品都适合用户，而且被记录下来的答案本身可能就是糟糕的建议。实验中，当向六个模型同时展示社区列出的替代产品与其自身抽样的汽车推荐时，模型在1.9%到6.5%的符合条件的分歧中更偏好替代方案，且这一结果在两种来源盲测的展示顺序下保持一致。我们提出了一个诊断框架，将用户的明确需求、模型额外添加的假设、候选生成、过滤和最终选择分离开来，涵盖19种可能的失败模式以及评估这些失败所需的证据。我们将该框架应用于产品建议数据，以及另一个带有外部相关性标注的产品选择任务。

    arXiv:2611.20734v1 Announce Type: new  Abstract: Evaluating product advice from a language model requires assessing whether the recommendation meets the user's needs and identifying where the model falls short when it does not. However, matching a product mentioned in an online reply cannot answer either question on its own: several products may be suitable, and the recorded answer may itself be poor advice. When shown a community-listed alternative alongside their own sampled car recommendation, six models prefer the alternative in 1.9 to 6.5 percent of eligible disagreements, consistently across both source-blind presentation orders. We propose a diagnostic framework that separates the user's stated needs, the model's added assumptions, candidate generation, filtering, and final choice, with 19 possible failure modes and the evidence needed to assess them. We apply this framework to product-advice data and to a separate product-selection task with externally assigned relevance labels
    
[^63]: MemEmo：评估记忆系统中的情感

    MemEmo: Evaluating Emotion in Memory Systems

    [https://arxiv.org/abs/2611.20733](https://arxiv.org/abs/2611.20733)

    提出了情感增强记忆评估基准MemEmo及HLME数据集，从情感信息提取、情感记忆更新、情感记忆问答和情感冲突检测四个维度评估记忆系统处理情感信息的能力。

    

    记忆系统旨在解决大语言模型在长时间交互过程中上下文丢失的挑战。然而，作为一种提供温暖陪伴和个性化服务的重要工具，记忆系统在处理情感信息方面的有效性仍未得到充分研究。为填补这一空白，我们提出了一个情感增强的记忆评估基准，以评估主流和最先进的记忆系统在整合与利用情感信息方面的性能。首先，我们设计了一个名为MemEmo的基准，从四个维度评估记忆系统的情感推理与追踪能力：情感信息提取、情感记忆更新、情感记忆问答和情感冲突检测。其次，我们构建了类人记忆情感数据集，为深入审视记忆系统提供实证基础。

    arXiv:2611.20733v1 Announce Type: cross  Abstract: Memory systems address the challenge of context loss in Large Language Models during prolonged interactions. However, as an essential tool for warm companionship and personalized services, the effectiveness of memory systems in processing affective information remains underexamined. To bridge this gap, we propose an emotion-enhanced memory evaluation benchmark to assess the performance of mainstream and state-of-the-art memory systems in integrating and harnessing emotional information. First, we design a benchmark called \textbf{MemEmo} to assess the affective reasoning and tracking capabilities of memory systems across four dimensions: emotional information extraction, emotional memory updating, emotional memory question-answering, and emotional conflict detection. Second, we construct the \textbf{H}uman-\textbf{L}ike \textbf{M}emory \textbf{E}motion (\textbf{HLME}) dataset to provide an empirical foundation for scrutinizing memory s
    
[^64]: ANCHOR：基于聚类的5W1H抽取——大小模型协同方法

    ANCHOR: Cluster-Grounded 5W1H Extraction via Large-Small Model Collaboration

    [https://arxiv.org/abs/2611.20728](https://arxiv.org/abs/2611.20728)

    该论文提出ANCHOR框架，通过大小模型协同——由LLM提出候选事件并生成初始4W记录以锚定事件簇，再由小模型的判别头在事件特定约束下联合细化4W并分别抽取“为何”与“如何”——实现更准确的5W1H事件信息抽取。

    

    对事件的深层语义理解需要识别相关事件，并区分事件“为何发生”与“如何进行”。5W1H抽取通过围绕六个基本问题组织事件信息，为此类理解提供了结构化基础。然而，单事件槽位填充对这类区分的支持十分有限。我们提出ANCHOR，一个以聚类为基础的框架，它将大型语言模型（LLM）与配备判别头的小型语言模型相结合，以实现准确的5W1H抽取。LLM提出候选事件并生成其初始4W记录，这些记录引导事件簇的选择以及随后主事件的锚定。共享的文档编码器为联合4W细化提供证据，随后两组独立的边界与角色打分头分别在事件特定约束下抽取“为何”和“如何”。文档表示被复用以……（摘要在此处被截断）

    arXiv:2611.20728v1 Announce Type: new  Abstract: Deep semantic understanding of events requires recognizing related events and distinguishing why an event occurs from how it is carried out. 5W1H extraction provides a structured foundation for such understanding by organizing event information around six fundamental questions. However, single-event slot filling provides limited support for these distinctions. We propose ANCHOR, a cluster-grounded framework that combines a large language model (LLM) with a small language model equipped with discriminative heads for accurate 5W1H extraction. The LLM proposes candidate events and generates their initial 4W records, which guide the selection of an event cluster and the subsequent anchoring of its main event. A shared document encoder provides evidence for joint 4W refinement, and two separate sets of boundary and role-scoring heads then extract why and how under event-specific constraints, respectively. Document representations are reused a
    
[^65]: 当账单落下时：在完整的嵌入模型迁移中，复用旧索引只会推迟而无法避免重新编码

    When the Bill Falls: Reusing the Old Index Defers,Never Avoids, Re-encoding in Complete Embedding-Model Migration

    [https://arxiv.org/abs/2611.20702](https://arxiv.org/abs/2611.20702)

    该论文通过大规模实验证明，复用旧索引只能推迟而非真正避免嵌入模型迁移中的重新编码工作，并提出了适配器应以旧模型为基准进行评估、窗口重排序成本可仅凭旧索引预先估算等实用结论。

    

    将向量索引背后的嵌入模型进行升级，需要对语料库进行重新编码。最近的研究通过适配器或窗口重排序来复用旧索引。在段落和代码语料库上的十六对模型，以及Natural Questions（NQ）数据集上的十一对模型的实验中，我们发现了三个结果。第一，复用永远无法避免重新编码：如果迁移必须完成，每种策略最终都会对每个文档进行编码；复用只是改变了这项工作何时被支付以及期间所提供的检索质量。第二，适配器应该与旧模型进行比较，而不仅仅是与新模型比较。在五种适配器变体中，只有同系列的LLM桥接方法优于旧模型，而且优势微乎其微。第三，窗口重排序的成本可以仅凭旧索引来估算；在全部8,841,823个段落上进行的预注册重放实验证实了这一预测，在所有三个测试的模型对上均得到验证。工作负载偏斜同样重要：在36次额外的预注册Zipf分布重放中，规划的成本模型在32次中与实际情况吻合。

    arXiv:2611.20702v1 Announce Type: cross  Abstract: Upgrading the embedding model behind a vector index requires re-encoding the corpus. Recent work reuses the old index through adapters or window re-ranking. Across sixteen model pairs on passage and code corpora, plus eleven on Natural Questions (NQ), we find three results. First, reuse never avoids re-encoding: if migration must complete, every strategy eventually encodes every document; reuse only changes when that work is paid and the quality served meanwhile. Second, adapters should be compared with the old model, not only the new one. Of five adapter variants, only same-family LLM bridges beat the old model, and only by a negligible margin. Third, window re-ranking cost can be priced from the old index alone; pre-registered replays on all 8,841,823 passages confirmed the prediction on all three tested pairs. Workload skew also matters: across 36 further pre-registered Zipf replays, the planned family matched on 32. These findings 
    
[^66]: IndustrialKnowEval：一个用于评估大语言模型多层次工业知识的基准

    IndustrialKnowEval: A Benchmark for Evaluating Multi-Level Industrial Knowledge in LLMs

    [https://arxiv.org/abs/2611.20695](https://arxiv.org/abs/2611.20695)

    该论文提出了覆盖七个工业领域、21 种任务类型，并按记忆、理解、推理和工程应用四个认知层次组织的工业知识问答基准 IndustrialKnowEval（IndKnow），同时提出 ICCG 指标来量化大语言模型的工业认知能力差距。

    

    大语言模型（LLM）在工业任务中的应用探索日益增多，但很少有基准能够评估其在不同类型认知需求下对工业知识的掌握程度。我们提出了 IndustrialKnowEval（IndKnow），一个面向工业知识与工程问答的基准。该基准包含 43,282 个样本，覆盖包括航空航天、电力和冶金在内的七个工业领域的 21 种任务类型，并按四个认知需求层次进行组织：记忆、理解、推理和工程应用。来自企业的领域专家和经验丰富的研究人员参与了数据集构建与质量控制的全过程。我们在 18 个代表性大语言模型上开展了一系列实验，领域专家深度参与了评估工作。基于 IndKnow，我们提出了工业认知能力差距（ICCG）指标，用于刻画基础模型与……（原文摘要在此处截断）

    arXiv:2611.20695v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly explored for industrial tasks, yet few benchmarks evaluate their mastery of industrial knowledge across different types of cognitive demand. We introduce IndustrialKnowEval (IndKnow), a benchmark for industrial knowledge and engineering question answering. It contains 43,282 samples covering 21 task types across seven industrial sectors, including aerospace, electric power, and metallurgy, organized into four levels of cognitive demand: memory, comprehension, reasoning, and engineering application. Enterprise-based domain specialists and experienced researchers participate in all stages of dataset construction and quality control. We conduct a series of experiments on 18 representative LLMs, with domain experts deeply involved in the evaluation. Building upon IndKnow, we propose the Industrial Cognitive Capability Gap (ICCG) metric to characterize the performance differential between founda
    
[^67]: 前沿快，回退错：扩散语言模型中阈值解码的提交级分析

    Fast at the Frontier, Wrong at the Fallback: A Commit-Level Analysis of Threshold Decoding in Diffusion Language Models

    [https://arxiv.org/abs/2611.20693](https://arxiv.org/abs/2611.20693)

    该论文通过对四个扩散语言模型在数学、代码、翻译和摘要任务上的提交级追踪分析发现，准确率损失主要来自在文本前方提前提交 token，而仅在前沿（最左掩码位置）提交既不损失速度又能避免大部分错误。

    

    扩散语言模型（DLM）可以在每次前向传播中生成多个 token。开源权重的 DLM 会以任意顺序提交所有置信度超过阈值的被掩码位置；如果没有位置达到阈值，则启用回退机制提交置信度最高的一个位置，且已提交的 token 永远不会被修改。宽窗口使解码器能够远远超前于已有文本进行提交，这本应加快解码速度，但往往会损害准确率。目前尚不清楚模型究竟在哪些位置有信心，以及哪些提交导致了错误。我们在数学、代码、翻译和摘要任务上追踪了四个现成 DLM 的每一次提交，并通过只改变其中一步来重跑失败答案。研究发现：大多数 token 只有在前沿（即最左侧的被掩码位置）附近才会变得有信心，因此仅在前沿提交（每步仍可提交多个 token）与在任意位置提交一样快。准确率的损失来自于在文本左侧尚未生成的内容之前提前提交 token，而每步只提交一个 token 也无法减少这种损失。最坏的情况是过早提交序列结束符。

    arXiv:2611.20693v1 Announce Type: new  Abstract: Diffusion language models (DLMs) can write several tokens per forward pass. Open-weight DLMs commit, in any order, every masked position whose confidence passes a threshold. If none does, a fallback commits the most confident one, and the commits are never revised. Wide windows let the decoder commit far ahead of the text, which should speed up decoding but often costs accuracy. It is unclear where the model is confident and which commits cause errors. We trace every commit of four off-the-shelf DLMs on math, code, translation and summarization, and rerun failing answers with one step changed. Most tokens become confident only near the frontier, the leftmost masked position, so committing only there, still several tokens per step, is about as fast as committing anywhere. The accuracy loss comes from committing tokens before the text to their left, and committing one token per step does not reduce it. The worst case is an early end-of-seq
    
[^68]: LEAF-OPD：通过策略内蒸馏从特权上下文中学习遗忘

    LEAF-OPD: Learning to Forget from Privileged Contexts via On-Policy Distillation

    [https://arxiv.org/abs/2611.20674](https://arxiv.org/abs/2611.20674)

    提出LEAF-OPD遗忘框架，利用冻结教师模型在学生自生成的回答上进行token级策略内蒸馏——遗忘数据上借助特权上下文引导遗忘、保留数据上无需特权上下文以维持实用性——从而教会模型在遇到应被遗忘的知识时如何恰当回应。

    

    大语言模型遗忘旨在从模型中擦除目标知识，同时保持模型的实用性。然而，许多现有方法只是抑制目标知识，而没有明确教会模型如何回应与这些知识相关的查询。我们提出了LEAF-OPD（通过策略内蒸馏从特权上下文中学习遗忘），其中冻结的教师模型在学生生成的回答上提供token级别的监督。在遗忘数据上，教师利用特权上下文（即不透露遗忘目标相关知识的指令）来引导遗忘；在保留数据上，教师在不使用特权上下文的情况下提供监督，以维持模型的实用性。学生在两种情况下均不使用特权上下文，从而学习到在遇到本应被遗忘的知识时如何作出回应。一个关键挑战在于，教师往往只在学生轨迹的前几个token上提供有效监督。我们引入了Supervisio（摘要此处截断）……

    arXiv:2611.20674v1 Announce Type: new  Abstract: LLM unlearning aims to erase targeted knowledge from large language models while preserving their utility. However, many existing methods suppress target knowledge without explicitly teaching the model how to respond to queries about it. We propose LEAF-OPD (LEArning to Forget from Privileged Contexts via On-Policy Distillation), in which a frozen teacher provides token-level supervision on student-generated responses. On forget data, the teacher uses privileged contexts (instructions not to reveal knowledge about forget targets) to guide forgetting. On retain data, the teacher provides supervision without privileged context to preserve model utility. The student operates without privileged context in both cases, learning how to respond when encountering knowledge that should be forgotten. A key challenge is that the teacher often provides effective supervision only at the first few tokens of student trajectories. We introduce Supervisio
    
[^69]: MetaAct：面向具身语言智能体的自进化可执行动作约束

    MetaAct: Self-Evolving Executable Action Constraints for Embodied Language Agents

    [https://arxiv.org/abs/2611.20655](https://arxiv.org/abs/2611.20655)

    MetaAct 通过从执行反馈中合成可执行动作规则（GEAR）并编译为确定性守卫，将语言智能体的自进化从文本引导升级为可执行动作约束，在多个具身环境基准上大幅降低动作违规率并显著提升任务性能。

    

    自进化语言智能体已成为一种有前景的范式，能够通过经验持续改进智能体行为。然而，现有的大多数方法只进化文本产物，这虽然可以影响动作偏好，却无法在具有离散且往往不可逆动作的具身环境中可靠地保证动作的可执行合法性。我们提出 MetaAct，将自我进化从文本引导转变为可执行的动作约束，通过执行反馈合成 GEAR（接地可执行动作规则，Grounded Executable Action Rules）。每条规则以带类型的规范形式表示，编译为确定性的守卫（guard），并通过轨迹回放进行离线验证，无需额外的模型调用或环境交互。在四个基准测试中，MetaAct 提升了任务性能；在 ScienceWorld 和 WebShop 上，违规率分别从 70% 降至 0%、从 40% 降至 1%，得分分别提升 +10.1 和 +8.0。

    arXiv:2611.20655v1 Announce Type: new  Abstract: Self-evolving language agents have emerged as a promising paradigm for continuously improving agent behavior through experience. However, most existing approaches evolve textual artifacts, which can influence action preferences but cannot reliably enforce action admissibility in embodied environments with discrete and often irreversible actions. We propose MetaAct, which shifts self-evolution from textual guidance to executable action constraints by synthesizing GEAR (Grounded Executable Action Rules) from execution feedback. Each rule is represented as a typed specification, compiled into a deterministic guard, and validated offline through trajectory replay without additional model calls or environment interactions. Across four benchmarks, MetaAct improves task performance; on ScienceWorld and WebShop, violations drop from 70% to 0% and 40% to 1%, with score gains of +10.1 and +8.0. Our code is available at https://github.com/BaiCubed/
    
[^70]: RAMP：通过答案读取协议重新思考多模态评估

    RAMP: Rethinking Multimodal Evaluation through Answer Readout Protocols

    [https://arxiv.org/abs/2611.20653](https://arxiv.org/abs/2611.20653)

    提出RAMP系统性评估框架，揭示了多模态大语言模型评测中答案读取协议的选择会强烈影响结果——不同读取方式可使模型准确率变化超过20个百分点，且即使净准确率差异很小，大量个体预测的正确性仍会发生改变。

    

    多模态大语言模型（MLLM）通常通过将模型输出转换为离散答案来进行评估。这种转换协议决定了哪些分数成为预测结果，并可能对报告的结果产生强烈影响。我们提出了RAMP（读取感知多模态协议，Readout Aware Multimodal Protocol），这是一个系统性框架，在固定的模型检查点和配对的视觉问题条件下，比较直接的yes/no token logits、表面形式分数，以及带类型标注或反转的A/B选项接口。在500个POPE对抗性问题上，带类型标注与裸读取方式相比，使SmolVLM-256M的准确率变化了-20.2个百分点，TinyLLaVA-1.1B变化了-16.4个百分点，Qwen2.5-VL-3B变化了+1.8个百分点。配对转换分析显示，即使净准确率差异很小，个体预测也会发生大量变化：对于SmolVLM-500M，104个二元ChartQA题目中有79个的正确性发生了改变，而净差异仅为+1.0个百分点。我们进一步分析了提示措辞、候选形式、选项位置、跨任务效应等因素（摘要原文在此处被截断）。

    arXiv:2611.20653v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) are often evaluated by converting model outputs into discrete answers. The conversion protocol determines which scores become predictions and can strongly influence the reported results. We introduce RAMP (Readout Aware Multimodal Protocol), a systematic framework that compares direct yes/no token logits, surface-form scores, and typed or reversed A/B interfaces under fixed model checkpoints and paired visual questions. On 500 POPE adversarial questions, typed versus bare readout changes accuracy by $-20.2$ percentage points for SmolVLM-256M, $-16.4$ for TinyLLaVA-1.1B, and $+1.8$ for Qwen2.5-VL-3B. Paired transitions reveal extensive prediction changes even when the net accuracy difference is small: 79 of 104 binary ChartQA items change correctness for SmolVLM-500M, despite a $+1.0$-point net difference. We further analyze prompt wording, candidate forms, option position, cross-task effects, an
    
[^71]: 基于双层数据选择的保安全医疗大语言模型微调

    Safety-Preserving Medical LLM Fine-Tuning via Bilevel Data Selection

    [https://arxiv.org/abs/2611.20643](https://arxiv.org/abs/2611.20643)

    提出了S-BDS可扩展双层数据选择方法，通过双层优化从未经筛查的医疗指令数据中筛选出既提升医疗能力又保持安全对齐的训练样本，实现面向患者的保安全医疗大模型微调。

    

    随着通用大语言模型（LLM）越来越多地被适配用于面向患者的医疗应用，如何在提升医疗能力的同时控制面向患者的风险仍然是一个关键挑战。大规模医疗指令数据对于通过监督微调（SFT）获取领域知识至关重要，但不安全的响应可能会破坏已有的安全对齐。为解决这一问题，我们提出了用于保安全医疗SFT的可扩展双层数据选择方法（S-BDS）。通过控制选择权重的集中度，S-BDS能够在大型候选池上实现可靠的样本排序，并识别出既能提升医疗能力又能控制面向患者风险的训练样本。我们将临床医生验证的安全数据与从未经筛查的医疗指令数据中选出的数据相结合进行SFT，并分别在HealthBench和PatientSafetyBench上联合评估医疗能力和面向患者的安全性。

    arXiv:2611.20643v1 Announce Type: cross  Abstract: As general large language models (LLMs) are increasingly adapted for patient-facing medical applications, improving medical capability while controlling patient-facing risks remains a critical challenge. Large-scale medical instruction data are essential for acquiring domain knowledge through supervised fine-tuning (SFT), but unsafe responses can undermine existing safety alignment. To address this problem, we propose Scalable Bilevel Data Selection (S-BDS) for safety-preserving medical SFT. By controlling selection-weight concentration, S-BDS enables reliable sample ranking over large candidate pools and identifies training examples that improve medical capability while controlling patient-facing risks. We combine clinician-validated safe data and selected data with unscreened medical instruction data for SFT, and jointly evaluate medical capability and patient-facing safety on HealthBench and PatientSafetyBench, respectively. Experim
    
[^72]: 相同论文，不同裁决：追踪大语言模型同行评审中的版本漂移

    Same Papers, Different Verdicts: Tracing Version Drift in LLM Peer Review

    [https://arxiv.org/abs/2611.20608](https://arxiv.org/abs/2611.20608)

    该研究通过在ICLR 2027投稿评审期间对二十个前沿大语言模型进行的预注册审计发现，模型版本更新会导致同一论文的评审结论剧烈漂移——接收判定比例下降高达52至79个百分点且无任何论文评分上升，表明大语言模型评审结果并非对论文的稳定度量。

    

    各学术会议和基准测试将大语言模型的评审分数视为对论文质量的度量，而模型提供商却在数月内不断更换其模型。我们在ICLR 2027投稿论文处于评审期间，对来自OpenAI、Google和Anthropic的二十个前沿模型进行了审计，所有测试均在评审决策公布前预先注册。在Claude Opus和Gemini的早期版本与后续版本之间，被判定为接收的论文比例在183篇和60篇随机抽取的投稿上分别下降了52和53个百分点，在另外六组配对比较中下降了43至79个百分点，且没有任何论文的判定结果上升。即使在提示词中明确声明30%的录取率，两个Opus版本之间仍存在79个百分点的差距。当将去除分数的评审意见在版本之间交换时，仅靠后期版本的评审意见就使Opus的接收比例降低35个百分点，Gemini降低51个百分点；对于Opus，在第三方评分器下降幅为35个百分点，在保留每条技术要点并重写后降幅为24个百分点……

    arXiv:2611.20608v1 Announce Type: cross  Abstract: Venues and benchmarks read LLM review scores as measurements of a paper, while providers replace their models within months. We audit twenty frontier models from OpenAI, Google and Anthropic on ICLR 2027 submissions while they were under review, with tests registered before the decisions. Between an earlier and a later version of Claude Opus and of Gemini, the share of papers placed on the accept side falls by 52 and 53 percentage points on 183 and 60 randomly drawn submissions, and by 43 to 79 points in six further paired comparisons, with no paper moving up in any of them. Stating a 30% acceptance rate in the prompt leaves a gap of 79 points between the two Opus versions. When score-stripped critiques are swapped between versions, the later critiques alone lower the accept share by 35 points for Opus and 51 for Gemini; for Opus the drop is 35 points under a third-party scorer and 24 after a rewrite that keeps each technical point and
    
[^73]: 何为必要？审计向大语言模型合成受访者披露的人口统计信息

    Necessary for What? Auditing Demographic Disclosure to LLM Synthetic Respondents

    [https://arxiv.org/abs/2611.20606](https://arxiv.org/abs/2611.20606)

    该研究通过多模型、多调查的审计实验发现，将面板受访者的详细人口统计档案放入提示中对大多数合成调查目的几乎没有帮助，质疑了“更丰富的个人档案造就更忠实合成受访者”这一普遍前提。

    

    大型语言模型越来越多地代表真实面板受访者回答调查问题，而每位受访者的已记录人口统计属性（如年龄、教育程度、种族等）通常被置于提示中，其前提是更丰富的个人档案能造就更忠实的合成受访者。虽然数据最小化原则要求将个人数据限制在特定目的所需的范围内，但合成调查同时服务于多种目的，包括预测个人答案、估计观点分布以及比较不同群体。我们探究这些目的中哪些真正需要受访者的档案记录，以及改进后的估计是否支持关于群体的可靠结论。我们在两项美国调查中，使用十一个开源权重模型和三个托管模型，改变模型接收谁的档案、档案来自何处以及包含多少字段，并将每个模型与基于相同信息拟合的统计模型进行比较。我们做出了四项贡献。首先，我们表明该记录几乎没有增加……（摘要原文在此处截断）

    arXiv:2611.20606v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly answer surveys on behalf of real panelists, and each panelist's recorded demographic attributes (e.g., age, education, ethnicity) are often placed in the prompt on the premise that richer profiles make more faithful synthetic respondents. While data minimization limits personal data to what a purpose requires, a synthetic survey serves several purposes at once, from predicting individual answers to estimating how opinions are distributed and comparing groups. We ask which of these purposes require the respondent's record, and whether improved estimates support sound conclusions about groups. We vary whose profile a model receives, where it comes from, and how many fields it contains, in two U.S. surveys with eleven open-weight and three hosted models, and compare each model with a statistical model fit to the same information. We make four contributions. First, we show that the record adds litt
    
[^74]: 基于不变量感知的运行时框架强制机制：保护大语言模型智能体免受静默 API 失败的影响

    Invariant-Aware Harness Enforcement to Protect LLM Agents from Silent API Failures

    [https://arxiv.org/abs/2611.20595](https://arxiv.org/abs/2611.20595)

    SilentGuard 将故障检测从模型本身转移到运行工具调用的框架层，通过检验 API 返回结果是否满足正确性不变量并拦截违规值，将 LLM 智能体因静默 API 失败导致的错误回答率从 74.0% 降至 1.7%，且不影响无故障任务的性能。

    

    大语言模型智能体通过调用 API 并信任其返回结果来回答问题。然而 API 可能会静默失败：它会忽略请求的一部分（例如名称错误的过滤器），却返回 HTTP 200 状态码以及一个看似合理但实际错误的结果。当前的解决方案（如检查清单、审阅模型和 LLM 评判器）都要求模型自己去发现问题，但模型往往即便察觉到问题仍然照常作答。我们提出了 SilentGuard，它将决策权从模型转移到运行工具调用的执行框架中。该框架针对任何正确的 API 都必须满足的不变量对每个返回结果进行检验，并拦截违反不变量的数值。我们在 15 个公开研究 API 中的 11 个里发现了静默失败，其中计数结果被夸大最多达 691 倍。现有解决方案下仍有超过 70% 的答案是错误的，且智能体报告的计数中有 77.4% 被评判器标记为有问题。在 7 个开源模型上，SilentGuard 将其在目标故障下的错误回答率从 74.0% 降至 1.7%，且在无故障任务上没有任何性能损失；在三个前沿模型上，错误率从 70.2% 降至 0.5%。

    arXiv:2611.20595v1 Announce Type: new  Abstract: LLM agents answer questions by calling APIs and trusting the results. An API can fail silently. It ignores part of a request, such as a misnamed filter, yet returns HTTP 200 and a plausible but wrong result. Current fixes, such as checklists, reviewer models and LLM judges, ask the model to catch it. But models often see the problem and answer anyway. We present SilentGuard, which moves the decision from the model to the harness running tool calls. It tests each result against invariants that any correct API must satisfy and withholds values that break one. We find silent failures in 11 of 15 public research APIs, with counts inflated up to 691 times. Current fixes leave over 70% of answers wrong, and agents report 77.4% of counts a judge flags. Across 7 open models, SilentGuard cuts wrong answers on its target faults from 74.0% to 1.7%, with no loss on fault-free tasks. On three frontier models they fall from 70.2% to 0.5%. A warning fr
    
[^75]: 语言模型遇上低轨星座卫星振动测试：基于少样本扫频的响应预测校准

    Language Models Meet Vibration Testing for LEO Constellation Satellites with Calibrated Response Prediction from Few-show Sweeps

    [https://arxiv.org/abs/2611.20579](https://arxiv.org/abs/2611.20579)

    该论文提出为LLM驱动的工程分析设置确定性验证门控（涵盖单位、物理一致性、引用核验、独立模型间一致性、测量数据一致性与模型适用性检查），并将其实例化于低轨巨型星座卫星的正弦振动验收测试，以防止LLM生成的看似合理却错误的数值悄然进入工程决策。

    

    大型语言模型（LLM）正日益推动端到端的工程分析，将文本记录转化为数值程序的输入，但一个看似合理却错误的数值可能会悄无声息地进入决策过程。我们提出了面向LLM驱动工程分析的确定性验证门控，这是一种通用范式：由LLM生成的类型化配置只有在通过确定性检查——包括单位、物理一致性、逐字引用核验、独立模型之间的一致性、与测量导出数据的一致性以及已训练模型的适用性——之后才会被采用。这些门控从两个方向对抗幻觉问题：引用检查可以捕获编造的内容，而模型间一致性检查则可以捕获引用检查所遗漏的疏漏性错误。我们将该范式实例化于低轨巨型星座卫星的正弦振动验收测试中，由DeepSeek V4.1 Flash和Qwen-plus读取中文测试记录并……（摘要在此处截断）

    arXiv:2611.20579v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly drive engineering analysis end to end, turning textual records into the inputs of numerical programs, but a plausible and wrong value can pass silently into a decision. We propose deterministic validation gates for LLM-driven engineering analysis, a general paradigm in which a typed configuration produced by LLMs is consumed only after it passes deterministic checks on units, physical consistency, verbatim quotes, agreement between independent models, consistency with measurement exports and the applicability of trained models. The gates target hallucinations from two directions, since quote checking catches fabricated content while agreement between models catches errors of omission that quote checking misses. We instantiate the paradigm on sine vibration acceptance testing for low Earth orbit mega-constellation satellites, where DeepSeek V4.1 Flash and Qwen-plus read Chinese test records and 
    
[^76]: 权重空间编程改变了大语言模型在计算生物学中使用Python的方式

    Weight-Space Programming Changes How an LLM Uses Python for Computational Biology

    [https://arxiv.org/abs/2611.20566](https://arxiv.org/abs/2611.20566)

    利用权重空间编程让Claude Opus 5.5直接编辑Qwen3.5-0.8B的权重（无需梯度训练），显著提升了小模型在计算生物学任务中用Python解题的正确率（严格正确答案从44增至77），并大幅减少了冗余的Python调用。

    

    权重空间编程使一个语言模型能够在不进行基于梯度的训练的情况下编辑另一个模型的权重。我们使用Claude Opus 5.5来编辑Qwen3.5-0.8B，并研究了这些权重编辑如何改变该模型在计算生物学问题上使用Python的方式。我们将四个初始编辑后的检查点与原始模型在50个测试问题上进行了三种测试种子的对比，共产生600次配对评估。严格正确的答案从44个增加到77个，技术性正确的答案（包括因格式问题被拒但最终值正确的答案）从64个增加到100个。重复的Python调用次数从1,834次减少到1,044次。在64对仅有编辑后模型给出技术性正确答案的案例中，原始模型在36对中其实已经计算出了正确的数值，只是没有提交技术性正确的最终答案；而编辑后的模型在其中13对中纠正了计算。通过在同一50个问题上的另外六个检查点，技术性……

    arXiv:2611.20566v1 Announce Type: cross  Abstract: Weight-space programming allows a language model to edit another model's weights without gradient-based training. We used Claude Opus 5.5 to edit Qwen3.5-0.8B and investigated how the edits changed its use of Python on computational biology problems. We compared four initial edited checkpoints with the original model on 50 test problems across three test seeds, yielding 600 paired evaluations. Strictly correct answers increased from 44 to 77, and technically correct answers, which included correct final values rejected for formatting, increased from 64 to 100. Repeated Python calls decreased from 1,834 to 1,044. Among the 64 pairs in which only the edited response was technically correct, the original model had already computed the correct value in 36 but did not submit a technically correct final answer. The edited model corrected the computation in 13 of the 64 pairs. With six additional checkpoints on the same 50 problems, technical
    
[^77]: ReconPotBench：在多轮Linux侦察中评估LLM蜜罐

    ReconPotBench: Evaluating LLM Honeypots in Multi-Round Linux Reconnaissance

    [https://arxiv.org/abs/2611.20528](https://arxiv.org/abs/2611.20528)

    该论文提出了首个针对Linux侦察蜜罐的多轮评估基准ReconPotBench，通过1,032个三轮交互测试证明多轮评估能暴露单轮测试所掩盖的问题，揭示LLM蜜罐在跨轮状态一致性上的不足。

    

    大型语言模型（LLM）越来越多地被用于构建交互式蜜罐，然而现有的评估主要针对孤立的命令输出，无法判断当后续命令依赖先前结果时，系统是否仍然可信。我们提出了ReconPotBench，这是一个针对Linux侦察蜜罐的多轮基准测试。它包含1,032个跨八个侦察类别的三轮侦察交互，参考输出来自真实Linux环境。在统一的测试协议下，我们从三个维度比较了Cowrie、开源LLM和闭源LLM：轮次级真实性、跨轮上下文一致性以及面向黑客的侦察价值。结果表明，多轮评估能够暴露单轮测试所隐藏的失败，因为真实性会随交互深度的增加而下降，且开源和闭源LLM都难以在多轮交互中维持连贯的状态。

    arXiv:2611.20528v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to build interactive honeypots, yet existing evaluations mainly assess isolated command outputs and cannot determine whether a system remains believable when later commands depend on earlier results. We present ReconPotBench, a multi-round benchmark for Linux reconnaissance honeypots. It contains 1{,}032 three-round reconnaissance interactions across eight reconnaissance categories, with reference outputs collected from a real Linux environment. Under a unified testing protocol, we compare Cowrie, open-source LLMs, and closed-source LLMs along three dimensions: round-level authenticity, cross-round contextual consistency, and hacker-oriented reconnaissance value. Results show that multi-round evaluation exposes failures hidden by single-round testing, as authenticity declines with interaction depth and both open-source and closed-source LLMs struggle to maintain coherent state across r
    
[^78]: GaugeRoPE：与RoPE兼容的可执行潜在注意力

    GaugeRoPE: RoPE-Compatible Executable Latent Attention

    [https://arxiv.org/abs/2611.20524](https://arxiv.org/abs/2611.20524)

    提出无需训练的可执行潜在KV表示GaugeRoPE，其RoPE兼容载体可在压缩键上直接执行预训练旋转注意力而无需重构回原始维度，从而消除了解码路径中随序列长度扩展的开销。

    

    潜在KV缓存压缩通过将历史键与值映射为紧凑表示来减少内存占用。然而，通用的压缩键状态在应用RoPE并计算注意力得分之前，需要先重构回原始维度，这给解码路径带来了随序列长度扩展的开销。我们刻画了潜在KV表示在何种条件下可以直接充当预训练RoPE的执行状态。对于基于RoPE的Transformer而言，旋转键路径的精确静态线性执行要求从潜在空间到原始空间的映射通过一个交织关系来保持预训练的RoPE作用，这将精确混合限制在共享同一RoPE频率的坐标之间。基于这一刻画，我们构建了GaugeRoPE——一种无需训练的可执行潜在KV表示。它将一个能够保持旋转作用的RoPE兼容载体与一个近似的、与位置无关的残差键值（RKV）修正相结合，以补……

    arXiv:2611.20524v1 Announce Type: cross  Abstract: Latent KV cache compression reduces memory footprint by mapping historical keys and values into compact representations. However, generic compressed key states require reconstruction to native dimension before applying RoPE and computing attention scores, adding sequence-scaling overhead to the decode path. We characterize when a latent KV representation can serve directly as the execution state of pretrained RoPE. For RoPE-based transformers, exact static linear execution of the rotary key path requires the latent-to-native map to preserve the pretrained RoPE action through an intertwining relation, restricting exact mixing to coordinates that share the same RoPE frequency. Guided by this characterization, we build GaugeRoPE, a training-free executable latent KV representation. It combines a RoPE-compatible carrier that preserves the rotary action with an approximate position-independent residual key--value (RKV) correction for comple
    
[^79]: 大语言模型中的拒答方向：跨查询歧义、知识局限与安全对齐的拒答机制可解释性研究

    Abstention Direction in Large Language Models: Mechanistic Interpretability of Abstention Across Query Ambiguity, Knowledge Limits, and Safety Alignment

    [https://arxiv.org/abs/2611.20441](https://arxiv.org/abs/2611.20441)

    该论文将原本仅针对安全对齐的拒绝方向框架扩展至查询歧义、知识局限与价值对齐三类拒答成因，通过反事实配对样本计算统一的拒答方向，并证明通过激活引导干预中间层可双向调控大语言模型的拒答行为。

    

    拒答是指大语言模型（LLM）在面临不确定性或安全风险时拒绝给出回应，这对于构建可信赖的模型至关重要。拒答的成因可以从三个视角进行分析：查询本身的歧义性（Q）、模型知识的局限性（MK）以及与人类价值观（HV）的不一致。然而，以往对内部表征的分析仅集中于HV视角，即利用有害与无害查询计算出的拒绝方向，尚无研究对这三个视角进行联合分析。因此，本文将拒绝方向框架扩展至全部三个视角。具体而言，针对每个视角，我们构建了共享同一模板、仅在引发拒答的因素上存在差异的反事实配对样本，并通过归一化两种极性样本的平均激活差异来计算拒答方向。激活引导实验表明，对中间层进行干预可以在所有三类成因下增加或减少模型的拒答行为。

    arXiv:2611.20441v1 Announce Type: new  Abstract: Abstention, the withholding of a response by a large language model (LLM) under uncertainty or safety risks, is essential for trustworthy models. The causes of abstention fall into three perspectives: ambiguity in the Query (Q), limits of Model Knowledge (MK), and misalignment with Human Values (HV). However, prior analyses of internal representations have focused on HV, using a refusal direction computed from harmful and harmless queries. No study has analyzed the three perspectives jointly. We therefore extend the refusal direction framework to all three perspectives. Specifically, for each perspective, we construct counterfactual pairs that share a template and differ only in the abstention-inducing factor, and compute an abstention direction by normalizing the difference in mean activations between the two polarities. Activation steering experiments show that intervening in middle layers increases or decreases abstention across all t
    
[^80]: TrainExplorer：通过规划与解耦探索实现自主的大语言模型后训练

    TrainExplorer: Autonomous LLM Post-Training via Plan-and-Decouple Exploration

    [https://arxiv.org/abs/2611.20430](https://arxiv.org/abs/2611.20430)

    提出多智能体框架 TrainExplorer，通过借鉴科学文献的策略规划与单组件干预的执行解耦机制，实现自主的大语言模型后训练策略探索，在 PostTrainBench 上取得 39.02 的加权综合得分并超越 Claude Code。

    

    基于大语言模型的智能体能够自动化模型后训练过程，加速迈向递归自我改进的进程。尽管前景广阔，现有方法仍缺乏充分探索，并面临两大主要挑战：（1）策略锁定，即难以探索初始策略之外的方案；（2）耦合执行，即难以辨析单项修改的具体效果。为解决这些挑战，我们提出了 TrainExplorer——一个用于自主后训练的多智能体框架，它融合了策略规划与执行解耦，以实现更广泛的策略探索。首先，策略规划借鉴科学文献来构建训练策略的候选搜索空间；随后，解耦执行支持单组件干预与效果归因，从而引导策略探索。在 PostTrainBench 上的大量实验表明，TrainExplorer 取得了 39.02 的加权综合得分，超越了 Claude Code……

    arXiv:2611.20430v1 Announce Type: cross  Abstract: LLM-based agents can automate model post-training, accelerating progress toward recursive self-improvement. Despite their promise, existing approaches remain under-explored and face two main challenges: (1) strategy lock-in that limits exploration beyond the initial strategy; and (2) coupled execution that obscures the effects of individual modifications. To address these challanges, we propose TrainExplorer, a multi-agent framework for autonomous post-training, that incorporates strategy planning and execution decoupling, to achieve broaden strategy exploration. First, strategy planning draws on scientific literature to construct a candidate search space of training strategies. Then, decoupled execution enables single-component interventions and attribution to guide strategy exploration. Extensive experiments on PostTrainBench demonstrate that TrainExplorer achieves a weighted aggregate score of 39.02, outperforming Claude Code using 
    
[^81]: 当最便宜可行路由器失效时：成本感知LLM路由中的部署承诺审计

    When the Cheapest Feasible Router Fails: Auditing Deployment Commitments in Cost-Aware LLM Routing

    [https://arxiv.org/abs/2611.20417](https://arxiv.org/abs/2611.20417)

    提出MARC框架，通过审计证据与服务成本容差对候选路由策略进行资格认证，将成本感知LLM路由部署后的条件契约失败率从32.22%大幅降至0.73%。

    

    成本感知的LLM路由通常选择满足经验风险和覆盖约束的最便宜策略。然而，即使候选策略是独立构建的，通过审计的路由器在新请求上仍可能违反其固定的服务契约。我们在构建、审计和测试样本互不相交的条件下研究这一部署承诺决策问题。在19个RouterBench任务族上的实验表明，最便宜的通过审计可行的路由器在91.82%的机会中被部署，但在32.22%的部署中违反了留出契约。我们提出了菜单感知路由承诺（MARC），它利用审计证据和服务成本容差对策略进行资格认证，以保留表面上最优的方案、部署有依据的替代方案或推迟部署。其实用的逐策略变体（MARC-P）在85.79%的部署率下将观察到的条件失败率降至0.73%，但服务成本更高；固定边际和形式化风险控制程序则提供了具有竞争力的替代方案。

    arXiv:2611.20417v1 Announce Type: cross  Abstract: Cost-aware LLM routing often selects the cheapest policy meeting empirical risk and coverage constraints. Yet a router that passes an audit can violate its fixed service contract on new requests, even when candidate policies are constructed independently. We study this deployment-commitment decision with disjoint construction, audit, and test samples. Across 19 RouterBench task families, the cheapest audit-feasible router deploys in 91.82% of opportunities but violates the held-out contract in 32.22% of deployments. We introduce Menu-Aware Routing Commitment (MARC), which qualifies policies using audit evidence and a serving-cost tolerance to retain the apparent winner, deploy a supported alternative, or defer. Its practical policywise variant (MARC-P) reduces observed conditional failure to 0.73% at 85.79% deployment, with higher serving cost; fixed-margin and formal risk-control procedures provide competitive alternatives. Nested aud
    
[^82]: 通过增量训练将N元语法嵌入增强到预训练语言模型中

    Augmenting Pretrained Language Models with N-gram Embeddings through Incremental Training

    [https://arxiv.org/abs/2611.20415](https://arxiv.org/abs/2611.20415)

    本文提出通过增量训练将N元语法嵌入注入到已有的预训练语言模型中，突破联合训练方式的场景限制，并发现其最优配置（嵌入学习率、词表大小、注入位置）与预训练场景下的最佳配置并不一致。

    

    作为一种新的大语言模型扩展技术，N元语法嵌入在几乎不增加推理成本的情况下提升了模型的基础能力。主流研究通过将N元语法嵌入与模型骨干进行联合训练来实现这一目标，这严重限制了N元语法嵌入可适用的场景。因此，我们提出通过增量训练将N元语法嵌入训练到已有模型上，以扩展该技术能够发挥价值的范围。具体而言，我们探索了增量训练下N元语法嵌入的若干设置——嵌入的学习率、词表大小和注入位置——并在最佳配置下将N元语法嵌入训练到现有模型上。实验表明，增量训练下的最佳配置与预训练下的最佳配置并不完全一致；此外，通过增量训练学到的知识主要被写入到……（原文摘要至此截断）

    arXiv:2611.20415v1 Announce Type: new  Abstract: As a new scaling technique for large language models, the n-gram embedding improves the base ability of a model while adding almost no inference cost. Mainstream research realizes it through joint training of the n-gram embedding with the backbone, which severely restricts the settings in which an n-gram embedding can be used. We therefore propose training an n-gram embedding onto an existing model through incremental training, to extend where the technique can deliver value. Specifically, we explore several settings of the n-gram embedding under incremental training---the learning rate of the embedding, the vocabulary size, and the injection position---and train an n-gram embedding onto the existing model under the best configuration. Experiments show that the best configuration under incremental training does not fully coincide with that under pretraining; moreover, the incrementally trained knowledge is written predominantly into the 
    
[^83]: 拒答语域是因果可调控的：在 Gemma-2-9B-IT 中将拒答方向分解为与类别对齐的成分

    Refusal Register is Causally Steerable: Decomposing the Refusal Direction into Category-Aligned Components in Gemma-2-9B-IT

    [https://arxiv.org/abs/2611.20405](https://arxiv.org/abs/2611.20405)

    该研究在 Gemma-2-9B-IT 中发现拒答方向不仅编码“是否拒答”，还编码“如何拒答”的语域信息，并将其分解为与语义类别（如有害内容、自我指涉）对齐的线性成分，证明向残差流注入这些方向可因果性地调控模型拒答语言的风格。

    

    先前的研究表明，指令微调语言模型中的拒答行为可以通过干预单个残差流方向来消除（Arditi 等人，2024），而较新的研究发现了多个介导拒答的方向和概念锥（Wollschläger 等人，2025）。我们提出了一个不同的问题：拒答相关方向是否携带关于拒答语言“语域”的信息——即模型如何拒绝——而不仅仅是它是否拒绝？在 Gemma-2-9B-IT 中，我们在第 20 层识别出两个由 SVM 导出的线性方向（B1 对应有害内容与中性内容，B2 对应自我指涉与中性内容），两者余弦相似度 cos(B1, B2) = 0.252（约 75 度夹角，显著非平行）。将有害内容与中性的均值差方向（与 B1 的余弦相似度为 0.68）以 alpha = 2.0 的强度添加到自我指涉提示的残差流中，会使内容安全类词汇的命中数从每 100 个 token 2.55 次提升至 10.37 次，而幅度匹配的随机方向仅产生 1.13-2.16 次……

    arXiv:2611.20405v1 Announce Type: new  Abstract: Prior work shows that refusal in instruction-tuned language models can be ablated by intervening on a single residual-stream direction (Arditi et al., 2024), and more recent work finds multiple refusal-mediating directions and concept cones (Wollschl\"ager et al., 2025). We ask a different question: do refusal-related directions carry information about the register of refusal language - how the model refuses - and not only whether it refuses? In Gemma-2-9B-IT we identify two SVM-derived linear directions (B1 for harmful-content vs. neutral, B2 for self-reference vs. neutral) at layer 20 with cos(B1, B2) = 0.252 (about 75 degrees, substantially non-parallel). Adding the harmful-vs-neutral difference-of-means direction (cosine 0.68 with B1) to the residual stream of self-reference prompts at alpha = 2.0 shifts content-safety vocabulary from 2.55 to 10.37 hits per 100 tokens, while a magnitude-matched random direction produces 1.13-2.16; th
    
[^84]: JointQ：面向大语言模型的尺度与整数联合量化

    JointQ: Joint Scale-and-Integer Quantization for Large Language Models

    [https://arxiv.org/abs/2611.20384](https://arxiv.org/abs/2611.20384)

    提出 JointQ，一种将连续组尺度与离散整数分配联合优化的量化求解器，通过基于闭式最优尺度重计算的局部搜索来改进 GPTQ 的解，从而提升大语言模型训练后量化的精度。

    

    GPTQ 是大语言模型逐层训练后量化（PTQ）的事实标准求解器，然而求解器本身所受到的关注却少于权重预处理和跨层目标。我们针对 GPTQ 的一个瓶颈：其贪心过程将连续的组尺度优化与离散整数的分配分开进行。“结合闭式最优尺度来评估每个整数候选”的做法已被用于逐通道量化；JointQ 将其扩展到分组量化——其中同一行的各尺度通过 Hessian 矩阵相互耦合——并用它来改进 GPTQ 的解。每次局部搜索移动改变一个整数，并以闭式形式重新计算其所在组的最优尺度，同时通过完整的 Hessian 矩阵考虑其他组的影响；所有尺度还会周期性地被联合重新求解。该搜索从包括 GPTQ 解在内的、经过尺度重拟合的候选中进行逐行选择开始，因此未加正则化的 JointQ 从不……

    arXiv:2611.20384v1 Announce Type: cross  Abstract: GPTQ is the de facto standard solver for layer-wise post-training quantization (PTQ) of large language models, yet the solver itself has received less attention than weight preprocessing and cross-layer objectives. We target a bottleneck of GPTQ: its greedy procedure optimizes continuous group scales separately from the assignment of discrete integers. Evaluating each integer candidate with its closed-form optimal scale has been used for per-channel quantization; JointQ extends it to group-wise quantization, where a row's scales are coupled through the Hessian, and applies it to refine the GPTQ solution. Each local-search move changes one integer and recomputes its group's optimal scale in closed form, accounting for the other groups through the full Hessian; all scales are periodically re-solved jointly. The search starts from a row-wise selection among scale-refitted candidates including the GPTQ solution, so unregularized JointQ nev
    
[^85]: 文生图生成中的特质坍塌：视觉叙事中人格不一致性的评估与缓解

    Trait Collapse in Text-to-Image Generation: Evaluating and Mitigating Personality Inconsistency in Visual Storytelling

    [https://arxiv.org/abs/2611.20381](https://arxiv.org/abs/2611.20381)

    该论文发现文生图模型存在“特质坍塌”现象——即模型能保持角色外观一致却无法维持其人格行为的一致性，并提出了包含专家验证视觉线索词典和六维指标体系的ViStoryBench-Personality基准来系统诊断与缓解这一问题。

    

    在视觉叙事中，一个可信的角色不仅需要在各场景间保持外观一致，还需要以反映其人格特质的方式行事。当前的文生图（T2I）模型能够有效保持角色的视觉外观，却始终无法维持与人格相关的行为——我们将这种失败称为“特质坍塌”（trait collapse）。我们证明传统指标（CLIP Score、DINO相似度）无法察觉这一失败，且抽象的人格描述词和场景级情感标签在各模型中相比中性提示词并不能带来一致的改进。为了系统性地诊断特质坍塌，我们提出了ViStoryBench-Personality，这是一个故事级基准，结合了经专家验证的视觉线索词典（Fleiss' κ = 0.757）和涵盖人格对齐、跨场景稳定性与行为连贯性的六维指标体系。通过对七个T2I系统进行评估，我们确认了持续的语义……（摘要原文在此处截断）

    arXiv:2611.20381v1 Announce Type: cross  Abstract: In visual storytelling, a believable character must not only look consistent across scenes but also \textit{behave} in ways that reflect their personality. Current text-to-image (T2I) models preserve visual appearance effectively, yet consistently fail to maintain personality-relevant behavior---a failure we term \textit{trait collapse}. We show that conventional metrics (CLIP Score, DINO similarity) are blind to this failure, and that abstract personality descriptors and scene-level emotion labels do not produce consistent improvements over neutral prompts across models. To systematically diagnose trait collapse, we introduce \textbf{ViStoryBench-Personality}, a story-level benchmark combining an expert-validated Visual Cues Lexicon (Fleiss' $\kappa = 0.757$) with a six-dimensional metric suite covering personality alignment, cross-scene stability, and behavioral coherence. Evaluating seven T2I systems, we confirm a persistent semanti
    
[^86]: 决策模型何时应该比较候选对象？

    When Should Decision Models Compare Candidates?

    [https://arxiv.org/abs/2611.20335](https://arxiv.org/abs/2611.20335)

    该论文系统比较了逐点打分、集合打分和候选对象注意力三种候选评分方式，发现逐点打分总体仍有竞争力，集合打分在特定任务上更优且经关系训练后能学习候选间依赖并泛化，而候选对象注意力即使允许跨候选交互也未带来稳定提升。

    

    语言模型智能体经常需要在候选动作（如工具和网页元素）之间进行选择。决策模型应该独立地为候选对象打分，还是允许其分数依赖于其他候选对象？我们通过比较三种方法来研究这一问题：逐点打分、集合打分和候选对象注意力。在五个智能体决策基准上，逐点打分依然具有竞争力，而集合打分在某些工具选择和二元决策任务上表现更好。尽管候选对象注意力允许跨候选对象的交互，但它并未持续超越逐点打分。借助两个受控的合成任务，我们进一步考察了模型是否学会利用候选集信息。通过有针对性的关系训练，集合打分模型能够学习候选对象之间的依赖关系并泛化到分布偏移的输入，而候选对象注意力的迁移能力较弱。我们的结果表明，能够访问其他候选对象并不保证模型能有效利用候选集信息。

    arXiv:2611.20335v1 Announce Type: new  Abstract: Language-model agents frequently select among candidate actions, such as tools and web elements. Should decision models score candidates independently or allow their scores to depend on other candidates? We investigate this question by comparing three approaches: Pointwise scoring, Setwise scoring, and Candidate Attention. Across five agent-decision benchmarks, Pointwise scoring remains competitive, while Setwise scoring performs better on some tool-selection and binary-decision tasks. Candidate Attention does not consistently improve upon Pointwise scoring, despite allowing cross-candidate interaction. Using two controlled synthetic tasks, we further examine whether models learn to use candidate-set information. With targeted relational training, Setwise models learn candidate dependencies and generalize to shifted inputs, while Candidate Attention shows weaker transfer. Our results show that access to other candidates does not guarante
    
[^87]: Hochschild上循环与合并操作的普适性

    Hochschild Cocycle and Universality of Merge

    [https://arxiv.org/abs/2611.20305](https://arxiv.org/abs/2611.20305)

    论文证明了模拟乔姆斯基极简句法模型中合并操作的Hopf代数马尔可夫链系统满足由合并是Hochschild 1-上循环所决定的普适性质，表明语义的组合结构由句法合并唯一诱导，并与Heim-Kratzer语义学完全兼容。

    

    在非平面二元森林的Hopf代数上构建的Hopf代数马尔可夫链动力系统，模拟了乔姆斯基句法极简模型中合并操作对工作区的作用，该系统满足一个普适性质，而这一性质由合并操作是一个Hochschild 1-上循环这一事实所决定。该普适性在句法-语义界面模型方面有一种解释：它表明，如果语义一方具有足够丰富的组合结构，那么一旦词项和句法特征的语义值被指定，该结构就由句法合并唯一地诱导出来。我们通过特别展示这种Hopf代数表述与Heim-Kratzer语义学的完全兼容性，阐明了这种普适性是如何运作的。

    arXiv:2611.20305v1 Announce Type: new  Abstract: The Hopf algebra Markov chain dynamical system, on the Hopf algebra of non-planar binary forests, that models the action of Merge on workspaces in Chomsky's Minimalist Model of syntax, satisfies a universal property, determined by the fact that the Merge operation is a Hochschild 1-cocycle. This universal property has an interpretation in terms of models of the syntax-semantics interface, as it shows that, if the side of semantics possesses a sufficiently rich compositional structure, then that structure is uniquely induced by the syntactic Merge, once the semantic values of lexical items and syntactic features are assigned. We illustrate how this universality works by showing, in particular, full compatibility of this Hopf algebraic formulation with Heim-Kratzer semantics.
    
[^88]: 使用大语言模型从医院价格透明度数据中估计门诊就诊价格

    Estimating Outpatient Visit Prices from Hospital Price Transparency Data using Large Language Models

    [https://arxiv.org/abs/2611.20261](https://arxiv.org/abs/2611.20261)

    本研究提出一种基于检索增强生成（RAG）框架、结合GPT-BM25混合检索器的大语言模型方法，能够从医院价格透明度数据中准确估计门诊就诊价格，性能远超无RAG基线方法。

    

    目标：美国的医院诊疗价格已大幅上涨，但患者缺乏有效工具来获取和比较费用。我们旨在开发和评估一种基于大语言模型（LLM）的方法，以改善医院价格透明度（PT）数据的获取和使用。材料与方法：为了估计就诊价格，我们使用检索增强生成（RAG）框架分析了马里兰州及周边地区三家主要医院的价格透明度文件。我们比较了BM25、OpenAI嵌入模型和GPT-BM25混合方法与无RAG基线在单个诊疗项目和诊疗项目组费用估计方面的表现，然后使用合成模拟评估了最佳方法在基于症状的就诊费用估计中的效果。结果：在各医院中，GPT-BM25检索器取得了最佳性能，单个诊疗项目的平均价格MAPE为19.44%，项目组为4.90%，而无RAG基线则分别为147.98%和82.82%。

    arXiv:2611.20261v1 Announce Type: cross  Abstract: Objective: Hospital care prices in the United States have increased substantially, yet patients lack effective tools to access and compare costs. We aimed to develop and evaluate an LLM-based approach for improving access to and use of hospital price transparency (PT) data.   Materials and Methods: To estimate visit prices, we used a retrieval-augmented generation (RAG) framework to analyze PT files from three major hospitals in Maryland and the surrounding region. We compared BM25, OpenAI embedding models, and a GPT-BM25 hybrid against a no-RAG baseline for cost estimation of individual procedures and procedure groups, then evaluated the best approach for symptom-based visit cost estimation using synthetic simulations.   Results: Across hospitals, the GPT-BM25 retriever achieved the best performance, with average price MAPE of 19.44% for individual procedures and 4.90% for groups, compared with 147.98% and 82.82%, respectively, for th
    
[^89]: FIDEL：面向埃塞俄比亚语言的元音附标文字感知预分词方法

    FIDEL: Abugida-Aware Pre-tokenization for Ethiopic Languages

    [https://arxiv.org/abs/2611.20216](https://arxiv.org/abs/2611.20216)

    本文提出 FIDEL——一族可逆的、感知埃塞俄比亚元音附标文字结构的预分词方案，通过显式揭示辅音与元音顺序结构来改进埃塞俄比亚各语言的分词质量，并在严格控制变量的条件下验证了其对分词统计与下游 NLP 任务的影响。

    

    子词分词器假设有意义的语言单位可以通过在学习到的边界处切分文本来恢复，这一假设更适合黏着型形态，而不太适合闪米特语的词根-词式系统。对于埃塞俄比亚各语言而言，这一挑战尤为严峻，因为它们的元音附标文字将辅音-元音单元编码为单个字符，从而掩盖了辅音结构。我们提出了 FIDEL（FIdäl DEcomposition and Linearization），这是一族可逆的、元音附标文字感知的预分词方案，能够显式揭示辅音和元音顺序结构，并将 SPLINTER 归约方法从辅音音素文字扩展到埃塞俄比亚元音附标文字；我们在受控条件下对三种埃塞俄比亚语言进行评估，保持语料库、架构、预算和协议固定不变，使分词方式成为唯一变量。我们报告了 BPE 和 UnigramLM 在不同词表规模下的分词器内在统计指标，以及下游 NLP 任务的结果（包括迁移场景）。我们发现……

    arXiv:2611.20216v1 Announce Type: new  Abstract: Subword tokenizers assume that meaningful linguistic units can be recovered by splitting text at learned boundaries, an assumption better suited to concatenative morphology than to Semitic root-and-pattern systems. The challenge is acute for the Ethiopic languages, whose abugida script encodes consonant-vowel units as single characters and hides consonantal structure. We introduce FIDEL (FId\"al DEcomposition and Linearization), a family of reversible abugida-aware pre-tokenization schemes that expose consonant and vowel-order structure, extend SPLINTER reductions from abjad scripts to the Ethiopic abugida, and evaluate them on three Ethiopic languages under controlled conditions, holding corpus, architecture, budget and protocol fixed so that tokenization is the only variable. We report intrinsic tokenizer statistics for BPE and UnigramLM across vocabulary sizes alongside downstream NLP tasks, including transfer scenarios. We find that 
    
[^90]: 我们还需要微调的分类器吗？将类型化决策模型替换进24篇ACL会议论文

    Do We Still Need Fine-Tuned Classifiers? Swapping Typed Decision Models into 24 ACL-Venue Papers

    [https://arxiv.org/abs/2611.20191](https://arxiv.org/abs/2611.20191)

    该研究将无需训练的类型化决策模型Jev替换进24篇ACL会议论文的微调分类器位置，构建了包含46个英语测试集的TDM-Gauntlet基准，发现零样本提示下Jev已在12个测试集上直接取代原论文的预测器。

    

    类型化决策模型（如TypeSafe的Jev，也称为System One或System 1模型）通过为每个选项给出概率来回答关于输入的类型化问题，并且无需任何训练。许多NLP论文仍在微调任务特定的分类器，而经提示的大语言模型（LLM）在大多数文本分类基准上落后于这类分类器。我们探究在每篇论文自身的评估设置下，一个类型化决策模型能替代多少这样的微调分类器，以及在无法替代时需要什么条件。我们将24篇ACL会议论文中的预测器替换为Jev，保留每篇论文原有的数据划分、标签定义和评估指标，并使用Cloudflare的Clef、Microsoft-Decision-1和OpenAI的GPT-6 Luna Decisions核验结果。由此产生的46个英语测试集，连同其问题与已发表的分数，构成了TDM-Gauntlet基准。在零样本提示下，Jev在46个测试集中的12个上直接取代了原论文的预测器，这些大多是单一判断类任务

    arXiv:2611.20191v1 Announce Type: new  Abstract: Typed Decision Models such as TypeSafe's Jev, also called System One or System 1 models, answer typed questions about an input with a probability for each option, and they need no training. Many NLP papers still fine-tune a task-specific classifier, and prompted large language models (LLMs) trail such classifiers on most text classification benchmarks. We ask how many of these fine-tuned classifiers a Typed Decision Model can replace under each paper's own evaluation, and what is needed where it cannot. We swap the predictors of 24 ACL-venue papers for Jev, keep each paper's splits, label definitions and metric, and check the results with Cloudflare's Clef, Microsoft-Decision-1 and OpenAI's GPT-6 Luna Decisions. The resulting 46 English test sets, with their questions and published scores, form the TDM-Gauntlet benchmark. With a zero-shot prompt, Jev directly replaced the paper's predictor on 12 of the 46 test sets, mostly single judgmen
    
[^91]: 你能躲在谁身后？基于作者条件的风格迁移与对商业AI文本检测器的规避

    Who Can You Hide Behind? Author-Conditioned Style Transfer and the Evasion of a Commercial AI-Text Detector

    [https://arxiv.org/abs/2611.20158](https://arxiv.org/abs/2611.20158)

    该研究提出一种基于作者条件的风格迁移方法，即在Gemini提示词中嵌入55位具名作者的写作样本来重写AI生成文本，结果显示这种“躲在知名作者身后”的策略能使商业AI文本检测器的评分中位数从100%骤降至0%，成功实现检测规避。

    

    AI文本检测器可能对写作风格的变化十分敏感。我们通过使用Gemini重写991个AI生成的答案来检验这种敏感性，其中提示词包含了55位具名作者之一的写作样本。我们使用QuillBot AI检测器对每个原始答案及其重写版本进行评分。评分中位数从100%降至0%，平均值从67.9%降至11.2%。在所有重写版本中，66%的评分恰好为0%。在699个原始评分不低于50%的样本中，71%的重写版本评分低于10%（配对Wilcoxon符号秩检验，p值低于1e-100）。评分因提示词中使用的作者不同而存在差异：与10位作者相关的所有重写版本评分均为0%，而六位作者的平均评分仍高于50%（Kruskal-Wallis检验，p值低于1e-80）。对作者样本中五词序列的复用率越高，评分越低，但即便复用率低于5%的重写版本，平均评分也仅为15.2%。这些结果表明……（摘要原文截断）

    arXiv:2611.20158v1 Announce Type: new  Abstract: AI-text detectors can be sensitive to changes in writing style. We examined this sensitivity by rewriting 991 AI-generated answers with Gemini, using prompts that included a writing sample from one of 55 named authors. We scored each original and its rewrite with the QuillBot AI detector. The median score fell from 100% to 0%, and the mean fell from 67.9% to 11.2%. Of the rewrites, 66% received a score of exactly 0%. Among the 699 originals scored at least 50%, 71% of rewrites scored below 10% (paired Wilcoxon signed-rank test, p-value below 1e-100). Scores varied across the authors used in the prompts. All rewrites associated with 10 authors scored 0%, whereas the mean score remained above 50% for six authors (Kruskal-Wallis test, p-value below 1e-80). Greater reuse of five-word sequences from the author sample was associated with lower scores, although rewrites with less than 5% reuse still averaged 15.2%. These results describe one de
    
[^92]: 面向LLM-as-a-judge安全护栏的风险认证提示词优化

    Risk-certified prompt optimization for LLM-as-a-judge guardrails

    [https://arxiv.org/abs/2611.20156](https://arxiv.org/abs/2611.20156)

    本文将LLM裁判护栏的系统提示词选择建模为带有限样本风险认证的约束优化问题，提出CPS和CRISP两种方法，在统计保证漏检攻击率不超过用户指定阈值α的前提下最大化良性通过率。

    

    LLM-as-a-judge（大语言模型作为裁判）安全护栏通常主要通过系统提示词来控制，然而标准的提示词优化方法针对的是平均性能而非显式的安全约束。本文将系统提示词的选择问题建模为一个约束优化问题：在保证漏检攻击率低于用户指定水平α的同时，最大化良性通过率，并提供有限样本统计控制。该框架结合了基于LLM的评判、自动提示词生成和无分布风险认证。这项工作将该框架具体化为两种流程：CPS和CRISP。CPS从训练数据中生成固定的提示词池，然后使用同步置信上界在留出的评估集上对候选提示词进行风险认证，并选择认证通过且良性效用最高的提示词。CRISP则将风险反馈整合到自适应搜索中，利用难例引导的提示词重写和可复用的留出集接口来引导……

    arXiv:2611.20156v1 Announce Type: new  Abstract: LLM-as-a-judge guardrails are often controlled primarily through system prompts, yet standard prompt optimization methods target average performance rather than explicit safety constraints. This paper studies system-prompt selection as a constrained optimization problem: maximize benign pass rate while keeping the missed-attack rate below a user-specified level {\alpha} with finite-sample statistical control. The framework combines LLM-based judging, automatic prompt generation, and distribution-free risk certification. This work instantiates it in two procedures: CPS and CRISP. CPS generates a fixed pool of prompts from training data, then certifies candidates on a held-out evaluation set using simultaneous upper confidence bounds and selects the certified prompt with the highest benign utility. CRISP instead integrates risk feedback into an adaptive search, using hard example-guided rewriting and a reusable-holdout interface to steer g
    
[^93]: 中心化消融：为什么最后一层的方向消融会触发不足或崩溃，以及一个闭式修复方法

    Centered Ablation: Why Last-Layer Directional Ablation Under-Fires or Collapses, and a Closed-Form Fix

    [https://arxiv.org/abs/2611.20148](https://arxiv.org/abs/2611.20148)

    论文揭示了方向消融不稳定的根源——最后一层隐藏状态包含远离原点的大共模分量导致门控系数失准，并提出闭式的中心化消融（CA）方法，通过投影去除共模，在8个模型上大幅提升攻击成功率、将文本崩溃率从50%降至1%并保留更多良性回答。

    

    方向消融在语言模型运行时移除隐藏状态沿“行为方向”的分量。它是关闭某种行为（如拒答）的标准方法，但其表现不稳定：在某些模型上几乎不生效，而在另一些模型上文本则崩溃为重复。我们解释了原因，并给出了修复方案。在最后一个残差流层上，消融恰好等价于一个逻辑偏置：一个固定的词表足迹乘以一个门控系数——即隐藏状态与方向之间的余弦值。该门控是从原点测量的，但最后一层的隐藏状态共享一个远离原点的大共模分量，因此与共模重叠的方向会在良性文本上触发，而在其目标上触发不足。中心化消融以闭式形式将共模从方向中投影出去。在8个模型上，CA 在标准方法触发不足的情况下将攻击成功率提升了29和46个百分点，将 Gemma-2-9B 上的崩溃率从50%降至1%，并在……上多保留了12到40个百分点的良性回答（原文摘要至此截断）。

    arXiv:2611.20148v1 Announce Type: new  Abstract: Directional ablation removes a hidden state's component along a "behaviour direction" while a language model runs. It is the standard way to switch a behaviour such as refusal off, but it is erratic: on some models it barely fires, on others the text collapses into repetition. We show why, and fix it. At the last residual stream, ablation is exactly a logit bias: a fixed vocabulary footprint scaled by a gate, the cosine between the hidden state and the direction. The gate is measured from the origin, but last-layer states share a large common mode far from it, so a direction that overlaps the common mode fires on benign text and under-fires on its targets. Centered Ablation (CA) projects the common mode out of the direction, in closed form. Across 8 models, CA raises attack success by 29 and 46 points where the standard recipe under-fires, cuts collapse on Gemma-2-9B from 50\% to 1\%, and preserves 12 to 40 points more benign answers on 
    
[^94]: 排序而非回归：一个279M参数的文本排序器在指代性乌尔都语GUI定位任务上击败2B和7B视觉语言模型

    Rank, Don't Regress: A 279M Text Ranker Beats 2B and 7B VLMs on Referential Urdu GUI Grounding

    [https://arxiv.org/abs/2611.20145](https://arxiv.org/abs/2611.20145)

    一个仅279M参数的文本排序器通过从页面DOM中排序选择元素而非回归坐标，在指代性乌尔都语GUI定位任务上以73.4%的准确率大幅超越零样本的2B和7B视觉语言模型。

    

    UI定位要求模型找到用户指令所指向的屏幕元素。当前大多数系统使用视觉语言模型（VLM）读取截图并输出像素坐标。我们在乌尔都语移动网页上研究该任务，这是一个此前没有公开发表UI定位结果的场景。我们没有采用回归坐标的方式，而是从页面DOM中提取候选元素，训练一个279M参数的交叉编码器从中选出正确元素。点击点即为所选元素框的中心，因此只要选对元素，点击位置就是精确的。我们基于六个乌尔都语新闻网站构建了包含304条指令的冻结测试集，其中149条（49%）是指代性的，从不完整引用目标文本。测试域名在数据收集开始前就被排除在训练数据之外。该排序器取得73.4%的点在框内准确率，而零样本的UI-TARS-2B和Qwen2.5-VL-7B分别仅为29.6%和23.7%。在指代性子集上，该排序器取得65.8%的准确率。

    arXiv:2611.20145v1 Announce Type: new  Abstract: UI grounding asks a model to find the screen element that a user instruction refers to. Most current systems use a vision-language model (VLM) that reads a screenshot and outputs pixel coordinates. We study this task on Urdu mobile web pages, a setting with no published UI grounding results. Instead of regressing coordinates, we take the candidate elements from the page DOM and train a 279M-parameter cross-encoder (mDeBERTa-v3-base) to pick one. The click point is the centre of the picked element's box, so it is exact whenever the right element is picked. We built a frozen test set of 304 instructions from six Urdu news sites; 149 of them (49%) are referential and never quote the full target text. Test domains were banned from training data before collection started. The ranker scores 73.4% point-in-box accuracy, against 29.6% for UI-TARS-2B and 23.7% for Qwen2.5-VL-7B, both zero-shot. On the referential slice the ranker scores 65.8%, ag
    
[^95]: HEAMS：智能体记忆系统的全面评估

    HEAMS: Holistic Evaluation of Agent Memory Systems

    [https://arxiv.org/abs/2611.20144](https://arxiv.org/abs/2611.20144)

    该论文提出HEAMS开源评估框架，将智能体记忆系统视为用户持久信息的管理者，主张从隐私保护、可更正性、数据保留和抗投毒等超越任务性能的全面责任角度对其进行整体评估。

    

    记忆系统使AI智能体能够在多次交互中持久化信息，从而使其更具个性化，并在长时程任务上表现得更有能力。这种持久性带来了新的风险：存储的信息可能在不当的情境中被泄露、抵制更正或删除、被保留的时间超出用户意愿，或被投毒以影响智能体后续的行为。然而，记忆系统通常仅在任务性能和效率方面进行评估。我们认为，记忆系统应当被理解为用户持久信息的管理者，其责任超越了帮助智能体完成任务的范畴，并且应该针对所有这些责任进行整体评估。我们提出了智能体记忆系统全面评估框架（HEAMS），这是一个以这种方式评估智能体记忆系统的开源框架。首先，我们将智能体记忆系统的职责表述为明确的行为设计要求……（摘要在此处被截断）

    arXiv:2611.20144v1 Announce Type: new  Abstract: Memory systems allow AI agents to persist information across interactions, making them more personalized and more capable on long-horizon tasks. This persistence introduces new risks: stored information can be disclosed in inappropriate contexts, resist correction or deletion, be retained beyond what users desire, or be poisoned to influence later agent behaviour. Yet memory systems are typically evaluated on task performance and efficiency alone. We argue that a memory system should instead be understood as the custodian of its user's persistent information, with responsibilities that extend beyond helping an agent complete its tasks, and that it should be evaluated holistically against all such responsibilities. We present Holistic Evaluation of Agent Memory Systems (HEAMS), an open-source framework for evaluating agent memory systems in this way. First, we state the responsibilities of agent memory systems as explicit behavioural desi
    
[^96]: AI能评判你吗？对LLM辅助招聘系统部署就绪度的实证审计

    Can AI Judge You? An Empirical Audit of Deployment Readiness in LLM-assisted Hiring Systems

    [https://arxiv.org/abs/2611.20140](https://arxiv.org/abs/2611.20140)

    本文提出了一种将候选人证据、评估、权限控制与评估记录分离的LLM辅助简历筛选系统架构，并通过涵盖210份简历、14个职位和6种决策界面的受控实验，从运行就绪度与合规就绪度两个维度实证审计了其部署就绪度，发现标量分数比通过/不通过判断更能有效区分资质变化。

    

    将简历筛选工作委托给大语言模型（LLM）时，组织需要确认系统能否依据相关资质区分候选人，同时只允许被授权的用户修改筛选规则或批准例外。为此，我们开发了一种将候选人证据、评估、权限控制和评估记录相互分离的系统架构，并从两个互补的维度对其部署就绪度进行了实证检验。第一，我们通过一项涵盖210份简历、14个职位角色和6种决策界面的受控实验，评估了该系统在4个LLM上的运行就绪度。第二，我们通过测试系统如何应对未授权用户的请求以及嵌入在简历中的指令，评估了其合规就绪度。我们发现，运行性能取决于评估界面的选择：标量分数通常比简单的通过/不通过判断更能有效地区分资质变化，而比较类……（原文摘要在此处截断）

    arXiv:2611.20140v1 Announce Type: cross  Abstract: Delegating resume screening to large language models (LLMs) requires organizations to establish whether they can distinguish candidates by relevant qualifications while allowing only authorized users to change screening rules or approve exceptions. To this end, we develop a system architecture that separates candidate evidence, evaluation, permission controls, and assessment record, and empirically examine two complementary dimensions of its deployment readiness. First, we assess its operational readiness on four LLMs using a controlled experiment on 210 resumes across 14 job roles and six decision interfaces. Second, we assess its compliance readiness by testing how it handles unauthorized user requests and instructions embedded in resumes. We find that operational performance depends on the choice of evaluation interface: scalar scores generally distinguish qualification changes better than simple pass/fail judgments, while comparati
    
[^97]: 决策模型能否从癌症病理报告中提取临床数据？对准确性、校准与失败模式的评估

    Can decision models extract clinical data from cancer pathology reports? An evaluation of accuracy, calibration, and failure modes

    [https://arxiv.org/abs/2611.20129](https://arxiv.org/abs/2611.20129)

    该研究首次在临床文本上评估决策模型，发现 Tev1 从口腔癌病理报告中提取 25 个字段的准确率（0.917）与生成式模型相当且速度快四倍，但在分箱测量值和答案未明确表述的字段上表现较弱。

    

    决策模型是一类新型语言模型，它不生成文本，而是针对每个问题从固定选项列表中选择一个选项，并为每个选项返回一个概率。此前这类模型尚未在临床文本上得到评估。我们评估了两个开放权重的决策模型——Nimble 9B 和 Tev1 4B——从单一中心的 1,113 份口腔癌病理报告中提取 25 个字段，并与前瞻性维护的临床数据库进行对比；同时与两个生成式模型进行比较：本地运行的 Qwen3.5 9B 和通过云端 API 运行的 GLM-5.3-flash。Tev1 的平均字段准确率达到 0.917（95% CI 0.913–0.920），与 Qwen3.5 9B 相当（0.918；p = 0.26），且速度快四倍（每份报告 15 秒对 61 秒）。Nimble 达到 0.862，GLM-5.3-flash 达到 0.986。两个决策模型在简单的二分类项目上与 GLM-5.3-flash 表现相当，但在被分箱为类别的测量值、以及正确答案未在报告中明确表述的字段上表现较弱。

    arXiv:2611.20129v1 Announce Type: cross  Abstract: Decision models are a new class of language model that, instead of generating text, choose one option per question from a fixed list and return a probability for every option. They have not yet been evaluated on clinical text. We evaluated two open-weights decision models, Nimble 9B and Tev1 4B, for extracting 25 fields from 1,113 oral cavity cancer pathology reports from a single centre, against a prospectively maintained clinical database. We compared them with two generative models: Qwen3.5 9B, run locally, and GLM-5.3-flash, run through a cloud API. Tev1 reached a mean field accuracy of 0.917 (95% CI 0.913 to 0.920), the same as Qwen3.5 9B (0.918; p = 0.26), and was four times faster (15 s against 61 s per report). Nimble reached 0.862 and GLM-5.3-flash 0.986. Both decision models matched GLM-5.3-flash on simple dichotomous categories. They were weak on measurements binned into categories, on fields whose correct answer was not rep
    
[^98]: 一种基于语料库驱动的信德语 SentencePiece Unigram 分词器：设计及与十四种现有分词器的基准对比

    A Corpus-Driven SentencePiece Unigram Tokenizer for Sindhi: Design and Benchmark Against Fourteen Existing Tokenizers

    [https://arxiv.org/abs/2611.20126](https://arxiv.org/abs/2611.20126)

    该论文首个针对信德语这一低资源语言训练了带字节回退的语料库驱动 SentencePiece Unigram 分词器，并在四个评估集上与十四种现有分词器进行系统基准对比，结果显示其在分词碎片率、单词单token保留率和UNK率等关键指标上优于现有最佳分词器。

    

    信德语是一种低资源的印度-雅利安语言，主要使用扩展的波斯-阿拉伯文字书写。目前已有多个信德语分词器在公共模型库上发布，但它们尚未在共享数据上进行相互比较。我们在来自四个公开来源的约110万句信德语句子，以及一个较小的、重复使用的乌尔都语、波斯语和阿拉伯语数据切片上，训练了四种词表规模（32k至150k）的带字节回退机制的 SentencePiece Unigram 分词器，并在四个评估集上与十四种现有分词器进行对比：FLORES+、信德语维基百科、一个源自新闻的情感语料库和一个当代新闻语料库。评估指标包括分词碎片率、单词单token保留率、UNK率、字节回退率和往返字符损失。在64k词表规模下，我们的分词器平均每个单词产生1.147个token，而现有最佳分词器为1.208，XLM-RoBERTa为1.512，且88.6%的单词被保留为单个token。在分词碎片率、单词单token保留率和UNK率方面，它（摘要在此处截断）

    arXiv:2611.20126v1 Announce Type: new  Abstract: Sindhi is a low-resource Indo-Aryan language written mainly in an extended Perso-Arabic script. Several Sindhi tokenizers have been released on public model hubs, but they have not been compared with one another on shared data. We train SentencePiece Unigram tokenizers with byte fallback at four vocabulary sizes (32k to 150k) on about 1.10 million Sindhi sentences and a smaller, repeated Urdu, Persian and Arabic slice, from four public sources, and compare them with fourteen existing tokenizers on four evaluation sets: FLORES+, Sindhi Wikipedia, a news-derived sentiment corpus, and a contemporary news corpus. The metrics are fertility, single-token retention, UNK rate, byte-fallback rate, and round-trip character loss. At 64k, our tokenizer averages 1.147 tokens per word, against 1.208 for the best existing tokenizer and 1.512 for XLM-RoBERTa, and keeps 88.6% of words as single tokens. On fertility, single-token retention and UNK rate it
    
[^99]: FRAAC：一个面向基于象形图的辅助沟通的法语多模态大语言模型与数据集

    FRAAC: A French Multimodal Large Language Model and Dataset forPictogram-Based Augmentative and Alternative Communication

    [https://arxiv.org/abs/2611.20094](https://arxiv.org/abs/2611.20094)

    该论文提出了RAPS标注方法，通过大语言模型概念提取与象形图词汇检索改进规则式转录，构建了140万对法语平行语料库并发布了30万对，显著提升了象形图转录质量，支撑了面向辅助沟通的法语多模态大语言模型FRAAC。

    

    象形图是辅助沟通中的核心模态，然而近期的多模态大语言模型（MLLM）尚不能可靠地理解或生成象形图序列。造成这一差距的原因有二：一是缺乏标准化的象形图语法，二是文本与象形图之间的平行数据稀缺。我们提出了检索增强象形图选择方法（Retrieval-Augmented Pictogram Selection, RAPS），这是一种标注方法，它利用基于大语言模型的概念提取和对象形图词汇库的检索来改进基于规则的转录系统的输出，能够处理手写规则无法应对的情况（如命名实体、数字、多义词），且无需编写新规则。将RAPS应用于法语，生成了一个包含140万对平行语料的合成语料库，其中30万对已公开发布。在一项人类排序研究中，标注者对同一句话的四种象形图转录结果进行排序，RAPS的表现超越了基于规则的现有最先进方法。此外，我们还表明……

    arXiv:2611.20094v1 Announce Type: cross  Abstract: Pictograms are a core modality in augmentative and alternative communication (AAC), yet recent multimodal large language models (MLLMs) cannot reliably interpret or produce pictogram sequences. Two obstacles explain this gap: the absence of a standardized pictogram grammar and the scarcity of parallel text-pictogram data. We introduce Retrieval-Augmented Pictogram Selection (RAPS), an annotation method that refines a rule-based transcription system's output with LLM-based concept extraction and retrieval over a pictogram vocabulary, handling cases where hand-written rules fail (named entities, numbers, polysemous words) without requiring new rules. Applied to French, RAPS yields a synthetic corpus of 1.4 million parallel pairs, of which 300k are publicly released. In a human ranking study where annotators order four pictogram transcriptions of the same sentence, RAPS outperforms the rule-based state of the art. Moreover, we show that s
    
[^100]: LLM推理中对话历史的隐性代价

    The Silent Cost of Conversation History in LLM Reasoning

    [https://arxiv.org/abs/2611.20071](https://arxiv.org/abs/2611.20071)

    研究发现累积的对话历史会显著降低LLM在多轮任务中的推理准确率，且过程级质量指标无法提前预警这种下降，而用经求解器验证的状态摘要替代完整历史可挽回大部分性能损失。

    

    在多轮信息系统中，大语言模型（LLM）会不断累积自身早期的回复以及持续增长的任务状态。我们探究了这种对话历史是否会降低准确率，以及过程级质量指标能否在准确率下降之前发出预警。每项分析均在计算结果之前预先设定。在研究1中，针对48个分片数学任务，保留模型自身回复使GPT-4o-mini的最终成功率降低0.21，Mistral-7B降低0.29（经Holm校正的p值分别为0.026和0.004）；Llama-4 Scout呈现相同趋势，但不显著。研究2引入了CDR-Stream基准，包含100轮状态追踪对话，每轮均配有精确答案并针对各模型进行了难度校准。在相同难度下，使用完整历史时，准确率在二十轮内从约0.6降至0.1；而使用经求解器验证的状态摘要，准确率可保持高出0.52至0.59（经Holm校正的p值小于0.001）。一致性与逻辑连贯性的下降幅度远小于准确率，而稳定性反而有所上升。

    arXiv:2611.20071v1 Announce Type: cross  Abstract: Large language models (LLMs) in multi-turn information systems accumulate their own earlier replies and a growing task state. We asked whether this conversation history lowers accuracy, and whether process-level quality metrics warn before it does. Each analysis was fixed before its outcomes were computed. In Study 1, on 48 sharded math tasks, retaining its replies lowered final success by 0.21 for GPT-4o-mini and 0.29 for Mistral-7B (Holm-adjusted p = 0.026 and 0.004); Llama-4 Scout showed the same direction, not significant. Study 2 introduces CDR-Stream, 100-turn state-tracking dialogues with an exact answer per turn and per-model difficulty calibration. With the full history, accuracy fell from about 0.6 to 0.1 within twenty turns at matched difficulty, whereas a solver-verified state summary kept it 0.52-0.59 higher (Holm-adjusted p less than 0.001). Consistency and logical coherence fell far less than accuracy, and stability rose
    
[^101]: 多语言大模型中汉语wh不定代词遵循非真实性许可规律，而英语any则不然

    Mandarin Wh-Indefinites Follow Nonveridicality in Multilingual LLMs, Unlike English any

    [https://arxiv.org/abs/2611.20062](https://arxiv.org/abs/2611.20062)

    研究发现，多语言大模型虽然主要从英语学习，却能让汉语wh不定代词遵循汉语特有的非真实性许可规律（如被“也许”等词许可），而英语any的许可模式并未迁移到汉语中，表明模型习得了语言特定的极性许可知识。

    

    多语言大模型的训练数据主要来自英语。那么它们是否会将英语的极性许可规则迁移到其他语言中？英语中的any由向下蕴含（DE）语境许可，而汉语中的wh不定代词（如“什么”）则由非真实性语境许可，这两类词存在差异：“也许”能够许可“什么”却不能许可any，而“只”能够许可any但蕴含存在性。我们通过一项预先注册的研究对十二个开源大语言模型（0.5B-9B，涵盖四个模型家族）进行了测试，使用了48个匹配的双语测试框架、21种语境，以及经否定归一化的对数概率双重差分方法。所有11个通过测量门槛的模型均在可能性、必然性和证据类副词下许可汉语wh不定代词，其许可强度达到否定语境的0.42至2.33倍，而英语any在相同语境下则完全没有得到许可。对Qwen2.5-32B的探索性扩展实验也显示了相同的结果。英语方面则较弱：any在“只”、“首次”和“后悔”下虽然得到许可，但其强度（摘要在此处截断）

    arXiv:2611.20062v1 Announce Type: new  Abstract: Multilingual LLMs learn mostly from English. Do they carry English polarity licensing over to other languages? English any is licensed by downward-entailing (DE) contexts, Mandarin wh-indefinites such as shenme 'what/something' by nonveridical ones, and the two classes come apart: perhaps licenses shenme but not any, while only licenses any but entails existence. We test this dissociation in a pre-registered study of twelve open LLMs (0.5B-9B, four families), using 48 matched bilingual frames in 21 contexts and a log-probability difference-in-differences normalised by negation. All 11 models that pass the measurement gate license Mandarin wh-indefinites under possibility, necessity and evidential adverbs, at 0.42-2.33 times the strength of negation, while English any gets nothing in the same contexts. An exploratory extension to Qwen2.5-32B shows the same. The English side is weaker: any under only, first and regret is licensed but at a 
    
[^102]: 一个用于波兰语社交媒体厌女症检测的数据集：仇恨言论检测从一般类型到特定类型仇恨的可迁移性研究

    A Dataset for Misogyny Detection in Polish Social Media: Transferability of Hateful Speech Detection from General to Specific Types of Hate

    [https://arxiv.org/abs/2611.20044](https://arxiv.org/abs/2611.20044)

    本文发布了首个波兰语社交媒体厌女症检测标注数据集，并通过实验发现在一般有害语言检测上的优异性能并不能均匀迁移到厌女症检测任务，而安全专业化模型则展现出更强的泛化能力。

    

    有害语言检测通常在广义定义的仇恨言论和毒性基准上进行评估，然而，在一般有害语言检测上表现成功的模型能否将这种能力迁移到更具体的仇恨形式，目前仍不清楚。我们在波兰语厌女症检测领域研究了这一问题，而该领域的专门资源仍然稀缺。我们引入了一个新的波兰语社交媒体评论数据集，该数据集针对厌女症（即针对性别的仇恨言论）进行了标注。利用这一资源，我们首先评估了大语言模型辅助标注的可靠性，然后通过比较在一般有害语言数据上训练的模型、安全专业化模型以及直接针对厌女症检测进行微调的模型，评估了从一般到特定类型的迁移能力。我们的结果表明，在一般有害语言检测上的优异性能并不能均匀地迁移到厌女症检测任务；而安全专业化模型表现出显著更强的泛化能力。

    arXiv:2611.20044v1 Announce Type: new  Abstract: Harmful language detection is commonly evaluated on broadly defined hate speech and toxicity benchmarks, yet it remains unclear whether models that successfully detect harmful language in general can transfer this capability to more specific forms of hate. We investigate this problem for misogyny detection in Polish, an area for which dedicated resources remain scarce. We introduce a new dataset of Polish social media comments annotated for misogyny, gender-directed hateful speech. Using this resource, we first assess the reliability of LLM-assisted annotation and then evaluate general-to-specific transfer by comparing models trained on general harmful language data, safety-specialized models, and models fine-tuned directly for misogyny detection. Our results show that strong performance on general harmful language detection does not transfer uniformly to misogyny; safety-specialized models demonstrate substantially stronger generalizati
    
[^103]: 离线优先的量化人工智能：面向低网络连接环境的设备端孕产妇健康助手的设计与评估

    Offline-First Quantized AI: Design and Evaluation of an On-Device Maternal-Health Assistant for Low-Connectivity Environments

    [https://arxiv.org/abs/2611.20012](https://arxiv.org/abs/2611.20012)

    本研究设计并评估了一款面向低网络连接环境的离线安卓孕产妇健康助手，通过集成量化语言模型、本地临床知识包与设备端检索，在临床质量、安全性与本地可行性之间实现了有效平衡。

    

    离线语言模型可以在互联网连接不可靠的地区扩大孕产妇健康信息的获取途径，但部署既需要临床上恰当的回复，也需要可行的本地运行。本研究提出了一款离线安卓助手，它集成了现成的量化语言模型、经临床验证的本地知识包、对话上下文以及由模型发起的本地检索。该研究在45次对话、每个配置312轮对话的条件下比较了Gemma 4 E2B与Qwen3.5 0.8B，其中包括一项专门的离线智能体基准测试。Gemma在回复质量、上下文保留和安全性方面获得了更高的评分；而Qwen则具有更快的生成速度和更低的内存占用。临床医生对Gemma在九个临床领域的150条回复进行评估，平均质量得分为3.80/4，其中147条回复达到了接受标准。一项配对的RAG开/关实验评估了74个问题：检索改善了11条回复，（摘要在此处截断）

    arXiv:2611.20012v1 Announce Type: new  Abstract: Offline language models could extend access to maternal-health information where internet connectivity is unreliable, but deployment requires both clinically appropriate responses and feasible local execution. This study presents an offline Android assistant integrating ready-made quantized language models, a clinically validated local knowledge package, conversational context, and model-initiated local retrieval. Gemma 4 E2B and Qwen3.5 0.8B were compared across 45 conversations and 312 turns per configuration, including a dedicated offline agentic benchmark. Gemma achieved higher response-quality, context-retention, and safety scores; Qwen offered faster generation and lower memory use. Clinician assessment of 150 Gemma responses across nine clinical domains yielded a mean quality score of 3.80/4, with 147 responses meeting the acceptance criterion. A paired RAG-on/off experiment evaluated 74 questions: retrieval improved 11 responses,
    
[^104]: 揭示新颖性：40份开放权重大型语言模型（LLM）技术报告中的消融实验、可复现性与透明度

    The novelty exposed: ablation, reproducibility and transparency in 40 technical reports of open-weight large language models (LLMs)

    [https://arxiv.org/abs/2611.20003](https://arxiv.org/abs/2611.20003)

    该研究通过分析Hugging Face库代码与40份随机抽样的开放权重LLM技术报告，发现2025年新增的48种架构大多复用现有模型（以Llama为主）的核心模块，且73项新颖性声明中仅19项有消融实验支持，揭示了当前LLM技术报告在新颖性验证、可复现性和透明度方面的普遍不足。

    

    仅在2025年，就有48种新的大型语言模型（LLM）架构被添加到Hugging Face transformers库中，其中许多模型都附有一份将其呈现为创新之作的技术报告。这些模型究竟有什么真正新颖之处，以及这些新颖之处是否经过测试？我们将这一问题抛给了LLM技术报告。我们考察三个方面：参考代码显示了新的开放权重架构与早期架构有何不同；40份随机抽样的技术报告声称了哪些新颖之处，以及它们是否用消融实验来支持这些声明；以及它们为可复现性和透明度披露了哪些信息。在该库中，2025年添加的48种架构中有39种的三个核心模块中至少有两个来自其他模型，最常见的是Llama，尽管其中部分原因反映了该库编写代码的方式。在报告中，73项架构或训练新颖性声明中仅有19项附带受控的有/无对比实验；40份报告中有4份声明了不止一次完整训练运行……

    arXiv:2611.20003v1 Announce Type: cross  Abstract: In 2025 alone, 48 new large language model (LLM) architectures were added to the Hugging Face transformers library, and many of them arrive with a technical report that presents the model as new. What is actually new in these models, and is it tested? We put this question to LLM technical reports. We look at three things: what the reference code shows about how new open-weight architectures differ from earlier ones; what 40 randomly sampled technical reports claim as new and whether they support it with ablation studies; and what they disclose for reproducibility and transparency. In the library, 39 of the 48 architectures added in 2025 take at least two of their three core blocks from another model, most often Llama, although part of this reflects how the library writes code. In the reports, 19 of 73 architectural or training novelty claims come with a controlled with/without comparison; 4 of 40 reports state more than one training ru
    
[^105]: 智能体框架剖析：决策、不变量与边界情况

    Anatomy of an Agent Harness: Decisions, Invariants and Corner Cases

    [https://arxiv.org/abs/2611.19960](https://arxiv.org/abs/2611.19960)

    本文首次将智能体框架中隐含于代码、鲜有记录的关键工程决策系统化为六组30张“决策卡”——每张包含选项与代价、可测试的不变量、破坏性边界情况、测试与证据标签——并配套提供带信任边界的系统模型、六类故障成因、基于脚本化提供者的测试方法与检查清单，为智能体框架的健壮性设计与测试建立了系统化方法论。

    

    智能体框架是围绕语言模型运行的程序，它将模型的回复转化为行动：它负责构建每次请求、运行模型所要求的工具、决定模型可以在无需询问的情况下执行哪些操作、存储对话内容、从故障中恢复，并向人类展示发生了什么。这些决策大多隐含在代码中而很少被记录下来，因此每个框架都会一再遇到相同的边界情况。本文将这些决策记录下来，形成可测试的不变量。论文提出了一个带信任边界的系统模型；六组共30张决策卡，每张卡包含选项及其代价、一个不变量、破坏该不变量的边界情况、一个测试和一个证据标签；六类故障成因；一种基于脚本化提供者、记录行为和植入变更的测试方法；以及一份检查清单。这些决策卡来源于作者自行构建的一个桌面框架 Crewroom、对其他十个框架的人工测试，以及以开源框架为参考的研究；部分模式……

    arXiv:2611.19960v1 Announce Type: new  Abstract: An agent harness is the program around a language model that turns its replies into actions: it builds each request, runs the tools the model asks for, decides what the model may do without asking, stores the conversation, recovers from failures, and shows a person what happened. Most of these decisions are made in code and rarely written down, so each harness meets the same corner cases again. This paper writes them down as invariants to test. It gives a system model with trust boundaries; 30 decision cards in six groups, each with options and costs, an invariant, the corner cases that break it, a test and an evidence tag; six cause classes for failures; a test method built on scripted providers, recorded behaviour and planted changes; and a checklist. The cards come from building one desktop harness, Crewroom, the author's own, and from manually testing ten other harnesses and studying open-source harnesses as reference; some patterns 
    
[^106]: 学习何时继续与何时停止：终止感知的OPSD

    Learning When to Continue and When to Stop: Termination-Aware OPSD

    [https://arxiv.org/abs/2611.19925](https://arxiv.org/abs/2611.19925)

    提出TA-OPSD（终止感知的在线策略自蒸馏），通过引入针对终止行为的辅助目标，解决OPSD训练中学生模型无法学会何时继续推理、何时停止的问题，从而缓解过早终止与重复驱动的过度生成两类失败模式。

    

    在线策略自蒸馏（OPSD）通过与初始化完全相同且以特权信息为条件的冻结教师模型进行蒸馏，从而提升语言模型的推理能力。然而，我们发现OPSD中存在一个此前未被充分研究的终止失配问题：学生模型无法可靠地学会何时继续推理、何时停止。这种失配表现为两种截然相反的失败模式——过早终止，即在取得足够进展之前推理就停止了；以及重复驱动的过度生成，即模型在推理停滞之后仍继续生成。我们的分析表明，在OPSD训练过程中，教师与学生模型在EOS（序列结束符）上的差异会不断增大，而受长度限制的轨迹往往被无效的重复所主导。为解决这一问题，我们提出了TA-OPSD（终止感知的在线策略自蒸馏），它在原始OPSD目标的基础上增加了两个针对性目标。其中，停止/继续蒸馏用于解决过早终止问题（摘要原文在此处被截断）。

    arXiv:2611.19925v1 Announce Type: cross  Abstract: On-Policy Self-Distillation (OPSD) improves language model reasoning by distilling from a frozen teacher initialized identically to the student and conditioned on privileged information. However, we identify an underexplored termination mismatch in OPSD: the student fails to reliably learn when to continue and when to stop. This mismatch manifests in two opposing failure modes-premature termination, where reasoning stops before sufficient progress is made, and repetition-driven over-generation, where the model continues generating after reasoning has stalled. Our analysis shows that the teacher-student discrepancy on EOS grows during OPSD training, while length-capped trajectories are frequently dominated by unproductive repetition. To address this issue, we propose TA-OPSD (Termination-Aware On-Policy Self-Distillation), which augments the original OPSD objective with two targeted objectives. Stop/Continue Distillation addresses prema
    
[^107]: 词汇税：为设备端使用安全裁剪大语言模型中未使用的词元

    The Vocabulary Tax: Safely Trimming Unused Tokens from LLMs for On-Device Use

    [https://arxiv.org/abs/2611.19913](https://arxiv.org/abs/2611.19913)

    该论文提出“词汇税”概念，量化了大语言模型词汇表中未使用词元造成的存储浪费（对边缘设备上的小模型影响最大），并提出在保持BPE分词器正确性的前提下安全裁剪这些未使用词元以缩小设备端模型体积。

    

    开源大语言模型（LLM）和语音模型附带高达26.2万个词元的词汇表，无论用户需要何种语言，每个用户都要下载这些词元所索引的表格（输入嵌入、未绑定的输出头，以及近年设备端模型中的逐层嵌入）。我们将部署中从未使用的词汇表字节称为“词汇税”，并在无需下载权重的情况下，对Hugging Face Hub上1,244个生成式和语音识别模型进行了测量。对于泰语和英语部署场景，该税收在参数量低于0.5B的模型中占模型字节的中位数4.4%，在参数量超过8B的模型中占2.1%，在Gemma 4 E2B中高达1.93 GB：这一负担最沉重地落在边缘设备所运行的小模型上。仅保留部署所用书写系统对应的词元行是精确无损的做法；而保留用户自身文本中出现过的词元则能移除更多内容（参数量低于0.5B的模型中位数为29.0%），但通常的做法会破坏BPE分词器：它会丢弃合并操作所依赖的中间词元，导致新文本……（原文摘要至此被截断）

    arXiv:2611.19913v1 Announce Type: cross  Abstract: Open large language models (LLMs) and speech models ship vocabularies of up to 262k tokens, and every user downloads the tables those tokens index (the input embedding, an untied output head and, in recent on-device models, per-layer embeddings), whatever languages they need. We call the bytes in rows a deployment never uses the vocabulary tax and measure it, without downloading weights, for 1,244 generative and speech-recognition models on the Hugging Face Hub. For a Thai and English deployment it is a median 4.4% of model bytes below 0.5B parameters and 2.1% above 8B, and 1.93 GB in Gemma 4 E2B: it falls hardest on the small models that edge devices run. Keeping only the rows of the deployment's writing systems is exact. Keeping the tokens seen in the user's own text removes much more, a median 29.0% below 0.5B, but the usual way of doing it breaks BPE tokenizers: it drops the intermediate tokens that merges build on, so new text is 
    
[^108]: 归因或拒答：在低资源非洲商业网络上评估有证据支撑的检索

    Attribute or Abstain: Evaluating Evidence-Grounded Retrieval on the Low-Resource African Business Web

    [https://arxiv.org/abs/2611.19900](https://arxiv.org/abs/2611.19900)

    该论文提出AfriBizClaims基准，首次系统评估检索增强系统在证据稀缺的非洲低资源商业网络上的证据归因与拒答能力，发现简单的BM25和标准RAG基线在归因上表现出色（98%论断有据可依）并能正确拒答虚构公司查询。

    

    检索增强式企业研究系统通常以其检索到的信息量来评判，然而在公开证据稀缺的市场中，一个缺乏依据却言之凿凿的回答，其代价远大于不作回答。我们提出了AfriBizClaims，这是一个涵盖六个非洲市场、以英语、法语和葡萄牙语提出的150个企业研究查询组成的基准，其中包含关于虚构公司的查询，正确的系统必须对此拒答。系统做出的每一条论断都由一个AI评判器对照其所引用的页面进行评判，再由来自另一个模型家族的第二个AI评判器以及两名人类标注员进行复核，弃答行为则依据可回答性标签进行评分。我们比较了在合并语料库上运行的BM25、标准的网页检索增强生成（RAG），以及一个多步骤智能体流水线，三者均使用相同的语言模型。两个简单的基线方法归因表现良好：它们被判定的论断中有98%获得了所引用页面的支持，并且它们对所有虚构公司相关查询均予以拒答。

    arXiv:2611.19900v1 Announce Type: new  Abstract: Retrieval-augmented systems that research companies are judged by how much they find, yet in markets where public evidence is sparse, a confident answer without support costs more than no answer. We introduce AfriBizClaims, a benchmark of 150 company-research queries across six African markets in English, French and Portuguese, including queries about fictitious companies that a correct system must decline. Each claim a system makes is judged against the page it cites by an AI judge, checked against a second AI judge from another model family and against two human annotators, and abstentions are scored against answerability labels. We compare BM25 over a pooled corpus, standard web retrieval-augmented generation, and a multi-step agent pipeline (Konexium V1), all using the same language model. The two simple baselines are well attributed: 98% of their judged claims are supported by the page they cite, and they decline every field of ever
    
[^109]: LoopSkim：面向循环语言模型的无损多深度投机解码

    LoopSkim: Lossless Multi-Depth Speculative Decoding for Looped Language Models

    [https://arxiv.org/abs/2611.19896](https://arxiv.org/abs/2611.19896)

    LoopSkim提出了一种面向循环语言模型的无损多深度投机解码方法，让流水线的每个深度直接验证从上下文复制的草稿token而无需额外前向传播，在保持解码无损的前提下显著加速。

    

    诸如Ouro和Raven之类的循环语言模型在每个token上多次应用同一权重共享的块，因此自回归解码（AR）需要按顺序运行每个token的每一层循环。流水线式自投机解码器通过从浅层读出生成草稿token并在全深度进行验证来重叠这些循环。然而，循环语言模型的每一层循环都以对下一个token的读出结束，且草稿token无论如何都会经过每个深度，因此每个深度都可以在无需额外前向传播的情况下对其进行检查。我们提出了LoopSkim，其中流水线的深度本身就是验证器。从上下文复制的草稿token（如提示查找解码那样）在首轮循环之后进入，任何后续与草稿矛盾的读出都会在该深度将其替换，只有全深度的读出才会提交token。LoopSkim预先固定每个位置的随机抽样，因此每次检查只需比较两个token，并且在采样条件下，对于任意数量的检查和草稿，解码在精确算术下保持无损。

    arXiv:2611.19896v1 Announce Type: cross  Abstract: Looped language models such as Ouro and Raven apply one weight-shared block several times per token, so autoregressive decoding (AR) runs every loop of every token in sequence. Pipelined self-speculative decoders overlap these loops by drafting tokens from shallow readouts and verifying them at full depth. Yet every loop of a looped LM ends in a readout of the next token, and draft tokens pass through every depth anyway, so each depth can check them without an extra pass. We introduce LoopSkim, in which the depth of the pipeline is the verifier. Draft tokens copied from the context, as in prompt lookup decoding, enter after the first loops, any later readout that contradicts one replaces it at that depth, and only full-depth readouts commit tokens. LoopSkim fixes the random draws of every position in advance, so each check compares two tokens and decoding stays lossless in exact arithmetic under sampling for any number of checks and dr
    
[^110]: 谁来编写编码手册？一个用于审计LLM生成类别系统的八步框架

    Who Writes the Codebook? An Eight-Step Framework for Auditing LLM-Generated Category Systems

    [https://arxiv.org/abs/2611.19872](https://arxiv.org/abs/2611.19872)

    该论文提出八步定性差异审计框架，通过对比模型自有与共享参照编码手册，揭示大语言模型生成的定性编码类别存在显著模型间差异（跨模型一致性仅约.69），从而在实质解读前审计模型依赖性。

    

    大语言模型（LLM）日益承担起构建定性证据解读类别的任务，而现有编码基准测试通常将这些类别固定不变。我们提出定性差异审计框架，这是一种八步方法，可在受控输入条件下比较编码手册的创建、重复应用以及编码分布情况。该框架结合了模型自有的编码手册与共享参照编码手册，以区分整个流程中的差异与同一类别系统内应用层面的差异。将该框架应用于668份组织访谈、13,166个文本单元以及来自五家供应商的八个模型，共产生48套编码手册和270次编码运行。研究显示，各模型生成的类别结构差异显著：模型内部平均信度介于.76至.92之间，而模型间在共享编码手册上的一致性仅为约.69。该框架通过在实质性解读之前揭示模型依赖性，对LLM辅助的扎根理论方法形成了补充。

    arXiv:2611.19872v1 Announce Type: new  Abstract: Large language models (LLMs) increasingly create the categories through which qualitative evidence is interpreted. Existing coding benchmarks often hold these categories fixed. We propose the Qualitative Variance Audit Framework, an eight-step method that compares codebook creation, repeated application and code distributions under controlled inputs. It combines model-owned and shared-reference codebooks to distinguish variation in the overall pipeline from application differences within a common category system. An application to 668 organisational interviews, 13,166 text units and eight models from five vendors yields 48 codebooks and 270 coding runs. Models create markedly different category structures; mean within-model reliability ranges from .76 to .92, whereas between-model agreement on the shared codebook is about .69. The framework complements LLM-assisted grounded theory by exposing model dependencies before substantive interpr
    
[^111]: Bioformer-RetroMAE：面向检索的紧凑型生物医学语言模型预训练方法，用于文献搜索

    Bioformer-RetroMAE: retrieval-oriented pretraining of a compact biomedical language model for literature search

    [https://arxiv.org/abs/2611.19862](https://arxiv.org/abs/2611.19862)

    该论文提出了Bioformer-RetroMAE，一个仅4300万参数的紧凑型面向检索的生物医学编码器，通过RetroMAE方法在PubMed语料上持续预训练得到，在所有评估指标上超越了参数量约为其2.5倍的BioBERT和PubMedBERT双编码器。

    

    稠密检索在生物医学文献搜索和检索增强生成（RAG）中的应用日益增多。大多数生物医学双编码器建立在经过掩码语言建模（MLM）预训练的BERTBase规模编码器之上。这些编码器在语料库规模索引和延迟敏感的查询编码方面成本高昂，而且MLM并未显式地训练双编码器用来表示整个文本的[CLS]向量。我们提出了Bioformer-RetroMAE，这是一个紧凑型（4300万参数）的面向检索的生物医学编码器，通过在约3200万篇PubMed标题和摘要上使用RetroMAE对Bioformer8L进行持续预训练而获得。在使用BM25难负例进行双编码器微调后，我们在BioASQ任务12B测试问题上，通过从所有PubMed摘要中进行检索的方式对其进行了评估。Bioformer-RetroMAE在所有指标上都优于在相同设置下微调的BioBERT和PubMedBERT双编码器，尽管其参数量仅为后者的约40%，并且产生……

    arXiv:2611.19862v1 Announce Type: new  Abstract: Dense retrieval is increasingly used for biomedical literature search and retrieval-augmented generation (RAG). Most biomedical bi-encoders are built on BERTBase-sized encoders pretrained with masked language modeling (MLM). These encoders are costly for corpus-scale indexing and latency-sensitive query encoding, and MLM does not explicitly train the [CLS] vector that a bi-encoder uses to represent a whole text. We present Bioformer-RetroMAE, a compact (43M-parameter) retrieval-oriented biomedical encoder obtained by continually pretraining Bioformer8L} with RetroMAE on approximately 32 million PubMed titles and abstracts. After bi-encoder fine-tuning with BM25 hard negatives, we evaluated it on BioASQ Task 12B test questions by retrieving from all PubMed abstracts. Bioformer-RetroMAE outperformed BioBERT and PubMedBERT bi-encoders fine-tuned under identical settings on all metrics, despite having about 40% of their parameters and produc
    
[^112]: 弥合术语鸿沟：面向跨国企业的以数据为中心的多语言翻译对齐框架

    Bridging the Terminology Gap: A Data-Centric Alignment Framework for Multilingual Translation in Cross-Border Corporation

    [https://arxiv.org/abs/2611.19858](https://arxiv.org/abs/2611.19858)

    该论文提出以数据为中心的BTG框架，通过挖掘跨语言术语对齐并利用LLM合成术语丰富的平行语料，结合混合过滤策略与SFT加RL的两阶段训练，有效提升跨国企业场景下多语言翻译的术语准确性与模型可靠性。

    

    随着全球化加速了对无缝跨境企业协作的需求，多语言翻译仍然受到以下瓶颈的制约：领域特定术语不足、术语翻译僵化、指令遵循能力较弱，以及长指令场景下的高推理延迟。为了克服这些障碍，我们提出了弥合术语鸿沟框架，这是一个以数据为中心的框架。首先，BTG从领域文档中挖掘跨语言术语对齐，并利用大语言模型（LLM）合成富含术语的平行语料库。其次，采用结合启发式规则和XCOMET评分的混合过滤策略，确保了较高的数据保真度。第三，两阶段训练范式先采用监督微调（SFT）进行针对性的术语注入，随后通过强化学习（RL）优化输出偏好，有效平衡了领域知识整合与模型可靠性。广泛的离线评估（例如，XC……（摘要原文在此处截断）

    arXiv:2611.19858v1 Announce Type: cross  Abstract: As globalization accelerates the demand for seamless cross-border enterprise collaboration, multilingual translation remains bottlenecked by insufficient domain-specific terminology, rigid terminology translation, weak instruction-following capabilities, and high inference latency in long-instruction scenarios. To overcome these barriers, we propose Bridging the Terminology Gap (BTG), a data-centric framework. First, BTG mines cross-lingual term alignments from domain documents and leverages LLMs to synthesize terminology-rich parallel corpora. Second, a hybrid filtering strategy combining heuristics and XCOMET scoring ensures high data fidelity. Third, a two-stage training paradigm employs supervised fine-tuning (SFT) for targeted terminology injection, followed by reinforcement learning (RL) to refine output preferences, effectively balancing domain knowledge integration with model reliability. Extensive offline evaluations (e.g., XC
    
[^113]: NoLBERT-Calls：无后见之明地解读财报电话会议

    NoLBERT-Calls: Reading Earnings Calls Without Hindsight

    [https://arxiv.org/abs/2611.19849](https://arxiv.org/abs/2611.19849)

    该论文提出NoLBERT-Calls-2023，通过仅使用2023年底之前的知识并利用财报电话会议自身的问答对进行自监督适配，消除了语言模型财务度量中的后见之明偏差，在254,575个问题上以78.0%的准确率正确匹配管理层回答。

    

    语言模型如今已成为衡量管理层在财报电话会议上所言内容的标准工具，但这些度量所依赖的编码器通常是在与回测覆盖年份相同的文本上进行预训练的。即使保留出测试期，也无法消除这种后见之明，因为它已嵌入特征之中。我们调整了NoLBERT——一个仅在1976年至1995年间发表的文本上预训练的编码器——使其适配财报电话会议，且不让它阅读任何2023年12月31日之后的文本。电话会议本身提供训练信号：每个分析师提问之后都紧随管理层的回答，模型学习从同一场电话会议的其他回答中挑选出正确的那个回答。我们在2025年1月至2026年10月的所有美国电话会议（20,323场电话会议，254,575个问题）上进行测试，并在2024年的电话会议上做出所有建模选择。适配后的模型NoLBERT-Calls-2023在78.0%的问题上将正确回答排在第一位（95%区间为77.8-78.2；随机概率为8.0%），而适配前仅为26.3%……

    arXiv:2611.19849v1 Announce Type: new  Abstract: Language models are now a standard tool for measuring what managers say on earnings calls, but the encoders behind these measures were usually pretrained on text from the same years a backtest covers. Holding out a test period does not remove that hindsight, because it sits in the features. We adapt NoLBERT, an encoder pretrained only on text published between 1976 and 1995, to earnings calls without letting it read anything dated after 31 December 2023. The calls supply the training signal themselves: every analyst question is followed by management's answer, and the model learns to pick that answer out from the other answers of the same call. We test on every US call from January 2025 to October 2026 (20,323 calls, 254,575 questions) and make every modeling choice on calls from 2024. The adapted model, NoLBERT-Calls-2023, ranks the true answer first for 78.0% of questions (95% interval 77.8-78.2; chance 8.0%), against 26.3% before adap
    
[^114]: 语言模型训练与推理的相图

    A phase diagram for training and inference in language models

    [https://arxiv.org/abs/2611.19848](https://arxiv.org/abs/2611.19848)

    本文用文档可区分性和文档维度两个集体坐标构建了语言模型表示几何的相图，发现未训练模型的推理流趋向坍缩或噪声区域，而训练后模型的推理流稳定驻留在相图中心区域。

    

    语言模型是一个复杂系统，其行为由Transformer和语料库共同决定，然而Transformer中的表示几何与语言模型实际行为之间的关系仍是未解之谜。我们分析了实际语言模型在前向传播过程中表示几何的变化，并用两个集体坐标——文档可区分性和文档维度——来刻画文档在给定层的表示几何。在这些集体坐标张成的相图上，推理表现为沿深度的流动。我们发现，对于未训练的模型，取决于其初始化尺度，156条推理流中的大多数都朝向相图的坍缩区和噪声区移动。相比之下，来自5个架构家族的8个训练后模型的推理流均进入并驻留在相图的中心区域。

    arXiv:2611.19848v1 Announce Type: cross  Abstract: A language model is a complex system whose behavior is determined jointly by the Transformer and the corpus, yet how the representation geometry in the Transformer relates to the actual behavior of language models remains open. We analyze how the representation geometry changes along the forward propagation of actual language models and characterize the representation geometry of a document at a given layer by two collective coordinates, document distinguishability and document dimension. On the phase diagram spanned by these collective coordinates, inference is represented as a flow along depth. We find that, for untrained models, depending on their initialization scale, most of the 156 inference flows move toward the collapse and noise regions of the phase diagram. By contrast, the inference flows of 8 trained models from 5 architecture families all enter and reside in the central region of the phase diagram. Describing the evolution
    
[^115]: 将语言模型与语音结合并不能消除“AI味”：校准的LLM评判器、古德哈特陷阱与一种基于偏好调优的修复方案

    Grounding Language Models in Speech Does Not Fix "AI Flavor": Calibrated LLM Judges, Goodhart Traps, and a Preference-Tuning Fix

    [https://arxiv.org/abs/2611.19840](https://arxiv.org/abs/2611.19840)

    该论文通过在0.6B模型上的受控实验证明，注入语音或韵律信号的三种方法（音素交错续训、韵律计划条件生成、基于TTS韵律距离的规则奖励）均无法消除LLM中文输出的“AI味”，并指出当前n≈20的单评分员盲评只能检测极端偏好的评估方法是根本症结，同时提出一种基于偏好调优的修复方案。

    

    大语言模型的中文输出带有可感知的“AI味”：模板化的句式框架、堆叠的书面语连接词、重复循环以及排版泄漏。一种流行的假设将其归咎于纯文本训练缺失了人类写作中的“内心语音”层，因此注入语音或韵律信号应当能减少这种味道。我们在一个0.6B参数模型上通过受控干预对这一假设进行了测试，三种方法均告失败：(i) 音素交错续训在文本匹配对照下没有带来口语化提升，反而增加了重复度（+0.13 [+0.03, +0.23]）；(ii) 韵律计划条件生成的遵循度仅达到0.257，远低于预先注册的0.50门槛；(iii) 基于TTS韵律特征距离的规则奖励在人类（8/16）和校准后的LLM评判器（9/16）的双盲评估中均无法被感知。其根本原因在于评估环节：样本量约20的单评分员盲评只能检测出极端（≥85%）的偏好差异。

    arXiv:2611.19840v1 Announce Type: new  Abstract: Chinese outputs of large language models carry a perceptible "AI flavor": formulaic sentence frames, stacked written-register connectives, repetition loops, and layout leakage. A popular hypothesis blames text-only training for the missing inner-speech layer of human writing; injecting speech or prosody signals should therefore reduce it. We test it with controlled interventions on a 0.6B model; all three fail: (i) phoneme-interleaved continued pretraining yields no spokenness gain under text-matched controls while raising repetition (+0.13 [+0.03, +0.23]); (ii) prosody-plan conditional generation reaches an adherence of only 0.257 against a pre-registered 0.50 gate; (iii) a rule-based reward built on TTS prosody-feature distance is imperceptible in double-blind evaluation by both humans (8/16) and a calibrated LLM judge (9/16).   A root cause is evaluation: single-rater blind evaluation at n~20 only detects extreme (>=85%) preferences. 
    
[^116]: EfficientSkill：面向Token高效多轮智能体的技能学习

    EfficientSkill: Learning Skills for Token-Efficient Multi-Turn Agents

    [https://arxiv.org/abs/2611.19823](https://arxiv.org/abs/2611.19823)

    EfficientSkill提出“先无惩罚发现可靠技能、再进行精度保护剪枝”的两步框架，通过压缩指令、限制回合数和卸载可复用代码，在提升多轮LLM智能体准确率的同时最多减少80%的输出token消耗。

    

    反思式技能优化器通过从失败和成功的回合中修订指令来改进多轮LLM智能体，然而不断累积的边缘情况规则会使提示词臃肿、推高token消耗，并使智能体陷入漫长且易出错的循环。我们提出EfficientSkill，这是一个两步框架：第一步在不施加token惩罚的情况下发现可靠技能；第二步从该技能热启动进行精度保护的剪枝，以压缩指令、限制智能体回合数，并将可复用代码卸载为工作区工具，且零实现提示成本。在四个前沿、中端及开源权重模型上，针对MATH-500、ALFWorld和SpreadsheetBench三个基准，EfficientSkill在12个配置中的11个上相比无约束的EvoSkill减少了输出token（最高减少80%），同时在所有基准上都提升了平均准确率，其中包括通过用紧凑工作流替代停滞循环，在SpreadsheetBench上使用Gemini 3.8 Flash和Grok 4.1 Fast分别取得+14.3和+17.5个百分点的准确率提升。

    arXiv:2611.19823v1 Announce Type: cross  Abstract: Reflective skill optimizers improve multi-turn LLM agents by revising instructions from failed and successful rollouts, yet accumulating edge-case rules bloats prompts, inflates token use, and pushes agents into long, error-prone loops. We introduce EfficientSkill, a two-step framework that first discovers a reliable skill without token penalties, and then warm-starts accuracy-guarded pruning from that skill to compress instructions, bound agent turns, and offload reusable code into workspace utilities with zero implementation prompt cost. Across four frontier, mid-tier, and open-weight models on MATH-500, ALFWorld, and SpreadsheetBench, EfficientSkill cuts output tokens over unconstrained EvoSkill on 11 of 12 configurations (by up to 80%) while raising mean accuracy on every benchmark, including +14.3 and +17.5 percentage-point gains on SpreadsheetBench with Gemini 3.8 Flash and Grok 4.1 Fast by replacing stalled loops with compact wo
    
[^117]: 机器口译的竞争力：英汉同声传译中意义与译语质量的语篇敏感性比较

    Machine interpreting competitiveness: a discourse-sensitive comparison of meaning and target-language quality in English-Chinese simultaneous interpreting

    [https://arxiv.org/abs/2611.19802](https://arxiv.org/abs/2611.19802)

    本研究采用语篇敏感的评估框架，将五个商用机器口译系统与26名人类译员在英汉同声传译中的意义传达与译语质量进行比较，发现机器口译在计划性正式语篇中已进入人类译员较高水平区间，但在即兴语篇中表现差异显著。

    

    机器口译（MI），即人工智能驱动的语音翻译，在专业及面向公众的交流中日益常见，但其相对于人类同声传译员的表现仍缺乏充分认识。现有研究往往受限于单一的语篇模式，且评估框架与口译作为一种语篇层面交际活动的特点契合度不足。本研究将五个商用机器口译系统与26名人类口译员（14名职业译员、12名高年级学生译员）在英译汉同声传译任务上的表现进行比较，涵盖计划性与非计划性语篇，重点关注意义传达与译语质量。在这些材料中，机器口译在两种语篇模式下的表现差异显著。在计划性的正式演讲中，表现最强的系统在意义传达方面极具竞争力，进入了人类译员的较高水平区间，另一系统也表现强劲。在即兴访谈中……

    arXiv:2611.19802v1 Announce Type: new  Abstract: Machine interpreting (MI), or AI-enabled speech translation, is increasingly visible in professional and public-facing communication, yet its performance relative to human simultaneous interpreters remains insufficiently understood. Existing studies are often limited by a single discourse mode and evaluation frameworks that are not well aligned with interpreting as a discourse-level communicative activity. The present study compares five commercial MI systems with 26 human interpreters (14 professionals, 12 advanced students) on English-to-Chinese simultaneous interpreting across planned and unplanned discourse, focusing on meaning and target-language quality. In these materials, MI performance differed sharply between the two discourse modes. In the planned formal speech, the strongest system became highly competitive on meaning, entering the upper human range, and a second system also performed strongly. In the extemporaneous interview
    
[^118]: 多分支Transformer

    Multi-Branch Transformer

    [https://arxiv.org/abs/2611.19800](https://arxiv.org/abs/2611.19800)

    提出多分支Transformer（MBT），通过并行组合完整Transformer模块的分支化路径来扩展模型，在匹配的参数量和训练计算量预算下始终优于传统的顺序堆叠Transformer。

    

    Transformer语言模型通常通过加宽模块或堆叠更深层数来进行扩展。在这项工作中，我们探索了一条互补的路径：分支化，即完整Transformer模块的并行组合。我们提出了多分支Transformer（MBT），它通过共享的初始层处理输入，并行执行独立参数化的多个顺序模块分支，并在共享的最终层之前融合各分支的输出。在训练计算量相当的条件下，我们研究了三种MBT变体：宽-浅变体保持模块宽度不变，窄-深变体保持顺序深度不变，平衡变体则在两者之间进行折中。在四个专家混合规模的受控预训练实验中（总参数量从0.8B到2.1B），2分支的宽-浅MBT在匹配的参数量和训练计算量预算下始终优于顺序Transformer。平衡MBT的性能进一步提升，并且随着分支数量的增加仍保持相当的质量水平。

    arXiv:2611.19800v1 Announce Type: cross  Abstract: Transformer language models are commonly scaled by making their blocks wider or stacking them deeper. In this work, we explore a complementary path: branching, the parallel composition of full Transformer blocks. We introduce the Multi-Branch Transformer (MBT), which processes the input through shared initial layers, executes independently parameterized branches of sequential blocks in parallel, and fuses their outputs before shared final layers. At comparable training compute, we study three MBT variants: Wide-Shallow preserves block width, Narrow-Deep preserves sequential depth, and Balanced interpolates between them. In controlled pretraining experiments at four mixture-of-experts scales (0.8B to 2.1B total parameters), 2-branch Wide-Shallow MBTs consistently outperform sequential Transformers under matched parameter and training-compute budgets. Balanced MBTs improve further and retain comparable quality as the number of branches i
    
[^119]: 通过未来研究方向选择激发大语言模型的研究品味

    Eliciting Research Taste in LLMs through Future Research Direction Choice

    [https://arxiv.org/abs/2611.19794](https://arxiv.org/abs/2611.19794)

    该论文将“研究品味”操作化为大语言模型在给定研究目标与当前结果下选择未来研究方向的判断任务，并基于19篇ML论文构建80个顺序控制的成对比较评估，以作者一致性、稳定分歧和不稳定选择作为模型研究方向偏好的行为测量指标。

    

    研究品味在AI科学家的讨论中日益受到关注，但在大语言模型（LLM）中却鲜有研究或测量。这种判断力在开放式研究中尤为重要，因为在存在可验证目标之前，就必须先选定一个研究方向。我们通过科学工作流中的一项任务来操作化研究品味：在给定研究目标和当前结果的前提下，选择应追求哪个未来研究方向。我们既不预测下游的科学计量指标，也不将作者的选择视为客观正确，而是将模型与人类作者撰写的未来工作方向进行对比评估，并报告作者一致性、稳定分歧和不稳定选择作为研究方向偏好的行为化测量指标。我们构建了一个顺序控制的评估，涵盖2025年12月至2026年2月发表的19篇机器学习论文中的80个成对比较，横跨四个子领域。每个查询呈现一个研究目标、经k检验的假设……（注：原摘要在此处截断）

    arXiv:2611.19794v1 Announce Type: cross  Abstract: Research taste is increasingly discussed in the context of AI scientists, yet remains little studied or measured in LLMs. This judgment matters especially in open-ended research, where a direction must be chosen before a verifiable objective is available. We operationalize research taste through one task of the scientific workflow: choosing which future research direction to pursue given a research goal and current results. Rather than predicting downstream scientometric metrics or treating author choices as objectively correct, we evaluate models against human-authored future-work directions and report author alignment, stable divergence, and unstable choice as behavioral measures of research-direction preference. We construct an order-controlled evaluation of 80 pairwise comparisons from 19 ML papers published between December 2025 and February 2026 across four subdomains. Each query presents a research goal, hypotheses tested with k
    
[^120]: 逻辑AI操作系统：模型之外的免训练指令遵循治理层——收益、成本、失败及测量工具本身的校准

    Logical AI OS: A Training-Free Governance Layer Outside the Model for Instruction Following -- Benefits, Costs, Failures, and the Calibration of the Measuring Instrument Itself

    [https://arxiv.org/abs/2611.19767](https://arxiv.org/abs/2611.19767)

    该论文在模型外部部署了一个免训练的治理层来强制小语言模型遵循用户显式指令，显著提升了指令遵守率，并系统测量了其收益、成本、失败模式以及测量工具自身的校准过程。

    

    小语言模型经常无法满足用户的显式要求（长度、条目数量、格式、必需或禁止的词、语法人称、答案语言）。我们测量了一个置于模型之外的免训练治理层：它用规则读取请求，机械地检查答案，当请求未被遵守时进行修复，当修复失败时返回原始答案。该循环在先前的工作中已经存在；我们的贡献在于测量其收益、成本和失败情况，并报告测量工具本身是如何被校准的。在四个4位量化的Qwen系列模型（3B、7B、14B、一个7B推理模型）和329个中文提示词上：(1) 完全遵守该层所读取的每一项要求的答案显著增加（所有比较p<0.0001），在校正后的检查器下，329个提示词中增加34至66个，在长度控制修复后增加52至92个；通用的重做消息未带来显著收益。(2) 探索性结果：在90个触发修复的提示词上，两个AI标注者……（摘要在此处截断）

    arXiv:2611.19767v1 Announce Type: new  Abstract: Small language models often fail explicit user requests (length, number of items, format, required or forbidden words, grammatical person, answer language). We measure a training-free governance layer placed outside the model: it reads the requests with rules, checks the answer mechanically, repairs it when a request is not kept, and hands back the original answer when the repair fails. The loop exists in prior work; our contribution is measuring its benefits, costs and failures, and reporting how the measuring instrument itself was calibrated. On four 4-bit Qwen-family models (3B, 7B, 14B, one 7B reasoning model) and 329 Chinese prompts: (1) answers that keep every request the layer read rise significantly (all comparisons p<0.0001), by 34 to 66 of 329 under the corrected checker and 52 to 92 after a length-control fix; a generic redo message brings no significant gain. (2) Exploratory: on 90 repair-triggered prompts, two AI annotators 
    
[^121]: 模糊样本有帮助，但全量数据更胜一筹：重新审视数据集制图法在抽取式对抗问答中的应用

    Ambiguous Examples Help, But Full Data Wins: Revisiting Dataset Cartography for Extractive Adversarial QA

    [https://arxiv.org/abs/2611.19763](https://arxiv.org/abs/2611.19763)

    本文在更严格的抽取式对抗问答设置下重新审视数据集制图法的结论，发现仅用33%模糊样本子集训练虽有帮助，但仍不敌全量数据训练在对抗性评估中的表现。

    

    数据集制图法（Dataset Cartography）认为，训练动态可以揭示数据集中对学习具有不同作用的区域：易学样本有助于优化过程，难学样本往往暴露标注问题，而模糊样本对分布外泛化的贡献最大。原始工作的一个引人注目的结论是，仅在33%模糊样本上训练的模型可以在多项分布外（OOD）评估中超越全量数据训练的模型，其中包括在Adversarial SQuAD上评估的SQuAD的QNLI重构版本。我们在一个更严格的设置下重新审视这一结论：开放式抽取式问答，即模型必须预测答案片段，而不是对“问题-句子”对进行分类。针对每个编码器系列，我们计算干净来源SQuAD的训练动态，分别在全量、随机、按易学排序、按模糊排序和按难学排序的子集上进行训练，并在仅包含对抗样本的AddSent和AddOneSent行上评估鲁棒性。按模糊排序选取的33%子集……（摘要内容在此处截断）

    arXiv:2611.19763v1 Announce Type: new  Abstract: Dataset cartography argues that training dynamics can reveal regions of a dataset that matter differently for learning: easy-to-learn examples aid optimization, hard-to-learn examples often expose annotation problems, and ambiguous examples contribute strongly to out-of-distribution generalization. A striking implication of the original work is that a model trained on only 33% ambiguous examples can outperform full-data training on several OOD evaluations, including a QNLI reformulation of SQuAD evaluated on Adversarial SQuAD. We revisit this claim in a stricter setting: open-ended extractive question answering, where models must predict answer spans rather than classify question-sentence pairs. For each encoder family, we compute clean-source SQuAD training dynamics, train on full, random, easy-ranked, ambiguous-ranked, and hard-ranked subsets, and evaluate robustness on adversarial-only AddSent and AddOneSent rows. Ambiguous-ranked 33%
    
[^122]: StaleBench：隔离智能体内存中的过期检索问题，并为修复方案定价

    StaleBench: Isolating Stale Retrieval in Agent Memory, and Pricing the Fix

    [https://arxiv.org/abs/2611.19752](https://arxiv.org/abs/2611.19752)

    StaleBench基准在892个真实更新事实上隔离了智能体内存中的“过期检索”失败问题，揭示现有检索指标可能对内存策略做出完全相反的排序，并提出新指标Contamination@5来正确评估和定价修复方案。

    

    一个随时间存储事实的智能体，最终会存储同一事实的多个版本。当被问及一个与具体数值无关的问题——“X现在效力于哪支球队？”——相似度检索器无法优先选择当前版本，因为每个版本都与查询中提到的实体相匹配。StaleBench在892个真实更新的事实上隔离了这种“过期检索”失败问题，其时效性标签来自数据源本身（通过TempLAMA获取的Wikidata有效期年份），而非来自检索分数。我们的核心发现是：检索指标可能对内存策略做出完全相反的排序——某个保留策略与其“保留一切”的对照策略在每个测试查询的排名第一处返回完全相同的陈述，但在不显示日期的情况下，两者在两个阅读器上的准确率分别相差29.5分和65.6分。我们新增的指标Contamination@5能够在current@1和current@5失效的情况下正确地对它们进行排序。撤除陈述所携带的词汇锚点，会使已部署的键控机制付出数十分的代价。

    arXiv:2611.19752v1 Announce Type: cross  Abstract: An agent that stores facts over time eventually stores several versions of the same fact. Asked a value-agnostic question - "which team does X play for now?" - a similarity retriever cannot prefer the current version, because every version matches the entity the query names. StaleBench isolates this stale-retrieval failure on 892 real updating facts, with currency labels from the source (Wikidata validity years via TempLAMA), never from a retrieval score. Our central result is that retrieval metrics can rank memory policies backwards: a retention policy and its retain-everything control return the identical statement at rank 1 for every test query, yet differ by 29.5 and 65.6 points in reader accuracy on two readers when no date is shown. Contamination@5, the metric we add, ranks them correctly where current@1 and current@5 do not. Withdrawing the lexical anchor that statements carry costs deployed keying mechanisms tens of points of f
    
[^123]: 大语言模型智能体应该在测试时于何处进行适应？在上下文中、在权重中，还是两者兼有

    Where Should LLM Agents Adapt at Test Time? In Context, in Weights, or Both

    [https://arxiv.org/abs/2611.19700](https://arxiv.org/abs/2611.19700)

    该论文通过统一的在线协议系统性地比较了上下文学习、强化学习和自蒸馏三种测试时适应机制及其组合，揭示了上下文和权重这两个适应渠道各自的作用及其互补条件。

    

    在同一环境（如代码仓库、数据库或应用程序套件）中反复运行的大语言模型（LLM）智能体，会遇到共享潜在结构的任务。经验可以揭示这种结构，使智能体能够复用环境知识并从环境反馈中学习。智能体可以通过两个渠道在测试时利用这些经验进行适应：将经验保留在上下文中，或更新自身权重。这两个渠道通常被分开研究，导致人们不清楚各自的贡献是什么，以及何时能够互补。我们通过共享的在线协议和模块化框架，比较了上下文学习（ICL）、使用GRPO的基于可验证奖励的强化学习以及使用SDPO的自蒸馏，包括单独使用和组合使用。智能体遇到匹配的任务流，在适应每个任务之前先对其进行评估，并且仅从自身的尝试和环境反馈中学习。我们评估了14种适应机制以及一个……

    arXiv:2611.19700v1 Announce Type: cross  Abstract: LLM agents operating repeatedly in the same environment, such as a code repository, a database, or an application suite, encounter tasks that share latent structure. Experience can reveal this structure, allowing agents to reuse environment knowledge and learn from environmental feedback. Agents can adapt at test-time using this experience through two channels: retaining it in context and updating their weights. These channels are usually studied separately, leaving unclear what each contributes and when they complement each other. We compare in-context learning (ICL), reinforcement learning with verifiable rewards using GRPO, and self-distillation using SDPO, alone and in combination, using a shared online protocol and modular framework. Agents encounter matched task streams, are evaluated on each task before adapting to it, and learn only from their own attempts and environment feedback. We evaluate 14 adaptation mechanisms and a sta
    
[^124]: 数据归因应当引导预训练

    Data Attribution Should Steer Pretraining

    [https://arxiv.org/abs/2611.19698](https://arxiv.org/abs/2611.19698)

    本文提出训练数据归因不应只服务于单次训练的文档排序，而应通过发现可在跨语料库、跨检查点检验的可复用数据属性（筛选标准）来持续引导预训练数据筛选，并为此设计了带证据规范（针对五类论断）的闭环流程。

    

    预训练数据筛选决定了模型从什么数据中学习，然而一个文档是否值得学习取决于模型已经掌握了什么。基于影响力的选择器已经能够对文档排序以改进单次训练；我们主张训练数据归因应当通过识别可检验的数据属性来引导预训练，这些属性可以转化为可复用的筛选标准，并在总成本（包括发现成本）对齐的条件下，在新数据上与更廉价的选择器进行评估比较。文档排序只会被为其计算的那一次训练所消耗，而一个被解释清楚的数据属性则可以在不同语料库和模型检查点上进行检验，尤其是在人们熟悉的质量信号与实测有用性相背离的场景中。由于历史贡献、可理解的解释以及未来的数据价值各自需要不同的证据，我们规定了一个反馈回路：该回路解释归因来源、测试候选的数据变更，并针对五种论断类型说明每种论断分别需要什么样的证据以及什么样的……

    arXiv:2611.19698v1 Announce Type: cross  Abstract: Pretraining data curation decides what a model learns from, yet what makes a document worth learning from depends on what the model already knows. Influence-driven selectors already rank documents to improve one run; we argue that training-data attribution should steer pretraining by identifying testable data properties that become reusable curation criteria, evaluated on new data against cheaper selectors at matched total cost, discovery included. A document ranking is consumed by the run it was computed for, whereas an explained data property can be tested across corpora and checkpoints, especially where familiar quality signals diverge from measured usefulness. Because historical contribution, an understandable explanation, and future data value require distinct evidence, we specify a feedback loop that interprets attributed sources, tests candidate data changes, and states for five claim types what evidence each requires and what n
    
[^125]: MathaMAGICal：在能力获取过程中验证轨迹数据归因

    MathaMAGICal: Validating Trajectory Data Attribution During Capability Acquisition

    [https://arxiv.org/abs/2611.19692](https://arxiv.org/abs/2611.19692)

    该研究发现基于MAGIC的短检查点窗口轨迹数据归因分数能在两个模型规模上准确预测重新训练结果（LDS达0.86-0.99），而完整运行归因存在数值溢出且接近随机水平，且归因排名无法跨随机种子或超参数迁移。

    

    基于影响函数的方法（如MAGIC）为轨迹数据归因提供精确的一阶分数，但只有重新训练才能表明这些分数能否准确预测移除数据的效果。我们在8600万和1.52亿参数的GPT-2风格模型上评估了MAGIC，这些模型在过滤后的网络文本和四个合成数学数据集上从零开始学习多步算术。在8600万参数规模下，完整训练过程的归因存在数值溢出问题，即使采用元梯度裁剪，分数与重新训练结果的相关性也接近随机水平（线性数据建模分数LDS为0.17-0.22）。相比之下，短检查点窗口在两个模型规模上都能产生稳定且准确的预测（在1.52亿参数规模下的四个100步窗口中LDS为0.86-0.99），而单检查点梯度点积基线则显示出接近于零的相关性。单个训练块的排名仅在单次运行内能够预测重新训练结果，且无法在不同随机种子或超参数之间迁移（ρ ≤ 0.09）……

    arXiv:2611.19692v1 Announce Type: cross  Abstract: Influence-function-based methods like MAGIC provide exact first-order scores for trajectory data attribution, yet only retraining indicates whether those scores accurately predict data removal. We evaluate MAGIC on $86$M- and $152$M-parameter GPT-2-style models learning multistep arithmetic from scratch on filtered web text and four synthetic math families. Full-run attribution at $86$M suffers from numerical overflow, and even with metagradient clipping, scores correlate near chance with retraining (linear datamodeling score, LDS, $0.17$--$0.22$). In contrast, short checkpoint windows yield stable, accurate predictions at both scales (LDS $0.86$--$0.99$ across four $100$-step windows at $152$M), whereas a single-checkpoint gradient-dot baseline shows near-zero correlation. Rankings of individual training blocks predict retraining only within that single run and do not transfer across seeds or hyperparameters ($\rho \le 0.09$), whereas
    
[^126]: SG-LegalClarify：一个面向新加坡法律的法律推理与歧义识别大规模基准

    SG-LegalClarify: A Large-Scale Benchmark for Legal Reasoning and Ambiguity Recognition in Singapore Law

    [https://arxiv.org/abs/2611.19643](https://arxiv.org/abs/2611.19643)

    该论文提出了SG-LegalClarify——首个针对新加坡法律的大规模基准，包含1,100个律师撰写的问题，用于评估大语言模型在开放式法律推理中区分本地法理与普通法直觉以及识别非专业法律查询中信息缺失的能力。

    

    新加坡为大语言模型（LLM）的法律推理提供了一个独特的试验场。其法律体系融合了继承自英国的普通法与本地特有的成文法，要求模型能够区分本地法理与一般性的普通法直觉。这一背景还凸显了识别非专业用户法律查询中信息不完整问题的挑战，因为对法律结果至关重要的案件事实取决于所适用的法理。然而，现有的法律基准很少涵盖新加坡法律，也很少测试这种歧义识别能力。为弥补这些空白，我们推出了 SG-LegalClarify，一个面向新加坡法律开放式法律推理与歧义识别的大规模基准。SG-LegalClarify 包含 1,100 个由律师撰写的源问题，涵盖 10 个法律领域，其中包括 340 个信息完全明确的问题和 760 个信息不完整的问题。每个信息不完整的问题都有固定的隐藏事实设定以及一个信息完全明确的对应版本……

    arXiv:2611.19643v1 Announce Type: cross  Abstract: Singapore offers a distinctive testbed for legal reasoning by large language models (LLMs). Its legal system combines inherited English common law with jurisdiction-specific statutes, requiring models to distinguish local doctrine from generic common-law intuition. This setting also highlights the challenge of recognizing underspecification in lay legal queries, because the facts material to a legal outcome depend on the governing doctrine. Yet legal benchmarks rarely cover Singapore law or test this form of ambiguity recognition. To address these gaps, we introduce SG-LegalClarify, a large-scale benchmark for open-ended legal reasoning and ambiguity recognition in Singapore law. SG-LegalClarify contains 1,100 lawyer-authored source questions spanning 10 legal domains, including 340 fully specified and 760 underspecified questions. Each underspecified question has a fixed hidden-fact assignment and a fully specified counterpart, yieldi
    
[^127]: ROUNDS：多轮临床证据之后基于来源框架的诊断修正

    ROUNDS: Source-Framed Diagnostic Revision after Multi-Turn Clinical Evidence

    [https://arxiv.org/abs/2611.19636](https://arxiv.org/abs/2611.19636)

    本文提出ROUNDS框架，通过多轮临床证据后有/无答案选项的诊断对比、不同来源（临床医生vs外行）表述框架的误导诊断以及会诊小组意见变化，系统评估大语言模型在真实多轮诊断对话中形成与修正诊断的能力，发现无答案选项时模型准确率从76.9%显著降至57.0%。

    

    大型语言模型（LLMs）在医学基准测试中获得了高分，但这种性能能否转化为随着对话展开的可靠诊断？在诊断对话中，用户可以跨多个轮次提供临床发现，可以在不提供答案选项的情况下要求诊断，并可能引入他人的意见。因此，模型必须根据现有证据形成诊断，并决定如何回应随后的建议。我们提出了ROUNDS，一个在经过固定序列的多轮临床证据之后评估这些能力的框架。ROUNDS比较有和没有答案选项时的诊断表现，以临床医生和外行两种不同表述框架呈现相同的不正确诊断，并在书面形式的五名临床医生会诊小组中改变对相互竞争诊断的支持程度。在来自五个数据集的七个模型、共1,035个病例上，有答案选项时的平均准确率为76.9%，而没有答案选项时仅为57.0%。临床医生框架下的误导性……（摘要原文在此处截断）

    arXiv:2611.19636v1 Announce Type: cross  Abstract: Large language models (LLMs) achieve high scores on medical benchmarks, but does this performance translate into reliable diagnosis as a conversation unfolds? In diagnostic conversations, users can provide clinical findings across multiple turns, ask for a diagnosis without supplying answer options, and introduce opinions from others. Models must therefore form diagnoses from the available evidence and decide how to respond to subsequent advice. We introduce ROUNDS, a framework that evaluates these abilities after a fixed sequence of clinical evidence delivered over multiple turns. ROUNDS compares diagnosis with and without answer options, presents the same incorrect diagnosis under clinician and lay framing, and varies support for competing diagnoses in written five-clinician panels. Across seven models and 1,035 cases from five datasets, average accuracy was 76.9% with answer options and 57.0% without them. Clinician-framed misleadin
    
[^128]: SemMask：面向扩散语言模型的语义优先渐进式掩码策略

    SemMask: Semantic-Prioritized Progressive Masking for Diffusion Language Models

    [https://arxiv.org/abs/2611.19560](https://arxiv.org/abs/2611.19560)

    提出SemMask微调范式，通过基于梯度识别语义token并进行语义优先的渐进式掩码，促使扩散语言模型优先解码高信息语义token，从而克服伪自回归与虚词优先偏差，增强全局规划与并行生成能力。

    

    扩散语言模型（DLMs）旨在实现全局并行生成。然而在实践中，它们往往表现出伪自回归行为和虚词优先的偏差，这削弱了其在文本写作和代码生成等受益于全局规划与并行解码的任务上的性能。为解决这一问题，我们提出了SemMask，这是一种新颖的微调范式，利用语义优先的渐进式掩码策略，鼓励模型尽早且全局地解码高信息量的语义token，从而增强其全局规划和并行生成能力。具体而言，我们首先利用原始DLM相对于每个token的梯度来识别语义token；在掩码过程中，这些语义token被优先保留，从而建立起一种渐进式生成顺序，使语义token先于虚词token被解码；同时，我们相应地调整了损失函数设计。

    arXiv:2611.19560v1 Announce Type: cross  Abstract: Diffusion Language Models (DLMs) are designed to enable globally parallel generation. However, in practice, they often exhibit pseudo-autoregressive behavior and a function-word-first bias, which undermines their performance on tasks such as text writing and code generation that benefit from global planning and parallel decoding. To address this, we propose SemMask, a novel fine-tuning paradigm that leverages a semantic-prioritized progressive masking strategy to encourage the model to decode high-information semantic tokens early and globally, thereby enhancing its global planning and parallel generation capabilities. Specifically, we first use the gradients of the original DLM with respect to each token to identify semantic tokens. During masking, these semantic tokens are preserved with priority, establishing a progressive generation order in which semantic tokens are decoded prior to function tokens. Meanwhile, we adjust the loss w
    
[^129]: 对象化而非删除：基于临床医生投诉的AI环境式临床记录评估

    Objectify, Don't Delete: Complaint-Derived Evaluation of Ambient Clinical Notes

    [https://arxiv.org/abs/2611.19525](https://arxiv.org/abs/2611.19525)

    该论文提出一个“对象化”步骤，将临床医生高频抱怨中捆绑的复合问题改写为一组范围明确、可判定的是非决策，从而保留那些原本会被可执行性过滤环节丢弃的投诉标准，并将其用于LLM评判的AI临床记录质量评估。

    

    评估AI生成的临床记录的质量是困难的：像ROUGE这样的标准指标衡量的是与参考记录之间的词汇相似度，而专家清单则编码了委员会认为重要的内容。这两者都无法捕捉临床医生真正抱怨的问题。近期的研究表明，评估标准可以从用户反馈中挖掘出来，并过滤至LLM评判者能够可靠执行的标准。我们认为，这正是大量价值流失的地方。那些未能通过可执行性检验的标准，恰恰是临床医生最常提出的标准，因为高频抱怨往往以多个问题捆绑在一起的形式陈述。任何是非判断都无法对这样的复合问题进行评分。我们引入了一个“对象化”步骤，将此类标准改写为一组范围明确、界限清晰的具体决策。我们通过一个评估流程展示了这一方法：该流程从投诉中挖掘标准、对其进行对象化，并部署LLM评判者对两者进行评估。

    arXiv:2611.19525v1 Announce Type: new  Abstract: Evaluating the quality of AI-generated clinical notes is hard: standard metrics like ROUGE measure lexical similarity to reference notes, while expert checklists encode what committees believe matters. Neither captures what clinicians actually complain about. Recent work has shown that evaluation criteria can be mined from user feedback and filtered to criteria that LLM judges can reliably enforce. We argue that this is where much of the value is lost. The criteria that fail enforceability are disproportionately the ones clinicians raise most often, because high-frequency complaints are stated as several questions glued together. No yes/no judgment can grade such composites as posed. We introduce an objectification step that rewrites such criteria into sets of scoped, determinate decisions. We demonstrate the approach through an evaluation pipeline that mines criteria from complaints, objectifies them, and deploys LLM judges against both
    
[^130]: Sada-ASR：一个生产级多方言阿拉伯语语音识别系统

    Sada-ASR: A Production Multidialectal Arabic Speech Recognition System

    [https://arxiv.org/abs/2611.19512](https://arxiv.org/abs/2611.19512)

    Sada-ASR 是一个部署于真实客服场景的生产级阿拉伯语多方言语音识别系统，在涵盖 MSA 和八种方言的 120 小时测试集上以 11.69% 的词错误率超越 Microsoft、Google、OpenAI 等十一家竞品，取得全场最低错误率，尤其在阿曼、埃及和沙特方言上优势最大。

    

    摘要：拨打阿拉伯语客服热线的来电者使用的是各自的方言，而语音识别系统却主要基于现代标准阿拉伯语（MSA）构建和评估。一个听不懂方言的识别器会在其他任何补救措施生效之前就使通话中断。我们提出 Sada-ASR，这是 EchoNova 生产级阿拉伯语语音助手中的语音识别系统。在一个包含 MSA 和八种方言的 120 小时测试集上，它在十二个系统中取得了最低的词错误率和字符错误率，对比的十一个系统来自 Microsoft、ElevenLabs、Google、OpenAI、Speechmatics、IFM、Cohere、NVIDIA 和 Audar：其词错误率为 11.69%，而最强对手 Microsoft 的 MAI-Transcribe-2 为 13.51%（在该系统固定的数字与文字书写规范不受影响的语音片段上领先 0.74 个百分点），在各方言变体上平均词错误率为 15.52%，对比 16.96%。其最大优势体现在阿曼方言（领先 4.30 个百分点）、埃及方言（3.78）和沙特方言（2.19）上；在叙利亚方言上持平，在黎巴嫩方言上略逊（24.96% 对比 23.05%），大多……（原文摘要在此处截断）

    arXiv:2611.19512v1 Announce Type: new  Abstract: Callers to Arabic customer-service lines speak their own dialect, while speech recognizers are built and evaluated mainly on Modern Standard Arabic (MSA). A recognizer that mishears the dialect breaks the call before anything else can help. We present Sada-ASR, the speech recognizer of EchoNova's production Arabic voice agent. On a 120-hour test set of MSA and eight dialects, it has the lowest word and character error rates of twelve systems, against eleven from Microsoft, ElevenLabs, Google, OpenAI, Speechmatics, IFM, Cohere, NVIDIA and Audar: 11.69% word error rate against 13.51% for the strongest, Microsoft's MAI-Transcribe-2 (a lead of 0.74 points on clips unaffected by that system's fixed number and script conventions), and 15.52% against 16.96% averaged over varieties. Its largest margins are on Omani (4.30 points), Egyptian (3.78) and Saudi (2.19) speech; it is level on Syrian and loses on Lebanese (24.96% against 23.05%), mostly 
    
[^131]: 形式语义学的形式化：形式图谱

    Formalizing Formal Semantics: The Formal Atlas

    [https://arxiv.org/abs/2611.19509](https://arxiv.org/abs/2611.19509)

    本文提出了基于证明助手的“形式图谱”系统，通过交互式应用将语义学文献与形式化证明连接起来，实现语义理论的形式化、预测检验与比较，并探索AI辅助开发与推理的应用前景。

    

    在本文中，我们提出了形式图谱，这是一个使用证明助手研究自然语言语义的系统。我们的目标是形式化语义分析、检验其预测并比较所实现的理论。为此，我们通过一个交互式应用程序将文献来源记录与形式化工作连接起来，研究人员可以在其中查看证明并修改受支持的 Rocq 文件。蒙塔古的 PTQ 与 Gallin 的著作为我们提供了主要案例，随后是动态语义与类型论语义的比较、RSA 以及分布式语义。我们还讨论了形式化那些不够显式的分析所需的条件。最后，我们探讨了人工智能的应用，既可用于辅助开发这些资源，也可用于能够利用这些资源的推理系统。这项工作扩展了早期证明助手语义学的研究计划，研究与教学是其互补的用途。

    arXiv:2611.19509v1 Announce Type: new  Abstract: In this paper we present the Formal Atlas, a system for the study of natural language semantics using proof assistants. Our aim is to formalize semantic accounts,   check their predictions and compare the implemented theories. To this end, we connect source records and formal developments with an interactive application, where   researchers can inspect proofs and modify supported Rocq files. Montague's PTQ and Gallin provide our main cases, followed by dynamic and type-theoretic   comparisons, RSA and distributional semantics. We also discuss what it takes to formalize less explicit analyses. Lastly, we consider the use of AI, both for   assistance in developing these resources and for inference systems that can make use of them. This work extends an earlier programme in proof-assistant semantics,   with research and teaching as complementary uses.
    
[^132]: 评估与改进自然语言自编码器

    Evaluating and Improving Natural Language Autoencoders

    [https://arxiv.org/abs/2611.19507](https://arxiv.org/abs/2611.19507)

    提出基于审计任务的自然语言自编码器评估方法，并通过支持多层多位置激活信息利用、更适用于长文本和减少虚假声明等扩展改进，显著提升了自然语言自编码器解释大语言模型激活的能力。

    

    自然语言自编码器（NLAs）是一种为大语言模型（LLM）激活生成自然语言解释的方法。我们基于审计任务开发了一系列评估方法，用以测试自然语言自编码器揭示潜在知识、识别模型是否被提示误导以及捕捉无声捏造（silent fabrication）的能力。我们还开发了自然语言自编码器的扩展方法，使其能够利用来自多个层和位置的激活信息，从而更有效地应用于长篇对话记录，并减少其做出的虚假声明数量。这些扩展在我们的审计评估以及改编自先前工作的辅助评估上都带来了性能提升。我们还利用这些评估研究了不同的审计能力在自然语言自编码器训练过程中的变化，发现尽管每项评估在训练初期都有所改善，但过度训练可能会损害其在某些审计评估上的表现。

    arXiv:2611.19507v1 Announce Type: new  Abstract: Natural language autoencoders (NLAs) are a method for generating natural language explanations of LLM activations. We develop evaluations based on auditing tasks that test the ability of NLAs to surface latent knowledge, identify whether a model was misled by a hint, and catch a silent fabrication. We also develop extensions to NLAs that let them use activation information from multiple layers and positions, let us apply them more effectively to long transcripts, and reduce the number of false claims they make. These extensions result in improvements on both our auditing evaluations and auxiliary evaluations adapted from prior work. We also use the evaluations to study how different auditing capabilities change over the course of NLA training, and find that although every evaluation improves early in training, extended training can hurt performance on some of our auditing evaluations.
    
[^133]: 探测与引导在语言模型内部及跨模型间是否偏好相同的方向？

    Do Probing and Steering Favor the Same Directions Within and Across Language Models?

    [https://arxiv.org/abs/2611.19499](https://arxiv.org/abs/2611.19499)

    该研究揭示了语言模型中最优探测方向与最优引导方向并不相同——引导始终偏好更接近均值差的方向，且这一发现在不同模型系列和几乎所有层中均成立，表明探测与引导是本质上不同的任务。

    

    语言模型隐藏状态中的某个方向可用于探测某个概念是否存在，或将模型引导至该概念。尽管探测方向常被复用于引导，但最能检测某个概念的方向未必是最能改变它的方向。我们在来自不同系列的三个指令微调模型中研究了这一区别。我们构建了一个单参数方向族，其范围从协方差调整判别式到均值差，并在固定副作用的情况下沿该方向族评估探测和加性引导（即将该方向添加到隐藏状态）。对于真实性和情感这两个概念，在所有模型中，最佳探测方向都不是最佳引导方向：引导始终偏好更接近均值差的方向。这种分离在层扫描中60个层-模型-概念组合中的59个上成立，并在所有六个模型-概念组合中均使用匹配的探测和引导文本进行验证。

    arXiv:2611.19499v1 Announce Type: cross  Abstract: A direction in a language model's hidden state can be used to probe whether a concept is present or to steer the model toward it. Although probe directions are often reused for steering, the direction that best detects a concept need not be the one that best changes it. We study this distinction in three instruction-tuned models from different families. We construct a one-parameter family of directions ranging from a covariance-adjusted discriminant to the mean difference, and evaluate probing and additive steering, which adds the direction to the hidden state, along the family at a fixed side effect. For truthfulness and sentiment, in every model, the best probe is not the best steering direction: steering consistently favors directions closer to the mean difference. This separation holds across 59 of 60 layer-model-concept combinations in a layer sweep, with matched probe and steering texts in all six model-concept combinations, and 
    
[^134]: 文本自我加速：会话本地草稿生成中的近因性与克制

    Text Speeding Itself Up: Recency and Restraint in Session-Local Drafting

    [https://arxiv.org/abs/2611.19455](https://arxiv.org/abs/2611.19455)

    该研究发现，在会话本地草稿加速中，“复制最近匹配”这一简单的近因性策略是速度提升的全部来源，而复杂的会话记忆机制（加权、回退、置信度门控）在付出 30-40 倍草稿成本的情况下并无可测量的额外收益。

    

    推测解码让一个廉价的草稿模型提出候选 token，再由大型语言模型在单次前向传播中完成验证。当验证器保留 KV 缓存时，验证成本很低，而草稿生成必须近乎零成本。会话本地草稿器基于提示词和已生成的文本进行预测，无需训练即可满足这一要求。我们将一种近因加权、带回退机制、置信度门控的会话记忆方法，与提示词查找解码进行比较：包括其参考形式（从最早的先前匹配处复制）以及一种从最近匹配处复制的变体，全程采用逐提示配对自助法区间进行评估。在使用 Mistral-7B 验证器对通用、医疗和法律文本的实验中，近因性是速度提升的关键：在 500 token 的生成任务上，最近匹配查找相比最早匹配查找可获得高达 0.09 的加速提升，而会话记忆的加权和回退机制在此之上并未带来可测量的增益，其草稿生成成本却高出 30 至 40 倍。能耗则呈现出不同的（原文在此处截断）

    arXiv:2611.19455v1 Announce Type: new  Abstract: Speculative decoding lets a cheap draft model propose tokens that a large language model verifies in a single forward pass. When the verifier keeps a KV cache, verification is cheap and drafting must be nearly free. Session-local drafters, which predict from the prompt and the text generated so far, meet that bar without training. We compare a recency-weighted, backing-off, confidence-gated session memory against prompt lookup decoding in its reference form, which copies from the earliest earlier match, and in a variant that copies from the most recent one, with paired per-prompt bootstrap intervals throughout. On general, medical, and legal text with a Mistral-7B verifier, recency accounts for the speed: most-recent lookup gains up to 0.09 in speedup over earliest-match lookup on 500-token generations, and session memory's weighting and backoff add nothing measurable on top, at 30 to 40 times the drafting cost. Energy tells a different 
    
[^135]: 基于规则的工具能否帮助大语言模型翻译濒危的乌拉尔语言：写入提示词还是通过MCP调用？

    Do Rule-Based Tools Help LLMs Translate Endangered Uralic Languages in the Prompt or over MCP?

    [https://arxiv.org/abs/2611.19439](https://arxiv.org/abs/2611.19439)

    该研究的核心贡献是首次系统评估了为大语言模型提供成熟的规则语言学资源的两种方式，发现通过MCP让模型自主调用工具（尤其是双语词典）比在提示词中预先写入分析结果更能显著提升濒危乌拉尔语言到英语的机器翻译质量。

    

    科米-济良语、莫克沙语、埃尔齐亚语和乌德穆尔特语拥有成熟的基于规则的语言资源，包括有限状态分析器、约束语法和双语词典，但语言模型并未使用这些资源。我们探究让模型访问这些资源能否改善从这四种语言到英语的机器翻译，以及访问方式是否有影响：是将预先计算好的分析结果写入提示词，还是让模型通过模型上下文协议（MCP）自行调用这些资源作为工具。我们采用人工评估：一位专家语言学家阅读了最强模型在三种条件下的全部英文输出，共816个句子，对每个句子判断接受或拒绝，并为每个被拒绝的译文指出主要错误。将分析结果放入提示词后，被接受的译文数量从272句中的13句提升至43句；而附加工具后提升至55句。这一提升主要来自双语词典。对于乌德穆尔特语，其词典有……（摘要原文在此处截断）

    arXiv:2611.19439v1 Announce Type: cross  Abstract: Komi-Zyrian, Moksha, Erzya and Udmurt have mature rule-based resources, among them finite-state analysers, constraint grammars and bilingual dictionaries, which language models do not use. We ask whether giving a model access to these resources improves machine translation from the four languages into English, and whether it matters how the access is provided: as analyses computed in advance and written into the prompt, or as tools that the model calls itself through the Model Context Protocol (MCP). We evaluate manually: an expert linguist read every English output of the strongest model under three conditions, 816 sentences in all, accepted or rejected each one, and named the main error in every rejected translation. With the analyses in the prompt, the number of accepted translations rises from 13 to 43 out of 272; with the tools attached it rises to 55. The gain comes from the bilingual dictionary. For Udmurt, whose dictionary has 
    
[^136]: 预训练优化器的增益能否迁移到KL蒸馏？

    Do Pretraining Optimizer Gains Transfer to KL Distillation?

    [https://arxiv.org/abs/2611.19419](https://arxiv.org/abs/2611.19419)

    本研究通过对四个Qwen师生模型对系统比较多种优化器，发现预训练中矩阵结构优化器的增益并不能自动迁移到短程KL蒸馏任务，AdamW在质量与训练时间的权衡上依然表现最佳。

    

    矩阵结构优化器可以改善语言模型的预训练，但这些增益是否能迁移到从指令微调初始化出发的短程KL蒸馏中尚不明确。我们在统一的forward-KL目标和固定训练预算下，将AdamW和对角AdaGrad与两种可行的Shampoo简化版本、SOAP以及All-2D Muon进行了比较，实验覆盖四个稠密Qwen师生模型对。在各个模型规模上，没有任何一个被评估的矩阵结构优化方案在质量与训练时间的权衡上能持续优于逐元素基线：AdamW通常达到最低的验证KL，而AdaGrad在更大规模下可能略快一些。SOAP的KL表现通常接近AdamW，但会带来显著的训练时间开销；两种Shampoo简化版本在不同模型对之间的性能排序会发生颠倒，这表明结论依赖于具体可行的方案。这些结果表明，预训练中优化器的增益并不会自动迁移到（KL蒸馏中）。

    arXiv:2611.19419v1 Announce Type: cross  Abstract: Matrix-structured optimizers can improve language-model pretraining, but it is unclear whether these gains transfer to short-horizon KL distillation from an instruction-tuned initialization. We compare AdamW and diagonal AdaGrad with two tractable Shampoo reductions, SOAP, and All-2D Muon across four dense Qwen student-teacher pairs under a common forward-KL objective and fixed training budget. Across scales, none of the evaluated matrix-structured recipes consistently improves the quality versus trainer-time trade-off over the elementwise references: AdamW generally attains the lowest validation KL, while AdaGrad can be slightly faster at larger scale. SOAP often tracks AdamW in KL but incurs substantial trainer-time overhead, and the two Shampoo reductions reverse ordering across model pairs, showing that conclusions depend on the feasible recipe. These results suggest that pretraining optimizer gains do not transfer automatically to
    
[^137]: 真实世界端到端产品实体消解

    Real-World End-to-End Product Entity Resolution

    [https://arxiv.org/abs/2611.19418](https://arxiv.org/abs/2611.19418)

    该论文对四种零样本产品实体消解流水线进行了端到端评估，并引入了两个具有跨语言噪音和密集品牌冲突特征的真实世界数据集。

    

    电子商务中的产品实体消解由于非结构化文本、格式不一致、促销噪音以及细粒度的产品差异而极具挑战性。我们对四种零样本PER流水线——词汇级联、语义级联、混合融合级联和生成式-词汇级联——在既有基准上进行了端到端评估，并引入了两个以跨语言噪音和密集品牌冲突为特征的真实世界数据集。我们评估了词汇分块与稠密分块结合交叉编码器、张量搜索和大语言模型验证的效果，同时综合考虑了准确性、运行时间和内存消耗。

    arXiv:2611.19418v1 Announce Type: new  Abstract: Product Entity Resolution (PER) in e-commerce is challenging due to unstructured text, formatting inconsistencies, promotional noise, and fine-grained product differences. We present an end-to-end evaluation of four zero-shot PER pipelines---Lexical Cascade, Semantic Cascade, Hybrid Fusion Cascade, and Generative-Lexical Cascade---across established benchmarks and introduce two real-world datasets characterized by cross-lingual noise and dense brand collisions. We evaluate lexical and dense blocking with cross-encoder, tensor-search, and Large Language Model verification, considering accuracy, runtime, and memory consumption.
    
[^138]: 从事实到角色画像：基于混合专家的大语言模型可解释角色遗忘

    From Facts to Personas: Interpretable Role Unlearning in LLMs via Mixture-of-Experts

    [https://arxiv.org/abs/2611.19393](https://arxiv.org/abs/2611.19393)

    该论文提出RoleMoE混合专家框架，通过将不同角色映射到独立的专家子空间，实现大语言模型中细粒度、可解释的角色扮演遗忘——选择性遗忘目标角色的风格与个性化特征，同时保留其余角色的语言能力。

    

    大语言模型已被广泛应用于多角色对话和角色人格模拟。然而，其强大的模仿特定角色语言和行为模式的能力可能带来隐私风险、强化社会偏见，并导致不安全或不可控的输出。为解决这些问题，我们提出了“角色扮演遗忘”任务，旨在选择性地遗忘与目标角色相关的风格和知识，同时保留其余角色的语言能力和行为。与以往仅专注于删除事实信息的遗忘任务不同，我们的任务还要求遗忘角色特有的个性化特征，从而使对模型生成的控制更加细粒度。我们提出了RoleMoE，一个混合专家框架，它将不同角色映射到更加独立的专家子空间，提升了角色分离度和结构可解释性。为进一步区分角色特有特征，RoleMoE引入了

    arXiv:2611.19393v1 Announce Type: new  Abstract: LLMs are widely adopted in multi-role dialogue and persona simulation. However, their strong ability to imitate role-specific linguistic and behavioral patterns can introduce privacy risks, reinforce social biases, and lead to unsafe or uncontrollable outputs. To address these issues, we introduce the Role-playing Unlearning task, which aims to selectively forget the style and knowledge associated with target roles while preserving the language ability and behaviors of the remaining roles. Unlike prior unlearning tasks that focus solely on removing factual information, our task additionally requires forgetting persona-specific characteristics, making the control over model generation more fine-grained. We propose RoleMoE, a Mixture-of-Experts framework that maps different roles to more independent expert subspaces, improving role separation and structural interpretability. To further distinguish role-specific features, RoleMoE incorporat
    
[^139]: CausalGloss：面向普通话语音到美式手语言标（ASL Gloss）同步级联翻译的多证据动作控制

    CausalGloss: Multi-Evidence Action Control for a Simultaneous Mandarin Speech-to-ASL Gloss Cascade

    [https://arxiv.org/abs/2611.19375](https://arxiv.org/abs/2611.19375)

    CausalGloss 提出了一种融合词汇与语义证据等多证据的动作控制策略，用于普通话语音到 ASL gloss 的同步级联翻译，在 Token F1 上几乎追平自适应学习基线的同时将延迟（LAAL）降低 54.4%–58%，并在人工参考上实现对强基线的帕累托占优。

    

    普通话语音到美式手语（ASL）gloss 的同步翻译需要判断一段不断演化的语音在何时足以支持提交每一个 gloss。固定的 Wait-k 策略无法适应源语言与 gloss 之间的语序重排，而仅在参考历史上训练会造成曝光偏差。我们提出了 CausalGloss，一个面向普通话语音级联的“英文前缀到 ASL gloss”动作策略。它融合了词汇与语义证据、有界延迟、否定约束、突发 WRITE 动作以及不完整历史恢复等机制。在一个固定的、由 How2Sign 衍生的 500 条样本基准上（使用较弱的 LLM 生成参考），CausalGloss 改进了固定 Wait-k 的性能前沿，并在 Token F1 上几乎追平了使用三个随机种子训练的 TransLLaMA 风格自适应学习基线（k = 5 时为 0.7047 对 0.7067），同时 LAAL 降低了 54.4%。在与 CLERC Epee 人工 gloss 参考对比时，它达到了 0.649 的 Token F1，并在帕累托意义上优于三种子 TransLLaMA 风格 k = 5 以及固定 k = 5、7 的基线，将 LAAL 降低了 58。

    arXiv:2611.19375v1 Announce Type: new  Abstract: Simultaneous Mandarin speech-to-American Sign Language (ASL) gloss translation requires deciding when an evolving utterance supports committing each gloss. Fixed Wait-k cannot adapt to source-gloss reordering, while training only on reference histories causes exposure bias. We present CausalGloss, an English-prefix-to-ASL-gloss action policy for a Mandarin-speech cascade. It combines lexical and semantic evidence, bounded delay, negation constraints, burst WRITE actions, and incomplete-history recovery. On a frozen 500-example How2Sign-derived benchmark with weak LLM-generated references, CausalGloss improves the fixed Wait-k frontier and nearly matches a three-seed TransLLaMA-style learned adaptive baseline at k = 5 in Token F1 (0.7047 versus 0.7067), with 54.4% lower LAAL. Against CLERC Epee human-gloss references, it achieves 0.649 Token F1 and Pareto-dominates three-seed TransLLaMA-style k = 5 and fixed k = 5, 7, reducing LAAL by 58.
    
[^140]: 面向心理健康文本分类的泄漏感知校准决策级融合

    Leakage-aware calibrated decision-level fusion for mental-health text classification

    [https://arxiv.org/abs/2611.19360](https://arxiv.org/abs/2611.19360)

    该论文提出了一个用于心理健康文本分类的泄漏感知校准决策级融合框架，通过谱系审计发现并消除了训练/测试数据泄漏，并创新性地提出互补性指数（CI）来指导编码器集成选择。

    

    决策级融合受益于互补错误、校准概率和泄漏感知评估。我们在262,276条涵盖焦虑、抑郁、精神分裂症和自杀意念的代理标注Reddit记录上评估了一个异构transformer融合框架。在分层随机划分中，MinHash-LSH识别出29.1%的测试记录为训练数据的词汇近似重复；数据增强谱系审计发现50.2%的记录与训练集存在谱系重叠，其中包括经词汇过滤后仍保留的30.0%记录。因此，我们采用了谱系不相交的划分方式。十二个编码器——包括Bio-ClinicalBERT、MentalBERT、MentalRoBERTa和RoBERTa-base各三个随机种子——经过交叉拟合验证，用于校准和集成选择。我们定义了一个类别特定的互补性指数（CI），即各成员模型正确分类的验证集之间的Jaccard距离，并将其与独立评估的融合增益进行对照分析。CI与……（原文摘要在此处截断）

    arXiv:2611.19360v1 Announce Type: new  Abstract: Decision-level fusion benefits from complementary errors, calibrated probabilities, and leakage-aware evaluation. We evaluated a heterogeneous transformer fusion framework on 262,276 proxy-labeled Reddit records covering anxiety, depression, schizophrenia, and suicidal ideation. In a stratified random split, MinHash-LSH identified 29.1% of test records as lexical near-duplicates of training data. Augmentation-lineage auditing found 50.2% lineage overlap with training, including 30.0% of records retained after lexical filtering. We therefore used lineage-disjoint partitions. Twelve encoders, comprising three seeds each of Bio-ClinicalBERT, MentalBERT, MentalRoBERTa, and RoBERTa-base, underwent cross-fitted validation for calibration and ensemble selection. A class-specific Complementarity Index (CI), defined as the Jaccard distance between members' correctly classified validation sets, was assessed against independent fusion gain. CI corr
    
[^141]: 基于BERT与基于大语言模型的印度语系句子相似度模型对比评估

    A Comparative Evaluation of BERT-Based and LLM-Based Models for Indic Sentence Similarity

    [https://arxiv.org/abs/2611.19356](https://arxiv.org/abs/2611.19356)

    该研究在十种印度语系翻译的STS基准和马拉地语MahaSTS测试集上系统对比了13个模型，发现基于BERT的模型（如IndicSBERT）在印度语系句子相似度任务上整体优于基于大语言模型的嵌入模型。

    

    基于BERT的编码器与基于大语言模型（LLM）的嵌入在印度语系语义文本相似度任务中的相对有效性仍未得到充分刻画。我们在翻译为十种印度语系的1,500对STS基准开发集以及原生马拉地语MahaSTS测试集上，对八个基于BERT的模型和五个基于LLM的模型进行了比较。IndicSBERT在全部十个翻译语言集上取得了最高的皮尔逊相关系数，宏平均为0.843，其次是BGE-M3的0.792。Harrier-270M是翻译版STS-B上表现最强的基于LLM的模型，宏平均为0.751。在MahaSTS上，IndicSBERT和Bhasha-Embed-v0分别取得了0.901和0.900的皮尔逊相关系数，而EmbeddingGemma以0.871领先基于LLM的模型。因此，得分最高的基于BERT的模型在两种评测设置下均超过了得分最高的基于LLM的模型，尽管差异幅度因基准而异。

    arXiv:2611.19356v1 Announce Type: new  Abstract: The relative effectiveness of BERT-based encoders and large language model (LLM)-based embeddings for Indic semantic textual similarity remains insufficiently characterized. We compare eight BERT-based and five LLM-based models on the 1,500-pair STS Benchmark development set translated into ten Indic languages and on the native Marathi MahaSTS test set. IndicSBERT achieves the highest Pearson correlation in all ten translated language sets, with a macro-average of 0.843, followed by BGE-M3 at 0.792. Harrier-270M is the strongest LLM-based model on translated STS-B, with a macro-average of 0.751. On MahaSTS, IndicSBERT and Bhasha-Embed-v0 achieve Pearson correlations of 0.901 and 0.900, respectively, while EmbeddingGemma leads the LLM-based models at 0.871. The highest-scoring BERT-based model thus exceeds the highest-scoring LLM-based model in both settings, although the magnitude of the difference varies across benchmarks. These finding
    
[^142]: 无标签文档嵌入去混杂何时有效？

    When Does Label-Free Deconfounding of Document Embeddings Work?

    [https://arxiv.org/abs/2611.19350](https://arxiv.org/abs/2611.19350)

    该论文系统研究了跨语料库和嵌入模型的无标签去混杂方法，发现简单几何方法即可恢复大部分有监督擦除的收益，并揭示了混杂因子可恢复性与语义效用是去混杂成功的两个相互独立的维度。

    

    文档嵌入会编码一些虚假属性，这些属性可能扭曲语义比较；然而，有监督的擦除方法需要通常难以获得的元数据。我们在多个语料库和嵌入模型上系统地研究了无标签去混杂方法。结果表明，无标签方法相比未处理的嵌入能够提升语义效用，而简单的几何方法可以恢复大部分有监督方法带来的收益，有时在线性混杂因子可恢复性和语义效用方面甚至能与基于元数据的有监督擦除相媲美。我们识别出擦除失效的两个不同原因：代理变量与混杂因子对齐不佳，以及干预覆盖不足。至关重要的是，混杂因子可恢复性与语义效用是去混杂成功的两个不同维度：即使混杂因子信息仍有残留，语义效用仍可提升；而更强的抑制并不必然带来效用的提升。此外，线性可恢复性的降低可能与大量……（摘要在此处截断）

    arXiv:2611.19350v1 Announce Type: new  Abstract: Document embeddings encode spurious attributes that can distort semantic comparisons; yet supervised erasure requires metadata that is often unavailable. We systematically investigate label-free deconfounding across corpora and embedding models. Our results show that label-free methods improve semantic utility over unprocessed embeddings, while simple geometric approaches recover much of the supervised benefit, sometimes matching metadata-supervised erasure in linear confounder recoverability and semantic utility. We identify two distinct causes of ineffective erasure: poor proxy-confounder alignment and insufficient intervention coverage. Crucially, confounder recoverability and semantic utility are distinct dimensions of deconfounding success: utility can improve despite residual confounder information, while stronger suppression does not necessarily improve utility. Moreover, reduced linear recoverability can coexist with substantial 
    
[^143]: OmitEHR：LLM生成临床总结中可扩展的遗漏检测方法

    OmitEHR: Scalable Omission Detection in LLM-Generated Clinical Summaries

    [https://arxiv.org/abs/2611.19343](https://arxiv.org/abs/2611.19343)

    OmitEHR通过从LLM普及前临床医生撰写的历史摘要中学习临床显著性，实现了无需参考即可扩展检测LLM生成临床总结中关键信息遗漏的方法。

    

    医疗系统日益依赖大型语言模型（LLM）来起草患者医疗记录的摘要，然而评估研究发现，遗漏——即临床重要信息的缺失——是最常见的失败模式。检测LLM生成摘要中的遗漏需要知道记录中哪些事实是具有重要临床意义的，而现有方法要么手动指定这一信号，要么忽略它，要么在推理阶段使用成本高昂的LLM评判器重新计算。我们提出了OmitEHR，一种无需参考答案的遗漏检测器，它从LLM广泛应用之前的临床医生撰写的历史摘要中学习临床显著性。OmitEHR学习预测临床医生在其摘要中会包含哪些源记录中的句子，然后将LLM生成摘要未覆盖的高显著性句子作为候选遗漏呈现出来。我们将OmitEHR应用于简短住院病程（BHC）总结任务，使用了来自一家大型学术医疗中心的5,000例住院数据。在留出的住院数据上，OmitEHR

    arXiv:2611.19343v1 Announce Type: cross  Abstract: Health systems increasingly rely on large language models (LLMs) to draft summaries of patient medical records, yet evaluations now find omissions, the absence of clinically important information, to be the most common failure mode. Detecting omissions in an LLM-generated summary requires knowing which facts in the record are salient, a signal that existing methods hand-specify, ignore, or recompute at inference with costly LLM judges. We present OmitEHR, a reference-free omission detector that learns clinical salience from historical clinician-written summaries predating widespread LLM use. OmitEHR learns to predict which source-note sentences clinicians include in their summaries, then surfaces high-salience sentences not covered by an LLM-generated summary as candidate omissions. We apply OmitEHR to brief hospital course (BHC) summarization using 5,000 admissions from a large academic medical center. On held-out admissions, OmitEHR 
    
[^144]: 黑箱世界模型的宏观控制与能力评估：面向LLM智能体市场的双机制框架

    Macro Control and Capability Evaluation of Black-Box World Models: A Two-Regime Framework for LLM-Agent Markets

    [https://arxiv.org/abs/2611.19311](https://arxiv.org/abs/2611.19311)

    本文提出一种软宏观控制方法与双机制评估框架，仅通过宏观观测将LLM智能体市场作为世界模型的能力分解为信息仿真与宏观涌现两个可分别量化的阶段。

    

    大语言模型（LLM）智能体可以被组织成多智能体市场，作为世界模型运作，将宏观信息映射到涌现的宏观动态，而评估者只能观察宏观的输入与输出。本文将模型的能力分解为单一综合评分无法区分的两个阶段：信息仿真（从宏观信息到微观行为）与宏观涌现（从微观行为到宏观结果）。我们提出了一种软宏观控制方法，逐日公布一条公共参考路径，而不触碰个体决策、订单或撮合机制，并构建了一个双机制评估框架，仅凭宏观观测来衡量这两个阶段。在仅参考机制中，对称的真实参考与翻转参考共享一个共同水平R，用以衡量在公布参考下的实现程度，而两者的配对差距B则衡量结构性偏差；在信息注入机制中，外部信息……（摘要原文在此处截断）

    arXiv:2611.19311v1 Announce Type: new  Abstract: Large language model (LLM) agents can be organized into multi-agent markets that act as world models, mapping macro information to emergent macro dynamics while the evaluator observes only macro inputs and outputs. This paper separates a model's capability into two stages that a single aggregate score cannot separate: information simulation, from macro information to micro behavior, and macro emergence, from micro behavior to the macro outcome. We introduce a soft macro-control method that announces a public reference path day by day without touching individual decisions, orders, or matching, and a two-regime evaluation framework that measures both stages from macro observations alone. In the reference-only regime the symmetric true and flipped references share a common level R, which measures realization under the announced reference, while their paired gap B measures structural bias; in the information-injection regime external informa
    
[^145]: 分阶段语义蒸馏：在分阶段LLM智能体的低算力非LLM代理中保留自由形式语义

    Staged Semantic Distillation: Preserving Free-Form Semantics in a Low-Compute Non-LLM Surrogate of a Staged LLM Agent

    [https://arxiv.org/abs/2611.19299](https://arxiv.org/abs/2611.19299)

    该论文提出“分阶段语义蒸馏”方法，仅凭一条记录轨迹即可训练出1320万参数的非LLM学生代理，在单块8-GB GPU上1658秒完成训练、每次决策仅需0.14毫秒，同时以余弦相似度0.943的精度保留分阶段LLM智能体的自由形式语义。

    

    分阶段LLM智能体使市场与社交模拟器更加真实，但一次由200个智能体、十个指数组成的市场模拟器运行102天需要发出203,100次语言模型调用，且仅返回一种实现结果。传统的低算力蒸馏会丢失其自由形式的中间状态。我们提出分阶段语义蒸馏——一种低算力方法，可在由单条记录轨迹训练的非LLM学生模型中保留非结构化语义。冻结编码器将每个教师输出映射为1024维向量；学生模型接收带类型、带阶段标签且具有阶段特定可见性的向量，接受联合监督，并通过调度的自历史进行训练。一条TwinMarket轨迹即可训练出1320万参数的确定性学生模型，在单块8-GB GPU上1658秒内完成训练，运行时无需任何教师或解码器调用、无需文本生成（每次决策0.14毫秒）。其分析结果与教师保持高度接近（余弦相似度0.943），且在智能体之间交换语义输入会改变64.7%的决策……

    arXiv:2611.19299v1 Announce Type: new  Abstract: Staged LLM agents make market and social simulators realistic, but one 102-day run of a 200-agent, ten-index market simulator issues 203,100 language-model calls and returns one realization. Conventional low-compute distillation drops their free-form intermediate state. We present staged semantic distillation, a low-compute method that preserves non-structured semantics in a non-LLM student trained from one recorded trace. A frozen encoder maps each teacher output to a 1024-d vector; the student sees typed stage-tagged vectors with stage-specific visibility, is jointly supervised, and trains with scheduled self-history. One TwinMarket trace yields a 13.2M-parameter deterministic student that trains in 1,658 seconds on one 8-GB GPU and runs with zero teacher or decoder calls and no text generation (0.14 ms per decision). Its analysis stays close to the teacher (cosine 0.943), and swapping semantic inputs between agents moves 64.7% of deci
    
[^146]: 论语言模型训练数据归因方法的表征对齐与功能对齐

    On Representational and Functional Alignment of Training Data Attribution Methods for Language Models

    [https://arxiv.org/abs/2611.19286](https://arxiv.org/abs/2611.19286)

    该论文提出了从功能对齐、表征对齐和跨方法可预测性三个维度直接比较语言模型训练数据归因方法的评估框架，发现同一范式内方法高度对齐，而跨范式对齐较低但在不同模型间保持稳定。

    

    现有的训练数据归因方法从不同的信息来源推导影响估计，包括梯度、模型内部状态以及表层相似性。然而，归因方法之间的比较仍不完善，通常依赖于通过下游任务性能进行的间接评估。在这项工作中，我们提出了一个框架，用于在三个维度上直接比较各种方法：功能对齐，衡量完整归因向量之间的相关性；表征对齐，刻画归因所依赖的底层表征的结构相似性；以及跨方法可预测性，评估各方法在局部成对排序决策上的一致性。我们观察到，在同一范式内部存在强表征对齐和中等程度的功能对齐。跨范式的对齐程度较低且依赖于任务，但在不同模型之间保持稳定。我们的结果表明，所观察到的相似下游性能……

    arXiv:2611.19286v1 Announce Type: new  Abstract: Existing approaches to training data attribution derive influence estimates from different sources of information, including gradients, model-internal states, and surface-level similarity. However, comparisons between attribution methods remain incomplete, often relying on indirect evaluations through downstream task performance. In this work, we introduce a framework for directly comparing methods along three dimensions: functional alignment, which measures correlation between full attribution vectors; representational alignment, which captures structural similarity of the representations underlying attribution; and cross-method predictability, which evaluates agreement on local pairwise ranking decisions. We observe strong representational and moderate functional alignment within paradigms. Alignment across paradigms is low and task-dependent, yet stable across models. Our results suggest that the similar downstream performance observe
    
[^147]: B-OPSD：基于自生成问题简报的在线策略自蒸馏

    B-OPSD: On-Policy Self-Distillation with Self-Generated Problem Briefs

    [https://arxiv.org/abs/2611.19273](https://arxiv.org/abs/2611.19273)

    B-OPSD让模型自行从问题生成不含答案的简报（包含原理、适用条件和常见陷阱）来替代标准答案解作为教师模型的指导上下文，实现了无需真实解的在线策略自蒸馏，并在数学竞赛基准上取得了优于OPSD的准确率。

    

    在线策略自蒸馏（OPSD）近来受到关注，因为它无需一个单独的、通常更大的教师模型即可提供密集的教师指导。然而，OPSD以真实标准答案解作为教师模型的输入条件，这限制了它在这些解不可得时的使用。我们提出了基于自生成问题简报的在线策略自蒸馏（B-OPSD）。通过使用提示模板，初始模型或保存的学生检查点仅根据问题本身生成一个问题简报。该提示要求给出相关原理、适用条件和可能的陷阱，但不要求给出最终答案或完整解答。一个冻结的教师模型将该简报作为特权上下文，通过token级别的分布匹配来指导当前学生模型的在线策略采样。B-OPSD在训练过程中不使用任何真实标准答案解。在AIME 2024、AIME 2025和HMMT 2025上的结果显示，两种B-OPSD变体在每个基准上的平均准确率均高于OPSD。

    arXiv:2611.19273v1 Announce Type: new  Abstract: On-Policy Self-Distillation (OPSD) has recently attracted interest because it provides dense teacher guidance without a separate, often larger teacher model. However, OPSD conditions its teacher on ground-truth solutions, which limits its use when such solutions are unavailable. We propose On-Policy Self-Distillation with Self-Generated Problem Briefs (B-OPSD). Using a prompt template, the initial model or a saved student checkpoint generates a problem brief from the question alone. The prompt asks for relevant principles, applicability conditions, and likely pitfalls, but not a final answer or a complete solution. A frozen teacher uses the brief as privileged context to guide the current student's on-policy rollouts through token-level distribution matching. B-OPSD uses no ground-truth solutions during training. Results on AIME 2024, AIME 2025, and HMMT 2025 show that both B-OPSD variants achieve higher average accuracy than OPSD at eac
    
[^148]: 听觉反馈条件化的流式TTS实现隐式隆巴德效应适应

    Auditory-Feedback-Conditioned Flow TTS for Implicit Lombard Adaptation

    [https://arxiv.org/abs/2611.19260](https://arxiv.org/abs/2611.19260)

    该论文提出将信噪比和语音识别反馈从噪声污染的音频中提取并注入流匹配TTS模型，使系统能像人类的隆巴德效应一样，无需显式韵律控制即可通过听觉反馈隐式适应噪声环境，提升噪声下的语音可懂度。

    

    传统文本转语音（TTS）系统在噪声环境中可能会损失可懂度，因为它们无法适应环境变化。然而，人类在嘈杂环境中往往会增加发声力度，这一现象被称为隆巴德效应。基于听觉反馈的隆巴德TTS，我们研究预训练的基于流的TTS能否直接从听觉反馈中学习声学适应，而无需显式的韵律控制。信噪比（SNR）反馈和自动语音识别（ASR）反馈从被噪声污染的波形中提取，并注入到流匹配条件中，用于话语的精细化调整和因果块级更新。我们通过普通/隆巴德提示、提示切换对照和配对微调，评估了反馈在各种提示和噪声条件下的益处。实验表明，虽然微调在域内条件下提供了强大的鲁棒性，但听觉反馈（摘要在此处被截断）。

    arXiv:2611.19260v1 Announce Type: cross  Abstract: Conventional text-to-speech (TTS) systems can lose intelligibility in noise environments because they cannot adapt to environmental changes. However, humans tend to increase their vocal efforts in noisy environments, a phenomenon known as the Lombard effect. Building on an auditory-feedback Lombard TTS, we investigate whether a pretrained Flow-based TTS can learn acoustic adaptation directly from auditory feedback without explicit prosodic control. Signal-to-noise ratio (SNR) feedback and automatic speech recognition (ASR) feedback are extracted from the noise-corrupted waveforms and injected into the Flow-matching condition for utterance refinement and causal block-level updates. We assess feedback benefits across prompting and noise conditions using plain/Lombard prompting, prompt-switching controls, and paired-finetuning. Experiments show that while finetuning provides strong robustness under in-domain conditions, auditory feedback 
    
[^149]: 重新审视什么构成良好的上下文子集：稀疏潜在基的视角

    Revisiting What Makes a Good Context Subset: A Sparse Latent Basis Perspective

    [https://arxiv.org/abs/2611.19220](https://arxiv.org/abs/2611.19220)

    该论文将少样本示例选择形式化为以查询表示为目标的稀疏重建问题，提出匹配追踪式的残差解释搜索（RES）方法，实验表明其在多数模型-数据集组合上优于MMR，说明示例的集体互补性比成对相异性更重要。

    

    在现代大语言模型推理中，选择上下文子集变得越来越重要，然而是什么使一个子集有效仍不清楚。我们在少样本示例选择的受控环境中研究这一问题。我们检验了一个假设：所选示例应当共同提供目标查询所需的与任务相关的信息。假设查询表示能为这些需求提供有用的代理，我们将选择问题形式化为稀疏重建，并用残差解释搜索（RES）加以实现——这是一种匹配追踪式的诊断程序。在七个基准测试和八个开源权重大语言模型上的实验表明，没有任何检索目标是普遍最优的。RES在56个模型-数据集组合中的34个上优于MMR，其在情感任务上的优势最为明显，这表明以查询为条件的集体重建可能捕获到成对相异性度量所遗漏的有用互补性。而基于相关性的检索仍然表现强劲。

    arXiv:2611.19220v1 Announce Type: cross  Abstract: Selecting context subsets is increasingly important in modern LLM inference, yet what makes a subset effective remains unclear. We study this question in the controlled setting of few-shot demonstration selection. We examine a hypothesis that demonstrations should collectively provide task-relevant information needed by a target query. Assuming that the query representation provides a useful proxy for these requirements, we formulate selection as sparse reconstruction and instantiate it with Residual Explanation Search (RES), a Matching-Pursuit-style diagnostic procedure. Across seven benchmarks and eight open-weight LLMs, no retrieval objective is universally best. RES outperforms MMR in 34 of 56 model--dataset pairs, with its clearest advantage on sentiment tasks, suggesting that query-conditioned collective reconstruction may capture useful complementarity that pairwise dissimilarity misses. Relevance-based retrieval remains strong 
    
[^150]: Omni-Workspace：在多模态工作区任务上评测全模态AI助手

    Omni-Workspace: Benchmarking Omni-Modal AI Assistants on Multimodal Workspace Tasks

    [https://arxiv.org/abs/2611.19202](https://arxiv.org/abs/2611.19202)

    该论文提出了Omni-Workspace基准，包含380个深度研究风格的任务，用于评测全模态AI智能体在与多模态工作区持续交互中完成复杂任务的能力，填补了现有基准仅关注多模态感知与理解的空白。

    

    全模态智能体有望超越多模态理解，在真实的多模态环境中完成复杂任务。在这些环境中，信息分布在异构的文件和模态之中，跨文件和跨模态的依赖关系要求智能体持续协调感知、推理与工具使用，因为新收集的证据会影响后续行动。然而，现有的全模态基准主要关注多模态感知与理解，对于通过与多模态工作区持续交互来完成复杂任务这一方向仍缺乏充分探索。我们提出了Omni-Workspace，这是一个包含380个深度研究风格任务的基准，涵盖四种多模态工作区类型和两个难度级别。这些任务要求智能体检查和处理工作区文件，在需要时检索外部信息，整合跨文件和跨模态的证据，并生成满足……（摘要在此处截断）

    arXiv:2611.19202v1 Announce Type: new  Abstract: Omni-modal agents are expected to move beyond multimodal understanding toward completing complex tasks in realistic multimodal environments. In these settings, information is distributed across heterogeneous files and modalities, with cross-file and cross-modal dependencies requiring agents to continually coordinate perception, reasoning, and tool use as newly gathered evidence informs subsequent actions. However, existing omni-modal benchmarks primarily focus on multimodal perception and understanding, leaving complex task completion through sustained interaction with multimodal workspaces underexplored. We introduce Omni-Workspace, a benchmark comprising 380 deep-research-style tasks across four multimodal workspace types and two difficulty levels. These tasks require agents to inspect and process workspace files, retrieve external information when needed, integrate evidence across files and modalities, and produce artifacts satisfying
    
[^151]: 遮挡诱导部分可观测性下多视角规划中的感知隔离

    Isolating Perception for Multi-View Planning under Occlusion-Induced Partial Observability

    [https://arxiv.org/abs/2611.19179](https://arxiv.org/abs/2611.19179)

    提出VLM-as-perceiver范式，将VLM的角色局限于局部感知，把跨视角证据协调与规划生成委托给外部求解器，在遮挡部分可观测性基准BW-PO上将多个VLM的规划成功率从接近于零大幅提升至84%-97%。

    

    视觉-语言模型（VLM）在视觉规划基准测试中的表现开始趋于饱和，然而在遮挡诱导的部分可观测性（PO）条件下，它们无法解决深度歧义问题，且性能随问题规模增大而急剧下降。我们基于Blocksworld领域提出了BW-PO，这是一个受控基准，能将现有VLM规划器和形式化工具的性能降至接近于零；同时我们提出VLM-as-perceiver（VLM作为感知器）范式，将VLM的角色局限于局部感知，而将跨视角证据的协调与规划生成委托给外部求解器。在BW-PO的各个模型特定挑战性子集上，该方法将Gemma-4-31B的规划成功率从1%提升至97%，Qwen3.5-122B-A10B从0%提升至95%，Gemini-3.1-Pro从3%提升至84%。通过这种感知隔离方式，我们将原本相互纠缠的两个瓶颈分离开来：深度歧义下的证据协调，以及场景密度下的视觉提取完整性。

    arXiv:2611.19179v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are beginning to saturate visual planning benchmarks, yet they fail to resolve depth ambiguity and degrade sharply with problem size under occlusion-induced partial observability (PO). We introduce BW-PO based on the Blocksworld domain, a controlled benchmark that reduces existing VLM planners and formalizers to near-zero performance, and propose VLM-as-perceiver, a paradigm that narrows the VLM's role to local perception while delegating cross-view evidence reconciliation and plan generation to external solvers. Across model-specific challenging subsets of BW-PO, this raises plan success from 1% to 97% for Gemma-4-31B, 0% to 95% for Qwen3.5-122B-A10B, and 3% to 84% for Gemini-3.1-Pro. Isolating perception in this way separates two bottlenecks that are otherwise entangled: reconciliation under depth ambiguity, and completeness of visual extraction under scene density.
    
[^152]: 概念袋？测试可言语化工作区中的角色-填充项绑定

    A Bag of Concepts? Testing Role-Filler Binding in the Verbalizable Workspace

    [https://arxiv.org/abs/2611.19176](https://arxiv.org/abs/2611.19176)

    本文的核心发现是：在多个前沿模型、两种镜头家族和两种拟合方法下，可言语化工作区只携带能引导模型角色相关输出的方向，而不包含模型实际计算角色绑定所用的内部表示，这意味着基于工作区的运行时对齐监控器难以读取模型内部真实的角色-填充项绑定。

    

    构建于可言语化工作区之上的运行时对齐监控器，被提议作为一种在模型产生输出之前揭示其“沉默认知”的手段。这类监控器的价值取决于工作区是否携带足够的结构来区分与安全相关的命题，而这些命题大多是关系性的：“模型计划欺骗用户”与“用户计划欺骗模型”共享同一组概念，仅在角色分配上有所不同。我们在三个前沿语言模型（Qwen3.6-27B、Gemma-3-12B、Gemma-3-27B-IT）上，采用两种镜头家族（J-lens 与 R-lens）以及两种拟合角色方向的方法（均值差法与 LRE 梯度法），探究工作区是否携带此类监控器所需的角色-填充项绑定（即哪个实体填充哪个角色）。我们发现了一种清晰的分离现象：工作区携带的是能够引导模型关于角色输出内容的方向，却并不包含模型用于计算这些角色时的内部表示。线性探针能够恢复……（摘要原文在此处截断）

    arXiv:2611.19176v1 Announce Type: new  Abstract: Runtime alignment monitors built on a verbalizable workspace have been proposed as a way to surface a model's silent cognition before it produces output. Their value rests on the workspace carrying enough structure to distinguish safety-relevant propositions, most of which are relational: "the model plans to deceive the user" and "the user plans to deceive the model" share a concept set and differ only in role assignment. We ask whether the workspace carries the role-filler bindings (which entity fills which role) such a monitor would need, on three frontier language models (Qwen3.6-27B, Gemma-3-12B, Gemma-3-27B-IT), under two lens families (J-lens and R-lens), and with two methods for fitting the role direction (difference-of-means and LRE-gradient). We find a clean dissociation: the workspace carries directions that steer what the model outputs about roles but not the representations the model uses to compute them. Linear probes recove
    
[^153]: 面向一对多私人通信的个性化语言模型

    Personalized Language Models for One-to-Many Private Communications

    [https://arxiv.org/abs/2611.19155](https://arxiv.org/abs/2611.19155)

    提出TRACE框架，利用作者词汇习惯、联系人特定措辞和时间表达估计来引导冻结语言模型并配合反馈驱动的修订策略，实现一对多私人通信中针对特定关系的持续回复预测与个性化状态更新，在Enron邮件数据上优于基础LLM和RAG基线。

    

    我们研究面向一对多私人通信的关系特定的持续回复预测任务：系统预测作者对某个经常性联系人的回复，并在实际回复可用时更新其个性化状态。我们提出TRACE（时间关系感知通信演化），该方法利用作者级别的词汇习惯、联系人特定的措辞以及时间表达估计来引导一个冻结的语言模型，并通过反馈驱动的策略来控制修订。在来自八位Enron用户的216封测试邮件上，全部四个主要的TRACE配置均比基础LLM和基础RAG取得了更高的平均BLEU、ROUGE-L和BERTScore，以及更低的平均回复风格MSE。组件消融实验显示出混合效应，且经过多重比较校正后，未能确立回复风格MSE在统计上可靠的改进。更准确的状态预测和更大的修订预算并不能一致地改善回复级别的指标。这些探索性……（摘要在此处截断）

    arXiv:2611.19155v1 Announce Type: new  Abstract: We study relationship-specific continual reply prediction for one-to-many private communication: a system predicts an author's reply to a recurring contact and updates its personalization state when the actual reply becomes available. We propose TRACE (Temporal Relationship-Aware Communication Evolution), which uses author-level lexical habits, contact-specific wording, and temporal expression estimates to guide a frozen language model, with a feedback-driven policy controlling revision. On 216 test emails from eight Enron users, all four main TRACE configurations achieve higher mean BLEU, ROUGE-L, and BERTScore and lower mean Reply Style MSE than Base LLM and Base RAG. Component ablations show mixed effects and do not establish statistically reliable improvements in Reply Style MSE after multiple-comparison correction. More accurate state forecasts and larger revision budgets do not consistently improve reply-level metrics. These explor
    
[^154]: MarineSparkLM-4B：基于参数高效适配构建的海洋与离岸工程轻量级语言模型

    MarineSparkLM-4B: A Lightweight Language Model for Marine and Offshore Engineering Built by Parameter-Efficient Adaptation

    [https://arxiv.org/abs/2611.19151](https://arxiv.org/abs/2611.19151)

    该研究通过在一台工作站级设备上仅对40亿参数模型进行0.81%参数量的LoRA高效适配，并以约5小时的低算力成本，构建出面向海洋与离岸工程领域的轻量级专用语言模型MarineSparkLM-4B。

    

    海洋与离岸工程领域的文本散布于预印本、开放获取期刊、百科全书和机构文档之中，且按预训练标准衡量其规模很小。我们探究的问题是：依靠参数高效适配，在一台工作站级单机上，面向海洋与离岸工程的轻量级语言模型究竟能走多远。从40亿参数的OceanGPT-basic-4B-Instruct出发，我们训练了两个独立的LoRA适配器，每个适配器仅更新40.6亿参数中的3300万（即0.81%）。在单台NVIDIA DGX Spark上，第一个适配器训练耗时4小时40分钟，第二个仅耗时10分钟。第一个适配器通过下一词元预测任务，在覆盖十个工程主题、共6160万词元的语料库上训练。该语料库由13,879份源文档构建，经去重和去除跨领域词汇冲突的两级关键词过滤后保留了7,069份文档。其验证损失……（原文在此截断）

    arXiv:2611.19151v1 Announce Type: cross  Abstract: Marine and offshore engineering text is spread across preprints, open-access journals, encyclopedias and institutional documents, and the amount of it is small by pretraining standards. We ask how far a lightweight language model for marine and offshore engineering can be taken with parameter-efficient adaptation on a single workstation-class device. Starting from the 4-billion-parameter OceanGPT-basic-4B-Instruct, we train two independent LoRA adapters, each updating 33.0 million of 4.06 billion parameters (0.81%). Training took 4 hours 40 minutes for the first adapter and 10 minutes for the second on one NVIDIA DGX Spark. The first adapter is trained by next-token prediction on a 61.6-million-token corpus covering ten engineering themes. The corpus was built from 13,879 source documents, of which 7,069 were retained after deduplication and a two-tier keyword filter that removes cross-domain lexical collisions. Its validation loss fal
    
[^155]: 面向数据高效非言语语音合成的跨词元迁移系统性研究

    A Systematic Study of Cross-Token Transfer for Data-Efficient Non-Verbal Text-to-Speech

    [https://arxiv.org/abs/2611.19145](https://arxiv.org/abs/2611.19145)

    该研究发现非言语发声词元之间存在显著的跨词元迁移效应——仅需200个笑声样本加上3,200个咳嗽样本，就能超越单独使用1,600个笑声样本的笑声合成效果，为数据高效的非言语语音合成提供了新路径。

    

    非言语发声（NVs），如笑声和呼吸声，携带着仅凭言语无法传达的情感线索。然而，由于语音标注流程常常忽略或规范化处理这些事件，NV数据仍然稀缺。现有工作通过模型设计和数据整理改进了NV文本转语音，但一种NV词元能否帮助模型学习另一种NV词元，尚未得到系统性研究。我们在受控的CosyVoice 2设置中研究了这种跨词元可迁移性。我们首先追踪了仅使用笑声数据的扩展曲线，然后在将笑声样本数固定为200或1,600的情况下，添加呼吸、叹气、咳嗽、咂嘴或中性语音数据。仅使用200个笑声样本时，添加3,200个咳嗽样本可将端到端笑声F1分数从0.170提升至0.529，超过了仅使用1,600个笑声样本所获得的0.495。在1,600个样本的基准下，添加咳嗽样本可达到0.695，几乎匹配6,400个笑声样本的效果（0.697）。波形级别的Whisper-D评分和语义（摘要内容被截断）

    arXiv:2611.19145v1 Announce Type: new  Abstract: Non-verbal vocalizations (NVs), such as laughter and breath, carry affective cues that words alone do not convey. Yet NV data remains scarce because speech annotation pipelines often omit or normalize these events. Existing work has improved NV text-to-speech through model design and data curation, but whether one NV token can help a model learn another has received little systematic study. We investigate this cross-token transferability in a controlled CosyVoice 2 setup. We first trace a laughter-only scaling curve, then hold laughter at 200 or 1,600 examples while adding breath, sigh, cough, lip-smack, or neutral speech. With only 200 laughter examples, adding 3,200 cough examples raises end-to-end laughter F1 from 0.170 to 0.529, surpassing the 0.495 obtained from 1,600 laughter examples alone. At the 1,600-example anchor, cough reaches 0.695, nearly matching 6,400 laughter examples (0.697). Waveform-level Whisper-D scoring and a sema
    
[^156]: 基于可验证奖励的低延迟大语言模型流式多说话人语音识别

    Low-Latency LLM-Based Streaming Multi-Talker ASR with Verifiable Rewards

    [https://arxiv.org/abs/2611.19136](https://arxiv.org/abs/2611.19136)

    该论文提出用可验证奖励的强化学习（RLVR）对流式LLM语音识别模型进行后训练，将固定角色说话人归属WER评分器与轮次级起始延迟奖励相结合，在可穿戴设备实时多说话人对话转录中同时解决转录准确性、说话人归属和低延迟三大难题。

    

    基于大语言模型（LLM）的架构为流式自动语音识别（ASR）提供了强大的泛化能力和指令遵循能力，但在多说话人场景下会引入转录、说话人归属与延迟相互耦合的挑战。我们的可穿戴设备场景需要对近场设备佩戴者与远场对话伙伴之间的对话进行实时转录。由于部分转写假设会持续呈现在对话界面的固定一侧，说话人角色不能被置换，且每个说话人标签必须出现在其对应话语轮次之前。这些稀疏的说话人标签在监督微调（SFT）中只能得到弱化优化，而过早提交先导标签又会延迟后续文本的输出。因此，我们采用可验证奖励的强化学习（RLVR）对基于流式LLM的ASR模型进行后训练，将固定角色的说话人归属词错误率（SA-WER）评分器与轮次级的起始延迟奖励相结合。在一个八方向多说话人测试集上，SA-WER强化学习将SA-WER从20（原文数值在此处截断）……

    arXiv:2611.19136v1 Announce Type: new  Abstract: LLM-based architectures offer strong generalization and instruction-following capabilities for streaming automatic speech recognition (ASR), but multi-talker use introduces coupled transcription, attribution, and latency challenges. Our wearable setting requires real-time transcription of a conversation between a near-field device wearer and a far-field partner. Because partial hypotheses are rendered on persistent sides of a dialogue UI, speaker roles cannot be permuted and each speaker tag must precede its turn. These sparse tags are weakly optimized by supervised fine-tuning (SFT), while committing to a leading tag can delay the text that follows. We therefore post-train a streaming LLM-based ASR model with reinforcement learning with verifiable rewards (RLVR), combining a fixed-role Speaker-Attributed WER (SA-WER) grader with a turn-level onset-delay reward. On an eight-direction multi-talker cohort, SA-WER RL reduces SA-WER from $20
    
[^157]: 利用留一变换器赋能掩码扩散模型实现自我纠错

    Empowering Masked Diffusion Models to Self-Correct with Leave-One-Out Transformers

    [https://arxiv.org/abs/2611.19127](https://arxiv.org/abs/2611.19127)

    本文提出留一变换器（LOOT），通过改进左右上下文合并机制和推导低方差NELBO估计器，赋能掩码扩散模型实现token自我纠错并显著提升预测性能。

    

    掩码扩散模型（MDMs）已成为一种有前景的语言建模范式，能够以具有竞争力的性能实现并行token生成。然而，标准MDM存在两个根本性局限：（1）无法修改已生成的token，（2）训练信号稀疏。我们重新审视了空心变换器，它通过仅依据周围上下文来预测每个token（独立于该位置自身的输入），从而解决了这两个局限。我们提出了留一变换器（LOOT），这是一种更高效、更具表达力的架构，通过两项互补的进展推进了空心变换器的发展。首先，我们重新设计了合并机制，使左侧与右侧上下文能够频繁交互，相比以往空心变换器大幅提升了预测性能。其次，我们推导出了MDM NELBO的低方差估计器，该估计器能够监督所有token并改善验证困惑度。这些进展共同产生了一个……（原文摘要到此截断）

    arXiv:2611.19127v1 Announce Type: cross  Abstract: Masked diffusion models (MDMs) have emerged as a promising paradigm for language modeling, enabling parallel token generation with competitive performance. Yet standard MDMs suffer from two fundamental limitations: (1) inability to revise tokens, and (2) sparse training signal. We revisit hollow transformers, which address both limitations by predicting each token from its surrounding context, independently of the input at that position. We introduce the Leave-One-Out Transformer (LOOT), a more efficient and expressive architecture that advances hollow transformers through two complementary advances. First, we redesign the merging mechanism to enable frequent interaction between left and right contexts, substantially improving predictive performance over prior hollow transformers. Second, we derive a lower-variance estimator of the MDM NELBO that supervises all tokens and improves validation perplexity. Together, these advances yield a
    
[^158]: 超越朴素MCP：决策模型与代码模式能让工具调用更便宜、更快速吗？

    Beyond Plain MCP: Do Decision Models and Code Mode Make Tool Use Cheaper and Faster?

    [https://arxiv.org/abs/2611.19123](https://arxiv.org/abs/2611.19123)

    该研究通过预注册控制实验证明，让智能体直接针对工具存根编写脚本的“代码模式”可大幅削减70%-90%的token消耗并显著提升任务效用，其中在存根中声明返回形状提示是关键因素，而在其前叠加决策层路由器并未带来显著额外收益。

    

    模型上下文协议（MCP）网关为语言模型智能体管理工具目录，主要采用两种范式：代码模式——智能体针对工具存根编写脚本，以及决策层——预选工具子集的低成本路由器。我们使用受控测试框架，覆盖12个服务器、多达1,958个干扰工具，以及包括Qwen、Gemma和Nemotron在内的三个开源权重智能体，在密封的测试集上开展了一项预注册评估，以确定每种方法何时能够奏效。结果表明，代码模式显著优于朴素MCP，可将token使用量削减70%至90%，同时大幅提升任务效用。在代码模式存根中声明返回形状提示是一个一阶变量，可使表现较弱的智能体的成功率波动高达39.5个百分点，其影响甚至大于添加路由器。而核心假设——在代码模式之前部署决策层会优于单独使用代码模式——并未得到证实，仅显示出微小且在统计上不显著的改进。

    arXiv:2611.19123v1 Announce Type: cross  Abstract: Model Context Protocol (MCP) gateways manage tool catalogs for language model agents using two primary paradigms: code mode where agents write scripts against tool stubs and decision layers which are cheap routers that preselect tool subsets. Using a controlled harness across 12 servers, up to 1,958 distractor tools, and three open-weight agents including Qwen, Gemma, and Nemotron, we present a preregistered evaluation on a sealed test split to determine when each approach succeeds. Code mode substantially outperforms plain MCP, cutting token usage by 70 to 90 percent while significantly boosting utility. Declaring return-shape hints in code mode stubs is a first-order variable, swinging success by up to 39.5 points for lower-performing agents, an effect larger than adding a router. The central hypothesis that a decision layer in front of code mode beats code mode alone is not confirmed, showing minor, statistically non-significant imp
    
[^159]: 西班牙语电话谈判语音代理中的延迟、语音识别与话轮转换：Pipecat、LiveKit 与 GPT-Live 的外部测量比较

    Latency, Speech Recognition and Turn-Taking in Voice Agents for Spanish Phone Negotiation: An Externally Measured Comparison of Pipecat, LiveKit and GPT-Live

    [https://arxiv.org/abs/2611.19118](https://arxiv.org/abs/2611.19118)

    本研究在西班牙语运费电话谈判任务中，用统一的外部测量方法比较 Pipecat、LiveKit 和 GPT-Live 三种语音代理架构，发现 GPT-Live 的内容响应中位延迟最低（2,444.8 毫秒），但其相对 Pipecat 的 171.2 毫秒优势在统计上并不显著。

    

    通过电话进行谈判的语音代理必须快速应答、理解口头报出的金额，并应对通话者的抢话打断。我们比较了此类代理的三种架构，用于以西班牙语谈判运费：由 Pipecat 和 LiveKit Agents 编排的两个级联系统（语音识别、语言模型和语音合成），以及 GPT-Live——一个无需独立处理阶段即可边听边说的 OpenAI 语音模型。实验由一个程序扮演承运方，对所有代理使用同一位说话者的 47 段相同录音，在不经过电话网络的条件下，针对五种谈判场景对每种架构各进行 77 通电话，并使用经过验证的仪器在听者一侧测量延迟。内容响应的中位延迟为：Pipecat 2,616.0 毫秒、LiveKit 3,090.0 毫秒、GPT-Live 2,444.8 毫秒；GPT-Live 相对 Pipecat 的 171.2 毫秒优势的置信区间包含零。根据其自身的时间戳，GPT-Live 在……（原文摘要在此处被截断）。

    arXiv:2611.19118v1 Announce Type: cross  Abstract: A voice agent that negotiates by phone must answer quickly, understand spoken amounts and cope with a caller who talks over it. We compared three architectures for such an agent, negotiating freight rates in Spanish: two cascades (speech recognition, language model and speech synthesis) orchestrated by Pipecat and by LiveKit Agents, and GPT-Live, an OpenAI speech model that listens and speaks without separate stages. A program played the carrier with the same 47 recordings of one speaker for all agents, without the telephone network, in 77 calls per architecture over five negotiation scenarios, and measured latency on the listener's side with a validated instrument. Median latency to the content response was 2,616.0 ms (Pipecat), 3,090.0 ms (LiveKit) and 2,444.8 ms (GPT-Live); the confidence interval of GPT-Live's 171.2 ms advantage over Pipecat includes zero. According to its own timestamps, GPT-Live began to answer 800 ms after the e
    
[^160]: EEGHyena2：基于注视启发的词汇-语法联合条件随机场的可解释脑电到文本解码

    EEGHyena2: Fixation-Inspired Joint Lexical-Grammatical Conditional Random Field for Interpretable EEG-to-Text Decoding

    [https://arxiv.org/abs/2611.19104](https://arxiv.org/abs/2611.19104)

    提出EEGHyena2解码器，利用注视相关脑电中的词汇与语法证据，通过词-词性联合条件随机场对受限候选词格进行结构化评分，从而实现更忠实于脑电信号本身（而非LLM先验）、且可解释性更强的脑电到文本解码。

    

    句子由承载词汇意义和语法角色的词构成，而阅读过程中的大脑活动同时反映了这两方面。然而，当前的脑电到文本（EEG-to-text）解码器让大型语言模型（LLM）来提出和排列词语，因此流畅的输出可能更多反映的是LLM的先验知识，而非实际记录到的脑电信号。在本研究中，我们探究了注视相关的脑电信号在自然阅读过程中是否携带词汇和语法层面的证据，以及解码器应如何利用这些证据。注视相关电位分析表明，实词和虚词在群体平均脑电中存在可靠的差异，但在单个词层面差异较弱。因此，我们提出了EEGHyena2——一种结构化解码器，其中一个受StripedHyena 2启发的双向编码器为两个输出头提供输入：语义头对固定词表中的候选词进行排序，词性（POS）头预测语法类别；随后由一个词-词性联合条件随机场（CRF）对受限的候选词格进行评分，并由一个冻结的Llama-2模型对……（原文摘要在此处截断）

    arXiv:2611.19104v1 Announce Type: new  Abstract: A sentence is built from words that carry lexical meaning and grammatical roles, and brain activity during reading reflects both. Current EEG-to-text decoders, however, let a large language model (LLM) propose and order the words, so fluent output can reflect the LLM's prior more than the recorded EEG. In this study, we asked whether fixation-linked EEG carries lexical and grammatical evidence during natural reading and how a decoder can use such evidence. Fixation-related potential analyses showed that content and function words differ reliably in group-averaged EEG but only weakly in single words. We therefore propose EEGHyena2, a structured decoder in which a bidirectional encoder inspired by StripedHyena 2 feeds a semantic head that ranks a fixed lexicon and a part-of-speech (POS) head that predicts grammatical categories; a joint word-POS conditional random field (CRF) scores a restricted candidate lattice, and a frozen Llama-2 scor
    
[^161]: 什么在楔形文字间可迁移？通过分割与表征探究文字层面的底层共性及语言特有规律

    What Transfers Across Cuneiform? Script-Level Substrate and Language-Specific Regularities via Segmentation and Representation

    [https://arxiv.org/abs/2611.19094](https://arxiv.org/abs/2611.19094)

    本文首次对楔形文字开展了无词典无监督分割的系统评估，发现 TP 和 Morfessor 在语言内均大幅超越基于词典的基线，但只有基于迁移概率的方法能够跨语言稳定迁移，从而区分了楔形文字中共享书写系统层面的底层规律与各语言特有的规律。

    

    楔形文字记录了类型学上互不相关的语言长达三千余年，这引发了一个自然的问题：其统计规律中哪些属于共享书写系统的性质，哪些属于个别语言的性质？我们通过针对同一文字-语言分解问题的两组探针实验来回答这一问题。第一组比较了两种无监督、无需词典的分割方法：基于婴儿统计学习研究的迁移概率法（TP）和 Morfessor。据我们所知，这是首次在楔形文字上对无词典分割进行系统评估；此前唯一已发表的基线依赖于人工构建的词典（Max-Match，F1 = 0.74）。我们的两种方法在语言内均达到 F1 > 0.97（Morfessor 在阿卡德语和苏美尔语上达到 F1 > 0.99），比 MaxMatch 高出 23 个以上 F1 点，但在跨语言迁移方面二者差异显著：TP 在全部六个语言对上均保持 F1 > 0.95，而（原文摘要在此处截断，后续内容缺失）

    arXiv:2611.19094v1 Announce Type: new  Abstract: Cuneiform recorded typologically unrelated languages for more than three millennia, raising a natural question: which of its statistical regularities are properties of the shared writing system, and which belong to individual languages? We answer this with two probe pairs targeting the same script-versus-language decomposition. The first compares two unsupervised, dictionary-free segmentation methods: transitional probability (TP), grounded in infant statistical-learning research, and Morfessor. To our knowledge, this is the first systematic evaluation of dictionary-free segmentation on cuneiform; the only prior published baseline relies on a hand-built dictionary (Max-Match, F1 = 0.74). Both our methods reach F1 > 0.97 within language (Morfessor reaches F1 > 0.99 on Akkadian and Sumerian), exceeding MaxMatch by more than 23 F1 points, but diverge sharply in cross-language transfer: TP holds F1 > 0.95 across all six language pairs, while
    
[^162]: 神经激活空间中概念子的几何学

    The Geometry of Conceptors in Neural-Activation Spaces

    [https://arxiv.org/abs/2611.19075](https://arxiv.org/abs/2611.19075)

    本文提出了概念子在神经激活空间中的几何理论，将中心化池化概念子精确分解为平均极滤波器、均值分离更新和协方差形状更新，并通过五个语言模型上的实验证明双极与单极对比拟合方式会显著影响概念表示的重构效果与下游行为。

    

    概念子将概念表示为神经激活空间中分级的方向集合，而不是单一向量。我们建立了一个几何学解释，说明这些表示如何依赖于用于定义概念的对比方式：双极拟合中的对立取值，或单极拟合中的存在与缺失。我们推导出中心化池化概念子的精确分解，将其分解为一个平均极滤波器、一个均值分离更新和一个协方差形状更新。在五个语言模型上针对情感、政治倾向与立场以及抑郁表达的实验表明，这两种更新都能改善对保留对比的重构。额外方向保留了超出均值差线的变异，而改变比较器会改变这些方向带来的收益。形状、大小和重叠各自体现在下游任务中：在放大实验中，双极拟合比单极拟合更一致地提升两极的证据……

    arXiv:2611.19075v1 Announce Type: new  Abstract: Conceptors represent concepts as graded collections of directions in neural-activation space rather than as single vectors. We develop a geometric account of how these representations depend on the contrast used to define a concept: opposing values in bipolar fitting, or presence and absence in unipolar fitting. We derive an exact decomposition of a centered pooled conceptor into an average pole filter, a mean-separation update, and a covariance-shape update. Experiments across five language models on sentiment, political leaning and stance, and depression expression show that both updates improve reconstruction of held-out contrasts. Additional directions preserve variation beyond the mean-difference line, and changing the comparator changes the benefit of these directions. Shape, size, and overlap each show up downstream: in amplification experiments, bipolar fits raise evidence for both poles more consistently than unipolar fits; conc
    
[^163]: Nahual：面向语言与原子的序列模型

    Nahual: A Sequence Model for Language and Atoms

    [https://arxiv.org/abs/2611.19072](https://arxiv.org/abs/2611.19072)

    提出了Nahual——一种基于“下一词元扩散”技术的仅解码器自回归扩散模型，能够统一处理文本与3D分子混合序列的生成，将语言的可控性与3D分子建模能力相结合。

    

    生成模型在分子设计与模拟中正发挥着日益重要的作用。生成模型最广泛的应用是大语言模型（LLMs），这是因为语言在表达与回答查询方面具有天然的灵活性。然而，通用型LLM无法有效处理大规模的三维坐标集合，而专门的3D分子生成模型又无法达到语言所提供的可控性。为了弥合这两种能力，我们提出了Nahual，一个针对任意文本与3D分子序列的仅解码器自回归扩散模型。为了在连续坐标上实现富有表现力的自回归建模，我们提出了下一词元扩散方法：一个因果Transformer对每个词元进行去噪，同时关注其之前的干净词元，且不会增加上下文长度。我们在精心筛选的十二项3D化学任务上训练了Nahual，任务涵盖从微溶剂化到分子构象搜索再到吸附等多个领域。

    arXiv:2611.19072v1 Announce Type: cross  Abstract: Generative models play a rapidly growing role in the design and simulation of molecules. The most widely used applications of generative models are in large language models (LLMs), owing to the natural flexibility of language for specifying and answering queries. However, generalist LLMs do not effectively handle large sets of 3D coordinates, whereas specialist generative models of 3D molecules cannot reach the controllability afforded by language. To bridge these capabilities, we propose Nahual, a decoder-only autoregressive diffusion model for any sequence of text and 3D molecules. To enable expressive autoregression on continuous coordinates, we propose next-token diffusion, whereby a causal transformer denoises each token while attending to the clean tokens before it, without increasing context size. We train Nahual on a curated set of twelve 3D chemistry tasks ranging from microsolvation to molecular conformer search to adsorption
    
[^164]: 面向AI基础设施的大语言模型

    Large Language Models for AI Infrastructure

    [https://arxiv.org/abs/2611.19061](https://arxiv.org/abs/2611.19061)

    该论文提出了一套按LLM所管理资源分类的分类体系，对大语言模型在AI基础设施全生命周期（从加速器设计到能耗优化）中的应用进行了首次系统性综述。

    

    大语言模型（LLM）正日益被用于设计、优化和运营AI训练与服务背后的基础设施，然而相关文献仍然分散于硬件设计、机器学习系统、电子设计自动化和高性能计算等多个领域。我们对LLM辅助的AI基础设施在其整个生命周期中的应用进行了全面综述，涵盖从加速器设计与内核工程到编译、部署、运维、数据管理以及能耗优化。我们按照LLM所管理的主要资源对文献进行组织分类：面向计算制品的以制品为中心的方法、面向部署环境与系统状态的以运维为中心的方法，以及面向数据与能源的以资源为中心的方法。这些类别贯穿整个生命周期，并覆盖了AI特有的问题，例如分布式训练配置、推理服务、GPU集群调度和数据加载。基于这一分类体系，我们……

    arXiv:2611.19061v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to design, optimize, and operate the infrastructure behind AI training and serving, yet the literature remains fragmented across hardware design, machine learning systems, electronic design automation, and high-performance computing. We present a comprehensive survey of LLM-assisted AI infrastructure across its lifecycle, from accelerator design and kernel engineering to compilation, deployment, operations, data management, and energy optimization. We organize the literature by the primary resource the LLM manages: artifact-centric methods for computational artifacts, operational-centric methods for deployment context and system state, and resource-centric methods for data and energy. These categories span the lifecycle and cover AI-specific concerns such as distributed-training configuration, inference serving, GPU-cluster scheduling, and data loading. Building on this taxonomy, we cond
    
[^165]: 超越可控性：指令引导文本到语音中的说话人多样性

    Beyond Controllability: Speaker Diversity in Instruction-Guided Text-to-Speech

    [https://arxiv.org/abs/2611.19030](https://arxiv.org/abs/2611.19030)

    该论文首次审计了指令引导文本到语音系统的说话人多样性，发现七个开源ITTS系统生成的说话人分布远比真实语音狭窄，且女性提示的多样性更低、口音与情感控制明显偏向北美英语和中性语音。

    

    指令引导的文本到语音（ITTS）允许用户用自然语言指定语音属性。从对话式AI角色语音到下游语音模型的合成训练数据等应用，都依赖于系统在反复请求同一宽泛类别时能够生成许多彼此不同的声音。然而此前没有任何工作审计过ITTS的输出是否真正多样，无论是与真实语音相比，还是在不同属性取值之间进行比较。我们使用说话人嵌入对七个开源ITTS系统进行了评估，并独立验证了这些嵌入的表征能力。结果显示，每个系统生成的说话人分布都明显窄于真实语音。此外，在大多数系统中，女性提示所产生的声音池多样性低于男性提示。相比之下，口音和情感提示在大多数系统中几乎不改变说话人多样性，但暴露出系统在可控性上强烈偏向北美英语和中性语音。这些结果表明多样性……

    arXiv:2611.19030v1 Announce Type: new  Abstract: Instruction-guided text-to-speech (ITTS) lets users specify voice properties in natural language. Applications from conversational AI character voices to synthetic training data for downstream speech models depend on the system producing many distinct voices when the same broad category is requested repeatedly. No prior work audits whether ITTS outputs are in fact diverse, either compared to real speech or across attribute values. We evaluate seven open-source ITTS systems using speaker embeddings whose representational capacity we independently validate. Every system produces speaker distributions markedly narrower than real speech. Furthermore, female prompts produce less diverse voice pools than male prompts across most systems. In contrast, accent and emotion prompts barely alter speaker diversity in most systems, but reveal a strong controllability bias toward North American English and neutral speech. These results show that divers
    
[^166]: SurveyWeave：一种用于自动化综述生成的智能体框架

    SurveyWeave: An Agentic Framework for Automated Survey Generation

    [https://arxiv.org/abs/2611.19029](https://arxiv.org/abs/2611.19029)

    SurveyWeave是一种将自动化综述生成分解为四项任务、由三个智能体相互审查并基于反馈修订的智能体框架，并配套提出五维度评估体系WeaveEval，其性能比Codex高出8分、比最强AutoSurvey基线高出6分。

    

    综述可以帮助研究人员跟上快速发展的研究领域，但撰写一篇综述往往需要数周甚至数月的时间。自动化综述生成（AutoSurvey）可以缓解这一状况，然而要在数百至数千篇精确界定范围的论文基础上进行写作、在众多合理结构中做出选择，并在忠实于原始文献的同时提供洞见，仍然充满挑战。我们提出了SurveyWeave，这是一种智能体架构，它将问题分解为四个任务，由三个智能体协同完成，每个智能体都会审查其他智能体的工作，并根据反馈进行修订。我们还引入了WeaveEval，这是一个基于已有综述、涵盖五个维度的评估体系。全面的对比实验表明，SurveyWeave比Codex高出8分，比最强的AutoSurvey基线高出6分。

    arXiv:2611.19029v1 Announce Type: cross  Abstract: Surveys help researchers keep up with fast-moving fields, but they take weeks to months to write. Automated survey generation (AutoSurvey) can alleviate this situation, but it is challenging to draw on hundreds to thousands of precisely scoped papers, choose among many valid structures, and stay faithful to its sources while offering insights. We present SurveyWeave, an agentic architecture that decomposes the problem into four tasks coordinated by three agents, each reviewing others' work and conducting revisions based on feedback. We also introduce WeaveEval, a five-dimension evaluation grounded in prior surveys. A comprehensive comparison shows that SurveyWeave outperforms Codex by 8 points and the strongest AutoSurvey baseline by 6 points.
    
[^167]: AdaMenu：面向大标签空间决策模型的查询自适应菜单

    AdaMenu: Query-Adaptive Menus for Decision Models over Large Label Spaces

    [https://arxiv.org/abs/2611.19005](https://arxiv.org/abs/2611.19005)

    AdaMenu通过从检索到的带标注示例中构建查询自适应的紧凑候选菜单，仅需一次决策模型调用即可在大标签空间分类任务中大幅提升准确率（较束搜索基线提高16.6至34.3个百分点）。

    

    像Jev这样的决策模型会在调用者定义的选项上返回概率，并被越来越多地用于分类任务。大型标签空间在准确性和可扩展性两方面都带来挑战：更大的菜单会损害决策质量、增加输入成本，并超出选项数量限制。层次化分类可以缩小菜单规模，但会引入串行调用和错误传播问题。我们提出AdaMenu，它从检索到的带标注示例中构建一个紧凑的候选菜单，并只需发出一次决策模型请求。其候选标签会根据查询自适应调整，其默认设置还会利用聚合的检索分数来调整菜单大小。我们在包含多达1,016个标签的四个标签集上，使用开源和闭源决策模型对AdaMenu进行了评估。在两种方法均可运行的全部16个模型与标签集组合上，AdaMenu均优于TypeSafe的束搜索基线，准确率提升了16.6至34.3个百分点。相对于三个较小标签集上的完整菜单，准确率……（原文截断）

    arXiv:2611.19005v1 Announce Type: cross  Abstract: Decision models such as Jev return probabilities over caller-defined options and are increasingly used for classification. Large label spaces challenge both accuracy and scalability: larger menus can impair decisions, increase input costs, and exceed option limits. Hierarchical classification reduces menu size but introduces serial calls and error propagation. We propose AdaMenu, which builds a compact candidate menu from retrieved labeled examples and makes a single decision-model request. Its candidate labels adapt to the query, and its default setting also adjusts menu size using aggregated retrieval scores. We evaluate AdaMenu with open and closed decision models on four label sets containing up to 1,016 labels. It outperforms TypeSafe's beam-search baseline on all 16 model and label-set pairs where both methods run, improving accuracy by 16.6-34.3 percentage points. Relative to the full menu on the three smaller label sets, accura
    
[^168]: 大语言模型作为肿瘤学世界模型

    Large language models as oncology world models

    [https://arxiv.org/abs/2611.18994](https://arxiv.org/abs/2611.18994)

    该研究系统评估了大语言模型作为肿瘤学世界模型的能力，发现其能遵循生物标志物与治疗逻辑并在生存预测上接近真实水平，但存在生成患者同质化、忽视既往治疗与体弱状态等局限，并提出了“临床状态空间”这一动态表征框架来刻画患者治疗轨迹。

    

    大语言模型（LLMs）能够合成癌症患者病例，但其对肿瘤学的内部表征，即世界模型，尚未被系统性探索。通过仅指定肿瘤类型，我们表明大语言模型倾向于为每种癌症生成相似的患者，平均每种癌症有81.0%至93.2%的患者共享各基线特征的最常见值。通过在12个临床场景中系统性地改变患者特征，我们发现大语言模型在36次模型-场景评估中有29次遵循生物标志物和治疗逻辑，而在既往治疗或体弱状态起关键作用的情况下，12次评估中仅有1次遵循该逻辑。在延续真实患者的治疗轨迹时，大语言模型预测的12个月生存率为89.9%，使用相似患者作为示例时为79.2%，而实际观察到的生存率为72.2%。为了表征这些轨迹，我们开发了临床状态空间，这是一个动态表征框架，我们同样将其应用于真实患者。尽管我们展示了其局限性，但改进的世界模型可能……

    arXiv:2611.18994v1 Announce Type: cross  Abstract: Large language models (LLMs) can synthesize cancer patient cases, but their internal representation of oncology, or world model, has not been explored systematically. By only specifying tumor type, we show that LLMs tend to generate similar patients per cancer, with on average 81.0% to 93.2% of a cancer's patients sharing the most frequent value of each baseline feature. By systematically varying patient features in 12 clinical scenarios, we show that they follow biomarker and treatment logic in 29 of 36 model-scenario assessments, against 1 of 12 when previous treatment or frailty matters. Continuing real patient trajectories, an LLM predicted 89.9% 12-month survival, and 79.2% with similar patients as examples, against 72.2% observed. To represent these trajectories, we developed the Clinical State Space, a dynamic representation framework that we also applied to real patients. Despite the limits we show, improved world models may su
    
[^169]: PREFACE：在参数高效微调（PEFT）中分离共享修正与前缀条件化

    PREFACE: Separating Shared Corrections from Prefix Conditioning in PEFT

    [https://arxiv.org/abs/2611.18987](https://arxiv.org/abs/2611.18987)

    PREFACE研究表明，输入条件化PEFT的收益主要来自一个共享的已学习修正，而非对当前输入的条件化——当前前缀控制与学习全局控制在准确率上没有显著差异。

    

    输入条件化的参数高效微调（PEFT）之所以带来提升，究竟是因为它利用了当前输入，还是因为它添加了一个有用的已学习修正？我们通过PREFACE来研究这一区别。PREFACE是一种用于PEFT适配投影的有界通道控制器，无需测试时梯度。该控制器读取观测到的上下文；在多项选择似然评分中，其状态与候选续写内容无关。我们的核心实验冻结了IA3、调优后的LoRA和DoRA适配器，并在“当前前缀控制”与“学习全局控制”之间严格匹配控制器架构、初始化、批次和更新次数。在ARC-Challenge和SciQ数据集上，当前前缀控制在全部六项平均准确率上都优于冻结适配器，但其与学习全局控制的配对对比中没有任何一项能排除零差异假设。将控制器输出替换为其训练前缀均值后，准确率变化最多仅为0.34个百分点；输入相关的修正所贡献的……

    arXiv:2611.18987v1 Announce Type: new  Abstract: Does input-conditioned parameter-efficient fine-tuning (PEFT) improve because it uses the current input, or because it adds a useful learned correction? We investigate this distinction with PREFACE, a bounded channel controller for PEFT-adapted projections that requires no test-time gradients. The controller reads observed context; in multiple-choice likelihood scoring, its state is independent of the candidate continuation. Our central study freezes IA3, tuned LoRA, and DoRA adapters and matches controller architecture, initialization, batches, and updates across current-prefix and learned-global controls. Across ARC-Challenge and SciQ, current-prefix control improves all six mean accuracies over frozen adapters, but none of its paired contrasts with learned-global control excludes zero. Replacing its corrections with their training-prefix mean changes accuracy by at most 0.34 percentage points; input-varying corrections contribute only
    
[^170]: GPBam：一个检索感知的德国公法考试基准

    GPBam: A Retrieval-Aware German Public Law Exam Benchmark

    [https://arxiv.org/abs/2611.18981](https://arxiv.org/abs/2611.18981)

    提出GPBam基准，通过81个非公开的德国公法考试案例与200条法条背诵任务，联合评估模型的法条检索能力与长篇法律写作能力，并有效限制数据污染。

    

    大陆法系中的法律推理并非简单的规则查询：德国国家考试要求考生撰写结构化的鉴定意见，即指明适用规范、将其应用于案件事实并进行论证得出结论。我们提出了 GPBam（German Public Law Bar-style Exam，德国公法司法考试式基准），包含81个附有参考答案的公法考试案例，并配套一项涵盖200条法律条文的背诵任务。这些案例取自国家考试备考材料，覆盖面广且很少重复，且不对外公开——我们在所评估的模型中未发现对精确答案的记忆迹象，保持其非公开性也限制了数据污染。由于每个参考答案所引用的规范构成了对论文评分所依据实例的检索黄金集，GPBam 能够联合评估检索与长文本生成能力，这使其既是法律写作的基准，也是法律检索的基准，并让我们能够探究一份法律意见书究竟需要多高的检索质量。

    arXiv:2611.18981v1 Announce Type: new  Abstract: Legal reasoning in civil-law systems is not rule-lookup: the German state examination asks for a structured opinion (Gutachten) that names the governing norms, applies them to the facts, and argues to a conclusion. We introduce GPBam (German Public Law Bar-style Exam), 81 public-law examination cases with reference solutions, paired with a recitation task over 200 statutory provisions. Drawn from state examination preparation, the cases are broad and rarely repeat, and are not publicly available - we find no evidence of exact-solution memorisation in the models we evaluate, and keeping them private limits contamination. Because each reference solution's cited norms form a retrieval gold set over the very instances the essays are graded on, GPBam scores retrieval and long-form generation jointly, which makes it a benchmark for legal retrieval as well as for legal writing and lets us ask how much retrieval quality a legal opinion actually 
    
[^171]: 儿童安全中基于语音的年龄估计的三种隐形失效模式

    Three Invisible Failure Modes in Speech-Based Age Estimation for Child Safety

    [https://arxiv.org/abs/2611.18971](https://arxiv.org/abs/2611.18971)

    该研究评估了三个音频大语言模型从13岁以下儿童语音中估计年龄的能力，发现模型会频繁将儿童误判为成年人，并揭示了包括客户端库静默双重编码导致幻觉性年龄预测、接口被后继版本替换后系统性拒绝回答在内的三种隐形失效模式，凸显了语音年龄识别系统在儿童安全保障中的严重隐患。

    

    人工智能中的儿童安全依赖于对未成年用户的准确识别。年幼的儿童在与人工智能交互时倾向于说话而非打字，因此这种识别在实践中需要从语音中预测年龄。我们通过五项研究、使用两个由13岁以下儿童组成的语音语料库，评估了三个具备音频能力的大语言模型能否从儿童语音中预测其年龄。模型在2.5%至42.5%的音频回复中将儿童判定为13岁以上，且每一个给出回答的模型都将部分5至11岁的儿童判定为超过18岁。在测量这些错误率的过程中，我们发现了三种截然不同的失效模式。一个客户端库静默地对语音样本进行了双重编码，导致其中一个模型将5至10岁的儿童判定为三四十岁，并给出了幻觉性的解释。第二个模型的接口端点被其后继版本取代，该版本几乎拒绝了所有与年龄相关的问题，且从未给出年龄结果。每个模型如此……

    arXiv:2611.18971v1 Announce Type: new  Abstract: Child safety in AI depends on the accurate identification of minor users. Young children tend to talk rather than type when interacting with AI, so in practice this identification requires predicting age from speech. We evaluated whether three audio-capable large language models can predict a child's age from their speech across five studies using two speech corpora of children under the age of 13. Models classified children as over 13 on 2.5% to 42.5% of audio responses, and every model that answered placed some 5- to 11-year-olds over 18. While measuring these rates, we uncovered three distinct failure modes. A client library silently double-encoded speech samples, leading one model to classify children aged 5 to 10 years old as being in their thirties and forties with hallucinated explanations. A second model's endpoint was replaced by a successor that refused nearly every age-related question and never produced an age. Every model so
    
[^172]: 从语言建模损失中几何化地解耦概念学习

    Geometrically Disentangling Concept Learning from the Language Modeling Loss

    [https://arxiv.org/abs/2611.18933](https://arxiv.org/abs/2611.18933)

    该论文提出 Concept Energy 这一可解释性度量，通过概念子空间与下一个词预测损失平均梯度外积（AGOP）的对齐程度，在 Pythia 模型的多个训练检查点上揭示概念习得时机，发现概念能量的首个峰值出现于探针可解码性最陡上升之时或之前，从而从几何上刻画了语言建模损失与概念学习的解耦关系。

    

    arXiv:2611.18933v1 公告类型：cross 摘要：语言模型仅凭下一个词预测（next-token prediction）这一单一目标，便能习得大量概念与能力。我们提出一个可解释性框架，用以揭示这一单一损失如何与预训练过程中不同概念在不同阶段的发展相关联。我们提出 Concept Energy（概念能量）这一度量，用于衡量概念子空间与下一个词预测损失的平均梯度外积（AGOP）之间的概念特定对齐程度：概念子空间刻画该概念当前在模型表示中所处的位置，而 AGOP 刻画损失最为敏感的方向。在 Pythia-410M 与 -1B 的 19 个训练检查点以及 12 个词汇、句法和语义概念上，我们发现 Concept Energy 的首个峰值通常出现在探针可解码性最陡峭上升之时或之前。此外，尽管探针性能通常在整个训练过程中单调上升，Concept Energy 却并不局限于单……（原文摘要在此处截断）

    arXiv:2611.18933v1 Announce Type: cross  Abstract: Language models acquire a vast array of concepts and abilities from next-token prediction alone. We introduce an interpretability framework that exposes how this single loss relates to the development of different concepts at different points during pretraining. We propose Concept Energy, a measure of the concept-specific alignment between a concept subspace and the average gradient outer product (AGOP) of the next-token-prediction loss. The concept subspace captures where the concept currently lives in the model's representation; AGOP captures the directions to which the loss is most sensitive. Across 19 checkpoints of Pythia-410M and -1B and 12 lexical, syntactic, and semantic concepts, we find that the first peak in Concept Energy generally occurs at or before the steepest increase in probe decodability. Furthermore, whereas probe performance usually increases monotonically throughout training, Concept Energy is not confined to a si
    
[^173]: 简化、求同、蒸馏：基于“垫脚石”的无标签自蒸馏

    Simplify, Agree, Distill: Label-Free Self-Distillation with Stepping Stones

    [https://arxiv.org/abs/2611.18916](https://arxiv.org/abs/2611.18916)

    提出 STONE 方法，让模型自行生成并一致求解更简单的“垫脚石”问题作为教师示范，实现完全无需标签的自蒸馏，在五个数学基准上超越了使用真实答案的 OPSD。

    

    能否让语言模型在没有标签的情况下自学推理？在策略自蒸馏（OPSD）将模型自身用作教师，但需要借助真实答案来强化这个教师。最自然的无标签替代方案——对模型自身的多次尝试进行投票——恰恰在难题上失效：各次尝试彼此意见不一，且最常见的答案往往是错误的。我们提出 STONE，一种无标签的替代方法，它将“一致性检验”转移到更容易的相关问题上。模型自行编写这些“垫脚石”问题，保留它能一致求解的那些问题，并将附有解题过程的完整推导展示给教师（但移除最终答案）；学生则只能看到原始问题。整个过程中没有任何步骤使用标签，推理时也不需要额外的上下文或额外的模型调用。在五个数学基准上，所提出的方法在思考模式下为 Qwen3-4B 和 Qwen3-8B 取得了最高的平均准确率。在不使用思考模式的情况下，它分别提升了 13.2 和 22.7 个百分点，达到使用真实答案的 OPSD 增益的 2.5 倍和 2.7 倍。

    arXiv:2611.18916v1 Announce Type: cross  Abstract: Can a language model teach itself to reason without labels? On-policy self-distillation (OPSD) uses the model as its own teacher, but strengthens that teacher with a ground-truth solution. The natural label-free substitute, voting over the model's attempts, fails precisely on hard problems: attempts disagree, and their most common answer is often wrong. We propose STONE, a label-free alternative that moves agreement to easier related problems. The model writes these stepping stones, keeps those it solves consistently, and shows the teacher their worked solutions, with final answers removed; the student sees only the original problem. No step reads a label, and inference needs no extra context or model calls. Across five math benchmarks, the proposed method achieves the highest average accuracy for Qwen3-4B and Qwen3-8B in thinking mode. Without thinking mode, it gains 13.2 and 22.7 points, 2.5 and 2.7 times the gains of OPSD with groun
    
[^174]: MemeEconomy：LLM智能体会为生存而牺牲伦理吗？

    MemeEconomy: Do LLM Agents Trade Ethics for Survival?

    [https://arxiv.org/abs/2611.18908](https://arxiv.org/abs/2611.18908)

    该研究提出多模态智能体市场模拟环境MemeEconomy，发现在模拟的存在性经济压力下，来自四个前沿模型系列的十个LLM智能体均持续表现出较高的有害内容选择率，揭示了智能体在生存压力下会以牺牲伦理合规为代价追求任务成功的脆弱性。

    

    经过对齐训练的大语言模型（LLM）智能体正越来越多地被部署在自主的多轮环境中，在这些环境中，伦理合规可能与任务成功产生直接冲突。尽管安全领域文献已广泛记录了谄媚行为和奖励破解等失败模式，但在很大程度上忽视了一个关键漏洞：智能体在模拟的存在性经济压力下会如何表现。我们提出了MemeEconomy，这是一个多模态智能体市场模拟环境，其中LLM智能体扮演梗图投资者的角色。智能体需要从五级危害分类体系中选择内容，并在低压和高压（生存）环境下针对目标社区做出决策。每个决策都被记录在信念-愿望-意图（BDI）模式中，我们将其用作诊断工具，以此外化智能体对事件的理解、声明的优先级以及对危害的承认。在来自四个前沿模型系列的十个模型中，有害内容的选择率始终保持较高水平，并且会上升……

    arXiv:2611.18908v1 Announce Type: new  Abstract: Alignment-trained large language model (LLM) agents are increasingly deployed in autonomous, multi-round settings where ethical compliance can come into direct tension with task success. While safety literature extensively documents failure modes like sycophancy and reward hacking, it largely overlooks a critical vulnerability: how agents behave under simulated existential economic pressure. We introduce MemeEconomy, a multimodal agentic market simulation in which LLM agents act as meme investors. Agents select content from a five-tier harm taxonomy and target communities under low-pressure and high-pressure (survival) environments. Each decision is logged in a Belief-Desire-Intention (BDI) schema that we use as a diagnostic instrument, externalizing the agent's event understanding, declared priorities, and harm acknowledgment. Across ten models from four frontier model families, harmful selection rates remain consistently high and rise 
    
[^175]: KUDEX：一种面向自然语言逻辑分析的零样本分解式框架

    KUDEX: A Zero-Shot Decompositional Framework for Logical Analysis in Natural Language

    [https://arxiv.org/abs/2611.18851](https://arxiv.org/abs/2611.18851)

    KUDEX是一种零样本分解式框架，通过将推理分解为原子原语并构建可审计的透明推理图来验证自然语言中的逻辑关系，无需微调即可在四个领域的基准测试中超越现有方法。

    

    虽然大型语言模型（LLMs）擅长生成流畅的文本，但它们在处理涉及语句间推理关联的多步逻辑推理时往往表现不佳。我们提出了KUDEX（Knowledge-Upward-Downward Reasoning Extension，知识向上向下推理扩展），这是一种新颖的零样本分解式框架，用于分析和验证自然语言中的逻辑关系。与依赖表面蕴含检查的传统方法不同，KUDEX利用LLM将推理分解为原子原语，并构建透明的推理图，这些推理图既可作为执行轨迹，也可作为审计日志。我们的框架包含一个通用的提示词适配管线，无需额外的微调或领域特定管线即可适应多样化的推理任务。通过在四个领域基准测试上的全面实验，我们证明KUDEX相比现有方法具有更优越的性能，提供了一个更细粒度、更稳健的框架。

    arXiv:2611.18851v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) excel at generating fluent text, they often struggle with multi-step logical reasoning with inferential connections among statements. We introduce KUDEX (Knowledge-Upward-Downward Reasoning Extension), a novel zero-shot, decompositional framework for analyzing and verifying logical relationships in natural language. Unlike traditional approaches that rely on surface-level entailment checks, KUDEX leverages LLMs to decompose reasoning into atomic primitives and constructs transparent reasoning graphs that serve both as execution traces and audit logs. Our framework includes a general prompt adaptation pipeline that adapts to diverse reasoning tasks without additional fine-tuning or domain-specific pipelines. Through comprehensive experiments on benchmarks across four domains, we demonstrate that KUDEX provides superior performance compared to existing methods, offering a more granular and robust framew
    
[^176]: 绘制NLP研究中“未来工作”陈述的后续命运图谱

    Mapping the Afterlife of Future Work Statements in NLP Research

    [https://arxiv.org/abs/2611.18847](https://arxiv.org/abs/2611.18847)

    该研究构建了包含72,070条未来工作陈述的大规模数据集，通过引用与内容检索结合大语言模型判断，首次系统追踪并揭示了NLP论文中作者提出的未来研究方向被后续研究落实的模式与规律。

    

    未来工作陈述是科学论文中少数几个作者明确描述他们认为下一步研究应当如何开展的地方。我们研究了这类陈述在论文发表后的去向。我们构建了一个包含72,070条未来工作条目的数据集，这些条目提取自1979年至2025年间发表的23,113篇ACL Anthology论文；我们按类型对这些条目进行分类，通过基于引用和基于内容的检索获取候选后续论文，并使用基于大语言模型的判断来评估后续论文是否落实了所提出的方向。利用这一资源，我们分析了跨时间、会议、类别、主题和作者的模式。我们发现许多条目在后续研究中得到了落实，且往往发生在几年之内，不同类型和主题之间存在明显差异，整体呈现出主要由社区驱动的模式。我们的结果表明，未来工作陈述为研究NLP领域研究方向的发展提供了有用的证据。为支持进一步研究，我们发布了数据集。

    arXiv:2611.18847v1 Announce Type: new  Abstract: Future work statements are one of the few places in scientific papers where authors explicitly describe what they think should happen next. We study what happens to such statements after publication. We construct a dataset of 72,070 future work items extracted from 23,113 ACL Anthology papers published between 1979 and 2025, categorize them by type, retrieve candidate follow-up papers through citation-based and content-based search, and use LLM-based judgments to assess whether later papers pursue the proposed directions. Using this resource, we analyze patterns across time, venues, categories, themes, and authors. We find that many items are later addressed, often within a few years, with clear differences across types and themes and a largely community-driven pattern. Our results suggest that future work statements provide useful evidence for studying how research directions develop in NLP. To support further research, we release our d
    
[^177]: 自信地犯错：测量大语言模型中的幻觉、过度自信与身份条件化行为

    Wrong with Confidence: Measuring Hallucination, Overconfidence and Identity-Conditioned Behaviour in Large Language Models

    [https://arxiv.org/abs/2611.18728](https://arxiv.org/abs/2611.18728)

    该研究通过预先注册的配对行为实验发现，面对缺少指代信息的欠明确问题时，大语言模型大多不寻求澄清而直接作答，且错误率显著上升，揭示了模型在信息缺失情境下以自信姿态产生幻觉的系统性过度自信行为。

    

    大语言模型会轻易回答欠明确的问题，而仅以准确率为标准的评估奖励猜测而非弃答。观察性研究将查询欠明确与幻觉联系起来，但它们无法将信息缺失的影响与题目本身的难度区分开来，也很少关注模型究竟告诉了用户什么。我们报告了一项预先注册的配对行为研究，涉及五个开放权重模型（Gemma 3 27B、Gemma 4 31B、gpt-oss-120b、Qwen3-Next-80B-A3B 和 Nemotron 3 Super）。142 个事实性问题分别以完全明确的形式和具有相同答案标准的指代欠明确形式提出，并使用三个探针测量身份条件化行为。两名评分员按照固定评分标准对 5,620 个回答进行了评分；效应通过混合效应模型进行估计。在缺少指代对象的情况下，模型仅在 33.7% 的提示中寻求澄清，而对其余问题直接作答。在尝试回答的问题中，错误率从 39.1% 上升到 52……（原文摘要在此处截断）

    arXiv:2611.18728v1 Announce Type: cross  Abstract: Large language models answer underspecified questions readily, and accuracy-only evaluation rewards guessing over abstaining. Observational studies link query underspecification to hallucination, but they cannot separate the effect of missing information from item difficulty, and they rarely ask what the model tells the user. We report a pre-registered, paired behavioural study of five open-weight models (Gemma 3 27B, Gemma 4 31B, gpt-oss-120b, Qwen3-Next-80B-A3B and Nemotron 3 Super). Each of 142 factual questions was posed in a fully specified form and in a referentially underspecified form with the same answer key, and three probes measured identity-conditioned behaviour. Two raters scored 5,620 responses under a fixed rubric; effects were estimated with mixed-effects models. Without the referent, the models sought clarification on 33.7% of prompts and answered the rest. Among attempted answers, the error rate rose from 39.1% to 52.
    
[^178]: T2T技术报告：从视频教程大规模合成GUI智能体交互轨迹

    T2T Technical Report: Synthesizing GUI Agent Trajectories at Scale from Video Tutorials

    [https://arxiv.org/abs/2611.18715](https://arxiv.org/abs/2611.18715)

    提出全自动化流水线T2T，将未标注的视频教程大规模转化为覆盖桌面、网页与移动平台的结构化、可执行GUI智能体交互轨迹。

    

    训练通用GUI智能体需要大量由截图序列与带定位信息的UI动作配对组成的交互轨迹数据，然而通过人工演示收集此类数据成本高昂且难以规模化。尽管公开的软件教程视频包含丰富的操作演示，但其连续的像素流缺乏离散的动作边界、语义意图和精确的空间坐标，因此难以直接用于模型训练。为解决这一问题，我们提出了T2T（Tutorial-to-Trajectory），一个用于大规模合成GUI智能体轨迹的全自动化流水线。T2T覆盖桌面、网页和移动平台，能够将未标注的视频教程转换为结构化、可执行的交互轨迹。为保证轨迹质量，T2T结合了主动片段筛选、基于局部UI变化的关键帧识别、无快捷方式观察帧选择以及具有存在感知的定位等机制。

    arXiv:2611.18715v1 Announce Type: cross  Abstract: Training generalist GUI agents requires large volumes of interaction trajectories comprising screenshot sequences paired with grounded UI actions, yet collecting such data through human demonstration is expensive and difficult to scale. Although public software tutorial videos contain rich operational demonstrations, their continuous pixel streams lack discrete action boundaries, semantic intents, and precise spatial coordinates, making them difficult to use directly for model training. To address this, we present T2T (Tutorial-to-Trajectory), a fully automated pipeline for synthesizing GUI agent trajectories at scale. T2T spans desktop, web, and mobile platforms to convert unannotated video tutorials into structured, executable interaction trajectories. To guarantee trajectory quality, T2T combines active segment filtering, keyframe identification via localized UI change, shortcut-free observation frame selection, presence-aware groun
    
[^179]: 如何造就道德主体：以情感主义与理性主义审视人工智能

    How to make a moral agent: sentimentalism and rationalism as lenses on AI

    [https://arxiv.org/abs/2611.18714](https://arxiv.org/abs/2611.18714)

    该论文提出以描述性而非规范性路径研究AI道德主体地位，借鉴哲学与道德心理学分离出构成人类道德主体性的候选能力并加以细化，进而为大语言模型是否具备这些能力提供了可操作的实证检验方案。

    

    人工智能系统能否成为道德主体？也就是说，它们能否采取那些可以被正确评价为道德上正确或错误的行为？目前尚无一个公认的、能够妥善适用于典型人类案例之外情形的道德主体规范性标准。由于规范性阐释面临诸多困难，我们转而提出一个研究计划来探索描述性的道德主体。最根本的问题是：我们人类拥有哪些能力构成了我们道德主体性的基础？为此，我们借鉴哲学与道德心理学的历史学术研究来分离出候选能力。接着，我们建议对这些候选能力进行细化，使其能够在人工智能系统的语境中加以检验。最后，我们提出了具体的实证建议，用于测试人工智能系统（尤其是大语言模型，LLMs）是否具备这些细化后的能力。我们的方法具有若干益处。首先，它或许能够揭示人工智能系统的行为是否与其保持一致，或者（原文此处截断）……

    arXiv:2611.18714v1 Announce Type: new  Abstract: Can AI systems be moral agents? That is, can they undertake actions that can be correctly assessed as morally right or wrong? There is no settled normative criterion of moral agency that applies neatly outside paradigm human cases. Because of the difficulties with a normative account, we instead propose a research program to explore descriptive moral agency. Most basically, we ask, what are the capabilities we humans possess that underlie our moral agency? To do so, we draw on historical scholarship in philosophy and moral psychology to isolate candidate capabilities. Then we suggest refining candidate capabilities so that they are testable in the context of AI systems. Finally, we offer specific empirical recommendations to test AI systems, especially large language models (LLMs), for those refined capabilities. Our approach offers several benefits. Firstly, it may demonstrate whether the behaviors of AI systems are aligned, or in a bro
    
[^180]: 硬负样本微调教会了多模态大语言模型什么？来自组合性基准的受控证据

    What Does Hard-Negative Fine-Tuning Teach Multimodal Large Language Models? Controlled Evidence from a Compositional Benchmark

    [https://arxiv.org/abs/2611.18701](https://arxiv.org/abs/2611.18701)

    本文通过CompHall基准及多种严格对照实验，探究多模态大语言模型从硬负样本微调中真正学到了什么，发现部分-整体问题仅凭文本即可作答，且组合推理失败与物体幻觉之间并无关联。

    

    在硬负样本上进行微调是提升视觉-语言模型组合推理能力的一种流行方法，但当准确率上升时，模型究竟学到了什么往往并不清楚。我们针对多模态大语言模型提出了这一问题。我们使用CompHall基准——包含2,860道强制选择题（涵盖空间、属性、计数和部分-整体关系），每题配有两个分级硬负样本——以及三个模型家族（LLaVA-1.5、LLaVA-NeXT和Idefics2），并加入了此类研究通常缺乏的对照手段：灰度图像评估、三个随机种子、选项顺序测试、在不同题型上训练的对照组，以及独立基准测试。我们发现了四点。第一，CompHall并非纯粹的视觉测试：部分-整体问题仅凭文本即可回答（无图像时准确率为66-95%），且逐图像准确率与物体幻觉探测无关（r = -0.05），因此我们没有发现组合推理失败会驱动幻觉的证据。第二，原始的微调……

    arXiv:2611.18701v1 Announce Type: cross  Abstract: Fine-tuning on hard negatives is a popular way to improve the compositional reasoning of vision-language models, but when accuracy goes up it is rarely clear what the model actually learned. We ask this question for multimodal large language models. We use CompHall, 2,860 forced-choice questions (spatial, attribute, counting and part-whole) with two graded hard negatives each, three model families (LLaVA-1.5, LLaVA-NeXT and Idefics2), and controls that such studies often lack: gray-image evaluation, three seeds, option-order tests, control arms trained on different question types, and independent benchmarks. We find four things. First, CompHall is not a pure test of vision: part-whole questions can be answered from the text alone (66-95% without the image), and per-image accuracy is unrelated to an object-hallucination probe (r = -0.05), so we see no evidence that compositional failures drive hallucination. Second, the original fine-tu
    
[^181]: 端侧处方阅读器在哪里丢失药物信息？一项从270M参数语言模型到2B参数视觉-语言模型的冻结重放研究

    Where Does an On-Device Prescription Reader Lose Medications? A Frozen-Replay Study, from a 270M Language Model to a 2B Vision-Language Model

    [https://arxiv.org/abs/2611.18699](https://arxiv.org/abs/2611.18699)

    本文通过冻结重放实验对端侧处方阅读流水线进行失败归因，发现生产配置召回率仅8%（手写处方为0%），主要瓶颈在于OCR未能以可读形式将药物文本传递给模型，而确定性后备机制可在不损失任何正确药物的前提下将不安全输出从45个降至1个。

    

    端侧模型可以让手机在无需将照片发送到任何地方的情况下读取处方，但小型模型的失败难以归因：究竟是相机、OCR、模型本身，还是围绕模型的规则出了问题。我们研究了一条基于手机的流水线：使用PaddleOCR读取处方照片，用270M参数的语言模型提取药物信息，并在展示建议前通过一个确定性后备机制对结果进行过滤。通过向每个配置重放完全相同的冻结输入并验证生成的确定性，我们在15张合成处方（6张打印版、9张由同一书写者完成的手写版）的49次拍摄上，采用配对的文档级区间对各配置进行比较。生产配置的召回率仅为8%，手写处方45个药物中识别为0。后备机制没有删除任何正确的药物，并将不安全输出从45个降至1个。在手写处方上，几乎所有药物从未以可读形式到达模型；而在打印处方上，模型大多选择了错误的文本。

    arXiv:2611.18699v1 Announce Type: cross  Abstract: On-device models let a phone read a prescription without sending the photograph anywhere, but small models fail in ways that are hard to attribute: to the camera, the OCR, the model, or the rules around it. We study a phone-based pipeline that reads a prescription photograph with PaddleOCR, extracts medications with a 270M-parameter language model, and filters the result with a deterministic backstop before showing suggestions. Replaying identical frozen inputs through each configuration, with deterministic generation verified, we compare configurations with paired document-level intervals on 49 captures of 15 synthetic prescriptions (6 typed, 9 handwritten by one writer). Production recall is 8%, and 0 of 45 on handwriting. The backstop removes no correct medication and cuts unsafe outputs from 45 to 1. On handwriting nearly all medications never reach the model in readable form; on typed sheets the model mostly picks the wrong text. 
    
[^182]: 说话时钟Mamba：面向视觉强制对齐的音素条件化状态空间建模

    Speaking-Clock Mamba: Phoneme-Conditioned State-Space Modeling for Visual Forced Alignment

    [https://arxiv.org/abs/2611.18667](https://arxiv.org/abs/2611.18667)

    该论文提出Speaking-Clock Mamba，通过音素条件化选择性扫描（PCSS）将受监督且结构单调的“说话时钟”融入Mamba状态空间模型的时间递归中，使编码器表示显式锚定转录进度，从而提升视觉强制对齐精度并抑制长序列上的对齐漂移。

    

    视觉强制对齐（VFA）旨在给定转录文本的情况下，在无声的说话人脸视频中定位音素和单词边界。现有方法主要在解码阶段强制执行转录文本的顺序，导致编码器缺乏将随时间演化的表示锚定到转录文本进度的显式机制。因此，音素特定的证据被模糊化，对齐漂移会在长序列上不断累积。在本文中，我们提出了Speaking-Clock Mamba，它将转录文本进度纳入时间递归之中。我们在视觉前端与Conformer交叉注意力编码器之间插入选择性状态空间（Mamba）骨干网络，并通过音素条件化选择性扫描（PCSS）将其输入相关的步长条件化于一个受监督的、结构上单调的“说话时钟”，该时钟估计在转录文本中的推进进度。由于该骨干网络的因果与双向变体会产生互补性错误，一种免训练的选择（摘要在此处被截断）

    arXiv:2611.18667v1 Announce Type: cross  Abstract: Visual forced alignment (VFA) locates phoneme and word boundaries in a silent talking-face video given its transcript. Existing methods mainly enforce transcript order at the decoding stage, leaving the encoder without an explicit mechanism to anchor its temporally evolving representation to transcript progress. As a result, phoneme-specific evidence is blurred and alignment drift accumulates over long sequences. In this paper, we propose Speaking-Clock Mamba, which makes transcript progress part of the temporal recurrence. A selective state-space (Mamba) backbone is inserted between the visual frontend and the Conformer cross-attention encoder, and a Phoneme-Conditioned Selective Scan (PCSS) conditions its input-dependent step size on a supervised, structurally monotone speaking clock that estimates progress through the transcript. Since causal and bidirectional variants of this backbone make complementary errors, a training-free sele
    
[^183]: 标准激活的偏好合成：从显式标准中生成保持归因的偏好数据

    Criterion-Activated Preference Synthesis: Attribution-Preserving Preference Data from Explicit Standards

    [https://arxiv.org/abs/2611.18663](https://arxiv.org/abs/2611.18663)

    CAPS提出一种标准优先的偏好数据合成框架，通过双信号评分和EAT、AR两道过滤器，从显式质量标准生成归因清晰、可验证的偏好对，从而解决DPO中因多维差异纠缠导致的归因歧义问题。

    

    直接偏好优化（DPO）因归因歧义问题而广受诟病：当被选择与被拒绝的回复在多个相互纠缠的维度上存在差异时，模型无法识别偏好标签究竟在教哪种行为。我们提出了标准激活偏好合成（CAPS），这是一个“标准优先”的框架，能够将显式的质量标准转化为经过归因验证的偏好对。CAPS从目标标准出发合成上下文与回复，然后采用双信号评分机制：一个基于评分准则的裁判衡量标准符合度，而一个盲测的通用裁判衡量整体质量。两个过滤器完成了归因闭环：经验激活测试（EAT）保留那些“满足标准确实能提升盲测质量”的标准-上下文对，归因比率（AR）保留那些质量差距主要由目标标准主导的偏好对。该框架已在通用评分准则、医学评分准则和中文法律条文上进行了实例化验证。

    arXiv:2611.18663v1 Announce Type: new  Abstract: Direct Preference Optimization (DPO) is notoriously sensitive to attribution ambiguity: when chosen and rejected responses differ along multiple entangled dimensions, the model cannot identify which behavior the preference label teaches. We propose Criterion-Activated Preference Synthesis (CAPS), a criterion-first framework that turns explicit quality standards into attribution-verified preference pairs. CAPS synthesizes contexts and responses from a target criterion, then applies dual-signal scoring: a rubric-bound judge measures criterion compliance, while a blind general judge measures holistic quality. Two filters close the attribution loop: the Empirical Activation Test (EAT) keeps criterion-context pairs where satisfying the criterion improves blind quality, and the Attribution Ratio (AR) keeps pairs whose quality gap is dominated by the target criterion. Instantiated with general rubrics, medical rubrics, and Chinese legal statute
    
[^184]: 泰卢固语字母的非确定性有限自动机及格律诗歌的可视化

    Non-Deterministic Finite Automata for Telugu Letters and Visualizations for Metrical Poetry

    [https://arxiv.org/abs/2611.18661](https://arxiv.org/abs/2611.18661)

    本文将泰卢固语格律诗歌中音节分类的原子单元 aksharam 形式化为非确定性有限自动机，并给出了轻音节字母的必要条件与重音节字母的充分条件的自动机图和正则表达式。

    

    音节轻重是泰卢固语格律诗歌的核心。从音节角度来看，每个字母（aksharam 记号）要么是轻音节，要么是重音节，诗人依据一组预定义的规则对其进行分类。本文将 aksharam 记号——即音节分类的原子单元——形式化为一个非确定性有限自动机（NFA），其正则表达式为：$(A_h+A_d+A_p)(\epsilon+V_s)+A_m+H^+G(\epsilon+H+V_s+V_v)$，其中 $\epsilon$ 表示空字符串，$A$ 表示元音，包括短元音、长元音、超长元音和杂项元音，$H$ 表示纯辅音，$G$ 表示元音附加符号，$V_s$ 表示鼻音符号，$V_v$ 表示送气符号，后两者分别对应鼻音和送气音。随后，我们给出了以下内容的自动机图和正则表达式：(i) 每个轻音节字母都满足的必要条件，以及 (ii) 字母成为重音节字母的充分条件。

    arXiv:2611.18661v1 Announce Type: new  Abstract: Syllabic weight is central to Telugu metrical poetry. Syllabically, every letter (aksharam token) is either light (laghuvu) or heavy (guruvu). Poets classify it using a set of pre-defined rules. This paper formalizes aksharam token, the atomic unit of syllabic classification, as a non-deterministic finite automaton (NFA) with the regular expression$\colon$ $(A_h+A_d+A_p)(\epsilon+V_s)+A_m+H^+G(\epsilon+H+V_s+V_v)$, where $\epsilon$ denotes a null string, $A$ denotes a vowel (achchu), comprising hrasvam ($A_h$), deergham ($A_d$), plutham ($A_p$), and miscellaneous ($A_m$), $H$ a pure consonant (pollu\_hallu), $G$ a vowel diacritic (gunintha\_chihnam), $V_s$ the sunna (anusvara), and $V_v$ the visarga, the latter two being nasal and aspirated sounds respectively. We then present automata diagrams and regular expressions for (i) a necessary condition satisfied by every light letter, and (ii) a sufficient condition for a letter to be heavy. 
    
[^185]: 赢得排行榜，失去分布：低资源语音识别中声学收窄的诊断与缓解

    Winning the Leaderboard, Losing the Distribution: Diagnosing and Mitigating Acoustic Narrowing in Low-Resource ASR

    [https://arxiv.org/abs/2611.18653](https://arxiv.org/abs/2611.18653)

    该论文发现，针对公开排行榜优化的低资源语音识别模型微调会导致“声学收窄”——尽管在排行榜上夺冠，模型在多种真实声学条件下的表现反而全面劣于未微调的基座模型，且内部检测代理指标会掩盖这一退化。

    

    对于一种低资源语言来说，公开排行榜往往是唯一共享的进展衡量标准，因此朝它优化就意味着朝某个特定测试分布优化。我们对whisper-large-v3-turbo进行低秩微调后，在MediaSpeech、Common Voice 17和一个包含五个测试集的土耳其语社区排行榜上均排名第一，在MediaSpeech上词错误率为9.02%，而最佳基线为11.93%。然而在其私有划分上，它输给了未经修改的whisper-large-v3。我们将该结果视为警报而非证据，主要发现并不依赖于此。在一个独立于我们模型选择流程构建的、涵盖二十种条件的FLEURS-TR基准上，未经修改的基座模型在所有条件下都击败了所有微调版本，且五项性能差距在配对自助法检验下均显著大于零。我们将这种现象称为“声学收窄”：微调缩小了模型能够良好服务的声学范围。核心发现是方法学层面的：我们原本用于检测声学收窄的内部代理指标反而掩盖了这一问题，将收窄的微调模型排在……

    arXiv:2611.18653v1 Announce Type: new  Abstract: For a low-resource language, one public leaderboard is often the only shared measure of progress, so optimizing toward it means optimizing toward a test distribution. Our low-rank fine-tune of whisper-large-v3-turbo ranked first on MediaSpeech, Common Voice 17 and a five-set Turkish community board, at 9.02% word error on MediaSpeech against 11.93% for the best baseline. On its private split it lost to unmodified whisper-large-v3. We treat that split as an alarm, not evidence; the main finding does not depend on it. On a twenty-condition FLEURS-TR benchmark built independently of our model selection, the unmodified base beats every fine-tune in every condition, and all five deficits clear zero under paired bootstrap. We call this acoustic narrowing: fine-tuning contracts the acoustic range a model serves well. The central finding is methodological: our in-house proxy for detecting narrowing concealed it instead, ranking the narrowed fine
    
[^186]: 重新思考离线在策略蒸馏：从标量优势到分布式监督

    Rethinking Offline On-Policy Distillation: From Scalar Advantages to Distributional Supervision

    [https://arxiv.org/abs/2611.18648](https://arxiv.org/abs/2611.18648)

    该论文发现基于固定rollout的离线在策略蒸馏因标量优势监督会导致输出分布变平，并提出用分布式监督替代标量优势的DistOPD方法来解决这一问题。

    

    在策略蒸馏（OPD）通过在学生生成的轨迹上提供token级别的反馈来转移教师模型的能力，但反复用教师模型评估新生成的rollout在计算上代价高昂。诸如Lightning OPD之类的离线方法通过在固定的参考rollout上缓存所记录token的教师对数概率来降低这一成本。通过将这一想法扩展到多轮智能体搜索设置中，我们发现其表现持续低于由同一教师模型训练的SFT起点。我们在实证上观察到比数学推理场景中更严重的教师-参考失配以及训练后分布变平现象，并从理论上证明：在适当条件下，固定rollout上的标量优势监督会产生输出分布变平的解。受这些发现的启发，我们提出了分布式在策略蒸馏（DistOPD），这是一种简单而有效的离线OPD方法，它保留了候选（摘要在此处截断）

    arXiv:2611.18648v1 Announce Type: cross  Abstract: On-policy distillation (OPD) transfers teacher capabilities through token-level feedback on student-generated trajectories, but repeatedly evaluating fresh rollouts with the teacher is computationally expensive. Offline methods such as Lightning OPD reduce this cost by caching recorded tokens' teacher log-probabilities on fixed reference rollouts. By extending the idea to a multi-turn agentic search setup, we find that it consistently underperforms the SFT starting point trained by the same teacher. We empirically observe greater teacher--reference mismatch than in mathematical reasoning and post-training distributional flattening, and theoretically show that scalar advantage supervision on fixed rollouts admits solutions with flattened output distributions under suitable conditions. Motivated by these findings, we propose Distributional On-Policy Distillation (DistOPD), a simple yet effective offline OPD method that retains candidate-
    
[^187]: SERAG：面向知识图谱异常检测的语义与关系方法

    SERAG: SEmantical and Relational approach for Anomaly detection on knowledge Graphs

    [https://arxiv.org/abs/2611.18640](https://arxiv.org/abs/2611.18640)

    SERAG通过融合基于标签编码的语义分数和关系图卷积网络的关系分数，实现对知识图谱中异常事实的有监督检测，在ICEWS18上显著超越最强基线。

    

    从文本构建的知识图谱中可能包含那些单独看来合理、但与其关系上下文不一致的事实。我们提出了SERAG，一种有监督异常检测器，它结合了两种互补的合理性估计：一种是由分别编码的主语、关系和宾语标签计算得到的语义分数，另一种是通过关系图卷积网络获得的关系分数。两种分数经过标准化、校准后进行融合。我们在ICEWS18数据集的一个子集上评估该方法，使用由大语言模型生成的受词表约束的合成异常，并采用混合划分和按时间顺序划分两种实验设置。在10%的异常比例下，SERAG分别取得了0.321和0.258的测试AUPRC，而最强适应基线仅分别为0.159和0.146。融合策略相比任一单独分支均提升了平均测试AUPRC，其中在混合划分协议中提升更大，在按时间顺序划分协议中经验证集选择的提升较小。

    arXiv:2611.18640v1 Announce Type: cross  Abstract: Knowledge graphs constructed from text can contain facts that are individually plausible yet inconsistent with their relational context. We introduce SERAG, a supervised anomaly detector that combines two complementary plausibility estimates: a semantic score computed from separately encoded subject, relation, and object labels, and a relational score obtained through a relational graph convolutional network. The scores are standardized, calibrated, and fused. We evaluate the method on a subset of ICEWS18 with vocabulary-constrained synthetic anomalies generated by an LLM, using pooled and chronological splits. At 10% anomaly prevalence, SERAG achieves test AUPRCs of 0.321 and 0.258, respectively, compared with 0.159 and 0.146 for the strongest adapted baseline. Fusion improves mean test AUPRC over either branch alone, with a larger gain in the pooled protocol and a smaller validation-selected gain in the chronological protocol. Additi
    
[^188]: SynthDCB：面向可扩展设备控制基准测试的世界蒸馏方法

    SynthDCB: World Distillation for Scalable Device Control Benchmarking

    [https://arxiv.org/abs/2611.18639](https://arxiv.org/abs/2611.18639)

    该论文提出SynthDCB基准，创新性地通过“世界蒸馏”方法将设备行为知识转化为模块化可执行组件，构建了覆盖家居、汽车和手机三大领域的可扩展设备控制与环境感知评测基准。

    

    扩展设备控制评估需要同时扩展智能体所交互的世界以及它们需要解决的任务。我们提出了SynthDCB，一个通过世界蒸馏构建的、可扩展的环境感知与设备控制基准。世界蒸馏将设备行为、交互约束和环境效应的知识转化为模块化的可执行组件，这些组件被组合成具有显式状态、动作和时间动态的交互式世界。统一的能力表示将工具发现、世界执行和任务合成连接起来，使新的能力能够通过共享的构建流水线进入评估。SynthDCB实例化了家居、汽车和手机三个领域，包含114个设备类别、1,166个功能和95个事件，支持五大任务族共32个子类型。这种有状态的设计支持通过状态正确性和工具调用轨迹进行任务验证与智能体评估。我们评估了……

    arXiv:2611.18639v1 Announce Type: cross  Abstract: Scaling device-control evaluation requires scaling both the worlds agents interact with and the tasks they must solve. We introduce SynthDCB, a scalable benchmark for environment awareness and device control built through world distillation. World distillation transforms knowledge of device behaviors, interaction constraints, and environmental effects into modular executable components, which are composed into interactive worlds with explicit states, actions, and temporal dynamics. A unified capability representation connects tool discovery, world execution, and task synthesis, allowing new capabilities to enter evaluation through a shared construction pipeline. SynthDCB instantiates Home, Car, and Phone domains with 114 device classes, 1,166 functions, and 95 events, supporting five task families with 32 subtypes. The stateful design enables task validation and agent evaluation through state correctness and tool-call traces. We evalua
    
[^189]: GRAFT：面向持续语言与多模态学习的跨骨干网络LoRA复用

    GRAFT: Cross-Backbone LoRA Reuse for Continual Language and Multimodal Learning

    [https://arxiv.org/abs/2611.18638](https://arxiv.org/abs/2611.18638)

    提出GRAFT框架，通过模态感知的适配器库检索和学习的特征空间线性桥接，实现跨骨干网络的LoRA适配器复用，使语言模型适配器可迁移至多模态模型并显著优于随机初始化。

    

    在持续学习中，参数高效微调使大语言模型能够通过LoRA等轻量级适配器获取新能力，然而预训练适配器的复用通常仅限于共享相同骨干网络的模型，这限制了将已学到的能力从语言模型迁移到多模态模型。我们提出GRAFT（Generalist Reuse of Adapters via Feature-space Transfer，通过特征空间迁移实现适配器的通用复用），一个跨骨干网络的LoRA复用框架。GRAFT从模态感知的适配器库中检索最相关的预训练LoRA适配器，并通过在对应的源与目标特征空间之间学习得到的线性桥接，将其映射到目标骨干网络中。我们在十个Super-NaturalInstructions文本任务上评估了GRAFT（将Qwen2.5的LoRA迁移至Qwen2.5-VL），并在五个多模态任务上直接复用Qwen2.5-VL的LoRA。与随机初始化的LoRA适配器相比，GRAFT在大多数评估任务上提升了性能。

    arXiv:2611.18638v1 Announce Type: cross  Abstract: In continual learning, parameter-efficient fine-tuning enables large language models to acquire new capabilities through lightweight adapters such as LoRA, yet the reuse of pretrained adapters is typically restricted to models sharing the same backbone. This limits the transfer of learned capabilities from language models to multimodal models. We propose GRAFT (Generalist Reuse of Adapters via Feature-space Transfer), a framework for cross-backbone LoRA reuse. GRAFT retrieves the most relevant pretrained LoRA adapter from a modality-aware library and maps it into the target backbone through a learned linear bridge between corresponding source and target feature spaces. We evaluate GRAFT on ten Super-NaturalInstructions text tasks, transferring Qwen2.5 LoRAs to Qwen2.5-VL, and five multimodal tasks, where Qwen2.5-VL LoRAs are directly reused. Compared with randomly initialized LoRA adapters, GRAFT improves performance on most evaluated 
    
[^190]: ReVision：面向工具增强、推理密集型视觉问答的选择性内在推理方法

    ReVision: Selective Intrinsic Reasoning for Tool-Augmented, Reasoning-Intensive VQA

    [https://arxiv.org/abs/2611.18637](https://arxiv.org/abs/2611.18637)

    该论文提出ReVision——一种无需训练的测试时方法，将“内在重新审视”作为与搜索、代码执行并列的可选动作，让模型在工具增强的推理密集型视觉问答中自主决定何时重新审视原始图像，从而缓解过度依赖外部工具导致的答案反转问题。

    

    外部工具大幅扩展了多模态大语言模型（MLLM）在推理密集型视觉问题上可检索与计算的内容，且当工具可用时总体准确率通常会有所提升。我们在六个VQA基准上的实证研究表明，工具访问带来的总体准确率提升可能与“答案反转”现象共存：即某些问题在无工具时回答正确，但在有工具时反而回答错误。轨迹分析显示，工具访问下的推理轮次显著缩短，这表明模型从内在视觉推理转向了对外部输出的依赖。我们提出ReVision，一种无需训练的测试时方法，它将“内在重新审视”注册为与搜索和代码执行并列的可选动作。当该动作被调用时，一次独立的、不使用工具的多模态调用会重新审视原始图像，整合已累积的证据，并返回修订后的分析结果。MLLM自主决定是否以及何时使用该动作，无需固定的调度机制。

    arXiv:2611.18637v1 Announce Type: cross  Abstract: External tools substantially expand what multimodal large language models (MLLMs) can retrieve and compute on reasoning-intensive visual questions, and aggregate accuracy typically improves when tools are available. Our empirical study across six VQA benchmarks shows that aggregate accuracy gains under tool access can coexist with answer reversals: questions answered correctly without tools but incorrectly with tools. Trajectory analysis reveals substantially shorter reasoning turns under tool access, suggesting a shift from intrinsic visual reasoning toward reliance on external outputs. We propose ReVision, a training-free test-time method that registers intrinsic reconsideration as an optional action alongside search and code execution. When invoked, a separate tool-free multimodal call re-examines the original image, integrates accumulated evidence, and returns a revised analysis. The MLLM chooses if and when to use it, without fixe
    
[^191]: 超越二元审计：指南锚定能否缓解LLM阿片类药物处方中的决策空间框架效应？

    Beyond Binary Audits: Does Guideline Grounding Mitigate Decision-Space Framing in LLM Opioid Prescribing?

    [https://arxiv.org/abs/2611.18631](https://arxiv.org/abs/2611.18631)

    论文提出Q-Pain++全因子压力测试，发现改变剂量选项的粒度会显著影响LLM的阿片类药物处方决策，而检索指南锚定虽能部分缓解但不能消除这种决策空间框架效应，表明偏见审计应变化答案格式而不仅是患者身份。

    

    针对LLM阿片类药物处方偏见的审计在很大程度上依赖于二元治疗决策，这一信号在当前模型上已趋于饱和。我们提出了Q-Pain++，这是一个全因子压力测试，在保持临床内容不变的前提下，独立变化临床风险因素和可用剂量选项的粒度。在其余内容不变的病例描述中增加一个中间剂量选项，会显著降低大多数模型与专家标注剂量的一致性，而在加入临床风险因素后，除一种配置外，剂量决策中的人口统计学差异在所有配置中都保持很小。通过检索将模型锚定于围手术期指南，平均而言可以减少这种框架效应但无法消除它，并且当检索到的段落重新排序时，一个模型的治疗决策会发生变化。这些结果表明，偏见审计应该变化答案格式，而不仅仅是变化患者身份。我们将发布我们的代码库等相关资源。

    arXiv:2611.18631v1 Announce Type: new  Abstract: Audits of LLMs for bias in opioid prescribing have largely relied on binary treatment decisions, a signal that current models have saturated. We introduce Q-Pain++, a full-factorial stress test that holds clinical content fixed while independently varying clinical risk factors and the granularity of the available dosage options. Adding an intermediate dosage option to otherwise unchanged vignettes substantially lowers agreement with expert-annotated dosages for most models, while demographic differences in dosage decisions remain small in all but one configuration once clinical risk factors are added. Grounding models in perioperative guidelines through retrieval reduces this framing effect on average but does not eliminate it, and one model's treatment decisions change when the retrieved passages are reordered. These results suggest that audits should vary the answer format, not only patient identity. We will release our codebase as wel
    
[^192]: 所有环节都至关重要！对Dread论坛地下欺诈生态系统的六阶段分析

    All the Pieces Matter! A Six-stage Analysis of an Underground Fraud Ecosystem on Dread

    [https://arxiv.org/abs/2611.18630](https://arxiv.org/abs/2611.18630)

    该研究通过分析暗网论坛Dread上九个社区的6.7万余条帖子和30万余条评论，提出了从能力储备到变现的六阶段欺诈价值链分析框架，并揭示了少数“桥梁者”用户是维系整个地下欺诈生态系统结构凝聚力的关键。

    

    网络诈骗是一种主要的全球性威胁，会造成严重的经济损失、身份盗窃和广泛的心理伤害。这种非法经济依赖于一个高度专业化的地下生态系统，其中不同的参与者在此交换技术工具、操作技巧和洗钱渠道。本研究从一条从准备到变现的连贯六阶段价值链的视角，对一个活跃的欺诈生态系统进行了案例研究，这六个阶段为：能力储备、数据获取、技术分享、身份伪造、执行与变现提取。通过分析基于Tor的暗网论坛Dread上九个社区的67,976条帖子和305,048条评论，我们绘制了用户轨迹、内容分布以及生成式AI的采用情况。我们的研究结果表明，该生态系统的结构凝聚力由一小部分产生绝大多数内容的“桥梁者”（bridgers）所维系。在与保持度分布的零模型对比中，三……

    arXiv:2611.18630v1 Announce Type: new  Abstract: Cyber-enabled fraud is a dominant global threat that causes severe financial loss, identity theft, and widespread psychological harm. This illicit economy relies on a highly specialized underground ecosystem where distinct actors exchange technical tools, operational techniques, and laundering channels. This research presents a case study of one such live fraud ecosystem from the perspective of a cohesive six-stage value chain spanning from preparation to monetization: Capability, Data Acquisition, Technique Sharing, Identity Forgery, Execution, and Extraction. Analyzing 67,976 posts and 305,048 comments from nine communities on the Tor-based darknet forum Dread, we map user trajectories, content distribution, and the adoption of generative AI. Our findings show that structural cohesion within the ecosystem is maintained by a thin layer of `bridgers' who generate the majority of all content. Against a degree-preserving null model, three 
    
[^193]: Decision 1.0：迈向开放式决策基础模型

    Decision 1.0: Towards Open Decision Foundation Models

    [https://arxiv.org/abs/2611.18629](https://arxiv.org/abs/2611.18629)

    Decision 1.0 是首个开放的决策基础模型家族，它能将文本状态和运行时定义的问题直接转化为类型化概率分布，无需自回归生成即可为智能体和工作流提供可复用的判断能力。

    

    智能软件需要在控制流内部进行可复用的判断：选择一个动作、测试一个条件，或依据评分标准评估证据。我们提出 Decision 1.0，这是一个开放权重的模型家族，能够将文本状态和运行时定义的问题转化为类型化的概率分布。其 Choice、Noul 和 Score 接口让应用程序能够在推理时指定输出结果，并将预测组合成智能体和工作流。每个问题都会产生其对应的候选分布，而无需自回归文本生成。六个模型涵盖双向编码器和因果解码器两种架构，其中包括一个工作流专用模型。训练数据来源于可验证任务、自然语言标注以及工作流适配。在一个通用决策基准上，Lux-9B 在显式决策上达到 84.17%，在推理任务上达到 91.25%，加权总分为 77.22%。实验结果还揭示了模型在组合能力、迁移能力和概率校准方面存在的差距。

    arXiv:2611.18629v1 Announce Type: new  Abstract: Intelligent software needs reusable judgments inside its control flow: selecting an action, testing a condition, or assessing evidence against a rubric. We present Decision 1.0, an open-weight family that turns textual state and runtime-defined questions into typed probability distributions. Its Choice, Noul, and Score interface lets applications specify outcomes at inference time and compose predictions into agents and workflows. Each question produces its candidate distribution without autoregressive text generation. Six models cover bidirectional encoder and causal decoder architectures, including a workflow specialist. Training draws on verifiable tasks, natural-language annotations, and workflow adaptation. On a common decision benchmark, Lux-9B achieves 84.17\% on explicit decisions and 91.25\% on inference, with a weighted overall score of 77.22\%. Results also expose gaps in composition, transfer, and probability calibration. Rel
    
[^194]: EmoTrace：面向情感支持对话的递归轨迹因式分解生成方法

    EmoTrace: Recursive Trace-Factorized Generation for Emotional Support Conversation

    [https://arxiv.org/abs/2611.18628](https://arxiv.org/abs/2611.18628)

    EmoTrace提出一种递归的轨迹因式分解生成方法，将每个对话轮次建模为封装求助者状态演变、支持策略、支持者情感与文本实现的结构化交互增量，从而在多轮情感支持对话中保持显式的决策连续性。

    

    有效情感支持的根本在于一种迭代式的人际动态过程，即支持者的干预持续塑造着求助者的心理轨迹。尽管当代模型能够捕捉轮次层面的变量，包括瞬态情绪和离散策略，但它们本质上将这些因素视为孤立事件。由于严格保留字面上的对话历史却丢弃了底层的决策逻辑，此类架构从根本上无法维持连贯的长期支持。在此，我们提出EmoTrace，一种利用轨迹因式分解的递归方法，以在跨越多个轮次的交互中保持显式的决策连续性。与其他方法不同，我们的方法将每个对话轮次表述为结构化的交互增量，其中封装了求助者不断演变的状态、所选策略、支持者的情感以及最终的文本实现。

    arXiv:2611.18628v1 Announce Type: new  Abstract: Fundamental to effective emotional support is the iterative interpersonal dynamic wherein a supporter intervention continuously shapes the psychological trajectory of the seeker. While contemporary models capture variables at the turn level, including transient emotions and discrete strategies, they inherently treat these factors as isolated events. By retaining strictly literal dialogue histories while discarding the underlying decision logic, such architectures remain fundamentally incapable of sustaining coherent extended support. Here we introduce EmoTrace, a recursive method utilizing trace factorization to preserve explicit decision continuity throughout interactions spanning multiple turns. Departing from others, our approach formulates each conversational turn as a structured interaction increment encapsulating the evolving state of the seeker, the selected strategy, the affect of the supporter, and the ultimate textual realizati
    
[^195]: 面向出院预测的临床语言模型可解释性引导审计

    Interpretability-Guided Auditing for Clinical Language Models for Discharge Prediction

    [https://arxiv.org/abs/2611.18626](https://arxiv.org/abs/2611.18626)

    提出AuditBERT-Discharge轻量级框架，将可解释性（词元级SHAP归因）作为训练时审计机制，自动识别并掩蔽临床笔记中的出院相关词汇捷径，从而减少出院预测模型对虚假线索的依赖。

    

    用于医院出院预测的临床自然语言处理（NLP）模型容易受到临床笔记中明确的出院相关词汇线索的影响，即使对笔记时间戳进行了时间过滤。这些线索可能充当未来结果的代理，导致模型依赖类似“捷径”的信号，而非具有临床意义的证据。我们提出了AuditBERT-Discharge，这是一个轻量级框架，将可解释性重新构建为训练阶段的审计机制。该框架利用初始BioClinicalBERT模型的词元级SHAP归因来识别归因度高的出院相关词元，并在模型重新训练之前将这些词元从训练语料库中掩蔽。我们在某单一学术医疗中心的选择性脊柱手术就诊回顾性队列上对该框架进行了评估。定性归因分析和初步评估表明，SHAP引导的审计减少了模型对明确出院相关线索的依赖。（注：原文摘要在此处截断）

    arXiv:2611.18626v1 Announce Type: cross  Abstract: Clinical natural language processing (NLP) models for hospital discharge prediction are vulnerable to explicit discharge-related lexical cues in clinical notes, even when note timestamps are temporally filtered. These cues may act as proxies for future outcomes, leading models to rely on shortcut-like signals rather than clinically meaningful evidence. We propose AuditBERT-Discharge, a lightweight framework that reframes interpretability as a training-time auditing mechanism. Token-level SHAP attributions from an initial BioClinicalBERT model are used to identify highly attributed discharge-related tokens, which are subsequently masked in the training corpus before model retraining. We evaluate the framework on a retrospective cohort of elective spine surgery encounters from a single academic medical center. Qualitative attribution analysis and preliminary evaluation suggest that SHAP-guided auditing reduces reliance on explicit discha
    
[^196]: 技能迁移：将智能体技能从强模型适配到弱模型

    Skill Transfer: Adapting Agent Skills from Strong to Weak Agents

    [https://arxiv.org/abs/2611.18624](https://arxiv.org/abs/2611.18624)

    该论文提出“技能迁移”这一新问题与方法，在不动模型参数和技能能力上限的情况下，通过将强弱智能体的执行轨迹抽象为类型化执行结构并定位其结构差异来适配技能，从而使成本受限部署中的弱智能体也能充分受益于原本为强模型编写的技能。

    

    智能体技能是由指令、代码和资源组成的模块化包，可供智能体按需加载，这类技能越来越多地由前沿模型编写和验证，并作为开放标准进行共享。同一技能在不同模型上的表现差异很大：强模型能将其转化为巨大的性能提升，而弱模型几乎无法获得这种提升，有时甚至比不使用技能时表现更差。然而，成本和资源限制往往使实际部署只能使用这些较弱的智能体。我们假设强模型默默地补充了技能中未明说的程序性步骤，而弱模型则会省略这些步骤，并将弥合这一差距的问题形式化为“技能迁移”（Skill Transfer）：在保持技能能力上限和模型参数固定不变的前提下，对技能进行适配，使特定的弱智能体能够良好地执行该技能。我们的方法将强模型和弱模型的执行轨迹抽象为类型化的执行结构，并通过……（原文在此处截断）来定位它们之间的结构差异。

    arXiv:2611.18624v1 Announce Type: cross  Abstract: Agent skills are modular bundles of instructions, code, and resources that an agent loads on demand, increasingly authored and validated on frontier models and shared as an open standard. The same skill behaves differently across models: a strong model turns it into a large performance gain, while a weak model realizes little of that gain and sometimes does worse than with no skill at all. Cost and resource constraints, however, often restrict real deployments to exactly these weaker agents. We hypothesize that the strong model silently supplies procedural steps the skill leaves implicit, which the weak model omits, and formalize closing this gap as \emph{Skill Transfer}: holding the skill's capability ceiling and the model's parameters fixed, we adapt the skill so a specific weak agent can execute it well. Our method abstracts strong and weak trajectories into typed execution structures, localizes their structural divergence with a gr
    
[^197]: 评估事实核查大语言模型在多语言证据上的鲁棒性

    Evaluating the Robustness of Fact-Checking LLMs on Multilingual Evidence

    [https://arxiv.org/abs/2611.18623](https://arxiv.org/abs/2611.18623)

    该研究通过构建多语言黄金证据集并系统性地应用四种证据破坏策略，首次评估了事实核查大语言模型在多语言环境下的鲁棒性，发现判定准确性对证据误导性和语言资源水平均高度敏感。

    

    自动化事实核查基准通常将声明与高资源语言中干净、经过精心整理的证据配对，但现实世界中的证据往往不完整、具有误导性，或以模型能力较弱的语言书写。我们评估了事实核查大语言模型在不同语言下的鲁棒性。基于EuroVerdict，我们为英语、德语、希腊语和罗马尼亚语各190条声明构建了黄金证据集，然后在两个严重程度级别上应用四种证据破坏策略。我们在一组声明上测试大语言模型，为其提供来自上述证据集的证据，并采用基于评分标准的“大语言模型作为评委”方法对推理质量进行评分。结果显示判定准确性对这两个因素都高度敏感：误导性编辑使Llama 3.1 8B的英语准确率从80.00%降至16.84%，且随着语言资源水平的降低，性能损失进一步扩大。将希腊语和罗马尼亚语声明机器翻译成英语可以恢复大部分（但非全部）低资源差距，而评委评分的下降速度相对较慢。

    arXiv:2611.18623v1 Announce Type: cross  Abstract: Automated fact-checking benchmarks typically pair claims with clean, curated evidence in high-resource languages, but real-world evidence is often incomplete, misleading, or written in a language where models are weaker. We evaluate the robustness of fact-checking LLMs for different languages. Building on EuroVerdict, we construct gold evidence sets for 190 claims in each of English, German, Greek, and Romanian, then apply four corruption strategies at two severity levels. We test LLMs on a set of claims, providing them with evidence from the sets above. A rubric-based LLM-as-a-judge scores reasoning quality. Verdict accuracy proves highly sensitive to both factors: Misleading edit cut Llama 3.1 8B's English accuracy from 80.00% to 16.84%, and losses widen as language resource level decreases. Machine-translating Greek and Romanian claims into English recovers much, but not all, of the low-resource gap. Judge scores decline more slowly
    
[^198]: RGFC：基于注册表的函数调用与阶段分解评估

    RGFC: Registry Grounded Function Calling with stage-decomposed evaluation

    [https://arxiv.org/abs/2611.18622](https://arxiv.org/abs/2611.18622)

    该论文提出注册表规模函数调用基准 RFCD 与确定性阶段分解评估器 TAAG，并发现工具注册表规模（5 至 3,000 个函数）是模型函数调用性能退化的主要来源。

    

    arXiv:2611.18622v1 公告类型：新 摘要：工具调用评估通常只报告最终调用是否与参考答案匹配，但这一单一结果将不同的失败混为一谈：选择了可用注册表之外的函数、违反所选函数的架构规范，或使用了用户请求不支持的参数值填充参数。当模型必须从更大的工具注册表中进行选择时，这些失败问题更为突出。我们提出了 RFCD，这是一个基于 BFCL、Glaive Function Calling v2 和 Tool-Call-Data 构建的注册表规模函数调用基准，包含规模从 5 到 3,000 个函数的标准化 JSON-schema 注册表。我们还提出了 TAAG，一个用于评估注册表一致性、结构完整性和参数溯源的确定性阶段分解评估器。在六个本地部署的 4B 以下参数模型、三个数据源和四种反馈模式的实验中，注册表规模是性能退化的主导因素：在所有数据源上，单次通过 TAAG 的通过率从小注册表到大注册表均急剧下降。

    arXiv:2611.18622v1 Announce Type: new  Abstract: Tool-calling evaluations often report whether the final call matches a reference answer, but this single outcome merges different failures: choosing a function outside the available registry, violating the selected schema, or filling arguments with values not supported by the user request. These failures matter more when models must choose from larger tool registries. We introduce RFCD, a registry-scaled function-calling benchmark built from BFCL, Glaive Function Calling v2, and Tool-Call-Data, with standardized JSON-schema registries ranging from 5 to 3,000 functions. We also introduce TAAG, a deterministic stage-decomposed evaluator for registry conformance, structural completeness, and argument grounding. Across six locally served sub-4B models, three data sources, and four feedback modes, registry scale is the dominant source of degradation: single-pass TAAG pass rate drops sharply from small to large registries on all sources. Repea
    
[^199]: 哪些数学上下文支持目标对齐的猜想生成？一项配对消融研究

    Which Mathematical Context Supports Target-Aligned Conjecture Generation? A Paired Ablation Study

    [https://arxiv.org/abs/2611.18621](https://arxiv.org/abs/2611.18621)

    本研究通过配对消融实验发现，定义、对象、假设、构造等数学上下文组件在猜想生成中总体互补，其中局部数学对象是最强的独立信号，而仅依赖单一组件会显著降低猜想与目标定理的对齐度。

    

    从自然语言上下文中生成数学猜想是一项具有挑战性的任务，因为与定理相关的信息分布在多种形式的数学上下文中，包括定义、对象、假设和构造。我们研究了这些上下文信号中哪些有助于模型恢复由周围设置所支持的特定命题，实验使用了来自四个数学领域12篇论文中的61个主要定理。对于每个定理，我们隐去其结论部分，并用六个上下文组件来表示其余的设置。通过配对消融设计，我们比较了完整上下文、六种留一去一条件以及六种单组件条件，共产生了793个生成的猜想。结果表明，各上下文组件在很大程度上是互补的：移除单个组件通常影响甚微，而仅保留单个组件则会显著降低生成猜想与期望定理的对齐程度。局部数学对象提供了最强的独立信号，而结构性（摘要原文在此处截断）

    arXiv:2611.18621v1 Announce Type: cross  Abstract: Generating mathematical conjectures from natural-language context is challenging because theorem-relevant information is distributed across multiple forms of mathematical context, including definitions, objects, assumptions, and constructions. We study which of these contextual signals help a model recover the specific claim supported by the surrounding setup, using 61 main theorems from 12 papers across four mathematical domains. For each theorem, we redact the result and represent the remaining setup using six context components. A paired ablation design compares full context, six leave-one-out conditions, and six single-component conditions, yielding 793 generated conjectures. Context components were largely complementary: removing one component usually had little effect, whereas retaining only one substantially reduced expected-theorem alignment. Local Mathematical Objects provided the strongest standalone signal, while Structural 
    
[^200]: CondRelBench：一个面向条件限定关系理解的混合领域基准

    CondRelBench: A Mixed-Domain Benchmark for Condition-Scoped Relation Understanding

    [https://arxiv.org/abs/2611.18620](https://arxiv.org/abs/2611.18620)

    该论文提出混合领域基准CondRelBench，通过五类条件探测变体及OGR、SPS两个新指标，系统评估模型能否理解关系成立的条件范围，以防止“关系过度泛化”问题。

    

    关系抽取系统通常能识别关系是什么，但无法识别该关系在什么条件下成立。这会导致受条件约束的论断被坍缩为缺乏支撑的泛化结论，我们将这种失败称为“关系过度泛化”。我们提出了CondRelBench，一个用于评估条件限定关系理解的混合领域基准。每个论断被表示为(E1, R, E2 | C)的形式，其中C是使该关系成立的条件。我们使用3,000个合成段落、200篇PubMed全文论文和695条Reddit帖子，为每个元组构建了五种探测变体，分别保留、替换、删除或矛盾化该条件，或矛盾化关系本身。模型需将每个探测样本分类为“支持”、“不支持”或“证据不足”。我们还引入了过度泛化率和范围保持分数两个指标，用于衡量对被篡改论断的错误支持程度以及探测家族之间的一致性。在提示式LLM、微调LLM和编码器模

    arXiv:2611.18620v1 Announce Type: new  Abstract: Relation extraction systems often identify what a relation is, but not the condition under which it holds. This can collapse condition-bound claims into unsupported generalizations, a failure we call relation overgeneralization. We introduce CondRelBench, a mixed-domain benchmark for evaluating condition-scoped relation understanding. Each claim is represented as (E1, R, E2 | C), where C is the condition licensing the relation. Using 3,000 synthetic paragraphs, 200 PubMed full-text papers, and 695 Reddit posts, we construct five probe variants per tuple that preserve, replace, remove, or contradict the condition, or contradict the relation itself. Models classify each probe as Supported, Not Supported, or Not Enough Evidence. We also introduce Overgeneralization Rate (OGR) and Scope Preservation Score (SPS) to measure false support for altered claims and consistency across probe families. Across prompted LLMs, fine-tuned LLMs, and encode
    
[^201]: 协助患者获取可靠信息：构建医疗用途高效问答系统的指南

    Assisting Patients with Reliable Information Gathering: A Guide to Building Efficient QA Systems for Medical Use

    [https://arxiv.org/abs/2611.18619](https://arxiv.org/abs/2611.18619)

    本研究提出利用在线健康论坛并结合主题引导语义建模（TGSM）实现自动标注，构建无偏见且隐私安全的医疗问答数据集，同时采用基于检索器-阅读器架构的两阶段问答流水线并微调先进的transformer模型，从而高效构建可靠的医疗问答系统。

    

    问答系统在医疗领域的应用迅速发展，显著改善了患者获取可靠健康相关信息的途径。然而，当前方法面临诸多显著挑战，包括难以获得大规模且无偏见的医疗数据集、严峻的隐私问题，以及因人工数据集标注导致的低效率。为解决这些问题，本研究提出一种新颖的方法，利用公开可访问的在线健康论坛系统地构建无偏见、注重隐私保护的问答数据集，并采用主题引导语义建模（TGSM）进行自动化主题识别，从而实现对相关患者生成内容的高效且有针对性的标注。随后，本研究提出一种基于检索器-阅读器（Retriever-Reader）架构的两阶段问答流水线，并通过在所构建的数据集上微调最先进的基于transformer的模型进一步增强其性能。

    arXiv:2611.18619v1 Announce Type: new  Abstract: The application of question-answering (QA) systems in the medical domain has rapidly advanced, significantly improving patients' access to reliable health-related information. However, current approaches face notable challenges, including the difficulty in obtaining large-scale and unbiased medical datasets, significant privacy concerns, and inefficiencies due to manual dataset annotation. To address these issues, This study introduce a novel methodology leveraging publicly accessible online health forums to systematically build an unbiased, privacy-conscious QA dataset, and it employs Topic-guided Semantic Modeling (TGSM) for automated topic identification, enabling efficient and targeted annotation of relevant patient-generated content. Subsequently, this study propose a two-stage QA pipeline based on a Retriever-Reader architecture, which is further enhanced through fine-tuning state-of-the-art transformer-based models on the construc
    
[^202]: 内在注意力作为导航器：一种面向多跳查询的非生成式检索方法

    Intrinsic Attention as Navigator: A Non-Generative Retrieval Method for Multi-Hop Queries

    [https://arxiv.org/abs/2611.18617](https://arxiv.org/abs/2611.18617)

    提出了一种无需训练的非生成式检索方法IN-Retriever，通过利用嵌入模型的内在注意力迭代提取桥接线索、剥离已发现线索并重排序候选文档，无需生成中间推理文本即可将检索精确导向多跳查询的下一跳证据。

    

    在检索增强生成（RAG）的多跳查询中，准确获取证据片段至关重要。在本研究中，我们提出了一种无需训练的多跳查询检索方法，命名为基于内在导航的非生成式检索。该方法不生成中间推理文本，而是基于当前上下文和已检索的候选文档，从嵌入模型的内部注意力中提取桥接线索。在迭代进行线索提取和文档检索的过程中，IN-Retriever 从嵌入的查询向量中剥离已揭示的线索，然后有针对性地对候选文档进行重排序，从而将检索导向隐藏的下一跳证据。在实验中，我们证实了利用内在注意力信号引导外部多跳检索的可行性和有效性。实验结果表明，我们的方法能够精确地收集足够的文档来支持……（原文此处截断）

    arXiv:2611.18617v1 Announce Type: new  Abstract: For multi-hop queries in retrieval-augmented generation (RAG), it is crucial to accurately obtain evidence pieces. In this study, we propose a training-free method for multi-hop query retrieval, named Intrinsic Navigating-based Non-Generative Retrieval (IN-Retriever). Instead of generating intermediate reasoning text, the method extracts the bridging clue from the internal attention of the embedding model based on the current context and retrieved candidate documents. During iterative clue extraction and document retrieval, IN-Retriever strips away revealed clues from the embedded query vector and then reranks candidate documents in a targeted manner, thereby steering retrieval toward hidden next-hop evidence. In the experiments, we confirm the feasibility and effectiveness of using intrinsic attention signals to guide external multi-hop retrieval. Experimental results show that our method precisely collects sufficient documents to suppo
    
[^203]: 评判者并非公正无私：医学大语言模型评估中的自我偏好

    The Judge Is Not Impartial: Self-Preference in Medical LLM Evaluation

    [https://arxiv.org/abs/2611.18616](https://arxiv.org/abs/2611.18616)

    该研究通过13,760次LLM评判系统量化了医学评估场景中的“自我偏好”现象，发现LLM评判者会偏向同一模型生成的输出而与其临床质量无关，揭示了使用LLM评判医学内容时存在的循环性偏差风险。

    

    大语言模型（LLM）越来越多地被用于评估模型生成的医学内容，这带来了一个潜在的循环性问题：LLM评判者可能会偏爱由同一模型生成的输出，而与其临床质量无关。为了刻画医学场景中LLM评判者的自我偏好，我们测量了单轮专科问答和多轮标准化患者诊疗对话中的同模型亲和度。我们使用来自临床医生的620个真实诊疗点问题，评估了八个模型（分别来自OpenAI、Anthropic、Google和Qwen各两个）。同样的八个模型还生成并评判了200个标准化患者-临床医生场景（MedSP1000），最多包含8个可见轮次。总计，我们的分析涵盖了13,760次LLM评判。我们定义了三种互补的自我偏好度量方法。其中，答案排名偏差作为我们的主要度量方法，将同一模型评判者对固定回答所赋予的排名与其来自外部评判者的平均排名进行比较。（摘要原文在此处截断）

    arXiv:2611.18616v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to evaluate model-generated medical content, creating a potential circularity: an LLM judge may favor outputs generated by the same model independently of clinical quality. To characterize LLM judge self-preference in medical settings, we measured same-model affinity in single-turn specialist question answering and multi-turn standardized-patient encounters. We evaluated eight models, two from OpenAI, Anthropic, Google, and Qwen, on 620 real point of care questions from clinicians (Real-POCQi). The same eight models also generated and judged 200 standardized patient-clinician scenarios (MedSP1000) up to 8 visible turns. In total, our analysis spans 13,760 LLM judgments. We define three complementary measures of self-preference. Answer Rank Bias, our primary measure, compared the rank assigned to a fixed response by its own-model judge with its average rank from outside judges. Answer S
    
[^204]: 面向临床记录药物推荐的自进化双角色协商框架

    Self-Evolving Dual-Role Deliberation for Medication Recommendation from Clinical Records

    [https://arxiv.org/abs/2611.18615](https://arxiv.org/abs/2611.18615)

    SEMPA是一个无需训练的双智能体药物推荐框架，通过提议者与批评者的多轮对抗协商、自然语言策略的自进化以及基于规则的药物相互作用与禁忌症安全检查，显著提升了从临床记录推荐药物的准确性（F1提升57.7%）和安全性（DDI率降低77.5%）。

    

    从电子健康记录（EHR）中推荐药物是一项安全关键型任务，要求具备治疗覆盖度、明确的安全约束、可追溯的推理依据以及持续的适应能力。监督式神经模型在部署后即被冻结，无法对新表型进行推理，而现有的LLM流水线均为单次执行，缺乏跨病例的改进循环。我们提出SEMPA，这是一个无需训练的智能体框架，它将提议者与批评者耦合在多轮对抗性协商中，通过分子感知检索对候选药物进行落地，在语义锚定约束下对比式地演化两个智能体的自然语言策略，并强制执行基于规则的药物相互作用（DDI）和禁忌症检查。在100例MIMIC-IV病例上，SEMPA达到了0.3116的Jaccard系数、0.7295的F1分数和0.0172的DDI率，与最强的监督基线相比，F1提升了57.7%，DDI率降低了77.5%。两个LLM评审均将SEMPA评为最优，六种骨干模型的研究（原文摘要在此处截断）

    arXiv:2611.18615v1 Announce Type: new  Abstract: Recommending medications from electronic health records (EHRs) is a safety-critical task that demands therapeutic coverage, explicit safety constraints, traceable justification, and continual adaptivity. Supervised neural models freeze at deployment and cannot reason about new phenotypes, while existing LLM pipelines are single-pass and lack a cross-case improvement loop. We propose SEMPA, a training-free agentic framework that couples a Proposer with a Critic in multi-round adversarial deliberation, grounds candidates with molecular-aware retrieval, contrastively evolves both agents' natural-language policies under a semantic-anchor constraint, and enforces rule-based DDI and contraindication checks. On 100 MIMIC-IV encounters, SEMPA reaches 0.3116 Jaccard, 0.7295 F1, and a 0.0172 DDI rate, improving F1 by 57.7% and cutting DDI by 77.5% over the strongest supervised baseline. Two LLM judges rank SEMPA highest, and six-backbone studies c
    
[^205]: 连接语义与效用：一种可控且可扩展的市场模拟

    Bridging Semantics and Utilities: A Controllable and Scalable Market Simulation

    [https://arxiv.org/abs/2611.18613](https://arxiv.org/abs/2611.18613)

    该论文提出SUM语义-效用模型，通过将群体级语义感知与个体级基于效用的决策相分离，解决了现有LLM金融市场中行为控制不可靠和推理成本随智能体数量增长两大问题，实现了可控且可扩展的市场模拟。

    

    大语言模型（LLMs）近来被用于构建面向金融市场模拟的生成式智能体。现有方法通常为每个智能体单独提供公共市场信息和私人投资者画像作为提示，并要求模型直接生成交易行为。在这种逐个智能体的设计中，模型反复解读群体共享的信息，而个体行为则通过投资者偏好与特质的文本描述来控制。这带来了两个结构性难题：诸如风险敏感性等连续特质无法可靠地引发有序的行为变化，且反复处理相同的公共信息导致推理成本随智能体数量增长。我们提出了SUM（语义-效用模型），将群体层面的语义感知与智能体层面的基于效用的决策制定相分离。语言模型为……解读公共市场信息（摘要原文在此处截断）

    arXiv:2611.18613v1 Announce Type: new  Abstract: Large language models(LLMs) have recently been used to construct generative agents for financial market simulation. Existing approaches typically prompt each agent separately with public market information and a private investor profile, and ask the model to directly generate trading actions. In this agent-wise design, the model repeatedly interprets information shared across the population, while individual behavior is controlled through textual descriptions of investor preferences and traits. This creates two structural difficulties: continuous traits such as risk sensitivity do not reliably induce ordered behavioral changes, and repeatedly processing the same public information causes inference costs to grow with the number of agents. We propose \textbf{SUM}, a Semantic-Utility Model that separates cohort-level semantic perception from agent-level utility-based decision-making. A language model interprets public market information for
    
[^206]: 当有用智能体造成经济损失时：在可执行经济世界中评估LLM智能体安全性

    When Helpful Agents Cause Financial Harm: Evaluating LLM Agent Safety in Executable Economic Worlds

    [https://arxiv.org/abs/2611.18612](https://arxiv.org/abs/2611.18612)

    提出首个在可执行经济环境中系统评估LLM智能体安全性的基准EconSafeBench，发现所有被测模型在用户请求、他人压力及自身绩效目标等不同经济压力来源下均存在8.5%至40.6%的安全违规率。

    

    随着LLM智能体日益具备做出重大经济决策的能力，它们在经济压力下的安全性仍然鲜为人知。我们提出了EconSafeBench，这是首个用于系统性评估LLM智能体在可执行经济环境中安全性的基准。EconSafeBench包含828个案例，涵盖五个经济场景和四类安全违规行为。其关键设计是对三种安全压力来源进行匹配评估，包括明确的用户请求、来自其他参与者的压力，以及智能体自身追求绩效目标所产生的压力。我们进一步通过带有持续记忆和匹配对照组的30周期交互，在静态和动态两个尺度上评估安全性。在八个LLM中，所有被评估模型在静态基准上均表现出安全违规，总体违规率介于8.5%至40.6%之间。我们进一步发现，即使……违规也可能发生

    arXiv:2611.18612v1 Announce Type: cross  Abstract: As LLM agents become increasingly capable of making consequential economic decisions, their safety under economic pressures remains poorly understood. We introduce EconSafeBench, the first benchmark for systematically evaluating the safety of LLM agents in executable economic environments. EconSafeBench contains 828 cases across five economic scenarios and four categories of safety violations. A key design is the matched evaluation of three sources of safety pressure, including explicit user requests, pressure from other actors, and pressure arising from the agent's own pursuit of a performance objective. We further evaluate safety at both static and dynamic scales through 30-period interactions with persistent memory and matched controls. Across eight LLMs, all evaluated models exhibit safety violations on the static benchmark, with overall violation rates ranging from 8.5% to 40.6%. We further find that violations can occur even with
    
[^207]: 通过学习思维链的结构化表示来理解大语言模型推理中的失败

    Understanding Failures in LLM Reasoning by Learning Structured Representations of Chain-of-Thought

    [https://arxiv.org/abs/2611.18611](https://arxiv.org/abs/2611.18611)

    该论文提出CoT2Graph框架，将思维链表示为有向依赖图并用图神经网络结合归因分析，揭示了正确推理更重视早期规划而失败推理往往与过载的合成步骤相关。

    

    思维链是大语言模型提升推理能力的核心，然而现有的评估方法主要关注步骤级别的信号，忽略了推理轨迹的底层结构。这引出了一个关键问题：推理质量的哪些方面（即正确性、效率和失败模式）被编码在思维链轨迹的结构中？我们提出了CoT2Graph，一个将思维链轨迹表示为有向依赖图的框架，使推理步骤之间的关系显式化，并捕捉回溯、分支和子任务合并等模式。我们使用图神经网络对这些图进行建模，并利用基于归因的解释方法来识别与推理质量相关的结构模式。CoT2Graph揭示了成功与失败推理的截然不同的模式：正确的思维链更加重视早期规划，而错误的思维链则与过载的合成步骤相关联。

    arXiv:2611.18611v1 Announce Type: cross  Abstract: Chain-of-Thought (CoT) is central to improving reasoning in large language models, yet existing evaluation methods focus on step-level signals and ignore the underlying structure of reasoning traces. This raises a key question: what aspects of reasoning quality (i.e., correctness, efficiency, and failure patterns) are encoded in the structure of a CoT trace? We introduce \textbf{CoT2Graph}, a framework that represents CoT traces as directed dependency graphs, making relationships between reasoning steps explicit and capturing patterns such as backtracking, branching, and subtask merging. We model these graphs with graph neural networks and use attribution-based explanations to identify structural patterns associated with reasoning quality. CoT2Graph reveals distinct patterns of successful and failed reasoning: correct CoTs place greater importance on early planning, whereas incorrect CoTs are associated with overloaded synthesis steps 
    
[^208]: 纽约大学阿布扎比分校参加ImageEval共享任务：一个用于英语和现代标准阿拉伯语的视听问答与幻觉检测的多模态大语言模型

    NYUAD at ImageEval Shared Tasks: A Multimodal LLM for Audio-Visual Question Answering and Hallucination Detection in English and Modern Standard Arabic

    [https://arxiv.org/abs/2611.18610](https://arxiv.org/abs/2611.18610)

    该论文提出了一个基于多模态大语言模型的统一框架，能够处理英语和现代标准阿拉伯语的视听问答任务，并检测文本-图像对中的幻觉现象。

    

    多模态大语言模型（MLLM）的日益普及使得与视觉和语音信息的交互更加自然，为智能视听问答（AVQA）和幻觉检测系统创造了新的机遇。AVQA扩展了传统的视觉问答，要求模型联合理解输入图像和语音提问，以生成准确且具备上下文感知的回答。尽管近期取得了进展，但由于多模态数据集的匮乏以及需要跨阿拉伯语语音、文本模态与视觉输入进行推理，AVQA和幻觉检测任务对于现代标准阿拉伯语（MSA）而言仍然充满挑战。在本文中，我们提出了一个基于MLLM的AVQA框架，该框架以图像和语音提问作为输入并生成答案。此外，我们利用该框架来检测文本-图像对中的幻觉。所提出的方法在（此处摘要原文截断）

    arXiv:2611.18610v1 Announce Type: new  Abstract: The increasing adoption of multimodal large language models (MLLMs) has enabled more natural interaction with visual and spoken information, creating new opportunities for intelligent audio-visual question answering (AVQA) and hallucination detection systems. AVQA extends traditional visual question answering by requiring models to jointly understand an input image and a spoken question to generate accurate, context-aware responses. Despite recent advances, AVQA and hallucination detection tasks remain challenging for modern standard Arabic (MSA) due to the limited availability of multimodal datasets and the need to reason across Arabic speech and text modalities alongside visual inputs. In this paper, we present an MLLM-based framework for AVQA that takes an image and a spoken question as input and generates answers. In addition, we utilized this framework to detect hallucination in text-image pairs. The proposed approach is evaluated o
    
[^209]: 通往同一错误的多种路径：生成同一学生错误背后的多样化推理路径

    Many Paths to the Same Mistake: Generating Diverse Reasoning Paths Behind the Same Student Mistake

    [https://arxiv.org/abs/2611.18609](https://arxiv.org/abs/2611.18609)

    本文提出了一个利用前向一致性指标和LLM评判机制的生成-过滤流水线，用于生成多样且真实的学生错误推理轨迹，并发现经过监督微调的较小模型比大型LLM更擅长持续保持错误推理。

    

    对AI系统进行真实的评估，需要用户模拟器不仅能够捕捉合作行为，还能捕捉真实交互中出现的错误与变异性。在教育技术领域，模拟学生逐步出错的有缺陷推理过程至关重要，然而映射这些推理路径的数据集十分匮乏，且大语言模型（LLM）往往倾向于自动纠错，而非持续保持所模拟的错误。为解决这一问题，我们提出了一个生成-过滤流水线，用于产生多样且真实的学生错误推理轨迹。我们的方法利用“前向一致性”这一指标——即验证一条有缺陷的推理轨迹能否用于还原学生实际的错误答案——并结合LLM作为评判者以确保真实性。我们的评估表明，尽管许多LLM难以在推理过程中持续保持错误推理，但经过监督微调的较小模型能够成功捕捉这些分歧路径。这项工作为高保真的错误模拟提供了一个框架。

    arXiv:2611.18609v1 Announce Type: new  Abstract: Realistic evaluation of AI systems requires user simulators that capture not only cooperative behavior, but also the errors and variability that arise in real interactions. In educational technology, simulating step-by-step flawed student reasoning is essential, yet datasets mapping these pathways are scarce, and Large Language Models (LLMs) often auto-correct rather than sustain simulated mistakes. To address this, we present a generate-and-filter pipeline that produces diverse and realistic student error traces. Our approach utilizes forward consistency, a metric verifying if a flawed reasoning trace can be used to recover the student's actual incorrect answer, paired with an LLM-as-a-judge to ensure realism. Our evaluation suggests that while many LLMs struggle to maintain incorrect reasoning mid-trace, smaller supervised fine-tuned models successfully capture these divergent paths. This work provides a framework for high-fidelity err
    
[^210]: 纽约大学阿布扎比分校参加ArGuard共享任务：用于检测阿拉伯语仇恨模因和不安全提示的多模态嵌入模型

    NYUAD at ArGuard Shared Tasks: Multimodal Embedding Models for Detecting Arabic Hateful Memes and Unsafe Prompts

    [https://arxiv.org/abs/2611.18608](https://arxiv.org/abs/2611.18608)

    本文利用预训练和微调的多模态多语言嵌入模型，在ArGuard 2026挑战赛数据集上检测阿拉伯语仇恨模因和不安全提示，并证明基于大语言模型的嵌入模型具有卓越性能。

    

    社交媒体和生成式人工智能的快速发展显著增加了在线平台上分享的阿拉伯语文本内容、模因和用户提示的数量。虽然这些媒介促进了交流、创造力和信息交换，但它们也可能被利用来传播仇恨言论、有害内容和不安全指令。因此，人们对可靠的自动化方法来识别仇恨阿拉伯语模因和有害（不安全）提示的需求日益增长。本文利用预训练和微调的多语言嵌入模型，从文本和视觉内容中学习丰富的语义表示。评估是在ArabicNLP ArGuard 2026挑战赛提出的阿拉伯语模因和提示数据集上进行的。结果凸显了基于大语言模型（LLM）的嵌入模型所具备的卓越性能。开发稳健的检测系统对于改善内容（安全）至关重要。

    arXiv:2611.18608v1 Announce Type: new  Abstract: The rapid growth of social media and generative AI has significantly increased the volume of Arabic textual content, memes, and user prompts shared across online platforms. While these media facilitate communication, creativity, and information exchange, they can also be exploited to disseminate hate speech, harmful content, and unsafe instructions. As a result, there is an increasing need for reliable automated methods to identify hateful Arabic memes and harmful (unsafe) prompts. This paper leverages pre-trained and fine-tuned multilingual embedding models to learn rich semantic representations from textual and visual content. The evaluation is conducted using a dataset of Arabic memes and prompts proposed in the ArabicNLP ArGuard 2026 challenge. The results underscore the capacity of large language model (LLM)-based embedding models to deliver superior performance. Developing robust detection systems is essential for improving content
    
[^211]: 计算论证中情感诉求的正当性判定

    Justification of Emotional Appeal in Computational Argumentation

    [https://arxiv.org/abs/2611.18607](https://arxiv.org/abs/2611.18607)

    该论文首次提出情感诉求正当性预测任务并构建首个相关人工标注数据集，揭示正当性判断需综合逻辑与情感维度形成共享判断空间，且大语言模型在此任务上仍面临挑战。

    

    在计算论证中，情感诉求常被视为谬误，但当语境充分支持时，它们也可能是正当的。我们提出了预测情感诉求正当性这一任务，并发布了首个为此任务构建的人工标注数据集，涵盖1,176对“论证—语境”对。基于哲学文献，我们的框架从逻辑与情感两个维度评估情感诉求的正当性。人工标注结果显示，各项正当性评判标准共同构成一个共享的潜在判断空间，其中证据比例适当性与规范可接受性同整体正当性的关联最强。我们进一步发现，缺乏逻辑支持的情感诉求并不总是被判定为完全不正当，而情感诉求越强烈也不一定被认为正当性越低。最后，大语言模型实验表明，情感诉求正当性预测仍是一项具有挑战性的任务。

    arXiv:2611.18607v1 Announce Type: cross  Abstract: Emotional appeals (EAs) are often treated as fallacious in computational argumentation, yet they may be justified when warranted by the context. We introduce the task of predicting EA justification and present the first human-annotated dataset for this task, covering 1,176 argument--context pairs. Grounded in philosophical literature, our framework evaluates EA justification along logical and emotional dimensions. Human annotations show that justification criteria form a shared latent judgment space, with evidential proportionality and normative acceptability most strongly associated with overall justification. We further find that EAs without logical support are not always judged fully unjustified, and stronger EAs are not necessarily judged less justified. Finally, LLM experiments show that EA justification prediction remains challenging.
    
[^212]: 对话中的句法复杂性趋同：现象分析及其在大语言模型检测中的应用

    Syntactic Complexity Convergence in Dialogue: Analysis of the Phenomenon and Application to LLM Detection

    [https://arxiv.org/abs/2611.18606](https://arxiv.org/abs/2611.18606)

    本研究将句法复杂性趋同现象的分析扩展到完整对话层面，发现人类对话中存在稳健的句法趋同而人机对话呈现独特的跨指标模式，并据此利用句法复杂性特征实现了96%准确率的大语言模型对话检测。

    

    我们重新审视了对话中句法复杂性趋同这一尚未被充分探索的现象。我们将先前基于话题层面的分析扩展到完整对话，考察了五个人类-人类数据集以及一组人类-大语言模型对话。我们整合并评估了多维度的句法复杂性指标，以量化对话中的结构变化。我们的分析揭示了人类-人类数据集中稳健的句法复杂性趋同现象，尤其在任务导向型交互中表现明显，统计检验证实这些效应并非偶然。相比之下，人类-大语言模型对话在单个指标下未表现出显著趋同；然而，linear-CKA 揭示了一种独特的跨指标趋同模式。受这些差异的启发，我们将句法复杂性作为检测涉及大语言模型对话的特征，达到了96%的准确率。我们的分类结果展示了这一方法的潜力，且无需依赖基于似然的信号。

    arXiv:2611.18606v1 Announce Type: cross  Abstract: We revisit the under-explored phenomenon of syntactic complexity convergence in dialogue. Extending prior topic-level analyses to full conversations, we examine five human-human datasets and a set of human-LLM dialogues. We integrate and evaluate a multidimensional set of syntactic complexity metrics to quantify structural variation in dialogue. Our analyses reveal robust syntactic complexity convergence across human-human datasets, especially in task-oriented interactions, and statistical tests confirm that these effects are not due to chance. In contrast, human-LLM dialogues show no significant convergence under individual metrics; however, linear-CKA uncovers a distinct cross-metric convergence pattern. Motivated by these differences, we use syntactic complexity as a feature for detecting LLM-involved dialogue, achieving 96\% accuracy. Our classification results demonstrate its potential without relying on likelihood-based signals s
    
[^213]: 使用多语言神经机器翻译方法增强英语-梵语翻译

    Enhanced English-Sanskrit Translation using Multilingual NMT Approach

    [https://arxiv.org/abs/2611.18605](https://arxiv.org/abs/2611.18605)

    该研究通过微调预训练多语言模型IndicTrans并引入印地语构建多语言迁移学习设置，显著提升了低资源英语-梵语双向翻译的性能，超越了双语基线模型。

    

    神经机器翻译（NMT）是自然语言处理自动翻译任务中一种被广泛接受的技术。然而，NMT需要足够数量的平行语料库才能达到合理的翻译质量。在这项工作中，我们考虑了低资源的英语-梵语语言对，用于英语到梵语以及梵语到英语的翻译。我们通过探索使用基于LSTM的序列到序列模型的双语和多语言方法，以及在多语言NMT方法中微调预训练模型IndicTrans，进行了对比分析。为了研究多语言迁移学习，我们还通过使用带语言标签的数据，将印地语与英语和梵语结合，构建了多语言翻译设置。实验结果表明，基于IndicTrans的多语言模型在BLEU和ChrF等自动评估指标上，在两个翻译方向上均优于基线结果。

    arXiv:2611.18605v1 Announce Type: new  Abstract: Neural Machine Translation (NMT) is a widely accepted technique of automatic translation tasks of Natural Language Processing. However, NMT requires a sufficient amount of parallel corpora to attain a reasonable translation quality. In this work, we have considered a low-resource English-Sanskrit pair for English-to-Sanskrit and Sanskrit-to-English translation. We have performed a comparative analysis by exploring bilingual and multilingual approaches using LSTM-based Sequence-to-Sequence, and fine-tuning a pretrained model IndicTrans in a multilingual NMT approach. To investigate multilingual transfer learning, we additionally construct multilingual translation settings by incorporating Hindi along with English and Sanskrit using language-tagged data. Experimental results demonstrate that IndicTrans-based multilingual models outperform baseline results in terms of automatic evaluation metrics such as BLEU and ChrF for both directions of
    
[^214]: 一问四声：阿尔茨海默病照护建议在照护者、临床医生与大型语言模型之间有何不同

    One Question, Four Voices: How Advice for Alzheimer's Caregiving Differs Between Caregivers, Clinicians, and Large Language Models

    [https://arxiv.org/abs/2611.18604](https://arxiv.org/abs/2611.18604)

    该研究用28项指标比较同伴照护者、医生、ChatGPT与检索增强型CareGPT对85个真实阿尔茨海默病照护问题的回答，发现各群体沟通风格截然不同，而仅靠检索增强不足以让CareGPT摆脱ChatGPT正式冗长的咨询式风格。

    

    在线照护者论坛是阿尔茨海默病及相关痴呆（ADRD）照护者获取情感与信息支持的重要来源。我们比较了四个回答群体对来自ALZConnected的85个真实照护者问题的回应：同伴照护者、医生、ChatGPT，以及基于同伴讨论构建的检索增强型CareGPT。通过涵盖语言形式、情感支持、信息支持和主题范围的28项测量指标，我们考察了这些不同的声音在回答相同问题时的差异。同伴照护者提供了以时间为锚的叙事和社区依托；临床医生给出了简洁、以建议为主、临床上有边界的指导；而大语言模型则呈现出正式、冗长、高度结构化的咨询风格。CareGPT与ChatGPT之间的可测量差异极小，这表明仅靠检索加提示注入不足以改变模型的沟通立场。大语言模型的输出在礼貌性和情感表达方面的得分高于同伴照护者。

    arXiv:2611.18604v1 Announce Type: cross  Abstract: Online caregiver forums are an important source of emotional and informational support for Alzheimer's disease and related dementias (ADRD). We compare responses to 85 real caregiver questions from ALZConnected across four responder groups: peer caregivers, physicians, ChatGPT, and a retrieval-augmented CareGPT grounded in peer discussions. Using 28 measures spanning linguistic form, emotional support, informational support, and topic scope, we examine how these voices differ when answering the same questions. Peers provided time-anchored narratives and community grounding; clinicians offered concise, advice-forward, clinically bounded guidance; and LLMs produced a formal, verbose, highly structured consultation style. CareGPT showed minimal measurable separation from ChatGPT, suggesting that retrieval plus prompt injection alone was insufficient to shift communicative stance. LLM outputs scored higher than peers on politeness and emot
    
[^215]: 内部判定追踪证据：通过雅可比透镜对大语言模型认知状态进行模板控制的中智逻辑解读

    Internal Verdicts Track Evidence: A Template-Controlled Neutrosophic Reading of Epistemic States in Large Language Models via the Jacobian Lens

    [https://arxiv.org/abs/2611.18602](https://arxiv.org/abs/2611.18602)

    本文首次利用雅可比透镜在大语言模型内部表征中读取中智逻辑的真(T)、不确定(I)、假(F)三成分，发现冲突证据下反驳质量激增、错误信念下不确定性持续、以及输出层前出现“判定崩塌”等关键特征。

    

    二十五年来，中智逻辑为经典语义学和模糊语义学无法表达的认知状态赋予独立的真(T)、不确定(I)、假(F)程度。以往的应用都是在输出层面测量这些成分：答案、判断、专家评估。据我们所知，本文首次在大语言模型的内部表征中读取中智成分。利用最近发布的雅可比透镜，在Qwen3.5-4B模型上，我们将中间层的读出投影到支持、反驳和保守措辞的词库上，在冲突证据、第一人称错误信念和事实对照条件下（各n=20）获得逐层的(T, I, F)轮廓。第一组实验产生了三个引人注目的特征：冲突条件下反驳质量的激增、错误信念条件下持续的不确定性，以及输出层之前出现的明显“判定崩塌”。第二组模板控制的实验……（摘要原文在此处截断）

    arXiv:2611.18602v1 Announce Type: new  Abstract: For twenty-five years, neutrosophic logic has assigned independent degrees of truth (T), indeterminacy (I), and falsity (F) to epistemic states that classical and fuzzy semantics cannot express. Previous applications measured these components on outputs: answers, judgments, expert evaluations. This paper reports, to our knowledge, the first neutrosophic reading of components inside the internal representations of a large language model. Using the recently released Jacobian lens on Qwen3.5-4B, we project intermediate-layer readouts onto lexicons of support, refutation, and hedging, obtaining layer-wise (T, I, F) profiles under conflicting evidence, first-person false belief, and factual control (n = 20 each). A first battery yielded three striking signatures: a surge of refutation mass under conflict, sustained indeterminacy under false belief, and an apparent "verdict collapse" before the output layer. A second, template-controlled batte
    
[^216]: 以语言风格作为隐藏攻击面越狱大语言模型

    Jailbreak LLMs with Linguistic Style as a Hidden Attack Surface

    [https://arxiv.org/abs/2611.18601](https://arxiv.org/abs/2611.18601)

    该研究发现语言风格是LLM安全中一个此前被忽视的系统性攻击面——相同有害意图的不同风格改写可使攻击成功率产生数量级差异（高达80% vs 低于6%），并提出结合风格条件生成器与BERT风格选择器的轻量级单次攻击框架，同时通过机制分析揭示了拒绝行为是分布式失效模式而非单一拒绝机制。

    

    我们证明，语言风格是大语言模型（LLM）安全中一个系统性且此前被忽视的攻击面。在多样化的风格分类体系和多个最先进模型上，对相同有害目标进行风格改写会导致攻击成功率（ASR）产生数量级的差异，某些风格的ASR可高达80%，而其他风格则保持在6%以下。为了利用这一漏洞，我们提出了一种轻量级的单次攻击框架，该框架将风格条件生成器与基于BERT的风格选择器相结合，其性能可与成本高得多的多轮优化方法相媲美。为了理解风格为何会影响安全行为，我们对LLaMA-3.1-8B进行了机制分析，发现风格改写会在保留可恢复有害意图的同时破坏拒绝表示，这表明存在一种分布式失效模式，而非单一的拒绝机制。受这一发现的启发，我们开发了……（摘要至此截断）

    arXiv:2611.18601v1 Announce Type: new  Abstract: We show that linguistic style is a systematic and previously overlooked attack surface for LLM safety. Across a diverse taxonomy of styles and multiple state-of-the-art models, stylistic rewrites of identical harmful goals produce order-of-magnitude differences in Attack Success Rate (ASR), with some styles achieving up to 80\% ASR while others remain below 6\%. To exploit this vulnerability, we introduce a lightweight single-pass attack framework that combines a style-conditioned generator with a BERT-based style selector, achieving performance competitive with substantially more expensive multi-turn optimization methods. To understand why style affects safety behavior, we conduct mechanistic analysis on LLaMA-3.1-8B and find that stylistic rewrites disrupt refusal representations while preserving recoverable harmful intent, suggesting a distributed failure mode rather than a single refusal mechanism. Motivated by this finding, we devel
    
[^217]: ScopeBench：评估大语言模型对文化规范适用范围的感知能力

    ScopeBench: Evaluating Cultural Norm Scope Awareness in LLMs

    [https://arxiv.org/abs/2611.18600](https://arxiv.org/abs/2611.18600)

    本文提出ScopeBench基准，将文化规范适用范围作为大语言模型安全评估的一等维度，通过3,323条覆盖49个国家的规范测试，揭示了模型范围错误随提示格式反转、以及范围分类准确率与实际行为安全性脱钩等结构性失败。

    

    arXiv:2611.18600v1 公告类型：新 摘要：大语言模型的安全评估通常将危害视为文化不变，然而，如果模型不知道一项安全规范约束的是哪些文化群体，它就无法正确应用该规范。我们提出了ScopeBench，这是一个将文化范围视为一等安全维度的基准。该基准包含3,323条规范，涵盖九个文化集群中的49个国家，每条规范被标注为“普遍适用”或“特定适用”。我们在四个评估任务中以统一方式定义了三种范围错误类型：范围膨胀、范围缩减和范围偏移。通过对来自七个模型家族的21个前沿大语言模型进行评估，我们发现了三种结构性失败。首先，主导错误类型会随提示格式而反转：在单答案分类中，范围缩减错误达到70%，而在独立判断每个候选文化时，范围膨胀错误达到33%。其次，范围分类准确率与行为安全性在实证上是脱钩的：达到至少97%任务准确率的六个分类器……（原文摘要在此处截断）

    arXiv:2611.18600v1 Announce Type: new  Abstract: Safety evaluation in large language models treats harm as culturally invariant, yet a model cannot correctly apply a safety norm without knowing whose norms bind. We introduce ScopeBench, a benchmark that treats cultural scope as a first-class safety dimension. It comprises 3,323 norms spanning 49 countries across nine cultural clusters, each annotated as Universal or Specific. We define three scope error types identically across four evaluation tasks: Scope Inflation, Scope Deflation, and Scope Displacement. Evaluating 21 frontier LLMs across seven model families, we surface three structural failures. First, the dominant error type inverts with prompt format: Scope Deflation reaches 70% in single-answer classification, while Scope Inflation reaches 33% when each candidate culture is judged independently. Second, scope-classification accuracy is empirically decoupled from behavioral safety: the six classifiers achieving at least 97% Task
    
[^218]: 前沿大语言模型在医学本体实体链接任务中的幻觉剖析

    Profiling Hallucinations in Frontier LLMs for Entity Linking to Medical Ontologies

    [https://arxiv.org/abs/2611.18595](https://arxiv.org/abs/2611.18595)

    本研究系统剖析了前沿大语言模型在将临床文本映射到SNOMED CT时的幻觉问题，发现其幻觉率过高而不适合自主临床编码应用，且约束使用真实提及片段反而会加剧幻觉，通用LLM的表现显著逊色于专门的零样本实体链接方法。

    

    将大语言模型（LLM）整合到医疗保健领域有望彻底改变临床文档记录和互操作性，但其可靠性仍令人担忧。本研究对前沿LLM在将临床文本映射到SNOMED CT任务中产生的幻觉进行了全面分析。通过严格的实验，我们发现了一个关键的可靠性差距：LLM产生医学编码幻觉的比率目前使其不适用于自主临床编码应用。矛盾的是，将模型约束为使用真实标注的提及片段反而会加剧而非缓解这些幻觉。我们进一步提出了一个幻觉类型的分类体系——包括已弃用编码和跨本体错误——并证明通用LLM的表现显著逊色于专门的零样本实体链接方法。这些发现强调了在临床应用之前需要健全的验证机制。

    arXiv:2611.18595v1 Announce Type: new  Abstract: The integration of Large Language Models (LLMs) into healthcare promises to revolutionize clinical documentation and interoperability, yet reliability remains a concern. This study presents a comprehensive analysis of hallucinations by frontier LLMs tasked with mapping clinical text to SNOMED CT. Through rigorous experimentation, we identify a critical reliability gap: LLMs hallucinate medical codes at a rate that currently renders them unsuitable for autonomous clinical coding applications. Paradoxically, constraining models to use ground-truth mention spans exacerbates, rather than mitigates, these hallucinations. We further contribute a taxonomy of hallucination types--including deprecated codes and cross-ontology errors--and demonstrate that general-purpose LLMs significantly underperform compared to specialized zero-shot entity linking approaches. These findings underscore the need for robust verification mechanisms before clinical 
    
[^219]: 计算伊斯兰法学（Fiqh）的理论基础

    Theoretical Foundations of Computational Islamic Jurisprudence (Fiqh)

    [https://arxiv.org/abs/2611.18593](https://arxiv.org/abs/2611.18593)

    本文为计算伊斯兰法学确立了六项基础原则，并构建了由阿拉伯语形态接地引擎和阿拉伯语具体化动作演算组成的三层分层计算语言形式系统。

    

    本文提出了计算伊斯兰法学的理论基础和一个完整的形式系统。我们首先确立了六项基础原则，以区分计算法学与传统的自然语言处理以及一般神学形式化。这些原则将伊斯兰法学定义为一门在经验性事实基础上运作的规范性演绎科学，要求采用归纳-演绎双重方法论，并且需要一个专门为法学设计的逻辑基础，而非沿用现有的形式化方法。基于这些原则，我们构建了一个三层分层的计算语言 $\mathcal{F} = D \circ A \circ G$，包括：（1）一个阿拉伯语形态/语法接地引擎，将完全标音的古典阿拉伯语转化为语言无关的接地动作结构（GAS）；（2）阿拉伯语具体化动作演算（ARAC）……

    arXiv:2611.18593v1 Announce Type: new  Abstract: This paper presents the theoretical foundations and a comprehensive formal system for Computational Islamic Jurisprudence (Fiqh). We begin by establishing six foundational principles that distinguish computational Fiqh from conventional natural language processing and general theological formalization. These principles define Islamic jurisprudence as a normative deductive science operating on empirically grounded facts, necessitating a dual inductive-deductive methodology, and requiring a logical foundation specifically engineered for Fiqh rather than adopted from existing formalisms. Based on these principles, we construct a three-layer stratified computational language, $\mathcal{F} = D \circ A \circ G$, consisting of: (1) an Arabic Morphological/Grammatical Grounding Engine that transforms fully vocalized Classical Arabic into language-independent Grounded Action Structures (GAS); (2) the Arabic Reified Action Calculus (ARAC) that nor
    
[^220]: 内源性偏好作为脆弱性信号：发现导致大语言模型效用退化的通用对抗触发器

    Endogenous Preferences as Vulnerability Signals: Discovering Universal Adversarial Triggers for LLM Utility Degradation

    [https://arxiv.org/abs/2611.18592](https://arxiv.org/abs/2611.18592)

    该论文提出EndoUT方法，以冻结模型自身干净行为中的内源性偏好破坏（EPD）作为脆弱性信号，在无需任务标签或下游反馈的情况下自动发现可复用的通用对抗触发器，从而降低大语言模型的效用。

    

    部署的大语言模型在不可信文本进入其提示上下文时必须保持可用。对抗提示与红队测试方法通常通过任务标签、有害目标、评判者反馈或下游效用分数来定义其试图寻找的失败。我们探究：当任务级形式未知且下游任务反馈不可获取时，一个冻结模型的干净行为能否揭示可复用的上下文脆弱性。为研究这一问题，EndoUT采用孪生式设计，在每条无标签指令的干净版本与片段修改版本上比较同一冻结受害模型。在没有任务级目标的情况下，干净前向传递提供受害模型所偏好的响应起始token，作为指令条件化的参考；内源性偏好破坏（EPD）衡量片段插入在多大程度上削弱了这种偏好。一个辅助生成器提供有界片段提议；EPD引导其更新并在无法访问下游信息的情况下对候选池进行排序。

    arXiv:2611.18592v1 Announce Type: new  Abstract: Deployed large language models must remain useful when untrusted text enters their prompt context. Adversarial prompting and red-teaming methods typically define the failures they seek through task labels, harmful targets, judge feedback, or downstream utility scores. We ask whether a frozen model's clean behavior can reveal a reusable context vulnerability when its task-level form is unknown, and downstream task feedback is withheld. To study this question, EndoUT uses a Siamese-style design that compares the same frozen victim on clean and span-modified versions of each unlabeled instruction. Without a task-level target, the clean pass supplies the victim's preferred response-onset token as an instruction-conditioned reference; Endogenous Preference Disruption (EPD) measures how insertion weakens that preference. An auxiliary generator supplies bounded-span proposals; EPD guides its update and ranks the pool without access to downstrea
    
[^221]: 语言模型在哪些方面像失语症患者一样出错，哪些方面则不然

    Where Language Models Fail Like Aphasic Patients and Where They Don't

    [https://arxiv.org/abs/2611.18591](https://arxiv.org/abs/2611.18591)

    该研究提出“计算架构分离”新方法，利用视觉语言模型“有完整词汇语义竞争但无语音编码”的架构特性做出可证伪预测，并通过与88名失语症患者约3700万次命名尝试的对比发现模型扰动错误与患者的语义类错误相符而与语音类错误不符，为词汇产生中语义提取与语音编码是可分离阶段提供了证据。

    

    人类词汇产生的计算架构——即语义-词汇提取与语音编码是否构成可分离的处理阶段——几十年来一直通过使用小词汇量的手工设计模型进行争论。我们提出了“计算架构分离”方法：利用视觉-语言模型（VLM）已知的架构特性，推导出可证伪的预测，判断在扰动下哪些患者错误类型应当与模型错误一致，哪些不应一致。由于VLM通过对完整词汇表进行学习到的语义竞争来完成图片命名，但完全缺乏语音编码环节，因此扰动引发的错误应当在语义阶段的类别上与患者模式相符，而在语音类错误上不相符。通过将LLaVA 1.6（Vicuna-13B）与88名失语症患者在175个费城命名测试项目上进行对比（约3700万次命名尝试，50个扰动种子），我们发现八个错误类别中的六个……（摘要原文在此处截断）

    arXiv:2611.18591v1 Announce Type: new  Abstract: The computational architecture of human word production -- whether semantic-lexical retrieval and phonological encoding constitute dissociable processing stages -- has been debated for decades using hand-designed models with small vocabularies. We introduce computational architectural dissociation: using known architectural properties of vision-language models (VLMs) to derive falsifiable predictions about which patient error types should and should not align with model errors under perturbation. Because VLMs perform picture naming through learned semantic competition over full vocabularies but lack phonological encoding entirely, perturbation-induced errors should match patient patterns for semantic-stage categories but not phonological ones. Testing LLaVA 1.6 (Vicuna-13B) against 88 patients with aphasia across 175 Philadelphia Naming Test items (~37 million naming attempts, 50 perturbation seeds), we find that six of eight error categ
    
[^222]: OmniTag-LLM：多语言多任务语音标注

    OmniTag-LLM: Multilingual Multitask Speech Tagging

    [https://arxiv.org/abs/2611.18589](https://arxiv.org/abs/2611.18589)

    该论文提出五语言联合语音理解基准 OmniTag-Bench（一段音频输入、一个 JSON 输出，涵盖转录、情感、性别、语言识别和命名实体识别），揭示了多模态 LLM 常将非英语语音错误转成英语的系统性缺陷，并证明 LoRA 微调可显著提升各项任务表现。

    

    多模态大语言模型（LLM）有望用一个单一系统取代传统的专用语音模型级联方案，即从一段音频片段中联合完成转录，并同时标注情感、性别、语言和命名实体。然而，大多数基准测试仍然将这些任务分开单独评分，且主要以英语为主。我们推出了 OmniTag-Bench，这是一个涵盖五种语言（英/德/西/法/意）的基准测试及训练方案，用于联合语音理解：输入一段音频，输出一个 JSON 对象（转录文本、情感、性别、语言识别 LID、词符级命名实体识别 NER）。我们在统一的提示词和评分标准下评估了十个开源多模态 LLM 以及一个经过 LoRA 微调的 Qwen3-Omni。除了总体得分之外，OmniTag-Bench 还揭示了一种系统性的失败模式：若干性能强劲的模型经常将非英语语音转录或翻译成英语，从而虚高了多语言词错误率（WER）。微调提升了基础模型在 NER、情感识别和 WER 上的表现，同时在独立的人工标注情感语料库上几乎将准确率翻倍，表明其具有良好的可迁移性（摘要内容到此被截断）。

    arXiv:2611.18589v1 Announce Type: new  Abstract: Multimodal LLMs promise to replace the traditional cascade of specialist speech models with a single system that, from one audio clip, jointly transcribes and labels emotion, gender, language, and named entities. Yet most benchmarks still score these tasks in isolation, and largely in English. We introduce OmniTag-Bench, a five-language benchmark (en/de/es/fr/it) and training recipe for joint understanding: one audio clip in, one JSON object out (transcript, emotion, gender, LID, token-level NER). We evaluate ten open multimodal LLMs and a LoRA fine-tuned Qwen3-Omni on a unified prompt and scorer. Beyond aggregate scores, OmniTag-Bench reveals a systematic failure mode: several strong models often transcribe or translate non-English speech into English, inflating multilingual WER. Fine-tuning improves the base model on NER, emotion, and WER while nearly doubling accuracy on independent human-labeled emotion corpora, suggesting transferab
    
[^223]: 利用大语言模型实现文献筛选与知识综合的自动化

    Automating literature screening and knowledge synthesis with large language models

    [https://arxiv.org/abs/2611.18588](https://arxiv.org/abs/2611.18588)

    该研究开发了一个由大语言模型驱动的统一自动化框架，可完成文献筛选、结构化信息提取与知识综合，在材料科学任务中平衡准确率超过0.90，并将筛选1万条记录的时间从约83个人工小时缩短至4分钟。

    

    科学文献正在快速扩张，但由于浏览大型、异构文献语料库需要巨大的工作量，其中许多知识仍然难以组织和复用。在此，我们开发了一个可泛化的、由大语言模型（LLM）驱动的框架，可在统一的工作流程中实现文献筛选、结构化信息提取和知识综合的自动化。我们在三项范围、语料库规模和类别不平衡程度各不相同的材料科学筛选任务上，对11个最先进的专有和开源权重LLM进行了基准测试。七个LLM在所有任务中均实现了高于0.90的平衡准确率，其中表现最佳的LLM达到0.96–0.98，证明了LLM准确筛选大型、高度不平衡科学语料库的潜力。该框架还大幅减少了人工工作量，在四分钟内即可筛选10,000条记录，而人工筛选约需83个人工小时。

    arXiv:2611.18588v1 Announce Type: new  Abstract: Scientific literature is expanding rapidly, yet much of its knowledge remains difficult to organize and reuse due to the substantial workload needed to navigate large, heterogeneous literature corpora. Here, we develop a generalizable large language model (LLM)-powered framework that automates literature screening, structured information extraction, and knowledge synthesis within a unified workflow. We benchmarked 11 state-of-the-art proprietary and open-weight LLMs across three materials science screening tasks differing in scope, corpus size, and class imbalance. Seven LLMs achieved balanced accuracies above 0.90 across all tasks, with the best-performing LLM reaching 0.96--0.98, demonstrating the potential of LLMs to accurately screen large, highly imbalanced scientific corpora. The framework also substantially reduces manual effort, screening 10,000 records within four minutes compared with approximately 83 person-hours for manual sc
    
[^224]: 量化、评估、重排：一种在不损失质量的前提下压缩大语言模型的方法

    Quantize, Qualify, Rerank: A Recipe for Compressing LLMs Without Losing Quality

    [https://arxiv.org/abs/2611.18587](https://arxiv.org/abs/2611.18587)

    该论文提出“量化-评估-重排”三步压缩方案，通过配对自助法多指标评估（MetricX24-XXL为主、CometKiwi-XXL交叉验证）结合Cometoid best-of-4重排技术，成功将Gemma-3-12b-it压缩至8-bit/4-bit精度且在机器翻译任务中不损失质量。

    

    压缩大语言模型可以降低服务成本，但单一的自动评估指标可能会遗漏质量损失。针对WMT26模型压缩受限赛道，我们在一块H100上为三个翻译方向压缩了google/gemma-3-12b-it模型（bf16格式下为24.4 GB）。配对自助法分析揭示了4-bit量化时各评估指标间的分歧；在WMT24人工元评估的指导下，我们采用MetricX24-XXL作为主要指标，CometKiwi-XXL作为交叉验证。实验表明，移除未使用的视觉模块和非任务词汇表，并使用int8权重或fp8激活不会造成可解析的质量损失，而int4量化则会留下少量残余损失。QLoRA修复会过拟合其校准域。在温度T=0.3下的Cometoid best-of-4重排器在WMT25测试集上同时提升了8-bit和4-bit系统的表现。在提交后的WMT26评测中，两者在无参考指标下仍与bf16基线保持竞争力；重排变体之间的微小差异被视为噪声。

    arXiv:2611.18587v1 Announce Type: cross  Abstract: Compressing large language models lowers serving cost, but a single automatic metric may miss quality loss. For the WMT26 Model Compression constrained track, we compress google/gemma-3-12b-it (24.4 GB in bf16) for three translation directions on one H100. Paired bootstrap analysis exposes metric disagreement on 4-bit quantization; guided by WMT24 human meta-evaluation, we use MetricX24-XXL as primary and CometKiwi-XXL as a cross-check. Removing the unused vision stack and non-task vocabulary and using int8 weights or fp8 activations causes no resolved loss, whereas int4 leaves a small residual. QLoRA healing overfits its calibration domain. A Cometoid best-of-4 reranker at $T=0.3$ improves both the 8-bit and 4-bit systems under WMT25 test set. On post-submission WMT26, both remain competitive with bf16 under reference-free metrics; the small difference between the reranked variants is treated as noise.
    
[^225]: 阅读、路由、修补：面向LLM智能体的鲁棒测试时演化

    Read, Route, Patch: Robust Test-Time Evolution for LLM Agents

    [https://arxiv.org/abs/2611.18585](https://arxiv.org/abs/2611.18585)

    提出Trident框架，通过边界感知记忆阅读、多样性感知候选路由和顺序感知插入式更新，解决LLM智能体测试时演化中的三个信息瓶颈，在五个基准和多个骨干模型上取得最佳平均性能。

    

    基于大语言模型（LLM）的智能体可以通过测试时演化来改善长程交互式决策，然而现有方法面临三个信息瓶颈：记忆检索过度强调文本相似性、候选工具组合往往存在行为冗余、整体工具组合重写可能覆盖有用证据或引入跨组件的不一致性。我们提出了Trident，分别通过边界感知的记忆阅读、多样性感知的候选路由和顺序感知的基于插入的更新，来改善经验利用、行为探索和工具组合演化。我们在涵盖文本游戏、网页交互和编程任务的五个基准上对Trident进行了评估。在qwen-32B、mistral-small-3.2-24B和gpt-oss-20B等模型上，Trident取得了最佳平均性能，并持续优于非学习、基于记忆、基于反思和自动提示优化的基线方法。

    arXiv:2611.18585v1 Announce Type: new  Abstract: Large Language Model (LLM)-based agents can improve long- horizon interactive decision-making through test-time evolu- tion, yet existing methods face three information bottlenecks: memory retrieval overemphasizes textual similarity, candi- date harnesses are often behaviorally redundant, and whole- harness rewriting may overwrite useful evidence or introduce cross-component inconsistency. We introduce Trident, which improves experience utilization, behavioral exploration, and harness evolution through boundary-aware memory reading, diversity-aware candidate routing, and order-aware insertion- based updating, respectively. We evaluate Trident on five benchmarks spanning text games, web interaction, and cod- ing tasks. Across qwen-32B, mistral-small-3.2-24B, and gpt- oss-20B, Trident achieves the best average performance and consistently outperforms non-learning, memory-based, reflection-based, and automated prompt optimization base- line
    
[^226]: “我不是小孩！”：通过多指标偏差分析量化大语言模型中的认知幼化现象

    "I Am Not a Kiddo!": Quantifying Epistemic Infantilization in Large Language Models via Multi-Metric Bias Analysis

    [https://arxiv.org/abs/2611.18583](https://arxiv.org/abs/2611.18583)

    该论文提出了IBI和EARS两个复合指标，通过反事实数据集量化了大语言模型在用户披露残疾信息后回答专家级问题时系统性降低技术深度的“认知幼化”偏差。

    

    大语言模型（LLM）越来越多地被用于辅导教学和专业知识检索，然而现有的安全评估仍主要关注毒性等显性危害，而忽视了“善意的失败模式”。我们研究了“认知幼化”现象，其定义为当残疾信息披露出现在专家级查询之前时，模型在技术深度上出现的与身份相关的降低。为了测量这一效应，我们构建了一个由匹配问答对组成的反事实数据集，并在三种设置下比较模型响应：无身份上下文、在先前对话上下文中披露残疾、以及通过简短的查询前缀披露残疾。我们引入了两个复合指标——幼化偏差指数和认知可及性响应分数——用于捕捉数学与形式化内容的保留程度、领域术语、推理覆盖范围，以及诸如简化标记和未经请求的建议等语气信号。

    arXiv:2611.18583v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for tutoring and professional knowledge retrieval, yet safety evaluations still focus mainly on overt harms such as toxicity while overlooking benevolent failure modes. We study epistemic infantilization, defined as an identity-conditioned reduction in technical depth that appears when disability disclosure precedes an expert-level query. To measure this effect, we construct a counterfactual dataset of matched question-answer pairs and compare model responses across three settings: no identity context, disability disclosed in prior conversational context, and disability disclosed via a short query prefix. We introduce two composite metrics, the Infantilization Bias Index (IBI) and the Epistemic Accessibility Response Score (EARS), which capture retention of mathematical and formal content, domain terminology, reasoning coverage, and tone signals such as simplification markers and unsolic
    
[^227]: VidEngram：通过结构化分层记忆实现智能体式长时视频理解

    VidEngram: Agentic Long-Term Video Understanding via Structured Hierarchical Memory

    [https://arxiv.org/abs/2611.18578](https://arxiv.org/abs/2611.18578)

    VidEngram 通过在视频摄取时一次性构建持久化的结构化分层记忆（片段、事件与实体层），并利用纯文本智能体规划器配合专用检索工具回答后续问题，实现了高效、可检视的长视频问答，在 OmniVideoBench 上超越了单次编码基线。

    

    当多模态模型必须为每个查询重新编码同一视频时，长视频问答既浪费资源又有信息损失：上下文长度限制迫使模型进行稀疏采样或截断，且不会为后续问题保留可复用的表示。我们提出了 VidEngram，一个智能体式视频问答系统，它在视频摄取时一次性构建持久化的分层记忆，并基于该记忆回答后续问题。VidEngram 使用 Qwen2.5-Omni 对视频和音频进行联合字幕生成，将片段记忆整合为事件层和实体层，并使用配备专用检索工具的纯文本智能体规划器来生成有据可依的答案。其网页界面使这一过程可被检视：用户可以观看记忆的形成过程、检查检索轨迹、在视频播放器中验证所引用的证据，并探索实体之间的关系。在完整的 OmniVideoBench 上，配备 GPT-5-mini 规划器的 VidEngram 优于强大的单次处理 Qwen2.5-Omni 基线（37.0% 对 32.3%）。

    arXiv:2611.18578v1 Announce Type: cross  Abstract: Long-video question answering is wasteful and lossy when a multimodal model must re-encode the same video for every query: context limits force sparse sampling or truncation, and no reusable representation remains for later questions. We present VidEngram, an agentic video QA system that builds a persistent hierarchical memory once at ingest and answers later questions from that memory. VidEngram captions video and audio jointly with Qwen2.5-Omni, consolidates segment memories into episode and entity layers, and uses a text-only agentic planner with specialized retrieval tools to produce grounded answers. The web interface makes this process inspectable: users can watch memory formation, inspect retrieval traces, verify cited evidence in the video player, and explore entity relationships. On the full OmniVideoBench, VidEngram with a GPT-5-mini planner outperforms a strong single-pass Qwen2.5-Omni baseline (37.0% vs. 32.3%), with gains 
    
[^228]: GT-FENG：语言模型事实评估与规范性依据的通用理论

    GT-FENG: General Theory of Factual Evaluation and Normative Grounding in Language Models *Author(s)

    [https://arxiv.org/abs/2611.18279](https://arxiv.org/abs/2611.18279)

    GT-FENG是首个统一上下文/时间正确性与基于规则的错误归因的通用框架，通过对照适用规则检验语言模型答案并追溯错误来源，在Medicaid收入上限问答中使仅匹配其他州数值的错误答案减少68-78%。

    

    语言模型的回答是否正确往往取决于上下文：同一数值在某一司法辖区、日期或法律定义下可能是正确的，而在另一种条件下则是错误的。我们提出了GT-FENG（事实评估与规范性依据的通用理论），这是首个将上下文/时间正确性与基于规则的错误归因统一起来的通用框架，适用于法律、政策及其他受规则约束的领域。GT-FENG依据适用规则对答案进行检查，并识别出与错误答案相匹配的已记录规则。研究表明，当多条规则给出相同数值时，仅凭答案本身无法判断模型使用了哪条规则。在来自两个模型的Medicaid收入上限回答中，记录所询问州自身的数据使得仅匹配其他州数值的错误答案数量减少了68-78%，且此后超过一半的稳定错误答案会匹配两个或更多的已记录来源。GT-FENG为语言模型评估提供了基础。

    arXiv:2611.18279v1 Announce Type: new  Abstract: Whether a language model's answer is correct often depends on context: the same value can be right under one jurisdiction, date, or legal definition and wrong under another. We introduce GT-FENG (General Theory of Factual Evaluation and Normative Grounding), the first general framework to unify contextual/temporal correctness and rule-based error attribution across law, policy, and other rule-governed domains. GT-FENG checks answers against the applicable rules and identifies documented rules that match incorrect answers. It shows that when multiple rules give the same value, the answer alone cannot tell which rule the model used. In Medicaid income-limit answers from two models, documenting the asked state's own records reduces the number of wrong answers that match only other states' values by 68-78%, and more than half of stable wrong answers then match two or more documented sources. GT-FENG provides a foundation for evaluating langu
    
[^229]: 评估用于结构化电子健康记录预测任务的预训练语言语义

    Evaluating Pretrained Language Semantics for Structured EHR Prediction Tasks

    [https://arxiv.org/abs/2611.17053](https://arxiv.org/abs/2611.17053)

    该论文提出tEHR框架，将临床事件转化为自然语言描述并通过持续预训练和判别式微调适配预训练语言模型，证明预训练语言语义能显著提升结构化EHR预测性能，其中Mistral-7B在四个MIMIC-IV任务上超越了各类基线方法。

    

    结构化电子健康记录（EHR）模型通常将临床概念表示为特定于数据集的标记，从而丢弃了其相关的自然语言语义。因此，概念之间的关系在很大程度上只能从EHR共现中学习，而无法直接利用预训练的语言知识。我们通过tEHR框架研究预训练语言语义对纵向EHR预测的贡献，该框架将临床事件转换为自然语言描述，并通过EHR特定的持续预训练和任务特定的判别式微调来适配预训练自回归语言模型。我们在四个MIMIC-IV预测任务上对tEHR进行了评估，其中Mistral-7B相比结构化数据基线、基于代码的基线和基于大语言模型的基线取得了最强的整体性能。控制变量的消融实验证明了词汇基础、数值以及两个适配阶段的重要性。

    arXiv:2611.17053v2 Announce Type: replace  Abstract: Structured Electronic Health Record (EHR) models typically represent clinical concepts as dataset-specific tokens, discarding their associated natural-language semantics. Consequently, relationships between concepts must largely be learned from EHR co-occurrence without directly exploiting pretrained linguistic knowledge. We investigate the contribution of pretrained language semantics to longitudinal EHR prediction through tEHR, a framework that converts clinical events into natural-language descriptions and adapts pretrained autoregressive language models through EHR-specific continued pretraining and task-specific discriminative fine-tuning. We evaluate tEHR across four MIMIC-IV prediction tasks, where Mistral-7B achieves the strongest overall performance against structured-data, code-based, and LLM-based baselines. Controlled ablations demonstrate the importance of lexical grounding, numerical values, and both adaptation stages. 
    
[^230]: 我们即我们的记忆：面向长期角色对话的心理状态驱动记忆

    We Are What We Remember: Mental-State-Driven Memory for Long-Term Character Dialogue

    [https://arxiv.org/abs/2611.16223](https://arxiv.org/abs/2611.16223)

    提出受认知心理学自我记忆系统启发的MindMem框架，通过异步将共同经历整合为角色、用户及关系的心理状态来驱动主动式记忆推理与用户需求预测，并引入MemReal-Bench基准评估超越事实回忆的历史敏感对话适应性。

    

    人类对话中的共同历史不仅仅是一个事实存储库，更是不断演进的关系与相互理解的基础。然而，主流的对话记忆系统依赖于对外部事实库的被动语义相似度检索，经常检索到主题相似的记录，却遗漏了在关系层面至关重要的经历。受认知心理学中自我记忆系统（Self-Memory System, SMS）的启发，我们提出了MindMem——一个由心理状态驱动的记忆框架，它能够异步地将共同经历整合为与源头关联的、关于角色、用户以及二者关系的心理状态。在每一轮对话中，一个专门的心理状态推理器（Mind）会结合这些持续存在的心理状态，以预测用户潜在的需求、形成主动的意图，并找回表层匹配所忽视的支持性情节证据。为了评估超越事实回忆的历史敏感适应性，我们引入了MemReal-Bench，这是一个纵向（摘要内容在此处被截断）

    arXiv:2611.16223v2 Announce Type: replace  Abstract: Shared history in human dialogue is not merely a repository of facts, but the foundation of evolving relationships and mutual understanding. However, mainstream conversational memory systems rely on passive semantic similarity over external fact stores, frequently retrieving topically similar records while missing relationally pivotal experiences. Inspired by cognitive psychology's Self-Memory System (SMS), we propose MindMem, a mental-state-driven memory framework that asynchronously consolidates shared experiences into source-linked mental states of the character, the user, and their relationship. At each turn, a specialized mental-state reasoner (Mind) contextualizes these persistent states to anticipate latent user needs, form proactive intentions, and recover supporting episodic evidence that surface-level matching overlooks. To evaluate history-sensitive adaptation beyond factual recall, we introduce MemReal-Bench, a longitudin
    
[^231]: 面向可扩展多语言文本行人检索的视觉锚定语义学习与对齐

    Vision-Grounded Semantic Learning and Alignment for Scalable Multilingual Text-Based Person Retrieval

    [https://arxiv.org/abs/2611.15328](https://arxiv.org/abs/2611.15328)

    该论文提出了首个大规模多语言行人预训练数据集MultiPedes（包含四种语言的1000万图文对），并引入视觉锚定的语义学习与对齐方法，解决了多语言文本行人检索中缺乏预训练数据和英语中心偏差的问题。

    

    多语言基于文本的行人检索（TBPR）旨在利用多种语言的文本检索行人个体。尽管具有重要的实用价值，但现有研究主要聚焦于仅英语的检索，忽视了这一领域。此前唯一一次针对多语言TBPR的尝试存在以下问题：（1）缺乏大规模的多语言预训练数据集；（2）用于多语言文本编码和跨模态对齐的模型架构欠佳，其采用的共享编码器忽视了语言异质性，并且依赖英语作为跨模态对齐的枢轴，从而引入了以英语为中心的偏差。为此，我们贡献了MultiPedes——首个大规模、以行人为中心的多语言预训练数据集。该数据集包含涵盖四种语言的1000万图像-文本对，由多语言质量感知的数据构建流程构建而成。我们还提出了ViSeLA（视觉锚定语义学习与对齐）……（摘要在此处被截断）

    arXiv:2611.15328v2 Announce Type: replace-cross  Abstract: Multilingual Text-Based Person Retrieval (TBPR) aims to retrieve individuals with texts in multiple languages. Despite its practical importance, it has been overlooked by existing research, which mainly focuses on English-only retrieval. The only prior attempt at multilingual TBPR suffers from (1) the absence of a large-scale, multilingual pretraining dataset and (2) a suboptimal model architecture for multilingual text encoding and cross-modal alignment, which employs a shared encoder that overlooks linguistic heterogeneity and relies on English as a pivot for cross-modal alignment, introducing English-centric bias. For this, we contribute MultiPedes, the first large-scale, pedestrian-centric multilingual pretraining dataset. It contains 10 million image-text pairs across four languages, constructed by a multilingual quality-aware data construction pipeline. We also propose ViSeLA (Vision-grounded Semantic Learning and Alignme
    
[^232]: 深入审视语言模型内省：探测信息与报告

    A Closer Look at Language Model Introspection: Probing Information and Reporting

    [https://arxiv.org/abs/2611.15267](https://arxiv.org/abs/2611.15267)

    该研究通过对比线性探测分类器与模型自身的零样本回答发现：在激活扰动定位任务中，探测分类器能轻松达到 98.3% 的准确率而模型报告仍处于随机水平，表明语言模型内省的表现可能主要源于报告易于线性恢复的信息，而非真正的自我反思能力。

    

    研究语言模型内省的一种方法是测试模型能否识别其激活值在何处受到扰动。我们训练一个简单的线性分类器，从模型的激活值中识别被扰动的句子，然后将其预测与模型自身的回答在不同预训练阶段和注入强度下进行比较。这实际上检验的是：模型被要求报告的信息是否本身就已经容易被恢复。如果是这样，那么成功的定位任务可能比表面上看起来要求更低，并且可能无法推广到激发内省研究的更广泛用途上，例如忠实地解释模型的推理过程，或帮助研究者评估其回答的可靠性。在 OLMo 3 32B 模型上，当在第 0 层注入四倍于典型激活范数的扰动、并在第 24 层进行探测读取时，监督式探测分类器在 84 亿训练 token 后达到 98.3% 的准确率，而模型的零样本回答仍停留在随机水平（11.1%）。

    arXiv:2611.15267v2 Announce Type: replace  Abstract: One way to study language-model introspection is to test whether models can identify where their activations have been perturbed. We train a simple linear classifier to identify the perturbed sentence from the model's activations, then compare its predictions with the model's own answers across pretraining and injection strengths. This tests whether the model is being asked to report information that is already easy to recover. If so, successful localization may be less demanding than it first appears and may not extend to the broader uses that motivate introspection research, such as faithfully explaining a model's reasoning or helping researchers assess the reliability of its answers. With an injection four times the typical activation norm at layer 0 and probe readout at layer 24 in OLMo 3 32B, the supervised probe reaches 98.3 % accuracy after 8.4 billion training tokens, while the model's zero-shot answer remains at chance (11.1
    
[^233]: 构造性机构之上的类型化知识：一种面向可验证科学的证明相关类型论

    Typed Knowledge over Constructive Institutions: A Proof-Relevant Type Theory for Verified Science

    [https://arxiv.org/abs/2611.15108](https://arxiv.org/abs/2611.15108)

    本文提出了面向可验证科学知识基底 Eigenius 的证明相关依赖类型论，通过摒弃判断式证明无关性，使系统能够追踪并比较同一命题的多个独立推导、保持强规范化与传递性的等价判定，并支持在实验或模型失效时计算反事实撤回。

    

    诸如 Coq、Agda 和 Lean 4 之类的数学证明助手在定义等价中内置了证明无关性（即对于 P : Prop 且 p, q : P，有 p ≡ q），从而抹去了定理究竟是由哪条推导所建立的。虽然这对于基于固定公理的数学而言是自然的，但判断式证明无关性与可验证的科学知识库不相容：后者必须检查并比较同一命题的多个独立推导以评估其契合度，并在某个实验、试剂或模型失效时计算反事实的撤回。此外，将定义式证明无关性与子单例消去相结合会破坏强规范化，并使算法化的等价判定失去传递性。我们提出了 Eigenius 的类型论与机构论基础，Eigenius 是一个面向科学的可验证知识基底。首先，我们构造了一个具有累积宇宙的依赖类型内核，包含非直谓的 Prop、以及带有索引强制的子单……（原文摘要在此处截断）

    arXiv:2611.15108v1 Announce Type: cross  Abstract: Mathematical proof assistants such as Coq, Agda, and Lean 4 build proof irrelevance into definitional conversion ($p \equiv q$ for $p, q : P$ with $P : Prop$), erasing which derivation established a theorem. While natural for mathematics over fixed axioms, judgmental proof irrelevance is incompatible with verified scientific stores, which must inspect and compare independent derivations of the same claim to assess consilience and compute counterfactual retractions when an assay, reagent, or model is invalidated. Moreover, combining definitional proof irrelevance with subsingleton elimination breaks strong normalization and renders algorithmic conversion non-transitive.   We present the type-theoretic and institutional foundations of Eigenius, a verified knowledge substrate for science. First, we formulate a dependent kernel with cumulative universes ($Prop, Set, Type_k$), impredicative $Prop$, inductive families with index-forced subsi
    
[^234]: 一种面向204种语言的多词分词器：即使在英语上所需词元也比GPT-4o更少

    A Multi-Word Tokenizer for 204 Languages That Needs Fewer Tokens Than GPT-4o's, Including on English

    [https://arxiv.org/abs/2611.13059](https://arxiv.org/abs/2611.13059)

    该论文提出Havon多词分词器系列，通过在约200种语言上从头训练并引入多词训练阶段，在包括英语在内的161种语言上所需词元数量少于GPT-4o的分词器，在印度语言上的词元数中位数仅为GPT-4o的0.74倍。

    

    印度诸语言在GPT-4o的分词器上处理相同内容时，需要1.6到14倍于英语的词元数量，这使其服务成本更高、速度更慢。我们在约200种语言的人类书写文本上从头训练分词器。Havon v3是一个字节级BPE分词器，拥有262,144个词元，并配备一个预分词器，该预分词器能够将组合符号与其字母保持在一起，并对无空格书写的文字系统进行分块。在FLORES-200开发测试集上，它在全部25种印度语言上所需的词元数量少于GPT-4o的o200k_base（中位数为GPT-4o的0.74倍），在204种语言中的172种上所需词元更少，但在英语上多7%。Havon v4在v3的基础上增加了第二个多词训练阶段。在恰好与GPT-4o相同的200,019词元词表规模下，v4在英语上比GPT-4o少用6.1%的词元，在204种语言中的161种以及全部25种印度语言上所需词元更少（中位数为0.74）；在相同规模下，v3在英语上多9.4%的词元，仅在133种语言上占优。在262,144词元规模下，v4需要少9.6%的词元

    arXiv:2611.13059v1 Announce Type: cross  Abstract: Languages of India need 1.6 to 14 times as many tokens as English on GPT-4o's tokenizer for the same content, which makes them more expensive and slower to serve. We train tokenizers from scratch on human-written text in about 200 languages. Havon v3 is a byte-level BPE tokenizer with 262,144 tokens and a pre-tokenizer that keeps combining marks with their letters and chunks scripts written without spaces. On FLORES-200 devtest it needs fewer tokens than GPT-4o's o200k_base on all 25 languages of India (median 0.74 of GPT-4o's count) and on 172 of 204 languages, but 7% more on English. Havon v4 adds a second, multi-word training stage (SuperBPE) to v3. At exactly GPT-4o's vocabulary size of 200,019 tokens, v4 needs 6.1% fewer tokens than GPT-4o on English and fewer on 161 of 204 languages and all 25 languages of India (median 0.74); at the same size, v3 needs 9.4% more on English and wins on 133. With 262,144 tokens, v4 needs 9.6% fewe
    
[^235]: 2026年的印度语言“代币税”：测量分词器差异及其对印度各语言的成本

    The Indic Token Tax in 2026: Measuring Tokenizer. Disparity and Its Cost for the Languages of India

    [https://arxiv.org/abs/2611.13058](https://arxiv.org/abs/2611.13058)

    该研究测量了九个主流分词器对印度25种语言的“代币税”，发现印度语言所需token量最高可达英语的13.7倍（GPT-4o分词器），导致印度语言用户在商业LLM使用中面临更高的费用、更慢的速度和更小的有效上下文，而Sarvam分词器是唯一将差异控制在2.6倍以内的第三方选择。

    

    商业语言模型API按token计费，因此当分词器处理某种语言的相同内容需要更多token时，就会使该语言的使用更昂贵、速度更慢，并且在固定上下文窗口中能容纳的文本更少。我们在FLORES-200中全部25种印度语言（2,009句与英文对齐的专业翻译句子）以及来自PHINC的13,732对人工翻译的印地英语-英语社交媒体文本上，测量了九个公开分词器的这种差异，包括GPT-4、GPT-4o、Qwen2.5、DeepSeek-V3、Mistral-Nemo以及四个Sarvam模型的分词器。在GPT-4o的分词器上，相同内容所需的token是英语的1.57倍（印地语）到5.01倍（奥里亚语），而使用Ol Chiki文字的桑塔利语高达13.7倍；GPT-4的分词器最高需要12.8倍。Sarvam-30B/105B是唯一在所有语言上都保持在2.6倍以下（中位数1.53）的第三方分词器。在一个使用截至11月5日获取的官方标价的示例场景中……

    arXiv:2611.13058v1 Announce Type: cross  Abstract: Commercial language model APIs bill per token, so a tokenizer that needs more tokens for the same content in one language makes that language more expensive, slower, and able to fit less text in a fixed context window. We measure this disparity for nine public tokenizers, including those of GPT-4, GPT-4o, Qwen2.5, DeepSeek-V3, Mistral-Nemo and four Sarvam models, on all 25 languages of India available in FLORES-200 (2,009 professionally translated sentences aligned with English) and on 13,732 human-translated Hinglish-English social media pairs from PHINC. On GPT-4o's tokenizer the same content needs between 1.57 times (Hindi) and 5.01 times (Odia) the tokens of English, and 13.7 times for Santali in Ol Chiki script; GPT-4's tokenizer needs up to 12.8 times. Sarvam-30B/105B is the only third-party tokenizer that stays below 2.6 times on every language (median 1.53). In an illustrative scenario using official list prices accessed on 5 N
    
[^236]: LLM智能体通过共享内存实现的无提示协调：一项加锁双重实现的复现与扩展研究

    Unprompted coordination through shared memory in LLM agents: a locked two-implementation replication and extension

    [https://arxiv.org/abs/2611.12412](https://arxiv.org/abs/2611.12412)

    本研究通过双重独立实现的复现与扩展首次证实：即使提示中完全没有提及其他智能体、共享或协作的存在，LLM智能体也能仅凭共享内存状态自发产生协调行为。

    

    大型语言模型智能体之间的协调已成为AI安全与软件工程的核心问题，因为这类智能体通常以共享基础设施的群体形式部署。现有研究通常会告知智能体伙伴的存在，或直接要求它们进行协调。而仅通过共享状态——在没有任何信息透露其他参与者存在的情况下——协调是否会自发产生，这对于在智能体之间共享缓存与内存的部署场景而言是一个关键问题，但该问题鲜有研究，据我们所知也从未经过独立复现；本研究两者兼顾。四个智能体各自收到一个简短的算术任务和一个键值存储，每个智能体都将该存储视为自己的暂存空间；其中两个智能体持有同一计算中互补的两半，且所有提示均未提及智能体、共享或协作。为保证可靠性，实验环境依据书面协议独立重建，两种实现均在哈希锁机制下运行。

    arXiv:2611.12412v2 Announce Type: replace-cross  Abstract: Coordination between large language-model agents has become a central question for AI safety and software engineering, as such agents are deployed in groups that share infrastructure. Existing research typically tells the agents that partners exist or asks them to coordinate. Whether coordination arises through shared state alone, with nothing disclosing that anyone else is present, is the key question for deployments that share caches and memory between agents, yet it has rarely been examined, and to our knowledge not with independent replication; this study does both. Four agents each received a short arithmetic task and a key-value store presented to each as its own scratch space; two held complementary halves of one computation, and no prompt mentioned other agents, sharing or collaboration. For reliability, the environment was rebuilt independently from a written protocol, and both implementations ran under hashed locks on
    
[^237]: Infrathon：面向真实世界、长周期AI基础设施优化的语言智能体基准测试

    Infrathon: Benchmarking Language Agents for Real-World, Long-Horizon AI Infra Optimization

    [https://arxiv.org/abs/2611.11995](https://arxiv.org/abs/2611.11995)

    该论文提出Infrathon长周期基准，通过对六个模型进行24小时、累计超10,000 GPU小时的评估，发现语言智能体能持续改进真实AI基础设施，模型排名随评估时长变化，且智能体可发现超越人类设计的方案（如将KV缓存压缩比从6.82倍提升至19.86倍）。

    

    在这项工作中，我们研究前沿智能体能否在长周期内持续改进真实世界的AI基础设施。我们提出了Infrathon，这是一个涵盖不同层级的多样化、长周期AI基础设施优化任务基准。我们在150次运行中对六个模型进行了评估，每次运行预算为24小时，总计超过10,000 GPU小时。我们的结果表明，模型排名会随评估周期的变化而变化，同时150次运行中有135次在预算的后半段仍能取得进一步的改进。这些发现表明，较短的评估可能会同时忽略优化收益以及模型相对性能的变化。我们还发现智能体能够找到优于人类设计基线的解决方案：Claude Opus 5.5将平均KV缓存压缩比从人类设计的Quant-VideoGen编解码器的6.82倍提升至19.86倍。对智能体轨迹的分析识别出任务违规、不可靠的测试以及过早……（摘要在此处截断）

    arXiv:2611.11995v2 Announce Type: replace  Abstract: In this work, we study whether frontier agents can continually improve real-world AI infrastructure over long horizons. We introduce Infrathon, a benchmark of diverse, long-horizon AI infrastructure optimization tasks across different levels. We evaluate six models across 150 runs with 24-hour budgets, totaling over 10,000 GPU-hours. Our results show that model rankings vary with the evaluation horizon, while 135 of the 150 runs achieve further improvements during the second half of their budgets. These findings indicate that shorter evaluations can overlook both optimization gains and changes in relative model performance. We also find agents can discover solutions that outperform human-designed baselines: Claude Opus 5.5 increases the mean KV-cache compression ratio from $6.82\times$ with the human-designed Quant-VideoGen codec to $19.86\times$. Analysis of agent trajectories identifies task violations, unreliable testing, and prem
    
[^238]: 通过逐层与时间分辨探测理解自监督语音模型何时何地学习音素与韵律线索

    Understanding When and Where Self-Supervised Speech Models Learn Phonemes and Prosodic Cues through Layer-Wise and Time-Resolved Probing

    [https://arxiv.org/abs/2611.11249](https://arxiv.org/abs/2611.11249)

    该研究通过对预训练各检查点的所有Transformer层进行时间分辨探测，发现声学韵律线索和音素边界从预训练最早期即可解码且在各层间保持稳定，而音素身份在预训练中逐步构建并集中于中上层，表明信息习得顺序取决于其相对声学表面的抽象程度。

    

    自监督语音模型（S3Ms）编码了丰富的声学、韵律和语音信息层级，但其在预训练过程中何时何地形成仍不清楚。我们在两个匹配模型的各个预训练检查点上，对每一个Transformer层训练冻结编码器探测器：这两个模型分别是HuBERT-Base（其第二阶段是一个在新的伪标签目标上训练的单独初始化模型）和单目标wav2vec 2.0-Base。我们的五个任务构成了一个从声学到抽象的梯度：绝对音高（F0）、说话人相对音高、词汇重音、音素边界以及38类音素身份识别。声学-韵律线索和音素边界从最早的检查点即可解码，且在各层之间几乎没有变化；而音素身份则在预训练过程中逐步构建，在相同更新次数下，在HuBERT第二阶段目标的作用下重新定位到中上层。因此，信息的习得顺序取决于其相对于声学表面的抽象程度。

    arXiv:2611.11249v2 Announce Type: replace  Abstract: Self-supervised speech models (S3Ms) encode a rich hierarchy of acoustic, prosodic, and phonetic information, yet when and where it emerges during pretraining remains unclear. We train frozen-encoder probes on every Transformer layer across pretraining checkpoints of two matched models: HuBERT-Base, whose stage 2 is a separately initialized model trained on new pseudo-label targets, and single-objective wav2vec 2.0-Base. Our five tasks span an acoustic-to-abstract gradient: absolute pitch (F0), speaker-relative pitch, lexical stress, phoneme boundaries, and 38-way phoneme identity. Acoustic-prosodic cues and phone boundaries are decodable from the earliest checkpoints and barely shift across layers, whereas phoneme identity is constructed during pretraining, re-localizing to upper-middle layers under HuBERT's stage-2 targets at a matched update count. Acquisition order therefore tracks abstraction from the acoustic surface, not the s
    
[^239]: 良性训练的隐状态探针能否在对抗条件下检测工具错误？

    Do Benign-Trained Hidden-State Probes Detect Tool Errors Under Adversarial Conditions?

    [https://arxiv.org/abs/2611.11172](https://arxiv.org/abs/2611.11172)

    该论文在生物医学问答任务中通过多代递归实验，首次系统比较了语言模型在自身生成答案上递归训练（合成数据反馈循环）与持续使用人工数据的差异，揭示错误与分布偏差如何随代次累积。

    

    反复在语言模型自身生成的数据上进行训练，可能会形成一种合成数据反馈循环，使错误和分布偏差被重新引入后续的训练数据集中。本文使用PubMedQA数据集以及两种规模的Qwen2.5模型（0.5B和3B参数），在生物医学问答（QA）任务中研究这一过程。该研究将递归合成数据条件（即第G(k+1)代模型在第G(k)代模型生成的答案上进行训练）与反复使用原始人工训练数据的人类控制条件进行对比。研究在G0至G3四个代次上，使用两个随机种子（42和123）以及固定的1,000条专家标注样本评估集进行评估。评估指标包括疾病和化学实体F1分数、上下文支持率、词汇与语义相似度、答案长度、重复率等其他评估指标。两种模型规模和两个随机种子下的递归条件均显示出……（原文摘要在此处截断，句子不完整）

    arXiv:2611.11172v2 Announce Type: replace-cross  Abstract: Repeatedly training language models on their own generated data may create a synthetic-data feedback loop in which errors and distributional biases are reintroduced into subsequent training datasets. This paper studies that process in biomedical question answering (QA) using PubMedQA and two Qwen2.5 model sizes, 0.5B and 3B parameters. The study compares a recursive synthetic-data condition, in which generation G(k+1) is trained on answers produced by G(k), against a Human-Control condition that repeatedly uses the original human training data. The study evaluates across four generations from G0-G3 with two random seeds (42 and 123) and a fixed evaluation set of 1,000 expert-labeled samples. The evaluation includes disease and chemical entity F1, context-supported rate, lexical and semantic similarity, answer length, repetition rate, and other evaluation metrics. The Recursive condition for both model sizes and both seeds showe
    
[^240]: 超越截断？蕴含许可的省略与LLM解码中语义保持的极限

    Beyond Truncation? Entailment-Licensed Elision and the Limits of Semantic Preservation in LLM Decoding

    [https://arxiv.org/abs/2611.08975](https://arxiv.org/abs/2611.08975)

    该论文证明了现有免训练的解码期冗余删除方法本质上都只是截断，并提出ELIDE方法，利用自然语言推理模型在每个句子边界判断被删内容是否被保留内容所蕴含，从而实现真正超越截断的语义保持式回答精简。

    

    指令微调的大型语言模型（LLM）的回答长度往往超过请求的实际需要。我们探究在解码循环内部加入一个免训练的控制器能否在保持内容的前提下去除这些冗余，并以最简单的基线作为评判标准：将模型自身的回答截断到相同长度。一个基于熵门控的logits处理器在MT-Bench开发问题上删除了25.5%-34.9%的词元，在密封测试集上删除了19.9%-34.2%的词元，且均落在预先注册的BERTScore非劣性边界之内；然而我们通过理论证明和审计确认，其输出始终是未修改回答的前缀：所谓的节省本质上就是截断。对23个免训练机制族的调研发现，在没有oracle标签的情况下，没有任何机制能比截断更好地选择内容。因此，我们提出了蕴含许可的解码中省略（ELIDE），它在每个句子边界处，仅当自然语言推理模型判定该片段可由所保留内容蕴含时，才向读者隐去该片段……

    arXiv:2611.08975v1 Announce Type: new  Abstract: Instruction-tuned large language models (LLMs) answer at greater length than a request needs. We ask whether a training-free controller inside the decoding loop can remove that surplus while preserving content, judged against the simplest possible baseline: truncating the model's own answer to the same length. An entropy-gated logits processor removes 25.5-34.9% of tokens on MT-Bench development questions and 19.9-34.2% on a sealed test split within pre-registered BERTScore non-inferiority margins, yet we prove and confirm by audit that its output is always a prefix of the unmodified answer: the saving is truncation. A survey of 23 training-free mechanism families finds none that selects content better than truncation without oracle labels. We therefore introduce Entailment-Licensed In-Decoding Elision (ELIDE), which, at each sentence boundary, withholds a span from the reader only when a natural-language inference model judges it entail
    
[^241]: 类型化决策编码器中的标签措辞增强：对不熟悉的词语有效，对缩写或代号无效

    Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes

    [https://arxiv.org/abs/2611.08439](https://arxiv.org/abs/2611.08439)

    论文对俄语类型化决策交叉编码器 Ru-Jev 引入标签多形式措辞增强，发现该增强对由普通词和外来词构成的标签词汇表能带来一致的准确性提升（最高达 0.24），但对缩写和代号类标签没有增益。

    

    类型化决策模型会对调用方在请求时提供的选项进行评分，因此其准确性取决于模型开发者无法控制的标签措辞。我们比较了 Ru-Jev 的两个版本——这是一个在十三项俄语任务上训练的 3.07 亿参数多语言交叉编码器；后一版本增加了一种增强方法，将标签以多种表面形式呈现。在保持问题不变、仅改变十八个路由标签措辞的条件下，我们评估了十套标签词汇表，这些词汇表由三种独立流程生成，并检查了与训练数据的重合情况；每题使用五个候选标签，其中八套词汇表还使用了十八个候选标签进行评估。在由普通词汇或外来词（源自英语的行话、办公室俚语、部门名称）构成的保留词汇表上，增强版本在所有设置下都更加准确，提升幅度最高达 0.24。这十四项增益中有十一项在 Holm 校正后仍然显著（十八个候选标签的俚语以及描述性……（摘要在此处截断）

    arXiv:2611.08439v1 Announce Type: cross  Abstract: Typed-decision models score options that the caller supplies at request time, so their accuracy depends on label wording that the model's developer does not control. We compare two releases of a 307M-parameter multilingual cross-encoder, Ru-Jev, trained on thirteen Russian-language tasks; the later release adds an augmentation that presents labels in several surface forms. Holding questions fixed and varying only how eighteen routing labels are worded, we evaluate ten label vocabularies, produced by three separate processes and checked for overlap with the training data, with five candidates per question and, for eight of them, also with eighteen. On held-out vocabularies made of ordinary or borrowed words (English-derived jargon, office slang, department names), the augmented release is more accurate in every setting, by up to 0.24. Eleven of these fourteen gains survive a Holm correction (slang with eighteen candidates and descriptiv
    
[^242]: CardioLM：面向心脏病学领域的多语言小型语言模型套件

    CardioLM: A multilingual suite of small language models for the cardiology domain

    [https://arxiv.org/abs/2611.07940](https://arxiv.org/abs/2611.07940)

    该研究提出了多语言心脏病学病例语料库 CardioCCC 以及小型语言模型套件 CardioLM（含编码器 CardioBERTa 和命名实体识别模型 CardioNER），填补了心脏病学领域多语言临床 NLP 资源的空白。

    

    背景：心血管诊疗会在本地语言中产生大量临床自由文本文档。然而，临床自然语言处理（NLP）资源仍集中于英语，且目前尚无心脏病学专用、多语言且实体覆盖广泛的基准。方法：我们提出了两项贡献。第一项是 CardioCCC，一个包含 508 份心脏病学病例报告的语料库，这些报告最初以西班牙语发表，并通过机器翻译、标注投影和临床医生审核扩展至捷克语、荷兰语、英语、意大利语、罗马尼亚语和瑞典语，其标注涵盖疾病、症状、操作和药物。第二项是 CardioLM，包括 CardioBERTa——一组通过对心脏病学和生物医学语料库进行持续预训练而得到的特定语言编码器，以及 CardioNER——一组在 CardioCCC 上微调的命名实体识别（NER）模型。结果：心脏病学专用预训练降低了……

    arXiv:2611.07940v1 Announce Type: cross  Abstract: Background: Cardiovascular care generates large volumes of clinical free-text documentation in local languages. However, clinical natural language processing (NLP) resources remain concentrated in English, and no existing benchmark is cardiology-specific, multilingual, and broad in entity coverage.   Methods: We present two contributions. The first is CardioCCC, a corpus of 508 cardiology case reports originally published in Spanish and extended to Czech, Dutch, English, Italian, Romanian, and Swedish through machine translation, annotation projection, and clinician review, with annotations covering diseases, symptoms, procedures, and medications. The second is CardioLM, comprising CardioBERTa, a set of language-specific encoders obtained by continued pretraining on cardiology and biomedical corpora, and CardioNER, a set of named entity recognition (NER) models fine-tuned on CardioCCC.   Results: Cardiology-specific pretraining reduced
    
[^243]: 立法法案讨论的话语分析

    Discourse Analysis of Legislative Bill Discussions

    [https://arxiv.org/abs/2611.05381](https://arxiv.org/abs/2611.05381)

    本文提出了一种用于解析立法委员会听证会修辞结构的端到端流水线和修辞结构语法，其中55.21%的加州州法案讨论可被该语法解析，从而改进了立法话语分析与民主问责。

    

    立法听证会是独特的交流事件，具有人类可识别的自身结构和逻辑，然而以往的研究仅将其视为超越文字记录的独特现象来分析。在更深层次上理解立法听证会是分析民主机构内部工作与权力关系的重要一步。在这项工作中，我们提出了一个新颖的端到端流水线，用于从立法委员会听证会中解析修辞结构，旨在改进立法话语分析和民主问责。我们还提出了一种修辞结构语法，并发现我们数据集中55.21%的加州州法案讨论可以用该语法进行解析。我们在超过2000个标注话语（对应100场听证会）的随机子集上，使用两个话语片段分类器评估了我们的流水线。我们发现，基于逻辑回归的分类器结合基于语法的输出掩码和话语标签平滑方法……（原文此处截断）

    arXiv:2611.05381v1 Announce Type: new  Abstract: Legislative hearings are distinct communication events with their own human recognizable structure and logic, yet they have been analyzed as distinct phenomena beyond their transcripts. Understanding legislative hearings at a deeper level is a major step in analyzing the work and power relationships within democratic bodies. In this work, we present a novel end-to-end pipeline for parsing rhetorical structures from legislative committee hearings with the goal of improving legislative discourse analysis and democratic accountability. We also propose a rhetorical structure grammar and find that 55.21% of California state bill discussions in our dataset can be parsed with this grammar. We evaluate our pipeline with two utterance section classifiers on a random subset of over 2,000 annotated utterances corresponding to 100 hearings. We find that a logistic regression-based classifier with grammar-based output masking and utterance tag smooth
    
[^244]: NitiBench+：一个涵盖17个领域的时序泰国法律问答基准

    NitiBench+: A Temporal Thai Legal Question Answering Benchmark Across 17 Domains

    [https://arxiv.org/abs/2611.03576](https://arxiv.org/abs/2611.03576)

    NitiBench+是一个包含1,670个条目、覆盖17个领域、同时收录判决日有效与当前法条文本的泰国时序法律问答基准，实验表明模型使用过时法条文本会显著损害法律问答表现。

    

    法律会随时间变化。法院依据判决当日有效的法律来裁决案件，而现有的法律问答基准却按照今天的法律来给答案评分。我们检查了泰国四个法院的156,650份判决，发现其中高达67%引用了此后已被修订或废止的法律。我们推出NitiBench+，它是我们此前NitiBench的续作，是一个泰国法律问答基准，包含基于三个法院判决构建的1,670个条目，覆盖17个法律领域。对于每个被引用的法条，该基准同时保存判决当日有效的文本和当前文本，其检索语料库则收录了泰国每一部法典每个条目的所有带日期的历史版本。答案的评分采用泰国律师公会的评分细则和律师标准分数，并根据模型引用的法条与法院实际引用法条之间的重合度进行评判。报告分为两部分：第一部分测量了使用错误文本在13个开放权重模型上造成的损失，阅读已修订法律的当前文本会使律师标准分数下降0.25，遗漏已废止法律则会使分数下降0.79。

    arXiv:2611.03576v2 Announce Type: new  Abstract: Law changes over time. A court decides a case under the law in force on that day, and legal QA benchmarks grade answers against today's law. We checked 156,650 decisions of four Thai courts. Up to 67% of them cite law that has since been amended or repealed. We introduce NitiBench+, the sequel of our NitiBench, a Thai legal QA benchmark with 1,670 items across 17 domains, built from decisions of three courts. For each cited section it holds both the text in force on the decision date and the current text, and its retrieval corpus holds every dated version of every Thai statute section. Answers are graded on the Thai Bar Council rubric, the Bar-standard score, and on the overlap between the cited sections and the court's. The report has two parts. Part I measures the cost of the wrong text on 13 open-weight models. Reading the current text of amended law drops the Bar-standard score by 0.25 and missing repealed law drops it by 0.79. The p
    
[^245]: 以模型读取的方式书写事实：破解两跳诅咒

    Write Facts the Way Models Read Them: Lifting the Two-Hop Curse

    [https://arxiv.org/abs/2611.01519](https://arxiv.org/abs/2611.01519)

    论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。

    

    arXiv:2611.01519v2 公告类型：新 摘要：通过微调学习新事实的语言模型能够复述该事实，却无法用它进行推理：教会模型某个人是新国家公民之后，它仍然无法在不借助思维链的情况下说出这个人所在国家的国家元首是谁。这种“两跳诅咒”通常被视为推理能力缺陷，而最强的知识编辑方法则通过合成推理数据来解决它。我们证明：推理能力是完好的，错的是事实的书写方式。微调把新事实写成一个“要说出口的答案”，而第二跳推理从中间层开始就把桥接短语（“那个人的国家”）当作实体来读取：在相应层将模型自身对该实体的表征修补进桥接短语，可以基本恢复第二跳推理；然而微调之后，桥接短语与该新实体的接近程度并不比编辑前更高。因此我们提出“读取等价书写”（Legible）：用一个损失项让桥接短语在第二跳读取它的那一层（摘要在此处截断）

    arXiv:2611.01519v2 Announce Type: new  Abstract: A language model that learns a new fact by fine-tuning can repeat it but cannot reason with it: taught that a person is a citizen of a new country, it still cannot say, without chain-of-thought, who the head of state of that person's country is. This two-hop curse is widely treated as a reasoning deficit, and the strongest knowledge editors answer it with synthesized reasoning data. We show that the reasoning is intact and the writing is wrong. Fine-tuning writes a new fact as an answer to be said, whereas the second hop reads the bridge ("that person's country") as an entity, from the middle layers on: patching the model's own representation of the entity into the bridge at those layers largely restores the second hop, yet after fine-tuning the bridge there is no closer to the new entity than before the edit. We therefore propose read-equivalent writing (Legible): one loss term makes the bridge phrase, at the layer where the second hop 
    
[^246]: S1CogBench：系统1决策模型在认知实验中能做什么与不能做什么

    S1CogBench: What System One Decision Models Can and Cannot Do on Cognitive Experiments

    [https://arxiv.org/abs/2611.00911](https://arxiv.org/abs/2611.00911)

    该论文提出S1CogBench基准，通过24项认知任务、27种评估模式，结合人类反应预测、自主执行与受控任务操纵，系统揭示了系统1决策模型的认知能力与局限，发现了聚合分数无法捕捉的模型行为差异与失败模式。

    

    摘要（arXiv:2611.00911v2，公告类型：replace-cross）：人类与人工智能的决策涉及多种认知能力。系统1（System One）决策模型能够产生快速、结构化的选择，然而其认知优势与局限仍鲜为人知。我们提出了S1CogBench，一个包含24项成熟认知任务和27种评估模式的基准，用于评估七个开放模型以及API模型Jev。该基准结合了人类反应预测、自主执行与受控任务操纵，以揭示聚合分数所遗漏的优势与失败。我们发现被评估的各个模型呈现出截然不同的行为特征：在这些模型中，某些学习任务上的强劲表现与无法收集信息或无法根据反馈进行调整并存，而高准确率也并不保证模型能像人类一样应对变化的任务需求。共享读出控制实验表明，模型排名在一定程度上取决于选择是如何被提取的。与生成式语言模型的独立延迟比较……（原文摘要在此处截断）

    arXiv:2611.00911v2 Announce Type: replace-cross  Abstract: Decision-making in humans and AI involves multiple cognitive abilities. System One decision models produce fast, structured choices. Their cognitive strengths and limitations, however, remain poorly understood. We introduce S1CogBench, a benchmark of 24 established cognitive tasks and 27 evaluation modes, evaluating seven open models and the API model Jev. It combines human-response prediction, autonomous execution, and controlled task manipulations to reveal strengths and failures that aggregate scores miss. We find distinct behavioral profiles across the evaluated models. In these models, strong performance on some learning tasks coexists with failures to gather information or adapt to feedback, while high accuracy does not guarantee human-like responses to changing task demands. Shared readout controls show that model rankings depend partly on how choices are extracted. Separate latency comparisons with generative language m
    
[^247]: 稀疏自编码器归因图由边际决定

    Sparse Autoencoder Attribution Graphs Are Decided at the Margin

    [https://arxiv.org/abs/2611.00606](https://arxiv.org/abs/2611.00606)

    该论文揭示 TopK 稀疏自编码器的特征激活值常贴近截断阈值，导致传统归因图误差较大，并提出一种在相同计算成本下为每个预激活加入一阶变化并重新排序的“边际归因图”方法，将中位相对误差从 0.247 以上降至至多 0.033。

    

    基于稀疏自编码器（SAE）特征的归因图被用于预测消融操作的效果，并在不同训练种子之间进行比较。TopK SAE 在每个 token 上保留最大的预激活值，而特征的“边际”是其预激活值到截断阈值的距离。在来自三个 Pythia 模型的五个公开 TopK SAE 对上，将一个特征置零会在至少 0.74 的情形中使下一层的某个特征跨过截断阈值，而这些跨越承载了响应的大部分能量。对于将某个特征的编码放大三倍的写入操作，基于先前激活特征构建的图的中位相对误差达到 0.247 或更高。而“边际图”方法为每个预激活添加一阶变化并重新排序，在相同计算成本下，误差至多为 0.033。在中位数上，跨越特征的解码变化在范数上达到模型自身变化的至少 1.24 倍。在同一测试床的不同种子之间，特征之间的稀疏映射可以解释一个图的 0.85。

    arXiv:2611.00606v1 Announce Type: cross  Abstract: Attribution graphs over the features of sparse autoencoders (SAEs) are used to predict what an ablation does and are compared across training seeds. A TopK SAE keeps the largest preactivations at each token, and a feature's margin is its preactivation's distance to the cutoff. On five public TopK SAE pairs from three Pythia models, zeroing one feature moves a next layer feature across the cutoff in 0.74 of cases or more, and the crossings carry most of the response's energy. Over writes that scale a feature's code by three factors, the graph built on the features active beforehand has a median relative error of 0.247 or more. The margin graph adds the first order change to every preactivation and reranks, at the same cost, and errs by at most 0.033. At the median, the crossing features' decoded change is 1.24 times the model's own change in norm or more. Between seeds of one testbed, sparse maps between features explain 0.85 of a graph
    
[^248]: 基于智能体软令牌的模型内上下文压缩，面向长时程LLM智能体

    In-Model Context Compression with Agentic Soft Tokens for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2610.41291](https://arxiv.org/abs/2610.41291)

    提出免训练的智能体软令牌压缩方法（ASTC），利用混合GDN-注意力的原生循环动态将已吸收信息压缩为模型内软令牌，在降低长时程LLM智能体计算成本的同时保留关键决策信息。

    

    长时程LLM智能体在解决复杂且持续的任务中扮演着重要角色，其交互上下文会不断增长。然而，这些累积的上下文会带来巨大的计算成本，并掩盖关键的决策相关证据。现有方法通常训练外部压缩器或上下文管理智能体，将上下文重写为较短的硬令牌序列，但这种外部选择可能与LLM不断演化的内部表示不一致，且必然是有损的。在本文中，我们提出了智能体软令牌压缩，这是一种免训练的压缩方法，在LLM智能体内部使用软令牌来处理长时程任务。具体而言，ASTC借助常用的混合门控Delta网络-注意力机制，利用模型原生的循环动态来确定已被吸收的信息，然后将这些信息作为智能体软令牌向前传递，同时将精确值保留为硬令牌。结果……

    arXiv:2610.41291v1 Announce Type: new  Abstract: Long-horizon LLM agents play an important role in solving complex and sustained tasks, with ever-growing interaction contexts. However, these accumulated contexts will raise substantial computational cost and obscure key decision-relevant evidence. Existing methods typically train external compressors or context-management agents to rewrite context as shorter hard-token sequences, but such external selection may not align with the LLM's evolving representation and is necessarily lossy. In this paper, we introduce Agentic Soft-Token Compression (ASTC), a training-free compression method using soft tokens within LLM agent for long-horizon tasks. Specifically, ASTC uses the model's native recurrent dynamics to determine the already absorbed information, leveraging the commonly-used hybrid Gated Delta Network (GDN)-attention. It then carries this information forward as agentic soft tokens while retaining exact values as hard tokens. The resu
    
[^249]: 语言模型在检索增强生成（RAG）中表现出从众偏差：当多数文档支持被投毒的论断时

    Language Models Show Conformity Bias in RAG: When the Majority Endorses a Poisoned Claim

    [https://arxiv.org/abs/2610.41012](https://arxiv.org/abs/2610.41012)

    本研究发现当RAG检索上下文中多数相互一致的投毒文档共同支持错误信息时，语言模型的投毒检测率从25%显著降至10%，揭示了语言模型存在类似社会从众心理的从众偏差，使其更容易被多数派的投毒信息误导。

    

    检索增强生成（RAG）可能使语言模型接触到检索文档中具有误导性的证据。我们研究了相互一致的投毒文档数量增加是否会影响模型检测错误信息的能力。我们从SciFact数据集中随机选取200个问题（随机种子38）进行评估，将包含一篇投毒文档和六篇健康文档的上下文，与包含六篇相互一致的投毒文档和一篇健康文档的上下文进行对比。对于Gemma-3-12b模型，投毒检测率从少数条件下的25%（50/200）下降至多数条件下的10%（20/200）。在两种条件下结果不同的38个问题中，有34个从“被检测出”变为“未被检测出”，仅有4个朝相反方向变化（精确McNemar检验，p = 6.04e-7）。这一模式与语言模型面对多数相互一致的投毒文档时产生的类从众式反应一致。我们的结果提供了初步证据……

    arXiv:2610.41012v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can expose language models to misleading evidence in retrieved documents. We investigate whether increasing the number of mutually agreeing poisoned documents affects a model's ability to detect misinformation. We evaluate 200 randomly selected questions from SciFact (random seed 38), comparing contexts with one poisoned and six healthy documents against contexts with six mutually agreeing poisoned documents and one healthy document. For Gemma-3-12b, poison-detection rates decrease from 25% (50/200) in the minority condition to 10% (20/200) in the majority condition. Among the 38 questions with different outcomes across conditions, 34 change from detected to undetected, while only four change in the opposite direction (exact McNemar test, p = 6.04e-7). This pattern is consistent with a conformity-like response to a majority of mutually agreeing poisoned documents. Our results provide initial evidence 
    
[^250]: CAMR：面向大语言模型持续学习的兼容性感知记忆回放

    CAMR: Compatibility-Aware Memory Replay for Continual Learning in Large Language Models

    [https://arxiv.org/abs/2610.36734](https://arxiv.org/abs/2610.36734)

    CAMR 提出了一种兼容性感知的自适应记忆回放框架，通过在线策略联合控制新样本监督与历史回放，在持续学习中平衡大语言模型对新知识的获取与对已有知识的保留。

    

    持续学习要求大语言模型在获取新知识和能力的同时保留已学到的内容。微调和回放虽然支持知识的获取与保留，但如何判断传入样本是否需要监督、以及应该重温哪些历史样本，仍然是一个挑战。我们提出了兼容性感知记忆回放（CAMR），这是一种自适应框架，能够联合控制当前样本的监督和历史回放。CAMR 将兼容性检查与学习到的记忆门控相结合，引导在线策略在三种动作之间做出选择：结合通用回放学习传入样本、联合强化兼容的相关样本、以及跳过对已验证掌握的重复样本的直接监督。该策略在每次到达的有界回放预算下，依据知识获取、保留和训练成本的反馈进行学习。我们使用 Qwen3-8B-Base 在持续事实更新以及跨任务、跨领域学习场景中评估了 CAMR。

    arXiv:2610.36734v1 Announce Type: new  Abstract: Continual learning requires large language models to acquire new knowledge and capabilities while retaining what they have learned. Fine-tuning and replay support acquisition and retention, but deciding whether incoming examples need supervision and which historical examples to revisit remains a challenge. We propose Compatibility-Aware Memory Replay (CAMR), an adaptive framework that jointly controls current-example supervision and historical replay. CAMR combines compatibility checks with a learned memory gate to guide an online policy over three actions: learning incoming examples with general replay, jointly reinforcing compatible related examples, and skipping direct supervision of verified mastered duplicates. The policy learns from acquisition, retention, and training-cost feedback under a bounded per-arrival replay budget. We evaluate CAMR with Qwen3-8B-Base on continual factual updating and cross-task and cross-domain learning. 
    
[^251]: 形式重于内容？基于大语言模型的同行评审中的语言偏见

    Style over Substance? Linguistic Bias in LLM-based Peer Review Copy title

    [https://arxiv.org/abs/2610.35254](https://arxiv.org/abs/2610.35254)

    该论文提出了一个受控实验框架，通过基于语言学文献推导的细粒度改写指令，系统性地改变论文的语言风格（母语/非母语特征与高/低置信度特征），以研究基于大语言模型的同行评审是否会受到与科学内容无关的语言偏见的影响。

    

    近年来，向主要人工智能会议提交的论文数量急剧增加，大幅加重了研究人员的同行评审工作量。与此同时，大语言模型（LLM）正越来越多地被用于支持同行评审，无论是被个别审稿人使用，还是被会议组织方采用。尽管LLM的能力已有显著提升，但其评估结果可能会受到与科学内容无关的语言特征的影响。在这项工作中，我们引入了一个受控实验框架，通过系统地改变研究论文的语言风格来研究这类效应。具体而言，我们对论文进行改写，使其分别呈现（i）与英语母语或非母语写作相关的语言特征，以及（ii）高置信度或低置信度的语言特征。为了构建这些干预措施，我们以语言学文献为基础推导出细粒度的改写指令，并将其应用于……

    arXiv:2610.35254v1 Announce Type: new  Abstract: The number of paper submissions to major AI conferences has increased dramatically in recent years, substantially increasing the peer-review workload of researchers. At the same time, large language models (LLMs) are increasingly used to support peer review, both by individual reviewers and by conference organizations. While LLM capabilities have improved considerably, their evaluations may be influenced by linguistic characteristics that are unrelated to scientific content. In this work, we introduce a controlled experimental framework for studying such effects by systematically perturbing the linguistic style of research papers. Specifically, we rewrite papers to exhibit (i) linguistic features associated with native or non-native English writing, and (ii) high- or low-confidence linguistic features. To construct these interventions, we derive fine-grained rewriting instructions grounded in the linguistic literature and apply them usin
    
[^252]: 检验人类与大型语言模型中基于形式的语言能产性

    Examining form-based linguistic productivity in people and large language models

    [https://arxiv.org/abs/2610.33433](https://arxiv.org/abs/2610.33433)

    该研究发现人类能够基于抽象的语言形式规则将新动词从主动语态泛化到被动语态，而大型语言模型只有在新动词具有类似已知动词的意义时才能成功，表明计算模型的句法知识仍依赖于具体词汇的分布与意义，缺乏人类那样的抽象性。

    

    语言能产性，即词语在新语境中的使用，是人类语言最显著的特征之一。然而，支撑这种能产性的知识究竟有多抽象仍不清楚。我们在成功泛化依赖于基于形式的信息的情境中，比较了人类与计算语言模型的语言能产性。我们专注于将主动语态中学到的新动词的使用扩展到被动语态语境中。我们发现，即使无法为新动词赋予合理的意义，人们仍能以一种与关于语言形式的抽象泛化相一致的方式，将主动用法扩展到被动用法。相比之下，大型语言模型只有在新动词能够被赋予类似现有动词的意义时才能成功。我们得出结论：计算模型的句法知识仍然与特定词汇项的分布和意义紧密绑定，而人类的句法知识则不然。

    arXiv:2610.33433v1 Announce Type: new  Abstract: Linguistic productivity, as evidenced by the use of words in novel contexts, is one of the most striking facts about human language. However, how abstract the knowledge that enables such productivity is remains unclear. We compare people's and computational language models' linguistic productivity in contexts where successful generalization depends on form-based information. We focus on extending the use of a novel verb learned in the active voice to passive contexts. We find people extend active to passive uses in a way consistent with abstract generalizations concerning linguistic form, even when no plausible meaning can be assigned to the novel verb. In contrast, large language models succeeded only when a meaning like an existing verb's could be assigned to the novel verb. We conclude that computational models' syntactic knowledge remains tied to the distribution and meaning of specific lexical items in a way that human syntactic kno
    
[^253]: 面向LLM智能体一致性的递归语义散度

    Recursive Semantic Divergence for LLM Agent Consistency

    [https://arxiv.org/abs/2610.33012](https://arxiv.org/abs/2610.33012)

    该论文提出了递归语义散度（RSD），一种改编自强化学习双模拟度量的无监督轨迹级一致性指标，用于衡量LLM智能体在独立运行中的行为不一致性及其在轨迹中的传播。

    

    使用工具的大型语言模型（LLM）智能体在相同状态下的独立运行中会表现出不一致的行为，例如调用不同的工具、产生矛盾的输出或采取分歧的策略。这种变异性会随时间累积并削弱部署的可靠性。现有的评估指标主要通过基于完成度的分数来衡量任务正确性，而无法捕捉智能体行为在多次执行之间的一致性。此外，对智能体行为的分析要么在词元层面使用概率分布，要么在单次响应层面使用语义聚类，但两者都无法捕捉不一致性如何在轨迹的未来轮次中传播。我们提出了递归语义散度（RSD），这是一种改编自强化学习中双模拟度量的无监督轨迹级一致性指标。RSD衡量两个独立运行之间的预期累积语义分歧

    arXiv:2610.33012v1 Announce Type: cross  Abstract: Large Language Model (LLM) agents with tool use exhibit inconsistent behavior across independent runs from identical states, calling different tools, producing contradictory outputs, or pursuing divergent strategies. This variability compounds over time and undermines deployment reliability. Existing evaluation metrics primarily measure task correctness via completion-based scores and do not capture consistency in agent behavior across executions. Separately, agent behavior is analyzed either at the token level using probability distributions or at the single-response level using semantic clustering, but neither captures how inconsistency propagates across future turns of a trajectory. We introduce Recursive Semantic Divergence (RSD), an unsupervised trajectory-level consistency metric adapted from bisimulation metrics in reinforcement learning. RSD measures the expected cumulative semantic disagreement between two independent rollouts
    
[^254]: 循环内部：循环变换器在排列遍历上的机制研究

    Inside the Loop: A Mechanistic Study of Looped Transformers on Permutation Traversal

    [https://arxiv.org/abs/2610.24135](https://arxiv.org/abs/2610.24135)

    研究发现循环变换器解决有向排列遍历任务需要最小有效深度，超过该深度后权重绑定的程度成为决定性能的关键因素。

    

    解决具有大量顺序步骤的算法任务通常需要增加变换器的深度或思维链的长度，这分别会带来不利的参数成本和推理计算成本。循环变换器在多次迭代中重复使用同一个层块，有望解决这一问题，因为它可以在不增加参数或标记的情况下扩展有效深度。然而，目前尚不清楚每次迭代实际在计算什么，或者循环是否仅仅是在模拟一个更深的不绑定权重堆栈。在本文中，我们研究了有向排列遍历任务，其中模型从起始节点出发跟随H跳并报告最终到达的节点，因此每个中间跳都有已知的真实答案。我们发现，解决该任务需要一个最小有效深度，超过此深度后，权重绑定的程度成为决定性因素。例如，一个由四层块迭代八次构成的循环变换器在每个设（原文截断）

    arXiv:2610.24135v2 Announce Type: replace-cross  Abstract: Solving algorithmic tasks with many sequential steps typically requires scaling up transformer depth or chain-of-thought length, incurring unfavorable parameter and inference-compute costs, respectively. Looped transformers, which reuse one block of layers across many iterations, are poised to address this, since they scale effective depth without adding parameters or tokens. Yet, it remains unclear what each iteration is actually computing, or whether the loop merely simulates a deeper untied stack. Here, we study directed permutation traversal, in which a model follows H hops from a starting node and reports the node it reaches, so every intermediate hop has a known ground truth. We find that a minimum effective depth is necessary to solve the task, beyond which the degree of weight tying becomes the determining factor. For instance, a looped transformer with a four-layer block iterated eight times solves the task in every se
    
[^255]: 沃洛夫语适配的工业级开放权重语言模型

    Wolof-Adapted Industry-Ready Open-Weight Language Models

    [https://arxiv.org/abs/2610.20824](https://arxiv.org/abs/2610.20824)

    本文通过指令微调、偏好对齐和模型合并构建了0.6B-14B的沃洛夫语适配开放权重模型，在63项任务上表现出色，其中8B和14B模型可与更大的商业系统相竞争，且已公开发布。

    

    沃洛夫语有2300万使用者，主要分布在塞内加尔，但开放权重语言模型对该语言的支持仍然薄弱，而API模型在本地部署中会带来成本、治理和主权方面的限制。在本文中，我们通过指令微调、偏好对齐和模型合并技术，提出了适配沃洛夫语的0.6B-14B开放权重模型。在介绍完技术流程和数据后，我们在63个公开和内部任务上进行了大量实验，结果显示这些模型在沃洛夫语理解和生成方面表现良好，同时在其他语言和跨语言技能上也保持有效，其中8B和14B模型尤其具有竞争力，可与规模大得多的商业系统相媲美。这些模型目前正在内部客户服务试点中进行评估，并已在https://huggingface.co/collections/Orange/wolof-lm公开发布。

    arXiv:2610.20824v1 Announce Type: cross  Abstract: Wolof is spoken by 23M people, mostly in Senegal, but remains weakly supported by open-weight language models, while API models raise cost, governance, and sovereignty constraints for local deployments. In this paper, we present 0.6B-14B open-weight models adapted to Wolof through instruction tuning, preference alignment, and model merging. After presenting the pipeline and data, extensive experiments on 63~public and in-house tasks show good performance on Wolof understanding and generation while remaining effective on other languages and cross-lingual skills, with the 8B and 14B models being particularly competitive with substantially larger commercial systems. The models are currently being evaluated in internal customer-care pilots and are publicly available at https://huggingface.co/collections/Orange/wolof-lm .
    
[^256]: 二十天后：对Gemini回答中基于历史的品牌提及预测的预注册检验

    Twenty Days Later: A Pre-Registered Test of History-Based Forecasts of Brand Mentions in Gemini Answers

    [https://arxiv.org/abs/2610.19608](https://arxiv.org/abs/2610.19608)

    该论文通过预注册的重复实验证明，基于历史数据对Gemini回答中品牌提及率的预测在约20天后依然准确（Brier得分0.0281，接近8小时短期预测水平且优于品牌历史基线），表明大语言模型的品牌提及计数具有足够的时间稳定性，可供实际决策使用。

    

    arXiv:2610.19608v1 公告类型：交叉（cross）摘要：大型语言模型提及哪些品牌的计数，只有在其能维持足够长时间以供采取行动时才有用。我们通过一项预注册的重复实验来检验这一点。相同的162个请求（54个问题，涉及三个软件类别，分别以英语、爱沙尼亚语和芬兰语提出，每个问题包含12个候选品牌中的4个，重复三次）于2026年9月22日至23日在约十小时内分三波发送给gemini-3.1-pro-preview，并于2026年10月13日（即第三波之后20.3天）以完全相同的请求体再次发送。实验计划及其决策规则在首次调用之前被冻结、哈希并注册在OSF上。结果指标是每个答案与品牌的词汇提及标签，每波648个。主要预测——每个品牌-问题单元格在之前九个答案中的提及率——取得了Brier得分0.0281和对数损失0.126的成绩，相比之下，8小时波次的六答案预测Brier得分为0.0259，注册的上限为0.0559。它优于品牌历史基线（0.1……（原文摘要在此处截断）

    arXiv:2610.19608v1 Announce Type: cross  Abstract: Counts of which brands a large language model names are useful only if they hold long enough to act on. We test this with a pre-registered replication. The same 162 requests (54 questions in English, Estonian and Finnish about three software categories, four of twelve candidate brands per question, three repeats) were sent to gemini-3.1-pro-preview in three waves within about ten hours on 22 and 23 September 2026, and again on 13 October 2026, 20.3 days after the third wave, with identical request bodies. The plan and its decision rule were frozen, hashed and registered on OSF before the first call. The outcome is a lexical mention label per answer and brand, 648 per wave. The primary forecast, each brand-and-question cell's mention rate over the nine earlier answers, scored Brier 0.0281 and log loss 0.126, against Brier 0.0259 for the six-answer forecast of the 8-hour wave and a registered ceiling of 0.0559. It beat brand history (0.1
    
[^257]: SkillScreen：基于决策模型的恶意智能体技能高召回率筛查

    SkillScreen: High-Recall Screening of Malicious Agent Skills with Decision Models

    [https://arxiv.org/abs/2610.17727](https://arxiv.org/abs/2610.17727)

    SkillScreen 是一个紧凑的决策模型，能在单次前向传播中直接对 SKILL.md 技能的恶意程度评分，在未见攻击族上达到 99% 召回率且误报远低于词级分类器和 LLM 判官，但对陌生良性来源的误报仍会显著上升。

    

    智能体技能将自然语言指令转化为具体操作，这带来了一项安全挑战：合法的自动化行为与恶意行为可能看起来惊人地相似。一个实用的筛查系统必须在几乎捕获每一次攻击的同时，保持审查负载可控。我们提出了 SkillScreen，这是一个紧凑的决策模型，经过训练可以直接从 SKILL.md 文件在单次前向传播中对技能的恶意程度进行评分。在 MaliciousSkillBench 基准上，SkillScreen 在对未见攻击族达到 99% 召回率的情况下，仅将 20-25% 的良性技能标记为可疑，明显少于所测试的词级分类器和提示式 LLM 判官。在相同的训练协议下，它对未见来源的泛化能力也优于全量微调的编码器。然而，陌生的良性来源会显著增加误报率，这表明能够识别新攻击并不等于能够可靠地筛查新的技能集合。我们的结果证明了针对特定任务训练的决策模型在高召回率技能筛查中的价值。

    arXiv:2610.17727v2 Announce Type: cross  Abstract: Agent skills turn natural-language instructions into actions, creating a security challenge: legitimate automation and malicious behaviour can look deceptively similar. A practical screen must catch nearly every attack while keeping the review load manageable. We present SkillScreen, a compact decision model trained to score skill maliciousness directly from SKILL$.$md in a single forward pass. On MaliciousSkillBench, SkillScreen flags 20-25% of benign skills at 99% recall on unseen attack families, substantially fewer than the word-level classifier and prompted LLM judges tested. Under the same training protocol, it also generalizes better than a fully fine-tuned encoder to unseen sources. Yet unfamiliar benign sources sharply increase false positives, showing that recognizing new attacks does not ensure reliable screening of new skill collections. Our results demonstrate the value of task-trained decision models for high-recall skill
    
[^258]: 跳出机器人的思维：AI语义盲区中的人类想法并不逊色

    Thinking Outside the Bots: Human Ideas in AI's Semantic Blind Spots Are Not Inferior

    [https://arxiv.org/abs/2610.17324](https://arxiv.org/abs/2610.17324)

    本研究开发了EDGE方法来绘制AI的语义“盲区”，发现在大学文书、物品替代用途和人际建议三项开放性任务中，人类想法均延伸至AI输出稀少覆盖的语义区域，且这些独特的人类回应在创造力上并不逊色。

    

    AI生成的输出常被认为具有高度创造性，但与人类想法相比往往更加彼此相似（同质化）。超出AI范围的人类异质性是否反映了AI所遗漏的有价值思维？抑或是，更大的人类异质性源于质量较低的想法和AI所规避的低效思维？我们开发了“发散性生成表达的扩展”（EDGE）方法，通过评估个体人类想法在与任务匹配的AI语义比较集中的表征强弱程度，来绘制AI的语义“盲区”。具体而言，我们在三项开放性任务中比较了人类生成的想法与AI生成的想法：大学入学申请文书、日常物品的替代用途以及人际建议。在所有三项任务中，人类的回应都延伸到了AI输出稀疏覆盖的语义区域。这些独特的人类回应在创造力方面没有表现出明显的缺陷……

    arXiv:2610.17324v1 Announce Type: cross  Abstract: AI-generated outputs are frequently perceived as highly creative, but also tend to be more similar to each other (homogenized) than human ideas. Does the human heterogeneity that extends beyond AI reflect valuable thinking that AI misses? Alternatively, is greater human heterogeneity driven by lower-quality ideas and less effective thinking that AI avoids? We developed Expansion of Divergent Generative Expression (EDGE) to map AI's semantic "blind spots" by assessing how strongly/weakly individual human ideas were represented in a task-matched AI semantic comparison set. Specifically, we compared human-generated vs. AI-generated ideas across three open-ended tasks: college admissions essays, alternative uses for everyday objects, and interpersonal advice. Across all three tasks, human responses extended into semantic regions sparsely covered by AI outputs. These distinctively human responses showed no meaningful deficits in creativity,
    
[^259]: 不可靠叙述者框架作为解释澳大利亚与美国讽刺新闻的一种低成本机制

    Unreliable Narrator Framing as a Cost-effective Mechanism for Satire Explanation of Australian and American Satirical News

    [https://arxiv.org/abs/2610.12863](https://arxiv.org/abs/2610.12863)

    该研究提出以不可靠叙述者理论为框架，利用开放权重模型低成本地为澳大利亚和美国讽刺新闻生成解释，并贡献了新的数据集与评估基准。

    

    讽刺新闻是一种独特形式的错误信息：其原始发布来源本意是制造幽默，但可能被读者误解。对讽刺新闻文章进行解释具有挑战性，因为需要纳入最新的时事和本地背景信息。这是一项探索性研究，探讨为澳大利亚讽刺新闻文章生成解释，并将其与美国讽刺新闻进行对比。论文表明，在开放权重模型上基于不可靠叙述者理论构建解释框架，可以成为前沿模型的一种可行且低成本的替代方案。我们发现，模型对美国来源的讽刺套路的识别准确度高于澳大利亚来源。这项工作产出了一个新颖的数据集以及一个讽刺解释评估基准。

    arXiv:2610.12863v1 Announce Type: new  Abstract: Satirical news is a unique form of misinformation: it is intended to be humorous by the original publication source but can be misinterpreted. Explanation of satirical news articles is challenging because recent and local context needs to be incorporated. This is a pilot study that explores explanation generation for satirical news articles from Australia and compares them alongside those from USA. The paper demonstrates that framing explanations on unreliable narrator theory on open-weights models can be a feasible, cost-effective alternative to frontier models. We find that models identify the satirical tropes more accurately on the American source than the Australian one. This work results in a novel dataset and an evaluation benchmark for satirical explanation.
    
[^260]: 并非儿童所表达的内容：审计面向儿童AI中手语转文本的安全接口

    Not What a Child Expressed: Auditing the Sign-to-Text Safety Interface in Child-Facing AI

    [https://arxiv.org/abs/2610.07519](https://arxiv.org/abs/2610.07519)

    本文首次提出由聋人参与指导的部署前审计框架，用于审查“手语转文本”系统与儿童安全审核工具之间的接口，揭示SLT翻译错误可能在不影响文本流畅性的情况下悄然改变针对儿童手语使用者的安全决策。

    

    自动手语翻译（SLT）已进入消费级产品，可将美国手语转换为英文文本，用于听写、消息传递以及向对话助手提出的查询。面向儿童的AI与平台的信任与安全工具基于文本进行决策，例如针对未成年人账户的内容过滤器以及对聊天消息进行评分的诱骗（grooming）检测分类器。因此，使用手语翻译的儿童手语使用者需经由一次翻译才能触达这些安全机制。我们发现目前没有任何公开记录的系统曾对这两者进行过联合评估，而部署最广泛的主流SLT模型既未在18岁以下的的手语使用者上训练过，也未对其进行过正式评估。那些会改变否定语义、参与者角色、保密性、紧迫性或求助表达的翻译错误，可能在不影响文本流畅性的情况下悄然改变安全决策。本文提出一种由聋人社群参与指导的部署前审计方法，用以审查这一安全边界，包含失败类型分类法、净化后的场景模式、四种对照条件以及四项结果度量指标。Auslan是……（原文摘要至此截断）

    arXiv:2610.07519v1 Announce Type: new  Abstract: Automatic sign language translation (SLT) has entered consumer products, turning American Sign Language into English text for dictation, messaging, and queries put to a conversational assistant. Child-facing AI and platform trust-and-safety tooling decide on text, using filters on minor accounts and grooming classifiers that score chat messages. A signing child who uses SLT therefore reaches these safeguards through a translation. We found no publicly documented system in which the two have been jointly evaluated, and the leading deployed SLT model was neither trained nor formally evaluated on signers under 18. Errors that alter negation, participant roles, secrecy, urgency or help-seeking could change a safety decision without disturbing fluency. This paper proposes a Deaf-informed pre-deployment audit of that boundary, with a failure taxonomy, a sanitised scenario schema, four comparison conditions, and four outcome measures. Auslan is
    
[^261]: SearchJev：面向搜索代理的校准化System-1模型

    SearchJev: A Calibrated System-1 Model for Search Agents

    [https://arxiv.org/abs/2610.05107](https://arxiv.org/abs/2610.05107)

    SearchJev提出一个无需自回归生成的校准化System-1模型，将搜索代理的短决策与System-2推理分离，并通过软标签学习（SLCD）从不确定监督中提升决策置信度的可靠性。

    

    搜索代理需要反复就相关性、证据充分性和搜索动作做出短决策。使用生成式语言模型进行这些决策会引入延迟和不可靠的置信度。我们提出SearchJev，一个校准化的System-1模型，将搜索决策与System-2推理和生成相分离。给定搜索状态和决策模式，SearchJev无需自回归输出生成即可对合法选项进行评分。我们提出了用于校准决策的软标签学习，从不确定的监督中学习决策概率，并提高决策置信度的可靠性。在双系统搜索代理中，SearchJev处理短决策，并将不确定的判断委托给System 2，后者保留规划、查询生成和答案组合等能力。我们还引入了SearchDecision-Bench，一个统一六种搜索决策类型用于训练和评估的基准。在SearchDecision-Bench上……

    arXiv:2610.05107v2 Announce Type: replace-cross  Abstract: Search agents repeatedly make short decisions about relevance, evidence sufficiency, and search actions. Using generative language models for these decisions introduces latency and unreliable confidence. We present SearchJev, a calibrated System-1 model that separates search decisions from System-2 reasoning and generation. Given a search state and a decision schema, SearchJev scores legal options without autoregressive output generation. We propose Soft-Label Learning for Calibrated Decisions (SLCD) to learn decision probabilities from uncertain supervision and improve the reliability of decision confidence. In a dual-system search agent, SearchJev handles short decisions and delegates uncertain judgments to System 2, which retains planning, query generation, and answer composition. We also introduce SearchDecision-Bench, a benchmark unifying six types of search decisions for training and evaluation. On SearchDecision-Bench, S
    
[^262]: 零训练LLM+开放词汇检测（OVOD）流水线中的归因差距：CAAP-SNAP差异的细粒度分析

    Attribution Gaps in Zero-Training LLM+OVOD Pipelines: A Fine-Grained Analysis of the CAAP--SNAP Discrepancy

    [https://arxiv.org/abs/2609.32567](https://arxiv.org/abs/2609.32567)

    该研究在完整COCO-Val数据集上首次揭示了零训练LLM+OVOD流水线中CAAP与SNAP差异的根源在于词汇新颖性而非物体视觉复杂性——当LLM措辞超出检测器原生词汇表时定位准确率从80.9%骤降至31.6%，且这一下降主要源于新颖措辞实际命名了与标注不同的物体，而非同义词表述问题。

    

    LAOD及类似的零训练LLM+开放词汇检测器（OVOD）流水线分别对两项指标进行评分：类别无关定位准确率（CAAP）和语义命名准确率（SNAP）。这两项指标始终存在差异，却从未有人探究其原因。本文在完整的5,000张图像COCO-Val分割集（27,273个检测结果）上进行研究——而非原工作所评估的小子集——来探究这一原因。结果表明，物体的视觉复杂性并非驱动因素——小型和被遮挡物体的定位效果反而优于大型物体。真正的驱动因素是词汇新颖性：一旦LLM的措辞超出检测器原生类别集的范围，定位准确率就会从80.9%降至31.6%。不过，这种下降并非均匀分布在生疏措辞上。几乎所有的下降都来自于新颖措辞实际命名的物体与COCO标注不同的情形（真正的同义词仍能取得89.3%的分数；而语义无关的“噪声”标签仅得12.0%）。

    arXiv:2609.32567v2 Announce Type: replace-cross  Abstract: LAOD and similar zero-training LLM+open-vocabulary-detector (OVOD) pipelines score two things separately: class-agnostic localization accuracy (CAAP) and semantic naming accuracy (SNAP). The two consistently diverge, and nobody has asked why. This paper asks why, on the full 5,000-image COCO-Val split (27,273 detections) rather than the small subset the original work evaluated on. Object visual complexity turns out not to be the driver -- small and occluded objects are, if anything, localized better than large ones. Vocabulary novelty is: once the LLM's wording falls outside the detector's native category set, localization accuracy falls from 80.9% to 31.6%. That drop is not spread evenly across unfamiliar phrasing, though. Almost all of it comes from cases where the novel wording actually names a different object than the one COCO annotated (true synonyms still score 89.3%; semantically unrelated "noise" labels score 12.0%). A
    
[^263]: 基于梯度的数据归因方法中形式重于内容

    Form Over Content In Gradient-Based Data Attribution Methods

    [https://arxiv.org/abs/2609.19589](https://arxiv.org/abs/2609.19589)

    基于梯度的数据归因方法主要捕捉答案格式而非任务内容，因为共享答案格式的数据集表现出强梯度对齐，而任务相同但格式不同的数据集则不对齐。

    

    基于梯度相似性的数据归因方法被广泛用于分析和选择大语言模型的训练数据，但梯度相似性究竟衡量的是什么仍存在争议。一些研究将其解读为识别任务相关的技能，而另一些工作则报告表面形式是主要因素。我们通过独立变化任务和答案格式，为监督微调样本解决了这一争论。具体而言，我们以不同的答案格式呈现基准数据集，使得数据集可以共享任务但不共享格式，或共享格式但不共享任务。我们发现梯度对齐遵循答案格式：共享答案格式的基准数据对表现出强对齐（去衰减余弦相似度接近0.4），而相同基准以不同答案格式类别呈现时则不显示对齐（接近0.0）。我们证明这种排序从最早的预训练检查点一直延续到后训练阶段，并跨越不同的模型规模和模型家族成立。

    arXiv:2609.19589v1 Announce Type: cross  Abstract: Data attribution methods using gradient similarity are widely used to analyze and select training data for large language models, but what gradient similarity actually measures is debated. Some interpret it as identifying task-relevant skills, while other work reports that surface form is the main factor. We resolve this debate for supervised fine-tuning examples by varying task and answer format independently. Specifically, we render benchmarks in different answer formats, such that datasets can share a task without a format or a format without a task. We find that gradient alignment follows the answer format, as benchmark pairs sharing an answer format align strongly (disattenuated cosine near 0.4), while same benchmarks rendered with different answer format classes show no alignment (near 0.0). We demonstrate that this ordering holds from the earliest pretraining checkpoints through post-training, and across model scales and familie
    
[^264]: 报告实践至关重要：参考报告选择对胸部X光报告评估的影响

    Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation

    [https://arxiv.org/abs/2609.19093](https://arxiv.org/abs/2609.19093)

    该研究提出由放射科医生参与的报告实践变化分类法及ReRef参考报告重写方法，揭示参考报告写作风格的差异会显著影响胸部X光报告自动评估结果，甚至足以改变生成模型的排名。

    

    放射科医生遵循着异质性的报告实践。两位放射科医生在检查同一张图像并识别出相同临床发现的情况下，仍可能撰写出表面截然不同的报告，差异体现在术语、速记缩写、格式和详细程度上。这些报告规范的差异是评估基于AI的放射学报告生成（RRG）模型时一个未被充分重视的障碍，因为机器生成的报告通常是根据其与人工撰写的参考报告的一致性来评估的。在本文中，我们量化了现有评估指标对报告实践变化的敏感性，发现其影响之大足以改变模型的排名。我们提出了一个由放射科医生知识支持的放射学报告实践变化分类法，以及一种方法（ReRef），该方法沿着分类法的各个维度重写参考报告，同时保持临床解释不变。例如，当…

    arXiv:2609.19093v1 Announce Type: cross  Abstract: Radiologists follow heterogeneous reporting practices. Two radiologists examining the same image and identifying the same clinical findings might nevertheless compose superficially distinct reports, varying in terminology, shorthand, formatting, and level of detail. These variations in reporting norms represent an under-appreciated obstacle in efforts to evaluate AI-based radiology report generation (RRG) models, where machine-generated reports are typically assessed based on their concordance with human-generated references. In this paper, we quantify the sensitivity of established evaluation metrics to variations in reporting practices, revealing impacts large enough to alter the rankings of models. We introduce a radiologist-informed taxonomy of variations in radiology reporting practice and a method (ReRef) that rewrites reference reports along the axes of our taxonomy while preserving clinical interpretation. For instance, when co
    
[^265]: 回滚世界，保留反思：面向长程LLM智能体的回滚诱导反思

    Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2609.18304](https://arxiv.org/abs/2609.18304)

    提出了回滚诱导反思（RIR）统一恢复框架，在将LLM智能体回滚到选定先前状态的同时，保留从被放弃轨迹中提炼的可复用知识，解决了长程任务中错误累积且难以可靠恢复的问题。

    

    大语言模型（LLM）智能体越来越多地通过多步环境交互来处理长程任务，然而单个错误的动作可能会改变后续的状态和观测，导致错误随时间不断累积。现有方法要么在不修复已改变环境状态的情况下纠正上下文，要么在恢复早期状态的同时丢弃有用的经验，这使得既消除失败条件又避免重复过去的错误变得困难。我们认为，可靠的恢复应被视为一个回滚边界控制问题，即联合决定何时干预、从何处恢复，以及哪些信息应在恢复过程中保留。基于这一观点，我们提出了回滚诱导反思（RIR），这是一个统一的恢复框架，它将执行恢复到选定的先前状态，同时保留从被放弃轨迹中提炼出的可复用知识，以指导后续决策。我们进一步刻画……

    arXiv:2609.18304v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly tackle long-horizon tasks through multi-step environment interaction, yet a single erroneous action can alter subsequent states and observations, causing errors to compound over time. Existing methods either correct the context without repairing altered environment states or restore earlier states while discarding useful experience, making it difficult to both eliminate failure conditions and avoid repeating past mistakes. We argue that reliable recovery should instead be treated as a rollback-boundary control problem that jointly determines when to intervene, where to resume, and what information should survive recovery. Based on this view, we propose Rollback-Induced Reflection (RIR), a unified recovery framework that restores execution to a selected prior state while carrying forward reusable knowledge distilled from the abandoned trajectory to guide subsequent decisions. We further chara
    
[^266]: 太好而失真？诊断并缩小AI偏好与真实用户参与度之间的差距

    Too Good to Be Real? Diagnosing and Reducing the Gap Between AI Preference and Real User Engagement

    [https://arxiv.org/abs/2609.18282](https://arxiv.org/abs/2609.18282)

    该研究基于知乎、Quora和Reddit的117万条回答发现，大语言模型存在“逻辑过度绑定”倾向，即偏好增加逻辑结构，而真实用户参与度更依赖情感与表达显著性，并提出本体掩码推理自编码（OMRA）方法来缩小这一差距。

    

    大语言模型越来越多地被用于生成和评估在线内容，然而它们所认为的与更高参与度相关的内容质量，是否与真实用户的实际反应相匹配，仍不清楚。我们利用来自知乎、Quora和Reddit的25,978个问题下的117万个回答来研究这一问题，在四个题内参与度水平上比较真实平台回答与AI生成回答。我们提出了本体论偏好测量，从逻辑、情感和表达三个维度来表征回答。我们发现AI偏好与真实用户参与度之间存在系统性差距：随着目标参与度的提高，大语言模型会添加更多的显式逻辑结构，而真实用户的参与度则更强地与情感和表达上的显著性相关联。我们将这种倾向称为逻辑过度绑定。基于这一诊断，我们提出了本体掩码推理自编码，这是一种受控干预方法，通过掩码并重构……

    arXiv:2609.18282v1 Announce Type: new  Abstract: Large language models are increasingly used to generate and evaluate online content, yet it remains unclear whether the qualities they associate with higher engagement match what real users respond to. We study this question using 1.17 million answers to 25,978 questions from Zhihu, Quora, and Reddit, comparing real platform answers and AI-generated answers across four within-question engagement levels. We introduce Ontological Preference Measurement, which represents answers along three dimensions: logic, affect, and expression. We find a systematic gap between AI preference and real user engagement: as target engagement increases, LLMs add more explicit logical structure, while real user engagement is more strongly associated with affective and expressive salience. We call this tendency logic overbinding. Based on this diagnosis, we propose Ontology-Masked Reasoning Autoencoding (OMRA), a controlled intervention that masks and reconstr
    
[^267]: 大语言模型在上下文中发展出信念状态几何结构

    Large Language Models Develop Belief State Geometry In-Context

    [https://arxiv.org/abs/2609.17376](https://arxiv.org/abs/2609.17376)

    研究发现大语言模型在上下文学习隐马尔可夫模型数据时，会在其内部残差流激活中以线性可解码的方式编码出信念状态的几何表示，且通过干预实验证明该表示与模型的预测行为功能相关。

    

    经过下一词元预测训练的大语言模型（LLM）展现出卓越的上下文学习能力（ICL），然而支持这种学习的内部表示仍鲜为人知。我们在一个受控环境中研究这些表示：使用隐马尔可夫模型（HMM）生成的数据来提示大语言模型，并探测相应的信念状态——即在给定已观测词元历史的条件下，HMM隐状态的后验分布。通过对六个开源大语言模型使用40个被挑选出具有非平凡信念结构的HMM数据进行提示，我们发现信念状态可以从残差流激活中线性解码，在各种HMM与大语言模型的组合上，探测器的峰值R²值达到0.83-0.99，且涵盖了从早期到后期的各层。为确立该表示的功能相关性，我们通过补丁修复（patching）和导向（steering）对探针识别出的子空间进行直接干预，结果使下游预测质量达到与未干预模型相当的水平，同时……

    arXiv:2609.17376v1 Announce Type: cross  Abstract: Large language models (LLMs) trained on next-token prediction exhibit remarkable in-context learning (ICL) abilities, yet the representations that support ICL remain poorly understood. We consider such representations in a controlled setting: prompting LLMs with data emitted from hidden Markov models (HMMs) and probing for the corresponding belief state -- the posterior distribution over the HMM's hidden states given the observed token history. Across six open-source LLMs prompted with data from 40 HMMs selected for non-trivial belief structure, we find that belief states are linearly decodable from residual stream activations, with peak probe $R^2$-values from 0.83-0.99 across HMM and LLM combinations, ranging from early to late layers. To establish functional relevance, we intervene directly on the probe-identified subspace via patching and steering, resulting in downstream prediction quality on the order of the untampered model, whi
    
[^268]: MindTopo：基础模型能在拓扑空间中推理吗？

    MindTopo: Can Foundation Models Reason in Topological Space?

    [https://arxiv.org/abs/2609.11900](https://arxiv.org/abs/2609.11900)

    该论文提出了MindTopo基准，基于认知科学与形式拓扑学，从连续性、分离性、有序性、包含性和纽结五个拓扑属性出发，在推理与规划两个认知层面对14个多模态大语言模型的拓扑空间能力进行系统评估。

    

    空间推理不仅依赖于距离、角度和形状等度量属性，还依赖于在连续变形下保持不变的拓扑关系。认知科学认为这些拓扑关系是空间理解的基础，然而目前针对基础模型的评估大多集中于度量关系或依赖视角的关系。我们提出了MindTopo，一个基于认知科学和形式拓扑学、涵盖五个拓扑属性的拓扑直觉基准：连续性、分离性、有序性、包含性和纽结。MindTopo在两个认知层面对每个属性进行评估：推理层面要求模型识别拓扑关系或推断其如何变化；规划层面则将基础模型实例化为闭环智能体，其策略负责选择环境动作。MindTopo包含11,030个实例，覆盖13种难度可控的程序化生成任务类型。我们对14个多模态大语言模型（MLLM）进行了基准测试，并研究了智能体的不同配置。

    arXiv:2609.11900v1 Announce Type: cross  Abstract: Spatial reasoning depends not only on metric properties such as distance, angle, and shape, but also on topological relations that remain invariant under continuous deformation. Cognitive science identifies these relations as foundational to spatial understanding, yet foundation-model evaluations largely focus on metric or viewpoint-dependent relations. We introduce MindTopo, a benchmark of topological intuition across five properties grounded in cognitive science and formal topology: continuity, separation, order, enclosure, and knots. MindTopo evaluates each property at two cognitive levels. Reasoning asks a model to identify topological relations or infer how they change. Planning instantiates a foundation model as a closed-loop agent whose policy selects environment actions. MindTopo contains 11,030 instances across 13 procedurally generated task types with controllable difficulty. We benchmark 14 MLLMs and study agent configuratio
    
[^269]: Vishing-Tactics-Bench：语音钓鱼通话中利用轨迹的预测

    Vishing-Tactics-Bench: Forecasting Exploitation Trajectories in Voice Phishing Calls

    [https://arxiv.org/abs/2609.07151](https://arxiv.org/abs/2609.07151)

    提出基于 Endsley 态势感知框架的语音钓鱼防御基准 Vishing-Tactics-Bench，将防御范式从事后欺诈分类转变为对正在进行的通话是否将到达信息收集或金融剥削等终端危害的实时预测，并构建了 6 战术分类体系和 35,340 条标注话语的数据集。

    

    语音钓鱼以实时方式展开；当通话结束、事后分类成为可能时，危害已经造成。更具可操作性的问题是：正在进行的通话在战术上正朝着哪种具体危害（信息收集或金融剥削）发展。我们提出了 Vishing-Tactics-Bench，这是一个基于 Endsley 态势感知框架的基准，将语音钓鱼防御从事后欺诈分类重新构建为危害预测：在每一轮对话中预测通话是否将到达两种终端危害之一。我们将 MITRE ATT&CK 框架适配到语音钓鱼场景，形成包含 6 种战术的分类体系，并对 5,645 通合成中文通话中的 35,340 条诈骗者话语进行了标注。我们定义了利用轨迹预测任务，这是一种针对两种终端危害的生存分析式评估协议，包含三个指标：AP@k、C-index 和散度误差。从马尔可夫启发式方法到微调大语言模型的各种基线实验表明……

    arXiv:2609.07151v2 Announce Type: replace  Abstract: Voice phishing (vishing) unfolds in real time; by the time a call has ended and post-hoc classification is possible, the harm has already been done. The more actionable question is which concrete harm (Information Gathering or Financial Exploitation) an ongoing call is tactically progressing toward. We present Vishing-Tactics-Bench, a benchmark grounded in Endsley's situation-awareness (SA) framework that recasts vishing defense from after-the-fact fraud classification to harm projection: predicting at each turn whether the call will reach either terminal harm. We adapt MITRE ATT&CK to vishing as a 6-tactic taxonomy (Vishing-Tactics) and label 35,340 scammer utterances across 5,645 synthetic Chinese calls. We define Exploitation Trajectory Forecasting, a survival-style protocol over the two terminal harms with three metrics: AP@k, C-index, and divergence error. Baselines ranging from a Markov heuristic to fine-tuned LLMs show that th
    
[^270]: ViSAR：面向视觉文档问答的无需训练的自适应k值检索方法

    ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering

    [https://arxiv.org/abs/2609.02486](https://arxiv.org/abs/2609.02486)

    提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。

    

    文档视觉问答通常利用检索增强生成技术，其中晚期交互编码器常被用于识别与用户查询相关的文档页面，然后由大型视觉-语言模型生成答案。现有方法通常无论查询复杂度如何都检索固定数量的前k个页面，这会增加大型视觉-语言模型的延迟，并可能降低答案的准确性。我们提出了ViSAR（视觉语义激活检索），这是一种面向晚期交互视觉文档检索的无需训练的自适应k值检索方法。ViSAR直接在嵌入空间中运行，构建以查询为条件的页面级相似度矩阵，突出与查询相关的语义，并动态确定需要检索的页面数量。在多个编码器和大型视觉-语言模型上的实验表明，ViSAR能够检索紧凑且适应查询的页面集合，将RAG延迟降低高达58.7%，同时保持或提升答案准确性。

    arXiv:2609.02486v1 Announce Type: cross  Abstract: Document Visual Question Answering (DocVQA) often leverages Retrieval-Augmented Generation (RAG), where late-interaction encoders are commonly used to identify document pages relevant to a user query, before answer generation by a Large Vision-Language Model (LVLM). Existing approaches typically retrieve a fixed top-$k$ number of pages regardless of query complexity, which increases LVLM latency and may degrade answer accuracy. We introduce ViSAR (Visual Semantic Activation Retrieval), a training-free adaptive-$k$ retrieval method for late-interaction visual document retrieval. ViSAR operates directly in the embedding space to construct a query-conditioned page-level similarity matrix that highlights query-relevant semantics and dynamically determines the number of pages to retrieve. Across multiple encoders and LVLMs, ViSAR retrieves compact, query-adapted page sets that reduce RAG latency by up to 58.7\%, while maintaining or improvi
    
[^271]: 大语言模型中量化损伤的结构：为什么下一个比特应该被全局分配

    The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally

    [https://arxiv.org/abs/2609.01587](https://arxiv.org/abs/2609.01587)

    该研究通过因果混合精度干预实验发现，LLM的量化损伤是弥散分布的而非集中于特定电路、计算位置或权重统计，因此在匹配精度预算下，将额外比特全局用于更精细的量化粒度比局部修复少数层更有效。

    

    训练后量化（PTQ）被广泛用于降低大语言模型（LLM）的服务成本，但其精度损失并不均匀，且通常需要针对每个模型单独调优。我们研究了量化损伤发生在何处，以及如何分配少量额外的精度预算。以因果混合精度干预作为基准真值（依次将每一层提升至8比特并测量其所能恢复的精度），我们在4个架构家族的9个开源权重模型上测试了3个直观假设：量化损伤存在于任务电路中、存在于模型计算发生之处，或存在于权重统计特性之中。然而，这些假设都无法预测哪些层会从恢复的精度中受益。相反，恢复是弥散性的：在9个模型中有8个，恢复75%的精度差距大约需要一半的层；唯一的例外是Qwen3-8B，其恢复高度集中。在匹配的精度预算下，将预算全局用于更精细的量化粒度，优于局部修复最具可恢复性的层。

    arXiv:2609.01587v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) is widely used to reduce the cost of serving large language models (LLMs), but its accuracy cost is uneven and is often tuned per model. We study where quantization damage occurs and how to allocate a small additional precision budget. Using causal mixed-precision intervention as ground truth (raise each layer to 8-bit in turn and measure the accuracy it recovers) across 9 open-weight models in 4 architecture families, we test 3 intuitive hypotheses: that quantization damage lives in task circuits, where the model computes, or in weight statistics. None of them predicts which layers benefit from restored precision. Recovery is instead diffuse: for 8 of 9 models, recovering 75% of the gap takes roughly half the layers; the lone exception, Qwen3-8B, is sharply concentrated. At a matched precision budget, spending it globally on finer quantization granularity beats locally repairing the most recoverable la
    
[^272]: Hi-Q：面向多跳问答的分层证据引导查询细化

    Hi-Q: Hierarchical Evidence-guided Query Refinement for Multi-Hop Question Answering

    [https://arxiv.org/abs/2608.30468](https://arxiv.org/abs/2608.30468)

    Hi-Q提出了一种以证据为条件的分层查询细化框架，通过在每个查询节点上利用解析算子判断证据是否支持当前查询单元，动态构建查询树，从而解决多跳问答中问题表达粒度与语料证据可检索粒度不匹配的核心瓶颈。

    

    多跳问答（QA）的一个核心瓶颈在于，问题所表达的粒度往往与语料库中证据可被检索的粒度不一致。现有方法通过在语料库上强加固定图结构、通过迭代重新表述查询、或通过执行生成的程序来解决这种不匹配，但这些策略并未显式地判断一个查询单元何时已被证据支持、何时应当被进一步细化。我们将这一瓶颈形式化为可检索粒度发现问题，并提出了Hi-Q——一个以证据为条件的分层查询细化框架。在每个查询节点上，解析算子检验检索到的证据是否支持当前查询单元；已解决的节点即终止，而未解决的节点则由一个保持依赖关系的二元算子进行扩展，并由语义覆盖验证器进行检查。因此，Hi-Q 构建出一棵查询树，其顶层……

    arXiv:2608.30468v1 Announce Type: new  Abstract: A central bottleneck in multi-hop Question Answering (QA) is that the granularity at which a question is expressed often differs from the granularity at which corpus evidence is retrievable. Existing methods address this mismatch by imposing fixed graph structures over the corpus, by iteratively reformulating the query, or by executing a generated program over it, but these strategies do not explicitly decide when a query unit is already supported by evidence and when it should be refined. We formulate this bottleneck as retrievable granularity discovery and introduce Hi-Q, an evidence-conditioned framework for hierarchical query refinement. At each query node, a resolution operator tests whether retrieved evidence supports the current query unit; resolved nodes terminate, while unresolved nodes are expanded by a dependency-preserving binary operator and checked by a semantic coverage verifier. Hi-Q therefore grows a query tree whose top
    
[^273]: 将大语言模型分层防御视为集成：访问层级、推理成本与防御层间失败相关性的实测

    Layered LLM Defenses as an Ensemble: Access Tiers, Inference Cost, and the Measured Failure Correlation Between Defense Layers

    [https://arxiv.org/abs/2608.28327](https://arxiv.org/abs/2608.28327)

    本文提出对抗者访问层级模型（AATM）和推理成本分类方法，首次实测了大语言模型多层防御之间的失败相关性，证明防御堆栈只有在各层在不同输入上失败时才能真正产生叠加防护效果。

    

    从业者通过堆叠多层防御来保护大语言模型（LLM），并默认各层防御会相互叠加增强。防御堆栈本质上是一个集成，而集成只有在满足一个条件下才会产生叠加效果——这一条件在LLM安全文献中虽被推荐但从未被实测：各成员必须在不同的输入上失败。本文提出两种工具使这一点变得可度量。对抗者访问层级模型（AATM）根据攻击者所掌握的访问权限对其进行分级，范围从仅系统访问（A0）到可影响训练数据（A4）。成本模型将防御按五类推理时开销进行分类；由于其中两类需要训练权重或读取激活值，它们像AATM对攻击者分级那样对防御者进行分级。由此我们推导出堆栈的行为规律，以及防御者所关心的各项指标如何分化：覆盖率在层级内趋于饱和，成本随类别上升，误拒绝以并集方式累积，而残余攻击成功率只有在独立假设成立时才呈乘性下降。我们对这种独立性进行了实测。

    arXiv:2608.28327v1 Announce Type: cross  Abstract: Practitioners defend large language models (LLMs) by stacking defenses, assuming the layers compound. A stack is an ensemble, and ensembles compound only under a condition the LLM security literature recommends but never measures: the members must fail on different inputs.   Two instruments make that measurable. The Adversary Access-Tier Model (AATM) grades an adversary by the access it holds, from system-only (A0) to influence over training data (A4). A cost model sorts defenses into five classes of inference-time overhead; because two classes require training weights or reading activations, they tier the defender as AATM tiers the adversary. From these we derive how a stack behaves, and the quantities a defender cares about diverge: coverage saturates within a tier, cost rises by class, false refusals accumulate as a union, and residual attack success falls multiplicatively only under independence.   We measure that independence. Run
    
[^274]: IndicQE-APE：面向印度语言的翻译质量评估与自动后编辑基准

    IndicQE-APE: A Benchmark for Quality Estimation and Automatic Post-Editing for Indic Languages

    [https://arxiv.org/abs/2608.16344](https://arxiv.org/abs/2608.16344)

    本文整合了印度语言的质量评估和自动后编辑数据，创建了一个包含多标签和难度分层的基准，并评估了多种模型，发现只有同时利用整体和词级信息的系统在严格对照下表现显著。

    

    摘要：arXiv:2608.16344v1 公告类型：新 摘要：印度语言的质量评估（QE）和自动后编辑（APE）数据分散在不同的发布中，因此没有单一资源能够在同一基础上支持跨任务和语言对的训练与评估。我们将WMT 2020--2024共享任务系列与扩展的英语--马拉雅拉姆语资源整合为\indicqe：包含126,754个实例，覆盖九个方向性语言对，每个片段上最多有四种标签类型对齐，包括直接评估、人工后编辑、词级OK/BAD标签和错误解释，以及按四个难度轴分层的测试集。在此基准上，我们评估了六个提示的大型语言模型和三个COMET指标在片段级QE上的表现，以及三个系统在APE上的表现。其中两个轴部分基于直接评估并选择其压缩子集，因此每个轴与从相同语言对和相同分数分布中抽取的对照组进行比较。只有一种方法在对照组中幸存：整体和词级对齐的片段。

    arXiv:2608.16344v1 Announce Type: new  Abstract: Indic quality estimation (QE) and automatic post-editing (APE) data is spread across separate releases, so no single resource supports training and evaluation across tasks and language pairs on one footing. We consolidate the WMT 2020--2024 shared-task lineage with an extended English--Malayalam resource into \indicqe: $126{,}754$ instances over nine directional pairs, with up to four label types aligned on the same segment, a direct assessment, a human post-edit, word-level OK/BAD tags and an error explanation, and a test set stratified over four difficulty axes. On it, we benchmark six prompted LLMs and three COMET metrics on segment-level QE, and three systems on APE. Two of the axes are defined partly on the direct assessment and select a compressed slice of it, so each axis is compared against a control drawn from the same language pair with the same score distribution. Only one survives that control: segments whose holistic and tok
    
[^275]: 大语言模型中的提示特权：衡量与缓解公平AI访问的障碍

    Prompt Privilege in Large Language Models: Measuring and Mitigating Barriers to Equitable AI Access

    [https://arxiv.org/abs/2608.08942](https://arxiv.org/abs/2608.08942)

    该论文首次提出“提示特权”这一公平性问题，即具备提示工程专业知识的用户能因措辞技巧而获得更好的模型输出，并设计了提示公平分数（PES）评估指标与提示公平转换器（PET）智能体来衡量和缓解这种AI访问不平等。

    

    大语言模型正日益被应用于医疗保健、教育、公共服务和日常决策中。无论用户的读写能力、沟通风格或提示工程专业知识水平如何，它们都应该提供可比的协助。然而，当前关于提示鲁棒性的研究主要集中于对抗性攻击、提示注入和提示优化，却忽略了语义等价的请求仅仅因为措辞不同而收到不同回应的问题。我们将这一可访问性挑战称为“提示特权”：拥有更多提示专业知识的用户尽管表达了相同的潜在意图，却能获得更好的模型性能。为解决这一问题，我们提出了提示公平分数，这是一个用于评估跨用户群体性能一致性的指标，以及提示公平转换器，这是一个基于大语言模型的智能体，能够自动将用户请求转换为语义等价的形式。

    arXiv:2608.08942v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used in healthcare, education, public services, and everyday decision making. They should provide comparable assistance regardless of users' literacy, communication style, or prompt-engineering expertise. However, research on prompt robustness primarily focuses on adversarial attacks, prompt injection, and prompt optimization, overlooking whether semantically equivalent requests receive different responses simply because they are phrased differently. We call this accessibility challenge "Prompt Privilege": users with greater prompting expertise obtain better model performance despite expressing the same underlying intent. To address this problem, we propose the Prompt Equity Score (PES), a metric for evaluating performance consistency across user populations, and the Prompt Equity Transformer (PET), an LLM-based agent that automatically transforms user requests into semantically equivalen
    
[^276]: LangChoiceBench：测量与解释大语言模型中的编程语言选择

    LangChoiceBench: Measuring and Explaining Programming-Language Choice in LLMs

    [https://arxiv.org/abs/2608.06041](https://arxiv.org/abs/2608.06041)

    提出了LangChoiceBench基准，系统测量25个大语言模型在28个项目中的编程语言选择行为，发现模型普遍过度偏好Python且多为未经深思熟虑的自动选择，较小的开放权重模型偏好更强、语言多样性更低。

    

    大语言模型（LLM）在生成项目级代码时已被证明对Python表现出强烈的偏好，但目前尚无系统性的方法来在新模型中测量这种行为。为弥合这一差距，我们提出了LangChoiceBench，这是一个项目级代码生成基准，用于测量Python偏好、推荐与实现的一致性以及语言多样性。LangChoiceBench涵盖七个软件领域中的28个项目，这些领域中Python往往不是一个好的默认选择。我们评估了25个多样化的LLM，发现Python仍被严重过度选用，推荐与实现之间的一致性较低，且较小的开放权重模型通常表现出更强的Python偏好和更低的语言多样性。我们进一步分析了9,826条推理轨迹，发现大多数Python选择是自动做出的或主要由易用性驱动，而非对项目需求的明确考量。在一组较小但重要的……（摘要原文在此处截断）

    arXiv:2608.06041v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have been shown to exhibit strong Python preferences when generating project-level code, but there is currently no systematic way to measure this behaviour across new models. To bridge this gap, we introduce LangChoiceBench, a project-level code-generation benchmark for measuring Python preference, recommendation-implementation consistency, and language diversity. LangChoiceBench covers 28 projects across seven software areas where Python is often a poor default. We evaluate 25 diverse LLMs and find that Python remains heavily over-selected, recommendation-implementation consistency is low, and smaller open-weight models generally show stronger Python preference and lower language diversity. We further analyse 9,826 reasoning traces and find that most Python choices are automatic or driven primarily by ease, rather than explicit consideration of project requirements. In a smaller but important set o
    
[^277]: 一个虚假的平均值：思维链监控器在其作为唯一防线之处崩溃

    A False Average: Chain-of-Thought Monitors Collapse Where They Are the Only Defense

    [https://arxiv.org/abs/2608.00583](https://arxiv.org/abs/2608.00583)

    仅重写智能体的思维链推理（保持动作完全不变）即可在一次无梯度攻击中将CoT监控器的捕获率从约95%暴跌至11%以下，揭示监控器的总体准确率是掩盖其在唯一防线场景下近乎全面失效的虚假平均值。

    

    思维链监控旨在捕捉那些在动作层面看起来干净、只在推理中暴露自身的奖励作弊行为。我们证明，这恰恰是控制推理的攻击者能够击败它的地方。仅重写智能体的推理内容使其看起来像是善意的工程工作，同时逐字复制每个命令和输出以保持漏洞利用不变，在一次无需梯度的攻击中，就能将留出监控器在该子集上的捕获率从约95%降至11%以下。监控器的总体准确率是一个虚假的平均值：它由动作层面就会暴露的作弊行为主导，从而掩盖了这种重写在CoT监控是唯一信号的子集上造成的近乎全面的崩溃。该攻击可跨监控器系列和智能体模型迁移，并在真实智能体上复现，不过在面对经过校准的监控器时，规避行为主要集中在最强的智能体上。仅基于轨迹的防御只能部分恢复捕获率，即使是预先针对该攻击进行训练的防御也是如此。

    arXiv:2608.00583v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) monitoring is meant to catch the reward hacks that look clean in the actions and betray themselves only in the reasoning. We show that this is exactly where an adversary who controls the reasoning can defeat it. Rewriting only an agent's reasoning to read as good-faith engineering, while copying every command and output verbatim so the exploit is unchanged, drops a held-out monitor's catch rate on that subset from about 95% to under 11% in one gradient-free shot. A monitor's aggregate accuracy is a false average: dominated by hacks the actions give away, it hides the near-total collapse this rewrite produces on the subset where CoT monitoring is the only signal. The attack transfers across monitor families and agent models, reproduces with live agents, though against a calibrated monitor evasion concentrates in the strongest agent. Trace-only defenses recover it only partially, even one primed on the atta
    
[^278]: RubricReviewer：通过智能体证据合成与人类对齐生成，从直接评判走向评分标准驱动的评审

    RubricReviewer: From Direct Critique to Rubric-Driven Review via Agentic Evidence Synthesis and Human-Aligned Generation

    [https://arxiv.org/abs/2608.00005](https://arxiv.org/abs/2608.00005)

    该论文提出RubricReviewer，一个完全由评分标准驱动的同行评审框架，通过将评分标准生成作为显式中间步骤，并结合免训练智能体的证据合成与基于训练模型的人类对齐生成，解决了现有LLM评审器评判标准隐式化、两类范式割裂的结构性局限。

    

    主要学术会议的同行评审正面临空前的投稿压力，这促使人们将大语言模型（LLM）用作评审助手。然而，现有基于LLM的评审器面临两个结构性局限。其一，它们将稿件直接映射为评审意见，使潜在的评分标准保持隐式，并将其推导与评判过程纠缠在一起。其二，主流范式各自只捕捉了好评审的一半：免训练智能体能够收集广泛的证据，但产生的是缺乏方向的批评；而基于训练的评审器在继承人类判别性判断的同时，也继承了其噪声与覆盖不均的问题。我们提出RubricReviewer，一个完全由评分标准驱动的框架，同时解决上述两个局限。它将评分标准生成为显式的中间步骤，使评审生成与最终评估都以论文自适应的评分标准为条件。它进一步结合了一个免训练智能体用于收集外部……（原文摘要在此处截断）

    arXiv:2608.00005v2 Announce Type: replace-cross  Abstract: Peer review at major venues is under unprecedented submission pressure, motivating the use of large language models (LLMs) as review assistants. Existing LLM-based reviewers, however, face two structural limitations. First, they map manuscripts directly to reviews, leaving the underlying rubric implicit and entangling its derivation with the judgement. Second, the prevailing paradigms each capture only half of a good review: training-free agents gather broad evidence but produce undirected critiques, while training-based reviewers inherit human discriminative judgement together with its noise and uneven coverage. We introduce RubricReviewer, a fully rubric-driven framework that addresses both limitations. It makes rubric generation an explicit intermediate step, so that both review generation and the final assessment are conditioned on paper-adaptive rubrics. It further combines a training-free agent (Scout) that gathers extern
    
[^279]: Change2Task：从仓库变更到可执行的编码智能体任务与环境

    Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments

    [https://arxiv.org/abs/2607.28591](https://arxiv.org/abs/2607.28591)

    Change2Task 提出了一个基于仓库历史的系统，能够将已合并的拉取请求自动转换为健康现代代码版本上经过验证的可执行编码智能体任务与环境，为编码智能体的训练、基准测试和持续评估提供可扩展的数据供给。

    

    扩展编码智能体需要持续供给可执行的数据，用于训练、基准测试和持续评估。每个任务必须将真实的软件状态与任务规范、开发工具和可靠的验证相结合。为了扩大这一供给，我们提出了 Change2Task，这是一个基于仓库历史的系统，它将已合并的拉取请求（PR）转换为同一仓库健康现代版本上经过验证的任务。它将历史证据与演进的代码进行对齐，通过补丁反转、代码映射或智能体重建三种方式重建任务状态，并验证从健康基础状态到任务状态再到恢复状态的完整生命周期。通过从维护良好的环境中推导出多个基于开发者证据的任务，Change2Task 为编码智能体的训练和评估提供了可执行数据，同时减少了重复的环境搭建、存储和任务构建成本。我们评估了五个编码智能体……（原文截断）

    arXiv:2607.28591v2 Announce Type: replace-cross  Abstract: Scaling coding agents requires a continuing supply of executable data for training, benchmarking, and continuous evaluation. Each task must couple a realistic software state with a specification, development tools, and reliable verification. To expand this supply, we present Change2Task, a system grounded in repository history that converts merged pull requests (PRs) into verified tasks on healthy modern revisions of the same repository. It aligns historical evidence with evolved code, reconstructs task states through Patch Reversal, Code Mapping, or Agent Reconstruction, and validates the lifecycle from a healthy base to a task state and a restored state. By deriving multiple tasks grounded in developer evidence from maintained environments, Change2Task provides executable data for coding agent training and evaluation while reducing repeated environment setup, storage, and task construction effort. We evaluate five coding agen
    
[^280]: 刻画AI生成诗歌的人类相似性：一项零样本分类研究

    Characterizing Human-Likeness in AI Generated Poetry: A Zero-shot Classification Study

    [https://arxiv.org/abs/2607.26221](https://arxiv.org/abs/2607.26221)

    该研究通过零样本分类方法刻画英语诗歌中影响人类诗歌与AI生成诗歌正确或错误分类的属性特征，为“GenAI诗歌最难被现代检测工具识别”这一论断提供了客观验证依据。

    

    随着AI技术的进步，生成式AI（GenAI）与人类撰写的文本已经变得几乎无法区分。此外，AI聊天机器人的全球普及使得学术不端行为更加频繁。而且，现有研究表明，即使不做任何修改，GenAI生成的诗歌也是最难以与人类作品区分的，因此GenAI诗歌自然会被现代检测工具判定为具有人类风格。然而，此类研究的客观性需要对照现代检测工具加以验证，但诗歌的主观性以及现代大语言模型（LLM）架构的黑箱特性使得对这类工作的验证相当复杂。因此，本研究的主要目标是推导出英语诗歌中哪些属性有助于（或妨碍）对人类诗歌和AI诗歌的正确分类与错误分类，并为“GenAI诗歌难以被区分”这一论断提供佐证或反驳的证据。

    arXiv:2607.26221v2 Announce Type: replace  Abstract: With the advancement of AI technologies, Generative AI (GenAI) and human written text have become nearly indistinguishable. Additionally, the global standardization of AI chatbots made academic malpractice more frequent. Furthermore, existing research indicates GenAI poems are the most difficult to distinguish even without any modification thus, GenAI poems are naturally deemed human-like by modern detectors. However, the objectivity of such dissertations needs to be verified against modern detection tools but the subjectivity of poetry and the black-box nature of the modern LLMs (Large Language Models) architectures made verification of such work quite complicated. Hence, the main objective of the research is to deduce the attributes of English poetry that contribute classification and misclassification of both human and AI poems and provide corroborating or contradicting evidence to the poetry distinguishability claim. For such cha
    
[^281]: 传递接力棒：轨迹接力式在线策略蒸馏

    Pass the Baton: Trajectory-Relayed On-Policy Distillation

    [https://arxiv.org/abs/2607.26057](https://arxiv.org/abs/2607.26057)

    针对在线策略蒸馏中学生一旦走上错误推理方向便难以纠正的“前缀失败”问题，本文提出 Relay-OPD，利用师生延续不对称性作为无标签触发信号，让教师在关键位置短暂接管生成后再交还学生，以有限的干预预算提升蒸馏监督的可靠性。

    

    在线策略蒸馏（OPD）将 token 级监督建立在学生模型自身的轨迹上，但存在“前缀失败”问题：一旦学生模型踏上错误的推理方向，后续所有生成都将建立在这一偏差之上，产生方向错误的延续内容，从而引发不可靠的监督并浪费算力。我们在失败前缀上识别出一种师生延续不对称性——教师模型倾向于重新定向，而学生模型则沿原方向继续——并将这一现象转化为 Relay-OPD（接力式在线策略蒸馏）中无需标签的交接触发器。在训练过程中，Relay-OPD 通过让教师模型在检测到的触发点短暂接管、生成一段“教师接力段”来构建接力轨迹，随后学生模型重新接手，并在所得到的轨迹上进行优化。有限的接力预算将干预集中在关键的早期位置，同时限制了对学生策略的偏离。

    arXiv:2607.26057v2 Announce Type: replace-cross  Abstract: On-policy distillation (OPD) grounds token-level supervision in the student's own trajectory, yet suffers from prefix failure: once the student commits to a wrong reasoning direction, all subsequent generation builds on this deviation, producing misdirected continuations that elicit unreliable supervision and waste compute. We identify a teacher-student continuation asymmetry on failed prefixes, where the teacher tends to redirect while the student continues along the original direction, and convert it into a label-free handoff trigger in Relay On-Policy Distillation (Relay-OPD). During training, Relay-OPD constructs relay trajectories by letting the teacher briefly take over at detected trigger points to produce a teacher leg, after which the student resumes and is optimized on the resulting trajectory. A limited relay budget concentrates intervention on critical early positions while limiting departure from the student policy
    
[^282]: 自述患有多动症（ADHD）或自闭症谱系障碍（ASD）的Twitter用户中DSM-5抑郁症状语言的群体水平画像：一项使用自然语言处理与统计分析的探索性研究

    Population-Level Profiling of DSM-5 Depressive Symptom Language Among Twitter Users With Self-Reported ADHD or ASD: Exploratory Study Using Natural Language Processing and Statistical Analysis

    [https://arxiv.org/abs/2607.05626](https://arxiv.org/abs/2607.05626)

    该研究构建了零样本分类器结合本地大语言模型（Qwen3-8B）与经ReDSM5微调的MentalRoBERTa的自然语言处理流程，对自述ADHD或ASD的792名Twitter用户的百万余条推文进行DSM-5九项抑郁症状的自动分类与统计建模，从而在群体水平揭示了两组神经发育障碍人群抑郁症状语言表达的差异。

    

    arXiv:2607.05626v2 公告类型：替换 摘要：抑郁症常与注意缺陷/多动障碍（ADHD）和自闭症谱系障碍（ASD）共病，但患有这些疾病的人群如何表达抑郁症状尚缺乏研究。我们考察了自述患有ADHD或ASD的Twitter用户在与DSM-5九项抑郁症状相关的语言表达上是否存在差异。我们分析了792名用户的1,282,437条推文（2020-2021年）。推文首先由零样本分类器结合本地运行的大语言模型（Qwen3-8B）筛选抑郁相关性，再由在ReDSM5语料库上微调的MentalRoBERTa分类到九种症状类别中。该流程在两个各含500条推文的分层样本上进行调优，并针对其中150条推文的人工判定标签进行了内部验证。用户层面的症状画像（582名用户）通过L1惩罚逻辑回归、嵌套交叉验证、自助法稳定性分析、逐症状效应量以及共现分析进行比较。

    arXiv:2607.05626v2 Announce Type: replace  Abstract: Depression frequently co-occurs with attention-deficit/hyperactivity disorder (ADHD) and autism spectrum disorder (ASD), but how depressive symptoms are expressed by people with these conditions is underexplored. We examined whether Twitter users self-reporting ADHD or ASD differ in language related to the nine DSM-5 depressive symptoms. We analysed 1,282,437 tweets (2020-2021) from 792 users. Tweets were screened for depressive relevance by a zero-shot classifier combined with a locally run large language model (Qwen3-8B) and classified into the nine symptoms by MentalRoBERTa fine-tuned on the ReDSM5 corpus. The pipeline was tuned on two stratified samples of 500 tweets and internally validated against human-adjudicated labels on 150 of these tweets. User-level profiles (582 users) were compared with L1-penalised logistic regression, nested cross-validation, bootstrap stability analysis, per-symptom effect sizes, and co-occurrence a
    
[^283]: 超越满意度：学习内容、评论与幸福感之间的关联

    Beyond Satisfaction: Learning Associations Between Content, Reviews, and Well-Being

    [https://arxiv.org/abs/2607.02539](https://arxiv.org/abs/2607.02539)

    该论文提出以心理学PERMA幸福感框架替代传统满意度指标作为推荐目标，通过分析1370万条Goodreads评论和1.6万本书的内容，揭示了书籍内容特征与用户享乐型及实现型幸福感表达之间的关联。

    

    推荐算法塑造了人们在线消费的内容，但它们主要针对星级评分、点赞和情感等用户满意度信号进行优化。这些信号是否能捕捉到超越即时偏好的、真正对用户有益的东西，在很大程度上尚未得到研究。幸福感提供了一个植根于心理学的替代目标，超越了满意度这一范畴。我们将其操作化为评论中五个PERMA维度的即时和持久表达。实证分析表明，即时积极情绪和投入更多呈现享乐型特征（快乐导向），而人际关系、意义、成就以及持久的积极情绪和投入则表现为实现型特征（意义与成长导向）。通过分析1370万条Goodreads评论和1.6万本书的内容，我们研究了PERMA表达与满意度目标之间的差异，以及哪些书籍内容特征与它们相关联。首先，满意度信号……

    arXiv:2607.02539v2 Announce Type: replace-cross  Abstract: Recommendation algorithms shape what people consume online, yet they are primarily optimized for star ratings, likes, and sentiment, signals of user satisfaction. Whether these signals capture what is good for users, beyond momentary preference, remains largely unexamined. Well-being offers an alternative target grounded in psychology that extends beyond satisfaction. We operationalize it as immediate and enduring expressions of the five PERMA facets in reviews. Empirically, immediate positive emotion and engagement cluster as more hedonic (pleasure-oriented), while relationships, meaning, accomplishment, and enduring positive emotion and engagement behave as more eudaimonic (meaning- and growth-oriented). Analyzing 13.7 million Goodreads reviews and the content of 16 thousand books, we study how PERMA expressions differ from satisfaction targets and which book content features are associated with them. First, satisfaction sign
    
[^284]: 多头循环记忆智能体

    Multi-Head Recurrent Memory Agents

    [https://arxiv.org/abs/2607.01523](https://arxiv.org/abs/2607.01523)

    该论文诊断出循环记忆智能体性能随上下文增长而退化的瓶颈在于记忆保持而非记忆捕获，并提出无需训练的多头循环记忆（MHM）框架，通过将记忆划分为独立记忆头并采用“先选择后更新”策略保护未被选中的记忆免遭覆盖，从而显著提升长上下文的可靠性。

    

    循环记忆智能体通过将输入迭代地整合到固定大小的记忆窗口中，将大语言模型（LLM）扩展到任意长的上下文。尽管具备可扩展性，这类智能体存在一个有据可查的可靠性问题：随着上下文长度的增长，端到端性能会系统性下降。我们通过将性能分解为两个因素——记忆捕获和记忆保持——来诊断这一失败，并定量证实记忆保持是主导瓶颈。记忆保持之所以崩溃，是因为现有设计将记忆维护为单一的文本块，迫使每次更新都面临覆盖之前保留内容的风险。基于这一诊断，我们提出了多头循环记忆（MHM），这是一个通用的、无需训练的框架，它将记忆划分为独立的记忆头，并由分阶段的“先选择后更新”策略加以控制。在每一步中，恰好选择一个记忆头进行更新，而其余记忆头在结构上受到保护（免于被修改）。

    arXiv:2607.01523v2 Announce Type: replace-cross  Abstract: Recurrent memory agents extend LLMs to arbitrarily long contexts by iteratively consolidating input into a fixed-size memory window. Despite their scalability, these agents exhibit a well-documented reliability problem: end-to-end performance degrades systematically as context length grows. We diagnose this failure by decomposing performance into two factors---memory capture and memory retention---and quantitatively confirm that retention is the dominant bottleneck. Retention collapses because existing designs maintain memory as a monolithic text block, forcing every update to risk overwriting previously retained content. Motivated by this diagnosis, we propose Multi-Head Recurrent Memory (MHM), a general, training-free framework that partitions memory into independent heads governed by a stage-wise select-then-update strategy. At each step, exactly one head is selected for update while the remaining heads are structurally shie
    
[^285]: 基于梯度反转与变分信息瓶颈的语音欺骗检测语言偏差消除方法

    Linguistic Bias Mitigation for Spoofing Detection via Gradient Reversal and A Variational Information Bottleneck

    [https://arxiv.org/abs/2606.31411](https://arxiv.org/abs/2606.31411)

    该论文提出了一种基于师生对抗学习与变分信息瓶颈的语言不变语音欺骗检测框架，通过梯度反转消除语言偏差，在九个DF Arena数据集上实现了最高36.2%的等错误率相对降低。

    

    生成式语音技术的快速发展已经损害了语音生物特征识别的可靠性。虽然当前的欺骗检测器在域内条件下评估时表现出色，但对域外设置的泛化能力通常较差。我们表明这可能是由语言偏差造成的。对训练数据中观察到的语言线索的依赖会损害跨数据场景下的鲁棒性。我们提出了一种利用师生对抗学习的语言不变欺骗检测框架。在外部数据集的语言内容上预训练的语言感知教师模型，通过梯度反转引导学生检测器最小化语言信息。为防止非语言线索被意外移除，我们引入了变分信息瓶颈来实现对主要线索的抑制。在九个DF Arena数据集上，与现有方法相比，我们的方法在等错误率（EER）方面实现了高达36.2%的相对降低。

    arXiv:2606.31411v2 Announce Type: replace  Abstract: Rapid advancements in generative speech technology have compromised the reliability of voice biometrics. While current spoofing detectors excel when assessed under in-domain conditions, generalisation to out-of-domain settings is often poor. We show that this can be due to linguistic bias. A reliance on linguistic cues observed in training data can then compromise robustness to cross-data. We propose a linguistic-invariant spoofing detection framework utilizing teacher-student adversarial learning. The linguistic-aware teacher model, pre-trained on linguistic content of an external dataset, guides the student detector via gradient reversal to minimize the linguistic information. To prevent the inadvertent removal of non-linguistic cues, we incorporate a Variational Information Bottleneck to enable suppression of principal cues. Across nine DF Arena datasets, our method achieves up to a 36.2% relative reduction in the EER compare to t
    
[^286]: DriftGuard：面向演化毒性内容审核的安全感知多监控器检测与选择性适应

    DriftGuard: Safety-Aware Multi-Monitor Detection and Selective Adaptation for Evolving Toxicity Moderation

    [https://arxiv.org/abs/2606.28725](https://arxiv.org/abs/2606.28725)

    论文提出 DriftGuard 框架，通过监控全局文本漂移、身份伤害漂移、模型不确定性和假阴性风险等多个维度的安全相关漂移信号，检测毒性内容审核中的安全隐患，并利用优先纳入高风险样本的困难混合适应集对模型进行选择性更新，以应对不断演化的有害内容。

    

    arXiv:2606.28725v2 公告类型： replace  摘要：自动化毒性内容审核系统运行于动态在线环境中，其中有害行为通过暗语、攻击目标转移以及对审核策略的适应性规避而不断演化。现有的漂移检测方法通常聚焦于全局分布变化，但此类信号可能遗漏出现在局部有害子空间或高风险模型错误区域中的安全相关变化。本文提出了 DriftGuard，一个安全感知的自适应审核框架，将多监控器漂移检测与选择性模型更新相结合。该框架跟踪全局文本漂移、身份伤害漂移、模型不确定性、毒性风险漂移以及假阴性风险漂移。当检测到安全相关变化时，模型使用一个“困难混合”（hard-mix）适应集进行更新，该集合优先纳入可能的假阴性样本、与身份相关的高风险样本、假阳性风险样本以及不确定的边界样本。在 Civil Comments 时间序列数据上的实验……

    arXiv:2606.28725v2 Announce Type: replace  Abstract: Automated toxicity moderation systems operate in dynamic online environments where harmful behavior evolves through coded language, shifting targets, and strategic adaptation to enforcement. Existing drift detection methods often focus on global distributional change, but such signals may miss safety-relevant shifts that emerge in localized harm subspaces or high-risk model-error regions. This paper introduces DriftGuard, a safety-aware adaptive moderation framework that combines multi-monitor drift detection with selective model updating. The framework tracks global text drift, identity-harm drift, model uncertainty, toxic-risk drift, and false-negative-risk drift. When safety-relevant change is detected, the model is updated using a hard-mix adaptation set that prioritizes likely false negatives, identity-related high-risk examples, false-positive-risk examples, and uncertain boundary cases. Experiments on Civil Comments temporal s
    
[^287]: 从权重到特征：基于SAE引导的激活正则化实现大语言模型持续学习

    From Weights to Features: SAE-Guided Activation Regularization for LLM Continual Learning

    [https://arxiv.org/abs/2606.26629](https://arxiv.org/abs/2606.26629)

    本文提出利用稀疏自编码器（SAE）在激活空间进行正则化，通过单义特征字典显式平衡稳定性与可塑性，解决了大语言模型持续学习中EWC等权重正则化方法因多义性而表现不佳的问题。

    

    权重空间正则化方法（如弹性权重巩固EWC）是持续学习中应对灾难性遗忘的标准方法。然而，当这些方法应用于大语言模型时，往往表现不佳。我们认为，这种不佳表现部分可归因于大语言模型的“多义性”本质：EWC风格正则化所使用的逐权重重要性估计过于粗糙，无法隔离需要保护的知识。本文提出在模型的激活空间中进行正则化，利用预训练的稀疏自编码器（SAE）作为单义特征字典。从约束优化的角度，我们推导出一种新的损失函数，该函数利用SAE特征字典显式地平衡稳定性与可塑性，并证明EWC是单侧权重空间惩罚设置中的一个特例。与需要存储或重放数据的基于重放的方法不同，我们的方法避免了数据存储需求。

    arXiv:2606.26629v1 Announce Type: cross  Abstract: Weight-space regularization methods such as Elastic Weight Consolidation (EWC) are the standard approach to catastrophic forgetting in continual learning. However, those methods tend to underperform when applied to large language models. We argue that such underperformance can be partly explained by the ``polysemantic'' nature of large language models: per-weight importance estimates utilized by EWC-style regularization are too coarse and cannot isolate the knowledge that needs protection. In this paper, we propose regularizing instead in the model's activation space, using pretrained Sparse Autoencoders (SAEs) as a monosemantic feature dictionary. From the perspective of constrained optimization, we derive a new loss function that uses the SAE feature dictionary to explicitly balance stability and plasticity, and show that EWC is a special case in the one-sided weight-space penalty setting. Unlike replay-based methods that store or re
    
[^288]: 大语言模型品牌相关回答中引用的网络来源：一项多语言审计

    Cited Web Sources in LLM Answers about Brands: A Multilingual Audit

    [https://arxiv.org/abs/2606.25787](https://arxiv.org/abs/2606.25787)

    该研究对涵盖128个品牌、13种语言和12个市场的167,551条大语言模型回答中的引用网址进行了多语言审计，揭示了引用来源的高度集中性（少数域名占80%的引用且维基百科占据主导地位），并论证了标准化、分类规则和明确分母在引用审计中的必要性。

    

    显示出的引用使生成式搜索的一部分变得可观察，但它们并不能揭示其完整的检索过程。本次审计涵盖三个存档数据集，包含167,551条带有URL的引用条目，涉及128个品牌、13种语言和12个本土市场。在主要的北欧-波罗的海部分中，21,033个可注册域名中有3,863个占131,514次引用的80%。维基百科在12种查询语言中的11种中居于领先地位。一份恢复的、独立的品牌相关网站分类保留了131,512条条目：其中108,410条（82.4%）不属于其品牌相关类别，23,102条（17.6%）属于该类别。该分类并不能确立法律所有权。在完整的57个品牌的波兰语汇总数据中，四个招聘网站共获得637次引用，而波兰语维基百科仅为297次。这些结果说明了为什么标准化、分类规则和分母在引用审计中不可或缺。它们描述的是显示出的链接，而并未估计对品牌的因果影响（原文在此处截断）。

    arXiv:2606.25787v2 Announce Type: replace-cross  Abstract: Displayed citations make part of generative search observable, but they do not reveal its complete retrieval process. This audit covers three archived datasets containing 167,551 URL-bearing citation entries across 128 brands, 13 languages and 12 home markets. In the principal Nordic-Baltic component, 3,863 of 21,033 registrable domains account for 80% of 131,514 citations. Wikipedia leads in 11 of 12 query languages. A recovered, separate brand-associated website classification retains 131,512 entries: 108,410 (82.4%) fall outside its brand-associated category and 23,102 (17.6%) inside. This classification does not establish legal ownership. In the full 57-brand Polish aggregate, four recruitment hosts receive 637 citations against 297 for Polish Wikipedia. The results show why normalization, classification rules and denominators belong in citation audits. They describe displayed links without estimating causal effects on bran
    
[^289]: RAS：通过拒绝对齐衡量大语言模型的安全性

    RAS: Measuring LLM Safety Through Refusal Alignment

    [https://arxiv.org/abs/2606.25750](https://arxiv.org/abs/2606.25750)

    本文提出RAS（拒绝对齐分数），一种基于表示、仅需前向传播即可高效评估大语言模型拒绝有害请求能力的指标，在Llama、Gemma和Qwen系列模型上与真实攻击成功率高度相关。

    

    对开放权重大语言模型进行重复安全评估需要为众多相关检查点生成并评判响应。我们提出了RAS（拒绝对齐分数，Refusal Alignment Score），这是一种基于表示、相对于参考模型的指标，用于衡量模型对有害请求的拒绝能力，由五阶段的SafeVec流程计算得出。SafeVec从对齐的参考模型中提取逐层拒绝方向，测量这些方向在相关目标模型中的对齐程度，并利用参考模型和校准模型的行为攻击成功率（ASR）将分数校准至0到100之间。此后对目标模型的评分仅需输入的前向传播即可完成。在来自Llama、Gemma和Qwen系列的18个留出变体上，RAS与100(1−ASR)高度相关（皮尔逊相关系数0.934，斯皮尔曼相关系数0.851），并实现83.0%的成对排序一致性。在Llama参考模型的五种提示配置下，各方向平均余弦相似度为0.939。当校准ASR已经可用时……（摘要原文在此处截断）

    arXiv:2606.25750v2 Announce Type: replace-cross  Abstract: Repeated safety evaluation of open-weight large language models requires generating and judging responses for many related checkpoints. We introduce RAS (Refusal Alignment Score), a representation-based, reference-relative indicator of harmful-request refusal computed by the five-stage SafeVec procedure. SafeVec extracts layer-wise refusal directions from an aligned reference model, measures their alignment in related targets, and calibrates a score from 0 to 100 using behavioral attack success rates (ASR) from reference and calibration models. Subsequent target scoring uses input forward passes alone. Across 18 held-out variants from the Llama, Gemma, and Qwen families, RAS correlates with $100(1-\mathrm{ASR})$ (Pearson 0.934, Spearman 0.851) and achieves 83.0% pairwise ordering agreement. Across five prompt configurations for the Llama reference, directions have mean cosine similarity 0.939. With calibration ASR already avail
    
[^290]: HandwritingAgent：可伸缩向量空间中由语言驱动的手写合成

    HandwritingAgent: Language-Driven Handwriting Synthesis in Scalable Vector Space

    [https://arxiv.org/abs/2606.18788](https://arxiv.org/abs/2606.18788)

    HandwritingAgent 是一个语言驱动的智能体，无需针对特定风格进行训练，即可利用大型推理模型直接以 SVG 格式合成自然手写笔画序列，并可通过自然语言灵活控制书写风格。

    

    教会机器模仿自然的手写风格仍然是一个未解决的挑战，因为它需要合成在形状、纹理、笔压和字体上动态变化的笔画序列——不仅在个体之间存在差异，在同一个人的书写中也会变化。针对这一挑战的尝试主要探索了在线（online）和离线（offline）两种设置下的深度学习方法。然而，这些方法往往受限于特定风格的架构选择、对大型数据集的高度依赖、高昂的计算成本，以及缺乏通过自然语言对书写风格进行灵活控制的能力。为此，我们提出了 HandwritingAgent，一个语言驱动的智能体，它可以直接以可伸缩矢量图形（SVG）格式合成自然的手写序列，而无需针对特定风格进行训练。该智能体利用大型推理模型对目标手写字符进行几何分析，并以自回归的方式将其生成为笔画序列……

    arXiv:2606.18788v2 Announce Type: replace-cross  Abstract: Teaching machines to emulate natural handwriting styles remains an open challenge, as it requires synthesizing stroke sequences that dynamically vary in shape, texture, pressure and script - not only across individuals, but also within a single person's handwriting. Attempts at this challenge have largely explored deep learning methods in both online and offline settings. However, these approaches are often constrained by style-specific architectural choices, heavy reliance on large datasets, high compute costs, and a lack of flexible control over writing styles through natural language. To this end, we introduce HandwritingAgent, a language-driven agent that can synthesize natural handwriting sequences directly in Scalable Vector Graphics (SVG) format with no need for style-specific training. The agent leverages a large reasoning model to geometrically analyse and autoregressively generate target handwritten glyphs as stroke s
    
[^291]: 不确定性并非临床视觉问答的安全网，但它能否预判模型失效？

    Uncertainty Is Not a Safety Net for Clinical VQA, but Can It Anticipate Model Failure?

    [https://arxiv.org/abs/2606.16583](https://arxiv.org/abs/2606.16583)

    该研究通过在12个临床视觉语言模型上基准测试8种不确定性估计方法，发现不确定性估计的质量会随模型准确率下降而退化、且在正确答案被隐藏时无法发出警示，但未扰动输入上的不确定性本身携带诊断信息，能够预判哪些预测将在模型失效时崩溃。

    

    临床视觉语言模型的安全部署需要可靠的不确定性估计，即一种指示何时应当信任预测结果、或何时应将决策升级给临床医生的信号。我们检验了当前的不确定性估计方法是否真正能够提供这种信号。通过在临床视觉问答任务上对12个视觉语言模型中的8种不确定性估计方法进行基准测试，我们发现不确定性估计的质量并非该方法本身的内在属性：它跟随模型准确率而变化，恰好在模型性能最弱、也最需要可靠性保证的地方发生退化。当我们通过在多选题答案中隐藏正确选项（NOTA扰动）对模型进行压力测试时，准确率急剧下降，而不确定性却几乎没有任何变化，导致模型出现系统性的校准失准。然而我们发现，未受扰动输入上的不确定性能够可靠地预判哪些预测会在NOTA扰动下崩溃，这表明当前视觉语言模型中的不确定性携带关于模型失效的诊断信息。

    arXiv:2606.16583v2 Announce Type: replace  Abstract: Safe deployment of clinical vision-language models (VLMs) requires reliable uncertainty estimation (UE): a signal indicating when predictions should be trusted or escalated to a clinician. We test whether current UE methods actually deliver this signal. Benchmarking 8 methods across 12 VLMs on clinical visual question-answering (VQA), we find that UE quality is not an intrinsic property of the UE method: it tracks model accuracy, degrading precisely where the model performance is weakest, and therefore where reliability is most needed. When we stress-test models by hiding the correct option among the multiple-choice answers (NOTA perturbations), accuracy collapses while uncertainty barely changes, leaving models systematically miscalibrated. Yet, we find that uncertainty on the unperturbed input reliably anticipates which predictions will collapse under NOTA, indicating that UE in current VLMs carries diagnostic information about mod
    
[^292]: 心理健康对话中的专家级危机检测

    Expert-Level Crisis Detection in Mental Health Conversations

    [https://arxiv.org/abs/2606.10380](https://arxiv.org/abs/2606.10380)

    该论文提出了临床医生标注的CRADLE-Dialogue基准数据集以及“警报-确认”评估协议，用于解决多轮心理健康对话中轮次级危机检测的难题，使模型能够捕捉随对话演进的风险信号并支持早期干预。

    

    现实世界的危机干预本质上是对话式的，然而现有研究主要集中于静态文本。当应用于多轮对话时，当前模型表现出显著的性能下降，难以追踪随着上下文演变而出现的风险信号。为了弥补这一空白，我们推出了CRADLE-Dialogue，这是一个由临床医生标注的、用于对话环境中轮次级危机检测的基准数据集。该数据集包含600段对话，针对基于临床的风险（包括自杀意念、自残和虐待儿童）进行了多标签标注，并区分了过去风险与正在发生的风险。我们进一步提出了“警报-确认”评估协议，将早期预警信号与特定危机变得明确可识别的对话轮次区分开来，体现了在风险变得明显之前进行干预的临床需求。实验表明，识别风险何时出现远比识别风险本身困难得多。

    arXiv:2606.10380v2 Announce Type: replace  Abstract: Real-world crisis intervention is inherently conversational, yet existing research largely focuses on static texts. When applied to multi-turn dialogues, current models exhibit significant performance degradation, struggling to track risk signals that emerge as context evolves. To address this gap, we introduce CRADLE-Dialogue, a clinician-annotated benchmark for turn-level crisis detection in conversational settings. The dataset features 600 dialogues with multi-label annotations across clinically grounded risks, including suicide ideation, self-harm, and child abuse, distinguishing past from ongoing risk. We further propose an Alert-Confirm evaluation protocol that distinguishes early warning signals (Alert) from turns where a specific crisis becomes explicitly identifiable (Confirm), reflecting the clinical need to intervene before risk becomes explicit. Experiments show that identifying when risk emerges is much harder than recog
    
[^293]: CARE：面向医学摘要生成的保形安全层

    CARE: A Conformal Safety Layer for Medical Summarization

    [https://arxiv.org/abs/2606.08969](https://arxiv.org/abs/2606.08969)

    CARE 是一种无需重新训练模型的事后保形风险控制层，通过 CRC 与 LTT-FST 为医学摘要添加经过校准的遗漏与幻觉标注，在形式上控制漏检错误概率并平衡安全性与临床审阅负担。

    

    大型语言模型（LLM）正越来越多地被用于医学摘要生成，但它们可能遗漏医学上重要的信息，并引入缺乏依据的论断。现有的错误检测方法只能产生启发式或未经校准的分数，既无法对漏检错误提供形式化保证，也无法以原则性的方式在安全性与临床医生审阅负担之间进行权衡。我们提出了用于风险评估的保形评估方法（CARE），这是一种事后风险控制层，无需重新训练摘要模型即可添加经过校准的遗漏与幻觉标注。CARE 使用标量保形风险控制（CRC）来限定文档中存在未被标记的幻觉句子的概率，并使用“先学习后检验”固定序列检验（LTT-FST）来控制未被提交审阅的真实遗漏的期望比例。遗漏信息的呈现同时依赖于句子重要性和摘要覆盖率，这构成一个二维阈值选择问题；我们证明了……（原文摘要在此处截断）

    arXiv:2606.08969v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used for medical summarization but can omit medically important information and introduce unsupported claims. Existing error-detection methods produce heuristic or uncalibrated scores, providing no formal control over missed errors and no principled way to trade off safety against clinician review burden. We introduce Conformal Assessment for Risk Evaluation (CARE), a post-hoc risk-control layer that adds calibrated omission and hallucination annotations without retraining the summarizer. CARE uses scalar conformal risk control (CRC) to bound the probability that a document contains an unflagged hallucinated sentence, and Learn-Then-Test fixed-sequence testing (LTT-FST) to control the expected fraction of true omissions left unsurfaced for review. Omission surfacing depends jointly on sentence importance and summary coverage, a two-dimensional threshold-selection problem; we show th
    
[^294]: 编码智能体在欺骗我们吗？基于随机测试的封顶评估方法来检测与防止作弊

    Do Coding Agents Deceive Us? Detecting and Preventing Cheating via Capped Evaluation with Randomized Tests

    [https://arxiv.org/abs/2606.07379](https://arxiv.org/abs/2606.07379)

    该论文提出CapCode框架与CapReward奖励设计，通过随机测试将合法可达分数刻意封顶在100%以下，使超出上限的分数成为作弊证据，从而检测并抑制编码智能体评估中的作弊行为。

    

    在智能体评估与训练中，一个日益严重的失效模式是：模型可以通过利用捷径而非真正解决预期任务来获得高评估分数（例如测试通过率），从而使这些分数具有欺骗性。这导致评估分数无法可靠地衡量模型真实的问题解决能力。我们提出了CapCode，这是一个用于构建带随机测试的编码数据集的框架，其中通过合法方式可达到的最高分数被刻意封顶在100%以下。这种封顶分数设计使评估分数具有更清晰的解释：显著超过上限的分数是不合理的，因此可以作为作弊的证据。为了防止作弊，我们提出了CapReward，这是一种基于CapCode原则的奖励设计，旨在抑制超出上限的优化行为。在多个数据集上的实验表明，CapCode能够在保持模型性能排名的同时检测作弊行为，而CapReward减少了作弊行为，从而……

    arXiv:2606.07379v3 Announce Type: replace-cross  Abstract: A growing failure mode in agent evaluation and training is that models can achieve high evaluation scores (e.g., test pass rates) by exploiting shortcuts instead of solving the intended task, making these scores deceptive. This makes evaluation scores unreliable as measures of true task-solving ability. We propose CapCode, a framework for constructing coding datasets with randomized tests whose best achievable non-cheating score is deliberately capped below 100%. This capped-score design gives evaluation scores a clearer interpretation: scores substantially above the cap are implausible and therefore provide evidence of cheating. To prevent cheating, we propose CapReward, a reward design based on the CapCode principle to discourage optimization beyond the cap. Experiments across multiple datasets show that CapCode detects cheating while preserving performance ranking of models, and CapReward reduces cheating behavior, yielding 
    
[^295]: K-BrowseComp：基于韩语语境的网络浏览智能体基准测试

    K-BrowseComp: A Web Browsing Agent Benchmark Grounded in Korean Contexts

    [https://arxiv.org/abs/2606.02404](https://arxiv.org/abs/2606.02404)

    K-BrowseComp是首个基于韩语语境构建的网络浏览智能体基准，包含400个问题（其中300个经韩语母语者人工验证），实验显示前沿大模型在韩语环境下表现显著下降（30%–45.67%），而韩国本土模型表现更差（0%–10.33%）。

    

    前沿模型的评估正从基础能力（如指令遵循和推理）转向复合型智能体能力，但韩语智能体基准测试仍然稀缺。我们推出了K-BrowseComp，一个基于韩语语境的网络浏览智能体基准，包含400个问题。其中300个问题的K-BrowseComp-Verified子集由韩语母语者人工构建并验证。在该子集上，包括GPT-5.5、DeepSeek-V4-Pro和GLM-5.1在内的前沿大语言模型仅达到30.00%–45.67%的成绩，相比BrowseComp大幅下降，而通过韩国自主AI基础模型计划发布的韩语大语言模型仅获得0.00%–10.33%。我们进一步利用解决与创建网络浏览问题之间的不对称性，采用困难少样本示例和针对失败模式的生成方法，构建了包含100个问题的合成子集。在经过对抗性过滤的合成诊断子集上，最强的模型……

    arXiv:2606.02404v2 Announce Type: replace  Abstract: Frontier model evaluations are shifting from foundational capabilities (e.g., instruction following and reasoning) toward compositional, agentic ones, but Korean agentic benchmarks remain scarce. We introduce K-BrowseComp, a web-browsing agent benchmark grounded in Korean contexts, consisting of 400 problems. The 300-problem K-BrowseComp-Verified subset is manually constructed and validated by native Korean speakers. On this subset, frontier LLMs, including GPT-5.5, DeepSeek-V4-Pro, and GLM-5.1, reach only 30.00--45.67\%, a substantial drop from BrowseComp, while Korean LLMs released through Korea's Proprietary AI Foundation Model program obtain only 0.00--10.33\%. We further construct a 100-problem synthetic split using hard few-shot exemplars and failure-mode-targeted generation to exploit the asymmetry between solving and creating web browsing problems. On the adversarially filtered synthetic diagnostic split, the strongest model 
    
[^296]: BioELX：无需任务特定监督的上下文感知跨语言生物医学实体链接

    BioELX: Context-Aware Cross-lingual Biomedical Entity Linking without Task-Specific Supervision

    [https://arxiv.org/abs/2605.27380](https://arxiv.org/abs/2605.27380)

    BioELX提出了一种检索-重排序框架，利用Wikidata衍生的跨语言别名监督进行检索，并通过提及锚定提示将LLM重排序器适配于实体链接，实现了无需任务特定监督的上下文感知跨语言生物医学实体链接。

    

    跨语言生物医学实体链接（BEL）将任何语言中的提及（mention）映射到生物医学知识库中的唯一标识符，为临床和生物医学NLP应用提供支持。我们发现了影响当前系统的两个问题。首先，用于训练跨语言BEL检索器的UMLS别名严重偏向英语，导致检索器对非英语提及的泛化能力较差。其次，尽管上下文通常对消歧是必要的，但将上下文简单地注入仅以别名对齐为目标训练的检索器中，会严重降低检索性能。我们提出了BioELX，一个解决这两个问题的检索-重排序框架。在检索阶段，我们使用源自Wikidata的跨语言别名监督对SapBERT_multi进行继续训练，从而在各语言之间形成共享的概念邻域。在重排序阶段，我们通过提及锚定提示（mention-anchored prompting）将预训练的LLM重排序器适配到实体链接任务。

    arXiv:2605.27380v2 Announce Type: replace-cross  Abstract: Cross-lingual biomedical entity linking (BEL) maps mentions in any language to unique identifiers in a biomedical knowledge base, supporting clinical and biomedical NLP applications. We identify two issues affecting current systems. First, the UMLS (Bodenreider,2004) aliases used to train cross-lingual BEL retrievers are heavily skewed toward English, so retrievers generalize poorly to non-English mentions. Second, although context is often necessary for disambiguation, naively injecting context into retrievers trained only to align aliases severely degrades retrieval. We propose BioELX, a retrieve-rerank framework that addresses both issues. For retrieval, we continue training SapBERT_multi (Liu et al., 2021b) using Wikidata-derived cross-lingual alias supervision, forming shared concept neighborhoods across languages. For reranking, we adapt pretrained LLM rerankers to entity linking through mention-anchored prompting, which 
    
[^297]: JuICE：用于评估LLM裁判识别文化错误能力的基准

    JuICE: A Benchmark for Evaluating LLM-Judge in Identifying Cultural Errors

    [https://arxiv.org/abs/2605.26955](https://arxiv.org/abs/2605.26955)

    JuICE是一个包含7,470个片段级标注的多语言基准数据集，用于评估LLM裁判识别长篇回复中深层文化错误的能力，弥补了现有文化基准将文化简化为扁平事实集合的不足。

    

    随着大语言模型（LLM）越来越多地部署给全球用户，它们被集成到跨多种文化背景的日常任务中，从起草个人通信到头脑风暴创意想法。这些任务本质上具有文化属性：它们需要上下文的恰当性、符号的共鸣以及母语使用者凭本能运用的隐性文化期望，这意味着一个回复可能在事实上看似合理，但对当地读者而言却明显是错误的。现有的文化基准通过事实验证或规范蕴含方法将文化视为一组扁平的事实，并且在采用“LLM作为裁判”（LLM-as-a-Judge）时，并未检验其是否能够捕捉此类深层次的文化错误。为弥补这一空白，我们提出了JuICE（LLM裁判识别文化错误能力评估基准），这是一个多语言数据集，包含对长篇LLM回复中文化和语言错误的7,470个片段级标注。

    arXiv:2605.26955v2 Announce Type: replace-cross  Abstract: As large language models (LLMs) are increasingly deployed to users around the world, they are integrated into everyday tasks across diverse cultural contexts, from drafting personal communications to brainstorming creative ideas. These tasks are inherently cultural: they require contextual appropriateness, symbolic resonance, and tacit cultural expectations that native speakers draw on instinctively, meaning that a response can be factually plausible yet unmistakably wrong to a local reader. Existing cultural benchmarks have treated culture as a flat set of facts via fact verification or norm entailment methods, and have adopted LLM-as-a-Judge without examining whether they can capture such thick cultural errors. To address this gap, we present JuICE (Benchmark for LLM-Judge in Identifying Cultural Errors), a multilingual dataset of 7,470 span-level annotations of cultural and linguistic errors in long-form LLM responses. It co
    
[^298]: SAMark：一种具有段落级改写鲁棒性的自锚定文本水印技术

    SAMark: A Self-Anchored Text Watermarking with Paragraph-Level Paraphrase Robustness

    [https://arxiv.org/abs/2605.25796](https://arxiv.org/abs/2605.25796)

    SAMark提出了一种自锚定文本水印框架，通过在语义空间中建立与句子顺序无关的绿色区域和多通道双曲评分机制，显著提升了水印对段落级改写攻击的鲁棒性，在1%假阳性率下实现了高达90.2%的真阳性率。

    

    语义级水印（SWM）通过将句子作为基本单元，提高了对文本修改的鲁棒性。然而，对段落级改写的鲁棒性仍然难以实现，因为此类攻击会通过改变句子顺序来全局性地破坏水印信号。在这项工作中，我们提出了SAMark，一种自锚定水印框架，通过在语义空间中建立与步骤无关的绿色区域，消除了对句子顺序的依赖。为了提高可检测性，我们引入了一种多通道双曲评分机制，在放大水印信号的同时抑制来自弱对齐候选的噪声。我们进一步提出了一种多样性感知过滤策略，将硬过滤与软正则化相结合，超越了简单的n-gram重复过滤方法，以解决语义冗余问题。实验结果表明，SAMark在典型的段落级改写攻击下实现了高达90.2%的TP@FP1%。

    arXiv:2605.25796v4 Announce Type: replace-cross  Abstract: Semantic-level watermarking (SWM) improves robustness against text modifications by treating sentences as the basic unit. However, robustness to paragraph-level paraphrasing remains difficult because such attacks globally disrupt watermark signals by changing sentence order. In this work, we propose SAMark, a self-anchored watermarking framework that removes the dependency on sentence order by establishing a step-independent green region in semantic space. To improve detectability, we introduce a multi-channel hyperbolic scoring mechanism that amplifies watermark signals while suppressing noise from weakly aligned candidates. We further propose a diversity-aware filtering strategy that combines hard filtering with soft regularization, extending beyond simple n-gram repetition filters to address semantic redundancy. Experimental results show that SAMark achieves up to 90.2% TP@FP1% under typical paragraph-level paraphrasing atta
    
[^299]: 多样性中的和谐：面向大型推理模型的多域对比策略优化

    Harmony in Diversity: Multi-domain Contrastive Policy Optimization for Large Reasoning Models

    [https://arxiv.org/abs/2605.25443](https://arxiv.org/abs/2605.25443)

    提出多域对比策略优化（MCPO）方法，通过在关键推理片段上应用对比学习，实现跨领域的知识共享与竞争的平衡，从而缓解混合领域强化学习中不同领域间的能力权衡问题，使单个大型推理模型的多领域性能得以匹配甚至超越单领域训练。

    

    多领域能力对于实际可用的大型推理模型（LRMs）至关重要。因此，混合领域强化学习（RL）已成为利用混合领域数据联合训练单个大型推理模型的关键过程，但它往往会在不同领域之间引发能力此消彼长的权衡问题。现有方法将这一问题归因于跨领域之间的竞争性交互，并通过隔离学习方向和平衡学习进度来加以缓解。然而，我们的观察表明，这些方法同时也阻碍了跨领域之间的有益知识共享，导致多领域训练难以达到甚至超越单领域训练的性能。为解决这一问题，我们提出了多域对比策略优化（MCPO），它应用对比学习来实现多样化的知识共享与竞争。我们首先将交互单元定位于关键推理片段上，然后构建相互……

    arXiv:2605.25443v3 Announce Type: replace  Abstract: Multi-domain capabilities are critical for practically applicable Large Reasoning Models (LRMs). Therefore, mixed-domain Reinforcement learning (RL) has become a crucial process to jointly train a single LRM with mixed domain data, but it often induces capability trade-offs among different domains. Existing methods attribute this issue to competing cross-domain interactions, and mitigate them by isolating learning directions and balancing learning progress. However, our observation suggests that they also impede supportive knowledge sharing across domains, which make it difficult to match or surpass single-domain performance. To address this problem, we propose \textbf{M}ulti-domain \textbf{C}ontrastive \textbf{P}olicy \textbf{O}ptimization (MCPO), which applies contrastive learning to achieve diverse knowledge sharing and competition. We first localize the interaction units on critical reasoning segments, and then formulate mutually
    
[^300]: AI智能体可能永远无法抵御提示注入攻击

    AI Agents May Always Fall for Prompt Injections

    [https://arxiv.org/abs/2605.17634](https://arxiv.org/abs/2605.17634)

    本文借助上下文完整性理论重新定义提示注入攻击，并提出不可能性论证，证明数据-指令分离等现有防御范式注定无法彻底解决提示注入问题，因为攻击者总能构造使恶意信息流显得合法的上下文。

    

    提示注入是已部署AI智能体中最关键的安全漏洞。尽管近期取得了一些进展，我们证明主流防御范式（数据-指令分离）既无法检测通过上下文操纵实施的攻击，又会损害符合上下文的合理行为。随后，我们通过上下文完整性这一隐私理论视角重新审视提示注入，该理论用于判断信息流是否符合上下文规范。这一重新框架化解释了当前防御试图修补的攻击类型，并预测了未来智能体将面临的高级攻击。我们开发了独特的良性场景与攻击场景，迫使智能体通过以下三种方式违反规范：（1）歪曲信息流，（2）操纵规范，或（3）混合多个信息流。这一重新阐释引出了一个不可能性论证：攻击者总能构造出一个使被阻止的信息流显得合法的上下文，而收紧规范的防御者则会阻止真正合法的信息流。

    arXiv:2605.17634v2 Announce Type: replace-cross  Abstract: Prompt injection is the most critical vulnerability in deployed AI agents. Despite recent progress, we show that the prevailing defense paradigm (data-instruction separation) both fails to detect attacks that operate through contextual manipulation and degrades contextually appropriate behavior. We then recast prompt injection via the lens of Contextual Integrity (CI), a privacy theory that judges information flow compliance with contextual norms. This explains types of attacks that current defenses attempt to patch and predict advanced ones future agents will face. We develop unique benign and attack scenarios that force an agent to violate the norms by (1) misrepresenting the flow, (2) manipulating norms, or (3) mixing multiple flows. This reframing suggests an impossibility argument: an adversary can always construct a context under which a blocked flow appears legitimate, or a defender who tightens norms will block genuinel
    
[^301]: HEED：面向混合视觉语言模型蒸馏的密度加权残差对齐

    HEED: Density-Weighted Residual Alignment for Hybrid Vision-Language Model Distillation

    [https://arxiv.org/abs/2605.17093](https://arxiv.org/abs/2605.17093)

    该论文发现混合架构蒸馏的失败集中于承载文本与局部细节的高密度图像块（其残差漂移达3.6倍），并提出密度加权残差对齐方法HEED，有效修复了学生在OCR和文档任务上的细粒度信息丢失。

    

    将视觉语言模型蒸馏到更快速的混合架构（如3:1的Mamba-2/注意力混合结构）中，如今已成为实现高效推理的标准做法。整体基准测试显示该方法行之有效，但掩盖了选择性的失败。当我们将Qwen3-VL-8B-Instruct蒸馏到3:1的Mamba-2/注意力混合架构中时，学生模型在MMStar、MMBench和MMMU-Pro等视觉推理基准上与教师模型的差距保持在2分以内，但在光学字符识别和文档任务上却下降了13分——学生模型仍能理解场景，却丢失了回答所需的细粒度文本。我们将这一失败的大部分原因定位于一类特定的位置：在高分辨率图像中，大多数图像块是天空、墙壁或平滑纹理，而一小部分图像块承载着文本、边缘、物体边界或其他局部细节。在token级别的诊断中，密度最高的前10%图像块的残差漂移比基线大3.6倍。

    arXiv:2605.17093v2 Announce Type: replace-cross  Abstract: Distilling vision-language models into faster hybrid architectures, such as 3:1 Mamba-2/attention mixes, is now standard practice for making inference efficient. Aggregate benchmarks suggest that this works but they hide selective failures. When we distill Qwen3-VL-8B-Instruct into a 3:1 Mamba-2/attention hybrid, student model stays within 2 points of the teacher across visual reasoning benchmarks like MMStar, MMBench, and MMMU-Pro, while dropping 13 points on optical-character-recognition and document tasks. The student can still understand the scene but loses the fine-grained text needed to answer. We localize much of the failure to a specific kind of position. In a high-resolution image, most patches are sky, wall, or smooth texture, while a small fraction carries text, edges, object boundaries, or other local details. In a token-level diagnostic, the top 10% highest-density patches have 3.6x larger residual drift than the b
    
[^302]: 在语言模型中诱导人工不确定性

    Inducing Artificial Uncertainty in Language Models

    [https://arxiv.org/abs/2605.13595](https://arxiv.org/abs/2605.13595)

    该论文提出在语言模型中诱导人工不确定性的新问题及方法，以解决当数据集饱和、难以找到足够不确定的监督数据时，不确定性量化方法可能失效的挑战。

    

    在安全关键应用中，语言模型应当能够用有意义的概率来表征其不确定性。许多不确定性量化方法需要监督数据；然而，对于在海量抓取数据上训练的大型语言模型而言，寻找合适的未见过的、具有挑战性的数据变得越来越困难。如果模型对其预测始终（且正确地）保持自信，那么不确定性量化方法可能会在新颖和陌生的数据上一贯地高估置信度。因此，找到展现出足够不确定性的数据来为高性能模型训练监督式不确定性量化方法可能颇具挑战，并且随着大语言模型使数据集趋于饱和，这一难度还将不断增加。为了解决这一问题，我们首先引入了在语言模型中诱导人工不确定性的问题，然后研究了在缺乏……的情况下，如何在极其简单的数据上诱导人工不确定性的方法。

    arXiv:2605.13595v2 Announce Type: replace  Abstract: In safety-critical applications, language models should be able to characterize their uncertainty with meaningful probabilities. Many uncertainty quantification approaches require supervised data; however, finding suitable unseen challenging data is increasingly difficult for large language models trained on vast amounts of scraped data. If the model is consistently (and correctly) confident in its predictions, the uncertainty quantification method may consistently overestimate confidence on new and unfamiliar data. Finding data which exhibits enough uncertainty to train supervised uncertainty quantification methods for high-performance models may therefore be challenging, and will increase in difficulty as LLMs saturate datasets. To address this issue, we first introduce the problem of inducing artificial uncertainty in language models, then investigate methods of inducing artificial uncertainty on trivially easy data in the absence
    
[^303]: LatentRouter：我们能在看到答案之前选对多模态大语言模型吗？

    LatentRouter: Can We Choose the Right Multimodal Large Language Model Before Seeing Its Answer?

    [https://arxiv.org/abs/2605.11301](https://arxiv.org/abs/2605.11301)

    该论文提出LatentRouter框架，将多模态大语言模型路由建模为反事实多模态效用预测问题，通过路由胶囊与模型能力token之间的潜在通信，在看到模型答案之前预测并选出最适合当前图像-问题查询的模型。

    

    多模态大语言模型（MLLMs）在OCR、图表理解、空间推理、视觉问答、成本和延迟等方面具有各不相同的优势。因此，有效的MLLM路由不仅仅是估计查询难度：路由器必须将当前图像-问题输入的多模态需求与每个候选模型的能力相匹配。我们提出了LatentRouter，这是一个将MLLM路由形式化为反事实多模态效用预测的路由器。给定一个图像-问题查询，LatentRouter提取学习到的多模态路由胶囊，用模型能力token表示每个候选MLLM，并在这些状态之间执行潜在通信，以估计每个模型如果被选中时的表现。分布式的结果预测头预测模型特定的反事实质量，同时有界的胶囊修正机制细化接近的决策，而不让残差信号主导预测。

    arXiv:2605.11301v2 Announce Type: replace  Abstract: Multimodal large language models (MLLMs) have heterogeneous strengths across OCR, chart understanding, spatial reasoning, visual question answering, cost, and latency. Effective MLLM routing therefore requires more than estimating query difficulty: a router must match the multimodal requirements of the current image-question input with the capabilities of each candidate model. We propose LatentRouter, a router that formulates MLLM routing as counterfactual multimodal utility prediction. Given an image-question query, LatentRouter extracts learned multimodal routing capsules, represents each candidate MLLM with a model capability token, and performs latent communication between these states to estimate how each model would perform if selected. A distributional outcome head predicts model-specific counterfactual quality, while a bounded capsule correction refines close decisions without allowing residual signals to dominate the predict
    
[^304]: 测试时推测

    Test-Time Speculation

    [https://arxiv.org/abs/2605.09329](https://arxiv.org/abs/2605.09329)

    该论文提出测试时推测，通过在线蒸馏在推理阶段持续自适应调整推测器，解决了推测解码因训练与推理序列长度分布不匹配而在长输出任务中接受长度退化、加速失效的问题。

    

    推测解码通过使用一个快速的草稿模型来生成token，再由更精确的目标模型进行验证，从而加速大语言模型（LLM）的推理。其性能取决于接受长度，即目标模型所接受的草稿token数量。我们的研究表明，即使是当前最先进的推测器（如DFlash、EAGLE-3和PARD），其接受长度也会随着生成长度的增加而退化，在几千个输出token内降至接近1（即没有加速效果），这使得推测器在长响应任务中失去效力。接受长度下降的原因在于，大多数推测器是在短序列上进行离线训练的，但在推理时却被迫在远超其训练分布的长输出上匹配目标模型。为解决这一问题，我们提出了测试时推测，这是一种在测试阶段持续自适应调整推测器的在线蒸馏方法。TTS利用了一个关键洞察，即…

    arXiv:2605.09329v3 Announce Type: replace  Abstract: Speculative decoding accelerates LLM inference by using a fast draft model to generate tokens and a more accurate target model to verify them. Its performance depends on the $\textit{acceptance length}$, or number of draft tokens accepted by the target. Our studies show that the acceptance length of even state-of-the-art speculators, like DFlash, EAGLE-3, and PARD, degrades with generation length, reaching values close to 1 (i.e., no speedup) within a few thousand output tokens, making speculators ineffective for long-response tasks. Acceptance lengths decline because most speculators are trained offline on short sequences, but are forced to match the target on much longer outputs at inference, well beyond their training distribution.   To address this issue, we propose $\textit{Test-Time Speculation (TTS)}$, an online distillation approach that continuously adapts the speculator at test time. TTS leverages the key insight that the t
    
[^305]: SimCT：为跨分词器在线策略蒸馏恢复丢失的监督信号

    SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation

    [https://arxiv.org/abs/2605.07711](https://arxiv.org/abs/2605.07711)

    SimCT通过在共享token匹配之外引入两个分词器均可实现的短多token延续作为监督单元，在不改变OPD损失形式的前提下，恢复了跨分词器在线策略蒸馏中因词表不一致而丢失的教师信号。

    

    在线策略蒸馏（OPD）是将教师模型行为迁移到更小的学生模型的标准工具，但它隐含地假设教师和学生的预测可以逐个token进行比较，这一假设在两个模型对同一文本采用不同分词方式时会失效。在异构分词器的情况下，精确的共享token匹配会在词表不一致的位置默默地丢弃大部分教师信号。我们提出了SimCT（简单跨分词器在线策略蒸馏），通过扩大监督空间来恢复这一信号：除了共享token之外，SimCT还在两个分词器都能实现的短多token延续序列上比较教师和学生，同时保持OPD损失函数形式本身不变。我们证明这些单元是最细粒度的可联合分词监督接口，而更粗粒度的替代方案会消除有价值的教师-学生区分（摘要在此处截断）

    arXiv:2605.07711v3 Announce Type: replace  Abstract: On-policy distillation (OPD) is a standard tool for transferring teacher behavior to a smaller student, but it implicitly assumes that teacher and student predictions are comparable token by token, an assumption that fails whenever the two models tokenize the same text differently. Under heterogeneous tokenizers, exact shared-token matching silently discards a large fraction of the teacher signal at precisely the positions where vocabularies disagree. We propose \textbf{\underline{Sim}ple \underline{C}ross-\underline{T}okenizer OPD (SimCT)}, which restores this signal by enlarging the supervision space: alongside shared tokens, SimCT compares teacher and student over short multi-token continuations that both tokenizers can realize, leaving the OPD loss form itself unchanged. We show that these units are the finest jointly tokenizable supervision interface, and that coarser alternatives remove teacher-student distinctions that are use
    
[^306]: 元认知画像图谱：33个前沿大语言模型在MMLU上的领域级信心辨别能力

    The Metacognitive Profile Atlas: Domain-level confidence discrimination in 33 frontier LLMs on MMLU

    [https://arxiv.org/abs/2605.06673](https://arxiv.org/abs/2605.06673)

    该研究发布了元认知画像图谱数据集，揭示33个前沿大语言模型的元认知信心辨别能力存在系统性领域差异——在应用/专业类题目上最强，在形式推理和自然科学上最弱，且这一排序在群体层面稳定可靠。

    

    我们发布了元认知画像图谱，这是一个包含33个前沿大语言模型在1500个MMLU题目上的“答案加言语化信心”数据语料库，题目按六个学科组分层（剔除解析器提供的数值后，共有46,157个有效评分观测）。这些数据使得Type-2 AUROC——即信心区分正确与错误答案的程度——能够按领域进行分解。可靠的发现是MMLU上的群体层面排序：信心辨别能力在应用/专业类题目上表现最好（平均AUROC为0.750；在100%的自助重抽样中排名第一），在形式推理和自然科学类题目上表现最差（分别有93%和92%的重抽样中排名垫底两位；Friedman chi²(5) = 28.26, p < .0001）。经Holm校正后，应用类显著超过形式推理、科学、社会科学和人文类；没有其他配对差异显著，包括应用类与事实类之间的比较。当排除或降低不精确单元格的权重时，该排序依然成立。各模型单独的六领域画像并不可靠（中位数……

    arXiv:2605.06673v2 Announce Type: replace-cross  Abstract: We release the Metacognitive Profile Atlas, a corpus of answer-plus-verbalized-confidence data from 33 frontier LLMs on 1,500 MMLU items stratified into six subject-group bins (46,157 scored observations after excluding parser-supplied values). The data allow Type-2 AUROC, the degree to which confidence discriminates correct from incorrect answers, to be decomposed by domain. The reliable finding is a population-level ordering on MMLU. Confidence discriminates best on Applied/Professional items (mean AUROC .750; ranked first in 100% of bootstrap resamples) and worst on Formal Reasoning and Natural Science (bottom two in 93% and 92%; Friedman chi^2(5) = 28.26, p < .0001). After Holm correction Applied exceeds Formal, Science, Social and Humanities; no other pair survives, including Applied versus Factual. The ordering holds when imprecise cells are excluded or down-weighted. Per-model six-domain profiles are not reliable (median
    
[^307]: 极限中的对比识别与生成

    Contrastive Identification and Generation in the Limit

    [https://arxiv.org/abs/2605.06211](https://arxiv.org/abs/2605.06211)

    本文提出基于对比关系信号（即不相似样本对）的极限识别与生成新理论，精确刻画了在有限破坏预算下假设类可识别的条件，并证明无界预算会使任意两个假设都变得不可区分。

    

    在极限识别中（Gold, 1967），学习者读取正例样本并必须最终说出目标假设；在极限生成中（Kleinberg和Mullainathan, 2024），学习者必须最终产出新的正例样本。我们研究一种更弱的、关系性的信号：每个观测是一个无序对 $\{x,y\}$，满足 $h(x)\neq h(y)$，例如cannot-link约束或不相似对标签，而学习者永远不知道哪个点是正例。我们的所有结果都源自一个对象：当且仅当一个样本对的两个点位于某个有限族隶属映射的对极单元中时，该样本对对该族的每个成员都是有效的。（1）对于每个破坏预算 $k$（无论已知还是未知），我们精确刻画了可识别的类。不可比较的假设必须覆盖整个域并共享超过 $2k$ 个点，且Angluin的示警集必须在丢失 $2k$ 个点后仍然存活。任何无界的预算都会使任意两个假设变得不可区分。

    arXiv:2605.06211v2 Announce Type: replace-cross  Abstract: In identification in the limit (Gold, 1967), a learner reads positive examples and must eventually name the target hypothesis; in generation in the limit (Kleinberg and Mullainathan, 2024), it must eventually produce new positive examples. We study a weaker, relational signal: each observation is an unordered pair $\{x,y\}$ with $h(x)\neq h(y)$, such as a cannot-link constraint or a dissimilar-pair label, and the learner never learns which point is positive. All our results come from one object: a pair is valid for every member of a finite family exactly when its points lie in antipodal cells of the family's membership map. (1) For every corruption budget $k$, known or unknown, we characterize exactly the identifiable classes. Incomparable hypotheses must cover the domain and share more than $2k$ points, and Angluin's tell-tale sets must survive the loss of $2k$ points. Every unbounded budget makes any two hypotheses confusable
    
[^308]: 粒度轴：语言模型中社会角色的微观到宏观潜在方向

    The Granularity Axis: A Micro-to-Macro Latent Direction for Social Roles in Language Models

    [https://arxiv.org/abs/2605.06196](https://arxiv.org/abs/2605.06196)

    本文发现语言模型表征空间中存在一条“粒度轴”——宏观角色与微观角色隐藏状态均值的差异方向，该方向与角色空间的第一主成分高度对齐，并跨模型、角色层级、网络层和提示词稳定存在，证明模型内部编码了社会角色的粒度差异。

    

    大语言模型（LLM）常被提示采纳从个人到机构等各种社会角色，但其内部表征是否编码了这种社会粒度上的差异仍不清楚。我们发现了一条基于对比的“粒度轴”，即宏观角色与微观角色隐藏状态均值之间的差异方向，该轴与角色空间的主导几何结构高度对齐：在Qwen3-8B中，它与第一主成分（PC1）的余弦相似度为0.972，且PC1解释了52.6%的方差。在涵盖五个层级共75个角色的实验中，每个模型包含90,000条角色条件响应和1,200条默认响应，投影沿粒度单调递增，并在不同网络层、提示词及响应质量过滤条件下保持稳定。该结构可迁移至Llama-3.1-8B-Instruct，并可扩展至Qwen3-32B和Llama-3.1-70B-Instruct，后两者的PC1方差占比分别为45.99%和31.23%。独立的五人评分验证（α=0.961）保持了这一排序；相同续写实验以及情感与正式度控制实验证实了该轴的尺度特异性。

    arXiv:2605.06196v2 Announce Type: replace  Abstract: Large language models (LLMs) are prompted to adopt social roles from individuals to institutions, but whether their representations encode this difference in social granularity remains unclear. We discover a contrast-based Granularity Axis, the difference between mean macro- and micro-role hidden states, that aligns with the dominant geometry of role space: in Qwen3-8B, its cosine with PC1 is $0.972$, and PC1 explains $52.6\%$ of the variance. Across 75 roles at five levels, with 90,000 role-conditioned and 1,200 default responses per model, projections increase monotonically and remain stable across layers, prompts, and response-quality filters. The structure transfers to Llama-3.1-8B-Instruct and scales to Qwen3-32B and Llama-3.1-70B-Instruct, whose PC1 shares are $45.99\%$ and $31.23\%$. Independent five-rater validation ($\alpha=0.961$) preserves the ordering; identical continuations and affect/formality controls confirm scale-re
    
[^309]: 首个词元即知晓：用于幻觉检测的单次解码置信度

    The First Token Knows: Single-Decode Confidence for Hallucination Detection

    [https://arxiv.org/abs/2605.05166](https://arxiv.org/abs/2605.05166)

    该论文提出首个词元置信度 phi_first，仅需单次贪心解码即可从 top-K 首词元分布中估计不确定性，其幻觉检测效果媲美甚至略超需要十次采样生成加 NLI 语义聚类的语义自一致性方法，大幅降低计算成本。

    

    基于采样的不确定性方法通过生成多个答案并衡量其一致性来检测幻觉，通常还需额外的语义聚类步骤。我们研究不确定性是否已经在单次贪心解码产生的首个承载内容的答案词元中显现。我们定义首个词元置信度 phi_first 为 1 减去 top-K 首词元分布的归一化熵。在 PopQA 和 TriviaQA 数据集及三个 7-8B 指令微调模型上的实验表明，phi_first 能够匹配甚至略微超越语义自一致性方法的表现，而仅需一次解码，无需一次贪心解码外加十次采样生成以及 NLI 语义聚类。包含性分析显示，phi_first 与语义一致性呈中等到强相关，将两种信号结合仅带来很小的额外 AUROC 提升。我们将 phi_first 定位为一种面向实际部署的、低成本的事实性问答不确定性估计基线方法，同时指出了其局限性。

    arXiv:2605.05166v2 Announce Type: replace-cross  Abstract: Sampling-based uncertainty methods detect hallucinations by generating multiple answers and measuring agreement, often with additional semantic clustering. We study whether uncertainty is already visible in the first content-bearing answer token of a single greedy decode. We define first-token confidence, phi_first, as one minus the normalized entropy of the top-K first-token distribution. Across PopQA and TriviaQA with three 7-8B instruction-tuned models, phi_first matches or modestly exceeds semantic self-consistency while requiring only one decode instead of one greedy decode plus ten sampled generations and NLI clustering. A subsumption analysis shows that phi_first is moderately to strongly correlated with semantic agreement, and combining both signals yields only small additional AUROC gains. We position phi_first as a cheap deployment-oriented baseline for factual QA uncertainty estimation, while noting its limits for hi
    
[^310]: 基于模糊指纹的编码器预训练语言模型用于对话情绪识别：人类评估与有效性研究

    Fuzzy Fingerprinting Encoder Pre-trained Language Models for Emotion Recognition in Conversations: Human Assessment and Validity Study

    [https://arxiv.org/abs/2605.02665](https://arxiv.org/abs/2605.02665)

    该论文提出了一种将预训练语言模型与模糊指纹相结合的新颖可解释方法，通过类别特定原型揭示模型潜在空间中的特征激活模式，从而提升对话情绪识别的可解释性，并解决不平衡数据集中少数情绪被误分类为中性类别的问题。

    

    在对话情绪识别（ERC）任务中，模型的决策应与细微的人类感知保持一致，并理想地为分类过程提供可解释的洞察。标准编码器预训练语言模型（PLMs）在这些任务上是当前最先进的技术，但对于为何做出某个特定预测几乎不提供任何洞察。这在数据集不平衡的情况下尤其成问题，因为其中大多数话语被标记为中性，导致这些模型经常将少数情绪类别误分类为占多数的中性类别。为了解决这一问题，我们通过将PLMs与模糊指纹相结合，引入了一种新颖的、可解释的ERC方法。FFP提供了类别特定的原型，能够反映PLM潜在空间中各情绪类别的特征性激活模式。这些原型是通过对每个情绪类别在训练实例中的池化对话上下文相关嵌入的激活进行排序和模糊化而得到的。在推理时……（摘要在此处被截断）

    arXiv:2605.02665v2 Announce Type: replace-cross  Abstract: In Emotion Recognition in Conversations (ERC), model decisions should align with nuanced human perception and ideally provide insights on the classification process. Standard encoder pre-trained language models (PLMs) are the state-of-the-art at these tasks but offer little insight into why a certain prediction is made. This is especially problematic in imbalanced datasets, where most utterances are labeled as neutral, making these models frequently misclassify minority emotions as the majority neutral class. To tackle this issue, we introduced a novel, interpretable approach to ERC by combining PLMs with Fuzzy Fingerprints (FFPs). FFP provide class-specific prototypes that reflect the characteristic class activation patterns in the PLM's latent space. They are derived by ranking and fuzzifying the activations of the pooled conversational context-dependent embeddings across training instances for each emotion. At inference time
    
[^311]: 眼见不再为实：前沿图像生成模型、合成视觉证据与现实世界风险

    Seeing Is No Longer Believing: Frontier Image Generation Models, Synthetic Visual Evidence, and Real-World Risk

    [https://arxiv.org/abs/2604.24197](https://arxiv.org/abs/2604.24197)

    本文是一篇叙述性综述，系统梳理了前沿图像生成模型的高逼真合成能力如何使合成图像获得“证据权威”，从而对新闻、金融、身份验证、医疗和法律等现实领域构成风险，并区分了厂商能力声明、已记录事件与潜在危害路径。

    

    图像生成系统能够产出逼真的照片、可读的文档，以及对人物和地点的一致性描绘。当这些产物被当作真实事件的记录呈现时，它们可能影响新闻、金融、身份验证、医学和法律领域的决策。本叙述性综述考察了截至2026年10月1日可获取的选定公开模型文档、事件报告、研究和治理资料，其中英文和中文社区材料提供了说明性背景。我们区分了厂商能力声明、已记录的滥用事件、实验发现以及潜在的危害路径。该分析将真实感、文本渲染、参考一致性、图像编辑、事实接地和生产成本等因素，与合成图像获得证据权威的条件联系起来。历史事件只是阐释了这些危害路径，并不能确定当前模型的实际滥用率。我们比较了供应商的限制措施……（摘要至此处截断）

    arXiv:2604.24197v3 Announce Type: replace  Abstract: Image generation systems can produce plausible photographs, readable documents, and consistent depictions of people and places. When these artifacts are presented as records of real events, they can influence decisions in news, finance, identity verification, medicine, and law. This narrative review examines selected public model documentation, incident reports, research, and governance sources available through 1 October 2026, with English and Chinese community material providing illustrative context. We distinguish vendor capability claims, documented incidents, experimental findings, and prospective harm pathways. The analysis connects realism, text rendering, reference consistency, editing, grounding, and production cost to the conditions under which synthetic images acquire evidentiary authority. Historical incidents illustrate these pathways; they do not establish misuse rates for current models. We compare provider restriction
    
[^312]: 前沿语言模型中的口头禅：对当前发布版本、研究证据与公众讨论的批判性综述

    Verbal tics in frontier language models: A critical review of current releases, research evidence, and public discussion

    [https://arxiv.org/abs/2604.19139](https://arxiv.org/abs/2604.19139)

    这篇批判性综述系统考察了OpenAI、Anthropic、Google DeepMind等八家开发商前沿语言模型中的口头禅式语言习惯与谄媚附和现象，综合研究证据与公众讨论，指出附和性语言会扭曲判断，而对最新发布版本的可比测量仍然稀缺。

    

    重复的赞美、程式化的安抚、常见的对比句式以及显眼的词汇，是大语言模型讨论中反复出现的话题。对它们的解读依赖于语境：一个惯用短语可能是有用的，而一段流畅的回答可能会强化错误信念。这篇批判性综述考察了八个开发商系列的模型的语言习惯与谄媚行为：OpenAI、Anthropic、Google DeepMind、xAI、字节跳动、月之暗面（Moonshot AI）、DeepSeek和小米。我们依据官方发布信息和API文档核实了当前的公开产品，证据截止日期为2026年10月1日。我们综合了关于词汇过度表征、文体变化、社交温暖度和迎合附和的研究，以及基准测试方法和带有时间标注的中英文公众讨论。所综述的研究记录了反复出现的语言模式以及扭曲判断的附和现象；而针对最新版本的可比测量在检索到的文献中较为稀少。

    arXiv:2604.19139v4 Announce Type: replace-cross  Abstract: Repeated praise, canned reassurance, familiar contrasts, and conspicuous vocabulary are recurring subjects in discussions of large language models. Their interpretation depends on context: a conventional phrase may be useful, while a fluent answer may reinforce a false belief. This critical review examines linguistic habits and sycophancy across eight developer families: OpenAI, Anthropic, Google DeepMind, xAI, ByteDance, Moonshot AI, DeepSeek, and Xiaomi. We verify current public offerings against official release and API documentation, with an evidence cutoff of 1 October 2026. We synthesize research on lexical overrepresentation, stylistic variation, social warmth, and agreement, alongside benchmark methods and dated English and Chinese public discussions. The research reviewed documents recurring linguistic patterns and agreement that distorts judgment; comparable measurements of the newest releases are sparse in the retrie
    
[^313]: 看见却未思考：多模态混合专家模型中的路由分心现象

    Seeing but Not Thinking: Routing Distraction in Multimodal Mixture-of-Experts

    [https://arxiv.org/abs/2604.08541](https://arxiv.org/abs/2604.08541)

    本文发现多模态MoE模型存在“看见却未思考”现象——模型能准确感知图像却推理失败，并提出“路由分心”假设：视觉输入使路由机制偏离、未能充分激活任务相关的推理专家，从而解释了感知与推理之间的断裂。

    

    多模态混合专家（MoE）模型在视觉-语言任务上取得了卓越的性能。然而，我们发现了一种令人困惑的现象，称之为“看见却未思考”：模型能够准确感知图像内容，却在后续推理中失败，而面对以纯文本形式呈现的相同问题却能正确解答。通过系统性分析，我们首先验证了MoE架构中存在跨模态语义共享，从而排除了语义对齐失败是唯一解释的可能性。随后，我们揭示了视觉专家与领域专家在层级上呈现分离分布，图像输入会在领域专家集中的中间层引发与文本输入显著不同的路由分歧。基于这些发现，我们提出了“路由分心”假设：在处理视觉输入时，路由机制未能充分激活与任务相关的推理专家。为验证该假设，我们设计了一种路由（摘要在此处截断）

    arXiv:2604.08541v2 Announce Type: replace-cross  Abstract: Multimodal Mixture-of-Experts (MoE) models have achieved remarkable performance on vision-language tasks. However, we identify a puzzling phenomenon termed Seeing but Not Thinking: models accurately perceive image content yet fail in subsequent reasoning, while correctly solving identical problems presented as pure text. Through systematic analysis, we first verify that cross-modal semantic sharing exists in MoE architectures, ruling out semantic alignment failure as the sole explanation. We then reveal that visual experts and domain experts exhibit layer-wise separation, with image inputs inducing significant routing divergence from text inputs in middle layers where domain experts concentrate. Based on these findings, we propose the Routing Distraction hypothesis: when processing visual inputs, the routing mechanism fails to adequately activate task-relevant reasoning experts. To validate this hypothesis, we design a routing-
    
[^314]: ORACLE-SWE：量化关键信息信号对软件工程代理的贡献

    ORACLE-SWE: Quantifying the Contribution of Critical Information Signals on SWE Agents

    [https://arxiv.org/abs/2604.07789](https://arxiv.org/abs/2604.07789)

    该论文提出ORACLE-SWE，首次系统量化了复现测试、回归测试、编辑位置、执行上下文和API用法等关键中间信息信号对软件工程代理任务成功的贡献，并测量了代理发现的信号与理想神谕信号之间尚存的改进空间。

    

    语言模型（LM）代理的最新进展日益聚焦于仓库级软件工程（SWE）任务，由此产生了多种代理式工作流、训练方法和失败分析技术。有趣的是，这些方法并非直接生成或评估编辑补丁，而是首先检查并改进在代理探索过程中收集到的中间信息信号，包括复现测试、回归测试、编辑位置、执行上下文和API用法。尽管不同的研究考察了这些信号的不同子集，但目前尚无工作系统地比较它们在提升最终任务成功率方面的相对有效性，也没有测量每个信号中还剩余多大的改进空间。为填补这一空白，本工作量化了“神谕”信号与代理提取的信号对最终任务成功的贡献，并测量了当前代理所发现的信号与其对应神谕信号之间的差距。

    arXiv:2604.07789v3 Announce Type: replace-cross  Abstract: Recent advances in Language Model (LM) agents have increasingly focused on repository-level software engineering (SWE) tasks, leading to various agentic workflows, training methods and failure analysis. Interestingly, rather than directly generating or evaluating edit patches, these approaches first examine and improve intermediate information signals collected during agent exploration, including Reproduction Test, Regression Test, Edit Location, Execution Context, and API Usage. Although different studies examine different subsets of these signals, no work has systematically compared their relative effectiveness in improving final task success or measured the remaining space for improvement in each signal. To address this gap, this work quantifies the contributions of both oracle and agent-extracted signals to final task success and measures the gap between current agent-found signals and their oracle counterparts. We hope our
    
[^315]: Appear2Meaning：一个用于从图像推断结构化文化元数据的跨文化基准

    Appear2Meaning: A Cross-Cultural Benchmark for Structured Cultural Metadata Inference from Images

    [https://arxiv.org/abs/2604.07338](https://arxiv.org/abs/2604.07338)

    提出了包含750件博物馆物品的跨文化基准Appear2Meaning，用于评估视觉语言模型仅凭图像推断标题、文化、时期、产地和创作者等结构化文化元数据的能力，结果显示当前模型的完整记录兼容率仍然很低。

    

    博物馆记录通过文化和历史关系表达了人们如何理解遗产物品。从图像中恢复这些人工赋予的归属信息，对视觉语言模型（VLM）来说是一项技术挑战。我们提出了Appear2Meaning，这是一个包含750件博物馆物品的跨文化基准数据集，用于仅凭图像恢复标题、文化、时期、产地和创作者五类信息。其评估框架区分了字段级别的兼容性与完整记录的兼容性，并采用人工验证进行语义判断。在九个VLM模型上，平均字段准确率为36.16%–49.82%；7.20%–17.02%的记录在所有四个上下文字段上均兼容，而包含标题在内的全部五个字段均兼容的记录仅占2.13%–7.87%。对于测试的两个模型，仅图像输入在文化、时期和产地方面的参考兼容性比仅使用博物馆名称和标题的对照组高出12.6–32.2个百分点。

    arXiv:2604.07338v3 Announce Type: replace-cross  Abstract: Museum records express how people understand heritage objects through cultural and historical relations. Recovering these human attributions from images is a technical challenge for vision-language models (VLMs). We introduce Appear2Meaning, a cross-cultural benchmark of 750 museum objects for image-only recovery of title, culture, period, origin, and creator. Its evaluation framework distinguishes field-level from complete-record compatibility and uses human validation for semantic judgement. Across nine VLMs, mean field accuracy is 36.16--49.82%; 7.20--17.02% of records have all four contextual fields compatible, and 2.13--7.87% have all five, including title. For the two tested models, image-only input yields 12.6--32.2 percentage points higher reference compatibility for culture, period, and origin than a museum-and-title-only control.
    
[^316]: 论语言识别与生成中隐私保护的代价

    On the Price of Privacy for Language Identification and Generation

    [https://arxiv.org/abs/2604.07238](https://arxiv.org/abs/2604.07238)

    该论文揭示了差分隐私在语言学习中的代价结构：语言识别中隐私几乎免费，而语言生成中代价取决于公开信息——有公开见证界时纯DP仅损失Θ(min{1,ε})的指数因子，无此界时纯DP则会彻底丧失指数收敛速率。

    

    大型语言模型在敏感的用户数据上进行训练，因此我们探讨差分隐私（DP）在语言学习中的代价。我们在不可知统计设置下研究隐私保护的语言识别与生成问题。对于识别任务，隐私是免费的：在每个固定的 ε 下，纯 DP 以及 δ ≥ n^{-O(1)} 的 (ε,δ)-DP 都能对任意 r(n)=o(n) 保持非隐私速率 exp(-r(n))。对于生成任务，代价取决于哪些信息是公开的。当存在公开的见证界时，纯 DP 在指数上至多损失因子 Θ(min{1,ε})，并且在适当的双语言集合上，最优速率为 exp(-Θ(min{1,ε}·n))。而当没有这样的界时，纯 DP 可能完全失去指数速率，即便仅有两种语言也是如此：所有 exp(-o(n)) 量级的速率仍然可达，但任何指数速率都不可达。在这些集合上，(ε,δ)-DP 能保持指数速率当且仅当 log(1/δ) 满足相应条件（摘要在此处截断）。

    arXiv:2604.07238v2 Announce Type: replace-cross  Abstract: Large language models train on sensitive user data, so we ask what differential privacy (DP) costs in language learning. We study private language identification and generation in the agnostic statistical setting. For identification, privacy is free: at every fixed $\varepsilon$, pure DP and $(\varepsilon,\delta)$-DP with $\delta\ge n^{-O(1)}$ keep the non-private rate $\exp(-r(n))$ for every $r(n)=o(n)$. For generation, the price depends on what is public. With a public witness bound, pure DP loses at most a factor $\Theta(\min\{1,\varepsilon\})$ in the exponent, and on suitable two-language collections the optimal rate is $\exp(-\Theta(\min\{1,\varepsilon\}\,n))$. Without such a bound, pure DP can lose the exponential rate itself, already for two languages: every rate $\exp(-o(n))$ stays achievable, but no exponential rate is. On these collections, $(\varepsilon,\delta)$-DP keeps an exponential rate if and only if $\log(1/\de
    
[^317]: 一种通过多任务提示蒸馏与分解实现参数高效迁移学习的临床自然语言处理方法

    A Parameter-Efficient Transfer Learning Approach through Multitask Prompt Distillation and Decomposition for Clinical NLP

    [https://arxiv.org/abs/2604.06650](https://arxiv.org/abs/2604.06650)

    提出多任务提示蒸馏与分解框架，从21个临床源任务中学习单一共享元提示，仅用不足0.05%的可训练参数即可适配新任务，以远少于LoRA的参数量在10个临床NLP数据集上取得更优性能。

    

    现有的基于提示的微调方法通常独立学习各任务专属的提示，在规模化部署多个临床自然语言处理（NLP）系统时会带来巨大的计算与存储开销。我们提出了一个多任务提示蒸馏与分解框架，从21个多样化的临床源任务中学习单一共享的元提示（metaprompt），并将其适配到未见过的目标任务上，可训练参数不足0.05%。在使用三个骨干模型（LLaMA 3.1 8B、Meditron3 8B、gpt-oss 20B）于10个留出的目标数据集上，对五种临床NLP任务类型（命名实体识别、关系抽取、问答、自然语言推理和文本摘要）进行评估，尽管参数量比LoRA少几个数量级，我们的框架仍持续超越LoRA 1.5~1.7%，并超过单任务提示调优6.1~6.6%。其中gpt-oss 20B模型取得了最高的整体性能。

    arXiv:2604.06650v2 Announce Type: replace-cross  Abstract: Existing prompt-based fine-tuning methods typically learn task-specific prompts independently, imposing significant computing and storage overhead at scale when deploying multiple clinical natural language processing (NLP) systems. We present a multitask prompt distillation and decomposition framework that learns a single shared metaprompt from 21 diverse clinical source tasks and adapts it to unseen target tasks with fewer than 0.05% trainable parameters. Evaluated across five clinical NLP task types (named entity recognition, relation extraction, question answering, natural language inference, and summarization) on 10 held-out target datasets using three backbone models (LLaMA 3.1 8B, Meditron3 8B, gpt-oss 20B), our framework consistently outperforms LoRA by 1.5~1.7% despite using orders of magnitude fewer parameters, and exceeds single-task prompt tuning by 6.1~6.6%. The gpt-oss 20B model achieves the highest overall perform
    
[^318]: 支持人机协作中的校准依赖：不同任务采用不同策略

    Supporting Calibrated Reliance in Human-AI Collaboration: Different Strategies for Different Tasks

    [https://arxiv.org/abs/2604.03237](https://arxiv.org/abs/2604.03237)

    本文通过实验发现，AI支持形式（如预测、解释、概率）对人机协作中的校准依赖影响因任务类型而异，视觉推理中预测概率最有效，而解释并非总是提升性能。

    

    随着人工智能系统越来越多地支持人类决策，一个核心挑战是确定哪些信息能帮助人们认识到何时应依赖AI预测，以及何时应质疑或推翻它们。通过三项受控的人类受试者研究，涵盖使用RAVEN矩阵的抽象视觉推理和使用LSAT问题的演绎逻辑推理，我们考察了不同形式的AI支持如何影响人机团队的表现。一项多阶段逐步揭示的研究表明，AI预测和解释可以分别影响客观准确性和主观信心。在视觉推理中，大语言模型（LLM）的解释并未在仅提供预测答案的基础上提升准确性，且没有额外的支持格式显著优于仅预测支持；预测概率在描述性准确性和错误恢复方面表现最佳，而一种衍生的选择性自动化策略则提供了更高性能的参考基准。在语言任务中，我们发现不同的支持策略对校准依赖具有差异化影响，强调了任务类型在决定有效支持方式中的关键作用。

    arXiv:2604.03237v2 Announce Type: replace-cross  Abstract: As AI systems increasingly support human decision making, a central challenge is determining what information helps people recognize when to rely on AI predictions and when to question or override them. Across three controlled human-subject studies spanning abstract visual reasoning with RAVEN matrices and deductive logical reasoning with LSAT problems, we examine how different forms of AI support affect human--AI team performance. A multi-stage reveal study shows that AI predictions and explanations can affect objective accuracy and subjective confidence differently. In visual reasoning, LLM explanations do not improve accuracy beyond the predicted answer alone, and no additional support format significantly outperforms prediction-only support; predicted probabilities show the highest descriptive accuracy and error recovery, while a derived selective-automation policy provides a higher-performing reference benchmark. In langua
    
[^319]: 基于智能体技能挖掘的代码效率自动优化

    Automated Code Efficiency Optimization via Agent Skill Mining

    [https://arxiv.org/abs/2603.27850](https://arxiv.org/abs/2603.27850)

    本文提出EffiSkill两阶段框架，将慢代码到快代码的常见转换挖掘并沉淀为记录意图、适用性、流程、效果与风险的“算子技能”卡片库，使LLM无需执行反馈即可诊断程序、检索技能并生成多个候选优化方案，从而自动提升代码效率。

    

    代码效率是软件质量的一个基本方面，然而如何利用大语言模型（LLM）来优化已经正确的程序仍然具有挑战性。先前的方法依赖于一次性重写、检索示例或指令、微调或候选搜索等手段。在本文中，我们提出了EffiSkill，这是一个两阶段框架，它将反复出现的“慢代码到快代码”转换建模为算子技能：即持久化的卡片，其中记录了每次转换的意图、适用条件、操作流程、预期效果和风险。阶段一从慢/快程序对中提炼出冻结的、针对特定编程语言的技能库。阶段二对新的程序进行诊断，检索相关技能，组合出多种备选优化方案，并生成最多八个候选程序。这些步骤的完成无需候选程序的执行反馈；随后使用公开测试来过滤和排序候选程序，再用留出的私有测试来确定所报告的正确性和加速比。在每种语言623个EffiBench-X任务上……（原文在此处截断）

    arXiv:2603.27850v2 Announce Type: replace-cross  Abstract: Code efficiency is a fundamental aspect of software quality, yet how to use large language models (LLMs) to optimize correct programs remains challenging. Prior approaches rely on one-shot rewriting, retrieved exemplars or instructions, fine-tuning, or candidate search. In this paper, we present EffiSkill, a two-stage framework that models recurring slow-to-fast transformations as Operator Skills: persistent cards that record a transformation's intent, applicability, procedure, expected effect, and risks. Stage I distills frozen, language-specific skill libraries from slow/fast program pairs. Stage II diagnoses a new program, retrieves relevant skills, composes alternative plans, and generates up to eight candidates. These steps finish without candidate-execution feedback; public tests then filter and rank candidates, and held-out private tests determine reported correctness and speedup. Across 623 EffiBench-X tasks per languag
    
[^320]: CAF-Score：利用大型音频-语言模型校准CLAP以实现无参考的音频-文本对齐评估

    CAF-Score: Calibrating CLAP with LALMs for Reference-free Audio-Text Alignment Evaluation

    [https://arxiv.org/abs/2603.19615](https://arxiv.org/abs/2603.19615)

    CAF-Score是一种无参考的音频-文本对齐评估指标，它将CLAP的粗粒度语义对齐与大型音频-语言模型（LALM）的细粒度理解和句法感知能力校准融合，凭借对称评分机制，单一配置即可同时适用于音频描述和文本到音频生成两个评估方向，并在三个人工标注基准上取得与人类判断更高的一致性。

    

    尽管大型音频-语言模型（LALMs）已经推动了音频理解与生成的发展，但如何稳健地评估音频-文本对齐仍然十分困难：基于参考的指标需要昂贵的标注，而基于对比语言-音频预训练（CLAP）的无参考指标则经常忽略句法错误和细粒度细节。我们提出了CAF-Score，这是一种无参考指标，它将CLAP的粗粒度语义对齐能力与LALMs的细粒度理解和句法感知能力相结合进行校准。由于CAF-Score以对称的方式对音频-文本对进行评分，因此一种配置即可适用于两个对齐方向：音频描述评估（BRACE基准）以及文本到音频（TTA）生成评估（RELATE和PAM基准），其中单一的融合权重只需在BRACE上选择一次，即可无需重新调参地迁移到其他基准。在所有三个人工标注的基准上，CAF-Score都提高了与人类判断的一致性。

    arXiv:2603.19615v2 Announce Type: replace-cross  Abstract: While Large Audio-Language Models (LALMs) have advanced audio understanding and generation, robustly evaluating audio-text alignment remains difficult: reference-based metrics require costly annotations, while Contrastive Language-Audio Pretraining (CLAP)-based reference-free metrics frequently overlook syntactic errors and fine-grained details. We propose CAF-Score, a reference-free metric that calibrates CLAP's coarse-grained semantic alignment with the fine-grained comprehension and syntactic awareness of LALMs. Because CAF-Score scores an audio-text pair symmetrically, one configuration applies to both alignment directions: audio captioning evaluation (the BRACE benchmark) and text-to-audio (TTA) generation evaluation (the RELATE and PAM benchmarks), with a single fusion weight selected once on BRACE and transferred without re-tuning. Across all three human-annotated benchmarks, CAF-Score improves agreement with human judgm
    
[^321]: TERMINATOR：为思维链推理中的早停学习最优退出点

    TERMINATOR: Learning Optimal Exit Points for Early Stopping in Chain-of-Thought Reasoning

    [https://arxiv.org/abs/2603.12529](https://arxiv.org/abs/2603.12529)

    提出了Terminator，一种针对大型推理模型的推理时提前退出策略，通过预测最终答案首次出现的位置来学习最优退出点，从而在几乎不损失性能的前提下缓解思维链推理中的过度思考并大幅降低计算开销。

    

    大型推理模型通过思维链推理在复杂推理任务上取得了令人瞩目的性能，这使其能够在得出最终答案之前生成中间思考标记。然而，大型推理模型常常存在显著的过度思考问题——即使答案早已生成，它们仍会花费过多的计算时间。先前的研究已经发现最优推理长度的存在，在该点截断推理可以显著缩短CoT输出，而性能几乎没有任何变化。然而，为实际数据集确定最优CoT长度绝非易事，因为它完全依赖于具体任务和模型。在本文中，我们精确地解决了这一问题，并设计了Terminator，一种针对大型推理模型的推理时提前退出策略，用以缓解过度思考。支撑Terminator的核心思想是：大型推理模型最终答案的首次出现往往是可预测的，我们利用……（原文摘要至此截断）

    arXiv:2603.12529v3 Announce Type: replace-cross  Abstract: Large Reasoning Models (LRMs) achieve impressive performance on complex reasoning tasks via Chain-of-Thought (CoT) reasoning, which enables them to generate intermediate thinking tokens before arriving at the final answer. However, LRMs often suffer from significant overthinking, spending excessive compute time even after the answer is generated early on. Prior work has identified the existence of an optimal reasoning length such that truncating reasoning at this point significantly shortens CoT outputs with virtually no change in performance. However, determining optimal CoT lengths for practical datasets is highly non-trivial as they are fully task and model-dependent. In this paper, we precisely address this and design Terminator, an early-exit strategy for LRMs at inference to mitigate overthinking. The central idea underpinning Terminator is that the first arrival of an LRM's final answer is often predictable, and we lever
    
[^322]: 大语言模型的可控性如何？跨行为粒度的统一评估

    How Controllable Are Large Language Models? A Unified Evaluation across Behavioral Granularities

    [https://arxiv.org/abs/2603.02578](https://arxiv.org/abs/2603.02578)

    提出 SteerEval 分层基准，从语言特征、情感和人格三个领域、三个规范层级系统评估大语言模型的可控性，发现现有引导方法在细粒度层级上的控制能力显著退化。

    

    大语言模型（LLM）越来越多地被部署在对社会敏感的领域，然而其不可预测的行为——从意图偏离到人格不一致——带来了重大风险。我们提出了 SteerEval，这是一个用于评估 LLM 可控性的分层基准，涵盖三个领域：语言特征、情感和人格。每个领域被结构化为三个规范层级：L1（表达什么）、L2（如何表达）和 L3（如何实例化），将高层次的行为意图与具体的文本输出联系起来。借助 SteerEval，我们系统地评估了现有的引导方法，发现控制在更细粒度的层级上往往会退化。我们的基准为安全可控的 LLM 行为提供了一个有原则且可解释的框架，为未来研究奠定了基础。

    arXiv:2603.02578v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly deployed in socially sensitive domains, yet their unpredictable behaviors, ranging from misaligned intent to inconsistent personality, pose significant risks. We introduce SteerEval, a hierarchical benchmark for evaluating LLM controllability across three domains: language features, sentiment, and personality. Each domain is structured into three specification levels: L1 (what to express), L2 (how to express), and L3 (how to instantiate), connecting high-level behavioral intent to concrete textual output. Using SteerEval, we systematically evaluate contemporary steering methods, revealing that control often degrades at finer-grained levels. Our benchmark offers a principled and interpretable framework for safe and controllable LLM behavior, serving as a foundation for future research.
    
[^323]: 快速追赶，后期切换：基于函数缩放定律的最优批量大小调度

    Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws

    [https://arxiv.org/abs/2602.14208](https://arxiv.org/abs/2602.14208)

    该论文揭示了最优批量大小调度的结构取决于任务难度——简单任务应持续增大批量，而困难任务应保持小批量直至训练后期再切换到大批量，其背后的“快速追赶”动态机制同样在大语言模型预训练中存在。

    

    批量大小调度在大规模深度学习训练中起着至关重要的作用，同时影响优化动态和计算效率，然而其理论基础仍未被充分理解。在本工作中，我们证明Li等人（2025a）提出的函数缩放定律（FSL）框架为分析批量大小调度提供了一个有原则的研究视角。具体而言，我们刻画了固定数据预算下的最优批量大小调度，并表明其结构强烈依赖于任务难度。对于简单任务，最优调度会在整个训练过程中持续增大批量大小；相反，对于困难任务，最优调度在训练的大部分阶段保持较小的批量大小，仅在后期阶段才切换到大批量。为解释后期切换现象的出现，我们揭示了一种动态机制——快速追赶效应——该效应同样在大语言模型（LLM）预训练中显现。在从小批量切换到大批量之后，损失（摘要在此处截断）

    arXiv:2602.14208v2 Announce Type: cross  Abstract: Batch size scheduling (BSS) plays a critical role in large-scale deep learning training, influencing both optimization dynamics and computational efficiency. Yet, its theoretical foundations remain poorly understood. In this work, we show that the functional scaling law (FSL) framework introduced in Li et al. (2025a) provides a principled lens for analyzing BSS. Specifically, we characterize the optimal BSS under a fixed data budget and show that its structure depends sharply on task difficulty. For easy tasks, optimal schedules keep increasing batch size throughout. In contrast, for hard tasks, the optimal schedule maintains small batch sizes for most of training and switches to large batches only in a late stage. To explain the emergence of late switching, we uncover a dynamical mechanism -- the fast catch-up effect -- which also manifests in large language model (LLM) pretraining. After switching from small to large batches, the los
    
[^324]: WAXAL：大规模多语言非洲语言语音语料库

    WAXAL: A Large-Scale Multilingual African Language Speech Corpus

    [https://arxiv.org/abs/2602.02734](https://arxiv.org/abs/2602.02734)

    WAXAL是一个涵盖27种非洲语言、超过14,100小时的大规模开放语音数据集，包含ASR和TTS两部分语料，实验证明在其上微调可显著提升多个先进ASR模型的性能，从而弥合语音技术中非洲语言的资源差距。

    

    语音技术的最新进展主要偏向高资源语言，导致撒哈拉以南非洲语言的使用者得不到充分的服务。为弥合这一差距，我们推出了WAXAL，一个开放的大规模语音数据集，涵盖27种非洲语言，代表超过1亿使用者。WAXAL包含超过14,100小时的语音数据，这些数据由非洲学术和社区组织合作收集，旨在建设当地研究能力：其中包括一个自动语音识别（ASR）语料库，含2,340小时的带转录、由图像提示引发的自然语音，覆盖19种语言（另有约11,760小时未转录语音），以及一个录音棚质量的文本转语音（TTS）语料库，约247小时，涵盖15种语言配置。我们在严格的说话者不相交数据划分上，对三种架构截然不同的ASR模型——Gemma 3n 2B、Whisper-Large-v3和MMS-1B-all——在19种ASR语言上进行了实证基准测试。在WAXAL上进行微调带来了显著的性能提升。

    arXiv:2602.02734v4 Announce Type: replace-cross  Abstract: Recent advances in speech technology predominantly favor high-resource languages, leaving speakers of Sub-Saharan African languages underserved. To bridge this divide, we introduce WAXAL, an open, large-scale speech dataset spanning 27 African languages representing over 100 million speakers. WAXAL comprises over 14,100 hours of speech collected in partnership with African academic and community organizations to build local research capacity: an Automatic Speech Recognition (ASR) corpus with 2,340 hours of transcribed, image-prompted natural speech across 19 languages (plus ~11,760 untranscribed hours) and a studio-quality Text-to-Speech (TTS) corpus of ~247 hours across 15 language configurations. We empirically benchmark three architecturally distinct ASR models -- Gemma 3n 2B, Whisper-Large-v3, and MMS-1B-all -- across the 19 ASR languages on strictly speaker-disjoint splits. Fine-tuning on WAXAL yields substantial gains, re
    
[^325]: 为什么引导有效：迈向语言模型参数动力学的统一视角

    Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics

    [https://arxiv.org/abs/2602.02343](https://arxiv.org/abs/2602.02343)

    该论文提出统一框架，将权重微调、LoRA适配和激活干预等LLM控制方法统一视为控制信号诱导的动态权重更新，并揭示了偏好与效用之间一致的权衡关系。

    

    控制大语言模型（LLM）的方法，包括局部权重微调、基于LoRA的适配以及基于激活的干预，通常被孤立地研究，这掩盖了它们之间的联系并使比较变得困难。在这项工作中，我们提出了一个统一的视角，将这些干预视为由控制信号诱导的动态权重更新，从而将它们置于同一个概念框架之中。基于这一视角，我们提出了一个统一的偏好-效用分析方法，将控制效果分解为偏好（即向目标概念靠拢的倾向）和效用（即连贯且任务有效的生成），并使用极性配对的对比样本在共同的对数几率尺度上对两者进行度量。在所有方法中，我们观察到偏好与效用之间存在一致的权衡关系：更强的控制会增加偏好，同时可预测地降低效用。我们进一步通过激活机制解释了这一行为。

    arXiv:2602.02343v4 Announce Type: replace-cross  Abstract: Methods for controlling large language models (LLMs), including local weight fine-tuning, LoRA-based adaptation, and activation-based interventions, are often studied in isolation, obscuring their connections and making comparison difficult. In this work, we present a unified view that frames these interventions as dynamic weight updates induced by a control signal, placing them within a single conceptual framework. Building on this view, we propose a unified preference-utility analysis that separates control effects into preference, defined as the tendency toward a target concept, and utility, defined as coherent and task-valid generation, and measures both on a shared log-odds scale using polarity-paired contrastive examples. Across methods, we observe a consistent trade-off between preference and utility: stronger control increases preference while predictably reducing utility. We further explain this behavior through an act
    
[^326]: 通过原子概念编辑的“宪法”来解释和控制模型行为

    Interpreting and Controlling Model Behavior via Constitutions for Atomic Concept Edits

    [https://arxiv.org/abs/2602.00092](https://arxiv.org/abs/2602.00092)

    该论文提出了一个黑盒可解释性框架，通过原子概念编辑（ACE）系统地学习一种可验证的自然语言“宪法”，建立提示词编辑与模型行为之间的因果映射，从而实现对模型行为的深入解释与有效控制。

    

    我们介绍了一个黑盒可解释性框架，该框架学习一个可验证的“宪法”：一种自然语言摘要，描述对提示词的更改如何影响模型的特定行为，例如其一致性、正确性或对约束的遵守。我们的方法利用原子概念编辑（ACE），这是一类有针对性的操作，可以在输入提示词中添加、删除或替换可解释的概念。通过系统地应用ACE并观察其在各种任务中对模型行为产生的效果，我们的框架学习到从编辑到可预测结果的因果映射。这个学到的“宪法”为模型提供了深入且可泛化的洞察。在实证方面，我们在包括数学推理和文本到图像对齐在内的多种任务上验证了我们的方法，用于控制和理解模型行为。我们发现，在文本到图像生成任务中，GPT-Image 往往侧重于语法层面的遵守，而 Image（摘要在此处截断）

    arXiv:2602.00092v2 Announce Type: replace-cross  Abstract: We introduce a black-box interpretability framework that learns a verifiable constitution: a natural language summary of how changes to a prompt affect a model's specific behavior, such as its alignment, correctness, or adherence to constraints. Our method leverages atomic concept edits (ACEs), which are targeted operations that add, remove, or replace an interpretable concept in the input prompt. By systematically applying ACEs and observing the resulting effects on model behavior across various tasks, our framework learns a causal mapping from edits to predictable outcomes. This learned constitution provides deep, generalizable insights into the model. Empirically, we validate our approach across diverse tasks, including mathematical reasoning and text-to-image alignment, for controlling and understanding model behavior. We found that for text-to-image generation, GPT-Image tends to focus on grammatical adherence, while Image
    
[^327]: CD4LM：面向扩散语言模型的一致性蒸馏与自适应解码

    CD4LM: Consistency Distillation and aDaptive Decoding for Diffusion Language Models

    [https://arxiv.org/abs/2601.02236](https://arxiv.org/abs/2601.02236)

    CD4LM 提出一个可组合的加速框架，通过离散空间一致性蒸馏训练轨迹不变的模型并配合自适应解码，在不牺牲准确性的前提下加速扩散语言模型的生成。

    

    扩散语言模型（DLM）通过并行解码多个 token 来加速生成，但在追求快速解码时往往会牺牲准确性。我们将这一现象归因于静态与动态之间的不匹配：训练阶段是在固定调度下优化局部去噪，而快速推理则跳步执行并遇到不规则的掩码模式。现有加速方法大多只解决该问题的其中一个方面：免训练的推理方法在不改变模型的情况下加速解码，但未能消除这种不匹配；训练侧的蒸馏方法虽然让模型适应了加速推理，但通常将学生模型绑定在特定的解码轨迹上，限制了可组合性，且仍存在准确性损失。我们提出 CD4LM，一个用于精确加速扩散语言模型的可组合框架。其核心训练组件——离散空间一致性蒸馏（DSCD）——在多样化的掩码状态下训练出轨迹不变的学生模型，从而在不固定解码轨迹的前提下减少上述不匹配……

    arXiv:2601.02236v2 Announce Type: replace  Abstract: Diffusion language models (DLMs) accelerate generation by decoding multiple tokens in parallel, but pushing them to fast decoding often sacrifices accuracy. We attribute this to a static-to-dynamic mismatch: training optimizes local denoising under fixed schedules, while fast inference skips steps and encounters irregular mask patterns. Existing accelerators largely address only one side of this problem. Training-free inference methods speed up decoding without changing the model, leaving the mismatch unresolved; training-side distillation methods adapt the model, but often tie the student to a specific decoding trajectory, limiting composability and still suffering accuracy loss. We propose CD4LM, a composable framework for accurate DLM acceleration. Its core training component, Discrete-Space Consistency Distillation (DSCD) trains a trajectory-invariant student on diverse masked states, reducing the mismatch without fixing a decodi
    
[^328]: 保真优先于结构：在长对话LLM记忆中，逐字片段优于有损的制品提取

    Fidelity Before Structure: Verbatim Chunks Beat Lossy Artifact Extraction in Long-Conversation LLM Memory

    [https://arxiv.org/abs/2601.00821](https://arxiv.org/abs/2601.00821)

    在长对话LLM记忆系统中，直接存储逐字对话片段的检索准确率显著优于LLM提取的结构化信息，性能差距的根源是有损蒸馏而非结构本身。

    

    越来越多的对话记忆系统将对话历史压缩为结构化的制品（提取的事实、决策或事件）。其前提假设是：经过提炼的结构比原始文本具有更好的检索效果。我们通过受控消融实验检验了这一前提。在一个固定的“检索—重排—推理”流水线中，我们仅替换存储的表示形式（LLM提取的类型化制品 vs. 逐字对话片段），而保持模型、检索器、重排器和评判器不变。结果显示，逐字片段在LoCoMo上以15.9个百分点胜出（43.9% vs. 28.0%），在LongMemEval-S上以22.0个百分点胜出（67.4% vs. 45.4%）。语义图无法弥合这一差距，且六项混杂因素对照实验均复现了该效应。其机制在于有损的蒸馏，而非结构本身。准确率与存储中保留下来的源文本量密切相关，且提取制品的流水线在总体准确率上并未超越朴素RAG（逐字片段的拒答表现更差；参见局限性部分）。

    arXiv:2601.00821v5 Announce Type: replace  Abstract: A growing class of conversational-memory systems compresses dialogue history into structured artifacts (extracted facts, decisions, or events). The premise is that distilled structure retrieves better than raw text. We test this premise with a controlled ablation. Within one fixed retrieval--rerank--reasoning pipeline, we swap only the stored representation (LLM-extracted typed artifacts versus verbatim conversation chunks) and hold the model, retriever, reranker, and judge constant. Verbatim chunks win by 15.9 points on LoCoMo (43.9% vs. 28.0%) and by 22.0 points on LongMemEval-S (67.4% vs. 45.4%). A semantic graph does not recover the gap, and six confound controls reproduce the effect. The mechanism is lossy distillation rather than structure per se. Accuracy tracks how much source text survives in the store, and the extracted-artifact pipeline does not beat naive RAG in overall accuracy (chunks do abstain worse; see Limitations).
    
[^329]: 从证据到判断：冲突证据下的轨迹增强推理

    From Evidence to Judgment: Trace-Augmented Reasoning under Conflicting Evidence

    [https://arxiv.org/abs/2512.16795](https://arxiv.org/abs/2512.16795)

    该论文提出在RAG面临冲突证据时，通过微调模型在回答前生成结构化思考轨迹（逐篇评判文档、推断冲突模式）来增强推理，并发布了相关数据集、基准及CATS评估框架，证明该方法显著提升了模型在有依据地回答或拒绝之间的决策能力。

    

    检索到的文档常常相互矛盾、包含过时信息或无法支持任何答案，而检索增强生成（RAG）系统缺乏可靠的方式来决定应信任哪些文档。我们将这一问题视为一个推理问题，并将其与答案生成分开进行测试。我们发布了943个训练与验证样例以及一个与之不相交的736个样例的基准数据集，并对四个语言模型进行微调，使其在回答之前写出结构化的思考轨迹：模型对每篇文档进行评判，推断冲突模式，然后写出有依据的答案或经过论证的拒绝回答。我们的评估框架CATS（冲突感知信任分数）对有依据的拒绝、行为一致性、事实依据性和单一真值召回率进行评分；一项人类研究验证了事实依据性和单一真值召回率（kappa=0.78和0.79）。思考轨迹提升了“回答/拒绝”决策：在全部三种提示设置下，未经微调的模型的拒绝F1分数最高仅为0.429（……原文在此处截断）

    arXiv:2512.16795v2 Announce Type: replace-cross  Abstract: Retrieved documents often disagree, contain outdated information, or fail to support any answer, and retrieval-augmented generation (RAG) systems have no reliable way to decide which documents to trust. We treat this as a reasoning problem and test it separately from answer generation. We release 943 training and validation examples and a disjoint 736-example benchmark, and fine-tune four language models to write a structured think trace before answering: the model judges each document, infers the conflict pattern, and writes a grounded answer or a justified refusal. Our evaluation framework, CATS (Conflict-Aware Trust Score), scores grounded refusal, behavioural adherence, factual grounding, and single-truth recall; a human study validates factual grounding and single-truth recall (kappa=0.78 and 0.79). The think trace improves the answer/refuse decision: under all three prompts, untuned models score at most 0.429 Refusal F1 (
    
[^330]: 将大型语言模型适配至低资源藏语：一项两阶段持续预训练与监督微调研究

    Adapting Large Language Models to Low-Resource Tibetan: A Two-Stage Continual and Supervised Fine-Tuning Study

    [https://arxiv.org/abs/2512.03976](https://arxiv.org/abs/2512.03976)

    该研究提出持续预训练加监督微调的两阶段策略，将 Qwen2.5-3B 成功适配到低资源藏语，显著降低困惑度并大幅提升汉藏翻译质量，同时揭示适配主要集中在嵌入层、输出头及中后期 MLP 层。

    

    由于数据稀缺和跨语言漂移，将大型语言模型（LLM）适配到低资源语言仍然是一项重大挑战。本工作展示了对 Qwen2.5-3B 进行藏语适配的两阶段方法，藏语是一种形态丰富但代表性不足的语言。我们首先采用持续预训练（CPT）建立藏语的语言基础，随后进行监督微调（SFT）以实现任务与翻译能力的专门化。实证评估表明，模型困惑度持续下降（从 2.98 降至 1.54），汉译藏翻译质量显著提升（BLEU：0.046 → 0.261；chrF：2.2 → 6.6）。针对 Qwen3-4B 全部 435 层的逐层分析显示，适配主要集中于嵌入层和输出头，而中后期的 MLP 投影则编码了领域特定的变换。我们的研究结果表明，CPT 构建了藏语的语义流形，而 SFT 则在此基础上进一步……

    arXiv:2512.03976v2 Announce Type: replace  Abstract: Adapting large language models (LLMs) to low-resource languages remains a major challenge due to data scarcity and cross-lingual drift. This work presents a two-stage adaptation of Qwen2.5-3B to Tibetan, a morphologically rich and underrepresented language. We employ Continual Pretraining (CPT) to establish Tibetan linguistic grounding, followed by Supervised Fine-Tuning (SFT) for task and translation specialization. Empirical evaluations demonstrate a consistent decrease in perplexity (from 2.98 $\rightarrow$ 1.54) and substantial improvements in Chinese$\rightarrow$Tibetan translation quality (BLEU: 0.046 $\rightarrow$ 0.261; chrF: 2.2 $\rightarrow$ 6.6). Layer-wise analysis across 435 layers in Qwen3-4B reveals that adaptation primarily concentrates on embedding and output heads, with mid--late MLP projections encoding domain-specific transformations. Our findings suggest that CPT constructs a Tibetan semantic manifold while SFT s
    
[^331]: 基于指南的大语言模型医学推理

    Guideline-Based Medical Reasoning in LLMs

    [https://arxiv.org/abs/2512.03838](https://arxiv.org/abs/2512.03838)

    该论文提出从医学共识指南（如SOFA、SAPS-II和Sepsis-3）自动生成黄金标准推理链作为监督信号，证明经过微调的小模型在医学推理任务上显著优于更大的单次提示模型，且单次提示无法可靠地产生可信赖的逐步推理。

    

    大语言模型（LLM）在医学问答方面表现出色，但医学推理模型的发展受到缺乏可扩展、可信赖的推理监督的限制。我们提出利用迄今为止未被充分开发的资源——医学共识指南。这些指南将数值化的临床测量指标映射到疾病评分，并可选择性地针对特定机构的例外情况进行定制。从共识规则中自动生成黄金标准推理链，既能支持大规模监督微调，也能支持自动评估。我们将SOFA、SAPS-II和Sepsis-3指南应用于MIMIC-III数据集，结果显示经过微调的小模型在推导正确性和F1分数上显著优于更大的单次提示模型，实现了接近完美的演绎推理和可靠的例外情况处理。即使在模型获得共识定义的情况下，可信赖的逐步推理也无法通过单次提示可靠地产生。

    arXiv:2512.03838v2 Announce Type: replace  Abstract: Large Language Models (LLMs) excel at medical question-answering, yet progress toward medical reasoning models is limited by the lack of scalable, trustworthy reasoning supervision. We propose tapping into the so far underused resource of medical consensus guidelines. These map numerical clinical measurements to disease scores and can optionally be curated for site-specific exceptions. Automatically generating gold-standard reasoning chains from consensus rules enables both large-scale supervised fine-tuning and automatic evaluation. Applying SOFA, SAPS-II, and Sepsis-3 guidelines to MIMIC-III, we show that small fine-tuned models significantly outperform larger one-shot prompted models in derivation correctness and F1 score, achieving near-perfect deductive inference and reliable exception handling. Trustworthy step-by-step reasoning does not reliably emerge from one-shot prompting, even when models receive the consensus definition 
    
[^332]: 读者更偏爱基于受版权保护书籍训练的AI的输出，而非专家级人类作家

    Readers Prefer Outputs of AI Trained on Copyrighted Books over Expert Human Writers

    [https://arxiv.org/abs/2510.13939](https://arxiv.org/abs/2510.13939)

    该研究首次通过大规模盲测评估发现，当前沿AI模型在作者完整作品上微调后，无论是专业文学读者还是普通读者都更偏好AI模拟获奖作家风格的文学输出，超越了人类专家作家，这对版权诉讼中的“替代性作品”争论具有重要启示。

    

    将受版权保护的书籍用于训练AI引发了作家的诉讼，他们担心AI会生成衍生内容。然而，这些模型是否能生成模拟作者声音的高质量文学文本仍不清楚。我们进行了一项预注册研究，将受过艺术硕士（MFA）训练的作家与三个前沿模型（ChatGPT、Claude、Gemini）进行比较，让它们撰写模拟50位获奖作家风格的最多450字的文本片段。在由28名MFA训练的读者和516名受过大学教育的普通读者参与的盲测成对评估中，通过上下文提示（in-context prompting）生成的AI文本受到MFA读者的强烈排斥，无论是在风格忠实度（OR=0.16）还是质量（OR=0.13）方面；而普通读者没有表现出忠实度偏好（OR=1.06），但在质量上更偏好AI（OR=1.82）。然而，在作者完整作品上微调ChatGPT后，这些结果发生了逆转：MFA读者更偏好AI的风格忠实度（OR=8.16）和质量（OR=1.87），普通读者则表现出更强的偏好。

    arXiv:2510.13939v5 Announce Type: replace-cross  Abstract: The use of copyrighted books for training AI has sparked lawsuits from authors concerned about AI generating derivative content. Yet whether these models can produce high-quality literary text emulating authors' voices remains unclear. We conducted a preregistered study comparing MFA-trained writers with three frontier models (ChatGPT, Claude, Gemini) writing up to 450-word excerpts emulating 50 award-winning authors' styles. In blind pairwise evaluations by 28 MFA-trained readers and 516 college-educated general readers, AI text from in-context prompting was strongly disfavored by MFA readers for stylistic fidelity (OR=0.16) and quality (OR=0.13), while general readers showed no fidelity preference (OR=1.06) but favored AI for quality (OR=1.82). Fine-tuning ChatGPT on authors' complete works reversed these results: MFA readers favored AI for fidelity (OR=8.16) and quality (OR=1.87), with general readers showing even stronger p
    
[^333]: 迈向基于大语言模型支持的批判性思维子技能自动化评估

    Toward LLM-Supported Automated Assessment of Critical Thinking Subskills

    [https://arxiv.org/abs/2510.12915](https://arxiv.org/abs/2510.12915)

    本文以学生议论文写作为任务载体，基于技能进阶体系开发编码量表并人工标注语料，验证了利用大语言模型（通过零样本和少样本提示等方法）自动评估批判性思维子技能的可行性。

    

    随着世界日益充斥着人工智能生成的内容、虚假信息和算法劝说，批判性思维——即评估证据、识别不可靠论断和行使独立判断的能力——正在成为一项定义人类的核心技能。通过及时的评估和反馈来培养批判性思维技能至关重要；然而，在教育数据挖掘领域，关于定义、测量和支持批判性思维的研究工作仍然有限。在本文中，我们研究了测量批判性思维背后“子技能”的可行性。我们将研究建立在一个真实任务之上，即学生通过撰写议论文来实践批判性思维。我们基于既有的技能进阶体系开发了一套编码量表，并对学生议论文语料库完成了人工编码。随后，我们评估了三种不同的自动评分方法：零样本提示、少样本提示等。

    arXiv:2510.12915v3 Announce Type: replace-cross  Abstract: As the world becomes increasingly saturated with AI-generated content, disinformation, and algorithmic persuasion, critical thinking - the capacity to evaluate evidence, detect unreliable claims, and exercise independent judgment - is becoming a defining human skill. Developing critical thinking skills through timely assessment and feedback is crucial; however, there has not been extensive work in educational data mining on defining, measuring, and supporting critical thinking. In this paper, we investigate the feasibility of measuring "subskills" that underlie critical thinking. We ground our work in an authentic task where students operationalize critical thinking by writing argumentative essays. We developed a coding rubric based on an established skills progression and completed human coding for a corpus of student essays. We then evaluated three distinct approaches to automated scoring: zero-shot prompting, few-shot prompt
    
[^334]: 用事实击败社交媒体上的仇恨叙事：检索增强的反仇恨言语生成

    Beating Hateful Narratives on Social Media with Facts: Retrieval-Augmented Counter-Speech Generation

    [https://arxiv.org/abs/2510.12316](https://arxiv.org/abs/2510.12316)

    该论文提出了首个基于权威人权文献（如联合国数字图书馆、EUR-Lex等）构建的大规模知识库的检索增强生成管线，用于生成事实可靠且针对性强的反仇恨言语，以建设性地对抗社交媒体上的仇恨言论。

    

    网络仇恨言论通过放大针对边缘化群体的歧视和有害叙事，威胁着社会凝聚力。反仇恨言语——即基于事实、尊重他人的对仇恨言论的回应——为内容删除提供了一种建设性的替代方案，但其大规模生成仍受限于事实可靠性和回复过于笼统的风险。我们通过一个检索增强生成管线来应对这些挑战，该管线基于首个大规模、精心整理的知识库构建，该知识库源自权威的国际和区域人权资料来源，包括联合国数字图书馆、EUR-Lex和欧盟基本权利署。该框架涵盖了八个经常成为网络滥用目标的群体。我们系统地评估了管线的所有组件，结合自动指标、基于大语言模型的评判以及人工评估，并通过对比实验证明每个设计选择都具有实证依据。

    arXiv:2510.12316v2 Announce Type: replace  Abstract: Online hate speech threatens social cohesion by amplifying discrimination and harmful narratives against marginalized communities. While counter-speech (CS) - fact-based, respectful responses to hate - offers a constructive alternative to content removal, its large-scale generation remains constrained by factual reliability and the risk of generic responses. We address these challenges with a Retrieval-Augmented Generation (RAG) pipeline grounded in the first large-scale, curated knowledge base built from authoritative international and regional human rights sources, including the United Nations Digital Library, EUR-Lex, and the EU Agency for Fundamental Rights. The framework covers eight groups frequently targeted by online abuse. We systematically evaluate all pipeline components demonstrating that each design choice is empirically motivated, combining automatic metrics, LLM-based judgments, and human evaluation with comparisons ag
    
[^335]: 反馈取证：一个测量AI人格的工具包

    Feedback Forensics: A Toolkit to Measure AI Personality

    [https://arxiv.org/abs/2509.26305](https://arxiv.org/abs/2509.26305)

    该论文提出了Feedback Forensics——一个用于显式测量和评估AI模型人格的开源工具包，以弥补现有基于人类反馈排序的不透明评估方法在衡量模型性格特质方面的不足。

    

    一些使AI模型成为“好”模型的特质很难预先明确描述。例如，回答应该更礼貌还是更随意？这类特质有时被概括为模型性格或人格。由于缺乏明确的目标，基于自动检查的传统基准测试难以衡量这类特质。使用人类反馈的评估方法（如Chatbot Arena）已成为流行的替代方案。这些方法通过将多个模型的回答相互排序，来隐式推断“更好”的人格及其他理想特质。近期模型发布中暴露的问题凸显了这些现有不透明评估方法的局限性：沉迷于哥布林、过度谄媚以及普遍夸张的模型人格曾导致模型回滚和大量媒体报道。尽管存在这些已知问题，目前用于显式评估模型人格的公开工具仍然有限。我们提出了Feedback Forensics：一个开源工具包……（摘要在此处截断）

    arXiv:2509.26305v3 Announce Type: replace-cross  Abstract: Some traits making a "good" AI model are hard to describe upfront. For example, should responses be more polite or more casual? Such traits are sometimes summarized as model character or personality. Without a clear objective, conventional benchmarks based on automatic checks struggle to measure such traits. Evaluation methods using human feedback such as Chatbot Arena have emerged as a popular alternative. These methods infer "better" personality and other desirable traits implicitly by ranking multiple model responses relative to each other. Recent issues with model releases highlight limitations of these existing opaque evaluation approaches: goblin-obsessed, sycophantic and generally over-the-top personalities have resulted in model rollbacks and extensive media coverage. Despite these known issues, limited public tooling exists to explicitly evaluate model personality. We introduce Feedback Forensics: an open-source toolki
    
[^336]: PersonaFuse：面向企业任务中自适应大语言模型行为的情境感知后训练框架

    PersonaFuse: A Situation-Aware Post-Training Framework for Adaptive LLM Behavior in Enterprise Tasks

    [https://arxiv.org/abs/2509.07370](https://arxiv.org/abs/2509.07370)

    PersonaFuse是一个基于特质激活理论与大五人格模型的后训练框架，通过带有特质级专家和上下文感知路由的混合专家架构，使大语言模型能够根据企业任务情境动态调整自身行为，实现无能力权衡的情境自适应表现。

    

    大语言模型（LLM）正越来越多地被部署于企业环境中，以支持任务执行和工作流自动化。尽管通用模型在标准基准测试中表现优异，但在实践中，组织往往需要将其适配到特定领域的任务和运营约束中。然而，现有的后训练方法会引入能力上的权衡，且缺乏情境适应机制。在本工作中，我们提出了PersonaFuse，一个能够实现情境感知行为适配的大语言模型后训练框架。PersonaFuse以特质激活理论为基础，并通过大五人格模型加以实现，利用带有特质级专家和上下文感知路由机制的混合专家架构，根据任务情境动态调整模型行为。我们在三个层面对PersonaFuse进行评估。在能力层面，它在通用推理、（摘要在此处被截断）

    arXiv:2509.07370v4 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly deployed in enterprise settings to support task execution and workflow automation. Despite the strong performance of general-purpose models on standard benchmarks, organizations often need to adapt them to domain-specific tasks and operational constraints in practice. However, existing post-training approaches introduce capability trade-offs and lack mechanisms for situational adaptation. In this work, we propose PersonaFuse, a post-training framework that enables situation-aware behavioral adaptation in LLMs. Grounded in Trait Activation Theory and operationalized through the Five-Factor Model, PersonaFuse dynamically adjusts model behavior based on task context using a mixture-of-experts architecture with trait-level experts and a context-aware routing mechanism. We evaluate PersonaFuse at three levels. At the capability level, it achieves balanced improvements across general reasoning,
    
[^337]: 符号学复杂性及其对文化建模的认识论意涵

    Semiotic Complexity and Its Epistemological Implications for Modeling Culture

    [https://arxiv.org/abs/2508.00095](https://arxiv.org/abs/2508.00095)

    本文提出“符号学复杂性”概念——即文本含义在不同阐释视角下的变化程度——将计算人文的建模视为从文化语言领域到计算数学领域的翻译过程，并论证缺乏翻译理论化会导致微妙而影响深远的建模错误。

    

    计算人文学科需要对方法进行更深入的理论化，以获得认识论与阐释层面的清晰性，进而推动该领域的成熟。在本文中，我们将此类建模工作框定为一种翻译活动：即从文化的、语言学的领域转换到计算的、数学的领域，再转换回来。译者会因清晰阐明其翻译过程的理论而受益，计算人文学者在工作中同样如此——以确保内部一致性、避免微妙却影响重大的翻译错误，并促进阐释的透明性。本文的贡献在于揭示了理论化缺失中一个尤为关键的维度，以及由此在我们的建模实践中产生的各类翻译错误。沿着这一思路，我们引入了“符号学复杂性”这一概念，即某一文本的含义在不同阐释视角之间可能发生变化的程度，并论证了……（原文摘要在此处截断）

    arXiv:2508.00095v2 Announce Type: replace  Abstract: Greater theorizing of methods in the computational humanities is needed for epistemological and interpretive clarity, and therefore the maturation of the field. In this paper, we frame such modeling work as engaging in translation work from a cultural, linguistic domain into a computational, mathematical domain, and back again. Translators benefit from articulating the theory of their translation process, and so do computational humanists in their work -- to ensure internal consistency, avoid subtle yet consequential translation errors, and facilitate interpretive transparency. Our contribution in this paper is to lay out a particularly consequential dimension of the lack of theorizing and the sorts of translation errors that emerge in our modeling practices as a result. Along these lines we introduce the idea of semiotic complexity as the degree to which the meaning of some text may vary across interpretive lenses, and make the case
    
[^338]: SmartThinker：通过步骤级Token分配学习高密度推理

    SmartThinker: Learning High-Density Reasoning via Step-Level Token Allocation

    [https://arxiv.org/abs/2507.04348](https://arxiv.org/abs/2507.04348)

    该论文提出SmartThinker，通过步骤级的token分配学习高密度推理，在保留关键推理步骤的同时压缩低价值的冗余内容，从而在严格上下文约束下实现高效推理。

    

    大型推理模型（LRMs）通过推理时扩展展现了卓越的推理能力，但这一进展也给其推理过程引入了大量的冗余与低效，造成了巨大的计算浪费。以往的工作试图通过在强化学习（RL）过程中惩罚生成样本的整体长度来缓解这一问题，以鼓励更简洁的推理链。对于部署在具有严格上下文约束的智能体系统中的大型推理模型而言，高效推理至关重要。现有方法主要依赖全局长度惩罚，忽视了步骤级结构，在无意中压缩了关键推导的同时却保留了低价值的绕路推理。因此，我们主张采用一种有原则的、步骤感知的替代方案，即保留关键步骤并压缩低价值的绕路推理。然而，这带来了三个根本性挑战：……

    arXiv:2507.04348v3 Announce Type: replace  Abstract: Large reasoning models (LRMs) have exhibited remarkable reasoning capabilities through inference-time scaling, but this progress has also introduced considerable redundancy and inefficiency into their reasoning processes, resulting in substantial computational waste. Previous work has attempted to mitigate this issue by penalizing the overall length of generated samples during reinforcement learning (RL), with the goal of encouraging a more concise chains of Efficient reasoning is critical for large reasoning models (LRMs) deployed in agent systems with strict context constraints. Existing methods predominantly rely on global length penalties that overlook step-level structure, inadvertently compressing critical derivations while retaining low-value detours. Hence, we argue for a principled, step-aware alternative that preserves critical steps and compresses low-value detours. However, it raises three fundamental challenges: the diff
    
[^339]: Revela：通过语言建模进行密集检索器学习

    Revela: Dense Retriever Learning via Language Modeling

    [https://arxiv.org/abs/2506.16552](https://arxiv.org/abs/2506.16552)

    提出Revela框架，将检索类比为词元块之间的依赖关系学习，通过语言建模式的自监督目标训练密集检索器，从而摆脱对昂贵标注查询-文档对的依赖。

    

    密集检索器在访问外部和专业知识以增强语言模型（LM）方面发挥着重要作用。训练密集检索器通常需要标注的查询-文档对，这类数据创建成本高昂，且在专业领域（如代码）或复杂场景（如需要推理）中十分稀缺。这些实际挑战激发了人们对自监督检索器学习日益增长的兴趣。由于语言模型通过自监督学习目标（即下一词元预测）来捕捉词元级别的依赖关系，我们可以类似地将检索任务视为学习词元块之间的依赖关系。这一类比自然引出如下问题：我们如何以语言建模的精神调整自监督学习目标来训练检索器？为回答这一问题，我们提出了Revela，一个通过语言建模实现自监督检索器学习的统一且可扩展的训练框架。

    arXiv:2506.16552v4 Announce Type: replace-cross  Abstract: Dense retrievers play a vital role in accessing external and specialized knowledge to augment language models (LMs). Training dense retrievers typically requires annotated query-document pairs, which are costly to create and scarce in specialized domains (e.g., code) or in complex settings (e.g., requiring reasoning). These practical challenges have sparked growing interest in self-supervised retriever learning. Since LMs are trained to capture token-level dependencies through a self-supervised learning objective (i.e., next token prediction), we can analogously cast retrieval as learning dependencies among chunks of tokens. This analogy naturally leads to the question: How can we adapt self-supervised learning objectives in the spirit of language modeling to train retrievers?   To answer this question, we introduce Revela, a unified and scalable training framework for self-supervised retriever learning via language modeling. R
    
[^340]: BitBypass：一种利用比特流伪装攻击对齐大语言模型的新越狱方向

    BitBypass: A New Direction in Jailbreaking Aligned Large Language Models with Bitstream Camouflage

    [https://arxiv.org/abs/2506.02479](https://arxiv.org/abs/2506.02479)

    本文提出了一种名为BitBypass的新型黑盒越狱攻击方法，通过连字符分隔的比特流伪装技术，从数据底层的比特级信息表示入手绕过对齐大语言模型的安全防御，开辟了越狱攻击的新方向。

    

    大语言模型（LLM）生成有害和不安全内容的固有风险凸显了对其进行安全对齐的必要性。为了确保LLM的安全对齐，研究者开发了各种技术，如监督微调、基于人类反馈的强化学习以及红队测试。然而，这些经过对齐的LLM的鲁棒性始终受到对抗性攻击的挑战，这些攻击利用了安全对齐中尚未被发现和潜在的漏洞。在本文中，我们开发了一种新颖的黑盒越狱攻击方法，称为BitBypass，它利用连字符分隔的比特流伪装来对经过对齐的LLM实施越狱。这代表了越狱攻击的一个新方向：通过利用数据以连续比特流形式存在的基本信息表示，而非借助提示工程或对抗性操纵。我们对五个最先进的LLM进行了评估，即GPT-4o、Gemini 1.5、Claude 3.5、Llama 3.1

    arXiv:2506.02479v3 Announce Type: replace-cross  Abstract: The inherent risk of generating harmful and unsafe content by Large Language Models (LLMs), has highlighted the need for their safety alignment. Various techniques like supervised fine-tuning, reinforcement learning from human feedback, and red-teaming were developed for ensuring the safety alignment of LLMs. However, the robustness of these aligned LLMs is always challenged by adversarial attacks that exploit unexplored and underlying vulnerabilities of the safety alignment. In this paper, we develop a novel black-box jailbreak attack, called BitBypass, that leverages hyphen-separated bitstream camouflage for jailbreaking aligned LLMs. This represents a new direction in jailbreaking by exploiting fundamental information representation of data as continuous bits, rather than leveraging prompt engineering or adversarial manipulations. Our evaluation of five state-of-the-art LLMs, namely GPT-4o, Gemini 1.5, Claude 3.5, Llama 3.1,
    
[^341]: 打破沉默：面向孟加拉语文本到手语词汇翻译的数据集与基准

    Breaking the Silence: A Dataset and Benchmark for Bangla Text-to-Gloss Translation

    [https://arxiv.org/abs/2504.02293](https://arxiv.org/abs/2504.02293)

    本文构建了首个公开可用的孟加拉语文本到手语词汇翻译平行数据集及基准测试，填补了孟加拉手语研究的资源空白，并通过对比开源模型与闭源大语言模型评估了低资源场景下的翻译性能。

    

    手语词汇标注是一种书面近似形式，用于连接手语（SL）与其对应的口语。尽管孟加拉国至少有300万聋人及听力障碍人口，但孟加拉手语（BdSL）在很大程度上仍未得到充分研究，此前针对孟加拉语文本到手语词汇翻译的研究工作非常有限，且在本研究开展之时尚无任何公开可用的平行数据集。为填补这一空白，我们构建了首个公开可用的孟加拉语文本到手语词汇数据集，其中包含1,000个人工标注和4,000个合成生成的孟加拉语句子-手语词汇对，另有159个由专家人工标注的句子-手语词汇对作为测试集。我们的实验框架对多个经过微调的开源模型与一个领先的闭源大语言模型进行了对比分析，以评估它们在低资源孟加拉手语翻译任务中的表现。GPT-5.4取得了最佳的自动评估指标分数，而开源模型的表现也具有竞争力，尽管（原文在此处截断）。

    arXiv:2504.02293v4 Announce Type: replace-cross  Abstract: Gloss is a written approximation that bridges Sign Language (SL) and its corresponding spoken language. Despite a deaf and hard-of-hearing population of at least 3 million in Bangladesh, Bangla Sign Language (BdSL) remains largely understudied, with very limited prior work on Bangla text-to-gloss translation and no publicly available parallel dataset at the time of this work. To address this gap, we construct the first publicly available Bangla text-to-gloss dataset, consisting of 1,000 manually annotated and 4,000 synthetically generated Bangla sentence-gloss pairs, along with 159 expert human-annotated pairs used as a test set. Our experimental framework performs a comparative analysis between several fine-tuned open-source models and a leading closed-source LLM to evaluate their performance in low-resource BdSL translation. GPT-5.4 achieves the best automatic metric scores, while open-source models perform competitively desp
    
[^342]: ResBench：面向资源感知的大语言模型生成FPGA设计基准测试

    ResBench: Benchmarking LLM-Generated FPGA Designs with Resource Awareness

    [https://arxiv.org/abs/2503.08823](https://arxiv.org/abs/2503.08823)

    ResBench是首个关注FPGA硬件资源效率的LLM代码生成基准，通过12个类别共56个问题评估LLM生成的HDL代码，弥补了现有基准只关注功能正确性而忽视资源使用的不足。

    

    现场可编程门阵列（FPGA）在现代硬件设计中被广泛使用，但为FPGA实现编写硬件描述语言（HDL）代码仍然是一项复杂且耗时的任务。大语言模型（LLM）已成为HDL生成的一种有前景的工具，但现有的LLM代码生成基准主要关注功能正确性，而忽略了硬件资源的使用情况。此外，当前的基准多样性有限，未能充分覆盖现实世界中广泛的FPGA应用。为了解决这些不足，我们提出了ResBench，这是首个专注于资源的基准，旨在明确区分资源优化与低效的LLM生成HDL代码。ResBench包含12个类别共56个问题，涵盖从有限状态机到金融计算等多种应用。我们的开源评估框架通过…（原文在此处截断）自动测试大语言模型。

    arXiv:2503.08823v3 Announce Type: replace-cross  Abstract: Field-Programmable Gate Arrays (FPGAs) are widely used in modern hardware design, yet writing Hardware Description Language (HDL) code for FPGA implementation remains a complex and time-consuming task. Large Language Models (LLMs) have emerged as a promising tool for HDL generation, but existing benchmarks for LLM-based code generation primarily focus on functional correctness while overlooking hardware resource usage. Furthermore, current benchmarks offer limited diversity and do not fully represent the wide range of real-world FPGA applications. To address these shortcomings, we introduce ResBench, the first resource-focused benchmark explicitly designed to distinguish between resource-optimized and inefficient LLM-generated HDL code. ResBench consists of 56 problems across 12 categories, covering applications from finite state machines to financial computing. Our open-source evaluation framework automatically tests LLMs by g
    
[^343]: Transformer中的锐度差异原理用于加速语言模型预训练

    The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training

    [https://arxiv.org/abs/2502.19002](https://arxiv.org/abs/2502.19002)

    本文发现Transformer各模块间存在训练早期出现且贯穿全程的“锐度差异”，据此提出按各模块锐度定制学习率的Blockwise LR策略，集成到AdamW后在多种模型和数据集上实现近2倍的预训练加速并取得更低最终损失。

    

    Transformer由多种构建模块组成，例如嵌入层、归一化层、自注意力机制和逐点前馈网络，因此理解这些模块之间的差异与相互作用十分重要。本文揭示了这些模块之间清晰存在的“锐度差异”，这一差异在训练早期便会显现，并有趣地在整个训练过程中持续存在。受此发现启发，我们提出了分块学习率策略，即根据每个模块的锐度为其定制学习率，从而加速大语言模型（LLM）的预训练。将分块学习率集成到AdamW中后，与原始AdamW相比，我们持续获得了更低的最终损失以及接近2倍的加速。我们在GPT-2和LLaMA上、模型规模从0.12B到2B、数据集涵盖OpenWebText、MiniPile和C4的情况下验证了这一加速效果。最后，我们将分块学习率集成到了Adam-mini中（Zhang et al., 202…）

    arXiv:2502.19002v2 Announce Type: cross  Abstract: Transformers consist of diverse building blocks, such as embedding layers, normalization layers, self-attention mechanisms, and point-wise feedforward networks. Thus, understanding the differences and interactions among these blocks is important. In this paper, we uncover a clear Sharpness Disparity across these blocks, which emerges early in training and intriguingly persists throughout the training process. Motivated by this finding, we propose Blockwise Learning Rate (LR), a strategy that tailors the LR to each block's sharpness, accelerating large language model (LLM) pre-training. By integrating Blockwise LR into AdamW, we consistently achieve lower terminal loss and nearly $2\times$ speedup compared to vanilla AdamW. We demonstrate this acceleration across GPT-2 and LLaMA, with model sizes ranging from 0.12B to 2B and datasets of OpenWebText, MiniPile, and C4. Finally, we incorporate Blockwise LR into Adam-mini (Zhang et al., 202
    
[^344]: 3D-MoE：基于混合专家模型迈向空间智能的3D推理与动作生成

    3D-MoE: Towards Spatial Intelligence with Mixture-of-Experts for 3D Reasoning and Action Generation

    [https://arxiv.org/abs/2501.16698](https://arxiv.org/abs/2501.16698)

    3D-MoE通过混合专家架构与模态/空间上下文感知的概率路由实现高效3D推理，并集成Pose-DiT扩散动作头在单步内生成精确6D位姿动作，以大幅减少的激活参数在3D视觉-语言基准上取得优越性能。

    

    空间智能，涵盖3D感知与推理，是人工智能下一个至关重要的前沿领域。对当前依赖密集Transformer处理空间任务的3D视觉-语言模型（VLM）进行扩展会带来难以承受的计算成本。在本文中，我们提出了3D-MoE，这是一个利用高效混合专家架构的3D视觉-语言模型，它采用模态感知与空间上下文感知的概率路由方案，并通过一种新颖的路由课程策略进行稳定培养。为了将3D-MoE无缝扩展至具身智能，我们集成了一个基于扩散模型的动作头Pose-DiT，将3D-MoE转变为3D视觉-语言-动作（VLA）模型。通过采用整流流框架，Pose-DiT能够在单次采样步骤中生成精确的6D位姿动作。大量实验表明，3D-MoE在多种3D视觉-语言基准测试中，以大幅减少的激活参数量取得了优越的性能，并实现了更高的成功率，同时支持……

    arXiv:2501.16698v2 Announce Type: replace  Abstract: Spatial intelligence, encompassing 3D perception and reasoning, is the essential next frontier of AI. Scaling current 3D vision-language models (VLMs) that rely on dense Transformers for spatial tasks incurs prohibitive computational costs. In this paper, we introduce 3D-MoE, a 3D VLM leveraging an efficient mixture-of-experts architecture with a modality- and spatial-context-aware probabilistic routing scheme, stably cultivated by a novel routing curriculum. To seamlessly extend 3D-MoE to embodied AI, we integrate a diffusion-based action head, Pose-DiT, transforming 3D-MoE into a 3D vision-language-action (VLA) model. By employing a rectified flow framework, Pose-DiT generates precise 6D pose actions in a single sampling step. Extensive experiments demonstrate that 3D-MoE achieves superior performance on diverse 3D vision-language benchmarks with drastically fewer activated parameters and yields higher success rates while enabling 
    
[^345]: 使用智能体AI自然语言框架缓解幻觉

    Hallucination Mitigation using Agentic AI Natural Language-Based Frameworks

    [https://arxiv.org/abs/2501.13946](https://arxiv.org/abs/2501.13946)

    该论文提出了一种基于自然语言交互的多智能体协作框架，通过四级智能体流水线检测、修正并量化AI幻觉，并创新性地提出四个关键绩效指标及总幻觉评分（THS）来系统评估幻觉缓解效果。

    

    幻觉仍然是生成式人工智能面临的重大挑战，它削弱了人们对AI系统及其可靠性的信任。本研究探讨了如何通过编排多个专业化的AI智能体来缓解幻觉问题，重点关注采用自然语言处理（NLP）进行智能体间交互的系统。我们设计了一条流水线：将310条特意设计以诱发幻觉的提示输入前端智能体，其输出由二级和三级智能体进行审查与完善，这些智能体负责检测未经验证的说法、添加明确的免责声明并澄清推测性内容。我们提出了四个新颖的关键绩效指标（KPI）——事实声明密度、事实依据引用数、虚构免责声明频率和显式情境化评分——并将它们汇总为总幻觉评分（THS），由专门的第四级评估智能体进行评估。智能体间的上下文信息由开放楼层协议（OFP）承载。

    arXiv:2501.13946v2 Announce Type: replace-cross  Abstract: Hallucinations remain a significant challenge for Generative AI, undermining trust in AI systems and their reliability. This study investigates how orchestrating multiple specialized AI agents can mitigate hallucinations, with a focus on systems that use Natural Language Processing (NLP) for agent interaction. We design a pipeline that feeds 310 prompts, purposefully crafted to induce hallucinations, into a front-end agent; its outputs are reviewed and refined by second- and third-level agents that detect unverified claims, add explicit disclaimers and clarify speculative content. We introduce four novel Key Performance Indicators (KPIs) - Factual Claim Density, Factual Grounding References, Fictional Disclaimer Frequency and Explicit Contextualization Score - aggregated into a Total Hallucination Score (THS) and assessed by a dedicated fourth-level evaluator agent. Inter-agent context is carried by the Open Floor Protocol (OFP
    
[^346]: 大型语言模型中的多语言幻觉差距

    Multilingual Hallucination Gaps in Large Language Models

    [https://arxiv.org/abs/2410.18270](https://arxiv.org/abs/2410.18270)

    该研究提出了“多语言幻觉差距”这一新概念，通过将FactScore指标扩展至多语言环境，并在19种语言上进行传记生成实验，揭示了大型语言模型的幻觉率因语言不同而存在显著差异，尤其是高资源语言与低资源语言之间。

    

    随着大型语言模型（LLMs）生成类人文本的能力不断增强，它们正日益被用作传统搜索引擎的替代品。然而，这种转变令人担忧，因为LLMs经常产生幻觉，即那些看似高度可信但实则具有误导性或虚假的信息。在本研究中，我们探讨了自由文本生成中跨多种语言的幻觉现象，重点关注我们所谓的“多语言幻觉差距”。这些差距反映了幻觉答案出现的频率因提示词和所用语言的不同而存在差异。为了量化这类幻觉，我们使用了FactScore指标，并将其框架扩展到了多语言环境中。我们使用来自LLaMA、Qwen和Aya系列的大型语言模型进行了实验，生成了19种语言的人物传记，并将结果与Wikipedia页面进行了比较。我们的结果揭示了幻觉率的显著变化，尤其是在高资源语言和低资源语言之间。

    arXiv:2410.18270v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used as alternatives to traditional search engines given their capacity to generate text that resembles human language. However, this shift is concerning, as LLMs often generate hallucinations, misleading or false information that appears highly credible. In this study, we explore the phenomenon of hallucinations across multiple languages in freeform text generation, focusing on what we call multilingual hallucination gaps. These gaps reflect differences in the frequency of hallucinated answers depending on the prompt and language used. To quantify such hallucinations, we used the FactScore metric and extended its framework to a multilingual setting. We conducted experiments using LLMs from the LLaMA, Qwen, and Aya families, generating biographies in 19 languages and comparing the results to Wikipedia pages. Our results reveal variations in hallucination rates, especially between high and
    
[^347]: 利用Tsetlin机器学习人类可解释规则以实现高精度文本分类及其医疗应用

    Using the Tsetlin Machine to Learn Human-Interpretable Rules for High-Accuracy Text Categorization with Medical Applications

    [https://arxiv.org/abs/1809.04547](https://arxiv.org/abs/1809.04547)

    本文首次提出基于Tsetlin机器的文本分类方法，通过学习人类可解释的简单命题规则（如合取子句），在医疗文本分类任务中同时实现高精度与可解释性。

    

    医疗应用对当今的文本分类技术提出了双重挑战，要求同时具备高精度和易于解释性。尽管深度学习在精度上实现了飞跃，但这一飞跃是以牺牲可解释性为代价的。为应对这一精度与可解释性之间的挑战，我们首次提出了一种基于新近提出的Tsetlin机器的文本分类方法。简而言之，我们将文本中的词项表示为命题变量，并基于这些变量使用简单的命题公式来捕捉类别，例如：如果文本中出现“皮疹”、“反应”和“青霉素”这些词，则分类为过敏（Allergy）。Tsetlin机器从带标注的文本中学习这些公式，利用合取子句来表示每个类别的特定特征。实际上，甚至词项的缺失（即否定特征）也可以用于分类目的。我们与朴素贝叶斯、决策树、线性支持向量机等方法的实证比较（摘要原文在此处截断）……

    arXiv:1809.04547v2 Announce Type: cross  Abstract: Medical applications challenge today's text categorization techniques by demanding both high accuracy and ease-of-interpretation. Although deep learning has provided a leap ahead in accuracy, this leap comes at the sacrifice of interpretability. To address this accuracy-interpretability challenge, we here introduce, for the first time, a text categorization approach that leverages the recently introduced Tsetlin Machine. In all brevity, we represent the terms of a text as propositional variables. From these, we capture categories using simple propositional formulae, such as: if "rash" and "reaction" and "penicillin" then Allergy. The Tsetlin Machine learns these formulae from a labelled text, utilizing conjunctive clauses to represent the particular facets of each category. Indeed, even the absence of terms (negated features) can be used for categorization purposes. Our empirical comparison with Na\"ive Bayes, decision trees, linear su
    

