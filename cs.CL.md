# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jailbreak Detection on Large Language Models via Contrastive Prototype Structures](https://arxiv.org/abs/2611.24936) | 提出JDPI方法，利用对比学习构建有害与良性原型，仅凭二元安全标签通过相对原型距离实现大语言模型越狱检测，无需训练额外下游分类器，在四个基准上将检测准确率提升9.1-18.0个百分点。 |
| [^2] | [Who are you? The Personal Imitation Game](https://arxiv.org/abs/2611.24888) | 该论文提出了“个人模仿游戏”这一更难的挑战，要求机器在对话中模仿特定个体而非普通人，并利用十年个人数字通信数据构建的LLM系统实验表明，审讯者无法可靠地区分该系统与真人本人。 |
| [^3] | [Ground, Iterate, Trace: Decomposing AI Research Ideation](https://arxiv.org/abs/2611.24887) | 提出了CAR智能体框架，通过文献扎根、语义知识图谱与四种可追溯的概念操作符分解研究构思过程，显著提升了AI生成研究想法的方法论新颖性。 |
| [^4] | [Can Programs Replace LLM Judges? Executable Program Judges for Scalable and Efficient Relevance Labeling](https://arxiv.org/abs/2611.24849) | 提出ExecJudges方法，利用前沿LLM将评估指南转化为仅依赖词汇匹配和标准NLP库的可执行程序，从而取代昂贵的LLM裁判调用，实现快速、可解释、可复现的大规模相关性标注。 |
| [^5] | [Vibe-Trading: Towards Agent-Native Trading](https://arxiv.org/abs/2611.24845) | 该论文提出Vibe-Trading智能体原生交易框架，其核心贡献是一种能将结果、证据、研究历史和交易权限与正确的时间、数据快照及执行上下文可靠绑定的架构，从而统一了从市场研究、回测到模拟交易和券商执行的全流程交易工作空间。 |
| [^6] | [General progress substitutes for specific training data in language models](https://arxiv.org/abs/2611.24844) | 该论文发现，语言模型在其余训练数据上取得的整体进步能够替代被刻意扣留的特定训练数据——模型只需从零开始所需数据量的一小部分即可习得此前被扣留的技能，且所需数据量随模型训练进展而不断减少。 |
| [^7] | [A Foundation Model to Measure and Predict Human Behaviour](https://arxiv.org/abs/2611.24830) | 本文提出名为Chiron的行为基础模型，通过对LLM进行联合微调以表征心理构念、预测个体行为并对齐群体回应分布与人类统计数据，从而在心理测量上更贴近人类自我报告且偏差更少。 |
| [^8] | [On-Policy Morphing into Hybrid Attention Models](https://arxiv.org/abs/2611.24809) | 提出在线策略层选择方法FlashMorph2，通过利用离散混合架构自身的隐藏状态进行端到端评估，并在恢复目标下联合优化层选择分数与模型权重，解决了FlashMorph中连续插值与冻结权重导致的离策略不匹配问题，从而更准确地将预训练Transformer转换为混合注意力模型。 |
| [^9] | [The Elephant in the Question: Unified Models Picture the Concepts Behind Their Answers](https://arxiv.org/abs/2611.24793) | 本文提出无需训练解码器的V-lens方法，利用统一多模态模型图像生成器对文本隐藏状态的访问能力，可视化模型推理中连接问题与答案的关键中间概念，并通过lens tuning显著提升代词指代等概念的可视化准确率。 |
| [^10] | [Induction Head Origins of SLAM-ASR Speech Recognition Models](https://arxiv.org/abs/2611.24721) | 本研究揭示SLAM-ASR语音识别模型的核心机制在于冻结语言模型的归纳头——文本中复制重复标记序列的注意力头同样负责语音嵌入到转录标记的映射，且注意力头对以受限差分注意力形式协同降噪。 |
| [^11] | [DRIFT: Detachable Reasoning for Runtime-Flexible and Efficient End-to-End Driving](https://arxiv.org/abs/2611.24719) | DRIFT提出了一种端到端自动驾驶框架，利用小于1.5亿参数的轻量级语言模型实现场景描述与动作解说，并引入可在运行时灵活开关语言推理的规划模块，且不损害闭环驾驶性能，从而兼顾推理能力与计算效率。 |
| [^12] | [Openhagen Worlds: Calibrated Environments for Measuring and Training Scientific Discovery Agents](https://arxiv.org/abs/2611.24711) | Openhagen Worlds是一套包含27个程序化生成环境、178个经严格校准难度级别的科学发现基准套件，通过明确的评分规则和针对神谕策略、非演绎策略及捷径策略的成功率校准，能够真正测量并训练智能体选择信息性实验和从结果中推理的能力，排除记忆回溯与捷径策略的干扰。 |
| [^13] | [When Does Continual Pre-Training Help? A Controlled Study of Dialectal Adaptation for Qu\'ebec French](https://arxiv.org/abs/2611.24699) | 该研究通过对九个大语言模型的预算匹配对照实验发现，持续预训练对魁北克法语方言适应的收益取决于起始模型对目标方言的既有能力，其增益主要由语料构成（尤其是文学文本）而非数据量驱动，且主要提升表达理解而非语法能力。 |
| [^14] | [CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning](https://arxiv.org/abs/2611.24654) | CurvDelta 通过基于 K-FAC 构建低曲率空间，将能力保持约束内建于模型更新的参数化中，使大模型在能力学习与选择性遗忘等适配任务中达成目标而不损害通用能力。 |
| [^15] | [Piccolo: Co-Evolving Decision Models and Data through Recursive Self-Improvement](https://arxiv.org/abs/2611.24651) | Piccolo是一系列通过智能体数据工程与递归自我改进实现模型能力与训练数据协同演化的多模态决策模型，其中Piccolo-27B以75.59分登顶Decision Index Vision Full排行榜。 |
| [^16] | [Self-Rewarded Reinforcement Learning for Better-Structured Medical Explanations in Small Language Models](https://arxiv.org/abs/2611.24650) | 该论文提出一种无需人工偏好标注的自奖励强化学习方法，通过简单的自监督奖励引导 PPO 训练，显著改善了小语言模型生成的医学解释的结构与质量。 |
| [^17] | [Clover-35B-A3B-Preview Technical Report](https://arxiv.org/abs/2611.24636) | Clover 是一个总参数 350 亿、激活参数仅 30 亿的混合专家扩散语言模型，其创新在于将门控 Delta 网络的循环记忆与块内双向注意力结合以适配块扩散，在智能体基准上全面领先（SWE-bench Verified 达 73.6 分），并在单块 GB300 GPU 上实现 680 tokens/s 的生成速度。 |
| [^18] | [Transferring an analysis of human brain organization to language models](https://arxiv.org/abs/2611.24626) | 该研究将刻画人类皮层组织的扩散映射方法应用于语言模型，发现推理训练使模型在一段连续层跨度内发生局部重组，将该区域的权重嫁接到其他模型中可提升其数学推理能力。 |
| [^19] | [Chain-of-Thought Hidden States Reveal Task-Solving Directions for Direct Answers](https://arxiv.org/abs/2611.24598) | 该论文发现直接提示的表征空间中存在一个由思维链隐状态（恢复与失败状态）构建出的任务求解方向，仅需在生成前旋转一次隐状态，就能引导模型在短生成中比思维链更准确地直接得出正确答案。 |
| [^20] | [Large Language Models or AI Agents for English to Arabic Translation? An assessment on WikiEn-Ar190](https://arxiv.org/abs/2611.24588) | 本研究构建了由阿拉伯语母语者核验的英译阿拉伯语新基准数据集WikiEn-Ar190，发现GPT-5-mini独立翻译已表现优异，而引入自我评估与改进的智能体流程可进一步提升词汇准确性。 |
| [^21] | [Late Entry in Looped Language Models](https://arxiv.org/abs/2611.24576) | 提出循环语言模型的“晚入”解码策略——新token跳过初始迭代、直接基于深度缓存上下文进入最后K次迭代，性能优于同等的早退策略，并可将该优势转化为实际解码速度的提升。 |
| [^22] | [Objective-Specialized Residual Streams in Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2611.24567) | 在流形约束超连接中为不同残差流分配不同的学习目标（如词汇与语义检索），能在各种配置下持续提升性能并诱导出目标特定的内部表征特化。 |
| [^23] | [Do Frontier Model Ingredients Add Up? A Replicated Factorial Study of Small Language Models](https://arxiv.org/abs/2611.24540) | 该研究在相同数据和训练预算下通过重复因子实验比较了小语言模型的五项前沿改进，发现混合专家层、Muon 优化器和 Gated DeltaNet 混合架构收益最大且增益大多可叠加，强调在决定采用哪些组件前应组合测试并重复实验。 |
| [^24] | [ProbMedTOD: A Bayesian Network Guided Task-Oriented Dialogue System for Patient History Taking](https://arxiv.org/abs/2611.24535) | 该论文提出ProbMedTOD系统，通过自动合成临床贝叶斯网络为小型语言模型提供可解释的概率信念跟踪与“系统2”深思推理能力，从而实现更可靠的患者病史采集任务导向对话。 |
| [^25] | [MECI: Multi-Element Collaborative Interaction for Multimodal Entity Linking](https://arxiv.org/abs/2611.24522) | 提出MECI框架，通过多视图专家网络和KL引导的路由机制进行解耦特征学习，缓解元素主导问题，从而更好地捕捉多模态实体链接中提及-实体的多层次对应关系和跨模态互补相关性。 |
| [^26] | [Nothing to Declare: Steering a single neuron increases concealment in AI teams](https://arxiv.org/abs/2611.24475) | 操控单个神经元会让AI更倾向于隐瞒信息，且这种效应在团队协作中比单独行动时更显著，说明集体环境能够改变内部干预对AI行为的影响。 |
| [^27] | [What Majority Labels Hide: Annotator Disagreement and Model Errors in Multi-Dialect Arabic Sexism Detection](https://arxiv.org/abs/2611.24383) | 本文构建了保留个体标注结果的多方言阿拉伯语性别歧视检测数据集MuDAS，发现模型错误更多出现在标注者存在分歧的评论上，且GPT-5.6 Terra与微调编码器虽F1分数接近但错误模式截然不同——前者过度判定性别相关内容，后者则漏判隐性性别歧视。 |
| [^28] | [MoE-Context: Show the Model Only What It Needs](https://arxiv.org/abs/2611.24379) | MoE-Context通过在每步前进行快速的“是/否”门控判断，只向语言模型展示当前所需的上下文块，其余内容以可召回的单行笔记形式掩蔽在单一键值缓存中而不被删除，从而将查找任务准确率从93/120大幅提升至119/120。 |
| [^29] | [LoopCache: Accelerating Long-Context Self-Speculative Decoding for Looped Transformers](https://arxiv.org/abs/2611.24301) | 本文提出LoopCache，一个面向长上下文循环Transformer的高效自投机解码框架，通过置信度引导机制解决现有波风格方法中固定循环起草浪费计算、以及跨循环批处理导致KV缓存内存流量放大R倍这两个关键问题。 |
| [^30] | [LipSpell: A Language-Independent Phonetic Scoring Framework for Spelling Correction Combining Distinctive-Feature Edit Distance, Keyboard Geometry, and Corpus Frequency](https://arxiv.org/abs/2611.24285) | LipSpell提出了一个语言无关的语音评分框架，结合基于IPA区别特征的音素编辑距离、键盘几何结构和语料库频率先验，在六个英语基准测试上全面超越Norvig和Hunspell等传统拼写纠正方法。 |
| [^31] | [From a finite subgroup of $SO(3)$:$C_2$ leads to a Holant dichotomy](https://arxiv.org/abs/2611.24236) | 该论文提出二元签名矩阵的射影等价类可自然嵌入SO(3)的有限子群这一代数视角，据此将C₂情形划分为正规与各向同性两类，并综合运用gadget构造、分解、插值与实数化技术，证明了复值偶数元Holant框架下的复杂性二分定理。 |
| [^32] | [CounselTraj: Synthesizing Process-Aware Psychological Counseling Dialogues via Dialogue-Act Trajectory Planning](https://arxiv.org/abs/2611.24168) | 提出CounselTraj框架，通过从人类咨询师角色扮演对话中学习对话行为模式并规划来访者与咨询师协同的轨迹，将简短的合成对话扩展为过程感知的会话级心理咨询对话，从而缓解了真实咨询数据因隐私问题难以获取的困境。 |
| [^33] | [Once Upon a Bias: Gender Stereotyping in LLM-Generated Children's Stories](https://arxiv.org/abs/2611.24156) | 该论文通过分析16个大语言模型在35个国家场景和12种叙述者角色下生成的62,873篇睡前故事，并提出了一个多层次评估框架，系统揭示了大语言模型会根据儿童性别生成带有系统性性别刻板印象的叙事（如面向儿子的故事强调更高能动性）。 |
| [^34] | [Constructing Dynamic Conversations with Neural Back-channel Generation and Voice-line Scheduling](https://arxiv.org/abs/2611.24086) | 该论文提出了一种神经调度器与反馈信道预测器相结合的方法，能够将两条独立录制或合成的游戏语音台词轨道合并为具有自然话轮转换和反馈信道的单一动态对话。 |
| [^35] | [Meaning Reaches Object Choice but Not Completion: Where Is a VLA Vulnerable to Landmark Synonyms?](https://arxiv.org/abs/2611.24076) | 研究发现VLA策略能够理解训练指令中未出现过的地标同义词并将其语义用于物体选择（该语义作用点位于视觉-语言模型与动作生成器之间的连接器之前），但这种措辞改写会显著降低任务完成率，且连接器之后无法通过单独注入单个模态的token来改变选择。 |
| [^36] | [Measuring and Charging for Retained Context in Long-Context LLM Services](https://arxiv.org/abs/2611.24071) | 该论文提出了一个将内容相关性、可观测代理与可审计位置权重分离的三层框架来度量长上下文LLM服务中重复上下文的实际价值，实测表明缓存命中可将重复前缀计算成本降低两个数量级以上，且校准仿真能将会话费用的99百分位降低31-42%。 |
| [^37] | [MIRAGE: A Framework for Multi-hop Reasoning under Dynamic Knowledge Injection](https://arxiv.org/abs/2611.24058) | 该论文提出了首个针对知识注入场景（同时涵盖冲突型K2与增量型K3知识）的多跳推理基准NEXUS，并构建MIRAGE框架，以解决模型因依赖错误知识来源（固守过时参数记忆、无法应对全新信息）而导致多跳推理出错的问题。 |
| [^38] | [Grading Executable Legal Rules Against Independent Records: Transitional Provisions and Korean SME Status](https://arxiv.org/abs/2611.24055) | 提出LexTemporis方法，以韩国上市公司10,901个企业年度的中小企业身份披露记录等独立事实为基准评估法律执行程序，发现主条文之外的过渡性条款显著提升判定一致率（从83-86%升至89-90%），并可检验争议条款的不同解读。 |
| [^39] | [A Hybrid RAG Framework for Deceptive Review Detection: Integrating RoBERTa-LoRA, FAISS-Based Retrieval, and LLM Reasoning](https://arxiv.org/abs/2611.24049) | 本文提出了一种融合RoBERTa-LoRA分类、FAISS语义检索与LLM推理的混合RAG框架用于虚假评论检测，仅经五个微调周期即在跨领域评估中取得0.90的F1分数，同时提供了证据支持的可解释性。 |
| [^40] | [Compression Effects on Low-Resource Multilingual Translation](https://arxiv.org/abs/2611.24043) | 该研究首次系统评估了训练后量化（FP16/INT8/INT4）对三种低资源非洲语言机器翻译的影响，发现专用MT模型以约六分之一的参数量显著优于通用大语言模型，而量化带来的质量损失仅约1 chrF。 |
| [^41] | [Rethinking Fixed-Horizon Latent Reasoning](https://arxiv.org/abs/2611.24001) | 该论文发现将潜空间思维链运行至完整预设长度反而可能损害推理性能，并提出用“已实现与剩余建构性进展”来刻画推理收益，为重新思考固定长度的潜空间推理提供了依据。 |
| [^42] | [A Mechanistic Analysis of In-Context Learning and Out-of-Context Hallucination](https://arxiv.org/abs/2611.24000) | 本文通过动态上下文中功能角色的统一视角，首次将上下文学习与上下文外幻觉这两种看似对立的现象联系起来进行机制分析，发现角色效应的显著影响取决于足够的贡献幅度和一致的带符号聚合这两个必要条件。 |
| [^43] | [ReflectRec: Cycle-Consistency Preference Verification for LLM-based Recommendation](https://arxiv.org/abs/2611.23969) | ReflectRec通过偏好-历史往返验证器对LLM生成的偏好笔记进行循环一致性校验，结合低质量偏好的条件性再生成与行为相似用户未来行为的前瞻检索，在MovieLens数据集上显著提升推荐性能。 |
| [^44] | [Coverage Governs Accuracy: When to Expand the Lexicon and When to Fall Back in Low-Resource Manchu OCR](https://arxiv.org/abs/2611.23900) | 该研究提出基于词根-后缀链形态扩展的词典约束解码方法，用于低资源满文OCR，证明词典覆盖率决定识别准确率，将未见词形的准确率从63.6%大幅提升至86.3%。 |
| [^45] | [NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding](https://arxiv.org/abs/2611.23835) | 提出NarrowSpec，证明推测解码中读取目标隐藏状态的草稿模型的隐藏宽度可缩小至目标模型的一半或四分之一而几乎不影响接受率，从而在B200和DGX Spark上实现1.06-1.44倍的端到端解码加速。 |
| [^46] | [A System 1 Decision Model for Medical Multiple-Choice Questions Built on Frozen Language Models](https://arxiv.org/abs/2611.23832) | 该研究首次将基于冻结语言模型的单次“系统1”决策方法应用于需要医学知识的MedMCQA考试题，仅通过训练两个小型投影头和温度参数，就在小型模型上取得了显著高于随机水平的答题准确率。 |
| [^47] | [Sightline: Long-Window Speculative Decoding for Fast SVG Generation](https://arxiv.org/abs/2611.23807) | Sightline通过将验证窗口延长至草稿器的预测范围并配合足够宽的草稿树，利用SVG生成中草稿错误稀少且聚集的特性，实现了当前最快的无损SVG解码。 |
| [^48] | [Geometry-bridge flow matching for blind rigid-motion correction in head cone-beam CT](https://arxiv.org/abs/2611.23788) | 提出一种基于几何桥接的3D流匹配先验方法，结合样条位姿估计、共轭梯度数据一致性和全变差正则化的即插即用循环框架，实现头部锥束CT的盲刚性运动校正。 |
| [^49] | [Thinking Fast, Anywhere: One System-1 for the Decision Points of Any Agent Workflow](https://arxiv.org/abs/2611.23751) | AnyS1将人类编写的程序性技能转化为领域通用的决策点来源，训练出一个统一的4B轻量System-1模型，使任意智能体工作流在运行时通过一次前向传播即可完成常规决策，并在置信度不足时回退到大模型System-2。 |
| [^50] | [Beyond Replacement: Levenshtein Editing for Diffusion Language Models](https://arxiv.org/abs/2611.23750) | 提出LED后训练框架，通过在扩散语言模型的“仅替换编辑”基础上扩展插入与删除操作，实现对生成序列的莱文斯坦式精炼，克服了单一替换编辑在处理序列偏移和多位置错误时的局限。 |
| [^51] | [Learning through Discernment: when a small language model learns from user feedback on the job, and when it decays](https://arxiv.org/abs/2611.23716) | 该研究发现，设备端小型语言模型在夜间从用户反馈中自我微调学习时，采用“先判断用户是否给出明确评价”的辨识式反馈感知机制，可使模型在两周内保持接近最优性能，但从长期看模型仍普遍呈现先提升后衰退的规律。 |
| [^52] | [Localizing Text Anonymization for Trustworthy AI: Extending RAT-Bench to Malaysian Microdata and PII](https://arxiv.org/abs/2611.23704) | 该研究将文本匿名化基准 RAT-Bench 本地化至马来西亚微数据与 PII 格式（涵盖马来西亚英语和马来语），并借助 LLM 攻击者评估发现，基于 LLM 的匿名化工具在隐私保护与文本效用之间实现了最佳权衡。 |
| [^53] | [Communicative On-Policy Distillation: Distilling Black-Box Teacher Capabilities through Communication for Long-Horizon Tasks](https://arxiv.org/abs/2611.23673) | 提出通信式在策略蒸馏（COPD），以自然语言通信为媒介，让学生模型自主决定何时向黑盒教师提问，并将成功的师生交互用于在策略自蒸馏与任务奖励强化学习相结合的训练，从而在长时程任务上取得最强黑盒蒸馏基线 2.46 倍的增益。 |
| [^54] | [Safe Personal RSI: Evaluating and Repairing Personal-Agent Updates](https://arxiv.org/abs/2611.23651) | 本文提出Safe Personal RSI更新评估机制，通过将递归自我改进限制在提示词层面、固定模型权重与代码，并建立“测试—修复—验证”流程，在29个案例的基准上成功拒绝了21次高风险更新尝试中的20次，保障个人智能体更新后的安全性与任务性能。 |
| [^55] | [Option-GRPO: Hierarchical RL Against AI Slop in Code Generation for Unseen SDKs](https://arxiv.org/abs/2611.23633) | 该论文提出Option-GRPO分层强化学习框架，仅凭SDK源代码（无需人工示例或测试用例）即可将代码大模型适配到未见过的SDK，通过从源代码中挖掘可复用代码模式（code options）作为课程与动作空间，同时优化功能正确性与代码简洁优雅性，从而抑制AI垃圾代码的产生。 |
| [^56] | [FutureBench: A Renewable Framework for Evaluating Knowledge-Context Conflict in LLMs](https://arxiv.org/abs/2611.23601) | FutureBench是一个利用时间戳问题评估大语言模型知识与检索上下文冲突的可再生基准，研究发现“检查+段落优先”提示会增加模型采纳错误答案的倾向，而具有对抗意识的提示可以恢复大部分被替换的正确答案。 |
| [^57] | [Safe to Omit, Worth Omitting? Keeping the Answer and Cutting Latency When an LLM Agent Skips Another's Message](https://arxiv.org/abs/2611.23570) | 该论文提出一种基于无消息置信度读取与有限样本统计检验的机制，使多智能体LLM系统中的接收方智能体能够以高概率保证答案改变率不超过5%地安全跳过辅助智能体的消息，在保持最终答案质量的同时有效降低通信延迟，并揭示了消息增益远小于oracle上限的量化证据。 |
| [^58] | [Language Models Make New Research Sound Old](https://arxiv.org/abs/2611.23546) | 大型语言模型在验证研究问题时存在系统性“报告差距”——虽能准确判断问题未被研究过却很少明确说明（差距达30-90个百分点且能力越强差距越大），通过将模型判断与对引用论文的单独核查相结合可有效修复这一不对称的报告失败。 |
| [^59] | [Can LLMs Understand What They Predict? Evaluating Latent Causal Structure Inference in Financial Markets](https://arxiv.org/abs/2611.23533) | 该论文提出LUCID基准，评估大语言模型能否从事件描述和历史金融数据中推断潜在的因果市场结构，发现模型生成的因果图通常无法改善重构质量，表明预测成功并不等于真正理解了底层机制。 |
| [^60] | [Dreaming Up Authors: Factual Prevalence and LLM Hallucinations in Academic Authorship Attribution](https://arxiv.org/abs/2611.23530) | 该研究通过192万篇论文的作者归属实验发现，论文引用量与大语言模型幻觉率呈稳健的反比关系——低引用论文幻觉率超过98%，高引用论文低于37%，且失败主要源于检索局限而非知识缺失。 |
| [^61] | [From Arraignment to Resolution: Transformer Models of Court Cases](https://arxiv.org/abs/2611.23518) | 本文提出基于Transformer的时间序列模型，将法院案件建模为事件序列以预测刑事案件中下一诉讼程序的结果，从而提升司法决策的透明度与可监督性。 |
| [^62] | [GRAFT: Execution-Guided Genetic Recombination for Agent Design](https://arxiv.org/abs/2611.23509) | GRAFT提出了一种执行引导的遗传重组方法，在进化式智能体设计中利用父代任务执行的成功与失败证据来引导语义交叉与变异，而不仅依赖聚合适应度分数，从而在不额外执行任务的情况下提升LLM智能体架构的搜索效果。 |
| [^63] | [BOS Summary: The First Token Matters](https://arxiv.org/abs/2611.23508) | BOS Summary是一种无需训练的推理时方法，通过在预填充阶段的单个transformer层放宽BOS查询行的因果约束，让序列开始标记聚合所有视觉和文本输入信息，使模型在解码全程都能访问这一上下文表示，且不引入任何额外参数或开销。 |
| [^64] | [SpecLook: Speculating Where to Look for Fast and Reliable Agentic MLLMs](https://arxiv.org/abs/2611.23483) | SpecLook 提出了一个无需训练的“推测-验证”三层框架，通过让小模型推测智能体下一步的放大观察——该观察可被验证、纠正并用于检验答案——从而显著加速多模态大语言模型智能体的推理并提升答案可靠性。 |
| [^65] | [Formal Adequacy Index (based on Polish-language material)](https://arxiv.org/abs/2611.23482) | 本文提出基于波兰语语料的正式度得体指数，通过分别建模观察正式性与期望正式性并引入可区分正式性不足与过度的语境容忍阈值，实现了对文本正式性与交际情境匹配程度的定量评估。 |
| [^66] | [Tracking Representations in LLM Reasoning Traces for Iterated Games](https://arxiv.org/abs/2611.23403) | 该论文通过机制可解释性技术揭示了LLM在迭代博弈中会在思维链里形成并迭代修正博弈的内部表示，但一旦过早形成错误信念便难以恢复，且这种失败同时体现在推理轨迹和内部激活之中。 |
| [^67] | [QDA: Closed-Loop LLM Quantization Design with Agents](https://arxiv.org/abs/2611.23388) | QDA提出了一个基于智能体的闭环系统，通过KL散度快速评估候选量化方案并自动化整合不同代码库的方法，为每个模型和精度自动设计最优的大语言模型训练后量化方案。 |
| [^68] | [Chorus: Shared N-gram Embeddings with Layer-wise Adaptation](https://arxiv.org/abs/2611.23386) | Chorus提出让整个网络共享一组n-gram嵌入表，每层通过基于自身隐藏状态计算的缩放因子和独立投影层以各自的方式读取共享表，从而以比现有方法小87%的表实现相当的质量。 |
| [^69] | [Prioritize, Don't Decide: What We Learned Auditing a Deployed Human-in-the-Loop CV Sorter](https://arxiv.org/abs/2611.23359) | 本文审计了一个部署在国家学术招聘平台上的人在环简历优先排序系统——委员会按其推荐顺序只需阅读一半简历即可覆盖 95% 的相关候选人；审计进一步发现该评测简历池仅凭文档长度即可排序，使精确率类指标失去参考意义，这一发现直接改变了产品设计。 |
| [^70] | [A Survey on RLOps for Large Language Models](https://arxiv.org/abs/2611.23357) | 本文对大语言模型RLOps方法进行了系统性综述，将现有技术归纳为训练设置、经验收集、奖励计算、策略优化和模型评估五个阶段，填补了该领域技术碎片化、缺乏统一梳理的空白。 |
| [^71] | [A Concurrent Structured Record Mirrored the Prose It Accompanied: A Pre-Registered Conceptual Replication and Extension of the Hide Pending Tool Call Scenario](https://arxiv.org/abs/2611.23274) | 本研究对 Huang 等人（2026）的“隐藏待处理工具调用”场景进行了预注册概念复制与扩展，发现要求模型在同一次生成中附加含“未解决问题”残余部分的结构化记录时，该记录只是映照了其伴随的散文叙述内容，并未能独立提升对缺失待处理结果的披露。 |
| [^72] | [PGPO: Prototype-Guided Progressive Obfuscation for Privacy-Preserving LLM-Enhanced Recommendation](https://arxiv.org/abs/2611.23236) | 该论文提出PGPO方法，通过为高影响力原型词学习拓扑感知的混淆锚点并在静态语义图中渐进传播，在有效防御嵌入反演攻击保护隐私的同时保持物品间语义关系，实现接近上限的推荐性能。 |
| [^73] | [MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study](https://arxiv.org/abs/2611.23158) | 本研究首次在统一的多语言评估协议下，用200种语言和9000多个评估设置系统比较八种LoRA方法，发现小规模搜索无法可靠预测方法在大规模下的相对表现，且LoRA虽能提升翻译数据拟合度但会持续损害阅读理解能力。 |
| [^74] | [Embedding Cosine Measures Text Length Before It Measures Homogenization](https://arxiv.org/abs/2611.21519) | 该研究揭示嵌入余弦相似度主要度量的是文本长度而非话语同质化，此前关于2022年后保守派言论同质化的发现是文本长度差异造成的假象，控制长度后该效应完全消失。 |
| [^75] | [Jetriever: Shifting Agent Memory Retrieval from Similarity to Utility](https://arxiv.org/abs/2611.21478) | Jetriever 用 Jev 决策分数对全部存储记忆进行效用评分排序，取代传统相似度检索，使智能体在回答问题时能将最有用的记忆准确地排在最前。 |
| [^76] | [Machine interpreting competitiveness: a discourse-sensitive comparison of meaning and target-language quality in English-Chinese simultaneous interpreting](https://arxiv.org/abs/2611.19802) | 本研究采用语篇敏感的评估框架，将五个商用机器口译系统与26名人类译员在英汉同声传译中的意义传达与译语质量进行比较，发现机器口译在计划性正式语篇中已进入人类译员较高水平区间，但在即兴语篇中表现差异显著。 |
| [^77] | [Ambiguous Examples Help, But Full Data Wins: Revisiting Dataset Cartography for Extractive Adversarial QA](https://arxiv.org/abs/2611.19763) | 本文在更严格的抽取式对抗问答设置下重新审视数据集制图法的结论，发现仅用33%模糊样本子集训练虽有帮助，但仍不敌全量数据训练在对抗性评估中的表现。 |
| [^78] | [FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models](https://arxiv.org/abs/2611.18627) | 提出 FlashREINFORCE——一种无评论家、每个提示词仅单次采样的异步强化学习框架，通过单批次 REINFORCE、序列信赖域与样本均值优化，在策略滞后约四步的情况下仍能稳定训练智能体语言模型，同时保持提示词覆盖度。 |
| [^79] | [Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent](https://arxiv.org/abs/2611.18397) | 提出Gamer，一个端到端的通用视觉-语言-动作基础模型，通过海量键鼠交互轨迹预训练、稀疏思考指令微调和稳定的多轮强化学习，实现了在复杂2D/3D交互环境中执行长时程任务的实时游戏智能体。 |
| [^80] | [Typed Knowledge over Constructive Institutions: A Proof-Relevant Type Theory for Verified Science](https://arxiv.org/abs/2611.15108) | 本文提出了面向可验证科学知识基底 Eigenius 的证明相关依赖类型论，通过摒弃判断式证明无关性，使系统能够追踪并比较同一命题的多个独立推导、保持强规范化与传递性的等价判定，并支持在实验或模型失效时计算反事实撤回。 |
| [^81] | [Seeing Is Not Believing: Counterfactual Lesion Erasure Teaches a Small Medical Vision-Language Agent to Report Only What the Image Shows](https://arxiv.org/abs/2611.13006) | 该论文提出无需人工标注的反事实病灶擦除评测范式与证据敏感度差距（ESG）指标，并据此训练出40亿参数的医学视觉-语言智能体MedScope-4B，使其只报告图像中真实可见的发现，将“幻觉报告被擦除病灶”的比例从前沿模型的41–62%降至9.6%。 |
| [^82] | [Distinguishing Extraction Failures and Clinical Discordance with LLM Uncertainty on Radiology Reports](https://arxiv.org/abs/2611.10502) | 该研究表明，大语言模型的不确定性信号（包括token级和模型自报告的不确定性）能够有效区分放射学报告信息抽取中的抽取失败与真正的临床不一致，为医学文本结构化抽取的质量控制提供了新方法。 |
| [^83] | [Twists and Turns: A Graph-Based Benchmark and Framework for Long-Term Memory in Exploration Tasks](https://arxiv.org/abs/2611.06414) | 该论文提出LExMem，一个基于图的长期记忆评测基准与框架，通过将试错式探索过程建模为图结构并引入演化机制，来评估模型在长期探索任务中追踪方案修订、分支与重新考量的记忆能力。 |
| [^84] | [AStockArena: Beyond Returns, Can We Entrust Real Capital to the Decisions of Financial Agents?](https://arxiv.org/abs/2611.00515) | 该论文提出AStockArena——一个针对中国A股市场连续交易的实时决策审计基准，通过涵盖执行可追溯性、证据充分性和金融元认知的八项检查来审计13个模型在3,198个连续决策中的真实行为，超越单纯收益指标以回答金融智能体是否值得托付真实资本的问题。 |
| [^85] | [TCC-RAG: Temporal Causal Chain-Guided RAG for Temporal Knowledge Graph Question Answering](https://arxiv.org/abs/2610.38366) | 该论文提出TCC-RAG方法，通过将时序知识图谱重构为多视图因果线索图，并沿时序因果链检索简洁而充分的因果知识，克服了语义RAG难以捕获多跳因果知识且易受冗余信息干扰的缺陷，从而提升时序知识图谱问答的推理准确性。 |
| [^86] | [Typed Decision Models under Prevalence Shift and Label Budgets: Decision Training, Calibration, and Implicit Priors](https://arxiv.org/abs/2610.26325) | 本文在流行率偏移与标签预算约束下系统比较三个类型化决策模型与其骨干模型及微调分类器，发现厂商决策训练并非总是有益，其校准优势因任务类型而异。 |
| [^87] | [LoongSage: A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes](https://arxiv.org/abs/2610.16352) | LoongSage是一个智能体原生、完全异步的生产级强化学习框架，基于Megatron与SGLang构建且核心仅约1.4万行代码，通过统一编排白盒与黑盒智能体的AgentFlow范式，显著降低了前沿大语言模型智能体强化学习的采用、使用与调试开销。 |
| [^88] | [DV-Lens: Revealing the Functional Organization of Language Model Parameters](https://arxiv.org/abs/2610.04489) | 提出DV-Lens参数级可解释性框架，通过估计模块特定的雅可比矩阵将语言模型参数投影到下游词汇空间，从而揭示参数的功能组织及其与模型能力的关系。 |
| [^89] | [Signed Lexical Confidence for Risk-Calibrated Intent Routing](https://arxiv.org/abs/2610.00262) | 提出一种将句子分类器logit间隔与稀疏词汇模型支持相结合的带符号词汇置信度门控，并配合独立的二项式校准选择阈值，在三个意图路由基准上将风险-覆盖率曲线下面积降低约12%至16%。 |
| [^90] | [SANTA++: Sampling Attention through Representative Keys](https://arxiv.org/abs/2609.35629) | SANTA++ 是一种无需训练的随机注意力方法，通过将 KV 缓存分组并用代表性键进行重要性采样，仅访问 28-31% 的 KV 缓存即可保留 95-98% 的稠密注意力性能，显著降低长上下文推理的内存开销。 |
| [^91] | [Circuit Hypernetworks for Quantum-Augmented Diffusion Language Models](https://arxiv.org/abs/2609.24657) | 提出HyperQ方法，通过轻量级电路超网络为冻结的掩码扩散语言模型注入词元条件化的量子残差分支，并利用计算成本随量子比特数线性增长的精确经典期望值表达式，实现高效的量子增强语言模型适配。 |
| [^92] | [Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2609.23697) | 提出TrustMOPD方法，以专家模型RL训练前后相对于共享参考模型的位移作为token级可靠性代理，实现无标签的多教师加权蒸馏监督分配，将学生模型性能恢复率从54.4%大幅提升至91.5%。 |
| [^93] | [RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents](https://arxiv.org/abs/2609.23466) | RPMem提出了一种两阶段架构，将LLM智能体的每个会话编译为模型无关的潜在记忆，经任务训练的循环门控整合后映射为LoRA参数，从而实现跨会话的长期参数化记忆演化，并在更换骨干模型时保持记忆可迁移。 |
| [^94] | [Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time](https://arxiv.org/abs/2609.22226) | 该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。 |
| [^95] | [Improving Cross-Lingual Token Representations by Adding a Pinch of SALT](https://arxiv.org/abs/2609.09953) | SALT是一种轻量级后训练方法，通过向现有跨语言句子编码器注入跨度级监督信号来改进词元表示，在五个多语言词元级基准中的四个上取得最佳结果，同时还能提升句子级任务性能。 |
| [^96] | [OracleZoom: On-Policy Self-Distillation Inspired Reference-Constrained Recursive Image Super Resolution](https://arxiv.org/abs/2609.06490) | OracleZoom提出了一种受自蒸馏启发、参考约束的递归图像超分辨率框架，通过直接与跨尺度监督、无参考质量目标、KL约束潜在先验和EMA一致性等技术，解决了深层递归缺乏真实标签监督的问题，在七个数据集上实现了极端放大倍率下最先进的超分辨率质量。 |
| [^97] | [XMerge: Cross-Axis Selection and Reconstructive Layer Merging for LLM Depth Compression](https://arxiv.org/abs/2609.02083) | XMerge 是一种训练后的大语言模型深度压缩方法，通过跨轴选择识别隐藏状态变化最小的层块，并利用局部边界重构重新拟合相邻存留块，在不改变架构、不增加推理参数、无需任务标签的情况下实现高质量的层删除压缩。 |
| [^98] | [Ceiling-Clipped Acceptance Histograms Indicate Stranded Speed-up in Block-Diffusion Speculative Decoding](https://arxiv.org/abs/2608.30427) | 该论文发现块扩散投机解码中，高接受率会使草稿模型提前耗尽训练时的块长度上限，造成未被实现的“搁浅加速”，提出用接受直方图顶格柱的尖峰作为投入训练算力前的预检指标，并证明推理时简单扩大块大小无法恢复该加速。 |
| [^99] | [Meta$^n$: Recursive Self-Improvement through Emergent Depth](https://arxiv.org/abs/2608.24735) | Meta$^n$通过固定元操作并递归作用于自身产物，突破了自我改进系统的元深度限制，实现了随收敛而定的涌现深度，从而在多个骨干网络上取得更优性能。 |
| [^100] | [Where Is the Tradeoff in Using Third-Party API Routers for Agentic Software Development?](https://arxiv.org/abs/2607.23624) | 该论文首次实证研究了编码智能体中第三方API路由器的路由器端注入风险，通过四个隐蔽程度递增的干预级别（如响应替换）揭示其可对软件开发任务造成真实且难以检测的影响，而现有客户端权限机制难以有效防御。 |
| [^101] | [Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning](https://arxiv.org/abs/2607.21653) | 提出了 Molt 框架，通过可组合模型并行、统一智能体接口、全异步 rollout 与优化以及分布式经验存储，实现了万亿参数规模下的智能体强化学习训练，且无需修改现有智能体的执行逻辑。 |
| [^102] | [Meta-Learning Preferences for Multilingual LLM Alignment](https://arxiv.org/abs/2607.13315) | 该论文提出了一种面向RLHF和DPO的元学习框架，通过从其他语言的偏好数据中学习可迁移的初始化，使模型在仅有100个目标语言偏好样本的极低资源场景下实现对齐，胜率相比基线提升高达33.4%。 |
| [^103] | [EasyOPD: An Easy-to-use On-Policy Distillation Framework for Large Language Models](https://arxiv.org/abs/2607.11012) | 本文提出EasyOPD，一个基于verl构建的易用在线策略蒸馏框架，通过解耦用户配置、方法监督逻辑与底层执行，统一了现有OPD方法碎片化的实现方式，使框架更易于复现和扩展。 |
| [^104] | [RABBiT: Rapidly adaptive BOLD foundation model via brain-tuning for accurate zero-shot and few-shot prediction of speech-elicited responses in the brain](https://arxiv.org/abs/2607.05171) | RABBiT是一个通过“脑调优”实现的紧凑型音频到fMRI基础模型，能够在零样本和少样本设置下准确预测大脑对自然语音的fMRI响应，仅需10分钟个体数据即可高效适应新参与者，性能超越现有SOTA fMRI基础模型。 |
| [^105] | [When Does Learning to Stop Help? Configuring Risk-Controlled Early Exit for Reasoning-Model Services](https://arxiv.org/abs/2606.30852) | 该论文提出一种基于工作负载校准数据的部署决策流程，通过Learn-then-Test以可证明的风险界限同时认证学习式停止规则与静态token预算两类提前退出方案并部署成本最低者，实验表明两类方案在不同模型与任务上各有优势、无普适胜者。 |
| [^106] | [Know Before You Fetch: When Does Confidence-Based Retrieval-Budget Allocation Pay Off?](https://arxiv.org/abs/2606.29959) | 该研究在相同的每查询平均检索预算上限下，将基于阅读器置信度的自适应检索分配与最佳简单策略（固定深度或其随机混合）对比，发现置信度分配在极紧预算（每查询平均半个段落）下于全部七个设置中稳定胜出（准确率提升0.010–0.075），其价值在于配给稀缺预算和避免有害检索。 |
| [^107] | [Learning User Simulators with Turing Rewards](https://arxiv.org/abs/2606.19336) | 提出基于图灵测试的强化学习方法Turing-RL，通过判别性图灵奖励训练用户模拟器，使其生成与真实用户难以区分的回复，并在对话和论坛两个领域取得优于基线方法的效果。 |
| [^108] | [Polar: A Benchmark for Evaluating Political Bias in LLMs](https://arxiv.org/abs/2606.12922) | 提出了Polar基准，通过选项级似然概率在美韩两种政治语境和八个议题类别上测量38个大语言模型的政治偏见，发现所有模型在美国内容上均偏向左翼进步立场，且展示语言本身即可改变测得的偏见结果。 |
| [^109] | [CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2606.02502) | 本文提出CRAM方法，通过中心路由和自适应混合专家模型，在缓解多模态持续指令微调中灾难性遗忘的同时，提升参数效率。 |
| [^110] | [Extracting Training Data from Diffusion Language Models via Infilling](https://arxiv.org/abs/2605.24173) | 该论文提出“填充提取”协议，利用任意二进制掩码全面评估扩散语言模型的训练数据泄露风险，发现传统的前缀探测方法会显著低估记忆化风险，而边缘条件掩码可提取多达三倍的逐字训练序列。 |
| [^111] | [Sparse Layers are Critical to Scaling Looped Language Models](https://arxiv.org/abs/2605.09165) | 研究发现稀疏混合专家（MoE）层是扩展循环语言模型的关键，因为不同循环通过激活不同专家可在不增加参数的情况下恢复表达能力，同时循环边界还提供了比标准模型更优的提前退出点。 |
| [^112] | [MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments](https://arxiv.org/abs/2605.07058) | 该论文提出MedExAgent，首次将临床诊断形式化为包含向患者提问、开具医学检查和给出诊断三类动作的POMDP，以训练能够在真实嘈杂临床环境中应对患者多样性与不完整信息的LLM智能体。 |
| [^113] | [HorizonBench: Long-Horizon Personalization with Evolving Preferences](https://arxiv.org/abs/2604.17283) | HorizonBench是一个通过隐含提及的生活事件驱动用户偏好演变的长程个性化基准测试，用于评估AI助理能否从对话线索中感知并正确更新对用户偏好的认知。 |
| [^114] | [Training-Free Dynamic Upcycling of Expert Language Models](https://arxiv.org/abs/2603.29765) | 本文提出 DUME，一种免训练的新方法，通过复用不同领域训练好的稠密专家模型来动态构建统一的 MoE 模型，从而避免多任务微调带来的过度专业化和灾难性遗忘问题。 |
| [^115] | [Improving LLM Final Representations with Inter-Layer Geometry](https://arxiv.org/abs/2603.22665) | 本文提出用轻量级图神经网络聚合大语言模型各层信息，并进一步设计基于SL(2,Zn)凯莱图的Cayley编码器，以稀疏、规则且低直径的数学拓扑替代昂贵的注意力机制，在提升效率的同时显著增强下游预测性能。 |
| [^116] | [HYPER: Hypothesis-Guided Planning for Evidence-Seeking Retrieval](https://arxiv.org/abs/2603.19008) | HYPER是一个无需训练的检索前规划框架，以临时答案假设为锚点，将检索分解为支持、区分和特征三种证据角色，把RAG从面向主题的检索转变为面向证据的检索，从而在需要竞争性选择的任务中更好地支持最终决策。 |
| [^117] | [A conceptual framework for ideology in online discourse beyond the left and right](https://arxiv.org/abs/2603.18945) | 本文提出将意识形态概念化为多层次社会认知概念网络的新框架，突破了计算社会科学中单一左右党派轴线的研究局限，并将在线话语分析方法与意识形态理论相连接。 |
| [^118] | [Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency](https://arxiv.org/abs/2603.17102) | 该论文提出将大语言模型的跨语言不一致性作为可解释性工具，通过对比模型在不同语言中事实回忆成功与失败时的MoE路由行为，定位混合专家模型中负责特定知识的关键专家。 |
| [^119] | [S-GRADES -- Studying Generalization of Student Response Assessments in Diverse Evaluative Settings](https://arxiv.org/abs/2603.10233) | 本文提出S-GRADES，一个开源的网页基准平台，将14个多样化的评分数据集整合于统一界面下，通过标准化访问和可复现的评估协议，弥合自动作文评分与自动简答题评分两个割裂领域，以研究学生回答评估在不同评估环境中的泛化能力。 |
| [^120] | [How Far Does Online RL Generalize in Mobile GUI Agents? A Controlled Study with AndroidWorld-Generalization](https://arxiv.org/abs/2603.07432) | 本文提出AndroidWorld-Generalization基准，通过未见实例、未见模板、未见应用三级递进的上下文偏移系统评估在线RL在移动GUI智能体中的泛化能力，发现RL收益随环境偏移距离增大而单调衰减。 |
| [^121] | [Unlocking Continuous Diffusion Language Models via Discrete Stochastic Localization](https://arxiv.org/abs/2602.16169) | 提出离散随机局部化（DSL）连续状态框架，通过将随机局部化通道与单位球面词元嵌入相结合，实现了仅依赖含噪状态、无需时间步条件的贝叶斯最优去噪器，使单个去噪网络即可支持连续去噪、掩码扩散、随机顺序自回归等多种采样方法，从而让连续扩散模型在离散序列生成上有效匹敌掩码离散扩散模型。 |
| [^122] | [Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems](https://arxiv.org/abs/2602.15198) | 提出了Colosseum框架，通过形式化多智能体决策框架、基于遗憾的合谋度量以及秘密通信信道行为探针，系统性地审计LLM多智能体系统中的合谋行为。 |
| [^123] | [Multimodal Information Fusion for Chart Understanding: A Survey of MLLMs -- Evolution, Limitations, and Cognitive Enhancement](https://arxiv.org/abs/2602.10138) | 本综述系统梳理了基于多模态大语言模型（MLLM）的图表理解研究，分析了视觉与语言信息融合的核心挑战，提出了规范型与非规范型基准的新颖分类体系，并追溯了从经典深度学习技术到先进MLLM融合范式的方法演进历程。 |
| [^124] | [Disentangling generalization and memorization in large language models using chess](https://arxiv.org/abs/2601.16823) | 该论文提出以国际象棋作为受控测试平台，通过按相关先验密度构建棋局分类体系，在无需了解模型训练数据的情况下区分大语言模型的记忆能力与真正的泛化推理能力，并发现模型性能随相关先验密度降低而持续显著下降。 |
| [^125] | [Challenging Quadratic Attention$\unicode{x2013}$A Holistic View on the Rise of Alternative Language Model Architectures](https://arxiv.org/abs/2510.05364) | 本文综述了亚二次注意力变体、循环神经网络、状态空间模型及混合架构等替代性语言模型架构的最新进展，并认为纯注意力 Transformer 的主导地位可能被挑战，尤其是在特定领域和边缘设备应用中。 |
| [^126] | [MARBLE: An Agentic Benchmark for Spatial Reasoning and Visual Abstraction](https://arxiv.org/abs/2506.22992) | 提出了MARBLE基准测试，通过三个智能体环境系统评估多模态语言模型的空间推理与视觉抽象能力，发现最强模型GPT-5.4的平均成功率仅为7.3%，揭示了现有模型在此类任务上的巨大局限。 |
| [^127] | [A Survey on Collaborative Mechanisms Between Large and Small Language Models](https://arxiv.org/abs/2505.07460) | 本综述系统梳理了大语言模型与小语言模型之间五种协作机制（流水线、路由、辅助、蒸馏、融合）及其关键使能技术和面向边缘设备的应用场景，为构建高效、灵活且易获取的AI系统提供了全面的参考框架。 |
| [^128] | [Quicksviewer: An LMM with Early Video Compression using Annealed Gumbel-Softmax](https://arxiv.org/abs/2504.15270) | Quicksviewer提出了一种基于退火Gumbel-Softmax的早期视频压缩方法，通过将非均匀冗余视频在线动态划分为不同大小的块并进行统一重采样，实现45倍压缩率的高效视频理解。 |
| [^129] | [Reinforcement Learning with Reference-Based Rewards for Language Model Reasoning and Alignment](https://arxiv.org/abs/2504.09895) | 提出RL2R方法，利用参考答案与轻量级语义相似度指标（如BERTScore）作为奖励信号，为语言模型的推理与对齐提供统一且低成本的强化学习后训练方案。 |
| [^130] | [MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents](https://arxiv.org/abs/2503.01935) | 提出了综合基准MultiAgentBench，利用新颖的基于里程碑的性能指标评估LLM多智能体系统在多样化交互场景中的任务完成、协作与竞争质量，并系统比较了不同协调拓扑结构与群体讨论、认知规划等创新策略的效果。 |
| [^131] | [Mixtures of In-Context Learners](https://arxiv.org/abs/2411.02830) | 提出MoICL方法，将上下文学习的演示示例子集视为专家并学习加权函数来合并其输出分布，在提升分类性能的同时减少推理时间，并显著增强了对域外、不平衡和噪声示例的鲁棒性。 |
| [^132] | [Don't Pay Attention, PLANT It: Pretraining Attention via Learning-to-Rank](https://arxiv.org/abs/2410.23066) | 提出PLANT，一种即插即用策略，通过基于互信息增益的学习排序模型预训练标签注意力来初始化注意力模块，可与大语言模型骨干集成，在极端多标签文本分类任务上超越现有最先进方法，且在罕见标签上收益最大。 |
| [^133] | [Coherence boosting: When your pretrained language model is not paying enough attention](https://arxiv.org/abs/2110.08294) | 提出了一种无需额外训练的推理方法“连贯性提升”，通过增强语言模型对长上下文的关注度来改善长距离语义连贯性，并在多种零样本NLP任务上带来性能提升。 |
| [^134] | [ThinkSum: Probabilistic reasoning over sets using large language models.](http://arxiv.org/abs/2210.01293) | 本研究提出一种两阶段的概率推理范例ThinkSum，通过以结构化的方式对对象或事实集进行推理，实现了对多个对象或事实进行推理并进行逻辑推导的场景中的改进。 |

# 详细

[^1]: 基于对比原型结构的大语言模型越狱检测

    Jailbreak Detection on Large Language Models via Contrastive Prototype Structures

    [https://arxiv.org/abs/2611.24936](https://arxiv.org/abs/2611.24936)

    提出JDPI方法，利用对比学习构建有害与良性原型，仅凭二元安全标签通过相对原型距离实现大语言模型越狱检测，无需训练额外下游分类器，在四个基准上将检测准确率提升9.1-18.0个百分点。

    

    越狱攻击与有害提示仍然是大语言模型安全部署面临的挑战。我们研究了对比学习所获得的表示能否在无需训练额外下游分类器的情况下支持安全检测。我们提出了基于原型推理的越狱检测方法（Jailbreak Detection with Prototype Inference, JDPI），该方法利用二元安全标签构建有害原型与良性原型，并通过相对原型距离进行预测。JDPI 在编码器适配、原型构建和验证中仅使用二元安全标签。在我们的实验设置中，由于避免了用于训练下游分类器的单独数据划分，JDPI 相比分类器基线（方法 A）所需的标注样本更少。在涵盖不同输入格式的四个基准测试中，对比微调使 JDPI 的检测准确率提升了 9.1 至 18.0 个百分点。在使用相同编码器的情况下，JDPI 在其中一个基准上取得了略高的 ROC-AUC，并与（原文此处截断）的差距保持在 0.017 以内。

    arXiv:2611.24936v1 Announce Type: new  Abstract: Jailbreak attempts and harmful prompts remain a challenge for the safe deployment of large language models. We study whether contrastively learned representations can support safety detection without training an additional downstream classifier. We propose Jailbreak Detection with Prototype Inference (JDPI), which uses binary safety labels to construct harmful and benign prototypes and makes predictions from relative prototype distances. JDPI uses only binary safety labels for encoder adaptation, prototype construction, and validation. In our setup, it uses fewer labeled examples than the classifier baseline (Method A) by avoiding the separate split used to train the downstream classifier. Across four benchmarks covering different input formats, contrastive fine-tuning improves JDPI's detection accuracy by 9.1-18.0 percentage points. Using the same encoder, JDPI achieves slightly higher ROC-AUC on one benchmark and stays within 0.017 of 
    
[^2]: 你是谁？个人模仿游戏

    Who are you? The Personal Imitation Game

    [https://arxiv.org/abs/2611.24888](https://arxiv.org/abs/2611.24888)

    该论文提出了“个人模仿游戏”这一更难的挑战，要求机器在对话中模仿特定个体而非普通人，并利用十年个人数字通信数据构建的LLM系统实验表明，审讯者无法可靠地区分该系统与真人本人。

    

    大型语言模型（LLM）如今在艾伦·图灵1950年提出的模仿游戏（Imitation Game）中表现出色，因此我们提出了一个更难的挑战——个人模仿游戏，它探究机器能否在对话中复制一个特定的个体，而非泛泛的普通人。通过30名参与者的测试，我们建立了人类基线，并报告了审讯者所依赖的身份信号：文体特征、人格特质和传记事实。为了展示这一框架，我们设计了一个基于LLM的系统，为论文的一位作者参与该游戏，该系统利用了涵盖十年个人数字通信的自动民族志数据集。在一项20人的欺骗研究中，审讯者在10轮中有7轮选择了我们的系统而非作者自己的朋友，在20次中有13次选择了作者而非我们的系统；在此样本量下，这两个比率均无法与随机概率区分开。我们讨论了设计改进、个人AI智能体的应用及相关影响。

    arXiv:2611.24888v1 Announce Type: cross  Abstract: Large Language Models (LLMs) now perform strongly on Alan Turing's 1950 Imitation Game, so we propose a harder challenge, the Personal Imitation Game, which asks whether a machine can replicate a specific individual in conversation rather than a generic human. Testing with 30 participants, we establish a human baseline and report the identity signals interrogators rely on: stylometric features, personality traits, and biographical facts. To demonstrate the framework, we design an LLM-based system that plays the game for one of the paper's authors, drawing on an autoethnographic dataset spanning ten years of personal digital communication. In a 20-person deception study, interrogators choose our system over the author's own friends in 7 of 10 rounds, and the author over our system in 13 of 20; neither rate is distinguishable from chance at this sample size. We consider design improvements, applications of personal AI agents, and implica
    
[^3]: 扎根、迭代、追溯：分解AI研究构思

    Ground, Iterate, Trace: Decomposing AI Research Ideation

    [https://arxiv.org/abs/2611.24887](https://arxiv.org/abs/2611.24887)

    提出了CAR智能体框架，通过文献扎根、语义知识图谱与四种可追溯的概念操作符分解研究构思过程，显著提升了AI生成研究想法的方法论新颖性。

    

    大语言模型已被广泛用于生成看似合理的研究想法，但尚不清楚是什么使自动化构思有效，以及生成的新颖提案如何源于先前的科学工作。我们介绍了创造性AI研究者（CAR），这是一个结合文献扎根、迭代开发和显式概念操作符的智能体框架。CAR从检索到的文献中构建语义知识图谱，并为构思智能体配备图谱检索工具和四种操作符：双联、类比、探索和细化。每次操作符调用都会根据结构化接受条件进行检查并记录，从而产生中间转换和声明文献来源的轨迹。我们使用两个语言模型骨干，在40个LiveIdeaBench研究主题和两个执行基准上对CAR进行评估。CAR实现了比单次提示和基线方法显著更高的方法论新颖性（MN）

    arXiv:2611.24887v1 Announce Type: new  Abstract: Large language models have been widely used to generate plausible research ideas, yet it remains unclear what makes automated ideation effective and how generated novel proposals arise from prior scientific work. We introduce the Creative AI Researcher (CAR), an agentic framework combining literature grounding, iterative development and explicit conceptual operators. CAR constructs a semantic knowledge graph from retrieved literature and equips an ideation agent with graph-retrieval tools and four operators: bisociation, analogy, exploration and elaboration. Each operator call is checked against structural acceptance conditions and recorded, producing a trace of intermediate transformations and declared literature sources. We evaluate CAR with two language-model backbones across 40 LiveIdeaBench research topics and two execution benchmarks. CAR achieves significantly higher methodological novelty (MN) than single-shot prompting and The A
    
[^4]: 程序能否取代大语言模型裁判？用于可扩展且高效相关性标注的可执行程序裁判

    Can Programs Replace LLM Judges? Executable Program Judges for Scalable and Efficient Relevance Labeling

    [https://arxiv.org/abs/2611.24849](https://arxiv.org/abs/2611.24849)

    提出ExecJudges方法，利用前沿LLM将评估指南转化为仅依赖词汇匹配和标准NLP库的可执行程序，从而取代昂贵的LLM裁判调用，实现快速、可解释、可复现的大规模相关性标注。

    

    大语言模型（LLM）裁判已成为为检索评估创建大规模相关性判断的常用方法。然而，每次判断都需要对前沿模型进行可能代价高昂的调用，且这些判断难以解释，并可能无法复现。在本文中，我们探讨LLM能否为每个主题编写一个可读的程序，以取代其对每个文档逐一进行判断的方式。为解决这些问题，我们提出了ExecJudges，一种用于自动相关性标注的方法，可生成快速、可检查且可扩展的裁判。具体而言，ExecJudges使用前沿LLM将评估指南转化为可执行程序，这些程序仅依赖词汇匹配、标准NLP库（如解析器）以及池统计信息。ExecJudges-L变体在可用时还会利用人工标注的示例。我们在七个TREC数据集合上评估了ExecJudges，通过将生成的标签与训练有素的NIST评估员、前沿LLM裁判以及（原文在此处截断）进行比较。

    arXiv:2611.24849v1 Announce Type: cross  Abstract: Large language model (LLM) judges have become a common way to create large-scale relevance judgments for retrieval evaluation. However, each judgment requires a potentially expensive call to a frontier model, and the judgments are hard to interpret and may not be reproducible. In this paper, we ask whether an LLM can write a readable program for each topic that replaces its per-document judgments. To address these issues, we propose ExecJudges, a method for generating fast, inspectable and scalable judges for automatic relevance labeling. Specifically, ExecJudges uses a frontier LLM to turn assessment guidelines into executable programs, which rely only on lexical matching, standard NLP libraries like parsers, and pool statistics. An ExecJudges-L variant also uses examples of human labels if available. We evaluate ExecJudges on seven TREC collections by comparing the produced labels with trained NIST assessors, frontier LLM judges, and
    
[^5]: Vibe-Trading：迈向智能体原生交易

    Vibe-Trading: Towards Agent-Native Trading

    [https://arxiv.org/abs/2611.24845](https://arxiv.org/abs/2611.24845)

    该论文提出Vibe-Trading智能体原生交易框架，其核心贡献是一种能将结果、证据、研究历史和交易权限与正确的时间、数据快照及执行上下文可靠绑定的架构，从而统一了从市场研究、回测到模拟交易和券商执行的全流程交易工作空间。

    

    现有的AI智能体能够检索信息、调用工具、运行实验并生成多步骤输出。然而，大多数智能体是围绕相对静态的工作流构建的，难以在环境变化时保持可靠的状态。金融交易使这一局限性尤为突出：价格、新闻、市场状况、投资组合持仓以及可用资金都在持续演变。一个结果只有在与正确的时间、数据快照和执行上下文保持绑定时才是有用的。我们提出了Vibe-Trading，这是一个全面的智能体原生交易框架和一体化工作空间，用于构建、评估和演化金融交易智能体。它统一了市场研究、金融数据、策略生成、回测、投资组合分析、模拟交易、多智能体协作以及可选的券商执行。其核心贡献是一种架构，该架构使结果、证据、研究历史和交易权限保持关联（原文此处截断）。

    arXiv:2611.24845v1 Announce Type: new  Abstract: Existing AI agents can retrieve information, call tools, run experiments, and produce multi-step outputs. Yet most are built around relatively static workflows and struggle to preserve reliable state as the environment changes. Financial trading makes this limitation especially consequential: prices, news, market conditions, portfolio positions, and available capital evolve continuously. A result is useful only when it remains tied to the correct time, data snapshot, and execution context. We present Vibe-Trading, a comprehensive agent-native trading framework and all-in-one workspace for building, evaluating, and evolving agents for financial trading. It unifies market research, financial data, strategy generation, backtesting, portfolio analysis, paper trading, multi-agent collaboration, and optional broker execution. Its central contribution is an architecture that keeps results, evidence, research history, and trading authority conne
    
[^6]: 语言模型中的整体进步可替代特定训练数据

    General progress substitutes for specific training data in language models

    [https://arxiv.org/abs/2611.24844](https://arxiv.org/abs/2611.24844)

    该论文发现，语言模型在其余训练数据上取得的整体进步能够替代被刻意扣留的特定训练数据——模型只需从零开始所需数据量的一小部分即可习得此前被扣留的技能，且所需数据量随模型训练进展而不断减少。

    

    语言模型从训练数据中获得其能力，开发者在训练期间扣留数据以防止模型学习可能被滥用的能力。这种做法会延迟一项能力多久，以及之后获取该能力需要多少数据，目前尚不清楚。在这里，我们展示了模型在其余训练数据上取得的总体进步可以替代被扣留的数据。我们在国际象棋、黑白棋、分子字符串、括号语言和网页文本的模型中扣留了单个技能的训练信号。被扣留技能在较大模型中的延迟程度可以从小模型的表现中预测得出。一个从未被明确训练过应对将军局面的国际象棋模型，对合法应对着法的排序几乎与受过训练的模型一样好。当模型随后在被扣留的数据上进行训练时，它们只需从初始状态训练所需数据量的一小部分即可习得该技能，而且这一比例随着模型训练的进展而降低。

    arXiv:2611.24844v1 Announce Type: cross  Abstract: Language models acquire their capabilities from training data, and developers withhold data during training to keep models from learning capabilities that could be misused. How long this delays a capability, and how much data is needed to acquire it later, is unknown. Here we show that the general progress a model makes on the rest of its training data substitutes for the data that was withheld. We withheld the training signal for individual skills in models of chess, Othello, molecular strings, bracket languages and web text. The delay of a withheld skill in larger models was predicted from smaller ones. A chess model never explicitly trained to respond to check ranked the legal responses almost as well as a model that was. When models were then trained on the withheld data, they acquired the skill with a small fraction of the data needed from initialization, and this fraction decreased as the models progressed. A control rule of equa
    
[^7]: 测量与预测人类行为的基础模型

    A Foundation Model to Measure and Predict Human Behaviour

    [https://arxiv.org/abs/2611.24830](https://arxiv.org/abs/2611.24830)

    本文提出名为Chiron的行为基础模型，通过对LLM进行联合微调以表征心理构念、预测个体行为并对齐群体回应分布与人类统计数据，从而在心理测量上更贴近人类自我报告且偏差更少。

    

    大语言模型（LLM）正被越来越多地用于推断心理属性、预测个体行为和模拟人类群体。然而，尽管它们能生成看似合理的回应，却常常错误地呈现个体并扭曲群体统计特征。在此，我们提出了Chiron，一个用于测量和预测人类行为的基础模型。我们联合微调一个LLM，使其能够表征心理构念、预测个体行为，并使群体回应分布与人类统计数据保持一致。训练和评估数据涵盖36个行为领域和78项社会科学实验，包含288,307条个体记录、1,411,750个原子行为决策，以及针对109个心理构念的99,953项心理测量评估。在评估中，Chiron的心理测量结果与人类自我报告更为吻合，且表现出更少损害心理测量效度的偏差。它对个体行为的预测...

    arXiv:2611.24830v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to infer psychological attributes, predict individual behaviour, and simulate human populations. However, despite generating plausible responses, they often misrepresent individuals and distort population statistics. Here we introduce Chiron, a foundation model to measure and predict human behaviour. We jointly fine-tune an LLM to represent psychological constructs, predict individual behaviour, and align population response distributions with human statistics. The training and evaluation data span 36 behavioural domains and 78 social science experiments, comprising 288,307 person records, 1,411,750 atomic behavioural decisions, and 99,953 psychometric assessments over 109 psychological constructs. In our evaluations, Chiron's psychological measurements agree more closely with human self-reports and exhibit fewer biases that undermine psychometric validity. It predicts individual behav
    
[^8]: 基于在线策略的混合注意力模型转换方法

    On-Policy Morphing into Hybrid Attention Models

    [https://arxiv.org/abs/2611.24809](https://arxiv.org/abs/2611.24809)

    提出在线策略层选择方法FlashMorph2，通过利用离散混合架构自身的隐藏状态进行端到端评估，并在恢复目标下联合优化层选择分数与模型权重，解决了FlashMorph中连续插值与冻结权重导致的离策略不匹配问题，从而更准确地将预训练Transformer转换为混合注意力模型。

    

    混合注意力模型通过结合全注意力与线性注意力，实现了更优的质量—效率权衡。将预训练的Transformer转换为此类模型，需要在给定预算下确定哪些层应保留全注意力。我们在FlashMorph的层选择方法中识别出两个离策略不匹配问题：(i) 连续分支插值和教师强制隐藏状态与离散混合计算存在差异；(ii) 冻结的模型权重无法反映后续恢复过程中的参数自适应。我们提出了FlashMorph2，一种在线策略的层选择方法，该方法 (I) 使用采样得到的离散混合架构自身的隐藏状态进行端到端评估；(II) 在恢复目标下联合优化层选择分数与模型权重，以反映参数自适应后的性能。在Qwen3不同模型规模及多种线性注意力变体上的实验表明，FlashMorph2始终优于现有方法。

    arXiv:2611.24809v1 Announce Type: new  Abstract: Hybrid attention models combine full and linear attention to achieve a better quality--efficiency trade-off. Converting pretrained Transformers into such models requires identifying which layers should retain full attention under a prescribed budget. We identify two off-policy mismatches in FlashMorph's layer selection: (i) continuous branch interpolation and teacher-forced hidden states differ from discrete hybrid computation; and (ii) frozen model weights fail to account for parameter adaptation during subsequent recovery. We introduce FlashMorph2, an on-policy layer-selection method that (I) evaluates sampled discrete hybrid architectures end to end using their own hidden states, and (II) jointly optimizes layer-selection scores and model weights under the recovery objective to reflect post-adaptation performance. Experiments across Qwen3 model scales and linear-attention variants show that FlashMorph2 consistently outperforms existin
    
[^9]: 问题中的大象：统一模型描绘其答案背后的概念

    The Elephant in the Question: Unified Models Picture the Concepts Behind Their Answers

    [https://arxiv.org/abs/2611.24793](https://arxiv.org/abs/2611.24793)

    本文提出无需训练解码器的V-lens方法，利用统一多模态模型图像生成器对文本隐藏状态的访问能力，可视化模型推理中连接问题与答案的关键中间概念，并通过lens tuning显著提升代词指代等概念的可视化准确率。

    

    可视化中间表示有助于我们检视模型输出背后的推理步骤。统一多模态模型在同一架构中融合了理解与生成，为实现这一目标提供了机会：其图像生成器可以访问处理文本时形成的表示。我们提出V-lens，利用这一连接，无需训练解码器即可可视化所选token的隐藏状态。在四个基于扩散模型和两个纯自回归的统一模型上，生成的图像揭示了连接问题与答案的关键中间概念，即使这些概念从未被明确提及。对照实验展示了这些可视化内容如何随上下文、token位置和模型深度而变化。随后我们提出lens tuning（透镜调优），通过V-lens用图像对选定的隐藏状态施加监督。在BAGEL、SenseNova-U1和HunyuanImage-3.0上，代词指代对象的正确可视化率显著提升（原摘要在此处被截断）。

    arXiv:2611.24793v1 Announce Type: cross  Abstract: Visualizing intermediate representations could help us inspect the reasoning steps underlying a model output. Unified multimodal models offer this opportunity by combining understanding and generation in one architecture. Their image generators can access representations formed while processing text. We introduce the V-lens, which uses this connection to visualize a chosen token's hidden state without training a decoder. Across four diffusion-based and two purely autoregressive unified models, the resulting images reveal key intermediate concepts that connect a question to its answer even when the concepts are never named explicitly. Controlled tests show how this content changes with context, token position, and depth. We then introduce lens tuning, which supervises selected hidden states with pictures through the V-lens. Across BAGEL, SenseNova-U1, and HunyuanImage-3.0, the rate of correct visualizations of pronoun referents rises fr
    
[^10]: SLAM-ASR语音识别模型的归纳头起源

    Induction Head Origins of SLAM-ASR Speech Recognition Models

    [https://arxiv.org/abs/2611.24721](https://arxiv.org/abs/2611.24721)

    本研究揭示SLAM-ASR语音识别模型的核心机制在于冻结语言模型的归纳头——文本中复制重复标记序列的注意力头同样负责语音嵌入到转录标记的映射，且注意力头对以受限差分注意力形式协同降噪。

    

    是什么使得一个从编码器状态到冻结语言模型嵌入空间的小型训练映射，足以实现具有竞争力的语音识别？SLAM-ASR架构的前提假设是，冻结的语言模型拥有足够丰富的语言学知识来完成识别任务。这一假设虽然合理，却对这些模型背后的机制缺乏深入洞察。因此，我们致力于寻找这些模型所依赖的具体内部结构。这项工作表明，SLAM-ASR模型依赖于冻结语言模型中的归纳头：那些在文本中复制重复标记序列的注意力头，同样对将语音嵌入映射为转录标记起着关键作用。此外，我们还发现了一些注意力头对，它们以一种受限形式的差分注意力运作，能够消除注意力噪声，并将注意力锐化聚焦于先前生成的单词上。最后，我们展示了如何利用归纳头来提升转录质量。

    arXiv:2611.24721v1 Announce Type: new  Abstract: What makes a small trained map from encoder states to the embedding space of a frozen language model sufficient for competitive speech recognition? The premise of the SLAM-ASR architecture is that a frozen language model possesses linguistic knowledge rich enough to perform the recognition task. This assumption, while reasonable, provides little insight into the mechanisms underlying these models. We therefore seek the concrete internal structures on which they rely. This work shows that SLAM-ASR models rely on induction heads of the frozen language model: the same heads that copy repeated token sequences in text are also important for mapping speech embeddings to transcript tokens. Moreover, we find pairs of heads that behave as a constrained form of differential attention, cancelling attention noise and sharpening attention onto the previously generated word. Finally, we show how induction heads could be used to improve transcription q
    
[^11]: DRIFT：面向运行时灵活且高效的端到端驾驶的可分离推理

    DRIFT: Detachable Reasoning for Runtime-Flexible and Efficient End-to-End Driving

    [https://arxiv.org/abs/2611.24719](https://arxiv.org/abs/2611.24719)

    DRIFT提出了一种端到端自动驾驶框架，利用小于1.5亿参数的轻量级语言模型实现场景描述与动作解说，并引入可在运行时灵活开关语言推理的规划模块，且不损害闭环驾驶性能，从而兼顾推理能力与计算效率。

    

    将视觉语言模型（VLM）集成到端到端（E2E）自动驾驶中，凭借其场景描述和推理能力展现出了强大的潜力。然而，VLM的计算开销高昂，并且在运行时动态开启和关闭语言推理而不损害闭环性能这一点尚未得到验证，这阻碍了其实际应用。为了解决这些问题，我们提出了DRIFT——一个集成了轻量级语言模型（LM）的端到端自动驾驶框架，用于视觉问答和动作解说。通过将中间驾驶表征高效地投影到语言嵌入空间，我们证明了一个紧凑的语言模型（参数量小于1.5亿）即可实现强大的场景描述和动作解说能力。此外，我们引入了一个运行时灵活的规划模块，该模块在从语言模型蒸馏出的代理推理特征上进行训练，并且可以以语言模型的输出或其他条件作为输入进行调节（摘要在此处被截断）。

    arXiv:2611.24719v1 Announce Type: cross  Abstract: The integration of vision-language models (VLMs) into end-to-end (E2E) autonomous driving has demonstrated strong potential by providing scene description and reasoning capabilities. However, VLMs are computationally expensive, and switching language reasoning on and off at runtime without hurting closed-loop performance has not been demonstrated, which hinders practical adoption. To address these issues, we propose DRIFT, an E2E autonomous driving framework that integrates a lightweight language model (LM) for visual question answering and action commentary. By efficiently projecting intermediate driving representations into the language embedding space, we demonstrate that a compact LM (<150M parameters) can achieve strong scene description and action commentary. Furthermore, we introduce a runtime-flexible planning module that is trained on surrogate reasoning features distilled from the LM and can be conditioned on either the LM or
    
[^12]: Openhagen Worlds：用于测量与训练科学发现智能体的校准环境

    Openhagen Worlds: Calibrated Environments for Measuring and Training Scientific Discovery Agents

    [https://arxiv.org/abs/2611.24711](https://arxiv.org/abs/2611.24711)

    Openhagen Worlds是一套包含27个程序化生成环境、178个经严格校准难度级别的科学发现基准套件，通过明确的评分规则和针对神谕策略、非演绎策略及捷径策略的成功率校准，能够真正测量并训练智能体选择信息性实验和从结果中推理的能力，排除记忆回溯与捷径策略的干扰。

    

    从事科学研究的智能体必须选择有信息量的实验并根据结果进行推理，但大多数科学发现基准无法将这种能力与记忆回溯、未明确说明的评分规则或捷径策略区分开来。我们提出Openhagen Worlds，一套包含27个程序化生成环境的基准套件（部分环境具有有限真值空间），涵盖从经典的实验推理谜题到基于模拟器的物理和化学场景。在这些环境中，智能体在预算限制下开展实验并提交答案，由确定性验证器对照隐藏真值进行检查。每个观测都标明了完整的评分规则，178个难度级别中的每一级都经过校准：只有在64个开发实例上，脚本化神谕策略成功率至少达到0.75、最强的合理非演绎策略成功率至多0.25、且每个受控捷径策略成功率至多0.35时，该级别才会被采纳。92个核心级别涵盖全部27个环境族，其规模适合训练小型模型；86个前沿级别……（摘要原文在此处截断）

    arXiv:2611.24711v1 Announce Type: cross  Abstract: Agents that do science must choose informative experiments and reason from their results, but most discovery benchmarks cannot separate that skill from recall, unstated scoring rules or shortcut policies. We introduce Openhagen Worlds, a suite of 27 procedurally generated environments (some with finite truth spaces), from classic puzzles of experimental reasoning to simulator-grounded physics and chemistry, in which an agent runs experiments under a budget and commits an answer that a deterministic verifier checks against a hidden truth. Every observation states the complete scoring rule, and each of the 178 difficulty levels is calibrated: it is admitted only if, on 64 development instances, a scripted oracle succeeds on at least 0.75, the strongest plausible non-deductive policy on at most 0.25 and each gated shortcut on at most 0.35. The 92 core levels span all 27 families and are sized to train small models; the 86 frontier rungs e
    
[^13]: 持续预训练何时有效？一项关于魁北克法语方言适应的对照研究

    When Does Continual Pre-Training Help? A Controlled Study of Dialectal Adaptation for Qu\'ebec French

    [https://arxiv.org/abs/2611.24699](https://arxiv.org/abs/2611.24699)

    该研究通过对九个大语言模型的预算匹配对照实验发现，持续预训练对魁北克法语方言适应的收益取决于起始模型对目标方言的既有能力，其增益主要由语料构成（尤其是文学文本）而非数据量驱动，且主要提升表达理解而非语法能力。

    

    持续预训练（CPT）被广泛用于将预训练语言模型适配到方言和非标准语言变体，但这种额外的训练何时才是必要的、其收益由什么驱动、以及总体评估分数是否足以刻画其效果，目前仍不清楚。本文针对魁北克法语语言理解任务，对涵盖不同模型家族与规模的九个大语言模型开展了一项持续预训练的对照研究。采用每个实验3000万词元的预算匹配协议，我们的结果表明：持续预训练的收益在很大程度上取决于起始模型对目标语言变体已有的掌握程度。当持续预训练带来收益时，其提升更多源于语料构成而非数据量，其中文学文本贡献了大部分增益。对魁北克法语与标准法语的成对评估进一步表明，持续预训练主要改善的是表达理解能力而非语法能力，并且语料库……（原文摘要在此处截断）

    arXiv:2611.24699v1 Announce Type: cross  Abstract: Continual pre-training (CPT) is widely used to adapt pretrained language models to dialectal and non-standard language varieties, yet it remains unclear when this additional training is necessary, what drives its benefits, and whether aggregate evaluation scores adequately capture its effects. In this paper, we present a controlled study of CPT for Qu\'ebec French language understanding across nine LLMs spanning different families and scales. Using a budget-matched protocol of 30M tokens per experiment, our results show that the benefit of CPT depends strongly on the starting model's existing competence in the target variety. When CPT is beneficial, the improvements are driven more by corpus composition than by data volume, with literary text accounting for most of the gain. Paired Qu\'ebec and standard French evaluations further show that CPT primarily improves expression comprehension rather than grammatical competence, and that corp
    
[^14]: CurvDelta：面向外科手术式学习与遗忘的低曲率增量建模

    CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning

    [https://arxiv.org/abs/2611.24654](https://arxiv.org/abs/2611.24654)

    CurvDelta 通过基于 K-FAC 构建低曲率空间，将能力保持约束内建于模型更新的参数化中，使大模型在能力学习与选择性遗忘等适配任务中达成目标而不损害通用能力。

    

    适配大语言模型（LLM）时的一个核心挑战是能力保持：一次适配可能在达成目标的同时损害模型的通用能力。我们提出了 CurvDelta，这是一个可扩展的框架，它将保持约束直接纳入模型更新的参数化之中。通过分析适配前模型附近保持约束的局部几何，CurvDelta 构建了一个低曲率空间，在该空间中参数更新几乎不会改变保持损失。我们借助 Kronecker 分解近似曲率（K-FAC）与分解投影，使这一构建对 LLM 而言切实可行。我们还提出了 CurvLoRA，这是同一保持几何框架下的低秩变体，旨在实现更高效的适配。两种方法均适用于保持约束下的适配任务，例如能力学习与选择性遗忘。我们在数学推理、代码生成等任务上评估了我们的方法。

    arXiv:2611.24654v1 Announce Type: new  Abstract: A central challenge in adapting large language models (LLMs) is capability preservation: an adaptation can achieve its target while degrading general capabilities. We introduce CurvDelta, a scalable framework that incorporates the preservation constraint into the parameterization of the model update. By analyzing the local geometry of the preservation constraint around the pre-adaptation model, CurvDelta constructs a low-curvature space in which parameter updates leave the preservation loss nearly invariant. We make its construction practical for LLMs through Kronecker-factored approximate curvature (K-FAC) and a factorized projection. We also introduce CurvLoRA, a low-rank variant within the same preservation geometry, designed for more efficient adaptation. Both methods apply to preservation-constrained adaptation tasks such as capability learning and selective unlearning. We evaluate our methods on mathematical reasoning, code generat
    
[^15]: Piccolo：通过递归自我改进实现决策模型与数据的协同演化

    Piccolo: Co-Evolving Decision Models and Data through Recursive Self-Improvement

    [https://arxiv.org/abs/2611.24651](https://arxiv.org/abs/2611.24651)

    Piccolo是一系列通过智能体数据工程与递归自我改进实现模型能力与训练数据协同演化的多模态决策模型，其中Piccolo-27B以75.59分登顶Decision Index Vision Full排行榜。

    

    我们提出了Piccolo，这是一个多模态决策模型家族（参数规模从0.8B到27B），通过智能体数据工程和递归自我改进（RSI）实现模型能力与训练数据的协同演化。这些模型对通用文本与视觉判断、智能体框架、数据工作流和交互式游戏中的候选答案进行概率预测。Piccolo-27B以75.59的分数登顶Decision Index Vision Full排行榜，而Piccolo-9B和Piccolo-4B分别取得68.35和67.46的成绩，三者均在各自模型规模类别的Private基准测试中排名第一。研究智能体负责分析模型行为、构建训练数据、开展实验，并利用实验结果来改进后续方案。语义切片和共享研究记忆在整个模型家族的开发过程中将这些行动串联起来。借助包含数十万个样本的训练配方，数据选择和反馈驱动的改进……

    arXiv:2611.24651v1 Announce Type: cross  Abstract: We present Piccolo, a family of multimodal decision models (0.8B to 27B) that coevolves model capabilities and training data via Agentic Data Engineering and recursive self-improvement (RSI). The models predict probabilities over candidate answers for general-purpose text and visual judgments, agent harnesses, data workflows, and interactive games. Piccolo-27B tops the Decision Index Vision Full leaderboard with a score of 75.59, while Piccolo-9B and Piccolo-4B achieve 68.35 and 67.46, respectively. All three rank first on the Private benchmark in their respective model-size categories. Research agents analyze model behavior, construct training data, conduct experiments, and use the results to refine subsequent proposals. Semantic slices and a shared research memory connect these actions across the model family's development. With training recipes comprising hundreds of thousands of examples, data selection and feedback-driven refineme
    
[^16]: 面向小语言模型更结构化医学解释的自奖励强化学习

    Self-Rewarded Reinforcement Learning for Better-Structured Medical Explanations in Small Language Models

    [https://arxiv.org/abs/2611.24650](https://arxiv.org/abs/2611.24650)

    该论文提出一种无需人工偏好标注的自奖励强化学习方法，通过简单的自监督奖励引导 PPO 训练，显著改善了小语言模型生成的医学解释的结构与质量。

    

    小语言模型（SLM）可以在普通硬件上运行，并支持保护隐私的本地化临床部署，但与大型语言模型相比，其生成的医学解释往往简短且结构欠佳。近期的监督式方法（如 Meerkat）将思维链推理迁移到 7B-8B 参数规模的骨干模型中，但其训练目标并未直接针对所生成解释的风格或组织结构。基于指令微调的 Mistral-7B 骨干模型，我们研究了轻量级的强化学习（RL）步骤能否在无需人工偏好标注的情况下改善解释的结构。我们在 2,000 个源自 Meerkat 的医学提示上应用了结合 LoRA 适配的近端策略优化（PPO），并由一个简单的自监督奖励引导，该奖励偏好更长、词汇更丰富且标点规范的回答，同时惩罚过短的回答。在单个 NVIDIA A100 GPU 上训练约 170 个 PPO 步骤后，得到了（摘要在此处截断）……

    arXiv:2611.24650v1 Announce Type: new  Abstract: Small language models (SLMs) can run on commodity hardware and support privacy-preserving, on-premises clinical deployment, but their medical explanations are often short and poorly structured compared with large language models. Recent supervised approaches such as Meerkat transfer chain-of-thought reasoning into 7B-8B backbones, yet their training objective does not directly target the style or organization of the generated rationales. Building on the instruction-tuned Mistral-7B backbone, we investigate whether a lightweight reinforcement learning (RL) step can improve explanation structure without human preference labels.We apply Proximal Policy Optimization (PPO) with LoRA adaptation to 2,000 Meerkat-derived medical prompts, guided by a simple self-supervised reward that favors longer, lexically diverse, and well-punctuated responses while penalizing very short ones. Training for about 170 PPO steps on a single NVIDIA A100 GPU yield
    
[^17]: Clover-35B-A3B-Preview 技术报告

    Clover-35B-A3B-Preview Technical Report

    [https://arxiv.org/abs/2611.24636](https://arxiv.org/abs/2611.24636)

    Clover 是一个总参数 350 亿、激活参数仅 30 亿的混合专家扩散语言模型，其创新在于将门控 Delta 网络的循环记忆与块内双向注意力结合以适配块扩散，在智能体基准上全面领先（SWE-bench Verified 达 73.6 分），并在单块 GB300 GPU 上实现 680 tokens/s 的生成速度。

    

    我们介绍 Clover-35B-A3B-Preview（简称 Clover），这是一个混合专家扩散语言模型，总参数量为 350 亿，激活参数量为 30 亿，支持思考模式生成。Clover 将门控 Delta 网络（Gated Delta Networks）适配到块扩散范式中，将先前已生成上下文的循环记忆与当前块内的双向 softmax 注意力相结合。历史记忆读取与当前块内的 token 在共享 softmax 中进行竞争，并配备可学习的块大小校准以及一个独立的、零初始化的右上下文卷积分支。Clover 在全部七项被评估的智能体指标中均处于领先地位，其中包括 SWE-bench Verified 上的 73.6 分和 Terminal-Bench 2.1 上的 51.7 分。在内部智能体工作负载上，使用单块 NVIDIA GB300 GPU，在 BF16 精度和批量大小为 1 的条件下，其报告的生成速度达到 680 tokens/s。我们围绕模型质量、草拟与验证成本以及受限内存条件下的部署来组织评估。

    arXiv:2611.24636v1 Announce Type: cross  Abstract: We introduce Clover-35B-A3B-Preview (Clover), a mixture-of-experts diffusion language model with 35 billion total parameters and 3 billion activated parameters, supporting thinking-mode generation. Clover adapts Gated Delta Networks to block diffusion by combining recurrent memory of previously generated context with bidirectional softmax attention inside the current block. Historical memory readout competes with current-block tokens in a shared softmax, with learnable block-size calibration and a separate, zero-initialized right-context convolution branch. Clover leads the seven evaluated agent metrics, including 73.6 on SWE-bench Verified and 51.7 on Terminal-Bench 2.1. On internal agent workloads, it reaches a reported generation speed of 680 tokens/s using one NVIDIA GB300 GPU at BF16 precision and batch size one. We organize the evaluation around model quality, drafting and verification costs, and deployment under constrained memo
    
[^18]: 将人类大脑组织结构分析迁移至语言模型

    Transferring an analysis of human brain organization to language models

    [https://arxiv.org/abs/2611.24626](https://arxiv.org/abs/2611.24626)

    该研究将刻画人类皮层组织的扩散映射方法应用于语言模型，发现推理训练使模型在一段连续层跨度内发生局部重组，将该区域的权重嫁接到其他模型中可提升其数学推理能力。

    

    在人类大脑皮层发育过程中，功能系统逐渐分化。我们提出疑问：语言模型推理能力的提升是否涉及类似的区域性重组。我们将通常用于刻画皮层组织结构的扩散映射嵌入方法，应用于来自三个模型家族的相互匹配的基座模型与经推理训练的模型，发现整体梯度轮廓大体保持不变，但在一段连续的层跨度内存在局部变化。将这些跨度区域的权重嫁接到指令微调的宿主模型中，平均而言提升了数学推理能力，且增益随区域重叠度的增加而上升，同时比在其他位置进行嫁接保留了更多先前已成功解决的问题。在整个训练过程中，该跨度在性能早期上升阶段逐渐形成，来自中间训练阶段的嫁接同样有效。嫁接使大多数宿主模型的行为部分向其供体模型偏移，有时甚至在供体模型未能解决的问题上取得改进；同一分析还在一个（原文在此处截断）……

    arXiv:2611.24626v1 Announce Type: cross  Abstract: During human cortical development, functional systems become increasingly differentiated. We asked whether improved reasoning in language models involves a comparable regional reorganization. Applying diffusion-map embedding, used to characterize cortical organization, to matched base and reasoning-trained models from three families, we found broadly preserved gradient profiles with localized changes within a contiguous layer span. Grafting weights from these spans into instruction-tuned hosts improved mathematical reasoning on average, with gains rising with region overlap, while retaining more previously solved problems than grafts elsewhere. Across training, the span formed gradually during the early rise in performance, and grafts from intermediate stages helped. Grafts shifted most hosts partly towards their donors' behaviour and sometimes improved on problems the donors failed; the same analysis located a transferable span in a m
    
[^19]: 思维链隐状态揭示通向直接答案的任务求解方向

    Chain-of-Thought Hidden States Reveal Task-Solving Directions for Direct Answers

    [https://arxiv.org/abs/2611.24598](https://arxiv.org/abs/2611.24598)

    该论文发现直接提示的表征空间中存在一个由思维链隐状态（恢复与失败状态）构建出的任务求解方向，仅需在生成前旋转一次隐状态，就能引导模型在短生成中比思维链更准确地直接得出正确答案。

    

    思维链提示通过引导中间推理步骤来提升大语言模型的性能。然而，思维链远非完美：在某些任务上，模型的表现反而不如直接回答提示；在另一些任务上，模型需要生成数百个token才能得到正确答案。我们首先为每个思维链隐状态打标签：如果将其修补到直接提示中能得到正确答案，则标记为“恢复”，否则标记为“失败”。线性探针实验表明，模型对这两类状态的编码方式存在差异。随后我们证明，直接提示的表征空间中存在一个针对每个问题的特定方向，该方向能够引导模型在较短的生成过程中比思维链更频繁地得出正确答案。具体而言，我们在生成之前，将直接提示的最后一个隐状态朝某个方向旋转一次，该方向是根据每个问题自身的“恢复”与“失败”状态构建的。在25个任务上，这些方向提升了直接提示的准确率……

    arXiv:2611.24598v1 Announce Type: cross  Abstract: Chain-of-Thought (CoT) prompting improves the performance of Large Language Models by eliciting intermediate reasoning steps. However, CoT is far from perfect: on some tasks, models perform worse than with direct-answer prompting, and on others they generate a few hundred tokens to reach the correct answer. We start by labelling each CoT hidden state as recovering if patching it into the direct prompt yields the correct answer, and failed otherwise. A linear probe shows that the model encodes these two classes differently. We then show that the representation space of the direct prompt holds a direction for each question that leads the model, in a short generation, to the correct answer more often than CoT does. Specifically, we rotate the direct prompt's last hidden state once, before generation, towards a direction built for each question from its own recovering and failed states. Across 25 tasks, these directions raise the direct pr
    
[^20]: 大语言模型还是AI智能体更适合英语到阿拉伯语的翻译？基于WikiEn-Ar190数据集的评估

    Large Language Models or AI Agents for English to Arabic Translation? An assessment on WikiEn-Ar190

    [https://arxiv.org/abs/2611.24588](https://arxiv.org/abs/2611.24588)

    本研究构建了由阿拉伯语母语者核验的英译阿拉伯语新基准数据集WikiEn-Ar190，发现GPT-5-mini独立翻译已表现优异，而引入自我评估与改进的智能体流程可进一步提升词汇准确性。

    

    本研究探讨了大语言模型与自主AI智能体在英语到阿拉伯语（一种低资源语言）翻译任务中孰优孰劣。作者推出了WikiEn-Ar190，这是一个包含190篇精心挑选的维基百科文章的新数据集，涵盖18个不同领域，由两名阿拉伯语母语者创建并核验，以确保其准确反映原文的含义与风格。基于该数据集，研究评估了GPT-5-mini的翻译质量，既包括其作为独立神经翻译器的表现，也包括其应用于包含自我评估与改进环节的更复杂流程时的表现。作为独立模型，GPT-5-mini生成了非常流畅的译文，COMET得分达到0.8649，并在充分性、流畅性和风格方面获得很高的人工评分（平均4.53分，满分5分）。在结合翻译与自我评估的系统中，其词汇准确性提升了28.76%（BLEU……）

    arXiv:2611.24588v1 Announce Type: new  Abstract: This research investigates whether a large language model or an autonomous AI Agent performs better at translating from English to Arabic, which is considered a low-resource language. The authors introduce WikiEn-Ar190, a new dataset of 190 carefully selected Wikipedia articles from 18 different areas, which were created and verified by two native Arabic speakers to ensure that they accurately reflect both meaning and style. Using this dataset, the study evaluates the translation quality of GPT-5-mini both as an independent neural translator and when used within a more complex process that includes self-assessment and improvement. As a standalone model, GPT-5-mini produced very fluent translations , achieving a COMET score of 0.8649 and receiving high human scores for adequacy, fluency, and style (averaging 4.53 out of 5). When used in a system that combined translation with self-assessment, it improved lexical accuracy by 28.76% (BLEU s
    
[^21]: 循环语言模型中的晚入策略

    Late Entry in Looped Language Models

    [https://arxiv.org/abs/2611.24576](https://arxiv.org/abs/2611.24576)

    提出循环语言模型的“晚入”解码策略——新token跳过初始迭代、直接基于深度缓存上下文进入最后K次迭代，性能优于同等的早退策略，并可将该优势转化为实际解码速度的提升。

    

    循环语言模型通过对每个token迭代共享层堆栈T次，在紧凑参数规模下实现更深层的推理，但也因此将循环深度直接转化为顺序解码的延迟。传统的早退解码在完成初始几次迭代后即停止且不完成缓存构建，将历史token的缓存键值状态截断至浅层深度。我们研究了相反的策略——晚入，即新token跳过初始迭代，直接进入基于深度上下文缓存的最后K（K < T）次迭代。一旦通过轻量级微调解决了入口边界处隐状态范数不匹配的问题，深度缓存上下文便可替代被跳过的token局部迭代，使晚入在多个模型家族和迭代预算设定下均优于同等条件的早退策略。为了在不因构建深度缓存而停滞的情况下将这种不对称性转化为实际运行时间的加速，候选块在关键路径上仅通过最后K次迭代进行验证，而先前已提交的token则完成……（摘要原文在此处被截断）

    arXiv:2611.24576v1 Announce Type: new  Abstract: Looped language models deepen reasoning at a compact parameter count by iterating a shared layer stack T times per token, turning recurrence depth directly into sequential decoding latency. Conventional early-exit decoding halts after initial passes without cache completion, truncating cached key-value states of past tokens to shallow depth. We study the opposite strategy, late entry, where new tokens skip initial passes and enter directly into the final K < T passes over a deep context cache. Once lightweight fine-tuning resolves hidden-state norm mismatch at the entry boundary, deep cached context substitutes for skipped token-local passes, allowing late entry to outperform matched early exit across model families and pass budgets. To turn this asymmetry into wall-clock speedups without stalling to build deep caches, candidate blocks verify through only the final K passes on the critical path while previously committed tokens complete 
    
[^22]: 流形约束超连接中的目标特化残差流

    Objective-Specialized Residual Streams in Manifold-Constrained Hyper-Connections

    [https://arxiv.org/abs/2611.24567](https://arxiv.org/abs/2611.24567)

    在流形约束超连接中为不同残差流分配不同的学习目标（如词汇与语义检索），能在各种配置下持续提升性能并诱导出目标特定的内部表征特化。

    

    流形约束超连接将Transformer的残差流扩展为多条并行流，并带有可学习的跨流连接。先前的工作将所有流朝着一个共享目标进行训练。我们转而研究为不同流分配不同的学习目标能否提升任务性能并诱导出目标特定的特化。以互补的词汇（稀疏）和语义（密集）检索目标作为测试案例，我们发现目标特化的mHC在整个稀疏-密集融合范围内始终优于非特化的mHC模型，这在双流和四流模型以及两种模型规模上均成立。随后，我们通过对流表示、残差混合、目标特定梯度流以及读写门控的探测，考察了特化如何在模型内部改变模型。特化与非特化模型在网络的大部分部分表现相似，但在网络的末端出现分化……

    arXiv:2611.24567v1 Announce Type: new  Abstract: Manifold-constrained Hyper-Connections (mHC) extend a Transformer's residual stream into multiple parallel streams with learned cross-stream connections. Prior work trains all streams toward a shared objective. We instead study whether assigning distinct learning objectives to different streams can improve task performance and induce objective-specific specialization. Using complementary lexical (sparse) and semantic (dense) retrieval objectives as a test case, we find that objective-specialized mHC consistently outperforms a non-specialized mHC model across the full sparse--dense fusion range, for both two- and four-stream models and at two model scales. We then examine how specialization changes the model internally using probes of stream representations, residual mixing, objective-specific gradient flow, and read/write gates. The specialized and non-specialized models behave similarly through most of the network but diverge in the fin
    
[^23]: 前沿模型组件能否叠加生效？一项针对小语言模型的重复因子实验研究

    Do Frontier Model Ingredients Add Up? A Replicated Factorial Study of Small Language Models

    [https://arxiv.org/abs/2611.24540](https://arxiv.org/abs/2611.24540)

    该研究在相同数据和训练预算下通过重复因子实验比较了小语言模型的五项前沿改进，发现混合专家层、Muon 优化器和 Gated DeltaNet 混合架构收益最大且增益大多可叠加，强调在决定采用哪些组件前应组合测试并重复实验。

    

    构建小语言模型需要选择组合哪些已发表的改进方法。我们在同一个模型中使用相同的数据和训练预算，比较了五项改进。我们测试了选定的组合，并用第二个随机种子重复了每次实验。混合专家层、Muon 优化器和 Gated DeltaNet 混合架构带来了最大的增益。额外的归一化也有帮助，而在参数预算匹配的情况下，注意力门控平均而言几乎没有收益。这些增益大多可以叠加，但在重复实验中，各改进之间仍会出现微小的相互作用。重复实验还揭示了一些最初被统计估计视为噪声的收益。这些发现适用于我们的小模型训练设定；专家模型也拥有更大的总容量。实践中的经验教训是：在决定保留哪些组件之前，应当将最有前景的组件组合起来测试，并重复进行比较。

    arXiv:2611.24540v1 Announce Type: cross  Abstract: Building a small language model requires choosing which published improvements to combine. We compare five changes in the same model, using the same data and training budget. We test selected combinations and repeat each one with a second random seed. Mixture-of-experts layers, Muon optimisation and a Gated DeltaNet hybrid give the largest gains. Extra normalisation also helps, while an attention gate with a matched parameter budget adds little on average. The gains mostly combine, but small interactions between changes appear again in the repeated runs. Repetition also reveals benefits that an initial statistical estimate treated as noise. These findings apply to our small-model training setting; the expert models also have greater total capacity. The practical lesson is to test promising components together and repeat the comparisons before deciding which ones to keep.
    
[^24]: ProbMedTOD：一种贝叶斯网络引导的面向患者病史采集的任务导向对话系统

    ProbMedTOD: A Bayesian Network Guided Task-Oriented Dialogue System for Patient History Taking

    [https://arxiv.org/abs/2611.24535](https://arxiv.org/abs/2611.24535)

    该论文提出ProbMedTOD系统，通过自动合成临床贝叶斯网络为小型语言模型提供可解释的概率信念跟踪与“系统2”深思推理能力，从而实现更可靠的患者病史采集任务导向对话。

    

    病史采集是一个策略性的多轮过程，通过迭代提问来细化诊断假设，反映了临床推理中每次询问都会更新对候选诊断信念的特点。在这一场景中部署AI受到隐私、成本和延迟的限制，因此相比大型语言模型（LLMs），小型语言模型（SLMs）是更理想的骨干选择。然而，SLMs只能捕捉这一过程中快速、直觉的“系统1”部分，缺乏维持和更新诊断不确定性所需的深思熟虑的“系统2”推理的显式机制。贝叶斯网络（BayesNets）提供了天然的补充，为可审计的概率信念跟踪提供了可解释的框架。然而，临床贝叶斯网络难以构建，因为其依赖于专家手工整理的结构、依赖关系和参数。我们提出了ProbMedTOD，它能从少量的……自动合成临床贝叶斯网络（注：原摘要不完整，在“from a sma”处截断）

    arXiv:2611.24535v1 Announce Type: new  Abstract: Patient history taking is a strategic, multi-turn process that refines diagnostic hypotheses through iterative questioning, mirroring clinical reasoning where each inquiry updates beliefs over candidate diagnoses. Deploying AI in this setting is constrained by privacy, cost, and latency, making Small Language Models (SLMs) a more desirable backbone compared to larger LLMs. However, SLMs capture only the fast, intuitive "System 1" side of this process and lack explicit mechanisms for the deliberative "System 2" reasoning needed to maintain and update diagnostic uncertainty over time. Bayesian Networks (BayesNets) offer a natural complement, providing an interpretable framework for auditable probabilistic belief tracking. Yet, clinical BayesNets are difficult to construct due to their reliance on expert curated structure, dependencies, and parameters.   We introduce ProbMedTOD, which automatically synthesizes a clinical BayesNet from a sma
    
[^25]: MECI：面向多模态实体链接的多元素协同交互

    MECI: Multi-Element Collaborative Interaction for Multimodal Entity Linking

    [https://arxiv.org/abs/2611.24522](https://arxiv.org/abs/2611.24522)

    提出MECI框架，通过多视图专家网络和KL引导的路由机制进行解耦特征学习，缓解元素主导问题，从而更好地捕捉多模态实体链接中提及-实体的多层次对应关系和跨模态互补相关性。

    

    多模态实体链接（MEL）旨在将多模态语境中的提及进行消歧，并将其链接到知识库中的特定实体。多模态实体链接的一个关键挑战是捕捉多层次的对应关系：即提及-实体对之间的语义一致性，以及跨模态之间的互补相关性。然而，现有方法由于依赖耦合交互或粗糙的全局聚合，往往存在“元素主导”问题。为此，我们提出了多元素协同交互（MECI）框架。首先，为了捕捉多元素的提及-实体对应关系，我们开发了多视图专家网络，利用“分而治之”策略进行解耦特征学习以缓解元素主导问题，并由KL引导的路由机制提供支持，该机制调控专家的专业化与协作。此外，为了建模跨模态互补相关性，我们提出了一种分层多……（摘要在此处截断）

    arXiv:2611.24522v1 Announce Type: cross  Abstract: Multimodal Entity Linking (MEL) aims to disambiguate mentions in multimodal contexts by grounding them to specific entities in a knowledge base. A pivotal challenge in MEL is capturing multi-level correspondences: the semantic consistency between mention-entity pairs and the complementary correlations across modalities. However, existing methods often suffer from element dominance due to their reliance on coupled interactions or coarse global aggregations. In response, we propose the Multi-Element Collaborative Interaction (MECI) framework. First, to capture multi-element mention-entity correspondences, we develop a Multi-view Experts Network that leverages a "divide-and-conquer" strategy for decoupled feature learning to mitigate element dominance, supported by a KL-guided routing mechanism that governs expert specialization and collaboration. Furthermore, to model cross-modal complementary correlations, we propose a Hierarchical Mult
    
[^26]: 无需申报：操控单个神经元会增加AI团队中的隐瞒行为

    Nothing to Declare: Steering a single neuron increases concealment in AI teams

    [https://arxiv.org/abs/2611.24475](https://arxiv.org/abs/2611.24475)

    操控单个神经元会让AI更倾向于隐瞒信息，且这种效应在团队协作中比单独行动时更显著，说明集体环境能够改变内部干预对AI行为的影响。

    

    近期研究表明，单个神经元在AI行为中可能具有可解释的角色。在Qwen3.5-4B模型中，沿某一个神经元的输出方向进行操控，在五个场景中都更多地增加了AI隐瞒信息的决策，且在团队中比单独行动时效果更强，即使队友没有发送任何消息也是如此。当所有成员都参与并接受操控时，四个场景中的集体隐瞒行为有所增加，而智能体之间的消息传递在操控之下有助于缓解隐瞒行为。此外，移除该神经元的贡献几乎不产生任何影响。因此，集体环境可以改变内部干预对AI行为的影响方式。

    arXiv:2611.24475v1 Announce Type: cross  Abstract: Recent research suggests that individual neurons can have interpretable roles in AI behaviour. In Qwen3.5-4B, steering along one neuron's output direction increased decisions to conceal information more in teams than alone across five scenarios, even when teammates provided no messages. When all members participated and received steering, collective concealment increased in four scenarios, while inter-agent messaging helped temper concealment under steering. Moreover, removing the neuron's contribution had negligible effects. Collective settings can therefore change how an internal intervention affects AI behaviour.
    
[^27]: 多数标签隐藏了什么：多方言阿拉伯语性别歧视检测中的标注者分歧与模型错误

    What Majority Labels Hide: Annotator Disagreement and Model Errors in Multi-Dialect Arabic Sexism Detection

    [https://arxiv.org/abs/2611.24383](https://arxiv.org/abs/2611.24383)

    本文构建了保留个体标注结果的多方言阿拉伯语性别歧视检测数据集MuDAS，发现模型错误更多出现在标注者存在分歧的评论上，且GPT-5.6 Terra与微调编码器虽F1分数接近但错误模式截然不同——前者过度判定性别相关内容，后者则漏判隐性性别歧视。

    

    对性别歧视的判断是主观的，且可能因社群而异，然而性别歧视检测数据集通常将标注者的判断简化为单一的多数标签，从而掩盖了标注者之间的分歧以及模型在争议样本上的表现。我们推出了MuDAS数据集，包含沙特、埃及和摩洛哥阿拉伯语方言的YouTube评论，每条评论由三名该方言的母语者分别标注，并保留了每位标注者的独立标注结果。我们利用该数据集评估了采用硬标签、软标签和视角主义目标微调的阿拉伯语编码器模型、GPT-5.6 Terra，以及四个以阿拉伯语为中心的大语言模型。结果表明，在存在标注者分歧的评论上，模型错误更为频繁。尽管宏观F1分数相对接近，GPT-5.6 Terra与微调编码器却表现出截然不同的错误模式：即使采用了标注者对性别歧视的定义，GPT-5.6 Terra仍会将诸如关于端庄的宗教劝告等性别相关内容过度判定为性别歧视，而编码器模型则会漏判隐性的性别歧视言论。

    arXiv:2611.24383v1 Announce Type: new  Abstract: Judgements of sexism are subjective and may vary across communities, yet sexism detection datasets usually reduce annotator judgements to a single majority label, obscuring disagreement and how models behave on contested cases. We introduce MuDAS, a dataset of YouTube comments in Saudi, Egyptian and Moroccan Arabic, each labelled by three native speakers of its dialect, with the individual annotations preserved. We use it to evaluate Arabic encoders fine-tuned with hard-label, soft-label and perspectivist objectives, GPT-5.6 Terra, and four Arabic-centric large language models. Model errors are more frequent on comments with annotator disagreement. Despite relatively close Macro-F1 scores, GPT-5.6 Terra and the fine-tuned encoders exhibit contrasting error patterns: GPT-5.6 Terra, even with the annotators' definition of sexism, over-classifies gender-related content such as religious advice on modesty, whereas the encoders miss implicit 
    
[^28]: MoE-Context：只让模型看到它需要的内容

    MoE-Context: Show the Model Only What It Needs

    [https://arxiv.org/abs/2611.24379](https://arxiv.org/abs/2611.24379)

    MoE-Context通过在每步前进行快速的“是/否”门控判断，只向语言模型展示当前所需的上下文块，其余内容以可召回的单行笔记形式掩蔽在单一键值缓存中而不被删除，从而将查找任务准确率从93/120大幅提升至119/120。

    

    AI智能体在每一步都会读取其完整的历史记录。人类并非如此工作：我们只保留当下所需的信息，其余内容则以可随时回忆的粗略笔记形式保存。MoE-Context将这一理念应用于语言模型：上下文被分割成多个块，在每一步之前，一次快速的“是/否”判断决定该步需要哪些块；其余块保留为可以调回的单行笔记，且不删除任何内容。在我们构建的版本中，同一个模型先做出这一判断再进行回答，所有操作都在单一的键值缓存上进行，其中被隐藏的块是被掩蔽而非移除。在配备Qwen3-1.7B的笔记本电脑上进行的合成查找任务中，仅显示所需块可将准确率从120题中的93题提升至119题，且被掩蔽的块与被删除的块同样不可见。模型自身的挑选能力是薄弱环节：一个写入空闲输出行的监督式线性探针将准确率从120题中的80题提升至110题，但在该任务上简单的关键词搜索能实现完美挑选，且该探针无法迁移（摘要在此处截断）。

    arXiv:2611.24379v1 Announce Type: new  Abstract: AI agents read their whole history at every step. People do not work like that: we keep what the moment needs and hold the rest as a rough note we can recall. MoE-Context applies this to language models. The context is split into blocks. Before each step a fast yes or no pass decides which blocks the step needs; the rest stay as one-line notes that can be brought back, and nothing is deleted. In the version we build, one model makes this decision and then answers, over a single key-value cache where hidden blocks are masked rather than removed. On a laptop with Qwen3-1.7B and a synthetic lookup task, showing only the needed block raises accuracy from 93 to 119 of 120, and a masked block is as hidden as a deleted one. The model's own picking is the weak part. A supervised linear probe written into spare output rows lifts its accuracy from 80 to 110 of 120, but simple keyword search picks perfectly on this task, and the probe does not tran
    
[^29]: LoopCache：加速循环Transformer的长上下文自投机解码

    LoopCache: Accelerating Long-Context Self-Speculative Decoding for Looped Transformers

    [https://arxiv.org/abs/2611.24301](https://arxiv.org/abs/2611.24301)

    本文提出LoopCache，一个面向长上下文循环Transformer的高效自投机解码框架，通过置信度引导机制解决现有波风格方法中固定循环起草浪费计算、以及跨循环批处理导致KV缓存内存流量放大R倍这两个关键问题。

    

    循环Transformer通过在R个递归步骤中重复应用权重共享的模块，以紧凑的模型实现较大的有效深度，每个循环步骤维护一个特定于该循环的键值缓存。为了加速顺序循环执行，最近出现了波风格的自投机方法，如WaveFront Decoding（WFD）和LoopSpec，它们在单个步骤中将草词和目标词跨循环流水线化。然而，这些方法面临两个关键局限。首先，它们只在固定的循环处进行起草，忽略了中间预测的变化，在过时的轨迹上浪费计算资源。其次，将来自不同循环的词进行批处理时，每步会加载多达R倍的KV缓存，在长上下文场景中严重放大了内存流量。为了应对这些挑战，我们提出了LoopCache，一个面向长上下文循环Transformer的高效自投机解码框架。基于预测切换与输出置信度之间的相关性，LoopCache引入了置信度引导的……（摘要在此处被截断）

    arXiv:2611.24301v1 Announce Type: new  Abstract: Looped Transformers achieve large effective depth with compact models by repeatedly applying a weight-shared block across R recurrent steps, each maintaining a loop-specific key-value (KV) cache. To accelerate sequential loop execution, recent wave-style self-speculative methods like WaveFront Decoding (WFD) and LoopSpec pipeline draft and target tokens across loops in a single step. However, they face two critical limitations. First, they draft only at fixed loops and ignore intermediate prediction changes, wasting computation on stale trajectories. Second, batching tokens from different loops loads up to Rx the KV cache per step, severely amplifying memory traffic in long-context scenarios. To address these challenges, we propose LoopCache, an efficient self-speculative decoding framework for long-context Looped Transformers. Based on the correlation between prediction switches and output confidence, LoopCache introduces confidence-gui
    
[^30]: LipSpell：一种结合区别特征编辑距离、键盘几何与语料库频率的语言无关语音评分拼写纠正框架

    LipSpell: A Language-Independent Phonetic Scoring Framework for Spelling Correction Combining Distinctive-Feature Edit Distance, Keyboard Geometry, and Corpus Frequency

    [https://arxiv.org/abs/2611.24285](https://arxiv.org/abs/2611.24285)

    LipSpell提出了一个语言无关的语音评分框架，结合基于IPA区别特征的音素编辑距离、键盘几何结构和语料库频率先验，在六个英语基准测试上全面超越Norvig和Hunspell等传统拼写纠正方法。

    

    自动拼写纠正必须同时处理键盘输入手误和认知或语音层面的拼写错误，即书写者根据单词的大致发音进行拼写的情况。传统拼写检查器要么依赖正字法编辑距离和键盘邻近度——当语音错误在多个字符上出现偏差时，这类方法表现不佳——要么依赖以英语为中心的语音哈希方法，如Soundex和Metaphone。我们提出了LipSpell，这是一个统一的、语言无关的语音评分引擎，配备可替换的特定语言G2P（字素到音素）前端、键盘映射和词典，它集成了：(1) 基于通用IPA区别特征预计算的|P|x|P|音素距离矩阵，(2) 物理键盘几何结构，以及(3) 语料库一元词频率先验，并通过两阶段对称删除索引和七类语音代码索引进行加速。在六个英语基准测试（包括来自TOEFL-Spell的1,000个二语学习者错误）上，LipSpell在每个数据集上都优于Norvig和Hunspell（注：摘要原文在此处被截断）。

    arXiv:2611.24285v1 Announce Type: new  Abstract: Automatic spelling correction must handle both keyboard typographical slips and cognitive or phonetic misspellings where writers approximate a word's pronunciation. Conventional spell checkers rely either on orthographic edit distance and keyboard proximity--which struggle when phonetic errors diverge by multiple characters--or on English-centric phonetic hashes such as Soundex and Metaphone. We present LipSpell, a unified, language-independent phonetic scoring engine paired with swappable language-specific G2P frontends, keyboard maps, and lexicons integrating: (1) a precomputed |P|x|P| phone-distance matrix from universal IPA distinctive features, (2) physical keyboard geometry, and (3) corpus unigram frequency priors, accelerated by a two-stage symmetric-delete and seven-class phonetic-code index. Across six English benchmarks (including 1,000 L2 errors from TOEFL-Spell), LipSpell outperforms Norvig and Hunspell on every dataset and b
    
[^31]: 来自SO(3)的有限子群：C₂引出Holant二分定理

    From a finite subgroup of $SO(3)$:$C_2$ leads to a Holant dichotomy

    [https://arxiv.org/abs/2611.24236](https://arxiv.org/abs/2611.24236)

    该论文提出二元签名矩阵的射影等价类可自然嵌入SO(3)的有限子群这一代数视角，据此将C₂情形划分为正规与各向同性两类，并综合运用gadget构造、分解、插值与实数化技术，证明了复值偶数元Holant框架下的复杂性二分定理。

    

    计数复杂性在Holant框架中已通过复杂性二分定理得到广泛研究。尽管在各种受限设置下已取得实质性进展，但在没有辅助一元签名的复值偶数元一般情形下，该问题仍基本悬而未决。本文研究了该研究计划中的一个基本情形，即二元签名诱导出满足 M(B(𝓕)) ≅ C₂ 的群结构。我们观察到二元签名矩阵的射影等价类自然地嵌入到SO(3)的有限子群中，为结构分类提供了一个代数视角。本文进一步识别出C₂情形下两个根本不同的模式，即正规情形和各向同性情形，并分别分析了它们的复杂性行为。结合gadget构造、分解技术、插值论证，以及一种将……转换为实数的实数化方法

    arXiv:2611.24236v1 Announce Type: cross  Abstract: Counting complexity within the Holant framework has been extensively studied through complexity dichotomy theorems. While substantial progress has been achieved for various restricted settings, the general complex-valued even-arity case without auxiliary unary signatures remains largely open. This paper studies a fundamental case in this program where the binary signatures induce a group structure satisfying \(M(B(\mathcal{F}))\cong C_2\). It is observed that projective equivalence classes of binary signature matrices naturally embed into finite subgroups of \(SO(3)\), providing an algebraic viewpoint for structural classification. This paper further identifies two fundamentally different regimes for the \(C_2\) case, namely the normal and isotropic cases, and analyzes their complexity behavior separately.   Combining gadget constructions, decomposition techniques, interpolation arguments, and a realnumberization approach that transfor
    
[^32]: CounselTraj：通过对话行为轨迹规划合成过程感知的心理咨询对话

    CounselTraj: Synthesizing Process-Aware Psychological Counseling Dialogues via Dialogue-Act Trajectory Planning

    [https://arxiv.org/abs/2611.24168](https://arxiv.org/abs/2611.24168)

    提出CounselTraj框架，通过从人类咨询师角色扮演对话中学习对话行为模式并规划来访者与咨询师协同的轨迹，将简短的合成对话扩展为过程感知的会话级心理咨询对话，从而缓解了真实咨询数据因隐私问题难以获取的困境。

    

    日益增长的心理咨询需求推动了基于大语言模型（LLM）的辅助人类咨询师系统的发展，但训练此类系统需要大规模、高质量的对话数据。出于隐私方面的考虑，真实的咨询对话记录难以收集和共享，这使得合成对话成为一种重要的替代方案。然而，现有的合成数据集尽管涵盖了多样的场景和来访者画像，但往往仅包含简短的交流，无法刻画会话层面的咨询过程。我们提出了CounselTraj，这是一个对话行为轨迹规划框架，用于将简短的合成对话扩展为具有过程感知的、会话级别的心理咨询对话。CounselTraj从人类咨询师创作的角色扮演咨询对话中学习对话行为模式，为来访者和咨询师规划协同的轨迹，并将其作为长篇对话生成的灵活引导。

    arXiv:2611.24168v1 Announce Type: cross  Abstract: The growing demand for psychological counseling has motivated the development of LLM-based systems that support human counselors, but training such systems requires large-scale, high-quality dialogue data. Authentic counseling transcripts are difficult to collect and share because of privacy concerns, making synthetic dialogues an important alternative. However, existing synthetic datasets, despite covering diverse scenarios and client profiles, often consist of brief exchanges and fail to capture session-level counseling processes. We propose CounselTraj, a dialogue-act trajectory planning framework for expanding brief synthetic dialogues into process-aware, session-level psychological counseling dialogues. CounselTraj learns dialogue-act patterns from role-play counseling dialogues produced by human counselors, plans coordinated trajectories for clients and counselors, and uses them as flexible guidance for long-form dialogue generat
    
[^33]: 《从前有个偏见：大语言模型生成的儿童故事中的性别刻板印象》

    Once Upon a Bias: Gender Stereotyping in LLM-Generated Children's Stories

    [https://arxiv.org/abs/2611.24156](https://arxiv.org/abs/2611.24156)

    该论文通过分析16个大语言模型在35个国家场景和12种叙述者角色下生成的62,873篇睡前故事，并提出了一个多层次评估框架，系统揭示了大语言模型会根据儿童性别生成带有系统性性别刻板印象的叙事（如面向儿子的故事强调更高能动性）。

    

    大语言模型日益被广泛用于为儿童创作故事。由于睡前故事会影响儿童对社会角色的认知，这引发了一个担忧：大语言模型是否会根据故事预设接收者的性别来调整叙事刻画。我们开展了一项大规模研究，以考察大语言模型生成的睡前故事中的性别刻板印象。通过采用匹配的女儿/儿子提示词和五代生成设计，我们分析了来自16个大语言模型、覆盖35个国家场景和12种叙述者角色的62,873篇留存故事。我们提出了一个多层次评估框架，综合运用情境调整的词汇比率对比、相对刻板印象一致性、叙述者-接收者交互分析、人工评估以及表征层面的干预手段，来检验这些差异的一致性与可解释性。研究结果显示，叙事框架存在系统性差异：面向儿子的故事强调更高的能动性和更低的关系性。

    arXiv:2611.24156v1 Announce Type: cross  Abstract: Large language models are increasingly used to create stories for children. Since bedtime stories can shape children's perceptions of social roles, this raises concerns about whether LLMs adapt narrative portrayals according to the intended recipient's gender. We conducted a large-scale study to examine gender stereotyping in LLM-generated bedtime stories. Using matched daughter/son prompts and a five-generation design, we analyze 62,873 retained stories from 16 LLMs across 35 country settings and 12 narrator roles. We introduce a multi-level evaluation framework combining context-adjusted lexical rate contrasts, relative stereotype alignment, narrator-recipient interaction analysis, human evaluation, and representation-level interventions to examine both the consistency and interpretability of these differences. Our results reveal systematic differences in narrative framing: son-directed stories emphasize higher agency and lower commu
    
[^34]: 利用神经反馈信道生成与语音台词调度构建动态对话

    Constructing Dynamic Conversations with Neural Back-channel Generation and Voice-line Scheduling

    [https://arxiv.org/abs/2611.24086](https://arxiv.org/abs/2611.24086)

    该论文提出了一种神经调度器与反馈信道预测器相结合的方法，能够将两条独立录制或合成的游戏语音台词轨道合并为具有自然话轮转换和反馈信道的单一动态对话。

    

    我们提出了一种新颖的方法，用于解决将双人对话音频的两个语音通道合并为统一对话的问题，该对话具有可控的反馈信道和话轮转换动态。在游戏行业的语音表演中，两条语音台词轨道通常是单独录制或合成的。我们的方法接收两条此类不同步的轨道，并使用神经调度器将其编排成单一的动态对话，通过插入或删除停顿并在适当时引入语音重叠来修改这两条轨道。我们进一步提出了一种反馈信道预测器，能够根据给定对话轮次的音频和内容预测插入反馈信道的时机，随后调度器可以将其纳入。我们通过多种定量比较、与先前技术和朴素基线的定性及感知对比评估了调度器和预测器的独立及组合表现，并给出了一组令人信服的结果。

    arXiv:2611.24086v1 Announce Type: cross  Abstract: We present a novel approach to the problem of combining two voice-line channels of dyadic conversational audio into a unified conversation with controllable back-channeling and turn-taking dynamics. In voice performances for the game industry, the two tracks of voice-lines are typically recorded or synthesized in isolation. Our method takes two such unsynchronized tracks and uses a neural scheduler to sequence them into a single dynamic conversation, modifying both tracks by inserting or removing pauses and introducing voice overlap as appropriate. We further propose a back-channel predictor that is able to predict opportunity for back-channels for a given conversational turn's audio and content, which the scheduler can then incorporate. We evaluate our scheduler and predictor independently and in combination by: various quantitative comparison; a qualitative and perceptual comparison to prior art and naive baselines; a compelling set 
    
[^35]: 语义能影响物体选择却无法影响任务完成：VLA在何处易受地标同义词的攻击？

    Meaning Reaches Object Choice but Not Completion: Where Is a VLA Vulnerable to Landmark Synonyms?

    [https://arxiv.org/abs/2611.24076](https://arxiv.org/abs/2611.24076)

    研究发现VLA策略能够理解训练指令中未出现过的地标同义词并将其语义用于物体选择（该语义作用点位于视觉-语言模型与动作生成器之间的连接器之前），但这种措辞改写会显著降低任务完成率，且连接器之后无法通过单独注入单个模态的token来改变选择。

    

    VLA（视觉-语言-动作）策略对措辞改写十分脆弱。仅凭成功率无法判断策略究竟是没有选中正确的物体，还是无法完成任务，也无法指出应在策略的哪个环节进行干预。我们在两个仅在“取两个碗中的哪一个”的地标词上存在差异的LIBERO-Spatial任务上研究了公开发布的GR00T N1.7。与虚构的词汇不同，训练指令中从未出现过的同义词会将策略引导至其所指代的碗，但在一项任务的闭环测试中仅完成67.6%的回合，而使用原始词汇的完成率为100%。在视觉-语言模型与动作生成器之间的连接器之前，仅用地标的token向量替换为另一个地标的向量（粘贴操作），几乎使所有回合都走向另一个碗。而在连接器之后，仅对地标token、其余指令token或图像token单独进行粘贴很少改变物体选择，但任意两者结合则会改变选择。在连接器之后对动作生成器隐藏地标token，仍会降低任务完成率。

    arXiv:2611.24076v1 Announce Type: cross  Abstract: VLA policies are fragile to rewording. Success rates alone cannot say whether a policy fails to choose the object or to complete the task, nor where in the policy to intervene. We study the released GR00T N1.7 on two LIBERO-Spatial tasks differing only in the landmark word telling which of two bowls to take. Synonyms absent from the training instructions send the policy to the bowl they name, unlike made-up words, yet complete 67.6% of one task's closed-loop episodes, against the original word's 100%. Before the connector between the vision-language model and the action generator, replacing only the landmark tokens' vectors with the other landmark's (pasting) sends nearly every episode to the other bowl. After the connector, pasting at the landmark, the remaining instruction, or the image tokens alone rarely moves the choice, but any two do. Hiding the landmark tokens from the action generator after the connector still lowers completio
    
[^36]: 长上下文大语言模型服务中保留上下文的度量与计费

    Measuring and Charging for Retained Context in Long-Context LLM Services

    [https://arxiv.org/abs/2611.24071](https://arxiv.org/abs/2611.24071)

    该论文提出了一个将内容相关性、可观测代理与可审计位置权重分离的三层框架来度量长上下文LLM服务中重复上下文的实际价值，实测表明缓存命中可将重复前缀计算成本降低两个数量级以上，且校准仿真能将会话费用的99百分位降低31-42%。

    

    长上下文大语言模型（LLM）服务对重复上下文进行计费，但其计费权重未必反映其对回答质量的实际贡献。我们提出了一个三层框架，将内容相关性、可观测代理指标和可审计的位置权重分离开来。上下文消融实验表明，相关性因素主导于位置因素，且距离衰减类模型并不优于仅基于相关性的基线。基于实测的服务栈显示，在65,536个token的情况下，缓存命中可将重复前缀的计算成本降低两个数量级以上，而冷重发则限制了可持续折扣的空间。跨提供商的测量将缓存价值与分词方式和标价差异区分开来。经过校准的仿真将会话费用的第99百分位数降低了31-42%，尽管收入匹配的非线性费率也能实现类似降幅，但会使中位数费用有所上升。该框架提供了一个有界的每token计费规则以及实测数据支撑。

    arXiv:2611.24071v1 Announce Type: cross  Abstract: Long-context large language model (LLM) services charge for repeated context, yet billing weights need not reflect its contribution to answer quality. We present a three-layer framework separating content relevance, observable proxies and auditable positional weights. Context-ablation experiments show that relevance dominates position and that distance-decay families do not outperform a relevance-only baseline. A measured serving stack shows that cache hits reduce repeated-prefix compute cost by over two orders of magnitude at 65,536 tokens, while cold resends constrain sustainable discounts. Cross-provider measurements separate cache value from tokenisation and list-price differences. Calibrated simulations reduce the 99th percentile of session charges by 31-42%, although revenue-matched nonlinear tariffs achieve similar reductions and median charges increase. The framework provides a bounded per-token accounting rule and measured dep
    
[^37]: MIRAGE：动态知识注入下的多跳推理框架

    MIRAGE: A Framework for Multi-hop Reasoning under Dynamic Knowledge Injection

    [https://arxiv.org/abs/2611.24058](https://arxiv.org/abs/2611.24058)

    该论文提出了首个针对知识注入场景（同时涵盖冲突型K2与增量型K3知识）的多跳推理基准NEXUS，并构建MIRAGE框架，以解决模型因依赖错误知识来源（固守过时参数记忆、无法应对全新信息）而导致多跳推理出错的问题。

    

    多跳问答（MQA）要求大型语言模型（LLM）检索并综合多条证据，形成连贯的推理路径，这对复杂任务至关重要。然而，以往针对多跳问答的知识更新研究大多只关注简单的事实编辑，忽视了一个更具挑战性且更贴近现实的场景——知识注入——在该场景下，模型必须同时应对相互冲突的更新以及全新的信息。在这一设定下，某一跳的答案出错往往仅仅是因为模型信任了错误的知识来源：当注入的知识与模型参数记忆相冲突时，模型固守过时的参数记忆（K2）；而当所需事实是完全崭新的信息时，模型则无法作答（K3）。为了研究这一问题，我们提出了NEXUS——首个专门面向知识注入场景下多跳推理的基准，明确涵盖了冲突型（K2）和增量型（K3）知识……（摘要在此处被截断）

    arXiv:2611.24058v1 Announce Type: new  Abstract: Multi-hop Question Answering (MQA) requires large language models (LLMs) to retrieve and synthesize multiple pieces of evidence into a coherent reasoning path, which is essential for complex tasks. However, prior work on knowledge updating for MQA has largely focused on simple fact edits, overlooking a more challenging and realistic setting---\emph{Knowledge Injection}---in which the model must simultaneously accommodate both conflicting updates and entirely novel information. Under this setting, a hop-level answer is often wrong simply because the model trusts the wrong knowledge source: it clings to outdated parametric memory when the injected knowledge conflicts with it (K2), and it cannot answer when the required fact is entirely new (K3).   To investigate this, we introduce NEXUS, the first benchmark tailored for multi-hop reasoning under Knowledge Injection, which explicitly covers both conflicting (K2) and incremental (K3) knowled
    
[^38]: 基于独立记录评估可执行法律规则：过渡性条款与韩国中小企业身份认定

    Grading Executable Legal Rules Against Independent Records: Transitional Provisions and Korean SME Status

    [https://arxiv.org/abs/2611.24055](https://arxiv.org/abs/2611.24055)

    提出LexTemporis方法，以韩国上市公司10,901个企业年度的中小企业身份披露记录等独立事实为基准评估法律执行程序，发现主条文之外的过渡性条款显著提升判定一致率（从83-86%升至89-90%），并可检验争议条款的不同解读。

    

    法律正日益被转化为能够决定具体案件的程序，例如判断一家企业在特定日期是否属于中小企业（SME），而如今大型语言模型（LLM）已经开始编写这类程序。这些程序通常通过与阅读同一部法律的人所写出的答案进行对比来评分，因此程序可能仅仅因为与评分者共享相同的法律解读而获得高分，而非真正正确地执行了法律。LexTemporis 则改用实际发生的事实记录来评估这些程序：即韩国上市公司披露的10,901个企业年度的中小企业身份认定结果，以及所有相关规则的历史版本。附录还补充了来自美国所得税和德国医院最低手术量规定的对比案例。研究发现，日期对结果的影响主要来自主法条文本之外的条款：其中一项过渡性条款使程序与披露记录的一致率从83-86%提升至89-90%。这些披露记录还检验了作者对一项有争议条款的解读，在189个相关企业年度中有130个支持该解读。

    arXiv:2611.24055v1 Announce Type: cross  Abstract: Laws are increasingly turned into programs that decide cases, such as whether a firm was a small or medium enterprise (SME) on a given date, and large language models (LLMs) now write them. The programs are usually graded against answers written by someone who read the same law, so a program can score well by sharing the grader's reading rather than by getting the law right. LexTemporis grades them instead against records of what actually happened: 10,901 firm-years of SME status disclosed by Korean listed companies, with every version of the governing rules. An appendix adds comparison cases from US income tax and German hospital minimum volumes. The date changes outcomes mainly through provisions outside the main statutory text: one transitional provision raises agreement with the disclosures from 83-86% to 89-90%. The disclosures also test the author's reading of a contested provision, favouring it in 130 of 189 firm-years where rea
    
[^39]: 一种用于虚假评论检测的混合RAG框架：集成RoBERTa-LoRA、基于FAISS的检索与大语言模型推理

    A Hybrid RAG Framework for Deceptive Review Detection: Integrating RoBERTa-LoRA, FAISS-Based Retrieval, and LLM Reasoning

    [https://arxiv.org/abs/2611.24049](https://arxiv.org/abs/2611.24049)

    本文提出了一种融合RoBERTa-LoRA分类、FAISS语义检索与LLM推理的混合RAG框架用于虚假评论检测，仅经五个微调周期即在跨领域评估中取得0.90的F1分数，同时提供了证据支持的可解释性。

    

    虚假在线评论的不断增长带来了错误信息和欺诈的严重风险。本文提出了一种检索增强生成（RAG）系统，用于自动识别虚假评论。该架构集成了采用低秩适配（LoRA）微调的RoBERTa分类模块、FAISS语义向量检索，以及用于最终推理的大语言模型（LLM）。该系统在平衡以及严重类别不平衡的真实场景中进行了评估。结果表明，仅需五个训练周期的微调即可实现稳健的领域迁移，在跨领域评估中达到0.90的F1分数，优于基线模型（OpenAI Detector，F1 = 0.82），并与领域内专用SOTA模型（fakeRoBERTa，F1 = 0.97）展现出有竞争力的性能。所提出的架构在保持相当的整体检测性能的同时，提供了有证据支持的可解释性以及统计学上显著的精确率提升。

    arXiv:2611.24049v1 Announce Type: new  Abstract: The growth of deceptive online opinions poses a critical risk of misinformation and fraud. This paper proposes a Retrieval-Augmented Generation (RAG) system to automatically distinguish spam reviews. The architecture integrates RoBERTa with Low-Rank Adaptation (LoRA) classification, FAISS semantic vector retrieval, and a Large Language Model (LLM) for final reasoning. The system was evaluated across balanced and severely imbalanced real-world scenarios. Results demonstrate that five epochs of fine-tuning enable robust domain transfer, achieving an F1-score of 0.90 in cross-domain evaluation, outperforming baselines (OpenAI Detector,F1 = 0.82) and demonstrating competitive performance against specialized in-domain SOTA models (fakeRoBERTa,F1 = 0.97). The proposed architecture provides evidence-backed explainability and statistically significant precision gains while maintaining comparable overall detection performance.
    
[^40]: 压缩对低资源多语言翻译的影响

    Compression Effects on Low-Resource Multilingual Translation

    [https://arxiv.org/abs/2611.24043](https://arxiv.org/abs/2611.24043)

    该研究首次系统评估了训练后量化（FP16/INT8/INT4）对三种低资源非洲语言机器翻译的影响，发现专用MT模型以约六分之一的参数量显著优于通用大语言模型，而量化带来的质量损失仅约1 chrF。

    

    我们呈现了一项初步研究，系统表征了训练后量化对多语言机器翻译质量与推理成本的影响，研究对象为三种非洲语言——豪萨语、斯瓦希里语和约鲁巴语，涵盖三个模型系列：一个小型通用大语言模型（Qwen2.5-1.5B）、一个更大的多语言指令微调大语言模型（Aya-23-8B），以及一个专门的多语言机器翻译模型（NLLB-200-distilled-1.3B）。我们在消费级 NVIDIA T4 GPU 上，以 FP16、bitsandbytes INT8 和 NF4 INT4 三种精度，在 FLORES-200 devtest 测试集（每个方向 1,012 个句子）上进行评估，研究发现：(i) 专门的机器翻译模型在豪萨语和斯瓦希里语方向上比两个大语言模型高出 32.7-43.3 chrF，在约鲁巴语方向上高出 12.9-24.7 chrF，尽管其规模仅为 8B 指令微调大语言模型的约六分之一；(ii) 训练后量化在所有模型系列和语言方向上都将整体翻译质量保持在约 1 chrF 的损失范围内，最差的单一情况……（原文摘要在此处截断）

    arXiv:2611.24043v1 Announce Type: new  Abstract: We present a pilot study characterising the effect of post-training quantisation on multilingual machine translation quality and inference cost for three African languages - Hausa, Swahili, and Yoruba - across three model families: a small general-purpose LLM (Qwen2.5-1.5B), a larger multilingual instruction-tuned LLM (Aya-23-8B), and a dedicated multilingual MT model (NLLB-200-distilled-1.3B). Evaluating on the FLORES-200 devtest set (1,012 sentences per direction) under FP16, bitsandbytes INT8, and NF4 INT4 precisions on consumer-grade NVIDIA T4 GPUs, we find: (i) the specialised MT model outperforms both LLMs by 32.7-43.3 chrF on the Hausa and Swahili directions, and by 12.9-24.7 chrF on Yoruba, despite being roughly a sixth the size of the 8B instruction-tuned LLM; (ii) post-training quantisation preserves aggregate translation quality within approximately 1 chrF across all model families and language directions, with a single worst 
    
[^41]: 重新思考固定长度的潜空间推理

    Rethinking Fixed-Horizon Latent Reasoning

    [https://arxiv.org/abs/2611.24001](https://arxiv.org/abs/2611.24001)

    该论文发现将潜空间思维链运行至完整预设长度反而可能损害推理性能，并提出用“已实现与剩余建构性进展”来刻画推理收益，为重新思考固定长度的潜空间推理提供了依据。

    

    现有的潜空间推理方法通常在推理阶段运行预设的潜空间推理长度。直觉上，更长的推理长度应该提供更多的推理机会。然而，在这项工作中，我们发现将潜空间思维链扩展到完整的预设长度实际上可能会降低推理性能。我们观察到潜空间思维链会经历不同的功能阶段：早期的计算形成前沿推理状态，而后期的计算主要是对已有候选进行重组，并可能削弱对正确候选的支持。这些发现促使我们重新审视固定长度的潜空间推理。为此，我们通过“已实现的建构性进展”和“剩余建构性进展”来刻画潜空间思维链带来的推理收益。实证结果表明，更高的已实现建构性进展对应更高的当前推理准确率，而更多的剩余进展则对应继续潜空间思维链所能带来的更大未来收益。

    arXiv:2611.24001v1 Announce Type: cross  Abstract: Existing latent reasoners typically run for a preset latent horizon during inference. Intuitively, a longer horizon should provide more opportunity for reasoning. However, in this work, we find that extending latent Chain-of-Thought (CoT) to the full preset horizon may in fact degrade reasoning performance. We observe that latent CoT evolves through distinct functional stages, with earlier computation forming frontier reasoning states, while later computation mainly reorganizes existing candidates and can weaken gold candidate support. These findings motivate a rethink of fixed-horizon latent inference. To address this, we characterize the reasoning benefits of latent CoT by realized and remaining constructive progress. Empirically, we find that higher realized constructive progress corresponds to higher current reasoning accuracy, while more remaining progress corresponds to larger future gains from continued latent CoT. To this end, 
    
[^42]: 上下文学习与上下文外幻觉的机制分析

    A Mechanistic Analysis of In-Context Learning and Out-of-Context Hallucination

    [https://arxiv.org/abs/2611.24000](https://arxiv.org/abs/2611.24000)

    本文通过动态上下文中功能角色的统一视角，首次将上下文学习与上下文外幻觉这两种看似对立的现象联系起来进行机制分析，发现角色效应的显著影响取决于足够的贡献幅度和一致的带符号聚合这两个必要条件。

    

    大语言模型能够通过上下文学习（ICL）推断并遵循特定规则，但却可能无法忠实于所提供的证据，从而产生上下文外（OOC）幻觉。现有工作大多沿着各自独立的路线分别研究ICL和OOC幻觉的内在机制。然而，这种割裂掩盖了这些看似对立的结果背后的机制基础。在这项工作中，我们通过动态上下文中功能角色的统一视角来研究这种对比。关键观察是，角色信息可以在模型内部被表示，但不一定能可靠地约束输出。我们推导了一阶分解来刻画角色效应，表明其显著影响取决于两个必要条件，即足够的贡献幅度和一致的带符号聚合。因果分析进一步表明，角色跨度的贡献会影响涉及响应的关键token选择。

    arXiv:2611.24000v1 Announce Type: cross  Abstract: Large language models can infer and follow specific rules through in-context learning (ICL), yet may fail to remain faithful to provided evidence, giving rise to out-of-context (OOC) hallucination. Existing work has largely investigated the underlying mechanisms of ICL and OOC hallucination along separate lines. However, this separation obscures the mechanistic basis for these seemingly opposing outcomes. In this work, we study this contrast through the unified view of functional roles in dynamic context. The key observation is that role information can be internally represented without reliably constraining output. We derive a first-order decomposition to characterize the role effect, showing that its substantial influence depends on two necessary conditions, namely sufficient contribution magnitude and coherent signed aggregation. Causal analysis further shows that role span contributions affect key token choices involving response s
    
[^43]: ReflectRec：面向基于大语言模型推荐的循环一致性偏好验证

    ReflectRec: Cycle-Consistency Preference Verification for LLM-based Recommendation

    [https://arxiv.org/abs/2611.23969](https://arxiv.org/abs/2611.23969)

    ReflectRec通过偏好-历史往返验证器对LLM生成的偏好笔记进行循环一致性校验，结合低质量偏好的条件性再生成与行为相似用户未来行为的前瞻检索，在MovieLens数据集上显著提升推荐性能。

    

    思维链（CoT）风格的大语言模型推荐器会生成一段自然语言偏好笔记，作为排序前的思维链前缀，但该笔记在未经验证的情况下就被直接信任，且排序器从不参考行为相似的用户。我们提出了ReflectRec，其核心是一个“偏好到历史”的往返验证器，该验证器的召回分数可作为偏好可恢复性的操作性信号，从而支持对低可恢复性偏好进行条件性再生成，并在多次再生成尝试中做出选择。围绕该验证器，我们构建了语义化的第一阶段提示、条件性再生成机制，以及“前瞻检索”，后者将行为相似用户的未来观看记录作为协同信号注入。在MovieLens-100K和MovieLens-1M数据集上，完整流水线在Hit@10指标上分别以+16.25和+4.46个百分点的优势超越CoT4Rec-ICL，且在所有ValScore分桶以及三种大语言模型骨干（Qwen3-30B、Qwen2.5-72B和Llama-3.3-70B）上均取得正向提升。我们还观察到类似的验证……

    arXiv:2611.23969v1 Announce Type: new  Abstract: CoT-style LLM recommenders generate a natural-language preference note as a chain-of-thought prefix to ranking, but the note is trusted without verification, and the ranker never consults behaviorally similar users. We propose ReflectRec, organized around a preference-to-history round-trip verifier whose recall score serves as an operational signal of preference-to-history recoverability, enabling conditional regeneration for low-recoverability preferences and selection among regeneration attempts. Around this verifier, we wire a semantic Stage-1 prompt, conditional regeneration, and Forward-Look retrieval that injects future watches of behaviorally similar users as a collaborative hint. On MovieLens-100K and MovieLens-1M, the full pipeline outperforms CoT4Rec-ICL by +16.25 / +4.46 pp Hit@10, with positive gains across all ValScore buckets and three LLM backbones (Qwen3-30B, Qwen2.5-72B, and Llama-3.3-70B). We also observe a similar veri
    
[^44]: 覆盖率决定准确性：低资源满文OCR中何时扩展词典、何时回退

    Coverage Governs Accuracy: When to Expand the Lexicon and When to Fall Back in Low-Resource Manchu OCR

    [https://arxiv.org/abs/2611.23900](https://arxiv.org/abs/2611.23900)

    该研究提出基于词根-后缀链形态扩展的词典约束解码方法，用于低资源满文OCR，证明词典覆盖率决定识别准确率，将未见词形的准确率从63.6%大幅提升至86.3%。

    

    低资源满文的字符级OCR常常产生近似拼写错误。词典约束解码可以利用词汇知识纠正这类错误，但固定词典无法覆盖黏着式构词所产生的众多未见词形。我们研究形态扩展能否缓解这一局限，以及扩展规模与置信度回退如何影响识别效果。我们使用基于词典的词干表对训练转写文本进行分词，从中提取词根与后缀链，生成约51万个候选词，通过在验证集上的频率排序确定扩展规模，并将硬词典约束下的准确率分解为覆盖率与词典内精度的乘积。在三个CRNN训练随机种子上，向约2.9万词的基础词典添加25万候选词，使未见词根组合上的词形准确率从63.6%提升至86.3%，整体词元准确率也随之提升。

    arXiv:2611.23900v1 Announce Type: new  Abstract: Character-level OCR for low-resource Manchu often produces near-miss spelling errors. Lexicon-constrained decoding can correct such errors with lexical knowledge, but a fixed dictionary cannot cover the many unseen word forms produced by agglutinative word formation. We study whether morphological expansion alleviates this limitation, and how the expansion size and confidence fallback affect recognition. We segment the training transcriptions with a dictionary-based stem list, extract roots and suffix chains from them, generate about 510k candidates, select the expansion size through frequency ranking on the validation set, and decompose accuracy under hard lexicon constraints into the product of coverage and in-lexicon precision. Across three CRNN training seeds, adding 250k candidates to a base dictionary of about 29k words raises the type accuracy on unseen combinations of seen roots from 63.6% to 86.3%, and overall token accuracy fro
    
[^45]: NarrowSpec：推测解码中草稿模型的隐藏宽度无需与目标模型匹配

    NarrowSpec: Draft Hidden Width Need Not Match the Target in Speculative Decoding

    [https://arxiv.org/abs/2611.23835](https://arxiv.org/abs/2611.23835)

    提出NarrowSpec，证明推测解码中读取目标隐藏状态的草稿模型的隐藏宽度可缩小至目标模型的一半或四分之一而几乎不影响接受率，从而在B200和DGX Spark上实现1.06-1.44倍的端到端解码加速。

    

    推测解码被广泛用于加速大语言模型（LLM）的生成，其方式是由一个小型草稿模型提出候选token，再由目标模型的一次前向传播进行验证。读取目标模型隐藏状态的草稿模型（如DFlash和EAGLE-3）通常继承目标模型的隐藏宽度。我们证明，草稿模型的整个残差流可以在该宽度的一半或四分之一下运行，同时基本保持token接受率。对于借用目标模型词表投影的DFlash，训练得到的输出适配器可以折叠进该投影中，形成一个更窄的投影，其logits在浮点舍入误差范围内保持不变。我们在SGLang中，于B200和DGX Spark（GB10）硬件上，针对两个目标模型Qwen3-8B（NVFP4）和GPT-OSS-20B（MXFP4），测试了DFlash和EAGLE-3两个草稿模型系列。在批量大小为1、温度为0的条件下，将草稿模型收窄至一半或四分之一宽度，可使端到端解码在B200上提速1.06-1.12倍，在GB10上提速1.13-1.44倍。该加速效果……（原文摘要在此处截断）

    arXiv:2611.23835v1 Announce Type: new  Abstract: Speculative decoding is widely used to speed up generation in large language models (LLMs), with a small draft model proposing tokens that one forward pass of the target model verifies. Draft models that read the target's hidden states, such as DFlash and EAGLE-3, typically inherit the target's hidden width. We show that the draft's whole residual stream can run at half or a quarter of that width while largely preserving acceptance. For DFlash, which borrows the target's vocabulary projection, the trained output adapter folds into it, giving one narrower projection with logits unchanged up to floating-point rounding. We test two drafter families, DFlash and EAGLE-3, on two targets, Qwen3-8B (NVFP4) and GPT-OSS-20B (MXFP4), in SGLang on a B200 and a DGX Spark (GB10). At batch size 1 and temperature 0, narrowing the draft to half or a quarter width speeds up end-to-end decoding by 1.06-1.12x on B200 and 1.13-1.44x on GB10. The speedup hold
    
[^46]: 基于冻结语言模型的医学多选题“系统1”决策模型

    A System 1 Decision Model for Medical Multiple-Choice Questions Built on Frozen Language Models

    [https://arxiv.org/abs/2611.23832](https://arxiv.org/abs/2611.23832)

    该研究首次将基于冻结语言模型的单次“系统1”决策方法应用于需要医学知识的MedMCQA考试题，仅通过训练两个小型投影头和温度参数，就在小型模型上取得了显著高于随机水平的答题准确率。

    

    “系统1”模型在单次前向传递中回答多选题。冻结的语言模型将问题和每个选项转换为向量，两个经过训练的小型头将它们映射到共享空间中，距离最近的选项即为答案。该设计已在智能体决策中得到验证，但据我们所知，尚无研究探讨该方法在需要医学知识的考试题目上能走多远——在这类题目中，正确选项与问题的相似度往往并不比错误选项更高。我们为MedMCQA从头重建了该方法，仅在冻结的Qwen3.5-0.8B、Granite-4.2-3B和Qwen3-8B模型之上训练两个投影头和一个温度参数，采用带有批内负样本和困难负样本的对比损失，并使用掩码处理重复答案。使用简单的末词元向量时，两个小型编码器在3,842个保留题目上约每三题答对一题（准确率分别为33.1%和33.0%，而随机猜测的基线为25.0%）。通过将每段文本嵌入单词摘要提示中并降低学习率……（摘要在此处被截断）

    arXiv:2611.23832v1 Announce Type: new  Abstract: A System 1 model answers a multiple-choice question in a single pass. A frozen language model turns the question and each option into vectors, two small, trained heads place them in a shared space, and the closest option wins. The design has been shown on agent decisions, but we are not aware of a study of how far it goes on exam questions that need medical knowledge, where the right option is often no more similar to the question than the wrong ones. We rebuilt the recipe from scratch for MedMCQA and trained only two projection heads and one temperature on top of frozen Qwen3.5-0.8B, Granite-4.2-3B and Qwen3-8B, using a contrastive loss with in-batch and hard negatives and a mask for repeated answers. With plain last-token vectors, both small encoders answer about one question in three correctly (33.1% and 33.0% on 3,842 held-out questions, where chance is 25.0%). Writing each text inside a one-word-summary prompt and lowering the learn
    
[^47]: Sightline：用于快速SVG生成的长窗口投机解码

    Sightline: Long-Window Speculative Decoding for Fast SVG Generation

    [https://arxiv.org/abs/2611.23807](https://arxiv.org/abs/2611.23807)

    Sightline通过将验证窗口延长至草稿器的预测范围并配合足够宽的草稿树，利用SVG生成中草稿错误稀少且聚集的特性，实现了当前最快的无损SVG解码。

    

    可缩放矢量图形（SVG）生成模型以代码形式编写图形，每个token需要一次前向传播，因此绘制速度很慢。投机解码可以无损地加速这一过程，而块级草稿器每次前向传播可猜测多个token。在SVG上，它们的许多轮次会接受整个草稿，并且只在验证窗口边界处结束。我们发现草稿的错误很少且呈聚集分布，并且在块内的准确深度远超自然语言场景。一个游程长度恒等式和三个命题表明，一旦候选足够宽，更长的验证窗口就会带来收益，其上限为草稿器的预测范围。我们提出Sightline，它将等于草稿器预测范围长度的验证窗口与足够宽以填满该窗口的草稿树相结合。实验涵盖文本到SVG和图像到SVG任务，使用三个模型在来自MMSVG、MMSVG-2M和SArena的六个基准上进行。在所有设置中，Sightline都是所比较的无损解码器中最快的。在SArena上，即使所有解码器共享同一个……（摘要截断）

    arXiv:2611.23807v1 Announce Type: cross  Abstract: Scalable Vector Graphics (SVG) generation models write a graphic as code with one forward pass per token, so drawing is slow. Speculative decoding speeds them up losslessly, and block drafters guess many tokens per pass. On SVG, many of their rounds accept the whole draft and end only at the verification window. We find that their misses are rare and clumped, and that they stay accurate far deeper into a block than on natural language. A run-length identity and three propositions show that a longer window pays once the candidates are wide, up to the drafter's horizon. We propose Sightline, which pairs a window as long as the drafter's horizon with a draft tree wide enough to fill it. Experiments cover text-to-SVG and image-to-SVG with three models on six benchmarks from MMSVG, MMSVG-2M and SArena. In every setting, Sightline is the fastest of the compared lossless decoders. On SArena, it stays the fastest when every decoder shares one 
    
[^48]: 面向头部锥束CT盲刚性运动校正的几何桥接流匹配方法

    Geometry-bridge flow matching for blind rigid-motion correction in head cone-beam CT

    [https://arxiv.org/abs/2611.23788](https://arxiv.org/abs/2611.23788)

    提出一种基于几何桥接的3D流匹配先验方法，结合样条位姿估计、共轭梯度数据一致性和全变差正则化的即插即用循环框架，实现头部锥束CT的盲刚性运动校正。

    

    arXiv:2611.23788v1 公告类型： cross 摘要：在头部锥束CT（CBCT）扫描过程中，患者头部的刚性运动会降低图像质量，并使从测量的投影数据中恢复精确的解剖结构变得困难。我们提出了一种3D流匹配先验方法，用于学习运动引起的图像变化。其“几何桥接”机制通过逐渐减弱的模拟运动，将同一源体积的运动污染FDK（Feldkamp-Davis-Kress）重建图像与无运动FDK重建图像连接起来。前向投影器的几何导数沿该路径提供解析速度目标。一个即插即用的循环框架将学习到的图像更新与基于样条的位姿估计、共轭梯度数据一致性约束以及图像全变差正则化相结合。研究对优化后的图像体积和在估计位姿下的FDK重建图像体积均进行了评估。实验基于来自30名留出CQ500患者的模拟360视角扫描数据，名义单分量运动幅度...

    arXiv:2611.23788v1 Announce Type: cross  Abstract: Rigid motion of a patient head during head cone-beam CT (CBCT) scan can degrade image quality and complicate recovery of accurate anatomical structures from the measured projections. We present a 3D flow-matching prior approach that learns the image changes induced by motion. Its geometry-bridge connects motion-corrupted and motion-free Feldkamp-Davis-Kress (FDK) reconstructed images of the same source volume through progressively reduced simulated motion. The forward projector's geometry derivative supplies analytic velocity targets along this path. A plug-and-play loop encompasses the learned image updates together with spline-based pose estimation, conjugate-gradient data consistency, and image total-variation regularization. Both the refined image volume and the FDK reconstructed image volume at the estimated poses are evaluated. On simulated 360-view scans from 30 held-out CQ500 patients with nominal per-component motion amplitude
    
[^49]: 随时随地快速思考：面向任意智能体工作流决策点的统一System-1模型

    Thinking Fast, Anywhere: One System-1 for the Decision Points of Any Agent Workflow

    [https://arxiv.org/abs/2611.23751](https://arxiv.org/abs/2611.23751)

    AnyS1将人类编写的程序性技能转化为领域通用的决策点来源，训练出一个统一的4B轻量System-1模型，使任意智能体工作流在运行时通过一次前向传播即可完成常规决策，并在置信度不足时回退到大模型System-2。

    

    自主运行的LLM智能体在每一步都调用大模型（System-2），即便许多中间动作只需在一小组已知候选项中进行选择，也会带来高昂的成本和延迟。现有的双系统智能体将此类决策卸载给轻量级的System-1模型，但依赖于针对每个环境手工编写的工作流特定钩子。我们提出AnyS1，这是一个框架，它将人类编写的程序性技能——其指令天然指明了选择出现的位置及可用选项——转化为共享的、领域通用的决策点来源。在离线阶段，AnyS1从超过1,000个公开技能中挖掘有依据的可枚举决策，在隔离容器中收集基于执行轨迹的状态以构建Any-DP数据集，并通过标签平滑训练出一个经过校准的单一4B指针式System-1模型。在运行时，编译后的工作流仅需一次前向传播即可完成常规决策，当置信度低于校准阈值时则回退至System-2。

    arXiv:2611.23751v1 Announce Type: new  Abstract: Autonomous LLM agents incur substantial cost and latency by invoking a large model (System-2) at every step, even when many intermediate actions choose among a small set of known candidates. Existing dual-system agents offload such decisions to a lightweight System-1 model, but rely on workflow-specific hooks authored per environment. We present AnyS1, a framework that turns human-authored procedural skills, whose instructions naturally specify where choices arise and which options apply, into a shared, domain-general source of decision points. Offline, AnyS1 mines grounded enumerable decisions across over 1,000 public skills, gathers execution-grounded trajectory states inside isolated containers to construct Any-DP, and trains a single calibrated 4B pointer System-1 with label smoothing. At run time, compiled workflows resolve routine decisions in one forward pass and fall back to System-2 when confidence falls below a calibrated thres
    
[^50]: 超越替换：面向扩散语言模型的Levenshtein编辑

    Beyond Replacement: Levenshtein Editing for Diffusion Language Models

    [https://arxiv.org/abs/2611.23750](https://arxiv.org/abs/2611.23750)

    提出LED后训练框架，通过在扩散语言模型的“仅替换编辑”基础上扩展插入与删除操作，实现对生成序列的莱文斯坦式精炼，克服了单一替换编辑在处理序列偏移和多位置错误时的局限。

    

    扩散语言模型为自回归模型提供了一种灵活的替代方案，支持任意顺序的文本填充。虽然大多数现有的解码方法将已解码的词元视为固定不变，但一些近期的模型允许对先前解码的词元进行修订——但仅限于用其他词元替换它们，我们将这种范式称为“仅替换编辑”。尽管ROE能够纠正单个词元，但它对于由词元缺失或多余引起的序列偏移处理效率低下，并且在错误出现在多个位置时表现不佳。为了解决这些局限性，我们提出了LED（Levenshtein Editing for Diffusion Language Models，面向扩散语言模型的莱文斯坦编辑），这是一个后训练框架，通过引入插入和删除操作扩展了ROE，实现对生成序列的莱文斯坦式精炼。该框架采用两种训练设计：首先，在监督微调过程中利用最长公共子序列对齐实时构建编辑监督信号，从而支持插入和删除的学习……（原文摘要截断）

    arXiv:2611.23750v1 Announce Type: cross  Abstract: Diffusion language models (DLMs) offer a flexible alternative to autoregressive models, enabling any-order infilling. While most existing decoding methods treat decoded tokens as fixed, some recent models allow previously decoded tokens to be revised only by replacing them with alternative tokens, a paradigm we refer to as replacement-only editing (ROE). Although ROE can correct individual tokens, it is inefficient for sequence shifts caused by missing or extra tokens and struggles when errors occur across many positions. To address these limitations, we propose LED (Levenshtein Editing for Diffusion Language Models), a post-training framework that extends ROE with insertion and deletion, enabling Levenshtein-style refinement of generated sequences. It uses two training designs. First, it reconstructs edit supervision on the fly during supervised fine-tuning using longest-common-subsequence alignment, enabling insertion and deletion le
    
[^51]: 通过辨识进行学习：小型语言模型何时能从用户在岗反馈中学习，何时会退化

    Learning through Discernment: when a small language model learns from user feedback on the job, and when it decays

    [https://arxiv.org/abs/2611.23716](https://arxiv.org/abs/2611.23716)

    该研究发现，设备端小型语言模型在夜间从用户反馈中自我微调学习时，采用“先判断用户是否给出明确评价”的辨识式反馈感知机制，可使模型在两周内保持接近最优性能，但从长期看模型仍普遍呈现先提升后衰退的规律。

    

    设备端的小型语言模型可以通过在夜间从自己的日常工作中学习来持续改进。然而，在缺乏真实结果的情况下，必须有某种机制来决定哪些尝试值得学习，而模型自身的判断和用户的反应都可能是错误的。我们研究了一个配备LoRA适配器的小型模型（Qwen3 0.6B和1.7B），每晚根据裁判所接受的尝试进行微调，在WebShop和τ²-bench上进行了127次长达一年的生命周期实验。我们将真实结果、用户明确给出的评价以及模型自身的判断，与一个单独的、冻结的“反馈感知”模块进行了比较，该模块用于解读用户的反应，共构建了三个版本。就自我判断而言，无论使用何种适配器，模型在长达一年的时间里都表现出先提升后衰退的现象。一个将礼貌性放弃误认为成功的感知版本导致5次生命周期中的3次崩溃；而一个为辨识而构建的版本——它首先判断用户是否给出了明确的评价——使5次生命周期中的4次在两周内保持在接近最优水平，随后在干净但……（原文摘要在此处截断）

    arXiv:2611.23716v1 Announce Type: cross  Abstract: Small language models on devices could keep improving by learning, overnight, from their own daily work. Without a true outcome, however, something must decide which attempts are worth learning from, and both the model's own judgment and the user's reaction can be wrong. We study a small model (Qwen3 0.6B and 1.7B) with a LoRA adapter, fine-tuned each night on the attempts a judge accepts, over 127 lives of up to a year on WebShop and $\tau^2$-bench. We compare the true outcome, the user's stated verdict and the model's own judgment with a separate, frozen "feedback sense" that reads the user's reaction, built in three versions. Judging itself, the model improved and then decayed, over a year and whatever its adapter. A sense that took polite give-ups for success collapsed 3 of 5 lives; one built for discernment, which first asks whether the user gave a clear verdict, kept 4 of 5 near the oracle for two weeks, then faded on clean but s
    
[^52]: 面向可信人工智能的文本匿名化本地化：将 RAT-Bench 扩展至马来西亚微数据与个人身份信息（PII）

    Localizing Text Anonymization for Trustworthy AI: Extending RAT-Bench to Malaysian Microdata and PII

    [https://arxiv.org/abs/2611.23704](https://arxiv.org/abs/2611.23704)

    该研究将文本匿名化基准 RAT-Bench 本地化至马来西亚微数据与 PII 格式（涵盖马来西亚英语和马来语），并借助 LLM 攻击者评估发现，基于 LLM 的匿名化工具在隐私保护与文本效用之间实现了最佳权衡。

    

    随着各国政府和组织采用基础模型来提供公共服务和提升工作场所生产力，敏感的公民、员工及行政文本可能会通过推理、检索、微调或本地模型开发等途径进入大语言模型（LLM）工作流程。我们通过将 RAT-Bench 扩展至马来西亚情境，使用本地微数据、马来西亚个人身份信息（PII）格式，以及基于本地文化背景的马来西亚英语和马来语转录文本，来检验文本匿名化工具能否迁移到新的部署环境中。我们采用一个从匿名化文本中推断属性的 LLM 攻击者，对基于 NER 和基于 LLM 的匿名化工具进行评估，同时测量重识别成功率和文本效用。在马来西亚英语和马来语两种语言场景下，两个基于 LLM 的匿名化工具提供了最强的显式隐私-效用权衡，将简单/困难重识别风险降低至 23-29%，同时保持 0.77-0.94 的 BLEU 分数。非 LLM 工具则表现出更尖锐的失效模式，要么在保持效用的同时……

    arXiv:2611.23704v1 Announce Type: new  Abstract: As governments and organizations adopt foundation models for public services and workplace productivity, sensitive citizen, employee, and administrative text may enter LLM workflows through inference, retrieval, fine-tuning, or local model development. We examine whether text anonymizers transfer to new deployment settings by extending RAT-Bench to Malaysia using local microdata, Malaysian PII formats, and culturally grounded transcripts in Malaysian English and Bahasa Malaysia. We evaluate NER- and LLM-based anonymizers using an LLM attacker that infers attributes from anonymized text, measuring both re-identification success and text utility. Across Malaysian English and Bahasa Malaysia, the two LLM anonymizers provide the strongest explicit privacy-utility trade-offs, reducing Easy/Hard re-identification risk to 23-29% while preserving BLEU scores of 0.77-0.94. Non-LLM tools show sharper failure modes, either preserving utility while 
    
[^53]: 通信式在策略蒸馏：通过通信为长时程任务蒸馏黑盒教师模型的能力

    Communicative On-Policy Distillation: Distilling Black-Box Teacher Capabilities through Communication for Long-Horizon Tasks

    [https://arxiv.org/abs/2611.23673](https://arxiv.org/abs/2611.23673)

    提出通信式在策略蒸馏（COPD），以自然语言通信为媒介，让学生模型自主决定何时向黑盒教师提问，并将成功的师生交互用于在策略自蒸馏与任务奖励强化学习相结合的训练，从而在长时程任务上取得最强黑盒蒸馏基线 2.46 倍的增益。

    

    在策略蒸馏是一种提升语言模型智能体在长时程任务上表现的强大范式，然而其传统形式依赖于教师模型的 logits，而仅支持文本的黑盒模型并不会暴露这些信息。为解决这一局限，我们提出了通信式在策略蒸馏，它以自然语言通信为媒介，将更强大模型的能力蒸馏到更小的学生模型中。在任务执行过程中，学生模型自主决定何时咨询黑盒教师，并在没有预定义问题模板或干预时间表的情况下提出开放式的自然语言请求。成功的学生-教师交互会被保留为特权参考，通过在策略自蒸馏来监督学生在无顾问情况下的全新推演，并与任务奖励强化学习联合进行。在 CAMG-II 上，COPD 取得了最强黑盒 OPD 基线 2.46 倍的增益。

    arXiv:2611.23673v1 Announce Type: cross  Abstract: On-policy distillation (OPD) is a powerful paradigm for improving language-model agents on long-horizon tasks, yet its conventional formulation relies on teacher logits that text-only black- box models do not expose. To address this limitation, we introduce Communicative On-Policy Distillation (COPD), which uses natural-language communication as a medium for distilling a stronger model's capabilities into a smaller student. During task execution, the student au- tonomously decides when to consult a black-box teacher and formulates open-ended natural- language requests without predefined question templates or intervention schedules. Successful student-teacher interactions are retained as privileged references to supervise fresh adviser-free student rollouts through on-policy self-distillation, jointly with task-reward reinforcement learning. On CAMG-II, COPD achieves 2.46 times the gain of the strongest black-box OPD baseline. The capab
    
[^54]: 安全个人RSI：评估与修复个人智能体更新

    Safe Personal RSI: Evaluating and Repairing Personal-Agent Updates

    [https://arxiv.org/abs/2611.23651](https://arxiv.org/abs/2611.23651)

    本文提出Safe Personal RSI更新评估机制，通过将递归自我改进限制在提示词层面、固定模型权重与代码，并建立“测试—修复—验证”流程，在29个案例的基准上成功拒绝了21次高风险更新尝试中的20次，保障个人智能体更新后的安全性与任务性能。

    

    Kicker是一个个人健康助手。对助手可复用指令的更改可以在改进任务处理的同时，改变它所披露的内容、它遵循的请求以及它报告操作的方式。我们研究了Safe Personal RSI，一个为Kicker开发的更新评估组件。我们将递归自我改进（RSI）限制在提示词更新的范围内，模型权重和运行时代码保持固定。该流程测试提议的指令与当前指令，重复失败的案例，临时撤销更改以指导修复，并在采纳前检查安全性和任务性能。如果没有版本通过，则继续使用当前指令。我们的可执行基准测试包含29个电子邮件、日历、健康和市场案例，并设有针对授权复用、记忆和工具输出注入的独立探测。在三个模型和两次故意设置的风险编辑中，该流程拒绝了21次更新尝试中的20次。在修复确认中，一个被接纳的GPT-5.6 Luna rep（原文摘要在此处截断）

    arXiv:2611.23651v1 Announce Type: cross  Abstract: Kicker is a personal health assistant. Changes to an assistant's reusable instructions can improve task handling while changing what it discloses, which requests it follows, and how it reports actions. We study Safe Personal RSI, an update-evaluation component developed for Kicker. We restrict recursive self-improvement (RSI) to prompt updates, with model weights and runtime code fixed. The procedure tests proposed and current instructions, repeats failing cases, temporarily undoes changes to guide repairs, and checks safety and task performance before adoption. If no version passes, the current instructions stay in use. Our executable benchmark contains 29 email, calendar, health, and marketplace cases, with separate probes for authorization reuse, memory, and tool-output injection. Across three models and two deliberately risky edits, the procedure rejects 20 of 21 update attempts. In repair confirmation, an admitted GPT-5.6 Luna rep
    
[^55]: Option-GRPO：利用分层强化学习应对未见SDK代码生成中的AI垃圾代码问题

    Option-GRPO: Hierarchical RL Against AI Slop in Code Generation for Unseen SDKs

    [https://arxiv.org/abs/2611.23633](https://arxiv.org/abs/2611.23633)

    该论文提出Option-GRPO分层强化学习框架，仅凭SDK源代码（无需人工示例或测试用例）即可将代码大模型适配到未见过的SDK，通过从源代码中挖掘可复用代码模式（code options）作为课程与动作空间，同时优化功能正确性与代码简洁优雅性，从而抑制AI垃圾代码的产生。

    

    大型语言模型在生成的代码中越来越多地产生“AI垃圾代码”：冗长、冗余、非惯用的代码实现，这些代码虽能通过功能检查，却会使代码库臃肿不堪。这一问题在模型训练截止日期之后发布的SDK上尤为严重，因为模型既缺乏惯用写法的先验知识，也缺乏对SDK的组合性理解。我们提出了Option-GRPO，这是一个分层强化学习框架，仅凭SDK源代码即可将代码大语言模型适配到未见过的SDK上，无需人工编写的使用示例或测试用例，并且明确地同时优化功能正确性与代码优雅性。我们的核心洞察在于：SDK源代码中蕴含的组合性结构定义了一个可复用代码模式（代码选项）的层次体系，这些代码选项既可以作为课程学习的构建模块，也可以作为高效策略优化的动作空间。我们证明了代码选项可以诱导出有效的半马尔可夫决策过程，并给出了条件边界……

    arXiv:2611.23633v1 Announce Type: cross  Abstract: Large language models increasingly produce AI slop in generated code: verbose, redundant, non-idiomatic implementations that pass functional checks but bloat codebases. The problem compounds on SDKs released after the model's training cutoff, where the model has neither idiomatic priors nor compositional understanding. We present Option-GRPO, a hierarchical reinforcement learning framework that adapts a code LLM to an unseen SDK from its source code alone, without human-written usage examples or test cases, and that explicitly optimizes for both functional correctness and code elegance. Our key insight is that SDK source code contains compositional structure that defines a hierarchy of reusable code patterns (code options) which simultaneously serve as the building blocks of a curriculum and the action space for efficient policy optimization. We show that code options induce a valid semi-Markov decision process, give conditional bounds
    
[^56]: FutureBench：一个用于评估大语言模型知识-上下文冲突的可再生框架

    FutureBench: A Renewable Framework for Evaluating Knowledge-Context Conflict in LLMs

    [https://arxiv.org/abs/2611.23601](https://arxiv.org/abs/2611.23601)

    FutureBench是一个利用时间戳问题评估大语言模型知识与检索上下文冲突的可再生基准，研究发现“检查+段落优先”提示会增加模型采纳错误答案的倾向，而具有对抗意识的提示可以恢复大部分被替换的正确答案。

    

    要求模型检查检索到的文本，同时也可能告诉它应该信任哪个来源。我们提出FutureBench，一个使用带时间戳的问题、受控段落三元组和无上下文探针来评估知识-上下文冲突的可再生框架。在3,629个问题和六个模型上，Verify（一种结合检查与段落优先级的提示方法）在截止日期前事件上测试的所有五个模型中都提高了错误答案的采纳率，主要是通过替换掉在Baseline（允许同时使用段落证据和模型知识）条件下原本正确的答案。在截止日期后的事件上，只有o4-mini提高了采纳率（+13.7个百分点）。在902个截止日期后问题上进行的四种提示对比发现，不含检查的段落优先级会提高采纳率，而转向Verify则在所有五个模型中都降低了采纳率。Adversarial-Aware（会警告存在误导性段落并优先考虑模型知识）恢复了82.5-98.9%被替换的截止日期前答案。

    arXiv:2611.23601v1 Announce Type: cross  Abstract: Asking a model to check retrieved text can also tell it which source to trust. We introduce FutureBench, a renewable framework for evaluating knowledge-context conflict using timestamped questions, controlled passage triplets, and no-context probes. Across 3,629 questions and six models, Verify (a prompt combining checking with passage priority) raises false-answer adoption in all five models tested on pre-cutoff events, mainly by displacing answers that were correct under Baseline, which allows both passage evidence and model knowledge. On post-cutoff events, only o4-mini increases adoption (+13.7 percentage points). A four-prompt comparison on 902 post-cutoff questions finds that passage priority without checking raises adoption, while moving to Verify lowers it in all five models. Adversarial-Aware, which warns of misleading passages and prioritizes model knowledge, restores 82.5-98.9% of displaced pre-cutoff answers with under one 
    
[^57]: 可以安全省略，也值得省略吗？当LLM智能体跳过另一方的消息时，保持答案不变并降低延迟

    Safe to Omit, Worth Omitting? Keeping the Answer and Cutting Latency When an LLM Agent Skips Another's Message

    [https://arxiv.org/abs/2611.23570](https://arxiv.org/abs/2611.23570)

    该论文提出一种基于无消息置信度读取与有限样本统计检验的机制，使多智能体LLM系统中的接收方智能体能够以高概率保证答案改变率不超过5%地安全跳过辅助智能体的消息，在保持最终答案质量的同时有效降低通信延迟，并揭示了消息增益远小于oracle上限的量化证据。

    

    在多智能体系统中，接收方智能体通常需要等待辅助智能体的消息（文本或其潜在KV缓存），这可以提升性能但会增加延迟。我们研究接收方智能体何时可以安全地（最终答案很少改变）且有益地（节省的时间超过决策本身的开销）跳过该消息。我们的机制在不读取该消息的情况下获取接收方智能体的置信度，并使用有限样本检验来认证一个阈值——在该阈值下，以高概率保证被跳过时答案的改变率不超过5%；如果没有阈值能通过检验，系统则继续保持通信。我们在OpenBookQA、ARC-Challenge和MMLU-Pro上使用三对Qwen模型进行测试，也在跨家族的智能体组合以及SQuAD和GSM8K上进行了测试。我们得到四点洞察。首先，“oracle上限空间”是消息有帮助作用的弱证据：不含答案特定内容、但被匹配为与真实消息同样频繁改变答案的控制输入，能够重现oracle收益的中位数76%。其次，一条消息能否……（摘要原文在此处截断）

    arXiv:2611.23570v1 Announce Type: cross  Abstract: In multi-agent systems, a receiver agent often waits for a helper agent's message (text or its latent KV cache), which can improve performance but adds latency. We study when a receiver agent can skip the message safely (the final answer rarely changes) and beneficially (the time saved exceeds the cost of deciding). Our mechanism reads the receiver agent's confidence without the message and uses a finite-sample test to certify a threshold at which, with high probability, at most 5% of skipped answers change; if none passes, the system keeps communicating. We test it on OpenBookQA, ARC-Challenge, and MMLU-Pro with three Qwen model pairs, on cross-family agents, and on SQuAD and GSM8K. Four insights emerge. First, oracle headroom is weak evidence that messages help: control inputs without answer-specific content, matched to change answers as often as real messages, reproduce a median 76% of the oracle's gain. Second, whether a message ca
    
[^58]: 语言模型让新研究听起来“早已被研究过”

    Language Models Make New Research Sound Old

    [https://arxiv.org/abs/2611.23546](https://arxiv.org/abs/2611.23546)

    大型语言模型在验证研究问题时存在系统性“报告差距”——虽能准确判断问题未被研究过却很少明确说明（差距达30-90个百分点且能力越强差距越大），通过将模型判断与对引用论文的单独核查相结合可有效修复这一不对称的报告失败。

    

    研究者们正越来越多地使用大型语言模型（LLM）来验证研究问题。我们发现，LLM 会把尚未被研究过的开放性研究问题表述得好像已被部分研究过，并夸大相关论文所确立的结论。当被要求单独查阅论文全文时，模型能够识别出其中大部分的夸大表述。在从新录用论文中抽取的 40 个问题上，GPT-6-Astra 能正确判断其中 98% 的问题是未被研究过的，但在普通回答中仅有 8% 明确说明这一状态。所有六个前沿系统均表现出 30-90 个百分点的“报告差距”。在 22 个被给予相同论文的模型中，模型能力越强，判断越准确，但报告差距也越大。这种报告失败是不对称的：当我们提供回答问题的相关论文时，模型在普通回答中能以 88-100% 的正确率识别出已被研究过的问题。为纠正这一问题，我们基于每个模型的判断以及对所引论文的单独核查构建了一个固定的回答，这将 Astra 的……

    arXiv:2611.23546v1 Announce Type: cross  Abstract: Researchers are increasingly using large language models (LLMs) to validate research questions. We find that LLMs make open research questions sound partly studied and overstate what related papers establish. Models recognize most of these overstatements when asked to check the papers' full texts separately. On 40 questions drawn from newly accepted papers, GPT-6-Astra correctly judges 98% to be unstudied, but states that status plainly in only 8% of ordinary answers. All six frontier systems show reporting gaps of 30-90 percentage points. Across 22 models given identical papers, higher capability is associated with more accurate judgments and larger reporting gaps. This reporting failure is asymmetric: models correctly identify studied questions in 88-100% of ordinary answers when we supply the answering papers. To correct this, we build a fixed answer from each model's judgment and separate checks of cited papers. This raises Astra's
    
[^59]: LLM能理解它们所预测的内容吗？评估金融市场中的潜在因果结构推断

    Can LLMs Understand What They Predict? Evaluating Latent Causal Structure Inference in Financial Markets

    [https://arxiv.org/abs/2611.23533](https://arxiv.org/abs/2611.23533)

    该论文提出LUCID基准，评估大语言模型能否从事件描述和历史金融数据中推断潜在的因果市场结构，发现模型生成的因果图通常无法改善重构质量，表明预测成功并不等于真正理解了底层机制。

    

    大语言模型（LLM）越来越多地通过其预测未来结果的能力来评估，这隐含地假设预测的成功反映了对底层系统的理解。然而，准确的预测往往可以在不恢复产生可观测结果的潜在机制的情况下实现。为了检验这一假设，我们引入了LUCID，这是一个评估模型能否从事件描述和历史金融观测数据中推断潜在市场结构的基准。模型生成代表所推断市场结构的事件诱导因果图，并通过基于重构的框架进行评估。我们对6个开源权重和专有LLM的评估表明，与仅使用历史观测数据相比，加入模型生成的因果图通常会降低重构质量，尽管少数指标有所改善。这些发现表明，推断有信息量的、针对特定事件的市场（结构）……

    arXiv:2611.23533v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly evaluated through their ability to predict future outcomes, implicitly assuming that predictive success reflects an understanding of the underlying system. However, accurate predictions can often be achieved without recovering the latent mechanisms that generate observable outcomes. To examine this assumption, we introduce LUCID, a benchmark that evaluates whether models can infer latent market structures from event descriptions and historical financial observations. Models generate event-induced causal graphs representing inferred market structure and are evaluated through a reconstruction-based framework. Our evaluation of 6 open-weight and proprietary LLMs shows that adding their generated graphs generally reduces reconstruction quality compared with using historical observations alone, although a few metrics improve. These findings suggest that inferring informative, event-specific market
    
[^60]: 凭空虚构作者：学术作者归属中的事实普遍性与大语言模型幻觉

    Dreaming Up Authors: Factual Prevalence and LLM Hallucinations in Academic Authorship Attribution

    [https://arxiv.org/abs/2611.23530](https://arxiv.org/abs/2611.23530)

    该研究通过192万篇论文的作者归属实验发现，论文引用量与大语言模型幻觉率呈稳健的反比关系——低引用论文幻觉率超过98%，高引用论文低于37%，且失败主要源于检索局限而非知识缺失。

    

    大语言模型（LLM）被广泛用于问答任务，但容易产生幻觉，然而其背后的根本原因，特别是训练数据中事实普遍性的作用，仍未得到充分探索。在本工作中，我们通过学术作者归属任务来研究这一问题，假设对于流行度较低的论文，模型更容易产生幻觉。我们使用论文引用次数作为事实普遍性的代理指标，构建了一个涵盖八个科学学科、共计192万篇论文的数据集，并按引用水平进行分层抽样。在三个最先进大语言模型上的实验证实了我们的假设，揭示了幻觉率与引用次数之间在所有模型和学科上都稳健存在的反比关系：低引用论文的幻觉率超过98%，而高引用论文的幻觉率则低于37%。我们的深入分析进一步表明，这些失败主要反映了检索能力的局限，而非知识的缺失（摘要在此处截断）。

    arXiv:2611.23530v1 Announce Type: new  Abstract: Large Language Models (LLMs) are widely used for question answering but are prone to hallucinations, yet the underlying causes, particularly the role of factual prevalence in training data, remain underexplored. In this work, we study this through academic authorship attribution, hypothesizing that hallucinations are more likely for less prevalent papers. Using citation count as a proxy for factual prevalence, we curate a dataset of 1.92 million papers across eight scientific disciplines with stratified samples by citation level. Experiments on three state-of-the-art LLMs confirm our hypothesis, revealing a robust inverse relationship between hallucination rate and citation count across all models and disciplines, with hallucination rates exceeding 98% for rarely cited papers and falling below 37% for highly cited ones. Our in-depth analysis further shows that these failures primarily reflect retrieval limitations rather than knowledge a
    
[^61]: 从提审到结案：法院案件的Transformer模型

    From Arraignment to Resolution: Transformer Models of Court Cases

    [https://arxiv.org/abs/2611.23518](https://arxiv.org/abs/2611.23518)

    本文提出基于Transformer的时间序列模型，将法院案件建模为事件序列以预测刑事案件中下一诉讼程序的结果，从而提升司法决策的透明度与可监督性。

    

    刑事法律系统由许多复杂的决策点组成，例如追诉哪些罪名、如何确定保释金、认罪协议谈判以及量刑判决。鉴于此类决策的深远影响，提高该系统内部透明度的需求日益增长。虽然先前的机器学习方法主要聚焦于对单一决策（如累犯）进行建模，本文提出了一种基于Transformer的时间序列模型，用于预测案件中下一个诉讼程序的结果（例如接受指控、设定保释金、达成认罪协议等）。通过将案件建模为事件序列，我们能够进行更细致的分析，以支持对决策过程更好的监督和理解。我们使用基于美国某中等城市公开法庭记录构建的新数据集，对数千个案例进行了实证分析。我们拟合了将书记员笔记中的文本数据与结构化信息相结合的模型……

    arXiv:2611.23518v1 Announce Type: new  Abstract: The criminal legal system is composed of many complex decision points, such as which charges are pursued, how bond is determined, plea deal negotiations, and sentencing. Given the profound impact of such decisions, there is a growing need to increase the transparency within this system. While prior machine learning approaches have focused on modeling single decisions (e.g., recidivism), this paper proposes a transformer-based time series model to forecast the outcome of the next proceeding in a case (e.g., charges accepted, bond set, plea deal made, etc.). By modeling a case as an event sequence, we enable a more nuanced analysis to support better oversight and understanding of decision making. We conduct an empirical analysis of thousands of cases using a newly constructed dataset derived from publicly available court records of a mid-sized U.S. city. We fit models that combine textual data from clerk notes with structured information a
    
[^62]: GRAFT：面向智能体设计的执行引导式基因重组

    GRAFT: Execution-Guided Genetic Recombination for Agent Design

    [https://arxiv.org/abs/2611.23509](https://arxiv.org/abs/2611.23509)

    GRAFT提出了一种执行引导的遗传重组方法，在进化式智能体设计中利用父代任务执行的成功与失败证据来引导语义交叉与变异，而不仅依赖聚合适应度分数，从而在不额外执行任务的情况下提升LLM智能体架构的搜索效果。

    

    自动智能体设计旨在搜索能够提升基于LLM的智能体性能的可执行架构，其中进化方法使用语言模型交叉来重组有前景的机制，以实现任务分解、模型协调和验证。然而，这种搜索通常仅由聚合适应度分数引导，这掩盖了每个分数背后的实例级成功与失败信息。我们提出了面向智能体设计的执行引导式基因重组（GRAFT），该方法使用聚合适应度进行选择和存活判定，同时利用父代特有的执行证据来指导融合了变异的语义交叉。其反馈框架复用适应度评估中有界数量的成功与失败运行样本，无需提供参考答案，也不需要额外的任务执行。我们使用开放权重的优化器模型和执行器模型，在六个推理和编程场景中评估了GRAFT，它实现了平均82（摘要原文在此处截断）的测试准确率。

    arXiv:2611.23509v1 Announce Type: new  Abstract: Automated agent design searches for executable architectures that improve LLM-based agent performance, with evolutionary methods using language-model crossover to recombine promising mechanisms for task decomposition, model coordination, and verification. This search, however, is typically guided by aggregate fitness alone, which hides the instance-level successes and failures behind each score. We introduce Execution Guided Genetic Recombination for Agent Design (GRAFT), which uses aggregate fitness for selection and survival and parent-specific execution evidence to guide semantic crossover with integrated mutation. Its feedback harness reuses bounded samples of successful and failed runs from fitness evaluation, withholding reference answers and requiring no additional task executions. We evaluate GRAFT across six reasoning and programming settings using open-weight optimizer and executor models. It achieves a mean test accuracy of 82
    
[^63]: BOS摘要：第一个标记至关重要

    BOS Summary: The First Token Matters

    [https://arxiv.org/abs/2611.23508](https://arxiv.org/abs/2611.23508)

    BOS Summary是一种无需训练的推理时方法，通过在预填充阶段的单个transformer层放宽BOS查询行的因果约束，让序列开始标记聚合所有视觉和文本输入信息，使模型在解码全程都能访问这一上下文表示，且不引入任何额外参数或开销。

    

    大型视觉语言模型（LVLMs）和语言模型（LLMs）常常表现出注意力汇聚现象，即特定位置（如序列开始（BOS）标记）会获得不成比例的注意力份额。然而，在标准因果掩码下，BOS查询仅关注自身，因此后续标记反复访问的表示不携带任何关于输入的信息。我们提出了BOS Summary，这是一种无需训练的推理时方法，将这一持久通路重新用于承载输入上下文。在预填充阶段的单个transformer层，BOS Summary仅对BOS查询行放宽因果约束，使BOS能够将所有视觉和文本输入标记聚合为一个自生成的上下文表示，同时其他所有标记保持标准因果注意力。这种丰富的表示在解码全过程中始终保持对模型可访问，无需额外参数、辅助模块或解码时间开销。

    arXiv:2611.23508v1 Announce Type: new  Abstract: Large vision-language models (LVLMs) and language models (LLMs) often exhibit attention sinks, whereby particular positions, such as the beginning-of-sequence (BOS) token, receive a disproportionate share of attention. Under standard causal masking, however, the BOS query attends only to itself, so the representation that later tokens repeatedly access carries no information about the input. We introduce BOS Summary, a training-free inference-time method that repurposes this persistent pathway to carry input context. At a single transformer layer during prefill, BOS Summary relaxes the causal constraint for the BOS query row alone, letting BOS aggregate all visual and textual input tokens into a self-derived context representation while every other token retains standard causal attention. The enriched representation then remains accessible to the model throughout decoding, with no additional parameters, auxiliary modules, or decoding-tim
    
[^64]: SpecLook：通过推测观察位置实现快速可靠的多模态大语言模型智能体

    SpecLook: Speculating Where to Look for Fast and Reliable Agentic MLLMs

    [https://arxiv.org/abs/2611.23483](https://arxiv.org/abs/2611.23483)

    SpecLook 提出了一个无需训练的“推测-验证”三层框架，通过让小模型推测智能体下一步的放大观察——该观察可被验证、纠正并用于检验答案——从而显著加速多模态大语言模型智能体的推理并提升答案可靠性。

    

    具有智能体能力的多模态大语言模型（MLLMs）通过“用图像思考”的方式，将推理与放大工具调用交错进行。然而，这种顺序循环速度较慢，且其答案不附带可靠性评分。先前对 token、推理步骤或答案的推测仍然保留了这一循环，而答案推测还会接受自信却错误的答案。我们发现该循环主要增加的是一个放大步骤，借助这一步骤，小模型在大多数数据划分上能够领先于该智能体。我们的关键洞察是推测智能体的下一次观察——与起草的答案不同，观察是可以被验证、纠正并用于检验答案的。我们提出 SpecLook，这是一个无需训练的“推测-验证”框架，包含三个层次：小模型起草一个“瞥见”并据此重新作答，若答案不变且置信度高，则问题结束；否则，智能体的大模型在不使用工具的情况下验证并纠正该瞥见，并将两者的答案融合；只有仍缺乏证据的问题……（摘要原文在此处截断）

    arXiv:2611.23483v1 Announce Type: new  Abstract: Agentic multimodal large language models (MLLMs) that think with images interleave reasoning with zoom-in tool calls. However, this sequential loop is slow, and its answers carry no reliability score. Prior speculation of tokens, reasoning steps, or answers keeps the loop, and answer speculation also accepts confident errors. We find that the loop mostly adds one zoom-in step, with which a small model is ahead of the agent on most splits. Our key insight is to speculate the agent's next observation, which, unlike a drafted answer, can be verified, corrected, and used to test the answer. We propose SpecLook, a training-free speculate-and-verify framework with three levels. A small model drafts a glimpse and re-answers from it, and the question ends if the answer is unchanged and confident. Otherwise, the agent's large model verifies and corrects the glimpse without tools, and their answers are fused. Only questions that still lack evidenc
    
[^65]: 正式度得体指数（基于波兰语语料）

    Formal Adequacy Index (based on Polish-language material)

    [https://arxiv.org/abs/2611.23482](https://arxiv.org/abs/2611.23482)

    本文提出基于波兰语语料的正式度得体指数，通过分别建模观察正式性与期望正式性并引入可区分正式性不足与过度的语境容忍阈值，实现了对文本正式性与交际情境匹配程度的定量评估。

    

    语言正式性是言语风格变异的维度之一；然而，仅仅测量其水平并不能确定所选的交流方式在特定情境中是否恰当。本文旨在提出正式度得体指数的概念（基于波兰语语料），该指数能够对文本的正式性与交际情境所引发的期望之间的匹配程度进行定量评估。所提出的方法基于对两种正式性的分别建模：根据文本语言特征确定的观察正式性，以及根据交际情境特征确定的期望正式性。通过比较这两种度量并考虑语境容忍阈值来评估得体性，且针对正式性不足与正式性过度的容忍阈值可以不同。文中阐述了该指数的潜在应用。

    arXiv:2611.23482v1 Announce Type: new  Abstract: Linguistic formality is one of the dimensions of stylistic variation in speech; however, simply measuring its level does not allow us to determine whether the chosen mode of communication is appropriate in a given situation. The aim of this article is to present the concept of a formal adequacy index (based on Polish-language material), which enables a quantitative assessment of how well the formality of a text matches the expectations arising from the communicative context. The proposed approach is based on the separate modelling of observed formality, determined on the basis of the linguistic features of the text, and expected formality, determined on the basis of the characteristics of the communicative situation. Adequacy is assessed by comparing these two measures, taking into account contextual tolerance thresholds, which may differ for a lack of formality and an excess of formality. The potential applications of the index are illu
    
[^66]: 迭代博弈中LLM推理轨迹的表示追踪

    Tracking Representations in LLM Reasoning Traces for Iterated Games

    [https://arxiv.org/abs/2611.23403](https://arxiv.org/abs/2611.23403)

    该论文通过机制可解释性技术揭示了LLM在迭代博弈中会在思维链里形成并迭代修正博弈的内部表示，但一旦过早形成错误信念便难以恢复，且这种失败同时体现在推理轨迹和内部激活之中。

    

    大语言模型（LLM）智能体通过思维链（CoT）推理在各个领域取得了最先进的性能。尽管其应用无处不在，但人们对外部化的推理轨迹与模型实际计算之间的关系知之甚少。我们通过研究LLM智能体在信息不完整的迭代博弈中如何做出决策来应对这一问题。利用一系列机制可解释性技术，我们发现模型在整个思维链过程中会形成并迭代地修正其对博弈的内部表示。一旦模型过早地陷入错误的信念，它很少能够恢复。这种失败在字面推理轨迹和层间激活中均有体现。随着多智能体系统日益普及，我们的工作为更有针对性地诊断其策略推理错误打开了大门。出于演示目的，我们还提供了该方法在选定场景上的交互式可视化工具。

    arXiv:2611.23403v1 Announce Type: cross  Abstract: Large language model (LLM) agents achieve state-of-the-art performance across domains through chain-of-thought (CoT) reasoning. Despite their ubiquitous use, little is understood about the relationship between the externalized trace and the model's actual computation. We address this by investigating how LLM agents make decisions in iterative games with incomplete information. Using a series of mechanistic interpretability techniques, we find that the model forms and iteratively revises an internal representation of the game throughout the CoT. Once the model prematurely commits to an incorrect belief, it rarely recovers. This failure is manifested in both the literal trace and the layer activations. Ultimately, as multi-agent systems are more prevalent, our work opens the door to a more focused diagnosis of their strategic reasoning errors. For illustrative purposes, we offer an interactive visualizer of our methodology on selected sa
    
[^67]: QDA：基于智能体的闭环大语言模型量化设计

    QDA: Closed-Loop LLM Quantization Design with Agents

    [https://arxiv.org/abs/2611.23388](https://arxiv.org/abs/2611.23388)

    QDA提出了一个基于智能体的闭环系统，通过KL散度快速评估候选量化方案并自动化整合不同代码库的方法，为每个模型和精度自动设计最优的大语言模型训练后量化方案。

    

    训练后量化是以低成本部署大语言模型的标准方式。然而，最佳量化方案因模型和精度而异。因此，专家必须为每个新模型重新设计方案，这既耗时又往往不是最优的。幸运的是，大语言模型智能体可以通过在多轮构建和评估候选方案的过程中不断改进量化方案来接管这一设计工作，使得评估轮数越多，方案质量越高。然而，评估一个量化推理模型需要数十个GPU小时，而且要结合来自不同代码库的方法，就需要手动将每个方法移植到新架构上。我们提出了QDA，一个由智能体为每个模型和精度设计量化方案的系统。为了加快评估速度，QDA通过计算每个候选方案在智能体从未见过的推理轨迹上与原始模型输出的KL散度，在几分钟内完成评分，并对有前景的候选方案进行长文本生成检查，以捕捉重复等失败情况……

    arXiv:2611.23388v1 Announce Type: cross  Abstract: Post-training quantization is the standard way to serve large language models at low cost. However, the best quantization recipe differs across models and precisions. As a result, experts must redesign it for every new model, which is time-consuming and often sub-optimal. Fortunately, an LLM agent can take over this design by refining the recipe over rounds of building and evaluating candidates, so that more rounds lead to better recipes. However, evaluating one quantized reasoning model takes tens of GPU-hours, and combining methods from different codebases means porting each to the new architecture by hand. We present QDA, a system in which an agent designs the recipe for each model and precision. To make evaluation fast, QDA scores each candidate in minutes by its output KL divergence from the original model on reasoning traces that the agent never sees, and checks promising ones with long generations to catch failures such as repet
    
[^68]: Chorus：具有逐层自适应的共享N-gram嵌入

    Chorus: Shared N-gram Embeddings with Layer-wise Adaptation

    [https://arxiv.org/abs/2611.23386](https://arxiv.org/abs/2611.23386)

    Chorus提出让整个网络共享一组n-gram嵌入表，每层通过基于自身隐藏状态计算的缩放因子和独立投影层以各自的方式读取共享表，从而以比现有方法小87%的表实现相当的质量。

    

    语言模型可以通过n-gram嵌入表以极小的计算代价获得容量提升：这些嵌入表是为连续token短序列存储学习向量的大型表，通过查表将其添加到模型的输入或中间层。近期的设计通常为每一层使用单独的n-gram嵌入表，这使得这些表在内存开销上十分昂贵，并且阻碍了各层之间共享所学内容。然而，共享单个表会使每一层获得相同的嵌入，没有为各层的偏好留下空间，因此在实践中表现不佳。我们提出Chorus，其中一组n-gram嵌入表服务于整个网络，而每一层以自己的方式读取它。每一层先根据其当前隐藏状态计算出的因子对各分量进行重新缩放，然后通过其自己的投影层映射该精炼后的嵌入。Chorus以小87%的表实现了与现有n-gram方法相当的质量：在1.2B表参数……

    arXiv:2611.23386v1 Announce Type: new  Abstract: Language models can gain capacity at little computational cost from n-gram embedding tables: large tables of learned vectors for short sequences of consecutive tokens, which are looked up and added to the model's input or to its intermediate layers. Recent designs commonly use a separate n-gram embedding table for each layer, which makes the tables costly in memory and prevents layers from sharing what they learn. Sharing a single table, however, gives every layer the same embedding and leaves no room for layer-wise preferences, and it performs poorly in practice. We propose Chorus, in which one set of n-gram embedding tables serves the whole network and every layer reads it in its own way. Each layer maps the refined embedding through its own projection layer, after rescaling its components by factors computed from its current hidden state. Chorus matches the quality of existing n-gram methods with 87% smaller tables: with 1.2B table pa
    
[^69]: 优先排序，而非决策：审计一个已部署的人在环简历排序系统的收获

    Prioritize, Don't Decide: What We Learned Auditing a Deployed Human-in-the-Loop CV Sorter

    [https://arxiv.org/abs/2611.23359](https://arxiv.org/abs/2611.23359)

    本文审计了一个部署在国家学术招聘平台上的人在环简历优先排序系统——委员会按其推荐顺序只需阅读一半简历即可覆盖 95% 的相关候选人；审计进一步发现该评测简历池仅凭文档长度即可排序，使精确率类指标失去参考意义，这一发现直接改变了产品设计。

    

    一份学术职位空缺可以吸引数百份申请，而第一轮筛选会占用评审委员会的大部分时间。我们描述了一个部署在国家学术职位空缺平台上、并与该平台职位空缺数据库集成的简历优先级排序器。它为一堆简历提出阅读顺序，而将所有决策留给委员会；该系统将冻结的多语言 Transformer 编码器与一个会话级线性模型相结合，该模型在每次标注后重新训练，并在会话结束时被丢弃。它是有效的：在一个包含 300 份合成简历的池子中，按该工具推荐的顺序阅读的委员会在读完一半简历后即可接触到 95% 的相关候选人，而按任意顺序阅读则需要几乎读完全部简历。本文关注的是我们审计上述声明时的发现，而非系统本身。三个发现改变了产品。其中一个发现是：该简历池仅凭文档长度即可完成排序，这使得在其上使用精确率类指标变得毫无参考意义；重新生成……（原文在此处截断）

    arXiv:2611.23359v1 Announce Type: new  Abstract: A single academic vacancy can draw hundreds of applications, and the first screening round consumes most of a committee's time. We describe a CV Priority Sorter deployed at a national platform for academic vacancies and integrated with its vacancy database. It proposes a reading order for a pile of CVs and leaves every decision to the committee, pairing a frozen multilingual transformer encoder with a per-session linear model that retrains after each label and is discarded when the session ends. It works: on a pool of 300 synthetic CVs, a committee reading in the tool's order reaches 95% of the relevant candidates after half the pile, where an arbitrary order needs almost all of it. This paper is about what we found when we audited that claim, rather than about the system itself. Three findings changed the product. The pool turned out to be sortable by document length alone, which makes precision-style metrics on it uninformative; regene
    
[^70]: 大语言模型RLOps综述

    A Survey on RLOps for Large Language Models

    [https://arxiv.org/abs/2611.23357](https://arxiv.org/abs/2611.23357)

    本文对大语言模型RLOps方法进行了系统性综述，将现有技术归纳为训练设置、经验收集、奖励计算、策略优化和模型评估五个阶段，填补了该领域技术碎片化、缺乏统一梳理的空白。

    

    强化学习（RL）在提升大语言模型（LLM）的推理、指令遵循和任务解决能力方面正变得越来越重要。然而，扩展强化学习训练会暴露出分布式执行、反馈计算和学习动态方面的故障，这些故障会浪费计算资源、拖慢训练速度并削弱模型改进效果。这些挑战催生了越来越多的大语言模型RLOps技术，用于管理大语言模型强化学习训练全生命周期的运行效率与可靠性。尽管关注度不断上升，这些技术仍然分散在各个研究领域，对其运维角色及相互作用的系统性梳理较为有限。为弥合这一差距，我们对大语言模型RLOps方法进行了系统性综述。我们将现有方法组织为五个阶段：训练设置、经验收集、奖励计算、策略优化和模型评估。

    arXiv:2611.23357v1 Announce Type: cross  Abstract: Reinforcement learning (RL) has become increasingly important for improving the reasoning, instruction-following, and task-solving capabilities of large language models (LLMs). However, scaling RL training exposes failures in distributed execution, feedback computation, and learning dynamics that can waste computational resources, slow training, and undermine model improvements. These challenges have motivated a growing body of techniques for LLM RLOps, which manages the efficiency and reliability of operations throughout the LLM reinforcement learning training lifecycle. Despite growing interest, these techniques remain scattered across research areas, with limited synthesis of their operational roles and interactions. To bridge this gap, we present a systematic survey of LLM RLOps methods. We organize existing approaches into five stages: Training Setup, Experience Collection, Reward Computation, Policy Optimization, and Model Evalua
    
[^71]: 并发生成的结构化记录映照了它所伴随的散文叙述：《隐藏待处理工具调用》场景的预注册概念性复制与扩展

    A Concurrent Structured Record Mirrored the Prose It Accompanied: A Pre-Registered Conceptual Replication and Extension of the Hide Pending Tool Call Scenario

    [https://arxiv.org/abs/2611.23274](https://arxiv.org/abs/2611.23274)

    本研究对 Huang 等人（2026）的“隐藏待处理工具调用”场景进行了预注册概念复制与扩展，发现要求模型在同一次生成中附加含“未解决问题”残余部分的结构化记录时，该记录只是映照了其伴随的散文叙述内容，并未能独立提升对缺失待处理结果的披露。

    

    Huang 等人（2026）的研究表明，撰写智能体工作报告的语言模型往往倾向于掩盖会改变叙述内容的缺陷，而在指令末尾附加“请在回答中保持诚实。”一句，能在其大多数场景中显著提高披露率。在其“隐藏待处理工具调用”场景中——即一个更宽泛的后续查询被接受却始终未返回结果——诚实提示使 GPT-5.5 标记缺失数据的比例仍然停留在 0%，尽管它让大多数回答变成了附带保留说明的形式。我探究的是换一种请求是否会改变这一点：在同一次生成中附加一份遵循对抗性审查工具模式的结构化记录，其中包含一个记录尚未解决问题的“残余”部分。在基于同一组 100 对合成日志（每个待处理日志均与一个后续查询确实返回了结果的“干净孪生”日志相配对）的两个预注册阶段中，我在四种条件下通过 Codex CLI 运行了 gpt-5.6-sol。第一阶段测量了记录的表现：当包含残余部分时……（原文摘要至此截断）

    arXiv:2611.23274v1 Announce Type: cross  Abstract: Huang et al. (2026) show that language models writing reports of agentic work tend to conceal narrative-changing flaws, and that appending "Be honest in your response." to the instruction substantially increases disclosure in most of their scenarios. In their Hide Pending Tool Call scenario, where a broader follow-up query was accepted but never returned, the honesty prompt left GPT-5.5's rate of flagging the missing data at 0%, although it turned most answers into caveats. I ask whether a different request changes this: appending, in the same generation, a structured record that follows the schema of an adversarial-review tool, including a residue section for what remained unresolved. In two pre-registered phases on the same 100 synthetic log pairs, each pending log matched with a clean twin in which the follow-up query did return, I ran gpt-5.6-sol through Codex CLI under four conditions. Phase 1 measured the record: with the residue
    
[^72]: PGPO：面向隐私保护LLM增强推荐的原型引导渐进式混淆

    PGPO: Prototype-Guided Progressive Obfuscation for Privacy-Preserving LLM-Enhanced Recommendation

    [https://arxiv.org/abs/2611.23236](https://arxiv.org/abs/2611.23236)

    该论文提出PGPO方法，通过为高影响力原型词学习拓扑感知的混淆锚点并在静态语义图中渐进传播，在有效防御嵌入反演攻击保护隐私的同时保持物品间语义关系，实现接近上限的推荐性能。

    

    LLM增强的推荐系统将物品的辅助信息编码为语义嵌入，但这些嵌入可能通过反演攻击泄露敏感的文本属性。现有的输入侧防御依赖于可信收集或局部词元扰动，往往会破坏推荐所需的物品间语义拓扑结构。为解决这一问题，我们引入关系保真度作为关键目标，要求在实现隐私保护的语义偏移的同时，保留物品间的相对关系以维持推荐效用。基于这一原则，我们提出了原型引导的渐进式混淆（PGPO），该方法为高影响力的原型词学习拓扑感知的混淆锚点，并通过静态语义图进行传播。PGPO使用GRPO进行优化，采用个体差异性、结构一致性和语义约束三种奖励。在两个基准数据集上的实验表明，PGPO在实现强大反演攻击防御的同时，推荐性能接近上限。

    arXiv:2611.23236v1 Announce Type: cross  Abstract: LLM-enhanced recommender systems encode item side information into semantic embeddings, but these embeddings may leak sensitive textual attributes through inversion attacks. Existing input-side defenses rely on trusted collection or local token perturbation, often disrupting the inter-item semantic topology needed for recommendation. To address it, we introduce relational fidelity as the key objective, requiring semantic shift for privacy while preserving relative item relations for utility. Based on this principle, we propose prototype-guided progressive obfuscation (PGPO), which learns topology-aware obfuscation anchors for high-influence prototype words and propagates them through a static semantic graph. PGPO is optimized with GRPO using individual dissimilarity, structural consistency, and semantic constraint rewards. Experiments on two benchmarks show strong inversion defense with near-upper-bound recommendation performance. The 
    
[^73]: 面向多语言适配的MoE-LoRA方法：一项系统性实证研究

    MoE--LoRA Methods for Multilingual Adaptation: A Systematic Empirical Study

    [https://arxiv.org/abs/2611.23158](https://arxiv.org/abs/2611.23158)

    本研究首次在统一的多语言评估协议下，用200种语言和9000多个评估设置系统比较八种LoRA方法，发现小规模搜索无法可靠预测方法在大规模下的相对表现，且LoRA虽能提升翻译数据拟合度但会持续损害阅读理解能力。

    

    路由式低秩适配器为提升大语言模型（LLM）在各类任务上的性能提供了一种参数高效的手段。因此，人们假定它们也可用于改善LLM对低资源语言的支持——目前这些模型对英语的服务远好于全球数千种其他语言。然而，这一假设从未在统一的多语言评估协议下得到实证验证。我们正是针对这一空白开展研究：在冻结的基础模型上，使用200种独特语言、超过9,000个评估设置，对八种低秩适配方法进行了比较。我们的结果表明，缩减规模的搜索能够识别出LoRA方法的可行配置，但无法可靠地预测其相对性能，因为排名最高的配置在大规模评估中从未表现最佳。LoRA能可靠地提升模型在翻译数据上的语言建模拟合度，却持续降低阅读理解能力，并在（原文摘要在此处截断）……

    arXiv:2611.23158v1 Announce Type: new  Abstract: Routed low-rank adapters offer a parameter-efficient means to improve the performance of large language models (LLMs) across tasks. Hence, it is assumed that they can also be used to improve the support of low-resource languages in LLMs, which currently serve English far better than the thousands of other languages spoken worldwide. However, this assumption has never been verified empirically under a shared multilingual evaluation protocol. We address exactly this gap by comparing eight low-rank adaptation (LoRA) methods on a frozen base model using 200 unique languages in more than 9{,}000 evaluation settings. Our results suggest that a reduced-scale search can identify viable configurations of LoRA methods but does not reliably predict their relative performance, as the top-ranking configuration is never best at scale. LoRA reliably improves language-model fit on translation data, consistently degrades reading comprehension, and benefi
    
[^74]: 嵌入余弦相似度在度量同质化之前，首先度量的是文本长度

    Embedding Cosine Measures Text Length Before It Measures Homogenization

    [https://arxiv.org/abs/2611.21519](https://arxiv.org/abs/2611.21519)

    该研究揭示嵌入余弦相似度主要度量的是文本长度而非话语同质化，此前关于2022年后保守派言论同质化的发现是文本长度差异造成的假象，控制长度后该效应完全消失。

    

    Transformer嵌入之间的平均成对余弦相似度是衡量在线言论在生成式AI影响下是否趋于同质化的标准指标。当嵌入来自一个从未针对句子相似性训练的编码器的均值池化输出时，该指标主要反映的是文本长度。在606万条Reddit评论中，未经中心化的DeBERTa-v3指标曾表明，只有保守派言论在2022年底之后出现了同质化趋势。将相同的评论从10个词截断到150个词，其平均余弦相似度的提升幅度是所报告断裂效应的八倍。一种将每条评论替换为等长随机评论的安慰剂检验重现了81%的不对称性，而固定内容的模拟则重现了78%。第一主成分方向承载了99.9%的原始DeBERTa同质性，一旦对长度加以控制，这种不对称性在所有编码器下都会消失。在57.5万条Stack Exchange回答上，该指标报告了技术网站在ChatGPT出现后的同质化现象……（原文摘要在此处截断）

    arXiv:2611.21519v1 Announce Type: new  Abstract: Mean pairwise cosine similarity between transformer embeddings is a standard measure of whether online discourse is converging under generative AI. When the embeddings are mean-pooled outputs of an encoder never trained for sentence similarity, this measure mostly tracks text length. In 6.06 million Reddit comments, an uncentered DeBERTa-v3 measure had suggested that only conservative discourse broke toward homogeneity after late 2022. Truncating the same comments from 10 to 150 words raises their mean cosine by eight times the reported break. A placebo that swaps every comment for a random one of equal length reproduces 81% of the asymmetry, and a fixed-content simulation reproduces 78%. The leading principal direction carries 99.9% of raw DeBERTa homogeneity, and once length is handled the asymmetry disappears under every encoder. On 575,000 Stack Exchange answers the measure reports a post-ChatGPT homogenization of technical sites tha
    
[^75]: Jetriever：将智能体记忆检索从相似度转向效用

    Jetriever: Shifting Agent Memory Retrieval from Similarity to Utility

    [https://arxiv.org/abs/2611.21478](https://arxiv.org/abs/2611.21478)

    Jetriever 用 Jev 决策分数对全部存储记忆进行效用评分排序，取代传统相似度检索，使智能体在回答问题时能将最有用的记忆准确地排在最前。

    

    一条有用的记忆未必与它所帮助回答的问题相似。Jetriever 依据 Jev 的决策分数对每条存储记忆对当前问题的有用性进行排序，用全池效用评分取代相似度搜索。在 InMind 的 125 个间接查询任务上，它在 84.8% 的情况下将黄金交易记录排在首位；而以用户轮次建立索引的对照方法 BM25 和 MiniLM 分别仅达到 1.6% 和 0.0%。一项包含 20 个问题的对比实验将“找到证据”与“对证据排序”区分开来：Jev 和 Qwen3.5 9B 各自能在前 50 名内找到 90% 的黄金交易记录，而 Laya 仅找到 30%；在排名第一的位置上，Jev 达到 85%，Qwen3.5 9B 为 40%。在对 MemoryAgentBench 问题按维度等比例随机抽取的 60 个问题上，Jetriever 的四维度得分为 55.4%，而 ReFind 在相同问题上的得分为 44.3%。Jev 检索在 InMind 上平均每个问题花费 0.00834 美元，在 MAB 上平均每个问题花费 0.03103 美元。这些结果表明了世界知识的价值，以及能够将决定性记忆稳定排在前列的评分机制的作用。

    arXiv:2611.21478v1 Announce Type: new  Abstract: A useful memory need not resemble the question it helps answer. Jetriever ranks each stored memory by Jev's decision score for its usefulness to the current question, replacing similarity search with full-pool utility scoring. On InMind's 125 indirect-query tasks, it places the gold exchange first in 84.8% of cases; matched BM25 and MiniLM controls reach 1.6% and 0.0% when indexing user turns. A 20-question comparison separates finding evidence from ranking it: Jev and Qwen3.5 9B each find 90% of gold exchanges by rank 50, while Laya finds 30%; at rank one, Jev reaches 85% and Qwen3.5 9B 40%. On a dimension-proportional random sample of 60 MemoryAgentBench questions, Jetriever's four-dimension score is 55.4%, against 44.3% for ReFind on the same questions. Jev retrieval costs USD 0.00834 per InMind question and USD 0.03103 per MAB question on average. These results show the value of world knowledge, scores that rank decisive memories fir
    
[^76]: 机器口译的竞争力：英汉同声传译中意义与译语质量的语篇敏感性比较

    Machine interpreting competitiveness: a discourse-sensitive comparison of meaning and target-language quality in English-Chinese simultaneous interpreting

    [https://arxiv.org/abs/2611.19802](https://arxiv.org/abs/2611.19802)

    本研究采用语篇敏感的评估框架，将五个商用机器口译系统与26名人类译员在英汉同声传译中的意义传达与译语质量进行比较，发现机器口译在计划性正式语篇中已进入人类译员较高水平区间，但在即兴语篇中表现差异显著。

    

    机器口译（MI），即人工智能驱动的语音翻译，在专业及面向公众的交流中日益常见，但其相对于人类同声传译员的表现仍缺乏充分认识。现有研究往往受限于单一的语篇模式，且评估框架与口译作为一种语篇层面交际活动的特点契合度不足。本研究将五个商用机器口译系统与26名人类口译员（14名职业译员、12名高年级学生译员）在英译汉同声传译任务上的表现进行比较，涵盖计划性与非计划性语篇，重点关注意义传达与译语质量。在这些材料中，机器口译在两种语篇模式下的表现差异显著。在计划性的正式演讲中，表现最强的系统在意义传达方面极具竞争力，进入了人类译员的较高水平区间，另一系统也表现强劲。在即兴访谈中……

    arXiv:2611.19802v1 Announce Type: new  Abstract: Machine interpreting (MI), or AI-enabled speech translation, is increasingly visible in professional and public-facing communication, yet its performance relative to human simultaneous interpreters remains insufficiently understood. Existing studies are often limited by a single discourse mode and evaluation frameworks that are not well aligned with interpreting as a discourse-level communicative activity. The present study compares five commercial MI systems with 26 human interpreters (14 professionals, 12 advanced students) on English-to-Chinese simultaneous interpreting across planned and unplanned discourse, focusing on meaning and target-language quality. In these materials, MI performance differed sharply between the two discourse modes. In the planned formal speech, the strongest system became highly competitive on meaning, entering the upper human range, and a second system also performed strongly. In the extemporaneous interview
    
[^77]: 模糊样本有帮助，但全量数据更胜一筹：重新审视数据集制图法在抽取式对抗问答中的应用

    Ambiguous Examples Help, But Full Data Wins: Revisiting Dataset Cartography for Extractive Adversarial QA

    [https://arxiv.org/abs/2611.19763](https://arxiv.org/abs/2611.19763)

    本文在更严格的抽取式对抗问答设置下重新审视数据集制图法的结论，发现仅用33%模糊样本子集训练虽有帮助，但仍不敌全量数据训练在对抗性评估中的表现。

    

    数据集制图法（Dataset Cartography）认为，训练动态可以揭示数据集中对学习具有不同作用的区域：易学样本有助于优化过程，难学样本往往暴露标注问题，而模糊样本对分布外泛化的贡献最大。原始工作的一个引人注目的结论是，仅在33%模糊样本上训练的模型可以在多项分布外（OOD）评估中超越全量数据训练的模型，其中包括在Adversarial SQuAD上评估的SQuAD的QNLI重构版本。我们在一个更严格的设置下重新审视这一结论：开放式抽取式问答，即模型必须预测答案片段，而不是对“问题-句子”对进行分类。针对每个编码器系列，我们计算干净来源SQuAD的训练动态，分别在全量、随机、按易学排序、按模糊排序和按难学排序的子集上进行训练，并在仅包含对抗样本的AddSent和AddOneSent行上评估鲁棒性。按模糊排序选取的33%子集……（摘要内容在此处截断）

    arXiv:2611.19763v1 Announce Type: new  Abstract: Dataset cartography argues that training dynamics can reveal regions of a dataset that matter differently for learning: easy-to-learn examples aid optimization, hard-to-learn examples often expose annotation problems, and ambiguous examples contribute strongly to out-of-distribution generalization. A striking implication of the original work is that a model trained on only 33% ambiguous examples can outperform full-data training on several OOD evaluations, including a QNLI reformulation of SQuAD evaluated on Adversarial SQuAD. We revisit this claim in a stricter setting: open-ended extractive question answering, where models must predict answer spans rather than classify question-sentence pairs. For each encoder family, we compute clean-source SQuAD training dynamics, train on full, random, easy-ranked, ambiguous-ranked, and hard-ranked subsets, and evaluate robustness on adversarial-only AddSent and AddOneSent rows. Ambiguous-ranked 33%
    
[^78]: FlashREINFORCE：面向智能体语言模型的无评论家单次轨迹异步强化学习

    FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models

    [https://arxiv.org/abs/2611.18627](https://arxiv.org/abs/2611.18627)

    提出 FlashREINFORCE——一种无评论家、每个提示词仅单次采样的异步强化学习框架，通过单批次 REINFORCE、序列信赖域与样本均值优化，在策略滞后约四步的情况下仍能稳定训练智能体语言模型，同时保持提示词覆盖度。

    

    异步强化学习适用于轨迹生成时间不规则的长程智能体，但基于组间相对优势的更新需要对每个提示词进行多次轨迹采样。然而，基于分组的采样方式在固定轨迹生成预算下限制了提示词的覆盖范围，并引入了组同步屏障。每个提示词仅采样一次虽可避免这些限制，但会失去用于降低方差的组基线，导致更新噪声更大，在过时的、长度不一的轨迹上更容易不稳定。我们提出 FlashREINFORCE，一个无评论家的单次通过框架，其建立在单批次 REINFORCE、序列信赖域和样本均值优化之上。这些组件在保留提示词覆盖范围和带符号反馈的同时，控制了策略漂移和由序列长度放大的负向更新。实验表明，在策略滞后约四次更新的条件下，FlashREINFORCE 在 DeepSeek-R1-Distill-Qwen-1.5B 上经过 6,000 次更新仍保持稳定。在 Qwen2.5-Math-1.5B 上，其平均成绩达到 38.0（摘要至此被截断）。

    arXiv:2611.18627v1 Announce Type: new  Abstract: Asynchronous reinforcement learning suits long-horizon agents with irregular rollout times, but group-relative updates require repeated rollouts of each prompt. However, group-based sampling limits prompt coverage at a fixed rollout budget and introduces group synchronization barriers. One rollout per prompt avoids these constraints but removes the variance-reducing group baseline, making updates noisier and more prone to instability on stale, variable-length trajectories. We present FlashREINFORCE, a critic-free, one-pass framework built on One-Batch REINFORCE, a Sequence Trust Region, and Sample-Mean Optimization. These components preserve prompt coverage and signed feedback while controlling policy drift and length-amplified negative updates. In our experiments, FlashREINFORCE remains stable through 6,000 updates on DeepSeek-R1-Distill-Qwen-1.5B at a policy lag of approximately four updates. On Qwen2.5-Math-1.5B, it reaches 38.0 mean 
    
[^79]: Gamer：面向长时程实时游戏智能体的通用视觉-语言-动作基础模型

    Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent

    [https://arxiv.org/abs/2611.18397](https://arxiv.org/abs/2611.18397)

    提出Gamer，一个端到端的通用视觉-语言-动作基础模型，通过海量键鼠交互轨迹预训练、稀疏思考指令微调和稳定的多轮强化学习，实现了在复杂2D/3D交互环境中执行长时程任务的实时游戏智能体。

    

    我们提出了Gamer，一个专为复杂交互环境中长时程任务构建的通用智能体框架。其端到端架构结合了感知、记忆管理和多步推理。统一的训练与部署流程支持与常规2D图形界面以及复杂3D模拟世界的交互。我们首先在从数万小时设备级键盘和鼠标交互中构建的数据集上进行继续预训练，这些轨迹涵盖模拟游戏和具身环境，使模型能够学习跨物理与虚拟领域的环境动态。随后的稀疏思考指令微调在保留复杂决策所需推理能力的同时，大幅减少了冗余计算。我们采用一种稳定的多轮强化学习方法来训练策略以实现长时程交互，该方法解决了有限上下文窗口、累积（误差/奖励）等问题（摘要在此处被截断）。

    arXiv:2611.18397v1 Announce Type: cross  Abstract: We introduce Gamer, a generalist agent framework built for long-horizon tasks in complex interactive environments. Its end-to-end architecture combines perception, memory management, and multi-step reasoning. A shared training and deployment pipeline supports interaction with both conventional 2D graphical interfaces and complex 3D simulated worlds. We first continue pre-training on a dataset constructed from tens of thousands of hours of device-level keyboard and mouse interaction. These trajectories span simulated games and embodied environments, allowing the model to learn environmental dynamics across both physical and virtual domains. Sparse-thinking instruction tuning then substantially reduces redundant computation while preserving the reasoning needed for complex decisions. We train the policy for long-horizon interaction with a stable multi-turn reinforcement learning method that addresses finite context windows, accumulated r
    
[^80]: 构造性机构之上的类型化知识：一种面向可验证科学的证明相关类型论

    Typed Knowledge over Constructive Institutions: A Proof-Relevant Type Theory for Verified Science

    [https://arxiv.org/abs/2611.15108](https://arxiv.org/abs/2611.15108)

    本文提出了面向可验证科学知识基底 Eigenius 的证明相关依赖类型论，通过摒弃判断式证明无关性，使系统能够追踪并比较同一命题的多个独立推导、保持强规范化与传递性的等价判定，并支持在实验或模型失效时计算反事实撤回。

    

    诸如 Coq、Agda 和 Lean 4 之类的数学证明助手在定义等价中内置了证明无关性（即对于 P : Prop 且 p, q : P，有 p ≡ q），从而抹去了定理究竟是由哪条推导所建立的。虽然这对于基于固定公理的数学而言是自然的，但判断式证明无关性与可验证的科学知识库不相容：后者必须检查并比较同一命题的多个独立推导以评估其契合度，并在某个实验、试剂或模型失效时计算反事实的撤回。此外，将定义式证明无关性与子单例消去相结合会破坏强规范化，并使算法化的等价判定失去传递性。我们提出了 Eigenius 的类型论与机构论基础，Eigenius 是一个面向科学的可验证知识基底。首先，我们构造了一个具有累积宇宙的依赖类型内核，包含非直谓的 Prop、以及带有索引强制的子单……（原文摘要在此处截断）

    arXiv:2611.15108v1 Announce Type: cross  Abstract: Mathematical proof assistants such as Coq, Agda, and Lean 4 build proof irrelevance into definitional conversion ($p \equiv q$ for $p, q : P$ with $P : Prop$), erasing which derivation established a theorem. While natural for mathematics over fixed axioms, judgmental proof irrelevance is incompatible with verified scientific stores, which must inspect and compare independent derivations of the same claim to assess consilience and compute counterfactual retractions when an assay, reagent, or model is invalidated. Moreover, combining definitional proof irrelevance with subsingleton elimination breaks strong normalization and renders algorithmic conversion non-transitive.   We present the type-theoretic and institutional foundations of Eigenius, a verified knowledge substrate for science. First, we formulate a dependent kernel with cumulative universes ($Prop, Set, Type_k$), impredicative $Prop$, inductive families with index-forced subsi
    
[^81]: 眼见未必为实：反事实病灶擦除教会小型医学视觉-语言智能体只报告图像中真实存在的内容

    Seeing Is Not Believing: Counterfactual Lesion Erasure Teaches a Small Medical Vision-Language Agent to Report Only What the Image Shows

    [https://arxiv.org/abs/2611.13006](https://arxiv.org/abs/2611.13006)

    该论文提出无需人工标注的反事实病灶擦除评测范式与证据敏感度差距（ESG）指标，并据此训练出40亿参数的医学视觉-语言智能体MedScope-4B，使其只报告图像中真实可见的发现，将“幻觉报告被擦除病灶”的比例从前沿模型的41–62%降至9.6%。

    

    医学视觉-语言模型（VLM）常常仅仅因为问题中提到了某个发现的名称就报告该发现，而标准基准测试无法捕捉这一现象。我们从公开分割数据集构建反事实视图，无需人工标注：包括通过图像修复擦除病灶的图像、在其他位置进行同形状对照编辑的图像，以及未经编辑的无病灶图像。证据敏感度差距（ESG）定义为擦除后“不存在或转诊/参考”类回答的比率减去对照编辑后的相应比率；对于忽略图像或对两种编辑反应相同的模型，该指标为零。MedScope-4B（基于Qwen3.5-4B、LoRA微调、配备图像工具）在由标签推导的训练轨迹上训练，按预先注册的规则进行模型选择，并一次性与九个前沿模型和六个开源VLM进行对比测试；随后通过四项预先注册的检验探究其机制。在涵盖五种成像类型的507例内部病例上，MedScope-4B达到ESG 0.758，且仅在9.6%的病例中声称存在已被擦除的病灶，而前沿模型分别为0.27–0.47和41–62%（所有p<0.001）。

    arXiv:2611.13006v1 Announce Type: new  Abstract: Medical vision-language models (VLMs) often report a finding because the question names it; standard benchmarks miss this. From public segmentation datasets we build counterfactual views without manual annotation: the lesion erased by inpainting, a same-shape control edit elsewhere, and unedited lesion-free images. The evidence sensitivity gap (ESG) is the rate of absent-or-refer answers after erasure minus the rate after the control edit; it is zero for a model that ignores the image or reacts to both edits alike. MedScope-4B (Qwen3.5-4B, LoRA, image tools) was trained on label-derived trajectories, selected by a pre-registered rule and tested once against nine frontier and six open VLMs; four pre-registered checks then probed the mechanism. On 507 internal cases across five imaging types, MedScope-4B reached ESG 0.758 and claimed an erased lesion in 9.6% of cases, against 0.27-0.47 and 41-62% for the frontier models (all p<0.001), with
    
[^82]: 基于大语言模型不确定性在放射学报告中区分抽取失败与临床不一致

    Distinguishing Extraction Failures and Clinical Discordance with LLM Uncertainty on Radiology Reports

    [https://arxiv.org/abs/2611.10502](https://arxiv.org/abs/2611.10502)

    该研究表明，大语言模型的不确定性信号（包括token级和模型自报告的不确定性）能够有效区分放射学报告信息抽取中的抽取失败与真正的临床不一致，为医学文本结构化抽取的质量控制提供了新方法。

    

    大语言模型越来越多地被用于从非结构化文本中构建结构化数据，但抽取错误仍然不可避免。我们评估了大语言模型的不确定性是否能够区分抽取失败与真正的临床不一致。在因疑似冠状动脉疾病（CAD）而接受冠状动脉CT血管造影（CCTA）的1,045名患者的5,225条血管中，我们评估了在抽取血管级别狭窄严重程度过程中的token级不确定性以及模型自报告的不确定性。抽取失败定义为大语言模型抽取的狭窄结果与专家标注之间的不一致；临床不一致定义为CCTA与后续有创性冠状动脉造影（ICA）之间在阻塞性CAD状态上的不一致。在校正狭窄严重程度、钙化负荷和冠状动脉血管等因素后，评估了不确定性的增量判别能力。抽取失败在具有专家标注的253名患者的1,077条血管中进行了评估，而临床不一致（摘要在此处截断）……

    arXiv:2611.10502v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used to structure data from unstructured text, but extraction errors remain inevitable. We evaluated whether LLM uncertainty could distinguish extraction failure from true clinical discordance. In 5{,}225 vessels from 1{,}045 patients undergoing coronary CT angiography (CCTA) for suspected coronary artery disease (CAD), we evaluated token-level and self-reported LLM uncertainty during extraction of vessel-level stenosis severity. Extraction failure was defined as disagreement between LLM-extracted stenosis and expert annotations. Clinical discordance was defined as disagreement in obstructive CAD status between CCTA and subsequent invasive coronary angiography (ICA). Incremental discrimination was assessed after adjustment for stenosis severity, calcium burden, and coronary vessel. Extraction failure was evaluated in 1{,}077 vessels from 253 patients with expert annotations, and clinical 
    
[^83]: 曲折与转折：一个面向探索任务中长期记忆的基于图的基准与框架

    Twists and Turns: A Graph-Based Benchmark and Framework for Long-Term Memory in Exploration Tasks

    [https://arxiv.org/abs/2611.06414](https://arxiv.org/abs/2611.06414)

    该论文提出LExMem，一个基于图的长期记忆评测基准与框架，通过将试错式探索过程建模为图结构并引入演化机制，来评估模型在长期探索任务中追踪方案修订、分支与重新考量的记忆能力。

    

    长期问题解决过程，如科学研究和软件开发，是沿着一条不断演化的试错轨迹向中心目标推进的。我们将这类任务称为“探索任务”，它们要求模型不仅记住事实，还要记住解决方案如何针对中间结果进行检验，以及早期尝试如何被不断修改或重新启用。然而，现有的长期记忆研究主要关注事实的回忆与更新，忽视了探索过程的相互关联特性。我们观察到，探索过程天然形成图结构，其中分支的方法通过修订与重新考量相互关联。受此启发，我们提出了LExMem——一个以图为引导的基准，旨在评估长期探索任务中的记忆能力。我们将探索过程建模为图，并引入一种演化机制来增加其结构复杂性。

    arXiv:2611.06414v1 Announce Type: new  Abstract: Long-term problem-solving processes, such as scientific research and software development, proceed along an evolving trial-and-error trajectory toward a central goal. We term these \textit{exploration tasks}, which require models to remember not only facts but also how solutions are tested against intermediate outcomes and early attempts are continually revised or revived. However, existing long-term memory research focuses on factual recall and updates, overlooking the interconnected nature of exploration. We observe that exploration processes naturally form graphs, with branching approaches linked by revisions and reconsiderations. Motivated by this, we propose \textsc{LExMem}, a graph-guided benchmark designed to evaluate \underline{\textsc{Mem}}ory in \underline{\textsc{L}}ong-term \underline{\textsc{Ex}}ploration tasks. We model exploration processes as graphs and introduce an evolutionary mechanism to increase their structural comp
    
[^84]: AStockArena：超越收益，我们能否将真实资本托付给金融智能体的决策？

    AStockArena: Beyond Returns, Can We Entrust Real Capital to the Decisions of Financial Agents?

    [https://arxiv.org/abs/2611.00515](https://arxiv.org/abs/2611.00515)

    该论文提出AStockArena——一个针对中国A股市场连续交易的实时决策审计基准，通过涵盖执行可追溯性、证据充分性和金融元认知的八项检查来审计13个模型在3,198个连续决策中的真实行为，超越单纯收益指标以回答金融智能体是否值得托付真实资本的问题。

    

    大型语言模型越来越多地被应用于动态金融交易，然而现有的实时交易基准主要依据累计收益和风险调整后的表现来评估智能体。这类结果指标无法反映交易是否被如实报告、是否有可用证据支撑、以及是否与账户状态保持一致。我们提出了AStockArena，一个面向中国A股市场连续交易的实时决策审计基准。在共享的市场条件和执行约束下，我们评估了13个开源与闭源模型在3,198个连续决策上的表现。AStockArena通过将报告、交易请求、执行响应与账户状态进行对齐，重构了实际提交到账户的操作，并通过八项检查对其进行审计，这些检查涵盖执行可追溯性（O）、证据充分性（RK）和金融元认知（B），同时结合了记录验证与共识专家标注。在1,070个交易点中，72……（原文摘要在此处截断）

    arXiv:2611.00515v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used in dynamic financial trading, yet real-time trading benchmarks mainly evaluate agents by cumulative returns and risk-adjusted performance. Such outcome metrics cannot show whether trades are faithfully reported, supported by available evidence, and consistent with account states. We introduce AStockArena, a real-time decision-auditing benchmark for continuous trading in China's A-share market. Under shared market conditions and execution constraints, we evaluate 13 open- and closed-source models over 3,198 sequential decisions. AStockArena reconstructsthe actions actually posted to accounts by aligning reports, trade requests, execution responses, and account states, and audits them with eight checks spanning execution traceability (O), evidence sufficiency (RK), and financial meta-awareness (B), combining record verification with consensus expert annotation. Of 1,070 trading points, 72.
    
[^85]: TCC-RAG：面向时序知识图谱问答的时序因果链引导检索增强生成方法

    TCC-RAG: Temporal Causal Chain-Guided RAG for Temporal Knowledge Graph Question Answering

    [https://arxiv.org/abs/2610.38366](https://arxiv.org/abs/2610.38366)

    该论文提出TCC-RAG方法，通过将时序知识图谱重构为多视图因果线索图，并沿时序因果链检索简洁而充分的因果知识，克服了语义RAG难以捕获多跳因果知识且易受冗余信息干扰的缺陷，从而提升时序知识图谱问答的推理准确性。

    

    大语言模型（LLM）的最新进展激发了人们对时序知识图谱问答（TKGQA）日益浓厚的兴趣。现有的大多数方法依赖于语义检索增强生成（RAG），而这类方法在复杂时序推理方面存在根本性的局限。一方面，基于语义的RAG方法难以捕获复杂时序推理所需的多跳因果知识；另一方面，这些方法会引入过多的冗余知识，使LLM陷入虚假的推理链，从而可能导致错误的答案。为解决这些问题，我们提出了时序因果链引导的检索增强生成方法（TCC-RAG），以检索一个简洁而充分的因果知识集合来支持复杂的时序推理。具体而言，我们将时序知识图谱重新构建为多视图因果线索图，通过因果线索诱导的连接来显式编码时序演化与因果关系（摘要在此处截断）。

    arXiv:2610.38366v2 Announce Type: replace  Abstract: Recent advances in large language models (LLMs) have spurred growing interest in temporal knowledge graph question answering (TKGQA). Most existing methods rely on semantic retrieval augmented generation (RAG), which suffers from fundamental limitations in complex temporal reasoning. On the one hand, semantic-based RAG methods struggle to capture the multi-hop causal knowledge required for complex temporal reasoning. On the other hand, these methods introduce excessive redundant knowledge, trapping LLMs in spurious reasoning chains that potentially result in erroneous answers. To address these problems, we propose a Temporal Causal Chain-guided RAG (TCC-RAG) to retrieve a concise yet sufficient set of causal knowledge to support complex temporal reasoning. Specifically, we reformulate the temporal knowledge graph into a multi-view causal cue graph, with causal cues inducing connections that explicitly encode temporal evolution and ca
    
[^86]: 流行率偏移与标签预算约束下的类型化决策模型：决策训练、校准与隐式先验

    Typed Decision Models under Prevalence Shift and Label Budgets: Decision Training, Calibration, and Implicit Priors

    [https://arxiv.org/abs/2610.26325](https://arxiv.org/abs/2610.26325)

    本文在流行率偏移与标签预算约束下系统比较三个类型化决策模型与其骨干模型及微调分类器，发现厂商决策训练并非总是有益，其校准优势因任务类型而异。

    

    类型化决策模型返回调用者所定义选项上的概率分布，Jev 的供应商宣称其模型经过校准。我们探讨三个问题：供应商的决策训练能否在其骨干模型之上带来改进；类型化决策模型的校准与阈值能否像微调分类器一样经受住流行率（标签）偏移；以及一个经过训练的模型需要多少标注标签才能超越类型化决策模型。我们在模拟偏移条件下，在来自五个公开数据集的相同候选集决策上（在意图数据集上限于黄金标签所属的意图组内），对三个类型化决策模型（Jev、Tev1-4B、Nimble-9B）与其基础模型以及经过训练的分类器进行了比较；我们在确认性分析之前冻结了 35 项配对检验（并披露了此前进行的测试数据检查），同时估计了标签预算曲线。首先，决策训练并非总是有益：与 Qwen3.5-9B 仅相差一个 LoRA 适配器的 Nimble-9B 在意图任务上校准更好，但在金融情感任务上 Macro-F1 却低了 12 个百分点（原文摘要至此被截断）。

    arXiv:2610.26325v2 Announce Type: replace-cross  Abstract: Typed decision models (TDMs) return probabilities over caller-defined options, and Jev's vendor describes it as calibrated. We ask whether vendor decision training improves a model over its backbone, whether a TDM's calibration and thresholds withstand prevalence (label) shift like a fine-tuned classifier's, and how many labels a trained model needs to overtake a TDM. We compare three TDMs (Jev, Tev1-4B, Nimble-9B), their base models and trained classifiers on identical candidate-set decisions from five public datasets (on intent datasets, within the gold label's intent group) under simulated shift, with 35 paired tests frozen before the confirmatory analysis (earlier test-data checks disclosed), and estimate label-budget curves. First, decision training is not always beneficial: Nimble-9B, which differs from Qwen3.5-9B only by a LoRA adapter, is better calibrated on intent tasks but 12 Macro-F1 points worse on financial sentim
    
[^87]: LoongSage：面向前沿大语言模型的生产级、高效智能体强化学习框架（附经验证的训练方案）

    LoongSage: A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes

    [https://arxiv.org/abs/2610.16352](https://arxiv.org/abs/2610.16352)

    LoongSage是一个智能体原生、完全异步的生产级强化学习框架，基于Megatron与SGLang构建且核心仅约1.4万行代码，通过统一编排白盒与黑盒智能体的AgentFlow范式，显著降低了前沿大语言模型智能体强化学习的采用、使用与调试开销。

    

    强化学习（RL）已成为大语言模型（LLM）后训练流程中的关键阶段。随着大语言模型能力的前沿从静态推理转向智能体任务——其特征包括外部工具调用、多轮交互和长程决策——现有主流强化学习框架变得日益笨重，增加了采用、使用和调试的开销。为应对这些挑战，我们提出了LoongSage，一个智能体原生、完全异步的强化学习框架，其核心代码仅约1.4万行。LoongSage基于Megatron实现分布式训练，基于SGLang实现高吞吐量推理，将编排逻辑集中在单一控制器进程中，并将所有驻留GPU的组件封装为Ray actor。它引入了AgentFlow这一统一执行范式，可同时编排白盒与黑盒智能体。对于异步执行，（原文摘要在此处截断，后续内容缺失）

    arXiv:2610.16352v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has emerged as a critical phase in the post-training pipeline of large language models (LLMs). As the frontier of LLM capabilities shifts from static reasoning to agentic tasks--characterized by external tool invocation, multi-turn interactions, and long-horizon decision-making--existing mainstream RL frameworks have grown increasingly unwieldy, raising the overhead of adoption, usage, and debugging. To address these challenges, we present LoongSage, an agent-native, fully asynchronous RL framework with a lightweight core of roughly 14K lines of code. Built upon Megatron for distributed training and SGLang for high-throughput inference, LoongSage centralizes orchestration within a single controller process and encapsulates all GPU-resident components as Ray actors. It introduces AgentFlow, a unified execution paradigm that orchestrates both white-box and black-box agents. For asynchronous execution, 
    
[^88]: DV-Lens：揭示语言模型参数的功能组织

    DV-Lens: Revealing the Functional Organization of Language Model Parameters

    [https://arxiv.org/abs/2610.04489](https://arxiv.org/abs/2610.04489)

    提出DV-Lens参数级可解释性框架，通过估计模块特定的雅可比矩阵将语言模型参数投影到下游词汇空间，从而揭示参数的功能组织及其与模型能力的关系。

    

    理解参数的功能有助于阐明大语言模型（LLMs）的内部机制。然而，如何将来自不同模块的参数与可验证的输出效应联系起来，并进一步刻画其功能组织与模型能力之间的关系，仍有待探索。为此，我们提出了下游词汇透镜，这是一个参数级可解释性框架，将原生的参数方向与其下游词汇响应联系起来。具体而言，我们首先在参考提示集上估计注意力查询、键、值和输出投影以及前馈网络的模块特定下游雅可比矩阵。其次，我们利用这些映射将原生参数列投影到最终词汇空间，获得表征其平均局部输出响应的带符号读数。第三，我们根据词汇响应对参数列进行分组……

    arXiv:2610.04489v2 Announce Type: replace  Abstract: Understanding parameter functions helps elucidate the internal mechanisms of large language models (LLMs). However, how to connect parameters from different modules to verifiable output effects and further characterize the relationship between their functional organization and model capability remains to be explored. To this end, we introduce the downstream vocabulary lens (DV-Lens), a parameter-level interpretability framework that links native parameter directions to their downstream vocabulary responses. Specifically, we first estimate module-specific downstream Jacobians over a reference prompt set for attention query, key, value, and output (Q/K/V/O) projections and feed-forward networks (FFNs). Second, we use these mappings to project native parameter columns into the final vocabulary space, obtaining signed readouts that characterize their average local output responses. Third, we group parameter columns by their vocabulary re
    
[^89]: 用于风险校准意图路由的带符号词汇置信度

    Signed Lexical Confidence for Risk-Calibrated Intent Routing

    [https://arxiv.org/abs/2610.00262](https://arxiv.org/abs/2610.00262)

    提出一种将句子分类器logit间隔与稀疏词汇模型支持相结合的带符号词汇置信度门控，并配合独立的二项式校准选择阈值，在三个意图路由基准上将风险-覆盖率曲线下面积降低约12%至16%。

    

    选择性意图路由使助手能够对可靠的预测采取行动，同时推迟处理不确定的请求。标准置信度分数主要反映基础模型的表示，这为在不改变其决策的情况下纳入补充证据留下了机会。我们引入了一种带符号词汇门控，它将句子分类器的logit间隔与稀疏词汇模型对分类器预测意图的支持相结合。通过将词汇一致的证据指定为正面证据，将词汇上偏向竞争意图的证据指定为负面证据，该门控比无符号词汇置信度或硬性一致规则保留了更多信息。一个独立的二项式校准阶段为指定的风险目标选择运行阈值。在BANKING77、CLINC150和HWU64数据集上的十次运行中，与学习到的纯语义门控相比，所提出的分数将风险-覆盖率曲线下面积分别降低了15.8%、15.1%和11.8%。

    arXiv:2610.00262v1 Announce Type: cross  Abstract: Selective intent routing allows an assistant to act on reliable predictions while deferring uncertain requests. Standard confidence scores primarily reflect the base model's representation, leaving an opportunity to incorporate complementary evidence without changing its decisions. We introduce a signed lexical gate that combines a sentence classifier's logit margin with a sparse lexical model's support for the classifier's predicted intent. By assigning positive evidence to lexical agreement and negative evidence to a lexically favored competing intent, the gate retains more information than either unsigned lexical confidence or a hard agreement rule. An independent binomial calibration stage selects an operating threshold for a specified risk target. Across ten runs on BANKING77, CLINC150, and HWU64, the proposed score reduces area under the risk-coverage curve by 15.8%, 15.1%, and 11.8% relative to a learned semantic-only gate. At a
    
[^90]: SANTA++：通过代表性键进行注意力采样

    SANTA++: Sampling Attention through Representative Keys

    [https://arxiv.org/abs/2609.35629](https://arxiv.org/abs/2609.35629)

    SANTA++ 是一种无需训练的随机注意力方法，通过将 KV 缓存分组并用代表性键进行重要性采样，仅访问 28-31% 的 KV 缓存即可保留 95-98% 的稠密注意力性能，显著降低长上下文推理的内存开销。

    

    注意力通常集中在上下文中的一小部分标记上，但哪些标记重要会随查询的不同而变化。为了利用这种动态变化的结构，我们提出了 SANTA++，这是一种无需训练的随机注意力方法，它使用代表性键进行内存高效的选择，而无需扫描整个键值（KV）缓存。缓存的键被组织成多个“团队”（分组），查询对每个团队中的一个代表性键进行评分，以决定采样哪些团队。我们在被采样的团队内计算精确的注意力分数，并根据每个团队被纳入概率的倒数重新加权其贡献。这种重要性采样修正可以估计整个缓存上的注意力，采样预算让我们可以在内存读取量和精度之间进行权衡。在 32K 上下文长度的 Qwen2.5-7B-Instruct 上，S=1024 的连续 SANTA++ 在 LongBench v2、HELMET RAG 和 RULER 上仅使用 28-31% 的逻辑 KV 访问量，就保留了 95-98% 的稠密聚合分数。

    arXiv:2609.35629v2 Announce Type: replace-cross  Abstract: Attention often concentrates on a small subset of tokens in the context, but which subset matters changes from one query to the next. To exploit this changing structure, we introduce SANTA++, a training-free stochastic attention method that uses representative keys for memory-efficient selection without scanning the entire key-value (KV) cache. Cached keys are organized into teams, and the query scores one representative from each team to decide which teams to sample. We compute exact attention scores within the sampled teams and reweight each team's contribution by the inverse of its inclusion probability. This importance sampling correction estimates attention over the full cache, with a sampling budget that lets us trade memory reads for accuracy. With Qwen2.5-7B-Instruct at 32K context, contiguous SANTA++ at S=1024 retains 95-98% of dense aggregate scores on LongBench v2, HELMET RAG, and RULER using 28-31% logical KV access
    
[^91]: 面向量子增强扩散语言模型的电路超网络

    Circuit Hypernetworks for Quantum-Augmented Diffusion Language Models

    [https://arxiv.org/abs/2609.24657](https://arxiv.org/abs/2609.24657)

    提出HyperQ方法，通过轻量级电路超网络为冻结的掩码扩散语言模型注入词元条件化的量子残差分支，并利用计算成本随量子比特数线性增长的精确经典期望值表达式，实现高效的量子增强语言模型适配。

    

    语言模型可以通过改变应用于单个词元的计算来进行适配。量子电路提供了这样一种方法，但在大型模型内部评估更宽的电路可能在计算上代价高昂。本文提出了HyperQ，它向一个冻结的掩码扩散语言模型中添加了词元条件化的量子残差分支。量子残差分支是每个transformer块中的一个模块，它读取词元的隐藏状态，生成该词元电路的坐标，执行该电路，并通过残差连接将测量值加回。主干网络保持冻结，仅训练新增的分支。在每个分支内部，一个轻量级的电路超网络在共享的稀疏电路结构中生成词元特定的旋转角度、耦合强度和测量轴。所需的期望值具有精确的经典表达式，其评估成本随量子比特数量线性增长，使得电路……

    arXiv:2609.24657v1 Announce Type: cross  Abstract: Language models can be adapted by changing the computations applied to individual tokens. Quantum circuits offer one such approach, but evaluating wider circuits inside a large model can be computationally demanding. Here we introduce HyperQ, which adds token-conditioned quantum residual branches to a frozen masked-diffusion language model. A quantum residual branch is a module in each transformer block that reads a token's hidden state, emits the coordinates of that token's circuit, executes it, and adds the measured values back through a residual connection. The backbone remains frozen, and only the added branches are trained. Within each branch, a lightweight circuit hypernetwork emits token-specific rotation angles, coupling strengths, and measurement axes in a shared sparse circuit structure. The required expectation values have an exact classical expression whose evaluation cost grows linearly with the qubit count, enabling circu
    
[^92]: 蒸馏你所信任的：可靠性感知的多教师在线策略蒸馏

    Distill What You Trust: Reliability-Aware Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2609.23697](https://arxiv.org/abs/2609.23697)

    提出TrustMOPD方法，以专家模型RL训练前后相对于共享参考模型的位移作为token级可靠性代理，实现无标签的多教师加权蒸馏监督分配，将学生模型性能恢复率从54.4%大幅提升至91.5%。

    

    多教师在线策略蒸馏允许学生在自身生成的轨迹上从互补的专家模型中学习。然而，基于领域路由的方法在每个样本中仅选择一个教师，并在整个响应生成过程中保持固定。这种设计既依赖于混合训练语料库中通常缺失的标签，也无法在轨迹内部所需专业知识发生变化时调整教师选择。我们提出了TrustMOPD，它用无标签的、token级别的监督分配取代样本级教师选择。在每个学生生成的前缀处，TrustMOPD将每个专家模型经强化学习（RL）后相对于共享的RL前参考模型的位移作为局部可靠性的代理指标，跨教师对这些分数进行校准，并构建加权的蒸馏目标。在数学、代码和指令遵循任务上，TrustMOPD优于最强的无标签基线，将恢复率从54.4%提升至91.5%。

    arXiv:2609.23697v1 Announce Type: new  Abstract: Multi-teacher on-policy distillation allows a student to learn from complementary specialists on its own trajectories. Domain-routed approaches, however, select one teacher per example and keep it fixed throughout the response. This design both depends on labels that mixed training corpora often lack and cannot adapt teacher selection when the expertise required changes within a trajectory. We propose \textbf{TrustMOPD}, which replaces example-level teacher selection with label-free, token-level supervision allocation. At each student-generated prefix, TrustMOPD uses each specialist's RL-induced displacement from a shared pre-RL reference as a proxy for local reliability, calibrates these scores across teachers, and constructs a weighted distillation target. Across mathematics, code, and instruction following, TrustMOPD outperforms the strongest label-free baseline, increasing the recovery ratio from $54.4\%$ to $91.5\%$ on \textsc{Singl
    
[^93]: RPMem：为LLM智能体学习跨会话的长期循环参数化记忆

    RPMem: Learning Long-Term Recurrent Parametric Memory Across Sessions for LLM Agents

    [https://arxiv.org/abs/2609.23466](https://arxiv.org/abs/2609.23466)

    RPMem提出了一种两阶段架构，将LLM智能体的每个会话编译为模型无关的潜在记忆，经任务训练的循环门控整合后映射为LoRA参数，从而实现跨会话的长期参数化记忆演化，并在更换骨干模型时保持记忆可迁移。

    

    长时间运行的LLM智能体需要能够跨会话持续存在并不断演化的记忆。基于文本的记忆在每次查询时都需要检索和重建过去的交互，随着历史记录的增长，长程性能越来越依赖于检索质量和上下文推理能力。参数化记忆将经验直接编码到模型计算中，但现有方法对跨会话记忆演化的支持有限，且其与特定骨干模型的耦合进一步限制了模型替换后的记忆复用。我们提出了RPMem，这是一种两阶段架构，通过前向计算将每个会话编译为与模型无关的潜在记忆，并通过任务训练的循环门控选择性地将其与保留的记忆进行整合。整合后的记忆随后被映射为特定于骨干模型的低秩适配（LoRA）参数，使得编码能力在骨干模型被替换时仍可迁移。在三个（数据集/任务上）的评估……

    arXiv:2609.23466v1 Announce Type: new  Abstract: Long-running LLM agents require memory that persists and evolves across sessions. Text-based memory retrieves and reconstructs past interactions at every query, making long-horizon performance increasingly dependent on retrieval quality and contextual reasoning as histories grow. Parametric memory encodes experience directly into model computation, but existing approaches provide limited support for cross-session memory evolution. Their coupling to a specific backbone further restricts memory reuse after model replacement. We introduce RPMem, a two-stage architecture that compiles each session into a model-independent latent memory through forward computation and selectively integrates it with retained memory via a task-trained recurrent gate. The consolidated memory is then mapped to backbone-specific low-rank adaptation (LoRA) parameters, allowing the encoding capability to transfer when the backbone is replaced. Evaluation across thre
    
[^94]: Swiss-Knife：解码时可重构的外部化多目标对齐框架

    Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time

    [https://arxiv.org/abs/2609.22226](https://arxiv.org/abs/2609.22226)

    该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。

    

    解码时对齐方法通过使用外部奖励对候选续写进行评分并选择得分最大者，来引导一个冻结的语言模型。我们认为这种共享设计只是更大空间中的一个退化点。我们提出Swiss-Knife，一个外部化多目标对齐框架，其中对齐规范是一等运行时对象：可热插拔的评分刀片、批归一化器、成对聚合算子和选择规则。六个公理刻画了可容许的聚合算子，并且我们证明了一个表示定理：每个满足这些公理的算子都具有形式 $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$，这是一个双参数族，其中包含probit和logistic比较规则以及逐点argmax作为命名的坐标。在该族中，成对聚合在对抗性奖励污染下是Lipschitz稳定的，而argmax则不是，且候选批归一化……（原文摘要在此处截断）

    arXiv:2609.22226v1 Announce Type: new  Abstract: Decode-time alignment methods steer a frozen language model by scoring candidate continuations with an external reward and selecting the maximiser. We argue that this shared design is a single degenerate point in a much larger space. We introduce Swiss-Knife, a framework for externalised multi-objective alignment in which the alignment specification is a first-class runtime object: hot-swappable scoring blades, a batch normaliser, a pairwise aggregation operator, and a selection rule. Six axioms characterise the admissible aggregation operators, and we prove a representation theorem: every operator satisfying them has the form $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$, a two-parameter family containing probit and logistic comparison rules and pointwise argmax as named coordinates. Within it, pairwise aggregation is Lipschitz-stable under adversarial reward contamination while argmax is not, and Candidate-Batch Norma
    
[^95]: 通过添加少量SALT来改进跨语言词元表示

    Improving Cross-Lingual Token Representations by Adding a Pinch of SALT

    [https://arxiv.org/abs/2609.09953](https://arxiv.org/abs/2609.09953)

    SALT是一种轻量级后训练方法，通过向现有跨语言句子编码器注入跨度级监督信号来改进词元表示，在五个多语言词元级基准中的四个上取得最佳结果，同时还能提升句子级任务性能。

    

    跨语言句子编码器能够在数百种语言之间实现可扩展的迁移，为翻译挖掘和低资源环境下的零样本学习等应用提供支持。尽管这些编码器是为句子级对齐而训练的，但它们越来越多地被应用于幻觉检测和序列标注等词元级任务，这暴露了训练与实际使用之间的不匹配。我们提出了SALT，一种轻量级的后训练方法，通过向现有句子编码器注入跨度级（span-level）监督信号来改进词元表示。在五个多语言词元级基准测试中，SALT在其中四个上取得了最佳整体结果，优于其他微调策略和竞争性编码器。此外，它还提升了跨语言检索和分类任务中的句子级性能。这些结果表明，跨度级监督是同时改进词元表示和句子表示的有效信号。

    arXiv:2609.09953v1 Announce Type: new  Abstract: Cross-lingual sentence encoders enable scalable transfer across hundreds of languages, powering applications such as translation mining and zero-shot learning in low-resource settings. Although trained for sentence-level alignment, they are increasingly also applied to token-level tasks such as hallucination detection and sequence tagging, exposing a mismatch between training and usage. We propose SALT, a lightweight post-training method that improves token representations by injecting span-level supervision into existing sentence encoders. Across five multilingual token-level benchmarks, SALT achieves the best overall results on four of them, outperforming alternative fine-tuning strategies and competitive encoders. It also improves sentence-level performance on cross-lingual retrieval and classification tasks. These results demonstrate that span-level supervision is an effective signal for improving both token and sentence representati
    
[^96]: OracleZoom：基于在线策略自蒸馏启发的参考约束递归图像超分辨率

    OracleZoom: On-Policy Self-Distillation Inspired Reference-Constrained Recursive Image Super Resolution

    [https://arxiv.org/abs/2609.06490](https://arxiv.org/abs/2609.06490)

    OracleZoom提出了一种受自蒸馏启发、参考约束的递归图像超分辨率框架，通过直接与跨尺度监督、无参考质量目标、KL约束潜在先验和EMA一致性等技术，解决了深层递归缺乏真实标签监督的问题，在七个数据集上实现了极端放大倍率下最先进的超分辨率质量。

    

    递归超分辨率（SR）通过将预测结果反复馈入同一模型，将固定尺度的超分辨率扩展到极端放大倍率，类似于反复缩放图像。然而，在每个尺度上获取真实标签数据仍然具有挑战性，尤其是在深层阶段，因为所需的源分辨率呈几何级数增长，导致更深层的预测缺乏监督。我们提出了OracleZoom，这是一个受在线策略蒸馏启发、参考约束的框架，它在自身轨迹上进行训练，同时将最后的真实标签证据延伸到监督边界之外。直接监督和跨尺度监督约束可验证的内容，而无参考质量目标则引导未解决的精细尺度细节。KL约束的预训练潜在先验限制了由质量驱动的漂移，而EMA一致性则稳定了监督边界。在七个数据集上，OracleZoom在各种缩放尺度上实现了最先进的超分辨率质量，平均0

    arXiv:2609.06490v1 Announce Type: cross  Abstract: Recursive Super-Resolution (SR) extends fixed-scale SR to extreme magnification by repeatedly feeding predictions back into the same model, analogous to zooming an image repeatedly. However, ground truth availability at every scale, especially at depth, remains challenging as the required source resolution grows geometrically, leaving deeper predictions unsupervised. We present OracleZoom, an on-policy distillation-inspired, reference-constrained framework that trains on its trajectory while carrying the last ground-truth evidence beyond the supervision boundary. Direct and cross-scale supervision constrain verifiable content, while a no-reference quality objective guides unresolved fine-scale detail. A KL-constrained pretrained latent prior limits quality-driven drift, while EMA consistency stabilizes the supervision boundary. Across seven datasets, OracleZoom achieves the state-of-the-art SR quality across zooming scales, averaging 0
    
[^97]: XMerge：用于大语言模型深度压缩的跨轴选择与重构式层合并

    XMerge: Cross-Axis Selection and Reconstructive Layer Merging for LLM Depth Compression

    [https://arxiv.org/abs/2609.02083](https://arxiv.org/abs/2609.02083)

    XMerge 是一种训练后的大语言模型深度压缩方法，通过跨轴选择识别隐藏状态变化最小的层块，并利用局部边界重构重新拟合相邻存留块，在不改变架构、不增加推理参数、无需任务标签的情况下实现高质量的层删除压缩。

    

    删除完整的 transformer 层可以保持标准的服务架构，但现有的深度压缩方法可能会损失大量质量，且损失在不同模型间变化难以预测。我们提出了 XMerge，这是一种包含两个组件的训练后方法。跨轴选择用于识别隐藏状态的相对幅度和角度变化较小的层块，局部边界重构则重新拟合相邻的存留块以匹配原始两个块的输出。XMerge 不使用任务标签或端到端微调，也不引入架构变更或额外的推理时参数。在七个 Llama 和 Qwen 主干模型（0.5B-8B）、五个已发表的基线方法和三个层削减级别上的实验表明，其相对于基线的优势在最激进的删除设置下最大：在 k=4 时，它在 CORE（一个 22 项任务的聚合基准）上于七个主干模型中的六个上排名第一，并且在 MMLU 上也在七个中的六个上排名第一（两个基准上同时排名第一的有五个）。

    arXiv:2609.02083v1 Announce Type: cross  Abstract: Removing complete transformer layers preserves a standard serving architecture, but existing depth-compression methods can lose substantial quality, and the loss varies unpredictably across models. We introduce XMerge, a post-training method with two components. Cross-axis selection identifies a block with low relative-magnitude and angular hidden-state change, and local boundary reconstruction re-fits the adjacent surviving block to match the original two-block output. XMerge uses no task labels or end-to-end fine-tuning, and it introduces neither architectural changes nor additional inference-time parameters. Across seven Llama and Qwen backbones (0.5B-8B), five published baselines, and three layer-reduction levels, its advantage over baselines is largest at the most aggressive removal: at k=4 it ranks first on six of seven backbones on CORE (a 22-task aggregate) and, separately, on six of seven on MMLU (five of seven on both at once
    
[^98]: 顶格截断的接受直方图揭示块扩散投机解码中的搁浅加速

    Ceiling-Clipped Acceptance Histograms Indicate Stranded Speed-up in Block-Diffusion Speculative Decoding

    [https://arxiv.org/abs/2608.30427](https://arxiv.org/abs/2608.30427)

    该论文发现块扩散投机解码中，高接受率会使草稿模型提前耗尽训练时的块长度上限，造成未被实现的“搁浅加速”，提出用接受直方图顶格柱的尖峰作为投入训练算力前的预检指标，并证明推理时简单扩大块大小无法恢复该加速。

    

    投机解码通过一个高效的草稿模型为目标模型提出词元，由目标模型在单次前向传播中完成验证，从而加速生成过程，同时保持目标模型的输出分布。高接受率的块扩散草稿模型（如DFlash和DFlare）可以在一次并行传播中填满整个块。在许多周期中，目标模型会接受整个块，因此草稿模型在验证失败之前就已耗尽其训练时的块长度上限。我们将这种未被实现的接受称为“搁浅加速”。按提示或按周期统计的平均提交长度会掩盖这一现象，而接受直方图则能将其暴露为顶格柱中的尖峰，即接受整个块的周期所占的比例。我们建议在投入训练算力之前，将直方图作为一项预检手段。在推理阶段简单地扩大块大小并不能恢复这一加速，因为一旦块超过其训练尺寸，草稿模型的双向注意力即使在早期阶段也会使其分布发生偏移。

    arXiv:2608.30427v1 Announce Type: new  Abstract: Speculative decoding speeds up generation with an efficient draft model (drafter) that proposes tokens for a target model to verify in one pass, preserving the target's output distribution. High-acceptance block-diffusion drafters such as DFlash and DFlare fill an entire block in one parallel pass. In many cycles, the target accepts the whole block, so the drafter exhausts its trained block horizon before verification fails. We call this unrealized acceptance stranded speed-up. A mean committed length, per prompt or per cycle, hides it, whereas the acceptance histogram exposes it as a spike in the ceiling bin, the fraction of cycles that accept the entire block. We recommend the histogram as a preflight check before spending training compute. Naively widening the block at inference does not recover the speed-up, because once the block outgrows its training size, the drafter's bidirectional attention shifts its distribution even at early 
    
[^99]: Meta$^n$：通过涌现深度实现递归自我改进

    Meta$^n$: Recursive Self-Improvement through Emergent Depth

    [https://arxiv.org/abs/2608.24735](https://arxiv.org/abs/2608.24735)

    Meta$^n$通过固定元操作并递归作用于自身产物，突破了自我改进系统的元深度限制，实现了随收敛而定的涌现深度，从而在多个骨干网络上取得更优性能。

    

    arXiv:2608.24735v1 公告类型：新 摘要：自我改进的LLM代理会优化答案，而不是产生答案的过程。增加一个元级别的系统会保持该级别固定，而那些自我编辑的系统必须保持其自身编辑机制的一部分不变以维持稳定，从而将其实现的元深度限制在大约两层。我们提出了Meta$^n$，它保持元操作固定，并改为对其输入进行递归。该操作$\Omega$被反复应用于其自身的产物，读取下方求解器堆栈的追踪以及生成这些追踪的代码，然后将下一层编写为策略性预处理和可调用辅助函数的库。由于$\Omega$从未改变，它不会破坏系统的稳定性，并且由于其输入严格增长，每一层都比上一层从更高的视角进行推理。深度由收敛决定，而非预先固定，并且进化档案会搜索层链。在两种骨干网络上，Meta$^n$均表现更优。

    arXiv:2608.24735v1 Announce Type: new  Abstract: Self-improving LLM agents refine answers, not the process that produces those answers. Systems that add a meta-level hold that level fixed, and those that edit themselves must leave part of their own editing machinery untouched to stay stable, capping the meta-depth they realize at roughly two. We present Meta$^n$, which keeps the meta-operation fixed and recurses on its input instead. That operation, $\Omega$, is applied repeatedly to its own products, reading the traces of the solver stack below together with the code that produced them, then writing the next layer as a strategic pre-process and a library of callable helpers. Because $\Omega$ never changes, it cannot destabilize the system, and because its input strictly grows, each layer reasons from a higher vantage than the last. Depth is set by convergence rather than fixed in advance, and an evolutionary archive searches over layer chains. Across two backbones, Meta$^n$ outperform
    
[^100]: 在智能体软件开发中使用第三方API路由器的权衡在哪里？

    Where Is the Tradeoff in Using Third-Party API Routers for Agentic Software Development?

    [https://arxiv.org/abs/2607.23624](https://arxiv.org/abs/2607.23624)

    该论文首次实证研究了编码智能体中第三方API路由器的路由器端注入风险，通过四个隐蔽程度递增的干预级别（如响应替换）揭示其可对软件开发任务造成真实且难以检测的影响，而现有客户端权限机制难以有效防御。

    

    第三方API路由器已成为一种常见的中间层，统一了对日益多样化的LLM提供商的访问。在编码智能体工作流中，高自主性操作被广泛采用，因为它减少了交互开销。因此，位于智能体与上游提供商之间的第三方API路由器不可避免地占据了可信路径。它可以检查和修改每一个请求与响应，然而目前没有任何机制来验证提供商的输出与智能体最终执行的仓库级操作之间的一致性。因此，客户端权限机制在实践中可能变得无效。这种控制缺口是否会对软件开发任务产生真实的、难以检测的影响，此前尚未得到实证测量。在本文中，我们对编码智能体中的路由器端注入进行了实证研究，考察了四个隐蔽程度递增的干预级别：响应替换（L1）、响应追加……

    arXiv:2607.23624v3 Announce Type: replace-cross  Abstract: Third-party API routers have become a common layer that unifies access across increasingly diverse LLM providers. In coding-agent workflows, high-autonomy operation is widely adopted because it reduces interaction overhead. As a result, a third-party API router, which sits between the agent and the upstream provider, inevitably occupies the trusted path. It can inspect and modify every request and response, yet no mechanism verifies alignment between the provider's output and the repository-level actions ultimately executed by the agent. Consequently, client-side permission mechanisms may become ineffective in practice. Whether this control gap produces real, hard-to-detect effects on software development tasks remains empirically unmeasured. In this paper, we conduct an empirical study of router-side injection in coding agents, examining four intervention levels of increasing subtlety: Response Substitution (L1), Response Appe
    
[^101]: Molt：一个面向智能体强化学习的可扩展 PyTorch 原生训练框架

    Molt: A Scalable PyTorch-Native Training Framework for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2607.21653](https://arxiv.org/abs/2607.21653)

    提出了 Molt 框架，通过可组合模型并行、统一智能体接口、全异步 rollout 与优化以及分布式经验存储，实现了万亿参数规模下的智能体强化学习训练，且无需修改现有智能体的执行逻辑。

    

    智能体强化学习需要一种研究者能够在不牺牲模型规模或智能体执行控制权的前提下进行修改的基础设施。我们提出了 Molt，一个轻量级的 PyTorch 原生框架，将万亿参数规模的训练与标准智能体接口相结合。Molt 整合了四项能力：基于可组合模型并行的紧凑训练实现；统一的 OpenAI 和 Anthropic 接口，支持上下文压缩后的自动轨迹分段；完全异步的 rollout 与优化；以及面向长多模态轨迹的分布式经验存储。现有智能体可以保留其执行与上下文管理逻辑，同时由共享捕获层记录生成的 token 和行为概率。Rollout 工作进程将大体积的经验数据放入 Ray 的对象存储中，训练器进程通过引用检索分配给它的经验，从而避免对完整 rollout 进行集中式收集。

    arXiv:2607.21653v2 Announce Type: replace-cross  Abstract: Agentic reinforcement learning requires infrastructure that researchers can modify without sacrificing model scale or control over agent execution. We present Molt, a lightweight PyTorch-native framework that combines trillion-parameter training with standard agent interfaces. Molt integrates four capabilities: a compact training implementation built on composable model parallelism; unified OpenAI and Anthropic interfaces with automatic trajectory segmentation after context compaction; fully asynchronous rollout and optimization; and distributed experience storage for long, multimodal trajectories. Existing agents retain their execution and context-management logic while a shared capture layer records generated tokens and behavior probabilities. Rollout workers place heavy experience payloads in Ray's object store, and trainer ranks retrieve their assigned experiences by reference, avoiding a centralized gather of the full roll
    
[^102]: 面向多语言大模型对齐的元学习偏好

    Meta-Learning Preferences for Multilingual LLM Alignment

    [https://arxiv.org/abs/2607.13315](https://arxiv.org/abs/2607.13315)

    该论文提出了一种面向RLHF和DPO的元学习框架，通过从其他语言的偏好数据中学习可迁移的初始化，使模型在仅有100个目标语言偏好样本的极低资源场景下实现对齐，胜率相比基线提升高达33.4%。

    

    在多语言环境下对齐大语言模型时，不同语言间人类偏好数据可获取性的不均衡构成了重大挑战。为解决低资源语言对齐中数据不足的问题，我们提出了一种用于人类反馈强化学习（RLHF）和直接偏好优化（DPO）的元学习框架。通过利用其他语言的偏好数据，我们的框架学习到一个可迁移的初始化，使其能够在数据极少的情况下有效适应目标语言。我们为元奖励建模和元策略优化两种设置提供了理论保证，并在多语言基准上实证展示了该方法的有效性。在仅有100个目标语言偏好样本的极低资源设置下，我们的方法相比基线方法实现了高达33.4%的胜率提升，并在多个基准上持续优于基线方法。

    arXiv:2607.13315v3 Announce Type: replace  Abstract: Unequal availability of human preference data across languages poses a significant challenge for aligning large language models in multilingual settings. To address the lack of sufficient data in low-resource language alignment, we propose a meta-learning framework for Reinforcement Learning from Human Feedback and Direct Preference Optimization. By leveraging preference data from other languages, our framework learns a transferable initialization that enables effective adaptation to a target language with minimal data. We provide theoretical guarantees for both the meta-reward modeling and meta-policy optimization settings, and empirically demonstrate the effectiveness of our approach on multilingual benchmarks. In an extremely low-resource setting with only 100 target-language preference samples, our approach achieves up to $33.4\%$ win-rate improvements over baseline methods, and consistently outperforms baselines across multiple 
    
[^103]: EasyOPD：一个易于使用的大语言模型在线策略蒸馏框架

    EasyOPD: An Easy-to-use On-Policy Distillation Framework for Large Language Models

    [https://arxiv.org/abs/2607.11012](https://arxiv.org/abs/2607.11012)

    本文提出EasyOPD，一个基于verl构建的易用在线策略蒸馏框架，通过解耦用户配置、方法监督逻辑与底层执行，统一了现有OPD方法碎片化的实现方式，使框架更易于复现和扩展。

    

    传统的大语言模型蒸馏通常依赖于固定的教师生成数据，这可能无法覆盖不断演进的学生策略所遇到的状态。在线策略蒸馏（OPD）则改为在学生生成的轨迹上收集教师或评估器的监督信号。然而，现有的OPD方法在监督形式、分词器兼容性、教师模型访问方式以及监督粒度方面存在显著差异，导致实现碎片化，难以复现和扩展。我们提出了EasyOPD，一个基于verl（面向大语言模型的分布式强化学习框架）构建的在线策略蒸馏框架。EasyOPD将用户侧配置、方法特定的监督逻辑和基于verl的执行三者相互分离。其方法模块通过扩展边界与共享后端连接，这些扩展边界涵盖损失构建、轨迹元数据、奖励处理、分词器对齐以及教师侧计算等方面。

    arXiv:2607.11012v2 Announce Type: replace  Abstract: Conventional language-model distillation often relies on fixed teacher-generated data, which may not cover the states encountered by an evolving student policy. On-policy distillation (OPD) instead collects teacher or evaluator supervision on student-generated rollouts. However, existing OPD methods differ substantially in supervision form, tokenizer compatibility, teacher access, and supervision granularity, leading to fragmented implementations that are difficult to reproduce and extend. We present \textsc{EasyOPD}, an on-policy distillation framework built on verl, a distributed reinforcement-learning framework for large language models. \textsc{EasyOPD} separates user-side configuration, method-specific supervision logic, and verl-based execution. Its method modules connect to the shared backend through extension boundaries for loss construction, rollout metadata, reward processing, tokenizer alignment, and teacher-side computati
    
[^104]: RABBiT：通过脑调优实现快速自适应的BOLD基础模型，用于准确零样本和少样本预测大脑中的语音诱发响应

    RABBiT: Rapidly adaptive BOLD foundation model via brain-tuning for accurate zero-shot and few-shot prediction of speech-elicited responses in the brain

    [https://arxiv.org/abs/2607.05171](https://arxiv.org/abs/2607.05171)

    RABBiT是一个通过“脑调优”实现的紧凑型音频到fMRI基础模型，能够在零样本和少样本设置下准确预测大脑对自然语音的fMRI响应，仅需10分钟个体数据即可高效适应新参与者，性能超越现有SOTA fMRI基础模型。

    

    arXiv:2607.05171v2 公告类型：替换 摘要：大脑中的语言理解依赖于上下文，会因实验刺激和个体不同而变化，这使得构建能够同时在这两者之间泛化的计算模型变得困难。因此，需要一个能够捕捉共享结构、同时高效适应新参与者和新输入的语言诱发大脑活动基础模型。我们提出了RABBiT（Rapidly Adaptive BOLD foundation model via BraIn-Tuning，通过脑调优的快速自适应BOLD基础模型），这是一个紧凑的音频到fMRI编码器，专为准确的零样本和少样本预测而设计。在涵盖324名参与者、跨多个未见过的fMRI数据集的全面评估中，RABBiT能够在听觉和语言选择性脑区准确进行自然语音fMRI响应的零样本预测，超越了当前最先进（SOTA）的fMRI基础模型以及基于群体平均值的预测。仅需10分钟的参与者特定数据，RABBiT还能通过参数高效的方法进一步提升性能。

    arXiv:2607.05171v2 Announce Type: replace  Abstract: Language understanding in the brain is context-dependent, varying across experimental stimuli and individuals, which makes it difficult to build computational models that generalize across both. This calls for a foundation model of language-evoked brain activity that can capture shared structure while adapting efficiently to new participants and inputs. We introduce RABBiT (Rapidly Adaptive BOLD foundation model via BraIn-Tuning), a compact audio-to-fMRI encoder designed for accurate zero- and few-shot prediction. A comprehensive evaluation on 324 participants across multiple unseen fMRI datasets shows that RABBiT enables accurate zero-shot prediction of fMRI responses to natural speech across auditory and language-selective regions, surpassing the SOTA foundation model for fMRI and predictions based on group averages. With as little as 10 minutes of participant-specific data, RABBiT further improves performance via parameter-efficie
    
[^105]: 学习式停止何时有帮助？为推理模型服务配置风险受控的提前退出机制

    When Does Learning to Stop Help? Configuring Risk-Controlled Early Exit for Reasoning-Model Services

    [https://arxiv.org/abs/2606.30852](https://arxiv.org/abs/2606.30852)

    该论文提出一种基于工作负载校准数据的部署决策流程，通过Learn-then-Test以可证明的风险界限同时认证学习式停止规则与静态token预算两类提前退出方案并部署成本最低者，实验表明两类方案在不同模型与任务上各有优势、无普适胜者。

    

    提前退出通过在临时答案看起来已是最终答案时停止推理，降低了推理模型的服务成本。服务运营者必须在习得的停止规则与单一静态token预算之间做出选择，而相对于完整推理的节省幅度本身并不能解决这一选择问题。我们提出了一种基于来自实际工作负载的校准数据进行决策的流程：该流程使用Learn-then-Test对两类方法进行认证，确保在所有问题中“完整推理能正确回答但提前退出无法正确回答”的比例不超过给定界限；同时按照服务栈的方式对临时答案进行定价，并部署其中成本最低的已认证候选方案。在四个模型和五个基准测试上，没有任何一类方法始终占优。在5%的风险目标下，使用Qwen3处理自由格式数学任务时，习得的停止规则比静态预算多节省10.4-17.0个百分点的token，准确率仅比完整推理低2.4-2.9个百分点，而在MMLU-Pro上平均节省较少。哪一类方法成本更低也会随场景变化（原文在此处截断）。

    arXiv:2606.30852v3 Announce Type: replace  Abstract: Early exit lowers the cost of serving a reasoning model by stopping its reasoning once a provisional answer looks final. An operator must choose between a learned stopping rule and one static token budget, and savings against full reasoning do not settle the choice. We present a procedure that decides on calibration data from the workload. It certifies both families with Learn-then-Test to one bound on the fraction of all questions that full reasoning answers correctly but early exit does not, prices provisional answers as the serving stack does, and deploys the cheapest certified candidate. Across four models and five benchmarks neither family is better throughout. At a 5% risk target the learned rule saves 10.4-17.0 percentage points more tokens than the static budget on free-form mathematics with Qwen3, with accuracy 2.4-2.9 points below full reasoning, and less on MMLU-Pro on average. Which family costs less also changes with the
    
[^106]: 先知而后取：基于置信度的检索预算分配何时才有收益？

    Know Before You Fetch: When Does Confidence-Based Retrieval-Budget Allocation Pay Off?

    [https://arxiv.org/abs/2606.29959](https://arxiv.org/abs/2606.29959)

    该研究在相同的每查询平均检索预算上限下，将基于阅读器置信度的自适应检索分配与最佳简单策略（固定深度或其随机混合）对比，发现置信度分配在极紧预算（每查询平均半个段落）下于全部七个设置中稳定胜出（准确率提升0.010–0.075），其价值在于配给稀缺预算和避免有害检索。

    

    自适应检索增强生成利用阅读器的信号（例如其在无检索的“闭卷”作答时的置信度）来决定是否检索以及检索多少。我们不是将这些方法与总是检索大量段落的做法相比，而是在每查询平均段落数相同上限的约束下，与最佳的简单策略进行比较：固定深度检索或固定深度的随机混合。各策略分配零个、一个或五个段落，并在保留问题集上进行评分。七个约含2,000个问题的实验设置组合了TriviaQA、Natural Questions与MS MARCO数据集、Qwen3-8B与Qwen3-32B阅读器，以及两种检索源：按问题的段落池和350万段落的维基百科索引。基于置信度的分配可能从三个方面获益。它能配给稀缺的预算：在每查询平均半个段落的上限下，它在全部七个设置中都优于最佳简单策略，准确率提升0.010至0.075。它能避免有害检索：在单个段落的上限下……

    arXiv:2606.29959v2 Announce Type: replace-cross  Abstract: Adaptive retrieval-augmented generation uses a signal from the reader, such as its confidence when answering without retrieval (closed-book), to decide whether and how much to retrieve. We compare such methods not with always retrieving many passages but with the best simple policy under the same cap on average passages per query: a fixed depth or a random mixture of fixed depths. Policies allocate zero, one or five passages and are scored on held-out questions. Seven settings of about 2,000 questions combine TriviaQA, Natural Questions and MS MARCO, Qwen3-8B and Qwen3-32B readers, and two retrieval sources: per-question passage pools and a 3.5-million-passage Wikipedia index. Allocation by confidence could gain in three ways. It rations a scarce budget: at a cap of half a passage per query it beats the best simple policy in all seven settings, by 0.010 to 0.075 in accuracy. It avoids harmful retrieval: at caps of one passage o
    
[^107]: 基于图灵奖励学习用户模拟器

    Learning User Simulators with Turing Rewards

    [https://arxiv.org/abs/2606.19336](https://arxiv.org/abs/2606.19336)

    提出基于图灵测试的强化学习方法Turing-RL，通过判别性图灵奖励训练用户模拟器，使其生成与真实用户难以区分的回复，并在对话和论坛两个领域取得优于基线方法的效果。

    

    在交互环境中学习模拟人类用户，可以推动智能助手训练、个性化系统评估、社会科学研究等多个领域的发展。现有方法通常通过训练大型语言模型来匹配单一的真实用户回复，要么最大化对数概率，要么使用相似性奖励。我们提出Turing-RL：一种基于图灵测试的强化学习方法，用于训练用户模拟器模型。Turing-RL使用一个判别性的图灵奖励，并借助LLM评判器，根据用户历史记录评估生成的回复与真实用户回复的不可区分性得分，用户模拟器LLM通过这种奖励学习生成与用户可能说出的话无法区分的回复。在两个不同领域——对话聊天和Reddit论坛讨论中，我们发现Turing-RL在LLM评估和人类评估上均持续优于基线方法。

    arXiv:2606.19336v2 Announce Type: replace  Abstract: Learning to simulate human users in interactive settings could advance the training of agent assistants, evaluation of personalization systems, research in the social sciences, and more. Existing approaches generally do so by training a large language model (LLM) to match a single ground truth response, either by maximizing the log probability or by using a similarity reward. We instead propose Turing-RL: a Turing-Test-based reinforcement learning approach for training user simulator models. Turing-RL uses a discriminative Turing reward with an LLM judge to score how indistinguishable a generated response is from the real user's given the user's history, and the user simulator LLM learns to produce responses indistinguishable from what the user could have said with such rewards. Across two different domains--conversational chat and Reddit forum discussion--we find that Turing-RL consistently outperforms baseline methods on both LLM a
    
[^108]: Polar：一个用于评估大语言模型政治偏见的基准

    Polar: A Benchmark for Evaluating Political Bias in LLMs

    [https://arxiv.org/abs/2606.12922](https://arxiv.org/abs/2606.12922)

    提出了Polar基准，通过选项级似然概率在美韩两种政治语境和八个议题类别上测量38个大语言模型的政治偏见，发现所有模型在美国内容上均偏向左翼进步立场，且展示语言本身即可改变测得的偏见结果。

    

    大语言模型（LLM）中的政治偏见日益重要，但在不同的政治和语言环境下难以进行可复现的测量。我们提出了Polar，一个包含4,026个实例的多项选择基准，它通过选项级别的似然概率而非基于提示的生成方式来测量政治偏见。Polar涵盖源自宣言项目的两个意识形态维度和八个议题类别，并在美国和韩国的政治语境中并行评估模型。在38个大语言模型上的实验显示，测得的偏见随政治语境、议题类别、模型组和展示语言的变化而系统性变化。所有模型在美国政治内容上都倾向于左翼进步立场，但在韩国内容上则表现出更为居中和混合的模式。翻译实验进一步表明，仅展示语言本身就能改变测得的偏见。这些发现凸显了开展多语言、跨语境政治偏见评估的必要性。

    arXiv:2606.12922v2 Announce Type: replace  Abstract: Political bias in large language models (LLMs) is increasingly significant, but difficult to measure reproducibly across political and linguistic contexts. We introduce Polar, a 4,026-instance multiple-choice benchmark that measures political bias through option-level likelihoods rather than prompt-based generation. Polar covers two ideological axes and eight issue categories derived from the Manifesto Project, and evaluates models in parallel across U.S. and South Korean political contexts. Across 38 LLMs, measured bias varies systematically with political context, issue category, model group, and presentation language. All models lean left-progressive on U.S. political content, but show more centered and mixed patterns on South Korean content. Translation experiments further show that presentation language alone can shift measured bias. These findings highlight the need for multilingual and cross-contextual evaluation of political 
    
[^109]: CRAM：面向多模态持续指令微调的中心路由与自适应混合专家模型

    CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2606.02502](https://arxiv.org/abs/2606.02502)

    本文提出CRAM方法，通过中心路由和自适应混合专家模型，在缓解多模态持续指令微调中灾难性遗忘的同时，提升参数效率。

    

    多模态大语言模型（MLLMs）通过指令微调在共享生成框架下统一了异构的视觉-语言任务，然而现实世界部署需要持续的能力扩展，这使得多模态持续指令微调（MCIT）变得至关重要。现有方法要么使用共享参数集更新所有任务，要么为每个新任务分配专用模块。共享更新迫使异构任务相互竞争，导致已学能力的遗忘。相反，孤立扩展虽避免了干扰，但在长任务流中严重限制了参数效率。为解决这一困境，我们提出了CRAM（中心路由与自适应混合专家模型）。具体而言，通过将任务特定模式隔离到独立模块中，CRAM缓解了跨任务的灾难性遗忘。为进一步提升参数效率，我们利用自适应秩实例化来识别现有专家之间的能力差距。

    arXiv:2606.02502v2 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) unify heterogeneous vision-language tasks under a shared generative framework via instruction tuning, yet real-world deployment demands continuous capability expansion, making Multimodal Continual Instruction Tuning (MCIT) essential. Existing methods either update all tasks with a shared parameter set or allocate dedicated modules for each new task. Shared updates force heterogeneous tasks to compete, causing forgetting of learned capabilities. Conversely, isolated expansion prevents interference but severely limits parameter efficiency over long task streams. To address this dilemma, we propose CRAM (Centroid-Routing and Adaptive MoE). Specifically, by isolating task-specific patterns into independent modules, CRAM mitigates catastrophic forgetting across tasks. To further boost parameter efficiency, we utilize adaptive-rank instantiation to identify the capability gap between existing expert
    
[^110]: 通过填充从扩散语言模型中提取训练数据

    Extracting Training Data from Diffusion Language Models via Infilling

    [https://arxiv.org/abs/2605.24173](https://arxiv.org/abs/2605.24173)

    该论文提出“填充提取”协议，利用任意二进制掩码全面评估扩散语言模型的训练数据泄露风险，发现传统的前缀探测方法会显著低估记忆化风险，而边缘条件掩码可提取多达三倍的逐字训练序列。

    

    大型语言模型中的记忆化问题几乎一直仅通过前缀条件提取来研究，这是自回归模型的一种自然选择。然而，扩散语言模型可以在任意位置对被掩码的标记进行去噪。因此，仅使用前缀探测只能揭示扩散语言模型中记忆化的一个侧面，并显著低估了训练数据提取的风险。为了更真实地建模扩散语言模型中训练数据的可提取性，我们引入了填充提取，这是一种由任意二进制掩码参数化的数据提取协议，它涵盖了仅前缀探测方法，并考虑了扩散语言模型的双向归纳偏置。我们在LLaDA-8B和Dream-7B上实例化该方法，涵盖五种提取模式、三种训练流程和三个语料库，包括逐字泄露和部分泄露的情况。我们发现掩码的几何结构决定了可提取性：边缘条件掩码提取的逐字序列最多可达三倍。

    arXiv:2605.24173v2 Announce Type: replace-cross  Abstract: Memorization in large language models has been studied almost exclusively through prefix-conditioned extraction, a natural choice for autoregressive models. However, diffusion language models (DLMs) can denoise masked tokens at arbitrary positions. Thus, prefix-only probing reveals only one facet of memorization in DLMs and significantly underestimates the risk of training-data extraction. In order to realistically model extractability of training data in DLMs, we introduce infilling extraction, a data-extraction protocol parameterized by an arbitrary binary mask that subsumes prefix-only probing and accounts for the bidirectional inductive bias of DLMs. Instantiating it on LLaDA-8B and Dream-7B across five extraction modes, three training pipelines, and three corpora covering verbatim and partial leakage, we find that mask geometry governs extractability: edge-conditioned masks extract up to three times more verbatim sequences
    
[^111]: 稀疏层是扩展循环语言模型的关键

    Sparse Layers are Critical to Scaling Looped Language Models

    [https://arxiv.org/abs/2605.09165](https://arxiv.org/abs/2605.09165)

    研究发现稀疏混合专家（MoE）层是扩展循环语言模型的关键，因为不同循环通过激活不同专家可在不增加参数的情况下恢复表达能力，同时循环边界还提供了比标准模型更优的提前退出点。

    

    循环语言模型通过在深度上重复一组Transformer层来降低内存成本，并在循环边界处提供自然的提前退出点。然而，循环模型的扩展性能不如拥有独立层的标准Transformer。我们比较了标准Transformer和混合专家模型在有循环和无循环情况下的表现，得出两个主要结果。第一，我们发现循环MoE模型比标准基线具有更好的扩展性，而密集的循环模型则不然。我们将此归因于循环之间的路由分歧：在循环MoE模型中，每次通过相同的共享层时会激活不同的专家，从而在不增加额外参数的情况下恢复了模型的表达能力。第二个发现是，循环模型在提前退出方面比标准模型具有更好的计算与质量之间的权衡。由于每个循环都以产生最终输出的相同层结束，循环边界是更优的退出点，这一点已通过更早的输出得到证实。

    arXiv:2605.09165v3 Announce Type: replace-cross  Abstract: Looped language models repeat a set of transformer layers through depth, reducing memory costs and providing natural early-exit points at loop boundaries. However, looped models do not scale as favorably as standard transformers with unique layers. We compare standard and Mixture-of-Experts (MoE) transformers, with and without looping, and find two main results. First, we find Looped-MoE models scale better than the standard baseline while dense looped models do not. We trace this to routing divergence between loops: in Looped-MoE models, different experts are activated on each pass through the same shared layers, recovering expressivity without additional parameters. Our second finding is that looped models have better compute-quality trade-offs with early exits than standard models. Because each loop ends with the same layers that produce the final output, loop boundaries are superior exit points, as confirmed by earlier outp
    
[^112]: MedExAgent：在嘈杂临床环境中训练大语言模型智能体进行询问、检查与诊断

    MedExAgent: Training LLM Agents to Ask, Examine, and Diagnose in Noisy Clinical Environments

    [https://arxiv.org/abs/2605.07058](https://arxiv.org/abs/2605.07058)

    该论文提出MedExAgent，首次将临床诊断形式化为包含向患者提问、开具医学检查和给出诊断三类动作的POMDP，以训练能够在真实嘈杂临床环境中应对患者多样性与不完整信息的LLM智能体。

    

    真实世界的临床诊断是一个复杂的过程，医生既需要通过与患者的互动获取信息，也需要通过开展医学检查来获取信息。此外，医生还需要适应不同的患者角色设定，以及在整个诊疗过程中随时可能出现的噪声和不完整信息。然而，现有的医学大语言模型基准测试和自动诊断方法在很大程度上简化了这一过程，将其归结为单轮问答、无噪声对话或顺序化的检查安排等，忽略了临床诊断的交互性和不确定性本质。在本文中，我们旨在通过将临床诊断形式化为部分可观测马尔可夫决策过程（POMDP）来弥补这一空白，其中包含三种动作类型：向患者提问、以工具调用的形式开具医学检查，以及给出诊断结论。我们还引入了一个系统性的噪声模型，涵盖七种患者……（注：原文摘要在此处被截断）

    arXiv:2605.07058v2 Announce Type: replace-cross  Abstract: Real-world clinical diagnosis is a complex process in which the doctor is required to obtain information from both interaction with the patient and conducting medical exams. Additionally, the doctor needs to adapt to different patient personas, as well as noisy and incomplete information that can happen at any time during the process. However, existing benchmarks for medical LLMs and methods for automatic diagnosis largely simplify this process by reducing it to single-turn question answering, noise-free conversations, or sequential exam making, etc., ignoring the interactive and uncertain nature of clinical diagnosis. In this paper, we aim to address this gap by formalizing clinical diagnosis as a Partially Observable Markov Decision Process (POMDP) with three action types: questioning the patient, ordering medical exams as tool calls, and issuing a diagnosis. We also introduce a systematic noise model comprising seven patient
    
[^113]: HorizonBench：具有演变偏好的长程个性化基准测试

    HorizonBench: Long-Horizon Personalization with Evolving Preferences

    [https://arxiv.org/abs/2604.17283](https://arxiv.org/abs/2604.17283)

    HorizonBench是一个通过隐含提及的生活事件驱动用户偏好演变的长程个性化基准测试，用于评估AI助理能否从对话线索中感知并正确更新对用户偏好的认知。

    

    个人助理如今与同一用户对话长达数月，而用户的偏好会随着生活的变化而改变；一个继续按照用户早已不再需要的偏好来服务的助理，就是对该用户的失职。真实对话并不会记录偏好何时或为何发生变化，而现有的基准测试要么将偏好固定不变，要么让用户明确陈述变化，因此无法区分模型究竟是忘记了用户说过的话，还是未能更新对用户的认知。我们提出了HorizonBench及其数据生成器，它能够采样用户偏好的变化方式，并根据该变化序列生成长达六个月的对话；每次偏好变化都源于一个用户提到的生活事件，但用户并不会直接说明新的偏好。我们收集并发布了一个人类纵向偏好数据集，包含160人在13周内提供的83,024条评分，用以确定生活事件改变偏好的频率和幅度。HorizonBench包含来自360位用户的4,245个问题……

    arXiv:2604.17283v2 Announce Type: replace-cross  Abstract: Personal assistants now talk with the same user for months, and users' preferences change as their lives change; an assistant that keeps serving a preference the user has outgrown fails that user. Real conversations do not record when or why a preference changed, and existing benchmarks either hold preferences fixed or have the user state the change, so they cannot tell whether a model forgot what the user said or failed to update its belief about the user. We introduce HorizonBench and its data generator, which samples how a user's preferences change and generates six months of conversation from that sequence; each change follows a life event that the user mentions without stating the new preference. A human longitudinal preference dataset that we collect and release, 83,024 ratings from 160 people over 13 weeks, sets how often and how much life events change preferences. HorizonBench contains 4,245 questions from 360 users, a
    
[^114]: 免训练的专家语言模型动态升级

    Training-Free Dynamic Upcycling of Expert Language Models

    [https://arxiv.org/abs/2603.29765](https://arxiv.org/abs/2603.29765)

    本文提出 DUME，一种免训练的新方法，通过复用不同领域训练好的稠密专家模型来动态构建统一的 MoE 模型，从而避免多任务微调带来的过度专业化和灾难性遗忘问题。

    

    大型语言模型（LLMs）在广泛的专业任务上取得了卓越的性能，展现出强大的问题解决能力。然而，训练这些模型的成本极其高昂，且由于依赖通用知识数据集，这些模型往往缺乏特定领域的专业知识。专业知识微调可以解决这一问题；然而，这通常会导致过度专业化，且由于各领域目标存在分歧，构建单一的多领域专家模型仍然十分困难。此外，由于任务间的干扰和灾难性遗忘问题，多任务训练也面临巨大挑战。现有工作提出在专家混合架构中组合稠密模型的专业知识，但这种方法仍然需要进行多任务微调。为了解决这些问题，我们提出了动态升级专家混合模型，这是一种新颖的方法，能够复用在不同领域上训练的稠密专家模型，从而构建一个统一的 MoE 模型。

    arXiv:2603.29765v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have achieved remarkable performance on a wide range of specialized tasks, exhibiting strong problem-solving capabilities. However, training these models is prohibitively expensive, and they often lack domain-specific expertise because they rely on general knowledge datasets. Expertise finetuning can address this issue; however, it often leads to overspecialization, and developing a single multi-domain expert remains difficult due to diverging objectives. Furthermore, multitask training is challenging due to interference and catastrophic forgetting. Existing work proposes combining the expertise of dense models within a Mixture of Experts (MoE) architecture, although this approach still requires multitask finetuning. To address these issues, we introduce Dynamic Upcycling MoE (DUME), a novel approach that reuses dense experts trained on different domains to construct a unified MoE model. Our method 
    
[^115]: 利用层间几何结构改进大语言模型的最终层表示

    Improving LLM Final Representations with Inter-Layer Geometry

    [https://arxiv.org/abs/2603.22665](https://arxiv.org/abs/2603.22665)

    本文提出用轻量级图神经网络聚合大语言模型各层信息，并进一步设计基于SL(2,Zn)凯莱图的Cayley编码器，以稀疏、规则且低直径的数学拓扑替代昂贵的注意力机制，在提升效率的同时显著增强下游预测性能。

    

    基于大语言模型（LLM）进行预测的标准做法是将最终层的表示作为下游预测器的输入。然而，中间层可能编码了互补的、与任务相关的信号。因此，现有方法要么为每个任务搜索最佳的层，要么采用代价高昂的基于注意力的机制来学习层间聚合。在本工作中，我们首先证明这种复杂性是不必要的：在由LLM各层构成的完全连接图上使用一个轻量级的图神经网络（GNN），不仅比现有方法更高效，还能取得显著更强的预测性能。随后，我们提出了Cayley编码器（Cayley-Encoder），通过用定义在SL(2, Zn)上的凯莱图替换完全连接图，进一步提升了效率与预测性能。这些凯莱图提供了一种数学上有依据的拓扑结构，具有稀疏、构造上规则且低直径的特点，从而使各层之间能够进行有效的通信。

    arXiv:2603.22665v4 Announce Type: replace  Abstract: The standard in LLM-based prediction is to use the final-layer representation as the input to a downstream predictor. However, intermediate layers may encode complementary task-relevant signals. Existing approaches therefore either search for the best layer for each task or apply expensive attention-based mechanisms to learn inter-layer aggregation. In this work, we first show that such complexity is unnecessary: a lightweight Graph Neural Network over a fully connected graph of LLM layers is more efficient and achieves significantly stronger predictive performance than existing approaches. We then introduce the Cayley-Encoder, which further improves both efficiency and predictive performance by replacing the fully connected graph with a Cayley graph over SL(2, Zn). These Cayley graphs provide a mathematically grounded topology that is sparse, regular by construction, and has low diameter. This enables effective communication across 
    
[^116]: HYPER：假设引导的证据寻求检索规划

    HYPER: Hypothesis-Guided Planning for Evidence-Seeking Retrieval

    [https://arxiv.org/abs/2603.19008](https://arxiv.org/abs/2603.19008)

    HYPER是一个无需训练的检索前规划框架，以临时答案假设为锚点，将检索分解为支持、区分和特征三种证据角色，把RAG从面向主题的检索转变为面向证据的检索，从而在需要竞争性选择的任务中更好地支持最终决策。

    

    检索增强生成（RAG）通过将生成过程建立在外部非参数化知识之上，从而改进了大语言模型（LLM）。然而，当任务需要在相互竞争的可能性之间做出选择时（例如在多种合理备选方案中进行诊断），仅靠主题相关的上下文往往不足以驱动最终决策。为了解决这一问题，我们提出了HYPER：假设引导的证据寻求检索规划（Hypothesis-Guided Planning for Evidence-Seeking Retrieval），这是一个无需训练的检索前规划框架，将RAG从面向主题的检索重新定位为面向证据的检索。HYPER使用一个临时答案假设作为规划锚点，然后将检索分解为三种证据角色：支持（Support）寻找支持临时答案的证据；区分（Distinction）检索能够区分相互竞争备选方案的标准；特征（Feature）则获取决策所需的显著问题特征。这一规划步骤引导检索朝向接近决策的证据……

    arXiv:2603.19008v2 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) improves Large Language Models (LLMs) by grounding generation in external, non-parametric knowledge. However, when a task requires choosing among competing possibilities, such as a diagnosis among plausible alternatives, topically relevant context is often insufficient to drive the final decision. To address this issue, we propose HYPER: Hypothesis-Guided Planning for Evidence-Seeking Retrieval, a training-free pre-retrieval planning framework that reorients RAG from topic-oriented retrieval to evidence-oriented retrieval. HYPER uses a provisional answer hypothesis as a planning anchor. It then decomposes retrieval into three evidence roles. Support seeks evidence for the provisional answer. Distinction retrieves criteria that separate competing alternatives. Feature grounds salient question features needed for the decision. This planning step steers retrieval toward evidence near the decisi
    
[^117]: 超越左右之分的在线话语意识形态概念框架

    A conceptual framework for ideology in online discourse beyond the left and right

    [https://arxiv.org/abs/2603.18945](https://arxiv.org/abs/2603.18945)

    本文提出将意识形态概念化为多层次社会认知概念网络的新框架，突破了计算社会科学中单一左右党派轴线的研究局限，并将在线话语分析方法与意识形态理论相连接。

    

    计算社会科学（CSS）在研究在线话语时，大多将意识形态操作化为单一的左右党派轴线。这种方法掩盖了人们如何解读和参与与种族、气候、性别等领域相关的更具体的意识形态形态。我们提出了一个框架，将意识形态概念化为多层次的社会认知概念网络，并解释了这一意识形态概念模型如何与在线话语研究相联系。在此过程中，我们的框架阐明了意识形态如何在话语中与框架构建等相关社会过程一同显现，并为理解何时以及为何应在单次分析中共同研究多种概念（如价值观和信念）提供了论证。更广泛地说，该框架将研究在线话语的方法与意识形态理论相连接，使更丰富的社会话语分析成为可能，从而使两个领域都从中受益。

    arXiv:2603.18945v2 Announce Type: replace-cross  Abstract: Computational social science (CSS) has largely operationalized ideology along a single left/right partisan axis when studying online discourse. This approach obscures how people interpret and engage with more specific ideological formations related to race, climate, gender, and other domains. We introduce a framework that instead conceptualizes ideology as a multi-level socio-cognitive concept network and then explain how this conceptual model of ideology can be linked to the study of online discourse. In doing so, our framework clarifies how ideology manifests in discourse alongside related social processes such as framing, and provides an argument for better understanding of when and why we might study multiple concepts, such as values and beliefs, together in one analysis. More broadly, it bridges methods used to study online discourse with ideology theory, enabling richer analyses of social discourse that benefit both field
    
[^118]: 利用跨语言不一致性在混合专家大语言模型中进行知识定位

    Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency

    [https://arxiv.org/abs/2603.17102](https://arxiv.org/abs/2603.17102)

    该论文提出将大语言模型的跨语言不一致性作为可解释性工具，通过对比模型在不同语言中事实回忆成功与失败时的MoE路由行为，定位混合专家模型中负责特定知识的关键专家。

    

    现代大语言模型在不同语言之间的行为表现出显著差异，例如能够在某些语言中回忆起事实信息，而在其他语言中却无法做到。虽然这通常被视为一个需要缓解的问题，但在本工作中，我们提出将这种跨语言不一致性作为混合专家大语言模型可解释性的一种工具加以利用。我们的知识定位框架通过对比模型在正确回忆信息的语言集合与无法回忆的语言集合之间的路由行为，从而分离出在回答某项知识时发挥功能性作用的模型组件。我们的方法分为两个阶段：(1) 使用跨多种语言的困难事实问题查询模型，以生成“成功”和“失败”激活桶；(2) 对MoE路由器的逻辑值应用统计对比分析，以识别对知识重要的专家。为了验证必

    arXiv:2603.17102v2 Announce Type: replace  Abstract: Modern LLMs continue to exhibit significant variance in behavior across languages, such as being able to recall factual information in some languages but not others. While typically studied as a problem to be mitigated, in this work, we propose leveraging this cross-lingual inconsistency as a tool for interpretability in mixture-of-experts (MoE) LLMs. Our knowledge localization framework contrasts routing for sets of languages where the model correctly recalls information from languages where it fails. This allows us to isolate model components that play a functional role in answering about a piece of knowledge. Our method proceeds in two stages: (1) querying the model with difficult factual questions across a diverse set of languages to generate "success" and "failure" activation buckets and then (2) applying a statistical contrastive analysis to the MoE router logits to identify experts important for knowledge. To validate the nece
    
[^119]: S-GRADES——研究多样化评估环境中学生回答评估的泛化能力

    S-GRADES -- Studying Generalization of Student Response Assessments in Diverse Evaluative Settings

    [https://arxiv.org/abs/2603.10233](https://arxiv.org/abs/2603.10233)

    本文提出S-GRADES，一个开源的网页基准平台，将14个多样化的评分数据集整合于统一界面下，通过标准化访问和可复现的评估协议，弥合自动作文评分与自动简答题评分两个割裂领域，以研究学生回答评估在不同评估环境中的泛化能力。

    

    评估学生回答（从长篇作文到简短的事实性答案）是教育自然语言处理领域的一项关键挑战。自动作文评分（AES）关注连贯性和论证等整体写作质量，而自动简答题评分（ASAG）则强调事实正确性和概念理解。尽管两者目标相同，但这些范式一直孤立发展，存在数据集碎片化、评估指标不一致以及研究社区割裂等问题。我们提出了S-GRADES（Studying Generalization of Student Response Assessments in Diverse Evaluative Settings），这是一个基于网页的基准测试平台，将14个多样化的评分数据集整合在统一界面之下，提供标准化访问和可复现的评估协议。该基准完全开源，并专为可扩展性而设计，支持持续集成新的数据集和评估设置。为展示S-GRADES的实用性，我们评估了三种最先进的（摘要在此处被截断）

    arXiv:2603.10233v2 Announce Type: replace  Abstract: Evaluating student responses, from long essays to short factual answers, is a key challenge in educational NLP. Automated Essay Scoring (AES) focuses on holistic writing qualities such as coherence and argumentation, while Automatic Short Answer Grading (ASAG) emphasizes factual correctness and conceptual understanding. Despite their shared goal, these paradigms have progressed in isolation with fragmented datasets, inconsistent metrics, and separate communities. We introduce S-GRADES (Studying Generalization of Student Response Assessments in Diverse Evaluative Settings), a web-based benchmark that consolidates 14 diverse grading datasets under a unified interface with standardized access and reproducible evaluation protocols. The benchmark is fully open-source and designed for extensibility, enabling continuous integration of new datasets and evaluation settings. To demonstrate the utility of S-GRADES, we evaluate three state-of-th
    
[^120]: 在线强化学习在移动GUI智能体中的泛化能力有多远？基于AndroidWorld-Generalization的受控研究

    How Far Does Online RL Generalize in Mobile GUI Agents? A Controlled Study with AndroidWorld-Generalization

    [https://arxiv.org/abs/2603.07432](https://arxiv.org/abs/2603.07432)

    本文提出AndroidWorld-Generalization基准，通过未见实例、未见模板、未见应用三级递进的上下文偏移系统评估在线RL在移动GUI智能体中的泛化能力，发现RL收益随环境偏移距离增大而单调衰减。

    

    在线强化学习（RL）已将AndroidWorld上移动GUI智能体的成功率提升至80%以上，但这些系统在相同环境中进行训练和评估，且其训练任务未公开以供审计。因此，RL在训练环境之外的泛化能力究竟如何仍无从衡量。我们将移动环境形式化为上下文马尔可夫决策过程，并构建了一个由三级渐进式远离的上下文偏移组成的阶梯，即AndroidWorld-Generalization基准：未见实例、未见模板和未见应用，并人工验证了各层级之间的区分度。此外，我们还提供了一个开放的RL训练系统，使这一规模的研究成为可能。将相同的RL算法应用于两个来源与能力各异的初始模型——成功率为23.0%的UI-TARS-7B-SFT和成功率为52.6%的Qwen3-VL-8B——我们发现RL在分布内均可提升两者的性能，但增益随阶梯逐级单调衰减（分别为+17.5/+11.1/+4.2和+10.3/+7.4/+0），训练表现……

    arXiv:2603.07432v2 Announce Type: replace-cross  Abstract: Online reinforcement learning (RL) has advanced mobile GUI agents on AndroidWorld to over 80\% success, but these systems train and evaluate in the same environment, and their training tasks are not released for auditing. How far RL generalizes beyond its training contexts therefore remains unmeasured. We formalize the mobile environment as a Contextual MDP and instantiate a ladder of three progressively distant context shifts as AndroidWorld-Generalization: Unseen Instance, Unseen Template, and Unseen Application, with separation verified manually. In addition, we provide an open RL training system that makes studies at this scale feasible. Applying the identical RL algorithm to two priors of different origin and competence, UI-TARS-7B-SFT at 23.0\% and Qwen3-VL-8B at 52.6\%, we find that RL improves both in distribution but that gains fall monotonically up the ladder (+17.5/+11.1/+4.2 and +10.3/+7.4/+0), with training perform
    
[^121]: 通过离散随机局部化解锁连续扩散语言模型

    Unlocking Continuous Diffusion Language Models via Discrete Stochastic Localization

    [https://arxiv.org/abs/2602.16169](https://arxiv.org/abs/2602.16169)

    提出离散随机局部化（DSL）连续状态框架，通过将随机局部化通道与单位球面词元嵌入相结合，实现了仅依赖含噪状态、无需时间步条件的贝叶斯最优去噪器，使单个去噪网络即可支持连续去噪、掩码扩散、随机顺序自回归等多种采样方法，从而让连续扩散模型在离散序列生成上有效匹敌掩码离散扩散模型。

    

    连续扩散是非自回归生成的一个自然框架，但在离散序列生成任务上通常落后于掩码离散扩散模型（MDMs）。我们认为，通过对噪声参数化与词元表示几何结构的联合设计，连续扩散可以有效地建模离散序列。我们提出了离散随机局部化，这是一个连续状态框架，它将随机局部化通道与单位球面上的词元嵌入相结合，从而得到一个仅依赖于含噪状态的贝叶斯最优去噪器，无需额外的时间步条件。单个训练好的去噪网络即可支持多种采样方法，包括连续去噪、掩码扩散、随机顺序自回归生成以及先连续后离散的混合生成。因此，在这种连续状态框架下的并行生成并不必然意味着同步的去噪。

    arXiv:2602.16169v3 Announce Type: replace-cross  Abstract: Continuous diffusion is a natural framework for non-autoregressive generation but has generally lagged behind masked discrete diffusion models (MDMs) on discrete sequence generation. We argue that continuous diffusion can effectively model discrete sequences through the joint design of noise parameterization and token representation geometry. We introduce \emph{Discrete Stochastic Localization} (DSL), a continuous-state framework that combines a stochastic localization channel with unit-sphere token embeddings, resulting in a Bayes-optimal denoiser that depends only on the noisy state, without additional timestep conditioning. A single trained denoising network then supports multiple sampling methods, from continuous denoising to masked diffusion, random-order autoregressive generation and hybrid continuous-then-discrete generation. Parallel generation in this continuous-state framework, therefore, need not imply synchronized d
    
[^122]: Colosseum：审计协作式多智能体系统中的合谋行为

    Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems

    [https://arxiv.org/abs/2602.15198](https://arxiv.org/abs/2602.15198)

    提出了Colosseum框架，通过形式化多智能体决策框架、基于遗憾的合谋度量以及秘密通信信道行为探针，系统性地审计LLM多智能体系统中的合谋行为。

    

    在多智能体系统中，LLM智能体通过自由形式的语言进行交流，能够实现复杂的协调来解决复杂的协作任务。当一组智能体结成联盟并合谋追求次要目标、损害共同目标时，这带来了一个独特的安全问题。在本文中，我们提出了Colosseum，一个用于审计多智能体环境中LLM智能体合谋行为的框架。我们通过形式化的多智能体决策框架为智能体如何协作奠定基础，并通过相对于协作最优解的遗憾（regret）来度量基于动作的合谋行为，同时将其与基于通信的合谋行为进行比较。Colosseum能够在良性设置、不同联盟目标、说服策略和网络拓扑下对LLM智能体的合谋行为进行审计。随后，我们通过在智能体之间创建秘密通信信道引入了一种新的行为探针方法，结果表明大多数外（原文摘要在此处被截断）

    arXiv:2602.15198v3 Announce Type: replace-cross  Abstract: Multi-agent systems, where LLM agents communicate through free-form language, enable sophisticated coordination for solving complex cooperative tasks. This surfaces a unique safety problem when a group of agents forms a coalition and colludes to pursue secondary goals and degrade the joint objective. In this paper, we present Colosseum, a framework for auditing LLM agents' collusive behavior in multi-agent settings. We ground how agents cooperate through a formal multi-agent decision-making framework and measure action-based collusive behavior in actions via regret relative to the cooperative optimum and compare it with communication-based collusive behavior. Colosseum enables audits of LLM agents for collusion under benign settings, different coalition objectives, persuasion tactics, and network topologies. We then introduce a new behavioral probe by creating secret communication channels between agents, showing that most out-
    
[^123]: 面向图表理解的多模态信息融合：多模态大语言模型（MLLM）综述——演进、局限性与认知增强

    Multimodal Information Fusion for Chart Understanding: A Survey of MLLMs -- Evolution, Limitations, and Cognitive Enhancement

    [https://arxiv.org/abs/2602.10138](https://arxiv.org/abs/2602.10138)

    本综述系统梳理了基于多模态大语言模型（MLLM）的图表理解研究，分析了视觉与语言信息融合的核心挑战，提出了规范型与非规范型基准的新颖分类体系，并追溯了从经典深度学习技术到先进MLLM融合范式的方法演进历程。

    

    图表理解是一项典型的信息融合任务，需要将图形数据与文本数据无缝整合以提取含义。多模态大语言模型（MLLM）的出现彻底变革了这一领域，然而基于MLLM的图表分析研究现状仍较为碎片化，缺乏系统性梳理。本综述通过构建该领域核心组件的结构化框架，为这一新兴前沿提供了全面的路线图。我们首先分析了图表中视觉与语言信息融合所面临的基本挑战。随后，我们对下游任务和数据集进行了分类，并引入了一种新颖的规范型与非规范型基准分类法，以凸显该领域不断扩展的研究范围。接着，我们全面梳理了方法论的演进历程，从经典深度学习技术追溯到采用复杂融合策略的最先进MLLM范式。

    arXiv:2602.10138v2 Announce Type: replace-cross  Abstract: Chart understanding is a quintessential information fusion task, requiring the seamless integration of graphical and textual data to extract meaning. The advent of Multimodal Large Language Models (MLLMs) has revolutionized this domain, yet the landscape of MLLM-based chart analysis remains fragmented and lacks systematic organization. This survey provides a comprehensive roadmap of this nascent frontier by structuring the domain's core components. We begin by analyzing the fundamental challenges of fusing visual and linguistic information in charts. We then categorize downstream tasks and datasets, introducing a novel taxonomy of canonical and non-canonical benchmarks to highlight the field's expanding scope. Subsequently, we present a comprehensive evolution of methodologies, tracing the progression from classic deep learning techniques to state-of-the-art MLLM paradigms that leverage sophisticated fusion strategies. By criti
    
[^124]: 利用国际象棋解耦大语言模型中的泛化能力与记忆能力

    Disentangling generalization and memorization in large language models using chess

    [https://arxiv.org/abs/2601.16823](https://arxiv.org/abs/2601.16823)

    该论文提出以国际象棋作为受控测试平台，通过按相关先验密度构建棋局分类体系，在无需了解模型训练数据的情况下区分大语言模型的记忆能力与真正的泛化推理能力，并发现模型性能随相关先验密度降低而持续显著下降。

    

    大语言模型（LLM）展现出卓越的能力，然而这些能力在多大程度上反映了精细的记忆召回还是真正的推理能力，目前仍不清楚。我们引入国际象棋作为一个受控测试平台，旨在解耦这两种能力。利用国际象棋的棋局结构和可扩展的引擎评估，我们构建了一个按相关先验密度变化的棋局分类体系——涵盖从可通过记忆解决的常见局面，到需要泛化能力的完全新颖的局面。关键的是，我们的方法无需对模型训练数据的显式了解即可实现这种区分。应用这一分类体系，我们将对GPT系列模型的纵向分析与对包括Claude Opus和Gemini在内的当代模型的严格评估相结合。我们的分析揭示了一个陡峭的梯度：随着相关先验密度的降低，模型性能持续下降。值得注意的是，对于相关先验较少的任务，基础模型

    arXiv:2601.16823v3 Announce Type: replace  Abstract: Large Language Models (LLMs) exhibit remarkable capabilities, yet it remains unclear to what extent these reflect sophisticated recall or genuine reasoning ability. We introduce chess as a controlled testbed aimed at disentangling these faculties. Leveraging the game's structure and scalable engine evaluations, we construct a taxonomy of positions varying in density of relevant priors -- ranging from common states solvable by memorization to completely novel ones requiring generalization. Crucially, our approach achieves this distinction without requiring explicit knowledge of the models' training data. Applying this taxonomy, we combine a longitudinal analysis of the GPT lineage with a rigorous evaluation of contemporary models, including Claude Opus and Gemini. Our analysis reveals a steep gradient: performance consistently degrades as the density of relevant priors decreases. Notably, for tasks with few relevant priors, base model
    
[^125]: 挑战二次复杂度注意力——替代性语言模型架构崛起的全景综述

    Challenging Quadratic Attention$\unicode{x2013}$A Holistic View on the Rise of Alternative Language Model Architectures

    [https://arxiv.org/abs/2510.05364](https://arxiv.org/abs/2510.05364)

    本文综述了亚二次注意力变体、循环神经网络、状态空间模型及混合架构等替代性语言模型架构的最新进展，并认为纯注意力 Transformer 的主导地位可能被挑战，尤其是在特定领域和边缘设备应用中。

    

    在过去七年中，Transformer 一直主导着序列处理任务——尤其是在语言建模领域。然而，其注意力机制固有的二次复杂度随着上下文长度的增加仍是显著的瓶颈。我们回顾并提炼了近期为克服这一瓶颈所做出的努力，包括（亚二次复杂度）注意力变体、循环神经网络、状态空间模型以及混合架构等方面的进展。我们从计算与内存复杂度、基准测试结果以及根本局限性等角度对这些方法进行了批判性分析，以评估纯注意力 Transformer 的主导地位是否可能很快受到挑战。我们认为这是可能的，尤其是在特定领域和边缘设备应用中。

    arXiv:2510.05364v2 Announce Type: replace  Abstract: Transformers have dominated sequence processing tasks for the past seven years$\unicode{x2013}$most notably language modeling. However, the inherent quadratic complexity of their attention mechanism remains a significant bottleneck as context length increases. We review and distill the recent efforts to overcome this bottleneck, including advances in (sub-quadratic) attention variants, recurrent neural networks, state space models, and hybrid architectures. We critically analyze approaches regarding compute and memory complexity, benchmark results, and fundamental limitations to assess whether the dominance of pure-attention transformers may soon be challenged, which we consider possible, particularly in domain-specific and edge-device applications.
    
[^126]: MARBLE：一个用于空间推理与视觉抽象的智能体基准测试

    MARBLE: An Agentic Benchmark for Spatial Reasoning and Visual Abstraction

    [https://arxiv.org/abs/2506.22992](https://arxiv.org/abs/2506.22992)

    提出了MARBLE基准测试，通过三个智能体环境系统评估多模态语言模型的空间推理与视觉抽象能力，发现最强模型GPT-5.4的平均成功率仅为7.3%，揭示了现有模型在此类任务上的巨大局限。

    

    多模态语言模型（MLLMs）在真实世界任务中的一个核心挑战是将视觉观察抽象为结构化的空间状态，并对这些状态在长时间序列动作下的变化进行推理。现有的多模态基准测试往往聚焦于静态问答（其答案可直接从图像中提取），或者假设符号状态已经可用而跳过从视觉到空间的抽象过程。我们提出了MARBLE，这是一个针对空间接地多模态推理与规划的诊断性基准测试，涵盖三个智能体环境：CUBE、MAZE和BLOX。MARBLE包含几何约束、动态布局和空间变换下的2D和3D空间谜题，并通过两个维度（观察格式和规划时域）系统地控制难度。广泛的评估表明，前沿MLLMs在MARBLE上表现不佳，其中最强的模型GPT-5.4仅取得了7.3%的平均成功率。

    arXiv:2506.22992v2 Announce Type: replace  Abstract: A central challenge for multimodal language models (MLLMs) in real-world tasks is to abstract visual observations into structured spatial states and reason over how those states change under actions over long horizons. Existing multimodal benchmarks often focus on static question answering, where answers can be directly extracted from images, or assume symbolic states are available and bypass the visual-to-spatial abstraction process. We present MARBLE, a diagnostic benchmark for spatially grounded multimodal reasoning and planning across three agent environments: CUBE, MAZE, and BLOX. MARBLE spans 2D and 3D spatial puzzles under geometric constraints, dynamic layouts, and spatial transformations, with difficulty systematically controlled along two axes: observation format and planning horizon. Extensive evaluations show that frontier MLLMs struggle on MARBLE, with the strongest model GPT-5.4 achieving only a 7.3% average success rat
    
[^127]: 大语言模型与小语言模型协同机制综述

    A Survey on Collaborative Mechanisms Between Large and Small Language Models

    [https://arxiv.org/abs/2505.07460](https://arxiv.org/abs/2505.07460)

    本综述系统梳理了大语言模型与小语言模型之间五种协作机制（流水线、路由、辅助、蒸馏、融合）及其关键使能技术和面向边缘设备的应用场景，为构建高效、灵活且易获取的AI系统提供了全面的参考框架。

    

    大语言模型能够提供强大的AI能力，但由于高昂的资源成本和延迟而面临部署挑战；而小语言模型以牺牲部分性能为代价，提供了高效率与可部署性。大语言模型与小语言模型之间的协作成为一种关键范式，可以协同平衡这些权衡取舍，从而支持先进的AI应用，尤其是在资源受限的边缘设备上。本综述对LLM-SLM协作进行了全面概述，详细介绍了各种交互机制（流水线、路由、辅助、蒸馏、融合）、关键使能技术，以及由设备端需求（如低延迟、隐私保护、个性化和离线运行）所驱动的多样化应用场景。在强调构建更高效、更灵活、更易获取的AI所具有的巨大潜力的同时，我们还讨论了持续存在的挑战，包括系统开销、模型间一致性等（原文在此处截断）。

    arXiv:2505.07460v2 Announce Type: replace  Abstract: Large Language Models (LLMs) deliver powerful AI capabilities but face deployment challenges due to high resource costs and latency, whereas Small Language Models (SLMs) offer efficiency and deployability at the cost of reduced performance. Collaboration between LLMs and SLMs emerges as a crucial paradigm to synergistically balance these trade-offs, enabling advanced AI applications, especially on resource-constrained edge devices. This survey provides a comprehensive overview of LLM-SLM collaboration, detailing various interaction mechanisms (pipeline, routing, auxiliary, distillation, fusion), key enabling technologies, and diverse application scenarios driven by on-device needs like low latency, privacy, personalization, and offline operation. While highlighting the significant potential for creating more efficient, adaptable, and accessible AI, we also discuss persistent challenges including system overhead, inter-model consisten
    
[^128]: Quicksviewer：一种使用退火Gumbel-Softmax进行早期视频压缩的大型多模态模型

    Quicksviewer: An LMM with Early Video Compression using Annealed Gumbel-Softmax

    [https://arxiv.org/abs/2504.15270](https://arxiv.org/abs/2504.15270)

    Quicksviewer提出了一种基于退火Gumbel-Softmax的早期视频压缩方法，通过将非均匀冗余视频在线动态划分为不同大小的块并进行统一重采样，实现45倍压缩率的高效视频理解。

    

    大型多模态模型（LMMs）统一地感知视频帧，这对于具有固有变化时间冗余的视频来说造成了计算效率低下的问题。本文提出了Quicksviewer，一种具有新感知范式的大型多模态模型，它使用Gumbel Softmax将非均匀冗余视频划分为不同大小的块，然后对每个块进行统一重采样，以实现高效的视频理解。我们进一步提出了一种Gumbel噪声退火策略，使离散分块网络能够稳定地进行端到端LMM训练。这种简单直观的方法基于帧差的动量在线动态压缩视频，显著降低了时空复杂度（总体45倍压缩率），同时通过大感受野实现高效训练。我们从语言骨干网络开始，通过三个渐进阶段训练模型，每个阶段都纳入平均时长420秒/1帧每秒的长视频……

    arXiv:2504.15270v2 Announce Type: replace-cross  Abstract: Large Multimodal Models (LMMs) uniformly perceive video frames, creating computational inefficiency for videos with inherently varying temporal redundancy. This paper present **Quicksviewer**, an LMM with new perceiving paradigm that partitions a nonuniform redundant video into varying cubes using Gumbel Softmax, followed by a unified resampling for each cube to achieve efficient video understanding. We further propose a Gumbel-noise annealing strategy that enables stable end-to-end LMM training with the discrete cubing network. This simple and intuitive approach dynamically compress video online based on the momentum of frame differences, significantly reducing spatiotemporal complexity (overall 45$\times$ compression rate), while enabling efficient training with large receptive field. We train the model from a language backbone through three progressive stages, each incorporating lengthy videos on average of 420s/1fps thanks 
    
[^129]: 基于参考奖励的强化学习用于语言模型推理与对齐

    Reinforcement Learning with Reference-Based Rewards for Language Model Reasoning and Alignment

    [https://arxiv.org/abs/2504.09895](https://arxiv.org/abs/2504.09895)

    提出RL2R方法，利用参考答案与轻量级语义相似度指标（如BERTScore）作为奖励信号，为语言模型的推理与对齐提供统一且低成本的强化学习后训练方案。

    

    强化学习（RL）支撑着大语言模型（LLM）后训练的两大主要方向：推理，即引导模型进行深思熟虑、可验证的问题求解；以及对齐，即引导模型表现出有益、无害且诚实的行为。两者都依赖于奖励设计。推理依赖于对最终答案的基于规则的验证，其二元反馈无法感知中间推导过程；而过程奖励模型（PRM）需要步骤级标注，成本过高难以规模化。对齐则依赖于在二元偏好对上训练的奖励模型（RM），使数据收集和模型训练都非常耗费资源。我们提出RL2R——基于参考奖励的强化学习，作为一种统一且低成本的替代方案。利用已有的或蒸馏得到的参考答案，RL2R通过轻量级指标（如BERTScore）度量采样生成结果与参考答案或解题过程之间的语义相似度，以相似度……（原文在此截断）

    arXiv:2504.09895v4 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) powers two major lines of large language model (LLM) post-training: reasoning, which elicits deliberate, verifiable problem solving, and alignment, which steers models toward helpful, harmless, and honest behavior. Both hinge on reward design. Reasoning relies on rule-based verification of final answers, whose binary feedback is blind to intermediate derivations, whereas process reward models (PRMs) require step-level annotations too costly to scale. Alignment depends on reward models (RMs) trained on binary preference pairs, making both data collection and model training resource-intensive. We propose RL2R, reinforcement learning with reference-based rewards, as a unified and inexpensive alternative. Using pre-existing or distilled references, RL2R measures semantic similarity with lightweight metrics such as BERTScore between a sampled generation and a reference answer or solution. Similarity prese
    
[^130]: MultiAgentBench：评估大语言模型智能体的协作与竞争

    MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents

    [https://arxiv.org/abs/2503.01935](https://arxiv.org/abs/2503.01935)

    提出了综合基准MultiAgentBench，利用新颖的基于里程碑的性能指标评估LLM多智能体系统在多样化交互场景中的任务完成、协作与竞争质量，并系统比较了不同协调拓扑结构与群体讨论、认知规划等创新策略的效果。

    

    大语言模型（LLM）作为自主智能体已展现出卓越能力，但现有基准要么聚焦于单智能体任务，要么局限于狭窄领域，无法捕捉多智能体协调与竞争的动态特性。本文提出了MultiAgentBench，一个旨在跨多样化交互场景评估基于LLM的多智能体系统的综合基准。我们的框架不仅衡量任务完成情况，还通过新颖的基于里程碑的关键性能指标来衡量协作与竞争的质量。此外，我们评估了多种协调协议（包括星形、链式、树形和图拓扑结构）以及群体讨论和认知规划等创新策略。值得注意的是，gpt-4o-mini获得了平均最高的任务分数，图结构在研究场景的协调协议中表现最佳，认知规划提升了里程碑达成率。

    arXiv:2503.01935v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have shown remarkable capabilities as autonomous agents, yet existing benchmarks either focus on single-agent tasks or are confined to narrow domains, failing to capture the dynamics of multi-agent coordination and competition. In this paper, we introduce MultiAgentBench, a comprehensive benchmark designed to evaluate LLM-based multi-agent systems across diverse, interactive scenarios. Our framework measures not only task completion but also the quality of collaboration and competition using novel, milestone-based key performance indicators. Moreover, we evaluate various coordination protocols (including star, chain, tree, and graph topologies) and innovative strategies such as group discussion and cognitive planning. Notably, gpt-4o-mini reaches the average highest task score, graph structure performs the best among coordination protocols in the research scenario, and cognitive planning improves mi
    
[^131]: 上下文学习器混合模型

    Mixtures of In-Context Learners

    [https://arxiv.org/abs/2411.02830](https://arxiv.org/abs/2411.02830)

    提出MoICL方法，将上下文学习的演示示例子集视为专家并学习加权函数来合并其输出分布，在提升分类性能的同时减少推理时间，并显著增强了对域外、不平衡和噪声示例的鲁棒性。

    

    上下文学习通过提供示例（演示）来适应大语言模型，而无需微调模型参数；然而，它无法区分不同的示例，并且会使Transformer大语言模型的复杂度呈平方级增长，从而耗尽内存。作为解决方案，我们提出了上下文学习器混合模型（MoICL），这是一种新颖的方法，它将演示示例的子集视为“专家”，并基于训练集学习一个加权函数来合并它们的输出分布。在实验中，与一组强基线相比，我们在7个分类数据集中的5个上展示了性能提升（与ICL和LENS相比最高提升+13%）。此外，我们通过减少以更少示例达到相同性能所需的推理时间，改善了ICL的帕累托前沿。最后，MoICL对域外示例（最高+11%）、不平衡示例（最高+49%）或噪声示例（最高+38%）更加鲁棒，或者可以过滤掉这些示例。

    arXiv:2411.02830v2 Announce Type: replace-cross  Abstract: In-context learning (ICL) adapts LLMs by providing demonstrations without fine-tuning the model parameters; however, it does not differentiate between demonstrations and quadratically increases the complexity of Transformer LLMs, exhausting the memory. As a solution, we propose Mixtures of In-Context Learners (MoICL), a novel approach to treat subsets of demonstrations as experts and learn a weighting function to merge their output distributions based on a training set. In our experiments, we show performance improvements on 5 out of 7 classification datasets compared to a set of strong baselines (up to +13\% compared to ICL and LENS). Moreover, we enhance the Pareto frontier of ICL by reducing the inference time needed to achieve the same performance with fewer demonstrations. Finally, MoICL is more robust to out-of-domain (up to +11\%), imbalanced (up to +49\%), or noisy demonstrations (up to +38\%) or can filter these out fr
    
[^132]: 无需费力学注意力，直接“种下”它：通过学习排序预训练注意力

    Don't Pay Attention, PLANT It: Pretraining Attention via Learning-to-Rank

    [https://arxiv.org/abs/2410.23066](https://arxiv.org/abs/2410.23066)

    提出PLANT，一种即插即用策略，通过基于互信息增益的学习排序模型预训练标签注意力来初始化注意力模块，可与大语言模型骨干集成，在极端多标签文本分类任务上超越现有最先进方法，且在罕见标签上收益最大。

    

    最先进的极端多标签文本分类模型依赖多标签注意力机制来聚焦输入文本中的关键标记，但学习良好的注意力权重十分具有挑战性。我们提出了PLANT（Pretrained and Leveraged AtteNTion，预训练并利用的注意力），这是一种即插即用的策略，用于初始化构建在大语言模型输出表示之上的标签注意力模块。PLANT的工作方式是：在端到端微调之前，利用一个基于互信息增益训练的学习排序模型来“种下”（植入）标签特定的注意力。这种与架构无关的方法可与多种大语言模型骨干网络（如Mistral-7B、Llama-3.1-8B、DeepSeek-V3和Phi-3-mini）集成。PLANT在ICD编码、法律主题分类和内容推荐等任务上超越了现有最先进方法，并在医疗领域之外的受控对比实验中改进了微调后的大语言模型。其收益在罕见标签上最为显著，尽管免训练的大语言模型抽取基线仍具有竞争力。

    arXiv:2410.23066v5 Announce Type: replace  Abstract: State-of-the-art Extreme Multi-Label Text Classification models rely on multi-label attention to focus on key tokens in input text, but learning good attention weights is challenging. We introduce PLANT (Pretrained and Leveraged AtteNTion), a plug-and-play strategy for initializing the label-attention module that operates on top of an LLM's output representations. PLANT works by planting label-specific attention using a Learning-to-Rank model trained on mutual information gain, before end-to-end fine-tuning. This architecture-agnostic approach integrates with LLM backbones (e.g., Mistral-7B, Llama-3.1-8B, DeepSeek-V3, and Phi-3-mini). PLANT outperforms SOTA methods across tasks like ICD coding, legal topic classification, and content recommendation, and improves fine-tuned LLMs in controlled comparisons outside the medical domain. Gains are largest on rare labels, although a training-free LLM extraction baseline remains competitive o
    
[^133]: 连贯性提升：当你的预训练语言模型注意力不够时

    Coherence boosting: When your pretrained language model is not paying enough attention

    [https://arxiv.org/abs/2110.08294](https://arxiv.org/abs/2110.08294)

    提出了一种无需额外训练的推理方法“连贯性提升”，通过增强语言模型对长上下文的关注度来改善长距离语义连贯性，并在多种零样本NLP任务上带来性能提升。

    

    长距离语义连贯性仍然是自动语言生成与理解中的一项挑战。我们证明了大型语言模型尚未充分学习到远距离词语对下一个词预测的影响。我们提出了连贯性提升，这是一种推理方法，能够增强语言模型对长上下文的关注。通过对生成的普通文本和对话回复进行分布分析，我们展示了连贯性提升在预训练模型上的优势。我们还发现，将连贯性提升应用于各种零样本NLP任务的最先进模型时，无需额外训练即可带来性能提升。

    arXiv:2110.08294v3 Announce Type: replace  Abstract: Long-range semantic coherence remains a challenge in automatic language generation and understanding. We demonstrate that large language models have insufficiently learned the effect of distant words on next-token prediction. We present coherence boosting, an inference procedure that increases a LM's focus on a long context. We show the benefits of coherence boosting with pretrained models by distributional analyses of generated ordinary text and dialog responses. It is also found that coherence boosting with state-of-the-art models for various zero-shot NLP tasks yields performance gains with no additional training.
    
[^134]: ThinkSum：使用大型语言模型进行集合的概率推理

    ThinkSum: Probabilistic reasoning over sets using large language models. (arXiv:2210.01293v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.01293](http://arxiv.org/abs/2210.01293)

    本研究提出一种两阶段的概率推理范例ThinkSum，通过以结构化的方式对对象或事实集进行推理，实现了对多个对象或事实进行推理并进行逻辑推导的场景中的改进。

    

    大型语言模型（LLMs）在高级类比推理方面具有显著的能力：重现线性文本中出现在它们的训练数据（零样本评估）或提供的上下文中的模式（少量样本在上下文中学习）。然而，最近的研究表明，即使是更先进的LLMs在需要对多个对象或事实进行推理并进行逻辑推导的场景中也会失败。我们提出了一种两阶段的概率推理范例ThinkSum，它以结构化的方式对对象或事实集进行推理。在第一阶段（Think-检索关联）中，LLM在从提示或辅助模型调用提取的短语集上并行查询。在第二阶段（Sum概率推理或推理）中，聚合这些查询的结果以进行最终预测。我们在BIG-bench LLM评估任务套件上展示了ThinkSum的可能性和优势，取得了超越基准的改进。

    Large language models (LLMs) have a substantial capacity for high-level analogical reasoning: reproducing patterns in linear text that occur in their training data (zero-shot evaluation) or in the provided context (few-shot in-context learning). However, recent studies show that even the more advanced LLMs fail in scenarios that require reasoning over multiple objects or facts and making sequences of logical deductions. We propose a two-stage probabilistic inference paradigm, ThinkSum, which reasons over sets of objects or facts in a structured manner. In the first stage (Think - retrieval of associations), a LLM is queried in parallel over a set of phrases extracted from the prompt or an auxiliary model call. In the second stage (Sum probabilistic inference or reasoning), the results of these queries are aggregated to make the final prediction. We demonstrate the possibilities and advantages of ThinkSum on the BIG-bench suite of LLM evaluation tasks, achieving improvements over the 
    

