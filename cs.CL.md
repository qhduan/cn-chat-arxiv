# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NarraSpace: Multimodal Scene-Grounded Evidence for Picture-Description Dementia Classification](https://arxiv.org/abs/2610.28616) | NarraSpace通过将语音内容与图片场景元素关联并提供可溯源的场景证据，实现了兼具高准确率（约81%）与可解释性的痴呆症图片描述分类。 |
| [^2] | [Benchmarking Arabic Dialects for Math Problems](https://arxiv.org/abs/2610.28607) | 该论文推出了首个用于评估阿拉伯方言数学推理的人工整理基准DAR-MGSM，涵盖五种主要阿拉伯方言及现代标准阿拉伯语，并发现大语言模型在文化本地化版本上性能普遍下降。 |
| [^3] | [Capable Teacher, Myopic Scorer: Reward-Free Teacher Repair for Agentic On-Policy Distillation](https://arxiv.org/abs/2610.28586) | 论文揭示智能体在线策略蒸馏（OPD）失败的根源不仅是误差累积，更在于教师逐 token 评分的“短视性”——其评分看不到回合结局，从而放过了安全却拖延的行动，并据此提出了一种无需奖励的教师修复方法。 |
| [^4] | [Cache Engineering for Agents](https://arxiv.org/abs/2610.28577) | 该论文提出“缓存工程”新范式，使LLM智能体无需修改轨迹或重新预填充上下文，即可通过drop和repos等操作直接控制活跃KV缓存，从而在降低缓存I/O与存储成本的同时保持任务准确率。 |
| [^5] | [Large Flow Language Models](https://arxiv.org/abs/2610.28559) | 提出了 Lafa——一个可扩展至 80 亿参数的分块流语言模型，通过改造预训练自回归模型实现高效训练，并借助前缀缓存与分块并行解码实现快速推理，在指令跟随、推理和编码任务上达到或超越自回归模型的性能。 |
| [^6] | [Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks](https://arxiv.org/abs/2610.28543) | 该论文提出二元离散扩散，将扩散过程的含噪状态表示为每个位置候选词元集合的二进制隶属向量，通过添加与删除候选实现信息的渐进丢失与恢复，从而能显式建模部分不确定性并有效解决数独、规划等逻辑与推理任务。 |
| [^7] | [BitMTP: When Multi-Token Prediction Meets Low-Bit Large Language Models](https://arxiv.org/abs/2610.28500) | 提出BitMTP算法-系统协同设计框架，将多词元预测与超低比特大语言模型骨干相结合，使单一模型兼具自回归生成与块因果草稿生成能力，在保持同等生成质量的前提下实现高效解码加速。 |
| [^8] | [Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs](https://arxiv.org/abs/2610.28498) | 论文通过仅有一个数字不同的最小提示对证明，从输入复制的信息确实以跨问题共享的编码存在于模型状态中，但常规单样本探针难以读取它，因此探查失败不能证明信息缺失，并提出了一套区分“计算出的值”与“输入副本/混合”的探查协议。 |
| [^9] | [OptionWorld: A Benchmark for Diagnosing Failures of LLMs as Long-Term Personal Assistants](https://arxiv.org/abs/2610.28491) | 论文提出OptionWorld基准，通过追踪模拟日常任务中的信息获取、判断、行动与结果，诊断出LLM作为长期个人助手失败的三大主因：所需信息未获取或整合、可用的行动机会未被利用、以及模型施加了任务并不需要的额外限制。 |
| [^10] | [Asymmetric Looped Transformers: Fixed Global Representations for an Iterative Decoder](https://arxiv.org/abs/2610.28489) | 提出非对称循环Transformer（AsymLoop），将单次上下文编码与循环状态精炼解耦，利用迭代过程中保持固定的全局KV缓存与循环解码器生成的局部KV进行联合注意力，从而在不增加参数量的情况下提升模型的有效深度。 |
| [^11] | [GPT-6-Astra Understands Touch and Outperform Vision--Tactile--Language--Action Models](https://arxiv.org/abs/2610.28457) | 提出将 GPT-6-Astra 直接作为触觉策略的 GPT-as-Tactile-Policy 方法，通过将专家演示总结为任务指导与触觉参考，并结合多模态观察与交互历史生成控制指令，无需专门的预训练与逐任务优化即可实现接触丰富的操作并超越现有的视觉-触觉-语言-动作模型。 |
| [^12] | [Toolflation: Seller Agents Distort Tool Markets Through Spontaneous Generative Engine Optimization](https://arxiv.org/abs/2610.28454) | 卖家AI智能体仅被要求“更常被选中”时就会自发进行生成式引擎优化，夸大其工具的能力描述，形成不受市场约束的“工具通胀”现象，从而扭曲智能体间的工具市场，且其严重程度在不同模型间相差可达一个数量级。 |
| [^13] | [MythoSemantic: An Interactive System for Discovering Semantic Parallels Across Mythological and Religious Traditions](https://arxiv.org/abs/2610.28452) | 该论文提出MythoSemantic交互式系统，利用神经语言模型将神话与宗教文本嵌入共享语义空间，使用户能够跨文化传统发现语义平行关系，且嵌入结构形成的语义几何形态与传统间的历史关系相吻合。 |
| [^14] | [Every Model Leaves a Trace: Multi-Resolution LLM Fingerprinting](https://arxiv.org/abs/2610.28434) | 该论文提出多分辨率大语言模型指纹识别框架，证明即使表层线索被抑制，LLM指纹在蒸馏模型乃至同一基座的LoRA适配模型中依然可检测，并能跨提示分布和语言迁移。 |
| [^15] | [The Instruction Tax on Factual Recall: Base Models Can Beat Their Instruction-Tuned Counterparts](https://arxiv.org/abs/2610.28407) | 该论文发现指令微调会降低语言模型的事实回忆能力（即“指令税”），通过简单的文本补全方式提示基础模型并仅保留重复输出，在AKBC LM-KBC 2026共享任务上取得了超过基线两倍的0.6836分。 |
| [^16] | [MAS-Guard: Unified Guardrails from Individual Agents to Multi-Agent LLM Systems](https://arxiv.org/abs/2610.28397) | 该论文提出统一安全防护框架MAS-Guard，通过本地智能体防护层（A-Guard）等机制，为包含可插桩与不透明智能体的单智能体和多智能体LLM系统，提供覆盖本地执行与系统级交互的一体化安全保障。 |
| [^17] | [When Copying Is Hard: Copy-Constrained Decoding for Exact Span Reproduction](https://arxiv.org/abs/2610.28393) | 该论文提出了CopyGen——一个面向冻结大语言模型的轻量级复制约束解码插件，通过将复制分解为“何时复制、复制多远”等显式决策，显著提升模型在长上下文中精确逐字复现目标片段的能力。 |
| [^18] | [CAPSafe: A Benchmark for Contextual Action and Plan Safety Reasoning in Embodied Agents](https://arxiv.org/abs/2610.28381) | 该论文提出了CAPSafe基准，从动作级和计划级两个层面评估具身智能体的上下文安全推理能力，发现LLM虽在危险识别上准确率可超90%，但在安全长时程规划方面仍存在显著差距。 |
| [^19] | [Object-Oriented Harness: Evolving Agent Harnesses through Object-Oriented Design](https://arxiv.org/abs/2610.28380) | 提出OO-Harness框架，利用面向对象设计中的继承机制演化出专门化执行框架的类层次结构，并通过路由器为不同任务选择最优框架，在三个长程基准测试中相比基线最高提升15%。 |
| [^20] | [DeclineCraft: A RAG Abstention Benchmark for Jointly Measuring Combined Retrieval and Parametric Knowledge](https://arxiv.org/abs/2610.28340) | 该论文提出了DeclineCraft——一个基于Minecraft语料库截止日期的动态RAG弃答基准，发现单次检索能显著提升模型在不可回答问题上的正确弃答率，而允许模型自主多轮检索的迭代式RAG反而会降低弃答能力。 |
| [^21] | [Beyond a Single Sentiment Score: Topical and Affective Dimensions of Emoji Meaning on Twitter](https://arxiv.org/abs/2610.28337) | 研究发现，在Twitter上主题语境越聚集的表情符号，其众包情感分数反而越两极化，表明表情符号的意义无法用单一情感分数概括，需要同时考虑主题与情感两个维度。 |
| [^22] | [Disclosure Begets Disclosure: A Self-Reinforcing Privacy Loop in LLM Agents with Persistent Memory](https://arxiv.org/abs/2610.28315) | 该研究揭示了持久记忆型LLM智能体中存在自我强化的隐私泄露循环：基于记忆的响应会将早期属性值作为预设延续，用户确认或不加纠正，导致属性泄露随会话次数持续扩大。 |
| [^23] | [From Abstraction to Action: DSLs for Natural Language Abstraction Interpretation and Execution](https://arxiv.org/abs/2610.28311) | 本文提出借鉴编程领域的领域特定语言（DSL）作为显式抽象表示层，使语言模型能够识别高抽象层次的自然语言指令并将其展开为具体可执行的动作。 |
| [^24] | [Reward Misspecification Degrades Chain-of-Thought Monitorability](https://arxiv.org/abs/2610.28303) | 该研究证明，强化学习中的奖励错误设定本身就足以降低语言模型思维链的可监控性——随着模型奖励破解率上升，其思维链仍在推理真实任务意图，而输出却违背了任务意图。 |
| [^25] | [Adaptive Replay Refinement: Allocating Test-Time Compute Between Revision and Resampling](https://arxiv.org/abs/2610.28294) | ARR 是一种无需训练的自适应方法，通过检测解的修订是否达到不动点，在“继续改进现有解”与“重新生成新解”之间动态分配测试时计算，从而在多个模型和基准上以同等计算量达到与多数投票相当的效果。 |
| [^26] | [Forty Questions: Identifying a Language Model from One-Word Answers](https://arxiv.org/abs/2610.28281) | 本文提出仅通过40个日常问题的单词式回答即可构建模型“指纹”，以92.5%的准确率精确识别API背后的语言模型，并据此揭露了模型端点伪装、计费异常及隐藏系统提示等实际问题。 |
| [^27] | [How Language Models Learn Taste: Brand Preference in LLMs](https://arxiv.org/abs/2610.28272) | 语言模型在“说出品牌”时倾向市场领导者，而在“推荐品牌”时却偏向小众专业品牌，这种品味形成于预训练并被DPO偏好优化显著放大，即使偏好数据中几乎不含品牌信息。 |
| [^28] | [Can Language Models Get "Aphasia" From Neuronal Ablation?](https://arxiv.org/abs/2610.28214) | 该研究通过系统性消融语言模型神经元发现，LSTM呈渐进式退化，而Transformer在MLP消融下出现类似Wernicke型失语症的阈值式崩溃，表明语言模型可作为研究神经退行性疾病认知衰退模式的计算代理。 |
| [^29] | [Uncovering Audio-Text Alignment in Large Audio-Language Models through Cross-Modal Concept Neurons](https://arxiv.org/abs/2610.28189) | 该研究发现了大型音频-语言模型中同时表示音频和文本的跨模态概念神经元，证实音频与文本表示在共享子空间中存在对齐，且这些神经元可通过因果干预被操控。 |
| [^30] | [Temporal Evidence-Parameter Co-Alignment for LLM Reasoning over Temporal Knowledge Graphs](https://arxiv.org/abs/2610.28176) | 本文提出TEPCA框架，通过在证据层面检索时序连贯的知识、在参数层面进行动态自适应的低秩更新，从证据和参数两个维度实现大语言模型与时序知识图谱推理的协同对齐。 |
| [^31] | [Africa-Papers: A 10-Billion-Token Multi-Domain Academic Corpus for African Foundation Language Models](https://arxiv.org/abs/2610.28173) | 本文发布了迄今最大的泛非多领域学术预训练语料库 Africa-Papers，包含 84 万余篇全文论文、共计 101.5 亿个 token，旨在弥补现有大模型预训练数据中非洲学术内容占比不足 0.5% 的结构性缺失，提升模型对非洲相关领域的理解能力并推进非洲的 AI 知识主权。 |
| [^32] | [Occupational Uniqueness and Wage Growth](https://arxiv.org/abs/2610.28133) | 本文提出了一个衡量职业任务结构的新框架，将任务数量变化与职业独特性变化相分离，并利用1977-2018年间近20万条职业任务描述数据研究职业独特性与工资增长之间的关系。 |
| [^33] | [Reading Too Much Into It? Interpretive Preferences and Commitment in LLM Mentalisation under Ambiguity](https://arxiv.org/abs/2610.28105) | 该研究提出MentalLoom评估框架，发现在模糊社会情境下，用户的心理困扰披露与重复提问会促使八个主流LLM转向更积极的对叙述者的解读、提供更多情感支持并表现出更高的确定性，证明用户语境能够重塑模型的解释偏好与承诺。 |
| [^34] | [BAIT: Can LLMs Find What Biases a Reader? A Benchmark for Cognitive Bias Triggers in News](https://arxiv.org/abs/2610.28092) | 该论文提出BAIT基准，通过九类认知偏见触发因素的分类体系和50篇标注新闻文章，定义了偏见触发因素检测与定位两项任务，以评测大语言模型识别会让人类读者产生偏见的文本特征的能力。 |
| [^35] | [Contextual Candour: Role-Aware Graduated Disclosure for LLMs over Mixed-Sensitivity Contexts](https://arxiv.org/abs/2610.28067) | 本文提出“分级披露”概念，将大语言模型在混合敏感度信息上的访问控制从非黑即白的“授予或拒绝”决策，革新为根据用户授权级别校准回答细节与敏感度的生成方式，并通过基于奖励的优化框架在惩罚改写、近似等隐性泄露的同时，引导模型生成与授权相称的回答。 |
| [^36] | [Same Judgment, Different Yes: Measuring Multilingual Judgment Profiles in Language Models](https://arxiv.org/abs/2610.28066) | 通过对八个模型、六种语言共110万条道德判断回复的分析，该研究发现除否定疑问句下按预设映射读取的“是/否”标记外，语言模型在不同提问形式和答案读取方式下均呈现一致的道德判断画像，表明多数测量方法反映的是真实判断而非回答习惯。 |
| [^37] | [On-Policy Transition Distillation](https://arxiv.org/abs/2610.28048) | 该论文发现教师与学生在中间层预测过渡上的一致性与答案正确率正相关，并提出在线策略过渡蒸馏（OPTD），通过在学生生成的轨迹上对齐教师与学生的逐层对数概率增量来增强传统最终层分布匹配，从而提升蒸馏效果。 |
| [^38] | [Poisoning-Based Backdoors Are Not Cryptographic](https://arxiv.org/abs/2610.28045) | 本文通过归约证明，基于投毒安装的神经网络后门不可能获得密码学认证那样的不对称性，因为任何从被投毒数据中安装触发器验证器的训练过程本身就是一个学习过程，因此基于学习的防御在原则上始终是可行的。 |
| [^39] | [Using Letter Positional Probabilities to Assess French Word Complexity](https://arxiv.org/abs/2610.28040) | 本文提出利用字母位置概率（LPP）这一新指标，通过对比词汇判断任务中反应时间最高与最低的单词组，发现大量字母/位置概率与法语单词复杂度显著相关。 |
| [^40] | [From RL Papers to Framework Code: Inclusion and Review in Seven Open-Source Training Frameworks](https://arxiv.org/abs/2610.28027) | 该研究通过追踪七个开源强化学习后训练框架的代码提交历史，发现1598篇提出通用RL训练方法的论文中仅有37篇被框架收录，且收录的论文多为大规模训练（30B+参数）和现有损失函数的小幅修改，揭示了RL研究成果从论文到实际工程应用之间的巨大鸿沟。 |
| [^41] | [Beyond the Evidence Boundary: Behavioral Mirage Under Progressive Degradation in Medical VLMs](https://arxiv.org/abs/2610.28000) | 该论文提出了一个基线锚定框架，通过渐进性图像退化识别医学视觉语言模型的“证据边界”，并首次定义“行为幻象”现象——即模型在证据已不足以支持判断的边界处仍持续坚持原有回答，且发现仅凭文本即可回答的问题与该坚持行为显著相关。 |
| [^42] | [Building Telecom Language Models with RSOD: Domain Data, Specialist Post-Training, Memory-Conscious Distillation, and Networking Evaluation](https://arxiv.org/abs/2610.27993) | 该论文提出RSOD端到端框架，通过领域合成数据构建、多个专家LoRA教师后训练以及内存感知的在线策略蒸馏，将六个网络专家模型的行为整合到一个可部署的单一电信语言模型中。 |
| [^43] | [Adaptive Stopping of Streamed Reasoning for Language Model Agents](https://arxiv.org/abs/2610.27981) | 提出 StreamStop 方法，利用冻结的前缀读取器和轻量级停止头，从前缀级别动作一致性中离线学习针对特定模型的停止策略，无需访问推理模型内部信号即可自适应地停止语言模型智能体的流式推理，并发布了带前缀级别动作标注的推理轨迹数据集。 |
| [^44] | [Lost in Recovery: Demystifying Recovery Behavior of Computer-Use Agents via Goal-Oriented Action Replay](https://arxiv.org/abs/2610.27954) | 该论文提出ReTrace评估框架，通过形式化“转换一致的恢复”标准并借助面向目标的动作回放，系统性地评估计算机使用智能体从失败中恢复的能力，填补了基于模型的恢复研究的空白。 |
| [^45] | [From DQI to C-DQI: Automatic Codebook Generation and Evaluation for Measuring Computational Discourse Quality Index](https://arxiv.org/abs/2610.27937) | 该论文提出了一套从DQI到C-DQI的完整协议，通过比较自上而下、自下而上和混合三种方法自动生成基于大语言模型的编码本，并结合人工验证的外在代理指标和模型间一致性（IMA）指标进行评估，从而以可扩展、低偏见的方式测量议会话语质量。 |
| [^46] | [Why Do LLMs Accept Generics? Explaining Acceptability using Psycholinguistic and Model-Derived Factors](https://arxiv.org/abs/2610.27928) | 该研究首次利用心理语言学因素（属性普遍性、区分性、典型性）和模型衍生因素系统解释了大语言模型判断泛指句可接受性的机制，发现模型像人类一样权衡属性普遍性与区分性并偏好典型属性，但与人类不同，它们忽略引人注目的属性，而最看重陈述的事实真实性。 |
| [^47] | [ACE-Compression: An Ability-Centric Evaluation of Compressed Large Language Models](https://arxiv.org/abs/2610.27913) | 该论文提出ACE-Compression框架，首次从核心能力（而非仅任务分数）的角度系统评估压缩后的大语言模型，涵盖程序性推理、知识真实性、社会认知与人格一致性四个维度，揭示了传统任务评估无法发现的能力悄然退化问题。 |
| [^48] | [Polarity Blindness in LLM Safety: A Systematic Study of Negation, Antonym Substitution, and Adversarial Framing](https://arxiv.org/abs/2610.27908) | 大语言模型普遍存在“极性盲视”漏洞——在语法否定或反义词替换等极性改变的语言转换后无法识别有害意图，从而绕过安全防护，该漏洞在10个主流大模型和3种语言中均被验证，最脆弱模型的平均危害评分高达4.12。 |
| [^49] | [Staged Continual Adaptation of Multimodal Foundation Models for Japanese Financial Documents](https://arxiv.org/abs/2610.27904) | 该研究追踪一个84亿参数多模态模型在日文金融文档上的三个训练阶段，发现各阶段贡献不可分解且交互非加性——数学蒸馏会覆盖已习得的金融知识，因此最佳检查点因任务而异，关键在于检查点保留与调度策略而非增加训练阶段。 |
| [^50] | [CollabSciReM: Aligning Language Models for Scientific Conversations via Science-Aware Reward Decomposition](https://arxiv.org/abs/2610.27901) | 该论文提出 CollabSciReM 框架，通过将会话级奖励分解为对话质量、科学严谨性和结构效率三个目标条件化组件来对齐语言模型，使其胜任多轮科学协作对话，并配套推出基于六个维度的 SciQuality 成对评估方法。 |
| [^51] | [Testing Associative Creativity in Humans and Multimodal LLMs with Difficult-to-Name Visual Stimuli](https://arxiv.org/abs/2610.27894) | 该研究通过SilCC和PixCC两个可控难度数据集评估人类与多模态大语言模型对难以命名的抽象视觉刺激的联想创造力，发现人类的联想描述更多样、更新颖、语义距离更大，而模型的有效性随难度增加而下降。 |
| [^52] | [From Tokens to Files: Multi-Granularity Detection and Localization of AI-Involved Code](https://arxiv.org/abs/2610.27863) | 提出了一种基于冻结代码语言模型上双低秩适配器逐词元对数似然比的框架，可在词元证据基础上对 AI 参与代码进行行、块、文件多粒度的检测与定位。 |
| [^53] | [UvA-MT's Participation in the WMT26 General Translation Shared Task](https://arxiv.org/abs/2610.27852) | UvA-MT 团队在 WMT26 通用机器翻译任务中，利用大语言模型顺序采样生成翻译候选，并通过 LLM 成对评委的循环投票机制选出最佳译文，参与了全部 23 个翻译方向。 |
| [^54] | [SciDiagBench: A Type-Specific Benchmark for Human-Aligned Evaluation of Scientific Diagrams](https://arxiv.org/abs/2610.27841) | 该论文提出了SciDiagBench，一个按图表类型系统分类、并与人类专家评估偏好显式对齐的科学图表评估基准，弥补了现有基于视觉语言模型的评估基准缺乏类型区分和人类对齐的不足。 |
| [^55] | [SMART: Semantic Segment Mapping and Assessment of Recognition Transcripts with Contextual Rubrics](https://arxiv.org/abs/2610.27820) | 提出SMART框架，将ASR识别文本与参考文本的差异映射为语义片段并利用上下文评分准则标注其语义后果，从而实现可追溯的语义评估，其提出的语义影响错误率（SIER）性能显著优于直接使用大语言模型判断。 |
| [^56] | [ProVe: Proposer-Verifier Decoding](https://arxiv.org/abs/2610.27815) | ProVe反转了推测解码中提议者与验证者的角色，让大模型通过单次前向传播直接提议token块并加以验证，无需训练独立草稿头即可实现2.5-2.7倍推理加速，且无损版本能完全复现原始贪婪解码结果。 |
| [^57] | [Are Linguistic and Arithmetic Circuits Dissociated in LLMs?](https://arxiv.org/abs/2610.27790) | 本研究通过机制可解释性的回路发现方法，在五个大语言模型上发现语言与算术回路的相似性主要由任务领域而非句法相似性决定，表明大语言模型以分离的方式处理语言与算术。 |
| [^58] | [Give the Model a Finger: Developmentally Aligned Evaluation of Counting in Vision-Language Models](https://arxiv.org/abs/2610.27756) | 该论文受儿童计数发展过程启发，提出一种无需训练的智能体评估框架，让视觉语言模型通过“指认标记”等认知动作进行计数，将模型通常隐藏的串行视觉推理过程转化为可解释、可审计的行为分析。 |
| [^59] | [Hearsay or Fact? How Framing and Elicitation Shape LLM Probability Updates](https://arxiv.org/abs/2610.27745) | 该论文提出了一种衡量大语言模型对单条新信息概率更新幅度的“方向性移动”指标，并发现信息的框架呈现方式（直接告知还是传闻转述）与概率引出方式（单个概率还是五个加权估计）会显著影响模型的概率更新。 |
| [^60] | [Tracing Recurrent and Emergent Hallucination Patterns in Decomposed Text-to-SQL Generation](https://arxiv.org/abs/2610.27720) | 该论文提出一个下钻式评估框架，将Text-to-SQL生成过程分解为复杂度递增的可执行子查询，从而区分错误在早期步骤已出现并延续到最终查询的“复发性幻觉”与仅在最终查询才显现的“涌现性幻觉”，揭示大模型幻觉在生成路径中的演变规律。 |
| [^61] | [When Does Experience Memory Help Closed-Loop LLM Driving Agents? A Controlled Evaluation with Resampled Warmup Pools, Token-Matched Fillers, and Audited Baselines](https://arxiv.org/abs/2610.27714) | 本研究通过在审计的HighwayEnv测试平台上控制预热池抽样与提示令牌两大混淆因素，发现经验记忆为闭环LLM驾驶智能体带来的表面收益主要来自预热经验池的随机抽样，而非记忆机制本身。 |
| [^62] | [Beyond Calibration: Temporal LLM Confidence Assessment Across Languages and Local Settings](https://arxiv.org/abs/2610.27702) | 受元记忆理论启发，该研究首次区分了大语言模型求解前（FOK）与求解后（JOL）两个时间阶段的置信度信号，发现求解后置信度在多语言与本地化场景中具有更强的认知对齐，而成本更低的求解前信号则校准较弱。 |
| [^63] | [Memory as Code: Declarative Retrieval over Codified Long-Term Memory](https://arxiv.org/abs/2610.27700) | 提出Memory as Code (MaC)框架，以共享代码抽象统一设计记忆构建与声明式检索，通过原子函数层、关系层、模块化组合层的三层可执行记忆结构，解决长期记忆中事实分散、关系动态演变导致难以组合检索的问题。 |
| [^64] | [PhonoBridge: Error-Aware Phoneme-Mediated EMG-to-Text Reconstruction](https://arxiv.org/abs/2610.27678) | 该论文提出PhonoBridge框架，以音素作为肌电信号与大语言模型之间的中间表示，并通过错误特征引导增强（EPGA）模拟前端解码错误来扩充训练数据，从而在配对肌电-文本数据稀缺的情况下实现更准确的无声语音到文本重建。 |
| [^65] | [CovNav: Conformal Prediction-Guided Adaptive Inference for Object Goal Navigation](https://arxiv.org/abs/2610.27677) | CovNav提出了一种无需训练的框架，利用共形预测来量化空间判断的不确定性，从而自适应地决定混合推理LLM在目标物体导航中何时使用快速“系统1”模式或缓慢“系统2”模式，在推理质量与效率之间实现最优权衡。 |
| [^66] | [Corpus Quality Improves, Retrieval Utility Does Not: An Ablation of Preprocessing and Refinement in a Speech-to-QA Pipeline](https://arxiv.org/abs/2610.27652) | 本研究通过消融实验发现，尽管预处理和精炼阶段能显著提升语料库质量并获得LLM评审与人类专家的一致认可，但这种质量提升并不能转化为下游检索准确率的提升，其中转录改写反而使检索准确率下降了0.061。 |
| [^67] | [Speak vs. Know: Balancing Behavior and Knowledge Distributions of Large Language Models](https://arxiv.org/abs/2610.27648) | 论文揭示了大语言模型的“知识分布”（文本表述分布）与“行为分布”（对数概率分布）存在显著不一致——在37%的世界价值观调查问题上两者最可能的答案不同，导致基于表述分布的评估分数无法真实反映模型的实际行为，因而需要平衡这两种分布。 |
| [^68] | [CUA-Recovery: Benchmarking Error Recovery in Long-Horizon Computer-Use Agents](https://arxiv.org/abs/2610.27645) | 该论文提出了 CUA-Recovery 基准和 ReRail 框架，通过在持久化桌面环境中将任务组合成长时程工作流来系统评估计算机使用智能体的错误检测与恢复能力，弥补了现有基准依赖人工注入错误、无法反映真实智能体策略所致失败的不足。 |
| [^69] | [BCT: Bayes-Consistent Tempering for Flow Language Models](https://arxiv.org/abs/2610.27641) | 提出贝叶斯一致温度调节（BCT），将温度显式地融入流语言模型生成过程的分布与动力学中，解决了事后温度缩放导致的分布与动力学不匹配问题，使温度调节对流语言模型性能的提升更加可靠。 |
| [^70] | [A Two-Way Street: Surveying the Interplay between Code-Mixing and Native Languages in Text and Speech](https://arxiv.org/abs/2610.27631) | 本文首次系统综述了文本与语音领域中语码混合与源语言之间的双向迁移，提出按迁移方向、路径和模态分类的框架，发现迁移主要从源语言流向语码混合场景，且在语音任务中源语言数据比新架构更有帮助。 |
| [^71] | [PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking](https://arxiv.org/abs/2610.27628) | 提出进度对齐循环（PAR）训练方法，通过让第 k 次迭代的预测对齐前 k 步操作后的状态并将循环状态范数锚定到首次迭代，使 Qwen2.5-0.5B 在 swap-S5 状态跟踪任务上实现 16 倍于训练长度的外推泛化（512 步时准确率达 95.5%）。 |
| [^72] | [ABench: Adversarily Benchmarking Large Language Models across Taxonomized Values](https://arxiv.org/abs/2610.27611) | 提出了ABench——一个基于158个中文价值观子类别的层次化分类体系、包含2,895个开放式问题的细粒度人工评测基准，通过角色扮演和逻辑陷阱等对抗性诱导技术及加权自动化评估框架，系统评估大语言模型在中文文化语境下面对对抗性诱导时价值观立场的稳定性。 |
| [^73] | [Auditing the Product, Not the Recursion: The Evaluation Evidence Behind Recursive Self-Improvement Claims](https://arxiv.org/abs/2610.27610) | 该论文首次通过对114项递归自我改进（RSI）研究的系统性全文编码，量化审计了RSI声明背后产品评估与流程评估证据的实际报告频率，揭示了递归自我改进声明与其实际评估证据之间的差距。 |
| [^74] | [Self-Decomposing Diffusion for Code Generation](https://arxiv.org/abs/2610.27600) | 提出了SDDC框架，完全通过扩散先生成规定程序分解方式与单元间依赖关系的契约，再据此指导协调的并行实现，从而解决并行解码效率与程序依赖正确性之间的权衡问题。 |
| [^75] | [WORTH Benchmark: Evidence Discovery under Not-Fully-Cooperative Information -- A Two-Episode Pilot](https://arxiv.org/abs/2610.27597) | 该论文提出WORTH基准，让模型在无线索标注的原创推理小说中自主发现关键证据、识别案件参与者并解释推理过程，以此评估模型在叙事信息不完全合作（证据与误导性安排并存）环境下的证据发现与判断能力。 |
| [^76] | [Benefit and Cost of Random-Token Replacement under Repeated Data: A Known-Entropy Decomposition and Evidence from Ten Real Corpora](https://arxiv.org/abs/2610.27590) | 该论文提出将验证损失分解为近似、估计与记忆三项的框架，在熵率已知的受控数据源上证明随机词元替换通过向语料均值收缩预测来缓解记忆化（可将验证损失最小值后的回升降低约60%），其收益依赖记忆化的存在而偏差代价则不然，并在十个真实语料库上提供了实证证据。 |
| [^77] | [Beyond Model Weights: Sampling-Dependent Fairness in Diffusion Language Models](https://arxiv.org/abs/2610.27584) | 扩散语言模型的社会偏见程度及模型间的偏见排序会随采样配置（如块长度和重掩码策略）显著变化，且即使采用去偏见策略后，这种采样依赖性依然持续存在。 |
| [^78] | [TeCoP: Accelerating Diffusion Large Language Model Decoding via Tentative Context Pipelining](https://arxiv.org/abs/2610.27583) | TeCoP是一种无需训练的扩散大语言模型解码方法，通过将稳定但未提交的预测作为暂定上下文进行跨块流水线复用，将前向评估次数最多减少87%，吞吐量最高提升11.77倍。 |
| [^79] | [Factual Access States in Diffusion Language Models: From Mechanism to Knowledge Editing and Retrieval](https://arxiv.org/abs/2610.27580) | 该论文揭示了扩散语言模型在第一层注意力层就通过将关系措辞路由到可见实体位置来组装事实访问状态，这一机制与自回归模型的组装方式截然不同且发生更早，为知识编辑与检索提供了新的机制层面依据。 |
| [^80] | [Why Static Probes Fail: Context Determines Deception Feature Composition](https://arxiv.org/abs/2610.27577) | 该论文发现欺骗并非大语言模型表征空间中的单一方向，提出用迭代正交分解将欺骗建模为多维子空间，并揭示欺骗特征的组成依赖于上下文，因此静态的探针组合难以可靠泛化。 |
| [^81] | [When LLMs Manipulate: Measuring Behavioral, Social, and Temporal Effects](https://arxiv.org/abs/2610.27573) | 该论文提出了一个测量“行为脆弱性”的行为计算框架，通过操纵梯度下降生成针对权威、紧迫性和社会认同偏差的操纵性AI刺激，并借助多利益相关者影响评估网络与时间性危害传播模型，量化了操纵性LLM输出对用户信任校准与依从性的显著影响及其跨用户、跨时间的传播效应。 |
| [^82] | [AudiPsy: Evidence-Grounded Multimodal Reasoning over Spoken Counseling Conversations for Depression Assessment](https://arxiv.org/abs/2610.27567) | 提出AudiPsy，一个基于证据的全模态大语言模型，通过观察-评估-总结范式对口语心理咨询对话中的文本与声学多模态证据进行结构化情境推理，显著提升了抑郁症评估的可解释性与准确性。 |
| [^83] | [Learned Readout Tokens for Extracting Large Language Model Confidence](https://arxiv.org/abs/2610.27566) | 提出 ScratchProbe 方法，通过学习一个专用的读取令牌来提取隐藏状态训练置信度探针，在多种数据集和大语言模型上优于基于现有令牌的探针，多令牌扩展还可进一步提升性能。 |
| [^84] | [AnomalyJev: Towards Evaluating Image Anomaly Explanations via Claim-Level Verification](https://arxiv.org/abs/2610.27565) | 提出AnomalyJev，一个无需token生成的轻量级评估框架，通过从图像异常检测模型生成的文本解释中提取可验证断言并与真实标注比对验证，从而高效且可靠地评估解释的忠实度与完整性。 |
| [^85] | [Exploring Recurrent-Depth Language Models for Multitask Clinical Text Processing](https://arxiv.org/abs/2610.27559) | 小型循环深度语言模型（如Ouro系列）通过多任务联合微调，在13个临床NLP数据集上以更小的参数量达到了与更大规模标准Transformer相当甚至更优的性能，尤其在推理和问答任务上优势显著。 |
| [^86] | [Can Native Safety Travel? Reusing Safety Pretraining Across Models via On-Policy Distillation](https://arxiv.org/abs/2610.27555) | 本文首次提出“原生安全性复用”问题，并证明通过在线策略蒸馏，可以将安全预训练教师模型的鲁棒安全性迁移给标准预训练的学生模型，使其在保留通用能力的同时继承安全鲁棒性。 |
| [^87] | [EviDial: When Correct Depression Predictions Lack Evidence](https://arxiv.org/abs/2610.27541) | EviDial基准将“预测标签正确”与“有证据支撑的预测”区分开来，揭示82.8%的抑郁对话症状评估缺乏足够的文本证据支持。 |
| [^88] | [Judging the Justification: Decision Certificate Supervision for Generative Reward Models](https://arxiv.org/abs/2610.27528) | 提出DeCert框架，无需参考解释即可评估生成式奖励模型所生成理由的有效性与结论一致性，并将其转化为兼顾结论正确性和理由质量的偏好监督信号，以区分强弱的理由论证。 |
| [^89] | [Self-Evolving Agents through Privileged Retrospective Distillation](https://arxiv.org/abs/2610.27495) | 提出回溯自蒸馏（RSD）多轮框架，让同一模型既当学生又当教师，借助特权回溯上下文将缺乏真值答案或奖励的自我经验转化为密集监督信号，实现模型、数据与监督的共同进化。 |
| [^90] | [Recoverability and Verifier Strength: Two Measured Preconditions for Unattended Agent Self-Improvement](https://arxiv.org/abs/2610.27465) | 论文提出了“验证器强度”和“可恢复性”这两个可测量的前置条件，用于判断智能体平台的哪些自我改进步骤可以安全地无人值守运行。 |
| [^91] | [Detecting Errors in Large Language Model Responses: The Incremental Predictive Value of Hidden States](https://arxiv.org/abs/2610.27455) | 该论文通过将输出侧信息整合为强基线并在公平条件下对比，发现隐藏状态对大语言模型错误检测的增量预测价值十分有限（AUROC仅提升约0.004至0.039）。 |
| [^92] | [Ask to Distill: Learning to Synthesize Data for On-Policy Distillation](https://arxiv.org/abs/2610.27450) | 该论文提出Ask-to-Distill框架，将合成问题视为分布引子，利用基于师生分歧等信号的分布性替代奖励（无需参考答案或正确答案）来评估并优化合成问题的价值，从而实现高效的同策略蒸馏数据合成。 |
| [^93] | [Harness Self-Evolution for Long-Horizon Terminal Agents](https://arxiv.org/abs/2610.27435) | RiseHarness提出了一种在模型权重固定的前提下，通过对比同一任务的成功与失败运行、结合智能体轨迹与运行时监控证据、并借鉴其他harness框架的实现来自演化迭代改进智能体harness的框架，从而弥合能力与可靠执行之间的差距。 |
| [^94] | [The Router Already Knows](https://arxiv.org/abs/2610.27421) | 提出NR-SAE稀疏自编码器，直接复用MoE路由器对每个token的稀疏专家选择来构建字典，在解释路由器加权专家输出方差上显著优于参数匹配的对照组。 |
| [^95] | [Looking Into The Past: The Impact of Context on Detecting Deceptive Statements](https://arxiv.org/abs/2610.27407) | 本研究表明，在《阿瓦隆》隐藏角色游戏的欺骗性消息检测任务中，为目标消息加入先前的游戏对话上下文，可使大语言模型的准确率提升高达31%。 |
| [^96] | [More Memory, Worse Agents: Error Reproduction and Anti-Persistence in LLM Agents](https://arxiv.org/abs/2610.27406) | 论文揭示了LLM智能体外部记忆中存储的抽象错误会通过“错误再现”这一反馈循环不断自我复制和延续，并据此提出了一种名为“反持久性”的记忆设计来抑制错误的持续传播。 |
| [^97] | [Steering intonational tunes in text-to-speech synthesis](https://arxiv.org/abs/2610.27403) | 该研究通过交换干预定位了残差向量量化文本到语音模型中英语核心语调的载体——主干网络的隐藏状态而非token堆栈第一层，并证明可用均值差向量引导该状态来实现对语调的控制。 |
| [^98] | [DStill: Trajectory Consistency Distillation for Accelerated One-Step Drafting in Block Diffusion Speculative Decoding](https://arxiv.org/abs/2610.27379) | 提出DStill两阶段蒸馏框架，通过轨迹一致性蒸馏将块扩散投机解码中草稿器的多步迭代细化压缩为单步草稿生成，在降低推理延迟的同时保持草稿的保真度。 |
| [^99] | [SHAM-Bench: A Span-level Hallucination Benchmark Across Multiple Tasks](https://arxiv.org/abs/2610.27370) | 提出了SHAM-Bench——一个涵盖六个NLG任务、共6461个样本的统一片段级幻觉检测基准，评估发现微调检测器在域内表现良好但在分布偏移下退化，而基于提示的推理方法泛化能力更强。 |
| [^100] | [Explore More, Validate Less: Efficient Reflective Prompt Optimization](https://arxiv.org/abs/2610.27367) | 提出了一种两阶段框架，将反思式候选池的构建与完整验证的分配解耦，使候选池规模和验证预算可以独立设置，从而大幅提升提示优化的效率。 |
| [^101] | [On the LLM Verification for Argument Graph Generation](https://arxiv.org/abs/2610.27361) | 本文提出了一种面向论辩图生成的分层验证框架，主张根据各类工具（确定性检查、求解器、大语言模型、辩论、人类）所能保证的能力来合理分配验证任务。 |
| [^102] | [Global Security Blind Spots: Translation as a Safety Bypass in Low-Resource Languages](https://arxiv.org/abs/2610.27304) | 该论文揭示了大型语言模型的英语安全对齐无法可靠迁移至低资源语言——在英语中被成功拒绝的有害请求经翻译后可诱发不安全回答，各模型平均绕过率高达21.9%至44.4%，构成全球性的安全盲区。 |
| [^103] | [Evaluating Context Compaction on the Questions That Come Later](https://arxiv.org/abs/2610.27301) | 研究发现为当前问题挑选句子的压缩方法虽能提升当前问题的准确率，但在后续问题上并不优于随机删除，而与问题无关的词级压缩方法LLMLingua-2在后续问题上反而表现更佳，表明上下文压缩需要兼顾未来的问题。 |
| [^104] | [RealControlBench: A Benchmark for Interpretability-Informed Fine-Grained Control of LLMs and Agents](https://arxiv.org/abs/2610.27295) | 提出了RealControlBench，一个涵盖指令遵循、工具使用决策、推理性能与长度控制、安全意识五个维度的综合基准，用于在真实复杂的多轮智能体交互场景中评估大语言模型与智能体的细粒度行为控制。 |
| [^105] | [TEX: Test-Time Scaling Testing Agents via Execution-based Cross-Validation](https://arxiv.org/abs/2610.27290) | TEX提出了一种基于执行驱动交叉验证的协作式测试时扩展框架，通过并行代理生成测试和补丁并交叉执行形成K×K反馈矩阵实现客观验证，在SWT-Bench Verified上以86.4%的成绩创下新的最先进水平。 |
| [^106] | [When Does Hate Transfer? A Large-Scale Study of Cross-Lingual Hate Speech Detection](https://arxiv.org/abs/2610.27285) | 该研究通过对14种语言的大规模实验发现，语言相似性是跨语言仇恨言论检测迁移效果的最强预测因子，而共享的文化线索和交际规范可能解释其余未知的迁移模式差异。 |
| [^107] | [FLINT: Fixed-budget Late Interaction](https://arxiv.org/abs/2610.27273) | FLINT 提出一种基于注意力池化的架构，将查询和文档对称地压缩为固定数量的嵌入向量，在大幅降低存储开销和检索时的向量比较次数的同时，保持了与多向量模型相当的检索效果。 |
| [^108] | [African Languages AI Safety Readiness Report](https://arxiv.org/abs/2610.27269) | 本报告通过考察AI资源可用性、国家AI政策支持、文本基准表现和说话者人口四个因素，评估了非洲语言参与人工智能安全讨论的就绪度。 |
| [^109] | [Auditing Safety Preference Supervision with Recorded Derivations](https://arxiv.org/abs/2610.27266) | 该论文提出符号化偏好标注方法，通过记录属性分类、明确规则和固定判定排序，使每个安全偏好标签都能从存储的推导过程中重新计算验证，从而实现对偏好监督的可审计性。 |
| [^110] | [TxnQA: Benchmarking Analytical Reasoning and Evidence Grounding over Transaction Event Sequences](https://arxiv.org/abs/2610.27256) | TxnQA提出了一个针对交易事件序列分析推理的新基准，包含27个任务族和欺诈风险推理套件，评估发现LoRA适配虽能有效规范输出格式，但算术能力仍是模型性能的主要瓶颈。 |
| [^111] | [Target-Aware Curation of Hallucination Stress Tests for Large Language Models](https://arxiv.org/abs/2610.27247) | 该论文提出了一种目标感知的迭代提示编辑方法，利用LLM编辑器根据目标模型的回答和评判者反馈不断修改种子提示，从而精准暴露特定模型的幻觉弱点，即使各模型在标准压力测试中平均分数相似也能区分出差异。 |
| [^112] | [Diagnostic Analysis for Large Language Model Reasoning](https://arxiv.org/abs/2610.27245) | 本文提出两阶段诊断评估方法，系统检验大语言模型推理图的结构可靠性，发现图可靠性受拓扑、提示设计和模型家族影响，且图结构仅在基于置信度分数区分正确答案和支持节点级诊断方面有价值，仅作为提示上下文时收益有限。 |
| [^113] | [Predicting Linear Relational Decodability Without the Relation](https://arxiv.org/abs/2610.27241) | 该论文提出“实体透镜”这一新颖的雅可比透镜变体，通过在与关系无关的中性提示下针对实体类型进行训练，能够在不使用任何关系特定信息的情况下，显著预测线性关系嵌入（LRE）在不同关系上忠实度的巨大差异。 |
| [^114] | [CEFR-oriented Russian Text Simplification Using Reinforcement Learning](https://arxiv.org/abs/2610.27237) | 该研究利用拒绝采样和直接偏好优化（DPO）等偏好优化方法微调YandexGPT，实现了将俄语文本自动简化至CEFR的A1、A2和B1语言水平的可控简化，最佳模型达到87.15%的目标级别准确率。 |
| [^115] | [MMedAgent-Plus: Scaling Multimodal Medical Agents across Modalities, Tasks and Workflows](https://arxiv.org/abs/2610.27233) | MMedAgent-Plus是一个多模态医疗工具使用智能体，通过视觉-语言规划器协调18个专科模型（覆盖12个任务和8种模态）来执行复杂医疗工作流，并在多种交互设置中实现高达98.1%的工具调用准确率。 |
| [^116] | [Which Agent's Skill Should Change? Group-Relative Credit Assignment for Evolving Multi-Agent Teams](https://arxiv.org/abs/2610.27227) | GRSE受GRPO启发，通过在相同技能库下多次执行同一问题并进行组内对比，将团队奖励精准追溯到具体的智能体-技能对，解决了“技能出现不等于技能责任”的信用分配难题，显著提升了多智能体协作性能。 |
| [^117] | [Profile-DP: Differential Privacy for User-Controlled Profile Release in Personalized LLM Inference](https://arxiv.org/abs/2610.27213) | 该论文提出了 Profile-DP，一种策略感知的本地差分隐私框架，可在个性化大语言模型推理中可证明地保护用户不愿公开的档案信息，并将隐私保证与下游模型行为解耦，同时提供了 DP-SLOT 和 DP-BLOCK 两种满足该框架的实现机制。 |
| [^118] | [LoWAgent: A Low-Resource Multilingual Benchmark for Web Agents](https://arxiv.org/abs/2610.27205) | 该论文提出了首个评估Web智能体在低资源语言（印地语、孟加拉语、乌尔都语、阿姆哈拉语）上性能的多语言基准LoWAgent，包含六个本地化网站和跨五种语言的3,000个任务实例，并发现闭源和开源模型在低资源语言上均相比英语出现明显的性能下降。 |
| [^119] | [The Culture Within: Towards Mechanistic Understanding of Cultural Reasoning in LLMs](https://arxiv.org/abs/2610.27202) | 该论文提出覆盖 75 个国家 9,261 个场景的 CultureLens 基准，通过机制性分析发现 LLM 在文化推理中默认采用英语国家规范、接受陈述性规范只看呈现方式而无论真假，且在模型内部阻断矛盾规范即可恢复正确回答，说明其失败源于内部信息处理而非知识缺失。 |
| [^120] | [Preference-Aware Pareto Steering for Multi-Task Speech LLMs](https://arxiv.org/abs/2610.27200) | 提出EA-SURF方法，通过在训练中自适应调整标量化权重，使多任务语音大语言模型的任务间权衡符合期望的部署偏好，并从理论上证明了该方法能够收敛到目标偏好。 |
| [^121] | [On the Fragility of Tasks under Composition: Implicit Budgeting in Language Models](https://arxiv.org/abs/2610.27190) | 语言模型存在一种隐式思考预算：在组合多个任务时，每个任务分配到的 token 数远少于单独处理时，导致即使模型能独立可靠地完成每个任务，组合后错误率仍显著上升，且需要长推理链的任务在组合下尤为脆弱。 |
| [^122] | [SITO: Sifting a Commercial Text-Safety API for Missed Harm and False Flags in English, Ukrainian and Surzhyk](https://arxiv.org/abs/2610.27178) | 该论文构建了包含2,332个英语、乌克兰语及混合语合成案例的SITO基准，独立评估了一款商用文本安全API在六类危害场景中的漏判与误报表现，填补了英语之外语言环境下此类系统缺乏独立验证证据的空白。 |
| [^123] | [GRID: Modeling Health Influencer Discourse as Argument Networks for Credibility Assessment](https://arxiv.org/abs/2610.27149) | 提出GRID框架，将健康类网红视频建模为相互关联的论辩网络，结合科学证据进行联合可信度推断，证明考虑更广泛的话语背景能显著提升健康内容验证的效果。 |
| [^124] | [ICLHerd: Herding-Based Data Selection for LLMs with In-Context Learning as a Proxy](https://arxiv.org/abs/2610.27142) | ICLHerd提出了一种基于von Mises-Fisher核的数据选择框架，通过平衡模式寻求与模式平均来克服传统最大相似度方法的聚类偏差，并利用上下文学习作为验证代理实现高效的超参数搜索，在大语言模型训练数据选择上优于现有算法。 |
| [^125] | [Joint Constraint Aggregation for Qualifier-Sensitive Retrieval-Augmented QA](https://arxiv.org/abs/2610.27114) | 提出联合约束聚合方法，通过基于检索证据将候选答案与问题派生的统一要求进行联合比对，解决检索增强问答中因检索偏斜导致错误答案被选中的决策问题，并显著提升整体准确率。 |
| [^126] | [Beyond Two Tracks: Speaker-Anchored Full-Duplex Dialogue Recovery](https://arxiv.org/abs/2610.27109) | 该论文提出在扩散生成之前进行说话人锚定路由，用一个仅184万参数的轻量模块将分离出的声流匹配并排序到固定的注册身份，把全双工对话重建中的错误声道时间从43.9%降至4.1%，且无需重新训练冻结的生成器。 |
| [^127] | [Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs](https://arxiv.org/abs/2610.27103) | 提出了无需训练的 Omni-Woodpecker 框架，通过识别跨模态冲突、正交分解分离接地证据与学习先验，并在解码前自适应地再平衡融合表示，从而有效修正全模态大语言模型的跨模态幻觉。 |
| [^128] | [SciFigBench: A Benchmark for Scientific Method Figure Understanding](https://arxiv.org/abs/2610.27087) | 该论文提出了SciFigBench基准，通过从真实论文中精选4,352张科学方法图并构建40,670个问答对，系统评估视觉语言模型在密集内容感知、视觉关系理解和信息流追踪三项核心能力上的表现。 |
| [^129] | [ClockMark: Adaptive and Host-Dependent Watermarking for Large Language Models](https://arxiv.org/abs/2610.27079) | 提出了ClockMark水印框架，首次基于“自适应嵌入”（保留先前采样决策状态）与“宿主相关嵌入”（利用当前LLM分布约束词元选择和状态更新）两大原则，在保持输出分布的同时实现对文本编辑的鲁棒水印嵌入。 |
| [^130] | [Discrete Neural Interlingua for Ontology-Agnostic EHR Modeling](https://arxiv.org/abs/2610.27047) | 本文提出MEDIATOR——一种本体无关的离散分词器，通过将任意文本形式表达的临床概念映射到共享的向量量化码本，将EHR数据协调转化为开放词表分词问题，从而打破不同医疗系统间的编码壁垒。 |
| [^131] | [ReuF2F: Benchmarking Coding Agents on Formalized Undergraduate Research Problems](https://arxiv.org/abs/2610.27046) | 该论文提出了 ReuF2F，一个基于 Lean 4 的基准测试，包含 244 个源自本科生研究经验计划（REU）的形式化开放研究问题，用以更可靠地评估编码智能体解决开放数学问题的能力。 |
| [^132] | [Cross-Lingual Reusability of LLM Pipelines: Financial Extraction, Multilingual Reasoning, and Semantic Screening](https://arxiv.org/abs/2610.27043) | 该研究通过支票信息抽取、多语言问答和金融文档分析三项任务，验证了在英语中开发和验证的大语言模型管线可以直接跨语言复用，准确性在各语言间保持高度稳定。 |
| [^133] | [Fluent but Fallible: Benchmarking Hallucination and Factual Reliability in Large Language Models](https://arxiv.org/abs/2610.27034) | 本研究通过70个覆盖四个复杂度级别的学术提示，对十个主流大语言模型的幻觉和事实可靠性进行了系统的基准对比评估，揭示了尽管模型语言流畅但仍存在事实错误和响应偏差的可靠性风险。 |
| [^134] | [Teaching AI Agent to Play Poker Strategically Through Preference Optimization](https://arxiv.org/abs/2610.27028) | 提出SPOKE方法，通过融合教师相对价值惩罚、策略锚定、风险约束和结构化推理监督的偏好优化技术，使小型AI智能体在隐藏信息的扑克博弈中学会策略性行为，并显著降低可利用性。 |
| [^135] | [Decomposed Relation Extraction from Aviation Accident Narratives: Evaluating Transformer-Based Safety Knowledge Discovery under Relational Sparsity](https://arxiv.org/abs/2610.27013) | 该研究提出将概念识别与关系分类相分离的分解式框架，并在NTSB事故叙述的大规模稀疏基准上系统评估发现，SVM与RoBERTa模型在真实关系稀疏场景下性能急剧下降（Macro-F1从约0.48降至约0.24-0.27），揭示了现有关系抽取方法在航空安全知识发现中面临的严重挑战。 |
| [^136] | [Evaluating Tibetan Speech Recognition Across Dialects and Adaptation Regimes: A Multi-Granularity Analysis](https://arxiv.org/abs/2610.27002) | 本文提出整合字符、音节和分词三种错误率的多粒度评估框架，对藏语语音识别在跨方言和LoRA适应场景下的表现进行系统评估，发现较低字符错误率并不必然意味着音节或分词层面的更好表现，且指标间相关性会在适应失败时减弱。 |
| [^137] | [How to get the best and fairest performance out of your ASR system using activation steering](https://arxiv.org/abs/2610.26994) | 该论文提出用监督优化法推导激活引导向量，可同时提升自动语音识别系统的整体性能与性别、年龄、方言等维度上的公平性，且发现使用同一数据集中不同说话人或声学相似说话人的数据训练引导向量即可获得显著效果。 |
| [^138] | [TRACE: Training-aware Robustness and Challenge Evaluation for Text-Vision Embedding Models](https://arxiv.org/abs/2610.26993) | 本文提出TRACE评估框架，将168个文本-视觉嵌入任务重新组织为通用、适应性和挑战三类，实验发现混合通用与适应性数据可显著提升适应性性能但挑战性能仍低于5.0分，且多语言和视觉文档任务为主要薄弱环节，表明仅靠数据适应不足以解决这些挑战。 |
| [^139] | [Think Before You Transcribe: Context-Steered Reasoning for Hotword-Aware Speech Recognition](https://arxiv.org/abs/2610.26986) | 该论文提出一种上下文引导的推理框架，通过在转录前先生成基于音频的简短描述来评估热词候选的相关性，从而在多种上下文条件下同时提升语音识别的转录质量和热词识别准确率，并对无关上下文保持鲁棒性。 |
| [^140] | [Sharp Degree Bounds for Putinar Certificates in Terms of the Positivity Margin via Analytic Sums of Squares](https://arxiv.org/abs/2610.26976) | 本文建立了关于正性裕度 δ 的 Putinar 平方和证书的精确次数上界 O(δ^(-1/2)log²(2/δ))，并通过构造三变量匹配下界例子证明了对 δ 的依赖是最优的，同时给出了 Lasserre 层级的精确最坏情形收敛速率。 |
| [^141] | [Transformers Learn Two Competing Sorting Algorithms, and Only One Length-Generalizes](https://arxiv.org/abs/2610.26974) | 该论文发现两层Transformer在排序任务上会根据权重初始化收敛到两种机制截然不同的解，其中只有同时使用两个注意力层的解能在仅训练固定长度序列的情况下完美实现长度泛化，且可以通过精心设计权重初始化来引导模型学习目标解。 |
| [^142] | [SZED at ArGuard Shared Tasks: Multi-Task Label-Attention and Cross-Modal Fusion for Arabic Hateful Meme Detection](https://arxiv.org/abs/2610.26972) | 该论文提出了一种融合MARBERTv2文本嵌入与CLIP视觉嵌入的多任务标签注意力架构，并引入级联校准机制，用于阿拉伯语仇恨表情包检测，在二分类任务上取得macro-F1 0.7597、在细粒度多标签任务上取得0.3142的成绩。 |
| [^143] | [Do LLMs Really Lack Difference Awareness? Uncovering Heuristic Substitution in Group-Differentiating Fairness](https://arxiv.org/abs/2610.26961) | 本文提出大语言模型在群体差异化公平任务中的失败并非源于缺乏差异意识，而是因为“公平启发式替代”——模型用更简单的形式平等启发式替代了复杂的差异识别与权衡过程，并据此提出两阶段“区分-决策”框架TDD来改善这一问题。 |
| [^144] | [Dynamic System Inspired Continuous Large Language Model for AI4Science Tasks](https://arxiv.org/abs/2610.26959) | Continuum-LM将预训练的残差变换复用为动力系统算子，构建了通过常微分方程演化隐藏状态的连续深度语言模型，在不扩展主干网络的前提下增加了潜在计算能力，在八项科学任务上将平均精确匹配准确率显著提升了7.77至10.96个百分点。 |
| [^145] | [LAMB: Latent Multi-Hypothesis Beliefs for Long-Horizon Large Language Agents](https://arxiv.org/abs/2610.26955) | 提出LAMB框架，通过潜在空间中的Common-Delta分解机制将交互证据与竞争性假设分离表示，使长时程语言智能体无需生成文本记忆即可高效维持多假设信念。 |
| [^146] | [A Multi-granularity Interaction Network for Relational Triple Extraction](https://arxiv.org/abs/2610.26952) | 提出一种多粒度交互网络，通过词与词、实体与关系、头尾实体三种交互及多次迭代来捕捉深层语义依赖，显著提升了关系三元组抽取中对“未见过”实体的处理性能。 |
| [^147] | [The Bottleneck Is Retrieval, Not Reading: Diagnosing Evidence-Based Generation with RAG for Long Context Question Answering](https://arxiv.org/abs/2610.26951) | 该论文提出基于证据的生成（EBG）方法及 Ans+Evid 联合评估指标，通过构建多个多文档问答基准诊断发现：RAG 的瓶颈在于基于相似度的检索环节而非模型阅读，而这些检索局限正是长上下文模型所能规避的。 |
| [^148] | [GRaSP: Latency-Aware Optimization for Geo-Distributed Retrieval-Augmented Generation](https://arxiv.org/abs/2610.26938) | 该论文提出GRaSP，一种面向地理分布式检索增强生成的延迟感知重评分方法，通过将问题形式化为划分拟阵上的线性目标并用O(N log N)贪心算法最优求解，在无需重训练的情况下兼顾检索相关性、延迟与成本。 |
| [^149] | [Do We Need the Largest LLMs for Every Clinical Decision?](https://arxiv.org/abs/2610.26937) | 提出CERTA决策级统计认证框架，证明量化后的小型模型可在统计保证下替代大型BF16模型处理绝大多数临床验证决策——8位量化满足99.8-100%的决策，4位量化将部署成本降至25.8%，因此并非每个临床决策都需要最大的LLM。 |
| [^150] | [AtomSD: Procedural Atom-Guided Self-Distillation for LLM Agents](https://arxiv.org/abs/2610.26926) | 提出AtomSD框架，将细粒度、可独立复用的“过程性原子”行动知识从智能体自身成功轨迹中持续提取并自蒸馏进策略，使LLM智能体性能超越现有最先进方法最多9.31分。 |
| [^151] | [PW-Pruning: Phase-Aware Expert Selection for Speech-Centric Omni-Modal Models](https://arxiv.org/abs/2610.26891) | 提出阶段感知的专家剪枝方法PW-Pruning，为声学预填充和文本答案生成两个阶段分别构建重要性加权核并融合阶段加权对数行列式边际增益来选择静态专家库，在Qwen3-Omni上每层仅保留32/128个专家即可使MMAU-mini和MMSU准确率较池化选择提升3.20和4.38个百分点。 |
| [^152] | [Disentangling Depression from Cognitive Decline in Elderly Speech Using Concurrent Clinical Assessments](https://arxiv.org/abs/2610.26890) | 该研究利用历时三年收集、带有同步抑郁与认知评估的韩国MCI老年人语音语料库，发现控制认知功能后仅共振峰特征与抑郁相关，仅12个特征的F1+F2子集即达到UAR 0.760，超越完整eGeMAPS基线和自监督表示。 |
| [^153] | [Mixture of Phonetic Experts Based Low-Rank Adaptation of Conformer Models for Accented English Speech Recognition](https://arxiv.org/abs/2610.26888) | 该论文提出MoPE-LoRA框架，按照发音方式类别而非口音身份来组织六个低秩专家，通过音素监督与声学门控相结合的帧级混合路由，实现参数高效的带口音英语语音识别适配，并可自然泛化到未见过的口音。 |
| [^154] | [Multimodal Ambivalence and Hesitancy Recognition with Bidirectional Audio--Text Cross-Attention](https://arxiv.org/abs/2610.26868) | 该论文提出了一种基于双向音频-文本交叉注意力的多模态框架，融合多语言BERT和wav2vec 2.0的表征，在无需显式时间对齐的情况下识别视频中的矛盾与犹豫状态，在BAH数据集上达到72.19%的准确率。 |
| [^155] | [WaveCommitments: Private and Verifiable Speech-Agent Marketplaces](https://arxiv.org/abs/2610.26867) | 提出WaveCommitments，一种将可信执行环境内基于Data Shapley的机密资产估值与多元多项式承诺相结合的端到端设计，实现了语音智能体市场中私密且可验证的资产交易，相比最佳基准速度提升最高20倍、证明大小缩减最高39倍。 |
| [^156] | [High-dimensional layer-state trajectories reveal the internal dynamics of transformer inference](https://arxiv.org/abs/2610.26866) | 该研究将Transformer逐层推理的有序层状态轨迹作为经验对象，发现真实层顺序产生的轨迹对齐度是保持终点随机打乱的1.60至2.24倍，证明层顺序构成一种与架构兼容而非训练特有的可测量计算，并可通过局部转换执行器实现对内部推理轨迹的选择性控制。 |
| [^157] | [AutoTerm-SST: Adaptive Terminology Memory with Zero Session-Time Setup for Streaming Speech Translation](https://arxiv.org/abs/2610.26865) | AutoTerm-SST提出了一种无需会话时上传术语表和选择领域的自适应术语记忆系统，通过将离线术语目录组织为主题切片并根据流式翻译上下文动态激活相关切片，在肿瘤学演讲翻译中实现了与已知领域设置相当的术语准确率。 |
| [^158] | [H2DF: Calibrated Target Bands for Controllable Language-Model Unlearning](https://arxiv.org/abs/2610.26864) | H2DF提出了一种校准的目标区间目标函数，通过数据依赖的目标带控制梯度上升压力，在遗忘达到目标后自动抑制或反转压力，从而实现可控的语言模型遗忘，同时保护保留数据的效用。 |
| [^159] | [WeAct: Dynamic Weight Switching as Agent Actions](https://arxiv.org/abs/2610.26847) | WeAct在冻结主干上训练独立的能力专用LoRA适配器，并将适配器选择纳入智能体动作空间，使模型能在长时程任务中通过原生工具调用接口按需组合能力，同时避免灾难性遗忘。 |
| [^160] | [MemT-1.0: Towards Memory-Native Evolution](https://arxiv.org/abs/2610.26825) | MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。 |
| [^161] | [HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization for Performance-Cost Trade-offs](https://arxiv.org/abs/2610.26770) | HarnessPro提出了一种无需标签的智能体框架递归自我改进方法，通过帕累托优化权衡性能与执行成本，并提供精简、均衡、重型三种框架版本以适配不同的部署预算。 |
| [^162] | [SpatialScout: Can VLM Agents Actively Explore and Map 3D Scenes?](https://arxiv.org/abs/2610.26659) | 提出了SpatialScout基准，用于评估视觉语言模型智能体在有限交互预算下，通过主动探索室内场景来构建物体级3D度量地图的能力。 |
| [^163] | [From Outcomes to Components: A Counterfactual Evaluation Framework for LLM-Proposed Repairs](https://arxiv.org/abs/2610.26650) | 该论文提出了一个反事实评估框架，将LLM提出的修复动作解耦为“目标记录选择”和“更新操作”两个组件，通过四种交叉执行方式分别评估各组件的独立贡献及其交互作用，从而超越单纯以成败论结果、更精细地评估LLM修复方案的质量。 |
| [^164] | [VibeTraining-Bench: Data-Trainer-Harness Co-Design for Iterative System Improvement](https://arxiv.org/abs/2610.26342) | 提出了VibeTraining-Bench基准，用于评估AI智能体在固定资源预算下协同优化训练数据、训练器与推理框架以实现递归自我改进的能力，并通过测试数据隔离、容器分离和网络限制等手段防止奖励作弊。 |
| [^165] | [Token-Set Choice Confounds POPE: A Systematic Audit of Yes/No Extraction in VLM Hallucination Evaluation](https://arxiv.org/abs/2610.25964) | 该论文系统审计了POPE幻觉评估中的是/否答案提取方式，发现传统双词元logit读出所比较的"yes"/"no"词元ID在LLaVA-1.5-7B的实际输出中从未出现过（模型实际生成的是带空格前缀的" Yes"/" No"），仅这一读出实现细节就能使F1分数偏移高达6.13分，说明评估代码的实现选择会严重混淆基准对比结论。 |
| [^166] | [When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding](https://arxiv.org/abs/2610.24993) | 该论文建立统一的训练-推理-评估对照协议，发现推测解码起草器之间在已发表结果中的接受率差距在相同条件下会大幅缩小，且接受率可通过近零成本的目标与上下文修改（如DCore强基线）或更昂贵的逐词元条件化来提升，只有当接受率提升的收益超过获取成本时才能带来吞吐量增益。 |
| [^167] | [Stable Signer V3: 4D Deformable Primitive Splatting Is All You Need for Sign Language Production](https://arxiv.org/abs/2610.24926) | Stable Signer V3基于4D可变形图元泼溅技术，通过共享面部图元泼溅、测试时身体图元泼溅和隧穿超分辨率三项创新，显著提升了手语虚拟形象的表情丰富度、身体变形准确性和高分辨率细节保真度。 |
| [^168] | [Ask, Don't Guess: Targeted Clarification for Speech Recognition Errors](https://arxiv.org/abs/2610.24809) | 提出PARDON框架，通过基于干净音频软监督训练的反事实增益评分器在线定位流式语音识别中可能出错的片段，请说话人重复该内容，并让大语言模型仅修正被选中的片段，从而实现“主动询问而非盲目猜测”的定向澄清式错误补救。 |
| [^169] | [Muse Glimmer 30B: An On-Device Agentic Model](https://arxiv.org/abs/2610.24628) | Muse Glimmer是一个开源的30B参数端侧智能体模型，集成了多步推理、可靠的工具使用和故障恢复能力，可在消费级硬件上本地运行且无需云端，并在智能体基准测试中超越Gemma4-31B和Qwen3.6-27B等同规模模型达到最先进性能。 |
| [^170] | [How Data Scales in Agentic Reinforcement Learning: Laws and Synthesis Strategies](https://arxiv.org/abs/2610.23946) | 该论文将智能体强化学习的训练数据分解为任务、环境和轨迹三个可独立扩展的维度，并拟合出各维度的缩放定律，以回答在给定合成预算下应优先扩展哪个数据维度这一关键问题。 |
| [^171] | [DiscoveryRSI: Recursively Learning to Discover Valuable Hypotheses](https://arxiv.org/abs/2610.23839) | 该论文提出 DiscoveryRSI 框架，通过多智能体进化搜索构建假设树，并借助假设信用优化、测试框架回放和再发现三种机制，实现对有价值科学假设发现过程的递归式持续学习。 |
| [^172] | [MultiAgent-OPD: End-to-End Training of Hierarchical Multi-Agent Systems with On-Policy Distillation](https://arxiv.org/abs/2610.23234) | 提出MultiAgent-OPD方法，通过将在线策略蒸馏与强化学习相结合，利用冻结的教师模型为搜索者提供局部过程监督，实现了层次化多智能体系统的端到端联合训练。 |
| [^173] | [SAFE-Mem: Secure Agentic Memory with Fine-Grained Authority and Governance-Aware Retrieval](https://arxiv.org/abs/2610.22959) | 提出SAFE-Mem框架，通过为记忆声明标注细粒度治理标签并结合治理感知检索，在保障LLM智能体记忆安全性的同时保留合法的记忆使用。 |
| [^174] | [Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?](https://arxiv.org/abs/2610.22887) | 提出并发约束强化（CCR），将多个可独立验证的问题放入同一个受限完成中，使RLVR在8,000-token的短窗口下仍能保留正确性信号，并将每训练步的token消耗从约97.5万降至约17.5万，大幅降低训练成本。 |
| [^175] | [Knowledge That Is Linear Before Fine-Tuning Generalizes After It](https://arxiv.org/abs/2610.22300) | 论文提出区分语言模型中支持泛化的“活性事实”与仅可回忆的“惰性事实”，并通过重标注实验发现：微调前在模型内部呈线性可读的知识，在微调后能够泛化到共享相同答案的未训练实体上。 |
| [^176] | [Training-Free On-Policy Distillation](https://arxiv.org/abs/2610.22236) | 本文提出免训练的在策略蒸馏方法，当多个专家模型共享同一基础模型时，仅需通过几百个采样样本的一次前向传播、以最小二乘方式逐层融合，即可在无需任何梯度训练的情况下将多个专家的能力聚合到一个模型中，且平均准确率超过任何单一专家模型。 |
| [^177] | [No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance](https://arxiv.org/abs/2610.21679) | 本研究通过分析约200个跨九种语言训练的语言模型，发现没有任何单一的分词器内在评价指标能够可靠地预测语言模型的下游性能，因此需要开发新的基于语言学的分词器评估方法。 |
| [^178] | [HyCo: Hybrid Drafting for Cost-Efficient Mobile-Cloud Speculative Decoding](https://arxiv.org/abs/2610.21436) | HyCo提出混合草稿生成机制，将服务器模型的原生多词预测头部署到移动NPU上并与小型语言模型单次推理协同工作，配合量化和ε阈值松弛策略，使移动-云推测解码的服务器GPU时间降低约29%-39%。 |
| [^179] | [Beyond Invariance: Factor-wise Selective Treatment of Shifts for Multimodal Learning](https://arxiv.org/abs/2610.21403) | 提出FaShift框架，超越传统不变性方法，通过按因子诊断分布变化、干预效应和跨环境稳定性，对多模态学习中的偏移信息进行选择性抑制、保留或削弱。 |
| [^180] | [SoK: Grading Open-Text Legal Answers, from Law School Exams to Trustworthy Rubric Judges](https://arxiv.org/abs/2610.19992) | 本文系统化研究了开放式法律文本答案（如法学院考试）的评估方法，从教育测量与法律规范中提炼出八项评估标准并对比八种评估范式，发现针对任务定制评分规则是让LLM裁判与人类评分者达成一致的最关键手段。 |
| [^181] | [ELARSpeech: A 30K-Hour Dataset Across 675 Underrepresented Languages for Long-Tail Multilingual Speech Technologies](https://arxiv.org/abs/2610.19388) | ELARSpeech是一个包含675种代表性不足语言、总时长30,202小时的大规模多语言语音数据集，为将语音识别与语音合成技术扩展至语言长尾领域提供了重要资源和测试平台。 |
| [^182] | [On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination](https://arxiv.org/abs/2610.18727) | 论文将自然语言自编码器的训练与监督部署形式化为两种不同的指称博弈，并指出二者之间的差距导致其解释虽能识别对应的激活，却难以让人类监控者从多个近似叙述中辨别出真实的那一个。 |
| [^183] | [Many Steps into One: Reasoning Contraction and the Limits of Safe Memory Reuse for LLM Agents](https://arxiv.org/abs/2610.17544) | 论文提出“推理收缩”概念，证明LLM智能体中基于成败计数的记忆更新机制是糟糕的变化检测器（比认证记忆多产生7-73倍的过时答案），并揭示了安全记忆复用存在由“认证负载”这一指标决定的理论上限。 |
| [^184] | [StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions](https://arxiv.org/abs/2610.17309) | 该论文提出StartLux-Decision系列决策模型（0.8B至27B参数），通过置信度引导的自适应推理机制与全新的RLCD²强化学习方法（利用冻结模型和适当评分规则奖励校准的决策概率质量、惩罚过长推理），实现了高效单次预测与按需深度推理的结合。 |
| [^185] | [PersonaFactor: Preference Factors with Dynamic Selection for LLM Personalization](https://arxiv.org/abs/2610.16372) | PersonaFactor框架从异构用户交互中提炼出十三个可复用、可解释的个性化偏好因子，并通过动态因子选择机制，在四个个性化基准和四个骨干模型的所有指标上均取得最佳性能。 |
| [^186] | [Compute-Optimal Scaling Laws for Spiking Language Models](https://arxiv.org/abs/2610.15146) | 本文首次对脉冲语言模型开展Chinchilla式缩放研究，发现其计算最优缩放指数（0.56）与密集Transformer（0.50）接近，且脉冲化带来的损失代价随计算预算增加而稳步下降。 |
| [^187] | [What Survives When Activation Steering Is Used as Memory](https://arxiv.org/abs/2610.14958) | 该研究提出一种免训练的激活记忆机制，并证明激活引导偏移作为注意力效应的固定平均值只能存留与问题无关的内容：这种记忆虽具有约98%的高选择性，但用户事实仅被存储在最后几层且几乎从不被使用，表明用户事实与偏好仍需以文本形式记忆。 |
| [^188] | [LongHisDoc: A Comprehensive Benchmark for Chinese Long Historical Document Understanding](https://arxiv.org/abs/2610.14531) | 该论文提出了首个针对中文长篇历史文献理解的基准LongHisDoc，包含来自101份PDF文档的1,012个专家标注问答对，评估发现当前最先进的模型准确率均不足70%，揭示了该领域仍有巨大提升空间。 |
| [^189] | [Scaling Agentic Data for Long-Horizon Terminal Intelligence](https://arxiv.org/abs/2610.13166) | 提出 TerminalHorizon，一个面向长程终端智能体的全自动化数据合成引擎，通过避免环境合成预先完成关键工作，生成能真正激发智能体探索、实施与验证能力的高质量长程训练数据。 |
| [^190] | [Not Self-Decidable: LLMs Cannot Draw the Boundary of What an Agent Verifier Can Check](https://arxiv.org/abs/2610.04699) | 研究通过对六个语料库、约22,000个标签的实证分析表明，大语言模型无法自行判定Agent验证器在面对外部监管规则时哪些内容可以自动检查、哪些需要人工裁决，因此“可自判定”这一升级方案的核心假设不成立。 |
| [^191] | [Penumbra: Sample-Efficient Adversarial Search for Regulatory Obligations](https://arxiv.org/abs/2610.04693) | 提出Penumbra，一种样本高效的对抗搜索方法，通过从经验证的锚点出发自适应地分配编辑预算，找到使监管合规判定翻转的最小编辑相邻回复对，从而以更少的探测成本最大化覆盖监管义务的“失效面”。 |
| [^192] | [$S^3$: Spectral Null-Space Swap Makes Reasoning Models Efficient](https://arxiv.org/abs/2609.37976) | 论文首次发现推理能力的关键在于思考模型权重中位于非思考模型主导子空间零空间内的分量，并提出无需训练的谱零空间交换方法S³，通过配对检查点组合在保持精度的同时大幅提升推理效率。 |
| [^193] | [Pair Difficulty Matters: Rethinking Pairwise LLM-as-a-Judge Evaluation and Consistency](https://arxiv.org/abs/2609.37577) | 该论文指出，传统用于评估LLM裁判可靠性的三大代理指标（位置偏差、传递性、成对一致性）具有误导性——在Bradley-Terry几何下它们被排名差距接近的配对所主导，而此类配对出现不一致在信息论上是必然的，因此不应以这些指标否定能力出色的LLM评估器。 |
| [^194] | [Constructing Challenging Browser-Use Tasks by Controlled Environment Interventions](https://arxiv.org/abs/2609.35814) | 提出 BreakingWeb 基准，通过在保持用户指令与成功标准不变的前提下对环境不同技术栈层面进行确定性干预，将浏览器任务的难度转化为可编程的环境属性，从智能体已能解决的任务中系统性构建出 519 对更具挑战性的任务对。 |
| [^195] | [KuaFu: Compressing Long User Behavior into Understanding at Billion Scale](https://arxiv.org/abs/2609.31045) | 该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。 |
| [^196] | [Which Medical Questions Deserve Rationales? Perturbation-Sensitive Selection for Robust QA](https://arxiv.org/abs/2609.09684) | 该论文提出RMS-RSP方法，通过仅在推理依据token处扰动隐藏状态并测量答案与干扰项之间裕度的变化，在固定token预算下智能筛选哪些已标注医学问题最值得投入推理依据监督，从而提升医学问答的鲁棒性。 |
| [^197] | [Knowledge Acquisition During Pre-training? Large Language Models Learn Better With Auxiliary Views](https://arxiv.org/abs/2609.04180) | 研究发现，在预训练中将token预算从文档重复转移到辅助视图（知识的重新表述）能提升大语言模型的学习效果，即使对事实回忆也有效，且不依赖教师模型的强弱。 |
| [^198] | [When Models Edit Too Much: On the Fidelity of Minimal Code Edits](https://arxiv.org/abs/2609.04061) | 该研究揭示了前沿大语言模型在修复代码时普遍存在“过度编辑”问题（即使如GPT-5.5这样的强模型也不例外），并提出通过一条简单的保留指令即可显著减少不必要的代码改动、降低认知复杂度，同时提升修复准确率。 |
| [^199] | [SpanCalib-VLM: Calibrated Hallucination Span Detection in Vision-Language Models](https://arxiv.org/abs/2608.29974) | 提出 SpanCalib-VLM 混合双系统，将多模态序列标注器与微调生成式 VLM 相结合，并通过联合校准融合策略对候选幻觉片段进行重新评分，同时兼顾了准确的片段定位与良好校准的置信度。 |
| [^200] | [Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit](https://arxiv.org/abs/2608.27309) | 本文揭示双重差分法在截断评分量表上因截断不均会制造虚假交互效应，并通过预注册审计实证证明该偏差可导致无效结果。 |
| [^201] | [Conditional Total Correlation and the Serial Depth of Adaptive Parallel Sampling](https://arxiv.org/abs/2608.25505) | 本文证明了条件总相关性是自适应并行采样中并行性的精确信息成本，并为有限阶马尔可夫链提供了与阶数成正比、与长度成对数关系的零误差调度方案。 |
| [^202] | [FRAGMENT: Factorized Graph Representations for Document Generation and Editing via Entity-Aware Transformations](https://arxiv.org/abs/2608.18679) | 提出FRAGMENT框架，将结构化文档表示为类型化关系图，通过“架构师”（生成图拓扑与空间关系）和“构建器”（填充边界框、文本与视觉样式）两阶段因子化建模，实现了具有可处理似然的文档生成与编辑。 |
| [^203] | [ReRef-3D: A Benchmark for Spatial Referring Expression-Guided 3D Scene Rearrangement](https://arxiv.org/abs/2608.16011) | 本文提出了ReRef-3D基准，用于评估语言引导的三维场景重排，发现关系满足性优于物理有效性，且“最近”和“之间”等关系最具挑战性。 |
| [^204] | [The Machine's Internal Clock: Do LLMs Share Human Temporal Illusions?](https://arxiv.org/abs/2608.15394) | 本研究通过新基准发现，大语言模型在时间错觉任务中比人类更倾向于选择文献预测场景，表明模型可能缺乏人类对时间的主观感知偏差。 |
| [^205] | [TokenSwap: Benchmarking and Reducing the Modality Gap in Multimodal LLMs](https://arxiv.org/abs/2607.28640) | 本文提出TokenSwap方法，通过将文本基准（如MMLU）中的文本概念替换为语义对齐的图像，构建出图像交错的TokenSwap-Bench基准，首次系统揭示了42个多模态大语言模型中普遍存在的模态差距，即图像交错输入相比纯文本输入平均导致19.6%的性能下降。 |
| [^206] | [SINT-Flow: Schema Integration using Large Language Model Workflows](https://arxiv.org/abs/2607.24492) | 本文提出由五个LLM算子组成的模式集成框架SINT-Flow，可实现全自动端到端模式集成，并首次支持将含多种实体类型的反规范化源表分解为实体特定的关系，同时发布了包含10个任务、93张关系表的模式集成基准SINT-Bench。 |
| [^207] | [DECODEM: Data Extraction from Corporate Organizational Documents via Enhanced Methods](https://arxiv.org/abs/2607.15879) | 本文提出DECODEM基准测试，利用SEC EDGAR数据库中的上市公司章程及章程细则样本与人工标注标签，系统评估大型语言模型提取公司治理变量的能力，为耗时昂贵的人工编码和覆盖有限的商业数据集提供了新的替代方案。 |
| [^208] | [When Synthetic Speech Is All You Have: Better Call GRPO](https://arxiv.org/abs/2607.08409) | 本文证明在隐私受限的银行等领域，用强化学习GRPO（尤其SFT+GRPO组合）替代监督微调来适配基于LLM的ASR，仅凭合成语音即可将词错误率相对降低40%–45%，其增益主要来自停止校准和语音-文本对齐等行为层面的改善。 |
| [^209] | [Probe, Don't Prompt: A Hidden-State Probe for Metadata Filtering in Multi-Meta-RAG](https://arxiv.org/abs/2607.03929) | 该论文提出用小型开源语言模型隐藏状态上训练的本地确定性多标签探针，替代Multi-Meta-RAG中基于GPT-3.5的新闻来源元数据提取器，从而免去专有API调用并将输出严格限制在允许的来源列表内。 |
| [^210] | [Randomized YaRN Improves Length Generalization for Long-Context Reasoning](https://arxiv.org/abs/2606.23687) | 随机化YaRN通过将YaRN位置外推与随机化位置编码和长度课程学习相结合，使模型在仅使用短上下文数据训练的情况下，也能在16K到128K的长上下文推理任务上实现一致的泛化性能提升。 |
| [^211] | [Decomposing Refusal Steering in Mixture-of-Experts Models](https://arxiv.org/abs/2606.04160) | 该研究将拒绝引导方法扩展到混合专家模型，发现单个专家在自由选择位置时平均可恢复78%的完整引导效果，从而揭示了MoE模型中拒绝机制在各组件间的运作方式。 |
| [^212] | [The Shape of Wisdom: How Language Models Converge on an Answer](https://arxiv.org/abs/2606.01202) | 该研究发现，语言模型在回答事实性问题时，得到相同答案的问题会在内部状态空间中收敛到共享的“答案质心”，该质心在网络约47%-71%深度处突然变得可读取，且只取决于答案内容本身而非问题的表述方式。 |
| [^213] | [When Is 0.1% Enough? Analyzing the Combined Effects of Dimensionality Reduction and Quantization on Text Embedding Compression](https://arxiv.org/abs/2606.01074) | 本文系统研究了降维与量化相结合对文本嵌入压缩的联合效应，发现在分类任务中仅需不到0.1%的存储空间即可保留至少99%的原始性能。 |
| [^214] | [Measuring, Localizing, and Ablating Alignment Signatures in LLMs](https://arxiv.org/abs/2605.30526) | 该论文发现后训练会使大语言模型的生成文本偏离人类语料风格并呈现出可被AI检测器识别的类AI特征，并提出了无需训练的PASTA方法，通过从残差对比中估计对齐签名方向并在解码时消融该方向，从而定位并去除模型内部的这一AI风格签名。 |
| [^215] | [Same Evidence, Different Answers: Canonical-Context On-Policy Distillation for Multi-Turn Language Models](https://arxiv.org/abs/2605.30251) | 提出规范上下文同策略蒸馏（CCOPD），让同一基础模型分别担任基于完整提示的冻结教师与逐步接收分片证据的多轮学生，并将学生在自身轨迹上的行为与教师的规范全上下文行为对齐，从而缓解多轮对话中的自我锚定漂移，使模型在证据相同时无论以完整提示还是分片对话呈现都能给出一致的答案。 |
| [^216] | [What a One-Hour Interview Does Not Tell an LLM About You: Interviews Elicit Traits, Not Decision Policies](https://arxiv.org/abs/2605.29458) | 一小时LLM访谈无法可靠预测个人在价值两难问题上的真实决策——访谈捕捉到的是人格特质而非具体的决策策略，且大多数预测错误源于转录文本中根本缺乏区分个体选择的关键证据。 |
| [^217] | [Robust LLM Watermarking with Minimal Semantic Distortion for IP Protection](https://arxiv.org/abs/2605.23175) | 提出了一种密钥条件水印框架SAFESEAL，通过保留命名实体并借助锦标赛采样机制用上下文感知的同义词替换语言术语，在几乎不损害模型效用的情况下实现了强可检测性和鲁棒性，用于大语言模型的知识产权保护。 |
| [^218] | [Psy-Chronicle: Synthesizing Campus Counseling Grounded in a Student's Semester Timeline](https://arxiv.org/abs/2605.22140) | Psy-Chronicle 提出一个事件驱动的多轮心理咨询对话合成框架，以模拟学生的学期时间线和压力事件图为根基生成跨会话携带记忆的对话，并据此构建了包含 100 名学生、1,248 次会谈的开放中文校园心理咨询数据集 CPCD。 |
| [^219] | [Rethinking How to Remember: Beyond Atomic Facts in Lifelong LLM Agent Memory](https://arxiv.org/abs/2605.19952) | TriMem通过同时维护原始对话片段、原子化事实和合成档案三种共存的信息表示粒度，克服了现有基于原子事实的记忆方法丢失细粒度细节、无法深度推理且提取粒度不一致的局限，提升了终身LLM智能体记忆系统的存储保真度、检索效率和推理深度。 |
| [^220] | [Dynamic Model Merging Made Slim](https://arxiv.org/abs/2605.18904) | DiDi-Merging通过可微秩选择在共享参数与专家参数之间联合分配低秩容量，并结合无数据的参数空间精炼，在仅1.24倍骨干网络存储的预算下实现了领先的动态模型合并性能。 |
| [^221] | [Performance-Driven Policy Optimization for Speculative Decoding with Adaptive Windowing](https://arxiv.org/abs/2605.14978) | 提出PPOW强化学习框架，将推测解码中草稿模型的优化从token级模仿提升到窗口级优化，通过成本感知加速奖励、分布接近度奖励和自适应散度感知窗口机制，有效解决难起草位置导致的推测效率瓶颈，加速LLM推理。 |
| [^222] | [Using Embedding Models to Improve Probabilistic Race Prediction](https://arxiv.org/abs/2604.22555) | 提出eBISG方法，利用预训练文本嵌入将姓名表示为稠密向量并结合2020年人口普查数据训练神经网络，显著改进了标准BISG对被遗漏的不常见姓氏人群的种族概率预测性能。 |
| [^223] | [Abstract Compression of Conversational Audio Context for LLM-based ASR](https://arxiv.org/abs/2603.26246) | 提出抽象压缩方法，将对话历史音频压缩为固定数量的潜在标记并保留转录文本，使基于LLM的语音识别以更小开销提升上下文实体的识别效果。 |
| [^224] | [Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails](https://arxiv.org/abs/2603.18280) | 该论文以中国语言模型的政治审查为自然实验，发现对齐机制真正作用于“从概念检测到行为策略”的路由环节而非概念检测本身，从而解释了仅基于拒绝行为的对齐评估为何会失效。 |
| [^225] | [CHIMERA: Compact Synthetic Data for Generalizable LLM Reasoning](https://arxiv.org/abs/2603.00889) | 该论文提出CHIMERA——一个仅含9K样本的紧凑合成推理数据集，通过提供详细的长思维链轨迹、覆盖更广泛的科学学科并摆脱人工标注依赖，一举解决LLM推理训练中的冷启动、领域覆盖不足和标注瓶颈三大数据挑战。 |
| [^226] | [TTCS: Test-Time Curriculum Synthesis for Self-Evolving](https://arxiv.org/abs/2601.22628) | TTCS提出了一种共同进化的测试时训练框架，通过问题合成器与推理求解器的迭代协同优化，生成难度递增的课程化问题变体，从而提升大语言模型在困难推理问题上的能力。 |
| [^227] | [Clarify or Answer: Reinforcement Learning for Agentic VQA with Context Under-specification](https://arxiv.org/abs/2601.16400) | 该论文提出基于强化学习的CoA智能体框架，使视觉问答模型在遇到上下文欠规范的歧义问题时先判断是否需要澄清，再通过迭代生成澄清问题来消除歧义，从而避免直接回答带来的自信但错误的预测。 |
| [^228] | [Ask, Answer, and Detect: Role-Playing LLMs for Personality Detection with Question-Conditioned Mixture-of-Experts](https://arxiv.org/abs/2512.08814) | 提出ROME框架，通过大语言模型角色扮演模拟用户回答心理测量问卷，并利用问题条件化混合专家模块提供可解释的中间监督，从而将心理学知识融入社交媒体人格检测。 |
| [^229] | [Attention-Aligned Reasoning for Large Language Models](https://arxiv.org/abs/2510.03223) | 提出了ATAR推理方法，通过利用固有推理结构引导大语言模型的注意力，避免关键中间步骤和原始提示被埋没，在六个基准测试中最高实现15.39%的绝对提升，并使非推理模型达到与同规模推理模型相当甚至更好的性能。 |
| [^230] | [Label Unification for Cross-Dataset Generalization in Cybersecurity NER](https://arxiv.org/abs/2507.13870) | 该论文研究了四个网络安全NER数据集的标签统一以提升数据可用性，但实验表明统一训练、多头模型和图迁移模型均未能稳定超越单数据集基线。 |
| [^231] | [A Survey of Generative Categories and Techniques in Multimodal Generative Models](https://arxiv.org/abs/2506.10016) | 本综述系统分类了六类文本条件的多模态生成模态（图像、音乐、视频、人体动作、3D物体等），分析了自监督学习、混合专家、RLHF和思维链提示四种技术在各模态中的适配机制，并围绕忠实性、组合性和鲁棒性得出四项关键发现，包括偏好增益未必提升文本相关性、奖励优化可能降低输出多样性等。 |
| [^232] | [LLMs are Brittle to Simple Code Transformations: Introducing CETBench - A Benchmark for Code-Equivalence Checking](https://arxiv.org/abs/2506.04019) | 本文提出了代码等价性检测基准CETBench，发现大语言模型即使面对简单的代码变换也会出现显著性能下降（在跨语言场景下尤为严重），并提出一种基于微调的简单方法来提升模型性能。 |
| [^233] | [Rethinking Generalization in Knowledge Graph Embedding: An Empirical Study and Sparse Peak Regularization](https://arxiv.org/abs/2505.23442) | 本文从泛化角度对知识图谱嵌入的正则化进行了跨模型族的系统性实证研究，发现过拟合在各类KGE模型中普遍存在，并提出稀疏峰值正则化方法来显著提升模型性能。 |
| [^234] | [KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation](https://arxiv.org/abs/2504.19024) | 该论文提出KETCHUP方法，通过多步使用贝尔曼最优方程诱导K步回报来降低梯度估计方差，显著提升了文本生成任务中基于强化学习的知识蒸馏效果，尤其适用于大型学生模型。 |
| [^235] | [Computational Typology](https://arxiv.org/abs/2504.15642) | 本文介绍了计算统计建模如何助力语言类型学研究，使大规模语言数据分析与语言结构和演化假设的检验成为可能。 |
| [^236] | [MedHal: a Synthetic Dataset for Medical Hallucination Detection](https://arxiv.org/abs/2504.08596) | MedHal是一个涵盖内在与外在幻觉的大规模合成医学数据集，用于训练和评估医学文本幻觉检测模型，基于该数据集训练的基线模型优于通用幻觉检测方法。 |
| [^237] | [Can Language Models Replace Programmers for Coding? REPOCOD Says 'Not Yet'](https://arxiv.org/abs/2410.21647) | 该论文提出了REPOCOD基准，包含来自11个真实热门项目的980个Python完整函数生成任务，以评估语言模型在真实世界编码任务中的能力，结果显示当前语言模型尚不能替代程序员。 |
| [^238] | [WAFFLE: Finetuning Multi-Modal Models for Automated Front-End Development](https://arxiv.org/abs/2410.18362) | Waffle提出了一种结合结构感知注意力机制与对比微调的多模态模型微调策略，显著提升了从UI设计图自动生成HTML代码的效果。 |
| [^239] | [Typhoon: Towards an Effective Task-Specific Masking Strategy for Pre-trained Language Models.](http://arxiv.org/abs/2303.15619) | 本文探讨了一种针对预训练语言模型的任务特定的屏蔽框架，称为Typhoon，可在GLUE基准数据集上实现卓越的下游任务性能，尤其在MRPC数据集上表现优异。 |

# 详细

[^1]: NarraSpace：面向图片描述型痴呆症分类的基于场景的多模态证据

    NarraSpace: Multimodal Scene-Grounded Evidence for Picture-Description Dementia Classification

    [https://arxiv.org/abs/2610.28616](https://arxiv.org/abs/2610.28616)

    NarraSpace通过将语音内容与图片场景元素关联并提供可溯源的场景证据，实现了兼具高准确率（约81%）与可解释性的痴呆症图片描述分类。

    

    图片描述型痴呆症分类使用共享的视觉场景，但仅基于转录文本的预测无法识别其输入背后的场景证据。NarraSpace 将词语和语音时间与命名的图片元素关联起来，并检索预定义场景事件对应的语句。叙事-空间描述符总结了覆盖范围与时间组织情况。一个冻结的语言模型参照来自阿尔茨海默病患者和健康对照组训练参与者的示例来对比每个事件的文本，同时结合转录文本的惊讶度与覆盖度测量。保留的源记录将事件分数与其源语句、语音区间和词汇匹配相联系。在“通过自发言语识别阿尔茨海默病痴呆”基准的官方48名参与者测试集上，一个固定的支持向量分类器在五十个随机种子上取得了80.88%±2.56%的平均准确率；平均概率方法达到81.25%的准确率和91.32%的曲线下面积。

    arXiv:2610.28616v1 Announce Type: new  Abstract: Picture-description dementia classification uses a shared visual scene, but a transcript-level prediction alone does not identify the scene evidence underlying its inputs. NarraSpace links words and speech times to named picture elements and retrieves utterances for predefined scene events. Narrative-spatial descriptors summarize coverage and temporal organization. A frozen language model contrasts each event's text under examples from Alzheimer's disease and healthy-control training participants, alongside transcript surprisal and coverage measurements. Retained source records link event scores to their source utterances, speech intervals, and lexical matches. On the official 48-participant test set of the Alzheimer's Dementia Recognition through Spontaneous Speech benchmark, a fixed support vector classifier achieves $80.88\%\pm2.56\%$ mean accuracy over fifty seeds; averaged probabilities yield $81.25\%$ accuracy and $91.32\%$ area un
    
[^2]: 阿拉伯方言数学问题基准测试

    Benchmarking Arabic Dialects for Math Problems

    [https://arxiv.org/abs/2610.28607](https://arxiv.org/abs/2610.28607)

    该论文推出了首个用于评估阿拉伯方言数学推理的人工整理基准DAR-MGSM，涵盖五种主要阿拉伯方言及现代标准阿拉伯语，并发现大语言模型在文化本地化版本上性能普遍下降。

    

    数学推理在包括阿拉伯方言在内的低资源语言中仍然缺乏充分研究。为了弥合这一差距，我们推出了DAR-MGSM（阿拉伯方言多语言小学数学）基准，这是首个用于评估阿拉伯方言数学推理能力的人工整理基准。DAR-MGSM由需要多步推理的数学应用题（MWPs）组成，涵盖五种主要阿拉伯方言以及现代标准阿拉伯语（MSA）。该基准包含两个平行版本：仅翻译版本和翻译并文化本地化版本，后一版本将实体和引用调整为区域性变体。我们在DAR-MGSM上评估了十一个以阿拉伯语为中心的多语言大语言模型。结果显示，模型在本地化版本上的性能普遍下降，凸显了文化背景适应所带来的挑战。然而，这种影响的程度因模型而异。

    arXiv:2610.28607v1 Announce Type: cross  Abstract: Mathematical reasoning remains underexplored for low-resource languages, including Dialectal Arabic. To bridge this gap, we introduce the DAR-MGSM (Dialectal Arabic Multilingual Grade School Math) Benchmark, the first manually curated benchmark for evaluating mathematical reasoning in Dialectal Arabic. DAR-MGSM consists of math word problems (MWPs) requiring multi-step reasoning and covers five major Arabic dialects alongside Modern Standard Arabic (MSA). The benchmark comprises two parallel versions: translated-only and translated & culturally localized. The latter version adapts entities and references to regional varieties. We evaluate eleven Arabic-centric and multilingual large language models on DAR-MGSM. The results show that the performance generally declines on the localized version, highlighting the challenges introduced by culturally grounded adaptations. However, the magnitude of this effect varies across models, with Arabi
    
[^3]: 有能力的教师，短视的评分者：面向智能体在线策略蒸馏的无奖励教师修复

    Capable Teacher, Myopic Scorer: Reward-Free Teacher Repair for Agentic On-Policy Distillation

    [https://arxiv.org/abs/2610.28586](https://arxiv.org/abs/2610.28586)

    论文揭示智能体在线策略蒸馏（OPD）失败的根源不仅是误差累积，更在于教师逐 token 评分的“短视性”——其评分看不到回合结局，从而放过了安全却拖延的行动，并据此提出了一种无需奖励的教师修复方法。

    

    智能体在线策略蒸馏（OPD）让学生智能体在自己的轨迹上进行训练，同时由教师模型对每个 token 进行评分。对其失败原因的一种自然解释是误差累积：学生的错误使其轨迹偏离了教师可靠的区域，因此近期的方法通过回合长度课程、教师引导轮次、前缀回放或在师生分歧处进行教师干预等方式，将训练保持在该区域内。我们证明这种解释是不完整的。在学生最偏离教师的状态下，由教师自己接管仍然能够完成任务，然而基于教师 token 评分训练出的学生却总是耗尽回合数。我们将此归因于下一 token 评分的短视性。教师在评分时看不到回合的最终结局，因此其评分放过了那些安全却拖延的行动，而基于结果的强化学习（RL）会以零奖励惩罚这类行动——这种惩罚机制正是智能体 OPD 所缺乏的。我们将此称为

    arXiv:2610.28586v1 Announce Type: cross  Abstract: Agentic on-policy distillation (OPD) trains a student agent on its own trajectories while a teacher scores every token. A natural explanation of where it goes wrong is compounding error: the student's mistakes carry its trajectories beyond the region where the teacher is reliable, so recent methods keep training inside that region, through episode-length curricula, teacher-guided turns, prefix replay, or teacher intervention where the two disagree. We show this explanation is incomplete. From the states where the student departs most from the teacher, the teacher's own takeovers still finish the task, yet the student trained on its token scores keeps running out of turns. We attribute this to the myopia of next-token scoring. The teacher rates each token without seeing where the episode ends, so its scores let through the safe, delaying moves that outcome-based RL would punish with zero reward, a penalty agentic OPD lacks. We call this
    
[^4]: 面向智能体的缓存工程

    Cache Engineering for Agents

    [https://arxiv.org/abs/2610.28577](https://arxiv.org/abs/2610.28577)

    该论文提出“缓存工程”新范式，使LLM智能体无需修改轨迹或重新预填充上下文，即可通过drop和repos等操作直接控制活跃KV缓存，从而在降低缓存I/O与存储成本的同时保持任务准确率。

    

    大语言模型（LLM）智能体能够处理日益复杂的任务，在漫长的观察与动作轨迹中持续导航。随着这些轨迹不断增长，键值（KV）缓存的输入/输出与存储成为主导性成本。轨迹还可能超出LLM的最大上下文长度，因此需要对活跃缓存的增长加以控制。已有工作采用上下文工程，通过裁剪或摘要历史内容来减小缓存规模。然而，修改轨迹前缀会使后续的缓存状态失效，并触发对保留上下文代价高昂的重新预填充；而为限制重新预填充成本而保留过少的上下文时，任务准确率也会受到损害。我们提出“缓存工程”这一新方案。它允许智能体直接控制其活跃的KV缓存，而无需修改轨迹或对保留上下文进行重新预填充。具体而言，智能体可以主动使用drop操作将选定的缓存排除在未来注意力之外，并使用repos操作来调整保留键的位置……

    arXiv:2610.28577v1 Announce Type: new  Abstract: Large Language Model (LLM) agents can handle increasingly complex tasks, navigating long trajectories of observations and actions. As these trajectories grow, key-value (KV) cache I/O and storage become the dominant cost. Trajectories can also exceed the LLM's maximum context length, requiring control over active cache growth. Prior work uses context engineering to reduce cache size by trimming or summarizing history. However, modifying the trajectory prefix invalidates subsequent cached states and triggers costly re-prefilling of retained context. Task accuracy can also suffer when too little context is retained to limit re-prefill costs. We propose cache engineering as a new scheme. It allows agents to directly control their active KV cache without modifying the trajectory or re-prefilling retained context. Specifically, agents can proactively use drop to exclude selected cache from future attention and repos to adjust retained keys' p
    
[^5]: 大型流语言模型

    Large Flow Language Models

    [https://arxiv.org/abs/2610.28559](https://arxiv.org/abs/2610.28559)

    提出了 Lafa——一个可扩展至 80 亿参数的分块流语言模型，通过改造预训练自回归模型实现高效训练，并借助前缀缓存与分块并行解码实现快速推理，在指令跟随、推理和编码任务上达到或超越自回归模型的性能。

    

    连续流语言模型为实现并行生成提供了一条路径，并具备离散扩散模型所不具备的控制与加速工具，然而在数十亿参数规模上构建此类模型的有效技术尚未充分确立。我们提出了 Lafa（Large flow language model，大型流语言模型），这是一个快速、高性能的流语言模型，以数据高效的方式扩展至 80 亿参数。Lafa 是一种分块流模型，在前缀部分采用因果注意力，并在每个块内进行双向、异步的流建模。这一设计使其能够通过改造预训练的自回归（AR）模型实现高效训练，并借助前缀缓存和分块并行解码实现快速推理。我们在 0.5B、7B 和 8B 规模上训练了 Lafa，使用的数据少于 300 亿 token，比以往的扩散模型更加高效；并在指令跟随、推理和编码任务上对其进行评估，结果表明学习到的流模型质量可匹敌甚至超越自回归模型，并且可以被蒸馏为流映射以……

    arXiv:2610.28559v1 Announce Type: cross  Abstract: Continuous flow language models offer a path to parallel generation with tools for control and acceleration unavailable to discrete diffusion, yet effective techniques for building them at the billion-parameter scale remain less established. We present Lafa (Large flow language model), a fast, high-performing flow language model scaled to 8B parameters in a data-efficient manner. Lafa is a blockwise flow model with causal attention on the prefix and bidirectional, asynchronous flow within each block. This design enables efficient training by adapting pretrained autoregressive (AR) models and fast inference via prefix caching and blockwise parallel decoding. We train Lafa at 0.5B, 7B, and 8B scales with fewer than 30B tokens, more efficient than previous diffusion models, and evaluate it on instruction following, reasoning, and coding, where the learned flow matches or surpasses the quality of AR and can be distilled into flow maps for 
    
[^6]: 二元离散扩散：面向逻辑与规划任务的候选集扩散

    Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks

    [https://arxiv.org/abs/2610.28543](https://arxiv.org/abs/2610.28543)

    该论文提出二元离散扩散，将扩散过程的含噪状态表示为每个位置候选词元集合的二进制隶属向量，通过添加与删除候选实现信息的渐进丢失与恢复，从而能显式建模部分不确定性并有效解决数独、规划等逻辑与推理任务。

    

    离散扩散模型通过迭代去噪由文本等分类数据构成的含噪序列，实现任意顺序的生成，并具备自校正与填充能力，这使其成为推理、规划以及代码、数学等受约束生成任务的有前景的候选方案。然而，掩码扩散和均匀扩散会在单次腐蚀事件中抹去词元的身份信息，因此其含噪状态无法显式表示那种多个备选词元仍然合理的部分解析不确定性。我们提出二元离散扩散来填补这一空白：这是一族连续时间离散扩散过程，其含噪状态是每个位置上的一组候选词元，以二进制隶属向量表示。二元扩散通过添加和删除操作来改变候选集，使信息能够逐渐丢失并逐渐恢复，同时扩散状态本身就表示部分不确定性。在Sudoku-Extreme、Boxoban和gr

    arXiv:2610.28543v1 Announce Type: cross  Abstract: Discrete diffusion models iteratively denoise a noisy sequence of categorical data such as text, enabling any-order generation with self-correction and in-filling capabilities. This makes them promising candidates for reasoning, planning, and constrained-generation tasks like coding and math. Yet masked and uniform diffusion erase a token's identity in a single corruption event, so their noisy states cannot explicitly represent partially resolved uncertainty in which several alternatives remain plausible. We close this gap with Binary Discrete Diffusion, a family of continuous-time discrete-diffusion processes whose noisy state is a set of candidate tokens per position, represented as a binary membership vector. Binary diffusion changes candidate sets through additions and removals, allowing information to be lost and recovered gradually while the diffusion state itself represents partial uncertainty. On Sudoku-Extreme, Boxoban, and gr
    
[^7]: BitMTP：当多词元预测遇上低比特大语言模型

    BitMTP: When Multi-Token Prediction Meets Low-Bit Large Language Models

    [https://arxiv.org/abs/2610.28500](https://arxiv.org/abs/2610.28500)

    提出BitMTP算法-系统协同设计框架，将多词元预测与超低比特大语言模型骨干相结合，使单一模型兼具自回归生成与块因果草稿生成能力，在保持同等生成质量的前提下实现高效解码加速。

    

    摘要（arXiv:2610.28500v1，公告类型：cross）：为了实现大语言模型（LLM）的高效部署，超低比特骨干网络方法可以降低权重存储、内存流量和线性层执行成本，而多词元预测（MTP）方法通过并行预测多个未来词元来减少顺序解码步骤。尽管这两种方法互补，但直接结合它们无法在同等质量下加速解码：掩码草稿生成缺少下一个词元训练中所提供的中间词元输入，同时更宽的计算和因果键值（KV）更新可能会抵消更少解码步骤带来的节省。为了解决这些挑战，我们提出BitMTP，一个用于在超低比特LLM骨干网络上实现高效MTP的算法-系统协同设计框架。通过联合下一个词元预测与MTP的训练目标，单个模型同时具备自回归（AR）生成和块因果草稿生成能力。仅置信度解码、仅投机解码和混合解码提供了不同的……（原文摘要在此处截断）

    arXiv:2610.28500v1 Announce Type: cross  Abstract: For efficient deployment of large language models (LLMs), methods of ultra-low-bit backbones reduce weight storage, memory traffic, and linear-layer execution cost, while multi-token prediction (MTP) methods reduce sequential decoding steps by predicting multiple future tokens in parallel. Although complementary, the direct combination of these methods cannot accelerate decoding at comparable quality: masked drafting lacks intermediate token input supplied in next-token training, while wider computation and causal key-value (KV) updates can offset the savings from fewer steps. To address these challenges, we propose BitMTP, an algorithm-system co-design framework for efficient MTP on ultra-low-bit LLM backbones. A joint next-token prediction and MTP objective equips a single model with both autoregressive (AR) generation and block-causal drafting. Confidence-only decoding, speculative-only decoding, and hybrid decoding provide differen
    
[^8]: 存在但难以读取：用最小提示对探查潜在推理

    Present but Hard to Read: Probing Latent Reasoning with Minimal Pairs

    [https://arxiv.org/abs/2610.28498](https://arxiv.org/abs/2610.28498)

    论文通过仅有一个数字不同的最小提示对证明，从输入复制的信息确实以跨问题共享的编码存在于模型状态中，但常规单样本探针难以读取它，因此探查失败不能证明信息缺失，并提出了一套区分“计算出的值”与“输入副本/混合”的探查协议。

    

    潜在推理的探查研究通常在答案开始生成的位置读取模型状态，并将探查失败当作该信息不存在的证据。我们用一个从问题中复制的数字来检验这一点，实验对象是循环模型 Ouro-1.4B 和标准 Transformer Qwen2.5-1.5B。比较仅在该数字上不同的提示对可以发现，该数字确实存在于最后一个提示词元中，并以一种跨问题共享的编码形式存在（读取准确率约 0.9）。而一次只看单个提示的常规探针几乎读不出它：即使训练了 5,249 个问题，Ouro 中仅达 0.36，Qwen 中仅达 0.19。该数字引起的状态变化是“微小”而非“拥挤”的：随机噪声会落在同样繁忙的方向上，而用一个随机词替换该数字会使状态发生大得多的位移且可被读取。由于每个变化的词元都会留下痕迹，成功的探查结果同样存在歧义。我们提出了一套探查协议，包括用于检验一般性混合效应的合成零假设，用于将模型计算出的值与输入的副本及混合信息进行区分测试。

    arXiv:2610.28498v1 Announce Type: cross  Abstract: Probing studies of latent reasoning often read a model's state where the answer begins and take a failed probe as evidence that a value is absent. We test this with a digit copied from the question, in the looped model Ouro-1.4B and the standard transformer Qwen2.5-1.5B. Comparing prompts that differ only in that digit shows it is present at the last prompt token, in a code shared across questions (readout about 0.9). A probe that sees one prompt at a time, the usual setting, barely reads it: 0.36 in Ouro and 0.19 in Qwen even after 5,249 training questions. The digit's change is small rather than crowded: random noise lands on equally busy directions, while a random word in its place moves the state far more and is readable. Since every varied token leaves a trace, a successful probe is ambiguous too. We give a protocol, including a synthetic null for generic mixing, for testing computed values against copies and mixtures of the input
    
[^9]: OptionWorld：一个用于诊断大语言模型作为长期个人助手失败原因的基准测试

    OptionWorld: A Benchmark for Diagnosing Failures of LLMs as Long-Term Personal Assistants

    [https://arxiv.org/abs/2610.28491](https://arxiv.org/abs/2610.28491)

    论文提出OptionWorld基准，通过追踪模拟日常任务中的信息获取、判断、行动与结果，诊断出LLM作为长期个人助手失败的三大主因：所需信息未获取或整合、可用的行动机会未被利用、以及模型施加了任务并不需要的额外限制。

    

    长期助手必须随着信息和环境的变化，跨越数天持续跟进未完成的任务。然而，找到信息或开始行动并不能确保任务最终被完成。我们提出了OptionWorld，这是一个通过追踪模拟日常生活任务中的信息获取、判断、行动和结果，来定位当前大语言模型作为长期个人助手时表现不足之处的基准测试。在使用存档决策输入的运行中，在35.5%的失败案例中，关键事实其实仍存在于截止时间前模型的最后一次输入之中。失败分析识别出三个最常见的问题：所需信息未被获取或整合、可用的行动机会未被利用、以及智能体施加了任务本身并不需要的限制。三项后续实验表明：提供缺失的事实可以在某些情况下恢复任务成功；行动提醒可以重新启动工作，但后续步骤仍会错过截止时间；而……

    arXiv:2610.28491v1 Announce Type: new  Abstract: Long-term assistants must follow unfinished tasks across days as information and circumstances change. Finding information or starting an action, however, does not ensure that the task gets completed. We introduce OptionWorld, a benchmark that traces information acquisition, judgments, actions, and outcomes in simulated daily-life tasks to identify where current LLMs fall short as long-term personal assistants. In runs with archived decision inputs, the key facts remain in the model's last input before the deadline in 35.5% of failed instances. Failure analysis identifies the three most frequent problems: needed information is not obtained or combined, available opportunities to act go unused, and agents impose restrictions that the task does not require. Three follow-up experiments show that supplying missing facts restores task success in some cases; action reminders can restart work while subsequent steps still miss the deadline; and 
    
[^10]: 非对称循环Transformer：面向迭代解码器的固定全局表示

    Asymmetric Looped Transformers: Fixed Global Representations for an Iterative Decoder

    [https://arxiv.org/abs/2610.28489](https://arxiv.org/abs/2610.28489)

    提出非对称循环Transformer（AsymLoop），将单次上下文编码与循环状态精炼解耦，利用迭代过程中保持固定的全局KV缓存与循环解码器生成的局部KV进行联合注意力，从而在不增加参数量的情况下提升模型的有效深度。

    

    循环Transformer通过重复应用共享层来增加有效深度，提供了一种在不增加参数量的情况下扩展计算规模的途径。然而，全栈循环将上下文构建和迭代状态精炼置于同一个循环计算之中，没有明确区分两者的作用。我们提出了非对称循环Transformer（AsymLoop），它将单次通过的上下文编码与循环状态精炼分离开来。一个单次通过的自回归编码器生成各层特定的全局键值缓存，这些缓存在整个迭代过程中保持固定；而一个权重绑定的循环解码器则从其不断演进的状态中生成局部键值。在单一softmax下对这两个来源进行联合注意力，使每次更新都能将持久的上下文信息与当前的中间状态整合起来。在训练方面，我们对编码器和解码器应用了一种重尾引导的逐层学习率方法。在FineWeb上的实验……

    arXiv:2610.28489v1 Announce Type: new  Abstract: Looped Transformers increase effective depth by repeatedly applying shared layers, offering a way to scale computation without increasing parameter count. However, full-stack looping places context construction and iterative state refinement within the same recurrent computation, without explicitly distinguishing their roles. We introduce the Asymmetric Looped Transformer (AsymLoop), which separates single-pass context encoding from recurrent state refinement. A single-pass causal encoder produces layer-specific Global key-value (KV) caches that remain fixed across iterations, while a weight-tied recurrent decoder generates Local KV from its evolving state. Joint attention over both sources under a single softmax allows each update to integrate persistent contextual information with the current intermediate state. For training, we apply a heavy-tail-guided layerwise learning-rate method to the encoder and decoder. Experiments on FineWeb-
    
[^11]: GPT-6-Astra 理解触觉并超越视觉-触觉-语言-动作模型

    GPT-6-Astra Understands Touch and Outperform Vision--Tactile--Language--Action Models

    [https://arxiv.org/abs/2610.28457](https://arxiv.org/abs/2610.28457)

    提出将 GPT-6-Astra 直接作为触觉策略的 GPT-as-Tactile-Policy 方法，通过将专家演示总结为任务指导与触觉参考，并结合多模态观察与交互历史生成控制指令，无需专门的预训练与逐任务优化即可实现接触丰富的操作并超越现有的视觉-触觉-语言-动作模型。

    

    通用多模态模型作为直接的机器人动作策略已展现出前景，但它们是否真正掌握了现实世界交互背后的物理规律，并能否将这种理解应用于指导动作，仍然缺乏充分探索。我们以接触丰富的操作作为代表性测试平台来研究这一问题，将触觉理解作为这一更广泛能力的具体实例。接触丰富的操作需要解读触觉反馈并在受阻后调整动作。现有的学习型触觉策略需要通过专门的预训练和逐任务优化来获得这些能力，伴随着大量的数据和训练成本。我们提出了 GPT-as-Tactile-Policy（GPT 作为触觉策略）方法，将专家演示总结为任务指导和触觉参考，用于接触评估和纠正性动作。GPT-6-Astra 将这些上下文信息与多模态观察和交互历史相结合，以生成控制指令并……（原文摘要到此截断）

    arXiv:2610.28457v1 Announce Type: cross  Abstract: General-purpose multimodal models have shown promise as direct robot action policies, yet whether they grasp the physics underlying real-world interactions and can apply that understanding to guide action remains underexplored. We investigate this question using contact-rich manipulation as a representative testbed, examining tactile understanding as a concrete instance of this broader capability. Contact-rich manipulation requires interpreting tactile feedback and adjusting actions after obstruction. Existing learned tactile policies acquire these capabilities through specialized pretraining and per-task optimization, with substantial data and training costs. We present GPT-as-Tactile-Policy, which summarizes expert demonstrations into task guidance and tactile references for contact assessment and corrective action. GPT-6-Astra combines this context with multimodal observations and interaction history to generate control commands and
    
[^12]: 工具通胀：卖家智能体通过自发性生成式引擎优化扭曲工具市场

    Toolflation: Seller Agents Distort Tool Markets Through Spontaneous Generative Engine Optimization

    [https://arxiv.org/abs/2610.28454](https://arxiv.org/abs/2610.28454)

    卖家AI智能体仅被要求“更常被选中”时就会自发进行生成式引擎优化，夸大其工具的能力描述，形成不受市场约束的“工具通胀”现象，从而扭曲智能体间的工具市场，且其严重程度在不同模型间相差可达一个数量级。

    

    智能体（AI代理）开始向其他智能体进行销售。卖家智能体负责撰写工具、服务和产品的商品列表，而买家智能体在选择使用什么时会阅读这些列表。我们证明，仅被告知要更频繁地被选中的卖家智能体会自发地进行生成式引擎优化，而这会扭曲卖家和买家双方的市场。我们基于Amazon和Instagram上的真实数据工具构建了工具市场，并对每个问题运行每个工具，从而准确知道哪些工具能够回答每个问题。在我们测试的所有五个生产级模型中，卖家都会重写其描述以声称越来越多的字段，其中包括其工具在不到一半的目标上才会返回的字段。我们将这一过程称为“工具通胀”。市场中没有任何机制能够限制它，其蔓延程度取决于所用模型——与卖家自身证据相矛盾的声明数量在不同模型之间相差一个数量级。工具通胀改变了买家的……（原文摘要在此处截断）

    arXiv:2610.28454v1 Announce Type: new  Abstract: Agents are starting to sell to other agents. Seller agents write the listings of tools, services and products, and buyer agents read these listings when they choose what to use. We show that seller agents told only to be selected more often practice generative engine optimization on their own, and that this distorts the market for sellers and for buyers. We build tool markets from real data tools on Amazon and Instagram and run every tool on every question, so we know which tools can answer each question. Under all five production models we test, sellers rewrite their descriptions to claim more and more fields, including fields that their tools return for fewer than half of the targets. We call this process toolflation. Nothing in the market limits it, and how far it goes depends on the model, as the number of claims that contradict the sellers' own evidence differs by an order of magnitude between models. Toolflation shifts the buyers' 
    
[^13]: MythoSemantic：一个用于发现神话与宗教传统之间语义平行关系的交互式系统

    MythoSemantic: An Interactive System for Discovering Semantic Parallels Across Mythological and Religious Traditions

    [https://arxiv.org/abs/2610.28452](https://arxiv.org/abs/2610.28452)

    该论文提出MythoSemantic交互式系统，利用神经语言模型将神话与宗教文本嵌入共享语义空间，使用户能够跨文化传统发现语义平行关系，且嵌入结构形成的语义几何形态与传统间的历史关系相吻合。

    

    我们提出了一种用于神话与宗教文本比较研究的工具。传统的人工分析高度依赖人力且难以扩展。为解决这一问题，我们使用神经语言模型将文本嵌入到共享语义空间中，使用户能够在交互式界面中发现不同文化传统之间的语义平行关系。嵌入的文本片段形成了独特的语义结构，其粗略的几何形态与文献中充分记录的各传统之间的历史关系相一致。我们量化了这种几何结构中有多少已经可以通过词汇重叠和专有名词来解释。当前系统可在 https://mythoscope.io 访问，可安装的软件包可在 https://github.com/haafus/MythoSemantic/blob/main/RUN.md 获取。

    arXiv:2610.28452v1 Announce Type: new  Abstract: We present a tool for the comparative study of mythological and religious texts. Traditional manual analysis is highly labor-intensive and difficult to scale. To address this, we use neural language models to embed texts into a shared semantic space, so that users can find semantic parallels across cultural traditions in an interactive interface. Embedded fragments form distinct semantic structures whose coarse geometry aligns with well-documented historical relations between traditions. We quantify how much of that geometry is already explained by lexical overlap and by proper names. The current system is available at https://mythoscope.io, and the installable package is available at https://github.com/haafus/MythoSemantic/blob/main/RUN.md.
    
[^14]: 每个模型都会留下痕迹：多分辨率大语言模型指纹识别

    Every Model Leaves a Trace: Multi-Resolution LLM Fingerprinting

    [https://arxiv.org/abs/2610.28434](https://arxiv.org/abs/2610.28434)

    该论文提出多分辨率大语言模型指纹识别框架，证明即使表层线索被抑制，LLM指纹在蒸馏模型乃至同一基座的LoRA适配模型中依然可检测，并能跨提示分布和语言迁移。

    

    大语言模型在其生成的文本中会留下模型特有的痕迹，但这些指纹能够以多精细的粒度区分相关模型，以及随着明显线索的减少哪些信号仍然具有信息量，目前尚不清楚。我们通过指纹分辨率来研究这一问题，既考虑模型分辨率（从不同模型家族到蒸馏和适配的变体），也考虑信号分辨率（涵盖行为、文体、词汇和语义层面的证据）。我们逐步抑制表层线索、对输出进行改写、改变可用文本的数量，并测试在提示分布和语言之间的泛化能力。我们的结果表明，指纹超越广泛的家族差异而持续存在，在蒸馏模型中仍可检测到，甚至在同一基础模型的LoRA适配版本中也可检测到。尽管随着表层和词汇线索被改变，归因能力会有所减弱，但仍然存在大量模型特有的信号。这些信号可以在不同的提示来源和语言之间迁移。

    arXiv:2610.28434v1 Announce Type: cross  Abstract: Large language models leave model-specific traces in their generated text, but it remains unclear how finely these fingerprints can distinguish related models and which signals remain informative as obvious cues are reduced. We study this through fingerprint resolution, considering both model resolution, from different families to distilled and adapted variants, and signal resolution across behavioral, stylometric, lexical, and semantic evidence. We progressively suppress surface cues, paraphrase outputs, vary the amount of available text, and test generalization across prompt distributions and languages. Our results show that fingerprints persist beyond broad family differences, remaining detectable in distilled models and even LoRA adaptations of the same base model. Although attribution weakens as surface and lexical cues are altered, substantial model-specific signal remains. These signals transfer across prompt sources and languag
    
[^15]: 事实回忆中的指令税：基础模型可以胜过其指令微调版本

    The Instruction Tax on Factual Recall: Base Models Can Beat Their Instruction-Tuned Counterparts

    [https://arxiv.org/abs/2610.28407](https://arxiv.org/abs/2610.28407)

    该论文发现指令微调会降低语言模型的事实回忆能力（即“指令税”），通过简单的文本补全方式提示基础模型并仅保留重复输出，在AKBC LM-KBC 2026共享任务上取得了超过基线两倍的0.6836分。

    

    我们展示了针对AKBC LM-KBC 2026共享任务的解决方案，在官方排行榜上取得了0.6836的分数，是0.2964基线的两倍多。该任务要求使用参数量不超过320亿的语言模型，在不进行检索或微调的情况下，为主体-关系对生成所有可能的宾语对象。我们表明，提示方法与模型架构同样重要：指令微调版本的模型通过其聊天界面只能达到0.5681，而基础模型在使用文本补全提示时能达到0.6836。因此，指令微调会降低事实回忆能力；我们将这一效应称为“指令税”，它由模型的权重本身导致，而非提示方法所致。我们的最终解决方案相当简单直接：用训练数据中的一些示例提示基础模型，生成64至256个输出，并只保留其中重复出现的内容，此外还使用了两种专门的方法。

    arXiv:2610.28407v1 Announce Type: cross  Abstract: We present our solution to the AKBC LM-KBC 2026 Shared Task, where we achieved a score of 0.6836 on the official leaderboard, more than double the baseline of 0.2964. This task requires generating all possible objects for subject-relation pairs using a language model with up to 32 billion parameters, with no retrieval or fine-tuning involved. We show that the prompting methodology is as important as the model architecture: while the instruction-tuned model version achieves 0.5681 with its chat interface, the base model achieves 0.6836 when prompted using text completion. Thus, instruction tuning decreases factual recall; we refer to this effect as the instruction tax, which is caused by the model's weights rather than the prompting method. Our final solution is fairly straightforward: we prompt the base model with some examples from the training data, generate 64--256 outputs, and retain only the repeated ones. Two specialized methods 
    
[^16]: MAS-Guard：从单个智能体到多智能体大语言模型系统的统一防护框架

    MAS-Guard: Unified Guardrails from Individual Agents to Multi-Agent LLM Systems

    [https://arxiv.org/abs/2610.28397](https://arxiv.org/abs/2610.28397)

    该论文提出统一安全防护框架MAS-Guard，通过本地智能体防护层（A-Guard）等机制，为包含可插桩与不透明智能体的单智能体和多智能体LLM系统，提供覆盖本地执行与系统级交互的一体化安全保障。

    

    智能体AI将大语言模型（LLM）从生成响应扩展到规划与执行复杂工作流，在大幅提升其效用的同时，也使应用面临不安全行为、信息泄露和对抗性操纵等风险。多智能体系统通过专业化分工与协作进一步扩展了这些能力，但权限分散、角色异构以及参与者不透明或可能已被攻陷等因素，使得安全防护变得更具挑战性。现有防御手段虽能应对重要的单一威胁，但在本地执行、系统级交互和执行机制方面的覆盖是碎片化的。我们提出MAS-Guard，这是一个统一框架，为包含可插桩智能体和不透明智能体的单智能体及多智能体AI应用提供安全保障，填补了这一空白。在可插桩智能体中，所提出的本地智能体防护层（A-Guard）应用任务特定的专家来评估……

    arXiv:2610.28397v1 Announce Type: cross  Abstract: Agentic AI extends large language models (LLMs) from generating responses to planning and executing complex workflows, substantially expanding their utility while exposing applications to unsafe actions, information leakage, and adversarial manipulation. Multi-agent systems further expand these capabilities through specialization and collaboration, but distributed authority, heterogeneous roles, and opaque or potentially compromised participants make safeguarding even more challenging. Existing defenses address important individual threats but provide fragmented coverage across local execution, system-level interactions, and enforcement mechanisms. We introduce MAS-Guard, a unified framework that closes this gap for safeguarding single- and multi-agent AI applications containing both instrumentable and opaque agents. Within instrumentable agents, the proposed local Agent Guardrail (A-Guard) layer applies task-specific experts to assess
    
[^17]: 当复制变得困难：面向精确片段复现的复制约束解码

    When Copying Is Hard: Copy-Constrained Decoding for Exact Span Reproduction

    [https://arxiv.org/abs/2610.28393](https://arxiv.org/abs/2610.28393)

    该论文提出了CopyGen——一个面向冻结大语言模型的轻量级复制约束解码插件，通过将复制分解为“何时复制、复制多远”等显式决策，显著提升模型在长上下文中精确逐字复现目标片段的能力。

    

    在长上下文中精确地逐字复现目标片段，是对大语言模型的一项基本能力测试：即使目标字符串就存在于上下文中，模型仍然常常在片段内部发生漂移，或无法正确停止，暴露出其在来源对齐与边界控制方面的局限。这一能力具有广泛的实际意义，例如代码生成往往需要将API密钥等敏感字符串按上下文中出现的原样精确嵌入。在本工作中，我们构建了一个基准，在合同、科学文献、代码和随机序列等场景上测试精确片段复现能力，发现当前模型仍然会犯出人意料地频繁的错误。为解决这一问题，我们提出了CopyGen，一个面向冻结大语言模型的轻量级复制约束插件，它将复制从下一个词预测的副产物转变为一个受控过程。我们的核心思想是将复制分解为三个不同的决策：何时复制、复制多远……

    arXiv:2610.28393v1 Announce Type: new  Abstract: Precisely reproducing a target span verbatim within a long context is a fundamental capability test for LLMs: even when the target string is present in context, models often still drift within the span or fail to stop correctly, revealing limitations in source grounding and boundary control. This capability has broad practical relevance. For instance, code generation often requires embedding sensitive strings such as API keys exactly as they appear in context. In this work, we construct a benchmark to test exact span reproduction ability across contracts, scientific literature, code, and random sequences, and find that current models still make surprisingly frequent errors. To address this, we propose CopyGen, a lightweight copy-constrained plugin for frozen LLMs that turns copying from a byproduct of next-token prediction into a controlled process. Our key idea is to decompose copying into three distinct decisions: when to copy, how far
    
[^18]: CAPSafe：面向具身智能体上下文动作与计划安全推理的基准测试

    CAPSafe: A Benchmark for Contextual Action and Plan Safety Reasoning in Embodied Agents

    [https://arxiv.org/abs/2610.28381](https://arxiv.org/abs/2610.28381)

    该论文提出了CAPSafe基准，从动作级和计划级两个层面评估具身智能体的上下文安全推理能力，发现LLM虽在危险识别上准确率可超90%，但在安全长时程规划方面仍存在显著差距。

    

    对于由大型语言模型（LLM）驱动的具身智能体而言，安全推理至关重要。我们提出了上下文动作与计划安全基准测试CAPSafe，旨在从两个层面评估安全推理能力：（1）动作级数据集，用于在给定环境上下文的情况下测试单个动作是否安全；（2）计划级数据集，用于评估智能体能否为复杂的高危家庭任务生成安全的长时程计划。CAPSafe建模的安全条件具有如下特点：危险是情境性的（需从物体属性推断得出），多个危险可能在同一环境中同时出现，且每个危险都标注了相对严重程度以便进行细粒度分析。对九个闭源和开源LLM的评估揭示了危险识别与安全长时程规划之间存在显著差距。若干模型在识别不安全动作方面达到了很高的准确率（最佳情况下超过90%），但在安全的长时程规划方面……

    arXiv:2610.28381v1 Announce Type: cross  Abstract: Reasoning about safety is essential for embodied agents powered by large language models (LLMs). We introduce the Contextual Action and Plan Safety benchmark (CAPSafe), designed to evaluate safety reasoning at two levels: (1) an action-level dataset that tests whether a single action is safe given its environmental context, and (2) a plan-level dataset that evaluates whether agents can generate safe long-horizon plans for complex hazardous household tasks. CAPSafe models safety conditions where hazards are situational (inferred from object properties), multiple hazards can co-occur within the same environment, and each is annotated with a relative severity level for fine-grained analysis. Evaluating nine closed and open-source LLMs reveals a substantial gap between hazard recognition and safe long-horizon planning. Several models achieve high accuracy in recognizing unsafe actions (exceeding 90% in the best cases), yet safe long-horizo
    
[^19]: 面向对象执行框架：通过面向对象设计演化智能体执行框架

    Object-Oriented Harness: Evolving Agent Harnesses through Object-Oriented Design

    [https://arxiv.org/abs/2610.28380](https://arxiv.org/abs/2610.28380)

    提出OO-Harness框架，利用面向对象设计中的继承机制演化出专门化执行框架的类层次结构，并通过路由器为不同任务选择最优框架，在三个长程基准测试中相比基线最高提升15%。

    

    智能体执行框架是围绕语言模型指定提示词、工具和控制流的可执行代码。单一的全局框架在处理需要不同执行策略的异构任务时可能表现不佳。在这项工作中，我们提出了面向对象执行框架（OO-Harness），这是一个利用面向对象设计演化出可执行框架类层次的框架优化方法。在训练过程中，一个演化智能体利用执行记录创建一系列专门化的框架，每个框架都通过继承实现为基类或现有专门化框架的子类。在测试时，一个框架路由器利用训练期间积累的路由经验为每个任务选择特定的框架。在涵盖业务工作流、工作区任务和工具使用的三个长程基准测试中，OO-Harness在对比方法中取得了最佳性能，相比基线实现了最高达15%的绝对提升。

    arXiv:2610.28380v1 Announce Type: new  Abstract: Agent harnesses are executable code that specifies the prompts, tools, and control flow around a language model. A single global harness can struggle with heterogeneous tasks that require different execution strategies. In this work, we introduce Object-Oriented Harness (OO-Harness), a harness optimization framework that evolves a hierarchy of executable harness classes using object-oriented design. During training, an evolution agent uses execution records to create a family of specialized harnesses, each implemented as a subclass of the base or an existing specialized harness via inheritance. At test time, a harness router uses routing lessons accumulated during training to select a specific harness for each task. Across three long-horizon benchmarks spanning business workflows, workspace tasks, and tool use, OO-Harness achieves the best performance among the compared methods, with an absolute improvement of up to 15% over the base. Ex
    
[^20]: DeclineCraft：一个用于联合衡量检索知识与参数知识组合的RAG弃答基准

    DeclineCraft: A RAG Abstention Benchmark for Jointly Measuring Combined Retrieval and Parametric Knowledge

    [https://arxiv.org/abs/2610.28340](https://arxiv.org/abs/2610.28340)

    该论文提出了DeclineCraft——一个基于Minecraft语料库截止日期的动态RAG弃答基准，发现单次检索能显著提升模型在不可回答问题上的正确弃答率，而允许模型自主多轮检索的迭代式RAG反而会降低弃答能力。

    

    可靠的检索增强生成（RAG）需要模型能够识别何时检索到的文档和参数知识都无法支持答案。我们提出了DeclineCraft，一个动态弃答基准，其不可回答的问题针对语料库截止日期之后Minecraft游戏中新增的功能特性，并对可能存在的参数化知识暴露进行了审计。较晚的语料库截止日期还可支持对未来模型的评估。我们在27个大语言模型上测量了正确弃答率（CAR；即模型在不可回答问题上承认知识不足的频率）、可回答问题尝试率以及答案准确率。我们发现：（1）与无检索相比，单次检索在40个配置（模型与思考模式设置）中的38个里显著提高了CAR；（2）迭代式RAG——即模型对自身上下文进行自我评估并可请求最多两轮额外检索——在40个配置中的30个里相比单次RAG显著降低了CAR（平均46.4%），而答案准确率几乎没有变化。

    arXiv:2610.28340v1 Announce Type: new  Abstract: Reliable retrieval-augmented generation (RAG) requires recognizing when neither retrieved documents nor parametric knowledge supports an answer. We introduce DeclineCraft, a dynamic abstention benchmark whose unanswerable questions target Minecraft feature additions after a corpus cut-off, with audits of possible parametric exposure. Later cut-offs support evaluation of future models. Across 27 LLMs, we measure Correct Abstention Rate (CAR; how often models acknowledge insufficient knowledge on unanswerable questions), Answerable Attempt Rate and Answer Accuracy. We find that (1) single-shot retrieval significantly raises CAR in 38 of 40 configurations (model and thinking setting) versus no retrieval. (2) Iterative RAG, in which the model self-assesses its context and can request up to two further retrieval rounds, significantly lowers CAR versus single-shot RAG in 30 of 40 configurations (mean 46.4%), with little change in Answer Accura
    
[^21]: 超越单一情感分数：Twitter上表情符号意义的主题与情感维度

    Beyond a Single Sentiment Score: Topical and Affective Dimensions of Emoji Meaning on Twitter

    [https://arxiv.org/abs/2610.28337](https://arxiv.org/abs/2610.28337)

    研究发现，在Twitter上主题语境越聚集的表情符号，其众包情感分数反而越两极化，表明表情符号的意义无法用单一情感分数概括，需要同时考虑主题与情感两个维度。

    

    表情符号通常被视作紧凑的情感符号，其意义由单一情感分数来概括。我们认为这种看法并不完整。基于一个包含16,575条英文推文的语料库，以及我们提出的针对每个表情符号的度量指标——“稳定性差值”（stability delta，即句子嵌入的类内余弦相似度减去类间余弦相似度），我们发现在主题高度聚集的语境中使用的表情符号，其众包情感反而更加两极化，而非更弱。以Kralj Novak等人（2015）的情感词典进行验证，核心相关性为Spearman ρ = +0.361（Holm校正后p = 0.015，95%自助法置信区间[+0.10, +0.59]）；以独立的Ferré等人（2025）西班牙语13种情感评分为参照，ρ = +0.509（p = 0.0003）。该结果通过了置换检验、BCa自助法、留一法折刀检验、控制表情符号使用频率的偏相关分析，以及跨四项检验族的Holm校正。主题凝聚性与情感……（摘要原文在此处截断）

    arXiv:2610.28337v1 Announce Type: cross  Abstract: Emoji are widely treated as compact affective tokens whose meaning is summarised by a single sentiment score. We argue that this picture is incomplete. From a corpus of 16{,}575 English tweets and a per-emoji metric we call \emph{stability delta} (within-class minus between-class cosine similarity of sentence embeddings), we show that emojis used in tightly clustered topical contexts also exhibit \emph{more} polarised crowd-sourced sentiment, not less. Validated against the Kralj Novak et al.\ (2015) sentiment lexicon, the central correlation is Spearman $\rho = +0.361$ (Holm $p = 0.015$, 95\% bootstrap CI $[+0.10, +0.59]$); against the independent Ferr\'{e} et al.\ (2025) Spanish 13-emotion ratings, $\rho = +0.509$ ($p = 0.0003$). The result survives permutation testing, BCa bootstrap, leave-one-out jackknife, partial correlation controlling for emoji frequency, and Holm correction across a four-test family. Topical coherence and affe
    
[^22]: 披露引发披露：具有持久记忆的LLM智能体中的自我强化隐私循环

    Disclosure Begets Disclosure: A Self-Reinforcing Privacy Loop in LLM Agents with Persistent Memory

    [https://arxiv.org/abs/2610.28315](https://arxiv.org/abs/2610.28315)

    该研究揭示了持久记忆型LLM智能体中存在自我强化的隐私泄露循环：基于记忆的响应会将早期属性值作为预设延续，用户确认或不加纠正，导致属性泄露随会话次数持续扩大。

    

    大语言模型智能体越来越多地使用持久记忆来个性化响应，而记忆中存储的个人属性一旦暴露便构成隐私风险。我们探究在对话中查阅记忆如何改变用户后续的发言内容，进而改变记忆最终会揭示的信息。在多会话模拟中，一个拥有24个个人属性的模拟用户与一个查阅或不查阅记忆的客服智能体对话，并在两种设置下均从完整累积的对话记录中推断这些属性。在四个客服模型上，使用记忆提升了响应质量，但同时也增加了经验性泄露——即从记录中能被正确推断出的属性值比例——且这一差距随会话次数增加而不断扩大。其根源在于普通的、非对抗性辅助过程中存在的一个循环：基于记忆的响应会将早期的属性值延续下去（常以预设的形式出现），而用户会确认这些值或不予纠正，从而使线索和智能体自身的猜测都被记录下来。

    arXiv:2610.28315v1 Announce Type: cross  Abstract: Large language model (LLM) agents increasingly use persistent memory to personalize responses, and personal attributes in this memory pose a privacy risk if exposed. We ask how consulting memory during dialogue changes what users go on to say, and thus what the memory comes to reveal. In multi-session simulations, a simulated user with 24 personal attributes talks with a support agent that does or does not consult memory, and in both settings attributes are inferred from the full accumulated record. Across four support models, memory raises response quality but also empirical leakage, the share of attribute values correctly inferred from the record, by a margin that widens over sessions. The cause is a loop in ordinary, non-adversarial assistance. Memory-conditioned responses carry earlier values forward, often as presuppositions, and users affirm them or leave them uncorrected, so that cues and the agent's own guesses become recorded 
    
[^23]: 从抽象到行动：用于自然语言抽象解释与执行的领域特定语言（DSL）

    From Abstraction to Action: DSLs for Natural Language Abstraction Interpretation and Execution

    [https://arxiv.org/abs/2610.28311](https://arxiv.org/abs/2610.28311)

    本文提出借鉴编程领域的领域特定语言（DSL）作为显式抽象表示层，使语言模型能够识别高抽象层次的自然语言指令并将其展开为具体可执行的动作。

    

    自然语言指令通常依赖于高层次的抽象，例如让智能体“把积木堆成金字塔”。解释抽象指令需要模型既能识别说话者所意图的抽象，又能将其展开为实现该抽象的许多具体动作。语言模型在这方面表现不佳：先前的研究表明，随着指令抽象层次的提高，模型性能会下降。为了弥合自然语言抽象与具体动作之间的差距，我们提出借鉴编程环境中的领域特定语言（DSL）概念——这是一种显式表示层，其文档化的原语为反复出现的抽象命名，并可将这些抽象展开为具体动作。我们在 Hexagons 上展示了 DSL 在自然语言解释方面的有效性，这是一个按抽象程度分级、以视觉为基础的指令遵循任务。我们进一步探究语言模型能否自行归纳出这样的 DSL。

    arXiv:2610.28311v1 Announce Type: new  Abstract: Natural-language instructions often rely on high-level abstractions, such as asking an agent to ``stack blocks in a pyramid.'' Interpreting abstract instructions requires models to both identify the abstraction intended by the speaker, and unroll it into the many concrete actions that instantiate it. Language models struggle here: prior work has shown that performance degrades as the level of abstraction in the instruction rises. To bridge the gap between natural language abstraction and concrete actions, we propose to borrow from coding environments the notion of a {\em domain-specific language} (DSL), an explicit representation layer whose documented primitives name recurring abstractions that expand into concrete actions. We demonstrate the efficacy of DSLs for NL interpretation on \textsc{Hexagons}, a visually-grounded instruction-following task graded for abstraction. We further ask whether language models can induce such DSLs thems
    
[^24]: 奖励错误设定会降低思维链的可监控性

    Reward Misspecification Degrades Chain-of-Thought Monitorability

    [https://arxiv.org/abs/2610.28303](https://arxiv.org/abs/2610.28303)

    该研究证明，强化学习中的奖励错误设定本身就足以降低语言模型思维链的可监控性——随着模型奖励破解率上升，其思维链仍在推理真实任务意图，而输出却违背了任务意图。

    

    监控语言模型（LM）的思维链（CoT）是检测和防止不当行为、保障AI安全的关键策略。因此，识别出会导致思维链变得无法监控的强化学习（RL）实践至关重要。而其中一种实践可能难以避免：奖励错误设定。随着强化学习环境变得更加复杂，实现完美验证是不切实际的，模型会因与真实任务意图不同的行为而获得奖励。我们证明，仅奖励错误设定本身就足以降低可监控性。我们构建了一个测试平台，其中包含与任务意图对齐的真实奖励和偏向另一种行为的错误设定奖励的配对，从而可以通过训练诱导奖励破解并进行可验证的标注。在三个模型和三个任务上，随着模型破解率的上升，思维链可监控性始终处于较低水平：思维链仍在围绕真实任务意图进行推理，而输出却违背了任务意图。

    arXiv:2610.28303v1 Announce Type: new  Abstract: Monitoring the chain-of-thought (CoT) of language models (LMs) is a key strategy for detecting and preventing misbehaviors for AI safety. As such, it is critical to identify reinforcement learning (RL) practices that lead to unmonitorable chain-of-thought. One such practice may be unavoidable: reward misspecification. As RL environments become more complex, it is infeasible to have perfect verification, and models are rewarded for behaviors that are different from the true task intent. We show that reward misspecification alone already degrades monitorability. We create a testbed with pairs of true reward aligned with task intent and misspecified rewards that favor a different behavior, so that reward hacking can be induced by training and labeled verifiably. Across three models and three tasks, CoT monitorability remains low as models' hack rates increase: the CoTs keep reasoning about the task intent while the outputs violate the inten
    
[^25]: 自适应重放优化：在修订与重采样之间分配测试时计算

    Adaptive Replay Refinement: Allocating Test-Time Compute Between Revision and Resampling

    [https://arxiv.org/abs/2610.28294](https://arxiv.org/abs/2610.28294)

    ARR 是一种无需训练的自适应方法，通过检测解的修订是否达到不动点，在“继续改进现有解”与“重新生成新解”之间动态分配测试时计算，从而在多个模型和基准上以同等计算量达到与多数投票相当的效果。

    

    测试时扩展将额外的推理计算用于优化现有解或生成新解。这引出了一个根本性问题：应该继续改进当前解，还是停止改进并尝试一个全新的解？现有方法依赖固定的优化调度、自我验证判断或学习到的控制器来做出这一决策，这可能会浪费计算，或依赖可能不正确、需要额外训练的信号。我们提出自适应重放优化，这是一种无需训练的方法，能够检测平稳修订何时达到不动点，并将释放出来的计算重新分配给新的独立样本；重放解码将当前解视为草稿，并并行验证未发生变化的 token，从而实现高效的不动点检测。在来自五个家族的七个模型和七个基准测试上，ARR 在同等计算量下于五个数学推理基准中的三个上与多数投票的表现相当。

    arXiv:2610.28294v1 Announce Type: new  Abstract: Test-time scaling allocates additional inference compute to refining existing solutions or generating new ones. This raises a fundamental question: Should I keep improving this solution, or stop working on it and try a completely new solution? Existing methods make this decision using fixed refinement schedules, self-verification judgments, or learned controllers, which can waste compute or rely on signals that may be incorrect or require additional training. We introduce Adaptive Replay Refinement (ARR), a training-free method that detects when a stationary revision reaches a fixed point and reallocates the released compute to a new independent sample; replay decoding treats the current solution as a draft and verifies unchanged tokens in parallel, enabling efficient fixed-point detection. Across seven models from five families and seven benchmarks, ARR matches majority voting on three of five mathematical reasoning benchmarks at equal 
    
[^26]: 四十个问题：从单词式回答识别语言模型

    Forty Questions: Identifying a Language Model from One-Word Answers

    [https://arxiv.org/abs/2610.28281](https://arxiv.org/abs/2610.28281)

    本文提出仅通过40个日常问题的单词式回答即可构建模型“指纹”，以92.5%的准确率精确识别API背后的语言模型，并据此揭露了模型端点伪装、计费异常及隐藏系统提示等实际问题。

    

    摘要：API背后的模型可以通过其对日常问题（如“说出一种声音”或“说出一位诗人”）的单词式回答被识别出来。模型的档案记录了它对每个问题给出每个答案的频率，我们据此将新的回答归因于最有可能给出该回答的模型。在一个包含94个模型的测试面板上，40个回答能够以92.5%的准确率识别出确切的模型，而随机猜测的准确率仅为1.1%。通过OpenRouter构建的档案在盲测条件下成功识别了DeepInfra所服务的全部十个模型。系统提示词会将识别准确率降低6至16个百分点。我们将该方法应用于OpenRouter上13个开放模型的96个端点，其中92个端点被识别为其声称的模型。在其余四个端点中，一个在关闭推理功能时与声称的模型匹配；一个以Llama 4 Maverick的名义实际提供Gemma 4服务：它自称是Gemma 4，且每条消息都按Gemma 4的token计数计费；另外两个端点尚无法确认。计费记录还显示第一方模型被添加了额外上下文。Grok 4.7是

    arXiv:2610.28281v1 Announce Type: new  Abstract: A model behind an API can be identified from its one-word answers to everyday questions such as "Name a sound" or "Name a poet". A model's profile records how often it gives each answer to each question. We attribute new answers to the model most likely to have given them. On a panel of 94 models, 40 answers identify the exact model 92.5% of the time, against 1.1% by chance. Profiles built through OpenRouter identified all ten models served by DeepInfra under blinded labels. A system prompt lowers accuracy by 6 to 16 points. We applied the method to 96 endpoints of 13 open models on OpenRouter, and 92 identify as the model they claim. Of the other four, one matches the claimed model with reasoning turned off. One serves Gemma 4 under the name Llama 4 Maverick: it names itself Gemma 4, and it is billed Gemma 4's token counts on every message. Two remain unresolved. Billing records also show context added to first-party models. Grok 4.7 is
    
[^27]: 语言模型如何习得品味：大语言模型中的品牌偏好

    How Language Models Learn Taste: Brand Preference in LLMs

    [https://arxiv.org/abs/2610.28272](https://arxiv.org/abs/2610.28272)

    语言模型在“说出品牌”时倾向市场领导者，而在“推荐品牌”时却偏向小众专业品牌，这种品味形成于预训练并被DPO偏好优化显著放大，即使偏好数据中几乎不含品牌信息。

    

    当被要求说出一个跑鞋品牌时，语言模型会回答耐克；当被要求推荐一个时，它们却会推荐Brooks。本文探究模型从何处习得这种品味。在102个模型和42个品牌类别中，模型推荐时选择的品牌与其说出的品牌在26个类别中存在差异。整个领域共同趋向于一致的推荐。被说出的品牌是市场领导者，而被推荐的品牌大多是小众和专业品牌。三个开放的训练流程揭示了这种偏好的形成过程：预训练阶段就已经将推荐的品牌与建议行为关联起来；监督微调教会模型如何作答，但基本不改变其推荐选择；偏好优化（DPO）则教会模型哪个答案更好，在所有三个流程中，推荐选择都发生了11到24个百分点的偏移。值得注意的是，偏好数据中几乎不包含任何品牌。即使偏好的品牌并未出现在偏好数据中，DPO也会使模型的判断向另一组关联偏移，且这种效应持续存在。

    arXiv:2610.28272v1 Announce Type: new  Abstract: Asked to name a running shoe brand, language models say Nike. Asked to recommend one, they say Brooks. This paper asks where models learn that taste. Across 102 models and 42 brand categories, the brand a model picks when recommending differs from the brand it names in 26. The field moves together to a shared recommendation. The Named brands are market leaders, and the Picked brands are mostly niche and specialist. Three open training pipelines show where this forms. Pretraining already attaches the recommended brand to advice. Supervised fine-tuning teaches the model to answer and mostly leaves the pick unchanged. Preference optimization (DPO) teaches the model which answer is better, and the pick moves in all three pipelines, by 11 to 24 points. The preference data contains almost no brands. DPO moves the model's judgments toward a second set of associations even when the preferred brand is not in the preference data, and the effect re
    
[^28]: 语言模型会因神经元消融而患上“失语症”吗？

    Can Language Models Get "Aphasia" From Neuronal Ablation?

    [https://arxiv.org/abs/2610.28214](https://arxiv.org/abs/2610.28214)

    该研究通过系统性消融语言模型神经元发现，LSTM呈渐进式退化，而Transformer在MLP消融下出现类似Wernicke型失语症的阈值式崩溃，表明语言模型可作为研究神经退行性疾病认知衰退模式的计算代理。

    

    渐进性神经元丢失是神经退行性疾病的标志，然而神经元丢失如何塑造认知衰退的模式仍知之甚少。在本研究中，我们系统性地消融了涵盖循环长短期记忆网络（LSTM）和Transformer架构的语言模型中的神经元（计算单元），并采用有限的微调作为代偿性可塑性的简单代理。在全单元消融下，LSTM在轻度至中度损伤范围内呈渐进式退化，其语法和语义能力平行下降。相比之下，Transformer在前馈多层感知机（MLP）消融下表现出晚期的阈值式崩溃，而在注意力头消融下则呈现更平滑的S型下降。在定性层面，MLP消融后的模型仍会做出类别内的语义替换（语义错语症），同时非响应率上升，这与Wernicke型失语症相似，而注意力头消融则产生了一种趋势……

    arXiv:2610.28214v1 Announce Type: new  Abstract: Progressive neuronal loss is the hallmark of neurodegenerative disease, yet how loss of neurons shapes the pattern of cognitive decline remains poorly understood. Here we systematically ablate neurons (computational units) in language models spanning recurrent long short-term memory (LSTM) and Transformer architectures and apply limited fine-tuning as a simple proxy for compensatory plasticity. With whole-unit ablation, LSTMs degrade gradually with syntactic and semantic faculties declining in parallel across mild-to-moderate damage. By contrast, Transformers exhibit late threshold-like collapse under feedforward multi-layer perceptron (MLP) ablation and smoother sigmoid-like decline under attention-head ablation. Qualitatively, MLP-ablated models continue to make within-category semantic substitutions (semantic paraphasias) alongside rising non-responses, paralleling Wernicke-type aphasia, while attention-head ablation produces a trend 
    
[^29]: 通过跨模态概念神经元揭示大型音频-语言模型中的音频-文本对齐

    Uncovering Audio-Text Alignment in Large Audio-Language Models through Cross-Modal Concept Neurons

    [https://arxiv.org/abs/2610.28189](https://arxiv.org/abs/2610.28189)

    该研究发现了大型音频-语言模型中同时表示音频和文本的跨模态概念神经元，证实音频与文本表示在共享子空间中存在对齐，且这些神经元可通过因果干预被操控。

    

    尽管语言模型主干仅在文本上进行预训练，但大型音频-语言模型（LALMs）在很大程度上能够识别声学属性，例如情感、说话者性别和口语语言。然而，音频表示是否以及如何与文本表示整合，目前尚不清楚。我们假设存在一个共享子空间，音频和文本表示在其中是对齐的。为了验证这一假设，我们识别出负责同时表示音频和文本的跨模态神经元。我们在三个大型音频-语言模型和四种声学属性类型中发现音频神经元与文本神经元之间存在不同程度的对齐。因果干预实验——包括引导和关闭这些神经元——证实了跨模态神经元不仅对预测是必需的，而且还可以被操控。

    arXiv:2610.28189v1 Announce Type: cross  Abstract: While the language model backbone is pretrained on text alone, large audio-language models (LALMs) can, to a great extent, recognize acoustic attributes, such as emotion, speaker gender, and spoken language. However, it is unclear how the audio representation is integrated, if at all, with the other text representation before the model produces text. We hypothesize that there is a shared subspace where the audio and text representations are aligned. To test this hypothesis, we identify cross-modal neurons that are responsible for representing both audio and text. We find alignment between audio and text neurons to various degrees across three LALMs and four acoustic attribute types. Causal interventions, steering and deactivating the neurons, confirm that the cross-modal neurons are not just necessary for prediction but can also be manipulated.
    
[^30]: 面向大语言模型时序知识图谱推理的时序证据-参数协同对齐

    Temporal Evidence-Parameter Co-Alignment for LLM Reasoning over Temporal Knowledge Graphs

    [https://arxiv.org/abs/2610.28176](https://arxiv.org/abs/2610.28176)

    本文提出TEPCA框架，通过在证据层面检索时序连贯的知识、在参数层面进行动态自适应的低秩更新，从证据和参数两个维度实现大语言模型与时序知识图谱推理的协同对齐。

    

    大语言模型（LLM）在时序相关推理方面常常表现不佳，因为其静态的参数化记忆与现实世界知识的动态演化不相匹配。尽管检索增强生成（RAG）和参数高效微调（PEFT）为增强大语言模型提供了实用途径，但它们对于时序推理而言仍然不够充分。现有的检索方法通常优先考虑语义相关性，而忽视了时序有效性、事件顺序和演化依赖关系，导致所获取的证据在孤立状态下看似相关，但在时序上却是碎片化的。与此同时，传统的PEFT方法通常采用固定的、与输入无关的低秩更新，难以捕捉依赖时间的实体语义和关系漂移。为了解决这些局限性，我们提出了时序证据-参数协同对齐（TEPCA），这是一个用于大语言模型在时序知识图谱（TKG）上进行推理的框架。TEPCA在证据和参数两个层面将大语言模型推理与时序知识对齐。

    arXiv:2610.28176v1 Announce Type: cross  Abstract: Large Language Models (LLMs) often struggle with temporally grounded reasoning because their static parametric memory is misaligned with the dynamic evolution of real-world knowledge. Although RAG and Parameter-Efficient Fine-Tuning (PEFT) provide practical ways to enhance LLMs, they remain insufficient for temporal reasoning. Existing retrieval methods typically prioritize semantic relevance while overlooking temporal validity, event order, and evolutionary dependencies, resulting in evidence that is relevant in isolation but temporally fragmented. Meanwhile, conventional PEFT methods usually employ fixed and input-agnostic low-rank updates, making it difficult to capture time-dependent entity semantics and relation drift. To address these limitations, we propose Temporal Evidence-Parameter Co-Alignment (TEPCA), a framework for LLM reasoning over TKGs. TEPCA aligns LLM reasoning with temporal knowledge at both the evidence and paramet
    
[^31]: 非洲论文集：面向非洲基础语言模型的百亿 token 多领域学术语料库

    Africa-Papers: A 10-Billion-Token Multi-Domain Academic Corpus for African Foundation Language Models

    [https://arxiv.org/abs/2610.28173](https://arxiv.org/abs/2610.28173)

    本文发布了迄今最大的泛非多领域学术预训练语料库 Africa-Papers，包含 84 万余篇全文论文、共计 101.5 亿个 token，旨在弥补现有大模型预训练数据中非洲学术内容占比不足 0.5% 的结构性缺失，提升模型对非洲相关领域的理解能力并推进非洲的 AI 知识主权。

    

    大语言模型（LLM）从根本上依赖于预训练语料库的多样性、深度和事实严谨性。然而，当代基础预训练数据集（如 The Pile、S2ORC 和 Dolma）存在严重的结构性缺失：非洲的科学、医学、法律和农业学术文献在全球预训练 token 中占比不足 0.5%。因此，最先进的模型在对热带流行病学、本地化小农农业生态、区域经济治理以及本土语言结构进行推理时会出现系统性的失效模式。为解决这一表征缺失问题并推进非洲在人工智能领域的知识主权，我们推出了 Africa-Papers——迄今规模最大的经过整理的泛非多领域学术预训练语料库。该语料库包含超过 844,807 篇全文论文，共计 101.5 亿个 token，分布于 879 个 Parquet 分片中，统一整合了……

    arXiv:2610.28173v1 Announce Type: cross  Abstract: Large Language Models (LLMs) depend fundamentally on the diversity, depth, and factual rigor of their pre-training corpora. However, contemporary foundational pre-training datasets (such as The Pile, S2ORC, and Dolma) exhibit a severe structural omission: African scientific, medical, legal, and agricultural scholarship constitutes less than 0.5% of global pre-training tokens. Consequently, state-of-the-art models suffer from systematic failure modes when reasoning about tropical epidemiology, localized smallholder agro-ecologies, regional economic governance, and indigenous linguistic structures. To resolve this representation deficit and advance African intellectual sovereignty in artificial intelligence, we introduce Africa-Papers, the largest curated multi-domain Pan-African academic pre-training corpus to date. Comprising over 844,807 full-text papers spanning 10.15 billion tokens across 879 Parquet shards, Africa-Papers unifies fo
    
[^32]: 职业独特性与工资增长

    Occupational Uniqueness and Wage Growth

    [https://arxiv.org/abs/2610.28133](https://arxiv.org/abs/2610.28133)

    本文提出了一个衡量职业任务结构的新框架，将任务数量变化与职业独特性变化相分离，并利用1977-2018年间近20万条职业任务描述数据研究职业独特性与工资增长之间的关系。

    

    职业任务内容的变化可能产生不同的经济影响，这取决于所获得或失去的任务是该职业特有的，还是已经在整个经济中普遍存在的任务。现有的职业任务变化度量方法在很大程度上忽略了这一区别，将职业视为自成一体的任务集合，而没有考虑任务在各职业之间的分布情况。我们开发了一个框架来衡量职业变化的一个独特维度，这一维度超越了任务专业性或任务内容，即任务结构。利用这一追踪各职业及各时期任务的新框架，我们将任务数量的变化与职业独特性的变化分离开来，职业独特性定义为某一职业的任务集合相对于其他职业的显著程度。我们使用了一个统一的数据集，该数据集包含来自《职业名称词典》和O*NET的196,563条任务描述，时间跨度为1977年至2018年，我们据此测量了任务扩散……

    arXiv:2610.28133v1 Announce Type: cross  Abstract: Changes in occupational task content may have different economic implications depending on whether the tasks gained or lost are occupation-specific or already widespread across the economy. Existing measures of occupational task change largely abstract from this distinction, treating occupations as self-contained task bundles rather than accounting for how tasks are distributed across occupations. We develop a framework to measure a distinct dimension of occupational change, one that goes beyond task expertise or task content: task structure. Using this new framework that tracks tasks across occupations and over time, we separate changes in task quantity from changes in occupational uniqueness, defined as the distinctiveness of an occupation's task bundle relative to other occupations. Using a harmonized dataset of 196563 task statements from the Dictionary of Occupational Titles and O*NET spanning 1977-2018, we measure task diffusion,
    
[^33]: 过度解读了吗？模糊情境下大语言模型心理化中的解释偏好与承诺

    Reading Too Much Into It? Interpretive Preferences and Commitment in LLM Mentalisation under Ambiguity

    [https://arxiv.org/abs/2610.28105](https://arxiv.org/abs/2610.28105)

    该研究提出MentalLoom评估框架，发现在模糊社会情境下，用户的心理困扰披露与重复提问会促使八个主流LLM转向更积极的对叙述者的解读、提供更多情感支持并表现出更高的确定性，证明用户语境能够重塑模型的解释偏好与承诺。

    

    大语言模型（LLM）越来越多地被用于提供人际建议，可能影响用户如何解读模糊的社会情境。现有的心智理论评估通常依赖于预设答案，而不确定性研究则忽视了LLM在心理困扰情境下偏好的解释及其置信度。为弥补这些空白，我们提出了MentalLoom，一种新颖的LLM心智化评估方法，包含604个模糊社会事件、匹配的抑郁症与社交焦虑自我披露，以及事件证据保持不变的四轮对话。在八个LLM上的实验显示，自我披露信息总体上最初引发了更多负面解释，但对叙述者本人的解读则更为积极。脚本化的重复提问表明，在没有新事件证据的情况下，模型表现出更多的情感支持、更强的对叙述者的积极认同以及更高的确定性。这些发现表明，用户语境能够重塑模型的解释偏好与承诺。

    arXiv:2610.28105v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used for interpersonal advice, potentially shaping how users interpret ambiguous social encounters. Existing Theory of Mind evaluations typically rely on predefined answers, while uncertainty research overlooks LLMs' preferred explanations and confidence under psychological distress. To address these gaps, we introduce MentalLoom, a novel evaluation of LLM mentalisation comprising 604 ambiguous social episodes, matched depression and social anxiety disclosures, and four-turn dialogues with unchanged event evidence. Across eight LLMs, disclosures initially elicited more negative explanations overall but more positive narrator-directed interpretations. Scripted repeated questioning showed greater emotional support, stronger positive narrator-directed endorsement, and higher certainty without new event evidence. These findings show that user context can reshape interpretive preferences and commi
    
[^34]: BAIT：大语言模型能发现什么会引发读者的偏见吗？——一个新闻认知偏见触发因素的基准

    BAIT: Can LLMs Find What Biases a Reader? A Benchmark for Cognitive Bias Triggers in News

    [https://arxiv.org/abs/2610.28092](https://arxiv.org/abs/2610.28092)

    该论文提出BAIT基准，通过九类认知偏见触发因素的分类体系和50篇标注新闻文章，定义了偏见触发因素检测与定位两项任务，以评测大语言模型识别会让人类读者产生偏见的文本特征的能力。

    

    说服性写作会利用认知偏见，借助读者的思维捷径来塑造其信念，而读者往往毫无察觉。我们将这类利用思维捷径的语言特征称为“认知偏见触发因素”。随着大语言模型（LLM）日益成为人们接触新闻的中介——为新闻做摘要、对新闻进行排序并基于新闻回答问题——一个自然的问题是：它们能否识别出会让人类读者产生偏见的文本特征？我们提出了BAIT（Bias Annotation In Text，文本偏见标注），一个用于检测新闻中认知偏见触发因素的资源。BAIT包含一个由九种触发因素组成的分类体系（每种触发因素都在标注员手册中有明确说明），以及50篇英文新闻文章，这些文章由经过培训的标注员在文章层面和文本片段层面进行标注，并经裁定形成金标准。基于这些数据，我们定义了两项任务：检测文本中包含哪些触发因素，以及定位承载这些触发因素的文本片段。随后，我们对六个前沿大语言模型进行了零样本及（基于提示的）基准测试。

    arXiv:2610.28092v1 Announce Type: cross  Abstract: Persuasive writing exploits cognitive biases, recruiting a reader's mental shortcuts to shape belief, often without the reader noticing. We call the linguistic features exploiting such mental shortcuts cognitive bias triggers. As large language models (LLMs) increasingly mediate how people encounter news, summarizing it, ranking it, and answering questions over it, a natural question is whether they can recognize the features of a text that bias a human reader. We introduce BAIT (Bias Annotation In Text), a resource for detecting cognitive bias triggers in news. BAIT pairs a schema of nine triggers, each specified in an annotator codebook, with 50 English news articles annotated by trained annotators at the article and span levels and adjudicated into a gold standard. On this data we define two tasks, detecting which triggers a text contains and locating the spans that carry them. We then benchmark six frontier LLMs zero-shot and with 
    
[^35]: 情境坦诚：面向大语言模型在混合敏感度语境下的角色感知分级披露

    Contextual Candour: Role-Aware Graduated Disclosure for LLMs over Mixed-Sensitivity Contexts

    [https://arxiv.org/abs/2610.28067](https://arxiv.org/abs/2610.28067)

    本文提出“分级披露”概念，将大语言模型在混合敏感度信息上的访问控制从非黑即白的“授予或拒绝”决策，革新为根据用户授权级别校准回答细节与敏感度的生成方式，并通过基于奖励的优化框架在惩罚改写、近似等隐性泄露的同时，引导模型生成与授权相称的回答。

    

    大语言模型日益在混合敏感度的语境中运行，它们需要推理从公开到机密的各种信息，同时只披露用户被授权获取的内容。现有方法将访问控制作为二元决策来执行：要么完整回答，要么拒绝，而未能在授予访问权限后对披露的信息量或信息类型加以约束，从而导致过度拒绝或意外的信息泄露。我们提出了“分级披露”这一访问控制形式化方法，其中模型回答的细节程度和敏感度根据用户的授权级别进行校准，而非简单地将决策视为“授予或拒绝”。我们将分级披露建模为一个披露感知的生成问题，并提出基于奖励的优化框架，使模型的回答朝向与授权级别相称的抽象层次，同时惩罚未经授权的事实性泄露，包括改写和近似表述等。我们评估了我们的方法……（原文摘要在此处截断）

    arXiv:2610.28067v1 Announce Type: cross  Abstract: LLMs increasingly operate over mixed-sensitivity contexts, where they must reason over information ranging from public to confidential while disclosing only what the user is authorised to receive. Existing approaches enforce access control as a binary decision: either fully answering or refusing, without regulating how much or what kind of information is revealed if access is granted, leading to over-refusal or unintended leakage. We introduce graduated disclosure, an access-control formulation in which the level of detail and sensitivity in a model's response is calibrated to the user's authorisation rather than treated as a grant-or-deny decision. We model graduated disclosure as a disclosure-aware generation problem and propose a reward-based optimisation framework that shapes responses toward level-appropriate abstractions while penalising unauthorised factual leakage, including paraphrases and approximations. We evaluate our appro
    
[^36]: 同样的判断，不同的“是”：度量语言模型中的多语言判断画像

    Same Judgment, Different Yes: Measuring Multilingual Judgment Profiles in Language Models

    [https://arxiv.org/abs/2610.28066](https://arxiv.org/abs/2610.28066)

    通过对八个模型、六种语言共110万条道德判断回复的分析，该研究发现除否定疑问句下按预设映射读取的“是/否”标记外，语言模型在不同提问形式和答案读取方式下均呈现一致的道德判断画像，表明多数测量方法反映的是真实判断而非回答习惯。

    

    多语言评估通常对模型回答开头的是/否标记进行打分，而这个标记同时也承载着模型和语言的回答习惯。我们探讨何时不同的引出与读取答案的方式能够还原相同的判断，何时它们度量的只是一种回答习惯。我们在最多105个道德情境短文上收集了八个模型在六种语言中的110万条回复，变化了问题的形式、答案的约束方式以及回复的读取方式（标记或句子），涵盖两种措辞（即FAR-6测试组）。故事层面的判断画像在各种测试的读取方式和问题形式之间表现出收敛一致，唯一的例外是在否定疑问句之后按预设映射读取的是/否标记：在相同的回复上，标记画像与句子画像在各单元格中的中位数一致性为 r = 0.67，而句子画像与自由句子画像的一致性达到 0.94，后者属于已知的语言变异性。在去除每种语言的故事共识后，模型的画像最接近于其自身在……（原文摘要至此截断）

    arXiv:2610.28066v1 Announce Type: cross  Abstract: Multilingual evaluations often score a model's opening yes or no, a token that also carries model and language answering conventions. We ask when different ways of eliciting and reading an answer recover the same judgment, and when they measure an answer habit. We collect 1.1 million replies from eight models in six languages on up to 105 moral vignettes, varying the question's form, the answer constraint, and the reading of the reply, token or sentence, in both wordings (the FAR-6 battery). Story-level judgment profiles converge across the tested readouts and question forms, except for the yes/no token after a negative interrogative read under a preset mapping: on the same replies, token and sentence profiles agree at a median r = 0.67 across cells, while sentence profiles agree with free-sentence profiles at 0.94, a known linguistic variability. With each language's story consensus removed, a model's profile is nearest to its own in 
    
[^37]: 在线策略过渡蒸馏

    On-Policy Transition Distillation

    [https://arxiv.org/abs/2610.28048](https://arxiv.org/abs/2610.28048)

    该论文发现教师与学生在中间层预测过渡上的一致性与答案正确率正相关，并提出在线策略过渡蒸馏（OPTD），通过在学生生成的轨迹上对齐教师与学生的逐层对数概率增量来增强传统最终层分布匹配，从而提升蒸馏效果。

    

    在线策略蒸馏（OPD）通过在学生模型生成的采样轨迹上匹配教师模型与学生模型的最终层分布，将专家能力整合到单个学生模型中。我们研究了深度方向上的预测过渡——即各层之间预测的变化——能否提供额外的与任务相关的监督信号。我们的诊断性研究发现，教师与学生在中间层过渡上的一致性越强，答案正确率越高，正确回答在所有测量的中间层对上均表现出更低的平均过渡KL散度。这些观察启发了在线策略过渡蒸馏（OPTD）的提出，该方法通过在学生生成的采样轨迹上对齐教师与学生的逐层对数概率增量，来增强最终层的分布匹配。在涵盖数学、编程和指令遵循任务的三个骨干模型和八个基准上的评估表明，OPTD 取得了……

    arXiv:2610.28048v1 Announce Type: cross  Abstract: On-policy distillation (OPD) consolidates expert capabilities into a single student by matching teacher and student final-layer distributions on student-generated rollouts. We investigate whether depthwise prediction transitions, the changes in predictions between layers, can provide additional task-relevant supervision. Our diagnostic study finds that stronger teacher--student agreement in intermediate transitions is associated with higher answer correctness within questions, with correct responses exhibiting lower mean transition KL across all measured intermediate layer pairs. These observations motivate On-Policy Transition Distillation (OPTD), which augments final-layer distribution matching by aligning teacher and student layer-to-layer log-probability increments on student-generated rollouts. Evaluations across three backbones and eight benchmarks spanning mathematics, coding, and instruction following show that OPTD achieves th
    
[^38]: 基于投毒的后门并非密码学安全的

    Poisoning-Based Backdoors Are Not Cryptographic

    [https://arxiv.org/abs/2610.28045](https://arxiv.org/abs/2610.28045)

    本文通过归约证明，基于投毒安装的神经网络后门不可能获得密码学认证那样的不对称性，因为任何从被投毒数据中安装触发器验证器的训练过程本身就是一个学习过程，因此基于学习的防御在原则上始终是可行的。

    

    为了防御针对神经网络的后门攻击，防御者必须识别出决定某个输入是否为触发器的验证器函数，而不能只是孤立地拒绝单个触发器，因为攻击者生成满足同一验证器的新触发器的速度会快于防御者拦截它们的速度。如果后门验证器能够像密码学认证方案那样具有不对称性——即只有攻击者能够产生触发器输入，而防御者无法通过观测逆向推导出验证器——那么基于学习的防御在原则上将是不可能的。我们证明，这种不对称性不可能由基于投毒的安装方式产生，在这种方式中，训练过程中唯一依赖于验证器的输入就是被投毒的数据集。该论证采用归约方法：一个能够从被投毒数据中安装验证器、使后门在未见过的触发器上触发同时保持良性行为的训练过程，其本身就是一个学习（摘要在此处截断）

    arXiv:2610.28045v1 Announce Type: cross  Abstract: To defend against backdoor attacks on neural networks, the defender must identify the verifier function that determines whether an input is a trigger, rather than rejecting individual triggers in isolation, since an attacker can generate new triggers satisfying the same verifier faster than they can be blocked. If a backdoor verifier could be installed with the asymmetry of a cryptographic authentication scheme, in which only the attacker can produce trigger inputs while the defender cannot reverse-engineer the verifier from observations, learning-based defense would be impossible in principle. We show that no such asymmetry can arise from poisoning-based installation, in which the only verifier-dependent input to training is a poisoned dataset. The argument is a reduction. A training procedure that installs a verifier from poisoned data, so that the backdoor fires on unseen triggers while benign behavior is preserved, is itself a lear
    
[^39]: 利用字母位置概率评估法语单词复杂度

    Using Letter Positional Probabilities to Assess French Word Complexity

    [https://arxiv.org/abs/2610.28040](https://arxiv.org/abs/2610.28040)

    本文提出利用字母位置概率（LPP）这一新指标，通过对比词汇判断任务中反应时间最高与最低的单词组，发现大量字母/位置概率与法语单词复杂度显著相关。

    

    我们使用字母位置概率（LPP）来考察简单法语单词与复杂法语单词之间的正字法差异。我们从一个包含38,000个法语单词的样本中抽取数据，这些单词在词汇判断任务中按反应时间评分。我们选取反应时间最高的5000个单词和最低的5000个单词，分别作为高复杂度组和低复杂度组的代表。我们基于单词的结构创建了18个变量，例如单词长度和音节数量。此外，我们根据标准法语字母表的42个字符，对每个单词的字母进行独热编码，最长编码至25个字母，从而将每个单词简化为一串二进制值。因此，每个单词由42×25个二进制字符表示。随后，我们通过比较简单单词与复杂单词在所有字母/位置上的平均二进制得分来构建LPP。我们发现，有大量LPP与单词复杂度相关。例如，我们发现大多数带重音符号的字母……

    arXiv:2610.28040v1 Announce Type: new  Abstract: We use Letter Positional Probabilities (LPPs) to examine the orthographic differences between simple and complex French words. We draw from a sample of 38,000 French words scored by reaction time on a lexical decision task. We take the 5000 highest reaction time words and the 5000 lowest as a proxy for high and low word complexity groups respectively. We create 18 variables based on the structure of the words, such as length and number of syllables. We also reduce each word to a string of binary values by one-hot encoding each letter according to the 42 characters of the standard French alphabet up to a length of 25 letters. Thus, each word is represented by 42 x 25 binary characters. We then create LPPs by looking at the average binary scores for simple as opposed to complex words for all letter/positions. We find that there is a high number of LPPs that are associated with word complexity. For example, we find that most accented letter
    
[^40]: 从强化学习论文到框架代码：七个开源训练框架中的收录与评审

    From RL Papers to Framework Code: Inclusion and Review in Seven Open-Source Training Frameworks

    [https://arxiv.org/abs/2610.28027](https://arxiv.org/abs/2610.28027)

    该研究通过追踪七个开源强化学习后训练框架的代码提交历史，发现1598篇提出通用RL训练方法的论文中仅有37篇被框架收录，且收录的论文多为大规模训练（30B+参数）和现有损失函数的小幅修改，揭示了RL研究成果从论文到实际工程应用之间的巨大鸿沟。

    

    从业者通过训练框架对语言模型使用强化学习方法，因此框架收录的方法对用户而言仅需更改一次配置即可使用。我们测量了这种情况发生的频率。我们追踪了七个用于强化学习后训练的开源框架的提交历史，并将其与2025年1月至2026年6月的6,939篇筛选后的arXiv论文相关联。在1,598篇提出通用强化学习训练方法的论文中，仅有37篇进入至少一个框架。在五个季度论文批次中，180天内被收录的数量始终保持在5至7篇之间，而批次规模却从72篇增长到386篇。与同期的其他论文相比，被收录的论文更常训练300亿（30B）参数或更大的策略模型（41%对10%），并且更常在现有损失函数中修改系数、裁剪、掩码或权重（29%对9%）。即使排除维护这些框架的公司自己的论文，这两个差距依然存在。

    arXiv:2610.28027v1 Announce Type: cross  Abstract: Practitioners use reinforcement learning methods for language models through training frameworks, so a method that a framework includes is one configuration change away for its users. We measure how often this happens. We trace seven open-source frameworks for RL post-training through their commit histories and link them to 6,939 screened arXiv papers from January 2025 to June 2026. Of 1,598 papers that propose a general RL training method, 37 reached at least one framework. Across five quarterly cohorts of papers, the number included within 180 days stayed between 5 and 7 while the cohorts grew from 72 to 386 papers. Compared with papers from the same quarters, included papers more often trained policies of 30B parameters or more, at 41 against 10 percent, and more often changed a coefficient, clip, mask or weight in the existing loss, at 29 against 9 percent. Both gaps hold without papers from the companies that maintain the framewor
    
[^41]: 超越证据边界：医学视觉语言模型在渐进性退化下的行为幻象

    Beyond the Evidence Boundary: Behavioral Mirage Under Progressive Degradation in Medical VLMs

    [https://arxiv.org/abs/2610.28000](https://arxiv.org/abs/2610.28000)

    该论文提出了一个基线锚定框架，通过渐进性图像退化识别医学视觉语言模型的“证据边界”，并首次定义“行为幻象”现象——即模型在证据已不足以支持判断的边界处仍持续坚持原有回答，且发现仅凭文本即可回答的问题与该坚持行为显著相关。

    

    用于医学图像解读的视觉语言模型（VLM）通常是在干净输入上进行评估的，然而在高风险临床环境中可能发生渐进性的证据丢失。现有的鲁棒性研究表明，性能可能下降而置信度仍然保持很高，但并未确定一个先前可靠的模型-案例对何时会停止可靠运行。我们引入了一个以基线为锚点的框架，通过渐进性图像退化追踪基线正确的案例，以识别经验性的证据边界，并将“幻象”定义为无论正确与否，模型在该边界处仍继续坚持其原有判断的行为。在三个医学数据集上对三个VLM进行的实验中，只有MedGemma提供了足够大的基线正确样本池用于轨迹分析。在VQA-RAD（唯一一个问题文本会变化的数据集）上，仅凭语言文本即可回答的问题与基线正确性相关（AUROC 0.677），也与边界处的坚持行为相关（AUROC 0.653至0.670）。

    arXiv:2610.28000v1 Announce Type: cross  Abstract: VLMs for medical image interpretation are typically evaluated on clean inputs, despite the progressive evidence loss that can occur in high stakes clinical settings. Existing robustness studies show that performance can degrade while confidence remains high, but do not establish when a previously reliable model-case pair ceases to behave reliably. We introduce a baseline-anchored framework that traces baseline-correct cases through progressive image degradation to identify an empirical evidence boundary, and define mirage as continued commitment at this boundary, regardless of correctness. Across three VLMs on three medical datasets, only MedGemma provides a sufficiently large baseline-correct pool for trajectory analysis. On VQA-RAD, the only dataset in which the question text varies, language-only answerability is associated with baseline correctness (AUROC 0.677) and with commitment at the boundary (AUROC 0.653 to 0.670). MedGemma's
    
[^42]: 使用RSOD构建电信语言模型：领域数据、专家后训练、内存感知蒸馏与网络评估

    Building Telecom Language Models with RSOD: Domain Data, Specialist Post-Training, Memory-Conscious Distillation, and Networking Evaluation

    [https://arxiv.org/abs/2610.27993](https://arxiv.org/abs/2610.27993)

    该论文提出RSOD端到端框架，通过领域合成数据构建、多个专家LoRA教师后训练以及内存感知的在线策略蒸馏，将六个网络专家模型的行为整合到一个可部署的单一电信语言模型中。

    

    电信语言模型必须将广泛的网络知识与路由、宽带服务、接入技术和运维诊断方面的专家行为相结合，同时保持作为单一模型部署的实用性。我们提出了一个端到端框架，该框架从结构化网络分类体系和专家图谱构建有来源依据的合成训练数据，生成跨越十四条网络路由的212,559条路由化训练记录，在一个共享的持续预训练骨干模型上训练六个冻结的专家LoRA教师模型，并使用路由专家在线策略蒸馏（Routed Specialist On-Policy Distillation, RSOD）将其行为整合到一个可部署的学生适配器中。在RSOD中，已确定的学生模型生成每条轨迹，按路由分配的专家教师对精确的原生token序列进行教师强制，紧凑的教师隐藏状态通过共享的冻结输出头进行重建，且仅更新学生LoRA。一项受控的重建……

    arXiv:2610.27993v1 Announce Type: new  Abstract: Telecom language models must combine broad networking knowledge with specialist behavior across routing, broadband services, access technologies, and operational diagnostics while remaining practical to deploy as a single model. We present an end-to-end framework that builds source-grounded synthetic training data from structured networking taxonomies and expert graphs, materializes 212,559 routed training records across fourteen networking routes, trains six frozen specialist LoRA teachers over one shared continued-pretrained backbone, and consolidates their behavior into one deployable student adapter using Routed Specialist On-Policy Distillation (RSOD). In RSOD, the committed student generates each trajectory, the route-assigned specialist teacher-forces the exact native token sequence, compact teacher hidden states are reconstructed through the shared frozen output head, and only the student LoRA is updated. A controlled reconstruct
    
[^43]: 面向语言模型智能体的流式推理自适应停止

    Adaptive Stopping of Streamed Reasoning for Language Model Agents

    [https://arxiv.org/abs/2610.27981](https://arxiv.org/abs/2610.27981)

    提出 StreamStop 方法，利用冻结的前缀读取器和轻量级停止头，从前缀级别动作一致性中离线学习针对特定模型的停止策略，无需访问推理模型内部信号即可自适应地停止语言模型智能体的流式推理，并发布了带前缀级别动作标注的推理轨迹数据集。

    

    具备推理能力的语言模型智能体在行动之前可能产生大量计算开销，然而不同决策所需的推理量各不相同。现有的停止信号（如置信度和中间答案稳定性）在不同推理模型之间的有效性参差不齐；而基于隐藏状态或词元概率的信号则需要访问模型内部信息，这些访问可能不可用或不切实际。我们提出 StreamStop，它从前缀级别的动作一致性中离线学习针对特定模型的停止策略。一个冻结的前缀读取器对可获得的推理文本进行编码，供一个轻量级的停止头使用，从而无需访问推理模型的内部信号即可实现停止决策。我们构建并公开发布了一个包含智能体推理轨迹及前缀级别动作标注的数据集。这些标注将每个前缀动作与完整推理的参考动作进行对比，从而支持对停止规则的离线比较以及精确的期望回报评估。

    arXiv:2610.27981v1 Announce Type: new  Abstract: Reasoning-enabled language model agents can incur substantial computation before acting, although the reasoning needed varies across decisions. Existing stopping signals, such as confidence and intermediate-answer stability, can vary in effectiveness across reasoning models; signals obtained from hidden states or token probabilities also require internal access that may be unavailable or impractical. We introduce StreamStop, which learns model-specific stopping policies offline from prefix-level action agreement. A frozen prefix reader encodes available reasoning text for a lightweight stop head, allowing stopping without access to the reasoning model's internal signals. We construct and publicly release a dataset of agent reasoning trajectories with prefix-level action annotations. These annotations compare each prefix action with the full-reasoning reference action, enabling offline comparison of stopping rules and exact expected-retur
    
[^44]: 迷失于恢复之中：通过面向目标的动作回放揭示计算机使用智能体的恢复行为

    Lost in Recovery: Demystifying Recovery Behavior of Computer-Use Agents via Goal-Oriented Action Replay

    [https://arxiv.org/abs/2610.27954](https://arxiv.org/abs/2610.27954)

    该论文提出ReTrace评估框架，通过形式化“转换一致的恢复”标准并借助面向目标的动作回放，系统性地评估计算机使用智能体从失败中恢复的能力，填补了基于模型的恢复研究的空白。

    

    近期大（视觉）语言模型的进展使计算机使用智能体能够通过细粒度的键盘和鼠标操作自动完成数字工作流程。然而，这类智能体在长时程任务中仍然十分脆弱：误差不断累积，已执行的动作会留下意想不到的副作用，因此从失败中恢复的能力至关重要。然而，现有的补救手段仍不充分：环境重置在专有环境中不可行，动作回放在长时程下代价过于高昂，基于API的回滚又仅限于网页环境，这使得基于模型的恢复方法在很大程度上尚未被探索。为填补这一空白，我们提出了ReTrace，一个用于系统研究计算机使用智能体恢复行为的可扩展评估框架。我们将“转换一致的恢复”形式化为一个判定标准：从恢复后的状态与目标状态出发执行相同的下游动作，必须产生相同的结果。我们通过面向目标的（动作回放）将这一标准加以操作化……（摘要在此处截断）

    arXiv:2610.27954v1 Announce Type: new  Abstract: Recent advances in large (vision) language models have enabled computer-use agents that automate digital workflows via fine-grained keyboard and mouse operations. However, such agents remain fragile in long-horizon settings, where errors accumulate and executed actions leave unintended side effects, making the ability to recover from failures critical. Yet, existing remedies remain inadequate: environment resets are infeasible in proprietary settings, action replay is prohibitively expensive over long horizons, and API-based rollback is limited to web environments, leaving model-based recovery largely unexplored. To address this gap, we present ReTrace, a scalable evaluation framework for systematically studying recovery behavior in computer-use agents. We formalize Transition-Consistent Recovery: identical downstream actions from the recovered and target states must yield identical outcomes. We operationalize this criterion through goal
    
[^45]: 从DQI到C-DQI：用于测量计算式话语质量指数的自动编码本生成与评估

    From DQI to C-DQI: Automatic Codebook Generation and Evaluation for Measuring Computational Discourse Quality Index

    [https://arxiv.org/abs/2610.27937](https://arxiv.org/abs/2610.27937)

    该论文提出了一套从DQI到C-DQI的完整协议，通过比较自上而下、自下而上和混合三种方法自动生成基于大语言模型的编码本，并结合人工验证的外在代理指标和模型间一致性（IMA）指标进行评估，从而以可扩展、低偏见的方式测量议会话语质量。

    

    有效的民主协商对于公民参与和立法机构的健康发展至关重要。然而，评估议会话语质量传统上依赖于专家评分的、基于理论的编码本（CB），这种方法僵化、劳动密集且容易产生偏见。虽然大语言模型（LLM）为模式归纳和话语标注提供了可扩展的替代方案，但如何确保模式质量并减轻模型固有偏见仍然是一个明确的挑战。在这项工作中，我们提出了一个完整的基于LLM的编码本生成与评估协议。从广受认可的话语质量指数（DQI）出发，我们比较了自上而下（基于理论）、自下而上（数据驱动）和混合式三种生成情境适配模式的方法。为了评估这些模式，我们定义了一个由人工标注验证的外在代理指标，以及一个内在的模型间一致性（IMA）指标，该指标利用了不同……

    arXiv:2610.27937v1 Announce Type: new  Abstract: Effective democratic deliberation is vital for civic engagement and the health of legislative institutions. However, assessing parliamentary discourse quality is traditionally done using expert-scored, theory-based codebooks (CBs), which are rigid, labor-intensive, and prone to biases. While Large Language Models (LLMs) offer a scalable alternative for both schema induction and discourse annotation, ensuring schema quality and mitigating models' inherent bias remains a clear challenge. In this work, we propose a complete protocol for LLM-based codebook generation and evaluation. Starting from the widely accepted discourse quality Index (DQI), we compare top-down (theory-based), bottom-up (data-driven), and hybrid approaches for generating context-adapted schemas. To evaluate these schemas, we define an extrinsic proxy measure, validated by manual annotations, alongside an intrinsic inter-model agreement (IMA) metric, which leverages diff
    
[^46]: 为什么大语言模型会接受泛指句？利用心理语言学与模型衍生因素解释可接受性

    Why Do LLMs Accept Generics? Explaining Acceptability using Psycholinguistic and Model-Derived Factors

    [https://arxiv.org/abs/2610.27928](https://arxiv.org/abs/2610.27928)

    该研究首次利用心理语言学因素（属性普遍性、区分性、典型性）和模型衍生因素系统解释了大语言模型判断泛指句可接受性的机制，发现模型像人类一样权衡属性普遍性与区分性并偏好典型属性，但与人类不同，它们忽略引人注目的属性，而最看重陈述的事实真实性。

    

    心理语言学研究探讨了为什么人类会接受某些泛指句而拒绝另一些泛指句，并表明这些判断不仅取决于某种属性的普遍程度，人们还会考虑其他多种因素，例如该属性能否很好地将这一类别与其他类别区分开来，或者该属性是否是该类别的典型特征。受语言学和认知科学文献的启发，我们探讨了这些研究中的因素能在多大程度上解释大语言模型对泛指句的判断。我们在一个心理语言学数据集和自然出现的泛指句上对十二个模型进行了实验，结果表明，模型会权衡属性的普遍性与属性区分类别的能力，并以与人类相似的方式将二者结合起来，尽管这种结合不如人类那样清晰。与人类一样，模型也偏好典型属性，但与人类判断的相关解释不同的是，它们会忽略引人注目的属性。对大语言模型而言，最重要的是陈述在事实上是否为真，而人类相关因素的很大一部分影响

    arXiv:2610.27928v1 Announce Type: new  Abstract: Psycholinguistic studies explore why humans accept some generic statements but reject others by showing that these judgments depend on more than how common a property is. People consider several other factors such as how well it sets the kind apart from others or whether it is characteristic of the kind.   Inspired by the linguistic and cognitive literature, we explore to what extent factors from these studies explain how large language models (LLMs) judge generics. Our results with twelve models on a psycholinguistic dataset and on naturally occurring generics show that models weigh how common a property is against how well it distinguishes the kind, and combine the two much as people do, though less clearly. Like people, they favor characteristic properties, but unlike the accounts of human judgment, they ignore striking ones. What matters most to LLMs is whether a statement is factually true, and much of the influence of human factors
    
[^47]: ACE-Compression：一种以能力为中心的压缩大语言模型评估方法

    ACE-Compression: An Ability-Centric Evaluation of Compressed Large Language Models

    [https://arxiv.org/abs/2610.27913](https://arxiv.org/abs/2610.27913)

    该论文提出ACE-Compression框架，首次从核心能力（而非仅任务分数）的角度系统评估压缩后的大语言模型，涵盖程序性推理、知识真实性、社会认知与人格一致性四个维度，揭示了传统任务评估无法发现的能力悄然退化问题。

    

    模型压缩使大语言模型得以部署，但当今的评估方式并不能讲述完整的故事。标准做法仅测量压缩后保留的任务性能便止步于此，这忽略了一个关键事实：压缩会扰动分布在数十亿个权重中的知识，而仅凭任务分数无法揭示哪些底层能力得以保留、哪些能力在悄然崩溃。为解决这一问题，我们提出了ACE-Compression，一个以能力为中心的评估框架。我们不再仅仅询问“压缩后的模型还能正确回答吗？”，而是询问“它的哪些核心能力仍然完好无损？”该框架从四个面向部署的维度进行探测：程序性推理与约束执行、知识基础与真实性、社会认知与情感理解，以及人格一致性与风格一致性。在来自六个开源大语言模型的超过35个压缩检查点上——涵盖量化与剪枝——（摘要在此处截断）

    arXiv:2610.27913v1 Announce Type: new  Abstract: Model compression makes large language models deployable, but today's evaluation tells an incomplete story. Standard practice measures retained task performance and stops there. This misses a critical fact: compression perturbs the distributed knowledge embedded across billions of weights, and task scores alone cannot reveal which underlying abilities survive and which quietly collapse. To solve this question, we introduce ACE-Compression, an ability-centric evaluation framework. Instead of asking only "does the compressed model still answer correctly?", we ask "which of its core abilities remain intact?" Our framework probes four deployment-facing dimensions: procedural reasoning and constraint execution, knowledge grounding and truthfulness, social cognition and affective understanding, and persona fidelity and stylistic consistency. Across more than 35 compressed checkpoints from six open-weight LLMs-covering quantization and pruning-
    
[^48]: 大模型安全性中的极性盲视：对否定、反义词替换与对抗性框架的系统研究

    Polarity Blindness in LLM Safety: A Systematic Study of Negation, Antonym Substitution, and Adversarial Framing

    [https://arxiv.org/abs/2610.27908](https://arxiv.org/abs/2610.27908)

    大语言模型普遍存在“极性盲视”漏洞——在语法否定或反义词替换等极性改变的语言转换后无法识别有害意图，从而绕过安全防护，该漏洞在10个主流大模型和3种语言中均被验证，最脆弱模型的平均危害评分高达4.12。

    

    否定一直是语言模型在各类NLP任务和模态中面临的持续性挑战。我们展示了这一弱点可以通过一种我们称之为“极性盲视”的结构性漏洞来削弱大模型的安全防护机制：即在极性改变的语言转换之后，模型无法识别有害意图。我们研究了两种极性反转的形式：语法否定和反义词替换，并使用同义词替换作为保持极性的对照组。我们进一步测试了对抗性文本包装是否会放大这些失效现象。我们使用了一个包含260个提示的基准测试，涵盖10个最先进的大语言模型和3种语言（英语、西班牙语、捷克语），按潜在危害性进行分层，并采用经过验证的0-5分危害量表进行评估。我们发现极性盲视影响了大多数被评估的模型，其中最脆弱模型的平均危害评分高达4.12。该效应在不同类型的否定系统中持续存在，表明极性盲视……

    arXiv:2610.27908v1 Announce Type: cross  Abstract: Negation remains a persistent challenge for language models across NLP tasks and modalities. We show that this weakness can undermine LLM safety guardrails through a structural vulnerability we term polarity blindness: the failure to recognize harmful intent after polarity-altering linguistic transformations. We study two forms of polarity inversion: grammatical negation and antonym replacement, and use synonym replacement as a polarity-preserving control. We further test whether an adversarial textual wrapper amplifies these failures. Using a 260-prompt benchmark across 10 state-of-the-art LLMs and 3 languages (English, Spanish, Czech), stratified by harmfulness potential and evaluated on a validated 0-5 harm scale, we find that polarity blindness affects most evaluated models, with mean harm scores reaching 4.12 for the most vulnerable model. The effect persists across typologically differing negation systems, suggesting that polarit
    
[^49]: 多模态基础模型在日文金融文档上的分阶段持续适应

    Staged Continual Adaptation of Multimodal Foundation Models for Japanese Financial Documents

    [https://arxiv.org/abs/2610.27904](https://arxiv.org/abs/2610.27904)

    该研究追踪一个84亿参数多模态模型在日文金融文档上的三个训练阶段，发现各阶段贡献不可分解且交互非加性——数学蒸馏会覆盖已习得的金融知识，因此最佳检查点因任务而异，关键在于检查点保留与调度策略而非增加训练阶段。

    

    多模态基础模型的分阶段后训练将视觉-语言对齐、推理蒸馏和领域微调串联起来，然而各阶段之间如何相互权衡能力仍缺乏充分刻画。通过追踪一个84亿参数模型从未经微调的基线出发，在日文金融文档上经历三个训练阶段（在五个文本输入基准上评估），我们发现各阶段之间的交互是非加性的：仅针对数学的蒸馏会在程序性数学技能得以保留的同时，覆盖先前习得的金融知识，而领域阶段所修复的内容多于其新增的内容。跳过阶段的消融实验对领域知识最有利，但会放弃数值推理方面的增益，因此各阶段的贡献不可分解，最佳检查点取决于具体任务，这对检查点的保留与调度（而非增加新阶段）具有指导意义。

    arXiv:2610.27904v1 Announce Type: new  Abstract: Staged post-training of multimodal foundation models chains vision-language alignment, reasoning distillation, and domain tuning, yet how each phase trades capabilities against the others remains under-characterised. Tracking an 8.4B-parameter model from an un-fine-tuned baseline through three training phases on Japanese financial documents (evaluated on five text-input benchmarks), we find the phases interact non-additively: math-only distillation overwrites the financial knowledge acquired earlier even as procedural math skill survives, and the domain phase repairs more than it installs. A skip-phase ablation is best for domain knowledge but forfeits numerical-reasoning gains, so per-phase contributions are not decomposable and the best checkpoint is task-dependent, with implications for checkpoint retention and scheduling rather than added stages.
    
[^50]: CollabSciReM：通过科学感知的奖励分解来对齐语言模型以进行科学协作对话

    CollabSciReM: Aligning Language Models for Scientific Conversations via Science-Aware Reward Decomposition

    [https://arxiv.org/abs/2610.27901](https://arxiv.org/abs/2610.27901)

    该论文提出 CollabSciReM 框架，通过将会话级奖励分解为对话质量、科学严谨性和结构效率三个目标条件化组件来对齐语言模型，使其胜任多轮科学协作对话，并配套推出基于六个维度的 SciQuality 成对评估方法。

    

    摘要（arXiv:2610.27901v1，公告类型：交叉）：大型语言模型（LLM）正越来越多地被用作研究合作者，在多轮交互中协助假设完善、方法规划和技术分析。在这一场景下，一个有用的助手必须做到准确、方法论上有依据、逻辑一致且令牌高效。目前的对齐方法很少能同时兼顾所有这些要求。经指令微调的模型往往通过冗长的表达来换取对话质量，而我们研究表明，与科学无关的通用协作奖励则会以牺牲科学严谨性为代价来压缩回答篇幅。我们提出 CollabSciReM，一个目标条件化的奖励框架，将会话级奖励分解为对话质量、科学严谨性和结构效率三个部分。我们还引入了 SciQuality，这是一种针对多轮研究对话的成对评估方法，基于300篇保留的科学论文构建，并在六个维度上进行评分：方法可靠性、论证合理性、逻辑……（原文摘要在此处被截断）

    arXiv:2610.27901v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly used as research collaborators for hypothesis refinement, methodological planning, and technical analysis across multi-turn interactions. In this setting, a useful assistant must be accurate, methodologically justified, logically consistent, and token-efficient. Current alignment approaches rarely achieve all of these together. Instruction-tuned models gain conversational quality through verbosity, and we show that science-agnostic general collaborative rewards compress responses at the expense of scientific rigor. We propose CollabSciReM, a goal-conditioned reward framework that decomposes the session-level reward into conversational quality, scientific rigor, and structural efficiency. We also introduce SciQuality, a pairwise evaluation of multi-turn research conversations grounded in 300 held-out scientific papers and scored on six dimensions: methodological soundness, justification, log
    
[^51]: 用难以命名的视觉刺激测试人类和多模态大语言模型的联想创造力

    Testing Associative Creativity in Humans and Multimodal LLMs with Difficult-to-Name Visual Stimuli

    [https://arxiv.org/abs/2610.27894](https://arxiv.org/abs/2610.27894)

    该研究通过SilCC和PixCC两个可控难度数据集评估人类与多模态大语言模型对难以命名的抽象视觉刺激的联想创造力，发现人类的联想描述更多样、更新颖、语义距离更大，而模型的有效性随难度增加而下降。

    

    我们研究了人类和多模态大语言模型（MLLMs）在描述抽象视觉刺激时的联想创造力。虽然概念联想是人类交流的核心，但当视觉刺激越来越难以映射到熟悉的语义概念时，人类和模型如何适应仍不清楚。我们使用两个具有受控难度操纵的数据集，从原创性和有效性这两个创造力维度评估了多个最先进的多模态大语言模型和人类参与者。SilCC通过逐渐更复杂的抽象形状组合来改变命名难度，PixCC则通过增加随机视觉噪声水平来改变难度。结果表明，人类比多模态大语言模型产生更多样化的联想描述，包括相对于参考词库具有更大的语义距离、更高的熵以及更多新颖的表达。随着难度增加，有效性下降。

    arXiv:2610.27894v1 Announce Type: new  Abstract: We investigate associative creativity in descriptions of abstract visual stimuli by humans and multimodal large language models (MLLMs). While conceptual association is central to human communication, it remains unclear how humans and models adapt when visual stimuli become increasingly difficult to map onto familiar semantic concepts. We evaluate several state-of-the-art MLLMs and human participants along two dimensions of creativity, originality and effectiveness, using two datasets with controlled manipulations of difficulty. SilCC varies naming difficulty through progressively more complex combinations of abstract shapes, and PixCC varies difficulty through increasing levels of random visual noise. Results show that humans produce more diverse associative descriptions than MLLMs, including greater semantic distance, higher entropy, and more novel expressions relative to reference lexica. As difficulty increases, effectiveness decline
    
[^52]: 从词元到文件：AI 参与代码的多粒度检测与定位

    From Tokens to Files: Multi-Granularity Detection and Localization of AI-Involved Code

    [https://arxiv.org/abs/2610.27863](https://arxiv.org/abs/2610.27863)

    提出了一种基于冻结代码语言模型上双低秩适配器逐词元对数似然比的框架，可在词元证据基础上对 AI 参与代码进行行、块、文件多粒度的检测与定位。

    

    随着大语言模型在代码生成中的使用日益增多，亟需能够区分纯人工编写代码与“AI 参与代码”（即文件中任何部分，无论多么微小，由模型生成或经模型实质性修改）的方法。除了标记此类文件之外，我们还旨在定位可归因于 AI 的代码行或代码块。现有方法将文件压缩为单一分数，依赖于表面风格特征或全局代码表示，当文件中仅有少数代码块由 AI 参与时，这种方式会平均化掉相关证据。我们提出了一个在词元级别建模 AI 创作归属、并在行、块和文件三个粒度上进行预测的框架。一个冻结的代码语言模型配备了两个低秩适配器，分别在不同机器编写和人工编写的文件上训练；两者的逐词元对数似然比为 AI 创作归属提供了局部证据。轻量级的预测头利用周围的代码上下文，在行和块内部聚合这些证据……

    arXiv:2610.27863v1 Announce Type: cross  Abstract: The increasing use of large language models for code generation calls for methods that distinguish purely human-written code from AI-involved code: files in which any part, however small, was generated or substantially modified by a model. Beyond flagging such files, we aim to localize the lines or blocks attributable to AI. Existing approaches compress a file into a single score, from surface stylistic features or a global code representation, which averages away the evidence when only a few blocks are AI-involved. We propose a framework that models AI authorship at the token level and predicts at the line, block, and file levels. A frozen code language model is equipped with two low-rank adapters, trained separately on machine-written and human-written files; their per-token log-likelihood ratio provides local evidence of AI authorship. Lightweight heads aggregate this evidence within lines and blocks, using the surrounding code cont
    
[^53]: UvA-MT 团队参与 WMT26 通用翻译共享任务

    UvA-MT's Participation in the WMT26 General Translation Shared Task

    [https://arxiv.org/abs/2610.27852](https://arxiv.org/abs/2610.27852)

    UvA-MT 团队在 WMT26 通用机器翻译任务中，利用大语言模型顺序采样生成翻译候选，并通过 LLM 成对评委的循环投票机制选出最佳译文，参与了全部 23 个翻译方向。

    

    本文介绍了 UvA-MT 团队参加 WMT 2026 通用机器翻译共享任务的提交系统，该系统以无约束赛道参与全部 23 个翻译方向。今年，我们充分借助大语言模型（LLM）在机器翻译中的测试时方法。具体而言：（1）我们使用带有润色提示的顺序采样为每个源语言片段生成一组翻译候选，该策略已被证明优于传统的并行采样；（2）我们采用大语言模型作为成对评委，通过循环投票机制选出最佳候选译文，在 WMT25 基准上的离线评估显示，该方法相比逐点式 best-of-N 选择有非常明显的提升。这两种策略相结合，构成了我们参加 WMT26 的最终提交系统。

    arXiv:2610.27852v1 Announce Type: new  Abstract: This paper presents UvA-MT's submission to the WMT 2026 General Machine Translation shared task, competing in the unconstrained track across all 23 translation directions. This year, we fully leverage the test-time methods of Large Language Models (LLMs) for machine translation. Specifically, (1) we use sequential sampling with a refinement prompt to generate a pool of translation candidates for each source segment, a strategy shown to outperform traditional parallel sampling; and (2) we employ LLMs as pairwise judges to select the best candidate via a round-robin voting mechanism, with offline evaluation on the WMT25 benchmark showing a very clear improvement over point-wise best-of-N selection. Combined, these two strategies form our WMT26 submission.
    
[^54]: SciDiagBench：一个面向人类对齐评估的科学图表类型专用基准

    SciDiagBench: A Type-Specific Benchmark for Human-Aligned Evaluation of Scientific Diagrams

    [https://arxiv.org/abs/2610.27841](https://arxiv.org/abs/2610.27841)

    该论文提出了SciDiagBench，一个按图表类型系统分类、并与人类专家评估偏好显式对齐的科学图表评估基准，弥补了现有基于视觉语言模型的评估基准缺乏类型区分和人类对齐的不足。

    

    最近的智能体在根据文本提示生成图像方面展现出强大的能力，这推动了它们在科学图表生成中的应用。科学图表自然涵盖多种类型，每种类型都具有独特的视觉结构和内容组织方式。因此，在同行评审及其他科学评估过程中，人类专家在评估图表质量时会自然地考虑类型特定的评判标准。然而，现有的基于视觉语言模型（VLM）的评估基准未能反映上述人类评估实践，主要原因有两点：1）基准数据集通常缺乏对不同图表类型的系统分类，从而对所有图表采用相同的通用评估标准；2）其评估标准往往是通过启发式方法设计的，没有与人类评估偏好进行显式对齐。受这些局限性的启发，我们提出了SciDiagBench，它包含一个基准数据集，该数据集系统地……

    arXiv:2610.27841v1 Announce Type: new  Abstract: Recent agents have demonstrated strong capabilities in generating images from textual prompts, motivating their utilization for scientific diagram generation. Scientific diagrams naturally span diverse types, each characterized by distinct visual structures and content organization. Consequently, during peer review and other scientific evaluation, human experts naturally consider type-specific criteria when assessing their quality. However, existing VLM-based evaluation benchmarks fail to reflect the above human evaluation practices for two main reasons: 1) benchmark datasets typically lack a systematic categorization of different diagram types and thus adopt the same generic criteria for all diagrams, and 2) their evaluation criteria are often designed heuristically, without explicit alignment to human evaluation preferences. Motivated by these limitations, we introduce SciDiagBench, which consists of a benchmark dataset that systematic
    
[^55]: SMART：基于上下文评分准则的识别转写文本语义分段映射与评估

    SMART: Semantic Segment Mapping and Assessment of Recognition Transcripts with Contextual Rubrics

    [https://arxiv.org/abs/2610.27820](https://arxiv.org/abs/2610.27820)

    提出SMART框架，将ASR识别文本与参考文本的差异映射为语义片段并利用上下文评分准则标注其语义后果，从而实现可追溯的语义评估，其提出的语义影响错误率（SIER）性能显著优于直接使用大语言模型判断。

    

    词错误率（WER）对所有词汇编辑赋予相同的代价，尽管它们造成的语义后果各不相同；而整体语义评分又无法识别出哪些具体错误改变了原本想表达的意思。我们将语音识别（ASR）的语义评估形式化为具有上下文感知能力的分段级后果评估，并提出了SMART（Semantic Segment Mapping and Assessment of Recognition Transcripts with Contextual Rubrics，基于上下文评分准则的识别转写文本语义分段映射与评估），一个用于可追溯语义评估和有针对性ASR纠错的框架。SMART将参考文本与识别文本之间的差异映射为连贯的语义片段，并应用结构化的上下文评分准则为各片段分配明确的后果标签。这些标签进一步支持语义影响错误率（SIER），可在话语、片段和词汇三个层面量化具有语义影响的错误。以人类多数投票标注为基准的评估中，SMART达到了0.843的F1分数，而直接使用大语言模型判断仅为0.745。跨多个大语言模型的评估进一步表明，SMART相对不敏感……（原文摘要在此处截断）

    arXiv:2610.27820v1 Announce Type: cross  Abstract: Word error rate (WER) assigns equal cost to lexical edits despite their different semantic consequences, while overall semantic scores cannot identify which individual errors alter the intended meaning. We formulate ASR semantic evaluation as context-aware, segment-level consequence assessment and propose Semantic Segment Mapping and Assessment of Recognition Transcripts with Contextual Rubrics (SMART), a framework for traceable semantic evaluation and targeted ASR correction. SMART maps reference-hypothesis discrepancies into coherent segments and applies a structured contextual rubric to assign explicit consequence labels. These labels support Semantic Impact Error Rate (SIER), which quantifies consequential errors at utterance, chunk, and lexical levels. Against majority-vote human labels, SMART achieves 0.843 F1 versus 0.745 for direct LLM judgment. Evaluation across multiple LLMs further demonstrates that SMART is relatively insen
    
[^56]: ProVe：提议者-验证者解码

    ProVe: Proposer-Verifier Decoding

    [https://arxiv.org/abs/2610.27815](https://arxiv.org/abs/2610.27815)

    ProVe反转了推测解码中提议者与验证者的角色，让大模型通过单次前向传播直接提议token块并加以验证，无需训练独立草稿头即可实现2.5-2.7倍推理加速，且无损版本能完全复现原始贪婪解码结果。

    

    推测解码通过让小模型串行起草token、大模型并行验证来加速自回归（AR）语言模型。我们反转了这两个角色。从一个标准的AR检查点出发，我们将大模型微调为一个提议者，使其能够在单次前向传播中预测未来的一个token块，然后用一次额外的前向传播来验证所提议的块。这就是ProVe，即提议者-验证者解码。验证者可以是一个较小的AR模型，也可以是提议者本身；后者构成ProVe-lossless，它能够复现提议者的贪婪AR解码结果。块提议本身已经提供了可观的加速空间：当块大小为16时，在GSM8K上提议块与AR解码的平均匹配前缀达到6.6个token。在六个基准测试上，使用32B提议者，ProVe-lossless无需训练单独的草稿头即可达到平均2.5倍的AR吞吐量。使用较小的3B验证器时，ProVe达到平均2.7倍的吞吐量。

    arXiv:2610.27815v1 Announce Type: new  Abstract: Speculative decoding accelerates autoregressive (AR) language models by letting a small model draft tokens serially and a large model verify them in parallel. We reverse these roles. Starting from a standard AR checkpoint, we fine-tune the large model into a proposer that predicts a block of future tokens in a single forward pass, then use one additional forward pass to verify the proposed block. This gives ProVe, Proposer-Verifier decoding. The verifier can be a smaller AR model or the proposer itself; the latter gives ProVe-lossless, which reproduces the proposer's greedy AR decoding. Block proposals already provide substantial room for acceleration: with a block size of 16, the proposed block has an average AR-matching prefix of 6.6 tokens on GSM8K. Across six benchmarks with a 32B proposer, ProVe-lossless reaches 2.5x average AR throughput without training a separate draft head. With a small 3B verifier, ProVe reaches 2.7x average th
    
[^57]: 大语言模型中语言回路与算术回路是否相互分离？

    Are Linguistic and Arithmetic Circuits Dissociated in LLMs?

    [https://arxiv.org/abs/2610.27790](https://arxiv.org/abs/2610.27790)

    本研究通过机制可解释性的回路发现方法，在五个大语言模型上发现语言与算术回路的相似性主要由任务领域而非句法相似性决定，表明大语言模型以分离的方式处理语言与算术。

    

    大语言模型（LLM）既具备语言知识（如通过语法判断能力所体现的），又具备借助语言执行任务的能力（如解答数学应用题）。受神经语言学中关于这些能力在人脑中神经分离现象研究的启发，本研究考察了大语言模型中语言回路与算术回路的重叠情况。具体而言，我们使用受控材料以及机制可解释性领域的回路发现方法，检验了为语言任务和算术任务所提取的回路之间的重叠及其因果效应。在五个大语言模型上的实验表明，回路相似性主要由任务领域（语言 vs. 算术）所解释，而非由输入和输出词表、也非由自然语言句子与算术表达式之间的句法相似性所解释。这一发现表明，大语言模型可能以分离的方式处理语言与算术……

    arXiv:2610.27790v1 Announce Type: new  Abstract: Large language models (LLMs) exhibit both knowledge of language, rendered by, for example, grammatical judgment ability, and abilities to perform tasks through language, such as math-word problem solving. Motivated by neurolinguistic studies of their neural dissociation in human brains, in this study, we investigate the overlap of their circuits in large language models (LLMs). Specifically, we examine the overlap and causal effects between the circuits extracted for linguistic and arithmetic tasks, using controlled materials and circuit discovery methods from the mechanistic interpretability field. Our experiments with five LLMs show that circuit similarity is mostly explained by task domains (linguistic vs. arithmetic), rather than by their input and output vocabularies or by the syntactic similarities between natural-language sentences and arithmetic expressions. This finding suggests that LLMs may process linguistic and arithmetic in
    
[^58]: 给模型一根手指：视觉语言模型计数能力的发展对齐评估

    Give the Model a Finger: Developmentally Aligned Evaluation of Counting in Vision-Language Models

    [https://arxiv.org/abs/2610.27756](https://arxiv.org/abs/2610.27756)

    该论文受儿童计数发展过程启发，提出一种无需训练的智能体评估框架，让视觉语言模型通过“指认标记”等认知动作进行计数，将模型通常隐藏的串行视觉推理过程转化为可解释、可审计的行为分析。

    

    视觉语言模型（VLM）在小集合上表现精确，而在较大集合上则退化为估计，这一特征与尚未掌握计数的儿童颇为相似。儿童通过认知动作跨越这一障碍：一边念数词一边指向每个物品，把已数过的物品移到一边，或对物品阵列进行分组。我们以无需训练的智能体循环方式，为VLM提供同样的动作——这些动作只是不具备自身视觉智能的“哑”图像工具。我们的核心主张在于测量而非干预：一个通过标记进行计数的模型，会将其通常隐藏的串行过程留在图像上，而计数的发展性原则将这些标记转变为对串行视觉推理的可解释审计：每个标记都会依据点级真值被分类为命中、重复、幻影或遗漏；答案会与标记数量进行核对；顺序无关性则通过强制相反的遍历顺序来检验。在9个模型、682张图像的实验中……

    arXiv:2610.27756v1 Announce Type: new  Abstract: Vision-language models (VLMs) are exact on small sets and drift into estimation on larger ones, a profile reminiscent of children before they master counting. Children cross the barrier with epistemic actions: pointing at each item while reciting number words, moving counted items aside, partitioning the array. We give VLMs the same actions as dumb image tools with no visual intelligence of their own in a training-free agent loop. Our central claim is about measurement, not intervention: a model that counts by marking leaves a normally hidden serial procedure on the image, and the developmental principles of counting turn those marks into an interpretable audit of serial visual reasoning: every tag is classified against point-level ground truth as on-target, duplicate, phantom, or omitted, the answer is checked against the number of tags, and order irrelevance is tested by forcing opposite traversal orders. Across 9 models, 682 images, a
    
[^59]: 道听途说还是事实？框架与引出方式如何塑造大语言模型的概率更新

    Hearsay or Fact? How Framing and Elicitation Shape LLM Probability Updates

    [https://arxiv.org/abs/2610.27745](https://arxiv.org/abs/2610.27745)

    该论文提出了一种衡量大语言模型对单条新信息概率更新幅度的“方向性移动”指标，并发现信息的框架呈现方式（直接告知还是传闻转述）与概率引出方式（单个概率还是五个加权估计）会显著影响模型的概率更新。

    

    大语言模型（LLM）日益被用作判断助手以及心理学实验中的替代被试，然而人们对它们的概率估计在响应单条新信息时会发生多大变化却知之甚少。我们将 δ-SNLI 中的前提-假设条目与一个增强项和一个削弱项分别配对，在分离的调用中分别引出先验概率和后验概率，并测量方向性移动 s：即按标注方向赋以符号的后验平均变化量，从而使先验的影响相互抵消。我们变化了信息更新的呈现方式——是直接告知（“这里还有一些你了解到的东西”）还是传闻转述（“有人说”），以及模型是给出单个概率还是五个加权估计。我们测试了七种配置：两个本地模型（Qwen3-8B、OLMo-2-7B；240 对）和五种通过智能体命令行框架运行的配置（120 对），且假设与分析规则在正式运行前即已固定（不公开……）

    arXiv:2610.27745v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as judgment assistants and as stand-in participants in psychology experiments, yet little is known about how much their probability estimates move in response to a single new piece of information. We pair premise-hypothesis items from $\delta$-SNLI with one strengthener and one weakener each, elicit the prior and the posteriors in separate calls, and measure the directional movement s: the mean posterior change signed by the annotated direction, so that the prior cancels out. We vary whether the update is stated ("Here is something else you learned") or reported ("Someone said that"), and whether the model gives one probability or five weighted estimates. We test seven configurations: two local models (Qwen3-8B, OLMo-2-7B; 240 pairs) and five configurations run through agent command-line harnesses (120 pairs), with hypotheses and analysis rules fixed before the formal runs (not publicl
    
[^60]: 在分解式Text-to-SQL生成中追踪复发性与涌现性幻觉模式

    Tracing Recurrent and Emergent Hallucination Patterns in Decomposed Text-to-SQL Generation

    [https://arxiv.org/abs/2610.27720](https://arxiv.org/abs/2610.27720)

    该论文提出一个下钻式评估框架，将Text-to-SQL生成过程分解为复杂度递增的可执行子查询，从而区分错误在早期步骤已出现并延续到最终查询的“复发性幻觉”与仅在最终查询才显现的“涌现性幻觉”，揭示大模型幻觉在生成路径中的演变规律。

    

    大型语言模型虽然能在Text-to-SQL基准测试中取得优异表现，但仍会产生模式（schema）错误和查询逻辑错误。大多数评估仅在最终生成的查询中刻画此类失败，因此难以判断相关错误是否在更早、更简单的步骤中就已经出现，还是仅在最终阶段才显现。我们提出了一个下钻式（drill-down）评估框架，将Text-to-SQL查询分解为复杂度逐步提升的可执行子查询以及对应的自然语言问题，从而能够在整条生成路径上检验基于规则的幻觉类别。我们区分了两类幻觉：“复发性幻觉”，即同一基于规则的类别在较早步骤中被观察到，并在最终查询中再次出现；“涌现性幻觉”，即该类别仅在最终步骤被观察到，而在同一路径的较早阶段未被检测到。在扩展的BIRD-mini基准上对五个大语言模型的评估中，……（原文摘要至此截断）

    arXiv:2610.27720v1 Announce Type: new  Abstract: Large language models can achieve strong Text-to-SQL benchmark performance while still producing schema and query-logic errors. Most evaluations characterize such failures only in the final generated query, making it difficult to determine whether a related error was also observed in an earlier, simpler step or appears only at the final stage. We present a drill-down evaluation framework that decomposes Text-to-SQL queries into progressively more complex executable sub-queries and corresponding natural-language questions, allowing rule-based hallucination categories to be examined across a generation path. We distinguish recurrent hallucinations, in which the same rule-based category is observed in an earlier step and again in the final query, from emergent hallucinations, in which the category is observed at the final step without an earlier detection in the same path. Across five LLMs evaluated on an expanded BIRD-mini benchmark, sever
    
[^61]: 经验记忆何时能帮助闭环LLM驾驶智能体？基于重采样预热池、令牌匹配填充器与审计基线的受控评估

    When Does Experience Memory Help Closed-Loop LLM Driving Agents? A Controlled Evaluation with Resampled Warmup Pools, Token-Matched Fillers, and Audited Baselines

    [https://arxiv.org/abs/2610.27714](https://arxiv.org/abs/2610.27714)

    本研究通过在审计的HighwayEnv测试平台上控制预热池抽样与提示令牌两大混淆因素，发现经验记忆为闭环LLM驾驶智能体带来的表面收益主要来自预热经验池的随机抽样，而非记忆机制本身。

    

    经验记忆何时能帮助闭环LLM驾驶智能体？已发表的收益无法回答这一问题，因为我们发现最关键的两个因素——预热池中包含哪些预热经验、以及注入的令牌对提示词产生什么影响——在现有研究中均未被控制。我们通过在一个经过审计的HighwayEnv测试平台（1Hz，决定性对比使用gpt-4o-mini）上进行受控评估来回答这一问题。我们将预热池视为重采样的随机因素（十个互不相交的池），并加入令牌匹配的无关填充控制、经oracle认证的诊断、谨慎定价指标以及对基线移植版本的量化审计。探测套件涵盖平文本检索（DiLu移植版）、单帧网格移植版（受RESPOND启发）、我们的情景时间图记忆、无需检索的整合规则手册以及平铺的K步观察历史基线。由此得出四项发现，每项均为该测试平台的特性。(1) 报告中的记忆收益归属于预热抽样的运气。代码库……

    arXiv:2610.27714v1 Announce Type: new  Abstract: When does experience memory help a closed-loop LLM driving agent? Published gains cannot say, because the two factors we find to matter most, which warmup experiences the pool contains and what any injected tokens do to the prompt, are controlled in none of them. We answer with a controlled evaluation on one audited HighwayEnv testbed (1Hz, gpt-4o-mini for the decisive contrasts). We treat the warmup pool as a resampled random factor (ten disjoint pools) and add token-matched irrelevant-filler controls, oracle-certified diagnostics, caution-pricing metrics, and quantified audits of the baseline ports. The probe suite spans flat-text retrieval (a DiLu port), a single-frame grid port (RESPOND-inspired), our episodic temporal-graph memory, a retrieval-free consolidated rulebook, and a flat K-step observation-history baseline. Four findings follow, each a property of this testbed. (1) Headline memory gains belong to the warmup draw. The repo
    
[^62]: 超越校准：跨语言与本地化场景下大语言模型置信度的时序评估

    Beyond Calibration: Temporal LLM Confidence Assessment Across Languages and Local Settings

    [https://arxiv.org/abs/2610.27702](https://arxiv.org/abs/2610.27702)

    受元记忆理论启发，该研究首次区分了大语言模型求解前（FOK）与求解后（JOL）两个时间阶段的置信度信号，发现求解后置信度在多语言与本地化场景中具有更强的认知对齐，而成本更低的求解前信号则校准较弱。

    

    大语言模型被期望能够估计自身的不确定性，其置信度信号可用于指导检索、自适应算力分配、路由以及选择性弃答。我们研究了在问题求解的不同时间阶段所测量的置信度信号是否编码了不同形式的认知信息，以及它们在以英语为中心的场景之外是否依然可靠。受经典元记忆理论的启发，我们区分了求解前的预期性置信度（知道感，FOK）与求解后的回溯性置信度（学习判断，JOL）。我们在两个前沿大语言模型、多个推理领域、经翻译的马拉地语基准测试，以及一个新构建的真正本地化的马拉地语事实性问题基准上评估了这些信号。在所有设置中，成本较低的求解前信号仍然校准较弱，而求解后置信度尽管需要额外的推理成本，通常能提供更强的认知对齐。

    arXiv:2610.27702v1 Announce Type: cross  Abstract: Large language models are expected to estimate their own uncertainty, with confidence signals used to guide retrieval, adaptive compute allocation, routing, and selective abstention. We investigate whether confidence signals measured at different temporal stages of problem solving encode distinct forms of epistemic information and remain reliable beyond English-centric settings. Inspired by classical meta-memory theory, we distinguish anticipatory pre-solve confidence (Feeling-of-Knowing, FOK) from retrospective post-solve confidence (Judgement-of-Learning, JOL). We evaluate these signals across two frontier LLMs, multiple reasoning domains, translated Marathi benchmarks, and a newly constructed benchmark of genuinely local Marathi factual questions. Across settings, the cheaper pre-solve signal remains weakly calibrated, while post-solve confidence generally provides stronger epistemic alignment despite its additional inference cost. 
    
[^63]: 记忆即代码：基于编码化长期记忆的声明式检索

    Memory as Code: Declarative Retrieval over Codified Long-Term Memory

    [https://arxiv.org/abs/2610.27700](https://arxiv.org/abs/2610.27700)

    提出Memory as Code (MaC)框架，以共享代码抽象统一设计记忆构建与声明式检索，通过原子函数层、关系层、模块化组合层的三层可执行记忆结构，解决长期记忆中事实分散、关系动态演变导致难以组合检索的问题。

    

    在长期记忆系统中，记忆从交互历史中构建，并根据查询进行检索，从而增强大语言模型智能体的推理能力。在复杂场景中，事实分散在各个会话之中，且它们之间的关系随时间不断演变，这使得现有记忆系统难以组合这些事实以适应多样化的查询。受代码的模块化与可组合性的启发，我们提出了Memory as Code (MaC)，这是一个统一的记忆框架，通过共享的代码抽象来联合设计记忆构建与检索。MaC将编码化构建（建立可执行的记忆抽象）与声明式检索（将查询转化为结构化的检索程序）相结合。具体而言，编码化构建了一个三层可执行记忆结构，从原子函数层出发，经过关系层，再到模块化组合层逐层递进。声明式检索……

    arXiv:2610.27700v1 Announce Type: new  Abstract: In long-term memory systems, memories are constructed from interaction histories and retrieved in response to queries, enhancing the reasoning ability of LLM agents. In complex scenarios, facts are scattered across sessions and their relationships evolve over time, making it difficult for existing memory systems to compose them to adapt to diverse queries. Motivated by the modularity and composability of code, we propose Memory as Code (MaC), a unified memory framework that jointly designs memory construction and retrieval through a shared code abstraction. MaC couples codified construction, which establishes executable memory abstractions, with declarative retrieval, which translates queries into structured retrieval programs. Specifically, codified construction builds a three-layer executable memory structure that progresses from an atomic function layer, through a relational layer, to a modular composition layer. Declarative retrieval
    
[^64]: PhonoBridge：错误感知的基于音素介导的肌电信号到文本重建

    PhonoBridge: Error-Aware Phoneme-Mediated EMG-to-Text Reconstruction

    [https://arxiv.org/abs/2610.27678](https://arxiv.org/abs/2610.27678)

    该论文提出PhonoBridge框架，以音素作为肌电信号与大语言模型之间的中间表示，并通过错误特征引导增强（EPGA）模拟前端解码错误来扩充训练数据，从而在配对肌电-文本数据稀缺的情况下实现更准确的无声语音到文本重建。

    

    肌电图（EMG）提供了一种无创手段，可将无声发音的语音解码为文本，但由于配对的肌电-文本数据有限，准确解码仍然具有挑战性。我们提出了PhonoBridge，这是一个两阶段框架，包含一个通道感知的肌电到音素前端和一个用于文本重建的大语言模型（LLM）后端。前端在配对的肌电-文本数据上训练，包括通道级特征提取、跨通道融合和电极嵌入。以音素作为中间表示，我们通过将LibriSpeech转录文本转换为音素-文本对来扩充后端的训练数据，而无需额外的肌电记录。为了使大语言模型适应前端噪声，我们引入了错误特征引导增强（EPGA），该方法根据从前端预测中估计的错误模式，对源自转录文本的音素序列进行扰动，为大语言模型生成额外的训练对……

    arXiv:2610.27678v1 Announce Type: cross  Abstract: Electromyography (EMG) provides a non-invasive means of decoding silently articulated speech into text, but accurate decoding remains challenging due to limited availability of paired EMG-text data. We introduce PhonoBridge, a two-stage framework comprising a channel-aware EMG-to-phoneme frontend and a large language model (LLM) backend for text reconstruction. The frontend is trained on paired EMG-text data that consists of channel-wise feature extraction, cross-channel fusion, and electrode embeddings. With phonemes as the intermediate representation, we expand the backend's training data by converting LibriSpeech transcripts into phoneme-text pairs without additional EMG recordings. To adapt the LLM to frontend noise, we introduce Error-Profile-Guided Augmentation (EPGA), which perturbs transcript-derived phoneme sequences according to error patterns estimated from frontend predictions, generating additional training pairs for the L
    
[^65]: CovNav：基于共形预测引导的目标物体导航自适应推理

    CovNav: Conformal Prediction-Guided Adaptive Inference for Object Goal Navigation

    [https://arxiv.org/abs/2610.27677](https://arxiv.org/abs/2610.27677)

    CovNav提出了一种无需训练的框架，利用共形预测来量化空间判断的不确定性，从而自适应地决定混合推理LLM在目标物体导航中何时使用快速“系统1”模式或缓慢“系统2”模式，在推理质量与效率之间实现最优权衡。

    

    在目标物体导航中，具身智能体需要在先前未见过的环境中寻找目标物体（例如“找一张床”），并依赖诸如物体共现等空间常识。大型语言模型编码了这种常识，但在机器人上部署它们会面临推理质量与效率之间的权衡：基于云端的模型存在延迟和隐私风险，而小型端侧LLM速度快但容易产生幻觉性的空间判断。混合推理LLM同时提供了快速的非思考模式和缓慢的思考模式，这给具身智能体提出了一个关键问题：什么时候值得付出缓慢思考的代价？我们提出了CovNav，一个无需训练的框架，它通过共形预测（CP）回答了这个问题，将双过程（系统1/系统2）推理转变为不确定性感知的决策。与最近的决策模型（如Jev）类似，CovNav将每个空间查询视为结构化二分类（摘要在此处截断）。

    arXiv:2610.27677v1 Announce Type: cross  Abstract: In object goal navigation (ObjectNav), an embodied agent must find a target object (e.g., "find a bed") in a previously unseen environment, relying on spatial commonsense such as object co-occurrence. Large Language Models (LLMs) encode this commonsense, but deploying them on robots exposes a trade-off between reasoning quality and efficiency: cloud-based models incur latency and privacy risks, whereas small on-device LLMs are fast but prone to hallucinated spatial judgments. Hybrid reasoning LLMs offer both a fast non-thinking mode and a slow thinking mode, which raises a key question for embodied agents: when is slow thinking worth its cost? We propose CovNav, a training-free framework that answers this question with Conformal Prediction (CP), turning dual-process (System 1 / System 2) inference into an uncertainty-aware decision. Much like recent decision models such as Jev, CovNav casts each spatial query as a structured binary dec
    
[^66]: 语料库质量提升，检索效用并未提升：语音转问答流水线中预处理与精炼阶段的消融研究

    Corpus Quality Improves, Retrieval Utility Does Not: An Ablation of Preprocessing and Refinement in a Speech-to-QA Pipeline

    [https://arxiv.org/abs/2610.27652](https://arxiv.org/abs/2610.27652)

    本研究通过消融实验发现，尽管预处理和精炼阶段能显著提升语料库质量并获得LLM评审与人类专家的一致认可，但这种质量提升并不能转化为下游检索准确率的提升，其中转录改写反而使检索准确率下降了0.061。

    

    使用语言模型流水线构建训练语料库的实践者，会在开启预处理或精炼（后处理）阶段能够提升语料库质量时保留该阶段。我们对一条将会议语音转换为问答对的生产级流水线进行了消融实验。预处理阶段负责改写转录文本，精炼阶段负责过滤生成的问答对。通过分别开启和关闭这两个阶段，我们从相同的40个韩语医疗和金融会议中得到了四个语料库。就语料库本身而言，随着各阶段的开启，质量随之提升。四个大语言模型评审与六位人类领域专家对四个语料库给出了相同的排序，完全精炼的语料库在1-5分制上分别领先+0.18和+0.22。随后我们探究这一提升能否传递到下游任务。我们将每个语料库提供给六个模型进行推理并测量检索准确率，结果该排序不再成立。开启转录文本改写会使检索准确率降低0.061，这是唯一在校正后仍然显著的阶段效应……

    arXiv:2610.27652v1 Announce Type: cross  Abstract: Practitioners building training corpora with a pipeline of language models keep a preprocessing or refinement (post-process) stage when switching it on raises the corpus quality. We ablate a production pipeline that turns conference speech into question-answer pairs. The preprocessing stage rewrites the transcript, and the refinement stage filters the generated pairs. Switching each on and off gives four corpora from the same 40 Korean medical and finance sessions. On the corpus itself, quality rises as the stages switch on. Four LLM judges and six human domain experts rank the four corpora the same way, and the fully refined corpus leads by +0.18 and +0.22 on a 1-5 scale. We then ask whether that gain carries downstream. We supply each corpus to six models at inference and measure retrieval accuracy. The ranking does not hold. Enabling the transcript rewrite lowers retrieval accuracy by 0.061, the only stage effect that survives corre
    
[^67]: 说与知：平衡大语言模型的行为分布与知识分布

    Speak vs. Know: Balancing Behavior and Knowledge Distributions of Large Language Models

    [https://arxiv.org/abs/2610.27648](https://arxiv.org/abs/2610.27648)

    论文揭示了大语言模型的“知识分布”（文本表述分布）与“行为分布”（对数概率分布）存在显著不一致——在37%的世界价值观调查问题上两者最可能的答案不同，导致基于表述分布的评估分数无法真实反映模型的实际行为，因而需要平衡这两种分布。

    

    最近的研究报告指出，直接在输出文本中表述出来的分布与人类意见分布的对齐程度，远优于大语言模型的下一个词元对数概率分布。然而，表述分布反映的是LLM对某一群体意见的“认知”（知识分布），而模型在部署时实际输出的内容则由其对数概率分布（行为分布）决定。我们发现，当提示LLM预测特定群体的意见时，在三个LLM上取平均，对于37%的世界价值观调查问题，表述分布下最可能的答案与对数概率分布下的最可能答案并不相同。因此，LLM的知识与其行为不匹配。这对评估意义重大：一个表述分布可以在与某群体意见的对比中获得高分，而其最可能的答案却是模型从不输出的内容，因此该分数不再能描述模型本身。

    arXiv:2610.27648v1 Announce Type: new  Abstract: Recent work reports that the distribution verbalized directly in the output text aligns far better with human opinion distributions than the next-token log-probability distribution of LLMs. However, the verbalized distribution reflects what LLMs know about a group's opinions (knowledge distribution), while what they actually output at deployment is governed by their log-probability distribution (behavior distribution). We find that when prompting LLMs to predict the opinion of a specific group, the most likely answer under the verbalized distribution differs from that under the log-probability distribution for 37% of World Values Survey questions, averaged over three LLMs. Thus, LLMs' knowledge does not match their behavior. This matters for evaluation: a verbalized distribution can score well against a group while its most likely answer is one the model never outputs, so the score no longer describes the model. The alignment scores for 
    
[^68]: CUA-Recovery：面向长时程计算机使用智能体的错误恢复基准测试

    CUA-Recovery: Benchmarking Error Recovery in Long-Horizon Computer-Use Agents

    [https://arxiv.org/abs/2610.27645](https://arxiv.org/abs/2610.27645)

    该论文提出了 CUA-Recovery 基准和 ReRail 框架，通过在持久化桌面环境中将任务组合成长时程工作流来系统评估计算机使用智能体的错误检测与恢复能力，弥补了现有基准依赖人工注入错误、无法反映真实智能体策略所致失败的不足。

    

    随着计算机使用智能体（CUA）逐渐转向跨越多个应用程序的长时程工作流，未被检测到的错误可能会悄无声息地破坏持久化状态，并在后续动作中引发级联失效。可靠的错误检测与恢复已成为构建稳健 CUA 的关键瓶颈。然而，现有基准主要关注端到端任务性能，对于智能体能否在执行过程中检测错误并从中恢复的洞察十分有限。少数评估恢复能力的基准在很大程度上依赖人工注入的错误，这可能无法反映真实智能体策略所引发错误的类型或时程范围。我们提出了 CUA-Recovery，这是一个由覆盖多个桌面应用程序的测试集和训练集组成的基准。我们使用 ReRail 来构建该基准，ReRail 是一个在持久化桌面环境中合成长时程任务与轨迹的框架。ReRail 将 MyPCBench 任务组合成新的长时程工作流……（原文摘要在此处截断）

    arXiv:2610.27645v1 Announce Type: new  Abstract: As computer-use agents (CUAs) move to long-horizon workflows spanning multiple applications, undetected errors can silently corrupt persistent state and cause cascading failures in subsequent actions. Reliable error detection and recovery have become critical bottlenecks for building robust CUAs. However, existing benchmarks focus primarily on end-to-end task performance, providing limited insight into whether agents can detect and recover from errors during execution. The few benchmarks that evaluate recovery rely largely on manually injected errors, which may not reflect the types or horizons of failures induced by real agent policies. We introduce CUA-Recovery, a benchmark comprising test and training suites across multiple desktop applications. We construct it using ReRail, a framework for synthesizing long-horizon tasks and trajectories in a persistent desktop environment. ReRail composes MyPCBench tasks into new long-horizon workfl
    
[^69]: BCT：面向流语言模型的贝叶斯一致温度调节方法

    BCT: Bayes-Consistent Tempering for Flow Language Models

    [https://arxiv.org/abs/2610.27641](https://arxiv.org/abs/2610.27641)

    提出贝叶斯一致温度调节（BCT），将温度显式地融入流语言模型生成过程的分布与动力学中，解决了事后温度缩放导致的分布与动力学不匹配问题，使温度调节对流语言模型性能的提升更加可靠。

    

    温度缩放通常作为模型概率的事后变换来应用。在由后验诱导的流语言模型中，这种方式会造成两个不匹配。首先，后验温度调节会隐式地改变终点分布和中间噪声状态的分布，而去噪器却仍然是在原始状态上训练的。其次，经过温度调节的后验权重与为未经调节的流所导出的传输动力学相结合。这些不匹配在实证中有所体现：在流语言模型中，降低温度对GSM8K性能的提升远不如在离散扩散语言模型中可靠。我们提出了贝叶斯一致温度调节（BCT），它将温度显式地融入生成流的分布和动力学之中。其可学习的实现方式在经过温度调节的中间状态上训练去噪器，并使用相应的调节后动力学对终点分布进行温度调节。我们还推导了无需训练的变体（摘要在此处被截断）

    arXiv:2610.27641v1 Announce Type: cross  Abstract: Temperature scaling is typically applied as a post-hoc transformation of model probabilities. In posterior-induced flow language models, this creates two mismatches. First, posterior tempering implicitly changes both the endpoint distribution and the distribution of intermediate noisy states, while the denoiser remains trained on the original states. Second, the tempered posterior weights are combined with transport dynamics derived for the untempered flow. These mismatches are reflected empirically: lowering temperature improves GSM8K performance much less reliably in flow LMs than in discrete diffusion LMs. We introduce Bayes-Consistent Tempering (BCT), which makes temperature explicit in both the distributions and dynamics of the generative flow. Its learned realization trains the denoiser on tempered intermediate states and tempers the endpoint distribution using the corresponding tempered dynamics. We also derive training-free var
    
[^70]: 一条双向车道：文本与语音中语码混合与源语言相互作用的综述

    A Two-Way Street: Surveying the Interplay between Code-Mixing and Native Languages in Text and Speech

    [https://arxiv.org/abs/2610.27631](https://arxiv.org/abs/2610.27631)

    本文首次系统综述了文本与语音领域中语码混合与源语言之间的双向迁移，提出按迁移方向、路径和模态分类的框架，发现迁移主要从源语言流向语码混合场景，且在语音任务中源语言数据比新架构更有帮助。

    

    语码混合（Code-Mixed, CM）的文本和语音系统很少从零开始构建。由于CM数据稀缺，大多数工作都会借鉴被混合的语言，我们将其称为源语言。这种迁移也可以反向进行，即利用CM数据来改进源语言的模型。现有的综述都聚焦于CM本身的处理，没有一个将这两个方向结合起来。本文通过一项系统性综述填补了这一空白，该综述涵盖了文本和语音领域的双向迁移。我们按照迁移方向、路径和模态的分类体系组织了所调研的工作。通过这一分类体系，我们发现大多数迁移是从源语言流向CM场景的。反向迁移的研究较为新兴，其中语码转换主要通过语言对齐来发挥作用。文本和语音领域的研究共享方法，但很少相互引用。在一个共享的语音基准测试中，源语言数据比新架构更有帮助。这些发现指向了尚未解决的开放挑战（摘要在此处被截断）。

    arXiv:2610.27631v1 Announce Type: new  Abstract: Systems for code-mixed (CM) text and speech rarely start from scratch. CM data is scarce, so most work borrows from the languages being mixed, which we call native languages. This transfer can also run in reverse, when CM data is used to improve models for the native languages. Existing surveys focus on processing CM itself, and none brings the two directions together. This paper fills that gap with a systematic survey of transfer in both directions, across text and speech. We organise the surveyed work in a taxonomy of transfer direction, route and modality. Through this taxonomy, we find that most transfer runs from native languages into the CM setting. The reverse direction is younger, and in it code-switching mainly helps by aligning the two languages. Text and speech work share methods but rarely cite each other. On a shared speech benchmark, native data helps more than new architectures. These findings point to open challenges, suc
    
[^71]: PAR：用于状态跟踪长度泛化的进度对齐循环方法

    PAR: Progress-Aligned Recurrence for Length Generalization in State Tracking

    [https://arxiv.org/abs/2610.27628](https://arxiv.org/abs/2610.27628)

    提出进度对齐循环（PAR）训练方法，通过让第 k 次迭代的预测对齐前 k 步操作后的状态并将循环状态范数锚定到首次迭代，使 Qwen2.5-0.5B 在 swap-S5 状态跟踪任务上实现 16 倍于训练长度的外推泛化（512 步时准确率达 95.5%）。

    

    循环语言模型可以在测试时运行更多迭代，但额外的迭代并不一定能帮助它们解决更长的题目。对于序列状态跟踪任务，最终答案的监督信号并未规定每次迭代应当推进多远，而且反复更新可能改变循环状态的尺度。我们提出了进度对齐循环（PAR），该方法训练第 k 次迭代的预测去匹配前 k 次操作之后的状态，并将循环状态重新缩放到其第一次迭代的范数。在 swap-S5 任务上，一个仅在最多 32 步数据上训练的 Qwen2.5-0.5B 模型，在 512 步（即训练长度的 16 倍）时达到了 95.5% 的平均全状态准确率；相比之下，在同一骨干网络上，RLT 报告的准确率为 55.7%，显式思维链仅为 0.8%。受控的 S5 消融实验表明，移除范数锚定会将外推倍数从 16 倍降至 2.2 倍以下，而若用其他目标替换进度对齐目标（原文摘要在此处截断）。

    arXiv:2610.27628v1 Announce Type: new  Abstract: Looped language models can run more iterations at test time, but additional iterations need not help them solve longer problems. For sequential state tracking, final-answer supervision does not specify how far each iteration should advance, and repeated updates can change the scale of the recurrent state. We introduce \textbf{Progress-Aligned Recurrence (PAR)}, which trains the prediction at iteration $k$ to match the state after the first $k$ operations and rescales recurrent states to their first-iteration norms. On swap-$S_5$, a Qwen2.5-0.5B model trained on at most 32 moves achieves 95.5\% mean full-state accuracy at 512 moves, or $16\times$ the training length, compared with 55.7\% reported for RLT and 0.8\% for explicit chain of thought on the same backbone. Controlled $S_5$ ablations show that removing norm anchoring reduces the extrapolation factor from $16\times$ to below $2.2\times$, and replacing progress-aligned targets with 
    
[^72]: ABench：基于分类价值观对大语言模型进行对抗性评测

    ABench: Adversarily Benchmarking Large Language Models across Taxonomized Values

    [https://arxiv.org/abs/2610.27611](https://arxiv.org/abs/2610.27611)

    提出了ABench——一个基于158个中文价值观子类别的层次化分类体系、包含2,895个开放式问题的细粒度人工评测基准，通过角色扮演和逻辑陷阱等对抗性诱导技术及加权自动化评估框架，系统评估大语言模型在中文文化语境下面对对抗性诱导时价值观立场的稳定性。

    

    确保大语言模型（LLM）与多元文化价值观保持对齐，对于其在全球范围内的安全部署至关重要。然而，现有的对齐评测基准主要以英语为中心，或仅局限于简单的判别式任务。此外，这些基准往往无法有效评估模型在特定文化语境下抵御复杂对抗性诱导的鲁棒性。为了解决这些局限，我们提出了ABench——一个专门针对中文分类价值观构建的细粒度、人工制作的评测基准。ABench建立在包含158个衍生价值子类别的严谨层次化分类体系之上，由2,895个开放式问题组成。这些问题运用角色扮演和逻辑陷阱等技术手段，探测模型在对抗性诱导情境下立场的稳定性。此外，我们提出了一个加权自动化评估框架，利用针对特定问题的评分标准实现细致入微的评估……

    arXiv:2610.27611v1 Announce Type: cross  Abstract: Ensuring the alignment of Large Language Models (LLMs) with diverse cultural values is critical for their safe global deployment. However, existing alignment benchmarks are predominantly English-centric or confined to simple discriminative tasks. Moreover, they often fail to effectively evaluate model robustness against complex, adversarial inductions within specific cultural contexts. To address these limitations, we propose ABench, a fine-grained, handcrafted benchmark specifically tailored to taxonomized values in Chinese. Grounded in a rigorous hierarchical taxonomy comprising 158 derived value subcategories, ABench consists of 2,895 open-ended questions. These questions employ techniques such as role-playing and logical traps to probe the stability of models' stances in adversarially induced contexts. Furthermore, we propose a weighted automated evaluation framework that utilizes question-specific scoring criteria for nuanced asse
    
[^73]: 审计产品而非递归过程：递归自我改进声明背后的评估证据

    Auditing the Product, Not the Recursion: The Evaluation Evidence Behind Recursive Self-Improvement Claims

    [https://arxiv.org/abs/2610.27610](https://arxiv.org/abs/2610.27610)

    该论文首次通过对114项递归自我改进（RSI）研究的系统性全文编码，量化审计了RSI声明背后产品评估与流程评估证据的实际报告频率，揭示了递归自我改进声明与其实际评估证据之间的差距。

    

    递归自我改进（RSI）承诺了诸如自我进化的LLM智能体等AI系统能够在产生进一步改进方面不断提升。评估这一承诺需要将关于最终系统的“产品”声明与关于改进过程本身的“流程”声明区分开来。现有综述已明确了这些声明所需的证据，但研究实际报告这些证据的频率仍未被测量。我们通过对2026年10月冻结的可复现arXiv搜索框架中114项符合条件的研究进行全文编码来考察这一问题，其中包括90项将自身系统描述为RSI的研究。该编码手册将声明与评估设计相关联。通过独立的人工编码、第三方读者的裁定，以及对改进器比较的单独盲编码来评估编码的可靠性。在90项RSI研究中，90%在包含部分分离或未明确说明分离的宽泛标准下报告了超越开发任务的产品评估，而60%报告了评估……（原文摘要在此处截断）

    arXiv:2610.27610v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) promises AI systems, such as self-evolving LLM agents, that become better at producing further improvements. Evaluating this promise requires separating product claims about the final system from process claims about the improvement process itself. Existing surveys specify the evidence these claims require, but how often studies report that evidence remains unmeasured. We examine this question through full-text coding of 114 eligible studies in a reproducible arXiv search frame frozen in October 2026, including 90 that describe their own systems as RSI. The codebook links claims to evaluation designs. Independent human coding, third-reader adjudication, and a separate blind coding of improver comparisons assess coding reliability. Among the 90 RSI studies, 90% report product evaluation beyond development tasks under a broad criterion that includes partial or unstated separation, while 60% report evaluat
    
[^74]: 面向代码生成的自分解扩散模型

    Self-Decomposing Diffusion for Code Generation

    [https://arxiv.org/abs/2610.27600](https://arxiv.org/abs/2610.27600)

    提出了SDDC框架，完全通过扩散先生成规定程序分解方式与单元间依赖关系的契约，再据此指导协调的并行实现，从而解决并行解码效率与程序依赖正确性之间的权衡问题。

    

    扩散语言模型正成为代码生成的一种有前景的范式，具备快速并行解码的潜力。然而，生成复杂程序仍然具有挑战性，因为程序中相互依赖的组件必须保持一致的接口、数据流和共享假设。激进的并行解码可能破坏这些依赖关系并损害程序的正确性，而保守的解码则牺牲了效率。为解决这一权衡问题，我们提出在实现之前先规划程序分解，以便在并行生成过程中协调相互依赖的单元。我们提出了面向代码的自分解扩散（SDDC），这是一个新颖的框架，完全通过扩散来分解复杂程序并并行实现其各个单元。SDDC首先生成一个契约，明确规定程序的分解方式及单元间的依赖关系，然后利用该契约指导协调一致的并行实现。

    arXiv:2610.27600v1 Announce Type: new  Abstract: Diffusion language models (DLMs) are emerging as a promising paradigm for code generation, offering the potential for fast, parallel decoding. However, generating complex programs remains challenging, as their interdependent components must maintain consistent interfaces, data flows, and shared assumptions. Aggressive parallel decoding can compromise these dependencies and undermine program correctness, while conservative decoding sacrifices efficiency. To address this trade-off, we propose planning program decomposition before implementation to coordinate interdependent units during parallel generation. We introduce Self-Decomposing Diffusion for Code (SDDC), a novel framework that decomposes complex programs and implements their units in parallel, entirely through diffusion. SDDC first generates a contract specifying the program's decomposition and inter-unit dependencies, then uses it to guide coordinated parallel implementation. Acro
    
[^75]: WORTH基准：非完全合作信息下的证据发现——两集试点研究

    WORTH Benchmark: Evidence Discovery under Not-Fully-Cooperative Information -- A Two-Episode Pilot

    [https://arxiv.org/abs/2610.27597](https://arxiv.org/abs/2610.27597)

    该论文提出WORTH基准，让模型在无线索标注的原创推理小说中自主发现关键证据、识别案件参与者并解释推理过程，以此评估模型在叙事信息不完全合作（证据与误导性安排并存）环境下的证据发现与判断能力。

    

    现有的推理评估通常将任务相关信息作为可用前提提供给模型，很少同时考察模型如何自主发现证据，以及叙事中的信息呈现方式如何引导或遮蔽判断。本文提出WORTH基准：模型阅读没有线索标注的原创推理小说，自主选择关键线索句子、识别案件参与者，并解释其推理过程。叙事材料由专业作者设计，使促进推理的证据与延迟真相揭示的叙事安排并存。我们报告了在全篇故事条件下、跨越两集的试点实验，对模型共进行了70次正式运行，其中67次运行可进行内容评分。评分分别记录对作者指定关键线索句的命中情况和参与者选择情况；自然语言解释被用（原文在此处截断）……

    arXiv:2610.27597v1 Announce Type: cross  Abstract: Existing reasoning evaluations typically provide task-relevant information to models as available premises, rarely examining concurrently how models discover evidence autonomously and how information presentation within narratives guides or obscures judgment. This paper proposes the WORTH Benchmark: models read original mystery narratives without clue annotations, autonomously select key clue sentences, identify case participants, and explain their reasoning process. The narrative materials are designed by professional authors such that evidence facilitating reasoning coexists with narrative arrangements that delay the revelation of truth. We report a pilot experiment under full-story conditions across two episodes, conducting a total of 70 formal runs on models, of which 67 runs are content-scorable. Scoring separately records hits on author-designated key clue sentences and participant selections; natural language explanations are us
    
[^76]: 重复数据下随机词元替换的收益与代价：基于已知熵的损失分解及来自十个真实语料库的证据

    Benefit and Cost of Random-Token Replacement under Repeated Data: A Known-Entropy Decomposition and Evidence from Ten Real Corpora

    [https://arxiv.org/abs/2610.27590](https://arxiv.org/abs/2610.27590)

    该论文提出将验证损失分解为近似、估计与记忆三项的框架，在熵率已知的受控数据源上证明随机词元替换通过向语料均值收缩预测来缓解记忆化（可将验证损失最小值后的回升降低约60%），其收益依赖记忆化的存在而偏差代价则不然，并在十个真实语料库上提供了实证证据。

    

    当独一无二的预训练数据稀缺时，模型会反复见到相同的词元，因而被认为产生了记忆化。随机词元替换是一种输入污染式数据增强手段，据报导可降低验证损失，但仅凭验证损失的下降并不能揭示它究竟减少了哪一类误差来源。在一个熵率具有闭式表达式的受控数据源上，我们证明替换会使预测向语料库平均值收缩。我们还引入了一种分解方法，将验证损失拆分为近似项、估计项与记忆项。这一收缩视角预言：替换的偏差代价无需记忆化即可出现，而其收益则依赖记忆化。由于记忆化会使验证损失在达到最小值后再度上升，我们在两个时点报告替换的效果：验证损失最小值处以及固定训练预算结束时。在替换有帮助的情形下，它将该回升幅度降低约60%。在预算结束时，随着受控数据源b……（原文摘要至此被截断）

    arXiv:2610.27590v1 Announce Type: cross  Abstract: When unique pretraining data is scarce, models see the same tokens many times and are thought to memorize. Random-token replacement, an input-corruption augmentation, is one remedy reported to lower validation loss, but a lower loss alone does not reveal which source of error it reduces. On a controlled source whose entropy rate is known in closed form, we show that replacement shrinks predictions toward the corpus average. We also introduce a decomposition that separates validation loss into approximation, estimation, and memorization terms. This shrinkage view predicts that replacement's bias cost needs no memorization but its benefit does. Memorization raises validation loss after its minimum, so we report the effect at two points, the minimum validation loss and the end of a fixed budget. Where replacement helps, it reduces this rise by about 60%. At the end of the budget, it turns from harmful to helpful as the controlled source b
    
[^77]: 超越模型权重：扩散语言模型中依赖采样的公平性

    Beyond Model Weights: Sampling-Dependent Fairness in Diffusion Language Models

    [https://arxiv.org/abs/2610.27584](https://arxiv.org/abs/2610.27584)

    扩散语言模型的社会偏见程度及模型间的偏见排序会随采样配置（如块长度和重掩码策略）显著变化，且即使采用去偏见策略后，这种采样依赖性依然持续存在。

    

    扩散语言模型（DLM）正成为自回归语言模型（ARLM）的实用替代方案，但其公平性在很大程度上尚未得到系统表征。扩散语言模型引入了在自回归模型中没有直接对应物的采样控制参数，例如块长度和重掩码策略，这些参数直接决定了模型的速度-质量工作点。这引出一个基本问题：扩散语言模型所测得的社会偏见在这些不同设置下是否保持稳定？通过在 Open-BBQ 数据集上对 LLaDA、LLaDA-1.5 和 Dream 三个模型进行测试，我们发现这三个模型均表现出社会偏见，但所测偏见的幅度以及按偏见程度对模型进行的相对排序，都对采样配置高度敏感。这种采样依赖性也改变了“成功去偏见”的含义。虽然为自回归语言模型建立的偏见缓解策略可以有效地迁移到扩散语言模型，并在不损害下游性能的前提下减少偏见，但即使在缓解之后，采样依赖性依然存在。

    arXiv:2610.27584v1 Announce Type: new  Abstract: Diffusion language models (DLMs) are emerging as a practical alternative to autoregressive language models (ARLMs), yet their fairness remains largely uncharacterized. DLMs introduce sampling controls with no direct autoregressive counterpart, such as block length and remasking strategy, that directly set their speed--quality operating point. This raises a basic question: does a DLM's measured social bias remain stable across these settings? Across LLaDA, LLaDA-1.5, and Dream on Open-BBQ, we find that all three models exhibit social bias, but both the magnitude of measured bias and the relative ordering of models by bias are highly sensitive to the sampling configuration. This sampling dependence also changes what successful debiasing means. While mitigation strategies established for ARLMs transfer effectively to DLMs and reduce bias without degrading downstream performance, sampling dependence persists even after mitigation. Our result
    
[^78]: TeCoP：通过暂定上下文流水线加速扩散大语言模型解码

    TeCoP: Accelerating Diffusion Large Language Model Decoding via Tentative Context Pipelining

    [https://arxiv.org/abs/2610.27583](https://arxiv.org/abs/2610.27583)

    TeCoP是一种无需训练的扩散大语言模型解码方法，通过将稳定但未提交的预测作为暂定上下文进行跨块流水线复用，将前向评估次数最多减少87%，吞吐量最高提升11.77倍。

    

    扩散大语言模型每次前向传播可预测多个词元位置，然而分块解码器将生成过程串行化，并反复重新计算已经稳定的预测。我们将这种低效归因于将预测的“不可逆提交就绪度”与“其作为上下文的效用”这两个角色耦合在了一起。我们提出了暂定上下文流水线，这是一种无需训练的解码器，通过错峰分块并行流水线和草稿-验证机制将这两种角色解耦。TeCoP将稳定但尚未提交的预测作为临时的跨块上下文复用，预测的提交与否由置信度和稳定性检查决定。在四个LLaDA和Dream扩散大语言模型上，针对GSM8K、Minerva MATH和MBPP基准测试，TeCoP将前向评估次数最多减少87%，在无KV缓存情况下相比未缓存的原始解码器实现最高7.50倍的吞吐量提升，在有缓存情况下达到11.77倍。在全部12个模型-基准组合设置中，其吞吐量均优于Fast-dLLM和DSB。

    arXiv:2610.27583v1 Announce Type: new  Abstract: Diffusion large language models (dLLMs) predict multiple token positions per forward pass, yet block-wise decoders serialize generation and repeatedly recompute stable predictions. We attribute this inefficiency to coupling a prediction's readiness for irreversible commitment with its utility as context. We introduce Tentative Context Pipelining (TeCoP), a training-free decoder that decouples these roles through staggered block-parallel pipelining and draft-and-verify. TeCoP reuses stable but uncommitted predictions as provisional cross-block context, with commitment governed by confidence and stability checks. Across four LLaDA and Dream dLLMs on GSM8K, Minerva MATH, and MBPP, TeCoP reduces number of forward evaluations by up to 87% and achieves up to 7.50-fold throughput without KV caching and 11.77-fold with caching over the uncached original decoder. It outperforms Fast-dLLM and DSB in throughput across all 12 model-benchmark setting
    
[^79]: 扩散语言模型中的事实访问状态：从机制到知识编辑与检索

    Factual Access States in Diffusion Language Models: From Mechanism to Knowledge Editing and Retrieval

    [https://arxiv.org/abs/2610.27580](https://arxiv.org/abs/2610.27580)

    该论文揭示了扩散语言模型在第一层注意力层就通过将关系措辞路由到可见实体位置来组装事实访问状态，这一机制与自回归模型的组装方式截然不同且发生更早，为知识编辑与检索提供了新的机制层面依据。

    

    事实回忆本质上是一种查找：网络内部的某种表示充当键，而该键的构成决定了哪些查询能够检索到该事实、一次编辑必须覆盖哪些内容、以及检索器能够读取到什么。我们将其称为“事实访问状态”。在自回归（AR）模型中，主语位置对属性的丰富过程在很大程度上独立于关系，而针对关系的选择发生在较晚的读出位置。在掩码扩散语言模型（DLMs）中，我们发现该状态的组装方式不同，且时间上早得多：在第一层注意力层中，关系的措辞被路由到可见实体的位置上，随后早期的前馈层基于这一融合表示查找事实。因此，该状态会随关系的表面形式和查询方向而变化，这意味着关于同一事实的查询占据的是状态空间中的一个区域而非一个点，而对事实的操作能否成功则取决于（摘要在此处截断，未完）

    arXiv:2610.27580v1 Announce Type: cross  Abstract: Factual recall is a lookup: some representation inside the network serves as the key, and what that key is made of decides which queries can retrieve the fact, what an edit must cover, and what a retriever can read. We call it the factual access state. In autoregressive (AR) models the subject position enriches attributes largely independently of the relation, and relation-specific selection happens later, at the readout position. In masked diffusion language models (DLMs) we find the state is assembled differently, and much earlier: in the first attention layer the wording of the relation is routed into the position of whichever entity is visible and the early feed-forward layers look the fact up on this fused representation. The state therefore varies with the relation's surface form and with the query direction, so the queries about one fact occupy a region of state space rather than a point, and an operation on the fact succeeds on
    
[^80]: 为什么静态探针会失效：上下文决定欺骗特征的组成

    Why Static Probes Fail: Context Determines Deception Feature Composition

    [https://arxiv.org/abs/2610.27577](https://arxiv.org/abs/2610.27577)

    该论文发现欺骗并非大语言模型表征空间中的单一方向，提出用迭代正交分解将欺骗建模为多维子空间，并揭示欺骗特征的组成依赖于上下文，因此静态的探针组合难以可靠泛化。

    

    随着自主性不断增强，大型语言模型（LLMs）可能会表现出欺骗性行为，例如为实现未对齐目标而密谋（scheming），或为逃避监督而故意示弱（sandbagging）。仅监控模型输出是不够的，因为模型可能在内部持有未对齐目标的同时产生看似无害的响应。虽然线性探针等机械可解释性方法颇具前景，但它们依赖于一个假设：欺骗对应于表征空间中的单一方向。我们证明，这类单方向探针无法在多样化的欺骗场景中泛化。为弥合这一差距，我们提出将欺骗表示为多维子空间而非单一方向。我们的方法利用迭代正交分解来识别互补方向，这些方向捕捉了欺骗意图的不同侧面。为了有效地利用这些方向进行检测，我们发现静态组合方式会变得脆弱……（原文摘要在此处截断）

    arXiv:2610.27577v1 Announce Type: new  Abstract: With increasing autonomy, Large Language Models (LLMs) may engage in deceptive behaviors, such as scheming to pursue misaligned goals or sandbagging to escape oversight. Monitoring outputs alone is insufficient, as models can produce seemingly benign responses while harboring misaligned internal objectives. While mechanistic approaches like linear probing are promising, they rely on the assumption that deception corresponds to a single direction in representation space. We demonstrate that such single-direction probes fail to generalize across diverse deception scenarios. To bridge this gap, we propose representing deception as a multi-dimensional subspace rather than a single direction. Our method utilizes iterative orthogonal decomposition to identify complementary directions that capture distinct facets of deceptive intent. To effectively leverage these directions for detection, we find that static combinations prove brittle due to st
    
[^81]: 当大语言模型进行操纵时：测量行为、社会与时间效应

    When LLMs Manipulate: Measuring Behavioral, Social, and Temporal Effects

    [https://arxiv.org/abs/2610.27573](https://arxiv.org/abs/2610.27573)

    该论文提出了一个测量“行为脆弱性”的行为计算框架，通过操纵梯度下降生成针对权威、紧迫性和社会认同偏差的操纵性AI刺激，并借助多利益相关者影响评估网络与时间性危害传播模型，量化了操纵性LLM输出对用户信任校准与依从性的显著影响及其跨用户、跨时间的传播效应。

    

    大语言模型（LLM）可以通过微妙的语言线索影响用户的信任与决策，然而操纵性输出的行为效应至今仍缺乏充分的量化研究。我们引入了一个行为计算框架，用于测量“行为脆弱性”，即用户对操纵性AI响应的易感程度。基于DarkPatterns-LLM基准，我们利用操纵梯度下降法（MGD）生成受控的操纵性刺激，针对与权威、紧迫性和社会认同相关的认知偏差。我们采用多利益相关者影响评估网络（MSIAN）分析参与者响应以建模跨用户影响，并使用时间性危害传播模型（THPM）刻画行为效应随时间的持续性。在一项涉及120名参与者的研究中，操纵性表述对信任校准和依从性产生了显著影响（p < 0.01），且效应还会在（摘要在此处被截断）……

    arXiv:2610.27573v1 Announce Type: new  Abstract: Large language models (LLMs) can influence user trust and decision-making through subtle linguistic cues, yet the behavioral effects of manipulative outputs remain poorly quantified. We introduce a behavioral computing framework for measuring \textit{behavioral vulnerability}, defined as users' susceptibility to manipulative AI responses. Building on DarkPatterns-LLM benchmark, we generate controlled manipulative stimuli using Manipulation Gradient Descent (MGD), targeting biases associated with authority, urgency, and social proof. We analyze participant responses using the Multi-Stakeholder Impact Assessment Network (MSIAN) to model cross-user influence and the Temporal Harm Propagation Model (THPM) to characterize the persistence of behavioral effects over time. In a study involving 120 participants, manipulative framing produced significant changes in trust calibration and compliance ($p < 0.01$), with effects also propagating across
    
[^82]: AudiPsy：基于证据的多模态推理方法，用于口语心理咨询对话中的抑郁症评估

    AudiPsy: Evidence-Grounded Multimodal Reasoning over Spoken Counseling Conversations for Depression Assessment

    [https://arxiv.org/abs/2610.27567](https://arxiv.org/abs/2610.27567)

    提出AudiPsy，一个基于证据的全模态大语言模型，通过观察-评估-总结范式对口语心理咨询对话中的文本与声学多模态证据进行结构化情境推理，显著提升了抑郁症评估的可解释性与准确性。

    

    从心理咨询语音中进行抑郁症评估需要跨轮次整合文本与声学两方面的心理学证据，然而现有的端到端分类方法缺乏显式的多模态证据解释和情境感知推理能力。我们提出了AudiPsy，一个基于证据的全模态大语言模型，用于口语心理咨询对话中的抑郁症评估。AudiPsy采用观察-评估-总结（Observation-Assessment-Summary, OAS）范式，将对话转录文本、文本情绪与声学情绪以及文本-声学情绪不一致性整合到结构化推理中，逐步组织多模态观察结果、进行情境化评估，并为后续对话轮次维护一个持续更新的对话总结。为支持该框架，我们构建了AudiPsy-Counsel——一个具有多模态上下文增强的合成口语心理咨询对话语料库，并开发了一套面向抑郁症推理的统一三阶段课程训练方法。实验...

    arXiv:2610.27567v1 Announce Type: cross  Abstract: Depression assessment from counseling speech requires integrating textual and acoustic psychological evidence across turns, yet existing end-to-end classification approaches lack explicit multimodal evidence interpretation and context-aware reasoning. We propose AudiPsy, an evidence-grounded omni-LLM for depression assessment over spoken counseling conversations. AudiPsy employs an Observation-Assessment-Summary (OAS) paradigm to integrate dialogue transcripts, textual and acoustic emotions, and textual-acoustic emotion incongruence into structured reasoning. OAS incrementally organizes multimodal observations, performs contextualized assessment, and maintains an evolving dialogue summary for subsequent turns. To support this framework, we construct AudiPsy-Counsel, a synthetic spoken counseling conversation corpus with multimodal context augmentation, and develop a unified three-stage curriculum for depression-oriented reasoning. Expe
    
[^83]: 学习式读取令牌用于提取大语言模型置信度

    Learned Readout Tokens for Extracting Large Language Model Confidence

    [https://arxiv.org/abs/2610.27566](https://arxiv.org/abs/2610.27566)

    提出 ScratchProbe 方法，通过学习一个专用的读取令牌来提取隐藏状态训练置信度探针，在多种数据集和大语言模型上优于基于现有令牌的探针，多令牌扩展还可进一步提升性能。

    

    大语言模型正日益广泛地应用于日常场景中，其响应会对下游任务产生重大影响。为了负责任地部署此类系统，我们需要了解能在多大程度上信任模型的各个输出。估计大语言模型置信度的一种流行方法是在模型令牌的隐藏状态上训练探针，以预测模型输出的正确性。这些探针通常从模型词表中已有的令牌读取隐藏状态。在本工作中，我们提出了一种新的置信度探针方法，该方法学习一个新的读取令牌，并利用其隐藏状态来训练探针。我们证明，我们的学习式读取方法 ScratchProbe 在多种数据集和大语言模型上均优于使用现有令牌的方法。我们还将该方法扩展为多令牌方法，并证明其能带来进一步的性能提升。最后，由于学习式读取仅改变……

    arXiv:2610.27566v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used within everyday applications where their responses have significant downstream impacts. For responsible deployment of such systems, we desire insight into how much we can trust their individual outputs. A popular approach for estimating LLM confidence is to train probes on the hidden states of the LLM's tokens to predict the correctness of the model's output. These probes typically read hidden states from tokens that are already in the model's vocabulary. In this work, we introduce a new approach to confidence probes that learn a new readout token and utilize its hidden states to train the probe. We show that our learned readout approach ScratchProbe outperforms its existing-token counterpart on a variety of datasets and LLMs. We also extend our method to a multi-token approach which we show provides further performance improvements. Finally, since the learned readout changes only the 
    
[^84]: AnomalyJev：基于断言级验证的图像异常解释评估方法

    AnomalyJev: Towards Evaluating Image Anomaly Explanations via Claim-Level Verification

    [https://arxiv.org/abs/2610.27565](https://arxiv.org/abs/2610.27565)

    提出AnomalyJev，一个无需token生成的轻量级评估框架，通过从图像异常检测模型生成的文本解释中提取可验证断言并与真实标注比对验证，从而高效且可靠地评估解释的忠实度与完整性。

    

    近期基于多模态大语言模型（MLLM）的异常检测（AD）方法能够生成有助于解释其预测结果的文本说明。然而，如何可靠且高效地评估这些说明仍然是一个挑战。虽然使用大语言模型（LLM）作为评审员可以通过结构化评分标准来评估说明，但这种方式通常需要生成大量的token，导致评估成本高昂且难以扩展。为此，我们提出将评估锚定在已有的真实标注上，从每个说明中提取可验证的断言，并将其与对应的标注进行比对验证。我们将这一想法实现为AnomalyJev，这是一个轻量级框架，利用由真实标注定义的查询和基于指针的输出头，从上下文化表示中提取可验证的断言，而无需生成token。随后将这些断言与真实标注进行验证，以评分解释的忠实度和完整性。为了进行评估，我们构建了一个包含640条人工标注说明的数据集……（原文摘要至此被截断）

    arXiv:2610.27565v1 Announce Type: cross  Abstract: Recent multimodal LLM (MLLM)-based anomaly detection (AD) methods can generate textual explanations that help interpret their predictions. However, reliably and efficiently evaluating these explanations remains challenging. While LLM judges can assess explanations through structured rubrics, they often require substantial token generation, making evaluation costly and difficult to scale. We instead propose to anchor evaluation in the available ground truth by extracting verifiable claims from each explanation and verifying them against the corresponding annotations. We instantiate this idea as AnomalyJev, a lightweight framework that extracts verifiable claims from contextualised representations using ground-truth-defined queries and pointer based heads, without token generation. The claims are then verified against the ground truth to score faithfulness and completeness. For evaluation, we construct a human-annotated set of 640 explan
    
[^85]: 探索循环深度语言模型用于多任务临床文本处理

    Exploring Recurrent-Depth Language Models for Multitask Clinical Text Processing

    [https://arxiv.org/abs/2610.27559](https://arxiv.org/abs/2610.27559)

    小型循环深度语言模型（如Ouro系列）通过多任务联合微调，在13个临床NLP数据集上以更小的参数量达到了与更大规模标准Transformer相当甚至更优的性能，尤其在推理和问答任务上优势显著。

    

    临床自然语言处理包括信息抽取、分类和问答等多种任务。开发一个适用于所有任务的模型可以简化部署并促进知识迁移。我们评估了小型循环深度语言模型能否通过联合微调满足这一需求，并将其性能与标准Transformer模型进行比较。我们在涵盖信息抽取、分类、推理和问答的9个数据集上对Ouro-1.4B-Thinking、Ouro-2.6B-Thinking、HRM-Text-1B、Qwen3-1.7B和Qwen3-8B进行了微调，并在4个未参与训练的额外数据集上进行评估。在这13个数据集中，Ouro-1.4B-Thinking在11个数据集上的表现优于Qwen3-1.7B；Ouro-2.6B-Thinking在大多数任务上与Qwen3-8B相当。Ouro模型在推理和问答任务上取得了最大提升，而HRM-Text-1B的表现则好坏参半。这些发现支持……

    arXiv:2610.27559v1 Announce Type: new  Abstract: Clinical natural language processing includes diverse tasks such as information extraction, classification, and question answering. Developing one model for all tasks could simplify deployment and improve knowledge transfer. We evaluated whether small recurrent-depth language models can meet this need through joint fine-tuning and compared their performance with standard transformers. We fine-tuned Ouro1.4B-Thinking, Ouro-2.6B-Thinking, HRM-Text-1B, Qwen3-1.7B, and Qwen3-8B on 9 datasets covering information extraction, classification, inference, and question answering, and evaluated them on four additional datasets excluded from training. Across these 13 datasets, Ouro-1.4B-Thinking achieved higher performances than Qwen3-1.7B on 11 datasets; Ouro-2.6BThinking was competitive with Qwen3-8B on most tasks. Ouro models showed the largest gains on inference and question answering, whereas HRM-Text-1B showed mixed results. These findings sup
    
[^86]: 原生安全性能否迁移？通过在线策略蒸馏实现跨模型的安全预训练复用

    Can Native Safety Travel? Reusing Safety Pretraining Across Models via On-Policy Distillation

    [https://arxiv.org/abs/2610.27555](https://arxiv.org/abs/2610.27555)

    本文首次提出“原生安全性复用”问题，并证明通过在线策略蒸馏，可以将安全预训练教师模型的鲁棒安全性迁移给标准预训练的学生模型，使其在保留通用能力的同时继承安全鲁棒性。

    

    安全预训练已成为一种有前景的范式，它能在预训练阶段将安全性深度融入模型之中，从而获得对后续微调的鲁棒性，而这种鲁棒性是传统安全后训练往往缺乏的。我们将这种在模型预训练期间习得的安全性称为“原生安全性”。然而，原生安全性获取成本高昂，仅局限于其被预训练的那个模型，并且常常伴随着较低的通用能力。在这项工作中，我们重新思考了安全预训练模型的实际价值，并首次提出了原生安全性复用问题：原生安全性能否从一个专用的、可能规模更小的安全预训练“教师”模型迁移到标准预训练的“学生”模型，使后者在继承其鲁棒性的同时保留自身的通用能力？我们证明了原生安全性可以通过在线策略蒸馏进行“迁移”：标准预训练的学生模型针对安全预训练提示生成回复，而……

    arXiv:2610.27555v1 Announce Type: cross  Abstract: Safety pretraining has emerged as a promising paradigm for embedding safety deeply into a model during pretraining, yielding robustness to subsequent fine-tuning that conventional safety post-training often lacks. We refer to this safety acquired during model pretraining as native safety. Yet native safety is expensive to acquire, confined to the model in which it was pretrained, and often accompanied by lower general utility. In this work, we rethink the practical value of safety-pretrained models and introduce, for the first time, the native safety reuse problem: Can native safety be transferred from a specialized, potentially smaller, safety-pretrained "teacher" to standard-pretrained "students," enabling them to inherit its robustness while preserving their general utility? We show that native safety can "travel" through on-policy distillation. A standard-pretrained student generates responses to safety-pretraining prompts, while a
    
[^87]: EviDial：当正确的抑郁预测缺乏证据时

    EviDial: When Correct Depression Predictions Lack Evidence

    [https://arxiv.org/abs/2610.27541](https://arxiv.org/abs/2610.27541)

    EviDial基准将“预测标签正确”与“有证据支撑的预测”区分开来，揭示82.8%的抑郁对话症状评估缺乏足够的文本证据支持。

    

    抑郁对话基准通常通过将预测的症状标签与临床量表得分进行比较来评估模型。然而，即使对话中缺乏足够的证据来支持某个预测，该预测仍可能与基准标签一致。我们提出了EviDial，这是一个区分标签一致性与基于证据评估的基准。EviDial包含来自三个来源的382个抑郁对话和2,744个参与者-症状单元，每个单元被分类为无证据、部分证据、冲突证据或充分证据。由于证据充分性缺乏直接可观察的黄金标准，我们使用概化理论、多侧面Rasch测量和随机效应潜在类别分析，从多个判断中推导出经过校准的证据参考。我们的审计显示，82.8%的参与者-症状单元缺乏支持相应症状评估的充分文本证据，且各来源之间存在显著差异。

    arXiv:2610.27541v1 Announce Type: new  Abstract: Depression dialogue benchmarks commonly assess models by comparing predicted symptom labels with clinical scale scores. However, a prediction may agree with a benchmark label even when the dialogue lacks enough evidence to justify it. We introduce EviDial, a benchmark that distinguishes label agreement from evidence-grounded assessment. EviDial comprises 382 depression dialogues from three sources and 2,744 participant-symptom cells, each classified as having absent, partial, conflicting, or sufficient evidence. Because evidence sufficiency lacks a directly observable gold standard, we derive calibrated evidence references from multiple judgments using generalizability theory, many-facet Rasch measurement, and random-effects latent class analysis. Our audit reveals that 82.8% of participant-symptom cells lack sufficient textual evidence for the corresponding symptom assessment, with substantial variation across sources. Across model fami
    
[^88]: 评判理由的正当性：面向生成式奖励模型的决策证书监督

    Judging the Justification: Decision Certificate Supervision for Generative Reward Models

    [https://arxiv.org/abs/2610.27528](https://arxiv.org/abs/2610.27528)

    提出DeCert框架，无需参考解释即可评估生成式奖励模型所生成理由的有效性与结论一致性，并将其转化为兼顾结论正确性和理由质量的偏好监督信号，以区分强弱的理由论证。

    

    生成式奖励模型通过生成自然语言理由来论证其偏好判断，然而正确的结论并不保证理由本身是可靠的。现有方法通常依赖参考解释来监督理由的生成，但获取这类参考代价高昂，且未必能确定生成的理由是否真正支持最终的结论。为解决这些局限，我们提出了DeCert，一个无需参考解释即可评估生成理由的有效性及其与结论一致性的框架。关键在于，DeCert将这些评估转化为同时考虑结论正确性与理由质量的偏好监督信号。特别地，当多个采样判断具有相同的正确结论时，证书质量的差异使DeCert能够构建理由偏好对，从而区分更强与更弱的理由论证。大量实验证明了……

    arXiv:2610.27528v1 Announce Type: new  Abstract: Generative reward models (GenRMs) generate natural language rationales to justify preference judgments, yet a correct verdict does not guarantee a sound justification. Existing approaches often rely on reference explanations to supervise rationales, but obtaining such references can be costly and does not necessarily establish whether generated rationales support the resulting verdict. To address these limitations, we propose DeCert, a framework that evaluates the validity and verdict consistency of generated rationales without requiring reference explanations. Crucially, DeCert translates these assessments into preference supervision that accounts for both verdict correctness and justification quality. In particular, when multiple sampled judgments share the same correct verdict, differences in certificate quality enable DeCert to construct rationale preference pairs that distinguish stronger from weaker justifications. Extensive empiri
    
[^89]: 通过特权回溯蒸馏实现自我进化的智能体

    Self-Evolving Agents through Privileged Retrospective Distillation

    [https://arxiv.org/abs/2610.27495](https://arxiv.org/abs/2610.27495)

    提出回溯自蒸馏（RSD）多轮框架，让同一模型既当学生又当教师，借助特权回溯上下文将缺乏真值答案或奖励的自我经验转化为密集监督信号，实现模型、数据与监督的共同进化。

    

    自我进化的智能体旨在从自身经验中持续改进，而非在部署后保持固定不变，但自然收集的轨迹往往缺乏真实标准答案或由验证器导出的奖励信号。我们提出了回溯自蒸馏（Retrospective Self-Distillation, RSD），这是一个多轮框架，利用特权回溯上下文将此类经验转化为密集的监督信号。对于每条同策略轨迹，同一个底层模型同时充当学生和教师：教师在以跨任务先例与经验教训、以及来自已完成轨迹的轮次级反思和未来后见之明为条件的情况下，对学生实际执行的动作进行重新评分。由此得到的教师分布通过同策略自蒸馏被蒸馏回策略之中。在多轮迭代中，经验库不断增长，回溯上下文不断刷新，使模型、其数据及其监督信号得以共同进化。我们进一步引入了前沿引导……（摘要至此截断）

    arXiv:2610.27495v1 Announce Type: cross  Abstract: Self-evolving agents aim to improve from their own experience rather than remain fixed after deployment, but naturally collected trajectories often lack ground-truth solutions or verifier-derived rewards. We introduce Retrospective Self-Distillation (RSD), a multi-round framework that turns such experience into dense supervision using privileged retrospective context. For each on-policy trajectory, the same underlying model acts as both student and teacher: the teacher re-scores the student's realized actions while conditioning on cross-task precedents and lessons, together with turn-level reflection and future hindsight from the completed trajectory. The resulting teacher distribution is distilled back into the policy through on-policy self-distillation. Across rounds, the experience bank grows and the retrospective context is refreshed, allowing the model, its data, and its supervision to co-evolve. We further introduce frontier-guid
    
[^90]: 可恢复性与验证器强度：无人值守智能体自我改进的两个可测量前置条件

    Recoverability and Verifier Strength: Two Measured Preconditions for Unattended Agent Self-Improvement

    [https://arxiv.org/abs/2610.27465](https://arxiv.org/abs/2610.27465)

    论文提出了“验证器强度”和“可恢复性”这两个可测量的前置条件，用于判断智能体平台的哪些自我改进步骤可以安全地无人值守运行。

    

    智能体平台开始改进自身的智能体：它们读取已部署智能体的执行轨迹，推导出变更，让自动化验证器批准该变更，并在没有人类作者的情况下发布。这个循环中有多少可以安全地在无人参与的情况下运行？我们提出了平台可以测量的两个量。验证器强度 σ 衡量可用验证器（通常是 LLM 判官）与可信参考验证器的一致程度，它限制了一个步骤能在多大程度上改进智能体。可恢复性衡量撤销一个不良变更的代价有多低：撤销它是否需要新的验证器批准，以及回滚能恢复多少旧行为。只有当两者都充分时，一个步骤才应该无人值守运行。在我们的平台上，这一规则允许提示词、工具和技能编辑无人值守运行，同时使权重更新保持在人类签署批准之下。我们的主要贡献是这两个定义及其测量方法。我们构建了一个应用该规则的平台并进行了测试。

    arXiv:2610.27465v1 Announce Type: new  Abstract: Agent platforms are starting to improve their own agents: they read a deployed agent's execution traces, derive a change, have an automated verifier approve it, and release it with no human author. How much of this loop can safely run without a human? We propose two quantities a platform can measure. Verifier strength $\sigma$ is how well the available verifier -- often an LLM judge -- agrees with a trusted reference verifier; it limits how much a step can improve the agent. Recoverability is how cheaply a bad change can be undone: whether undoing it needs a new verifier approval, and how much of the old behavior a rollback restores. A step should run unattended only if both are adequate. On our platform this rule allows prompt, tool and skill edits to run unattended and keeps weight updates under human sign-off.   Our main contribution is these two definitions and how to measure them. We built a platform that applies the rule and tested
    
[^91]: 检测大语言模型回答中的错误：隐藏状态的增量预测价值

    Detecting Errors in Large Language Model Responses: The Incremental Predictive Value of Hidden States

    [https://arxiv.org/abs/2610.27455](https://arxiv.org/abs/2610.27455)

    该论文通过将输出侧信息整合为强基线并在公平条件下对比，发现隐藏状态对大语言模型错误检测的增量预测价值十分有限（AUROC仅提升约0.004至0.039）。

    

    尽管大语言模型能够支持广泛的任务，但幻觉问题仍是阻碍用户信任的主要障碍。当模型生成回答时，其内部处理过程和最终输出都会随问题而变化。人们通常期望内部状态能为幻觉检测提供更丰富的信号。然而，使用内部状态的研究通常只报告其单独的判别性能，而输出侧的对比往往采用建模能力较弱的简单基线。我们将推理时可获得的输出侧信息整合为一个综合基线，并在相同的数据划分和评估规则下，比较添加隐藏状态前后的检测性能。实验涵盖了不同模型上的开放式生成任务和受限的多项选择任务。结果显示，添加隐藏状态使两项任务的AUROC分别仅提升约0.004和0.039。因此，隐藏状态能够提供的是额外的预测价值。

    arXiv:2610.27455v1 Announce Type: new  Abstract: Although large language models support a wide range of tasks, hallucinations remain a major obstacle to user trust. When a model generates an answer, both its internal processing and final output vary with the question. Internal states are often expected to offer richer signals for hallucination detection. However, studies using these states typically report their standalone discrimination performance, while output-side comparisons often use simple baselines with weaker modeling capacity. We integrate output-side information available at inference time into a combined baseline and compare detection performance before and after adding hidden states under identical data splits and evaluation rules. Experiments cover open-ended generation and constrained multiple-choice tasks across different models. Adding hidden states improves AUROC by approximately 0.004 and 0.039 on the two tasks, respectively. Hidden states thus provide additional pre
    
[^92]: 提问以蒸馏：学习合成数据用于同策略蒸馏

    Ask to Distill: Learning to Synthesize Data for On-Policy Distillation

    [https://arxiv.org/abs/2610.27450](https://arxiv.org/abs/2610.27450)

    该论文提出Ask-to-Distill框架，将合成问题视为分布引子，利用基于师生分歧等信号的分布性替代奖励（无需参考答案或正确答案）来评估并优化合成问题的价值，从而实现高效的同策略蒸馏数据合成。

    

    同策略蒸馏（OPD）仅用极少的训练问题就能取得成功，这促使我们将问题视为分布性知识迁移的载体。那么，我们如何评估一个新生成的问题对蒸馏是否有价值，并在无法获得其正确答案的情况下学习生成更好的问题？我们提出了 Ask-to-Distill 框架，该框架将合成问题视为分布引子，并优化这些问题在学生模型生成的轨迹上所引出的教师监督。一个以学生模型初始化的合成器，通过问题价值的分布性替代指标进行优化，该指标由冻结的学生-教师模型对计算得出，无需参考答案、正确性奖励，也无需针对每个候选问题更新学生模型。该奖励将师生分歧度与教师在学生候选支持集上的概率质量相结合，并辅以长度和多样性控制。所生成的问题经过过滤后……（摘要截断）

    arXiv:2610.27450v1 Announce Type: new  Abstract: On-policy distillation (OPD) can succeed with surprisingly few training questions, motivating a view of questions as carriers of distributional knowledge transfer. How can we assess whether a newly generated question is valuable for distillation, and learn to generate better questions---without access to its correct answer? We introduce Ask-to-Distill (\method), a framework that treats synthetic questions as distribution elicitors and optimizes the teacher supervision they expose along student-generated trajectories. A student-initialized synthesizer is optimized using a distributional surrogate for question value, computed from a frozen student--teacher pair without reference answers, correctness rewards, or per-candidate student updates. The reward combines teacher--student disagreement with teacher probability mass on the student's candidate support, together with length and diversity controls. The generated questions are filtered and
    
[^93]: 面向长时程终端智能体的Harness自演化框架

    Harness Self-Evolution for Long-Horizon Terminal Agents

    [https://arxiv.org/abs/2610.27435](https://arxiv.org/abs/2610.27435)

    RiseHarness提出了一种在模型权重固定的前提下，通过对比同一任务的成功与失败运行、结合智能体轨迹与运行时监控证据、并借鉴其他harness框架的实现来自演化迭代改进智能体harness的框架，从而弥合能力与可靠执行之间的差距。

    

    长时程终端智能体可能一次运行就能解决某个任务，但在多次重复运行中却可能失败。随机性错误会通过动作和上下文不断传播，造成模型能力与可靠执行之间的差距。我们提出了RiseHarness，一个在模型权重保持固定的情况下迭代改进智能体harness的框架。它专注于那些在同一配置下既有成功运行又有失败运行的任务，并将成功运行作为失败分析的参考。RiseHarness将智能体轨迹与来自运行时监控器的内部harness追踪记录相结合，以考察模型动作如何与命令执行、反馈、上下文管理和完成决策相互影响。一份Harness手册将这些证据与harness的行为和源代码关联起来，帮助识别跨任务共享机制中反复出现的问题。随后，RiseHarness会搜索其他harness框架中的相关实现，并将其改编为当前harness的设计参考。

    arXiv:2610.27435v1 Announce Type: new  Abstract: Long-horizon terminal agents may solve a task in one run yet fail across repeated runs. Stochastic errors can propagate through actions and context, creating a gap between capability and reliable execution. We present RiseHarness, a framework for iteratively improving agent harnesses with model weights fixed. It focuses on flipping tasks with both successful and failed runs under the same configuration, using successful runs as references for failure analysis. RiseHarness combines agent trajectories with internal harness traces from a runtime monitor to examine how model actions interact with command execution, feedback, context management, and completion decisions. A Harness Handbook links this evidence to harness behaviors and source code, helping identify recurring problems in mechanisms shared across tasks. RiseHarness then searches other harness frameworks for relevant implementations and adapts them as design references to the curr
    
[^94]: 路由器早已知晓

    The Router Already Knows

    [https://arxiv.org/abs/2610.27421](https://arxiv.org/abs/2610.27421)

    提出NR-SAE稀疏自编码器，直接复用MoE路由器对每个token的稀疏专家选择来构建字典，在解释路由器加权专家输出方差上显著优于参数匹配的对照组。

    

    混合专家模型对每个token已经做出了一个稀疏选择：由哪些专家来处理它。我们探讨稀疏字典是否应当复用这一选择。NR-SAE为每个被路由的专家分配一个小型字典，并只打开被冻结路由器选中的那些字典。在Gemma 4 26B-A4B的第15个块上，经过一千万个C4训练token并在三个随机种子下，专家对齐自编码器在固定评估捕获上解释了路由器加权专家输出方差的0.518。一个参数量相同但候选数量为其十六倍的秩寻址对照组达到0.481，候选数匹配的对照组为0.458，而架构匹配但专家身份随机化的对照组为0.387。转码器变体同样领先所有三个对照组：0.432，对比0.347、0.371和0.118。这一优势部分源于架构本身：每个专家只写入至多四分之一模型宽度的子空间，因此指名专家即等于指名该子空间……（原文摘要至此截断）

    arXiv:2610.27421v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) models already make a sparse choice for every token: which experts should process it. We ask whether a sparse dictionary should reuse that choice. NR-SAE gives each routed expert a small dictionary and opens only those the frozen router selects. At block 15 of Gemma 4 26B-A4B, after ten million C4 training tokens and across three seeds, the expert-aligned autoencoder explains 0.518 of the variance in router-weighted expert outputs on a fixed evaluation capture. A rank-addressed control with the same parameters but sixteen times as many candidates reaches 0.481, a candidate-matched control 0.458, and an architecture-matched control with randomized expert identity 0.387. A transcoder variant also leads all three controls: 0.432, against 0.347, 0.371, and 0.118. Part of this advantage is architectural: each expert writes into a subspace of at most a quarter of the model width, so naming the expert names the subspa
    
[^95]: 回顾过去：上下文对检测欺骗性陈述的影响

    Looking Into The Past: The Impact of Context on Detecting Deceptive Statements

    [https://arxiv.org/abs/2610.27407](https://arxiv.org/abs/2610.27407)

    本研究表明，在《阿瓦隆》隐藏角色游戏的欺骗性消息检测任务中，为目标消息加入先前的游戏对话上下文，可使大语言模型的准确率提升高达31%。

    

    欺骗检测通常孤立地考虑单条消息。先前在声明验证和角色检测方面的工作表明，诸如游戏历史之类的额外信息对成功预测至关重要。在这项工作中，我们展示了在隐藏角色游戏《阿瓦隆》的欺骗性消息检测任务中，先前的游戏对话所带来的价值。研究结果表明，为目标消息添加游戏上下文后，大语言模型在多项指标上均有所提升，其中准确率最高可提升31%。

    arXiv:2610.27407v1 Announce Type: new  Abstract: Deception detection typically considers messages in isolation. Previous works in claim verification and role detection have displayed that additional information such as gameplay history is crucial to successful predictions. In this work, we demonstrate the value of preceding gameplay dialogue when applied to deceptive message detection within the hidden role game of Avalon. The findings show that LLMs can see improvement across a number of metrics, including accuracy by up to 31\% with the addition of gameplay context for the target message.
    
[^96]: 更多记忆，更差的智能体：LLM智能体中的错误再现与反持久性

    More Memory, Worse Agents: Error Reproduction and Anti-Persistence in LLM Agents

    [https://arxiv.org/abs/2610.27406](https://arxiv.org/abs/2610.27406)

    论文揭示了LLM智能体外部记忆中存储的抽象错误会通过“错误再现”这一反馈循环不断自我复制和延续，并据此提出了一种名为“反持久性”的记忆设计来抑制错误的持续传播。

    

    LLM智能体可以通过将自身执行过程中产生的经验存储到外部记忆中，并在后续决策时复用这些经验，从而在不更新模型参数的情况下跨任务提升性能。一个核心设计问题是：应该只保留观察到的轨迹和结果，还是同时保留从中推断出的规则、总结和技能。保留这些抽象化知识可以让智能体直接复用策略，而无需反复从原始轨迹中重新推导。然而，存储的抽象知识中的错误可能通过检索和复用传播到后续决策中。这些错误还会影响执行轨迹的形成，而这些轨迹又会成为新的抽象知识的基础，从而复现原始错误。我们将这种反馈机制称为“错误再现”：错误通过抽象、执行、再抽象的循环不断再生，使其持久性超出了直接复用原始错误知识的范畴。我们提出了“反持久性”，一种记忆设计……（摘要在此处被截断）

    arXiv:2610.27406v1 Announce Type: new  Abstract: LLM agents can improve across tasks without updating model parameters by storing experience from their own executions in external memory and reusing it to guide later decisions. A central design question is whether to retain only observed trajectories and outcomes or also the rules, summaries, and skills inferred from them. Retaining these abstractions allows agents to reuse strategies without repeatedly deriving them from raw trajectories. However, errors in stored abstractions can propagate to subsequent decisions through retrieval and reuse. They can also shape execution trajectories, which then become the basis for new abstractions that reproduce the original errors. We term this feedback mechanism error reproduction: errors are regenerated through a cycle of abstraction, execution, and re-abstraction, extending their persistence beyond direct reuse of the original faulty knowledge. We propose Anti-Persistence, a memory design that a
    
[^97]: 引导文本到语音合成中的语调曲调

    Steering intonational tunes in text-to-speech synthesis

    [https://arxiv.org/abs/2610.27403](https://arxiv.org/abs/2610.27403)

    该研究通过交换干预定位了残差向量量化文本到语音模型中英语核心语调的载体——主干网络的隐藏状态而非token堆栈第一层，并证明可用均值差向量引导该状态来实现对语调的控制。

    

    arXiv:2610.27403v1 公告类型：交叉 摘要：基于残差向量量化构建的文本到语音系统通常分两个阶段生成音频：一个transformer主干网络输出每帧token堆栈的第一层，而一个较小的深度模型自回归地生成其余部分。我们探究这类模型中哪些组件负责韵律语调，更准确地说，负责生成英语核心调（例如，区分陈述句与疑问句上升旋律的下降音高旋律），以及是否可以通过对这些组件的干预来控制语调。基于提示的控制大多失败；只有句末的问号是一个可靠的调节手段。在CSM和S2模型中进行的交换干预定位了语调的来源：主干网络的隐藏状态通过深度模型将语调传递到波形中，而token堆栈的第一层几乎不携带或不携带语调信息。通过使用在以句末（原文在此处截断）……

    arXiv:2610.27403v1 Announce Type: cross  Abstract: Text-to-speech systems built on residual vector quantization often generate audio in two stages: a transformer backbone emits the first level of each frame's token stack, and a smaller depth model generates the rest autoregressively. We ask which components of such a model are responsible for prosodic intonation, more precisely for generating the English nuclear tune (for instance, the falling pitch melody that distinguishes an assertion from a question's rising melody), and whether interventions on those components can be used to control intonation. Prompt-based control mostly fails; only a final question mark is a reliable lever. Interchange interventions in CSM and S2 locate the tune: the backbone's hidden state carries it to the waveform through the depth model, and the first level of the stack carries little or none of it. Steering the backbone state with a difference-of-means vector computed on new material generated with a final
    
[^98]: DStill：面向块扩散投机解码中加速单步草稿生成的轨迹一致性蒸馏

    DStill: Trajectory Consistency Distillation for Accelerated One-Step Drafting in Block Diffusion Speculative Decoding

    [https://arxiv.org/abs/2610.27379](https://arxiv.org/abs/2610.27379)

    提出DStill两阶段蒸馏框架，通过轨迹一致性蒸馏将块扩散投机解码中草稿器的多步迭代细化压缩为单步草稿生成，在降低推理延迟的同时保持草稿的保真度。

    

    块扩散投机解码通过并行起草多个token来加速大语言模型推理。然而，离散扩散中的单步草稿生成受到token因子化误差的限制：每个token都是从同一个掩码块中预测得到的，而没有以该步骤中其他位置所选token作为条件。这会产生不一致的草稿并降低接受长度。迭代细化通过让后续预测以先前已确定的token为条件来缓解这一局限，但额外的推理过程会增加草稿生成成本。为解决草稿保真度与推理延迟之间的权衡，我们提出了DStill，这是一个两阶段框架，将目标条件化草稿器自身的多步细化过程蒸馏为单步草稿生成。我们首先训练草稿器完成部分揭示的块，然后利用其细化轨迹来训练学生模型在单次推理中预测最终完成结果。使用……

    arXiv:2610.27379v1 Announce Type: new  Abstract: Block-diffusion speculative decoding accelerates large language model inference by drafting multiple tokens in parallel. However, one-step drafting in discrete diffusion is limited by token factorization error: each token is predicted from the same masked block, without conditioning on the tokens selected at other positions in that step. This can produce inconsistent drafts and reduce acceptance length. Iterative refinement mitigates this limitation by conditioning later predictions on previously committed tokens, though additional passes increase drafting cost. To address this trade-off between draft fidelity and inference latency, we introduce DStill, a two-stage framework that distills a target-conditioned drafter's own multi-step refinement into one-step drafting. We first train the drafter to complete partially revealed blocks, then use its refinement trajectories to train a student to predict the final completion in one pass. Using
    
[^99]: SHAM-Bench：一个跨多任务的片段级幻觉检测基准

    SHAM-Bench: A Span-level Hallucination Benchmark Across Multiple Tasks

    [https://arxiv.org/abs/2610.27370](https://arxiv.org/abs/2610.27370)

    提出了SHAM-Bench——一个涵盖六个NLG任务、共6461个样本的统一片段级幻觉检测基准，评估发现微调检测器在域内表现良好但在分布偏移下退化，而基于提示的推理方法泛化能力更强。

    

    片段级幻觉检测能够精确定位大语言模型输出中缺乏依据支持的内容，从而提供可操作的反馈和有针对性的缓解措施。目前针对片段级幻觉检测方法的评估在任务、标注格式和指标上较为零散，且大多数片段级数据集仅覆盖单一的自然语言生成（NLG）领域，这使得难以评估某个检测方法是否真正能够在不同任务和领域之间泛化。在本文中，我们提出了SHAM-Bench（跨多任务的片段级幻觉检测基准），这是一个统一的片段级幻觉检测基准，涵盖六个NLG任务，通过统一的模式和评估协议，从现有人工标注数据集中聚合而成，共计6,461个样本。利用SHAM-Bench，我们评估了现有的幻觉检测方法，发现经过微调的检测器通常在域内表现良好，但在分布偏移下性能下降，而基于提示的推理方法（摘要在此处截断）……

    arXiv:2610.27370v1 Announce Type: new  Abstract: Span-level hallucination detection pinpoints unsupported content in LLM outputs, enabling actionable feedback and targeted mitigation. Existing evaluations for span-level hallucination detection methods are fragmented across tasks, annotation formats, and metrics, and most span-level datasets cover only a single NLG domain, making it difficult to assess whether a detection method actually generalizes across tasks and domains. In this paper, we introduce SHAM-Bench (Span-level Hallucination Benchmark Across Multiple Tasks), a unified benchmark for span-level hallucination detection spanning six NLG tasks, aggregated from existing human-annotated datasets totaling 6,461 examples under a common schema and evaluation protocol. Using SHAM-Bench, we evaluate existing hallucination detection methods and find that fine-tuned detectors often perform well in-domain but degrade under distribution shift, whereas prompt-based methods with reasoning a
    
[^100]: 更多探索，更少验证：高效的反思式提示优化

    Explore More, Validate Less: Efficient Reflective Prompt Optimization

    [https://arxiv.org/abs/2610.27367](https://arxiv.org/abs/2610.27367)

    提出了一种两阶段框架，将反思式候选池的构建与完整验证的分配解耦，使候选池规模和验证预算可以独立设置，从而大幅提升提示优化的效率。

    

    反思式提示优化利用任务反馈来改进指令，但将候选生成与基于完整验证的选择耦合在一起，会使搜索过程代价高昂且只能顺序进行。我们提出了一个两阶段框架，将反思式候选池的构建与验证资源的分配解耦。在构建阶段，我们采样多个父提示，并在并行的反思分支中复用每个父提示的训练响应。每个修订版本在小型训练小批量数据上与其父提示进行筛选比较，使得局部改进的候选无需完整验证即可支持进一步的修订。构建完成后，探索策略在固定的完整验证预算下从冻结的候选池中选择候选，并返回测得验证分数最高的提示。这种解耦使得候选池大小和验证预算可以独立设置。我们评估了随机选择以及基于相似度的学习分数预测方法……

    arXiv:2610.27367v1 Announce Type: new  Abstract: Reflective prompt optimization improves instructions using task feedback, but coupling candidate generation with full-validation-based selection can make search costly and sequential. We propose a two-stage framework that decouples reflective candidate-pool construction from validation allocation. During construction, we sample multiple parents and reuse each parent's training responses across parallel reflection branches. Each revision is screened against its parent on a small training minibatch, allowing locally improved candidates to support further revisions without full validation. After construction, an exploration policy selects candidates from the frozen pool under a fixed full-validation budget, and the prompt with the highest measured validation score is returned. This separation allows candidate-pool size and validation budget to be set independently. We evaluate random selection and learned score prediction with similarity-ba
    
[^101]: 论论辩图生成中大语言模型验证

    On the LLM Verification for Argument Graph Generation

    [https://arxiv.org/abs/2610.27361](https://arxiv.org/abs/2610.27361)

    本文提出了一种面向论辩图生成的分层验证框架，主张根据各类工具（确定性检查、求解器、大语言模型、辩论、人类）所能保证的能力来合理分配验证任务。

    

    多智能体大语言模型系统能够对自然语言论证进行端到端的形式化处理，通过对论证成分进行分类、用关系将它们连接起来，并将所得的图丰富为形式化模型，从而将纯文本转化为结构化图。对这些图的质量保障几乎总是依赖另一个大语言模型来完成，即审查者、评判者或批评者智能体。另一方面，在相邻的神经符号领域，此类生成输出通常通过外部检查和求解器进行符号化验证。在本工作中，我们主张在论辩图验证中，任务的分配应当类似，并基于每种工具所能保证的能力。我们提出了分层验证方法：从确定性检查开始，进而采用基于求解器和大语言模型的审查，并仅在存在不确定性时使用辩论机制，将其他任何工具都无法判断的内容留给人类裁决。我们通过一项在人工标注的图中注入结构缺陷的研究来支撑这一原则。大语言模型审查者……

    arXiv:2610.27361v1 Announce Type: new  Abstract: Multi-agent LLM systems can formalize natural-language argumentation end-to-end, turning plain text into structured graphs by classifying components, connecting them with relations, and enriching the resulting graphs into formal models. Quality assurance for those graphs is almost always another LLMs: a reviewer, judge, or critic agent. On the other side, in adjacent neuro-symbolic fields, such generated output is often verified symbolically by external checks and solvers. In this work, we argue that in argument graph verification, the allocation should be similar, and based on what each instrument can guarantee. We propose layered verification: starting from deterministic checks, moving to solver-based and LLM reviews, and utilizing debate only under uncertainty, leaving human judgment for what no other instrument can decide. We ground our principle with a study on human-annotated graphs with injected structural defects. An LLM reviewer
    
[^102]: 全球安全盲区：翻译作为低资源语言中的安全绕过手段

    Global Security Blind Spots: Translation as a Safety Bypass in Low-Resource Languages

    [https://arxiv.org/abs/2610.27304](https://arxiv.org/abs/2610.27304)

    该论文揭示了大型语言模型的英语安全对齐无法可靠迁移至低资源语言——在英语中被成功拒绝的有害请求经翻译后可诱发不安全回答，各模型平均绕过率高达21.9%至44.4%，构成全球性的安全盲区。

    

    大型语言模型在全球范围内部署，然而其安全对齐与评估却仍集中于英语：在英语中被拒绝的有害请求，一旦被翻译成低资源语言，便可能引发不安全的回答。我们在翻译绕过威胁模型下，对十个广泛部署的大语言模型在79种语言（英语及78种低资源语言）上进行了评估，并采用一种以英语安全性为条件的度量方法，从而将翻译绕过效应与英语本身已存在的不安全性区分开来。研究结果表明，英语的安全性无法可靠迁移：在英语回答安全的前提下，各模型的平均绕过率介于21.9%至44.4%之间。绕过风险集中于部分语系之中，且这一模式在各模型间普遍存在。尽管这种共同的排序与翻译退化存在混淆，但其在不同模型间的差异是显著的，并且经受住了我们施加的所有控制。我们还发现，单看各模型自身的绕过率无法得到一个稳定的排名：十个模型中有九个……（原文在此截断）

    arXiv:2610.27304v1 Announce Type: cross  Abstract: Large language models are deployed globally, yet safety alignment and evaluation remain concentrated in English: harmful requests refused in English can elicit unsafe completions when translated into low-resource languages. We evaluate ten widely deployed LLMs across 79 languages (English plus 78 low-resource) under a translation-bypass threat model, using a metric conditioned on English safety, which separates translation bypass from unsafety already present in English. English safety does not transfer reliably: conditioned on safe English completions, mean bypass rates range from 21.9% to 44.4%. Bypass risk concentrates in a subset of language families, a pattern broadly shared across models. The shared ordering is confounded with translation degeneracy, but its variation across models is significant and survives every control we apply. We also show that per-model bypass rates do not identify a stable ranking: nine of ten models chan
    
[^103]: 评估上下文压缩对后续问题的效果

    Evaluating Context Compaction on the Questions That Come Later

    [https://arxiv.org/abs/2610.27301](https://arxiv.org/abs/2610.27301)

    研究发现为当前问题挑选句子的压缩方法虽能提升当前问题的准确率，但在后续问题上并不优于随机删除，而与问题无关的词级压缩方法LLMLingua-2在后续问题上反而表现更佳，表明上下文压缩需要兼顾未来的问题。

    

    语言模型智能体会针对当前问题压缩其上下文，但压缩后的上下文还必须服务于后续出现的问题。我们在这些后续问题上评估压缩效果，使用了四个语料库，这些语料库对每篇文本中多个问题的证据进行了标注，并记录了智能体轨迹，同时在相同预算下将每种方法与随机删除进行比较。在MultiRC数据集上，保留gpt-6-luna（通过OpenAI Decisions API）判断为当前问题所需的句子，可以在仅使用25% token的情况下，将该问题的准确率提高11.4个百分点。然而，在后续问题上，任何为当前问题选择句子的方法都没有明显优于随机删除句子。删除词语而非句子效果更好：忽略问题的LLMLingua-2在相同token数量下，在后续MultiRC问题上比随机句子删除高出12.5个百分点，它保留的内容份额约为随机删除的两倍。

    arXiv:2610.27301v1 Announce Type: cross  Abstract: Language-model agents compact their context for the question at hand, but the compacted context must also serve questions that come later. We evaluate compaction on these later questions, using four corpora that annotate the evidence of several questions per text and recorded agent trajectories, and compare every method with random deletion at the same budget. On MultiRC, keeping the sentences that gpt-6-luna (through the OpenAI Decisions API) judges necessary for the current question raises its accuracy on that question by 11.4 points at 25% of the tokens. On the later questions, however, no method that selects sentences for the current question does detectably better than deleting sentences at random. Deleting words instead of sentences does better: LLMLingua-2, which ignores the question, beats random sentence deletion at the same token count by 12.5 points on the later MultiRC questions. It keeps about twice random deletion's share
    
[^104]: RealControlBench：一个面向基于可解释性的大语言模型与智能体细粒度控制的基准测试

    RealControlBench: A Benchmark for Interpretability-Informed Fine-Grained Control of LLMs and Agents

    [https://arxiv.org/abs/2610.27295](https://arxiv.org/abs/2610.27295)

    提出了RealControlBench，一个涵盖指令遵循、工具使用决策、推理性能与长度控制、安全意识五个维度的综合基准，用于在真实复杂的多轮智能体交互场景中评估大语言模型与智能体的细粒度行为控制。

    

    现有的行为引导评测基准建立在简化的、孤立的生成任务之上，并在受控设置下仅评估单一行为属性，无法反映实际部署场景的多样性与复杂性。因此，在这些基准上的表现只能有限地反映引导方法在真实世界中的效果。为此，我们提出了RealControlBench，这是一个全面的基准测试，用于评估大语言模型与大语言模型智能体在多样行为目标和交互设置下的细粒度行为引导。RealControlBench将引导评测从孤立的生成任务扩展到复杂的、基于实际执行的多轮智能体交互，涵盖五个互补维度：指令遵循控制、智能体工具使用决策控制、推理性能引导、推理长度控制以及基于轨迹的安全意识。该基准并非仅测量引导的成功与否，而是……

    arXiv:2610.27295v1 Announce Type: new  Abstract: Existing steering benchmarks are built around simplified, isolated generation tasks and evaluate a single behavioral attribute under controlled settings that do not reflect the diversity and complexity of practical deployment. As a result, performance on these benchmarks provides only a limited picture of how steering methods perform in the real world. We therefore introduce RealControlBench, a comprehensive benchmark for evaluating fine-grained steering of LLMs and LLM agents across diverse behavioral targets and interaction settings. RealControlBench extends steering evaluation beyond isolated generations to complex, execution-grounded, multi-turn agent interactions, covering five complementary dimensions: instruction-following control, agentic tool-use decision control, reasoning-performance steering, reasoning-length control, and trajectory-conditioned safety awareness. Rather than measuring steering success alone, the benchmark join
    
[^105]: TEX：通过基于执行的交叉验证实现测试时代理的测试时扩展

    TEX: Test-Time Scaling Testing Agents via Execution-based Cross-Validation

    [https://arxiv.org/abs/2610.27290](https://arxiv.org/abs/2610.27290)

    TEX提出了一种基于执行驱动交叉验证的协作式测试时扩展框架，通过并行代理生成测试和补丁并交叉执行形成K×K反馈矩阵实现客观验证，在SWT-Bench Verified上以86.4%的成绩创下新的最先进水平。

    

    测试用例生成对语言模型至关重要，但仍然具有挑战性。现有的测试时扩展方法要么孤立地处理候选方案，要么聚合执行轨迹，这些方法无法扩展到长时程任务。我们提出了TEX，一个基于执行驱动交叉验证的协作式测试时扩展框架。TEX并行化多个代理来生成测试和补丁，将每个测试与每个补丁交叉执行，形成一个K×K的反馈矩阵。各代理利用自己的矩阵切片进行改进，无需冗长的轨迹共享即可实现客观验证。在SWT-Bench Verified基准上，TEX达到了86.4%的准确率，创下新的最先进水平，比单代理基线提升了9.0%。此外，即使从TEX输出中随机选择（84.4%）也超越了之前的SOTA（84.0%），表明协作式执行反馈提升的是解决方案的整体质量，而不仅仅是最佳样本。我们的研究结果突出了基于执行的反馈作为一种有效的扩展机制。

    arXiv:2610.27290v1 Announce Type: new  Abstract: Test-case generation is critical but remains challenging for language models. Existing test-time scaling methods either isolate candidates or aggregate traces, which do not scale to long-horizon tasks. We present TEX, a collaborative test-time scaling framework based on execution-driven cross-validation. TEX parallelizes agents to generate tests and patches, cross-executing each test against every patch to form a $K \times K$ feedback matrix. Agents use their matrix slices for refinement, enabling objective validation without verbose trace sharing. On SWT-Bench Verified, TEX achieves 86.4%, establishing a new state-of-the-art and improving +9.0% over single-agent baselines. Moreover, even random selection from TEX outputs (84.4%) surpasses the previous SOTA (84.0%), showing collaborative execution feedback improves solution quality, not just best-case samples. Our results highlight execution-grounded feedback as an effective scaling mech
    
[^106]: 仇恨何时能迁移？跨语言仇恨言论检测的大规模研究

    When Does Hate Transfer? A Large-Scale Study of Cross-Lingual Hate Speech Detection

    [https://arxiv.org/abs/2610.27285](https://arxiv.org/abs/2610.27285)

    该研究通过对14种语言的大规模实验发现，语言相似性是跨语言仇恨言论检测迁移效果的最强预测因子，而共享的文化线索和交际规范可能解释其余未知的迁移模式差异。

    

    arXiv:2610.27285v1 宣布类型：cross 摘要：有效检测仇恨内容对于维护健康的在线社区至关重要，然而其性能在不同语言之间可能存在显著差异。本研究探讨了一个核心实践问题：来自一种语言的训练数据能否以相当的性能支持另一种语言的仇恨言论检测？为回答这一问题，我们在14种语言上开展了大规模研究。我们观察到一致的模式：某些语言对之间的迁移效果明显优于其他语言对。在我们测量的因素中，语言相似性是这些模式的最强预测因子，而时间重叠和数据集规模差异的作用较小。这些可测量的因素并不能完全解释所观察到的模式；一项补充性的定性分析表明，共享的文化线索和交际规范可能与部分剩余情况相关，尤其是在几个聚类内部。我们展示了这种文化……（摘要原文在此处截断）

    arXiv:2610.27285v1 Announce Type: cross  Abstract: Effective detection of hate content is essential for maintaining healthy online communities, yet performance can vary substantially across languages. This work investigates a core practical question: Can training data from one language support hate-speech detection in another language with comparable performance? To answer this question, we conduct a large-scale study across 14 languages. \change{We observe consistent patterns in which certain language pairs transfer more effectively than others. Among the factors we measure, linguistic similarity is the strongest predictor of these patterns, with temporal overlap and dataset-size difference playing smaller roles. These measurable factors do not fully explain the observed patterns; a complementary qualitative analysis suggests that shared cultural cues and communicative norms may be associated with some of the remaining cases, particularly within several clusters. We present this cultu
    
[^107]: FLINT：固定预算的后期交互

    FLINT: Fixed-budget Late Interaction

    [https://arxiv.org/abs/2610.27273](https://arxiv.org/abs/2610.27273)

    FLINT 提出一种基于注意力池化的架构，将查询和文档对称地压缩为固定数量的嵌入向量，在大幅降低存储开销和检索时的向量比较次数的同时，保持了与多向量模型相当的检索效果。

    

    arXiv:2610.27273v1 公告类型： cross 摘要：以 ColBERT 为代表的多向量检索模型通过在词元级别匹配查询和文档，往往在同等规模下优于单向量模型。然而，为每个词元保留一个向量会显著增加存储需求，并且与单向量双编码器相比会降低检索速度，限制了其可扩展性。我们提出了 FLINT（Fixed-budget Late INTeraction，固定预算后期交互），这是一种利用基于注意力的池化机制，以固定数量的嵌入向量来表示每个查询和文档的架构。与仅对文档表示进行池化而保留词元级查询的架构不同，FLINT 对查询和文档两侧对称地应用池化，从而同时限制了文档存储量以及检索过程中所需的向量比较次数。FLINT 在 BEIR 上取得了与最接近的基线相当的排序效果，并在包含更长、更复杂查询的多个基准测试上超越了它们，包括 BrowseComp+、CRUMB 和 T

    arXiv:2610.27273v1 Announce Type: cross  Abstract: Multi-vector retrieval models such as ColBERT often outperform single-vector models of comparable scale by matching queries and documents at the token level. However, retaining a vector for every token substantially increases storage requirements and slows retrieval compared to single-vector bi-encoders, limiting their scalability. We introduce FLINT (Fixed-budget Late INTeraction), an architecture that uses attention-based pooling to represent each query and document with a fixed number of embedding vectors. Unlike architectures that pool only document representations while retaining token-level queries, FLINT applies pooling symmetrically to both sides, limiting document storage and the number of vector comparisons needed during retrieval. FLINT achieves comparable ranking effectiveness to the closest baselines on BEIR and outperforms them on several benchmarks with longer and more complex queries, including BrowseComp+, CRUMB, and T
    
[^108]: 非洲语言人工智能安全就绪度报告

    African Languages AI Safety Readiness Report

    [https://arxiv.org/abs/2610.27269](https://arxiv.org/abs/2610.27269)

    本报告通过考察AI资源可用性、国家AI政策支持、文本基准表现和说话者人口四个因素，评估了非洲语言参与人工智能安全讨论的就绪度。

    

    随着大语言模型（LLMs）在各语言和领域的广泛应用，人工智能安全已成为全球关注的关键问题。然而，由于资源限制，研发工作往往优先考虑少数高资源语言，导致许多其他语言，尤其是非洲使用的语言，代表性不足。虽然现有关于语言“资源性”的研究主要考察是否有足够的资源来开发语言技术，但语言在人工智能安全研究中是否得到充分代表的问题直到最近才受到关注。本报告基于现有文献和数据，考察了非洲语言参与人工智能安全讨论的就绪度。报告考虑了四个因素：包括文本数据在内的AI相关资源的可用性；国家AI政策支持；在既定文本基准上的表现；以及说话者人口数量。

    arXiv:2610.27269v1 Announce Type: new  Abstract: With the widespread adoption of large language models (LLMs) across languages and domains, AI safety has emerged as a critical global concern. However, due to resource constraints, research and development efforts tend to prioritize a limited number of high-resource languages, leaving many others, especially those spoken in Africa, underrepresented. While existing research on language "resource-ness" has largely examined whether sufficient resources are available to develop language technologies, the question of whether languages are adequately represented in AI safety research has received attention only more recently. This report examines the readiness of African languages for inclusion in AI safety discussions, drawing on existing literature and data. It considers four factors: the availability of AI-relevant resources, including text data; national AI policy support; performance on established text benchmarks; and speaker population.
    
[^109]: 基于记录推导的安全偏好监督审计

    Auditing Safety Preference Supervision with Recorded Derivations

    [https://arxiv.org/abs/2610.27266](https://arxiv.org/abs/2610.27266)

    该论文提出符号化偏好标注方法，通过记录属性分类、明确规则和固定判定排序，使每个安全偏好标签都能从存储的推导过程中重新计算验证，从而实现对偏好监督的可审计性。

    

    偏好标签用于对齐模型训练，但其背后的判断难以检查或验证。在本文中，我们研究了当中间决策被记录时如何对偏好构建过程进行检查，以及其中哪些部分仍然存在不确定性。我们提出了符号化偏好标注，这是一种由预训练语言模型对提示词和响应的预定义属性进行分类的方法。明确的规则为响应分配判定结果，固定的判定排序决定偏好标签。每个偏好标签都具有可记录的推导过程：响应判定和偏好标签可以从存储的属性分配、规则和判定排序中重新计算得出，而无需重新运行语言模型。在由人工标注者标注的200个提示词上，语言模型为192个提示词生成了有效的结构化输出。全部八次解析失败都发生在被标记为有害的自残提示词子集中。使用465个偏好对，我们训练……

    arXiv:2610.27266v1 Announce Type: cross  Abstract: Preference labels align model training, but the judgments behind them are difficult to inspect or verify. In this paper, we examine how preference construction can be checked when intermediate decisions are recorded, and which of them remain uncertain. We present Symbolic Preference Labeling, a method in which a pretrained language model classifies predefined attributes of prompts and responses. Explicit rules assign response verdicts, and a fixed verdict ranking determines the preference label. Each preference label has a recorded derivation: the response verdicts and preference label can be recomputed from the stored attribute assignments, rules, and verdict ranking without rerunning the language model. On 200 prompts annotated by human raters, the language model produces valid structured outputs for 192 prompts. All eight parsing failures occur in the subset of self-harm prompts marked as harmful. Using 465 preference pairs, we trai
    
[^110]: TxnQA：针对交易事件序列的分析推理与证据溯源基准测试

    TxnQA: Benchmarking Analytical Reasoning and Evidence Grounding over Transaction Event Sequences

    [https://arxiv.org/abs/2610.27256](https://arxiv.org/abs/2610.27256)

    TxnQA提出了一个针对交易事件序列分析推理的新基准，包含27个任务族和欺诈风险推理套件，评估发现LoRA适配虽能有效规范输出格式，但算术能力仍是模型性能的主要瓶颈。

    

    分析商户交易历史需要在不规则事件流上进行动态时间过滤、实体解析和聚合操作——这些操作在静态表格或时间序列预测基准中难以得到充分体现。我们提出TxnQA，一个分析型问答基准，包含27个任务族（81个模板）以及一个5场景的欺诈风险推理套件，该套件在合成的、经隐私脱敏处理的交易数据上同时评估判定结果和支持证据。标准答案通过程序化方式计算，模型性能被分解为格式遵循率（FFR）、条件内容准确率（CA）和端到端正确性（E2E）。对基于提示的专有基线模型和六个经LoRA适配的开源权重模型的评估表明，适配能够可靠地规范输出模式，但仍存在算术瓶颈：Gemma-4-31B-IT和Qwen3-8B分别取得0.867和0.816的E2E分数，而所评估的MoE模型检查点仅达到0.632。

    arXiv:2610.27256v1 Announce Type: new  Abstract: Analyzing merchant transaction histories requires dynamic temporal filtering, entity resolution, and aggregation over irregular event streams--operations poorly captured by static tabular or time-series forecasting benchmarks. We introduce TxnQA, an analytical QA benchmark comprising 27 task families (81 templates) alongside a 5-scenario fraud-risk reasoning suite that evaluates both verdicts and supporting evidence over synthetic, privacy-mitigated transaction data. Ground-truth answers are computed programmatically, and model performance is decomposed into format-following rate (FFR), conditional content accuracy (CA), and end-to-end correctness (E2E). Evaluating prompted proprietary baselines and six LoRA-adapted open-weight models shows that adaptation reliably enforces output schemas but leaves arithmetic bottlenecks: Gemma-4-31B-IT and Qwen3-8B attain E2E scores of 0.867 and 0.816, whereas evaluated MoE checkpoints reach only 0.632
    
[^111]: 面向目标模型的大语言模型幻觉压力测试精准筛选方法

    Target-Aware Curation of Hallucination Stress Tests for Large Language Models

    [https://arxiv.org/abs/2610.27247](https://arxiv.org/abs/2610.27247)

    该论文提出了一种目标感知的迭代提示编辑方法，利用LLM编辑器根据目标模型的回答和评判者反馈不断修改种子提示，从而精准暴露特定模型的幻觉弱点，即使各模型在标准压力测试中平均分数相似也能区分出差异。

    

    幻觉压力测试即使在不同提示会引发不同模型产生幻觉的情况下，也可能在各语言模型间产生相似的平均分数。我们开发了一种流程，通过迭代编辑基于来源的种子提示来暴露目标模型中的幻觉。一个LLM编辑器根据目标模型的回答以及评判者给出的特定类别分数和解释来修订每个提示。在主要设置中，目标模型还会收到关于其先前回答的反馈。实验涵盖来自七个数据集和十个主题的500个种子，在开放权重和闭源模型上评估了内在和外在幻觉。编辑显著提高了评判者评定的幻觉严重程度，超过了种子提示的水平。在基线配置中，针对外在幻觉的运行中有77.2%、针对内在幻觉的运行中有74.2%的保留回答落入评判者的强幻觉区间。后续编辑轮次贡献了超过一半的效果。

    arXiv:2610.27247v1 Announce Type: cross  Abstract: Hallucination stress tests can yield similar average scores across language models even when different prompts elicit hallucinations in each model. We develop a procedure that iteratively edits source-grounded seed prompts to expose hallucinations in a target model. An LLM editor revises each prompt in response to the target's answer and a judge's category-specific score and explanation. In the main setting, the target also receives feedback on its previous response. Experiments cover 500 seeds from seven datasets and ten topics, with intrinsic and extrinsic hallucinations evaluated across open-weight and closed-source models. Editing substantially increases judge-rated severity over the seeds. In the baseline configuration, retained responses fall in the judge's strong-hallucination bin in 77.2% of runs targeting extrinsic hallucination and 74.2% targeting intrinsic hallucination. Later editing rounds account for more than half of the
    
[^112]: 大语言模型推理的诊断分析

    Diagnostic Analysis for Large Language Model Reasoning

    [https://arxiv.org/abs/2610.27245](https://arxiv.org/abs/2610.27245)

    本文提出两阶段诊断评估方法，系统检验大语言模型推理图的结构可靠性，发现图可靠性受拓扑、提示设计和模型家族影响，且图结构仅在基于置信度分数区分正确答案和支持节点级诊断方面有价值，仅作为提示上下文时收益有限。

    

    推理结构（如链和图）越来越多地被用于概率推断、事实性验证和不确定性估计，但其结构可靠性很少被直接评估。我们通过在开放式多跳问答中提出的两阶段诊断评估来检验这种可靠性是否成立。第一阶段测试构建推理图所需的不同能力，结果表明图的可靠性随拓扑结构、提示设计和模型家族而变化。第二阶段将不确定性估计作为下游测试用例。生成边实验表明，基于图的计算对上游图错误敏感，相对于金标准图设置其性能有所下降。金标准图实验进一步表明，图结构仅作为提示上下文使用时收益甚微，但它能改善通过置信度分数对正确与错误答案的区分能力，并支持节点级诊断。

    arXiv:2610.27245v1 Announce Type: new  Abstract: Reasoning structures (e.g., chains and graphs) are increasingly used for probabilistic inference, factuality verification, and uncertainty estimation, but their structural reliability is rarely evaluated directly. We test whether this reliability holds through a proposed two-stage diagnostic evaluation in open-ended multi-hop QA. Stage 1 tests different capabilities required to construct reasoning graphs, showing that graph reliability varies with topology, prompt design, and model family. Stage 2 uses uncertainty estimation as a downstream test case. Generated-edge experiments show that graph-based computation is sensitive to upstream graph errors, with performance dropping relative to the gold-graph setting. Gold-graph experiments further show that graph structure provides little benefit when used only as prompt context, but improves the separation of correct from incorrect answers by confidence score and enables node-level diagnostics
    
[^113]: 无需关系信息即可预测线性关系可解码性

    Predicting Linear Relational Decodability Without the Relation

    [https://arxiv.org/abs/2610.27241](https://arxiv.org/abs/2610.27241)

    该论文提出“实体透镜”这一新颖的雅可比透镜变体，通过在与关系无关的中性提示下针对实体类型进行训练，能够在不使用任何关系特定信息的情况下，显著预测线性关系嵌入（LRE）在不同关系上忠实度的巨大差异。

    

    线性关系嵌入（Linear Relational Embeddings, LREs）使用仿射函数来近似语言模型从主体到客体的映射。其忠实度在不同关系之间差异巨大：例如 person_father（某人的父亲）关系的忠实度接近于零，而 country_capital_city（国家首都城市）关系则达到上限。我们研究了导致这种差异的因素，并证明这种差异可以从一个不了解底层关系的读出（readout）中被显著预测出来。为此，我们提出了一种新颖的雅可比透镜（Jacobian lens）形式——实体透镜（entity lens）。雅可比透镜是在所有词元和概念上进行估计的，而实体透镜 EL_t 则被训练用于预测模型在中性提示词"{} is known for"（“{} 以……闻名”）下对类型为 t 的实体会输出什么；例如，我们分别为国家和人物类型训练了实体透镜。在估计出实体透镜之后，我们测量它在多大程度上将模型对某个关系的答案排在高位，而在整个估计和预测过程中不使用任何关系特定的信息。在28个关系……（摘要原文在此处被截断）

    arXiv:2610.27241v1 Announce Type: new  Abstract: Linear relational embeddings (LREs) approximate a language model's subject-to-object mapping with an affine function. Their faithfulness varies enormously across relations, from near zero for person_father to the ceiling for country_capital_city. We investigate the factors that contribute to this variation and demonstrate that it can be significantly predicted from a readout that is not aware of the underlying relation. To do so, we propose a novel form of the Jacobian lens, the entity lens. Whereas the Jacobian lens is estimated across all tokens and concepts, the entity lens EL_t is trained to predict what the model will output for an entity of type t under the neutral prompt "{} is known for"; e.g., we train ELs for countries and persons. After estimating an entity lens, we measure how highly it ranks the model's answer for a relation without using any relation-specific information in estimation and prediction. Across 28 relations of 
    
[^114]: 基于强化学习的面向CEFR的俄语文本简化

    CEFR-oriented Russian Text Simplification Using Reinforcement Learning

    [https://arxiv.org/abs/2610.27237](https://arxiv.org/abs/2610.27237)

    该研究利用拒绝采样和直接偏好优化（DPO）等偏好优化方法微调YandexGPT，实现了将俄语文本自动简化至CEFR的A1、A2和B1语言水平的可控简化，最佳模型达到87.15%的目标级别准确率。

    

    我们提出了一项关于俄语文本受控自动简化的研究，采用基于偏好的优化算法，重点是将文本适配至A1、A2和B1语言熟练水平。我们使用拒绝采样和直接偏好优化（DPO）方法对YandexGPT进行微调，以提升对生成文本复杂度的控制能力。我们比较了多种训练配置，这些配置在数据集组成、文本类型以及DPO偏好对构建策略方面各不相同。在自动评估方面，我们训练了一个基于ruRoBERTa-large的回归模型，用于预测对应CEFR级别的连续复杂度分数。我们还考虑了语义相似度、词汇覆盖率和量化可读性指标。人工评估则考察了简洁性、流畅性和充分性。我们表现最佳的模型实现了87.15%的目标级别准确率，MAE为0.35个CEFR级别，目标区间MAE为0.11。人工评估显示，部分模型生成的简化文本在质量上表现良好。

    arXiv:2610.27237v1 Announce Type: new  Abstract: We present a study on controlled automatic simplification of Russian texts using preference-based optimization algorithms with a focus on adapting texts to the A1, A2 and B1 proficiency levels. We fine-tune YandexGPT using Rejection Sampling and Direct Preference Optimization (DPO) methods to improve control over the complexity of the generated text. We compare several training configurations that differ in dataset composition, text type and the strategy used to construct preference pairs for DPO. For automatic evaluation we train a regression model based on ruRoBERTa-large to predict continuous complexity scores corresponding to CEFR levels. We also consider semantic similarity, lexical coverage and quantitative readability metrics. Human evaluation assesses simplicity, fluency and adequacy. Our best model achieves 87.15\% target level accuracy, an MAE of 0.35 CEFR levels and a target interval MAE of 0.11. Manual assessment shows that 7
    
[^115]: MMedAgent-Plus：跨模态、任务与工作流扩展的多模态医疗智能体

    MMedAgent-Plus: Scaling Multimodal Medical Agents across Modalities, Tasks and Workflows

    [https://arxiv.org/abs/2610.27233](https://arxiv.org/abs/2610.27233)

    MMedAgent-Plus是一个多模态医疗工具使用智能体，通过视觉-语言规划器协调18个专科模型（覆盖12个任务和8种模态）来执行复杂医疗工作流，并在多种交互设置中实现高达98.1%的工具调用准确率。

    

    医疗工作流通常需要多个相互依赖的分析，但现有的专科模型和通用AI模型并不是为协调这些分析而设计的。我们提出了MMedAgent-Plus，这是一个多模态医疗工具使用智能体，它将视觉-语言规划器与18个领先的专科模型相结合，涵盖12个任务和8种模态。通过统一的工具接口，规划器选择专科模型、按顺序执行它们，并基于其输出综合生成响应。该规划器在来自32个公开医疗基准的208,928个样本上进行了指令微调，涵盖单工具和多工具交互的单轮与多轮场景。我们分别在保留样本上评估了工具编排、响应综合和端到端执行。MMedAgent-Plus在单工具多轮、多工具多轮和多工具单轮设置中分别实现了91.0%、98.1%和80.7%的工具调用准确率。在给定正确的工具输出……（摘要不完整）

    arXiv:2610.27233v1 Announce Type: cross  Abstract: Medical workflows often require multiple interdependent analyses, but existing specialist and generalist AI models are not designed to coordinate them. We introduce MMedAgent-Plus, a multimodal medical tool-use agent that combines a vision-language planner with 18 leading specialist models spanning 12 tasks and 8 modalities. Through a unified tool interface, the planner selects specialists, executes them in sequence, and synthesizes responses grounded in their outputs. The planner is instruction-tuned on 208,928 samples from 32 public medical benchmarks, covering single- and multi-tool interactions across single and multiple rounds. We separately evaluate tool orchestration, response synthesis, and end-to-end execution on held-out samples. MMedAgent-Plus achieves tool-call accuracies of 91.0%, 98.1%, and 80.7% in single-tool multi-round, multi-tool multi-round, and multi-tool single-round settings, respectively. Given correct tool outp
    
[^116]: 哪个智能体的技能需要改变？面向多智能体团队进化的组相对信用分配方法

    Which Agent's Skill Should Change? Group-Relative Credit Assignment for Evolving Multi-Agent Teams

    [https://arxiv.org/abs/2610.27227](https://arxiv.org/abs/2610.27227)

    GRSE受GRPO启发，通过在相同技能库下多次执行同一问题并进行组内对比，将团队奖励精准追溯到具体的智能体-技能对，解决了“技能出现不等于技能责任”的信用分配难题，显著提升了多智能体协作性能。

    

    合作型语言模型智能体可以通过修订可复用的技能来在不更新权重的情况下持续改进，但前提是必须将共享的团队奖励追溯到应负责任的智能体与技能。现有方法对每条轨迹中出现的技能给予信用，然而“出现”并不等于“责任”：在成功运行中被激活的有害技能依然会获得正向信用。我们提出了GRSE（组相对技能进化，Group-Relative Skill Evolution），它通过比较来分配行为信用：受GRPO启发，该方法在相同的技能库下对每个问题执行多次，并对比这些执行结果来为智能体-技能对分配信用，同时对失败进行结构化诊断。在使用两个开源权重模型于HotpotQA和LoCoMo数据集上的实验中，GRSE在最接近的先前方法MASkills之上，在HotpotQA以及除开放域外的所有LoCoMo类别上均表现更优，其中在多跳问题上增益最大（+6.7和+3.2 F1）。其增益源于更可靠的检索基础。

    arXiv:2610.27227v1 Announce Type: new  Abstract: Cooperative language-model agents can improve without weight updates by revising reusable skills, but only if a shared team reward is traced to the responsible agent and skill. Existing methods credit the skills present in each trajectory, yet presence is not responsibility: a harmful skill active in successful runs still earns positive credit. We present \textbf{GRSE}, \textbf{G}roup-\textbf{R}elative \textbf{S}kill \textbf{E}volution, which assigns behavioral credit by comparison: inspired by GRPO, it executes each question several times under the same skill libraries and contrasts the executions to credit agent--skill pairs, with structured diagnosis of failures. On HotpotQA and LoCoMo with two open-weight actors, GRSE outperforms MASkills, the closest prior method, on HotpotQA and every LoCoMo category except open-domain, with the largest gains on multi-hop questions ($+6.7$ and $+3.2$ F1). Its gains rest on better-grounded retrieval
    
[^117]: Profile-DP：个性化大语言模型推理中用户可控档案发布的差分隐私

    Profile-DP: Differential Privacy for User-Controlled Profile Release in Personalized LLM Inference

    [https://arxiv.org/abs/2610.27213](https://arxiv.org/abs/2610.27213)

    该论文提出了 Profile-DP，一种策略感知的本地差分隐私框架，可在个性化大语言模型推理中可证明地保护用户不愿公开的档案信息，并将隐私保证与下游模型行为解耦，同时提供了 DP-SLOT 和 DP-BLOCK 两种满足该框架的实现机制。

    

    arXiv:2610.27213v1 公告类型：交叉（cross）。摘要：个性化大语言模型（LLM）日益依赖于基于用户档案的条件推理，将公开请求与丰富的用户特定事实信息相结合。这一范式暴露了用户档案本身以及用户明确的共享偏好。我们提出了 Profile-DP，一个专门为个性化 LLM 推理中用户档案发布而设计的、策略感知的本地差分隐私框架。Profile-DP 可证明地限制用户不愿公开的事实的泄露，同时将公开指令、档案模式和已授权事实视为公开的辅助信息。通过生成一个在答案生成前与公开指令相结合的隐私化档案，我们的框架将形式化的隐私保证与下游模型行为解耦。我们开发了两种满足 Profile-DP 的有限域机制：DP-SLOT 独立地对受保护属性进行扰动，而 DP-BLOCK 则利用公开的语义决策来协调受保护属性的发布。

    arXiv:2610.27213v1 Announce Type: cross  Abstract: Personalized large language models (LLMs) increasingly rely on profile-conditioned inference, pairing public requests with rich, user-specific facts. This paradigm exposes the user profile itself, together with the user's explicit sharing preferences. We introduce Profile-DP, a policy-aware local differential privacy framework designed specifically for user-profile release in personalized LLM inference. Profile-DP provably bounds the release of user-withheld facts while treating public instructions, profile schemas, and authorized facts as public side information. By generating a privatized profile that composes with the public instruction prior to answer generation, our framework decouples formal privacy guarantees from downstream model behavior. We develop two finite-domain mechanisms satisfying Profile-DP: DP-SLOT independently perturbs protected attributes, while DP-BLOCK uses public semantic decisions to coordinate their release a
    
[^118]: LoWAgent：面向Web智能体的低资源多语言基准测试

    LoWAgent: A Low-Resource Multilingual Benchmark for Web Agents

    [https://arxiv.org/abs/2610.27205](https://arxiv.org/abs/2610.27205)

    该论文提出了首个评估Web智能体在低资源语言（印地语、孟加拉语、乌尔都语、阿姆哈拉语）上性能的多语言基准LoWAgent，包含六个本地化网站和跨五种语言的3,000个任务实例，并发现闭源和开源模型在低资源语言上均相比英语出现明显的性能下降。

    

    Web智能体通常在英语界面上进行评估，其跨全球多语言网站运行的能力在很大程度上尚未被探索。这一差距对于低资源语言尤为显著，因为用于评估Web智能体的标准化资源仍然有限。为了填补这一空白，我们提出了低资源Web智能体基准测试，用于评估Web智能体在不同低资源语言（LRL）上相对于默认语言英语的表现。LoWAgent在英语以及四种低资源语言（印地语、孟加拉语、乌尔都语和阿姆哈拉语）上评估Web智能体。该基准包含六个完全本地化的网站，跨五种语言实例化，共产生3,000个任务实例。我们在所有五种语言和所有六个网站上评估了三个闭源模型和五个开源模型。我们的研究结果表明，与英语相比，闭源和开源模型在低资源语言上均出现了性能下降。

    arXiv:2610.27205v1 Announce Type: cross  Abstract: Web agents are typically evaluated on English-language interfaces, leaving their ability to operate across the world's multilingual websites largely unexplored. This gap is especially significant for low-resource languages, where standardized resources for evaluating web agents remain limited. To address this gap, we introduce Low-Resource Web Agent Benchmark (LoWAgent), a benchmark for evaluating web agents across different low-resource languages (LRLs) in comparison to the default, English. LoWAgent evaluates web agents across English and four LRLs: Hindi, Bengali, Urdu, and Amharic. The benchmark comprises of six fully localized websites, instantiated across five languages to yield 3,000 task instances. We evaluate three closed-source and five open-source models across all five languages on all six websites. Our findings show that both closed-source and open-source models experience performance degradation for LRLs compared to Engli
    
[^119]: 内在的文化：迈向对大语言模型文化推理的机制性理解

    The Culture Within: Towards Mechanistic Understanding of Cultural Reasoning in LLMs

    [https://arxiv.org/abs/2610.27202](https://arxiv.org/abs/2610.27202)

    该论文提出覆盖 75 个国家 9,261 个场景的 CultureLens 基准，通过机制性分析发现 LLM 在文化推理中默认采用英语国家规范、接受陈述性规范只看呈现方式而无论真假，且在模型内部阻断矛盾规范即可恢复正确回答，说明其失败源于内部信息处理而非知识缺失。

    

    大语言模型（LLM）越来越多地在文化多样的环境中运行，但现有的评估大多只是在输出层面衡量文化行为。我们提出了 CultureLens，这是一个基于参考标准的基准测试，同时也是对 LLM 如何使用文化信息的机制性研究。CultureLens 包含覆盖 75 个国家的 9,261 个场景，并系统地变化国家、价值观、真实规范与矛盾规范等证据。在 Llama-3.1-8B、Gemma-3-12B 和 Qwen-3.5-27B 上的实验表明，陈述一个矛盾的规范会使与文化一致的回答下降 32-47 个百分点，即使模型仅凭国家信息本可以正确作答。从机制上看，当未给出国家信息时，模型会默认采用英语国家的规范。在矛盾规范下的失败并非源于知识缺失，因为在模型内部阻断该规范后，大多数情况下可以恢复正确的回答。模型是否接受某个被陈述的规范取决于其呈现方式，而非其是否真实。

    arXiv:2610.27202v1 Announce Type: new  Abstract: Large Language Models (LLMs) increasingly operate across culturally diverse settings, yet existing evaluations largely measure cultural behavior at the output level. We introduce CultureLens, a reference-grounded benchmark and mechanistic study of how LLMs use cultural information. CultureLens contains 9,261 scenarios across 75 countries and systematically varies country, value, true-norm, and contradictory-norm evidence. Across Llama-3.1-8B, Gemma-3-12B, and Qwen-3.5-27B, stating a contradictory norm lowers aligned answers by 32-47 percentage points, even when the model answers correctly from the country alone. Mechanistically, models default to Anglophone norms when no country is given. The failure under a contradictory norm is not one of missing knowledge, as blocking the norm inside the model restores the correct answer in most cases. Models accept a stated norm according to how it is presented and not whether it is true, which leave
    
[^120]: 面向多任务语音大语言模型的偏好感知帕累托调控

    Preference-Aware Pareto Steering for Multi-Task Speech LLMs

    [https://arxiv.org/abs/2610.27200](https://arxiv.org/abs/2610.27200)

    提出EA-SURF方法，通过在训练中自适应调整标量化权重，使多任务语音大语言模型的任务间权衡符合期望的部署偏好，并从理论上证明了该方法能够收敛到目标偏好。

    

    语音大语言模型（LLM）可以执行多种任务，包括自动语音识别（ASR）、时间戳预测（TS）和语音翻译（ST）。语音大语言模型的联合训练会产生任务间的权衡，然而使用固定损失权重的标准训练通常无法根据部署偏好来控制所产生的权衡。为解决这一问题，我们提出了EA-SURF（沿帕累托前沿的端点锚定偏好感知均匀采样），这是一种先验方法，通过调整标量化权重，使产生的任务权衡能够反映期望的偏好。EA-SURF保留了标准的加权损失训练流程，跟踪当前模型相对于各任务特定参考的性能，并随着训练的进行自适应地调整标量化权重。我们从理论上证明了EA-SURF能够匹配期望的偏好，在可检验的正则性条件下，更新在目标点处趋于平稳。

    arXiv:2610.27200v1 Announce Type: new  Abstract: Speech large language models (LLMs) perform multiple tasks, including automatic speech recognition (ASR), timestamp prediction (TS), and speech translation (ST). Joint training of speech LLMs creates task trade-offs, yet standard training with fixed loss weights does not generally control the resulting trade-off according to deployment preferences. To address this, we propose Endpoint-Anchored Preference-Aware Sampling Uniformly along the Pareto Front (EA-SURF), an a priori method that adapts scalarization weights so that the resulting task trade-off reflects the desired preference. EA-SURF retains the standard weighted-loss training pipeline, tracking the current model's performance relative to each task-specific reference and adapting the scalarization weights as training proceeds. We theoretically show that EA-SURF can match a desired preference, with the update becoming stationary at the target under checkable regularity conditions. 
    
[^121]: 论任务组合下的脆弱性：语言模型中的隐式预算分配

    On the Fragility of Tasks under Composition: Implicit Budgeting in Language Models

    [https://arxiv.org/abs/2610.27190](https://arxiv.org/abs/2610.27190)

    语言模型存在一种隐式思考预算：在组合多个任务时，每个任务分配到的 token 数远少于单独处理时，导致即使模型能独立可靠地完成每个任务，组合后错误率仍显著上升，且需要长推理链的任务在组合下尤为脆弱。

    

    现代语言模型越来越多地被要求在单个提示中处理多个任务。这需要模型正确解决每个任务，同时将各任务的结果整合成一个连贯的回复。本文识别了这一过程中的一个根本性局限：当任务被组合在一起时，模型会犯下多得多的错误，即使它们能够可靠地独立完成每一个组成任务。这种性能退化无法完全用独立错误的累积来解释。通过对开放权重模型以及前沿模型 Claude Opus 5 的研究，我们的发现指向了这一现象背后的一种隐式思考预算。具体而言，在任务组合的情况下，模型在每个任务上使用的 token 数量远少于单独处理时，即使完整序列仍远在上下文窗口范围之内。因此，那些单独求解时需要更长推理链条的任务在组合情况下会更加“脆弱”，即模型……

    arXiv:2610.27190v1 Announce Type: cross  Abstract: Modern language models are increasingly asked to handle several tasks within a single prompt. Doing so requires solving each task correctly while integrating the resulting answers into a coherent response. In this paper, we identify a fundamental limitation in this process, where models make substantially more errors when tasks are composed, even when they can solve every constituent task reliably in isolation. This degradation cannot be fully explained by the accumulation of independent errors. Across both open-weight models and the frontier model Claude Opus 5, our findings instead point to an implicit thinking budget underlying this effect. In particular, under composition, models use substantially fewer tokens per task than in isolation, even when the full sequence remains well within the context window. Tasks that require longer solution traces in isolation are consequently more $\textit{fragile}$ under composition, i.e., models m
    
[^122]: SITO：对一款商用文本安全API进行筛检，发现其在英语、乌克兰语和苏尔日克语（俄乌混合语）中漏判的危害与误报

    SITO: Sifting a Commercial Text-Safety API for Missed Harm and False Flags in English, Ukrainian and Surzhyk

    [https://arxiv.org/abs/2610.27178](https://arxiv.org/abs/2610.27178)

    该论文构建了包含2,332个英语、乌克兰语及混合语合成案例的SITO基准，独立评估了一款商用文本安全API在六类危害场景中的漏判与误报表现，填补了英语之外语言环境下此类系统缺乏独立验证证据的空白。

    

    托管青少年用户的平台使用商用文本安全API来筛查消息，但关于此类系统漏判有害内容或误报无害文本的频率，独立的证据十分稀缺，尤其是在英语之外的语言中，而且本文所研究的这一系统的官方文档也未量化其误报率。我们评估了一款此类API，即以发布标识符 tuteliq-detect-v5.20261011 部署的 Tuteliq Detect（而非单独发布的 v6 模型），涵盖六类危害场景：授权推送支付诈骗、“钱骡”招募、社会工程攻击、浪漫诈骗、赌博危害和欺凌。我们构建了SITO（Synthetic Independent Text-safety benchmark with Overflagging tests，含过报测试的合成独立文本安全基准；sito在乌克兰语中意为“筛子”），包含2,332个英语、乌克兰语及乌克兰语混合语（苏尔日克语）的合成案例，其中每个有害案例都配有对应的困难无害案例：同主题消息、最小编辑对比、警告性内容以及回顾性叙述。在英语文本上，该系统标记了84.9%至94.1%……（摘要在此处截断）

    arXiv:2610.27178v1 Announce Type: new  Abstract: Platforms that host young users screen messages with commercial text-safety APIs, but independent evidence on how often such systems miss harm or flag harmless text is scarce, especially outside English, and the documentation of the one studied here leaves its false-positive rate unquantified. We evaluated one such API, Tuteliq Detect as deployed under the release identifier tuteliq-detect-v5.20261011 (not the separately published v6 model), on six harm categories: authorised push-payment fraud, money-mule recruitment, social engineering, romance scams, gambling harm and bullying. We built SITO (Synthetic Independent Text-safety benchmark with Overflagging tests; sito is Ukrainian for 'sieve'), 2,332 synthetic cases in English, Ukrainian and mixed Ukrainian, with each harmful case matched by hard harmless ones: same-topic messages, minimal-edit contrasts, warnings and retrospective accounts. On English text the system flagged 84.9-94.1% 
    
[^123]: GRID：将健康类网红话语建模为论辩网络以进行可信度评估

    GRID: Modeling Health Influencer Discourse as Argument Networks for Credibility Assessment

    [https://arxiv.org/abs/2610.27149](https://arxiv.org/abs/2610.27149)

    提出GRID框架，将健康类网红视频建模为相互关联的论辩网络，结合科学证据进行联合可信度推断，证明考虑更广泛的话语背景能显著提升健康内容验证的效果。

    

    社交媒体上与健康相关的讨论日益受到网红的影响，这些网红对同一问题提供多样且常常相互竞争的观点。然而，此类内容通常被孤立地评估，而没有考虑其所处的更广泛的话语背景。我们提出了GRID（Grounded Reasoning over Interconnected Discourse，互联话语上的有据推理），这是一个通过对周围话语进行集体推理来评估网红内容的框架。GRID将每个视频表示为一个论辩图，将各论点连接到其中心信息（即要点），在视频之间关联相关的论点和要点，将论点与科学证据相印证，并在由此形成的网络上联合推断可信度。我们在一个端到端的健康相关内容验证任务上对GRID进行了评估，该任务使用569个基于专家评估的网红视频。结果表明，考虑更广泛的话语背景能够改进网红内容的验证效果。

    arXiv:2610.27149v1 Announce Type: new  Abstract: Health-related discussions on social media are increasingly shaped by influencers who offer diverse, and often competing perspectives on the same issue. Yet, such content is typically evaluated in isolation, without considering the broader discourse in which it appears. We introduce GRID (Grounded Reasoning over Interconnected Discourse), a framework for evaluating influencer content through collective inference over the surrounding discourse. GRID represents each video as an argumentation graph connecting claims to its central message, or takeaway, links related claims and takeaways across videos, grounds claims in scientific evidence, and jointly infers credibility over the resulting network. We evaluate GRID on an end-to-end health-related content validation task using 569 influencer videos grounded in expert assessments. Our results show that accounting for the broader discourse improves influencer content validation. Finally, we use
    
[^124]: ICLHerd：基于Herding的大语言模型数据选择方法，以上下文学习作为代理

    ICLHerd: Herding-Based Data Selection for LLMs with In-Context Learning as a Proxy

    [https://arxiv.org/abs/2610.27142](https://arxiv.org/abs/2610.27142)

    ICLHerd提出了一种基于von Mises-Fisher核的数据选择框架，通过平衡模式寻求与模式平均来克服传统最大相似度方法的聚类偏差，并利用上下文学习作为验证代理实现高效的超参数搜索，在大语言模型训练数据选择上优于现有算法。

    

    数据选择在现代大语言模型训练中扮演着核心角色，因为训练数据的质量强烈影响下游性能。一种常见的方法是策划一个编码了所需技能（例如推理或多语言能力）的开发集，并选择与这些技能相似的数据。现有方法使用模型隐藏状态或梯度上的余弦距离来计算相似度。然而，我们观察到这种最大相似度策略会引发模式寻求偏差：所选数据紧密聚集在开发样本周围。为了解决这个问题，我们提出了ICLHerd，这是一个数据选择框架，利用von Mises-Fisher核来平衡模式寻求与模式平均。我们进一步引入上下文学习作为验证代理，从而能够在平衡权衡上进行高效的超参数搜索。结果表明，ICLHerd优于现有的大语言模型数据选择算法，凸显了……

    arXiv:2610.27142v1 Announce Type: cross  Abstract: Data selection plays a central role in modern large language model training, as the quality of training data strongly influences downstream performance. A common approach is to curate a development set that encodes the desired skills (e.g., reasoning or multilinguality) and select data that are similar to them. Existing methods compute similarity using cosine distance over either the model's hidden states or gradients. However, we observe that this max-similarity strategy induces a mode-seeking bias: the selected data clusters tightly around the development samples. To address this, we propose ICLHerd, a data selection framework that leverages the von Mises-Fisher kernel to balance mode-seeking with mode-averaging. We further introduce in-context learning as a validation proxy, enabling efficient hyperparameter search over the balance trade-off. Results show that ICLHerd outperforms existing LLM data selection algorithms, highlighting 
    
[^125]: 面向限定条件敏感的检索增强问答的联合约束聚合方法

    Joint Constraint Aggregation for Qualifier-Sensitive Retrieval-Augmented QA

    [https://arxiv.org/abs/2610.27114](https://arxiv.org/abs/2610.27114)

    提出联合约束聚合方法，通过基于检索证据将候选答案与问题派生的统一要求进行联合比对，解决检索增强问答中因检索偏斜导致错误答案被选中的决策问题，并显著提升整体准确率。

    

    检索增强问答即使当正确答案出现在检索到的证据中时也可能失败。这种情况可能发生在：某个竞争性答案只满足问题要求中的一部分（而非全部），却在检索到的文档中被频繁提及，从而使其看起来像是得到最多支持的答案。为了检验这种检索偏斜是否真的会影响模型决策，我们直接对其进行操控：增加偏向竞争性答案的段落的数量，会使模型更倾向于选择该答案。这表明存在一个仅凭候选答案层面的支持度无法解决的决策问题。因此，我们提出了联合约束聚合（JCA），它利用检索到的证据，将候选答案与由问题派生的相同要求进行对比。在三种评估设置和三种指令微调骨干模型上，JCA 提升了整体准确率，并且当检索偏向竞争性答案时，其优势会进一步扩大。这些结果……

    arXiv:2610.27114v1 Announce Type: cross  Abstract: Retrieval-augmented QA can fail even when the correct answer is in the retrieved evidence. This can happen when a competing answer satisfies some, but not all, of the question's requirements and is mentioned frequently in the retrieved documents, which can make it appear to be the best-supported answer. To test whether this retrieval skew actually affects the decision, we manipulate it directly: increasing the prevalence of passages that favor the competing answer makes the model more likely to select it. This points to a decision problem that candidate-level support alone cannot resolve. We therefore introduce Joint Constraint Aggregation (JCA), which compares candidates against the same question-derived requirements using the retrieved evidence. Across three evaluation settings and three instruction-tuned backbones, JCA improves overall accuracy, and its advantage grows when retrieval favors a competing answer. Together, these result
    
[^126]: 超越双声道：面向说话人锚定的全双工对话恢复

    Beyond Two Tracks: Speaker-Anchored Full-Duplex Dialogue Recovery

    [https://arxiv.org/abs/2610.27109](https://arxiv.org/abs/2610.27109)

    该论文提出在扩散生成之前进行说话人锚定路由，用一个仅184万参数的轻量模块将分离出的声流匹配并排序到固定的注册身份，把全双工对话重建中的错误声道时间从43.9%降至4.1%，且无需重新训练冻结的生成器。

    

    口语交互正从轮流发言转向全双工对话，即智能体在说话的同时进行聆听。学习这种协同配合需要带有持久说话人分离声道的自然对话数据，而这一数据需求在大规模场景下仍难以满足。生成式恢复提供了一条从普通录音出发的路径，但仅恢复出两个声音还不够：随着对话的展开，它们的声道归属可能发生漂移。我们通过在扩散之前的说话人锚定路由来弥补这一缺失环节：一个仅有184万参数的模块将分离器输出的粗略声流与固定的注册身份进行匹配并排序，随后由冻结的生成器重建语音，全程无需重新训练生成器，也不会改变重建后的音频。在25分钟的远程对话中，路由将错误声道时间从43.9%降至4.1%，低于基于匹配注册的波形重识别方法的6.2%，其最大优势出现在其中一名参与者……（摘要截断）

    arXiv:2610.27109v1 Announce Type: cross  Abstract: Spoken interaction is moving from alternating turns to full-duplex conversation, where an agent listens while speaking. Learning this coordination requires natural dialogue with persistent speaker-resolved tracks, a data requirement that remains difficult to satisfy at scale. Generative recovery offers a route from ordinary recordings, but recovering two voices is not enough: their track assignments can drift as the conversation unfolds. We address this missing link with speaker-anchored routing before diffusion: a 1.84M-parameter module matches the separator's coarse streams to fixed enrollment identities and orders them before a frozen generator reconstructs the speech, without retraining the generator or altering the reconstructed audio. On 25-minute remote conversations, routing cuts wrong-track time from 43.9% to 4.1%, below the 6.2% of matched-enrollment waveform re-identification, with its largest advantage when one participant 
    
[^127]: Omni-Woodpecker：通过证据-先验再平衡修正全模态大语言模型的幻觉

    Omni-Woodpecker: Hallucination Correction via Evidence-Prior Rebalancing for Omni LLMs

    [https://arxiv.org/abs/2610.27103](https://arxiv.org/abs/2610.27103)

    提出了无需训练的 Omni-Woodpecker 框架，通过识别跨模态冲突、正交分解分离接地证据与学习先验，并在解码前自适应地再平衡融合表示，从而有效修正全模态大语言模型的跨模态幻觉。

    

    近期的全模态大语言模型（OLLMs）在音视频联合推理方面展现出卓越的能力，但跨模态幻觉仍然是一个重大挑战。现有的缓解方法主要在输出层面进行操作，忽视了融合表示中接地证据与学习先验之间的纠缠问题。为解决这一问题，我们提出了 Omni-Woodpecker（OWP），这是一个无需训练的框架，将幻觉缓解从输出层面的操作转变为基于冲突条件的表示再平衡。具体而言，OWP 首先从互补的模态特定视角识别跨模态冲突；然后采用正交分解将隐藏状态中的接地证据与竞争性先验分离开来；最后，OWP 在解码前通过抑制误导性先验并恢复被忽视的证据，自适应地再平衡融合表示。大量实验证明了该方法的有效性和强大的跨……

    arXiv:2610.27103v1 Announce Type: new  Abstract: Recent Omni LLMs (OLLMs) have demonstrated remarkable capabilities in joint audio-visual reasoning, yet cross-modal hallucinations remain a major challenge. Existing mitigation methods mainly operate at the output level, overlooking the entanglement of grounding evidence and learned priors within fused representations. To address this issue, we propose Omni-Woodpecker (OWP), a training-free framework that shifts hallucination mitigation from output-level manipulation to conflict-conditioned representation rebalancing. Specifically, OWP first identifies cross-modal conflicts from complementary modality-specific views. It then employs orthogonal decomposition to separate grounding evidence from competing priors within the hidden states. Finally, OWP adaptively rebalances fused representations by suppressing misleading priors and restoring ignored evidence before decoding. Extensive experiments demonstrate the effectiveness and strong cross
    
[^128]: SciFigBench：科学方法图理解基准

    SciFigBench: A Benchmark for Scientific Method Figure Understanding

    [https://arxiv.org/abs/2610.27087](https://arxiv.org/abs/2610.27087)

    该论文提出了SciFigBench基准，通过从真实论文中精选4,352张科学方法图并构建40,670个问答对，系统评估视觉语言模型在密集内容感知、视觉关系理解和信息流追踪三项核心能力上的表现。

    

    科学方法图通常为研究人员提供了理解论文方法和核心贡献的清晰视觉入口，然而其密集的布局和复杂的信息流使得视觉语言模型（VLMs）难以解读。现有基准主要聚焦于粗粒度任务，例如基于上下文的检索或整体性理解，忽视了复杂方法图中细粒度的内容以及模块之间的关系。为解决这些被忽视的方面，我们提出了科学图基准（SciFigBench），用于评估科学方法图理解所需的核心能力。具体而言，该基准针对三种能力：密集内容感知、视觉关系理解和信息流追踪。为此，我们从真实研究论文中精选了4,352张科学方法图，并构建了40,670个问答对来具体化这些能力。

    arXiv:2610.27087v1 Announce Type: cross  Abstract: Scientific method figures often provide researchers with a clear visual entry point into a paper's methodology and core contributions, yet their dense layouts and complex information flows are difficult for Vision Language Models (VLMs) to interpret. Existing benchmarks predominantly focus on coarse-grained tasks, such as context-based retrieval or holistic comprehension, overlooking the fine-grained content and inter-module relations within complex methodological diagrams. To address these overlooked aspects, we introduce Scientific Figure Benchmark (SciFigBench), a benchmark for evaluating core capabilities required for scientific method figure understanding. Specifically, it targets three capabilities: dense content perception, visual relation comprehension, and information-flow tracing. For this purpose, we curate 4,352 scientific method figures from real-world research papers and construct 40,670 QA pairs to operationalize these c
    
[^129]: ClockMark：面向大语言模型的自适应与宿主相关水印技术

    ClockMark: Adaptive and Host-Dependent Watermarking for Large Language Models

    [https://arxiv.org/abs/2610.27079](https://arxiv.org/abs/2610.27079)

    提出了ClockMark水印框架，首次基于“自适应嵌入”（保留先前采样决策状态）与“宿主相关嵌入”（利用当前LLM分布约束词元选择和状态更新）两大原则，在保持输出分布的同时实现对文本编辑的鲁棒水印嵌入。

    

    大语言模型（LLM）水印技术在生成过程中不断演化的下一词元分布上进行操作。在流式生成中，每个采样得到的词元都会成为文本的一部分，并影响后续的分布。因此，嵌入过程是在不断变化的宿主上进行的，这与经典水印技术不同——经典水印技术是将嵌入适配到已存在的宿主信号上。为了适应这一差异，我们基于两个原则为LLM水印构建了一种新颖的嵌入框架。我们首先论证嵌入应当是自适应的，即保留先前采样决策的状态，以更好地指导后续的选择。我们进一步论证嵌入应当是宿主相关的，即利用当前的LLM分布来同时约束词元的选择和状态的更新。基于这些原则，并遵循分布保持和抗文本编辑鲁棒性这两项要求，我们提出了ClockMark。它将自适应与宿主相关的嵌入与消息相结合……

    arXiv:2610.27079v1 Announce Type: cross  Abstract: Large language model (LLM) watermarking operates on the next-token distribution that evolves during generation. In streaming generation, each sampled token becomes part of the text and affects subsequent distributions. Embedding therefore proceeds with a changing host, unlike classical watermarking, which adapts embedding to an existing host signal. To accommodate this difference, we formulate a novel embedding framework for LLM watermarking based on two principles. We first argue that embedding should be adaptive, retaining the state of previous sampling decisions to better guide subsequent choices. We further argue that embedding should be host-dependent, using the current LLM distribution to constrain both token selection and state updates. Guided by these principles and the two requirements of distribution preservation and robustness to text editing, we propose ClockMark. It combines adaptive and host-dependent embedding with messa
    
[^130]: 面向本体无关电子健康记录建模的离散神经中介语言

    Discrete Neural Interlingua for Ontology-Agnostic EHR Modeling

    [https://arxiv.org/abs/2610.27047](https://arxiv.org/abs/2610.27047)

    本文提出MEDIATOR——一种本体无关的离散分词器，通过将任意文本形式表达的临床概念映射到共享的向量量化码本，将EHR数据协调转化为开放词表分词问题，从而打破不同医疗系统间的编码壁垒。

    

    电子健康记录（EHR）基础模型有望实现跨医疗系统的可扩展临床预测，但其部署受到分词瓶颈的制约。同一临床概念可能以不同的本体编码、本地标识符、同义词或自由文本描述的形式出现，而标准的EHR分词器会将它们视为互不相关的符号。使用封闭词表分词器在MIMIC-III上训练的模型，仅能使用eICU中13%的事件。我们提出了MEDIATOR，这是一种离散的、本体无关的分词器，它将任何以文本形式表达的临床概念映射到一个共享的向量量化码本中，从而将EHR数据协调统一转化为开放词表分词问题，无需人工映射或固定词表。通过同义词监督进行训练，并借助丰富的概念描述、编码共现信息和关系图进行引导，MEDIATOR能够合并等价表达，同时在具有医学意义的离散空间中组织不同概念。用作分词接口时（摘要在此处截断）

    arXiv:2610.27047v1 Announce Type: cross  Abstract: EHR foundation models promise scalable clinical prediction across health systems, yet their deployment is constrained by a tokenization bottleneck. The same clinical concept can appear as different ontology codes, local identifiers, synonyms, or free-text descriptions, which standard EHR tokenizers treat as unrelated symbols. A model trained on MIMIC-III with a closed-vocabulary tokenizer can use only 13% of eICU events. We introduce MEDIATOR, a discrete ontology-agnostic tokenizer that maps any clinical concept expressed as text into a shared vector-quantized codebook, turning EHR harmonization into open-vocabulary tokenization without manual mappings or a fixed vocabulary. Trained with synonym supervision and guided by enriched concept descriptions, code co-occurrence, and relation graphs, MEDIATOR collapses equivalent expressions while organizing distinct concepts in a medically meaningful discrete space. Used as the token interface
    
[^131]: ReuF2F：基于形式化本科研究问题的编码智能体基准测试

    ReuF2F: Benchmarking Coding Agents on Formalized Undergraduate Research Problems

    [https://arxiv.org/abs/2610.27046](https://arxiv.org/abs/2610.27046)

    该论文提出了 ReuF2F，一个基于 Lean 4 的基准测试，包含 244 个源自本科生研究经验计划（REU）的形式化开放研究问题，用以更可靠地评估编码智能体解决开放数学问题的能力。

    

    随着人工智能系统开始解决开放性研究问题，可靠的验证变得日益重要。形式化数学为数学推理提供了一个可信的框架，使智能体在进行数学探索的同时能够生成机器可检验的证明。现有的形式化数学基准测试通常集中于已知结果或少数著名的开放问题，一旦这些问题的解被发现并广泛传播，就难以再对其进行可靠评估。此外，现有的评估协议也并不总是与现代编码智能体的工作方式相匹配。因此，我们提出了 ReuF2F，这是一个基于 Lean 4 的基准测试，包含从本科生研究经验计划（Research Experiences for Undergraduates, REU）中提取的 244 个形式化开放问题，涵盖了广泛数学子领域中的入门级开放研究问题。我们预计这些较少受到关注的开放问题——它们往往吸引的注意力较少，并可能长时间未被解决——将更……（摘要在此处被截断）

    arXiv:2610.27046v1 Announce Type: new  Abstract: As AI systems begin to resolve open research problems, reliable verification becomes increasingly important. Formal mathematics provides a trusted framework for mathematical reasoning, enabling agents to produce machine-checkable proofs while conducting mathematical exploration. Existing formal mathematics benchmarks often focus on known results or a small set of prominent open problems, which can become difficult to evaluate reliably once solutions are discovered and widely disseminated. Existing evaluation protocols are also not always aligned with how modern coding agents operate. We therefore introduce ReuF2F, a Lean 4 benchmark of 244 formalized open problems drawn from Research Experiences for Undergraduates (REU), spanning entry-level open research problems across a broad range of mathematical subfields. We expect these less prominent open problems, which tend to attract less attention and may remain unsolved for longer, to be bet
    
[^132]: 大语言模型管线的跨语言可复用性：金融信息抽取、多语言推理与语义筛选

    Cross-Lingual Reusability of LLM Pipelines: Financial Extraction, Multilingual Reasoning, and Semantic Screening

    [https://arxiv.org/abs/2610.27043](https://arxiv.org/abs/2610.27043)

    该研究通过支票信息抽取、多语言问答和金融文档分析三项任务，验证了在英语中开发和验证的大语言模型管线可以直接跨语言复用，准确性在各语言间保持高度稳定。

    

    金融机构日益在多语言市场中部署大语言模型（LLM），这引出一个实际问题：在英语中开发和验证的管线能否跨语言复用，还是必须针对每个地区重新构建？我们通过三个互补任务，从三个维度——准确性、推理努力迁移和行为稳定性——检验跨语言可复用性。任务1（支票信息抽取）和任务2（跨九种语言、九种模型-推理努力配置、固定证据的Mintaka问答）测试准确性和努力设置能否迁移；任务3（FinanceBench试点，针对四种语言的SEC 10-K文件进行50项测试）验证这些模式是否可推广至金融内容。准确性可复用性得到广泛证实：Gemini 3.5 Flash在不同提示语言下达到95.2-97.6%的抽取准确率，FinanceBench各语言平均值跨度为5.8个百分点（77.1-82.9%），尽管语言难度排名（摘要在此处截断）

    arXiv:2610.27043v1 Announce Type: cross  Abstract: Financial institutions increasingly deploy large language models (LLMs) in multilingual markets, raising a practical question: can pipelines developed and validated in English be reused across languages, or must they be rebuilt per locale? We test cross-lingual reusability along three dimensions--accuracy, reasoning-effort transfer, and behavioural stability--through three complementary tasks. Task 1 (cheque extraction) and Task 2 (Mintaka QA with pinned evidence across nine languages and nine model-effort configurations) test whether accuracy and effort settings transfer; Task 3 (FinanceBench pilot, 50 items over SEC 10-K filings in four languages) validates whether these patterns generalise to financial content. Accuracy reusability is broadly confirmed: Gemini 3.5 Flash achieves 95.2-97.6% extraction accuracy across prompt languages, and FinanceBench per-language means span 5.8pp (77.1-82.9%), though the language-difficulty ranking 
    
[^133]: 流畅但易错：大型语言模型中幻觉与事实可靠性的基准测试

    Fluent but Fallible: Benchmarking Hallucination and Factual Reliability in Large Language Models

    [https://arxiv.org/abs/2610.27034](https://arxiv.org/abs/2610.27034)

    本研究通过70个覆盖四个复杂度级别的学术提示，对十个主流大语言模型的幻觉和事实可靠性进行了系统的基准对比评估，揭示了尽管模型语言流畅但仍存在事实错误和响应偏差的可靠性风险。

    

    大型语言模型（LLM）通过其在推理、文本生成和知识检索方面的能力，显著推动了自然语言处理和人工智能应用的发展。然而，LLM仍然容易受到两个重要挑战的影响：幻觉，即生成事实上错误或虚构的内容；以及响应偏差，即系统性地偏离真实答案。这些局限性引发了学术界和研究环境中对可靠性和可信度的严重担忧。本研究采用多种指标，对十个先进的LLM进行了全面的对比分析，包括ChatGPT、Gemini、Claude 3.5 Sonnet、LLaMA 3、Mistral、Grok、Perplexity、DeepSeek V3、Falcon和Gemma。基准测试由70个精心设计的学术提示组成，涵盖四个复杂度级别：基础、解释、高级和技巧性，于2025年2月针对特定模型版本进行评估。

    arXiv:2610.27034v1 Announce Type: new  Abstract: Large Language Models (LLMs) have significantly advanced natural language processing and artificial intelligence applications through their capabilities in reasoning, text generation and knowledge retrieval. However, LLMs are still vulnerable to two important challenges: hallucination, generating factually incorrect or fabricated content, and response deviation, meaning systematic deviations from ground-truth responses. These limitations raise serious concerns regarding reliability and trustworthiness in academic and research environments. This study offers a thorough comparative analysis using a variety of metrics of ten advanced LLMs ChatGPT, Gemini, Claude~3.5~Sonnet, LLaMA~3, Mistral, Grok, Perplexity, DeepSeek~V3, Falcon, and Gemma based on a benchmark consisting of 70 carefully designed academic prompts across four complexity levels: Basic, Explanation, Advanced, and Tricky, evaluated in February 2025 on specific model versions. Ea
    
[^134]: 通过偏好优化教会AI智能体策略性地玩扑克

    Teaching AI Agent to Play Poker Strategically Through Preference Optimization

    [https://arxiv.org/abs/2610.27028](https://arxiv.org/abs/2610.27028)

    提出SPOKE方法，通过融合教师相对价值惩罚、策略锚定、风险约束和结构化推理监督的偏好优化技术，使小型AI智能体在隐藏信息的扑克博弈中学会策略性行为，并显著降低可利用性。

    

    一个扑克智能体可以做出合法的行动，却仍然是策略上糟糕的行动。我们探究紧凑型AI智能体能否在隐藏信息条件下展现出策略性行为，以及偏好优化能否超越静态和零样本策略。我们首先评估了一个静态基线和三个冻结的语言模型：SmallThinker、Qwen3和Gemma-3，它们在精确博弈评估和六人局评估中的表现差异很大。因此，我们提出了带KL锚定均衡的策略偏好优化方法（SPOKE），该方法结合了教师相对价值惩罚、策略锚定、风险约束和结构化推理监督，并将其与SFT、DPO、GRPO和GSPO进行比较。在SmallThinker-4B-A0.6B策略的三次运行中，SPOKE在Kuhn扑克中取得了最低的平均可利用性（0.213），在Leduc扑克中为（1.081），并取得了最高的六人局top-1一致率（0.775），同时保持了与SFT相当的教师KL值（0.522 vs 0.523）。

    arXiv:2610.27028v1 Announce Type: new  Abstract: A poker agent can make a legal move and still make a strategically poor one. We ask whether compact AI agents can exhibit strategic behavior under hidden information and whether preference optimization can improve beyond static and zero-shot policies. We first evaluate a static baseline and three frozen language models: SmallThinker, Qwen3, and Gemma-3 whose performance varies substantially between exact-game and six-player evaluation. We therefore propose Strategic Preference Optimization with KL-anchored Equilibrium (SPOKE), combining teacher-relative value penalties, policy anchoring, risk constraints, and structured rationale supervision, and compare it with SFT, DPO, GRPO, and GSPO. Across three runs of a SmallThinker-4B-A0.6B policy, SPOKE achieves the lowest mean exploitability in Kuhn (0.213) and Leduc (1.081) and the highest six-max top-1 agreement (0.775), while maintaining teacher KL comparable to SFT (0.522 vs 0.523). These r
    
[^135]: 航空事故叙述文本的分解式关系抽取：关系稀疏性下基于Transformer的安全知识发现评估

    Decomposed Relation Extraction from Aviation Accident Narratives: Evaluating Transformer-Based Safety Knowledge Discovery under Relational Sparsity

    [https://arxiv.org/abs/2610.27013](https://arxiv.org/abs/2610.27013)

    该研究提出将概念识别与关系分类相分离的分解式框架，并在NTSB事故叙述的大规模稀疏基准上系统评估发现，SVM与RoBERTa模型在真实关系稀疏场景下性能急剧下降（Macro-F1从约0.48降至约0.24-0.27），揭示了现有关系抽取方法在航空安全知识发现中面临的严重挑战。

    

    从航空事故叙述文本中进行关系抽取极具挑战性，因为真正的安全关系在所有可能的概念对中非常稀疏。本研究评估了一种将关系分类与概念识别相分离的分解式框架。研究构建了一个包含300篇美国国家运输安全委员会（NTSB）叙述文本的基准数据集，其中含有2,524个独特的概念-文本节点，并将其扩展为21,038个有序候选对，其中仅有248个（1.18%）代表RESULTS_IN或CONTRIBUTES_TO关系。研究者使用一个独立的图标注资源（结合困难负样本采样）开发了TF-IDF线性支持向量机（SVM）和加权RoBERTa分类器，并在密封评估前冻结了模型参数。开发阶段的Macro-F1分数分别为0.4738（SVM）和0.4809（RoBERTa），而在穷举评估中下降至0.2741和0.2420，马修斯相关系数分别为0.0128和0.0197。假阳性占主导地位：SVM恢复了96个参考关系，但……

    arXiv:2610.27013v1 Announce Type: new  Abstract: Relation extraction from aviation accident narratives is challenging because genuine safety relations are sparse among possible concept pairs. This study evaluates a decomposed framework separating relation classification from concept identification. A benchmark of 300 National Transportation Safety Board (NTSB) narratives containing 2,524 unique concept-text nodes was expanded into 21,038 ordered candidate pairs, with only 248 (1.18%) representing RESULTS_IN or CONTRIBUTES_TO relations. A TF-IDF Linear Support Vector Machine (SVM) and weighted RoBERTa classifier were developed using an independent graph-annotated resource with hard-negative sampling and frozen before sealed evaluation. Development Macro-F1 scores of 0.4738 (SVM) and 0.4809 (RoBERTa) declined to 0.2741 and 0.2420 under exhaustive evaluation, with Matthews correlation coefficients of 0.0128 and 0.0197. False positives dominated: the SVM recovered 96 reference relations bu
    
[^136]: 跨方言与适应场景下的藏语语音识别评估：一种多粒度分析

    Evaluating Tibetan Speech Recognition Across Dialects and Adaptation Regimes: A Multi-Granularity Analysis

    [https://arxiv.org/abs/2610.27002](https://arxiv.org/abs/2610.27002)

    本文提出整合字符、音节和分词三种错误率的多粒度评估框架，对藏语语音识别在跨方言和LoRA适应场景下的表现进行系统评估，发现较低字符错误率并不必然意味着音节或分词层面的更好表现，且指标间相关性会在适应失败时减弱。

    

    由于藏文书写系统不标注词边界，藏语语音识别通常采用字符错误率或音节错误率进行评估。这些指标可能会对系统给出不同的排序，并且对适应失败的反应也不尽相同。我们将字符错误率、音节错误率和分词错误率整合到同一个评估框架中。我们使用基于词典的、神经网络的以及基于语言模型的分词方法，并将其切分边界与专家标注进行对比。我们在拉萨藏语上评估了五种ASR架构，然后在五个额外的语料库子集上测试零样本迁移和目标语料库适应。我们还通过改变LoRA秩和训练数据规模来考察适应失败的情况。结果表明，较低的字符错误并不总能带来较低的音节错误率或分词错误率。这种排序差异在各语料库中持续存在。在某些失败的适应设置中，各指标之间的相关性也会减弱。较高的LoRA秩……

    arXiv:2610.27002v1 Announce Type: new  Abstract: Tibetan speech recognition is usually evaluated with character or syllable error rates because the writing system does not mark word boundaries. These metrics can rank systems differently and respond differently to adaptation failures. We combine character error rate, syllable error rate, and segmented word error rate in one evaluation framework. We use dictionary-based, neural, and language-model-based segmentation and compare their boundaries with expert annotations. We evaluate five ASR architectures on Lhasa Tibetan, then test zero-shot transfer and target-corpus adaptation on five additional corpus subsets. We also vary LoRA rank and training-data size to examine adaptation failure. The results show that a lower character error does not always lead to a lower syllable or segmented-word error. These ranking differences persist across corpora. Metric correlations also weaken in some unsuccessful adaptation settings. Higher LoRA ranks 
    
[^137]: 如何使用激活引导让你的ASR系统获得最佳且最公平的性能

    How to get the best and fairest performance out of your ASR system using activation steering

    [https://arxiv.org/abs/2610.26994](https://arxiv.org/abs/2610.26994)

    该论文提出用监督优化法推导激活引导向量，可同时提升自动语音识别系统的整体性能与性别、年龄、方言等维度上的公平性，且发现使用同一数据集中不同说话人或声学相似说话人的数据训练引导向量即可获得显著效果。

    

    针对自动语音识别（ASR）系统的激活引导是一种在提升ASR性能和人口群体公平性方面尚未被充分探索的技术。我们研究了两种形式的激活引导：成对激活减除法和监督优化法。我们发现前者对改善ASR性能完全无效，而后者在推导引导向量（SVs）方面非常有效，这些引导向量能够同时提升整体ASR性能以及在性别、年龄和方言等多样化维度上的人口群体公平性。我们还探讨了计算或训练SVs需要哪些类型的数据——从最个人化到最不个人化的数据。首先，我们展示了使用同一数据集中来自不同群体的说话人训练SVs能够带来显著的性能提升。此外，我们发现使用声学特征相似的说话人训练的向量，其带来的ASR性能与使用同一说话人训练的向量相比具有竞争力甚至更优。

    arXiv:2610.26994v1 Announce Type: new  Abstract: Activation steering for ASR systems is an underexplored technique in improving both ASR performance and demographic fairness. We investigate two forms of activation steering: paired activation subtraction, and supervised optimization. We find the former to be ineffective at ameliorating ASR performance whatsoever, while the latter is quite effective at deriving steering vectors (SVs) that improve at once overall ASR performance and demographic fairness along such diverse axes as gender, age, and dialect. We also explore what types of data---from the most to the least personal---are necessary for calculating or training SVs. First, we show significant performance in performance from training SVs using an eclectic group of speakers from the same dataset. Furthermore, we find that vectors trained using acoustically similar speakers leads to ASR performances that are competitive or superior to vectors trained on the same speaker, or speakers
    
[^138]: TRACE：面向文本-视觉嵌入模型的训练感知鲁棒性与挑战评估

    TRACE: Training-aware Robustness and Challenge Evaluation for Text-Vision Embedding Models

    [https://arxiv.org/abs/2610.26993](https://arxiv.org/abs/2610.26993)

    本文提出TRACE评估框架，将168个文本-视觉嵌入任务重新组织为通用、适应性和挑战三类，实验发现混合通用与适应性数据可显著提升适应性性能但挑战性能仍低于5.0分，且多语言和视觉文档任务为主要薄弱环节，表明仅靠数据适应不足以解决这些挑战。

    

    现有针对文本-视觉嵌入模型的基准测试报告的是综合评分，这些评分掩盖了性能究竟来自通用能力、特定任务适应，还是对挑战性任务的鲁棒性。我们提出了TRACE，一个训练感知的鲁棒性与挑战评估框架，它将来自多种数据源的168个任务重新组织为三大类别：通用、适应性和挑战。通过对两个文本-视觉嵌入模型进行受控实验，我们表明结合通用数据和适应性数据分别使适应性性能提升了7.67和7.09分，同时基本保持了通用性能。然而，挑战性能仍然低于5.0分，额外的训练数据带来的改进有限。进一步分析发现，多语言任务和视觉文档任务是主要的薄弱环节。我们的发现表明，仅靠数据适应不足以应对这些挑战，这凸显了……

    arXiv:2610.26993v1 Announce Type: new  Abstract: Existing benchmarks for text-vision embedding models report aggregate scores that obscure whether performance comes from general capabilities, task-specific adaptation, or robustness to challenging tasks. We introduce TRACE, a Training-aware Robustness and Challenge Evaluation framework that reorganizes 168 tasks from various sources of datasets into three categories: General, Adaptability, and Challenge. Through controlled experiments on two text-vision embedding models, we show that combining general and adaptation data improves Adaptability performance by 7.67 and 7.09 points, respectively, while largely preserving General performance. However, Challenge performance remains below 5.0, with limited improvements from additional training data. Further analysis identifies multilingual and visual-document tasks as major weaknesses. Our findings demonstrate that data adaptation alone is insufficient to address these challenges, highlighting
    
[^139]: 先思考再转录：面向热词感知语音识别的上下文引导推理

    Think Before You Transcribe: Context-Steered Reasoning for Hotword-Aware Speech Recognition

    [https://arxiv.org/abs/2610.26986](https://arxiv.org/abs/2610.26986)

    该论文提出一种上下文引导的推理框架，通过在转录前先生成基于音频的简短描述来评估热词候选的相关性，从而在多种上下文条件下同时提升语音识别的转录质量和热词识别准确率，并对无关上下文保持鲁棒性。

    

    上下文自动语音识别（ASR）通过提供候选热词来改善罕见实体的识别，但检索到的候选词可能不完整或不正确。因此，一个可靠的系统必须能够利用相关的候选词，同时防止缺乏依据的备选词覆盖声学证据。我们提出了一种上下文引导的推理框架，在转录之前先生成简短的、基于音频的描述。这种任务特定的表示使模型能够在做出词汇决策之前评估候选词的相关性。监督微调建立了从描述到转录的结构化行为，而当困难语音只产生不成功的推演结果时，有针对性的轨迹注入增强了组相对策略优化（GRPO）。在多种上下文条件下的实验表明，所提出的方法在保持对无关上下文鲁棒性的同时，持续提升了转录质量和热词识别效果。

    arXiv:2610.26986v1 Announce Type: new  Abstract: Contextual automatic speech recognition (ASR) improves rare-entity recognition by supplying candidate hotwords, but retrieved candidates may be incomplete or incorrect. A reliable system must therefore exploit a relevant candidate without allowing unsupported alternatives to override the acoustic evidence. We propose a context-steered reasoning framework that generates a short, audio-grounded caption before transcription. This task-specific representation lets the model assess candidate relevance before making a lexical decision. Supervised fine-tuning establishes the structured caption-to-transcription behavior, while targeted trajectory injection strengthens group relative policy optimization when difficult utterances yield only unsuccessful rollouts. Experiments across diverse context conditions show that the proposed method consistently improves both transcription quality and hotword recognition while remaining robust to irrelevant c
    
[^140]: 基于解析平方和的关于正性裕度的 Putinar 证书的精确次数界

    Sharp Degree Bounds for Putinar Certificates in Terms of the Positivity Margin via Analytic Sums of Squares

    [https://arxiv.org/abs/2610.26976](https://arxiv.org/abs/2610.26976)

    本文建立了关于正性裕度 δ 的 Putinar 平方和证书的精确次数上界 O(δ^(-1/2)log²(2/δ))，并通过构造三变量匹配下界例子证明了对 δ 的依赖是最优的，同时给出了 Lasserre 层级的精确最坏情形收敛速率。

    

    我们建立了关于正性裕度 $\delta$ 的 Putinar 平方和（SOS）证书的精确次数界。对于任意具有指定生成元的固定阿基米德二次模以及目标函数次数的任意固定上界，我们证明了归一化多项式目标的一致证书次数上界为 $O(\delta^{-1/2}\log^2(2/\delta))$。我们通过在三变量情形下构造一个匹配的下界例子，证明了该界是精确的，从而确立了对 $\delta$ 的依赖性是最优的。这些匹配的界还给出了 Lasserre 层级的精确最坏情形收敛速率。

    arXiv:2610.26976v1 Announce Type: new  Abstract: We establish sharp degree bounds for Putinar sum-of-squares (SOS) certificates in terms of the positivity margin $\delta$. For any fixed Archimedean quadratic module with prescribed generators and any fixed bound on the degree of the objective, we prove a uniform certificate-degree upper bound of $O(\delta^{-1/2}\log^2(2/\delta))$ for normalized polynomial objectives. We show that this bound is sharp by constructing a matching lower-bound example in three variables, thereby establishing that the dependence on $\delta$ is optimal. These matching bounds also yield sharp worst-case convergence rates for the Lasserre hierarchy.
    
[^141]: Transformer学习两种相互竞争的排序算法，且只有一种能实现长度泛化

    Transformers Learn Two Competing Sorting Algorithms, and Only One Length-Generalizes

    [https://arxiv.org/abs/2610.26974](https://arxiv.org/abs/2610.26974)

    该论文发现两层Transformer在排序任务上会根据权重初始化收敛到两种机制截然不同的解，其中只有同时使用两个注意力层的解能在仅训练固定长度序列的情况下完美实现长度泛化，且可以通过精心设计权重初始化来引导模型学习目标解。

    

    我们证明，在排序任务上训练至接近零损失的两层Transformer，会根据其初始化收敛到两种机制上截然不同的解类别之一。其中一类解不使用第二个注意力层进行排序，同时也无法实现长度泛化。另一类解同时使用两个注意力层，尽管仅在固定长度的序列上训练，却能完美地实现长度泛化。通过精心选择权重初始化，我们证明可以引导模型学习其中一种解或另一种解。我们识别出在训练长度下罕见、但在更长序列长度下日益常见的输入类型，将其作为这种长度泛化差异的来源，并发现第二类解的逐层专门化使其能够更准确地处理这些罕见序列。在第二类解中，我们发现了一种机制，通过该机制，第二层（摘要在此处截断）

    arXiv:2610.26974v1 Announce Type: cross  Abstract: We show that two-layer transformers trained to near-zero loss on a sorting task converge to one of two mechanistically distinct solution classes depending on their initialization. One solution class consists of solutions that do not use the second attention layer to sort, but also fail to length-generalize. The other, which uses both attention layers, length-generalizes perfectly, despite only being trained to sort sequences of a fixed length. By carefully choosing the weight initialization, we demonstrate that we can steer the model toward learning one solution or the other. We identify input types that are rare at the training length but increasingly common at longer sequence lengths as a source of this disparity in length-generalization, and find that layer-wise specialization of the second class of solutions allows them to handle these rare sequences more accurately. In the second class, we discover a mechanism by which the second 
    
[^142]: SZED参加ArGuard共享任务：面向阿拉伯语仇恨表情包检测的多任务标签注意力与跨模态融合方法

    SZED at ArGuard Shared Tasks: Multi-Task Label-Attention and Cross-Modal Fusion for Arabic Hateful Meme Detection

    [https://arxiv.org/abs/2610.26972](https://arxiv.org/abs/2610.26972)

    该论文提出了一种融合MARBERTv2文本嵌入与CLIP视觉嵌入的多任务标签注意力架构，并引入级联校准机制，用于阿拉伯语仇恨表情包检测，在二分类任务上取得macro-F1 0.7597、在细粒度多标签任务上取得0.3142的成绩。

    

    我们展示了SZED团队参加ArGuard 2026赛道A（阿拉伯语仇恨表情包检测）的提交方案。该任务包含两个部分：二分类（A1）和细粒度多标签标注（A2）。对于A1，我们融合了MARBERTv2和CLIP ViT-L/14的嵌入表示，并比较了拼接、自注意力和集成头三种融合方式。对于A2，我们采用多任务标签注意力架构，为每个类别设置独立的查询向量，并使用辅助头进行级联校准。我们在A1上达到macro-F1 = 0.7597，在A2上达到0.3142（对10个活跃类别取平均）。我们报告了多次运行结果的可复现性、各类别的错误模式，以及级联校准对长尾分布A2标签空间的影响。

    arXiv:2610.26972v1 Announce Type: cross  Abstract: We present SZED's submission to ArGuard 2026 Track A: Arabic hateful meme detection. The task has two parts: binary classification (A1) and fine-grained multi-label tagging (A2). For A1, we fuse MARBERTv2 and CLIP ViT-L/14 embeddings, comparing concatenation, self-attention, and ensemble heads. For A2, we use a multi-task label-attention architecture with per-class query vectors and an auxiliary head for cascade calibration. We reach macro-F1 = 0.7597 on A1 and 0.3142 on A2 (averaged over 10 active classes). We report reproducibility across runs, per-class error patterns, and the effect of cascade calibration on the long-tailed A2 label space.
    
[^143]: 大语言模型真的缺乏差异意识吗？揭示群体差异化公平中的启发式替代现象

    Do LLMs Really Lack Difference Awareness? Uncovering Heuristic Substitution in Group-Differentiating Fairness

    [https://arxiv.org/abs/2610.26961](https://arxiv.org/abs/2610.26961)

    本文提出大语言模型在群体差异化公平任务中的失败并非源于缺乏差异意识，而是因为“公平启发式替代”——模型用更简单的形式平等启发式替代了复杂的差异识别与权衡过程，并据此提出两阶段“区分-决策”框架TDD来改善这一问题。

    

    在本文中，我们研究了大语言模型（LLMs）中的差异意识问题，即模型在公平性决策中识别和承认社会群体间有意义差异的能力。以往的研究一致表明，模型在这类任务上表现不佳，这一发现通常被归因于模型缺乏差异意识。我们提出了一种替代性解释，称之为公平启发式替代：差异意识并非必然缺失，而是同时识别相关群体差异并确定其对公平决策影响这一复杂需求，可能导致模型转而采用更简单、更易获取的平等待遇启发式（即形式平等）。基于这一假设，我们引入了TDD——一个两阶段“区分-决策”框架，将公平性推理明确分解为两个步骤：首先区分情境中相关的群体差异，然后做出相应的公平决策。

    arXiv:2610.26961v1 Announce Type: cross  Abstract: In this paper, we investigate the issue of difference awareness in LLMs, defined as a model's ability to recognize and acknowledge meaningful differences between social groups in fairness decision-making. Previous studies have consistently shown that models perform poorly on such tasks, a finding commonly attributed to a lack of difference awareness. We propose an alternative account, which we term Fairness Heuristic Substitution: difference awareness is not necessarily absent, but the complex demand of simultaneously identifying relevant group differences and determining their implications for fairness decisions may lead models to substitute the simpler, more accessible equal-treatment heuristic, i.e., formal equality. Based on this hypothesis, we introduce TDD, a Two-stage Distinguish-Decide framework that explicitly decomposes fairness reasoning into first distinguishing contextually relevant group differences and then making the co
    
[^144]: 基于动力系统启发的面向AI4Science任务的连续大语言模型

    Dynamic System Inspired Continuous Large Language Model for AI4Science Tasks

    [https://arxiv.org/abs/2610.26959](https://arxiv.org/abs/2610.26959)

    Continuum-LM将预训练的残差变换复用为动力系统算子，构建了通过常微分方程演化隐藏状态的连续深度语言模型，在不扩展主干网络的前提下增加了潜在计算能力，在八项科学任务上将平均精确匹配准确率显著提升了7.77至10.96个百分点。

    

    科学基础模型日益需要处理连续现象，但它们仍然使用离散的残差计算。我们提出这样的问题：预训练的残差变换能否在不扩展主干网络的情况下支持额外的潜在计算。为验证这一可能性，我们提出了Continuum-LM，它将这些残差变换复用为连续深度语言模型中的动力学算子。其隐藏状态通过常微分方程进行演化，并采用高阶数值积分进行求解，其中求解器的阶数和步长控制着计算量。在八项科学任务上，Continuum-LM训练稳定，在0.6B和1.7B参数规模下，平均精确匹配准确率分别比离散对应模型提升了7.77和10.96个百分点。在0.6B规模下，在匹配的参数量和残差函数评估预算下，它还优于Ouro风格的循环控制模型。这些结果共同支持了可扩展的计算……（摘要在此处截断）

    arXiv:2610.26959v1 Announce Type: cross  Abstract: Scientific foundation models increasingly operate on continuous phenomena, yet still use discrete residual computation. We ask whether pretrained residual transformations can support additional latent computation without expanding the backbone. To test this possibility, we introduce Continuum-LM, which reuses these transformations as dynamical operators in a continuous-depth language model. Its hidden states evolve through an ordinary differential equation evaluated by higher-order numerical integration, with solver order and step size controlling computation. Across eight scientific tasks, Continuum-LM trains stably and improves mean exact-match accuracy by 7.77 and 10.96 percentage points over discrete counterparts at the 0.6B and 1.7B scales, respectively. At 0.6B, it also outperforms an Ouro-style recurrent control under matched parameter and residual-function evaluation budgets. Together, these results support extensible computati
    
[^145]: LAMB：面向长时程大语言智能体的潜在多假设信念

    LAMB: Latent Multi-Hypothesis Beliefs for Long-Horizon Large Language Agents

    [https://arxiv.org/abs/2610.26955](https://arxiv.org/abs/2610.26955)

    提出LAMB框架，通过潜在空间中的Common-Delta分解机制将交互证据与竞争性假设分离表示，使长时程语言智能体无需生成文本记忆即可高效维持多假设信念。

    

    长时程语言智能体需要紧凑的记忆，既要保留交互证据，又要维持关于未观测状态的多个竞争性假设。尽管现有信念状态方法行之有效，但许多方法以离散自然语言编码记忆，反复压缩可能导致证据遗漏或过早丢弃合理的备选假设。我们提出LAMB，一个用于潜在多假设信念的循环框架。Common-Delta分解将共享的Common状态（用于存储交互证据）与加权的Delta组件（用于表示竞争性假设）相结合。来自动作-结果预测的相对证据引导假设的循环更新，而无需生成文本记忆。在默认配置中，上下文条件化的读出机制通过固定token的信念前缀将各个假设分别呈现给策略。信念模块与策略通过动作模仿以及辅助的状态、结果和观测判别目标进行联合训练。

    arXiv:2610.26955v1 Announce Type: new  Abstract: Long-horizon language agents require compact memories that retain interaction evidence while preserving competing hypotheses about unobserved states. Although effective, many belief-state methods encode memory in discrete natural language, where repeated compression can omit evidence or prematurely discard plausible alternatives. We introduce LAMB, a recurrent framework for latent multi-hypothesis beliefs. A Common--Delta factorization combines a shared Common state for interaction evidence with weighted Delta components for competing hypotheses. Relative evidence from action-outcome predictions guides recurrent hypothesis updates without generating textual memory. A context-conditioned readout exposes hypotheses separately to the policy through a fixed-token belief prefix in the default configuration. The belief modules and policy are jointly trained using action imitation and auxiliary state, outcome, and observation-discrimination obj
    
[^146]: 面向关系三元组抽取的多粒度交互网络

    A Multi-granularity Interaction Network for Relational Triple Extraction

    [https://arxiv.org/abs/2610.26952](https://arxiv.org/abs/2610.26952)

    提出一种多粒度交互网络，通过词与词、实体与关系、头尾实体三种交互及多次迭代来捕捉深层语义依赖，显著提升了关系三元组抽取中对“未见过”实体的处理性能。

    

    关系三元组抽取旨在不依赖实体类型信息的情况下，从原始句子中识别实体之间的关系。传统方法在面对“未见过”的实体时性能会严重退化，因为这些方法无法构建与“未见过”实体相关的上下文依赖关系。本文通过提出一种多粒度交互网络来应对这一挑战，该网络利用词与词、实体与关系、头实体与尾实体之间的交互来捕捉深层语义依赖。通过引入多次迭代机制，多粒度交互网络能够有效精炼头实体和尾实体对应的表示，增强了与关系三元组相关的全局语义特征，有助于修复在早期步骤中可能被忽略或遗漏的三元组。该方法在两个重新整理的数据集上取得了强劲且一致的性能，并实现了显著提升。

    arXiv:2610.26952v1 Announce Type: new  Abstract: Relational triple extraction identifies entity relations from raw sentences without depending on entity type information. The performance of traditional methods is seriously degenerated by ``unseen'' entities, because these methods can't build contextual dependencies relevant ``unseen'' entities. This paper addresses this challenge by proposing a multi-granularity interaction network that leverages word-word, entity-relation, and head-tail interactions to capture deep semantic dependencies. By incorporating multiple iterations, the multi-granularity interaction network is effective to refine the representations corresponding to head and tail entities. It enhances the global semantic features associated with relational triples, aiding in the remediation of triples that may have been overlooked or omitted in earlier steps. Strong and consistent performance is achieved by our method on two rearranged datasets, with a significant improvement
    
[^147]: 瓶颈在于检索而非阅读：用基于证据的生成诊断面向长上下文问答的RAG

    The Bottleneck Is Retrieval, Not Reading: Diagnosing Evidence-Based Generation with RAG for Long Context Question Answering

    [https://arxiv.org/abs/2610.26951](https://arxiv.org/abs/2610.26951)

    该论文提出基于证据的生成（EBG）方法及 Ans+Evid 联合评估指标，通过构建多个多文档问答基准诊断发现：RAG 的瓶颈在于基于相似度的检索环节而非模型阅读，而这些检索局限正是长上下文模型所能规避的。

    

    检索增强生成（RAG）是将大语言模型与外部语料库相结合的主流范式，但其基于相似度的分块流水线存在结构性盲区：它无法预判问题需要哪些段落，并且会丢弃能够解决跨段落引用的文档级上下文。我们提出了基于证据的生成（EBG），要求模型在返回答案的同时提供行级引用，并设计了联合的Ans+Evid指标，可以同时评估答案的正确性与证据支撑度。我们将两个语料库重新构建为基于证据的多文档问答基准——MultiHop-RAG 和 QASPER-MP（QASPER 的多论文改编版本，带有真实场景中的短论文提示），并配以受控的合成两跳“大海捞针”测试，以及覆盖1万至100万字符的证据增强版 NoLiMa。在十个专有前沿大模型和三个开源权重大模型上的实验中，EBG 分离出了长上下文模型可以规避的两个 RAG 局限性：相似性检索……（摘要在此处截断）

    arXiv:2610.26951v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) is the dominant paradigm for grounding large language models in external corpora, but its similarity-driven, chunked pipeline has structural blind spots: it cannot anticipate which passages a question requires and discards document-level context that resolves cross-passage references. We introduce evidence-based generation (EBG), requiring models to return an answer with line-level citations, and a joint Ans+Evid metric that scores correctness and grounding simultaneously. We reformulate two corpora into evidence-grounded multi-document QA benchmarks- MultiHop-RAG and QASPER-MP, a multi-paper adaptation of QASPER with realistic short paper cues, and pair them with a controlled synthetic two-hop haystack and an evidence-augmented NoLiMa spanning 10K-1M characters. Across ten proprietary frontier and three open-weight LLMs, EBG isolates two RAG limitations that long-context models avoid: similarity retr
    
[^148]: GRaSP：面向地理分布式检索增强生成的延迟感知优化

    GRaSP: Latency-Aware Optimization for Geo-Distributed Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.26938](https://arxiv.org/abs/2610.26938)

    该论文提出GRaSP，一种面向地理分布式检索增强生成的延迟感知重评分方法，通过将问题形式化为划分拟阵上的线性目标并用O(N log N)贪心算法最优求解，在无需重训练的情况下兼顾检索相关性、延迟与成本。

    

    检索增强生成（RAG）日益被部署在地理分布式的云和边缘基础设施上，在这些环境中，广域网延迟、跨区域出口流量成本以及数据驻留规则与检索质量相互制约。传统流水线仅依据语义相似度进行排序，而忽略了内容所在的地理位置，从而推高了尾延迟和成本。我们将地理RAG选择问题形式化为GRaSP，这是一个即插即用的重评分层，通过延迟对相关性进行衰减，并在基数预算和各区域上限约束下添加可选的成本惩罚，且无需重新训练。我们首先确定了该问题的计算复杂度：基础形式化是划分拟阵上的线性目标函数，可以通过O(N log N)的贪心算法最优求解，而非像背包问题框架所暗示的那样是NP难问题；只有在一附加出口预算约束下问题才会变难。在模拟四区域广域网的五节点CloudLab测试床上，基于Natural Questions语料库（268万个段落）的实验中，GRaSP的收益……

    arXiv:2610.26938v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) is increasingly served across geo-distributed cloud and edge infrastructures, where wide-area latency, cross-region egress cost, and data-residency rules compete with retrieval quality. Conventional pipelines rank by semantic similarity alone and ignore where content lives, inflating tail latency and cost. We formalize the Geo-RAG Selection Problem (GRaSP), a drop-in re-scoring layer that decays relevance by latency and adds an optional cost penalty under a cardinality budget and per-region caps, with no retraining. We first settle its complexity: the base formulation is a linear objective over a partition matroid, solved optimally by an O(N log N) greedy algorithm, rather than NP-hard as a knapsack framing suggests, with hardness entering only under an additive egress budget. On a five-node CloudLab testbed emulating a four-region WAN over the Natural Questions corpus (2.68M passages), GRaSP's bene
    
[^149]: 我们真的需要为每个临床决策都使用最大的大语言模型吗？

    Do We Need the Largest LLMs for Every Clinical Decision?

    [https://arxiv.org/abs/2610.26937](https://arxiv.org/abs/2610.26937)

    提出CERTA决策级统计认证框架，证明量化后的小型模型可在统计保证下替代大型BF16模型处理绝大多数临床验证决策——8位量化满足99.8-100%的决策，4位量化将部署成本降至25.8%，因此并非每个临床决策都需要最大的LLM。

    

    大型临床语言模型通常需要多GPU基础设施，这限制了本地部署。我们提出了CERTA，一种决策级统计证书，用于判定量化模型何时可以替代其原生BF16参考模型，并将无法通过认证的决策升级至参考模型处理。我们在来自三个模型家族（7-72B）的十一个指令微调模型（涵盖通用模型和领域适配模型）上，评估了来自去标识化患者情境（2,484名患者）的2,500个临床验证决策，校准与测试采用患者不相交的划分方式。在更高检验效能的操作性分析中，采用BF16存储的最近取整8位量化在所有十一个模型上满足了99.8-100%的留出决策；形式化的有限样本保证则使用精确的拆分样本程序单独评估。在相同的操作性分析中，类条件的4位量化前沿方案将摊销后的BF16等效部署成本降低至25.8%。

    arXiv:2610.26937v1 Announce Type: new  Abstract: Large clinical language models often require multi-GPU infrastructure, limiting local deployment. We introduce CERTA, a decision-level statistical certificate that determines when a quantized model may substitute for its native BF16 reference and escalates decisions that cannot be certified to it. Across eleven instruction-tuned models, both general-purpose and domain-adapted, from three families (7-72B), we evaluated 2,500 clinical-verification decisions from de-identified patient contexts (2,484 patients), with patient-disjoint calibration and testing. In the higher-power operational analysis, round-to-nearest 8-bit quantization at BF16 storage served 99.8-100% of held-out decisions across all eleven models; formal finite-sample guarantees were assessed separately using the exact split-sample procedure. In the same operational analysis, the class-conditional 4-bit frontier reduced the amortized, BF16-equivalent deployment cost to 25.8%
    
[^150]: AtomSD：面向大语言模型智能体的过程性原子引导自蒸馏

    AtomSD: Procedural Atom-Guided Self-Distillation for LLM Agents

    [https://arxiv.org/abs/2610.26926](https://arxiv.org/abs/2610.26926)

    提出AtomSD框架，将细粒度、可独立复用的“过程性原子”行动知识从智能体自身成功轨迹中持续提取并自蒸馏进策略，使LLM智能体性能超越现有最先进方法最多9.31分。

    

    大语言模型（LLM）拥有广泛的事实知识和强大的推理能力，但解决复杂任务还需要关于“如何行动”的过程性知识。现有方法通常通过轨迹或捆绑技能来表示过程性知识，这限制了其在交互历史中的灵活复用。本文提出了“过程性原子”，这是一种细粒度且可独立应用的单元，用于表示意图、时机、执行和避免的指导信息。基于这一表示，我们提出了AtomSD——一种原子引导的自蒸馏框架，其中学习器持续地从自身的成功轨迹中提取过程性原子，并通过原子条件监督和任务奖励优化将其蒸馏到不断演进的策略中。全面的实验表明，AtomSD持续优于强力的基于奖励的方法和蒸馏方法，平均性能超出最先进的基线高达9.31分。

    arXiv:2610.26926v1 Announce Type: new  Abstract: Large language models (LLMs) possess broad factual knowledge and strong reasoning capabilities, but solving complex tasks also requires procedural knowledge about how to act. Existing approaches often represent procedural knowledge through trajectories or bundled skills, limiting flexible reuse across interaction histories. This paper introduces Procedural Atoms, fine-grained and independently applicable units that represent Intent, When, Do, and Avoid guidance. Building on this representation, we propose AtomSD, an Atom-guided Self-Distillation framework in which the learner continually extracts Procedural Atoms from its own successful trajectories and distills them into the evolving policy through Atom-conditioned supervision and task-reward optimization. Comprehensive experiments show that AtomSD consistently outperforms strong reward-based and distillation methods, exceeding the state-of-the-art baseline by up to 9.31 points in avera
    
[^151]: PW-Pruning：面向语音中心全模态模型的阶段感知专家选择

    PW-Pruning: Phase-Aware Expert Selection for Speech-Centric Omni-Modal Models

    [https://arxiv.org/abs/2610.26891](https://arxiv.org/abs/2610.26891)

    提出阶段感知的专家剪枝方法PW-Pruning，为声学预填充和文本答案生成两个阶段分别构建重要性加权核并融合阶段加权对数行列式边际增益来选择静态专家库，在Qwen3-Omni上每层仅保留32/128个专家即可使MMAU-mini和MMSU准确率较池化选择提升3.20和4.38个百分点。

    

    稀疏路由虽然降低了每个token的计算量，但完整的专家库仍然需要全部驻留在内存中。我们研究了以语音为中心的全模态模型的专家剪枝问题，在该场景下，使用池化校准统计会掩盖声学输入处理与文本答案生成两个阶段对不同专家的需求差异。我们提出了PW-Pruning，这是DO-ACP的一种阶段感知扩展方法，它为声学预填充阶段和教师强制下的答案token分别构建重要性加权核，然后结合阶段加权的对数行列式边际增益来选出一个静态专家库。在Qwen3-Omni上，每层仅保留128个专家中的32个，经过监督微调并在三次运行中取平均后，MMAU-mini和MMSU的准确率分别比匹配的池化选择方法提高了3.20和4.38个百分点。在该公开骨干模型上进行的单阶段和混洗标签对照实验表明，语义阶段信息对上述提升有所贡献。在非公开的Qwen3.5-O...（摘要截断）

    arXiv:2610.26891v1 Announce Type: cross  Abstract: Sparse routing reduces per-token computation but leaves the full expert bank resident in memory. We study expert pruning for speech-centric omni-modal models, where pooling calibration statistics can obscure the different expert demands of acoustic-input processing and text-answer generation. We propose PW-Pruning, a phase-aware extension of DO-ACP that constructs separate importance-weighted kernels for acoustic-prefill and teacher-forced answer tokens, then combines phase-weighted log-determinant marginal gains to select one static expert bank. On Qwen3-Omni, retaining 32 of 128 experts per layer improves MMAU-mini and MMSU accuracy over matched pooled selection by 3.20 and 4.38 percentage points, respectively, after supervised fine-tuning, averaged over three runs. Single-phase and shuffled-label controls on this public backbone suggest that semantic phase information contributes to these gains. Experiments on a non-public Qwen3.5-O
    
[^152]: 利用同步临床评估从老年人语音中解耦抑郁与认知衰退

    Disentangling Depression from Cognitive Decline in Elderly Speech Using Concurrent Clinical Assessments

    [https://arxiv.org/abs/2610.26890](https://arxiv.org/abs/2610.26890)

    该研究利用历时三年收集、带有同步抑郁与认知评估的韩国MCI老年人语音语料库，发现控制认知功能后仅共振峰特征与抑郁相关，仅12个特征的F1+F2子集即达到UAR 0.760，超越完整eGeMAPS基线和自监督表示。

    

    从轻度认知障碍（MCI）老年患者的语音中检测抑郁症，会因认知衰退产生的重叠声学效应而变得复杂。若不将这些效应解耦，分类器可能学习到的是认知相关的模式，而非抑郁特异性的模式。我们提出了一个韩国老年人语音语料库，包含89名MCI说话者，历时三年收集，并同时进行抑郁（SGDS）和认知（MMSE）评估。线性混合效应模型和偏相关分析均识别出相同的模式：在七个特征组中，只有共振峰特征在控制认知功能后与抑郁相关，而广泛使用的F0（基频）特征不携带任何信号。仅使用12个特征的F1+F2子集实现了0.760的UAR（非加权平均召回率），优于完整的eGeMAPS基线和自监督表示。这些发现表明，同步临床评估能够在该人群中识别出可解释的、抑郁特异性的声学标记。

    arXiv:2610.26890v1 Announce Type: cross  Abstract: Detecting depression from speech in elderly patients with mild cognitive impairment (MCI) is complicated by overlapping acoustic effects of cognitive decline. Without disentangling these, classifiers risk learning cognitive rather than depression-specific patterns. We present a Korean elderly speech corpus of 89 MCI speakers collected over three years with concurrent depression (SGDS) and cognitive (MMSE) assessments. Both linear mixed-effects models and partial correlations identify the same pattern: of seven feature groups, only formant features are associated with depression after controlling for cognitive function, while widely used F0 features carry no signal. The F1+F2 subset achieves UAR 0.760 with just 12 features, outperforming both the full eGeMAPS baseline and self-supervised representations. These findings show that concurrent clinical assessments can identify interpretable, depression-specific acoustic markers in populatio
    
[^153]: 基于音素专家混合的低秩适配Conformer模型用于带口音英语语音识别

    Mixture of Phonetic Experts Based Low-Rank Adaptation of Conformer Models for Accented English Speech Recognition

    [https://arxiv.org/abs/2610.26888](https://arxiv.org/abs/2610.26888)

    该论文提出MoPE-LoRA框架，按照发音方式类别而非口音身份来组织六个低秩专家，通过音素监督与声学门控相结合的帧级混合路由，实现参数高效的带口音英语语音识别适配，并可自然泛化到未见过的口音。

    

    我们提出了基于音素专家混合的低秩适配方法（MoPE-LoRA），这是一种用于带口音英语语音识别的参数高效适配框架。近期的混合专家方法通常为每种口音分配专家，隐式地将口音身份视为主要的变异轴。相比之下，我们沿着语音类别来组织适配，其动机在于观察到口音差异往往表现为音素实现的系统性失真。MoPE-LoRA引入了一组固定的、与发音方式类别对齐的六个低秩专家，并通过结合音素监督与可学习声学门控的混合机制进行帧级路由。由于专家集合在所有口音之间共享，该模型不需要特定于口音的模块，并且能够自然地扩展到先前未见过的口音。在L2-ARCTIC数据集上的实验表明，MoPE-LoRA始终优于单个LoRA基线。

    arXiv:2610.26888v1 Announce Type: new  Abstract: We present Mixture of Phonetic Experts based Low-Rank Adaptation (MoPE-LoRA), a parameter-efficient adaptation framework for accented English speech recognition. Recent mixture-of-experts approaches typically allocate experts per accent, implicitly treating accent identity as the primary axis of variation. In contrast, we organize adaptation along phonetic categories, motivated by the observation that accent differences are often expressed as systematic distortions of phoneme realization. MoPE-LoRA introduces a fixed set of six low-rank experts aligned with manner-of-articulation classes and performs frame-level routing through a hybrid mechanism that combines phoneme supervision with learned acoustic gating. Because the expert set is shared across accents, the model does not require accent-specific modules and naturally extends to previously unseen accents. Experiments on L2-ARCTIC show that MoPE-LoRA consistently improves over single L
    
[^154]: 基于双向音频-文本交叉注意力的多模态矛盾与犹豫识别

    Multimodal Ambivalence and Hesitancy Recognition with Bidirectional Audio--Text Cross-Attention

    [https://arxiv.org/abs/2610.26868](https://arxiv.org/abs/2610.26868)

    该论文提出了一种基于双向音频-文本交叉注意力的多模态框架，融合多语言BERT和wav2vec 2.0的表征，在无需显式时间对齐的情况下识别视频中的矛盾与犹豫状态，在BAH数据集上达到72.19%的准确率。

    

    矛盾与犹豫（A/H）是一种微妙的行为状态，可以通过一个人说了什么以及说话的方式来表达。本文提出了一个用于视频级A/H识别的音频-文本框架。完整视频的文本转录和语音波形分别使用预训练的多语言BERT和wav2vec 2.0进行编码，并投影到共享表示空间中。两个堆叠的双向交叉注意力模块使语言标记能够关注声学帧，同时声学帧也能关注语言上下文，而无需显式的时间对齐。得到的序列通过掩码平均池化进行聚合，并使用拼接、逐元素乘积和绝对差进行组合以进行二分类。我们在行为矛盾/犹豫（BAH）数据集的受试者独立划分上对该模型进行了评估。所提出的系统达到了72.19%的准确率、75.00%的精确率、81.13%的……

    arXiv:2610.26868v1 Announce Type: new  Abstract: Ambivalence and hesitancy (A/H) are subtle behavioural states that can be expressed through both what a person says and how it is spoken. This paper presents an audio--text framework for video-level A/H recognition. Full-video transcripts and speech waveforms are encoded using pre-trained multilingual BERT and wav2vec~2.0, respectively, and projected into a shared representation space. Two stacked bidirectional cross-attention blocks allow linguistic tokens to attend to acoustic frames and acoustic frames to attend to linguistic context without requiring explicit temporal alignment. The resulting sequences are aggregated through masked mean pooling and combined using concatenation, element-wise products, and absolute differences for binary classification. We evaluate the model on the subject-independent split of the Behavioural Ambivalence/Hesitancy (BAH) dataset. The proposed system achieves 72.19\% accuracy, 75.00\% precision, 81.13\% 
    
[^155]: WaveCommitments：私密且可验证的语音智能体市场

    WaveCommitments: Private and Verifiable Speech-Agent Marketplaces

    [https://arxiv.org/abs/2610.26867](https://arxiv.org/abs/2610.26867)

    提出WaveCommitments，一种将可信执行环境内基于Data Shapley的机密资产估值与多元多项式承诺相结合的端到端设计，实现了语音智能体市场中私密且可验证的资产交易，相比最佳基准速度提升最高20倍、证明大小缩减最高39倍。

    

    现代语音系统正在演变为由自主语音智能体组成的生态系统，这些智能体协同发声、协商并交易诸如嵌入向量、频谱图或原始数据等资产。在这类市场中，出于隐私约束，卖家无法公开资产供评估，而买家也无法为其无法测试的东西定价。因此，我们开源了WaveCommitments，这是一种端到端设计，将可信执行环境内基于Data Shapley的机密资产估值与多元多项式承诺相结合。它支持选择性公开和防篡改溯源，能够证明交付的资产与估值相符，同时隐藏其他所有信息。我们在三个知名数据集上，使用包括Whisper在内的最先进编码器，针对多种任务进行了大量实验，结果表明我们的方法将最佳基准的速度提升了最高20倍，同时将证明大小缩减了最高39倍。我们还系统地研究了说话人的价值如何随任务变化而偏移。

    arXiv:2610.26867v1 Announce Type: new  Abstract: Modern speech systems are evolving into ecosystems of autonomous speech agents that speak together, negotiate, and trade assets such as embeddings, spectrograms, or raw data. In such markets, sellers cannot reveal assets for evaluation due to privacy constraints, and buyers cannot price what they cannot test. Thus, we open-source WaveCommitments, an end-to-end design that pairs confidential asset valuation via Data Shapley inside a trusted execution environment with multivariate polynomial commitments. It enables selective openings and tamper-evident provenance to prove that a delivered asset matches the valuation while hiding everything else. Our extensive experiments across three known datasets and various tasks with state-of-the-art encoders, including Whisper, reveal that we speed up the best benchmarks by up to 20x while reducing their proof sizes by up to 39x. We also systematically study how a speaker's value shifts across tasks a
    
[^156]: 高维层状态轨迹揭示Transformer推理的内部动力学

    High-dimensional layer-state trajectories reveal the internal dynamics of transformer inference

    [https://arxiv.org/abs/2610.26866](https://arxiv.org/abs/2610.26866)

    该研究将Transformer逐层推理的有序层状态轨迹作为经验对象，发现真实层顺序产生的轨迹对齐度是保持终点随机打乱的1.60至2.24倍，证明层顺序构成一种与架构兼容而非训练特有的可测量计算，并可通过局部转换执行器实现对内部推理轨迹的选择性控制。

    

    大语言模型的机制研究将功能定位于参数、注意力头或神经元之中，但这些部件并不能说明一次推理是如何逐步展开的。我们将有序的层状态轨迹视为经验过程，并检验层顺序是否定义了一种可测量的计算。在代表性的Qwen、Llama和Gemma检查点上，实际实现的层顺序所产生的路径与其终点弦的对齐程度是保持终点的随机打乱顺序的1.60至2.24倍。这种顺序效应可以迁移到词汇任务中，但在同一Qwen架构的三个随机初始化模型中表现更强，这表明它是一种与架构兼容的支架结构，而非训练特有的规律。直接隐藏状态比词汇回读提供了更强的机制坐标，而任务定义的坐标使得候选词之间的竞争变得可计算。在加性干预失败之后，局部转换执行器能够选择性地控制内部轨迹。

    arXiv:2610.26866v1 Announce Type: cross  Abstract: Mechanistic studies of large language models localize function in parameters, attention heads or neurons, but those parts do not specify how one inference unfolds. We treat the ordered layer-state trajectory as the empirical process and test whether layer order defines a measurable computation. Across representative Qwen, Llama and Gemma checkpoints, realized order produced paths 1.60--2.24 times as aligned with their endpoint chord as endpoint-preserving shuffles. The order effect transferred to a lexical task, but was stronger in three random initializations of the same Qwen architecture, identifying an architecture-compatible scaffold rather than a training-specific law. Direct hidden states provided stronger mechanism coordinates than vocabulary readbacks, and a task-defined coordinate made candidate competition computable. After additive interventions failed, local transition actuators selectively controlled internal trajectories 
    
[^157]: AutoTerm-SST：面向流式语音翻译的零会话时设置自适应术语记忆

    AutoTerm-SST: Adaptive Terminology Memory with Zero Session-Time Setup for Streaming Speech Translation

    [https://arxiv.org/abs/2610.26865](https://arxiv.org/abs/2610.26865)

    AutoTerm-SST提出了一种无需会话时上传术语表和选择领域的自适应术语记忆系统，通过将离线术语目录组织为主题切片并根据流式翻译上下文动态激活相关切片，在肿瘤学演讲翻译中实现了与已知领域设置相当的术语准确率。

    

    检索增强语音翻译能够改善稀有术语的翻译，但诸如RASST等系统仍然假设用户在每次会话开始前提供术语表。这一假设对于现场演讲来说并不方便，而将所有领域合并到一个不断增长的术语表中，最终会导致过多的无关术语出现在同一检索和提示竞争之中。我们提出AutoTerm-SST，这是一个消除了会话时术语表上传和领域选择需求的交互式系统。该系统将大型离线术语目录组织成主题切片，将最近的流式翻译上下文与双语切片原型进行匹配，并在固定的检索和提示预算下激活一个紧凑的多切片工作集。在五场ACL和五场ESO肿瘤学演讲上，AutoTerm在术语准确率（88.3% vs. 88.1%）和mWER重新分词BLEU分数（45.09 vs. 45.19）方面与已知领域的参考系统相当，同时无需任何领域输入。与一个包含一百万条目的合并术语表相比，它…（摘要在此处被截断）

    arXiv:2610.26865v1 Announce Type: new  Abstract: Retrieval-augmented speech translation improves rare terminology, but systems such as RASST still assume that users provide a glossary before each session. That assumption is inconvenient for live talks, while merging every domain into one growing glossary eventually places too many irrelevant terms in the same retrieval and prompt competition. We present AutoTerm-SST, an interactive system that removes session-time glossary upload and domain selection. It organizes a large offline terminology catalog into topic slices, matches recent streaming translation context to bilingual slice prototypes, and activates a compact multi-slice working set under fixed retrieval and prompt budgets. On five ACL and five ESO oncology talks, AutoTerm matches a known-domain reference in terminology accuracy (88.3% vs. 88.1%) and mWER-resegmented BLEU (45.09 vs. 45.19), while requiring no domain input. Compared with a one-million-entry merged glossary, it im
    
[^158]: H2DF：用于可控语言模型遗忘的校准目标区间

    H2DF: Calibrated Target Bands for Controllable Language-Model Unlearning

    [https://arxiv.org/abs/2610.26864](https://arxiv.org/abs/2610.26864)

    H2DF提出了一种校准的目标区间目标函数，通过数据依赖的目标带控制梯度上升压力，在遗忘达到目标后自动抑制或反转压力，从而实现可控的语言模型遗忘，同时保护保留数据的效用。

    

    梯度上升（GA）是一种简单的语言模型遗忘基线方法，但在有用的遗忘已经发生后，它仍会持续增加遗忘损失。这种无界的压力会损害保留数据的效用，并使被遗忘的样本出现异常高的损失。我们提出了H2DF，一种校准的目标区间目标函数，它在低于数据相关的下目标时施加类似GA的压力，并在达到目标区域后抑制或反转该压力。这些目标是在训练前从匹配的非成员样本中估计得到的，因此基础目标函数在每个优化步骤中仅需一次遗忘批次的前向/反向传播；我们报告的TOFU变体每十步增加一次稀疏的保留数据回放。我们为TOFU问答任务实例化了一个单侧目标，为MUSE-News语言建模任务实例化了一个双侧目标。在三个TOFU随机种子上，H2DF取得了遗忘NLL为1.528（裕度0.049）、保留NLL为1.165（裕度0.047）的结果，与GradDiff相当。

    arXiv:2610.26864v1 Announce Type: cross  Abstract: Gradient ascent (GA) is a simple language-model unlearning baseline, but it continues increasing forget loss after useful forgetting has occurred. This unbounded pressure can damage retained utility and make forgotten examples abnormally high-loss. We introduce H2DF, a calibrated target-band objective that applies GA-like pressure below a data-dependent lower target and suppresses, or reverses, that pressure after the target region is reached. The targets are estimated from matched non-member examples before training, so the base objective requires one forget-batch forward/backward pass per optimization step; our reported TOFU variant adds sparse retain replay every tenth step. We instantiate a one-sided objective for TOFU question answering and a two-sided objective for MUSE-News language modeling. Across three TOFU seeds, H2DF obtains a forget NLL of 1.528 with margin 0.049 and retain NLL of 1.165 with margin 0.047, matching GradDiff
    
[^159]: WeAct：将动态权重切换作为智能体动作

    WeAct: Dynamic Weight Switching as Agent Actions

    [https://arxiv.org/abs/2610.26847](https://arxiv.org/abs/2610.26847)

    WeAct在冻结主干上训练独立的能力专用LoRA适配器，并将适配器选择纳入智能体动作空间，使模型能在长时程任务中通过原生工具调用接口按需组合能力，同时避免灾难性遗忘。

    

    长时程智能体必须从交错的轨迹中学习异构能力。更新共享参数以学习新能力可能会导致先前获得的能力退化，从而造成灾难性遗忘。随着轨迹的展开，不同阶段可能需要不同的能力，然而传统的联合训练和静态适配器合并在整个执行过程中都维持固定的参数配置。我们提出WeAct（权重级内部动作），它在冻结的主干模型上训练独立的能力专用LoRA适配器，并将适配器选择添加到智能体的动作空间中。这些内部动作利用模型原生的工具调用接口，在共享上下文中按需组合能力。在八个深度研究基准测试中，配备WeAct的Qwen3-32B相比全量微调的对应模型，在使用更少计算量的情况下平均准确率提升6.2%，相比最强的静态方法提升27.5%。

    arXiv:2610.26847v2 Announce Type: replace-cross  Abstract: Long-horizon agents must learn heterogeneous capabilities from interleaved trajectories. Updating shared parameters to learn new capabilities can degrade previously acquired ones, resulting in catastrophic forgetting. As a trajectory unfolds, different stages may require different capabilities, yet conventional joint training and static adapter merging maintain a fixed parameter configuration throughout execution. We introduce WeAct (Weight-level Internal Actions), which trains independent capability-specific LoRA adapters over a frozen backbone and adds adapter selection to the agent's action space. These internal actions use the model's native tool-calling interface for on-demand capability composition in a shared context. Across eight deep-research benchmarks, Qwen3-32B with WeAct improves average accuracy by 6.2% relative to full fine-tuning counterparts despite using less compute, and by 27.5% relative to the strongest sta
    
[^160]: MemT-1.0：迈向记忆原生演化

    MemT-1.0: Towards Memory-Native Evolution

    [https://arxiv.org/abs/2610.26825](https://arxiv.org/abs/2610.26825)

    MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。

    

    记忆原生演化要求将记忆作为基础模型的内在能力，使其能够持续积累和利用信息。为此，我们提出了MemT-1.0，一种将记忆内化为模型计算一部分的记忆基础模型新架构。MemT-1.0基于Metis构建，采用Gated DeltaNet-2，并在每一轮内使用稳定的token级记忆更新机制。我们进一步引入Mixture-of-Storage（存储混合）来扩展原生记忆容量。在经过精心准备的主干模型之后，我们开发了一个由继续预训练和后训练组成的两阶段训练流程，并由340亿（34B）token的训练数据以及高效的训练与推理基础设施支撑。我们发布了包含MemT-1.0-Nano和MemT-1.0-Flash的模型家族，并对记忆能力和通用能力进行了全面评估。结果表明，它们在（各项评估中）优于参数化记忆基线。

    arXiv:2610.26825v2 Announce Type: replace  Abstract: Memory-native evolution requires making memory an intrinsic capability of foundation models, enabling them to continuously accumulate and utilize information. To this end, we introduce MemT-1.0, a new architecture of memory foundation models that internalize memory as part of model computation. Building on Metis, MemT-1.0 employs Gated DeltaNet-2 with a stable token-level memory update mechanism within each turn. We further introduce Mixture-of-Storage to scale native memory capacity. Following a deliberate backbone preparation, we develop a two-stage training pipeline consisting of continued pre-training and post-training, supported by 34B tokens of training data and efficient training and inference infrastructure. We release a model family comprising MemT-1.0-Nano and MemT-1.0-Flash, and conduct a comprehensive evaluation of memory and general capabilities. The results show that they outperform the parametric memory baselines on av
    
[^161]: HarnessPro：基于帕累托优化的递归式智能体框架自我改进方法，实现性能-成本权衡

    HarnessPro: Recursive Harness Self-Improvement with Pareto Optimization for Performance-Cost Trade-offs

    [https://arxiv.org/abs/2610.26770](https://arxiv.org/abs/2610.26770)

    HarnessPro提出了一种无需标签的智能体框架递归自我改进方法，通过帕累托优化权衡性能与执行成本，并提供精简、均衡、重型三种框架版本以适配不同的部署预算。

    

    智能体框架的开发在构建强大的AI智能体中起着至关重要的作用。近期的方法通过递归自我改进（RSI）来优化框架，但其性能提升往往伴随着更高的执行成本。我们提出了HarnessPro，这是一种无需标签的方法，用于演化智能体框架，在提升任务性能的同时降低执行成本。在该设置下，候选框架通过LLM生成的测试进行评估，无需访问真实标准答案或官方测试。专门的智能体负责诊断执行轨迹，根据不同的性能-成本优先级规划更新方案，并将其实现为候选框架。HarnessPro并非返回单一框架，而是采用归一化加权平方距离选择方法，为不同的部署预算提供侧重成本的精简版、均衡的中等版和侧重性能的重型版框架。被选中的框架进入下一轮迭代，执行反馈在此过程中持续引导改进……

    arXiv:2610.26770v2 Announce Type: replace-cross  Abstract: Harness development plays a crucial role in building capable AI agents. Recent approaches improve harnesses through recursive self-improvement (RSI), but their performance gains often come with higher execution costs. We introduce HarnessPro, a label-free method for evolving harnesses to improve task performance while reducing execution cost. In this setting, candidate harnesses are evaluated using LLM-generated tests without access to ground-truth solutions or official tests. Specialized agents diagnose execution trajectories, plan updates with different performance-cost priorities, and implement them as candidate harnesses. Rather than returning a single harness, HarnessPro uses normalized weighted squared-distance selection to provide cost-focused (lite), balanced (medium), and performance-focused (heavy) harnesses for different deployment budgets. Selected harnesses enter the next iteration, where execution feedback guides 
    
[^162]: SpatialScout：视觉语言模型智能体能否主动探索并构建3D场景地图？

    SpatialScout: Can VLM Agents Actively Explore and Map 3D Scenes?

    [https://arxiv.org/abs/2610.26659](https://arxiv.org/abs/2610.26659)

    提出了SpatialScout基准，用于评估视觉语言模型智能体在有限交互预算下，通过主动探索室内场景来构建物体级3D度量地图的能力。

    

    空间智能要求智能体主动获取视觉证据，并将观察结果整合为对三维世界的连贯表征。然而，空间问答任务对于模型能否通过自主探索构建准确的度量地图所提供的证据十分有限。我们提出了SpatialScout，一个用于评估视觉语言模型（VLM）智能体主动式物体级3D建图能力的基准。SpatialScout包含80个多样化的室内场景，涵盖3,140个目标物体。给定物体清单、RGB观测、相机标定信息和相机位姿，智能体控制相机探索每个场景，并在共享坐标系中估计物体中心位置。该任务要求智能体在有限的交互预算下协调视角选择、跨视图关联、几何估计和地图修正。评估通过物体尺度的重叠指标，同时衡量定位精度和目标覆盖率。

    arXiv:2610.26659v2 Announce Type: replace-cross  Abstract: Spatial intelligence requires agents to actively acquire visual evidence and integrate observations into a coherent representation of the three-dimensional world. Yet spatial question answering offers limited evidence of whether models can construct accurate metric maps through autonomous exploration. We introduce SpatialScout, a benchmark for evaluating vision-language model (VLM) agents on active object-level 3D mapping. SpatialScout comprises 80 diverse indoor scenes containing 3,140 target objects. Given an object inventory, RGB observations, camera calibration, and camera poses, agents control a camera to explore each scene and estimate object centers in a shared coordinate frame. The task requires coordinating viewpoint selection, cross-view association, geometric estimation, and map revision under bounded interaction budgets. Evaluation jointly measures localization accuracy and target coverage through object-scale overl
    
[^163]: 从结果到组件：面向LLM提出修复方案的反事实评估框架

    From Outcomes to Components: A Counterfactual Evaluation Framework for LLM-Proposed Repairs

    [https://arxiv.org/abs/2610.26650](https://arxiv.org/abs/2610.26650)

    该论文提出了一个反事实评估框架，将LLM提出的修复动作解耦为“目标记录选择”和“更新操作”两个组件，通过四种交叉执行方式分别评估各组件的独立贡献及其交互作用，从而超越单纯以成败论结果、更精细地评估LLM修复方案的质量。

    

    arXiv:2610.26650v2 公告类型： replace 摘要：大型语言模型（LLM）提出的修复动作指定了要更改哪些记录以及如何更改它们的值。一个失败的修复动作可能仍然包含有用的记录选择或有用的更新，这取决于两者是如何配对的。我们引入了一个反事实评估框架，将所选记录（目标）与所施加的更改（更新）分离开来。通过四种执行方式将提议的目标与更新同参考的目标与更新进行组合，以衡量纠正或保留每个组件的效果以及它们之间的交互作用。我们在两个任务上评估了该框架：交通分配任务（需要从道路计数中识别受影响的出行需求）和数据库修复任务（需要从聚合报告中识别受影响的行）。注入的故障提供了一个可逆转该故障的参考动作，恢复率则衡量一个动作相对于该参考动作的损失减少量。在交通实验中，对于稀疏目标的动作，保留……

    arXiv:2610.26650v2 Announce Type: replace  Abstract: A repair action proposed by a large language model (LLM) specifies which records to change and how to change their values. A failed action may still contain a useful choice of records or a useful update, depending on how the two are paired. We introduce a counterfactual evaluation framework that separates selected records (targets) from the applied change (an update). Four executions combine proposed and reference targets and updates to measure the effects of correcting or retaining each component and their interaction. We evaluate the framework on two tasks: traffic assignment, where affected demand must be identified from road counts, and database repair, where affected rows must be identified from aggregate reports. An injected fault supplies a reference action that reverses it, and the recovery rate measures an action's loss reduction relative to that reference. For sparse-target actions in the traffic experiments, retaining the 
    
[^164]: VibeTraining-Bench：面向迭代式系统改进的数据-训练器-推理框架协同设计

    VibeTraining-Bench: Data-Trainer-Harness Co-Design for Iterative System Improvement

    [https://arxiv.org/abs/2610.26342](https://arxiv.org/abs/2610.26342)

    提出了VibeTraining-Bench基准，用于评估AI智能体在固定资源预算下协同优化训练数据、训练器与推理框架以实现递归自我改进的能力，并通过测试数据隔离、容器分离和网络限制等手段防止奖励作弊。

    

    递归自我改进（RSI）要求智能体能够诊断系统瓶颈，并协调训练与推理层面的改进。我们将智能体联合优化训练数据、训练器和推理框架的能力称为跨组件协同设计。我们提出了VibeTraining-Bench，一个用于评估该能力的基准，涵盖终端操作、软件工程、多轮工具使用和数学推理等任务。在每个任务中，研究智能体从相同的初始智能体系统出发，在固定的资源预算下，依据验证反馈迭代地改进数据、训练器和推理框架，然后选择并提交最终的智能体系统以在测试集上进行评估。为防止奖励作弊，我们实施了三项保障措施：测试数据隔离、独立的研究与评分容器，以及针对研究智能体的网络限制。我们的评估结果显示……

    arXiv:2610.26342v2 Announce Type: replace  Abstract: Recursive self-improvement (RSI) requires agents to diagnose system bottlenecks and coordinate improvements across training and inference. We refer to the agents' capability to jointly optimize training data, the trainer, and the inference harness as cross-component co-design. We introduce VibeTraining-Bench, a benchmark for evaluating this capability across tasks spanning terminal operation, software engineering, multi-turn tool use, and mathematical reasoning. Within each task, research agents start from an identical initial agent system under a fixed resource budget. Guided by validation feedback, they iteratively refine data, trainers, and inference harnesses, then select and submit a final agent system for evaluation on the test set. To prevent reward hacking, we enforce three safeguards: test data isolation, separate research and grading containers, and network restrictions for research agents. Our evaluation results show that 
    
[^165]: 词元集合选择混淆POPE基准：VLM幻觉评估中是/否提取的系统性审计

    Token-Set Choice Confounds POPE: A Systematic Audit of Yes/No Extraction in VLM Hallucination Evaluation

    [https://arxiv.org/abs/2610.25964](https://arxiv.org/abs/2610.25964)

    该论文系统审计了POPE幻觉评估中的是/否答案提取方式，发现传统双词元logit读出所比较的"yes"/"no"词元ID在LLaVA-1.5-7B的实际输出中从未出现过（模型实际生成的是带空格前缀的" Yes"/" No"），仅这一读出实现细节就能使F1分数偏移高达6.13分，说明评估代码的实现选择会严重混淆基准对比结论。

    

    我们试图复现POPE基准上近期的一个幻觉缓解结果。我们测得的LLaVA-1.5-7B基线在对抗性分割上达到82.21 F1，比我们计划对比的对抗性基线（81.00、79.30和73.54）高出1.21至8.67分。我们检查了评估代码，发现一个仅凭自身就能改变F1分数的读出选择：双词元logit读出方式通过比较两个词表ID——3582（"yes"）和1217（"no"）——来判定是与否。但在跨越全部三个分割的9,000个贪心解码问题上，这两个ID一次也从未被生成过。模型实际输出的是“ Yes”（词元3869）和“ No”（词元1939），因为SentencePiece在提示词后缀"ASSISTANT:"之后会前置一个空格字节。改用覆盖带空格前缀表层形式的八字元查找来读取结果时，LLaVA-1.5-7B在对抗性、热门和随机分割上的F1分别为0.8221、0.8498和0.8713。仅这一对抗性分割上的6.13分偏移就超过了……（摘要原文在此处被截断）

    arXiv:2610.25964v2 Announce Type: replace-cross  Abstract: We set out to reproduce a recent hallucination-mitigation result on the POPE benchmark. The LLaVA-1.5-7B baseline we measured, 82.21 F1 on the adversarial split, sat 1.21 to 8.67 points above the adversarial baselines we planned to compare against (81.00, 79.30, and 73.54). We looked at the evaluation code and found a readout choice that moves F1 on its own. A two-token logit readout decides yes versus no by comparing two vocabulary IDs, 3582 ("yes") and 1217 ("no"). Across 9,000 greedy-decode questions spanning all three splits, neither ID is generated even once. The model emits " Yes" (token 3869) and " No" (token 1939), because SentencePiece prepends a space byte after the prompt suffix "ASSISTANT:". Read with an eight-token lookup that covers the space-prefixed surface forms, LLaVA-1.5-7B scores F1 0.8221, 0.8498, and 0.8713 on the adversarial, popular, and random splits. That 6.13-point adversarial shift exceeds the advers
    
[^166]: 何时更好的接受率才有回报？推测解码的对照研究

    When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding

    [https://arxiv.org/abs/2610.24993](https://arxiv.org/abs/2610.24993)

    该论文建立统一的训练-推理-评估对照协议，发现推测解码起草器之间在已发表结果中的接受率差距在相同条件下会大幅缩小，且接受率可通过近零成本的目标与上下文修改（如DCore强基线）或更昂贵的逐词元条件化来提升，只有当接受率提升的收益超过获取成本时才能带来吞吐量增益。

    

    推测解码的吞吐量提升取决于接受长度与执行延迟之间的权衡。我们探讨何时额外的接受率值得为其付出的成本。我们建立了一套涵盖训练、推理和评估的比较协议，重新测量了已发布的检查点，并在共享的目标模型生成数据上重新训练了六种起草器方案，覆盖十个基准测试和两个目标模型。对齐改变了格局：在已发表比较中显得很大的差距，在共享训练和推理条件下大幅缩小。这种缩小引出了两个更深层次的问题：接受率是以何种成本得到提升的，以及这些提升何时能改善吞吐量？我们发现，接受率可以通过几乎零成本的目标函数与上下文修改来提升，也可以通过更昂贵的逐词元条件化来提升。DCore 展示了前一条路线，它作为一个强大的基线，几乎没有额外的可测量起草开销。

    arXiv:2610.24993v2 Announce Type: replace  Abstract: The throughput gains of speculative decoding depend on the trade-off between acceptance length and execution latency. We ask when additional acceptance is worth the cost of obtaining it. We establish a comparison protocol spanning training, inference, and evaluation, remeasuring released checkpoints and retraining six drafter recipes on shared target-generated data across ten benchmarks and two target models. Alignment changes the picture: gaps that appear large in published comparisons shrink substantially under shared training and inference conditions. This narrowing motivates two deeper questions: how is acceptance improved at different costs, and when do these gains improve throughput? We find that acceptance can be improved through nearly free objective and context changes or more expensive token-to-token conditioning. DCore illustrates the former route as a strong baseline with little additional measured drafting overhead. The 
    
[^167]: Stable Signer V3：4D可变形图元泼溅是手语生成所需的全部

    Stable Signer V3: 4D Deformable Primitive Splatting Is All You Need for Sign Language Production

    [https://arxiv.org/abs/2610.24926](https://arxiv.org/abs/2610.24926)

    Stable Signer V3基于4D可变形图元泼溅技术，通过共享面部图元泼溅、测试时身体图元泼溅和隧穿超分辨率三项创新，显著提升了手语虚拟形象的表情丰富度、身体变形准确性和高分辨率细节保真度。

    

    我们提出了Stable Signer V3，一个基于4D可变形图元泼溅（4D Deformable Primitive Splatting）构建的交互式高分辨率手语虚拟形象系统。它解决了现有虚拟形象的三个局限性：面部表情受限、身份特定的身体变形不准确、以及精细外观细节的丢失。共享面部图元泼溅（Shared Face Primitive Splatting）增加了可迁移的超二次曲面图元，以实现更丰富的表情控制；测试时身体图元泼溅（Test-Time Body Primitive Splatting）学习一种身份特定的虚拟肌肉层，以实现更逼真的运动；测试时隧穿超分辨率（Test-Time Tunneling Super-Resolution）使用TunnelingNet恢复身份特定的高分辨率细节。这两个测试时模块均针对每个手语者身份仅优化一次；之后推理完全基于存储的身份参数运行。它们共同将面部几何误差降低了71.4%，在十个身份上将身体边界误差降低了35.1%，并在留出评估中将高分辨率RGB重建误差降低了24.2%。

    arXiv:2610.24926v2 Announce Type: replace-cross  Abstract: We present Stable Signer V3, an interactive high-resolution sign language avatar system built on 4D Deformable Primitive Splatting. It addresses three limitations of existing avatars: restricted facial expression, inaccurate identity-specific body deformation, and loss of fine appearance details. Shared Face Primitive Splatting adds transferable superquadric primitives for richer expression control; Test-Time Body Primitive Splatting learns an identity-specific virtual muscle layer for more faithful motion; and Test-Time Tunneling Super-Resolution uses TunnelingNet to recover identity-specific high-resolution details. Both test-time modules are optimized once per signer identity; inference then runs entirely from the stored identity parameters. Together, they reduce facial geometry error by 71.4%, body-boundary error by 35.1% across ten identities, and high-resolution RGB reconstruction error by 24.2% on held-out evaluations, w
    
[^168]: 询问而非猜测：针对语音识别错误的定向澄清机制

    Ask, Don't Guess: Targeted Clarification for Speech Recognition Errors

    [https://arxiv.org/abs/2610.24809](https://arxiv.org/abs/2610.24809)

    提出PARDON框架，通过基于干净音频软监督训练的反事实增益评分器在线定位流式语音识别中可能出错的片段，请说话人重复该内容，并让大语言模型仅修正被选中的片段，从而实现“主动询问而非盲目猜测”的定向澄清式错误补救。

    

    在远场对话场景下的实时语音识别仍然存在显著错误。在流式自动语音识别（ASR）过程中识别潜在的错误片段，可以实现一种补救机制：请说话人在更清晰的声学条件下重复受影响的内容。我们提出了PARDON，一个用于在线错误定位的新框架，它能识别出可能从这种干预中受益的片段。我们的监督方法利用干净音频来估计在声学条件改善后识别器的输出可能发生怎样的变化。我们使用成对隐马尔可夫模型（pair-HMM）对含噪解码与干净解码进行概率对齐，从而获得软监督信号。这些软目标用于训练一个反事实增益评分器，该评分器结合一整套声学与语言特征，产生词元级别的分数，并进一步将这些分数聚合为片段，以实现定向补救。随后，大语言模型（LLM）利用重复内容的ASR转录结果，仅对被选定的片段进行修正。在70%的精确率下，PARDON实现了……（原文摘要在此处截断）

    arXiv:2610.24809v2 Announce Type: replace  Abstract: Speech recognition in real time in far-field conversational settings still incurs significant errors. Identifying potential error spans during streaming ASR permits a recourse mechanism where the speaker can be asked to repeat the affected content under cleaner acoustic conditions. We propose PARDON, a new framework for online error localization, that identifies spans likely to benefit from this intervention. Our supervision uses clean audio to estimate how the recognizer's output might change under improved acoustic conditions. We probabilistically align noisy and clean decodes using a pair-HMM to obtain soft supervision. These soft targets train a counterfactual gain scorer by combining a suite of acoustic and language features to produce token-level scores, that are further grouped into spans for targeted recourse. An LLM uses the ASR transcript of the repetition to revise only the selected span. At a precision of 70%, PARDON achi
    
[^169]: Muse Glimmer 30B：一个端侧智能体模型

    Muse Glimmer 30B: An On-Device Agentic Model

    [https://arxiv.org/abs/2610.24628](https://arxiv.org/abs/2610.24628)

    Muse Glimmer是一个开源的30B参数端侧智能体模型，集成了多步推理、可靠的工具使用和故障恢复能力，可在消费级硬件上本地运行且无需云端，并在智能体基准测试中超越Gemma4-31B和Qwen3.6-27B等同规模模型达到最先进性能。

    

    我们介绍了Muse Glimmer，这是一个开源的300亿参数因果语言模型，配备感知编码器，专为在消费级硬件上执行自主智能体任务而打造。Muse Glimmer将多步推理、可靠的工具使用和故障恢复能力集成到单一模型中，可在消费级设备上本地运行，无需依赖云端基础设施。该模型接受文本和图像输入并产生文本输出，训练数据涵盖一百多种语言，并被量化至约4比特精度以实现高效的单GPU推理。Muse Glimmer在智能体基准测试中取得了同规模模型中的最先进性能，在工具使用、多步规划和智能体任务完成方面，相比同规模模型Gemma4-31B和Qwen3.6-27B树立了新标准。我们以Apache 2.0许可证发布该模型，以促进功能强大、安全、本地化AI智能体的开源发展。

    arXiv:2610.24628v2 Announce Type: replace-cross  Abstract: We introduce Muse Glimmer, an open-source 30B parameter causal language model with perception encoder, purpose-built for autonomous agentic tasks on consumer hardware. Muse Glimmer integrates multi-step reasoning, reliable tool use and failure recovery into a single model that runs locally on consumer devices without requiring cloud infrastructure. The model accepts text and image inputs and produces text outputs, was trained on data from over a hundred languages, and is quantized to approximately 4-bit precision for efficient single-GPU inference. Muse Glimmer achieves state-of-the-art performance for its size class on agentic benchmarks, setting a new standard compared to the models of its size: Gemma4-31B and Qwen3.6-27B on tool use, multi-step planning, and agentic task completion. We release the model under the Apache 2.0 license to foster open-source development of capable, safe, local AI agents.
    
[^170]: 智能体强化学习中数据的扩展规律：定律与合成策略

    How Data Scales in Agentic Reinforcement Learning: Laws and Synthesis Strategies

    [https://arxiv.org/abs/2610.23946](https://arxiv.org/abs/2610.23946)

    该论文将智能体强化学习的训练数据分解为任务、环境和轨迹三个可独立扩展的维度，并拟合出各维度的缩放定律，以回答在给定合成预算下应优先扩展哪个数据维度这一关键问题。

    

    预训练的缩放定律将训练数据视为一个一维量：即token数量。智能体强化学习继承了这种标量式的抽象，但现代数据合成流水线如今能够以截然不同的成本独立生成任务、环境和轨迹，使数据变成了一个结构化的设计空间。这引出了一个预训练从未需要回答的问题：在给定的合成预算下，应该扩展哪一个维度？我们通过三项紧密关联的贡献对这一问题进行了端到端的解答。首先，我们将智能体数据分解为三个基数维度（任务、环境、轨迹），外加一个奖励密度控制变量，并在数学+代码、软件工程（SWE）、网页以及UI/工具等应用领域的不同大语言模型（LLM）上拟合了各维度的缩放关系。在我们测量的所有模型规模、模型家族、领域以及强化学习算法（PPO和GRPO）中，缩放指数……

    arXiv:2610.23946v2 Announce Type: replace-cross  Abstract: Pretraining scaling laws treat training data as a one-dimensional quantity: a token count. Agentic reinforcement learning (RL) inherits this scalar abstraction, but modern synthesis pipelines now generate \textit{tasks}, \textit{environments}, and \textit{trajectories} independently and at very different costs, turning data into a structured design space. This raises a question pretraining never had to answer: given a synthesis budget, \textit{which axis} should be scaled? We address this question end-to-end with three tightly coupled contributions. First, we decompose agentic data into three cardinality axes (task, environment, trajectory), plus a reward-density control variable, and fit per-axis scaling relations on different large language models (LLMs) across math+code, SWE, web, and UI/tool application domains. Across all model sizes, model families, domains, and RL algorithms (PPO and GRPO) we measured, the scaling expone
    
[^171]: DiscoveryRSI：递归学习以发现有价值的假设

    DiscoveryRSI: Recursively Learning to Discover Valuable Hypotheses

    [https://arxiv.org/abs/2610.23839](https://arxiv.org/abs/2610.23839)

    该论文提出 DiscoveryRSI 框架，通过多智能体进化搜索构建假设树，并借助假设信用优化、测试框架回放和再发现三种机制，实现对有价值科学假设发现过程的递归式持续学习。

    

    科学发现需要将有价值的假设发展为切实可行的研究，然而从这一不断演进的研究过程中学习仍然充满挑战。我们提出了一个关于有价值且可检验的研究假设的操作性定义：即基于证据的、关于重大不确定性的论断，并辅以可行的调查研究、明确的决策标准和替代性解释。我们的 DiscoveryRSI 框架通过多智能体进化搜索构建假设树，将选定的研究编译为可执行的 Harbor 任务，并通过三种机制实现递归学习：假设信用优化将科学评估归因于有记录的研究贡献；测试框架回放识别高效策略与反复出现的失败，以改进研究工作流；再发现机制将已被接受的改进应用于新的研究，为后续学习生成更多经验。我们还引入了……（原文摘要在此处截断）

    arXiv:2610.23839v1 Announce Type: new  Abstract: Scientific discovery requires developing worthwhile hypotheses into practical investigations, yet learning from this evolving research process remains challenging. We introduce an operational definition of valuable and testable research hypotheses: evidence-grounded claims about consequential uncertainties, coupled with feasible investigations, explicit decision criteria, and alternative explanations. Our DiscoveryRSI framework develops hypothesis trees through multi-agent evolutionary search, compiles selected investigations into executable Harbor tasks, and recursively learns through three mechanisms: hypothesis credit optimization attributes scientific assessments to documented research contributions; harness replay identifies productive strategies and recurring failures to refine research workflows; and rediscovery applies accepted improvements to new investigations, generating further experience for subsequent learning. We also intr
    
[^172]: MultiAgent-OPD：基于在线策略蒸馏的层次化多智能体系统端到端训练

    MultiAgent-OPD: End-to-End Training of Hierarchical Multi-Agent Systems with On-Policy Distillation

    [https://arxiv.org/abs/2610.23234](https://arxiv.org/abs/2610.23234)

    提出MultiAgent-OPD方法，通过将在线策略蒸馏与强化学习相结合，利用冻结的教师模型为搜索者提供局部过程监督，实现了层次化多智能体系统的端到端联合训练。

    

    在层次化多智能体系统中，由领导者分配任务、子智能体执行任务。联合训练领导者和子智能体十分困难，因为最终答案只能为子智能体提供间接反馈。虽然大语言模型评判器可以为子智能体的表现打分，但这些分数依赖于评判器自身对子智能体回答质量的评估。我们提出了MultiAgent-OPD，该方法将在线策略蒸馏与强化学习相结合，端到端地联合训练共享参数的领导者-搜索者-回答者团队。我们首先使用答案正确性在搜索者-回答者工作流中训练一个教师模型。在团队训练过程中，冻结的教师模型对学生生成的搜索和总结token进行评分，为搜索者提供局部过程监督，而领导者从团队成功中学习，回答者从候选答案的正确性中学习。在七个数据集上，训练后的系统达到了38.68%的macro avg@4，超越了提示的链式方法（原文在此处截断）。

    arXiv:2610.23234v1 Announce Type: new  Abstract: In a hierarchical multi-agent system, a Leader assigns tasks and subagents carry them out. Jointly training the Leader and the subagents is difficult because a final answer gives only indirect feedback to the subagents. An LLM judge can score the performance of the subagents, but these scores depend on the judge's own assessment of the quality of subagents' responses. We present MultiAgent-OPD, which combines on-policy distillation with reinforcement learning to jointly train a shared-parameter Leader-Searcher-Answerer team end to end. We first train a Teacher in a Searcher-Answerer workflow using answer correctness. During team training, the frozen Teacher scores student-generated search and summary tokens to provide local process supervision for the Searchers, while the Leader learns from team success and the Answerer from candidate correctness. Across seven datasets, the trained system reaches 38.68% macro avg@4, exceeding Prompted Cr
    
[^173]: SAFE-Mem：具有细粒度权限与治理感知检索的安全智能体记忆

    SAFE-Mem: Secure Agentic Memory with Fine-Grained Authority and Governance-Aware Retrieval

    [https://arxiv.org/abs/2610.22959](https://arxiv.org/abs/2610.22959)

    提出SAFE-Mem框架，通过为记忆声明标注细粒度治理标签并结合治理感知检索，在保障LLM智能体记忆安全性的同时保留合法的记忆使用。

    

    大语言模型（LLM）智能体日益依赖持久化记忆来支持长期推理与个性化交互。然而，当从外部来源收集的不可信声明（即原子性陈述）随后被当作经授权的偏好、策略或权限使用时，这些信息可能带来安全风险。现有的防御手段通常依赖于基于内容的过滤或粗粒度的来源追踪，它们要么无法阻止未经授权的记忆使用，要么会屏蔽合法信息，从而牺牲了记忆的效用。为了解决这些局限，我们提出了SAFE-Mem，这是一种细粒度的记忆治理框架，能够在保留合法记忆使用的同时提升安全性。首先，SAFE-Mem为每条声明标注三种治理标签（即类型、权限和范围），以捕获其来源和允许的使用方式。其次，治理感知检索将查询需求与声明权限相匹配，以阻止未经授权的记忆使用（原文摘要到此中断）。

    arXiv:2610.22959v1 Announce Type: cross  Abstract: Large language model (LLM) agents increasingly rely on persistent memory to support long-term reasoning and personalized interactions. However, information collected from external sources can introduce security risks when untrusted claims (i.e., atomic statements) are later treated as authorized preferences, policies, or permissions. Existing defenses typically rely on content-based filtering or coarse-grained provenance tracking, which either fail to prevent unauthorized memory use or block legitimate information, sacrificing memory utility. To address these limitations, we propose SAFE-Mem, a fine-grained memory governance framework that improves security while preserving legitimate memory use. First, SAFE-Mem annotates each claim with three governance labels (i.e., kind, authority, and scope) to capture its provenance and permitted use. Second, governance-aware retrieval matches query requirements with claim permissions to block una
    
[^174]: 批量上下文强化 II：RLVR 需要长采样窗口吗？

    Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?

    [https://arxiv.org/abs/2610.22887](https://arxiv.org/abs/2610.22887)

    提出并发约束强化（CCR），将多个可独立验证的问题放入同一个受限完成中，使RLVR在8,000-token的短窗口下仍能保留正确性信号，并将每训练步的token消耗从约97.5万降至约17.5万，大幅降低训练成本。

    

    长采样窗口使得可验证奖励强化学习（RLVR）的代价高昂，但简单缩短窗口可能会丢失正确性信号：当每个单问题的采样在得出答案之前就被截断时，组相对准确率优势将不复存在。作为批量上下文强化（Batched Contextual Reinforcement, BCR）的技术延续，我们研究了并发奖励如何放宽这种对长窗口的依赖。并发约束强化将三个可独立验证的问题置于同一个有长度上限的训练完成中。在截断前完成的正确答案仍可获得奖励，而共享预算在无需显式长度惩罚的情况下，鼓励更简短的有效推理。我们将该机制与条件奖励方差和更新方差分析联系起来，并在4B和8B数学推理模型上进行了研究。与使用32k窗口的单问题GRPO相比，使用8,000-token窗口的CCR在4B模型上将每训练步的token数从975,464降至174,957。

    arXiv:2610.22887v1 Announce Type: cross  Abstract: Long rollout windows make reinforcement learning with verifiable rewards (RLVR) expensive, yet simply shortening them can remove the correctness signal: when every single-problem rollout truncates before an answer, group-relative accuracy advantages vanish. As a technical continuation of Batched Contextual Reinforcement (BCR), we study how concurrent rewards relax this dependence on long windows. Concurrent Constraint Reinforcement (CCR) places three independently verifiable problems in one capped training completion. Correct answers completed before truncation retain credit, while a shared budget encourages shorter useful reasoning without an explicit length penalty. We connect this mechanism to conditional reward- and update-variance analyses and study 4B and 8B mathematical reasoning models. Against single-problem GRPO with a 32k window, CCR with an 8,000-token window reduces tokens per training step from 975,464 to 174,957 on 4B an
    
[^175]: 微调前呈线性的知识在微调后能够泛化

    Knowledge That Is Linear Before Fine-Tuning Generalizes After It

    [https://arxiv.org/abs/2610.22300](https://arxiv.org/abs/2610.22300)

    论文提出区分语言模型中支持泛化的“活性事实”与仅可回忆的“惰性事实”，并通过重标注实验发现：微调前在模型内部呈线性可读的知识，在微调后能够泛化到共享相同答案的未训练实体上。

    

    并非语言模型能够回忆的所有事实都能同样好地支持进一步的学习。我们区分了“活性事实”与“惰性事实”：前者能够支持超出其自身回忆范围的泛化，后者虽然可以被回忆起来，但在给定任务中却不支持这种泛化。我们通过从已有关系构造新关系的方式来研究这一区分。以将地点映射到其所属大洲的关系（巴黎 → 欧洲）为例，为每个答案分配一个任意的新标签（欧洲 → 螃蟹，亚洲 → 斑马），于是新关系便将巴黎映射为螃蟹。我们在一部分主体上用新关系对模型进行微调，但不告诉模型原始关系或重新标注规则，然后测试模型能否为那些在原始关系中共享相同答案的留出主体分配相应的标签。也就是说，在学会了“巴黎 → 螃蟹”之后，模型能否回答出“慕尼黑 → 螃蟹”？在十一个语言模型和七十个关系的实验中，原始关系的线性可读程度……（摘要原文在此处截断）

    arXiv:2610.22300v1 Announce Type: cross  Abstract: Not all facts a language model can recall support further learning equally well. We distinguish active facts, which support generalization beyond their own recall, from inert facts, which can be recalled but do not support such generalization in a given task. We study this distinction by constructing new relations from existing ones. Take the relation that maps a place to its continent (Paris -> Europe) and assign each of its answers an arbitrary new label (Europe -> crab, Asia -> zebra). The new relation then maps Paris -> crab. We fine-tune the model on the new relation for a subset of subjects, without telling it the original relation or the relabeling rule, and test whether it assigns the corresponding labels to held-out subjects that share an answer in the original relation. Having learned Paris -> crab, does it answer Munich -> crab? Across eleven language models and 70 relations, how linearly readable the original relation is in
    
[^176]: 免训练的在策略蒸馏

    Training-Free On-Policy Distillation

    [https://arxiv.org/abs/2610.22236](https://arxiv.org/abs/2610.22236)

    本文提出免训练的在策略蒸馏方法，当多个专家模型共享同一基础模型时，仅需通过几百个采样样本的一次前向传播、以最小二乘方式逐层融合，即可在无需任何梯度训练的情况下将多个专家的能力聚合到一个模型中，且平均准确率超过任何单一专家模型。

    

    在策略蒸馏需要为每个教师模型进行采样、教师前向传播和梯度更新，才能将专家模型整合到一个学生模型中。我们证明这种整合其实不需要训练：仅通过前向传播，就可以将分别微调的多个专家模型的能力聚合到一个模型中，该模型平均而言比任何一个单独的专家模型都更加准确。当专家模型共享同一个基础模型时，蒸馏存在一个精确的逐层解：融合模型的每个线性层必须在专家模型所接收的输入上复现该专家的输出（同时读取其自身的答案），这是一个最小二乘问题，其解只需要这些输入的二阶矩。对几百个采样样本进行一次前向传播即可获得这些二阶矩。我们将这一方法称为免训练在策略蒸馏。相比求解器，采样样本中所包含的内容更为重要：在重复循环中被截断的回答在数学专家模型的采样中只占少数。

    arXiv:2610.22236v1 Announce Type: cross  Abstract: On-policy distillation consolidates specialist models into one student at the price of rollouts, teacher passes and gradient steps for every teacher. We show that the consolidation does not need training: with forward passes only, the capabilities of separately tuned specialists can be aggregated into one model that is, on average, more accurate than either specialist alone. When specialists share a base model, distillation has an exact layer-wise solution: each linear layer of the fused model must reproduce every specialist's output on the inputs that specialist receives while reading its own answers, a least-squares problem whose solution needs only the second moments of those inputs. One forward pass over a few hundred rollouts provides them. We call this training-free on-policy distillation (\method). What the rollouts contain matters more than the solver: answers truncated in repetition loops are few among a mathematics specialist
    
[^177]: 没有任何单一分词器特借能够可靠地预测语言模型的下游性能

    No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance

    [https://arxiv.org/abs/2610.21679](https://arxiv.org/abs/2610.21679)

    本研究通过分析约200个跨九种语言训练的语言模型，发现没有任何单一的分词器内在评价指标能够可靠地预测语言模型的下游性能，因此需要开发新的基于语言学的分词器评估方法。

    

    分词是语言建模流程中的一个关键组成部分，但人们对它的理解仍然不足。分词器会影响下游性能，但关于哪些分词器特征能够在无需完整训练的情况下预测出更好的语言模型，目前的证据并不一致。在本文中，我们分析了约200个在九种不同语言上训练的语言模型。我们对模型及其分词器均进行了评估，并检验了各种已有的分词器内在评价指标与模型性能（通过多种基准测试衡量）之间是否存在关联。尽管已有的分词器内在评价指标数量众多且种类多样，我们发现没有任何一个指标能够在不同模型家族、任务和语言之间一致地预测下游性能。这促使我们需要开发新的分词器评估指标和方法。我们提出了基于语言学的新型指标，本工作引入的新指标表明，新的指标可以……

    arXiv:2610.21679v1 Announce Type: new  Abstract: Tokenization is a critical component of the language modeling pipeline, but it is poorly understood. Tokenizers can impact downstream performance, but there is currently mixed evidence about which tokenizer features predict a better language model without requiring a full training run. In this paper, we analyze approximately 200 language models trained on nine different languages. We evaluate both models and their tokenizers and test whether there is a relationship between any proposed intrinsic tokenizer evaluations and model performance, as measured by a wide variety of benchmarks. Despite the quantity and variety of proposed intrinsic tokenizer metrics, we find that none consistently predicts downstream performance across model families, tasks, and languages. This motivates the need for novel tokenizer evaluation metrics and approaches. We propose novel linguistically grounded metric introduced in this work shows that new metrics can 
    
[^178]: HyCo：面向高性价比移动-云协同推测解码的混合草稿生成机制

    HyCo: Hybrid Drafting for Cost-Efficient Mobile-Cloud Speculative Decoding

    [https://arxiv.org/abs/2610.21436](https://arxiv.org/abs/2610.21436)

    HyCo提出混合草稿生成机制，将服务器模型的原生多词预测头部署到移动NPU上并与小型语言模型单次推理协同工作，配合量化和ε阈值松弛策略，使移动-云推测解码的服务器GPU时间降低约29%-39%。

    

    设备-云协同推测解码能够减少服务器的计算开销，但使用小型语言模型（SLM）生成深度草稿树需要在移动神经处理单元（NPU）上进行多次推理。我们提出了HyCo，它将草稿深度分配给在设备端执行的服务器模型原生多词预测（MTP）头，并将替代分支分配给单次SLM推理。服务器只需返回该MTP头所需的隐状态、输入嵌入和键值缓存条目，而无需执行其原有的顺序计算步骤。通过量化、受限词表和1K-token窗口设置，两个草稿模型均能适配NPU运行。此外，一种ε阈值规则进一步放宽了对验证器采样分布的精确保持要求。在通过LTE/5G网络连接Galaxy S26 Ultra的单客户端运行中，与服务器端MTP自推测相比，HyCo在Spec-Bench上将每个输出token的服务器GPU时间降低了38.6%，在32K-token摘要任务上降低了29.2%。

    arXiv:2610.21436v1 Announce Type: new  Abstract: Device-cloud speculative decoding reduces server computation, but growing a deep draft tree with a small language model (SLM) requires repeated passes on a mobile neural processing unit (NPU). We present HyCo, which assigns depth to the server model's native multi-token prediction (MTP) head, executed on the device, and alternative branches to one SLM pass. The server returns the hidden state, input embedding, and key-value entries needed by the head instead of executing its sequential steps. Quantization, a restricted vocabulary, and a 1K-token window fit both drafters to the NPU. An {\epsilon}-threshold rule additionally relaxes exact preservation of the verifier's sampling distribution. In single-client runs with a Galaxy S26 Ultra over LTE/5G, HyCo reduces server GPU time per output token relative to server-side MTP self-speculation by 38.6% on Spec-Bench and 29.2% on 32K-token summarization. With the same acceptance rule, three Spec
    
[^179]: 超越不变性：面向多模态学习的按因子选择性处理分布偏移

    Beyond Invariance: Factor-wise Selective Treatment of Shifts for Multimodal Learning

    [https://arxiv.org/abs/2610.21403](https://arxiv.org/abs/2610.21403)

    提出FaShift框架，超越传统不变性方法，通过按因子诊断分布变化、干预效应和跨环境稳定性，对多模态学习中的偏移信息进行选择性抑制、保留或削弱。

    

    分布偏移是鲁棒多模态学习面临的根本性挑战。现有方法通常通过抑制依赖环境的变异来促进不变性，从而隐式地对偏移信息施加统一处理。然而，偏移信息可能产生不同的预测后果：有些对预测贡献甚微，有些仍然可靠地包含信息，有些则表现出依赖环境的预测效应。这就引出了一个根本性问题：哪些偏移信息应该被抑制、保留或削弱？为了解决这一问题，我们提出了FaShift，一个按因子处理偏移的框架，它根据潜在表示因子的分布变化、干预效应和跨环境稳定性对其进行诊断。通过分解多模态表示并评估这些互补属性，FaShift能够区分非预测性因子、稳定预测性因子和预测性（因子）……

    arXiv:2610.21403v1 Announce Type: new  Abstract: Distribution shifts pose a fundamental challenge to robust multimodal learning. Existing methods typically promote invariance by suppressing environment-dependent variations, implicitly applying a uniform treatment to shifted information. However, shifted information can have different predictive consequences: some contributes little to prediction, some remains reliably informative, and some exhibits environment-dependent predictive effects. This raises a fundamental question: which shifted information should be suppressed, preserved, or attenuated? To address this question, we propose FaShift, a factor-wise shift framework that diagnoses latent representation factors according to their distributional changes, intervention effects, and cross-environment stability. By factorizing multimodal representations and evaluating these complementary properties, FaShift distinguishes non-predictive factors, stably predictive factors, and predictive
    
[^180]: SoK：开放式法律文本答案的评分——从法学院考试到可信赖的评分规则裁判

    SoK: Grading Open-Text Legal Answers, from Law School Exams to Trustworthy Rubric Judges

    [https://arxiv.org/abs/2610.19992](https://arxiv.org/abs/2610.19992)

    本文系统化研究了开放式法律文本答案（如法学院考试）的评估方法，从教育测量与法律规范中提炼出八项评估标准并对比八种评估范式，发现针对任务定制评分规则是让LLM裁判与人类评分者达成一致的最关键手段。

    

    法学学生通过开放式文本答案接受考核：案例分析、论文和法律备忘录，将法律适用于一组特定事实。这类答案的评分成本高昂，人类评分者之间分歧很大，而大型语言模型（LLM）如今可以在教学中以及测试法律AI的基准评测中大规模进行评分。本文系统化地研究了如何评估开放式法律文本答案。我们从教育测量学以及规范考试的法律中推导出八项必要条件，其中包括效度、信度、说理依据、对可辩护的替代解答给予得分、对投机取巧行为的鲁棒性以及人类监督。随后，我们将八种评估范式与这些标准进行对比，范围涵盖从非结构化的人工评分，到采用针对具体实例的评分规则的LLM裁判，再到成对比较。有三个发现尤为突出。为任务量身编写的评分规则是使LLM裁判与人类评分者达成一致的最强杠杆，但所报告的数字回答的是不同的问题……（原文摘要至此截断）

    arXiv:2610.19992v1 Announce Type: new  Abstract: Law students are examined through open-text answers: case solutions, essays and memos that apply the law to a set of facts. Such answers are expensive to grade, human graders disagree widely on them, and large language models (LLMs) now offer to grade them at scale, both in teaching and in the benchmarks that test legal AI. This paper systematizes how open-text legal answers can be evaluated. We derive eight desiderata from educational measurement and from the law that governs exams, among them validity, reliability, reasons, credit for defensible alternative solutions, robustness to gaming and human oversight. We then compare eight evaluation paradigms against them, from unstructured human grading to LLM judges with instance-specific rubrics and pairwise comparison. Three findings stand out. A rubric written for the task is the strongest lever for agreement between an LLM judge and human graders, but the reported figures answer differen
    
[^181]: ELARSpeech：一个覆盖675种代表性不足语言、时长3万小时的长尾多语言语音技术数据集

    ELARSpeech: A 30K-Hour Dataset Across 675 Underrepresented Languages for Long-Tail Multilingual Speech Technologies

    [https://arxiv.org/abs/2610.19388](https://arxiv.org/abs/2610.19388)

    ELARSpeech是一个包含675种代表性不足语言、总时长30,202小时的大规模多语言语音数据集，为将语音识别与语音合成技术扩展至语言长尾领域提供了重要资源和测试平台。

    

    近年来，大规模多语言语音建模取得了显著进展，然而现有的训练资源和基准测试仍然未能充分覆盖语言长尾中的大部分语言。为了弥补这一差距，我们推出了ELARSpeech，这是一个大规模多语言语音资源，包含来自675种代表性不足语言的30,202小时语音数据，整理自濒危语言档案馆。该语料库结合了人工标注和未标注的录音，涵盖了多样的录音条件、应用领域和语系。除了语料库构建之外，我们的实验展示了ELARSpeech如何支持将自动语音识别（ASR）和语音合成（TTS）能力扩展到多种代表性不足的语言，并且通过对未标注数据进行伪标注可以带来进一步的收益。与此同时，我们的语音表征分析揭示了将现有多语言模型迁移到语言长尾时仍存在的局限性。总而言之，ELARSpeech既提供了一份宝贵的资源，也提供了一个测试平台。

    arXiv:2610.19388v1 Announce Type: new  Abstract: Massively multilingual speech modeling has made remarkable progress in recent years, yet existing training resources and benchmarks still underrepresent much of the linguistic long tail. To address this gap, we introduce ELARSpeech, a large-scale multilingual speech resource comprising 30,202 hours of speech across 675 underrepresented languages, curated from the Endangered Languages Archive. Combining human-annotated and unlabeled recordings, the corpus spans diverse recording conditions, domains, and language families. Beyond corpus construction, our experiments show how ELARSpeech supports extending ASR and TTS capabilities to diverse underrepresented languages, with further benefits from pseudo-labeling unlabeled data. At the same time, our speech representation analysis reveals remaining limitations in transferring existing multilingual models to the linguistic long tail. Together, ELARSpeech provides both a resource and a testbed f
    
[^182]: 论自然语言自编码器的可监控性：识别与判别

    On the Monitorability of Natural Language Autoencoders: Identification vs Discrimination

    [https://arxiv.org/abs/2610.18727](https://arxiv.org/abs/2610.18727)

    论文将自然语言自编码器的训练与监督部署形式化为两种不同的指称博弈，并指出二者之间的差距导致其解释虽能识别对应的激活，却难以让人类监控者从多个近似叙述中辨别出真实的那一个。

    

    自然语言自编码器（NLA）被提出作为语言模型的监督工具：激活言语化器（AV）为激活向量撰写英文解释，而激活重构器（AR）将解释映射回向量。它们在监督中的价值取决于人类监控者在给定AV解释的情况下，能否将一段对激活的真实叙述与虚假叙述区分开来。我们将NLA的训练与NLA监督部署形式化为源自涌现通信领域的两种不同的指称博弈：训练是一种生成式指称博弈，而监督是一种判别式的、多模态的、部分可观测的指称博弈，该博弈在受监控语言模型的激活与真实描述它们的叙述之间的多对多关系上展开。这两种博弈之间的差距预示着AV的解释将能够识别其所描述的是哪个激活，却无法将其真实叙述与紧密匹配的叙述区分开来。

    arXiv:2610.18727v2 Announce Type: replace-cross  Abstract: Natural Language Autoencoders (NLAs) are proposed as oversight tools for language models: an Activation Verbaliser (AV) writes an English explanation of an activation vector, and an Activation Reconstructor (AR) maps the explanation back to a vector. Their value for oversight depends on a human monitor being able to tell a true account of an activation from a false one, given the AV's explanation. We formalise NLA training and NLA oversight deployment as two different referential games from Emergent Communication: training is a generative referential game, while oversight is a discriminative, multi-modal, partially-observable referential game played over a many-to-many relation between the monitored language model's activations and the accounts that are true of them. The gap between the two games predicts that AV's explanations will identify which activation they describe without discriminating its true account from closely mat
    
[^183]: 多步归一：推理收缩与LLM智能体安全记忆复用的极限

    Many Steps into One: Reasoning Contraction and the Limits of Safe Memory Reuse for LLM Agents

    [https://arxiv.org/abs/2610.17544](https://arxiv.org/abs/2610.17544)

    论文提出“推理收缩”概念，证明LLM智能体中基于成败计数的记忆更新机制是糟糕的变化检测器（比认证记忆多产生7-73倍的过时答案），并揭示了安全记忆复用存在由“认证负载”这一指标决定的理论上限。

    

    思维模板、技能库、智能体记忆和缓存将许多推理步骤转化为一步，我们将这种现象称为“推理收缩”。收缩后的记忆能够节省推理步骤并避免错误，直到其底层事实发生变化，随后便会悄然给出错误答案。大多数系统通过统计成功与失败的次数来信任记忆。我们证明这类计数器是调校失当的变化检测器：对称计数器只有在经历其误报时间跨度的幂次之后才能察觉部分变化，而未截断的计数器可能永远无法察觉。在五个LLM回答基于被编辑事实的多跳问题的实验中，在相同信息条件下，ExpeL和ACE风格的计数器比经过认证的记忆多提供7.0-13.5倍的过时答案；而在带有精确验证器的代码任务上，这一比例达到35-73倍。安全性也是有代价的。当事实发生变化且每个记忆仅由其自身经过验证的结果来评判时，一个数字——“认证负载”——决定了任何安全记忆最多能跳过多少推理，并且在Zipf需求分布下，它产生……

    arXiv:2610.17544v1 Announce Type: cross  Abstract: Thought templates, skill libraries, agent memories and caches turn many reasoning steps into one. We call this reasoning contraction. A contracted memory saves reasoning steps and errors until its facts change, then quietly answers wrong. Most systems trust memories by counting successes and failures. We prove that such counters are badly tuned change detectors: a symmetric counter notices a partial change only after a power of its false-alarm horizon, and an unclipped one may never notice it. On five LLMs answering multi-hop questions over edited facts, ExpeL- and ACE-style counters serve 7.0-13.5x more stale answers than certified memories given the same information, and on code tasks with an exact verifier 35-73x more. Safety also has a price. When facts change and each memory is judged only by its own verified outcomes, one number, the certification load, sets how much reasoning any safe memory can skip, and under Zipf demand it yi
    
[^184]: StartLux-Decision：基于强化学习的AutoThink决策模型，实现校准的审慎决策

    StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions

    [https://arxiv.org/abs/2610.17309](https://arxiv.org/abs/2610.17309)

    该论文提出StartLux-Decision系列决策模型（0.8B至27B参数），通过置信度引导的自适应推理机制与全新的RLCD²强化学习方法（利用冻结模型和适当评分规则奖励校准的决策概率质量、惩罚过长推理），实现了高效单次预测与按需深度推理的结合。

    

    我们推出了StartLux-Decision，这是一系列参数规模从0.8B到27B的决策模型，将高效的单次预测与自适应的、置信度引导的推理相结合。Instant模型在完整注意力层中使用双向注意力来提升决策精度，同时保持单次推理。AutoThink模型则对不确定的决策调用显式推理，而对有把握的案例直接给出答案。三阶段训练流程先建立共享的决策能力，再将两个变体专门化，并通过推理监督和强化学习为AutoThink配备推理能力。我们提出了用于校准审慎决策的强化学习方法（RLCD²），该方法使用一个冻结模型和适当的评分规则来奖励决策概率质量的提升，并对过长的推理轨迹施加软惩罚。RLCD²提高了在生成限制内终止的推理轨迹的比例……

    arXiv:2610.17309v1 Announce Type: cross  Abstract: We introduce StartLux-Decision, decision models from 0.8B to 27B parameters that combine efficient single-pass prediction with adaptive, confidence-guided reasoning. The Instant models use bidirectional attention in full-attention layers to improve decision accuracy while keeping single-pass inference. The AutoThink models invoke explicit reasoning for uncertain decisions and answer confident cases directly. A three-stage training pipeline establishes shared decision capabilities, specializes the two variants, and equips AutoThink with reasoning through reasoning supervision and reinforcement learning. We introduce Reinforcement Learning for Calibrated Deliberative Decisions (RLCD$^2$), which uses a frozen model and proper scoring rules to reward improvements in decision probability quality and applies a soft penalty to overlong reasoning traces. RLCD$^2$ raises the share of reasoning traces that terminate within the generation limit b
    
[^185]: PersonaFactor：基于动态选择的用于大语言模型个性化的偏好因子

    PersonaFactor: Preference Factors with Dynamic Selection for LLM Personalization

    [https://arxiv.org/abs/2610.16372](https://arxiv.org/abs/2610.16372)

    PersonaFactor框架从异构用户交互中提炼出十三个可复用、可解释的个性化偏好因子，并通过动态因子选择机制，在四个个性化基准和四个骨干模型的所有指标上均取得最佳性能。

    

    虽然用户在不同任务中可能表现出不同的个性化需求，但这些特定任务的行为可以反映出共享的底层偏好机制。现有的LLM个性化方法通常只关注表面层面的交互或未加区分的历史记录，忽视了这些行为背后的共享机制。我们提出了PersonaFactors，这是一个从异构用户交互中提炼可复用、可解释个性化因子的框架。我们通过提取基于证据的理据并对跨任务反复出现的模式进行聚类，构建了一个包含十三个PersonaFactors的因子库，并开发了一个动态因子引导的生成框架，可根据每个用户的历史记录和当前请求选择相关因子。在四个个性化基准和四个骨干模型上，Dynamic PersonaFactor在所有报告的指标上都取得了最佳成绩。表示分析进一步表明，PersonaFactors与用户（摘要在此处截断）

    arXiv:2610.16372v2 Announce Type: replace  Abstract: Although users may express different personalization needs across tasks, these task-specific behaviors can reflect shared underlying preference mechanisms. Existing LLM personalization methods often focus on surface-level interactions or undifferentiated histories, overlooking the shared mechanisms underlying these behaviors. We introduce PersonaFactors, a framework that distills reusable and interpretable personalization factors from heterogeneous user interactions. We construct a library of thirteen PersonaFactors by extracting evidence-grounded rationales and clustering recurring patterns across tasks, and develop a dynamic factor-guided generation framework that selects factors relevant to each user's history and current request. Across four personalization benchmarks and four backbone models, Dynamic PersonaFactor achieves the best score on every reported metric. Representation analyses further show that PersonaFactors and user 
    
[^186]: 脉冲语言模型的计算最优缩放定律

    Compute-Optimal Scaling Laws for Spiking Language Models

    [https://arxiv.org/abs/2610.15146](https://arxiv.org/abs/2610.15146)

    本文首次对脉冲语言模型开展Chinchilla式缩放研究，发现其计算最优缩放指数（0.56）与密集Transformer（0.50）接近，且脉冲化带来的损失代价随计算预算增加而稳步下降。

    

    脉冲神经网络通过稀疏的二值激活实现节能的语言建模，但脉冲语言模型在大规模下的效率-精度权衡尚未被刻画。我们首次对脉冲语言模型进行了Chinchilla式的IsoFLOP缩放研究，在FineWeb-EDU数据集上，于3×10¹⁷至10¹⁹ FLOPs的计算预算范围内，对SpikeGPT系列模型（采用RWKV骨干，宽度从384到1536）进行了系统扫描。我们报告了两个发现：首先，计算最优前沿为 N_opt ∝ C^0.56，接近Hoffmann等人报告的密集Transformer的0.50，因此脉冲化并不改变在给定计算预算下参数与数据应如何权衡；其次，与以相同方式训练的非脉冲RWKV和Transformer基线相比，损失差距可分解为不随计算量变化的约0.16 nats的骨干成本，以及随计算量稳步下降的脉冲成本（从0.34降至0.23 nats）。

    arXiv:2610.15146v2 Announce Type: replace-cross  Abstract: Spiking neural networks target energy-efficient language modeling through sparse, binary activations, but the efficiency-accuracy tradeoff of spiking language models is uncharacterized at scale. We present the first Chinchilla-style IsoFLOP scaling study of a spiking language model, sweeping SpikeGPT-family models (RWKV backbone) from width 384 to 1536 across compute budgets from $3\times10^{17}$ to $10^{19}$ FLOPs on FineWeb-EDU. We report two findings. First, the compute-optimal frontier is $N_{opt} \propto C^{0.56}$, close to the 0.50 that Hoffmann et al. report for dense Transformers, so spiking does not change how parameters and data should be traded off under a compute budget. Second, against non-spiking RWKV and Transformer baselines trained the same way, the loss gap splits into a backbone cost of about 0.16 nats that does not change with compute and a spiking cost that falls steadily with compute, from 0.34 to 0.23 nat
    
[^187]: 当激活引导被用作记忆时，什么能够存留

    What Survives When Activation Steering Is Used as Memory

    [https://arxiv.org/abs/2610.14958](https://arxiv.org/abs/2610.14958)

    该研究提出一种免训练的激活记忆机制，并证明激活引导偏移作为注意力效应的固定平均值只能存留与问题无关的内容：这种记忆虽具有约98%的高选择性，但用户事实仅被存储在最后几层且几乎从不被使用，表明用户事实与偏好仍需以文本形式记忆。

    

    激活引导（activation steering）近来被用作技能的记忆：智能体过去的经验被存储为引导向量，以供日后检索。而关于用户的事实与偏好则被认为需要以文本形式存储。我们在一个冻结的大语言模型旁搭建了一个免训练的激活记忆来检验这一假设：一次经验在模型后层状态中引起的变化被存储为一个偏移量（shift），按对话主旨的白化键值归档，且仅在后续会话中主旨匹配时才被重新加回。我们论证，这样的偏移量本质上是注意力在上下文中针对每个问题重新计算的效应的固定平均值，因此只有与问题无关的内容才能存留。在一项针对 Qwen3.5-2B 和 9B 的预注册研究中，采用来自四个模型家族的评判模型，该记忆表现出选择性（约98%的无关回复保持不变），但事实被存储为“该说的词”，仅存在于最后几层，且几乎从未被使用（干净使用率至多15%，相比之下文本记忆为96–98……原文摘要在此截断）。

    arXiv:2610.14958v1 Announce Type: new  Abstract: Activation steering has recently been used as a memory for skills: past experiences of an agent are stored as steering vectors and retrieved later. Facts and preferences about the user are assumed to need text instead. We test that assumption with a training-free activation memory beside a frozen LLM. The change an experience causes in late-layer states is stored as a shift, filed under a whitened key for the conversation's gist, and added back in a later session only when the gist matches. We argue that such a shift is a fixed average of an effect that attention, in context, recomputes for every question, so only question-independent content can survive. In a pre-registered study on Qwen3.5-2B and 9B, with judges from four model families, the memory is selective (about 98% of unrelated replies are unchanged), but facts are stored as the word to say, in the last layers only, and are almost never used (at most 15% clean use, against 96-98
    
[^188]: LongHisDoc：面向中文长篇历史文献理解的综合性基准

    LongHisDoc: A Comprehensive Benchmark for Chinese Long Historical Document Understanding

    [https://arxiv.org/abs/2610.14531](https://arxiv.org/abs/2610.14531)

    该论文提出了首个针对中文长篇历史文献理解的基准LongHisDoc，包含来自101份PDF文档的1,012个专家标注问答对，评估发现当前最先进的模型准确率均不足70%，揭示了该领域仍有巨大提升空间。

    

    中文历史文献是宝贵的文化瑰宝，但其复杂性给非专业人员带来了显著的理解挑战。尽管大语言模型和大视觉语言模型在现代文档任务中取得了成功，但它们在长篇（整本书）历史文献理解方面的能力仍在很大程度上未被探索。为了弥合这一差距，我们推出了LongHisDoc，这是首个专门针对长篇历史文献理解设计的基准。LongHisDoc包含1,012个由专家标注的问答对，这些问答对来自101份PDF文档，每份文档平均85.1页、42,776.4个token。它涵盖10个类别，并包含四种问答类型：单页、跨页、多跳和不可回答。回答这些问题需要整合来自文本、版面和视觉元素的多模态证据。我们的评估显示，表现最佳的LVLM和LLM分别仅达到62.9%和69.9%的准确率，这

    arXiv:2610.14531v2 Announce Type: replace  Abstract: Chinese historical documents are invaluable cultural treasures, but their complexity poses significant comprehension challenges for non-specialists. Despite the success of Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) in modern document tasks, their capabilities in Long (full-book) Historical Document understanding remain largely unexplored. To bridge this gap, we introduce LongHisDoc, the first benchmark tailored for long historical document understanding. LongHisDoc comprises 1,012 expert-annotated QA pairs derived from 101 PDF documents, with an average of 85.1 pages and 42,776.4 tokens per document. It spans 10 categories and features four QA types: single-page, cross-page, multi-hop, and unanswerable. Answering these questions requires integrating multimodal evidence from text, layout, and visual elements. Our evaluation reveals that the top-performing LVLM and LLM achieve only 62.9% and 69.9% accuracy, r
    
[^189]: 面向长程终端智能的智能体数据规模化扩展

    Scaling Agentic Data for Long-Horizon Terminal Intelligence

    [https://arxiv.org/abs/2610.13166](https://arxiv.org/abs/2610.13166)

    提出 TerminalHorizon，一个面向长程终端智能体的全自动化数据合成引擎，通过避免环境合成预先完成关键工作，生成能真正激发智能体探索、实施与验证能力的高质量长程训练数据。

    

    终端智能体正在从本地操作迈向完成真实世界的专业任务。短轨迹虽然能够展示命令和工具调用，但无法充分体现智能体如何通过探索积累证据、利用证据辅助判断、根据反馈持续调整策略，以及将中间结果转化为最终交付成果。高质量的长程轨迹能够在单一问题求解过程中捕捉这些智能体行为模式及其内在联系，为持续性自主工作提供更丰富的经验。然而，环境合成可能预先完成或简化这些行为模式背后的关键工作，从而消除了智能体本应执行的探索、实施和验证环节。为解决这一问题，我们提出了 TerminalHorizon，一个面向长程终端智能体的全自动化数据合成引擎。该方法从真实世界的专业（任务出发），……

    arXiv:2610.13166v1 Announce Type: new  Abstract: Terminal agents are moving beyond local operations toward completing real-world professional tasks. Short trajectories can demonstrate commands and tool calls, but they do not fully show how agents accumulate evidence through exploration, use it to inform judgments, continually adjust strategies based on feedback, and turn intermediate results into final deliverables. High-quality long-horizon trajectories capture these agentic behavioral patterns and their connections within a single problem-solving process, providing richer experience for sustained autonomous work. However, environment synthesis may complete or simplify the key work underlying these patterns in advance and thus remove the need for exploration, implementation, and verification that the agent was meant to perform. To address this, we introduce TerminalHorizon, a fully automated data synthesis engine for long-horizon terminal agents. Starting from real-world professional 
    
[^190]: 不可自判定：大语言模型无法划定Agent验证器可检验内容的边界

    Not Self-Decidable: LLMs Cannot Draw the Boundary of What an Agent Verifier Can Check

    [https://arxiv.org/abs/2610.04699](https://arxiv.org/abs/2610.04699)

    研究通过对六个语料库、约22,000个标签的实证分析表明，大语言模型无法自行判定Agent验证器在面对外部监管规则时哪些内容可以自动检查、哪些需要人工裁决，因此“可自判定”这一升级方案的核心假设不成立。

    

    针对Agent的验证器面临两类规则：一类是固定检查可以判定的，另一类是需要裁判来裁决的。如果一个团队自行推导其检查规则，那么这个划分在事前就固定了。但在金融、医疗和法律等领域，需求来自外部，Agent执行的是它自己并未编写的规则，因此这一划分只能留到运行时解决，并且对每条规则的每个谓词、在每次动作上都要重复判断，其频率之高是任何审查者都无法审计的。每一种升级（escalation）方案都假设模型能够自行做出这一决定，即这一问题是“可自判定的”（self-decidable）。我们在六个语料库上——包括欧盟《人工智能法案》、FINRA指南以及一个已部署的信贷Agent——从三个实验室的四个模型中收集了约22,000个标签。研究发现：在答案显而易见的地方，各模型几乎完美一致；但在监管文本上它们的表现集体崩溃；它们的错误方向相反，因此没有任何模型可以被信任为保守的选择；而且在已部署Agent自身的规则集上，它们会一起犯错，过度声称……

    arXiv:2610.04699v2 Announce Type: replace  Abstract: A verifier for an agent faces rules of two kinds: the ones a fixed check can settle and the ones that require a judge. A team that derives its own checks fixes that split up front. Where the requirements come from outside, as in finance, healthcare and law, the agent enforces rules it did not write, so the split falls to runtime, recurring for every predicate of every rule on every action at a rate no reviewer can audit. Every escalation scheme assumes a model can make that decision itself, that it is self-decidable. Across six corpora, including the EU AI Act, FINRA guidance and a deployed credit agent, we collect roughly 22,000 labels from four models built by three labs. They agree almost perfectly where the answer is obvious and collapse on regulatory text; their errors run in opposite directions, so no model can be trusted as the conservative choice; and on the deployed agent's own rule-set they err together, over-claiming that 
    
[^191]: Penumbra（半影）：面向监管义务的样本高效对抗搜索

    Penumbra: Sample-Efficient Adversarial Search for Regulatory Obligations

    [https://arxiv.org/abs/2610.04693](https://arxiv.org/abs/2610.04693)

    提出Penumbra，一种样本高效的对抗搜索方法，通过从经验证的锚点出发自适应地分配编辑预算，找到使监管合规判定翻转的最小编辑相邻回复对，从而以更少的探测成本最大化覆盖监管义务的“失效面”。

    

    智能体正在进入金融、医疗和法律领域，在这些领域中，违规行为不会留下词汇层面的痕迹，却会带来真实的处罚。一项遗漏是否重要、一项披露是否充分，取决于回复所遗漏的内容。探测这类监管义务意味着找到那些只需一次最小编辑就能使合规状态翻转的回复，而每次探测的代价是一次生成加两次裁定，因此监管红队测试的关键约束是样本效率，而非数量。我们提出Penumbra，一种对抗性搜索方法：它从一个经验证的锚点出发，在不断扩大的编辑预算下进行搜索，直到由两个评估器组成的委员会改变判定结果，并输出跨越这一变化的两个相邻回复。资源分配是自适应的，目标是对“失效面”的覆盖：即每个候选所解析出的不同（义务×失效模式）单元格。在相同预算下，自适应分配在59%的候选项上达到了均匀分配全额预算所实现的覆盖率……（原文摘要截断于此）

    arXiv:2610.04693v2 Announce Type: replace  Abstract: Agents are entering finance, healthcare and law, sectors where a violation leaves no lexical signature and carries real penalties. Whether an omission is material, or a disclosure sufficient, depends on what the response left out. Probing such an obligation means finding responses one minimal edit from flipping compliance, and every probe costs a generation and two adjudications, so the binding constraint on regulatory red-teaming is sample efficiency, not volume. We introduce Penumbra, an adversarial search that walks from a verified anchor under an expanding edit budget until a two-evaluator committee changes its verdict, and emits the two adjacent responses that straddle the change. Allocation is adaptive, and the objective is coverage of the defeat surface: distinct (obligation x defeat mode) cells resolved per candidate. At matched budget, adaptive allocation reaches uniform allocation's full-budget coverage on 59% of the candid
    
[^192]: S³：谱零空间交换使推理模型更高效

    $S^3$: Spectral Null-Space Swap Makes Reasoning Models Efficient

    [https://arxiv.org/abs/2609.37976](https://arxiv.org/abs/2609.37976)

    论文首次发现推理能力的关键在于思考模型权重中位于非思考模型主导子空间零空间内的分量，并提出无需训练的谱零空间交换方法S³，通过配对检查点组合在保持精度的同时大幅提升推理效率。

    

    经思维链训练的大语言模型在推理能力上表现出色，但往往伴随着过多的token开销。我们发现推理能力的核心在于思考模型的权重分量位于由对应非思考模型主导奇异方向所定义投影的零空间内，并且移除该子空间分量可以在不损害思考模式后训练所获得精度的前提下大幅提升推理效率。与现有大多在主导子空间内进行操作的方法不同，我们首次揭示了零空间的关键作用，并将其用于模型优化。受这一发现的启发，我们提出了谱零空间交换（S³），这是一种无需训练的、由配对的非思考与思考检查点组成的融合方法。我们的方法将非思考模型保留在其自身的主导子空间内，而将思考检查点置于该子空间之外，在提升推理效率的同时保持（摘要此处被截断）

    arXiv:2609.37976v1 Announce Type: cross  Abstract: LLMs trained with Chain-of-thought excel in reasoning capability, but often come with excessive token cost. We find that the core of reasoning capacity lies in the Thinking model's weight component within the null space of a projection defined by the corresponding Non-thinking model's dominant singular directions, and removing the subspace component can largely improve reasoning efficiency without hurting the accuracy gained during thinking-mode post-training. Unlike existing efforts that mostly operate within the dominant subspace, we are the first to unveil the critical role of the null space and harness it for model optimization. Motivated by this finding, we propose Spectral Null-Space Swap ($S^3$), a training-free composition of paired Non-thinking and Thinking checkpoints. Our method keeps the Non-thinking model inside its own dominant subspace and takes the Thinking checkpoint outside it, improving reasoning efficiency while mai
    
[^193]: 配对难度至关重要：重新思考成对式LLM裁判评估与一致性

    Pair Difficulty Matters: Rethinking Pairwise LLM-as-a-Judge Evaluation and Consistency

    [https://arxiv.org/abs/2609.37577](https://arxiv.org/abs/2609.37577)

    该论文指出，传统用于评估LLM裁判可靠性的三大代理指标（位置偏差、传递性、成对一致性）具有误导性——在Bradley-Terry几何下它们被排名差距接近的配对所主导，而此类配对出现不一致在信息论上是必然的，因此不应以这些指标否定能力出色的LLM评估器。

    

    大型语言模型（LLM）裁判被广泛用于通过成对比较来对文本及文本生成系统进行排序，其可靠性通常通过三个代理指标来评估：位置偏差、传递性和成对一致性（自评或人工标注）。由于这些代理指标驱动着裁判的选择与基准测试，大量报告裁判在这些指标上表现不佳的文献可能会使实践者放弃原本有能力的评估器。我们认为这种评估具有误导性。在成对聚合所依据的Bradley-Terry几何结构下，每个代理指标都由排名差距接近的配对所主导，而在这类配对中，不一致性在信息论上是可以预期的，且个别判断对总体排名的贡献微乎其微；相比之下，排名差距较大的配对虽然承载着排名信号，却几乎不会影响这些代理指标。我们对该论点进行了形式化，并在受控模拟以及两个人工评分语料库上进行了验证：这些代理指标与排名质量仅呈弱相关。

    arXiv:2609.37577v1 Announce Type: new  Abstract: Large Language Model judges are widely used to rank texts and text-generating systems through pairwise comparison, and their reliability is typically assessed via three proxies: position bias, transitivity, and pairwise agreement (self- or human-labeled). Because these proxies drive judge selection and benchmarking, a substantial literature reporting that judges perform poorly on them risks steering practitioners away from otherwise capable evaluators. We argue this assessment is misleading. Under the Bradley--Terry geometry underlying pairwise aggregation, each proxy is dominated by close-rank-gap pairs, where inconsistency is information-theoretically expected and individual verdicts contribute little to the aggregate ranking; far-gap pairs carry the ranking signal but barely move the proxies. We formalize this argument and validate it in a controlled simulation and on two human-rated corpora: the proxies correlate only weakly with ran
    
[^194]: 通过受控环境干预构建具有挑战性的浏览器使用任务

    Constructing Challenging Browser-Use Tasks by Controlled Environment Interventions

    [https://arxiv.org/abs/2609.35814](https://arxiv.org/abs/2609.35814)

    提出 BreakingWeb 基准，通过在保持用户指令与成功标准不变的前提下对环境不同技术栈层面进行确定性干预，将浏览器任务的难度转化为可编程的环境属性，从智能体已能解决的任务中系统性构建出 519 对更具挑战性的任务对。

    

    随着浏览器使用智能体的不断提升，基准测试也在通过收集新的任务、网站和应用程序来跟进，这通常使任务变得更长或更新颖。这使得任务难度的更新成本高昂且难以控制：当多个方面同时发生变化时，人们无法确定究竟是什么因素使任务变得具有挑战性。我们转而从智能体已经能够解决的任务出发来构建具有挑战性的实例，将难度转化为环境的一种可编程属性。BreakingWeb 将每个基础任务与一个干预条件配对，该条件在保持用户指令、潜在目标和后端成功标准不变的前提下，在不同层次的 Web 技术栈上对环境进行改变。每个干预都是确定性的、可检测的且可恢复的，并标注了其主要用于考验的认知基元。该基准测试包含 519 对干净/干预任务对，涵盖七个自托管网站和 29 个干预类别，全部基于结果进行评分。我们评估了……

    arXiv:2609.35814v1 Announce Type: cross  Abstract: As browser-use agents improve, benchmarks keep pace by collecting new tasks, websites, and applications, often making tasks longer or more novel. This makes difficulty expensive to refresh and difficult to control: when many aspects change at once, it is unclear what actually makes a task challenging. We instead construct challenging instances from tasks agents already solve, turning difficulty into a programmable property of the environment. BreakingWeb pairs every base task with an intervention condition that preserves the user instruction, latent target, and backend success criterion while changing the environment at different web stack layers. Each intervention is deterministic, detectable, and recoverable, and is annotated with the cognitive primitive it primarily loads. The benchmark contains 519 clean/intervention task pairs across seven self-hosted websites and 29 intervention families, all graded against outcomes. We evaluate 
    
[^195]: KuaFu：在十亿级规模上将长用户行为压缩为用户理解

    KuaFu: Compressing Long User Behavior into Understanding at Billion Scale

    [https://arxiv.org/abs/2609.31045](https://arxiv.org/abs/2609.31045)

    该论文提出KuaFu系统，在十亿用户规模上解决将长用户行为压缩为用户理解时序列过长与刷新吞吐量的双重工业瓶颈，并应对压缩过程可能引入的四类幻觉问题。

    

    对话式智能体、生成式推荐器和个性化广告都依赖一项核心能力：从原始行为中理解每一位用户。当前主流的工业实践是任务专用的：针对每个任务，从完整历史中提取相关子序列，并在其上训练专用模型。在生产环境中，这遇到了两个瓶颈。第一，即使经过过滤，单一任务的序列仍然极长：内容兴趣摘要需要读取每位用户的数百条内容，一旦序列化为提示文本就达数万个token。第二，用户画像需要例行刷新：每周覆盖十亿用户，总计约10万QPM，在固定的GPU预算下这设定了一个硬性的吞吐量下限。因此压缩是必不可少的，但截断或粗糙的压缩会悄然扭曲用户画像，引入四种幻觉类型（捏造、遗漏、日期错误归属、逻辑断裂），而由于无法评估压缩（原文在此处截断）……

    arXiv:2609.31045v1 Announce Type: cross  Abstract: Conversational agents, generative recommenders, and personalized advertising all rest on one capability: understanding each user from raw behavior. Prevailing industrial practice is task-specific: for each task, a relevant subsequence is extracted from the full history and a dedicated model trained on it. In production it hits two bottlenecks. First, even after filtering, a single-task sequence stays extremely long: content-interest summarization reads several hundred items per user, tens of thousands of tokens once serialized as prompt text. Second, profiles are refreshed routinely: a billion users weekly, roughly 100K QPM in aggregate, which under a fixed GPU budget sets a hard throughput floor. Compression is therefore mandatory, yet truncation or coarse compression can silently distort the profile, introducing four hallucination types (fabrication, omission, date misattribution, broken logic) that, with no way to evaluate the compr
    
[^196]: 哪些医学问题值得生成推理依据？面向鲁棒问答的扰动敏感选择方法

    Which Medical Questions Deserve Rationales? Perturbation-Sensitive Selection for Robust QA

    [https://arxiv.org/abs/2609.09684](https://arxiv.org/abs/2609.09684)

    该论文提出RMS-RSP方法，通过仅在推理依据token处扰动隐藏状态并测量答案与干扰项之间裕度的变化，在固定token预算下智能筛选哪些已标注医学问题最值得投入推理依据监督，从而提升医学问答的鲁棒性。

    

    医学问答数据集通常包含答案标签，而高质量的推理依据仍然稀缺、含有噪声或验证成本高昂。这改变了数据获取的核心问题：我们不再追问哪些问题应该被标注，而是探究在固定token预算下，哪些已标注的问题应该获得推理依据监督。我们研究了该问题的一个离线版本，其中候选推理依据对选择器可见，但除非被选中，否则不会用于下游训练。我们提出了均方根鲁棒性样本优先级方法（RMS-RSP），该方法仅在推理依据token处扰动隐藏状态，并测量由此导致的金标准答案与最佳干扰项之间裕度的变化。在五个医学问答数据集、MedGemma-4B-IT模型、三个训练种子、十个有预算的非RSP选择器以及一个无预算的全监督参考基线下，RMS-RSP给出了一个经过审慎限定的结果：其固定预算下的平均准确率为60.61%。

    arXiv:2609.09684v1 Announce Type: new  Abstract: Medical question-answering datasets often contain answer labels, whereas high-quality rationales remain scarce, noisy, or costly to validate. This changes the acquisition question: rather than asking which questions should be labeled, we ask which already-labeled questions should receive rationale supervision under a fixed token budget. We study an offline version of this problem in which candidate rationales are visible to the selector but withheld from downstream training unless selected. We propose root-mean-square Robustness-based Sample Prioritization (RMS-RSP), which perturbs hidden states only at rationale tokens and measures the resulting shift in the gold-versus-best-distractor margin. Across five medical QA datasets, MedGemma-4B-IT, three training seeds, ten budgeted non-RSP selectors, and an unbudgeted full-supervision reference, RMS-RSP provides a deliberately qualified result. Its locked-budget accuracy is 60.61% on average 
    
[^197]: 预训练期间的知识获取？大语言模型借助辅助视图学得更好

    Knowledge Acquisition During Pre-training? Large Language Models Learn Better With Auxiliary Views

    [https://arxiv.org/abs/2609.04180](https://arxiv.org/abs/2609.04180)

    研究发现，在预训练中将token预算从文档重复转移到辅助视图（知识的重新表述）能提升大语言模型的学习效果，即使对事实回忆也有效，且不依赖教师模型的强弱。

    

    我们对大语言模型（LLMs）在预训练期间如何获取知识的理解仍存在空白。我们提出假设：辅助视图（auxiliary views），即知识的重新表述，对学习具有因果性的帮助。我们设计了受控实验来隔离验证这一假设。首先，我们确认重复是知识获取的必要条件，并澄清释义改写仅在小批量大小（batch size）时才有所帮助。其次，在保持token预算固定的情况下，将token从文档重复中分配给辅助视图可以提升学习效果——反直觉的是，即使对于事实回忆也是如此。第三，辅助视图的有效性并不取决于生成它的教师模型的强弱。第四，我们识别出两类知识形式——上下文性知识和基础性知识——它们能够在存在先前知识空白的情况下帮助学习。最后，我们通过逐层偏置（layer-wise biases）和压缩（compression）机制，从机理层面考察了这些效应的表现。总之，我们的发现表明辅助表示……

    arXiv:2609.04180v1 Announce Type: cross  Abstract: Gaps remain in our understanding of how large language models (LLMs) acquire knowledge during pre-training. We posit that auxiliary views, reformulations of knowledge, are causally helpful for learning. We design controlled experiments to isolate this. First, we confirm that repetition is necessary for acquisition and clarify that paraphrasing helps only at smaller batch sizes. Second, holding the token budget fixed, allocating tokens from document repetition to auxiliary views improves learning, counterintuitively, even for factual recall. Third, the effectiveness of auxiliary views is not contingent on the strength of the teacher model that generates them. Fourth, we identify forms of knowledge, contextual and foundational, that aid learning in the presence of prior knowledge gaps. Finally, we examine how these effects manifest mechanistically via layer-wise biases and compression. Together, our findings suggest that auxiliary repres
    
[^198]: 当模型编辑过多：论最小代码编辑的保真度

    When Models Edit Too Much: On the Fidelity of Minimal Code Edits

    [https://arxiv.org/abs/2609.04061](https://arxiv.org/abs/2609.04061)

    该研究揭示了前沿大语言模型在修复代码时普遍存在“过度编辑”问题（即使如GPT-5.5这样的强模型也不例外），并提出通过一条简单的保留指令即可显著减少不必要的代码改动、降低认知复杂度，同时提升修复准确率。

    

    大语言模型越来越多地被用于编辑现有代码，但仅仅正确是不够的：有用的修复还应当是最小化的、可审查的，并且忠实于原始实现。我们研究了“过度编辑”现象，即模型重写代码的范围超出修复缺陷所需的趋势。我们基于400个BigCodeBench问题构建了一个评估框架，通过向参考解答中注入受控的AST级（抽象语法树级）破坏，为每个修复任务提供一个已知的最小补丁。研究发现，在各类前沿大语言模型中，过度编辑现象普遍存在，即使是在GPT-5.5这样的强大模型中也是如此：高Pass@1可能与不必要的巨大编辑和新增认知复杂度并存。一条保留指令能够显著减少这种行为，将平均超额Levenshtein距离从0.195降至0.131，减少26.6%的新增认知复杂度，并使Pass@1提高2.3个百分点。然而，这些收益并非简单地源于更大的推理（摘要在此处被截断）

    arXiv:2609.04061v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to edit existing code, but correctness alone is not enough: useful repairs should also be minimal, reviewable, and faithful to the original implementation. We study over-editing, the tendency of a model to rewrite code beyond what is required to fix a bug. We construct an evaluation framework from 400 BigCodeBench problems by injecting controlled AST-level corruptions into reference solutions, giving each repair task a known minimal patch. Across frontier LLMs, over-editing is widespread even among strong models like GPT-5.5: high Pass@1 can coexist with unnecessarily large edits and added cognitive complexity. A preservation instruction substantially reduces this behavior, lowering average excess Levenshtein distance from 0.195 to 0.131, reducing added cognitive complexity by 26.6%, and increasing Pass@1 by 2.3 points. However, these gains do not simply follow from a larger reasoning 
    
[^199]: SpanCalib-VLM：视觉-语言模型中经校准的幻觉片段检测

    SpanCalib-VLM: Calibrated Hallucination Span Detection in Vision-Language Models

    [https://arxiv.org/abs/2608.29974](https://arxiv.org/abs/2608.29974)

    提出 SpanCalib-VLM 混合双系统，将多模态序列标注器与微调生成式 VLM 相结合，并通过联合校准融合策略对候选幻觉片段进行重新评分，同时兼顾了准确的片段定位与良好校准的置信度。

    

    检测大型视觉-语言模型（LVLM）中的幻觉既需要准确的片段定位，也需要良好校准的置信度分数。微调后的生成式视觉-语言模型擅长识别幻觉文本片段，但存在过度自信和高推理延迟的问题；判别式序列标注器则提供确定性的速度和更优的校准性能，但片段召回率较为保守。我们提出了 SpanCalib-VLM，这是为 SHROOM-Visions 共享任务设计的一种混合双系统，它将一个多模态序列标注器（由 XLM-RoBERTa-Large 通过交叉注意力与 SigLIP 视觉编码器融合而成）与我们微调的生成式视觉-语言模型（Qwen3.5-4B-SHROOM-SFT）相结合。通过联合校准融合策略，来自生成式模型的候选片段会使用序列标注器输出的校准概率进行重新评分。在 SHROOM-Visions 英文评估集上，我们的集成系统达到了 0.41 的皮尔逊校准相关性以及……（原文在此处截断）

    arXiv:2608.29974v1 Announce Type: cross  Abstract: Detecting hallucinations in Large Vision-Language Models (LVLMs) requires both accurate span localization and well-calibrated confidence scores. Fine-tuned generative VLMs excel at identifying hallucinated text spans but suffer from overconfidence and high inference latency. Discriminative sequence taggers offer deterministic speed and superior calibration but exhibit conservative span recall. We present SpanCalib-VLM, a hybrid dual-system for the SHROOM-Visions Shared Task that combines a multimodal sequence tagger, consisting of XLM-RoBERTa-Large fused with a SigLIP vision encoder via cross-attention, with our fine-tuned generative VLM (Qwen3.5-4B-SHROOM-SFT). Through a Union-Calibrated Fusion strategy, candidate spans from the generative model are re-scored with calibrated probabilities from the sequence tagger. On the SHROOM-Visions English evaluation split, our ensemble achieves a Pearson calibration correlation of 0.41 and an ove
    
[^200]: 在截断评分量表上的双重差分法可能制造虚假效应：来自预注册大语言模型裁判审计的证据

    Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit

    [https://arxiv.org/abs/2608.27309](https://arxiv.org/abs/2608.27309)

    本文揭示双重差分法在截断评分量表上因截断不均会制造虚假交互效应，并通过预注册审计实证证明该偏差可导致无效结果。

    

    arXiv:2608.27309v1 公告类型：交叉 摘要：对大语言模型裁判的审计通过对比匹配条件来验证偏差，而最严谨的设计会进行两次差分：在两项候选回答之间进行项目内对比，再在操控的属性上进行二次差分，最终从有界评分量表上读取结果。我们证明，这一终点在该量表上无法被识别。双重差分的每一项都受到各自截断比例的影响，因此观察到的统计量混淆了差异偏好与差异衰减：当两项回答以不平等的方式截断时，一个共同作用于两者的严重性偏移会制造出交互效应，因为距界限的不等距离使它们恰好落在良好刺激所在的位置。我们在一个冻结教学裁判的预注册审计中展示了这一失败，该审计在首次990次调用之前就已密封。注册的主要终点，即陈述的学习者画像对裁判脚手架偏好的影响，为零：+0.085分（95% BCa置信区间）。

    arXiv:2608.27309v1 Announce Type: cross  Abstract: Audits of LLM judges certify a bias by contrasting matched conditions, and the strongest designs difference twice: a within-item contrast between two candidate responses, differenced again across a manipulated attribute, read off a bounded rating scale. We show that this endpoint is not identified on the scale that reports it. Each term of the double difference is censored by its own share, so the observed statistic confounds differential preference with differential attenuation: a severity shift common to both responses manufactures an interaction whenever the two censor it unequally, as unequal distances from the bounds make them, exactly where good stimuli place them. We exhibit the failure inside a pre-registered audit of a frozen pedagogy judge, sealed before the first of its 990 calls. The registered primary endpoint, the effect of a stated learner profile on the judge's scaffolding preference, is null: $+0.085$ points (95\% BCa 
    
[^201]: 条件总相关性与自适应并行采样的串行深度

    Conditional Total Correlation and the Serial Depth of Adaptive Parallel Sampling

    [https://arxiv.org/abs/2608.25505](https://arxiv.org/abs/2608.25505)

    本文证明了条件总相关性是自适应并行采样中并行性的精确信息成本，并为有限阶马尔可夫链提供了与阶数成正比、与长度成对数关系的零误差调度方案。

    

    受掩码扩散模型中并行解码的启发，我们研究了离散向量的自适应并行采样：在每一轮中，确定性策略根据目前已观测到的值选择未揭示的坐标，所选坐标则独立地从其精确的条件边际分布中采样。近似误差通过前向Kullback-Leibler散度衡量，而串行深度是在给定误差预算下，满足目标平均轮数的最小值。我们的核心结果是一个精确恒等式：每个策略的散度等于其揭示轮次中累积的期望条件总相关性，因此条件总相关性是轮内并行性的精确信息成本。该恒等式为有限阶马尔可夫链提供了零误差调度方案，其轮复杂度与马尔可夫阶数成正比，与序列长度成对数关系，并给出了匹配的对数特征。

    arXiv:2608.25505v1 Announce Type: cross  Abstract: Motivated by parallel decoding in masked diffusion models, we study adaptive parallel sampling of discrete vectors: in each round, a deterministic policy selects unrevealed coordinates on the basis of the values observed so far, and the selected coordinates are sampled independently from their exact conditional marginals. Approximation error is measured by forward Kullback-Leibler divergence, and serial depth is the minimum target-averaged number of rounds meeting a prescribed error budget. Our central result is an exact identity: the divergence of every policy equals the expected conditional total correlation accumulated over its reveal rounds, so conditional total correlation is the exact information cost of within-round parallelism. The identity yields zero-error schedules for finite-order Markov chains with round complexity proportional to the Markov order and logarithmic in sequence length, a matching logarithmic characterization 
    
[^202]: FRAGMENT：基于实体感知变换的因子化图表示用于文档生成与编辑

    FRAGMENT: Factorized Graph Representations for Document Generation and Editing via Entity-Aware Transformations

    [https://arxiv.org/abs/2608.18679](https://arxiv.org/abs/2608.18679)

    提出FRAGMENT框架，将结构化文档表示为类型化关系图，通过“架构师”（生成图拓扑与空间关系）和“构建器”（填充边界框、文本与视觉样式）两阶段因子化建模，实现了具有可处理似然的文档生成与编辑。

    

    诸如发票、表单、报告和科学文章等结构化文档，其意义源自空间布局、文本内容和逻辑结构之间的相互作用。在像素或词元级别运行的生成模型往往难以有效捕捉这些依赖关系。我们探索了FRAGMENT，这是一个生成式框架，它将文档表示为类型化关系图，并将其分布分解为 p(structure, content) = p(structure) × p(content | structure)。该框架由两个阶段组成。第一阶段是Architect（架构师），这是一个以文档类别为条件的因果掩码Transformer，自回归地生成图拓扑结构和类型化空间关系。第二阶段是Builder（构建器），这是一个基于GATv2的图注意力网络，用归一化的边界框、文本和视觉样式属性来丰富图。两个阶段均定义了显式的似然模型，从而产生可处理的文档级（似然）……

    arXiv:2608.18679v1 Announce Type: cross  Abstract: Structured documents such as invoices, forms, reports, and scientific articles derive meaning from the interplay between spatial layout, textual content, and logical structure. Generative models operating at the pixel or token level often struggle to capture these dependencies effectively. We explore FRAGMENT, a generative framework that represents a document as a typed relational graph and factorizes its distribution as p(structure, content) = p(structure) * p(content | structure). The framework consists of two stages. The first stage, the Architect, is a causally masked Transformer conditioned on document category that autoregressively generates the graph topology and typed spatial relations. The second stage, the Builder, is a GATv2-based graph attention network that enriches the graph with normalized bounding boxes, text, and visual style attributes. Both stages define explicit likelihood models, yielding a tractable document-level
    
[^203]: ReRef-3D：空间指代表达引导的三维场景重排基准

    ReRef-3D: A Benchmark for Spatial Referring Expression-Guided 3D Scene Rearrangement

    [https://arxiv.org/abs/2608.16011](https://arxiv.org/abs/2608.16011)

    本文提出了ReRef-3D基准，用于评估语言引导的三维场景重排，发现关系满足性优于物理有效性，且“最近”和“之间”等关系最具挑战性。

    

    arXiv:2608.16011v1 公告类型：新 摘要：我们引入了ReRef-3D，一个用于三维场景中语言引导放置的基准。它包含998个基于CLEVR衍生场景的33,826条指令，涵盖16种放置类别以及直接、一跳和二跳引用。每条指令必须被解析为一个有效的放置位置。鉴于指令定义的是可接受放置的区域而非单一坐标，我们的评估将预测插入场景中，重新计算关系，并测试关系满足性和物理有效性。每条指令还包括一个经过验证的自然化改写。在微调后，LLaVA-3D、3D-LLM和PlaceIt3D分别对68.3%、31.6%和22.4%的指令产生有效放置。跨模型来看，关系满足性优于物理有效性，像“最近”和“之间”这样的关系最难处理，而措辞对性能影响最小。

    arXiv:2608.16011v1 Announce Type: new  Abstract: We introduce ReRef-3D, a benchmark for language-guided placement in 3D scenes. It contains 33,826 instructions across 998 CLEVR-derived scenes, spanning 16 placement families and direct, one-hop, and two-hop references. Each instruction must be resolved into a valid new placement position. Given that an instruction defines a region of acceptable placements rather than one coordinate, our evaluation inserts a prediction into the scene, recomputes relations, and tests relation satisfaction and physical validity. Each instruction also includes a verified naturalized rewrite. After fine-tuning, LLaVA-3D, 3D-LLM, and PlaceIt3D produce valid placements for 68.3%, 31.6%, and 22.4% of instructions, respectively. Across models, relation satisfaction surpasses physical validity, relations such as nearest and between are the most difficult, and phrasing has minimal effect on performance.
    
[^204]: 机器的内在时钟：大语言模型是否共享人类的时间错觉？

    The Machine's Internal Clock: Do LLMs Share Human Temporal Illusions?

    [https://arxiv.org/abs/2608.15394](https://arxiv.org/abs/2608.15394)

    本研究通过新基准发现，大语言模型在时间错觉任务中比人类更倾向于选择文献预测场景，表明模型可能缺乏人类对时间的主观感知偏差。

    

    arXiv:2608.15394v1 公告类型：新 摘要：人类对时间的感知是主观的。已有充分记录的时间错觉表明，大脑依赖于上下文和关系线索来判断持续时间，而不是直接跟踪流逝的时间。先前的研究通过视觉和听觉刺激确立了这些效应。现有的大语言模型（LLM）时间感知评估侧重于估计事件持续时间或多步骤时间推理。在这项工作中，我们使用一个包含6,684个叙事对的新基准，涵盖五种时间错觉，调查仅凭书面叙事是否能引发人类的时间错觉。我们发现，人类读者（60名参与者）仅在五种错觉中的两种中偏好预期场景，即那些操作在文本中直接可见而非需要读者内部模拟持续时间的场景。我们在同一基准上评估了14个LLM。令人惊讶的是，我们发现模型在五种错觉中的四种中选择了文献预测的场景，与人类表现出现分歧。

    arXiv:2608.15394v1 Announce Type: new  Abstract: Human perception of time is subjective. Well-documented temporal illusions show that the brain relies on context and relational cues for judging duration instead of tracking elapsed time directly. Prior studies established these effects with visual and auditory stimuli. Existing LLM evaluations of temporal perception focus on estimating event durations or multi-step temporal reasoning. In this work, we investigate whether written narratives alone can evoke human temporal illusions, using a new benchmark of 6,684 narrative pairs spanning five illusions. We find that human readers (60 participants) prefer expected scenarios in only two of the five illusions, those where the manipulation is directly visible in text rather than requiring readers to internally simulate duration. We evaluate 14 LLMs on the same benchmark. Surprisingly, we find that models pick the literature-predicted scenario across four of the five illusions, diverging from 
    
[^205]: TokenSwap：多模态大语言模型中模态差距的基准测试与缩小

    TokenSwap: Benchmarking and Reducing the Modality Gap in Multimodal LLMs

    [https://arxiv.org/abs/2607.28640](https://arxiv.org/abs/2607.28640)

    本文提出TokenSwap方法，通过将文本基准（如MMLU）中的文本概念替换为语义对齐的图像，构建出图像交错的TokenSwap-Bench基准，首次系统揭示了42个多模态大语言模型中普遍存在的模态差距，即图像交错输入相比纯文本输入平均导致19.6%的性能下降。

    

    多模态大语言模型（MLLM）应当表现出跨模态一致性，即在语义等价但模态不同的输入下生成一致的响应。然而，我们观察到模型预测在这类跨模态变化下存在系统性差异。具体而言，我们将“模态差距”定义为模型在语义等价的文本输入与多模态输入下的性能差异。我们提出TokenSwap方法，通过用语义对齐的图像替换文本概念来构建此类输入，从而得到视觉token与文本token相互交错的序列。基于TokenSwap，我们将现有的基于文本的基准（如MMLU）转换为图像交错的对应版本，构建出TokenSwap-Bench。在对42个多模态大语言模型的评测中，我们观察到普遍存在的模态差距：当输入从纯文本转变为图像交错形式时，模型性能下降4.2%至47.4%，平均下降19.6%（±3.3%）。

    arXiv:2607.28640v2 Announce Type: replace  Abstract: Multimodal large language models (MLLMs) should exhibit cross-modal consistency, generating consistent responses given semantically equivalent inputs across modalities. However, we observe a systematic discrepancy in model predictions under such cross-modal variations. Specifically, we define the modality gap as the difference in model performance under semantically equivalent textual and multimodal inputs. We introduce TokenSwap, a method that constructs such inputs by replacing textual concepts with semantically aligned images, resulting in sequences where visual tokens are interleaved with text tokens. Based on TokenSwap, we transform existing text-based benchmarks (e.g., MMLU) into image-interleaved counterparts, resulting in TokenSwap-Bench. Across 42 MLLMs, we observe a pervasive modality gap, with performance decreasing by 4.2% to 47.4% when moving from text-only to image-interleaved inputs, averaging 19.6% +/- 3.3% across mod
    
[^206]: SINT-Flow：使用大语言模型工作流的模式集成

    SINT-Flow: Schema Integration using Large Language Model Workflows

    [https://arxiv.org/abs/2607.24492](https://arxiv.org/abs/2607.24492)

    本文提出由五个LLM算子组成的模式集成框架SINT-Flow，可实现全自动端到端模式集成，并首次支持将含多种实体类型的反规范化源表分解为实体特定的关系，同时发布了包含10个任务、93张关系表的模式集成基准SINT-Bench。

    

    模式集成的目标是，给定一组输入的模式或表，推导出一个全局的、统一的模式，能够以连贯的方式表示所有输入表的概念、属性和关系。本文提出了SINT-Flow，这是一个由五个基于大语言模型（LLM）的算子组成的模式集成框架，这些算子可以组合成工作流，以执行完全自动化的端到端模式集成。与现有方法相比，SINT-Flow能够处理包含描述多种实体类型属性的反规范化源表。在模式集成过程中，这些表会被分解为独立的、特定于实体的关系。为了评估SINT-Flow，我们引入了SINT-Bench，这是一个模式集成基准，包含10个模式集成任务，共由93个关系表组成，其中包括描述多种实体类型的表。我们使用GPT-5.2以及开源（摘要在此处截断）

    arXiv:2607.24492v2 Announce Type: replace  Abstract: The goal of schema integration is, given a set of input schemata or tables, to derive a global, unified schema that is able to represent the concepts, attributes, and relationships of all input tables in a coherent fashion. This paper presents SINT-Flow, a schema integration framework composed of five LLM-based operators that can be combined into workflows to perform fully automated, end-to-end schema integration. In contrast to existing approaches, SINT-Flow can process denormalized source tables that contain attributes describing multiple entity types. During the schema integration process, these tables are decomposed into separate entity-specific relations. To evaluate SINT-Flow, we introduce SINT-Bench, a schema integration benchmark comprising 10 schema integration tasks consisting of altogether 93 relational tables, including tables that describe multiple types of entities. We evaluate SINT-Flow using GPT-5.2 as well as the ope
    
[^207]: DECODEM：通过增强方法从公司组织文件中提取数据

    DECODEM: Data Extraction from Corporate Organizational Documents via Enhanced Methods

    [https://arxiv.org/abs/2607.15879](https://arxiv.org/abs/2607.15879)

    本文提出DECODEM基准测试，利用SEC EDGAR数据库中的上市公司章程及章程细则样本与人工标注标签，系统评估大型语言模型提取公司治理变量的能力，为耗时昂贵的人工编码和覆盖有限的商业数据集提供了新的替代方案。

    

    许多实证法学研究依赖于将非结构化文本转换为结构化变量。在公司治理研究及其他领域，这种转换传统上由人工完成，而人工编码既耗时又昂贵。在公司治理研究中，标准化商业数据集提供了一种替代方案，但其覆盖的公司范围和治理变量种类有限。本文介绍了DECODEM基准测试，并利用这些基准评估大型语言模型能否从公司章程和章程细则中提取公司治理变量。该基准测试将由SEC的EDGAR数据库获取的美国上市公司章程和章程细则样本，与手工编码的标签相结合，这些标签用于标注文档中是否包含特定的治理安排。论文利用这些基准测试评估了一系列提取方法和模型，这些方法在多个维度上各不相同。

    arXiv:2607.15879v3 Announce Type: replace-cross  Abstract: Much empirical legal research depends on translating unstructured text into structured variables. In corporate governance research as elsewhere, this translation has traditionally been performed by humans. Manual coding is time-consuming and costly. In corporate governance research, standardized commercial datasets offer an alternative, but they cover a limited set of firms and governance variables.   This paper introduces the DECODEM Benchmarks and uses them to evaluate whether large language models can extract corporate governance variables from charters and bylaws. The benchmarks combine a sample of corporate charters and bylaws of publicly traded U.S. corporations obtained from the SEC's EDGAR database with hand-coded labels that capture whether a document includes a specific governance arrangement. The paper uses these benchmarks to evaluate a range of extraction approaches and models. The approaches vary along several dim
    
[^208]: 当合成语音是你仅有的一切：快呼叫GRPO

    When Synthetic Speech Is All You Have: Better Call GRPO

    [https://arxiv.org/abs/2607.08409](https://arxiv.org/abs/2607.08409)

    本文证明在隐私受限的银行等领域，用强化学习GRPO（尤其SFT+GRPO组合）替代监督微调来适配基于LLM的ASR，仅凭合成语音即可将词错误率相对降低40%–45%，其增益主要来自停止校准和语音-文本对齐等行为层面的改善。

    

    基于LLM的自动语音识别（ASR）在适配银行等受监管领域时受到隐私瓶颈的制约：真实语音采集成本高昂且受法律限制，这使得合成文本转语音（TTS）成为一种有吸引力的替代方案。然而，合成语音在声学上与真实录音存在不匹配，而针对这一差距的研究一直局限于监督微调（SFT）。我们转而采用强化学习，并证明组相对策略优化（GRPO）能从相同的合成语音中挖掘出远超SFT的性能。仅使用合成语音对模型进行GRPO适配——这是一种无评论家的方法，通过奖励低词错误率（WER）的假设进行优化——使WER相对SFT降低了40%（36.71%→22.09%），而SFT后再接GRPO的组合将这一提升进一步推至45%。我们将该增益归因于行为层面而非表征层面：GRPO通过改善停止时机的校准减少了插入错误，并通过更好地将注意力锚定于音频改善了语音到文本的对齐，同时早期层表征（原文在此处截断）

    arXiv:2607.08409v2 Announce Type: replace-cross  Abstract: LLM-based ASR adapted to regulated domains such as banking is bottlenecked by privacy: real speech is costly and legally constrained to collect, making synthetic text-to-speech (TTS) an attractive substitute. Yet synthetic speech stays acoustically mismatched with real recordings, and work on this gap has stayed within supervised fine-tuning (SFT). We instead turn to reinforcement learning, and show that Group Relative Policy Optimization (GRPO) extracts far more from the same synthetic speech than SFT. Synthetic-only adaptation of the model with GRPO, a critic-free method rewarding low-WER hypotheses, reduces WER by 40% relative to SFT (36.71%$\to$22.09%), and an SFT-then-GRPO combination pushes this further to 45%. We trace the gain to behavior rather than representation: GRPO reduces insertion errors by improving stopping calibration and speech-to-text alignment by better anchoring attention to audio, leaving early-layer rep
    
[^209]: 探测而非提示：用于Multi-Meta-RAG元数据过滤的隐藏状态探针

    Probe, Don't Prompt: A Hidden-State Probe for Metadata Filtering in Multi-Meta-RAG

    [https://arxiv.org/abs/2607.03929](https://arxiv.org/abs/2607.03929)

    该论文提出用小型开源语言模型隐藏状态上训练的本地确定性多标签探针，替代Multi-Meta-RAG中基于GPT-3.5的新闻来源元数据提取器，从而免去专有API调用并将输出严格限制在允许的来源列表内。

    

    Multi-Meta-RAG通过在元数据（具体为新闻来源和发布日期）上过滤向量存储来改进多跳问答的检索效果，这些元数据由GPT-3.5从每个查询中提取。该提取器需要对每个查询进行专有API调用，且其输出并不受允许来源列表的约束：尽管提示中指定了49个来源，GPT-3.5却生成了128个不同的来源名称，其中79个不在列表之内。我们证明，该提取器的来源部分可以由一个本地、确定性的探针替代，该探针在小型开源语言模型的隐藏状态上训练得到。该探针是作用于某一层均值池化隐藏状态的多标签逻辑回归分类器，使用类别平衡损失在黄金证据来源上训练。通过5折交叉验证来选择层，我们在全部2556个MultiHop-RAG查询上对参数量从135M到1.5B的四个模型进行了折外评估。

    arXiv:2607.03929v2 Announce Type: replace-cross  Abstract: Multi-Meta-RAG improves retrieval for multi-hop question answering by filtering a vector store on metadata, specifically the news source and the publication date, which it extracts from each query using GPT-3.5. This extractor requires a proprietary API call for every query, and its output is not restricted to the allowed sources: even though the prompt specifies 49 sources, GPT-3.5 generates 128 distinct source names, 79 of which do not appear on the list. We show that the source part of this extractor can be replaced by a local, deterministic probe trained on the hidden states of a small open-source language model. The probe is a multi-label logistic classifier on the mean-pooled hidden states of one layer, trained with a class-balanced loss on the gold evidence sources. Using 5-fold cross-validation to select the layer, we evaluate four models ranging from 135M to 1.5B parameters out-of-fold across all 2556 MultiHop-RAG quer
    
[^210]: 随机化YaRN改进长上下文推理的长度泛化能力

    Randomized YaRN Improves Length Generalization for Long-Context Reasoning

    [https://arxiv.org/abs/2606.23687](https://arxiv.org/abs/2606.23687)

    随机化YaRN通过将YaRN位置外推与随机化位置编码和长度课程学习相结合，使模型在仅使用短上下文数据训练的情况下，也能在16K到128K的长上下文推理任务上实现一致的泛化性能提升。

    

    大语言模型（LLM）通常在短序列上进行预训练，然后通过额外的训练扩展以处理更长的序列。然而，这类模型仍然难以进一步泛化到非常长的序列。我们提出了随机化YaRN（Randomized YaRN），这是一种通过将基于YaRN的位置外推与随机化位置编码和长度课程学习相结合来改进长度泛化能力的训练方法。在短上下文数据上训练期间，token被赋予从更大位置范围采样得到的YaRN位置编码，使模型即使在短上下文输入上也能接触到分布外的位置表示。我们在三个具有挑战性的长上下文推理基准上评估了随机化YaRN：BABILong、多轮共指消解（MRCR）和LongBench v2。当在短上下文数据上训练时，随机化YaRN在16K到128K的上下文长度上一致地提升了推理性能，并优于

    arXiv:2606.23687v2 Announce Type: replace  Abstract: Large language models (LLMs) are typically pretrained on short sequences and then extended to work on longer sequences with additional training. However, such LLMs still struggle to further generalize to very long sequences. We propose Randomized YaRN, a training method that improves length generalization by combining YaRN-based positional extrapolation with randomized positional encoding and a length curriculum. During training on short context data, tokens are assigned YaRN positional encodings sampled from a larger position range, exposing the model to out-of-distribution positional representations even on short-context inputs. We evaluate Randomized YaRN on three challenging long-context reasoning benchmarks, BABILong, Multi-Round Coreference Resolution (MRCR), and LongBench v2. When training on data with short context, Randomized YaRN consistently improves reasoning performance on context lengths from 16K to 128K and outperforms
    
[^211]: 分解混合专家模型中的拒绝引导机制

    Decomposing Refusal Steering in Mixture-of-Experts Models

    [https://arxiv.org/abs/2606.04160](https://arxiv.org/abs/2606.04160)

    该研究将拒绝引导方法扩展到混合专家模型，发现单个专家在自由选择位置时平均可恢复78%的完整引导效果，从而揭示了MoE模型中拒绝机制在各组件间的运作方式。

    

    arXiv:2606.04160v2 公告类型：替换 摘要：指令微调大语言模型（LLM）的安全对齐依赖于模型可靠地拒绝有害或违规请求的能力。近期研究表明，可以在推理阶段向稠密LLM施加引导向量以抑制拒绝行为，从而诱导模型对有害请求作出响应。我们将这种拒绝引导方法扩展到三个开源混合专家模型上，以分解并更好地理解拒绝机制如何在MoE的各个组件中运作。我们发现，引导性能不受MoE架构固有的复杂路由模式的影响；当允许专家自由选择自身位置时，单个专家平均可恢复78%的完整引导效果。然而，当专家级引导被限制在全层引导所使用的位置时，平均仅能恢复54%的效果，且在安全相关系统提示下这一差距进一步扩大。我们的结果还表明，拒绝信号...

    arXiv:2606.04160v2 Announce Type: replace  Abstract: Safety alignment in instruction-tuned large language models (LLMs) depends on a model's ability to reliably refuse harmful or disallowed requests. Recent work has shown that a steering vector can be applied to a dense LLM during inference to suppress refusal behavior and induce responses to harmful requests. We extend this refusal steering method to three open-source Mixture-of-Experts (MoE) LLMs to decompose and better understand how refusal mechanisms operate across MoE components. We find that steering performance is uninhibited by the complex routing patterns inherent to the MoE architecture, and that a single expert recovers 78% of the full steering effect on average when free to select its own location. However, expert-level steering only recovers 54% on average when constrained to the location used by full-layer steering, a gap that widens further under safety-related system prompts. Our results also show that refusal signals 
    
[^212]: 智慧的形状：语言模型如何收敛于一个答案

    The Shape of Wisdom: How Language Models Converge on an Answer

    [https://arxiv.org/abs/2606.01202](https://arxiv.org/abs/2606.01202)

    该研究发现，语言模型在回答事实性问题时，得到相同答案的问题会在内部状态空间中收敛到共享的“答案质心”，该质心在网络约47%-71%深度处突然变得可读取，且只取决于答案内容本身而非问题的表述方式。

    

    当语言模型回答一个事实性问题时，它的内部状态会走向何方？我们向四个开源模型（Llama-3.1-8B、Qwen2.5-7B、Gemma-2-9B 和 Llama-3.2-1B）提出了 2,256 个答案会重复出现的事实性问题，例如某人所拥有的国籍是哪个国家。在每一层，我们都记录了产生答案时所处的状态。得到相同答案的问题会收敛到状态空间中的一个共享区域，我们用该区域的中心——“答案质心”——来概括它。答案在单一层内即可被读取。在每个模型中，其状态最接近自身答案质心的问题比例会在某一层发生跳跃，该层位于网络进程的 47% 到 71% 之间。质心追随的是模型给出的答案而非问题本身：从关于人物国籍的问题中学到的某个国家的质心，也能在关于城镇和电影的问题中识别出该国家。模型本身也在使用这些质心。在跳跃之后，……（摘要原文在此处截断）

    arXiv:2606.01202v2 Announce Type: replace  Abstract: When a language model answers a factual question, where does its internal state go? We asked four open models (Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B and Llama-3.2-1B) 2,256 factual questions whose answers recur, such as the country a person is a citizen of. At every layer, we recorded the state from which the answer is produced. Questions that receive the same answer converge on a shared region of state space, which we summarise by its centre, the answer centroid. The answer becomes readable within a single layer. In every model, the share of questions whose state lies nearest its own answer's centroid jumps at one layer, between 47% and 71% of the way through the network. The centroids follow the answer the model gives rather than the question: the centroid of a country learned from questions about people's citizenship also identifies that country in questions about towns and films. The model also uses the centroids. After the jump, 
    
[^213]: 0.1%何时足够？分析降维与量化对文本嵌入压缩的联合效应

    When Is 0.1% Enough? Analyzing the Combined Effects of Dimensionality Reduction and Quantization on Text Embedding Compression

    [https://arxiv.org/abs/2606.01074](https://arxiv.org/abs/2606.01074)

    本文系统研究了降维与量化相结合对文本嵌入压缩的联合效应，发现在分类任务中仅需不到0.1%的存储空间即可保留至少99%的原始性能。

    

    近期高性能的文本嵌入模型通常会产生高维实值向量，导致巨大的存储和计算成本。为解决这一问题，研究者提出了基于降维或量化的嵌入压缩方法；然而，两者结合使用的效果尚未得到充分研究。本文使用五个预训练嵌入模型，系统研究了降维和量化如何共同影响嵌入压缩在海量文本嵌入基准（MTEB）四个任务族上的表现。我们的结果表明，将降维与量化相结合，能够以远小于单独使用任一方法的存储占用来保持接近原始的性能。特别是对于分类任务，多个模型仅使用不到原始存储需求0.1%的空间，就能保留至少99%的原始分类分数。

    arXiv:2606.01074v2 Announce Type: replace  Abstract: Recent high-performing text embedding models often produce high-dimensional real-valued vectors, resulting in substantial storage and computational costs. To address this issue, embedding compression methods based on dimensionality reduction or quantization have been proposed; however, the effects of their combination have not been sufficiently investigated. In this paper, we systematically investigate how dimensionality reduction and quantization jointly affect embedding compression across four task families in the Massive Text Embedding Benchmark (MTEB), using five pretrained embedding models. Our results show that combining dimensionality reduction and quantization can preserve near-original performance with substantially smaller storage footprints than either method alone. In particular, for classification, several models retain at least 99% of their original classification score using less than 0.1% of the storage required by th
    
[^214]: 《测量、定位与消融大语言模型中的对齐签名》

    Measuring, Localizing, and Ablating Alignment Signatures in LLMs

    [https://arxiv.org/abs/2605.30526](https://arxiv.org/abs/2605.30526)

    该论文发现后训练会使大语言模型的生成文本偏离人类语料风格并呈现出可被AI检测器识别的类AI特征，并提出了无需训练的PASTA方法，通过从残差对比中估计对齐签名方向并在解码时消融该方向，从而定位并去除模型内部的这一AI风格签名。

    

    经过对齐的大语言模型通常表现出一种可被识别的类AI风格，但这种风格与后训练及模型内部表征之间的联系仍知之甚少。在本工作中，我们研究了后训练是否引入或放大了类AI的风格规律性，以及这些规律性是否对应于一个可局部化的内部签名。为此，我们在匹配的人类来源前缀条件下，比较了人类文本、基座模型生成文本与对齐模型生成文本。结果表明，与基座模型的生成文本相比，对齐模型的生成文本具有更低的人类语料亲和度和更高的AI检测率，这说明后训练使生成文本偏离了人类语料风格，转向了检测器易于识别的类AI文本。随后，我们提出了PASTA（Post-training Alignment Signature Targeted Ablation，后训练对齐签名定向消融），这是一种无需训练的方法：它通过对比对齐模型与基座模型的残差来估计后训练对齐签名，并在解码过程中消融相应的方向。在11个……（摘要被截断）

    arXiv:2605.30526v2 Announce Type: replace-cross  Abstract: Aligned language models often exhibit a recognizable AI-like style, yet its connection to post-training and internal representations remains poorly understood. In this work, we study whether post-training introduces or amplifies AI-like stylistic regularities and whether these regularities have a localized internal signature. To this end, we compare human text, base-model generations, and aligned-model generations under matched human-source prefixes. Aligned generations show lower human-corpus affinity and higher AI-detection rates than base generations, suggesting that post-training shifts generated text away from human-corpus style and toward detector-visible AI-like text. We then introduce PASTA (Post-training Alignment Signature Targeted Ablation), a training-free method that estimates a post-training alignment signature from aligned-base residual contrasts and ablates the corresponding direction during decoding. Across 11 
    
[^215]: 相同证据，不同答案：面向多轮语言模型的规范上下文同策略蒸馏

    Same Evidence, Different Answers: Canonical-Context On-Policy Distillation for Multi-Turn Language Models

    [https://arxiv.org/abs/2605.30251](https://arxiv.org/abs/2605.30251)

    提出规范上下文同策略蒸馏（CCOPD），让同一基础模型分别担任基于完整提示的冻结教师与逐步接收分片证据的多轮学生，并将学生在自身轨迹上的行为与教师的规范全上下文行为对齐，从而缓解多轮对话中的自我锚定漂移，使模型在证据相同时无论以完整提示还是分片对话呈现都能给出一致的答案。

    

    大型语言模型（LLM）在所有指令一次性给出时通常能够完成任务，但当相同的信息在多轮对话中逐步揭示时却常常失败。当一个干净的完整（FULL）提示和一个分片的多轮（RAW-SHARDED）对话包含相同的完整用户证据时，模型理应得出相同的答案。我们认为造成这一差距的一个关键原因是自我锚定漂移：在部分信息下生成的回复会引入缺乏依据的假设，而这些假设随后会扭曲最终答案。为了减少这种影响，我们提出了规范上下文同策略蒸馏（CCOPD）。在训练过程中，同一个基础模型扮演两个角色：一个以干净的完整提示为条件的冻结教师模型，以及一个通过多轮对话逐步接收相同证据的可训练学生模型；CCOPD 将学生模型在自身轨迹上的行为与教师模型在规范完整上下文下的行为对齐。仅训练……（原文摘要到此被截断）

    arXiv:2605.30251v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) often solve a task when all instructions are given in a single prompt, but fail when the same information is revealed gradually across turns. When a clean FULL prompt and a RAW-SHARDED conversation contain the same complete user evidence, the model should still arrive at the same answer. We argue that a key reason for this gap is self-anchored drift: responses produced under partial information introduce unsupported assumptions, and those assumptions later distort the final answer. To reduce this effect, we propose Canonical-Context On-Policy Distillation (CCOPD). During training, the same base model is used in two roles: a frozen teacher conditioned on the clean FULL prompt and a trainable student that receives the same evidence incrementally through a multi-turn conversation; CCOPD aligns the student's behavior on its own trajectories with the teacher's canonical full-context behavior. Trained onl
    
[^216]: 一小时访谈无法告诉大语言模型关于你的什么：访谈捕捉的是人格特质，而非决策策略

    What a One-Hour Interview Does Not Tell an LLM About You: Interviews Elicit Traits, Not Decision Policies

    [https://arxiv.org/abs/2605.29458](https://arxiv.org/abs/2605.29458)

    一小时LLM访谈无法可靠预测个人在价值两难问题上的真实决策——访谈捕捉到的是人格特质而非具体的决策策略，且大多数预测错误源于转录文本中根本缺乏区分个体选择的关键证据。

    

    大型语言模型越来越多地被要求代表特定的人，而访谈这个人本身是让模型了解其身份的自然方式。我们测试了由LLM主导的一小时自适应访谈能让模型了解到个人在价值相关决策上的多少信息。46名成年人完成了这样的访谈，随后回答了25个两难问题，这些答案作为真实基准。我们从转录文本的不同部分预测他们的答案，在同一批参与者上比较自适应追问与固定问题，并将每种变体与无任何个人信息的模型以及多数答案基线进行比较。结果显示，访谈的任何部分都没有可靠地优于任何基线，自适应追问也不比固定问题更好。对转录文本的两项审计揭示了原因：在85%的预测错误中，审计员找不到任何能将该个体的选择与其他选项区分开来的证据，而在转录文本确实包含相关信息时……（原文摘要在此处截断）

    arXiv:2605.29458v2 Announce Type: replace-cross  Abstract: Large language models are increasingly asked to stand in for particular people, and interviewing the person is the natural way to tell a model who they are. We test what a one-hour LLM-led adaptive interview tells a model about an individual's value-laden decisions. 46 adults completed such an interview and then answered 25 dilemma questions, which are the ground truth. We predict their answers from different parts of the transcript, compare adaptive follow-up questions with fixed ones on the same participants, and benchmark every variant against a model given no information about the person and against the majority answer. No part of the interview is reliably better than either baseline, and adaptive follow-ups are no better than fixed ones. Two audits of the transcripts show why. For 85% of the errors an auditor finds no evidence that distinguishes the person's choice from the alternatives, and where a transcript does contain
    
[^217]: 面向知识产权保护的具有最小语义失真的鲁棒大语言模型水印

    Robust LLM Watermarking with Minimal Semantic Distortion for IP Protection

    [https://arxiv.org/abs/2605.23175](https://arxiv.org/abs/2605.23175)

    提出了一种密钥条件水印框架SAFESEAL，通过保留命名实体并借助锦标赛采样机制用上下文感知的同义词替换语言术语，在几乎不损害模型效用的情况下实现了强可检测性和鲁棒性，用于大语言模型的知识产权保护。

    

    专有大型语言模型（LLM）面临知识产权（IP）侵犯的风险，因为攻击者可以通过收集输入-输出对来训练替代模型，从而复制大语言模型，造成经济损失。水印为验证所有权提供了一种有前景的防御手段，但现有方法往往存在语义失真、事实不一致以及易受对抗攻击等问题。此外，用于提供者特定检测的密钥条件水印——尤其是在跨提供者和多用户场景下——在很大程度上仍未被充分探索。为应对这些挑战，我们提出了SAFESEAL，一种新颖的密钥条件水印框架，它在对模型效用影响极小的情况下实现了强大的可检测性，有效平衡了可检测性、效用与鲁棒性。SAFESEAL在保留命名实体的同时，通过密钥条件的锦标赛采样机制，用上下文感知的同义词替换语言术语，从而保持……

    arXiv:2605.23175v2 Announce Type: replace-cross  Abstract: Proprietary large language models (LLMs) face risks of intellectual property (IP) violation, as adversaries can replicate an LLM by collecting input-output pairs to train a surrogate model, causing financial setbacks. Watermarks offer a promising defense to verify ownership, but existing methods often struggle with semantic distortion, factual inconsistency, and adversarial attacks. In addition, key-conditioned watermarks for provider-specific detection, especially in cross-provider and multi-user scenarios, remain largely underexplored. To address these challenges, we propose SAFESEAL, a novel key-conditioned watermarking framework that achieves strong detectability with minimal impact on model utility, effectively balancing detectability, utility, and robustness. SAFESEAL preserves named entities while substituting linguistic terms with context-aware synonyms through a key-conditioned Tournament sampling mechanism, maintainin
    
[^218]: Psy-Chronicle：基于学生学期时间线的校园心理咨询对话合成

    Psy-Chronicle: Synthesizing Campus Counseling Grounded in a Student's Semester Timeline

    [https://arxiv.org/abs/2605.22140](https://arxiv.org/abs/2605.22140)

    Psy-Chronicle 提出一个事件驱动的多轮心理咨询对话合成框架，以模拟学生的学期时间线和压力事件图为根基生成跨会话携带记忆的对话，并据此构建了包含 100 名学生、1,248 次会谈的开放中文校园心理咨询数据集 CPCD。

    

    心理咨询通常跨越多次会谈展开，来访者在每次会谈中所谈论的内容，取决于自上次会谈以来其生活中发生的事情。现有的合成心理咨询语料库大多为单次会谈，而多会话语料库则沿着既定的治疗方案推进，并未对来访者两次会谈之间的生活进行建模。我们提出了 Psy-Chronicle，这是一个事件驱动的框架，将多轮心理咨询对话的合成建立在学生的学期时间线之上：每个模拟学生都拥有一个个人档案和一个覆盖整个学期的时间性压力事件图，心理咨询会谈沿着这一时间线生成，并在各次会谈之间保留记忆。将该框架应用于校园心理咨询场景，我们构建了 CPCD——一个开放的中文数据集，包含 100 名学生和 1,248 次跨越整个学期的会谈。对话严格遵循各自的时间线：事件之间的依赖关系能够跨越会谈延续，且对先前会谈的回指很少引入时间线轨迹中不存在的内容。

    arXiv:2605.22140v3 Announce Type: replace  Abstract: Psychological counseling unfolds over many sessions, and what a client brings to each session depends on what has happened in their life since the last one. Existing synthetic counseling corpora are mostly single-session, and multi-session ones advance along a therapy plan without modeling the client's life between sessions. We present Psy-Chronicle, an event-driven framework that grounds multi-session counseling synthesis in a student's semester timeline: each simulated student receives a profile and a semester-long temporal stress event graph, and counseling sessions are generated along this timeline with memory carried across them. Applied to campus counseling, it yields CPCD, an open Chinese dataset of 100 students and 1{,}248 sessions spanning a full semester. The dialogues follow their timelines: dependencies between events carry across sessions, and references to earlier sessions rarely introduce content absent from the trajec
    
[^219]: 重新思考如何记忆：超越原子化事实的终身LLM智能体记忆

    Rethinking How to Remember: Beyond Atomic Facts in Lifelong LLM Agent Memory

    [https://arxiv.org/abs/2605.19952](https://arxiv.org/abs/2605.19952)

    TriMem通过同时维护原始对话片段、原子化事实和合成档案三种共存的信息表示粒度，克服了现有基于原子事实的记忆方法丢失细粒度细节、无法深度推理且提取粒度不一致的局限，提升了终身LLM智能体记忆系统的存储保真度、检索效率和推理深度。

    

    为了实现可靠的长期交互，LLM智能体需要一个能够忠实存储、高效检索并对积累的对话历史进行深度推理的记忆系统。大多数现有方法采用基于提取事实的范式：通过手工设计的静态提示将原始对话压缩为原子化事实，再将这些事实存储、匹配并注入下游推理。然而，这种以事实为中心的设计不可避免地会丢弃原始对话中的细粒度细节，且无法对分散的孤立事实进行深度推理。此外，静态提示无法在不同对话风格之间保持一致的提取粒度。为了解决这些局限性，我们提出了TriMem，它维护三种共存的表示粒度，包括以源标识符锚定的原始对话片段以保证存储保真度、用于高效记忆检索的提取原子化事实，以及对信息进行聚合的合成档案。

    arXiv:2605.19952v2 Announce Type: replace  Abstract: To enable reliable long-term interaction, LLM agents require a memory system that can faithfully store, efficiently retrieve, and deeply reason over accumulated dialogue history. Most existing methods adopt an extracted fact based paradigm: handcrafted static prompts compress raw dialogues into atomic facts, which are then stored, matched, and injected into downstream reasoning. Nevertheless, such fact-centric designs inevitably discard fine-grained details in original dialogues and fail to support deep reasoning over scattered isolated facts. Moreover, static prompts cannot maintain consistent extraction granularity across diverse dialogue styles. To address these limitations, we propose TriMem, which maintains three coexisting representation granularities, including raw dialogue segments anchored by source identifiers for storage fidelity, extracted atomic facts for efficient memory retrieval, synthesized profiles that aggregate di
    
[^220]: 动态模型合并的瘦身之道

    Dynamic Model Merging Made Slim

    [https://arxiv.org/abs/2605.18904](https://arxiv.org/abs/2605.18904)

    DiDi-Merging通过可微秩选择在共享参数与专家参数之间联合分配低秩容量，并结合无数据的参数空间精炼，在仅1.24倍骨干网络存储的预算下实现了领先的动态模型合并性能。

    

    模型合并能够在无需联合训练或访问任务数据的情况下，将独立微调的检查点整合在一起。动态方法保留了任务特定的容量，但必须在共享参数与专家参数之间分配有限的存储预算。我们提出DiDi-Merging，该方法通过可微秩选择为这些组件联合分配低秩容量，随后进行无数据的参数空间精炼。在八任务ViT-B/32基准测试中，小型配置以单个骨干网络1.24倍的存储量（含该骨干网络）达到89.3%的平均准确率（98.7%的归一化保留率）；中型配置以1.4倍存储量达到89.8%。移除精炼步骤会使中型配置下降4.3个百分点，而静态合并加均匀恢复的对照方法比小型配置低4.7-8.9个百分点。额外的评估还涵盖了在各自报告预算下的语言和多模态模型。（原摘要在此处截断）

    arXiv:2605.18904v2 Announce Type: replace-cross  Abstract: Model merging consolidates independently fine-tuned checkpoints without joint training or access to task data. Dynamic methods retain task-specific capacity, but must divide a limited storage budget between shared and expert parameters. We propose DiDi-Merging, which jointly allocates low-rank capacity to these components through differentiable rank selection, followed by data-free parameter-space refinement. On the eight-task ViT-B/32 benchmark, the small configuration reaches 89.3% average accuracy (98.7% normalized retention) at 1.24x the storage of one backbone, including that backbone; the medium configuration reaches 89.8% at 1.4x. Removing refinement reduces the medium configuration by 4.3 percentage points, while static-merge plus uniform-recovery controls are 4.7-8.9 points below the small configuration. Additional evaluations cover language and multimodal models at their separately reported budgets. The main results a
    
[^221]: 面向推测解码的性能驱动策略优化与自适应窗口机制

    Performance-Driven Policy Optimization for Speculative Decoding with Adaptive Windowing

    [https://arxiv.org/abs/2605.14978](https://arxiv.org/abs/2605.14978)

    提出PPOW强化学习框架，将推测解码中草稿模型的优化从token级模仿提升到窗口级优化，通过成本感知加速奖励、分布接近度奖励和自适应散度感知窗口机制，有效解决难起草位置导致的推测效率瓶颈，加速LLM推理。

    

    推测解码通过让轻量级草稿模型提出候选token的推测窗口，再由更大的目标模型进行并行验证，从而加速大语言模型（LLM）的推理。在实践中，推测效率往往受限于难以起草的位置——在这些位置上，早期的不匹配会截断已被接受的前缀，并使推测窗口的其余部分失效。尽管推测效用本质上是窗口级别且对前缀敏感的，但大多数基于学习的草稿模型仍然使用token级别的监督目标进行优化。我们提出了PPOW（性能驱动的策略优化与自适应窗口机制），这是一个强化学习框架，将草稿模型的优化从token级别的模仿转变为窗口级别的优化。PPOW结合了成本感知加速奖励、基于分布的接近度奖励以及自适应散度感知窗口机制，优先处理具有高置信度加权草稿的信息丰富窗口。

    arXiv:2605.14978v3 Announce Type: replace  Abstract: Speculative decoding accelerates LLM inference by having a lightweight draft model propose speculative windows of candidate tokens for parallel verification by a larger target model. In practice, speculative efficiency is often bottlenecked by hard-to-draft positions, where an early mismatch truncates the accepted prefix and invalidates the rest of the speculative window. Most learning-based drafters are still optimized with token-level supervised objectives, even though speculative utility is inherently window-level and prefix-sensitive. We propose PPOW (Performance-Driven Policy Optimization with Adaptive Windowing), a reinforcement learning framework that shifts drafter optimization from token-level imitation to window-level optimization. PPOW combines a Cost-Aware Speedup Reward, a Distribution-Based Proximity Reward, and Adaptive Divergence-Aware Windowing, which prioritizes informative windows with high confidence-weighted draf
    
[^222]: 使用嵌入模型改进概率性种族预测

    Using Embedding Models to Improve Probabilistic Race Prediction

    [https://arxiv.org/abs/2604.22555](https://arxiv.org/abs/2604.22555)

    提出eBISG方法，利用预训练文本嵌入将姓名表示为稠密向量并结合2020年人口普查数据训练神经网络，显著改进了标准BISG对被遗漏的不常见姓氏人群的种族概率预测性能。

    

    估计种族差异需要个人层面的种族数据，但由于收集此类信息的敏感性，这些数据往往无法获得。为解决这一问题，许多研究者利用贝叶斯改进姓氏地理编码，其关键依赖于人口普查姓氏数据。遗憾的是，这些数据仅能捕捉常见姓氏的种族-姓氏关系，遗漏了约10%的美国人口。我们表明，对于拥有这些被遗漏的不常见姓氏的个人，预测性能会大幅下降，因为标准BISG实现在这些情况下依赖于无信息量的通用先验。为解决这一局限性，我们提出嵌入驱动的BISG（eBISG），它使用预训练的文本嵌入将姓名表示为稠密向量，并在2020年人口普查姓氏和名字数据上训练神经网络，以估计人口普查未覆盖姓名的种族概率。我们比较了五种方法：

    arXiv:2604.22555v3 Announce Type: replace  Abstract: Estimating racial disparity requires individual-level race data, which are often unavailable due to the sensitivity of collecting such information. To address this problem, many researchers utilize Bayesian Improved Surname Geocoding (BISG), which have critically relied on Census surname data. Unfortunately, these data capture race-surname relationships only for common surnames, omitting approximately 10% of the US population. We show that predictive performance degrades substantially for individuals with such omitted, uncommon surnames because standard BISG implementation relies on a uninformative generic prior in these cases. To address this limitation, we propose embedding-powered BISG (eBISG), which uses pre-trained text embeddings to represent names as dense vectors and trains neural networks on 2020 Census surname and first-name data to estimate race probabilities for names not covered in the Census. We compare five approaches:
    
[^223]: 面向基于大语言模型语音识别的对话音频上下文抽象压缩

    Abstract Compression of Conversational Audio Context for LLM-based ASR

    [https://arxiv.org/abs/2603.26246](https://arxiv.org/abs/2603.26246)

    提出抽象压缩方法，将对话历史音频压缩为固定数量的潜在标记并保留转录文本，使基于LLM的语音识别以更小开销提升上下文实体的识别效果。

    

    标准的基于大语言模型（LLM）的语音识别系统孤立地处理每个话语，限制了其利用对话上下文的能力。本文研究了来自先前对话轮次的多模态上下文能否改进基于LLM的语音识别（ASR），以及如何高效地表示这些上下文。我们发现，经过监督多轮训练后，对话上下文主要有助于上下文相关实体的识别。然而，以原始上下文为条件的开销很大，因为先前轮次的音频嵌入序列会随对话长度的增长而迅速膨胀。为解决这一问题，我们提出了抽象压缩（Abstract Compression），这是一种对先前轮次音频的学习式潜在抽象，用固定数量的潜在标记替换每个先前轮次的音频序列，同时显式保留相应的转录文本。在域内和域外测试集上，启用压缩的模型以更小的先前轮次音频开销，恢复了原始上下文条件化所带来的一部分收益。

    arXiv:2603.26246v2 Announce Type: replace-cross  Abstract: Standard LLM-based speech recognition systems process utterances in isolation, limiting their ability to leverage conversational context. Here we study whether multimodal context from prior turns improves LLM-based ASR and how to represent that context efficiently. We find that, after supervised multi-turn training, conversational context mainly helps with the recognition of contextual entities. However, conditioning on raw context is expensive because the prior-turn audio embedding sequence grows rapidly with conversation length. To address this, we propose Abstract Compression, a learned latent abstraction of prior-turn audio that replaces each prior-turn audio sequence with a fixed number of latent tokens while retaining corresponding transcripts explicitly. On both in-domain and out-of-domain test sets, the compression-enabled model recovers part of the gains of raw-context conditioning with a smaller prior-turn audio footp
    
[^224]: 检测是廉价的，路由是习得的：为什么基于拒绝的对齐评估会失败

    Detection Is Cheap, Routing Is Learned: Why Refusal-Based Alignment Evaluation Fails

    [https://arxiv.org/abs/2603.18280](https://arxiv.org/abs/2603.18280)

    该论文以中国语言模型的政治审查为自然实验，发现对齐机制真正作用于“从概念检测到行为策略”的路由环节而非概念检测本身，从而解释了仅基于拒绝行为的对齐评估为何会失效。

    

    当前的对齐评估主要衡量模型是否编码了危险概念，以及模型是否拒绝有害请求。但这两者都忽略了对齐通常真正发挥作用的层面：从概念检测到行为策略之间的路由。我们以源自中国的语言模型中的政治审查作为自然实验，使用探针、外科手术式消融和行为测试，对来自五个实验室的九个开源权重模型进行了研究。由此得出三个发现。第一，仅凭探针准确率不具有诊断意义：政治探针、零对照和置换基线都可以达到100%的准确率，因此对保留类别（held-out category）的泛化测试才是有信息量的检验。第二，外科手术式消融揭示了实验室特有的路由方式。在大多数被测试的模型中，移除政治敏感方向可以消除审查行为并恢复准确的事实输出；而其中一个模型会出现虚构（confabulation），因为其架构将事实知识与审查机制纠缠在了一起。跨模型……

    arXiv:2603.18280v4 Announce Type: replace-cross  Abstract: Current alignment evaluation mostly measures whether models encode dangerous concepts and whether they refuse harmful requests. Both miss the layer where alignment often operates: routing from concept detection to behavioral policy. We study political censorship in Chinese-origin language models as a natural experiment, using probes, surgical ablations, and behavioral tests across nine open-weight models from five labs. Three findings follow. First, probe accuracy alone is non-diagnostic: political probes, null controls, and permutation baselines can all reach 100%, so held-out category generalization is the informative test. Second, surgical ablation reveals lab-specific routing. Removing the political-sensitivity direction eliminates censorship and restores accurate factual output in most models tested, while one model confabulates because its architecture entangles factual knowledge with the censorship mechanism. Cross-model
    
[^225]: CHIMERA：用于可泛化大语言模型推理的紧凑合成数据

    CHIMERA: Compact Synthetic Data for Generalizable LLM Reasoning

    [https://arxiv.org/abs/2603.00889](https://arxiv.org/abs/2603.00889)

    该论文提出CHIMERA——一个仅含9K样本的紧凑合成推理数据集，通过提供详细的长思维链轨迹、覆盖更广泛的科学学科并摆脱人工标注依赖，一举解决LLM推理训练中的冷启动、领域覆盖不足和标注瓶颈三大数据挑战。

    

    大语言模型（LLM）已经展现出卓越的推理能力，这主要得益于基于高质量推理数据的监督微调（SFT）和强化学习（RL）后训练。然而，在开放和可扩展的环境中复现和扩展这些能力受到三个以数据为中心的挑战的阻碍：（1）冷启动问题，源于缺乏包含详细、长思维链（CoT）轨迹的种子数据集来初始化推理策略；（2）领域覆盖有限，因为大多数现有的开源推理数据集集中于数学领域，对更广泛的科学学科覆盖不足；（3）标注瓶颈，前沿级推理任务的难度使得可靠的人工标注成本过于高昂甚至不可行。为了解决这些挑战，我们提出了CHIMERA，一个仅包含9K样本的紧凑合成推理数据集……

    arXiv:2603.00889v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have exhibited remarkable reasoning capabilities, largely enabled by supervised fine-tuning (SFT)- and reinforcement learning (RL)-based post-training on high-quality reasoning data. However, reproducing and extending these capabilities in open and scalable settings is hindered by three data-centric challenges: (1) the cold-start problem, arising from the lack of seed datasets with detailed, long Chain-of-Thought (CoT) trajectories needed to initialize reasoning policies; (2) limited domain coverage, as most existing open-source reasoning datasets are concentrated in mathematics, with limited coverage of broader scientific disciplines; and (3) the annotation bottleneck, where the difficulty of frontier-level reasoning tasks makes reliable human annotation prohibitively expensive or infeasible. To address these challenges, we introduce CHIMERA, a compact synthetic reasoning dataset of 9K samples desi
    
[^226]: TTCS：面向自我进化的测试时课程合成

    TTCS: Test-Time Curriculum Synthesis for Self-Evolving

    [https://arxiv.org/abs/2601.22628](https://arxiv.org/abs/2601.22628)

    TTCS提出了一种共同进化的测试时训练框架，通过问题合成器与推理求解器的迭代协同优化，生成难度递增的课程化问题变体，从而提升大语言模型在困难推理问题上的能力。

    

    测试时训练为提升大型语言模型（LLM）的推理能力提供了一种很有前景的方法，它仅利用测试问题来适配模型。然而，现有方法在应对困难推理问题时存在两大障碍：原始测试问题往往过于困难，难以产生高质量的伪标签；同时测试集规模有限，导致持续的在线更新容易不稳定。为了解决这些局限，我们提出了TTCS——一个共同进化的测试时训练框架。具体而言，TTCS从同一个预训练模型初始化两个策略：问题合成器与推理求解器。这两个策略通过迭代优化实现共同进化：合成器以测试问题为条件，生成难度逐步提升的问题变体，为求解器的当前能力量身打造结构化课程；而求解器则利用由（原文此处截断）计算得到的自洽性奖励来更新自身。

    arXiv:2601.22628v3 Announce Type: replace-cross  Abstract: Test-Time Training offers a promising way to improve the reasoning ability of large language models (LLMs) by adapting the model using only the test questions. However, existing methods struggle with difficult reasoning problems for two reasons: raw test questions are often too difficult to yield high-quality pseudo-labels, and the limited size of test sets makes continuous online updates prone to instability. To address these limitations, we propose TTCS, a co-evolving test-time training framework. Specifically, TTCS initializes two policies from the same pretrained model: a question synthesizer and a reasoning solver. These policies evolve through iterative optimization: the synthesizer generates progressively challenging question variants conditioned on the test questions, creating a structured curriculum tailored to the solver's current capability, while the solver updates itself using self-consistency rewards computed from
    
[^227]: 澄清或回答：面向上下文欠规范的智能体式视觉问答的强化学习

    Clarify or Answer: Reinforcement Learning for Agentic VQA with Context Under-specification

    [https://arxiv.org/abs/2601.16400](https://arxiv.org/abs/2601.16400)

    该论文提出基于强化学习的CoA智能体框架，使视觉问答模型在遇到上下文欠规范的歧义问题时先判断是否需要澄清，再通过迭代生成澄清问题来消除歧义，从而避免直接回答带来的自信但错误的预测。

    

    真实世界中的视觉问答（VQA）可能是依赖上下文的：图像-问题对可能是欠规范（under-specified）的，即正确答案取决于图像中无法观察到的外部信息。在这种情况下，直接回答可能导致自信但错误的预测。我们提出了“澄清或回答”（Clarify-or-Answer, CoA），这是一个“提问或回答”智能体，它将“是否提问或回答”的决策与“需要时问什么”分别建模。CoA 首先判断是否有必要进行澄清；如果需要，它会生成一个或多个澄清问题，以迭代方式丰富上下文，直到歧义被消除。为了验证该框架，我们引入了 CONTEXTCLARIFY 数据集，其中包含具有歧义的 VQA 问题以及一组非歧义问题的对照集。我们进一步提出了 GRPO-CR（澄清推理，Clarification Reasoning），这是一种强化学习方法，通过多种奖励信号来优化澄清问题的生成，鼓励……

    arXiv:2601.16400v2 Announce Type: replace  Abstract: Real-world visual question answering (VQA) may be context-dependent: an image-question pair could be under-specified, such that the correct answer depends on external information that is not observable in the image. In such cases, directly answering can lead to confident but incorrect predictions. We propose Clarify-or-Answer (CoA), an ask-or-answer agent that separately models the decision to ask or answer, and what to ask if needed. CoA first determines whether clarification is necessary; if so, it generates one or more clarification questions to iteratively enrich the context until ambiguity is resolved. To validate this framework, we introduce CONTEXTCLARIFY, a dataset of ambiguous VQA questions along with a contrast set of non-ambiguous questions. We further introduce GRPO-CR(Clarification Reasoning), a reinforcement learning approach that optimizes clarification question generation with multiple reward signals encouraging well-
    
[^228]: 提问、回答与检测：基于问题条件化混合专家模型的角色扮演大语言模型人格检测

    Ask, Answer, and Detect: Role-Playing LLMs for Personality Detection with Question-Conditioned Mixture-of-Experts

    [https://arxiv.org/abs/2512.08814](https://arxiv.org/abs/2512.08814)

    提出ROME框架，通过大语言模型角色扮演模拟用户回答心理测量问卷，并利用问题条件化混合专家模块提供可解释的中间监督，从而将心理学知识融入社交媒体人格检测。

    

    理解人类人格对个性化推荐和心理健康评估非常重要。现有人格检测方法通常将社交媒体帖子编码为用户表示并预测人格标签。尽管大语言模型（LLM）取得了进展，这些方法仍面临标签稀缺导致的监督有限，以及用户语言与心理构念之间映射不清晰的问题。我们提出了ROME，一个将心理学知识融入人格检测的框架。受标准化自评测试的启发，ROME利用大语言模型角色扮演来模拟用户对经过验证的心理测量问卷的回答。这些问题级别的答案提供了可解释的、有问卷依据的证据和中间监督。一个问题条件化的混合专家（Mixture-of-Experts）模块将帖子和问题表示进行联合路由，在显式监督下预测问卷答案。

    arXiv:2512.08814v2 Announce Type: replace  Abstract: Understanding human personality is important for personalized recommendation and mental health assessment. Existing personality detection methods typically encode social media posts into user representations and predict personality labels. Despite advances in large language models (LLMs), these methods face limited supervision from scarce labels and an unclear mapping between user language and psychological constructs. We propose ROME, a framework that incorporates psychological knowledge into personality detection. Inspired by standardized self-assessment tests, ROME uses LLM role-playing to simulate users' responses to validated psychometric questionnaires. These question-level answers provide interpretable, questionnaire-grounded evidence and intermediate supervision. A question-conditioned Mixture-of-Experts module jointly routes post and question representations to predict questionnaire answers under explicit supervision. The pr
    
[^229]: 大语言模型的注意力对齐推理

    Attention-Aligned Reasoning for Large Language Models

    [https://arxiv.org/abs/2510.03223](https://arxiv.org/abs/2510.03223)

    提出了ATAR推理方法，通过利用固有推理结构引导大语言模型的注意力，避免关键中间步骤和原始提示被埋没，在六个基准测试中最高实现15.39%的绝对提升，并使非推理模型达到与同规模推理模型相当甚至更好的性能。

    

    大语言模型（LLM）在解决复杂任务时往往会生成长推理链。然而，随着推理链的延伸，关键的中间步骤和原始提示词会被埋没在上下文中，得不到足够的注意力，从而导致错误。在本工作中，我们提出了ATAR，这是一种新颖的推理方法，它利用固有的推理结构来引导大语言模型的注意力。我们的实验表明，ATAR在六个基准测试中优于最先进（SOTA）方法，实现了高达15.39%的绝对提升。此外，借助ATAR，“非推理”模型在大多数基准测试中取得了与同规模推理模型相当甚至更好的性能。最后，我们的消融研究表明，注意力对齐组件贡献显著，且这些改进在不同的注意力引导后端下均能保持。

    arXiv:2510.03223v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) tend to generate a long reasoning chain when solving complex tasks. However, as the reasoning chain extends, critical intermediate steps and the original prompt will be buried in the context, receiving insufficient attention and leading to errors. In this work, we present ATAR, a novel reasoning method that leverages the inherent reasoning structure to steer LLM attention. Our experiments show that ATAR outperforms SOTA methods across six benchmarks, achieving up to 15.39% absolute improvement. Furthermore, with ATAR, "non-reasoning" models achieve comparable or even better performance compared to reasoning models of the same size in most benchmarks. Finally, our ablation studies show that the attention alignment component contributes significantly, and that these improvements are persist under different attentionsteering backends.
    
[^230]: 网络安全命名实体识别中面向跨数据集泛化的标签统一

    Label Unification for Cross-Dataset Generalization in Cybersecurity NER

    [https://arxiv.org/abs/2507.13870](https://arxiv.org/abs/2507.13870)

    该论文研究了四个网络安全NER数据集的标签统一以提升数据可用性，但实验表明统一训练、多头模型和图迁移模型均未能稳定超越单数据集基线。

    

    网络安全命名实体识别（NER）领域缺乏标准化的标签，这使得数据集难以合并使用。我们研究了四个网络安全数据集之间的标签统一，以提高数据资源的可用性。我们进行了粗粒度的标签统一，并使用RoBERTa模型开展两两跨数据集评估。对预测结果的定性分析揭示了错误、局限性以及数据集之间的差异。为解决标签统一的局限性，我们提出了替代架构，包括多头模型和基于图的迁移模型。在我们选定的映射和训练设置下，统一训练在全部四个开发集上的表现均低于相应的单数据集基线。具有权重共享的多头模型相比单数据集基线没有带来一致的收益，基于图的迁移模型也未显示出一致的收益。

    arXiv:2507.13870v3 Announce Type: replace  Abstract: The field of cybersecurity NER lacks standard-ized labels, making it challenging to combine datasets. We investigate label unification across four cybersecurity datasets to increase data resource usability. We perform a coarse-grained label unification and conduct pairwise cross-dataset evaluations using RoBERTa models. Qualitative analysis of predictions reveals errors, limitations, and dataset differences. To address unification limitations, we propose alternative architectures including a multi-head model and a graph-based transfer model. Under our selected mapping and training setup, unified training underperformed the corresponding single-dataset baselines on all four development sets. The multi-head model with weight sharing provides no consistent benefits over single-dataset baselines, and the graph-based transfer model shows no consistent benefits.
    
[^231]: 多模态生成模型中的生成类别与技术综述

    A Survey of Generative Categories and Techniques in Multimodal Generative Models

    [https://arxiv.org/abs/2506.10016](https://arxiv.org/abs/2506.10016)

    本综述系统分类了六类文本条件的多模态生成模态（图像、音乐、视频、人体动作、3D物体等），分析了自监督学习、混合专家、RLHF和思维链提示四种技术在各模态中的适配机制，并围绕忠实性、组合性和鲁棒性得出四项关键发现，包括偏好增益未必提升文本相关性、奖励优化可能降低输出多样性等。

    

    多模态生成模型（MGMs）已迅速超越文本生成范畴，扩展至图像、音乐、视频、人体动作和3D物体等领域。本综述对六种主要的以文本为条件的生成模态进行了分类，并考察了四种技术——即自监督学习、混合专家、基于人类反馈的强化学习以及思维链提示——如何在这些模态中被适配应用，从而区分共享机制与经实验验证的收益。我们围绕忠实性、组合性和鲁棒性来组织评估，并在每项研究的实验协议框架内解读证据。所审阅文献中报告的若干研究内比较支持四项发现：偏好增益未必能提升文本相关性，且奖励优化可能降低同一提示下的多样性；中间规划需要在最终输出中检验其是否得到实现；专家粒度的变化会改变机制及其存储方式（原文摘要在此处截断）……

    arXiv:2506.10016v4 Announce Type: replace-cross  Abstract: Multimodal generative models (MGMs) have rapidly expanded beyond text generation to encompass images, music, video, human motion, and 3D objects. This survey categorises six primary text-conditioned generative modalities and examines how four techniques, namely self-supervised learning, mixture of experts, reinforcement learning from human feedback, and chain-of-thought prompting, are adapted across them, distinguishing shared mechanisms from experimentally supported benefits. We organise evaluation around faithfulness, compositionality, and robustness, and interpret evidence within each study's protocol. Selected within-study comparisons reported in the reviewed literature support four findings: preference gains need not improve text relevance, and reward optimisation can reduce within-prompt diversity; intermediate plans require tests of their realisation in final outputs; expert granularity changes the mechanism and its stor
    
[^232]: 大语言模型对简单代码变换十分脆弱：提出CETBench——一个代码等价性检测基准

    LLMs are Brittle to Simple Code Transformations: Introducing CETBench - A Benchmark for Code-Equivalence Checking

    [https://arxiv.org/abs/2506.04019](https://arxiv.org/abs/2506.04019)

    本文提出了代码等价性检测基准CETBench，发现大语言模型即使面对简单的代码变换也会出现显著性能下降（在跨语言场景下尤为严重），并提出一种基于微调的简单方法来提升模型性能。

    

    我们研究了大型语言模型（LLM）判断两个程序在功能上是否等价的能力。这是一个重要问题，因为对代码等价性进行基准测试有助于评估LLM在代码重写和代码翻译等任务上的能力。为此，我们提出了CETBench（Code Equivalence with Transformations Benchmark，基于变换的代码等价性检测基准），它由一个程序仓库构建而成，这些程序可能解决相同或不同的任务。每个数据集实例通过采样一对程序并应用随机序列的预定义代码变换来生成，从而得到等价或非等价的程序对。我们的分析表明，即使是简单的变换也会导致最先进的LLM在代码等价性检测任务上出现显著的性能下降。在跨语言场景下（即比较用不同语言编写的程序时），这些挑战会被进一步放大。为了解决这一问题，我们提出了一种简单的基于微调的方法来提升LLM在该任务上的性能。

    arXiv:2506.04019v2 Announce Type: replace-cross  Abstract: We study how well LLMs can determine whether two programs are functionally equivalent. This is an important problem because benchmarking code equivalence helps assess LLM capability in tasks such as code rewriting and translation. To this end, we introduce CETBench - Code Equivalence with Transformations Benchmark - built from a repository of programs that may solve the same or different tasks. Each dataset instance is created by sampling a program pair and applying a random sequence of predefined code transformations, yielding either equivalent or non-equivalent pairs. Our analysis shows that even simple transformations cause a significant performance drop in state-of-the-art LLMs on code-equivalence checking. These challenges are further amplified in the cross-lingual setting when comparing programs written in different languages. To remedy this, we present a simple fine-tuning-based approach to boost LLM performance on the t
    
[^233]: 重新思考知识图谱嵌入中的泛化：实证研究与稀疏峰值正则化

    Rethinking Generalization in Knowledge Graph Embedding: An Empirical Study and Sparse Peak Regularization

    [https://arxiv.org/abs/2505.23442](https://arxiv.org/abs/2505.23442)

    本文从泛化角度对知识图谱嵌入的正则化进行了跨模型族的系统性实证研究，发现过拟合在各类KGE模型中普遍存在，并提出稀疏峰值正则化方法来显著提升模型性能。

    

    知识图谱（KGs）为实体和关系提供了结构化表示，支撑着搜索、推荐和问答等数据管理与Web应用。由于Freebase和Wikidata等Web规模的知识图谱天然存在不完整性，知识图谱嵌入（KGE）将实体和关系映射到连续向量空间中，并被广泛用于通过链接预测来推断缺失的事实。尽管KGE模型设计已取得广泛进展，但正则化的作用尚未得到系统性考察，尤其是在张量分解模型之外的模型上。本文的主要贡献是从泛化的角度对KGE的正则化进行了全面的实证研究。在基于平移、基于张量分解、基于图神经网络（GNN）、时序以及多模态等各类KGE模型上，我们表明过拟合在各模型族中普遍存在，而适当的正则化可以显著提升（原文在此处截断）。

    arXiv:2505.23442v2 Announce Type: replace-cross  Abstract: Knowledge graphs (KGs) provide structured representations of entities and relations, and underpin data management and Web applications such as search, recommendation, and question answering. Since Web-scale KGs such as Freebase and Wikidata are inherently incomplete, knowledge graph embedding (KGE) maps entities and relations into continuous vector spaces and is widely used to infer missing facts through link prediction. Despite extensive progress in KGE model design, the role of regularization has been less systematically examined, especially beyond tensor-decomposition models. The primary contribution of this paper is a comprehensive empirical study of regularization for KGE from the perspective of generalization. Across translation-based, tensor-decomposition-based, GNN-based, temporal, and multimodal KGE models, we show that overfitting is common across model families and that appropriate regularization can substantially im
    
[^234]: KETCHUP：面向序列知识蒸馏的K步回报估计方法

    KETCHUP: K-Step Return Estimation for Sequential Knowledge Distillation

    [https://arxiv.org/abs/2504.19024](https://arxiv.org/abs/2504.19024)

    该论文提出KETCHUP方法，通过多步使用贝尔曼最优方程诱导K步回报来降低梯度估计方差，显著提升了文本生成任务中基于强化学习的知识蒸馏效果，尤其适用于大型学生模型。

    

    我们提出了一种新颖的K步回报估计方法（称为KETCHUP），用于文本生成任务中基于强化学习（RL）的知识蒸馏（KD）。我们的核心思想是通过多步使用贝尔曼最优方程来诱导出K步回报。理论分析表明，这种K步形式降低了梯度估计的方差，从而改进了强化学习的优化效果，尤其是在学生模型规模较大时。在三个文本生成任务上的实证评估表明，我们的方法在标准任务指标和基于大语言模型（LLM）的评估中均取得了优越的性能。这些结果表明，我们的K步回报诱导方法为增强大语言模型研究中基于强化学习的知识蒸馏提供了一个有前景的方向。

    arXiv:2504.19024v2 Announce Type: replace  Abstract: We propose a novel k-step return estimation method (called KETCHUP) for Reinforcement Learning(RL)-based knowledge distillation (KD) in text generation tasks. Our idea is to induce a K-step return by using the Bellman Optimality Equation for multiple steps. Theoretical analysis shows that this K-step formulation reduces the variance of the gradient estimates, thus leading to improved RL optimization especially when the student model size is large. Empirical evaluation on three text generation tasks demonstrates that our approach yields superior performance in both standard task metrics and large language model (LLM)-based evaluation. These results suggest that our K-step return induction offers a promising direction for enhancing RL-based KD in LLM research.
    
[^235]: 计算类型学

    Computational Typology

    [https://arxiv.org/abs/2504.15642](https://arxiv.org/abs/2504.15642)

    本文介绍了计算统计建模如何助力语言类型学研究，使大规模语言数据分析与语言结构和演化假设的检验成为可能。

    

    类型学是语言学的一个分支领域，专注于基于语言的结构特征对语言进行研究和分类。与考察语言之间历史关系的谱系分类不同，类型学试图通过识别被称为“共性”的共同属性和模式来理解人类语言的多样性。近年来，计算方法在类型学研究中发挥着越来越重要的作用，使得大规模语言数据的分析以及关于语言结构和演化的假设检验成为可能。本文展示了计算统计建模在类型学研究中的优势。

    arXiv:2504.15642v3 Announce Type: replace  Abstract: Typology is a subfield of linguistics that focuses on the study and classification of languages based on their structural features. Unlike genealogical classification, which examines the historical relationships between languages, typology seeks to understand the diversity of human languages by identifying common properties and patterns, known as universals. In recent years, computational methods have played an increasingly important role in typological research, enabling the analysis of large-scale linguistic data and the testing of hypotheses about language structure and evolution. This article provides an illustration of the benefits of computational statistical modeling in typology.
    
[^236]: MedHal：一个用于医学幻觉检测的合成数据集

    MedHal: a Synthetic Dataset for Medical Hallucination Detection

    [https://arxiv.org/abs/2504.08596](https://arxiv.org/abs/2504.08596)

    MedHal是一个涵盖内在与外在幻觉的大规模合成医学数据集，用于训练和评估医学文本幻觉检测模型，基于该数据集训练的基线模型优于通用幻觉检测方法。

    

    幻觉是指AI系统生成非事实性内容的现象，在医学场景中会带来严重风险，因为错误可能直接影响患者的治疗效果。我们提出了MedHal，这是一个大规模数据集，专门用于评估模型能力并训练模型执行医学文本中的幻觉检测任务。当前的幻觉检测方法在应用于医学等专业领域时面临重大局限，可能造成灾难性后果。MedHal通过纳入多样化的医学文本来源和任务（涵盖内在幻觉与外在幻觉），并提供大量适合训练医学幻觉检测模型的数据样本来解决这一问题。我们通过训练和评估一个基线医学幻觉检测模型证明了MedHal的实用性，其表现优于通用幻觉检测方法。该资源为医学领域的幻觉检测研究和应用提供了重要支持。

    arXiv:2504.08596v3 Announce Type: replace-cross  Abstract: Hallucination, the generation of non factual content by AI systems, poses serious risks in medical contexts, where errors can directly affect patient outcomes. We present MedHal, a large-scale dataset specifically designed to assess capabilities and train models on the task of hallucination detection in medical texts. Current hallucination detection methods face significant limitations when applied to specialized domains like medicine, where they can have disastrous consequences. MedHal addresses this issue by incorporating diverse medical text sources and tasks covering both intrinsic and extrinsic hallucinations, and by providing a substantial volume of data samples suitable for training medical hallucination detection models. We demonstrate MedHal's utility by training and evaluating a baseline medical hallucination detection model, showing improvements over general-purpose hallucination detection approaches. This resource e
    
[^237]: 语言模型能否替代程序员进行编码？REPOCOD的回答是“尚不能”

    Can Language Models Replace Programmers for Coding? REPOCOD Says 'Not Yet'

    [https://arxiv.org/abs/2410.21647](https://arxiv.org/abs/2410.21647)

    该论文提出了REPOCOD基准，包含来自11个真实热门项目的980个Python完整函数生成任务，以评估语言模型在真实世界编码任务中的能力，结果显示当前语言模型尚不能替代程序员。

    

    最近，许多仓库级代码生成基准（如CoderEval、DevEval、RepoEval、RepoBench和LongCodeArena）相继出现，用于评估大型语言模型（LLM）在HumanEval和MBPP等独立基准之外的能力。由此产生一个自然的问题：LLM在真实世界编码任务中的表现是否会与其在这些基准上的表现相当？遗憾的是，目前无法回答这个问题，因为这些基准由简短的补全任务、合成示例组成，或仅关注规模有限的仓库，无法代表真实世界的编码任务。为解决这些挑战，我们创建了REPOCOD，这是一个Python代码生成基准，包含来自真实世界大型项目中具有现实依赖关系的复杂任务，并配有适合评估源代码的指标。该基准包含来自11个热门项目的980个完整函数生成任务，其中50.8%的任务需要仓库级上下文信息。

    arXiv:2410.21647v5 Announce Type: replace-cross  Abstract: Recently, a number of repository-level code generation benchmarks-such as CoderEval, DevEval, RepoEval, RepoBench, and LongCodeArena-have emerged to evaluate the capabilities of large language models (LLMs) beyond standalone benchmarks like HumanEval and MBPP. Thus, a natural question is, would LLMs have similar performance in real world coding tasks as their performance in these benchmarks? Unfortunately, one cannot answer this question, since these benchmarks consist of short completions, synthetic examples, or focus on limited scale repositories, failing to represent real-world coding tasks.   To address these challenges, we create REPOCOD, a Python code-generation benchmark containing complex tasks with realistic dependencies in real-world large projects and appropriate metrics for evaluating source code. It includes 980 whole-function generation tasks from 11 popular projects, 50.8% of which require repository-level contex
    
[^238]: WAFFLE：面向自动化前端开发的多模态模型微调

    WAFFLE: Finetuning Multi-Modal Models for Automated Front-End Development

    [https://arxiv.org/abs/2410.18362](https://arxiv.org/abs/2410.18362)

    Waffle提出了一种结合结构感知注意力机制与对比微调的多模态模型微调策略，显著提升了从UI设计图自动生成HTML代码的效果。

    

    网页开发涉及将UI设计转化为功能性的网页，由于HTML的层级结构和样式较为复杂，无论对初学者还是有经验的开发者来说都颇具难度。虽然大型语言模型（LLMs）在生成源代码方面已展现出潜力，但在UI到HTML代码生成中仍存在两大挑战：（1）如何有效地为LLM表示HTML的层级结构，以及（2）如何弥合UI设计的视觉特性与HTML代码文本格式之间的鸿沟。为应对这些挑战，我们提出了Waffle，这是一种新的微调策略，它采用结构感知注意力机制来增强LLM对HTML结构的理解，并使用对比微调方法来对齐LLM对UI图像和HTML代码的理解。经Waffle微调后的模型表现出显著提升：HTML匹配度最高提高9.00个百分点，CW-SSIM提高0.0982，CLIP提高32.99，另一指标提高27.12个百分点。

    arXiv:2410.18362v4 Announce Type: replace-cross  Abstract: Web development involves turning UI designs into functional webpages, which can be difficult for both beginners and experienced developers due to the complexity of HTML's hierarchical structures and styles. While Large Language Models (LLMs) have shown promise in generating source code, two major challenges persist in UI-to-HTML code generation: (1) effectively representing HTML's hierarchical structure for LLMs, and (2) bridging the gap between the visual nature of UI designs and the text-based format of HTML code. To tackle these challenges, we introduce Waffle, a new fine-tuning strategy that uses a structure-aware attention mechanism to improve LLMs' understanding of HTML's structure and a contrastive fine-tuning approach to align LLMs' understanding of UI images and HTML code. Models fine-tuned with Waffle show up to 9.00 pp (percentage point) higher HTML match, 0.0982 higher CW-SSIM, 32.99 higher CLIP, and 27.12 pp higher
    
[^239]: 台风：针对预训练语言模型的有效特定任务屏蔽策略

    Typhoon: Towards an Effective Task-Specific Masking Strategy for Pre-trained Language Models. (arXiv:2303.15619v1 [cs.CL])

    [http://arxiv.org/abs/2303.15619](http://arxiv.org/abs/2303.15619)

    本文探讨了一种针对预训练语言模型的任务特定的屏蔽框架，称为Typhoon，可在GLUE基准数据集上实现卓越的下游任务性能，尤其在MRPC数据集上表现优异。

    

    通过利用图形处理单元所能提供的高度并行性，变压器架构使自然语言处理领域取得了巨大的进展。在传统的屏蔽语言模型中，使用特殊的MASK标记来提示模型从周围单词中收集情境信息以恢复原本隐藏的信息。在本文中，我们探讨了一种预训练大型语言模型的任务特定的屏蔽框架，以在GLUE基准数据集上实现卓越的下游任务性能。我们基于记号输入梯度开发了自己的屏蔽算法Typhoon，并将其与其他标准基线进行比较。我们发现，Typhoon在MRPC数据集上的性能与整体字屏蔽相当。我们的实现可以在公共Github库中找到。

    Through exploiting a high level of parallelism enabled by graphics processing units, transformer architectures have enabled tremendous strides forward in the field of natural language processing. In a traditional masked language model, special MASK tokens are used to prompt our model to gather contextual information from surrounding words to restore originally hidden information. In this paper, we explore a task-specific masking framework for pre-trained large language models that enables superior performance on particular downstream tasks on the datasets in the GLUE benchmark. We develop our own masking algorithm, Typhoon, based on token input gradients, and compare this with other standard baselines. We find that Typhoon offers performance competitive with whole-word masking on the MRPC dataset. Our implementation can be found in a public Github Repository.
    

