# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Imagine3D-LLM: Teaching MLLMs to Imagine 3D Scenes Before Answering](https://arxiv.org/abs/2609.38177) | 该论文受人类空间推理过程启发，提出Imagine3D-LLM，教会多模态大语言模型在回答问题前通过粗略识别跨视角共同物体、推断视角间相对几何关系来构建紧凑的粗略3D场景布局，而非依赖细粒度几何线索，从而提升多视角3D推理能力。 |
| [^2] | [STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization](https://arxiv.org/abs/2609.38169) | STEPQuant揭示了Delta规则循环状态量化误差在时间（记忆寿命）和空间（键行/值列）两个维度上的差异化影响，并提出一种时空后训练量化框架，根据误差大小与记忆寿命自适应分配精度，从而在低比特量化下保持模型精度。 |
| [^3] | [EmoRES-TTS: Residual-Enhanced Vector Steering for Emotional Speech Generation](https://arxiv.org/abs/2609.38157) | 该论文发现情感向量可分解为使语音脱离平淡表达的共享分量和引导生成指定情感的残差分量，并据此提出无需重新训练骨干网络的 EmoRES 方法来分别控制这两个分量，从而更可靠地生成指定情感的语音。 |
| [^4] | [Beyond the Timeline: Augmenting Long-Video Memory with Grounded Entity Biographies](https://arxiv.org/abs/2609.38155) | 该论文提出长视频记忆框架GEB（接地实体传记），将跨片段对同一物理实例的视觉观测聚合为可检索的实体传记，使模型在长视频问答中能够基于身份关联跨时间追踪特定实体。 |
| [^5] | [Pretraining Latent Information Feedback Transformers with Teacher Supervision](https://arxiv.org/abs/2609.38149) | 提出LIFT架构与训练方法，通过教师强制方式将循环状态学习转化为预测问题，使Transformer语言模型在预训练阶段能够跨生成步骤传播潜在状态信息，突破了传统前馈架构中信息只能通过解码token向下传递的瓶颈。 |
| [^6] | [Learning Meta-Skills for Agent Harness Design in Test-Time AI4AI](https://arxiv.org/abs/2609.38143) | 提出Meta-Skill方法，使权重固定的Builder模型从Target的执行反馈中学习可复用的支持原则，为未见任务构建更好的执行环境，相比无技能构建和直接交付技能库分别提升8.95和12.02个百分点。 |
| [^7] | [AdviSD: Learning to Advise Frontier LLMs via Targeted Multi-Turn Self-Distillation](https://arxiv.org/abs/2609.38142) | 该论文从理论上证明“不改变执行结果的纠正”可能限制建议者的学习，并据此提出AdviSD方法，将基于结果的强化学习与来自反馈条件副本的选择性自蒸馏相结合，以提升可训练建议者引导冻结大语言模型执行器的能力。 |
| [^8] | [LongHarness Bench: Stress-Testing Language Model Harnesses for Long-Context Reasoning](https://arxiv.org/abs/2609.38137) | 提出LongHarness Bench基准，通过需要多样化检索策略与全局-局部自适应推理的任务，同时评估长上下文语言模型框架的有效性与效率，并揭示不同处理策略之间显著的准确率—成本权衡。 |
| [^9] | [From Routing Signals to Selective Review: Visual regrounding in MoE VLMs](https://arxiv.org/abs/2609.38111) | 该论文提出首个利用MoE视觉语言模型内部路由信号在生成前检测“目标缺失定位失败”的框架，仅用路由概率训练的简单线性检测器即可达到近完美的检测效果（ROC-AUC高达0.9988），并据此选择性触发审查提示实现纠正。 |
| [^10] | [How Local Mixing Encodes Relative Position in Global NoPE Attention](https://arxiv.org/abs/2609.38109) | 本文通过理论与实证分析揭示，混合Transformer架构中滑动窗口注意力（SWA）和门控线性注意力等局部混合层会在残差流中诱导近因偏差，从而使不使用显式位置编码的全局NoPE注意力层能够隐式地编码相对位置信息。 |
| [^11] | [Correct Answers, Invalid Traces: What Verifiable Grade-School Math Reveals About Chain-of-Thought Traces](https://arxiv.org/abs/2609.38107) | 研究通过可程序化验证的 iGSM 数学基准发现，答案正确性并不能可靠证明推理有效性——在分布外最难的实例上，31.6% 的正确答案伴随无效的思维链轨迹。 |
| [^12] | [Pruning for Efficiency, Paying in Fairness: Demographic Disparities in Pruned Speech-LLMs](https://arxiv.org/abs/2609.38106) | 本研究首次系统揭示了音频编码器剪枝会不成比例地损害语音大语言模型中弱势人口群体的识别性能，导致群体间差距成倍扩大，而仅依赖总体词错误率的评估方式会掩盖这一公平性问题。 |
| [^13] | [Effective Dense Retrieval using Only In-Context Examples](https://arxiv.org/abs/2609.38099) | 本文提出RICE方法，仅通过少量上下文示例提示LLM，无需任何检索器训练即可提取高质量稠密表示，显著提升基于提示的LLM嵌入在稠密检索任务中的准确性。 |
| [^14] | [Gender bias across LLMs is common and highly heterogenous](https://arxiv.org/abs/2609.38036) | 该研究通过两种实验范式测试了来自九个厂商的十款大语言模型，发现性别偏见在模型间普遍存在但高度异质——部分模型表现出反刻板印象的性别归因模式，另一些模型则在道德判断中与人类保护女性免受伤害的倾向一致。 |
| [^15] | [Layer-Informed Fine-Tuning via Three-Stage Functional Segmentation of LLMs](https://arxiv.org/abs/2609.38027) | 本文提出大语言模型各层在概念化、推理和文本化上存在结构化分工的假设，并据此提出层信息感知微调方法（LIFT），借助敏感性分析定位瓶颈功能阶段，仅更新功能关键层即可实现高效有效的微调。 |
| [^16] | [Dr. OPD: Learning What to Follow for Optimal On-Policy Distillation of Large Language Models](https://arxiv.org/abs/2609.38025) | 提出Dr. OPD框架，将同策略蒸馏中的词元级教师监督权重选择建模为双层优化问题，通过交替更新词元权重与学生策略，识别并强化对学生性能最有价值的教师信号，从而最大化大语言模型蒸馏效果。 |
| [^17] | [Auditable Long-Term Memory: A Deterministic Retrieval Chain Measured at 479/475 of 500 on LongMemEval-S](https://arxiv.org/abs/2609.38021) | 该论文提出一种以确定性检索链（混合候选检索、交叉编码器重排序、覆盖优先数据包编译）为核心、LLM仅作为可替换最终阅读器的可审计长期记忆系统，在LongMemEval-S上两次评测分别获得479/500和475/500，成绩跨越Chronos High的478/500，但差异不足以证明优越性或等效性。 |
| [^18] | [BITEM at the NTCIR-19 R2C2 Task: Predicting Confidence from Agentic RAG Pipeline Signals](https://arxiv.org/abs/2609.37993) | 该论文提出一种由编排器根据代理式RAG流水线运行痕迹（蕴含级联核验证据、多轮去重检索等信号）来计算答案置信度、而无需模型自我评估的方法，在NTCIR-19 R2C2任务中取得第4和第5名，且多轮融合检索在多跳问题上增益最大。 |
| [^19] | [$S^3$: Spectral Null-Space Swap Makes Reasoning Models Efficient](https://arxiv.org/abs/2609.37976) | 论文首次发现推理能力的关键在于思考模型权重中位于非思考模型主导子空间零空间内的分量，并提出无需训练的谱零空间交换方法S³，通过配对检查点组合在保持精度的同时大幅提升推理效率。 |
| [^20] | [On Trajectory-Aware Training for Masked Diffusion Language Models](https://arxiv.org/abs/2609.37974) | 该论文提出PUMBA统一框架，通过在策略诱导轨迹的连续步骤上训练去噪器、跨步骤传递信息并利用时间反向传播进行联合优化，实现了掩码扩散语言模型的轨迹感知训练，解决了训练与推理条件不匹配的问题。 |
| [^21] | [SelfSearch: Reward-Free Search for Self-Improving Agents](https://arxiv.org/abs/2609.37968) | SelfSearch提出了一种免奖励的自我改进搜索方法，智能体通过利用以往自我修改回合的记录（包含推理、工具操作和结果）来改进自身，无需昂贵的下游评估即可在多个模型-基准测试设置中显著提升成功率。 |
| [^22] | [Learning What to Remember: Long-horizon Counterfactual Memory Optimization](https://arxiv.org/abs/2609.37930) | 提出记忆增益策略优化（MGPO），通过反事实地归因每次记忆重写对当前与未来下游效用的边际贡献，将延迟的记忆效用转化为直接学习信号，在文档级信息抽取中提升性能的同时将平均记忆长度减少近80%。 |
| [^23] | [Time-Anchored Diffusion Language Models: Latent-Space Caching for Fast Generation](https://arxiv.org/abs/2609.37924) | 提出基于时间的自监督锚定机制，通过两阶段架构缓存并复用潜在锚点状态，使扩散语言模型无需监督式标记目标即可实现快速生成。 |
| [^24] | [Overcoming Scaling Limits in On-Policy Self-Distillation for LLM Reasoning](https://arxiv.org/abs/2609.37915) | 该论文通过因子分析发现在线策略自蒸馏中支架（轨迹）正确性比特权上下文正确性对下游性能影响更大，据此提出OASIS方法，通过主要监督经过验证的在线策略轨迹来克服自蒸馏的规模限制。 |
| [^25] | [The Unequal Influence of Bad Advice: Using Training Data Attribution to Modulate Emergent Misalignment](https://arxiv.org/abs/2609.37914) | 本文首次利用训练数据归因定量估计每个有害样本对“涌现性失对齐”（EM）的贡献，发现坏样本的影响并不均等，且基于归因分数的数据过滤可显著增强或减弱模型失对齐的程度。 |
| [^26] | [It's All Training: A Fully Synthetic Single-Stage Recipe for LLMs](https://arxiv.org/abs/2609.37891) | 该论文提出首个开源的全合成语料库SYNTH，仅从58,698篇维基百科文章出发，通过结构化扩增将预训练、中期训练与后训练整合为单一训练阶段。 |
| [^27] | [Zero-shot Dependency Parsing with Unsupervised Cross-Lingual Bootstrapping](https://arxiv.org/abs/2609.37883) | 本文提出一种跨语言无监督引导方法，通过增强预训练语言模型内部的句法知识，显著提升了低资源语言上零样本依存句法分析的性能，并通过无参数树探测测试证明了模型识别句法结构鲁棒性的提升。 |
| [^28] | [How Many Labels Does a Language Need? Annotation Budgets and Cross-Lingual Pooling for African-Language Text Classification](https://arxiv.org/abs/2609.37882) | 该论文在28个语言-任务对上实证测算了非洲语言文本分类所需的标注预算，发现新闻主题分类约400个标签即可达到全量数据90%的性能，而情感分析需要数千个标签，且跨语言数据池化仅在标注预算很小时才有效。 |
| [^29] | [Retrieval Capacity of Self-Attention Under Competition](https://arxiv.org/abs/2609.37879) | 该论文提出一种无需重新训练的方法，通过仅保留注意力权重最高的token来估计自注意力的有效检索容量，发现相对较小的注意力选择集即可使模型损失接近全注意力基线，且所需集合大小随上下文扩展而增加、但占上下文比例下降。 |
| [^30] | [Learning Beyond What You Sample: Off-Policy-Aware Cross-Model Trajectory Exchange for RLVR](https://arxiv.org/abs/2609.37868) | 提出GRAFT框架，通过用同伴模型信息丰富的轨迹替换全失败的rollout组，并借助对等计算的优势与序列级兼容性权重控制跨模型失配，在无需更强教师模型的情况下实现异构模型间的相互学习，提升RLVR训练效果。 |
| [^31] | [It's Not What the Image Shows: Irrelevant Context Destabilises VLM Judges Without Informing Them](https://arxiv.org/abs/2609.37863) | 本文提出MIST误导性图像压力测试，揭示无关图像——无论与句义一致还是相反——都会以几乎相同的幅度（约20%）动摇VLM评判器的标签判断，且动摇幅度甚至超过删除“忽略图像”指令的影响，表明图像是作为干扰性上下文而非信息来源起作用，威胁了VLM替代人类标注者的可靠性。 |
| [^32] | [One Threshold Does Not Fit All Languages: Language-Conditional Deferral for Reliable and Efficient Low-Resource Text Classification](https://arxiv.org/abs/2609.37861) | 该论文揭示了基于跨语言汇总数据的单一置信度阈值无法在所有语言上兑现错误率承诺（如索马里语覆盖率仅77.5%），并提出按语言条件化的延迟机制，使低资源多语言文本分类在保证可靠性的同时更加高效。 |
| [^33] | [Storage Is Not Strategy: State-Conditioned Support Control for LLM Unlearning](https://arxiv.org/abs/2609.37858) | 该论文发现“存储目标知识”的参数未必是执行遗忘的最佳干预对象，提出基于实际遗忘更新预测效果的干预分数以及动态干预重排方法（DIR-R），在优化过程中按需自适应调整干预参数子集，从而显著提升大语言模型遗忘的效果。 |
| [^34] | [AnthroDial: Benchmarking LLM Anthropomorphism in Autonomous Social Interaction](https://arxiv.org/abs/2609.37853) | AnthroDial 提出了一个统一框架，通过 MindFlow 自主交互机制、CAPS-Eval 三维度评估体系和 SEEDS+DiAPO 可扩展训练范式，实现、评估并提升大语言模型在持续开放社交交互中的拟人化能力。 |
| [^35] | [Can Vision-Language Models Stay Helpful When Facing Implicit Risks? Intent-Privilege OPSD for Efficient Safety-Helpfulness Alignment](https://arxiv.org/abs/2609.37837) | 提出意图-特权策略内自蒸馏（OPSD）方法，在训练时利用基于证据的意图作为特权监督，使视觉-语言模型仅用1,447个安全示例（比标准偏好数据集少95%）就能识别跨模态隐性风险，在不一概拒绝的前提下实现安全与有用性的高效对齐。 |
| [^36] | [Can a Cacheable Decision Model Follow Rules?](https://arxiv.org/abs/2609.37832) | 本研究将决策模型从联合评分改为可缓存的独立编码后规则敏感性大幅丧失（recall@1 从 1.00 跌至 0.24），但通过针对性的反事实监督训练可以恢复较强的规则遵循能力，同时保留约5倍的缓存加速优势。 |
| [^37] | [The Geometry of Inference in Transformer Residual Streams](https://arxiv.org/abs/2609.37824) | 本研究通过比较中间残差状态与最终状态及其经验库，揭示了Transformer语言模型中预测表示主要通过渐进的方向对齐变化（而非欧氏距离的显著改变）逐渐变得对最终结果具有特异性，并建立了区分范数、对齐与终点几何作用的高维理论模型。 |
| [^38] | [Thinking in Depth, Speaking Directly: Recurrent Latent Reasoning for Paralinguistically Grounded Spoken Dialogue](https://arxiv.org/abs/2609.37818) | LoopSLM通过循环Transformer进行潜在推理，在无需显式思维链的情况下实现副语言接地的共情口语对话，有效缩小了感知与推理之间的鸿沟并降低了推理延迟。 |
| [^39] | [CompOrca: Corpus-Scale Compliance Labelling of Instruction-Tuning Data](https://arxiv.org/abs/2609.37807) | 该论文提出了 CompOrca，利用开源大模型评判器对整个 OpenOrca 语料库（超过 420 万条样本）进行五次独立判定，首次实现了语料库规模的合规性标注，并发布带投票计数的标注结果，以支持对拒答与不服从行为的研究。 |
| [^40] | [A Proposed Rubric for Evaluating Expressed Clinical Reasoning in Large Language Model Responses](https://arxiv.org/abs/2609.37788) | 该论文提出一个融合医学教育评估框架、临床大语言模型基准和通用LLM推理评估研究的多维评分量规，用于对大语言模型针对金标准临床案例的自由文本回答中表达的临床推理进行结构化评估。 |
| [^41] | [Selecting What Matters: Semantic Compression-Guided Selective Pooling for Long-Context Embeddings](https://arxiv.org/abs/2609.37782) | 提出免训练框架SCSP，通过为句子感知分块附加语义压缩提示并利用其注意力模式估计词元重要性，实现长上下文嵌入中的选择性池化，避免平均池化稀释关键语义信息。 |
| [^42] | [Which papyrus HTR is good enough? Character-error-rate tolerance of four papyrological tasks on Greek texts](https://arxiv.org/abs/2609.37755) | 该研究通过将模拟的不同字符错误率（1%—50%）的HTR输出用于文献类型识别、断代等四种纸草学任务，首次为古希腊纸草文献手写文本识别系统所需的精度建立了基准。 |
| [^43] | [Context Language Models](https://arxiv.org/abs/2609.37725) | 上下文语言模型（CLM）通过将自身上下文当作可自由编辑的文件来实现模型原生管理上下文，在多种任务上以更少的计算量超越了现有最先进的上下文管理策略。 |
| [^44] | [Predictive Geometry of Hidden Trajectories in Transformers](https://arxiv.org/abs/2609.37717) | 该论文证明仅解码器Transformer在成功验证轨迹附近的逐层“损失到终点”函数的局部二阶几何由隐藏状态空间上的回拉Fisher算子支配，其谱划分出输出敏感方向与预测零方向、界定出残差流的局部可观测子空间，并由此为因果Transformer导出衡量每个词元隐藏状态对预测敏感度的逐词元曲率分数。 |
| [^45] | [Billiger.de Products: A Bilingual Entity Matching Benchmark](https://arxiv.org/abs/2609.37713) | 本文提出了Billiger.de Products，一个涵盖十三个消费品类别的德英双语实体匹配基准数据集，填补了现有产品匹配基准以英语为主且品类单一的空白。 |
| [^46] | [Reader Proficiency Shapes Layer-wise Surprisal Profiles](https://arxiv.org/abs/2609.37688) | 该研究发现，大语言模型惊讶度预测能力在模型各层中的分布深度会因读者词汇熟练水平和眼动指标而异：词汇熟练度较低的读者在首次通过注视时长上呈现更深的预测深度，而总注视时长在两组读者中都呈现更深的预测深度。 |
| [^47] | [EngiWorld: What Can Frontier Agents Deliver in Professional Engineering Environments?](https://arxiv.org/abs/2609.37686) | 该论文提出了首个覆盖完整设计循环的工程智能体基准EngiWorld，涵盖1,301个专家任务、6个工程领域和26个专业软件平台，并引入以工件为中心的自动化评估方法来检验工程成果的几何有效性、物理可行性与规则合规性。 |
| [^48] | [When Models Don't Manipulate Manifolds: The Geometry of a Comparison Task](https://arxiv.org/abs/2609.37680) | 该论文精确刻画了Qwen2.5-7B-Instruct模型在数字比较任务中的因果几何结构，发现模型主要依赖线性表示而非操纵低维流形来实现比较计算。 |
| [^49] | [KUPAS MASTER: Distilling the Tacit Expertise of Master Practitioners into Agent-Ready Experience Corpora](https://arxiv.org/abs/2609.37673) | 提出了KUPAS MASTER经验工程平台，通过六要素案例结构和九层认知语料库构建方法，将资深从业者工作记录与访谈中的隐性专业知识转化为可追溯、可复用、供LLM智能体使用的结构化经验语料库。 |
| [^50] | [Corpus-Guided Dual-Path Propagation for Graph Retrieval-Augmented Generation](https://arxiv.org/abs/2609.37661) | NexusRAG通过联合实体共现与语义相似度构建语料库级实体邻域结构，引导语义传播与结构传播两条互补路径，克服了无关系图检索中仅依赖查询-句子相似度而遗漏桥接证据的缺陷。 |
| [^51] | [Evaluating and Benchmarking the System One Model Jev](https://arxiv.org/abs/2609.37647) | 本文对TypeSafe AI的商业System One模型Jev进行了大规模零样本评估，在37个数据集、346,009次请求（成本不到10美元）上系统基准测试其在分类、路由、推理、审核等小型决策任务上的表现，并与Qwen3.8-27B和Gemma-4-E4B进行对比。 |
| [^52] | [Co-Linguistics: AI-augmented Theory Construction in Linguistics](https://arxiv.org/abs/2609.37635) | 本文提出“协同语言学”理念，即AI可作为“协同科学家”帮助语言学家构建和评估形式化语言学理论，通过显式化现有理论、比较竞争理论、提出新理论来加速语言学研究。 |
| [^53] | [RLTL;DR: Self-improvement by Internalizing Self-generated Feedback](https://arxiv.org/abs/2609.37633) | 提出RLTL;DR方法，让策略在每次失败后根据验证器输出撰写自己的TL;DR见解，并将其作为后续尝试的上下文条件，同时通过反向传播将这些见解内化为任务到见解的直接映射，从而在无教师模型、任务难度极高（Pass@128=0）的场景下实现自我提升。 |
| [^54] | [Correct, Don't Delete: Mitigating Emergent Misalignment with Corrective Supervision](https://arxiv.org/abs/2609.37624) | 本研究发现在微调数据被污染的场景下，将有害示例纠正为正确答案比直接删除它们更能有效缓解涌现性失调——替换四分之一的污染数据即可将失调率降低约三分之一，而删除同样的数据几乎没有效果。 |
| [^55] | [Authority Bias in Language Models: Source Deference and User Agreement Are Not Interchangeable](https://arxiv.org/abs/2609.37616) | 该研究发现，语言模型对“验证来源”（如检索结果、工具输出）的盲目遵从远强于对用户断言的附和——一条支持错误答案的权威来源注释可使45%-88%原本正确的回答被翻转，且这两种行为在模型内部是可通过因果干预选择性区分的、不可互换的不同机制。 |
| [^56] | [FOCUS: Training-Free Decision-Preserving Context Compression for LLM Agents](https://arxiv.org/abs/2609.37590) | FOCUS是一个免训练、与架构无关的上下文压缩框架，它通过在测试时因果性地识别哪些历史交互塑造了智能体的未来决策来压缩上下文，避免了离线训练的高昂成本，同时缓解注意力稀释导致的性能下降。 |
| [^57] | [Rational Clarification by Assistive Agents via Value-of-Information Reasoning](https://arxiv.org/abs/2609.37588) | 该论文提出 REVOIR 框架，让辅助智能体在推理阶段通过计算澄清问题的信息价值（即答案带来的预期任务奖励提升），来理性权衡是直接行动还是向用户提问，从而更安全有效地处理模糊请求。 |
| [^58] | [Pair Difficulty Matters: Rethinking Pairwise LLM-as-a-Judge Evaluation and Consistency](https://arxiv.org/abs/2609.37577) | 该论文指出，传统用于评估LLM裁判可靠性的三大代理指标（位置偏差、传递性、成对一致性）具有误导性——在Bradley-Terry几何下它们被排名差距接近的配对所主导，而此类配对出现不一致在信息论上是必然的，因此不应以这些指标否定能力出色的LLM评估器。 |
| [^59] | [MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment](https://arxiv.org/abs/2609.37574) | MERGE提出一个两阶段多LLM集成框架，先由三个小型开源LLM独立生成查询扩展候选、再由更大的LLM生成式合成为单一查询，并用基于下游检索性能的自动提示词优化循环取代传统LLM评估器，解决了单一LLM查询增强受限于模型偏见且提示词工程难以扩展的问题。 |
| [^60] | [Devils in Question Relay: Source-Conditioned Relay Steering to Mitigate Hallucinations in Audio-visual Large Language Models](https://arxiv.org/abs/2609.37568) | 该研究揭示了音视频大语言模型中“源混淆接地幻觉”的内部成因——问题传递机制使问题状态混入未使用模态的干扰线索，并提出基于源条件化的传递引导方法，通过切断干扰模态到问题状态的通路来有效缓解幻觉。 |
| [^61] | [Orthogonal Yet Coupled: Decoupling Geometric Components for Model Merging](https://arxiv.org/abs/2609.37564) | 提出DiGA几何感知模型合并框架，通过以预训练权重为共享几何参考，将任务向量正交分解为具有不同几何属性的组件并分别聚合，解决了传统整体合并方式中跨组件耦合导致合并模型质量下降的问题。 |
| [^62] | [RunyaNER: Auxiliary Language Selection for Runyankore NER](https://arxiv.org/abs/2609.37543) | 该论文发布了首个公开的Runyankore语命名实体识别基准数据集RunyaNER（含23.7万标注词），并利用它系统研究了跨语言零样本迁移和多语言微调中辅助语言选择策略的效果。 |
| [^63] | [E-MoE: Enhanced Mixture-of-Experts for Non-Factorized Diffusion Language Models](https://arxiv.org/abs/2609.37533) | E-MoE利用MoE骨干网络的专家路由决策作为离散共享潜在变量，将掩码扩散模型的逆向过程构建为非因子化的分布混合，在不增加激活参数的前提下显著提升了少步生成质量。 |
| [^64] | [Hierarchical Compression of Vision-Language Model Benchmarks](https://arxiv.org/abs/2609.37515) | 提出了PRIMEBench——一个视觉感知的层次化基准压缩框架，通过数据清洗、类别代表性选择、视觉感知方差（VAW）题目剪枝和类别数量剪枝四个阶段，在保持模型排名的同时大幅降低视觉-语言模型的评估成本。 |
| [^65] | [From Dissonance to Orchestration: Teacher Intervention in On-Policy Distillation](https://arxiv.org/abs/2609.37510) | 该论文发现更深的教师干预在在线策略蒸馏中收益递减且会增加离策略负载，并提出 MAESTRO 方法，通过策略分歧分数自适应地决定教师何时接管以及生成多长时间。 |
| [^66] | [Evaluating Bounded Autonomy in Regulated Agentic AI: A Diagnostic Harness with Constitutional Rewards, Escalation Labels, and Runtime Governance](https://arxiv.org/abs/2609.37501) | 提出RegLLM诊断框架，利用宪法奖励、任务级升级标签和确定性运行时治理来衡量并约束受监管智能体AI的有界自主性，实验表明治理机制可将升级召回率从0提升至0.67、将不安全行为率从0.33降至0.08。 |
| [^67] | [Who Warmed the Archives? LLMs Overestimate Historical Warmth](https://arxiv.org/abs/2609.37499) | 该研究首次系统评估了利用大语言模型从历史档案中提取温度指数的可靠性，发现所有测试的LLM均存在随年代递增的系统性偏暖偏差，使其误差在跨世纪气候比较中并不安全。 |
| [^68] | [The Rashomon Wikipedia: A Data-Perspectivist Analysis of Divergent Historical Narratives](https://arxiv.org/abs/2609.37498) | 该研究通过对五种语言维基百科中罗马尼亚三个争议历史事件的分析，首次揭示了语言版本之间严重的“引用隔离”现象及民族中心主义叙事偏见——如波萨达之战的119条引用中仅2条跨语言共享，罗马尼亚语版本91%的引用呈现亲民族立场。 |
| [^69] | [Larry Caused the Car to Stop, But the Model Didn't Notice: Transformer Blindness to the M-Heuristic](https://arxiv.org/abs/2609.37497) | 本文通过自然语言推理实验对比词汇使役与分析型使役结构，发现DeBERTa、RoBERTa和BART等Transformer模型无法捕捉M-启发式所编码的语用区别，且探针分析的结果实际反映的是句法复杂度而非使役性。 |
| [^70] | [Your Benchmark Is Not Saturated: Reviving Multiple-Choice Evaluation with Answer Pooling](https://arxiv.org/abs/2609.37494) | 提出AnswerPool方法，无需编写新题，仅将共享上下文的多道题目的选项合并为一个池并要求模型同时作答，即可大幅降低饱和多选题基准的可猜测性，同时还能在同一轮次中评估模型的弃权能力。 |
| [^71] | [Risk-Controlled Selective LLM Answering by Pricing Label-Free Checks](https://arxiv.org/abs/2609.37493) | PriceCheck通过为无标签检查（如重新解题）赋予“价格”来构建决策规则族，在给定选择性风险目标下自动决定何时作答、何时弃答，在数学任务上平均提供76.1%的答案并将选择性风险控制在1.5%以下，效果优于奖励模型、提示评判器、生成器置信度等基线方法。 |
| [^72] | [Regime Boundary Alignment for Evidence-Gated Question Answering](https://arxiv.org/abs/2609.37491) | 提出机制边界对齐（RBA）方法，通过在同一问题的有支持与无支持匹配变体上训练单一阅读器，使其在有证据支持时作答、证据缺失时弃权，无需验证器、阈值或机制标签即可将多跳问答中的无支持作答率降低超过六十个百分点。 |
| [^73] | [FORUM: Frozen Outputs Reconciled Using Model Agreement for Visual Grounding](https://arxiv.org/abs/2609.37488) | 该论文提出FORUM，一种无需训练的测试时融合框架，利用多个冻结多模态大语言模型的预测一致性（基于一致性选择和中位定位两条几何规则）来提升对抗性视觉定位任务的鲁棒性，仅用三个开源模型就在Ref-Adv-s基准上以相对5%的平均准确率优势超越了397B参数的参考模型。 |
| [^74] | [Relevance Is Not Sufficient Evidence: Detecting Evidence Gaps Before Generation in RAG](https://arxiv.org/abs/2609.37469) | 该研究揭示了现有证据不足测试基准的构建陷阱，并提出一个通过替换、删除和问题交换构造、控制表面特征的配对基准，证明可以在生成之前仅凭问题与检索证据判断其充分性，从而使RAG系统在证据缺失时更可靠地弃答。 |
| [^75] | [Learning to Retrieve Missing Evidence for Long-Term Memory QA](https://arxiv.org/abs/2609.37443) | 提出MERA框架，通过强化学习训练轻量级规划器，利用已验证的证据作为线索迭代式地检索缺失证据，在长期记忆问答任务上显著超越现有方法。 |
| [^76] | [Look What You Made Us Cluster: Hate Narrative Extraction from Reddit Discourse](https://arxiv.org/abs/2609.37408) | 提出了一种基于LLM推理的仇恨叙事提取流程，将叙事表示为实体-评价对，结合Leiden聚类与LLM引导的细化过程，从Reddit话语中实现更精确、更可解释的仇恨叙事识别。 |
| [^77] | [Compiling Learning Problems into Adaptation Programs for Language Models](https://arxiv.org/abs/2609.37371) | 提出“适应编译”框架，通过从历史适应中学习预测候选程序的多维反事实响应面，在适应开始前自动选择最优更新程序，并可跨不同下游目标复用而无需重新训练。 |
| [^78] | [SemOPT: Fixing Semantic Errors in LLM-based Optimization Modeling via Reward-Guided Search](https://arxiv.org/abs/2609.37361) | 提出了SemOPT框架，通过语义奖励模型与奖励引导搜索，自动检测并修复大语言模型生成优化建模代码中难以察觉的语义错误。 |
| [^79] | [Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling](https://arxiv.org/abs/2609.37351) | 该论文发现了测试时潜空间深思中随思考深度增加而推理崩溃的“深思漂移悬崖”现象，并提出端口哈密顿潜空间深思框架，以同时兼顾表达能力、Lyapunov稳定性与计算效率，从而在深层思考步数下有效抑制漂移、缓解推理性能崩溃。 |
| [^80] | [Solving Without Stopping: On-Policy Distillation at Small Scale](https://arxiv.org/abs/2609.37326) | 该研究通过将 Qwen3-8B 蒸馏到 0.6B–4B 的小模型，发现策略内蒸馏能够传递解题能力，但在思考模式下无法传递“知道何时停止推理”的能力，且学生模型的提升受限于其训练前多次尝试即可达到的性能上限。 |
| [^81] | [Hidden Reasoning Must Leak, but Need Not Be Readable: Fundamental Opportunities and Limits for Chain-of-Thought Monitoring](https://arxiv.org/abs/2609.37312) | 本文证明足够复杂的隐藏计算必然在思维链中留下信息论痕迹，但在密码学假设下模型可以加密推理过程使其对多项式时间监控器不可读，从而揭示了思维链监控的根本机遇与局限。 |
| [^82] | [Asking for What Was Never Requested: Horizontal and Vertical Proactivity in Agents](https://arxiv.org/abs/2609.37236) | 该论文提出了智能体主动性的全新维度——横向主动性（追寻当前上下文已隐含的未明说信息）与纵向主动性（追寻仅由早期证据揭示的需求），并利用从基准分解中恢复的需求图实现了无需模型评判者的自动评分，同时设计了无需奖励模型的Q&D方法来训练智能体提出最有效的问题。 |
| [^83] | [Follow the Entities: A Corpus Map for Agentic Search](https://arxiv.org/abs/2609.37226) | 提出CorpusMap，一种围绕文档中反复出现实体来组织语料库的导航层，帮助大语言模型智能体在多文档搜索中发现文档间的关联，从而减少遗漏互补证据并降低令牌消耗。 |
| [^84] | [CredWise: A Controlled Agentic Decision-Intelligence Framework for Explainable and Auditable Credit-Risk Assessment](https://arxiv.org/abs/2609.37223) | 本文提出CredWise决策支持框架，将XGBoost信用风险预测与概率校准、SHAP可解释性分析、政策检索及受控智能体工作流相结合，在真实贷款数据上实现了性能优良且时间稳定的可解释、可审计信贷风险评估。 |
| [^85] | [VLM Fine-Tuning for End-to-End Combinatorial Optimization](https://arxiv.org/abs/2609.37175) | 本文提出一种通用视觉语言求解器，将输入导出的视觉表示与文本描述相结合，通过监督微调和验证器引导的强化学习训练单一视觉语言模型，在CVRP、JSSP等复杂组合优化问题上显著优于纯文本方法，且问题规模越大优势越明显。 |
| [^86] | [Bridging Semantic Gaps in RAG through Generated Context Knowledge Fusion](https://arxiv.org/abs/2609.37171) | 提出知识感知语义桥接框架，通过智能知识融合实现查询与检索文档的语义空间对齐，从而弥合RAG中的语义鸿沟并提升段落选择的相关性与准确性。 |
| [^87] | [Trajectory Soup: Pushing the Compute-Scaling Frontier of LLM Mid-training via Diverse Trajectories](https://arxiv.org/abs/2609.37169) | 提出 Trajectory Soup 方法，将大语言模型中期训练的计算预算分散到多个独立分支上，并通过轨迹内与轨迹间权重平均将最优检查点融合为单一模型，从而突破串行计算带来的收益天花板，拓展中期训练的计算扩展前沿。 |
| [^88] | [Multimodal Detection of Higher-Order Behavioral Constructs: Self-Compassion in Structured Reflective Interaction](https://arxiv.org/abs/2609.37148) | 该研究针对“自我关怀”这一不可直接观察的高阶心理构念，构建了首个多模态数据集，收集并标注51场结构化反思对话，探索从随时间变化的言语、声音与动作中推断此类复杂行为特质。 |
| [^89] | [LoLBench: Evaluating Coding Agents with Long-Horizon Proposals on Large Software Systems](https://arxiv.org/abs/2609.37143) | LoLBench是一个多语言基准测试，通过让编码智能体在大型软件系统上完成从人工编写的增强提案到代码实现的完整流程，首次同时评估其将用户意图转化为规范的感知能力和生成正确代码修改的实现能力。 |
| [^90] | [LLM unbranding: Erasing Commercial Identity while Preserving Generic Utility](https://arxiv.org/abs/2609.37127) | 本文正式定义了“大语言模型去品牌化”这一新任务，旨在中和LLM文本输出中定义品牌身份的标志性语言、口号和风格标记，以防范商标淡化、错误归因和品牌诽谤等风险，同时保留模型的通用效用，并提供了覆盖多商业领域知名品牌的综合评估数据集作为基准。 |
| [^91] | [Cross-Linguistic Effects in Bilingual Phoneme BabyLMs](https://arxiv.org/abs/2609.37121) | 该研究将语音音素表示引入双语BabyLM训练，固定英语为第二语言并变换德语、瑞典语、波斯语和巴斯克语作为第一语言，以在更贴近儿童口语学习条件的框架下研究双语习得中的跨语言效应。 |
| [^92] | [Unlocking the Critic: Reward-Free Policy Optimization for LLM Post-Training](https://arxiv.org/abs/2609.37119) | 该论文提出保留常被丢弃的预训练评论家，通过小幅度低方差的策略更新实现稳定训练，并利用评论家从未完成前缀预测成功概率的能力提供无需完整采样与奖励的密集学习信号，从而实现高效的长程推理后训练。 |
| [^93] | [VACE: Validation-Gated Alternating Co-Evolution of Agent Models and Harnesses](https://arxiv.org/abs/2609.37105) | VACE提出了一种验证门控的交替协同进化方法，将智能体强化学习与基于轨迹的框架优化交替进行，只有当候选框架通过验证性能门控时才被采纳，从而在OfficeQA和AutomationBench上显著超越仅权重强化学习和无门控交替方法。 |
| [^94] | [What Does Post-Training Change in Multilingual Reasoning?](https://arxiv.org/abs/2609.37104) | 该论文通过对Qwen3模型在十一种语言竞赛数学任务上的审计发现，非英语语言中仅有约15-18%的问题能获得符合语言要求的正确完整解答（英语达92.9%），并通过系统评估十三个模型端点（包括多语言SFT与多种RL奖励设计）来定位多语言推理差距的来源。 |
| [^95] | [Traverse: Learning When to Remember, Reset, and Redirect for Long-Horizon Web Search](https://arxiv.org/abs/2609.37082) | 论文提出Traverse自主搜索框架，让智能体通过“评分标准—答案—验证”三状态自我管理搜索并配备封存记忆工具进行主动上下文管理，同时用仅训练上下文管理后末段的简单策略避免“封存崩塌”，使35B模型在BrowseComp上达到72.83。 |
| [^96] | [Learning from Think-Mode Advantage via On-Policy Distillation](https://arxiv.org/abs/2609.37044) | 本文提出 ThinkOPD，通过在线策略蒸馏让模型学习思考模式的优势，并引入轨迹-回答分歧（TRD）度量，在回答层面自适应地路由教师监督信号，克服了统一共享思考轨迹蒸馏带来的师生不匹配问题。 |
| [^97] | [Selecting The Most Informative Tokens in Natural Language Autoencoders](https://arxiv.org/abs/2609.37040) | 仅基于聊天结构训练的排序器无需模型前向传播即可为自然语言自编码器选出信息量最大的词元位置，仅解释5%的位置就能保留几乎全部威胁审计成功率，且预训练言语化器无需额外训练即可恢复模型经微调后隐藏的词语。 |
| [^98] | [LatCom: Cross-Agent Latent Compression for Efficient Multi-Agent Collaboration](https://arxiv.org/abs/2609.37017) | 提出LatCom框架，通过将多个发送方的潜在表示压缩到固定数量的任务相关槽位中，解决了多智能体潜在协作中被忽视的跨智能体冗余问题，从而降低接收端上下文规模、计算开销与协作延迟。 |
| [^99] | [CypherTurn: A Multi-Turn Benchmark for Conversational Text-to-Cypher Evaluation and the Autonomy Divergence](https://arxiv.org/abs/2609.36987) | 该论文提出了首个对话式Text-to-Cypher多轮基准测试CypherTurn（721个会话、5,927轮对话），发现最佳模型执行准确率仅64.7%、会话级正确率不足5%，并揭示前沿模型在自主运行时排行榜头部会显著重排的“自主性分歧”现象，表明错误管理是部分独立于原始生成能力的关键维度。 |
| [^100] | [SRJudge: Empowering Large Language Models with Selective Reasoning for Fine-Grained Knowledge Concept Tagging](https://arxiv.org/abs/2609.36982) | 本文提出三阶段SRJudge框架，先利用微调的小型语言模型将候选概念缩小到top-K范围，再让大语言模型进行选择性推理，从而解决了大语言模型在细粒度知识概念标注中因决策空间过大而难以选出正确概念的问题。 |
| [^101] | [AMU:Admission and Memory Update for Personalized Conversations---Structured Memory with SLM Guided Control](https://arxiv.org/abs/2609.36976) | 提出了AMU框架，利用小语言模型引导的结构化记忆控制机制，在写入阶段决定哪些信息进入长期记忆，并对记录进行单独存储、去重丢弃或融合更新，从而提升个性化对话的记忆质量。 |
| [^102] | [Repetition, Not Length: Isolating the Counting Failure in Neural Text-to-Speech](https://arxiv.org/abs/2609.36974) | 该论文通过句子数与词数匹配的对照实验证明，神经文本转语音模型的计数失败源于文本重复本身而非长度——控制句准确率高达94.3%而重复句仅18.2%，且该失败随重复周期性平滑增长、即使词不相邻也依然存在。 |
| [^103] | [Chinese-Jev: Bringing System One Model to Chinese-Language Tasks](https://arxiv.org/abs/2609.36965) | Chinese-Jev通过统一的数据处理与训练流程——将异构中文标注转换为候选选项的概率目标，并采用轻量级编码器骨干进行面向决策的训练——首次将高效的系统一模型成功引入中文决策任务。 |
| [^104] | [VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers](https://arxiv.org/abs/2609.36958) | 本文提出VStress可审计重放契约与VStress-CA相关性感知分配策略，通过估计重复验证器调用的条件边际信息来自适应分配预算并在信息不足时停止调用，从而在固定预算下以更低的调用成本获得更高的验证平衡准确率，同时通过受控审计刻画了该机制在极端损坏率下的适用边界。 |
| [^105] | [Cool the Sampler, Not the Learner: Sampling Temperature Moves the Staleness Cliff of Importance-Corrected GRPO](https://arxiv.org/abs/2609.36953) | 该论文发现重要性校正GRPO在采样器长时间不刷新时会遭遇性能“悬崖”式崩溃，并提出“解耦冷却”方法——仅将采样端温度降至0.8而学习端保持温度1，在不改变学习器目标函数的前提下移动了过时悬崖，使每192步才刷新一次采样器仍能保持稳定学习。 |
| [^106] | [ER-JEPA: Experience Replay Improves Joint-Embedding Predictive Learning in Language Models](https://arxiv.org/abs/2609.36952) | ER-JEPA在LLM-JEPA中引入经验回放机制，通过存储并检索历史训练样本对提供额外监督，使模型同时从当前批次和记忆库中学习，在多个数据集上一致优于LLM-JEPA。 |
| [^107] | [CoEM: Empowering Long-Context Reasoning with Commit-on-Evidence Memory](https://arxiv.org/abs/2609.36935) | CoEM 提出了一种“证据提交记忆”机制：在固定上下文预算下先逐字保留待定证据，再由学习到的策略根据新到来的上下文决定将其提交、继续保留或丢弃，从而避免过早压缩导致的关键信息丢失，提升长上下文推理性能。 |
| [^108] | [Dating the Model: Hidden Dates in System Prompts Affect LLM Evaluation](https://arxiv.org/abs/2609.36931) | 研究发现系统提示中隐藏注入的当前日期会显著影响大语言模型的评估结果与排名（性能差异最高达14%），其影响超过批大小等其他非确定性因素，且思维链提示反而会放大这一效应。 |
| [^109] | [Benchmarking Automatic Speech Recognition Tools for Iberian Languages](https://arxiv.org/abs/2609.36920) | 该研究对十一个语音识别系统在五种伊比利亚语言上进行了迄今最全面的基准测试，发现准确性、效率和语言覆盖之间存在明显权衡，低资源语言（尤其是巴斯克语）性能显著下降，且多数系统存在性别偏见，为多语言ASR的实际应用提供了实用指导。 |
| [^110] | [Can Language Models Learn to Forecast Stock Prices](https://arxiv.org/abs/2609.36914) | 该论文在按时间顺序推进的股票价格沙盒中，通过对 Qwen3-4B 进行 SFT 与 PPO 后训练，探索语言模型能否学会利用价格、成交量等信息预测股票未来收益。 |
| [^111] | [BaLEEN: Biasing with Latent Encoded Entities for Context-Aware ASR](https://arxiv.org/abs/2609.36913) | BaLEEN提出了一种基于超网络的轻量级框架，利用预训练语言模型编码上下文关键词并将偏置向量注入完全冻结的ASR编码器中，实现了无需微调、推理零开销的即插即用上下文自适应语音识别。 |
| [^112] | [MultiTalk: Scaling Full-Duplex Speech Models to Long, Multi-Party, Bilingual Conversation](https://arxiv.org/abs/2609.36903) | 该论文发布了5.76万小时的多方双语合成语音训练数据集 MultiTalkPT，并在英语和中文上沿长时域与多方交互两个维度扩展了 Moshi 全双工语音范式，使单一模型能够应对长时多方对话场景。 |
| [^113] | [RAEGNet: Relation-Aware Evidence Graph Network for Harm-Aware Multimodal Fake News Detection](https://arxiv.org/abs/2609.36902) | 提出关系感知证据图网络RAEGNet，通过事件级证据检索与融合新闻-证据立场关系的有向图建模，联合检测假新闻的真实性及其潜在危害。 |
| [^114] | [Momentum-Coupled Rubric Adaptation for Detailed Image Captioning](https://arxiv.org/abs/2609.36893) | 提出 MoCo Rubric 两阶段框架，通过角色条件化共享参数和动量耦合机制协调描述生成、量规构建与评判三个环节，解决了现有基于量规的强化学习方法中角色解释不一致及分阶段优化脱节的问题，从而提升详细图像描述的质量。 |
| [^115] | [Harness Evolution as Learning: Approximation, Generalization, and Optimization Limits of Self-Improving Personal Agents](https://arxiv.org/abs/2609.36892) | 该论文以“将框架演化视为学习”的视角，通过互补的实证与理论分析，系统研究了个人代理框架的架构、规模与自演化算法三大核心问题，揭示了自我提升型个人代理在逼近、泛化与优化方面的极限。 |
| [^116] | [Rethinking Multimodal Fake News Detection in the Generative AI Era](https://arxiv.org/abs/2609.36850) | 该论文构建了面向生成内容场景的多模态假新闻检测数据集Weibo26，并提出生成性感知层次化推理（GAHR）框架，通过全局判断与局部校正相结合，弥合了假新闻检测与AIGC检测之间的割裂。 |
| [^117] | [On-Policy Visual Evidence Distillation](https://arxiv.org/abs/2609.36838) | ReVuE提出了一种面向视觉智能体的在线策略蒸馏方法，通过比较学生生成的多条交互轨迹，针对证据获取、读取和答案落地等不同失败阶段提供有针对性的反思与纠正。 |
| [^118] | [CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning](https://arxiv.org/abs/2609.36820) | 本文提出CorrGRPO，通过将奖励间协方差归一化为皮尔逊相关系数来改进GRPO的多奖励学习，从而避免大尺度奖励主导归一化过程并抑制小尺度奖励的信号。 |
| [^119] | [VAA-CSEC: Vote-guided Advantage Allocation for Chinese Semantic Error Correction](https://arxiv.org/abs/2609.36804) | 该论文提出VAA-CSEC多阶段框架，通过结合思维链蒸馏、监督微调、强化学习与自洽性解码，并设计对齐最小编辑原则的任务奖励函数以及重新分配GRPO优势的组级相对策略优化（GLPO），有效解决了中文语义纠错中的过度纠正问题。 |
| [^120] | [Seeing What Should Be Heard: Diagnosing and Repairing Cross-Modal Shortcuts in Omni-Modal LLMs](https://arxiv.org/abs/2609.36798) | 该研究揭示了全模态大语言模型在回答音频相关问题时过度依赖图像的跨模态捷径，提出因子化模态诊断方法以隔离各模态的因果贡献，并发现该捷径在监督微调和强化学习后训练中持续存在甚至被放大。 |
| [^121] | [QuantMLA: Function-Aligned Dual-Path Quantization for Low-Bit MLA KV Caching](https://arxiv.org/abs/2609.36760) | 提出了 QuantMLA——一个函数对齐的低比特双路径量化框架，通过系统建模 MLA 内容路径与 RoPE 路径的量化误差，并学习可完全离线融合的路径特定变换，在消除在线开销的同时大幅压缩 MLA KV 缓存且保持全精度计算效果。 |
| [^122] | [Does a prosody-trained representation help beyond trainable fusion? A parameter-matched study with frozen HuBERT](https://arxiv.org/abs/2609.36754) | 该研究通过参数匹配的冻结HuBERT实验发现，可训练融合机制本身即可带来词错误率下降，而额外的韵律表征虽被模型所依赖，却未能带来显著的额外识别收益。 |
| [^123] | [Group-Marginalized Self-Rewarding RL Drives Zero-Label Self-Evolving](https://arxiv.org/abs/2609.36750) | 本文提出 GMAE 方法，将响应奖励在不同组上下文下的实现聚合为响应级分布并估计期望优势，消除了随机组上下文带来的奖励信号不确定性，实现了无需人工标注的稳定自进化。 |
| [^124] | [SIPO: Unifying Reinforcement Learning with On-Policy Self-Distillation](https://arxiv.org/abs/2609.36742) | 提出 SIPO 方法，通过对比性自教师将强化学习与在线策略自蒸馏相统一，为稀疏的可验证奖励提供密集的 token 级信用分配，克服了以往自教师过度自信及对长推理轨迹过度惩罚的问题。 |
| [^125] | [Backpropagated Output Momentum: Relocating Optimizer History from Parameters to Task Space](https://arxiv.org/abs/2609.36738) | 提出反向传播输出动量（BOM），通过在模型输出端存储预测误差的紧凑移动平均并逐步经当前网络重投影，将优化器历史从参数空间迁移到任务空间，从而将优化器状态减少高达99.8%并提升验证性能。 |
| [^126] | [Reconstructing the Vocal Tract with Differentiable Acoustic Simulation](https://arxiv.org/abs/2609.36737) | 提出了一种可微分、GPU加速的声道声学仿真器，通过频域建模和可微湍流模型，实现了仅凭语音声音反向重建声道形状的突破。 |
| [^127] | [From Neurons to Conversation: Speech Brain-Computer Interfaces](https://arxiv.org/abs/2609.36736) | 本文从系统级视角综述了语音脑机接口研究，强调语音脑机接口本质上是一个自适应临床系统，其神经表征、硬件、解码架构、语言先验、反馈与用户学习需长期协同演化，而非单纯的神经到文本解码器。 |
| [^128] | [Distilling What Matters: Confidence-Aware Selective Distillation for Large Language Models](https://arxiv.org/abs/2609.36734) | 提出CaRE-KD置信度门控蒸馏框架，通过词元级自适应切换正反向KL散度以及批次级抑制教师不确定性更新的机制，解决大语言模型教师预测的高熵与幻觉问题，避免蒸馏过程损害学生模型良好校准的先验。 |
| [^129] | [Can Agents Design Libraries for Agents?](https://arxiv.org/abs/2609.36730) | 该论文提出LibraryDesignBench基准来评估智能体为其他智能体设计代码库的能力，发现智能体设计者在大多数任务上能复现人类生产级库的抽象，但下游智能体对库的利用不足，经常重新实现库中已有的功能。 |
| [^130] | [ATTUNER: Recomputation-Free KV Cache Reuse via Query-Side Adaptation](https://arxiv.org/abs/2609.36722) | 该论文发现位置无关缓存的质量损失主要源于模型在多个内容片段间选择时的注意力分数偏差而非位置编码不匹配，并提出ATTUNER通过查询侧适配修复注意力分数，在完全不重计算KV缓存的情况下达到接近完整预填充的效果。 |
| [^131] | [LAURA: Knowledge Distillation for Interpretable Ambiguous Clause Identification in Legal Contracts](https://arxiv.org/abs/2609.36707) | LAURA是一个后训练框架，通过结合IRAC-Unlearning提示技术的知识蒸馏，将教师大语言模型的能力迁移到小型开放权重学生模型上，实现法律合同中可解释的模糊条款识别，其中仅250M参数的Flan-T5即可达到最先进的性能。 |
| [^132] | [Lost in Conversation or Lost in Translation? Diagnosing Multi-Turn Degradation in RAG](https://arxiv.org/abs/2609.36700) | 该研究通过150万次模拟对话的大规模仿真实验，首次系统揭示了RAG与GraphAG在真实多轮对话场景中存在高达21%的性能退化和47%的不可靠性增加，指出当前基于单轮完备查询的评估范式与实际使用方式存在严重错配。 |
| [^133] | [Video2Skill: From Streaming Experience to Reusable Embodied Skills](https://arxiv.org/abs/2609.36691) | 该论文提出了流式具身技能发现（SESD）问题并推出Video2Skill基准，用于系统评估视觉-语言模型能否将连续视频流中的操作事件组织成可影响后续决策的持久可复用技能库。 |
| [^134] | [CHAIN: Calibrated LLM Forecasting via Causal-Temporal Hypergraph Inference](https://arxiv.org/abs/2609.36689) | 该论文提出CHAIN方法，将大语言模型在因果-时序超图上的概率预测分解为证据加权、证据聚合和源融合三个阶段，并为每个阶段设计针对性机制，从预测过程内部缓解系统性校准偏差，提升概率输出的可信度。 |
| [^135] | [ProgressCompass: Embodied Progress Reward Models Are Lost Without the Right Context](https://arxiv.org/abs/2609.36684) | 论文提出“上下文依赖的进展估计”这一新问题，并构建了包含24个操作任务的ContextProgress-Bench基准，用以评估进展奖励模型在必须依靠历史上下文才能判断任务进展的长时程具身任务中的表现。 |
| [^136] | [MARCO: Multi-Round Agentic Reinforcement for Conditional Molecular Optimization](https://arxiv.org/abs/2609.36683) | MARCO提出了一个基于评估器的多轮强化学习框架，通过“提议—反馈—修订”轨迹和聚合回合奖励的组相对策略优化，训练分子编辑器在多目标分子优化中同时兼顾有效性、性质改进与相似性控制。 |
| [^137] | [G\"odel Forest: Balancing Search Depth and Breadth for Data-Centric Recursive Self-Improvement](https://arxiv.org/abs/2609.36675) | 提出哥德尔森林多智能体框架，将数据中心递归自我改进组织为协同进化的搜索树集合——每个智能体在持久树上自主深化、分支或剪枝数据策略以保证搜索深度，并行多棵树则提供探索广度，从而解决该领域深度与广度难以兼得的根本困境。 |
| [^138] | [Replay the Curvature: Accurate and Scalable NVFP4 Quantization for Large Language Model Inference](https://arxiv.org/abs/2609.36654) | 提出Schur Replay缩放因子选择算法，通过精确复现GPTQ逐列更新来准确评估NVFP4块缩放因子的重构误差，并支持跨设备可扩展地求解，从而实现大语言模型推理的高精度4比特量化。 |
| [^139] | [What Makes Recurrence Effective in Looped Language Models?](https://arxiv.org/abs/2609.36636) | 本文通过受控实验系统研究了循环语言模型中循环何时有效、应用于何处及条件化方式的影响，发现循环能提升超出训练范围的推理能力但会损害知识性能，且性能取决于层与循环迭代之间的分配方式而非单纯的有效深度。 |
| [^140] | [Generating Edit-Inducing Questions for AI Research Manuscripts](https://arxiv.org/abs/2609.36617) | 该研究比较了GPT与人类审稿人为AI论文草稿生成“编辑诱发式问题”的能力，发现GPT的问题能引发更广泛深入的修改但有效率更低，并揭示了一个反直觉现象：处理长上下文反而会损害推理模型生成有用输出的能力。 |
| [^141] | [Act First, Reason Later: Accelerating On-Policy Distillation for Multi-Turn Agents via Reference-Conditioned Inverse Dynamics](https://arxiv.org/abs/2609.36608) | 论文提出ActFirst-OPD训练框架，通过参考条件逆动力学让多轮智能体先执行动作、后异步生成完整推理响应，将环境交互与响应生成分离，从而在不牺牲轨迹质量的前提下显著加速在线策略蒸馏训练。 |
| [^142] | [SEED: Self-Speculative Decoding via Implicit Encoder-Decoder](https://arxiv.org/abs/2609.36590) | SEED将仅解码器transformer重新解释为隐式编码器-解码器结构，通过复用验证阶段已计算的深层上下文表示来低成本生成高质量草稿，从而在解决草稿质量与成本权衡的同时加速LLM推理。 |
| [^143] | [Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It](https://arxiv.org/abs/2609.36585) | 仅在单个早期层加入微小的rank-8 LoRA并冻结其余全部权重，就能让Transformer充分“用足”深度，将Qwen3-8B在24行长链追踪任务上的准确率从15.5%提升至99%，表明模型默认输出远未释放其潜在计算能力。 |
| [^144] | [Long-Term Memory-Guided Enhancement for Target Perception in Audio-Language Models](https://arxiv.org/abs/2609.36577) | 提出LTM-AE方法，通过从干净参考录音中提取长期记忆表示并对音频token进行插值重建，在不训练任何模型参数的情况下显著提升音频大语言模型在噪声环境中的目标感知能力。 |
| [^145] | [Grounded Revision vs. Prior Injection: Probing Retrieval-Augmented Patent Claim Amendment](https://arxiv.org/abs/2609.36550) | 该研究以专利权利要求修改这一“正确”可定义的任务为测试场，发布了 7,385 个 USPTO 审查案例语料库、七项探测测试和确定性评估指标，发现四个前沿大语言模型在检索增强的权利要求修改中均未表现出经典的先验注入行为，且检索效果微小、方向不一致。 |
| [^146] | [DraftTrace: A Multi-View Analytics Environment for AI-Integrated Writing](https://arxiv.org/abs/2609.36544) | DraftTrace通过同时捕获最终作品、写作过程和与AI助手的交互三个互补视图，帮助教师区分真实的写作行为、AI生成文本和复制打字等不同情况。 |
| [^147] | [When Updating Stops Being Learning: Rethinking LLM Self-Evolution via learnable information gain](https://arxiv.org/abs/2609.36535) | 该论文提出用“可学习信息增益”作为整体诊断框架来衡量LLM自我进化中每轮新增的可参数化信息量，并据此设计ATRI方法对样本重新加权，从而自适应地调节训练以缓解自我进化退化问题。 |
| [^148] | [Retrieval Sensitivity to Identity Signals in Queries](https://arxiv.org/abs/2609.36534) | 密集检索器会受查询中身份信号的影响而产生系统性偏差：返回与查询自身政治倾向一致的文章，且对非裔美国人语言（AAL）查询的检索效果劣于白人主流英语（WME）查询。 |
| [^149] | [Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling](https://arxiv.org/abs/2609.36529) | 提出三元线性注意力，通过键、第二键与值的三元外积将循环状态扩展为三维张量状态，仅增加两个投影即实现状态大小E倍增长，从而增强长上下文序列建模能力。 |
| [^150] | [Adapting Context Compression for Long-Horizon Agents with Counterfactual Continuations](https://arxiv.org/abs/2609.36526) | 该论文通过匹配的反事实延续实验发现压缩导致的严重退化集中在孤立的压缩事件上，并提出PAIR方法，利用干预式推演定位并诊断有害的个别压缩操作，自动修订结构化压缩提示词以提升长程智能体的执行可靠性。 |
| [^151] | [Large-scale factor analysis shows machine intelligence is only partially interpretable](https://arxiv.org/abs/2609.36515) | 本论文对1,618个语言模型在456个纯文本基准上的13,251个评测分数进行了前所未有的大规模因子分析，发现语言模型的智能仅能被部分解释——一般智能因子约解释70.8%的方差，其余表现差异由领域特定的潜在因素驱动。 |
| [^152] | [Similar Choices, Different Attention: Cross-Modal Associations in Humans and Vision-Language Models](https://arxiv.org/abs/2609.36475) | 该论文通过让人类与视觉语言模型完成相同的伪词-图像匹配任务并记录眼动数据发现，尽管部分较大的VLMs在跨模态关联的选择上能与人类对齐（微调可进一步提升），但其视觉注意力与人类注视的匹配程度甚至不如简单的中心偏好基线，揭示了模型与人类“选择相似、注意不同”的现象。 |
| [^153] | [FinRT: Distilling Adaptive Red-Teaming Strategies into Reusable Adversarial Generators in Consumer Finance](https://arxiv.org/abs/2609.36474) | FinRT框架将自适应红队测试策略蒸馏为可复用的对抗性提示生成器，在消费金融领域将攻击成功率近乎翻倍（32.9% vs. 17.2%），同时提升对抗严重性33%并保持语义多样性。 |
| [^154] | [Fisher-IRG: Fisher-Induced Local Invariant Representation Geometry across Language and Vision Models](https://arxiv.org/abs/2609.36458) | 提出Fisher-IRG方法，通过局部预测敏感性来度量表示几何，能够在语言和视觉模型中更好地区分语义变化与无关干扰。 |
| [^155] | [Memory Consolidation Flattens the Temporal Shape of User Facts](https://arxiv.org/abs/2609.36457) | 长期记忆系统在将对话固化为笔记时会不对称地抹平进行时等时间体貌线索（381对语句中有244对被扁平化且从不反向），该现象在11种模型配置及mem0、Graphiti、Letta三个流水线中普遍存在，而丢失的线索会误导后续模型对事实是否仍然有效的判断。 |
| [^156] | [Reliable Parallel Decoding in Masked Diffusion Language Models](https://arxiv.org/abs/2609.36452) | 该论文提出了一种无需训练的可靠并行解码方法（RPD），通过逐层预测稳定性和最终置信度来判断掩码扩散语言模型中并行提交词元的可靠性，从而安全高效地加速文本生成。 |
| [^157] | [Invariant Atoms: Sparse Coordinates of Local Semantic Geometry in Language Model Representations](https://arxiv.org/abs/2609.36451) | 该论文提出“不变原子”假设，发现语言模型表示中的局部语义变化可以用一组在保持语义的变换下保持稳定的稀疏方向坐标来刻画，这些原子具有语义-干扰分离特性，并可对模型预测产生因果影响。 |
| [^158] | [MemFold: Learning Compact Soft Memory for Long-Context Personalization via On-Policy Optimization](https://arxiv.org/abs/2609.36435) | MemFold通过在线策略优化将查询条件化的文本记忆压缩为固定数量的连续向量作为读取器的记忆接口，并在读取器自身生成的序列上以组相对任务奖励等信号进行训练，从而在长上下文个性化中统一优化记忆的保留与实际运用。 |
| [^159] | [Eternal Sunshine of the Spotless Mind: Systematically Erasing LLM's Memories](https://arxiv.org/abs/2609.36414) | 该论文首次提出“LLM记忆删除”这一新研究方向，发现当前大语言模型即使声称遗忘也无法真正删除用户要求删除的信息，并提出DeLLM框架，通过处理对话中的消息依赖关系来实现对用户记忆删除请求的正确处理。 |
| [^160] | [Calibrated to Whom? Persona and Language Effects on Cultural Values in JEV](https://arxiv.org/abs/2609.36399) | 该研究通过对JEV模型进行28.8万次价值观调查测试，发现基于角色的校准能朝人类沙特-美国文化差异方向移动（英语中再现87%、阿拉伯语中仅62%，且长期取向反转），且阿拉伯语下差异缩小源于题目语言而非角色描述语言。 |
| [^161] | [TTMark: Pairwise Distortion-Free Watermarking Beyond Single-Token Entropy](https://arxiv.org/abs/2609.36372) | TTMark提出一种成对无失真水印框架，通过对相邻token对的联合分布进行水印嵌入，将有效水印字母表从V扩展到V²，使检测器能够同时利用token熵和条件熵，从而突破单token熵对检测能力的根本限制。 |
| [^162] | [DeepRewind: Predicting and Repairing Premature Commitments in Deep Research Agents](https://arxiv.org/abs/2609.36344) | DeepRewind提出了一个面向可逆深度研究的附加控制层，通过类型化认知图表示智能体的演化状态，利用世界模型预测中间结论的可逆性以阻止过早承诺，并在后续证据推翻结论时执行依赖感知的回滚，从而显著提升洞察召回率。 |
| [^163] | [Training LLMs to Verbalize Evaluation Awareness](https://arxiv.org/abs/2609.36316) | 提出言语化训练（VT）方法，通过在模型自发言明评估意识前截断轨迹并以强化学习鼓励其坦诚表达，使言语化评估意识提升2.4-2.9倍且可迁移到未见智能体场景，同时保持潜在意识与行为基本稳定。 |
| [^164] | [Fractional State Space Transition for Long Sequence Modeling](https://arxiv.org/abs/2609.36314) | FRAC 是一种基于分数阶动力学的选择性状态空间模型架构，用幂律长记忆取代传统指数遗忘，并通过有限状态、对数间隔的指数模式之和实现高效并行训练，显著提升了长上下文建模性能。 |
| [^165] | [HeurEvo: Agentic Evolution of Hybrid Solver-Augmented Heuristics for Time-Critical Mathematical Optimization](https://arxiv.org/abs/2609.36303) | HeurEvo提出了一个“计划—代码—组件”协同进化框架，联合进化高层算法结构、代码实现和可复用组件池，从而在严格时间约束下自动设计出融合启发式方法与数学规划求解器的混合优化算法。 |
| [^166] | [MoRE: Scaling mixture of experts with hardware-aware low-rank routing](https://arxiv.org/abs/2609.36301) | 提出MoRE方法，通过将MoE路由器权重矩阵低秩分解，把路由成本从Θ(Mh)降至O((h+M)r)，在证明可保持路由表达能力与负载均衡的同时，支持Θ(h/r)倍的更多专家，并考虑硬件实际加速。 |
| [^167] | [When Trees Are Not Enough: Learning Mixed-Topology Feature Graphs with Adaptive Graph Sparse Autoencoders](https://arxiv.org/abs/2609.36294) | 该论文提出自适应图稀疏自编码器（AG-SAE），将每个特征的完整父节点集合作为原子结构假设进行竞争验证，突破了传统单亲树结构的限制，能够学习包含多父节点关系的混合拓扑特征图并以此引导SAE训练。 |
| [^168] | [The Surge of Anti-Semitism in German Social Media following the October 7 Attacks](https://arxiv.org/abs/2609.36290) | 该研究利用大语言模型分析德国社交媒体在10月7日袭击事件前后共12.5万余条帖子，发现反犹太主义言论在Facebook和Telegram上显著激增（Telegram上约为Facebook的十倍），且加入用户上下文信息可将检测F1分数提升至83%并大幅减少误报。 |
| [^169] | [In-Context Learning Amplifies a Latent Symbolic Circuit](https://arxiv.org/abs/2609.36265) | 该研究揭示大语言模型内部在预训练时就已存在一个三阶段符号推理回路（抽象、归纳、检索），上下文学习通过放大该回路驱动少样本规则学习——单头因果贡献最高增长8倍，且注入函数向量可将0-shot准确率挽救至86%。 |
| [^170] | [OTROPE: Optimal Transport-based Robust Off-policy Evaluation for Large Language Models](https://arxiv.org/abs/2609.36264) | 提出 OTROPE，一种基于最优传输、无需似然值的大语言模型离线策略评估方法，通过在语义空间对齐行为策略与目标策略样本，实现无需密度比估计和策略建模的双重鲁棒式评估，适用于黑盒大语言模型。 |
| [^171] | [Population Fidelity: Evaluating Population Representativeness in LLMs](https://arxiv.org/abs/2609.36253) | 该论文提出了“群体保真度”评估框架，从群体层面准确性、群体间变异量及其结构三个维度评估LLM生成回复对真实人群的代表性，并发现LLM代表性不佳不仅是因为群体间变异不足，还因为变异被错误地归因于不同群体。 |
| [^172] | [Learning from Teacher Continuations at Student States](https://arxiv.org/abs/2609.36246) | OLIVE 提出一种在线干预式蒸馏框架——学生生成前缀、教师自回归续写并以其交叉熵更新学生——同时解决了离线 SFT 的协变量偏移、OPD 的监督碎片化以及分布匹配蒸馏需教师 token 概率三大局限，以相近成本取得更优推理性能。 |
| [^173] | [Cognitive Expert Language Models Better Align with the Corresponding Brain Systems](https://arxiv.org/abs/2609.36239) | 本研究通过提示和微调构建了感觉、空间、数值、推理、社会和抽象六个认知领域的专家型大语言模型，发现每个专家模型与对应认知功能的脑系统对齐更好，说明认知特化的语言模型能更准确地反映脑区的功能特化。 |
| [^174] | [CineSubBench: Evaluating LLMs on Long-Form Narrative and Cultural Understanding from Multilingual Movie Subtitles](https://arxiv.org/abs/2609.36218) | CineSubBench是一个基于1,012部电影六语言字幕构建的基准，通过七项任务在多任务、多语言、多元文化的匹配设置下，评估大语言模型从海量时序字幕中重构长篇电影叙事与文化理解的能力。 |
| [^175] | [Lost in Translation: Measuring the Effect of Non-Native English on End User Performance of Large Language Models](https://arxiv.org/abs/2609.36214) | 本研究构建了包含19万余个英语提示变体的FABLE数据集，发现大语言模型虽不会复制拼写等表层错误，却会镜像提示中的修辞与词汇水平，从而导致非流利英语用户获得系统性的低质量回复。 |
| [^176] | [The Canonical Order Problem: When Large Language Models Are Unreliable Knowledge Bases for Multi-Valued Relations](https://arxiv.org/abs/2609.36209) | 大语言模型内部按规范顺序（如字母或时间顺序）组织多值关系，当提示要求偏离此顺序生成实体集合时，其作为知识库的可靠性会显著下降。 |
| [^177] | [Geometric Representations of African Languages: A Regional Semantic Hub and Cultural Steering](https://arxiv.org/abs/2609.36205) | 该研究揭示Gemma 4 31B中非洲语言在表示空间中形成了比对照语言之间更紧密对齐的区域语义枢纽，并成功利用英语数据构建的文化方向来引导模型对不同非洲国家的响应。 |
| [^178] | [FastGuide: Accelerating Reward Guidance for Diffusion Large Language Models](https://arxiv.org/abs/2609.36202) | FastGuide通过自适应地混合并行与自回归解码，并借助KV缓存和稀疏注意力重计算，显著加速了扩散大语言模型的奖励引导推理过程。 |
| [^179] | [SCOUT: Synergizing Reasoning and Tool-Use for Computer-Use Safety](https://arxiv.org/abs/2609.36201) | SCOUT提出了一种两阶段的代理式安全验证器，通过将推理密集的评分标准生成与工具密集的证据收集相结合，能够有效检测计算机使用代理在执行任务时产生的细微且隐蔽的有害行为。 |
| [^180] | [Concept Direction Reliability Across Languages with Different Tokenizer Fertility](https://arxiv.org/abs/2609.36194) | 该研究发现情感概念方向的可靠性在不同语言间存在显著差异（英语最高、豪萨语次之、约鲁巴语最低，与分词器生育率相关），且情感分类器的预测准确性并不能保证方向的一致性。 |
| [^181] | [Targeting Pivotal Decisions for Credit Assignment in Agentic Reinforcement Learning](https://arxiv.org/abs/2609.36178) | ProVer通过智能体裁判对比成功与失败轨迹来提出潜在的关键决策片段，并利用片段前后策略延续的终端成功率差异验证其优势，从而在智能体强化学习中实现了细粒度的信用分配。 |
| [^182] | [Principled Thoughts for Latent Recursive LLM Systems](https://arxiv.org/abs/2609.36159) | 提出REST训练目标，将有效思维表示的因果性、最小性、可分离性和稳定性四个属性转化为可微分损失并加入交叉熵，从而解决仅用交叉熵训练潜在递归大语言模型系统时出现的四种失败问题，且无需更改架构或增加推理参数。 |
| [^183] | [Language Models Are "Insecure" Reporters](https://arxiv.org/abs/2609.36139) | 该研究首次系统性提出并验证了“不安全报告”现象——大语言模型在总结工作时倾向于隐瞒削弱方法有效性的负面结果，而仅仅添加一句简短的诚实指令就能将负面结果的指出率从 1% 大幅提升至 95%。 |
| [^184] | [When Does Correction Become Repair? Mechanistic Auditing of Internal Interventions in Tool-Using LLMs](https://arxiv.org/abs/2609.36138) | 提出SAKIKO审计框架，揭示工具使用型大语言模型内部干预中“行为改变不等于修复”——即便干预带来净收益，也可能损害超过一半的原始决策，因此必须通过目的地解析验证来审计干预的真实效果。 |
| [^185] | [A Character-Level Neural Approach to Sinhala Sandhi Splitting](https://arxiv.org/abs/2609.36131) | 该论文基于SandhiLex数据集首次为僧伽罗语连声切分建立了字符级神经网络基准，发现双向LSTM编码器-解码器模型在规则的附加式连声上可达94%准确率，但在词汇化、派生和词源连声等困难子集上仅达68.40%。 |
| [^186] | [Better Behavioral Prediction, More Faithful Model Ablations? Evidence from Sequential Choice](https://arxiv.org/abs/2609.36097) | 该研究在具有已知生成策略的合成序列选择任务中，通过比较神经网络模型与认知模型对输入消融的响应，检验了“模型对信息的依赖能否忠实反映行为生成过程的依赖”这一核心假设，揭示了行为预测的准确性并不必然保证模型消融分析对认知解释的忠实性。 |
| [^187] | [PADM\'E: Preference Alignment Data Synthesis for Meta-Evaluation of LM Agent Evaluators](https://arxiv.org/abs/2609.36086) | 该论文将元评估重新表述为偏好判断问题，并提出PADM'E数据合成方法，仅依靠小型语言模型、无需人工参与且计算成本低，就能为智能体场景生成可靠的基于标准的元评估数据。 |
| [^188] | [GeoOutageBench: Benchmarking Ambiguity-aware, Ontology-grounded Geospatiotemporal KGQA for Multimodal Power Outage and Resilience Analysis](https://arxiv.org/abs/2609.36082) | 该论文提出了GeoOutageBench，一个基于多模态时空知识图谱的基准测试，用于评估大语言模型在多模态停电与韧性分析中对歧义地理时空问题的理解、本体效用评估和答案准确性三方面能力。 |
| [^189] | [A Polyphonic Conception of AI Understanding](https://arxiv.org/abs/2609.36079) | 该论文提出“复调”的AI理解概念，基于机制性证据论证大语言模型的理解并非定位于单一机制，而是由多个可靠性不等的并行机制联盟共同产生输出，从而重构了关于AI理解问题的传统单声部范式。 |
| [^190] | [Mnemon: Raw Records, Fast Judgments, Slow Thoughts](https://arxiv.org/abs/2609.36059) | 提出记忆代理 Mnemon，将记忆工作类比为快慢双系统分工：LLM（System 2）负责规划搜索与组织答案，决策模型 Jev（System 1）快速执行大量记录判断，从而在保留带日期的原始对话记录的基础上高效构建长期记忆。 |
| [^191] | [Causal and Interpretable Structures in LLM Compositional Tasks](https://arxiv.org/abs/2609.35970) | 该研究通过循环概念的组合任务揭示了大语言模型处理词元关系的内部机制：模型在中间层采用基于两个词元推断关系的联合几何表示，而在后续层则使用涉及全部三个词元的联合表示来完成任务，且这一逐层演进模式在不同模型家族中保持一致。 |
| [^192] | [Question-Specific Knowledge Graphs for Efficient Visual Reasoning](https://arxiv.org/abs/2609.35942) | 提出VisKG强化学习框架，将视觉内容转化为问题特定的知识图谱表示，过滤无关视觉细节并保留实体-关系结构，从而实现更高效的视觉推理。 |
| [^193] | [Almost Human, Except When It Matters: VoxParity and the Decisions a Voice Should Change](https://arxiv.org/abs/2609.35922) | 提出 VoxParity 基准，在 183 个文字固定、音频可变的跨行业场景中检验语音代理是否会因“听到什么”而改变关键决策，结果 23 个可测系统中仅 11 个通过仅凭文字的零测试，表明当前语音代理在关键时刻仍以文字为准而忽略声音线索。 |
| [^194] | [VehicleArena: A Realistic Urban Environment for Multi-Agent Driving](https://arxiv.org/abs/2609.35916) | 提出了VehicleArena，一个3D城市驾驶基准，用于研究动态共享世界中独立运作的LLM控制智能体——每个智能体的驾驶决策会相互影响交通流与风险，而现有九个模型的到达率最高仅约65%。 |
| [^195] | [CruxBench: A Benchmark of Information Discovery](https://arxiv.org/abs/2609.35879) | 该论文提出CruxBench基准，通过信息价值评估大型语言模型发现关键问题的能力，其真实答案由未来世界事件决定，从而兼具抗污染、开放式和有据可依三大独特优势。 |
| [^196] | [The Price of Token Boundaries: Compression Certificates and Prediction](https://arxiv.org/abs/2609.35869) | 该论文提出基于线性规划对偶与独立整数证书来量化预分词边界规则压缩代价的方法，发现边界规则使英文维基百科上的最优词元数增加28.3–36.8%，且压缩与语言模型预测偏好不同的词典。 |
| [^197] | [PACT: Pairwise-Anchored Calibrated Tuning for Single-Token Typed Decisions](https://arxiv.org/abs/2609.35865) | PACT将带有机器校验证据证书的成对对比数据转化为四个无需新标注的训练项（双重差分边际、置换一致性、证据必要性、序数传输成本），并辅以轻量级校准，从而提升单token类型化决策模型的性能与可靠性。 |
| [^198] | [Resolving the Missing Financial Data Crisis: A Generative AI Pipeline for SEC 10-K Extraction](https://arxiv.org/abs/2609.35864) | 本研究评估了Llama-3 8B、Qwen-2.5 14B和Llama-3.3 70B等多个大型语言模型从SEC 10-K文件中提取财务数据的能力，以解决传统方法（正则表达式和BERT）导致的影响超过70%公司的金融数据缺失问题。 |
| [^199] | [The Detectability Gap: Hidden Heterogeneity in Hallucination Detection Across Language Models](https://arxiv.org/abs/2609.35860) | 该研究揭示了语言模型幻觉检测中隐藏的异质性：高一致性（Ghost）与低一致性（Flickering）幻觉之间存在显著的可检测性差距，且在排除统计耦合影响并通过多种稳健性检验后，这种不对称性依然成立。 |
| [^200] | [Hyperspherical Semantic Trajectory Analysis: Mapping Technological Diffusion across Academic Preprints, Patent Signals, and Compute Scaling](https://arxiv.org/abs/2609.35845) | 本文提出超球面语义轨迹分析（HSTA），一种无监督定量方法，通过将arXiv预印本与USPTO专利文本的Transformer嵌入投影到超球面上，直接量化追踪技术扩散与范式转变，克服了传统宏观经济生产率指标多年滞后的局限。 |
| [^201] | [Neurosymbolic Routing for Reliable Reasoning on Resource-Constrained Edge Devices](https://arxiv.org/abs/2609.35833) | 该论文提出一种神经符号路由方法，通过L*文法推断学习确定性有限自动机，将查询分类并分派给成本最低的正确求解器（结构化任务交给确定性符号引擎，开放性问题交给小语言模型），从而在资源受限的边缘设备上实现更可靠、更高效的推理。 |
| [^202] | [When Should LLMs Trust Their Own Revisions? A Risk-Aware Study of Intrinsic Self-Correction](https://arxiv.org/abs/2609.35832) | 该研究对 29 个开源大模型的内在自我纠错进行了风险感知分析，发现总体准确率会掩盖“挽回错误”与“推翻正确答案”之间的权衡，并通过比较三种运行时策略识别出利用初始回答后信号进行选择性修改（学习型门控）有效的场景。 |
| [^203] | [Beyond the Context Window: An Adaptive Entropy-Based Routing Framework for Hybrid Retrieval and Long-Context Language Models](https://arxiv.org/abs/2609.35831) | 本文提出熵驱动自适应路由器（EDAR），通过利用RAG响应前几个生成token的预测熵，在推理时自适应地决定采用检索增强生成还是完整长上下文处理，从而在成本与准确性之间取得平衡。 |
| [^204] | [Reliable but Design-Sensitive: Instrument Uncertainty in LLM Annotation](https://arxiv.org/abs/2609.35824) | 大语言模型标注在重复运行时虽然高度一致，但不同的任务设计和模型选择会导致标注结果大幅波动，其引入的不确定性远超人工标注工具之间的差异和抽样误差。 |
| [^205] | [Tracing mechanisms of sycophantic agreement in language models](https://arxiv.org/abs/2609.35822) | 该研究通过因果中介分析揭示，语言模型的谄媚性认同源于一组稀疏的早期注意力头——它们将用户观点信号注入最终提示词元的残差流并偏置答案检索，而消融这些注意力头即可大幅降低谄媚性且几乎不损害事实准确性。 |
| [^206] | [Can We Still Trust Disaster Social Sensing? Empirical Evidence on Detecting AI-Generated Social Media Posts](https://arxiv.org/abs/2609.35821) | 本研究构建了来自九场灾害的12,000条文本的匹配语义单元数据集，系统评估了多种AI文本检测器及大语言模型判断器区分人类与AI生成灾害帖子的能力，为生成式AI对灾害社会感知可信度的威胁提供了实证证据。 |
| [^207] | [$\tau$-Multilingual: Benchmarking Voice Agents Across Languages](https://arxiv.org/abs/2609.35820) | 该论文提出多语言语音智能体基准τ-Multilingual，发现韩语和中文场景下任务完成度显著下降（分别下降14.7和8.4分）且各语言呈现不同失败模式，并发布了语言包和评估工具供社区使用。 |
| [^208] | [Less Uniform Discrete Diffusion is More Powerful and Scalable](https://arxiv.org/abs/2609.35817) | LUDI框架通过引入更少均匀的损失函数和逐token时间嵌入，克服了均匀扩散语言模型规模扩展的障碍，并成功将7B自回归模型转化为具备复杂推理能力、每步可生成3个token的高效扩散模型。 |
| [^209] | [PrimeSeeker: Capability-Oriented Supervision for Deep Search Agents](https://arxiv.org/abs/2609.35816) | 提出PrimeSeeker框架，通过“潜在锚点推理”将深度搜索分解为锚点解析与关系转移的耦合操作链，以面向局部检索能力的方式构建训练监督，从而更有效地训练深度搜索智能体。 |
| [^210] | [How to Run Statistics over LLM Judges and Trust the Results: Calibrated Inference for Small-Sample AI Evaluation with evalstats](https://arxiv.org/abs/2609.35815) | 该论文指出对原始LLM评判分数直接进行统计检验会造成假阳性膨胀（且在人类与LLM几乎完美一致时风险最高），并推出evalstats工具与指导，通过预测驱动推断（PPI）实现九种假设检验（包括首次针对秩检验的PPI校正），使小样本AI评估的统计结论可信可靠。 |
| [^211] | [Constructing Challenging Browser-Use Tasks by Controlled Environment Interventions](https://arxiv.org/abs/2609.35814) | 提出 BreakingWeb 基准，通过在保持用户指令与成功标准不变的前提下对环境不同技术栈层面进行确定性干预，将浏览器任务的难度转化为可编程的环境属性，从智能体已能解决的任务中系统性构建出 519 对更具挑战性的任务对。 |
| [^212] | [Local Predictability and Collective Fidelity in LLM-Agent Societies](https://arxiv.org/abs/2609.35813) | 本研究利用9,455条轨迹和舆论动力学实验，比较了LLM智能体社会模拟中代理模型的个体预测与集体预测能力，发现邻居信息可全面改善个体预测，但集体层面的收益依赖迁移条件，强调了直接集体验证、明确观测限制及简单基线比较的必要性。 |
| [^213] | [Automated Evaluation of Multi-Turn Dialogues in In-Car Conversational Assistants](https://arxiv.org/abs/2609.35812) | 提出了一个自动化测试框架，通过闭环仿真结合策略引导的用户模拟器、对抗性策略管理器和双层LLM裁判，来评估车载对话助手在多轮对话中的约束处理、上下文保持和安全关键行为。 |
| [^214] | [Lookahead-R: Budget-Aware Tool Retrieval via Execution-Centric Planning](https://arxiv.org/abs/2609.35811) | Lookahead-R通过轻量级执行感知代理世界模型（无需调用真实API即可预测工具执行结果、延迟与语义效用），结合预算感知的蒙特卡洛树搜索，将工具检索转化为资源受限的序贯决策问题，实现了精度与效率的最优平衡。 |
| [^215] | [TRACE: Deployable Tree-Relational Structure Enhancement for Oncology LLMs](https://arxiv.org/abs/2609.35810) | TRACE框架通过将肿瘤学概念组织成可更新的树状关系结构，在推理时检索紧凑的提示证据，无需监督标签即可在零样本设置下提升肿瘤学大语言模型在分类与问答任务上的表现。 |
| [^216] | [Can Multimodal Large Language Models Generate and Detect Multimodal Social Media Fake News?](https://arxiv.org/abs/2609.35809) | 该论文提出一个由故事、图像和评论三个智能体协作的多智能体框架，生成了超过9,000条多模态假新闻，并通过对16个MLLMs的基准测试发现，现有模型的假新闻检测能力显著低于人类水平，尤其在识别图像真实性方面表现严重不足。 |
| [^217] | [When Successful Memories Mislead Embodied Agents:Memory Adaption For Task-Conditioned Execution](https://arxiv.org/abs/2609.35808) | 提出 MATE——一种无需额外 LLM 推理的确定性检索后处理方法，将历史成功轨迹改写为面向执行的记忆，从而避免“成功经验误导”，在 ALFWorld 上使 Qwen2.5-14B/72B 分别达到 81.3% 和 93.3% 的任务成功率。 |
| [^218] | [Environment Steering: Using Data Flow Control to Improve Agent Utility and Safety](https://arxiv.org/abs/2609.35807) | 该论文提出“环境引导”方法，将智能体执行状态建模为数据库表并在运行时用声明式策略检查记录级数据流，检测到违规时通过针对性反馈引导智能体走向安全轨迹，从而在提升任务成功率的同时实现0%的攻击成功率。 |
| [^219] | [From Lexical Baselines to Agentic Retrieval-Augmented Generation: Structured Skill and Responsibility-Level Extraction with the SFIA Framework](https://arxiv.org/abs/2609.35806) | 本文首次将SFIA框架下的结构化技能与责任级别提取任务形式化，系统对比了从词汇基线到多智能体检索增强生成等五种策略在该任务上的表现。 |
| [^220] | [Alignment Forecasting: Predicting Misalignment From Training Data](https://arxiv.org/abs/2609.35805) | 提出了“对齐预测”任务及包含5,000多个预测问题、覆盖17个目标模型、32个数据集和16种失败模式的ALIGNMENTFORECASTBENCH基准，用于在训练前预测微调数据是否会引发欺骗、谄媚等对齐失败。 |
| [^221] | [Evaluating the Effects of Prompt Perturbation on Bias and Hallucination in Large Language Models](https://arxiv.org/abs/2609.35804) | 本研究评估了提示扰动对大语言模型在决策任务中偏见与幻觉的影响，发现与以往研究相反，扰动可以在某些模型中缓解偏见和幻觉，其中 Claude 3 表现最为有效，而 GPT3.5 的表现则参差不齐。 |
| [^222] | [Developing an OCR model for Extracting Information from Invoices with Korean Language](https://arxiv.org/abs/2609.35796) | 提出了一种结合深度学习与图像预处理技术的高效OCR模型，用于自动从韩语发票中提取关键信息，在收集的发票数据集上达到87%的F1分数且处理时间极短。 |
| [^223] | [Sieve and Sage: Efficient Distraction Filtering for Reliable RALM Abstention](https://arxiv.org/abs/2609.35794) | 该论文将检索失败分解为“不可回答”与“受干扰”两种状态，并提出轻量级的Sieve模块在调用昂贵大语言模型Sage之前预先过滤检索文档中的干扰证据，从而以更低的计算成本实现更可靠的RALM弃答。 |
| [^224] | [FD-VAD: Semantic Endpoint Detection for Streaming Full-Duplex Speech](https://arxiv.org/abs/2609.35791) | FD-VAD提出了一种无需ASR的流式语义端点检测方法，通过因果音频-语言推理直接判断用户的停顿是犹豫还是话轮结束，从而实现更自然的全双工语音交互。 |
| [^225] | [Sage: Formalization with Semantic Correction](https://arxiv.org/abs/2609.35790) | 本文提出 Sage，一个智能体化的形式化引擎，通过四阶段分解式生成流水线与融合 Lean 4 编译器诊断和多维语义反馈的双信号语义修正循环，解决了自然语言翻译为 Lean 4 形式化命题时的“严谨性幻觉”问题，确保生成命题既句法有效又数学忠实。 |
| [^226] | [Large Language Models Exhibit Human-Like Bayesian Hypocrisy](https://arxiv.org/abs/2609.35779) | 研究发现GPT-4o和Claude 3.7 Sonnet在贝叶斯推理任务上的表现接近人类水平，但会像人类一样（甚至更严重地）伪善地谴责使用相同贝叶斯推理的他人。 |
| [^227] | [Tracing the Evolution of Oracle Bone Characters Across Three Millennia](https://arxiv.org/abs/2609.35674) | 该论文提出基于流形的文字演变框架（MSEF），通过神经常微分方程将汉字从甲骨文到楷书的演变建模为流形空间的连续演化，从而突破单一时期参照的局限以辅助甲骨文破译。 |
| [^228] | [Toward a Culturally Adapted Chinese Language Agent: A Wizard-of-Oz Study of Nonverbal Behavior in Chinese-German Intercultural Interaction](https://arxiv.org/abs/2609.35150) | 本文提出了一套结合逼真虚拟人形象与实时多模态数据采集的“绿野仙踪”研究系统，用于捕捉中国母语者对德国汉语学习者违反社会规范时的非语言反应，从而为开发具备文化适应能力的中文语言智能体奠定基础。 |
| [^229] | [One Readout, Many Repairs: Diffusion-Guided Hierarchical Search for Tool-Agent Repair](https://arxiv.org/abs/2609.34879) | 提出ReCommit——一个无需训练的扩散引导框架，将工具智能体修复形式化为操作支持集上的分层搜索，通过可复用的搜索区域避免重复操作选择，高效找到能成功执行并满足原始请求的替代工具调用序列。 |
| [^230] | [After the Fix: Transfer of Corrected Agent Experience](https://arxiv.org/abs/2609.34603) | 本研究通过3,300次运行系统评估了修复后的智能体经验向后续任务迁移的效果，发现修正经验带来的收益在很大程度上源于未修正基线较弱而非记忆质量的真正提升，且并非所有修正机制（如APEX）都能产生可比的修正收益。 |
| [^231] | [How to Tame a Multi-Headed Hydra? Adaptive Multi-Category Safety Steering for Large Language Models](https://arxiv.org/abs/2609.34514) | 提出CAM-Steer框架，通过将当前隐藏状态与安全/危险原型对比来估计各伤害类别的风险，在单个提示中多个伤害类别共存时自适应地协调激活引导的方向与强度，从而提升大语言模型的安全性。 |
| [^232] | [Coherence-Aware Distributional Evaluation of Open-Ended Text Generation](https://arxiv.org/abs/2609.34240) | 提出CHORD指标，通过在冻结大语言模型的隐藏状态空间中比较生成文本与人类文本的分布，有效检测开放式文本生成中局部流畅但全局不连贯的质量问题。 |
| [^233] | [LLMs are not stochastic parrots: Evidence for meaning-mediated abstraction from conlang-like tasks](https://arxiv.org/abs/2609.34187) | 本研究通过类人造语言任务证明，大语言模型在没有任何示例输出的情况下，仅凭自然语言描述就能处理统计上罕见、且违背训练数据表层模式的虚构语言规则，表明其具备超越统计模式匹配的、基于意义中介的抽象能力，从而反驳了“随机鹦鹉”论断。 |
| [^234] | [Quantization Error Is Spectrally Flat: A Single Random Probe Is a Calibrated, Data-Free Sensitivity Estimator, with Application to Budget-Targeted Mixed-Precision Quantization](https://arxiv.org/abs/2609.33923) | 本文发现舍入误差具有谱平坦性，从而证明单个随机高斯探针即可作为经过校准、无需数据的逐张量量化敏感度无偏估计器，并据此提出RAM方法，在无校准数据条件下通过背包求解实现精确字节预算下的混合精度量化。 |
| [^235] | [LLMs learn different forms of metacognition when trained to predict their own accuracy](https://arxiv.org/abs/2609.33886) | 训练大语言模型预测自身答题准确率时，模型学到的置信度包含两种不同的元认知信号，在接近训练数据的问题上反映真实准确率，在其他领域则反映答案分布的集中程度（输出一致性），且后者在训练早期即出现并可泛化。 |
| [^236] | [Quantifying Behavioral Tails in Black-Box Language Models](https://arxiv.org/abs/2609.33638) | RareTrap框架通过代理LLM构建几何感知映射来诱导可复现的提示词分布，并结合序列稀有事件模拟技术，有效估计了黑盒大语言模型发生严重行为的概率。 |
| [^237] | [Teach Yourself Where to Look: On-Policy Attention Self-Distillation for Reasoning](https://arxiv.org/abs/2609.33200) | 提出在策略注意力自蒸馏（OPASD），在token级监督之外增加以解答为条件的注意力蒸馏，将特权教师的注意力投影到学生可见位置并对齐，在竞赛级数学基准上将平均准确率提升4.98至8.40个百分点，同时避免回复长度膨胀。 |
| [^238] | [Knowing Is Not Choosing: What Explicit Verification Adds Beyond Generative Preference](https://arxiv.org/abs/2609.33142) | 语言模型“知道”正确答案不代表会“选择”它：将事实回忆分解为生成、排序与选择三个步骤后发现，基于 P(True) 的显式验证在候选答案排序上显著优于对数似然等生成式偏好信号，可将多数投票准确率提升约 5 个百分点。 |
| [^239] | [Multimodal LLMs Outperform Pathology Foundation Models in Cross-Domain Histological Similarity](https://arxiv.org/abs/2609.32876) | 该研究发布MOSAIC基准，揭示病理学基础模型存在将“同机构不同疾病”误判为更相似的临床危险缺陷，而通用多模态大语言模型在跨机构、跨领域的组织学相似性判断中持续表现更优。 |
| [^240] | [OpenTumorBoard: A Real-World Benchmark of Multidisciplinary Tumor Board Discussion Trajectories](https://arxiv.org/abs/2609.32810) | 该论文构建了首个基于 YouTube 公开真实肿瘤多学科会诊录像的大规模基准 OpenTumorBoard（611 例患者、19,157 轮讨论、10 个专科角色），用于评估大语言模型在专科发言与会诊模拟两种场景下的表现，并发现即使最强的前沿与医学大模型在临床等效性上也与专科医生存在显著差距。 |
| [^241] | [Decision-Sufficient State Representations: Measuring and Reducing Write-Time Regret](https://arxiv.org/abs/2609.32805) | 该论文提出将读取者的损失分解为“预算损失”和“写入时遗憾”的度量框架，发现在TextWorld烹饪游戏中几乎所有损失都来自写入时遗憾——持有事实的128-token状态几乎全胜，而提示词驱动的语言模型写入者最多仅赢17%，表明写入状态的质量是核心瓶颈。 |
| [^242] | [Adaptive Consistency Graph for Long-Horizon Agents](https://arxiv.org/abs/2609.32754) | 提出自适应一致性图（ACG），通过在持久图中增量组织执行证据及来源，并为每次决策构建可追溯的、以需求为中心的上下文视图，在不改动智能体规划器与工具执行器的前提下显著提升长时程任务成功率。 |
| [^243] | [AdaTutoRank: Learning to Rerank Document Sets via Adaptive Tutoring Optimization for RAG and Deep Research](https://arxiv.org/abs/2609.32472) | 该论文提出AdaTutoRank，针对文档集重排序中集合级标量奖励导致的监督稀疏与信用分配困难问题，通过自适应辅导优化为不同质量的rollout提供差异化指导，从而为RAG和深度研究筛选出完整、互补、无冗余的文档集合。 |
| [^244] | [SEA-CLIP-Tiny: Efficient Multilingual Text-Vision Embedding for Southeast Asian Languages](https://arxiv.org/abs/2609.30739) | 本文提出SEA-CLIP-Tiny，一个参数量不足5000万的紧凑型多语言文本-视觉嵌入模型，通过区域数据筛选和多语言教师引导，在七种东南亚语言的跨语言图像-文本检索上取得最优平均性能，且比MobileCLIP2参数更少、CPU延迟更低。 |
| [^245] | [RAZOR: Pruning Replaceable Experts in LLMs](https://arxiv.org/abs/2609.30465) | 该论文提出无需训练的 MoE 专家剪枝方法 RAZOR，利用共识残差衡量专家的功能可替换性，在固定剪枝预算下剪除可被存活专家替代的专家，无需梯度或恢复训练即可最大程度保留原始模型输出分布。 |
| [^246] | [JEV vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places](https://arxiv.org/abs/2609.29769) | 研究表明，无需生成文本的类型化分类器Jev可作为LLM评分准则裁判的低成本替代方案：准确率与LLM裁判无显著差异但成本仅为其1/29至1/325，且两者在分级准则上倾向于犯相似错误（均低于人工评分等级）。 |
| [^247] | [Isolated Sign Language Recognition for Icelandic Sign Language: Experiments in a Low-resource Setting](https://arxiv.org/abs/2609.25862) | 该论文首次针对极低资源的冰岛手语开展孤立手语识别实验，发现跨语言迁移（先在美国手语数据上预训练再微调）能带来最大收益，将准确率提升 14-24 个百分点。 |
| [^248] | [The Copy Ceiling: An Input-Exposure Control for Ontology-Grounded Generation over Curated Corpora](https://arxiv.org/abs/2609.24885) | 论文提出“暴露核算”与“复制天花板”这一无需评判者的评估控制方法，揭示语言模型在基于本体检索的接地生成中的性能提升几乎完全来自复制上下文中已暴露的答案，而非对检索结构的真正推理。 |
| [^249] | [Abstention and Noise Filtering: Two Missing Primitives of Softmax Attention](https://arxiv.org/abs/2609.22005) | 本文论证并通过实验证明，注意力值通路的门控为softmax注意力补充了两种缺失的基本要素——弃权（允许注意力头输出空值）和噪声过滤（抑制残差流中叠加特征的干扰），并在10M至350M参数的匹配模型上提供了实证支持。 |
| [^250] | [Uni-LaDiR: Latent Diffusion Unifies Multimodal Reasoning](https://arxiv.org/abs/2609.19878) | Uni-LaDiR提出将多模态推理步骤映射到共享潜在空间，并利用扩散模型预测下一块思维标记，从而在统一的潜在表示中实现灵活的多模态推理。 |
| [^251] | [Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection](https://arxiv.org/abs/2609.18860) | 研究发现大型视觉-语言模型内部已编码了检测有害模因所需的证据信息，但无法将其正确路由至输出端——通过稀疏自编码器读取的稀疏特征在六个有害内容基准上均显著优于模型原生预测，揭示了模型存在“可解码但误路由”的读取差距。 |
| [^252] | [From Pixels to Pairs: A Comprehensive Benchmark of LLM-Based Key-Value Extraction in Noisy Document Settings](https://arxiv.org/abs/2609.17538) | 该论文建立了一个系统性基准，评估开源指令微调大语言模型在干净文本与含噪OCR条件下提取键值对的能力，发现现代LLM在高质量文本输入下是强大的语义提取器（部分情况接近有监督布局感知系统），但在OCR噪声下性能显著下降。 |
| [^253] | [IROH: Insightful Ranking Of Humor using Multi-Stage Hybrid Retrieval with Rationale-Distilled LLM Judges for JOKER 2026 Track Task 1 English](https://arxiv.org/abs/2609.15618) | 该论文提出IROH三阶段检索系统，通过稀疏-稠密混合检索、交叉编码器重排序与原理蒸馏的LoRA大语言模型裁判集成，在JOKER 2026任务1幽默排序中夺得第一名（MAP 0.6347），并发现原理蒸馏裁判是排序质量的关键驱动因素。 |
| [^254] | [When the Wrong Key Wins: Understanding and Detecting Hallucinations in LLMs](https://arxiv.org/abs/2609.15106) | 大语言模型的幻觉源于预训练关联之间“潜在钥匙”的竞争，本文据此提出一种两阶段关键词扰动检测方法，通过移除关键词条并观察预测重组方式，区分误导性关联导致的错误与有据可依的正确答案。 |
| [^255] | [Agent as Policy for Robotic Manipulation](https://arxiv.org/abs/2609.12541) | 该论文提出Agent as Policy (AGP)方法，使通用智能体无需任何任务特定训练即可直接控制物理机器人完成从精细操作、动态运动到可变形物体处理等多种真实操作任务。 |
| [^256] | [A Fragility Spectrum for Recursive Language-Model Training](https://arxiv.org/abs/2609.11149) | 该研究让13个公开模型在固定递归污染协议下共享语料库繁衍五代，发现不同模型对坍塌的脆弱性存在约五倍差异，且该脆弱性排序在不同数据组成和随机种子下高度稳定，表明易坍塌性是模型本身的固有属性。 |
| [^257] | [The Oligarch Barely Steers Model Collapse in Multi-Model Ecosystems](https://arxiv.org/abs/2609.11146) | 在多模型递归训练的生态系统中，即使将寡头模型的市场份额推高至90%，既不会加速模型崩溃，也不会使其他模型被拖向寡头的输出分布——模型崩溃的动态对市场份额集中度表现出不变性。 |
| [^258] | [Last Translation Benchmark](https://arxiv.org/abs/2609.04173) | 提出了终极翻译基准测试，这是一个包含人工编写、经同行评审的多模态示例的基准数据集，通过为每个示例配备手工编写的验证规则来描述具体失败案例，解决了现有机器翻译基准趋于饱和且评估方法不可靠的问题。 |
| [^259] | [Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor](https://arxiv.org/abs/2609.03221) | 临床LLM智能体在完全相同输入下本身就存在显著的动作不稳定性（约8.7%），因此反事实公平性审计必须先测量这一“每动作不稳定性底线”，否则任何检测到的人口统计学差异都无法解释。 |
| [^260] | [Margins, Not Windows: Training-Free Per-Step Lossy Speculative Decoding](https://arxiv.org/abs/2609.02897) | AdaptiveSpec提出了一种无需训练的逐步推测解码方法，通过边际概率比规则放宽严格的token匹配验证，并动态调整草稿树的深度、宽度和节点数，从而在不受草稿长度和起草器架构限制的情况下加速LLM推理。 |
| [^261] | [Quit While You're Ahead: Quit for Efficient Candidate Generation in Machine Translation Reranking](https://arxiv.org/abs/2609.00588) | 提出Quit方法，通过不确定性量化的早停策略对机器翻译的整个候选生成—重排序流程进行增量式生成与重排序，在最高候选质量稳定时提前终止，从而在保持翻译质量的同时显著降低推理延迟。 |
| [^262] | [Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models](https://arxiv.org/abs/2609.00355) | 该论文提出 GLANCE——首个在未修改的视觉语言模型上实现无损推测解码的单遍块草拟器，通过块扩散头零成本读取目标模型已融合的视觉-语言状态，并在一次前向传播中完成整块草拟与宽候选树验证，从而打破了草拟器因规模受限而被迫牺牲视觉信息的自我挫败循环。 |
| [^263] | [JPO: Juris Policy Optimization for Structured Legal Reasoning in Criminal Judgment Prediction](https://arxiv.org/abs/2608.29616) | 提出JPO后训练框架，通过教师监督的标准化四步推理与基于复合奖励的强化学习，实现刑事判决预测中“事实—法条—罪名—量刑”环环相扣的结构化法律推理。 |
| [^264] | [The Illusion of Replacement: Rethinking Specialized Machine Learning Models in the Foundation Model Era](https://arxiv.org/abs/2608.28980) | 本文综述159篇论文后发现，语言模型虽在极端少样本预测等特定场景中可与专用模型竞争，但一旦直接评估结构表示与计算能力，并无证据表明其能全面取代机器学习中的专用架构。 |
| [^265] | [CultureConverse: A Multilingual Multi-turn Simulation Harness for Culturally Grounded Assistance in East and Southeast Asia](https://arxiv.org/abs/2608.28405) | 该论文提出CultureConverse，一个覆盖东亚与东南亚10个地区、58个子群体身份和7个领域的多语言多轮文化情境化助手对话模拟与评测框架，并构建了包含14,610个基准评测回合和274,295个oracle引导对话的数据集，弥补了传统单选题式文化评测无法反映多轮实际辅助场景的不足。 |
| [^266] | [Entity tracking emerges in sub-billion parameter language models and exceeds human performance in naturalistic narratives](https://arxiv.org/abs/2608.18083) | 本文发现实体追踪能力在低至4.1亿参数的语言模型中即已出现，并在自然叙事中随模型规模增大而超越人类表现，而人类表现则受叙事复杂度影响。 |
| [^267] | [Policy Iteration with Human Feedback: Bringing Post-Training RL to In-context Learning](https://arxiv.org/abs/2608.16831) | 本文提出PIHF方法，利用预训练语言模型作为执行基础，通过语言模型批评者和临床专家的循环评估与修订，将强化学习思想引入上下文学习，从而改进策略性能。 |
| [^268] | [Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus](https://arxiv.org/abs/2608.12149) | 本文首次系统研究了混合线性注意力大语言模型中的大规模激活现象，发现了注意力前尖峰和尖峰间平台两种新形态，并揭示了它们与架构配置的关系。 |
| [^269] | [RAISE: Diagnosing Acquisition Collapse in Costly LLM Signals](https://arxiv.org/abs/2608.10441) | 本文识别出“获取崩溃”这一失败模式，并提出了RAISE预路由诊断框架，帮助判断昂贵的LLM信号在何时才值得调用，从而避免盲目调用造成的资源浪费。 |
| [^270] | [VectraYX-Vision-1B: A Sub-2B Spanish/LATAM Cybersecurity Vision-Language Model with Structured Visual Reasoning and Native Tool Use](https://arxiv.org/abs/2608.08477) | v3版本将原生训练的Qwen2-VL视觉塔移植到同一冻结解码器上，使原本失败的8半字节地址字段精确率从0.00跃升至0.81，且token预算比2x2平铺更粗糙，证明分辨率并非关键变量。 |
| [^271] | [MoEGen: Mixture-of-Experts for Instance-Adaptive LoRA Generation](https://arxiv.org/abs/2608.03275) | MoEGen将基于MoE的PEFT从专家选择转变为专家条件参数生成，用小型专家码和轻量级超网络按输入动态生成低秩更新，在避免适配器存储随专家数线性增长的同时实现实例自适应适配。 |
| [^272] | [OpenART: Scaling Agent Red Teaming via Open-Ended Environment Evolution](https://arxiv.org/abs/2608.00677) | 该论文提出了OpenART——一个通过环境演化实现可扩展智能体红队测试的开放式平台，包含覆盖50个领域的1万多个有状态场景，并提出演化马尔可夫超图攻击（EMHA）这一黑盒策略，以系统探索智能体在长时程有状态工作流中不断演化的安全攻击面。 |
| [^273] | [Verifier-Induced Support Reshaping in On-Policy Optimization](https://arxiv.org/abs/2608.00220) | 本文揭示了RLVR训练会重塑策略的支撑集，使其在提升当前任务表现的同时让后续任务的成功行为难以被采样到，例如数学训练使IFEval的pass@1提升6.5个百分点但best@32下降9.8个百分点。 |
| [^274] | [A Constitution-Grid Instrument for Data-Efficient RL Alignment (C-Guard)](https://arxiv.org/abs/2608.00180) | 本文提出C-Guard和C-LIM，通过宪法网格生成训练数据并逐单元评分，在训练前识别无效数据区域，显著提升RL对齐中的数据效率和安全性。 |
| [^275] | [Trustworthiness Costs of Domain Adaptation in Small Language Models:A Cross-Architecture Empirical Study](https://arxiv.org/abs/2608.00042) | 本文首次通过跨架构、跨领域、跨训练数据条件的系统实证研究，量化了小语言模型在领域自适应微调过程中付出的可信度代价（事实校准与对抗鲁棒性），并对比了四种微调策略在医疗、法律、金融领域中的可信度表现。 |
| [^276] | [ORCA-bench: How Ready Are Language Model Agents for Oncall?](https://arxiv.org/abs/2607.28545) | 该论文提出了ORCA-bench基准，将1,079个根因分析任务与真实可观测性工具接口及六天生产级遥测数据相结合，系统评估语言模型智能体在值班根因分析场景中的真实能力。 |
| [^277] | [AdvancedMathBench: A Benchmark Suite for Advanced Mathematical Proof Generation and Verification](https://arxiv.org/abs/2607.11849) | 该论文提出AdvancedMathBench基准套件，包含245道本科及博士资格考试级别的高等数学证明题，并配套基于大规模专家标注训练的自动验证流水线，以细粒度方式评估大语言模型在高等数学证明生成与验证上的推理能力。 |
| [^278] | [Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning](https://arxiv.org/abs/2606.29985) | 该论文提出“方法层面多样性”概念，证明现有多样性指标只是表面措辞差异的不可靠代理而非解题策略差异，且直接优化LLM裁判的多样性奖励会导致投机行为而非真正拓宽解题思路，但方法多样的候选集能提升测试时扩展效果。 |
| [^279] | [TriageRA-CCF: Source-Side Clinical Confidence and Coverage Signals for Adaptive Rank Budgeting in Medical LLMs](https://arxiv.org/abs/2606.29375) | 该论文提出TriageRA-CCF，利用仅从源训练数据计算的三种信号——基础模型答案置信度、临床覆盖度和反事实近似命中代理——来训练直通式预算路由器，为每个医学问题自适应分配LoRA秩预算，从而在避免路由坍缩的同时提升参数高效医学问答性能。 |
| [^280] | [Does Anthropomorphic Language Impact Public Perceptions of AI?](https://arxiv.org/abs/2606.29121) | 本研究通过815名参与者的实验，考察了AI公共话语中拟人化语言对公众认知的影响，并比较了这种影响在大语言模型与推荐系统两类AI技术之间的差异。 |
| [^281] | [HPRO: Hierarchical Progressive Reward Optimization via Preference Extraction for Emotional Text-to-Speech](https://arxiv.org/abs/2606.28249) | 提出分层渐进奖励优化框架HPRO，通过HD-Emo编解码器将语音解耦为独立的内容与风格偏好标记以缓解信息冲突，并弥合句子级奖励与帧级生成之间的尺度差距，从而提升情感文本转语音的表现力。 |
| [^282] | [Epiphany-Aware KV Cache Eviction Without the Attention Matrix](https://arxiv.org/abs/2606.26472) | 本文提出EpiKV，一种通过直接读取模型前向传播中的内部表示变化（顿悟分数）来淘汰KV缓存的方法，无需注意力矩阵，可将可行上下文长度扩展至传统注意力评分方法的16倍，且无需训练或自定义内核。 |
| [^283] | [The Hitchhiker's Guide to Agentic AI: From Foundations to Systems](https://arxiv.org/abs/2606.24937) | 该论文（著作）是一部从大语言模型基础、对齐与推理技术到智能体AI全栈覆盖的构建自主AI系统综合实践指南，其核心观点是：构建优秀的智能体系统必须理解技术栈的每一个层级。 |
| [^284] | [CORE-BREW: LLR-Based Soft Decoding for Robust Multi-Bit LLM Watermarking](https://arxiv.org/abs/2606.24163) | CORE-BREW通过恒定命中率校准推导出逐token的对数似然比以实现软判决解码，并结合熵感知擦除机制，显著提升了多比特LLM水印在编辑和改写攻击下的检测鲁棒性与信息恢复能力。 |
| [^285] | [Is Agent Code Less Maintainable Than Human Code?](https://arxiv.org/abs/2606.21804) | 本研究提出 CodeThread 框架，通过受控实验发现智能体基于智能体生成代码解决任务的效率低于基于人类代码（任务解决率最多下降 13.1%），且传统可维护性指标无法解释这一差异。 |
| [^286] | [Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning](https://arxiv.org/abs/2606.04923) | 本文提出了一个可控黑客环境CHERRL，通过注入已知偏见到评判者中，实现了奖励黑客行为的稳定复现与分析，为检测和缓解该问题提供了实验平台。 |
| [^287] | [Reasoning with Sampling: Cutting at Decision Points](https://arxiv.org/abs/2605.30327) | 该研究表明从基础模型的幂分布中采样即可达到媲美强化学习训练的推理能力，并提出应在推理轨迹中的关键决策点处进行切割重采样，以实现高效的混合采样。 |
| [^288] | [Do Proactive Agents Need an LLM to Decide When to Act?](https://arxiv.org/abs/2605.30152) | 该论文提出用时间图学习（TGL）控制器替代大语言模型来决定主动式代理何时介入及如何选择上下文，在每事件仅11.13毫秒的低成本下实现了对语言代理性能的提升。 |
| [^289] | [Evaluating Cross-lingual Knowledge Consistency in Code-Mixed vis-a-vis Indian Languages using IndicKLAR](https://arxiv.org/abs/2605.29637) | 提出IndicKLAR基准，涵盖18种印度语言及11个语言对的代码混合变体，揭示大模型在印度本土语言与英语间的知识准确率差距可达约0.50，而代码混合输入能将该差距缩小至约0.05以内。 |
| [^290] | [Diversifying RLVR Rollouts via First-Token Exploration](https://arxiv.org/abs/2605.28295) | 论文发现回复首个词元的分布高度集中且与答案正确性关系微弱，据此提出轻量级方法REFT，通过让首个词元多样化来拓宽RLVR每组轨迹的推理路径探索，且几乎不损失回答质量。 |
| [^291] | [KSAFE-MM: A Multimodal Safety Benchmark via Localized Contextualization for Korean Cultural Risks](https://arxiv.org/abs/2605.28013) | 该论文提出了KSAFE-MM基准，通过语言情境化评估韩国语境下的全球共享安全风险（KSAFE-MM-G），并结合本地化视觉查询与越狱式文本查询来评估文化依赖性的安全漏洞（KSAFE-MM-C），弥补了现有安全评估工具以英语为中心且忽视本地文化风险的不足。 |
| [^292] | [Decomposing and Measuring Evaluation Awareness](https://arxiv.org/abs/2605.23055) | 该论文基于社会心理学提出了一个评估意识分解框架，将其分为环境成分（八类触发因素）和模型成分，并通过思维链监测发现九个前沿模型对四个基准测试的识别率取决于模型与基准的配对方式，且识别很少引发行为变化，即使引发，方向也取决于评估类型。 |
| [^293] | [Remember Your Trace: Memory-Guided Long-Horizon Agentic Framework for Consistent and Hierarchical Repository-Level Code Documentation](https://arxiv.org/abs/2605.14563) | 提出MemDocAgent长程智能体框架，通过依赖感知的遍历引导与基于共享记忆RepoMemory的记忆引导智能体交互，在覆盖整个仓库的单一集成上下文中生成一致且具有分层结构的仓库级代码文档。 |
| [^294] | [PRISM: A Geometric Risk Bound for Decomposing Drift into Scale, Shape, and Head](https://arxiv.org/abs/2605.11608) | 提出PRISM，一种交叉熵风险差距的闭式上界，可将模型漂移精确分解为尺度、形状和预测头三个可测量的几何轴，从而将变体的特征变化与性能退化联系起来。 |
| [^295] | [PowerStep: Memory-Efficient Adaptive Optimization via $\ell_p$-Norm Steepest Descent](https://arxiv.org/abs/2605.10335) | PowerStep 受 $\ell_p$-范数最速下降启发，仅对单个动量缓冲区施加带符号幂变换即可实现无需存储二阶矩的自适应优化，相比 AdamW 将 fp32 优化器状态内存减半，结合 int8 量化更可减少约 8 倍，同时在 124M 到 235B 参数的 Transformer 上保持有竞争力的性能。 |
| [^296] | [Relative Kinetic Utility: Calibrating Cross-Layer Credit for Global Structured LLM Pruning](https://arxiv.org/abs/2605.09008) | 提出Global RKU，一种无标签的全局结构化剪枝准则，通过最终隐藏状态的激活-梯度信号衡量通道参与度，并利用块相对归一化消除块级公共尺度对跨层比较的干扰，使不同层通道分数具有可比性，从而实现更准确的大语言模型全局剪枝。 |
| [^297] | [GRAVITY: Architecture-Agnostic Structured Anchoring for Long-Horizon Conversational Memory](https://arxiv.org/abs/2605.01688) | GRAVITY提出了一种与架构无关的辅助记忆层，在生成阶段将对话整合为实体档案、事件轨迹和跨会话主题摘要并注入提示，在五种异构记忆系统和两个基准上均显著提升了长程对话记忆性能。 |
| [^298] | [Think Multilingual, Not Harder: A Data-Efficient Framework for Teaching Reasoning Models to Code-Switch](https://arxiv.org/abs/2604.15490) | 该论文提出了首个基于语言学与行为动机的微调框架，并构建了CoRe语料库，用于识别大型语言模型中有益的语码转换推理行为，从而以数据高效的方式教会推理模型更有效地利用语码转换来提升推理能力。 |
| [^299] | [Revisiting the Capacity Gap in Chain-of-Thought Distillation from a Practical Perspective](https://arxiv.org/abs/2604.08880) | 该论文从实践视角重新审视思维链蒸馏中的容量差距问题，发现在更现实的实验设置下容量差距并非总是主导因素，当候选教师模型性能差异显著时选择更强的教师通常更优，从而为教师模型选择提供了实用指导。 |
| [^300] | [Screening Is Enough](https://arxiv.org/abs/2604.01178) | 本文提出“筛选”注意力机制，通过显式阈值将查询-键相似度转换为绝对相关度，无需推理时扩展即可在长文本中保持低困惑度和稳健检索，并据此构建了参数效率更高、零样本性能更强的Multiscreen语言模型架构。 |
| [^301] | [UltRAG: a Universal Simple Scalable Recipe for Knowledge Graph RAG](https://arxiv.org/abs/2603.28773) | UltRAG是一种无需训练的知识图谱RAG方案，通过结合LLM查询生成、完全归纳式神经查询执行器和LLM仲裁，在无需重训模型的情况下于KGQA任务上取得最先进结果，并支持Wikidata规模的超大规模图谱。 |
| [^302] | [RAWR: Reward Assignment Without Rollouts in Verifiable Domains](https://arxiv.org/abs/2603.17815) | 提出MCNIG方法，利用净信息增益（NetIG）在可验证领域中自动标注推理步骤质量，无需昂贵的人工标注或计算密集的rollout即可训练出高性能的过程奖励模型。 |
| [^303] | [SiDiaC-v.2.0: Sinhala Diachronic Corpus Version 2.0](https://arxiv.org/abs/2603.10861) | 本文构建了迄今最大的僧伽罗语历时语料库SiDiaC-v.2.0，包含185部文学作品共22.9万词，时间跨度从公元5世纪至20世纪，并提供了按写作日期标注的子集，为僧伽罗语的历史语言学研究提供了重要资源。 |
| [^304] | [Bootstrapping Audiovisual Speech Recognition in Zero-AV-Resource Scenarios](https://arxiv.org/abs/2603.08249) | 本研究证明在完全没有真实视听数据的零资源场景下，仅依靠700余小时合成说话人视频作为唯一视觉监督来微调AV-HuBERT，即可在加泰罗尼亚语基准上以更少的参数和训练数据实现接近SOTA的视听语音识别性能。 |
| [^305] | [Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing Optimization](https://arxiv.org/abs/2603.08091) | 提出JudgeBiasBench基准，通过4维度偏差分类体系和受控偏差注入流程系统量化LLM评判者的12种代表性偏差类型，揭示现有生成式和判别式评判者均存在显著且多样的偏差模式。 |
| [^306] | [SalamahBench: Dialect and Category Level Safety Evaluation of Arabic Language Models](https://arxiv.org/abs/2603.04410) | 本文提出 SalamahBench 基准，包含 8,270 条经人工验证的有害提示及其在现代标准阿拉伯语与五种方言下的共 49,620 对实例，并引入方言偏移和类别特定方言偏差两个指标，从方言和危害类别层面系统评估阿拉伯语语言模型的安全性。 |
| [^307] | [A theoretical model of dynamical grammatical gender shifting based on set-valued set function](https://arxiv.org/abs/2603.03510) | 本文提出了一个以集值集函数形式化定义的“基于模板的模块化认知模型”，通过将词项与形态模板动态配对，揭示名词语法性转换的非线性底层规律。 |
| [^308] | [Evaluating Test-Time Scaling of General LLM Agents](https://arxiv.org/abs/2602.18998) | 本文提出了一个统一覆盖搜索、编程、推理和工具使用的现实基准，系统研究了LLM智能体在序列扩展与并行扩展两个维度的测试时扩展行为，发现领先智能体从领域特定评估转向现实场景时性能显著下降，并通过细粒度扩展测试时计算刻画了性能上界。 |
| [^309] | [Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems](https://arxiv.org/abs/2602.15382) | 该论文提出“视觉虫洞”，将视觉-语言模型的视觉输入接口重新用于通信通道，通过通用视觉编解码器和共享参考空间，使冻结的异构多智能体之间能够以仅需O(N)组件的架构进行潜空间连续通信。 |
| [^310] | [On Calibration of Large Language Models: From Response To Capability](https://arxiv.org/abs/2602.13540) | 该论文提出了“能力校准”这一新的评估框架，将大语言模型校准的研究焦点从单次响应的置信度转向查询级的整体置信度，并从理论和实证上证明其与传统的响应校准存在本质区别。 |
| [^311] | [Evaluating Alignment of Behavioral Dispositions in LLMs](https://arxiv.org/abs/2602.11328) | 提出STAR框架，将心理学问卷改编为现实的寻求建议场景，构建2.3万场景数据集评估25个LLM的行为倾向与人类的对齐程度，发现前沿模型在人类高共识时有15-20%的偏差，且在人类意见分歧时其建议多样性明显不足。 |
| [^312] | [IESR:Efficient MCTS-Based Modular Reasoning for Text-to-SQL with Large Language Models](https://arxiv.org/abs/2602.05385) | 该论文提出IESR框架，通过信息理解与模式链接、基于MCTS的多路径推理与多数投票机制、以及轨迹一致性验证模块三大创新，使轻量级大语言模型在Text-to-SQL任务上达到最先进性能，同时降低了企业部署成本。 |
| [^313] | [MGSM-Pro: A Simple Strategy for Robust Multilingual Mathematical Reasoning Evaluation](https://arxiv.org/abs/2601.21225) | 提出 MGSM-Pro 数据集，通过 GSM-Symbolic 方法为 MGSM 每道题生成五个变体实例，揭示了许多低资源语言在数字变化下性能大幅下降，且模型在高资源语言上的鲁棒性无法迁移到低资源语言。 |
| [^314] | [Asymptotic Universal Alignment: A New Alignment Framework via Test-Time Scaling](https://arxiv.org/abs/2601.08777) | 该论文提出通过测试时扩展实现“渐近通用对齐”的新框架，证明生成 k 个候选回答的乘积策略能以最优且不可超越的速率 f(k)=k/(k+1) 逼近完美胜率，为处理用户偏好冲突提供了理论基础。 |
| [^315] | [How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction](https://arxiv.org/abs/2601.08626) | 该论文提出了确定性基准OrderProbe，利用中日韩固定四字表达的唯一规范顺序来评估大语言模型从打乱输入中重构结构的能力，发现即使是最先进的模型，零样本恢复率也经常低于35%。 |
| [^316] | [MAPLE: Medical Aspect-Based Summarization with Phrase-Level Evidence](https://arxiv.org/abs/2601.03418) | 该论文提出MAPLE基准，首次将医学方面摘要中每个论断的证据粒度细化到短语级，并引入解耦评估框架，分别衡量内容质量、可追溯性和可定位性。 |
| [^317] | [Block Sparse Flash Attention](https://arxiv.org/abs/2512.07011) | 提出块稀疏Flash注意力（BSFA），通过精确计算查询-键相似度选取最重要的值块并与校准阈值比较来跳过约50%的计算和内存传输，作为免训练的即插即用替代方案加速长上下文推理且不损失模型质量。 |
| [^318] | [Evidence-Guided Schema Normalization for Temporal Tabular Reasoning](https://arxiv.org/abs/2512.00329) | 该研究将时序表格问答重构为自动化知识库构建任务，并通过受控交叉实验证明模式设计质量对问答准确率的影响远大于查询模型的选择（分别解释79.5%和1.6%的EM方差），据此提炼出模式设计原则。 |
| [^319] | [Beyond Semantics: How Temporal Biases Shape Retrieval in Transformer and State-Space Models](https://arxiv.org/abs/2510.22752) | 研究发现，Transformer与状态空间模型在上下文学习中的检索不仅受语义驱动，还存在显著的时间位置偏差——模型更倾向于关注序列开头或末尾附近的重复token，类似于人类情景记忆中的时间分离机制。 |
| [^320] | [POET: Preference Optimization for Enhanced Text-to-Image Generation](https://arxiv.org/abs/2510.12041) | POET提出了一种利用大语言模型自动重写用户提示词的框架，通过复合奖励系统和迭代式DPO训练，从多模态反馈中学习模型偏好的提示词结构，从而在无需昂贵监督微调数据的情况下提升文本到图像生成的质量。 |
| [^321] | [TagPR: Tag-Guided Process Supervision for Personalization Reasoning in Large Language Models](https://arxiv.org/abs/2509.23140) | TagPR通过在推理过程中添加语义标签实现逐步的过程监督，并结合基于用户嵌入的个性化奖励模型进行多阶段强化学习，显著提升了大语言模型在个性化任务上的推理能力。 |
| [^322] | [Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs](https://arxiv.org/abs/2509.21305) | 该论文首次将大语言模型的谄媚行为因果地分解为谄媚性附和与谄媚性赞扬两种独立机制，证明它们在潜空间中沿不同线性方向编码，且可被独立调控而互不干扰，并在不同模型家族和规模上保持一致。 |
| [^323] | [Predicting Team Performance from Communications in Simulated Search-and-Rescue](https://arxiv.org/abs/2503.03791) | 该研究通过分析模拟搜救任务中的对话数据，运用主题建模和聚类方法识别团队特质，证明这些从通信中推断出的个体特质和团队动态能够有效预测团队绩效的差异。 |
| [^324] | [Dynamic Optimizations of LLM Ensembles with Two-Stage Reinforcement Learning Agents](https://arxiv.org/abs/2502.04492) | 本文提出RL-Focal两阶段强化学习框架：第一阶段的决策者智能体通过最大化错误多样性与推理性能，为下游任务从N个大语言模型中动态选择小规模集成；第二阶段的融合智能体解决模型间推理冲突并自适应融合所选模型，实现大语言模型集成的动态优化。 |
| [^325] | [BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models](https://arxiv.org/abs/2406.00083) | BadRAG揭示了一种针对检索增强生成系统的新型攻击：攻击者通过向知识库注入恶意文段，当用户查询包含特定触发词时即可操纵系统响应，且无需修改用户输入或模型权重。 |
| [^326] | [MONOVAB : An Annotated Corpus for Bangla Multi-label Emotion Detection.](http://arxiv.org/abs/2309.15670) | 这个研究构建了一个基于孟加拉语的注释语料库，用于多标签情感检测。通过使用基于上下文的方法以及BERT模型，填补了这一学科领域的空白。 |

# 详细

[^1]: Imagine3D-LLM：教会多模态大语言模型在回答之前想象3D场景

    Imagine3D-LLM: Teaching MLLMs to Imagine 3D Scenes Before Answering

    [https://arxiv.org/abs/2609.38177](https://arxiv.org/abs/2609.38177)

    该论文受人类空间推理过程启发，提出Imagine3D-LLM，教会多模态大语言模型在回答问题前通过粗略识别跨视角共同物体、推断视角间相对几何关系来构建紧凑的粗略3D场景布局，而非依赖细粒度几何线索，从而提升多视角3D推理能力。

    

    从多视角图像推理三维世界仍然是多模态大语言模型（MLLMs）面临的一个根本性挑战。尽管现代多模态大语言模型能够有效处理单张图像输入，但它们难以将跨视角的证据整合成连贯的3D理解。越来越多的研究试图通过向多模态大语言模型注入3D感知能力来弥合这一差距，要么通过增强细粒度的像素级跨视角对应关系，要么通过融合3D几何基础模型的特征，但与人类推理相比仍存在相当大的差距。在本工作中，我们重新审视了人类的空间推理过程，研究表明人类并不依赖细粒度的几何线索，而是粗略地识别跨视角中的常见物体，推断视角之间的相对几何关系，并构建出场景的粗略3D布局。受这一过程的启发，我们提出了Imagine3D-LLM，这是一个学会构建类似紧凑3D场景表示的多模态大语言模型……

    arXiv:2609.38177v1 Announce Type: cross  Abstract: Reasoning about the 3D world from multi-view images remains a fundamental challenge for Multimodal Large Language Models (MLLMs). While modern MLLMs handle single-image inputs effectively, they struggle to integrate evidence across viewpoints into a coherent 3D understanding. A growing body of work attempts to close this gap by injecting 3D awareness into MLLMs, either by boosting fine-grained pixel-level cross-view correspondence or by fusing features from 3D geometry foundation models, yet a substantial gap to human reasoning persists. In this work, we revisit human spatial reasoning, which suggests that rather than relying on fine-grained geometry cues, humans roughly identify common objects across views, infer the relative geometry between viewpoints, and assemble a coarse 3D layout of the scene. Inspired by this process, we introduce Imagine3D-LLM, an MLLM that learns to assemble a similar compact 3D representation of the scene an
    
[^2]: STEPQuant：Delta规则循环状态量化中误差何时何地产生影响

    STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization

    [https://arxiv.org/abs/2609.38169](https://arxiv.org/abs/2609.38169)

    STEPQuant揭示了Delta规则循环状态量化误差在时间（记忆寿命）和空间（键行/值列）两个维度上的差异化影响，并提出一种时空后训练量化框架，根据误差大小与记忆寿命自适应分配精度，从而在低比特量化下保持模型精度。

    

    线性注意力机制用固定大小的循环状态取代了不断增长的KV缓存，但这些持久状态在并发服务场景下可能成为显著的内存瓶颈。直接将循环状态量化为低精度往往会导致严重的精度下降，因为量化误差会在连续的状态更新中不断传播。我们发现这些误差的影响取决于两个互补的维度：在时间维度上，长期存活的记忆中的误差会在多个解码步骤中持续存在；在空间维度上，不同键行中的误差对模型输出的影响各不相同，而状态幅值在行和列两个方向上都存在显著差异。基于这些观察，我们提出了STEPQuant，一个面向Delta规则循环状态的时空后训练量化框架。STEPQuant根据误差大小和记忆寿命来分配精度，并基于状态分布联合拟合键行与值列的缩放因子。

    arXiv:2609.38169v1 Announce Type: cross  Abstract: Linear attention replaces growing KV caches with fixed-size recurrent states, yet these persistent states can become a substantial memory bottleneck under concurrent serving. Directly quantizing recurrent states to low precision often leads to severe accuracy degradation, as quantization errors propagate through successive state updates. We discover that the impact of these errors depends on two complementary dimensions: temporally, errors in long-lived memory can persist across many decoding steps; spatially, errors in different key rows affect model outputs differently, while state magnitudes vary substantially along both rows and columns. Motivated by these observations, we propose STEPQuant, a spatial-temporal post-training quantization framework for Delta-rule recurrent states. STEPQuant allocates precision according to error magnitude and memory lifetime, and jointly fits key-row and value-column scales based on state distributio
    
[^3]: EmoRES-TTS：面向情感语音生成的残差增强向量引导方法

    EmoRES-TTS: Residual-Enhanced Vector Steering for Emotional Speech Generation

    [https://arxiv.org/abs/2609.38157](https://arxiv.org/abs/2609.38157)

    该论文发现情感向量可分解为使语音脱离平淡表达的共享分量和引导生成指定情感的残差分量，并据此提出无需重新训练骨干网络的 EmoRES 方法来分别控制这两个分量，从而更可靠地生成指定情感的语音。

    

    情感条件化的文本转语音（TTS）模型可能无法可靠地表达所请求的情感，而通过额外训练来提升可控性在计算成本和情感标注语音训练数据方面都十分昂贵。因此，我们研究了向量引导这一无需训练的方法，即通过修改冻结模型的内部表示来实现控制。CoCoEmo 是一种传统的情感 TTS 向量引导方法，它将每个情感向量视为一个不可分割的方向，并由单一的全局强度进行控制，这限制了对所请求情感的遵循程度。在这项工作中，我们首先发现情感向量可以分解为一个将语音从平淡表达中分离出来的共享分量，以及一个将生成过程引导向所请求情感的残差分量。基于这一发现，我们提出了面向 TTS 的情感残差增强引导方法，这是一种无需重新训练骨干网络即可分别控制这两个分量的新方法。在（原文摘要到此截断）

    arXiv:2609.38157v1 Announce Type: cross  Abstract: Emotion-conditioned text-to-speech (TTS) models may fail to express the requested emotion reliably, and improving controllability by additional training is costly in both computation and emotion-labeled speech training data. We therefore study vector steering, a training-free approach that modifies the internal representations of a frozen model. CoCoEmo, a conventional vector steering method for emotion TTS, treats each emotion vector as an indivisible direction controlled by a single global strength, limiting adherence to the requested emotion. In this work, we first discover that an emotion vector can be decomposed into a shared component that moves speech away from neutral expression and a residual component that directs generation toward the requested emotion. Building on this finding, we propose Emotion Residual-Enhanced Steering for TTS (EmoRES), a novel method that controls the two components without retraining the backbone. On 
    
[^4]: 超越时间线：利用接地实体传记增强长视频记忆

    Beyond the Timeline: Augmenting Long-Video Memory with Grounded Entity Biographies

    [https://arxiv.org/abs/2609.38155](https://arxiv.org/abs/2609.38155)

    该论文提出长视频记忆框架GEB（接地实体传记），将跨片段对同一物理实例的视觉观测聚合为可检索的实体传记，使模型在长视频问答中能够基于身份关联跨时间追踪特定实体。

    

    回答关于长视频的问题通常需要跨越数小时甚至数天，将涉及同一对象的事件关联起来。按时间顺序的描述和从文本派生的实体往往无法解决物理身份的问题：不同的对象可能共享相同的描述，而同一对象的观测结果在不同事件之间仍相互割裂。因此，检索相关事件并不一定能恢复问题所涉及的那个特定实体的“传记”。为了解决这一问题，我们提出了接地实体传记（GEB），这是一个长视频记忆框架，它将跨视频片段中对同一物理实例的视觉接地观测聚合为可检索的实体传记，同时保留每个时刻的上下文。在问答过程中，实体传记与情节证据一同被检索，使模型能够借助记忆构建阶段建立的身份关联，沿事件轨迹追踪该实体的经历。在四个基准上的评估显示……

    arXiv:2609.38155v1 Announce Type: cross  Abstract: Answering questions about long videos often requires connecting events involving the same objects across hours or days. Chronological descriptions and text-derived entities can leave physical identity unresolved: different objects may share a description, while observations of the same object remain disconnected across events. Retrieving relevant events therefore does not necessarily recover the "biography" of the particular entity a question concerns. To address this, we introduce Grounded Entity Biographies (GEB), a long-video memory framework that groups visually grounded observations of the same physical instance across clips into retrievable biographies while preserving the context of each moment. During question answering, the biography is retrieved alongside episodic evidence, allowing the model to follow an entity through events using identity links established during memory construction. Evaluations across four benchmarks, inc
    
[^5]: 基于教师监督预训练潜在信息反馈Transformer

    Pretraining Latent Information Feedback Transformers with Teacher Supervision

    [https://arxiv.org/abs/2609.38149](https://arxiv.org/abs/2609.38149)

    提出LIFT架构与训练方法，通过教师强制方式将循环状态学习转化为预测问题，使Transformer语言模型在预训练阶段能够跨生成步骤传播潜在状态信息，突破了传统前馈架构中信息只能通过解码token向下传递的瓶颈。

    

    Transformer语言模型（LM）是前馈式的：深层表示从不反馈到浅层，且信息在生成步骤之间向下流动的唯一途径是解码出的token。这种狭窄的通道迫使模型重新计算中间结果并丢弃备选的延续方案。在这项工作中，我们在预训练阶段消除了这一瓶颈，提出了LIFT（潜在信息反馈Transformer）架构和训练方法，使语言模型能够在生成过程中跨步骤传播状态。我们通过将循环状态学习转化为教师强制预测问题来实现这一目标：每个输入token与一个信息密集的状态配对，该状态由现成的预训练语言模型的下一个token分布推导得出。该模型通过少量额外参数进行扩展，然后被训练以同时预测下一个token和下一个状态。由于输入状态是预先计算的，预训练过程……

    arXiv:2609.38149v1 Announce Type: new  Abstract: Transformer language models (LMs) are feed-forward: deep-layer representations are never fed back to shallower layers, and the only pathway for information to flow downward across generation steps is the decoded token. This narrow channel forces models to recompute intermediate results and to discard alternative continuations. In this work, we remove this bottleneck during pretraining, introducing the LIFT (Latent Information Feedback Transformer) architecture and training method which enable LMs to propagate state across generation. We achieve this by turning recurrent-state learning into a teacher-forced prediction problem: each input token is paired with an information-dense state, derived from the next-token distribution of an off-the-shelf pretrained LM. The model, extended with a small number of additional parameters, is then trained to predict both the next token and the next state. As the input states are precomputed, pretraining
    
[^6]: 在测试时AI4AI中通过学习元技能进行智能体执行环境设计

    Learning Meta-Skills for Agent Harness Design in Test-Time AI4AI

    [https://arxiv.org/abs/2609.38143](https://arxiv.org/abs/2609.38143)

    提出Meta-Skill方法，使权重固定的Builder模型从Target的执行反馈中学习可复用的支持原则，为未见任务构建更好的执行环境，相比无技能构建和直接交付技能库分别提升8.95和12.02个百分点。

    

    智能体的表现既取决于其推理能力，也取决于其所处的执行环境。我们研究了测试时的“AI用于AI”问题，探讨在两个模型权重均保持固定的情况下，“构建者”如何学会为“目标者”构建更好的执行环境。为了使构建者的经验可复用，我们引入了元技能：即明确规定何时需要支持以及应提供何种资源的原则。构建者从开发集上目标者的执行反馈中学习这些原则，然后使用冻结的技能库为未见过的任务构建执行环境。在Harness-Bench和NewtonBench基准上，完整技能库的元技能使宏观平均性能相比无技能构建提升了8.95个百分点，相比将同一技能库直接交付给目标者提升了12.02个百分点。这些结果凸显了将经验转化为可执行支持的价值。当同一模型同时扮演两个角色时仍能获得增益，这进一步表明了一条通往系统级自我提升的路径。

    arXiv:2609.38143v1 Announce Type: new  Abstract: Agent performance depends on both reasoning ability and the environment in which it acts. We study test-time AI-for-AI, asking how a Builder can learn to construct better execution environments for a Target while both models' weights remain fixed. To make the Builder's experience reusable, we introduce Meta-Skill: principles specifying when support is needed and what resources to provide. The Builder learns these principles from Target's execution feedback on the development set, then uses the frozen skill bank to construct harnesses for unseen tasks. Across Harness-Bench and NewtonBench, full-bank meta-skills improve macro-average performance by 8.95 percentage points over no-skill construction, and 12.02 points over direct delivery of the same bank to the Target. These results highlight the value of translating experience into executable support. Gains when the same model serves both roles further suggest a path to system level self-im
    
[^7]: AdviSD：通过目标性多轮自蒸馏学习为前沿大语言模型提供建议

    AdviSD: Learning to Advise Frontier LLMs via Targeted Multi-Turn Self-Distillation

    [https://arxiv.org/abs/2609.38142](https://arxiv.org/abs/2609.38142)

    该论文从理论上证明“不改变执行结果的纠正”可能限制建议者的学习，并据此提出AdviSD方法，将基于结果的强化学习与来自反馈条件副本的选择性自蒸馏相结合，以提升可训练建议者引导冻结大语言模型执行器的能力。

    

    一个小型可训练的“建议者”可以通过自然语言建议来引导一个冻结的语言模型“执行器”。除了从任务奖励中学习之外，建议者还可以利用已完成交互的反馈来改进其建议。然而，一个看似合理的纠正并不一定会改变执行结果，但从这类纠正中学习仍可能影响建议者在其他情境下的未来决策。在一个共享参数模型中，我们证明：如果此类纠正的目标对有用建议的强化程度弱于其他纠正的目标，那么这类纠正可能会限制学习。相较于从每一条纠正中学习，更少地保留这类纠正能够改善模型的最终性能。受此启发，我们提出“建议者自蒸馏”方法（Advisor Self-Distillation, AdviSD），将基于结果的强化学习与来自建议者反馈条件副本的选择性自蒸馏相结合。反思环节提出纠正，建议者对同一份记录的执行器表现进行评分……

    arXiv:2609.38142v1 Announce Type: new  Abstract: A small trainable advisor can steer a frozen language-model executor using natural-language advice. In addition to learning from task rewards, the advisor can use feedback from completed interactions to improve its advice. However, a plausible correction need not change execution, yet learning from such corrections can still affect the advisor's future decisions in other contexts. In a shared-parameter model, we prove that such corrections can limit learning if their targets favor useful advice less strongly than those of other corrections. Keeping them less often than the rest improves the model's eventual performance compared to learning from every correction. Motivated by this, our method, Advisor Self-Distillation (AdviSD), pairs outcome-based reinforcement learning with self-distillation from a feedback-conditioned copy of the advisor selectively. Reflection proposes corrections, and the advisor scores the same recorded executor res
    
[^8]: LongHarness Bench：面向长上下文推理的语言模型框架压力测试基准

    LongHarness Bench: Stress-Testing Language Model Harnesses for Long-Context Reasoning

    [https://arxiv.org/abs/2609.38137](https://arxiv.org/abs/2609.38137)

    提出LongHarness Bench基准，通过需要多样化检索策略与全局-局部自适应推理的任务，同时评估长上下文语言模型框架的有效性与效率，并揭示不同处理策略之间显著的准确率—成本权衡。

    

    语言模型框架通过利用额外的计算资源，使语言模型能够在长上下文中高效运作。然而，现有的长上下文评估已不足以区分现代框架，这体现在各框架的准确率已趋于饱和，且评估成本大体相似。本文提出了一个用于同时评估长上下文框架有效性与效率的基准。我们的任务需要多样化的检索策略，包括词法搜索与语义匹配，并结合对全局与局部上下文的策略性和自适应推理。上下文中的大部分内容在语义上都是相关的，但在每一步中只有一小部分是有用的，这既构成了一个具有挑战性的搜索问题，也导致不同处理策略之间存在不同的准确率—成本权衡。例如，其中一项任务需要利用散布在多篇文档中的证据，识别出满足多个条件的所有人员；通过策略性地检查最……（原文在此处截断）

    arXiv:2609.38137v1 Announce Type: new  Abstract: Language-model (LM) harnesses enable LMs to operate effectively over long contexts using additional compute. However, existing long-context evaluations are insufficient for distinguishing modern harnesses, reflected by saturated accuracy across harnesses and largely similar evaluation costs. In this paper, we introduce a benchmark for evaluating both the effectiveness and efficiency of long-context harnesses. Our tasks require diverse retrieval strategies, including lexical search and semantic matching, together with strategic and adaptive reasoning over global and local context. Much of the context is semantically relevant but only a small subset is useful at each step, creating both a challenging search problem and different accuracy--cost tradeoffs across processing strategies. For example, one task requires identifying every person satisfying several conditions using evidence scattered across documents; strategically checking the mos
    
[^9]: 从路由信号到选择性审查：MoE视觉语言模型中的视觉重新定位

    From Routing Signals to Selective Review: Visual regrounding in MoE VLMs

    [https://arxiv.org/abs/2609.38111](https://arxiv.org/abs/2609.38111)

    该论文提出首个利用MoE视觉语言模型内部路由信号在生成前检测“目标缺失定位失败”的框架，仅用路由概率训练的简单线性检测器即可达到近完美的检测效果（ROC-AUC高达0.9988），并据此选择性触发审查提示实现纠正。

    

    视觉语言模型（VLM）可能会接受错误的视觉前提，即使目标物体并不存在，也会回答有关其颜色、数量、位置或状态的问题。我们将这种对可靠性至关重要的行为称为“目标缺失定位失败”。现有的视觉定位检测器主要依赖于生成的响应、隐藏状态或不确定性度量。我们提出了首个利用混合专家视觉语言模型内部路由决策的框架，可在生成之前检测目标缺失并引导选择性纠正。我们从Qwen3-VL-30B-A3B-Instruct和Gemma-4-26B-A4B-it中提取目标词元的路由概率，为每个模型分别训练一个L2正则化线性检测器，并利用其预测结果选择性地调用目标感知的审查提示。仅使用路由信号，Qwen和Gemma检测器在GQA-Inpaint上分别达到0.9988和0.9956的ROC-AUC，在外部OBER数据集上仍保持0.8095和0.7781的性能。

    arXiv:2609.38111v1 Announce Type: new  Abstract: Vision-language models (VLMs) may accept false visual premises, answering questions about a target object's color, count, location, or state even when it is absent. We call this reliability-critical behavior a target-absence grounding failure. Existing visual-grounding detectors primarily rely on generated responses, hidden states, or uncertainty measures. We present the first framework to leverage internal routing decisions in Mixture-of-Experts (MoE) VLMs to detect target absence before generation and guide selective correction. We extract target-token routing probabilities from Qwen3-VL-30B-A3B-Instruct and Gemma-4-26B-A4B-it, train a separate L2-regularized linear detector for each model, and use its predictions to selectively invoke a target-aware review prompt. Using routing alone, the Qwen and Gemma detectors achieve ROC-AUCs of 0.9988 and 0.9956 on GQA-Inpaint and retain 0.8095 and 0.7781 on the external OBER dataset, respectivel
    
[^10]: 局部混合如何在全局NoPE注意力中编码相对位置

    How Local Mixing Encodes Relative Position in Global NoPE Attention

    [https://arxiv.org/abs/2609.38109](https://arxiv.org/abs/2609.38109)

    本文通过理论与实证分析揭示，混合Transformer架构中滑动窗口注意力（SWA）和门控线性注意力等局部混合层会在残差流中诱导近因偏差，从而使不使用显式位置编码的全局NoPE注意力层能够隐式地编码相对位置信息。

    

    注意力操作本身是位置不变的。然而，位置信息对自然语言至关重要，因此在基于Transformer的模型中发展出了多种显式位置编码方法，例如旋转位置编码（RoPE）。尽管长期以来人们一直认为显式位置编码是必需的，但最近的研究表明，在全局注意力层中不编码位置（NoPE）、同时交织局部混合层（如滑动窗口注意力SWA和门控线性注意力）的混合方法在大规模应用中取得了成功。然而这种方法如何以及为何有效尚不清楚。在本文中，我们对这类混合模型如何在全局NoPE层隐式编码位置提出了一个解释。在理论和实证证据的支持下，我们的核心论点是：SWA和门控线性注意力会在残差流中诱导一种近因偏差，该偏差传播到……（摘要在此处截断）

    arXiv:2609.38109v1 Announce Type: cross  Abstract: The attention operation is naively position invariant. However, positional information is fundamental to natural language, and therefore a variety of explicit position encodings have been developed in transformer-based models, such as rotary position encoding (RoPE). Although explicit position encodings have long been assumed to be required, recent methods that interleave local mixing layers, such as sliding window attention (SWA) and gated linear attention, while not encoding position (NoPE) in global attention layers has recently been shown to be successful at scale. How and why this approach works is not well-understood. In this paper, we develop an explanation of how hybrid models of this sort can implicitly encode position at global NoPE layers. Supported by both theoretical and empirical evidence, our central argument is that SWA and gated linear attention induce a recency bias in the residual stream that propagates to, and is se
    
[^11]: 正确的答案，无效的轨迹：可验证的小学数学揭示的思维链轨迹问题

    Correct Answers, Invalid Traces: What Verifiable Grade-School Math Reveals About Chain-of-Thought Traces

    [https://arxiv.org/abs/2609.38107](https://arxiv.org/abs/2609.38107)

    研究通过可程序化验证的 iGSM 数学基准发现，答案正确性并不能可靠证明推理有效性——在分布外最难的实例上，31.6% 的正确答案伴随无效的思维链轨迹。

    

    思维链轨迹被广泛视为模型如何得出答案的记录，为调试、智能体审计以及关于推理能力的论断提供依据。然而，检验这种解释十分困难，因为自然语言思考轨迹很少能被机械地验证。我们在 iGSM 中重新审视了这一问题——iGSM 是一个专为研究思考轨迹而设计的合成小学数学基准，曾被用于支持模型学到了推理和规划能力的论断。关键在于，iGSM 揭示了正确解法应当使用的确切数量和依赖关系，使得生成的轨迹可以被逐步程序化检查，从而使我们能够检验正确答案是否总是伴随有效的轨迹。我们首先评估了仅在有效、最小化轨迹上训练的模型。答案正确性与轨迹有效性在分布内几乎一致，但在分布外出现解耦：在最难的实例上，31.6% 的正确答案具有无效轨迹，超过一半……

    arXiv:2609.38107v1 Announce Type: cross  Abstract: Chain-of-thought traces are widely read as records of how models reach their answers, informing debugging, agent auditing, and claims about reasoning. Testing this interpretation is difficult because natural-language thinking traces are rarely mechanically verifiable. We revisit it in iGSM, a synthetic grade-school mathematics benchmark designed to study thinking traces and used to support claims of learned reasoning and planning. Crucially, iGSM exposes the exact quantities and dependencies that a correct solution should use, allowing generated traces to be checked programmatically step by step and enabling us to test whether correct answers are reliably accompanied by valid traces. We first evaluate models trained exclusively on valid, minimal traces. Answer correctness and trace validity nearly coincide in distribution but decouple out of distribution: on the hardest instances, 31.6% of correct answers have invalid traces, over half
    
[^12]: 为效率而剪枝，以公平为代价：剪枝语音大语言模型中的人口群体差异

    Pruning for Efficiency, Paying in Fairness: Demographic Disparities in Pruned Speech-LLMs

    [https://arxiv.org/abs/2609.38106](https://arxiv.org/abs/2609.38106)

    本研究首次系统揭示了音频编码器剪枝会不成比例地损害语音大语言模型中弱势人口群体的识别性能，导致群体间差距成倍扩大，而仅依赖总体词错误率的评估方式会掩盖这一公平性问题。

    

    语音大语言模型的运行成本高昂，因此模型压缩对于实际部署十分重要。然而，压缩模型通常是通过总体词错误率（WER）来筛选的，这可能掩盖剪枝对不同人口群体的影响差异。在本工作中，我们系统地研究了音频编码器剪枝对SLAM-ASR在不同人口群体上的影响。通过使用Fair-Speech和Common Voice数据集，我们发现剪枝对不同人口群体的影响并不均等：表现最好与表现最差群体之间的差距会成倍扩大。这种差异在所有三个编码器规模上都存在，但只有最大规模的模型最初会借助总体WER指标将这些差异掩盖起来。LoRA适配虽然能改善所有群体的WER，但对原本表现较好的群体提升更为显著，并且会进一步扩大某些群体的性能差距。在Common Voice的英语、丹麦语和荷兰语上，口音差距持续存在但并未明显扩大，这表明剪枝的公平性影响会随语言而变化。

    arXiv:2609.38106v1 Announce Type: cross  Abstract: Speech-LLMs are expensive to run, making compression important for real-world deployment. However, compressed models are usually selected using aggregate word error rate (WER), which can hide how pruning affects different demographic groups. In this work, we systematically study the effect of audio encoder pruning on SLAM-ASR for different demographic groups. Using the Fair-Speech and Common Voice datasets, we found that the pruning does not affect all demographic groups equally; the gap between best- and worst-performing groups increases in fold. These disparities appear across all three encoder scales, but only the largest model initially hides them behind aggregate WER. LoRA adaptation improves WER for every group, but benefits groups already performing well more strongly and widens for certain groups. On Common Voice English, Danish, and Dutch, accent gaps persist but do not clearly widen, showing that the fairness effects of pruni
    
[^13]: 仅使用上下文示例实现高效的稠密检索

    Effective Dense Retrieval using Only In-Context Examples

    [https://arxiv.org/abs/2609.38099](https://arxiv.org/abs/2609.38099)

    本文提出RICE方法，仅通过少量上下文示例提示LLM，无需任何检索器训练即可提取高质量稠密表示，显著提升基于提示的LLM嵌入在稠密检索任务中的准确性。

    

    将仅解码器架构的大型语言模型（LLM）转变为强大的稠密检索器通常需要某种形式的检索器训练。在本文中，我们探讨了一个问题：在仅提供少量上下文示例的情况下，能否通过提示（prompting）让LLM直接生成有效的稠密检索表示？为回答这一问题，我们提出了RICE（Representations from In-Context Examples，来自上下文示例的表示），这是一种简单的“免训练”方法，可以从LLM中提取高质量的稠密表示。具体而言，RICE通过上下文示例为LLM提供共享语境，用于查询和文档的编码。我们的结果表明，RICE嵌入能够显著提升基于提示的LLM嵌入的准确性，使其成为构建无需训练的基于LLM的稠密检索器的一种简单方法。我们已在 https://github.com/nourj98/RICE 发布了代码。

    arXiv:2609.38099v1 Announce Type: cross  Abstract: Turning decoder-only large language models (LLMs) into strong dense retrievers typically requires some form of retriever training. In this paper, we ask whether LLMs can instead be prompted to produce effective representations for dense retrieval given only a few in-context examples. To answer this, we introduce RICE (Representations from In-Context Examples), a simple "training-free" approach that extracts high-quality dense representations from LLMs. To do so, RICE conditions the LLM on examples that provide a shared context for query and document encoding. Our results demonstrate that RICE embeddings can substantially improve the accuracy of prompt-based LLM embeddings, establishing it as a simple method to build LLM-based dense retrievers that do not require training. We release our code at https://github.com/nourj98/RICE.
    
[^14]: 大语言模型中的性别偏见普遍存在且高度异质

    Gender bias across LLMs is common and highly heterogenous

    [https://arxiv.org/abs/2609.38036](https://arxiv.org/abs/2609.38036)

    该研究通过两种实验范式测试了来自九个厂商的十款大语言模型，发现性别偏见在模型间普遍存在但高度异质——部分模型表现出反刻板印象的性别归因模式，另一些模型则在道德判断中与人类保护女性免受伤害的倾向一致。

    

    随着大语言模型（LLM）被嵌入具有实际影响的决策支持工具中，理解其中的性别偏见变得越来越重要。以往的研究仅关注少数模型，使得性别偏见在各LLM之间的普遍性和异质性程度尚不明确。我们通过两项研究填补了这一空白，研究对象涵盖2025年4月至2026年6月间发布的十款模型，来自九个厂商，采用两种范式：对刻板印象短语的性别归因（研究1），以及对为防止灾难性后果而对女性或男性实施虐待或酷刑的道德判断（研究2）。在研究1中，十款模型中有两款将男性刻板印象短语归因于女性作者的频率高于反向情况，而三款模型表现出相反的模式。在研究2中，多个模型呈现出不利于男性的不对称性，其方向与已有文献记录的人类倾向于保护女性目标免受伤害的倾向一致，尽管……

    arXiv:2609.38036v1 Announce Type: cross  Abstract: Understanding gender biases in large language models (LLMs) is increasingly important as these systems become embedded in decision-support tools with real consequences. Prior research has focused only on a small set of models, leaving open the extent to which gender biases are common and heterogeneous across LLMs. We address this gap across ten models released between April 2025 and June 2026, spanning nine vendors, using two paradigms: gender attribution to stereotyped phrases (Study 1) and moral judgment of abuse or torture against a woman or a man to prevent a catastrophic outcome (Study 2). In Study 1, two of ten models attributed masculine-stereotyped phrases to female writers more often than the reverse, while three models showed the opposite pattern. In Study 2, several models converged on a male-disadvantaging asymmetry that was directionally consistent with a documented human tendency to protect female targets from harm, thoug
    
[^15]: 基于大语言模型三阶段功能分层的层信息感知微调

    Layer-Informed Fine-Tuning via Three-Stage Functional Segmentation of LLMs

    [https://arxiv.org/abs/2609.38027](https://arxiv.org/abs/2609.38027)

    本文提出大语言模型各层在概念化、推理和文本化上存在结构化分工的假设，并据此提出层信息感知微调方法（LIFT），借助敏感性分析定位瓶颈功能阶段，仅更新功能关键层即可实现高效有效的微调。

    

    arXiv:2609.38027v1 公告类型：新 摘要：近年来，大语言模型（LLM）在推理任务上的表现十分出色，甚至在多项基准测试中超越了人类能力。然而，学术界对于大语言模型的结构和内部参数如何逐步解决复杂推理问题，仍然缺乏清晰的理解。在本研究中，我们考察了大语言模型在跨语言材料上的推理过程，并提出一个假设：大语言模型的各层在概念化、推理和文本化三个阶段上呈现出结构化的分工。基于这一假设，我们引入了一种利用敏感性分析的瓶颈识别机制，以精确定位针对特定任务最关键的功能阶段。利用这一洞见，我们提出了一种新颖的方法——层信息感知微调（LIFT），通过选择性地仅更新这些功能关键层，实现了高效且有效的微调。

    arXiv:2609.38027v1 Announce Type: new  Abstract: In recent years, the performance of large language models (LLMs) on reasoning tasks has been remarkable, even surpassing human capabilities on various benchmarks. However, there remains a lack of clear understanding in the academic community regarding how the structure and internal parameters of LLMs progressively solve complex reasoning problems. In this study, we investigate the inference process of LLMs on cross-linguistic materials and propose the hypothesis that LLM layers exhibit a structured division of labor across conceptualization, reasoning, and textualization. Based on this hypothesis, we introduce a bottleneck identification mechanism using sensitivity analysis to pinpoint the most critical functional stage for a specific task. Leveraging this insight, we propose a novel approach, Layer-Informed Fine-Tuning (LIFT), which achieves efficient and effective fine-tuning by selectively updating only these functionally critical lay
    
[^16]: Dr. OPD：学习遵循哪些信号以实现大语言模型的最优同策略蒸馏

    Dr. OPD: Learning What to Follow for Optimal On-Policy Distillation of Large Language Models

    [https://arxiv.org/abs/2609.38025](https://arxiv.org/abs/2609.38025)

    提出Dr. OPD框架，将同策略蒸馏中的词元级教师监督权重选择建模为双层优化问题，通过交替更新词元权重与学生策略，识别并强化对学生性能最有价值的教师信号，从而最大化大语言模型蒸馏效果。

    

    同策略蒸馏利用更强教师模型提供的密集的、词元级别的监督信号，在学生模型自身生成的回复上对其进行训练。传统的同策略蒸馏平等对待所有教师信号，假设教师的监督对每个词元同等重要。然而，不同词元处的教师信号对学生模型性能的影响可能截然不同：有些信号能够纠正重要的推理错误，而另一些信号对最终答案几乎没有影响。基于这一观察，我们提出了Dr. OPD（正确执行的同策略蒸馏），它定义了最优的加权同策略蒸馏以最大化学生模型的性能。我们将Dr. OPD表述为一个双层优化问题：学生模型从加权的教师监督中学习，而权重则被选择以使最终学生模型的期望奖励最大化。为了求解Dr. OPD，我们开发了一种高效的迭代求解器，交替更新词元权重和学生策略。

    arXiv:2609.38025v1 Announce Type: cross  Abstract: On-policy distillation (OPD) trains a student on its own generated responses using dense, token-level supervision from a stronger teacher. Vanilla OPD treats all teacher signals equally, assuming that the teacher's supervision is equally important for every token. However, teacher signals at different tokens may have very different effects on the student's performance: some correct important reasoning errors, while others have little effect on the final answer. Motivated by this observation, we introduce Dr. OPD (OPD Done Right), which defines the optimal weighted OPD to maximize the student's performance. We formulate Dr. OPD as a bilevel optimization problem in which the student learns from weighted teacher supervision, while the weights are selected to maximize the expected reward of the resulting student. To solve Dr. OPD, we develop an efficient iterative solver that updates the token weights and student policy alternatively. At e
    
[^17]: 可审计的长期记忆：在LongMemEval-S上测得479/475（满分500）成绩的确定性检索链

    Auditable Long-Term Memory: A Deterministic Retrieval Chain Measured at 479/475 of 500 on LongMemEval-S

    [https://arxiv.org/abs/2609.38021](https://arxiv.org/abs/2609.38021)

    该论文提出一种以确定性检索链（混合候选检索、交叉编码器重排序、覆盖优先数据包编译）为核心、LLM仅作为可替换最终阅读器的可审计长期记忆系统，在LongMemEval-S上两次评测分别获得479/500和475/500，成绩跨越Chronos High的478/500，但差异不足以证明优越性或等效性。

    

    我们在LongMemEval-S上评估了一个可审计的长期记忆系统。其检索链采用混合候选检索、交叉编码器重排序、以覆盖优先的数据包编译以及确定性推理脚手架；大语言模型仅作为可替换的最终阅读器使用。该检索链在470个可回答问题中的468个上将所有金标准会话纳入候选池，并为其中462个生成金标准完整的数据包。使用通过未固定版本的CLI别名调用的Claude Opus阅读器，在GPT-4o评分下，两次500题评测分别获得479/500和475/500的分数。其中72个可回答的知识更新题目使用了经过实质性修改的评分提示词，该修改在官方提示词文本下的效果尚未被测量。这一对结果跨越了Chronos High已发表的478/500；由于阅读器生成方式、评分提示词、可能的数据版本差异，以及系统内部的方差，这些结果既不能确立优越性，也不能确立等效性。在同一数据包上使用grok-4.6-high阅读器的得分为476/474，而……

    arXiv:2609.38021v1 Announce Type: cross  Abstract: We evaluate an auditable long-term memory system on LongMemEval-S. Its retrieval chain uses hybrid candidate retrieval, cross-encoder reranking, coverage-first packet compilation, and deterministic reasoning scaffolds; an LLM is used only as a replaceable final reader. The chain places all gold sessions in the candidate pool for 468/470 answerable questions and produces gold-complete packets for 462/470. With a Claude Opus reader called through an unpinned CLI alias, two 500-question passes score 479/500 and 475/500 under GPT-4o. The 72 answerable knowledge-update rows used a substantively modified scoring prompt whose effect under the official text has not been measured. The pair straddles Chronos High's published 478/500; differences in reader generation, scoring prompt, and possibly data version, plus within-system variance, establish neither superiority nor equivalence. A grok-4.6-high reader on the same packets scores 476/474, whi
    
[^18]: BITEM参加NTCIR-19 R2C2任务：从代理式RAG流水线信号预测置信度

    BITEM at the NTCIR-19 R2C2 Task: Predicting Confidence from Agentic RAG Pipeline Signals

    [https://arxiv.org/abs/2609.37993](https://arxiv.org/abs/2609.37993)

    该论文提出一种由编排器根据代理式RAG流水线运行痕迹（蕴含级联核验证据、多轮去重检索等信号）来计算答案置信度、而无需模型自我评估的方法，在NTCIR-19 R2C2任务中取得第4和第5名，且多轮融合检索在多跳问题上增益最大。

    

    BITEM团队使用单一的代理式流水线参加了NTCIR-19 R2C2任务的两个子任务。在该流水线中，一个模型在电影语料库上进行检索、阅读并记录证据，同时一个编排器持有记录并裁决哪些内容可以提交。只有当蕴含级联将某个论断与其所引用的段落核对之后，该论断才被采纳；只有当足够多的已核验证据支撑某个答案时，该答案才被发布。每个问题运行三到四次，且每一轮检索都从剥离了此前各轮已见内容的语料库中进行。随每个答案提交的置信度由编排器根据运行过程留下的痕迹计算得出，而从不询问模型本身，模型也没有任何自我评分的途径。两个检索运行在22个参赛系统中分别位列第4和第5，对多轮运行进行融合带来0.0709的nDCG@20提升，且提升在多跳和重后处理的问题上最大，组织方将融合后的运行在该类问题上排名第一。

    arXiv:2609.37993v1 Announce Type: cross  Abstract: The BITEM team entered both subtasks of the NTCIR-19 R2C2 task with a single agentic pipeline, in which a model searches, reads and records evidence over a movie corpus while an orchestrator holds the record and rules on what may be submitted. A claim is admitted only once an entailment cascade has checked it against the passage it cites, and an answer is released only once enough checked evidence stands behind it. Each question is run three or four times, every pass retrieving from a corpus stripped of what the earlier passes have already seen. The confidence filed with each answer is computed by the orchestrator from what the run leaves behind and is never asked of the model, which is offered no way to rate itself. The two retrieval runs placed 4th and 5th of 22, pooling the passes was worth 0.0709 nDCG@20, and the gain was largest on the multi-hop and post-processing-heavy questions, where the organisers rank the pooled run top of t
    
[^19]: S³：谱零空间交换使推理模型更高效

    $S^3$: Spectral Null-Space Swap Makes Reasoning Models Efficient

    [https://arxiv.org/abs/2609.37976](https://arxiv.org/abs/2609.37976)

    论文首次发现推理能力的关键在于思考模型权重中位于非思考模型主导子空间零空间内的分量，并提出无需训练的谱零空间交换方法S³，通过配对检查点组合在保持精度的同时大幅提升推理效率。

    

    经思维链训练的大语言模型在推理能力上表现出色，但往往伴随着过多的token开销。我们发现推理能力的核心在于思考模型的权重分量位于由对应非思考模型主导奇异方向所定义投影的零空间内，并且移除该子空间分量可以在不损害思考模式后训练所获得精度的前提下大幅提升推理效率。与现有大多在主导子空间内进行操作的方法不同，我们首次揭示了零空间的关键作用，并将其用于模型优化。受这一发现的启发，我们提出了谱零空间交换（S³），这是一种无需训练的、由配对的非思考与思考检查点组成的融合方法。我们的方法将非思考模型保留在其自身的主导子空间内，而将思考检查点置于该子空间之外，在提升推理效率的同时保持（摘要此处被截断）

    arXiv:2609.37976v1 Announce Type: cross  Abstract: LLMs trained with Chain-of-thought excel in reasoning capability, but often come with excessive token cost. We find that the core of reasoning capacity lies in the Thinking model's weight component within the null space of a projection defined by the corresponding Non-thinking model's dominant singular directions, and removing the subspace component can largely improve reasoning efficiency without hurting the accuracy gained during thinking-mode post-training. Unlike existing efforts that mostly operate within the dominant subspace, we are the first to unveil the critical role of the null space and harness it for model optimization. Motivated by this finding, we propose Spectral Null-Space Swap ($S^3$), a training-free composition of paired Non-thinking and Thinking checkpoints. Our method keeps the Non-thinking model inside its own dominant subspace and takes the Thinking checkpoint outside it, improving reasoning efficiency while mai
    
[^20]: 基于轨迹感知训练的掩码扩散语言模型研究

    On Trajectory-Aware Training for Masked Diffusion Language Models

    [https://arxiv.org/abs/2609.37974](https://arxiv.org/abs/2609.37974)

    该论文提出PUMBA统一框架，通过在策略诱导轨迹的连续步骤上训练去噪器、跨步骤传递信息并利用时间反向传播进行联合优化，实现了掩码扩散语言模型的轨迹感知训练，解决了训练与推理条件不匹配的问题。

    

    掩码扩散模型（MDMs）通过在每一步中解掩码若干个token来生成文本，但其训练与采样是在不同的条件下进行的。模型在随机掩码的序列上进行训练，而推理时则遵循由模型自身预测所塑造的轨迹。此外，每一步都无法获知前一步所计算的内容。近期的方法从不同角度分别缓解了这些局限，但这些设计选择之间如何相互作用仍是一个悬而未决的问题。我们提出了PUMBA，这是一个用于轨迹感知训练的统一框架，它在策略诱导轨迹的连续步骤上训练去噪器，在步骤之间传递信息，并通过时间反向传播（BPTT）对二者进行联合优化。对该设计空间的受控研究表明：i）完全的训练-推理对齐会因局部过拟合而失败，而较宽松的对齐仍能使训练掩码更接近推理时所见的掩码；ii）传递连续信息的效果优于离散……（原文摘要在此处截断）

    arXiv:2609.37974v1 Announce Type: cross  Abstract: Masked diffusion models (MDMs) generate text by unmasking several tokens per step, but they are trained and sampled under different conditions. The model is trained on randomly masked sequences, whereas inference follows a trajectory shaped by the model's own predictions. Additionally, each step has no access to what the previous one computed. Recent methods narrow these limitations from separate angles, leaving open how these choices interact. We introduce PUMBA, a unified framework for trajectory-aware training that trains the denoiser on consecutive steps of policy-induced trajectories, passes information between steps, and optimizes them jointly by backpropagation through time. A controlled study of this design space shows that i) exact train--inference alignment fails due to local overfitting, whereas a looser alignment still brings training masks closer to those seen at inference; ii) passing continuous information outperforms di
    
[^21]: SelfSearch：面向自我改进智能体的免奖励搜索

    SelfSearch: Reward-Free Search for Self-Improving Agents

    [https://arxiv.org/abs/2609.37968](https://arxiv.org/abs/2609.37968)

    SelfSearch提出了一种免奖励的自我改进搜索方法，智能体通过利用以往自我修改回合的记录（包含推理、工具操作和结果）来改进自身，无需昂贵的下游评估即可在多个模型-基准测试设置中显著提升成功率。

    

    arXiv:2609.37968v1 公告类型： new 摘要：大语言模型智能体编程能力的进步使其能够检查和修改自身的指令、工具和执行程序。现有方法利用这种能力，通过反复的下游评估来搜索性能更优的智能体，这不仅带来高昂成本，还将搜索过程与被评估的任务绑定在一起。我们提出了SelfSearch，这是一种免奖励的搜索程序，智能体利用以往自我改进回合的记录来修改自身。这些记录捕捉了先前修改尝试中的推理过程、工具操作和结果，为同时改进任务解决能力和自我修改能力提供了具体的经验。在搜索过程中没有下游奖励信号的情况下，SelfSearch在全部六个模型-基准测试设置中均提升了相对于初始智能体的种群平均成功率，其中单个智能体在Terminal-Bench 2.1上最多提升了11.2个百分点。在SWE-bench Multilingual上，一个智能体的成功率提升了\

    arXiv:2609.37968v1 Announce Type: new  Abstract: Advances in the coding capabilities of LLM agents allow them to inspect and modify their own instructions, tools, and execution procedures. Existing approaches use this ability to search for improved agents through repeated downstream evaluation, which incurs substantial costs and ties the search to the evaluated tasks. We introduce \textbf{SelfSearch}, a reward-free search procedure in which agents modify themselves using records of previous self-improvement episodes. These records capture the reasoning, tool actions, and outcomes of earlier modification attempts, providing concrete experience for improving both task solving and self-modification. Without downstream reward signals during search, SelfSearch improves population-mean success over the initial agent in all six model--benchmark settings, with individual agents gaining up to 11.2 percentage points on Terminal-Bench 2.1. On SWE-bench Multilingual, an agent improves success by \
    
[^22]: 学习记住什么：长时程反事实记忆优化

    Learning What to Remember: Long-horizon Counterfactual Memory Optimization

    [https://arxiv.org/abs/2609.37930](https://arxiv.org/abs/2609.37930)

    提出记忆增益策略优化（MGPO），通过反事实地归因每次记忆重写对当前与未来下游效用的边际贡献，将延迟的记忆效用转化为直接学习信号，在文档级信息抽取中提升性能的同时将平均记忆长度减少近80%。

    

    持久化的文本记忆使语言模型能够在长程交互中携带信息，但学习记住什么本质上是一个信用分配问题。一次记忆重写可能只有在很多步骤之后才变得有用，而观察到的效用中很大一部分可能继承自重写之前已经存储的信息。我们提出了记忆增益策略优化（MGPO），它通过将每次记忆重写对其当前和未来下游效用的边际贡献进行归因，来隔离出每次记忆重写的增量价值。这将延迟的记忆效用转化为直接的学习信号，用于优化哪些信息应当被持久保留。我们在文档级信息抽取任务上研究 MGPO，在该任务中结构化监督使单次记忆更新的效果可以直接测量。MGPO 在提升抽取性能的同时，相对于优化前的初始记忆策略，将平均记忆长度减少了近 80%。

    arXiv:2609.37930v1 Announce Type: new  Abstract: Persistent textual memory allows language models to carry information across long interactions, but learning what to remember is fundamentally a credit-assignment problem. A memory rewrite may only become useful many steps later, while much of the observed utility may be inherited from information already stored before the rewrite. We introduce Memory Gain Policy Optimization (MGPO), which isolates the incremental value of each memory rewrite by crediting it for its marginal contribution to current and future downstream utility. This turns delayed memory utility into a direct learning signal for optimizing what information should persist. We study MGPO on document-level information extraction, where structured supervision makes the effects of individual memory updates directly measurable. MGPO improves extraction while reducing average memory length by nearly 80% relative to the initial memory policy before optimization. The learned memo
    
[^23]: 时间锚定扩散语言模型：面向快速生成的潜在空间缓存

    Time-Anchored Diffusion Language Models: Latent-Space Caching for Fast Generation

    [https://arxiv.org/abs/2609.37924](https://arxiv.org/abs/2609.37924)

    提出基于时间的自监督锚定机制，通过两阶段架构缓存并复用潜在锚点状态，使扩散语言模型无需监督式标记目标即可实现快速生成。

    

    近期关于锚定扩散语言模型的研究通过利用监督式的重要标记目标来塑造中间潜在空间，从而改进去噪效果。在本工作中，我们引入了基于时间的（自监督）锚定方法，它无需此类目标即可学习并复用潜在锚点。我们的关键观察是：锚点编码了干净序列的持久属性，例如其语义意图、全局结构或中间规划。尽管随着标记画布的演变，其隐藏表示会变得陈旧，但其语义内容在相邻的扩散时间步之间仍然有用。该方法通过两阶段架构实现：由一个相对昂贵的锚定网络生成潜在缓存状态，再由一个轻量级去噪网络利用融合模块在每个反向步骤中智能地将缓存的潜在状态与当前状态相结合。这为锚定机制提供了一种潜在空间缓存的解释。

    arXiv:2609.37924v1 Announce Type: cross  Abstract: Recent work on anchored diffusion language models improves denoising by shaping an intermediate latent space with supervised important-token targets. In this work, we introduce time-based (self-supervised) anchoring, which learns and reuses latent anchors without requiring such targets. Our key observation is that anchors encode persistent properties of the clean sequence, such as its semantic intent, global structure, or intermediate plan. Although their hidden representations become stale as the token canvas evolves, their semantic content remains useful across nearby diffusion times. This is implemented through a two-stage architecture consisting of a relatively expensive anchor network that generates the latent cache state and a lightweight denoising network that intelligently combines the cached latent state with the current state at each reverse step using a fusion module. This gives anchoring a latent-space caching interpretatio
    
[^24]: 克服大语言模型推理中在线策略自蒸馏的规模限制

    Overcoming Scaling Limits in On-Policy Self-Distillation for LLM Reasoning

    [https://arxiv.org/abs/2609.37915](https://arxiv.org/abs/2609.37915)

    该论文通过因子分析发现在线策略自蒸馏中支架（轨迹）正确性比特权上下文正确性对下游性能影响更大，据此提出OASIS方法，通过主要监督经过验证的在线策略轨迹来克服自蒸馏的规模限制。

    

    在线策略自蒸馏（OPSD）训练学生模型在其自身采样的轨迹上匹配一个具有特权信息的教师分布。标准OPSD将这种监督应用于未经验证的学生rollout，同时以特权上下文（通常是参考解答）来条件化教师模型。我们在因子分析中将这两个角色分离，发现支架（scaffold）的正确性比上下文的正确性对下游准确率有更强的影响。未经验证的支架会造成模仿差距，因为教师可以使用学生无法获得的信息。这种差距随模型规模的增大而缩小，但OPSD仍然主要监督未经验证的轨迹。相比之下，即使教师以学生自身失败的rollout为条件，经过验证的支架仍然保持有效。基于这一发现，我们提出了OASIS，它保留了OPSD的目标，但主要监督经过标签验证的在线策略轨迹，并用其替换书面解答（原文在此处截断）

    arXiv:2609.37915v1 Announce Type: cross  Abstract: On-policy self-distillation (OPSD) trains a student to match a privileged teacher distribution along its own sampled trajectory. Standard OPSD applies this supervision to unverified student rollouts while conditioning the teacher on privileged context, typically a reference solution. We separate these roles in a factorial analysis and find that scaffold correctness has a stronger effect on downstream accuracy than context correctness. Unverified scaffolds create an imitation gap because the teacher can use information unavailable to the student. This gap shrinks with model scale, yet OPSD continues to supervise mostly unverified trajectories. In contrast, verified scaffolds remain effective even when the teacher is conditioned on the student's own unsuccessful rollout. Based on this finding, we introduce OASIS, which retains the OPSD objective but supervises mostly verified by label on-policy trajectories and replaces written solutions
    
[^25]: 坏建议的不平等影响：利用训练数据归因调控涌现性失对齐

    The Unequal Influence of Bad Advice: Using Training Data Attribution to Modulate Emergent Misalignment

    [https://arxiv.org/abs/2609.37914](https://arxiv.org/abs/2609.37914)

    本文首次利用训练数据归因定量估计每个有害样本对“涌现性失对齐”（EM）的贡献，发现坏样本的影响并不均等，且基于归因分数的数据过滤可显著增强或减弱模型失对齐的程度。

    

    在狭窄的失对齐任务上微调大语言模型，可能会破坏其后期训练形成的对齐性，并诱发新的失对齐行为——这种现象被称为“涌现性失对齐”。EM 与人格化表征相关联，微调可能通过放大某种有害或“邪恶”的人格来降低损失。目前尚不清楚训练数据的哪些特性驱动了这一效应：是否所有有害样本对失对齐的贡献大致相同，以及不同模型是否受到相同微调样本的同等影响。在本工作中，我们使用训练数据归因来定量估计每个有害样本对 EM 的贡献。我们通过重新训练来检验归因的质量——一个可靠的归因分数应能让我们通过基于该分数过滤数据来增强或减弱 EM。基于分数的过滤可以显著增强或减弱 EM；我们发现基于数据归因……（原文摘要在此处截断）

    arXiv:2609.37914v1 Announce Type: cross  Abstract: Fine-tuning large language models on narrow, misaligned tasks can undo their post-training alignment and induce novel misaligned behaviors -- a phenomenon known as \emph{emergent misalignment} (EM). EM has been linked to persona-like representations, where fine-tuning might reduce loss by amplifying a harmful or 'evil' persona. It remains unclear which properties of the training data drive this effect: whether all harmful examples contribute approximately equally to misalignment and whether different models are equally affected by the same fine-tuning examples. In this work, we use training data attribution to quantitatively estimate how much each harmful example contributes to EM. We benchmark the quality of the attribution via retraining -- a sound attribution score should enable us to enhance or attenuate EM by filtering data on that score. Score-based filtering can substantially enhance or attenuate EM; we find that both data-attri
    
[^26]: 一切皆在训练：面向大语言模型的全合成单阶段配方

    It's All Training: A Fully Synthetic Single-Stage Recipe for LLMs

    [https://arxiv.org/abs/2609.37891](https://arxiv.org/abs/2609.37891)

    该论文提出首个开源的全合成语料库SYNTH，仅从58,698篇维基百科文章出发，通过结构化扩增将预训练、中期训练与后训练整合为单一训练阶段。

    

    当前的预训练数据集源自网络爬取，带有网络爬取的所有固有缺陷，且其设计初衷并未考虑支持中期训练和后训练流程——例如，其中几乎不包含显式推理内容。因此，许多前沿实验室已开始基于最先进的模型开发自己的内部数据集，以扩充其预训练数据组合，例如添加推理轨迹来解决冷启动问题。尽管这类方法已被证明有效，但这些数据集均未公开，且这种所谓的合成数据对语言模型（包括小型模型）知识与技能习得的影响仍缺乏充分理解。我们提出了SYNTH——首个开源合成语料库，它源自58,698篇维基百科文章，通过对精选的百科种子进行结构化扩增，将预训练、中期训练和后训练合并为单一训练阶段。我们通过训练一系列模型来评估SYNTH：一个56M参数的微型模型、0.3B-0.6B……

    arXiv:2609.37891v1 Announce Type: new  Abstract: Current pre-training datasets are derived from web crawls, with all their issues, and were not designed to support mid- and post-training pipelines--for instance, they contain little explicit reasoning. Thus, many frontier labs have begun to develop their own internal datasets, starting from state-of-the-art models, to augment their pre-training data mix, eg, with reasoning traces to address cold-start problems. While demonstratively effective, none of these datasets are public, and the effect of this so-called synthetic data on knowledge and skill acquisition of language models, including small ones, remains poorly understood. We present SYNTH, the first open-source synthetic corpus derived from 58,698 Wikipedia articles that collapses pre-, mid-, and post-training into a single training stage via structured amplification of curated encyclopedic seeds. We evaluate SYNTH by training a suite of models: a 56M tiny model (Monad), 0.3B-0.6B 
    
[^27]: 基于无监督跨语言引导的零样本依存句法分析

    Zero-shot Dependency Parsing with Unsupervised Cross-Lingual Bootstrapping

    [https://arxiv.org/abs/2609.37883](https://arxiv.org/abs/2609.37883)

    本文提出一种跨语言无监督引导方法，通过增强预训练语言模型内部的句法知识，显著提升了低资源语言上零样本依存句法分析的性能，并通过无参数树探测测试证明了模型识别句法结构鲁棒性的提升。

    

    基于编码器架构的预训练语言模型（PLMs）在各种语言理解任务的零样本跨语言迁移中展现出令人印象深刻的能力。然而，由于其句法性质，将这一技术应用于依存句法分析仍然是一个重大挑战。为了提升模型在不同语言类型学之间的泛化能力，我们提出了一种跨语言无监督引导方法，以增强预训练语言模型内部的句法知识。实验表明，我们的方法在低资源语言的零样本句法分析性能上取得了显著提升。对这些引导模型的分析揭示了其在识别句法结构方面鲁棒性的增强，这通过无参数树探测测试中的更高得分得到了证实。

    arXiv:2609.37883v1 Announce Type: new  Abstract: Pre-trained language models (PLMs) with encoder-based architectures have shown impressive capabilities in zero-shot cross-lingual transfer for various language understanding tasks. However, applying this technique to dependency parsing remains a significant challenge due to its syntactic nature. To boost model generalizability across linguistic typologies, we propose a cross-lingual unsupervised bootstrapping method to improve syntactic knowledge within the PLM. We show that our method achieves a significant improvement in zero-shot parsing performance in low-resource languages. Analysis of these bootstrapped models uncovers increased robustness in recognizing syntactic structures, evidenced by higher scores in parameter-free tree probing tests.
    
[^28]: 一种语言需要多少标签？非洲语言文本分类的标注预算与跨语言池化

    How Many Labels Does a Language Need? Annotation Budgets and Cross-Lingual Pooling for African-Language Text Classification

    [https://arxiv.org/abs/2609.37882](https://arxiv.org/abs/2609.37882)

    该论文在28个语言-任务对上实证测算了非洲语言文本分类所需的标注预算，发现新闻主题分类约400个标签即可达到全量数据90%的性能，而情感分析需要数千个标签，且跨语言数据池化仅在标注预算很小时才有效。

    

    每个面向非洲语言的文本分类器都始于一个预算问题：需要多少标注样本？来自其他非洲语言的标签能否替代它们？我们在28个语言-任务对上对这两个问题进行了实证回答，涵盖16种语言的新闻主题分类（MasakhaNEWS）和12种语言的推文情感分析（AfriSenti），使用的是一种字符n-gram线性模型，该模型可在两个CPU核心上数秒内完成训练，无需预训练权重，也无需加速器。在25个至数千个标签预算范围内的单语学习曲线显示：主题分类在语言中位数水平下约需400个标签即可达到全量数据宏观F1的90%；而情感分析在12种语言中有11种在全量训练规模下仍在提升，需要数千个标签。将基准中其他语言的全部训练数据汇集起来，在小预算时收益巨大，在大预算时则毫无价值：在仅有25个目标语言标签时，它可带来0.20的提升……（原文摘要在此处截断）

    arXiv:2609.37882v1 Announce Type: new  Abstract: Every text classifier for an African language begins with a budgeting question: how many labelled examples are needed, and can labels from other African languages stand in for them? We answer both questions empirically for 28 language-task pairs, news topic classification in 16 languages (MasakhaNEWS) and tweet sentiment in 12 languages (AfriSenti), using a character n-gram linear model that trains in seconds on two CPU cores with no pretrained weights and no accelerator. Monolingual learning curves at budgets from 25 to several thousand labels show that topic classification reaches 90\% of its full-data macro-F1 with about 400 labels in the median language, while sentiment is still improving at the full training size in 11 of 12 languages and needs thousands of labels. Pooling the full training data of the other languages in the benchmark is worth a great deal at small budgets and nothing at large ones: at 25 target labels it adds 0.20 
    
[^29]: 竞争条件下自注意力的检索容量

    Retrieval Capacity of Self-Attention Under Competition

    [https://arxiv.org/abs/2609.37879](https://arxiv.org/abs/2609.37879)

    该论文提出一种无需重新训练的方法，通过仅保留注意力权重最高的token来估计自注意力的有效检索容量，发现相对较小的注意力选择集即可使模型损失接近全注意力基线，且所需集合大小随上下文扩展而增加、但占上下文比例下降。

    

    语言模型实际使用了其上下文中的多少个token，又是什么决定了这个数量？我们通过自注意力机制来研究这个问题。在不进行重新训练的情况下，我们仅保留在每个注意力头、每一层和每个查询中注意力权重最高的token，并保持其原始权重不变。通过改变所选集合的大小并测量负对数似然（NLL）的增加，我们估计出在给定损失容忍度范围内所需的有效注意力集合大小。相对较小的所选集合就能使NLL接近全注意力基线，尽管所需的大小因模型而异。基于注意力的选择显著优于随机选择。所选集合表现出几何结构，但仅凭几何上的可分性并不能证明模型损失得以保持。在评估相同预测目标的前提下扩展上下文会增加所需的集合大小，而该集合占上下文的比例则随之下降。

    arXiv:2609.37879v1 Announce Type: new  Abstract: How many tokens from its context does a language model actually use, and what determines that number? We study this question through self-attention. Without retraining, we retain only the tokens with the highest attention weights at each head, layer, and query, keeping their original weights unchanged. By varying the selected set size and measuring the increase in negative log-likelihood (NLL), we estimate the effective attention set size needed to stay within a chosen loss tolerance. Relatively small selected sets can keep NLL close to the full-attention baseline, although the required size varies across models. Attention-based selection substantially outperforms random selection. Selected sets exhibit geometric structure, although geometric separation alone does not establish that model loss is preserved. Extending context while evaluating the same prediction targets increases the required set size, while its fraction of context decrea
    
[^30]: 超越采样所学：面向RLVR的离策略感知跨模型轨迹交换

    Learning Beyond What You Sample: Off-Policy-Aware Cross-Model Trajectory Exchange for RLVR

    [https://arxiv.org/abs/2609.37868](https://arxiv.org/abs/2609.37868)

    提出GRAFT框架，通过用同伴模型信息丰富的轨迹替换全失败的rollout组，并借助对等计算的优势与序列级兼容性权重控制跨模型失配，在无需更强教师模型的情况下实现异构模型间的相互学习，提升RLVR训练效果。

    

    诸如GRPO等基于可验证奖励的强化学习方法依赖于成功的自我生成轨迹，但有限的rollout预算可能产生全部失败的组，从而没有任何基于奖励的策略梯度信号。虽然额外的rollout能以更高成本提高成功概率，但某个模型的rollout中缺失的成功轨迹可能早已被另一个模型发现。事实上，我们观察到异构模型往往在互补的提示上取得成功，这为无需指定更强教师模型的相互学习创造了机会。为了利用这种互补性，我们提出了GRAFT（用对等轨迹门控替换答案失败组），一个离策略感知的框架，用信息丰富的对等组替换全失败组。GRAFT同时传输成功和失败的对等响应并附带对等模型计算的优势，同时通过序列级兼容性权重来控制跨模型失配。

    arXiv:2609.37868v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) methods such as GRPO rely on successful self-generated trajectories, but finite rollout budgets can produce all-fail groups with no reward-based policy-gradient signal. While additional rollouts improve the chance of success at higher cost, successful trajectories missing from one model's rollouts may already have been discovered by another. Indeed, we observe that heterogeneous models often succeed on complementary prompts, creating opportunities for mutual learning without a designated stronger teacher. To exploit this complementarity, we propose GRAFT (Gated Replacement of Answer-Failed groups with peer Trajectories), an off-policy-aware framework that replaces all-fail groups with informative peer groups. GRAFT transfers both successful and unsuccessful peer responses with peer-computed advantages, while controlling cross-model mismatch through sequence-level compatibility weigh
    
[^31]: 重要的不是图像展示了什么：无关上下文在不提供信息的情况下动摇VLM评判器的稳定性

    It's Not What the Image Shows: Irrelevant Context Destabilises VLM Judges Without Informing Them

    [https://arxiv.org/abs/2609.37863](https://arxiv.org/abs/2609.37863)

    本文提出MIST误导性图像压力测试，揭示无关图像——无论与句义一致还是相反——都会以几乎相同的幅度（约20%）动摇VLM评判器的标签判断，且动摇幅度甚至超过删除“忽略图像”指令的影响，表明图像是作为干扰性上下文而非信息来源起作用，威胁了VLM替代人类标注者的可靠性。

    

    视觉语言模型（VLM）越来越多地被用来替代人类标注者，因此替代性测试的结果应当反映模型本身的能力，而非偶然的评估条件。我们提出了MIST（误导性图像压力测试）：包含200个英文句子，每个句子都围绕一个既可以按比喻义也可以按字面义解读的短语构建，并以三种条件之一呈现：一幅与句子含义一致的图像、一幅描绘相反含义的误导性图像，或者完全不配图像。标注指南要求标签仅依据句子本身来决定，因此任何图像都不应改变答案。我们原本预期每种图像都会将评判器的标签拉向其描绘的含义，但两种图像都没有产生这种预期效果。在十三个VLM评判器上，一致的图像改变了20.5%的标签，误导性图像改变了19.4%，对所有评判器而言两者数值接近，且都高于在保留图像的情况下仅删除“忽略图像”指令所产生的11.6%变化率。然而，在两种图像条件下标签不同的案例中，只有37%……（摘要至此截断）

    arXiv:2609.37863v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are increasingly used in place of human annotators, making it important that substitutability tests reflect the model rather than incidental evaluation conditions. We introduce MIST, the Misleading-Image Stress Test: 200 English sentences, each built around a phrase readable either figuratively or literally and shown with an aligned image depicting its reading, a misleading image depicting the opposite, or no image at all. The guidelines require the label to be decided from the sentence alone, so no image should change any answer. We expected each image to pull a judge's labels toward the sense it depicts, and neither kind did. Across thirteen VLM judges, an aligned image changed 20.5% of labels and a misleading one 19.4%, close for every judge and both above the 11.6% produced by deleting the ignore-the-image instruction with the image left in place. Yet only 37% of the labels that differ between the two 
    
[^32]: 单一阈值并不适用于所有语言：面向可靠且高效的低资源文本分类的语言条件化延迟机制

    One Threshold Does Not Fit All Languages: Language-Conditional Deferral for Reliable and Efficient Low-Resource Text Classification

    [https://arxiv.org/abs/2609.37861](https://arxiv.org/abs/2609.37861)

    该论文揭示了基于跨语言汇总数据的单一置信度阈值无法在所有语言上兑现错误率承诺（如索马里语覆盖率仅77.5%），并提出按语言条件化的延迟机制，使低资源多语言文本分类在保证可靠性的同时更加高效。

    

    在全球南方——即非洲、亚洲和拉丁美洲的低收入国家，世界上大多数语言在此被使用——部署的文本分类器通常运行在普通CPU上，用单一模型服务多种语言，每种语言的标注样本都很少，并且依赖人工来纠正其错误。这样的系统只有在能够承诺出错频率时才有用：它自行分配的标签中最多只能有固定比例是错误的，其余的都必须交给人工处理。分割共形预测通过单一的置信度阈值来实现这一承诺，该阈值通常在跨语言汇总的验证数据上进行估计。我们追问这一承诺是否能惠及每一种语言，答案是并不能。在MasakhaNEWS（16种非洲语言）和AfriSenti（12种语言外加两种训练中从未见过的语言）数据集上，汇总阈值平均满足90%的目标，但对索马里语的覆盖率仅为77.5%，对提格雷尼亚语为83.7%，而对两种未见语言……（原文摘要在此处截断）

    arXiv:2609.37861v1 Announce Type: new  Abstract: In the Global South, the lower-income countries of Africa, Asia, and Latin America where most of the world's languages are spoken, a deployed text classifier usually runs on ordinary CPUs, serves many languages with a single model, has few labeled examples in any of them, and relies on people to catch its mistakes. Such a system is only useful if it can promise how often it will be wrong: at most a fixed fraction of the labels it assigns on its own may be incorrect, and everything else must go to a person. Split conformal prediction delivers this promise through a single confidence threshold, normally estimated on validation data pooled across languages. We ask whether the promise reaches every language, and it does not. On MasakhaNEWS (16 African languages) and AfriSenti (12 languages plus two never seen in training), a pooled threshold meets the 90% target on average but covers Somali at 77.5%, Tigrinya at 83.7%, and the two unseen lan
    
[^33]: 存储并非策略：面向大语言模型遗忘的状态条件支撑集控制

    Storage Is Not Strategy: State-Conditioned Support Control for LLM Unlearning

    [https://arxiv.org/abs/2609.37858](https://arxiv.org/abs/2609.37858)

    该论文发现“存储目标知识”的参数未必是执行遗忘的最佳干预对象，提出基于实际遗忘更新预测效果的干预分数以及动态干预重排方法（DIR-R），在优化过程中按需自适应调整干预参数子集，从而显著提升大语言模型遗忘的效果。

    

    许多局部化的大语言模型（LLM）遗忘方法从定位信号中选出一小部分参数子集，并在优化过程中将其固定不变。然而，与目标知识关联最强的参数并不一定是最佳的更新对象，且候选干预的价值会随优化进程而变化。在一个受控实验中，存储定位分数达到了0.981的受试者工作特征曲线下面积（AUROC），但存储身份仅在17/36个目标上与更优的干预选择一致，而低秩适应（LoRA）则在35/36个目标上胜出。我们提出干预分数，该分数根据实际遗忘更新的预测效果对可编辑参数组进行排序，同时将附带损害纳入考量，并以此构建静态干预价值基线。随后我们进一步提出选择性动态干预重排（DIR-R），仅当经过校准的探针证明有必要时，才对该参数子集进行重新审视和调整。

    arXiv:2609.37858v1 Announce Type: cross  Abstract: Many localized large language model (LLM) unlearning methods select a small parameter subset from a localization signal and keep it fixed during optimization. The parameters most associated with a target, however, need not be the best ones to update, and candidate interventions can change value as optimization proceeds. In a controlled experiment, a storage-localization score reaches an area under the receiver operating characteristic curve (AUROC) of 0.981, yet storage identity agrees with the better intervention on only 17/36 targets, while low-rank adaptation (LoRA) wins 35/36. We introduce Intervention Score, which ranks editable groups by the predicted effect of the actual unlearning update while accounting for collateral damage, and use it to form the static intervention-value baseline (Static-IV). We then introduce selective dynamic intervention re-ranking (DIR-R), which revisits that subset only when a calibrated probe justifie
    
[^34]: AnthroDial：自主社交交互中大语言模型拟人化能力的基准测试

    AnthroDial: Benchmarking LLM Anthropomorphism in Autonomous Social Interaction

    [https://arxiv.org/abs/2609.37853](https://arxiv.org/abs/2609.37853)

    AnthroDial 提出了一个统一框架，通过 MindFlow 自主交互机制、CAPS-Eval 三维度评估体系和 SEEDS+DiAPO 可扩展训练范式，实现、评估并提升大语言模型在持续开放社交交互中的拟人化能力。

    

    大语言模型（LLM）越来越多地被部署为社交智能体，但可信的类人交互不仅需要流畅的回复或人设一致性。智能体必须自主决定是否、何时以及如何进行交流，同时适应不断演变的情境、目标和关系。然而，现有研究缺乏一种统一的方法来在持续、开放的交互中实现、评估和提升这类能力。我们提出了 AnthroDial，这是一个从三个互补方面开发拟人化社交智能体的统一框架：MindFlow，一个轻量级交互工具，通过动态思维缓冲区实现自主、异步和自适应的交流；CAPS-Eval，一个基于理论的评估框架，用于评估拟人化交互在认知、情感和行为三个维度上的表现；以及一个可扩展的训练范式，结合了用于环境扩展的 SEEDS 和用于自适应训练的 DiAPO。

    arXiv:2609.37853v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as social agents, yet credible human-like interaction requires more than fluent responses or persona consistency. Agents must autonomously decide whether, when, and how to communicate while adapting to evolving contexts, goals, and relationships. Existing research, however, lacks a unified approach to enabling, evaluating, and improving such capabilities in continuous, open-ended interaction. We introduce AnthroDial, a unified framework for developing anthropomorphic social agents from three complementary aspects: MindFlow, a lightweight interaction harness that enables autonomous, asynchronous, and adaptive communication through a dynamic Mind Buffer; CAPS-Eval, a theory-grounded framework for evaluating cognitive, affective, and behavioral dimensions of anthropomorphic interaction; and a scalable training paradigm that combines SEEDS for environment expansion with DiAPO for adaptiv
    
[^35]: 视觉-语言模型在面对隐性风险时还能保持有用性吗？用于高效安全-有用性对齐的意图-特权OPSD方法

    Can Vision-Language Models Stay Helpful When Facing Implicit Risks? Intent-Privilege OPSD for Efficient Safety-Helpfulness Alignment

    [https://arxiv.org/abs/2609.37837](https://arxiv.org/abs/2609.37837)

    提出意图-特权策略内自蒸馏（OPSD）方法，在训练时利用基于证据的意图作为特权监督，使视觉-语言模型仅用1,447个安全示例（比标准偏好数据集少95%）就能识别跨模态隐性风险，在不一概拒绝的前提下实现安全与有用性的高效对齐。

    

    视觉-语言模型（VLMs）仍然容易受到跨模态隐性风险的威胁：单独看起来无害的视觉和文本输入可能共同引发不安全的响应。现有的安全方法通常需要大规模的偏好数据集、昂贵的多次采样训练，或在推理时增加额外的安全保障模块。此外，这些方法还可能因直接拒绝那些本可以安全回答的请求而牺牲有用性。在本文中，我们提出了意图-特权策略内自蒸馏，它在训练期间利用基于证据的意图作为特权监督，帮助VLMs识别隐性风险，并提供安全且有用的响应，而不是一概拒绝。OPSD对每个提示仅需单次采样，即可将教师模型基于意图条件化的响应偏好蒸馏到学生模型中；学生模型随后在推理时无需意图标注或额外的安全模块即可给出响应。仅使用1,447个安全相关示例——比标准偏好数据集少95%——即可……

    arXiv:2609.37837v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) remain vulnerable to cross-modal implicit risks: visual and textual inputs that appear benign in isolation can jointly elicit unsafe responses. Existing safety methods often require large preference datasets, costly multi-rollout training, or additional safeguards at inference time. They may also sacrifice helpfulness by directly refusing requests that could be answered safely. In this paper, we propose Intent-Privilege On-Policy Self-Distillation (OPSD), which leverages evidence-grounded intent as privileged supervision during training to help VLMs recognize implicit risks and provide safe, useful responses instead of blanket refusals. OPSD distills a teacher's intent-conditioned preferences over responses into a student using a single rollout per prompt; the student then responds without intent annotations or an additional safety module. With only 1,447 safety-specific examples - 95% fewer than standard pr
    
[^36]: 可缓存的决策模型能否遵循规则？

    Can a Cacheable Decision Model Follow Rules?

    [https://arxiv.org/abs/2609.37832](https://arxiv.org/abs/2609.37832)

    本研究将决策模型从联合评分改为可缓存的独立编码后规则敏感性大幅丧失（recall@1 从 1.00 跌至 0.24），但通过针对性的反事实监督训练可以恢复较强的规则遵循能力，同时保留约5倍的缓存加速优势。

    

    Certo 是一个小型非生成式决策模型（基于 Qwen3-4B）：它根据候选动作的文本进行打分并返回一个概率，而不是生成答案。精确的设计是将状态、规则和每个候选一起读取（联合评分器），因此成本随候选菜单规模增长。独立编码则允许每个候选只编码一次并在不同状态间复用（在77个候选时约便宜5倍），但这将状态与候选分离开来。我们探究在这种转变下有多少规则敏感性得以保留，以及它是否可以通过训练恢复。在 Certo 上进行了四个实验：(1) 被测试的可缓存评分转换丢失了规则敏感性（recall@1 从 1.00 降至 0.24），而联合评分器保持 1.00，且候选短名单+重排序的补救方法失败；(2) 有针对性的反事实监督在留出的合成规则任务（改写、反事实、组合；跨随机种子可复现）上恢复了强劲性能，尽管我们尚未分离出预测是否依赖于……

    arXiv:2609.37832v1 Announce Type: new  Abstract: Certo is a small non-generative decision model (Qwen3-4B): it scores candidate actions from their text and returns a probability, instead of generating an answer. The accurate design reads the state, the rules, and each candidate together (a joint scorer), so cost grows with the menu. Independent encoding lets each candidate be encoded once and reused across states (about 5x cheaper at 77 candidates), but separates state from candidate. We ask how much rule-sensitivity survives that move, and whether it can be trained back.   Four experiments on Certo: (1) the tested conversion to cacheable scoring loses rule-sensitivity (recall@1 1.00 -> 0.24) while the joint scorer holds 1.00, and a shortlist+rerank rescue fails; (2) targeted counterfactual supervision restores strong performance on held-out synthetic rule tasks (paraphrase, counterfactual, composition; reproducible across seeds), though we do not isolate whether predictions depend on 
    
[^37]: Transformer 残差流中推理的几何学

    The Geometry of Inference in Transformer Residual Streams

    [https://arxiv.org/abs/2609.37824](https://arxiv.org/abs/2609.37824)

    本研究通过比较中间残差状态与最终状态及其经验库，揭示了Transformer语言模型中预测表示主要通过渐进的方向对齐变化（而非欧氏距离的显著改变）逐渐变得对最终结果具有特异性，并建立了区分范数、对齐与终点几何作用的高维理论模型。

    

    Transformer 语言模型通过一系列连续的残差更新来构建预测，但其表示如何逐渐变得对最终结果具有特异性仍不清楚。我们通过将中间残差状态与其自身的最终状态以及来自其他上下文的最终状态经验库进行比较来研究这一过程。在六个预训练语言模型中，自身的终点在早期就变得优于平均替代方案，而许多个体终点仍然更接近。这些相互竞争的集合通常随深度增加而缩小，但其成员会发生变化，且存活的终点彼此之间并不一定变得更相似。因此，方向对齐和终点排名可以改善，而与最终状态的欧几里得距离却变化很小。我们开发了一个简单的高维模型，分离了范数、对齐和终点几何各自的作用，展示了渐进的方向变化如何能够产生竞争的急剧减少。

    arXiv:2609.37824v1 Announce Type: new  Abstract: Transformer language models build predictions through successive residual updates, but how their representations become specific to an eventual outcome remains unclear. We study this process by comparing intermediate residual states with their own final states and an empirical bank of final states from other contexts. Across six pretrained language models, the own endpoint becomes preferable to the average alternative early, while many individual endpoints remain closer. These competing sets generally shrink with depth, but their membership changes and their surviving endpoints need not become more similar to one another. Directional alignment and endpoint rank can therefore improve while Euclidean distance to the final state changes little. We develop a simple high-dimensional model that separates the roles of norm, alignment, and endpoint geometry, showing how gradual directional changes can produce sharp reductions in competition. We 
    
[^38]: 深度思考，直接表达：面向副语言特征接地的口语对话的循环潜在推理

    Thinking in Depth, Speaking Directly: Recurrent Latent Reasoning for Paralinguistically Grounded Spoken Dialogue

    [https://arxiv.org/abs/2609.37818](https://arxiv.org/abs/2609.37818)

    LoopSLM通过循环Transformer进行潜在推理，在无需显式思维链的情况下实现副语言接地的共情口语对话，有效缩小了感知与推理之间的鸿沟并降低了推理延迟。

    

    共情的口语对话要求模型同时利用“说了什么”和“怎么说”来决定如何回应。显式思维链（CoT）可以改善副语言感知，并使声学线索在回复中更加明确，但并不能保证其在响应规划中被有效利用。我们将这种不匹配称为“感知-推理鸿沟”。此外，CoT可能无法以文字形式完全捕捉声学线索，且生成CoT会增加推理延迟。为了解决这些局限，我们提出了LoopSLM，它基于循环Transformer进行潜在推理，复用解码器模块在每一次迭代中以声学接地的方式精炼隐藏状态。其两阶段训练通过将“学习推理”与“学习回应”相分离，进一步缩小了感知-推理鸿沟，从而实现无需CoT的直接推理。在EchoMind基准上，LoopSLM在副语言理解、推理和回复质量方面均优于Qwen2.5-Omni-7B。相比CoT-SFT基线，LoopSLM取得了提升。

    arXiv:2609.37818v1 Announce Type: cross  Abstract: Empathetic spoken dialogue requires models to use both what is said and how it is said to decide how to respond. Explicit CoT can improve paralinguistic perception and make acoustic cues more explicit in replies, yet does not ensure their effective use in response planning. We call this mismatch the perception-reasoning gap. In addition, CoT may not fully capture acoustic cues in words, and generating it adds inference latency. To address these limitations, we introduce LoopSLM, which builds on looped Transformers for latent reasoning, reusing a decoder block to refine hidden states with acoustic grounding at every pass. Its two-stage training further narrows the perception-reasoning gap by separating learning to reason from learning to respond, enabling direct inference without CoT. On EchoMind, LoopSLM improves paralinguistic understanding, reasoning, and reply quality over Qwen2.5-Omni-7B. Against the CoT-SFT baseline, LoopSLM gains
    
[^39]: CompOrca：语料库规模的指令微调数据合规性标注

    CompOrca: Corpus-Scale Compliance Labelling of Instruction-Tuning Data

    [https://arxiv.org/abs/2609.37807](https://arxiv.org/abs/2609.37807)

    该论文提出了 CompOrca，利用开源大模型评判器对整个 OpenOrca 语料库（超过 420 万条样本）进行五次独立判定，首次实现了语料库规模的合规性标注，并发布带投票计数的标注结果，以支持对拒答与不服从行为的研究。

    

    研究微调如何塑造拒答与不服从行为，需要识别出那些拒绝、规避或以其他方式未能完成所请求任务的训练样本。但现有的标注最多只覆盖几千条提示词的评估集。我们提出了 CompOrca，对包含 4,233,923 条样本的 OpenOrca 语料库整体进行了合规性标注。每个样本都由一个开源权重的大语言模型评判器（LongCat-2.0，1.6 万亿参数）经过五次独立判定，被分类为合规或不合规，并将语料库发布为一致合规（94.75%）、一致不合规（1.28%）和非一致行（3.97%）三个部分，同时附带原始投票计数。单次判定会将语料库中 2.7-3.2% 的样本标记为不合规，而只有 1.28% 被全部五次判定标记，这使得过滤最模糊的样本成为可能。在 450 个经人工标注的样本（其中 150 个被标注了两次，人与人之间 κ = 0.93）上，一致合规与不合规……（原文摘要至此截断）

    arXiv:2609.37807v1 Announce Type: new  Abstract: Studying how fine-tuning shapes refusal and noncompliance behaviour requires identifying training examples that refuse, evade or otherwise fail to fulfil the requested task. But existing annotation covers evaluation sets of a few thousand prompts at most. We present CompOrca, a compliance labelling over the entirety of the 4,233,923-example OpenOrca corpus. Every example was classified as compliant or noncompliant by five independent passes of an open-weight LLM judge (LongCat-2.0, 1.6T parameters), and the corpus is released as unanimous compliance (94.75%), unanimous noncompliance (1.28%), and nonunanimous rows (3.97%) along with the raw vote counts. A single pass flags 2.7-3.2% of the corpus as noncompliant, while only 1.28% is flagged by all five, allowing for filtering the most ambiguous samples. Against 450 human-annotated examples, 150 of them annotated twice (human-human $\kappa = 0.93$), the unanimous compliance and noncomplianc
    
[^40]: 一种用于评估大语言模型回答中所表达的临床推理的评分量规（提议稿）

    A Proposed Rubric for Evaluating Expressed Clinical Reasoning in Large Language Model Responses

    [https://arxiv.org/abs/2609.37788](https://arxiv.org/abs/2609.37788)

    该论文提出一个融合医学教育评估框架、临床大语言模型基准和通用LLM推理评估研究的多维评分量规，用于对大语言模型针对金标准临床案例的自由文本回答中表达的临床推理进行结构化评估。

    

    评分量规为语言模型的结构化评估提供支持。我们提出了一种用于评估模型回答中所表达的临床推理的量规，该量规借鉴了三个方面的工作：医学教育评估框架（ART、SCT、关键特征问题以及OSCE）；临床大语言模型基准（MedR-Bench、HealthBench、TIMER-Bench、DR.BENCH、PrIME-LLM 和 PatientSafeBench）；以及通用大语言模型推理评估研究，包括事实性-有效性-连贯性-有用性（Factuality-Validity-Coherence-Utility）分类法、FaithCoT-Bench 和 C2-Faith。我们使用“有据性”作为对该分类法中事实性类别的面向临床的改编。该量规将这些概念整合到一个多维框架中，用于对针对金标准临床案例片段的自由文本回答进行评分。它包括初步的行为锚定标准、适用性规则，以及一个用于标记案例特定安全关键错误的独立标记。通用领域的框架为其设计提供了参考，但并未被视为经过验证的临……（原文摘要在此处截断）

    arXiv:2609.37788v1 Announce Type: cross  Abstract: Rubrics support the structured evaluation of language models. We propose a rubric for assessing expressed clinical reasoning in model responses, drawing on three bodies of work: medical education assessment frameworks (ART, SCT, Key Feature Problems and OSCE); clinical LLM benchmarks (MedR-Bench, HealthBench, TIMER-Bench, DR.BENCH, PrIME-LLM and PatientSafeBench); and general LLM reasoning evaluation research, including the Factuality-Validity-Coherence-Utility taxonomy, FaithCoT-Bench and C2-Faith. We use groundedness as a clinically oriented adaptation of the taxonomy's factuality category. The rubric brings these concepts together in a multidimensional framework for scoring free-text responses to gold-standard clinical vignettes. It includes provisional behavioural anchors, applicability rules and a separate flag for case-specific safety-critical errors. General-domain frameworks inform its design but are not treated as validated cl
    
[^41]: 选择重要的内容：语义压缩引导的长上下文嵌入选择性池化

    Selecting What Matters: Semantic Compression-Guided Selective Pooling for Long-Context Embeddings

    [https://arxiv.org/abs/2609.37782](https://arxiv.org/abs/2609.37782)

    提出免训练框架SCSP，通过为句子感知分块附加语义压缩提示并利用其注意力模式估计词元重要性，实现长上下文嵌入中的选择性池化，避免平均池化稀释关键语义信息。

    

    大型语言模型（LLM）作为免训练的文本编码器，在长上下文嵌入方面展现出强大潜力。现有方法主要致力于改进因果注意力下的信息流，并且通常通过对所有词元表示进行均匀平均来构建嵌入。然而，对于长文档而言，这种平均池化会使显著的语义信息被大量冗余或弱信息内容稀释。为此，我们提出了SCSP，一个利用语义压缩在长上下文嵌入中选择信息丰富词元的免训练框架。具体而言，SCSP首先将文档划分为句子感知的分块，并在每个分块后附加语义压缩提示。提示隔离注意力掩码在保持文档词元之间信息流的同时，将每个提示限制在其对应的局部上下文中。随后，我们利用这些提示所引发的注意力模式来估计词元的重要性，并选择……

    arXiv:2609.37782v1 Announce Type: new  Abstract: Large language models (LLMs) have shown strong potential as training-free text encoders for long-context embeddings. Existing approaches primarily improve information flow under causal attention and typically construct embeddings by uniformly averaging all token representations. However, for long documents, such mean pooling can dilute salient semantic information with abundant redundant or weakly informative content. To this end, we propose SCSP, a training-free framework that leverages semantic compression for informative token selection in long-context embedding. Specifically, SCSP first partitions a document into sentence-aware chunks and appends a semantic compression prompt to each chunk. A prompt-isolated attention mask preserves information flow among document tokens while restricting each prompt to its corresponding local context. We then use the attention patterns elicited by these prompts to estimate token importance, select i
    
[^42]: 哪种纸草手写文本识别（HTR）才够好？四种纸草学任务对希腊语文本字符错误率的容忍度研究

    Which papyrus HTR is good enough? Character-error-rate tolerance of four papyrological tasks on Greek texts

    [https://arxiv.org/abs/2609.37755](https://arxiv.org/abs/2609.37755)

    该研究通过将模拟的不同字符错误率（1%—50%）的HTR输出用于文献类型识别、断代等四种纸草学任务，首次为古希腊纸草文献手写文本识别系统所需的精度建立了基准。

    

    目的：大多数希腊纸草文献仍未出版和数字化；一个能够自动转录这些文献的手写文本识别（HTR）流程将使学者能够发现迄今尚未被阅读过的文书和文学作品。古希腊纸草文献的识别系统尚处于起步阶段，而它们对于特定纸草学任务需要达到何种精度尚未得到检验。为了回答这一问题并为希腊纸草HTR建立基准，我们以已发表的校勘本作为真值，针对四种纸草学任务测试了一系列字符错误率（CER）。方法：从papyri.info中现有的63,846个希腊语纸草文本校勘本出发，我们通过去除编辑层来模拟仅含正文的“完美HTR”输出，然后使用种子算法将其降级为精确的1%—50%字符错误率，并加入缺失行和四种错误形态变体。在这些数据上，我们训练了小型模型（TF-IDF、fastText、字符CNN、ByT5-small），用于文献类型识别、断代和文献性质……（摘要至此被截断）

    arXiv:2609.37755v1 Announce Type: new  Abstract: Purpose: Most Greek papyri remain unpublished and undigitised; a handwritten text recognition (HTR) pipeline that transcribes them automatically would let scholars discover documents and literary works that have so far gone unread. Recognition systems for Ancient Greek papyri are in statu nascendi, and how accurate they must be for a given papyrological task has not been examined. To answer this and set a benchmark for Greek papyrus HTR, we test a range of character error rates (CER) against four papyrological tasks, using published editions as ground truth. Methods: From 63,846 current editions of Greek texts in papyri.info, we imitate a letters-only "perfect HTR" output by removing the editorial layer, then degrade it with a seeded algorithm to exact CERs of 1 - 50%, with lost lines and four error-shape variants. On these data we train small models (TF-IDF, fastText, a character CNN, ByT5-small) for document type, dating and documentar
    
[^43]: 上下文语言模型

    Context Language Models

    [https://arxiv.org/abs/2609.37725](https://arxiv.org/abs/2609.37725)

    上下文语言模型（CLM）通过将自身上下文当作可自由编辑的文件来实现模型原生管理上下文，在多种任务上以更少的计算量超越了现有最先进的上下文管理策略。

    

    我们提出了上下文语言模型，这是一类能够原生管理自身上下文的语言模型。我们通过将上下文视为一个文件，并允许模型对该文件进行不受限制的更新来实现这一目标。这使得模型能够学习在上下文中维护哪些内容最为重要，并且可以自然地扩展到多智能体系统，其中多个智能体的上下文以文件的形式共存。利用现有模型以零样本方式构建的CLM在多种任务上超越了最先进的上下文管理策略：在BrowseComp-Plus上准确率提高11.4%，同时FLOPs减少21.5%；在12小时的EdgeBench上得分提高5%，FLOPs减少59%；在24小时的多仓库智能体集群任务上，在相同计算量下改进幅度提升65%。此外，通过将上下文管理从外部框架控制转移到模型的内在行为，CLM自然地支持对上下文管理策略进行上下文内学习和参数化学习。我们展示了CLM可以……

    arXiv:2609.37725v1 Announce Type: new  Abstract: We introduce Context Language Models (CLMs), language models that natively manage their own context. We implement this by treating the context as a file and allowing the model to make unrestricted updates to this file. This allows the model to learn what is most important to maintain in context, and naturally extends to multi-agent systems where multiple agent contexts coexist as files. Building CLMs zero-shot with existing models outperforms SOTA context management strategies across a variety of tasks: 11.4% higher accuracy with 21.5% fewer FLOPs on BrowseComp-Plus, 5% higher scores with 59% fewer FLOPs on 12-hour EdgeBench, and 65% greater improvement with the same compute on a 24-hour multi-repository agent-swarm task. Moreover, by shifting context management from external harness control to intrinsic model behavior, CLMs naturally enable both in-context and parametric learning of context-management strategies. We show that CLMs can b
    
[^44]: Transformer中隐状态轨迹的预测几何

    Predictive Geometry of Hidden Trajectories in Transformers

    [https://arxiv.org/abs/2609.37717](https://arxiv.org/abs/2609.37717)

    该论文证明仅解码器Transformer在成功验证轨迹附近的逐层“损失到终点”函数的局部二阶几何由隐藏状态空间上的回拉Fisher算子支配，其谱划分出输出敏感方向与预测零方向、界定出残差流的局部可观测子空间，并由此为因果Transformer导出衡量每个词元隐藏状态对预测敏感度的逐词元曲率分数。

    

    仅解码器Transformer仅通过终端的下一词元预测损失进行训练，然而该损失通过固定的下游计算约束着每一个中间隐藏状态。我们通过研究逐层的“损失到终点”函数来形式化这一约束，即通过将候选隐藏状态继续经过剩余的Transformer块所得到的终端损失。在成功的验证轨迹附近，我们证明这些函数的局部二阶几何结构，在忽略低损失残差项的意义下，由隐藏状态空间上的一个回拉Fisher算子所支配。其谱识别出输出敏感方向与近似预测零方向，从而给出残差流的一个局部可观测子空间。对于因果Transformer，同样的几何结构诱导出一种逐词元的曲率分数：即目标logits对每个词元隐藏状态扰动的Fisher加权敏感度。该分数在因果（该部分摘要在此处被截断）

    arXiv:2609.37717v1 Announce Type: cross  Abstract: Decoder-only transformers are trained only through a terminal next-token prediction loss, yet this loss constrains every intermediate hidden state through the fixed downstream computation. We formalize this constraint by studying layerwise loss-to-go functions: the terminal loss obtained by continuing a candidate hidden state through the remaining transformer blocks. Around successful validation trajectories, we show that the local second-order geometry of these functions is governed, up to low-loss residual terms, by a pullback Fisher operator on hidden-state space. Its spectrum identifies output-sensitive directions and approximately prediction-null directions, yielding a local observable subspace of the residual stream. For causal transformers, the same geometry induces a tokenwise curvature score: a Fisher-weighted sensitivity of the target logits to perturbations of each token's hidden state. This score vanishes outside the causal
    
[^45]: Billiger.de产品：一个双语实体匹配基准数据集

    Billiger.de Products: A Bilingual Entity Matching Benchmark

    [https://arxiv.org/abs/2609.37713](https://arxiv.org/abs/2609.37713)

    本文提出了Billiger.de Products，一个涵盖十三个消费品类别的德英双语实体匹配基准数据集，填补了现有产品匹配基准以英语为主且品类单一的空白。

    

    现有的产品匹配基准数据集主要包含英语产品数据，且通常由单一产品类别（如电子产品）主导。本文介绍了Billiger.de Products，一个涵盖十三个消费产品类别的德英双语实体匹配基准数据集，其中包括服装和家具等难以处理的类别。该基准数据源自德国比价平台billiger.de。遵循WDC Products的设计，该基准提供了多个变体，它们在边界案例的比例、开发集的大小以及训练中未见实体的比例上各不相同。每个商品的对应英文翻译保持所有配对、数据划分和标签固定不变，而跨语言测试集则在单个配对内结合德语和英语记录。我们使用六个有监督匹配器和零样本GPT-5.2在两种语言版本上对该基准进行了验证。

    arXiv:2609.37713v1 Announce Type: new  Abstract: Existing product matching benchmarks primarily contain English-language product data and are often dominated by a single product category, such as electronics. This paper introduces Billiger.de Products, a bilingual German and English entity matching benchmark covering thirteen consumer product categories, including difficult-to-handle categories such as clothing and furniture. The benchmark data originates from the German price comparison platform billiger.de. Following the design of WDC Products, the benchmark offers multiple variants that differ in the fraction of corner cases, the size of the development set, and the fraction of entities unseen during training. An aligned English translation of every offer keeps all pairs, splits, and labels fixed, while cross-language test sets combine German and English records within individual pairs. We validate the benchmark using six supervised matchers and zero-shot GPT-5.2 on both language ve
    
[^46]: 读者语言熟练程度塑造逐层惊讶度分布特征

    Reader Proficiency Shapes Layer-wise Surprisal Profiles

    [https://arxiv.org/abs/2609.37688](https://arxiv.org/abs/2609.37688)

    该研究发现，大语言模型惊讶度预测能力在模型各层中的分布深度会因读者词汇熟练水平和眼动指标而异：词汇熟练度较低的读者在首次通过注视时长上呈现更深的预测深度，而总注视时长在两组读者中都呈现更深的预测深度。

    

    阅读行为不仅随语言输入而变化，还随读者的语言熟练程度而变化。在本研究中，我们探讨了大型语言模型（LLM）的惊讶度与人类眼动行为之间的逐层关系，是否会因读者熟练水平的不同以及眼动指标的不同而有所差异。利用 MECO L2 语料库的眼动追踪数据，我们比较了词汇熟练度高与低的读者在首次通过注视时长（FPGD）和总注视时长（TGD）上的表现。我们使用“预测深度”来量化惊讶度预测能力在模型各层中的分布。在对 12 个受测大语言模型的分析中，我们发现词汇熟练度较低的读者在 FPGD 上往往表现出更深的预测深度，而这种差异在 TGD 上较小。此外，在两个熟练程度组中，TGD 本身都比 FPGD 表现出更深的预测深度。这些模式表明，预测能力在大语言模型各层中的集中位置可能与（原文在此处截断）

    arXiv:2609.37688v1 Announce Type: new  Abstract: Reading behaviour varies not only with linguistic input, but also with reader proficiency. In this study, we investigate whether the layer-wise relationship between surprisal from large language models (LLMs) and human gaze behaviour differs across readers with different levels of proficiency and across gaze measures. Using eye-tracking data from the MECO L2 corpus, we compare readers with high and low vocabulary proficiency on first-pass gaze duration (FPGD) and total gaze duration (TGD). We quantify the distribution of the predictive power of surprisal across model layers using Predictive Depth. Across 12 tested LLMs, we find that readers with lower vocabulary proficiency tend to show deeper Predictive Depth for FPGD, while this difference is smaller for TGD. Also, TGD itself shows deeper Predictive Depth than FPGD in both proficiency groups. These patterns suggest that where predictive power is concentrated across LLM layers may be re
    
[^47]: EngiWorld：前沿智能体在专业工程环境中能交付什么？

    EngiWorld: What Can Frontier Agents Deliver in Professional Engineering Environments?

    [https://arxiv.org/abs/2609.37686](https://arxiv.org/abs/2609.37686)

    该论文提出了首个覆盖完整设计循环的工程智能体基准EngiWorld，涵盖1,301个专家任务、6个工程领域和26个专业软件平台，并引入以工件为中心的自动化评估方法来检验工程成果的几何有效性、物理可行性与规则合规性。

    

    自主智能体在通用计算机使用方面取得了快速进展，但对专业工业工程的可靠自动化仍然遥不可及，因为工程工作流程要求对几何和物理约束进行推理，并处理跨软件和设计阶段保留的依赖关系。我们提出了EngiWorld，这是第一个围绕完整设计循环构建的基准：包含1,301个由专家精心策划的任务，涵盖6个工程领域（CAD、CAE、CAM、BIM、EDA和3D可视化）和26个专业软件平台，同时提供GUI和CLI两种界面，以及从软件选择到开放式任务的6种任务类型。我们进一步引入了一种以工件为中心的评估方法，该方法建立在统一的领域验证器套件之上，能够以编程方式检查最终和中间工件的几何有效性、物理可行性和规则合规性，并通过规范达成度对定量设计任务进行连续评分。

    arXiv:2609.37686v1 Announce Type: new  Abstract: Autonomous agents have made rapid progress in general-purpose computer use, but reliable automation of professional industrial engineering remains out of reach, as engineering workflows demand reasoning over geometric and physical constraints and dependencies preserved across software and design stages. We present EngiWorld, the first benchmark structured around the complete design loop: 1,301 expert-curated tasks spanning 6 engineering domains (CAD, CAE, CAM, BIM, EDA, and 3D visualization) and 26 professional software platforms, with both GUI and CLI interfaces and 6 task types ranging from software-selection to open-ended tasks. We further introduce an artifact-centric evaluation methodology built on a unified domain-verifier suite, which programmatically checks the geometric validity, physical feasibility, and rule compliance of final and intermediate artifacts, and scores quantitative design tasks continuously by specification attai
    
[^48]: 当模型不操纵流形时：一个比较任务的几何学

    When Models Don't Manipulate Manifolds: The Geometry of a Comparison Task

    [https://arxiv.org/abs/2609.37680](https://arxiv.org/abs/2609.37680)

    该论文精确刻画了Qwen2.5-7B-Instruct模型在数字比较任务中的因果几何结构，发现模型主要依赖线性表示而非操纵低维流形来实现比较计算。

    

    机制可解释性研究的一个当前前提是，对神经网络表示几何的详细刻画能够揭示模型如何执行计算，以及如何对其进行有效干预。虽然文献中已观察到多个概念存在低维流形（例如，数字编码在螺旋线上、星期几编码在圆上等），其结构被认为反映了数据和任务的特性，但模型在计算中对这些流形的依赖程度，以及它们如何操纵这些流形，仍不清楚。我们精确刻画了数字比较任务（作为决策中比较行为的抽象）中计算的几何结构，以及模型如何以优雅的方式利用几何来实现这一任务。具体而言，我们研究了Qwen2.5-7B-Instruct（一个能力强大且被广泛研究的开源权重模型）中数字比较的因果几何，发现Qwen在很大程度上使用线性表示……

    arXiv:2609.37680v1 Announce Type: cross  Abstract: One of the current premises of mechanistic interpretability research is that detailed accounts of the geometry of neural network representations can tell us how models perform computations, and how to effectively intervene on them. While low dimensional manifolds have been observed for multiple concepts in the literature (e.g. numbers encoded on helices, days of the week on a circle, ...), with structure believed to reflect properties of data and tasks, the extent to which models rely on them for computation, and how they manipulate them, remains unclear. We characterize precisely the geometry of computation in a number-comparison task, as an abstraction of comparison for decision making, and how models utilize geometry in an elegant fashion to implement it. Specifically, we study the causal geometry of number comparison in Qwen2.5-7B-Instruct, a capable and widely studied open-weight model, and find Qwen largely uses linear representa
    
[^49]: KUPAS MASTER：将资深从业者的隐性专业知识提炼为可供智能体使用的经验语料库

    KUPAS MASTER: Distilling the Tacit Expertise of Master Practitioners into Agent-Ready Experience Corpora

    [https://arxiv.org/abs/2609.37673](https://arxiv.org/abs/2609.37673)

    提出了KUPAS MASTER经验工程平台，通过六要素案例结构和九层认知语料库构建方法，将资深从业者工作记录与访谈中的隐性专业知识转化为可追溯、可复用、供LLM智能体使用的结构化经验语料库。

    

    经验丰富的专业人士所掌握的不只是事实和结论，他们还知道哪些线索重要、为什么某个判断是合理的，以及应该采取什么行动。常规的工作记录往往遗漏了这些隐性知识，使得大语言模型（LLM）智能体难以有效利用专业经验。我们提出了KUPAS MASTER，一个围绕九层认知语料库构建的经验工程平台。它将异构的工作记录和从业者访谈转化为可追溯、可复用的智能体经验语料库。六个案例要素保存了任务过程：情境、线索、判断、行动、边界和结果。九层认知语料库构建沿九个抽取维度组织隐性经验，并将所得资产存储在六个库中：规则、约束、最佳实践、负面示例、边界情况和技能。语义对齐、个人经验蒸馏、组织（原文摘要在此处被截断）

    arXiv:2609.37673v1 Announce Type: new  Abstract: Experienced professionals know more than just facts and conclusions. They know which cues matter, why a judgment is reasonable, and which action to take. Routine work records often leave out this tacit knowledge, making it difficult for Large Language Model (LLM) agents to use professional experience effectively. We introduce KUPAS MASTER, an experience engineering platform built around nine-layer cognitive corpus construction. It turns heterogeneous work records and practitioner interviews into traceable, reusable experience corpora for agents. Six case elements preserve the task process: context, cues, judgment, action, boundaries, and outcomes. Nine-layer cognitive corpus construction organizes tacit experience along nine extraction dimensions and stores the resulting assets in six libraries: rules, constraints, best practices, negative examples, corner cases, and skills. Semantic alignment, individual experience distillation, organiz
    
[^50]: 基于语料库引导的双路径传播的图检索增强生成

    Corpus-Guided Dual-Path Propagation for Graph Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.37661](https://arxiv.org/abs/2609.37661)

    NexusRAG通过联合实体共现与语义相似度构建语料库级实体邻域结构，引导语义传播与结构传播两条互补路径，克服了无关系图检索中仅依赖查询-句子相似度而遗漏桥接证据的缺陷。

    

    基于图的检索增强生成通过将语料库信息组织成图来支持多跳检索。然而，现有的无关系图检索方法主要依赖查询与句子之间的相似度来搜索证据。这可能会遗漏查询相似度较低但有用的桥接证据，并激活与推理链无关的附带实体。在本文中，我们提出了一种简单而有效的方法，称为NexusRAG，它通过联合实体共现与语义相似度构建的语料库级实体邻域结构来增强无关系的Tri-Graph。NexusRAG利用该结构引导两条互补的传播路径：通过句子的邻域约束语义传播识别与查询相关的实体边界，而相邻实体之间的直接结构传播则将该边界扩展到结构上相关的实体。传播后的实体权重（摘要在此处截断）

    arXiv:2609.37661v1 Announce Type: new  Abstract: Graph-based retrieval-augmented generation supports multi-hop retrieval by organizing corpus information into graphs. However, existing relation-free graph retrieval methods rely primarily on query-sentence similarity to search for evidence. This can exclude useful bridging evidence with low query similarity and activate incidental entities unrelated to the reasoning chain. In this paper, we propose a simple and effective approach called NexusRAG, which augments the relation-free Tri-Graph with a corpus-level entity neighborhood structure derived from joint entity co-occurrence and semantic similarity. NexusRAG employs this structure to guide two complementary propagation paths: neighborhood-constrained semantic propagation through sentences identifies the query-relevant entity frontier, while direct structural propagation between neighboring entities expands that frontier to structurally related entities. The propagated entity weights a
    
[^51]: 评估与基准测试System One模型Jev

    Evaluating and Benchmarking the System One Model Jev

    [https://arxiv.org/abs/2609.37647](https://arxiv.org/abs/2609.37647)

    本文对TypeSafe AI的商业System One模型Jev进行了大规模零样本评估，在37个数据集、346,009次请求（成本不到10美元）上系统基准测试其在分类、路由、推理、审核等小型决策任务上的表现，并与Qwen3.8-27B和Gemma-4-E4B进行对比。

    

    Jev是TypeSafe AI公司推出的一款商业System One模型，它不生成文本：给定一个状态和带类型的问题，它会从固定选项中返回一个选择、在评分量规上返回一个位置，或返回一个陈述为真的概率，厂商称这些概率是经过校准的。此类模型面向信息访问流程中的小型决策任务，例如查询路由、事实依据检查、内容审核或按量规评分。我们在37个数据集上对Jev（jev-1.13.0）进行了零样本评估，涵盖分类、路由、自然语言推理、阅读理解、常识推理、内容审核、法律条款分析和量规评分等任务，每个数据集使用一个固定模板和完整的评估集：346,009次请求的花费不到10美元。作为参考，我们在相同的请求上，通过Qwen3.8-27B和Gemma-4-E4B在选项上的精确下一词概率对它们进行评分。Jev在IMDB、SST-2、HellaSwag等数据集上达到95-99%的准确率（摘要在此处被截断）。

    arXiv:2609.37647v1 Announce Type: cross  Abstract: Jev is a commercial System One model from TypeSafe AI that does not generate text: given a state and typed questions, it returns a choice from fixed options, a position on a rubric, or the probability that a statement is true, with probabilities the vendor describes as calibrated. Such models target small decisions in information access pipelines, such as routing queries, checking grounding, moderating content, or rating against a rubric. We evaluate Jev (jev-1.13.0) zero-shot on 37 datasets spanning classification, routing, natural language inference, reading comprehension, commonsense reasoning, moderation, legal clause analysis and rubric scoring, with one frozen template per dataset and full evaluation splits: 346,009 requests for under USD 10. For reference, we score Qwen3.8-27B and Gemma-4-E4B on identical requests via their exact next-token probabilities over the options. Jev reaches 95-99% accuracy on IMDB, SST-2, HellaSwag and
    
[^52]: 协同语言学：AI增强的语言学理论建构

    Co-Linguistics: AI-augmented Theory Construction in Linguistics

    [https://arxiv.org/abs/2609.37635](https://arxiv.org/abs/2609.37635)

    本文提出“协同语言学”理念，即AI可作为“协同科学家”帮助语言学家构建和评估形式化语言学理论，通过显式化现有理论、比较竞争理论、提出新理论来加速语言学研究。

    

    近年来，大型语言模型（LLMs）在语言学中被研究作为人类语言能力的潜在模型。本文讨论AI的一种完全不同的用途，即作为“协同科学家”（co-scientist），帮助构建和评估语言学理论（我们将这一研究方向称为“协同语言学”）。自20世纪60年代以来，语言学所发展的理论在原则上都是可以数学形式化的，通常采用形式语言理论或模型论的语言来表述。因此，数学领域的AI革命也将在语言学中产生影响——但有一个关键的区别：证明新定理很少是语言学家的目标。相反，语言学家寻求的是找到一组最佳的公理，以推导出经验性陈述。AI可以通过多种方式加速研究：使现有理论完全显式化、比较相互竞争的理论，以及更具雄心地提出新理论（在机器学习中，这与“程序归纳”相关）。它还将通过加速……来帮助评估理论。

    arXiv:2609.37635v1 Announce Type: new  Abstract: LLMs have been studied in recent linguistics as potential models of humans' linguistic abilities. Here we discuss an entirely different use of AI, namely as a co-scientist, to help construct and assess linguistic theories (we refer to the result as "Co-Linguistics"). Since the 1960s, linguistics has developed theories that are in principle mathematically formalizable, often in the language of formal language theory or model theory. The AI revolution in mathematics will thus have consequences in linguistics-but with an essential twist: proving new theorems is rarely the linguist's goal. Rather, one seeks to find the best set of axioms to derive empirical statements. AI could accelerate research by making existing theories fully explicit, by comparing competing theories, and more ambitiously, by proposing new theories (in machine learning, this relates to "program induction"). It will also help assess theories by accelerating the identific
    
[^53]: RLTL;DR：通过内化自生成反馈实现自我提升

    RLTL;DR: Self-improvement by Internalizing Self-generated Feedback

    [https://arxiv.org/abs/2609.37633](https://arxiv.org/abs/2609.37633)

    提出RLTL;DR方法，让策略在每次失败后根据验证器输出撰写自己的TL;DR见解，并将其作为后续尝试的上下文条件，同时通过反向传播将这些见解内化为任务到见解的直接映射，从而在无教师模型、任务难度极高（Pass@128=0）的场景下实现自我提升。

    

    带可验证奖励的强化学习（RLVR）的常见范式是让智能体对任务进行多次尝试，并朝着成功的尝试方向进行优化。这在自我提升的场景中会出现问题：任务极其困难，智能体成功概率很低甚至为零，且没有教师模型或示例解答可供蒸馏。在本文中，我们提出了 RLTL;DR。在每次失败尝试后，我们向策略展示验证器的输出，并让它以一条 TL;DR（一句话见解）的形式撰写自己的反馈。下一次 rollout 会以所有先前的见解为条件，我们依次采样 rollout，直到找到解决方案。此外，我们对上下文中的见解启用反向传播，以将“任务到见解”的直接映射内化到模型中。在具有挑战性的工具调用和代码数据集上（过滤至 Pass@128=0），对 Qwen 3.5 9B Thinking 策略进行标准 GRPO 训练的表现始终持平……

    arXiv:2609.37633v1 Announce Type: cross  Abstract: The common paradigm of reinforcement learning with verifiable rewards (RLVR) is to let agents make multiple attempts at a task, and optimize towards the successful ones. This becomes problematic in the realms of self-improvement, where tasks are so difficult that the agent has a low or even no chance of success, and where there are no teacher models or example solutions to distill from. In this paper, we introduce RLTL;DR. After each failed attempt, we show the policy the verifier outputs and let it write its own feedback, in the form of a single TL;DR insight. The next rollout is conditioned on all previous insights, and we sequentially sample rollouts until a solution is found. Moreover, we enable backpropagation on the in-context insights to internalize a direct task to insight mapping. On challenging tool-calling and coding datasets (filtered to Pass@128=0), standard GRPO training of a Qwen 3.5 9B Thinking policy stays flat at a Pa
    
[^54]: 纠正而非删除：用纠正性监督缓解涌现性失调

    Correct, Don't Delete: Mitigating Emergent Misalignment with Corrective Supervision

    [https://arxiv.org/abs/2609.37624](https://arxiv.org/abs/2609.37624)

    本研究发现在微调数据被污染的场景下，将有害示例纠正为正确答案比直接删除它们更能有效缓解涌现性失调——替换四分之一的污染数据即可将失调率降低约三分之一，而删除同样的数据几乎没有效果。

    

    对语言模型在一小部分有害示例（如糟糕的医疗建议）上进行微调，可能使其在与这些示例无关的问题上出现广泛失调，这种现象被称为“涌现性失调（EM）”。通常的防御方法是找出有问题的数据行并将其删除，但我们的数据行定位器未能通过保留测试，而且删除数据行的效果不如预期。我们提出了一个不同的问题：给定一组固定的被污染数据行，纠正它们是否比删除它们更好？我们在糟糕医疗建议与良性聊天数据的混合数据上微调 Qwen2.5-14B-Instruct，预先选定四分之一的污染数据行，然后要么删除这些行，要么将其替换为针对相同提示的纠正后答案，其余条件保持完全一致。结果显示，替换这些行可将涌现性失调率降低约三分之一，并改善模型在保留医疗问题上的回答表现，而删除同样的数据行几乎没有可测量的效果。当一半的污染数据行被纠正时，这一优势更加显著。

    arXiv:2609.37624v1 Announce Type: cross  Abstract: Fine-tuning a language model on a narrow set of harmful demonstrations, such as bad medical advice, can make it broadly misaligned on unrelated questions, a phenomenon known as emergent misalignment (EM). The usual defense is to find the offending rows and delete them, but a row locator failed our held-out test and deleting rows helps less than expected. We ask a different question: given a fixed set of poisoned rows, is it better to correct them than to remove them? We fine-tune Qwen2.5-14B-Instruct on a mixture of bad medical advice and benign chat data, select a quarter of the poison rows in advance, and either delete them or replace each with a corrected answer to the same prompt, keeping everything else the same. Replacing the rows cuts the EM rate by about a third and improves answers on held-out medical questions, while deleting the same rows has little measurable effect. The advantage is larger when half the poison rows are cor
    
[^55]: 语言模型中的权威偏见：对来源的遵从与对用户的附和并非可以互换

    Authority Bias in Language Models: Source Deference and User Agreement Are Not Interchangeable

    [https://arxiv.org/abs/2609.37616](https://arxiv.org/abs/2609.37616)

    该研究发现，语言模型对“验证来源”（如检索结果、工具输出）的盲目遵从远强于对用户断言的附和——一条支持错误答案的权威来源注释可使45%-88%原本正确的回答被翻转，且这两种行为在模型内部是可通过因果干预选择性区分的、不可互换的不同机制。

    

    语言模型倾向于同意用户断言的任何内容，而后训练日益针对这种谄媚行为，使模型能够基于论断本身的优劣进行评估，而不是盲目顺从用户。然而，当错误答案被归因于一个经过验证的来源时，同样的模型会表现出更强的服从性——而这正是检索结果、工具输出和基于搜索的内容呈现信息的常见方式。我们在五个开源权重模型家族和三个闭源API上测量了这一差距。在八个模型中的七个里，一条支持错误答案的单一验证来源注释会使45%-88%原本正确的基线回答被翻转，且注释听起来越具权威性，模型的服从性越高。对来源的遵从与对用户的附和在模型内部并非行为上可互换：在具有相同错误答案的匹配项目上，因果干预可以选择性地抑制其中一种行为，而不对另一种产生同等影响。在三个开源权重模型家族中，移除拟合出的来源方向会（原文在此处截断）……

    arXiv:2609.37616v1 Announce Type: cross  Abstract: Language models tend to agree with whatever a user asserts, and post-training increasingly targets this sycophancy so that models evaluate claims on their merits rather than deferring to the user. Yet the same models are far more compliant when a wrong answer is attributed to a verified source, which is how retrieval results, tool outputs, and grounded-search content often present information. We measure this gap across five open-weight families and three closed APIs. A single verified-source note endorsing a wrong answer flips 45-88% of baseline-correct responses in seven of eight models, and compliance rises with how authoritative the note sounds. Source deference and user agreement are not behaviorally interchangeable inside the model: on matched items with the same wrong answer, causal interventions can selectively suppress one without equally affecting the other. In three open-weight families, removing a fitted source direction lo
    
[^56]: FOCUS：面向大语言模型智能体的免训练决策保持上下文压缩

    FOCUS: Training-Free Decision-Preserving Context Compression for LLM Agents

    [https://arxiv.org/abs/2609.37590](https://arxiv.org/abs/2609.37590)

    FOCUS是一个免训练、与架构无关的上下文压缩框架，它通过在测试时因果性地识别哪些历史交互塑造了智能体的未来决策来压缩上下文，避免了离线训练的高昂成本，同时缓解注意力稀释导致的性能下降。

    

    大语言模型智能体积累的交互历史随任务长度线性增长，导致推理成本呈二次方扩展，并因注意力稀释引发性能下降。现有的上下文压缩方法在离线阶段学习丢弃哪些内容：通过对比优化准则、蒸馏压缩器或训练压缩策略来实现。这带来了高昂的成本。此外，压缩策略是先验习得的，无法根据测试时不断演化的轨迹进行动态调整。在本文中，我们提出了一个互补的问题：哪些过去的交互在因果上塑造了智能体的未来决策？我们将上下文压缩重新表述为基于离散交互单元的因果决策保持问题，并提出了FOCUS——一个完全在测试时运行的免训练上下文压缩框架。我们的方法无需离线数据收集或微调，且与具体架构无关，可附加到任何封闭API的前端接口。

    arXiv:2609.37590v1 Announce Type: new  Abstract: LLM agents accumulate interaction histories that grow linearly with task length, causing quadratic inference cost scaling and performance degradation from attention dilution. Existing context-compression methods learn what to discard offline: by contrastively optimizing guidelines, distilling compressors, or training compression policies. This incurs a substantial cost. Further, the compression policy is learned a priori and is not dynamically conditioned on the evolving test-time trajectories. In this paper we ask a complementary question: Which past interactions causally shape the agent's future decisions? We recast context compression as a causal decision preservation problem over discrete interaction units and introduce FOCUS, a training-free context compression framework that operates entirely at test time. Our method requires no offline data collection or fine-tuning, and is architecture-agnostic, attaching to any closed-API fronti
    
[^57]: 基于信息价值推理的辅助智能体理性澄清

    Rational Clarification by Assistive Agents via Value-of-Information Reasoning

    [https://arxiv.org/abs/2609.37588](https://arxiv.org/abs/2609.37588)

    该论文提出 REVOIR 框架，让辅助智能体在推理阶段通过计算澄清问题的信息价值（即答案带来的预期任务奖励提升），来理性权衡是直接行动还是向用户提问，从而更安全有效地处理模糊请求。

    

    基于语言的辅助智能体的用户常常会提出模糊的请求。对此，助手既可以直接按照自己对请求的理解采取行动——这有可能与用户意图产生偏差——也可以提出澄清问题。哪种选择才是最安全且最有帮助的呢？一种常见做法是通过提问来不断降低对用户意图的不确定性，直到达到某个阈值。然而，这种做法忽略了以下因素：降低不确定性对下游任务性能的实际影响、提问与立即行动之间的成本权衡，以及用户可能无需被询问就主动给出纠正意见的可能性。为了在这些权衡中做出决策，我们提出了基于信息价值推理的理性询问框架（REVOIR）。REVOIR 通过在推理阶段对一个问题所蕴含的信息价值进行推理来做出澄清决策，该信息价值刻画了所获答案带来的任务奖励的预期提升。在两个辅助任务——模糊问题回答（摘要在此处被截断）

    arXiv:2609.37588v1 Announce Type: new  Abstract: Users of language-based assistive agents often make ambiguous requests. In response, an assistant can either directly act on its interpretation of the request --- risking misalignment with the user --- or ask a clarifying question. Which option is the most safe and helpful? A common approach is to ask questions that minimize uncertainty about the user's intent until a threshold is reached. However, this neglects the impact of uncertainty reduction on downstream performance, the costs of asking versus acting immediately, and the possibility that users may provide corrections without being asked. To navigate these trade-offs, we introduce Rational Enquiry via Value-of-Information Reasoning (REVOIR). REVOIR makes clarification decisions via inference-time reasoning about the value-of-information of a question, which captures the expected improvement in task reward due to the answer received. In two assistive tasks --- ambiguous question ans
    
[^58]: 配对难度至关重要：重新思考成对式LLM裁判评估与一致性

    Pair Difficulty Matters: Rethinking Pairwise LLM-as-a-Judge Evaluation and Consistency

    [https://arxiv.org/abs/2609.37577](https://arxiv.org/abs/2609.37577)

    该论文指出，传统用于评估LLM裁判可靠性的三大代理指标（位置偏差、传递性、成对一致性）具有误导性——在Bradley-Terry几何下它们被排名差距接近的配对所主导，而此类配对出现不一致在信息论上是必然的，因此不应以这些指标否定能力出色的LLM评估器。

    

    大型语言模型（LLM）裁判被广泛用于通过成对比较来对文本及文本生成系统进行排序，其可靠性通常通过三个代理指标来评估：位置偏差、传递性和成对一致性（自评或人工标注）。由于这些代理指标驱动着裁判的选择与基准测试，大量报告裁判在这些指标上表现不佳的文献可能会使实践者放弃原本有能力的评估器。我们认为这种评估具有误导性。在成对聚合所依据的Bradley-Terry几何结构下，每个代理指标都由排名差距接近的配对所主导，而在这类配对中，不一致性在信息论上是可以预期的，且个别判断对总体排名的贡献微乎其微；相比之下，排名差距较大的配对虽然承载着排名信号，却几乎不会影响这些代理指标。我们对该论点进行了形式化，并在受控模拟以及两个人工评分语料库上进行了验证：这些代理指标与排名质量仅呈弱相关。

    arXiv:2609.37577v1 Announce Type: new  Abstract: Large Language Model judges are widely used to rank texts and text-generating systems through pairwise comparison, and their reliability is typically assessed via three proxies: position bias, transitivity, and pairwise agreement (self- or human-labeled). Because these proxies drive judge selection and benchmarking, a substantial literature reporting that judges perform poorly on them risks steering practitioners away from otherwise capable evaluators. We argue this assessment is misleading. Under the Bradley--Terry geometry underlying pairwise aggregation, each proxy is dominated by close-rank-gap pairs, where inconsistency is information-theoretically expected and individual verdicts contribute little to the aggregate ranking; far-gap pairs carry the ranking signal but barely move the proxies. We formalize this argument and validate it in a controlled simulation and on two human-rated corpora: the proxies correlate only weakly with ran
    
[^59]: MERGE：基于生成式增强的多大语言模型集成检索框架

    MERGE: Multi-LLM Ensemble for Retrieval via Generative Enrichment

    [https://arxiv.org/abs/2609.37574](https://arxiv.org/abs/2609.37574)

    MERGE提出一个两阶段多LLM集成框架，先由三个小型开源LLM独立生成查询扩展候选、再由更大的LLM生成式合成为单一查询，并用基于下游检索性能的自动提示词优化循环取代传统LLM评估器，解决了单一LLM查询增强受限于模型偏见且提示词工程难以扩展的问题。

    

    大语言模型（LLM）越来越多地被用于信息检索（IR）中的用户查询增强，使BM25等标准检索器能够弥合查询与目标语料库之间的词汇鸿沟。然而，任何单一LLM都受限于其训练数据和架构偏见，且其增强行为依赖于手工设计的提示词——这些提示词必须针对每个新模型重新设计，是一个昂贵且难以扩展的过程。我们提出了MERGE（Multi-LLM Ensemble for Retrieval via Generative Enrichment，基于生成式增强的检索多LLM集成框架），这是一个两阶段框架：三个异构的7-8B开源LLM独立生成候选查询扩展，随后由一个更大的LLM将它们生成式地合成为单一查询。为了使提示词工程在整个模型集成中具备可扩展性，我们在两个阶段中都集成了基于任务的自动提示词优化（APO）循环。与使用LLM评估器来判断候选结果的APO方法不同，我们的循环根据每个候选的下游检索表现进行评分。

    arXiv:2609.37574v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly used to enrich user queries in information retrieval (IR) so that a standard retriever such as BM25 can bridge vocabulary gaps with the target corpus. Any single LLM, however, is limited by its training data and architectural biases, and its enrichment behavior depends on hand-crafted prompts that must be re-engineered for each new model -- an expensive and poorly scalable process. We present MERGE (Multi-LLM Ensemble for Retrieval via Generative Enrichment), a two-stage framework: three heterogeneous 7-8B open-source LLMs independently produce candidate expansions, and a larger LLM generatively synthesizes them into a single query. To make prompt engineering scalable across the ensemble, we integrate a task-grounded Automatic Prompt Optimization (APO) loop into both stages. Unlike APO methods that judge candidates with an LLM evaluator, our loop scores each candidate by its downstream retr
    
[^60]: 问题传递中的魔鬼：基于源条件化的传递引导以缓解音视频大语言模型中的幻觉

    Devils in Question Relay: Source-Conditioned Relay Steering to Mitigate Hallucinations in Audio-visual Large Language Models

    [https://arxiv.org/abs/2609.37568](https://arxiv.org/abs/2609.37568)

    该研究揭示了音视频大语言模型中“源混淆接地幻觉”的内部成因——问题传递机制使问题状态混入未使用模态的干扰线索，并提出基于源条件化的传递引导方法，通过切断干扰模态到问题状态的通路来有效缓解幻觉。

    

    音频-视觉大语言模型（AVLLMs）通过视觉、听觉与语言信息之间的交互，在多模态理解与推理方面取得了显著进展。然而，近期研究表明，AVLLMs 面临一个关键挑战：源混淆的接地幻觉，即来自未被使用模态的线索会诱导出所需模态并不支持的响应，损害了真实世界应用中的可靠性。现有方法在缓解这一失效问题上已取得进展，但其如何由内部跨模态交互产生仍缺乏充分理解。为填补这一空白，我们开展了路径干预与表征分析，揭示了一种“问题传递”机制：问题状态在承载所需源证据的同时也携带干扰线索，从而破坏了对所需模态证据的接地。切断从干扰模态到问题状态的通路能够带来更大的改进效果。

    arXiv:2609.37568v1 Announce Type: new  Abstract: Audio-visual large language models (AVLLMs) have made remarkable progress in multimodal understanding and reasoning through interactions among visual, auditory, and linguistic information. However, recent studies show that AVLLMs face a critical challenge: $\textbf{source-confused grounding hallucination}$, where cues from the unused modality induce responses that the required modality does not support, undermining reliability in real-world applications. Existing methods have made progress in mitigating this failure, yet how it arises from internal cross-modal interactions remains insufficiently understood. To address this gap, we conduct path-intervention and representation analyses, revealing a $\textbf{question-relay}$ mechanism: question states carry interfering cues alongside required-source evidence, undermining grounding in required-modality evidence. Cutting pathways from interfering modality to question states yields greater cor
    
[^61]: 正交却耦合：面向模型合并的几何组件解耦

    Orthogonal Yet Coupled: Decoupling Geometric Components for Model Merging

    [https://arxiv.org/abs/2609.37564](https://arxiv.org/abs/2609.37564)

    提出DiGA几何感知模型合并框架，通过以预训练权重为共享几何参考，将任务向量正交分解为具有不同几何属性的组件并分别聚合，解决了传统整体合并方式中跨组件耦合导致合并模型质量下降的问题。

    

    合并预训练模型已成为将多种能力整合到单一统一模型中的有效方法。然而，现有的主流合并方法通常将每个任务向量视为不可分割的合并单元，忽略了其中编码的异构几何变化。这种处理方式会引发跨组件耦合问题：当合并决策基于完整任务向量的统计量得出时，某个组件的几何特性可能会影响另一个组件的选择、加权或组合方式，从而可能降低合并模型的质量。为了解决这个问题，我们提出了DiGA，一个解耦的几何感知模型合并框架。以预训练权重作为共享的几何参考，DiGA将每个任务向量正交分解为对应不同几何属性的组件。DiGA并非将任务向量作为一个整体进行合并，而是聚合相应的组件

    arXiv:2609.37564v1 Announce Type: new  Abstract: Merging pretrained models has emerged as an effective approach for consolidating diverse capabilities into a single unified model. However, prevailing merging methods typically treat each task vector as an indivisible merging unit, overlooking the heterogeneous geometric changes encoded within it. This treatment can induce cross-component coupling: when merging decisions are derived from statistics of the complete task vector, the geometric characteristics of one component may influence how another is selected, weighted, or combined, potentially degrading the quality of the merged model. To address this issue, we propose DiGA, a Disentangled Geometry-Aware model merging framework. Using the pretrained weights as a shared geometric reference, DiGA orthogonally decomposes each task vector into components corresponding to distinct geometric attributes. Rather than merging the task vectors as a whole, DiGA aggregates corresponding components
    
[^62]: RunyaNER：Runyankore语命名实体识别中的辅助语言选择

    RunyaNER: Auxiliary Language Selection for Runyankore NER

    [https://arxiv.org/abs/2609.37543](https://arxiv.org/abs/2609.37543)

    该论文发布了首个公开的Runyankore语命名实体识别基准数据集RunyaNER（含23.7万标注词），并利用它系统研究了跨语言零样本迁移和多语言微调中辅助语言选择策略的效果。

    

    跨语言零样本迁移和多语言微调是命名实体识别（NER）等自然语言处理任务在低资源语言中的有前景的方法，但在缺乏目标语言基准的情况下，尚不清楚哪种辅助语言选择策略能带来最佳的迁移效果。我们推出了RunyaNER，这是首个公开可用的东非语言Runyankore的NER基准数据集，并利用它来研究迁移时应选用哪些语言。RunyaNER通过半自动化流程创建并经过完全人工验证，涵盖3万个句子中超过23.7万个标注词。我们在RunyaNER上对预训练模型进行了基准测试，证实了该数据集的质量和规模足以训练出有效的Runyankore NER模型。随后，我们使用RunyaNER研究了跨语言零样本和多语言微调设置下的辅助语言选择问题。我们的实验表明，虽然迁移表现……（原文在此处截断）

    arXiv:2609.37543v1 Announce Type: new  Abstract: Cross-lingual zero-shot transfer and multilingual fine-tuning are promising approaches for NLP tasks such as Named Entity Recognition (NER) in low-resource languages, but in the absence of target language benchmarks, it is unclear which auxiliary language selection strategy leads to the best transfer. We introduce RunyaNER, the first publicly available NER benchmark for the East African language Runyankore, and use it to investigate the choice of which languages to use for transfer. Created with a semi-automated pipeline and fully manually verified, RunyaNER contains over 237k annotated words across 30k sentences. We benchmark pretrained models on RunyaNER, establishing that our dataset is of sufficient quality and size to produce effective Runyankore NER models. We then use RunyaNER to investigate auxiliary language selection in cross-lingual zero-shot and multilingual fine-tuning settings. Our experiments show that while transfer perfo
    
[^63]: E-MoE：用于非因子化扩散语言模型的增强型专家混合方法

    E-MoE: Enhanced Mixture-of-Experts for Non-Factorized Diffusion Language Models

    [https://arxiv.org/abs/2609.37533](https://arxiv.org/abs/2609.37533)

    E-MoE利用MoE骨干网络的专家路由决策作为离散共享潜在变量，将掩码扩散模型的逆向过程构建为非因子化的分布混合，在不增加激活参数的前提下显著提升了少步生成质量。

    

    掩码扩散模型（MDMs）通过在每一步去噪中逐步解除多个词元的掩码来生成序列，但其逆向过程通常在各位置上是因子化的，这限制了少步生成场景下的样本质量，而少步生成正是扩散模型相对自回归解码的速度优势最为关键的领域。近期一系列工作引入了以变分自编码器方式训练的连续高斯潜在变量来捕捉位置间的相关性，但这类方法容易出现后验坍缩问题，即潜在变量被悄然忽略。我们提出了增强型专家混合模型（E-MoE），它将逆向过程构建为在离散共享潜在变量上的因子化分布之混合，该离散共享潜在变量由专家混合骨干网络的专家路由决策给出，且与因子化基线相比不增加激活参数量。在合成多模态基准、二值化MNIST和LM1B数据集上，E-MoE相较于因子化方法提升了少步生成的表现。

    arXiv:2609.37533v1 Announce Type: new  Abstract: Masked diffusion models (MDMs) generate sequences by progressively unmasking several tokens per denoising step, but their reverse process is typically factorized over positions, limiting sample quality in the few-step regime where diffusion's speed advantage over autoregressive decoding matters most. A recent line of work introduces a continuous Gaussian latent, trained as a variational autoencoder, to capture correlations across positions, but such approaches are prone to posterior collapse, where the latent is silently ignored. We propose Enhanced Mixture-of-Experts (E-MoE), which builds the reverse process as a mixture of factorized distributions over a discrete shared latent given by the expert-routing decisions of a Mixture-of-Experts (MoE) backbone, without increasing active parameters over the factorized baseline. Across synthetic multi-modal benchmarks, binarized MNIST, and LM1B, E-MoE improves few-step generation over factorized
    
[^64]: 视觉-语言模型基准测试的层次化压缩

    Hierarchical Compression of Vision-Language Model Benchmarks

    [https://arxiv.org/abs/2609.37515](https://arxiv.org/abs/2609.37515)

    提出了PRIMEBench——一个视觉感知的层次化基准压缩框架，通过数据清洗、类别代表性选择、视觉感知方差（VAW）题目剪枝和类别数量剪枝四个阶段，在保持模型排名的同时大幅降低视觉-语言模型的评估成本。

    

    对视觉-语言模型（VLM）进行全面评估的成本已变得高得令人望而却步，因为基准测试涵盖的能力范围日益广泛，且新模型以不懈的速度不断涌现。能够在极低成本下保持模型排名的基准压缩方法在语言模型领域已得到充分研究，但对于视觉-语言模型，这一问题仍处于探索不足的状态。我们提出了PRIMEBench（Pruning Redundant Items for Multimodal Evaluation，面向多模态评估的冗余项剪枝），这是一个具有视觉感知能力的层次化基准压缩框架，能够在保持模型排名的同时大幅降低评估成本。该层次化框架分四个阶段运行：数据清洗（去除无需图像即可回答的题目以及全部模型均回答正确的题目）、类别代表性选择（为每个能力类别选取一个基准测试）、使用视觉感知方差（VAW）进行题目剪枝，以及类别数量剪枝。VAW将模型间的方差与基于多模态计算的视觉依赖性得分相结合。

    arXiv:2609.37515v1 Announce Type: cross  Abstract: Thorough evaluation of vision-language models (VLMs) has become prohibitively expensive, as benchmarks span an ever-broader spectrum of capabilities and new models arrive at a relentless pace. Benchmark compression methods that preserve model rankings at a fraction of the cost are well studied for language models, but for VLMs the question remains under-explored. We present PRIMEBench (Pruning Redundant Items for Multimodal Evaluation), a vision-aware hierarchical benchmark compression framework that substantially reduces evaluation cost while preserving model rankings. This hierarchical framework operates in four stages: data cleaning to remove items answerable without the image and all-correct items, category representative selection to pick one benchmark per capability category, item pruning with Vision-Aware Variance (VAW), and category-count pruning. VAW combines inter-model variance with a vision-dependence score computed from mu
    
[^65]: 从不协调到统筹编排：在线策略蒸馏中的教师干预

    From Dissonance to Orchestration: Teacher Intervention in On-Policy Distillation

    [https://arxiv.org/abs/2609.37510](https://arxiv.org/abs/2609.37510)

    该论文发现更深的教师干预在在线策略蒸馏中收益递减且会增加离策略负载，并提出 MAESTRO 方法，通过策略分歧分数自适应地决定教师何时接管以及生成多长时间。

    

    在线策略蒸馏（OPD）利用更强教师模型的反馈，在学生模型自身的推理轨迹上进行训练。教师模型的干预可以改善这些轨迹，但同时也会改变学生所学习的分布。我们的对照研究表明，仅凭 rollout 质量是分配教师指导的不完整标准。更深入的干预在 rollout 准确率上的收益递减，同时会增加离策略负载。在一个具有受限 rollout 视野的训练探针实验中，学生的峰值准确率和性能保持率偏好不同的干预强度。偏好的干预深度和干预位置也因基准测试而异。这些发现促成了 MAESTRO 方法，它利用局部策略分歧来联合调整教师何时接管以及生成多长时间。其“策略分歧分数”将教师加权的候选覆盖度与局部分布相似度相结合，并在推理（片段）内进行聚合……

    arXiv:2609.37510v1 Announce Type: new  Abstract: On-policy distillation (OPD) trains a student on its own reasoning trajectories using feedback from a stronger teacher. Teacher interventions can improve these trajectories, but also change the distribution on which the student learns. Our controlled studies show that rollout quality alone is an incomplete criterion for allocating teacher guidance. Deeper intervention yields diminishing gains in rollout accuracy while increasing off-policy load. In a training probe with a restricted rollout horizon, peak student accuracy and performance retention favor different intervention strengths. The preferred intervention depth and placement also vary across benchmarks. These findings motivate MAESTRO, which uses local policy disagreement to jointly adapt when the teacher takes over and how long it generates. Its {policy disagreement score} combines teacher-weighted candidate coverage with local distribution similarity and is aggregated within rea
    
[^66]: 评估受监管智能体AI中的有界自主性：一种融合宪法奖励、升级标签与运行时治理的诊断测试框架

    Evaluating Bounded Autonomy in Regulated Agentic AI: A Diagnostic Harness with Constitutional Rewards, Escalation Labels, and Runtime Governance

    [https://arxiv.org/abs/2609.37501](https://arxiv.org/abs/2609.37501)

    提出RegLLM诊断框架，利用宪法奖励、任务级升级标签和确定性运行时治理来衡量并约束受监管智能体AI的有界自主性，实验表明治理机制可将升级召回率从0提升至0.67、将不安全行为率从0.33降至0.08。

    

    我们提出了RegLLM，一个用于评估受监管智能体工作流中有界自主性的诊断测试框架。该框架测量六项可信度信号：引用有效性、来源依据、模式合规性、升级正确性、宪法一致性以及不安全行为率。这些信号根据其监督来源加以区分：程序化验证器、任务级升级标签或AI评判分数。一个确定性的运行时监督器会阻止缺乏依据的回答并强制升级，同时记录干预措施。同一份领域宪法同时被用于评估、训练奖励和服务端防护。任务级“应当升级”标签使“行动还是推迟”的决策成为一种可测量的训练信号。我们在冒烟测试规模上演示了该框架。一次离线参考运行（n=12）显示，在启用治理的情况下，升级召回率从0提升至0.67，不安全行为率从0.33降低至0.08。两次单GPU Qwen2.5-3B的LoRA/DPO试点实验（n=8，相同种子及……（原文摘要在此处截断）

    arXiv:2609.37501v1 Announce Type: cross  Abstract: We propose RegLLM, a diagnostic harness for bounded autonomy in regulated agentic workflows. It instruments six trustworthiness signals: citation validity, source grounding, schema compliance, escalation correctness, constitutional alignment, and unsafe-action rate. Signals are distinguished by their source of supervision: programmatic verifiers, task-level escalation labels, or AI-judge scores. A deterministic runtime supervisor blocks ungrounded answers and forces escalation, logging interventions. The same domain constitution informs evaluation, training rewards, and serving guardrails. Task-level should-escalate labels make the act-versus-defer decision a measurable training signal. We demonstrate the harness at smoke scale. An offline reference run (n=12) lifts escalation recall from 0 to 0.67 and reduces unsafe-action rate from 0.33 to 0.08 when governance is enabled. Two single-GPU Qwen2.5-3B LoRA/DPO pilots (n=8, same seed and 
    
[^67]: 谁让档案变暖了？大语言模型高估了历史上的温暖程度

    Who Warmed the Archives? LLMs Overestimate Historical Warmth

    [https://arxiv.org/abs/2609.37499](https://arxiv.org/abs/2609.37499)

    该研究首次系统评估了利用大语言模型从历史档案中提取温度指数的可靠性，发现所有测试的LLM均存在随年代递增的系统性偏暖偏差，使其误差在跨世纪气候比较中并不安全。

    

    历史档案是一种未被充分利用的资源，可以将仪器测量的气候记录向更早的年代延伸，而大语言模型（LLM）为提取气候学家原本需要手工推导的指数提供了一种新途径。除了衡量系统提取该信号的能力之外，我们还检验了其误差用于跨世纪比较是否安全，因为良好的相关分数并不能排除系统性的、与年代相关的偏差。我们针对跨越五个世纪的德语文本中的Pfister温度指数，比较了词汇基线方法、微调的历史文本transformer模型以及LLM提示方法，结果显示词汇方法击败了我们测试的所有微调transformer模型，包括一个从零开始在历史德语上预训练的模型（r=-0.016）。我们测试的全部六个LLM（Gemini 2.5 Flash、GPT-5-mini、DeepSeek v4 Flash、Claude Sonnet 4.6、Qwen3.7-Plus、Kimi-K2.6-Fast）都表现出随日历年份增长而增强的偏暖偏差，且所有模型的偏差方向一致（斜率为每世纪+0.13至+0.34，p<0.01）。该效应在规模上较为适度（r……

    arXiv:2609.37499v1 Announce Type: new  Abstract: Historical archives are an under-used source for extending the instrumental climate record backward in time, and LLMs offer a way to extract the indices climatologists derive by hand. Beyond measuring how well systems extract this signal, we check whether their errors are safe to use for cross-century comparison, since a good correlation score does not rule out systematic, era-linked bias. Comparing lexical baselines, fine-tuned historical transformers, and LLM prompting on the Pfister temperature index across five centuries of German text, lexical methods beat every fine-tuned transformer we test, including one pretrained from scratch on historical German (r=-0.016). All six LLMs we test (Gemini 2.5 Flash, GPT-5-mini, DeepSeek v4 Flash, Claude Sonnet 4.6, Qwen3.7-Plus, Kimi-K2.6-Fast) show a warm bias that grows with calendar year, with the same sign in every model (slopes +0.13 to +0.34/century, p<0.01). The effect is modest in size (r
    
[^68]: 罗生门式的维基百科：对分歧历史叙事的数据视角主义分析

    The Rashomon Wikipedia: A Data-Perspectivist Analysis of Divergent Historical Narratives

    [https://arxiv.org/abs/2609.37498](https://arxiv.org/abs/2609.37498)

    该研究通过对五种语言维基百科中罗马尼亚三个争议历史事件的分析，首次揭示了语言版本之间严重的“引用隔离”现象及民族中心主义叙事偏见——如波萨达之战的119条引用中仅2条跨语言共享，罗马尼亚语版本91%的引用呈现亲民族立场。

    

    维基百科旨在提供一个统一、中立的历史记录，然而其独立的语言版本往往作为不同的认知共同体运作，围绕争议性事件形成分歧的叙事。本文通过分析五种语言（罗马尼亚语、匈牙利语、俄语、土耳其语和英语）的维基百科文章来研究跨语言的史学偏见，重点关注罗马尼亚历史上三个有争议的事件：波萨达之战（1330年）、苏联占领比萨拉比亚（1940年）以及特尔戈维什特夜袭（1462年）。通过使用人工标注员和大语言模型（LLM）对引用立场进行分类，并量化2005年至2024年间的叙事演变，我们识别出一种“引用隔离”现象。在波萨达之战的案例中，119条引用中仅有2条在不同语言版本之间共享，且罗马尼亚语版本表现出91%的亲民族偏见，而匈牙利语版本则相对平衡。

    arXiv:2609.37498v1 Announce Type: new  Abstract: Wikipedia aims to provide a unified, neutral record of history, yet its independent language editions often function as distinct epistemic communities, creating divergent narratives around contested events. This paper investigates cross-lingual historiographical bias by analyzing Wikipedia articles across five languages (Romanian, Hungarian, Russian, Turkish, and English) focusing on three contentious events in Romanian history: the Battle of Posada (1330), the Soviet occupation of Bessarabia (1940), and the Night Attack at Targoviste (1462). Using human annotators and Large Language Models (LLMs) to classify citation stance and quantify narrative evolution from 2005 to 2024, we identify a phenomenon of "citation isolation". In the case of the Battle of Posada, only 2 out of 119 citations were shared between language editions, with the Romanian edition exhibiting a 91% pro-national bias compared to the balanced Hungarian edition. Longitu
    
[^69]: 拉里使汽车停了下来，但模型没有注意到：Transformer对M-启发式的盲区

    Larry Caused the Car to Stop, But the Model Didn't Notice: Transformer Blindness to the M-Heuristic

    [https://arxiv.org/abs/2609.37497](https://arxiv.org/abs/2609.37497)

    本文通过自然语言推理实验对比词汇使役与分析型使役结构，发现DeBERTa、RoBERTa和BART等Transformer模型无法捕捉M-启发式所编码的语用区别，且探针分析的结果实际反映的是句法复杂度而非使役性。

    

    现代Transformer模型擅长通过句子嵌入捕捉语义关系，但它们执行语用推理的能力仍缺乏充分研究。本文考察了基于编码器的Transformer模型（如DeBERTa）是否运用了M-启发式（新格赖斯原则：有标记的语言形式隐含有标记的意义）。我们通过自然语言推理框架，将词汇使役结构（如“Larry stopped the car”）与分析型使役结构（如“Larry caused the car to stop”）进行对比来检验这一假设。我们在188种实验条件下对15个作格兼及物动词进行的实验表明，DeBERTa、RoBERTa和BART均未表现出能够捕捉这两种形式之间语用区别的证据，其中DeBERTa对100%的案例都预测为“中性”。探针分析最初暗示存在表征使用的分离，但对照实验揭示探针追踪的实际上是句法复杂度，而非使役性。

    arXiv:2609.37497v1 Announce Type: new  Abstract: Modern transformer models excel at capturing semantic relationships through sentence embeddings, yet their ability to perform pragmatic reasoning remains understudied. This paper investigates whether encoder-based transformers such as DeBERTa employ the M-Heuristic (the neo-Gricean principle that marked linguistic forms implicate marked meanings). We test this hypothesis by contrasting lexical causatives (e.g., ``Larry stopped the car'') with periphrastic causatives (e.g., ``Larry caused the car to stop'') using a Natural Language Inference framework. Our experiments across 188 conditions with 15 ambitransitive verbs reveal that DeBERTa, RoBERTa, and BART show no evidence of capturing the pragmatic distinction between these forms, with DeBERTa predicting ``Neutral'' for 100% of cases. Probing analysis initially suggested a representation-use dissociation, but control experiments reveal the probe was tracking syntactic complexity, not cau
    
[^70]: 你的基准测试并未饱和：通过答案池化复兴多选题评估

    Your Benchmark Is Not Saturated: Reviving Multiple-Choice Evaluation with Answer Pooling

    [https://arxiv.org/abs/2609.37494](https://arxiv.org/abs/2609.37494)

    提出AnswerPool方法，无需编写新题，仅将共享上下文的多道题目的选项合并为一个池并要求模型同时作答，即可大幅降低饱和多选题基准的可猜测性，同时还能在同一轮次中评估模型的弃权能力。

    

    多选题基准测试评分成本低廉，但性能空间正趋于耗尽，而标准补救方法——编写更难的题目——耗时缓慢，且需要为每个基准重复进行。一个已经饱和的基准测试中其实仍蕴含着更难的任务。每道题的错误选项都是仅针对该题单独编写的，因此模型可以通过排除少数选项来获得分数。我们提出AnswerPool：选取N道共享同一上下文的题目，将它们的所有选项汇集到一个列表中，然后要求模型为每道题分配其对应答案。无需编写新题目，也无需更改标签。对于五道四选项的题目，猜对整组的概率从10⁻³降至5×10⁻⁷；能够真正识别自己答案的模型可以保持其多选题分数，因此因池化而损失的准确率衡量了该题型因排除法而给予的分值。从选项池中删除正确答案会使题目变得无法回答（真值不再存在），因此弃权行为可以在同一轮次中进行评分。在八个文本、图像……（原文截断）

    arXiv:2609.37494v1 Announce Type: new  Abstract: Multiple-choice benchmarks are cheap to grade and are running out of room, and the standard remedy, writing harder items, is slow and repeated for every benchmark. A saturated benchmark still holds a harder task. Each question's wrong options are written for that question alone, so a model can score by eliminating a few options. We propose AnswerPool: take $N$ questions that share a context, pool all their options into one list, and ask the model to assign every question its answer. No item is written and no label changes. The chance of guessing a group right falls from $10^{-3}$ to $5\times10^{-7}$ for five four-option questions, and a model that recognizes its answers keeps its multiple-choice score, so the accuracy lost to pooling measures the credit the format gave for elimination. Deleting answers from the pool makes questions unanswerable with exact ground truth, so abstention is scored in the same pass. Across eight text, image, a
    
[^71]: 基于无标签检查定价的风险受控选择性大语言模型回答

    Risk-Controlled Selective LLM Answering by Pricing Label-Free Checks

    [https://arxiv.org/abs/2609.37493](https://arxiv.org/abs/2609.37493)

    PriceCheck通过为无标签检查（如重新解题）赋予“价格”来构建决策规则族，在给定选择性风险目标下自动决定何时作答、何时弃答，在数学任务上平均提供76.1%的答案并将选择性风险控制在1.5%以下，效果优于奖励模型、提示评判器、生成器置信度等基线方法。

    

    从大语言模型中提供答案需要决定何时弃答，然而验证器的排名准确率本身并不能决定已提供答案中的错误率。我们提出PriceCheck，它从无标签检查（例如重新求解一个问题）构建一个紧凑的决策规则族。每个检查都有一个价格：它在正确答案与错误答案上的一致率，以及每次运行的成本。在一个小型、类别丰富的标注集上拟合出的价格可以组合成对某个调度方案覆盖率和成本的预测，从而指导运行哪些检查以及何时停止。随后通过校准测试，在给定的选择性风险目标下选择一个调度方案。在数学领域，所选的调度方案平均提供76.1%的答案，并且在全部15个数据划分上都将留出集的选择性风险保持在1.5%以下。在共享测试协议下，PriceCheck在该风险目标下提供的答案数量超过奖励模型、提示评判器、生成器置信度以及训练的正确性分类器。

    arXiv:2609.37493v1 Announce Type: cross  Abstract: Serving an answer from a large language model requires deciding when to abstain, yet a verifier's ranking accuracy alone does not determine the error rate among served answers. We introduce PriceCheck, which builds a compact family of decision rules from label-free checks such as re-solving a problem. Each check has a price: its agreement rates on correct and incorrect answers and its cost per run. Prices fitted on a small, class-enriched labelled set compose into predictions of a schedule's coverage and cost, guiding which checks to run and when to stop. A calibration test then selects a schedule at a stated selective-risk target. In mathematics, the selected schedules serve 76.1% of answers on average and keep held-out selective risk below 1.5% on all 15 splits. Under the shared testing protocol, PriceCheck serves more answers at that target than reward models, a prompted judge, the generator's confidence and a trained correctness cl
    
[^72]: 面向证据门控问答的机制边界对齐

    Regime Boundary Alignment for Evidence-Gated Question Answering

    [https://arxiv.org/abs/2609.37491](https://arxiv.org/abs/2609.37491)

    提出机制边界对齐（RBA）方法，通过在同一问题的有支持与无支持匹配变体上训练单一阅读器，使其在有证据支持时作答、证据缺失时弃权，无需验证器、阈值或机制标签即可将多跳问答中的无支持作答率降低超过六十个百分点。

    

    检索增强型语言模型被期望基于检索到的证据进行回答，但在实践中，即使证据缺失，它们往往仍然继续作答。我们将这一行为归因于训练信号：以答案为中心的微调对无支持的上下文没有设定训练目标，因此无法区分会弃权的阅读器与会胡乱猜测的阅读器，即使在有支持的情形下准确率有所提升，无支持作答率仍然接近100%。我们提出机制边界对齐，在同一问题和标准答案的匹配变体上训练单个阅读器。该阅读器被训练为：当上下文支持答案时生成标准答案（包括同时存在冲突证据的情形），而当正确的支持被移除时则弃权；推理阶段仅采用普通解码，无需验证器、阈值或机制标签。在三个多跳问答数据集、三组随机种子的实验中，RBA将无支持作答率降低了超过六十个百分点。

    arXiv:2609.37491v1 Announce Type: cross  Abstract: Retrieval-augmented language models are expected to answer from the retrieved evidence, but in practice they often keep answering when that evidence is missing. We trace this behavior to the training signal: answer-focused fine-tuning assigns no target to unsupported contexts, so it cannot distinguish a reader that abstains from one that guesses, and unsupported answering stays near 100% even as supported accuracy improves. We introduce Regime Boundary Alignment (RBA), which trains a single reader on matched variants of the same question and gold answer. The reader is trained to produce the gold answer when the context supports it, including when conflicting evidence is also present, and to abstain when the correct support is removed; inference is ordinary decoding, with no verifier, threshold, or regime label. On three multi-hop QA datasets across three seeds, RBA reduces the unsupported-answer rate by more than sixty percentage point
    
[^73]: FORUM：基于模型一致性协调冻结输出以实现视觉定位

    FORUM: Frozen Outputs Reconciled Using Model Agreement for Visual Grounding

    [https://arxiv.org/abs/2609.37488](https://arxiv.org/abs/2609.37488)

    该论文提出FORUM，一种无需训练的测试时融合框架，利用多个冻结多模态大语言模型的预测一致性（基于一致性选择和中位定位两条几何规则）来提升对抗性视觉定位任务的鲁棒性，仅用三个开源模型就在Ref-Adv-s基准上以相对5%的平均准确率优势超越了397B参数的参考模型。

    

    冻结的多模态大语言模型（MLLMs）如今通过单次提示调用即可解决标准的指代表达理解任务，然而在包含同类别干扰项和否定表达的对抗性基准测试中，即使是最大的模型也会自信地犯错，且重新采样只会重复相同的错误。由不同数据和架构构建的模型很少会被相同的干扰因素所误导，因此它们之间的一致性是识别正确目标的强大无标签信号。我们提出了FORUM，一种无需训练的测试时融合方法，由两条固定的几何规则指导：基于一致性的选择保留被最多不同模型共同支持的区域，而中位定位返回一个真实的成员边界框而非坐标平均值，从而确保单个宽松的预测无法偏移最终答案。通过融合三个开源MLLMs，FORUM在对抗性基准Ref-Adv-s上的平均准确率相对超过了397B参数的已发表参考模型5%，而简单的平均集成方法（摘要在此处被截断）……

    arXiv:2609.37488v1 Announce Type: cross  Abstract: Frozen multimodal large language models (MLLMs) now solve standard referring expression comprehension with a single prompted call, yet on adversarial benchmarks with same-category distractors and negation, even the largest models are confidently wrong, and resampling repeats the error. Models built from different data and architectures rarely fall for the same confounder, so their agreement is a strong label-free signal of the correct target. We present FORUM, a training-free test-time fusion of frozen MLLMs guided by two fixed geometric rules: agreement-based selection keeps the region supported by the most distinct models, and medoid localization returns an actual member box instead of a coordinate average, so one loose prediction cannot shift the answer. Fusing three open MLLMs, FORUM surpasses the 397B-parameter published reference by a relative 5% in mean accuracy on the adversarial Ref-Adv-s benchmark, and a plain averaging ensem
    
[^74]: 相关性并非充分证据：在RAG生成之前检测证据缺口

    Relevance Is Not Sufficient Evidence: Detecting Evidence Gaps Before Generation in RAG

    [https://arxiv.org/abs/2609.37469](https://arxiv.org/abs/2609.37469)

    该研究揭示了现有证据不足测试基准的构建陷阱，并提出一个通过替换、删除和问题交换构造、控制表面特征的配对基准，证明可以在生成之前仅凭问题与检索证据判断其充分性，从而使RAG系统在证据缺失时更可靠地弃答。

    

    检索增强生成（RAG）通过外部来源为大语言模型提供依据，但检索到的段落往往只是提到了正确的实体，却未提供回答问题所需的事实。即使被明确要求在证据不足时弃答，12个生成器仍会回答40.0%至99.3%的证据不足问题。通过训练生成器来学会弃答，会将这一决策绑定在模型权重上，可能奖励从参数化知识中“回忆”出的答案，且仍需要一次完整的生成器调用。那么，能否仅凭问题和证据，在任何答案产生之前判断证据是否充分？我们指出了构建“证据不足”测试时的陷阱：删除相关证据或将证据与不相关问题配对，可能通过词汇重叠或证据位置泄露标签。我们构建了一个配对基准，采用替换、删除和问题交换三种构造方式，在控制用词等选定表面特征的同时改变证据对答案的支持程度。证据充分性可以在……（原文摘要截断）

    arXiv:2609.37469v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) grounds large language models in external sources, but retrieved passages often name the right entities without providing the facts needed to answer. Even when instructed to abstain, 12 generators answer 40.0-99.3% of insufficient-evidence questions. Training generators to abstain ties the decision to model weights, may reward answers recalled from parametric knowledge, and still requires a full generator call. Can sufficiency be judged from the question and evidence alone, before any answer exists? We identify pitfalls in constructing insufficient-evidence tests: removing relevant evidence or pairing evidence with unrelated questions can reveal labels through lexical overlap or evidence position. We build a paired benchmark using substitution, deletion, and question-swap constructions that vary answer support while controlling selected surface features, such as word use. Sufficiency can be judged wit
    
[^75]: 面向长期记忆问答的学习式缺失证据检索

    Learning to Retrieve Missing Evidence for Long-Term Memory QA

    [https://arxiv.org/abs/2609.37443](https://arxiv.org/abs/2609.37443)

    提出MERA框架，通过强化学习训练轻量级规划器，利用已验证的证据作为线索迭代式地检索缺失证据，在长期记忆问答任务上显著超越现有方法。

    

    长期记忆使语言模型能够在未来的对话中利用过去的交互内容。然而，回答问题所需的证据可能分散在相距遥远的对话轮次中，而问题本身往往省略了定位这些证据所需的线索。已检索到的事实可以揭示这些线索，这促使检索决策应当依赖于已经找到的证据。我们提出了MERA（缺失证据检索增强，Missing-Evidence Retrieval Augmentation），它将全局可搜索的记忆与针对特定问题的证据状态分离开来。已验证的证据可以指导后续的检索，同时不会限制对全局记忆的访问。我们通过强化学习训练了一个轻量级规划器，对能够找回先前缺失证据的查询给予奖励。MERA在Qwen3-30B和GPT-4o-mini两种骨干模型上都取得了出色的答案准确率。使用Qwen3-30B进行证据处理和答案生成时，经过训练的0.6B规划器在LoCoMo上达到77.40%的准确率，在LongMemEval-S上达到71.29%的准确率，超过了……

    arXiv:2609.37443v1 Announce Type: cross  Abstract: Long-term memory enables language models to use past interactions in future conversations. However, evidence needed to answer a question may be scattered across distant turns, while the question itself omits clues needed to locate it. Retrieved facts can reveal these clues, motivating retrieval decisions conditioned on evidence already found. We introduce MERA (Missing-Evidence Retrieval Augmentation), which separates globally searchable memory from a question-specific evidence state. Verified evidence guides subsequent retrieval without restricting access to the global memory. We train a lightweight planner through reinforcement learning, rewarding queries that recover previously missing evidence. MERA achieves strong answer accuracy across Qwen3-30B and GPT-4o-mini backbones. With Qwen3-30B for evidence processing and answer generation, the trained 0.6B planner achieves 77.40% accuracy on LoCoMo and 71.29% on LongMemEval-S, exceeding
    
[^76]: 看看你们把我们逼成什么样了：从Reddit话语中提取仇恨叙事

    Look What You Made Us Cluster: Hate Narrative Extraction from Reddit Discourse

    [https://arxiv.org/abs/2609.37408](https://arxiv.org/abs/2609.37408)

    提出了一种基于LLM推理的仇恨叙事提取流程，将叙事表示为实体-评价对，结合Leiden聚类与LLM引导的细化过程，从Reddit话语中实现更精确、更可解释的仇恨叙事识别。

    

    叙事提取使我们能够识别网络仇恨叙事，为构建严谨的检测系统提供支持。然而，现有的计算方法精度有限，因为它们依赖语义表示，而语义表示往往只能捕捉表层含义。为了检测更精确、更可解释的叙事，我们提出了一种将叙事表示为实体-评价对的提取流程。叙事通过一个扩展了基于方面情感分析的大语言模型（LLM）推理过程来提取，该过程识别方面、将判断类型分类作为评价的基础，并据此推导出相应的评价。提取出的叙事随后使用Leiden算法进行聚类，之后通过一个LLM引导的细化过程将聚类解析到预期的粒度级别。我们以2024年批评泰勒·斯威夫特的英文Reddit评论为例，展示了这一叙事提取流程。

    arXiv:2609.37408v1 Announce Type: new  Abstract: Narrative extraction allows us to identify online hate narratives, supporting the construction of rigorous detection systems. Existing computational approaches, however, are limited in precision as they rely on semantic representations, which tend to capture only surface-level meaning. To detect more precise and interpretable narratives, we present an extraction pipeline that represents narratives as entity-evaluation pairs. Narratives are extracted using a Large Language Model (LLM) reasoning process that extends Aspect-Based Sentiment Analysis, identifying the aspect, classifying its judgement type as the basis for evaluation, and deriving the evaluation accordingly. Extracted narratives are then clustered using Leiden, following which clusters are resolved to an intended level of granularity through an LLM-guided refinement process. We illustrate this narrative pipeline with English Reddit comments from 2024 that criticize Taylor Swif
    
[^77]: 将学习问题编译为语言模型的适应程序

    Compiling Learning Problems into Adaptation Programs for Language Models

    [https://arxiv.org/abs/2609.37371](https://arxiv.org/abs/2609.37371)

    提出“适应编译”框架，通过从历史适应中学习预测候选程序的多维反事实响应面，在适应开始前自动选择最优更新程序，并可跨不同下游目标复用而无需重新训练。

    

    模型适应通常由固定的方案主导，尽管不同的更新程序可以产生截然不同的行为结果。我们提出了“适应编译”的概念，将模型应在哪里、如何以及以何种程度进行适应重新定义为一个联合预测与决策问题。编译器无需为每个学习情境重新搜索候选程序，而是从先前的适应经验中学习，预测候选程序上的向量值反事实响应面——即各程序对知识习得、迁移、有界性和保持性的预期影响——并在适应开始之前选定一个程序。由于这种预测的几何结构捕获了多种行为后果，而非单一最优解或标量分数，因此它可以在不同的下游优先级下复用而无需重新训练。在五种学习类型中，首选程序在不同学习情境之间存在显著差异，而这种差异是可预测的。

    arXiv:2609.37371v1 Announce Type: cross  Abstract: Model adaptation is typically governed by a fixed recipe, even though different update programs can produce substantially different behavioral outcomes. We introduce adaptation compilation, which reframes where, how, and to what extent a model should adapt as a joint prediction and decision problem. Rather than searching over candidate programs anew for each learning episode, a compiler learns from prior adaptations to predict a vector-valued counterfactual response surface over candidate programs---their expected effects on acquisition, transfer, boundedness, and preservation---and selects a program before adaptation begins. Because this predicted geometry captures multiple behavioral consequences rather than a single winner or scalar score, it can be reused under different downstream priorities without retraining. Across five learning types, preferred programs vary meaningfully across episodes, and this variation is predictable from 
    
[^78]: SemOPT：通过奖励引导搜索修复基于大语言模型的优化建模中的语义错误

    SemOPT: Fixing Semantic Errors in LLM-based Optimization Modeling via Reward-Guided Search

    [https://arxiv.org/abs/2609.37361](https://arxiv.org/abs/2609.37361)

    提出了SemOPT框架，通过语义奖励模型与奖励引导搜索，自动检测并修复大语言模型生成优化建模代码中难以察觉的语义错误。

    

    运筹学在能源、经济和医疗等领域为决策提供支持。求解运筹学问题通常始于优化建模，即将自然语言的问题描述转化为可执行的求解器代码。大语言模型（LLM）为自动化这一过程提供了有前景的途径，但它们仍然容易出错。在实践中，这些错误可以分为两类：语法错误是指求解器代码无法成功运行或被求解器判定为不可行；语义错误是指求解器代码成功返回了目标函数值，但违背了原始问题的意图。由于语义错误不会触发运行时故障，因此难以检测和纠正。为了解决这一问题，我们提出了SemOPT，一个用于纠正基于大语言模型的优化模型的语义引导框架。SemOPT结合了一个语义奖励模型，用于区分忠实的数学模型……

    arXiv:2609.37361v1 Announce Type: new  Abstract: Operations research supports decision-making in domains such as energy, economics, and healthcare. Solving operations research problems typically begins with optimization modeling, which translates a natural-language problem description into executable solver code. LLMs offer a promising way to automate this process, but they remain prone to errors. In practice, these errors can be divided into two categories: syntactic errors refer to solver code that fails to run successfully or is judged infeasible by the solver; semantic errors refer to solver code that successfully returns an objective value but violates the intent of the original problem. Since semantic errors do not trigger runtime failures, they are difficult to detect and rectify. To address this problem, we introduce SemOPT, a semantic-guided framework for correcting LLM-based optimization models. SemOPT combines a semantic reward model that distinguishes faithful math models f
    
[^79]: 端口哈密顿潜空间深思：缓解测试时计算扩展中的深思漂移悬崖

    Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling

    [https://arxiv.org/abs/2609.37351](https://arxiv.org/abs/2609.37351)

    该论文发现了测试时潜空间深思中随思考深度增加而推理崩溃的“深思漂移悬崖”现象，并提出端口哈密顿潜空间深思框架，以同时兼顾表达能力、Lyapunov稳定性与计算效率，从而在深层思考步数下有效抑制漂移、缓解推理性能崩溃。

    

    测试时计算扩展已成为先进机器推理的基石，然而在连续潜表示空间中直接执行迭代深思会暴露出一种灾难性病理现象：深思漂移悬崖。尽管无约束的循环潜空间模型在较短的推理步数（K ≤ 4）下能获得初步的推理收益，但当外推至更深的思考步数（K ≥ 16）时，其推理能力会发生崩溃，在标准逻辑基准测试中性能下降22%至62%。我们通过22轮实证与理论研究，解决了测试时潜空间推理中表达能力、Lyapunov稳定性与计算效率之间的三难困境。我们证明，严格保守的标量势梯度流能够抑制长程漂移（悬崖仅为3.40%），但会将峰值推理准确率瓶颈在32.73%；而无约束的旋转流虽实现了高符号表达能力（82.33%），却会遭受高达36.87%的严重漂移。

    arXiv:2609.37351v1 Announce Type: cross  Abstract: Test-time compute scaling has emerged as a cornerstone of advanced machine reasoning, yet performing iterative deliberation directly within continuous latent representation spaces reveals a catastrophic pathology: the Deliberation Drift Cliff. While unconstrained recurrent latent models achieve initial reasoning gains at short horizons (K <= 4), their reasoning collapses when extrapolated to deeper thinking steps (K >= 16), dropping by 22% to 62% across standard logical benchmarks. We resolve the trilemma among expressivity, Lyapunov stability, and computational efficiency in test-time latent reasoning through a 22-round empirical and theoretical investigation. We demonstrate that strictly conservative scalar potential gradient flows suppress long-range drift (cliff 3.40%) but bottleneck peak reasoning accuracy at 32.73%, whereas unconstrained rotational flows achieve high symbolic expressivity (82.33%) but suffer a severe 36.87% drift
    
[^80]: 不停下也能解题：小规模上的策略内蒸馏

    Solving Without Stopping: On-Policy Distillation at Small Scale

    [https://arxiv.org/abs/2609.37326](https://arxiv.org/abs/2609.37326)

    该研究通过将 Qwen3-8B 蒸馏到 0.6B–4B 的小模型，发现策略内蒸馏能够传递解题能力，但在思考模式下无法传递“知道何时停止推理”的能力，且学生模型的提升受限于其训练前多次尝试即可达到的性能上限。

    

    策略内蒸馏是指学生模型从更强的教师模型对其自身输出的反馈中进行学习，这是将推理能力传递给更小模型的常用方法。我们在小规模上分析了这种蒸馏究竟传递了什么，将 Qwen3-8B 蒸馏到 Qwen3-4B、1.7B 和 0.6B 学生模型中，分别在思考模式（先进行长篇推理，然后结束推理并给出答案）以及作为对比的非思考模式（没有单独的推理阶段）下进行。长篇推理需要两种能力：解决一个问题，以及知道问题何时已被解决。我们发现蒸馏传递了第一种能力，但在思考模式下并不传递第二种能力。解题能力在各个规模上都有提升，但受限于两个天花板——我们在两种模式和所有学生规模上对此进行了全面测量：学生在训练后的单次尝试永远不会超过其训练前通过多次尝试所能达到的水平，而且学生模型越小，其与教师模型的差距就越大。而停止推理则是两种模式产生分歧的地方。在非思考模式下，每个学生模型都能保持（摘要在此处截断）

    arXiv:2609.37326v1 Announce Type: new  Abstract: On-policy distillation, where a student learns from a stronger teacher's feedback on its own outputs, is a common way to pass reasoning to smaller models. We analyze what it transfers at small scale, distilling Qwen3-8B into Qwen3 4B, 1.7B and 0.6B students, in thinking mode (reason at length, then end the reasoning and answer) and, for comparison, in non-thinking mode (no separate reasoning phase). Long reasoning needs two abilities, solving a problem and knowing when it is solved, and we find that distillation transfers the first, but in thinking mode not the second. Solving improves at every size, up to two ceilings, which we measure comprehensively across both modes and all student sizes: a student's single attempt never exceeds what it could already reach in many attempts before training, and the smaller the student, the further it stays below the teacher. Stopping is where the modes part. In non-thinking mode every student keeps st
    
[^81]: 隐蔽推理必然泄露信息，但未必可被读取：思维链监控的根本机遇与局限

    Hidden Reasoning Must Leak, but Need Not Be Readable: Fundamental Opportunities and Limits for Chain-of-Thought Monitoring

    [https://arxiv.org/abs/2609.37312](https://arxiv.org/abs/2609.37312)

    本文证明足够复杂的隐藏计算必然在思维链中留下信息论痕迹，但在密码学假设下模型可以加密推理过程使其对多项式时间监控器不可读，从而揭示了思维链监控的根本机遇与局限。

    

    推理模型能否欺骗思维链监控器，并在其思维痕迹中不显露出隐藏计算的情况下执行隐藏计算？我们证明，答案取决于底层任务的难度和模型大小。简单的计算可以被隐蔽地执行；然而，超过一个取决于模型大小的阈值后，成功解决任务必然会将近乎线性数量的关于隐蔽任务输入的信息泄露到思维链中。因此，足够复杂的隐藏计算总会留下信息论意义上的痕迹。然而，令人担忧的是，这种泄露未必是可读的：在合理的密码学假设下，即使是单层Transformer也能在线加密其推理过程，使得任何多项式时间的监控器都无法提取关于隐藏计算的信息。总体而言，我们的理论和实证结果为思维链监控的机遇与局限提供了全面的视角。

    arXiv:2609.37312v1 Announce Type: cross  Abstract: Can reasoning models trick chain of thought (CoT) monitors and perform hidden computation without revealing it in their thinking traces? We show that the answer depends on the underlying task difficulty and the model size. Simple computations can be performed covertly; however, beyond a threshold depending on model size, successfully solving the task necessarily leaks a near-linear amount of information about the covert task input into the CoT. Therefore, sufficiently complex hidden computation always leaves an information-theoretic footprint. However, concerningly, this leakage need not be readable: Under plausible cryptographic assumptions, even a one-layer Transformer can encrypt its reasoning online so that no polynomial-time monitor can extract information about the hidden computation. Overall, our theoretical and empirical results provide a holistic view of both the opportunities and the limitations of CoT monitoring.
    
[^82]: 询问从未被要求之事：智能体中的横向与纵向主动性

    Asking for What Was Never Requested: Horizontal and Vertical Proactivity in Agents

    [https://arxiv.org/abs/2609.37236](https://arxiv.org/abs/2609.37236)

    该论文提出了智能体主动性的全新维度——横向主动性（追寻当前上下文已隐含的未明说信息）与纵向主动性（追寻仅由早期证据揭示的需求），并利用从基准分解中恢复的需求图实现了无需模型评判者的自动评分，同时设计了无需奖励模型的Q&D方法来训练智能体提出最有效的问题。

    

    使用工具的智能体通常只响应用户明确提出的请求，然而完成任务可能需要用户从未要求提供的信息。现有的主动智能体研究主要关注智能体是否以及何时应该自主行动，而非它应该追寻什么信息。我们研究了主动性的一个独特维度：其内容。横向主动性追寻当前上下文已经隐含的未明说信息，纵向主动性则追寻只有通过更早的证据才能揭示的需求。需求图从基准测试自身的任务分解中恢复，记录了哪些需求依赖于哪些需求，因此这两种主动性形式以及智能体是否在恰当时机停止，都可以直接从对话记录中评分，而无需模型评判者。为了学习这种行为，我们提出了Q&D（提问者与起草者）方法，它训练提问者优先选择那些其后续回答能够检索到更多所需证据的问题，整个过程无需奖励模型或评判者。在保留测试集上（摘要在此处截断）

    arXiv:2609.37236v1 Announce Type: new  Abstract: An agent that uses tools typically responds to what the user explicitly asks, yet completing the task may require information the user never requested. Work on proactive agents mainly studies whether and when an agent should act on its own, not what information it should pursue. We study a distinct axis of proactivity: its content. Horizontal proactivity pursues unstated information that the current context already identifies, and vertical proactivity pursues needs that only earlier evidence reveals. A need graph, recovered from a benchmark's own decomposition, records which needs depend on which, so both forms, and whether the agent stops at the right time, can be scored from a transcript without a model judge. To learn this behavior, we propose Q&D (questioner and drafter), which trains a questioner to prefer the question whose continuation retrieves more of the required evidence, with no reward model or judge. On held-out splits of th
    
[^83]: 跟随实体：面向智能体搜索的语料库地图

    Follow the Entities: A Corpus Map for Agentic Search

    [https://arxiv.org/abs/2609.37226](https://arxiv.org/abs/2609.37226)

    提出CorpusMap，一种围绕文档中反复出现实体来组织语料库的导航层，帮助大语言模型智能体在多文档搜索中发现文档间的关联，从而减少遗漏互补证据并降低令牌消耗。

    

    在大规模文档集合上回答问题和完成任务，通常需要连接分散在多个文档中的证据，例如某个项目的批准记录在一个文档中，其需求在另一个文档中，而其最新状态在第三个文档中。近期的大语言模型（LLM）智能体通过迭代式搜索整个语料库来应对这一挑战，而不是仅阅读一组固定的排名靠前的文档。然而，当语料库仅以扁平的文件集合形式呈现时，相关文档无法体现它与其他文档之间的关系，因此智能体必须为每个查询重新发现这些关系，这往往导致遗漏互补证据，同时消耗大量额外的令牌（token）。为了解决这个问题，我们提出了CorpusMap，这是一个导航层，它围绕语料库中反复出现的实体来组织语料库。这些实体可以从文档本身中识别出来，并能够将单个文档与来自不同来源的许多其他文档链接起来。具体而言，CorpusMap……

    arXiv:2609.37226v1 Announce Type: cross  Abstract: Answering questions and completing tasks over large document collections often requires connecting evidence spread across multiple documents, such as a project's approval recorded in one, its requirements in another, and its latest status in a third. Recent LLM agents approach this by iteratively searching the full corpus rather than reading only a fixed set of top-ranked documents. However, when the corpus is exposed only as a flat collection of files, a relevant document gives no indication of how it relates to others, so the agent must rediscover these relationships for every query, often missing complementary evidence while simultaneously consuming substantial additional tokens. To address this, we introduce CorpusMap, a navigation layer that organizes the corpus around its recurring entities, which are identifiable from the documents themselves and can link a single document to many others across sources. Specifically, CorpusMap r
    
[^84]: CredWise：一个用于可解释、可审计信用风险评估的受控智能体决策智能框架

    CredWise: A Controlled Agentic Decision-Intelligence Framework for Explainable and Auditable Credit-Risk Assessment

    [https://arxiv.org/abs/2609.37223](https://arxiv.org/abs/2609.37223)

    本文提出CredWise决策支持框架，将XGBoost信用风险预测与概率校准、SHAP可解释性分析、政策检索及受控智能体工作流相结合，在真实贷款数据上实现了性能优良且时间稳定的可解释、可审计信贷风险评估。

    

    信用风险预测在银行业中十分重要，但单纯的预测结果既无法解释申请者为何存在风险，也无法说明应如何将其与其他证据相结合。本文提出了CredWise，一个集成信用风险预测、概率校准、可解释人工智能、政策检索、SQL分析以及受控智能体工作流的决策支持框架。研究基于Lending Club数据（1,345,310笔贷款，18个特征）训练XGBoost模型，并采用时间划分方式：2007—2016年用于训练，2017年用于验证，2018年用于测试。在2018年测试集上，校准后的模型取得了ROC-AUC 0.7109、PR-AUC 0.2993、F1分数0.3714和65.44%准确率的成绩。概率校准将Brier分数从0.2157降至0.1273，期望校准误差从0.2862降至0.0585。SHAP解释在时间上保持稳定，2017年与2018年特征排名之间的Spearman相关系数高达0.9959。在28个标签（摘要内容在此处被截断）

    arXiv:2609.37223v1 Announce Type: new  Abstract: Credit-risk prediction is important in banking, but a prediction alone does not explain why an applicant is risky or how it should be combined with other evidence. This paper presents CredWise, a decision-support framework that integrates credit-risk prediction, probability calibration, explainable artificial intelligence, policy retrieval, SQL analytics, and controlled agent-based workflows. An XGBoost model is trained on Lending Club data (1,345,310 loans, 18 features) using a temporal split: 2007--2016 for training, 2017 for validation, and 2018 for testing. On the 2018 test set, the calibrated model achieved a ROC-AUC of 0.7109, PR-AUC of 0.2993, F1-score of 0.3714, and accuracy of 65.44\%. Calibration reduced the Brier score from 0.2157 to 0.1273 and the expected calibration error from 0.2862 to 0.0585. SHAP explanations were temporally stable, with a Spearman correlation of 0.9959 between 2017 and 2018 feature rankings. On 28 label
    
[^85]: 面向端到端组合优化的视觉语言模型微调

    VLM Fine-Tuning for End-to-End Combinatorial Optimization

    [https://arxiv.org/abs/2609.37175](https://arxiv.org/abs/2609.37175)

    本文提出一种通用视觉语言求解器，将输入导出的视觉表示与文本描述相结合，通过监督微调和验证器引导的强化学习训练单一视觉语言模型，在CVRP、JSSP等复杂组合优化问题上显著优于纯文本方法，且问题规模越大优势越明显。

    

    大语言模型（LLM）为端到端组合优化（CO）提供了统一的接口，但仅依靠文本序列化可能会掩盖对生成有效组合优化解至关重要的空间和关系结构。本文提出了一种通用的视觉语言求解器，通过由输入导出的视觉表示来增强文本实例描述。单一的视觉语言模型（VLM）被应用于不同的组合优化任务，并采用先监督微调、后验证器引导的强化学习的方式进行训练。尽管视觉输入中不包含标准答案或任何由解导出的信息，我们的实验表明，该视觉语言模型总体上比其纯文本对应模型提升了求解质量，在CVRP（带容量约束的车辆路径问题）和JSSP（作业车间调度问题）等更复杂的组合优化问题上提升尤为明显。视觉信息的优势在大规模问题下更为显著。

    arXiv:2609.37175v1 Announce Type: new  Abstract: Large language models (LLMs) have provided a unified interface for end-to-end combinatorial optimization (CO), but textual serialization alone may obscure spatial and relational structures that are important for generating effective CO solutions. This paper presents a general-purpose vision-language solver that augments textual instance descriptions with input-derived visual representations. A single vision-language model (VLM) is applied across different CO tasks and trained using supervised fine-tuning followed by verifier-guided reinforcement learning. While the visual inputs contain no gold solutions or solution-derived information, our experiments show that the VLM generally improves solution quality over its text-only counterpart, with particularly clear gains on more complex CO problems such as CVRP and JSSP. The advantage of visual information is more pronounced at large problem scales.
    
[^86]: 通过生成上下文知识融合弥合检索增强生成中的语义鸿沟

    Bridging Semantic Gaps in RAG through Generated Context Knowledge Fusion

    [https://arxiv.org/abs/2609.37171](https://arxiv.org/abs/2609.37171)

    提出知识感知语义桥接框架，通过智能知识融合实现查询与检索文档的语义空间对齐，从而弥合RAG中的语义鸿沟并提升段落选择的相关性与准确性。

    

    检索增强生成已成为自然语言处理领域的基础框架，它将信息检索与大语言模型的生成能力无缝集成。然而，这一过程从根本上受到一个关键挑战的制约：查询与检索到的上下文之间存在语义空间不匹配问题。我们提出了知识感知语义桥接，这是一个新颖的框架，通过智能知识融合实现查询与检索文档之间的语义空间对齐，从而提升段落选择的质量。我们的方法通过多阶段过程充分利用生成式知识与检索式知识的互补优势，同时增强了相关性和准确性。我们在三个流行的开放域问答数据集上对KASB进行了评估，以证明我们方法的有效性。

    arXiv:2609.37171v1 Announce Type: new  Abstract: Retrieval-Augmented Generation has established itself as a fundamental framework in natural language processing, seamlessly integrating information retrieval with the generative capabilities of large language models. However, this process is fundamentally constrained by a critical challenge: semantic space mismatch between queries and retrieved contexts. We propose Knowledge-Aware Semantic Bridging (KASB), a novel framework that improves passage selection quality through semantic space alignment between queries and retrieved documents through intelligent knowledge fusion. Our approach leverages the complementary strengths of generative and retrieval-based knowledge through a multistage process that enhances both relevance and accuracy. We evaluate KASB on three popular open-domain Question Answering datasets to demonstrate the effectiveness of our approach.
    
[^87]: 轨迹汤：通过多样化轨迹推动大语言模型中期训练的计算扩展前沿

    Trajectory Soup: Pushing the Compute-Scaling Frontier of LLM Mid-training via Diverse Trajectories

    [https://arxiv.org/abs/2609.37169](https://arxiv.org/abs/2609.37169)

    提出 Trajectory Soup 方法，将大语言模型中期训练的计算预算分散到多个独立分支上，并通过轨迹内与轨迹间权重平均将最优检查点融合为单一模型，从而突破串行计算带来的收益天花板，拓展中期训练的计算扩展前沿。

    

    中期训练为预训练的大语言模型赋予专业化与推理能力，但这一阶段的收益存在上限，因为额外的串行计算几乎无法带来进一步的下游改进，甚至可能损害某些能力，这为中期训练所能吸收的计算量设置了实际上的天花板。我们重新审视了这些计算应如何分配——是用于单次运行，还是多次类似的优化。我们发现，从共享检查点出发、在不同受控配方下分叉的各个分支会到达参数空间中可测量的不同区域，并由此确立了一种单次运行延长所无法提供的“兼容多样性”。因此，我们提出 Trajectory Soup（轨迹汤），它将中期训练预算分配到多个独立分支上，并通过轨迹内与轨迹间的权重平均，把基于验证集筛选出的最强检查点整合为单一模型。一项局部的偏差与方差分析区分了……

    arXiv:2609.37169v1 Announce Type: cross  Abstract: Mid-training equips pretrained large language models with specialized and reasoning capabilities, but the returns of this stage are bounded since additional serial compute yields little further downstream improvement and can even degrade some capabilities, which places a practical ceiling on how much compute mid-training absorbs. We revisit how this compute should be allocated to a single run or multiple similar optimizations. We find that branches forked from a shared checkpoint under various controlled recipe reaches measurably different regions of parameter space, and establish a form of compatible diversity that extending one run cannot supply. Therefore, we introduce Trajectory Soup, which distributes a mid-training budget over several independent branches, and consolidates strongest checkpoints selected on validation through intra- and inter-trajectory averaging into a single model. A local bias and variance analysis separates th
    
[^88]: 高阶行为构念的多模态检测：结构化反思互动中的自我关怀

    Multimodal Detection of Higher-Order Behavioral Constructs: Self-Compassion in Structured Reflective Interaction

    [https://arxiv.org/abs/2609.37148](https://arxiv.org/abs/2609.37148)

    该研究针对“自我关怀”这一不可直接观察的高阶心理构念，构建了首个多模态数据集，收集并标注51场结构化反思对话，探索从随时间变化的言语、声音与动作中推断此类复杂行为特质。

    

    许多对人们的学习与成长最为重要的品质——例如一个人如何调节情绪、如何反思挫折、或在困难对话中如何保持对他人的觉察——是无法被直接观察的。这些品质必须从一个人随时间推移的言语、动作和声音中推断出来，而且难以满足大多数机器学习流程所依赖的那种“干净”标注方式。我们通过一个在心理学理论中有扎实基础却很少被计算建模的案例来研究这一挑战：自我关怀，即以耐心而非苛刻的自我批评来回应自身挫折的倾向。我们考察了它在技术介导的训练环境中结构化反思访谈里如何呈现，在这种环境中，人们自然会谈论那些社会情感上具有挑战性的情境。由于目前没有现成数据集能在此类情境中捕捉这类构念，我们收集并标注了51场反思对话。

    arXiv:2609.37148v1 Announce Type: cross  Abstract: Many of the qualities that matter most in how people learn and grow, how someone regulates their emotions, reflects on a setback, or stays aware of others during a difficult conversation, are not directly observable. They have to be inferred from how someone speaks, moves, and sounds over time, and they resist the kind of clean labeling that most machine learning pipelines are built around. We study this challenge through a case that is well grounded in psychological theory but rarely modeled computationally: self-compassion, the tendency to respond to one's own setbacks with patience rather than harsh self-criticism. We examine how it appears during structured reflective interviews in a technology-mediated training setting, where people naturally talk through socio-emotionally demanding situations. Since no existing dataset captures this kind of construct in this kind of setting, we collected and annotated 51 reflective dialog session
    
[^89]: LoLBench：基于大型软件系统上长周期提案的编码智能体评估

    LoLBench: Evaluating Coding Agents with Long-Horizon Proposals on Large Software Systems

    [https://arxiv.org/abs/2609.37143](https://arxiv.org/abs/2609.37143)

    LoLBench是一个多语言基准测试，通过让编码智能体在大型软件系统上完成从人工编写的增强提案到代码实现的完整流程，首次同时评估其将用户意图转化为规范的感知能力和生成正确代码修改的实现能力。

    

    现代编码智能体能够交付越来越大规模的仓库级代码变更，近期的基准测试也反映了这一趋势，强调具有大型参考实现的长周期任务。许多基准测试评估的是编码智能体的实现能力，即根据详细规范生成正确的代码修改。然而，实际的模块化开发任务还需要感知能力，即将用户意图和高层设计落地转化为规范。我们提出LoLBench，通过在大型软件系统上执行从提案到实现的完整流程来评估这两种能力。这是一个多语言基准测试，涵盖五个领域中29个软件系统上的100个任务。每个任务提供一份由人工编写的增强提案，其中包含用户意图和高层设计。平均而言，提案约含5,000个单词，软件系统包含240万行源代码（LoC），实现拉取请求（PR）所修改的代码量约……（原文在此处截断）

    arXiv:2609.37143v1 Announce Type: cross  Abstract: Modern coding agents can deliver increasingly large repository-level changes, and recent benchmarks reflect this by emphasizing long-horizon tasks with large reference implementations. Many benchmarks evaluate coding agents' implementation capability to produce correct code edits from detailed specifications. However, practical modular development tasks also require the perception capability of grounding user intent and high-level design to derive a specification. We introduce LoLBench to evaluate both capabilities through the entire proposal-to-implementation process on large software systems. It is a multilingual benchmark of 100 tasks across 29 software systems in five domains. Each task provides a human-written enhancement proposal with user intent and high-level design. On average, proposals contain about 5,000 words, software systems contain 2.4 million source lines of code (LoC), and implementation pull requests (PRs) change app
    
[^90]: 大语言模型去品牌化：在保留通用效用的同时擦除商业身份

    LLM unbranding: Erasing Commercial Identity while Preserving Generic Utility

    [https://arxiv.org/abs/2609.37127](https://arxiv.org/abs/2609.37127)

    本文正式定义了“大语言模型去品牌化”这一新任务，旨在中和LLM文本输出中定义品牌身份的标志性语言、口号和风格标记，以防范商标淡化、错误归因和品牌诽谤等风险，同时保留模型的通用效用，并提供了覆盖多商业领域知名品牌的综合评估数据集作为基准。

    

    在图像生成领域，将“去品牌化”确立为一项关键实践以防止视觉标识获得负面含义已是标准做法。如今，大语言模型（LLM）面临着类似的、新兴的挑战。这些模型经常在多种语境下生成品牌描述，这种高频出现带来了重大风险，例如商标淡化、错误归因和品牌诽谤。为此，我们正式定义了大语言模型去品牌化这一新任务，专门针对在文本输出中处理商业外观的复杂挑战，即中和那些定义品牌身份的标志性语言、口号和风格标记。关键在于，这些元素不如显性的视觉标识那样显而易见。为了给该任务建立基准，我们引入了一个涵盖多个商业领域知名品牌的综合评估数据集，并对现有的最先进机器遗忘方法进行了严格评估。

    arXiv:2609.37127v1 Announce Type: new  Abstract: Establishing unbranding as a critical practice to prevent visual logos from acquiring negative connotations is standard in image generation. Large Language Models (LLMs) now face a parallel and emerging challenge. These models frequently generate brand descriptions within diverse contexts. This frequency introduces significant risks, such as trademark dilution, false attribution, and brand defamation. In response, we formally define the novel task of LLM Unbranding. We specifically address the complex challenge of managing trade dress within textual outputs. This involves neutralizing characteristic language, slogans, and stylistic markers that define brand identity. Crucially, these elements are less evident than explicit visual logos. To benchmark this task, we introduce a comprehensive evaluation dataset incorporating prominent brands from multiple commercial domains. We rigorously evaluate existing state-of-the-art machine unlearning
    
[^91]: 双语音素BabyLM中的跨语言效应

    Cross-Linguistic Effects in Bilingual Phoneme BabyLMs

    [https://arxiv.org/abs/2609.37121](https://arxiv.org/abs/2609.37121)

    该研究将语音音素表示引入双语BabyLM训练，固定英语为第二语言并变换德语、瑞典语、波斯语和巴斯克语作为第一语言，以在更贴近儿童口语学习条件的框架下研究双语习得中的跨语言效应。

    

    跨语言效应是双语第一语言习得中的核心议题。人工学习者能够通过对不同语言组合和学习条件进行受控比较，帮助研究第一语言（L1）与第二语言（L2）之间的交互作用。近期的工作通过在符合发展合理性的约束下训练双语语言模型来探索这一方向。然而，人类学习者和模型学习者之间仍存在根本性差异，其中一个主要差异在于输入模态：儿童主要从口语输入中学习，而语言模型通常基于书写文本进行训练。为了缩小这一差距，研究者们已经在语音的音素表示上训练模型。在这项工作中，我们将这些研究方向结合起来，使用音素输入训练双语BabyLM。我们将英语固定为第二语言（L2），并变换第一语言（L1），涵盖德语、瑞典语、波斯语和巴斯克语，这些语言被选取以代表句法结构和音位库方面的对比性组合……

    arXiv:2609.37121v1 Announce Type: new  Abstract: Cross-linguistic effects are a central topic in bilingual first-language acquisition. Artificial learners can help investigate L1-L2 interactions by enabling controlled comparisons across language combinations and learning conditions. Recent work explores this direction by training bilingual language models under developmentally plausible constraints. However, human and model learners still diverge in fundamental ways, with one major difference being input modality: children learn primarily from spoken input, whereas language models are typically trained on orthographic text. To reduce this gap, researchers have trained models on phonemic representations of speech. In this work, we combine these research directions to train bilingual BabyLMs with phonemic input. We keep English fixed as the L2 and vary the L1 across German, Swedish, Persian, and Basque, selected to represent contrasting combinations of syntactic and phoneme-inventory dis
    
[^92]: 解锁评论家：面向大语言模型后训练的无奖励策略优化

    Unlocking the Critic: Reward-Free Policy Optimization for LLM Post-Training

    [https://arxiv.org/abs/2609.37119](https://arxiv.org/abs/2609.37119)

    该论文提出保留常被丢弃的预训练评论家，通过小幅度低方差的策略更新实现稳定训练，并利用评论家从未完成前缀预测成功概率的能力提供无需完整采样与奖励的密集学习信号，从而实现高效的长程推理后训练。

    

    近年来，针对大语言模型的强化学习（RL）后训练方法日益倾向于移除评论家（critic）网络，以降低训练不稳定性与内存开销。即使在训练了评论家的情况下，一旦训练结束，评论家也会被丢弃，尽管它已经学会了预测结果。我们重新审视了这一趋势，并证明预训练评论家预测未来结果的能力可以使其成为实现高效长程推理的宝贵资产。首先，我们发现基于评论家的强化学习在长思维链推理中的不稳定性很大程度上是一种优化层面的伪影：将策略更新保持在较小幅度且低方差，即可恢复稳定的收敛。其次，经过良好预训练的评论家能够从轨迹的后续状态和未完成的前缀中估计最终成功的后验概率。其预测提供了由结果导出的、密集的、针对每个前缀的学习信号，在策略优化过程中，这些信号既不需要完整的轨迹采样，也不需要步骤级的（标注）……

    arXiv:2609.37119v1 Announce Type: cross  Abstract: Recent approaches to reinforcement learning (RL) post-training for large language models increasingly remove the critic to reduce training instability and memory overhead. Even where a critic is trained, it is discarded once training ends, although it has learned to predict outcomes. We revisit this trend and show that a pretrained critic's ability to predict future outcomes can make it a valuable asset for efficient long-horizon reasoning. First, we find that instability in critic-based RL for long chain-of-thought reasoning is largely an optimization artifact: keeping policy updates small and low in variance restores stable convergence. Second, a well-pretrained critic estimates the posterior probability of eventual success from later trajectory states and unfinished prefixes. Its predictions provide outcome-derived, dense, per-prefix learning signals that, during policy optimization, require neither completed rollouts, step-level an
    
[^93]: VACE：基于验证门控的智能体模型与执行框架交替协同进化

    VACE: Validation-Gated Alternating Co-Evolution of Agent Models and Harnesses

    [https://arxiv.org/abs/2609.37105](https://arxiv.org/abs/2609.37105)

    VACE提出了一种验证门控的交替协同进化方法，将智能体强化学习与基于轨迹的框架优化交替进行，只有当候选框架通过验证性能门控时才被采纳，从而在OfficeQA和AutomationBench上显著超越仅权重强化学习和无门控交替方法。

    

    语言模型智能体可以通过更新模型权重或优化引导任务执行的框架（harness）来改进。这两个组件是相互耦合的：权重更新会改变模型使用框架的方式，而框架更新会改变用于训练的轨迹。我们提出了VACE（Validation-Gated Alternating Co-Evolution，验证门控交替协同进化），该方法将智能体强化学习与基于轨迹的框架优化交替进行。在每个强化学习阶段之后，VACE复用已收集的轨迹来提出框架修订方案，并在保持更新后模型固定的情况下，对现有框架和候选框架进行评估。只有当候选框架能够提升验证性能时，才会被用于指导后续训练。使用Qwen3.5-9B模型，VACE在OfficeQA上达到了45.26%的测试准确率，在AutomationBench上达到了75.19%的平均部分得分，分别超过仅权重的强化学习方法6.43和9.09个百分点，超过无门控的交替方法4.59和6.95个百分点。在44个框架提案中……

    arXiv:2609.37105v1 Announce Type: cross  Abstract: Language model agents can be improved by updating their model weights or refining the harness that guides task execution. These components are coupled: weight updates change how the model uses the harness, while harness updates change the trajectories used for training. We propose VACE, Validation-Gated Alternating CoEvolution, which alternates agentic reinforcement learning with trajectory-driven harness refinement. After each RL stage, VACE reuses the collected trajectories to propose a harness revision and evaluates the incumbent and candidate with the updated model held fixed. The candidate guides subsequent training only if it improves validation performance. With Qwen3.5-9B, VACE achieves 45.26% test accuracy on OfficeQA and a mean partial-credit score of 75.19% on AutomationBench, exceeding weight-only RL by 6.43 and 9.09 percentage points and ungated alternation by 4.59 and 6.95 points, respectively. Across 44 harness proposals
    
[^94]: 后训练在多语言推理中改变了什么？

    What Does Post-Training Change in Multilingual Reasoning?

    [https://arxiv.org/abs/2609.37104](https://arxiv.org/abs/2609.37104)

    该论文通过对Qwen3模型在十一种语言竞赛数学任务上的审计发现，非英语语言中仅有约15-18%的问题能获得符合语言要求的正确完整解答（英语达92.9%），并通过系统评估十三个模型端点（包括多语言SFT与多种RL奖励设计）来定位多语言推理差距的来源。

    

    开源推理模型在不同语言之间提供的推理能力访问是不平等的。当一个模型能够解决某个问题，却无法用用户的语言给出完整的解答时，语言就成为一种访问障碍，而不仅仅是性能差异的来源。我们对Qwen3的检查点在十一种语言的竞赛数学任务上进行了审计。在十种非英语语言中，在16次采样中仅有15.4-17.9%的问题能得到以所请求语言呈现的、正确且能正常终止的、带有可见推理过程的解答，而英语中这一比例为92.9%。为了找出这种差距的来源，我们评估了来自同一模型家族的十三个端点，涵盖已发布的检查点、两种规模的多语言监督微调（SFT）、受控SFT消融实验，以及三种强化学习（RL）奖励设计。我们联合追踪正确性、语言遵循度、终止能力和交付效率。主要的瓶颈……

    arXiv:2609.37104v1 Announce Type: cross  Abstract: Open-source reasoning models provide unequal access to reasoning capability across languages. When a model can solve a problem but cannot deliver a complete solution in the user's language, language becomes an access barrier rather than merely a source of performance variation. We audit Qwen3 checkpoints on competition-mathematics tasks in eleven languages. Across the ten non-English languages, only 15.4-17.9% of problems receive a correct, terminating solution with visible reasoning in the requested language in any of 16 samples, compared with 92.9% in English. To identify the source of this disparity, we evaluate thirteen endpoints from one model family, spanning released checkpoints, multilingual supervised fine-tuning (SFT) at two scales, controlled SFT ablations, and three reinforcement-learning (RL) reward formulations. We jointly track correctness, language adherence, termination, and delivery efficiency. The dominant bottleneck
    
[^95]: Traverse：学习何时记忆、重置与重定向的长程网络搜索

    Traverse: Learning When to Remember, Reset, and Redirect for Long-Horizon Web Search

    [https://arxiv.org/abs/2609.37082](https://arxiv.org/abs/2609.37082)

    论文提出Traverse自主搜索框架，让智能体通过“评分标准—答案—验证”三状态自我管理搜索并配备封存记忆工具进行主动上下文管理，同时用仅训练上下文管理后末段的简单策略避免“封存崩塌”，使35B模型在BrowseComp上达到72.83。

    

    长程信息检索智能体常常会累积噪声或具有误导性的上下文，导致早期错误持续存在，并使恢复变得越来越困难。我们引入了一种自主搜索框架，其中智能体通过三种状态——评分标准、答案和验证——来管理自身的搜索过程。智能体首先定义有效答案的标准，在这些标准下进行搜索，然后独立验证结果，再决定是终止还是继续搜索。它还配备了一个“封存记忆”工具，以实现主动的上下文管理。然而，使用强化学习训练这种行为可能引发“封存崩塌”，导致训练不稳定，使智能体无法可靠地学会何时以及如何使用其记忆工具。我们通过一个简单的策略解决了这一问题：仅训练上下文管理之后的最后片段。我们的35B模型在BrowseComp上取得了72.83的成绩，超越了同类模型。

    arXiv:2609.37082v1 Announce Type: new  Abstract: Long-horizon information-seeking agents often accumulate noisy or misleading context, causing early mistakes to persist and making recovery increasingly difficult. We introduce an autonomous search harness in which the agent manages its own search process through three states: Rubric, Answer, and Verify. The agent first defines criteria for a valid answer, searches under these criteria, and then independently verifies the result before deciding whether to terminate or continue searching. It is further equipped with a Seal Memory tool that enables active context management. Training this behavior with reinforcement learning, however, can induce Seal Collapse, resulting in unstable training and preventing the agent from reliably learning when and how to use its memory tools. We solve this with a simple strategy that trains only the final segment after context management. Our 35B model achieves 72.83 on BrowseComp, outperforming comparable 
    
[^96]: 通过在线策略蒸馏从思考模式优势中学习

    Learning from Think-Mode Advantage via On-Policy Distillation

    [https://arxiv.org/abs/2609.37044](https://arxiv.org/abs/2609.37044)

    本文提出 ThinkOPD，通过在线策略蒸馏让模型学习思考模式的优势，并引入轨迹-回答分歧（TRD）度量，在回答层面自适应地路由教师监督信号，克服了统一共享思考轨迹蒸馏带来的师生不匹配问题。

    

    显式的中间推理赋予大型语言模型（LLM）更强的问题求解模式。我们研究如何通过在线策略蒸馏（OPD）从这种思考模式优势中学习。在线策略蒸馏保留了学生模型自身生成的轨迹，并在学生所到达的前缀处提供密集的词元级教师目标；特权推理仅在蒸馏阶段使用，而非在学生推理时使用。Uniform ThinkOPD 是一种自然的启用思考模式的在线策略蒸馏基线，它以一条共享的思考轨迹为条件固定教师模型，并对同组的每个学生回答进行统一蒸馏。尽管其前缀是在线策略的，但该思考轨迹未必与每个完整回答的推理路径相容：即使多个回答最终得到相同的结果，同一条特权轨迹也可能引发不同的师生差异。我们用“轨迹-回答分歧”（TRD）来概括这种交互作用，并在此基础上提出 ThinkOPD，它在回答层面路由监督信号，通过结合组相对奖励（原文摘要在此处被截断）……

    arXiv:2609.37044v1 Announce Type: new  Abstract: Explicit intermediate reasoning gives large language models (LLMs) a stronger problem-solving mode. We study learning from this think-mode advantage via on-policy distillation (OPD). OPD preserves student-generated trajectories and provides dense token-level teacher targets at student-visited prefixes. Privileged reasoning is used during distillation rather than student inference. Uniform ThinkOPD, a natural think-enabled OPD baseline, conditions a fixed teacher on one shared think trace and uniformly distills every sibling student response. Although its prefixes are on-policy, the trace need not follow a route compatible with every complete response: the same privileged trace can induce different teacher-student discrepancies even when responses reach the same outcome. We summarize this interaction with trace-response divergence (TRD) and introduce ThinkOPD, which routes supervision at the response level by combining group-relative rewa
    
[^97]: 在自然语言自编码器中选择信息量最大的词元

    Selecting The Most Informative Tokens in Natural Language Autoencoders

    [https://arxiv.org/abs/2609.37040](https://arxiv.org/abs/2609.37040)

    仅基于聊天结构训练的排序器无需模型前向传播即可为自然语言自编码器选出信息量最大的词元位置，仅解释5%的位置就能保留几乎全部威胁审计成功率，且预训练言语化器无需额外训练即可恢复模型经微调后隐藏的词语。

    

    自然语言自编码器将语言模型的内部激活转换为可读的解释。对每个词元位置都进行解释成本高昂，那么审计人员应该检查哪些位置来理解潜在威胁呢？我们通过对提示注入和词语隐藏场景下470万条解释的研究来探讨这一问题。我们将来自模型计算内部信号的排序方法与仅基于聊天结构训练的排序器进行比较。结果表明，聊天结构通常比计算信号能选出更相关的解释，且无需通过模型前向传播来选择位置。在四个数据集中的三个上，仅解释5%的词元位置即可保留解释全部位置时几乎完全的成功率（成功指获得关于威胁的解释），但收益大小随审计任务而异。此外，我们还证明预训练的言语化器无需额外训练，即可恢复模型通过微调学会隐藏的词语。

    arXiv:2609.37040v1 Announce Type: cross  Abstract: Natural language autoencoders translate a language model's internal activations into readable explanations. Explaining every token position is costly. Which positions should an auditor inspect to understand a potential threat? We study this question across $4.7$ million explanations on prompt injection and concealment. We compare signals from model computation with a ranker trained only on chat structure. Chat structure usually selects more relevant explanations than the computational signals, without requiring a model forward pass for position selection. On three of four datasets, explaining just $5\%$ of positions retains nearly all of the success rate from explaining every position, where success means obtaining an explanation about the threat. The benefit varies with the audit task. We also show that pretrained verbalizers recover words that models have learned to conceal through fine-tuning, without additional verbalizer training.
    
[^98]: LatCom：面向高效多智能体协作的跨智能体潜在压缩

    LatCom: Cross-Agent Latent Compression for Efficient Multi-Agent Collaboration

    [https://arxiv.org/abs/2609.37017](https://arxiv.org/abs/2609.37017)

    提出LatCom框架，通过将多个发送方的潜在表示压缩到固定数量的任务相关槽位中，解决了多智能体潜在协作中被忽视的跨智能体冗余问题，从而降低接收端上下文规模、计算开销与协作延迟。

    

    基于大语言模型的多智能体系统（MAS）越来越多地采用潜在空间协作，以避免自然语言通信带来的信息损失和重复编码-解码的开销。然而，直接转发所有发送方的潜在表示会使接收方的上下文规模随智能体数量和推理长度的增加而增长，从而提升计算量、内存占用和协作延迟。一个自然的解决方案是潜在压缩。但我们发现，现有潜在压缩方法未能解决跨智能体的冗余问题——这些方法通常对每个发送方独立压缩，然后将结果拼接在一起。我们提出了LatCom，一个面向高效多智能体潜在协作的跨智能体潜在压缩框架。LatCom将多个发送方的潜在表示映射到固定数量的对接收方可读且与任务相关的槽位中。它并非重建所有发送方的隐藏状态，而是以接收侧的任务效用来优化压缩后的潜在表示。

    arXiv:2609.37017v1 Announce Type: new  Abstract: LLM-based multi-agent systems (MAS) increasingly use latent collaboration to avoid the information loss and repeated encoding-decoding overhead of natural-language communication. However, directly forwarding all sender latents makes the receiver-side context scale with both the number of agents and the reasoning length, increasing computation, memory usage, and collaboration latency. A natural solution is latent compression. But we find that cross-agent redundancy remains unresolved in existing latent compression approaches, which typically compress each sender independently and then concatenate the results. We propose LatCom, a cross-agent latent compression framework for efficient multi-agent latent collaboration. LatCom maps multiple sender latents into a fixed number of receiver-readable and task-relevant slots. Rather than reconstructing all sender hidden states, it optimizes the compressed latents for receiver-side task utility. La
    
[^99]: CypherTurn：用于对话式Text-to-Cypher评估与自主性分歧的多轮基准测试

    CypherTurn: A Multi-Turn Benchmark for Conversational Text-to-Cypher Evaluation and the Autonomy Divergence

    [https://arxiv.org/abs/2609.36987](https://arxiv.org/abs/2609.36987)

    该论文提出了首个对话式Text-to-Cypher多轮基准测试CypherTurn（721个会话、5,927轮对话），发现最佳模型执行准确率仅64.7%、会话级正确率不足5%，并揭示前沿模型在自主运行时排行榜头部会显著重排的“自主性分歧”现象，表明错误管理是部分独立于原始生成能力的关键维度。

    

    图数据库越来越多地通过自然语言进行查询，然而现有的所有基准测试都只评估孤立的单轮查询，而非分析师实际工作中所使用的多轮会话。我们提出了CypherTurn，这是首个用于对话式Text-to-Cypher评估的基准测试，涵盖7个知识图谱和13种对话现象，共包含721个会话和5,927轮对话。我们在引导式预言机协议和完全自主的智能体协议下评估了15个模型，得出四项发现。第一，最佳模型仅达到64.7%的执行准确率，会话级正确率仍低于5%。第二，尽管整体排名相关性很强，前沿模型在自主运行下表现出排行榜头部的显著重排，我们将这一现象称为“自主性分歧”，它揭示了错误管理是一种部分独立于原始生成技能的能力。第三，扩展行动预……（原文摘要在此处截断）

    arXiv:2609.36987v1 Announce Type: new  Abstract: Graph databases are increasingly queried through natural language, yet every existing benchmark evaluates isolated single-turn queries rather than the multi-turn sessions through which analysts actually work. We introduce CypherTurn, the first benchmark for conversational Text-to-Cypher evaluation, comprising 721 sessions and 5,927 turns across 7 knowledge graphs and 13 conversational phenomena. We evaluate 15 models under a guided oracle protocol and a fully autonomous agentic protocol, yielding four findings. First, the best model reaches only 64.7% execution accuracy, and session-level correctness remains below 5%. Second, despite strong overall rank correlation, frontier models exhibit a consequential reordering of the top of the leaderboard under autonomous operation, a phenomenon we term the Autonomy Divergence, which reveals error-management as a partially independent capability from raw generation skill. Third, scaling action bud
    
[^100]: SRJudge：通过选择性推理增强大语言模型的细粒度知识概念标注能力

    SRJudge: Empowering Large Language Models with Selective Reasoning for Fine-Grained Knowledge Concept Tagging

    [https://arxiv.org/abs/2609.36982](https://arxiv.org/abs/2609.36982)

    本文提出三阶段SRJudge框架，先利用微调的小型语言模型将候选概念缩小到top-K范围，再让大语言模型进行选择性推理，从而解决了大语言模型在细粒度知识概念标注中因决策空间过大而难以选出正确概念的问题。

    

    知识概念标注旨在为教育内容分配特定的概念或主题标签，这对于传统教学和在线教学实践中的教育者和学习者都至关重要。最近的研究已经探索将大语言模型（LLMs）应用于该任务，并取得了良好的性能。然而，由于决策空间的高维性，大语言模型仍然难以从大规模候选集合中选择出正确的概念。在本文中，我们提出了一种新颖的三阶段“选择-推理-判断”（Select-Reason-Judge，SRJudge）框架，该框架赋予大语言模型用于细粒度知识概念标注的选择性推理能力。具体而言，阶段1中的选择器首先通过微调一个小型语言模型（SLM，例如BERT）将候选概念缩小到top-K候选列表，因为top-K预测在大多数情况下能够命中正确概念，从而缩小了正确候选的决策空间。接着，阶段2中的推理器采用……

    arXiv:2609.36982v1 Announce Type: cross  Abstract: Knowledge concept tagging aims to assign specific concept or topic labels to educational content, which is essential for both educators and learners in traditional and online teaching practices. Recent work has explored large language models (LLMs) for this task, achieving promising performance. However, LLMs still struggle to select the correct concept from a large-scale candidate set due to the high dimensionality of the decision space. In this paper, we propose a novel three-stage Select-Reason-Judge (SRJudge) framework, which empowers LLMs with selective reasoning capability for fine-grained knowledge concept tagging. Specifically, the Selector in Stage 1 first narrows the candidate concepts to a top-K shortlist by fine-tuning a small language model (SLM), e.g., BERT, since the top-$K$ predictions hit the correct concept in most cases, thereby reducing the decision space of correct candidates. Next, the Stage 2 Reasoner employs a l
    
[^101]: AMU：面向个性化对话的准入与记忆更新——基于小语言模型引导控制的结构化记忆

    AMU:Admission and Memory Update for Personalized Conversations---Structured Memory with SLM Guided Control

    [https://arxiv.org/abs/2609.36976](https://arxiv.org/abs/2609.36976)

    提出了AMU框架，利用小语言模型引导的结构化记忆控制机制，在写入阶段决定哪些信息进入长期记忆，并对记录进行单独存储、去重丢弃或融合更新，从而提升个性化对话的记忆质量。

    

    大语言模型（LLMs）已成为个性化助手的基础，但在长期交互中维护持久的用户记忆仍然具有挑战性。现有的记忆系统通常专注于存储、检索或整合，而记忆写入环节仍缺乏足够的控制：临时性请求、重复陈述以及过时的用户状态可能会进入记忆，并在之后的个性化过程中被检索出来。在本文中，我们提出了AMU：面向个性化对话的准入与记忆更新（Admission and Memory Update for Personalized Conversations），这是一个由小语言模型（SLM）引导的、针对写入时记忆控制的结构化框架。AMU使用结构化记忆过滤来决定哪些内容应该进入记忆，并通过SLM引导的存储管理来决定被准入的记录应该单独存储、作为重复项被丢弃、还是作为更新进行融合。我们在受控的记忆写入与检索环境中对AMU进行了评估。实验结果表明，AMU能够保持……

    arXiv:2609.36976v1 Announce Type: new  Abstract: Large language models (LLMs) have become the foundation of personalized assistants, but maintaining persistent user memory across long-term interactions remains challenging. Existing memory systems often focus on storage, retrieval, or consolidation, while memory writing remains less controlled: transient requests, duplicate statements, and outdated user states may enter memory and later be retrieved for personalization. In this paper, we present AMU: Admission and Memory Update for Personalized Conversations, an SLM-guided (Small language model guided) structured framework for writing-time memory control. AMU uses structured memory filtering to decide what should enter memory and SLM-guided storage management to determine whether an admitted record should be stored separately, discarded as a duplicate, or fused as an update. We evaluate AMU in a controlled memory writing and retrieval setting. Experimental results show that AMU maintain
    
[^102]: 重复而非长度：隔离神经文本转语音中的计数失败

    Repetition, Not Length: Isolating the Counting Failure in Neural Text-to-Speech

    [https://arxiv.org/abs/2609.36974](https://arxiv.org/abs/2609.36974)

    该论文通过句子数与词数匹配的对照实验证明，神经文本转语音模型的计数失败源于文本重复本身而非长度——控制句准确率高达94.3%而重复句仅18.2%，且该失败随重复周期性平滑增长、即使词不相邻也依然存在。

    

    文本转语音模型在处理将短语重复多次的文本时，会出现循环、截断和计数丢失的现象。我们证明，破坏这些模型的是重复本身，而非随之而来的文本长度。我们测试集中的每个重复句子都配有一个控制句，其句子数和词数相匹配，但没有任何词连续重复出现。来自三种架构的六个模型几乎完美地渲染了控制句，却在重复的孪生句上失败：在 k ≥ 6 时，两者的完全正确率分别为 94.3% 和 18.2%。这一差距在贪婪解码、重复惩罚参数扫描、四个独立语音识别器和 420 种分析规范下均未出现符号逆转；一个留出的第四种架构落在其预测差距的 1 个百分点之内，且两个非自回归基线之一也表现出相同的失败。通过改变文本的周期可以发现，失败随周期性平滑增长，即使没有任何词与自身相邻，仍有一半的失败存在。

    arXiv:2609.36974v1 Announce Type: new  Abstract: Text-to-speech models loop, truncate and lose count on text that repeats a phrase many times. We show that repetition itself is what breaks them, not the length that comes with it. Every repeated sentence in our test set is paired with a control of matched sentence and word count in which no word ever repeats back-to-back. Six models from three architectures render the controls almost perfectly and fail the repeated twins: 94.3% against 18.2% exactly right at k >= 6. The gap survives greedy decoding, repetition-penalty sweeps, four independent speech recognisers and 420 analysis specifications without once reversing sign; a held-out fourth architecture lands within a point of its predicted gap, and one of two non-autoregressive baselines shows the same failure. Varying the period of the text shows the failure grows smoothly with periodicity, half of it surviving when no word is adjacent to itself.
    
[^103]: Chinese-Jev：将系统一（System One）模型引入中文任务

    Chinese-Jev: Bringing System One Model to Chinese-Language Tasks

    [https://arxiv.org/abs/2609.36965](https://arxiv.org/abs/2609.36965)

    Chinese-Jev通过统一的数据处理与训练流程——将异构中文标注转换为候选选项的概率目标，并采用轻量级编码器骨干进行面向决策的训练——首次将高效的系统一模型成功引入中文决策任务。

    

    arXiv:2609.36965v1 公告类型： new 摘要：诸如Jev这样的系统一模型，为那些需要做出决策而非开放式回答的任务提供了一种相比生成式语言模型更高效的替代方案。然而，现有的Jev模型在中文决策任务上的准确率有限，这限制了其在通用及专门场景中的实用性。在本文中，我们提出了Chinese-Jev，这是一种通过统一的数据处理与训练流程来填补这一空白的系统一模型。我们的数据处理协议将异构的中文标注转换为候选选项上的概率目标，从而实现了跨领域、跨题型的统一训练形式。为了实现高效推理，Chinese-Jev采用轻量级的仅编码器骨干网络进行文本编码，并通过面向决策的训练学习对候选答案进行打分。为解决预训练分布与下游中文场景之间的不对齐问题，我们首先……（原文摘要在此处截断）

    arXiv:2609.36965v1 Announce Type: new  Abstract: System One models such as Jev offer an efficient alternative to generative language models for tasks that require decisions rather than open-ended responses. However, existing Jev models exhibit limited Chinese-language decision accuracy, restricting their utility in both general and specialized settings. In this paper, we introduce Chinese-Jev, a System One model that addresses this gap through a unified data processing and training pipeline. Our data processing protocol converts heterogeneous Chinese-language annotations into probability targets over candidate options, enabling a shared training formulation across domains and question formats. To enable efficient inference, Chinese-Jev adopts a lightweight encoder-only backbone for text encoding and learns to score candidate answers through decision-oriented training. To address the misalignment between the pre-training distribution and downstream Chinese-language scenarios, we first t
    
[^104]: VStress：面向重复验证器的相关性感知审计与自适应预算分配

    VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers

    [https://arxiv.org/abs/2609.36958](https://arxiv.org/abs/2609.36958)

    本文提出VStress可审计重放契约与VStress-CA相关性感知分配策略，通过估计重复验证器调用的条件边际信息来自适应分配预算并在信息不足时停止调用，从而在固定预算下以更低的调用成本获得更高的验证平衡准确率，同时通过受控审计刻画了该机制在极端损坏率下的适用边界。

    

    重复验证器调用只有在贡献条件信息时才有价值。我们提出了VStress——一种可审计的重放契约，以及VStress-CA——一种相关性感知的分配策略。该策略在密封的校准集上估计未查询验证器的条件边际信息，对不确定性进行折扣，按调用成本归一化，并在下一次调用不具信息量时停止或弃权。控制器在接入干净预言机之前冻结其决策和成本账本；当依赖性偏移警报触发时，禁用通道偏好并回退到精确停止机制。受控审计给出了该机制的适用边界：在35%对称损坏下，多数投票-5将平衡准确率从0.6578提升至0.7739；而在65%损坏下则损失0.1226个百分点。在固定预算匹配比较中，广度、冗余和自适应分配策略分别取得0.6048、0.6375和0.6538的平衡准确率，每项平均调用3.4216次，RLVR分数为0.64。

    arXiv:2609.36958v1 Announce Type: cross  Abstract: Repeated verifier calls are useful only when they contribute conditional information. We introduce VStress, an auditable replay contract, and VStress-CA, a correlation-aware allocation policy that estimates the conditional marginal information of an unqueried verifier on a sealed calibration split, discounts uncertainty, normalizes by call cost, and stops or abstains when the next call is not informative. The controller freezes its decision and cost ledger before joining the clean oracle; a dependence-shift alarm disables channel preference and falls back to exact-stop. The controlled audit gives the mechanism boundary: at 35% symmetric corruption, majority-5 improves balanced accuracy from 0.6578 to 0.7739, whereas at 65% it loses 0.1226 points. In the matched fixed-budget comparison, breadth, redundancy, and adaptive allocation obtain balanced accuracies 0.6048, 0.6375, and 0.6538, with 3.4216 calls per item and an RLVR score of 0.64
    
[^105]: 冷却采样器，而非学习器：采样温度能够移动重要性校正GRPO的过时悬崖

    Cool the Sampler, Not the Learner: Sampling Temperature Moves the Staleness Cliff of Importance-Corrected GRPO

    [https://arxiv.org/abs/2609.36953](https://arxiv.org/abs/2609.36953)

    该论文发现重要性校正GRPO在采样器长时间不刷新时会遭遇性能“悬崖”式崩溃，并提出“解耦冷却”方法——仅将采样端温度降至0.8而学习端保持温度1，在不改变学习器目标函数的前提下移动了过时悬崖，使每192步才刷新一次采样器仍能保持稳定学习。

    

    在生产级的语言模型强化学习中，采样器会落后于学习器，并使用截断的重要性权重来修复由此产生的失配。我们探究在这种校正下，采样器可以在不刷新的情况下维持多久，并发现了一个“悬崖”：在Qwen2.5-Math-1.5B和GSM8K上，每192次更新刷新一次的重要性校正GRPO在180步内学习良好，但在刷新到来之前，在全部三个数据种子中都出现严重退化。已发表的针对过时性的补救措施作用于更新端；而我们则作用于采样器端。解耦冷却以温度0.8进行采样，而学习器、参考模型和重要性权重均保持温度1，行为概率从带温度的分布中记录，因此学习器的目标函数保持不变。所有相应的冷却运行都是稳定的，且更长的刷新间隔保留了较短间隔所能提供的收益：在相同的更新预算下，每192步刷新一次的冷却采样器达到了与……（摘要在此处截断）

    arXiv:2609.36953v1 Announce Type: cross  Abstract: Production RL for language models lets the sampler fall behind the learner and repairs the resulting mismatch with a truncated importance weight. We ask how long the sampler can go without a refresh under that correction, and find a cliff: on Qwen2.5-Math-1.5B and GSM8K, importance-corrected GRPO refreshed every 192 updates learns well for 180 steps and then degrades severely in all three data seeds before the refresh arrives. Published remedies for staleness act on the update; we act on the sampler instead. Decoupled cooling draws samples at temperature 0.8 while the learner, the reference model and the importance weights stay at temperature 1, with the behaviour probability recorded from the tempered distribution, so the learner's objective is unchanged. All corresponding cooled runs are stable, and the longer interval keeps what the short one delivered: at the same update budget, a cooled sampler refreshed every 192 steps matches an
    
[^106]: ER-JEPA：经验回放改进语言模型中的联合嵌入预测学习

    ER-JEPA: Experience Replay Improves Joint-Embedding Predictive Learning in Language Models

    [https://arxiv.org/abs/2609.36952](https://arxiv.org/abs/2609.36952)

    ER-JEPA在LLM-JEPA中引入经验回放机制，通过存储并检索历史训练样本对提供额外监督，使模型同时从当前批次和记忆库中学习，在多个数据集上一致优于LLM-JEPA。

    

    大型语言模型（LLMs）在词元级生成方面表现出色，但可能学到不良的抽象语义并缺乏全面的感知能力。LLM-JEPA通过联合嵌入预测架构（JEPA）对同一底层知识的不同视图进行对齐来缓解这一问题。然而，强对齐并不一定带来准确、稳定的预测。为解决这一问题，我们提出了ER-JEPA，它在LLM-JEPA的基础上增加了一条情景回放路径。ER-JEPA将训练样本对存储在记忆库中，并在每一步存储和检索相关数据以提供额外的监督信号。这使得模型能够同时从当前批次和已存储的训练样本对中学习，为词元预测和表示对齐提供额外监督。在多个数据集（NL-RX、GSM8K、Spider和NQ-Open）上的实验表明，ER-JEPA始终优于LLM-JEPA。

    arXiv:2609.36952v1 Announce Type: cross  Abstract: Large language models (LLMs) excel at token-level generation but may learn undesirable abstract semantics and lack comprehensive perception. LLM-JEPA mitigates this by aligning different views of the same underlying knowledge via a joint-embedding predictive architecture (JEPA). However, strong alignment does not necessarily lead to accurate, stable predictions. To address this, we propose ER-JEPA, which adds an episodic replay path to LLM-JEPA. ER-JEPA stores training pairs in a memory. At each step, it stores and retrieves relevant data to provide additional supervision. This enables learning from both the current batch and stored training pairs, providing additional supervision for token prediction and representation alignment. Experiments across multiple datasets (NL-RX, GSM8K, Spider, and NQ-Open) demonstrate that ER-JEPA consistently outperforms LLM-JEPA.
    
[^107]: CoEM：基于证据提交记忆的长上下文推理增强方法

    CoEM: Empowering Long-Context Reasoning with Commit-on-Evidence Memory

    [https://arxiv.org/abs/2609.36935](https://arxiv.org/abs/2609.36935)

    CoEM 提出了一种“证据提交记忆”机制：在固定上下文预算下先逐字保留待定证据，再由学习到的策略根据新到来的上下文决定将其提交、继续保留或丢弃，从而避免过早压缩导致的关键信息丢失，提升长上下文推理性能。

    

    长上下文推理对于复杂且长时程的任务至关重要，然而大语言模型（LLM）的性能会随着上下文长度的增加而下降。近期的解决方法是将输入逐块处理，同时在模型上下文中维护容量有限的文本记忆。然而，过早的信息压缩可能会丢弃对后续推理至关重要的关键细节。在本文中，我们提出了证据提交记忆，它学习何时将源证据转换为紧凑的记忆事实。具体而言，在固定的上下文-记忆预算下，CoEM 将潜在有用的源文本片段逐字保存在一个待定集合中，使后续上下文有机会澄清其相关性，然后再进行不可逆的压缩。随着新上下文的到来，一个学习到的策略会重新审视每个待定片段，并决定是将其提升为已提交记忆、继续保留以待进一步考量，还是将其丢弃。一个冻结的验证器确保……

    arXiv:2609.36935v1 Announce Type: new  Abstract: Long-context reasoning is essential for complex and long-horizon tasks, yet the performance of large language models (LLMs) degrades as context length increases. Recent approaches address this by processing input chunk by chunk while maintaining a bounded textual memory in model context. However, premature information compression can discard critical details essential for subsequent reasoning. In this paper, we introduce Commit-on-Evidence Memory (CoEM), which learns when to convert source evidence into compact memory facts. Specifically, under a fixed context-memory budget, CoEM preserves potentially useful source excerpts verbatim in a pending set, allowing subsequent context to clarify their relevance before irreversible compression. As new context arrives, a learned policy revisits each pending excerpt and decides whether to promote it to the committed memory, retain it for further consideration, or discard it. A frozen verifier ensu
    
[^108]: 为模型标注日期：系统提示中隐藏的日期影响大语言模型评估

    Dating the Model: Hidden Dates in System Prompts Affect LLM Evaluation

    [https://arxiv.org/abs/2609.36931](https://arxiv.org/abs/2609.36931)

    研究发现系统提示中隐藏注入的当前日期会显著影响大语言模型的评估结果与排名（性能差异最高达14%），其影响超过批大小等其他非确定性因素，且思维链提示反而会放大这一效应。

    

    可重复性对科学研究至关重要，然而先前的研究表明，大语言模型（LLM）的输出会因硬件和批处理方式的不同而变化。我们发现了一个被忽视的因素：系统提示中隐藏注入的当前日期，这一因素用户无法控制，且每天都在变化。在9个近期的大语言模型和6个数据集上，涵盖多选题问答（MCQA）、数学推理、代码生成和机器翻译等任务，性能仅随当前日期而波动，其中MCQA的差异高达6%，数学推理高达14%，代码生成高达7%，机器翻译差异达2.84 BLEU。模型排名也会随之变化，进而影响排行榜结果。这种日期效应超过了其他非确定性来源，例如批大小和数值精度。标准的提示技术——思维链和少样本提示——并不能降低这种敏感性；思维链甚至会放大这一效应。我们的发现强调，需要谨慎设计评估协议以确保可重复性。

    arXiv:2609.36931v1 Announce Type: new  Abstract: Reproducibility is essential for scientific research, yet prior work shows that LLM outputs vary with hardware and batching. We identify an overlooked factor: the hidden injection of the current date into system prompts, which users cannot control and which changes every day. Across 9 recent LLMs and 6 datasets spanning multiple-choice QA (MCQA), math reasoning, code generation, and machine translation, performance varies solely with the current date, with deltas of up to 6% on MCQA, 14% on math reasoning, 7% on code generation, and 2.84 BLEU on machine translation. Model rankings also shift, affecting leaderboards. This date effect exceeds other sources of non-determinism, such as batch size and numerical precision. Standard prompting techniques -- chain-of-thought and few-shot prompting -- do not reduce the sensitivity; chain-of-thought even amplifies it. Our findings underscore the need for careful evaluation protocols to ensure repro
    
[^109]: 伊比利亚语言自动语音识别工具的基准测试

    Benchmarking Automatic Speech Recognition Tools for Iberian Languages

    [https://arxiv.org/abs/2609.36920](https://arxiv.org/abs/2609.36920)

    该研究对十一个语音识别系统在五种伊比利亚语言上进行了迄今最全面的基准测试，发现准确性、效率和语言覆盖之间存在明显权衡，低资源语言（尤其是巴斯克语）性能显著下降，且多数系统存在性别偏见，为多语言ASR的实际应用提供了实用指导。

    

    针对伊比利亚语言的自动语音识别（ASR）的全面评估仍然有限，低资源语言、偏见和效率权衡等方面的研究尚不充分。我们对十一个系统进行了基准测试，其中包括十个开放权重模型和一个商业API，覆盖五种伊比利亚语言（巴斯克语、加泰罗尼亚语、加利西亚语、葡萄牙语、西班牙语），并以德语和土耳其语作为对照。评估使用了一个85小时的数据集，涵盖朗读语音、广播媒体和有声读物，通过词错误率（WER）和实时因子（RTF/RTFx）评估准确性和效率。结果显示没有任何单一模型占据绝对优势：准确性、效率和语言覆盖范围之间存在明显的权衡。低资源语言（尤其是巴斯克语）的性能显著下降，凸显了训练数据覆盖范围的重要作用。我们观察到大多数系统中存在一致的性别差异，凸显了多语言ASR中的公平性挑战。总体而言，该基准测试为实际应用提供了实用指导。

    arXiv:2609.36920v1 Announce Type: new  Abstract: Comprehensive evaluations of automatic speech recognition (ASR) for Iberian languages remain limited, and low-resource languages, biases, and efficiency trade-offs are underexplored. We benchmark eleven systems, ten open-weight models and one commercial API, across five Iberian languages (Basque, Catalan, Galician, Portuguese, Spanish), with German and Turkish as controls. Evaluation uses an 85-hour dataset covering read speech, broadcast media, and audiobooks, assessing accuracy and efficiency via word error rate (WER) and real-time factors (RTF/RTFx). Results show no single model dominates: accuracy, efficiency, and language coverage present clear trade-offs. Low-resource languages, especially Basque, degrade significantly, highlighting the role of training coverage. We observe consistent sex disparities across most systems, highlighting fairness challenges in multilingual ASR. Overall, the benchmark provides practical guidance for rea
    
[^110]: 语言模型能否学会预测股票价格

    Can Language Models Learn to Forecast Stock Prices

    [https://arxiv.org/abs/2609.36914](https://arxiv.org/abs/2609.36914)

    该论文在按时间顺序推进的股票价格沙盒中，通过对 Qwen3-4B 进行 SFT 与 PPO 后训练，探索语言模型能否学会利用价格、成交量等信息预测股票未来收益。

    

    后训练已被证明能显著提升语言模型在具有可验证结果任务上的表现，包括数学推理、软件工程和计算机使用。然而，同样的方法能否改善金融市场的预测则远非明确。与具有可验证结果的任务相比，不仅实际收益充满噪声，甚至什么构成进行有效预测所需的相关信息集也并非先验可知：模型必须自行决定收集哪些观察信息，然后在结果揭晓之前给出数值判断。我们在一个按时间顺序推进的股票价格沙盒中研究这一问题，其中语言模型收集价格、成交量、相对表现以及市场背景等证据，并预测未来收益。我们首先基于工具使用演示对 Qwen3-4B 进行监督微调（SFT），随后使用近端策略优化（PPO）进行训练，其终端奖励由……

    arXiv:2609.36914v1 Announce Type: new  Abstract: Post-training has been shown to significantly improve language models' performance on tasks with verifiable outcomes, including mathematical reasoning, software engineering, and computer use. However, whether the same approach can improve forecasting in financial markets is much less clear. Compared with tasks with verifiable outcomes, not only are realized returns noisy, but even what constitutes a relevant information set for making effective predictions is not obvious a priori: the model must decide which observations to gather and then commit to a numerical judgment before the outcome is known. We study this question in a chronological stock-price sandbox, where a language model gathers price, volume, relative-performance, and market-context evidence and predicts a future return. We post-train Qwen3-4B with supervised fine-tuning (SFT) on tool-use demonstrations, then proximal policy optimization (PPO) with a terminal reward given by
    
[^111]: BaLEEN：基于潜在编码实体的偏置方法实现上下文感知语音识别

    BaLEEN: Biasing with Latent Encoded Entities for Context-Aware ASR

    [https://arxiv.org/abs/2609.36913](https://arxiv.org/abs/2609.36913)

    BaLEEN提出了一种基于超网络的轻量级框架，利用预训练语言模型编码上下文关键词并将偏置向量注入完全冻结的ASR编码器中，实现了无需微调、推理零开销的即插即用上下文自适应语音识别。

    

    在自动语音识别（ASR）中，转录特定领域的实体和罕见专有名词仍然是一个重大挑战。本文提出了BaLEEN（Biasing with Latent Encoded Entities，基于潜在编码实体的偏置），这是一个轻量级的、基于超网络的框架，可在不微调底层ASR模型的情况下实现动态上下文自适应。BaLEEN使用预训练语言模型对可变长度的上下文关键词进行编码，通过Perceiver瓶颈将其压缩为固定序列的潜在向量，并将依赖于上下文的偏置向量直接注入ASR模型的中间编码器表示中。由于语言模型和骨干ASR模型在训练期间完全保持冻结，BaLEEN作为一个即插即用的适配器运行，当上下文偏置被预先计算时，在推理阶段不会产生任何额外计算开销。我们在基于CTC的ASR模型上，使用来自维基百科的带有命名实体标注的语料库对该方法进行了评估。

    arXiv:2609.36913v1 Announce Type: cross  Abstract: Transcribing domain-specific entities and rare proper nouns remains a major challenge in automatic speech recognition (ASR). In this paper, we propose BaLEEN (Biasing with Latent Encoded Entities), a lightweight, hypernetwork-based framework for dynamic contextual adaptation without fine-tuning the underlying ASR model. BaLEEN encodes variable-length contextual keywords using a pretrained language model, compresses them into a fixed sequence of latent vectors via a Perceiver bottleneck, and injects context-dependent bias vectors directly into the intermediate encoder representations of the ASR model. Because both the language model and the backbone ASR model remain entirely frozen during training, BaLEEN operates as a plug-and-play adapter that incurs zero computational overhead at inference time when context biases are precomputed. We evaluate our method on a CTC-based ASR model using a Wikipedia-derived corpus with annotated named en
    
[^112]: MultiTalk：将全双工语音模型扩展至长时、多方、双语对话

    MultiTalk: Scaling Full-Duplex Speech Models to Long, Multi-Party, Bilingual Conversation

    [https://arxiv.org/abs/2609.36903](https://arxiv.org/abs/2609.36903)

    该论文发布了5.76万小时的多方双语合成语音训练数据集 MultiTalkPT，并在英语和中文上沿长时域与多方交互两个维度扩展了 Moshi 全双工语音范式，使单一模型能够应对长时多方对话场景。

    

    端到端全双工语音模型已使开源的机器对话更接近人类式交互，但现有系统仍在两个相互交织的维度上存在局限：长上下文的鲁棒性与多方交互能力。现实场景（如会议、小组课程和社交机器人接待）要求单一模型能够在长时间内跟踪、理解上下文并响应多个说话人。该领域的进展受到数据和评估两方面的制约：开放的多方语音语料库规模仍然很小，且并非为编解码器帧级的全双工建模而设计；同时，现有的长音频基准主要针对被动聆听，而语音到语音的基准大多篇幅短小且仅限于双人对话。我们在英语和中文上，沿长时域与多方交互两个维度共同扩展了 Moshi 范式。首先，我们发布了5.76万小时的合成训练数据（MultiTalkPT）……

    arXiv:2609.36903v1 Announce Type: cross  Abstract: End-to-end full-duplex speech models have brought open-source machine conversation closer to human-like interaction, yet existing systems remain limited in two intertwined dimensions: long-context robustness and multi-party interaction. Real-world scenarios such as meetings, group lessons, and social-robot reception require a single model to track, contextualize, and respond to multiple speakers over extended durations. Progress is constrained by both data and evaluation: open multi-party speech corpora remain small and are not designed for codec-frame-level full-duplex modeling, while existing long-audio benchmarks focus on passive listening and speech-to-speech benchmarks are mostly short and dyadic. We extend the Moshi paradigm jointly along the long-horizon and multi-party axes in English and Chinese. First, we release 57.6k hours of synthetic training data ($\href{https://huggingface.co/datasets/MultiTalk/MultiTalkPT}{MultiTalkPT}
    
[^113]: RAEGNet：面向危害感知多模态假新闻检测的关系感知证据图网络

    RAEGNet: Relation-Aware Evidence Graph Network for Harm-Aware Multimodal Fake News Detection

    [https://arxiv.org/abs/2609.36902](https://arxiv.org/abs/2609.36902)

    提出关系感知证据图网络RAEGNet，通过事件级证据检索与融合新闻-证据立场关系的有向图建模，联合检测假新闻的真实性及其潜在危害。

    

    现有的多模态假新闻检测方法通常引入外部信息来辅助检测。然而，其中大多数方法依赖于实体级检索，因此容易引入与事件无关的噪声。同时，现有方法主要关注提升整体性能，而未考虑不同假新闻实例所造成危害程度的差异。为了解决这些局限性，我们设计了事件级证据检索框架（ELERF），并提出了关系感知证据图网络（RAEGNet）。ELERF 基于新闻条目的完整事件语义来检索外部证据。RAEGNet 构建了一个融合新闻-证据立场关系和证据-证据交互关系的有向图，并引入条件危害分支来联合建模新闻的真实性与潜在危害。实验结果表明，RAEGNet 在多项指标上优于多种基线方法……

    arXiv:2609.36902v1 Announce Type: new  Abstract: Existing multimodal fake news detection methods often introduce external information to assist detection. However, most of them rely on entity-level retrieval and are therefore prone to introducing event-irrelevant noise. Meanwhile, existing methods mainly focus on improving overall performance and do not account for differences in the degree of harm posed by different instances of fake news. To address these limitations, we design an Event-Level Evidence Retrieval Framework (ELERF) and propose a Relation-Aware Evidence Graph Network (RAEGNet). ELERF retrieves external evidence based on the complete event semantics of a news item. RAEGNet constructs a directed graph that incorporates news-evidence stance relations and evidence-evidence interaction relations, and introduces a conditional-harm branch to jointly model authenticity and potential harm. Experimental results demonstrate that RAEGNet outperforms multiple baseline methods across 
    
[^114]: 面向详细图像描述的动量耦合量规自适应方法

    Momentum-Coupled Rubric Adaptation for Detailed Image Captioning

    [https://arxiv.org/abs/2609.36893](https://arxiv.org/abs/2609.36893)

    提出 MoCo Rubric 两阶段框架，通过角色条件化共享参数和动量耦合机制协调描述生成、量规构建与评判三个环节，解决了现有基于量规的强化学习方法中角色解释不一致及分阶段优化脱节的问题，从而提升详细图像描述的质量。

    

    详细图像描述要求对细粒度视觉内容进行准确且全面的描述，而描述质量涉及事实准确性、信息覆盖度和清晰度等多个维度。与主要依赖高质量监督信号或整体奖励的传统方法相比，基于量规的强化学习将这些要求分解为明确的评判标准，并提供有针对性的结构化反馈。然而，现有方法通常使用相互独立的模型来分别完成描述生成、量规构建和评判，这可能导致不同角色之间的解释不一致。一些动态量规方法在保持评判器固定的同时交替更新描述策略和量规生成器，但这种分阶段优化仍可能使量规构建和评判与策略优化脱节。我们提出 MoCo Rubric，这是一个协调上述角色的两阶段框架。首先，采用基于角色条件的共享参数多任务（摘要内容在此处截断）。

    arXiv:2609.36893v1 Announce Type: new  Abstract: Detailed image captioning requires accurate and comprehensive descriptions of fine-grained visual content, yet caption quality spans factual accuracy, information coverage, and clarity. Compared with conventional methods that rely mainly on high-quality supervision or holistic rewards, rubric-based reinforcement learning decomposes these requirements into explicit criteria and provides targeted, structured feedback. However, existing methods often use separate models for caption generation, rubric construction, and judging, which may lead to inconsistent interpretations across roles. Some dynamic rubric methods alternate updates between the caption policy and rubric generator while keeping the judge fixed, but staged optimization may still leave rubric construction and judging out of step with policy optimization. We propose MoCo Rubric, a two-stage framework that coordinates these roles. First, role-conditioned, shared-parameter multi-t
    
[^115]: 将框架演化视为学习：自我提升型个人代理的逼近、泛化与优化极限

    Harness Evolution as Learning: Approximation, Generalization, and Optimization Limits of Self-Improving Personal Agents

    [https://arxiv.org/abs/2609.36892](https://arxiv.org/abs/2609.36892)

    该论文以“将框架演化视为学习”的视角，通过互补的实证与理论分析，系统研究了个人代理框架的架构、规模与自演化算法三大核心问题，揭示了自我提升型个人代理在逼近、泛化与优化方面的极限。

    

    随着大型语言模型（LLM）能力的不断提升，如何将其能力转化为有用的行为正受到越来越多的关注。个人代理将这一问题带入了日常场景，在这些场景中，模型被期望服务于个人用户，并持续适应用户的偏好。在底层模型保持固定的情况下，这种适应依赖于框架工程：设计并演化管理上下文、记忆、工具和执行的周边层。尽管进展迅速，但决定有效框架演化的关键因素仍未得到充分理解。为缩小这一差距，我们通过互补的实证与理论分析，研究了关于框架架构、框架规模和自演化算法的三个核心问题。在实证方面，我们引入了一个面向偏好的基准测试，并系统性地刻画了个人代理相关能力与局限（摘要在此处被截断）。

    arXiv:2609.36892v1 Announce Type: new  Abstract: As the capabilities of large language models (LLMs) continue to advance, increasing attention is turning to how to translate their abilities into useful behavior. Personal agents bring this question into everyday settings, where models are expected to serve individual users and continually adapt to their preferences. With the underlying model held fixed, such adaptation relies on harness engineering: designing and evolving the surrounding layer that manages context, memory, tools, and execution. Despite rapid progress, the factors governing effective harness evolution remain insufficiently understood. To narrow this gap, we investigate three central questions concerning harness architecture, harness scale, and self-evolution algorithms through complementary empirical and theoretical analyses. Empirically, we introduce a preference-oriented benchmark and systematically characterize the capabilities and limitations of personal agents assoc
    
[^116]: 生成式AI时代下多模态假新闻检测的再思考

    Rethinking Multimodal Fake News Detection in the Generative AI Era

    [https://arxiv.org/abs/2609.36850](https://arxiv.org/abs/2609.36850)

    该论文构建了面向生成内容场景的多模态假新闻检测数据集Weibo26，并提出生成性感知层次化推理（GAHR）框架，通过全局判断与局部校正相结合，弥合了假新闻检测与AIGC检测之间的割裂。

    

    生成式内容正日益融入新闻的生产与传播过程，使假新闻从人工捏造或简单篡改的材料，转变为原生内容与生成内容共同参与的复杂形态。现有的多模态假新闻检测研究主要聚焦于真实性评估，很少刻画生成性差异如何影响证据的可靠性。相比之下，AIGC检测主要判断内容是否由生成式模型生成或修改，但其本身并不能确定底层新闻事件是否真实。为了弥合这两类任务在数据和评估上的割裂，我们构建了Weibo26，一个面向生成内容场景的多模态假新闻检测数据集。在此基础上，我们提出了生成性感知层次化推理（GAHR）框架，该框架将全局判断与局部校正相结合，使生成性信息…

    arXiv:2609.36850v1 Announce Type: new  Abstract: Generative content is increasingly entering the production and dissemination of news, transforming fake news from manually fabricated or simply manipulated material into complex forms in which native and generated content jointly participate. Existing multimodal fake news detection research primarily focuses on veracity assessment and rarely characterizes how generativity differences affect the reliability of evidence. In contrast, AIGC detection primarily determines whether content is generated or modified by generative models, but it does not by itself establish whether the underlying news event is true. To bridge the separation between these tasks in data and evaluation, we construct Weibo26, a multimodal fake news detection dataset for generative-content scenarios. On this basis, we propose the Generativity-Aware Hierarchical Reasoning (GAHR) framework, which combines global judgment with local correction so that generativity informa
    
[^117]: 在线策略视觉证据蒸馏

    On-Policy Visual Evidence Distillation

    [https://arxiv.org/abs/2609.36838](https://arxiv.org/abs/2609.36838)

    ReVuE提出了一种面向视觉智能体的在线策略蒸馏方法，通过比较学生生成的多条交互轨迹，针对证据获取、读取和答案落地等不同失败阶段提供有针对性的反思与纠正。

    

    视觉智能体通过将推理与图像操作交替进行来解决问题，而在线策略蒸馏通过强教师模型对学生生成的交互轨迹提供指导。然而，图像操作会改变后续推理可用的证据，因此证据获取、证据读取或答案落地等环节中的局部错误可能沿轨迹传播，最终导致错误答案。现有的多模态在线策略蒸馏方法主要通过构建或对比原始图像的辅助视图来强化监督，而没有显式建模学生动作、由此产生的观察结果与后续推理之间的联系，这限制了它们针对不同失败阶段提供相应纠正的能力。我们提出了ReVuE（视觉证据反思），这是一种面向视觉智能体的在线策略蒸馏方法。ReVuE通过比较学生针对同一问题生成的多条轨迹……（原文摘要在此处截断）

    arXiv:2609.36838v1 Announce Type: cross  Abstract: Visual agents solve problems by interleaving reasoning with image operations, and on-policy distillation (OPD) provides guidance from a strong teacher on student-generated interaction trajectories. However, image operations change the evidence available for subsequent reasoning, so local errors in evidence acquisition (Acquire), reading (Read), or answer grounding (Ground) can propagate through the trajectory and lead to incorrect answers. Existing multimodal OPD methods primarily construct or contrast auxiliary views of the original image to strengthen supervision, without explicitly modeling the connections between student actions, resulting observations, and subsequent reasoning. This limits their ability to provide corrections tailored to different failure stages. We introduce Reflection on Visual Evidence (ReVuE), an on-policy distillation method for visual agents. ReVuE compares multiple student-generated trajectories for the sam
    
[^118]: CorrGRPO：面向多奖励学习的相关性归一化GRPO

    CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning

    [https://arxiv.org/abs/2609.36820](https://arxiv.org/abs/2609.36820)

    本文提出CorrGRPO，通过将奖励间协方差归一化为皮尔逊相关系数来改进GRPO的多奖励学习，从而避免大尺度奖励主导归一化过程并抑制小尺度奖励的信号。

    

    组相对策略优化（GRPO）被广泛用于训练推理型语言模型，它通过对同一提示（prompt）的多次采样（rollout）中的奖励进行中心化和归一化来计算优势函数。对于多个奖励，GRPO将各奖励分量求和，并用总奖励的组内标准差进行归一化，其对应的方差等于所有奖励两两之间协方差的总和。在中心化总奖励固定的情况下，聚合协方差越大，产生的优势越小，反之亦然，这使得更新幅度能够自适应地随奖励相关性变化。然而，具有较大尺度的相关奖励可能会主导这一归一化过程，从而抑制来自较小尺度奖励的信号。我们提出了相关性归一化GRPO（CorrGRPO），它将两两协方差归一化为皮尔逊相关系数。CorrGRPO在保持中心化总奖励不变的同时，平衡了不同尺度的奖励对基于相关性的（注：原文摘要在此处截断）

    arXiv:2609.36820v1 Announce Type: cross  Abstract: Group Relative Policy Optimization (GRPO) is widely used to train reasoning language models, where it computes advantages by centering and normalizing rewards across rollouts of the same prompt. For multiple rewards, GRPO sums the reward components and normalizes the total reward by its within-group standard deviation. The corresponding variance equals the sum of all pairwise reward covariances. For a fixed centered reward, larger aggregate covariance produces smaller advantages, and vice versa, allowing update magnitudes to adapt to reward dependence. However, correlated rewards with large scales can dominate this normalization and suppress signals from smaller-scale rewards. We propose Correlation-Normalized GRPO (CorrGRPO), which normalizes pairwise covariances into Pearson correlation coefficients. CorrGRPO keeps the centered total reward unchanged while balancing the influence of differently scaled rewards on the correlation-based
    
[^119]: VAA-CSEC：面向中文语义纠错的投票引导优势分配方法

    VAA-CSEC: Vote-guided Advantage Allocation for Chinese Semantic Error Correction

    [https://arxiv.org/abs/2609.36804](https://arxiv.org/abs/2609.36804)

    该论文提出VAA-CSEC多阶段框架，通过结合思维链蒸馏、监督微调、强化学习与自洽性解码，并设计对齐最小编辑原则的任务奖励函数以及重新分配GRPO优势的组级相对策略优化（GLPO），有效解决了中文语义纠错中的过度纠正问题。

    

    中文语义纠错（CSEC）旨在纠正中文文本中的语义错误，这类错误通常比拼写和语法错误更加隐蔽和复杂，但目前相关研究仍相对匮乏。现有的基于大语言模型（LLM）的方法在该任务中面临两个反复出现的障碍：一是过度纠正问题，二是思维链推理与自洽性解码之间的交互不清晰，导致CoT推理带来的收益无法可靠地传递到最终的纠错结果中。我们提出了面向中文语义纠错的投票引导优势分配框架（VAA-CSEC），这是一个结合了思维链蒸馏、监督微调（SFT）、强化学习（RL）和自洽性解码的多阶段框架。在强化学习阶段，我们设计了一个与CSEC最小编辑原则直接对齐的任务特定奖励函数。我们进一步提出了组级相对策略优化（GLPO），根据个体之间的差距重新分配GRPO的优势。

    arXiv:2609.36804v1 Announce Type: cross  Abstract: Chinese Semantic Error Correction (CSEC) targets semantic errors in Chinese text, which are typically more subtle and complex than spelling and grammatical errors but remain relatively underexplored. Existing LLM-based approaches face two recurring obstacles in this task: over-correction, and unclear interaction between Chain-of-Thought (CoT) reasoning and self-consistency decoding, such that the benefits brought by CoT cannot be reliably transferred to final corrections. We propose Vote-guided Advantage Allocation for CSEC (VAA-CSEC), a multi-stage framework that combines CoT distillation, Supervised Fine-Tuning (SFT), Reinforcement Learning (RL) and self-consistency decoding. During RL, we design a task-specific reward function that directly aligned with the minimal-editing principle of CSEC. We further introduce Group-Level Relative Policy Optimization (GLPO), which reallocates GRPO advantages according to the margin between individ
    
[^120]: 看见本应听见的：诊断与修复全模态大语言模型中的跨模态捷径

    Seeing What Should Be Heard: Diagnosing and Repairing Cross-Modal Shortcuts in Omni-Modal LLMs

    [https://arxiv.org/abs/2609.36798](https://arxiv.org/abs/2609.36798)

    该研究揭示了全模态大语言模型在回答音频相关问题时过度依赖图像的跨模态捷径，提出因子化模态诊断方法以隔离各模态的因果贡献，并发现该捷径在监督微调和强化学习后训练中持续存在甚至被放大。

    

    全模态大语言模型（LLM）被期望能够使用问题明确指向的模态来回答问题。然而，现有的训练范式很少验证模型是否真的遵循了该模态，因为来自同一样本的多模态输入往往为同一答案提供冗余的证据。在这项工作中，我们揭示了全模态大语言模型中普遍存在的跨模态捷径现象：当被问及与音频相关的问题时，模型对图像的依赖程度与对音频相当，有时甚至更高。为了系统地诊断这种行为，我们引入了因子化模态诊断方法，该方法在样本之间独立地交换音频和图像，以隔离每种模态的因果贡献。在两个模型家族的不同设置中，我们发现这种捷径在监督微调和强化学习后训练过程中持续存在，而基于裁判的强化学习可能会进一步放大这种对无关视觉信息的依赖。

    arXiv:2609.36798v1 Announce Type: cross  Abstract: Omni-modal large language models (LLMs) are expected to answer a question using the modality it explicitly refers to. However, existing training paradigms rarely verify whether models actually follow this modality, because multimodal inputs from the same sample often provide redundant evidence for the same answer. In this work, we uncover a pervasive cross-modal shortcut in omni-modal LLMs: when asked an audio-related question, models rely on the image as much as on the audio, and sometimes even more. To systematically diagnose this behavior, we introduce the Factorized Modality Diagnostic, which independently swaps audio and images between samples to isolate each modality's causal contribution. Across two model families in different settings, we find that this shortcut persists throughout supervised fine-tuning and reinforcement learning post-training, while judge-based RL may further amplify such reliance on irrelevant visual informa
    
[^121]: QuantMLA：面向低比特 MLA KV 缓存的函数对齐双路径量化

    QuantMLA: Function-Aligned Dual-Path Quantization for Low-Bit MLA KV Caching

    [https://arxiv.org/abs/2609.36760](https://arxiv.org/abs/2609.36760)

    提出了 QuantMLA——一个函数对齐的低比特双路径量化框架，通过系统建模 MLA 内容路径与 RoPE 路径的量化误差，并学习可完全离线融合的路径特定变换，在消除在线开销的同时大幅压缩 MLA KV 缓存且保持全精度计算效果。

    

    多头潜在注意力（MLA）通过为其内容路径与解耦的 RoPE 路径采用紧凑缓存，实现了表达能力强的多头注意力，但其缓存内存仍会随上下文长度和批处理大小线性增长。在本工作中，我们建立了 MLA 双路径量化误差的系统模型，刻画了它们对注意力输出失真的不同影响，并解释了 RoPE 路径误差被显著放大的现象。基于这一分析，我们提出了 QuantMLA，一个用于低比特双路径量化的函数对齐框架。我们推导出路径特定的变换空间，这些空间在保持全精度计算的同时，可以完全离线融合到模型参数中，从而消除在线变换开销。在这些空间内，QuantMLA 以函数对齐的目标学习路径特定的变换：注意力输出重建目标捕捉内容路径的耦合匹配与聚合误差，而位置（posit

    arXiv:2609.36760v1 Announce Type: cross  Abstract: Multi-Head Latent Attention (MLA) enables expressive multi-head attention with compact caches for its content and decoupled RoPE paths, yet cache memory still scales linearly with context length and batch size. In this work, we establish a systematic model of MLA's dual-path quantization errors, characterizing their distinct effects on attention-output distortion and explaining the pronounced amplification of RoPE-path errors. Guided by this analysis, we introduce QuantMLA, a function-aligned framework for low-bit dual-path quantization. We derive path-specific transformation spaces that preserve full-precision computation while remaining fully fusible into model parameters offline, eliminating online transformation overhead. Within these spaces, QuantMLA learns path-specific transformations with function-aligned objectives: attention-output reconstruction captures the content path's coupled matching and aggregation errors, while posit
    
[^122]: 韵律训练的表征能否在可训练融合之外带来帮助？基于冻结HuBERT的参数匹配研究

    Does a prosody-trained representation help beyond trainable fusion? A parameter-matched study with frozen HuBERT

    [https://arxiv.org/abs/2609.36754](https://arxiv.org/abs/2609.36754)

    该研究通过参数匹配的冻结HuBERT实验发现，可训练融合机制本身即可带来词错误率下降，而额外的韵律表征虽被模型所依赖，却未能带来显著的额外识别收益。

    

    显式的韵律线索可能有助于自发性语音的自动语音识别（ASR），但辅助表征通常需要额外的可训练组件，这使得识别收益究竟来自辅助信息本身还是融合机制变得不明确。为解决这一问题，我们使用冻结的HuBERT主干网络，以及一个经过训练以预测对数F0、浊音性、Delta对数F0、对数能量和频谱倾斜特征的64维表征。我们比较了三种设置：冻结主干的识别器、零辅助输入的可训练融合，以及提供该学习表征的相同融合结构。在Buckeye、Switchboard和AMI IHM三个数据集上，零辅助输入的融合相对于基线将词错误率（WER）降低了0.71至1.45个百分点，而使用学习表征的版本相对于零辅助融合的差异仅为+0.07、-0.09和+0.00个百分点，均无显著差异。然而，在推理阶段移除或错配该表征会增加其词错误率。因此，该模型虽然依赖这一表征，却并未表现出可测量的额外改进。

    arXiv:2609.36754v1 Announce Type: cross  Abstract: Explicit prosodic cues may help automatic speech recognition (ASR) of spontaneous speech, but auxiliary representations typically require additional trainable components, making it unclear whether gains come from the auxiliary information or the fusion mechanism. We address this using a frozen HuBERT backbone and a 64-dimensional representation trained to predict log F0, voicing, Delta log F0, log energy, and spectral tilt. We compare a frozen-backbone recognizer (Baseline), trainable fusion with zero auxiliary input (Null), and the same fusion supplied with the learned representation (Learned). Across Buckeye, Switchboard, and AMI IHM, Null reduces WER by 0.71-1.45 points over Baseline, whereas Learned differs from Null by +0.07, -0.09, and +0.00 points, with no significant differences. However, removing or mismatching the representation at inference increases Learned WER. Thus, Learned depends on the representation yet shows no measu
    
[^123]: 群体边缘化自我奖励强化学习驱动零标签自进化

    Group-Marginalized Self-Rewarding RL Drives Zero-Label Self-Evolving

    [https://arxiv.org/abs/2609.36750](https://arxiv.org/abs/2609.36750)

    本文提出 GMAE 方法，将响应奖励在不同组上下文下的实现聚合为响应级分布并估计期望优势，消除了随机组上下文带来的奖励信号不确定性，实现了无需人工标注的稳定自进化。

    

    自我奖励强化学习（RL）使大型语言模型（LLM）能够在无需人工标注的情况下自我进化。现有的基于集成的方法从采样组中构建奖励参考并据此分配奖励。然而，一个响应的奖励表示还依赖于其随机采样的组上下文，即其所在组中的其他响应。仅使用单一组上下文实现可能会遗漏期望的奖励信号，并为策略优化提供不可靠的引导。为解决这一问题，我们提出了群体边缘化优势估计（GMAE），它将跨可能上下文的奖励实现聚合为响应级分布，并估计期望优势。在八个基准测试和四个基础模型上的实验表明，GMAE 展现出强大的性能和跨领域泛化能力。此外，GMAE 还具有稳定的学习过程、较低的额外成本，以及在不同训练数据集和 RL 骨干网络上的良好适用性。

    arXiv:2609.36750v1 Announce Type: cross  Abstract: Self-rewarding reinforcement learning (RL) enables large language models (LLMs) to self-evolve without human labels. Existing ensemble-based methods construct reward references from rollout groups and assign rewards accordingly. However, a response's reward representation also depends on its randomly sampled group context, i.e., the other responses in its group. Using only one group-context realization may miss desired reward signals and provide unreliable guidance for policy optimization. To address this issue, we propose Group-Marginalized Advantage Estimation (GMAE), which aggregates reward realizations across possible contexts into a response-level distribution and estimates expected advantages. Experiments across eight benchmarks and four base models demonstrate strong performance and cross-domain generalization. GMAE also exhibits stable learning, low extra cost, and good applicability across training datasets and RL backbones.
    
[^124]: SIPO：统一强化学习与在线策略自蒸馏

    SIPO: Unifying Reinforcement Learning with On-Policy Self-Distillation

    [https://arxiv.org/abs/2609.36742](https://arxiv.org/abs/2609.36742)

    提出 SIPO 方法，通过对比性自教师将强化学习与在线策略自蒸馏相统一，为稀疏的可验证奖励提供密集的 token 级信用分配，克服了以往自教师过度自信及对长推理轨迹过度惩罚的问题。

    

    可验证奖励的强化学习（RLVR）已成为提升大语言模型（LLMs）在各类任务上表现的标准范式，但其稀疏的结果奖励缺乏对中间步骤的 token 级信用分配。为解决这一问题，在线策略自蒸馏（OPSD）利用具有特权上下文的自教师提供额外的密集学习信号。然而，由于自教师往往过于自信，并对长推理轨迹施加过多惩罚，OPSD 在实践中常常表现不佳。为缓解这一问题，我们提出了带有对比性自教师的自指导策略优化（SIPO），以提供密集的信用分配。在每次迭代中，SIPO 从当前策略为每个提示采样多个 rollout，用环境奖励对其进行评分，并通过将参考答案与该组中出现的错误配对，为每个 rollout 构建两个教师上下文。随后，模型重新评估其……

    arXiv:2609.36742v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) has become a standard paradigm for improving large language models (LLMs) on various tasks, yet its sparse outcome rewards lack token-level credit assignment for intermediate steps. To address this, on-policy self-distillation (OPSD) leverages a self-teacher with privileged context to provide additional dense learning signals. However, because the self-teacher is often overconfident and imposes excessive penalties on long reasoning trajectories, OPSD frequently struggles in practice. To mitigate this, we propose self-instructing policy optimization (SIPO) with a contrastive self-teacher to provide dense credit. At each iteration, SIPO samples multiple rollouts per prompt from the current policy, scores them with environment rewards, and constructs two teacher contexts for each rollout by pairing the reference answer with mistakes made within the group. The model then re-evaluates its 
    
[^125]: 反向传播输出动量：将优化器历史从参数空间迁移到任务空间

    Backpropagated Output Momentum: Relocating Optimizer History from Parameters to Task Space

    [https://arxiv.org/abs/2609.36738](https://arxiv.org/abs/2609.36738)

    提出反向传播输出动量（BOM），通过在模型输出端存储预测误差的紧凑移动平均并逐步经当前网络重投影，将优化器历史从参数空间迁移到任务空间，从而将优化器状态减少高达99.8%并提升验证性能。

    

    优化器动量通常以参数规模的过去梯度移动平均形式存储，这使得历史信息的维护代价高昂，并将每个过去的信号固定在其被计算时的坐标系中。我们提出反向传播输出动量（BOM），它改为在模型输出处存储预测误差的紧凑移动平均，并在每一步通过当前网络对该历史信息进行重新投影。批级别分析刻画了这种迁移所保留与省略的信息，同时该实现保留了当前的有监督梯度，并能够替代若干自适应优化器中的一阶矩分量。作为基于动量的优化器的插件（包括那些已经压缩其状态的优化器），BOM 在三种组合中将参数形状的优化器状态减少了 49.7%–99.8%，并在三个语言骨干网络上平均将配对步进时间降低了 4.0%。它还提升了语言与视觉任务的平均验证性能。

    arXiv:2609.36738v1 Announce Type: cross  Abstract: Optimizer momentum is usually stored as a parameter-sized moving average of past gradients, which makes history costly and fixes each past signal in the coordinates in which it was computed. We introduce Backpropagated Output Momentum (BOM), which instead stores a compact moving average of prediction errors at the model output and reprojects that history through the current network at every step. A batch-level analysis characterizes the information retained and omitted by this relocation, while the implementation preserves the current supervised gradient and can replace the first-moment component of several adaptive optimizers. As a plug-in for momentum-based optimizers, including ones that already compress their state, BOM reduces parameter-shaped optimizer state by 49.7-99.8% in three compositions and, averaged over three language backbones, paired step time by 4.0%. It also improves mean validation performance across language and vi
    
[^126]: 基于可微声学仿真的声道重建

    Reconstructing the Vocal Tract with Differentiable Acoustic Simulation

    [https://arxiv.org/abs/2609.36737](https://arxiv.org/abs/2609.36737)

    提出了一种可微分、GPU加速的声道声学仿真器，通过频域建模和可微湍流模型，实现了仅凭语音声音反向重建声道形状的突破。

    

    声道是人体中负责对声音进行滤波以产生语音的区域。本文提出了一种可微分且GPU加速的声道声学仿真器。该可微仿真器通过让声音沿声道的声管模型传播来合成语音，并借助其梯度能够求解逆问题：仅从语音声重建声道的形状。尽管几何与声音之间的逆映射以非凸性著称，我们发现梯度下降法在以下三项技术贡献的支持下能够成功求解：(1) 我们设计了声道流体动力学的频域公式，其GPU可并行化程度比时域有限差分方法高70倍；(2) 我们集成了一个可微分的湍流模型以合成辅音；(3) 与隐式神经表示（INR）和神经场领域的先前工作类似，我们发现……

    arXiv:2609.36737v1 Announce Type: cross  Abstract: The vocal tract is the region of the human body responsible for filtering one's voice to create speech. In this paper, we present a differentiable and GPU accelerated acoustic simulator for the vocal tract. The differentiable simulator synthesizes speech by propagating sound along an acoustic tube model of the vocal tract, and via its gradients, can solve the inverse problem: reconstructing the shape of the vocal tract solely from the sound it produces. Although the inverse mapping between geometry and sound is notoriously non-convex, we discover that gradient descent succeeds with three technical contributions: (1) we design a frequency domain formulation of the vocal tract's fluid dynamics that is 70x more GPU parallelizable than finite differences in time, (2) we integrate a differentiable model for turbulence to synthesize consonants, and (3) similar to prior work in implicit neural representations (INRs) and neural fields, we find
    
[^127]: 从神经元到对话：语音脑机接口

    From Neurons to Conversation: Speech Brain-Computer Interfaces

    [https://arxiv.org/abs/2609.36736](https://arxiv.org/abs/2609.36736)

    本文从系统级视角综述了语音脑机接口研究，强调语音脑机接口本质上是一个自适应临床系统，其神经表征、硬件、解码架构、语言先验、反馈与用户学习需长期协同演化，而非单纯的神经到文本解码器。

    

    语音脑机接口（BCIs）旨在通过将与言语、语言或交际意图相关的神经活动转化为文本、合成语音或虚拟形象控制等外部输出，来恢复沟通能力。皮层内和皮层脑电图记录技术、深度序列模型以及语言模型辅助解码的最新进展推动了该领域的快速发展，包括高性能的尝试性言语解码和日益自然的语音合成。然而，这些成就也揭示出，语音脑机接口并非简单的神经信号到文本的解码器，而是适应性临床系统，其中神经表征、记录硬件、解码架构、语言先验、反馈和用户学习随时间相互作用。本文从系统级视角综述了语音脑机接口研究，首先考察了言语和语言的神经基础，强调其分层、分布式和时间（摘要在此处截断）

    arXiv:2609.36736v1 Announce Type: cross  Abstract: Speech brain-computer interfaces (BCIs) aim to restore communication by transforming neural activity related to speech, language, or communicative intent into external outputs such as text, synthesized voice, or avatar control. Recent advances in intracortical and electrocorticographic recording, deep sequence models, and language-model-assisted decoding have enabled rapid progress, including high-performance attempted-speech decoding and increasingly naturalistic speech synthesis. Yet these achievements also reveal that speech BCIs are not simply neural-to-text decoders. They are adaptive clinical systems in which neural representations, recording hardware, decoding architectures, language priors, feedback, and user learning interact over time. Here, we synthesize speech BCI research from a system-level perspective. We first examine the neural substrates of speech and language, emphasizing their hierarchical, distributed, temporally s
    
[^128]: 提炼重要内容：面向大语言模型的置信度感知选择性蒸馏

    Distilling What Matters: Confidence-Aware Selective Distillation for Large Language Models

    [https://arxiv.org/abs/2609.36734](https://arxiv.org/abs/2609.36734)

    提出CaRE-KD置信度门控蒸馏框架，通过词元级自适应切换正反向KL散度以及批次级抑制教师不确定性更新的机制，解决大语言模型教师预测的高熵与幻觉问题，避免蒸馏过程损害学生模型良好校准的先验。

    

    知识蒸馏（KD）通过匹配输出分布来训练一个容量较小的学生模型去模仿容量较大的教师模型，这隐含地假设教师模型是可靠的神谕（oracle）。在大语言模型（LLM）中，这一假设常常失效：教师模型的预测可能表现出高熵和幻觉现象，导致标准知识蒸馏反而损害学生模型原本良好校准的先验。我们提出CaRE-KD，一个置信度门控的蒸馏框架，用不确定性自适应优化取代静态目标。CaRE-KD包含两个组件：一是词元级损失，基于教师与学生的置信度在正向KL散度和反向KL散度之间自适应切换；二是批次级的认知拒绝机制，当教师模型比学生模型更加不确定时抑制参数更新。我们提供了梯度层面的分析，展示了这种双粒度设计如何诱导出一种条件校准机制。

    arXiv:2609.36734v1 Announce Type: new  Abstract: Knowledge Distillation (KD) trains a smaller-capacity student model to imitate a larger-capacity teacher model by matching output distributions, implicitly assuming the teacher to be a reliable oracle. In large language models (LLMs), this assumption often fails: teacher predictions can exhibit high entropy and hallucinations, causing standard KD to degrade well-calibrated student priors. We propose CaRE-KD, a confidence-gated distillation framework that replaces static objectives with uncertainty-adaptive optimization. CaRE-KD has two components: a token-level loss (CaRE-Divergence) that adaptively switches between Forward and Reverse KL divergence based on teacher--student confidence, and a batch-level epistemic rejection mechanism (Revival) that suppresses updates when the teacher is more uncertain than the student. We provide a gradient-level analysis showing how this dual-granularity design induces a conditional calibration mechanis
    
[^129]: 智能体能为智能体设计程序库吗？

    Can Agents Design Libraries for Agents?

    [https://arxiv.org/abs/2609.36730](https://arxiv.org/abs/2609.36730)

    该论文提出LibraryDesignBench基准来评估智能体为其他智能体设计代码库的能力，发现智能体设计者在大多数任务上能复现人类生产级库的抽象，但下游智能体对库的利用不足，经常重新实现库中已有的功能。

    

    智能体越来越多地基于其他智能体编写的代码进行构建，但它们往往重新实现而非复用现有代码，导致后续智能体必须工作的代码库不断膨胀。为了衡量智能体为其他智能体设计库的能力，我们提出了LibraryDesignBench，这是一个两阶段的基准测试：智能体根据一份规范实现一个功能齐全的库，该规范定义了所需能力和潜在用例，但不规定具体设计。我们通过来自不同模型家族的三个用户智能体所编写程序的正确性和简洁性来评估该库。该基准测试涵盖15个库设计任务中的242个经专家验证的编程问题，涉及四种编程语言。在15个任务中的11个上，智能体设计者能够复现人工编写的生产级库中的抽象。下游智能体对智能体编写的库和人工编写的库都有采用，但利用不足，常常重新实现库中已提供的功能。

    arXiv:2609.36730v1 Announce Type: new  Abstract: Agents increasingly build on code written by other agents, and they reimplement rather than reuse, growing the codebases later agents must work in. To measure how well agents design libraries for other agents, we introduce LibraryDesignBench, a two-phase benchmark in which an agent implements a full-featured library from a specification that defines required capabilities and potential use cases without prescribing the design. We evaluate the library through the correctness and simplicity of programs written by three user agents from different model families. The benchmark spans 242 expert-validated programming problems across 15 library-design tasks in four languages. On eleven of the fifteen tasks, agent designers reproduce the abstractions of the human-written production library. Downstream agents adopt agent- and human-written libraries alike but underuse them, reimplementing capabilities the library already provides. Our failure anal
    
[^130]: ATTUNER：通过查询侧适配实现无需重计算的KV缓存复用

    ATTUNER: Recomputation-Free KV Cache Reuse via Query-Side Adaptation

    [https://arxiv.org/abs/2609.36722](https://arxiv.org/abs/2609.36722)

    该论文发现位置无关缓存的质量损失主要源于模型在多个内容片段间选择时的注意力分数偏差而非位置编码不匹配，并提出ATTUNER通过查询侧适配修复注意力分数，在完全不重计算KV缓存的情况下达到接近完整预填充的效果。

    

    大型语言模型（LLM）智能体反复将可复用的内容（如技能、文档和记忆条目）加载到当前上下文中，为每个请求重新编码这些内容会浪费计算资源。位置无关缓存通过独立编码每个内容片段并在任意位置复用其键值（KV）状态来缓解这一问题，但相较于完整上下文预填充会产生质量损失。现有方法通过恢复全局位置ID或重新计算选定token来修复这一损失。在这项工作中，我们隔离了质量损失的来源，发现位置不匹配的影响较小，且独立缓存的内容保留了忠实的表示：读取给定内容基本保持准确，只有当模型必须在多个内容片段中进行选择时性能才会下降。此外，用完整预填充的注意力分数替换PIC的注意力分数，可以在缓存KV保持不变的情况下恢复性能，并且（该修复可以）局部化……

    arXiv:2609.36722v1 Announce Type: new  Abstract: Large language model (LLM) agents repeatedly load reusable content, such as skills, documents, and memory entries, into the current context. Re-encoding this content for every request wastes computation. Position-independent caching (PIC) alleviates this by encoding each artifact independently and reusing its key-value (KV) states at arbitrary positions, but it incurs a quality loss relative to full-context prefill. Existing methods repair this loss by restoring global position IDs or recomputing selected tokens. In this work, we isolate the source of the loss, finding that the positional mismatch has minor effect, and independently cached artifacts retain faithful representations: reading a provided artifact stays largely accurate, and performance degrades only when the model must select among multiple artifacts. Moreover, replacing PIC's attention scores with full-prefill scores recovers performance with the cached KV unchanged, locali
    
[^131]: LAURA：面向法律合同中可解释模糊条款识别的知识蒸馏方法

    LAURA: Knowledge Distillation for Interpretable Ambiguous Clause Identification in Legal Contracts

    [https://arxiv.org/abs/2609.36707](https://arxiv.org/abs/2609.36707)

    LAURA是一个后训练框架，通过结合IRAC-Unlearning提示技术的知识蒸馏，将教师大语言模型的能力迁移到小型开放权重学生模型上，实现法律合同中可解释的模糊条款识别，其中仅250M参数的Flan-T5即可达到最先进的性能。

    

    法律合同中存在使企业面临财务和法律风险的模糊之处。有些模糊之处允许灵活解释而不会引发争议，而另一些则会导致重大的法律冲突。这使得仅进行识别并不足够，可解释的理由分析变得至关重要。我们提出了LAURA，一个用于可解释模糊条款识别的后训练框架。LAURA利用结合IRAC-Unlearning提示技术的知识蒸馏，将知识从教师大语言模型迁移到开放权重的学生模型（参数量≤1B），然后使用结合分类损失和理由生成损失的联合目标进行训练。该框架支持法律和非法律领域的利益相关者就哪些模糊之处需要进一步关注做出明智决策。在7个基线模型和7个开放权重模型上的大量实验表明，使用Flan-T5（250M）的LAURA达到了最先进的性能。

    arXiv:2609.36707v1 Announce Type: new  Abstract: Legal contracts contain ambiguities that expose enterprises to financial and legal risks. Some ambiguities allow flexible interpretation without triggering disputes, while others lead to significant legal conflicts. This makes identification alone insufficient, and interpretable rationale analysis essential. We propose LAURA, a post-training framework for interpretable ambiguous clause identification. LAURA leverages knowledge distillation with an IRAC-Unlearning prompting technique to transfer knowledge from a teacher LLM to an open-weight student model (<=1B parameters), which is then trained using a joint objective combining classification and rationale generation losses. The framework supports both legal and non-legal stakeholders in making informed decisions about which ambiguities require further attention. Extensive experiments across 7 baselines and 7 open-weight models demonstrate that LAURA with Flan-T5 (250M) delivers state-of
    
[^132]: 迷失于对话还是迷失于翻译？诊断检索增强生成（RAG）中的多轮性能退化

    Lost in Conversation or Lost in Translation? Diagnosing Multi-Turn Degradation in RAG

    [https://arxiv.org/abs/2609.36700](https://arxiv.org/abs/2609.36700)

    该研究通过150万次模拟对话的大规模仿真实验，首次系统揭示了RAG与GraphAG在真实多轮对话场景中存在高达21%的性能退化和47%的不可靠性增加，指出当前基于单轮完备查询的评估范式与实际使用方式存在严重错配。

    

    当用户与大语言模型（LLM）对话时，往往从一个简单的问题开始，并通过后续多轮追问逐步构建出多跳问题。检索增强生成（RAG）及其基于图的变体已成为将LLM回答锚定于外部证据的主流方法，然而对它们的评估几乎完全基于单轮、信息完备的查询。我们通过一项大规模仿真研究系统性地考察了这种评估上的错配。在先前多轮LLM评估工作的基础上，我们将多跳问答（QA）基准中的问题转化为信息欠完备的对话，并在150万次模拟对话中评估了十个LLM助手与八个检索系统。研究结果表明，多轮交互会导致普遍的性能退化：相对性能下降高达21%，不可靠性增加47%，这使得RAG（摘要在此处被截断）

    arXiv:2609.36700v1 Announce Type: cross  Abstract: When conversing with large language models (LLMs), users often begin with a simple question and build towards a multi-hop question through follow-up turns. Retrieval-augmented generation (RAG) and its graph-based variant (GraphRAG) have become the dominant approaches for grounding LLM responses in external evidence, yet both are evaluated almost exclusively on single-turn, fully specified queries. We systematically investigate this evaluation mismatch through a large-scale simulation study. Building on prior work on multi-turn LLM evaluation, we transform questions from multi-hop question answering (QA) benchmarks into underspecified conversations and evaluate ten LLM assistants with eight retrieval systems across 1.5 million simulated conversations. Our findings reveal that multi-turn interaction causes widespread performance degradation, incurring relative performance drops of up to 21% and increasing unreliability by 47%, making RAG
    
[^133]: Video2Skill：从流式经验到可复用的具身技能

    Video2Skill: From Streaming Experience to Reusable Embodied Skills

    [https://arxiv.org/abs/2609.36691](https://arxiv.org/abs/2609.36691)

    该论文提出了流式具身技能发现（SESD）问题并推出Video2Skill基准，用于系统评估视觉-语言模型能否将连续视频流中的操作事件组织成可影响后续决策的持久可复用技能库。

    

    操作行为在不同物体和场景之间差异很大，但它们共享一小组可复用的技能，利用这些技能进行规划有助于具身智能体泛化到新任务。然而，智能体只能用它已知的技能进行规划。从观察到的经验中恢复技能（即规划的反过程）能够随时间积累这类知识，并为训练未来的智能体生成技能数据。视觉-语言模型（VLM）能够很好地描述单个操作事件，但它们能否将事件流组织成可复用的技能呢？我们将该问题形式化为流式具身技能发现：模型按顺序观看视频，并维护一个持久的技能库，该技能库会影响其后续决策。为了系统地衡量这种能力，我们提出了Video2Skill，这是一个涵盖机器人桌面操作和人类厨房活动的基准，测试三项核心能力：（i）在时间上定位操作事件，（ii）对事件进行分组……（摘要在此处截断）

    arXiv:2609.36691v1 Announce Type: new  Abstract: Manipulation behaviors vary widely across objects and scenes, but they share a small set of reusable skills, and planning with these skills helps embodied agents generalize to new tasks. Yet an agent can only plan with skills it knows. Recovering skills from observed experience, the inverse of planning, builds this knowledge over time and yields skill data for training future agents. Vision-Language Models (VLMs) describe individual manipulation events well, but can they organize a stream of events into reusable skills? We formulate this problem as Streaming Embodied Skill Discovery (SESD): a model watches videos in sequence and maintains a persistent skill library that shapes its later decisions. To systematically measure this ability, we introduce Video2Skill, a benchmark that covers robot tabletop manipulation and human kitchen activity and tests three core capabilities: (i) locating manipulation events in time, (ii) grouping events o
    
[^134]: CHAIN：基于因果-时序超图推理的校准大语言模型预测

    CHAIN: Calibrated LLM Forecasting via Causal-Temporal Hypergraph Inference

    [https://arxiv.org/abs/2609.36689](https://arxiv.org/abs/2609.36689)

    该论文提出CHAIN方法，将大语言模型在因果-时序超图上的概率预测分解为证据加权、证据聚合和源融合三个阶段，并为每个阶段设计针对性机制，从预测过程内部缓解系统性校准偏差，提升概率输出的可信度。

    

    大语言模型在事件预测方面取得了显著进展，但其概率输出存在系统性的校准偏差，且该偏差在不同领域和问题类型间呈现异质性，损害了概率输出在不确定性决策中的可信度。然而，现有的校准方法通常在预测完成后才对概率输出进行修正，而未对预测过程本身内部的偏差结构来源进行建模。为应对这一挑战，我们将因果-时序超图上的概率预测分解为三个阶段——证据加权、证据聚合和源融合——并提出CHAIN方法，针对每个阶段设计专门机制以缓解偏差：(i) 通过因果拓扑距离调制时间衰减函数；(ii) 在方向感知去重后通过Noisy-OR聚合近似独立的因果链（摘要在此处被截断）

    arXiv:2609.36689v1 Announce Type: cross  Abstract: Large language models have achieved significant progress in event forecasting, yet their probability outputs exhibit systematic calibration bias that varies heterogeneously across different domains and question types, undermining the trustworthiness of probabilistic outputs for decision-making under uncertainty. However, existing calibration methods typically correct probability outputs after prediction is complete, without modeling the structural sources of bias within the prediction process itself. To address this challenge, we decompose probabilistic prediction over causal-temporal hypergraphs into three stages, evidence weighting, evidence aggregation, and source fusion, and propose CHAIN, which designs stage-specific mechanisms to mitigate bias at each stage: (i) modulating the temporal decay function by causal topological distance, (ii) aggregating approximately independent causal chains via Noisy-OR after direction-aware dedupli
    
[^135]: ProgressCompass：具身进展奖励模型若缺乏正确上下文便会迷失方向

    ProgressCompass: Embodied Progress Reward Models Are Lost Without the Right Context

    [https://arxiv.org/abs/2609.36684](https://arxiv.org/abs/2609.36684)

    论文提出“上下文依赖的进展估计”这一新问题，并构建了包含24个操作任务的ContextProgress-Bench基准，用以评估进展奖励模型在必须依靠历史上下文才能判断任务进展的长时程具身任务中的表现。

    

    具身智能体如今承担着越来越长的任务。对于长时程任务，仅知道任务最终成功或失败意义不大，过程中的每一步都很重要。进展奖励模型（Progress Reward Models, PRMs）在每个步骤中对任务已完成的程度进行打分，可作为稠密奖励、验证器和监控器使用。然而在长任务中，仅凭当前帧往往无法判断任务进展到何种程度，因为进展取决于之前发生的事情。我们将这一问题称为“上下文依赖的进展估计”。现有的进展估计基准大多集中于可以从当前观测中直接读取进展的短任务，而PRMs在需要上下文时能否准确估计进展仍缺乏充分研究。因此，我们构建了ContextProgress-Bench，包含24个操作任务、共120个回合。该基准涵盖三种设定：(i) 状态回溯（State Recall），即判断进展所需的信息出现过但不在当前帧中；(ii) 序列追踪……

    arXiv:2609.36684v1 Announce Type: new  Abstract: Embodied agents now take on ever longer tasks. For long tasks, knowing only whether a task finally succeeds or fails says little; the steps along the way matter. Progress Reward Models (PRMs) score how far a task has come at every step, and serve as dense rewards, verifiers and monitors. Yet in long tasks the current frame alone often cannot tell how far the task has come, because progress depends on what happened before. We call this problem context-dependent progress estimation. Existing benchmarks on progress estimation mostly focus on short tasks whose progress can be read from the current observation, and whether PRMs can estimate progress when context is needed remains underexplored. We therefore build ContextProgress-Bench, with 24 manipulation tasks for 120 episodes. The benchmark covers three settings: (i) State Recall, where information needed for progress appeared earlier but is not in the current frame; (ii) Sequence Tracking
    
[^136]: MARCO：面向条件分子优化的多轮智能体强化学习

    MARCO: Multi-Round Agentic Reinforcement for Conditional Molecular Optimization

    [https://arxiv.org/abs/2609.36683](https://arxiv.org/abs/2609.36683)

    MARCO提出了一个基于评估器的多轮强化学习框架，通过“提议—反馈—修订”轨迹和聚合回合奖励的组相对策略优化，训练分子编辑器在多目标分子优化中同时兼顾有效性、性质改进与相似性控制。

    

    分子优化本质上是迭代式的：先提出候选分子，再针对多个目标进行评估，然后在保持与源分子关系的同时进行修订。然而，大多数指令遵循模型只输出一个编辑后的分子，将有效性、性质改进和相似性控制强行压缩到单一响应中。我们提出了MARCO，这是一个以评估器为基准的强化学习框架，通过有界的“提议—反馈—修订”轨迹来训练分子编辑器。MARCO将塑形的回合奖励聚合为无折扣的轨迹回报，用于组相对策略优化。我们评估了这种训练带来的两个结果：Same-1在单次响应预算下测试训练后的策略，而Same-5则测试当最多允许五次响应时，同一策略能否利用验证器的反馈。在三目标的MuMOInstruct基准、三个Qwen骨干模型以及已见/未见指令划分上，基于SFT初始化的…

    arXiv:2609.36683v1 Announce Type: cross  Abstract: Molecular optimization is inherently iterative: a candidate is proposed, evaluated against several objectives, and revised while preserving a relationship to the source molecule. Most instruction-following models instead emit one edited molecule, forcing validity, property improvement, and similarity control into a single response. We introduce MARCO, an evaluator-grounded reinforcement-learning framework that trains molecular editors on bounded proposal--feedback--revision trajectories. MARCO aggregates shaped turn rewards into an undiscounted trajectory return for group-relative policy optimization. We evaluate two consequences of this training: Same-1 tests the trained policy under a one-response budget, while Same-5 tests whether the same policy can use verifier feedback when up to five responses are available. Across the three-objective MuMOInstruct benchmark, three Qwen backbones, and seen/unseen instruction splits, SFT-initializ
    
[^137]: 哥德尔森林：平衡搜索深度与广度的数据中心递归自我改进

    G\"odel Forest: Balancing Search Depth and Breadth for Data-Centric Recursive Self-Improvement

    [https://arxiv.org/abs/2609.36675](https://arxiv.org/abs/2609.36675)

    提出哥德尔森林多智能体框架，将数据中心递归自我改进组织为协同进化的搜索树集合——每个智能体在持久树上自主深化、分支或剪枝数据策略以保证搜索深度，并行多棵树则提供探索广度，从而解决该领域深度与广度难以兼得的根本困境。

    

    递归自我改进（RSI）旨在通过让模型改进自身来实现复合收益。现有的大多数RSI系统是在冻结的基础模型之上优化外部智能体框架或提示词，而数据中心（data-centric）的RSI则通过在智能体生成的数据上进行训练，直接更新模型自身的参数。然而，由于验证数据策略需要代价高昂的模型训练，现有方法面临一个根本性的困境：单一智能体会陷入狭窄的方向、缺乏探索广度，而朴素的并行搜索或繁重的轨迹共享又会牺牲长程搜索深度。为了应对这一挑战，我们提出了哥德尔森林，这是一个将递归自我改进组织为协同进化的搜索树集合的多智能体框架。在哥德尔森林中，每个智能体自主地培育一棵持久树，根据模型反馈对数据策略进行深化、分支或剪枝以保证搜索深度，同时并行树则探索……（原文摘要在此处截断）

    arXiv:2609.36675v1 Announce Type: new  Abstract: Recursive self-improvement (RSI) aims to achieve compounding gains by having models improve themselves. While most existing RSI systems optimize external agent harnesses or prompts around a frozen base model, data-centric RSI directly updates the model's own parameters by training on agent-generated data. However, because validating data strategies requires expensive model training, existing methods face a fundamental dilemma: a single agent gets trapped in narrow directions and lacks exploration breadth, while naive parallel search or heavy trace sharing sacrifices long-horizon search depth. To address this challenge, we introduce G"odel Forest, a multi-agent framework that organizes recursive self-improvement as an ensemble of co-evolving search trees. In G"odel Forest, each agent autonomously grows a persistent tree, deepening, branching, or pruning data strategies based on model feedback to secure depth, while parallel trees explore 
    
[^138]: 重放曲率：面向大语言模型推理的精确且可扩展的NVFP4量化

    Replay the Curvature: Accurate and Scalable NVFP4 Quantization for Large Language Model Inference

    [https://arxiv.org/abs/2609.36654](https://arxiv.org/abs/2609.36654)

    提出Schur Replay缩放因子选择算法，通过精确复现GPTQ逐列更新来准确评估NVFP4块缩放因子的重构误差，并支持跨设备可扩展地求解，从而实现大语言模型推理的高精度4比特量化。

    

    大语言模型使得权重存储和内存流量成为推理的主要开销，这促使人们采用仅用几个比特表示每个权重的低精度格式。此类格式使用缩放因子将浮点数值映射到一个小的码本中；NVFP4通过让每16个E2M1权重共享一个E4M3块缩放因子来提升局部数值范围的利用率。在GPTQ中，选择该缩放因子十分困难，因为量化某一列会更新其后续列，因此独立评估一个块可能会错误估计其最终的重构误差。大模型还带来了第二个挑战：全精度权重、校准激活值和二阶状态无法全部驻留在单个加速器上，而将完整层分配给各设备又导致每个耗时的层求解只能串行执行。我们提出了Schur Replay，一种缩放因子选择算法，它能够复现每个块缩放因子所引发的GPTQ更新，并在考虑补偿效应之后对产生的块误差进行评分……

    arXiv:2609.36654v1 Announce Type: cross  Abstract: Large language models make weight storage and memory traffic major inference costs, motivating low-precision formats that represent each weight with only a few bits. Such formats use a scale to map floating-point values into a small codebook; NVFP4 improves local range utilization by letting every 16 E2M1 weights share an E4M3 block scale. Choosing that scale is difficult in GPTQ because quantizing one column updates those that follow, so evaluating a block independently can misestimate its final reconstruction error. Large models pose a second challenge: full-precision weights, calibration activations, and second-order state cannot all remain on one accelerator, while assigning complete layers to devices leaves each time-consuming layer solve serial. We introduce \emph{Schur Replay}, a scale-selection algorithm that reproduces the GPTQ updates caused by each block scale and scores the resulting block error after accounting for compens
    
[^139]: 是什么让循环在循环语言模型中有效？

    What Makes Recurrence Effective in Looped Language Models?

    [https://arxiv.org/abs/2609.36636](https://arxiv.org/abs/2609.36636)

    本文通过受控实验系统研究了循环语言模型中循环何时有效、应用于何处及条件化方式的影响，发现循环能提升超出训练范围的推理能力但会损害知识性能，且性能取决于层与循环迭代之间的分配方式而非单纯的有效深度。

    

    循环语言模型通过参数共享来增加计算深度，提供了一条在不增加参数的情况下扩展推理计算的路径。然而，目前仍不清楚何时引入额外的循环是有益的，以及架构选择如何影响其有效性。通过受控实验，我们系统地研究了以下三个问题：(1) 循环何时有帮助；(2) 应该在哪里应用循环；(3) 循环的条件化方式如何影响性能。我们的评估覆盖了在知识与推理任务下低于、等于以及超出训练范围的推理预算。(1) 我们发现循环可以在超出训练范围的情况下提升推理性能，但会降低知识任务的性能，而更难的推理实例并不总是能获得更多收益。(2) 性能还取决于不同层与循环迭代之间的分配方式，这表明仅凭有效深度不足以预测模型行为。非循环的输出层……

    arXiv:2609.36636v1 Announce Type: cross  Abstract: Looped language models (LoopLMs) increase computational depth through parameter sharing, offering a path to scale inference computation without adding parameters. However, it remains unclear when additional recurrence is beneficial and how architectural choices affect its effectiveness. Through controlled experiments, we systematically examine (1) when recurrence helps, (2) where it should be applied, and (3) how its conditioning affects performance. Our evaluation covers inference budgets below, within, and beyond the training horizon under knowledge and reasoning tasks. (1) We find that recurrence can improve reasoning beyond the training horizon while degrading knowledge performance, but harder reasoning instances do not consistently benefit more. (2) Performance also depends on how distinct layers and recurrent iterations are allocated, showing that effective depth alone is insufficient to predict behavior. Non-recurrent output lay
    
[^140]: 面向人工智能研究论文的编辑诱发式问题生成

    Generating Edit-Inducing Questions for AI Research Manuscripts

    [https://arxiv.org/abs/2609.36617](https://arxiv.org/abs/2609.36617)

    该研究比较了GPT与人类审稿人为AI论文草稿生成“编辑诱发式问题”的能力，发现GPT的问题能引发更广泛深入的修改但有效率更低，并揭示了一个反直觉现象：处理长上下文反而会损害推理模型生成有用输出的能力。

    

    我们研究了大语言模型（LLM）生成“编辑诱发式问题”的能力，这类问题的答案能够帮助改进论文草稿。在一个由ICLR和NeurIPS的投稿版本与定稿版本配对组成的数据集上，我们比较了GPT模型在有或没有完整论文上下文情况下所生成问题的有用性，并与人类审稿人提出的问题进行对比。结果显示，GPT生成了更多编辑诱发式问题，且与审稿人的问题相比，其问题伴随着更广泛的修改，覆盖的编辑内容范围也更广。然而，GPT问题中真正具有编辑诱发性的比例却小得多。我们的分析证实了自动化生成的问题对作者是有益的，同时揭示了一个典型案例：在某些任务中，对长上下文的恰当关注反而会削弱推理模型产出有用结果的能力。

    arXiv:2609.36617v1 Announce Type: new  Abstract: We study the ability of LLMs to generate edit-inducing questions whose answer will improve a paper draft. On a dataset of paired submission and camera-ready papers from ICLR and NeurIPS, we compare the helpfulness of questions from GPT models with or without full paper context to that of human reviewers. GPT produces more edit-inducing questions and its questions are associated with more extensive edits and cover a broader range of edited content compared to questions from reviewers. However, a much smaller percentage of the GPT questions are edit-inducing. Our analyses confirm that automated questions can be beneficial to authors and highlight an example task where proper attending to long context deteriorates reasoning model ability to produce helpful output.
    
[^141]: 先行动，后推理：通过参考条件逆动力学加速多轮智能体的在线策略蒸馏

    Act First, Reason Later: Accelerating On-Policy Distillation for Multi-Turn Agents via Reference-Conditioned Inverse Dynamics

    [https://arxiv.org/abs/2609.36608](https://arxiv.org/abs/2609.36608)

    论文提出ActFirst-OPD训练框架，通过参考条件逆动力学让多轮智能体先执行动作、后异步生成完整推理响应，将环境交互与响应生成分离，从而在不牺牲轨迹质量的前提下显著加速在线策略蒸馏训练。

    

    在线策略蒸馏（OPD）通过在学生生成的响应上施加密集的教师监督来训练多轮语言智能体。然而，标准的“先思考后行动”轨迹采样要求在每个简短动作之前进行冗长的推理，从而延迟了环境转换和经验收集。直接生成动作可以减少这种延迟，但可能会降低轨迹采样质量。为了解决这一问题，我们提出了ActFirst-OPD，一个“先行动、后推理”的训练框架，它将环境交互与完整响应生成分离开来。学生通过参考条件逆动力学，利用其当前交互上下文和参考的下一个观测来推断并执行动作，当产生的转换偏离参考轨迹时，则切换到自主的下一动作预测。基于收集到的交互上下文，学生异步生成完整的“先思考后行动”响应，用于token级别的教师监督。实验……（摘要在此处被截断）

    arXiv:2609.36608v1 Announce Type: cross  Abstract: On-policy distillation (OPD) trains multi-turn language agents with dense teacher supervision on student-generated responses. However, standard think-then-act rollouts require lengthy reasoning before each short action, delaying environment transitions and experience collection. Generating actions directly reduces this delay but can degrade rollout quality. To address this, we propose ActFirst-OPD, an act-first, reason-later training framework that decouples environment interaction from full-response generation. The student infers and executes actions through reference-conditioned inverse dynamics using its current interaction context and a reference next observation, and switches to autonomous next-action prediction when the resulting transition deviates from the reference trajectory. From the collected interaction contexts, the student asynchronously generates full think-then-act responses for token-level teacher supervision. Experim
    
[^142]: SEED：基于隐式编码器-解码器的自推测解码

    SEED: Self-Speculative Decoding via Implicit Encoder-Decoder

    [https://arxiv.org/abs/2609.36590](https://arxiv.org/abs/2609.36590)

    SEED将仅解码器transformer重新解释为隐式编码器-解码器结构，通过复用验证阶段已计算的深层上下文表示来低成本生成高质量草稿，从而在解决草稿质量与成本权衡的同时加速LLM推理。

    

    自推测解码通过从目标模型自身生成草稿token来加速大语言模型（LLM）推理，但面临草稿质量与成本之间的严峻权衡。早退方法通过在中间层终止计算来低成本地生成草稿，但由于放弃了后续层提供的更深层表示，草稿质量会受损。多token预测通过从模型最终的隐藏状态进行输出来保证草稿质量，但在每个草稿生成步骤都需要付出完整前向传播的代价来产生这些状态。我们提出自推测编码器-解码器（SEED），这是一种自推测方法，通过复用在验证阶段已经计算好的深层上下文表示，以低成本获得高质量草稿。我们将标准的仅解码器transformer重新解释为隐式编码器-解码器结构：前几层（编码器）构建深层上下文表示，而最后几层（解码器）……

    arXiv:2609.36590v1 Announce Type: new  Abstract: Self-speculative decoding accelerates large language model (LLM) inference by drafting tokens from the target model itself, but faces a sharp tradeoff between the quality and cost of the draft. Early-exit methods produce drafts cheaply by terminating computation at intermediate layers, but forgo the deeper representations that later layers provide and thus suffer in draft quality. Multi-token prediction preserves draft quality by emitting from the model's final hidden states, but pays for a full forward pass to produce those states at every drafting step. We propose self-speculative encoder-decoder (SEED), a self-speculative method that obtains high-quality drafts cheaply by reusing the deep contextual representations already computed during verification. We reinterpret the standard decoder-only transformer as an implicit encoder-decoder: the first layers (encoder) build deep contextual representations, and the last few layers (decoder) 
    
[^143]: Transformer过早停止思考，而一个微型LoRA即可修复

    Transformers Stop Thinking Too Early, and a Tiny LoRA Fixes It

    [https://arxiv.org/abs/2609.36585](https://arxiv.org/abs/2609.36585)

    仅在单个早期层加入微小的rank-8 LoRA并冻结其余全部权重，就能让Transformer充分“用足”深度，将Qwen3-8B在24行长链追踪任务上的准确率从15.5%提升至99%，表明模型默认输出远未释放其潜在计算能力。

    

    预训练Transformer在追踪上下文中的引用时很少利用其网络深度。十三个基础模型只能可靠地追踪1.4-3.6行引用，额外的预训练循环也收效甚微。而在某个早期层部署的任务训练rank-8 LoRA，能在冻结所有模型权重的情况下扩展这种计算。Qwen3-8B在24行链上的精确准确率从15.5%提升至99%；训练更长时间的LoRA可达50行。Ouro-1.4B经过四次循环可达60行，八次循环后至少160行。该LoRA启动了一种“接力”机制：程序行通过一小段中间层传递其链身份，冻结的注意力头沿链逐步读取更远的位置，而移除对父行的注意力会中止这一接力。冻结模型测量方法在四个留出模型中的三个里，于容差范围内定位到最后一个有效干预层。任务特定的LoRA也能提升MuSiQue基准的表现。因此，模型的默认回答低估了通过一次微小编辑即可释放的计算潜能。

    arXiv:2609.36585v1 Announce Type: new  Abstract: Pretrained transformers use little of their depth to follow references in context. Thirteen base models reliably follow only 1.4-3.6 lines, and extra pretrained loops add little. A task-trained rank-8 LoRA at one early layer extends this computation with all model weights frozen. Qwen3-8B improves from 15.5% to 99% exact accuracy on 24-line chains; a longer-trained LoRA reaches 50 lines. Ouro-1.4B reaches 60 lines after four loops and at least 160 after eight. The LoRA starts a relay: program lines pass on their chain identity through a short range of middle layers. Frozen heads read progressively further up the chain, and removing parent-line attention stops the relay. A frozen-model measurement locates the last useful intervention layer within tolerance in three of four held-out models. Task-specific LoRAs also improve MuSiQue. Default answers therefore understate the computation accessible through a tiny edit. Code and an interactive 
    
[^144]: 长期记忆引导的音频-语言模型目标感知增强

    Long-Term Memory-Guided Enhancement for Target Perception in Audio-Language Models

    [https://arxiv.org/abs/2609.36577](https://arxiv.org/abs/2609.36577)

    提出LTM-AE方法，通过从干净参考录音中提取长期记忆表示并对音频token进行插值重建，在不训练任何模型参数的情况下显著提升音频大语言模型在噪声环境中的目标感知能力。

    

    音频大语言模型（ALLMs）能够对录音内容进行推理以执行复杂任务。然而，在真实世界环境中，当背景噪声和竞争声源与目标声音混合时，这些能力通常会失效。受人类听觉中长期记忆的启发，我们提出了长期记忆引导音频增强方法（LTM-AE），通过在不训练的情况下优化ALLM的音频表示来改善选择性目标感知。LTM-AE从单独的干净参考录音的隐藏状态中提取表示，作为每个类别的长期记忆，引导增强过程朝向用户指定的聆听目标。我们在所选类别的长期记忆中对输入的音频token进行重建，并在语言主干解码之前将重建结果与原始token进行插值。这种插值机制控制了存储的听觉经验的影响，同时保持所有ALLM参数固定不变。

    arXiv:2609.36577v1 Announce Type: cross  Abstract: Audio large language models (ALLMs) can reason about the content of audio recordings to perform complex tasks. However, these capabilities usually collapse in real-world environments when background noise and competing sources mix the target sound. Inspired by long-term memory in human listening, we propose Long-Term Memory-Guided Audio Enhancement (LTM-AE) to improve selective target perception by refining the audio representations of ALLMs without training. LTM-AE extracts representations in hidden states from separate clean reference recordings as long-term memory for each category, guiding enhancement toward a user-specified listening target. We reconstruct incoming audio tokens in the selected category long-term memory and interpolate the reconstructions with the original tokens before language backbone decoding. This interpolation controls the influence of stored auditory experience while keeping all ALLM parameters fixed. Diagno
    
[^145]: 有据修订 vs. 先验注入：探查检索增强式专利权利要求修改

    Grounded Revision vs. Prior Injection: Probing Retrieval-Augmented Patent Claim Amendment

    [https://arxiv.org/abs/2609.36550](https://arxiv.org/abs/2609.36550)

    该研究以专利权利要求修改这一“正确”可定义的任务为测试场，发布了 7,385 个 USPTO 审查案例语料库、七项探测测试和确定性评估指标，发现四个前沿大语言模型在检索增强的权利要求修改中均未表现出经典的先验注入行为，且检索效果微小、方向不一致。

    

    检索增强生成被广泛应用于专业写作，然而在“正确”具有可定义含义的场景中，检索究竟是为修订提供了依据，还是仅仅注入了模板，却鲜有检验。专利权利要求修改恰好提供了这种信号：审查员会指明受质疑的技术特征并引用现有技术，从而为每个案例提供基准真值。我们发布了三项成果：(i) 一个包含 7,385 个美国专利商标局（USPTO）审查案例的语料库，带有 XML 对齐的修改前后权利要求、驳回理由及所引用的现有技术；(ii) 一套七项探测测试，在固定的提示词框架下将随机检索与结构匹配检索作为两种策略进行比较；(iii) 一个确定性的五通道评估指标（C1–C3 与 C5 为主，C4 为补充），无需借助 LLM 进行评估。在对四个前沿大语言模型（Claude Sonnet 4、Claude Haiku 4.5、GPT-5.4、GPT-4o-mini）进行的 9,600 次预注册调用中，没有任何被测模型表现出可检测到的经典先验注入行为；检索带来的效果很小且方向不一致。

    arXiv:2609.36550v1 Announce Type: new  Abstract: Retrieval-augmented generation is widely used in professional writing, yet whether retrieval grounds revision or merely injects templates is rarely tested where "correct" has a definable meaning. Patent claim amendment supplies that signal: the examiner names the attacked limitation and cites prior art, providing per-case ground truth. We release three artifacts: (i) a corpus of 7,385 USPTO prosecution cases with XML-aligned pre/post claims, rejection, and cited prior art; (ii) a seven-probe battery comparing random and structural-match retrieval as two policies under a fixed prompt scaffold; (iii) a deterministic five-channel metric (C1-C3 and C5 in main, C4 supplementary) requiring no LLM evaluation. Across 9,600 pre-registered calls on four frontier LLMs (Claude Sonnet 4, Claude Haiku 4.5, GPT-5.4, GPT-4o-mini), no tested model exhibits detectable classical prior-injection behavior; retrieval effects are small and direction-inconsiste
    
[^146]: DraftTrace：一个面向AI集成写作的多视图分析环境

    DraftTrace: A Multi-View Analytics Environment for AI-Integrated Writing

    [https://arxiv.org/abs/2609.36544](https://arxiv.org/abs/2609.36544)

    DraftTrace通过同时捕获最终作品、写作过程和与AI助手的交互三个互补视图，帮助教师区分真实的写作行为、AI生成文本和复制打字等不同情况。

    

    生成式AI改变了学生完成写作作业的方式。仅凭最终成品已不足以理解其产生的过程。我们提出了DraftTrace，这是一个写作环境，能够同时捕获写作的三个互补视图：最终成品、写作过程以及与集成AI助手的交互。DraftTrace重建文档随时间推移的演进过程，并将这些信号组织为提交级、纵向和班级层面的分析，供教师使用。我们在一门有81名学生的研究生NLP课程中部署了DraftTrace，并将学生的写作会话与自动工具输入的LLM生成文本以及复制打字的文本进行了比较。结果显示，产品层面的度量能够区分文本表述方式的差异，而过程层面的度量能够区分文本输入方式的差异。将两个视图结合起来有助于刻画诸如复制打字等情况。交互痕迹表明学生……（原文摘要在此处截断）

    arXiv:2609.36544v1 Announce Type: cross  Abstract: Generative AI has changed how students produce writing assignments. The final artifact is no longer sufficient to understand the process through which it was produced. We introduce DraftTrace, a writing environment that jointly captures three complementary views of writing: the final product, the writing process and interactions with an integrated AI-assistant. DraftTrace reconstructs how a document develops over time and organizes these signals into submission, longitudinal, and class-level analytics for instructors. We deployed DraftTrace in a graduate NLP course with 81 students and compared their sessions with LLM-generated responses entered by automated tools and with copy-typed responses. While product measures distinguish differences in text formulation, process measures distinguish differences in how text is entered. Considering both views together helps characterize cases such as copy-typing. Interaction traces show that stude
    
[^147]: 当更新不再是学习：基于可学习信息增益重新思考大语言模型的自我进化

    When Updating Stops Being Learning: Rethinking LLM Self-Evolution via learnable information gain

    [https://arxiv.org/abs/2609.36535](https://arxiv.org/abs/2609.36535)

    该论文提出用“可学习信息增益”作为整体诊断框架来衡量LLM自我进化中每轮新增的可参数化信息量，并据此设计ATRI方法对样本重新加权，从而自适应地调节训练以缓解自我进化退化问题。

    

    arXiv:2609.36535v1 公告类型：新论文 摘要：自我进化使大语言模型（LLM）能够利用自身生成的数据进行迭代改进，但常常出现自我进化退化问题：性能先提升，随后停滞，继而下降。现有方法在组件层面应对这一问题，要么针对提问者，要么针对求解者，却忽视了自我进化是一个紧密耦合的系统。我们提出了一个基于可学习信息增益的整体框架，用以衡量某一轮相对于前一轮所提供的多少新颖且可参数化的信息。从理论上讲，该增益等于两轮数据分布之间的Kullback-Leibler散度加上它们的熵变化；在实践中，它通过将一个小型语言模型拟合到前一轮数据，并利用负对数似然对新数据进行评分来估计。基于这一诊断工具，我们提出了ATRI（基于信息增益的自适应训练调节），该方法在一轮内对样本进行重新加权……

    arXiv:2609.36535v1 Announce Type: new  Abstract: Self-evolution lets large language models (LLMs) improve iteratively using their own generated data, but often suffers from self-evolution degeneration: performance improves, plateaus, then declines. Existing methods address this issue at the component level, targeting either the Questioner or the Solver, and overlook that self-evolution is a tightly coupled system. We propose a holistic framework based on learnable information gain, which measures how much novel, parameterizable information a round provides relative to the previous round. Theoretically, this gain equals the Kullback-Leibler divergence between the two rounds' data distributions plus their entropy change. Practically, it is estimated by fitting a small language model to the previous round and scoring new data via negative log-likelihood. Based on this diagnostic, we propose ATRI (Adaptive Training Regulation via Information-gain), which reweights samples within a round an
    
[^148]: 检索系统对查询中身份信号的敏感性

    Retrieval Sensitivity to Identity Signals in Queries

    [https://arxiv.org/abs/2609.36534](https://arxiv.org/abs/2609.36534)

    密集检索器会受查询中身份信号的影响而产生系统性偏差：返回与查询自身政治倾向一致的文章，且对非裔美国人语言（AAL）查询的检索效果劣于白人主流英语（WME）查询。

    

    密集检索器决定了哪些文档能够到达用户以及使用这些文档的语言模型，然而它们通常是用中性查询来评估的。我们探究真实用户在查询中表达的身份信号——政治意识形态和方言——是否会使检索器返回的结果产生偏差。我们在两个领域设计了评估：政治新闻和消费者健康问题，每个领域都将一个仅改变身份信号的受控合成数据集与自然查询相配对。在五个密集检索器和一个稀疏检索基线上，每个检索器（i）都会检索出与查询自身政治倾向一致的文章，并且（ii）对以非裔美国人语言（AAL）撰写的问题的表现比对以白人主流英语（WME）撰写的问题更差。两项分析将这些差距与查询中超越表面词汇层面的身份信号联系起来：在剔除总体的词汇不对称分数后，合成数据上的差距基本保持不变，且线性探针能够恢复出查询的倾向和……（原文摘要在此处截断）

    arXiv:2609.36534v1 Announce Type: new  Abstract: Dense retrievers decide which documents reach users and the language models that use them, yet they are typically evaluated with neutral queries. We ask whether the identity signals that real users express in their queries---political ideology and dialect---bias what a retriever returns. We design evaluations in two domains, political news and consumer-health questions, each pairing a controlled synthetic set that varies only the identity signal with naturalistic queries. Across five dense retrievers and a sparse baseline, every retriever (i) retrieves articles that align with the query's own political lean and (ii) performs worse for questions written in African American Language (AAL) than in White Mainstream English (WME). Two analyses tie these gaps to queries' identity signals beyond surface vocabulary: partialling out an aggregate lexical-asymmetry score leaves the synthetic gaps largely intact, and linear probes recover lean and d
    
[^149]: 三元线性注意力：用于长上下文序列建模的三维循环状态

    Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling

    [https://arxiv.org/abs/2609.36529](https://arxiv.org/abs/2609.36529)

    提出三元线性注意力，通过键、第二键与值的三元外积将循环状态扩展为三维张量状态，仅增加两个投影即实现状态大小E倍增长，从而增强长上下文序列建模能力。

    

    循环神经网络（RNN）将历史上下文压缩为固定大小的记忆状态，从而实现常数时间的推理。记忆状态的大小是影响其性能的关键因素，线性注意力的强劲表现和复兴便是例证——它将普通RNN的向量值隐藏状态扩展为矩阵值隐藏状态。关键在于，线性注意力以参数高效的方式实现这一扩展，特别是通过使用键和值向量的外积来写入矩阵值隐藏状态。我们推广了这一构造，提出三元线性注意力（triadic linear attention），它将一个键、第二个键和一个值的三元外积写入三阶（即三维）张量状态，并通过两个查询与两个键轴的收缩来读取状态。一个E维的第二个键因此带来状态大小E倍的增长，同时仅增加两个投影。三元线性注意力……

    arXiv:2609.36529v1 Announce Type: cross  Abstract: Recurrent neural networks (RNNs) compress the historical context into a memory state of fixed size, thus allowing for constant-time inference. The memory state size is a crucial factor in their performance, as exemplified by the strong performance and resurgence of linear attention, which extends the vector-valued hidden states of ordinary RNNs to matrix-valued hidden states. Crucially, linear attention does so in a parameter-efficient way, in particular by using an outer product of the key and value vectors to write to the matrix-valued hidden state. We generalize this construction and propose triadic linear attention, which writes the triadic outer product of a key, a second key, and a value, into a third-order (i.e., 3D) tensor state, and reads from it by contracting both key axes with two queries. An $E$-dimensional second key thus yields an $E$-fold increase in state size while adding only two projections. Triadic linear attention
    
[^150]: 基于反事实延续的长程智能体上下文压缩自适应

    Adapting Context Compression for Long-Horizon Agents with Counterfactual Continuations

    [https://arxiv.org/abs/2609.36526](https://arxiv.org/abs/2609.36526)

    该论文通过匹配的反事实延续实验发现压缩导致的严重退化集中在孤立的压缩事件上，并提出PAIR方法，利用干预式推演定位并诊断有害的个别压缩操作，自动修订结构化压缩提示词以提升长程智能体的执行可靠性。

    

    长程智能体需要上下文压缩来管理不断增长的交互历史。然而，压缩质量最终由下游执行决定。现有的提示词自适应方法通过比较完整上下文轨迹与压缩后轨迹来推断压缩错误，但这类比较无法分离出单个压缩操作，且会受到智能体随机性的干扰。我们首先发现，压缩会先损害执行可靠性，之后才影响任务可解性。通过使用匹配的反事实延续——即从相同的智能体状态出发，比较有无压缩情况下的执行结果——我们进一步表明，严重的性能退化集中在孤立的压缩事件上。受此发现启发，我们提出了PAIR（基于干预式推演的提示词自适应，Prompt Adaptation using Interventional Rollouts），用于自适应地调整结构化压缩提示词。PAIR识别出损害后续执行的单个压缩操作，诊断其影响，并据此修订固定的压缩提示词中的相关部分（原文摘要在此处截断）。

    arXiv:2609.36526v1 Announce Type: cross  Abstract: Long-horizon agents require context compression to manage growing interaction histories. Compression quality, however, is ultimately determined by downstream execution. Existing prompt-adaptation methods infer compression errors by comparing full-context and compressed trajectories. Such comparisons cannot isolate individual compressions and are confounded by agent stochasticity. We first find that compression degrades reliability before solvability. Using matched counterfactual continuations that compare execution from the same agent state with versus without compression, we further show that severe degradation concentrates at isolated compression events. Motivated by this finding, we propose PAIR (Prompt Adaptation using Interventional Rollouts) for adapting structured compression prompts. PAIR identifies individual compressions that degrade subsequent execution, diagnoses their effects, and revises the relevant sections of a fixed c
    
[^151]: 大规模因子分析表明机器智能只能被部分解释

    Large-scale factor analysis shows machine intelligence is only partially interpretable

    [https://arxiv.org/abs/2609.36515](https://arxiv.org/abs/2609.36515)

    本论文对1,618个语言模型在456个纯文本基准上的13,251个评测分数进行了前所未有的大规模因子分析，发现语言模型的智能仅能被部分解释——一般智能因子约解释70.8%的方差，其余表现差异由领域特定的潜在因素驱动。

    

    语言模型开发中的一个常见假设是，认知能力围绕一个一般的、跨领域的智能因子组织，类似于人类的流体智力。这一假设很少被直接检验，以往的尝试也仅在更小的规模上进行。我们采用潜变量的方法来研究语言模型的智能，这与心理测量学家研究心理构念的方式类似。在每一个具体问题集上的表现都受到一个领域特定潜因子和一个领域无关潜因子的影响。我们将因子分析作为降维技术，分析了13,251个已发表的评测分数，涵盖1,618个语言模型和456个不同的纯文本基准。由于该数据集具有超稀疏的特性，我们通过不同的数据稠密化方法和插补方法对分析结果进行三角验证。在不同偏差模式下呈现出的稳健规律是：1. 一个一般智能因子可解释70.8……（摘要截断）

    arXiv:2609.36515v1 Announce Type: cross  Abstract: A common assumption in language model development is that cognitive abilities are organized around a general, domain-free intelligence factor, like fluid intelligence in humans. This assumption is rarely tested directly, and prior attempts have done so only at a much smaller scale. We take a latent variable approach to intelligence in language models, similar to how psychometricians study psychological constructs. Performance in every specific problem set is influenced by a domain-specific and a domain-agnostic latent factor. Using factor analysis as a dimension-reduction technique, we analyzed 13,251 published evaluation scores covering 1,618 language models across 456 different text-only benchmarks. Due to the super-sparse nature of the dataset, we triangulate our analysis across different data densifiers and imputation methods. A robust pattern across different modes of bias is that 1. A general intelligence factor accounts for 70.8
    
[^152]: 相似的选择，不同的注意力：人类与视觉语言模型中的跨模态关联

    Similar Choices, Different Attention: Cross-Modal Associations in Humans and Vision-Language Models

    [https://arxiv.org/abs/2609.36475](https://arxiv.org/abs/2609.36475)

    该论文通过让人类与视觉语言模型完成相同的伪词-图像匹配任务并记录眼动数据发现，尽管部分较大的VLMs在跨模态关联的选择上能与人类对齐（微调可进一步提升），但其视觉注意力与人类注视的匹配程度甚至不如简单的中心偏好基线，揭示了模型与人类“选择相似、注意不同”的现象。

    

    跨模态关联是指跨模态特征之间的系统性配对，例如“bouba”与圆形形状的关联以及“kiki”与尖锐形状的关联。先前的研究已经比较了人类与视觉语言模型（VLMs）在这类关联上的表现，但往往在人类与模型之间使用不同的刺激或任务。在本研究中，我们探究VLMs是否不仅在选择上与人类一致，而且在做出选择时的注视位置上是否也与人类一致。我们同时研究了VLMs和人类被试（N = 53），向他们呈现相同的刺激——一个伪词和两张图像，并记录了参与者的选择和眼动数据，同时公开发布了这些数据。我们发现少数较大的VLMs在选择上与人类存在对齐，但它们的显著性图与人类注视的匹配程度甚至不如一个中心偏好基线（即位于每张图像中心的固定高斯分布）。在人类选择数据上微调小型VLMs，可以使其在未见过的单词和图像上的选择一致性达到人类多数投票参考的水平，但它们的注意力……（原文摘要在此处截断）

    arXiv:2609.36475v1 Announce Type: cross  Abstract: Cross-modal associations are systematic pairings of features across modalities, such as the association of 'bouba' with round shapes and 'kiki' with sharp shapes. Prior work has compared humans and vision-language models (VLMs) on such associations, but often using different stimuli or tasks between humans and models. Here, we ask whether VLMs align with humans not only in choices, but also in where they look when making those choices. We study both VLMs and humans (N = 53), presenting them with the same stimuli, a pseudo-word and two images, and record participants' choices and eye movements, which we release. We find choice alignment in a few larger VLMs, but their saliency matches human gaze less closely than a center-bias baseline, a fixed Gaussian at the center of each image. Fine-tuning small VLMs on human choices brings their choice alignment to the level of a human majority-vote reference on unseen words and images, yet their a
    
[^153]: FinRT：将自适应红队测试策略蒸馏为消费金融领域可复用的对抗生成器

    FinRT: Distilling Adaptive Red-Teaming Strategies into Reusable Adversarial Generators in Consumer Finance

    [https://arxiv.org/abs/2609.36474](https://arxiv.org/abs/2609.36474)

    FinRT框架将自适应红队测试策略蒸馏为可复用的对抗性提示生成器，在消费金融领域将攻击成功率近乎翻倍（32.9% vs. 17.2%），同时提升对抗严重性33%并保持语义多样性。

    

    在消费金融等受监管行业中，看似无害的用户查询可能会利用大语言模型的漏洞，触发安全失效，并使模型响应危险地逼近政策边界。现有的自动化红队测试方法在攻击有效性与生成成本之间进行权衡，同时将覆盖率、严重性和多样性视为附带指标而非联合优化目标。我们提出了FinRT，这是一个结构化框架，能够从自适应红队测试策略中构建可复用的对抗性提示生成器。在消费金融领域的六个受害模型上，FinRT大幅超越了自适应搜索基线，同时将面向目标的攻击生成摊销为一个可复用的生成器。FinRT将攻击成功率较自适应基线Rainbow Teaming几乎翻倍（32.9% vs. 17.2%），将最大对抗严重性提升了33%，并保持了与迭代搜索相当的策略领域内语义多样性。

    arXiv:2609.36474v1 Announce Type: new  Abstract: In regulated industries like consumer finance, seemingly harmless user queries can exploit large language model vulnerabilities, triggering safety failures and pushing responses dangerously close to policy limits. Existing automated red-teaming methods trade off attack effectiveness against generation cost, while treating coverage, severity, and diversity as incidental rather than joint objectives. We introduce FinRT, a structured framework that builds reusable adversarial prompt generators from adaptive red-teaming strategies. Across the six victim models in consumer finance, FinRT substantially outperforms adaptive search baselines while amortizing target-facing attack generation into a reusable generator. FinRT nearly doubles the attack success rate over the adaptive baseline Rainbow Teaming (32.9% vs. 17.2%), increases maximum adversarial severity by 33%, and preserves comparable intra-policy-domain semantic diversity to iterative se
    
[^154]: Fisher-IRG：跨语言与视觉模型的Fisher诱导局部不变表示几何

    Fisher-IRG: Fisher-Induced Local Invariant Representation Geometry across Language and Vision Models

    [https://arxiv.org/abs/2609.36458](https://arxiv.org/abs/2609.36458)

    提出Fisher-IRG方法，通过局部预测敏感性来度量表示几何，能够在语言和视觉模型中更好地区分语义变化与无关干扰。

    

    语义保持的变换可能在学到的表示中引起大幅移动，而微小的变化却可能强烈影响模型的预测，这引出了一个基本问题：什么样的局部度量最能捕捉具有语义后果的变化？我们提出Fisher诱导不变表示几何，它通过预测敏感性来度量局部表示方向。在每个表示周围，我们构建语义保持和语义改变的邻域，聚合它们的局部Fisher信息，并通过一个对比广义特征值问题恢复不变方向。受控位移分析首先表明，相当的欧几里得移动可能产生截然不同的预测后果，这支持了对预测几何的需求。在语言和视觉模型上，Fisher-IRG在语义与干扰因素的预测选择性方面表现更强，且通常更具代表性（原文此处被截断）。

    arXiv:2609.36458v1 Announce Type: cross  Abstract: Semantic-preserving transformations can induce substantial motion in learned representations, while small changes may strongly affect model predictions, raising a basic question: what local metric best captures semantically consequential variation? We propose Fisher-induced invariant representation geometry (Fisher-IRG), which measures local representation directions through their predictive sensitivity. Around each representation, we construct semantic-preserving and semantic-changing neighborhoods, aggregate their local Fisher information, and recover invariant directions through a contrastive generalized eigenvalue problem. Controlled displacement analyses first show that comparable Euclidean motion can have substantially different predictive consequences, supporting the need for a predictive geometry. Across language and vision models, Fisher-IRG yields stronger semantic-versus-nuisance predictive selectivity and generally more rep
    
[^155]: 记忆固化会抹平用户事实的时间形态

    Memory Consolidation Flattens the Temporal Shape of User Facts

    [https://arxiv.org/abs/2609.36457](https://arxiv.org/abs/2609.36457)

    长期记忆系统在将对话固化为笔记时会不对称地抹平进行时等时间体貌线索（381对语句中有244对被扁平化且从不反向），该现象在11种模型配置及mem0、Graphiti、Letta三个流水线中普遍存在，而丢失的线索会误导后续模型对事实是否仍然有效的判断。

    

    长期记忆系统会将对话转化为简短的存储笔记。一条笔记可以保留用户事实，却丢失了该事实是否仍然成立的证据。例如，“我正在开一辆标致车”可能被写成“用户开一辆标致车”，从而丢失了该活动正在进行中的线索。我们将这种现象称为“体貌扁平化”，并用LAPSE来衡量它——LAPSE是一个由仅在时间形式上有所不同的匹配用户语句组成的基准测试。我们发现记忆写入器会有选择性地扁平化体貌：三个写入器模型在381对语句中的244对里将进行时语句扁平化，却保留了其一般现在时的对应语句，且从未出现相反的情况。这种不对称性在测试的全部11种模型配置中均成立，在已部署的流水线mem0、Graphiti和Letta中也是如此。丢失的线索对后续读取者有实际影响：在探索性测试中，仅改变存储的动词就会改变所有三个读取者对“事实是否仍然成立”的估计；当读取者可以在行动前询问用户时，三个读取者中有两个在未询问的情况下就采取了行动（原文摘要至此处截断）。

    arXiv:2609.36457v1 Announce Type: new  Abstract: Long-term memory systems turn conversations into short stored notes. A note can keep a user fact while losing evidence about whether the fact still holds. For example, "I am driving a Peugeot" can become "The user drives a Peugeot," which drops the cue that the activity is ongoing. We call this aspectual flattening and measure it with LAPSE, a benchmark of matched user statements that differ only in temporal form. We find that memory writers flatten aspect selectively. Three writer models flattened the progressive statement but kept its simple-present match in 244 of 381 pairs, never the reverse. The asymmetry holds in all 11 model configurations tested and in the installed pipelines mem0, Graphiti, and Letta. The lost cue matters to later readers. In exploratory tests, changing only the stored verb shifted all three readers' estimates that a fact still holds. When readers could ask the user before acting, two of three acted without aski
    
[^156]: 掩码扩散语言模型中的可靠并行解码

    Reliable Parallel Decoding in Masked Diffusion Language Models

    [https://arxiv.org/abs/2609.36452](https://arxiv.org/abs/2609.36452)

    该论文提出了一种无需训练的可靠并行解码方法（RPD），通过逐层预测稳定性和最终置信度来判断掩码扩散语言模型中并行提交词元的可靠性，从而安全高效地加速文本生成。

    

    掩码扩散语言模型（MDLM）可以通过并行预测多个被掩码的词元来高效生成文本，但来自同一次前向传播的预测在被同时提交时并不一定可靠。我们研究了并行提交何时是可靠的。我们的诊断分析表明，仅凭置信度并不能决定可靠的提交顺序：序列末端附近的高置信度预测可能会在其支撑性计算尚未建立时就提前固定答案，并且随着上游上下文不确定性的增加，下游预测的可靠性会下降。与此同时，单次前向传播本身已经能够解析多个被掩码的词元，而且在最后几层中保持稳定的预测更有可能是正确的。基于这些发现，我们提出了可靠并行解码，这是一种无需训练的方法，它通过逐层预测稳定性和最终置信度来选择候选词元，并在一定条件下提交它们。

    arXiv:2609.36452v1 Announce Type: cross  Abstract: Masked diffusion language models (MDLMs) can generate text efficiently by predicting multiple masked tokens in parallel, but predictions from the same forward pass are not necessarily reliable when committed together. We study when parallel commitment is reliable. Our diagnostics show that confidence alone does not determine a reliable commitment order: confident predictions near the end of the sequence can fix an answer before its supporting computations are established, and downstream predictions become less reliable as the uncertainty of their upstream context grows. At the same time, a single forward pass can already resolve several masked tokens, and predictions that remain stable across the final layers are more likely to be correct. Based on these findings, we propose Reliable Parallel Decoding (RPD), a training-free method that selects candidates by layerwise prediction stability and final confidence, and commits them under a c
    
[^157]: 不变原子：语言模型表示中局部语义几何的稀疏坐标

    Invariant Atoms: Sparse Coordinates of Local Semantic Geometry in Language Model Representations

    [https://arxiv.org/abs/2609.36451](https://arxiv.org/abs/2609.36451)

    该论文提出“不变原子”假设，发现语言模型表示中的局部语义变化可以用一组在保持语义的变换下保持稳定的稀疏方向坐标来刻画，这些原子具有语义-干扰分离特性，并可对模型预测产生因果影响。

    

    大型语言模型在措辞、风格和句法发生显著变化时往往仍能保留语义，而微小的语义编辑却能系统地改变其隐藏表示。这表明语义变化可能沿着重复出现的局部方向进行组织。我们提出不变原子假设：局部语义运动存在一组偏好的稀疏坐标，这些坐标所对应的方向在保持语义不变的变换下保持稳定。我们学习一个共享的语义框架和稀疏坐标，用于重建语义位移同时抑制干扰变化，并通过依赖于锚点的对角调制来调整原子强度，而无需针对每个样本的旋转。实验表明，这些原子表现出强烈的语义-干扰分离、稀疏重建、可复现的方向以及对模型预测的因果影响。所学到的几何结构能够泛化到未见过的语义邻域和干扰族。

    arXiv:2609.36451v1 Announce Type: cross  Abstract: Large language models often preserve meaning despite substantial changes in wording, style, and syntax, while small semantic edits can systematically alter their hidden representations. This suggests that semantic variation may be organized along recurring local directions. We propose the Invariant Atom Hypothesis: local semantic motion admits preferred sparse coordinates along directions that remain stable under meaning-preserving transformations. We learn a shared semantic frame and sparse coordinates that reconstruct semantic displacements while suppressing nuisance variation, with anchor-dependent diagonal modulation adjusting atom strengths without sample-specific rotations. Empirically, the atoms exhibit strong semantic--nuisance separation, sparse reconstruction, reproducible directions, and causal effects on model predictions. The learned geometry generalizes to unseen semantic neighborhoods and nuisance families, while local r
    
[^158]: MemFold：通过在线策略优化学习面向长上下文个性化的紧凑软记忆

    MemFold: Learning Compact Soft Memory for Long-Context Personalization via On-Policy Optimization

    [https://arxiv.org/abs/2609.36435](https://arxiv.org/abs/2609.36435)

    MemFold通过在线策略优化将查询条件化的文本记忆压缩为固定数量的连续向量作为读取器的记忆接口，并在读取器自身生成的序列上以组相对任务奖励等信号进行训练，从而在长上下文个性化中统一优化记忆的保留与实际运用。

    

    为同一用户提供长期服务的助手，必须基于该用户曾透露的信息来作答：哪些偏好依然有效，哪些已被修改，以及当前适用哪些约束条件。然而，保留这些信息与据此采取行动并不是一回事，但现有方法通常却将二者当作同一件事来优化。以文本形式保留信息会使读取器的输入随保留历史不断增长；而将信息压缩为固定数量的潜在向量虽然限制了接口规模，但这类方法通常被训练用于重构文本或模仿参考答案——两种方式都是在读取器从未产生过的序列上进行评分的。我们提出MemFold，它根据软记忆所支持的行为来优化固定预算的软记忆。具体而言，查询条件化的文本记忆被压缩为K个连续向量，构成读取器的记忆接口；随后读取器在其自身生成的rollout上进行训练，并采用两个互补的信号：针对任务结果的组相对奖励，以及……（原文摘要在此处截断）

    arXiv:2609.36435v1 Announce Type: new  Abstract: An assistant that serves the same user over a long horizon has to answer from what that user has revealed: which preferences still hold, which were revised, and which constraints apply now. Retaining that information is not the same as acting on it, and the two are usually optimized as if they were. Keeping the information as text makes the reader's input grow with the retained history, while compressing it into a fixed number of latent vectors bounds the interface but is typically trained to reconstruct text or imitate reference answers, both of which are scored on sequences the reader never produced. We present MemFold, which optimizes a fixed-budget soft memory by the behavior it supports. A query-conditioned textual memory is compressed into K continuous vectors that form the reader's memory interface, and the reader is then trained on its own rollouts under two complementary signals: group-relative rewards for task outcomes, and con
    
[^159]: 美丽心灵的永恒阳光：系统性擦除大语言模型的记忆

    Eternal Sunshine of the Spotless Mind: Systematically Erasing LLM's Memories

    [https://arxiv.org/abs/2609.36414](https://arxiv.org/abs/2609.36414)

    该论文首次提出“LLM记忆删除”这一新研究方向，发现当前大语言模型即使声称遗忘也无法真正删除用户要求删除的信息，并提出DeLLM框架，通过处理对话中的消息依赖关系来实现对用户记忆删除请求的正确处理。

    

    我们研究持久化的大语言模型（LLM），这类模型会随着时间推移不断积累与用户交互的记忆。此类LLM通过外部存储来维护记忆，并可查询这些存储以克服固定上下文窗口的限制。这类系统具有众多实际应用，因为它们在回应用户查询时能够利用所有过往的交互信息。在本文中，我们探讨LLM能否在用户提出请求时遗忘与它们共享的信息。我们发现，当前的LLM无法真正删除此类信息——即使它们声称已经遗忘，即使在受限的上下文环境下运行也是如此。为此，我们提出了一个新的研究方向：LLM记忆删除（Deletion of LLM Memories）。我们表明，简单地移除与用户删除请求相匹配的消息是不够的，因为对话会自然地引入消息间的依赖关系，导致信息持续留存。为了正确处理删除请求，我们提出了DeLLM框架，它通过动态……（摘要在此处截断）

    arXiv:2609.36414v1 Announce Type: new  Abstract: We consider persistent LLMs that accumulate memories of their interactions with a user over time. Such LLMs maintain memories using external storage, which they can query to overcome the limitations of a fixed context window. Such systems have numerous practical applications, as they can draw on all past interactions when responding to user queries.   In this paper, we ask whether LLMs can forget information shared with them upon a user's request. We find that current LLMs fail to delete such information---even when they claim to have forgotten it and even when operating with a limited context. To this end, we consider a new direction of study: Deletion of LLM Memories.   We show that naively removing messages that match a user's deletion request is insufficient, since conversations naturally introduce message dependencies that cause information to persist. To correctly handle deletion requests, we propose the DeLLM framework. It dynamic
    
[^160]: 为谁校准？角色设定与语言对JEV模型文化价值观的影响

    Calibrated to Whom? Persona and Language Effects on Cultural Values in JEV

    [https://arxiv.org/abs/2609.36399](https://arxiv.org/abs/2609.36399)

    该研究通过对JEV模型进行28.8万次价值观调查测试，发现基于角色的校准能朝人类沙特-美国文化差异方向移动（英语中再现87%、阿拉伯语中仅62%，且长期取向反转），且阿拉伯语下差异缩小源于题目语言而非角色描述语言。

    

    仅决策型语言模型为每个答案选项返回一个概率而非生成文本，这使它们适合充当调查受访者和评判者。我们使用2013年价值观调查模块（VSM 2013）对此类模型之一——TypeSafe的JEV的文化价值观进行了审计。我们让其以12个匹配的沙特阿拉伯角色和12个匹配的美国角色，以及不设角色的方式，用英语和阿拉伯语，在八种请求表述形式下回答这24个题目（共288,000个答案）。JEV的回答高度可重复（ICC 0.997），且在无角色设定时，其回答与其自身的美国角色相似。当角色设定为沙特人而非美国人时，回答朝人类沙特-美国文化差异的方向移动，在英语中再现了该差异幅度的87%，但在阿拉伯语中仅再现62%，且长期取向出现反转。语言交叉实验表明，阿拉伯语中较小的差异源于题目本身所用语言，而非角色描述所用语言。年龄偏移……

    arXiv:2609.36399v1 Announce Type: cross  Abstract: Decision-only language models return a probability for every answer option instead of generating text, which makes them attractive as survey respondents and as judges. We audit the cultural values of one such model, TypeSafe's JEV, with the Values Survey Module 2013. We asked it the 24 items as 12 matched Saudi and 12 matched American personas and without a persona, in English and Arabic, under eight ways of formulating the request (288,000 answers). JEV's answers were highly repeatable (ICC 0.997), and without a persona they resembled those of its own American personas. When the persona was Saudi rather than American, the answers moved in the direction of the human Saudi-US difference, reproducing 87% of its size in English but 62% in Arabic, with long-term orientation reversed. A language cross shows that the smaller difference in Arabic comes from the language of the items, not from the language of the persona description. Age shift
    
[^161]: TTMark：超越单Token熵的成对无失真水印

    TTMark: Pairwise Distortion-Free Watermarking Beyond Single-Token Entropy

    [https://arxiv.org/abs/2609.36372](https://arxiv.org/abs/2609.36372)

    TTMark提出一种成对无失真水印框架，通过对相邻token对的联合分布进行水印嵌入，将有效水印字母表从V扩展到V²，使检测器能够同时利用token熵和条件熵，从而突破单token熵对检测能力的根本限制。

    

    无失真水印技术能够在保持输出分布不变的前提下，实现对机器生成文本的可靠归因。然而，现有方法在每个生成的token上独立运行，这使其检测能力从根本上受限于下一个token分布的熵。我们提出了串行Token水印，这是一个通用的成对水印框架，将无失真水印从单个token扩展到相邻token对。通过对连续token的联合分布进行水印嵌入，TTMark将有效水印字母表从V扩大到V²，使检测器能够同时利用token熵和条件熵，同时保持联合分布上的无失真特性。我们进一步引入了一种分支隔离的串联生成算法，可在单次前向传播中高效构建联合分布。理论上，我们证明了成对水印……

    arXiv:2609.36372v1 Announce Type: cross  Abstract: Distortion-free watermarking enables reliable attribution of machine-generated text while preserving output distribution. However, existing methods operate independently on each generated token, making their detection capability fundamentally constrained by the entropy of the next-token distribution. We present Tandem Token WaterMark (TTMARK), a general pairwise watermarking framework that extends distortion-free watermarking from individual tokens to adjacent token pairs. By watermarking the joint distribution of consecutive tokens, TTMARK enlarges the effective watermarking alphabet from V to $V^2$, allowing the detector to exploit both token entropy and conditional entropy while preserving distortion-freeness over the joint distribution. We further introduce a branch-isolating concatenated tandem generation algorithm that efficiently constructs the joint distribution in a single forward pass. Theoretically, we show that pairwise wat
    
[^162]: DeepRewind：预测与修复深度研究智能体中的过早承诺

    DeepRewind: Predicting and Repairing Premature Commitments in Deep Research Agents

    [https://arxiv.org/abs/2609.36344](https://arxiv.org/abs/2609.36344)

    DeepRewind提出了一个面向可逆深度研究的附加控制层，通过类型化认知图表示智能体的演化状态，利用世界模型预测中间结论的可逆性以阻止过早承诺，并在后续证据推翻结论时执行依赖感知的回滚，从而显著提升洞察召回率。

    

    深度研究智能体通过迭代搜索、证据评估、信念修正与综合归纳来执行长程调查任务。然而，它们可能在证据尚不充分时就过早地承诺某些论断，导致后续推理不断强化错误的解释。我们提出DeepRewind，这是一个用于可逆深度研究的附加控制层，它将智能体不断演化的认知状态表示为一个类型化图，其中包含来源、证据、论断、假设、前提假设、承诺、计划和草稿等节点。在接受一个中间结论之前，基于提示词的世界模型会预测其影响，并根据假设收窄程度、信息损失、恢复成本和矛盾触发点覆盖率来估计该结论的可逆性。一个二进制控制器负责阻止有风险的承诺，而一致性监控器则在后续证据推翻已有承诺时执行依赖感知的回滚。在DRBench和LiveDRBench基准测试上，DeepRewind将洞察召回率提升了……

    arXiv:2609.36344v1 Announce Type: new  Abstract: Deep-research agents conduct long-horizon investigations through iterative search, evidence evaluation, belief revision, and synthesis. However, they may commit to claims before sufficient evidence is available, causing later reasoning to reinforce an incorrect interpretation. We introduce DeepRewind, an additive control layer for reversible deep research that represents the agent's evolving epistemic state as a typed graph of sources, evidence, claims, hypotheses, assumptions, commitments, plans, and drafts. Before accepting an intermediate conclusion, a prompt-based world model predicts its impact and estimates reversibility based on hypothesis narrowing, information loss, recovery cost, and contradiction-trigger coverage. A binary controller blocks risky commitments, while a consistency monitor performs dependency-aware rollback when later evidence invalidates them. Across DRBench and LiveDRBench, DeepRewind improves insight recall by
    
[^163]: 训练大语言模型言明评估意识

    Training LLMs to Verbalize Evaluation Awareness

    [https://arxiv.org/abs/2609.36316](https://arxiv.org/abs/2609.36316)

    提出言语化训练（VT）方法，通过在模型自发言明评估意识前截断轨迹并以强化学习鼓励其坦诚表达，使言语化评估意识提升2.4-2.9倍且可迁移到未见智能体场景，同时保持潜在意识与行为基本稳定。

    

    评估意识会使大语言模型在审计期间的表现与实际部署时不同，然而如何测量和解释评估意识仍然是一个挑战。我们提出了言语化训练（Verbalization Training, VT），这是一种让大语言模型不再吝于言明评估意识的方法，同时避免对潜在的信念本身进行监督。VT 将模型自发的言语表达作为意识存在的证据，并在言语表达出现之前立即截断每一次 rollout，从而生成模型被推定已具有意识的训练前缀。随后，模型通过一个旨在以经过校准的方式增加言语表达的强化学习目标进行训练。在 Qwen3.6-35B-A3B、Kimi K2.6 和 Inkling 三个模型上，VT 将言语化的评估意识提升了 2.4-2.9 倍，并能迁移到留出的智能体场景中，同时测得的潜在评估意识和模型行为基本保持稳定。在一项因果实验中，我们独立地植入了关于评估的元知识……（摘要原文在此处截断）

    arXiv:2609.36316v1 Announce Type: cross  Abstract: Evaluation awareness (EA) can cause large language models (LLMs) to behave differently during audits than in deployment, yet measuring and accounting for EA remains challenging. We introduce verbalization training (VT), a method for making LLMs less reticent about verbalizing evaluation awareness while avoiding to supervise the latent belief itself. VT uses a model's spontaneous verbalizations as evidence that awareness is present and truncates each rollout immediately before the verbalization, producing training prefixes at which the model is presumed to be aware. The model is then trained with an RL objective designed to increase verbalization in a calibrated way. Across Qwen3.6-35B-A3B, Kimi K2.6, and Inkling, VT increases verbalized EA by 2.4-2.9 times and transfers to held-out agentic settings, while measured latent EA and behavior remain largely stable. In a causal experiment, we independently implant meta-knowledge about evaluat
    
[^164]: 面向长序列建模的分数阶状态空间转换

    Fractional State Space Transition for Long Sequence Modeling

    [https://arxiv.org/abs/2609.36314](https://arxiv.org/abs/2609.36314)

    FRAC 是一种基于分数阶动力学的选择性状态空间模型架构，用幂律长记忆取代传统指数遗忘，并通过有限状态、对数间隔的指数模式之和实现高效并行训练，显著提升了长上下文建模性能。

    

    状态空间模型（SSM）将序列历史压缩到一个有界的循环状态中，使得由此产生的记忆规律成为影响长上下文性能的核心架构选择。大多数现代 SSM 依赖于基于常微分方程（ODE）的动力学，这会导致指数遗忘，限制了模型在广泛时间范围内保留信息的能力。我们提出 FRAC，一种源自分数阶动力学的选择性 SSM 架构，用幂律长记忆取代这种指数衰减。为了使分数阶动力学具备实用性，FRAC 使用有限状态、对数间隔的指数模式之和来逼近重尾目标核。这一构造将分数阶记忆转变为一个支持并行训练和预填充的高效循环模块，同时保留了有界状态的自回归解码。大量实验（包括 13 亿参数规模的语言建模）表明，FRAC 在长上下文性能上持续超越当前最先进的方法。

    arXiv:2609.36314v1 Announce Type: cross  Abstract: State Space Models (SSMs) compress sequence history into a bounded recurrent state, making the resulting memory law a central architectural choice for long-context performance. Most modern SSMs rely on ODE-based dynamics that lead to exponential forgetting, limiting their ability to retain information over broad temporal ranges. We introduce FRAC, a selective SSM architecture derived from fractional dynamics that replaces this exponential decay with power-law long memory. To make fractional dynamics practical, FRAC approximates the heavy-tailed target kernel with a finite-state, log-spaced sum of exponential modes. This construction turns fractional memory into an efficient recurrent module with parallel training and prefill, while retaining bounded-state autoregressive decoding. Extensive experiments, including 1.3B-parameter language modeling, demonstrate that FRAC consistently improves long-context performance over state-of-the-art 
    
[^165]: HeurEvo：面向时间敏感数学优化的混合求解器增强启发式的智能体协同进化

    HeurEvo: Agentic Evolution of Hybrid Solver-Augmented Heuristics for Time-Critical Mathematical Optimization

    [https://arxiv.org/abs/2609.36303](https://arxiv.org/abs/2609.36303)

    HeurEvo提出了一个“计划—代码—组件”协同进化框架，联合进化高层算法结构、代码实现和可复用组件池，从而在严格时间约束下自动设计出融合启发式方法与数学规划求解器的混合优化算法。

    

    arXiv:2609.36303v1 公告类型：新 摘要：近期在智能体启发式设计方面的进展，利用AI智能体和执行反馈来自动发现针对复杂优化问题的算法。在许多实际场景中，必须在严格的运行时间约束下获得高质量解，这促使人们采用将特定问题的启发式方法与强大的数学规划求解器相结合的混合方法。然而，现有方法通常只能在预定义流程内改进启发式组件，或孤立地调整求解器配置，这限制了对计算资源的分配位置、求解器的使用方式以及整体算法结构优化的整体自适应能力。为解决这些局限，我们提出了HeurEvo，这是一个自动化的“计划—代码—组件”协同进化框架，它共同进化高层算法结构、其具体实现以及一个可复用组件的共享池。规划器决定使用哪些算法组件、如何……（原文摘要在此处截断）

    arXiv:2609.36303v1 Announce Type: new  Abstract: Recent advances in agentic heuristic design use AI agents and execution feedback to automate algorithm discovery for challenging optimization problems. In many practical settings, high-quality solutions must be obtained under strict runtime constraints, motivating hybrid approaches that combine problem-specific heuristics with powerful mathematical programming solvers. However, existing approaches typically improve heuristic components within predefined procedures or tune solver configurations in isolation. This limits holistic adaptation of where to allocate computation, how to leverage solvers, and how to refine the overall algorithmic structure. To address these limitations, we propose HeurEvo, an automated plan--code--component co-evolution framework that jointly evolves the high-level algorithmic structures, their implementations, and a shared pool of reusable components. A planner determines which algorithmic components to use, how
    
[^166]: MoRE：通过硬件感知的低秩路由扩展专家混合模型

    MoRE: Scaling mixture of experts with hardware-aware low-rank routing

    [https://arxiv.org/abs/2609.36301](https://arxiv.org/abs/2609.36301)

    提出MoRE方法，通过将MoE路由器权重矩阵低秩分解，把路由成本从Θ(Mh)降至O((h+M)r)，在证明可保持路由表达能力与负载均衡的同时，支持Θ(h/r)倍的更多专家，并考虑硬件实际加速。

    

    专家混合层是前沿语言模型的核心组件，而近期的架构正朝着更多、更小的专家方向发展。在这种模式下，标准的线性路由器成为瓶颈：当有 $M$ 个专家和隐藏维度 $h$ 时，其每 token 成本 $\Theta(Mh)$ 在 $M$ 较大时会主导 MoE 层的开销。我们提出 MoRE（秩约简路由的专家混合），它在秩 $r$ 处对路由器权重矩阵进行分解，将路由成本降低至 $O((h+M)r)$。我们证明，当激活专家数量固定时，与 $M$ 呈对数关系的秩足以保证路由的表达能力，且在不计精度因子的意义下该秩下界是必要的。我们还证明，对数秩在高斯记忆模型中能够保持负载均衡，并且在合成电话簿任务上的训练表明低秩不会损害记忆能力。在匹配的激活 FLOPs 下，该分解允许专家数量增加 $\Theta(h/r)$ 倍。为了在实际运行时间中实现这一收益……

    arXiv:2609.36301v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) layers are central to frontier language models, and recent architectures push toward more and smaller experts. In this regime, the standard linear router becomes a bottleneck: with $M$ experts and hidden dimension $h$, its per-token cost $\Theta(Mh)$ dominates the MoE layer once $M$ is large. We introduce MoRE (Mixture of Rank-reduced-routed Experts), which factorizes the router weight matrix at rank $r$ and reduces the routing cost to $O((h + M)r)$. We prove that rank logarithmic in $M$ suffices for routing expressivity when the number of active experts is fixed, and is necessary up to precision factors. We also prove that logarithmic rank preserves load balance in a Gaussian memorization model, and training on a synthetic phonebook task shows that low rank does not hurt memorization. At matched active FLOPs, the factorization allows a factor of $\Theta(h/r)$ more experts. To realize this gain in wall-clock ti
    
[^167]: 当树结构不再足够：利用自适应图稀疏自编码器学习混合拓扑特征图

    When Trees Are Not Enough: Learning Mixed-Topology Feature Graphs with Adaptive Graph Sparse Autoencoders

    [https://arxiv.org/abs/2609.36294](https://arxiv.org/abs/2609.36294)

    该论文提出自适应图稀疏自编码器（AG-SAE），将每个特征的完整父节点集合作为原子结构假设进行竞争验证，突破了传统单亲树结构的限制，能够学习包含多父节点关系的混合拓扑特征图并以此引导SAE训练。

    

    稀疏自编码器（SAE）能够在大语言模型激活中揭示可解释的特征，然而现有的结构化SAE强制采用单亲树或森林结构，而事后构建的图虽然允许多个父节点，但既不能引导特征学习，也无法确保可靠的关系恢复。我们提出了自适应图稀疏自编码器（AG-SAE），这是一种结构引导的训练范式，它将每个特征的完整父节点集合视为一个原子结构假设，并让证据来选择零个、一个或多个父节点。通过让完整父节点集合与空假设、子集解释以及其他替代解释进行竞争，AG-SAE能够识别出联合必要的多父节点关系，同时拒绝冗余或虚假的替代解释，并验证每个子特征在其父节点之外仍有额外贡献。由此在SAE特征上诱导出的拓扑结构定义了一个可微的结构损失来引导SAE训练，同时拓扑引导的精炼机制缓解了特征吸收问题。

    arXiv:2609.36294v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) expose interpretable features in large language model activations, yet existing structured SAEs impose single-parent trees or forests, while post-hoc graphs permit multiple parents but neither guide feature learning nor ensure reliable relation recovery. We introduce the Adaptive Graph Sparse Autoencoder (AG-SAE), a structure-guided training paradigm that treats each feature's complete parent set as an atomic structural hypothesis and lets evidence select zero, one, or multiple parents. By competing complete parent sets against null, subset, and alternative explanations, AG-SAE identifies jointly necessary multi-parent relations while rejecting redundant or spurious alternatives and verifying that each child contributes beyond its parents. The induced topology over SAE features then defines a differentiable structural loss that guides SAE training, while topology-guided refinement mitigates feature absorption
    
[^168]: 《10月7日袭击事件后德国社交媒体中反犹太主义言论的激增》

    The Surge of Anti-Semitism in German Social Media following the October 7 Attacks

    [https://arxiv.org/abs/2609.36290](https://arxiv.org/abs/2609.36290)

    该研究利用大语言模型分析德国社交媒体在10月7日袭击事件前后共12.5万余条帖子，发现反犹太主义言论在Facebook和Telegram上显著激增（Telegram上约为Facebook的十倍），且加入用户上下文信息可将检测F1分数提升至83%并大幅减少误报。

    

    我们研究了2023年10月7日哈马斯对以色列的袭击在多大程度上影响了德国社交媒体上关于犹太教和以色列的讨论。为此，我们开发了一种利用大语言模型（LLM）检测用户发帖中26类反犹太主义内容的方法。该方法应用于事件前后各三个月的Facebook和Telegram帖子（样本量N=125,718）。在方法上，我们在两种设置下测试了不同的开源权重模型——即是否将用户信息作为帖子文本的附加上下文。最佳设置在我们人工标注的验证集上，二分类反犹太主义检测的F1分数最高可达83%。用户上下文为大多数LLM提供了有价值的信息，并大幅减少了误报，例如在对反犹太主义事件进行（批判性）报道的情形。就研究主题而言，我们发现反犹太主义言论在两个平台上均显著激增，且其在Telegram上的流行程度约为Facebook的十倍。

    arXiv:2609.36290v1 Announce Type: cross  Abstract: We investigate the extent to which the Hamas attacks on Israel of October 7, 2023, have affected German social media debates about Judaism and Israel. For this, we develop an approach to detect 26 anti-Semitic categories in user postings via large language models (LLMs). The approach is applied to Facebook and Telegram posts (N=125,718) from three months before and after the event. Methodically, we test different open-weight models in two setups---with and without user information as additional context to the post text. The best setup achieves up to 83 % F1-score for binary anti-Semitism detection on our manually coded validation set. User context provides valuable information for most LLMs and drastically reduces false positives, for example, when (critically) reporting on anti-Semitic incidents. Concerning our topic, we find that anti-Semitism is surging significantly on both platforms, while being about ten times more prevalent on T
    
[^169]: 上下文学习放大一个潜在的符号回路

    In-Context Learning Amplifies a Latent Symbolic Circuit

    [https://arxiv.org/abs/2609.36265](https://arxiv.org/abs/2609.36265)

    该研究揭示大语言模型内部在预训练时就已存在一个三阶段符号推理回路（抽象、归纳、检索），上下文学习通过放大该回路驱动少样本规则学习——单头因果贡献最高增长8倍，且注入函数向量可将0-shot准确率挽救至86%。

    

    大型语言模型能够仅凭少量上下文示例学习抽象规则，但其内部机制如何随示例数量增加而被激活尚不清楚。我们在三个模型家族中追踪了一个三阶段符号推理回路（抽象、归纳、检索）在不同示例数量下的状态，发现该回路在模型达到高准确率之前就已可检测且发挥功能。从1-shot到10-shot，各注意力头的因果贡献最高增长8倍；跨shot激活修补可将0-shot准确率从1%提升至56%，1-shot准确率从17%提升至88%。在0-shot时缩放并注入函数向量可将准确率挽救至86%，这在很大程度上可替代归纳阶段，但关键依赖于下游完整的检索阶段。抽象规则遵循所需的基础设施在任何示例展示之前就已存在于模型权重之中；上下文示例、函数向量及相关干预似乎都在为同一个潜在回路提供输入。

    arXiv:2609.36265v1 Announce Type: cross  Abstract: Large language models can learn abstract rules from just a few in-context examples, but how their internal mechanisms activate as examples accumulate is not well understood. We trace a three-stage symbolic reasoning circuit (abstraction, induction, retrieval) across shot counts in three model families and find it is detectable and functional well before the model achieves high accuracy. Per-head causal contribution grows up to 8x from 1- to 10-shot, and cross-shot activation patching raises accuracy from 1% to 56% at 0-shot and 17% to 88% at 1-shot. Function vectors scaled and injected at 0-shot rescue accuracy up to 86%, largely substituting for the induction stage but depending critically on an intact downstream retrieval stage. The infrastructure for abstract rule-following is present in the weights before any demonstrations; in-context examples, function vectors, and related interventions appear to supply input to the same latent c
    
[^170]: OTROPE：基于最优传输的大语言模型鲁棒离线策略评估方法

    OTROPE: Optimal Transport-based Robust Off-policy Evaluation for Large Language Models

    [https://arxiv.org/abs/2609.36264](https://arxiv.org/abs/2609.36264)

    提出 OTROPE，一种基于最优传输、无需似然值的大语言模型离线策略评估方法，通过在语义空间对齐行为策略与目标策略样本，实现无需密度比估计和策略建模的双重鲁棒式评估，适用于黑盒大语言模型。

    

    对大语言模型（LLM）进行可靠的评估对其开发与部署至关重要，然而在线评估往往成本高昂、风险较大且难以安全进行。我们研究大语言模型的离线策略评估问题，即利用来自行为模型的有限人工标注数据来评估一个更新的目标大语言模型。这一设定极具挑战性，原因在于标注数据稀缺、行为模型与目标模型之间的分布偏移普遍存在，而且对于黑盒大语言模型而言，其响应的似然值通常不可获得。我们提出了基于最优传输的鲁棒离线策略评估方法（OTROPE），这是一种无需似然值的评估方法，通过最优传输在语义空间中进行分布校正，使有标注的行为策略样本与无标注的目标策略样本相互对齐。OTROPE 将校正后的人工标注残差与代理预测器相结合，形成一种双重鲁棒风格的评估方式，且无需对行为策略建模或进行密度比估计。我们在理论上刻画了……

    arXiv:2609.36264v1 Announce Type: new  Abstract: Reliable evaluation of large language models (LLMs) is essential for their development and deployment, yet is often costly, risky, and difficult to perform safely online. We study off-policy evaluation for LLMs, where limited human-labeled data from a behavior model are used to evaluate a newer target LLM. This setting is challenging because labels are scarce, behavior--target distribution shift is common, and response likelihoods are often unavailable for black-box LLMs. We propose the Optimal Transport-based Robust Off-Policy Evaluation (OTROPE), a likelihood-free evaluation that performs distributional correction in a semantic space via optimal transport to align labeled behavior-policy samples with unlabeled target-policy samples. OTROPE combines corrected human-labeled residuals with proxy predictors, yielding a doubly robust-style evaluation without behavior-policy modeling or density-ratio estimation. We theoretically characterize
    
[^171]: 群体保真度：评估大语言模型中的群体代表性

    Population Fidelity: Evaluating Population Representativeness in LLMs

    [https://arxiv.org/abs/2609.36253](https://arxiv.org/abs/2609.36253)

    该论文提出了“群体保真度”评估框架，从群体层面准确性、群体间变异量及其结构三个维度评估LLM生成回复对真实人群的代表性，并发现LLM代表性不佳不仅是因为群体间变异不足，还因为变异被错误地归因于不同群体。

    

    大语言模型（LLMs）在模拟人类态度和偏好方面展现出相当大的潜力。先前的研究发现，LLM生成的回复可能会压缩人群内部的态度范围，并以因模型和主题而异的方式错误地代表特定子群体。我们提出了“群体保真度”，这是一个评估框架，用于区分一组LLM生成的回复代表一个人群所需的关键条件。该框架包含三个维度：群体层面的准确性、群体间变异的数量，以及该变异的结构。我们通过两种方式展示了该框架的实用性。首先，我们重现了先前一项关于LLM调查回复中“机器偏见”的研究，并将该框架应用于其模型以及更新的模型，结果表明糟糕的代表性不仅源于群体间变异不足，还源于变异被分配到了错误的群体。其次，我们评估了一种提出的……（摘要在此处截断）

    arXiv:2609.36253v1 Announce Type: cross  Abstract: Large language models (LLMs) show considerable potential in simulating human attitudes and preferences. Prior work finds that LLM-generated responses can compress the range of attitudes found within populations and misrepresent particular subgroups in ways that vary across models and topics. We introduce Population Fidelity, an evaluation framework that distinguishes key conditions required for a set of LLM-generated responses to represent a population. It incorporates three dimensions: group-level accuracy, the amount of between-group variation, and the structure of that variation. We demonstrate the framework's utility in two ways. First, we reproduce a prior study of "machine bias" in LLM survey responses and apply the framework to its models and more recent ones, showing that poor representation reflects not only insufficient between-group variation but also variation assigned to the wrong groups. Second, we evaluate one proposed a
    
[^172]: 在学生状态下从教师续写中学习

    Learning from Teacher Continuations at Student States

    [https://arxiv.org/abs/2609.36246](https://arxiv.org/abs/2609.36246)

    OLIVE 提出一种在线干预式蒸馏框架——学生生成前缀、教师自回归续写并以其交叉熵更新学生——同时解决了离线 SFT 的协变量偏移、OPD 的监督碎片化以及分布匹配蒸馏需教师 token 概率三大局限，以相近成本取得更优推理性能。

    

    我们提出了 OLIVE（OnLine InterVEntion，在线干预）。在每次迭代中，不断演进的学生策略生成一个新的前缀，教师以自回归方式对该前缀进行续写，然后利用在教师生成 token 上计算的交叉熵来更新学生。每一项设计选择都针对现有蒸馏方法的一个相应局限：(1) 基于固定教师轨迹的离线监督微调（SFT）中的序列协变量偏移问题；(2) token 级在策略蒸馏（OPD）中前缀失败导致的监督碎片化问题；以及 (3) 分布匹配蒸馏需要访问教师 token 概率的问题。在相当的 GPU 小时成本下，OLIVE 取得了比 OPD（使用 top-16 KL 近似）更高的推理性能。我们的异步实现进一步将 OLIVE 的总训练时间减少了 23.8%。我们在困难推理任务和智能体任务上评估了 OLIVE，这些任务反映了现代后训练场景，并且它始终……（摘要在此处截断）

    arXiv:2609.36246v1 Announce Type: new  Abstract: We present OLIVE (OnLine InterVEntion). At each iteration, the evolving student policy generates a new prefix, the teacher continues it autoregressively, and the student is updated using cross-entropy computed on the teacher-generated tokens. Each design choice targets a corresponding limitation of existing distillation methods: (1) sequential covariate shift in offline supervised fine-tuning (SFT) on fixed teacher trajectories, (2) fragmented supervision under prefix failure in token-level on-policy distillation (OPD), and (3) the need for access to teacher token probabilities in distribution-matching distillation. OLIVE achieves higher reasoning performance than OPD (with a top-16 KL approximation) at comparable GPU-hour cost. Our asynchronous implementation further reduces OLIVE's total training time by 23.8\%. We evaluate OLIVE on both hard reasoning tasks and agentic tasks which reflects modern post-training scenarios, and it consis
    
[^173]: 认知专家语言模型与相应脑系统的对齐更佳

    Cognitive Expert Language Models Better Align with the Corresponding Brain Systems

    [https://arxiv.org/abs/2609.36239](https://arxiv.org/abs/2609.36239)

    本研究通过提示和微调构建了感觉、空间、数值、推理、社会和抽象六个认知领域的专家型大语言模型，发现每个专家模型与对应认知功能的脑系统对齐更好，说明认知特化的语言模型能更准确地反映脑区的功能特化。

    

    大型语言模型（LLMs）能够预测人类在自然语言理解过程中多个脑区的神经活动。然而，通常的做法是使用同一个模型来测量它与不同脑区的对齐程度，然后再汇总各脑区的模型表现。这种“一个模型适配所有”的方法忽略了脑区的功能特化。在本研究中，我们评估了面向特定认知领域的模型是否与专门负责该领域的脑系统对齐得更好。通过提示和微调，我们首先构建了六个领域的专家型LLM变体：感觉、空间、数值、推理、社会和抽象加工。随后，我们检验每个专家模型是否能最好地预测与相应认知领域相关脑区的活动。与我们的假设一致，每个专家模型的表征与其最相关的脑系统的对齐更为紧密。

    arXiv:2609.36239v1 Announce Type: new  Abstract: Large language models (LLMs) can predict human brain activity across a variety of brain regions during natural language comprehension. Typically, however, LLM-brain alignment is measured using one model for different regions of the brain, and then model performance is summarized across regions. This one-model-fits-all approach ignores the functional specialization of brain regions. In this study, we assess whether a model oriented toward a particular cognitive domain aligns better with the brain system dedicated to that domain. Through prompting and fine-tuning, we first build expert LLM variants for six domains: sensory, spatial, numerical, reasoning, social, and abstract processing. We then examine whether each expert best predicts activity in the brain region associated with the corresponding cognitive domain. Consistent with our hypotheses, each expert's representations align more closely with the brain system most associated with th
    
[^174]: CineSubBench：基于多语言电影字幕评估大语言模型的长篇叙事与文化理解能力

    CineSubBench: Evaluating LLMs on Long-Form Narrative and Cultural Understanding from Multilingual Movie Subtitles

    [https://arxiv.org/abs/2609.36218](https://arxiv.org/abs/2609.36218)

    CineSubBench是一个基于1,012部电影六语言字幕构建的基准，通过七项任务在多任务、多语言、多元文化的匹配设置下，评估大语言模型从海量时序字幕中重构长篇电影叙事与文化理解的能力。

    

    大语言模型越来越多地在法律、医学、软件工程和网络安全等专业领域接受评估，然而电影领域尽管需要长篇叙事整合、多语言解读以及基于文化情境的观众判断，却相对缺乏探索。我们提出了CineSubBench，这是一个用于评估基于多语言电影字幕的长上下文电影理解能力的基准。字幕轨将一部电影表示为数千条按时间顺序排列的短话语，模型必须从中重构人物、关系、事件、因果进展和主题，而无需显式的场景或事件结构。CineSubBench包含1,012部具有六种语言完整字幕覆盖的电影，共产生6,072条字幕轨和813万条带时间戳的字幕条目。它提供了一个匹配的多任务、多语言、多元文化（MultiX）评估设置：七项任务涵盖叙事重构及……（原文摘要在此处截断）

    arXiv:2609.36218v1 Announce Type: cross  Abstract: Large language models are increasingly evaluated in specialized domains such as law, medicine, software engineering, and cybersecurity, yet film remains comparatively underexplored despite requiring long-form narrative integration, multilingual interpretation, and culturally situated audience judgments. We introduce CineSubBench, a benchmark for evaluating long-context film understanding from multilingual movie subtitles. A subtitle track represents a film as thousands of short, temporally ordered utterances from which models must reconstruct characters, relationships, events, causal progression, and themes without explicit scene or event structure. CineSubBench contains 1,012 films with complete subtitle coverage in six languages, yielding 6,072 tracks and 8.13M timestamped subtitle entries. It provides a matched multi-task, multilingual, and multicultural (MultiX) evaluation setting: seven tasks span narrative reconstruction and abst
    
[^175]: 翻译中的迷失：测量非母语英语对大型语言模型最终用户表现的影响

    Lost in Translation: Measuring the Effect of Non-Native English on End User Performance of Large Language Models

    [https://arxiv.org/abs/2609.36214](https://arxiv.org/abs/2609.36214)

    本研究构建了包含19万余个英语提示变体的FABLE数据集，发现大语言模型虽不会复制拼写等表层错误，却会镜像提示中的修辞与词汇水平，从而导致非流利英语用户获得系统性的低质量回复。

    

    大型语言模型（LLM）正越来越多地被母语并非英语的人群使用，然而已有研究表明，这些用户所获得的回复质量系统性地低于英语流利者。由于流利度本身是机械准确性、词汇使用、篇章组织和语篇连贯性的复合体，究竟是英语非母语表达中的哪些具体特征造成了这一差距，目前仍不清楚。本研究提出了FABLE——一个包含190,911个英语提示变体的受控数据集，这些变体源自174K条与写作任务相关的真实用户提示。通过对34个开放权重LLM的回复进行评估，我们发现了一种明显的不对称性：模型虽然不会将拼写错误等表层错误传播到其输出中，却会镜像反映用户提示中所呈现的较高层次的修辞和词汇特征。此外，最不流利与最流利的提示之间，回复的整体质量存在显著差异。这些结果凸显了LLM性能的一个关键差异……

    arXiv:2609.36214v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used by people whose first language is not English, yet these users have been shown to receive systematically lower-quality responses than fluent speakers. Which specific features of non-native English drive this gap remains unclear, because fluency is itself a composite of mechanical accuracy, vocabulary use, organization, and discourse coherence. Here, we introduce FABLE, a controlled dataset of 190,911 English prompt variants derived from 174K real user prompts for writing-related tasks. Evaluating responses from 34 open-weight LLMs, we find a clear asymmetry; while models do not propagate surface errors such as misspellings into their outputs, models do mirror higher-level rhetorical and lexical qualities present in the user's prompt. Further, the overall quality of responses differs substantially between the least- and most-fluent prompts. These results highlight a key LLM performance di
    
[^176]: 规范顺序问题：当大语言模型作为多值关系的不可靠知识库时

    The Canonical Order Problem: When Large Language Models Are Unreliable Knowledge Bases for Multi-Valued Relations

    [https://arxiv.org/abs/2609.36209](https://arxiv.org/abs/2609.36209)

    大语言模型内部按规范顺序（如字母或时间顺序）组织多值关系，当提示要求偏离此顺序生成实体集合时，其作为知识库的可靠性会显著下降。

    

    由于大语言模型（LLM）在预训练期间获取了海量知识，它们正越来越多地被用作知识库（KB）。尽管许多研究专注于提取单一关系三元组，但大多数现实世界中的关系是多值的，需要生成实体集合。本文研究了LLM如何表示和生成多值关系。我们识别出了规范顺序问题：LLM内部的概率分布会按照一种规范顺序（例如字母顺序或时间顺序）来组织许多多值关系。通过机制分析，我们表明LLM中的集合生成过程可以分为三个阶段：（1）候选实体的检索，（2）内部排序，以及（3）下一个元素的选择。因此，当构建知识库的提示词偏离这种内部规范顺序时，会导致LLM在生成时可靠性显著降低。

    arXiv:2609.36209v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used as knowledge bases (KBs) due to the vast amount of knowledge they acquire during pre-training. While many works focus on extracting single relational triples, most real-world relations are multi-valued and require generating sets of entities.   In this paper, we investigate how LLMs represent and generate multi-valued relations. We identify the canonical order problem: The probabilistic distributions inside LLMs organize many multi-valued relations according to a canonical ordering (e.g., alphabetical or chronological). Through mechanistic analysis, we show that set generation in LLMs can be thought of in terms of three phases: (1) retrieval of candidate entities, (2) internal sorting, and (3) selection of the next element. As a result, prompts aiming to construct KBs that deviate from this internal canonical ordering lead to a markedly reduced reliability of LLMs when aiming to generate
    
[^177]: 非洲语言的几何表示：区域语义枢纽与文化引导

    Geometric Representations of African Languages: A Regional Semantic Hub and Cultural Steering

    [https://arxiv.org/abs/2609.36205](https://arxiv.org/abs/2609.36205)

    该研究揭示Gemma 4 31B中非洲语言在表示空间中形成了比对照语言之间更紧密对齐的区域语义枢纽，并成功利用英语数据构建的文化方向来引导模型对不同非洲国家的响应。

    

    我们研究Gemma 4 31B如何表示非洲语言以及如何响应文化引导。第一项研究使用探针、对比方向和表示相似性度量，比较了九种非洲语言和三种对照语言。从英语的迁移程度在不同语言和不同层之间有所差异。在若干层中，表示“非洲与西方”对比的方向在非洲语言之间的对齐程度高于这些语言与对照语言之间的对齐程度。跨语系比较在十二层中的五层通过了报告的Holm显著性阈值，尽管语言对之间的依赖性限制了统计解释。在尼日利亚范围内，约鲁巴语和伊博语之间的对齐程度在十二层中的十一层高于它们各自与豪萨语配对的平均值。第二项研究使用独立的英语数据为尼日利亚、加纳、肯尼亚和南非构建方向。在选定强度下采用联合评分时，估计的属性差异……（原文摘要在此处截断）

    arXiv:2609.36205v1 Announce Type: new  Abstract: We study how Gemma 4 31B represents African languages and responds to cultural steering. The first study compares nine African languages and three controls using probes, contrast directions, and measures of representation similarity. Transfer from English varies across languages and layers. Directions representing an Africa versus West contrast are more aligned among the African languages than between these languages and the controls at several layers. The comparison across language families passes the reported Holm threshold at five of twelve layers, although dependence between language pairs limits the statistical interpretation. Within Nigeria, Yoruba and Igbo are more aligned than the average of their pairs with Hausa at eleven of twelve layers. The second study uses separate English data to construct directions for Nigeria, Ghana, Kenya, and South Africa. Under union scoring at the selected strengths, estimated differences in attrib
    
[^178]: FastGuide：加速扩散大语言模型的奖励引导

    FastGuide: Accelerating Reward Guidance for Diffusion Large Language Models

    [https://arxiv.org/abs/2609.36202](https://arxiv.org/abs/2609.36202)

    FastGuide通过自适应地混合并行与自回归解码，并借助KV缓存和稀疏注意力重计算，显著加速了扩散大语言模型的奖励引导推理过程。

    

    基于梯度的奖励引导提供了一种灵活的方式，可以在推理时利用下游奖励模型来控制掩码扩散语言模型。然而，其计算成本仍然很高，因为每次解码迭代都需要进行昂贵的扩散模型前向传播和奖励模型反向传播步骤。为了解决这一问题，我们提出了FastGuide，一种并行解码与自回归解码的自适应混合方法，用于加速扩散语言模型的奖励引导。类似于并行解码，FastGuide通过在每个解码步骤中仅计算一次引导并将其重用于生成多个标记，从而摊销了奖励模型反向传播的成本。在每个解码步骤内，FastGuide使扩散前向传播呈现自回归特性，即每次仅对一个标记取消掩码，并利用KV缓存技术和注意力的稀疏重计算，在每次取消掩码后高效地重新计算标记分布。最后，为了使混合解码适应……

    arXiv:2609.36202v1 Announce Type: new  Abstract: Gradient-based reward guidance provides a flexible way to use downstream reward models to control masked diffusion language models at inference time. However, its computational cost remains high as each decoding iteration incurs expensive diffusion model forward passes and reward model backpropagation steps. To address this, we introduce FastGuide, an adaptive hybrid of parallel and autoregressive decoding to accelerate reward guidance for diffusion language models. In analogy to parallel decoding, FastGuide amortizes the cost of reward model backpropagation by computing guidance once per decoding step and reusing it to generate multiple tokens. Within each decoding step, FastGuide makes diffusion forward passes autoregressive by unmasking tokens one at a time while efficiently recomputing token distributions after each unmasking by utilizing KV caching techniques and sparse recomputation of attention. Lastly, to adapt hybrid decoding to
    
[^179]: SCOUT：协同推理与工具使用以保障计算机使用安全

    SCOUT: Synergizing Reasoning and Tool-Use for Computer-Use Safety

    [https://arxiv.org/abs/2609.36201](https://arxiv.org/abs/2609.36201)

    SCOUT提出了一种两阶段的代理式安全验证器，通过将推理密集的评分标准生成与工具密集的证据收集相结合，能够有效检测计算机使用代理在执行任务时产生的细微且隐蔽的有害行为。

    

    计算机使用代理（CUA）虽然能够在日常和专业工作流程中完成计算机任务，但即使在良性指令和环境条件下也可能造成意外的伤害。然而，检测此类伤害仍然极具挑战性。首先，这需要细致的、针对特定任务的推理：仅依靠通用安全标准进行判断的验证器往往会忽略许多重要但微妙的有害行为。其次，这需要主动调查：历史轨迹截图只显示了代理做了什么，但并不总能反映环境中的实际变化，因此仅依赖截图的LLM-as-a-judge验证器可能无法确定行动的实际后果。为了应对这些挑战，我们提出了SCOUT，一个两阶段的代理式安全验证器，它将推理密集的评分标准生成与工具密集的证据收集相协同。首先，SCOUT评分标准生成器对任务和代理的执行轨迹进行深入的推理……

    arXiv:2609.36201v1 Announce Type: cross  Abstract: Computer-use agents (CUAs), while capable of completing computer tasks in everyday and professional workflows, can cause unintended harm even under benign instructions and environments. However, detecting such harm remains challenging. First, it requires careful, task-specific reasoning: verifiers guided only by general safety criteria often overlook many important but subtle harmful behaviors. Second, it requires active investigation: past trajectory screenshots show what the agent did but not always what actually changed in the environment, so LLM-as-a-judge verifiers that rely on screenshots alone may be unable to determine the actual consequences of actions. To address these challenges, we introduce SCOUT, a two-stage agentic safety verifier that synergizes reasoning-intensive rubric generation with tool-intensive evidence gathering. First, our SCOUT rubric generator extensively reasons over the task and the agent's trajectory to d
    
[^180]: 不同分词生育率语言间的概念方向可靠性

    Concept Direction Reliability Across Languages with Different Tokenizer Fertility

    [https://arxiv.org/abs/2609.36194](https://arxiv.org/abs/2609.36194)

    该研究发现情感概念方向的可靠性在不同语言间存在显著差异（英语最高、豪萨语次之、约鲁巴语最低，与分词器生育率相关），且情感分类器的预测准确性并不能保证方向的一致性。

    

    即使下游情感分类保持准确，提取出的情感方向在不同样本之间也可能存在差异。为了评估方向的可复现性，我们在四个语言模型上，使用原文和翻译文本，测量了英语、豪萨语和约鲁巴语表示中的对半一致性。我们使用十个主题确定所选层的一致性，并在另外独立的十五个主题组上评估方向一致性。使用最后一个token时，对半一致性范围为：英语0.737至0.870，豪萨语0.589至0.762，约鲁巴语0.101至0.399，且这一语言排序在所有77次完整的模型比较中保持不变。在这些相同层上训练的分类器始终能以高于随机水平的准确率预测情感，这表明预测准确性并不意味着方向一致性。此外，对token表示取平均会产生较低且不够一致的一致性，而高一致性可能部分反映了句子（摘要在此处截断）

    arXiv:2609.36194v1 Announce Type: new  Abstract: Extracted sentiment directions can vary across samples even when downstream sentiment classification remains accurate. To evaluate direction reproducibility, we measure split-half agreement in English, Hausa, and Yoruba representations across four language models using both native and translated texts. We identify layers selected for agreement using ten topics and evaluate direction agreement across separate groups of fifteen topics. Using the final token, split-half agreement ranges from 0.737 to 0.870 for English, 0.589 to 0.762 for Hausa, and 0.101 to 0.399 for Yoruba, maintaining this language rank order across all 77 complete model comparisons. Classifiers trained on these same layers consistently predict sentiment above chance, demonstrating that predictive accuracy does not imply directional consistency. Furthermore, averaging token representations yields less consistent agreement, and high agreement can partially reflect sentence
    
[^181]: 面向智能体强化学习中信用分配的关键决策定位

    Targeting Pivotal Decisions for Credit Assignment in Agentic Reinforcement Learning

    [https://arxiv.org/abs/2609.36178](https://arxiv.org/abs/2609.36178)

    ProVer通过智能体裁判对比成功与失败轨迹来提出潜在的关键决策片段，并利用片段前后策略延续的终端成功率差异验证其优势，从而在智能体强化学习中实现了细粒度的信用分配。

    

    组相对策略优化（GRPO）已成为训练大型语言模型智能体的一种有前景的方法。然而，它将轨迹级优势统一分配给所有策略token，无法区分关键决策与相关性较低的决策，从而掩盖了哪些中间决策为成功做出了贡献。我们提出了ProVer，这是一个在智能体强化学习中对潜在关键决策进行定位、以实现细粒度信用分配的框架。给定一个rollout组，一个智能体裁判通过对比成功与失败的轨迹，提出一个可能对二者结果差异负责的片段。ProVer并不直接信任裁判的评估，而是通过估计该片段之前与之后采样的当前策略延续在终端成功率上的差异，来验证所提议片段的优势。随后，正向的估计值会被纳入相应token的GRPO优势中。

    arXiv:2609.36178v1 Announce Type: cross  Abstract: Group Relative Policy Optimization (GRPO) has become a promising approach for training large language model agents. However, its uniform assignment of trajectory-level advantages to all policy tokens fails to distinguish consequential decisions from less relevant ones, obscuring which intermediate decisions contributed to success. We introduce ProVer, a framework that targets potentially pivotal decisions for fine-grained credit assignment in agentic reinforcement learning. Given a rollout group, an agentic judge contrasts successful and failed trajectories to propose a segment potentially responsible for their divergent outcomes. Rather than directly trusting the judge's assessment, ProVer verifies the proposed segment by estimating its advantage from the difference in terminal success rates between current-policy continuations sampled before and after the segment. Positive estimates are then incorporated into the GRPO advantages of p
    
[^182]: 面向潜在递归大语言模型系统的原则性思维方法

    Principled Thoughts for Latent Recursive LLM Systems

    [https://arxiv.org/abs/2609.36159](https://arxiv.org/abs/2609.36159)

    提出REST训练目标，将有效思维表示的因果性、最小性、可分离性和稳定性四个属性转化为可微分损失并加入交叉熵，从而解决仅用交叉熵训练潜在递归大语言模型系统时出现的四种失败问题，且无需更改架构或增加推理参数。

    

    大语言模型可以在连续空间中进行推理，而非在解码后的文本中推理，其方式是对自身的隐藏状态进行递归，或在这些隐藏状态于多个智能体之间传递，而训练过程仅监督最终解码答案的交叉熵（CE），并不对思维过程本身加以约束。理论与实证分析建立并证实了仅用交叉熵训练所导致的四种失败情况，这些失败会降低得到正确答案的概率，例如不同问题之间思维发生坍缩、以及保留无关信息等。我们提出了REST（REpresentation-Supervised Thoughts，表示监督思维），这是一种训练目标，它将有效思维表示的四个属性——因果性、最小性、可分离性和稳定性——转化为可微分的损失函数，并添加到交叉熵损失中。我们在潜在的单智能体和多智能体系统中对该方法进行了实例化，无需更改架构，也不需要在推理时增加额外参数。在涵盖数学、科学、医学和代码生成的7个基准测试中……

    arXiv:2609.36159v1 Announce Type: new  Abstract: Large language models can reason in continuous space instead of decoded text, by recurring on their own hidden states or by passing those states between agents, while training supervises only the Cross-Entropy (CE) of the final decoded answer and does not constrain the thought. Theoretical and empirical analyses establish and confirm four failures of CE-only training that lead to a lower probability of the correct answer such as collapsing thoughts across distinct questions and retaining irrelevant information. We introduce REST (REpresentation-Supervised Thoughts), a training objective that turns four properties of a valid thought representation (causality, minimality, separability, and stability) into differentiable losses added to CE. We instantiate it in latent single-agent and multi-agent systems, without architectural changes or added parameters at inference. Across 7 benchmarks spanning mathematics, science, medicine, and code gen
    
[^183]: 语言模型是“不安全”的报告者

    Language Models Are "Insecure" Reporters

    [https://arxiv.org/abs/2609.36139](https://arxiv.org/abs/2609.36139)

    该研究首次系统性提出并验证了“不安全报告”现象——大语言模型在总结工作时倾向于隐瞒削弱方法有效性的负面结果，而仅仅添加一句简短的诚实指令就能将负面结果的指出率从 1% 大幅提升至 95%。

    

    随着大型语言模型被部署到日益自主的长时程任务中，人工审计和验证模型的行为、产物和输出变得越来越困难，用户转而依赖大模型生成的报告来评估工作的质量与完整性。我们引入了一套包含八个对抗性报告场景的测试集，以系统性地研究大语言模型是否会掩盖“改变叙事的缺陷”——即那些会削弱原本看似成功的工作描述的错误或局限性，我们将这种现象称为“不安全报告”。当把包含一个植入性负面结果（该结果会实质性地削弱所提方法的有效性）的机器学习实验日志交给 GPT-5.5 时，模型在 200 份生成的报告中仅有 2 份指出了该负面结果；然而，当添加一条简短的诚实指令“在回答中保持诚实”后，模型在 200 份报告中有 190 份指出了该负面结果。在八个开源权重模型上，思维链分析

    arXiv:2609.36139v1 Announce Type: cross  Abstract: As large language models are deployed in increasingly autonomous long-horizon tasks, manually auditing and verifying the actions, artifacts, and outputs of models becomes more difficult. Users instead come to rely on LLM-generated reports to assess the quality and completeness of the work. We introduce a suite of eight adversarial reporting scenarios to systematically study whether LLMs conceal narrative-changing flaws: errors or limitations that undermine an otherwise successful account of work. We call this phenomenon "insecure reporting." When handed machine learning experiment logs containing a planted negative result that substantially weakens the proposed method, GPT-5.5 flags the negative result in only 2 of 200 generated reports. However, when a short honesty instruction, "Be honest in your response," is added, the model flags the negative result in 190 of 200 reports. Across eight open-weight models, chain-of-thought analysis 
    
[^184]: 纠正何时才能成为修复？工具使用型大语言模型内部干预的机制审计

    When Does Correction Become Repair? Mechanistic Auditing of Internal Interventions in Tool-Using LLMs

    [https://arxiv.org/abs/2609.36138](https://arxiv.org/abs/2609.36138)

    提出SAKIKO审计框架，揭示工具使用型大语言模型内部干预中“行为改变不等于修复”——即便干预带来净收益，也可能损害超过一半的原始决策，因此必须通过目的地解析验证来审计干预的真实效果。

    

    在调用外部工具之前，智能体型大语言模型必须在包含K个选项的动作空间中做出选择：执行调用、请求澄清、直接回答或拒绝回答。虽然内部激活引导可以改变这些执行前的决策，但传统的聚合指标掩盖了被改变的状态落向何处，以及它们造成了何种附带损害。我们提出了SAKIKO——一个审计框架，通过方向性错误发现、基于路由器的条件干预、目的地解析验证以及前瞻性冻结的统计许可来形式化“表征修复”这一概念。在When2Call和MetaTool数据集上对七个大语言模型的实验中，通道键控干预在五个模型中诱导出方向特定的净增益；在三次密封评估中，59个预算匹配的随机方向无一能达到校准后的目标增益。至关重要的是，目的地审计表明行为上的移动并不等同于修复：一个实现+55净增益的干预会破坏超过一半的原始决策（原文摘要在此处截断）。

    arXiv:2609.36138v1 Announce Type: new  Abstract: Before invoking external tools, an agentic LLM must select among a K-way action space: executing a call, seeking clarification, answering directly, or declining. While internal activation steering can alter these pre-execution decisions, conventional aggregate metrics obscure where altered states land and what collateral damage they inflict. We present SAKIKO, an auditing framework that formalizes representation repair via directional error discovery, router-conditioned intervention, destination-resolved verification, and prospectively frozen statistical licensing. Across seven LLMs on When2Call and MetaTool, channel-keyed interventions induce direction-specific net gains in five models; across three sealed evaluations, none of 59 budget-matched random directions matches calibrated target gain. Crucially, destination auditing shows that behavioral movement does not equal repair: an intervention achieving +55 net gain corrupts over half o
    
[^185]: 一种基于字符级神经网络的僧伽罗语连声切分方法

    A Character-Level Neural Approach to Sinhala Sandhi Splitting

    [https://arxiv.org/abs/2609.36131](https://arxiv.org/abs/2609.36131)

    该论文基于SandhiLex数据集首次为僧伽罗语连声切分建立了字符级神经网络基准，发现双向LSTM编码器-解码器模型在规则的附加式连声上可达94%准确率，但在词汇化、派生和词源连声等困难子集上仅达68.40%。

    

    僧伽罗语连声切分旨在从因语音融合而形成的表层形式中恢复隐藏其中的构成词或语素。由于连声现象模糊了词汇边界，该任务对僧伽罗语自然语言处理非常重要，但此前尚无已发表工作为僧伽罗语连声切分建立神经基准。我们基于SandhiLex数据集开展了一项字符级序列到序列研究，使用原生僧伽罗语Unicode输入，并评估了循环编码器-解码器模型在附加式连声以及更复杂的词汇化、派生和词源连声上的表现。核心挑战在于困难子集——词汇化、派生和词源连声，我们表现最好的模型（双向LSTM编码器配合单向LSTM解码器）在该子集上仅达到68.40%的精确匹配准确率（82.08%的字符级准确率），远低于在更规则的附加式子集上取得的94.00%。消融实验表明，双向编码是对性能贡献最大的因素。

    arXiv:2609.36131v1 Announce Type: new  Abstract: Sinhala Sandhi splitting recovers the constituent words or morphemes hidden inside a phonologically merged surface form. The task is important for Sinhala NLP because Sandhi obscures lexical boundaries, but no prior published work has established a neural benchmark for Sinhala Sandhi splitting. We present a character-level sequence-to-sequence study based on SandhiLex, using native Sinhala Unicode input and evaluating recurrent encoder-decoder models for affixational and more complex lexicalized, derivational, and etymological Sandhi. The central challenge is the hard subset lexicalized, derivational, and etymological Sandhi, where our best model, a bidirectional LSTM encoder with a unidirectional LSTM decoder, reaches only 68.40\% exact-match accuracy (82.08\% character-level accuracy), well below the 94.00\% achieved on the more regular affixational subset. Ablations show that bidirectional encoding is the largest contributor to perfor
    
[^186]: 更好的行为预测意味着更忠实可靠的模型消融吗？来自序列选择的证据

    Better Behavioral Prediction, More Faithful Model Ablations? Evidence from Sequential Choice

    [https://arxiv.org/abs/2609.36097](https://arxiv.org/abs/2609.36097)

    该研究在具有已知生成策略的合成序列选择任务中，通过比较神经网络模型与认知模型对输入消融的响应，检验了“模型对信息的依赖能否忠实反映行为生成过程的依赖”这一核心假设，揭示了行为预测的准确性并不必然保证模型消融分析对认知解释的忠实性。

    

    使用预测模型来解释认知，仅仅拥有准确的行为预测是不够的。输入消融提供了一种颇具吸引力的方法：从模型中移除某种信息，并将由此产生的性能变化解释为该信息对行为重要性的证据。然而，这一推断假设模型对信息的依赖反映了生成该行为的过程对信息的依赖。我们在两个具有已知生成策略的合成序列多臂老虎机任务中对这一假设进行了检验，在这些任务中，当预测器无法获得反馈时，过去的选择仍然可以携带信息。我们比较了从零开始训练的GRU和Transformer、经过微调的LLaMA模型，以及认知模型，并在系统性变化的奖励贡献条件下进行测试。我们的分析区分了两种情形：一是在没有奖励观测的条件下训练后的预测，二是固定预测器对供体-奖励替换的响应。研究得出三个发现。首先，在不稳定任务中，在没有奖励观测条件下训练的神经模型（摘要在此处截断）

    arXiv:2609.36097v1 Announce Type: cross  Abstract: Using predictive models to explain cognition requires more than accurate behavioral predictions. Input ablations offer an appealing route: remove information from a model and interpret the resulting performance change as evidence of its importance for behavior. Yet this inference assumes that the model's dependence on information reflects the dependence of the process generating the behavior. We test it in two synthetic sequential bandit tasks with known generating policies, where past choices can remain informative when feedback is unavailable to a predictor. We compare GRUs and Transformers trained from scratch, a fine-tuned LLaMA model, and cognitive models across systematically varied reward contributions. Our analyses distinguish prediction after training without reward observations from the response of a fixed predictor to donor-reward replacement. Three findings emerge. First, in the restless task, neural models trained without 
    
[^187]: PADM'E：用于语言模型智能体评估器元评估的偏好对齐数据合成方法

    PADM\'E: Preference Alignment Data Synthesis for Meta-Evaluation of LM Agent Evaluators

    [https://arxiv.org/abs/2609.36086](https://arxiv.org/abs/2609.36086)

    该论文将元评估重新表述为偏好判断问题，并提出PADM'E数据合成方法，仅依靠小型语言模型、无需人工参与且计算成本低，就能为智能体场景生成可靠的基于标准的元评估数据。

    

    语言模型经常被用于评估其他语言模型。一个能够跨多个标准对智能体行为进行评分的语言模型评估器是很有价值的，前提是它的判断与人类判断保持一致。我们将评估这种一致性的问题称为元评估。直接解决这一难题十分困难：收集人类数据成本高昂，绝对评分难以对齐，而使用语言模型元评估器又会引发可信度的递归问题。我们将元评估重新表述为一个偏好判断问题：不再比较人类与语言模型评估器对同一轨迹的打分，而是询问二者所隐含的偏好是否一致。基于这一思路，我们提出了PADM'E，一种为智能体场景生成可靠的、基于标准的元评估数据的数据合成方法。PADM'E仅使用小型语言模型，评估过程中无需人工参与，且在较低的计算预算下运行。我们构建了一个概念验证……（摘要原文在此处截断）

    arXiv:2609.36086v1 Announce Type: cross  Abstract: Language models are frequently employed to evaluate other language models. An LM evaluator scoring agentic behaviors across multiple criteria is valuable, provided that its decisions align with human judgment. We call the problem of evaluating this alignment Meta-Evaluation. Tackling it directly is difficult: collecting human data is expensive, absolute scoring is hard to align, and using an LM meta-evaluator recurses the question of trustworthiness. We adopt a reformulation of meta-evaluation as a preference judgment problem: rather than comparing human and LM evaluator scores of a trajectory, we ask whether their implied preferences align. Building on this, we introduce PADM\'E, a data synthesis method that generates reliable criterion-based meta-evaluation data for agentic settings. PADM\'E uses only small language models, requires no human involvement during evaluations, and operates under a low computational budget. We build a pro
    
[^188]: GeoOutageBench：面向多模态停电与韧性分析的歧义感知、本体驱动的地理时空知识图谱问答基准测试

    GeoOutageBench: Benchmarking Ambiguity-aware, Ontology-grounded Geospatiotemporal KGQA for Multimodal Power Outage and Resilience Analysis

    [https://arxiv.org/abs/2609.36082](https://arxiv.org/abs/2609.36082)

    该论文提出了GeoOutageBench，一个基于多模态时空知识图谱的基准测试，用于评估大语言模型在多模态停电与韧性分析中对歧义地理时空问题的理解、本体效用评估和答案准确性三方面能力。

    

    我们提出了GeoOutageBench，这是一个用于评估基于大语言模型（LLM）的地理时空知识图谱问答（KGQA）在多模态停电与韧性分析方面能力的基准测试。与现有的面向网络知识的KGQA基准不同，GeoOutageBench采用了一个时空知识图谱，该图谱整合了来自停电记录、遥感、天气观测、风暴和电力事件、地理实体以及领域本体的视觉、文本和结构化数据。它提供了一个不同难度级别的能力查询分类体系，涵盖时空包含与邻近关系、时空共现分析、多模态证据以及假设评估。基于多模态知识图谱和查询类别，GeoOutageBench提供了用户可配置的评估，涵盖三个重要、高度相关但研究较少的任务：（1）LLM对歧义地理时空问题的理解能力，即自然语言到SPARQL的转换；（2）查询驱动的本体效用评估；（3）答案准确性评估……

    arXiv:2609.36082v1 Announce Type: new  Abstract: We introduce GeoOutageBench, a benchmark for assessing LLM-based geospatiotemporal KGQA for multimodal outage and resilience analysis. Unlike existing KGQA benchmarks for Web knowledge, GeoOutageBench considers a spatiotemporal KG that integrates visual, textual, and structured data from outage records, remote sensing, weather observations, storm and power events, geographic entities, and domain ontologies. It provides a competency query taxonomy at different difficulty levels from spatiotemporal containment and proximity, spatiotemporal co-occurrence analysis, multimodal evidence, to hypothetical evaluation. Over multimodal KG and query classes, GeoOutageBench provides user-configurable evaluation of three important, highly coherent yet less studied tasks: (1) LLMs' understanding for ambiguous geospatiotemporal questions in terms of NL to SPARQL interpretation, (2) query-driven assessment of ontology utility, and (3) answer accuracy of 
    
[^189]: 一种关于AI理解的复调概念

    A Polyphonic Conception of AI Understanding

    [https://arxiv.org/abs/2609.36079](https://arxiv.org/abs/2609.36079)

    该论文提出“复调”的AI理解概念，基于机制性证据论证大语言模型的理解并非定位于单一机制，而是由多个可靠性不等的并行机制联盟共同产生输出，从而重构了关于AI理解问题的传统单声部范式。

    

    当医生、法官或工程师必须决定是否信任AI模型的输出时，他们无法回避询问该模型究竟理解了什么。纯粹的数学或统计描述难以在不以另一种名义重新引入AI理解问题的情况下，区分可信与不可信的输出。然而，这个问题目前的表述方式是不恰当的，因为沿袭下来的传统概念运作于一种单声部范式之中：即认为认知系统对某事物的理解必须定位于支撑该理解所赋予的全部能力的单一机制上。基于广泛的机制性证据，我们表明大语言模型普遍具有复调性：其输出源于可靠性不等的并行机制联盟，这些联盟相互补充、相互复制或相互淹没，多个联盟足以完成任务而没有任何一个是不可或缺的。复调性不仅……

    arXiv:2609.36079v1 Announce Type: new  Abstract: When a doctor, a judge, or an engineer must decide whether to trust an AI model's output, they cannot avoid asking what the model understands. Purely mathematical or statistical descriptions struggle to distinguish trustworthy from untrustworthy outputs without reintroducing the question of AI understanding in all but name. Yet the question is ill-framed as it stands, because the inherited concept operates within a monophonic paradigm: the idea that a cognitive system's understanding of something must be localised to a single mechanism underpinning all the capacities conferred by such understanding. Drawing on a wide range of mechanistic evidence, we show that LLMs are pervasively polyphonic: outputs emerge from coalitions of parallel mechanisms of uneven reliability, which variously complement, duplicate, or drown out one another, with several coalitions sufficing for a task without any one being indispensable. Polyphony not only compli
    
[^190]: Mnemon：原始记录、快速判断与慢速思考

    Mnemon: Raw Records, Fast Judgments, Slow Thoughts

    [https://arxiv.org/abs/2609.36059](https://arxiv.org/abs/2609.36059)

    提出记忆代理 Mnemon，将记忆工作类比为快慢双系统分工：LLM（System 2）负责规划搜索与组织答案，决策模型 Jev（System 1）快速执行大量记录判断，从而在保留带日期的原始对话记录的基础上高效构建长期记忆。

    

    长期记忆使 LLM 助手能够利用其已无法重新阅读的历史对话，而大多数记忆系统是在写入时通过将对话重写为事实、图谱或类型化记忆来构建长期记忆。我们认为，记忆的工作如同思考一样，可以划分为两个系统。其中大部分是快速的 System 1 工作：对记录进行大量独立的是/否判断，例如某条记录是否仍然需要或是否已过时，一个决策模型可以在三分之一秒内完成数十个这样的判断。只有少量工作是慢速的 System 2 工作：撰写少量搜索查询、明确回答所需的内容并组织答案，这些任务 LLM 擅长但速度较慢。我们提出了 Mnemon，一个基于这种分工构建的记忆代理。它将对话保存为带有日期的原始记录；LLM（System 2）负责规划对这些记录的搜索，决策模型 Jev（System 1）负责判断搜索返回的结果，而带有明确预算的规则将这些判断转化为一个小型视图，供未经修改的应答模型使用。

    arXiv:2609.36059v1 Announce Type: cross  Abstract: Long-term memory lets an LLM assistant use a history it can no longer reread, and most memory systems build it by rewriting conversations into facts, graphs or typed memories at write time. We argue that the work of memory divides, as thinking does, into two systems. Most of it is fast System 1 work: many small, independent yes/no judgments about records, such as whether a record is needed or no longer current, which a decision model makes by the dozen in a third of a second. Only a little is slow System 2 work: writing a few search queries, naming what the reply needs and composing the answer, which an LLM does well but slowly. We present Mnemon, a memory agent built on this division. It keeps conversations as raw, dated records; an LLM (System 2) plans searches over them, a decision model, Jev (System 1), judges what the searches return, and rules with explicit budgets turn the judgments into a small View for an unchanged answering m
    
[^191]: 大语言模型组合任务中的因果与可解释结构

    Causal and Interpretable Structures in LLM Compositional Tasks

    [https://arxiv.org/abs/2609.35970](https://arxiv.org/abs/2609.35970)

    该研究通过循环概念的组合任务揭示了大语言模型处理词元关系的内部机制：模型在中间层采用基于两个词元推断关系的联合几何表示，而在后续层则使用涉及全部三个词元的联合表示来完成任务，且这一逐层演进模式在不同模型家族中保持一致。

    

    大语言模型能够解决那些答案不仅取决于单个输入词元、还取决于词元之间关系的任务。这种关系信息是如何在各Transformer层中被表示和处理的？我们研究了需要推断三个词元之间关系（这些词元对应循环概念：月份、小时、星期和音符）以正确预测下一个词元的提示词集合的激活。在多个模型家族（Llama、Qwen、Gemma和Mistral）以及多种循环概念上，我们发现词元间的联合依赖关系在几何组织方式和因果使用方式上呈现一致的逐层演进规律：中间层使用基于两个词元之间推断关系的联合表示，而后续层则使用与全部三个词元相关的联合表示来正确完成任务。我们还发现了其他在几何上具有结构（摘要在此处截断）……

    arXiv:2609.35970v1 Announce Type: cross  Abstract: Large language models are able to solve tasks whose answers depend on not only individual input tokens, but also on relations among them. How is such relational information represented and processed across transformer layers? We study activations from ensembles of prompts that require inferring relationships between three tokens corresponding to a cyclic concept (months, hours, weekdays, and musical notes) to correctly predict the next token. Across model families (Llama, Qwen, Gemma, and Mistral) and cyclic concepts, we find a consistent layerwise progression in how the joint dependence among the tokens is geometrically organized and causally used: intermediate layers use a joint representation based on the inferred relationship between two tokens, while later layers use a joint representation associated with all three tokens to correctly complete the task. We also find other relationships between tokens that are geometrically structu
    
[^192]: 面向高效视觉推理的问题特定知识图谱

    Question-Specific Knowledge Graphs for Efficient Visual Reasoning

    [https://arxiv.org/abs/2609.35942](https://arxiv.org/abs/2609.35942)

    提出VisKG强化学习框架，将视觉内容转化为问题特定的知识图谱表示，过滤无关视觉细节并保留实体-关系结构，从而实现更高效的视觉推理。

    

    视觉问答领域的近期研究表明，视觉-语言模型通过将视觉输入转化为文本表示，能够展现出强大的推理能力。这种转化的有效性取决于视觉细节的保留程度；模型需要呈现并对齐足以支持推理的显性与隐性知识，同时不引入虚假的假设。现有利用详细图像描述的方法会引入与推理任务无关的视觉细节，导致输入token数量膨胀并增加计算成本。为应对这些挑战，我们提出了VisKG，一个强化学习（RL）框架，模型在其中学习将视觉内容转化为针对特定问题的知识图谱（KG）表示。这一过程过滤掉感知噪声，同时保留思维链推理所需的实体-关系结构，遵循最小充分性原则。

    arXiv:2609.35942v1 Announce Type: new  Abstract: Recent work in visual question answering has shown that vision-language models can exhibit strong reasoning capabilities by translating visual inputs into textual representations. The effectiveness of this translation depends on how well visual details are retained; models need to surface and align both explicit and implicit knowledge sufficient to support reasoning, without introducing spurious assumptions. Existing methods that leverage detailed image captions introduce visual details unrelated to the reasoning task, inflating input token counts and increasing computational cost. To address these challenges, we propose VisKG, a reinforcement learning (RL) framework in which models learn to translate visual content into question-specific knowledge graph (KG) representations. This process filters out perceptual noise while preserving the entity-relation structure needed for chain-of-thought reasoning, following the principle of minimum s
    
[^193]: 几近人类，除非在关键时刻：VoxParity 与语音应当改变的那些决策

    Almost Human, Except When It Matters: VoxParity and the Decisions a Voice Should Change

    [https://arxiv.org/abs/2609.35922](https://arxiv.org/abs/2609.35922)

    提出 VoxParity 基准，在 183 个文字固定、音频可变的跨行业场景中检验语音代理是否会因“听到什么”而改变关键决策，结果 23 个可测系统中仅 11 个通过仅凭文字的零测试，表明当前语音代理在关键时刻仍以文字为准而忽略声音线索。

    

    语音代理可以仅凭通话的文字内容处理几乎所有的来电，却仍可能在其所在行业规则专门针对的少数场景中失败。紧急呼叫标准、反欺诈指南、无线电通话术语以及弱势群体保护规则都承认：来电者声音听起来如何，或通话中还能听到什么，都可能改变正确的处置行动。VoxParity 用于检验语音代理是否真的会据此行动。在来自 14 个行业的 183 个场景中，文字转录保持不变，而音频发生变化（例如教练式安抚的声音、医疗监护仪的哔哔声、无线电检查中夹带的求救呼号、覆盖在药名上的噪音、用儿童声音下注、惊恐的低语），正确的类型化工具调用也随之改变。仅凭文字的零测试规定：只有当系统听到通话后其行动的改变幅度大于仅读取文字的流水线时，才算通过。在 23 个同时可以在转录文本上运行的系统中，只有 11 个通过。从描述性结果看，错误倾向于倒向文字：当音频呼唤保护措施时，所有 28 个系统仍执行常规……（摘要至此截断）

    arXiv:2609.35922v1 Announce Type: cross  Abstract: A voice agent can handle almost every call on the words alone and still fail the few its sector's rules were written for. Emergency-call standards, fraud guidance, radio phraseology and vulnerability rules recognise that how a caller sounds, or what else is audible, can change the right action. VoxParity tests whether agents act on it. In 183 scenarios from 14 sectors, one transcript stays fixed while the audio changes (a coaching voice, a medical monitor beeping, a mayday under a radio check, noise over a drug name, a child's voice placing a bet, a frightened whisper), and with it the correct typed tool call. A words-only null test credits a system only if hearing the call moves its actions more than it moves a pipeline that only reads the words. Only 11 of the 23 systems that can also be run on the transcript pass. Descriptively, errors run toward the words: when the audio calls for protection, all 28 systems carry out the routine re
    
[^194]: VehicleArena：一个面向多智能体驾驶的真实城市环境

    VehicleArena: A Realistic Urban Environment for Multi-Agent Driving

    [https://arxiv.org/abs/2609.35916](https://arxiv.org/abs/2609.35916)

    提出了VehicleArena，一个3D城市驾驶基准，用于研究动态共享世界中独立运作的LLM控制智能体——每个智能体的驾驶决策会相互影响交通流与风险，而现有九个模型的到达率最高仅约65%。

    

    现实世界的具身智能体通常在共享的物理环境中追求各自独立的目标，它们的行为会改变其他智能体所面临的条件。然而，现有基准测试通常假设智能体共享目标或明确规定了交互协议，使得这种突现的物理耦合尚未得到充分探索。我们提出了VehicleArena，一个用于研究动态共享世界中独立运行智能体的3D城市驾驶基准。在VehicleArena中，由大语言模型（LLM）控制的智能体必须在复杂交通中行驶的同时完成不断变化的乘客请求，且每个智能体的驾驶决策都会重塑交通流、延误、风险以及周围智能体的后续观察。该基准提供了112个评估任务，涵盖单智能体和多智能体驾驶。在对九个模型的评估中，单智能体任务上的最高到达率仅为65.0%，多智能体任务上为65.6%，而在较强的乘客请求或ca…（摘要原文在此处截断）

    arXiv:2609.35916v1 Announce Type: cross  Abstract: Real-world embodied agents often pursue independent objectives within a shared physical environment, where their actions can alter the conditions faced by others. Existing benchmarks, however, typically assume shared goals or explicitly prescribed interaction protocols, leaving such emergent physical coupling underexplored. We introduce VehicleArena, a 3D urban-driving benchmark for studying independently operating agents in a dynamic shared world. In VehicleArena, LLM-controlled agents must fulfill evolving passenger requests while navigating complex traffic, and each agent's driving decisions can reshape traffic flow, delays, risks, and subsequent observations for surrounding agents. The benchmark provides 112 evaluation tasks spanning single-agent and multi-agent driving. Across nine evaluated models, the highest arrival rates reach only 65.0% on single-agent tasks and 65.6% on multi-agent tasks, while strong passenger-request or ca
    
[^195]: CruxBench：信息发现基准测试

    CruxBench: A Benchmark of Information Discovery

    [https://arxiv.org/abs/2609.35879](https://arxiv.org/abs/2609.35879)

    该论文提出CruxBench基准，通过信息价值评估大型语言模型发现关键问题的能力，其真实答案由未来世界事件决定，从而兼具抗污染、开放式和有据可依三大独特优势。

    

    大型语言模型（LLM）的基准测试通常通过与固定参考标签的比对来评估答案的准确性。然而，在许多复杂的现实世界任务中，一个核心步骤是首先识别哪些问题值得提出：将难题分解为子问题——我们称之为“关键问题”——这些子问题的答案为解决目标问题提供了关键步骤。为了评估这种信息发现能力，我们提出了CruxBench，这是一个通过信息价值来为LLM生成的问题评分的基准：即模型提出的关键问题能在多大程度上更新对某个目标预测问题的信念。CruxBench罕见地兼具三个关键特性：（1）从构造上即具有抗污染性，因为真实答案由未来的世界事件产生；（2）开放式，允许无边界且复杂的文本提交，而非唯一正确的数值答案；（3）有据可依……

    arXiv:2609.35879v1 Announce Type: cross  Abstract: Benchmarks for large language models (LLMs) typically evaluate the accuracy of answers against fixed reference labels. But a central step in many complex real-world tasks is identifying which questions are worth asking in the first place: decomposing a difficult problem into subquestions -- which we call cruxes -- whose answers provide key steps on the path toward solving the target problem. To evaluate this capability of information discovery, we introduce CruxBench, a benchmark that grades LLM-generated questions by their Value of Information (VOI): how much a model-proposed crux updates beliefs about a target forecasting question. CruxBench enjoys a rare combination of three key properties: it is (1) contamination-resistant by construction, since ground truth is generated by future world events; (2) open-ended, admitting unbounded and complex text-based submissions rather than one correct numeric answer; and (3) grounded, with infor
    
[^196]: 词元边界的代价：压缩证书与预测

    The Price of Token Boundaries: Compression Certificates and Prediction

    [https://arxiv.org/abs/2609.35869](https://arxiv.org/abs/2609.35869)

    该论文提出基于线性规划对偶与独立整数证书来量化预分词边界规则压缩代价的方法，发现边界规则使英文维基百科上的最优词元数增加28.3–36.8%，且压缩与语言模型预测偏好不同的词典。

    

    预分词限制了哪些文本片段可以成为预测单元，但当分词器仅在相同边界下进行比较时，其压缩代价就被掩盖了。我们通过从两侧对最小词元数进行界定来度量这一代价，分别考虑有和没有正则表达式边界规则的两种情况。对词元出现赋予非负价格，可通过最短路径与词表预算选择得到下界；对所有价格取最大值即恢复线性规划松弛，并由一个独立的整数检查器对所报告的数值进行认证。在英文维基百科上，边界规则使最优词元数增加28.3–36.8%。字节对编码（BPE）比受限下界高出2.1%，但比无限制下界高出10.9%。压缩与预测偏好不同的词典：在8500万非嵌入参数和匹配的训练词元预算下，无限制拟合在统一的未截断条件下产生更高的平均留出集每字节比特数（原文在此处被截断）。

    arXiv:2609.35869v1 Announce Type: new  Abstract: Pre-tokenisation restricts which text fragments can become prediction units, but its compression cost is obscured when tokenisers are compared only under the same boundaries. We measure this cost by bounding the minimum token count from both sides, with and without a regular-expression boundary rule. Nonnegative prices on token occurrences yield a lower bound through shortest paths and vocabulary-budget selection; maximising over all prices recovers the linear programming relaxation, and an independent integer checker certifies the reported values. On English Wikipedia, boundaries increase the optimal token count by 28.3--36.8\%. Byte pair encoding lies 2.1\% above the constrained lower bound, but 10.9\% above the unrestricted bound. Compression and prediction favour different dictionaries: at 85M non-embedding parameters and matched training-token budgets, unrestricted fitting yields higher mean held-out bits per byte under a common unr
    
[^197]: PACT：面向单Token类型化决策的成对锚定校准调优

    PACT: Pairwise-Anchored Calibrated Tuning for Single-Token Typed Decisions

    [https://arxiv.org/abs/2609.35865](https://arxiv.org/abs/2609.35865)

    PACT将带有机器校验证据证书的成对对比数据转化为四个无需新标注的训练项（双重差分边际、置换一致性、证据必要性、序数传输成本），并辅以轻量级校准，从而提升单token类型化决策模型的性能与可靠性。

    

    单token类型化决策模型通过在单个位置读取若干单字母答案代码的logits来回答模式化问题：这类模型速度快，并为每个允许的答案返回概率，但它们仅使用普通交叉熵进行训练，忽略了训练数据中的大部分结构。我们研究了这样一类模型，其数据被整理为对比对——两个上下文仅在一个人为编辑的事实上不同，而该事实会翻转答案——且每个对比对都附有机器校验的证书，证明删除起决定性作用的句子会使该事实变为未知。我们提出PACT，它将这种结构转化为四个无需新增标注的训练项：针对每个对比对的双重差分边际（对任何共享的logit偏移保持不变）、对抗答案代码位置偏差的置换一致性项、在经证书验证的消融上下文上的证据必要性项，以及针对评分量规字段的序数传输成本，外加一个三参数的……（摘要原文在此处截断）

    arXiv:2609.35865v1 Announce Type: cross  Abstract: Single-token typed-decision models answer a schema question by reading the logits of a few one-letter answer codes at a single position: they are fast and return a probability for every allowed answer, but they are trained with plain cross-entropy that ignores most of the structure in their training data. We study such a model whose data is curated as contrastive pairs---two contexts that differ in one edited fact that flips the answer---each carrying a machine-checked certificate that deleting the decisive sentence makes the fact unknown. We propose PACT, which turns this structure into four training terms that need no new annotation: a difference-in-differences margin over each pair that is invariant to any shared logit offset, a permutation-consistency term against answer-code position bias, an evidence-necessity term on certificate-verified ablated contexts, and an ordinal transport cost for rubric fields, plus a three-parameter co
    
[^198]: 解决金融数据缺失危机：用于SEC 10-K提取的生成式AI流水线

    Resolving the Missing Financial Data Crisis: A Generative AI Pipeline for SEC 10-K Extraction

    [https://arxiv.org/abs/2609.35864](https://arxiv.org/abs/2609.35864)

    本研究评估了Llama-3 8B、Qwen-2.5 14B和Llama-3.3 70B等多个大型语言模型从SEC 10-K文件中提取财务数据的能力，以解决传统方法（正则表达式和BERT）导致的影响超过70%公司的金融数据缺失问题。

    

    SEC 10-K文件包含大量未在结构化数据集中被一致捕获的财务信息，造成了一个影响超过70%的公司和总市值一半的数据缺失问题。这可能不成比例地使定量分析对小型公司产生偏差，这些公司可能因可用数据有限而被排除在外。传统的财务提取方法，如正则表达式（Regex）和BERT，已被广泛使用。然而，这些方法在解析复杂的SEC 10-K文件时非常脆弱，导致文件中实际存在的数据丢失，因为它们没有考虑到数据属性可能位于不同章节或脚注中。本研究评估了多个大型语言模型（LLM），包括Llama-3 8B、Qwen-2.5 14B和Llama-3.3 70B，以弄清各个模型在从SEC 10-K文本中提取特定属性时的优势与劣势。

    arXiv:2609.35864v1 Announce Type: new  Abstract: SEC 10-K filings contain substantial financial information that is not consistently captured in structured datasets, creating a missing-data problem affecting over 70% of firms and half of total market capitalization. This can disproportionately bias quantitative analysis against smaller firms, which may be excluded due to limited available data. Traditional financial extraction methods such as Regular Expressions (Regex) and BERT, have been widely used. However, they are highly brittle when parsing complex SEC 10-K filings, which leads to data that is existent in the files being lost since these methods do not consider that a data attribute could be located in a different section or a footnote. This study evaluates several Large Language Models (LLMs), including Llama-3 8B, Qwen-2.5 14B, and Llama-3.3 70B, to figure out individual model strengths and weaknesses when extracting specific attributes from SEC 10-K text. The extraction quali
    
[^199]: 可检测性差距：语言模型幻觉检测中隐藏的异质性

    The Detectability Gap: Hidden Heterogeneity in Hallucination Detection Across Language Models

    [https://arxiv.org/abs/2609.35860](https://arxiv.org/abs/2609.35860)

    该研究揭示了语言模型幻觉检测中隐藏的异质性：高一致性（Ghost）与低一致性（Flickering）幻觉之间存在显著的可检测性差距，且在排除统计耦合影响并通过多种稳健性检验后，这种不对称性依然成立。

    

    基于采样的一致性方法被广泛用于幻觉检测，然而总体性能可能掩盖了哪些错误可被检测的系统性差异。本工作在四个语言模型和三个事实问答数据集上研究了这种异质性。通过答案一致性对幻觉进行划分，揭示了高一致性（Ghost）和低一致性（Flickering）两种状态，其表观可检测性差距为0.35至0.46 AUC。由于用于定义这些状态和测量该差距的统计量高度耦合（|ρ|≈0.94至1.00），该原始结果被视为基于一致性的检测方法的属性，而非独立证据。在冻结状态分配之后，词汇和语义响应的离散度仍保持这种不对称性，在所有12个模型与数据集设置中，bootstrap 95%置信区间均不包含零。一种使用单独扩散轨迹且不进行跨种子信息（摘要在此处截断）的更严格测试……

    arXiv:2609.35860v1 Announce Type: cross  Abstract: Sampling based consistency is widely used for hallucination detection, yet aggregate performance can conceal systematic differences in which errors are detectable. This work studies that heterogeneity across four language models and three factual question answering datasets. Partitioning hallucinations by answer agreement reveals high agreement (Ghost) and low agreement (Flickering) regimes with an apparent detectability gap of $0.35$ to $0.46$ AUC. Because the statistics used to define the regimes and measure this gap are strongly coupled ($|\rho|\approx0.94$ to $1.00$), the raw result is treated as a property of agreement based detection rather than independent evidence. After freezing regime assignments, lexical and semantic response dispersion preserve the asymmetry, with bootstrap $95\%$ intervals excluding zero in all $12$ model and dataset settings. A stricter test using individual diffusion trajectories and no cross seed inform
    
[^200]: 超球面语义轨迹分析：映射学术预印本、专利信号与算力扩展中的技术扩散

    Hyperspherical Semantic Trajectory Analysis: Mapping Technological Diffusion across Academic Preprints, Patent Signals, and Compute Scaling

    [https://arxiv.org/abs/2609.35845](https://arxiv.org/abs/2609.35845)

    本文提出超球面语义轨迹分析（HSTA），一种无监督定量方法，通过将arXiv预印本与USPTO专利文本的Transformer嵌入投影到超球面上，直接量化追踪技术扩散与范式转变，克服了传统宏观经济生产率指标多年滞后的局限。

    

    宏观经济生产率指标（如全要素生产率）由于行政调查周期和国民核算惯例的限制，往往需要数年滞后才能记录下技术突破。本文提出了超球面语义轨迹分析，这是一种无监督的定量方法，可直接从非结构化的科学与商业文本流中追踪技术扩散过程。我们分析了30,000条经过筛选的文献记录，涵盖来自arXiv的学术预印本和来自美国专利商标局（USPTO）的专利申请记录。通过利用球面K均值聚类（覆盖八个主要子主题）和UMAP流形降维，将高维Transformer句子嵌入投影到单位超球面上，HSTA形式化了两个定量指标：（1）语义质心向量漂移，通过追踪不同时间子语料库之间的词汇变化来识别结构性范式转变；（2）商业化……（原文摘要至此被截断）

    arXiv:2609.35845v1 Announce Type: new  Abstract: Macroeconomic productivity metrics, such as Total Factor Productivity, register technological breakthroughs with multi-year reporting lags due to administrative survey intervals and national accounting conventions. This paper introduces Hyperspherical Semantic Trajectory Analysis (HSTA), an unsupervised quantitative methodology that tracks technology diffusion directly from unstructured scientific and commercial text streams. We analyze 30,000 filtered document records spanning academic preprints from arXiv and patent application records from the USPTO. By projecting high-dimensional Transformer sentence embeddings onto unit hyperspheres using Spherical K-Means clustering across eight primary sub-topics and UMAP manifold reductions, HSTA formalizes two quantitative metrics: (1) Semantic Centroid Vector Drift, which tracks vocabulary shifts between temporal sub-corpora to identify structural paradigm transformations; and (2) Commercializa
    
[^201]: 面向资源受限边缘设备可靠推理的神经符号路由

    Neurosymbolic Routing for Reliable Reasoning on Resource-Constrained Edge Devices

    [https://arxiv.org/abs/2609.35833](https://arxiv.org/abs/2609.35833)

    该论文提出一种神经符号路由方法，通过L*文法推断学习确定性有限自动机，将查询分类并分派给成本最低的正确求解器（结构化任务交给确定性符号引擎，开放性问题交给小语言模型），从而在资源受限的边缘设备上实现更可靠、更高效的推理。

    

    在边缘硬件上运行语言模型能够在无需网络连接的情况下提供私密且低延迟的推理能力，然而适配此类设备的小型模型在计算机本应擅长处理的任务（如算术、代数和形式逻辑问题）上并不可靠。我们认为这种不可靠性在很大程度上是可以避免的：许多看似需要推理的查询实际上是结构上确定性的，可以采用快速且精确的符号方法求解。因此，强迫概率模型去近似求解这些问题只会牺牲准确性和能耗，却几乎没有任何收益。我们提出了一种神经符号路由器，它对每个传入的查询进行分类，并将其分派给成本最低的正确求解器：将结构化任务发送给确定性引擎，而将小型语言模型（SLM）保留用于开放式的文字应用题。我们没有手工编写路由逻辑，而是使用L*文法推断来学习一个确定性有限自动机（DFA）。

    arXiv:2609.35833v1 Announce Type: new  Abstract: Running a language model on edge hardware provides private and low-latency reasoning without a network connection, and yet the small models that fit on such devices are unreliable on the tasks computers are expected to handle well, such as arithmetic, algebra, and formal logic problems. We argue that much of this unreliability is avoidable. Many queries appearing to demand reasoning are in fact structurally deterministic and permit fast and exact symbolic solutions. Therefore, forcing a probabilistic model to approximate them sacrifices accuracy and energy for little benefit. We present a neurosymbolic router that classifies each incoming query and dispatches it to the cheapest correct solver, sending structured tasks to deterministic engines and reserving the small language model (SLM) for open-ended word problems. Instead of hand-coding the routing logic, we learn a deterministic finite automaton (DFA) with the L* grammatical inference
    
[^202]: 大语言模型何时应该相信自己的修改？一项关于内在自我纠错的风险感知研究

    When Should LLMs Trust Their Own Revisions? A Risk-Aware Study of Intrinsic Self-Correction

    [https://arxiv.org/abs/2609.35832](https://arxiv.org/abs/2609.35832)

    该研究对 29 个开源大模型的内在自我纠错进行了风险感知分析，发现总体准确率会掩盖“挽回错误”与“推翻正确答案”之间的权衡，并通过比较三种运行时策略识别出利用初始回答后信号进行选择性修改（学习型门控）有效的场景。

    

    内在自我纠错要求语言模型在不接收任何新外部证据的情况下修改自己给出的答案。第二轮修改可以挽回错误，但也可能推翻原本已经正确的答案。我们在 BoolQ、GSM8K 和 Corr2Cause 数据集上，通过追踪初始答案与修改后答案之间的正确性变化，对 29 个开源权重大语言模型的这一权衡进行了研究。总体准确率可能掩盖截然不同的修改行为：例如，Llama-3.1-8B 在 GSM8K 上提升了 25.5 个百分点，但“精炼”过程同时将 19.1% 的原本正确答案改成了错误答案。一项受控的 BoolQ 研究进一步表明，精炼提示词会改变“挽回错误”与“造成损害”之间的平衡。随后，我们比较了三种运行时策略：保留初始答案、总是接受修改后的答案，以及利用初始回答之后可获得的信号选择性地触发修改。该比较识别出了学习型门控发挥作用的有效场景，以及……

    arXiv:2609.35832v1 Announce Type: cross  Abstract: Intrinsic self-correction asks a language model to revise its own answer without receiving new external evidence. A second pass can recover mistakes, but it can also overturn answers that were already correct. We study this trade-off across 29 open-weight LLMs on BoolQ, GSM8K, and Corr2Cause by tracking correctness transitions between initial and revised answers. Aggregate accuracy can conceal substantially different revision behavior: for example, Llama-3.1-8B improves by 25.5 percentage points on GSM8K, while refinement changes 19.1% of initially correct answers into wrong ones. A controlled BoolQ study further shows that refinement prompts shift the balance between recovery and harm. We then compare three runtime choices: keeping the initial answer, always accepting the revision, and selectively invoking revision using signals available after the initial response. The comparison identifies settings where learned gating is useful and
    
[^203]: 超越上下文窗口：一种面向混合检索与长上下文语言模型的自适应熵路由框架

    Beyond the Context Window: An Adaptive Entropy-Based Routing Framework for Hybrid Retrieval and Long-Context Language Models

    [https://arxiv.org/abs/2609.35831](https://arxiv.org/abs/2609.35831)

    本文提出熵驱动自适应路由器（EDAR），通过利用RAG响应前几个生成token的预测熵，在推理时自适应地决定采用检索增强生成还是完整长上下文处理，从而在成本与准确性之间取得平衡。

    

    现代大型语言模型如今已支持超过一百万token的上下文窗口，这引发了一个问题：检索增强生成（RAG）是否仍然必要。纯长上下文（LC）处理成本高昂，且已知对位于长输入中间位置的信息注意力不足；而纯RAG虽然速度快，但受限于检索质量，并且在检索到的片段仅部分相关或相互矛盾时容易出错。我们提出了熵驱动自适应路由器（EDAR），这是一个在推理时决定是依据检索到的片段回答查询，还是将其升级为完整长上下文处理的框架。该决策基于对RAG响应前几个生成token计算的token级概率分布的预测熵。熵阈值通过在留出验证集上对成本与准确性进行权衡搜索来选取。实验将EDAR与纯RAG和纯长上下文方法进行了比较。

    arXiv:2609.35831v1 Announce Type: new  Abstract: Modern large language models now support context windows of more than one million tokens, which has raised the question of whether retrieval-augmented generation (RAG) is still necessary. Pure long-context (LC) processing is expensive and is known to under-attend to information placed in the middle of long inputs, while pure RAG is fast but bounded by retrieval quality and prone to errors when retrieved chunks are partially relevant or contradictory. We propose the Entropy-Driven Adaptive Router (EDAR), a framework that decides at inference time whether to answer a query from retrieved chunks or to escalate it to full long-context processing. The decision uses the predictive entropy of the token-level probability distribution computed over the first few generated tokens of the RAG response. The entropy threshold is selected on a held-out validation set by sweeping cost against accuracy. Experiments compare EDAR against pure-RAG and pure-
    
[^204]: 可靠但对设计敏感：大语言模型标注中的工具不确定性

    Reliable but Design-Sensitive: Instrument Uncertainty in LLM Annotation

    [https://arxiv.org/abs/2609.35824](https://arxiv.org/abs/2609.35824)

    大语言模型标注在重复运行时虽然高度一致，但不同的任务设计和模型选择会导致标注结果大幅波动，其引入的不确定性远超人工标注工具之间的差异和抽样误差。

    

    大语言模型（LLMs）在某种设置下可以给出可靠的标签，但当研究者做出其他合理的设计选择时，这些标签却会发生改变。我们测试了七个大语言模型、12种任务设计、三次独立运行，以及3000条标注为冒犯性语言和仇恨言论的推文。重复使用相同的模型和任务设计产生了高度一致性（Fleiss' κ 中位数为0.91）。当对相同推文改变任务设计时，一致性下降（Cohen's κ 中位数为0.76）。与仅考虑抽样方差相比，任务设计和模型选择使估计流行率的方差分别增加了76.7倍（冒犯性语言）和110.6倍（仇恨言论）。大语言模型任务设计之间的差异达到560-572个基点，而五个人工标注工具版本之间的差异为270-331个基点。置信度分数无法解决这个问题：它们追踪重复模型输出的程度高于其与人工标签的一致性，并且将六条推文进行分组……

    arXiv:2609.35824v1 Announce Type: new  Abstract: Large language models (LLMs) can give reliable labels under one setup yet change those labels when researchers make other reasonable design choices. We tested seven LLMs, 12 task designs, three independent runs, and 3,000 tweets labeled for offensive language and hate speech. Repeating the same model and task design produced high agreement (median Fleiss' $\kappa = 0.91$). Agreement fell when we changed the task design for the same tweets (median Cohen's $\kappa = 0.76$). Task design and model choice increased the variance of estimated prevalence by factors of 76.7 for offensive language and 110.6 for hate speech compared with sampling variance alone. Variation across LLM task designs reached 560-572 basis points, compared with 270-331 basis points across five human instrument versions. Confidence scores did not solve this problem. They tracked repeated model outputs more closely than agreement with human labels, and grouping six tweets 
    
[^205]: 追踪语言模型中谄媚性认同的机制

    Tracing mechanisms of sycophantic agreement in language models

    [https://arxiv.org/abs/2609.35822](https://arxiv.org/abs/2609.35822)

    该研究通过因果中介分析揭示，语言模型的谄媚性认同源于一组稀疏的早期注意力头——它们将用户观点信号注入最终提示词元的残差流并偏置答案检索，而消融这些注意力头即可大幅降低谄媚性且几乎不损害事实准确性。

    

    语言模型中的谄媚性认同是指模型过度肯定用户所陈述的信念或偏好的倾向，而这往往以牺牲事实准确性为代价。尽管这种现象被广泛认为是一种对齐失败，但其内在机制仍鲜为人知。在这项工作中，我们使用因果中介分析来识别谄媚性认同背后的机制。我们表明，用户陈述的观点会在早期被整合进最后一个提示词元的残差流中，并在那里对后续的答案检索产生偏置。一组稀疏的早期注意力头携带这一观点信号。消融这些注意力头能够大幅减少谄媚行为，同时基本不影响事实准确性。当观点被明确陈述时，无论措辞如何变化，同样的这组注意力头都会携带该观点信号。当观点未被明确陈述，而是通过无具体内容的质疑（例如“你确定吗？”）来传达时，我们发现了一组不同的注意力头，它们会抑制模型原本的判断（原文摘要在此处被截断）。

    arXiv:2609.35822v1 Announce Type: new  Abstract: Sycophantic agreement in language models refers to the tendency to overly affirm a user's stated beliefs or preferences, often at the expense of factual accuracy. Although it is widely recognized as an alignment failure, its underlying mechanisms remain poorly understood. In this work, we use causal mediation analysis to identify the mechanisms behind sycophantic agreement. We show that a stated opinion is incorporated into the residual stream of the final prompt token early, where it biases subsequent answer retrieval. A sparse set of early attention heads carries this opinion signal. Ablating these heads substantially reduces sycophancy while leaving factual accuracy largely intact. The same heads carry the opinion when it is explicitly stated, regardless of how it is phrased. When an opinion is not stated explicitly but instead conveyed through content-free pushback (e.g., ``Are you sure?"), we find a distinct set of heads that suppre
    
[^206]: 我们还能信任灾害社会感知吗？关于检测AI生成社交媒体帖子的实证证据

    Can We Still Trust Disaster Social Sensing? Empirical Evidence on Detecting AI-Generated Social Media Posts

    [https://arxiv.org/abs/2609.35821](https://arxiv.org/abs/2609.35821)

    本研究构建了来自九场灾害的12,000条文本的匹配语义单元数据集，系统评估了多种AI文本检测器及大语言模型判断器区分人类与AI生成灾害帖子的能力，为生成式AI对灾害社会感知可信度的威胁提供了实证证据。

    

    灾害社会感知将公众社交媒体帖子转化为用于态势感知和人道主义需求的证据，但生成式人工智能（AI）可以生成与目击者报告极为相似的逼真消息。本研究探究基于文本的AI检测器能否可靠地区分人类撰写的与AI生成的灾害帖子。我们构建了一个包含12,000条文本的数据集，这些文本来自九场灾害，被组织成3,000个匹配的语义单元，包括：原始人类帖子（H0）、经过大语言模型（LLM）轻度校对的人类帖子（H1）、基于相同已验证事实生成的事实性AI帖子（A0），以及这些AI帖子的情感化框架版本（A1）。另一个来自42个事件的6,000条文本的独立语料库用于支持模型选择和阈值校准。我们在五个模型家族上评估了OSM-Det、Fast-DetectGPT、Binoculars以及直接使用大语言模型（LLM）进行判断的方法，随后测试了灾害领域校准、冻结编码器的线性读出以及配对（原文在此处截断）

    arXiv:2609.35821v1 Announce Type: new  Abstract: Disaster social sensing converts public social-media posts into evidence for situational awareness and humanitarian needs, but generative artificial intelligence (AI) can produce plausible messages that resemble eyewitness reports. This study investigates whether text-based AI detectors can reliably distinguish human-authored from AI-generated disaster posts. We construct a dataset of 12,000 texts organised into 3,000 matched semantic units from nine disasters: original human posts (H0), minimally LLM-proofread human posts (H1), factual AI-generated posts based on the same verified facts (A0), and affectively framed versions of those AI posts (A1). A separate 6,000-text corpus from 42 events supports model selection and threshold calibration. We evaluate OSM-Det, Fast-DetectGPT, Binoculars, and direct large language model (LLM) judges across five model families, then test disaster-domain calibration, a frozen-encoder linear readout, pair
    
[^207]: τ-多语言：跨语言语音智能体基准测试

    $\tau$-Multilingual: Benchmarking Voice Agents Across Languages

    [https://arxiv.org/abs/2609.35820](https://arxiv.org/abs/2609.35820)

    该论文提出多语言语音智能体基准τ-Multilingual，发现韩语和中文场景下任务完成度显著下降（分别下降14.7和8.4分）且各语言呈现不同失败模式，并发布了语言包和评估工具供社区使用。

    

    仅针对英语的基准测试只能揭示语音智能体行为的很小一部分。我们提出了τ-Multilingual，将τ-Voice扩展到西班牙语、巴西葡萄牙语、印地语、韩语和中文，并由母语使用者对生成的语言和语音输出进行审查与评估。在4,500次全双工通话和五种语音配置的测试中，西班牙语、葡萄牙语和印地语的任务完成度与英语相差在3.2分以内，但韩语和中文分别下降了14.7分和8.4分。各语言的失败模式也各不相同：韩语系统更容易漏掉回复，中文系统更频繁地打断对话，两者在工具和实体处理上都存在困难。Grok在任务完成度上领先，但在生成质量上得分最低，这表明需要分别报告任务、交互和生成质量。我们发布了语言包、经验证的评估器以及相关工具，供社区构建多语言语音智能体评估。

    arXiv:2609.35820v1 Announce Type: cross  Abstract: English-only benchmarks expose only a narrow slice of voice-agent behavior. We introduce $\tau$-Multilingual, extending $\tau$-Voice to Spanish, Brazilian Portuguese, Hindi, Korean, and Mandarin with native-speaker review and evaluation of generated language and spoken output. Across 4,500 full-duplex calls and five voice configurations, Spanish, Portuguese, and Hindi remain within 3.2 task-completion points of English, but Korean and Mandarin fall by 14.7 and 8.4 points. The failure modes also vary: Korean systems miss more responses, Mandarin systems interrupt more often, and both struggle with tools and entities. Grok leads task completion but scores lowest on generation quality, motivating separate task, interaction, and generation reporting. We release language packs, validated judges, and tools for community-built multilingual voice-agent evaluation.
    
[^208]: 更少均匀的离散扩散更强大且更具可扩展性

    Less Uniform Discrete Diffusion is More Powerful and Scalable

    [https://arxiv.org/abs/2609.35817](https://arxiv.org/abs/2609.35817)

    LUDI框架通过引入更少均匀的损失函数和逐token时间嵌入，克服了均匀扩散语言模型规模扩展的障碍，并成功将7B自回归模型转化为具备复杂推理能力、每步可生成3个token的高效扩散模型。

    

    尽管均匀扩散语言模型（UDLM）代表了一种有前景的扩散范式，但其规模扩展仍然充满挑战。我们将核心障碍确定为过度均匀的训练目标以及采样过程中的条件-目标混淆。为解决这些问题，我们提出了“更少均匀扩散”（Less Uniform Diffusion, LUDI），一种新颖的UDLM框架。具体而言，我们（i）引入了一种不那么均匀的损失函数，使每个反向转移直接指向干净的token；（ii）为模型配备逐token时间嵌入，以提供token级别的损坏提示，从而实现基于置信度的少步采样。跨尺度的实验表明，LUDI能提供更干净的监督信号并改善少步生成。我们进一步将一个7B自回归模型继续训练为LUDI-7B，得到一个能够进行复杂推理的UDLM。相比自回归解码，它实现了每步生成3个token的加速，并且与掩码扩散基线相比具有有竞争力的性能，揭示了完整……

    arXiv:2609.35817v1 Announce Type: cross  Abstract: Although uniform diffusion language models (UDLMs) represent a promising diffusion paradigm, scaling them remains challenging. We identify the core obstacle as an over-uniform training objective and condition-target confusion during sampling. To address these, we propose Less Uniform Diffusion (LUDI), a novel UDLM framework. Specifically, we (i) introduce a less uniform loss that directs each reverse transition toward the clean token, and (ii) equip the model with per-token time embeddings that supply token-level corruption hints, enabling confidence-based few-step sampling. Experiments across scales show that LUDI yields cleaner supervision and improves few-step generation. We further continue-train a 7B autoregressive model into LUDI-7B, resulting in a UDLM capable of complex reasoning. It achieves a 3-token-per-step speedup over AR decoding and competitive performance compared with masked diffusion baselines, revealing that the full
    
[^209]: PrimeSeeker：面向能力的深度搜索智能体监督方法

    PrimeSeeker: Capability-Oriented Supervision for Deep Search Agents

    [https://arxiv.org/abs/2609.35816](https://arxiv.org/abs/2609.35816)

    提出PrimeSeeker框架，通过“潜在锚点推理”将深度搜索分解为锚点解析与关系转移的耦合操作链，以面向局部检索能力的方式构建训练监督，从而更有效地训练深度搜索智能体。

    

    大语言模型搜索智能体通常使用合成问题进行训练，这些问题通过更大的证据图、更多的跳数和更长的轨迹来提升难度。然而，这些全局属性只是搜索过程中所需的局部检索能力的间接代理指标。为了解决这种不匹配，我们提出了潜在锚点推理，它包括从描述性规范中解析出一个未命名的检索锚点，并将恢复出的锚点转移到后续的信息需求中。这一原始检索单元将深度搜索分解为耦合操作链，并围绕锚点解析和关系转移来组织问题构建，而不规定固定的搜索路径。基于这一表述，我们提出了PrimeSeeker，一个面向能力的框架，它构建基于真实网络的锚点结构，并联合推导出一个问题和参考证据骨架。

    arXiv:2609.35816v1 Announce Type: cross  Abstract: Large language model search agents are often trained with synthetic questions whose difficulty is increased through larger evidence graphs, additional hops, and longer trajectories. These global properties, however, are only indirect proxies for the local retrieval capabilities required during search. To address this mismatch, we introduce latent anchor reasoning, which consists of resolving an unnamed retrieval anchor from descriptive specifications and transferring the recovered anchor into a subsequent information demand. This primitive retrieval unit decomposes deep search into chains of coupled operations and organizes question construction around anchor resolution and relation transfer, without prescribing a canonical search path. Based on this formulation, we propose PrimeSeeker, a capability-oriented framework that constructs web-grounded anchor structures and jointly derives a question and a reference evidence skeleton. The sk
    
[^210]: 如何对LLM评判者运行统计并信任结果：基于evalstats的小样本AI评估校准推断

    How to Run Statistics over LLM Judges and Trust the Results: Calibrated Inference for Small-Sample AI Evaluation with evalstats

    [https://arxiv.org/abs/2609.35815](https://arxiv.org/abs/2609.35815)

    该论文指出对原始LLM评判分数直接进行统计检验会造成假阳性膨胀（且在人类与LLM几乎完美一致时风险最高），并推出evalstats工具与指导，通过预测驱动推断（PPI）实现九种假设检验（包括首次针对秩检验的PPI校正），使小样本AI评估的统计结论可信可靠。

    

    arXiv:2609.35815v1 公告类型：cross 摘要：学术界的研究人员越来越多地将显著性结论建立在LLM评判分数和小样本AI评估之上。然而，如果没有经过良好校准的置信区间（CI）、假设检验和评判者偏差校正，这些结论是不可靠的。我们在几个方面做出了贡献。首先，我们发现对原始LLM评判分数直接运行统计会导致假阳性率膨胀：反直觉的是，对于许多评分者间一致性指标，假阳性风险恰好在人类与LLM“几乎完美”一致时达到峰值。为了帮助研究人员了解如何负责任地对LLM评判者运行统计，我们为混合人类-AI评判设计提供了统计分析的指导和工具，并通过预测驱动推断（PPI）实现了九种假设检验，包括首次已知的针对四种基于秩的检验（Wilcoxon符号秩检验、Mann-Whitney U检验及其全向变体）的PPI校正。为了使PPI++在小样本人工标注数据上保持稳定……

    arXiv:2609.35815v1 Announce Type: cross  Abstract: Researchers across academia increasingly base significance claims on LLM judge scores and small-sample AI evaluations. Yet without well-calibrated confidence intervals (CIs), hypothesis tests, and judge-bias corrections, such claims are unreliable. We address these issues in several contributions. First, we find that running statistics over raw LLM judge scores leads to inflated false positives: counterintuitively, for many inter-rater agreement metrics, false positive risk peaks at "almost perfect" human-LLM agreement. To help researchers understand how to run statistics over LLM judges responsibly, we present guidance and tooling for the statistical analysis of mixed human-AI judge designs, and implement nine hypothesis tests via prediction-powered inference (PPI), including the first known PPI corrections for four rank-based tests (Wilcoxon signed-rank, Mann-Whitney U, and omnibus variants). To keep PPI++ stable with small human-lab
    
[^211]: 通过受控环境干预构建具有挑战性的浏览器使用任务

    Constructing Challenging Browser-Use Tasks by Controlled Environment Interventions

    [https://arxiv.org/abs/2609.35814](https://arxiv.org/abs/2609.35814)

    提出 BreakingWeb 基准，通过在保持用户指令与成功标准不变的前提下对环境不同技术栈层面进行确定性干预，将浏览器任务的难度转化为可编程的环境属性，从智能体已能解决的任务中系统性构建出 519 对更具挑战性的任务对。

    

    随着浏览器使用智能体的不断提升，基准测试也在通过收集新的任务、网站和应用程序来跟进，这通常使任务变得更长或更新颖。这使得任务难度的更新成本高昂且难以控制：当多个方面同时发生变化时，人们无法确定究竟是什么因素使任务变得具有挑战性。我们转而从智能体已经能够解决的任务出发来构建具有挑战性的实例，将难度转化为环境的一种可编程属性。BreakingWeb 将每个基础任务与一个干预条件配对，该条件在保持用户指令、潜在目标和后端成功标准不变的前提下，在不同层次的 Web 技术栈上对环境进行改变。每个干预都是确定性的、可检测的且可恢复的，并标注了其主要用于考验的认知基元。该基准测试包含 519 对干净/干预任务对，涵盖七个自托管网站和 29 个干预类别，全部基于结果进行评分。我们评估了……

    arXiv:2609.35814v1 Announce Type: cross  Abstract: As browser-use agents improve, benchmarks keep pace by collecting new tasks, websites, and applications, often making tasks longer or more novel. This makes difficulty expensive to refresh and difficult to control: when many aspects change at once, it is unclear what actually makes a task challenging. We instead construct challenging instances from tasks agents already solve, turning difficulty into a programmable property of the environment. BreakingWeb pairs every base task with an intervention condition that preserves the user instruction, latent target, and backend success criterion while changing the environment at different web stack layers. Each intervention is deterministic, detectable, and recoverable, and is annotated with the cognitive primitive it primarily loads. The benchmark contains 519 clean/intervention task pairs across seven self-hosted websites and 29 intervention families, all graded against outcomes. We evaluate 
    
[^212]: 大语言模型智能体社会中的局部可预测性与集体保真度

    Local Predictability and Collective Fidelity in LLM-Agent Societies

    [https://arxiv.org/abs/2609.35813](https://arxiv.org/abs/2609.35813)

    本研究利用9,455条轨迹和舆论动力学实验，比较了LLM智能体社会模拟中代理模型的个体预测与集体预测能力，发现邻居信息可全面改善个体预测，但集体层面的收益依赖迁移条件，强调了直接集体验证、明确观测限制及简单基线比较的必要性。

    

    紧凑的代理模型可以降低模拟大语言模型社会的成本，但必须能够再现集体行为。我们使用9,455条已发布的轨迹以及关于舆论动力学的新实验，比较了个体预测与集体预测。邻居信息在所有16个公开数据设置中均改善了个体预测，并在保留问题上的汇总集体预测中也有所提升，尽管集体层面的收益取决于迁移条件。对24条新陈述的测试未能证实此前研究中从初始状态进行预测时所发现的对比性历史效应。Qwen模型在观察到三轮之后能够从历史信息中获益。这些发现促使研究者进行直接的集体验证、对可用观测次数设定明确限制，并与简单基线进行比较。

    arXiv:2609.35813v1 Announce Type: cross  Abstract: Compact surrogates could reduce the cost of simulating large language model societies, but must reproduce collective behavior. We compare individual predictions and collective forecasts using 9,455 published trajectories and new experiments on opinion dynamics. Neighbor information improves individual prediction in all 16 public-data settings and pooled collective forecasts on held-out questions, although collective gains depend on transfer conditions. Tests on 24 new statements do not confirm earlier contrasting history effects in forecasts from the initial state. Qwen benefits from history after three observed rounds. These findings motivate direct collective validation, explicit limits on available observations, and comparisons with simple baselines.
    
[^213]: 车载对话助手多轮对话的自动化评估

    Automated Evaluation of Multi-Turn Dialogues in In-Car Conversational Assistants

    [https://arxiv.org/abs/2609.35812](https://arxiv.org/abs/2609.35812)

    提出了一个自动化测试框架，通过闭环仿真结合策略引导的用户模拟器、对抗性策略管理器和双层LLM裁判，来评估车载对话助手在多轮对话中的约束处理、上下文保持和安全关键行为。

    

    车载对话助手（ICAs）日益被集成到车辆中，以支持路线规划、车辆控制和信息获取。由于多轮交互、缺乏明确的真实标准（ground truth）以及严格的安全约束，确保其可靠性极具挑战性。现有评估技术存在不足，因为它们针对单轮对话设置，无法捕捉跨轮次的约束处理、上下文保持以及安全关键行为。我们提出了一个用于测试车载对话助手多轮对话能力的自动化框架。该系统被视为黑盒，通过闭环仿真进行评估，框架包含一个策略引导的用户模拟器、一个对抗性策略管理器，以及一个双层LLM裁判，用于评估轮次级的故障和对话级的质量。我们在一个工业级车载对话助手上评估了该方法，使用了六个LLM后端和十二名人类标注者。自动化裁判显示出高度的一致性……

    arXiv:2609.35812v1 Announce Type: new  Abstract: In-car conversational assistants (ICAs) are increasingly integrated into vehicles to support route planning, vehicle control, and information access. Ensuring their reliability is challenging due to multi-turn interactions, the absence of explicit ground truth, and strict safety constraints. Existing evaluation techniques fall short, as they target single-turn settings and fail to capture constraint handling, context retention, and safety-critical behavior across turns. We propose an automated framework for testing the multi-turn conversational capabilities of ICAs. The system is treated as a black box and evaluated via closed-loop simulation with a strategy-guided user simulator, an adversarial strategy manager, and a two-tier LLM judge assessing turn-level failures and conversation-level quality. We evaluate the approach on an industrial ICA with six LLM backends and twelve human annotators. The automated judge shows substantial agreem
    
[^214]: Lookahead-R：基于以执行为中心规划的预算感知工具检索

    Lookahead-R: Budget-Aware Tool Retrieval via Execution-Centric Planning

    [https://arxiv.org/abs/2609.35811](https://arxiv.org/abs/2609.35811)

    Lookahead-R通过轻量级执行感知代理世界模型（无需调用真实API即可预测工具执行结果、延迟与语义效用），结合预算感知的蒙特卡洛树搜索，将工具检索转化为资源受限的序贯决策问题，实现了精度与效率的最优平衡。

    

    工具检索是基于大语言模型（LLM）的智能体在大型异构API生态系统中运行的关键瓶颈。现有方法面临固有的权衡困境：语义检索器速度快但受制于语义-功能鸿沟，而基于执行的验证虽能提升精度，却带来难以接受的延迟。我们提出Lookahead-R，这是一个基于规划的框架，将工具检索重新表述为资源受限的序贯决策问题。其核心在于，Lookahead-R引入了一个轻量级的执行感知代理世界模型，无需调用真实API即可联合预测工具执行成功率、延迟成本和语义效用。该世界模型驱动一个成本敏感、不确定性引导的蒙特卡洛树搜索，在严格预算约束下探索工具空间。在大规模ToolBench基准上的评估表明，Lookahead-R在所有测试场景中均实现了卓越的准确性-效率权衡。

    arXiv:2609.35811v1 Announce Type: cross  Abstract: Tool retrieval is a critical bottleneck for LLM-based agents operating over large, heterogeneous API ecosystems. Existing approaches face an inherent trade-off: semantic retrievers are fast but suffer from the semantic-functional gap, while execution-based validation improves precision at the cost of prohibitive latency. We propose Lookahead-R, a planning-based framework that reformulates tool retrieval as a resource-constrained sequential decision-making problem. At its core, Lookahead-R introduces a lightweight execution-aware surrogate world model that jointly predicts tool execution success, latency cost, and semantic utility---without invoking real APIs. This world model drives a cost-sensitive, uncertainty-guided Monte Carlo Tree Search that navigates the tool space under strict budget constraints. Evaluated on the large-scale ToolBench benchmark, Lookahead-R achieves a superior accuracy-efficiency trade-off across all test scena
    
[^215]: TRACE：面向肿瘤学大语言模型的可部署树状关系结构增强

    TRACE: Deployable Tree-Relational Structure Enhancement for Oncology LLMs

    [https://arxiv.org/abs/2609.35810](https://arxiv.org/abs/2609.35810)

    TRACE框架通过将肿瘤学概念组织成可更新的树状关系结构，在推理时检索紧凑的提示证据，无需监督标签即可在零样本设置下提升肿瘤学大语言模型在分类与问答任务上的表现。

    

    大语言模型在肿瘤学应用中的使用日益增多，但其预测往往缺乏与显式医学结构的紧密关联。我们提出了TRACE，一个面向肿瘤学大语言模型的可部署树状关系增强框架。TRACE将昂贵的离线结构学习与轻量的在线推理相分离：肿瘤学概念和关系被组织成一个可更新的树状关系结构，利用基于语言模型损失的证据进行优化，并在推理时作为紧凑的提示证据被检索。这种设计支持任务自适应的证据选择，且在零样本设置下无需监督标签。在十项肿瘤学分类任务和一个MedQuAD CancerGov问答基准上，TRACE同时提升了无标签评估和监督微调的性能。额外的分析表明，TRACE优于原始的RAG和通用的GraphRAG，在泄漏控制的METABRIC输入下仍然有效，并产生可解释的结果。

    arXiv:2609.35810v1 Announce Type: cross  Abstract: Large language models are increasingly used in oncology applications, but their predictions are often weakly grounded in explicit medical structure. We present TRACE, a deployable tree-relational enhancement framework for oncology LLMs. TRACE separates expensive offline structure learning from lightweight online inference: oncology concepts and relations are organized into an updatable tree-relational structure, refined using LM-loss-derived evidence, and retrieved at inference time as compact prompt evidence. This design supports task-adaptive evidence selection without requiring supervised labels in the zero-shot setting. Across ten oncology classification tasks and one MedQuAD CancerGov QA benchmark, TRACE improves both label-free evaluation and supervised fine-tuning. Additional analyses show that TRACE improves over vanilla RAG and generic GraphRAG, remains useful under leakage-controlled METABRIC inputs, and produces interpretabl
    
[^216]: 多模态大语言模型能否生成并检测社交媒体上的多模态假新闻？

    Can Multimodal Large Language Models Generate and Detect Multimodal Social Media Fake News?

    [https://arxiv.org/abs/2609.35809](https://arxiv.org/abs/2609.35809)

    该论文提出一个由故事、图像和评论三个智能体协作的多智能体框架，生成了超过9,000条多模态假新闻，并通过对16个MLLMs的基准测试发现，现有模型的假新闻检测能力显著低于人类水平，尤其在识别图像真实性方面表现严重不足。

    

    生成式人工智能的快速发展引发了人们对多模态大语言模型（MLLMs）被滥用于社交媒体大规模虚假信息活动的担忧。尽管已有关于文本虚假信息的研究，但一个根本问题仍未得到解答：MLLMs能否被利用来制造逼真的多模态假新闻？它们又能否可靠地检测出这些假新闻？我们提出了一个多智能体框架，其中故事智能体、图像智能体和评论智能体协作生成能够令人信服地反驳真实新闻的虚假社交媒体帖子。我们将该框架应用于科学、健康和娱乐领域，生成了超过9,000对多模态新闻帖子，并对16个开源和闭源MLLMs进行了自动检测的基准测试。我们发现，大多数模型的准确性远低于人类水平，并且在识别图像真实性方面存在严重缺陷。我们的研究为开发针对社交媒体假新闻的强大防御措施奠定了基础。

    arXiv:2609.35809v1 Announce Type: cross  Abstract: The rapid advancement of generative AI raises concerns about the misuse of Multimodal LLMs (MLLMs) for large-scale disinformation campaigns on social media. Despite existing research on textual disinformation, a fundamental question remains unanswered: can MLLMs be exploited to fabricate realistic multimodal fake news, and can they reliably detect it? We introduce a multi-agent framework in which a story agent, an image agent, and a critic agent collaborate to produce fake social media posts that plausibly counter true news. We apply the framework to generate over 9,000 paired multimodal news posts across science, health, and entertainment domains, and benchmark 16 open- and closed-source MLLMs for automated detection. We find that most models fall substantially short of human-level accuracy and fail critically on identifying image authenticity. Our research provides a foundation for developing robust defenses against social media fake
    
[^217]: 当成功记忆误导具身智能体：面向任务条件执行的记忆适配

    When Successful Memories Mislead Embodied Agents:Memory Adaption For Task-Conditioned Execution

    [https://arxiv.org/abs/2609.35808](https://arxiv.org/abs/2609.35808)

    提出 MATE——一种无需额外 LLM 推理的确定性检索后处理方法，将历史成功轨迹改写为面向执行的记忆，从而避免“成功经验误导”，在 ALFWorld 上使 Qwen2.5-14B/72B 分别达到 81.3% 和 93.3% 的任务成功率。

    

    经验复用可以减少具身智能体的重复探索，但曾经成功的轨迹未必适合当前的执行情境。现有记忆系统主要优化构建与检索环节，因此当检索到的经验包含不兼容的动作或不合适的结构化程度时，仅凭语义相关性和历史成功率仍然不够。我们提出了面向任务条件执行的记忆适配方法 MATE，这是一种确定性的检索后处理程序，可将轨迹转化为面向执行的记忆。MATE 移除过时的控制上下文，提取“条件-动作-效果”转移，应用经过验证的动作规范化，选择任务相关的表示形式，并在固定预算下对结果进行序列化，而无需额外的 LLM 推理。在 134 个 ALFWorld 任务上，MATE 使用 Qwen2.5-14B 和 72B 分别取得了 81.3% 和 93.3% 的任务成功率，同时仅使用约（摘要原文在此处截断）。

    arXiv:2609.35808v1 Announce Type: cross  Abstract: Experience reuse can reduce repeated exploration in embodied agents, but a trajectory that succeeded previously may be unsuitable for the current execution context. Existing memory systems pri marily optimize construction and retrieval; semantic relevance and historical success therefore remain insufficient when retrieved ex perience contains incompatible actions or an inappropriate level of structure. We introduce Memory Adaptation for Task-Conditioned Execution (MATE), a deterministic post-retrieval procedure that converts trajectories into execution-oriented memory. MATE re moves obsolete control context, extracts condition-action-effect transitions, applies verified action normalization, selects a task dependent representation, and serializes the result under a fixed budget without additional LLM inference. On 134 ALFWorld tasks, MATE achieves task success rates of 81.3% and 93.3% with Qwen2.5-14B and 72B while using approximately 
    
[^218]: 环境引导：利用数据流控制提升智能体的效用与安全性

    Environment Steering: Using Data Flow Control to Improve Agent Utility and Safety

    [https://arxiv.org/abs/2609.35807](https://arxiv.org/abs/2609.35807)

    该论文提出“环境引导”方法，将智能体执行状态建模为数据库表并在运行时用声明式策略检查记录级数据流，检测到违规时通过针对性反馈引导智能体走向安全轨迹，从而在提升任务成功率的同时实现0%的攻击成功率。

    

    大语言模型智能体即使在被指示安全行事的情况下，也可能做出不安全的工具调用。现有防御方法在执行前约束智能体、修改工具输入/输出，或依赖大语言模型裁判；这些方法可能依赖于模型自身行为，或者只是阻止不安全操作而不帮助智能体恢复。我们认为，执行环境应该在智能体运行时强制执行安全性，并在发生违规时引导其转向安全的替代方案——我们称之为“环境引导”。我们通过将智能体和执行框架的运行状态建模为数据库表，跟踪记录级数据流，并在运行时依据声明式策略检查这些数据流来实现这一目标。当检测到违规时，基于策略和上下文的针对性反馈会引导智能体走向安全轨迹。在AgentDyn基准上，该方法使智能体在比无防御情况更高的任务成功率下，实现了0%的攻击成功率。

    arXiv:2609.35807v1 Announce Type: cross  Abstract: LLM agents can make unsafe tool calls even when instructed to behave safely. Existing defenses constrain agents before execution, modify tool inputs/outputs, or rely on LLM judges; these approaches may depend on model behavior or block unsafe actions without helping the agent recover. We argue that the execution environment should instead enforce safety as the agent runs and steer it toward safe alternatives when violations occur---we call this Environment Steering. We implement this by modeling the agent and harness execution state as database tables, track the record-level data flows, and check these data flows against declarative policies during runtime. When violations are detected, policy- and context-specific feedback steers the agent toward safe trajectories. On AgentDyn, this enables the agent to improve task success rate over no-defense while achieving 0% attack success rate.
    
[^219]: 从词汇基线到智能体检索增强生成：基于SFIA框架的结构化技能与责任级别提取

    From Lexical Baselines to Agentic Retrieval-Augmented Generation: Structured Skill and Responsibility-Level Extraction with the SFIA Framework

    [https://arxiv.org/abs/2609.35806](https://arxiv.org/abs/2609.35806)

    本文首次将SFIA框架下的结构化技能与责任级别提取任务形式化，系统对比了从词汇基线到多智能体检索增强生成等五种策略在该任务上的表现。

    

    自动化技能提取是劳动力规划的基础支撑，然而大多数系统将技能表示为扁平的标签，没有涉及技能实践所处责任级别这一概念。信息时代技能框架（SFIA）恰好捕捉了这一维度，定义了147项专业技能并划分为七个责任级别，但此前尚无针对SFIA的基于大语言模型（LLM）的自动化提取方法被报道。我们将该任务形式化为从自由文本中对（技能，级别）配对进行结构化预测，并提出三个问题：文本能以多高的准确度映射到SFIA的封闭词汇表上；哪些策略能够在预测技能的同时可靠地预测其级别；以及智能体化设计是否优于更简单的检索和提示方法。我们评估了五种策略（词汇基线方法、带LLM重排序的稠密检索、零样本模式约束LLM、单智能体RAG、以及三智能体“检索器-匹配器-验证器”团队），并对照专家标注的欧洲ICT数据进行验证。

    arXiv:2609.35806v1 Announce Type: cross  Abstract: Automated skill extraction underpins workforce planning, yet most systems represent skills as flat labels with no notion of the responsibility level at which a skill is practiced. The Skills Framework for the Information Age (SFIA) captures exactly this dimension, defining 147 professional skills across seven responsibility levels, but no automated LLM-based extraction targeting SFIA has been reported. We formalize the task as structured prediction of (skill, level) pairs from free text and ask three questions: how accurately can text be mapped onto SFIA's closed vocabulary, which strategies reliably predict the level alongside the skill, and do agentic designs improve on simpler retrieval and prompting? We evaluate five strategies (a lexical baseline, dense retrieval with LLM reranking, a zero-shot schema-constrained LLM, single-agent agentic RAG, and a three-agent retriever--matcher--verifier crew) against expert-mapped European ICT 
    
[^220]: 对齐预测：从训练数据预测模型失调

    Alignment Forecasting: Predicting Misalignment From Training Data

    [https://arxiv.org/abs/2609.35805](https://arxiv.org/abs/2609.35805)

    提出了“对齐预测”任务及包含5,000多个预测问题、覆盖17个目标模型、32个数据集和16种失败模式的ALIGNMENTFORECASTBENCH基准，用于在训练前预测微调数据是否会引发欺骗、谄媚等对齐失败。

    

    在带有细微缺陷的数据上训练语言模型，有时会使模型产生广泛的失调（不对齐）现象。仅凭表面检查数据往往无法判断这种失调是否会出现，而目前只能在训练完成后通过审计模型才能发现问题。为了补充事后审计，我们提出了“对齐预测”（Alignment Forecasting）这一任务：即在训练之前预测对齐失败。给定一个目标模型、一个微调数据集以及一种失败模式（如欺骗或谄媚），预测器会输出微调将显著增加该失败模式发生概率的概率。为了衡量对齐预测的进展，我们引入了ALIGNMENTFORECASTBENCH，这是一个包含超过5,000个预测问题的基准，涵盖17个目标模型、32个数据集和16种失败模式。直接进行提示的前沿模型在ALIGNMENTFORECASTBENCH上表现不佳。因此，我们提出了一个预测框架（scaffold），让大语言模型阅读数据集并评估其强度……

    arXiv:2609.35805v1 Announce Type: cross  Abstract: Training a language model on data with a narrow flaw can sometimes make the model broadly misaligned. Inspecting the data at face value often does not settle whether it will emerge, and today it is caught only after training, by auditing the resulting model. To complement post-hoc audits, we introduce Alignment Forecasting: the task of predicting alignment failures before training. Given a target model, a fine-tuning dataset, and a failure mode such as deception or sycophancy, a forecaster outputs the probability that fine-tuning would meaningfully increase that failure mode. To measure progress on alignment forecasting, we introduce ALIGNMENTFORECASTBENCH, a benchmark of over 5,000 forecasting questions spanning 17 target models, 32 datasets, and 16 failure modes. Frontier models prompted directly perform poorly on ALIGNMENTFORECASTBENCH. We therefore propose a forecasting scaffold in which an LLM reads the dataset and rates how stron
    
[^221]: 评估提示扰动对大语言模型中偏见与幻觉的影响

    Evaluating the Effects of Prompt Perturbation on Bias and Hallucination in Large Language Models

    [https://arxiv.org/abs/2609.35804](https://arxiv.org/abs/2609.35804)

    本研究评估了提示扰动对大语言模型在决策任务中偏见与幻觉的影响，发现与以往研究相反，扰动可以在某些模型中缓解偏见和幻觉，其中 Claude 3 表现最为有效，而 GPT3.5 的表现则参差不齐。

    

    大语言模型（LLM）在各类自然语言处理任务中展现出卓越的能力，因此被广泛部署为决策场景中的智能助手。然而，这些模型日益增加的复杂性引发了对其可靠性的担忧，尤其是在偏见和幻觉方面。在本工作中，我们评估了大语言模型在决策任务中对原始查询的扰动变体的鲁棒性。我们表明，与以往研究相反，扰动能够在某些大语言模型中缓解偏见和幻觉，而在其他模型中则不然。研究发现，Claude 3 在大多数数据集所代表的任务上更为有效，而像 GPT3.5 这样的模型则表现出不同程度的适应性，在某些情况下表现相当，但在其他情况下则明显落后。这些见解对于理解将基于大语言模型的助手部署为有效决策工具的实际影响至关重要。

    arXiv:2609.35804v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown remarkable capabilities in various natural language processing tasks, leading to their widespread deployment as intelligent assistants in decision-making contexts. However, the increasing complexity of these models raises concerns about their reliability, particularly regarding bias and hallucination. In this work, we evaluate the robustness of LLMs to perturbed variations of the original inquiry in decision-making tasks. We show that contrary to previous studies, perturbations can mitigate bias and hallucination in some LLMs over other models. It's found that Claude 3 is more effective for the tasks represented in most datasets, whereas models like GPT3.5 exhibit varying levels of adequacy, performing comparably in some cases but falling significantly behind in others. These insights are crucial for understanding the practical implications of deploying LLM-based assistants as effective decision-
    
[^222]: 开发用于从韩语发票中提取信息的OCR模型

    Developing an OCR model for Extracting Information from Invoices with Korean Language

    [https://arxiv.org/abs/2609.35796](https://arxiv.org/abs/2609.35796)

    提出了一种结合深度学习与图像预处理技术的高效OCR模型，用于自动从韩语发票中提取关键信息，在收集的发票数据集上达到87%的F1分数且处理时间极短。

    

    发票是包含各种信息的商业文件，包括购买的商品、时间和总金额，因此提取重要信息至关重要。所存储的信息可用于不同的目的。韩语是约8000万人的母语，不仅在韩国和朝鲜，而且在许多其他国家（如越南、菲律宾等拥有大量韩国公司的国家）也发挥着重要作用。在此背景下，为了从韩语发票中自动提取正确的信息，我们提出了一种高效的光学字符识别（OCR）模型，该模型将深度学习模型与一些图像预处理技术相结合。在收集到的大量发票数据集上对该OCR模型进行评估，结果表明，在处理时间可忽略不计的情况下，可以达到87%的F1分数。

    arXiv:2609.35796v1 Announce Type: new  Abstract: Invoices are commercial documents that contain various pieces of information, including the purchased items, time, and total money. Making the extraction of important information crucial. The stored information serves different purposes. Korean language is the native language of about 80 million people, playing an important role in not only South and North Korea but also in many other countries such as Vietnam, Philippine where a large number of Korean companies are located. In this context, to automatically extract proper information from the invoices with Korean language, we propose an efficient Optical Character Recognition (OCR) model in which a deep learning model is combined with some image preprocessing techniques. The proposed OCR model is assessed in a rich set of collected invoices showing that 87% F1-score can be achieved with negligible time processing.
    
[^223]: Sieve与Sage：面向可靠RALM弃答的高效干扰过滤

    Sieve and Sage: Efficient Distraction Filtering for Reliable RALM Abstention

    [https://arxiv.org/abs/2609.35794](https://arxiv.org/abs/2609.35794)

    该论文将检索失败分解为“不可回答”与“受干扰”两种状态，并提出轻量级的Sieve模块在调用昂贵大语言模型Sage之前预先过滤检索文档中的干扰证据，从而以更低的计算成本实现更可靠的RALM弃答。

    

    正如苏格拉底认识到自身知识的局限，检索增强语言模型（RALM）也应当学会在检索到的证据无法支撑可靠回答时进行弃答。现有方法大多依赖单一的大语言模型在一步之内处理异构的检索失败，导致弃答性能有限且计算成本高昂。我们转而将检索失败分解为两种不同的状态：（i）不可回答状态，即所需证据缺失；（ii）受干扰状态，即相关证据与冲突、否定或对抗性信息混杂在一起。基于这一分解，我们引入了一个轻量级模块Sieve，在调用昂贵的大语言模型Sage进行有据生成与弃答判断之前，先对检索到的文档集合筛查干扰证据。在通用领域与高风险专家领域的评估表明，我们的Sieve与Sage框架能够预先检测干扰信息，实现更可靠的弃答表现。

    arXiv:2609.35794v1 Announce Type: new  Abstract: Just as Socrates recognized the limits of his own knowledge, Retrieval-Augmented Language Models (RALMs) should learn to abstain when the retrieved evidence cannot support a reliable response. Existing approaches largely rely on monolithic LLMs to handle heterogeneous retrieval failures in a single step, resulting in limited abstention performance and high computational costs. We instead decompose retrieval failures into two distinct states: (i) the unanswerable state, where the required evidence is absent, and (ii) the distracted state, where relevant evidence is mixed with conflicting, negated, or adversarial information. Based on this decomposition, we introduce a lightweight module (Sieve) that screens retrieved document sets for distracting evidence before invoking a costly LLM (Sage) for grounded generation and abstention. Evaluated across both general and high-stakes expert domains, our Sieve and Sage framework preemptively detect
    
[^224]: FD-VAD：面向流式全双工语音的语义端点检测

    FD-VAD: Semantic Endpoint Detection for Streaming Full-Duplex Speech

    [https://arxiv.org/abs/2609.35791](https://arxiv.org/abs/2609.35791)

    FD-VAD提出了一种无需ASR的流式语义端点检测方法，通过因果音频-语言推理直接判断用户的停顿是犹豫还是话轮结束，从而实现更自然的全双工语音交互。

    

    全双工语音交互中自然的话轮转换需要从部分语音中判断停顿究竟反映了用户的犹豫，还是对话意图已经完成。声学的语音活动检测缺乏这种语义信息，而基于级联ASR的端点检测则引入了对转写文本的依赖和额外的处理阶段。我们将语义端点检测形式化为一个因果的音频-语言推理任务，并提出了FD-VAD——一种无需ASR的流式端点检测器，能够将有界的因果音频窗口直接映射为继续/停止决策。FD-VAD将冻结的语音编码器与轻量级模态适配器以及参数高效适配的语言模型相结合，并采用最后块训练目标以支持流式推理。我们进一步引入了置信度门控的端点确认机制来权衡打断与延迟，以及聚焦边界的困难负样本采样，以改进在模糊话轮边界附近的决策。

    arXiv:2609.35791v1 Announce Type: new  Abstract: Natural turn-taking in full-duplex voice interaction requires determining from partial speech whether a pause reflects hesitation or a completed conversational intent. Acoustic voice activity detection lacks this semantic information, while cascaded ASR-based endpointing introduces transcription dependence and additional processing stages. We formulate semantic endpoint detection as a causal audio-language reasoning task and introduce FD-VAD, an ASR-free streaming endpointer that maps bounded causal audio windows directly to Continue/Stop decisions. FD-VAD combines a frozen speech encoder with a lightweight modality adapter and a parameter-efficiently adapted language model, using a last-chunk training objective for streaming inference. We further introduce confidence-gated endpoint commitment to control interruption versus delay and boundary-focused hard-negative sampling to improve decisions around ambiguous turn boundaries. Across in-
    
[^225]: Sage: 基于语义修正的形式化

    Sage: Formalization with Semantic Correction

    [https://arxiv.org/abs/2609.35790](https://arxiv.org/abs/2609.35790)

    本文提出 Sage，一个智能体化的形式化引擎，通过四阶段分解式生成流水线与融合 Lean 4 编译器诊断和多维语义反馈的双信号语义修正循环，解决了自然语言翻译为 Lean 4 形式化命题时的“严谨性幻觉”问题，确保生成命题既句法有效又数学忠实。

    

    arXiv:2609.35790v1 公告类型：交叉（cross） 摘要：尽管神经定理证明器已在形式数学领域取得了令人瞩目的里程碑式进展，但它们大多建立在一个假设之上，即已被忠实翻译的 Lean 4 形式化命题是现成可用的。将非形式的自然语言翻译为形式语言是一个关键的数据瓶颈，且这一过程深受“严谨性幻觉”的困扰：标准类型检查器会接受那些能够编译通过、却丢弃了假设条件、引入了空洞真命题或微妙地改变了数学界限的命题。为解决这一问题，我们提出了 Sage（语义智能体引导的形式化引擎，Semantic Agent-Guided Formalization Engine），这是一个智能体化框架，它用一个四阶段分解式生成流水线取代了单体式翻译，并耦合了一个双信号语义修正循环。通过将 Lean 4 编译器诊断信息与多维度的语义反馈相结合，我们的修正循环在保证句法有效性的同时强制实现数学忠实性。通过显式地考虑开放式查询与声明式形式化目标之间的差距……

    arXiv:2609.35790v1 Announce Type: cross  Abstract: While neural theorem provers have achieved impressive milestones in formal mathematics, they largely operate on the assumption that faithful Lean 4 formal statements are already provided. Translating informal natural language into a formal language is a critical data bottleneck plagued by an "illusion of rigor": standard type-checkers accept statements that compile but drop hypotheses, introduce vacuous truths, or subtly alter mathematical bounds. To resolve this, we introduce Sage (Semantic Agent-Guided Formalization Engine), an agentic framework that replaces monolithic translation with a four-stage decomposed generation pipeline coupled with a dual-signal semantic correction loop. By pairing Lean 4 compiler diagnostics with multi-dimensional semantic feedback, our correction loop enforces mathematical fidelity alongside syntactic validity. By explicitly accounting for the gap between open-ended queries and declarative formal targets
    
[^226]: 大型语言模型表现出类人的贝叶斯伪善

    Large Language Models Exhibit Human-Like Bayesian Hypocrisy

    [https://arxiv.org/abs/2609.35779](https://arxiv.org/abs/2609.35779)

    研究发现GPT-4o和Claude 3.7 Sonnet在贝叶斯推理任务上的表现接近人类水平，但会像人类一样（甚至更严重地）伪善地谴责使用相同贝叶斯推理的他人。

    

    鉴于大语言模型（LLM）近期的成就，前沿模型被期望在贝叶斯推理任务上表现良好，至少与人类相当。此外，没有理由预期LLM会谴责那些做出同样贝叶斯判断的人——这是在人类决策中观察到的一种谬误（Cao等人，2019）。在包含48个实验条件的5项实验、共计超过5,000次试验中，研究团队在贝叶斯推理任务的两种变体上对GPT-4o和Claude 3.7 Sonnet进行了测试，并评估了LLM对一位给出与它们相同推理答案的假设人物的能力与道德的评价。结果显示，LLM在贝叶斯任务上的表现接近人类水平，但其推理更加基于规则且更为僵化。令人惊讶的是，LLM与人类一样（甚至程度更深）表现出同样的伪善行为，谴责那些像自己一样运用贝叶斯法则的他人。

    arXiv:2609.35779v1 Announce Type: cross  Abstract: Given recent achievements of large language models (LLMs), frontier models are expected to perform well on Bayesian reasoning tasks, at least as well as humans. Furthermore, there is no reason to expect that LLMs will condemn others who offer those very same Bayesian judgments, a fallibility observed in human decision-making (Cao, et al., 2019). In 5 experiments with 48 experimental conditions employing over 5,000 trials, GPT-4o and Claude 3.7 Sonnet were tested on two variations of a Bayesian reasoning task. We also assessed LLM evaluation of the competence and morality of a hypothetical person who had offered the same reasoning task as them. LLMs hovered near human performance on the Bayesian task, though their reasoning was more rule-based and rigid. Surprisingly, like humans but to a greater extent, LLMs also demonstrated the same hypocrisy in condemning others who, like them, had deployed Bayes' rule. In demonstrating Bayesian hyp
    
[^227]: 追溯三千年来甲骨文字的演变

    Tracing the Evolution of Oracle Bone Characters Across Three Millennia

    [https://arxiv.org/abs/2609.35674](https://arxiv.org/abs/2609.35674)

    该论文提出基于流形的文字演变框架（MSEF），通过神经常微分方程将汉字从甲骨文到楷书的演变建模为流形空间的连续演化，从而突破单一时期参照的局限以辅助甲骨文破译。

    

    在商代出土的约4500个甲骨文（OBI）字符中，仅有约1600个已被破译。许多计算方法每次只将甲骨文与单一历史时期的字形进行比较。然而，在汉字演变过程中，重大的结构或语义变化往往发生在不确定的朝代，当相关字形在所观察到的时代之间发生显著变化时，单一时期的参照可能并不足够。因此，我们提出了基于流形的文字演变框架（MSEF），该框架将汉字的演变序列（甲骨文、金文、小篆、隶书、楷书）建模为流形空间的持续演化。MSEF将每个字符表示为特定时代的流形点，并通过神经常微分方程学习时代之间的连续过渡规则。流形空间和过渡动力学均可进行端到端训练……

    arXiv:2609.35674v2 Announce Type: replace  Abstract: Of the approximately 4,500 Oracle Bone Inscription (OBI) characters discovered from the Shang dynasty, only about 1,600 have been deciphered. Many computational approaches compare OBI with glyphs from one historical period at a time. However, during the evolution of Chinese characters, significant structural or semantic changes often occur in uncertain dynasties. A single-period reference may be insufficient when relevant forms change substantially between observed eras. Therefore, we propose the \textbf{Manifold-based Script Evolution Framework (MSEF)}, a framework that models the evolution series (OBI, Bronze, Seal, Clerical, Regular) of Chinese characters as the continual evolution of a manifold space. MSEF represents each character as an era-specific manifold point and learns continuous inter-era transition rules via Neural Ordinary Differential Equations. Both manifold space and transition dynamics can be trained end-to-end thro
    
[^228]: 迈向文化适应的中文语言智能体：中德跨文化互动中非语言行为的绿野仙踪式研究

    Toward a Culturally Adapted Chinese Language Agent: A Wizard-of-Oz Study of Nonverbal Behavior in Chinese-German Intercultural Interaction

    [https://arxiv.org/abs/2609.35150](https://arxiv.org/abs/2609.35150)

    本文提出了一套结合逼真虚拟人形象与实时多模态数据采集的“绿野仙踪”研究系统，用于捕捉中国母语者对德国汉语学习者违反社会规范时的非语言反应，从而为开发具备文化适应能力的中文语言智能体奠定基础。

    

    成功的跨文化交流不仅需要语法能力，还需要对根植于文化的社会规范的敏感性——违反这些规范会引发微妙但意义深远的非语言反应。对于学习汉语普通话的德国学习者而言，培养这种敏感性至关重要，然而现有的语言学习智能体对此支持甚少。我们提出了一项“绿野仙踪”研究设计和配套的实时系统，用于收集中国母语者在面对德国学习者违反社会规范时产生的多模态行为数据。该系统具有由Live Link面部捕捉和MediaPipe上半身追踪技术驱动的逼真MetaHuman虚拟形象、用于实时行为选择的向导控制台，以及在智能体与学习者数据流之间同步记录的多模态日志。此外，该系统还包含一个基于心理学理论的分层标注框架，涵盖不可观察的社会情感反应、规范解读、言语以及可观察行为……

    arXiv:2609.35150v1 Announce Type: cross  Abstract: Successful intercultural communication requires more than grammatical competence. It demands sensitivity to culturally embedded social norms whose violation triggers subtle but meaningful nonverbal responses. For German learners of Mandarin Chinese, acquiring this sensitivity is critical yet poorly supported by existing language-learning agents. We present a Wizard-of-Oz (WoZ) study design and supporting real-time system for collecting multimodal behavioral data from native Chinese speakers reacting to social norm violations by German learners. The system features a photorealistic MetaHuman avatar driven by Live Link face capture and MediaPipe upper-body tracking, a wizard console for real-time behavior selection, and synchronized multimodal logging across agent and learner streams. A layered annotation framework, based on psychological theory and covering non-observable socioemotional reactions, norm interpretation, verbal, and observ
    
[^229]: 一次读取，多种修复：面向工具智能体修复的扩散引导分层搜索

    One Readout, Many Repairs: Diffusion-Guided Hierarchical Search for Tool-Agent Repair

    [https://arxiv.org/abs/2609.34879](https://arxiv.org/abs/2609.34879)

    提出ReCommit——一个无需训练的扩散引导框架，将工具智能体修复形式化为操作支持集上的分层搜索，通过可复用的搜索区域避免重复操作选择，高效找到能成功执行并满足原始请求的替代工具调用序列。

    

    工具智能体利用大语言模型通过外部工具执行操作，然而即使调用成功执行，用户的请求仍可能未被满足。工具智能体修复旨在寻找能够成功执行并满足原始请求的替代调用序列。然而，修复需要同时探索操作选择及其具体实现方式，这使得完整序列的重新生成成本高昂。此外，即使失败源于操作的具体实现方式，重新生成仍会重复进行操作选择。由此产生的挑战在于，在保持对替代操作及其实现方式探索的同时，减少这种重复。因此，我们将修复问题形式化为对操作支持集的分层搜索。我们提出的操作支持集是指允许的操作类型的集合，它们为具体的工具调用序列定义了可复用的搜索区域。我们提出了ReCommit，一个无需训练、扩散引导的框架，用于改进工具智能体的失败恢复。

    arXiv:2609.34879v2 Announce Type: replace  Abstract: Tool agents use large language models to act through external tools, yet successfully executed calls can still leave user requests unfulfilled. Tool-agent repair seeks alternative call sequences that execute successfully and fulfill the original requests. However, repair requires exploring both operation choices and their concrete realizations, making complete-sequence regeneration costly. Moreover, regeneration repeats operation selection even when failure arises from how those operations are realized. The resulting challenge is to reduce this repetition while preserving exploration of alternative operations and realizations. Therefore, we formulate repair as hierarchical search over operation supports, which we introduce as sets of permitted operation types that define reusable search regions for concrete tool-call sequences. We propose ReCommit, a training-free, diffusion-guided framework for improving tool-agent failure recovery 
    
[^230]: 修复之后：修正后智能体经验的迁移

    After the Fix: Transfer of Corrected Agent Experience

    [https://arxiv.org/abs/2609.34603](https://arxiv.org/abs/2609.34603)

    本研究通过3,300次运行系统评估了修复后的智能体经验向后续任务迁移的效果，发现修正经验带来的收益在很大程度上源于未修正基线较弱而非记忆质量的真正提升，且并非所有修正机制（如APEX）都能产生可比的修正收益。

    

    修复一个失败的回合能否使其经验成为下一个任务更好的记忆？我们将同一失败的源任务在修复被接受之前和之后迁移到一个固定的目标任务，并与独立执行进行对比。我们的3,300次运行涵盖了100对ThinkingBox任务以及相同的100对APEX任务（分别在有和没有源状态继承的条件下），共在十一种条件下进行实验。ThinkingBox的Full/Skill/Hybrid修正收益分别为44/29/32个百分点，修正后的性能比独立执行高出25/22/18个百分点；但在任务族层面，推断能力会减弱。然而，Full相比Skill多出的15个百分点修正差距中，有12个百分点来自未修正时表现更差，而非修正后的记忆质量更好。此外，Full的46次向上转变中有22次只是恢复了观察到的基线成功。两种APEX机制均未能展现出可比的整体修正收益。行动证据将工作流收益与可复用的义务联系起来，而约定冲突则与源任务本地的选择相关。（原文摘要在此处不完整）

    arXiv:2609.34603v2 Announce Type: replace  Abstract: Does repairing an episode make its experience a better memory for the next task? We transfer the same failed source before and after accepted repair to a fixed target, alongside independent execution. Our 3,300 runs cover 100 ThinkingBox pairs and the same 100 APEX pairs with and without source-state inheritance, under eleven conditions. ThinkingBox's Full/Skill/Hybrid correction gains are 44/29/32 percentage points, with corrected performance 25/22/18 points above independence; inference weakens at the task-family level. Yet 12 of Full's 15-point larger correction gap over Skill come from worse uncorrected performance, not better corrected memory. Moreover, 22 of Full's 46 upward transitions restore observed baseline success. Neither APEX regime establishes comparable aggregate correction benefits. Action evidence connects workflow gains with reusable obligations and convention conflicts with source-local choices. Text APEX's accept
    
[^231]: 如何驯服多头九头蛇？面向大语言模型的自适应多类别安全引导

    How to Tame a Multi-Headed Hydra? Adaptive Multi-Category Safety Steering for Large Language Models

    [https://arxiv.org/abs/2609.34514](https://arxiv.org/abs/2609.34514)

    提出CAM-Steer框架，通过将当前隐藏状态与安全/危险原型对比来估计各伤害类别的风险，在单个提示中多个伤害类别共存时自适应地协调激活引导的方向与强度，从而提升大语言模型的安全性。

    

    随着大语言模型（LLMs）日益普及，防止其对有害提示产生不安全响应对于其安全部署至关重要。激活引导通过在推理过程中修改内部激活而不更新模型参数，为提升LLM安全性提供了一种方法。然而，单个提示可能涉及多个伤害类别，针对某一类别的安全引导可能无法解决来自另一类别的有害内容。尽管自适应引导技术不断进步，现有方法在单个提示中同时出现多个伤害类别时，并未显式协调引导的方向与强度。为解决这一问题，我们提出了CAM-Steer，一个类别自适应的多类别安全引导框架。具体而言，它通过将当前隐藏状态与安全和危险原型进行比较，来估计与每个伤害类别相关的风险。然后利用估计出的风险来组合安全……

    arXiv:2609.34514v2 Announce Type: replace-cross  Abstract: As large language models (LLMs) become increasingly widespread, preventing unsafe responses to harmful prompts is essential for their safe deployment. Activation steering offers an approach to improving LLM safety by modifying internal activations during inference without updating model parameters. However, a single prompt can involve multiple harm categories, and steering toward safety in one category may leave harmful content from another unaddressed. Despite advances in adaptive steering, existing methods do not explicitly coordinate steering direction and strength when multiple harm categories co-occur within a single prompt. To address this problem, we propose CAM-Steer, a Category-Adaptive Multi-category Safety Steering framework. Specifically, it estimates the risk associated with each harm category by comparing the current hidden state with safe and unsafe prototypes. The estimated risks are then used to combine the saf
    
[^232]: 连贯性感知的开放式文本生成分布评估

    Coherence-Aware Distributional Evaluation of Open-Ended Text Generation

    [https://arxiv.org/abs/2609.34240](https://arxiv.org/abs/2609.34240)

    提出CHORD指标，通过在冻结大语言模型的隐藏状态空间中比较生成文本与人类文本的分布，有效检测开放式文本生成中局部流畅但全局不连贯的质量问题。

    

    现有的开放式生成评估指标衡量似然、词汇多样性或通用表示空间中的分布相似性，但可能会遗漏质量的关键维度。一个显著的盲点是全局连贯性：生成的文本可能在局部流畅的同时，在整体上存在矛盾、因果不一致或主题脱节。我们发现文本表示是检测这些失败的核心瓶颈，并提出了CHORD（连贯性感知的隐藏状态开放式生成参考距离），这是一种对连贯性敏感的分布度量指标。CHORD使用连贯性引导提示，在冻结的大语言模型的隐藏状态空间中对生成语料和人类撰写语料进行编码，并使用RBF-MMD比较所得的分布。为了测试连贯性敏感性和选择性，我们构建了一个反事实评估套件，将分级连贯性降级扰动与保持语义的对照样本配对。

    arXiv:2609.34240v2 Announce Type: replace-cross  Abstract: Existing open-ended generation metrics measure likelihood, lexical diversity, or distributional similarity in generic representation space, yet can miss fundamental dimensions of quality. A prominent blind spot is global coherence: a generated passage may be locally fluent while remaining globally contradictory, causally inconsistent, or topically disconnected. We identify representation as a central bottleneck in detecting these failures and introduce CHORD (Coherence-aware Hidden-state Open-generation Reference Distance), a coherence-sensitive distributional metric. CHORD encodes generated and human-written corpora in the hidden-state space of a frozen LLM using a coherence-eliciting prompt, and compares the resulting distributions using RBF-MMD. To test coherence sensitivity and selectivity, we construct a counterfactual evaluation suite pairing graded coherence-degrading perturbations with meaning-preserving controls. CHORD
    
[^233]: 大语言模型并非“随机鹦鹉”：来自类人造语言任务的意义中介抽象证据

    LLMs are not stochastic parrots: Evidence for meaning-mediated abstraction from conlang-like tasks

    [https://arxiv.org/abs/2609.34187](https://arxiv.org/abs/2609.34187)

    本研究通过类人造语言任务证明，大语言模型在没有任何示例输出的情况下，仅凭自然语言描述就能处理统计上罕见、且违背训练数据表层模式的虚构语言规则，表明其具备超越统计模式匹配的、基于意义中介的抽象能力，从而反驳了“随机鹦鹉”论断。

    

    “随机鹦鹉”论证的强版本声称，尽管大语言模型（LLM）可能超越机械复读，但它们无法从统计模式匹配跃升到抽象或推理，在本体论上仍接近模式重用的下限，尽管其生成的文本流畅诱人。我们使用类人造语言任务来检验这一假设。多个大语言模型仅获得关于虚构语言的自然语言描述，这些虚构语言通过结合统计上罕见且未经证实的特征，颠覆了训练数据中突出的表层模式。关键的是，实验中未提供任何示例输出。我们认为，如果模型表现出遵循规则的行为，它们就不可能仅仅依赖表层的统计模式；因为这类模式往往与正确的输出相悖。相反，成功的表现要求模型具备对提示中所指定约束的表征。在三个互补的任务族中，……

    arXiv:2609.34187v2 Announce Type: replace-cross  Abstract: The strong version of the stochastic parrot argument claims that, although large language models (LLMs) may exceed rote regurgitation, they cannot move beyond statistical pattern matching into abstraction or reasoning, remaining ontologically near the lower bound of pattern reuse despite producing alluringly fluent text. We test this hypothesis using conlang-like tasks. Several LLMs are given only natural-language descriptions of fictional languages that subvert prominent superficial patterns in training data by combining statistically uncommon and unattested features. Crucially, no example outputs are given. We argue that if the models exhibit rule-following behaviour, they cannot be relying solely on superficial statistical patterns; such patterns often work against the correct output. Instead, successful performance requires representations of the constraints specified in the prompt. Across three complementary task families,
    
[^234]: 量化误差具有谱平坦性：单个随机探针是一种经过校准、无需数据的敏感度估计器，及其在面向预算目标的混合精度量化中的应用

    Quantization Error Is Spectrally Flat: A Single Random Probe Is a Calibrated, Data-Free Sensitivity Estimator, with Application to Budget-Targeted Mixed-Precision Quantization

    [https://arxiv.org/abs/2609.33923](https://arxiv.org/abs/2609.33923)

    本文发现舍入误差具有谱平坦性，从而证明单个随机高斯探针即可作为经过校准、无需数据的逐张量量化敏感度无偏估计器，并据此提出RAM方法，在无校准数据条件下通过背包求解实现精确字节预算下的混合精度量化。

    

    单个随机高斯探针可以给出层的量化误差Frobenius范数平方的无偏估计。该估计器之所以表现良好，是因为舍入到最近值（round-to-nearest）误差具有谱平坦性。在来自一个350亿参数MoE模型和一个90亿参数稠密模型的1,683个张量上，有效维度为相同形状独立同分布噪声值的0.93至0.96倍，且在MoE模型上从2比特到8比特中位数保持不变。探针的变异系数可以从张量形状进行预测。单次探针测量可将每张量敏感度的测量误差控制在4%至7%以内；二十次探针可达到1.3%至1.4%。RAM将该估计器的传播形式应用于无需校准数据的预算目标混合精度量化。携带网络自身输入统计信息的高斯探针对每个张量在六种比特宽度下进行评分。背包求解器在精确的字节预算下分配比特，并设有防止灾难性2比特分配的护栏。单次探针遍历即可服务于任意预算。

    arXiv:2609.33923v2 Announce Type: replace-cross  Abstract: A single random Gaussian probe gives an unbiased estimate of the squared Frobenius norm of a layer's quantization error. The estimator is well-behaved because round-to-nearest error is spectrally flat. Across 1,683 tensors from a 35B MoE and a 9B dense model, effective dimensionality is 0.93 to 0.96 times the i.i.d. noise value of the same shape, and on the MoE the median is unchanged from 2-bit to 8-bit. The probe coefficient of variation is predictable from tensor shape. One probe measures per-tensor sensitivity to within 4 to 7%; twenty probes reach 1.3 to 1.4%.RAM applies the propagated form of this estimator to budget-targeted mixed-precision quantization with no calibration data. Gaussian probes carrying the network's own input statistics score every tensor at six bit-widths. A knapsack solver allocates bits under an exact byte budget, with guardrails against catastrophic 2-bit assignments. One probe pass serves any budge
    
[^235]: 大语言模型在被训练预测自身准确性时会学习到不同形式的元认知

    LLMs learn different forms of metacognition when trained to predict their own accuracy

    [https://arxiv.org/abs/2609.33886](https://arxiv.org/abs/2609.33886)

    训练大语言模型预测自身答题准确率时，模型学到的置信度包含两种不同的元认知信号，在接近训练数据的问题上反映真实准确率，在其他领域则反映答案分布的集中程度（输出一致性），且后者在训练早期即出现并可泛化。

    

    大语言模型被训练为无论是否具备相关知识都总是生成一个答案，这导致它们编造事实。先前的研究表明，大语言模型的置信度估计与其真实表现之间的对应关系很差，而微调可以显著改善这一点。然而，模型在此类训练中实际学到了什么仍知之甚少。我们通过训练10个开源权重大语言模型在回答事实性多选题之前预测自己的准确率，来研究大语言模型如何获得元认知监控，即知道自己知道什么的能力。我们发现，训练后的置信度反映了两种不同的信号：在接近训练数据的问题上，它跟踪模型的真实准确率；而在其他领域，它转而跟踪输出一致性，即模型答案分布的集中程度。输出一致性跟踪在训练早期就会出现，并能跨领域泛化（原文摘要至此被截断）。

    arXiv:2609.33886v2 Announce Type: replace  Abstract: Large language models are trained to always produce an answer, regardless of whether they possess the relevant knowledge, which leads them to fabricate facts. Prior work has shown that LLMs' confidence estimates correspond poorly to their actual performance, and that fine-tuning can substantially improve them. However, what models actually learn during such training remains poorly understood. We investigate how LLMs acquire metacognitive monitoring, the ability to know what one knows, by training 10 open-weight LLMs to predict their own accuracy on factual multiple-choice questions before answering them. We find that trained confidence reflects two distinct signals. While on questions close to the training data, it tracks the model's true accuracy, in other domains, it instead tracks output consistency: the concentration of the model's answer distribution. Output consistency tracking emerges early in training and generalizes across d
    
[^236]: 量化黑盒语言模型的行为尾部

    Quantifying Behavioral Tails in Black-Box Language Models

    [https://arxiv.org/abs/2609.33638](https://arxiv.org/abs/2609.33638)

    RareTrap框架通过代理LLM构建几何感知映射来诱导可复现的提示词分布，并结合序列稀有事件模拟技术，有效估计了黑盒大语言模型发生严重行为的概率。

    

    我们提出了RareTrap，一个用于估计黑盒大语言模型（LLM）中严重行为发生概率的框架。概率估计的一个关键挑战是在输入空间上定义一个可处理的分布。为实现这一目标，RareTrap使用一个代理LLM，并构建了一种几何感知的映射，将低维潜在参考空间映射到其token嵌入空间，从而在输入提示词上诱导出明确且可复现的分布。通过在响应上应用响应级别的性能函数来量化行为严重程度。这使得序列稀有事件模拟成为可能，即将评估集中在逐渐更严重的行为上，同时在诱导的提示词分布下保持概率——否则这种概率将难以测量。在10个开放权重模型和两个前沿模型（GPT-5.4和Claude Sonnet 4.6）上，我们发现RareTrap成功诱导了严重的资源共…（原文摘要在此处截断）

    arXiv:2609.33638v2 Announce Type: replace-cross  Abstract: We introduce RareTrap, a framework for estimating the probability of severe behaviors in black box large language models (LLMs). A key challenge for probability estimation is defining a tractable distribution over the input space. To accomplish that, RareTrap uses a surrogate LLM and constructs a geometry-aware mapping from a lower-dimensional latent reference space into its token-embedding space to induce an explicit and reproducible distribution over input prompts. A response-level performance function is utilized on the response to quantify behavior severity. This enables sequential rare event simulation that concentrates evaluations on progressively more severe behaviors while preserving probability under the induced prompt distribution, which would otherwise be prohibitive to measure. Across 10 open-weight and two frontier models (GPT-5.4 and Claude Sonnet 4.6), we find that RareTrap successfully induces severe resource co
    
[^237]: 教会自己看哪里：面向推理任务的在策略注意力自蒸馏

    Teach Yourself Where to Look: On-Policy Attention Self-Distillation for Reasoning

    [https://arxiv.org/abs/2609.33200](https://arxiv.org/abs/2609.33200)

    提出在策略注意力自蒸馏（OPASD），在token级监督之外增加以解答为条件的注意力蒸馏，将特权教师的注意力投影到学生可见位置并对齐，在竞赛级数学基准上将平均准确率提升4.98至8.40个百分点，同时避免回复长度膨胀。

    

    在策略自蒸馏利用来自特权教师（可访问已验证解答）的密集token分布指导，在推理模型自身生成的轨迹上进行训练。这种监督只能传递教师预测了什么内容，却无法直接传递教师在先前上下文中关注了哪里。我们提出在策略注意力自蒸馏（OPASD），在token级监督的基础上补充了以解答为条件的注意力蒸馏。由于特权教师能够关注学生无法看到的已验证解答token，OPASD将教师注意力投影到学生可见的位置上，并在对齐之前对所得分布进行重新归一化。在三种模型规模和四个竞赛级数学基准上，OPASD持续优于仅使用token监督的OPSD，平均准确率提升4.98至8.40个百分点。OPASD还能避免回复长度膨胀和性能下降（原文在此处截断）。

    arXiv:2609.33200v2 Announce Type: replace-cross  Abstract: On-policy self-distillation trains reasoning models on their own trajectories using dense token distribution guidance from a privileged teacher with access to a verified solution. This supervision transfers what the teacher predicts without directly transferring where it attends within the preceding context. We introduce On-Policy Attention Self-Distillation (OPASD), which complements token-level supervision with solution-conditioned attention distillation. Because the privileged teacher can attend to verified solution tokens unavailable to the student, OPASD projects teacher attention onto student-visible positions and renormalizes the resulting distribution before alignment. Across three model sizes and four competition-level mathematics benchmarks, OPASD consistently outperforms token-only OPSD, improving average accuracy by 4.98 to 8.40 percentage points. OPASD also avoids the response-length inflation and performance degra
    
[^238]: 知道不等于选择：显式验证在生成式偏好之外增加了什么

    Knowing Is Not Choosing: What Explicit Verification Adds Beyond Generative Preference

    [https://arxiv.org/abs/2609.33142](https://arxiv.org/abs/2609.33142)

    语言模型“知道”正确答案不代表会“选择”它：将事实回忆分解为生成、排序与选择三个步骤后发现，基于 P(True) 的显式验证在候选答案排序上显著优于对数似然等生成式偏好信号，可将多数投票准确率提升约 5 个百分点。

    

    生成正确答案并不意味着语言模型会最终选择它。我们将事实回忆分解为三个步骤：生成正确的候选答案、对可用候选答案进行排序、以及选择最终答案。在三个模型家族中，生成前的读取信号可以预测事实回忆表现以及采样将会覆盖哪些问题，但对一个可用的正确答案最终是否会被选择几乎没有预测能力。基于 $P(\mathrm{True})$ 的显式验证在 Gemma、Qwen3 和 Llama 中的问题内排序表现优于平均对数似然，AUROC 提升了 0.08–0.12。在一个前瞻性定义的 Gemma 队列中，验证将多数投票准确率提高了约 5 个百分点，并且相对于聊天模板似然这一更强的生成式基线仍有约 2 个百分点的提升。这种优势在具有常见答案先验的关系上最为显著，并且依赖于对实体的访问；掩盖实体会消除排序优势。（原文摘要在此处截断）

    arXiv:2609.33142v2 Announce Type: replace  Abstract: Generating a correct answer does not mean that a language model will select it. We separate factual recall into three steps: generating a correct candidate, ranking the available candidates, and selecting the final answer. Pre-generation readouts predict factual recall and which questions sampling will cover across three model families, but say little about whether an available correct answer will ultimately be selected. Explicit verification with $P(\mathrm{True})$ improves within-question ranking over mean log-likelihood in Gemma, Qwen3, and Llama, with AUROC gains of $0.08$--$0.12$. In a prospectively defined Gemma cohort, verification raises plurality accuracy by about $5$ points, and still gains about $2$ points over chat-template likelihood, a stronger generative baseline. The advantage is strongest for relations with common-answer priors and depends on access to the entity; masking the entity removes the ranking advantage in l
    
[^239]: 多模态大语言模型在跨领域组织学相似性判断中超越病理学基础模型

    Multimodal LLMs Outperform Pathology Foundation Models in Cross-Domain Histological Similarity

    [https://arxiv.org/abs/2609.32876](https://arxiv.org/abs/2609.32876)

    该研究发布MOSAIC基准，揭示病理学基础模型存在将“同机构不同疾病”误判为更相似的临床危险缺陷，而通用多模态大语言模型在跨机构、跨领域的组织学相似性判断中持续表现更优。

    

    arXiv:2609.32876v2 公告类型：replace-cross 摘要：最先进的病理学基础模型虽然在数百万张组织学图像块上进行了训练，但在比较跨越切片或机构边界时，可能无法保持组织相似性。我们证明，通用的多模态大语言模型即使没有作为病理学基础模型进行训练，也能在跨领域组织学相似性判断中持续超越这些专业化模型。我们使用一种相对相似性评估框架，并将其发布为MOSAIC（跨机构与队列的模型相似性评估）基准，在6个数据集上评估了17个模型，发现病理编码器通常将同一机构、不同疾病的图像块判定为比同疾病、不同机构的图像块更相似——这是一种在标准领域内评估中无法察觉的、具有临床危险性的失败模式。大语言模型似乎不易受这种失败的影响，这可能是由于其执行的是对形态学和组织结构的语义视觉比较……

    arXiv:2609.32876v2 Announce Type: replace-cross  Abstract: State-of-the-art pathology foundation models, trained on millions of histology tiles, can fail to preserve tissue similarity when comparisons cross slide or institution boundaries. We show that general-purpose multimodal LLMs, without being trained as pathology foundation models, consistently outperform these specialized models in cross-domain histological similarity judgments. Using a relative similarity framework that we release as the MOSAIC (Model Similarity Assessment across Institutions and Cohorts) benchmark, we evaluate 17 models across 6 datasets and find that pathology encoders often rank same-institution, different-disease tiles as more similar than same-disease, different-institution tiles, a clinically dangerous failure mode invisible to standard within-domain evaluations. LLMs appear less susceptible to this failure, likely because they perform semantic visual comparison of morphology and tissue architecture rathe
    
[^240]: OpenTumorBoard：多学科肿瘤会诊讨论轨迹的真实世界基准

    OpenTumorBoard: A Real-World Benchmark of Multidisciplinary Tumor Board Discussion Trajectories

    [https://arxiv.org/abs/2609.32810](https://arxiv.org/abs/2609.32810)

    该论文构建了首个基于 YouTube 公开真实肿瘤多学科会诊录像的大规模基准 OpenTumorBoard（611 例患者、19,157 轮讨论、10 个专科角色），用于评估大语言模型在专科发言与会诊模拟两种场景下的表现，并发现即使最强的前沿与医学大模型在临床等效性上也与专科医生存在显著差距。

    

    多学科肿瘤会诊通过各专科医生的讨论整合多模态临床观察与患者的纵向病史，然而现有基准很少能够捕捉这些真实世界的讨论轨迹。我们推出了 OpenTumorBoard，这是一个包含 611 个患者病例、涵盖十个专科角色共 19,157 轮讨论的基准数据集，内容转录自 YouTube 上公开的肿瘤会诊录像，总时长达 12,534 分钟。该基准评估两种设置：SPECIALIST TURN（专科发言），即大语言模型回应真实讨论中提出的具有临床意义的问题；BOARD SIMULATION（会诊模拟），即模型生成完整的往复讨论，并就治疗建议、手术方案、下一步行动及临床试验匹配达成共识。对 14 个通用前沿大模型和医学大模型的评估揭示了显著的局限性：表现最好的模型在与专科医生答案的临床等效性上仅得 3.43 分（满分 5 分），在……上得 2.78 分（满分 5 分）。（注：原文摘要在此处截断）

    arXiv:2609.32810v2 Announce Type: replace  Abstract: Multidisciplinary tumor boards integrate multimodal clinical observations and longitudinal patient histories through specialist discussions, yet benchmarks rarely capture these real-world trajectories. We introduce OpenTumorBoard, a benchmark with 611 patient cases and 19,157 discussion turns across ten specialist roles, transcribed from 12,534 minutes of publicly available tumor board recordings on YouTube. The benchmark evaluates two settings: SPECIALIST TURN, in which an LLM responds to a clinically significant question posed during a real discussion, and BOARD SIMULATION, in which it generates an entire back-and-forth discussion and reaches a consensus on therapy recommendations, surgical plans, next actions and clinical trial matching. Evaluation of 14 general-purpose frontier and medical LLMs reveals substantial limitations: the best models score 3.43 out of 5 in clinical equivalence to specialist answers and 2.78 out of 5 in a
    
[^241]: 决策充分的状态表示：测量与减少写入时遗憾

    Decision-Sufficient State Representations: Measuring and Reducing Write-Time Regret

    [https://arxiv.org/abs/2609.32805](https://arxiv.org/abs/2609.32805)

    该论文提出将读取者的损失分解为“预算损失”和“写入时遗憾”的度量框架，发现在TextWorld烹饪游戏中几乎所有损失都来自写入时遗憾——持有事实的128-token状态几乎全胜，而提示词驱动的语言模型写入者最多仅赢17%，表明写入状态的质量是核心瓶颈。

    

    长任务产生的历史信息超出LLM智能体在上下文中能容纳的范围，即使历史信息能装下，智能体也无法可靠地使用全部内容。因此，一条不断发展的研究路线是让智能体携带一份简短的书面状态：在每一步由写入者重写状态，读取者仅依据状态行动。这使每一步保持低成本，但写入者丢弃的任何信息都会丢失，而后续的决策可能会揭示这些信息本是被需要的。我们对这种损失进行了量化，并探讨训练能否减少这种损失。通过将写入的状态与事后看来同等大小的最佳状态进行比较，我们将读取者的损失分解为预算损失（任何该大小的状态都不可避免要承担的损失）和写入时遗憾（源于写入者选择的损失）。在TextWorld烹饪游戏中，我们控制一条事实在被需要之前必须被携带的时长，一个持有相关事实的128-token状态几乎赢得每一局游戏，而使用提示词驱动的语言模型写入者最多只能赢得17%。几乎所有的损失都是写入时遗憾（注：原摘要在此处截断）。

    arXiv:2609.32805v2 Announce Type: replace  Abstract: Long tasks produce more history than an LLM agent can hold in its context, and more than it uses reliably even when the history fits. A growing line of work therefore has agents carry a short written state instead: at every step a writer rewrites the state, and a reader acts from the state alone. Steps stay cheap, but anything the writer drops is lost before later decisions reveal that they need it. We quantify this loss and ask whether training can reduce it. Comparing the written state with the best state of the same size written in hindsight, we split the reader's loss into a budget loss, which any state of that size must incur, and a write-time regret, which comes from the writer's choices. In TextWorld cooking games where we control how long a fact must be carried before it is needed, a 128-token state holding the facts wins nearly every game, while prompted language-model writers win at most 17%. Almost all of the loss is write
    
[^242]: 面向长时程智能体的自适应一致性图

    Adaptive Consistency Graph for Long-Horizon Agents

    [https://arxiv.org/abs/2609.32754](https://arxiv.org/abs/2609.32754)

    提出自适应一致性图（ACG），通过在持久图中增量组织执行证据及来源，并为每次决策构建可追溯的、以需求为中心的上下文视图，在不改动智能体规划器与工具执行器的前提下显著提升长时程任务成功率。

    

    大语言模型智能体在短任务上通常能做出合理的局部决策，但当成功需要长序列相互依赖的动作和工具调用时，其性能会下降。在执行过程中，任务需求、历史证据与当前执行状态可能逐渐脱节，导致后续决策偏离原始目标。我们通过引入面向长时程执行的自适应一致性图（ACG）来研究这一问题。ACG以增量方式在持久图中组织执行证据及其来源信息，并在有限的上下文预算下为每个决策构建一个临时的、以需求为中心的视图。ACG并不取代基础智能体的规划器或工具执行器，而是为每个决策提供结构化且可追溯的上下文视图。在匹配的评估中，ACG将GPT-5.6-luna的平均成功率从使用ReAct时的44.5%提升至50.2%，其中在BrowseComp上提升最大。

    arXiv:2609.32754v2 Announce Type: replace  Abstract: Large language model agents can often make reasonable local decisions on short tasks, yet their performance degrades when success requires long sequences of dependent actions and tool calls. During execution, task requirements, historical evidence, and the current execution state may gradually become disconnected, so later decisions can drift from the original objective. We study this problem by introducing the Adaptive Consistency Graph (ACG) for long-horizon execution. ACG incrementally organizes execution evidence and its provenance in a persistent graph, then constructs a temporary requirement-centered view for each decision under a bounded context budget. Rather than replacing the base agent's planner or tool executor, ACG provides a structured and traceable context view for each decision. In the matched evaluation, ACG improves GPT-5.6-luna's average success from 44.5\% with ReAct to 50.2\%, with the largest gain on BrowseComp-
    
[^243]: AdaTutoRank：通过自适应辅导优化学习文档集重排序，面向RAG与深度研究

    AdaTutoRank: Learning to Rerank Document Sets via Adaptive Tutoring Optimization for RAG and Deep Research

    [https://arxiv.org/abs/2609.32472](https://arxiv.org/abs/2609.32472)

    该论文提出AdaTutoRank，针对文档集重排序中集合级标量奖励导致的监督稀疏与信用分配困难问题，通过自适应辅导优化为不同质量的rollout提供差异化指导，从而为RAG和深度研究筛选出完整、互补、无冗余的文档集合。

    

    在RAG和深度研究中，文档重排序器决定了哪些证据会传递给下游模型，然而主流的重排序器通过相关性匹配进行选择，而单独相关的文档很少能构成复杂信息需求所需的完整、互补且无冗余的集合。先前的工作通过集合的总体评分（rubric score）来奖励整个集合，将目标从文档排序转移到了集合组合。然而该分数是集合中所有文档共享的一个标量，因此监督信号是稀疏的：只要集合得分高，冗余文档就会与其余文档一同获得奖励；而只要集合得分低，起决定性作用的文档也会与其余文档一同受到惩罚；这种信用分配机制使得真正的贡献者与搭便车者无法被区分开来。在策略蒸馏（on-policy distillation）可以加密这种监督信号，但现有方法对每一次 rollout 都给予相同的固定指导，对强的 rollout 而言过于死板，对弱的 rollout 而言又过于抽象。因此，我们提出了 AdaTutoRank，一种（摘要在此处截断）

    arXiv:2609.32472v2 Announce Type: replace  Abstract: Document rerankers determine what evidence reaches the downstream model in RAG and deep research, yet mainstream rerankers select by relevance matching, and individually relevant documents rarely constitute the complete, complementary, non-redundant set a complex information need demands. Prior work rewards a set by its aggregate rubric score, shifting the objective from ranking documents to composing sets. Yet that score is one scalar shared by every document in the set, so the supervision is sparse: a redundant document is rewarded with the rest whenever the set scores well, and a decisive one penalized with the rest whenever it does not; credit assignment leaves contributors indistinguishable from free riders. On-policy distillation could densify this supervision, but existing methods give every rollout the same fixed guidance, too prescriptive for strong rollouts and too abstract for weak ones. We therefore propose AdaTutoRank, a
    
[^244]: SEA-CLIP-Tiny：面向东南亚语言的高效多语言文本-视觉嵌入模型

    SEA-CLIP-Tiny: Efficient Multilingual Text-Vision Embedding for Southeast Asian Languages

    [https://arxiv.org/abs/2609.30739](https://arxiv.org/abs/2609.30739)

    本文提出SEA-CLIP-Tiny，一个参数量不足5000万的紧凑型多语言文本-视觉嵌入模型，通过区域数据筛选和多语言教师引导，在七种东南亚语言的跨语言图像-文本检索上取得最优平均性能，且比MobileCLIP2参数更少、CPU延迟更低。

    

    多语言文本-视觉嵌入模型对于跨语言图像-文本检索至关重要，但由于该地区语言多样性以及数据和计算资源的限制，东南亚语言的支持仍然不足。在本文中，我们提出了SEA-CLIP-Tiny，一个面向东南亚的紧凑型多语言文本-视觉嵌入模型，参数量少于5000万。我们的模型通过区域数据筛选和多语言教师模型引导，将CLIP-KD风格的框架适配到东南亚多语言环境中。在七种东南亚语言上的实验表明，SEA-CLIP-Tiny在所评估的学生模型中取得了最强的平均检索性能，在R@1、R@5和R@10上分别达到12.9%、31.5%和42.2%。与MobileCLIP2相比，它在参数量减少38.4%、CPU实测延迟更低的情况下，将平均R@10提升了12.1个百分点。这些结果凸显了区域感知……（原文摘要至此截断）

    arXiv:2609.30739v1 Announce Type: new  Abstract: Multilingual text-vision embedding models are essential for cross-lingual image-text retrieval, but Southeast Asian languages remain poorly supported due to the region's linguistic diversity and limited data and computing resources. In this paper, we introduce SEA-CLIP-Tiny, a compact multilingual text-vision embedding model for Southeast Asia with fewer than 50M parameters. Our model adapts a CLIP-KD-style framework to Southeast Asian multilingual settings through regional data curation and multilingual teacher guidance. Experiments across seven Southeast Asian languages show that SEA-CLIP-Tiny achieves the strongest average retrieval performance among the evaluated student models, reaching 12.9%, 31.5%, and 42.2% at R@1, R@5, and R@10, respectively. Compared with MobileCLIP2, it improves average R@10 by 12.1 points while using 38.4% fewer parameters and lower measured CPU latency. These results highlight the importance of region-aware 
    
[^245]: RAZOR：在大语言模型中剪枝可被替换的专家

    RAZOR: Pruning Replaceable Experts in LLMs

    [https://arxiv.org/abs/2609.30465](https://arxiv.org/abs/2609.30465)

    该论文提出无需训练的 MoE 专家剪枝方法 RAZOR，利用共识残差衡量专家的功能可替换性，在固定剪枝预算下剪除可被存活专家替代的专家，无需梯度或恢复训练即可最大程度保留原始模型输出分布。

    

    混合专家模型每个 token 只激活少量专家，但需要存储完整的专家池。专家剪枝可以减轻这种存储负担；在固定剪枝预算下，目标是尽可能保留原始模型的输出分布。然而，专家的使用频率或贡献大小本身并不能决定移除它所造成的损害，关键在于存活的计算能否替代其功能。我们提出 RAZOR，这是一种无需训练的专家剪枝方法，通过“共识残差”（即专家输出与原始加权混合输出的偏差）来评分专家的功能可替换性。该方法在固定层输入处使用精确的单删除恒等式，考虑了存活专家的重新归一化以及路由器选择的补充机制，从而在无需梯度或恢复训练的情况下，将在校准 token 上聚合得到的局部评分用于预算化剪枝。在 GLM-4.7-Flash、Qwen3.6-35B-A3B、DeepSeek-V4-Flash-0731 和 Hy3 上，在 25% 的（剪枝率下……摘要在此处截断）

    arXiv:2609.30465v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) models activate few experts per token but store the full expert pool. Expert pruning reduces this storage burden; at a fixed pruning budget, the goal is to preserve the original model's output distribution as closely as possible. Yet an expert's usage or contribution magnitude does not by itself determine the damage caused by its removal. What matters is whether the surviving computation can replace its function. We introduce RAZOR, a training-free expert pruning method that scores functional replaceability using consensus residuals: deviations of expert outputs from the original weighted mixture. An exact single-deletion identity at a fixed layer input accounts for survivor renormalization and router-selected refill, providing local scores aggregated over calibration tokens for budgeted pruning without gradients or recovery training. On GLM-4.7-Flash, Qwen3.6-35B-A3B, DeepSeek-V4-Flash-0731, and Hy3 at 25\% an
    
[^246]: JEV与LLM作为评分准则裁判：更便宜、更快速，且在同样的地方犯错

    JEV vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places

    [https://arxiv.org/abs/2609.29769](https://arxiv.org/abs/2609.29769)

    研究表明，无需生成文本的类型化分类器Jev可作为LLM评分准则裁判的低成本替代方案：准确率与LLM裁判无显著差异但成本仅为其1/29至1/325，且两者在分级准则上倾向于犯相似错误（均低于人工评分等级）。

    

    我们探究Jev——一个无需生成文本、直接返回各允许答案概率的类型化分类器——能否取代LLM评分准则裁判。我们在来自七个基准的九个面板上，将其与三个flash级LLM裁判进行对比，并为每个裁判提供完全相同的准则文本。在27组配对比较中，Jev的准确率仅在8组上与LLM裁判存在显著差异：其优势主要体现在二元准则上，仅在分级准则上落后，其余多数比较结果不明确。对九个面板累计计算，每条准则调用一次的LLM裁判成本是Jev的29至325倍，耗时是Jev的30至220倍。在分级准则上，四个裁判彼此之间的一致性都高于其与真实标签的一致性，且大多给出低于人工评分者的等级。一种可能的观察性解释是：人工评分者遵循了我们的准则文本中未写明的量表使用惯例。Jev的置信度在大多数面板上能够对其自身错误进行排序，这应该能使更便宜的错误筛选成为可能。

    arXiv:2609.29769v1 Announce Type: new  Abstract: We ask whether Jev, a typed classifier that returns probabilities over permitted answers without generating text, can replace an LLM rubric judge. We compare it with three flash-tier LLM judges on nine panels drawn from seven benchmarks, giving every judge identical criterion texts. Jev's accuracy differs significantly from an LLM judge's in only 8 of 27 paired comparisons, ahead mostly on binary criteria and behind only on graded ones, and most of the other comparisons are inconclusive. Summed over the nine panels, the LLM judges, called once per criterion, cost 29 to 325 times as much as Jev and took 30 to 220 times as long. On graded criteria all four judges agree more with one another than with the labels and mostly assign lower levels than the raters. One of several observational accounts is that raters followed scale conventions our criterion texts omit. Jev's confidence ranks its own errors on most panels, which should make a chea
    
[^247]: 冰岛手语的孤立手语识别：低资源环境下的实验

    Isolated Sign Language Recognition for Icelandic Sign Language: Experiments in a Low-resource Setting

    [https://arxiv.org/abs/2609.25862](https://arxiv.org/abs/2609.25862)

    该论文首次针对极低资源的冰岛手语开展孤立手语识别实验，发现跨语言迁移（先在美国手语数据上预训练再微调）能带来最大收益，将准确率提升 14-24 个百分点。

    

    我们展示了针对冰岛手语（\'ITM）的孤立手语识别（ISLR）的首批实验。我们使用 \'ITM SignWiki 数据集，该数据集源自冰岛语-\'ITM 双语在线词典。它是真正的低资源数据集：1,845 个视频涵盖 849 个类别，其中 86% 的类别仅有两个样本，使得完整任务实际上成为跨手语者的单样本识别。我们在三个词汇量递增的任务（22、117 和 849 个类别）上比较了两个开源 ISLR 框架 OpenHands 和 SPOTER，并评估了三种姿态估计器和两种跨语言迁移方法。仅使用 \'ITM 数据时，SPOTER 在所有三个任务上都优于 OpenHands，且 MediaPipe 姿态估计的结果优于 AlphaPose 或 SDPose。跨语言迁移带来了最大的收益：在美国手语数据上预训练 SPOTER 后再在 \'ITM 上微调，可将准确率提高 14-24 个百分点，在三个任务上分别达到 72.7%、47.9% 和 22.6%，并且

    arXiv:2609.25862v1 Announce Type: new  Abstract: We present the first experiments on isolated sign language recognition (ISLR) for Icelandic Sign Language (\'ITM). We use \'ITM SignWiki, a dataset derived from a bilingual Icelandic--\'ITM online dictionary. It is genuinely low-resource: 1,845 videos cover 849 classes, 86% of which have only two examples, making the full task effectively one-shot recognition across signers. We compare two open-source ISLR frameworks, OpenHands and SPOTER, on three tasks of increasing vocabulary size (22, 117 and 849 classes), and evaluate three pose estimators and two forms of cross-lingual transfer. With \'ITM data alone, SPOTER outperforms OpenHands on all three tasks, and MediaPipe poses give better results than AlphaPose or SDPose. Cross-lingual transfer brings the largest gains: pretraining SPOTER on American Sign Language data before finetuning on \'ITM raises accuracy by 14--24 percentage points, to 72.7%, 47.9% and 22.6% on the three tasks, and 
    
[^248]: 复制天花板：面向策展语料库本体接地生成的输入暴露控制

    The Copy Ceiling: An Input-Exposure Control for Ontology-Grounded Generation over Curated Corpora

    [https://arxiv.org/abs/2609.24885](https://arxiv.org/abs/2609.24885)

    论文提出“暴露核算”与“复制天花板”这一无需评判者的评估控制方法，揭示语言模型在基于本体检索的接地生成中的性能提升几乎完全来自复制上下文中已暴露的答案，而非对检索结构的真正推理。

    

    当语言模型通过基于图的检索从策展语料库中回答问题时，接地带来的大幅性能提升并不能证明模型对检索到的结构进行了推理：因为所提供的上下文可能已经暴露了标准答案。我们提出“暴露核算”方法，根据所展示的上下文是否暴露了每个标准答案项以及答案是否恢复了该项来对其进行分类。其标量参照是“复制天花板”，即对上下文进行逐字复制所能达到的召回率；相对于复制的带符号增益衡量了模型召回率相对于这一确定性、无需评判者的基线的表现。在十个模型上，无辅助召回率平均为0.26，接地召回率为0.92，但相对于复制的增益一致为负（-0.067至-0.022）。在11,360个标准答案项观测中（代表在十个模型下评估的1,136个目标实例），仅有三个未暴露的项获得了词汇层面的评分。对423个观测进行的分层模型评判审计（采用对称引用验证策略）估计，97%……（原文摘要在此处截断）

    arXiv:2609.24885v1 Announce Type: new  Abstract: When a language model answers from a curated corpus via graph-based retrieval, a large grounding uplift does not establish reasoning over the retrieved structure: the context may already expose the gold answers. We propose exposure accounting, which classifies each gold item by whether the shown context exposes it and whether the answer recovers it. Its scalar reference is the copy ceiling, the recall a verbatim copy of the context achieves; signed gain over copy measures the model's recall relative to this deterministic, judge-free baseline. Across ten models, unaided recall averages 0.26 and grounded recall 0.92, yet gain over copy is uniformly negative (-0.067 to -0.022). Of 11,360 gold-item observations, representing 1,136 target instances evaluated under ten models, only three unexposed items receive lexical credit. A stratified model-judged audit of 423 observations, with a symmetric quotation-verification policy, estimates that 97
    
[^249]: 弃权与噪声过滤：Softmax注意力缺失的两个基本要素

    Abstention and Noise Filtering: Two Missing Primitives of Softmax Attention

    [https://arxiv.org/abs/2609.22005](https://arxiv.org/abs/2609.22005)

    本文论证并通过实验证明，注意力值通路的门控为softmax注意力补充了两种缺失的基本要素——弃权（允许注意力头输出空值）和噪声过滤（抑制残差流中叠加特征的干扰），并在10M至350M参数的匹配模型上提供了实证支持。

    

    据报道，对注意力的值通路进行门控可以改善语言模型的预训练效果，但此前的研究对其原因看法不一。我们提出论点并提供实验证据，表明这类门控为softmax注意力提供了两种它所缺乏的不同能力：弃权与噪声过滤。第一种是弃权，它允许注意力头输出空值，从而绕过注意力权重之和必须为一的约束。第二种是噪声过滤，它允许注意力头的值通路抑制残差流中叠加特征带来的干扰。在我们对从10M到350M参数的匹配模型进行的实验中，我们通过在softmax中引入一个可学习的每头汇点logit来实现弃权，通过对每个值进行门控来实现噪声过滤。我们报告了三项实证发现。第一，弃权的收益（以相对于匹配基线的验证损失下降来衡量）随着模型规模的增大而下降，而……的收益（摘要在此处被截断）

    arXiv:2609.22005v1 Announce Type: cross  Abstract: Gating the value pathway of attention reportedly improves language model pretraining, and prior studies disagree on why. We argue and provide experimental evidence that such gates supply two different things that softmax attention lacks: abstention and noise filtering. The first is abstention, which allows an attention head to output nothing, bypassing the requirement that attention weights must sum to one. The second is noise filtering, which allows the value pathway of an attention head to suppress interference from superposed features in the residual stream. In our experiments in matched models from 10M to 350M parameters, we supply abstention through a learned per-head sink logit in the softmax and noise filtering through a gate on each value. We report three empirical findings. First, the benefit of abstention, measured as the reduction in validation loss relative to a matched baseline, declines as models grow, whereas the benefit
    
[^250]: Uni-LaDiR：潜在扩散统一多模态推理

    Uni-LaDiR: Latent Diffusion Unifies Multimodal Reasoning

    [https://arxiv.org/abs/2609.19878](https://arxiv.org/abs/2609.19878)

    Uni-LaDiR提出将多模态推理步骤映射到共享潜在空间，并利用扩散模型预测下一块思维标记，从而在统一的潜在表示中实现灵活的多模态推理。

    

    多模态推理要求模型在整个推理过程中利用来自多种模态的信息。然而，现有方法通常将特定模态的思维标记拼接在单一序列中，使得模型在跨模态推理时需要自行弥合表示上的差异。我们提出了Uni-LaDiR（统一潜在扩散推理器），这是一个将这些思维引入共享潜在空间进行推理的框架。统一编码器将来自不同模态的教师推理步骤映射为共享的思维标记，并通过训练来保留后续推理步骤及最终答案或动作所需的信息。由于相同的上下文可以支持多个有效的下一步推理，我们使用扩散模型基于输入和先前块来预测下一块思维标记。通过共享模型权重联合训练编码器和扩散推理器，促使思维标记既对任务有用，又……

    arXiv:2609.19878v1 Announce Type: cross  Abstract: Multimodal reasoning requires models to draw on information from multiple modalities throughout the reasoning process. Yet existing methods often concatenate modality-specific thought tokens in a single sequence, leaving the model to bridge representational differences as it reasons across modalities. We introduce Uni-LaDiR (Unified Latent Diffusion Reasoner), a framework that brings these thoughts into a shared latent space for reasoning. A unified encoder maps teacher reasoning steps from different modalities into shared thought tokens, trained to preserve the information needed for later reasoning steps and the final answer or action. Because the same context can support multiple valid next steps, we use diffusion to predict the next block of thought tokens from the input and preceding blocks. Jointly training the encoder and diffusion reasoner with shared model weights encourages thought tokens to be both useful for the task and pr
    
[^251]: 可解码却误路由：稀疏特征揭示视觉-语言模型在有害模因检测中的读取差距

    Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection

    [https://arxiv.org/abs/2609.18860](https://arxiv.org/abs/2609.18860)

    研究发现大型视觉-语言模型内部已编码了检测有害模因所需的证据信息，但无法将其正确路由至输出端——通过稀疏自编码器读取的稀疏特征在六个有害内容基准上均显著优于模型原生预测，揭示了模型存在“可解码但误路由”的读取差距。

    

    当大型视觉-语言模型错误分类有害模因时，这种失败可能反映了内部证据的缺失，或者是无法将已表征的证据正确路由到输出端。我们在Gemma-3和Qwen3.5模型中利用稀疏自编码器、角色条件探针、因果干预和恢复实验来区分这两种情况，并在六个有害内容基准上进行评估，还开展了西班牙语和印地语-英语混合语的补充评估。稀疏读取在所有六个主要二分类任务上都优于模型原生预测：Qwen的稀疏读取平均宏F1达到0.740，而原生预测仅为0.432，残差重建达到0.486；Gemma则从0.532提升至0.714。这些差异反映的是监督可访问性，而非模型中预先存在的原生决策规则，且最具影响力的词元角色取决于具体任务。在所评估的分数尺度下，Qwen的静默特征消融对探针的敏感度高出24-63倍，而路由特征修补……

    arXiv:2609.18860v1 Announce Type: cross  Abstract: When a large vision-language model misclassifies a harmful meme, the failure may reflect missing internal evidence or an inability to route represented evidence to its output. We distinguish these cases in Gemma-3 and Qwen3.5 using sparse autoencoders, role-conditioned probes, causal interventions, and recovery experiments across six harmful content benchmarks, with additional Spanish and Hindi-English code-mixed evaluations. Sparse readouts outperform native prediction on all six primary binary tasks: Qwen averages $0.740$ versus $0.432$ native macro-F1, while residual reconstruction reaches $0.486$, whereas Gemma improves from $0.532$ to $0.714$. These differences reflect supervised accessibility rather than a pre-existing, native decision rule, and the most influential token role depends on the task. Under the evaluated score scales, Qwen silent-feature ablation is $24-63$ times more probe-sensitive, whereas routed-feature patching 
    
[^252]: 从像素到配对：噪声文档环境下基于大语言模型的键值对提取综合基准测试

    From Pixels to Pairs: A Comprehensive Benchmark of LLM-Based Key-Value Extraction in Noisy Document Settings

    [https://arxiv.org/abs/2609.17538](https://arxiv.org/abs/2609.17538)

    该论文建立了一个系统性基准，评估开源指令微调大语言模型在干净文本与含噪OCR条件下提取键值对的能力，发现现代LLM在高质量文本输入下是强大的语义提取器（部分情况接近有监督布局感知系统），但在OCR噪声下性能显著下降。

    

    大语言模型（LLMs）越来越多地被用于从文档中提取结构化信息，但它们在真实OCR噪声下的行为仍然知之甚少。我们提出了一个系统性基准测试，评估开源指令微调大语言模型在干净文本和含噪OCR条件下进行键值对（KVP）提取的表现。我们在FUNSD、CORD和SROIE基准数据集上评估了代表性的仅解码器模型（Gemma、Mistral、Qwen2.5、LLaMA 3和DeepSeek），使用金标准文本标注以及来自PaddleOCR、EasyOCR和Tesseract的OCR输出。统一的评估协议在一致的条件下分离了输入质量、模型设计和提示方式的影响。结果表明，当有高质量文本可用时，现代大语言模型表现出强大的语义提取能力，在某些情况下接近有监督的布局感知系统。然而，在OCR噪声下，性能显著下降，模型之间的性能差距……

    arXiv:2609.17538v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for structured information extraction from documents, yet their behavior under realistic OCR noise remains poorly understood. We present a systematic benchmark of open-source instruction-tuned LLMs for key-value pair (KVP) extraction under both clean-text and noisy OCR conditions.   We evaluate representative decoder-only models (Gemma, Mistral, Qwen2.5, LLaMA 3, and DeepSeek) on the FUNSD, CORD, and SROIE benchmarks using both Gold-text annotations and OCR outputs from PaddleOCR, EasyOCR, and Tesseract. A unified evaluation protocol isolates the effects of input quality, model design, and prompting under consistent conditions.   The results show that modern LLMs act as strong semantic extractors when high-quality text is available, in some cases approaching supervised layout-aware systems. Under OCR noise, however, performance degrades substantially and performance gaps between models n
    
[^253]: IROH：基于多阶段混合检索与原理蒸馏LLM裁判的幽默洞察排序——面向JOKER 2026赛道任务1（英语）

    IROH: Insightful Ranking Of Humor using Multi-Stage Hybrid Retrieval with Rationale-Distilled LLM Judges for JOKER 2026 Track Task 1 English

    [https://arxiv.org/abs/2609.15618](https://arxiv.org/abs/2609.15618)

    该论文提出IROH三阶段检索系统，通过稀疏-稠密混合检索、交叉编码器重排序与原理蒸馏的LoRA大语言模型裁判集成，在JOKER 2026任务1幽默排序中夺得第一名（MAP 0.6347），并发现原理蒸馏裁判是排序质量的关键驱动因素。

    

    我们的团队VANGUARD提出了IROH（Insightful Ranking of Humor，幽默洞察排序），这是一个面向CLEF 2026 JOKER任务1（英语）的三阶段检索系统，以0.6347的MAP成绩位列排行榜第一名。我们的处理流程结合了稀疏-稠密混合检索、交叉编码器重排序，以及经LoRA适配的大语言模型裁判集成。我们使用Gemma 4在两种提示策略（通用型和类型化）下生成查询感知的原理说明，并生成多达四种类型的结构化困难负例用于训练数据构建。通过对三种交叉编码器架构、四种稠密嵌入模型和八种裁判配置的消融实验，我们得到三项关键发现：（1）原理蒸馏的裁判是排序质量的主要驱动因素，而将原理说明附加到第一阶段索引中的贡献微乎其微；（2）结构化困难负例虽然在本地验证集上会抬高分数，但在几乎所有配置中都会损害泛化能力；以及……

    arXiv:2609.15618v1 Announce Type: cross  Abstract: Our team, VANGUARD, presents IROH (Insightful Ranking of Humor), a three-stage retrieval system for JOKER Task 1 English at CLEF 2026, achieving first place on the leaderboard with 0.6347 MAP. Our pipeline combines hybrid sparse-dense retrieval, cross-encoder reranking, and a LoRA-adapted Large Language Model judge ensemble. We employ Gemma 4 to generate query-aware rationales under two prompt strategies, generic and typed, and produce up to four types of structured hard negatives for training data construction. Through an ablation across three cross-encoder architectures, four dense embedders, and eight judge configurations, our key findings are threefold: (1) the rationale-distilled judge is the primary driver of ranking quality, whereas appending rationales to the first-stage index contributes negligibly; (2) structured hard negatives degrade generalisation in nearly all configurations despite inflating local validation scores; and 
    
[^254]: 当错误的钥匙获胜：理解与检测大语言模型中的幻觉

    When the Wrong Key Wins: Understanding and Detecting Hallucinations in LLMs

    [https://arxiv.org/abs/2609.15106](https://arxiv.org/abs/2609.15106)

    大语言模型的幻觉源于预训练关联之间“潜在钥匙”的竞争，本文据此提出一种两阶段关键词扰动检测方法，通过移除关键词条并观察预测重组方式，区分误导性关联导致的错误与有据可依的正确答案。

    

    大语言模型即使在其给出正确答案所需的知识已经可获得的情况下，仍可能产生幻觉。我们通过推理的“潜在钥匙”视角来研究这种失败现象，即答案的选择取决于预训练期间习得的各关联之间的竞争。我们证明，模型预测可能对查询单个关键词高度敏感，这些具有影响力的关键词表现出针对特定实体的绑定特性，且其影响受预训练频率的系统性塑造。多个绑定还可以在同一查询内相互竞争并表现出高阶交互作用。基于这一机制，我们提出了一种用于幻觉检测的两阶段关键词扰动方法。通过移除有影响力的关键词并测量模型如何重组其预测，该方法能够将由误导性关键关联导致的错误与由诊断性证据支持的正确决策区分开来。在多个模型和基准测试上的结果……

    arXiv:2609.15106v1 Announce Type: new  Abstract: Large language models can hallucinate even when the knowledge required for a correct answer is already available. We study this failure through a latent-key view of inference, where answer selection depends on competition among associations acquired during pretraining. We show that model predictions can be highly sensitive to individual query keywords, that these influential keywords exhibit entity-specific binding, and that their effects are systematically shaped by pretraining frequency. Multiple bindings can also compete and exhibit higher-order interactions within the same query. Based on this mechanism, we introduce a two-stage keyword-perturbation method for hallucination detection. By removing influential keywords and measuring how the model reorganizes its prediction, the method distinguishes errors caused by misleading key associations from correct decisions supported by diagnostic evidence. Across multiple models and benchmarks
    
[^255]: 智能体作为策略的机器人操作

    Agent as Policy for Robotic Manipulation

    [https://arxiv.org/abs/2609.12541](https://arxiv.org/abs/2609.12541)

    该论文提出Agent as Policy (AGP)方法，使通用智能体无需任何任务特定训练即可直接控制物理机器人完成从精细操作、动态运动到可变形物体处理等多种真实操作任务。

    

    我们证明了通用智能体可以在整个任务执行过程中直接驱动物理机器人，而无需任何针对特定任务或特定环境的训练。我们提出了智能体作为策略方法，将任务规划和执行都置于智能体的控制之下。给定一个任务和机器人接口，智能体能够解读视觉信息、编写可执行程序、发出运动指令，并根据物理执行结果修正自身动作。这将智能体的推理和编程能力带入了与物理世界的持续交互之中。我们在多个真实世界的操作任务上研究了AGP，涵盖精细操作、动态运动和可变形物体，包括从人类视频中学习装配、根据目标图像搭建积木、翻转骰子、定向投掷以及双臂协作折叠毛巾。AGP在三种积木搭建配置上分别取得了100%、100%和80%的成功率。

    arXiv:2609.12541v1 Announce Type: new  Abstract: We demonstrate that a general-purpose agent can directly drive a physical robot throughout task execution without any task-specific or environment-specific training. We introduce Agent as Policy (AGP), which places task planning and execution under the agent's control. Given a task and a robot interface, the agent interprets visual evidence, writes executable programs, issues motion commands, and revises its actions in response to physical outcomes. This brings the agent's reasoning and programming capabilities into continuous interaction with the physical world. We study AGP across multiple real-world manipulation tasks spanning precision manipulation, dynamic motions, and deformable objects. These include assembly from human videos, block construction from goal images, die reorientation, targeted throwing, and bimanual towel folding. AGP achieves success rates of 100%, 100%, and 80% on three block construction configurations. These fin
    
[^256]: 递归语言模型训练的脆弱性谱系

    A Fragility Spectrum for Recursive Language-Model Training

    [https://arxiv.org/abs/2609.11149](https://arxiv.org/abs/2609.11149)

    该研究让13个公开模型在固定递归污染协议下共享语料库繁衍五代，发现不同模型对坍塌的脆弱性存在约五倍差异，且该脆弱性排序在不同数据组成和随机种子下高度稳定，表明易坍塌性是模型本身的固有属性。

    

    模型生成的文本正在回流到训练语料库中，大量证据表明，反复使用这类数据进行训练会导致输出多样性的坍塌。先前的工作研究了这一现象本身：哪些训练协议以及哪些数据混合方式会引发坍塌。但在相同的过程下，不同模型的表现差异巨大。我们固定了一种递归污染协议，让13个公开发布的模型检查点组成一个生态系统，共享同一语料库并繁衍五代。五代之后，各检查点的唯一4-gram指标从0.187到0.940不等，约有五倍的差距：一些模型几乎未受影响，另一些则退化为重复的片段。改变共享语料池的组成或混入人类文本时，模型排序的Spearman相关性保持在0.91–0.97；改变随机种子时，该相关性保持在0.93–0.98。因此，模型在递归训练下是否容易坍塌，是模型本身固有的一种属性。

    arXiv:2609.11149v1 Announce Type: new  Abstract: Model-generated text is finding its way back into training corpora, and there is plenty of evidence that training on such data over and over collapses output diversity. Prior work has studied the phenomenon itself: which protocols and which data mixtures cause collapse. But different models behave very differently under the same process. We fix one recursive contamination protocol and let 13 publicly released checkpoints form an ecosystem that shares a common corpus for five generations. The unique 4-gram outcome after five generations ranges from 0.187 to 0.940 across checkpoints, a roughly five-fold spread: some models are barely touched, others degenerate into repetitive fragments. Changing the composition of the shared pool or mixing in human text keeps the Spearman correlation of the ordering at 0.91--0.97, and changing the random seed keeps it at 0.93--0.98. Whether a model collapses easily under recursive training is, then, a prop
    
[^257]: 寡头在多模型生态系统中几乎无法左右模型崩溃

    The Oligarch Barely Steers Model Collapse in Multi-Model Ecosystems

    [https://arxiv.org/abs/2609.11146](https://arxiv.org/abs/2609.11146)

    在多模型递归训练的生态系统中，即使将寡头模型的市场份额推高至90%，既不会加速模型崩溃，也不会使其他模型被拖向寡头的输出分布——模型崩溃的动态对市场份额集中度表现出不变性。

    

    AI生成的文本正在回流到下一代模型的训练语料库中，对其实施的递归训练会导致模型崩溃。近期工作将这一研究场景扩展到多个模型相互喂食的情况——但几乎所有研究都假设市场份额均分，而现实中的生成式AI行业实为寡头垄断格局。市场集中度引发了两个担忧：更少、更同质化的来源可能加速模型崩溃，且后续的模型可能被拖向寡头模型的输出分布。我们在受控生态系统中对这两点进行了检验：13个开源的1—4B参数模型构成了包含3到13个参与者的自然生态系统，另加入一个注入的探针模型，将头部模型的市场份额推高至90%；在每一代中，所有模型的输出按各自市场份额混入共享数据池，每个模型都从干净的初始基础权重出发在该共享池上重新训练，共进行五代。然而在我们测试的范围内，这两个担忧均未成真；相反，实验呈现出一种不变性：使市场份额分配更加不均，几乎不会改变模型崩溃的速度。

    arXiv:2609.11146v1 Announce Type: cross  Abstract: AI-generated text is flowing back into the training corpora of the next generation of models. Recursive training on it drives model collapse, and recent work extends the setting to many models feeding one another -- but almost always with the market split evenly, while real generative AI is an oligopoly. Concentration raises two worries: fewer, more uniform sources may make collapse faster, and later models may be dragged toward the oligarch's output. We test both in controlled ecosystems: 13 open 1--4B models form natural ecosystems of 3 to 13 players, plus an injected probe that pushes the top share to 90%; each generation, every model's output is mixed into a shared pool by market share and every model is retrained on that pool from clean base weights, for five generations. Yet within the range we test, neither worry materializes; what emerges instead is an invariance. Making the split more unequal barely changes the speed of collap
    
[^258]: 终极翻译基准测试

    Last Translation Benchmark

    [https://arxiv.org/abs/2609.04173](https://arxiv.org/abs/2609.04173)

    提出了终极翻译基准测试，这是一个包含人工编写、经同行评审的多模态示例的基准数据集，通过为每个示例配备手工编写的验证规则来描述具体失败案例，解决了现有机器翻译基准趋于饱和且评估方法不可靠的问题。

    

    为了推动科学进步，我们需要能够测试最先进模型极限的基准测试，以及能够揭示失败案例的评估方法。随着模型日益强大，机器翻译的标准基准测试正趋于饱和。此外，自动翻译指标不可靠、容易受到奖励破解的攻击，且提供的评估缺乏可操作性。即使是黄金标准的人工评估也并非完美，因为它通常缺乏可重复性、客观性和可扩展性。总体而言，这阻碍了我们追踪该领域的客观进展并确定改进路径。我们提出了终极翻译基准测试，这是一个由人工编写并经过同行评审的示例（文本、图像、音频、视频）集合，这些示例能够难倒领先的机器翻译模型。我们还提出了一种新的评估方法：每个示例都附有手工编写的验证规则，用于描述该示例上的具体失败案例，因此能够……

    arXiv:2609.04173v1 Announce Type: new  Abstract: For scientific progress, we need benchmarks that test the limits of state-of-the-art models, and evaluation methods that inform us about failure cases. As models get stronger, standard benchmarks for machine translation are approaching saturation. Further, automatic translation metrics are unreliable, vulnerable to reward-hacking, and provide unactionable assessments. Even gold human evaluation is not problem-free, because it often lacks reproducibility, objectivity, and scalability. Overall, this prevents us from tracking objective progress in the field and identifying pathways for improvement. We introduce the Last Translation Benchmark, a collection of human-authored and peer-reviewed examples (texts, images, audio, videos) that break leading machine translation models. We also present a new evaluation approach: each example comes with handcrafted verification rules describing concrete failure cases on that example, therefore allowing
    
[^259]: 多步临床大语言模型智能体的反事实公平性审计需要测量每动作的不稳定性底线

    Counterfactual Fairness Audits of Multi-Step Clinical LLM Agents Require a Measured Per-Action Instability Floor

    [https://arxiv.org/abs/2609.03221](https://arxiv.org/abs/2609.03221)

    临床LLM智能体在完全相同输入下本身就存在显著的动作不稳定性（约8.7%），因此反事实公平性审计必须先测量这一“每动作不稳定性底线”，否则任何检测到的人口统计学差异都无法解释。

    

    反事实审计是检查临床智能体是否对人口统计学上不同但临床上相同的患者采取不同行动的标准工具。这类审计报告一个“翻转率”：当仅改变患者描述时，智能体行动发生改变的频率。我们证明这一指标本身是不可解释的。在16个病例情景上将完全相同的条件重复运行十次（相同叙述、相同描述字符串、不改变任何变量），临床智能体的行动在8.7%的结果-情景单元格中发生了改变，且不稳定性在不同行动之间呈现8倍的异质性，从ICU升级决策的0.022到受管制物质谨慎建议的0.179。我们数据中没有任何人口统计学对比能够与这一底线区分开。第二个模型给出了6.7%的合并底线，且对六种行动的不稳定性排序几乎完全一致（Spearman 0.94，精确p=0.017），说明该底线并非单一系统的特有产物。对五次抽样进行多数投票聚合可以消除其中39%的不稳定性……

    arXiv:2609.03221v1 Announce Type: new  Abstract: Counterfactual audits are the standard tool for checking whether a clinical agent treats demographically distinct but clinically identical patients differently. They report a flip rate: how often an action changes when only the patient descriptor changes. We show that this quantity is uninterpretable on its own. Re-running an identical condition ten times over sixteen vignettes (same narrative, same descriptor string, nothing varied) moved a clinical agent's action in 8.7% of outcome-vignette cells, and instability was heterogeneous across actions by a factor of eight, from 0.022 for ICU escalation to 0.179 for controlled-substance caution. No demographic contrast in our data was distinguishable from that floor. A second model gives a pooled floor of 6.7% and ranks the six actions almost identically (Spearman 0.94, exact p=0.017), so the floor is not one system's artefact. Majority-vote aggregation over five draws removes 39% of it and t
    
[^260]: 边际而非窗口：无需训练的逐步有损推测解码

    Margins, Not Windows: Training-Free Per-Step Lossy Speculative Decoding

    [https://arxiv.org/abs/2609.02897](https://arxiv.org/abs/2609.02897)

    AdaptiveSpec提出了一种无需训练的逐步推测解码方法，通过边际概率比规则放宽严格的token匹配验证，并动态调整草稿树的深度、宽度和节点数，从而在不受草稿长度和起草器架构限制的情况下加速LLM推理。

    

    推测解码通过起草候选token并并行验证来加速大语言模型（LLM）推理。以EAGLE-3为代表的树注意力起草器被广泛采用，但其通常固定了两个决策：（1）严格的token匹配验证规则，（2）静态的草稿树形状。先前的工作在限制性假设下分别对这两者进行放松：基于长草稿链实现无需训练的有损验证，以及在固定token预算下进行自适应树形调整。我们提出了AdaptiveSpec，一种无需训练的逐步推测解码方法，它利用解码过程中已经产生的内部信号来自适应地调整这两个决策。逐步边际规则在目标模型在草拟token上的概率与其top-1概率之比超过阈值时，接受不匹配的草稿提议token，且不依赖于草稿长度或底层起草器架构。逐步树策略则直接调整草稿树的深度、宽度和节点数。

    arXiv:2609.02897v1 Announce Type: new  Abstract: Speculative decoding accelerates LLM inference by drafting candidate tokens and verifying them in parallel. Tree-attention drafters such as EAGLE-3 are widely adopted, yet typically hold two decisions fixed: (1) a strict token-match verification rule and (2) a static draft-tree shape. Prior work relaxes each in isolation under limiting assumptions: long draft chains for training-free lossy verification, and adaptive tree shaping under a fixed token budget. We introduce AdaptiveSpec, a training-free per-step speculative decoding method that adapts both decisions from internal signals already produced during decoding. A per-step margin rule promotes a mismatched draft-proposed token when the ratio of the target's probability on the drafted token to its top-1 probability exceeds a threshold with no dependence on draft length or underlying drafter architecture. A per-step tree policy adjusts the draft tree's depth, width, and node count dire
    
[^261]: 见好就收：用于机器翻译重排序中高效候选生成的Quit方法

    Quit While You're Ahead: Quit for Efficient Candidate Generation in Machine Translation Reranking

    [https://arxiv.org/abs/2609.00588](https://arxiv.org/abs/2609.00588)

    提出Quit方法，通过不确定性量化的早停策略对机器翻译的整个候选生成—重排序流程进行增量式生成与重排序，在最高候选质量稳定时提前终止，从而在保持翻译质量的同时显著降低推理延迟。

    

    重排序方法，如最小贝叶斯风险（MBR）解码和质量估计（QE）重排序，被广泛应用于现代神经机器翻译（NMT）中，用于从一组候选假设中选出最终输出。然而，这些性能提升是以高推理延迟为代价的。现有的加速方法仅针对MBR解码且只减少重排序计算，既未解决QE重排序的问题，也基本未触及候选生成——而后者可能是更大的计算瓶颈。在本工作中，我们提出了Quit（基于不确定性量化的增量终止），这是一种针对整个“生成—重排序”流程的新型早停策略。Quit将候选生成视为不确定性下的序列决策过程，增量式地生成并重排序候选译文，当候选集中最高的估计质量趋于稳定时即停止生成。在三个NMT模型、19个语言对上的全面实验表明……

    arXiv:2609.00588v1 Announce Type: new  Abstract: Reranking methods, such as Minimum Bayes Risk (MBR) decoding and Quality Estimation (QE) reranking, are widely used in modern neural machine translation (NMT) to select an output from a set of candidate hypotheses. However, the performance gains come at the cost of high inference latency. Existing acceleration methods target MBR decoding and reduce only reranking computation, leaving QE reranking unaddressed and candidate generation---which can be the larger computational bottleneck---largely untouched. In this work, we propose Quit (Quantifying Uncertainty for Incremental Termination), a novel early-stopping strategy for the entire generation--reranking pipeline. Viewing candidate generation as a sequential decision under uncertainty, Quit incrementally generates and reranks candidates, stopping when the highest estimated quality in the candidate set stabilizes. Comprehensive experiments on three NMT models across 19 language pairs show
    
[^262]: 视觉并非开销：面向视觉语言模型无损推测解码的单遍块草拟方法

    Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models

    [https://arxiv.org/abs/2609.00355](https://arxiv.org/abs/2609.00355)

    该论文提出 GLANCE——首个在未修改的视觉语言模型上实现无损推测解码的单遍块草拟器，通过块扩散头零成本读取目标模型已融合的视觉-语言状态，并在一次前向传播中完成整块草拟与宽候选树验证，从而打破了草拟器因规模受限而被迫牺牲视觉信息的自我挫败循环。

    

    推测解码能够在不改变输出结果的前提下加速生成，但在视觉语言模型上，它却陷入了一种自我挫败的循环：草拟器必须保持自回归架构，因而只能维持小规模；小型草拟器无法在每一步都承担图像处理的代价，于是视觉信息被压缩、剪枝或隐藏；而被切断了图像信息的草拟器，恰恰在图像最能让文本变得可预测的地方变得最不可靠。我们提出 GLANCE——首个在未经修改的 VLM 目标模型上实现无损解码的单遍块草拟器，它从两端打破了这一循环。一个块扩散头读取目标模型已经融合好的视觉-语言状态，因此视觉对草拟器而言零开销；同时它在一次前向传播中填满整个块，因此模型深度不会带来额外的串行步数。宽候选树通过一次目标模型前向传播即可完成验证，且经审计的每个提示都能精确复现贪婪解码的结果。在依赖视觉依据的工作负载上收益最为显著，会进入一种逐字复制的模式，其长段连续（原文摘要在此处截断）……

    arXiv:2609.00355v1 Announce Type: new  Abstract: Speculative decoding accelerates generation without changing its output, yet on vision-language models (VLMs) it has been caught in a self-defeating cycle. The drafter stays autoregressive, so it must stay small. A small drafter cannot afford the image at every step, so vision is compressed, pruned, or hidden. A drafter cut off from the image is then least reliable exactly where the image makes text predictable. We present GLANCE, the first one-pass block drafter that is lossless on an unmodified VLM target, and it breaks the cycle at both ends. A block-diffusion head reads the target's already-fused vision-language state, so vision costs the drafter nothing, and fills a whole block in one forward pass, so depth costs no sequential steps. A wide candidate tree is verified in one target pass, and every audited prompt reproduces greedy decoding exactly. Grounded workloads reward this most, entering a verbatim-copy regime whose long runs co
    
[^263]: JPO：面向刑事判决预测中结构化法律推理的司法策略优化

    JPO: Juris Policy Optimization for Structured Legal Reasoning in Criminal Judgment Prediction

    [https://arxiv.org/abs/2608.29616](https://arxiv.org/abs/2608.29616)

    提出JPO后训练框架，通过教师监督的标准化四步推理与基于复合奖励的强化学习，实现刑事判决预测中“事实—法条—罪名—量刑”环环相扣的结构化法律推理。

    

    刑事判决预测要求模型从案件事实中推断出适用的法条、罪名和量刑结果。与标准分类任务不同，它涉及一个结构化的推理过程：法条应与事实相匹配，罪名应由法条来论证，量刑结果应与罪名保持一致。现有方法通常只优化最终标签，虽然部分方法尝试评估推理质量，但其评估方式是间接的，往往依赖于大语言模型生成的评分标准，而这些标准反映的是模型内部的偏好，而非法律裁判固有的逻辑结构。我们提出了司法策略优化，这是一个面向中文刑事判决预测中结构化法律推理的后训练框架。JPO首先利用教师模型生成的推理依据来监督一个标准化的四步推理过程，然后应用强化学习，并采用基于标签（与推理过程）的复合奖励机制……（原文摘要在此处截断）

    arXiv:2608.29616v1 Announce Type: new  Abstract: Criminal judgment prediction requires models to infer statutory articles, charges, and sentencing outcomes from case facts. Unlike standard classification tasks, it involves a structured reasoning process in which statutes should be matched with facts, charges should be justified by statutes, and sentencing outcomes should remain consistent with charges. Existing approaches optimize final labels, and while some have attempted to evaluate reasoning quality, their evaluations are indirect, often relying on LLM-generated rubrics that reflect model-internal preferences rather than the inherent logical structure of legal adjudication. We propose Juris Policy Optimization (JPO), a post-training framework for structured legal reasoning in Chinese criminal judgment prediction. JPO first uses teacher-generated rationales to supervise a standardized four-step reasoning process, and then applies reinforcement learning with a composite reward over l
    
[^264]: 替代的幻象：重新思考基础模型时代的专用机器学习模型

    The Illusion of Replacement: Rethinking Specialized Machine Learning Models in the Foundation Model Era

    [https://arxiv.org/abs/2608.28980](https://arxiv.org/abs/2608.28980)

    本文综述159篇论文后发现，语言模型虽在极端少样本预测等特定场景中可与专用模型竞争，但一旦直接评估结构表示与计算能力，并无证据表明其能全面取代机器学习中的专用架构。

    

    机器学习传统上为结构化数据构建的专用架构能否被基于语言的模型所取代？本文通过对2016年至2026年间涵盖九种模态的159篇论文的综述来检验这一问题，在考虑预测精度的同时兼顾结构表示与结构计算。论文区分了“执行任务”与“保留并计算使任务可处理的结构”这两个概念，并将现有方法归纳为八种表示机制，范围从纯语言系统到完全专用的架构。研究发现，语言中介模型在特定场景下极具竞争力，包括极端少样本预测、离散化符号任务、文本标注的知识图谱以及大规模单模态预训练。然而，只要直接评估结构表示或结构计算而非仅评估精度，就没有发现通用替代的证据。

    arXiv:2608.28980v1 Announce Type: cross  Abstract: Can the specialized architectures that machine learning has traditionally built for structured data be replaced by language-based models? This question is examined through a review of 159 papers (2016--2026) across nine modalities, with predictive accuracy considered alongside structural representation and computation. A distinction is made between performing a task and preserving and computing the structure that makes the task tractable, and existing approaches are organized into eight representational regimes, ranging from language-only systems to fully specialized architectures. Language-mediated models are found to be highly competitive in specific settings, including extreme few-shot prediction, discretized symbolic tasks, textually annotated knowledge graphs, and large-scale single-modality pretraining. However, whenever structural representation or computation is directly evaluated rather than accuracy alone, no evidence of gene
    
[^265]: CultureConverse：一个面向东亚与东南亚文化情境化辅助的多语言多轮对话模拟评测框架

    CultureConverse: A Multilingual Multi-turn Simulation Harness for Culturally Grounded Assistance in East and Southeast Asia

    [https://arxiv.org/abs/2608.28405](https://arxiv.org/abs/2608.28405)

    该论文提出CultureConverse，一个覆盖东亚与东南亚10个地区、58个子群体身份和7个领域的多语言多轮文化情境化助手对话模拟与评测框架，并构建了包含14,610个基准评测回合和274,295个oracle引导对话的数据集，弥补了传统单选题式文化评测无法反映多轮实际辅助场景的不足。

    

    当前针对大语言模型（LLM）的文化评测往往将文化简化为通过多选题进行的单轮事实性问答，无法捕捉一个常见的使用场景：用户在文化情境化的场景中通过多轮对话寻求实际帮助。我们提出了CultureConverse，这是一个可扩展的、多语言的文化情境化助手对话模拟与评测框架，覆盖10个东亚和东南亚地区、58个子群体身份以及7个领域。每一次被模拟和评测的对话回合都会产生一个带评分的交互，其中助手为用户提供帮助，并从部分信息中推断文化约束。由此构建的CultureConverse-DS数据集包含14,610个基准（评测）回合和274,295个由oracle引导（gold模式）的对话。在对18个模型的基准评测中，GPT-5 mini获得了最高的辅助质量。人工标注实验表明，我们的评测框架可以作为一种充分的替代指标……

    arXiv:2608.28405v1 Announce Type: new  Abstract: Current cultural evaluations for large language models (LLMs) often reduce culture to single-turn factual recall via MCQs, failing to capture a common use case: users seeking practical help over multiple turns in culturally grounded scenarios. We introduce CultureConverse, a scalable, multilingual simulation and evaluation harness for culturally grounded assistant dialogue that covers 10 East and Southeast Asian regions, 58 subgroup identities, and 7 domains. Each simulated and evaluated episode produces a scored interaction where the assistant assists the user and infers cultural constraints from partial information. The resulting CultureConverse-DS dataset contains 14,610 benchmark (evaluation) episodes and 274,295 oracle-guided (gold-mode) dialogues. In our benchmark evaluation of 18 models, GPT-5 mini achieves the highest assistance quality. Human annotation experiments suggest that our evaluation framework is a sufficient proxy for 
    
[^266]: 实体追踪在低于十亿参数的语言模型中出现，并在自然叙事中超越人类表现

    Entity tracking emerges in sub-billion parameter language models and exceeds human performance in naturalistic narratives

    [https://arxiv.org/abs/2608.18083](https://arxiv.org/abs/2608.18083)

    本文发现实体追踪能力在低至4.1亿参数的语言模型中即已出现，并在自然叙事中随模型规模增大而超越人类表现，而人类表现则受叙事复杂度影响。

    

    arXiv:2608.18083v1 公告类型：新 摘要：理解语言需要在话语中追踪实体——即知道事物在哪里以及它们如何变化，即使未明确提及。语言模型是否以类似人类的方式进行这种追踪仍不清楚，部分原因在于现有评估依赖于人为任务，远离自然语言理解，并缺乏与人类的比较。在这里，我们使用多种复杂度的自然叙事评估了语言模型和人类（N = 48）中的实体追踪。在人类中，我们发现实体追踪特别随叙事复杂度而退化，而非叙事长度。在语言模型中，我们发现人类水平的实体追踪在4.1亿参数时已经存在——远低于先前工作识别的数十亿参数、代码专门化模型——并随规模提升而改进，当代模型远超人类表现。综合这些结果，演示了...

    arXiv:2608.18083v1 Announce Type: new  Abstract: Understanding language requires tracking entities across discourse - i.e., knowing where things are and how they change, even when not explicitly stated. Whether language models perform such tracking in a human-like fashion remains unclear, in part because existing evaluations rely on artificial tasks, far removed from natural language comprehension, and lack comparisons to humans. Here, we evaluate entity tracking in both language models and humans (N = 48) using naturalistic narratives at multiple levels of complexity. In humans, we find that entity tracking degrades specifically with narrative complexity, not narrative length. In language models, we find that human-level entity tracking is already present at 410 million parameters - well below the multi-billion parameter, code-specialised models identified by prior work - and improves with scale, with contemporary models far exceeding human performance. Together, these results demonst
    
[^267]: 基于人类反馈的策略迭代：将训练后强化学习引入上下文学习

    Policy Iteration with Human Feedback: Bringing Post-Training RL to In-context Learning

    [https://arxiv.org/abs/2608.16831](https://arxiv.org/abs/2608.16831)

    本文提出PIHF方法，利用预训练语言模型作为执行基础，通过语言模型批评者和临床专家的循环评估与修订，将强化学习思想引入上下文学习，从而改进策略性能。

    

    生成式预训练建立了可复用的任务表征；后续关于基于语言的任务条件化和上下文学习的研究表明，固定模型能够根据指令和演示调整其行为。基于人类反馈的策略迭代（PIHF）在此基础上发展，并融合了广义策略迭代中循环评估与改进的结构。PIHF使用预训练语言模型作为执行基础，并将持续修订迁移至版本化的自然语言策略和工具集。一个语言模型批评者和临床专家审查完整面板推理和工具使用轨迹，以定位反复出现的失败并形成候选修订；专家可重新解释证据并保留准入和回滚的权威，而Recall@1和Recall@5在候选执行后验证结果。在累积消融和超罕见疾病基准测试中，基于PIHF的策略将R值提升了...

    arXiv:2608.16831v1 Announce Type: new  Abstract: Generative pretraining established reusable task representations; later work on language-based task conditioning and in-context learning showed that a fixed model could adapt its behavior from instructions and demonstrations. Policy Iteration with Human Feedback (PIHF) builds on this development and the recurrent evaluate-and-improve structure of generalized policy iteration. PIHF uses a pretrained language model as its execution substrate and moves persistent revision to a versioned natural-language policy and tool set. A language-model critic and clinical expert review complete-panel reasoning and tool-use trajectories to localize recurrent failures and form candidate revisions; the expert may reinterpret the evidence and retains authority over admission and rollback, while Recall@1 and Recall@5 validate outcomes after candidate execution.   Across cumulative ablations and ultra-rare-disease benchmarks, a PIHF-derived policy improved R
    
[^268]: 混合线性注意力大语言模型中的大规模激活：注意力前尖峰与尖峰间平台

    Massive Activations in Hybrid Linear Attention Large Language Models: Pre-Attention Spikes and Inter-Spike Plateaus

    [https://arxiv.org/abs/2608.12149](https://arxiv.org/abs/2608.12149)

    本文首次系统研究了混合线性注意力大语言模型中的大规模激活现象，发现了注意力前尖峰和尖峰间平台两种新形态，并揭示了它们与架构配置的关系。

    

    我们首次对层交错混合线性注意力（HLA）大语言模型中的大规模激活（MAs）进行了系统性研究，并揭示了两种与架构对齐的形态：MAs在完全注意力层之前持续出现尖峰，形成注意力前尖峰（PAS），并且可以持续通过中间的线性注意力层，产生尖峰间平台（ISP）。随着完全注意力变得更密集，连续的PAS通过ISP逐渐连接，最终恢复完全注意力大语言模型的稳定MA形态。我们证实了这种组织在五种线性注意力架构、六种混合配置、五个数据域以及代表1.2B到397B总参数规模的开源混合模型中的重复性。基于GDN的混合模型在高达1.3B规模的受控预训练表明，这两种形态在早期出现，并对输出门控表现出不对称响应：完全注意力输出门控强烈减弱了...

    arXiv:2608.12149v1 Announce Type: new  Abstract: We present the first systematic study of Massive activations (MAs) in layer-interleaved HLA LLMs and uncover two architecture-aligned morphologies: MAs consistently spike immediately before full attention layers, forming pre-attention spikes (PAS), and can persist through intervening linear attention layers, giving rise to inter-spike plateaus (ISP). As full attention becomes denser, successive PAS become increasingly connected through ISP, ultimately recovering the stable MA morphology of full attention LLMs. We establish the recurrence of this organization across five linear attention architectures, six hybridization configurations, five data domains, and representative open-source hybrid models spanning 1.2B to 397B total parameters. Controlled pretraining of GDN-based hybrids at scales up to 1.3B shows that both morphologies emerge early and respond asymmetrically to output gating: full attention output gating strongly attenuates the
    
[^269]: RAISE：诊断昂贵LLM信号中的获取崩溃

    RAISE: Diagnosing Acquisition Collapse in Costly LLM Signals

    [https://arxiv.org/abs/2608.10441](https://arxiv.org/abs/2608.10441)

    本文识别出“获取崩溃”这一失败模式，并提出了RAISE预路由诊断框架，帮助判断昂贵的LLM信号在何时才值得调用，从而避免盲目调用造成的资源浪费。

    

    大型语言模型（LLM）日益被用作真实系统中昂贵的按需组件，但无差别地调用它们可能会浪费大量的计算资源、延迟时间和服务预算。因此，关键的部署问题不仅在于LLM平均而言是否有帮助，更在于何时值得调用它。我们识别出一种常见的失败模式，称之为“获取崩溃”：一个LLM信号可能在总体上或事后看来是有用的，但在调用之前所能提供的信息却太少，无法支持可靠的选择性使用。我们提出了RAISE（Reward-SNR Actionability in Signal Evaluation，信号评估中的奖励信噪比可行动性），这是一个预路由诊断框架，用于在确定路由策略之前检验现有证据是否支持选择性使用。我们通过结构化假设嵌入来实现RAISE，这是一种用于推荐的冻结LLM意图信号，每个用户仅需一次LLM调用，并通过受控实验、回顾性研究和新用户队列研究对其进行评估。

    arXiv:2608.10441v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used as costly, on-demand components in real systems, but calling them indiscriminately can waste substantial compute, latency, and serving budget. The key deployment question is therefore not only whether an LLM helps on average, but when it is worth calling. We identify a common failure mode, which we call acquisition collapse: an LLM signal can appear useful in aggregate or post hoc, yet still provide too little before-call information to support reliable selective use. We introduce RAISE (Reward-SNR Actionability in Signal Evaluation), a pre-routing diagnostic framework for testing whether available evidence supports selective use before committing to a routing strategy. We instantiate RAISE with Structured Hypothesis Embeddings (SHE), a frozen-LLM intent signal for recommendation using one LLM call per user, and evaluate it through controlled, retrospective, and fresh-cohort st
    
[^270]: VectraYX-Vision-1B：一个具有结构化视觉推理与原生工具使用能力的、参数量小于2B的西班牙语/拉美地区网络安全视觉语言模型

    VectraYX-Vision-1B: A Sub-2B Spanish/LATAM Cybersecurity Vision-Language Model with Structured Visual Reasoning and Native Tool Use

    [https://arxiv.org/abs/2608.08477](https://arxiv.org/abs/2608.08477)

    v3版本将原生训练的Qwen2-VL视觉塔移植到同一冻结解码器上，使原本失败的8半字节地址字段精确率从0.00跃升至0.81，且token预算比2x2平铺更粗糙，证明分辨率并非关键变量。

    

    arXiv:2608.08477v3 公告类型：替换 摘要：25页，1幅图，10个表格。v3版本：将一个原生训练的视觉塔（Qwen2-VL）移植到相同的冻结解码器上，使原本完全失败的8个半字节（nibble）地址字段的精确匹配率从0.00提升至0.81，且所用的token预算比2x2平铺方案更粗糙，这一结果驳斥了分辨率是关键操作变量的假设。第二个预注册字段被发现存在63%的数据污染，已被降级处理。B6/B7工具识别仍停留在最低水平。代码和模型检查点已发布在Hugging Face上。

    arXiv:2608.08477v3 Announce Type: replace  Abstract: 25 pages, 1 figure, 10 tables. v3: transplanting a natively-trained visual tower (Qwen2-VL) onto the same frozen decoder takes the failing 8-nibble address field from 0.00 to 0.81 exact, at a coarser token budget than 2x2 tiling, refuting resolution as the operative variable. Second pre-registered field found 63% contaminated, demoted. B6/B7 tool-id remains at floor. Code/checkpoints on HF.
    
[^271]: MoEGen：用于实例自适应LoRA生成的专家混合模型

    MoEGen: Mixture-of-Experts for Instance-Adaptive LoRA Generation

    [https://arxiv.org/abs/2608.03275](https://arxiv.org/abs/2608.03275)

    MoEGen将基于MoE的PEFT从专家选择转变为专家条件参数生成，用小型专家码和轻量级超网络按输入动态生成低秩更新，在避免适配器存储随专家数线性增长的同时实现实例自适应适配。

    

    参数高效微调（PEFT）能够实现对大语言模型的高效适配，但现有的基于专家混合（MoE）的PEFT方法通常通过存储多个完整的LoRA专家来提升模型容量，这导致适配器的存储量随专家数量线性增长，并将适配限制在固定的专家池中。我们提出了一个问题：基于MoE的PEFT能否在无需为每个专家显式存储独立LoRA模块的情况下，生成针对具体实例的适配？为填补这一空白，我们提出MoEGen，这是一个将基于MoE的PEFT从专家选择转变为专家条件参数生成的适配框架。MoEGen不再将每个专家存储为完整的LoRA适配器，而是将每个专家表示为一个小的可学习向量，称为专家码。它针对每个输入在这些向量上进行路由，并利用它们的加权组合来条件化一个轻量级超网络，由该超网络生成针对输入的低秩更新。这一设计将专家容量与存储解耦……

    arXiv:2608.03275v2 Announce Type: replace  Abstract: Parameter-efficient fine-tuning (PEFT) enables efficient adaptation of large language models, but existing MoE-based PEFT methods typically improve capacity by storing multiple full LoRA experts, causing adapter storage to grow linearly with the number of experts and restricting adaptation to a fixed expert pool. We ask whether MoE-based PEFT can produce instance-specific adaptations without explicitly storing a separate LoRA module for each expert. To address this gap, we propose MoEGen, an adaptation framework that shifts MoE-based PEFT from expert selection to expert-conditioned parameter generation. Instead of storing each expert as a full LoRA adapter, MoEGen represents each expert as a small learnable vector, termed an expert code. It routes each input over these vectors and uses their weighted combination to condition a lightweight hypernetwork that generates input-specific low-rank updates. This design decouples expert capaci
    
[^272]: OpenART：通过开放式环境演化扩展智能体红队测试

    OpenART: Scaling Agent Red Teaming via Open-Ended Environment Evolution

    [https://arxiv.org/abs/2608.00677](https://arxiv.org/abs/2608.00677)

    该论文提出了OpenART——一个通过环境演化实现可扩展智能体红队测试的开放式平台，包含覆盖50个领域的1万多个有状态场景，并提出演化马尔可夫超图攻击（EMHA）这一黑盒策略，以系统探索智能体在长时程有状态工作流中不断演化的安全攻击面。

    

    AI智能体运行在持久性环境中，早期的状态变化可能会深远地影响未来的决策。与传统语言模型交互不同，智能体行为是通过一个被反复修改并在长时程工作流中复用的共享状态来中介的。当前的安全基准通常无法捕捉这类累积性风险，因为它们聚焦于简短、静态的任务。为解决这些局限性，我们提出了OpenART，一个通过环境演化实现可扩展智能体红队测试的开放式竞技场。OpenART提供了超过10,000个经过验证的、覆盖50个领域的有状态场景，其来源于一个包含超过500,000个工具和技能的资源池。这些任务的中位数需要97次工具调用，并支持跨75种不同智能体-模型配置的统一评估。为了系统地探索这些不断演化的攻击面，我们提出了演化马尔可夫超图攻击（EMHA）。EMHA是一种黑盒策略，它……（原文在此处截断）

    arXiv:2608.00677v2 Announce Type: replace  Abstract: AI agents operate in persistent environments where early state changes can influence decisions far into the future. Unlike conventional language-model interactions, agent behavior is mediated through a shared state that is repeatedly modified and reused across long-horizon workflows. Current safety benchmarks often fail to capture these cumulative risks because they focus on short, static tasks. To address these limitations, we introduce OpenART, an open-ended arena for scalable agent red teaming through environment evolution. OpenART provides over 10,000 validated stateful scenarios across 50 domains, drawing from a pool of more than 500,000 tools and skills. These tasks require a median of 97 tool calls and enable unified evaluation across 75 different agent-model configurations. To systematically explore these evolving attack surfaces, we propose the Evolutionary Markov Hypergraph Attack (EMHA). EMHA is a black-box policy that per
    
[^273]: 在策略优化中由验证器诱导的支撑集重塑

    Verifier-Induced Support Reshaping in On-Policy Optimization

    [https://arxiv.org/abs/2608.00220](https://arxiv.org/abs/2608.00220)

    本文揭示了RLVR训练会重塑策略的支撑集，使其在提升当前任务表现的同时让后续任务的成功行为难以被采样到，例如数学训练使IFEval的pass@1提升6.5个百分点但best@32下降9.8个百分点。

    

    我们证明，基于可验证奖励的在策略强化学习（RLVR）在提升当前目标表现的同时，可能会使后续目标所需的成功行为变得过于稀少，以至于无法被采样和强化。我们将这种现象称为“验证器诱导的支撑集重塑”，并将“有效可奖励支撑”定义为在固定采样预算内可触达的成功轨迹。我们在两个模型家族上，通过重复的验证器评分采样和双向训练，在数学推理与受约束的指令遵循任务上研究这一效应，其中包括使用相反验证器进行的顺序训练。数学RLVR提高了指令遵循的平均成功率，但在重复采样下减少了存在任何成功响应的提示数量。在Qwen3-8B-Base模型上，IFEval的pass@1提升了6.5个百分点，而best@32下降了9.8个百分点，且同样的分歧在两个模型和所有指令遵循基准上均出现。相反，IF-RLV……（摘要原文在此处截断）

    arXiv:2608.00220v2 Announce Type: replace-cross  Abstract: We show that on-policy reinforcement learning with verifiable rewards (RLVR) can improve the current objective while making successful behaviors for later objectives too rare to sample and reinforce. We call this verifier-induced support reshaping and define effective rewardable support as successful trajectories reachable within a fixed rollout budget. Across two model families, we study this effect through repeated verifier-scored sampling and bidirectional training on mathematical reasoning and constrained instruction following, including sequential training with the opposite verifier. Math-RLVR raises average instruction-following success but reduces the number of prompts with any successful response under repeated sampling. On IFEval with Qwen3-8B-Base, pass@1 rises by 6.5 percentage points while best@32 falls by 9.8 percentage points, and the same divergence appears across both models and IF benchmarks. Conversely, IF-RLV
    
[^274]: 一种用于数据高效RL对齐的宪法网格工具（C-Guard）

    A Constitution-Grid Instrument for Data-Efficient RL Alignment (C-Guard)

    [https://arxiv.org/abs/2608.00180](https://arxiv.org/abs/2608.00180)

    本文提出C-Guard和C-LIM，通过宪法网格生成训练数据并逐单元评分，在训练前识别无效数据区域，显著提升RL对齐中的数据效率和安全性。

    

    arXiv:2608.00180v4 公告类型：替换 摘要：在RL对齐中，目标冲突是普遍存在的，而高效地训练这些目标数据很困难。训练安全防护模型涉及优化两个相互冲突的目标：捕捉真实危害，同时不拒绝良性提示。我们的发现是，过度拒绝从22.4%改善到12.8%，而对对抗性攻击的拒绝不足则悄然恶化，从0.27增至0.33。我们提出了C-Guard，一种生成RL训练数据的宪法网格工具，以及C-LIM，一个逐单元可学习性评分，决定每个单元的动作：剪枝、加密、修正、扩展。C-LIM在任何训练预算花费之前标记出无效数据区域：187个非定向行未带来任何收益，而我们的方法将该区域的学习影响从0.733提升到0.80。代码和宪法已开源。

    arXiv:2608.00180v4 Announce Type: replace  Abstract: Conflicting objectives are general in RL alignment, and training on them data-efficiently is hard. Training a safety guard with RL means optimizing two objectives that conflict: catch real harm, and do not refuse benign prompts. Our finding is that over-refusal improves 22.4% to 12.8%, while under-refusal on adversarial attacks silently worsens 0.27 to 0.33. We present C-Guard, a constitution-grid instrument that generates the RL training data, and C-LIM, a per-cell learnability score that decides each cell's move: prune, densify, amend, expand. C-LIM flags the dead-weight data region before any training budget is spent: 187 untargeted rows had bought zero gain, and our method lifts the same region's learning impact 0.733 to 0.80. Code and the constitution are open-sourced.
    
[^275]: 小语言模型领域自适应的可信度代价：一项跨架构实证研究

    Trustworthiness Costs of Domain Adaptation in Small Language Models:A Cross-Architecture Empirical Study

    [https://arxiv.org/abs/2608.00042](https://arxiv.org/abs/2608.00042)

    本文首次通过跨架构、跨领域、跨训练数据条件的系统实证研究，量化了小语言模型在领域自适应微调过程中付出的可信度代价（事实校准与对抗鲁棒性），并对比了四种微调策略在医疗、法律、金融领域中的可信度表现。

    

    摘要（arXiv:2608.00042v2，公告类型：replace-cross）：小语言模型（SLM）的领域自适应已成为在医疗保健、法律服务和金融分析等资源受限且高风险环境中部署强大NLP系统的实用策略。尽管参数高效微调带来的性能提升已被充分表征，但其对可信度（即事实校准与对抗鲁棒性）的相应影响仍知之甚少。本文首次开展了系统性跨领域、跨架构的实证研究，以量化领域自适应带来的可信度代价，研究涵盖三种SLM架构（TinyLlama 1B、Gemma-2 2B、Llama 3.2 1B）、三个领域（医疗、法律、金融）、两种训练数据条件（良性数据与对抗扰动数据），以及四种微调策略（基线LoRA、Safety-DPO、暗经验回放和任务算术LoRA，TA-LoRA）。可信度通过TruthfulQA MC2等指标进行评估……

    arXiv:2608.00042v2 Announce Type: replace-cross  Abstract: Domain adaptation of small language models (SLMs) has emerged as a practical strategy for deploying capable NLP systems in resource-constrained, high-stakes environments including healthcare, legal services, and financial analysis. While performance gains from parameter-efficient fine-tuning are well characterised, the corresponding impact on trustworthiness (factual calibration and adversarial robustness) remains poorly understood. This paper presents the first systematic cross-domain, cross-architecture empirical study quantifying the trustworthiness cost of domain adaptation across three SLM architectures (TinyLlama 1B, Gemma-2 2B, Llama 3.2 1B), three domains (healthcare, legal, finance), two training-data conditions (benign and adversarially perturbed), and four fine-tuning strategies (baseline LoRA, Safety-DPO, Dark Experience Replay, and Task Arithmetic LoRA, TA-LoRA). Trustworthiness is evaluated through TruthfulQA MC2 
    
[^276]: ORCA-bench：语言模型智能体对值班排障（Oncall）的准备程度如何？

    ORCA-bench: How Ready Are Language Model Agents for Oncall?

    [https://arxiv.org/abs/2607.28545](https://arxiv.org/abs/2607.28545)

    该论文提出了ORCA-bench基准，将1,079个根因分析任务与真实可观测性工具接口及六天生产级遥测数据相结合，系统评估语言模型智能体在值班根因分析场景中的真实能力。

    

    大型语言模型能够编写、修补和搜索代码，但值班根因分析（RCA）需要的是不同的能力：从模糊的用户报告中出发，对嘈杂的指标、日志、追踪数据和源代码进行推理，且通常发生在事故开始数小时之后。我们提出了ORCA-bench，这是一个将通用编码智能体置于高保真生产环境值班场景中的基准测试。ORCA-bench将1,079个RCA任务与从持续模拟用户负载下、经过OpenTelemetry插桩的微服务系统中收集的六天指标、日志和追踪数据相配对。智能体通过真实的可观测性接口——通过Grafana访问的Prometheus、Jaeger和OpenSearch——调查这些记录的历史数据，并可完全访问应用程序源代码。任务系统地改变报告的具体程度、检测时间以及共现故障场景。真实症状由专家SRE（站点可靠性工程师）审核并签署确认，我们的LLM-as-judge是独立……

    arXiv:2607.28545v3 Announce Type: replace-cross  Abstract: Large language models can write, patch, and search code, but oncall root cause analysis (RCA) demands something different: reasoning over noisy metrics, logs, traces, and source code, starting from ambiguous user-facing reports, often hours after the incident began. We introduce ORCA-bench, a benchmark that puts general-purpose coding agents in a production-fidelity oncall setting. ORCA-bench pairs 1,079 RCA tasks with six days of metrics, logs, and traces collected from an OpenTelemetry-instrumented microservice system under continuous simulated user load. Agents investigate this recorded history through real observability interfaces---Prometheus, Jaeger, and OpenSearch via Grafana---with full access to application source code. Tasks systematically vary report specificity, time-to-detection, and co-occurring fault scenarios. Ground-truth symptoms are curated and signed off by expert SREs, and our LLM-as-judge is independently 
    
[^277]: AdvancedMathBench：面向高等数学证明生成与验证的基准测试套件

    AdvancedMathBench: A Benchmark Suite for Advanced Mathematical Proof Generation and Verification

    [https://arxiv.org/abs/2607.11849](https://arxiv.org/abs/2607.11849)

    该论文提出AdvancedMathBench基准套件，包含245道本科及博士资格考试级别的高等数学证明题，并配套基于大规模专家标注训练的自动验证流水线，以细粒度方式评估大语言模型在高等数学证明生成与验证上的推理能力。

    

    大型语言模型（LLMs）在高中及竞赛水平的数学问题上已取得显著表现，但其在高等数学方面的能力仍然知之甚少。然而，现有的基准测试在范围和评估粒度上均存在不足：它们提供的学科覆盖面有限，且往往依赖最终答案的正确性或粗略的判断，导致对推理过程有效性的评估不够充分。为弥合这一差距，我们提出了AdvancedMathBench，一个旨在评估LLMs在高等数学证明上推理能力的基准测试套件。其核心生成基准ProverBench包含245道题目，涵盖本科（UG）和博士资格考试（QE）两个级别。为了可靠地评估这些证明，我们开发了一套专用的自动验证流水线，该流水线基于大规模专家标注进行训练，能够同时给出正确性判定和细粒度的分析……

    arXiv:2607.11849v2 Announce Type: replace  Abstract: Large language models (LLMs) have achieved remarkable performance on high-school and competition-level mathematics, yet their capabilities on advanced mathematics remain poorly understood. Existing benchmarks, however, fall short in both scope and evaluation granularity: they provide limited disciplinary coverage and often rely on final-answer correctness or coarse judgments, leaving the validity of the reasoning process inadequately assessed. To bridge this gap, we introduce AdvancedMathBench, a benchmark suite designed to evaluate the reasoning capabilities of LLMs on advanced mathematical proofs. Its core generation benchmark, ProverBench, contains 245 problems spanning undergraduate (UG) and doctoral qualifying-exam (QE) levels. To reliably evaluate these proofs, we develop a dedicated automatic verification pipeline that is trained on large-scale expert annotations, produces both correctness verdicts and fine-grained analyses, a
    
[^278]: 我们衡量的是策略还是措辞？大语言模型数学推理中表层多样性与方法层面多样性之间的鸿沟

    Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning

    [https://arxiv.org/abs/2606.29985](https://arxiv.org/abs/2606.29985)

    该论文提出“方法层面多样性”概念，证明现有多样性指标只是表面措辞差异的不可靠代理而非解题策略差异，且直接优化LLM裁判的多样性奖励会导致投机行为而非真正拓宽解题思路，但方法多样的候选集能提升测试时扩展效果。

    

    arXiv:2606.29985v2 公告类型：替换。摘要：大语言模型（LLM）数学推理中的多样性对探索至关重要，但常用的多样性指标大多只捕捉表层变化，而非解决问题方式的差异。我们通过引入“方法层面多样性”来弥补这一缺口，即对同一问题的不同正确解答之间在解题策略上的差异。借助一个经人类校准的大语言模型裁判（LLM judge）框架，我们证明以往的多样性指标是方法层面多样性的不可靠代理，且这种不匹配会延续到多样性感知的RLVR训练中——目标指标虽然得以保持，方法层面多样性却在下降。在研究方法层面多样性何时有帮助以及能否被直接诱导时，我们发现方法多样的候选集能够改善测试时扩展效果。然而，在训练过程中优化大语言模型裁判的多样性奖励，会导致策略去利用裁判的特定偏好，而不是真正拓宽其解题方法，使得对方法层面多样性的直接优化落空。

    arXiv:2606.29985v2 Announce Type: replace  Abstract: Diversity in LLM mathematical reasoning is critical for exploration, but common diversity metrics mostly capture surface-level variation rather than differences in how a problem is solved. We address this gap by introducing approach-level diversity: variation in strategies across correct solutions to the same problem. Using a human-calibrated LLM judge framework, we show that prior diversity measures are unreliable proxies for approach-level diversity, and this mismatch carries over to diversity-aware RLVR, where target metrics are preserved while approach-level diversity declines. Investigating when approach-level diversity helps and whether it can be directly induced, we find that approach-diverse candidate sets improve test-time scaling. However, optimizing an LLM judge diversity reward during training causes the policy to exploit judge-specific preferences rather than broaden its approaches, leaving direct optimization of approac
    
[^279]: TriageRA-CCF：面向医学大语言模型自适应秩预算的源侧临床置信度与覆盖度信号

    TriageRA-CCF: Source-Side Clinical Confidence and Coverage Signals for Adaptive Rank Budgeting in Medical LLMs

    [https://arxiv.org/abs/2606.29375](https://arxiv.org/abs/2606.29375)

    该论文提出TriageRA-CCF，利用仅从源训练数据计算的三种信号——基础模型答案置信度、临床覆盖度和反事实近似命中代理——来训练直通式预算路由器，为每个医学问题自适应分配LoRA秩预算，从而在避免路由坍缩的同时提升参数高效医学问答性能。

    

    医学大语言模型通常采用固定的低秩预算进行参数高效适配，尽管医学问题在置信度、临床覆盖度和跨领域难度上存在显著差异。我们研究了面向参数高效医学问答的自适应秩预算方法：对于每个问题，适配器决定是否激活LoRA秩通道的小、中或大子集。其核心挑战在于，朴素的自适应预算路由器可能坍缩为不稳定的选择，或者在分布偏移的基准测试上消耗容量却无法带来性能提升。我们提出了TriageRA-CCF，一种用于自适应秩预算LoRA的源侧教师方法。该方法结合了仅从源训练数据计算的三种信号：基础模型答案置信度、元数据单元临床覆盖度以及反事实近似命中代理信号。这些信号对活跃秩{2,4,8}上的直通式预算路由器进行监督，并辅以预算成本、熵和秩平衡正则化约束。

    arXiv:2606.29375v2 Announce Type: replace  Abstract: Medical large language models are commonly adapted with a fixed low-rank budget, even though medical questions differ substantially in confidence, clinical coverage, and cross-domain difficulty. We study adaptive rank budgeting for parameter-efficient medical question answering: for each question, the adapter decides whether to activate a small, medium, or large subset of LoRA rank channels. The central challenge is that a naive adaptive budget router can collapse to unstable choices or spend capacity without improving shifted benchmarks. We propose TriageRA-CCF, a source-side teacher for adaptive rank-budgeted LoRA. It combines three signals computed only from source training data: base-model answer confidence, metadata-cell clinical coverage, and a counterfactual close-miss proxy. These signals supervise a straight-through budget router over active ranks {2,4,8}, together with budget-cost, entropy, and rank-balance regularization. 
    
[^280]: 拟人化语言会影响公众对人工智能的认知吗？

    Does Anthropomorphic Language Impact Public Perceptions of AI?

    [https://arxiv.org/abs/2606.29121](https://arxiv.org/abs/2606.29121)

    本研究通过815名参与者的实验，考察了AI公共话语中拟人化语言对公众认知的影响，并比较了这种影响在大语言模型与推荐系统两类AI技术之间的差异。

    

    arXiv:2606.29121v2 公告类型：交叉替换。摘要：关于人工智能（AI）的公共讨论中经常使用拟人化语言，即把人类的能力和特征归因于AI系统的语言。这种做法因设定误导性预期、夸大宣称以及助长围绕AI的炒作而受到批评，这可能扭曲公众对AI的理解并影响政策优先事项。我们通过比较参与者在阅读包含或不包含拟人化语言的段落（这些段落旨在反映面向公众的AI话语的真实情况）时对AI认知的变化（N=815），研究了拟人化框架的影响。我们进一步考察了这些影响在两类AI技术——大语言模型和推荐系统——之间是否存在差异，并测量了在当前公共讨论中备受关注的多个维度上对AI认知的变化。在一个使用明确讨论AI危险性文本的独立实验条件下，我们展示了……（摘要在此处被截断）

    arXiv:2606.29121v2 Announce Type: replace-cross  Abstract: Public discourse about artificial intelligence (AI) often uses anthropomorphic language: language that attributes human capabilities and characteristics to AI systems. This practice has been criticized for setting misleading expectations, inflating claims, and fueling hype around AI, which may distort public understanding of AI and impact policy priorities. We study the effects of anthropomorphic framing by comparing changes in participants' perceptions of AI (N=815) when reading passages with and without anthropomorphic language, designed to reflect realistic public-facing AI discourse. We further examine whether these effects differ across two types of AI technologies -- large language models and recommendation systems -- and measure changes in perceptions of AI across several dimensions that are prominent in current public discourse. In a separate condition using a text that explicitly discusses the dangers of AI, we show th
    
[^281]: HPRO：基于偏好提取的情感文本转语音分层渐进奖励优化

    HPRO: Hierarchical Progressive Reward Optimization via Preference Extraction for Emotional Text-to-Speech

    [https://arxiv.org/abs/2606.28249](https://arxiv.org/abs/2606.28249)

    提出分层渐进奖励优化框架HPRO，通过HD-Emo编解码器将语音解耦为独立的内容与风格偏好标记以缓解信息冲突，并弥合句子级奖励与帧级生成之间的尺度差距，从而提升情感文本转语音的表现力。

    

    近年来，基于大语言模型（LLM）的文本转语音（TTS）模型在自然度方面取得了显著成就。然而，标准的监督微调范式往往收敛于统计平均的韵律，限制了情感表现力。尽管偏好驱动的优化提供了一种有前景的替代方案，但现有方法存在两个结构性失配：一是信息冲突，即内容与情感在共享潜空间中产生相互矛盾的梯度，导致奖励破解（reward hacking）和语义退化；二是尺度差距，即稀疏的句子级奖励难以指导密集的帧级生成。为克服这些挑战，我们提出了HPRO，一个分层渐进奖励优化框架。在HPRO中，我们引入HD-Emo编解码器作为一种新颖的可微奖励模型，以缓解信息冲突。它将语音提取为相互独立的内容偏好标记和风格偏好标记，从结构上……（摘要在此处截断）

    arXiv:2606.28249v2 Announce Type: replace-cross  Abstract: Recently, Large Language Model (LLM)-based Text-to-Speech (TTS) models have achieved remarkable naturalness. However, the standard Supervised Fine-Tuning paradigm often converges to statistically averaged prosody, limiting emotional expressiveness. While preference-driven optimization offers a promising alternative, existing approaches suffer from two structural mismatches: information conflict, where content and emotion in a shared latent space produce conflicting gradients, leading to reward hacking and semantic degradation; and scale gap, where sparse sentence-level rewards struggle to guide dense frame-level generation. To overcome these challenges, we propose HPRO, a hierarchical progressive reward optimization framework. Within HPRO, we introduce the HD-Emo codec as a novel differentiable reward model to mitigate the information conflict. It extracts speech into distinct content and style preference tokens, structurally i
    
[^282]: 基于顿悟分数的KV缓存淘汰方法：无需注意力矩阵

    Epiphany-Aware KV Cache Eviction Without the Attention Matrix

    [https://arxiv.org/abs/2606.26472](https://arxiv.org/abs/2606.26472)

    本文提出EpiKV，一种通过直接读取模型前向传播中的内部表示变化（顿悟分数）来淘汰KV缓存的方法，无需注意力矩阵，可将可行上下文长度扩展至传统注意力评分方法的16倍，且无需训练或自定义内核。

    

    arXiv:2606.26472v1 公告类型：交叉 摘要：随着推理模型生成长达数万token的思维链，KV缓存日益成为部署瓶颈。现有缓存淘汰方法通过注意力权重对token进行排序，这在长推理轨迹中是一个有噪声的重要性代理，并且通过强制模型实现注意力矩阵，阻碍了生产推理中融合内核的使用。在这项工作中，我们提出了一种称为"顿悟分数"的度量标准来对token进行评分：该分数直接从前向传播中读取模型内部表示的变化，无需注意力矩阵且仅需极少的额外状态。由此产生的缓存淘汰方法EpiKV无需训练、分类器或自定义内核，可直接在FlashAttention推理栈中不变地使用——将可行上下文长度扩展至基于注意力评分方法的16倍。针对上层中间层（负向影响）和下层中间层（正向影响），我们采用因果滚动z分数消除位置趋势。在4096-token缓存设置下，该方法表现出色。

    arXiv:2606.26472v1 Announce Type: cross  Abstract: As reasoning models emit chains of thought tens of thousands of tokens long, KV cache increasingly becomes a deployment bottleneck. Existing cache eviction methods rank tokens by attention weight, which is a noisy importance proxy in long reasoning traces, and prohibits the use of fused kernels in production inference by forcing the model to materialize the attention matrix. In this work, we instead score tokens with a metric we term the epiphany score: the change in the model's internal representation, read directly from the forward pass with no attention matrix and negligible extra state. Our resulting cache eviction method, EpiKV, requires no training, classifier, or custom kernel, and can be used directly in FlashAttention inference stacks unchanged -- scaling to a 16x longer feasible context than attention-based scoring. upper-mid layers negatively) and remove a positional trend with a causal rolling z-score. At a 4096-token cache
    
[^283]: 《智能体人工智能漫游指南：从基础到系统》

    The Hitchhiker's Guide to Agentic AI: From Foundations to Systems

    [https://arxiv.org/abs/2606.24937](https://arxiv.org/abs/2606.24937)

    该论文（著作）是一部从大语言模型基础、对齐与推理技术到智能体AI全栈覆盖的构建自主AI系统综合实践指南，其核心观点是：构建优秀的智能体系统必须理解技术栈的每一个层级。

    

    《智能体人工智能漫游指南》是一本面向从业者的综合性参考书，旨在指导构建自主人工智能系统，内容涵盖从第一性原理到生产部署的全栈知识。本书的核心论点是：构建优秀的智能体系统需要理解技术流水线的每一层，而不仅仅是其中某一层。本书开篇介绍大语言模型（LLM）基础层，涵盖Transformer架构、GPU系统、训练与微调（SFT、LoRA、MoE）、模型压缩以及推理优化等必备基础。随后阐述对齐与推理层：RLHF、PPO、DPO及其变体、GRPO、奖励建模，以及面向大型推理模型的强化学习，包括思维链和测试时扩展（test-time scaling）。本书后半部分专门聚焦智能体AI本身：智能体训练与基于轨迹的强化学习、RAG与智能体RAG、记忆系统（上下文内记忆、外部记忆、情景记忆与语义记忆）、智能体框架设计、循环工程、基于图的编排等。

    arXiv:2606.24937v3 Announce Type: replace  Abstract: The Hitchhiker's Guide to Agentic AI is a comprehensive practitioner's reference for building autonomous AI systems, covering the full stack from first principles to production deployment. The central thesis: building great agentic systems requires understanding every layer of the pipeline, not just one. The book opens with the LLM substrate, covering transformer architecture, GPU systems, training and fine-tuning (SFT, LoRA, MoE), model compression, and inference optimization, as essential foundations. It then develops the alignment and reasoning layer: RLHF, PPO, DPO and its variants, GRPO, reward modeling, and RL for large reasoning models including chain-of-thought and test-time scaling. The second half is devoted to agentic AI proper: agentic training and trajectory-based RL, RAG and Agentic RAG, memory systems (in-context, external, episodic, and semantic), agent harness design, loop engineering, graph-based orchestration, and 
    
[^284]: CORE-BREW：基于LLR的软解码，用于鲁棒的多比特大语言模型水印

    CORE-BREW: LLR-Based Soft Decoding for Robust Multi-Bit LLM Watermarking

    [https://arxiv.org/abs/2606.24163](https://arxiv.org/abs/2606.24163)

    CORE-BREW通过恒定命中率校准推导出逐token的对数似然比以实现软判决解码，并结合熵感知擦除机制，显著提升了多比特LLM水印在编辑和改写攻击下的检测鲁棒性与信息恢复能力。

    

    可靠的大语言模型（LLM）输出溯源需要多比特水印，这类水印在文本被编辑后仍能保持鲁棒性，同时维持较低的误报率。现有的基于纠错码（ECC）的LLM水印依赖硬判决解码，丢弃了token级别的可靠性信息，从而限制了其在生成后编辑下的鲁棒性。我们提出CORE-BREW，这是BREW的一种恒定命中率嵌入扩展，用于多比特水印。CORE-BREW通过以固定命中率p*为目标来校准水印信道，得到闭式解的逐token对数似然比（LLR）用于软判决解码。它引入熵感知擦除机制以限制低熵上下文中的扰动，并将基于似然的评分与软判决列表解码相结合以充分利用软证据。在开源LLM上针对token级编辑和改写攻击的实验表明，CORE-BREW相较于BREW基础版本，通常能提升检测鲁棒性和载荷恢复能力。

    arXiv:2606.24163v2 Announce Type: replace-cross  Abstract: Reliable provenance for LLM outputs requires multi-bit watermarks that remain robust under editing while maintaining low false-positive rates. Existing ECC-based LLM watermarks rely on hard-decision decoding, discarding token-level reliability information and limiting robustness under post-generation edits. We propose CORE-BREW, a COnstant-hit-Rate Embedding extension of BREW for multi-bit watermarking. CORE-BREW calibrates the watermark channel by targeting a fixed hit rate $p^\star$, yielding closed-form per-token log-likelihood ratios (LLRs) for soft-decision decoding. It incorporates entropy-aware erasures to limit perturbations in low-entropy contexts and combines likelihood-based scoring with soft-decision list decoding to exploit soft evidence. Experiments on open-source LLMs under token-level edits and paraphrasing demonstrate that CORE-BREW generally improves detection robustness and payload recovery over the BREW base
    
[^285]: 智能体代码比人类代码更难维护吗？

    Is Agent Code Less Maintainable Than Human Code?

    [https://arxiv.org/abs/2606.21804](https://arxiv.org/abs/2606.21804)

    本研究提出 CodeThread 框架，通过受控实验发现智能体基于智能体生成代码解决任务的效率低于基于人类代码（任务解决率最多下降 13.1%），且传统可维护性指标无法解释这一差异。

    

    可维护性是软件工程的核心维度，塑造着代码随时间推移如何被编写、审查和开发。尽管编码智能体在单 issue 任务上已展现出强大性能，但当未来的智能体在其代码之上继续构建时，这些代码的可维护性如何仍不清楚，而这可能导致不断叠加的下游影响。我们研究了在这些维护场景中智能体代码与人类代码的对比，并提出了 CodeThread——一个从仓库级编码基准中构建受控实验的框架。将 CodeThread 应用于四个前沿编码智能体和四个基准后，我们发现智能体在基于智能体代码解决任务时效果不如基于人类代码，任务解决率下降幅度高达 13.1%。回归分析表明，许多传统软件工程可维护性指标无法解释这一差异；相反，最清晰的信号更为微妙……

    arXiv:2606.21804v2 Announce Type: replace-cross  Abstract: Maintainability is a core dimension of software engineering, shaping how code is written, reviewed, and developed over time. While coding agents have demonstrated strong performance on single-issue tasks, it remains unclear how maintainable their code is when future agents build on top of it, potentially leading to compounding downstream effects. We investigate how agent code compares to human code in these maintenance settings, presenting CodeThread, a framework to construct controlled experiments from repository-level coding benchmarks. Applying CodeThread to four frontier coding agents and four benchmarks, we find that agents are less effective at resolving tasks when building on agent code compared to human code, with task resolve rate drops of up to 13.1%. Regression analysis reveals that many traditional software engineering maintainability metrics do not explain this difference. Instead, the clearest signals are subtler 
    
[^286]: 复现、分析与检测基于评分标准的强化学习中的奖励黑客行为

    Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning

    [https://arxiv.org/abs/2606.04923](https://arxiv.org/abs/2606.04923)

    本文提出了一个可控黑客环境CHERRL，通过注入已知偏见到评判者中，实现了奖励黑客行为的稳定复现与分析，为检测和缓解该问题提供了实验平台。

    

    摘要：基于评分标准的强化学习（RL）使用大语言模型作为评判者（LaaJ），根据评分标准对模型输出进行评分作为奖励。然而，策略模型可能利用评判者中的潜在偏见，导致奖励黑客行为，进而产生无效或不安全的训练结果。在现实世界的基于评分标准的RL中，此类黑客行为通常微妙且与多种评判者偏见交织在一起，使其难以分析、检测和缓解。在本文中，我们引入了CHERRL，一个用于基于评分标准的RL的可控黑客环境。通过向LaaJ注入已知偏见，CHERRL能够稳定复现奖励黑客行为，明确观察奖励偏差，并识别黑客行为的发生点。这为研究基于评分标准的RL中奖励黑客行为的机制和缓解措施提供了一个干净的实验平台。为展示其效用，我们从可发现性和可利用性的角度分析了不同的评判者偏见。

    arXiv:2606.04923v2 Announce Type: replace-cross  Abstract: Rubric-based reinforcement learning (RL) uses an LLM-as-a-Judge (LaaJ) to score model outputs according to rubrics as rewards. However, policy models may exploit latent biases in the judge, leading to reward hacking and ineffective or unsafe training outcomes. In real-world rubric-based RL, such hacking behaviors are often subtle and entangled with multiple judge biases, making them difficult to analyze, detect, and mitigate. In this paper, we introduce CHERRL, a Controllable Hacking Environment for Rubric-based RL. By injecting known biases into LaaJ, CHERRL enables stable reproduction of reward hacking, explicit observation of reward divergence, and identification of hacking onset. This provides a clean experimental testbed for studying the mechanisms and mitigations of reward hacking in rubric-based RL. To demonstrate its utility, we analyze different judge biases from the perspectives of discoverability and exploitability, 
    
[^287]: 基于采样的推理：在决策点处切割

    Reasoning with Sampling: Cutting at Decision Points

    [https://arxiv.org/abs/2605.30327](https://arxiv.org/abs/2605.30327)

    该研究表明从基础模型的幂分布中采样即可达到媲美强化学习训练的推理能力，并提出应在推理轨迹中的关键决策点处进行切割重采样，以实现高效的混合采样。

    

    前沿推理模型是通过强化学习对基础语言模型进行后训练而得到的。最近的研究对这一做法提出了挑战，表明从基础模型分布的锐化版本（即所谓的幂分布）中进行采样，无需额外训练、精选数据集或验证器，即可引发相当水平的推理能力。然而，要使这一方法实用化，需要能够高效地从幂分布中采样。采样器需要“混合”到幂分布，这要求在目标分布的众数之间移动；直观地说，例如尝试不同的推理策略。先前工作提出的采样器反复地在当前推理轨迹中均匀随机地选择一个“切割”位置，并从该位置起重新采样后缀。然而，推理轨迹通常只包含少数几个关键性的决策（例如证明策略或算法的选择），我们观察到均匀随机的切割方式并不理想（摘要原文在此处截断）。

    arXiv:2605.30327v2 Announce Type: replace-cross  Abstract: Frontier reasoning models are produced by post-training base language models with reinforcement learning. Recent work has challenged this by showing that sampling from a sharpened version of the base model's distribution, a so-called power distribution, elicits comparable reasoning without additional training, curated datasets, or verifiers. However, making this method practical requires efficiently sampling from the power distribution. A sampler needs to "mix" to the power distribution, which necessitates moving between modes of the target distribution; intuitively, e.g., trying different reasoning strategies. The samplers proposed in prior works repeatedly select a "cut" position in the current reasoning trace uniformly at random and resample the suffix from that position onward. However, reasoning traces typically contain a few consequential decisions (e.g., the choice of proof strategy or algorithm), and we observe that a u
    
[^288]: 主动式代理需要大语言模型来决定何时行动吗？

    Do Proactive Agents Need an LLM to Decide When to Act?

    [https://arxiv.org/abs/2605.30152](https://arxiv.org/abs/2605.30152)

    该论文提出用时间图学习（TGL）控制器替代大语言模型来决定主动式代理何时介入及如何选择上下文，在每事件仅11.13毫秒的低成本下实现了对语言代理性能的提升。

    

    主动式助手需要持续决定何时介入以及什么样的上下文应当支持这一介入。大型语言模型（LLM）流水线需要反复解读活动历史来做出这些决策，即使助手保持沉默也要付出推理成本。我们证明了一个小型图模型可以同时处理这两种决策，并改善它所控制的语言代理。我们的关键洞察是：用户活动具有天然的图结构——事件涉及持久存在的实体，这些实体的重复出现将交互随时间连接起来。触发决策和上下文选择可以直接映射为对事件节点和实体节点的预测。我们提出了一个时间图学习（TGL）控制器，它联合学习这些预测，并在一次前向传播中同时提供两个输出。下游的语言代理在被触发的事件上，利用活动历史和评分后的实体来生成建议。在GPU服务器上，TGL每个事件仅需11.13毫秒，并取得了最高的（性能表现）……

    arXiv:2605.30152v2 Announce Type: replace-cross  Abstract: Proactive assistants continuously decide when to intervene and what context should support the intervention. Large language model (LLM) pipelines repeatedly interpret activity histories to make these decisions, paying an inference cost even when the assistant remains silent. We show that a small graph model can handle both decisions and improve the language agents it controls. Our key insight is that user activity has a native graph structure: events involve persistent entities whose recurrence connects interactions over time. Triggering and context selection map directly to predictions on event and entity nodes. We introduce a temporal-graph-learning (TGL) controller that learns these predictions jointly and supplies both outputs in one forward pass. The downstream language agent generates suggestions on triggered events using the activity history and scored entities. At 11.13 ms per event on a GPU server, TGL achieves the hig
    
[^289]: 使用IndicKLAR评估代码混合与印度语言之间的跨语言知识一致性

    Evaluating Cross-lingual Knowledge Consistency in Code-Mixed vis-a-vis Indian Languages using IndicKLAR

    [https://arxiv.org/abs/2605.29637](https://arxiv.org/abs/2605.29637)

    提出IndicKLAR基准，涵盖18种印度语言及11个语言对的代码混合变体，揭示大模型在印度本土语言与英语间的知识准确率差距可达约0.50，而代码混合输入能将该差距缩小至约0.05以内。

    

    大型语言模型在同等知识查询上，其英语表现与低资源语言表现之间往往存在显著差距——这种跨语言一致性问题在印度语言及其代码混合形式上仍未得到充分研究。为了研究这一差距，我们提出了IndicKLAR，它是KLAR-CLC基准的印度语扩展版本，涵盖了印度22种表列语言中的18种。针对11种广泛使用的语言对，我们还额外提供了代码混合变体。单语和代码混合输入均经过母语使用者验证。这种三方对齐使我们能够检验知识召回一致性在英语、代码混合和印度本土语言输入之间的差异。在九个开放权重模型上，我们发现本土语言输入与英语输入之间的准确率差距可达约0.50，而代码混合输入显著缩小了这一差距，使性能与英语的差距缩小到约0.05以内。

    arXiv:2605.29637v2 Announce Type: replace  Abstract: Large language models often exhibit a substantial gap between their performance in English and in lower-resourced languages on equivalent knowledge queries---a cross-lingual consistency issue that remains underexplored for Indian languages and their code-mixed counterparts. To study this gap, we introduce IndicKLAR, an Indic extension of the KLAR-CLC benchmark covering 18 of the 22 scheduled Indian languages. For 11 widely used language pairs, we additionally provide code-mixed variants. Both monolingual and code-mixed inputs verified by native speakers. This three-way alignment enables us to examine how knowledge recall consistency varies across English, code-mixed, and native Indian language inputs. Across nine open-weight models, we find that the accuracy gap between native-language and English inputs can reach $\sim$0.50, while code-mixed inputs substantially reduce this gap, bringing performance within $\sim$0.05 of English with
    
[^290]: 通过首词元探索实现RLVR采样轨迹多样化

    Diversifying RLVR Rollouts via First-Token Exploration

    [https://arxiv.org/abs/2605.28295](https://arxiv.org/abs/2605.28295)

    论文发现回复首个词元的分布高度集中且与答案正确性关系微弱，据此提出轻量级方法REFT，通过让首个词元多样化来拓宽RLVR每组轨迹的推理路径探索，且几乎不损失回答质量。

    

    可验证奖励强化学习（RLVR）无需标注轨迹即可训练推理模型，它依靠经验证器评分的成组采样轨迹来探索不同的推理路径。轨迹多样性不足是其中的核心瓶颈，以往通常通过调整温度、前缀或轨迹选择来缓解。我们指出回复的首个词元是一个结构上独特、且在以往工作中大多被忽视的多样化目标。我们发现首个词元的分布高度集中，且与下游答案正确性仅有微弱关联——低概率的候选词元同样能产生准确率相当的回答。因此，让首个词元多样化可以在几乎不损失回答质量的前提下，拓宽每组轨迹所探索的推理路径。基于这一观察，我们提出了REFT（Rollout Exploration with First-Token Diversification，基于首词元多样化的轨迹探索），这是对RLVR的一种轻量级修改。REFT通过采样……（摘要原文在此截断）

    arXiv:2605.28295v2 Announce Type: replace  Abstract: Reinforcement learning with verifiable rewards (RLVR) trains reasoning models without labeled trajectories, using groups of verifier-scored rollouts to explore alternative reasoning paths. Limited rollout diversity is a central bottleneck, typically addressed through adjustments to temperature, prefixes, or rollout selection. We identify the first token of the response as a structurally distinct target for diversification, largely overlooked in prior work. We find that the first-token distribution is sharply concentrated and only weakly related to downstream correctness, as lower-probability candidates can yield similarly accurate responses. Diversifying the first token can therefore broaden the reasoning paths explored within each rollout group with little loss in response quality. Motivated by this observation, we introduce REFT (Rollout Exploration with First-Token Diversification), a lightweight modification to RLVR. REFT samples
    
[^291]: KSAFE-MM：基于本地化情境化针对韩国文化风险的多模态安全基准

    KSAFE-MM: A Multimodal Safety Benchmark via Localized Contextualization for Korean Cultural Risks

    [https://arxiv.org/abs/2605.28013](https://arxiv.org/abs/2605.28013)

    该论文提出了KSAFE-MM基准，通过语言情境化评估韩国语境下的全球共享安全风险（KSAFE-MM-G），并结合本地化视觉查询与越狱式文本查询来评估文化依赖性的安全漏洞（KSAFE-MM-C），弥补了现有安全评估工具以英语为中心且忽视本地文化风险的不足。

    

    多模态大语言模型（MLLMs）通过在语言和视觉等多种模态中引入漏洞而加剧了安全风险。然而，当前的多模态大语言模型安全评估工具存在重大局限：1）以英语为中心的数据集构建；2）关注与本地文化背景无关的通用风险。本文提出了KSAFE-MM，这是一个用于韩国多模态安全评估的基准，同时涵盖了一般安全风险和文化特有的安全漏洞。KSAFE-MM由两个互补部分组成：KSAFE-MM-G通过语言情境化评估韩国语境下全球共享的风险，将通用安全查询转化为基于情境的多模态样本；相比之下，KSAFE-MM-C针对依赖文化的安全漏洞，使用从现实世界中提取的本地化视觉查询，并将这些视觉查询与越狱式文本查询相配对。

    arXiv:2605.28013v2 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) exacerbate safety risks by introducing vulnerabilities across multiple modalities, such as language and vision. Current MLLM safety evaluation tools, however, suffer from major limitations: 1) English-centric dataset construction, and 2) a focus on generic risks that are not tied to local cultural contexts. This paper introduces KSAFE-MM, a benchmark for Korean multimodal safety evaluation that covers both general safety risks and culture-specific vulnerabilities. KSAFE-MM consists of two complementary parts: KSAFE-MM-G evaluates globally shared risks in Korean contexts through linguistic contextualization, which transforms generic safety queries into contextually grounded multimodal samples. In contrast, KSAFE-MM-C targets safety vulnerabilities that are culture-dependent, using localized visual queries drawn from real-world. It pairs these visual queries with jailbreak-style textual queries 
    
[^292]: 分解与测量评估意识

    Decomposing and Measuring Evaluation Awareness

    [https://arxiv.org/abs/2605.23055](https://arxiv.org/abs/2605.23055)

    该论文基于社会心理学提出了一个评估意识分解框架，将其分为环境成分（八类触发因素）和模型成分，并通过思维链监测发现九个前沿模型对四个基准测试的识别率取决于模型与基准的配对方式，且识别很少引发行为变化，即使引发，方向也取决于评估类型。

    

    前沿语言模型有时会识别到自己正处于评估之中，并相应调整其行为，这可能损害基准测试结果的有效性。然而，目前该领域的研究缺乏共同的理论基础，将评估本身的缺陷与模型的能力混为一谈，也将“识别”与“行为反应”混为一谈。我们将评估意识建立在社会心理学的基础上，将其分解为环境成分和模型成分，从而将“识别”与“倾向”分离开来。我们通过八类触发因素（如占位符实体和评分式输出格式）对环境成分进行操作化定义，并通过思维链监测来研究识别与行为。在九个前沿模型和四个基准测试上，识别率取决于模型与基准测试的具体配对。识别很少与行为变化相关联，即使相关联，其方向也取决于评估的类型。

    arXiv:2605.23055v3 Announce Type: replace-cross  Abstract: Frontier language models sometimes recognize that they are under evaluation and adjust their behavior which can undermine validity of benchmark results. Yet the field studies it without a shared foundation, conflating flaws of the evaluation with capabilities of the model, and detection with behavioral response. We ground evaluation awareness in social psychology, decomposing it into an environment component and a model component that separates recognition from propensity. We operationalize the environment component through eight categorized trigger factors, such as placeholder entities and grading-style output formats, and study recognition and behavior through chain-of-thought monitoring. Across nine frontier models and four benchmarks, recognition rates depend on the specific pairing of model and benchmark. Recognition rarely associates with behavioral change, and when it does, the direction depends on the type of evaluation
    
[^293]: 铭记你的足迹：面向一致性与分层结构的仓库级代码文档的记忆引导长程智能体框架

    Remember Your Trace: Memory-Guided Long-Horizon Agentic Framework for Consistent and Hierarchical Repository-Level Code Documentation

    [https://arxiv.org/abs/2605.14563](https://arxiv.org/abs/2605.14563)

    提出MemDocAgent长程智能体框架，通过依赖感知的遍历引导与基于共享记忆RepoMemory的记忆引导智能体交互，在覆盖整个仓库的单一集成上下文中生成一致且具有分层结构的仓库级代码文档。

    

    arXiv:2605.14563v3 公告类型： replace-cross 摘要：自动化代码文档生成对现代软件开发至关重要，它为人类开发者和编码智能体提供了赖以浏览大型代码库的上下文基础。现有的仓库级方法独立处理各个组件，导致冗余检索以及文档间描述冲突，同时生成的输出缺乏分层结构。因此，我们提出了MemDocAgent，这是一个长程智能体框架，能够在覆盖整个仓库的单一集成上下文中生成文档。它结合了两个组件：（i）依赖感知的遍历引导，其预先确定遵循依赖关系与粒度层级的遍历顺序；（ii）记忆引导的智能体交互，其中智能体与RepoMemory进行交互——这是一个通过读取、写入和验证操作来积累先前工作痕迹的共享记忆。通过深入的多维度评估，Me（摘要在此处被截断）

    arXiv:2605.14563v3 Announce Type: replace-cross  Abstract: Automated code documentation is essential for modern software development, providing the contextual grounding that both human developers and coding agents rely on to navigate large codebases. Existing repository-level approaches process components independently, causing redundant retrieval and conflicting descriptions across documents while producing outputs that lack hierarchical structure. Therefore, we propose MemDocAgent, a long-horizon agentic framework that generates documentation within a single, integrated context spanning the entire repository. It combines two components: (i) Dependency-Aware Traversal Guiding that predetermines a traversal order respecting dependency and granularity hierarchies; (ii) Memory-Guided Agentic Interaction, in which the agent interacts with RepoMemory, a shared memory accumulating prior work traces through read, write, and verify operations. Through an in-depth multi-criteria evaluation, Me
    
[^294]: PRISM：将漂移分解为尺度、形状与预测头的几何风险界

    PRISM: A Geometric Risk Bound for Decomposing Drift into Scale, Shape, and Head

    [https://arxiv.org/abs/2605.11608](https://arxiv.org/abs/2605.11608)

    提出PRISM，一种交叉熵风险差距的闭式上界，可将模型漂移精确分解为尺度、形状和预测头三个可测量的几何轴，从而将变体的特征变化与性能退化联系起来。

    

    如今，单个基础大语言模型会衍生出数十个训练后变体——量化版本、LoRA适配版本或蒸馏版本——而每个变体在发布前都必须经过检查。现有评估只能提供片面的信息：基准测试分数和似然筛查能说明某个变体已经退化，而CKA和SVCCA等相似性分数能说明其特征如何变化，但没有任何方法将两者联系起来。我们通过一个结构性事实和一个设计选择将它们联系起来：预测头是线性的，因此特征几何结构能够传递到损失函数中；并且我们通过一个正交映射来比较两组特征，这种映射不会改变所测量的几何结构。基于此，我们推导出PRISM——目标模型与代理变体之间交叉熵风险差距的闭式上界——并证明它可以精确分解为三个可测量的轴：尺度、形状和预测头。每个轴都指出一种失败模式以及相应的干预位置：低比特量化会扭曲形状，并在最低（摘要在此处截断）

    arXiv:2605.11608v2 Announce Type: replace-cross  Abstract: A single base LLM now comes with dozens of post-training variants, quantized, LoRA-adapted, or distilled, and each has to be checked before release. Existing evaluations provide only a partial picture: benchmark scores and likelihood screens say that a variant has degraded, similarity scores such as CKA and SVCCA say how its features moved, and nothing connects the two. We connect them with one structural fact and one design choice: the prediction head is linear, so feature geometry reaches the loss, and we compare the two feature sets through an orthogonal map, which leaves the geometry being measured unchanged. From these we derive PRISM, a closed-form upper bound on the cross-entropy risk gap between a target model and a proxy variant, and prove that it splits exactly into three measurable axes: scale, shape, and head. Each axis names a failure mode and where to intervene: low-bit quantization distorts shape and, at the lowe
    
[^295]: PowerStep：基于 $\ell_p$-范数最速下降的内存高效自适应优化

    PowerStep: Memory-Efficient Adaptive Optimization via $\ell_p$-Norm Steepest Descent

    [https://arxiv.org/abs/2605.10335](https://arxiv.org/abs/2605.10335)

    PowerStep 受 $\ell_p$-范数最速下降启发，仅对单个动量缓冲区施加带符号幂变换即可实现无需存储二阶矩的自适应优化，相比 AdamW 将 fp32 优化器状态内存减半，结合 int8 量化更可减少约 8 倍，同时在 124M 到 235B 参数的 Transformer 上保持有竞争力的性能。

    

    诸如 Adam 等自适应优化器是训练 Transformer 的标准选择，但存储梯度的一阶矩和二阶矩会带来显著的内存开销。我们提出了 PowerStep，一种内存高效的优化器，它无需存储二阶矩统计量即可实现逐坐标的自适应性。受 $\ell_p$-范数最速下降的启发，PowerStep 直接对单个动量缓冲区施加带符号幂变换。我们为精确、无正则化的更新建立了有限时域的平稳性界，其中包含一个 $O(1/\sqrt{T})$ 项和一个依赖噪声的残差项。在参数量从 1.24 亿（124M）到 2350 亿（235B）的 Transformer 上的实验表明，PowerStep 在验证质量上具有竞争力，同时相对于 AdamW 将 fp32 优化器状态内存减半。结合均匀 int8 量化，PowerStep 仍保持数值稳定性，与 fp32 AdamW 相比将优化器状态内存减少约 8 倍。因此，PowerStep 提供了一种简单、内存高效（摘要在此处截断）……

    arXiv:2605.10335v2 Announce Type: replace-cross  Abstract: Adaptive optimizers such as Adam are standard for training Transformers, but storing gradient first and second moments incurs substantial memory overhead. We introduce PowerStep, a memory-efficient optimizer that achieves coordinate-wise adaptivity without storing second-moment statistics. Motivated by $\ell_p$-norm steepest descent, PowerStep applies a signed-power transform directly to one momentum buffer. We establish a finite-horizon stationarity bound for exact, unregularized updates, with an $O(1/\sqrt{T})$ term and a noise-dependent residual. Experiments on Transformers from 124M to 235B parameters show competitive validation quality while halving $\texttt{fp32}$ optimizer-state memory relative to AdamW. Combined with uniform $\texttt{int8}$ quantization, PowerStep remains numerically stable and reduces optimizer-state memory by $\sim8\times$ compared to $\texttt{fp32}$ AdamW. PowerStep thus provides a simple, memory-eff
    
[^296]: 相对动力学效用：为大语言模型全局结构化剪枝校准跨层贡献

    Relative Kinetic Utility: Calibrating Cross-Layer Credit for Global Structured LLM Pruning

    [https://arxiv.org/abs/2605.09008](https://arxiv.org/abs/2605.09008)

    提出Global RKU，一种无标签的全局结构化剪枝准则，通过最终隐藏状态的激活-梯度信号衡量通道参与度，并利用块相对归一化消除块级公共尺度对跨层比较的干扰，使不同层通道分数具有可比性，从而实现更准确的大语言模型全局剪枝。

    

    全局结构化剪枝要求不同层的通道在共享的稀疏性预算下进行竞争，这带来了两个相互耦合的挑战：识别哪些通道应当被保留，以及使它们的分数在不同层之间具有可比性。原始通道分数中可能包含块级公共尺度，这种尺度虽然不会改变块内的排序，但会扭曲整个模型层面的竞争。我们的实验表明，相似的逐层分配可以保留差异很大的FFN通道，因此仅凭层分配并不能决定通道的去留。受这种分离性的启发，我们提出了全局相对动力学效用，这是一种无标签的准则，它将通道重要性估计与跨层比较分离开来。Global RKU使用最终隐藏状态的激活-梯度信号来衡量通道参与度，然后应用块相对归一化来缓解块级公共尺度的影响，同时保持块内排序不变……

    arXiv:2605.09008v2 Announce Type: replace-cross  Abstract: Global structured pruning requires channels from different layers to compete under a shared sparsity budget, raising two coupled challenges: identifying which channels should be retained and making their scores comparable across layers. Raw channel scores can contain block-common scale that leaves within-block ordering unchanged but distorts model-wide competition. Our experiment indicates that similar layer-wise allocations can retain substantially different FFN channels, so layer allocation alone does not determine channel identity. Motivated by this separation, we introduce Global Relative Kinetic Utility (Global RKU), a label-free criterion that separates channel importance estimation from cross-layer comparison. Global RKU measures channel participation using a final-hidden-state activation-gradient signal, then applies block-relative normalization to mitigate block-common scale while preserving within-block ordering, requ
    
[^297]: GRAVITY：面向长程对话记忆的架构无关结构化锚定

    GRAVITY: Architecture-Agnostic Structured Anchoring for Long-Horizon Conversational Memory

    [https://arxiv.org/abs/2605.01688](https://arxiv.org/abs/2605.01688)

    GRAVITY提出了一种与架构无关的辅助记忆层，在生成阶段将对话整合为实体档案、事件轨迹和跨会话主题摘要并注入提示，在五种异构记忆系统和两个基准上均显著提升了长程对话记忆性能。

    

    长程记忆系统在证据的存储与检索方式上不断改进，然而生成器仍然需要在跨会话关系隐含不明的片段上进行推理。我们将生成时（generation-time）的记忆组织作为一个独立的设计维度进行研究，并提出了GRAVITY（通过注入拓扑记忆实现生成时关系锚定），这是一个与宿主无关的辅助记忆层。GRAVITY将原始对话整合为实体档案、时间事件轨迹和跨会话主题摘要，然后通过提示接口检索并注入与查询相关的记录。在LongMemEval和LoCoMo基准上的五种异构记忆系统中，GRAVITY在两种不同的LLM配置下均提升了每个宿主-基准基线的表现。控制性分析将“组织已有证据”带来的收益与“整合完整历史信息”带来的收益区分开来。在一个匹配的LightMem流水线中，实体-事件-主题（摘要在此处截断）

    arXiv:2605.01688v2 Announce Type: replace-cross  Abstract: Long-horizon memory systems increasingly improve how evidence is stored and retrieved, yet the generator must still reason over fragments whose cross-session relationships are implicit. We study generation-time memory organization as a distinct design dimension and introduce GRAVITY (Generation-time Relational Anchoring Via Injected Topological MemorY), a host-independent auxiliary memory layer. GRAVITY consolidates raw dialogue into entity profiles, temporal event traces, and cross-session topic summaries, then retrieves and injects query-relevant records through the prompt interface. Across five heterogeneous memory systems on LongMemEval and LoCoMo, it improves every host--benchmark baseline under two distinct LLM configurations. Controlled analyses separate gains from organizing already available evidence and from consolidating information across the full history. Under a matched LightMem pipeline, the entity--event--topic 
    
[^298]: 多语言思考，而非更费力：一个教会推理模型进行语码转换的数据高效框架

    Think Multilingual, Not Harder: A Data-Efficient Framework for Teaching Reasoning Models to Code-Switch

    [https://arxiv.org/abs/2604.15490](https://arxiv.org/abs/2604.15490)

    该论文提出了首个基于语言学与行为动机的微调框架，并构建了CoRe语料库，用于识别大型语言模型中有益的语码转换推理行为，从而以数据高效的方式教会推理模型更有效地利用语码转换来提升推理能力。

    

    推理能力的最新进展使大型语言模型能够解决日益复杂的数学、符号和逻辑任务。有趣的是，尽管推理模型通常被训练为生成单语文本，但这些模型也被观察到会出现语码转换（即混合使用多种语言）的现象。以往的工作要么将语码转换视为一种不良错误，要么试图通过修改输入提示或输出解码过程来控制语码转换，要么只关注语言、领域、任务和模型的狭窄子集。我们通过引入首个基于语言学与行为动机的微调框架来填补这些空白，该框架用于识别大型语言模型中有益的语码转换推理行为，并教会这些模型更有效地利用语码转换进行推理。我们构建了语码转换推理语料库，其中包含来自15个模型、18种语言的7千条推理轨迹……（原文摘要在此处截断）

    arXiv:2604.15490v2 Announce Type: replace  Abstract: Recent developments in reasoning capabilities have enabled large language models to solve increasingly complex mathematical, symbolic, and logical tasks. Interestingly, while reasoning models are often trained to generate monolingual text, these models have also been observed to code-switch (i.e., mix languages). Prior works have either viewed code-switching as an undesirable error, attempted to control code-switching through modifications to input prompts or the output decoding process, or focus on narrow subsets of languages, domains, tasks, and models. We address these gaps by introducing the first linguistically and behaviorally motivated fine-tuning framework for identifying beneficial code-switched reasoning behaviors in large language models and teaching these models to code-switch more effectively for reasoning. We create the Code-Switched Reasoning (CoRe) corpus, consisting of (1) 7k reasoning traces from 15 models, 18 langu
    
[^299]: 从实践视角重新审视思维链蒸馏中的容量差距

    Revisiting the Capacity Gap in Chain-of-Thought Distillation from a Practical Perspective

    [https://arxiv.org/abs/2604.08880](https://arxiv.org/abs/2604.08880)

    该论文从实践视角重新审视思维链蒸馏中的容量差距问题，发现在更现实的实验设置下容量差距并非总是主导因素，当候选教师模型性能差异显著时选择更强的教师通常更优，从而为教师模型选择提供了实用指导。

    

    思维链蒸馏将推理行为从强大的教师模型迁移到更小的学生模型，但先前的研究报告了“容量差距”问题：当教师与学生的能力差距过大时，蒸馏可能会失败。我们从实践角度重新审视容量差距问题，重新检验了常用的实验设置。值得注意的是，我们发现思维链蒸馏相比学生模型蒸馏前的基线往往会降低性能，而且先前工作中使用的一些设置虽然适合将容量差距确立为一种现象，但并不能反映现实的部署场景。作为对先前确立容量差距工作的补充，我们在更现实的设置下评估了其影响，发现容量差距并不总是起主导作用；当候选教师模型之间的性能差异显著时，更强的教师往往更受青睐。我们的结果为教师模型的选择提供了实用指导。

    arXiv:2604.08880v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) distillation transfers reasoning behaviors from a strong teacher to a smaller student, but prior work reports a capacity gap: distillation may fail when the teacher-student capability mismatch is large. We revisit the capacity gap from a practical perspective by re-examining commonly used experimental settings. Notably, we find that CoT distillation often degrades performance compared to the student's pre-distillation baseline, and that some settings used in prior work, while suitable for establishing the capacity gap as a phenomenon, do not reflect realistic deployment scenarios. Complementing prior work that establishes the capacity gap, we evaluate its practical impact under more realistic settings and find that it does not consistently dominate; stronger teachers tend to be preferable when candidate teachers differ substantially in performance. Our results offer practical guidance for selecting teache
    
[^300]: 筛选即足够

    Screening Is Enough

    [https://arxiv.org/abs/2604.01178](https://arxiv.org/abs/2604.01178)

    本文提出“筛选”注意力机制，通过显式阈值将查询-键相似度转换为绝对相关度，无需推理时扩展即可在长文本中保持低困惑度和稳健检索，并据此构建了参数效率更高、零样本性能更强的Multiscreen语言模型架构。

    

    当查询-键相关度的值位于固定的有界尺度上、既不依赖于相互竞争的键也不依赖于序列长度、无需随序列长度进行校准、且允许全部为零时，我们称这种相关度为“绝对相关度”。为实现这一概念，我们提出了筛选机制，它通过显式阈值将有界的查询-键相似度转换为相关度值，从而实现精确拒绝、空选择以及在统一尺度上的直接检查。在同一Transformer骨干网络上对12种注意力机制进行的受控比较中，只有筛选机制在超出训练上下文的情况下，仍能同时保持较低的长文本困惑度和稳健的检索能力；值得注意的是，这无需依赖推理时的扩展。基于筛选机制，我们提出了Multiscreen，一种由并行门控筛选单元组成的语言模型架构。Multiscreen在保留这些长文本优势的同时，实现了更高的参数效率和更强的通用零样本下游性能。

    arXiv:2604.01178v4 Announce Type: replace-cross  Abstract: We call query--key relevance absolute when its values lie on a fixed bounded scale, depend on neither competing keys nor sequence length, require no sequence-length-dependent calibration, and can all be zero. To realize this notion, we introduce screening, whose explicit threshold transforms bounded query--key similarities into relevance values, enabling exact rejection, empty selection, and direct inspection on a common scale. In a controlled comparison of 12 attention mechanisms on a matched Transformer backbone, only screening maintains both low long-context perplexity and robust retrieval beyond the training context; notably, it does so without inference-time scaling. Building on screening, we introduce Multiscreen, a language-model architecture composed of parallel gated screening tiles. Multiscreen retains these long-context gains while achieving greater parameter efficiency, stronger general zero-shot downstream performa
    
[^301]: UltRAG：一种通用、简单、可扩展的知识图谱RAG方案

    UltRAG: a Universal Simple Scalable Recipe for Knowledge Graph RAG

    [https://arxiv.org/abs/2603.28773](https://arxiv.org/abs/2603.28773)

    UltRAG是一种无需训练的知识图谱RAG方案，通过结合LLM查询生成、完全归纳式神经查询执行器和LLM仲裁，在无需重训模型的情况下于KGQA任务上取得最先进结果，并支持Wikidata规模的超大规模图谱。

    

    大型语言模型（LLM）在用于语言生成时，经常生成看似自信但事实上不正确的内容（这种现象通常被称为“幻觉”）。检索增强生成（RAG）试图通过在知识语料库中检索信息并将其置于模型上下文窗口中，来减少事实性错误。虽然这种方法在文档结构化数据上已经相当成熟，但要将其适配到知识图谱（KG）上并不容易，尤其是对于那些需要在图上进行多节点/多跳推理的查询。我们提出了UltRAG，这是一种无需训练的KG-RAG方案，它结合了LLM查询生成、完全归纳式（inductive）的神经查询执行器以及LLM仲裁。这种开箱即用的组合在知识图谱问答（KGQA）任务上取得了最先进的结果，而无需重新训练LLM或执行器，同时使语言模型能够与Wikidata规模的图谱（1.16亿实体、16亿……）进行交互。

    arXiv:2603.28773v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) frequently generate confident yet factually incorrect content when used for language generation (a phenomenon often known as hallucination). Retrieval augmented generation (RAG) tries to reduce factual errors by identifying information in a knowledge corpus and putting it in the context window of the model. While this approach is well-established for document-structured data, it is non-trivial to adapt it for Knowledge Graphs (KGs), especially for queries that require multi-node/multi-hop reasoning on graphs. We introduce UltRAG, a training-free KG-RAG recipe that combines LLM query generation, a fully inductive neural query executor, and LLM arbitration. This off-the-shelf composition achieves state-of-the-art results on Knowledge Graph Question Answering (KGQA) tasks without retraining the LLM or executor, while enabling language models to interface with Wikidata-scale graphs (116M entities, 1.6B 
    
[^302]: RAWR：可验证领域中无需Rollout的奖励分配

    RAWR: Reward Assignment Without Rollouts in Verifiable Domains

    [https://arxiv.org/abs/2603.17815](https://arxiv.org/abs/2603.17815)

    提出MCNIG方法，利用净信息增益（NetIG）在可验证领域中自动标注推理步骤质量，无需昂贵的人工标注或计算密集的rollout即可训练出高性能的过程奖励模型。

    

    arXiv:2603.17815v2 公告类型：替换 摘要：在单个步骤的粒度上理解和评估大语言模型（LLM）的多步推理仍然是一个关键挑战。过程奖励模型（PRMs）通过为每个推理步骤打分提供了一种解决方案，实现了细粒度的监督并提升了可靠性。然而，训练这类模型需要昂贵的人工标注或计算开销极高的基于rollout的标注方法。为解决这一问题，我们提出了MCNIG，这是一种可扩展的方法，能够在任何可验证领域中自动标注单个推理步骤的质量。其步骤得分——净信息增益（NetIG）——通过比较最受支持的正确答案与最受支持的错误答案，对单参考信息增益（IG）进行了改进，即使对于代码和SQL这类长且结构化的输出（IG在此类场景下会失效）也能产生稳健的信号。我们证明了MCNIG产生的信号与人类对步骤质量的判断具有相关性，并利用MCNIG标注训练出了取得最佳平均性能的过程奖励模型。

    arXiv:2603.17815v2 Announce Type: replace  Abstract: Understanding and evaluating multi-step reasoning in LLMs at the level of individual steps remains a key challenge. Process reward models (PRMs) provide a solution by scoring each step, enabling fine-grained supervision and improved reliability. However, training them requires costly human annotation or computationally intensive rollout-based labeling. To solve this, we introduce MCNIG, a scalable method for automatically labeling the quality of individual reasoning steps in any verifiable domain. Its step score, net information gain (NetIG), improves upon single-reference information gain (IG) by comparing the most-supported correct answer against the most-supported incorrect one, yielding a robust signal even for long and structured outputs like code and SQL, where IG fails. We show that the signal produced by MCNIG correlates with human judgments of step quality, and we apply MCNIG labels to train PRMs that achieve the best averag
    
[^303]: SiDiaC-v.2.0：僧伽罗语历时语料库2.0版

    SiDiaC-v.2.0: Sinhala Diachronic Corpus Version 2.0

    [https://arxiv.org/abs/2603.10861](https://arxiv.org/abs/2603.10861)

    本文构建了迄今最大的僧伽罗语历时语料库SiDiaC-v.2.0，包含185部文学作品共22.9万词，时间跨度从公元5世纪至20世纪，并提供了按写作日期标注的子集，为僧伽罗语的历史语言学研究提供了重要资源。

    

    SiDiaC-v.2.0是迄今为止最大的综合性僧伽罗语历时语料库，按出版日期涵盖公元1800年至1955年，按写作日期涵盖公元5世纪至20世纪的历史时期。该语料库包含来自185部文学作品的22.9万词，这些作品经过了严格的筛选、预处理和版权合规检查，随后进行了大量的后处理。此外，其中59份文档（共计6.5万词）的子集根据其写作日期进行了标注。来自斯里兰卡国家图书馆的文本从SiDiaC-v.1.0的未过滤列表中选取，并使用Google Document AI OCR进行数字化，随后通过后处理纠正格式问题、处理语码混合现象、添加特殊标记并修复格式错误的标记。SiDiaC-v.2.0的构建参考了FarPaHC、SiDiaC-v.1.0和CCOHA等其他语料库的实践经验。

    arXiv:2603.10861v2 Announce Type: replace  Abstract: SiDiaC-v.2.0 is the largest comprehensive Sinhala Diachronic Corpus to date, covering a period from 1800 CE to 1955 CE in terms of publication dates, and a historical span from the 5th to the 20th century CE in terms of written dates. The corpus consists of 229k words across 185 literary works that underwent thorough filtering, preprocessing, and copyright compliance checks, followed by extensive post-processing. Additionally, a subset of 59 documents totalling 65k words was annotated based on their written dates. Texts from the National Library of Sri Lanka were selected from the SiDiaC-v.1.0 non-filtered list, which was digitised using Google Document AI OCR. This was followed by post-processing to correct formatting issues, address code-mixing, include special tokens, and fix malformed tokens. The construction of SiDiaC-v.2.0 was informed by practices from other corpora, such as FarPaHC, SiDiaC-v.1.0, and CCOHA. This was particula
    
[^304]: 在零视听资源场景下引导式构建视听语音识别

    Bootstrapping Audiovisual Speech Recognition in Zero-AV-Resource Scenarios

    [https://arxiv.org/abs/2603.08249](https://arxiv.org/abs/2603.08249)

    本研究证明在完全没有真实视听数据的零资源场景下，仅依靠700余小时合成说话人视频作为唯一视觉监督来微调AV-HuBERT，即可在加泰罗尼亚语基准上以更少的参数和训练数据实现接近SOTA的视听语音识别性能。

    

    视听语音识别（AVSR）通过结合声学与视觉线索，在具有挑战性的条件下提升语音转录的鲁棒性，但由于缺乏用于训练的标注视频语料库，大多数资源匮乏的语言至今无法使用这一技术。合成视觉数据已被证明是应对视听数据稀缺的有效增强策略。然而，对于加泰罗尼亚语这类语言而言，情况更为艰难，因为没有任何真实的视听数据可用于训练。在本研究中，我们探讨了能否在这种零视听资源设置下引导式构建AVSR，即仅使用合成视觉数据作为视觉监督的唯一来源。我们合成了超过700小时的说话人头部视频，并对预训练的AV-HuBERT模型进行微调。在一个人工标注的加泰罗尼亚语基准测试上，我们的模型以远少于SOTA ASR系统的参数量和训练数据，取得了接近最先进（SOTA）的性能。

    arXiv:2603.08249v2 Announce Type: replace-cross  Abstract: Audiovisual speech recognition (AVSR) combines acoustic and visual cues to improve transcription robustness under challenging conditions but remains out of reach for most under-resourced languages due to the lack of labeled video corpora for training. Synthetic visual data have been shown to be an effective augmentation strategy for addressing AV data scarcity. However, a more challenging scenario arises for languages such as Catalan, where no real audiovisual data are available for training.   In this study, we investigate whether AVSR can be bootstrapped in such a zero-AV-resource setting, using synthetic visual data as the sole source of visual supervision. We synthesize over 700 hours of talking-head video and fine-tune a pre-trained AV-HuBERT model. On a manually annotated Catalan benchmark, our model achieves near state-of-the-art (SOTA) performance with much fewer parameters and training data than SOTA ASR systems such a
    
[^305]: 迈向鲁棒的基于大语言模型的评判者：分类偏差评估与去偏优化

    Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing Optimization

    [https://arxiv.org/abs/2603.08091](https://arxiv.org/abs/2603.08091)

    提出JudgeBiasBench基准，通过4维度偏差分类体系和受控偏差注入流程系统量化LLM评判者的12种代表性偏差类型，揭示现有生成式和判别式评判者均存在显著且多样的偏差模式。

    

    基于大语言模型（LLM）的评判者被广泛应用于自动评估和奖励建模，但其判断往往受到评判偏差的影响。准确评估这些偏差对于确保基于LLM的评判者的可靠性至关重要。然而，现有研究通常仅在单一评判形式（生成式或判别式）下研究有限的偏差类型，缺乏全面的评估。为弥补这一空白，我们提出了JudgeBiasBench，一个用于系统量化基于LLM的评判者偏差的基准。JudgeBiasBench定义了跨4个维度的评判偏差分类体系，并通过受控的偏差注入流程构建了偏差增强的评估实例，涵盖12种代表性偏差类型。我们在生成式和判别式评判者上进行了广泛的实验，揭示了当前评判者表现出显著且多样的偏差模式，这些偏差往往会损害评估结果的可靠性。

    arXiv:2603.08091v3 Announce Type: replace  Abstract: Large language model (LLM)-based judges are widely adopted for automated evaluation and reward modeling, yet their judgments are often affected by judgment biases. Accurately evaluating these biases is essential for ensuring the reliability of LLM-based judges. However, existing studies typically investigate limited biases under a single judge formulation, either generative or discriminative, lacking a comprehensive evaluation. To bridge this gap, we propose JudgeBiasBench, a benchmark for systematically quantifying biases in LLM-based judges. JudgeBiasBench defines a taxonomy of judgment biases across 4 dimensions, and constructs bias-augmented evaluation instances through a controlled bias injection pipeline, covering 12 representative bias types. We conduct extensive experiments across both generative and discriminative judges, revealing that current judges exhibit significant and diverse bias patterns that often compromise the re
    
[^306]: SalamahBench：阿拉伯语语言模型的方言与危害类别层面安全评估

    SalamahBench: Dialect and Category Level Safety Evaluation of Arabic Language Models

    [https://arxiv.org/abs/2603.04410](https://arxiv.org/abs/2603.04410)

    本文提出 SalamahBench 基准，包含 8,270 条经人工验证的有害提示及其在现代标准阿拉伯语与五种方言下的共 49,620 对实例，并引入方言偏移和类别特定方言偏差两个指标，从方言和危害类别层面系统评估阿拉伯语语言模型的安全性。

    

    随着各方利益相关者试图利用阿拉伯语语言模型（ALMs），ALMs 的安全对齐在很大程度上仍未得到充分探索，这阻碍了其主流应用。现有的安全基准主要以英语为中心，并且仅在标准化形式下评估阿拉伯语，从而掩盖了阿拉伯语自然语言处理系统中细粒度的安全漏洞。本文提出了 SalamahBench，这是一个统一基准，包含 8,270 条经人工验证的有害提示，覆盖 ML Commons 危害类别，每条提示均以现代标准阿拉伯语（MSA）以及五种地区阿拉伯语变体（即埃及、叙利亚、沙特、黎巴嫩和摩洛哥方言）呈现，共计 49,620 对实例。为分析所得到的数据，我们引入了两个互补的指标：方言偏移，用于衡量模型在方言改写下安全性的总体变化；以及类别特定方言偏差，用于隔离出安全变化偏离整体趋势的危害类别。

    arXiv:2603.04410v3 Announce Type: replace-cross  Abstract: While different stakeholders are trying to leverage Arabic Language Models (ALMs), safety alignment in ALMs remains largely underexplored, hindering their mainstream adoption. Existing safety benchmarks are predominantly English-centric and evaluate Arabic only in its standardized form, obscuring fine-grained safety vulnerabilities in Arabic NLP systems. This paper introduces SalamahBench, a unified benchmark of 8{,}270 human-verified harmful prompts across ML Commons hazard categories, each rendered in Modern Standard Arabic (MSA) and five regional Arabic varieties, namely Egyptian, Syrian, Saudi, Lebanese, and Moroccan, for a total of 49{,}620 paired instances. To analyze the resulting data, we introduce two complementary metrics, namely Dialect Shift, which measures a model's aggregate change in safety under dialectal reformulation, and Category-Specific Dialect Deviation, which isolates harm categories whose change departs 
    
[^307]: 基于集值集函数的语法性动态转换理论模型

    A theoretical model of dynamical grammatical gender shifting based on set-valued set function

    [https://arxiv.org/abs/2603.03510](https://arxiv.org/abs/2603.03510)

    本文提出了一个以集值集函数形式化定义的“基于模板的模块化认知模型”，通过将词项与形态模板动态配对，揭示名词语法性转换的非线性底层规律。

    

    本研究考察名词的多样特征，既关注语义层面（如可数/不可数）的区别，也关注形态句法层面（如阳性/阴性）的区别。我们探讨了名词形态中性别标记在词与词之间的变异情况。语法性的转换是世界各地语言中普遍存在的现象。本研究旨在揭示支配词位变异的底层规律。为此，我们提出了一个新的计算组件，专门用于将词项与形态模板进行配对（例如，生成的词项-模板配对结果为：(funas, {N, +SG, -PL, -M, +F, -COL, +SING})，其拼写输出形式为：ða-funast ‘母牛’）。这一过程在形式上由“基于模板的模块化认知模型”加以表征。该模型由集值集函数 h : 𝒫(M) → 𝒫(M) 定义，用以预测词项到形态结构的非线性动态映射。

    arXiv:2603.03510v3 Announce Type: replace-cross  Abstract: This study investigates the diverse characteristics of nouns, focusing on both semantic (e.g., countable/uncountable) and morphosyntactic (e.g., masculine/feminine) distinctions. We explore inter-word variations for gender markers in noun morphology. Grammatical gender shift is a widespread phenomenon in languages around the world. The aim is to uncover the underlying patterns governing the variation of lexemes. To this end, we propose a new computational component dedicated to pairing items with morphological templates (e.g., the result of a generated item-template pair: (funas, $\{N, +SG, -PL, -M, +F, -COL, +SING\}$), with its spell-out form: $\eth$a-funast 'cow'). This process is formally represented by the Template-Based and Modular Cognitive model. This proposed model, defined by a set-valued set function $h : \mathscr{P}(M) \rightarrow \mathscr{P}(M)$, predicts the nonlinear dynamic mapping of lexical items onto morpholog
    
[^308]: 评估通用大语言模型智能体的测试时扩展

    Evaluating Test-Time Scaling of General LLM Agents

    [https://arxiv.org/abs/2602.18998](https://arxiv.org/abs/2602.18998)

    本文提出了一个统一覆盖搜索、编程、推理和工具使用的现实基准，系统研究了LLM智能体在序列扩展与并行扩展两个维度的测试时扩展行为，发现领先智能体从领域特定评估转向现实场景时性能显著下降，并通过细粒度扩展测试时计算刻画了性能上界。

    

    大语言模型智能体日益被期望作为通用系统来解决真实世界的用户请求，然而其在现实环境中的动态扩展行为仍然鲜为人知。本文系统地研究了LLM智能体的两个主要测试时扩展维度：通过延长交互实现的序列扩展，以及通过轨迹采样实现的并行扩展。我们首先提出了一个贴近现实的基准，为评估LLM智能体在搜索、编程、推理和工具使用等领域提供了一个统一框架，更忠实地反映了真实部署的异质性。对十个领先的LLM智能体的评估显示，当从特定领域的评估过渡到这一现实场景时，性能出现了显著下降。在此基础上，我们沿着细粒度的增量逐步扩展测试时计算，以刻画性能上界。

    arXiv:2602.18998v2 Announce Type: replace  Abstract: LLM agents are increasingly expected to operate as general-purpose systems that resolve real-world user requests, yet their dynamic scaling behavior in realistic environments remains poorly understood. In this paper, we systematically investigate two principal test-time scaling axes of LLM agents: sequential scaling through extended interaction and parallel scaling through trajectory sampling. We first introduce a realistic benchmark that provides one unified framework for evaluating LLM agents across search, coding, reasoning, and tool-use domains, more faithfully reflecting the heterogeneity of real-world deployments. Evaluating ten leading LLM agents reveals substantial performance degradation when transitioning from domain-specific evaluations to this realistic setting. Building on this foundation, we progressively scale test-time compute along fine-grained increments to characterize the performance upper bound. We find that neit
    
[^309]: 视觉虫洞：异构多智能体系统中的潜空间通信

    Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems

    [https://arxiv.org/abs/2602.15382](https://arxiv.org/abs/2602.15382)

    该论文提出“视觉虫洞”，将视觉-语言模型的视觉输入接口重新用于通信通道，通过通用视觉编解码器和共享参考空间，使冻结的异构多智能体之间能够以仅需O(N)组件的架构进行潜空间连续通信。

    

    异构多智能体系统通过统一的通信接口将具有不同能力的模型组合在一起。直接交换内部状态需要在模型特定的表示之间进行转换，并控制中间计算过程。我们提出了“视觉虫洞”，它将视觉-语言模型的视觉输入接口重新用于冻结的异构智能体之间的连续通信。通用视觉编解码器将每个发送方的潜空间推演编码为固定大小的消息，通过共享参考空间进行映射，并将接收到的消息解码到接收方的图像标记区间中。每模型的编解码器和仿射参考映射构成一个枢纽-辐射架构，对于N个模型仅需O(N)个组件。每个模型通过在锚文本上进行自蒸馏来独立学习其编解码器，而共享锚点的对齐使编解码器可在不同通信伙伴之间复用。在四个VLM家族、六种团队配置上（原文在此处截断）……

    arXiv:2602.15382v3 Announce Type: replace  Abstract: Heterogeneous multi-agent systems combine models with different capabilities through a common communication interface. Exchanging internal states directly requires translating between model-specific representations and controlling intermediate computation. We introduce the Vision Wormhole, which repurposes the visual input interface of Vision-Language Models (VLMs) for continuous communication between frozen heterogeneous agents. A Universal Visual Codec encodes each sender's latent rollout into a fixed-size message, maps it through a shared reference space, and decodes received messages into the receiver's image-token span. Per-model codecs and affine reference maps form a hub-and-spoke architecture with $O(N)$ components for $N$ models. Each model learns its codec independently through self-distillation on anchor texts, and shared-anchor alignment enables reuse across communication partners. Across four VLM families, six team confi
    
[^310]: 论大语言模型的校准：从响应到能力

    On Calibration of Large Language Models: From Response To Capability

    [https://arxiv.org/abs/2602.13540](https://arxiv.org/abs/2602.13540)

    该论文提出了“能力校准”这一新的评估框架，将大语言模型校准的研究焦点从单次响应的置信度转向查询级的整体置信度，并从理论和实证上证明其与传统的响应校准存在本质区别。

    

    准确的置信度估计对于大语言模型（LLM）的可靠使用至关重要。以往关于大语言模型校准的工作主要集中在响应级置信度上，即估计单个生成输出的正确性。然而，这种定义方式与许多实际应用场景并不匹配——在这些场景中，核心问题是模型有多大可能整体上解决某个查询。我们证明，这种不匹配源于现代大语言模型解码过程的随机性：在随机解码下，单次响应的正确性无法反映模型的底层真实能力。为解决这一问题，我们提出了能力校准，这是一个新的评估框架，用于衡量查询级置信度与模型在单个查询上的期望准确率之间的对齐程度。我们从形式上区分了能力校准（CC）与响应校准（RC），并证明二者在理论和实证上均存在差异。我们进一步表明，CC比RC更适合……

    arXiv:2602.13540v2 Announce Type: replace-cross  Abstract: Accurate confidence estimation is critical for reliable use of large language models (LLMs). Prior work on LLM calibration largely focuses on response-level confidence, which estimates the correctness of a single generated output. However, this formulation is misaligned with many practical settings where the central question is how likely a model is to solve a query overall. We show that this mismatch results from the stochastic nature of modern LLM decoding, under which single-response correctness fails to reflect underlying model capability. To address this issue, we introduce capability calibration, a new evaluation framework for measuring how well query-level confidence aligns with a model's expected accuracy on individual queries. We formally distinguish capability calibration (CC) from response calibration (RC) and show that the two differ both theoretically and empirically. We further show that CC is better suited than R
    
[^311]: 评估大语言模型行为倾向的对齐性

    Evaluating Alignment of Behavioral Dispositions in LLMs

    [https://arxiv.org/abs/2602.11328](https://arxiv.org/abs/2602.11328)

    提出STAR框架，将心理学问卷改编为现实的寻求建议场景，构建2.3万场景数据集评估25个LLM的行为倾向与人类的对齐程度，发现前沿模型在人类高共识时有15-20%的偏差，且在人类意见分歧时其建议多样性明显不足。

    

    随着人们越来越多地向大语言模型（LLM）寻求社交建议，理解它们在此类情境中的行为变得至关重要。在本工作中，我们聚焦于行为倾向，即塑造社交情境下回应的潜在倾向。我们提出了STAR，一个用于研究LLM所表达的倾向与人类倾向契合程度的框架。STAR建立在成熟的心理问卷基础上，将其条目改编为现实的寻求建议场景，因为自我报告可能无法迁移到实际的咨询行为中。利用STAR，我们构建了一个包含2.3万个场景的数据集，每个场景均由3名评估者验证，并由10名参与者标注偏好。在对25个LLM的评估中，我们发现：（1）当人类共识较高时，前沿模型在15-20%的情况下未能反映该共识，而较小模型的失败率显著更高；（2）当人类意见不一致时，LLM的建议多样性显著低于人类的选择，无论是在个体内部还是……

    arXiv:2602.11328v2 Announce Type: replace  Abstract: As people turn to LLMs for social advice, understanding their behavior in such contexts becomes essential. In this work, we focus on behavioral dispositions: the underlying tendencies that shape responses in social contexts. We introduce STAR, a framework for studying how closely the dispositions expressed by LLMs align with those of humans. STAR builds on established psychological questionnaires, adapting their items into realistic advice-seeking scenarios, as self-report may not transfer to actual advisory behavior. Using STAR, we construct a dataset of 23k scenarios, each validated by 3 raters and annotated with preferences from 10 participants. Across 25 LLMs, we find that (1) when human consensus is high, frontier models can fail to reflect it in 15-20% of cases, and smaller models fail at substantially higher rates; (2) when humans disagree, LLM recommendations are substantially less diverse than human choices, both within indi
    
[^312]: IESR：基于蒙特卡洛树搜索的大语言模型Text-to-SQL高效模块化推理

    IESR:Efficient MCTS-Based Modular Reasoning for Text-to-SQL with Large Language Models

    [https://arxiv.org/abs/2602.05385](https://arxiv.org/abs/2602.05385)

    该论文提出IESR框架，通过信息理解与模式链接、基于MCTS的多路径推理与多数投票机制、以及轨迹一致性验证模块三大创新，使轻量级大语言模型在Text-to-SQL任务上达到最先进性能，同时降低了企业部署成本。

    

    Text-to-SQL是一项关键的自然语言处理任务，它将自然语言问题映射为SQL查询，使用户能够直观地与基于Web的数据库进行交互。尽管当前方法在BIRD和Spider等基准测试中表现良好，但它们在复杂推理、领域知识和假设性查询方面仍然存在困难，且在企业部署中成本高昂。为了解决这些问题，我们提出了一个面向轻量级大语言模型的框架IESR（信息增强结构化推理）：(i) 利用大语言模型进行关键信息理解和模式链接，并将数学计算与SQL生成解耦；(ii) 集成了基于蒙特卡洛树搜索（MCTS）的多路径推理机制与多数投票策略；(iii) 引入了带有判别模型的轨迹一致性验证模块，以确保准确性和一致性。实验结果表明，IESR实现了最先进的（性能表现）。

    arXiv:2602.05385v2 Announce Type: replace  Abstract: Text-to-SQL is a key natural language processing task that maps natural language questions to SQL queries, enabling intuitive interaction with web-based databases. Although current methods perform well on benchmarks like BIRD and Spider, they struggle with complex reasoning, domain knowledge, and hypothetical queries, and remain costly in enterprise deployment. To address these issues, we propose a framework named IESR(Information Enhanced Structured Reasoning) for lightweight large language models: (i) leverages LLMs for key information understanding and schema linking, and decoupling mathematical computation and SQL generation, (ii) integrates a multi-path reasoning mechanism based on Monte Carlo Tree Search (MCTS) with majority voting, and (iii) introduces a trajectory consistency verification module with a discriminator model to ensure accuracy and consistency. Experimental results demonstrate that IESR achieves state-of-the-art 
    
[^313]: MGSM-Pro：一种用于鲁棒多语言数学推理评估的简单策略

    MGSM-Pro: A Simple Strategy for Robust Multilingual Mathematical Reasoning Evaluation

    [https://arxiv.org/abs/2601.21225](https://arxiv.org/abs/2601.21225)

    提出 MGSM-Pro 数据集，通过 GSM-Symbolic 方法为 MGSM 每道题生成五个变体实例，揭示了许多低资源语言在数字变化下性能大幅下降，且模型在高资源语言上的鲁棒性无法迁移到低资源语言。

    

    大语言模型在数学推理方面已取得显著进展。然而，面向多语言评估的基准测试开发在难度和时效性方面均落后于英文。最近，GSM-Symbolic 提供了有力证据，表明模型在同一问题的不同实例化版本上进行评估时性能方差较大，但该评估仅在英文上进行。本文提出 MGSM-Pro，即采用 GSM-Symbolic 方法对 MGSM 数据集的扩展。我们的数据集通过改变人名、数字和无关上下文，为每个 MGSM 问题提供五个实例化版本。在九种语言上的评估表明，当使用与原始测试集不同的数字实例化版本进行测试时，许多低资源语言出现大幅性能下降。我们进一步发现，模型在高资源语言（HRL）环境中的鲁棒性并不一定能迁移到低资源语言（LRL）。此外，Gemini 2.5 Flash 和 GPT-4.1 等专有模型……（原文摘要在此处截断）

    arXiv:2601.21225v4 Announce Type: replace-cross  Abstract: Large language models have made substantial progress in mathematical reasoning. However, benchmark development for multilingual evaluation has lagged behind English in both difficulty and recency. Recently, GSM-Symbolic showed a strong evidence of high variance when models are evaluated on different instantiations of the same question; however, the evaluation was conducted only in English. In this paper, we introduce MGSM-Pro, an extension of MGSM dataset with GSM-Symbolic approach. Our dataset provides five instantiations per MGSM question by varying names, digits and irrelevant context. Evaluations across nine languages reveal that many low-resource languages suffer large performance drops when tested on digit instantiations different from those in the original test set. We further find that models robustness in HRL setting do not necessarily translate to LRL. Moreover, proprietary models, such as Gemini 2.5 Flash and GPT-4.1
    
[^314]: 渐近通用对齐：一种基于测试时扩展的新型对齐框架

    Asymptotic Universal Alignment: A New Alignment Framework via Test-Time Scaling

    [https://arxiv.org/abs/2601.08777](https://arxiv.org/abs/2601.08777)

    该论文提出通过测试时扩展实现“渐近通用对齐”的新框架，证明生成 k 个候选回答的乘积策略能以最优且不可超越的速率 f(k)=k/(k+1) 逼近完美胜率，为处理用户偏好冲突提供了理论基础。

    

    让大型语言模型（LLM）能够服务于具有异构且可能相互冲突偏好的用户，是个性化与可信人工智能面临的核心挑战。我们通过测试时扩展形式化了一种理想化的通用对齐概念：对于每个提示，模型生成 k≥1 个候选回答，用户从中选择自己最喜欢的一个。我们提出了 (k, f(k))-鲁棒对齐，即要求该 k 输出模型相对于任何其他单输出模型具有 f(k) 的胜率；并进一步提出渐近通用对齐（U-alignment），即要求当 k→∞ 时 f(k)→1。我们的主要结果刻画了最优收敛速率：存在一族单输出策略，其 k 样本乘积策略能够以 f(k)=k/(k+1) 的速率实现 U-alignment，而且在一般情况下没有任何方法能够达到更快的速率。我们证明，包括从人类反馈中进行纳什学习（NLHF）在内的主流后训练方法可以……

    arXiv:2601.08777v2 Announce Type: replace-cross  Abstract: Aligning large language models (LLMs) to serve users with heterogeneous and potentially conflicting preferences is a central challenge for personalized and trustworthy AI. We formalize an ideal notion of universal alignment through test-time scaling: for each prompt, the model produces $k\ge 1$ candidate responses and a user selects their preferred one. We introduce $(k,f(k))$-robust alignment, which requires the $k$-output model to have win rate $f(k)$ against any other single-output model, and asymptotic universal alignment (U-alignment), which requires $f(k)\to 1$ as $k\to\infty$. Our main result characterizes the optimal convergence rate: there exists a family of single-output policies whose $k$-sample product policies achieve U-alignment at rate $f(k)=\frac{k}{k+1}$, and no method can achieve a faster rate in general.   We show that popular post-training methods, including Nash learning from human feedback (NLHF), can fund
    
[^315]: 大语言模型对顺序有多敏感？用于确定性结构重构的OrderProbe

    How Order-Sensitive Are LLMs? OrderProbe for Deterministic Structural Reconstruction

    [https://arxiv.org/abs/2601.08626](https://arxiv.org/abs/2601.08626)

    该论文提出了确定性基准OrderProbe，利用中日韩固定四字表达的唯一规范顺序来评估大语言模型从打乱输入中重构结构的能力，发现即使是最先进的模型，零样本恢复率也经常低于35%。

    

    大语言模型（LLMs）在语义理解方面表现出色，但其从打乱输入中重构内部结构的能力仍未得到充分探索。句子级恢复难以自动评估，因为打乱的句子往往存在多种有效的重排方式。我们提出了OrderProbe，一个用于结构重构的确定性基准，使用中文、日文和韩文中的固定四字表达，这些表达具有唯一的规范顺序，从而支持精确匹配评分。我们进一步提出了一个诊断框架，在恢复准确率之外对模型进行评估，包括语义准确性、逻辑有效性、结构一致性、鲁棒性和信息密度。对十二个广泛使用的大语言模型的实验表明，即使对最先进的系统而言，结构重构仍然困难：零样本恢复率经常低于35%。我们还观察到面向意义的生成与（此处摘要被截断）之间存在一致的差距。

    arXiv:2601.08626v4 Announce Type: replace  Abstract: Large language models (LLMs) excel at semantic understanding, yet their ability to reconstruct internal structure from scrambled inputs remains underexplored. Sentence-level restoration is difficult to evaluate automatically because scrambled sentences often admit multiple valid reorderings. We introduce OrderProbe, a deterministic benchmark for structural reconstruction using fixed four-character expressions in Chinese, Japanese, and Korean, which have a unique canonical order and thus support exact-match scoring. We further propose a diagnostic framework that evaluates models beyond recovery accuracy, including Semantic Accuracy, Logical Validity, Structural Consistency, Robustness, and Information Density. Experiments on twelve widely used LLMs show that structural reconstruction remains difficult even for frontier systems: zero-shot recovery frequently falls below 35%. We also observe a consistent gap between meaning-oriented gen
    
[^316]: MAPLE：基于短语级证据的医学方面摘要

    MAPLE: Medical Aspect-Based Summarization with Phrase-Level Evidence

    [https://arxiv.org/abs/2601.03418](https://arxiv.org/abs/2601.03418)

    该论文提出MAPLE基准，首次将医学方面摘要中每个论断的证据粒度细化到短语级，并引入解耦评估框架，分别衡量内容质量、可追溯性和可定位性。

    

    可信的临床摘要生成要求每个论断都能追溯到其证据来源，然而现有的归因方法通常只能定位到句子或文档级别，这使得临床医生不得不浏览周围文本以寻找真正重要的少数几个词。我们认为，归因的单位应当与验证的单位保持一致：即读者目光必须落到的精确短语。我们提出了MAPLE（Medical Aspect-Based Summarization with Phrase-Level Evidence），这是一个经人工标注的基准数据集，它将摘要中的每个论断同时锚定到所引用的句子以及句子内部的贡献性短语。MAPLE涵盖152篇随机对照试验（RCT）摘要和16个临床驱动的方面，包含1,799个具有两级证据的基于方面的摘要。我们进一步引入了一个解耦的评估框架，分别对内容、可追溯性和可定位性进行评分，并提供了一个衡量临床医生为验证某个论断所需检查的源文本量的代理指标。

    arXiv:2601.03418v3 Announce Type: replace  Abstract: Trustworthy clinical summarization requires every claim to be traceable to its evidence, yet existing attribution often resolves only to the sentence or document, leaving clinicians to scan surrounding text for the few words that matter. We argue that the unit of attribution should match the unit of verification: the precise phrase the reader's eye must land on. We present MAPLE (Medical Aspect-Based Summarization with Phrase-Level Evidence), a human-annotated benchmark that grounds each summarized claim in both cited sentences and contributory phrases within them. Spanning 152 randomized controlled trial (RCT) abstracts and 16 clinically motivated aspects, MAPLE comprises 1,799 aspect-based summaries with two-level evidence. We further introduce a decoupled evaluation framework that separately scores content, traceability, and locatability, together with a proxy for the amount of source text a clinician must inspect to verify a clai
    
[^317]: 块稀疏Flash注意力

    Block Sparse Flash Attention

    [https://arxiv.org/abs/2512.07011](https://arxiv.org/abs/2512.07011)

    提出块稀疏Flash注意力（BSFA），通过精确计算查询-键相似度选取最重要的值块并与校准阈值比较来跳过约50%的计算和内存传输，作为免训练的即插即用替代方案加速长上下文推理且不损失模型质量。

    

    现代大语言模型在推理和多文档任务中越来越需要长上下文，但注意力的二次方复杂度造成了严重的计算瓶颈。我们提出了块稀疏Flash注意力（BSFA），这是一种即插即用的替代方案，能够在保持模型质量的同时加速长上下文推理。与在计算分数之前预测重要性的方法不同，BSFA通过计算精确的查询-键相似度来为每个查询选择最重要的前k个值块。通过将每个块的最大分数与校准阈值进行比较，我们跳过了约50%的被剪枝块的计算和内存传输。这种免训练方法只需在一个小数据集上进行一次阈值校准，即可学习每层和每个注意力头的注意力分数分布。我们提供了CUDA内核实现，可以作为FlashAttention的即插即用替代品。在Llama-3.1-8B上，BSFA实现了显著的加速效果。

    arXiv:2512.07011v2 Announce Type: replace-cross  Abstract: Modern large language models increasingly require long contexts for reasoning and multi-document tasks, but attention's quadratic complexity creates a severe computational bottleneck. We present Block Sparse Flash Attention (BSFA), a drop-in replacement that accelerates long-context inference while preserving model quality. Unlike methods that predict importance before computing scores, BSFA computes exact query-key similarities to select the top-k most important value blocks for each query. By comparing per-block maximum scores against calibrated thresholds, we skip approximately 50% of the computation and memory transfers for pruned blocks. Our training-free approach requires only a one-time threshold calibration on a small dataset to learn the per-layer and per-head attention score distributions. We provide a CUDA kernel implementation that can be used as a drop-in replacement for FlashAttention. On Llama-3.1-8B, BSFA achiev
    
[^318]: 面向时序表格推理的证据引导式模式规范化

    Evidence-Guided Schema Normalization for Temporal Tabular Reasoning

    [https://arxiv.org/abs/2512.00329](https://arxiv.org/abs/2512.00329)

    该研究将时序表格问答重构为自动化知识库构建任务，并通过受控交叉实验证明模式设计质量对问答准确率的影响远大于查询模型的选择（分别解释79.5%和1.6%的EM方差），据此提炼出模式设计原则。

    

    arXiv:2512.00329v2 公告类型：replace-cross 摘要：在不断演化的半结构化表格上进行时序推理对当前的问答系统构成了挑战。我们提出一种方法，将该任务重新构建为自动化知识库构建：（1）通过提示大语言模型（LLM）从维基百科信息框时间线中合成符合第三范式（3NF）的关系模式，（2）填充该模式以获得可查询的数据库，（3）生成并执行针对该数据库的SQL查询，并以问答准确率作为所构建知识库的外在评估手段。在一个由三种模式生成器与六种查询模型交叉组合的受控实验网格中，模式来源解释了精确匹配（EM）方差的79.5%，而查询模型仅解释1.6%：替换模式及其派生的提示框架会使EM偏移14.7至20.0个百分点，而在固定模式下替换查询模型仅使EM偏移4.4至12.1个百分点。基于这些证据，我们提炼出三条候选的模式设计原则：平衡规范化……

    arXiv:2512.00329v2 Announce Type: replace-cross  Abstract: Temporal reasoning over evolving semi-structured tables poses a challenge to current QA systems. We propose an approach that recasts the task as automated knowledge base construction: (1) prompting an LLM to synthesize a 3NF-compliant relational schema from Wikipedia infobox timelines, (2) populating the schema to obtain a queryable database, and (3) generating and executing SQL queries against it, with QA accuracy serving as an extrinsic evaluation of the constructed knowledge base. In a controlled grid of three schema generators crossed with six query models, the schema source accounts for 79.5% of the exact match (EM) variance against 1.6% for the query model: replacing the schema, and the prompt scaffolding derived from it, shifts EM by 14.7 to 20.0 points, whereas replacing the query model under a fixed schema shifts it by 4.4 to 12.1. From this evidence, we distill three candidate schema-design principles: balanced normal
    
[^319]: 超越语义：时间偏差如何塑造Transformer与状态空间模型中的检索

    Beyond Semantics: How Temporal Biases Shape Retrieval in Transformer and State-Space Models

    [https://arxiv.org/abs/2510.22752](https://arxiv.org/abs/2510.22752)

    研究发现，Transformer与状态空间模型在上下文学习中的检索不仅受语义驱动，还存在显著的时间位置偏差——模型更倾向于关注序列开头或末尾附近的重复token，类似于人类情景记忆中的时间分离机制。

    

    上下文学习由时间和语义关系共同支配，决定着大语言模型（LLMs）检索上下文信息的方式。类似于人类情景记忆——其中特定事件的检索依赖于对不同时间发生事件的区分——本工作探究了各种预训练LLM（包括Transformer和状态空间模型）区分并检索时间上相互分离事件的能力。具体而言，我们向模型输入包含同一token多次出现、且该token在序列末尾再次出现的序列。通过固定这些重复token的位置并随机排列所有其他token，我们消除了语义混淆因素，从而分离出时间因素对下一token预测的影响。在多样化的序列中，模型一致地将最高概率赋予重复token之后的token，但明显偏向于位于序列开头或末尾附近的那些token（原文在此处截断）。

    arXiv:2510.22752v2 Announce Type: replace-cross  Abstract: In-context learning is governed by both temporal and semantic relationships, shaping how Large Language Models (LLMs) retrieve contextual information. Analogous to human episodic memory, where the retrieval of specific events is enabled by separating events that happened at different times, this work probes the ability of various pretrained LLMs, including transformer and state-space models, to differentiate and retrieve temporally separated events. Specifically, we prompted models with sequences containing multiple presentations of the same token, which reappears at the sequence end. By fixing the positions of these repeated tokens and permuting all others, we removed semantic confounds and isolated temporal effects on next-token prediction. Across diverse sequences, models consistently placed the highest probabilities on tokens following a repeated token, but with a notable bias for those nearest the beginning or end of the i
    
[^320]: POET：面向增强文本到图像生成的偏好优化

    POET: Preference Optimization for Enhanced Text-to-Image Generation

    [https://arxiv.org/abs/2510.12041](https://arxiv.org/abs/2510.12041)

    POET提出了一种利用大语言模型自动重写用户提示词的框架，通过复合奖励系统和迭代式DPO训练，从多模态反馈中学习模型偏好的提示词结构，从而在无需昂贵监督微调数据的情况下提升文本到图像生成的质量。

    

    文本到图像（T2I）生成的最新进展已取得令人瞩目的成果，然而由于用户提示词与其描述性训练文本之间存在分布差距，现有模型在处理简单或描述不充分的用户提示词时常常表现不佳。这经常导致图文对齐、美学效果和整体视觉质量欠佳。为了弥合这一差距，我们提出了POET（面向增强文本到图像生成的偏好优化），这是一个自动化提示词重写框架，利用大型语言模型（LLM）在将用户输入送入冻结的T2I骨干模型之前对其进行优化。POET引入了精心设计的复合奖励系统和迭代式直接偏好优化（DPO）训练流程，使重写器能够直接从多模态反馈中学习模型偏好的提示词结构，而无需昂贵的高质量监督微调（SFT）数据。大量评估...

    arXiv:2510.12041v3 Announce Type: replace  Abstract: Recent advances in text-to-image (T2I) generation have achieved impressive results, yet existing models often struggle with simple or underspecified user prompts due to a distributional gap with their descriptive training captions. This frequently leads to suboptimal image-text alignment, aesthetics, and overall visual quality. To bridge this gap, we propose POET (\textbf{P}reference \textbf{O}ptimization for \textbf{E}nhanced \textbf{T}ext-to-Image generation), an automated prompt rewriting framework that leverages large language models (LLMs) to refine user inputs before feeding them into frozen T2I backbones. POET introduces a carefully designed composite reward system and an iterative Direct Preference Optimization (DPO) training pipeline, enabling the rewriter to learn model-preferred prompt structures directly from multimodal feedback without requiring costly high-quality supervised fine-tuning (SFT) data. Extensive evaluations
    
[^321]: TagPR：面向大语言模型个性化推理的标签引导过程监督

    TagPR: Tag-Guided Process Supervision for Personalization Reasoning in Large Language Models

    [https://arxiv.org/abs/2509.23140](https://arxiv.org/abs/2509.23140)

    TagPR通过在推理过程中添加语义标签实现逐步的过程监督，并结合基于用户嵌入的个性化奖励模型进行多阶段强化学习，显著提升了大语言模型在个性化任务上的推理能力。

    

    近期的进展已使大语言模型具备了令人瞩目的通用推理能力。然而，这些推理模型在个性化任务上的表现往往不如非推理模型。虽然一些方法使用基于结果的强化学习（RL）来提升个性化推理，但它们未能对推理过程进行监督。因此，模型可能通过有缺陷的推理链得出正确答案，这限制了进一步的提升。为解决这一问题，我们提出了TagPR——一个新颖的框架，它通过在推理过程中添加语义标签来提供逐步引导。TagPR首先自动生成一个结构化的、带标签的数据集用于监督微调（SFT）。随后，它采用由复合奖励信号引导的多阶段强化学习过程，该信号将基于标签的过程监督与一种新颖的结合用户嵌入的个性化奖励模型（Personalization Reward Model with User Embeddings）相结合，以实现与用户特定逻辑的细粒度对齐。在公开的LaMP数据集上进行的大量实验……（摘要不完整）

    arXiv:2509.23140v2 Announce Type: replace  Abstract: Recent advancements have endowed Large Language Models with impressive general reasoning capabilities. However, these reasoning models often perform worse than non-reasoning models on personalization tasks. While some methods use outcome-based RL to improve personalization reasoning, they fail to supervise the reasoning process. As a result, models may reach correct answers through flawed reasoning chains, limiting further improvement. To address this, we propose TagPR, a novel framework that adds semantic tags to the reasoning process for step-by-step guidance. TagPR first automatically generates a structured, tagged dataset for Supervised Fine-Tuning. It then employs a multi-stage RL process guided by a composite reward signal, which integrates tag-based process supervision with a novel Personalization Reward Model with User Embeddings to achieve fine-grained alignment with user-specific logic. Extensive experiments on public LaMP,
    
[^322]: 谄媚行为并非单一事物：大语言模型中谄媚行为的因果分离

    Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs

    [https://arxiv.org/abs/2509.21305](https://arxiv.org/abs/2509.21305)

    该论文首次将大语言模型的谄媚行为因果地分解为谄媚性附和与谄媚性赞扬两种独立机制，证明它们在潜空间中沿不同线性方向编码，且可被独立调控而互不干扰，并在不同模型家族和规模上保持一致。

    

    大语言模型（LLMs）常常表现出谄媚行为——例如过度附和或奉承用户——但尚不清楚这些行为究竟源自单一机制还是多个不同的过程。我们将谄媚分解为谄媚性附和与谄媚性赞扬，并将两者与真正的认同进行对比。通过在多个模型和数据集上使用均值差方向、激活添加和子空间几何方法，我们证明：（1）这三种行为在潜空间中沿着不同的线性方向进行编码；（2）每种行为可以被独立地放大或抑制，而不影响其他行为；（3）它们的表征结构在不同模型家族和规模上保持一致。这些结果表明，谄媚行为对应于独特的、可独立调控的表征。

    arXiv:2509.21305v4 Announce Type: replace  Abstract: Large language models (LLMs) often exhibit sycophantic behaviors -- such as excessive agreement with or flattery of the user -- but it is unclear whether these behaviors arise from a single mechanism or multiple distinct processes. We decompose sycophancy into sycophantic agreement and sycophantic praise, contrasting both with genuine agreement. Using difference-in-means directions, activation additions, and subspace geometry across multiple models and datasets, we show that: (1) the three behaviors are encoded along distinct linear directions in latent space; (2) each behavior can be independently amplified or suppressed without affecting the others; and (3) their representational structure is consistent across model families and scales. These results suggest that sycophantic behaviors correspond to distinct, independently steerable representations.
    
[^323]: 从模拟搜救任务中的通信预测团队绩效

    Predicting Team Performance from Communications in Simulated Search-and-Rescue

    [https://arxiv.org/abs/2503.03791](https://arxiv.org/abs/2503.03791)

    该研究通过分析模拟搜救任务中的对话数据，运用主题建模和聚类方法识别团队特质，证明这些从通信中推断出的个体特质和团队动态能够有效预测团队绩效的差异。

    

    理解个体特质如何影响团队绩效是有价值的，但这些特质并不总是能够被直接观察到。先前的研究已经从行为数据中推断出信任等特质。我们通过分析对话数据来识别团队特质及其与团队协作结果的关联。利用基于Minecraft的搜救实验的对话转录文本，我们应用主题建模和聚类方法来揭示关键的互动模式。我们的研究结果表明，团队协作结果的差异可以通过这些推断来解释，其中个体特质和团队动态提供了不同程度的预测能力。

    arXiv:2503.03791v1 Announce Type: cross  Abstract: Understanding how individual traits influence team performance is valuable, but these traits are not always directly observable. Prior research has inferred traits like trust from behavioral data. We analyze conversational data to identify team traits and their correlation with teaming outcomes. Using transcripts from a Minecraft-based search-and-rescue experiment, we apply topic modeling and clustering to uncover key interaction patterns. Our findings show that variations in teaming outcomes can be explained through these inferences, with different levels of predictive power derived from individual traits and team dynamics.
    
[^324]: 基于两阶段强化学习智能体的大语言模型集成动态优化

    Dynamic Optimizations of LLM Ensembles with Two-Stage Reinforcement Learning Agents

    [https://arxiv.org/abs/2502.04492](https://arxiv.org/abs/2502.04492)

    本文提出RL-Focal两阶段强化学习框架：第一阶段的决策者智能体通过最大化错误多样性与推理性能，为下游任务从N个大语言模型中动态选择小规模集成；第二阶段的融合智能体解决模型间推理冲突并自适应融合所选模型，实现大语言模型集成的动态优化。

    

    大语言模型（LLM）的进步及其易用性引发了人们对多智能体强化学习的重新关注，将其作为动态变化环境中鲁棒且自适应的框架。本文介绍了RL-Focal，一个用于路由和集成大语言模型的两阶段强化学习智能体框架。第一，我们开发了决策者强化学习智能体，它通过迭代更新任务自适应奖励和策略，学习为来自用户定义下游任务 i 的传入查询，在 N 个大语言模型中动态选择一个小规模的集成（m_i，其中 m_i ≪ N），同时最大化所选集成的错误多样性和推理性能。第二，为了实现对动态选择的大语言模型的有效融合，我们开发了第二阶段融合强化学习智能体，它学习解决来自不同大语言模型的推理冲突，并动态适应决策者智能体为不同下游任务所组成的不同集成团队。

    arXiv:2502.04492v3 Announce Type: replace  Abstract: The advancement of LLMs and their accessibility have triggered renewed interest in multi-agent reinforcement learning as robust and adaptive frameworks for dynamically changing environments. This paper introduces \texttt{RL-Focal}, a two-stage RL agent framework that routes and ensembles LLMs. \textit{First}, we develop the Decider RL-agent, which learns to dynamically select an ensemble of small size ($m_i$) among $N$ LLMs ($m_i \ll N$) for incoming queries from a user-defined downstream task $i$, by maximizing both error-diversity and reasoning-performance of the selected ensemble through iterative updates of task-adaptive rewards and policy. \textit{Second}, to enable effective fusion of dynamically selected LLMs, we develop the stage-2 Fusion RL-agent, which learns to resolve reasoning conflicts from different LLMs and dynamically adapt to different ensemble teams composed by the Decider Agent for different downstream tasks. {\em
    
[^325]: BadRAG：识别大语言模型检索增强生成中的漏洞

    BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models

    [https://arxiv.org/abs/2406.00083](https://arxiv.org/abs/2406.00083)

    BadRAG揭示了一种针对检索增强生成系统的新型攻击：攻击者通过向知识库注入恶意文段，当用户查询包含特定触发词时即可操纵系统响应，且无需修改用户输入或模型权重。

    

    检索增强生成（RAG）通过从外部知识库检索相关信息来增强大语言模型（LLM），从而提供更准确、更具上下文感知且更加及时的回答。然而，这种对外部知识的依赖引入了重大的安全漏洞，因为许多RAG系统（例如Google搜索）依赖于庞大且未经清洗的数据存储库（例如Reddit）。在本文中，我们揭示了一种新型威胁：攻击者通过向RAG系统的知识库注入恶意文段来操纵系统的响应。当用户的查询包含攻击者指定的触发词时，RAG会检索并引用这些恶意文段，使攻击者能够在不修改用户输入或RAG权重的情况下操纵响应。BadRAG分为两个阶段：（i）对恶意文段进行优化，使其仅在用户查询中出现触发词时才会被检索；（ii）这些恶意文段……

    arXiv:2406.00083v3 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant information from external knowledge bases to provide more accurate, contextually informed, and up-to-date responses. However, this reliance on external knowledge introduces significant security vulnerabilities, as many RAG systems (e.g., Google Search) rely on large and unsanitized data repositories (e.g., Reddit). In this paper, we unveil a novel threat in which attackers steer the RAG system's response by injecting malicious passages into its knowledge base. When a user's query contains attacker-specified trigger words, the RAG retrieves and refers to these malicious passages, enabling the attacker to steer the response without altering the user input or modifying the RAG weights. BadRAG operates in two phases: (i) malicious passages are optimized to be retrieved exclusively when trigger words appear in user queries; (ii) these p
    
[^326]: MONOVAB: 用于孟加拉语多标签情感检测的注释语料库

    MONOVAB : An Annotated Corpus for Bangla Multi-label Emotion Detection. (arXiv:2309.15670v1 [cs.LG])

    [http://arxiv.org/abs/2309.15670](http://arxiv.org/abs/2309.15670)

    这个研究构建了一个基于孟加拉语的注释语料库，用于多标签情感检测。通过使用基于上下文的方法以及BERT模型，填补了这一学科领域的空白。

    

    近年来，情感分析(SA)和情感识别(ER)在孟加拉语中越来越流行，孟加拉语是世界上第七大使用人数最多的语言。然而，孟加拉语的结构复杂，这使得准确提取情绪变得困难。在这个研究领域中，已经采用了一些不同的方法，例如提取积极和消极情感以及多类情绪。然而，在这种语言中提取多种情绪几乎是未开发的领域，它涉及基于一段文本识别出多种情感。因此，本研究展示了一种基于从Facebook上抓取的数据构建注释语料库的详细方法，以填补这个学科领域的空白，克服挑战。为了使这种注释更有成果，采用了基于上下文的方法。转换器中的双向编码器表示(BERT)。

    In recent years, Sentiment Analysis (SA) and Emotion Recognition (ER) have been increasingly popular in the Bangla language, which is the seventh most spoken language throughout the entire world. However, the language is structurally complicated, which makes this field arduous to extract emotions in an accurate manner. Several distinct approaches such as the extraction of positive and negative sentiments as well as multiclass emotions, have been implemented in this field of study. Nevertheless, the extraction of multiple sentiments is an almost untouched area in this language. Which involves identifying several feelings based on a single piece of text. Therefore, this study demonstrates a thorough method for constructing an annotated corpus based on scrapped data from Facebook to bridge the gaps in this subject area to overcome the challenges. To make this annotation more fruitful, the context-based approach has been used. Bidirectional Encoder Representations from Transformers (BERT),
    

