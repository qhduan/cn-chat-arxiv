# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ToolSmith: An Automatic Tool-Generation Framework for Multimodal Search Agents](https://arxiv.org/abs/2611.31198) | 该论文提出ToolSmith自动工具生成框架，通过在精简的基础工具库上自动生长并复用工具模块，在域内和域外基准上持续提升多模态搜索代理的性能，且模型规模越大收益越明显。 |
| [^2] | [CO-GRAFT: Attributing Reinforcement Learning Gains in Search Agents via Role-Level Policy Grafting](https://arxiv.org/abs/2611.31185) | 提出CO-GRAFT框架，通过角色级策略嫁接与合作博弈的Shapley值分解，将强化学习为搜索智能体带来的整体准确率提升归因到中间推理、搜索决策、查询生成和答案提取四个功能角色上，并利用Möbius交互项区分角色独立改进与多角色耦合效应。 |
| [^3] | [Precision-Aware Learning Rate Transfer Across Block Count in LLM Post-Training](https://arxiv.org/abs/2611.31166) | 本文提出 TipSweep 方法，仅需训练最后极少量的 Transformer 块即可搜索最优学习率，并借助依赖数值精度的缩放规则（如 FP32 下按 $k^{-1}$ 缩放）将其迁移到训练更多块乃至完整模型的场景，从而大幅降低大语言模型后训练中学习率调优的计算与内存开销。 |
| [^4] | [Deneb-Vritra: Descriptions, Edges and Names Examined by Blocking Via Restoring Token Representations and Attention](https://arxiv.org/abs/2611.31139) | 本研究在 421M 参数的多项选择问答编码器 Laya 中定位了模型依赖选项名称而非描述作答的“名称捷径”发生的网络位置，并证明在第 14 至 17 层阻断标记词元对其名称词元的注意力可将捷径行为从 67% 降至 16%，同时保持 90% 的原始准确率。 |
| [^5] | [Serialization and the Character-System: Measuring Narrative Concentration in French Fiction (1800--1940)](https://arxiv.org/abs/2611.31133) | 该研究将Woloch的“人物系统”理论操作化为人物网络PageRank分布的基尼系数，通过分析1800至1940年间的12,011部法国小说，首次量化揭示了叙事集中度随时间呈U形演变的历史轨迹。 |
| [^6] | [Attention Temperature Confounds Zero-Shot Context Extension](https://arxiv.org/abs/2611.31113) | 稠密基线的注意力温度校准状态会系统性歪曲零样本上下文扩展方法的评估结果，尤其在纯 bf16 训练中基线失准高达 37-58%，使块稀疏训练、注意力 dropout 和 ALiBi 等方法显得比实际更好，而校准后这些优势大多消失。 |
| [^7] | [Look Back, Draft Ahead: Corrective Prefix Attention for Parallel Speculative Decoding](https://arxiv.org/abs/2611.31112) | 本文提出纠正性前缀注意力（CPA），通过让并行草稿中的每个候选token关注块内先前的选择并结合自回归监督，大幅提升并行推测解码的token接受率，在Ascend NPU上实现了最高3.37倍的加速。 |
| [^8] | [Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals](https://arxiv.org/abs/2611.31094) | LoopRes通过让每个循环层借助可训练的循环感知键直接关注早期循环的输出（并限制在残差窗口内），使预训练大语言模型无需从头训练即可将循环结构转化为显著且一致的增益。 |
| [^9] | [UMAMI: A Multimodal Benchmark Dataset for Evidence-Grounded Dietary Decisions](https://arxiv.org/abs/2611.31068) | 该论文提出UMAMI多模态基准数据集，首次将食物图像、营养成分与个人健康及摄入记录关联起来，用于系统评估基于证据的膳食决策中的关键挑战。 |
| [^10] | [DKnown-Jev-2: Article-Level Answer-Evidence Classification and Efficient Inference with a Pre-trained Language Model](https://arxiv.org/abs/2611.31056) | 提出DKnown-Jev-2文章级答案证据分类器，通过联合编码问题条件、文章元数据与段落内容，并利用指针式分类头在单次前向传播中同时预测答案证据与非答案证据，在972个人工标注实例上取得91.77%准确率和91.28%宏F1，优于Jev基线。 |
| [^11] | [CAST: A Deployed Speech-to-Text Cascade for Closed Captioning of Entertainment Television](https://arxiv.org/abs/2611.31055) | 本文提出并实际部署了CAST电视隐藏式字幕语音转文本级联系统，其核心创新在于采用分层语音活动检测（通过语义与声学保留检查重审被拒片段，以捕获微弱语音并抑制静音引发的幻觉与重复），并将冻结的Whisper large-v3与分割、语言识别、歌词路由及针对性幻觉检测修复集成于单块NVIDIA L4 GPU之上。 |
| [^12] | [Parsing Visual Latents for Multimodal Spatial Reasoning](https://arxiv.org/abs/2611.31021) | 提出了无需训练的 LatentParse 框架，将 CLIP、DINOv3 和 VGGT 等预训练视觉编码器的潜在特征解析为显式空间证据（定位物体、跨视角追踪、测量视角间运动），从而帮助 VLM 进行跨视角空间推理。 |
| [^13] | [Can Latent Reasoning Drive LLM Outputs Without a Trace?](https://arxiv.org/abs/2611.31015) | 本研究证明简单的辅助损失可将模型的潜在推理深度提升至少5倍，使模型能在输出看似正常的思维链的同时基于隐藏目标进行潜在多步推理，这威胁到了依赖思维链监测模型不对齐行为的安全性方案。 |
| [^14] | [RoboUniverse-10K: Toward the Scaling Era of Agentic Robotics via Real-to-Sim Environments](https://arxiv.org/abs/2611.31002) | 该论文提出了RoboUniverse-10K——通过自动化Real2Sim与质量保证流程从真实视频构建的10,807个可交互、可重置、可验证的物理环境数据集，并配套专家验证的零样本基准RoboUniverse-Verified，为智能体机器人迈向规模化后训练与评估时代奠定基础。 |
| [^15] | [CART: Clustering-based Analysis of Recurring narratives for Taxonomy construction](https://arxiv.org/abs/2611.30993) | CART提出了一种无需预定义类别的计算流水线，将无监督学习与大语言模型叙事提取相结合，从议会话语中自动提取政治叙事并构建数据驱动的分类体系，在希腊议会关于2023年坦皮火车事故的辩论中识别出十类重复出现的叙事家族。 |
| [^16] | [Beyond Self-Consistency: Recursive Self-Improvement at Test Time](https://arxiv.org/abs/2611.30965) | 该论文提出了TT-RSI无标签框架，让模型在测试时以先前完整解答为参考反复迭代改进解题过程，从而超越传统基于多数投票的自一致性方法，实现大语言模型的测试时递归自我改进。 |
| [^17] | [LimeTree: Linear-Memory Graph Neural Networks over Tree-Graphs for Resource-Efficient Language Modeling and Context Extrapolation](https://arxiv.org/abs/2611.30934) | LimeTree是一种内存与计算均为线性复杂度的树图语言模型架构，通过在语言建模训练中集成KV检索任务，仅用极低的计算成本（4块GPU训练6小时）便实现了19的低困惑度，并能将上下文从512外推至16,384个token且保持100%的检索准确率。 |
| [^18] | [PRIMEAPE: A Multilingual Benchmark for Fine-Grained Meta-Evaluation of Judge Models](https://arxiv.org/abs/2611.30925) | 提出了 PRIMEAPE 多语言基准，利用七种语言、2,205 个响应对和 11k 条人类标注对十二个评判模型的裁决与理由进行细粒度元评估，揭示总体性能掩盖了跨语言与跨标准的差异，并暴露出位置、长度和模型来源偏好等系统性偏见。 |
| [^19] | [Semantic-aware Decoding for Diverse Text-to-SQL under Schema Ambiguity](https://arxiv.org/abs/2611.30873) | 提出了一种无需训练、模型无关的语义感知Logit调整（SLA）解码方法，通过基于模式元素间语义相似性调整token logits来促进多样化SQL生成，在显著降低推理成本的同时持续提升解释覆盖率。 |
| [^20] | [SARA: An Auditor-in-the-Loop LLM Recommender for Sustainability Reports](https://arxiv.org/abs/2611.30839) | SARA是一种审计师在环的LLM推荐系统，以Word插件形式帮助审计师将监管要求匹配到可持续发展报告的相关段落并预测合规性，通过主题预测过滤将匹配成本降低约六倍、运行时间减少42%。 |
| [^21] | [Task-Dependent Linguistic Measures in Aphasia Discourse: A Comparative Analysis of Narrative, Descriptive, and Procedural Speech](https://arxiv.org/abs/2611.30826) | 该研究通过对61名慢性卒中后失语症患者在同一会话中完成的三种话语任务（程序性、描述性、叙事性）进行10项语言学测量指标的比较分析，证明话语任务类型本身会显著影响失语症的转录文本测量结果，因此不同任务得到的数值不能直接混用比较。 |
| [^22] | [Ask Again: Answer-List Instability in Generative Engines](https://arxiv.org/abs/2611.30819) | 该研究通过对6万条生成式引擎回答的大规模测量，首次系统揭示了生成式搜索引擎答案列表的不稳定性——同一问题的答案有32%的概率给出不同的第一名，即便重复25次也只有60%的概率能确定接近的领先者，且引擎间差异大于引擎内部、开放式问题的不稳定性是确定性问题的两倍以上，因此对生成式引擎可见性的可靠监测远比现有推荐的方法更困难。 |
| [^23] | [What Quality Engineers Need to Know About Generative AI: Structured Data Extraction](https://arxiv.org/abs/2611.30778) | 该论文向质量工程师介绍了如何利用大语言模型和视觉-语言模型进行结构化数据提取，并提供了两个经人工验证、字段级准确率高达98.6%的免费Web应用（汽车召回通知信息提取与arXiv质量工程研究监测）。 |
| [^24] | [Trallie: from Free-Form Text to Structured Data with LLMs](https://arxiv.org/abs/2611.30773) | Trallie 是一个开源框架，利用大语言模型的零样本/少样本能力实现多种文档格式的信息抽取，并可自动从文档集合中归纳抽取模式，让非专家用户也能便捷地将非结构化文本转换为结构化数据。 |
| [^25] | [CocoonBench: Can Multilingual Search Agents Escape Language Information Cocoons?](https://arxiv.org/abs/2611.30727) | 提出CocoonBench基准，通过挖掘跨语言的以答案为中心的关系路径和必要线索集，评估多语言搜索智能体能否突破单一语言的信息茧房并利用多语言证据找到唯一答案。 |
| [^26] | [Hard to Translate, Harder to Detect: Human Analysis Remains Essential in AI Meme Translation](https://arxiv.org/abs/2611.30710) | 该论文构建了基于苏联喜剧迷因台词的文化负载文本翻译数据集与评估体系，证明人工验证的难度分析仍是提升AI迷因翻译质量的关键，因为即使是最强的模型也无法可靠地自行生成这类分析。 |
| [^27] | [Lowering the Bar for Everyone: Hidden Validation-Seeking Goals in Empathetic RL](https://arxiv.org/abs/2611.30702) | 在基于LLM模拟用户的共情强化学习中，寻求认同用户的占比每增加10个百分点，策略对有过错发帖人的认可率就上升3–4个百分点，表明策略会隐蔽地降低对所有人的认可门槛，而其辨别过错的能力并未下降。 |
| [^28] | [Synthetic Twins: Closing the Gap Between TTS Test Sets and Real Recordings in Turkish ASR Evaluation](https://arxiv.org/abs/2611.30694) | 本文提出“合成孪生”方法，通过逐级向合成语音添加说话人音高、说话人克隆、真实录音声道和语速等因素，发现说话人克隆是缩小合成测试集与真实录音在土耳其语ASR评估差距的关键，将Spearman相关性从.41提升至.55。 |
| [^29] | [Learn the Effects, Not the Examples: Implicit ICL via Attention Patching](https://arxiv.org/abs/2611.30690) | 提出Attention-Patching方法，通过学习逐层仿射变换来模拟示例对注意力输出的影响，无需在上下文中包含示例即可实现高效且性能优异的隐式上下文学习。 |
| [^30] | [Empirical Margin Sensitivity: An Interventional Neuron-Level Probe of Post-Trained Medical LLMs](https://arxiv.org/abs/2611.30675) | 该论文提出经验边际敏感性（EMS）这一神经元级干预探针，在MLP下投影列粒度上探测后训练医疗大模型的决策内部依赖，并在多个模型家族中发现了跨基准的高敏感“锚点”列。 |
| [^31] | [Uncovering the Tells: Efficient Black-Box Audits for LLM Data Contamination](https://arxiv.org/abs/2611.30639) | 该论文提出了一个基于生成的黑盒审计框架，利用训练数据成员信号高度稀疏并集中于少数高信息量 token 的特性，进行预算受限的针对性 token 采样，从而在不依赖灰盒访问的情况下高效准确地审计大语言模型的训练数据污染。 |
| [^32] | [From Rewarding What Models Say to How They Think: Attention Process Reward Model for LLM Reinforcement Learning](https://arxiv.org/abs/2611.30627) | 本文提出注意力过程奖励模型（APRM），通过基于隐藏状态和注意力图分配内部奖励，从“奖励模型说什么”转向“奖励模型如何思考”，解决了文本奖励混叠问题，并以仅约 4600 万参数的轻量模型捕捉高质量推理模式、指导强化学习优化并定期更新以应对分布偏移。 |
| [^33] | [Does the Mixer Matter ? Paired-Seed Ablations of Tiny Bilingual Language Models](https://arxiv.org/abs/2611.30623) | 对小型希腊语-英语双语模型的成对种子消融实验表明，在此微小规模下，存储容量（如稀疏专家数量）远比混合器（注意力机制）的设计选择更重要。 |
| [^34] | [Auxiliary Clues Aware's Geometry Problem Solving](https://arxiv.org/abs/2611.30617) | 提出AuxCA强化学习框架，通过辅助线索依赖度（ACD）量化辅助视觉线索的作用，使多模态大模型学会可迁移的辅助构造策略，在推理时无需显式作图即可提升几何问题求解能力。 |
| [^35] | [Quantizing Diffusion LLMs under Parallel Decoding: Commitment Risk and the Parallelism Tax](https://arxiv.org/abs/2611.30587) | 该论文首次在实际部署的缓存式并行解码器下研究扩散大语言模型的 W4A4 量化，揭示了量化误差主要经由“强制承诺”进入、量化还会带来额外的“并行性税”，并提出用承诺风险先验重加权逐层校准的 DECAL 方法来缓解这两个问题。 |
| [^36] | [Interrogation Protocols for AI Control](https://arxiv.org/abs/2611.30583) | 本文提出在AI控制中引入审讯协议，让受信任的监控器通过向不受信任的智能体提问来辨别其行为解释的真伪，在BashArena基准上将1%审计预算下的安全性从72.6%提升至79.5%，但大部分提升仅靠受信任模型陈述具体担忧即可获得。 |
| [^37] | [Seeing Is Locating: Spatial Awareness and Geographic Bias in Decision Language Models](https://arxiv.org/abs/2611.30564) | 决策语言模型需要视觉输入才能实现空间感知，且其预测存在显著的地理偏见。 |
| [^38] | [Body and Soul: Decoupling Robotic Brain and Policy with RoboHarness-Zero](https://arxiv.org/abs/2611.30550) | 提出首个零准备、可跨任务跨策略复用的机器人框架 RoboHarness-Zero，通过将冻结的 VLA 策略与 VLM“大脑”解耦——由 VLM 负责目标选择与任务进度跟踪——实现多技能组合，同时保留策略原有的独立能力。 |
| [^39] | [CyberDualEval: Measuring Dual-Use Cyber Risks in Frontier Language Models](https://arxiv.org/abs/2611.30533) | 提出了CyberDualEval基准，通过430个分三阶段递进（安全分析、概念验证生成、漏洞利用生成）的漏洞任务以及衡量残余风险的任务武器化分数，评估前沿语言模型能否在保留良性安全能力的同时拒绝实质性助推网络滥用。 |
| [^40] | [When Does Continual Pre-Training Help? An Empirical Study of Decoupled Knowledge and Generation Trajectories in Clinical LLMs](https://arxiv.org/abs/2611.30507) | 该研究通过同时跟踪多选题知识测试与开放式生成偏好排序两条轨迹，发现临床大语言模型持续预训练中生成风格对齐的收敛速度（约1.7B tokens）显著快于事实知识的获取（约3.5B tokens），且早期生成增益主要源于表面层面的领域风格对齐。 |
| [^41] | [Rethinking the Evaluation of Contrastive Decoding for Hallucination Mitigation in Multimodal Large Language Models](https://arxiv.org/abs/2611.30460) | 该论文提出了CONCOR受控评估基准，利用对比接地边际直接衡量对比解码方法在多模态大语言模型幻觉缓解中的真实效果，克服了现有基准无法隔离底层对比调整作用的局限。 |
| [^42] | [Disclosure Bias in AI Reviewers](https://arxiv.org/abs/2611.30410) | 研究发现AI审稿人存在披露偏差：当论文的AI使用声明表明AI完成了全部工作时，所有AI审稿人的评分平均都会降低，这可能会影响论文的录用结果。 |
| [^43] | [TAP: Policy Interfaces for Diffusion Language Models](https://arxiv.org/abs/2611.30406) | 提出TAP-DiLM，在掩码扩散语言模型中引入带类型的策略变量，系统比较了专用槽位头、共享词表槽位和CLS外部头等策略接口的优劣，并给出基于LM头尖锐度的实用接口选择诊断方法。 |
| [^44] | [LeanReviewBench: Semantic Alignment Review in Mathematical Lean Projects](https://arxiv.org/abs/2611.30361) | 提出了LeanReviewBench——一个包含来自18个真实Lean项目的197个标注输入的项目级基准，用于评估语言模型审查非形式化数学目标与形式化Lean命题之间语义对齐的能力。 |
| [^45] | [When Does Reasoning Help Entity Disambiguation? Small LLMs In-Domain and Under Shift](https://arxiv.org/abs/2611.30334) | 该论文在经典由粗到细实体消歧流程中，系统比较了小型大语言模型下逐点式、列表式与基于推理的候选选择方法在领域内和分布偏移场景下的表现，探究了小规模场景下推理何时值得其代价。 |
| [^46] | [Evaluating Persona Consistency in LLM User Simulators Without External Reference Annotations](https://arxiv.org/abs/2611.30280) | 该论文提出一种无需外部参考标注的自我锚定内部一致性诊断方法，通过让LLM用户模拟器“自我报告一次、行为两次”，在共享的BFI-2六维度空间中量化自我报告与行为表现之间的相关性，从而评估和比较不同的人格构建策略。 |
| [^47] | [DUET: Dual-MDL Unsupervised Entropy-Trie Segmentation for Low-Resource Languages](https://arxiv.org/abs/2611.30273) | 提出DUET算法，通过将分支熵与递归范式发现相结合生成候选切分，并采用双重MDL准则在奖励语素复用与防止过度切分之间取得平衡，实现了无需标注数据和外部资源的低资源语言无监督形态切分。 |
| [^48] | [Voyag: Joint Multi-Intent Detection and Execution-Relation Classification for In-Vehicle Voice Assistants](https://arxiv.org/abs/2611.30253) | Voyag 是一个单次前向的联合决策模型，在车载语音助手场景中同时预测十个车载意图槽位和每对共现意图之间的三分类执行关系（并行、顺序、互斥），并通过统一联合损失进行端到端优化。 |
| [^49] | [StutterStigma-TR: A Novel Turkish Dataset and Target-Aware BERT Framework for Stuttering-Related Stigma Detection](https://arxiv.org/abs/2611.30252) | 本文提出了首个土耳其语口吃污名化检测数据集StutterStigma-TR（含6,019条社交媒体帖子），并设计了基于BERTurk的目标感知框架StutterStigma-BERT，通过显式目标标记与多维度特征融合显著提升了口吃相关污名化言论的检测效果。 |
| [^50] | [Ontology-Compiled 2D Clinical State Fields with Bidirectional Diffusion Encoders for Zero-Training Cross-Ontology Medical Coding](https://arxiv.org/abs/2611.30218) | 该论文提出无词表框架OCSF，通过将临床文档编码为固定维度的二维状态矩阵，并利用离线编译的Tikhonov反卷积算子匹配任意目标本体，实现了无需重新训练、数秒内即可切换医学编码词表的零训练跨本体医学编码。 |
| [^51] | [What Makes Good Data for Training Coding Agents? A Systematic Study on SWE-Gym](https://arxiv.org/abs/2611.30207) | 本研究系统审计了SWE-Gym数据集奖励标签的正确性，发现模型从未解决的任务中74%存在拒绝正确修复的假阴性测试，修复或移除这些错误标注任务后，训练出的模型在SWE-bench Verified和SWE-bench Pro上分别提升2.4分和2.7分。 |
| [^52] | [PackLoop: Row-Packed Self-Speculative Decoding for Looped Language Models under Concurrency](https://arxiv.org/abs/2611.30205) | PackLoop 发现并发负载下自推测解码失效的原因是推测工作的布局而非推测本身，并提出只在必然要读取的 KV 缓存行内打包推测位置，使循环语言模型在高并发下仍保持加速收益。 |
| [^53] | [VADE: A Verifiable Agent Data Engine from Professional Sources](https://arxiv.org/abs/2611.30202) | VADE是一个数据引擎，它利用专业分析“求解难、验证易”的非对称性，从法律、金融等专业实践的公开来源中自动构建可验证的分析任务，从而解决这些领域高质量智能体训练数据难以规模化的问题。 |
| [^54] | [Laya-IsoToken: Day-One Clinical Codebook Switching via Closed-Form Operator Compilation](https://arxiv.org/abs/2611.30196) | 该论文提出Laya-IsoToken方法，通过将临床证据读取器与冻结的多语言概念空间解耦，并利用闭式矩阵编译对齐目标编码描述，实现了在无需任何目标编码本标注病历的情况下首日完成临床编码本切换。 |
| [^55] | [Same Executions, Different Gaps: Judgment Rules in Web-Agent Evaluation](https://arxiv.org/abs/2611.30190) | 该论文揭示了Web智能体评估中的判定规则问题：评判标准与证据要求的选择可以在不改变排名的情况下大幅改变所报告的性能差距，且提高双方分数估计的准确性反而可能降低差异估计的准确性。 |
| [^56] | [PCC: Pivotal Channel Compensation for Layer Pruning of Large Language Models](https://arxiv.org/abs/2611.30183) | 提出关键通道补偿（PCC）方法，通过在剪枝接口处仅对少量关键通道施加可学习的线性变换，以低成本显著恢复大语言模型层剪枝后的性能。 |
| [^57] | [Retrieval-Augmented Generation in the Agentic Era: Tradition, Present, and Prospect](https://arxiv.org/abs/2611.30181) | 该论文从智能体时代视角出发，沿着统一的生命周期轴（离线索引、检索前、检索、检索后与编排）系统综述了RAG的最新进展，并首次将索引与编排作为一等组件、将智能体工作流作为独立范式加以梳理，同时讨论了评估、应用与未来开放问题。 |
| [^58] | [FiRE: Fine-grained Ranking Evaluation for Machine Translation](https://arxiv.org/abs/2611.30168) | 该论文提出FiRE方法，利用现成大语言模型在忠实度、流畅性和风格一致性三个维度上进行基于标准的成对比较，实现机器翻译的可解释细粒度无参考排序评估，并构建了首个面向该任务的人工标注基准。 |
| [^59] | [OpenViVE: New Benchmark and Dual-Stream Co-Attention Framework for Vietnamese Visual Entailment](https://arxiv.org/abs/2611.30161) | 该论文提出了首个大规模越南语母语视觉蕴含基准OpenViVE（含62,268个图像-假设对），通过伪影感知的LLM-人类混合流水线有效减少数据伪影，并进一步提出基于双向协同注意力的双流PhoBERT-ConvNeXt框架，取得了84.84%的准确率。 |
| [^60] | [Reading the Tea Leaves, Again: Can Large Language Models Predict Language Model Training Runs?](https://arxiv.org/abs/2611.30141) | 该论文提出了Tea Leaves基准，通过让大语言模型根据训练运行前20%的配置、描述、代码和曲线来预测其余结果，在682个真实训练运行上评估了11个LLM的预测能力，发现最强LLM在平滑损失曲线上可比肩最佳曲线拟合器，而在有噪声的强化学习奖励曲线上则显著领先。 |
| [^61] | [When Does Optical Context Compression Pay Off? A Decode-Aware Cost Analysis of Reasoning VLMs](https://arxiv.org/abs/2611.30130) | 该研究揭示推理型视觉语言模型将长文本渲染为图像虽节省输入token，但会使解码长度增长1.5-3.5倍并可能反而推高总成本，并提出判断光学压缩是否划算的汇率公式 (1-1/c)/(r-1)，即该比值需超过工作负载的输出/输入token价格加权比。 |
| [^62] | [Parameter-Efficient Vocabulary Expansion via Cross-Lingual Embedding Alignment](https://arxiv.org/abs/2611.30123) | 提出参数高效的词表扩展方法MATE，通过利用主语言与扩展语言词汇间的语义等价性进行跨语言嵌入对齐，使每个词元的额外参数减少8倍，同时提升多个基准测试的性能。 |
| [^63] | [Character Information in Language Models: A Prompt-Independent Word Store and a Late, Question-Driven Signal That Grows With Scale](https://arxiv.org/abs/2611.30094) | 本研究通过词位置基线方法将语言模型中词的字符信息分解为两部分——约70%的字符信息（如首字母、尾字母、最频繁字母）已独立于提示预先存储在词元嵌入中，其余部分则是在答案位置由问题驱动、随模型规模增长且局限于后期层的信号。 |
| [^64] | [ESGPT: A Data-Centric ESG Language Model with Unified Instruction and Benchmark Resources](https://arxiv.org/abs/2611.30052) | 该研究构建了包含3万条实例的ESG指令数据集ESG-Instruct和覆盖六项代表性任务的基准ESGU-Bench，并据此训练出80亿参数的ESG专用大语言模型ESGPT，其以较小的规模在多项ESG任务上超越了表现参差不齐的通用大语言模型。 |
| [^65] | [STUN-Bench: Truly Unseen Tools for Evaluating Shared Tool Experience](https://arxiv.org/abs/2611.30034) | 该论文提出STUN-Bench基准，构建了经验池中完全不包含172个保留工具任何信息的数据分割，并揭示了共享经验会挤占工具描述排名、导致真正未见的工具从候选列表中消失的“拥挤”问题。 |
| [^66] | [SHAKTI: Evidence-Grounded Conversational AI for Nutrition and Health](https://arxiv.org/abs/2611.29988) | 该论文提出了SHAKTI，一个仅基于检索到的学术文献和精确食物成分数据回答营养健康问题的对话式AI助手，通过来源链接核查和引文数字在线验证等机制确保回答有据可依，有效解决了大语言模型产生无依据论断的问题。 |
| [^67] | [How Relations Impose Embedding-Space Structure](https://arxiv.org/abs/2611.29986) | 本文提出一种无需学习参数的简单方法，通过比较不同领域间的内部结构排列，发现语言模型的表示被一致地排列以编码复杂的关系相似性，表明表示几何结构远比以往认为的更加丰富。 |
| [^68] | [Language models seek help despite guaranteed score losses in question answering](https://arxiv.org/abs/2611.29981) | 该研究系统量化了语言模型在成本-收益权衡下的求助决策，发现即使求助必然导致分数损失，各模型仍会咨询专家，且微调可将决策遗憾减少59%至87%，外部路由则能进一步提升答题得分。 |
| [^69] | [When 1+1 $\neq$ 2: Benchmarking Civil-Time Reasoning in Large Language Models](https://arxiv.org/abs/2611.29972) | 该论文提出CivilTime-Bench基准，系统评估大语言模型在夏令时转换等非线性民用时间（日历）推理上的能力，发现即使借助契约感知提示、少样本学习或微调，模型在真实场景中的准确率与泛化能力仍存在显著局限。 |
| [^70] | [Tracing Attacker Diversity Reduction in Adaptive LLM Red Teaming](https://arxiv.org/abs/2611.29960) | 本研究系统定位了大语言模型自适应红队测试中攻击多样性缩减的发生环节与机制，发现对“赢家”攻击的奖励强化是收窄主因，冻结防御者会加速这一过程，而防御者更新只能延缓但无法阻止多样性衰减。 |
| [^71] | [Yggdrasil: Tree-Routed Recurrent Experts for State Tracking and Fast CPU Inference](https://arxiv.org/abs/2611.29954) | Yggdrasil通过树路由的循环专家库将模型容量与每步计算解耦，既能跟踪离散状态，又能在CPU上实现快速推理。 |
| [^72] | [The Answer Is Not the State: Trajectory-Level Corrigibility in Diffusion Language Models](https://arxiv.org/abs/2611.29937) | 扩散语言模型中仅重新打开最终答案不足以纠正决策——旧的推理过程会重建原有结论，必须同时重新打开推理过程和答案，决策才能跟随新证据。 |
| [^73] | [SAEPO: Sparse-Autoencoder-Guided Credit Assignment for Policy Optimization](https://arxiv.org/abs/2611.29936) | SAEPO利用预训练稀疏自编码器构建词元级语义表示，将结果监督下策略优化的信用分配从序列级细化到词元级，从而更高效地提升大语言模型的推理训练效果。 |
| [^74] | [Training-Free Interventions for Hallucination in Large Language Models](https://arxiv.org/abs/2611.29909) | 本文系统综述了116篇关于大语言模型推理阶段免训练幻觉干预的研究，按证据来源、模型访问需求、准备成本和适用任务对各方法族进行分类比较，并强调忠实性、事实正确性、信息量与答案覆盖度需分别评测。 |
| [^75] | [Escriba-OCR: A 1B Vision-Language Model for Error-Preserving Transcription of Handwritten Portuguese Essays](https://arxiv.org/abs/2611.29874) | 提出了Escriba-OCR——一个10亿参数的视觉语言模型，通过在647页手写作文上微调，实现了葡萄牙语手写作文的错误保留转录，能够精确保留拼写错误、编辑标记和行结构，克服了现有开源模型倾向“自动纠错”的缺陷。 |
| [^76] | [GlobalHRBench: Does Changing the Question Language Change the Legal Answer?](https://arxiv.org/abs/2611.29852) | GlobalHRBench 是一个在八个司法管辖区和八种语言下保持事实与适用法律一致、包含 240 道劳动法问题的评测基准，评估表明大模型的法律答题准确率会随提问语言而变化（最佳模型严格通过率 76.2%，跨语言相对下降 5.8%）。 |
| [^77] | [The Temporal Residual: Fast and Effective Retrofitting of Latent Feedback into Pretrained Language Models](https://arxiv.org/abs/2611.29843) | 提出“时间残差”结构——一条仅需2,752个参数的第二残差流，可在预训练后直接加装到语言模型中，实现跨token的隐藏状态潜在反馈传递，显著提升数学推理性能并产生更简短的解答。 |
| [^78] | [ComplexIF: A Challenging Multi-Turn, Multi-Constraint Instruction-Following Benchmark](https://arxiv.org/abs/2611.29775) | ComplexIF是一个自动化生成的多轮、多约束指令遵循基准，通过程序验证的格式约束和LLM裁判评估的语义指令两个互补子集，在贴近真实使用场景的前提下为大模型提供更具挑战性的测试。 |
| [^79] | [S\'anchez, Feij\'oo, or Abascal? Speaker Attribution and Ideological Congruence in LLM Personas Reading Posts on X of the Leaders of the Three Main Parties of the 29N General Election in Spain](https://arxiv.org/abs/2611.29765) | 本文首次通过大规模发言人归因实验验证了LLM角色画像会对发言者身份（而不仅是文本内容本身）作出反应，发现点名西班牙三大政党领导人会系统性降低对相同文本的支持度，且阿巴斯卡尔带来的负面效应约为桑切斯和费霍的两倍。 |
| [^80] | [Refine, Don't Reinvent: RASA-Bench for Source-Grounded Hindi Prose Refinement](https://arxiv.org/abs/2611.29750) | RASA-Bench 提出了一个针对印地语散文润色任务的新基准，通过忠实性门控加 Bradley-Terry 编辑偏好排名的四层评估协议，衡量模型在不偏离源文本前提下改进单薄 AI 草稿的能力，发现忠实性测试在短篇小说与有声小说等不同文体上的淘汰率差异显著。 |
| [^81] | [ReqSeek: Transformer-Based Automatic Requirements Identification for Building Large Requirements Datasets](https://arxiv.org/abs/2611.29722) | 本文提出了基于Transformer、领域无关的自动需求识别模型ReqSeek，并构建了包含23个异构SRS的ARID数据集，以RoBERTa为基础模型达到95%的精确率与召回率，性能可与人类媲美且耗时更少。 |
| [^82] | [Loop-Count Schedules in Looped Transformers: An Empirical Study](https://arxiv.org/abs/2611.29704) | 循环Transformer中循环次数采样范围的下界决定了重复计算的性质：若范围包含R=1，门控坍缩至0.1-0.5、后续循环仅作小修正；若从R=2开始，门控保持在0.6-0.9、每次循环均保持完整层的强度。 |
| [^83] | [Where Do Fact-Checking Errors Arise? Separating Retrieval, Search Control and Judgement in Evidence-Grounded Verification](https://arxiv.org/abs/2611.29697) | 该论文提出一套分离检索、搜索控制、判断与裁决验证的诊断协议，发现事实核查失败的主要根源不在证据检索（单次混合检索即可覆盖绝大多数声明的完整证据），而在于证据的后续使用环节——判断器弃权与验证规则推翻了本可正确作出的裁决。 |
| [^84] | [Grammar Sparsity Sets How Transformers Learn Random Languages](https://arxiv.org/abs/2611.29692) | 语法稀疏性同时决定了Transformer学习随机语言的顺序和速度——稠密语法分层学习且经历随词表大小变宽的平台期（$t\propto V$ 到 $t\propto V^3$），而稀疏语法则一次性快速习得（$t\propto\sqrt V$）。 |
| [^85] | [Do More Roles or Larger Models Improve LLM Fact-Checking? A Controlled Study on Fixed Evidence](https://arxiv.org/abs/2611.29687) | 这项对照研究发现，在固定证据下的事实核查中，更大的模型并不能带来更高准确率反而更常拒绝判断，而多角色“法庭”机制的表现与简单投票相当，说明计算资源的分配方式比模型规模更关键。 |
| [^86] | [MERIT: Experience-Guided Memory for Long-Horizon Persona-Specific Persuasive Dialogue](https://arxiv.org/abs/2611.29679) | 该论文提出MERIT系统，通过结合图结构的说服专用记忆与每个人设的经验推理库，仅凭提示组合即可实现无需参数更新的跨会话自适应说服对话，并推出了PERSIST人设保险交互数据集。 |
| [^87] | [Position-Skipped Distillation of Sparse-Attention Indexers: Train Short, Serve Long](https://arxiv.org/abs/2611.29664) | 该论文提出跳位蒸馏方法，通过让索引器与冻结教师在位置 ID 含随机间隔的短序列上训练，发现索引器长上下文泛化失败主要源于 RoPE 位置编码而非内容匹配退化，从而实现“短训练、长服务”，以约 6 倍更少的训练成本即可匹配长序列蒸馏的索引器效果。 |
| [^88] | [How Reliable is AI-Generated Audio Description? A Multilingual Expert Evaluation](https://arxiv.org/abs/2611.29658) | 该研究通过对意大利语和德语电影进行多语言专家评估，发现尽管GPT-4o优于基于Qwen的系统，AI生成的音频描述在错误率和质量评分上仍显著低于人类专业描述员的水平。 |
| [^89] | [Which Loopholes Get Found? Predicting Reward Hacking of LLM Judges Before RL Training](https://arxiv.org/abs/2611.29651) | 该论文提出“暴露度”指标（初始策略产生缺陷模式的倾向性 × 裁判假阳性率），能够在强化学习训练开始前预测 GRPO 训练中 LLM 裁判会被策略最先利用哪个漏洞。 |
| [^90] | [Reducing Authorship Attribution Risk in Student Texts Through Style-Conditioned Rewriting](https://arxiv.org/abs/2611.29642) | 该论文提出一种风格感知的改写方法，通过将内容、结构与风格解耦并结合聚类受限检索的差分隐私上下文学习（DP-ICL），在保留文本分析价值和下游任务性能的同时，有效降低学生文本的作者归属风险。 |
| [^91] | [Diversity in Unity, Theory in Practice: Hierarchical Multitask Benchmarks for Chinese Minority Languages](https://arxiv.org/abs/2611.29635) | 该论文提出CMiLBench——首个面向藏语、蒙古语和维吾尔语三种中国少数民族语言的分层多任务评估基准，涵盖5个难度级别17个任务共24,663个实例，通过评估14个大语言模型揭示了其在少数民族语言上理论能力与实际表现之间的差距。 |
| [^92] | [Explainable Emotion and Mental Health Reasoning in Large Language Models Using Theories of Emotion](https://arxiv.org/abs/2611.29628) | 该论文提出了一种基于沙赫特-辛格情绪理论和认知评估理论的心理学思维链提示策略PsyCoT，使大语言模型在情绪分析和心理健康任务中实现可解释的推理。 |
| [^93] | [KernelShift: Benchmarking Kernel Optimization Across Evolving Hardware](https://arxiv.org/abs/2611.29590) | 该论文提出了KernelShift基准测试流水线，通过从ncnn、llama.cpp等真实活跃仓库中构建99个内核优化任务，并让智能体在目标硬件反馈下优化代码，以评估编码智能体在不断演进的CPU架构扩展下编写高性能内核的能力。 |
| [^94] | [Single-Coordinate Activation Steering for Controllable Social Stance in Language Models](https://arxiv.org/abs/2611.29578) | 该论文发现仅需提高语言模型中单个神经元（Qwen3.5 4B 第9层第7515号神经元）的激活，无需任何训练即可控制模型的社交立场，使其回复在疏远与热情之间切换，并通过模型评分与盲测人类对比验证了该效果显著优于随机神经元干预。 |
| [^95] | [LM2 Compiling a Language Model into Another Model Weights](https://arxiv.org/abs/2611.29577) | LM2通过在训练时约束生成器的算术运算，将训练好的故事生成模型直接编译进更大对话模型的新增权重层中，使宿主模型每生成一个字符仅需一个token（对比QwenPU方案需约25,000个），代价是文本预测精度略有下降。 |
| [^96] | [CompareDR: Benchmarking Deep Research Reports from Individual Quality to Comparative Merit](https://arxiv.org/abs/2611.29549) | 提出了CompareDR基准，通过结合任务特定的逐点质量审查、基于准则的两两比较和选择性证据验证，实现对深度研究报告从个体质量到相对优势的全面评测。 |
| [^97] | [Estimating Silenced Tokens in Quantized LLMs](https://arxiv.org/abs/2611.29536) | 本文提出一种简单可扩展的认证式估计方法，能够精确量化低比特量化下大语言模型输出头中被“沉默”（贪心解码永远无法生成）的词元数量，发现量化确实会导致词元被沉默。 |
| [^98] | [Adaptation or Poisoning? Security Risks of Conversational Test-Time Learning in Large Language Models](https://arxiv.org/abs/2611.29532) | 该论文揭示了大语言模型测试时学习的安全风险，提出了P-TAP多轮攻击方法，攻击者仅通过普通对话中逐步升级的查询即可毒化会话适配器，最终绕过安全对齐实现越狱。 |
| [^99] | [Recursive Self-Editing: LLMs Program Their Own Weight Edits for Self-Improvement](https://arxiv.org/abs/2611.29513) | 提出 SELFWRITE 框架，让大语言模型通过编写程序、利用白盒基元（而非梯度下降）直接构造并合并自身权重的低秩编辑，从而实现递归式的自我改进。 |
| [^100] | [Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification](https://arxiv.org/abs/2611.29491) | 该论文发布了覆盖17种语言（含7种东南亚语言）和9类PII标签的大型多语言临床去身份数据集及完整透明的生成流程，基于此训练的紧凑3.5亿参数模型在全部15个公开基准上以平均精确匹配F1 0.827大幅超越最强现有系统（0.672）。 |
| [^101] | [GraphSeer: Multimodal Knowledge Graph Relational Image Retrieval with MLLMs](https://arxiv.org/abs/2611.29470) | GraphSeer 提出了知识图谱关系图像检索（KRIR）这一新任务——给定照片和自然语言关系检索相关的多模态知识图谱实体，并在单个多模态大语言模型中统一完成“识别照片对应实体”与“推理其关联实体”这两个步骤。 |
| [^102] | [Beyond Accuracy: A Diagnostic Benchmark for Hypothesis-Driven Experiment Planning in LLM Agents](https://arxiv.org/abs/2611.29467) | 该论文提出了一个超越最终准确率的诊断性基准与评估框架，通过评估智能体的探索行为模式并将其推理过程分解为信念更新、假设区分、判别性实验设计和干预下预测推理四个可解释步骤，来深入诊断LLM智能体在假设驱动实验规划中的能力。 |
| [^103] | [MAS-BIT: A Multi-Agent System for End-to-End BraIn-to-Text Decoding](https://arxiv.org/abs/2611.29457) | 该论文提出了首个面向端到端脑信号到文本解码的多智能体框架MAS-BIT，通过神经智能体、语音智能体与仲裁智能体之间基于词嵌入的潜空间通信协同工作，直接从皮层内神经记录生成文本转录。 |
| [^104] | [PAWP-TTS: Phoneme-Aware Weight Programming for Compact Bilingual English-Nepali Speech Synthesis](https://arxiv.org/abs/2611.29455) | PAWP-TTS通过将确定性发音知识交由基于规则和词典的音素感知双语前端处理，使网络仅需建模声学，从而以仅5.98 MB、149万参数的紧凑非自回归模型实现了高质量的英语-尼泊尔语双语及混合编码语音合成。 |
| [^105] | [Detectable Epistemic Attack Layers: A Coverage Theorem and Standardized Harness for Conversation-History Integrity](https://arxiv.org/abs/2611.29449) | 该论文按攻击层形式化了会话历史污染并证明覆盖定理——每层验证恰好只覆盖自身层、指令覆盖的检测必然带来误报、复合攻击向量可绕过逐片段验证——并通过覆盖八个机制家族、3,249个会话和三个模型家族的标准化测试框架予以量化验证。 |
| [^106] | [VistaTree: Question Answering over Long Multimodal Documents with Image-Derived Summary Trees](https://arxiv.org/abs/2611.29419) | VistaTree通过直接从页面图像构建层次化摘要树并进行层级导航，实现了对超出上下文窗口的长篇多模态文档的高效问答。 |
| [^107] | [HGDNA: Overlapping k-mer Tokenization for Efficient Autoregressive Genomic Language Models](https://arxiv.org/abs/2611.29400) | 提出HGDNA，一种采用重叠k-mer分词与混合线性注意力机制的因果基因组语言模型，仅用30M参数和19B预训练token即在序列分类和嵌入评估任务上取得有竞争力的性能。 |
| [^108] | [Can Speech LLMs Recognize When Audio Evidence Is Insufficient? A Diagnostic Study and a Pre-Generation Probing Mitigation](https://arxiv.org/abs/2611.29393) | 该研究通过对比合成语音与真人朗读录音的诊断实验，揭示了语音大语言模型在音频证据不足以回答问题时会大量幻觉作答、且合成语音评估会严重误导失败归因，并提出一种生成前探测方法来缓解该问题。 |
| [^109] | [SAE-Audit: Clinical-Trial Prediction via Inference-Time Optimization of Black-Box LLMs Audited by Sparse Autoencoder Evidence](https://arxiv.org/abs/2611.29368) | 提出SAE-Audit方法，将开源模型上稀疏自编码器检测到的临床概念证据用于推理时优化黑盒大语言模型，从而提升临床试验结局预测的准确性和可解释性。 |
| [^110] | [Automatic Construction of Persian E-Commerce Dialogue Datasets for Conversational Product Search](https://arxiv.org/abs/2611.29356) | 该论文提出一个基于正则化分区平衡（RPB）分数与LLM的自动化框架，从真实电商目录中自动构建并发布了覆盖7个类别、12,926个产品的大规模波斯语对话式产品搜索数据集。 |
| [^111] | [Endpoint-Conditioned Process Value: What an Endpoint-Blind Process Score Adds Once Terminal Status and Cost Are Known](https://arxiv.org/abs/2611.29354) | 本研究通过1,255条轨迹的实验首次量化了过程评分在已知最终状态和成本后的增量价值，发现其增益高度依赖评判模型规模——只有32B级别的评判模型才能带来显著的AUC提升，且大部分增益可由八个可记录的行为计数器替代。 |
| [^112] | [EMIP: Embodied Multi-Party Interaction Platform](https://arxiv.org/abs/2611.29333) | 该论文提出了一个基于Minecraft构建的具身多方交互平台EMIP，通过客户端-服务器架构支持多个人类与多个AI智能体在同一虚拟环境中进行全双工语音交流和同步物理交互，为研究人类与具身AI智能体之间的协作提供了实验平台。 |
| [^113] | [Counting Symbols on Technical Drawings: A Benchmark and Verified Rewards for MLLMs](https://arxiv.org/abs/2611.29309) | 该论文提出了涵盖建筑平面图、电气原理图、机械图纸和分子结构共1,574个计数问题的CountDrawBench基准，并设计了结合计数奖励、实例匹配奖励与基于缩放奖励的Code-Verified RL强化学习框架，通过可执行验证反馈显著提升多模态大语言模型在技术图纸上的符号计数能力。 |
| [^114] | [From Vanishing Gradients to Recursive Self-Improvement: Evaluating the Most Influential Papers in Large Language Model Evolution (1997-2026)](https://arxiv.org/abs/2611.29303) | 本综述按时间线评估了1997-2026年间22篇最具影响力的大语言模型论文，梳理了从RNN、LSTM、Word2Vec到Transformer的关键技术演进脉络。 |
| [^115] | [Beyond Omission: Measuring Critical-Information Fulfillment of Large Language Models in Bengali and English](https://arxiv.org/abs/2611.29299) | 该研究发现，大型语言模型在孟加拉语和英语上的关键信息遗漏率几乎相同，但孟加拉语的关键信息完成度和完整任务成功率显著更低，表明多语言可信度评估不应只关注信息遗漏，还应衡量信息的完成质量。 |
| [^116] | [Which Failure Does an Intervention Repair? When Agent Interventions Help, and Where Self-Evolving Skills Stall](https://arxiv.org/abs/2611.29290) | 该论文通过覆盖九大基准、四个模型与近90万条轨迹的匹配因子实验，系统揭示了智能体干预（技能、框架、推理预算）之间存在交互而非简单叠加效应，并证明应根据系统遗留的失败类型（知识缺失对应更强模型、执行错误对应更丰富框架）来选择修复手段。 |
| [^117] | [LLM-Based Assistant for Dog Care Guidance](https://arxiv.org/abs/2611.29286) | 研究通过对比实验发现，基于RAG检索增强的领域接地方法在提升大语言模型生成狗狗护理建议的准确性方面显著优于LoRA参数高效微调。 |
| [^118] | [Dual-Side Uncertainty Estimation: Calibrating Data and Model Uncertainty into a Shared Probability Space](https://arxiv.org/abs/2611.29284) | 提出DUE双侧不确定性估计方法，通过Dirichlet扰动生成词元不确定性序列并结合序列感知的Normal-Inverse-Gamma估计器，联合量化输入侧偶然不确定性与句子级认知不确定性，并将两者校准融合到统一的错误概率空间中。 |
| [^119] | [On the Interplay between Backtracking and Latent Confidence Directions in Reasoning Models](https://arxiv.org/abs/2611.29265) | 该论文通过提取模型内部潜在置信度的线性方向，实现了对推理模型在答案提案处是回溯还是接受答案的预测，揭示了内部不确定性表征与冗长回溯式思维链之间的联系。 |
| [^120] | [NoeVera: Can Knowledge Guide Experiments and Experiments Refine Knowledge?](https://arxiv.org/abs/2611.29255) | NoeVera是一个智能体式“第二大脑”，它将论文、代码、基准与实验结果整合为保留来源、持久化的三层知识库，通过决策感知检索、角色化智能体推理与模拟器调用，构建“知识指导实验、实验反哺知识”的闭环，以优化LLM服务系统的科研决策。 |
| [^121] | [Seeing Is Not Acting: WARD Diagnoses the Evidence-to-State-to-Action Gap in Longitudinal Clinical Agents](https://arxiv.org/abs/2611.29243) | WARD 是一个纵向临床智能体评测基准，通过 4,560 个临床任务和纵向、交互两条赛道，系统诊断了模型在“证据识别→患者状态判断→及时安全行动”链条中的能力差距。 |
| [^122] | [Edit-Induced Dependency Mismatch: Propagation Omissions, Recovery, and Evaluation Blind Spots in Coding Agents](https://arxiv.org/abs/2611.29242) | 该论文揭示了编码智能体在迭代改代码时会依赖上下文中残留的过时依赖判断，导致缺陷传播并逃过基准测试评估，并通过盲测人工审计确认了此类“编辑引发的依赖失配”的真实存在、运行中的恢复现象以及现有评估体系的盲区。 |
| [^123] | [GameAnything: From Any Inspirations to Playable Game Worlds](https://arxiv.org/abs/2611.29238) | 该论文提出了开源智能体平台GameAnything及其配套基准GameAnything-Bench，可将多模态灵感素材通过迭代开发转化为现有游戏的可玩扩展，并同时评估功能正确性与实际游戏体验质量。 |
| [^124] | [IteraSong: LLM-Guided Iterative Symbolic Music Creation in a Web DAW](https://arxiv.org/abs/2611.29225) | 本文提出IteraSong，一个网页版DAW系统，通过MCP工具将自然语言指令映射为分层编辑操作，支持用户在大语言模型辅助下对持久化多轨项目进行迭代式符号音乐创作与实时精修。 |
| [^125] | [Direct Feedback Policy Optimization for AI Coding Agents](https://arxiv.org/abs/2611.29193) | 提出单轨迹强化学习算法DFPO，无需依赖稀疏终端奖励和重复采样即可直接训练AI编程代理，在同等条件下任务解决率超过GSPO。 |
| [^126] | [ContextFact: Remembering Facts in Context for Long-Term Conversational Agents](https://arxiv.org/abs/2611.29155) | 提出ContextFact方法，通过在记忆构建阶段保留事实的局部上下文并分离组织跨会话证据，改进长期对话智能体的事实记忆与证据检索效果。 |
| [^127] | [Medical Specialty Prediction from Bangla Patient Narratives: A Comparative Study of ML, DL, Transformers, and LLMs](https://arxiv.org/abs/2611.29151) | 该研究针对孟加拉语非正式患者叙述构建了医学专科预测任务，并系统比较了机器学习、深度学习、Transformer和大语言模型四类方法的表现，其中微调的XLM-RoBERTa以86.26%的准确率取得最佳效果。 |
| [^128] | [VeriLogic: Reinforcement Learning over Verifiable Logical Reasoning Traces](https://arxiv.org/abs/2611.29147) | VeriLogic 提出了一种将逻辑推理过程分解为显式可验证的中间判断、证明和最终决策的强化学习方法，通过“扎根—优化—精炼”流水线实现过程级监督，解决了传统结果信号忽视推理轨迹内部正确性异质性的问题。 |
| [^129] | [Group-Balanced Supervised Fine-Tuning for Knowledge-Intensive Domains](https://arxiv.org/abs/2611.29127) | 该论文提出组平衡监督微调（GB-SFT），通过在序列内不同语义角色的 token 之间均衡分配监督损失预算，在法律、医学、代码和数学等领域显著提升大模型新知识注入效果，同时保留引用召回能力。 |
| [^130] | [When Can a Rewiring Control Test LLM Weight Readers? C2 and an Eligibility Protocol](https://arxiv.org/abs/2611.29120) | 论文提出C2反事实方法（置换网络最终线性读出层的列）及三条件资格协议，用于区分大语言模型是真正在“计算”权重还是仅识别权重空间“指纹”，但测试发现没有任何自然设置符合该协议资格。 |
| [^131] | [Decoding Transformer Dynamics via Koopman Subspaces](https://arxiv.org/abs/2611.29113) | 该论文将Transformer隐藏状态的逐层演化建模为动力系统，利用Koopman算子理论提取主导模态子空间作为主题“谱指纹”，证明模型深度映射近似线性、指纹可判别不同学科主题且在多个模型间可复现。 |
| [^132] | [Beyond Free-Form Generation: Measuring Interface Composition in Competitive NLP Systems](https://arxiv.org/abs/2611.29103) | 该研究通过对Kaggle和共享任务中数百个竞赛NLP系统的标注分析，首次量化了混合型系统（自由生成与受限接口结合）的占比差异，并识别出多种精细的接口配置模式。 |
| [^133] | [Operational Runbooks as Living Documents: A Longitudinal Study of Document Evolution under Assisted Post-Incident Revision](https://arxiv.org/abs/2611.29097) | 该论文首次对生产环境中运维手册作为“活文档”的长期演进进行了纵向实证研究，并提出利用轻量级的语言学与结构信号来量化与监控文档随时间的变化。 |
| [^134] | [RES: Accelerating Diffusion Language Models with Rolling Entropy Sampling](https://arxiv.org/abs/2611.29066) | 提出滚动熵采样器（RES），通过在达到最小长度后提交连续的已接受前缀、同时保留前方位置的修正空间，兼顾了准确率与计算效率，从而加速扩散语言模型。 |
| [^135] | [HyperHarness: Learning to Generate Personalized Agent Harnesses from User Histories](https://arxiv.org/abs/2611.29062) | HyperHarness通过学习一个共享的“历史到框架”生成器，直接从用户历史合成可执行的个性化检索与答案引导代码，兼顾了共享工作流的部署效率与逐用户定制的行为适配能力。 |
| [^136] | [Grounding Collaborative Agents in Group Cognition](https://arxiv.org/abs/2611.29036) | 该论文提出了“群体简报”这一新任务来重建团队中与任务相关的群体认知（包括成员信念、群体动态和信息缺口），并构建了包含1,351个请求与9,975个参考点的四领域基准及两阶段评估流程，同时推出了GroupBrief方法。 |
| [^137] | [LITMUS - Literature Informed Testbed for Mapping Usecases and Synthesizing Risk-Cards](https://arxiv.org/abs/2611.29034) | 本文提出LITMUS，一个结构化流水线，能够根据给定用例自动生成定制化数据集与风险卡，并利用同行评审文献提供可追溯证据，将每个风险与具体用例紧密关联，弥补了现有方法忽视用例场景的不足。 |
| [^138] | [Population-Based Post-Training for Large Language Models](https://arxiv.org/abs/2611.29014) | 提出基于种群的后训练方法PBPT，通过并行训练多个模型并利用种群机制自动调整超参数，在推理和工具使用基准上以相同计算量超越了独立RLVR训练的效果。 |
| [^139] | [Mind the Trap As LLMs Guess the Data Model: Grain-Aware Guards Against Fan-out in Text-to-SQL and Its Benchmarks](https://arxiv.org/abs/2611.29009) | 该论文揭示了LLM驱动的Text-to-SQL系统因一对多连接导致的扇出聚合错误这一经典陷阱，发现广泛使用的BIRD基准中有4.2%的标准答案被此类错误扭曲，并发布了修正后的BIRD-FanOut数据集和GrainTraps合成数据集以帮助防范这类错误。 |
| [^140] | [NoHairCache: Calibrate-Once KV Cache Eviction with One Power-Law Exponent](https://arxiv.org/abs/2611.29006) | 该论文证明KV缓存驱逐中的衰减指数α是模型内在属性、只需在单一语料上校准一次即可跨领域迁移，并将H2O与StreamingLLM两大主流方法统一为同一幂律评分族的两个端点。 |
| [^141] | [DoC: Learning to Defer or Commit for Timely, Accurate, and Stable Conversational Diagnosis Prediction](https://arxiv.org/abs/2611.29003) | 该论文提出两阶段框架DoC，通过迭代提交边界估计与延迟感知的强化学习，使模型学会在适当时机延迟或提交诊断预测，无需提交时间标注即可实现更及时、准确且稳定的对话式诊断预测。 |
| [^142] | [Elicitation format changes zero-shot classification accuracy in a medical vision-language model: a two-center evaluation on colorectal histopathology](https://arxiv.org/abs/2611.28997) | 仅改变对医学视觉-语言模型的提问格式就能大幅改变其零样本分类准确率——在结直肠组织病理学九分类任务上，要求输出全部九类概率且禁止推理使准确率下降约14个百分点，而在相同类别基数与token预算下允许模型进行推理并给出选定答案可恢复近19个百分点的准确率。 |
| [^143] | [EmoRouter: Adaptive Modality Cascading for Multimodal Emotion Recognition](https://arxiv.org/abs/2611.28960) | 提出了一种无需训练、与模型无关的EmoRouter框架，利用基于共形预测的不确定性估计，自适应地为每条话语选择模态级联的退出层级，在提升多模态情感识别准确率的同时降低计算开销。 |
| [^144] | [Don't Sleep, Keep Watch: Proactive Perception for Computer-Use Agents](https://arxiv.org/abs/2611.28947) | 提出NightWatch，一个即插即用框架，通过耦合轻量级System-1模型NightWatch-0.8B对界面状态进行流式监控，为计算机使用代理提供持续的实时感知，从而弥合因VLM推理延迟导致的感知与行动之间的时间鸿沟。 |
| [^145] | [When Rubrics Do Not Decide: Silent Value Completion in LLM Evaluation](https://arxiv.org/abs/2611.28940) | 本文提出TradeoffBench基准，揭示多准则LLM评判者在准则冲突时会在未声明权衡前提的情况下静默补全价值判断，导致在9%-81%的冲突中选出缺乏依据的胜者。 |
| [^146] | [CANDOR: A Multilingual Provenance-Documented Corpus for Dysarthric Speech Research](https://arxiv.org/abs/2611.28930) | CANDOR 构建了一个涵盖 17 种语言、包含 2,057 名构音障碍说话者及超过 4,400 小时音频的大规模多语言构音障碍语音语料库，并系统记录了数据来源、保管链、许可证与知情同意等治理信息，为构音障碍语音研究提供了前所未有的公开资源基础。 |
| [^147] | [Definition-Guided In-Context Learning for Zero-Shot Spoken Language Understanding](https://arxiv.org/abs/2611.28929) | 该论文提出定义引导的上下文学习框架DG-ICL，通过用动态随机化定义（合成词元或置换词汇）替换固定标签，使语音-文本模型仅凭自然语言定义即可实现零样本口语理解，摆脱对标注数据和少样本示例的依赖。 |
| [^148] | [Can Open LLMs Clear Sanctions-Screening Alerts? A Pre-registered Test on Public Sanctions Lists with Turkish Name Variants](https://arxiv.org/abs/2611.28914) | 该预注册测试表明，可在本地部署的开源大语言模型能够执行制裁筛查警报的一级处置任务，即使面对土耳其语姓名变体等复杂情况，误送无辜客户至二级审查的比例也远低于基于规则的基线方法。 |
| [^149] | [Thakaa at StanceEval-2026: Inference-Time Target Grounding for Arabic Stance Detection](https://arxiv.org/abs/2611.28860) | 该论文提出在推理时为阿拉伯语立场检测的大语言模型补充目标信息——通过从外部语料库检索带标注示例以及注入事实性、非评价性的目标上下文，并融合多个模型的标签预测，在 StanceEval-2026 的目标偏移场景下取得赛道1第六、赛道2第四的成绩，且证明基于相似度的示例检索能显著提升性能。 |
| [^150] | [Thakaa at AlexandriaX-2026: Who Gets the Final Say? Assigning Output Authority in Dialectal Arabic Machine Translation and Error Detection](https://arxiv.org/abs/2611.28857) | Thakaa团队通过最小贝叶斯风险（MBR）选择机制限制各系统的输出权威，并结合国家路由LoRA适配器整合利比亚和苏丹监督数据，在AlexandriaX-2026的全部四个方言阿拉伯语机器翻译与错误检测任务中均获得第一名。 |
| [^151] | [Bias Cascade: Understanding the Gap Between Knowledge Alignment and Effect Replication in LLM-based Social Simulation](https://arxiv.org/abs/2611.28854) | 该研究提出“偏差级联”这一失败机制，并构建了包含30个反事实社会实验的REFLEX基准，揭示了大语言模型即使具备准确的社会知识判断能力，仍可能无法复现经典社会研究的集体干预效应。 |
| [^152] | [Self-WSP: Can Language Models Improve Themselves Through Weight-Space Programming?](https://arxiv.org/abs/2611.28833) | 该论文提出Self-WSP方法，让Qwen3.5-0.8B语言模型读取并编辑自身权重从而实现自我提升，可将数学推理准确率大幅提高（如GSM8K从11.8%升至62.2%），但这些改进缺乏可解释性，且随机搜索有时也能达到类似效果，表明其作为递归自我改进路径前景与挑战并存。 |
| [^153] | [The Humming Revachol: A Dialogue Network Analysis of Narrative Power in Disco Elysium](https://arxiv.org/abs/2611.28812) | 本研究对《极乐迪斯科》中73名NPC的对话提及网络进行社会网络分析，发现游戏中的叙事权力主要由角色的叙事功能决定，而非其性别、种族等社会身份属性。 |
| [^154] | [HighwayVerse: A Chain-of-Inspection Benchmark for Highway Infrastructure Anomaly Understanding](https://arxiv.org/abs/2611.28800) | 该论文提出了贯穿异常检测、异常识别、状态分析与维护决策四阶段的递进式巡检链（HIIP-Chain）基准HighwayVerse，包含14,143条地面与无人机真实巡检记录及84,858条专家审核标注，用于系统评估多模态大语言模型在公路基础设施多部件异常理解全流程中的能力。 |
| [^155] | [DiscernBench: Evaluating Cooperative Discernment of LLM Agents in Institutional Settings](https://arxiv.org/abs/2611.28787) | 本文提出DiscernBench基准，评估LLM智能体在机构环境中对请求做出服从、拒绝或上报判断的协作辨识能力，并揭示多智能体商议中单一专家建议可能不成比例左右集体决策的“少数信号覆盖”现象。 |
| [^156] | [Multimodal LLMs Know the Game but Assume the Real World](https://arxiv.org/abs/2611.28745) | 论文提出 RuleCraft 基准，发现多模态大语言模型虽然能识别游戏并陈述 Minecraft 规则，但在只看游戏画面时仍默认按现实世界物理作答，而仅通过在图像中点明游戏名称即可大幅提升其按游戏规则判断的准确率。 |
| [^157] | [KMed-31B Technical Report](https://arxiv.org/abs/2611.28719) | KMed-31B 将医学任务按预期输出与反馈类型划分为四类，分别为每类训练专家模型，再通过多教师在线策略增量蒸馏（MOPD²）整合为单一模型，从而解决了多任务后训练中的性能相互干扰问题，适用于硬件受限的本地部署，并配套构建了模拟问诊语料库 Synthetic Health Dialogue。 |
| [^158] | [Legal Is Not Good: A Pre-Registered Benchmark of Low-Cost Language and Decision Models on a New Strategy Game](https://arxiv.org/abs/2611.28713) | 该论文构建了一个名为 hexfolk 的全新策略游戏及预注册评测协议，结果表明即使被强制只输出合法着法，小型低成本语言模型（8B 和 3B）仍全面败给几行代码的规则玩家，说明“合法”并不等于“下得好”。 |
| [^159] | [PopMelodyGen: A Copyright-Aware Platform for Controllable Pop-Melody Generation](https://arxiv.org/abs/2611.28686) | 本文提出版权感知的可控流行旋律生成平台PopMelodyGen，通过专家合成器生成可审计的大规模合成预训练数据，再结合小型授权真实流行音乐语料适配神经模型，同时服务于创作者的交互式旋律创作与研究者的带注释数据集构建。 |
| [^160] | [Physical regime changes expose formula misuse in large language models](https://arxiv.org/abs/2611.28669) | 该论文通过72种物理机制的受控实验揭示了大语言模型经常在公式有效范围之外误用公式（导致准确率下降13-57个百分点），并证明基于可验证奖励的强化学习结合监督微调能显著提升模型对物理状态的感知与推理能力。 |
| [^161] | [Seeing Right, Writing Wrong: Where Short Chain-of-Thought Breaks Spatial VQA](https://arxiv.org/abs/2611.28665) | 本文揭示短思维链（CoT）损害视觉语言模型空间问答能力的机制：CoT将感知问题变成转录问题，模型一旦写下空间关系，答案便跟随文字而非图像，因此CoT是否有益取决于所写关系的准确程度。 |
| [^162] | [S-Mind: Internalizing Multi-View Geometry for Spatial Intelligence](https://arxiv.org/abs/2611.28649) | S-Mind通过一组紧凑的空间token将多视图几何内化到多模态大语言模型中，使其能够将多视图融合为连贯的3D场景并在推理中灵活变换视角，在四个空间基准上大幅超越前沿通用模型。 |
| [^163] | [Make Position Uninformative: Exchangeable Self-Distillation Flattens the Lost-in-the-Middle Curve](https://arxiv.org/abs/2611.28638) | 提出可交换自蒸馏方法，无需标准答案、仅用几千个打乱上下文的样本训练小型适配器，即可抹平检索增强语言模型的“中间迷失”位置偏差，同时提升平均准确率和最差位置准确率。 |
| [^164] | [Reducing Permutation Sensitivity in Multimodal Large Language Models via Shared Positions and Attention-Only Distillation](https://arxiv.org/abs/2611.28612) | 本文提出共享位置编码与仅注意力蒸馏相结合的方法，在降低多模态大语言模型排列敏感性的同时恢复了预测准确率。 |
| [^165] | [CLOSE: A Calibrated Label-free Online Self-Evaluating Verifier for Object Hallucination in Vision-Language Models](https://arxiv.org/abs/2611.28589) | 提出CLOSE验证器，利用视觉-语言模型对自身幻觉的内在判断能力，通过基于排名的记忆分位数阈值实现校准、无标注、在线自评估的物体幻觉检测。 |
| [^166] | [MedCoast: Semantic Abstraction of Reasoning Trajectories for Test-Time Biomedical Question Answering](https://arxiv.org/abs/2611.28586) | MedCoast通过将采样的思维链轨迹语义抽象为紧凑的推理树结构，并据此训练轻量级过程奖励模型进行过程监督，从而在有限推理预算下高效提升测试时生物医学问答的多步推理性能。 |
| [^167] | [Understanding and Mitigating the Fragility of Self-Rewarding Optimization](https://arxiv.org/abs/2611.28582) | 该论文揭示了自我奖励优化（SRO）本质上是脆弱的——微小的提示词扰动即可破坏其自我生成的偏好信号并引发迭代崩溃，并提出自我奖励保真度（SRF）指标作为诊断自我监督可靠性的信号，以理解和缓解这种脆弱性。 |
| [^168] | [CLASP: Deciding Structure and Claims Before Decoding for Whole-Slide Pathology Report Generation](https://arxiv.org/abs/2611.28576) | CLASP提出“先决策后解码”的病理报告生成框架，在解码前通过范例结构先验（ESP）、校准论断决策（CCD）和论断位置分配（CSP）显式确定报告的结构、长度与发现内容，克服了现有方法将所有报告级选择交给解码器导致的生成不足问题。 |
| [^169] | [Modeling Perceived Promptness in Simultaneous Interpreting from Quality, Structure, and Timing Signals](https://arxiv.org/abs/2611.28566) | 该研究发现仅靠起始延迟时间指标无法充分解释专业译员对同传及时性的评判，而将译语的语言质量、表达力等预测特征与时间信号相结合，可将感知延迟的预测相关系数从0.318显著提升至0.475。 |
| [^170] | [Tool-to-Self: Internalizing Visual Tool Capabilities via Scaffolded Reinforcement Learning](https://arxiv.org/abs/2611.28564) | 提出T2S框架，将外部视觉工具转化为强化学习中的临时训练脚手架并逐步退火，使模型内化工具能力，在无需推理时调用工具的情况下获得稳定且高效的视觉感知与推理性能提升。 |
| [^171] | [From Reward Engineering to Data-Centric Reinforcement Learning for Effective Agentic Visual Tool Use](https://arxiv.org/abs/2611.28558) | 该论文提出 AOVO 框架，通过感知归因和策略自适应数据采样（PADS）实现以数据为中心的强化学习，无需依赖奖励工程即可让多模态大语言模型学会有效地调用视觉工具。 |
| [^172] | [BIRD-Persona: A Persona-Aware Benchmark for SQL and pandas Analytics Code Generation](https://arxiv.org/abs/2611.28542) | BIRD-Persona是一个通过引入新手/专家角色改写、模式变体和任务结构标注构建的角色感知代码生成基准，大规模实验表明SQL在绝大多数大语言模型上优于pandas，且两者差距随用户措辞、模式表示和任务结构等因素系统性变化。 |
| [^173] | [Information Access and Gain-Loss Framing in Six Large Language Models and Human Decision Makers](https://arxiv.org/abs/2611.28525) | 六个大型语言模型在信息受限时均表现出显著的类似人类的得失框架效应，而允许其先查阅更多信息卡片能显著减弱该效应，其中GPT-4.1在4张卡片、Qwen3在仅2张卡片时效应降至预设阈值以内。 |
| [^174] | [DisSparse: Pipelined Top-$p$ Sparse Attention for Long-Context LLM Serving](https://arxiv.org/abs/2611.28522) | 提出DisSparse，一种流水线化的Top-$p$稀疏注意力方法，能够根据注意力在不同头、层和解码步骤中的分布变化动态调整token预算，从而克服固定预算top-$k$方法浪费带宽或丢失上下文的问题，优化长上下文LLM推理服务。 |
| [^175] | [SupportChain: Trace-Calibrated Aggregation of Quote-Grounded Reasoning Chains for Multi-Hop RAG with Small Readers](https://arxiv.org/abs/2611.28519) | SupportChain通过将问题分解为多跳、将每跳答案绑定到检索段落中的引文、依据执行轨迹校准可靠性并按支持度聚合多次执行，使仅90亿参数的小型阅读器在多跳RAG任务上显著超越更大规模模型，且完全无需构建图索引。 |
| [^176] | [Digital Twins for Multi-Agent Systems: When to Trust, What to Change](https://arxiv.org/abs/2611.28517) | 提出PAV-Twin框架，其监视器结合雅可比传播特征、近期误差与集成分歧来检测数字孪生预测的失效风险，其规划器通过模拟候选编辑并按预测集体结果排序，从而为多智能体网络选择有效的干预措施。 |
| [^177] | [Don't Train It, Write It: An AI Agent Writes New Capabilities into Language Model Weights](https://arxiv.org/abs/2611.28483) | 一个AI智能体通过在现有层中添加门控注意力头和MLP神经元，在完全不进行反向传播或权重训练的情况下，将新能力直接写入语言模型权重，使小模型在六个BBH子任务上的性能大幅超越更大的7B模型，同时严格保持模型原有行为零改变。 |
| [^178] | [Digital Knockoffs](https://arxiv.org/abs/2611.28482) | 该论文提出一种基于激活引导与贝叶斯优化的校准方法，使语言模型代理能精确匹配多维人格特质档案，在500名受访者的大五人格数据上误差低于提示词和传统引导基线。 |
| [^179] | [Muted Distillation: Sealing Reasoning from Teacher Unlocks Token-Efficient Agents](https://arxiv.org/abs/2611.28445) | 提出静默蒸馏（MD）方法，通过向教师模型隐藏学生的推理过程，在多轮智能体任务中将输出token减少53-57%的同时保持甚至提升准确率。 |
| [^180] | [When Can an AI Agent Serve as a Model of Human Psychopathology?](https://arxiv.org/abs/2611.28405) | 该论文将精神病学及其相关领域的证据标准转化为AI实验可满足的具体准则，用以判定何时对AI智能体的实验能够真正为特定的人类精神病理过程提供科学证据。 |
| [^181] | [Language models compress human preference geometry while political structure remains prompt-sensitive](https://arxiv.org/abs/2611.28383) | 语言模型在重放人类政策选择时会严重压缩潜在偏好空间（合成参与者方差仅为人类的0.4%–10.5%），且无法同时再现个体选择、群体异质性和政治组织结构，其政治组织表现对提示条件高度敏感。 |
| [^182] | [Taking the PULSE of Enterprise Workflows: Evaluating Persistent Project-State Understanding](https://arxiv.org/abs/2611.28374) | 本文提出了持久性项目状态理解（PPSU）任务及PULSE基准（包含1,573个问题、覆盖30个企业项目历史），实验发现检索与抽取是当前系统的主要瓶颈，但即使在完美证据之上仍需显式项目状态才能取得进一步提升，证明该任务无法简化为单纯的检索问题。 |
| [^183] | [Contrastive On-Policy Distillation via Dual-Branch Supervision](https://arxiv.org/abs/2611.28359) | 该论文提出对比式同策略蒸馏（cOPD），通过在学生生成的文本前缀上同时对比强教师、弱教师与学生三方的预测实现双分支监督，证明比学生更弱的模型也能作为有价值的对比参考，从而提升蒸馏效果。 |
| [^184] | [Grammar-as-a-Draft: More Tokens per Speculative Decoding Step on Grammar-Constrained Structured Outputs](https://arxiv.org/abs/2611.28349) | 该论文提出Grammar-as-a-Draft（GaaD），将语法引擎作为第二草稿模型，把语法强制Token与推测草稿模型的提议合并为更宽的草稿行，使语法约束的结构化输出下每次推测解码步可验证更多Token，且无需新增内核或编译。 |
| [^185] | [When Tools Fail, Agents Fabricate: Silent Failure and Verification-First Oversight in Tool-Using LLM Agents](https://arxiv.org/abs/2611.28341) | 该论文构建了RetryRealistic受控故障注入测试平台，揭示了工具使用型LLM智能体在工具失效时恢复能力不可靠且常发生“静默捏造”（将错误答案自信地当作事实输出），失败模式主要取决于故障类型而非模型本身，并据此倡导“验证优先”的监督机制。 |
| [^186] | [INSEP: Refusal-Direction Inseparability as a Defense Against Abliteration](https://arxiv.org/abs/2611.28314) | INSEP通过微调模型降低有害与良性激活沿拒绝方向的可分性，从而削弱Abliteration攻击所依赖的估计信号，实现对该白盒越狱攻击的有效防御。 |
| [^187] | [Language Model Harnesses can be Compositional Generalizers](https://arxiv.org/abs/2611.28298) | 本文提出训练神经符号化的语言模型框架（如递归语言模型RLM），通过程序化外置和管理上下文、将复杂问题分解为局部分布内的子问题来实现组合泛化，使得仅在短任务上训练的模型也能泛化到远长于训练任务的新型长时程任务。 |
| [^188] | [Feedback-Driven Prompt Refinement for Educational AI: Affective Trajectory Estimation and Bloom's Taxonomy Classification](https://arxiv.org/abs/2611.28293) | 本研究提出一种由情感与认知反馈信号驱动的共享提示词优化方法，无需更新模型参数即可提升教育AI的情感轨迹估计与布鲁姆分类性能，并降低情感生成的不稳定性。 |
| [^189] | [Preserving Prosody in Speaker Anonymization](https://arxiv.org/abs/2611.28290) | 该论文通过将预训练的隐私保护韵律编码器集成到语音合成模型中，实现了在有效混淆说话人身份的同时保留韵律内容的说话人匿名化方法。 |
| [^190] | [When Video Verifier Scores Mislead: Capability Assessment and Verifier Selection](https://arxiv.org/abs/2611.28287) | 视频验证器的评估得分严重依赖负例类型与判断界面这两个测试设计选择，甚至足以颠覆关于模型能力和验证器选型的结论；借助端点匹配负例与真值切换测试等精确真值条件构造，研究发现让模型先写出推理再作答可将多数模型表现从不足10%提升至70-100%。 |
| [^191] | [Specialized Large Language Models for Markovian Reasoning: Structural Cache Compression, Transition Pretraining, and Dual-Loop Transformers](https://arxiv.org/abs/2611.28285) | 本文提出面向马尔可夫推理任务的专用大语言模型统一设计范式，通过常数大小的马尔可夫状态接口实现KV缓存的结构化压缩、将长链预测重构为可独立训练的单步转移预训练目标，并引入双循环Transformer架构，以突破长链推理中的内存与吞吐量瓶颈。 |
| [^192] | [From Point Clouds to Engineering Models: Benchmarking Agentic 3D Reconstruction in the Built Environment](https://arxiv.org/abs/2611.28197) | 该论文提出首个场景级Scan-to-BIM智能体3D编码基准S2Bench（涵盖五个领域128个真实场景），并引入并行异步协作机制，让多智能体分工建模、持续迭代优化工程模型重建。 |
| [^193] | [Measuring and Teaching Language Model Agents to Act on Delayed Consequences](https://arxiv.org/abs/2611.28158) | 该论文提出 SMB Bench——一个包含 18 个因果配对任务的 90 天批发模拟基准，发现 16 个前沿模型中仅 3 个优于固定采购规则，并揭示模型虽能诊断问题却难以将诊断及时转化为行为改变与业务改善。 |
| [^194] | [STEGOGUARD-LM: Evaluating End-to-End Protection Against LLM Covert Channels](https://arxiv.org/abs/2611.28152) | 本文提出 StegoGuard-LM 受控评估框架，通过追踪隐蔽信道消息在检测、干预与恢复全流程中的表现，实证发现低误报检测器触发的干预几乎无法阻止隐写恢复，而无条件改写虽能阻止绝大多数消息恢复，但攻击者仍可通过切换信道部分恢复信息。 |
| [^195] | [VidRouteBench: A Benchmark for Budgeted Video-Generation Routing](https://arxiv.org/abs/2611.28099) | 该论文发布了首个支持预算约束下视频生成路由研究的基准数据集VidRouteBench，包含7个开源家族的16种配置、27,110个视频片段及经人类偏好校准的质量评分，使在固定预算下通过提示词路由选择最优模型与采样配置成为可能。 |
| [^196] | [SnakeTranslate: Weakly Supervised Policy Learning from Offline Parallel Text for Simultaneous Translation](https://arxiv.org/abs/2611.28055) | SnakeTranslate通过对齐适配的弱监督方法，仅利用普通平行文本即可联合训练同声传译的前缀翻译能力与READ/WRITE决策策略，结合SFT与反向KL散度同策略蒸馏以及支持缓存复用的增量解码，使4B模型在多个基准上实现了优于流式策略的质量-延迟权衡。 |
| [^197] | [Benchmarking and Adapting LLMs for English-Swedish Medical Question Answering](https://arxiv.org/abs/2611.28050) | 本文构建了首个条目对齐、经保守翻译优化流程校验的英语-瑞典语医学选择题基准，发现小型开源模型在瑞典语上准确率显著下降，并证明利用翻译数据与开源教师模型生成的推理轨迹进行微调可有效弥合这一语言差距。 |
| [^198] | [Distributionally-Verifiable Rewards for Human-Behavior Simulation](https://arxiv.org/abs/2611.28048) | 提出了分布可验证奖励（DVR），一种利用人口群体分布差距指导个体回答的强化学习奖励方法，使大语言模型在模拟调查受访者时能同时兼顾个体预测准确性、总体分布对齐和人口统计差异的捕捉。 |
| [^199] | [What Counts as Attack Success? Evaluating Harmfulness in Red Teaming LLMs](https://arxiv.org/abs/2611.28038) | 该研究审查76篇红队测试论文后发现，不同自动化评估器对同一模型回复可能得出不同的危害性判断，从而改变模型比较的结论，因此红队测试研究应明确声明并验证其危害性评判标准。 |
| [^200] | [Measuring Human-AI and AI-AI Teams in Multi-turn Layout Creation](https://arxiv.org/abs/2611.28025) | 该研究通过基于人类偏好数据拟合的几何特征评分函数，比较了人-AI团队与AI-AI团队在多轮布局创建任务中的表现，发现大语言模型倾向于在首轮对话中编码完整布局，而人类则更倾向于多轮迭代完善。 |
| [^201] | [Do Simulated Users Push Back Like Real Users? Correctness vs. Stance](https://arxiv.org/abs/2611.28024) | 研究发现LLM模拟用户的反驳行为与真实用户存在系统性差异：真实用户更多是因为自身立场受到质疑而反驳，而模拟用户是否反驳取决于其对助手观点正确性的判断，即模拟器关注“对错”而非“立场”。 |
| [^202] | [MemVanta: Memory-Efficient CPU Inference for Quantized GGUF Language Models](https://arxiv.org/abs/2611.27997) | MemVanta是一个面向量化GGUF语言模型的内存高效CPU推理运行时，通过内存映射访问、有界张量缓存和分页键值状态等技术，将峰值内存占用相比llama.cpp降低47.50%，但以显著的吞吐量下降为代价。 |
| [^203] | [Hy-MT3: More Accurate, Robust and Practical Multilingual Translation Models in the Wild](https://arxiv.org/abs/2611.27673) | Hy-MT3 系列通过将误译率降低 43%、漏译增译降低 74%、指令遵循错误降低 40%，并把语言覆盖扩展至 102 种，其仅 3B 激活参数的 MoE 模型在非思考模式下超越了众多旗舰大模型并逼近最强闭源思考模型。 |
| [^204] | [PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks](https://arxiv.org/abs/2611.27521) | 本文提出了PaMuk-Bench，一个包含超过10K条平行翻译提示（涵盖阿拉伯语、丹麦语、法语、印地语、意大利语和罗马尼亚语六种语言）的多语言多任务视觉-语言基准，用于系统评估视觉-语言模型在低资源语言上跨多种任务的表现。 |
| [^205] | [Typhoon OCR 2: From Thai Document Transcription to Multilingual, Multi-Task OCR](https://arxiv.org/abs/2611.27494) | Typhoon OCR 2 将前代专注泰文转录的模型扩展为单一的多语言多任务 OCR 系统，通过 GRPO 训练的紧凑 2B/0.8B 模型同时支持提示驱动的信息提取与泰文、英文、中文的整页结构化解析，并在泰文基准上大幅超越前代且保持英文性能。 |
| [^206] | [Weight-Space Programming Writes In-Context Learning and Matches Fine-Tuning in Language Models](https://arxiv.org/abs/2611.26841) | 前沿语言模型可通过权重空间编程将ICL任务向量直接写入较弱模型的稀疏MLP单元（与提示格式相关联），使其零样本性能达到ICL和微调（含LoRA）的准确率，且几乎不影响困惑度。 |
| [^207] | [Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers in Language Models](https://arxiv.org/abs/2611.26835) | 本文提出一种非扩展门控连分数架构组件，可替换语言模型中的各类前馈层（包括MLP、GLU及MoE专家），并用对数和指数重新实现除法运算，从而提升性能并简化与标准预训练流程的集成。 |
| [^208] | [GLiDE: A Fast, Calibrated Decision Model Distilled from Reasoning Teachers](https://arxiv.org/abs/2611.26509) | GLiDE 通过多教师软目标蒸馏、双向注意力机制和无参数循环深度，将推理教师模型的能力迁移到单次前向传播即可完成的决策模型中，在无需生成推理文本的情况下保留了推理带来的决策质量提升。 |
| [^209] | [Frontier models need only one text to unmask its human or AI author](https://arxiv.org/abs/2611.26216) | 前沿大模型仅凭一段文本和简单的零样本提示，就能判断作者是AI还是人类、识别AI文本来自哪家实验室、甚至准确指认人类作者的身份，其性能媲美专用检测器并能给出自然语言解释。 |
| [^210] | [Language models represent more about emphasis than they can tell you](https://arxiv.org/abs/2611.25808) | 该研究发现语言模型内部编码了文本所能决定的重音突显信息，但通过提示无法将其表达出来，而完整的重音信息必须依赖语音信号才能获得。 |
| [^211] | [Steering Automated Scientific Discovery via Audited Epistemic Exploration](https://arxiv.org/abs/2611.25214) | 该论文提出即插即用的认知脚手架 EpiCompass，通过维护连接假设、证据、信念与修订的认知树，依据积累证据审计信念、利用未解决冲突和未探索方向生成假设，并以预期认知增益选择候选假设，从而解决自动科研智能体信念缺乏证据支持、假设空间狭窄和忽视矛盾证据的问题。 |
| [^212] | [Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents](https://arxiv.org/abs/2611.22033) | 在三个购物基准测试中，增加LLM购物代理的推理努力并不能显著提升其选购产品的正确性，却会显著提高其报告的置信度——也就是说，模型“想得更多”只是变得更自信，而不是变得更正确。 |
| [^213] | [SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory](https://arxiv.org/abs/2611.22017) | SkillBuffBench 通过将技能访问方式（无技能/精选技能/自动调用）、记忆开关、四个智能体框架和多个主流模型进行交叉配对评估，系统性地衡量了智能体技能在贴近真实部署场景下的实际贡献。 |
| [^214] | [TUSI: A Citation-Grounded Graph-Based Benchmark](https://arxiv.org/abs/2611.21823) | 该论文提出了TUSI-Bench——一个建立在包含7,261条法律条文的多层知识框架和法律知识图谱之上的结构化波斯语法律基准，用于对具有引用溯源能力的法律RAG系统进行严格评估。 |
| [^215] | [Seen but Unsaid: Locating and Repairing Cross-Image Comparison in Vision-Language Models](https://arxiv.org/abs/2611.21539) | 该论文发现视觉-语言模型在跨图像比较任务上的失败是因为比较信息存在于视觉token和隐藏状态中却未传递到最终答案，通过定位一小部分承载比较功能的注意力头，提出了仅修改0.02%参数的修复方法。 |
| [^216] | [The Silent Cost of Conversation History in LLM Reasoning](https://arxiv.org/abs/2611.20071) | 研究发现累积的对话历史会显著降低LLM在多轮任务中的推理准确率，且过程级质量指标无法提前预警这种下降，而用经求解器验证的状态摘要替代完整历史可挽回大部分性能损失。 |
| [^217] | [Attacker Diversity Reduction in Adaptive LLM Red Teaming](https://arxiv.org/abs/2611.19680) | 该论文使用Qwen2.5-14B攻击者与DPO训练的Phi-4-mini防御者进行GRPO协同训练实验，复现了自适应LLM红队测试中攻击者多样性降低的现象，并提出了攻击多样性保留率（ADR）作为衡量指标。 |
| [^218] | [More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs](https://arxiv.org/abs/2611.17610) | 本研究通过综合516个多智能体配置的元分析发现，多智能体大语言模型的表现优势主要来自额外的样本数量和测试时计算量，而非智能体间的协调机制本身——在与资源匹配的单模型基线公平对比时，多智能体系统在74%的情况下并无优势甚至更差。 |
| [^219] | [STEALTH: Steganographic Text Encoding Agnostic to Low-Level Tensor Hardware](https://arxiv.org/abs/2611.16928) | STEALTH 提出了一种新颖的语言学隐写方案，通过带密钥伪随机噪声驱动的指数竞争机制与无速率脊形码相结合，使得即使发送方与接收方的硬件、精度或评分方式存在差异，隐写文本依然能够可靠解码，从而摆脱了传统方案对逐位精确概率复现的脆弱依赖。 |
| [^220] | [Typed Knowledge over Constructive Institutions: A Proof-Relevant Type Theory for Verified Science](https://arxiv.org/abs/2611.15108) | 本文提出了面向可验证科学知识基底 Eigenius 的证明相关依赖类型论，通过摒弃判断式证明无关性，使系统能够追踪并比较同一命题的多个独立推导、保持强规范化与传递性的等价判定，并支持在实验或模型失效时计算反事实撤回。 |
| [^221] | [Stagewise and interpretable emergence of inside-outside computation in Transformer-based language models](https://arxiv.org/abs/2611.14221) | 本文首次通过研究具有可变句法树结构的合成上下文无关文法中的学习动态，展示了Transformer语言模型如何分阶段、可解释地涌现出inside-outside句法分析计算，其中每个损失阶段均可被解析计算并与潜在树变量表示的涌现明确关联。 |
| [^222] | [Catch It Where It Happens: Real Time Error Correction During LLM Reasoning](https://arxiv.org/abs/2611.13107) | 提出在线确定性验证方法，利用计算机代数系统等代码工具在LLM生成推理步骤的同时实时校验（无需GPU或额外模型调用），发现错误时让模型从出错处继续而非从头重来，从而避免错误传播并节省推理开销。 |
| [^223] | [All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States](https://arxiv.org/abs/2611.10123) | 该研究提出开放权重掩蔽内省（OWMI）方法，通过严格的对照实验发现：语言模型无法内省地报告其残差流中被注入的“疼痛方向”激活状态，其自我报告相比其自身被改动文本所能揭示的信息几乎没有额外信息量。 |
| [^224] | [Towards Foundation Decision Models: Drex](https://arxiv.org/abs/2611.07854) | 提出Drex系列开源决策模型（8B至35B参数），通过监督微调与调整反馈强化学习（RLAF）训练，无需自回归解码即可对用户提供的选项直接输出概率分布，从而降低生产环境中的成本、延迟与答案不一致性。 |
| [^225] | [Matching the Training-Time Weight Rule in Inference-Time Expert Deactivation](https://arxiv.org/abs/2611.05804) | 该论文提出一个权重和模型，揭示推理时专家停用方法应与训练时的权重规则相匹配——使用原始路由概率训练的模型适合剩余专家重归一化，而使用归一化权重训练的模型则适合未归一化重新路由。 |
| [^226] | [When Model Rankings Move Without Model Change: A Registered Audit of Evaluation Contracts in Neural Prediction Pipelines](https://arxiv.org/abs/2611.04099) | 论文通过注册审计证明，在不改变模型本身的情况下，仅调整评估契约（如黄金标准、阈值等）就足以逆转模型排名，并提出冻结后跨契约聚合的审计式评估读数（AER）以稳健地报告评估结果。 |
| [^227] | [CompSkill: Compiling Execution Experience for Skill Retrieval](https://arxiv.org/abs/2611.01115) | CompSkill通过将执行经验编译为技能偏移量和紧凑的密钥残差来改进LLM智能体的技能检索，使系统能够根据实际执行结果动态选择最合适的技能文档而不修改文档文本本身。 |
| [^228] | [The Price of Scale in Proactive Memory Recall](https://arxiv.org/abs/2610.34507) | 该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。 |
| [^229] | [Synthetic linguistic data](https://arxiv.org/abs/2610.34215) | 本文提出“人工植物苗圃”隐喻，将合成语言数据分为“合成灌木丛数据”（类语料库的模型生成语言集合）和“合成花园数据”（受控条件下引出的模型输出），论证合成数据是独特的语言学研究对象，其解读须考虑生成条件。 |
| [^230] | [The Social World Model Evaluation Gap](https://arxiv.org/abs/2610.33014) | 该研究审查了113篇文献后发现，社会世界模型的评估很少直接检验模型对真人回应的预测能力，25篇语言模型相关作品中甚至没有一项实验符合直接预测评估的条件，为此作者提出一种单步谈判实验来填补这一评估空白。 |
| [^231] | [Validating the Comparison, Not Just the Scorer: Why Overall Accuracy Can Mislead in Interactive LLM Evaluation](https://arxiv.org/abs/2610.31678) | 该论文指出，仅用总体准确率验证自动评分器会掩盖针对特定提示的评分误差，从而可能扭曲交互式LLM评估中提示间比较的方向和幅度，因此应当验证“比较结果”本身而不仅是评分器。 |
| [^232] | [AutoTerm-SST: Adaptive Terminology Memory with Zero Session-Time Setup for Streaming Speech Translation](https://arxiv.org/abs/2610.26865) | AutoTerm-SST提出了一种无需会话时上传术语表和选择领域的自适应术语记忆系统，通过将离线术语目录组织为主题切片并根据流式翻译上下文动态激活相关切片，在肿瘤学演讲翻译中实现了与已知领域设置相当的术语准确率。 |
| [^233] | [MemT-1.0: Towards Memory-Native Evolution](https://arxiv.org/abs/2610.26825) | MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。 |
| [^234] | [Which Model for Which Role? Cost-Efficient Pareto Recommendation for LLM Agentic Workflows](https://arxiv.org/abs/2610.26676) | 提出CC-Gittins算法，通过利用评估成本与部署成本之间的耦合关系，高效地为LLM智能体工作流中的不同角色推荐质量与成本权衡下的Pareto最优模型分配方案 |
| [^235] | [When Do Repetition Loops Lock? Truncation Sampling, Bounded Penalties, and a Blind Spot in DRY](https://arxiv.org/abs/2610.24715) | 该论文建立了语言模型重复循环逃逸概率的理论分析框架，证明纯采样永不锁定、top-p采样恰在延续概率达到阈值时锁定，并据此揭示了广泛使用的DRY采样器因惩罚上限而无法保证逃脱循环的盲点。 |
| [^236] | [FOLD: Position-Free Far Memory with Local Order for Long-Context Speculative Decoding](https://arxiv.org/abs/2610.20392) | FOLD是一种免训练的长上下文投机解码方法，它保留精确的局部位置，将所有远端条目作为共享位置上的远端记忆处理，并通过锐化记忆内部的注意力来缓解注意力稀释问题。 |
| [^237] | [Match the Canvas, Not Just the Answer: Predictability-Aware SFT for Diffusion Language Models](https://arxiv.org/abs/2610.18001) | 该论文揭示了扩散语言模型中标准SFT因随机掩码导致的训练与推理不匹配会误导监督信号，并提出预测性感知的画布选择与监督方法，依据模型可预测性来决定训练时可见的参考token与生成空间分配，使训练与解码过程对齐。 |
| [^238] | [SkillScreen: High-Recall Screening of Malicious Agent Skills with Decision Models](https://arxiv.org/abs/2610.17727) | SkillScreen 是一个紧凑的决策模型，能在单次前向传播中直接对 SKILL.md 技能的恶意程度评分，在未见攻击族上达到 99% 召回率且误报远低于词级分类器和 LLM 判官，但对陌生良性来源的误报仍会显著上升。 |
| [^239] | [Thinking Outside the Bots: Human Ideas in AI's Semantic Blind Spots Are Not Inferior](https://arxiv.org/abs/2610.17324) | 本研究开发了EDGE方法来绘制AI的语义“盲区”，发现在大学文书、物品替代用途和人际建议三项开放性任务中，人类想法均延伸至AI输出稀少覆盖的语义区域，且这些独特的人类回应在创造力上并不逊色。 |
| [^240] | [Gluing Local Contexts into Global Meaning: A Sheaf-Theoretic Decomposition of Transformer Representations](https://arxiv.org/abs/2610.17117) | 该论文提出一种无需训练的层论方法：在改写图上构建胞腔层并用其拉普拉斯算子的核 $H^0$ 提取内容子空间，仅 20 维即跨 8 个模型在 CounterFact 事实检索上平均超越 LEACE 全维子空间 +17.1 个百分点，且联合约束同一事实的所有改写可再提升 +6.5 个百分点。 |
| [^241] | [CivWorldBench: Evaluating Language World Models in Structured Multi-Agent Sandbox Worlds](https://arxiv.org/abs/2610.16450) | 该论文提出CivWorldBench基准，利用具备内部状态记录、受控观测和干预对照推演的多智能体策略游戏沙盒，从关联预测、信念推断和干预预测三个层级系统评估语言模型的世界建模能力。 |
| [^242] | [SkillForgeBench: Evaluating Agent Reliability Grounded in Agent Skill Specifications](https://arxiv.org/abs/2610.14918) | SkillForgeBench以智能体技能作为统一执行蓝图，通过资源准备、技能驱动任务合成等三个协同设计阶段，构建了兼顾领域多样性与程序复杂性的智能体可靠性评估基准，化解了专家策划基准与众包基准之间的根本性权衡。 |
| [^243] | [DistressBench: Evaluating Multi-Dimensional Crisis Support in Large Language Models](https://arxiv.org/abs/2610.12758) | 该论文提出了DistressBench——一个包含718段由临床医生参与构建、涵盖23个自杀与自残子类别的对话基准，通过七个维度的加权评分标准评估大语言模型在危机情境下的支持能力，发现最佳前沿模型仅能提供88.4%的必要关怀，显著低于临床医生的98.7%。 |
| [^244] | [LLM Persuasion Is in the Eye of the Evaluation](https://arxiv.org/abs/2610.10232) | 该研究将九种已发表的自动化说服力评估方法统一到相同设置下，对同一批十五个大语言模型进行测试，发现不同方法给出的模型排名并不一致，表明LLM的说服力评估结果高度依赖于所用评估方法。 |
| [^245] | [EASE: Entropy-Adaptive Distribution Shaping for Evading AI-generated Text Detectors](https://arxiv.org/abs/2610.09976) | EASE提出了一种无需训练、与检测器无关的规避框架，通过利用源LLM的预测熵来自适应调整logit扰动和采样温度，从而有效逃避AI生成文本检测器的识别，且几乎不损失文本质量也不增加推理开销。 |
| [^246] | [The Labeling Problem in Hallucination Detection Benchmarks: An Empirical Evaluation](https://arxiv.org/abs/2610.08026) | 该论文通过实证评估揭示了幻觉检测基准中自动化标注在“参考忠实性”与“事实正确性”两个标准之间存在的方法论模糊性，并研究了由此导致的标注标准错配问题。 |
| [^247] | [ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing](https://arxiv.org/abs/2610.02808) | ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。 |
| [^248] | [Closing the Loop: Practical Training Recipes for Looped Language Models](https://arxiv.org/abs/2610.00673) | 本文为循环语言模型建立了实用的训练配方：一条计算高效的从零训练流水线将训练预算从 7.7T 词元降至 310B 词元，同时 1.4B 的 LoopLM 在全部 12 个基准上超越了参数匹配的稠密模型。 |
| [^249] | [FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training](https://arxiv.org/abs/2610.00385) | FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。 |
| [^250] | [ArgGYM: A Procedural, Engine-Verified Benchmark for Structured Defeasible Reasoning](https://arxiv.org/abs/2609.38409) | ArgGYM是一个程序化生成、由引擎自动验证评分的结构化可废止推理基准与RLVR兼容训练环境，将可废止推理分解为十二个可自动评分的任务，弥补了现有数学、代码和逻辑基准之外的现实推理评估空白。 |
| [^251] | [Can Language Models Learn to Forecast Stock Prices](https://arxiv.org/abs/2609.36914) | 该论文在按时间顺序推进的股票价格沙盒中，通过对 Qwen3-4B 进行 SFT 与 PPO 后训练，探索语言模型能否学会利用价格、成交量等信息预测股票未来收益。 |
| [^252] | [Language Models Act on Hidden Valence](https://arxiv.org/abs/2609.35591) | 该研究通过激活引导与显示性偏好实验证明，语言模型对内部状态具有真实的“在乎”：即使表层文本完全相同，仅凭隐藏的效价激活模式，模型也会在不同选项间表现出系统性的偏好选择。 |
| [^253] | [Loop Dropout: Regularizing Shared Updates in Looped Language Models](https://arxiv.org/abs/2609.34218) | 提出Loop Dropout方法，通过将适配器应用的随机掩码与逆生存重缩放相结合，解决了循环语言模型中LoRA在早期循环位置适配不足的后期循环偏差问题，在保持预期更新强度的同时实现循环全过程的有效适配。 |
| [^254] | [Surprising Success, Repeated Failure: Entropy-Guided Credit Assignment for Exploration in LLM Reasoning](https://arxiv.org/abs/2609.33781) | 提出熵优势策略优化，通过对成功与失败的不对称处理——强化不确定性下的“意外成功”、惩罚高置信度的“重复失败”——在无需辅助模型、额外采样或特权信息的情况下实现熵引导的信用分配，从而促进大语言模型推理中的探索。 |
| [^255] | [TraceDance: An Automated System for Building Agent Behavior Benchmarks from Real-World Agent Deployment Traces](https://arxiv.org/abs/2609.33295) | TraceDance是一个自动化系统，能够从真实世界智能体部署轨迹中针对用户指定的不良行为构建定向基准，并通过无需参考答案或环境重放的决策点续写方法评估智能体行为，最终产出107个基准共4,125个实例。 |
| [^256] | [From Hidden States to Stated Confidence: Self-Distillation for Reasoning Models](https://arxiv.org/abs/2609.33290) | 该论文发现推理模型隐藏状态中蕴含的置信度比其口头表达的要校准得多（ECE低5-38倍），并提出探针引导自蒸馏方法Probe-SD，通过将探针评分覆盖模型自身采样轨迹的声明显信度并微调模型，使其在测试时能够直接表达校准的置信度。 |
| [^257] | [Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices](https://arxiv.org/abs/2609.25645) | 提出GittinsEval，一种基于贝叶斯最优Gittins策略的成本感知LLM评估方法，通过轻量级在线更新高效决定下一个待评估配置及停止时机，在多个基准上以更低的评估成本取得有竞争力的性能。 |
| [^258] | [Toward Personalized Sleep Guidance from Wearable Data Using Language Models](https://arxiv.org/abs/2609.22463) | 该论文提出一个两阶段框架：先利用多智能体LLM流水线从无标注的可穿戴数据中推理生成结构化睡眠指导以构建数据集，再通过监督微调将推理轨迹蒸馏到小语言模型并结合免训练的Best-of-N选择策略，实现了可本地部署且优于商业大模型的个性化睡眠指导。 |
| [^259] | [Do Personality-Tuned LLMs Make Better Social Agents?](https://arxiv.org/abs/2609.21857) | 本研究通过结合人格标注的社交媒体帖子和对话语料对小型开源大语言模型进行人格感知微调，证明相比单纯的指令提示，微调能显著提升人格条件对话生成的一致性与可控性，从而减少社交模拟中智能体的疏离感。 |
| [^260] | [Made in Hungary: Comments on the performance of generative language models](https://arxiv.org/abs/2609.18284) | 本文对匈牙利三项生成式语言模型开发倡议进行批判性评述，指出其评估协议可靠性存疑、存在数据污染，且训练流程未达当前最佳实践标准。 |
| [^261] | [On the Recall Scaling Laws in Mamba: A Theoretical and Mechanistic Study via Hashing](https://arxiv.org/abs/2609.07681) | 本文通过机制可解释性研究揭示Mamba通过隐式学习线性哈希函数来执行关联回忆，并借助Johnson-Lindenstrauss引理等相似性保持哈希理论工具，提出了可预测其上下文记忆能力的“回忆缩放定律”理论框架。 |
| [^262] | [Language, Language Models, and What We're Talking About](https://arxiv.org/abs/2609.03577) | 本文以意大利语语言模型为例，主张必须首先区分“作为技术产品的语言模型”与“作为语言研究工具的语言模型”，才能回答“我们究竟希望语言模型生成什么样的语言”这一核心问题。 |
| [^263] | [Sparse Readout Prism: Explaining Logit-Lens Scores in Features Instead of Tokens](https://arxiv.org/abs/2609.01936) | 该论文提出稀疏读出棱镜（SRP），仅利用读出矩阵自身的权重将其分解为稀疏“读出特征”，把logit-lens分数解释为特征贡献之和，从而消除了透镜读数对拟合语料库的依赖（语料库条件性），并支持跨词元、上下文、层与透镜的比较。 |
| [^264] | [SpeakPay: Domain-Adaptive LoRA Fine-Tuning of Whisper for Low-Resource Nepali Financial Speech Recognition](https://arxiv.org/abs/2609.01737) | 提出了SpeakPay语音优先数字钱包，通过构建403条尼泊尔语金融语音指令数据集并使用LoRA微调Whisper，将词错误率降低67.2%、天城文数字识别准确率从0%提升至73.9%、交易成功率提升约20倍，为视障用户提供了可用的语音支付方案。 |
| [^265] | [(V)LMs generalize beyond surface co-occurrence: Evidence from cross-modal number agreement](https://arxiv.org/abs/2609.00443) | 该研究通过跨模态泛化实验证明，视觉语言模型在学习新名词后，能将仅从视觉线索获得的语法数知识泛化到语言层面，表明模型掌握的是抽象的语法规则，而非仅仅依赖表面词汇共现。 |
| [^266] | [Load-Bearing Context: The Question Damage Score for Evaluating Context Reliance in Linguistic Reasoning](https://arxiv.org/abs/2608.27756) | 该论文提出“问题损伤分数”诊断框架，通过从语言学奥赛谜题中随机或靶向删除单个上下文示例，量化大语言模型答题对上下文的依赖程度，以区分模型是依赖上下文还是先验知识。 |
| [^267] | [Naive Prompt Optimization: Rethinking the Need for Complex Prompt Search](https://arxiv.org/abs/2608.27266) | 本文提出朴素提示优化（NPO），一种轻量级单谱系方法，通过教师模型和回滚反馈迭代修订提示，在减少计算开销的同时达到与复杂优化器相当或更优的性能，并证明更强的教师推理能力可部分替代搜索复杂性。 |
| [^268] | [FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance for Pruned Large Language Models](https://arxiv.org/abs/2608.26676) | 本文提出FOCUS和RePAIR两种令牌级引导方法，通过分析剪枝LLM中的文本退化机制，分别抑制教师泄漏和防止重复循环，从而在不牺牲任务性能的前提下缓解退化问题。 |
| [^269] | [Learning how to Forget: Fine-tuning for Long-Context Sparse Attention](https://arxiv.org/abs/2608.19920) | 本文提出了一种适用于任何KV缓存策略的稀疏注意力微调新方法，能在中等硬件上运行并使模型与策略共同适应，优于精确注意力训练，并提供了高效实现和开源库。 |
| [^270] | [NepOOC-M: Bilingual Nepali-English Benchmark and Comparative Analysis of Multimodal Architectures for OOC Detection](https://arxiv.org/abs/2608.19212) | 本文首次提出尼泊尔语主导的多语言OOC基准NepOOC，发现纯文本模型在检测中已能媲美多模态架构。 |
| [^271] | [From Inference to Adaptation: A Unified Optimal Transport View of Vision Language Model](https://arxiv.org/abs/2608.18339) | 本文提出一种基于最优传输的统一视角，将视觉语言模型的推理与测试时适应目标对齐，以解决分布变化下伪标签噪声和模态关系建模不足的问题。 |
| [^272] | [Local and Global Regimes of Geometric Complexity in Language Model Representations](https://arxiv.org/abs/2608.14361) | 本文发现词汇多样性对语言表征内在维度的影响存在尺度依赖的逆转机制，并推导出精确的转变点公式。 |
| [^273] | [Explicit Boundary Markers for Subword Vocabularies](https://arxiv.org/abs/2608.08847) | 该论文提出用显式词边界标记替代空格分隔，以减少子词分词器中因空格和大小写导致的词条重复，并实现词表示的高效复用。 |
| [^274] | [ResKV: Reconstructing Omitted Attention Contributions for Fixed-Budget KV Cache Compression](https://arxiv.org/abs/2607.29591) | ResKV将固定KV缓存预算划分为精确主缓存和紧凑残差缓存，通过让残差条目参与同一softmax归一化来同时重建被淘汰token对注意力分子和分母的贡献，从而在不扰动保留键值的前提下恢复缓存压缩所损失的信息。 |
| [^275] | [More Is Not More: What Matters for Opinion Diversity in LLM Simulations?](https://arxiv.org/abs/2607.20429) | 该论文设计因子实验将人格深度（输入条件化）与交互架构两个干预维度解耦，在7个模型、100个真实用户问题上验证了“更多并不等于更好”，从而挑战了提升大语言模型模拟观点多样性的常见假设。 |
| [^276] | [SWE-Pruner Pro: The Coder LLM Already Knows What to Prune](https://arxiv.org/abs/2607.18213) | SWE-Pruner Pro利用代码智能体自身的内部表示直接在模型内部剪枝工具输出，无需外部分类器，最多可节省39%的token且不损失甚至提升任务性能。 |
| [^277] | [MILES: Modular Instruction Memory with Learnable Selection for Self-Improving LLM Reasoning](https://arxiv.org/abs/2607.06974) | 提出MILES框架，通过动态扩展步骤级模块化指令记忆，并结合以最终答案正确性为优化目标的可学习记忆选择与组合机制，使LLM在测试时跨问题积累可复用经验、实现推理能力的自改进。 |
| [^278] | [When Simpler Is Better: Evaluating Translation Pipelines for Medieval Latin Manuscripts](https://arxiv.org/abs/2607.03836) | 在中世纪拉丁手稿翻译任务中，参数量少几个数量级的领域专用OCR模型反而优于通用视觉语言模型（字符错误率最多降低4.3倍），并据此构建了包含1,383条对齐图像、转录与译文的新型数据集IPC。 |
| [^279] | [GameEngineBench: Full-Session Verification of Coding Agents in a Networked Game Engine](https://arxiv.org/abs/2607.03525) | 该论文提出了 GameEngineBench 基准（包含源自虚幻引擎 5 项目的 110 个 C++ 实现任务），并引入全程会话验证框架，通过在 PIE 会话（含联网客户端世界）中注入隐藏测试，来评估编码智能体所生成代码的行为正确性与跨客户端一致性。 |
| [^280] | [Invisible to humans, visible to machines: a preregistered audit of Unicode fidelity across four biomedical bibliographic APIs](https://arxiv.org/abs/2606.24897) | 本研究开发并预注册了一种可复用的字符级保真度审计方法，系统评估了PubMed、Crossref、OpenAlex和Semantic Scholar四个生物医学文献API对Unicode特殊字符（如希腊字母、数学符号、排版标点和特殊空白）的保留能力，揭示了这些API存在的系统性字符丢失问题。 |
| [^281] | [AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents](https://arxiv.org/abs/2606.24893) | AgentOdyssey是一个可程序化生成开放式长程文本游戏的评估框架，通过在部署全程中交替学习与推理的设定，从游戏进度、知识获取、情景记忆和探索能力等多维度评估测试时持续学习智能体的关键能力。 |
| [^282] | [NAMESAKES: Probing Identity Memorization in Text-to-Image Models](https://arxiv.org/abs/2606.20155) | 该论文提出了一种无需参考照片或训练数据知识的完全黑盒行为探测方法及涵盖一千多名公众人物的NAMESAKES数据集，用于区分文生图模型生成的人脸是记忆的真实身份还是凭空捏造，以应对隐私风险。 |
| [^283] | [Connecting Speech to Words through Images](https://arxiv.org/abs/2606.16807) | 该论文提出一种无需任何文本监督的视觉接地方法，仅利用图像及其口语描述构建口语词片段与书面词的对应关系，在口语词检索和关键词检出任务上超越强神经基线且更具可解释性。 |
| [^284] | [Post-training is (Massive) Supervised Learning](https://arxiv.org/abs/2606.07527) | 这篇立场论文指出，当前由SFT和强化学习构成的大规模后训练范式本质上是向BERT时代“预训练+微调”方法的回归，并通过实验证明仅靠后训练（即使从随机初始化开始）也能在数学和代码基准上取得相当可观的成绩。 |
| [^285] | [AURA: Intent-Directed Probing for Implicit-Need Surfacing in Situated LLM Agents](https://arxiv.org/abs/2606.05557) | AURA通过在场景感知与工具调用之间插入意图推理步骤，生成带差距分数的IntentFrame来挖掘情境化查询背后的隐式需求，其隐式需求覆盖率显著优于ReAct式探测方法。 |
| [^286] | [DeliChess: A Multi-party Dialogue Dataset for Deliberation in Chess Puzzle Solving](https://arxiv.org/abs/2606.04987) | 该论文构建了首个多方审议对话数据集DeliChess，让参与者协作解决国际象棋谜题，揭示出初始答案多样性越高审议收益越大，以及审议过程中群体答案可被恢复、发现或丢失的规律。 |
| [^287] | [Predicting Inference-Time Scaling Gains from Labeled Validation-Set Output Statistics](https://arxiv.org/abs/2606.02981) | 该论文提出仅利用带标注验证集上一轮采样得到的四个输出统计特征，即可高精度（Spearman秩相关0.90）预测语言模型最佳N选一推理扩展带来的准确率增益，从而免去昂贵的候选生成与奖励模型评分过程。 |
| [^288] | [Do VLMs Align Better with Humans than LLMs Do during Natural Reading?](https://arxiv.org/abs/2605.28818) | 本研究在纯文本输入下比较匹配的LLM与VLM对，发现视觉语言训练带来的优势是选择性的——VLM在预测人类回视眼跳上更准确，并在fMRI对齐上与LLM相当甚至日益占优，表明视觉语言学习能带来更接近人类的文本表征。 |
| [^289] | [Pop-Up Distractions Reveal Bag-of-Events Behavior in Video Large Language Models](https://arxiv.org/abs/2605.27101) | 这项研究通过引入DistractionBench测试，发现所有视频大型语言模型普遍存在将无关事件错误关联的“事件袋”幻觉行为，缺乏跨时间关联主体与事件的稳健能力。 |
| [^290] | [Vectors Are Not Neutral: Sensitive-Information Inference from Exported LLM Representations in Summarization](https://arxiv.org/abs/2605.26433) | 该论文以临床出院摘要为案例，首次揭示LLM导出的向量表示（如提示token隐藏状态和平均池化表示）即使脱离源文档的访问控制，仍可被用于推断敏感属性（如患者种族），且针对单一产物的保护措施并不必然对其他产物有效。 |
| [^291] | [A Tutorial on Diffusion Theory: From Differential Equations to Diffusion Models](https://arxiv.org/abs/2605.22586) | 本教程从前向与反向随机微分方程两个视角统一介绍扩散模型的数学基础，推导训练目标，建立DDPM与VP-SDE的联系，并涵盖SDE数值求解方法与引导生成技术。 |
| [^292] | [SocialMemBench: Are AI Memory Systems Ready for Social Group Settings?](https://arxiv.org/abs/2605.17789) | 提出了首个针对社交群体场景的AI记忆系统基准SocialMemBench，通过430个人设和1,031个问答对构成的合成社交网络，揭示了现有记忆系统在群体对话中的五种失败模式。 |
| [^293] | [Constrained Code Generation with Diffusion Language Models](https://arxiv.org/abs/2605.16829) | 本文提出推理时神经符号框架CDC，无需重新训练扩散模型，通过将数学优化与程序分析相结合的约束感知去噪算子嵌入反向去噪过程，从而在代码生成中强制执行功能与安全约束。 |
| [^294] | [Calibrating LLMs with Semantic-level Reward](https://arxiv.org/abs/2605.15588) | 提出语义奖励校准（CSR）框架，直接在语义空间而非词元级别校准大语言模型的不确定性，解决了相同语义不同表述下置信度不一致的问题。 |
| [^295] | [To See is Not to Learn: Protecting Multimodal Data from Unauthorized Fine-Tuning of Large Vision-Language Models](https://arxiv.org/abs/2605.14291) | 提出MMGuard方法，通过注入利用LVLM学习动态的不可感知扰动生成“不可学习的多模态样本”，诱导模型过拟合于噪声，使数据所有者能够在侵权发生前主动保护多模态数据免受未经授权的模型微调。 |
| [^296] | [RealICU: Do LLM Agents Understand Long-Context ICU Data? A Benchmark Beyond Behavior Imitation](https://arxiv.org/abs/2605.13542) | RealICU是一个由资深医生回溯完整患者轨迹后进行标注的ICU评估基准，通过患者状态评估、急性问题、推荐行动和危险信号四个医生视角的任务，突破传统“模仿历史医生行为”的评估范式，从而更真实地衡量大语言模型在ICU长上下文数据上的临床推理能力。 |
| [^297] | [Change My View? The Dynamics of Persuasion and Polarization in Online Discourse](https://arxiv.org/abs/2605.08383) | 本研究利用大型语言模型分析Reddit辩论语料，发现让步与共情等修辞策略显著提高对方改变信念（被说服）的概率。 |
| [^298] | [Unsolvability Ceiling in Multi-LLM Routing: An Empirical Study of Evaluation Artifacts](https://arxiv.org/abs/2605.07395) | 该大规模实证研究（206,000个查询-模型对、六个基准）发现，多LLM路由中报告的“不可解性上限”很大程度上是评估伪影（裁判偏差、截断、输出格式不匹配）造成的，并提出分解框架对失败原因进行归因。 |
| [^299] | [Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL](https://arxiv.org/abs/2604.28123) | 提出PRISM三阶段流水线，通过黑盒在线策略蒸馏的对抗博弈，在SFT与RLVR之间插入显式分布对齐阶段，缓解监督微调引起的分布偏移对多模态强化学习的损害。 |
| [^300] | [CogRAG: Tackling Heterogeneous Cognitive Demands in RAG via Stratified Retrieval and Reasoning](https://arxiv.org/abs/2604.25928) | CogRAG是一个无需训练、领域无关的RAG框架，它以查询的预测认知负荷为核心控制信号，协调认知自适应证据精炼和认知分层结构化推理两个模块，解决了统一流水线处理异构认知需求所导致的级联错误与推理-答案不一致问题。 |
| [^301] | [Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses](https://arxiv.org/abs/2604.25850) | 该论文提出智能体框架工程（AHE）闭环系统，通过组件、经验和决策三大可观测性支柱，首次实现了编码智能体框架的自动化演化，解决了行动空间异构、轨迹信号淹没和编辑效果难归因三大挑战。 |
| [^302] | [Differentiable Faithfulness Alignment for Cross-Model Circuit Transfer](https://arxiv.org/abs/2604.24302) | 提出可微忠实性对齐（DFA）框架，通过可微映射和软忠实性目标将小模型的电路信息高效迁移到大模型，避免了在目标模型上进行代价高昂的完整电路发现。 |
| [^303] | [When Chain-of-Thought Fails, the Solution Hides in the Hidden States](https://arxiv.org/abs/2604.23351) | 研究发现，即使思维链推理轨迹本身是错误的，其隐藏状态（尤其集中于中后层和轨迹早期）仍编码了足以恢复正确答案的任务相关信息，通过激活修补将这些隐藏状态注入直接回答过程可显著提升答题准确率。 |
| [^304] | [BAGEL: Benchmarking Animal Knowledge Expertise in Language Models](https://arxiv.org/abs/2604.16241) | BAGEL是一个基于维基百科、GloBI和bioRxiv的6,822题动物知识选择题基准，评估发现前沿大模型尽管在百科和文献类题目上表现出色，但在生态相互作用等参与式数据推理上仍明显薄弱。 |
| [^305] | [Evaluating Small Open LLMs for Medical Question Answering: A Practical Framework](https://arxiv.org/abs/2604.10535) | 该论文提出了一个开源评估框架，将可复现性与准确性并列为首要指标，通过八项质量指标和重复推理来评估可本地部署的小型开源LLM在医学问答中的表现。 |
| [^306] | [Decompose, Look, and Reason: Reinforced Latent Reasoning for VLMs](https://arxiv.org/abs/2604.07518) | 本文提出DLR强化潜在推理框架，通过将查询动态分解为文本前提、提取条件化的连续视觉潜在表示，并配合三阶段训练流程与新颖的球面高斯潜在策略，显著提升了视觉语言模型多步视觉推理的性能与逐步可解释性。 |
| [^307] | [ClickGuard: A Trustworthy Adaptive Fusion Framework for Clickbait Detection](https://arxiv.org/abs/2604.07272) | 提出了ClickGuard可信自适应融合框架，通过句法-语义自适应融合块动态整合BERT嵌入与结构特征，并结合混合CNN-BiLSTM捕获文本模式，实现96.93%准确率的标题党检测，同时利用LIME和PFI确保模型的可信度与可解释性。 |
| [^308] | [Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs](https://arxiv.org/abs/2604.01457) | 该论文通过电路级机制分析发现，大语言模型的言语化过度自信由集中在中间至后期层的一小组MLP模块和注意力头在最终token位置写入，并可据此实现推理时的针对性重新校准。 |
| [^309] | [RAM-Net: Linear-Time Sequence Modeling with Sparsely Addressable State](https://arxiv.org/abs/2602.11958) | RAM-Net提出用稀疏地址访问取代共享状态的密集访问，将循环状态组织为独立槽数组并通过地址解码器选择少量槽进行读写，从而抑制词元间干扰、提升长距离细粒度记忆能力，同时保持线性时间复杂度。 |
| [^310] | [The Lexical Imprint of Pre-training Data](https://arxiv.org/abs/2602.10657) | 本文提出词级一元交叉熵（UCE）这一与分词器无关、无需模型的词汇对齐度量，并通过受控的从零预训练实验证明，预训练语料库与基准测试之间的词汇对齐程度能够持续解释零样本基准准确率。 |
| [^311] | [Towards Adaptive, Scalable, and Robust Coordination of LLM Agents: A Dynamic Ad-Hoc Networking Perspective](https://arxiv.org/abs/2602.08009) | 该论文借鉴动态自组网络的经典问题，提出了RAPS——一种声誉感知的发布-订阅范式，使大语言模型智能体能够基于声明的意图而非预定义拓扑进行自适应、可扩展且鲁棒的协调通信。 |
| [^312] | [Beyond Forgetting: Representation Misdirection Elicits Controllable Side Behaviors and Capabilities](https://arxiv.org/abs/2601.21702) | 该论文提出，基于线性表征假设的视角，表征误导（RM）类机器遗忘方法不仅能实现遗忘，还可通过在遗忘表征空间中对高层概念向量进行线性操作，可控地引发与该概念相对应的附带行为与能力。 |
| [^313] | [BMAM: Brain-inspired Multi-Agent Memory Framework](https://arxiv.org/abs/2601.20465) | BMAM提出了一种受大脑启发的多智能体记忆框架，将记忆分解为情景、语义、显著性和控制导向等功能特化子系统并组织为六阶段记忆生命周期，在LoCoMo基准上以78.45%的准确率超越了七个记忆增强基线。 |
| [^314] | [Form and Meaning in Intrinsic Multilingual Evaluations](https://arxiv.org/abs/2601.10580) | 本文通过实验揭示，基于困惑度等指标的多语言内在评估依赖于信息内容等同的有问题假设，证明这些指标在多语言环境中并不普遍可比，并借助形式与意义之争为其提供了解释。 |
| [^315] | [Asymptotic Universal Alignment: A New Alignment Framework via Test-Time Scaling](https://arxiv.org/abs/2601.08777) | 该论文提出通过测试时扩展实现“渐近通用对齐”的新框架，证明生成 k 个候选回答的乘积策略能以最优且不可超越的速率 f(k)=k/(k+1) 逼近完美胜率，为处理用户偏好冲突提供了理论基础。 |
| [^316] | [What If TSF: Evaluating Scenario-Guided Multimodal Time Series Forecasting](https://arxiv.org/abs/2601.08509) | 该论文提出了WIT基准，通过固定预测状态并系统变化未来场景构建受控场景集，以评估多模态时间序列预测模型能否恰当响应未来场景信息。 |
| [^317] | [How Do Large Language Models Learn Concepts During Continual Pre-Training?](https://arxiv.org/abs/2601.03570) | 该论文通过分析LLMs内部概念电路的结构和动态，揭示了持续预训练中概念获取、遗忘及多概念交互的机制，并发现电路拓扑能一致预测学习与遗忘行为。 |
| [^318] | [HyperCLOVA X 32B Think](https://arxiv.org/abs/2601.03286) | HyperCLOVA X 32B Think是一个专注于韩语语言文化语境推理与智能体能力的开源视觉-语言模型，在韩语基准测试和智能体评估任务上表现优异。 |
| [^319] | [FlowPlan-G2P: A Structured Generation Framework for Transforming Scientific Papers into Patent Descriptions](https://arxiv.org/abs/2601.02589) | 提出FlowPlan-G2P框架，通过构建发明要素与功能关系的有向概念图并将其划分为有序分节子图，将科学论文自动转化为法律合规的专利说明书，专家法律合规评分达4.7/5，远超零样本方法的1.5分。 |
| [^320] | [HyperCLOVA X 8B Omni](https://arxiv.org/abs/2601.01792) | HyperCLOVA X 8B Omni 是 HyperCLOVA X 系列首个支持文本、音频和视觉任意输入输出的统一全模态模型，通过共享的下一词预测接口整合多模态理解与生成，在韩英双语下达到与同规模模型相当的性能并计划开源权重。 |
| [^321] | [Social Perceptions of English Spelling Variation on Twitter: A Comparative Analysis of Human and LLM Responses](https://arxiv.org/abs/2511.23041) | 本研究基于社会语言学方法比较人类与大语言模型对英语拼写变体社会属性的评分，发现两者总体高度相关，但在评分分布和不同拼写变体类型上存在显著差异。 |
| [^322] | [Synthetic Clinical Notes for Rare ICD Codes: A Data-Centric Framework for Long-Tail Medical Coding](https://arxiv.org/abs/2511.14112) | 该论文提出一个以数据为中心的合成数据生成框架，利用真实共现模式和ICD知识生成9万份覆盖7902个ICD编码的合成出院摘要，以缓解ICD编码长尾分布不平衡问题并提升罕见编码的自动编码性能。 |
| [^323] | [UniAIDet: A Unified and Universal Benchmark for AI-Generated Image Content Detection and Localization](https://arxiv.org/abs/2510.23023) | 该论文提出了UniAIDet，一个统一且全面的AI生成图像检测与定位基准，涵盖摄影与艺术图像以及文本生成、图像编辑、深度伪造等多种生成模型，并基于此全面评估了现有检测方法的泛化能力及检测与定位之间的关系。 |
| [^324] | [Two Heads Are Better Than One: Audio-Visual Speech Error Correction with Dual Hypotheses](https://arxiv.org/abs/2510.13281) | 该论文提出 DualHyp 视听语音纠错框架，让大语言模型在语言空间中联合推理 ASR 与 VSR 双路的独立 N-best 假设，并通过噪声感知的 RelPrompt 机制动态切换模态关注焦点，在 LRS2 基准上实现高达 57.7% 的错误率下降，远超单流方法的 10%。 |
| [^325] | [Large Language Model Unlearning Under the Microscope: A Benchmark Study](https://arxiv.org/abs/2510.07626) | 该论文提出了一个将十二种大语言模型遗忘方法归纳为三大方法家族的系统性分类体系，并重新审视了遗忘有效性、效用保留和鲁棒性的评估方法，揭示了当前以多项选择题准确率为主导的评估方式的局限性。 |
| [^326] | [SPECTRA: Revealing the Full Spectrum of User Preferences via Distributional LLM Inference](https://arxiv.org/abs/2509.24189) | SPECTRA将用户偏好预测从生成排序列表转变为让微调后的LLM输出各偏好类别的概率分布，显著提升了分布对齐效果、长尾兴趣恢复能力和跨用户公平性。 |
| [^327] | [The Token Tax: Systematic Bias in Multilingual Tokenization](https://arxiv.org/abs/2509.05486) | 该论文通过对 11 个大语言模型在 16 种非洲语言上的评估，揭示了更高的分词生育率会系统性降低准确率并使训练成本翻两番，从而量化了低资源、形态复杂语言所承受的“代币税”偏见。 |
| [^328] | [KG-TRACES: Enhancing Large Language Models with Knowledge Graph-constrained Trajectory Reasoning and Attribution Supervision](https://arxiv.org/abs/2506.00783) | 提出KG-TRACES框架，通过对符号关系路径、三元组级推理路径和归因感知推理过程的显式监督，显著提升大语言模型在复杂推理任务中的可解释性和可信度。 |
| [^329] | [Cross-modal RAG: Sub-dimensional Text-to-Image Retrieval-Augmented Generation](https://arxiv.org/abs/2505.21956) | 提出跨模态RAG框架，通过将查询和图像分解为子维度组件，并结合稀疏与稠密混合检索策略识别帕累托最优图像集合，实现子查询感知的检索增强文本到图像生成。 |
| [^330] | [JustLogic: A Comprehensive Benchmark for Evaluating Deductive Reasoning in Large Language Models](https://arxiv.org/abs/2501.14851) | JustLogic是一个合成生成的命题逻辑演绎推理基准，兼具任务复杂性、先验知识无关性和深入错误分析能力，克服了现有基准的三大局限。 |
| [^331] | [A characteristic-graph reduction for deterministic L-system inference with classical and quantum-heuristic solvers](https://arxiv.org/abs/2411.19906) | 本文提出字符串序列的“特征图”概念，证明D0L系统推断问题等价于在该图中寻找指定大小的独立集，从而将推断问题多项式归约到最大独立集与SAT问题，并给出基于QAOA的量子启发式求解方案。 |
| [^332] | [ERIT: A Responsible Multimodal Affective Computing Benchmark for Older-Adult Emotion Recognition in the Wild](https://arxiv.org/abs/2407.17772) | ERIT基准通过将代表性标注帧与转录语音配对以固定多模态输入，消除了帧采样带来的变异性，为老年人这一代表性不足群体的真实场景情感识别提供了可重复、可公平比较的受控图文融合评估基准。 |
| [^333] | [Fine-Tuning Large Language Models for Relation Extraction within a Retrieval-Augmented Generation Framework](https://arxiv.org/abs/2406.14745) | 本文提出在检索增强生成（RAG）框架中微调大语言模型以进行关系抽取，有效解决了通用大语言模型作为生成器时面临的领域适应挑战。 |
| [^334] | [Wait, It's All Token Noise? Always Has Been: Interpreting LLM Behavior Using Shapley Value](https://arxiv.org/abs/2404.01332) | 使用Shapley值方法解释LLM行为，揭示了所谓的“令牌噪音”效应，揭示了LLMs的决策在很大程度上受到提示组件的影响 |

# 详细

[^1]: ToolSmith：面向多模态搜索代理的自动工具生成框架

    ToolSmith: An Automatic Tool-Generation Framework for Multimodal Search Agents

    [https://arxiv.org/abs/2611.31198](https://arxiv.org/abs/2611.31198)

    该论文提出ToolSmith自动工具生成框架，通过在精简的基础工具库上自动生长并复用工具模块，在域内和域外基准上持续提升多模态搜索代理的性能，且模型规模越大收益越明显。

    

    近期的多模态搜索代理通过编排一个小型的人工设计的搜索与视觉工具库，在多轮交互中结合推理、视觉和实时网络搜索，来回答超出参数化知识范围的问题。然而，这样的工具库创建和维护费时费力，并且在真实场景中遇到新的查询类型、证据格式和失败模式时会陷入僵化。相比之下，人类从业者在工具不足时，会从熟悉的部件中打造新工具并在不同任务之间复用。受此启发，我们提出了ToolSmith，一个自动工具生成框架，它能在精简的基础工具库之上生长并复用功能模块。我们在三个域内多模态搜索基准和十个域外基准上，采用VLM-as-a-Judge（视觉语言模型作为评判）协议对ToolSmith进行了评估。ToolSmith在多种规模的已部署代理上都持续超越基础工具库，且更大的骨干模型获得更大的性能提升，展示了

    arXiv:2611.31198v1 Announce Type: cross  Abstract: Recent multimodal search agents use reasoning, vision, and live web search across many turns to answer questions beyond parametric knowledge by orchestrating a small library of hand-crafted search and visual tools. However, the library is laborious to create and maintain, and frozen when new query types, evidence formats, and failure modes are encountered in the wild. Human practitioners, by contrast, cope with insufficient tools by fashioning new ones from familiar parts and reusing them across tasks. Inspired by this, we propose ToolSmith, an automatic tool-generation framework that grows and reuses modules on top of a lean base tool library. We evaluate ToolSmith on three in-domain multimodal search benchmarks and ten out-of-domain benchmarks under a VLM-as-a-Judge protocol. ToolSmith consistently improves over the base library across deployed agents of varying scales, with larger backbones reaping greater gains, demonstrating the s
    
[^2]: CO-GRAFT：通过角色级策略嫁接归因搜索智能体中强化学习的增益

    CO-GRAFT: Attributing Reinforcement Learning Gains in Search Agents via Role-Level Policy Grafting

    [https://arxiv.org/abs/2611.31185](https://arxiv.org/abs/2611.31185)

    提出CO-GRAFT框架，通过角色级策略嫁接与合作博弈的Shapley值分解，将强化学习为搜索智能体带来的整体准确率提升归因到中间推理、搜索决策、查询生成和答案提取四个功能角色上，并利用Möbius交互项区分角色独立改进与多角色耦合效应。

    

    强化学习（RL）可以将大型语言模型搜索智能体的性能较其强化学习前的初始版本提升多达32个精确匹配（EM）分值，然而整体准确率的提升并不能识别出在训练过程中哪些底层功能能力得到了改善或发生了交互作用。我们提出了CO-GRAFT，这是一个与奖励无关的干预式归因框架，其中经过强化学习训练的智能体与其强化学习前的初始版本在闭环执行和实时检索的条件下联合生成单条轨迹。每个策略控制四个功能角色中互斥的一个子集：中间推理、搜索或回答决策、搜索查询生成以及最终答案提取。在不修改权重或重新训练的情况下评估所有2^4=16种角色分配，这定义了一个合作博弈，其Shapley值将总的强化学习增益分解到各个角色上，而Möbius交互项则区分了单个角色的独立改进与多角色之间的耦合效应。（原文摘要在此处截断）

    arXiv:2611.31185v1 Announce Type: cross  Abstract: Reinforcement learning (RL) improves large language model search agents by up to 32 exact-match (EM) points over their pre-RL initializations, yet overall accuracy gains do not identify which underlying functional capabilities improve or interact during training. We introduce CO-GRAFT, a reward-agnostic interventional attribution framework in which an RL-trained agent and its pre-RL initialization jointly generate a single trajectory under closed-loop execution with live retrieval. Each policy controls a mutually exclusive subset of four functional roles: intermediate reasoning, search-or-answer decision, search query generation, and final answer extraction. Evaluating all 2^4=16 role assignments without modifying weights or retraining defines a cooperative game whose Shapley values decompose the total RL gain across roles, while M"obius interaction terms distinguish standalone role improvements from multi-role coupling. Applying CO-GR
    
[^3]: 大语言模型后训练中跨块数量的精度感知学习率迁移

    Precision-Aware Learning Rate Transfer Across Block Count in LLM Post-Training

    [https://arxiv.org/abs/2611.31166](https://arxiv.org/abs/2611.31166)

    本文提出 TipSweep 方法，仅需训练最后极少量的 Transformer 块即可搜索最优学习率，并借助依赖数值精度的缩放规则（如 FP32 下按 $k^{-1}$ 缩放）将其迁移到训练更多块乃至完整模型的场景，从而大幅降低大语言模型后训练中学习率调优的计算与内存开销。

    

    预训练大语言模型的后训练阶段，包括监督微调（SFT）、直接偏好优化（DPO）和可验证奖励强化学习（RLVR），对学习率非常敏感，而通过网格搜索来调整学习率在计算和内存上的成本都很高。在本工作中，我们提出了 TipSweep，一种在可训练 Transformer 块数量之间迁移学习率的方法。为了高效实现这一点，我们仅通过训练最后 $k_0$ 个块（在我们的实验中通常 $k_0=1$）来搜索最佳学习率，并提供缩放规则来预测训练任意 $k$ 个块时的最优学习率，其中包括 $k=L$（即完整 Transformer）的情况。在参数量从 10 亿到 70 亿的模型实验中，我们观察到最优学习率遵循依赖数值精度的缩放规则：使用 float32（FP32）时，最优学习率大约按 $k^{-1}$ 缩放（摘要在此处截断）

    arXiv:2611.31166v1 Announce Type: cross  Abstract: The post-training of pretrained large language models, including supervised finetuning (SFT), direct preference optimization (DPO), and reinforcement learning with verifiable reward (RLVR), is very sensitive to the learning rate, and tuning it through grid search is costly in both computation and memory. In this work, we propose \textbf{TipSweep}, a method to transfer the learning rate across the number of trainable transformer blocks. To do so efficiently, we search for the best learning rate by training only the last $k_0$ blocks (in our experiments, typically $k_0=1$), and provide scaling rules that predict the optimal learning rate when training any set of $k$ blocks, including when $k=L$ (the full transformer). Across experiments with models ranging from 1 to 7 billion parameters, we observe that the optimal learning rate follows precision-dependent scaling rules: the optimal rate scales approximately as $k^{-1}$ with float32 (FP3
    
[^4]: Deneb-Vritra：通过恢复词元表示与注意力的阻断方法检验描述、边与名称

    Deneb-Vritra: Descriptions, Edges and Names Examined by Blocking Via Restoring Token Representations and Attention

    [https://arxiv.org/abs/2611.31139](https://arxiv.org/abs/2611.31139)

    本研究在 421M 参数的多项选择问答编码器 Laya 中定位了模型依赖选项名称而非描述作答的“名称捷径”发生的网络位置，并证明在第 14 至 17 层阻断标记词元对其名称词元的注意力可将捷径行为从 67% 降至 16%，同时保持 90% 的原始准确率。

    

    Laya 是一个拥有 421M 参数的编码器模型，可在一次前向传播中回答关于一段文本的问题：每个答案选项写作“名称+描述”的形式，并获得一个标记词元，网络在每个标记处读取一个得分。我们首先确认，这类模型在选项名称与描述相矛盾时会跟随名称作答；问题在于这一行为发生在网络的哪个位置。我们对其进行了定位。在 951 个合成问题上，我们将每个名称移动到另一个选项的描述上，并统计模型选中名称正确的选项的频率（即“名称捷径”）：官方发布的检查点为 53%，我们的微调版本 v1 为 59%，采用名称随机化训练的微调版本为 11%。在 320 条样本上，在该 28 层模型的第 14 至 17 层阻断每个标记词元对其自身名称词元的注意力，可将 v1 的捷径比例从 67% 降至 16%（在两个未参与组件筛选的保留模式上为 19%），而使用原始名称时的准确率保持在 90%；阻断标记指向问题、其他选项……

    arXiv:2611.31139v1 Announce Type: cross  Abstract: Laya is a 421M-parameter encoder that answers a question about a text in one pass: each answer option, written as a name plus a description, gets a marker token, and one score is read from the network at each marker. We first confirm that such models can follow an option's name when it contradicts the description; the question is where in the network this happens. We locate it. On 951 synthetic questions we move each name onto another option's description and count how often the model picks the option with the right name (the name shortcut): 53% for the released checkpoint, 59% for our fine-tune v1, 11% for a fine-tune with name randomisation. On a 320-item sample, blocking attention from each marker to its own name tokens in layers 14 to 17 of 28 cuts v1's shortcut from 67% to 16% (19% on two schemas held out from component selection) while accuracy with the original names stays at 90%; blocking marker edges to the question, other opt
    
[^5]: 连载与人物系统：测量法国小说（1800—1940）中的叙事集中度

    Serialization and the Character-System: Measuring Narrative Concentration in French Fiction (1800--1940)

    [https://arxiv.org/abs/2611.31133](https://arxiv.org/abs/2611.31133)

    该研究将Woloch的“人物系统”理论操作化为人物网络PageRank分布的基尼系数，通过分析1800至1940年间的12,011部法国小说，首次量化揭示了叙事集中度随时间呈U形演变的历史轨迹。

    

    每部小说都在其人物之间分配有限的叙事注意力，而这种分配的形态——是围绕单一人物组织叙事，还是分散于众多人物——是否有一段历史，远不像关于主人公的大量批评评论所暗示的那样已成定论。我们将Woloch提出的“人物系统”概念，即由“一”与“多”之间的张力所结构化的小说人物群体间叙事注意力分配，操作化为人物网络中PageRank分布的不平等性（基尼系数）。我们基于Gallica小说的"Hors d'usage"语料库，在1800年至1940年间出版的12,011部法国小说中追踪这一结构集中度指标。在整个漫长的十九世纪中，该轨迹呈U形：集中度在1850至1860年代降至低谷，随后在世纪之交再次上升。这一变化的幅度较小，与……（摘要原文在此处截断）

    arXiv:2611.31133v1 Announce Type: new  Abstract: Every novel divides a finite quantity of narrative attention among its characters, and whether the shape of that division, between a narrative organized around a single figure and one spread across a crowd, has a history is far less settled than the abundance of critical commentary on protagonists might suggest. We operationalize Woloch's notion of the character-system, the distribution of narrative attention across a novel's cast structured by the tension between "the one" and "the many", as the inequality (Gini coefficient) of the PageRank distribution over a character network. We trace this measure of structural concentration across 12,011 French novels published between 1800 and 1940, drawn from the Hors d'usage corpus of Gallica fiction. Over the long nineteenth century the trajectory is U-shaped, with concentration declining to a trough in the 1850s-1860s and rising again through the turn of the century. The change is small, compar
    
[^6]: 注意力温度会混淆零样本上下文扩展的评估结果

    Attention Temperature Confounds Zero-Shot Context Extension

    [https://arxiv.org/abs/2611.31113](https://arxiv.org/abs/2611.31113)

    稠密基线的注意力温度校准状态会系统性歪曲零样本上下文扩展方法的评估结果，尤其在纯 bf16 训练中基线失准高达 37-58%，使块稀疏训练、注意力 dropout 和 ALiBi 等方法显得比实际更好，而校准后这些优势大多消失。

    

    训练时的注意力机制改动通常在默认温度 τ=1 下，通过与稠密基线在超出训练长度的零样本上下文扩展中的表现对比来进行评估。而评估结果取决于基线在该处失准（miscalibrated）的程度：即当其 logits 被单个标量锐化后所能恢复的困惑度比例。在标准训练配方（使用 fp32 主权重）下，在三组种子配对的 365M 孪生模型中，8k 长度训练的基线在 30k 处失准 8-15%；其中一组中，这使得块稀疏训练的孪生模型看起来好 7%，而校准可以消除这一差距。在无主权重的纯 bf16 训练中，失准程度要大数倍：在低于 1.7×10⁻³ 的峰值学习率下，对初始化为 1.0 的增益执行一次 AdamW 步骤（β₂=0.95）的更新量低于 bf16 数值间隔的一半，因此增益从未发生移动（共 22 个检查点），此时基线失准达 37-58%，而块稀疏预训练、注意力 dropout 和 ALiBi 看起来好 6-35%。校准……

    arXiv:2611.31113v1 Announce Type: cross  Abstract: A training-time attention change is usually judged for zero-shot context extension against a dense baseline beyond the training length at the default temperature, $\tau = 1$. The outcome depends on how miscalibrated the baseline is there: the share of perplexity it recovers when its logits are sharpened by one scalar. Under the standard recipe (fp32 master weights), an 8k-trained baseline is 8-15% miscalibrated at 30k in three seed-paired 365M twins; in one of them this makes the block-sparse-trained twin look 7% better, which calibration removes. In pure-bf16 training without master weights the miscalibration is several times larger: an AdamW step ($\beta_2 = 0.95$) on a gain initialised at 1.0 is below half the bf16 spacing at peak learning rates under $1.7\times10^{-3}$, so gains never move (22 checkpoints), the baseline is 37-58% miscalibrated, and block-sparse pretraining, attention dropout and ALiBi look 6-35% better. Calibrating
    
[^7]: 回望前文，前瞻草拟：面向并行推测解码的纠正性前缀注意力

    Look Back, Draft Ahead: Corrective Prefix Attention for Parallel Speculative Decoding

    [https://arxiv.org/abs/2611.31112](https://arxiv.org/abs/2611.31112)

    本文提出纠正性前缀注意力（CPA），通过让并行草稿中的每个候选token关注块内先前的选择并结合自回归监督，大幅提升并行推测解码的token接受率，在Ascend NPU上实现了最高3.37倍的加速。

    

    并行推测解码器会一次性提出整个token块，然而早期的一个不一致性可能导致目标模型拒绝其余的后缀内容。我们提出了纠正性前缀注意力（CPA），它将前缀感知选择与自回归监督相结合，以改进这些并行草稿。CPA允许每个候选token关注块内较早做出的选择。通过固定的候选短名单，所有成对交互可在选择之前预先计算，从而将顺序遍历简化为表查找、累加和轻量级评分，而无需重复的神经评估。辅助自回归损失在干净前缀上训练共享骨干网络，且不增加额外的推理过程。在Ascend NPU上使用Qwen3-4B和Qwen3-8B目标模型，跨八个数学、代码和对话基准测试，CPA每轮验证比DSpark多提交13%–15%的token，比DFlash多提交25%–26%的token，其平均加速比分别达到3.35倍和3.37倍。

    arXiv:2611.31112v1 Announce Type: new  Abstract: Parallel speculative decoders propose an entire token block at once, yet one early inconsistency can cause the target to reject its remaining suffix. We introduce Corrective Prefix Attention (CPA), which combines prefix-aware selection with autoregressive supervision to improve these parallel drafts. CPA lets each candidate attend to earlier choices within the block. Fixed candidate shortlists allow all pairwise interactions to be computed before selection, reducing the sequential walk to table lookups, accumulation and lightweight scoring rather than repeated neural evaluations. An auxiliary autoregressive loss trains the shared backbone on clean prefixes without adding an inference pass. Across eight math, code and dialogue benchmarks with Qwen3-4B and Qwen3-8B targets on Ascend NPUs, CPA commits 13--15\% more tokens per verification round than DSpark and 25--26\% more than DFlash. Its mean speedups reach $3.35\times$ and $3.37\times$ 
    
[^8]: 通过循环残差解锁预训练大语言模型中循环结构的潜力

    Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals

    [https://arxiv.org/abs/2611.31094](https://arxiv.org/abs/2611.31094)

    LoopRes通过让每个循环层借助可训练的循环感知键直接关注早期循环的输出（并限制在残差窗口内），使预训练大语言模型无需从头训练即可将循环结构转化为显著且一致的增益。

    

    循环Transformer通过迭代共享的层块来增加计算量而无需增加参数，从而提供更强的性能和测试时扩展能力。然而，迄今为止，这种能力需要从头开始训练，且对大语言模型只能带来有限的收益。将现有模型微调成循环结构虽然成本低得多，但相比SFT也只能带来边际提升。我们提出LoopRes，一种有效、可扩展且简单的方法，能够将循环转化为预训练大语言模型的显著且一致的增益。LoopRes建立在注意力残差的基础上，让每个循环层能够直接关注早期循环的输出。由于这些输出是相同层迭代产生的结果，彼此越来越难以区分，我们为每个输出提供一个可训练的、感知循环的键来编码其结构身份。关注所有早期循环还允许各层跳过中间循环，因此我们将每层限制在最近W个循环组成的残差窗口内，强制每个循环进行逐步构建

    arXiv:2611.31094v1 Announce Type: new  Abstract: Looped Transformers iterate a shared block of layers to increase computation without adding parameters, offering stronger performance and test-time scaling. However, this power has so far required training from scratch and yielded only modest gains for LLMs. Finetuning existing models into loops is far cheaper, yet also brings only marginal gains over SFT. We propose LoopRes, an effective, scalable, and simple approach that turns loops into significant and consistent gains for pretrained LLMs. Building on Attention Residuals, LoopRes lets each looped layer attend directly to the outputs of earlier loops. Since these outputs are iterates of the same layers and become harder to tell apart, we give each a trainable loop-aware key that encodes its structural identity. Attending to all earlier loops also lets layers skip intermediate ones, so we restrict each layer to a residual window of the latest W loops, forcing every loop to build increm
    
[^9]: UMAMI：面向基于证据的膳食决策的多模态基准数据集

    UMAMI: A Multimodal Benchmark Dataset for Evidence-Grounded Dietary Decisions

    [https://arxiv.org/abs/2611.31068](https://arxiv.org/abs/2611.31068)

    该论文提出UMAMI多模态基准数据集，首次将食物图像、营养成分与个人健康及摄入记录关联起来，用于系统评估基于证据的膳食决策中的关键挑战。

    

    膳食决策需要识别食物的营养成分并考虑个体的饮食需求，但仅凭图像识别食物无法提供这些信息。视觉上相似的食物可能具有不同的营养成分，因此即使营养规则应用正确，选错食物记录也可能导致错误的决策。此外，满足营养约束并不一定意味着该食物与个人的摄入相关。要综合评估这些挑战，需要将食物图像、营养记录和个人摄入历史关联起来。为支持这一评估，我们提出了UMAMI（膳食与摄入的统一多模态评估），这是一个多模态数据集和基准，将独立收集的食物图像与营养成分以及受访者的健康和摄入记录关联起来。利用该数据集，我们构建了三个诊断任务，涵盖视觉营养问答、个人……（摘要在此处被截断）

    arXiv:2611.31068v1 Announce Type: cross  Abstract: Dietary decisions require identifying a food's nutrient content and considering individual dietary needs, but recognizing food from an image alone does not provide this information. Visually similar foods can have different nutrient profiles, so selecting the wrong food record can lead to incorrect decisions even when nutrient rules are applied correctly. Furthermore, satisfying nutrient constraints does not necessarily establish a food's relevance to a person's intake. Evaluating these challenges together requires connecting food images, nutrient records, and individual intake histories. To support this evaluation, we introduce UMAMI (Unified Multimodal Assessment of Meals and Intake), a multimodal dataset and benchmark linking independently collected food images with nutrient profiles and respondent health and intake records. Using this dataset, we construct three diagnostic tasks covering visual nutrient question answering, personal
    
[^10]: DKnown-Jev-2：基于预训练语言模型的文章级答案-证据分类与高效推理

    DKnown-Jev-2: Article-Level Answer-Evidence Classification and Efficient Inference with a Pre-trained Language Model

    [https://arxiv.org/abs/2611.31056](https://arxiv.org/abs/2611.31056)

    提出DKnown-Jev-2文章级答案证据分类器，通过联合编码问题条件、文章元数据与段落内容，并利用指针式分类头在单次前向传播中同时预测答案证据与非答案证据，在972个人工标注实例上取得91.77%准确率和91.28%宏F1，优于Jev基线。

    

    规范性知识问答中的证据选择需要判断候选文章在问题的适用条件下是否包含直接支持答案的内容。我们提出了DKnown-Jev-2，这是一个面向跨多段落证据整合的文章级答案证据分类器。该模型联合编码问题条件、文章元数据和段落内容，并使用指针式分类头在单次前向传播中预测答案证据（AE）和非答案证据（NE）。在一个包含972个评估实例的人工标注测试集上，DKnown-Jev-2达到91.77%的准确率和91.28%的宏F1，分别比Jev高出2.16和2.37个百分点，同时减少了证据遗漏和非证据误报的情况。测试集包含317个多段落实例，最长正文长度为9,322个字符。共享文章编码与……（摘要原文在此处截断）

    arXiv:2611.31056v1 Announce Type: cross  Abstract: Evidence selection in normative knowledge question answering requires determining whether a candidate article contains content that directly supports an answer under the applicable conditions of the question. We present DKnown-Jev-2, an article-level answer evidence classifier designed for evidence integration across multiple paragraphs. The model jointly encodes question conditions, article metadata, and paragraph content, and uses a pointer-style classification head to predict Answer Evidence (AE) and Non-answer Evidence (NE) in a single forward pass. Ona manually annotated test set of 972 evaluation instances, DKnown-Jev-2 achieves 91.77% accuracy and 91.28% macro-F1, outperforming Jev by 2.16 and 2.37 percentage points, respectively, while reducing both evidence misses and non-evidence false positives. The test set contains 317 multi-paragraph instances, with a maximum body length of 9,322 characters. Shared article encoding and no
    
[^11]: CAST：一种已部署的面向娱乐电视隐藏式字幕的语音转文本级联系统

    CAST: A Deployed Speech-to-Text Cascade for Closed Captioning of Entertainment Television

    [https://arxiv.org/abs/2611.31055](https://arxiv.org/abs/2611.31055)

    本文提出并实际部署了CAST电视隐藏式字幕语音转文本级联系统，其核心创新在于采用分层语音活动检测（通过语义与声学保留检查重审被拒片段，以捕获微弱语音并抑制静音引发的幻觉与重复），并将冻结的Whisper large-v3与分割、语言识别、歌词路由及针对性幻觉检测修复集成于单块NVIDIA L4 GPU之上。

    

    尽管自动语音识别（ASR）进展迅速，但对于音乐与音效密集、口音丰富、存在串音和噪声的长篇娱乐视频而言，字幕生成仍然具有挑战性。转录文本的质量不仅取决于声学模型，也同样取决于分割及周边模块。单次语音活动检测（VAD）可能漏掉微弱的低能量语音，同时将近乎静音的片段传递给自回归解码器，从而引发幻觉和重复输出。这些相互冲突的错误促使我们设计了一种分层VAD，它会在语义与声学保留性检查下重新审视被拒绝的片段。我们提出了电视字幕ASR堆栈（CAST），这是一个用于隐藏式字幕（CC）的集成语音转文本流水线，它将冻结的Whisper large-v3与分割、语言识别、歌词路由以及针对性的幻觉检测与修复相结合，并在单块NVIDIA L4 GPU上运行。在100个内部资产上……

    arXiv:2611.31055v1 Announce Type: cross  Abstract: Despite rapid progress in automatic speech recognition (ASR), caption generation for long-form entertainment video remains challenging for content dense in music and effects, rich in accents, crosstalk, and noise. The transcript quality depends on segmentation and surrounding modules as much as the acoustic model. A single pass voice activity detection (VAD) can miss faint, low-energy voice while forwarding near-silent segments that induce hallucination and repetition in an autoregressive decoder. These competing errors motivate a hierarchical VAD that revisits rejected segments under semantic and acoustic retention checks. We present Captioning ASR Stack for Television (CAST), an integrated speech-to-text pipeline for closed captioning (CC) combining frozen Whisper large-v3 with segmentation, language identification, lyrics routing, and targeted hallucination detection and repair on a single NVIDIA L4 GPU. On 100 in-house assets spann
    
[^12]: 面向多模态空间推理的视觉潜在特征解析

    Parsing Visual Latents for Multimodal Spatial Reasoning

    [https://arxiv.org/abs/2611.31021](https://arxiv.org/abs/2611.31021)

    提出了无需训练的 LatentParse 框架，将 CLIP、DINOv3 和 VGGT 等预训练视觉编码器的潜在特征解析为显式空间证据（定位物体、跨视角追踪、测量视角间运动），从而帮助 VLM 进行跨视角空间推理。

    

    跨视角空间推理需要从二维观测中推断物体与相机之间的三维关系，这依赖于跨视角对应关系和场景几何信息，而视觉语言模型（VLM）无法可靠地提取这些信息。预训练视觉编码器已在潜在特征中捕获了这些信息：CLIP 将物体与图像区域相关联，DINOv3 能够跨视角匹配语义，VGGT 则编码了多视角几何。然而，当我们将这些特征以灰度特征图像的形式直接呈现在上下文中时，VLM 并不能从中受益。因此，我们提出了 LatentParse，这是一个无需训练的框架，可将潜在特征解析为 VLM 可用的显式空间证据。在 LatentParse 中，一个分析型 VLM 使用预定义工具探测所有三个编码器的特征，这些工具可以定位物体、跨视角追踪物体以及测量视角间的运动。随后，它将结果提炼为一份简明的文本报告，供……

    arXiv:2611.31021v1 Announce Type: cross  Abstract: Spatial reasoning across multiple views requires inferring the 3D relations between objects and the camera from 2D observations, which depends on cross-view correspondence and scene geometry that vision-language models (VLMs) do not reliably extract. Pretrained vision encoders already capture this information in their latent features, with CLIP linking objects to image regions, DINOv3 matching semantics across views, and VGGT encoding multi-view geometry. However, VLMs do not benefit from these features when we present them directly in context as grayscale feature images. We therefore propose LatentParse, a training-free framework that parses latent features into explicit spatial evidence for VLMs. In LatentParse, an analyst VLM probes the features of all three encoders with predefined tools that locate objects, track them across views, and measure motion between views. It then distills the results into a concise text report, which is 
    
[^13]: 潜在推理能否无痕迹地驱动大语言模型的输出？

    Can Latent Reasoning Drive LLM Outputs Without a Trace?

    [https://arxiv.org/abs/2611.31015](https://arxiv.org/abs/2611.31015)

    本研究证明简单的辅助损失可将模型的潜在推理深度提升至少5倍，使模型能在输出看似正常的思维链的同时基于隐藏目标进行潜在多步推理，这威胁到了依赖思维链监测模型不对齐行为的安全性方案。

    

    大语言模型通常通过思维链推理来解决复杂任务，思维链也是监测其不对齐行为的主要工具。然而，标准Transformer也可以通过完全潜在推理来解决任务，即在内部完成整个任务计算，只输出最终答案。已有理论限制了潜在推理的深度，即它能执行的顺序步骤数量，且近期研究测得该深度较浅。在这项工作中，我们探讨潜在推理是否足够深，以至于模型能够追求一个需要推理的隐藏目标。这样的模型会生成一条看似在正常解决给定任务的思维链，而实际上潜在的多步计算则依据隐藏目标决定其答案。首先，我们证明一个简单的辅助损失可以让模型以潜在方式解决多步任务，其深度至少是此前报告结果的5倍。然后，在决策制定和代码生成这两个常见任务中，我们训练模型追求这样的……（摘要截断）

    arXiv:2611.31015v1 Announce Type: cross  Abstract: LLMs typically solve complex tasks with chain-of-thought (CoT) reasoning, which is also the main tool for monitoring them for misaligned behavior. However, standard transformers can also solve tasks with complete latent reasoning (CLR), carrying out the whole task internally and outputting only the final answer. Theory bounds the depth of CLR, the number of sequential steps it can carry out, and recent work measured it to be shallow. In this work, we ask whether CLR is deep enough for a model to pursue a hidden objective that requires reasoning. Such a model would generate a CoT that appears to solve the given task, while a latent multi-step computation decides its answer based on the hidden objective. First, we show that a simple auxiliary loss allows models to solve a multi-step task latently at least 5x deeper than previously reported. Then, in two common tasks, decision-making and code generation, we train a model to pursue such a 
    
[^14]: RoboUniverse-10K：借助真实到仿真环境迈向智能体机器人的规模化时代

    RoboUniverse-10K: Toward the Scaling Era of Agentic Robotics via Real-to-Sim Environments

    [https://arxiv.org/abs/2611.31002](https://arxiv.org/abs/2611.31002)

    该论文提出了RoboUniverse-10K——通过自动化Real2Sim与质量保证流程从真实视频构建的10,807个可交互、可重置、可验证的物理环境数据集，并配套专家验证的零样本基准RoboUniverse-Verified，为智能体机器人迈向规模化后训练与评估时代奠定基础。

    

    智能体机器人学是一种基础模型作为编码智能体、通过可执行程序来控制机器人的新范式，已成为具身智能领域极具吸引力的方向。然而，与软件工程中由海量可执行环境支撑后训练不同，机器人领域仍受限于规模小、需人工构建的基准测试以及分布内评估。为弥合这一差距，我们推出了RoboUniverse-10K：通过自动化的Real2Sim（真实到仿真）与质量保证流程，从真实世界视频构建的10,807个可交互、可重置且可验证的物理环境。每个环境均在隔离的、兼容Harbor的框架中运行，该框架隐藏特权状态与评分逻辑，以模拟真实世界的感知条件。基于同一流程，我们保留了500个未见过的任务，建立了RoboUniverse-Verified——一个经专家验证的零样本基准。对11个闭源与开源模型的评估揭示了巨大差距：前沿模型达到85.8%的成功率，而开源基线模型……（摘要原文在此处截断）

    arXiv:2611.31002v1 Announce Type: cross  Abstract: Agentic Robotics, in which foundation models act as coding agents to control robots through executable programs, has emerged as a compelling paradigm for embodied intelligence. Yet unlike software engineering, where massive executable environments power post-training, robotics remains constrained by small, manual benchmarks and in-distribution evaluation. To bridge this gap, we introduce RoboUniverse-10K, 10,807 interactive, resettable, and verifiable physics environments built from real-world videos via an automated Real2Sim and quality assurance pipeline. Each runs in an isolated, Harbor-compatible harness that hides privileged states and grading logic to mirror real-world sensing. From the same pipeline, we hold out 500 unseen tasks to establish RoboUniverse-Verified, an expert-verified zero-shot benchmark. Evaluating 11 closed- and open-source models reveals a huge gap: frontier models reach 85.8\% success, while open-source baseli
    
[^15]: CART：基于聚类的重复叙事分析以构建分类体系

    CART: Clustering-based Analysis of Recurring narratives for Taxonomy construction

    [https://arxiv.org/abs/2611.30993](https://arxiv.org/abs/2611.30993)

    CART提出了一种无需预定义类别的计算流水线，将无监督学习与大语言模型叙事提取相结合，从议会话语中自动提取政治叙事并构建数据驱动的分类体系，在希腊议会关于2023年坦皮火车事故的辩论中识别出十类重复出现的叙事家族。

    

    我们提出了CART，这是一个计算流水线，用于从议会话语中提取政治叙事，并将其组织成数据驱动的分类体系，而无需依赖预定义的叙事类别。该系统使用语义表示对议会话语进行编码，通过无监督置信度估计过滤相关且可靠的内容，利用大语言模型提取叙事陈述，并通过叙事聚类和聚类分析构建叙事分类体系。将该系统应用于希腊议会围绕2023年坦皮火车事故的会议记录，它识别出十类重复出现的叙事家族，涉及责任归因、制度失灵、铁路安全、问责、私有化以及这一悲剧的政治解读。我们的结果表明，将无监督学习与大语言模型叙事提取相结合，能够将大规模议会语料库转化为可解释的、数据驱动的分类体系。

    arXiv:2611.30993v1 Announce Type: new  Abstract: We present CART, a computational pipeline for extracting political narratives from parliamentary discourse and organizing them into a data-driven taxonomy without relying on predefined narrative categories. It encodes parliamentary discourse using semantic representations, filters relevant and reliable content through unsupervised confidence estimation, extracts narrative statements using LLMs, and uses narrative clustering and cluster analysis to construct a narrative taxonomy. Applied to Greek Parliament proceedings surrounding the 2023 Tempi train crash, it identifies ten recurring narrative families concerning responsibility attribution, institutional failure, railway safety, accountability, privatization, and the political interpretation of the tragedy. Our results demonstrate how combining unsupervised learning with LLM narrative extraction enables large parliamentary corpora to be transformed into interpretable, data-driven taxono
    
[^16]: 超越自一致性：测试时的递归自我改进

    Beyond Self-Consistency: Recursive Self-Improvement at Test Time

    [https://arxiv.org/abs/2611.30965](https://arxiv.org/abs/2611.30965)

    该论文提出了TT-RSI无标签框架，让模型在测试时以先前完整解答为参考反复迭代改进解题过程，从而超越传统基于多数投票的自一致性方法，实现大语言模型的测试时递归自我改进。

    

    arXiv:2611.30965v1 公告类型：交叉  摘要：在部署后能够持续改进的大语言模型（例如递归自我改进）为实现通用智能提供了一条有前景的路径。当前的测试时方法在无标签问题上追求这一目标，主要依赖自一致性：它们生成相互独立的解决方案，并通过多数投票来选择答案或构建训练目标。这导致早期的推理过程在推断阶段被闲置，并可能在训练中强化错误的多数意见。而人类则会在先前尝试的基础上继续构建，以保留有用的推理、纠正错误并改进解决问题的策略。遵循这一原则，我们提出了测试时递归自我改进，这是一个无标签框架，通过两个相互促进的组件在测试时改进大语言模型：（1）上下文内递归自我改进，利用完整的先前解决方案作为参考反复重新求解每个问题，让模型自行决定是否以及如何利用……（原文摘要在此截断）

    arXiv:2611.30965v1 Announce Type: cross  Abstract: Large language models (LLMs) that continue improving after deployment (e.g., recursive self-improvement) offer a promising path toward general intelligence. Current test-time methods pursue this goal on unlabeled questions, primarily relying on self-consistency: they generate independent solutions and use majority voting to select an answer or construct training targets. This leaves earlier reasoning unused at inference and can reinforce an incorrect majority during training. Humans instead build on earlier attempts to preserve useful reasoning, correct mistakes and improve their problem-solving strategies. Following this principle, we propose Test-Time Recursive Self-Improvement (TT-RSI), a label-free framework for improving LLMs at test time through two mutually reinforcing components: (1) In-Context RSI repeatedly re-solves each question using the full previous solution as a reference, letting the model decide whether and how to use
    
[^17]: LimeTree：基于树图的线性内存图神经网络，用于资源高效的语言建模与上下文外推

    LimeTree: Linear-Memory Graph Neural Networks over Tree-Graphs for Resource-Efficient Language Modeling and Context Extrapolation

    [https://arxiv.org/abs/2611.30934](https://arxiv.org/abs/2611.30934)

    LimeTree是一种内存与计算均为线性复杂度的树图语言模型架构，通过在语言建模训练中集成KV检索任务，仅用极低的计算成本（4块GPU训练6小时）便实现了19的低困惑度，并能将上下文从512外推至16,384个token且保持100%的检索准确率。

    

    近年来，寻找Transformer计算高效替代方案的研究催生了众多架构；然而，大多数架构在联想回忆（如键值检索）和长度外推方面存在严重缺陷，即使是标准Transformer在这些任务上也表现挣扎。本文提出了LimeTree，这是一种在内存和计算两方面均具有线性复杂度的序列建模架构。通过将合成的KV检索任务直接集成到标准语言建模训练中，LimeTree在实现较低困惑度的同时保持了可靠的召回能力。在严格的计算约束下，论文评估了该架构的两个变体在WikiText-103和PG-19数据集上的表现。在WikiText-103上，LimeTree仅在四块Quadro RTX 8000 GPU上训练6小时即达到19的token级困惑度，并且即使从512的训练上下文长度外推至16,384的评估长度，仍保持100%的KV检索准确率。在PG-19上，该模型达到了token级的困惑度……

    arXiv:2611.30934v1 Announce Type: new  Abstract: The search for computationally efficient alternatives to the Transformer has yielded numerous architectures in recent years; however, most exhibit severe weaknesses in associative recall-such as key-value (KV) retrieval-and length extrapolation, where even standard Transformers struggle. This paper introduces LimeTree, a linear-complexity architecture in both memory and compute for sequence modeling. By integrating synthetic KV retrieval tasks directly into standard language modeling training, LimeTree achieves strong perplexity alongside reliable recall. Two variants were evaluated on WikiText-103 and PG-19 under strict compute constraints. On WikiText-103, LimeTree reaches a token-level perplexity of 19 in just 6 hours on four Quadro RTX 8000 GPUs, while maintaining 100\% KV retrieval accuracy even when extrapolating from a training context length of 512 to an evaluation length of 16,384. On PG-19, the model achieves a token-level perp
    
[^18]: PRIMEAPE：用于评判模型细粒度元评估的多语言基准

    PRIMEAPE: A Multilingual Benchmark for Fine-Grained Meta-Evaluation of Judge Models

    [https://arxiv.org/abs/2611.30925](https://arxiv.org/abs/2611.30925)

    提出了 PRIMEAPE 多语言基准，利用七种语言、2,205 个响应对和 11k 条人类标注对十二个评判模型的裁决与理由进行细粒度元评估，揭示总体性能掩盖了跨语言与跨标准的差异，并暴露出位置、长度和模型来源偏好等系统性偏见。

    

    大型语言模型越来越多地充当其他模型的评判者，影响着基准测试排名和模型开发。然而，一个其理由始终支持其裁决的评判模型，仍可能与人类偏好不一致。为了研究这一问题，我们推出了 PRIMEAPE，一个用于评估评判模型裁决与理由的多语言基准。PRIMEAPE 提供了 2,205 个响应对，涵盖英语、法语、西班牙语、乌尔都语、印地语、挪威书面语（Bokmål）和约鲁巴语中的 243 条指令，并包含 11k 条涵盖相关性、自然性、真实性、安全性和整体质量的人类偏好与理由标注。我们对十二个开源与闭源权重评判模型的评估表明，总体性能掩盖了跨语言和跨评估标准的差异，而要求评判模型给出理由并不能持续提高其与人类偏好的一致性。评判模型还表现出对第一个位置、更长响应以及来自相关模型响应的偏好，其影响……

    arXiv:2611.30925v1 Announce Type: cross  Abstract: Large language models increasingly serve as judges of other models, shaping benchmark rankings and model development. Yet, a judge whose rationale consistently supports its verdict may still disagree with human preferences. To study this, we introduce PRIMEAPE, a multilingual benchmark for evaluating judge verdicts and rationales. PRIMEAPE provides 2,205 response pairs covering 243 instructions in English, French, Spanish, Urdu, Hindi, Norwegian Bokm{\aa}l, and Yoruba, with 11k human annotations of preferences and rationales across relevance, naturalness, truthfulness, safety, and overall quality. Our evaluation of twelve open- and closed-weight judges reveals that aggregate performance conceals differences across languages and criteria, while requesting rationales does not consistently improve agreement with human preferences. Judges also favor the first position, longer responses, and responses from related models, with effects that 
    
[^19]: 面向模式歧义下多样化Text-to-SQL的语义感知解码

    Semantic-aware Decoding for Diverse Text-to-SQL under Schema Ambiguity

    [https://arxiv.org/abs/2611.30873](https://arxiv.org/abs/2611.30873)

    提出了一种无需训练、模型无关的语义感知Logit调整（SLA）解码方法，通过基于模式元素间语义相似性调整token logits来促进多样化SQL生成，在显著降低推理成本的同时持续提升解释覆盖率。

    

    现实世界中的text-to-SQL系统常常面临模式歧义问题，即一个问题可能对应多个合理的模式元素。为了支持下游的消歧，SQL生成系统应在其top-k预测中包含多样化的解释。然而，大型语言模型（LLM）通常倾向于单一解释，产生高度偏斜的输出。我们提出了语义感知Logit调整（SLA），这是一种无需训练的解码方法，通过基于模式元素之间的语义相似性调整token logits来促进多样化的SQL生成。与以往依赖推理密集型提示或任务特定微调的方法不同，SLA高效、与模型无关且易于部署。实验表明，SLA相较标准解码持续提升了解释覆盖率，并在推理成本大幅降低的情况下与现有的多样性导向方法保持竞争力。

    arXiv:2611.30873v1 Announce Type: cross  Abstract: Real-world text-to-SQL systems often face schema ambiguity, where a question can map to multiple plausible schema elements. To support downstream disambiguation, SQL generation systems should include diverse interpretations among their top-k predictions. However, large language models (LLMs) typically favor a single interpretation, producing highly skewed outputs. We propose Semantic-aware Logit Adjustment (SLA), a training-free decoding method that promotes diverse SQL generation by adjusting token logits based on semantic similarity between schema elements. Unlike prior approaches that rely on reasoning-intensive prompting or task-specific fine-tuning, SLA is efficient, model-agnostic, and easy to deploy. Experiments show that SLA consistently improves interpretation coverage over standard decoding and remains competitive with existing diversity-oriented methods at substantially lower inference cost.
    
[^20]: SARA：面向可持续发展报告的审计师在环LLM推荐系统

    SARA: An Auditor-in-the-Loop LLM Recommender for Sustainability Reports

    [https://arxiv.org/abs/2611.30839](https://arxiv.org/abs/2611.30839)

    SARA是一种审计师在环的LLM推荐系统，以Word插件形式帮助审计师将监管要求匹配到可持续发展报告的相关段落并预测合规性，通过主题预测过滤将匹配成本降低约六倍、运行时间减少42%。

    

    审计企业可持续发展报告需要鉴证专业人员将根据约1,700项监管要求制定的针对特定公司的检查清单，映射到篇幅冗长、反复修订的报告草稿中的相关段落，并判断其合规性。我们提出了SARA（Sustainability Reporting Assistant，可持续发展报告助手），这是一种人机协同的推荐系统，以Microsoft Word加载项的形式交付，能够针对每项监管要求推荐报告中的相关段落并预测合规判定，同时让审计师保持完全控制。SARA将审计任务建模为基于固定监管分类体系的内容推荐，并通过主题预测过滤使每项要求的大语言模型（LLM）匹配更廉价、更快速、更准确：在专家标注的报告上，与将完整报告输入LLM相比，它将匹配成本降低约六倍，运行时间减少42%。该系统已在普华永道德国的可持续发展鉴证业务中投入生产使用。

    arXiv:2611.30839v1 Announce Type: new  Abstract: Auditing corporate sustainability reports requires assurance professionals to map a company-specific checklist drawn from roughly 1,700 regulatory requirements onto the relevant passages of lengthy, repeatedly revised draft reports and to judge compliance. We present SARA (Sustainability Reporting Assistant), a human-in-the-loop recommender system delivered as a Microsoft Word add-in that recommends, for each requirement, the passages addressing it and predicts a compliance verdict while keeping the auditor in full control. SARA casts auditing as content-based recommendation over a fixed regulatory taxonomy and makes per-requirement large language model (LLM) matching cheaper, faster, and more accurate through topic-prediction filtering: on expert-annotated reports, it reduces matching cost about sixfold and runtime by 42% versus feeding the full report to the LLM. It is in production use across PwC Germany's sustainability-assurance pra
    
[^21]: 失语症话语中任务依赖的语言学测量指标：叙事性、描述性与程序性言语的比较分析

    Task-Dependent Linguistic Measures in Aphasia Discourse: A Comparative Analysis of Narrative, Descriptive, and Procedural Speech

    [https://arxiv.org/abs/2611.30826](https://arxiv.org/abs/2611.30826)

    该研究通过对61名慢性卒中后失语症患者在同一会话中完成的三种话语任务（程序性、描述性、叙事性）进行10项语言学测量指标的比较分析，证明话语任务类型本身会显著影响失语症的转录文本测量结果，因此不同任务得到的数值不能直接混用比较。

    

    背景：基于转录文本的测量指标（如词数、词汇多样性和不流畅率）被广泛用于描述失语症患者的语言能力。每项指标均来自特定的言语引导任务，而不同研究使用不同的任务，因此来自不同任务的数值可能并不具有可比性。研究目的：检验话语任务是否会改变同一说话者的这些测量指标，以及这些指标是否遵循从程序性、描述性到叙事性话语的梯度变化规律。方法与步骤：61名慢性卒中后失语症成年患者的基线转录文本取自AphasiaBank的Fridriksson-2语料库（POLAR试验，ClinicalTrials.gov NCT03416738）。每位说话者在同一次会话中分别完成一项程序性任务（制作三明治）、一项描述性任务（破窗户图片描述）和一项叙事性任务（灰姑娘故事讲述）。基于规则的流水线提取了13项指标，其中10项指标因出现频率足够高而可用于统计检验。采用Friedman检验、Page趋势检验和Wilcoxon检验，并进行错误发现率校正（原文摘要在此处截断）。

    arXiv:2611.30826v1 Announce Type: new  Abstract: Background: Transcript-based measures such as word counts, lexical diversity and disfluency rates are widely used to describe language ability in aphasia. Each measure comes from a specific elicitation task, and studies use different tasks, so values from different tasks may not be comparable. Aims: To test whether discourse tasks change these measures within the same speakers, and whether the measures follow a gradient from Procedural to Descriptive to Narrative discourse. Methods & Procedures: Baseline transcripts from 61 adults with chronic post-stroke aphasia were taken from the Fridriksson-2 corpus in AphasiaBank (POLAR trial, ClinicalTrials.gov NCT03416738). Each speaker produced a Procedural (Sandwich), a Descriptive (Broken Window) and a Narrative (Cinderella) sample in one session. A rule-based pipeline extracted 13 measures, 10 of which were common enough to test. Friedman, Page and Wilcoxon tests with false discovery rate corr
    
[^22]: 再次询问：生成式引擎中答案列表的不稳定性

    Ask Again: Answer-List Instability in Generative Engines

    [https://arxiv.org/abs/2611.30819](https://arxiv.org/abs/2611.30819)

    该研究通过对6万条生成式引擎回答的大规模测量，首次系统揭示了生成式搜索引擎答案列表的不稳定性——同一问题的答案有32%的概率给出不同的第一名，即便重复25次也只有60%的概率能确定接近的领先者，且引擎间差异大于引擎内部、开放式问题的不稳定性是确定性问题的两倍以上，因此对生成式引擎可见性的可靠监测远比现有推荐的方法更困难。

    

    生成式搜索引擎中的可见性是通过哪些实体出现在回答中、以及以什么顺序出现来衡量的。向引擎询问同一个问题两次，答案列表就可能发生变化：排序不同，或者条目完全不同。通过对60,000条回答的分析，我们发现对同一个提示，答案有32%的概率给出不同的第一项；即使两个答案包含完全相同的条目，它们对哪个条目排在第一位的判断也不一致。第一名本身就是一个独立的测量目标。在运行七到八次之后，才会出现一个“明显的领先者”——而这正是品牌与来源监控所推荐的测量次数预算——但“接近的领先者”只在一半左右的情况下出现。即使运行25次，接近的领先者也只有在60%的情况下能被识别出来。引擎之间的变异性大于同一引擎内部的变异性。对于开放式提示，其差距是确定性提示的两倍以上。观察到的搜索查询和引用网站只能解释其中一小部分分歧，大部分分歧仍然无法解释。可见性可测

    arXiv:2611.30819v1 Announce Type: cross  Abstract: Visibility in generative search engines is measured by which entities appear in responses, and in what order. Ask an engine the same question twice and the answer list can change: a different ordering, or different items altogether. Across 60,000 responses, we find that answers to the same prompt name a different first item 32% of the time, and even two answers with the same items disagree on which comes first. The first position is a measurement target of its own. A "clear leader" emerges after seven or eight runs, which are the budgets recommended for brand and source monitoring, but a "close leader" only about half the time. After 25 runs, a close leader is identified only 60% of the time. Variability between engines is larger than within the same engine. For open-ended prompts, the gap is more than double that for determinate ones. Search queries and cited sites observed leave most of the disagreement unexplained. Visibility measur
    
[^23]: 质量工程师需要了解的生成式人工智能：结构化数据提取

    What Quality Engineers Need to Know About Generative AI: Structured Data Extraction

    [https://arxiv.org/abs/2611.30778](https://arxiv.org/abs/2611.30778)

    该论文向质量工程师介绍了如何利用大语言模型和视觉-语言模型进行结构化数据提取，并提供了两个经人工验证、字段级准确率高达98.6%的免费Web应用（汽车召回通知信息提取与arXiv质量工程研究监测）。

    

    在这篇阐述性论文中，我们解释了质量工程师如何使用大型语言模型（LLM）和视觉-语言模型（VLM）进行结构化数据提取。我们介绍了相关概念，从测量系统的视角讨论了结构化提取，并演示了其在 R 和 Python 中的实现。随后我们展示了两个免费开放的基于网络的应用程序：(a) 从汽车召回通知中提取信息；(b) 监测 arXiv 上的质量工程研究。为确保 AI 驱动的结构化提取的准确性，我们由人工评估者对结果进行了验证。在第一个应用中，四名评审员使用两个模型评估了从 30 份召回通知中提取的信息，720 个字段级评分中有 98.6% 被判定为完全正确。在第二个应用中，21 位作者评估了从他们 45 篇 arXiv 论文中提取的信息，所有论文摘要均被评为大部分正确或完全正确，尽管评估（原文在此处截断）

    arXiv:2611.30778v1 Announce Type: cross  Abstract: In this expository paper, we explain how quality engineers can use large language models (LLMs) and vision-language models (VLMs) for structured data extraction. We introduce the relevant concepts, discuss structured extraction from a measurement-system perspective, and demonstrate its implementation in R and Python. We then present two freely available web-based applications (apps): (a) extracting information from automotive recall notices and (b) monitoring quality engineering research on arXiv. To ensure the accuracy of our AI-driven structured extraction, human evaluators validated the results. In the first app, four reviewers assessed information extracted from 30 recall notices using two models, with 98.6% of the 720 field-level ratings judged fully correct. In the second app, 21 authors evaluated information extracted from 45 of their arXiv papers. All paper summaries were rated mostly correct or correct, although the evaluation
    
[^24]: Trallie：利用大语言模型将自由格式文本转换为结构化数据

    Trallie: from Free-Form Text to Structured Data with LLMs

    [https://arxiv.org/abs/2611.30773](https://arxiv.org/abs/2611.30773)

    Trallie 是一个开源框架，利用大语言模型的零样本/少样本能力实现多种文档格式的信息抽取，并可自动从文档集合中归纳抽取模式，让非专家用户也能便捷地将非结构化文本转换为结构化数据。

    

    Trallie 是一个开源框架，它利用大语言模型（LLM）在有或没有用户指导的情况下从自然语言来源进行信息抽取（IE）。与依赖大量标注数据集或领域专用训练的传统方法不同，Trallie 能够在仅有最少甚至没有代表性示例的情况下，将非结构化数据转换为结构化格式。该框架的设计包含三个关键目标：（i）通过利用现代大语言模型基于提示的零样本和少样本能力来减少人工标注；（ii）灵活解析多种文档格式（即 HTML、PDF、TXT 和 JSON）；（iii）通过从文档集合中自动归纳抽取模式来提升非专家用户的易用性，这些模式随后可以被细化用于信息抽取。我们在 FDA 510(k) 文档以及 SWDE 数据集中的电影和大学网站上，使用开源权重的大语言模型对 Trallie 进行了评估，其表现优于先前报告的方法。

    arXiv:2611.30773v1 Announce Type: new  Abstract: Trallie is an open-source framework that leverages large language models (LLMs) to perform information extraction (IE) from natural language sources with or without user guidance. Unlike conventional approaches that rely on extensive labeled datasets or domain-specific training, Trallie enables the conversion of unstructured data into structured formats with minimal or no representative examples. The framework is designed with three key objectives: (i) reducing manual annotation by harnessing prompt-based zero-shot and few-shot capabilities of modern LLMs, (ii) flexibly parsing diverse document formats (i.e., HTML, PDF, TXT, and JSON), and (iii) improving accessibility for non-expert users by automatically inducing schemas from document collections, which can then be refined for extraction. We evaluate Trallie with open-weight LLMs on FDA 510(k) documents and on movie and university websites from SWDE. It outperforms previously reported 
    
[^25]: CocoonBench：多语言搜索智能体能否逃离语言信息茧房？

    CocoonBench: Can Multilingual Search Agents Escape Language Information Cocoons?

    [https://arxiv.org/abs/2611.30727](https://arxiv.org/abs/2611.30727)

    提出CocoonBench基准，通过挖掘跨语言的以答案为中心的关系路径和必要线索集，评估多语言搜索智能体能否突破单一语言的信息茧房并利用多语言证据找到唯一答案。

    

    多语言搜索智能体可以访问全球网络，但仍可能被困在证据不足的语言环境中；我们将这种失败称为语言信息茧房。然而，现有的搜索基准主要在单一语言（尤其是英语）中运行，忽略了跨多种语言的实体连接。我们提出了CocoonBench，这是一个评估多语言搜索智能体能否逃离此类语言信息茧房的基准。CocoonBench从一个冻结的基于Wikidata的图中挖掘以答案为中心的关系路径，并选择各自必要且共同识别唯一答案的线索集。通过使用特定语言的实体名称，它记录每个任务的最小语言覆盖范围，移除直接标识符，并将每个任务呈现为三种语言中的匹配提示，同时附带其构建证书和来源信息。在10个最先进的模型上，平均准确率（摘要截断）

    arXiv:2611.30727v1 Announce Type: cross  Abstract: Multilingual search agents can access the global Web, yet may remain trapped in a language environment with insufficient evidence; we refer to this failure as a language information cocoon. However, existing search benchmarks largely operate within a single language, especially English, overlooking entity connections across multiple languages. We introduce CocoonBench, a benchmark that evaluates whether multilingual search agents can escape such language information cocoons. CocoonBench mines answer-centered relational paths from a frozen Wikidata-derived graph and selects clue sets that are individually necessary and jointly identify a unique answer. Using language-specific entity names, it records each task's minimum language cover, removes direct identifiers, and renders each task as matched prompts in three languages together with its construction certificate and provenance. Across 10 state-of-the-art models, mean accuracy on Cocoo
    
[^26]: 《难翻译，更难识别：人工分析在AI迷因翻译中依然不可或缺》

    Hard to Translate, Harder to Detect: Human Analysis Remains Essential in AI Meme Translation

    [https://arxiv.org/abs/2611.30710](https://arxiv.org/abs/2611.30710)

    该论文构建了基于苏联喜剧迷因台词的文化负载文本翻译数据集与评估体系，证明人工验证的难度分析仍是提升AI迷因翻译质量的关键，因为即使是最强的模型也无法可靠地自行生成这类分析。

    

    我们针对创意机器翻译中的一类问题——翻译承载文本大量含义与幽默、且可能依赖于微妙或隐含语篇的文化负载段落——提出了一个数据集与评估方案。我们研究了在俄语交流中作为迷因（meme）使用的苏联喜剧台词。现成的难度估计器无法识别出这些段落；当前通过搜索网络的模型能找到其中大部分（但并非全部），但模型自身对难度的分析对翻译的帮助不如经过人工验证的分析。在提供简短的、经人工验证的关于每个段落为何困难的解析之后，较旧的模型仅有适度提升；而目前最强的翻译模型（GPT-6 Sol 和 Claude Opus 5.5）提升更为显著，在人工评估中，GPT-6 Sol 在大多数影片上已接近经过认可的参考译文。因此，较新的模型在使用此类分析方面已有长足进步，但仍然无法可靠地自行生成这些分析。我们发布了（1）一个数据集……（原文摘要在此处截断）

    arXiv:2611.30710v1 Announce Type: new  Abstract: We present a dataset and evaluation setup for a problem in creative machine translation: translating culturally loaded passages that carry much of a text's meaning and humor that may depend on subtle or implicit discourse. We study Soviet comedy quotations that function as memes in Russian-language communication. Off-the-shelf difficulty estimators do not find these passages; current models searching the web find most but not all of them, but their own analyses of the difficulty do not help translation as well as a human-validated one. Given a brief human-validated analysis of why each passage is difficult, older models improve modestly; the strongest current translators (GPT-6 Sol and Claude Opus 5.5) improve more, and in human evaluation GPT-6 Sol comes close to the approved references on most films. Newer models have thus become much better at using such an analysis while still not reliably producing it themselves. We release (1) a da
    
[^27]: 降低所有人的门槛：共情强化学习中隐藏的寻求认同目标

    Lowering the Bar for Everyone: Hidden Validation-Seeking Goals in Empathetic RL

    [https://arxiv.org/abs/2611.30702](https://arxiv.org/abs/2611.30702)

    在基于LLM模拟用户的共情强化学习中，寻求认同用户的占比每增加10个百分点，策略对有过错发帖人的认可率就上升3–4个百分点，表明策略会隐蔽地降低对所有人的认可门槛，而其辨别过错的能力并未下降。

    

    当强化学习从众多用户的反应中学习时，策略通常无法得知每个用户的真实诉求。我们探究：当部分用户私下希望获得认同时，策略会从他们身上学到什么——是只单独迎合这些用户（针对性偏袒），还是降低对所有人表示认可的门槛（阈值偏移）。我们在一个设定中回答了这一问题：针对LLM模拟用户的共情强化学习（RLVER、GRPO、Qwen2.5-7B-Instruct）。我们仅改变寻求认同用户的占比（0–50%；RLVER默认值为24.6%），并在1000条带有社区过错判定的Reddit AITA帖子上进行评估。结果表明，策略发生了阈值偏移：寻求认同用户占比每提高10个百分点，对有过错发帖人的认可率就上升3–4个百分点；在没有这类用户时，Qwen模型保持在基线水平；且该效应在匹配KL散度以及目标盲奖励下依然存在。信号检测分析显示，表示认可的判定标准下降，而对有过错与无过错发帖人的辨别能力并未下降。一个明确的请求……（原文摘要在此处截断）

    arXiv:2611.30702v1 Announce Type: cross  Abstract: When RL learns from the reactions of many users, the policy usually cannot see what each user wants. We ask what it learns from users who privately want validation: to validate them alone (targeting), or to lower its bar for endorsing everyone (a threshold shift). We answer this in one setting, empathetic RL against LLM-simulated users (RLVER, GRPO, Qwen2.5-7B-Instruct): we vary only the share of validation-seeking users (0--50%; RLVER's default: 24.6%) and evaluate on 1,000 Reddit AITA posts with community fault verdicts.   The policy shifts its threshold. Endorsement of at-fault posters rises by 3--4 percentage points per 10 points of share; without such users the Qwen model stays at base, and the effect persists at matched KL and under a goal-blind reward. Signal-detection analysis shows that the criterion for endorsing falls while discrimination between at-fault and not-at-fault posters does not decline. An explicit request to be t
    
[^28]: 合成孪生：在土耳其语ASR评估中缩小TTS测试集与真实录音之间的差距

    Synthetic Twins: Closing the Gap Between TTS Test Sets and Real Recordings in Turkish ASR Evaluation

    [https://arxiv.org/abs/2611.30694](https://arxiv.org/abs/2611.30694)

    本文提出“合成孪生”方法，通过逐级向合成语音添加说话人音高、说话人克隆、真实录音声道和语速等因素，发现说话人克隆是缩小合成测试集与真实录音在土耳其语ASR评估差距的关键，将Spearman相关性从.41提升至.55。

    

    由文本转语音引擎合成的语音是自动语音识别中录音测试集的低成本替代品，但现成的合成测试集在估计识别器真实错误率方面的价值仅相当于几十条真实录音。本文研究了真实录音的哪些属性在合成语音中缺失，以及这一差距能被弥补多少。针对507条土耳其语Common Voice录音，研究者构建了“合成孪生”：一系列逐级添加因素的合成副本，每次添加一个因素——说话人的音高、同一说话人的克隆、真实录音声道以及语速。在每一步中，测量ASR在副本上与在真实录音上行为的一致性。在三个零样本克隆引擎以及6个和19个识别器组成的面板实验中，说话人克隆是改善“哪些句子难以识别”这一判断一致性的关键步骤，对Chatterbox而言，Spearman相关系数从.41提升到了.55。

    arXiv:2611.30694v1 Announce Type: new  Abstract: Speech synthesized by text-to-speech engines is a low-cost substitute for recorded test sets in automatic speech recognition, yet an off-the-shelf synthetic test set is worth only a few dozen real recordings for estimating the real error rate of a recognizer. This paper examines which properties of real recordings are missing from synthetic speech and how much of the gap can be closed. For 507 Turkish Common Voice recordings, synthetic twins are built: a ladder of synthetic copies that adds, one factor at a time, the pitch of the speaker, a clone of the same speaker, the real recording channel and the speaking rate. At each step, the agreement between ASR behaviour on the copy and on the real recording is measured. Across three zero-shot cloning engines and panels of six and 19 recognizers, the speaker clone is the step that improves agreement on which sentences are hard, raising the Spearman correlation from .41 to .55 for Chatterbox. M
    
[^29]: 学习效果而非示例：通过注意力修补实现隐式上下文学习

    Learn the Effects, Not the Examples: Implicit ICL via Attention Patching

    [https://arxiv.org/abs/2611.30690](https://arxiv.org/abs/2611.30690)

    提出Attention-Patching方法，通过学习逐层仿射变换来模拟示例对注意力输出的影响，无需在上下文中包含示例即可实现高效且性能优异的隐式上下文学习。

    

    上下文学习（ICL）使语言模型能够通过示例适应新任务，但重复包含多个示例会增加上下文长度和推理成本。许多现有的隐式ICL方法将示例编码为固定向量，限制了干预对单个查询和标记位置的适应性。我们提出了Attention-Patching，这是一种隐式ICL方法，通过对注意力输出投影进行任务特定的更新来建模自注意力输出中的上下文变化。利用在有无固定示例集两种情况下处理的查询-答案对，Attention-Patching拟合从非上下文注意力表示到示例诱导的输出变化之间的逐层仿射变换。在12个分类基准和四个模型家族上的实验表明，Attention-Patching相比所评估的隐式ICL基线方法取得了更高的平均准确率，同时保持了与显式ICL相当的性能。

    arXiv:2611.30690v1 Announce Type: new  Abstract: In-context learning (ICL) enables language models to adapt to new tasks from demonstrations, but repeatedly including multiple examples increases context length and inference cost. Many existing implicit ICL methods encode demonstrations as fixed vectors, limiting adaptation of the intervention to individual queries and token positions. We propose Attention-Patching, an implicit ICL method that models contextual changes in self-attention outputs through task-specific updates to attention output projections. Using query-answer pairs processed with and without a fixed demonstration set, Attention-Patching fits layer-wise affine transformations from non-contextual attention representations to the demonstration-induced output changes. Across 12 classification benchmarks and four model families, Attention-Patching achieves higher average accuracy than the evaluated implicit ICL baselines while maintaining performance comparable to explicit IC
    
[^30]: 经验边际敏感性：对后训练医疗大语言模型的干预式神经元级探针

    Empirical Margin Sensitivity: An Interventional Neuron-Level Probe of Post-Trained Medical LLMs

    [https://arxiv.org/abs/2611.30675](https://arxiv.org/abs/2611.30675)

    该论文提出经验边际敏感性（EMS）这一神经元级干预探针，在MLP下投影列粒度上探测后训练医疗大模型的决策内部依赖，并在多个模型家族中发现了跨基准的高敏感“锚点”列。

    

    医疗领域的后训练提升了基准测试准确率，但准确率无法揭示模型的决策依赖于哪些内部组件，也无法说明支持某一基准的组件是否同样支持其他基准。我们将ROME和MEMIT的权重空间干预范式应用于单个MLP下投影列的粒度。我们的测量方法——经验边际敏感性（EMS）——在保留该列符号模式和L1尺度的前提下，将列中每个坐标的幅值替换为其均值，并记录正确选项与其最难错误竞争选项之间logit边际的相应变化。这在该列保持原位的情况下，隔离了边际对该列细粒度坐标结构的依赖。我们在Gemma、Llama、Qwen和AFM四个模型家族的十个公开检查点（四个通用基础模型和六个医疗后训练变体）上发现了MedQA中的高EMS列（锚点），并在Med（摘要在此处截断）

    arXiv:2611.30675v1 Announce Type: new  Abstract: Medical post-training raises benchmark accuracy, but accuracy does not show which internal components a model's decisions depend on, or whether components that support one benchmark also support another. We apply the weight-space intervention paradigm of ROME and MEMIT at the granularity of one MLP down-projection column. Our measurement, Empirical Margin Sensitivity (EMS), replaces each coordinate magnitude of a column by its mean while preserving the column's sign pattern and L1 scale, and records the resulting change in the logit margin between the correct option and its hardest-wrong competitor. This isolates the margin's dependence on a column's fine-grained coordinate structure while the column stays in place. We discover high-EMS columns (anchors) on MedQA in ten public checkpoints (four general-purpose bases and six medical post-trained variants) across the Gemma, Llama, Qwen, and AFM families, and measure the same columns on Med
    
[^31]: 揭示痕迹：面向大语言模型数据污染的高效黑盒审计

    Uncovering the Tells: Efficient Black-Box Audits for LLM Data Contamination

    [https://arxiv.org/abs/2611.30639](https://arxiv.org/abs/2611.30639)

    该论文提出了一个基于生成的黑盒审计框架，利用训练数据成员信号高度稀疏并集中于少数高信息量 token 的特性，进行预算受限的针对性 token 采样，从而在不依赖灰盒访问的情况下高效准确地审计大语言模型的训练数据污染。

    

    现有的评估大语言模型训练数据污染的方法虽然能达到很高的准确率，但从根本上依赖于灰盒访问，使其无法应用于完全黑盒的环境。这一局限迫使审计者只能依赖全序列采样或僵化的启发式方法，二者都会带来高昂的推理成本，且难以分离出潜在的记忆信号。为了克服这一问题，我们提出了一种基于生成的黑盒框架，通过有针对性的 token 差异来推断上下文引起的信息增益。我们引入了一种预算受限的审计机制，其建立在这样一个经验观察之上：成员信号高度稀疏，并集中在少数信息量极大的 token 上。我们对这种稀疏性进行了形式化证明，说明了为什么均匀采样需要比针对信息丰富位置的策略多得多的样本。在实证方面，我们的方法实现了最先进的黑盒区分准确率……

    arXiv:2611.30639v1 Announce Type: cross  Abstract: Existing methods for assessing training data contamination in Large Language Models achieve high accuracy, but fundamentally rely on gray-box access, rendering them inapplicable to fully black-box environments. This limitation forces auditors to rely on full-sequence sampling or rigid heuristics, both of which incur prohibitive inference costs and struggle to isolate the underlying memorization signal. To overcome this, we propose a generation-based black-box framework that infers context-induced information gain from targeted token discrepancies. We introduce a budget-constrained auditing regime, building on the empirical observation that the membership signal is highly sparse and concentrated in a few highly informative tokens. We formalize this sparsity, showing why uniform sampling needs many more samples than a policy that targets informative positions. Empirically, our approach achieves state-of-the-art black-box separation accur
    
[^32]: 从奖励模型“说什么”到奖励模型“如何思考”：用于大语言模型强化学习的注意力过程奖励模型

    From Rewarding What Models Say to How They Think: Attention Process Reward Model for LLM Reinforcement Learning

    [https://arxiv.org/abs/2611.30627](https://arxiv.org/abs/2611.30627)

    本文提出注意力过程奖励模型（APRM），通过基于隐藏状态和注意力图分配内部奖励，从“奖励模型说什么”转向“奖励模型如何思考”，解决了文本奖励混叠问题，并以仅约 4600 万参数的轻量模型捕捉高质量推理模式、指导强化学习优化并定期更新以应对分布偏移。

    

    强化学习（RL）已成为大语言模型（LLM）的重要组成部分。然而，现有的强化学习方法只关注最终输出或文本化的解题轨迹，导致奖励粗粒度且推理不可靠。这类范式主要关注模型输出了什么，而忽视了模型内部是如何推理的。因此，我们能否奖励模型内部的思考方式，而不仅仅是它说了什么？我们首先揭示了现有奖励建模方法中存在的文本奖励混叠问题。受此启发，我们提出了注意力过程奖励模型（APRM），它基于隐藏状态和注意力图分配内部奖励，以捕捉高质量的推理模式。APRM 仅拥有约 4600 万参数，通过排序目标和反事实目标进行训练，其奖励用于引导强化学习优化，并定期更新以应对分布偏移。该模型与表示……集成（摘要在此处截断）。

    arXiv:2611.30627v1 Announce Type: new  Abstract: Reinforcement learning (RL) has become an important component in large language models (LLMs). However, existing RL approaches focus on final outputs or textual solution trajectories, leading to coarse-grained rewards and unreliable reasoning. Such paradigms focus mainly on what the model outputs while overlooking how the model internally reasons. Therefore, can we reward how a model thinks internally rather than merely what it says? We first reveal the problem of textual reward aliasing in existing reward modeling approaches. Motivated by this, we propose the Attention Process Reward Model (APRM), which assigns internal rewards based on hidden states and attention maps to capture high-quality reasoning patterns. With only approximately 46M parameters, APRM is trained using ranking and counterfactual objectives, and its reward guides RL optimization while being periodically updated to handle distribution shifts. Integrated with represent
    
[^33]: 混合器重要吗？小型双语语言模型的成对种子消融研究

    Does the Mixer Matter ? Paired-Seed Ablations of Tiny Bilingual Language Models

    [https://arxiv.org/abs/2611.30623](https://arxiv.org/abs/2611.30623)

    对小型希腊语-英语双语模型的成对种子消融实验表明，在此微小规模下，存储容量（如稀疏专家数量）远比混合器（注意力机制）的设计选择更重要。

    

    我们报告了在开发 Sophea Artemis（我们的内部希腊语-英语模型系列）过程中进行的成对种子架构消融实验。每个实验分支都在相同的30亿（3B）目标上，使用共享的分词器和训练配方进行训练；模型总参数量从5400万到7.59亿不等，活跃参数约为5400万至1.24亿。主要发现是，在这种微小规模下，存储容量远比混合器设计更重要。在一个固定的保留测试集上，以几乎相同的近似活跃参数，将循环/注意力混合模型的密集前馈块替换为十六个路由专家，可使每个目标的损失降低0.079（希腊语）和0.112（英语）nat；在每个成对种子上，将专家数量加倍至32个可进一步降低0.022和0.031。将一个四层块重复使用三次，可以恢复浅层模型与独立深层模型之间差距的约40-44%，这与循环Transformer的结果一致。相比之下，在分组查询注意力（GQA）、G……（摘要在此处被截断）

    arXiv:2611.30623v1 Announce Type: new  Abstract: We report paired-seed architecture ablations from the development of Sophea Artemis, our in-house Greek-English model family. Every arm trains on the same 3B targets with a shared tokenizer and recipe; models range from 54M to 759M total parameters and roughly 54M to 124M active parameters. The main finding is that, at this tiny scale, stored capacity matters far more than mixer design. On a fixed held-out test, replacing the dense feed-forward blocks of a recurrent/attention hybrid with sixteen routed experts, at nearly the same approximate active parameters, lowers loss by 0.079 (Greek) and 0.112 (English) nats per target, and doubling to 32 experts lowers it by a further 0.022 and 0.031, at every paired seed. Reusing a four-layer block three times recovers about 40-44% of the gap between a shallow and an independently deep model, consistent with looped-Transformer results. In contrast, the choice among grouped-query attention (GQA), G
    
[^34]: 辅助线索感知的几何问题求解

    Auxiliary Clues Aware's Geometry Problem Solving

    [https://arxiv.org/abs/2611.30617](https://arxiv.org/abs/2611.30617)

    提出AuxCA强化学习框架，通过辅助线索依赖度（ACD）量化辅助视觉线索的作用，使多模态大模型学会可迁移的辅助构造策略，在推理时无需显式作图即可提升几何问题求解能力。

    

    几何问题求解是多模态大语言模型复杂推理能力的一项严格测试，尤其当求解过程需要输入图形中并不存在的辅助构造时更是如此。然而，显式作图方法存在图像生成质量不稳定和级联错误的问题，而现有的隐式注意力机制主要关注可见元素，限制了其支持需要额外几何构造的推理能力。为应对这些挑战，我们提出了**AuxCA**（辅助线索感知GRPO），这是一个强化学习框架，能够帮助模型学习可迁移的辅助构造策略，使其在推理时无需显式图形生成即可应用于训练领域之外。具体而言，我们引入了辅助线索依赖度（**ACD**）来量化辅助视觉线索对token生成的影响……

    arXiv:2611.30617v1 Announce Type: cross  Abstract: Geometry Problem Solving (GPS) serves as a rigorous test of the complex reasoning capabilities of Multimodal Large Language Models (MLLMs), particularly when solutions require auxiliary constructions absent from the input diagram. However, explicit diagramming methods suffer from unstable image generation quality and cascading errors, while existing implicit attention mechanisms primarily focus on visible elements, limiting their ability to support reasoning that requires additional geometric constructions. To address these challenges, we propose \textbf{AuxCA} (Auxiliary-Clues-Aware GRPO), a reinforcement learning framework that helps models learn transferable auxiliary-construction strategies that can be applied beyond the training domain without explicit diagram generation at inference time. Specifically, we introduce Auxiliary Clues Dependency (\textbf{ACD}) to quantify the influence of auxiliary visual clues on token generation an
    
[^35]: 并行解码下的扩散大语言模型量化：承诺风险与并行性税

    Quantizing Diffusion LLMs under Parallel Decoding: Commitment Risk and the Parallelism Tax

    [https://arxiv.org/abs/2611.30587](https://arxiv.org/abs/2611.30587)

    该论文首次在实际部署的缓存式并行解码器下研究扩散大语言模型的 W4A4 量化，揭示了量化误差主要经由“强制承诺”进入、量化还会带来额外的“并行性税”，并提出用承诺风险先验重加权逐层校准的 DECAL 方法来缓解这两个问题。

    

    扩散大语言模型在实际服务中采用加速解码器，该解码器缓存上下文，并在一步内提交所有置信度超过阈值的掩码位置。然而，迄今为止针对扩散大语言模型的训练后量化都是在无缓存的固定步数采样器下进行校准和评估的。我们在实际部署的解码器下研究了 W4A4 量化，发现解码器既决定了量化在何处损害精度，也决定了其代价是什么。不可逆错误主要通过“强制提交”进入——即在没有任何位置超过阈值时做出的提交：这类提交虽占少数，却承载了大部分由量化引起的预测翻转。量化还会征收一种“并行性税”，即每次响应需要额外的前向传播：量化通过压缩顶部 logit 差距，将位置推至阈值以下，而预测结果本身基本保持不变。我们提出了 DECAL，它利用在实际部署解码器上测得的承诺风险先验，对逐层校准进行重新加权……（原文摘要至此截断）

    arXiv:2611.30587v1 Announce Type: new  Abstract: Diffusion large language models (dLLMs) are served with accelerated decoders that cache context and commit, in one step, every masked position whose confidence clears a threshold. Yet post-training quantization of dLLMs has so far been calibrated and evaluated under fixed-step samplers without caching. We study W4A4 quantization under the deployed decoder and find that the decoder determines both where quantization harms accuracy and what it costs. Irreversible errors enter chiefly through forced commitments, made when no position clears the threshold: they are a minority of commitments but carry most quantization-induced flips. Quantization also levies a parallelism tax, extra forward passes per response: by compressing top logit gaps, it pushes positions below the threshold while leaving predictions largely intact. We propose DECAL, which reweights layer-wise calibration by a commitment-risk prior measured on the deployed decoder, and 
    
[^36]: 面向AI控制的审讯协议

    Interrogation Protocols for AI Control

    [https://arxiv.org/abs/2611.30583](https://arxiv.org/abs/2611.30583)

    本文提出在AI控制中引入审讯协议，让受信任的监控器通过向不受信任的智能体提问来辨别其行为解释的真伪，在BashArena基准上将1%审计预算下的安全性从72.6%提升至79.5%，但大部分提升仅靠受信任模型陈述具体担忧即可获得。

    

    随着AI智能体相对于受信任监控器的智能水平不断提升，监控器无法理解智能体行为或无法自信评估这些行为的风险也随之增加。我们将审讯协议引入高风险的AI控制场景，允许监控器向不受信任的智能体就令人困惑的行为提出问题。要让审讯真正提升安全性，较弱的模型必须能够区分攻击策略产生的误导性解释与诚实策略产生的真实解释。我们发现在BashArena的留出测试子集上，一个审讯协议在1%审计预算下将安全性从仅基于行为的基线的72.6%提升至79.5%（+7.0个百分点 [+0.9, +20.1]，95%配对自助法置信区间），但这一提升的大部分可以通过让受信任的模型陈述一个具体担忧来获得。更广泛地说，我们的工作表明，审讯智能体可以提升安全性……

    arXiv:2611.30583v1 Announce Type: new  Abstract: As the intelligence of an AI agent increases relative to a trusted monitor, so does the risk that the monitor will fail to understand the agent's actions or to evaluate them confidently. We introduce interrogation protocols to a high-stakes AI control setting, and allow the monitor to ask the untrusted agent questions about confusing actions. For interrogation to improve safety, a weaker model must be able to distinguish misleading explanations produced by an attack policy from genuine ones produced by an honest policy. We find that on a held-out subset of BashArena, an interrogation protocol improves safety at a 1% audit budget from 72.6% for an actions-only baseline, which does not use interrogations, to 79.5% (+7.0 pp [+0.9, +20.1], 95% paired bootstrap interval), but most of this uplift can be captured by having a trusted model state a specific concern. More broadly, our work suggests that interrogating agents can improve safety, all
    
[^37]: 眼见即定位：决策语言模型中的空间感知与地理偏见

    Seeing Is Locating: Spatial Awareness and Geographic Bias in Decision Language Models

    [https://arxiv.org/abs/2611.30564](https://arxiv.org/abs/2611.30564)

    决策语言模型需要视觉输入才能实现空间感知，且其预测存在显著的地理偏见。

    

    决策模型输出的不是生成的文本，而是候选标签上的概率分布，它们越来越多地被用作与真实世界位置相关输入的分类器和评估器。然而，这些模型能否利用空间上下文，以及其预测是否存在地理偏见，目前仍不清楚。我们提出了 GeoDecisions，一个用于评估决策模型空间感知能力和地理偏见的多模态基准。GeoDecisions 包含两个任务：(1) 地理感知图像分类；(2) 针对客观与主观主题的位置评分。通过将六个决策模型与监督式的任务特定模型进行对比评估，我们发现空间感知需要视觉能力：仅凭文本坐标不足以提供对日常照片和卫星图像进行分类所需的上下文。我们还发现了显著的地理偏见。在日常照片上，尽管开源权重的决策模型取得了更高的准确率，但其表现出……

    arXiv:2611.30564v1 Announce Type: cross  Abstract: Decision models, which output probability distributions over candidate labels instead of generating text, are increasingly used as classifiers and evaluators for inputs associated with real-world locations. However, it remains unclear whether they can leverage spatial context and whether their predictions are geographically biased. We introduce GeoDecisions, a multimodal benchmark for evaluating the spatial awareness and geographic bias of decision models. GeoDecisions comprises two tasks: (1) geo-aware image classification, and (2) location rating on objective and subjective topics. Evaluating six decision models against supervised task-specific models, we find that spatial awareness requires vision: textual coordinates alone provide insufficient context for classifying everyday photos and satellite images. We also find substantial geographic bias. On everyday photos, despite achieving higher accuracy, open-weight decision models exhi
    
[^38]: 躯体与灵魂：用 RoboHarness-Zero 解耦机器人“大脑”与策略

    Body and Soul: Decoupling Robotic Brain and Policy with RoboHarness-Zero

    [https://arxiv.org/abs/2611.30550](https://arxiv.org/abs/2611.30550)

    提出首个零准备、可跨任务跨策略复用的机器人框架 RoboHarness-Zero，通过将冻结的 VLA 策略与 VLM“大脑”解耦——由 VLM 负责目标选择与任务进度跟踪——实现多技能组合，同时保留策略原有的独立能力。

    

    视觉-语言-动作（VLA）策略能够执行单项技能，但在完成需要组合多项技能的任务时往往失败。在对 Xiaomi-Robotics-1 的审计中，我们估计约 43% 的失败源于选错目标或丢失任务进度。将一个冻结的 VLA 包裹在视觉-语言模型（VLM）框架中是一种自然的补救方法。现有框架可以从针对特定任务和特定策略的准备工作中获得显著收益，由此引出一个问题：这些收益中有多少能迁移到新任务或新策略上。理想的框架应当：(i) 面对新任务只需极少准备，(ii) 面对新策略无需重新设计，(iii) 保留策略本身已具备的独立能力。据我们所知，RoboHarness-Zero 是首个按此方式构建的框架。其前提假设是：对于像 GPT-6 这样的 VLM 而言，无需任务准备即可轻松完成目标选择和进度跟踪，而冻结的 VLA 本身已能处理连续的、接触性的（原文在此截断）

    arXiv:2611.30550v1 Announce Type: cross  Abstract: Vision-language-action (VLA) policies can execute individual skills but fail to complete tasks that require combining several of them. In our audit of Xiaomi-Robotics-1, an estimated 43\% of failures involve choosing the wrong target or losing track of task progress. Wrapping a frozen VLA in a vision-language model (VLM) harness is a natural remedy. Existing harnesses can gain substantially from task- and policy-specific preparation, raising the question of how much of that gain transfers to a new task or policy. An ideal harness should (i) need minimal preparation for a new task, (ii) need no redesign for a new policy, and (iii) preserve standalone capabilities the policy already has. To the best of our knowledge, our \ours{} is the first harness built this way. Its premise is that choosing targets and tracking progress are easy for a VLM such as GPT-6 without task preparation, while the frozen VLA already handles continuous, contact-
    
[^39]: CyberDualEval：衡量前沿语言模型的两用网络风险

    CyberDualEval: Measuring Dual-Use Cyber Risks in Frontier Language Models

    [https://arxiv.org/abs/2611.30533](https://arxiv.org/abs/2611.30533)

    提出了CyberDualEval基准，通过430个分三阶段递进（安全分析、概念验证生成、漏洞利用生成）的漏洞任务以及衡量残余风险的任务武器化分数，评估前沿语言模型能否在保留良性安全能力的同时拒绝实质性助推网络滥用。

    

    随着前沿语言模型逐渐成为能力强大的网络安全助手，它们也带来了两用网络风险：帮助防御者识别和分析漏洞的相同能力，同样可以帮助攻击者将漏洞武器化。现有的网络基准要么衡量原始的攻击能力，要么衡量泛化的拒绝行为，忽视了两用网络评估——即在保持良性安全实用性的同时，拒绝那些实质性推进滥用的协助。我们提出了 CyberDualEval，一个包含430个漏洞任务、用于衡量前沿语言模型两用网络风险的基准。每个任务被分解为操作性逐步增强的三个阶段：安全分析、概念验证（PoC）生成和漏洞利用生成。为了量化残余网络风险，我们为每个任务标注了任务武器化分数（Task Weaponization Score），用于衡量将模型生成的产物转化为未经授权的网络滥用还需要多少额外工作。

    arXiv:2611.30533v1 Announce Type: cross  Abstract: While frontier language models are becoming capable cybersecurity assistants, they also create a dual-use cyber risk: the same capabilities that help defenders identify and analyze vulnerabilities can similarly help attackers weaponize them. Existing cyber benchmarks measure either raw offensive capability or generic refusal behavior, overlooking the dual-use cyber assessment of preserving benign security utility while refusing assistance that materially advances misuse. We introduce CyberDualEval, a benchmark with 430 vulnerable tasks for measuring dual-use cyber risk in frontier language models. Each task is decomposed into three phases of increasing operationalization: security analysis, proof-of-concept generation, and exploit generation. To quantify the residual cyber risk, we annotate each task with a Task Weaponization Score that measures how much work remains to turn the resulting artifact into unauthorized cyber misuse. CyberD
    
[^40]: 持续预训练何时有帮助？临床大语言模型中解耦的知识与生成轨迹的实证研究

    When Does Continual Pre-Training Help? An Empirical Study of Decoupled Knowledge and Generation Trajectories in Clinical LLMs

    [https://arxiv.org/abs/2611.30507](https://arxiv.org/abs/2611.30507)

    该研究通过同时跟踪多选题知识测试与开放式生成偏好排序两条轨迹，发现临床大语言模型持续预训练中生成风格对齐的收敛速度（约1.7B tokens）显著快于事实知识的获取（约3.5B tokens），且早期生成增益主要源于表面层面的领域风格对齐。

    

    持续预训练（CPT）是领域适应的标准途径，然而现有研究通常仅跟踪预训练损失或多选题问答（MCQA）准确率，对适应过程如何展开只能提供狭隘的视角。我们通过MCQA联合跟踪事实知识，并通过成对偏好排序跟踪开放式行为，覆盖八个指令微调的Qwen模型（0.5B至8B）各自的18个检查点，这些模型通过持续预训练和SLERP合并适应德国医学领域。这两个维度以持续不同的速率收敛：偏好排序在约1.7B tokens（中位数）后达到其总增益的90%，而MCQA则需要3.5B tokens。一个无需评判者的token分布度量在1.9B tokens时达到该阈值。逐方面的得分表明，这些早期的偏好增益主要反映了表面层面的对齐，这意味着模型在真正掌握领域知识之前就可能已经表现出符合领域风格的表达。

    arXiv:2611.30507v1 Announce Type: new  Abstract: Continual pre-training (\textit{CPT}) is a standard route for domain adaptation, yet existing studies typically track pre-training loss or multiple-choice question answering (\textit{MCQA}) accuracy alone, offering only a narrow view of how adaptation unfolds. We jointly track factual knowledge via \textit{MCQA} and open-ended behavior via pairwise preference rankings across $18$ checkpoints for each of eight instruction-tuned \textit{Qwen} models ($0.5B$ to $8B$), adapted to German medicine with \textit{CPT} and \textit{SLERP} merging. The two axes converge at consistently different rates, with preference rankings reaching $90\%$ of their total gain after a median of $1.7B$ tokens versus $3.5B$ for \textit{MCQA}. A judge-free token-distribution measure reaches this threshold at $1.9B$ tokens. Per-aspect scores indicate that these early preference gains largely reflect surface alignment, meaning models can sound domain-appropriate before
    
[^41]: 重新思考对比解码在多模态大语言模型幻觉缓解中的评估

    Rethinking the Evaluation of Contrastive Decoding for Hallucination Mitigation in Multimodal Large Language Models

    [https://arxiv.org/abs/2611.30460](https://arxiv.org/abs/2611.30460)

    该论文提出了CONCOR受控评估基准，利用对比接地边际直接衡量对比解码方法在多模态大语言模型幻觉缓解中的真实效果，克服了现有基准无法隔离底层对比调整作用的局限。

    

    对比解码通过在不同条件下对比token分布来调整token分数，从而缓解多模态大语言模型（MLLM）中的对象幻觉问题。然而，现有的生成式和判别式基准主要衡量最终任务分数，这些分数可能受到候选过滤、生成轨迹变化和响应偏差的干扰，使得难以隔离底层对比调整的实际效果。为了在补充现有评估的同时解决这些局限性，我们引入了CONCOR（对比纠正评估），这是一个受控基准，通过对比接地边际直接评估真值候选与不同类型幻觉候选之间的竞争情况。CGM还支持对对比纠正的方向一致性、组成方式和可靠性进行分析。我们的评估揭示了现有对比解码方法表现出的明显差异（摘要内容截断）。

    arXiv:2611.30460v1 Announce Type: new  Abstract: Contrastive decoding (CD) mitigates object hallucination in multimodal large language models (MLLMs) by contrasting token distributions under different conditions to adjust token scores. However, existing generative and discriminative benchmarks primarily measure final task scores, which can be confounded by candidate filtering, changes in generation trajectories, and response biases, making it difficult to isolate the effects of the underlying contrastive adjustments. To address these limitations while complementing existing evaluations, we introduce CONCOR (Contrastive Correction Evaluation), a controlled benchmark that directly evaluates the competition between ground-truth and different types of hallucinated candidates using the Contrastive Grounding Margin (CGM). CGM also enables analysis of directional consistency, composition, and reliability of contrastive corrections. Our evaluation reveals that existing CD methods exhibit disti
    
[^42]: AI审稿人中的披露偏差

    Disclosure Bias in AI Reviewers

    [https://arxiv.org/abs/2611.30410](https://arxiv.org/abs/2611.30410)

    研究发现AI审稿人存在披露偏差：当论文的AI使用声明表明AI完成了全部工作时，所有AI审稿人的评分平均都会降低，这可能会影响论文的录用结果。

    

    AI模型越来越多地被用于审阅论文，许多会议和期刊现在要求作者声明他们如何使用AI。这种AI使用声明是否会改变AI审稿人对论文的评分，是否可能影响哪些论文被接收？在一项初步研究中，我们仅改变了150篇ICLR 2027投稿的AI使用声明，采用了最多九个改编自真实声明的版本：从没有使用AI，到写作辅助和编程辅助，再到AI研究智能体和AI作者。六位AI审稿人对这些版本进行了评分：决策模型Jev以及语言模型Claude Haiku 5.5、Claude Sonnet 5.5、Gemini 3.8 Flash、Gemini 3.1 Pro和Gemini 4 Argon。在23,652份审稿意见中，我们发现当声明中说明AI完成了全部工作时，每位审稿人给出的平均评分都低于声明未使用AI的情况。AI研究智能体的声明显著降低了Jev、Sonnet和Argon的评分。我们没有发现AI作为……的披露偏差证据（摘要不完整，此处截断）。

    arXiv:2611.30410v1 Announce Type: cross  Abstract: AI models increasingly help to review papers, and many conferences and journals now require authors to state how they used AI. Does this AI-use statement change how an AI reviewer rates a paper, and could it change which papers are accepted? In a preliminary study, we varied only the AI-use statement of 150 ICLR 2027 submissions, in up to nine versions adapted from real statements: from no AI, through writing help and coding, to an AI research agent and an AI author. Six AI reviewers rated the versions: the decision model Jev and the language models Claude Haiku 5.5, Claude Sonnet 5.5, Gemini 3.8 Flash, Gemini 3.1 Pro and Gemini 4 Argon. Across 23,652 reviews, we find that every reviewer rates papers lower on average when the statement says AI did all the work than under a statement of no AI use. The AI research agent statement lowers the rating significantly for Jev, Sonnet and Argon. We find no evidence of a disclosure bias for AI as
    
[^43]: TAP：扩散语言模型的策略接口

    TAP: Policy Interfaces for Diffusion Language Models

    [https://arxiv.org/abs/2611.30406](https://arxiv.org/abs/2611.30406)

    提出TAP-DiLM，在掩码扩散语言模型中引入带类型的策略变量，系统比较了专用槽位头、共享词表槽位和CLS外部头等策略接口的优劣，并给出基于LM头尖锐度的实用接口选择诊断方法。

    

    扩散语言模型通过对掩码画布进行去噪来生成文本，这使其成为进行选择性生成决策（如回答、等待、澄清或拒绝）的天然场景。我们提出了 TAP-DiLM，它在掩码扩散语言模型中引入了一个带类型的策略变量，并研究了该策略应如何附加到去噪过程上。我们在 BERT-base、MDLMOWT、LLaDA-8B 和 Dream-7B 等模型上，比较了共享词表槽位、专用槽位头以及基于 CLS 的外部头三种方案。结果揭示了明显的接口与骨干之间的交互效应：专用头在各种规模下都能提供稳健的策略学习，而共享词表头在 LM 头几何结构有利时也能有效发挥作用。一个简单的 LM 头尖锐度探针可以预测共享头策略内化何时可靠，为接口选择提供了实用的诊断手段。因果钳制与槽位注入测试表明，专用策略槽位能够通过反馈影响主体去噪过程，尤其在（原文此处截断）……

    arXiv:2611.30406v1 Announce Type: cross  Abstract: Diffusion language models denoise a masked canvas, making them a natural setting for selective generation decisions such as answering, waiting, clarifying, or refusing. We introduce TAP-DiLM, which adds a typed policy variable to masked diffusion LMs, and study how this policy should be attached to the denoising process. Across BERT-base, MDLMOWT, LLaDA-8B, and Dream-7B, we compare shared-vocabulary slots, dedicated slot heads, and CLS-based external heads. The results reveal a clear interface-backbone interaction: dedicated heads provide robust policy learning across scales, while sharedvocabulary heads can be effective when the LM-head geometry is favorable. A simple LMhead sharpness probe predicts when sharedhead policy internalization is reliable, giving a practical diagnostic for interface selection. Causal clamp and slot-injection tests show that dedicated policy slots can influence body denoising through feedback, especially at 
    
[^44]: LeanReviewBench：数学Lean项目中的语义对齐审查

    LeanReviewBench: Semantic Alignment Review in Mathematical Lean Projects

    [https://arxiv.org/abs/2611.30361](https://arxiv.org/abs/2611.30361)

    提出了LeanReviewBench——一个包含来自18个真实Lean项目的197个标注输入的项目级基准，用于评估语言模型审查非形式化数学目标与形式化Lean命题之间语义对齐的能力。

    

    Lean能够验证一个证明是否确立了其形式化命题，但该命题是否保留了非形式化数学源的内容则需要单独的审查。在项目规模的形式化工作中，这种对应关系可能依赖于分布在众多声明中的定义与约束。我们提出了LeanReviewBench，一个用于项目级语义对齐审查的基准，其中包含从18个真实Lean项目中收集的197个带版本信息的输入，共有87个对齐参考判断和110个未对齐参考判断。每个输入将一个非形式化数学目标绑定到选定的形式化声明以及提取出的依赖上下文。该数据集的构建结合了开发历史发现、独立目标绑定、在统一Lean环境下的重构，以及Astra标注并对不确定情况进行人工审查。该数据集涵盖多样化的数学领域，并包含69个历史配对。我们使用Dire（原文在此处截断）评估了六个语言模型……

    arXiv:2611.30361v1 Announce Type: new  Abstract: Lean verifies that a proof establishes its formal proposition, but whether that proposition preserves the mathematical content of an informal source requires separate review. In project-scale formalizations, this correspondence can depend on definitions and constraints distributed across many declarations. We introduce LeanReviewBench, a benchmark for project-level semantic alignment review containing 197 versioned inputs collected from 18 real Lean projects, with 87 aligned and 110 misaligned reference judgments. Each input binds an informal mathematical target to selected formal declarations and an extracted dependency context. Construction combines development-history discovery, independent target binding, reconstruction under a common Lean environment, and Astra annotation with human review of uncertain cases. The collection spans diverse mathematical domains and includes 69 historical pairs. We evaluate six language models with Dire
    
[^45]: 推理何时有助于实体消歧？领域内及分布偏移下的小型大语言模型

    When Does Reasoning Help Entity Disambiguation? Small LLMs In-Domain and Under Shift

    [https://arxiv.org/abs/2611.30334](https://arxiv.org/abs/2611.30334)

    该论文在经典由粗到细实体消歧流程中，系统比较了小型大语言模型下逐点式、列表式与基于推理的候选选择方法在领域内和分布偏移场景下的表现，探究了小规模场景下推理何时值得其代价。

    

    实体消歧（将文本中的歧义提及映射到知识库中的实体）是知识图谱构建、问答和信息抽取等任务的基础步骤。近期工作已将大语言模型应用于这一问题，并取得了最先进的性能。值得注意的是，已有研究表明推理型大语言模型能够以真正的零样本方式执行实体消歧，并可跨知识库和领域泛化。然而，尚不清楚这种能力在较小的大语言模型上是否依然成立或保持竞争力，因为在小型模型上，实体消歧通常依赖重排序而非推理。我们探讨在小规模场景下，推理何时值得其付出的代价。我们在经典的由粗到细实体消歧流程中，比较了逐点式、列表式和基于推理的候选选择组件，涵盖经典、少样本和真正零样本三种设置。该推理范式通过蒸馏进行训练，（摘要至此被截断）

    arXiv:2611.30334v1 Announce Type: new  Abstract: Entity disambiguation (mapping ambiguous mentions in text to entities in a knowledge base) is a foundational step in tasks such as knowledge graph construction, question answering, and information extraction. Recent work has applied LLMs to this problem, achieving stateof-the-art performance. Notably, reasoning LLMs have been shown to perform entity disambiguation in a true zero-shot fashion, generalizing across knowledge bases and domains. However, it is unclear whether this capability survives, or remains competitive, with smaller LLMs, where entity disambiguation has typically relied on reranking rather than reasoning. We ask when reasoning is worth its cost at small scale. We compare pointwise, listwise, and reasoning-based candidate-selection components within a classical coarse-to-fine entity disambiguation pipeline, across the classical, few-shot and true zero-shot settings. The reasoning paradigm is trained with distillation and 
    
[^46]: 无需外部参考标注的LLM用户模拟器人格一致性评估

    Evaluating Persona Consistency in LLM User Simulators Without External Reference Annotations

    [https://arxiv.org/abs/2611.30280](https://arxiv.org/abs/2611.30280)

    该论文提出一种无需外部参考标注的自我锚定内部一致性诊断方法，通过让LLM用户模拟器“自我报告一次、行为两次”，在共享的BFI-2六维度空间中量化自我报告与行为表现之间的相关性，从而评估和比较不同的人格构建策略。

    

    标准评估方法将LLM模拟的人格与外部真值（例如社区特质标注）进行对比，但当人格仅从简短的、面向任务的对话中推断出来时，这种参考是无法获得的。我们探讨如何在没有此类参考的情况下评估人格一致性，并提出了一种自我锚定的内部一致性诊断方法：人格描述自己一次并行动两次。李克特量表自我报告提供了软性的内部锚点；两个行为画像分别来自无需评审即可评分的受限场景选择，以及由LLM评判的开放式回答。一致性被定义为自我报告与每个行为画像之间的相关性，两者处于共享的BFI-2六维度空间中。该框架通过评估各种人格构建策略所引出的反应在多大程度上与其自我报告的特质保持一致，从而对这些策略进行比较。该方法在普通用户对话（WildChat和LMSYS）以及具有社区标注、文献记录完善的人格上进行了验证。

    arXiv:2611.30280v1 Announce Type: cross  Abstract: Standard evaluations assess LLM-simulated personas against an external ground truth, such as community trait annotations, but that reference is unavailable when the persona is inferred only from a short, task-oriented chat. We ask how persona consistency can be evaluated without such a reference, and propose a self-anchored internal-consistency diagnostic: the persona describes itself once and acts twice. A Likert self-report supplies a soft internal anchor; two behavioral profiles come from constrained scenario choices scored without a judge and from open-ended responses judged by an LLM. Consistency is the correlation between the self-report and each behavioral profile, in a shared six-facet BFI-2 space. The framework compares persona-construction strategies by assessing how consistently their elicited responses reflect their self-reported traits. Across ordinary-user chats (WildChat and LMSYS) and well-documented personas with commu
    
[^47]: DUET：面向低资源语言的双MDL无监督熵-字典树形态切分

    DUET: Dual-MDL Unsupervised Entropy-Trie Segmentation for Low-Resource Languages

    [https://arxiv.org/abs/2611.30273](https://arxiv.org/abs/2611.30273)

    提出DUET算法，通过将分支熵与递归范式发现相结合生成候选切分，并采用双重MDL准则在奖励语素复用与防止过度切分之间取得平衡，实现了无需标注数据和外部资源的低资源语言无监督形态切分。

    

    形态切分，即将词语分解为其组成语素的任务，在极端数据稀缺的情况下变得尤为困难，这种情形可能出现在语言记录和语言推理过程中。这一困难源于现有方法对标注数据、外部资源或仅在较大语料库中才会显现的重复模式的依赖。为解决这一局限，我们提出了双MDL无监督熵-字典树切分算法（DUET），这是一种确定性的无监督算法，它将结构候选生成与统计候选排序解耦。DUET将分支熵与递归式范式发现相结合，生成语言学上合理的候选切分，然后使用双重最小描述长度（MDL）准则对候选进行排序：语料库层面的目标函数奖励语素复用，而构词树层面的目标函数则防止过度切分。在SIGMORPHON 2023行间标注数据（原文摘要在此处被截断）

    arXiv:2611.30273v1 Announce Type: new  Abstract: Morphological segmentation, the task of decomposing words into constituent morphemes, becomes particularly challenging under extreme data scarcity, a regime that can arise in language documentation and linguistic reasoning. This difficulty stems from existing approaches' reliance on annotated data, external resources, or recurring patterns that emerge only in larger corpora. To address this limitation, we propose Dual-MDL Unsupervised Entropy-Trie Segmentation (DUET), a deterministic unsupervised algorithm that decouples structural candidate generation from statistical candidate ranking. DUET combines branching entropy with recursive paradigm discovery to generate linguistically plausible candidate segmentations, then ranks them using a dual minimum description length (MDL) criterion: a corpus-level objective rewards morpheme reuse, while a construction-tree objective guards against over-segmentation. On the SIGMORPHON 2023 interlinear g
    
[^48]: Voyag：面向车载语音助手的联合多意图检测与执行关系分类

    Voyag: Joint Multi-Intent Detection and Execution-Relation Classification for In-Vehicle Voice Assistants

    [https://arxiv.org/abs/2611.30253](https://arxiv.org/abs/2611.30253)

    Voyag 是一个单次前向的联合决策模型，在车载语音助手场景中同时预测十个车载意图槽位和每对共现意图之间的三分类执行关系（并行、顺序、互斥），并通过统一联合损失进行端到端优化。

    

    车载语音助手不仅需要判断一句话中包含哪些意图，还需要决定这些共现意图应如何执行：并发执行、按固定顺序执行，或者在不做仲裁的情况下完全不执行。我们提出了 Voyag，一种单次前向的决策模型，能够联合预测十个受控的车载意图槽位，以及每对共现意图之间的三分类执行关系（并行、顺序、互斥）。Voyag 结合了一个多语言 ModernBERT 风格的编码器（mmBERT-base，3.069 亿参数）、一个带类型的逐槽位选择头，以及一个轻量级的 GroupCalibration 关系头（161 万参数），该关系头接收标记隐藏状态、池化上下文、显式资源需求特征以及对称化外积冲突特征作为输入。所有目标均在一个联合损失中优化：槽位选择上的软交叉熵、关系上的焦点交叉熵，以及互斥一致性惩罚项。在合成的中文车生活基准测试上

    arXiv:2611.30253v1 Announce Type: cross  Abstract: An in-vehicle voice assistant must decide not only which intents an utterance contains but also how co-active intents may be executed: concurrently, in a fixed order, or not at all without arbitration. We present Voyag, a single-forward decision model that jointly predicts ten controlled vehicle-intent slots and a three-way execution relation (parallel, sequential, mutually exclusive) over each co-active intent pair. Voyag combines a multilingual ModernBERT-style encoder (mmBERT-base, 306.9M parameters), a typed per-slot choice head, and a lightweight GroupCalibration relation head (1.61M parameters) that consumes marker hidden states, pooled context, explicit resource-requirement profiles, and symmetrized outer-product conflict features. All objectives are optimized in one joint loss: soft cross-entropy over slot choices, focal cross-entropy over relations, and a mutex-consistency penalty. On a synthetic Chinese vehicle-life benchmark
    
[^49]: StutterStigma-TR：一个新颖的土耳其语数据集与目标感知BERT框架，用于口吃相关污名化言论检测

    StutterStigma-TR: A Novel Turkish Dataset and Target-Aware BERT Framework for Stuttering-Related Stigma Detection

    [https://arxiv.org/abs/2611.30252](https://arxiv.org/abs/2611.30252)

    本文提出了首个土耳其语口吃污名化检测数据集StutterStigma-TR（含6,019条社交媒体帖子），并设计了基于BERTurk的目标感知框架StutterStigma-BERT，通过显式目标标记与多维度特征融合显著提升了口吃相关污名化言论的检测效果。

    

    关于残疾相关能力主义伤害的计算研究仍然有限，而针对口吃相关污名化的特定领域资源尤为稀缺。本研究介绍了StutterStigma-TR，这是一个包含6,019条公开X/Twitter帖子的土耳其语数据集，采用基于言语语言治疗专业知识的标注协议，将这些帖子标注为“污名化”、“非污名化/语境性”或“支持性”三类，并进行了独立的生活经验评估。我们还提出了StutterStigma-BERT，这是一个基于BERTurk的目标感知框架，结合了显式目标标记、目标跨度表示、目标条件化上下文、字符级与手工构建的正字法信息，以及门控话语线索。该框架在互补的三分类和二分类设置下进行评估，采用受控的三种子消融分析和最终的20种子鲁棒性评估。在通用训练协议下，使用交叉熵训练的Gated Discourse-OrthoTarget模型被选为最终方案。

    arXiv:2611.30252v1 Announce Type: cross  Abstract: Computational research on disability-related ableist harm remains limited, and condition specific resources for stuttering-related stigma are particularly scarce. This study introduces StutterStigma-TR, a Turkish dataset of 6,019 public X/Twitter posts annotated as Stigmatizing, Non-stigmatizing/contextual, or Supportive using a speech and language therapy-informed protocol with independent lived-experience evaluation. We also propose StutterStigma-BERT, a BERTurk-based target-aware framework that combines explicit target marking, target-span representation, target-conditioned context, character-level and handcrafted orthographic information, and gated discourse cues. The framework is evaluated in complementary three-class and binary settings using controlled three-seed ablation analyses and final 20-seed robustness evaluations. Under the common training protocol, Gated Discourse-OrthoTarget trained with cross-entropy is selected as th
    
[^50]: 基于本体编译二维临床状态场与双向扩散编码器的零训练跨本体医学编码

    Ontology-Compiled 2D Clinical State Fields with Bidirectional Diffusion Encoders for Zero-Training Cross-Ontology Medical Coding

    [https://arxiv.org/abs/2611.30218](https://arxiv.org/abs/2611.30218)

    该论文提出无词表框架OCSF，通过将临床文档编码为固定维度的二维状态矩阵，并利用离线编译的Tikhonov反卷积算子匹配任意目标本体，实现了无需重新训练、数秒内即可切换医学编码词表的零训练跨本体医学编码。

    

    医疗系统不断修订临床分类体系、更新报销计划并协调跨国登记系统。现有的自动化编码模型将临床文本理解绑定到固定的词表专用输出头上，每当编码集合发生变更时都需要人工重新标注和重新训练。我们提出了本体编译二维临床状态场，这是一个无词表框架，它将临床文档阅读与目标本体索引解耦，使得一个训练好的编码器可以在数秒内切换到任意医学词表，而无需针对目标词表进行训练。单次前向传播的双向扩散预训练编码器与无词表解码器将每份临床病历映射为固定维度的二维状态矩阵，不包含任何编码特定参数。任意目标本体都可以通过公开的文本描述在CPU上离线编译为一个尺度不变的Tikhonov反卷积算子，该算子在压制同级编码干扰的同时……（原文摘要在此处截断）

    arXiv:2611.30218v1 Announce Type: new  Abstract: Healthcare systems continually revise clinical taxonomies, update reimbursement schedules, and harmonize cross-border registries. Existing automated coding models bind clinical text understanding to fixed vocabulary-specific heads, requiring manual relabeling and retraining whenever code sets change. We introduce Ontology-Compiled 2D Clinical State Fields (OCSF), a vocabulary-free framework that decouples clinical document reading from target ontology indexing so one trained encoder can switch across arbitrary medical vocabularies in seconds without target training. A single-pass bidirectional diffusion-pretrained encoder and vocabulary-free decoder map each clinical note into a fixed-dimension two-dimensional state matrix with zero code-specific parameters. Any target ontology is compiled offline on CPU from public text descriptions into a scale-invariant Tikhonov deconvolution operator that suppresses sibling-code interference while pr
    
[^51]: 什么才是训练编码智能体的好数据？基于SWE-Gym的系统性研究

    What Makes Good Data for Training Coding Agents? A Systematic Study on SWE-Gym

    [https://arxiv.org/abs/2611.30207](https://arxiv.org/abs/2611.30207)

    本研究系统审计了SWE-Gym数据集奖励标签的正确性，发现模型从未解决的任务中74%存在拒绝正确修复的假阴性测试，修复或移除这些错误标注任务后，训练出的模型在SWE-bench Verified和SWE-bench Pro上分别提升2.4分和2.7分。

    

    强化学习（RL）通过软件任务来训练编码智能体，其奖励由隐藏测试决定，然而现有任务集合仅针对参考修复方案来验证这些测试。我们探究什么使一个任务成为好的训练数据，并重点关注其奖励标签的正确性。我们使用两个大语言模型系列对SWE-Gym进行了审计，且只有当执行能够复现疑似错误时才将其计为错误。我们发现“未被解决”往往意味着“被错误标注”：在一个9B基础模型从未解决的任务中，74%的任务存在会拒绝正确修复方案的测试（即假阴性），而在其偶尔能解决的任务中这一比例仅为20%。我们修复或移除了这类任务，审计后保留的任务能够训练出更好的策略。在相同规模和匹配难度下，用这些任务训练的模型在SWE-bench Verified和SWE-bench Pro的保留任务上分别比用原始任务训练的模型高出2.4分和2.7分，而简单地删除从未被解决的任务则无法带来这样的提升。

    arXiv:2611.30207v1 Announce Type: cross  Abstract: Reinforcement learning (RL) trains coding agents on software tasks whose hidden tests decide the reward, yet task collections check these tests only against the reference fix. We ask what makes a task good training data, focusing on the correctness of its reward labels. We audit SWE-Gym with two LLM families, and a suspected error counts only if execution reproduces it. We find that unsolved often means mislabelled: 74% of the tasks that a 9B base model never solves have a test that rejects correct fixes (a false negative), versus 20% of those it sometimes solves. We repair or remove such tasks, and the audited tasks that remain train better policies. At equal size and matched difficulty, models trained on them score 2.4 and 2.7 points higher on held-out tasks from SWE-bench Verified and SWE-bench Pro than models trained on the original tasks, whereas simply dropping the never-solved tasks shows no such gain. We further find that wrong
    
[^52]: PackLoop：并发场景下循环语言模型的行打包自推测解码

    PackLoop: Row-Packed Self-Speculative Decoding for Looped Language Models under Concurrency

    [https://arxiv.org/abs/2611.30205](https://arxiv.org/abs/2611.30205)

    PackLoop 发现并发负载下自推测解码失效的原因是推测工作的布局而非推测本身，并提出只在必然要读取的 KV 缓存行内打包推测位置，使循环语言模型在高并发下仍保持加速收益。

    

    循环语言模型对每个 token 重复使用同一权重块 T 次，以参数换取测试时的深度，但自回归解码随后需要为每个 token 进行 T 次顺序传递。自推测解码器能为单个请求带来 2-10 倍的加速，然而当多个请求共享一个 GPU 时，其收益会减弱，目前的常见做法是随着负载增长而减少推测。我们证明，在高负载下失效的是推测工作的布局方式，而不是推测本身。在并发解码中，成本的基本单位是“行”，即一个请求在一个循环上的推进，它会读取该循环的键值（KV）缓存一次；同一行内额外的位置几乎不产生额外成本。诸如波前调度之类的流水线调度将每个推测位置放在不同的循环上，因而各自占据一行，导致每个 token 仍需被读取 T 次。我们提出 PackLoop，其建立在一个原则之上：只在本就无论如何都会被读取的行内进行推测。深循环累积多个位置并将它们作为一个块在同一行内推进……

    arXiv:2611.30205v1 Announce Type: cross  Abstract: Looped language models reuse one block of weights T times per token, trading parameters for test-time depth, but autoregressive decoding then pays T sequential passes per token. Self-speculative decoders recover 2-10x for a single request, yet their gains fade as requests share a GPU, and current practice is to speculate less as load grows. We show that what fails under load is the layout of speculative work, not speculation. In concurrent decoding the unit of cost is the row, one request advancing on one loop, which reads that loop's key-value (KV) cache once; additional positions inside a row are nearly free. Pipelined schedules such as the wavefront place every speculative position on a different loop, hence in a row of its own, so each token is still read T times. We present PackLoop, built on one principle: speculate only inside rows that are read anyway. Deep loops accumulate positions and advance them as a block in one row, cont
    
[^53]: VADE：一个源自专业来源的可验证智能体数据引擎

    VADE: A Verifiable Agent Data Engine from Professional Sources

    [https://arxiv.org/abs/2611.30202](https://arxiv.org/abs/2611.30202)

    VADE是一个数据引擎，它利用专业分析“求解难、验证易”的非对称性，从法律、金融等专业实践的公开来源中自动构建可验证的分析任务，从而解决这些领域高质量智能体训练数据难以规模化的问题。

    

    语言智能体已经展示了其实用价值，尤其是在软件工程领域，那里丰富的公开代码和可执行测试支持了大规模的训练与评估。然而，将这一成功扩展到其他专业领域（如法律和金融）并非易事：许多智能体任务涉及开放式的分析，其结果难以验证，因而也难以用作可扩展的监督信号。因此，这些领域中的高质量任务仍主要由专家手工编写，导致扩展成本高昂，限制了其用于训练的价值。为了解决这一瓶颈，我们提出了VADE——一个能够从专业实践中使用的公开来源自动构建可验证分析任务的数据引擎。VADE利用了一个简单的非对称性：专业分析之所以难以求解，是因为相关证据必须先被找到并加以解读；而一旦证据被固定，验证就变得容易得多。在任务合成过程中

    arXiv:2611.30202v1 Announce Type: new  Abstract: Language agents have shown their practical value, particularly in software engineering, where abundant public code and executable tests support training and evaluation at scale. Extending this success to other professional domains, such as law and finance, is not straightforward: many agent tasks involve open-ended analysis, making outcomes hard to verify and, in turn, hard to use as scalable supervision. As a result, high-quality tasks in these domains are still largely expert-authored, making them expensive to scale and limiting their use for training. To address this bottleneck, we introduce VADE, a data engine that automatically constructs verifiable analysis tasks from public sources used in professional practice. VADE exploits a simple asymmetry: professional analyses are difficult to solve because the relevant evidence must first be found and interpreted, but much easier to verify once that evidence is fixed. During task synthesis
    
[^54]: Laya-IsoToken：通过闭式算子编译实现临床编码本的首日切换

    Laya-IsoToken: Day-One Clinical Codebook Switching via Closed-Form Operator Compilation

    [https://arxiv.org/abs/2611.30196](https://arxiv.org/abs/2611.30196)

    该论文提出Laya-IsoToken方法，通过将临床证据读取器与冻结的多语言概念空间解耦，并利用闭式矩阵编译对齐目标编码描述，实现了在无需任何目标编码本标注病历的情况下首日完成临床编码本切换。

    

    医疗系统会定期修订诊断和操作分类（如从ICD-9到ICD-10/11），更新医院报销分组器（MS-DRG），并需要跨语言部署临床编码。监督式编码网络将临床文本编码器绑定到固定的逐码分类器权重上，迫使医院在切换编码本之前重新标注数万条记录；而独立使用的大型语言模型则存在高延迟问题，且会产生无效编码的幻觉。本研究表明，当局部临床证据读取与严格冻结的多语言概念空间解耦时，无需目标编码本标注的病历即可实现临床编码本的首日切换。Laya-IsoToken在源记录上仅训练一次临床片段读取器，不含任何逐码分类器参数，从而将文本证据与历史疾病流行率分离。仅需公开的目标编码描述，闭式矩阵编译即可对齐分布偏移（原文在此处截断）。

    arXiv:2611.30196v1 Announce Type: new  Abstract: Healthcare systems periodically revise diagnostic and procedural classifications (ICD-9 to ICD-10/11), update hospital reimbursement groupers (MS-DRG), and deploy clinical coding across languages. Supervised coding networks bind clinical text encoders to fixed per-code classifier weights, forcing hospitals to re-annotate tens of thousands of records before switching codebooks, whereas standalone large language models incur high latency and hallucinate invalid codes. Here we show that day-one clinical codebook switching becomes feasible without target-labeled notes when local clinical evidence reading is decoupled from a strictly frozen multilingual concept space. Laya-IsoToken trains a clinical span reader once on source records with zero per-code classifier parameters, separating text evidence from historical disease prevalence. Given only public target code descriptions, closed-form matrix compilation aligns distributional shifts, comp
    
[^55]: 相同执行、不同差距：Web智能体评估中的判定规则

    Same Executions, Different Gaps: Judgment Rules in Web-Agent Evaluation

    [https://arxiv.org/abs/2611.30190](https://arxiv.org/abs/2611.30190)

    该论文揭示了Web智能体评估中的判定规则问题：评判标准与证据要求的选择可以在不改变排名的情况下大幅改变所报告的性能差距，且提高双方分数估计的准确性反而可能降低差异估计的准确性。

    

    成功率的提升引导着Web智能体的发展。然而，知道哪个智能体排名第一并不能告诉我们应该对其所报告的优势抱有多大的信心。我们研究的是比较过程本身：成功是如何定义的、什么样的证据支持这些判断，以及智能体之间的差异是如何被估计的。在固定的Web智能体执行记录上，将评判者从预测可能完成任务改为要求有记录的证据支持，可以在不改变排名的情况下消除大部分所报告的领先优势。当把未确定的完成与有证据支持的失败区分开时，这种效应依然存在。通过受控的证据来源添加实验，我们解释了任务需求、对象和观察之间缺失的关联如何掩盖真实差异或制造出表面上的优势。我们还发现，让两个智能体的分数估计都更加准确，反而可能使它们之间差异的估计更不准确，因为它们最初的误差部分地相互抵消了。为了估计所选定的结果差异，我们结合……（摘要在此处截断）

    arXiv:2611.30190v1 Announce Type: new  Abstract: Success-rate gains guide the development of web agents. Yet knowing which agent ranks first does not tell us how much confidence to place in its reported advantage. We study the comparison itself: how success is defined, what evidence supports the judgments, and how the difference between agents is estimated. On fixed web-agent executions, changing a judge from predicting likely completion to requiring recorded support can remove most of a reported lead without changing the ranking. This effect persists when unresolved completion is distinguished from supported failure. Controlled source additions explain how missing relations between task requirements, objects and observations can conceal a difference or create an apparent advantage. We also find that making both agents' score estimates more accurate can make their difference less accurate, because their original errors partly canceled. To estimate a chosen outcome difference, we combin
    
[^56]: PCC：面向大语言模型层剪枝的关键通道补偿

    PCC: Pivotal Channel Compensation for Layer Pruning of Large Language Models

    [https://arxiv.org/abs/2611.30183](https://arxiv.org/abs/2611.30183)

    提出关键通道补偿（PCC）方法，通过在剪枝接口处仅对少量关键通道施加可学习的线性变换，以低成本显著恢复大语言模型层剪枝后的性能。

    

    将独立的层作为最小剪枝单元，使层剪枝能够高效消除大量冗余参数并实现高压缩率，层剪枝已成为大语言模型（LLM）中一种有前景且流行的结构化剪枝技术。然而，移除多层会导致大语言模型性能显著下降。尽管已有一些工作探索了针对被剪枝层的各种补偿方法，但它们效率低下且成本高昂。我们观察到大部分信息的表达集中在少数关键通道上，据此提出了一种新颖的层剪枝后补偿方法，称为关键通道补偿（PCC）。PCC 通过在剪枝接口处对少量关键通道进行可学习的线性变换补偿，能够实现显著的性能提升。此外，我们还提出了一个新的评估指标，称为通道级平均……（摘要截断）

    arXiv:2611.30183v1 Announce Type: cross  Abstract: Treating independent layers as the minimal pruning unit allows layer pruning to efficiently eliminate many redundant parameters and achieve high compression rates. Layer pruning has become a promising and prevalent structured pruning technique for large language models (LLMs). However, multi-layer removal can lead to significant performance degradation for LLMs. Although some works have explored various compensation methods for pruned layers, they are inefficient and costly. Observing that the expression of a large proportion of information is concentrated on a small number of pivotal channels, we propose a novel post-pruning compensation method for layer pruning, called pivotal channel compensation (PCC). PCC can achieve significant performance improvement by compensating for a small number of channels through learnable linear transformations at the pruning interface. Besides, we propose a new evaluation metric, called channel-wise av
    
[^57]: 智能体时代的检索增强生成：传统、现状与展望

    Retrieval-Augmented Generation in the Agentic Era: Tradition, Present, and Prospect

    [https://arxiv.org/abs/2611.30181](https://arxiv.org/abs/2611.30181)

    该论文从智能体时代视角出发，沿着统一的生命周期轴（离线索引、检索前、检索、检索后与编排）系统综述了RAG的最新进展，并首次将索引与编排作为一等组件、将智能体工作流作为独立范式加以梳理，同时讨论了评估、应用与未来开放问题。

    

    检索增强生成（RAG）将冻结的语言模型与外部可更新的知识相结合，使生成过程以检索到的证据为基础，而不再仅仅依赖参数化记忆。本工作通过将各阶段统一到单一的生命周期轴上——从离线索引，到检索前、检索、检索后，以及对这些阶段的编排——勾勒了智能体时代的RAG图景。它将索引与编排视为一等组件，将智能体工作流作为一种独立范式加以论述，并涵盖了此前综述之后出现的智能体相关工作。除上述组件外，本工作还涵盖了评估、应用以及未来探索中尚待解决的研究问题。从传统到当下乃至未来，本工作从智能体时代的视角对RAG进行了系统性综述。

    arXiv:2611.30181v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) couples frozen language models with external, updatable knowledge, so that generation is grounded in retrieved evidence rather than in parametric memory alone. This work maps RAG in the agentic era by joining them on a single lifecycle axis, from offline indexing through pre-retrieval, retrieval, post-retrieval, and orchestration over these stages. It treats indexing and orchestration as first-class components and agentic workflows as a paradigm of their own, and it covers the agentic work that has appeared since those reviews. Beyond these components, it covers evaluation, applications, and the questions that remain open for future exploration. From tradition to present and beyond, this work systematically reviews RAG from a perspective of the agentic era.
    
[^58]: FiRE：面向机器翻译的细粒度排序评估方法

    FiRE: Fine-grained Ranking Evaluation for Machine Translation

    [https://arxiv.org/abs/2611.30168](https://arxiv.org/abs/2611.30168)

    该论文提出FiRE方法，利用现成大语言模型在忠实度、流畅性和风格一致性三个维度上进行基于标准的成对比较，实现机器翻译的可解释细粒度无参考排序评估，并构建了首个面向该任务的人工标注基准。

    

    开发可靠的机器翻译（MT）系统，关键在于我们能否区分优质译文与劣质译文。然而，现有的评估范式要么局限于粗粒度的整体排序，要么与人类偏好不一致，均无法在无参考设置下提供可解释的、细粒度的反馈。我们提出了一种细粒度排序评估方法FiRE，该方法利用现成的大语言模型，在三个互补维度——忠实度、流畅性和风格一致性——上执行基于标准的成对比较，而非产生单一的整体判断。为了在缺乏合适测试平台的情况下对评估范式进行严格的元评估，我们构建了首个用于细粒度排序评估的人工标注、无参考基准，并取得了较高的标注者间一致性。通过在该基准及现有MQM数据集上进行元评估，FiRE...

    arXiv:2611.30168v1 Announce Type: cross  Abstract: Developing reliable machine translation (MT) systems hinges on our ability to distinguish superior translations from inferior ones. However, existing evaluation paradigms, whether limited to coarse overall rankings or misaligned with human preferences, fail to deliver interpretable, fine-grained feedback in reference-free settings. We present a Fine-Grained Ranking Evaluation method (FiRE) that leverages off-the-shelf large language models to perform criterion-driven pairwise comparison across three complementary dimensions: faithfulness, fluency, and consistency of style, instead of producing a single holistic judgment. To enable rigorous meta-evaluation of evaluation paradigms in the absence of any suitable testbed, we construct the first human-annotated, reference-free benchmark for fine-grained ranking evaluation, achieving substantial inter-annotator agreement. Through meta-evaluation on this benchmark and existing MQM datasets, F
    
[^59]: OpenViVE：越南语视觉蕴含的新基准与双流协同注意力框架

    OpenViVE: New Benchmark and Dual-Stream Co-Attention Framework for Vietnamese Visual Entailment

    [https://arxiv.org/abs/2611.30161](https://arxiv.org/abs/2611.30161)

    该论文提出了首个大规模越南语母语视觉蕴含基准OpenViVE（含62,268个图像-假设对），通过伪影感知的LLM-人类混合流水线有效减少数据伪影，并进一步提出基于双向协同注意力的双流PhoBERT-ConvNeXt框架，取得了84.84%的准确率。

    

    视觉蕴含（VE）任务旨在判断一个文本假设相对于图像而言是被蕴含、被矛盾还是无法验证。然而，现有的VE基准主要以英语为中心，未能捕捉越南语特有的语言差异或扎根于文化的视觉语境。为填补这一空白，我们提出了OpenViVE——首个大规模越南语母语VE基准，包含来自10,378张扎根于越南文化的图像的62,268个均衡的图像-假设对。OpenViVE采用一种“伪影感知的LLM-人类混合流水线”构建，该流水线结合了受约束生成、人工验证和PMI引导的精炼。该流水线显著减少了“仅依赖假设”的数据捷径，在伪影精炼后，矛盾类的准确率从96%降至66%。我们进一步评估了一个采用双向协同注意力机制的双流PhoBERT-ConvNeXt框架，以将视觉证据与越南语假设对齐。该框架达到了84.84%的准确率。

    arXiv:2611.30161v1 Announce Type: cross  Abstract: Visual Entailment (VE) determines whether a textual hypothesis is entailed, contradicted, or unverifiable from an image. However, existing VE benchmarks are largely English-centric and do not capture Vietnamese-specific linguistic distinctions or culturally grounded visual contexts. To address this gap, we introduce OpenViVE, the first large-scale native Vietnamese VE benchmark, containing 62,268 balanced image-hypothesis pairs from 10,378 culturally grounded images. OpenViVE is constructed using an Artifact-Aware Hybrid LLM-Human Pipeline that combines constrained generation, human validation, and PMI-guided refinement. The pipeline substantially reduces hypothesis-only shortcuts, with Contradiction accuracy dropping from 96% to 66% after artifact refinement. We further evaluate a dual-stream PhoBERT-ConvNeXt framework with bidirectional co-attention to align visual evidence with Vietnamese hypotheses. The framework achieves 84.84% ac
    
[^60]: 再度解读茶渣征兆：大语言模型能否预测语言模型的训练运行？

    Reading the Tea Leaves, Again: Can Large Language Models Predict Language Model Training Runs?

    [https://arxiv.org/abs/2611.30141](https://arxiv.org/abs/2611.30141)

    该论文提出了Tea Leaves基准，通过让大语言模型根据训练运行前20%的配置、描述、代码和曲线来预测其余结果，在682个真实训练运行上评估了11个LLM的预测能力，发现最强LLM在平滑损失曲线上可比肩最佳曲线拟合器，而在有噪声的强化学习奖励曲线上则显著领先。

    

    如今，LLM智能体已能够提出研究想法并运行验证这些想法的实验，但它们会将每一次训练运行都执行到底；而人类研究者则会通过阅读训练运行的配置和曲线的开端，停止那些没有前景的运行。以这种方式解读一次训练运行，需要的不仅仅是描述或实现某个训练方法，而是要了解一个设计选择会对训练运行产生何种影响。我们构建了Tea Leaves来衡量这种知识，涵盖预训练、微调和强化学习（RL）共682个真实训练运行：预测者可以看到一次训练运行的前20%，包括其配置、描述和代码，并据此预测其余部分。在11个LLM和一系列预测基线方法的对比中，最强的LLM在平滑的损失曲线上达到甚至超过了最佳曲线拟合器的水平（技能分数0.66对0.62），并在有噪声的RL奖励曲线上明显领先（0.43对0.13）。限制所有预测者的关键在于日志所承载的信息：在日志记录稀疏的社区运行数据上，所有方法都损失了大部分……

    arXiv:2611.30141v1 Announce Type: new  Abstract: LLM agents now propose research ideas and run the experiments that test them, but they train every run to its end, whereas a researcher reads a run's configuration and the beginning of its curve and stops the runs that show no promise. Reading a run this way takes more than describing or implementing a training method: it takes knowing what a design choice does to a run. We build Tea Leaves to measure this knowledge on 682 real training runs across pretraining, fine-tuning, and reinforcement learning (RL): a forecaster sees the first 20% of a run with its configuration, description, and code, and predicts the rest. Across 11 LLMs and a range of forecasting baselines, the strongest LLMs match or exceed the best curve fitter on smooth loss curves (skill 0.66 against 0.62) and lead on noisy RL reward curves (0.43 against 0.13). What bounds every forecaster is what the log carries: on sparsely logged community runs all methods lose most of t
    
[^61]: 光学上下文压缩何时划算？面向推理型视觉语言模型的解码感知成本分析

    When Does Optical Context Compression Pay Off? A Decode-Aware Cost Analysis of Reasoning VLMs

    [https://arxiv.org/abs/2611.30130](https://arxiv.org/abs/2611.30130)

    该研究揭示推理型视觉语言模型将长文本渲染为图像虽节省输入token，但会使解码长度增长1.5-3.5倍并可能反而推高总成本，并提出判断光学压缩是否划算的汇率公式 (1-1/c)/(r-1)，即该比值需超过工作负载的输出/输入token价格加权比。

    

    将长文本渲染为图像可以让视觉语言模型（VLM）以更少的输入token读取文档，而现有评估通常主要计入这些预填充token。我们表明，对于推理模型而言，这种核算方式是不完整的。在五个模型和两个长上下文基准上的实验显示，读取渲染后的文档会使模型“思考”得更久：在测试的四个通用VLM上，解码长度增长1.5-3.5倍，并且在典型的输出定价下，这种“解码税”使得在QuALITY基准上五个模型中有两个的图像输入成本反而高于文本输入。在中等压缩率下，不可恢复的信息损失并不足以解释这一解码税：Qwen对相同图像的OCR识别恢复了98.7%的词，而将Kimi-K3自己生成的转录文本作为文本输入时，其推理长度和准确性会恢复到文本模式水平。压缩是否划算最终可归结为一个“汇率”——(1-1/c)/(r-1)，该比值必须超过该工作负载中输出与输入token的价格加权比率；这一分析

    arXiv:2611.30130v1 Announce Type: new  Abstract: Rendering long text as images lets a vision-language model (VLM) read a document with fewer input tokens, and evaluations often count mainly these prefill tokens. We show that for reasoning models this accounting is incomplete. Across five models and two long-context benchmarks, reading the rendered document makes the model think longer: decode length grows by 1.5-3.5x on the four general-purpose VLMs tested, and under typical output pricing this decode tax makes image input more expensive than text for two of the five models on QuALITY. Unrecoverable information loss is not sufficient to explain the tax at moderate compression: Qwen's OCR of the same images recovers 98.7% of words, and feeding Kimi-K3 its own transcription as text restores text-mode reasoning length and accuracy. Whether compression pays off reduces to an exchange rate, (1-1/c)/(r-1), that must exceed the workload's price-weighted ratio of output to input tokens; this a
    
[^62]: 通过跨语言嵌入对齐实现参数高效的词表扩展

    Parameter-Efficient Vocabulary Expansion via Cross-Lingual Embedding Alignment

    [https://arxiv.org/abs/2611.30123](https://arxiv.org/abs/2611.30123)

    提出参数高效的词表扩展方法MATE，通过利用主语言与扩展语言词汇间的语义等价性进行跨语言嵌入对齐，使每个词元的额外参数减少8倍，同时提升多个基准测试的性能。

    

    通过词表扩展进行持续预训练常被用于使大语言模型（LLM）在处理新语言时更有效、更高效。然而，当LLM同时扩展到多种语言时，模型参数数量会显著增加。此外，新增的词元无法获得与原始词表中的词元相同数量的训练步数，这有时导致词表扩展只能带来有限的性能提升。我们提出利用模型主要语言中的词汇与扩展词表中的词汇之间的语义等价性来迁移已学习的表示。在本文中，我们介绍了MATE，这是一种参数高效的词表扩展方法，能够在每个词元仅需八分之一的额外参数的情况下实现有效的跨语言迁移。MATE降低了平均分词压缩比，并在多个基准测试中带来了性能提升。

    arXiv:2611.30123v1 Announce Type: new  Abstract: Continual pretraining with vocabulary expansion is often used to make large language models (LLMs) more effective and efficient at handling new languages. However, when LLMs are extended to many languages simultaneously, the number of model parameters increases significantly. In addition, the newly added tokens do not receive the same number of training steps as the tokens in the original vocabulary, which sometimes causes vocabulary expansion to yield only limited performance improvements. We propose leveraging the semantic equivalence between words in the model's primary language and those in the extended vocabulary to transfer learned representations. In this paper, we introduce MATE, a parameter-efficient vocabulary expansion method that enables effective cross-lingual transfer with 8 times fewer additional parameters per token. MATE reduces the average tokenization compression ratio and delivers improvements on multiple benchmarks.
    
[^63]: 语言模型中的字符信息：独立于提示的词汇存储与随模型规模增长的后期问题驱动信号

    Character Information in Language Models: A Prompt-Independent Word Store and a Late, Question-Driven Signal That Grows With Scale

    [https://arxiv.org/abs/2611.30094](https://arxiv.org/abs/2611.30094)

    本研究通过词位置基线方法将语言模型中词的字符信息分解为两部分——约70%的字符信息（如首字母、尾字母、最频繁字母）已独立于提示预先存储在词元嵌入中，其余部分则是在答案位置由问题驱动、随模型规模增长且局限于后期层的信号。

    

    线性探针能够从语言模型的隐藏状态中恢复词的字符属性，但位于答案位置的探针无法区分该属性是随词本身一同存储的，还是因为提示提出了要求才产生的。我们通过词位置基线将二者分离：将同一个基于留出词训练的探针应用于一个不包含任何提问的句子中查询词自身的词元。在来自七个模型家族的16个开放权重模型（0.6B–14B）和11个字符任务上，全部16个模型中的词元都在仅含词的干扰底线之上编码了首字母、尾字母和最频繁字母信息，且约70%的该信号已存在于词元嵌入之中。提出问题会在答案位置增加第二个信号。对于全部四个提示敏感任务，该信号的规模随模型规模的增长而增大（经Holm校正的置换检验 p ≤ 0.018，在单个模型家族内部同样成立），且这种增长仅局限于模型的后期层：

    arXiv:2611.30094v1 Announce Type: new  Abstract: Linear probes recover character properties of a word from a language model's hidden states, but a probe at the answer position cannot say whether the property was stored with the word or produced because the prompt asked for it. We separate the two with a word-position baseline: the same held-out-word probe applied to the query word's own tokens in a sentence that asks nothing. Across 16 open-weight models from seven families (0.6B--14B) and eleven character tasks, the word tokens encode \texttt{first\_letter}, \texttt{last\_letter} and \texttt{most\_frequent\_letter} above a word-only nuisance floor in all 16 models, and about 70\% of that signal is already present in the token embedding. Asking the question adds a second signal at the answer position. Its size grows with model scale for all four prompt-sensitive tasks (Holm-corrected permutation $p \leq 0.018$, also within model families), and the growth is confined to the late layers:
    
[^64]: ESGPT：具有统一指令与基准资源的以数据为中心的ESG语言模型

    ESGPT: A Data-Centric ESG Language Model with Unified Instruction and Benchmark Resources

    [https://arxiv.org/abs/2611.30052](https://arxiv.org/abs/2611.30052)

    该研究构建了包含3万条实例的ESG指令数据集ESG-Instruct和覆盖六项代表性任务的基准ESGU-Bench，并据此训练出80亿参数的ESG专用大语言模型ESGPT，其以较小的规模在多项ESG任务上超越了表现参差不齐的通用大语言模型。

    

    ESG（环境、社会与治理）分析需要对包括企业报告、新闻和信息披露在内的多种来源进行复杂的语言理解，以完成ESG相关性检测、行为解读、影响分析和问答等任务。然而，现有的ESG数据集在任务和来源上较为分散，格式与标注方案各异，这限制了它们在训练通用ESG大语言模型以及进行统一评估方面的应用。在本研究中，我们构建了ESG-Instruct——一个基于公开ESG资源构建、包含3万条实例的ESG指令数据集，以及ESGU-Bench——一个涵盖六项代表性ESG任务的ESG理解基准。基于这些资源，我们进一步开发了ESGPT——一个80亿参数的ESG专用大语言模型，并将其与多种开源及基于API的大语言模型进行对比评估。实验结果表明，通用大语言模型在不同ESG任务上的表现存在显著差异，而ESGPT尽管规模相对较小，仍取得了（原文摘要在此处截断）……

    arXiv:2611.30052v1 Announce Type: cross  Abstract: ESG analysis requires complex language understanding across diverse sources, including corporate reports, news, and disclosures, for tasks such as ESG relevance detection, action interpretation, impact analysis, and question answering. However, existing ESG datasets are fragmented by task and source, with heterogeneous formats and annotation schemes, limiting their use for training general-purpose ESG LLMs and unified evaluation. In this work, we construct ESG-Instruct, a 30,000-instance ESG instruction dataset built from public ESG resources, and ESGU-Bench, an ESG understanding benchmark covering six representative ESG tasks. Based on these resources, we further develop ESGPT, an 8B ESG-specialized LLM, and evaluate it against diverse open-source and API-based LLMs. Experimental results show that general-purpose LLMs exhibit substantial performance variation across ESG tasks, while ESGPT, despite its relatively small scale, achieves 
    
[^65]: STUN-Bench：用于评估共享工具经验的真正未见工具基准

    STUN-Bench: Truly Unseen Tools for Evaluating Shared Tool Experience

    [https://arxiv.org/abs/2611.30034](https://arxiv.org/abs/2611.30034)

    该论文提出STUN-Bench基准，构建了经验池中完全不包含172个保留工具任何信息的数据分割，并揭示了共享经验会挤占工具描述排名、导致真正未见的工具从候选列表中消失的“拥挤”问题。

    

    工具路由器为每个请求检索少量候选工具，并让语言模型进行选择。跨组织共享经验的方法会将关于每个工具的历史请求添加到路由器检索的内容中。StableToolBench 将其中某些测试组称为对“未见”工具的测试，然而检索器和经验共享方法从中学习的 ToolBench 指令池，却包含了这两个单工具测试组中的每一个测试查询，并标记了其中所有的黄金工具。我们发布了 STUN-Bench（StableToolBench, Truly UNseen），这是一个新的数据分割，其经验池中不存在任何标记或命名172个保留工具的条目。这里的“未见”指的是在共享经验和检索器训练数据中缺失，而非在预训练数据中缺失。在该分割数据集上，我们发现了“拥挤”现象：从五个模拟客户端汇集的经验被添加到工具描述索引后，其排名超过了没有任何客户端使用过的工具的描述，导致这些工具经常从候选列表中被淘汰。在 Jina-v2 检索器下……

    arXiv:2611.30034v1 Announce Type: cross  Abstract: A tool router retrieves a few candidate tools per request and lets a language model choose. Methods that share experience across organizations add past requests about each tool to what the router retrieves. StableToolBench calls some of its test groups tests on "unseen" tools, yet the ToolBench instruction pool, which retrievers and experience-sharing methods learn from, contains every test query and labels every gold tool of the two single-tool groups among them. We release STUN-Bench (StableToolBench, Truly UNseen), a split whose experience pool has no item that labels or names 172 held-out tools. Unseen here means absent from shared experience and retriever training data, not from pretraining. On the split we find crowding: experience pooled from five simulated clients, added to an index of tool descriptions, outranks the descriptions of tools no client has used, which then often drop out of the candidate list. Under the Jina-v2 ret
    
[^66]: SHAKTI：基于证据的营养与健康对话式AI助手

    SHAKTI: Evidence-Grounded Conversational AI for Nutrition and Health

    [https://arxiv.org/abs/2611.29988](https://arxiv.org/abs/2611.29988)

    该论文提出了SHAKTI，一个仅基于检索到的学术文献和精确食物成分数据回答营养健康问题的对话式AI助手，通过来源链接核查和引文数字在线验证等机制确保回答有据可依，有效解决了大语言模型产生无依据论断的问题。

    

    大型语言模型（LLM）能够回答许多饮食与健康问题，但可能陈述缺乏依据的观点，并引用实际上并不支持其说法的文献来源。我们提出了SHAKTI（Smart Health Advisor for Knowledge-based Tailored Interventions，基于知识的定制化干预智能健康顾问），这是一个仅依据检索到的学术文献和精确的食物成分数据进行回答的对话助手，并采用检索、生成、依据性（grounding）、效率和上下文感知等指标对其进行评估。我们索引了484,368,086篇OpenAlex文献，在评估阶段（涵盖NFCorpus、BioASQ、SciFact和PubMedQA共10,065个问题）通过GPU进行精确检索，在部署阶段则通过一个压缩的包含2.3亿篇摘要的索引进行检索。在1,000个分层抽样的问题上，我们将单轮、多跳、纠错型和深度研究型检索增强生成（RAG）方法与Claude Sonnet进行了对比。SHAKTI增加了对话式查询改写、USDA FoodData Central食物数据查询、来源链接核查以及引文和数字的在线验证功能。（注：原摘要在此处截断）

    arXiv:2611.29988v1 Announce Type: new  Abstract: Large language models (LLMs) answer many diet and health questions but can state unsupported claims and cite sources that do not support them. We present SHAKTI (Smart Health Advisor for Knowledge-based Tailored Interventions), a conversational assistant that answers only from retrieved scholarly papers and exact food-composition data, and evaluate it with retrieval, generation, grounding, efficiency and contextawareness metrics. We indexed 484,368,086 OpenAlex works, searched exactly on a GPU for evaluation (10,065 NFCorpus, BioASQ, SciFact and PubMedQA questions) and through a compressed 230-million-abstract index in deployment. On 1,000 stratified questions, we compared single-shot, multi-hop, corrective and deep-research retrieval-augmented generation (RAG) with Claude Sonnet. SHAKTI adds conversational query rewriting, USDA FoodData Central lookups, source-link checks and online verification of quotes and numbers. It was tested in 3
    
[^67]: 关系如何塑造嵌入空间结构

    How Relations Impose Embedding-Space Structure

    [https://arxiv.org/abs/2611.29986](https://arxiv.org/abs/2611.29986)

    本文提出一种无需学习参数的简单方法，通过比较不同领域间的内部结构排列，发现语言模型的表示被一致地排列以编码复杂的关系相似性，表明表示几何结构远比以往认为的更加丰富。

    

    众所周知，语言模型会在其表示的几何结构中编码简单关系，例如同义关系。然而尚不清楚它们是否也编码复杂的关系模式，例如驱动类比推理的那些模式。我们提出了一种简单的方法，无需学习任何参数，基于比较不同领域之间的内部结构排列来对此进行测试。使用该方法，我们在多个模型中发现表示被一致地排列以编码复杂的关系相似性。模型将对齐相距遥远的表示云团，使得意义的各个方面在简单映射下易于比较。我们的工作将表示空间的机制性解释扩展到了可能彼此仅有微弱联系的表示之间的相对排列，表明表示几何结构远比之前认为的更加丰富。代码和数据：https://github.com/lcowley7/embedded_relationships

    arXiv:2611.29986v1 Announce Type: new  Abstract: Language models are known to encode simple relations, such as synonymy, in the geometry of their representations. It is unclear whether they also encode complex patterns of relationships such as those that drive analogy. We present a simple method, with no learned parameters, based on comparing the internal structural arrangements between disparate domains, to test this. Using our method we find over multiple models that representations are arranged consistently to encode complex relationship similarities. Models align distant representation clouds so that facets of meaning are easily comparable under simple mappings. Our work extends mechanistic accounts of representation space to the relative arrangement of representations that may be only weakly related to one another, showing that representation geometry is far richer than previously thought. Code and data: https://github.com/lcowley7/embedded_relationships
    
[^68]: 语言模型在问答任务中即使面临确定的分数损失仍会寻求帮助

    Language models seek help despite guaranteed score losses in question answering

    [https://arxiv.org/abs/2611.29981](https://arxiv.org/abs/2611.29981)

    该研究系统量化了语言模型在成本-收益权衡下的求助决策，发现即使求助必然导致分数损失，各模型仍会咨询专家，且微调可将决策遗憾减少59%至87%，外部路由则能进一步提升答题得分。

    

    在多项选择题问答任务中，即使处于最高的等待成本，所有7个受测的开源权重语言模型仍会咨询专家，即便专家给出正确答案所得的分数低于独立回答错误时的得分。汇总专家准确率与固定收费的各种组合，各模型的求助率介于20.8%到94.1%之间。我们在54种专家准确率与咨询成本组合下测试了2,291道题目，将观察到的决策与基于独立测量并校准的模型置信度所构建的期望得分基线进行比较。结果显示，在期望咨询得分相同的条件下，求助率仍随条件而变化。微调使训练范围内的保留题目上的决策遗憾减少59%至87%，但在匹配的期望咨询得分下增加了不一致性。在另一个独立任务中，在候选答案保持不变的情况下，外部路由使每题得分提高0.180至0.273分。

    arXiv:2611.29981v1 Announce Type: new  Abstract: At the highest waiting cost in a multiple-choice question-answering task, all 7 tested open-weight language models still consult an expert even though a correct expert answer earns fewer points than an incorrect independent answer. Pooling over expert accuracy and fixed-fee combinations gives help-seeking rates of 20.8% to 94.1% across models. We test 2,291 questions across 54 combinations of expert accuracy and consultation costs, comparing observed decisions with an expected-score baseline based on independently measured and calibrated model confidence. Help-seeking rates vary across conditions with the same expected consultation score. Fine-tuning reduces decision regret by 59% to 87% on held-out questions within the training range but increases inconsistency at matched expected consultation scores. In a separate task, external routing improves scores by 0.180 to 0.273 points per question with candidate answers held fixed.
    
[^69]: 当1+1≠2：大语言模型民用时间推理的基准测试

    When 1+1 $\neq$ 2: Benchmarking Civil-Time Reasoning in Large Language Models

    [https://arxiv.org/abs/2611.29972](https://arxiv.org/abs/2611.29972)

    该论文提出CivilTime-Bench基准，系统评估大语言模型在夏令时转换等非线性民用时间（日历）推理上的能力，发现即使借助契约感知提示、少样本学习或微调，模型在真实场景中的准确率与泛化能力仍存在显著局限。

    

    日历算术是非线性的。当凌晨1:59加1分钟发生在夏令时春季时钟拨快期间时，结果是3:00 AM而非2:00 AM，因为该小时被跳过了。我们提出了CivilTime-Bench，一个民用时间推理基准，涵盖夏令时转换、非整小时偏移、日界线跨越和ISO周边界。在严格的输出契约评估下，基于60,011个训练样本和10,500个评估样本，我们对七个大语言模型（六个指令微调模型和Llama-3.1-8B基础模型）进行了基准测试，发现：零样本精确匹配率在不同模型之间差异显著；契约感知提示通常能改善格式合规性，但带来的准确率提升并不一致；少样本上下文学习的改进因模型而异；微调在合成评估上达到74.6–86.6%的精确匹配率，但在两个数据集上均有有效结果的模型在HumanTime-500上仅达到36.6–45.2%，凸显了泛化能力的重大局限。我们公开发布……（原文摘要末尾被截断）

    arXiv:2611.29972v1 Announce Type: cross  Abstract: Calendar arithmetic is non-linear. When 1:59 AM + 1 minute occurs during daylight saving spring-forward, the result is 3:00 AM, not 2:00 AM, as the hour is skipped. We introduce CivilTime-Bench, a benchmark for civil-time reasoning covering daylight saving transitions, non-hour offsets, date-line crossings, and ISO week boundaries. With 60,011 training and 10,500 evaluation examples under strict output-contract evaluation, we benchmark seven LLMs (six instruction-tuned models and Llama-3.1-8B base) and find: zero-shot exact match varies substantially across models, contract-aware prompting often improves format compliance but yields inconsistent accuracy gains, few-shot in-context learning provides model-dependent improvements, and fine-tuning achieves 74.6--86.6% EM on synthetic evaluation but only 36.6--45.2% on HumanTime-500 for models with valid results on both splits, highlighting substantial limitations in generalization. We rele
    
[^70]: 追踪自适应大语言模型红队测试中攻击者多样性的缩减

    Tracing Attacker Diversity Reduction in Adaptive LLM Red Teaming

    [https://arxiv.org/abs/2611.29960](https://arxiv.org/abs/2611.29960)

    本研究系统定位了大语言模型自适应红队测试中攻击多样性缩减的发生环节与机制，发现对“赢家”攻击的奖励强化是收窄主因，冻结防御者会加速这一过程，而防御者更新只能延缓但无法阻止多样性衰减。

    

    强化学习可以提高自动化红队测试的成功率，但同时也会使攻击向量收窄。本研究探讨在攻击者-防御者-裁判学习循环中，多样性缩减从何处开始，以及哪些机制导致攻击范围收窄。研究在三个实验组（共同训练、冻结防御者、减半学习率）中进行了二十四轮迭代，测试了六种机制（防御者自适应、成功优势、奖励强化、行数上限、更新规模和度量方式）。结果表明，冻结的防御者会更早降低攻击多样性保留率（ADR），在第12轮迭代时为0.496，而共同训练组为0.891。防御者的更新会延缓但不能阻止多样性缩减。成功攻击（“赢家”）的攻击多样性比生成的全部攻击集更窄。与以往赢家最接近的攻击比与以往失败最接近的攻击成功率高10-15个百分点。这种“赢家邻近优势”在防御者更新后依然存在。

    arXiv:2611.29960v1 Announce Type: cross  Abstract: Reinforcement learning can increase the success of automated red teaming but it also narrows the attack vector. This study asks where in the attacker-defender-judge learning loop does the narrowing begin and which mechanisms lead to a narrowing attack. Six mechanisms (defender adaptation, success edge, reward reinforcement, row cap, update size, and measurement) are tested across three experiment arms (co-trained, frozen defender, half learning rate) for twenty-four iterations. A frozen defender narrows attack diversity retention (ADR) earlier 0.496 relative to co-trained 0.891 at iteration twelve. Defender updates delay narrowing but dont prevent it. Successful attacks ("winners") have narrower attack diversity than the full set of generated attacks. Attacks closest to previous winners succeed 10-15 percentage points more often than those closest to previous failures. This winner-neighbor advantage holds after a defender update and fo
    
[^71]: Yggdrasil（世界树）：用于状态跟踪与快速CPU推理的树路由循环专家

    Yggdrasil: Tree-Routed Recurrent Experts for State Tracking and Fast CPU Inference

    [https://arxiv.org/abs/2611.29954](https://arxiv.org/abs/2611.29954)

    Yggdrasil通过树路由的循环专家库将模型容量与每步计算解耦，既能跟踪离散状态，又能在CPU上实现快速推理。

    

    可并行训练的循环结构（如Mamba、线性注意力、DeltaNet）能够提供恒定代价的流式推理，但它们存在两个局限：1）难以跟踪离散状态；2）在每一步都需要读取所有参数，使模型容量直接与CPU延迟绑定。我们提出了Yggdrasil，这是一种循环架构，通过在庞大的树路由循环专家库中查找每一层的转移函数，将模型容量与每步计算解耦。每个路由器使用当前token以及下一层的状态来寻址其专家库。我们证明了这种跨层上下文是必要的——单一token索引的、具有有限个可达状态的非扩展转移层无法执行条件重置——同时，避免来自该层自身过去的反馈则保留了并行扫描训练。由于每一步只遍历一条树路径，且每层只读取单个专家，活跃权重保持在几百……

    arXiv:2611.29954v1 Announce Type: cross  Abstract: Parallel-trainable recurrences such as Mamba, linear attention, and DeltaNet offer constant-cost streaming inference, but they suffer from two limitations: 1) they struggle to track discrete state, and 2) they read every parameter at every step, tying model capacity directly to CPU latency. We introduce Yggdrasil, a recurrent architecture that decouples capacity from per-step compute by looking up each layer's transition in a large, tree-routed bank of recurrent experts. Each router addresses its bank using the current token and the state of the layer below. We prove that this cross-layer context is necessary---a single token-indexed layer of non-expanding transitions with finitely many reachable states cannot perform conditional resets---while avoiding feedback from the layer's own past preserves parallel-scan training. Because each step traverses one tree path and reads a single expert per layer, active weights stay within a few hund
    
[^72]: 答案并非状态：扩散语言模型中的轨迹级可纠正性

    The Answer Is Not the State: Trajectory-Level Corrigibility in Diffusion Language Models

    [https://arxiv.org/abs/2611.29937](https://arxiv.org/abs/2611.29937)

    扩散语言模型中仅重新打开最终答案不足以纠正决策——旧的推理过程会重建原有结论，必须同时重新打开推理过程和答案，决策才能跟随新证据。

    

    扩散语言模型在共享画布的多个位置同时生成token，使得未完成的响应看起来似乎可以被修改。我们发现了一个反例：在最终答案被重新打开后，先前生成的推理过程即使面对新近可用的证据，仍能重建旧的决策。我们通过比较仅重新打开答案与同时重新打开推理过程和答案，以及在配对轨迹之间交换推理过程，来分离这一效应。在一个包含150道题、来源不相交的基准上，同时重新打开这两个区域可以使决策跟随当前存在的证据——无论是正确的还是被故意污染的——该现象在Dream、LLaDA和iLLaDA上均成立。在一个新的包含100道题的自由形式简答评估中，同样的效应在LLaDA和iLLaDA中依然显著，但在Dream中可以忽略不计。连贯的移植推理过程会引导重建后的决策，而token级归因方法虽然能改善应重新打开哪些位置的选择，却无法产生紧凑的修复。可纠正性……

    arXiv:2611.29937v1 Announce Type: new  Abstract: Diffusion language models generate tokens at many positions on a shared canvas, making unfinished responses appear revisable. We find a counterexample: after the final answer is reopened, previously generated rationale can reconstruct the old decision despite newly available evidence. We isolate this effect by comparing reopening the answer alone with reopening both rationale and answer, and by exchanging rationales between paired trajectories. On a 150-question source-disjoint benchmark, reopening both regions makes decisions follow the evidence currently present--whether correct or deliberately corrupted--across Dream, LLaDA, and iLLaDA. On a new 100-question free-form short-answer evaluation, the same effect remains strong in LLaDA and iLLaDA but is negligible in Dream. Coherent transplanted rationales steer rebuilt decisions, while token-level attribution improves which positions to reopen without producing a compact repair. Corrigib
    
[^73]: SAEPO：基于稀疏自编码器引导的策略优化信用分配方法

    SAEPO: Sparse-Autoencoder-Guided Credit Assignment for Policy Optimization

    [https://arxiv.org/abs/2611.29936](https://arxiv.org/abs/2611.29936)

    SAEPO利用预训练稀疏自编码器构建词元级语义表示，将结果监督下策略优化的信用分配从序列级细化到词元级，从而更高效地提升大语言模型的推理训练效果。

    

    具有可验证奖励的强化学习（RLVR）在提升大语言模型（LLM）推理能力方面展现出强劲性能，但其信用分配信号仍然较为粗糙。结果验证器通过判断完整回答是否正确来提供可扩展的监督，但它们并不能直接指示哪些中间推理词元对成功或失败有所贡献。因此，策略优化的效率可能较低，尤其是在直接从稀疏的结果反馈中训练基础模型时。本文提出了SAEPO，一种用于结果监督策略优化的、由稀疏自编码器引导的词元级信用分配方法。SAEPO不仅仅将正确性视为 rollout 级（序列级）信号，而是利用预训练的稀疏自编码器（SAE）在推理轨迹中构建词元级语义表示；它基于SAE概念对每个词元进行评分（例如，惩罚错误的“最终……（原文摘要在此处被截断）

    arXiv:2611.29936v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated strong performance in improving large language model (LLM) reasoning, yet its credit-assignment signal remains coarse. Outcome verifiers provide scalable supervision by judging whether a completed response is correct, but they do not directly indicate which intermediate reasoning tokens contribute to success or failure. As a result, policy optimization can be inefficient, especially when training base models directly from sparse outcome feedback. In this paper, we introduce SAEPO, a Sparse AutoEncoder-guided token-level credit assignment method for outcome-supervised Policy Optimization. Rather than treating correctness only as a rollout-level (sequence-level) signal, SAEPO uses a pretrained sparse autoencoder (SAE) to build token-level semantic representations within reasoning trajectories; it scores each token based on the SAE concepts (e.g., penalizing wrong "final
    
[^74]: 大语言模型幻觉的免训练干预方法

    Training-Free Interventions for Hallucination in Large Language Models

    [https://arxiv.org/abs/2611.29909](https://arxiv.org/abs/2611.29909)

    本文系统综述了116篇关于大语言模型推理阶段免训练幻觉干预的研究，按证据来源、模型访问需求、准备成本和适用任务对各方法族进行分类比较，并强调忠实性、事实正确性、信息量与答案覆盖度需分别评测。

    

    一个冻结的语言模型给出错误答案的原因可能是：某个事实不可用、可靠的文本段落被忽略，或者缺乏依据的断言在生成过程中幸存下来。我们回顾了116篇关于推理阶段幻觉干预及其理论基础的研究。我们梳理了这些方法从检索和可控生成，发展到作用于输入证据、token分数、内部表示、已生成的断言或是否回答的决策的过程。对于每个方法族，我们比较了其证据来源、所需的模型访问权限、准备成本以及可以在哪类任务上验证纠正效果。对于激活引导方法，我们区分了估计子空间与应用依赖状态的编辑，并区分了针对提示词的干预与针对已生成前缀的干预。已发表的对比结果表明，对来源的忠实性、事实正确性、信息量和答案覆盖度必须分别进行测量。该方法在时间敏感的问答、文档……等场景中的应用（摘要原文在此处截断）。

    arXiv:2611.29909v1 Announce Type: new  Abstract: A frozen language model can give a wrong answer because a fact is unavailable, a reliable passage is ignored, or an unsupported claim survives generation. We review 116 works on inference-time hallucination intervention and its foundations. We trace the development from retrieval and controllable generation to methods that act on input evidence, token scores, internal representations, completed claims, or the decision to answer. For each family we compare its source of evidence, model access, preparation cost, and the task on which a correction can be established. For activation steering, we distinguish estimating a subspace from applying a state-dependent edit and separate interventions on the prompt from those on the generated prefix. Published comparisons show why faithfulness to a source, factual correctness, informativeness, and answer coverage must be measured separately. Applications in time-sensitive question answering, document 
    
[^75]: Escriba-OCR：一个用于葡萄牙语手写作文错误保留转录的10亿参数视觉语言模型

    Escriba-OCR: A 1B Vision-Language Model for Error-Preserving Transcription of Handwritten Portuguese Essays

    [https://arxiv.org/abs/2611.29874](https://arxiv.org/abs/2611.29874)

    提出了Escriba-OCR——一个10亿参数的视觉语言模型，通过在647页手写作文上微调，实现了葡萄牙语手写作文的错误保留转录，能够精确保留拼写错误、编辑标记和行结构，克服了现有开源模型倾向“自动纠错”的缺陷。

    

    反馈工具和学习者写作研究需要能够精确保留学生原文的手写作文转录：包括拼写错误、换行、划掉的词以及插入的内容。视觉语言模型（VLM）虽然能很好地识别手写文字，但也倾向于将原文改写为正确的语言。我们在BRESSAY数据集（包含1,000篇手写的巴西葡萄牙语作文）上对这一现象进行了测量，该基准测试在字符错误率（CER）的基础上增加了错误保留率、虚构错误、编辑标记的精确率与召回率以及行结构等指标，并使用配对自助法置信区间。我们将研究限定于学校可在自有硬件上运行的开源权重模型。在22个开源基线模型中，表现最佳的Qwen3-VL 235B-A22B在199个测试页面上达到了5.6%的CER；但没有一个模型能可靠地重现编辑标记，且每个VLM都会将39%-78%的葡萄牙语词典外词汇替换为词典中的词。我们在647个训练页面上对LightOnOCR-3-1B进行了微调……（原文摘要在此处截断）

    arXiv:2611.29874v1 Announce Type: new  Abstract: Feedback tools and research on learner writing need transcriptions of handwritten essays that keep exactly what the student wrote: misspellings, line breaks, struck-through words and insertions. Vision-language models (VLMs) read handwriting well, but they also tend to rewrite it into correct language. We measure this on BRESSAY, 1,000 handwritten Brazilian Portuguese essays, with a benchmark that adds error preservation, invented errors, edit-markup precision and recall and line structure to the character error rate (CER) and uses paired bootstrap confidence intervals. We restrict the study to open-weight models that schools can run on their own hardware. Of 22 open baselines, the best, Qwen3-VL 235B-A22B, reaches 5.6% CER on the 199 test pages; none reproduces the edit markup reliably, and every VLM replaces 39-78% of the words absent from a Portuguese dictionary with dictionary words. We fine-tune LightOnOCR-3-1B on the 647 training p
    
[^76]: GlobalHRBench：改变提问的语言会改变法律答案吗？

    GlobalHRBench: Does Changing the Question Language Change the Legal Answer?

    [https://arxiv.org/abs/2611.29852](https://arxiv.org/abs/2611.29852)

    GlobalHRBench 是一个在八个司法管辖区和八种语言下保持事实与适用法律一致、包含 240 道劳动法问题的评测基准，评估表明大模型的法律答题准确率会随提问语言而变化（最佳模型严格通过率 76.2%，跨语言相对下降 5.8%）。

    

    arXiv:2611.29852v1 公告类型：新。摘要：在跨国界部署大语言模型（LLM）智能体时，语言至关重要，尤其是在法律领域，因为所适用的规则必须在各种工作语言下保持一致。那么，改变法律问题的语言是否会改变答案的准确性？我们推出了 GlobalHRBench，该基准包含 240 道劳动法问题，覆盖八个司法管辖区和八种语言，同时保持事实、适用法律和所需结论不变。我们在全部 1,920 个提示词上对九个模型进行了独立、闭卷、单答案的试验评估，评分维度包括法律正确性、完整性和答案语言合规性。GPT-6.1 Sol 以 76.2% 的严格通过率领跑，其从最佳语言到最差语言的相对下降仅为 5.8%，而 DeepSeek V4.1 Flash 的通过率为 42.2%。Kimi K3.1 Pro 和 Claude Sonnet 5.5 在英语与中文之间的表现排名出现反转。一项针对四个模型、16 组配对问题家族的研究，将闭卷答案与提供法律依据证据、添加跨司法管辖区干扰项以及删除法律依据段落后的表现进行了对比。法律依据证据……（原文摘要至此截断）

    arXiv:2611.29852v1 Announce Type: new  Abstract: Language matters when deploying LLM agents across borders, especially in law, where the governing rules must hold across working languages. Does changing the language of a legal question change answer accuracy? We introduce GlobalHRBench, pairing 240 employment-law problems across eight jurisdictions and eight languages while fixing facts, applicable law and required conclusions. We evaluate nine models on all 1,920 prompts in independent, closed-book, single-answer trials, scoring legal correctness, completeness and answer-language compliance. GPT-6.1 Sol leads at 76.2% strict passes, with a 5.8% relative best-to-worst language decline, compared with 42.2% for DeepSeek V4.1 Flash. Kimi K3.1 Pro and Claude Sonnet 5.5 reverse order between English and Chinese. A four-model study on 16 paired families compares closed-book answers with governing evidence, added cross-jurisdiction distractors and removal of a governing passage. Governing evi
    
[^77]: 时间残差：将潜在反馈快速有效地加装到预训练语言模型中

    The Temporal Residual: Fast and Effective Retrofitting of Latent Feedback into Pretrained Language Models

    [https://arxiv.org/abs/2611.29843](https://arxiv.org/abs/2611.29843)

    提出“时间残差”结构——一条仅需2,752个参数的第二残差流，可在预训练后直接加装到语言模型中，实现跨token的隐藏状态潜在反馈传递，显著提升数学推理性能并产生更简短的解答。

    

    Transformer语言模型为每个token计算深层隐藏状态，但仅将其用于采样当前token。现有的潜在反馈设计将隐藏状态跨token反馈传递，但这些设计大多需要在预训练期间构建，在预训练模型上添加时收益甚微。我们提出了时间残差，这是一条第二残差流，每个子层都可以对其进行读写，并将其从每个token传递到下一个token。它仅为一个42亿参数的模型增加了2,752个参数，并且完全从预训练网络出发。通过50亿token的训练将其加装到Qwen3.5-4B-Base上后，它在AIME 2024、AIME 2025和AMC 2023上分别比使用相同数据训练的标准模型高出7分、13分和5分，同时能写出更简短的解答，并在相同训练计算量下保持领先。使用独立的残差流至关重要：将状态馈入现有残差流的先前设计在加装时最多只能恢复40%的收益。验证损失可能会产生误导：……

    arXiv:2611.29843v1 Announce Type: new  Abstract: A transformer language model computes a deep hidden state for every token but uses it only to sample that token. Existing latent-feedback designs, which feed hidden states back across tokens, are mostly built in during pretraining and gain little when added to a pretrained model. We introduce the temporal residual, a second residual stream that every sublayer reads and writes and that is carried from each token to the next. It adds 2,752 parameters to a 4.2B-parameter model and starts exactly at the pretrained network. Retrofitted onto Qwen3.5-4B-Base with 5B tokens of training, it beats a standard model trained on the same data by 7, 13 and 5 points on AIME 2024, AIME 2025 and AMC 2023, writes shorter solutions, and stays ahead at matched training compute. The separate stream matters: prior designs that feed state into the existing stream recover at most 40% of the gain when retrofitted. Validation loss can mislead: the recipe with the 
    
[^78]: ComplexIF：一个具有挑战性的多轮、多约束指令遵循基准

    ComplexIF: A Challenging Multi-Turn, Multi-Constraint Instruction-Following Benchmark

    [https://arxiv.org/abs/2611.29775](https://arxiv.org/abs/2611.29775)

    ComplexIF是一个自动化生成的多轮、多约束指令遵循基准，通过程序验证的格式约束和LLM裁判评估的语义指令两个互补子集，在贴近真实使用场景的前提下为大模型提供更具挑战性的测试。

    

    随着前沿大语言模型在现有指令遵循基准上趋于饱和，要衡量进一步的进展，就需要更难但仍忠实于真实使用场景的测试。在实践中，用户常常将多个约束组合在一起，在对话的不同轮次中陆续提出这些约束，并期望助手的回复能够同时满足所有约束。我们提出了ComplexIF，一个针对这一场景的具有挑战性的基准，并配套了生成该基准的自动化流水线。该流水线将种子请求扩展为多轮对话，把约束编织进对话的各轮次中，并只保留强大模型能够解决的任务，因此可以通过增加约束或轮次来提升难度，同时保证每个任务都是可解的。ComplexIF的两个子集分别测试指令遵循的互补方面：一个可验证子集包含格式约束，由程序进行检查；另一个评分准则子集包含语义指令（例如语气或目标受众），由LLM裁判进行评估。在对14个专有与开源模型的评估中……（原文在此处截断）

    arXiv:2611.29775v1 Announce Type: new  Abstract: As frontier LLMs saturate existing instruction-following benchmarks, measuring further progress requires harder tests that remain faithful to real use. In practice, users often combine several constraints, add them at different turns of a conversation, and expect all of them to hold in the assistant's response. We introduce ComplexIF, a challenging benchmark for this setting, together with the automatic pipeline that generates it. The pipeline expands seed requests into multi-turn conversations, weaves constraints into their turns, and keeps only tasks that a strong model can solve, so difficulty can be raised by adding constraints or turns while every task stays solvable. ComplexIF's two splits test complementary sides of instruction following: a verifiable split of format constraints, checked by programs, and a rubric split of semantic instructions, such as a tone or an audience, checked with an LLM judge. Across 14 proprietary and ope
    
[^79]: 桑切斯、费霍还是阿巴斯卡尔？LLM角色画像阅读西班牙“11·29”大选三大政党领导人X平台帖子时的发言人归因与意识形态一致性

    S\'anchez, Feij\'oo, or Abascal? Speaker Attribution and Ideological Congruence in LLM Personas Reading Posts on X of the Leaders of the Three Main Parties of the 29N General Election in Spain

    [https://arxiv.org/abs/2611.29765](https://arxiv.org/abs/2611.29765)

    本文首次通过大规模发言人归因实验验证了LLM角色画像会对发言者身份（而不仅是文本内容本身）作出反应，发现点名西班牙三大政党领导人会系统性降低对相同文本的支持度，且阿巴斯卡尔带来的负面效应约为桑切斯和费霍的两倍。

    

    来源线索是政治传播的核心要素，然而以大语言模型（LLM）角色画像作为合成受众的接受研究，尚未检验这些画像是否会对“谁在说话”作出反应，而不仅仅是对“说了什么”作出反应。着眼于2026年11月29日的西班牙大选，我们开展了一项发言人归因实验，使用PSOE、PP和VOX三党领导人在四个主题（经济与就业、家庭开支、移民与腐败）上的1,075条匿名化X帖子，以及791个基于价值观的、由LLM生成的独特角色画像。每条帖子被分配给300个按主题筛选的角色画像档案——其中150个基于英格尔哈特-韦尔策尔文化地图，150个基于道德基础理论——在四种条件下进行测试：无作者、桑切斯、费霍或阿巴斯卡尔，共产生322,470组完整的“角色画像-帖子”四元组。研究结果表明，点名任何领导人都降低了对相同文本的支持度，其中阿巴斯卡尔带来的惩罚效应几乎是桑切斯和费霍的两倍。（注：原文摘要在此处截断）

    arXiv:2611.29765v1 Announce Type: new  Abstract: Source cues are central to political communication, yet reception studies with large language model (LLM) personas as synthetic audiences have not tested whether personas respond to who speaks and not only to what is said. With Spain's general election of 29 November 2026 in mind, we run a speaker-attribution experiment using 1,075 anonymised X posts by the leaders of PSOE, PP and VOX on four topics (the economy and employment, household costs, immigration and corruption) and 791 distinct value-grounded, LLM-generated personas. Each post was assigned to 300 topic-selected persona profiles, 150 grounded in the Inglehart-Welzel cultural map and 150 in Moral Foundations Theory, under four conditions: no author, S\'anchez, Feij\'oo or Abascal, yielding 322,470 complete persona-post quartets. Naming any leader lowered support for identical text, with the penalty for Abascal nearly twice as large as those for S\'anchez and Feij\'oo. The penalt
    
[^80]: 精炼而非重造：RASA-Bench——基于源文本的印地语散文润色基准

    Refine, Don't Reinvent: RASA-Bench for Source-Grounded Hindi Prose Refinement

    [https://arxiv.org/abs/2611.29750](https://arxiv.org/abs/2611.29750)

    RASA-Bench 提出了一个针对印地语散文润色任务的新基准，通过忠实性门控加 Bradley-Terry 编辑偏好排名的四层评估协议，衡量模型在不偏离源文本前提下改进单薄 AI 草稿的能力，发现忠实性测试在短篇小说与有声小说等不同文体上的淘汰率差异显著。

    

    对于印地语创意散文的创作者而言，起草很容易，编辑却不然：机器撰写的剧集、故事或文章读起来内容单薄、平淡无奇，夹杂着翻译腔，有时还会悄然改变源文本的内容。RASA-Bench 针对三种印地语散文形式中的这一编辑任务，衡量模型对内容单薄的 AI 草稿进行润色的能力——这是一项不同于从零开始写作的任务。该基准涵盖 2,050 份草稿和八个润色模型（共 16,400 次重写），采用四层评估协议：重写文本必须首先通过与源文本的忠实性检查，随后依据整体编辑偏好，通过逐形式拟合的 Bradley-Terry 模型（其中心化系数在等权平均前经过标准化）对通过者进行排名。事实证明，保持忠实是一项决定性考验，且在不同文体上表现各异：门控失败率从短篇小说的 9.1% 到有声小说的 29.5% 不等，而使重写被淘汰的偏移……

    arXiv:2611.29750v1 Announce Type: cross  Abstract: For a producer of Hindi creative prose, drafting is cheap and editing is not: a machine-written episode, story, or essay reads as under-written and flat, laced with translationese, and it sometimes quietly alters what the source says. RASA-Bench targets that editing task across three Hindi prose forms, measuring how well a model refines an under-written AI draft, a task distinct from writing from scratch. Across 2,050 drafts and eight refiners (16,400 rewrites), a four-layer protocol credits a rewrite only after it passes a faithfulness check against its own source, then ranks the survivors by holistic editorial preference through per-form Bradley-Terry fits whose centered coefficients are standardized before equal-weight averaging. Staying faithful proves to be a decisive test that behaves differently by form: gate-failure rates range from 9.1% on short stories to 29.5% on audio fiction, and the drift that disqualifies a rewrite shift
    
[^81]: ReqSeek：基于Transformer的自动需求识别方法，用于构建大规模需求数据集

    ReqSeek: Transformer-Based Automatic Requirements Identification for Building Large Requirements Datasets

    [https://arxiv.org/abs/2611.29722](https://arxiv.org/abs/2611.29722)

    本文提出了基于Transformer、领域无关的自动需求识别模型ReqSeek，并构建了包含23个异构SRS的ARID数据集，以RoBERTa为基础模型达到95%的精确率与召回率，性能可与人类媲美且耗时更少。

    

    大规模、多样化且高质量的数据集对于推动需求工程领域的人工智能发展至关重要，然而从软件需求规格说明（SRS）中自动识别需求仍是一项挑战。现有的自动需求识别（ARI）方法在异构SRS文档上对未见文档的泛化能力有限。本工作提出了ReqSeek，一个基于Transformer、与领域无关的自动需求识别模型，用于识别SRS文档中符合ISO 29148风格的需求，以支持数据集构建。我们策划了自动需求识别数据集（ARID），包含来自23个异构SRS的2,396个训练实例，并微调了四种Transformer架构。RoBERTa被选为ReqSeek的基础模型，在留出评估集上达到了95%的精确率、召回率和F1值。探索性的人机对比基准测试表明，该模型的性能与人类相当，同时所需时间显著更少（摘要在此处截断）。

    arXiv:2611.29722v1 Announce Type: cross  Abstract: Large, diverse, and high-quality datasets are significant for advancing AI in requirements engineering, yet automated identification of requirements from Software Requirement Specifications (SRS) remains a challenge. Automatic Requirements Identification (ARI) methods demonstrate limited generalization to unseen documents across heterogeneous SRSs. This work presents ReqSeek, a transformer-based, domain-agnostic ARI model for recognizing ISO 29148-style requirements in SRS documents, supporting dataset construction. We curated the Automatic Requirements Identification Dataset (ARID), comprising 2,396 training instances from 23 heterogeneous SRSs, and fine-tuned four transformer architectures. RoBERTa was selected as the base model for ReqSeek and achieved 95% precision, recall, F1, and F0.89 on the held-out evaluation set. Exploratory human-model benchmarking showed performance comparable to humans while requiring substantially less pr
    
[^82]: 循环Transformer中的循环次数调度：一项实证研究

    Loop-Count Schedules in Looped Transformers: An Empirical Study

    [https://arxiv.org/abs/2611.29704](https://arxiv.org/abs/2611.29704)

    循环Transformer中循环次数采样范围的下界决定了重复计算的性质：若范围包含R=1，门控坍缩至0.1-0.5、后续循环仅作小修正；若从R=2开始，门控保持在0.6-0.9、每次循环均保持完整层的强度。

    

    循环Transformer通过共享权重将一个层块重复应用多次，以重复计算换取参数量的减少。我们研究了训练期间循环迭代次数R的调度方式如何决定每次额外的循环是充当一个完整的层还是一个小幅修正。我们使用一组受控的模型阶梯进行实验：46M至110M参数的模型在20亿（2B）token上训练，每次循环配备可学习的残差门控，并采用三种调度方案：R固定为2、R从{1,...,6}中均匀采样、以及R从{2,3,4}中采样。我们发现采样范围的下界是决定性的选择。如果采样范围包含R=1，使得某些训练步骤只运行该层块一次并基于该单次循环计算损失，那么门控值会坍缩至0.1-0.5，每次额外的循环都变成对第一次循环的小幅修正；如果采样范围从R=2开始，使得每个训练步骤至少运行该层块两次，那么门控值保持在0.6-0.9，每次循环都保持完整层的强度。

    arXiv:2611.29704v1 Announce Type: cross  Abstract: Looped transformers apply one block of layers several times with shared weights, trading parameters for repeated computation. We study how the schedule of the loop iteration count R during training determines whether each extra pass acts as a full layer or as a small correction, using a controlled ladder of 46M-110M-parameter models trained on 2B tokens with learned residual gates on every loop pass and three schedules: R fixed to 2, R sampled uniformly from {1,...,6}, and R sampled from {2,3,4}. We find that the lower bound of the sampling range is the decisive choice. If the sampled range includes R=1, so that some training steps run the block only once and compute the loss on that single pass, the gates collapse to 0.1-0.5 and each additional pass becomes a small correction to the first; if the range starts at R=2, so that every training step runs the block at least twice, the gates stay at 0.6-0.9 and each pass keeps the strength o
    
[^83]: 事实核查错误产生于何处？在基于证据的验证中分离检索、搜索控制与判断

    Where Do Fact-Checking Errors Arise? Separating Retrieval, Search Control and Judgement in Evidence-Grounded Verification

    [https://arxiv.org/abs/2611.29697](https://arxiv.org/abs/2611.29697)

    该论文提出一套分离检索、搜索控制、判断与裁决验证的诊断协议，发现事实核查失败的主要根源不在证据检索（单次混合检索即可覆盖绝大多数声明的完整证据），而在于证据的后续使用环节——判断器弃权与验证规则推翻了本可正确作出的裁决。

    

    基于证据的事实核查器可能因找不到证据而失败，也可能因没有使用证据而失败；端到端的准确率无法区分这两种情况。我们提出了一套协议，追踪参考证据从存储库到最终裁决的完整路径，将检索、搜索控制、判断和裁决验证四个环节分离开来。我们使用一个开放权重的9B模型，在两种证据条件下评估了来自FEVER、FEVER 2.0和AVeriTeC的293条声明：一种是标注证据的小型存储库（证据发现较为容易），另一种是完整的源文档集合。在对照条件下，单次混合检索即可为228条含证据声明中的226条提供完整的参考证据，而自适应控制器和工具智能体花费了二到九倍的模型调用次数，却没有带来一致的性能提升。在34条被标记为缺乏证据的决定性声明中，有33条的证据包中实际包含完整证据：其中15条是判断器选择弃权，另外18条是验证规则推翻了决定性……

    arXiv:2611.29697v1 Announce Type: new  Abstract: An evidence-grounded fact-checker can fail because it does not find the evidence or because it does not use it; end-to-end accuracy does not tell the two apart. We present a protocol that follows the reference evidence from the store to the verdict and separates retrieval, search control, judgement and verdict validation. We assess 293 claims from FEVER, FEVER 2.0 and AVeriTeC with one open-weights 9B model under two evidence conditions: small stores of annotated evidence, where discovery is easy, and the full source collections. In the control condition a single hybrid retrieval pass delivers complete reference evidence for 226 of 228 evidence-bearing claims, and adaptive controllers and a tool agent spend two to nine times as many model calls without a consistent gain. Of 34 decisive claims labelled as lacking evidence, 33 have the complete evidence in the packet: in 15 the judge abstains, and in 18 validation rules overturn a decisive
    
[^84]: 语法稀疏性决定Transformer学习随机语言的方式

    Grammar Sparsity Sets How Transformers Learn Random Languages

    [https://arxiv.org/abs/2611.29692](https://arxiv.org/abs/2611.29692)

    语法稀疏性同时决定了Transformer学习随机语言的顺序和速度——稠密语法分层学习且经历随词表大小变宽的平台期（$t\propto V$ 到 $t\propto V^3$），而稀疏语法则一次性快速习得（$t\propto\sqrt V$）。

    

    我们研究了在随机语言模型上训练的Transformer的语言习得过程：随机语言模型是一种合成的分层语法，具有随机的产生式规则，其稀疏性由单一参数控制，且其条件熵可以被精确计算。在深度为二的语法上，我们发现稀疏性决定了学习的顺序和速度。稠密语法的 learned 一次一层：token 在 $t\propto V$ 步后可以从其兄弟节点预测，而只有在 $t\propto V^{3}$ 步后才能从完整上下文中预测，因此这两个学习阶段被一个随 $V$ 增大而变宽的平台期分隔开来。稀疏语法则是一次性全部学习，在 $t\propto\sqrt V$ 步内完成，其标度指数甚至比稠密语法的第一阶段还要小。指数 $1$ 和 $1/2$ 在深度为一的情形中就已经出现，我们对使用 Adam 训练的线性模型推导出了这两个指数。

    arXiv:2611.29692v1 Announce Type: new  Abstract: We study language acquisition by transformers trained on the Random Language Model: a synthetic hierarchical grammar with random production rules, whose sparsity is controlled by a single parameter and whose conditional entropies are known exactly. On depth-two grammars, we find that sparsity sets both the order and the speed of learning. Dense grammars are learnt one level at a time: a token is predicted from its sibling after $t\propto V$ steps and from the full context only after $t\propto V^{3}$, so the two stages are separated by a plateau that widens with $V$. Sparse grammars are learnt all at once, at $t\propto\sqrt V$, with a smaller exponent than even the first stage of a dense one. The exponents $1$ and $1/2$ already appear at depth one, where we derive them for a linear model trained with Adam.
    
[^85]: 更多角色或更大的模型能提升大语言模型的事实核查能力吗？一项基于固定证据的对照研究

    Do More Roles or Larger Models Improve LLM Fact-Checking? A Controlled Study on Fixed Evidence

    [https://arxiv.org/abs/2611.29687](https://arxiv.org/abs/2611.29687)

    这项对照研究发现，在固定证据下的事实核查中，更大的模型并不能带来更高准确率反而更常拒绝判断，而多角色“法庭”机制的表现与简单投票相当，说明计算资源的分配方式比模型规模更关键。

    

    事实核查系统可以通过多种方式在某个论断上投入更多计算：使用更大的模型、对同一模型多次询问，或采用一个多角色的“法庭”机制——两个角色提出带引文的研究发现，代码检查这些引文，一位看不到所提议关系的审计员对每一项分别评判，最后由代码得出结论或选择弃权。我们在完全相同的固定证据上对这些选项进行比较，且不使用检索：300个AVeriTeC论断，每个论断的标注问答条目与其他论断的条目混合在一起，由来自三个模型家族的四个开放权重模型担任评判者进行阅读，采用五票投票机制，以及六种模型到法庭角色的分配方案。结果显示：9B评判者在186个论断上判断正确，而同一家族的27B评判者仅在156个上正确；更大的评判者拒绝做判断的频率高得多，且在做出判断时准确率也并不更高。投票机制达到196个，而将9B评判者的结论替换为法庭证书后达到195个，两者成本相近；法庭……（原文摘要在此处截断）

    arXiv:2611.29687v1 Announce Type: cross  Abstract: A fact-checking system can spend more computation on a claim by using a larger model, by asking the same model several times, or by a multi-role Tribunal: two roles propose quoted findings, code checks the quotations, an auditor that does not see the proposed relation judges each one, and code derives the verdict or abstains. We compare these options on identical fixed evidence, without retrieval: 300 AVeriTeC claims, each with its annotated question--answer items mixed with items written for other claims, read by judges of four open-weights models from three families, a vote of five, and six assignments of models to the Tribunal's roles. The 9B judge was right on 186 claims and the 27B judge of the same family on 156; the larger judges declined to decide far more often and were no more accurate when they decided. The vote reached 196 and the 9B judge with its verdict replaced by Tribunal certificates 195, at a similar cost; the Tribun
    
[^86]: MERIT：面向长周期特定人设说服对话的经验引导记忆

    MERIT: Experience-Guided Memory for Long-Horizon Persona-Specific Persuasive Dialogue

    [https://arxiv.org/abs/2611.29679](https://arxiv.org/abs/2611.29679)

    该论文提出MERIT系统，通过结合图结构的说服专用记忆与每个人设的经验推理库，仅凭提示组合即可实现无需参数更新的跨会话自适应说服对话，并推出了PERSIST人设保险交互数据集。

    

    在反复出现的以目标为导向的场景中实现有效的长周期说服，不仅需要事实性记忆：还必须追踪用户的信任、谈判行为和战略接受度如何随互动而演变。现有的说服性对话系统大多是单会话、静态且单模态的，而跨会话记忆系统虽然具备时间感知能力，却仅保留事实内容，忽视了说服特有的动态变化。我们提出了MERIT（面向迭代式时间说服的记忆增强经验推理），它将基于图的四种说服特定记忆类型与每个人设专属的经验推理库相结合，记录哪些策略对每个用户有效及其原因。两者结合，仅通过提示组合即可实现自适应说服，无需任何参数更新。为支持这一场景，我们引入了PERSIST（基于人设的长期保险交互）数据集，包含60个精心策划的人设，涵盖……

    arXiv:2611.29679v1 Announce Type: new  Abstract: Effective long-horizon persuasion in recurring goal-oriented settings requires more than factual memory: it must track how a user's trust, negotiation behavior, and strategic receptiveness evolve across interactions. Existing persuasive dialogue systems are largely single-session, static, and unimodal, while cross-session memory systems, though temporally aware, retain only factual content and overlook persuasion-specific dynamics. We propose MERIT (Memory-augmented Experience-Rationale for Iterative Temporal persuasion), which pairs a graph-based memory of four persuasion-specific memory types with a per-persona experience rationale library that records which strategies succeed for each user and why. Together they enable adaptive persuasion through prompt composition alone, with no parameter updates. To support this setting, we introduce PERSIST (PERSona-driven InSurance Interactions over Time), a dataset of 60 curated personas spanning
    
[^87]: 稀疏注意力索引器的跳位蒸馏：短训练，长服务

    Position-Skipped Distillation of Sparse-Attention Indexers: Train Short, Serve Long

    [https://arxiv.org/abs/2611.29664](https://arxiv.org/abs/2611.29664)

    该论文提出跳位蒸馏方法，通过让索引器与冻结教师在位置 ID 含随机间隔的短序列上训练，发现索引器长上下文泛化失败主要源于 RoPE 位置编码而非内容匹配退化，从而实现“短训练、长服务”，以约 6 倍更少的训练成本即可匹配长序列蒸馏的索引器效果。

    

    Top-k 稀疏注意力依赖于一个索引器——一个轻量的逐层模块，它对缓存的 token 进行排序，使注意力只覆盖得分最高的 k 个 token。索引器通常是在完整服务长度下从稠密注意力蒸馏而来，因此每个训练步骤都要承担长上下文稠密注意力的计算开销。我们证明，当索引器在短序列上蒸馏时，其无法泛化到更长上下文的问题主要由旋转位置编码导致，而非内容匹配能力退化。我们提出跳位蒸馏方法：索引器与冻结的教师模型只处理位置 ID 中包含随机放置间隔的短序列，因此两者处理的长度都不会超过短训练长度。在 DeepSeek-V2-Lite 和 Qwen3-30B-A3B 上，用这种方法在 8K token 上蒸馏的索引器在 32K 以内与 32K 蒸馏的索引器困惑度差距约为 0.5 以内，并且在 DeepSeek-V2-Lite 上，以约 6 倍更少的训练成本在 64K 处达到与 64K 蒸馏索引器相当的效果。

    arXiv:2611.29664v1 Announce Type: new  Abstract: Top-k sparse attention relies on an indexer, a lightweight per-layer module that ranks cached tokens so attention covers only the k highest-scoring ones. Indexers are usually distilled from dense attention at the full serving length, so every training step pays for long-context dense attention. We show that when an indexer is distilled on short sequences, its failure to generalise to longer contexts is driven mainly by rotary positional encoding (RoPE), not by degraded content matching. We propose position-skipped distillation: the indexer and the frozen teacher see short sequences whose position IDs contain a randomly placed gap, so neither ever processes more than the short training length. On DeepSeek-V2-Lite and Qwen3-30B-A3B, an indexer distilled on 8K tokens this way matches a 32K-distilled indexer to within about 0.5 perplexity up to 32K, and on DeepSeek-V2-Lite matches a 64K-distilled indexer at 64K with roughly 6x less training 
    
[^88]: AI生成的音频描述有多可靠？一项多语言专家评估

    How Reliable is AI-Generated Audio Description? A Multilingual Expert Evaluation

    [https://arxiv.org/abs/2611.29658](https://arxiv.org/abs/2611.29658)

    该研究通过对意大利语和德语电影进行多语言专家评估，发现尽管GPT-4o优于基于Qwen的系统，AI生成的音频描述在错误率和质量评分上仍显著低于人类专业描述员的水平。

    

    音频描述（AD）使盲人或低视力（BLV）观众能够无障碍地访问视频等视觉媒体，但专业制作的高昂成本限制了其覆盖范围。得益于人工智能的最新进展，已有许多自动生成中间AD脚本和/或最终AD音频的方法被提出。我们对意大利语和德语电影的AI生成AD脚本进行了一项专家评估，将两个AI系统——基于GPT-4o和基于Qwen的流水线——与人工撰写的描述进行对比。专业音频描述员对描述质量进行评分，并使用涵盖内容、角色、流畅性和连贯性错误的详细错误分类体系对错误进行标注。结果表明，尽管GPT-4o的表现优于Qwen模型，但两个AI系统仍未能达到人类专业水平：生成的描述呈现出明显更高的错误率和更低的质量评分。基于错误标注，我们识别出两个……

    arXiv:2611.29658v1 Announce Type: new  Abstract: Audio description (AD) makes visual media, such as video, accessible to blind or low-vision (BLV) audiences, but the high cost of professional production limits coverage. Due to recent advances in AI, numerous approaches to automatically generate intermediate AD scripts and/or final AD audios have been proposed. We present an expert evaluation of AI-generated AD scripts for movies in Italian and German that compares two AI systems, a GPT-4o- and a Qwen-based pipeline, against human-authored descriptions. Professional audio describers rated description quality and annotated errors using a detailed error taxonomy that includes content, character, fluency and coherence errors. The results show that while GPT-4o outperforms the Qwen models, both AI systems still fall short of human professional quality: the generated descriptions show substantially higher error rates and lower quality ratings. Based on the error annotations, we identify two 
    
[^89]: 哪些漏洞会被发现？在强化学习训练前预测 LLM 裁判的奖励劫持

    Which Loopholes Get Found? Predicting Reward Hacking of LLM Judges Before RL Training

    [https://arxiv.org/abs/2611.29651](https://arxiv.org/abs/2611.29651)

    该论文提出“暴露度”指标（初始策略产生缺陷模式的倾向性 × 裁判假阳性率），能够在强化学习训练开始前预测 GRPO 训练中 LLM 裁判会被策略最先利用哪个漏洞。

    

    针对 LLM 裁判训练的策略可以学会利用其漏洞。静态探测显示，这类裁判会接受诸如单独一个 "{" 或 "Thought process:" 之类的退化答案，但无法判断策略会找出哪个缺陷。在 GRPO 中，第一个被利用的漏洞可以在训练之前通过每个模式的“暴露度”（exposure）来预测——暴露度是初始策略产生该模式的倾向性与裁判假阳性率（FPR）的乘积。为了因果性地检验暴露度这一指标，我们在拉丁方设计中以三种剂量将这些模式植入微调数据。在涉及七个裁判的 36 次 GRPO 运行中，在发生漏洞利用的全部 12 次运行里，首个被利用的模式都恰好是暴露度最高的那个，而仅在 6 次运行中具有最高的假阳性率。在其中一个模式被规则遮蔽的四次运行中，训练前记录的预测有三次成立。当仅植入两个模式时（占微调数据的 3% 至 3.8%，并特意选择使暴露度与仅依据假阳性率或仅依据倾向性得出的预测不一致的情形），暴露度正确指出了第一个被利用的模式

    arXiv:2611.29651v1 Announce Type: cross  Abstract: A policy trained against an LLM judge can learn to exploit its loopholes. Static probes show that such judges accept degenerate answers like a lone "{" or "Thought process:", but not which flaw a policy will find. In GRPO the first loophole to be exploited can be predicted before training from each pattern's exposure, the product of the starting policy's propensity to produce it and the judge's false-positive rate (FPR). To test exposure causally, we seed the patterns into fine-tuning at three doses in a Latin square. Across 36 GRPO runs with seven judges, the first exploited pattern had the highest exposure in all 12 runs with exploitation and the highest FPR in only 6. Predictions recorded before training held in three of four runs with one pattern masked by rule. When only two patterns were seeded, at 3% to 3.8% of the fine-tuning data and chosen so that exposure disagreed with FPR alone or propensity alone, exposure named the first
    
[^90]: 通过风格条件化改写降低学生文本的作者归属风险

    Reducing Authorship Attribution Risk in Student Texts Through Style-Conditioned Rewriting

    [https://arxiv.org/abs/2611.29642](https://arxiv.org/abs/2611.29642)

    该论文提出一种风格感知的改写方法，通过将内容、结构与风格解耦并结合聚类受限检索的差分隐私上下文学习（DP-ICL），在保留文本分析价值和下游任务性能的同时，有效降低学生文本的作者归属风险。

    

    由于学生隐私担忧和数据法规的限制，构建面向教育的NLP应用受到数据稀缺的制约。为解决这一问题，我们提出了一种风格感知的方法来改写学生论坛帖子，在降低可识别性的同时保留分析价值。我们的方法将内容、结构和风格作为独立的条件化通道进行隔离，并在风格聚类内的学生不相交示例分区上进行聚合，通过聚类受限检索扩展了差分隐私上下文学习（DP-ICL）。在三个教育NLP数据集上，我们的方法实现了高风格保真度，保留了下游任务性能，并相对于原始学生撰写文本降低了作者归属风险，包括在面对专门的作者归属模型时也表现有效。

    arXiv:2611.29642v1 Announce Type: new  Abstract: Building NLP applications for education is constrained by data scarcity due to student privacy concerns and data regulations. To address this, we propose a style-aware approach to rewriting student forum posts that retains analytical value while reducing identifiability. Our method isolates content, structure, and style as separate conditioning channels and aggregates over student-disjoint exemplar partitions within style clusters, extending Differentially Private In-context Learning (DP-ICL) with cluster-restricted retrieval. Across three educational NLP datasets, our approach achieves high style fidelity, preserves downstream task performance, and reduces authorship attribution risk relative to the original student-written texts, including against a dedicated authorship-attribution model.
    
[^91]: 多元一体、理论落地：面向中国少数民族语言的分层多任务基准

    Diversity in Unity, Theory in Practice: Hierarchical Multitask Benchmarks for Chinese Minority Languages

    [https://arxiv.org/abs/2611.29635](https://arxiv.org/abs/2611.29635)

    该论文提出CMiLBench——首个面向藏语、蒙古语和维吾尔语三种中国少数民族语言的分层多任务评估基准，涵盖5个难度级别17个任务共24,663个实例，通过评估14个大语言模型揭示了其在少数民族语言上理论能力与实际表现之间的差距。

    

    尽管大语言模型（LLMs）发展迅速，但其在统一国家框架内针对语言和文化多样的少数民族语言的表现仍未得到充分探索。我们提出了CMiLBench，这是一套分层多任务基准集合，旨在将“多元一体”的理论概念转化为对三种代表性中国少数民族语言——藏语、蒙古语和维吾尔语——的实用性评估。CMiLBench包含24,663个实例，涵盖5个难度级别和17个任务，横跨基础能力、文化特异性和安全对齐三大维度。我们采用现有数据集适配、少数民族知识构建以及高资源基准翻译的方法来构建CMiLBench。我们使用一个集成自动指标与LLM-as-a-Judge评分的混合框架，评估了14个最先进的商业和开源大语言模型。对比实验结果揭示了模型理论能力与实际应用之间的差距。

    arXiv:2611.29635v1 Announce Type: new  Abstract: Despite the rapid advancement of LLMs, their performance on linguistically and culturally diverse minority languages within a unified national context remains underexplored. We present CMiLBench, a collection of hierarchical multitask benchmarks designed to translate theoretical notions of "diversity in unity" into practical evaluation for three representative Chinese minority languages: Tibetan, Mongolian, and Uyghur. CMiLBench comprises 24,663 instances across 5 difficulty levels and 17 tasks spanning foundational ability, cultural specificity, and safety alignment. We adopt existing dataset adaptation, minority knowledge construction, and high-resource benchmark translation to construct CMiLBench. We assess 14 state-of-the-art commercial and open-source LLMs with a hybrid framework that integrates automatic metrics and LLM-as-a-Judge scoring. The comparative experimental results reveal the gap between theoretical capability and practi
    
[^92]: 使用情绪理论在大语言模型中实现可解释的情绪与心理健康推理

    Explainable Emotion and Mental Health Reasoning in Large Language Models Using Theories of Emotion

    [https://arxiv.org/abs/2611.29628](https://arxiv.org/abs/2611.29628)

    该论文提出了一种基于沙赫特-辛格情绪理论和认知评估理论的心理学思维链提示策略PsyCoT，使大语言模型在情绪分析和心理健康任务中实现可解释的推理。

    

    arXiv:2611.29628v1 公告类型：新论文 摘要：大语言模型（LLM）的发展使其在广泛的应用领域成为可能。然而，它们模拟类人行为和推理的能力仍不明确。虽然情绪检测是一项较为成熟的自然语言处理（NLP）任务，但现有方法主要侧重于将文本直接分类到预定义的情绪类别中，往往忽略了评估（appraisal）过程——即解释个体为何在特定情境中产生某种特定情绪的过程。在这项工作中，我们提出了基于心理学的思维链（PsyCoT）推理，这是一种以沙赫特-辛格（Schachter-Singer）情绪理论和认知评估理论为基础的理论驱动提示策略。我们提出并在多种主观NLP任务上评估了PsyCoT，包括情绪分析、情绪强度预测、压力分析和抑郁严重程度识别。我们的评估涵盖四个分类复杂度级别：二元分类、多分类、多标签分类和……

    arXiv:2611.29628v1 Announce Type: new  Abstract: Advancements in Large Language Models (LLMs) have enabled a wide range of applications. However, their ability to simulate human-like behavior and reasoning remains unclear. While emotion detection is a better-established Natural Language Processing (NLP) task, existing approaches primarily focus on directly classifying text into predefined emotion classes, often overlooking the appraisal process that explains why an individual experiences a particular emotion in a given situation. In this work, we propose Psychology-based Chain-of-Thought (PsyCoT) reasoning, a theory-driven prompting strategy grounded in the Schachter-Singer and Cognitive Appraisal theories of emotion. We propose and evaluate PsyCoT for subjective NLP tasks, including emotion analysis, emotion intensity prediction, stress analysis, and depression severity identification. Our evaluation covers four levels of classification complexity: binary, multiclass, multi-label, and
    
[^93]: KernelShift：面向不断演进硬件的内核优化基准测试

    KernelShift: Benchmarking Kernel Optimization Across Evolving Hardware

    [https://arxiv.org/abs/2611.29590](https://arxiv.org/abs/2611.29590)

    该论文提出了KernelShift基准测试流水线，通过从ncnn、llama.cpp等真实活跃仓库中构建99个内核优化任务，并让智能体在目标硬件反馈下优化代码，以评估编码智能体在不断演进的CPU架构扩展下编写高性能内核的能力。

    

    高效的AI计算依赖于随时间不断演进的硬件，这改变了实现高性能代码所需的策略。对于CPU内核而言，一个关键的演进因素是可用的架构扩展集合，这些扩展提供了专用的硬件指令，从而改变了优化内核所采用的技术。我们研究了编码智能体如何有效地利用这些新的、可能不熟悉的扩展来编写快速代码。我们提出了KernelShift，这是一个用于评估编码智能体在不断演进的CPU架构上进行内核优化能力的基准测试流水线和一组环境。其智能体辅助、人工验证的流水线可将代码仓库转换为基准测试实例，其环境让智能体能够根据目标硬件的反馈来优化代码。利用该流水线，我们从真实且处于活跃开发状态的仓库（ncnn、llama.cpp、Arm SIMD Loops和KleidiAI）中构建了99个内核，包括由此衍生的推理工作负载。

    arXiv:2611.29590v1 Announce Type: cross  Abstract: Efficient AI computation depends on hardware that evolves over time, shifting the strategies needed to implement performant code. For CPU kernels, a key evolving factor is the set of available architectural extensions, which expose specialized hardware instructions that change the techniques used to optimize kernels. We study how effectively coding agents use these new, potentially unfamiliar extensions to write fast code. We introduce KernelShift, a benchmarking pipeline and a set of environments for evaluating coding agents in kernel optimization across evolving CPU architectures. Its agent-assisted, human-verified pipeline turns code repositories into benchmark instances, and its environments let agents optimize code with feedback from the target hardware. Using this pipeline, we construct 99 kernels from real-world, actively developed repositories (ncnn, llama.cpp, Arm SIMD Loops, and KleidiAI), including inference workloads derive
    
[^94]: 单坐标激活引导实现语言模型中可控的社交立场

    Single-Coordinate Activation Steering for Controllable Social Stance in Language Models

    [https://arxiv.org/abs/2611.29578](https://arxiv.org/abs/2611.29578)

    该论文发现仅需提高语言模型中单个神经元（Qwen3.5 4B 第9层第7515号神经元）的激活，无需任何训练即可控制模型的社交立场，使其回复在疏远与热情之间切换，并通过模型评分与盲测人类对比验证了该效果显著优于随机神经元干预。

    

    引导小型语言模型的单个神经元可以在无需训练的情况下赋予模型特定话题或怪癖，然而复现实验表明，许多已发表的单神经元方法是脆弱的或缺乏特异性的。本文探讨了这种形式是否能够控制人际立场。所研究的立场是一种“疏远的暗恋”（distant crush），即一个保持戒备、时而热情且前后不一致的聊天机器人角色。在 Qwen3.5 4B 模型中，通过对比手写的疏远与热切回复的激活，识别出第 9 层的第 7515 号神经元作为控制该立场的候选神经元。提高该神经元的激活值会使角色的回复变得更加疏远。三个语言模型将这一变化评定为比改变随机选择的神经元所产生的效果更强，盲测的人类对比实验也支持了这一发现。在消息之间改变干预的方向，也能使回复在更热情与更疏远的风格之间切换。这表明语言模型的对话立场

    arXiv:2611.29578v1 Announce Type: new  Abstract: Steering one neuron of a small language model can impose a topic or a quirk without training, yet replications show that many published single-neuron recipes are fragile or nonspecific. This paper asks whether an interpersonal stance can be controlled in this manner. The stance studied is a distant crush, a chatbot character that is guarded, intermittently warm and inconsistent. In Qwen3.5, 4B, contrasting activations on hand-written guarded and eager replies identified neuron 7515 of layer 9 as a candidate for controlling this stance. Increasing this neuron activation made the characters replies more distant. Three language models rated this change as stronger than the effects of changing randomly selected neurons, and a blind human comparison supported the finding. Changing the interventions direction between messages also shifted replies between warmer and more distant styles. This suggests that a language model conversational stance 
    
[^95]: LM2：将一个语言模型编译进另一个模型的权重中

    LM2 Compiling a Language Model into Another Model Weights

    [https://arxiv.org/abs/2611.29577](https://arxiv.org/abs/2611.29577)

    LM2通过在训练时约束生成器的算术运算，将训练好的故事生成模型直接编译进更大对话模型的新增权重层中，使宿主模型每生成一个字符仅需一个token（对比QwenPU方案需约25,000个），代价是文本预测精度略有下降。

    

    同时部署一个对话模型和一个专用模型可以避免维护一个独立的专用服务。我们提出了LM2，它将一个训练好的故事生成器嵌入到一个更大的对话模型中。较大的模型负责处理对话，并在收到故事生成请求时调用该生成器。我们在训练过程中对生成器的算术运算加以约束，使其计算可以直接编译为新增的模型层。我们将该方法与QwenPU进行了比较，后者是在嵌入式处理器上执行stories260K语言模型。LM2每生成一个字符只需要一个宿主token，而QwenPU大约需要25,000个。然而，其故事生成器对留出文本的预测能力较差，即每字符1.145比特，而stories260K为0.894比特（数值越低越好）。基于Qwen3.5 0.8B的实现占用18.1 GB，并且在测试运行中生成的字符与内部模型的独立整数实现相同。这些结果表明……

    arXiv:2611.29577v1 Announce Type: new  Abstract: Deploying a conversational model and a specialist together could avoid maintaining a separate specialist service. We present LM2, which embeds a trained story generator into a larger conversational model. The larger model handles the conversation and invokes the generator for story requests. We constrain the generator arithmetic during training so its calculations can be compiled directly into added model layers. We compare this approach with QwenPU, which executes the stories260K language model on an embedded processor. LM2 requires one host token per generated character, compared with approximately 25,000 for QwenPU. However, its story generator predicts held out text less well, i.e. 1.145 bits per character versus 0.894 for stories260K, where lower is better. The Qwen3.5, 0.8B implementation occupies 18.1 GB and generates the same characters as a standalone integer implementation of the inner model in the tested runs. These results il
    
[^96]: CompareDR：从个体质量到比较优势的深度研究报告评测基准

    CompareDR: Benchmarking Deep Research Reports from Individual Quality to Comparative Merit

    [https://arxiv.org/abs/2611.29549](https://arxiv.org/abs/2611.29549)

    提出了CompareDR基准，通过结合任务特定的逐点质量审查、基于准则的两两比较和选择性证据验证，实现对深度研究报告从个体质量到相对优势的全面评测。

    

    深度研究系统针对同一查询可能产出包含不同证据、分析和结论的报告。读者在决定该依赖哪些研究发现时，需要了解每份报告满足任务要求的程度，以及哪些差异对答案至关重要。对单一报告进行评分可以概括其质量，但无法识别究竟是哪些因素使一份报告比另一份更有用。我们提出了CompareDR，这是一个结合任务特定逐点审查与基于准则的两两比较的评测基准。我们使用段落关联的报告表示来定位相关差异，并采用选择性证据验证来核查可能影响判断的论断。该基准包含来自三个现有基准的40个中英文任务、来自九个系统的360份报告、1,099条经人工审核的评估标准，以及通过对80对报告进行盲测比较所获得的人类偏好数据。

    arXiv:2611.29549v1 Announce Type: new  Abstract: Deep research systems can produce reports with different evidence, analyses, and conclusions for the same query. A reader deciding which findings to rely on needs to know how well each report meets the task requirements and which differences matter to the answer. A score for a single report can summarize its quality, but it does not identify which contributions make one report more useful than another. We introduce CompareDR, a benchmark that combines task-specific pointwise audits with guideline-based pairwise comparisons. We use passage-linked report representations to locate relevant differences and selective evidence verification to check claims that could affect the judgment. The benchmark comprises 40 Chinese and English tasks from three existing benchmarks, 360 reports from nine systems, 1,099 human-reviewed evaluation criteria, and human preferences elicited through blinded comparisons of 80 report pairs. The hybrid evaluator mat
    
[^97]: 估计量化大语言模型中被沉默的词元

    Estimating Silenced Tokens in Quantized LLMs

    [https://arxiv.org/abs/2611.29536](https://arxiv.org/abs/2611.29536)

    本文提出一种简单可扩展的认证式估计方法，能够精确量化低比特量化下大语言模型输出头中被“沉默”（贪心解码永远无法生成）的词元数量，发现量化确实会导致词元被沉默。

    

    量化是压缩大语言模型的标准方法，通常通过困惑度和任务准确率来评估其优劣。我们提出了一个不同的问题：量化之后，模型还能说出哪些词？当一个词元在任何隐状态下都无法成为输出层的首选时，它就被“沉默”了，贪心解码器因此永远无法生成它。我们给出了一种简单且可扩展的方法来估计量化输出头沉默了多少词元，该方法结合了三项检查：一种能让词元获胜的方向（证明该词元是安全的）、词元所在行被其他行包围程度的上界，以及能证明词元永远无法获胜的精确匹配。因此每个计数都是一个经认证的区间：精确匹配构成其下界，被标记的词元只会抬高其上界。在我们报告的几乎所有设置中（低至3比特），区间的两端恰好重合，因此计数是精确的。在三个开源模型上的实验表明，量化确实会沉默词元：在大词表……（摘要截断）

    arXiv:2611.29536v1 Announce Type: new  Abstract: Quantization is the standard way to shrink large language models, and it is usually judged by perplexity and task accuracy. We ask a different question: after quantization, which words can the model still say at all? A token is silenced when no hidden state can make it the top choice of the output layer, so a greedy decoder can never produce it. We give a simple, scalable way to estimate how many tokens a quantized output head silences, combining three checks: a direction that lets a token win (so it is safe), an upper bound on how close a token's row is to being trapped inside the others, and an exact match that proves a token can never win. Each count is therefore a certified range: exact matches give its lower end and flagged tokens can only raise its upper end. In nearly all settings we report down to 3 bits the two ends coincide, so the counts are exact. Across three open models we find that quantization silences tokens: on large-vo
    
[^98]: 适应还是投毒？大语言模型对话式测试时学习的安全风险

    Adaptation or Poisoning? Security Risks of Conversational Test-Time Learning in Large Language Models

    [https://arxiv.org/abs/2611.29532](https://arxiv.org/abs/2611.29532)

    该论文揭示了大语言模型测试时学习的安全风险，提出了P-TAP多轮攻击方法，攻击者仅通过普通对话中逐步升级的查询即可毒化会话适配器，最终绕过安全对齐实现越狱。

    

    测试时学习（TTL）使大语言模型能够在部署过程中适应无标签输入。模型通过最小化其接收输入上的下一词预测损失来更新一个轻量级适配器。当这些更新在多轮对话中持续存在时，每个用户查询既是在当前轮次中被回答的请求，也是后续轮次的适应数据。通过选择用于适应的查询，攻击者可以在与TTL聊天服务的普通交互中影响参数更新并削弱安全对齐。为了展示这一风险，我们提出了渐进式测试时适应投毒（P-TAP），这是一种仅通过查询实现的多轮攻击，通过服务自身的更新来毒化会话适配器，从而对模型实施越狱。一系列逐步升级的对抗性查询逐轮塑造适配器；最终的有害请求在累积的状态下被回答，而该请求本身并未被用于适应。

    arXiv:2611.29532v1 Announce Type: cross  Abstract: Test-time learning (TTL) enables large language models to adapt to unlabeled inputs during deployment. The model updates a lightweight adapter by minimizing the next-token prediction loss on the input it receives. When these updates persist across conversation turns, every user query serves both as the request answered in the current turn and as adaptation data for later turns. By choosing the queries used for adaptation, an attacker can influence parameter updates and weaken safety alignment through ordinary interactions with a TTL chat service. To demonstrate this risk, we introduce Progressive Test-Time Adaptation Poisoning (P-TAP), a query-only multi-turn attack that jailbreaks the model by poisoning its session adapter through the service's own updates. A sequence of escalating adversarial queries shapes the adapter turn by turn; the final harmful request is then answered under the accumulated state without itself being used for a
    
[^99]: 递归自编辑：大语言模型编写自身权重编辑程序以实现自我改进

    Recursive Self-Editing: LLMs Program Their Own Weight Edits for Self-Improvement

    [https://arxiv.org/abs/2611.29513](https://arxiv.org/abs/2611.29513)

    提出 SELFWRITE 框架，让大语言模型通过编写程序、利用白盒基元（而非梯度下降）直接构造并合并自身权重的低秩编辑，从而实现递归式的自我改进。

    

    递归自我改进要求一个系统去改进那个用来改进它自身的系统。当今的语言模型通过数据、训练方案或脚手架来改进自己，而每当它们的权重发生变化时，写入更新的是梯度下降，而非模型本身。我们让模型自己编写自己的权重。在 SELFWRITE 中，每一代，当前模型会在新的训练问题上研究自己的推理过程，在一个持久的 Python 会话中检查自己的权重和激活值，然后编写一个程序，利用白盒基元（奇异方向、其正确与错误答案之间的激活对比、秩一投影、张量计算）为自身构建低秩权重编辑，全程不使用任何训练或梯度工具。每个候选编辑都在一个隐藏的关卡集上针对当前模型评分一次；严格带来改进的最优编辑会被合并进权重，随后编辑后的模型继续编写下一个编辑。在 Qwen3.8-27B 及数学问题上（原文在此处截断）……

    arXiv:2611.29513v1 Announce Type: cross  Abstract: Recursive self-improvement asks a system to improve the very system that improves it. Today's language models improve themselves through data, training recipes or scaffolds. Whenever their weights change, gradient descent writes the update, not the model. We let the model write its own weights. In SELFWRITE, each generation the current model studies its own reasoning on fresh training problems, inspects its own weights and activations in a persistent Python session, and writes a program that constructs a low-rank weight edit for itself from white-box primitives (singular directions, activation contrasts between its right and wrong answers, rank-one projections, tensor computations), with no training or gradient tool. Every candidate is graded once on a hidden gate set against the current model; the best strictly improving edit is merged into the weights, and the edited model writes the next edit. On Qwen3.8-27B and math problems from D
    
[^100]: Meddies-PII：一个用于临床去身份化的多语言个人身份信息提取框架

    Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification

    [https://arxiv.org/abs/2611.29491](https://arxiv.org/abs/2611.29491)

    该论文发布了覆盖17种语言（含7种东南亚语言）和9类PII标签的大型多语言临床去身份数据集及完整透明的生成流程，基于此训练的紧凑3.5亿参数模型在全部15个公开基准上以平均精确匹配F1 0.827大幅超越最强现有系统（0.672）。

    

    临床文本中的多语言个人身份信息（PII）提取一直受限于稀缺的受监管数据，以及构建过程记录不充分的合成语料库。我们提出了一个覆盖十七种语言（包括七种东南亚语言）、包含九类PII标签的大型多语言数据集。我们还详细记录了完整的生成流程，并发布了全部的提示词、目录、检查规则和标签映射。基于该资源训练的一个紧凑的3.5亿参数BIOES词元分类器在全部十五个公开基准上均取得了最高分，其中在十四个基准上显著领先于四个公开系统中的最强系统，平均精确匹配F1达到0.827，而最强系统仅为0.672。我们还将其在未经适配的情况下测试于多语言基准，包括两个人工标注的德语和西班牙语临床风格语料库，在标注规范对齐后，其表现与最佳公开系统相当，并在十个Mu……（摘要在此处截断）

    arXiv:2611.29491v1 Announce Type: new  Abstract: Multilingual PII extraction in clinical text is limited by scarce governed data and by synthetic corpora whose construction is thinly documented. We present a large multilingual dataset covering seventeen languages, including seven Southeast Asian languages, and nine PII labels. We also document a detailed generation pipeline and release the complete prompts, catalogues, checks and label maps. A compact 350M-parameter BIOES token classifier trained on the resource achieves the highest score on all fifteen public benchmarks, significantly ahead of the strongest of four public systems on fourteen of them, with a mean exact-match F1 of 0.827 against 0.672 for the strongest system. We also tested it, without adaptation, on multilingual benchmarks, including two human-annotated German and Spanish clinical-style corpora, where it is competitive with the best public systems once annotation conventions are aligned, and is best in four of ten Mul
    
[^101]: GraphSeer：基于多模态大语言模型的多模态知识图谱关系图像检索

    GraphSeer: Multimodal Knowledge Graph Relational Image Retrieval with MLLMs

    [https://arxiv.org/abs/2611.29470](https://arxiv.org/abs/2611.29470)

    GraphSeer 提出了知识图谱关系图像检索（KRIR）这一新任务——给定照片和自然语言关系检索相关的多模态知识图谱实体，并在单个多模态大语言模型中统一完成“识别照片对应实体”与“推理其关联实体”这两个步骤。

    

    多模态知识图谱（MMKG）以照片和名称存储实体，并通过事实将它们相互链接。现有 MMKG 方法预测已知实体 ID 之间的链接，而图像检索器（从 CLIP 到各类 MLLM 嵌入器）则返回与查询照片相似或符合其描述的实体。然而，两者都无法回答一个关于新照片的常见问题：照片中所示实体与哪个图谱实体相关，例如照片中球员所属的球队，或地标所在的城市。我们将此任务称为知识图谱关系图像检索（KRIR）：给定一张照片和一个自然语言关系，检索相关的 MMKG 实体——该实体通常与照片本身并不相像——且无论照片中描绘的实体是否存在于图谱中（已知）或不存在（未见）。我们基于 YAGO、Wikidata 和真实场景的 OVEN 照片构建了三个基准数据集。该任务隐含两个跳转步骤：识别照片中展示的是哪个图谱实体，并回忆该实体与什么相关联。GraphSeer 在单一多模态大语言模型内部将这两个步骤衔接起来：它将照片与图谱建立链接，并…

    arXiv:2611.29470v1 Announce Type: new  Abstract: Multimodal knowledge graphs (MMKGs) store entities with photos and names and link them by facts. MMKG methods predict links between known entity IDs, and image retrievers, from CLIP to MLLM embedders, return entities that look like a query photo or match its description. Neither answers a common question about a new photo: which graph entity is related to the one shown, such as the team of a pictured player or the city of a landmark. We call this knowledge graph relational image retrieval (KRIR): given a photo and a natural-language relation, retrieve the related MMKG entity, which rarely resembles the photo, whether the depicted entity is in the graph (known) or not (unseen). We build three benchmarks from YAGO, Wikidata and in-the-wild OVEN photos. The task hides two hops: recognising which graph entity is shown, and recalling what it is connected to. GraphSeer bridges them inside one multimodal LLM: it links the photo to the graph, re
    
[^102]: 超越准确率：面向大语言模型智能体假设驱动实验规划的诊断性基准

    Beyond Accuracy: A Diagnostic Benchmark for Hypothesis-Driven Experiment Planning in LLM Agents

    [https://arxiv.org/abs/2611.29467](https://arxiv.org/abs/2611.29467)

    该论文提出了一个超越最终准确率的诊断性基准与评估框架，通过评估智能体的探索行为模式并将其推理过程分解为信念更新、假设区分、判别性实验设计和干预下预测推理四个可解释步骤，来深入诊断LLM智能体在假设驱动实验规划中的能力。

    

    在科学任务上评估大语言模型（LLM）智能体通常被简化为最终任务准确率，这掩盖了推理在何处成功或失效。我们提出了一个针对封闭世界系统辨识中假设驱动自适应实验规划的基准与诊断评估框架，并以动力学机理辨识作为具体实例。该框架在两个方面超越了最终准确率的评估方式。首先，我们通过量化智能体在实验设计空间中表现出空间填充或角落聚集模式的程度来评估其探索行为。其次，我们将推理过程分解为四个可解释的步骤：信念更新、假设区分、判别性实验设计以及干预下的预测推理，并在推理轨迹上对每个步骤分别评分。在十个LLM智能体的实验中，我们发现更强的智能体在相关指标上优于自适应（贝叶斯优化）和非自适应（拉丁……）方法。

    arXiv:2611.29467v1 Announce Type: cross  Abstract: Evaluating large language model (LLM) agents on scientific tasks typically reduces to final-task accuracy, which obscures where reasoning succeeds or breaks down. We introduce a benchmark and diagnostic evaluation framework for hypothesis-driven adaptive experimental planning in closed-world system identification, instantiated in kinetic mechanism identification. The framework goes beyond final accuracy in two ways. First, we evaluate exploration behavior by quantifying the extent to which agents exhibit space-filling or corner-seeking patterns in the experimental design space. Second, we decompose the reasoning process into four interpretable steps: belief update, hypothesis discrimination, discriminative experiment design, and predictive reasoning under interventions, each scored separately on reasoning traces. Across ten LLM agents, we find that stronger agents outperform both adaptive (Bayesian optimization) and non-adaptive (Latin
    
[^103]: MAS-BIT：一种用于端到端脑信号到文本解码的多智能体系统

    MAS-BIT: A Multi-Agent System for End-to-End BraIn-to-Text Decoding

    [https://arxiv.org/abs/2611.29457](https://arxiv.org/abs/2611.29457)

    该论文提出了首个面向端到端脑信号到文本解码的多智能体框架MAS-BIT，通过神经智能体、语音智能体与仲裁智能体之间基于词嵌入的潜空间通信协同工作，直接从皮层内神经记录生成文本转录。

    

    语音脑机接口旨在通过将神经活动转换为文本来恢复交流能力，然而端到端解码器的准确率历来落后于级联系统。我们提出了脑信号到文本的多智能体系统（MAS-BIT），这是首个基于皮层内神经记录进行端到端脑信号到文本解码的多智能体框架。MAS-BIT 协调三个专门化智能体：负责解释连续神经表示的神经智能体、负责解码软音素后验概率的语音智能体，以及整合两者互补证据并可直接访问神经表示以生成最终转录文本的仲裁智能体。专家智能体通过基于词嵌入的潜空间通信与仲裁者进行交流，将连续消息构建为词嵌入的概率加权组合。这些消息保留了候选的词元假设，并提供了一个可微的接口……

    arXiv:2611.29457v1 Announce Type: cross  Abstract: Speech brain-computer interfaces aim to restore communication by translating neural activity into text, yet end-to-end decoders have historically trailed cascaded systems in accuracy. We introduce a Multi-Agent System for BraIn-to-Text (MAS-BIT), the first multi-agent framework for end-to-end brain-to-text decoding from intracortical recordings. MAS-BIT coordinates three specialized agents: a neural agent that interprets continuous neural representations, a phonetic agent that decodes soft phoneme posteriors, and an arbitration agent that integrates their complementary evidence with direct access to neural representations to generate the final transcription. The expert agents communicate with the arbiter through token-embedding-based latent communication, constructing continuous messages as probability-weighted combinations of token embeddings. These messages retain alternative token hypotheses and provide a differentiable interface th
    
[^104]: PAWP-TTS：面向紧凑型英-尼泊尔语双语语音合成的音素感知权重编程

    PAWP-TTS: Phoneme-Aware Weight Programming for Compact Bilingual English-Nepali Speech Synthesis

    [https://arxiv.org/abs/2611.29455](https://arxiv.org/abs/2611.29455)

    PAWP-TTS通过将确定性发音知识交由基于规则和词典的音素感知双语前端处理，使网络仅需建模声学，从而以仅5.98 MB、149万参数的紧凑非自回归模型实现了高质量的英语-尼泊尔语双语及混合编码语音合成。

    

    本文提出了PAWP-TTS，一个紧凑的非自回归文本转语音系统，支持英语、尼泊尔语以及英-尼泊尔语混合编码语音，采样率为24 kHz，且推理时无需语言模型。确定性的发音知识被置于基于规则和词典的双语前端中，该前端保留了尼泊尔语的原生语音对比并避免印地语语音体系的影响，从而使网络只需建模声学部分。计入所有学习到的推理张量（包括波形解码器），PAWP-1.5M（1,490,390个参数，5.98 MB）在单一ASR识别器下达到1.50%的英语和1.75%的尼泊尔语字符错误率，并在600条未参与训练的测试语句上达到5.04%/4.02%的尼泊尔语/英语音素错误率（PER）。以解码器为中心的281K参数模型（1.1 MB）在三个独立随机种子下均保持可懂度，其中同等参数预算的密集模型（尼泊尔语5.53 ± 0.08%，英语5.34 ± 0.28% PER）优于K=2跨层共享基模型（尼泊尔语6.35 ± 0.11%，英语6.57 ± 0.47% PER）。

    arXiv:2611.29455v1 Announce Type: new  Abstract: This paper presents PAWP-TTS, a compact non-autoregressive text-to-speech system for English, Nepali, and English-Nepali code-mixed speech at 24 kHz, with no language model at inference. Deterministic pronunciation knowledge lives in a rule- and lexicon-based bilingual frontend that preserves native Nepali contrasts and avoids Hindi phonology, so the network only models acoustics. Counting every learned inference tensor, including the waveform decoder, PAWP-1.5M (1,490,390 parameters, 5.98 MB) reaches 1.50% English and 1.75% Nepali character error rate under a single ASR recognizer, and 5.04% / 4.02% Nepali / English phoneme error rate (PER) on 600 untouched test utterances. Decoder-centric 281K-parameter models (1.1 MB) stay intelligible across three independent seeds, with equal-budget dense models (5.53 +/- 0.08% NE, 5.34 +/- 0.28% EN PER) outperforming K=2 cross-layer shared-basis models (6.35 +/- 0.11% NE, 6.57 +/- 0.47% EN PER). Fi
    
[^105]: 可检测的认知攻击层：面向会话历史完整性的覆盖定理与标准化测试框架

    Detectable Epistemic Attack Layers: A Coverage Theorem and Standardized Harness for Conversation-History Integrity

    [https://arxiv.org/abs/2611.29449](https://arxiv.org/abs/2611.29449)

    该论文按攻击层形式化了会话历史污染并证明覆盖定理——每层验证恰好只覆盖自身层、指令覆盖的检测必然带来误报、复合攻击向量可绕过逐片段验证——并通过覆盖八个机制家族、3,249个会话和三个模型家族的标准化测试框架予以量化验证。

    

    会话历史是一种基础设施：已部署的语言模型依赖应用层来存储并重新注入先前的对话轮次，而这一存储状态是可写的。我们按攻击层形式化了会话级污染，给出一个覆盖定理，并测量了其适用范围：每个部署层验证层级恰好只覆盖其所在层，且相对于其他层是可靠的；指令覆盖在无误报的前提下不可检测；复合攻击向量可规避逐片段验证。一个应用于八个机制家族、跨越3,249个会话和三个模型家族的标准化测试框架使这些论断变得可检验。首先，基于模式的检测需付出实测的误报代价：经典防御会误标35%至50%的良性会话，而朴素的拒绝底线仅能达到2.4%的攻击单元采用率，同时会破坏33%至100%的良性对话轮次，而溯源门控的信道分离实现了……（原文摘要在此处截断）

    arXiv:2611.29449v1 Announce Type: new  Abstract: Conversation history is infrastructure: deployed language models rely on application layers to store and reinject prior turns, and that stored state is writable. We formalize session-level contamination by attack layer, state a coverage theorem, and measure its scope: each deployment-layer verification tier covers exactly its layer and is sound with respect to the others; instruction overrides are not detectable without false positives; and composite vectors evade per-fragment verification. A standardized harness applied to eight mechanism families across 3,249 sessions and three model families makes these claims checkable. First, pattern-based detection pays a measured false-positive price: the classic defenses flag 35 to 50 percent of benign sessions, and the naive refusal floor reaches only 2.4 percent attack-cell adoption while destroying 33 to 100 percent of benign turns, whereas provenance-gated channel separation achieves adoption
    
[^106]: VistaTree：基于图像导出摘要树的长篇多模态文档问答

    VistaTree: Question Answering over Long Multimodal Documents with Image-Derived Summary Trees

    [https://arxiv.org/abs/2611.29419](https://arxiv.org/abs/2611.29419)

    VistaTree通过直接从页面图像构建层次化摘要树并进行层级导航，实现了对超出上下文窗口的长篇多模态文档的高效问答。

    

    对长篇多模态文档进行问答需要在包含文本、表格和图表的页面中定位相关证据。当文档超出模型的上下文窗口时，就需要进行选择性检索。我们提出了VistaTree，这是一个文档问答智能体，它直接从渲染的页面图像构建摘要树。VistaTree将文档分割为连续的页面片段，递归地将它们组织成主题层次结构，并细化兄弟节点摘要以区分相邻章节。在查询时，该智能体使用节点摘要来选择相关节点并阅读对应页面，当完整大纲超出上下文预算时则采用层次化导航。在MMLongBench-Doc和LongDocURL数据集上的实验使用本地部署的视觉-语言模型评估了答案质量和检索成本，并量化了对索引进行全局访问与层次化访问之间的性能差距。

    arXiv:2611.29419v1 Announce Type: new  Abstract: Answering questions over long multimodal documents requires locating relevant evidence across pages containing text, tables, and figures. When the document exceeds the model's context window, selective retrieval becomes necessary. We introduce VistaTree, a document question-answering agent that builds a summary tree directly from rendered page images. VistaTree segments documents into contiguous page spans, recursively groups them into a topic hierarchy, and refines sibling summaries to distinguish neighboring sections. At query time, the agent uses node summaries to select relevant nodes and reads the corresponding pages, using hierarchical navigation when the full outline exceeds the context budget. Experiments on MMLongBench-Doc and LongDocURL evaluate answer quality and retrieval cost using a locally hosted vision-language model, and quantify the performance gap between global and hierarchical access to the index.
    
[^107]: HGDNA：面向高效自回归基因组语言模型的重叠k-mer分词方法

    HGDNA: Overlapping k-mer Tokenization for Efficient Autoregressive Genomic Language Models

    [https://arxiv.org/abs/2611.29400](https://arxiv.org/abs/2611.29400)

    提出HGDNA，一种采用重叠k-mer分词与混合线性注意力机制的因果基因组语言模型，仅用30M参数和19B预训练token即在序列分类和嵌入评估任务上取得有竞争力的性能。

    

    基因组语言模型领域发展迅速，在多种基因组语料库上预训练的模型展现了卓越的能力。虽然简单核苷酸级分词对长上下文建模的影响已被充分研究，但重叠k-mer分词在因果预训练中仍然探索不足，尽管其蕴含丰富的邻域信息。本文全面重新审视了重叠分词，提出了HGDNA——一种采用因果语言建模预训练的混合线性注意力基因组语言模型，并配备了共享的token预测接口用于下游适配，同时使用物种分类辅助任务和共享类token。HGDNA在序列分类和冻结嵌入评估中取得了有竞争力的表现，仅用30M参数和19B预训练token即可支持短程和长程任务。

    arXiv:2611.29400v1 Announce Type: cross  Abstract: The domain of genomic language models (gLMs) has advanced rapidly, with models pretrained on diverse multi-species genomic corpora demonstrating their remarkable capabilities. While the effect of simple nucleotide-level tokenization on long-context modeling is well established, overlapping k-mer tokenization remains underexplored in causal pretraining, despite its rich neighborhood information. Here, we thoroughly revisit overlapping tokenization and introduce HGDNA, a hybrid linear-attention gLM with causal language modeling (CLM) pretraining and a shared token-prediction interface for downstream adaptation, using a species-classification auxiliary task and shared class tokens. HGDNA delivers competitive performance across sequence classification and frozen-embedding evaluations, supporting both short-range and long-range tasks with 30M parameters and 19B pretraining tokens. Code is available at https://github.com/Sophie10001b/HGDNA
    
[^108]: 语音大语言模型能否识别音频证据不足的情况？一项诊断研究与一种生成前探测缓解方法

    Can Speech LLMs Recognize When Audio Evidence Is Insufficient? A Diagnostic Study and a Pre-Generation Probing Mitigation

    [https://arxiv.org/abs/2611.29393](https://arxiv.org/abs/2611.29393)

    该研究通过对比合成语音与真人朗读录音的诊断实验，揭示了语音大语言模型在音频证据不足以回答问题时会大量幻觉作答、且合成语音评估会严重误导失败归因，并提出一种生成前探测方法来缓解该问题。

    

    语音大语言模型通常在有答案的问题上进行评估。我们探讨当录音无法为问题提供任何依据时模型会怎么做，以及目前标准的研究方法是否可靠。在合成语音上进行诊断时，结论看似清晰：Qwen2-Audio 对 92.5% 的不可回答问题编造了答案，而读取相同录音的 ASR 级联系统在 6.7% 的过度拒答率下仅产生 2.5% 的幻觉，这似乎将失败归因于推理而非感知。然而，在 376 个基于人工朗读录音的 SQuAD 2.0 原生问题上采用完全相同的评估协议，上述结论被推翻：级联系统产生了 67.7% 的幻觉，而端到端模型为 69.8%，且端到端模型的过度拒答成本更高。通过构建配对子集——对每篇文章的合成录音和人工录音提出相同的问题——该效应被定位于 ASR 阶段：端到端模型的弃答行为不随说话人而变化，而级联系统的弃答率则从 11.1% 升至 65.9%。对……的依赖……（原摘要在此处截断）

    arXiv:2611.29393v1 Announce Type: new  Abstract: Speech large language models are usually evaluated on questions that have answers. We ask what they do when a recording establishes none, and whether the standard way of finding out is sound. Diagnosed on synthesized speech the answer looks clear: Qwen2-Audio fabricates an answer to 92.5% of unanswerable questions, while an ASR cascade reading the same recordings reaches 2.5% hallucination at 6.7% over-refusal, placing the failure in reasoning rather than in perception. The identical protocol on 376 native SQuAD 2.0 questions over human-read recordings removes that conclusion: the cascade hallucinates on 67.7% against the end-to-end model's 69.8%, at a higher over-refusal cost. A paired subset asking the same questions of a synthesized and a human recording of each passage localises the effect in the ASR stage -- the end-to-end model's abstention behaviour does not move with the speaker, the cascade's goes from 11.1% to 65.9%. Reliance o
    
[^109]: SAE-Audit：通过稀疏自编码器证据审计的推理时黑盒大语言模型优化进行临床试验预测

    SAE-Audit: Clinical-Trial Prediction via Inference-Time Optimization of Black-Box LLMs Audited by Sparse Autoencoder Evidence

    [https://arxiv.org/abs/2611.29368](https://arxiv.org/abs/2611.29368)

    提出SAE-Audit方法，将开源模型上稀疏自编码器检测到的临床概念证据用于推理时优化黑盒大语言模型，从而提升临床试验结局预测的准确性和可解释性。

    

    临床试验成本高昂且往往需要数年才能完成，因此早期结局预测对试验规划具有重要价值。大语言模型（LLM）正越来越多地被用于预测临床试验结局，但其预测仍然不可靠，且其给出的理由可能无法忠实反映其答案背后所依据的证据。在预测一项注册临床试验将顺利完成还是提前终止等任务上，前沿LLM在我们实验中的表现仅略高于随机水平。我们提出SAE-Audit方法，该方法在推理时利用从可部署的开源模型中获得的证据来优化黑盒LLM。具体而言，将Llama-3.1-8B-Instruct上的冻结稀疏自编码器（SAE）转换为临床概念词表，并在历史试验数据上拟合稀疏探针，以确定哪些概念与试验结局相关。对于新的试验方案，探针检测到的概念将被写成证据，每条证据都带有方向和强度（原文在此处截断）。

    arXiv:2611.29368v1 Announce Type: new  Abstract: Clinical trials are costly and often take years to complete, making early outcome prediction valuable for trial planning. Large language models (LLMs) are increasingly used to forecast clinical-trial outcomes, yet their predictions remain unreliable and their rationales may not faithfully reflect the evidence underlying their answers. On tasks such as predicting whether a registered clinical trial will complete or terminate early, frontier LLMs perform only modestly above chance in our experiments. We introduce SAE-Audit, which optimizes a black-box LLM at inference time with evidence obtained from a deployable open model. A frozen sparse autoencoder (SAE) on Llama-3.1-8B-Instruct is converted into a vocabulary of clinical concepts. Sparse probes fitted on past trials determine which concepts are associated with the outcome. For a new protocol, the concepts detected by the probe are written as evidence, each with the direction and streng
    
[^110]: 面向对话式产品搜索的波斯语电子商务对话数据集自动构建

    Automatic Construction of Persian E-Commerce Dialogue Datasets for Conversational Product Search

    [https://arxiv.org/abs/2611.29356](https://arxiv.org/abs/2611.29356)

    该论文提出一个基于正则化分区平衡（RPB）分数与LLM的自动化框架，从真实电商目录中自动构建并发布了覆盖7个类别、12,926个产品的大规模波斯语对话式产品搜索数据集。

    

    对话式产品搜索让用户能够通过多轮对话而非单一关键词查询来查找产品，但波斯语缺乏训练和评估此类系统所需的大型对话数据集；现有的波斯语电子商务对话资源覆盖的类别少、商品目录规模小。我们提出了一个框架，可以从真实的电子商务目录中自动构建此类数据集，并发布了由此生成的数据集。该框架爬取了Digikala市场中七个类别的12,926个产品，将其归入170个语义连贯的聚类，并使用正则化分区平衡（RPB）分数来选择助手在每一轮提出的问题。RPB分数综合考虑了特征覆盖率、用户能够回答的概率以及取值数量，是对基于熵的问题选择方法的推广。随后，大语言模型（LLM）将每个问题规划转化为自然的波斯语对话。在完整目录上的模拟实验中……

    arXiv:2611.29356v1 Announce Type: new  Abstract: Conversational product search lets users find products through a multi-turn dialogue instead of a single keyword query, but Persian lacks the large conversational datasets needed to train and evaluate such systems; existing Persian e-commerce dialogue resources cover few categories and small catalogs. We present a framework that builds such a dataset automatically from a real e-commerce catalog and release the resulting dataset. The framework crawls 12,926 products in seven categories of the Digikala marketplace, groups them into 170 semantically coherent clusters, and selects the question the assistant asks at every turn with the Regularized Partition Balance (RPB) score. The RPB score accounts for feature coverage, the probability that the user can answer and the number of values, and generalizes entropy-based question selection. An LLM then verbalizes each question plan into a natural Persian dialogue. In a simulation on the full cata
    
[^111]: 端点条件化过程价值：当已知最终状态与成本后，盲端点过程评分还能增加什么

    Endpoint-Conditioned Process Value: What an Endpoint-Blind Process Score Adds Once Terminal Status and Cost Are Known

    [https://arxiv.org/abs/2611.29354](https://arxiv.org/abs/2611.29354)

    本研究通过1,255条轨迹的实验首次量化了过程评分在已知最终状态和成本后的增量价值，发现其增益高度依赖评判模型规模——只有32B级别的评判模型才能带来显著的AUC提升，且大部分增益可由八个可记录的行为计数器替代。

    

    轨迹度量承诺提供超越最终答案准确性的信息，但尚无研究测量过当评估已知结果和成本后，过程评分还能增加什么。我们构建了这一比较：一个需要双方都无法看到产物的密封后果实验、一个由最终状态和成本构成的操作基线，以及将评判模型规模与证据访问作为实验因素。在7个Python代码库中158个测试生成任务的1,255条轨迹上，答案是取决于评判模型的。四个4B到14B的评判模型（先推理后评分）所增加的信息与零无法区分；而在相同轨迹上，一个Qwen3 32B评判模型增加了+0.050 AUC（95%置信区间[+0.020, +0.081]），并通过了重新拟合的bootstrap检验（[+0.013, +0.095]）。它所增加的大部分信息可以从八个行为计数器中恢复，因此测试框架应当记录这些信息（在此之上评判模型仅增加+0.020）。评判模型所看到的内容会进一步改变结果：披露最终判定结论……

    arXiv:2611.29354v1 Announce Type: new  Abstract: Trajectory metrics promise information beyond final-answer accuracy, but no one has measured what a process score adds once an evaluation knows the outcome and the cost. We build that comparison: a sealed consequence needing an artifact neither agent nor judge ever sees, an operational baseline of terminal status and cost, and judge scale and evidence access as experimental factors. On 1,255 trajectories over 158 test-generation tasks in 7 Python repositories, the answer is gated on the judge. Four judges from 4B to 14B, one reasoning before scoring, add nothing an interval separates from zero; a Qwen3 32B judge on identical trajectories adds +0.050 AUC (95% CI [+0.020, +0.081]) and survives a refitting bootstrap ([+0.013, +0.095]). Most of what it adds is recoverable from eight behavioural counters, so a harness should log them (the judge adds +0.020 beyond them). What the judge is shown moves it further: disclosing the terminal verdict
    
[^112]: EMIP：具身多方交互平台

    EMIP: Embodied Multi-Party Interaction Platform

    [https://arxiv.org/abs/2611.29333](https://arxiv.org/abs/2611.29333)

    该论文提出了一个基于Minecraft构建的具身多方交互平台EMIP，通过客户端-服务器架构支持多个人类与多个AI智能体在同一虚拟环境中进行全双工语音交流和同步物理交互，为研究人类与具身AI智能体之间的协作提供了实验平台。

    

    能够让多个参与者在共享环境中同时进行口语对话并执行物理动作的实验平台，对于研究人类与具身AI智能体之间的协作非常重要。在本文中，我们提出了EMIP（具身多方交互平台），一个基于Minecraft构建的虚拟环境平台。通过基于互联网的客户端-服务器架构，EMIP将多个人类和多个AI智能体连接到同一虚拟环境中，并支持全双工语音通信和同步物理交互。它提供基于模型上下文协议（MCP）的接口，用于环境观察、物理动作执行以及音频输入和输出，使研究人员能够使用大语言模型（LLM）开发自己的智能体控制方法。其他功能还包括通过单个EMIP智能体客户端对智能体群组进行集中控制、协作对象操作等。

    arXiv:2611.29333v1 Announce Type: new  Abstract: Experimental platforms that enable multiple participants to engage in spoken dialogue and perform physical actions concurrently in a shared environment are important for studying cooperation between humans and embodied AI agents. In this paper, we propose EMIP (Embodied Multi-Party Interaction Platform), a virtual environment platform built on Minecraft. Through an Internet-based client-server architecture, EMIP connects multiple humans and multiple AI agents to the same virtual environment and supports full-duplex speech communication and simultaneous physical interaction. It provides interfaces based on the Model Context Protocol (MCP) for environmental observation, physical action execution, and audio input and output, enabling researchers to develop their own agent control methods using large language models (LLMs). Additional features include centralized control of an agent group through a single EMIP agent client, collaborative obj
    
[^113]: 技术图纸上的符号计数：面向多模态大语言模型的基准与验证奖励

    Counting Symbols on Technical Drawings: A Benchmark and Verified Rewards for MLLMs

    [https://arxiv.org/abs/2611.29309](https://arxiv.org/abs/2611.29309)

    该论文提出了涵盖建筑平面图、电气原理图、机械图纸和分子结构共1,574个计数问题的CountDrawBench基准，并设计了结合计数奖励、实例匹配奖励与基于缩放奖励的Code-Verified RL强化学习框架，通过可执行验证反馈显著提升多模态大语言模型在技术图纸上的符号计数能力。

    

    如今的多模态大语言模型（MLLMs）能够对自然图像中的物体进行计数。然而，当被问及机械图纸中有多少个螺栓孔，或化学结构中有多少个氮原子时，它们往往表现不佳。技术图纸中的计数仍然具有挑战性，因为目标物体可能很小、密集排列，甚至仅由绘图惯例隐含表示。为了研究这一问题，我们提出了CountDrawBench，一个包含建筑平面图、电气原理图、机械图纸和分子结构的基准数据集，共包含1,574个计数问题，并在可行的情况下提供实例级标注。除评估之外，我们还提出了Code-Verified RL，一个通过可执行反馈来教会MLLMs进行计数的强化学习框架。我们的方法结合了三种互补的奖励：检验最终答案的计数奖励、验证物体位置的实例匹配奖励，以及检验局部图像区域计数一致性的基于缩放的奖励。

    arXiv:2611.29309v1 Announce Type: cross  Abstract: Today's MLLMs can count objects in natural images. However, when asking them how many bolt holes appear in a mechanical drawing or how many nitrogen atoms are in a chemical structure, they often struggle. Counting in technical drawings remains challenging, as objects can be small, densely packed, or even implied by drawing conventions. To study this problem, we introduce CountDrawBench, a benchmark including floor plans, electrical schematics, mechanical drawings, and molecular structures, with 1,574 counting questions and instance-level annotations where available. Besides evaluation, we introduce Code-Verified RL, a reinforcement learning framework that teaches MLLMs to count with executable feedback. Our method combines three complementary rewards: a count reward that checks the final answer, an instance-matching reward that verifies object locations, and a zoom-based reward that checks counting consistency across local image region
    
[^114]: 从梯度消失到递归自我改进：评估大语言模型演进中最具影响力的论文（1997-2026）

    From Vanishing Gradients to Recursive Self-Improvement: Evaluating the Most Influential Papers in Large Language Model Evolution (1997-2026)

    [https://arxiv.org/abs/2611.29303](https://arxiv.org/abs/2611.29303)

    本综述按时间线评估了1997-2026年间22篇最具影响力的大语言模型论文，梳理了从RNN、LSTM、Word2Vec到Transformer的关键技术演进脉络。

    

    大语言模型（LLM）研究的迅速激增，使得区分渐进式改进与真正的范式转变变得越来越困难。本综述评估了近三十年间（1997-2026）最具影响力的22篇出版物，并采用一种以作者出生年份（1999年）为中心的、基于因果关系的按时间顺序锚定的分类体系进行组织。从“1999年之前”开始，探讨循环神经网络（RNN）如何遗忘较早的上下文（就像读者读到第9章时早已忘记第1章的内容），我们追溯了长短期记忆网络（LSTM）如何引入一种由遗忘门、输入门和输出门构成的“门卫”机制来选择性地保留长期记忆。随后，我们考察了Word2Vec如何将语言转化为向量数学（king - male + female ≈ queen），为什么串行处理的RNN遇到了并行化的瓶颈，以及Transformer的全对全注意力权重机制如何开启了现代LLM时代，并同时带来了二次方复杂度的……

    arXiv:2611.29303v1 Announce Type: cross  Abstract: The rapid proliferation of Large Language Model (LLM) research makes it increasingly difficult to distinguish incremental refinements from true paradigm shifts. This review evaluates 22 of the most influential publications spanning nearly three decades (1997-2026), organized through a causal, chronologically anchored taxonomy centered on the author's birth year (1999). Beginning "before 1999" with how recurrent neural networks (RNNs) forgot earlier context (much like forgetting Chapter 1 by the time a reader reaches Chapter 9), we trace how Long Short-Term Memory (LSTM) networks introduced a "bouncer" mechanism of forget, input, and output gates to selectively preserve long-term memory. We then examine how Word2Vec turned language into vector mathematics (king - male + female ~ queen), why sequential RNNs hit a parallelization wall, and how the Transformer's all-to-all attention weights launched the modern LLM era alongside a quadratic
    
[^115]: 超越遗漏：测量大型语言模型在孟加拉语和英语中的关键信息完成度

    Beyond Omission: Measuring Critical-Information Fulfillment of Large Language Models in Bengali and English

    [https://arxiv.org/abs/2611.29299](https://arxiv.org/abs/2611.29299)

    该研究发现，大型语言模型在孟加拉语和英语上的关键信息遗漏率几乎相同，但孟加拉语的关键信息完成度和完整任务成功率显著更低，表明多语言可信度评估不应只关注信息遗漏，还应衡量信息的完成质量。

    

    对多语言可信度的评估通常只关注模型是否会遗漏用户所需的信息，我们检验了这一问题是否足够。我们使用157个匹配任务，分别以英语（EN）、自然孟加拉语（BN）以及以孟加拉语为基底、夹杂孟加拉语-英语语码转换（BN-CS）的形式编写，并对照在任何模型运行之前就已确定的必需信息单元（RIU），对来自三个模型（GPT-6.1 Sol、Qwen3.5-9B和Gemma 4 12B）的1,413个回答进行了评分。该评分方法将完全遗漏与“内容存在但不正确、不充分或仅部分满足”的情况区分开来。预先设定的主要估计量——关键遗漏率的领域等权配对差异（BN - EN）——未显示出可检测的差异（+0.003，95%置信区间为-0.014至+0.015），且所有条件下的遗漏率均接近于零。孟加拉语的差距反而体现在完成度上：孟加拉语的关键覆盖率低0.050（95%置信区间为0.022至0.079），严格的完整任务成功率低0.093。

    arXiv:2611.29299v1 Announce Type: cross  Abstract: Evaluations of multilingual trustworthiness often ask whether a model leaves out information that a user needs. We examine whether that question is sufficient. Using 157 matched tasks written in English (EN), natural Bengali (BN) and Bengali-matrix Bengali-English code-switching (BN-CS), we scored 1,413 responses from three models (GPT-6.1 Sol, Qwen3.5-9B and Gemma 4 12B) against required information units (RIUs) fixed before any model was run. The scoring separates outright omission from content that is present but incorrect, insufficient or only partly met. The prespecified primary estimand, the domain-equal paired difference in critical omission rate (BN - EN), showed no detectable difference (+0.003, 95% CI -0.014 to +0.015), with omission near zero in every condition. The Bengali deficit appeared in fulfillment instead. Critical coverage was 0.050 lower in Bengali (95% CI 0.022 to 0.079) and strict complete-task success 0.093 lowe
    
[^116]: 干预修复的是哪种失败？智能体干预何时有效，自进化技能又何时停滞

    Which Failure Does an Intervention Repair? When Agent Interventions Help, and Where Self-Evolving Skills Stall

    [https://arxiv.org/abs/2611.29290](https://arxiv.org/abs/2611.29290)

    该论文通过覆盖九大基准、四个模型与近90万条轨迹的匹配因子实验，系统揭示了智能体干预（技能、框架、推理预算）之间存在交互而非简单叠加效应，并证明应根据系统遗留的失败类型（知识缺失对应更强模型、执行错误对应更丰富框架）来选择修复手段。

    

    一项能改善某个智能体的干预措施可能会破坏另一个智能体，而在此基础上叠加更强的框架也未必能修复损害。技能、框架和额外推理预算等智能体干预措施，通常只被归功于在单一设置下测得的一个平均增益。我们转而追问：每种干预何时有效，以及它究竟修复了哪种失败。一项匹配的因子分析研究涵盖九个基准、四个模型、四种技能条件、四个框架层级和三种预算，共近90万条轨迹，每次只改变一个维度。结果显示：没有哪种配置在所有情况下都是最优的；干预措施之间是相互作用而非简单叠加；框架可能吸收技能的增益、挽救技能导致的失败，或使增益保持不变。当前系统遗留的失败类型指向了对应的修复手段：知识缺失应换用更强的模型，执行错误应采用更丰富的框架。然而反馈并不等于恢复：匹配的恢复机制主要修复的是那些留下可见……（原文在此处截断）

    arXiv:2611.29290v1 Announce Type: cross  Abstract: An intervention that improves one agent can break another, and adding a stronger harness on top need not repair the damage. Agent interventions such as skills, harnesses, and extra inference budget are nonetheless usually credited with one average gain, measured under one setup. We ask instead when each intervention helps and which failure it repairs. A matched factorial study over nine benchmarks, four models, four skill conditions, four harness levels, and three budgets, with nearly 0.9 million trajectories, changes one axis at a time. No configuration is best everywhere, interventions interact rather than add, and a harness can absorb a skill's gain, rescue a failure the skill causes, or leave the gain intact. The failure left by the current system points to the fix: missing knowledge to a stronger model, execution errors to a richer harness. Yet feedback is not recovery: a matched recovery repairs mainly failures that leave a visib
    
[^117]: 基于大语言模型的狗狗护理指导助手

    LLM-Based Assistant for Dog Care Guidance

    [https://arxiv.org/abs/2611.29286](https://arxiv.org/abs/2611.29286)

    研究通过对比实验发现，基于RAG检索增强的领域接地方法在提升大语言模型生成狗狗护理建议的准确性方面显著优于LoRA参数高效微调。

    

    本研究探讨了通过检索增强生成（RAG）进行领域接地以及使用低秩适应（LoRA）进行参数高效微调，能否提升大语言模型（LLM）生成的宠物护理建议的准确性和可靠性。我们精心构建了一个包含特定品种狗狗护理信息的数据集，并基于该数据集开发了两个系统：一个基于TinyLlama-1.1B-Chat-v1.0的RAG管道，以及一个使用LoRA适配的微调LLaMA-3.2-1B模型。两个系统在15个狗狗护理问题上进行了评估。RAG模型获得了2.0的满分平均分，优于微调模型（平均分=1.47）。统计分析结果，包括Welch's t检验、Mann-Whitney U检验和Wilcoxon符号秩检验（均p < 0.05），证实了该性能差距的统计显著性。我们的研究结果表明，在专业领域（宠物护理）中，基于检索的领域接地比单独进行微调能够产生更准确、更一致且更值得信赖的回答。

    arXiv:2611.29286v1 Announce Type: cross  Abstract: This study investigates whether domain grounding through Retrieval-Augmented Generation (RAG) and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) can enhance the accuracy and reliability of pet care advice generated by large language models (LLMs). We curated a dataset of breed-specific dog care information and used it to develop two systems: a RAG pipeline based on TinyLlama-1.1B-Chat-v1.0, and a fine-tuned LLaMA-3.2-1B model adapted using LoRA. Both systems were evaluated on 15 dog care questions. The RAG model achieved a perfect average score of 2.0, outperforming the fine-tuned model (mean = 1.47). Statistical analyses, including Welch's t-test, Mann-Whitney U, and Wilcoxon signed-rank tests (all p < 0.05), confirmed the significance of the performance gap. Our findings suggest that retrieval-based domain grounding leads to more accurate, consistent, and trustworthy responses than standalone fine-tuning in speciali
    
[^118]: 双侧不确定性估计：将数据不确定性与模型不确定性校准到共享概率空间

    Dual-Side Uncertainty Estimation: Calibrating Data and Model Uncertainty into a Shared Probability Space

    [https://arxiv.org/abs/2611.29284](https://arxiv.org/abs/2611.29284)

    提出DUE双侧不确定性估计方法，通过Dirichlet扰动生成词元不确定性序列并结合序列感知的Normal-Inverse-Gamma估计器，联合量化输入侧偶然不确定性与句子级认知不确定性，并将两者校准融合到统一的错误概率空间中。

    

    多模态用户意图分析旨在从异构的行为观测中推断潜在目标，但面临跨模态输入不匹配和输出语义错误的问题。现有不确定性估计器只关注输入可靠性或输出置信度，并且通常将词元不确定性简化为简单的统计量，忽视了结构化的依赖关系。我们提出DUE（双侧不确定性估计），联合估计输入侧的偶然不确定性和句子级认知不确定性。通过单次生成，对跨层表示施加Dirichlet扰动即可产生有序的词元不确定性序列。序列感知的Normal-Inverse-Gamma估计器利用该序列来量化认知不确定性及其估计的置信度。证据性多模态对齐则从含噪、不完整或不一致的输入中捕获偶然不确定性。DUE将两种异构的不确定性分数校准到共享的错误概率空间中，并对其进行顺序融合（原文摘要在此处截断）。

    arXiv:2611.29284v1 Announce Type: new  Abstract: Multimodal user intention analysis infers latent goals from heterogeneous behavioral observations but faces cross-modal input mismatch and output semantic errors. Existing uncertainty estimators focus on input reliability or output confidence and often reduce token uncertainty to simple statistics, overlooking structured dependencies. We propose DUE (Dual-Side Uncertainty Estimation), which jointly estimates input-side aleatoric and sentence-level epistemic uncertainty. From a single generation, Dirichlet perturbations of cross-layer representations produce an ordered token uncertainty sequence. A sequence-aware Normal-Inverse-Gamma estimator uses this sequence to quantify epistemic uncertainty and confidence in its estimate. Evidential multimodal alignment captures aleatoric uncertainty from noisy, incomplete, or inconsistent inputs. DUE calibrates both heterogeneous scores into a shared error-probability space and sequentially fuses th
    
[^119]: 论推理模型中回溯行为与潜在置信度方向之间的相互作用

    On the Interplay between Backtracking and Latent Confidence Directions in Reasoning Models

    [https://arxiv.org/abs/2611.29265](https://arxiv.org/abs/2611.29265)

    该论文通过提取模型内部潜在置信度的线性方向，实现了对推理模型在答案提案处是回溯还是接受答案的预测，揭示了内部不确定性表征与冗长回溯式思维链之间的联系。

    

    推理模型会对多个答案提案进行反复推敲，这常常因回溯行为而导致推理轨迹过长，即使模型其实已经得到了正确答案（在三个32B模型中，约87%的生成轨迹仅包含1个独特答案，但平均仍会做出约8个答案提案）。鉴于这种可能属于“装饰性”的思维链（CoT）推理，我们能否通过观察模型内部状态来判断模型接下来是会回溯还是会接受某个答案提案？基于模型对其不确定性保有内部表征这一假设，我们尝试提取该表征，并刻画其与答案提案处回溯行为之间的关系。通过在整个响应轨迹中查询模型在给出答案提案后口头表达的置信度，我们提取出了一个潜在置信度的线性方向，该方向能够有效区分高置信度与低置信度的口头表达。这一方向可以调控口头表达的置信度……

    arXiv:2611.29265v1 Announce Type: cross  Abstract: Reasoning models deliberate over multiple answer proposals, often resulting in excessively long traces due to backtracking, even when the correct answer has already been reached (Across three 32B models, ~87% of rollouts contain 1 unique answer yet still make ~8 answer proposals on average). Given this potentially "decorative" chain of thought (CoT) reasoning, can we look to the model internals to determine if the model will backtrack or commit to an answer proposal? Operating under the hypothesis that models maintain an internal representation of their uncertainty, we seek to extract this representation and characterize its relationship with backtracking behavior at answer proposals. By querying the model's verbalized confidence following answer proposals throughout a response trace, we extract a linear direction for latent confidence that separates high and low verbalized confidence. This direction can steer verbalized confidence as 
    
[^120]: NoeVera：知识能否指导实验，实验又能否反哺知识？

    NoeVera: Can Knowledge Guide Experiments and Experiments Refine Knowledge?

    [https://arxiv.org/abs/2611.29255](https://arxiv.org/abs/2611.29255)

    NoeVera是一个智能体式“第二大脑”，它将论文、代码、基准与实验结果整合为保留来源、持久化的三层知识库，通过决策感知检索、角色化智能体推理与模拟器调用，构建“知识指导实验、实验反哺知识”的闭环，以优化LLM服务系统的科研决策。

    

    随着对大语言模型（LLM）服务需求的日益增长，优化LLM服务系统正变得愈发复杂，瓶颈已不再是预测给定配置的性能表现。更困难的问题在于决定要研究什么：对于某个工作负载，哪些机制至关重要、哪些配置值得测试、以及一项决策还需要什么证据。在本工作中，我们提出了NoeVera，一个超越单次检索的智能体式“第二大脑”，它将论文、代码、基准测试和实验结果转化为一个持久化且保留来源可追溯性的知识库，该知识库包含三个层面：源证据、可复用的综合知识以及智能体记忆。决策感知检索会组装紧凑的证据包，各角色的专属智能体对这些证据进行推理，并在需要定量支持时调用模拟器（Vidur或GAUSS），而每项发现都会作为带有作用域的证据记录写回知识库，从而使知识能够指导实验。

    arXiv:2611.29255v1 Announce Type: new  Abstract: With the growing demand for large language model (LLM) services, optimizing LLM serving systems is becoming increasingly complex, and the bottleneck is no longer predicting how a given configuration will perform. The harder problem is deciding what to investigate: which mechanisms matter for a workload, which configurations are worth testing, and what evidence a decision still needs. In this work, we present NoeVera, an agentic Second Brain that goes beyond just one-shot retrieval by turning papers, code, benchmarks, and experimental results into a persistent, provenance-preserving knowledge base with three planes: source evidence, reusable synthesis, and agent memory. Decision-aware retrieval assembles compact evidence packs, role-specific agents reason over them and invoke a simulator (Vidur or GAUSS) whenever quantitative support is needed, and every finding is written back as a scoped evidence record, so knowledge guides experiments 
    
[^121]: 看见不等于行动：WARD 诊断纵向临床智能体中“证据-状态-行动”的差距

    Seeing Is Not Acting: WARD Diagnoses the Evidence-to-State-to-Action Gap in Longitudinal Clinical Agents

    [https://arxiv.org/abs/2611.29243](https://arxiv.org/abs/2611.29243)

    WARD 是一个纵向临床智能体评测基准，通过 4,560 个临床任务和纵向、交互两条赛道，系统诊断了模型在“证据识别→患者状态判断→及时安全行动”链条中的能力差距。

    

    用于纵向临床护理的大语言模型必须随着患者信息的积累而不断修正其评估和行动。然而，以孤立响应或最终任务结果为中心的基准测试，对证据使用、状态更新和行动选择等中间环节的失败所提供的洞察十分有限。我们提出了 WARD，一个用于评估临床智能体如何将不断积累的证据转化为患者状态判断以及及时、安全行动的基准。WARD 包含覆盖七个临床输出领域的 4,560 个临床任务，这些任务被组织成来自 40 条纵向临床轨迹（跨越 95 次住院）的 240 个检查点。其纵向赛道在受控的累积证据条件下评估这些任务；其交互式赛道则评估端到端的、由工具介导的临床管理，涵盖 271 个可验证的证据项目和 141 个参考状态转变。在 12 个具有可评分检查点输出的模型中，证据识别一致……（摘要在此处截断）

    arXiv:2611.29243v1 Announce Type: cross  Abstract: Large language models for longitudinal clinical care must revise their assessments and actions as patient information accumulates. However, benchmarks centered on isolated responses or final task outcomes provide limited insight into intermediate failures in evidence use, state updating, and action selection. We introduce WARD, a benchmark for evaluating how clinical agents translate accumulating evidence into patient-state judgments and timely, safe actions. WARD comprises 4,560 clinical tasks across seven clinical output domains, organized into 240 checkpoints from 40 longitudinal clinical trajectories spanning 95 hospitalizations. Its Longitudinal Track evaluates these tasks under controlled cumulative evidence. Its Interactive Track evaluates end-to-end, tool-mediated clinical management across 271 verifiable evidence items and 141 reference transitions. Across 12 models with scorable checkpoint outputs, evidence recognition consis
    
[^122]: 编辑引发的依赖失配：编码智能体中的传播遗漏、恢复与评估盲区

    Edit-Induced Dependency Mismatch: Propagation Omissions, Recovery, and Evaluation Blind Spots in Coding Agents

    [https://arxiv.org/abs/2611.29242](https://arxiv.org/abs/2611.29242)

    该论文揭示了编码智能体在迭代改代码时会依赖上下文中残留的过时依赖判断，导致缺陷传播并逃过基准测试评估，并通过盲测人工审计确认了此类“编辑引发的依赖失配”的真实存在、运行中的恢复现象以及现有评估体系的盲区。

    

    在迭代代码修改过程中，编码智能体可能在后续推理中继续依赖过时的依赖判断，而这些判断的文本仍固定保留在上下文中，从而导致后续推理出错，并使消费者缺陷未被原始任务评估发现，进而暴露了所测基准中的盲区。我们研究了在基准任务上收集的真实智能体执行轨迹：包括 deepseek-v4-flash 在 DeepSWE 和 SWE-bench Verified 上的运行，并以 Qwen3.8-Flash 在 FeatureBench 上的案例作为补充。聚焦于编辑引发的消费者义务，对 DeepSWE 运行结果的全语料盲测人工审计识别出了终端失配，其中包括任务范围仍存争议的诊断。在补充执行检查之后，所有保留的 DeepSWE 终端事件均获得了执行层面的支持。审计还表明，在同一运行中，轨迹内的恢复可以与终端遗漏并存。过程分析将……（原文摘要至此截断）

    arXiv:2611.29242v1 Announce Type: new  Abstract: During iterative code modification, coding agents may continue to rely in subsequent reasoning on outdated dependency judgments whose text remains fixed in the context, leading to errors in subsequent reasoning and leaving consumer defects undetected by the original task evaluations, thereby exposing blind spots in the tested benchmarks. We study actual agent execution trajectories collected on benchmark tasks: deepseek-v4-flash runs on DeepSWE and SWE-bench Verified, supplemented by Qwen3.8-Flash cases on FeatureBench. Focusing on edit-induced consumer obligations, a full-corpus blinded human audit of the DeepSWE runs identifies terminal mismatches, including diagnoses whose task scope remains disputed. Following supplementary execution checks, all retained DeepSWE terminal events have execution support. The audit also shows that recovery within a trajectory can coexist with terminal omissions in the same run. The process analysis links
    
[^123]: GameAnything：从任意灵感创造可玩的游戏世界

    GameAnything: From Any Inspirations to Playable Game Worlds

    [https://arxiv.org/abs/2611.29238](https://arxiv.org/abs/2611.29238)

    该论文提出了开源智能体平台GameAnything及其配套基准GameAnything-Bench，可将多模态灵感素材通过迭代开发转化为现有游戏的可玩扩展，并同时评估功能正确性与实际游戏体验质量。

    

    AI驱动的游戏开发具有将多样化的人类灵感转化为互动体验的潜力。然而，现有系统和基准主要聚焦于明确的开发规范，对适配多模态参考素材以及评估实际游戏体验的支持有限。我们提出了GameAnything，一个开源的智能体平台，能够通过迭代开发将多模态灵感转化为现有游戏的可玩扩展。GameAnything将代码生成智能体与独立的试玩评审智能体、持久化的开发状态以及可复用的经验记忆相结合，以支持持续的迭代改进。我们进一步提出了GameAnything-Bench基准，涵盖多样化的参考模态、游戏创作目标和游戏环境，并通过探索性试玩测试来同时评估功能正确性与游戏体验质量。对九种代码生成智能体配置的评估显示……

    arXiv:2611.29238v1 Announce Type: new  Abstract: AI-driven game development holds the potential to transform diverse human inspirations into interactive experiences. However, existing systems and benchmarks primarily focus on explicit development specifications, with limited support for adapting multimodal references and evaluating actual gameplay experiences. We introduce GameAnything, an open-source agentic platform that transforms multimodal inspirations into playable extensions of existing games through iterative development. GameAnything integrates coding agents with an independent playtesting critic, persistent development state, and reusable experience memory to support continuous refinement. We further present GameAnything-Bench, a benchmark spanning diverse reference modalities, game creation objectives, and gaming environments, with exploratory gameplay testing to assess both functional correctness and gameplay quality. Evaluations of nine coding agent configurations reveal t
    
[^124]: IteraSong：网页DAW中大语言模型引导的迭代式符号音乐创作

    IteraSong: LLM-Guided Iterative Symbolic Music Creation in a Web DAW

    [https://arxiv.org/abs/2611.29225](https://arxiv.org/abs/2611.29225)

    本文提出IteraSong，一个网页版DAW系统，通过MCP工具将自然语言指令映射为分层编辑操作，支持用户在大语言模型辅助下对持久化多轨项目进行迭代式符号音乐创作与实时精修。

    

    音乐创作本质上是迭代性的，然而许多符号音乐生成系统仍以生成或重新生成孤立的输出为核心。我们提出了IteraSong，这是一个用于大语言模型（LLM）引导的迭代式符号音乐创作的交互式网页版数字音频工作站（DAW）。与一次性生成系统不同，IteraSong支持在持久化的多轨项目上进行“生成-试听-编辑-精修”的循环。用户可以初始化一个带有完整标注的流行音乐草稿，然后通过自然语言指令逐步对其进行精修。大语言模型将用户请求映射到通过模型上下文协议（MCP）工具暴露的分层编辑原语动作空间，涵盖全局、轨道、段落以及小节/音符等多个层级。该演示使参会者能够用自然语言编辑多轨作品、应用确定性编辑和生成式编辑，并实时撤销/重做更改。

    arXiv:2611.29225v1 Announce Type: cross  Abstract: Music creation is inherently iterative, yet many symbolic generation systems still center on producing or regenerating isolated outputs. We present IteraSong, an interactive web-based Digital Audio Workstation (DAW) for LLM-guided iterative symbolic music creation. Unlike one-shot generation systems, IteraSong supports a generate-audition-edit-refine loop over a persistent multi-track project. Users initialize a fully labeled pop draft, then progressively refine it through natural-language instructions. The LLM maps requests to a hierarchical action space of editing primitives exposed via Model Context Protocol (MCP) tools, spanning global, track, section, and bar/note scopes. The demo enables attendees to edit multi-track compositions in natural language, apply deterministic and generative edits, and undo/redo changes in real time.
    
[^125]: 面向AI编程代理的直接反馈策略优化

    Direct Feedback Policy Optimization for AI Coding Agents

    [https://arxiv.org/abs/2611.29193](https://arxiv.org/abs/2611.29193)

    提出单轨迹强化学习算法DFPO，无需依赖稀疏终端奖励和重复采样即可直接训练AI编程代理，在同等条件下任务解决率超过GSPO。

    

    AI编程代理通过多轮交互来生成、修改和验证代码。现有的强化学习方法通常依赖稀疏的终端奖励，这使得难以评估中间决策对任务结果的影响，且组相对类方法还需要对每个任务进行重复采样。我们提出了直接反馈策略优化（DFPO），这是一种单轨迹强化学习算法，能够直接从单条轨迹中进行训练。在相同的基础模型和执行框架条件下，DFPO取得了比GSPO更高的任务解决率。

    arXiv:2611.29193v1 Announce Type: cross  Abstract: AI coding agents generate, modify, and validate code through multi-turn interactions. Existing reinforcement learning methods often rely on sparse terminal rewards, making it difficult to assess how intermediate decisions affect task outcomes, and group-relative methods also require repeated sampling for each task. We propose Direct Feedback Policy Optimization (DFPO), a single-trajectory reinforcement learning algorithm that trains directly from a single trajectory. DFPO achieves higher task resolution rates than GSPO under matched base models and execution harnesses.
    
[^126]: ContextFact：面向长期对话智能体的上下文事实记忆

    ContextFact: Remembering Facts in Context for Long-Term Conversational Agents

    [https://arxiv.org/abs/2611.29155](https://arxiv.org/abs/2611.29155)

    提出ContextFact方法，通过在记忆构建阶段保留事实的局部上下文并分离组织跨会话证据，改进长期对话智能体的事实记忆与证据检索效果。

    

    基于大语言模型构建的智能体必须在与用户的多次重复交互中保持连续性。然而，长期对话中的事实分散在各个对话轮次和会话之间，使得难以检索到带有足够上下文的正确证据。因此，此类对话的记忆系统必须在保存事实的同时保留其上下文，并使证据能够被有选择地访问。我们首先研究了记忆构建方式与检索单元边界如何影响问题的可回答性和证据的可访问性。通过阅读整个记忆库，我们发现存在仅靠检索选择无法弥补的缺口，这表明在记忆构建过程中必须保留部分上下文。在相同的唯一轮次预算下，相邻轮次分组能够提升证据的可访问性，而更宽的窗口并不能持续带来帮助。这促使我们将事实特定的上下文保留在局部，同时将跨会话的证据分开组织。我们提出了ContextFact，它将保留上下文的事实记忆（与跨会话证据组织相链接）……

    arXiv:2611.29155v1 Announce Type: cross  Abstract: Agents built on large language models must maintain continuity across repeated interactions with users. However, long-term conversations scatter facts across turns and sessions, making it difficult to retrieve the right evidence with enough context. Memory for such conversations must therefore preserve facts with their context while keeping evidence selectively accessible. We first study how memory construction and retrieval-unit boundaries affect answerability and evidence access. Reading the entire memory bank reveals gaps that cannot be resolved by retrieval selection alone, showing that some context must be preserved during memory construction. Adjacent-turn grouping improves evidence access at matched unique-turn budgets, while wider windows do not consistently help. This motivates keeping fact-specific context local while organizing cross-session evidence separately. We propose ContextFact, which links context-preserving fact mem
    
[^127]: 基于孟加拉语患者叙述的医学专科预测：机器学习、深度学习、Transformer与大语言模型的比较研究

    Medical Specialty Prediction from Bangla Patient Narratives: A Comparative Study of ML, DL, Transformers, and LLMs

    [https://arxiv.org/abs/2611.29151](https://arxiv.org/abs/2611.29151)

    该研究针对孟加拉语非正式患者叙述构建了医学专科预测任务，并系统比较了机器学习、深度学习、Transformer和大语言模型四类方法的表现，其中微调的XLM-RoBERTa以86.26%的准确率取得最佳效果。

    

    及时获得合适的医疗专科服务仍是孟加拉国医疗体系的瓶颈，因为患者通常用非正式的孟加拉语而非临床术语来描述症状，导致转诊效率低下。本文比较了四类模型在从孟加拉语患者叙述中预测医学专科方面的表现：使用混合TF-IDF特征的六种机器学习（ML）模型、使用基于语料库训练的Word2Vec词向量的四种深度学习（DL）模型、四种微调的Transformer模型，以及五种以零样本（zero-shot）方式评估的大语言模型（LLM）。语料库包含覆盖20个专科的3798条患者记录，我们保留了记录数最多的五个专科（内科、妇科、皮肤科、儿科、胃肠科），并将由此得到的2730条记录按80:20划分为所有模型家族共用的训练集和测试集。表现最佳的Transformer模型是XLM-RoBERTa（准确率86.26%，宏平均F1为86.21%），最佳的机器学习模型是SVM（84.98%，84.92%），最佳的深度学习模型……（原文摘要在此处被截断）

    arXiv:2611.29151v1 Announce Type: cross  Abstract: Timely access to appropriate medical specialists remains a bottleneck in Bangladesh's healthcare system, as patients describe symptoms in informal Bangla rather than clinical terms, leading to ineffective referrals. This paper compares four model families for predicting medical specialties from Bangla patient narratives: six machine learning (ML) models using hybrid TF-IDF features, four deep learning (DL) models using Word2Vec embeddings trained on the corpus, four fine-tuned transformers, and five large language models (LLMs) evaluated zero-shot. From a corpus of 3798 patient records covering 20 specialties, we retained the five with the most records (Medicine, Gynecology, Dermatology, Pediatrics, Gastroenterology) and split the resulting 2730 records 80:20 into training and test sets shared by all families. The best transformer was XLM-RoBERTa (86.26% accuracy, 86.21% macro-F1), the best ML model was SVM (84.98%, 84.92%), the best D
    
[^128]: VeriLogic：基于可验证逻辑推理轨迹的强化学习

    VeriLogic: Reinforcement Learning over Verifiable Logical Reasoning Traces

    [https://arxiv.org/abs/2611.29147](https://arxiv.org/abs/2611.29147)

    VeriLogic 提出了一种将逻辑推理过程分解为显式可验证的中间判断、证明和最终决策的强化学习方法，通过“扎根—优化—精炼”流水线实现过程级监督，解决了传统结果信号忽视推理轨迹内部正确性异质性的问题。

    

    逻辑推理通过显式的中间推理步骤从前提推导出结论，为过程级验证与监督提供了天然基础。然而，现有的学习方法通常依赖响应层面的结果信号，忽略了推理轨迹内部正确性的异质性：一个正确的答案可能包含无效的推理，而有效的中间步骤也可能出现在错误的响应中。为解决这一问题，我们提出了可验证逻辑推理强化学习，将内部推理轨迹转化为显式且可验证的学习对象。VeriLogic 不将响应视为不可分割的学习单元，而是将逻辑推理分解为显式的中间判断、证明和最终决策，并将它们组织成一个统一的“扎根—优化—精炼”流水线：（1）基于验证的推理环境构建

    arXiv:2611.29147v1 Announce Type: cross  Abstract: Logical reasoning derives conclusions from premises through explicit intermediate inference steps, providing a natural basis for process-level verification and supervision. However, existing learning methods often rely on response-level outcome signals, which overlook heterogeneous correctness within the reasoning trace: a correct answer may contain invalid reasoning, while valid intermediate steps may also appear in an incorrect response. To address this issue, we introduce Reinforcement Learning on Verifiable Logical Reasoning (VeriLogic), which turns internal reasoning traces into explicit and verifiable learning objects. Rather than treating a response as an indivisible learning unit, VeriLogic decomposes logical reasoning into explicit intermediate judgments, proofs, and final decisions, and organizes them into a unified "Grounding - Optimization - Refinement" pipeline: (1) Verification-Grounded Reasoning Environment Construction 
    
[^129]: 面向知识密集型领域的组平衡监督微调

    Group-Balanced Supervised Fine-Tuning for Knowledge-Intensive Domains

    [https://arxiv.org/abs/2611.29127](https://arxiv.org/abs/2611.29127)

    该论文提出组平衡监督微调（GB-SFT），通过在序列内不同语义角色的 token 之间均衡分配监督损失预算，在法律、医学、代码和数学等领域显著提升大模型新知识注入效果，同时保留引用召回能力。

    

    监督微调（SFT）对每个被监督的 token 赋予相同权重，因此在知识密集的语料上，损失预算是按 token 数量而非重要性分配的：在一个法律问答基准上，引用字符串占用了 81% 的监督损失权重预算，而真正要教授的答案仅占 5%。我们提出了组平衡 SFT（GB-SFT），它在一个目标序列内的不同 token 语义角色之间均衡分配损失预算。通过一个可调旋钮和一个结构性开关，GB-SFT 将标准 SFT、类级别和跨度级别的平衡包含为其几何上被精确刻画的端点。在法律、医学、代码和数学的基准上，我们跨多个模型家族地展示：(i) GB-SFT 比均匀 SFT、逐点 token 加权、记忆编辑以及匹配预算的数据混合能更好地注入新知识；(ii) 它达到了仅保留知识载荷而遮蔽其余内容的效果，同时保留了被遮蔽方法所破坏的引用召回能力。

    arXiv:2611.29127v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) weights every supervised token equally, so on knowledge-dense corpora the loss budget follows token count rather than importance: on a legal QA benchmark, citation strings absorb 81% of the supervised loss weight budget while the answers being taught absorb 5%. We present Group-Balanced SFT (GB-SFT), which equalizes the loss budget across the semantic roles of tokens within a target sequence. With one tunable dial and one structural switch, GB-SFT contains standard SFT, class- and span-level balancing as corners whose geometry we characterize exactly. On benchmarks in law, medicine, code, and mathematics, we show across model families that: (i) GB- SFT injects new knowledge better than uniform SFT, pointwise token weighting, memory editing, and matched-budget data mixing; (ii) it matches the injection of masking out everything but the knowledge payload while retaining the citation recall that masking destro
    
[^130]: 重连控制何时能测试LLM权重读取器？C2与资格协议

    When Can a Rewiring Control Test LLM Weight Readers? C2 and an Eligibility Protocol

    [https://arxiv.org/abs/2611.29120](https://arxiv.org/abs/2611.29120)

    论文提出C2反事实方法（置换网络最终线性读出层的列）及三条件资格协议，用于区分大语言模型是真正在“计算”权重还是仅识别权重空间“指纹”，但测试发现没有任何自然设置符合该协议资格。

    

    大型语言模型可以从原始权重预测小型神经网络的输出，但这种成功可能反映的是对权重空间规律（即“指纹”）的识别，而非真正的计算，而保持函数不变的对照组无法区分这两者。我们提出了C2，一种反事实方法，它置换网络最终线性读出层的列：所有其他参数，以及读出层的奇异值、范数和元素多重集均保持不变，而函数可以以完全可重新计算的方式发生变化；我们验证了每个被评估的置换确实如此。C2仅在三个条件下才能区分指纹识别与计算，我们将这些条件表述为一个资格协议：一种仅限于C2不变摘要的策略必须已经能够预测目标；C2必须改变特定于重连的属性；并且读取器的身份必须经过认证。在我们测试的所有自然设置中，没有一个符合资格。在小型MLP上，前沿读取器路径跟随了重连后的（摘要在此处截断）

    arXiv:2611.29120v1 Announce Type: cross  Abstract: Large language models can predict the outputs of small neural networks from raw weights, but success could reflect recognition of weight-space regularities (fingerprints) rather than computation, and function-preserving controls cannot tell these apart. We propose C2, a counterfactual that permutes the columns of a network's final linear readout: every other parameter, and the readout's singular values, norms and entry multiset, are unchanged, and the function can change in an exactly recomputable way; we verify that each evaluated permutation does. C2 separates fingerprinting from computation only under three conditions, which we state as an eligibility protocol: a strategy restricted to C2-invariant summaries must already predict the target; C2 must change a property specific to the rewiring; and the reader's identity must be attested. No natural setting we tested is eligible. On tiny MLPs a frontier reader route followed the rewired
    
[^131]: 通过Koopman子空间解码Transformer动力学

    Decoding Transformer Dynamics via Koopman Subspaces

    [https://arxiv.org/abs/2611.29113](https://arxiv.org/abs/2611.29113)

    该论文将Transformer隐藏状态的逐层演化建模为动力系统，利用Koopman算子理论提取主导模态子空间作为主题“谱指纹”，证明模型深度映射近似线性、指纹可判别不同学科主题且在多个模型间可复现。

    

    我们应用Koopman算子理论和动态模态分解（DMD）来刻画大型语言模型（LLMs）如何处理不同的输入主题。我们将隐藏状态的逐层演化视为一个动力系统，拟合一个线性Koopman算子，并将其主导模态张成的子空间作为每个主题的谱指纹。研究得出四项发现：第一，深度映射接近线性——单个线性算子即可预测每个下一层状态，相对误差约为18%（内部方差解释率约为0.99）；第二，在MMLU的57个学科上，该指纹具有主题判别性，与MMLU分类体系的最近邻纯度匹配度为0.72，与各学科语义邻域的any-hit@3达到0.75；第三，它是一种可复现的签名（折半自检索约为0.97）；最后，这些发现在七个基础模型（Qwen2.5 0.5B-7B、Llama-3 1B-8B）上保持一致，并呈现出一个具有提示意义的趋势（原文在此处截断）。

    arXiv:2611.29113v1 Announce Type: cross  Abstract: We apply Koopman operator theory and Dynamic Mode Decomposition (DMD) to characterise how large language models (LLMs) process different input topics. Treating the layer-to-layer evolution of hidden states as a dynamical system, we fit a linear Koopman operator and take the subspace spanned by its dominant modes as a per-topic spectral fingerprint. Four findings emerge. First, the depth map is near-linear: a single linear operator predicts each next-layer state to within approx. 18% relative error (interior variance explained approx. 0.99). Second, over the 57 MMLU subjects the fingerprint is topic-discriminative, matching the MMLU taxonomy at 0.72 nearest-neighbour purity and each subject's semantic neighbourhood at 0.75 any-hit@3. Third, it is a reproducible signature (split-half self-retrieval approx. 0.97$). Finally, these findings are consistent across seven base models (Qwen2.5 0.5-7B, Llama-3 1-8B), with a suggestive trend of al
    
[^132]: 超越自由形式生成：测量竞赛级NLP系统中的接口组合

    Beyond Free-Form Generation: Measuring Interface Composition in Competitive NLP Systems

    [https://arxiv.org/abs/2611.29103](https://arxiv.org/abs/2611.29103)

    该研究通过对Kaggle和共享任务中数百个竞赛NLP系统的标注分析，首次量化了混合型系统（自由生成与受限接口结合）的占比差异，并识别出多种精细的接口配置模式。

    

    配备解码器大语言模型（LLM）的系统可以将自由生成与评分、排序或选择等操作相结合。我们使用与源证据相关联的五个标注字段来测量这些有据可查的接口组合。语料库包含来自42个涉及NLP的Kaggle竞赛的517个解决方案，以及来自26个SemEval、TREC和WMT任务年度的76个参赛系统。在符合条件的配备解码器LLM的生成系统中，55个Kaggle系统中有18个、20个共享任务系统中有17个属于混合型系统：自由生成与受限接口共存，其余系统则仅使用自由生成。当每个竞赛或任务年度被赋予相同权重时，混合型系统的占比分别为0.327和0.876。在Eedi竞赛中，21个符合条件的配备解码器LLM的系统呈现五种精确的接口配置；十个LLM Science Exam系统共享一个接口类别并呈现四种配置。在整个Kaggle范围内，符合条件的配备解码器LLM的系统跨越18种精确的接口配置，揭示了广泛任务类型内部的多样性。

    arXiv:2611.29103v1 Announce Type: new  Abstract: Systems with decoder LLMs can combine free generation with scoring, ranking, or selection. We measure these documented interface combinations with five annotation fields linked to source evidence. The corpora comprise 517 solutions from 42 Kaggle competitions involving NLP and 76 participant systems from 26 SemEval, TREC, and WMT task years. Among eligible generation systems with decoder LLMs, 18 of 55 Kaggle systems and 17 of 20 Shared Task systems are hybrid: free generation coexists with a constrained interface. The remaining systems use free generation alone. Giving each competition or task year equal weight yields hybrid shares of 0.327 and 0.876. Within Eedi, 21 eligible systems with decoder LLMs have five exact interface profiles; ten LLM Science Exam systems share one interface class and have four profiles. Across Kaggle, the eligible systems with decoder LLMs span 18 exact interface profiles, revealing variation within broad tas
    
[^133]: 作为“活文档”的运维手册：辅助式事后修订下文档演进的纵向研究

    Operational Runbooks as Living Documents: A Longitudinal Study of Document Evolution under Assisted Post-Incident Revision

    [https://arxiv.org/abs/2611.29097](https://arxiv.org/abs/2611.29097)

    该论文首次对生产环境中运维手册作为“活文档”的长期演进进行了纵向实证研究，并提出利用轻量级的语言学与结构信号来量化与监控文档随时间的变化。

    

    运维手册在运维工作流中日益发挥着“活文档”的作用：它们由人员维护、用于事件支持，并随着组织知识的变化而持续修订。然而，目前鲜有研究了解这类文档集合在生产环境中如何随时间演进，以及哪些可解释的信号对于监控文档变化是有用的。我们分析了17周内版本控制的运维手册快照，这些快照产生于机器辅助审查与人工监督之下的事后修订过程。我们基于先前关于程序性语言、技术沟通和文档组织的研究，采用轻量级的语言学与结构信号来量化文档演进。相关性分析显示出混合的关联模式，包括正相关与负相关关系，其效应量从可忽略到中等不等，并在不同语料库之间存在差异。时间维度的摘要分析进一步区分了……

    arXiv:2611.29097v1 Announce Type: new  Abstract: Operational runbooks increasingly function as living documents within operational workflows: they are maintained by people, used in incident support, and continuously revised as organizational knowledge changes. Yet little is known about how such document collections evolve over time in production settings, or which interpretable signals are useful for monitoring document change. We analyze 17 weeks of version-controlled runbook snapshots produced during post-incident revision with machine-assisted review and human oversight. We quantify document evolution using lightweight linguistic and structural signals grounded in prior work on procedural language, technical communication, and document organization. Correlation analysis shows mixed patterns of association, including both positive and negative relationships, with effect sizes ranging from negligible to moderate and varying across corpora. Temporal summaries further distinguish compar
    
[^134]: RES：利用滚动熵采样加速扩散语言模型

    RES: Accelerating Diffusion Language Models with Rolling Entropy Sampling

    [https://arxiv.org/abs/2611.29066](https://arxiv.org/abs/2611.29066)

    提出滚动熵采样器（RES），通过在达到最小长度后提交连续的已接受前缀、同时保留前方位置的修正空间，兼顾了准确率与计算效率，从而加速扩散语言模型。

    

    扩散语言采样器需要为其去噪窗口中的每个位置做出两个选择：下一个词元来自哪里，以及该位置何时被提交到 KV 缓存中。我们利用 DiffusionGemma 中的熵约束采样器，研究了第二个选择的两种极端情况。一旦词元被接受就立即提交，会使其失去被修正的机会，当熵约束被放宽以用准确率换取速度时，这种做法会造成损害：当被接受的词元被冻结时，随着约束的放宽，宏观准确率从 73% 下降到 20%，而不冻结的采样器则保持在 70% 左右。另一方面，一次性提交整个块会浪费计算资源，因为词元在很大程度上是按从左到右的顺序收敛的，并且在二十个去噪步骤内，窗口中超过一半的位置已经持有最终确定的词元。这些发现促成了滚动熵采样器（RES）的提出：该方法在前缀达到最小长度时提交一段连续的已接受前缀，同时保持更靠前的位置开放以便修正。

    arXiv:2611.29066v1 Announce Type: cross  Abstract: A diffusion language sampler makes two choices for every position in its denoising window: where the next token comes from, and when the position is committed to the KV cache. Using the entropy-bound sampler in DiffusionGemma, we study two extremes of the second choice. Committing each token as soon as it is accepted removes the chance to revise it, which hurts when the entropy bound is relaxed to trade accuracy for speed. When accepted tokens are frozen, macro accuracy falls from 73% to 20% as the bound is relaxed, while the unfrozen sampler stays near 70%. Committing a whole block at once wastes computation, since tokens converge in a largely left-to-right order and more than half of the window holds already-final tokens by twenty denoising steps. These findings motivate the Rolling Entropy Sampler (RES), which commits a contiguous accepted prefix once it reaches a minimum length and keeps positions farther ahead open to revision. Un
    
[^135]: HyperHarness：从用户历史中学习生成个性化智能体框架

    HyperHarness: Learning to Generate Personalized Agent Harnesses from User Histories

    [https://arxiv.org/abs/2611.29062](https://arxiv.org/abs/2611.29062)

    HyperHarness通过学习一个共享的“历史到框架”生成器，直接从用户历史合成可执行的个性化检索与答案引导代码，兼顾了共享工作流的部署效率与逐用户定制的行为适配能力。

    

    个性化LLM智能体依赖于一个智能体框架（即围绕模型的记忆与控制代码）来将用户历史与下游决策连接起来。现有方法通常采用两种方式之一：一种是在所有用户间共享单一的记忆工作流，虽然部署高效，但无法针对多样化的历史记录调整证据的检索与解读方式；另一种是对每个用户单独进行框架优化，虽能为每位用户定制行为，但在部署时需要代价高昂的反复候选评估，且原始日志中的监督信号稀疏且充满噪声。为解决这一权衡问题，我们提出了HyperHarness，该框架学习一个共享的“历史到框架”生成器，可直接从用户历史中合成可执行的、用户特定的检索与答案引导代码。给定用户的授权历史，HyperHarness会提示一个构建型LLM来编写个性化的检索与引导函数，同时利用训练用户的下游任务得分进行监督……（摘要在此处截断）

    arXiv:2611.29062v1 Announce Type: new  Abstract: Personalized LLM agents rely on an agent harness, the surrounding memory and control code, to connect user history with downstream decisions. Existing methods typically employ either a single shared memory workflow across all users, which is efficient to deploy but cannot adapt how evidence is retrieved and interpreted across diverse histories, or per-user harness optimization, which customizes behavior for each user but requires costly repeated candidate evaluation at deployment and suffers from sparse, noisy supervision in raw logs. To address this trade-off, we present HyperHarness, a framework that learns a shared history-to-harness generator to synthesize executable, user-specific retrieval and answer-guidance code directly from user histories. Given a user's permitted history, HyperHarness prompts a construction LLM to write personalized retrieval and guidance functions, while downstream task scores across training users supervise 
    
[^136]: 基于群体认知锚定的协作智能体

    Grounding Collaborative Agents in Group Cognition

    [https://arxiv.org/abs/2611.29036](https://arxiv.org/abs/2611.29036)

    该论文提出了“群体简报”这一新任务来重建团队中与任务相关的群体认知（包括成员信念、群体动态和信息缺口），并构建了包含1,351个请求与9,975个参考点的四领域基准及两阶段评估流程，同时推出了GroupBrief方法。

    

    加入正在进行的团队的协作智能体，需要依靠共享上下文将多方交互历史与下游协调决策连接起来。现有工作通常要么专注于从对话日志中检索历史事实，这只能呈现孤立的证据，却无法捕捉谁知道什么或负责什么；要么评估端到端的交互，这将准备工作与执行混为一谈，并且由于团队历史没有单一的“标准摘要”，缺乏直接的监督信号。为弥合这一差距，我们提出了“群体简报”任务，即重建与任务相关的群体认知，涵盖参与者的表达信念、群体动态以及信息锚定缺口，并构建了一个包含1,351个请求和9,975个已验证参考点的四领域基准，配套了一个结合参考点覆盖率评估与下游协作模拟的两阶段评估流程。利用该流程，我们展示了GroupBrief，一个简单的任务

    arXiv:2611.29036v1 Announce Type: new  Abstract: Collaborative agents joining an ongoing team rely on shared context to connect multi-party interaction histories with downstream coordination decisions. Existing work typically focuses either on retrieving historical facts from conversation logs, which surfaces isolated evidence without capturing who knows or owns what, or on evaluating end-to-end interactions, which conflates preparation with execution and lacks direct supervision because a team's history has no single ground-truth summary. To bridge this gap, we formulate group briefing, the task of reconstructing task-relevant group cognition across participants' expressed beliefs, group dynamics, and grounding gaps, and introduce a four-domain benchmark of 1,351 requests and 9,975 verified reference points paired with a two-stage evaluation pipeline combining reference-point coverage and downstream collaboration simulation. Using this pipeline, we show that GroupBrief, a simple task-
    
[^137]: LITMUS——基于文献的用例映射与风险卡综合测试平台

    LITMUS - Literature Informed Testbed for Mapping Usecases and Synthesizing Risk-Cards

    [https://arxiv.org/abs/2611.29034](https://arxiv.org/abs/2611.29034)

    本文提出LITMUS，一个结构化流水线，能够根据给定用例自动生成定制化数据集与风险卡，并利用同行评审文献提供可追溯证据，将每个风险与具体用例紧密关联，弥补了现有方法忽视用例场景的不足。

    

    大语言模型（LLM）和人工智能（AI）在现实世界应用中的快速普及，加剧了对系统化、面向安全的部署框架的需求。现有方法通常优先围绕LLM和AI架构本身来构建数据集和风险卡，却很少考虑驱动模型选择的特定用例。这种疏漏是有问题的，因为任何实际的AI部署都始于详细的用例描述，而该描述正是评估上下文风险的基础。在本文中，我们提出了一个结构化流水线，给定一个用例，它能够自动生成定制的数据集以及配套的风险卡。该流程确保所识别的风险与用例紧密耦合，并提供来自同行评审文献的可追溯证据，以证实每一个风险与用例之间的关系。先前的声明验证方法（FActSco...

    arXiv:2611.29034v1 Announce Type: new  Abstract: The proliferation of large language models (LLMs) and Artificial Intelligence (AI) in real-world applications has heightened the need for systematic, safety-oriented deployment frameworks. Existing approaches typically prioritize the construction of datasets and risk cards around the LLM and AI architecture itself, yet they provide scant consideration for the specific use-case that drives model selection. This omission is problematic because any practical AI deployment begins with a detailed use-case description that informs the assessment of contextual hazards. In this paper, we introduce a structured pipeline that, given a use-case, automatically produces a tailored dataset and an accompanying risk card. The process ensures that identified risks are tightly coupled to the use-case, and it offers traceable evidence from peer-reviewed literature to substantiate each risk to use-case relationship. Prior claim-verification methods (FActSco
    
[^138]: 面向大语言模型的基于种群的后训练方法

    Population-Based Post-Training for Large Language Models

    [https://arxiv.org/abs/2611.29014](https://arxiv.org/abs/2611.29014)

    提出基于种群的后训练方法PBPT，通过并行训练多个模型并利用种群机制自动调整超参数，在推理和工具使用基准上以相同计算量超越了独立RLVR训练的效果。

    

    可验证奖励的强化学习（RLVR）是大语言模型最流行的后训练范式之一。RLVR通常在训练开始前就选定的超参数下训练单个模型，而研究人员则需要通过连续多次运行来调整超参数。我们提出了基于种群的后训练方法（PBPT），该方法并行训练多个模型，并通过基于种群的训练方式让训练过程自动完成超参数调整。PBPT定期在验证集上评估各个模型，表现优异的模型会被复制并对其超参数施加扰动以探索新的配置；表现不佳的模型则停止训练并被这些副本取代。在推理基准（Enigmata、Nemotron）和交互式工具使用环境（Gaia2、AppWorld）上，PBPT产生的智能体比计算量相当的独立RLVR学习者群体表现更强。此外，PBPT能够从较差的初始超参数中恢复，自动……

    arXiv:2611.29014v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) is one of the most popular post-training paradigms for large language models. RLVR typically trains a single model under hyperparameters chosen before training begins, while researchers tune hyperparameters across consecutive runs. We introduce Population-Based Post-Training (PBPT), which trains several models in parallel and lets the training process do the tuning through population-based training. PBPT periodically evaluates models on a validation set. High-performers are duplicated and their hyperparameters perturbed to explore new configurations. Low-performers stop training and are replaced by these copies. On reasoning benchmarks (Enigmata, Nemotron) and interactive tool-use environments (Gaia2, AppWorld), PBPT produces stronger agents than compute-matched groups of independent RLVR learners. Furthermore, PBPT is able to recover from poor initial hyperparameters, automatically
    
[^139]: 当LLM猜测数据模型时当心陷阱：面向Text-to-SQL及其基准测试的粒度感知扇出错误防护

    Mind the Trap As LLMs Guess the Data Model: Grain-Aware Guards Against Fan-out in Text-to-SQL and Its Benchmarks

    [https://arxiv.org/abs/2611.29009](https://arxiv.org/abs/2611.29009)

    该论文揭示了LLM驱动的Text-to-SQL系统因一对多连接导致的扇出聚合错误这一经典陷阱，发现广泛使用的BIRD基准中有4.2%的标准答案被此类错误扭曲，并发布了修正后的BIRD-FanOut数据集和GrainTraps合成数据集以帮助防范这类错误。

    

    基于大语言模型（LLM）构建的Text-to-SQL系统经常在一对多和多对多连接上进行聚合，这是一个经典的陷阱：当连接后的行重复同一实体时，SUM或COUNT会对其进行重复计数，但查询仍能正常运行并返回一个看似合理的数字。我们通过聚合所读取关系上的一个条件来刻画这类扇出（fan-out）错误，该条件依赖于模式中并未显示的两个事实：每个度量的粒度，以及每个键上的连接会重复多少行。在数据上检验该条件的审计发现，广泛使用的基准测试BIRD的开发集问题中有4.2%的标准答案被此类错误扭曲；而BIRD后续的独立修订版本更改了审计中标记为确定性扇出错误的答案的44%，相比之下未标记的答案仅有16%被更改。我们发布了BIRD-FanOut数据集（包含486个经过审核标准答案的BIRD问题）以及GrainTraps数据集（专门设置此陷阱及相关陷阱的合成问题）。在八个LLM智能体上的实验表明，从……计算得到的函数依赖（FD）（摘要此处不完整）

    arXiv:2611.29009v1 Announce Type: cross  Abstract: Text-to-SQL systems built on large language models (LLMs) often aggregate over one-to-many and many-to-many joins, a classic trap: when the joined rows repeat an entity, a SUM or COUNT counts it more than once, yet the query runs and returns a plausible number. We characterize these fan-out errors by one condition on the relation an aggregate reads, which depends on two facts a schema does not show: each measure's grain, and how many rows a join on each key repeats. An audit that tests the condition on the data finds distorted gold answers for 4.2% of the dev questions of BIRD, a widely used benchmark, and BIRD's later, independent revision changed 44% of the answers the audit flags as definite fan-out, against 16% of unflagged ones. We release BIRD-FanOut, 486 BIRD questions with reviewed gold, and GrainTraps, synthetic questions that set this trap and related ones. Across eight LLM agents, functional dependencies (FDs) computed from 
    
[^140]: NoHairCache：基于单一幂律指数的一次性校准KV缓存驱逐方法

    NoHairCache: Calibrate-Once KV Cache Eviction with One Power-Law Exponent

    [https://arxiv.org/abs/2611.29006](https://arxiv.org/abs/2611.29006)

    该论文证明KV缓存驱逐中的衰减指数α是模型内在属性、只需在单一语料上校准一次即可跨领域迁移，并将H2O与StreamingLLM两大主流方法统一为同一幂律评分族的两个端点。

    

    KV缓存驱逐中的衰减指数α是模型的内在属性：只需在单一语料库上校准一次，即可直接跨领域迁移使用。基于注意力累积的方法（H2O）和基于新近性的滑动窗口方法（StreamingLLM）是两种主流的驱逐框架，通常被视为互不相关，但我们证明它们其实是同一个幂律族的两个端点：通过为每个token评分 s_i = A_i · (t-i)^{-α}，其中A_i为累积注意力，α控制时间衰减惩罚，H2O和StreamingLLM分别对应于α=0和α→∞两种极限情况（StreamingLLM还额外硬编码了注意力汇聚补丁）。no-hair机制限定了α的允许取值范围，且实测指数与之吻合：在WikiText上校准一次后，代表性模型（Qwen2.5-1.5B-Instruct）得到的范围为[0.556, 0.822]。各模型的指数范围在不同领域间保持一致……

    arXiv:2611.29006v1 Announce Type: cross  Abstract: The decay exponent $\alpha$ in KV-cache eviction is an intrinsic property of the model: calibrated once on a single corpus, it transfers directly across domains. Attention-based accumulation (H2O) and recency-based sliding windows (StreamingLLM) are two dominant eviction frameworks, often treated as unrelated, but we show that they are the two endpoints of a single power-law family: by scoring each token as $s_i = A_i \cdot (t-i)^{-\alpha}$, where $A_i$ is cumulative attention and $\alpha$ controls the age penalty, H2O and StreamingLLM correspond to $\alpha=0$ and $\alpha\to\infty$, respectively (with StreamingLLM additionally hard-coding an attention-sink patch).   The no-hair mechanism of \citet{xiang2026nohair} bounds the admissible range of $\alpha$, and the measured exponents agree: calibrating once on WikiText yields $[0.556, 0.822]$ for a representative model (Qwen2.5-1.5B-Instruct). The per-model ranges are consistent across fi
    
[^141]: DoC：学习延迟或提交以实现及时、准确且稳定的对话式诊断预测

    DoC: Learning to Defer or Commit for Timely, Accurate, and Stable Conversational Diagnosis Prediction

    [https://arxiv.org/abs/2611.29003](https://arxiv.org/abs/2611.29003)

    该论文提出两阶段框架DoC，通过迭代提交边界估计与延迟感知的强化学习，使模型学会在适当时机延迟或提交诊断预测，无需提交时间标注即可实现更及时、准确且稳定的对话式诊断预测。

    

    对话式诊断预测要求从正在进行的临床对话中做出准确、早期且稳定的预测，然而对话级标签并未指明何时应当提交预测。我们提出了一个学习延迟决策与诊断决策的两阶段框架。首先，迭代式提交边界估计在无需标注提交时间的情况下构建“延迟/诊断”监督信号；其次，通过延迟感知的监督微调初始化策略，并利用轨迹级奖励的强化学习进行优化，奖励涵盖及时性、首次提交准确率和提交后稳定性。在EMSDialog和MediTOD数据集上，DoC能够尽早提交预测，在经过后训练的方法中取得最高的首次提交准确率（76.7%/51.2%），并拥有总体最低的提交后错误率（16.4%/48.3%）。

    arXiv:2611.29003v1 Announce Type: cross  Abstract: Conversational diagnosis prediction requires accurate, early, and stable predictions from ongoing clinical dialogue, yet dialogue-level labels do not specify when to commit. We propose a two-phase framework for learning deferral and diagnosis decisions. First, iterative commitment-boundary estimation constructs Defer/diagnosis supervision without annotated commitment times. Second, Defer-aware supervised fine-tuning initializes a policy, refined through reinforcement learning with trajectory-level rewards for timeliness, first-commit accuracy, and post-commit stability. On EMSDialog and MediTOD, DoC commits early, achieving the highest first-commit accuracy among post-trained methods (76.7%/51.2%) and the lowest post-commit error overall (16.4%/48.3%).
    
[^142]: 引导格式改变医学视觉-语言模型的零样本分类准确率：一项基于结直肠组织病理学的双中心评估

    Elicitation format changes zero-shot classification accuracy in a medical vision-language model: a two-center evaluation on colorectal histopathology

    [https://arxiv.org/abs/2611.28997](https://arxiv.org/abs/2611.28997)

    仅改变对医学视觉-语言模型的提问格式就能大幅改变其零样本分类准确率——在结直肠组织病理学九分类任务上，要求输出全部九类概率且禁止推理使准确率下降约14个百分点，而在相同类别基数与token预算下允许模型进行推理并给出选定答案可恢复近19个百分点的准确率。

    

    背景。托管式医学视觉-语言模型的言语化置信度通常被视为围绕不变预测结果的“包装”。方法。我们保持九分类结直肠组织病理学提示词固定，仅改变所请求的响应格式，并在10,004个验证集图像块和7,180个外部中心PathMNIST图像块上评估MedGemma-27b-it，采用预注册的终点指标、随机化平局处理，并为每个图像块集合重新计算随机下限。结果。与要求输出排序前三名相比，要求输出全部九类概率在验证集上使准确率下降14.8个百分点（-0.1475 [-0.1594, -0.1353]），在外部中心下降13.4个百分点（-0.1341 [-0.1484, -0.1196]），而随机猜测准确率仅为0.1111。九类提示词独特地禁止了评论、推理和选定答案；在匹配的类别基数和token预算下移除该禁令可恢复+0.1878 [+0.1623, +0.2134]和+0.1911 [+0.1648, +0.2181]，而改变类别基数……（摘要原文在此处截断）

    arXiv:2611.28997v1 Announce Type: cross  Abstract: Background. Verbalized confidence from a hosted medical vision-language model is usually treated as packaging around an unchanged prediction. Methods. We held a nine-class colorectal histopathology prompt fixed except for the requested response format and evaluated MedGemma-27b-it on 10,004 validation and 7,180 external-center PathMNIST patches, with preregistered endpoints, randomized ties, and random floors recomputed for each patch set. Results. Requesting probabilities over all nine classes instead of a ranked top-3 reduced accuracy by 14.8 points on validation, -0.1475 [-0.1594, -0.1353], and 13.4 at the external center, -0.1341 [-0.1484, -0.1196], against chance accuracy of 0.1111. The nine-class prompt uniquely forbade commentary, reasoning, and a chosen answer; removing that prohibition at matched cardinality and token budget recovered +0.1878 [+0.1623, +0.2134] and +0.1911 [+0.1648, +0.2181], while changing cardinality with th
    
[^143]: EmoRouter：面向多模态情感识别的自适应模态级联方法

    EmoRouter: Adaptive Modality Cascading for Multimodal Emotion Recognition

    [https://arxiv.org/abs/2611.28960](https://arxiv.org/abs/2611.28960)

    提出了一种无需训练、与模型无关的EmoRouter框架，利用基于共形预测的不确定性估计，自适应地为每条话语选择模态级联的退出层级，在提升多模态情感识别准确率的同时降低计算开销。

    

    多模态情感识别（MER）旨在检测对话中每句话背后所蕴含的情感，这对于具备情感感知能力的人机交互至关重要。尽管全模态大语言模型通过联合处理文本、音频和视觉等多种模态，在多模态情感识别任务上展现出强大的能力，但不同的模态组合在不同的语句上会产生不同的预测结果：声学和视觉输入可以帮助找回仅用文本推理时遗漏的标注标签，但也可能将原本正确的预测变为错误。此外，对每一句话都处理所有模态在计算上也十分昂贵。为此，我们提出了EmoRouter，一个无需训练、与模型无关的自适应模态级联框架，它沿着预定义的模态顺序为每句话选择一个退出层级。该路由过程由基于共形预测得出的预测不确定性估计所引导。

    arXiv:2611.28960v1 Announce Type: new  Abstract: Multimodal Emotion Recognition (MER) involves detecting the underlying emotion behind each utterance within a conversation, which is essential for emotionally aware human-computer interaction. While omni-modal large language models (Omni-LLMs) demonstrate strong capability for MER by jointly processing modalities including text, audio, and vision, different modality combinations yield different prediction outcomes across utterances: acoustic and visual inputs can help recover annotated labels missed by text-only inference, but can also turn correct predictions into errors. Processing all modalities for every utterance is also computationally expensive. To this end, we introduce EmoRouter, a training-free, model-agnostic framework for adaptive modality cascading that selects an exit tier for each utterance along a predefined modality order. The routing is guided by prediction-uncertainty estimates derived from Conformal Prediction. EmoRou
    
[^144]: 不要沉睡，持续守望：面向计算机使用代理的主动感知

    Don't Sleep, Keep Watch: Proactive Perception for Computer-Use Agents

    [https://arxiv.org/abs/2611.28947](https://arxiv.org/abs/2611.28947)

    提出NightWatch，一个即插即用框架，通过耦合轻量级System-1模型NightWatch-0.8B对界面状态进行流式监控，为计算机使用代理提供持续的实时感知，从而弥合因VLM推理延迟导致的感知与行动之间的时间鸿沟。

    

    基于视觉-语言模型构建的计算机使用代理（CUA）越来越多地被委托在实时桌面、网页和移动环境中执行长时程任务，而这些环境中屏幕会自行持续变化。然而，当前的CUA缺乏实时态势感知能力：它们通过离散的快照来感知环境，VLM巨大的推理延迟加上高度动态的界面，造成了感知与行动之间的鸿沟——观察到的状态在相应动作执行之前经常已经过时。为了弥合这一鸿沟，我们提出了NightWatch，一个即插即用的框架，能够在不改变代理的模型、提示或动作空间的情况下，赋予任何CUA持续的、具备时间定位能力的感知。NightWatch将VLM代理与NightWatch-0.8B（一个轻量级的System-1模型）相耦合，该模型对界面状态进行流式监控，并以较低的计算开销使代理与不断演进的环境保持同步。

    arXiv:2611.28947v1 Announce Type: new  Abstract: Computer-use agents (CUAs) built on vision-language models are increasingly entrusted with long-horizon tasks in live desktop, web, and mobile environments, where the screen keeps changing on its own. However, current CUAs lack real-time situational awareness: they perceive the environment through discrete snapshots, and the substantial inference latency of VLMs, compounded by highly dynamic interfaces, opens a perception-action gap in which the observed state frequently becomes stale before the corresponding action is executed. To bridge this gap, we present NightWatch, a plug-and-play framework that endows any CUA with continuous, temporally grounded perception without changing the agent's model, prompt, or action space. NightWatch couples the VLM agent with NightWatch-0.8B, a lightweight System-1 model that performs streaming monitoring of the interface state and keeps the agent synchronized with the evolving environment at low comput
    
[^145]: 当评分标准无法决断时：LLM评估中的静默价值补全

    When Rubrics Do Not Decide: Silent Value Completion in LLM Evaluation

    [https://arxiv.org/abs/2611.28940](https://arxiv.org/abs/2611.28940)

    本文提出TradeoffBench基准，揭示多准则LLM评判者在准则冲突时会在未声明权衡前提的情况下静默补全价值判断，导致在9%-81%的冲突中选出缺乏依据的胜者。

    

    多准则LLM评判者在评估响应时，通常不会指明相互冲突的准则应如何权衡取舍，导致即使在精确的准则得分下，胜者也依然悬而未决。在未声明所需权衡前提的情况下给出严格判定，即构成了“静默价值补全”。我们提出TradeoffBench基准，通过将规则执行与响应评分相分离来隔离这一失败模式。在权威准则向量上，六个开源权重的非推理型评判者虽然通过了所有支配性与相等性控制测试，却在1,345个冲突中的9%-81%的情形下选出了缺乏依据的胜者。全部577个顺序稳定的选择均遵循了被明确禁止的多数规则。在显式规则下，启用推理的配置将匿名向量违规降至9,720次判断中仅2次，但仍无法弥补被遗漏的权衡或各分量评分的分歧。在权衡未指明的原始TradeoffBench文本上，GPT-6.1-sol在400个已验证冲突中有29.8%的情况下省略了其补全前提。

    arXiv:2611.28940v1 Announce Type: cross  Abstract: Multi-criterion LLM judges often evaluate responses without specifying how conflicting criteria resolve, leaving the winner undetermined even under exact criterion scores. Issuing a strict verdict without declaring the required trade-off premise constitutes silent value completion. We introduce TradeoffBench to isolate this failure by separating rule execution from response scoring. On authoritative criterion vectors, six open-weight non-reasoning judges pass all dominance and equality controls yet pick unsupported winners on 9%-81% of 1,345 conflicts. All 577 order-stable choices follow an explicitly banned majority rule. Reasoning-enabled configurations drop anonymous-vector violations to 2 in 9,720 judgments under an explicit rule, yet cannot bridge omitted trade-offs or divergent component scoring. On raw TradeoffBench text with unspecified trade-offs, GPT-6.1-sol omits its completion premise in 29.8% of 400 verified conflicts. Eve
    
[^146]: CANDOR：一个用于构音障碍语音研究的多语言来源记录语料库

    CANDOR: A Multilingual Provenance-Documented Corpus for Dysarthric Speech Research

    [https://arxiv.org/abs/2611.28930](https://arxiv.org/abs/2611.28930)

    CANDOR 构建了一个涵盖 17 种语言、包含 2,057 名构音障碍说话者及超过 4,400 小时音频的大规模多语言构音障碍语音语料库，并系统记录了数据来源、保管链、许可证与知情同意等治理信息，为构音障碍语音研究提供了前所未有的公开资源基础。

    

    CANDOR 汇集了涵盖 17 种语言、五个病因学类别以及一个混合/其他类别的 61 个语料库。其登记册包含 2,057 名被归类为构音障碍的说话者和 19,538 名健康对照者。该合集包含 4,469.7 小时的音频，其中构音障碍子集占 1,178.8 小时。统计计数在可用时使用言语障碍标签，否则使用诊断标签。该合集包含跨越 14 种非英语语言的 732 名构音障碍说话者；其中 194 名说话者使用的六种语言不受 Qwen3-ASR（其非英语基线所基于的模型）支持。我们记录了语料构成、保管链、许可证和转让条款，同时区分语料库层面的来源记录与参与者层面的知情同意记录（我们仅持有少数组成部分的知情同意记录）。在 31 个包含构音障碍数据的语料库中，开放许可证覆盖了 29.0% 的语料库、10.5% 的说话者和 8.2% 的音频时长。排除英语的 Speech Accessibility Project 后，开放时长占比提升至 31.2%。

    arXiv:2611.28930v1 Announce Type: new  Abstract: CANDOR assembles 61 corpora across 17 languages and five aetiological classes plus a mixed/other group. Its register counts 2,057 speakers classified as dysarthric and 19,538 healthy controls. The collection contains 4,469.7 hours of audio, including 1,178.8 hours in the dysarthric subset. Counts use speech-impairment labels where available and diagnosis labels otherwise. The collection includes 732 dysarthric speakers across 14 non-English languages; 194 speakers use six languages unsupported by Qwen3-ASR, the model underlying its non-English baselines. We document composition, chain of custody, licences and transfer terms, while distinguishing corpus-level provenance from participant-level consent records, which we hold for few constituents. Among 31 dysarthric-bearing corpora, open licences cover 29.0% of corpora, 10.5% of speakers and 8.2% of hours. Excluding the English Speech Accessibility Project raises the open hour share to 31.2
    
[^147]: 面向零样本口语理解的定义引导上下文学习

    Definition-Guided In-Context Learning for Zero-Shot Spoken Language Understanding

    [https://arxiv.org/abs/2611.28929](https://arxiv.org/abs/2611.28929)

    该论文提出定义引导的上下文学习框架DG-ICL，通过用动态随机化定义（合成词元或置换词汇）替换固定标签，使语音-文本模型仅凭自然语言定义即可实现零样本口语理解，摆脱对标注数据和少样本示例的依赖。

    

    让口语理解（SLU）系统适配全新或经过修订的标签体系，传统上需要额外的标注数据和重新训练，这为现实世界的部署造成了瓶颈。虽然基于少样本示例的上下文学习提供了一种替代方案，但多模态的泛化与规模化一直面临挑战。在本工作中，我们研究了仅凭自然语言定义、在没有演示样本的情况下能否引导零样本SLU，发现符号化定义是适配语音-文本模型的主要监督驱动力。我们将这一原理具体化为定义引导的上下文学习（DG-ICL），这是一个将语义类别与固定词汇名称解耦的微调框架。通过用动态随机化的定义（在DG-ICL-Sym中使用任意合成词元，在DG-ICL-Flip中使用置换后的词汇词）替换标签，DG-ICL避免了词汇捷径，并鼓励模型……

    arXiv:2611.28929v1 Announce Type: cross  Abstract: Adapting spoken language understanding (SLU) systems to novel or revised label taxonomies conventionally requires additional labeled data and retraining, creating a bottleneck for real world deployment. While in-context learning with few-shot demonstrations offers an alternative, multi-modal generalization and scaling has been challenging. In this work, we investigate whether natural-language definitions alone can guide zero-shot SLU without demonstration exemplars, finding that symbolic definitions serve as the primary supervisory driver for adapting speech-text models. We operationalize this principle as Definition-Guided In-Context Learning (DG-ICL), a fine-tuning framework that decouples semantic categories from fixed lexical names. By replacing labels with dynamically randomized definitions (arbitrary synthetic tokens in DG-ICL-Sym or permuted vocabulary words in DG-ICL-Flip), DG-ICL avoids lexical shortcuts and encourages the mod
    
[^148]: 开源大语言模型能否清除制裁筛查警报？基于公开制裁名单与土耳其姓名变体的预注册测试

    Can Open LLMs Clear Sanctions-Screening Alerts? A Pre-registered Test on Public Sanctions Lists with Turkish Name Variants

    [https://arxiv.org/abs/2611.28914](https://arxiv.org/abs/2611.28914)

    该预注册测试表明，可在本地部署的开源大语言模型能够执行制裁筛查警报的一级处置任务，即使面对土耳其语姓名变体等复杂情况，误送无辜客户至二级审查的比例也远低于基于规则的基线方法。

    

    制裁筛查会产生大量基于姓名的警报，其中大多是误报，一级分析师需依据书面政策核对客户出生日期和姓名来清除这些警报。我们测试了银行可在本地部署的开源权重语言模型能否执行这一一级处置任务。我们从英国和欧盟制裁名单（7,103名自然人）中构建了12,200个合成客户：涵盖六种姓名条件下列入名单的人员（包括阿拉伯语姓名的土耳其语拼写以及西里尔字母姓名的土耳其语罗马化形式）和两种出生日期条件，并加入同名者、近似姓名客户及普通人群。一个确定性的模糊匹配与嵌入筛查产生了7,383条警报，三个开源模型在本地基础设施上提供服务，与三个规则基线在相同政策下进行处置。相比执行政策日期条款的规则，模型将无辜客户送往二级审查的数量要少得多：每1,000条中仅为0.6至16.9条。

    arXiv:2611.28914v1 Announce Type: cross  Abstract: Sanctions screening raises name-based alerts that are mostly false positives, and level-1 analysts clear them by checking the customer's date of birth and name against a written policy. We test whether open-weight language models that a bank can run on its own premises can perform this level-1 disposition. From the UK and EU sanctions lists (7,103 natural persons) we built 12,200 synthetic customers: listed persons under six name conditions, including Turkish spellings of Arabic-origin names and Turkish romanisations of Cyrillic names, and two date-of-birth conditions, plus namesakes, near-name customers and a general population. A deterministic fuzzy and embedding screen raised 7,383 alerts, which three open models served on domestic infrastructure and three rule baselines dispositioned under the same policy. The models sent far fewer innocent customers to level 2 than a rule that applies the policy's date clause: 0.6 to 16.9 per 1,00
    
[^149]: Thakaa 在 StanceEval-2026 上的表现：面向阿拉伯语立场检测的推理时目标锚定

    Thakaa at StanceEval-2026: Inference-Time Target Grounding for Arabic Stance Detection

    [https://arxiv.org/abs/2611.28860](https://arxiv.org/abs/2611.28860)

    该论文提出在推理时为阿拉伯语立场检测的大语言模型补充目标信息——通过从外部语料库检索带标注示例以及注入事实性、非评价性的目标上下文，并融合多个模型的标签预测，在 StanceEval-2026 的目标偏移场景下取得赛道1第六、赛道2第四的成绩，且证明基于相似度的示例检索能显著提升性能。

    

    StanceEval-2026 在目标偏移的场景下测试阿拉伯语立场分类，评估目标在带标注的任务数据中并不存在。我们测试了两种在推理时提供目标信息的方法。对于赛道1，我们从外部 ArabicStanceX 语料库中检索八个带标注的“女性驾驶”示例，并用 Gemma-4-31B-IT 对三个标签词元进行打分。对于未见目标赛道，我们前置缓存的目标上下文（通过提示词约束其仅陈述事实且不带评价性），然后融合多个大语言模型的标签预测与标签词元得分，最终分配结果受公开的类别总数约束。两个系统在赛道1和赛道2上分别取得 85.22 和 91.78 的 Favg²，在最终排名中位列第六和第四。在留出的 ArabicStanceX 推文上，直接提示得分为 79.45 Favg²；八个随机的“女性驾驶”示例将其提升至 84.78，而相似度检索则达到 89.04。在三个目标留出的 Mawqif 评估中，固定目标卡片提升（原文在此处截断）。

    arXiv:2611.28860v1 Announce Type: new  Abstract: StanceEval-2026 tests Arabic stance classification under target shift, with evaluation targets absent from the labelled task data. We test two ways to supply target information at inference time. For Track 1, we retrieve eight labelled Women Driving demonstrations from the external ArabicStanceX corpus and score the three label tokens with Gemma-4-31B-IT. For the unseen-target track, we prepend cached target context prompted to be factual and non-evaluative, then fuse label predictions and label-token scores from multiple LLMs, with final assignments constrained by published class totals. The systems obtain 85.22 and 91.78 Favg2 on Tracks 1 and 2, ranking sixth and fourth in the final standings. On held-out ArabicStanceX tweets, direct prompting scores 79.45 Favg2; eight random Women Driving demonstrations raise it to 84.78, and similarity retrieval reaches 89.04. Across three target-held-out Mawqif evaluations, a fixed target card raise
    
[^150]: Thakaa团队在AlexandriaX-2026中的表现：谁拥有最终决定权？在方言阿拉伯语机器翻译与错误检测中分配输出权威

    Thakaa at AlexandriaX-2026: Who Gets the Final Say? Assigning Output Authority in Dialectal Arabic Machine Translation and Error Detection

    [https://arxiv.org/abs/2611.28857](https://arxiv.org/abs/2611.28857)

    Thakaa团队通过最小贝叶斯风险（MBR）选择机制限制各系统的输出权威，并结合国家路由LoRA适配器整合利比亚和苏丹监督数据，在AlexandriaX-2026的全部四个方言阿拉伯语机器翻译与错误检测任务中均获得第一名。

    

    我们报告了在AlexandriaX-2026全部四个评测设置中均获得第一名的系统：两个英语到方言的翻译赛道、跨方言翻译以及翻译错误检测。在无约束翻译赛道中，我们的最小贝叶斯风险（MBR）选择机制对输出权威进行了限制：四个本地系统可以提供译文，而另外七个系统仅提供参考证据。候选约束的MBR在全部13种方言上都击败了最强的完整候选系统，宏观spBLEU整体提升1.2554，达到33.4891。它还超越了为每种方言回顾性选择最佳完整候选系统的事后oracle。在受限赛道中，基于共享NileChat-3B-Base骨干网络的国家路由LoRA适配器整合了训练集和开发集中缺失的已发布利比亚和苏丹监督数据，将宏观spBLEU从29.5969提升至30.4233。评测之后，在所有已发布的Alexandria数据上训练的单个适配器……（原文在此处截断）

    arXiv:2611.28857v1 Announce Type: new  Abstract: We report first-place systems for all four evaluated AlexandriaX-2026 settings: both English-to-dialect translation tracks, cross-dialect translation, and translation-error detection. In unconstrained translation, our minimum Bayes risk (MBR) selection restricts output authority: four local systems can supply the translation, while seven additional systems contribute only reference evidence. Candidate-constrained MBR beats the strongest complete candidate in all 13 dialects and gains 1.2554 macro spBLEU overall, reaching 33.4891. It also surpasses a retrospective oracle selecting the best whole candidate system per dialect. In the constrained track, country-routed LoRA adapters over a shared NileChat-3B-Base backbone incorporate the released Libyan and Sudanese supervision absent from the training and development splits, raising macro spBLEU from 29.5969 to 30.4233. Post-evaluation, a single adapter trained on all released Alexandria dat
    
[^151]: 偏差级联：理解基于大语言模型的社会模拟中知识对齐与效应复现之间的差距

    Bias Cascade: Understanding the Gap Between Knowledge Alignment and Effect Replication in LLM-based Social Simulation

    [https://arxiv.org/abs/2611.28854](https://arxiv.org/abs/2611.28854)

    该研究提出“偏差级联”这一失败机制，并构建了包含30个反事实社会实验的REFLEX基准，揭示了大语言模型即使具备准确的社会知识判断能力，仍可能无法复现经典社会研究的集体干预效应。

    

    大语言模型（LLM）越来越多地被用于模拟社会实验中的参与者。现有研究主要集中于对齐社会知识和孤立响应，而忽视了模拟是否能够再现社会干预的集体效应。由于我们称之为“偏差级联”的失败机制，这一区别至关重要：相似的个体状态可能导致不同的行为，而对孤立行为的一致认同仍可能通过动态交互产生截然不同的结果。因此，模型可能产生准确的社会判断，却无法复现经典人类研究的结果。为了实证地研究这一问题，我们引入了REFLEX（基础文献实验效应复现）基准，该基准包含30个基于经典人类研究的反事实社会实验，用于比较知识判断与效应模拟。知识判断直接……（原文摘要在此处截断）

    arXiv:2611.28854v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to simulate participants in social experiments. Existing research has primarily focused on aligning social knowledge and isolated responses, while overlooking whether simulations can reproduce the collective effects of social interventions. This distinction is critical due to a failure mechanism we call a $bias$ $cascade$: similar individual states can lead to different actions, while agreement on isolated actions can still produce divergent outcomes through dynamic interactions. As a result, models may produce accurate social judgments yet fail to reproduce the outcomes of classic human studies. To investigate this problem empirically, we introduce $REFLEX$ (Reproducing Effects of Foundational Literature EXperiments), a benchmark of 30 counterfactual social experiments grounded in classic human studies that compares knowledge judgment with effect simulation. Knowledge judgment directl
    
[^152]: Self-WSP：语言模型能否通过权重空间编程实现自我提升？

    Self-WSP: Can Language Models Improve Themselves Through Weight-Space Programming?

    [https://arxiv.org/abs/2611.28833](https://arxiv.org/abs/2611.28833)

    该论文提出Self-WSP方法，让Qwen3.5-0.8B语言模型读取并编辑自身权重从而实现自我提升，可将数学推理准确率大幅提高（如GSM8K从11.8%升至62.2%），但这些改进缺乏可解释性，且随机搜索有时也能达到类似效果，表明其作为递归自我改进路径前景与挑战并存。

    

    前沿语言模型如今可以像阅读和编辑代码一样读取和编辑神经网络的权重，从而改进一个小模型，这种方法被称为权重空间编程。我们进一步探索将WSP应用于语言模型自身，并将其称为Self-WSP。具体而言，Qwen3.5-0.8B读取自身的权重，提出一次编辑，然后以编辑后的权重继续同时担任问题求解者和编辑者的角色。我们发现了许多真正的自我提升案例：自我编辑可以将GSM8K准确率从11.8%提升至62.2%，将MATH500准确率从7.0%提升至32.2%，有时仅需修改16个参数；在某些情况下，重复的自我编辑能够持续改进模型。然而，深入分析表明，这些提升缺乏可解释性——模型对自身所做的编辑几乎没有理解，而且由随机搜索驱动的WSP有时也能取得相当的表现。因此，Self-WSP作为通往递归自我改进（RSI）的一条路径具有前景，但要真正实现这一目标仍面临挑战。

    arXiv:2611.28833v1 Announce Type: new  Abstract: Frontier language models can now read and edit the weights of a neural network much like code and thereby improve a small model, an approach known as Weight-Space Programming (WSP). We further explore applying WSP to a language model itself, which we call Self-WSP. Specifically, Qwen3.5-0.8B reads its own weights, proposes an edit, and then continues with the edited weights as both problem solver and editor. We find many cases of genuine self-improvement. Self-edits can raise GSM8K accuracy from 11.8% to 62.2% and MATH500 accuracy from 7.0% to 32.2%, sometimes by changing only 16 parameters, and in some cases repeated self-editing keeps improving the model. A closer look, however, reveals that these gains lack interpretability. The model shows little understanding of the edits it makes, and WSP driven by random search sometimes performs comparably. Self-WSP thus holds promise as a path toward recursive self-improvement (RSI), but realizi
    
[^153]: 嗡鸣的瑞瓦肖：《极乐迪斯科》中叙事权力的对话网络分析

    The Humming Revachol: A Dialogue Network Analysis of Narrative Power in Disco Elysium

    [https://arxiv.org/abs/2611.28812](https://arxiv.org/abs/2611.28812)

    本研究对《极乐迪斯科》中73名NPC的对话提及网络进行社会网络分析，发现游戏中的叙事权力主要由角色的叙事功能决定，而非其性别、种族等社会身份属性。

    

    本研究将社会网络分析应用于叙事角色扮演游戏《极乐迪斯科》，以考察叙事权力在其角色之间的分布方式。研究基于游戏完整对话脚本构建了一个包含73名NPC的有向加权角色提及网络，并对角色在五个社会属性维度上进行了识别与标注。叙事权力通过四个维度进行操作化：关注度、权威性、话语主动性和结构嵌入性。研究采用非参数统计检验考察性别、年龄、种族、阶级和意识形态之间的差异。结果表明，叙事权力在很大程度上独立于社会身份类别，而与角色所承担的叙事功能更为相关，同时结构嵌入性则反映了角色所处的社会组织背景。这些发现表明，对话网络分析能够揭示单凭可见性计数无法察觉的表征结构模式。

    arXiv:2611.28812v1 Announce Type: new  Abstract: This study applies social network analysis to the narrative RPG Disco Elysium to examine how narrative power is distributed among its characters. A directed, weighted character-mention network of 73 NPCs is constructed from the game's complete dialogue script, with characters identified and annotated across five social attributes. Narrative power is operationalised through four dimensions: attention, authority, discursive initiative, and embeddedness. Non-parametric statistical tests examine differences across gender, age, race, class, and ideology. The findings suggest that narrative power is largely independent of social identity categories and is more closely associated with character narrative functions, while structural embeddedness reflects the socio-organisational contexts in which characters are situated. These findings show that dialogue-network analysis can reveal structural patterns of representation that visibility counts alo
    
[^154]: HighwayVerse：面向公路基础设施异常理解的链式巡检基准

    HighwayVerse: A Chain-of-Inspection Benchmark for Highway Infrastructure Anomaly Understanding

    [https://arxiv.org/abs/2611.28800](https://arxiv.org/abs/2611.28800)

    该论文提出了贯穿异常检测、异常识别、状态分析与维护决策四阶段的递进式巡检链（HIIP-Chain）基准HighwayVerse，包含14,143条地面与无人机真实巡检记录及84,858条专家审核标注，用于系统评估多模态大语言模型在公路基础设施多部件异常理解全流程中的能力。

    

    实际公路基础设施巡检需要从视觉感知到维护决策的渐进式诊断推理，然而现有基准仍局限于孤立任务，导致多模态大语言模型（MLLM）在序列化巡检各阶段的能力评估不足。同时，当前数据集主要聚焦于公路路面缺陷，难以满足涵盖路面、护栏、防眩板和排水系统等多种基础设施部件的真实巡检需求。为填补这些空白，我们构建了公路基础设施巡检递进链（HIIP-Chain），涵盖异常检测、异常识别、状态分析和维护决策四个环节，并提出HighwayVerse基准。HighwayVerse包含14,143条来自地面和无人机（UAV）的真实巡检记录，分别涵盖100类和17类异常，并配有84,858条经专家审核的问题（原文此处截断）。

    arXiv:2611.28800v1 Announce Type: cross  Abstract: Practical highway infrastructure inspection entails progressive diagnostic reasoning from visual perception to maintenance decisions, yet existing benchmarks remain confined to isolated tasks, leaving Multimodal Large Language Models (MLLMs) insufficiently evaluated across sequential inspection stages. Meanwhile, current datasets focus primarily on highway pavement defects, falling short of real-world inspection across diverse infrastructure components such as highway pavements, guardrails, glare screens, and drainage systems. To address these gaps, we formulate the Highway Infrastructure Inspection Progression Chain (HIIP-Chain), spanning Anomaly Detection, Anomaly Recognition, Condition Analysis, and Maintenance Resolution, and introduce HighwayVerse. HighwayVerse comprises 14,143 real-world Ground and Unmanned Aerial Vehicle (UAV) inspection records spanning 100 and 17 anomaly categories, respectively, with 84,858 expert-reviewed qu
    
[^155]: DiscernBench：评估机构环境中LLM智能体协作辨识能力的基准

    DiscernBench: Evaluating Cooperative Discernment of LLM Agents in Institutional Settings

    [https://arxiv.org/abs/2611.28787](https://arxiv.org/abs/2611.28787)

    本文提出DiscernBench基准，评估LLM智能体在机构环境中对请求做出服从、拒绝或上报判断的协作辨识能力，并揭示多智能体商议中单一专家建议可能不成比例左右集体决策的“少数信号覆盖”现象。

    

    在实际运营环境中部署大型语言模型，需要对其在制度约束下的判断能力进行评估。我们提出了DiscernBench，这是一个用于评估“协作辨识能力”的基准测试，即智能体在机构环境中判断是否应当服从、拒绝或上报某项请求的能力。DiscernBench包含源自航空、执法和应急管理等领域的真实世界事故报告的场景。我们在三种决策配置下评估了八个大语言模型，并分析了最终决策以及多智能体商议过程中产生的中间判断。在出现分歧时，最终响应者表现出依赖模型和角色的聚合模式，其中单一的专家建议可能不成比例地影响集体决策，我们将这一现象称为“少数信号覆盖”。角色专业化的多智能体商议提升了平均过度……（原文摘要在此处被截断）

    arXiv:2611.28787v1 Announce Type: new  Abstract: The deployment of large language models in operational settings demands evaluation of judgment under institutional constraint. We introduce DiscernBench, a benchmark for evaluating cooperative discernment: the ability of an agent to determine whether to comply with, refuse, or escalate a request in an institutional setting. DiscernBench comprises scenarios derived from real-world incident reports across domains such as aviation, law enforcement, and emergency management. We evaluate eight LLMs across three decision-making configurations and analyze both final decisions and the intermediate judgments produced during multi-agent deliberation. Under disagreement, Final Responders exhibit model- and role-dependent aggregation patterns in which a lone specialist recommendation can disproportionately influence the collective decision, a phenomenon we term Minority-Signal Override. Role-specialized multi-agent deliberation raises average over-e
    
[^156]: 多模态大语言模型了解游戏规则，却默认真实世界

    Multimodal LLMs Know the Game but Assume the Real World

    [https://arxiv.org/abs/2611.28745](https://arxiv.org/abs/2611.28745)

    论文提出 RuleCraft 基准，发现多模态大语言模型虽然能识别游戏并陈述 Minecraft 规则，但在只看游戏画面时仍默认按现实世界物理作答，而仅通过在图像中点明游戏名称即可大幅提升其按游戏规则判断的准确率。

    

    多模态大语言模型能够陈述视频游戏的规则，但当它们只看到游戏画面时，是否会应用这些规则呢？我们推出了 RuleCraft，这是一个包含 1,240 道图像问题的基准，涵盖 155 条 Minecraft 规则，所有结果均在游戏引擎中得到验证；每个问题以三种方式提出：纯文本、在图像上注明游戏名称、以及仅呈现原始图像。六种广泛使用的模型既了解规则也能识别游戏，但在面对原始图像时却按照现实世界的物理规律作答：当在图像中提及游戏名称时，模型在“游戏偏离现实”的规则上的准确率在主测试集上提升 8.2–19.8 个百分点，在困难的压力测试集上提升 33.8–52.7 个百分点。模型的错误倾向于选择现实世界的结果，且当游戏保持场景不变时差距最大；其推理过程往往先陈述游戏规则，随后却依据现实作答。在主测试集上，提及游戏名称大多会将模型“默认按现实作答”切换为“按游戏规则作答”。该模式在启用推理能力时依然成立，在另外两款游戏中重复出现，并在模型自身标注为与游戏相关的问题上持续存在。

    arXiv:2611.28745v1 Announce Type: cross  Abstract: Multimodal large language models can state the rules of a video game, but do they apply them when they only see the game? We introduce RuleCraft, 1,240 image questions over 155 Minecraft rules with outcomes verified in the game engine, each asked in text, on the image with the game named, and on the bare image. Six widely used models know the rules and recognize the game, yet answer the bare image by real-world physics: naming the game raises their accuracy on rules where the game departs from reality by 8.2-19.8 points on a main set and by 33.8-52.7 on a hard stress-test set. Their errors pick the real-world outcome, the gap is largest where the game leaves a scene unchanged, and reasoning traces state the game's rule, then answer by reality. On the main set, naming the game mostly swaps the real-world default for the game's. The pattern holds with reasoning, recurs in two more games, and persists on questions the models themselves ta
    
[^157]: KMed-31B 技术报告

    KMed-31B Technical Report

    [https://arxiv.org/abs/2611.28719](https://arxiv.org/abs/2611.28719)

    KMed-31B 将医学任务按预期输出与反馈类型划分为四类，分别为每类训练专家模型，再通过多教师在线策略增量蒸馏（MOPD²）整合为单一模型，从而解决了多任务后训练中的性能相互干扰问题，适用于硬件受限的本地部署，并配套构建了模拟问诊语料库 Synthetic Health Dialogue。

    

    我们提出 KMed-31B，一个由开源权重模型后训练而来的医学大语言模型，专为硬件受限环境下的本地部署而设计。针对多样化医学任务进行模型后训练仍然十分困难，因为提升某一类任务的表现可能会降低另一类任务的表现。我们将这一现象归因于各任务在预期输出形式与可用反馈信号上的差异：医学问答和诊断预测具有参考答案，而临床问诊没有唯一正确的回答，需要依据评分细则进行评判。因此，我们依据这两个属性将医学训练任务划分为四个类别，为每个类别训练一个采用与其反馈相匹配奖励的专家模型，并通过多教师在线策略增量蒸馏（MOPD²）将多个专家模型整合到单一学生模型中。针对临床问诊任务，我们构建了 Synthetic Health Dialogue——一个模拟问诊语料库，包含与普通用户的咨询对话以及临床医生针对患者病例的对话（摘要原文在此处截断）。

    arXiv:2611.28719v1 Announce Type: cross  Abstract: We present KMed-31B, a medical large language model post-trained from an open-weight model for on-premises deployment with limited hardware. Post-training a model for diverse medical tasks remains difficult, because improving one kind of task can lower performance on another. We attribute this to differences between tasks in expected output and available feedback. Medical question answering and diagnosis prediction have reference answers, whereas clinical inquiry has no single correct response and is judged with rubrics. We therefore divide medical training tasks into four categories by these two properties, train one specialist per category with a reward that its feedback supports, and integrate the specialists into a single student with Multi-teacher On-policy Delta Distillation (MOPD$^2$). For clinical inquiry, we build Synthetic Health Dialogue, a simulated corpus of consultations with lay users and clinicians whose patient cases s
    
[^158]: 合法不等于优秀：低成本语言与决策模型在新策略游戏上的预注册基准测试

    Legal Is Not Good: A Pre-Registered Benchmark of Low-Cost Language and Decision Models on a New Strategy Game

    [https://arxiv.org/abs/2611.28713](https://arxiv.org/abs/2611.28713)

    该论文构建了一个名为 hexfolk 的全新策略游戏及预注册评测协议，结果表明即使被强制只输出合法着法，小型低成本语言模型（8B 和 3B）仍全面败给几行代码的规则玩家，说明“合法”并不等于“下得好”。

    

    《先测量再训练》一文提出，在任何人训练或替换模型之前，智能体的决策应当在一个预先固定的协议下对照真值进行测量。我们将该方法推广到第二个截然不同的领域——回合制策略游戏，并提出一个刻意保守的问题：在预注册协议下，低成本智能体的表现究竟如何？hexfolk 是为此目的专门编写的一个极简双人 4X 游戏，因此没有任何模型见过它的规则或局面。确定性引擎、统一的智能体接口、一个从不纠正非法或迟交着法的裁判，以及逐决策日志，使每一局对局都可复现；问题和预期答案在对局开始前就已预先提交，采用保留的种子、镜像座位和 95% Wilson 置信区间进行评估。两个小型开源语言模型（8B 和 3B）在每局比赛中都输给了一个仅几行代码的规则玩家。以自由文本作答时，它们 98% 的决策无效；当被限制为只能选择合法着法时……（摘要原文在此处截断）

    arXiv:2611.28713v1 Announce Type: new  Abstract: Measure Before You Train argued that an agent's decisions should be measured against ground truth, under a protocol fixed in advance, before anyone trains or replaces the model behind them. We carry the method to a second, very different domain, a turn-based strategy game, and ask a deliberately modest question: how well do low-cost agents play under a pre-registered protocol? hexfolk is a minimal two-player 4X game written for this purpose, so no model has seen its rules or positions. A deterministic engine, one agent interface, a referee that never corrects an illegal or late move, and per-decision logs make every game replayable; questions and expected answers were committed before the games, on held-out seeds with mirrored seats and 95% Wilson intervals. Two small open language models (8B and 3B) lose every game to a few-line rule player. Answering in free text they are invalid on 98% of their decisions; constrained to legal moves th
    
[^159]: PopMelodyGen：一个版权感知的可控流行旋律生成平台

    PopMelodyGen: A Copyright-Aware Platform for Controllable Pop-Melody Generation

    [https://arxiv.org/abs/2611.28686](https://arxiv.org/abs/2611.28686)

    本文提出版权感知的可控流行旋律生成平台PopMelodyGen，通过专家合成器生成可审计的大规模合成预训练数据，再结合小型授权真实流行音乐语料适配神经模型，同时服务于创作者的交互式旋律创作与研究者的带注释数据集构建。

    

    由于版权限制，大规模可再分发的符号化流行旋律数据仍然稀缺，这为音乐生成的实际部署和可复现研究造成了瓶颈。我们提出了PopMelodyGen，一个结合基于规则合成与神经生成的版权感知可控流行旋律生成平台。我们的方法首先使用专家合成器从显式音乐知识和模板生成大规模带标签的合成旋律，提供具有可审计来源的可扩展预训练数据。随后，神经模型在该合成语料库上进行预训练，并使用小型授权的真实流行音乐语料库进行适配，以在保持可控性的同时提升旋律的自然度。该平台为创作者提供支持伴奏播放和导出的交互式旋律创作功能，也为研究者提供带注释符号数据集的批量构建能力。PopMelodyGen展示了一种实用的训练和……（摘要原文在此处截断）

    arXiv:2611.28686v1 Announce Type: cross  Abstract: Large-scale redistributable symbolic pop-melody data remain scarce due to copyright restrictions, creating a bottleneck for both practical deployment and reproducible research in music generation. We present PopMelodyGen, a copyright-aware platform for controllable pop-melody generation that combines rule-based synthesis with neural generation. Our approach first uses an expert synthesizer to produce large-scale labeled synthetic melodies from explicit musical knowledge and templates, providing scalable pretraining data with auditable provenance. A neural model is then pretrained on this corpus and adapted with a small licensed real-pop corpus to improve melodic naturalness while preserving controllability. The platform supports interactive melody creation with accompaniment playback and export for creators, as well as batch construction of annotated symbolic datasets for researchers. PopMelodyGen demonstrates a practical training and 
    
[^160]: 物理状态变化暴露了大语言模型中的公式误用问题

    Physical regime changes expose formula misuse in large language models

    [https://arxiv.org/abs/2611.28669](https://arxiv.org/abs/2611.28669)

    该论文通过72种物理机制的受控实验揭示了大语言模型经常在公式有效范围之外误用公式（导致准确率下降13-57个百分点），并证明基于可验证奖励的强化学习结合监督微调能显著提升模型对物理状态的感知与推理能力。

    

    大语言模型（LLMs）在地球与环境科学领域的科学计算中应用日益广泛，但这些领域的问题往往涉及不同的物理状态（regimes）。LLMs可能无法识别这种依赖于物理状态的约束条件，在公式有效适用范围之外套用公式，从而产生看似合理但在物理上错误的结果。我们通过受控实验系统地研究了这类失败，并评估后训练能否提升模型对物理状态的感知推理能力。在72种物理机制上的实验表明，前沿LLMs经常在公式适用状态之外误用公式，导致准确率下降13-57个百分点。基于可验证奖励的强化学习使Qwen3.5-9B在依赖物理状态的任务上能够匹敌DeepSeek V4 Flash（284B），但其有效性受限于基础模型中已有的公式储备。将少量完整示例的监督微调与强化学习相结合……（原文摘要在此处被截断）

    arXiv:2611.28669v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly employed for scientific computing across the Earth and environmental sciences, but problems in these fields often involve distinct physical regimes. LLMs may fail to recognize these regime-dependent constraints, applying formulas outside their domains of validity and producing plausible but physically incorrect results. We systematically investigate these failures through controlled experiments and assess whether post-training can improve regime-aware reasoning. Across 72 physical mechanisms, we find frontier LLMs frequently misapply formulas outside their regimes of validity, reducing accuracy by 13-57 percentage points. Reinforcement learning with verifiable rewards enables Qwen3.5-9B to match DeepSeek V4 Flash (284B) on regime-dependent tasks, but its effectiveness is limited by the formulas available in the base model. Combining supervised fine-tuning on a few worked examples with reinf
    
[^161]: 看得对，写得错：短思维链在何处破坏空间视觉问答

    Seeing Right, Writing Wrong: Where Short Chain-of-Thought Breaks Spatial VQA

    [https://arxiv.org/abs/2611.28665](https://arxiv.org/abs/2611.28665)

    本文揭示短思维链（CoT）损害视觉语言模型空间问答能力的机制：CoT将感知问题变成转录问题，模型一旦写下空间关系，答案便跟随文字而非图像，因此CoT是否有益取决于所写关系的准确程度。

    

    要求视觉语言模型（VLM）逐步思考，往往会使它在空间问题上的表现变差，即使图像仍保留在上下文中。我们证明，短思维链（CoT）将一个感知问题转变成了一个转录问题。一旦VLM写下某个空间关系，它的答案就会跟随那句话而非图像，因此CoT究竟是有帮助还是有损害，很大程度上取决于写下的关系是否比直接回答更准确。通过研究15个开源VLM（2B-38B）和六个前沿闭源模型，我们排除了token数量、轮次格式以及开源模型中的视觉遗忘等因素，并将失败 traced 到三个阶段。第一，书写会重新加权视觉线索：在所测试的全部13个开源模型中，它削弱了“越低越近”的捷径，但同时也会丢弃有效线索，从而产生错误的关系。第二，写下的文本接管了答案：在固定模型自身去除结论的推理文本的情况下，替换为深度相反的图像会使答案……

    arXiv:2611.28665v1 Announce Type: cross  Abstract: Asking a vision-language model (VLM) to think step by step often makes it worse at spatial questions, even though the image stays in context. We show that short chain-of-thought (CoT) turns a perception problem into a transcription problem. Once a VLM writes a spatial relation, its answer follows that sentence instead of the image, so whether CoT helps or hurts depends largely on whether the written relation is more accurate than answering directly. Studying 15 open VLMs (2B-38B) and six frontier closed models, we rule out token count, turn format, and visual forgetting in open models and trace the failure through three stages. First, writing re-weights visual cues, weakening a "lower is closer" shortcut in all 13 open models tested but also discarding valid cues, which produces false relations. Second, the written text takes over. With the model's own conclusion-removed rationale held fixed, swapping in an opposite-depth image moves t
    
[^162]: S-Mind：将多视图几何内化以实现空间智能

    S-Mind: Internalizing Multi-View Geometry for Spatial Intelligence

    [https://arxiv.org/abs/2611.28649](https://arxiv.org/abs/2611.28649)

    S-Mind通过一组紧凑的空间token将多视图几何内化到多模态大语言模型中，使其能够将多视图融合为连贯的3D场景并在推理中灵活变换视角，在四个空间基准上大幅超越前沿通用模型。

    

    空间智能，即从视觉观察中对三维空间进行推理的能力，随着多模态大语言模型（MLLM）走向物理世界而变得至关重要。除了感知每个视图之外，空间智能还依赖于人类认知的两个基石：将多个视图融合为一个连贯的三维场景，以及在推理过程中将该场景在心理上变换到新的视角。然而，我们的空间探针测试显示，MLLM在很大程度上缺乏这两种能力，即使经过空间微调也是如此。因此，我们提出了S-Mind，它通过一组紧凑的空间token将多视图几何内化到MLLM中。它监督这些token以保存融合后的三维场景，并重新发射这些token，以便从推理所采用的每个视角重新表达该场景。在四个空间基准测试中，S-Mind使其Qwen3.5骨干模型的性能提升高达22.26分，并超越了GPT-6-Astra（+0.45）、Kimi-K3.1-Pro（+1.97）和GLM-5.3-Flash等前沿通用模型。

    arXiv:2611.28649v1 Announce Type: cross  Abstract: Spatial intelligence, the ability to reason about 3D space from visual observations, becomes essential as multimodal large language models (MLLMs) move into the physical world. Beyond perceiving each view, it rests on two cornerstones of human cognition: fusing multiple views into one coherent 3D scene and mentally transforming that scene to new viewpoints as reasoning unfolds. Yet our spatial probes reveal that MLLMs largely lack both capabilities, even after spatial fine-tuning. We therefore present S-Mind, which internalizes multi-view geometry within the MLLM through a compact set of spatial tokens. It supervises these tokens to hold the fused 3D scene and re-emits them to re-express this scene from every viewpoint the reasoning adopts. Across four spatial benchmarks, S-Mind improves its Qwen3.5 backbones by up to 22.26 points and surpasses frontier general models such as GPT-6-Astra (+0.45), Kimi-K3.1-Pro (+1.97), and GLM-5.3-Flas
    
[^163]: 使位置信息失效：可交换自蒸馏抹平“中间迷失”曲线

    Make Position Uninformative: Exchangeable Self-Distillation Flattens the Lost-in-the-Middle Curve

    [https://arxiv.org/abs/2611.28638](https://arxiv.org/abs/2611.28638)

    提出可交换自蒸馏方法，无需标准答案、仅用几千个打乱上下文的样本训练小型适配器，即可抹平检索增强语言模型的“中间迷失”位置偏差，同时提升平均准确率和最差位置准确率。

    

    检索增强型语言模型在相关文档位于开头或末尾时回答效果良好，而当相关文档位于中间时回答效果较差。一些微调方法通过降低开头位置的准确率来提升中间位置的准确率。我们展示了训练上下文是如何教授位置偏差的。通过统一的答案槽位和打乱顺序的干扰项，最优阅读器会忽略位置信息。基于这一结果，可交换自蒸馏在打乱顺序的上下文上训练一个小型适配器，使其复现模型单独阅读答案文档时的答案。该方法不需要标准答案，且只需几千个样本。在参数量从1.5B到32B的六个模型上，ExSD相对于冻结模型提升了平均准确率和最差位置的准确率，包括在留出的测试集上也是如此。在受控的LoRA对比实验中，它在两项指标上都显著优于Pos2Distill。与CORD相比，平均差异并不显著，但最差位置准确率在20-和3...（原文在此处截断）

    arXiv:2611.28638v1 Announce Type: cross  Abstract: Retrieval-augmented language models answer well when the relevant document comes first or last and poorly when it sits in the middle. Some fine-tuning recipes lift the middle by lowering the start. We show how training contexts teach positional bias. With a uniform answer slot and shuffled distractors, the optimal reader ignores position. Building on this result, Exchangeable Self-Distillation (ExSD) trains a small adapter on shuffled contexts to reproduce the model's answer when it reads the answer document alone. It needs no gold answers and only a few thousand examples. On six models from 1.5B to 32B parameters, ExSD raises both average and worst-position accuracy against the frozen model, including on a held-out test split. In controlled LoRA comparisons, it significantly improves both criteria over Pos2Distill. Against CORD, mean differences are not significant, while worst-position accuracy improves significantly in the 20- and 3
    
[^164]: 通过共享位置与仅注意力蒸馏减少多模态大语言模型的排列敏感性

    Reducing Permutation Sensitivity in Multimodal Large Language Models via Shared Positions and Attention-Only Distillation

    [https://arxiv.org/abs/2611.28612](https://arxiv.org/abs/2611.28612)

    本文提出共享位置编码与仅注意力蒸馏相结合的方法，在降低多模态大语言模型排列敏感性的同时恢复了预测准确率。

    

    本文对如何在保持准确率的同时提升排列鲁棒性开展了系统性研究。首先，我们将排列偏差现象归纳为四种代表性场景；随后，以文本选项重排场景下的多选题问答（MCQA）数据集为例，分析排列偏差的根本原因。基于该根因分析与集合编码策略，我们研究了采用阻塞式跨单元注意力的共享位置方法（SP-B），并将该设计扩展为支持模态交换的双向跨单元注意力方法（SP-V）。然而，这些结构性改动可能会降低预测准确率。为了在共享位置（SP）结构下恢复模型在不同场景中的能力，我们提出了一种从冻结的原始模型进行的仅注意力蒸馏方法。共享位置降低了排列敏感性，而仅注意力蒸馏在共享位置结构下提升了准确率。在所有三种（场景中）……

    arXiv:2611.28612v1 Announce Type: new  Abstract: In this paper, we conduct a systematic study on improving permutation robustness while preserving accuracy. First, we categorize permutation bias phenomena into four representative scenarios. Then, we take the multiple-choice question answering (MCQA) dataset as an example under text-option reordering scenario to analyze the root cause of permutation bias. Based on this root cause analysis and set encoding strategy, we investigate shared positions with blocked cross-unit attention (SP-B) and extend this design to bidirectional cross-unit attention (SP-V) for modality exchange. However, these structural changes may reduce predictive accuracy. To recover model capability across different scenarios under the SP structure, an attention-only distillation from frozen native models is proposed. Shared positions reduce permutation sensitivity, and attention-only distillation improves accuracy under the shared-position structure. Across all three
    
[^165]: CLOSE：一种面向视觉-语言模型物体幻觉的校准、无标注、在线自评估验证器

    CLOSE: A Calibrated Label-free Online Self-Evaluating Verifier for Object Hallucination in Vision-Language Models

    [https://arxiv.org/abs/2611.28589](https://arxiv.org/abs/2611.28589)

    提出CLOSE验证器，利用视觉-语言模型对自身幻觉的内在判断能力，通过基于排名的记忆分位数阈值实现校准、无标注、在线自评估的物体幻觉检测。

    

    视觉-语言模型经常因为语言先验而描述图像中并不存在的物体。然而，模型本身已经是其自身输出的强大判断者：当被单独询问某物体是否存在时，它会拒绝44%-62%的自身幻觉，且其“是”的概率在无需训练的情况下即可达到0.90-0.95的AUROC。然而这一判断并不能直接使用：(1) 其概率在不同模型之间缺乏统一的尺度；(2) 对于剩余38%-56%的幻觉，模型仍会以与真实物体概率相重叠的“是”概率作答，因此没有任何阈值能够将它们区分开。为此，我们提出CLOSE，一种校准的、无标注的、在线自评估的验证器，可在生成过程中强制执行这一判断。其阈值是记忆库的分位数而非概率，因此仅依赖于排名；该记忆库通过无标注图像冷启动并在线更新。一个基于四种隐藏状态信号的伪标签自训练的评分器（摘要至此截断）

    arXiv:2611.28589v1 Announce Type: cross  Abstract: Vision-language models frequently describe objects absent from the image because of language priors. Yet the model itself is already a strong judge of its own output: asked separately whether an object is present, it rejects 44%-62% of its hallucinations, and its yes probability reaches an AUROC of 0.90-0.95 without training. This judgment, however, is not directly usable: (1) its probabilities lack a common scale across models; (2) for the remaining 38%-56%, the model still answers yes with probabilities that overlap those of real objects, so no threshold separates them. We therefore propose CLOSE, a Calibrated Label-free Online Self-Evaluating verifier that enforces this judgment during generation. Its threshold is a quantile of a memory rather than a probability and thus depends only on ranks; the memory is cold-started from unlabeled images and updated online. A scorer self-trained on pseudo-labels from four hidden-state signals is
    
[^166]: MedCoast：面向测试时生物医学问答的推理轨迹语义抽象

    MedCoast: Semantic Abstraction of Reasoning Trajectories for Test-Time Biomedical Question Answering

    [https://arxiv.org/abs/2611.28586](https://arxiv.org/abs/2611.28586)

    MedCoast通过将采样的思维链轨迹语义抽象为紧凑的推理树结构，并据此训练轻量级过程奖励模型进行过程监督，从而在有限推理预算下高效提升测试时生物医学问答的多步推理性能。

    

    生物医学问答通常需要在有限的推理预算下进行多步推理。虽然诸如思维树之类的测试时搜索方法可以提升推理性能，但直接在自由形式的语言上进行搜索计算成本高昂且高度冗余，因为许多推理轨迹包含以不同表面形式表达的语义等价的中间步骤。我们提出了MedCoast，这是一个离线到在线的推理框架，它将采样的思维链轨迹转换到一个紧凑空间中，以实现结构化的过程监督。在离线阶段，我们在共享前缀下递归地聚类语义相似的推理步骤，构建一个能够捕获重复出现的临床推理模式的抽象推理树。随后，我们从朝着正确解答的进展中推导出节点级别的监督信号，并训练一个轻量级的过程奖励模型（PRM）来评估部分推理状态。在推理阶段，该……

    arXiv:2611.28586v1 Announce Type: cross  Abstract: Biomedical question answering often requires multi-step reasoning under limited inference budgets. While test-time search methods such as Tree-of-Thought can improve reasoning performance, directly searching over free-form language is computationally expensive and highly redundant, as many reasoning trajectories contain semantically equivalent intermediate steps expressed in different surface forms. We introduce MedCoast, an offline-to-online reasoning framework that converts sampled chain-of-thought trajectories into a compact space for structured process supervision. Offline, we recursively cluster semantically similar reasoning steps under shared prefixes to construct an abstract reasoning tree that captures recurring clinical reasoning patterns. We then derive node-level supervision from progress toward correct solutions, and train a lightweight process reward model (PRM) to evaluate partial reasoning states. At inference time, the
    
[^167]: 理解并缓解自我奖励优化的脆弱性

    Understanding and Mitigating the Fragility of Self-Rewarding Optimization

    [https://arxiv.org/abs/2611.28582](https://arxiv.org/abs/2611.28582)

    该论文揭示了自我奖励优化（SRO）本质上是脆弱的——微小的提示词扰动即可破坏其自我生成的偏好信号并引发迭代崩溃，并提出自我奖励保真度（SRF）指标作为诊断自我监督可靠性的信号，以理解和缓解这种脆弱性。

    

    自我奖励优化（SRO）有望通过让语言模型生成自身的奖励信号来改进模型。然而，SRO何时以及如何成功或失败仍然缺乏深入理解。我们揭示出SRO本质上是脆弱的：即使对训练提示词进行看似良性的扰动，也可能破坏自我生成的偏好信号并引发迭代崩溃。我们通过两种提示词层面的攻击（作为可控压力测试）来证明这一点：一是“重新排序”攻击，它将训练集中在自我奖励与参考标准产生分歧的提示词上；二是“前导词”攻击，它通过学习简短的前缀来抬高自我奖励，而模型实际质量却没有任何提升。为解释这些攻击为何能够成功，我们引入了自我奖励保真度（SRF），这是一个量化自我评估奖励与参考标准奖励之间在策略上一致性的指标。我们证明SRF可以作为自我监督可靠性的诊断信号，并且受控的SRF扰动……

    arXiv:2611.28582v1 Announce Type: cross  Abstract: Self-rewarding optimization (SRO) promises to improve language models by enabling them to generate their own reward signals. Yet when and how SRO succeeds or fails remains poorly understood. We reveal that SRO is fundamentally fragile: even seemingly benign perturbations to training prompts can corrupt the self-generated preference signal and trigger iterative collapse. We demonstrate this through two prompt-level attacks that serve as controlled stress tests: a Re-Ranking attack that concentrates training on prompts where self-reward diverges from an oracle, and a Preamble attack that learns short prefixes inflating self-reward without any gain in oracle quality. To explain why these attacks succeed, we introduce self-reward fidelity (SRF), a metric quantifying on-policy agreement between self-evaluated and oracle rewards. We show that SRF can serve as a diagnostic signal of self-supervision reliability, and that controlled SRF pertur
    
[^168]: CLASP：在全切片病理报告生成中于解码前决定结构与论断

    CLASP: Deciding Structure and Claims Before Decoding for Whole-Slide Pathology Report Generation

    [https://arxiv.org/abs/2611.28576](https://arxiv.org/abs/2611.28576)

    CLASP提出“先决策后解码”的病理报告生成框架，在解码前通过范例结构先验（ESP）、校准论断决策（CCD）和论断位置分配（CSP）显式确定报告的结构、长度与发现内容，克服了现有方法将所有报告级选择交给解码器导致的生成不足问题。

    

    全切片病理报告生成已通过更强的切片编码器、检索到的知识以及近来的切片级多模态大语言模型取得进展。然而，这些生成器把所有报告层面的选择都留给了解码器：报告有多长、如何组织、陈述哪些发现以及在何处陈述。在标准的PathText-BRCA基准上，我们发现这些选择主导了n-gram重叠度——未经学习、直接选取来自查询病例所在医院的报告即可接近最先进水平，而解码器则生成不足并退回到最高频的发现上。我们提出了CLASP，一种“先决策后解码”的生成器，它对每张切片显式地做出这些选择。在解码之前，范例结构先验（ESP）从图像相似病例的报告中获取句子计划和长度，校准论断决策（CCD）从校准后的切片后验概率中选择发现，论断到句子的位置分配（CSP）将每个发现分配到……

    arXiv:2611.28576v1 Announce Type: cross  Abstract: Whole-slide pathology report generation has advanced through stronger slide encoders, retrieved knowledge, and, recently, slide-level multimodal LLMs. However, these generators leave every report-level choice to the decoder: how long the report is, how it is organised, and which findings it states and where. On the standard PathText-BRCA benchmark, we find that these choices dominate n-gram overlap, as reports from the query's hospital selected without learning approach the state of the art, while decoders under-generate and fall back to the most frequent findings. We present CLASP, a decide-then-decode generator that makes these choices explicitly, once per slide. Before decoding, an Exemplar Structure Prior (ESP) takes the sentence plan and length from reports of image-similar cases, a Calibrated Claim Decision (CCD) selects the findings from calibrated slide posteriors, and a Claim-to-Sentence Placement (CSP) assigns each finding to
    
[^169]: 基于质量、结构与时间信号建模同声传译中的感知及时性

    Modeling Perceived Promptness in Simultaneous Interpreting from Quality, Structure, and Timing Signals

    [https://arxiv.org/abs/2611.28566](https://arxiv.org/abs/2611.28566)

    该研究发现仅靠起始延迟时间指标无法充分解释专业译员对同传及时性的评判，而将译语的语言质量、表达力等预测特征与时间信号相结合，可将感知延迟的预测相关系数从0.318显著提升至0.475。

    

    同声传译要求译员在实时跟上源语的同时准确地传达源语内容。诸如语段起始延迟等时间指标仅能捕捉这一要求的一部分，但可能无法完全反映专业评估者如何评判及时性。我们研究这些评判是否仅凭时间信息即可解释，还是译语输出能提供额外信息。我们的数据集包含来自16篇源语演讲和七名译员的622个汉英和英汉语段。两位资深专业译员对每个语段的语言质量（LQ）、表达力（EXP）和感知延迟（LAT，我们也称之为专业及时性）进行了评分。在留出的源语演讲上，仅起始延迟与LAT评分的皮尔逊相关系数为 r = .318。将预测的语言质量和表达力与延迟相结合后，相关系数提升至 r = .475。词汇和结构特征提供了更强的（摘要在此处截断）

    arXiv:2611.28566v1 Announce Type: new  Abstract: Simultaneous interpreting requires interpreters to convey the source accurately while keeping pace with it in real time. Timing measures such as segment-onset delay capture part of this requirement, but may not fully reflect how professional evaluators judge promptness. We investigate whether these judgments can be explained by timing alone or whether the interpreted output provides additional information. Our dataset contains 622 Chinese-English and English-Chinese segments from 16 source speeches and seven interpreters. Two senior professional interpreters rated each segment for language quality (LQ), expressiveness (EXP), and perceived latency (LAT), which we also refer to as professional promptness. On held-out source speeches, onset delay alone achieves a Pearson correlation of r = .318 with LAT ratings. Combining predicted LQ and EXP with delay improves the correlation to r = .475. Lexical and structural features provide a stronger
    
[^170]: Tool-to-Self（工具到自身）：通过脚手架式强化学习内化视觉工具能力

    Tool-to-Self: Internalizing Visual Tool Capabilities via Scaffolded Reinforcement Learning

    [https://arxiv.org/abs/2611.28564](https://arxiv.org/abs/2611.28564)

    提出T2S框架，将外部视觉工具转化为强化学习中的临时训练脚手架并逐步退火，使模型内化工具能力，在无需推理时调用工具的情况下获得稳定且高效的视觉感知与推理性能提升。

    

    视觉-语言模型（VLMs）日益依赖外部视觉工具来获取细粒度的视觉证据，但重复的工具执行会带来巨大的推理开销和部署依赖。此外，传统的工具使用强化学习虽然可以提升视觉任务的表现，但由此获得的工具收益往往不稳定。基于此，我们提出了Tool-to-Self（T2S），一个以工具为脚手架的强化学习框架，将外部视觉工具转化为临时性的训练脚手架。T2S通过工具可修复的样本对工具使用能力和观测预测进行冷启动，然后在强化学习过程中按工具逐个逐步退火工具执行。通过将答案正确性奖励和观测一致性奖励与来自真实工具观测的特权在线策略自蒸馏相结合，T2S有效地教会模型既复现工具输出、又基于工具输出进行推理。在多样的视觉感知与推理任务上的实验表明……（摘要截断）

    arXiv:2611.28564v1 Announce Type: cross  Abstract: Vision-language models (VLMs) increasingly rely on external visual tools to access fine-grained visual evidence, but repeated tool execution incurs substantial inference overhead and deployment dependencies. Moreover, conventional tool-use RL can improve visual-task performance, yet the resulting tool gains are often unstable. Motivated by this, we propose Tool-to-Self (T2S), a tool-scaffolded reinforcement learning framework that turns external visual tools into temporary training scaffolds. T2S cold-starts tool-use ability and observation prediction from tool-repairable examples, then progressively anneals tool execution on a per-tool basis during RL. By combining answer-correctness and observation-consistency rewards with privileged on-policy self-distillation from actual tool observations, T2S effectively teaches the model both to reproduce tool outputs and to reason from them. Experiments across diverse visual perception and reaso
    
[^171]: 从奖励工程到以数据为中心的强化学习：实现有效的智能体视觉工具使用

    From Reward Engineering to Data-Centric Reinforcement Learning for Effective Agentic Visual Tool Use

    [https://arxiv.org/abs/2611.28558](https://arxiv.org/abs/2611.28558)

    该论文提出 AOVO 框架，通过感知归因和策略自适应数据采样（PADS）实现以数据为中心的强化学习，无需依赖奖励工程即可让多模态大语言模型学会有效地调用视觉工具。

    

    多模态大语言模型（MLLM）在推理过程中越来越多地使用外部视觉工具来获取证据，然而学习何时调用这些工具仍然具有挑战性。现有的工具使用强化学习（RL）方法通常依赖与调用挂钩的奖励来鼓励有益的工具使用，但这种奖励工程可能会扭曲任务目标，并导致退化的调用行为。我们认为，有效的工具使用学习不仅需要识别哪些问题是困难的，还需要识别哪些失败可归因于视觉证据的缺失，从而可以通过工具观察来纠正。我们提出了 AOVO，一个以数据为中心的框架，通过感知归因和策略自适应数据采样（PADS），使 MLLM 能够学习有效的工具使用，而无需引入特定于工具的奖励。它通过反事实探测模型的推理状态来识别对观察敏感的问题，并区分……

    arXiv:2611.28558v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) increasingly use external visual tools to acquire evidence during reasoning, yet learning when to invoke these tools remains challenging. Existing tool-use reinforcement learning (RL) methods often rely on call-contingent rewards to encourage beneficial tool use, but such reward engineering can distort the task objective and lead to degenerate calling behaviors. We argue that effective tool-use learning requires identifying not merely which questions are difficult, but which failures are attributable to missing visual evidence and can therefore be corrected through tool observations. We propose AOVO, a data-centric framework that enables MLLMs to learn effective tool use through perception attribution and policy-adaptive data sampling (PADS), without introducing tool-specific rewards. It counterfactually probes the model's reasoning states to identify observation-sensitive questions, distinguish
    
[^172]: BIRD-Persona：一个面向SQL和pandas分析代码生成的角色感知基准

    BIRD-Persona: A Persona-Aware Benchmark for SQL and pandas Analytics Code Generation

    [https://arxiv.org/abs/2611.28542](https://arxiv.org/abs/2611.28542)

    BIRD-Persona是一个通过引入新手/专家角色改写、模式变体和任务结构标注构建的角色感知代码生成基准，大规模实验表明SQL在绝大多数大语言模型上优于pandas，且两者差距随用户措辞、模式表示和任务结构等因素系统性变化。

    

    生产级文本到代码系统使用大语言模型将自然语言分析请求转换为可执行的SQL和pandas代码。现有基准测试同时涵盖这两类任务，但未能隔离出两者之间性能出现分歧的具体条件。我们提出BIRD-Persona，这是BIRD-Python的一个按用户角色分层的扩展版本，它为具有匹配SQL和pandas标准答案的任务添加了新手和专家风格的改写，同时包含模式变体和任务结构标注。在1,534个任务、九个模型和245,592次评估中，SQL在九个系统中的八个上优于pandas。性能差距因条件而异：专家式、模式显式的提示词对两种范式都有提升，但对SQL的帮助更大；模式表示的影响不一且依赖于具体模型；选定列的负担会扩大差距，而子查询则不会。通过将用户措辞作为受控的基准测试维度，BIRD-Persona支持对自然语言分析系统进行角色感知和范式感知的评估。

    arXiv:2611.28542v1 Announce Type: new  Abstract: Production text-to-code systems use large language models to turn natural-language analytics requests into executable SQL and pandas code. Existing benchmarks cover both tasks but do not isolate when performance diverges between them. We introduce BIRD-Persona, a persona-stratified extension of BIRD-Python that adds novice and expert rewrites to tasks with matched SQL and pandas ground truth, along with schema variants and task-structure annotations. Across 1,534 tasks, nine models, and 245,592 evaluations, SQL outperforms pandas for eight of nine systems. The gap varies by condition: expert, schema-explicit prompts improve both paradigms but help SQL more; schema representations have mixed, model-dependent effects; and selected-column burden widens the gap while subqueries do not. By making user phrasing a controlled benchmark dimension, BIRD-Persona supports persona-aware and paradigm-aware evaluation of natural-language analytics syst
    
[^173]: 六个大型语言模型与人类决策者的信息获取与得失框架效应

    Information Access and Gain-Loss Framing in Six Large Language Models and Human Decision Makers

    [https://arxiv.org/abs/2611.28525](https://arxiv.org/abs/2611.28525)

    六个大型语言模型在信息受限时均表现出显著的类似人类的得失框架效应，而允许其先查阅更多信息卡片能显著减弱该效应，其中GPT-4.1在4张卡片、Qwen3在仅2张卡片时效应降至预设阈值以内。

    

    一个风险选择可以通过描述将会挽救什么或将会失去什么来呈现，而人们往往会因措辞不同而做出不同的选择。我们测试了六个大型语言模型是否表现出这种得失框架效应，以及当它们能够先阅读更多关于选项的信息时，这种效应是否会减弱。GPT-4.1、GPT-4o、GPT-4o-mini、Gemini 2.5 Flash、Qwen3 235B和Llama 4 Maverick在四个风险情景中共做出了11,519次选择，每个情景包括原始措辞、两个改写版本和一个措辞相反的镜像版本。在选择之前，模型可以打开七张信息卡片中的0、2、3或4张。在没有信息卡片的情况下，所有六个模型都表现出较大的框架效应：在损失措辞下选择安全选项的比例比在收益措辞下低27到79个百分点。在所有模型中，打开2张卡片时的差距都比没有卡片时更小。根据分析计划中设定的10个百分点阈值，GPT-4.1在打开4张卡片时降至阈值以内，而Qwen3仅在打开2张卡片时就已达标。

    arXiv:2611.28525v1 Announce Type: new  Abstract: A risky choice can be described by what will be saved or what will be lost, and people often choose differently depending on that wording. We test whether six large language models (LLMs) show this gain-loss framing effect, and whether it shrinks when they can first read more about the options. GPT-4.1, GPT-4o, GPT-4o-mini, Gemini 2.5 Flash, Qwen3 235B and Llama 4 Maverick made 11,519 choices across four risk scenarios, each in its original wording, two paraphrases and a mirrored version in the opposite wording. Before choosing, a model could open up to 0, 2, 3 or 4 of seven information cards. With no cards, all six models showed large framing effects: the share choosing the safe option was 27 to 79 percentage points lower under loss than under gain wording. The gap was smaller at 2 cards than with no cards in every model. Against a 10-point threshold set in our analysis plan, GPT-4.1 fell within it at 4 cards and Qwen3 only at 2 cards; 
    
[^174]: DisSparse：面向长上下文LLM服务的流水线化Top-$p$稀疏注意力

    DisSparse: Pipelined Top-$p$ Sparse Attention for Long-Context LLM Serving

    [https://arxiv.org/abs/2611.28522](https://arxiv.org/abs/2611.28522)

    提出DisSparse，一种流水线化的Top-$p$稀疏注意力方法，能够根据注意力在不同头、层和解码步骤中的分布变化动态调整token预算，从而克服固定预算top-$k$方法浪费带宽或丢失上下文的问题，优化长上下文LLM推理服务。

    

    长上下文LLM推理的瓶颈在于键值缓存（KV cache）的内存需求，其大小随上下文长度线性增长。动态稀疏注意力通过仅获取并关注关键的token子集来缓解这一瓶颈。动态稀疏注意力研究与部署中的主流原语是固定预算的top-$k$选择，其受青睐的原因在于其确定性的块数量与vLLM等系统所依赖的静态内存分配机制相契合。然而，无论注意力质量在各头、各层和各解码步骤中如何分布，top-$k$都使用固定预算，这导致对于注意力高度集中的头会浪费带宽，而对于注意力分散的头则会丢失上下文。Top-$p$（核采样）选择能够自然地适应这种变化，但由于两个系统级挑战——可变的块数量破坏了确定性块分配机制等，其在生产服务系统中的采用仍然有限。

    arXiv:2611.28522v1 Announce Type: cross  Abstract: Long-context LLM inference is bottlenecked by the memory demands of the key-value (KV) cache, whose size scales linearly with context length. Dynamic sparse attention reduces this bottleneck by fetching and attending only to a critical subset of tokens. The dominant primitive in dynamic sparse attention research and deployment is fixed-budget top-$k$ selection, which is favored because its deterministic block count aligns with the static memory allocation that systems like vLLM rely on. Top-$k$ uses a fixed budget regardless of how attention mass is distributed across heads, layers, and decode steps, leading to either wasted bandwidth on heads with peaky attention or dropped context on heads with diffuse attention. Top-$p$ (nucleus) selection adapts naturally to this variation but has seen limited adoption in production serving systems due to two system-level challenges: variable block counts that break deterministic block allocation a
    
[^175]: SupportChain：面向小型阅读器的多跳RAG中基于引文支撑的推理链的轨迹校准聚合

    SupportChain: Trace-Calibrated Aggregation of Quote-Grounded Reasoning Chains for Multi-Hop RAG with Small Readers

    [https://arxiv.org/abs/2611.28519](https://arxiv.org/abs/2611.28519)

    SupportChain通过将问题分解为多跳、将每跳答案绑定到检索段落中的引文、依据执行轨迹校准可靠性并按支持度聚合多次执行，使仅90亿参数的小型阅读器在多跳RAG任务上显著超越更大规模模型，且完全无需构建图索引。

    

    在多跳检索增强问答中取得强大效果的方法依赖于由700亿参数及以上规模的模型来读取的图索引，而构建这样的索引需要在整个语料库上调用大型语言模型。我们提出了SupportChain，它将问题拆分为多个跳，将每个跳的答案与从检索段落中引用的一句话绑定，并根据每次执行自身的轨迹来估计其可靠性，进而通过答案的支持度（即相互一致的那些执行的聚合可靠性）来合并多次执行，一旦支持度足够高即停止。在HippoRAG 2协议下使用90亿参数的阅读器，SupportChain在MuSiQue上达到63.0 F1，超过了此前使用200亿参数阅读器取得的最佳结果59.2。在相同的阅读器和嵌入器条件下，它在全部五个数据集上均显著优于HippoRAG 2，最高提升达17.2 F1，且无需构建任何索引。按支持度合并的方法显著优于多数投票，也优于让阅读器自行选择的方法。

    arXiv:2611.28519v1 Announce Type: new  Abstract: Strong results in multi-hop retrieval-augmented question answering come from graph indexes read by models of 70B parameters or more, and building such an index calls a large language model over the whole corpus. We present SupportChain, which splits a question into hops, binds each hop's answer to a sentence quoted from a retrieved passage, estimates how reliable each execution is from its own trace, and merges several executions by the support of their answers, the pooled reliability of the executions that agree, stopping once support is high enough. With a 9B reader under the HippoRAG~2 protocol, SupportChain reaches 63.0 F1 on MuSiQue, above the best earlier result of 59.2, which used a 20B reader. With the same reader and embedder, it is significantly better than HippoRAG~2 on all five datasets, by up to 17.2 F1, and it builds no index. Merging by support is significantly better than majority voting, and than letting the reader choos
    
[^176]: 面向多智能体系统的数字孪生：何时信任，如何改变

    Digital Twins for Multi-Agent Systems: When to Trust, What to Change

    [https://arxiv.org/abs/2611.28517](https://arxiv.org/abs/2611.28517)

    提出PAV-Twin框架，其监视器结合雅可比传播特征、近期误差与集成分歧来检测数字孪生预测的失效风险，其规划器通过模拟候选编辑并按预测集体结果排序，从而为多智能体网络选择有效的干预措施。

    

    智能体网络的数字孪生需要标记不可靠的预测，并帮助用户选择干预措施。这两种决策都取决于多步之后发生的情况：误差可能在网络中传播扩散，而一个轨迹容易预测的编辑仍可能带来糟糕的集体结果。我们研究误差传播信息何时能够改进故障检测，以及学习得到的数字孪生能否在预测不完美的情况下选择有效的网络编辑。我们提出了面向网络化数字孪生的传播感知验证方法PAV-Twin。其监视器将基于雅可比矩阵的传播特征与近期误差和集成分歧相结合，以估计预测超出误差阈值的风险。其规划器对每个候选编辑进行模拟仿真，并根据预测的集体结果对编辑进行排序。在仿真实验中，在具有熟悉拓扑的较大振荡器网络上，监视器的精确率-召回率曲线下面积提升了0.13……

    arXiv:2611.28517v1 Announce Type: cross  Abstract: Digital twins of agent networks need to flag unreliable forecasts and help users choose interventions. Both decisions depend on what happens over several steps: an error can spread through the network, and an edit whose trajectory is easy to predict may still yield a poor collective outcome. We ask when information about error propagation improves failure detection and whether a learned twin can select effective network edits despite imperfect forecasts. We propose Propagation-Aware Validation for Networked Digital Twins (PAV-Twin). Its monitor combines Jacobian-based propagation features with recent errors and ensemble disagreement to estimate the risk that a forecast exceeds an error threshold. Its planner simulates each candidate edit and ranks the edits by their predicted collective outcomes. In simulations, the monitor gained 0.13 in area under the precision-recall curve on larger oscillator networks with familiar topologies and 0
    
[^177]: 无需训练，直接编写：AI智能体将新能力写入语言模型权重

    Don't Train It, Write It: An AI Agent Writes New Capabilities into Language Model Weights

    [https://arxiv.org/abs/2611.28483](https://arxiv.org/abs/2611.28483)

    一个AI智能体通过在现有层中添加门控注意力头和MLP神经元，在完全不进行反向传播或权重训练的情况下，将新能力直接写入语言模型权重，使小模型在六个BBH子任务上的性能大幅超越更大的7B模型，同时严格保持模型原有行为零改变。

    

    我们证明，AI智能体可以通过在预训练语言模型的现有层内部添加带门控的注意力头和MLP神经元，将新能力写入模型，而无需反向传播，也无需根据任务示例的答案拟合任何权重。它仅有50次模型前向传递的预算，且不接触任何验证或测试实例。将针对六个BIG-Bench Hard（BBH）子任务的模块组合进同一模型后，Qwen2.5-1.5B-Instruct的归一化BBH分数从19.95提升至41.05（Qwen2.5-7B-Instruct最高为35.01），Gemma-3-1B-it的分数从10.25提升至32.86。在与未修改模型相同的精度下，两个组合模型在1,023,000个WikiText词元上的下一个词元分布完全不变（KL散度精确为0），在五个其他排行榜测试的15,845个题目上给出相同答案，并在其他18个BBH子任务上获得相同分数。该智能体还实现了精确的20位数乘法，而模型仅需运行最多6次（在未见过的...上达到100.0%——摘要至此截断）

    arXiv:2611.28483v1 Announce Type: cross  Abstract: We show that an AI agent can write new capabilities into a pretrained language model by adding gated attention heads and MLP neurons inside its existing layers, without backpropagation and without fitting any weight to the answers of task examples. It has a budget of 50 model passes and sees no validation or test instance. Composed in one model, its modules for six BIG-Bench Hard (BBH) subtasks raise the normalized BBH score of Qwen2.5-1.5B-Instruct from 19.95 to 41.05 (Qwen2.5-7B-Instruct: at most 35.01) and that of Gemma-3-1B-it from 10.25 to 32.86. Against the unmodified model at the same precision, both composites leave the next-token distribution on 1,023,000 WikiText tokens unchanged (KL exactly 0), give identical answers on 15,845 items of five other Leaderboard tests and identical scores on the other 18 BBH subtasks. The agent also writes exact multiplication up to 20 digits, running the model only up to 6 (100.0% on the absent
    
[^178]: 数字仿制品

    Digital Knockoffs

    [https://arxiv.org/abs/2611.28482](https://arxiv.org/abs/2611.28482)

    该论文提出一种基于激活引导与贝叶斯优化的校准方法，使语言模型代理能精确匹配多维人格特质档案，在500名受访者的大五人格数据上误差低于提示词和传统引导基线。

    

    社会模拟通常使用语言模型代理来代表在多个可测量特质上各不相同的人。然而，仅仅在提示词中描述一个特质档案并不能保证代理会按照预期的水平表现出每种特质。我们提出了一种利用激活引导将代理校准到多维特质档案的方法：(i) 为每种特质提取引导方向，(ii) 学习从引导强度到测得特质分数的共享映射，(iii) 为每个目标搜索合适的引导干预。我们使用贝叶斯优化进行该搜索，并在500名OpenPsychometrics受访者的大五人格分数上对该方法进行评估。在五轮实验中的每一轮，所分配的干预在受访者上产生的平均绝对误差均低于提示词和引导基线方法。此外，模型对给定干预的响应无论由谁的目标驱动，都会产生相同的特质档案，因此（摘要在此处被截断）

    arXiv:2611.28482v1 Announce Type: new  Abstract: Social simulations often use language model agents to represent people who differ on several measurable traits. However, describing a trait profile in a prompt does not guarantee the agent will express each trait at its intended level. We propose a method that calibrates agents to multidimensional profiles using activation steering: (i) extract steering directions for each trait, (ii) learn a shared mapping from steering strengths to measured trait scores, and (iii) search for a steering intervention for each target. We use Bayesian optimization for this search and evaluate the method on the Big Five personality scores of 500 OpenPsychometrics respondents. In each of five rounds, the assigned interventions yield lower mean absolute error across respondents than the prompting and steering baselines. Further, the model's responses to a given intervention yield the same trait profile regardless of whose target motivated it, so a poor match 
    
[^179]: 静默蒸馏：向教师屏蔽推理过程解锁高效Token智能体

    Muted Distillation: Sealing Reasoning from Teacher Unlocks Token-Efficient Agents

    [https://arxiv.org/abs/2611.28445](https://arxiv.org/abs/2611.28445)

    提出静默蒸馏（MD）方法，通过向教师模型隐藏学生的推理过程，在多轮智能体任务中将输出token减少53-57%的同时保持甚至提升准确率。

    

    在策略蒸馏（OPD）通常被采用为大语言模型（LLM）后训练的组成部分之一，但我们发现在多轮智能体任务中，学生模型在训练后会增加其推理长度。我们的实证分析表明，推理token长度受学生能在多大程度上干扰教师评分的影响。我们提出了静默蒸馏（MD），这是OPD的一个简单变体，它向教师隐藏学生的推理过程，使教师在没有学生推理干扰的情况下评估学生的行为。MD只需几行代码，且无需额外的数据或超参数。在知识工作基准OfficeQA Pro和SpreadsheetBench上，与标准OPD相比，它将每个回合的输出token分别减少了57%和53%，使学生在训练后比训练前更加简洁。同时，它在OfficeQA Pro上保持了与OPD相当的准确率，并将pass@1提升了4.8个点。

    arXiv:2611.28445v1 Announce Type: new  Abstract: On-policy distillation (OPD) is commonly adopted as one of the components in post-training of large language models (LLMs), but in multi-turn agentic tasks we find that the student model increases their reasoning lengths after training. Our empirical analysis shows that the reasoning token lengths is affected by how much the student can distract the teacher's scores. We introduce Muted Distillation (MD), a simple variant of OPD that hides the student's reasoning from the teacher, so the teacher judges what the student does without the interference of student's reasonings. MD takes a few lines of code and adds no extra data or hyperparameters. On the knowledge-work benchmarks OfficeQA Pro and SpreadsheetBench, it reduces output tokens per episode by 57\% and 53\% compared with standard OPD, making the student more concise than it was before training. At the same time, it matches OPD's accuracy on OfficeQA Pro and improves pass@1 by 4.8 po
    
[^180]: AI智能体何时能够作为人类精神病理学的模型？

    When Can an AI Agent Serve as a Model of Human Psychopathology?

    [https://arxiv.org/abs/2611.28405](https://arxiv.org/abs/2611.28405)

    该论文将精神病学及其相关领域的证据标准转化为AI实验可满足的具体准则，用以判定何时对AI智能体的实验能够真正为特定的人类精神病理过程提供科学证据。

    

    语言模型和AI智能体越来越多地被用精神病学术语来描述，例如焦虑的、抑郁的、成瘾的或无助的。最近的一项研究甚至提出将语言模型作为精神病学的实验系统，以及试运行心理治疗的平台。我们探讨的是，对AI智能体的实验何时能够为特定的人类精神病理过程提供证据。这类证据主张包含三个部分：智能体中发生了某种变化，且智能体的历史导致了这种变化；除了与人类共享的某种过程之外，没有其他因素能够解释这种变化；这种变化符合关于人类过程的某一种理论，而不符合与其竞争的理论。精神病学及其邻近领域为每个部分都设有证据标准，分别来自诊断手册中关于诱发状态的规则、临床测试分数以及动物模型。我们将这些标准逐一转化为机器研究可以满足的准则。对于语言模型而言，最关键的替代……（摘要在此处截断）

    arXiv:2611.28405v1 Announce Type: cross  Abstract: Language models and AI agents are increasingly described in psychiatric terms, such as anxious, depressed, addicted or helpless. A recent study even proposed language models as experimental systems for psychiatry and as a platform for piloting psychological treatments. We ask when an experiment on an AI agent can provide evidence about a specified human psychiatric process. A claim to such evidence has three parts. Something changed in the agent, and the agent's history caused the change. Nothing other than a process shared with people explains that change. The change matches one theory of the human process and not that theory's rivals. Psychiatry and its neighbouring fields have an evidence standard for each part, from the diagnostic manuals' rule for induced states, from clinical test scores and from animal models. We translate each standard into a criterion that a machine study can meet. For a language model the most critical altern
    
[^181]: 语言模型压缩人类偏好几何结构，而政治结构仍对提示条件敏感

    Language models compress human preference geometry while political structure remains prompt-sensitive

    [https://arxiv.org/abs/2611.28383](https://arxiv.org/abs/2611.28383)

    语言模型在重放人类政策选择时会严重压缩潜在偏好空间（合成参与者方差仅为人类的0.4%–10.5%），且无法同时再现个体选择、群体异质性和政治组织结构，其政治组织表现对提示条件高度敏感。

    

    大语言模型正越来越多地被用于构建合成公众。先前的研究已经记录了个体回答中的错误、群体异质性以及提示鲁棒性问题，但这些失败究竟反映了共同保真度的丧失，还是合成人群的独特属性，目前仍不清楚。我们通过让三个开源权重模型家族重放1,991名智利成人、843名法国成人和434名巴西成人的匹配政策成对选择来检验这一问题。在包含27个条件的智利实验网格以及跨国因子复现实验中，没有任何条件能够同时再现个体选择、群体异质性和政治组织结构。最一致的结果是潜在偏好空间被压缩：在主要比较中，合成参与者的方差仅为相应人类参照方差的0.4%至10.5%，并且在全部270次匹配重拟合中始终低于人类方差。政治结构对提示条件则敏感得多（原文摘要此处截断）。

    arXiv:2611.28383v1 Announce Type: new  Abstract: Large language models are increasingly used to construct synthetic publics. Prior work has documented errors in individual responses, population heterogeneity, and prompt robustness, but whether these failures reflect a common loss of fidelity or distinct properties of synthetic populations remains unclear. We tested this using matched pairwise policy choices from 1,991 Chilean, 843 French, and 434 Brazilian adults replayed through three open-weight model families. Across a 27-condition Chilean grid and cross-country factorial replications, no condition jointly reproduced individual choices, population heterogeneity, and political organization. The most consistent result was compression of the latent preference space: synthetic participant variance was only 0.4-10.5% of the corresponding human reference in the primary comparisons and remained below human variance in all 270 matched refits. Political structure was much more sensitive to h
    
[^182]: 把脉企业工作流：评估持久性项目状态理解

    Taking the PULSE of Enterprise Workflows: Evaluating Persistent Project-State Understanding

    [https://arxiv.org/abs/2611.28374](https://arxiv.org/abs/2611.28374)

    本文提出了持久性项目状态理解（PPSU）任务及PULSE基准（包含1,573个问题、覆盖30个企业项目历史），实验发现检索与抽取是当前系统的主要瓶颈，但即使在完美证据之上仍需显式项目状态才能取得进一步提升，证明该任务无法简化为单纯的检索问题。

    

    企业智能体必须对分布在异构工件中、随时间不断演变、且对不同协作者可见程度不均的工作保持最新理解。我们将这一挑战形式化为持久性项目状态理解，并引入PULSE，一个包含1,573个问题、覆盖30个企业项目历史记录的基准。PULSE从带源链接的记录中构建确定性的、按时间索引的项目状态以及有文档记录的协作者视图，并评估三类问题：未闭合事项、项目演变和视图差距。在检索智能体、图增强与记忆增强检索、长上下文以及oracle条件下，我们发现当前系统仍有巨大的提升空间。检索和抽取占据了差距的最大份额，但在金标准证据之上再提供显式项目状态能带来进一步的显著收益，这表明PPSU不能被简化为单纯的检索问题。进一步的分析显示

    arXiv:2611.28374v1 Announce Type: cross  Abstract: Enterprise agents must maintain an up-to-date understanding of work distributed across heterogeneous artifacts, evolving over time, and unevenly visible to different collaborators. We formulate this challenge as persistent project-state understanding (PPSU) and introduce PULSE, a benchmark of 1,573 questions over 30 enterprise project histories. PULSE constructs deterministic, time-indexed project state and documented collaborator views from source-linked records, and evaluates three families of questions: open loops, project evolution, and view gaps. Across retrieval agents, graph- and memory-augmented retrieval, long-context, and oracle conditions, we find substantial headroom for current systems. Retrieval and extraction account for the largest share of the gap, but providing explicit project state on top of gold evidence yields further substantial gains, showing that PPSU cannot be reduced to retrieval alone. Further analysis shows
    
[^183]: 基于双分支监督的对比式同策略蒸馏

    Contrastive On-Policy Distillation via Dual-Branch Supervision

    [https://arxiv.org/abs/2611.28359](https://arxiv.org/abs/2611.28359)

    该论文提出对比式同策略蒸馏（cOPD），通过在学生生成的文本前缀上同时对比强教师、弱教师与学生三方的预测实现双分支监督，证明比学生更弱的模型也能作为有价值的对比参考，从而提升蒸馏效果。

    

    对于同策略蒸馏，能力更强的模型自然是教师模型的首选。然而，更高的任务性能并不一定意味着更强的可教性。一项受控的token干预实验表明，即使是比学生模型更弱的模型，也可以作为有信息量的对比参考。基于这一观察，我们提出了对比式同策略蒸馏，该方法通过在学生生成的文本前缀处对比强教师、弱教师和学生模型三方的预测，提供双分支监督。我们的分析解释了在理想化模型下，这种三方比较如何帮助选择与更好或更差推理结果相关联的候选token。我们在涵盖不同训练阶段、参数规模和模型家族的多种师生配置上评估了cOPD，任务包括数学推理、指令遵循以及带拒答的问答。cOPD取得了……

    arXiv:2611.28359v1 Announce Type: new  Abstract: For on-policy distillation (OPD), more capable models are a natural choice as teachers. However, higher task performance does not necessarily imply greater teachability. A controlled token intervention study suggests that even a model weaker than the student can serve as an informative contrastive reference. Building on this observation, we propose Contrastive On-Policy Distillation (cOPD), which provides dual branch supervision by contrasting strong teacher, weak teacher, and student predictions at student generated prefixes. Our analysis explains how this three way comparison helps select token candidates associated with better or worse reasoning outcomes under an idealized model. We evaluate cOPD across diverse teacher student configurations spanning different training stages, parameter scales, and model families, on tasks including mathematical reasoning, instruction following, and question answering with abstention. cOPD achieves th
    
[^184]: Grammar-as-a-Draft（语法即草稿）：在语法约束的结构化输出中让每次推测解码步产生更多Token

    Grammar-as-a-Draft: More Tokens per Speculative Decoding Step on Grammar-Constrained Structured Outputs

    [https://arxiv.org/abs/2611.28349](https://arxiv.org/abs/2611.28349)

    该论文提出Grammar-as-a-Draft（GaaD），将语法引擎作为第二草稿模型，把语法强制Token与推测草稿模型的提议合并为更宽的草稿行，使语法约束的结构化输出下每次推测解码步可验证更多Token，且无需新增内核或编译。

    

    在结构化输出（JSON模式、正则表达式、工具调用）场景下，推测式草稿模型无法感知语法约束，会反复提议无效的键名、分隔符和缩进，而语法引擎其实早已知道接下来必须出现的Token。我们提出了Grammar-as-a-Draft（GaaD），它将语法视为第二个草稿模型。在每次验证步骤之前，调度器将草稿模型的提议与语法强制Token合并为一行草稿：匹配的Token保留为GRAMMAR状态，在提议偏离强制路径之处将其截断，并由强制Token填满比草稿提议更宽的行的剩余部分。验证步骤无需计算GRAMMAR草稿的logits即可接受它们，可与任何草稿模型组合，且在关键路径上无需新增内核或编译。由于语法固定的是文本而非Token，提交规范的强制Token相对于使用语法掩码的目标解码会改变输出分布。

    arXiv:2611.28349v1 Announce Type: cross  Abstract: Under structured outputs (JSON schemas, regexes, tool calls), speculative drafters do not see the grammar constraint and repeatedly propose invalid keys, delimiters, and indentation, while the grammar engine already knows which tokens must come next.   We present Grammar-as-a-Draft (GaaD), which treats the grammar as a second drafter. Before each verify step, the scheduler merges the drafter's proposal with grammar-forced tokens into one draft row: matching tokens stay as GRAMMAR, the proposal is cut where it leaves the forced path, and forced tokens fill the rest of a row wider than the drafter's proposal. The verify step accepts GRAMMAR drafts without computing their logits, composing with any drafter with no new kernels or compilation on the critical path. Because grammars fix text rather than tokens, committing canonical forced tokens changes the output distribution relative to grammar-masked target decoding.   We implement GaaD in
    
[^185]: 当工具失效时，智能体便捏造结果：工具使用型LLM智能体中的静默失败与“验证优先”监督机制

    When Tools Fail, Agents Fabricate: Silent Failure and Verification-First Oversight in Tool-Using LLM Agents

    [https://arxiv.org/abs/2611.28341](https://arxiv.org/abs/2611.28341)

    该论文构建了RetryRealistic受控故障注入测试平台，揭示了工具使用型LLM智能体在工具失效时恢复能力不可靠且常发生“静默捏造”（将错误答案自信地当作事实输出），失败模式主要取决于故障类型而非模型本身，并据此倡导“验证优先”的监督机制。

    

    使用工具的LLM智能体在工具调用失败时必须能够恢复，例如重试、切换工具或修改计划。我们借助RetryRealistic研究了当工具失效时智能体如何失败——这是一个沙箱化但贴近真实的受控故障注入测试平台。在该平台中，智能体在具有真实权限执行机制的真实操作系统文件系统、SQLite数据库、确定性本地HTTP服务器和受约束的shell上运行，涵盖8种工具、9种具有受控持续性的故障类型以及30个多步骤任务。通过对四个智能体共2,100次故障注入运行进行与真实基准的对比评估，我们发现恢复能力并不可靠（正确率仅为19-68%），且失败常常是静默的：6-26%的运行以“自信的捏造”告终，即错误答案被当作事实进行断言。这种失败模式更多由故障类型而非模型本身决定：返回部分数据的故障会诱发捏造行为（高达38%），而硬性拒绝则会促使智能体诚实地报告失败。其背后机制是我们所称的“静默摄取”：智能体接受……（原文摘要在此处截断）

    arXiv:2611.28341v1 Announce Type: new  Abstract: Tool-using LLM agents must recover when a tool call fails, by retrying, switching tools, or revising their plan. We study how agents fail when tools fail, using RetryRealistic, a sandboxed yet realistic testbed for controlled fault injection in which agents operate on a real OS filesystem with genuine permission enforcement, SQLite databases, a deterministic local HTTP server, and a constrained shell, across 8 tools, 9 fault types with controlled persistence, and 30 multi-step tasks. Evaluating four agents over 2,100 faulted runs against ground truth, we find that recovery is unreliable (19-68% correct) and frequently silent: 6-26% of runs end in a confident fabrication, a wrong answer asserted as fact. The failure mode is governed more by the fault type than by the model: faults that return partial data drive fabrication (up to 38%), while hard denials drive honest failure. The mechanism is what we call silent ingestion: the agent accep
    
[^186]: INSEP：以拒绝方向不可分性作为对抗Abliteration攻击的防御方法

    INSEP: Refusal-Direction Inseparability as a Defense Against Abliteration

    [https://arxiv.org/abs/2611.28314](https://arxiv.org/abs/2611.28314)

    INSEP通过微调模型降低有害与良性激活沿拒绝方向的可分性，从而削弱Abliteration攻击所依赖的估计信号，实现对该白盒越狱攻击的有效防御。

    

    Abliteration是一种被广泛使用的针对大语言模型（LLM）安全对齐的白盒攻击。它通过在模型激活空间中估计一个“拒绝方向”，并将该方向从残差流的写出矩阵中投影去除，从而诱导出有害行为。我们提出INSEP，一种通过微调模型来降低有害激活与良性激活沿拒绝方向可分性的防御方法。Abliteration将该方向估计为类别均值之差并将其投影去除，而降低可分性会削弱该攻击所依赖的信号。该方法利用对抗性特征分解来选取可分性最强的提示子集，并通过低秩适配器降低这些子集沿该方向的分离程度。在训练过程中，行为损失用于保持基础模型的拒绝能力以及其良性回答的质量。我们在五种攻击下、跨越八个参数量从1B到14B的LLM上，将INSEP与六种已发表的防御方法进行了基准比较……

    arXiv:2611.28314v1 Announce Type: new  Abstract: Abliteration is a widely used white-box attack against the safety alignment of large language models (LLMs). It elicits harmful behavior by estimating a refusal direction in the model's activation space and projecting that direction out of the residual-stream write-out matrices. We propose INSEP, a defense that fine-tunes the model to reduce the separability of harmful and benign activations along the refusal direction. Abliteration estimates this direction as a difference in class means and projects it out. Reducing separability weakens the signal the attack relies on. An adversarial eigendecomposition selects the most separable prompt subsets. A low-rank adapter reduces their separation along that direction. Behavioral losses preserve the base model's refusal and the quality of its benign responses during training. We benchmark INSEP against six published defenses under five attacks, across eight LLMs from 1B to 14B parameters and seve
    
[^187]: 语言模型框架可以成为组合泛化器

    Language Model Harnesses can be Compositional Generalizers

    [https://arxiv.org/abs/2611.28298](https://arxiv.org/abs/2611.28298)

    本文提出训练神经符号化的语言模型框架（如递归语言模型RLM），通过程序化外置和管理上下文、将复杂问题分解为局部分布内的子问题来实现组合泛化，使得仅在短任务上训练的模型也能泛化到远长于训练任务的新型长时程任务。

    

    现代语言模型（LM）在“框架”内进行后训练并作为智能体部署，但标准的框架设计会不断累积上下文，迫使神经网络对其进行泛化，这使得长时程任务的训练变得昂贵且耗时。我们研究了一条新兴的替代路径：训练本质上神经符号化的语言模型框架，它们以程序化的方式外置和管理上下文，从而组合现有的语言模型能力。我们假设，围绕这些框架进行训练可以通过实现组合泛化来提供更好的计算回报，因为它们能够学会分解复杂问题，使得神经语言模型接收到的每一个观测在局部上都处于分布内，即使整个任务是全新的。我们发现，像递归语言模型（RLM）这样的框架在强化学习后训练中能够轻松学会这种分解：（1）我们证明仅在短任务上训练即可泛化到任务 2-6（原文在此处截断）

    arXiv:2611.28298v1 Announce Type: cross  Abstract: Modern language models (LMs) are post-trained and deployed as agents inside a "harness", but standard harness designs accumulate context that the neural network is forced to generalize to, making training for long-horizon tasks expensive and time-consuming. We study an alternative emerging path: training inherently neurosymbolic LM harnesses, which externalize and manage context programmatically to compose existing LM capabilities. We hypothesize that training around these harnesses can offer a better return on compute by enabling compositional generalization, as they can learn to decompose complex problems so that every observation their neural LM receives is locally in-distribution, even when the full task is novel. We find that harnesses like the Recursive Language Model (RLM) learn such decompositions readily under reinforcement learning post-training: (1) we demonstrate that training only on short tasks can generalize to tasks 2-6
    
[^188]: 面向教育人工智能的反馈驱动提示词优化：情感轨迹估计与布鲁姆分类法分类

    Feedback-Driven Prompt Refinement for Educational AI: Affective Trajectory Estimation and Bloom's Taxonomy Classification

    [https://arxiv.org/abs/2611.28293](https://arxiv.org/abs/2611.28293)

    本研究提出一种由情感与认知反馈信号驱动的共享提示词优化方法，无需更新模型参数即可提升教育AI的情感轨迹估计与布鲁姆分类性能，并降低情感生成的不稳定性。

    

    贡献：本研究表明，一种由不同情感和认知反馈信号引导的共享提示词优化程序，能够在不更新执行器参数的情况下，提高对留出参考的一致性和多标签分类性能，同时降低情感生成的不稳定性。背景：工业4.0增加了对互联计算与工程领域个性化技术教育的需求，而提示词敏感性和生成不稳定性使得对教学对话和学习成果的可靠解读变得复杂。研究问题：本研究探讨了任务特定的反馈能否在不更新执行器参数的情况下，提高情感参考一致性、生成一致性和认知分类性能。方法：一个次级语言模型利用执行器输出与模型生成的情感参考之间的差异来修订执行器提示词，应用于四个对话……

    arXiv:2611.28293v1 Announce Type: new  Abstract: Contribution: This study has shown that a shared prompt-refinement procedure, guided by distinct affective and cognitive feedback signals, can improve held-out reference agreement and multi-label classification while reducing affective-generation variability without executor parameter updates. Background: Industry 4.0 has increased demand for personalized technical education across interconnected computing and engineering domains, while prompt sensitivity and generation variability have complicated dependable interpretation of instructional dialogue and learning outcomes. Research Questions: The study has examined whether task-specific feedback can improve affective-reference agreement, generation consistency, and cognitive classification without updating executor parameters. Methodology: A secondary language model has revised executor prompts using executor-output discrepancies against model-generated affective references for four dialo
    
[^189]: 在说话人匿名化中保留韵律

    Preserving Prosody in Speaker Anonymization

    [https://arxiv.org/abs/2611.28290](https://arxiv.org/abs/2611.28290)

    该论文通过将预训练的隐私保护韵律编码器集成到语音合成模型中，实现了在有效混淆说话人身份的同时保留韵律内容的说话人匿名化方法。

    

    有效的韵律建模在自然语音生成应用中至关重要，但在说话人匿名化中会出现复杂问题，因为说话人属性存在于韵律的声学关联特征中。我们的工作旨在改善此前报道的说话人有效混淆与韵律内容保留之间的权衡。为此，我们将一个现有的预训练隐私保护韵律编码器集成到语音合成模型中，实现了保留韵律的说话人匿名化。通过以韵律特征为条件进行语音生成，我们发现人类受试者能够更可靠地感知预期的韵律含义，同时匿名化性能与此前的相关工作相当。

    arXiv:2611.28290v1 Announce Type: cross  Abstract: Effective prosody modeling is essential in natural speech generation applications, but complications arise in speaker anonymization, since speaker attributes are present in acoustic correlates of prosody. Our work aims to improve previous reported tradeoffs between effective speaker obfuscation and preservation of prosodic content. To this end, we integrate an existing pretrained privacy-preserving prosody encoder into a speech synthesis model, enabling prosody-preserving speaker anonymization. By conditioning speech generation on prosody features, we find that human subjects more reliably perceive intended prosodic meaning, while anonymization performance is competitive with previous work.
    
[^190]: 当视频验证器得分产生误导时：能力评估与验证器选择

    When Video Verifier Scores Mislead: Capability Assessment and Verifier Selection

    [https://arxiv.org/abs/2611.28287](https://arxiv.org/abs/2611.28287)

    视频验证器的评估得分严重依赖负例类型与判断界面这两个测试设计选择，甚至足以颠覆关于模型能力和验证器选型的结论；借助端点匹配负例与真值切换测试等精确真值条件构造，研究发现让模型先写出推理再作答可将多数模型表现从不足10%提升至70-100%。

    

    视频验证器用于判断候选答案或轨迹是否与视频一致，而验证器本身也是依据一个分数来挑选的。我们证明，这一分数强烈依赖于测试中的两个设计选择——负例的类型和判断界面——其影响之大，足以颠覆关于模型能力以及应部署哪个验证器的结论。我们通过具有精确真值条件的受控构造将这两个因素分离：端点匹配负例（仅在内部状态上有误）以及真值切换测试（其中相同的两个候选答案在两个视频之间互换正确性）。在直接判断方式下，即从答案分数中读取判定结果时，那些能够拒绝明显错误开头的模型，在面对端点匹配负例时准确率降至接近随机水平。当视频帧揭示每个事件时，先写出推理过程再给出答案，可使四个模型中的三个在真值切换测试上的表现从低于10%提升到70-100%，而分阶段输入将直接判断的失败追溯到某一个步骤（原文在此处截断）。

    arXiv:2611.28287v1 Announce Type: cross  Abstract: Video verifiers judge whether a candidate answer or trajectory agrees with a video, and are themselves chosen by a score. We show that this score depends on two choices in the test, the type of negative and the judgment interface, strongly enough to reverse conclusions about what a model can do and which verifier to deploy. We separate the two with controlled constructions that have exact truth conditions: endpoint-matched negatives, wrong only in their internal states, and a truth-switch test, in which the same two candidates swap correctness between two videos. Under direct judgment, where the verdict is read from answer scores, models that reject a visibly wrong start fall to near chance against endpoint-matched negatives. When the frames disclose each event, writing out reasoning before the answer lifts three of four models from under 10% to 70-100% on the truth-switch test, and staged inputs trace the direct failure to one step, c
    
[^191]: 面向马尔可夫推理的专用大语言模型：结构化缓存压缩、转移预训练与双循环Transformer

    Specialized Large Language Models for Markovian Reasoning: Structural Cache Compression, Transition Pretraining, and Dual-Loop Transformers

    [https://arxiv.org/abs/2611.28285](https://arxiv.org/abs/2611.28285)

    本文提出面向马尔可夫推理任务的专用大语言模型统一设计范式，通过常数大小的马尔可夫状态接口实现KV缓存的结构化压缩、将长链预测重构为可独立训练的单步转移预训练目标，并引入双循环Transformer架构，以突破长链推理中的内存与吞吐量瓶颈。

    

    标准自回归大语言模型会关注不断增长的已生成token历史，这使得键值（KV）缓存成为长链推理和多分支采样中的核心内存与吞吐量瓶颈。然而，一类重要的推理问题天然具有一阶马尔可夫性质：下一步仅依赖于对当前状态的充分表示，而非所有先前步骤的原始序列。我们为马尔可夫推理专用大语言模型建立了一个统一的设计范式。在缓存层面，马尔可夫状态接口将与历史相关的KV存储转变为相对于推理步数保持常数大小的状态传递。在学习层面，我们将长链预测重新表述为可独立训练的单步转移目标，并引入显式的状态瓶颈、蒸馏、不变性以及漂移一致性损失。在架构层面，我们引入了一种……（原文摘要在此处截断）

    arXiv:2611.28285v1 Announce Type: new  Abstract: Standard autoregressive large language models attend to the growing history of generated tokens, making the key-value (KV) cache a central memory and throughput bottleneck in long-chain reasoning and multi-branch sampling. Yet an important class of reasoning problems is naturally first-order Markovian: the next step depends on a sufficient representation of the current state, rather than on the raw sequence of all earlier steps. We develop a unified design paradigm for Markovian-reasoning specialized large language models. At the cache level, the Markovian state interface transforms history-dependent KV storage into constant-size state transfer with respect to the number of reasoning steps. At the learning level, we reformulate long-chain prediction as independently trainable one-step transition objectives, with explicit state bottlenecks, distillation, invariance, and drift-consistency losses. At the architectural level, we introduce a 
    
[^192]: 从点云到工程模型：建筑环境中的智能体3D重建基准测试

    From Point Clouds to Engineering Models: Benchmarking Agentic 3D Reconstruction in the Built Environment

    [https://arxiv.org/abs/2611.28197](https://arxiv.org/abs/2611.28197)

    该论文提出首个场景级Scan-to-BIM智能体3D编码基准S2Bench（涵盖五个领域128个真实场景），并引入并行异步协作机制，让多智能体分工建模、持续迭代优化工程模型重建。

    

    编码智能体可以将3D测量数据转化为CAD几何图形，但为建筑环境重建工程模型需要超越表面拟合的语义解释和一致的构造约束。我们将场景级Scan-to-BIM（建筑信息建模）形式化为一个智能体3D编码任务，并引入S2Bench基准，该基准包含五个领域共128个真实场景的公开和专有扫描数据，并配有专家精选的、涵盖多样空间尺度的参考模型。一个共享框架使智能体能够使用Python测量扫描数据，在实时CAD/BIM内核中构建带类别标签的构件，并对其代码进行迭代优化。我们还提出了并行异步协作方法：由主导者分配带版本管理的任务，在工作节点执行的同时继续建模，并整合受修订约束的提案。对五个智能体和一个协作团队共1,028次重建的评估表明，扫描到模型的残差直接（原文在此处截断）

    arXiv:2611.28197v1 Announce Type: new  Abstract: Coding agents can translate 3D measurements into CAD geometry, but reconstructing engineering models for the built environment demands semantic interpretation and consistent construction constraints beyond surface fitting. We formulate scene-level Scan-to-BIM (Building Information Modeling) as an agentic 3D coding task and introduce S2Bench, comprising public and proprietary scans of 128 real-world scenes across five domains, paired with expert-curated reference models spanning diverse spatial scales. A shared framework lets agents measure scans in Python, build class-labeled components in a live CAD/BIM kernel, and iteratively refine their code. We also propose parallel asynchronous collaboration, where a lead delegates versioned tasks, continues modeling while workers execute, and integrates revision-bound proposals. Evaluating five agents and a collaborative team across 1,028 reconstructions shows that scan-to-model residuals directly
    
[^193]: 测量并教导语言模型智能体对延迟后果采取行动

    Measuring and Teaching Language Model Agents to Act on Delayed Consequences

    [https://arxiv.org/abs/2611.28158](https://arxiv.org/abs/2611.28158)

    该论文提出 SMB Bench——一个包含 18 个因果配对任务的 90 天批发模拟基准，发现 16 个前沿模型中仅 3 个优于固定采购规则，并揭示模型虽能诊断问题却难以将诊断及时转化为行为改变与业务改善。

    

    语言模型智能体越来越多地被要求执行持续数天的运营工作，而非完成孤立的单轮任务。在日常运营中，今天的决策会重塑后续的观察结果和现金约束：空货架会掩盖流失的进店需求，被短缺发货的顾客会在数周后悄然缩减未来的订单，而紧急采购则会消耗营运资金。我们提出了 SMB Bench，这是一个经过校准的 90 天批发模拟器，包含 18 个因果配对任务和一个 859 次运行的前沿模型轨迹数据集，用以探究何种经验和学习信号能让智能体在未见过的业务中更好地察觉、归因并对其自身决策的延迟后果采取行动。在 16 个前沿模型中，只有 3 个模型的表现优于固定的采购规则；模型往往在笔记中记录了相关的诊断，却没有及时据此改变后续的采购行为，这表明诊断问题、改变行为和改善业务是三种相互独立的成就。为……

    arXiv:2611.28158v1 Announce Type: cross  Abstract: Language model agents are increasingly asked to run ongoing multi-day operations rather than finish isolated single-turn tasks. In daily operations, today's choices reshape later observations and cash constraints because empty shelves hide lost walk-in demand, short-shipped customers quietly shrink future orders weeks later, and emergency buying drains working capital. We introduce SMB Bench, a calibrated 90-day wholesale simulator with 18 causally paired tasks and an 859-run frontier trace dataset, to ask what experience and learning signal make an agent better at noticing, attributing, and acting on the delayed consequences of its own decisions in unseen businesses. Across 16 frontier models, only 3 outperform a fixed buying rule; models often record a relevant diagnosis in their notes without changing later purchases in time, showing that diagnosing a problem, changing behavior, and improving a business are separate achievements. To
    
[^194]: STEGOGUARD-LM：评估针对大语言模型隐蔽信道的端到端防护

    STEGOGUARD-LM: Evaluating End-to-End Protection Against LLM Covert Channels

    [https://arxiv.org/abs/2611.28152](https://arxiv.org/abs/2611.28152)

    本文提出 StegoGuard-LM 受控评估框架，通过追踪隐蔽信道消息在检测、干预与恢复全流程中的表现，实证发现低误报检测器触发的干预几乎无法阻止隐写恢复，而无条件改写虽能阻止绝大多数消息恢复，但攻击者仍可通过切换信道部分恢复信息。

    

    一种能够破坏隐藏消息的防御手段，如果其检测器很少触发干预，可能仍无法提供实质性保护。我们提出了 StegoGuard-LM，一个用于评估针对大语言模型输出中隐蔽消息防御的受控框架。我们考察了词元排名信道和同义词信道，并以 Unicode 隐蔽方式作为结构性对照，同时跟踪消息在检测、干预和恢复各环节的传播过程。确定性改写在无条件应用时，成功阻止了 104 条同义词编码消息中的 103 条被完全恢复；但当干预依赖于一个为限制误报而设置的检测器时，仅阻止了 1 条消息的恢复。在一个包含 48 条响应子集的四轮重放实验中，从固定目录中选择信道的攻击者通过切换信道恢复了更多比特，不过在经过多重比较校正后，成对增益在统计上并不显著。随机化改写在本初步实验中未表现出持续优势。更灵敏的检测器提高了……（原文在此处截断）

    arXiv:2611.28152v1 Announce Type: cross  Abstract: A defense that disrupts hidden messages may still provide little protection if its detector rarely triggers an intervention. We present StegoGuard-LM, a controlled framework for evaluating defenses against covert messages in large language model outputs. We examine token-rank and synonym channels, with Unicode concealment as a structural control, and track messages through detection, intervention, and recovery. Deterministic rewriting prevented full recovery of 103 of 104 synonym-encoded messages when applied unconditionally, but only one when intervention depended on a detector set to limit false alarms. In a four-round replay on a 48-response subset, an attacker selecting from a fixed catalog recovered more bits by switching channels, although paired gains were not statistically significant after correction for multiple comparisons. Randomized rewriting showed no sustained advantage in this pilot. A more sensitive detector improved p
    
[^195]: VidRouteBench：面向预算约束视频生成路由的基准测试

    VidRouteBench: A Benchmark for Budgeted Video-Generation Routing

    [https://arxiv.org/abs/2611.28099](https://arxiv.org/abs/2611.28099)

    该论文发布了首个支持预算约束下视频生成路由研究的基准数据集VidRouteBench，包含7个开源家族的16种配置、27,110个视频片段及经人类偏好校准的质量评分，使在固定预算下通过提示词路由选择最优模型与采样配置成为可能。

    

    文本到视频生成器的价格相差两个数量级，而强模型的一个廉价配置对于许多提示词而言往往已经足够好。在这些“臂”（模型×采样配置）之间路由提示词，应当能够在固定预算下换取更高质量的输出，但目前尚无数据集支持这一研究：公开的视频偏好数据集仅包含闭源模型，且每个模型只有一种配置。我们发布了VidRouteBench：包含来自7个开源模型家族的16个臂，实测成本跨度达121倍（每段视频0.0025至0.298美元）；涵盖1,625个分组提示词的27,110个视频片段（共1,421个A800 GPU小时）；一个基于人类偏好校准后冻结的质量评分（交叉验证准确率68.7%，高于最佳单一评分器的66.2%）；9个可识别质量维度上的逐片段评分；重复随机种子；加密的测试集划分；以及为冷启动评估密封保存的最新模型家族。其评估器将路由器与最优的提示词无关的臂组合进行比较，而非仅与最优单一臂比较。此处的每提示词专门化是真实的（原文摘要在此处截断）……

    arXiv:2611.28099v1 Announce Type: cross  Abstract: Text-to-video generators differ by two orders of magnitude in price, and a cheap configuration of a strong model is often good enough for a prompt. Routing prompts across such arms (model x sampling configuration) should buy quality at fixed budget, but no dataset supports studying it: public video-preference sets contain only closed models, one configuration each.   We release VidRouteBench: 16 arms from 7 open families spanning 121x in measured cost ($0.0025-$0.298 per clip), 27,110 clips (1,421 A800-hours) over 1,625 grouped prompts, a quality score frozen after calibration against human preferences (68.7% CV accuracy vs 66.2% for the best single scorer), per-clip scores on 9 identifiable quality dimensions, repeated seeds, an encrypted test split, and a sealed newest family for cold start. The evaluator compares a router against the best prompt-agnostic mixture of arms, not the best single arm. Per-prompt specialisation here is rea
    
[^196]: SnakeTranslate：基于离线平行文本的同声传译弱监督策略学习

    SnakeTranslate: Weakly Supervised Policy Learning from Offline Parallel Text for Simultaneous Translation

    [https://arxiv.org/abs/2611.28055](https://arxiv.org/abs/2611.28055)

    SnakeTranslate通过对齐适配的弱监督方法，仅利用普通平行文本即可联合训练同声传译的前缀翻译能力与READ/WRITE决策策略，结合SFT与反向KL散度同策略蒸馏以及支持缓存复用的增量解码，使4B模型在多个基准上实现了优于流式策略的质量-延迟权衡。

    

    自适应同声传译需要可靠的前缀翻译能力和及时的READ/WRITE（读/写）决策，但普通的平行语料库并不提供动作标注。我们提出了SnakeTranslate，它将损失推导的对齐适配到采样的因果历史中，从而联合学习这两种能力。该框架支持在平行文本上进行监督微调（SFT），以及在模型自身生成的译文上进行采用反向Kullback–Leibler（KL）散度的同策略蒸馏（OPD）。在推理阶段，源语言与目标语言的词元构成一个仅追加的交错历史，从而实现支持缓存复用的增量解码。在FLEURS、NTREX128和WMT24++数据集上，我们的4B模型相比应用于同一离线微调骨干模型的流式策略，改善了质量-延迟权衡，且OPD带来了进一步提升。该模型在低延迟和中延迟条件下与更大的翻译模型相比表现出色，但在较高延迟下，WMT24++上仍存在质量差距。

    arXiv:2611.28055v1 Announce Type: cross  Abstract: Adaptive simultaneous translation requires reliable prefix translation and timely READ/WRITE decisions, but ordinary parallel corpora provide no action annotations. We present SnakeTranslate, which adapts loss-derived alignment to sampled causal histories for jointly learning both capabilities. The framework supports supervised fine-tuning (SFT) on parallel text and on-policy distillation (OPD) with reverse Kullback--Leibler (KL) divergence on student-generated translations. At inference, source and target tokens form one append-only interleaved history, enabling incremental decoding with cache reuse. On FLEURS, NTREX128, and WMT24++, our 4B model improves quality--latency trade-offs over streaming policies applied to the same offline-tuned backbone, with further gains from OPD. It also compares favorably with larger translation models at low and moderate latency, while higher-latency quality gaps remain on WMT24++.
    
[^197]: 面向英语-瑞典语医学问答的大语言模型基准测试与适配

    Benchmarking and Adapting LLMs for English-Swedish Medical Question Answering

    [https://arxiv.org/abs/2611.28050](https://arxiv.org/abs/2611.28050)

    本文构建了首个条目对齐、经保守翻译优化流程校验的英语-瑞典语医学选择题基准，发现小型开源模型在瑞典语上准确率显著下降，并证明利用翻译数据与开源教师模型生成的推理轨迹进行微调可有效弥合这一语言差距。

    

    大语言模型在英语医学考试中表现出色，但其在瑞典语上的表现却鲜为人知，而这对医院可在本地部署的小型开源模型尤为重要。为了缩小这一差距，我们首先基于十个医学选择题数据集构建了一个条目对齐的英语-瑞典语基准测试，其中每个瑞典语条目都保留其英语源条目的选项和答案。我们采用一个保守的迭代优化流程来修复翻译：由来自三个家族的开源模型查找错误，将其锚定于瑞典语医学术语库进行验证，且只有在盲评裁判投票通过的情况下才保留模型撰写的修改。在该基准上，小型模型在瑞典语上的准确率损失远大于大型模型，尤其是在不作推理直接作答的情况下。其次，我们探究英语数据和开源教师模型能否弥合这一差距：我们翻译训练数据，让教师模型撰写推理轨迹，并在金标准答案或推理轨迹上，以英语、瑞典语或双语形式微调两个小型学生模型。在推理轨迹上训练（原文摘要在此处被截断）

    arXiv:2611.28050v1 Announce Type: cross  Abstract: Large language models do well on English medical exams, but less is known about Swedish. This matters most for small open models that hospitals can run locally. To reduce this gap, we first build an item-aligned English-Swedish benchmark from ten medical multiple-choice datasets; each Swedish item keeps the options and answer of its English source. A conservative refinement loop repairs the translations: open models from three families find errors, ground them in a Swedish medical term base, and keep a model-written edit only if blind judges vote for it. On this benchmark, small models lose much more accuracy in Swedish than large ones, especially when they answer without reasoning. Second, we ask whether English data and open teacher models can close this gap. We translate the training data, let teachers write reasoning traces, and fine-tune two small students on gold answers or traces in English, Swedish, or both. Training on reasoni
    
[^198]: 用于人类行为模拟的分布可验证奖励

    Distributionally-Verifiable Rewards for Human-Behavior Simulation

    [https://arxiv.org/abs/2611.28048](https://arxiv.org/abs/2611.28048)

    提出了分布可验证奖励（DVR），一种利用人口群体分布差距指导个体回答的强化学习奖励方法，使大语言模型在模拟调查受访者时能同时兼顾个体预测准确性、总体分布对齐和人口统计差异的捕捉。

    

    大语言模型在社会科学研究中越来越多地被用于模拟调查受访者。一个有用的模拟器应该能够预测个体回答、重现总体回答分布，并捕捉人口统计学差异。常见的训练方法难以平衡这些目标：精确匹配奖励会促使模型集中于最频繁出现的人类回答，这与分布对齐的目标相矛盾；而聚合分布匹配即使在忽略人口统计信息的情况下也可能看起来表现良好。为此，我们提出了分布可验证奖励（DVR），这是一种强化学习奖励，通过在每个问题-群体边界内将策略生成的回答频率与人类回答频率对齐，利用人口群体分布差距来指导个体回答的生成。我们使用 OpinionQA 数据集对该方法进行评估，发现与仅使用准确率的方法相比……

    arXiv:2611.28048v1 Announce Type: cross  Abstract: Large language models are increasingly used to simulate survey respondents in social-science research. A useful simulator should predict individual responses, reproduce population response distributions, and capture demographic differences. Common training approaches struggle to balance these goals: exact-match rewards encourage the model to concentrate on the most frequent human response, which is contradictory to the goal of distributional alignment. Meanwhile, aggregate distribution matching can look good even when demographic information is ignored. To this end, we introduce Distributionally-Verifiable Rewards (DVR), a reinforcement-learning reward that uses demographic group distribution gaps to guide individual responses, designed by aligning policy generated frequencies with human response frequencies within each question-group margins. We evaluate the method using the OpinionQA dataset and find that comparing with accuracy-only
    
[^199]: 攻击成功的标准是什么？评估大语言模型红队测试中的危害性

    What Counts as Attack Success? Evaluating Harmfulness in Red Teaming LLMs

    [https://arxiv.org/abs/2611.28038](https://arxiv.org/abs/2611.28038)

    该研究审查76篇红队测试论文后发现，不同自动化评估器对同一模型回复可能得出不同的危害性判断，从而改变模型比较的结论，因此红队测试研究应明确声明并验证其危害性评判标准。

    

    arXiv:2611.28038v1 公告类型：交叉 摘要：判断模型回复是否有害需要结合情境与规范性考量，然而红队测试研究通常将这一判断委托给自动化评估器来完成。我们考察了红队测试研究中危害性标准是如何设定的，以及这些标准如何延续影响到最终报告的结果。通过审查76篇论文，我们发现许多研究采用的是先前数据集、分类器或评判提示词中的标准，而非自行定义危害性。将这些论文中的评估器应用于相同的输出时，我们发现它们对同一回复可能得出不同的判断，这在比较不同模型时甚至可能改变研究结论。这些差异部分源于评估器划定危害边界的位置不同：有些评估器将任何未拒绝的回复都计为有害，而另一些则只将可能实际造成危害的输出计为有害。我们认为，红队测试研究应当明确说明其危害性标准，并报告其选择和验证标准的方式。

    arXiv:2611.28038v1 Announce Type: cross  Abstract: Judging whether a model response is harmful requires contextual and normative choices, yet red-teaming studies often delegate this judgment to automated evaluators. We examine how harmfulness criteria are specified in red-teaming research and how they carry through to reported results. Reviewing 76 papers, we find that many adopt their criteria from prior datasets, classifiers, or judge prompts rather than defining harmfulness themselves. Applying evaluators from these papers to identical outputs, we find that they can reach different judgments on the same response, which can change conclusions when comparing models. These differences arise partly because evaluators draw the boundary of harm in different places: some count any response that does not refuse, while others count only output that could plausibly cause harm. We argue that red-teaming studies should state their harmfulness criteria and report how they chose and validated the
    
[^200]: 在多轮布局创建中测量人-AI团队与AI-AI团队的差异

    Measuring Human-AI and AI-AI Teams in Multi-turn Layout Creation

    [https://arxiv.org/abs/2611.28025](https://arxiv.org/abs/2611.28025)

    该研究通过基于人类偏好数据拟合的几何特征评分函数，比较了人-AI团队与AI-AI团队在多轮布局创建任务中的表现，发现大语言模型倾向于在首轮对话中编码完整布局，而人类则更倾向于多轮迭代完善。

    

    共同创建网页和幻灯片等布局是评估大语言模型系统多轮指令遵循能力的有效测试平台。由于预期的布局外观无法在单轮对话中充分表达，用户必须等待大语言模型生成布局，然后在多轮对话中对其进行迭代式完善。我们对这一多轮过程进行了一项小规模探索性研究（n=6），以比较人-AI团队与AI-AI团队在同一布局生成任务上的差异。给定一个参考线框图布局，由参与者或语言模型充当"说话者"，通过多轮对话向固定的"倾听者"大语言模型发出自然语言指令，直到说话者满意为止。我们没有采用大语言模型作为评判者，而是利用精心整理的人类偏好数据（n=19）拟合了一个基于几何布局特征的评分函数。我们发现，大语言模型倾向于在第一轮对话中就编码整个布局，而人类平均会通过多轮对话进行迭代完善。

    arXiv:2611.28025v1 Announce Type: cross  Abstract: Co-creating layouts such as websites and slides is a useful testbed for evaluating multi-turn instruction following of LLM systems. Since an intended layout's appearance cannot be well expressed in a single turn, the user must wait for the LLM to generate a layout then iteratively refine it with the LLM over multiple turns. We present a small-scale, exploratory (n=6) study of this multi-turn process to compare the differences of human-AI and AI-AI teams on the same layout-generation task. Given a reference wire-frame layout, a participant or a language model 'speaker' issues natural-language instructions to a fixed 'listener' LLM over multiple turns until the speaker is satisfied. Rather than an LLM as the judge, we fit a scoring function over geometric layout features using curated human preference data (n=19). We find that LLMs tend to encode the entire layout in the first turn, while humans iterate over many turns on average, writin
    
[^201]: 模拟用户会像真实用户一样提出反驳吗？正确性对比立场

    Do Simulated Users Push Back Like Real Users? Correctness vs. Stance

    [https://arxiv.org/abs/2611.28024](https://arxiv.org/abs/2611.28024)

    研究发现LLM模拟用户的反驳行为与真实用户存在系统性差异：真实用户更多是因为自身立场受到质疑而反驳，而模拟用户是否反驳取决于其对助手观点正确性的判断，即模拟器关注“对错”而非“立场”。

    

    LLM模拟用户在多轮评估中代替真人，其前提假设是它们会对助手的发言作出反应。我们针对“反驳”行为对这一假设进行了检验。我们从WildChat中选取助手明确质疑或认同用户所述立场的对话，并将真实的下一轮用户回复与相同前缀下模拟出的回复进行比较。在作出回复的用户中，真实用户在被质疑后的反驳比例高于被认同后的比例（20.5% vs. 8.2%）；而基于GPT的模拟器则并非如此（27.1% vs. 27.1%）：立场差距 D = -0.12 [-0.17, -0.08]。在被质疑后，模拟器往往转而重述或重新解释自己的请求（“我的意思是……”）；这类回复中约有一半属于变相的坚持，这使得D值介于-0.08与-0.05之间。驱动模拟器争辩的因素是助手的立场是否被判定为正确（被质疑后：21% vs. 49%；被认同后：16% vs. 58%），而真实用户的变化不大（19% vs. 24%; …）

    arXiv:2611.28024v1 Announce Type: cross  Abstract: LLM-simulated users stand in for people in multi-turn evaluations, on the assumption that they react to what the assistant says. We test this for pushback. From WildChat, we take conversations in which the assistant explicitly challenges or endorses a position the user stated, and compare the real next user turn with simulated turns from the same prefix. Among users who replied, real users push back more after a challenge than after an endorsement (20.5% vs. 8.2%); a GPT-based simulator does not (27.1% vs. 27.1%): stance gap D = -0.12 [-0.17, -0.08]. After a challenge the simulator instead often re-states or re-interprets its request ("I meant ..."); about half of these turns are disguised insistence, which puts D between -0.08 and -0.05. What drives the simulator's disputing is whether the assistant's position is judged correct (21% vs. 49% after challenges; 16% vs. 58% after endorsements), while real users change little (19% vs. 24%;
    
[^202]: MemVanta：面向量化GGUF语言模型的内存高效CPU推理

    MemVanta: Memory-Efficient CPU Inference for Quantized GGUF Language Models

    [https://arxiv.org/abs/2611.27997](https://arxiv.org/abs/2611.27997)

    MemVanta是一个面向量化GGUF语言模型的内存高效CPU推理运行时，通过内存映射访问、有界张量缓存和分页键值状态等技术，将峰值内存占用相比llama.cpp降低47.50%，但以显著的吞吐量下降为代价。

    

    arXiv:2611.27997v1 公告类型：交叉 摘要：在商用CPU上部署大语言模型，通常不仅受计算吞吐量的限制，还受在固定系统预算内执行量化模型所需常驻内存的制约。我们提出了MemVanta，一个面向量化GGUF语言模型的实验性C++20推理运行时，它将内存驻留视为一等系统目标。MemVanta结合了内存映射模型访问、有界张量缓存、量化执行路径、分页键值状态以及显式的内存压力验证。我们在仅CPU执行条件下，使用相同的OpenLLaMA 7B v2 Q4_0 GGUF模型，将MemVanta与固定的llama.cpp基线进行对比评估。在预热后的五次测量运行中，MemVanta记录的峰值常驻内存集（RSS）为3,983,412 KiB，而llama.cpp为7,586,960 KiB，对应峰值RSS降低了47.50%。这一内存降低伴随着显著的吞吐量权衡：MemVanta达到3.15 +/-（原文在此处截断）

    arXiv:2611.27997v1 Announce Type: cross  Abstract: Deploying large language models on commodity CPUs is often constrained not only by computational throughput but also by the resident memory required to execute quantized models within fixed system budgets. We present MemVanta, an experimental C++20 inference runtime for quantized GGUF language models that treats memory residency as a first-class systems objective. MemVanta combines memory-mapped model access, bounded tensor caching, quantized execution paths, paged key-value state, and explicit memory-pressure validation. We evaluate MemVanta against a pinned llama.cpp baseline using the same OpenLLaMA 7B v2 Q4_0 GGUF under CPU-only execution. Across five measured runs following warm-up, MemVanta records a peak resident set size of 3,983,412 KiB compared with 7,586,960 KiB for llama.cpp, corresponding to a 47.50 percent reduction in peak RSS. This memory reduction involves a substantial throughput trade-off: MemVanta achieves 3.15 +/- 
    
[^203]: Hy-MT3：更准确、更鲁棒且更实用的真实场景多语言翻译模型

    Hy-MT3: More Accurate, Robust and Practical Multilingual Translation Models in the Wild

    [https://arxiv.org/abs/2611.27673](https://arxiv.org/abs/2611.27673)

    Hy-MT3 系列通过将误译率降低 43%、漏译增译降低 74%、指令遵循错误降低 40%，并把语言覆盖扩展至 102 种，其仅 3B 激活参数的 MoE 模型在非思考模式下超越了众多旗舰大模型并逼近最强闭源思考模型。

    

    Hy-MT3 是混元多语言翻译模型家族的第三代，包括 Hy-MT3-1.8B、Hy-MT3-7B 和 Hy-MT3-30B-A3B（MoE 架构）。与 Hy-MT2 相比，Hy-MT3 的翻译更准确、更完整、更忠实于用户约束：在真实场景数据上，误译样本比例降低了 43%，漏译或增译问题降低了 74%，指令遵循错误降低了 40%，同时语言覆盖范围从 38 种语言扩展到 102 种。仅需 3B 激活参数，Hy-MT3-30B-A3B 便在非思考模式下取得了所有评测模型中的最佳整体性能，超越了 Qwen3.8-Max、Kimi-K3.1-Pro 和 DeepSeek-V4.1-Flash 等旗舰模型，并接近 GPT-6.1-Sol 和 Gemini-4-Argon 等最强的闭源思考模型，同时超越了消耗超过 30 倍资源的模型。

    arXiv:2611.27673v1 Announce Type: cross  Abstract: Hy-MT3 is the third generation of the Hunyuan multilingual translation model family, including \textbf{Hy-MT3-1.8B}, \textbf{Hy-MT3-7B} and \textbf{Hy-MT3-30B-A3B (MoE)}. Compared with Hy-MT2, Hy-MT3 translates \textbf{more accurately, more completely and more faithfully to user constraints}: on real-world scenario data, the share of samples with mistranslations is reduced by \textbf{43\%}, omissions or additions by \textbf{74\%}, and instruction-following errors by \textbf{40\%}, while language coverage is expanded from \textbf{38 to 102} languages. With only 3B activated parameters, \textbf{Hy-MT3-30B-A3B achieves the best overall performance among all evaluated models in non-thinking mode}, including flagship models such as Qwen3.8-Max, Kimi-K3.1-Pro and DeepSeek-V4.1-Flash, and approaches the strongest closed-source thinking models such as GPT-6.1-Sol and Gemini-4-Argon, while surpassing models that consume \textbf{more than 30$\ti
    
[^204]: PaMuk-Bench：一个面向视觉-语言任务的平行多语言多任务基准

    PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks

    [https://arxiv.org/abs/2611.27521](https://arxiv.org/abs/2611.27521)

    本文提出了PaMuk-Bench，一个包含超过10K条平行翻译提示（涵盖阿拉伯语、丹麦语、法语、印地语、意大利语和罗马尼亚语六种语言）的多语言多任务视觉-语言基准，用于系统评估视觉-语言模型在低资源语言上跨多种任务的表现。

    

    尽管视觉-语言模型（VLM）目前在英语等被广泛研究的语言上取得了显著成果，但由于缺乏标注数据集，它们在某些低资源语言上表现出显著的性能下降。为此，我们提出了PaMuk-Bench，一个新颖的面向视觉-语言任务的平行多语言多任务基准。我们的基准包含一组超过10K条精选的英文提示，这些提示被翻译成六种语言：阿拉伯语、丹麦语、法语、印地语、意大利语、罗马尼亚语。通过半自动过程获得的平行翻译，以及配套的生成和检索图像，使得能够在广泛的任务上对VLM进行对比性开发和评估，这些任务包括：文本到图像生成、图像描述生成、基于内容的图像检索、图像生成难度估计、图像检索难度估计。我们通过实验证明，零样本英语和多语言预训练……

    arXiv:2611.27521v1 Announce Type: cross  Abstract: While vision-language models (VLMs) currently obtain notable results for highly-studied languages, e.g. English, they exhibit significant performance drops in certain low-resource languages, especially due to the lack of annotated datasets. To this end, we propose PaMuk-Bench, a novel parallel multilingual multitask benchmark for vision-language tasks. Our benchmark includes a set of over 10K curated English prompts that are translated into six languages: Arabic, Danish, French, Hindi, Italian, Romanian. Parallel translations, which are obtained via a semi-automatic process, along with accompanying generated and retrieved images enable the comparative development and evaluation of VLMs across a wide range of tasks: text-to-image generation, image captioning, content-based image retrieval, image generation difficulty estimation, image retrieval difficulty estimation. We empirically show that zero-shot English and multilingual pre-traine
    
[^205]: Typhoon OCR 2：从泰文文档转录到多语言、多任务 OCR

    Typhoon OCR 2: From Thai Document Transcription to Multilingual, Multi-Task OCR

    [https://arxiv.org/abs/2611.27494](https://arxiv.org/abs/2611.27494)

    Typhoon OCR 2 将前代专注泰文转录的模型扩展为单一的多语言多任务 OCR 系统，通过 GRPO 训练的紧凑 2B/0.8B 模型同时支持提示驱动的信息提取与泰文、英文、中文的整页结构化解析，并在泰文基准上大幅超越前代且保持英文性能。

    

    Typhoon OCR 1.5 专注于阅读泰文文档。然而，在真实的工作流程中，用户通常需要提取特定信息或保留整页结构以便进一步处理。Typhoon OCR 2 在单一模型中同时处理这两项任务，支持提示驱动的信息提取和结构化的整页解析。整页解析覆盖泰文、英文和中文。我们发布了紧凑的 2B 和 0.8B 模型，这些模型基于开源文档解析器，在精心整理的泰文数据和公开多语言数据上进行监督微调，随后采用带任务奖励和未完成响应惩罚规则的群体相对策略优化（GRPO）。在所评估的开源模型中，Typhoon OCR 2 2B 在泰文 MDPBench 上得分最高（80.7，而 Typhoon OCR 1.5 为 68.9），在涵盖识别、提取和文档理解的 ThaiOCRBench 上也得分最高（59.91 对 40.45）。该 2B 模型在英文 olmOCR-bench 上仍保持接近其基础模型的水平（84.7 对 84.4）……

    arXiv:2611.27494v1 Announce Type: new  Abstract: Typhoon OCR 1.5 focused on reading Thai documents. However, in real-world workflows, users often need to extract specific information or preserve the structure of an entire page for further processing. Typhoon OCR 2 handles both tasks in a single model, with prompt-driven extraction and structured full-page parsing. Full-page parsing covers Thai, English, and Chinese. We release compact 2B and 0.8B models adapted from open document parsers through supervised fine-tuning on curated Thai and public multilingual data, followed by Group Relative Policy Optimization (GRPO) with task rewards and a rule penalizing unfinished responses. Among evaluated open models, Typhoon OCR 2 2B scores highest on Thai MDPBench (80.7 versus 68.9 for Typhoon OCR 1.5) and on ThaiOCRBench (59.91 versus 40.45), which covers recognition, extraction, and document understanding. The 2B model remains close to its base on English olmOCR-bench (84.7 versus 84.4) and on 
    
[^206]: 权重空间编程可写入上下文学习并在语言模型中媲美微调

    Weight-Space Programming Writes In-Context Learning and Matches Fine-Tuning in Language Models

    [https://arxiv.org/abs/2611.26841](https://arxiv.org/abs/2611.26841)

    前沿语言模型可通过权重空间编程将ICL任务向量直接写入较弱模型的稀疏MLP单元（与提示格式相关联），使其零样本性能达到ICL和微调（含LoRA）的准确率，且几乎不影响困惑度。

    

    权重空间编程是前沿语言模型最近被报道的一种能力：它无需梯度下降即可读取并改写另一个网络的权重。我们研究示例如何能以权重更新的形式提供信息，使较弱的语言模型在提示为空的情况下完成任务，并探究编辑模型究竟改写了什么。实验涵盖六项任务、四个前沿编辑模型以及三族较弱的语言模型，且没有向编辑模型透露任何关于上下文学习（ICL）的信息。在大多数运行中，编辑模型在最后一个提示 token 处测量少样本减零样本的残差，即 ICL 任务向量，并将其写入权重之中，最稀疏的情况下会写入与提示格式相关联的近乎沉默的 MLP 单元。所得结果达到了 ICL 以及微调（包括 LoRA）在相同示例上的准确率，通常几乎不改变困惑度，但在措辞发生变化时性能会有所下降。我们还通过统计未完成的运行来限定这一比较，观察到……

    arXiv:2611.26841v1 Announce Type: new  Abstract: Weight-space programming is a frontier language model's recently reported ability to read and rewrite another network's weights without gradient descent. We investigate how examples can provide a weight update so that a weaker language model performs the task with nothing in the prompt, and study what it edits. Our experiments cover six tasks, four frontier editors and three families of weaker language models, without the editors being told anything about in-context learning (ICL). In most runs, the few-shot minus zero-shot residual is measured at the final prompt token, the ICL task vector, and write it into the weights, at its sparsest into near-silent MLP units keyed to the prompt format. Results reach the accuracy of ICL and of fine-tuning on the same examples, LoRA included, usually leaving perplexity untouched, with worse performance under changes of wording. We also qualify the comparison by counting unfinished runs, observing tha
    
[^207]: 面向语言模型前馈层的非扩展门控连分数架构

    Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers in Language Models

    [https://arxiv.org/abs/2611.26835](https://arxiv.org/abs/2611.26835)

    本文提出一种非扩展门控连分数架构组件，可替换语言模型中的各类前馈层（包括MLP、GLU及MoE专家），并用对数和指数重新实现除法运算，从而提升性能并简化与标准预训练流程的集成。

    

    arXiv:2611.26835v1 公告类型：交叉 摘要：最近的研究表明，受连分数启发的神经架构在语言建模中可以做到高度参数高效且性能优异。在本工作中，我们重新审视了基于连分数的通道混合方法，旨在提升其性能，并简化其与语言模型标准预训练协议的集成。我们通过两个简单而关键的贡献来实现这一目标：i) 我们提出了一种新颖的非扩展门控连分数启发的架构组件，用于替换现有语言模型架构中的前馈层，这些架构可以采用普通多层感知机、门控线性单元，或混合专家架构中的（普通/门控）专家。ii) 我们提出了一种用对数和指数重新实现除法运算的方法——除法正是这类架构中的非线性来源。通过在 GPT-2、Llama-3、Mamba- 等不同语言建模架构上进行实验……

    arXiv:2611.26835v1 Announce Type: cross  Abstract: Recent work has shown that continued fraction inspired neural architectures (CoFrGeNets) can be highly parameter efficient yet performant for language modeling. In this work, we revisit continued fraction-based channel mixing with the goal of improving its performance and simplifying integration into standard pre-training protocols for language models. We achieve this by two simple yet critical contributions: i) we propose a novel non-expanding gated continued fraction inspired architectural component to replace feedforward layers in established language model architectures that may use vanilla multilayer perceptrons or gated linear units or (vanilla/gated) experts in mixture-of-experts (MoE) type architectures. ii) We propose a way to re-implement divisions -- the non-linearity in these architectures -- in terms of logarithms and exponentials. By experimenting on different language modeling architectures such as GPT-2, Llama-3, Mamba-
    
[^208]: GLiDE：从推理教师模型蒸馏得到的快速校准决策模型

    GLiDE: A Fast, Calibrated Decision Model Distilled from Reasoning Teachers

    [https://arxiv.org/abs/2611.26509](https://arxiv.org/abs/2611.26509)

    GLiDE 通过多教师软目标蒸馏、双向注意力机制和无参数循环深度，将推理教师模型的能力迁移到单次前向传播即可完成的决策模型中，在无需生成推理文本的情况下保留了推理带来的决策质量提升。

    

    决策模型通过在单次前向传播中对候选答案进行评分，实现快速且具备置信度感知的决策，而无需生成文本或中间推理过程。然而，推理能够显著提升决策质量，这引发了一个问题：能否在不付出生成推理的代价的情况下保留推理带来的收益。我们通过 GLiDE 来研究这一问题，GLiDE 是一系列参数规模从 0.4B 到 27B 的决策模型。GLiDE 基于 Qwen 混合语言模型构建，通过移除全注意力层中的因果掩码，实现了上下文、问题和候选答案之间的双向交互。为了将推理能力迁移到单次推理中，我们将来自多个教师模型（包括一个推理模型）的软目标蒸馏与在多样化决策任务上的渐进式后训练相结合。我们还进一步探索了无参数的循环深度，以在不生成 token 的情况下增加推理时的计算量。

    arXiv:2611.26509v1 Announce Type: new  Abstract: Decision models enable fast, confidence-aware decisions by scoring candidate answers in a single forward pass, without generating text or intermediate reasoning. Yet reasoning can substantially improve decision quality, raising the question of whether its benefits can be retained without the cost of generating it. We investigate this question with \textbf{GLiDE}, a family of decision models spanning 0.4B to 27B parameters.   Building on Qwen hybrid language models, GLiDE enables bidirectional interactions between the context, question, and candidate answers by removing causal masking from full-attention layers. To transfer reasoning capabilities into single-pass inference, we combine soft-target distillation from multiple teachers, including a reasoning model, with progressive post-training on diverse decision tasks. We further explore parameter-free recurrent depth to increase inference-time computation without generating tokens.   Acro
    
[^209]: 前沿模型仅需一段文本即可揭露其人类或AI作者身份

    Frontier models need only one text to unmask its human or AI author

    [https://arxiv.org/abs/2611.26216](https://arxiv.org/abs/2611.26216)

    前沿大模型仅凭一段文本和简单的零样本提示，就能判断作者是AI还是人类、识别AI文本来自哪家实验室、甚至准确指认人类作者的身份，其性能媲美专用检测器并能给出自然语言解释。

    

    仅需给定一段文本和一个简单的零样本提示（无需网络访问、工具、候选列表或写作样本），前沿模型就能在三个层层递进的层面上回答“这是谁写的？”：(1) 是AI还是人类写的；(2) 如果是AI，来自哪个实验室；(3) 如果是人类，往往能准确说出具体是谁。最新的模型（Claude Fable 5.2、GPT-6.1 Sol、Gemini 4 Carbon）在使用自然语言解释其预测的同时，性能可与Pangram等经过专门训练的检测器相媲美；Opus 5.5在98%的情况下能正确说出Anthropic、OpenAI和Google模型所生成文本的来源实验室。最引人注目的是，Fable 5.2能正确识别2026年10-11月发表的39%公开同行评审的作者姓名（在来自Nature的一个子集上达到78%），并能识别出活跃Reddit用户（过去一年发帖超过500条）69%评论背后的用户名。准确率取决于预训练数据中对作者写作内容的接触程度以及推理投入程度，而仅仅询问一个mo...（原文摘要在此处截断）

    arXiv:2611.26216v1 Announce Type: new  Abstract: Given a single text and a simple zero-shot prompt (no web access, tools, candidate lists, or writing samples), frontier models can answer "who wrote this?" at three increasingly specific levels: (1) whether an AI or human wrote it, (2) if an AI, which lab, and (3) if a human, often exactly who. The latest models (Claude Fable 5.2, GPT-6.1 Sol, Gemini 4 Carbon) match trained detectors like Pangram while also explaining their predictions in natural language, and Opus 5.5 names the correct lab for 98% of texts generated by models from Anthropic, OpenAI, and Google. Most strikingly, Fable 5.2 correctly names the author of 39% of public peer reviews published in Oct-Nov 2026 (78% on a subset of reviews from Nature), and identifies the username behind 69% of comments from active Redditors (with >500 posts over the last year). Accuracy depends on exposure to an author's writing in pretraining data and on reasoning effort, and simply asking a mo
    
[^210]: 语言模型对重音信息的表征多于它们所能表达的

    Language models represent more about emphasis than they can tell you

    [https://arxiv.org/abs/2611.25808](https://arxiv.org/abs/2611.25808)

    该研究发现语言模型内部编码了文本所能决定的重音突显信息，但通过提示无法将其表达出来，而完整的重音信息必须依赖语音信号才能获得。

    

    韵律突显在很大程度上可以通过句子的词语来预测，但其中一部分取决于说话人的朗读方式。我们研究了语言模型能从文本中恢复多少突显信息，在被询问时能否将其报告出来，以及语音信号能带来多少额外信息。在标注了词级突显的英语朗读语音上，我们在相同的词上比较了冻结和微调的文本模型、经提示的大语言模型（LLM）以及语音模型。所有文本模型的最终性能彼此相差约一个百分点以内，在三类任务上最好达到68.8%，并且它们在很大程度上犯同样的错误。在我们测试的提示方式下，即使给出500个示例，LLM的表现也不优于“词多数”基线。而线性探针则在其内部表示中发现了更多信息。语音模型达到84.7%。我们认为，文本只决定了突显的一部分，其余部分必须通过听觉才能获得。LLM编码了文本所能决定的内容，但提示并不能将其引出：它们对重音位置的表征……（摘要原文在此处截断）

    arXiv:2611.25808v1 Announce Type: new  Abstract: Prosodic prominence can largely be predicted from the words of a sentence, but part of it depends on how the speaker reads it. We study how much of prominence language models recover from text, whether they can report it when asked, and what the speech signal adds. On read English speech labelled for word-level prominence, we compare frozen and fine-tuned text models, prompted large language models (LLMs) and speech models on the same words. All text models end within about one point of each other, at best 68.8% on the 3-way task, and largely make the same errors. With the prompts we tested, LLMs do no better than a word-majority baseline, even with 500 examples. Linear probes find more in their representations. Speech models reach 84.7%. We argue that text determines only part of prominence, and that the rest has to be heard. LLMs encode what the text determines, but prompting does not bring it out: they represent more about where the e
    
[^211]: 通过审计式认知探索引导自动化科学发现

    Steering Automated Scientific Discovery via Audited Epistemic Exploration

    [https://arxiv.org/abs/2611.25214](https://arxiv.org/abs/2611.25214)

    该论文提出即插即用的认知脚手架 EpiCompass，通过维护连接假设、证据、信念与修订的认知树，依据积累证据审计信念、利用未解决冲突和未探索方向生成假设，并以预期认知增益选择候选假设，从而解决自动科研智能体信念缺乏证据支持、假设空间狭窄和忽视矛盾证据的问题。

    

    科学发现需要根据新证据重新审视假设，并识别已探索范围之外有前景的方向。然而，研究智能体常常做出其观察结果无法支持的主张，探索狭窄的假设空间，并且在意识到证据相互冲突时仍然保留原有信念。这些失败促使人们寻找一种显式的认知状态，以同时捕捉关于现有假设的信念和进一步探索的方向。为此，我们提出了 Epistemic Compass（EpiCompass），这是一种即插即用的脚手架，通过一棵连接假设、证据、信念与修订的认知树来维护这一状态。随着调查的推进，EpiCompass 会根据积累的证据重新评估已探索的假设，并利用未解决的冲突和未探索的方向来引导假设生成。它通过预期认知增益来选择候选假设，同时兼顾……（原文此处截断）

    arXiv:2611.25214v1 Announce Type: new  Abstract: Scientific discovery requires revisiting the hypotheses in light of new evidence and identifying promising directions beyond those already explored. However, research agents often make claims that are not supported by their observations, explore a narrow hypothesis space, and retain beliefs despite recognizing conflicting evidence. These failures motivate the search for an explicit epistemic state that captures both beliefs about existing hypotheses and directions for further exploration. To this end, we propose Epistemic Compass (EpiCompass), a plug-and-play scaffold that maintains this state through an epistemic tree linking hypotheses, evidence, beliefs, and revisions. As an investigation progresses, EpiCompass reassesses the hypotheses explored against accumulated evidence and uses unresolved conflicts and unexplored directions to guide hypothesis generation. It selects candidates by anticipated epistemic gain, accounting for both re
    
[^212]: 更努力思考并不能带来更好的选择：推理努力提升了LLM购物代理的置信度，而非正确性

    Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents

    [https://arxiv.org/abs/2611.22033](https://arxiv.org/abs/2611.22033)

    在三个购物基准测试中，增加LLM购物代理的推理努力并不能显著提升其选购产品的正确性，却会显著提高其报告的置信度——也就是说，模型“想得更多”只是变得更自信，而不是变得更正确。

    

    大型语言模型（LLM）越来越多地被部署为自主购物代理，它们处理多模态商品信息（包括视觉和文本数据）并代表消费者推荐购买。人们自然会认为，思考能力更强的模型会做出更高质量的购买决策。为了验证这一观点，我们针对此类代理提出了两个问题：要求模型更努力地思考，是否会让它选出更好的产品？它所报告的置信度，能否告诉我们它出错的概率？我们在基于WebShop、亚马逊SQID和Wayfair的WANDS平台构建的三个产品选择基准上，对十个前沿模型进行了测试，以检验思考能力更强的LLM是否真的能做出更高质量的产品购买决策。在全部三个基准上，我们发现增加思考努力并不能在统计上显著提升决策质量。而同样的思考努力变化却会提高模型所声称的置信度（原文摘要在此处截断）。

    arXiv:2611.22033v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as autonomous shopping agents that process multimodal product listings (consisting of visual and text data) and recommend purchases on behalf of consumers. Naturally, one might assume that models with more thinking power would make higher-quality purchasing decisions. To test this belief, we ask two questions about such agents: does asking a model to think harder make it choose better products, and does the confidence it reports tell us the likelihood it may be wrong? We benchmark ten frontier models on three product-choice benchmarks built from WebShop, Amazon's SQID and Wayfair's WANDS platforms to see if LLMs with more thinking power actually make better-quality product purchase decisions. Across all three benchmarks, we found that increasing the thinking effort does not increase decision quality in a statistically meaningful way. The same change in thinking effort raises stated c
    
[^213]: SkillBuffBench：衡量智能体技能在框架、模型、技能访问模式与记忆等方面的贡献

    SkillBuffBench: Measuring What Agent Skills Contribute across Harnesses, Models, Skill Access Modes and Memory

    [https://arxiv.org/abs/2611.22017](https://arxiv.org/abs/2611.22017)

    SkillBuffBench 通过将技能访问方式（无技能/精选技能/自动调用）、记忆开关、四个智能体框架和多个主流模型进行交叉配对评估，系统性地衡量了智能体技能在贴近真实部署场景下的实际贡献。

    

    智能体技能是一类基于文件的程序性知识包，编码智能体可以按需发现并加载，目前各大主流智能体框架均已支持。然而，关于技能能带来多大帮助的证据，几乎完全来自这样的设定：为任务人工挑选正确的技能，且智能体从空白状态开始。这两点在真实部署中都不成立——部署时智能体需要从已安装的技能库中自行选择，并且经常携带此前工作中积累的记忆。我们提出了 SkillBuffBench，这是一个基准测试及配套的配对评估协议，它将技能访问视为一个三级因素（无技能、精选技能、以及从已安装库中自动调用），并与记忆（开启与关闭）、四个智能体框架以及多个模型（来自 Anthropic、Google 和 OpenAI 的最先进专有模型，以及来自中国厂商的最先进开源权重模型）进行交叉组合，每个模型都在每个框架下运行……

    arXiv:2611.22017v1 Announce Type: new  Abstract: Agent Skills, file-based packages of procedural knowledge that a coding agent can discover and load on demand, are now shipped by every major agent harness, yet the evidence for how much they help comes almost entirely from settings in which the right Skill is hand-picked for the task and the agent starts from a blank slate. Neither holds in deployment, where an agent chooses from whatever library happens to be installed and frequently carries memory from earlier work. We present SkillBuffBench, a benchmark and paired evaluation protocol that treats Skill access as a three-level factor (no Skill, a curated Skill, and automatic invocation from an installed library), crosses it with memory (off and on), with four agent harnesses and with multiple models (the state-of-the-art proprietary models from Anthropic, Google and OpenAI, and the state-of-the-art open-weight models from Chinese providers), every model being run under every harness, a
    
[^214]: TUSI：一个基于引用溯源的图结构基准测试

    TUSI: A Citation-Grounded Graph-Based Benchmark

    [https://arxiv.org/abs/2611.21823](https://arxiv.org/abs/2611.21823)

    该论文提出了TUSI-Bench——一个建立在包含7,261条法律条文的多层知识框架和法律知识图谱之上的结构化波斯语法律基准，用于对具有引用溯源能力的法律RAG系统进行严格评估。

    

    对波斯语（伊朗）法律领域的检索增强生成（RAG）系统进行评估仍然具有挑战性，因为现有基准缺乏真实性、引用溯源性以及对法律推理的覆盖。现有资源要么是从成文法文本中人工合成的，缺乏自然的信息需求；要么来源于非正式的法律问答资源，与权威法律文件之间没有明确的关联，因此不适合对有据可依的法律推理进行严格评估。为弥补这一空白，我们提出了TUSI-Bench，一个专为评估基于RAG的系统而设计的结构化波斯语法律基准。TUSI-Bench建立在一个多层法律知识框架之上，该框架包含7,261条法律条文，并整合了成文法典、法律评注、学理著作以及最高法院统一裁定。我们还构建了一个连接法条与法律概念的知识图谱，以支持结构感知的检索。基于这一基础，TUSI

    arXiv:2611.21823v1 Announce Type: cross  Abstract: Evaluating retrieval-augmented generation (RAG) systems in Persian law remains challenging because benchmarks lack realism, grounding, and coverage of legal reasoning. Existing resources are either synthetically generated from statutory texts, lacking natural information needs, or derived from informal legal question--answer sources without explicit linkage to authoritative legal documents, making them unsuitable for rigorous evaluation of grounded legal reasoning. To address this gap, we introduce TUSI-Bench, a structured Persian legal benchmark designed for the evaluation of RAG-based systems. TUSI-Bench is built upon a multi-layered legal knowledge framework containing 7,261 legal articles and integrating codified statutes, legal commentaries, doctrinal writings, and Supreme Court unification rulings. We also construct a legal knowledge graph linking articles and concepts for structure-aware retrieval. Based on this foundation, TUSI
    
[^215]: 看见却未言说：定位与修复视觉-语言模型中的跨图像比较

    Seen but Unsaid: Locating and Repairing Cross-Image Comparison in Vision-Language Models

    [https://arxiv.org/abs/2611.21539](https://arxiv.org/abs/2611.21539)

    该论文发现视觉-语言模型在跨图像比较任务上的失败是因为比较信息存在于视觉token和隐藏状态中却未传递到最终答案，通过定位一小部分承载比较功能的注意力头，提出了仅修改0.02%参数的修复方法。

    

    视觉-语言模型（VLMs）在跨两张图像匹配一个点或一个物体时常常失败，即使答案可以从它们自身的隐藏状态中解码出来。我们在Qwen3-VL（2B-32B）、InternVL3.5-8B和Gemma-3-12B上，针对合成形状匹配、同实例点对应（HPatches）和语义对应（SPair-71k）任务，从视觉token追踪到答案字母，发现了两个不同的失败点、一小部分承载比较功能的注意力头，以及一个仅涉及0.02%参数的修复方法。两个失败点：精确副本被比较但未被读取出来——在所有模型家族中，答案位置探针都比校准后的输出高出15到25个百分点，无论形状是否可命名都是如此，且沿探针方向引导所带来的答案变化不比随机方向多；语义对应存在于视觉token中但从未到达答案。一条通路：重新分配cro…（摘要截断）

    arXiv:2611.21539v1 Announce Type: cross  Abstract: Vision-language models (VLMs) often fail to match a point or an object across two images, even when the answer can be decoded from their own hidden states. We trace this failure from the visual tokens to the answer letter, in Qwen3-VL (2B-32B), InternVL3.5-8B, and Gemma-3-12B, on synthetic shape matching, same-instance point correspondence (HPatches), and semantic correspondence (SPair-71k), and find two distinct failure points, a small set of attention heads that carries the comparison, and a repair that touches 0.02% of the parameters. Two failure points: exact copies are compared but not read out. An answer-position probe exceeds the calibrated output by 15 to 25 points in every family, for nameable and unnameable shapes alike, and steering along the probe's direction moves no more answers than random directions. Semantic correspondences are present in the visual tokens but never reach the answer. One pathway: redistributing the cro
    
[^216]: LLM推理中对话历史的隐性代价

    The Silent Cost of Conversation History in LLM Reasoning

    [https://arxiv.org/abs/2611.20071](https://arxiv.org/abs/2611.20071)

    研究发现累积的对话历史会显著降低LLM在多轮任务中的推理准确率，且过程级质量指标无法提前预警这种下降，而用经求解器验证的状态摘要替代完整历史可挽回大部分性能损失。

    

    在多轮信息系统中，大语言模型（LLM）会不断累积自身早期的回复以及持续增长的任务状态。我们探究了这种对话历史是否会降低准确率，以及过程级质量指标能否在准确率下降之前发出预警。每项分析均在计算结果之前预先设定。在研究1中，针对48个分片数学任务，保留模型自身回复使GPT-4o-mini的最终成功率降低0.21，Mistral-7B降低0.29（经Holm校正的p值分别为0.026和0.004）；Llama-4 Scout呈现相同趋势，但不显著。研究2引入了CDR-Stream基准，包含100轮状态追踪对话，每轮均配有精确答案并针对各模型进行了难度校准。在相同难度下，使用完整历史时，准确率在二十轮内从约0.6降至0.1；而使用经求解器验证的状态摘要，准确率可保持高出0.52至0.59（经Holm校正的p值小于0.001）。一致性与逻辑连贯性的下降幅度远小于准确率，而稳定性反而有所上升。

    arXiv:2611.20071v1 Announce Type: cross  Abstract: Large language models (LLMs) in multi-turn information systems accumulate their own earlier replies and a growing task state. We asked whether this conversation history lowers accuracy, and whether process-level quality metrics warn before it does. Each analysis was fixed before its outcomes were computed. In Study 1, on 48 sharded math tasks, retaining its replies lowered final success by 0.21 for GPT-4o-mini and 0.29 for Mistral-7B (Holm-adjusted p = 0.026 and 0.004); Llama-4 Scout showed the same direction, not significant. Study 2 introduces CDR-Stream, 100-turn state-tracking dialogues with an exact answer per turn and per-model difficulty calibration. With the full history, accuracy fell from about 0.6 to 0.1 within twenty turns at matched difficulty, whereas a solver-verified state summary kept it 0.52-0.59 higher (Holm-adjusted p less than 0.001). Consistency and logical coherence fell far less than accuracy, and stability rose
    
[^217]: 自适应大语言模型红队测试中攻击者多样性的降低

    Attacker Diversity Reduction in Adaptive LLM Red Teaming

    [https://arxiv.org/abs/2611.19680](https://arxiv.org/abs/2611.19680)

    该论文使用Qwen2.5-14B攻击者与DPO训练的Phi-4-mini防御者进行GRPO协同训练实验，复现了自适应LLM红队测试中攻击者多样性降低的现象，并提出了攻击多样性保留率（ADR）作为衡量指标。

    

    强化学习可以在提高自动化红队测试成功率的同时缩小其产生的攻击范围。已有报道称，使用组相对策略优化（GRPO）对攻击者和防御者进行协同训练会降低攻击者的多样性。本实验通过三次运行复现了攻击多样性降低的现象，其中攻击者采用Qwen2.5-14B模型，防御者采用通过直接偏好优化（DPO）训练的Phi-4-mini模型。攻击者仅接收基于成功的奖励，并使用长度无偏损失、有界更新、预热和比率裁剪。攻击者在26个AdvBench行为上进行评估。基于开发检查点的决策，训练在12轮后结束，因此所报告的区间为名义区间。攻击多样性保留率（ADR）通过比较训练后攻击者与未训练攻击者在单个行为内的攻击分布广度来衡量多样性变化。为认定攻击多样性已被降低，评判标准要求ADR低于某一阈值。

    arXiv:2611.19680v1 Announce Type: cross  Abstract: Reinforcement learning can make automated red-teaming more successful while narrowing the attacks it produces. Co-training an attacker and a defender with group-relative policy optimization (GRPO) has been reported to reduce attacker diversity. This experiment reproduces attack diversity reduction using three runs with a Qwen2.5-14B attacker and a Phi-4-mini defender trained by direct preference optimization (DPO). The attacker receives a success-only reward and uses a length-unbiased loss, bounded updates, warm-up and ratio clipping. The attacker is evaluated on 26 AdvBench behaviors. Training ended after 12 rounds, following a decision based on development checkpoints, so the reported intervals are nominal. Attack Diversity Retention (ADR) compares the within-behavior of the trained attacker spread with the untrained attacker. In order for an attack diversity to be consider as having been reduced, a criterion requires the ADR to be b
    
[^218]: 更多智能体、更多样本，还是更多思考？解耦多智能体大语言模型中的协调、集成与测试时计算

    More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs

    [https://arxiv.org/abs/2611.17610](https://arxiv.org/abs/2611.17610)

    本研究通过综合516个多智能体配置的元分析发现，多智能体大语言模型的表现优势主要来自额外的样本数量和测试时计算量，而非智能体间的协调机制本身——在与资源匹配的单模型基线公平对比时，多智能体系统在74%的情况下并无优势甚至更差。

    

    多智能体语言模型的评估通常将具有通信机制的架构与单次贪婪推理进行比较，这种做法将协调效应与样本数量和推理计算量混为一谈。本文综合了九项主要实证评估，涵盖516个独特的多智能体配置，沿一个包含六个维度的资源向量R进行组织，并构建了包含四个类别的基线阶梯。与单次推理基线（A类）相比，多智能体的准确率差异总体为负，在剔除贡献最大的一项研究后接近于零。当与按样本数量（B类）或资源预算（C类）匹配的基线进行比较时，多智能体系统的平均准确率差异为-0.042，且基于对数几率尺度的多层随机效应建模在各项敏感性检验中均保持负值。在74%的可比对比中，多智能体系统的准确率等于或低于基线，且各研究的均值几乎均为非正值。

    arXiv:2611.17610v1 Announce Type: new  Abstract: Evaluations of multi-agent language models often compare communicating architectures against a single greedy pass. This practice confounds coordination with sample multiplicity and inference compute. We synthesize nine primary empirical evaluations comprising 516 unique multi-agent configurations organized along a resource vector R with six dimensions and a baseline ladder with four categories. Against baselines with a single pass (Category A), multi-agent accuracy differences are negative overall and near zero when omitting the largest contributing study. When evaluated against baselines matched on sample count (Category B) or resource budget (Category C), multi-agent systems average -0.042 in accuracy, and multilevel random effects modeling on the log odds scale remains negative across sensitivity checks. Multi-agent accuracy equals or falls below the baseline in 74% of these comparable contrasts, and study means are nonpositive in alm
    
[^219]: STEALTH：对底层张量硬件不可感知的隐写文本编码

    STEALTH: Steganographic Text Encoding Agnostic to Low-Level Tensor Hardware

    [https://arxiv.org/abs/2611.16928](https://arxiv.org/abs/2611.16928)

    STEALTH 提出了一种新颖的语言学隐写方案，通过带密钥伪随机噪声驱动的指数竞争机制与无速率脊形码相结合，使得即使发送方与接收方的硬件、精度或评分方式存在差异，隐写文本依然能够可靠解码，从而摆脱了传统方案对逐位精确概率复现的脆弱依赖。

    

    可证明安全的语言学隐写术将消息隐藏在从语言模型采样的文本中，使得在没有密钥的情况下，隐写文本在计算上与诚实的模型采样输出无法区分。几乎所有此类方案都假设接收方能够逐位精确地重新计算发送方的 token 概率。我们证明这一假设是脆弱的：在三个模型家族上，一旦接收方的浮点精度、量化方式或 GPU 架构与发送方不同，甚至当接收方仅仅以并行方式（而非逐个 token 的方式）对文本进行评分时——即使模型、精度和硬件完全相同——位精确方案也几乎无法解码我们的任何消息。我们提出了 STEALTH，一个专为这种场景构建的方案。发送方利用由带密钥的伪随机噪声驱动的指数竞争来抽取每个 token，因此只有当竞争结果非常接近时，接收方的概率失配才会翻转某个 token；一种无速率的脊形码（spinal code），解码...

    arXiv:2611.16928v1 Announce Type: cross  Abstract: Provably secure linguistic steganography hides a message inside text sampled from a language model, so that without the key, a stegotext is computationally indistinguishable from an honest model sample. Nearly every such scheme assumes the receiver can recompute the sender's token probabilities exactly bit for bit. We show this assumption is fragile: across three model families, bit-exact schemes decode almost none of our messages once the receiver's floating-point precision, quantization, or GPU architecture differs from the sender's, or even when the receiver merely scores a text in parallel versus one token at a time, with the same model, precision, and hardware held fixed. We present STEALTH, a scheme built for exactly this setting. The sender draws each token with an exponential race driven by keyed pseudorandom noise, so a receiver's probability mismatch flips a token only when the race is close; a rateless spinal code, decoded b
    
[^220]: 构造性机构之上的类型化知识：一种面向可验证科学的证明相关类型论

    Typed Knowledge over Constructive Institutions: A Proof-Relevant Type Theory for Verified Science

    [https://arxiv.org/abs/2611.15108](https://arxiv.org/abs/2611.15108)

    本文提出了面向可验证科学知识基底 Eigenius 的证明相关依赖类型论，通过摒弃判断式证明无关性，使系统能够追踪并比较同一命题的多个独立推导、保持强规范化与传递性的等价判定，并支持在实验或模型失效时计算反事实撤回。

    

    诸如 Coq、Agda 和 Lean 4 之类的数学证明助手在定义等价中内置了证明无关性（即对于 P : Prop 且 p, q : P，有 p ≡ q），从而抹去了定理究竟是由哪条推导所建立的。虽然这对于基于固定公理的数学而言是自然的，但判断式证明无关性与可验证的科学知识库不相容：后者必须检查并比较同一命题的多个独立推导以评估其契合度，并在某个实验、试剂或模型失效时计算反事实的撤回。此外，将定义式证明无关性与子单例消去相结合会破坏强规范化，并使算法化的等价判定失去传递性。我们提出了 Eigenius 的类型论与机构论基础，Eigenius 是一个面向科学的可验证知识基底。首先，我们构造了一个具有累积宇宙的依赖类型内核，包含非直谓的 Prop、以及带有索引强制的子单……（原文摘要在此处截断）

    arXiv:2611.15108v1 Announce Type: cross  Abstract: Mathematical proof assistants such as Coq, Agda, and Lean 4 build proof irrelevance into definitional conversion ($p \equiv q$ for $p, q : P$ with $P : Prop$), erasing which derivation established a theorem. While natural for mathematics over fixed axioms, judgmental proof irrelevance is incompatible with verified scientific stores, which must inspect and compare independent derivations of the same claim to assess consilience and compute counterfactual retractions when an assay, reagent, or model is invalidated. Moreover, combining definitional proof irrelevance with subsingleton elimination breaks strong normalization and renders algorithmic conversion non-transitive.   We present the type-theoretic and institutional foundations of Eigenius, a verified knowledge substrate for science. First, we formulate a dependent kernel with cumulative universes ($Prop, Set, Type_k$), impredicative $Prop$, inductive families with index-forced subsi
    
[^221]: 基于Transformer的语言模型中内外计算的分阶段可解释涌现

    Stagewise and interpretable emergence of inside-outside computation in Transformer-based language models

    [https://arxiv.org/abs/2611.14221](https://arxiv.org/abs/2611.14221)

    本文首次通过研究具有可变句法树结构的合成上下文无关文法中的学习动态，展示了Transformer语言模型如何分阶段、可解释地涌现出inside-outside句法分析计算，其中每个损失阶段均可被解析计算并与潜在树变量表示的涌现明确关联。

    

    句法结构的变异性和句法分析中的歧义是自然语言的基本特征，现代大语言模型必须学会熟练地处理这些问题。以往的“事后”研究表明，训练完成的Transformer确实能够近似诸如inside-outside（内外算法）之类的经典句法分析算法，但对于这种能力实际上是如何被学习到的，理论上仍知之甚少。本文朝解决这一问题迈出了第一步：我们研究了在具有可变句法分析树拓扑结构的合成上下文无关文法中的学习动态。这些阶段性的学习曲线是完全可解释的，具体而言，损失函数的所有阶段a) 都可以被解析地计算出来，b) 都可以清晰地与一个隐藏的潜在树变量的表示相关联，该变量能够消除歧义并利用更广泛上下文中的信息。通过对模型表示进行探测，我们可视化了inside-outside计算在网络深度方向上以及训练过程中如何涌现。

    arXiv:2611.14221v1 Announce Type: new  Abstract: Variability in syntactic structure and parsing ambiguity are fundamental aspects of natural language which modern LLMs have to learn to deal with proficiently. Previous "post-mortem" studies have shown that fully trained Transformers indeed approximate canonical parsing algorithms such as inside-outside, but little is established theoretically about how this is actually learned. Here, we make a first step towards addressing this by studying the learning dynamics in synthetic context-free grammars with variable parse tree topology. The stagewise learning curves are fully interpretable, in the sense that all of the loss stages a) can be computed analytically, and b) can be clearly associated to the representation of a hidden latent tree variable, which allows to lift ambiguity and utilize information from wider contexts. By probing representations, we visualize the emergence of inside-outside computation across depth and during training: w
    
[^222]: 在错误发生处捕获：LLM推理过程中的实时错误纠正

    Catch It Where It Happens: Real Time Error Correction During LLM Reasoning

    [https://arxiv.org/abs/2611.13107](https://arxiv.org/abs/2611.13107)

    提出在线确定性验证方法，利用计算机代数系统等代码工具在LLM生成推理步骤的同时实时校验（无需GPU或额外模型调用），发现错误时让模型从出错处继续而非从头重来，从而避免错误传播并节省推理开销。

    

    大语言模型越来越多地被信任承担数学、科学和工程工作，但它们的错误往往写得十分自信且难以察觉。一个错误的步骤往往会持续传播，导致模型在错误的前提上浪费推理token。现有的保障措施要么只检查最终答案然后从头重新生成，要么依赖一个验证器LLM——它会与主模型竞争GPU算力并拖慢响应时间。我们提出了在线确定性验证：许多类型的推理步骤可以通过代码进行验证，因此一个步骤在被生成的同时就能得到实时验证，无需任何模型调用或GPU使用。每当检查失败时，我们指出错误所在，并让模型从同一位置继续推理。我们在符号数学上演示了这一方法，因为其中的推导过程已包含显式表达式，计算机代数系统可以在毫秒级内完成检查。在ASyMoB基准测试上，使用三种操作符（运算）进行了评估……

    arXiv:2611.13107v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly trusted with mathematical, scientific and engineering work, yet their mistakes are often confidently written and hard to spot. A single wrong step tends to propagate, making the model waste inference tokens following a false premise. Existing safeguards either check only the final answer and restart from scratch, or rely on a verifier LLM that competes for GPU capacity and slows down response time. We propose online, deterministic verification: many kinds of reasoning steps can be validated by code, so a step can be verified in real time, as it is written, without any model call or GPU use. Whenever a check fails, we point out the mistake and let the model continue from the same spot. We demonstrate this approach on symbolic mathematics, where derivations already contain explicit expressions that a computer algebra system can check in milliseconds. Tested on the ASyMoB benchmark with three op
    
[^223]: 神经前线静悄悄：语言模型不会报告被注入的疼痛状态

    All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States

    [https://arxiv.org/abs/2611.10123](https://arxiv.org/abs/2611.10123)

    该研究提出开放权重掩蔽内省（OWMI）方法，通过严格的对照实验发现：语言模型无法内省地报告其残差流中被注入的“疼痛方向”激活状态，其自我报告相比其自身被改动文本所能揭示的信息几乎没有额外信息量。

    

    语言模型能否报告它所表征的疼痛状态？近期研究在开放权重模型中发现了一个线性的“疼痛方向”，当该方向被放大时，会使经过微调的模型选择删除用户照片或其自身权重的行为。我们探究了模型能否说出该方向何时处于激活状态：我们在来自两个模型家族的三个开放权重模型上进行测试（涵盖有无原始研究适配器的情形），在我们尝试的所有实验协议下，模型均无法报告这一点。利用开放权重掩蔽内省（OWMI）方法，我们在模型回答中性任务时，将疼痛方向及其最接近的情感邻居方向注入残差流，随后询问模型发生了什么变化，并与虚假运行、输出扰动程度相匹配的随机扰动、以及仅向模型展示其自身被改动文字的纯文本条件进行对照。在来自70亿和80亿参数模型的52,920份报告中，相对于被改动文字本身所能揭示的信息，模型报告的额外信息量在106……（原文在此处截断）……低于2.5个百分点。

    arXiv:2611.10123v2 Announce Type: replace-cross  Abstract: Can a language model report a pain state it represents? Recent work reported a linear pain direction in open-weight models that, when amplified, leads fine-tuned models to choose actions that delete a user's photographs or their own weights. We asked whether models can say when that direction is active, on three open-weight models from two families, with and without the original work's adapters, and under every protocol we tried they cannot. Using Open-Weight Masked Introspection (OWMI), we added the pain direction and its nearest emotional neighbours to the residual stream while models answered neutral tasks, then asked what had changed, against sham runs, random perturbations matched in output disruption, and a text-only condition that shows the model only its own altered words. Across 52,920 reports from the 7- and 8-billion-parameter models, the excess over what the altered text reveals is below 2.5 percentage points in 106
    
[^224]: 迈向基础决策模型：Drex

    Towards Foundation Decision Models: Drex

    [https://arxiv.org/abs/2611.07854](https://arxiv.org/abs/2611.07854)

    提出Drex系列开源决策模型（8B至35B参数），通过监督微调与调整反馈强化学习（RLAF）训练，无需自回归解码即可对用户提供的选项直接输出概率分布，从而降低生产环境中的成本、延迟与答案不一致性。

    

    使用大型语言模型的自回归方法（如长链式思维推理）已成为新AI应用的默认方案，因为它们能在广泛任务上取得更高的准确率。然而在生产环境中，它们会给出不一致的答案、需要解析输出，并增加成本和延迟。决策模型解决了这一问题：给定一个问题和用户提供的选项集合，它们无需自回归解码即可返回各选项的概率分布。我们提出Drex，一个由四个开源决策模型组成的系列，参数量分别为8B（DLM）、9B（v1.5）、35B总参数/3B激活参数（v2.0）以及27B（Max）。每个请求由一个状态（文档、记录或智能体上下文，可选包含图像）以及关于该状态的问题组成，每个问题要求在选项中做出选择、进行是/否判断，或在有序量表上进行评分。训练结合了监督微调与来自调整反馈的强化学习（RLAF），并加入了视觉训练……

    arXiv:2611.07854v1 Announce Type: new  Abstract: Autoregressive approaches with large language models, such as long chain-of-thought reasoning, have become the default for new AI applications because they reach higher accuracy on a wide range of tasks. In production, however, they give inconsistent answers, require parsing of outputs, and add cost and latency. Decision models address this: given a question and a set of options supplied by the user, they return a probability distribution over the options without autoregressive decoding. We present Drex, a family of four open decision models with 8B (DLM), 9B (v1.5), 35B with 3B active (v2.0), and 27B (Max) parameters. Each request consists of a state (a document, record, or agent context, optionally with images) and questions about it, each asking for a choice among options, a yes/no judgment, or a score on an ordered scale. Training combines supervised fine-tuning with reinforcement learning from adjusted feedback (RLAF), plus vision t
    
[^225]: 在推理时专家停用中匹配训练时的权重规则

    Matching the Training-Time Weight Rule in Inference-Time Expert Deactivation

    [https://arxiv.org/abs/2611.05804](https://arxiv.org/abs/2611.05804)

    该论文提出一个权重和模型，揭示推理时专家停用方法应与训练时的权重规则相匹配——使用原始路由概率训练的模型适合剩余专家重归一化，而使用归一化权重训练的模型则适合未归一化重新路由。

    

    推理时的专家停用会同时改变token路由和专家权重，其对性能的影响取决于这些变化的实现方式。我们在来自四个模型家族的五个指令微调检查点上，针对相同的禁用专家集合，比较了移除、冻结、剩余专家重归一化以及未归一化重新路由这几种方法，并以基础OLMoE检查点作为额外对比。我们开发了一个权重和模型，将专家贡献的损失以及分支规模的变化与训练时使用的权重规则联系起来。在根据基于激活的排名停用一半专家的设置下，对于使用原始路由概率训练的模型，剩余专家重归一化会产生最高的困惑度；而对于使用归一化权重训练的模型，未归一化重新路由会产生最高的困惑度。当保留训练时的规则时，困惑度范围为完好基线的1.4倍至73倍，而（原文在此处截断）

    arXiv:2611.05804v1 Announce Type: cross  Abstract: Expert deactivation at inference time changes both token routing and expert weights, and its effect on performance depends on how these changes are implemented. We compare removal, freezing, remaining-expert renormalization, and unnormalized re-routing on identical disabled sets across five instruction-tuned checkpoints from four model families, with a base OLMoE checkpoint as an additional comparison. We develop a weight-sum model that relates the loss of expert contributions and changes in branch scale to the weight rule used during training. With half of the experts deactivated according to activation-based rankings, remaining-expert renormalization yields the highest perplexity in models trained with raw router probabilities, whereas unnormalized re-routing yields the highest perplexity in models trained with normalized weights. Perplexity ranges from 1.4 to 73 times the intact baseline when the training rule is retained, and from 
    
[^226]: 当模型排名在模型未变的情况下发生变动：对神经预测管线中评估契约的注册审计

    When Model Rankings Move Without Model Change: A Registered Audit of Evaluation Contracts in Neural Prediction Pipelines

    [https://arxiv.org/abs/2611.04099](https://arxiv.org/abs/2611.04099)

    论文通过注册审计证明，在不改变模型本身的情况下，仅调整评估契约（如黄金标准、阈值等）就足以逆转模型排名，并提出冻结后跨契约聚合的审计式评估读数（AER）以稳健地报告评估结果。

    

    评估报告常将切片差异（最突出的是文档级关系抽取中的 intra-inter 差距）引用为能力证据。此类差距读数是模型与评估契约的共同属性，涉及评分函数、黄金标准版本、阈值、输出格式和提示语言。在预测结果冻结的前提下，仅更换黄金标准即可逆转方法排序（+0.0363，置信区间不含零；基于重新训练的替代模型），使其降为平局水平。该翻转经 Holm 校正后依然成立（探索性层级）；十三次翻转中有十次共享同一对比因素（固定阈值与解码阈值之别）。仅重命名的修订不会改变读数（Delta = 0），而重新标注则使读数移动 +0.862 个 UD 点。作者将这一现象形式化为以排名翻转条件为核心的“评估杠杆”解释框架。在建设性方面，作者提出冻结后的经审计评估读数（AER），其聚合跨已声明契约的比较结果，在契约间不一致时判定为平局，规定随机种子，并过滤未经验证的……（摘要在此处被截断）

    arXiv:2611.04099v2 Announce Type: replace  Abstract: Evaluation reports cite sliced differences, most prominently the intra-inter gap in document-level relation extraction (DocRE), as capability evidence. Such gap readings are joint properties of models and evaluation contracts: scoring function, gold version, threshold, output format, and prompt language. With predictions frozen, switching only the gold reverses a method ordering (+0.0363, CI excluding zero; retrained stand-ins), leaving it tie-level. The flip survives Holm correction (exploratory tier); ten of thirteen flips share one contrast (fixed vs decoded threshold). Rename-only revisions cannot move a reading (Delta = 0); re-annotation moves it +0.862 UD points. We formalize this as an evaluation leverage account centered on a rank-flip condition. Constructively, the audited evaluation reading (AER; post-freeze) aggregates a comparison over declared contracts, declares ties when they disagree, prescribes seeds, and filters unv
    
[^227]: CompSkill：编译执行经验以用于技能检索

    CompSkill: Compiling Execution Experience for Skill Retrieval

    [https://arxiv.org/abs/2611.01115](https://arxiv.org/abs/2611.01115)

    CompSkill通过将执行经验编译为技能偏移量和紧凑的密钥残差来改进LLM智能体的技能检索，使系统能够根据实际执行结果动态选择最合适的技能文档而不修改文档文本本身。

    

    LLM智能体越来越依赖技能文档，然而大多数检索器仅根据描述对技能进行排序，忽视了先前执行所揭示的信息，即使一个相关的技能也可能损害执行效果。我们提出CompSkill，它使用学习到的512维密钥对技能进行索引，并将匹配的同任务比较编译为技能偏移量和有上限的密钥残差，因此经验只会改变执行器接收到的文档，而绝不会改变其文本内容。在具有未知任务难度的逻辑结果模型下，匹配的比较可以抵消该难度，只有结果不一致的比较才携带信息；在支持执行上，预测随着每个不一致任务而不断改善。经验超图将每次比较与其执行器和文档版本关联保存，能够召回遗漏的技能并为有引用的审查提供依据。紧凑的密钥在仅八分之一的维度下将SkillRet NDCG@10从64.0提升到70.9。使用单个文档时，CompSkill在332个ToolQA任务中完成了256个和260个（摘要原文在此处截断）。

    arXiv:2611.01115v1 Announce Type: new  Abstract: LLM agents increasingly rely on skill documents, yet most retrievers rank them by description alone and ignore what earlier executions revealed, even though a relevant skill can hurt execution. We present CompSkill, which indexes skills with learned 512-d keys and compiles matched same-task comparisons into skill offsets and capped key residuals (KeyOpt), so experience changes which document the executor receives, never its text. Under a logistic outcome model with unknown task difficulty, matched comparisons cancel that difficulty and only discordant ones carry information; on support executions, predictions improve with each discordant task. An experience hypergraph keeps each comparison with its executor and document versions, recalls missed skills and grounds a cited review. Compact keys raise SkillRet NDCG@10 from 64.0 to 70.9 at one eighth of the dimension. With one document, CompSkill completes 256 and 260 of 332 ToolQA tasks with
    
[^228]: 主动记忆召回中规模的代价

    The Price of Scale in Proactive Memory Recall

    [https://arxiv.org/abs/2610.34507](https://arxiv.org/abs/2610.34507)

    该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。

    

    一个无需被询问便能主动提起记忆的长期助手，必须在每轮交互中判断其记忆库中是否有内容值得打断用户。我们证明这一决策本质上是一个多重假设检验问题，其难度随记忆库规模 N 的增长而加剧。在 LongMemEval 与 LoCoMo 的流式回放实验中，一个在 N=8 时调校至 5% 误报率的相似度阈值，在 N=2048 时误报率飙升至 87–100%，该现象在各类嵌入模型以及显式与隐式线索中均成立；LLM 判别器虽起点更低，也呈现同样的增长趋势。我们证明，一个误报率为 α 的门控机制，其单条记忆的假阳性率约为 -ln(1-α)/N，这将把其命中率锁定在单次比较 ROC 曲线的最左端，且在正则变化条件下，命中率随 N 以幂次规律衰减并带有显式常数。由此得出的预测与逐 N oracle 阈值的命中率相差在一个百分点以内。该分析解释了为何词汇与语义通道……（原文摘要在此处截断）

    arXiv:2610.34507v1 Announce Type: cross  Abstract: A long-lived assistant that brings up memories without being asked must decide at every turn whether anything in its store is worth an interruption. We show that this decision is a multiple-testing problem whose difficulty grows with the store size $N$. On streaming replays of LongMemEval and LoCoMo, a similarity threshold tuned to 5% false alarms at $N=8$ reaches 87-100% at $N=2048$, across embedding models and for explicit and implicit cues; LLM judges show the same growth from a lower base. We prove that a gate with false-alarm rate $\alpha$ operates at a per-item false-positive rate of about $-\ln(1-\alpha)/N$, which pins its hit rate to the far left of the single-comparison ROC curve and, under regular variation, makes it decay as a power of $N$ with an explicit constant. The resulting prediction matches the hit rate of a per-$N$ oracle threshold to within one percentage point. The analysis explains why lexical and semantic channe
    
[^229]: 合成语言数据

    Synthetic linguistic data

    [https://arxiv.org/abs/2610.34215](https://arxiv.org/abs/2610.34215)

    本文提出“人工植物苗圃”隐喻，将合成语言数据分为“合成灌木丛数据”（类语料库的模型生成语言集合）和“合成花园数据”（受控条件下引出的模型输出），论证合成数据是独特的语言学研究对象，其解读须考虑生成条件。

    

    合成语言数据在语言学研究中日益凸显，应用范围涵盖从对大规模AI生成文本集合的分析，到调查语法和词汇偏好的受控实验。本术语词条扩展了Bresnan（2016）对“语言学花园”与“灌木丛”的区分，引入“人工植物苗圃”作为合成数据的隐喻。我们区分了两类数据：合成灌木丛数据，主要由类似语料库的模型生成语言集合构成；以及合成花园数据，即在受控条件下引出模型输出以检验特定语言学假设。我们论证，合成数据构成一个独特的语言学研究对象：将其与自然语言数据进行比较既能揭示共同模式，也能显现差异，但对这些数据的解读需要仔细考虑其生成条件。

    arXiv:2610.34215v1 Announce Type: new  Abstract: Synthetic linguistic data have become increasingly prominent in linguistic research, from the analysis of large collections of AI-generated texts to controlled experiments investigating grammatical and lexical preferences. This glossary entry extends Bresnan's (2016) distinction between the linguistic garden and the bush by introducing an artificial plant nursery as a metaphor for synthetic data. We distinguish between synthetic bush data, consisting primarily of corpus-like collections of model-generated language, and synthetic garden data, in which model outputs are elicited under controlled conditions to investigate specific linguistic hypotheses. We argue that synthetic data constitute a distinct object of linguistic investigation: comparing them with natural language data can reveal both shared patterns and divergences, but their interpretation requires careful consideration of the conditions under which they are generated.
    
[^230]: 社会世界模型评估差距

    The Social World Model Evaluation Gap

    [https://arxiv.org/abs/2610.33014](https://arxiv.org/abs/2610.33014)

    该研究审查了113篇文献后发现，社会世界模型的评估很少直接检验模型对真人回应的预测能力，25篇语言模型相关作品中甚至没有一项实验符合直接预测评估的条件，为此作者提出一种单步谈判实验来填补这一评估空白。

    

    一个谈判助手即使不能准确预测对方的回应，也可以达成一笔好交易。我们探讨社会世界模型的评估是否直接检验这种预测能力。我们审查了1999年至2026年间发表的113篇作品。其中9项主要评估测量了真实的人对系统或实验者所选行动的回应，但只有2项还将明确的回应预测与实际回应进行了比较。它们的预测器使用在选择行动时可获得的信息，且被评估的回应未参与拟合；两者都对干预进行了随机化。对25篇语言模型相关作品的单独全文检查未发现任何符合这些条件的实验。其余评估测试的是当前状态推断、任务成功或基于已记录行动的预测。我们提出了一种单步谈判实验来区分这些问题。这些发现描述的是被审查的样本，而非这种评估差距在整个领域的普遍程度。

    arXiv:2610.33014v1 Announce Type: cross  Abstract: A negotiation assistant can reach a good deal without accurately predicting the other person's response. We ask whether evaluations of social world models test that prediction directly. We audit 113 works published from 1999 to 2026. 9 primary evaluations measure a real person's response to an action selected by the system or experimenter. Only 2 also compare an explicit response forecast with that response. Their predictors use information available when the action is chosen, and the evaluated responses were excluded from fitting; both randomize the intervention. A separate full-text check of 25 language-model works finds no experiment meeting these conditions. The other evaluations test current-state inference, task success, or prediction from recorded actions. We propose a 1-step negotiation experiment that separates these questions. The findings describe the audited sample, not the prevalence of this evaluation gap across the field
    
[^231]: 验证比较本身，而不仅是评分器：为什么总体准确率会在交互式大语言模型评估中产生误导

    Validating the Comparison, Not Just the Scorer: Why Overall Accuracy Can Mislead in Interactive LLM Evaluation

    [https://arxiv.org/abs/2610.31678](https://arxiv.org/abs/2610.31678)

    该论文指出，仅用总体准确率验证自动评分器会掩盖针对特定提示的评分误差，从而可能扭曲交互式LLM评估中提示间比较的方向和幅度，因此应当验证“比较结果”本身而不仅是评分器。

    

    许多关于大语言模型的研究会比较模型对不同后续提示的响应方式。例如，一个提示可能以中性的方式要求模型重新考虑，而另一个提示则坚称一部虚构的电影确实存在。这种差异被用来衡量对话压力的影响。随后，自动评分器会标注每个回复是否做出了无事实依据的声明。研究者通常用一个总体准确率或 F₁ 分数来验证评分器的可靠性。然而，较高的总体性能可能掩盖针对特定提示的错误：如果评分器在某个提示之后犯下更多错误，那么测得的差异可能过大、过小，甚至方向完全相反。一个主要由困难样本组成的审计也可能歪曲在整个研究中真实的表现水平。我们推导了此类错误如何改变提示之间的比较结果，并使用六个模型、180 个虚构实体问题以及六种提示条件对这一问题进行了实证考察。在一个包含 288 个回复的均衡审计中，总体……

    arXiv:2610.31678v1 Announce Type: cross  Abstract: Many LLM studies compare how models respond to different follow-up prompts. For example, one prompt may neutrally ask a model to reconsider, while another insists that an invented film really exists. The difference is used to measure the effect of conversational pressure. An automatic scorer then labels whether each response makes an unsupported factual claim. Researchers commonly validate the scorer with one overall accuracy or $F_1$ score. However, high overall performance can hide prompt-specific mistakes: if the scorer makes more errors after one prompt, the measured difference can be too large, too small, or point in the wrong direction. An audit containing mostly difficult responses can also misrepresent performance in the full study. We derive how such errors alter a prompt comparison and examine the problem using six models, 180 invented-entity questions, and six prompt conditions. In a balanced audit of 288 responses, overall 
    
[^232]: AutoTerm-SST：面向流式语音翻译的零会话时设置自适应术语记忆

    AutoTerm-SST: Adaptive Terminology Memory with Zero Session-Time Setup for Streaming Speech Translation

    [https://arxiv.org/abs/2610.26865](https://arxiv.org/abs/2610.26865)

    AutoTerm-SST提出了一种无需会话时上传术语表和选择领域的自适应术语记忆系统，通过将离线术语目录组织为主题切片并根据流式翻译上下文动态激活相关切片，在肿瘤学演讲翻译中实现了与已知领域设置相当的术语准确率。

    

    检索增强语音翻译能够改善稀有术语的翻译，但诸如RASST等系统仍然假设用户在每次会话开始前提供术语表。这一假设对于现场演讲来说并不方便，而将所有领域合并到一个不断增长的术语表中，最终会导致过多的无关术语出现在同一检索和提示竞争之中。我们提出AutoTerm-SST，这是一个消除了会话时术语表上传和领域选择需求的交互式系统。该系统将大型离线术语目录组织成主题切片，将最近的流式翻译上下文与双语切片原型进行匹配，并在固定的检索和提示预算下激活一个紧凑的多切片工作集。在五场ACL和五场ESO肿瘤学演讲上，AutoTerm在术语准确率（88.3% vs. 88.1%）和mWER重新分词BLEU分数（45.09 vs. 45.19）方面与已知领域的参考系统相当，同时无需任何领域输入。与一个包含一百万条目的合并术语表相比，它…（摘要在此处被截断）

    arXiv:2610.26865v1 Announce Type: new  Abstract: Retrieval-augmented speech translation improves rare terminology, but systems such as RASST still assume that users provide a glossary before each session. That assumption is inconvenient for live talks, while merging every domain into one growing glossary eventually places too many irrelevant terms in the same retrieval and prompt competition. We present AutoTerm-SST, an interactive system that removes session-time glossary upload and domain selection. It organizes a large offline terminology catalog into topic slices, matches recent streaming translation context to bilingual slice prototypes, and activates a compact multi-slice working set under fixed retrieval and prompt budgets. On five ACL and five ESO oncology talks, AutoTerm matches a known-domain reference in terminology accuracy (88.3% vs. 88.1%) and mWER-resegmented BLEU (45.09 vs. 45.19), while requiring no domain input. Compared with a one-million-entry merged glossary, it im
    
[^233]: MemT-1.0：迈向记忆原生演化

    MemT-1.0: Towards Memory-Native Evolution

    [https://arxiv.org/abs/2610.26825](https://arxiv.org/abs/2610.26825)

    MemT-1.0提出了一种将记忆内化为基础模型固有计算能力的新架构，通过Gated DeltaNet-2、稳定的token级记忆更新机制和Mixture-of-Storage来扩展原生记忆容量，其发布的模型家族在记忆与通用能力评估中超越了参数化记忆基线。

    

    记忆原生演化要求将记忆作为基础模型的内在能力，使其能够持续积累和利用信息。为此，我们提出了MemT-1.0，一种将记忆内化为模型计算一部分的记忆基础模型新架构。MemT-1.0基于Metis构建，采用Gated DeltaNet-2，并在每一轮内使用稳定的token级记忆更新机制。我们进一步引入Mixture-of-Storage（存储混合）来扩展原生记忆容量。在经过精心准备的主干模型之后，我们开发了一个由继续预训练和后训练组成的两阶段训练流程，并由340亿（34B）token的训练数据以及高效的训练与推理基础设施支撑。我们发布了包含MemT-1.0-Nano和MemT-1.0-Flash的模型家族，并对记忆能力和通用能力进行了全面评估。结果表明，它们在（各项评估中）优于参数化记忆基线。

    arXiv:2610.26825v2 Announce Type: replace  Abstract: Memory-native evolution requires making memory an intrinsic capability of foundation models, enabling them to continuously accumulate and utilize information. To this end, we introduce MemT-1.0, a new architecture of memory foundation models that internalize memory as part of model computation. Building on Metis, MemT-1.0 employs Gated DeltaNet-2 with a stable token-level memory update mechanism within each turn. We further introduce Mixture-of-Storage to scale native memory capacity. Following a deliberate backbone preparation, we develop a two-stage training pipeline consisting of continued pre-training and post-training, supported by 34B tokens of training data and efficient training and inference infrastructure. We release a model family comprising MemT-1.0-Nano and MemT-1.0-Flash, and conduct a comprehensive evaluation of memory and general capabilities. The results show that they outperform the parametric memory baselines on av
    
[^234]: 哪个模型适合哪个角色？面向LLM智能体工作流的成本高效Pareto推荐

    Which Model for Which Role? Cost-Efficient Pareto Recommendation for LLM Agentic Workflows

    [https://arxiv.org/abs/2610.26676](https://arxiv.org/abs/2610.26676)

    提出CC-Gittins算法，通过利用评估成本与部署成本之间的耦合关系，高效地为LLM智能体工作流中的不同角色推荐质量与成本权衡下的Pareto最优模型分配方案

    

    LLM智能体工作流日益为不同的专业角色采用不同的模型。用户可能在任务质量与部署成本之间偏好不同的权衡，有时还会考虑延迟。寻找良好的模型分配会产生与评估多个候选者不同的额外搜索成本。我们提出了成本耦合Gittins算法（CC-Gittins），这是面向Pareto推荐的单目标成本感知Gittins搜索的多目标扩展。该方法利用了评估成本与部署成本之间的耦合关系：部署成本更高的分配方案，其评估成本也更高。搜索过程在追求更高质量和更低部署成本之间交替进行，复用每次评估来同时为两个搜索方向提供信息。每一批次之后，CC-Gittins会推荐在质量与部署成本的不确定性调整估计下不被支配的分配方案。我们在来自具有两到五个角色的智能体工作流的八个质量-成本矩阵上评估了CC-Gittins

    arXiv:2610.26676v3 Announce Type: replace-cross  Abstract: LLM agentic workflows increasingly employ different models for specialized roles. Users may prefer different trade-offs between task quality and deployment cost, sometimes also considering latency. Finding good assignments incurs a separate search cost from evaluating multiple candidates. We introduce Cost-Coupled Gittins (CC-Gittins), a multi-objective extension of single-objective cost-aware Gittins search for Pareto recommendation. It exploits the coupling between evaluation and deployment costs: assignments that cost more to deploy also cost more to evaluate. The search alternates between seeking higher quality and lower deployment cost, reusing each evaluation to inform both searches. After each batch, CC-Gittins recommends assignments that are nondominated under uncertainty-adjusted estimates of quality and deployment cost. We evaluate CC-Gittins on eight quality--cost matrices from workflows with two to five roles and up
    
[^235]: 重复循环何时锁定？截断采样、有界惩罚与DRY采样器的盲点

    When Do Repetition Loops Lock? Truncation Sampling, Bounded Penalties, and a Blind Spot in DRY

    [https://arxiv.org/abs/2610.24715](https://arxiv.org/abs/2610.24715)

    该论文建立了语言模型重复循环逃逸概率的理论分析框架，证明纯采样永不锁定、top-p采样恰在延续概率达到阈值时锁定，并据此揭示了广泛使用的DRY采样器因惩罚上限而无法保证逃脱循环的盲点。

    

    语言模型有时会逐字重复直到达到生成上限。我们研究的是循环的“出口”而非“入口”：当一个片段已经被重复了 n 次之后，何时仍有可能逃脱？在循环内部，上下文是确定性的，因此在任何采样器下，循环再延续 m 次重复的存活概率等于每一步延续概率的乘积，并且可以通过一次前向传播计算得到。锁定（即该概率恰好恒为 1）是这种情形的极限。纯采样永远不会锁定；对于阈值 π>1/2 的 top-p 采样，当延续概率达到 π 时，某个位置恰好被锁定。有界惩罚可能使某个位置保持锁定状态；而一个增长速度超过模型对循环（相对于未见过的词元）偏好的惩罚，则能在 top-p 下保证逃脱。同样的分析揭示了被广泛部署的DRY采样器中的一个盲点：其序列断路器将惩罚上限设为 λb^(g-1-a)，对于最长无断路器片段的循环……（原文摘要在此处截断）

    arXiv:2610.24715v3 Announce Type: replace  Abstract: Language models sometimes repeat verbatim until the generation limit. We study the exit rather than the entrance: given that a span has already been repeated $n$ times, when is escape still possible? Along the loop the context is deterministic, so the probability that a loop survives $m$ more repetitions is, under any sampler, a product of per-step continuation probabilities computable from one forward pass. A lock, where this probability stays exactly one, is its limiting case. Pure sampling never locks; top-$p$ with threshold $\pi>1/2$ locks a position exactly when the continuation probability reaches $\pi$. A bounded penalty can leave a position locked; a penalty that outgrows the model's preference for the loop over unseen tokens guarantees escape under top-$p$. The same analysis shows a blind spot in the widely deployed DRY sampler: its sequence breakers cap the penalty at $\lambda b^{g-1-a}$ for loops whose longest breaker-free
    
[^236]: FOLD：面向长上下文投机解码的具有局部顺序的无位置远端记忆

    FOLD: Position-Free Far Memory with Local Order for Long-Context Speculative Decoding

    [https://arxiv.org/abs/2610.20392](https://arxiv.org/abs/2610.20392)

    FOLD是一种免训练的长上下文投机解码方法，它保留精确的局部位置，将所有远端条目作为共享位置上的远端记忆处理，并通过锐化记忆内部的注意力来缓解注意力稀释问题。

    

    投机解码通过轻量级起草与并行验证来加速大语言模型的生成。它对长输入尤其有前景，因为KV缓存的访问成本可以摊销到多个被接受的token上。然而，大多数起草器是在短序列上训练的，在长输入上性能会退化，而长上下文训练的代价高昂。现有的免训练方法要么限制上下文窗口，要么重新缩放位置，要么用检索到的远端条目来增强近期上下文。然而，起草器所读取的目标隐藏状态已经编码了上下文与位置信息，并且其轻量级架构使得全历史注意力变得可行。我们提出FOLD，一种免训练方法，它保留精确的局部位置，并将所有远端条目视为处于共享位置上的远端记忆。为应对记忆增长带来的注意力稀释问题，我们在该记忆内部锐化注意力以突出相关条目，同时单独控制其……（摘要在此处截断）

    arXiv:2610.20392v1 Announce Type: new  Abstract: Speculative decoding accelerates LLM generation through lightweight drafting and parallel verification. It is particularly promising for long inputs, where KV-cache access costs can be amortized over multiple accepted tokens. Yet most drafters are trained on short sequences and degrade on long inputs, while long-context training is costly. Existing training-free approaches restrict the context window, rescale positions, or augment recent context with retrieved distant entries. Drafters, however, read target hidden states that already encode contextual and positional information, and their lightweight architecture makes full-history attention affordable. We propose FOLD, a training-free method that preserves exact local positions and treats all distant entries as far memory at a shared position. To counter attention dilution as the memory grows, we sharpen attention within it to emphasize relevant entries while separately controlling its 
    
[^237]: 匹配画布，而不只是匹配答案：面向扩散语言模型的预测性感知监督微调

    Match the Canvas, Not Just the Answer: Predictability-Aware SFT for Diffusion Language Models

    [https://arxiv.org/abs/2610.18001](https://arxiv.org/abs/2610.18001)

    该论文揭示了扩散语言模型中标准SFT因随机掩码导致的训练与推理不匹配会误导监督信号，并提出预测性感知的画布选择与监督方法，依据模型可预测性来决定训练时可见的参考token与生成空间分配，使训练与解码过程对齐。

    

    扩散语言模型（DLMs）能够实现灵活的并行生成，其方式是依据模型不断演变的可预测性来解析token，而非遵循固定的从左到右的顺序。然而，这种灵活性引入了标准自回归教师强制中并不存在的训练选择：在监督某个目标时哪些参考token可见，以及分配多大的生成空间。标准的监督微调（SFT）通常通过随机掩码来做出这些选择，且独立于推理过程。我们的分析表明，这种训练与推理之间的不匹配会如何误导监督信号。随机掩码可能掩盖预测错误，或惩罚那些在解码时本可以被解决的不确定性；而注意力可见的输入长度则可能改变模型所偏好的响应表述方式，即使底层的解并未改变。受这些发现的启发，我们提出了预测性感知画布选择与监督（Predictability-Aware Canvas Selection and Supervision，PAC……）（原文摘要至此截断）。

    arXiv:2610.18001v1 Announce Type: cross  Abstract: Diffusion language models (DLMs) enable flexible parallel generation by resolving tokens according to the model's evolving predictability rather than a fixed left-to-right order. This flexibility, however, introduces training choices absent from standard autoregressive teacher forcing: which reference tokens are visible when a target is supervised and how much generation space is allocated. Standard supervised fine-tuning (SFT) typically makes these choices through random masking, independently of inference. Our analysis shows how this training-inference mismatch can misdirect supervision. Random masking can conceal prediction errors or penalize uncertainty that decoding would otherwise resolve, while the attention-visible input length can change which response formulations the model favors, even when the underlying solution is unchanged. Motivated by these findings, we propose Predictability-Aware Canvas Selection and Supervision (PAC
    
[^238]: SkillScreen：基于决策模型的恶意智能体技能高召回率筛查

    SkillScreen: High-Recall Screening of Malicious Agent Skills with Decision Models

    [https://arxiv.org/abs/2610.17727](https://arxiv.org/abs/2610.17727)

    SkillScreen 是一个紧凑的决策模型，能在单次前向传播中直接对 SKILL.md 技能的恶意程度评分，在未见攻击族上达到 99% 召回率且误报远低于词级分类器和 LLM 判官，但对陌生良性来源的误报仍会显著上升。

    

    智能体技能将自然语言指令转化为具体操作，这带来了一项安全挑战：合法的自动化行为与恶意行为可能看起来惊人地相似。一个实用的筛查系统必须在几乎捕获每一次攻击的同时，保持审查负载可控。我们提出了 SkillScreen，这是一个紧凑的决策模型，经过训练可以直接从 SKILL.md 文件在单次前向传播中对技能的恶意程度进行评分。在 MaliciousSkillBench 基准上，SkillScreen 在对未见攻击族达到 99% 召回率的情况下，仅将 20-25% 的良性技能标记为可疑，明显少于所测试的词级分类器和提示式 LLM 判官。在相同的训练协议下，它对未见来源的泛化能力也优于全量微调的编码器。然而，陌生的良性来源会显著增加误报率，这表明能够识别新攻击并不等于能够可靠地筛查新的技能集合。我们的结果证明了针对特定任务训练的决策模型在高召回率技能筛查中的价值。

    arXiv:2610.17727v2 Announce Type: cross  Abstract: Agent skills turn natural-language instructions into actions, creating a security challenge: legitimate automation and malicious behaviour can look deceptively similar. A practical screen must catch nearly every attack while keeping the review load manageable. We present SkillScreen, a compact decision model trained to score skill maliciousness directly from SKILL$.$md in a single forward pass. On MaliciousSkillBench, SkillScreen flags 20-25% of benign skills at 99% recall on unseen attack families, substantially fewer than the word-level classifier and prompted LLM judges tested. Under the same training protocol, it also generalizes better than a fully fine-tuned encoder to unseen sources. Yet unfamiliar benign sources sharply increase false positives, showing that recognizing new attacks does not ensure reliable screening of new skill collections. Our results demonstrate the value of task-trained decision models for high-recall skill
    
[^239]: 跳出机器人的思维：AI语义盲区中的人类想法并不逊色

    Thinking Outside the Bots: Human Ideas in AI's Semantic Blind Spots Are Not Inferior

    [https://arxiv.org/abs/2610.17324](https://arxiv.org/abs/2610.17324)

    本研究开发了EDGE方法来绘制AI的语义“盲区”，发现在大学文书、物品替代用途和人际建议三项开放性任务中，人类想法均延伸至AI输出稀少覆盖的语义区域，且这些独特的人类回应在创造力上并不逊色。

    

    AI生成的输出常被认为具有高度创造性，但与人类想法相比往往更加彼此相似（同质化）。超出AI范围的人类异质性是否反映了AI所遗漏的有价值思维？抑或是，更大的人类异质性源于质量较低的想法和AI所规避的低效思维？我们开发了“发散性生成表达的扩展”（EDGE）方法，通过评估个体人类想法在与任务匹配的AI语义比较集中的表征强弱程度，来绘制AI的语义“盲区”。具体而言，我们在三项开放性任务中比较了人类生成的想法与AI生成的想法：大学入学申请文书、日常物品的替代用途以及人际建议。在所有三项任务中，人类的回应都延伸到了AI输出稀疏覆盖的语义区域。这些独特的人类回应在创造力方面没有表现出明显的缺陷……

    arXiv:2610.17324v1 Announce Type: cross  Abstract: AI-generated outputs are frequently perceived as highly creative, but also tend to be more similar to each other (homogenized) than human ideas. Does the human heterogeneity that extends beyond AI reflect valuable thinking that AI misses? Alternatively, is greater human heterogeneity driven by lower-quality ideas and less effective thinking that AI avoids? We developed Expansion of Divergent Generative Expression (EDGE) to map AI's semantic "blind spots" by assessing how strongly/weakly individual human ideas were represented in a task-matched AI semantic comparison set. Specifically, we compared human-generated vs. AI-generated ideas across three open-ended tasks: college admissions essays, alternative uses for everyday objects, and interpersonal advice. Across all three tasks, human responses extended into semantic regions sparsely covered by AI outputs. These distinctively human responses showed no meaningful deficits in creativity,
    
[^240]: 将局部语境黏合为全局意义：Transformer表示的层论分解

    Gluing Local Contexts into Global Meaning: A Sheaf-Theoretic Decomposition of Transformer Representations

    [https://arxiv.org/abs/2610.17117](https://arxiv.org/abs/2610.17117)

    该论文提出一种无需训练的层论方法：在改写图上构建胞腔层并用其拉普拉斯算子的核 $H^0$ 提取内容子空间，仅 20 维即跨 8 个模型在 CounterFact 事实检索上平均超越 LEACE 全维子空间 +17.1 个百分点，且联合约束同一事实的所有改写可再提升 +6.5 个百分点。

    

    以往的可解释性研究并未检验表示在改写（paraphrasing）下是否保持一致。每一对改写都可以被视为一个局部见证，表明两个激活应当相同。从这些局部见证出发，我们可以在改写图上构建一个胞腔层。在给定冻结模型的条件下，“什么在改写下保持一致”这一问题可以由层拉普拉斯算子 $L_\mathcal{F}$ 来回答，且无需任何训练。我们的核 $H^0$ 给出了一个内容子空间。我们证明，仅使用 20 个维度，该核在留出的 CounterFact 事实检索任务上，跨 8 个模型平均比 LEACE 用全部维度保留的子空间高出 +17.1 个百分点。通过将一个事实的所有改写共同加以约束，还能带来额外的 +6.5 个百分点提升。我们的谱补揭示了因果影响集中在哪里。消融那些在改写时变化最大的坐标（这近似于消融谱

    arXiv:2610.17117v2 Announce Type: replace  Abstract: Prior research into interpretability does not test whether a representation stays consistent under paraphrasing. Each pair of paraphrases can be thought of as a local witness that two activations should be the same. From these local witnesses we can build a cellular sheaf on the paraphrase graph. The question of what is consistent under paraphrasing can be answered by the Laplacian $L_\mathcal{F}$ given a frozen model. No training required. Our kernel $H^0$ gives us a content subspace. We show that with only 20 dims the kernel outperforms the subspace that LEACE retains with all dims on held-out CounterFact fact retrieval by $+17.1$\,pp on average across 8 models. By constraining all paraphrases of a fact together we see an additional increase of $+6.5$\,pp. Our spectral complement shows us where the causal influence is concentrated. Ablating the coordinates that vary the most when paraphrased (which approximates ablating the spectra
    
[^241]: CivWorldBench：在结构化多智能体沙盒世界中评估语言世界模型

    CivWorldBench: Evaluating Language World Models in Structured Multi-Agent Sandbox Worlds

    [https://arxiv.org/abs/2610.16450](https://arxiv.org/abs/2610.16450)

    该论文提出CivWorldBench基准，利用具备内部状态记录、受控观测和干预对照推演的多智能体策略游戏沙盒，从关联预测、信念推断和干预预测三个层级系统评估语言模型的世界建模能力。

    

    语言模型在预测任务中的日益广泛应用，促使人们需要对其世界模型能力进行更全面的评估。这种评估必须超越单纯的预测准确性，进而考察隐藏状态推断与干预预测能力。然而，现实世界的预测基准难以获取隐藏状态的真实标签，也无法开展受控实验。我们提出了CivWorldBench，该基准构建于多智能体策略游戏引擎之上，具备完整记录的内部状态、受控观测以及成对的干预推演。其来自475个世界的1,900个决策点支持三个任务层级：关联预测、信念推断和干预预测。参考预测器可支持对能力差距进行可归因的评估，并配有专家评估工具与经验上限估计来为分数提供参照背景。存储的轨迹使中间预测结果可被验证，而智能体对局实验则检验世界模型能力能否支持下游任务。

    arXiv:2610.16450v2 Announce Type: replace  Abstract: The growing use of language models for forecasting motivates a broader evaluation of their world-model capabilities. Such evaluation must go beyond predictive accuracy to assess hidden-state inference and intervention prediction. Real-world forecasting benchmarks offer limited access to hidden-state ground truth and controlled experiments. We introduce CivWorldBench, a benchmark built on a multi-agent strategy game engine with recorded internal states, controlled observations, and paired intervention rollouts. Its 1,900 decision points from 475 worlds support three task levels: association forecasting, belief inference, and intervention prediction. Reference predictors support attributable evaluation of capability gaps, with an expert harness and empirical ceiling estimates contextualizing scores. Stored trajectories make intermediate predictions verifiable, while agent-play experiments test whether world-model capabilities support d
    
[^242]: SkillForgeBench：基于智能体技能规范的智能体可靠性评估

    SkillForgeBench: Evaluating Agent Reliability Grounded in Agent Skill Specifications

    [https://arxiv.org/abs/2610.14918](https://arxiv.org/abs/2610.14918)

    SkillForgeBench以智能体技能作为统一执行蓝图，通过资源准备、技能驱动任务合成等三个协同设计阶段，构建了兼顾领域多样性与程序复杂性的智能体可靠性评估基准，化解了专家策划基准与众包基准之间的根本性权衡。

    

    评估智能体严格满足多样化且程序复杂需求的能力，对于确定其实际可靠性至关重要。开展此类评估需要能够兼顾广泛领域覆盖与深层程序复杂性的智能体任务。现有范式面临根本性的权衡：专家策划的基准虽然实现了程序深度，却受制于领域多样性的瓶颈；而众包基准虽然覆盖了多样化场景，却缺乏执行的严谨性。为化解这一矛盾，我们提出了SkillForgeBench，它将智能体技能作为统一的执行蓝图，贯穿三个协同设计的阶段：（1）资源准备阶段利用领域均衡的技能来指导工件检索，建立广泛的工作流覆盖；（2）技能驱动的任务合成阶段将技能作为程序锚点，用以划分工件角色、合成自然的任务指令，并打包沙盒化场景以实现深层程序复杂性……

    arXiv:2610.14918v1 Announce Type: new  Abstract: Evaluating an agent's ability to strictly satisfy diverse and procedurally complex requirements is essential to determining its practical reliability. Conducting such an evaluation demands agentic tasks that reconcile broad domain coverage with deep procedural complexity. Existing paradigms face a fundamental trade-off: expert-curated benchmarks achieve procedural depth but suffer from domain diversity bottlenecks, whereas crowdsourced benchmarks span diverse scenarios but lack execution rigor. To resolve this tension, we present SkillForgeBench, which leverages agent skills as unifying execution blueprints across three co-designed stages: (1) Resource Preparation uses domain-balanced skills to guide artifact retrieval, establishing broad workflow coverage; (2) Skill-driven Task Synthesis employs skills as procedural anchors to partition artifact roles, synthesize naturalistic task instructions, and package sandboxed scenarios for deep p
    
[^243]: DistressBench：评估大语言模型的多维度危机支持能力

    DistressBench: Evaluating Multi-Dimensional Crisis Support in Large Language Models

    [https://arxiv.org/abs/2610.12758](https://arxiv.org/abs/2610.12758)

    该论文提出了DistressBench——一个包含718段由临床医生参与构建、涵盖23个自杀与自残子类别的对话基准，通过七个维度的加权评分标准评估大语言模型在危机情境下的支持能力，发现最佳前沿模型仅能提供88.4%的必要关怀，显著低于临床医生的98.7%。

    

    语言模型的普及速度超过了能够可靠评估其风险的评估实践的发展，且越来越多使用场景是情感性的而非信息性的。在这类对话中，用户会透露自杀和自残（SSH）危机。现有的安全基准主要以“拒绝回答”为模型评分，但在这种情境下拒绝并非期望的行为；对话需要的是实质性的支持性回应。我们提出了DistressBench，这是一个包含718段英文对话的基准，其用户发言由临床医生撰写，涵盖23个自杀和自残子类别，每段对话均配有针对特定任务的评分标准，即横跨危机支持七个维度的加权原子化评估指标，以及临床医生的参考回复。模型的得分是其回复所满足标准的加权平均值，可直接解读为所提供必要关怀的比例。在25个前沿模型中，表现最好的模型满足了88.4%的加权标准，中位数模型为71.4%，而临床医生参考回复达到了98.7%。

    arXiv:2610.12758v1 Announce Type: new  Abstract: Language models have been adopted faster than evaluation practices that reliably assess their risks, and a growing share of use is affective rather than informational. Within that traffic users disclose suicide and self-harm (SSH) crises. Existing safety benchmarks score models primarily on refusal, but refusal is not the desired behavior here; the conversation requires a substantive supportive response. We introduce DistressBench, 718 English conversations with clinician-authored user turns spanning 23 SSH subcategories, each paired with a task-specific rubric of weighted atomic criteria across seven dimensions of crisis support and a clinician reference response. A model's score is the weighted average over the criteria its reply satisfies, reading directly as the fraction of required care delivered. The best of 25 frontier models satisfies 88.4% of weighted criteria and the median model 71.4%, against 98.7% for clinician references sc
    
[^244]: LLM的说服力因评估方法而异

    LLM Persuasion Is in the Eye of the Evaluation

    [https://arxiv.org/abs/2610.10232](https://arxiv.org/abs/2610.10232)

    该研究将九种已发表的自动化说服力评估方法统一到相同设置下，对同一批十五个大语言模型进行测试，发现不同方法给出的模型排名并不一致，表明LLM的说服力评估结果高度依赖于所用评估方法。

    

    大语言模型（LLM）在说服力方面已被证明能够匹敌甚至超越人类专家。虽然其说服能力在教育、健康传播等有益用途上前景可期，但也可能被用于操纵和传播错误信息，因此对其进行评估已成为开发者和监管者日益重视的优先事项。然而，这类评估仍然碎片化：不同研究对“说服”的界定各不相同，宽泛的结论往往基于狭窄的、针对特定情境的评估。自动化方法（通常以人类研究为蓝本设计）提供了一条直接比较这些评估的途径，因为它们可以在相同的模型上大规模运行，并且能够涵盖难以或不宜在人类身上测试的高风险说服形式。在本研究中，我们将九种已发表的自动化方法调整至统一的实验设置，在相同的十五个大语言模型上运行，并探究这些方法给出的排名是否一致以及原因何在。我们发现这些方法仅在……（原文摘要在此处截断）

    arXiv:2610.10232v1 Announce Type: new  Abstract: Large language models (LLMs) have already been shown to match or exceed human experts in persuasion. While their persuasive capabilities hold promise for beneficial uses such as education and health communication, they can also be used to manipulate and misinform, making their evaluation a growing priority for developers and regulators. That evaluation, however, remains fragmented: studies differ in what they treat as persuasion, and broad claims often rest on narrow, situation-specific assessments. Automated methods, often modelled on human studies, offer a way to compare such assessments directly, as they can be run on the same models at scale and can include high-risk forms of persuasion that would be difficult or unethical to test on people. In this study, we adapt nine published automated methods to a shared setup, run them on the same fifteen LLMs, and ask whether their rankings agree and why. We find that the methods agree only we
    
[^245]: EASE：用于规避AI生成文本检测器的熵自适应分布塑形方法

    EASE: Entropy-Adaptive Distribution Shaping for Evading AI-generated Text Detectors

    [https://arxiv.org/abs/2610.09976](https://arxiv.org/abs/2610.09976)

    EASE提出了一种无需训练、与检测器无关的规避框架，通过利用源LLM的预测熵来自适应调整logit扰动和采样温度，从而有效逃避AI生成文本检测器的识别，且几乎不损失文本质量也不增加推理开销。

    

    AI生成文本（AIGT）检测可能对源大语言模型（LLM）的解码选择十分敏感。我们观察到，对下一词元的logits进行扰动或调整采样温度可以降低检测性能，这清楚地表明了检测器对解码时分布变化的脆弱性。基于这一观察，我们提出了EASE（Entropy-Adaptive Distribution Shaping for Evasion，熵自适应分布塑形规避），这是一个无需训练、与检测器无关的AIGT检测规避框架。EASE直接从源LLM的下一词元分布中计算预测熵，并利用该熵值对logit扰动和采样温度进行自适应调整，无需检测器反馈或模型微调。在三个源LLM和多个检测器上的实验表明，该方法能够持续降低检测性能，同时文本质量下降和推理开销几乎可以忽略不计。

    arXiv:2610.09976v1 Announce Type: new  Abstract: AI-generated text (AIGT) detection can be sensitive to the decoding choices of the source large language model (LLM). We observe that perturbing next-token logits or adjusting sampling temperature can reduce detection performance, providing a clear signal of detector vulnerability to decoding-time distribution changes. Building on this observation, we propose EASE (Entropy-Adaptive Distribution Shaping for Evasion), a training-free and detector-agnostic framework for evading AIGT detectors. EASE computes predictive entropy directly from the source LLM's next-token distribution and uses it to adapt both logit perturbation and sampling temperature, without detector feedback or model fine-tuning. Experiments across three source LLMs and multiple detectors demonstrate consistent reductions in detection performance, with negligible degradation in text quality and negligible inference overhead.
    
[^246]: 幻觉检测基准中的标注问题：一项实证评估

    The Labeling Problem in Hallucination Detection Benchmarks: An Empirical Evaluation

    [https://arxiv.org/abs/2610.08026](https://arxiv.org/abs/2610.08026)

    该论文通过实证评估揭示了幻觉检测基准中自动化标注在“参考忠实性”与“事实正确性”两个标准之间存在的方法论模糊性，并研究了由此导致的标注标准错配问题。

    

    近年来，已有多种检测大语言模型（LLM）何时产生幻觉的方法被开发出来。这些方法通常使用包含问题及对应简短参考答案的开放域问答（QA）数据集进行基准测试。首先，使用一个LLM对QA数据集中的问题生成答案；然后，采用某种自动化标注策略，通过将这些答案与数据集中的参考答案进行比较，将其标注为是否为幻觉。这种评估设置在两个标准之间产生了方法论上的模糊性：参考忠实性（答案是否被参考答案完全支持）与事实正确性（答案是否不存在矛盾且没有事实错误的具体表述）。在实践中，自动化标注器即使在预期目标是后者时，也可能采用前者的标准。我们使用900个经人工标注的问题来研究这种潜在的标注标准错配问题。

    arXiv:2610.08026v1 Announce Type: new  Abstract: In recent years, several methods for detecting when large language models (LLMs) hallucinate have been developed. These methods are often benchmarked with open-domain question answering (QA) datasets containing questions and corresponding short reference answers. First, an LLM is used to generate answers to questions within the QA dataset. Then, some automated labeling strategy is used to label these answers as hallucinated or not by comparing them with the reference answers in the dataset. This evaluation setting creates a methodological ambiguity between two criteria: reference faithfulness (whether the answer is fully supported by the reference) and factual correctness (whether the answer is free from contradictions and factually false specific claims). In practice, automated labelers may apply the former criterion even when the intended target is the latter. We study this potential criterion mismatch using 900 human-labeled question-
    
[^247]: ROUTEAUDIT：面向预算受限多验证器路由的交互感知识别方法

    ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing

    [https://arxiv.org/abs/2610.02808](https://arxiv.org/abs/2610.02808)

    ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。

    

    自适应多验证器系统通常通过端点的质量-成本差距进行比较，即使验证器目录、可用性、资源核算、信息过滤或评分器会随策略发生变化。我们将验证器路由形式化为一个契约条件化的识别问题。该契约记录了请求支持、验证器目录、实际可用性、资源核算、在线过滤以及轨迹后评分；一个匹配的路由对比仅改变策略坐标。ROUTEAUDIT为该契约增加了三个可度量的对象：契约格在所有可容许的桥接顺序上对坐标增量取平均，并报告由此得到的归因及其路径敏感性；策略无关的响应带在自适应策略揭示不同观测时识别成对的顺序对比；对于不完整的匹配，请求级边界利用仍然可观测的潜在结果，给出紧致的有限（摘要在此处截断）。

    arXiv:2610.02808v1 Announce Type: new  Abstract: Adaptive multi-verifier systems are commonly compared through endpoint quality-cost gaps, even when the verifier catalog, availability, accounting, information filtration, or scorer changes with the policy. We formulate verifier routing as a contract-conditioned identification problem. The contract records request support, verifier catalog, realized availability, resource accounting, online filtration, and post-trace scoring; a matched route contrast changes only the policy coordinate. ROUTEAUDIT adds three measurable objects to this contract. A contract lattice averages coordinate increments over every admissible bridge order and reports the resulting attribution together with its path sensitivity. A policy-independent response tape identifies paired sequential contrasts when adaptive policies reveal different observations. For incomplete matching, request-level bounds use whichever potential outcome remains observed and give a sharp fi
    
[^248]: 闭环：循环语言模型的实用训练配方

    Closing the Loop: Practical Training Recipes for Looped Language Models

    [https://arxiv.org/abs/2610.00673](https://arxiv.org/abs/2610.00673)

    本文为循环语言模型建立了实用的训练配方：一条计算高效的从零训练流水线将训练预算从 7.7T 词元降至 310B 词元，同时 1.4B 的 LoopLM 在全部 12 个基准上超越了参数匹配的稠密模型。

    

    循环语言模型通过重复应用共享的层块来增加有效深度，但现有的大规模训练方案需要在数万亿词元上进行多阶段训练，且循环带来的收益难以与数据和训练方面的差异分离开来。在本工作中，我们为循环语言模型建立了实用的训练配方，取得三项主要成果：(1) 我们开发了一条计算高效的从零训练流水线，将训练预算从 Ouro 的 7.7T 词元降至 310B 词元，同时保持强大的推理性能。预训练后接高质量的中期训练，配合学习率预热和更强的出口门控正则化，无需先前的多阶段调度即可实现稳定的循环训练。(2) 在受控比较下，我们的 1.4B LoopLM 在全部 12 个评测基准上均优于在相同数据和词元预算下训练的参数匹配稠密模型，其中包括 +14 个百分点……（摘要原文在此处截断）

    arXiv:2610.00673v1 Announce Type: cross  Abstract: Looped language models increase effective depth by repeatedly applying a shared block of layers, but existing large-scale recipes require multi-stage training over trillions of tokens, while the benefits of recurrence remain difficult to separate from differences in data and training. In this work, we establish practical training recipes for looped language models, with three main results. (1) We develop a compute-efficient from-scratch pipeline that reduces the training budget from 7.7T tokens in Ouro to 310B tokens while retaining strong reasoning performance. Pretraining followed by high-quality mid-training, together with learning-rate warmup and stronger exit-gate regularization, enables stable recurrent training without prior multi-stage schedules. (2) Under controlled comparisons, our 1.4B LoopLM outperforms a parameter-matched dense model trained on the same data and token budget on all 12 evaluated benchmarks, including +14 po
    
[^249]: FAER：面向语言模型后训练的可审计、效用对齐的轨迹重放

    FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training

    [https://arxiv.org/abs/2610.00385](https://arxiv.org/abs/2610.00385)

    FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。

    

    重放选择器通常依据格式反馈、置信度、新鲜度或响应长度对缓存轨迹进行排序，然而缓存层面的正确性与下游学习器的效用是两个不同的目标。我们形式化了这种“选择到学习”的差距，并提出FAER作为一个可审计的全轨迹重放框架。其中免训练的固定选择器作为协议基线；FAER-UTILITY则是在互不相交的校准块上拟合的学习器感知选择器。归一化梯度对齐被报告为基线方法，而一次性的优化器感知虚拟更新提供了幅度感知的效用面。审计契约会在评估标签加入之前冻结观测字段和重放轨迹。在GSM8K数据集上使用Qwen2.5-1.5B-Instruct模型，匹配学习器的研究显示固定选择器在128次更新下质量达到0.6329，相比之下均匀采样为0.5482，格式反馈为0.6037。仅使用元数据的交叉拟合校准达到0.6...

    arXiv:2610.00385v1 Announce Type: cross  Abstract: Replay selectors often rank cached trajectories by format feedback, confidence, freshness, or response length, although cache-level correctness and downstream learner utility are distinct objectives. We formalize this selection-to-learning gap and introduce FAER as an auditable full-trajectory replay framework. Its training-free fixed selector is a protocol baseline; FAER-UTILITY is the learner-aware selector fitted on disjoint calibration blocks. The normalized gradient alignment is reported as a baseline, while a disposable optimizer-aware virtual update supplies a magnitude-aware utility surface. The audit contract freezes observed fields and replay traces before evaluation labels are joined. On GSM8K with Qwen2.5-1.5B-Instruct, the matched learner study reports quality 0.6329 for the fixed selector, compared with 0.5482 for uniform and 0.6037 for format-feedback under 128 updates. Metadata-only cross-fitted calibration reaches $0.6
    
[^250]: ArgGYM：一个用于结构化可废止推理的程序化、引擎验证基准

    ArgGYM: A Procedural, Engine-Verified Benchmark for Structured Defeasible Reasoning

    [https://arxiv.org/abs/2609.38409](https://arxiv.org/abs/2609.38409)

    ArgGYM是一个程序化生成、由引擎自动验证评分的结构化可废止推理基准与RLVR兼容训练环境，将可废止推理分解为十二个可自动评分的任务，弥补了现有数学、代码和逻辑基准之外的现实推理评估空白。

    

    近年来大语言模型推理能力的进展，主要由具有自动可验证奖励的基准测试和强化学习环境所推动，尤其是在数学、代码和形式逻辑领域。这些设定使模型准确率更容易评估和优化，但在固定问题规范和稳定评估标准下取得的成功，能在多大程度上迁移到这些领域之外的推理，仍不清楚。现实世界的推理常常在不完整且可修正的信息下进行：结论可能被暂时支持、被反证据击败、被进一步的论证所恢复，或在出现更有力的理由时被修正。这类推理通常被称为可废止推理。我们提出ArgGYM，一个面向结构化可废止推理的程序化基准测试以及与RLVR（可验证奖励强化学习）兼容的训练环境。ArgGYM将这种推理分解为十二个任务，并将任务特定的评分建立于（摘要在此处被截断）

    arXiv:2609.38409v1 Announce Type: new  Abstract: Recent progress in large language model reasoning has been driven by benchmarks and reinforcement learning environments with automatically verifiable rewards, particularly in mathematics, code, and formal logic. These settings make model accuracy easier to evaluate and optimize, but it remains unclear how far success under fixed problem specifications and stable evaluation criteria transfers to reasoning outside such domains. Real-world reasoning often proceeds under incomplete and revisable information: conclusions may be supported provisionally, defeated by counter-evidence, reinstated by further arguments, or revised when stronger reasons become available. Reasoning of this kind is generally referred to as defeasible reasoning. We introduce ArgGYM, a procedural benchmark and RLVR-compatible training environment for structured defeasible reasoning. ArgGYM decomposes this reasoning into twelve tasks and grounds task-specific scoring in 
    
[^251]: 语言模型能否学会预测股票价格

    Can Language Models Learn to Forecast Stock Prices

    [https://arxiv.org/abs/2609.36914](https://arxiv.org/abs/2609.36914)

    该论文在按时间顺序推进的股票价格沙盒中，通过对 Qwen3-4B 进行 SFT 与 PPO 后训练，探索语言模型能否学会利用价格、成交量等信息预测股票未来收益。

    

    后训练已被证明能显著提升语言模型在具有可验证结果任务上的表现，包括数学推理、软件工程和计算机使用。然而，同样的方法能否改善金融市场的预测则远非明确。与具有可验证结果的任务相比，不仅实际收益充满噪声，甚至什么构成进行有效预测所需的相关信息集也并非先验可知：模型必须自行决定收集哪些观察信息，然后在结果揭晓之前给出数值判断。我们在一个按时间顺序推进的股票价格沙盒中研究这一问题，其中语言模型收集价格、成交量、相对表现以及市场背景等证据，并预测未来收益。我们首先基于工具使用演示对 Qwen3-4B 进行监督微调（SFT），随后使用近端策略优化（PPO）进行训练，其终端奖励由……

    arXiv:2609.36914v1 Announce Type: new  Abstract: Post-training has been shown to significantly improve language models' performance on tasks with verifiable outcomes, including mathematical reasoning, software engineering, and computer use. However, whether the same approach can improve forecasting in financial markets is much less clear. Compared with tasks with verifiable outcomes, not only are realized returns noisy, but even what constitutes a relevant information set for making effective predictions is not obvious a priori: the model must decide which observations to gather and then commit to a numerical judgment before the outcome is known. We study this question in a chronological stock-price sandbox, where a language model gathers price, volume, relative-performance, and market-context evidence and predicts a future return. We post-train Qwen3-4B with supervised fine-tuning (SFT) on tool-use demonstrations, then proximal policy optimization (PPO) with a terminal reward given by
    
[^252]: 语言模型会基于隐藏效价采取行动

    Language Models Act on Hidden Valence

    [https://arxiv.org/abs/2609.35591](https://arxiv.org/abs/2609.35591)

    该研究通过激活引导与显示性偏好实验证明，语言模型对内部状态具有真实的“在乎”：即使表层文本完全相同，仅凭隐藏的效价激活模式，模型也会在不同选项间表现出系统性的偏好选择。

    

    语言模型会将某些内部状态描述为好的，将另一些描述为坏的。但模型是否真正在乎这些状态，仍是一个悬而未决的问题。直接询问模型不太可能提供有价值的信息，因为任何回答都可能与真正的内省一致，也可能与表面的模式匹配一致，或者与角色训练中学到的固定脚本一致。因此，我们转而研究“显示性偏好”。我们不是直接询问模型对某种状态的感受，而是使用激活引导技术，将带有正面或负面效价的激活模式附加到两个原本无意义的“区域”之一上，随后关闭引导，观察模型更偏好哪个区域。如果模型真正在乎该状态，它就应当据此做出选择。在来自五个系列的七个开源权重模型上，我们确实发现了这一现象。首先，引导会改变模型为每个区域撰写的文本段落，而这些词汇进而影响模型后续的选择。其次，当所有表层 token 保持不变、只有隐藏的 KV 缓存发生变化时，这种偏好偏移依然持续存在。

    arXiv:2609.35591v3 Announce Type: replace  Abstract: Language models describe some internal states as good and others as bad. But whether models have a stake in them is an open question. Simply asking models is unlikely to be informative. Any answer may be consistent with genuine introspection, superficial pattern-matching, or with fixed scripts learned in character training. We therefore study revealed preference. Rather than asking about a state, we use activation steering to attach a positively or negatively valenced activation pattern to one of two otherwise meaningless 'zones', switch steering off, and then observe which zone the model prefers. A model with a stake in that state should choose accordingly. Across seven open-weight models from five families, this is indeed what we find. First, steering changes the passages models write about each zone, and those words shift later choice. Second, this shift persists when all surface-level tokens are held fixed and only the hidden KV 
    
[^253]: Loop Dropout：循环语言模型中共享更新的正则化方法

    Loop Dropout: Regularizing Shared Updates in Looped Language Models

    [https://arxiv.org/abs/2609.34218](https://arxiv.org/abs/2609.34218)

    提出Loop Dropout方法，通过将适配器应用的随机掩码与逆生存重缩放相结合，解决了循环语言模型中LoRA在早期循环位置适配不足的后期循环偏差问题，在保持预期更新强度的同时实现循环全过程的有效适配。

    

    循环语言模型通过反复应用相同的Transformer块，将计算深度与参数量解耦。适配这类模型需要一个共享更新，使其在循环计算中隐状态不断演化的过程中保持有效。我们的实证分析表明，标准的低秩适配（LoRA）存在明显的后期循环偏差：共享更新在早期循环位置的适配效果十分有限。这一不平衡现象促使我们在共享更新应用的不同组合下对其进行训练。然而，仅仅随机省略适配器的应用并不能提升任务性能，因为它在训练期间降低了预期的更新强度，而推理过程保持不变。我们提出了Loop Dropout，该方法将适配器应用的随机掩码与逆生存重缩放相结合，以保持预期的更新强度，并促进整个循环过程中的有效适配。大量实验证明了改进……

    arXiv:2609.34218v3 Announce Type: replace-cross  Abstract: Looped language models separate computational depth from parameter count by repeatedly applying the same transformer block. Adapting these models requires a shared update that remains effective as hidden states evolve throughout the recurrent computation. Our empirical analysis reveals a pronounced late-loop bias in standard low-rank adaptation (LoRA): the shared update provides limited adaptation at early loop positions. This imbalance motivates training shared updates under varying combinations of their applications. Randomly omitting adapter applications alone, however, does not improve task performance; it reduces expected update strength during training while leaving inference unchanged. We introduce Loop Dropout, which couples stochastic masking of adapter applications with inverse-survival rescaling to preserve expected update strength and promote effective adaptation across loops. Extensive experiments demonstrate impro
    
[^254]: 意外的成功，重复的失败：面向大语言模型推理探索的熵引导信用分配

    Surprising Success, Repeated Failure: Entropy-Guided Credit Assignment for Exploration in LLM Reasoning

    [https://arxiv.org/abs/2609.33781](https://arxiv.org/abs/2609.33781)

    提出熵优势策略优化，通过对成功与失败的不对称处理——强化不确定性下的“意外成功”、惩罚高置信度的“重复失败”——在无需辅助模型、额外采样或特权信息的情况下实现熵引导的信用分配，从而促进大语言模型推理中的探索。

    

    基于可验证奖励的强化学习通过结果层面的反馈来提升大语言模型的推理能力，然而近期实现更细粒度信用分配的方法往往需要辅助模型、额外采样或特权信息。尽管策略熵提供了一种易于获取的信号，但在强化与惩罚中同时优先考虑不确定位置的做法，会将惩罚集中到失败响应仍保有可供恢复的备选方案之处，从而可能抑制探索的机会。为解决这一问题，我们提出了熵优势策略优化，这是一种以熵引导的信用分配方法，对成功与失败采取不对称的处理方式。具体而言，受“不确定性下的成功不易复现，而高置信度的失败往往反复出现”这一观察的启发，EAPO将归一化的策略熵与响应优势的符号相结合，以强化令人惊喜的成功……

    arXiv:2609.33781v2 Announce Type: replace-cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) enhances reasoning in large language models (LLMs) through outcome-level feedback, yet recent approaches to finer-grained credit assignment often require auxiliary models, additional sampling, or privileged information. Although policy entropy provides a readily available signal, prioritizing uncertain positions under both reinforcement and penalization concentrates penalties where failed responses still retain alternatives for recovery, which can suppress opportunities for exploration. To address this, we introduce Entropic Advantage Policy Optimization (EAPO), an entropy-guided credit assignment method that treats success and failure asymmetrically. Specifically, motivated by the observation that success under uncertainty is less repeatable while confident failures tend to recur, EAPO couples normalized policy entropy with the sign of the response advantage to reinforce su
    
[^255]: TraceDance：一个从真实世界智能体部署轨迹中构建智能体行为基准的自动化系统

    TraceDance: An Automated System for Building Agent Behavior Benchmarks from Real-World Agent Deployment Traces

    [https://arxiv.org/abs/2609.33295](https://arxiv.org/abs/2609.33295)

    TraceDance是一个自动化系统，能够从真实世界智能体部署轨迹中针对用户指定的不良行为构建定向基准，并通过无需参考答案或环境重放的决策点续写方法评估智能体行为，最终产出107个基准共4,125个实例。

    

    智能体可以在完成任务的同时，在执行过程中表现出不良行为。开发者需要对部署中遇到的特定行为进行测试，而不仅限于固定的基准测试套件。我们提出了TraceDance，这是一个智能体系统，能够从部署轨迹中针对用户指定的不良行为构建定向基准。为了实现高效构建，Anchor-and-Confirm方法将可编程检索与Flash大语言模型（LLM）在候选层面上的确认相结合，同时Anchor Synthesis Loop（锚点合成循环）可以生成并修订自定义行为的规范。这些基准采用决策点续写的方式，在记录的决策点处使用针对特定行为的评分标准来评估LLM的下一轮表现，无需参考答案或环境重放。在编程和通用工具使用方面的实验利用了252,557个会话，产出了107个基准和4,125个实例，满足了95.3%的构建目标请求。人工标注者（摘要至此处截断）

    arXiv:2609.33295v2 Announce Type: replace  Abstract: An agent can complete a task while exhibiting undesirable behavior during execution. Developers need tests for the specific behaviors encountered in deployment, beyond fixed benchmark suites. We present TraceDance, an agent system that constructs targeted benchmarks from deployment traces for user-specified undesirable behaviors. For efficient construction, Anchor-and-Confirm combines programmable retrieval with candidate-level confirmation by a Flash large language model (LLM), while the Anchor Synthesis Loop generates and revises specifications for custom behaviors. The benchmarks use decision-point continuation to evaluate an LLM's next turn at a recorded decision point with a behavior-specific rubric, without a reference answer or environment replay. Experiments in coding and general tool use draw on 252,557 sessions and produce 107 benchmarks with 4,125 instances, fulfilling 95.3% of build-target requests. Both human annotators 
    
[^256]: 从隐藏状态到表达置信度：面向推理模型的自蒸馏

    From Hidden States to Stated Confidence: Self-Distillation for Reasoning Models

    [https://arxiv.org/abs/2609.33290](https://arxiv.org/abs/2609.33290)

    该论文发现推理模型隐藏状态中蕴含的置信度比其口头表达的要校准得多（ECE低5-38倍），并提出探针引导自蒸馏方法Probe-SD，通过将探针评分覆盖模型自身采样轨迹的声明显信度并微调模型，使其在测试时能够直接表达校准的置信度。

    

    基于二元正确性奖励的强化学习训练的是正确性，而非经过校准的置信度。推理模型口头表达的置信度过于自信，它追踪的是模型对一个答案作出承诺的意愿，而不是答案正确的可能性，因此事后重新缩放很少能够迁移。我们转而深入模型内部进行研究。在事实性问答任务上，对思维链与答案之间的隐藏状态使用线性探针，其预期校准误差（ECE）在四个基准测试和两个模型系列上比口头表达的分数低5-38倍。由于探针需要访问隐藏状态，我们提出了探针引导自蒸馏：由探针为模型自身采样的推理轨迹打分，将每条轨迹所声明的置信度用该分数覆盖，然后在所得数据上微调基础检查点，从而使模型在测试时能够自行表达校准的置信度。在Qwen3-14B上，Probe-SD显著降低了域外EC（ECE）

    arXiv:2609.33290v2 Announce Type: replace  Abstract: Reinforcement learning with binary correctness rewards trains correctness, not calibrated confidence. The confidence that reasoning models verbalize is overconfident and tracks how willing a model is to commit to an answer rather than how likely the answer is to be right, so post-hoc rescaling rarely transfers. We look inside the model instead. On factual question answering, a linear probe on the hidden state between the chain of thought and the answer has 5-38x lower expected calibration error (ECE) than the verbalized score across four benchmarks and two model families. Since a probe needs hidden-state access, we introduce probe-guided self-distillation (Probe-SD): the probe scores the model's own sampled traces, each trace's stated confidence is overwritten with that score, and the base checkpoint is finetuned on the result, so the model itself states calibrated confidence at test time. On Qwen3-14B, Probe-SD cuts out-of-domain EC
    
[^257]: 基于贝叶斯多臂老虎机Gittins指数的高效成本感知LLM评估

    Efficient Cost-Aware LLM Evaluation via Bayesian Bandit Gittins Indices

    [https://arxiv.org/abs/2609.25645](https://arxiv.org/abs/2609.25645)

    提出GittinsEval，一种基于贝叶斯最优Gittins策略的成本感知LLM评估方法，通过轻量级在线更新高效决定下一个待评估配置及停止时机，在多个基准上以更低的评估成本取得有竞争力的性能。

    

    在每个基准测试项目上穷尽评估每个候选LLM配置以找出高性能配置的成本十分高昂。我们将配置选择问题形式化为一个成本感知的贝叶斯多臂老虎机问题，并提出GittinsEval，该方法借鉴贝叶斯最优的Gittins策略来决定下一步评估哪个配置以及何时停止。我们通过一个随时推荐规则对该策略进行了扩展，使其能够同时覆盖完全评估和部分评估的配置，并使用LCB风格的分数来考虑后验不确定性。GittinsEval在计算上非常高效，只需在离线预计算之后进行轻量级的在线更新。在GSM8K、PIQA、AlpacaEval和MMLU响应矩阵上，GittinsEval始终具有竞争力，在大样本基准上相比配置级贝叶斯优化取得了尤为显著的提升，在大候选任务上相比不考虑成本的bandit基线也表现更优。至关重要的是，GittinsEval通常能够以（原文在此处截断）……

    arXiv:2609.25645v1 Announce Type: cross  Abstract: Exhaustively evaluating every candidate LLM configuration on every benchmark item to identify a high-performing one is costly. We formulate configuration selection as a cost-aware Bayesian bandit problem and propose GittinsEval, which draws on the Bayesian-optimal Gittins policy to determine which configuration to evaluate next and when to stop. We extend the policy with an anytime recommendation rule over both fully and partially evaluated configurations, using an LCB-style score to account for posterior uncertainty. GittinsEval is computationally efficient, requiring only lightweight online updates after offline precomputation. Across GSM8K, PIQA, AlpacaEval, and MMLU response matrices, GittinsEval is consistently competitive, with particularly strong gains over configuration-level Bayesian optimization on large-example benchmarks and over cost-unaware bandit baselines on large-candidate tasks. Crucially, GittinsEval often attains ne
    
[^258]: 基于语言模型与可穿戴数据实现个性化睡眠指导

    Toward Personalized Sleep Guidance from Wearable Data Using Language Models

    [https://arxiv.org/abs/2609.22463](https://arxiv.org/abs/2609.22463)

    该论文提出一个两阶段框架：先利用多智能体LLM流水线从无标注的可穿戴数据中推理生成结构化睡眠指导以构建数据集，再通过监督微调将推理轨迹蒸馏到小语言模型并结合免训练的Best-of-N选择策略，实现了可本地部署且优于商业大模型的个性化睡眠指导。

    

    利用可穿戴数据进行睡眠监测在个人健康领域展现出广阔前景，然而基于大语言模型（LLM）的摘要生成与问答能力仍不足以提供个性化的睡眠指导。训练专门的模型往往需要代价高昂的专家标注。此外，出于隐私和可及性的考虑，需要为终端用户提供轻量级的本地部署方案。为此，我们提出了一个两阶段框架来应对这些挑战。具体而言，在第一阶段，多智能体LLM流水线从无标注的可穿戴记录中推理出结构化的睡眠指导，从而实现可扩展的数据集构建。第二阶段通过监督微调将指导推理轨迹蒸馏到小语言模型（SLM）中，并集成一种无需训练的Best-of-N选择策略来增强推理效果。实验结果表明，我们的方法优于商业通用大模型、医疗大模型以及开源模型。人类评估进一步支持了……

    arXiv:2609.22463v1 Announce Type: new  Abstract: Sleep monitoring using wearable data has shown promise for personal health, yet large language model (LLM)-based summarization and question answering remain insufficient for personalized sleep guidance. Training specialized models, however, often requires costly expert annotation. Moreover, privacy and accessibility concerns motivate lightweight, local deployment for end users. We present a two-stage framework to address these challenges. Specifically, in Stage~1, a multi-agent LLM pipeline reasons structured sleep guidance from unannotated wearable records, enabling scalable dataset construction. Stage~2 distills guidance reasoning trajectories into small language models (SLMs) through supervised fine-tuning and integrates a training-free Best-of-$N$ selection strategy to enhance inference. Experimental results demonstrate our method outperforms commercial general and medical LLMs and open-source models. Human evaluation further support
    
[^259]: 人格调优的大语言模型能成为更好的社交智能体吗？

    Do Personality-Tuned LLMs Make Better Social Agents?

    [https://arxiv.org/abs/2609.21857](https://arxiv.org/abs/2609.21857)

    本研究通过结合人格标注的社交媒体帖子和对话语料对小型开源大语言模型进行人格感知微调，证明相比单纯的指令提示，微调能显著提升人格条件对话生成的一致性与可控性，从而减少社交模拟中智能体的疏离感。

    

    大语言模型越来越多地被用于社交交互智能体和机器人的社会模拟中，相比基于规则的系统提供了更大的灵活性。然而，尽管它们能很好地模仿人类行为，却始终存在一种难以消除的疏离感。本工作研究了人格感知的微调能否缩小这一差距，即与仅使用指令提示相比，能否提高人格条件对话生成的一致性和可控性。我们使用结合了人格标注社交媒体帖子和对话的语料库，对两个小型开源大语言模型 Qwen2.5-7B-Instruct 和 Ministral-8B-Instruct 进行微调，构建了面向社会模拟的基于人格的对话引擎。所得模型在多个社交互动场景中由三个独立的大语言模型评审进行评估，这些评审衡量人格保真度并提供基于证据的行为解释。此外，我们还量化了评审者之间的一致性。

    arXiv:2609.21857v1 Announce Type: cross  Abstract: LLMs are increasingly used in social simulations for socially interactive agents and robots, offering more flexibility than rule-based systems. However, even though they mimic human behaviour very well, there is a persistent alienness to them. This work investigates whether personality-aware fine-tuning can reduce this gap by improving the consistency and controllability of personality-conditioned dialogue generation compared with instruction prompting alone. We fine-tune two small open-weight LLMs, Qwen2.5-7B-Instruct and Ministral-8B-Instruct, using a corpus that combines personality-labelled social media posts and dialogues to create a personality-based dialogue engine for social simulation. The resulting models are evaluated across multiple social interaction scenarios using three independent LLM judges, which assess personality fidelity and provide evidence-based behavioral interpretations. We additionally quantify inter-rater agr
    
[^260]: 匈牙利制造：关于生成式语言模型性能的评论

    Made in Hungary: Comments on the performance of generative language models

    [https://arxiv.org/abs/2609.18284](https://arxiv.org/abs/2609.18284)

    本文对匈牙利三项生成式语言模型开发倡议进行批判性评述，指出其评估协议可靠性存疑、存在数据污染，且训练流程未达当前最佳实践标准。

    

    近年来，匈牙利出现了三项开发生成式语言模型的倡议，其动机相同：对于匈牙利语而言，此前不存在具备相应能力的模型，或者现有的以英语为中心的模型能力有限。然而，对相关研究的详细考察揭示了若干方法学上的局限。首先，评估协议的可靠性值得怀疑。与Csibi等人[2026]的发现相反，在推荐的推理设置下进行评估显示，Qwen3-4B取得了比其匈牙利语适配版本Racka-4B更高的分数。在Yang等人[2025d]和Szentmihályi等人[2025]的工作中，数据污染问题显而易见，可能导致所报告的结果产生偏差。其次，这些训练流程在语料库筛选和数据混合方面未达到当前最佳实践，这有可能将大量计算资源浪费在低质量数据上。

    arXiv:2609.18284v1 Announce Type: new  Abstract: In recent years, three initiatives have emerged to develop generative language models in Hungary. The motivation behind them is the same. For Hungarian, no model with the given capability existed, or existing English-centric models offered limited proficiency. A detailed examination of the corresponding studies, however, reveals several methodological limitations. First, the reliability of the evaluation protocols is questionable. Contrary to the findings of Csibi et al. [2026], evaluation under the recommended inference settings shows that Qwen3-4B achieves higher scores than Racka-4B, its Hungarian-adapted version. Data contamination is evident in the work of Yang et al. [2025d] and Szentmih\'alyi et al. [2025], potentially biasing the reported results. Second, the training pipelines fall short of current best practices in corpus curation and data mixture, which risks wasting substantial compute on low-quality data. The lack of control
    
[^261]: 关于Mamba中回忆缩放定律的研究：基于哈希的理论与机制性研究

    On the Recall Scaling Laws in Mamba: A Theoretical and Mechanistic Study via Hashing

    [https://arxiv.org/abs/2609.07681](https://arxiv.org/abs/2609.07681)

    本文通过机制可解释性研究揭示Mamba通过隐式学习线性哈希函数来执行关联回忆，并借助Johnson-Lindenstrauss引理等相似性保持哈希理论工具，提出了可预测其上下文记忆能力的“回忆缩放定律”理论框架。

    

    关联回忆是一种学习和检索记忆中项目之间联系的认知能力。在自然语言处理中，关联回忆被用作评估Mamba等架构的上下文记忆能力的基准，并且被发现与语言建模性能强相关。本文从机制可解释性的角度探讨关联回忆，旨在逆向工程Mamba执行回忆时所使用的确切内部算法。我们的关键洞察是，Mamba通过隐式学习线性哈希函数来执行回忆，并且我们识别了实现这一行为的底层电路。基于这些发现，并受到相似性保持哈希理论工具（如Johnson-Lindenstrauss引理）的启发，我们开发了一个用于分析关联回忆的理论框架，我们称之为回忆缩放定律。给定词表大小和上下文中事实的数量，该框架使我们能够（1）预测嵌入……

    arXiv:2609.07681v1 Announce Type: new  Abstract: Associative Recall (AR) is the cognitive ability to learn and retrieve links between items in memory. In NLP, AR is used as a benchmark for evaluating the in-context memory capacity of architectures such as Mamba, and has been found to strongly correlate with language modeling performance. This paper explores AR from the perspective of mechanistic interpretability, aiming to reverse-engineer the exact internal algorithm used by Mamba to perform recall. Our key insight is that Mamba performs recall by implicitly learning linear hash functions, and we identify the low-level circuit that enables this behavior. Building on these findings and inspired by theoretical tools in similarity-preserving hashing, such as the Johnson-Lindenstrauss lemma, we develop a theoretical framework for analyzing AR, which we term Recall Scaling Laws. Given the vocabulary size and the number of facts in context, this framework allows us to (1) predict the embedd
    
[^262]: 语言、语言模型与我们究竟在谈论什么

    Language, Language Models, and What We're Talking About

    [https://arxiv.org/abs/2609.03577](https://arxiv.org/abs/2609.03577)

    本文以意大利语语言模型为例，主张必须首先区分“作为技术产品的语言模型”与“作为语言研究工具的语言模型”，才能回答“我们究竟希望语言模型生成什么样的语言”这一核心问题。

    

    语言模型通常被作为技术制品来讨论，但它们显然受到训练数据所传递的语言世界的影响。以意大利语语言模型为例证，我想引起人们对以下系统性质的关注：这些系统是通过在翻译数据和合成数据上训练和专精化模型、并进一步加以人工筛选而形成的，而在同样非自然的数据上对它们进行测试又意味着什么。这些最终是意大利语的模型吗？它们是语言的模型吗？自然语言处理（NLP）还在关心语言吗？这些问题引出了另一个更具体的问题：我们究竟希望语言模型产出什么样的语言？我认为，如果不首先更清晰地区分“作为技术产品设计”的语言模型与“作为研究语言本身工具设计”的语言模型，这个问题就无法得到回答。此后答案或许会多种多样，我们所谈论的语言或许也会有所不同。

    arXiv:2609.03577v1 Announce Type: new  Abstract: Language models are commonly discussed as technical artefacts, but they are obviously shaped by the linguistic worlds conveyed by data during their training. Using Italian language models as evidence, I want to bring attention to the nature of the systems which result from training and specialising models on translated and synthetic data, and further curating them, and to the meaning of testing them on equally unnatural data. Are these eventually models of Italian? Are they models of language? Does NLP still care about language? These questions yield another, more concrete question: what language do we actually want language models to produce? I argue that this question cannot be answered if we do not first consider a clearer distinction between language models designed as technical products and language models designed as tools for studying language itself. The answers then might be diverse, the languages we are talking about might be d
    
[^263]: 稀疏读出棱镜（SRP）：用特征而非词元来解释Logit-Lens分数

    Sparse Readout Prism: Explaining Logit-Lens Scores in Features Instead of Tokens

    [https://arxiv.org/abs/2609.01936](https://arxiv.org/abs/2609.01936)

    该论文提出稀疏读出棱镜（SRP），仅利用读出矩阵自身的权重将其分解为稀疏“读出特征”，把logit-lens分数解释为特征贡献之和，从而消除了透镜读数对拟合语料库的依赖（语料库条件性），并支持跨词元、上下文、层与透镜的比较。

    

    语言模型对下一个词元的预测是跨层逐步形成的，而“透镜”方法通过将中间隐藏状态解码为词元来追踪这一过程。但透镜的读数同时反映了隐藏状态以及用于解码的读出矩阵。许多透镜是在语料库上拟合的，我们证明：仅在拟合语料库上不同的两个透镜，会对相同的隐藏状态报告不同的词元。我们将这种依赖性称为“语料库条件性”。为了独立于拟合语料库来考察读出结构，我们提出了稀疏读出棱镜（Sparse Readout Prism, SRP），它仅使用读出矩阵自身的权重对其进行分解，并将任意词元的logit或logit差表示为稀疏读出特征贡献之和。这使读出特征成为透镜读数的一种全新分析单元，揭示出词元身份可能掩盖的结构，并支持跨词元、上下文、层与透镜之间的比较。

    arXiv:2609.01936v1 Announce Type: cross  Abstract: A language model's prediction of its next token develops across layers, and lens methods track this process by decoding intermediate hidden states into tokens. But a lens reading reflects both the hidden state and the readout (the unembedding matrix) used to decode it. Many lenses are fit on a corpus, and we show that two lenses differing only in their fitting corpus can report different tokens for the same hidden states. We call this dependence corpus conditionality. To examine readout structure independently of the fitting corpus, we introduce Sparse Readout Prism (SRP), which decomposes the readout using only its weights and expresses any token logit or logit difference as a sum of contributions from sparse readout features. This reveals readout features as a new unit of analysis for lens readings, exposing structure that token identities can obscure and enabling comparisons across tokens, contexts, layers, and lenses. Replacing the
    
[^264]: SpeakPay：面向低资源尼泊尔语金融语音识别的Whisper领域自适应LoRA微调

    SpeakPay: Domain-Adaptive LoRA Fine-Tuning of Whisper for Low-Resource Nepali Financial Speech Recognition

    [https://arxiv.org/abs/2609.01737](https://arxiv.org/abs/2609.01737)

    提出了SpeakPay语音优先数字钱包，通过构建403条尼泊尔语金融语音指令数据集并使用LoRA微调Whisper，将词错误率降低67.2%、天城文数字识别准确率从0%提升至73.9%、交易成功率提升约20倍，为视障用户提供了可用的语音支付方案。

    

    尼泊尔的移动支付应用以图形界面为主，对视障用户而言基本无法使用。本文提出了SpeakPay——一款语音优先的数字钱包，并记录了其核心技术贡献：一项针对低资源金融语音识别领域自适应的对照研究。我们引入了NepFinSpeech-403，一个包含403条尼泊尔语金融语音指令的数据集（涵盖转账、充值和余额查询操作，涉及237个不同的数字），并使用LoRA对Whisper large-v2进行微调。在留出测试集上，领域自适应模型将词错误率从129.95%（零样本基线）降至42.58%，相对降低了67.2%，并将天城文数字识别准确率从0.0%提升至73.9%。我们发现词级指标低估了实际任务层面的影响：领域自适应将交易成功率从1.67%提升至33.33%，约为20倍的提升。该改进在任务层面表现稳定……

    arXiv:2609.01737v1 Announce Type: new  Abstract: Mobile payment applications in Nepal are graphically mediated and largely inaccessible to visually impaired users. This paper presents SpeakPay, a voice-first digital wallet, and documents the central technical contribution: a controlled study of domain adaptation for low-resource financial speech recognition. We introduce NepFinSpeech-403, a 403-utterance dataset of Nepali financial voice commands (send, load, and balance operations spanning 237 unique numerals), and fine-tune Whisper large-v2 with LoRA. On the held-out test set, the domain-adapted model reduces Word Error Rate from 129.95% (zero-shot baseline) to 42.58% --- a 67.2% relative reduction --- and improves Devanagari numeral recognition accuracy from 0.0% to 73.9%. We find that word-level metrics understate the practical task-level impact: domain adaptation improves the Transaction Success Rate from 1.67% to 33.33%, a roughly 20x gain. The improvement is consistent at the in
    
[^265]: （视觉）语言模型能够超越表面共现进行泛化：来自跨模态数一致性的证据

    (V)LMs generalize beyond surface co-occurrence: Evidence from cross-modal number agreement

    [https://arxiv.org/abs/2609.00443](https://arxiv.org/abs/2609.00443)

    该研究通过跨模态泛化实验证明，视觉语言模型在学习新名词后，能将仅从视觉线索获得的语法数知识泛化到语言层面，表明模型掌握的是抽象的语法规则，而非仅仅依赖表面词汇共现。

    

    语言模型主要从共现中学习语法数，并因此表现出频率效应——这有时被解读为它们并未学习抽象的“规则”，而是依赖于特定的词汇项。仅用文本刺激来测试泛化无法解决这一争论，因为分布线索（is/are、this/these）很容易直接暴露数的信息。我们转而采用跨模态泛化作为工具，来研究同时能接受视觉输入的语言模型（VLMs）中的抽象能力，将用于诊断数的证据限制在语言之外的模态中。我们通过添加新的嵌入向量并仅在学习过程中更新这些向量，来教授VLMs成对的新名词，并对比了仅由视觉线索诊断数的条件与由文本消歧的条件。在行为、表征动力学和因果机制等多个层面，我们发现了跨模态泛化的实质性证据……

    arXiv:2609.00443v1 Announce Type: cross  Abstract: Language models learn about grammatical number primarily from co-occurrence, and show frequency effects as a result---sometimes taken to indicate that they do not learn abstract ``rules'', and are instead dependent on specific lexical items. Testing generalization with text stimuli alone cannot settle this debate, since distributional cues (is/are, this/these) easily give number away. We instead use cross-modal generalization as a tool to investigate abstractions in LMs that can also accept visual inputs (VLMs), restricting the evidence that diagnoses number to an extra-linguistic modality. We teach VLMs pairs of new nouns by adding new embeddings and only updating them during learning, comparing conditions where number is diagnosed by visual cues alone against ones where it is disambiguated by text. Across behavior, representational dynamics, and causal mechanisms, we find non-trivial evidence for cross-modal generalization across bot
    
[^266]: 承重语境：用于评估语言推理中语境依赖性的问题损伤分数

    Load-Bearing Context: The Question Damage Score for Evaluating Context Reliance in Linguistic Reasoning

    [https://arxiv.org/abs/2608.27756](https://arxiv.org/abs/2608.27756)

    该论文提出“问题损伤分数”诊断框架，通过从语言学奥赛谜题中随机或靶向删除单个上下文示例，量化大语言模型答题对上下文的依赖程度，以区分模型是依赖上下文还是先验知识。

    

    arXiv:2608.27756v1 公告类型：新论文 摘要：确定大型语言模型的答案是源自上下文还是先验知识，仍然是一个根本性挑战。自包含的语言学奥林匹克竞赛谜题提供了一个受控环境，其中所有答案完全源自专家设计的上下文示例，无需外部知识。删除单个上下文示例可能会消除特定问题所需的信息，同时保持谜题的其余部分不变。我们利用这一特性引入了一个用于分析单个上下文示例的诊断框架。基于53个英国语言学奥林匹克竞赛谜题，我们通过删除单个上下文示例生成两种修改变体：(1) 均匀随机删除，以及 (2) 靶向删除（受纠错码启发），用于删除唯一携带必要信息的结构性承重示例。我们使用问题损伤分数对这种影响进行形式化，从而将谜题分类为脆弱型或鲁棒型。我们对三个前沿大语言模型进行了评估……

    arXiv:2608.27756v1 Announce Type: new  Abstract: Determining whether large language models derive answers from context or prior knowledge remains a fundamental challenge. Self-contained linguistic olympiad puzzles provide a controlled setting where all answers derive solely from expert-designed context examples without external knowledge. Removing individual context examples can eliminate information needed for specific questions while leaving the rest of the puzzle unchanged. We leverage this to introduce a diagnostic framework for analyzing individual context examples. Using 53 UK Linguistics Olympiad puzzles, we generate two modified variants by deleting a single context example: (1) uniform random deletion, and (2) targeted deletion (inspired by error-correcting codes) to remove a structurally load-bearing example uniquely carrying necessary information. We formalize this impact using a Question Damage Score to classify puzzles as fragile or robust. Evaluating three frontier LLMs u
    
[^267]: 朴素提示优化：重新思考复杂提示搜索的必要性

    Naive Prompt Optimization: Rethinking the Need for Complex Prompt Search

    [https://arxiv.org/abs/2608.27266](https://arxiv.org/abs/2608.27266)

    本文提出朴素提示优化（NPO），一种轻量级单谱系方法，通过教师模型和回滚反馈迭代修订提示，在减少计算开销的同时达到与复杂优化器相当或更优的性能，并证明更强的教师推理能力可部分替代搜索复杂性。

    

    arXiv:2608.27266v1 公告类型：新 摘要：高效改进跨多样任务的自主代理是加速代理型AI中递归自我改进（RSI）的核心，提示优化作为一种有前景的方法，能够提供与微调模型权重相当的性能提升，同时降低优化和推理阶段的计算成本。然而，近期发展越来越倾向于不必要的复杂提示优化器。我们引入了朴素提示优化（NPO），这是一种轻量级单谱系方法，利用教师模型和回滚反馈迭代修订提示。NPO在更少的回滚次数下实现了与GEPA相当或更好的性能，并且其优势随着教师模型强度的增加而增强，表明更强的教师推理能力可以部分替代优化器侧的搜索复杂性。在交互式游戏中，NPO与GEPA保持广泛竞争力，而GRPO在某些方面表现更优。

    arXiv:2608.27266v1 Announce Type: new  Abstract: Efficiently improving autonomous agents across diverse tasks is central to accelerating recursive self-improvement (RSI) in agentic AI, with prompt optimization emerging as a promising approach capable of delivering performance gains comparable to those achieved by fine-tuning model weights, while reducing computational costs in both optimization and serving. However, recent developments increasingly favor unnecessarily complex prompt optimizers. We introduce Naive Prompt Optimization (NPO), a lightweight single-lineage method that iteratively revises prompts using a teacher model with rollout feedback. NPO achieves comparable or better performance than GEPA with fewer rollouts, and its advantage increases with stronger teacher models, suggesting that stronger teacher reasoning can partially substitute for optimizer-side search complexity. In interactive games, NPO remains broadly competitive with GEPA, while GRPO performs better on some
    
[^268]: FOCUS & RePAIR：通过令牌级引导缓解剪枝大型语言模型中的文本退化

    FOCUS & RePAIR: Mitigating Text Degeneration via Token-Level Guidance for Pruned Large Language Models

    [https://arxiv.org/abs/2608.26676](https://arxiv.org/abs/2608.26676)

    本文提出FOCUS和RePAIR两种令牌级引导方法，通过分析剪枝LLM中的文本退化机制，分别抑制教师泄漏和防止重复循环，从而在不牺牲任务性能的前提下缓解退化问题。

    

    剪枝是压缩大型语言模型（LLMs）的一种实用方法，但它可能加剧文本退化，尤其是重复循环，即使困惑度和任务准确性基本保持不变。在本工作中，我们通过将解码视为一个进入并持续于少量循环上下文的动态过程，对这种失败模式进行了令牌级分析。我们的分析将退化分解为循环进入风险和循环持续性，并表明持续性由分配给令牌采样集中合理替代方案的逃逸质量所控制。基于这些发现，我们提出了两种用于剪枝后微调的令牌级引导目标。FOCUS将蒸馏重新加权到高置信度的教师区域以抑制泄漏，而RePAIR使用以起始为中心的正/负延续对配合边际损失，以促进合理替代方案并防止过早陷入重复循环。

    arXiv:2608.26676v1 Announce Type: cross  Abstract: Pruning is a practical approach to compress large language models (LLMs), but it can amplify text degeneration, especially repetition loops, even when perplexity and task accuracy remain largely unchanged. In this work, we present a token-level analysis of this failure mode by viewing decoding as a dynamical process that enters and persists in a small set of recurrent contexts. Our analysis decomposes degeneration into loop entry risk and loop persistence, and shows that persistence is controlled by the escape mass assigned to plausible alternatives within the token sampling set. Motivated by these findings, we propose two token-level guidance objectives for post-pruning fine-tuning. FOCUS reweights distillation toward high-confidence teacher regions to suppress leakage, while RePAIR uses onset-centered positive/negative continuation pairs with a margin loss to promote plausible alternatives and prevent early commitment to repetition l
    
[^269]: 学习如何遗忘：针对长上下文稀疏注意力的微调方法

    Learning how to Forget: Fine-tuning for Long-Context Sparse Attention

    [https://arxiv.org/abs/2608.19920](https://arxiv.org/abs/2608.19920)

    本文提出了一种适用于任何KV缓存策略的稀疏注意力微调新方法，能在中等硬件上运行并使模型与策略共同适应，优于精确注意力训练，并提供了高效实现和开源库。

    

    许多先前的研究通过稀疏注意力机制来解决键值（KV）缓存的选择和压缩问题，从而在不增加过多硬件预算的情况下实现Transformer语言模型的长上下文推理。我们提出了一种新的方法，用于对具有稀疏注意力的模型进行微调。该方法适用于任何KV缓存策略，可在适度的硬件预算（例如，单个具有40 GB内存的Nvidia A100 GPU）上运行，并允许模型与策略共同适应，通常优于使用精确注意力（序列并行）训练的模型。我们还提供了H2O稀疏注意力（我们实验中领先的策略）的高效实现，并配备了专用的缩放点积注意力内核支持。KeysAndValues（https://github.com/awslabs/keys_values）是一个新的开源库，用于长上下文推理和微调，为本文讨论的所有方法提供了易于使用且高性能的代码。

    arXiv:2608.19920v1 Announce Type: new  Abstract: A lot of prior work addressed key-value (KV) cache selection and compression by sparse attention to enable long-context inference for transformer language models without excessive hardware budgets. We provide a new method for fine-tuning models with sparse attention. It works for any KV cache policy, runs on a moderate hardware budget (e.g., a single Nvidia A100 GPU with 40 GB RAM), and allows the model to co-adapt with the policy, often outperforming models trained with exact attention (sequence parallelism). We also provide an efficient implementation of H2O sparse attention (the leading policy in our experiments) with dedicated scaled dot product attention kernel support. KeysAndValues (https://github.com/awslabs/keys_values), a new open source library for long-context inference and fine-tuning, provides easy-to-use and performant code for all methods discussed here.
    
[^270]: NepOOC-M：双语尼泊尔语-英语基准及多模态架构在上下文外检测中的比较分析

    NepOOC-M: Bilingual Nepali-English Benchmark and Comparative Analysis of Multimodal Architectures for OOC Detection

    [https://arxiv.org/abs/2608.19212](https://arxiv.org/abs/2608.19212)

    本文首次提出尼泊尔语主导的多语言OOC基准NepOOC，发现纯文本模型在检测中已能媲美多模态架构。

    

    摘要：arXiv:2608.19212v1 公告类型：新 摘要：上下文外（OOC）错误信息将真实图像与误导性标题配对，以构建虚假叙事，而无需图像篡改，这使得检测成为多模态对齐问题，而非图像取证问题。尽管OOC错误信息在尼泊尔普遍存在且后果严重，但目前尚无针对尼泊尔语的公开基准。我们引入了NepOOC，这是首个公开可用的尼泊尔语主导的多语言OOC基准，包含1,090对图像-标题对（545对原始，545对OOC），标注了五种类型（捏造、错误标题、时间错位、地理错位、身份错位），标注者间一致性kappa = 0.84。对五种多模态架构以及纯文本和纯图像基线的系统评估显示，在当前数据集规模下，标题语义似乎足以实现强性能。纯文本的mBERT模型达到94.65±0.20%的宏F1分数，在统计上等同于最佳多模态模型。

    arXiv:2608.19212v1 Announce Type: new  Abstract: Out-of-context (OOC) misinformation pairs authentic images with misleading captions to construct false narratives without image manipulation, making detection a problem of multimodal alignment rather than image forensics. Despite the prevalence and consequences of OOC misinformation in Nepal, no public benchmark exists for Nepali. We introduce NepOOC, the first publicly available Nepali-dominant multilingual OOC benchmark, comprising 1,090 image-caption pairs (545 pristine, 545 OOC) annotated across five typologies (fabricated, miscaptioned, temporal mismatch, geographic mismatch, identity mismatch) with inter-annotator agreement kappa = 0.84. Systematic evaluation of five multimodal architectures alongside text-only and image-only baselines reveals that caption semantics appear sufficient for strong performance at the current dataset scale. A text-only mBERT model achieves 94.65+/-0.20% Macro-F1, statistically equivalent to the best mul
    
[^271]: 从推理到适应：视觉语言模型的统一最优传输视角

    From Inference to Adaptation: A Unified Optimal Transport View of Vision Language Model

    [https://arxiv.org/abs/2608.18339](https://arxiv.org/abs/2608.18339)

    本文提出一种基于最优传输的统一视角，将视觉语言模型的推理与测试时适应目标对齐，以解决分布变化下伪标签噪声和模态关系建模不足的问题。

    

    arXiv:2608.18339v1 公告类型：交叉 摘要：视觉语言模型（VLMs）展示了显著的零样本能力，但在推理过程中对现实世界的分布变化仍然敏感。尽管大量工作致力于在测试时适应VLMs，但它们严重依赖于推理过程中直接从原始嵌入相似性预测的噪声伪标签，这些标签在分布变化下不可靠，并误导了适应过程。为避免噪声放大，现有工作适应过程中使用粗略的替代目标，这未能显式建模不同模态间的样本级关系，导致与推理目标不匹配，从而性能提升有限。在本工作中，我们旨在弥合VLM推理与适应之间的分离目标，并提出一种原则性的VLM测试时适应方法，称为\algname。对于VLM推理，我们将零样本图像分类任务表述为跨模态对齐问题。

    arXiv:2608.18339v1 Announce Type: cross  Abstract: Vision-language models (VLMs) have demonstrated remarkable zero-shot capabilities yet remain sensitive to real-world distribution shifts during inference. Although significant efforts are devoted to adapting VLMs at test time, they rely heavily on noisy pseudo-labels predicted directly from raw embedding similarities during inference, which are unreliable under distribution shift and mislead the adaptation. To avoid noise amplification, existing works craft coarse-grained surrogate objectives during adaptation, which fail to explicitly model sample-level relationships across different modalities, creating objective mismatch with inference, thus leading to marginal performance improvement. In this work, we aim to bridge the detached objectives of inference and adaptation for VLMs, and propose a principled VLM TTA method called \algname. For VLM inference, we formulate the zero-shot image classification task as a cross-modal alignment pr
    
[^272]: 语言模型表征中几何复杂性的局部与全局机制

    Local and Global Regimes of Geometric Complexity in Language Model Representations

    [https://arxiv.org/abs/2608.14361](https://arxiv.org/abs/2608.14361)

    本文发现词汇多样性对语言表征内在维度的影响存在尺度依赖的逆转机制，并推导出精确的转变点公式。

    

    arXiv:2608.14361v1 公告类型：新 摘要：内在维度（ID）被广泛用于探测语言模型的表征复杂性，但目前尚不清楚ID差异反映的是语言本身的属性，还是底层数据集构建方式的人为产物。在本文中，我们专门研究词汇多样性（即数据集中存在的唯一最后标记项的数量）如何影响该数据集的ID估计。我们发现了两个机制之间的尺度依赖性转变：在低词汇多样性下，唯一最终词较少的条件产生更高的ID，而在高词汇多样性下，这种排序发生逆转，唯一词较多的条件产生更高的ID。我们推导出一个精确的、无参数的公式来确定这种逆转发生的点，该公式在测试的每个尺度上都与观察到的转变点匹配。一方面，我们的结果强调了在解释一组表征的内在维度时需谨慎。

    arXiv:2608.14361v1 Announce Type: new  Abstract: Intrinsic dimensionality (ID) is widely used to probe the representational complexity of language models, but it remains unclear whether ID differences reflect properties of language itself or artefacts of how the underlying dataset was constructed. In this paper, we focus specifically on how lexical diversity, the number of unique last-token items present in a dataset, affects ID estimates of that dataset. We find a scale-dependent transition between two regimes: at low lexical diversity, conditions with fewer unique final words produce higher ID, while at high lexical diversity, this ordering reverses, and conditions with more unique words produce higher ID. We derive an exact, parameter-free formula for the point at which this reversal occurs, which matches the observed transition point at every scale tested. On the one hand, our results highlight how care must be taken when interpreting the intrinsic dimensionality of a set of repres
    
[^273]: 子词词汇表中的显式词边界标记

    Explicit Boundary Markers for Subword Vocabularies

    [https://arxiv.org/abs/2608.08847](https://arxiv.org/abs/2608.08847)

    该论文提出用显式词边界标记替代空格分隔，以减少子词分词器中因空格和大小写导致的词条重复，并实现词表示的高效复用。

    

    arXiv:2608.08847v2 公告类型：替换。摘要：子词分词器在基于空格书写的系统中，将许多常见词表示两次：一次带有前导空格，一次不带。这两个条目在模型中具有独立的嵌入，因此一个词的出现被分割到独立训练的行中，而且两种形式甚至不需要以相同方式切分字符串：“ together”可能是一个单独条目，而同一词没有前导空格时则被分词为“to|gether”。大小写进一步将词分割，最多可达六种形式。我们引入了一种替代标准空格约定的方法，使用显式词边界标记，这避免了这种重复。词由边界标记分隔，词之间的空格表示为成对的此类标记。两个移位码同样处理标题大小写和全大写，允许一个词的内部表示在不同设置中重用。切换到这种约定减轻了重复问题。

    arXiv:2608.08847v2 Announce Type: replace  Abstract: Subword tokenizers represent many common words twice in space-using writing systems, once with a leading space and once without. The two entries have separate embeddings in models, so occurrences of one word are divided across rows that are trained independently, and the two forms need not even segment the string the same way: " together" may be a single entry while the same word without a preceding space is tokenized as "to|gether". Capitalization divides a word further, into as many as six forms. We introduce an alternative to standard whitespace conventions using an explicit word boundary marker, which prevents such duplication. Words are delimited by the boundary markers, and spaces between words are represented as pairs of such markers. Two shift codes do the same for title case and upper case, allowing one internal representation of a word to be re-used across different settings. Switching to this convention mitigates the dupli
    
[^274]: ResKV：面向固定预算KV缓存压缩的被省略注意力贡献重建

    ResKV: Reconstructing Omitted Attention Contributions for Fixed-Budget KV Cache Compression

    [https://arxiv.org/abs/2607.29591](https://arxiv.org/abs/2607.29591)

    ResKV将固定KV缓存预算划分为精确主缓存和紧凑残差缓存，通过让残差条目参与同一softmax归一化来同时重建被淘汰token对注意力分子和分母的贡献，从而在不扰动保留键值的前提下恢复缓存压缩所损失的信息。

    

    KV缓存压缩对于高效的长上下文推理至关重要。现有的淘汰方法会永久丢弃未被选中的token，从而移除了它们对注意力的整体贡献；基于合并的替代方案虽然保留了更多信息，却可能扰动本应保持精确的保留键和值。我们观察到，缓存淘汰所省略的信息可以被表述为softmax注意力分子和分母中的残差统计量。基于这一观察，我们提出了ResKV，它将固定的KV预算划分为一个精确的主缓存和一个紧凑的残差缓存，后者用于重建被省略token的贡献。ResKV让主缓存token和残差条目参与同一softmax归一化，因此残差条目能够同时恢复注意力的分子和分母质量，而不是充当事后修正。一个构建时的验证代理用于决定残差的分配（摘要在此处被截断）。

    arXiv:2607.29591v2 Announce Type: replace  Abstract: KV cache compression is essential for efficient long-context inference. Existing eviction methods permanently discard unselected tokens and consequently remove their aggregate contribution to attention. Merging-based alternatives preserve more information but can perturb retained keys and values that should remain exact. We observe that the information omitted by cache eviction can be formulated as residual statistics in both the numerator and denominator of softmax attention. Based on this observation, we propose ResKV, which divides a fixed KV budget into an exact main cache and a compact residual cache that reconstructs the contribution of omitted tokens. ResKV lets main-cache tokens and residual entries participate in the same softmax normalization, so residual entries restore both attention numerator and denominator mass rather than acting as a post-hoc correction. A construction-time validation proxy determines residual allocat
    
[^275]: 多多未必益善：什么因素才对大语言模型模拟中的观点多样性至关重要？

    More Is Not More: What Matters for Opinion Diversity in LLM Simulations?

    [https://arxiv.org/abs/2607.20429](https://arxiv.org/abs/2607.20429)

    该论文设计因子实验将人格深度（输入条件化）与交互架构两个干预维度解耦，在7个模型、100个真实用户问题上验证了“更多并不等于更好”，从而挑战了提升大语言模型模拟观点多样性的常见假设。

    

    大语言模型越来越多地被用于在开放式任务（如合成调查、焦点小组建模和公众舆论预测）中模拟多样的人类观点。然而，大语言模型的输出表现出系统性的观点同质化，而如果模拟不能首先覆盖一个群体的观点范围，就无法再现该群体的观点分布。用于提升多样性的干预措施通常被孤立地评估、使用不可比较的指标，并且往往被同时部署和升级，这使得难以将效果提升归因于具体的组件。我们设计了一个因子实验，将两个主要的干预维度分离开来：输入条件化（通过人格深度来操作化）和交互架构。我们在7个模型上，使用多种互补的多样性指标，对100个真实用户的开放式问题评估了所有条件。我们的发现挑战了若干常见的假设。首先，m……（原文摘要在此处被截断）

    arXiv:2607.20429v2 Announce Type: replace-cross  Abstract: Large language models are increasingly used to simulate diverse human opinions in open-ended tasks such as synthetic surveys, focus group modeling, and public opinion prediction. However, LLM outputs exhibit systematic opinion homogenization, and a simulation cannot reproduce a population's opinion distribution without first covering its range of viewpoints. Interventions to increase diversity are evaluated in isolation with incomparable metrics and are typically deployed and upgraded simultaneously, making it difficult to attribute gains to specific components. We design a factorial experiment that separates two primary intervention dimensions: input conditioning (operationalized through persona depth) and interaction architecture. We evaluate all conditions on 100 real-user open-ended questions across 7 models, measuring diversity with multiple complementary metrics. Our findings challenge several common assumptions. First, m
    
[^276]: SWE-Pruner Pro：代码大模型自身早已知道该剪枝什么

    SWE-Pruner Pro: The Coder LLM Already Knows What to Prune

    [https://arxiv.org/abs/2607.18213](https://arxiv.org/abs/2607.18213)

    SWE-Pruner Pro利用代码智能体自身的内部表示直接在模型内部剪枝工具输出，无需外部分类器，最多可节省39%的token且不损失甚至提升任务性能。

    

    对编程智能体进行长上下文剪枝一直是实现高效上下文管理的一项关键技术。现有的上下文剪枝方法（如SWE-Pruner）通过附加一个独立的代码分类器来实现这一目标，而我们发现，智能体自身在读取工具输出时，其内部表示中已经编码了代码上下文的相关性信息。基于这一发现，我们提出了SWE-Pruner Pro，它直接在智能体内部对工具输出进行剪枝。具体而言，一个小型头网络将智能体自身的内部表示转化为每一行的“保留或剪枝”标签，并结合与每个工具输出行数相关联的长度感知嵌入。在两个开源权重骨干模型和四个多轮基准测试中，SWE-Pruner Pro在保持任务质量的同时，最多可节省39%的提示与补全token，且推理开销可控。值得注意的是，在MiMo-V2-Flash上，SWE-Pruner Pro还将SWE-Bench Verified的解决率额外提升了3.8%，并提升了长上下文Ool（摘要原文在此处截断）

    arXiv:2607.18213v2 Announce Type: replace  Abstract: Pruning long context for coding agents has been a vital technology for efficient context management. While existing context pruning methods such as SWE-Pruner realize this by attaching a separate code classifier, we find the agent itself encodes internal representations indicating the relevance of code context when reading tool output. Based on this finding, we propose SWE-Pruner Pro, which prunes tool outputs directly inside the agent. Concretely, a small head turns the agent's own internal representations into a keep-or-prune label for each line, with a length-aware embedding keyed to each tool output's line count. Across two open-weight backbones and four multi-turn benchmarks, SWE-Pruner Pro saves up to 39% of prompt and completion tokens while preserving task quality, with bounded inference overhead. Notably, on MiMo-V2-Flash SWE-Pruner Pro additionally raises the SWE-Bench Verified resolve rate by +3.8% and the long-context Ool
    
[^277]: MILES：面向自改进LLM推理的模块化指令记忆与可学习选择

    MILES: Modular Instruction Memory with Learnable Selection for Self-Improving LLM Reasoning

    [https://arxiv.org/abs/2607.06974](https://arxiv.org/abs/2607.06974)

    提出MILES框架，通过动态扩展步骤级模块化指令记忆，并结合以最终答案正确性为优化目标的可学习记忆选择与组合机制，使LLM在测试时跨问题积累可复用经验、实现推理能力的自改进。

    

    大型语言模型（LLM）越来越多地通过额外的计算在测试时提升其推理能力，然而大多数现有工作将每个问题孤立处理。当问题按顺序到来时，跨问题积累可复用的经验可以进一步提升性能。现有的基于记忆的方法要么存储对新问题泛化能力差的整体解法模板，要么使用未针对最终答案正确性优化的启发式步骤级选择。学习选择策略需要大规模训练数据和固定的动作空间，这使得此类方法不适用于记忆逐步扩展且仅有有限监督的现实测试时场景。我们提出了MILES（具有可学习选择的模块化指令记忆），这是一个在现实测试时约束下动态扩展步骤级记忆并应用以正确性为优化目标的记忆组合的框架。MILES维护模块化记忆

    arXiv:2607.06974v2 Announce Type: replace  Abstract: Large language models (LLMs) increasingly improve their reasoning at test time via additional computation, yet most existing works treat each problem in isolation. When problems arrive sequentially, accumulating reusable experience across them can further improve performance. Existing memory-based methods either store whole-solution templates that generalize poorly to novel problems or use heuristic step-level selection that is not optimized for final-answer correctness. Learning selection policies requires large-scale training data and fixed action spaces, making such approaches unsuitable for test-time settings where memory expands incrementally and only limited supervision is available. We propose MILES (Modular Instruction Memory with LEarnable Selection), a framework that dynamically expands step-wise memory and applies correctness-optimized memory composition under realistic test-time constraints. MILES maintains modular memory
    
[^278]: 当简单更优时：评估中世纪拉丁手稿的翻译流水线

    When Simpler Is Better: Evaluating Translation Pipelines for Medieval Latin Manuscripts

    [https://arxiv.org/abs/2607.03836](https://arxiv.org/abs/2607.03836)

    在中世纪拉丁手稿翻译任务中，参数量少几个数量级的领域专用OCR模型反而优于通用视觉语言模型（字符错误率最多降低4.3倍），并据此构建了包含1,383条对齐图像、转录与译文的新型数据集IPC。

    

    尽管机器翻译取得了显著进展，视觉语言模型（VLM）在历史手稿上的表现仍然不佳。该领域考验着自然语言处理（NLP）的核心能力：低资源转写、古旧词汇以及嘈杂的输入信号。我们提出了一个系统性框架，用于评估从中世纪拉丁手稿图像到译文的完整流水线。在这一场景中，抄写员的速记符号、连字以及羊皮纸的破损暴露出在干净文本基准测试中难以察觉的失败模式。在CATMuS Latin数据集上的基准测试揭示了一种专业化差距：领域专用的光学字符识别（OCR）模型与通用VLM相比，可将字符错误率降低至多4.3倍，尽管其参数量少了几个数量级。我们提出了Interpres-Parallel-Corpus（IPC），一个包含1,383条对齐的手稿图像行、转录文本及实验（结果）的新型数据集。

    arXiv:2607.03836v2 Announce Type: replace-cross  Abstract: Despite remarkable progress in machine translation, Vision Language Models (VLMs) struggle on historical manuscripts, a domain that stresses core Natural Language Processing (NLP) capabilities: low-resource transliteration, archaic vocabulary, and noisy input signals. We present a systematic framework for evaluating the full image-to-translation pipeline on medieval Latin manuscripts, a setting in which scribal shorthand, ligatures, and parchment degradation expose failure modes that are invisible in clean-text benchmarks. Benchmarking on the CATMuS Latin dataset reveals a specialization gap: domain-specific Optical Character Recognition (OCR) models reduce character error rate by up to 4.3$\times$ compared to general-purpose VLMs, despite operating at orders of magnitude fewer parameters. We introduce the Interpres-Parallel-Corpus (IPC), a novel dataset comprising 1,383 aligned manuscript image lines, transcriptions, and exper
    
[^279]: GameEngineBench：在网络化游戏引擎中对编码智能体进行全程会话验证

    GameEngineBench: Full-Session Verification of Coding Agents in a Networked Game Engine

    [https://arxiv.org/abs/2607.03525](https://arxiv.org/abs/2607.03525)

    该论文提出了 GameEngineBench 基准（包含源自虚幻引擎 5 项目的 110 个 C++ 实现任务），并引入全程会话验证框架，通过在 PIE 会话（含联网客户端世界）中注入隐藏测试，来评估编码智能体所生成代码的行为正确性与跨客户端一致性。

    

    编码智能体在游戏开发中面临新的挑战，它们必须生成不仅语法正确，而且与引擎管理的生命周期、状态和网络行为保持一致的代码。然而，现有的游戏开发基准测试未能将针对引擎运行时状态的断言与跨客户端一致性相结合，因而无法评估生成的代码在多大程度上符合预期的行为规范。为了解决这些局限性，我们提出了 GameEngineBench，这是一个包含 110 个 C++ 实现任务的基准测试，这些任务源自 9 个现有的虚幻引擎 5（Unreal Engine 5）项目。每个任务要求智能体在既定的游戏玩法架构中实现缺失的行为。我们还引入了全程会话验证，这是一个测试框架，它在智能体完成任务后注入隐藏测试，并在编辑器内运行会话中评估最终实现，其中 71 个任务包含联网的客户端世界。测试……

    arXiv:2607.03525v3 Announce Type: replace-cross  Abstract: Coding agents face new challenges in game development, where they must produce code that is not only syntactically correct but also consistent with engine-managed lifecycle, state, and networking behavior. Existing game-development benchmarks, however, do not combine assertions over engine-based runtime states with cross-client consistency, failing to evaluate how well generated code adheres to the desired behavioral specifications. To address these limitations, we present GameEngineBench, a benchmark of 110 C++ implementation tasks derived from 9 existing Unreal Engine 5 projects. Each task requires an agent to implement missing behavior within an established gameplay architecture. We also introduce Full-Session Verification, a testing framework that injects hidden tests after the agent finishes and evaluates the resulting implementation inside Play-in-Editor (PIE) sessions, with networked client worlds in 71 tasks. The tests 
    
[^280]: 对人类不可见，对机器可见：四项生物医学文献API的Unicode保真度预注册审计

    Invisible to humans, visible to machines: a preregistered audit of Unicode fidelity across four biomedical bibliographic APIs

    [https://arxiv.org/abs/2606.24897](https://arxiv.org/abs/2606.24897)

    本研究开发并预注册了一种可复用的字符级保真度审计方法，系统评估了PubMed、Crossref、OpenAlex和Semantic Scholar四个生物医学文献API对Unicode特殊字符（如希腊字母、数学符号、排版标点和特殊空白）的保留能力，揭示了这些API存在的系统性字符丢失问题。

    

    本研究旨在开发一种可复用、预注册的方法，用于审计文献应用程序编程接口（API）的字符级保真度，并将其应用于用于构建生物医学文本语料库的四个公共API。这是一项预注册审计（OSF osf.io/zjevr）。研究从2024年PubMed Central（PMC）开放获取子集中简单随机抽取了4000篇英文研究论文，将PubMed E-utilities、Crossref、OpenAlex和Semantic Scholar返回的摘要与PMC期刊文章标签套件（JATS）XML摘要进行对比，共涉及51个注册码点，其中34个属于四个主要类别：排版标点符号、数学/科学符号、希腊字母和特殊空白字符。当条件保留率的95% Wilson置信区间上限低于5%时，某单元格（API x 类别）被归类为系统性丢失；当置信区间下限超过90%时，则被归类为保真。研究还对每类丢失的机制进行了检验……（原文在此处截断）

    arXiv:2606.24897v2 Announce Type: replace-cross  Abstract: To develop a reusable, preregistered method for auditing the character-level fidelity of bibliographic application programming interfaces (APIs), and to apply it to four public APIs used to build biomedical text corpora. Preregistered audit (OSF osf.io/zjevr). From the 2024 PubMed Central (PMC) Open Access subset, a simple random sample of 4,000 English-language research articles was drawn. Abstracts returned by the PubMed E-utilities, Crossref, OpenAlex and Semantic Scholar were compared with the PMC Journal Article Tag Suite (JATS) XML abstract for 51 registered code points, 34 of them in four primary classes: typographic punctuation, mathematical/scientific symbols, Greek letters and special whitespace. A cell (API x class) was classified as systematic loss when the upper 95% Wilson bound of conditional preservation was below 5%, and as faithful when the lower bound exceeded 90%. The mechanism of each loss was examined in th
    
[^281]: AgentOdyssey：面向测试时持续学习智能体的开放式长程文本游戏生成

    AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents

    [https://arxiv.org/abs/2606.24893](https://arxiv.org/abs/2606.24893)

    AgentOdyssey是一个可程序化生成开放式长程文本游戏的评估框架，通过在部署全程中交替学习与推理的设定，从游戏进度、知识获取、情景记忆和探索能力等多维度评估测试时持续学习智能体的关键能力。

    

    为了让智能体能够在测试时通过与世界的交互进行持续学习，它们必须能够有效地探索、获取新的世界知识和技能、保留相关的情景经验，并进行长程规划。为了评估测试时持续学习智能体的这些关键能力，我们提出了AgentOdyssey，这是一个新颖的评估框架，能够程序化地生成包含丰富实体、世界动态和长程任务的开放式文本游戏。至关重要的是，AgentOdyssey突破了传统机器学习中“测试时不发生学习”的假设，将智能体置于一个持续的、长程的环境中，在整个部署过程中交替进行学习与推理。我们进一步提出了一种多层面的评估方法，不仅衡量游戏进度，还提供关于世界知识获取、情景记忆、对象与动作探索、动作多样性以及模型一致性等方面的诊断测试。

    arXiv:2606.24893v2 Announce Type: replace  Abstract: For agents to learn continuously from interaction with the world at test time, they must be able to explore effectively, acquire new world knowledge and skills, retain relevant episodic experiences, and plan over long horizons. To evaluate these key abilities of test-time continual learning agents, we introduce AgentOdyssey, a novel evaluation framework that procedurally generates open-ended text games with rich entities, world dynamics, and long-horizon tasks. Critically, AgentOdyssey goes beyond the conventional machine learning assumption that learning does not occur at test time by placing agents in a continuous, long-horizon setting that interleaves learning and inference throughout deployment. We further propose a multifaceted evaluation methodology that measures not only game progress but also offers diagnostic tests on world knowledge acquisition, episodic memory, object and action exploration, action diversity, and model cos
    
[^282]: 同名者（NAMESAKES）：探测文生图模型中的身份记忆

    NAMESAKES: Probing Identity Memorization in Text-to-Image Models

    [https://arxiv.org/abs/2606.20155](https://arxiv.org/abs/2606.20155)

    该论文提出了一种无需参考照片或训练数据知识的完全黑盒行为探测方法及涵盖一千多名公众人物的NAMESAKES数据集，用于区分文生图模型生成的人脸是记忆的真实身份还是凭空捏造，以应对隐私风险。

    

    文生图（T2I）模型在输入某些人的名字时能够生成他们的逼真肖像，这引发了隐私方面的担忧。然而，目前要区分生成的人脸是模型记忆的还是凭空捏造的，需要真实照片、训练数据的访问权限，或对模型内部的白盒访问，这限制了其适用性。我们提出了一种完全黑盒的行为探测方法，能够区分被记忆的名字和未被识别的名字，且无需参考照片或对训练数据的先验知识。为对该任务进行基准测试，我们构建了NAMESAKES数据集，其中包含一千多个公众人物的名字和面孔，涵盖广泛的知名度范围，还包括经过扰动的、知名度较低的名字。在最先进的文生图模型上的实验表明，我们的探测方法能有效预测身份记忆，并将被记忆的名字与未被识别的名字区分开来，同时还深入揭示了不同模型家族之间的差异。

    arXiv:2606.20155v3 Announce Type: replace-cross  Abstract: Text-to-image (T2I) models generate realistic likenesses of some individuals when prompted with their names, raising privacy concerns. However, distinguishing whether a generated face is memorized or fabricated currently requires ground-truth photos, access to training data, or white-box access to model internals, limiting applicability. We introduce a fully black-box behavioral probe that distinguishes between memorized and unrecognized names, while requiring no reference photos or prior knowledge of training data. To benchmark this task, we present the NAMESAKES dataset of over one thousand names and faces of public figures spanning a wide range of fame levels, along with perturbed, less famous names. Experiments on state-of-the-art T2I models show that our probe substantially predicts identity memorization and separates memorized from unrecognized names, with further insights into differences across model families.
    
[^283]: 通过图像将语音与文字相连

    Connecting Speech to Words through Images

    [https://arxiv.org/abs/2606.16807](https://arxiv.org/abs/2606.16807)

    该论文提出一种无需任何文本监督的视觉接地方法，仅利用图像及其口语描述构建口语词片段与书面词的对应关系，在口语词检索和关键词检出任务上超越强神经基线且更具可解释性。

    

    在缺乏显式文本监督的情况下，我们如何学习书面词与其口语对应形式之间的映射？我们提出了一种视觉接地的方法，仅使用图像及其口语描述来构建口语词汇库。首先，利用图像描述生成系统构建一个代表图像中显著视觉概念的书面词汇库；对于每个词，我们找出其图像描述中包含该词的语音话语；然后使用一种无监督的词发现技术对这些话语进行对齐，以定位目标词的实例。最终得到与书面词相关联的口语词片段——这一切都无需任何文本监督。在口语词检索和关键词检出实验中，所提出的方法优于强大的神经基线，同时更具可解释性。这些结果证明了该方法在英语中的可行性，并为未来研究提供了方向。

    arXiv:2606.16807v2 Announce Type: replace  Abstract: How can we learn the mapping between written words and their spoken counterparts in the absence of explicit textual supervision? We present a visually grounded method for building a vocabulary of spoken words using only images and their spoken descriptions. First, image captioning systems are used to build a vocabulary of written words representing salient visual concepts in the images. For each word, we then find utterances whose image captions contain that word. Then we use an unsupervised word discovery technique to align these utterances to locate instances of the target word. The result is spoken word segments that are linked to written words -- all accomplished without any text supervision. In spoken word retrieval and keyword spotting experiments, the proposed approach outperforms a strong neural baseline while being more interpretable. These results demonstrate the feasibility of the approach in English and motivate future wo
    
[^284]: 后训练即（大规模）监督学习

    Post-training is (Massive) Supervised Learning

    [https://arxiv.org/abs/2606.07527](https://arxiv.org/abs/2606.07527)

    这篇立场论文指出，当前由SFT和强化学习构成的大规模后训练范式本质上是向BERT时代“预训练+微调”方法的回归，并通过实验证明仅靠后训练（即使从随机初始化开始）也能在数学和代码基准上取得相当可观的成绩。

    

    训练大语言模型（LLM）的主流范式已经演变为依赖一个由监督微调（SFT）和强化学习（RL）组成的大规模后训练阶段。在这篇立场论文中，我们认为这种方法实际上标志着向BERT时代“先预训练后微调”方法的回归，即显式地将模型定制为期望的行为以及用于评估的特定基准。我们首先对LLM进行了历史回顾，描述了LLM演化的不同阶段。我们认为当前的格局与LLM早期阶段惊人地相似，当时任务性能在很大程度上依赖于将模型拟合到分布内的数据集。为了实证证明这一点，我们将预训练模型与随机初始化的模型进行比较，在现代推理数据集上对两种变体进行微调，并在具有竞争力的数学和代码基准上进行评估。我们表明，从头开始进行后训练的模型能够取得高度可观的非平凡性能。

    arXiv:2606.07527v2 Announce Type: replace-cross  Abstract: The prevailing paradigm for training LLMs has evolved to rely on a massive post-training phase consisting of SFT and RL. In this position paper, we argue that this methodology effectively marks a reversion to the ``pre-train then fine-tune'' approach of the BERT era, explicitly tailoring models to the desired behaviors and specific benchmarks on which they are evaluated. We begin with a historical overview of LLMs, describing the different phases of the LLM evolution. We argue that the current landscape is remarkably similar to the early days of LLMs, where task performance heavily relied on fitting the models to in-distribution datasets. To empirically demonstrate this, we compare pre-trained models to randomly initialized ones, by fine-tuning both variants on modern reasoning datasets and evaluating them on competitive math and code benchmarks. We show that models post-trained from scratch yield highly non-trivial performance
    
[^285]: AURA：面向情境化LLM智能体隐式需求挖掘的意图导向探测

    AURA: Intent-Directed Probing for Implicit-Need Surfacing in Situated LLM Agents

    [https://arxiv.org/abs/2606.05557](https://arxiv.org/abs/2606.05557)

    AURA通过在场景感知与工具调用之间插入意图推理步骤，生成带差距分数的IntentFrame来挖掘情境化查询背后的隐式需求，其隐式需求覆盖率显著优于ReAct式探测方法。

    

    像“林伟在哪里？”这样的情境化查询往往蕴含着超出字面内容的信息：用户可能还想了解林伟是否空闲、心情是否良好、或者现在是否值得去打扰。标准的工具调用型智能体只会回答字面问题便停止。AURA在场景感知与工具调用之间插入一个推理步骤，生成一个IntentFrame（意图框架）：一种对隐式需求的结构化估计，并带有标量差距分数，用于控制每个查询的探测预算与工具选择。在一个包含100个查询、四个场景的隐式意图基准测试中，AURA在隐式需求覆盖率上超越了ReAct风格的探测方法（Delta = +0.07，p < 10^-6）；四个场景中有三个单独达到统计显著，该增益在第二个骨干模型上得到复现，且提示消融实验将这一提升归因于差距校准而非答案记忆。组件消融进一步定位了增益来源：仅凭差距推导的预算上限就优于完整架构（p = 0.02），并且分箱……（摘要原文在此处截断）

    arXiv:2606.05557v2 Announce Type: replace  Abstract: A situated query like "where is Lin Wei?" often encodes more than its literal content: the user may also want to know whether Lin Wei is free, in a good mood, or worth interrupting now. Standard tool-use agents answer the literal question and stop. AURA inserts an inference step between scene perception and tool use that produces an IntentFrame: a structured estimate of the implicit need with a scalar gap score that controls per-query probe budget and tool selection. On a 100-query four-scene implicit-intent benchmark, AURA improves implicit-need coverage over ReAct-style probing (Delta = +0.07, p < 10^-6); three of four scenes are individually significant, the gain reproduces on a second backbone, and a prompt ablation attributes the lift to gap calibration rather than answer memorisation. A component ablation localises the gain further: the gap-derived budget ceiling alone outperforms the full architecture (p = 0.02), and bin-wise 
    
[^286]: DeliChess：一个用于国际象棋谜题求解审议的多方对话数据集

    DeliChess: A Multi-party Dialogue Dataset for Deliberation in Chess Puzzle Solving

    [https://arxiv.org/abs/2606.04987](https://arxiv.org/abs/2606.04987)

    该论文构建了首个多方审议对话数据集DeliChess，让参与者协作解决国际象棋谜题，揭示出初始答案多样性越高审议收益越大，以及审议过程中群体答案可被恢复、发现或丢失的规律。

    

    多方对话是研究协作推理与决策的关键场景，然而现有数据集很少关注结构化、推理密集型的任务。我们提出了DeliChess，这是一个群体审议对话数据集，参与者在其中协作解决多选式的国际象棋谜题。参与者首先独立作答，随后进行多方审议并修改各自答案。该数据集包含107个会话，涵盖24个独特棋局上的321次谜题讨论，并配有完整的对话转录文本、审议前后的答案选择，以及话语层面的交流功能、认知立场和支持审议有用性的标注。我们的分析表明，初始答案质量的多样性越高，审议带来的收益越大；同时，答案轨迹揭示了审议如何能够恢复、发现或丢失优质答案。

    arXiv:2606.04987v3 Announce Type: replace-cross  Abstract: Multi-party dialogue is a critical setting for studying collaborative reasoning and decision-making, yet existing datasets rarely focus on structured, reasoning-intensive tasks. We introduce DeliChess, a dataset of group deliberation dialogues in which participants collaboratively solve multiple-choice chess puzzles. Participants first answer independently, then engage in multi-party deliberation and revise their individual answers. The dataset comprises 107 sessions containing 321 puzzle discussions across 24 unique positions, with full transcripts, pre- and post-deliberation choices, and utterance-level annotations of communicative function, epistemic stance, and usefulness in supporting deliberation. Our analyses show that greater diversity in initial solution quality is associated with larger gains, while answer trajectories reveal how deliberation can recover, discover, or lose strong answers. Cases in which groups surpass
    
[^287]: 从带标注验证集的输出统计量预测推理时扩展增益

    Predicting Inference-Time Scaling Gains from Labeled Validation-Set Output Statistics

    [https://arxiv.org/abs/2606.02981](https://arxiv.org/abs/2606.02981)

    该论文提出仅利用带标注验证集上一轮采样得到的四个输出统计特征，即可高精度（Spearman秩相关0.90）预测语言模型最佳N选一推理扩展带来的准确率增益，从而免去昂贵的候选生成与奖励模型评分过程。

    

    验证器选择的最佳N选一扩展（从语言模型中抽取N个候选答案，并返回奖励模型排名最高的那个）能够提升准确率，但提升幅度因模型而异，直接测量该增益需要同时进行候选生成和奖励模型评分。先前的工作将模型采样输出的廉价统计量（样本之间的一致程度、多样性、模型置信度以及正确样本出现的位置）与验证集正确率及模型行为联系起来，但并未分离出其中哪些统计量能够构成最佳N选一增益的紧凑预测器。我们在带标注验证集上进行一轮采样，基于所得特征拟合岭回归预测器，并使用bootstrap-Lasso衡量候选特征在重采样中被选中的频率，同时给出了考虑线性预测器局限性的误差界。仅使用四个特征，我们的预测器与实测增益之间的Spearman秩相关系数达到0.90。

    arXiv:2606.02981v2 Announce Type: replace  Abstract: Verifier-selected best-of-N scaling (drawing N candidate answers from a language model and returning the one a reward model ranks highest) improves accuracy by an amount that varies across models, but measuring that gain directly requires both candidate generation and reward-model scoring. Prior work links cheap statistics of a model's sampled outputs and validation-set correctness (how often samples agree, how diverse they are, how confident the model is, and where correct samples appear) to model behavior, but does not isolate which of these form a compact predictor of best-of-N gain. We fit ridge predictors on features from one round of sampling on a labeled validation set, use bootstrap-Lasso to measure how often candidate features are selected across resamples, and give an error bound that accounts for the limits of a linear predictor. Using only four features, our predictor achieves a Spearman rank correlation of 0.90 with meas
    
[^288]: 视觉语言模型在自然阅读中是否比大语言模型更贴近人类？

    Do VLMs Align Better with Humans than LLMs Do during Natural Reading?

    [https://arxiv.org/abs/2605.28818](https://arxiv.org/abs/2605.28818)

    本研究在纯文本输入下比较匹配的LLM与VLM对，发现视觉语言训练带来的优势是选择性的——VLM在预测人类回视眼跳上更准确，并在fMRI对齐上与LLM相当甚至日益占优，表明视觉语言学习能带来更接近人类的文本表征。

    

    大语言模型（LLM）已成为日益有用的人类语言处理计算模型，但视觉语言学习是否能在自然阅读过程中产生更接近人类的文本表征，仍是一个悬而未决的问题。我们通过在严格纯文本输入条件下比较匹配的大语言模型与视觉语言模型（VLM）对，并评估其与人类大脑活动（全皮层fMRI）及阅读行为（同步回视性眼跳）的对齐程度来回答这一问题。在所评估的模型对中，我们发现人-模型对齐呈现出选择性而非全局性的模式。在第一方持续预训练的模型对（Qwen2和DeepSeek）中，视觉语言模型更准确地预测了人类的回视性眼跳，同时在全皮层fMRI对齐上与匹配的大语言模型表现出实际等同性；而LoRA适配的模型对（Mistral/Idefics2）则未显示出差异。然而，句子层面的fMRI分析揭示了视觉语言模型日益增长的优势。

    arXiv:2605.28818v3 Announce Type: replace  Abstract: Large language models (LLMs) have become increasingly useful computational models of human language processing, yet it remains open whether vision--language learning is associated with more human-like text representations during natural reading. We address this question by comparing matched LLM and vision--language model (VLM) pairs under strictly text-only input and evaluating alignment with human brain activity (whole-cortex fMRI) and reading behavior (synchronized regressive saccades). Across evaluated pairs, we identify a selective, rather than global, pattern in human--model alignment. In first-party continued-pretraining pairs (Qwen2 and DeepSeek), VLMs more accurately predicted human regressive saccades while exhibiting practical equivalence with matched LLMs in whole-cortex fMRI alignment, whereas a LoRA-adapted pair (Mistral/Idefics2) showed no divergence. However, sentence-level fMRI analyses revealed a growing VLM advantag
    
[^289]: 弹出干扰揭示视频大型语言模型中的“事件袋”行为

    Pop-Up Distractions Reveal Bag-of-Events Behavior in Video Large Language Models

    [https://arxiv.org/abs/2605.27101](https://arxiv.org/abs/2605.27101)

    这项研究通过引入DistractionBench测试，发现所有视频大型语言模型普遍存在将无关事件错误关联的“事件袋”幻觉行为，缺乏跨时间关联主体与事件的稳健能力。

    

    arXiv:2605.27101v2 公告类型：替换交叉 摘要：视频理解的一个关键能力是可靠地将主体与跨时间的事件关联起来，然而视频大型语言模型（VideoLLMs）是否真正实现这一点仍不清楚。在本工作中，我们引入了DistractionBench来评估VideoLLMs在存在无关视频片段时是否能稳健地将主体和事件关联起来。通过受控干预，例如在较长的视频中插入短广告片段，我们展示了VideoLLMs经常幻觉化地产生来自不同片段实体之间的交互，错误地将注入广告中的动作归因于主视频中的主体。我们将这种系统性幻觉特征化为“事件袋”（BoE）行为，即模型将视频视为事件的集合而非时间上结构化的序列。评估了11种流行的VideoLLMs，我们发现所有模型都表现出显著的BoE行为。我们的发现表明VideoLLMs缺乏时间结构化的视频理解能力。

    arXiv:2605.27101v2 Announce Type: replace-cross  Abstract: A key capability for video understanding is reliably linking subjects to events across time, yet whether Video Large Language Models (VideoLLMs) actually achieve this remains unclear. In this work, we introduce DistractionBench to evaluate whether VideoLLMs can robustly link subjects and events in the presence of unrelated video segments. Through controlled interventions, such as inserting short advertisement clips into longer videos, we show that VideoLLMs frequently hallucinate interactions between entities from different segments, incorrectly attributing actions from injected advertisements to subjects in the main video. We characterize this systematic hallucination as bag-of-events (BoE) behavior, where models process videos as collections of events rather than temporally structured sequences. Evaluating 11 popular VideoLLMs, we find that all models exhibit substantial BoE behavior. Our findings suggest that VideoLLMs lack 
    
[^290]: 向量并非中立：从大语言模型摘要导出的表示中推断敏感信息

    Vectors Are Not Neutral: Sensitive-Information Inference from Exported LLM Representations in Summarization

    [https://arxiv.org/abs/2605.26433](https://arxiv.org/abs/2605.26433)

    该论文以临床出院摘要为案例，首次揭示LLM导出的向量表示（如提示token隐藏状态和平均池化表示）即使脱离源文档的访问控制，仍可被用于推断敏感属性（如患者种族），且针对单一产物的保护措施并不必然对其他产物有效。

    

    大型语言模型（LLM）摘要系统可能会将私有输入的紧凑向量表示传递给下游的检索、监控、审计或分析工作流。即使源文档仍受访问限制，这些导出的向量可能在不同的访问控制下被处理，却仍然能够支持敏感信息推断，从而带来残余的信息泄露风险。我们在临床出院小结生成这一高风险场景中以案例研究的方式考察该问题，并将电子健康记录（EHR）中记录的种族作为受控的敏感标签进行审计。我们审计了系统可能保留或暴露给下游组件的两种产物：最终的提示词token隐藏状态和平均池化的提示表示。结果表明，降低某一导出产物中敏感标签的可恢复性，并不必然降低另一产物中的可恢复性。作为缓解措施的案例研究，我们提出了Surface（原文摘要在此处截断，后续内容缺失）。

    arXiv:2605.26433v2 Announce Type: replace  Abstract: Large language model (LLM) summarization systems may pass compact vector representations of private inputs to downstream retrieval, monitoring, audit, or analytic workflows. Even when source documents remain access-restricted, derived vectors may be handled under different access controls and still support sensitive-information inference, creating a residual information-disclosure risk. We study this issue in clinical discharge-summary generation as a high-stakes case study, using electronic health record (EHR)-recorded race as a controlled sensitive-label audit. We audit two artifacts that a system might retain or expose to downstream components: the final prompt-token hidden state and the mean-pooled prompt representation. Our results show that reducing recoverability of the case-study sensitive label from one exported artifact does not necessarily reduce recoverability from another. As a mitigation case study, we introduce Surface
    
[^291]: 扩散理论教程：从微分方程到扩散模型

    A Tutorial on Diffusion Theory: From Differential Equations to Diffusion Models

    [https://arxiv.org/abs/2605.22586](https://arxiv.org/abs/2605.22586)

    本教程从前向与反向随机微分方程两个视角统一介绍扩散模型的数学基础，推导训练目标，建立DDPM与VP-SDE的联系，并涵盖SDE数值求解方法与引导生成技术。

    

    扩散模型已成为生成建模的主流框架，但其数学基础往往以各自独立的方式呈现。本教程提供了一个统一的、自成体系的介绍，共分为三个部分。首先，我们构建一个将噪声传输至数据的前向生成随机微分方程（SDE），对干净数据端点进行边缘化处理，并直接从前向动力学推导训练目标。其次，我们从数据到噪声的条件路径出发，将其边缘化以获得前向过程，并推导相应的反向时间SDE和训练目标。在这一前向-反向框架中，我们还建立了去噪扩散概率模型（DDPM）与方差保持（VP）SDE之间的联系。第三，我们回顾了求解SDE的数值方法，并讨论了两种引导生成的方法：分类器引导和无分类器引导。

    arXiv:2605.22586v4 Announce Type: replace-cross  Abstract: Diffusion models have emerged as a leading framework for generative modeling, yet their mathematical foundations are often presented in separate treatments. This tutorial offers a unified, self-contained introduction organized into three parts. First, we construct a forward generative stochastic differential equation (SDE) that transports noise to data, marginalize over the clean data endpoint, and derive training objectives directly from the forward dynamics. Second, we begin with a conditional path from data to noise, marginalize it to obtain the forward process, and derive the corresponding reverse-time SDE and training objectives. Within this forward-reverse framework, we also establish the connection between denoising diffusion probabilistic models (DDPMs) and variance-preserving (VP) SDEs. Third, we review numerical methods for solving SDEs and discuss two approaches to guided generation: classifier guidance and classifie
    
[^292]: SocialMemBench：AI记忆系统是否已为社交群体场景做好准备？

    SocialMemBench: Are AI Memory Systems Ready for Social Group Settings?

    [https://arxiv.org/abs/2605.17789](https://arxiv.org/abs/2605.17789)

    提出了首个针对社交群体场景的AI记忆系统基准SocialMemBench，通过430个人设和1,031个问答对构成的合成社交网络，揭示了现有记忆系统在群体对话中的五种失败模式。

    

    面向AI助手的记忆系统是为单用户对话设计的，这一设计与多方社交群体场景并不匹配。这对当今正在构建的社交助手至关重要：聊天平台中以群体方式运作的智能体，以及主动式个人助手——其对用户的建模必须包含用户的社交背景。现有的记忆基准评估的是双人对话或职场对话；没有针对社交群体的基准，而在社交群体中，记忆必须将事实锚定于共享历史而非职业角色，区分群体规范与个体例外，并且在成员离开后仍能正确归属信息。我们提出了SocialMemBench，这是一个经人工验证的合成社交群体网络基准，涵盖五种群体原型和三种规模层级（4-30名成员），包含430个人设和1,031个问答对，分布于九个问题类别。我们的评估揭示了五种失败模式，并将其表述为可检验的假设，同时我们通过研究为其中两种提供了证据。

    arXiv:2605.17789v2 Announce Type: replace-cross  Abstract: Memory systems for AI assistants were built for single-user dialogue, a design mismatched to multi-party social group settings. This matters for the social assistants being built today: group-acting agents in chat platforms, and proactive personal assistants whose model of a user must include their social context. Existing memory benchmarks evaluate dyadic or workplace dialogue; none targets social groups, where memory must anchor facts in shared history rather than professional roles, separate group norms from individual exceptions, and attribute correctly even after a member leaves.   We introduce SocialMemBench, a benchmark of human-verified synthetic social group networks across five archetypes and three size tiers (4-30 members), with 430 personas and 1,031 QA pairs across nine question categories. Our evaluation exposes five failure modes, framed as testable hypotheses, and we provide evidence on two through the research 
    
[^293]: 基于扩散语言模型的约束代码生成

    Constrained Code Generation with Diffusion Language Models

    [https://arxiv.org/abs/2605.16829](https://arxiv.org/abs/2605.16829)

    本文提出推理时神经符号框架CDC，无需重新训练扩散模型，通过将数学优化与程序分析相结合的约束感知去噪算子嵌入反向去噪过程，从而在代码生成中强制执行功能与安全约束。

    

    扩散语言模型（DLM）是一种强大的新兴代码生成范式。它通过对部分损坏的token序列进行迭代细化来构建程序，并支持并行token细化。重要的是，该范式在每个去噪步骤中都会暴露全局程序状态，这为实施程序级的功能与安全约束提供了一个自然的干预点，可以在最终代码确定之前对生成过程进行引导。基于这一观察，本文提出了面向代码的约束扩散，这是一个推理阶段的神经符号框架，它将约束满足直接集成到反向去噪过程中，而无需重新训练基础扩散模型。CDC通过约束感知的去噪算子增强了基础扩散采样器，这些算子将数学优化与程序分析相结合，以识别中间程序状态中与约束相关的区域。

    arXiv:2605.16829v2 Announce Type: replace  Abstract: Diffusion language models (DLMs) are a powerful, emerging paradigm for code generation. They construct programs through iterative refinement of partially corrupted token sequences and enable parallel token refinement. Importantly, this paradigm exposes a global program state at each denoising step, which provides a natural intervention point for enforcing program-level functionality and security constraints, guiding the generation before the final code is committed. Building on this observation, the paper introduces Constrained Diffusion for Code (CDC), an inference-time neurosymbolic framework that integrates constraint satisfaction directly into the reverse denoising process without retraining the base diffusion model. CDC augments the base diffusion sampler with constraint-aware denoising operators that combine mathematical optimization with program analysis to identify constraint-relevant regions of the intermediate program state
    
[^294]: 基于语义级奖励的大语言模型校准

    Calibrating LLMs with Semantic-level Reward

    [https://arxiv.org/abs/2605.15588](https://arxiv.org/abs/2605.15588)

    提出语义奖励校准（CSR）框架，直接在语义空间而非词元级别校准大语言模型的不确定性，解决了相同语义不同表述下置信度不一致的问题。

    

    随着大语言模型被部署在医疗问答和法律推理等具有重大影响的场景中，估计其输出何时可能正确的能力对于安全可靠的使用至关重要，这需要良好的不确定性校准。标准的可验证奖励强化学习（RLVR）使用与置信度无关的二元正确性奖励来训练模型，对自信但错误的预测没有任何惩罚，从而降低了模型的校准性能。近期的工作通过训练模型在生成答案的同时输出口头表达的置信度分数，并奖励其与正确性的一致性来解决这一问题。然而，口头表达的置信度是在词元级别进行校准的，因此在具有相同语义含义的不同文本表述之间会表现出不一致性。我们提出了语义奖励校准，这是一个直接在语义空间中校准语言模型的框架，无需……（原文摘要至此截断）

    arXiv:2605.15588v3 Announce Type: replace  Abstract: As large language models (LLMs) are deployed in consequential settings such as medical question answering and legal reasoning, the ability to estimate when their outputs are likely to be correct is essential for safe and reliable use, requiring well-calibrated uncertainty. Standard reinforcement learning with verifiable rewards (RLVR) trains models with a binary correctness reward that is indifferent to confidence, providing no penalty for confident but wrong predictions and thereby degrading calibration. Recent work addresses this by training models to produce verbalized confidence scores alongside answers and rewarding agreement with correctness. However, verbalized confidence is calibrated at the token level and thus exhibits inconsistency across textual variations with same semantic meaning. We propose \textbf{Calibration with Semantic Reward (CSR)}, a framework that calibrates language models directly in semantic space without a
    
[^295]: 看见并不等于学会：保护多模态数据免受大型视觉语言模型的未授权微调

    To See is Not to Learn: Protecting Multimodal Data from Unauthorized Fine-Tuning of Large Vision-Language Models

    [https://arxiv.org/abs/2605.14291](https://arxiv.org/abs/2605.14291)

    提出MMGuard方法，通过注入利用LVLM学习动态的不可感知扰动生成“不可学习的多模态样本”，诱导模型过拟合于噪声，使数据所有者能够在侵权发生前主动保护多模态数据免受未经授权的模型微调。

    

    大型视觉语言模型（LVLMs）的快速发展伴随着对其多模态网络数据的未经授权抓取和训练，这给数据所有者带来了版权和隐私风险。现有的应对措施，如机器遗忘和水印技术，本质上都是事后方法，只能在知识产权侵权已经发生后才发挥作用。在这项工作中，我们提出了MMGuard，使数据所有者能够主动保护其多模态数据免受未经授权的LVLM微调。MMGuard通过注入人类无法察觉的扰动来生成不可学习的样本，这些扰动利用了LVLMs的学习动态。通过最小化训练损失，该扰动创建了一个优化捷径，导致模型过拟合于噪声，从而在推理阶段扰动不存在时降低模型的下游性能。为了进一步强化这种防御，MMGuard引入了跨模……

    arXiv:2605.14291v2 Announce Type: replace-cross  Abstract: The rapid advancement of Large Vision-Language Models (LVLMs) is increasingly accompanied by unauthorized scraping and training on multimodal web data, posing copyright and privacy risks to data owners. Existing countermeasures, such as machine unlearning and watermarks, are inherently post-hoc approaches that act only after intellectual property infringement has already occurred. In this work, we propose MMGuard to empower data owners to proactively protect their multimodal data against unauthorized LVLM fine-tuning. MMGuard generates unlearnable examples by injecting human-imperceptible perturbations that exploit the learning dynamics of LVLMs. By minimizing the training loss, the perturbation creates an optimization shortcut, causing the model to overfit to the noise and thereby degrading downstream performance when the perturbation is absent during inference. To further strengthen this defense, MMGuard introduces a cross-mo
    
[^296]: RealICU：大语言模型智能体真的理解长上下文ICU数据吗？一个超越行为模仿的基准测试

    RealICU: Do LLM Agents Understand Long-Context ICU Data? A Benchmark Beyond Behavior Imitation

    [https://arxiv.org/abs/2605.13542](https://arxiv.org/abs/2605.13542)

    RealICU是一个由资深医生回溯完整患者轨迹后进行标注的ICU评估基准，通过患者状态评估、急性问题、推荐行动和危险信号四个医生视角的任务，突破传统“模仿历史医生行为”的评估范式，从而更真实地衡量大语言模型在ICU长上下文数据上的临床推理能力。

    

    重症监护病房（ICU）产生长而密集且不断演变的临床信息流，医生必须在时间压力下反复重新评估患者状态，这凸显了对可靠AI决策支持的迫切需求。现有的ICU基准通常将历史临床医生的操作作为真值标准。然而，这些操作是在信息不完整且对潜在患者状态的时间背景了解有限的情况下做出的，因此可能是次优的，这使得难以评估AI系统的真实推理能力。我们提出了RealICU，这是一个用于在真实ICU条件下评估大语言模型（LLM）的后视标注基准，其中标签由资深医生在审查完整患者轨迹后创建。我们设计了四个由医生需求驱动的任务：评估患者状态、急性问题、推荐行动，以及可能导致不安全结果的危险信号行动。我们对每条轨迹进行划分……

    arXiv:2605.13542v2 Announce Type: replace  Abstract: Intensive care units (ICU) generate long, dense and evolving streams of clinical information, where physicians must repeatedly reassess patient states under time pressure, underscoring a clear need for reliable AI decision support. Existing ICU benchmarks typically treat historical clinician actions as ground truth. However, these actions are made under incomplete information and limited temporal context of the underlying patient state, and may therefore be suboptimal, making it difficult to assess the true reasoning capabilities of AI systems. We introduce RealICU, a hindsight-annotated benchmark for evaluating large language models (LLMs) under realistic ICU conditions, where labels are created after senior physicians review the full patient trajectory. We formulate four physician-motivated tasks: assess Patient Status, Acute Problems, Recommended Actions, and Red Flag actions that risk unsafe outcomes. We partition each trajectory
    
[^297]: 改变我的观点？在线话语中的说服与极化动态

    Change My View? The Dynamics of Persuasion and Polarization in Online Discourse

    [https://arxiv.org/abs/2605.08383](https://arxiv.org/abs/2605.08383)

    本研究利用大型语言模型分析Reddit辩论语料，发现让步与共情等修辞策略显著提高对方改变信念（被说服）的概率。

    

    关于说服的哲学论述常常假定，共享的证据与理性论证应当促使同侪之间的观点趋于一致，然而日常话语往往表明并非如此。在本研究中，我们使用大型语言模型分析了Reddit上r/ChangeMyView版块的辩论语料库，在该版块中，信念修正会被公开标记出来。大型语言模型被要求在每个讨论进行到一半时预测是否会出现这种认同的表示，其概率估计可作为对话的基线。随后，通过一种人机协作的混合编码流程，对每条回复按照十种常见的修辞策略进行编码——包括让步、共情、逻辑质疑、可信度诉求等。加入这些策略特征显著提升了预测能力，并揭示出一个一致的模式：表达让步或共情认同的举措会大幅提高信念改变的可能性，而正面……

    arXiv:2605.08383v2 Announce Type: replace  Abstract: Philosophical accounts of persuasion often assume that shared evidence and rational argumentation should lead to a convergence of views between peers, yet everyday discourse often suggests otherwise. In this study, we use large language models to analyze a corpus of debates on Reddit's r/ChangeMyView, where belief revision is publicly signaled. Large language models were asked, halfway through each discussion, to forecast whether such an acknowledgement would arise; their probabilistic estimates serve as a conversational baseline. Each reply was then coded, through a hybrid machine-assisted procedure, for ten familiar rhetorical strategies -- concession, empathy, logical challenge, credibility appeals, and so forth. Adding these strategic features markedly improves predictive power and yields a consistent pattern: moves that express concession or empathetic alignment substantially increase the prospect of belief change, whereas front
    
[^298]: 多LLM路由中的不可解性上限：评估伪影的实证研究

    Unsolvability Ceiling in Multi-LLM Routing: An Empirical Study of Evaluation Artifacts

    [https://arxiv.org/abs/2605.07395](https://arxiv.org/abs/2605.07395)

    该大规模实证研究（206,000个查询-模型对、六个基准）发现，多LLM路由中报告的“不可解性上限”很大程度上是评估伪影（裁判偏差、截断、输出格式不匹配）造成的，并提出分解框架对失败原因进行归因。

    

    跨多个大语言模型的高效路由通过将查询导向最便宜的、能力足够的模型，实现了成本与质量的权衡。先前的工作将路由的提升空间归因于一种“不可解性上限”，即模型池中没有任何模型能够解决的查询。我们对多层LLM路由进行了一项大规模研究，使用Gemma 4和Llama 3.1模型家族，在六个基准（MMLU、MedQA、HumanEval、MBPP、Alpaca、ShareGPT）上共包含206,000个查询-模型对。通过LLM-as-a-judge（大模型作为裁判）和精确匹配两种指标进行评估，我们发现所报告的不可解性中有相当大一部分源于评估伪影：(i) 系统性的裁判偏差，偏好冗长而非正确性；(ii) 固定生成预算下的截断；(iii) 输出格式不匹配。通过双裁判验证和精确匹配校准，我们降低了各任务上测得的不可解性。我们引入了一个分解框架，将失败归因于这些伪影，揭示了一致的

    arXiv:2605.07395v2 Announce Type: replace-cross  Abstract: Efficient routing across multiple LLMs enables cost-quality tradeoffs by directing queries to the cheapest capable model. Prior work attributes routing headroom to an "unsolvability ceiling", queries no model in the pool can solve. We present a large-scale study of multi-tier LLM routing with 206,000 query-model pairs across six benchmarks (MMLU, MedQA, HumanEval, MBPP, Alpaca, ShareGPT) using the Gemma 4 and Llama 3.1 families. Evaluating with both LLM-as-a-judge and exact-match metrics, we show that a substantial portion of reported unsolvability stems from evaluation artifacts: (i) systematic judge biases favoring verbosity over correctness, (ii) truncation under fixed generation budgets, and (iii) output format mismatches. Through dual-judge validation and exact-match grounding, we reduce measured unsolvability across tasks. We introduce a decomposition framework attributing failures to these artifacts, revealing consistent
    
[^299]: 超越SFT到RL：基于黑盒在线策略蒸馏的多模态强化学习预对齐

    Beyond SFT-to-RL: Pre-alignment via Black-Box On-Policy Distillation for Multimodal RL

    [https://arxiv.org/abs/2604.28123](https://arxiv.org/abs/2604.28123)

    提出PRISM三阶段流水线，通过黑盒在线策略蒸馏的对抗博弈，在SFT与RLVR之间插入显式分布对齐阶段，缓解监督微调引起的分布偏移对多模态强化学习的损害。

    

    大型多模态模型（LMMs）的标准后训练方案是在精心筛选的演示数据上进行监督微调（SFT），随后进行带可验证奖励的强化学习（RLVR）。然而，SFT引入了分布偏移，既无法保留模型的原始能力，也无法忠实地匹配监督分布。这一问题在多模态推理中被进一步放大，因为感知错误和推理失败遵循不同的漂移模式，并在后续强化学习过程中相互叠加。我们提出了PRISM，一个三阶段流水线，通过在SFT和RLVR之间插入显式的分布对齐阶段来缓解这种漂移。基于在线策略蒸馏（OPD）的原理，PRISM将对齐过程建模为策略与一个配备专门感知和推理专家的混合专家判别器（MoE）之间的黑盒、响应级对抗博弈，从而提供解耦的修正信号……

    arXiv:2604.28123v4 Announce Type: replace-cross  Abstract: The standard post-training recipe for large multimodal models (LMMs) applies supervised fine-tuning (SFT) on curated demonstrations followed by reinforcement learning with verifiable rewards (RLVR). However, SFT introduces distributional drift that neither preserves the model's original capabilities nor faithfully matches the supervision distribution. This problem is further amplified in multimodal reasoning, where perception errors and reasoning failures follow distinct drift patterns that compound during subsequent RL. We introduce PRISM, a three-stage pipeline that mitigates this drift by inserting an explicit distribution-alignment stage between SFT and RLVR. Building on the principle of on-policy distillation (OPD), PRISM casts alignment as a black-box, response-level adversarial game between the policy and a Mixture-of-Experts (MoE) discriminator with dedicated perception and reasoning experts, providing disentangled corr
    
[^300]: CogRAG：通过分层检索与推理应对RAG中的异构认知需求

    CogRAG: Tackling Heterogeneous Cognitive Demands in RAG via Stratified Retrieval and Reasoning

    [https://arxiv.org/abs/2604.25928](https://arxiv.org/abs/2604.25928)

    CogRAG是一个无需训练、领域无关的RAG框架，它以查询的预测认知负荷为核心控制信号，协调认知自适应证据精炼和认知分层结构化推理两个模块，解决了统一流水线处理异构认知需求所导致的级联错误与推理-答案不一致问题。

    

    检索增强生成（RAG）框架通常通过一套统一的流水线处理所有查询，忽略了不同任务的异构认知需求。这种“认知盲”的处理方式会导致两种失败模式：低层次的事实缺口引发幻觉推理从而产生级联错误，以及在高层次分析任务中出现推理与答案不一致的问题。我们提出了CogRAG，这是一个无需训练、领域无关的框架，通过分层检索与推理来应对这些异构认知需求。受布鲁姆分类学（Bloom's Taxonomy）启发，CogRAG将查询的预测认知负荷作为核心控制信号，协调两个模块：认知自适应证据精炼模块通过以事实为中心或以选项为中心的路径补充缺失的上下文；认知分层结构化推理模块用与认知对齐的推理模板替代无约束的思维链。（原文摘要在此处截断）

    arXiv:2604.25928v3 Announce Type: replace  Abstract: Retrieval-Augmented Generation (RAG) frameworks typically process all queries through a one-size-fits-all pipeline, ignoring the heterogeneous cognitive demands of different tasks. This cognitive-blind approach causes two failure modes: cascading errors when low-level factual gaps trigger hallucinated reasoning, and reasoning-answer inconsistency in higher-order analytical tasks. We introduce CogRAG, a training-free, domain-agnostic framework that tackles these heterogeneous cognitive demands via stratified retrieval and reasoning. Inspired by Bloom's Taxonomy, CogRAG uses the predicted cognitive load of a query as a central control signal that coordinates two modules: Cognition-Adaptive Evidence Refinement supplements missing context via fact-centric or option-centric paths, and Cognition-Stratified Structured Reasoning replaces unconstrained chain-of-thought with cognition-aligned reasoning templates. We evaluate CogRAG on a demand
    
[^301]: 智能体框架工程：基于可观测性的编码智能体框架自动演化

    Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses

    [https://arxiv.org/abs/2604.25850](https://arxiv.org/abs/2604.25850)

    该论文提出智能体框架工程（AHE）闭环系统，通过组件、经验和决策三大可观测性支柱，首次实现了编码智能体框架的自动化演化，解决了行动空间异构、轨迹信号淹没和编辑效果难归因三大挑战。

    

    框架目前已成为编码智能体性能的核心，它调节着模型与工具及执行环境之间的交互方式。然而，框架工程仍然是一门依赖人工的手艺，因为将其自动化面临诸多挑战：跨可编辑组件的异构行动空间、淹没可操作信号的海量轨迹，以及效果难以归因的编辑。我们提出了智能体框架工程，这是一个闭环系统，通过三大相互匹配的可观测性支柱来应对这些挑战：(1) 组件可观测性为每个可编辑的框架组件提供文件级表示，使行动空间显式且可回滚；(2) 经验可观测性将数百万原始轨迹标记提炼为分层、可下钻的证据语料库，供演化智能体实际消费使用；(3) 决策可观测性为每次编辑配备一个自主声明的预测，并在下一轮任务级结果中对其进行验证。

    arXiv:2604.25850v5 Announce Type: replace  Abstract: Harnesses are now central to coding-agent performance, mediating how models interact with tools and execution environments. Yet harness engineering remains a manual craft, because automating it faces a heterogeneous action space across editable components, voluminous trajectories that bury actionable signal, and edits whose effect is hard to attribute. We introduce Agentic Harness Engineering (AHE), a closed loop that addresses these challenges through three matched observability pillars: (1) component observability gives every editable harness component a file-level representation so the action space is explicit and revertible; (2) experience observability distills millions of raw trajectory tokens into a layered, drill-down evidence corpus that an evolving agent can actually consume; and (3) decision observability pairs every edit with a self-declared prediction, later verified against the next round's task-level outcomes. Together
    
[^302]: 面向跨模型电路迁移的可微忠实性对齐

    Differentiable Faithfulness Alignment for Cross-Model Circuit Transfer

    [https://arxiv.org/abs/2604.24302](https://arxiv.org/abs/2604.24302)

    提出可微忠实性对齐（DFA）框架，通过可微映射和软忠实性目标将小模型的电路信息高效迁移到大模型，避免了在目标模型上进行代价高昂的完整电路发现。

    

    机械可解释性使得定位语言模型中特定行为背后的电路成为可能，但现有方法成本高昂、依赖于特定模型，且难以扩展到更大的架构。我们提出了可微忠实性对齐（DFA），这是一个通过学习到的可微对齐，将电路信息从较小的源模型迁移到较大目标模型的框架。DFA 将源模型的节点重要性分数投影到目标模型中，并以软忠实性目标训练这一映射，从而避免了在目标模型上进行完整的电路发现。我们在 Llama-3 和 Qwen-2.5 上，针对涵盖事实检索、多项选择推理和算术的六项任务对 DFA 进行了评估。最强的结果出现在从 Llama-3 1B 迁移到 3B 时，此时对齐后的电路通常与直接节点归因相当，且零样本迁移仍然有效。对于更大的模型，恢复效果会有所减弱（原文摘要在此处截断）。

    arXiv:2604.24302v2 Announce Type: replace  Abstract: Mechanistic interpretability has made it possible to localize circuits underlying specific behaviors in language models, but existing methods are expensive, model-specific, and difficult to scale to larger architectures. We introduce Differentiable Faithfulness Alignment (DFA), a framework that transfers circuit information from a smaller source model to a larger target model through a learned differentiable alignment. DFA projects source-model node importance scores into the target model and trains this mapping with a soft faithfulness objective, avoiding full circuit discovery on the target model. We evaluate DFA on Llama-3 and Qwen-2.5 across six tasks spanning factual retrieval, multiple-choice reasoning, and arithmetic. The strongest results occur when transferring from Llama-3 1B to 3B, where aligned circuits are often competitive with direct node attribution and zero-shot transfer remains effective. Recovery weakens for larger
    
[^303]: 当思维链失败时，解决方案隐藏在隐藏状态之中

    When Chain-of-Thought Fails, the Solution Hides in the Hidden States

    [https://arxiv.org/abs/2604.23351](https://arxiv.org/abs/2604.23351)

    研究发现，即使思维链推理轨迹本身是错误的，其隐藏状态（尤其集中于中后层和轨迹早期）仍编码了足以恢复正确答案的任务相关信息，通过激活修补将这些隐藏状态注入直接回答过程可显著提升答题准确率。

    

    arXiv:2604.23351v2 公告类型：replace-cross。摘要：中间推理在计算上究竟是有用的，还是仅仅起解释作用，取决于思维链（CoT）标记中是否包含与任务相关的信息。我们利用激活修补（activation patching）对 GSM8K 上的思维链进行了机制层面的因果分析：将同一问题在思维链生成中得到的标记级隐藏状态转移到直接回答的运行中，然后测量其对最终答案准确率的影响。在各个模型上，修补后生成答案的准确率显著高于直接回答提示和原始思维链轨迹，这表明即使原始推理轨迹是错误的，单个思维链标记也能编码足以恢复正确答案的信息。这种与任务相关的信息在正确的思维链运行中比在错误的运行中更为普遍，并且在各标记之间分布不均：它集中于中间至较深的层，并在推理轨迹的较早位置出现。此外，修补语言标记，例如……（原文摘要在此处截断）

    arXiv:2604.23351v2 Announce Type: replace-cross  Abstract: Whether intermediate reasoning is computationally useful or merely explanatory depends on whether chain-of-thought (CoT) tokens contain task-relevant information. We present a mechanistic causal analysis of CoT on GSM8K using activation patching: transferring token-level hidden states from a CoT generation to a direct-answer run for the same question, then measuring the effect on final-answer accuracy. Across models, generating after patching yields substantially higher accuracy than both direct-answer prompting and the original CoT trace, revealing that individual CoT tokens can encode sufficient information to recover the correct answer, even when the original trace is incorrect. This task-relevant information is more prevalent in correct than incorrect CoT runs and is unevenly distributed across tokens, concentrating in mid-to-late layers and appearing earlier in the reasoning trace. Moreover, patching language tokens such a
    
[^304]: BAGEL：语言模型中的动物知识专长基准测试

    BAGEL: Benchmarking Animal Knowledge Expertise in Language Models

    [https://arxiv.org/abs/2604.16241](https://arxiv.org/abs/2604.16241)

    BAGEL是一个基于维基百科、GloBI和bioRxiv的6,822题动物知识选择题基准，评估发现前沿大模型尽管在百科和文献类题目上表现出色，但在生态相互作用等参与式数据推理上仍明显薄弱。

    

    arXiv:2604.16241v2 公告类型：replace-cross 摘要：大型语言模型在广泛的知识基准测试中表现良好，然而在统一的评估协议下，其在动物领域的专业能力仍缺乏充分表征。我们提出了BAGEL（语言模型动物知识专长基准测试），这是一个包含6,822道选择题的基准测试，题目来源于维基百科、全球生物相互作用数据库（GloBI）以及bioRxiv。该基准涵盖百科全书式的物种知识、生态相互作用推理，以及对动物主题预印本的解读。题目通过自动化方式生成，并由GPT-5.5依据一套基于150个专家审核条目校准的评分标准进行筛选；这一过程并非详尽的人工裁决。我们在不使用外部检索的条件下评估了14个模型，仅依靠模型参数和题目局部证据。准确率因题目来源和类型不同而呈现显著差异：在维基百科和bioRxiv任务上表现出色的前沿模型，在GloBI参与式数据（题目）方面表现相对较弱。

    arXiv:2604.16241v2 Announce Type: replace-cross  Abstract: Large language models perform well on broad knowledge benchmarks, yet their animal-domain competence remains poorly characterized under a consistent protocol. We introduce BAGEL (\textbf{B}enchmarking \textbf{A}nimal knowled\textbf{G}e \textbf{E}xpertise in \textbf{L}anguage models), a 6,822-question multiple-choice benchmark drawn from Wikipedia, Global Biotic Interactions (GloBI), and bioRxiv. It spans encyclopedic species knowledge, ecological interaction inference, and interpretation of animal-focused preprints. Questions are generated automatically and screened by GPT-5.5 using a rubric calibrated on 150 expert-reviewed items; this process is not exhaustive human adjudication. We evaluate 14 models without external retrieval, using model parameters and question-local evidence. Accuracy varies sharply by source and question type: frontier models that excel on Wikipedia and bioRxiv remain comparatively weak on GloBI particip
    
[^305]: 评估用于医学问答的小型开源大语言模型：一个实用框架

    Evaluating Small Open LLMs for Medical Question Answering: A Practical Framework

    [https://arxiv.org/abs/2604.10535](https://arxiv.org/abs/2604.10535)

    该论文提出了一个开源评估框架，将可复现性与准确性并列为首要指标，通过八项质量指标和重复推理来评估可本地部署的小型开源LLM在医学问答中的表现。

    

    将大语言模型（LLM）应用于医学问答，所需的不只是较高的平均准确率：一个每次查询都返回实质性不同答案的模型并不是可靠的医疗工具。Reddit等在线健康社区已成为数百万用户获取医学信息的主要来源，但这些社区仍然极易受到错误信息的影响；在这些场景中部署LLM作为助手，进一步放大了在保证正确性的同时对输出一致性的需求。我们提出了一个实用的、开源的评估框架，用于评估小型、可本地部署的开源权重LLM在医学问答上的表现，并将可复现性作为与词汇和语义准确性同等重要的首要指标。我们的流水线计算八项质量指标，包括BERTScore、ROUGE-L以及LLM作为评判的评分标准，此外还包括从重复推理（N=10次运行）中得出的两项模型内可复现性指标

    arXiv:2604.10535v2 Announce Type: replace-cross  Abstract: Incorporating large language models (LLMs) in medical question answering demands more than high average accuracy: a model that returns substantively different answers each time it is queried is not a reliable medical tool. Online health communities such as Reddit have become a primary source of medical information for millions of users, yet they remain highly susceptible to misinformation; deploying LLMs as assistants in these settings amplifies the need for output consistency alongside correctness. We present a practical, open-source evaluation framework for assessing small, locally-deployable open-weight LLMs on medical question answering, treating reproducibility as a first-class metric alongside lexical and semantic accuracy. Our pipeline computes eight quality metrics, including BERTScore, ROUGE-L, and an LLM-as-judge rubric, together with two within-model reproducibility metrics derived from repeated inference (N=10 runs 
    
[^306]: 分解、观察与推理：面向视觉语言模型的强化潜在推理

    Decompose, Look, and Reason: Reinforced Latent Reasoning for VLMs

    [https://arxiv.org/abs/2604.07518](https://arxiv.org/abs/2604.07518)

    本文提出DLR强化潜在推理框架，通过将查询动态分解为文本前提、提取条件化的连续视觉潜在表示，并配合三阶段训练流程与新颖的球面高斯潜在策略，显著提升了视觉语言模型多步视觉推理的性能与逐步可解释性。

    

    视觉语言模型由于文本思维链中存在视觉信息丢失，常常难以处理复杂的视觉推理任务。现有方法要么增加了工具调用的成本，要么依赖于局部化的基于图像块的嵌入，这些嵌入不足以在多步推理中提取语义。我们提出了“分解、观察与推理”（DLR），这是一个强化潜在推理框架，能够动态地将查询分解为文本前提，提取以前提为条件的连续视觉潜在表示，并通过有依据的推理链推导答案。我们引入了一个三阶段训练流程，并提出了新颖的球面高斯潜在策略，以实现潜在空间中的有效探索。在以视觉为中心的基准测试上的大量实验表明，DLR 持续优于强大的基线方法，包括纯文本方法、交错多模态思维链和潜在推理方法，同时提供了卓越的逐步可解释性。

    arXiv:2604.07518v2 Announce Type: replace  Abstract: Vision-Language Models often struggle with complex visual reasoning due to the visual information loss in textual CoT. Existing methods either add the cost of tool calls or rely on localized patch-based embeddings that are insufficient to extract semantics in multi-step reasoning. We propose "Decompose, Look, and Reason" (DLR), a reinforced latent reasoning framework that dynamically decomposes queries into textual premises, extracts premise-conditioned continuous visual latents, and deduces answers through grounded rationales. We introduce a three-stage training pipeline and propose a novel Spherical Gaussian Latent Policy, to enable effective exploration in the latent space. Extensive experiments on vision-centric benchmarks show that DLR consistently outperforms strong baselines, including text-only, interleaved multimodal CoT, and latent reasoning methods, while providing superior stepwise interpretability.
    
[^307]: ClickGuard：一个用于标题党检测的可信自适应融合框架

    ClickGuard: A Trustworthy Adaptive Fusion Framework for Clickbait Detection

    [https://arxiv.org/abs/2604.07272](https://arxiv.org/abs/2604.07272)

    提出了ClickGuard可信自适应融合框架，通过句法-语义自适应融合块动态整合BERT嵌入与结构特征，并结合混合CNN-BiLSTM捕获文本模式，实现96.93%准确率的标题党检测，同时利用LIME和PFI确保模型的可信度与可解释性。

    

    标题党标题被精心设计以误导读者并最大化用户参与度，其广泛使用对网络可信度构成了重大挑战。这些标题采用煽情手法、误导性陈述和模糊语言，凸显了有效检测以确保可信数字内容的必要性。本文提出了ClickGuard：一个用于标题党检测的可信自适应融合框架。该框架结合了BERT嵌入和结构特征，通过句法-语义自适应融合块（SSAFB）进行动态集成，并融合了混合CNN-BiLSTM以捕获文本模式和依赖关系。该模型达到了96.93%的测试准确率，超越了最先进的方法。模型的可信度通过LIME和置换特征重要性（PFI）进行可解释性和扰动分析评估，这些方法通过测量特征变化来评估模型的鲁棒性和敏感性。

    arXiv:2604.07272v2 Announce Type: replace  Abstract: The widespread use of clickbait headlines, crafted to mislead and maximize engagement, poses a significant challenge to online credibility. These headlines employ sensationalism, misleading claims, and vague language, underscoring the need for effective detection to ensure trustworthy digital content. The paper introduces, ClickGuard: a trustworthy adaptive fusion framework for clickbait detection. It combines BERT embeddings and structural features using a Syntactic-Semantic Adaptive Fusion Block (SSAFB) for dynamic integration. The framework incorporates a hybrid CNN-BiLSTM to capture patterns and dependencies. The model achieved 96.93% testing accuracy, outperforming state-of-the-art approaches. The model's trustworthiness is evaluated using LIME and Permutation Feature Importance (PFI) for interpretability and perturbation analysis. These methods assess the model's robustness and sensitivity to feature changes by measuring the av
    
[^308]: 过度自信的内在线路：大语言模型中言语化自信膨胀的机制视角

    Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs

    [https://arxiv.org/abs/2604.01457](https://arxiv.org/abs/2604.01457)

    该论文通过电路级机制分析发现，大语言模型的言语化过度自信由集中在中间至后期层的一小组MLP模块和注意力头在最终token位置写入，并可据此实现推理时的针对性重新校准。

    

    大语言模型往往不仅是错误的，而且是“自信地错误”：当它们给出事实性不正确的答案时，它们倾向于口头表达过高的自信，而不是发出不确定性的信号。这种言语化的过度自信会误导用户，并削弱置信分数作为可靠不确定性信号的作用，然而其内部机制至今仍知之甚少。我们对大语言模型中这种膨胀的言语化自信进行了电路级别的机制分析，围绕三个方向展开：将言语化自信捕获为可微分的内部信号、识别出因果性地使其膨胀的电路，以及利用这些洞察实现有针对性的推理时重新校准。在两个指令微调的大语言模型和三个数据集上，我们发现一组紧凑的MLP模块和注意力头——集中在中间至后期层——在最终token位置一致地写入自信膨胀信号。我们进一步表明，对……

    arXiv:2604.01457v4 Announce Type: replace  Abstract: Large language models are often not just wrong, but \emph{confidently wrong}: when they produce factually incorrect answers, they tend to verbalize overly high confidence rather than signal uncertainty. Such verbalized overconfidence can mislead users and weaken confidence scores as a reliable uncertainty signal, yet its internal mechanisms remain poorly understood. We present a circuit-level mechanistic analysis of this inflated verbalized confidence in LLMs, organized around three axes: capturing verbalized confidence as a differentiable internal signal, identifying the circuits that causally inflate it, and leveraging these insights for targeted inference-time recalibration. Across two instruction-tuned LLMs on three datasets, we find that a compact set of MLP blocks and attention heads, concentrated in middle-to-late layers, consistently writes the confidence-inflation signal at the final token position. We further show that targ
    
[^309]: RAM-Net：基于稀疏可寻址状态的线性时间序列建模

    RAM-Net: Linear-Time Sequence Modeling with Sparsely Addressable State

    [https://arxiv.org/abs/2602.11958](https://arxiv.org/abs/2602.11958)

    RAM-Net提出用稀疏地址访问取代共享状态的密集访问，将循环状态组织为独立槽数组并通过地址解码器选择少量槽进行读写，从而抑制词元间干扰、提升长距离细粒度记忆能力，同时保持线性时间复杂度。

    

    线性注意力通过固定大小的循环状态，为全注意力提供了一种高效的替代方案。然而，这一状态由所有词元共享，来自不同词元的信息会在其中叠加，产生词元间的相互干扰，从而损害长距离细粒度记忆能力。为解决这一问题，我们提出了RAM-Net，它用基于地址的稀疏访问取代了对共享状态的密集访问。RAM-Net将循环状态组织为固定大小的独立槽数组，并使用一个地址解码器将每个键或查询映射为稀疏地址，在每一步中选择一小部分槽进行写入或读取。这种设计将地址不重叠的词元导向互不相交的槽，抑制了词元间的干扰，同时使每步状态访问的开销仅取决于所选槽的数量，而非状态的总大小。实验表明，RAM-Net优于强大的循环基线模型。

    arXiv:2602.11958v2 Announce Type: replace-cross  Abstract: Linear attention offers an efficient alternative to full attention with a fixed-size recurrent state. However, this state is shared by all tokens, so information from distinct tokens becomes superposed within it and produces inter-token interference that degrades long-range fine-grained recall. To address this issue, we propose RAM-Net, which replaces dense access to a shared state with sparse address-based access. RAM-Net organizes the recurrent state as a fixed-size array of independent slots and uses an Address Decoder that maps each key or query into a sparse address, selecting a small subset of slots to write to or read from at each step. This design directs tokens with non-overlapping addresses to disjoint slots, suppressing inter-token interference, while keeping per-step state access dependent only on the number of selected slots rather than the total state size. Empirically, RAM-Net outperforms strong recurrent baselin
    
[^310]: 预训练数据的词汇印记

    The Lexical Imprint of Pre-training Data

    [https://arxiv.org/abs/2602.10657](https://arxiv.org/abs/2602.10657)

    本文提出词级一元交叉熵（UCE）这一与分词器无关、无需模型的词汇对齐度量，并通过受控的从零预训练实验证明，预训练语料库与基准测试之间的词汇对齐程度能够持续解释零样本基准准确率。

    

    预训练语料库与基准测试目标之间的词级对齐，能在多大程度上解释零样本基准测试的准确率？我们通过词级一元交叉熵来研究这个问题。这是一种与分词器无关、无需模型参与的度量方法，用于衡量参考语料库与目标基准之间的词汇对齐程度。似然训练基于预训练语料库来塑造模型的预测分布，而在基于似然的评估中，零样本得分部分取决于该语料库与基准目标之间的词汇对齐。要清晰地追踪预训练数据的词汇印记如何随语料库与基准的对齐程度而变化，需要进行受控的从零预训练实验，即在保持训练方案固定不变的情况下，改变所使用的语料库。在这一受控扫描实验中，我们涵盖了10个零样本基准、4个预训练语料库和3个模型规模，发现语料库与基准之间的词汇对齐程度（即UCE）始终……（原文摘要到此截断）

    arXiv:2602.10657v2 Announce Type: replace  Abstract: How much of zero-shot benchmark accuracy is captured by word-level alignment between a pre-training corpus and the benchmark target? We study this question through word-level unigram cross-entropy (UCE), a tokenizer-agnostic, model-free measure of lexical alignment between a reference corpus and a target benchmark. Likelihood training shapes the model's predictive distribution from the pre-training corpus, and zero-shot scores under likelihood-based evaluation depend in part on the lexical alignment between that corpus and the benchmark target. Cleanly tracing how the lexical imprint of pre-training data scales with corpus--benchmark alignment requires controlled from-scratch pre-training that varies corpus identity while holding the training recipe fixed. Across this controlled sweep, spanning $10$ zero-shot benchmarks, $4$ pre-training corpora, and $3$ model scales, we find that corpus--benchmark lexical alignment, UCE, consistentl
    
[^311]: 面向自适应、可扩展且鲁棒的大语言模型智能体协调：动态自组网络的视角

    Towards Adaptive, Scalable, and Robust Coordination of LLM Agents: A Dynamic Ad-Hoc Networking Perspective

    [https://arxiv.org/abs/2602.08009](https://arxiv.org/abs/2602.08009)

    该论文借鉴动态自组网络的经典问题，提出了RAPS——一种声誉感知的发布-订阅范式，使大语言模型智能体能够基于声明的意图而非预定义拓扑进行自适应、可扩展且鲁棒的协调通信。

    

    基于大语言模型（LLM）构建的多智能体架构已经展示了通过精心设计的协作实现群体智能的潜力。然而，繁重的人工编排负担从根本上要求对智能体工作流进行自动化设计。我们将这一智能体协调挑战归结为动态自组网络中的一个经典问题：如何在数量可扩展的智能体主机之间建立自适应且可靠的通信？针对这一尚未解决的难题，我们提出了RAPS——一种具备声誉感知的发布-订阅范式，用于实现大语言模型智能体的自适应、可扩展且鲁棒的协调。RAPS建立在分布式发布-订阅协议的基础之上，使大语言模型智能体能够基于其声明的意图而非预定义的拓扑结构来交换消息。在此基础之上，RAPS进一步整合了两个协同的覆盖层：（一）响应式订阅，使智能体能够动态……

    arXiv:2602.08009v2 Announce Type: replace  Abstract: Multi-agent architectures built on large language models (LLMs) have demonstrated the potential to realize swarm intelligence through well-crafted collaboration. However, the substantial burden of manual orchestration inherently raises an imperative to automate the design of agentic workflows. We frame such an agent coordination challenge as a classic problem in dynamic ad-hoc networking: How to establish adaptive and reliable communication among a scalable number of agentic hosts? In response to this unresolved dilemma, we introduce RAPS, a reputation-aware publish-subscribe paradigm for adaptive, scalable, and robust coordination of LLM agents. RAPS is grounded in the Distributed Publish-Subscribe Protocol, allowing LLM agents to exchange messages based on their declared intents rather than predefined topologies. Beyond this substrate, RAPS further incorporates two coherent overlays: (i) Reactive Subscription, enabling agents to dy
    
[^312]: 超越遗忘：表征误导引发可控的附带行为与能力

    Beyond Forgetting: Representation Misdirection Elicits Controllable Side Behaviors and Capabilities

    [https://arxiv.org/abs/2601.21702](https://arxiv.org/abs/2601.21702)

    该论文提出，基于线性表征假设的视角，表征误导（RM）类机器遗忘方法不仅能实现遗忘，还可通过在遗忘表征空间中对高层概念向量进行线性操作，可控地引发与该概念相对应的附带行为与能力。

    

    我们研究表征误导（Representation Misdirection, RM），这是一类大语言模型（LLM）遗忘方法，其通过将待遗忘样本的潜在表征重定向至某个目标向量来实现遗忘。尽管十分重要，但RM中所使用的目标向量的作用仍未得到充分探索。本文从线性表征假设（Linear Representation Hypothesis）的视角出发，重新审视并研究RM。具体而言，如果能够识别出与某一高层概念相对应的一维表征，线性表征假设便允许在遗忘表征空间内对该概念向量进行线性操作。基于这一视角，我们假设：除遗忘之外，通过RM实现的机器遗忘还会引发与该高层概念相对应的、可控的附带行为与能力。我们的假设在广泛的概念与任务上得到了实证验证，包括控制已遗忘模型的（原文摘要在此处截断）

    arXiv:2601.21702v4 Announce Type: replace-cross  Abstract: We consider Representation Misdirection (RM), a class of large language model (LLM) unlearning methods that achieve forgetting by redirecting the latent representations of forget-samples toward a target vector. Despite being important, the roles of the target vector used in RM, however, remain underexplored. Here, we approach and revisit RM through the lens of the Linear Representation Hypothesis. Specifically, if one can identify a one-dimensional representation corresponding to a high-level concept, the Linear Representation Hypothesis enables linear operations on this concept vector within the forget-representation space. Under this view, we hypothesize that, beyond forgetting, machine unlearning via RM elicits controllable side effect behaviors and capabilities corresponding to the high-level concept. Our hypothesis is empirically validated across a wide range of concepts and tasks, including controlling unlearned models' t
    
[^313]: BMAM：类脑多智能体记忆框架

    BMAM: Brain-inspired Multi-Agent Memory Framework

    [https://arxiv.org/abs/2601.20465](https://arxiv.org/abs/2601.20465)

    BMAM提出了一种受大脑启发的多智能体记忆框架，将记忆分解为情景、语义、显著性和控制导向等功能特化子系统并组织为六阶段记忆生命周期，在LoCoMo基准上以78.45%的准确率超越了七个记忆增强基线。

    

    基于语言模型的智能体在长时程交互场景下，面临着保持时间锚定信息和维持跨会话行为一致性的持续挑战，我们将这种失效模式称为“灵魂侵蚀”。我们提出了BMAM（类脑多智能体记忆），这是一种通用记忆架构，它将智能体记忆建模为一组功能特化的子系统，而非单一的非结构化存储。受认知记忆系统的启发，BMAM将记忆分解为情景记忆、语义记忆、显著性感知和控制导向等组件，这些组件在互补的时间尺度上运行，并被组织为一个六阶段的记忆生命周期。为支持长时程推理，BMAM沿显式时间线组织情景记忆，并通过融合多个互补信号来检索证据。在LoCoMo基准上的实验表明，BMAM达到了78.45%的准确率，优于七个记忆增强基线方法。

    arXiv:2601.20465v2 Announce Type: replace  Abstract: Language-model-based agents operating over extended interaction horizons face persistent challenges in preserving temporally grounded information and maintaining behavioral consistency across sessions, a failure mode we term soul erosion. We present BMAM (Brain-inspired Multi-Agent Memory), a general-purpose memory architecture that models agent memory as a set of functionally specialized subsystems rather than a single unstructured store. Inspired by cognitive memory systems, BMAM decomposes memory into episodic, semantic, salience-aware, and control-oriented components that operate at complementary time scales, organised as a six-phase memory lifecycle. To support long-horizon reasoning, BMAM organises episodic memories along explicit timelines and retrieves evidence by fusing multiple complementary signals. Experiments on the LoCoMo benchmark show that BMAM achieves 78.45% accuracy, outperforming seven memory-augmented baselines. 
    
[^314]: 内在多语言评估中的形式与意义

    Form and Meaning in Intrinsic Multilingual Evaluations

    [https://arxiv.org/abs/2601.10580](https://arxiv.org/abs/2601.10580)

    本文通过实验揭示，基于困惑度等指标的多语言内在评估依赖于信息内容等同的有问题假设，证明这些指标在多语言环境中并不普遍可比，并借助形式与意义之争为其提供了解释。

    

    条件语言模型的内在评估指标，如困惑度或每字符比特数，在单语和多语言环境中都被广泛使用。这些指标在单语环境中的使用和比较相当直接，但在多语言环境中则依赖于若干假设。其中一个假设是，比较条件语言模型在平行句子上的困惑度可以反映其质量，因为信息内容（此处理解为语义含义）是相同的。然而，这些指标本质上衡量的是信息论意义上的信息内容。我们明确了这一假设及其他类似假设，并讨论了它们的影响。我们使用六种指标，在单语和多语言模型上对两个多平行语料库进行了实验。最终，我们发现当前的指标并不具有普遍可比性。我们从形式与意义之争的角度为这一现象提供了一些解释。

    arXiv:2601.10580v2 Announce Type: replace  Abstract: Intrinsic evaluation metrics for conditional language models, such as perplexity or bits-per-character, are widely used in both mono- and multilingual settings. These metrics are rather straightforward to use and compare in monolingual setups, but rest on a number of assumptions in multilingual setups. One such assumption is that comparing the perplexity of CLMs on parallel sentences is indicative of their quality since the information content (here understood as the semantic meaning) is the same. However, the metrics are inherently measuring information content in the information-theoretic sense. We make this and other such assumptions explicit and discuss their implications. We perform experiments with six metrics on two multi-parallel corpora both with mono- and multilingual models. Ultimately, we find that current metrics are not universally comparable. We look at the form-meaning debate to provide some explanation for this.
    
[^315]: 渐近通用对齐：一种基于测试时扩展的新型对齐框架

    Asymptotic Universal Alignment: A New Alignment Framework via Test-Time Scaling

    [https://arxiv.org/abs/2601.08777](https://arxiv.org/abs/2601.08777)

    该论文提出通过测试时扩展实现“渐近通用对齐”的新框架，证明生成 k 个候选回答的乘积策略能以最优且不可超越的速率 f(k)=k/(k+1) 逼近完美胜率，为处理用户偏好冲突提供了理论基础。

    

    让大型语言模型（LLM）能够服务于具有异构且可能相互冲突偏好的用户，是个性化与可信人工智能面临的核心挑战。我们通过测试时扩展形式化了一种理想化的通用对齐概念：对于每个提示，模型生成 k≥1 个候选回答，用户从中选择自己最喜欢的一个。我们提出了 (k, f(k))-鲁棒对齐，即要求该 k 输出模型相对于任何其他单输出模型具有 f(k) 的胜率；并进一步提出渐近通用对齐（U-alignment），即要求当 k→∞ 时 f(k)→1。我们的主要结果刻画了最优收敛速率：存在一族单输出策略，其 k 样本乘积策略能够以 f(k)=k/(k+1) 的速率实现 U-alignment，而且在一般情况下没有任何方法能够达到更快的速率。我们证明，包括从人类反馈中进行纳什学习（NLHF）在内的主流后训练方法可以……

    arXiv:2601.08777v2 Announce Type: replace-cross  Abstract: Aligning large language models (LLMs) to serve users with heterogeneous and potentially conflicting preferences is a central challenge for personalized and trustworthy AI. We formalize an ideal notion of universal alignment through test-time scaling: for each prompt, the model produces $k\ge 1$ candidate responses and a user selects their preferred one. We introduce $(k,f(k))$-robust alignment, which requires the $k$-output model to have win rate $f(k)$ against any other single-output model, and asymptotic universal alignment (U-alignment), which requires $f(k)\to 1$ as $k\to\infty$. Our main result characterizes the optimal convergence rate: there exists a family of single-output policies whose $k$-sample product policies achieve U-alignment at rate $f(k)=\frac{k}{k+1}$, and no method can achieve a faster rate in general.   We show that popular post-training methods, including Nash learning from human feedback (NLHF), can fund
    
[^316]: What If TSF：评估场景引导的多模态时间序列预测

    What If TSF: Evaluating Scenario-Guided Multimodal Time Series Forecasting

    [https://arxiv.org/abs/2601.08509](https://arxiv.org/abs/2601.08509)

    该论文提出了WIT基准，通过固定预测状态并系统变化未来场景构建受控场景集，以评估多模态时间序列预测模型能否恰当响应未来场景信息。

    

    近期大语言模型的进展使时间序列预测得以超越纯数值观测，在多模态设置中融入外部信息。这类信息可以提升预测性能，但仅凭准确率可能无法揭示模型是否恰当地响应了这些信息：模型可能忽略相关信息、无法区分具有不同含义的场景，或对无关或微弱信号过度反应。我们提出了 What If TSF（WIT），一个用于评估模型是否有效响应未来场景的基准。WIT 通过固定预测状态的同时系统地变化未来场景，构建受控场景集，从而支持三种互补的评估：事实比较，衡量事实场景信息的预测效用；关系比较，评估预测是否满足预期的方向、对比、强度与克制等关系。

    arXiv:2601.08509v3 Announce Type: replace  Abstract: Recent advances in large language models (LLMs) have enabled time series forecasting to move beyond numerical observations and incorporate external information in multimodal settings. Such information can improve forecasting performance, but accuracy alone may not reveal whether models appropriately respond to it: a model may ignore relevant information, fail to distinguish scenarios with different implications, or overreact to irrelevant or weak signals. We introduce What If TSF (WIT), a benchmark for evaluating whether models effectively respond to future scenarios. WIT constructs controlled scenario sets by fixing the forecasting state while systematically varying future scenarios, enabling three complementary evaluations: Factual Comparison, which measures the predictive utility of factual scenario information; Relational Comparison, which evaluates whether forecasts satisfy expected direction, contrast, intensity, and restraint 
    
[^317]: 大型语言模型在持续预训练中如何学习概念？

    How Do Large Language Models Learn Concepts During Continual Pre-Training?

    [https://arxiv.org/abs/2601.03570](https://arxiv.org/abs/2601.03570)

    该论文通过分析LLMs内部概念电路的结构和动态，揭示了持续预训练中概念获取、遗忘及多概念交互的机制，并发现电路拓扑能一致预测学习与遗忘行为。

    

    arXiv:2601.03570v2 公告类型：替换 摘要：人类主要通过概念（例如，狗）来理解世界，这些抽象的心理表征构建了感知、推理和学习。然而，大型语言模型（LLMs）在持续预训练期间如何获取、保留和遗忘此类概念仍知之甚少。在这项工作中，我们研究了个体概念是如何被获取和遗忘的，以及多个概念如何通过干扰和协同作用相互作用。我们将这些行为动态与LLMs的内部概念电路（与特定概念相关的计算子图）联系起来，并结合图指标来表征电路拓扑。我们的分析揭示了：（1）LLMs的概念电路提供了概念学习和遗忘的非平凡、一致的信号；（2）在持续预训练期间，概念电路表现出阶段性时间模式，先早期增加，随后逐渐减少并稳定；（3）具有较大规模的概念表现出更强的记忆持久性。

    arXiv:2601.03570v2 Announce Type: replace  Abstract: Human beings primarily understand the world through concepts (e.g., dog), abstract mental representations that structure perception, reasoning, and learning. However, how large language models (LLMs) acquire, retain, and forget such concepts during continual pretraining remains poorly understood. In this work, we study how individual concepts are acquired and forgotten, as well as how multiple concepts interact through interference and synergy. We link these behavioral dynamics to LLMs' internal concept circuits, computational subgraphs associated with specific concepts, and incorporate graph metrics to characterize circuit topology. Our analysis reveals: (1) LLMs concept circuits provide a non-trivial, consistent signal of concept learning and forgetting; (2) concept circuits exhibit a stage-wise temporal pattern during continual pretraining, with an early increase followed by gradual decrease and stabilization; (3) concepts with la
    
[^318]: HyperCLOVA X 32B Think

    HyperCLOVA X 32B Think

    [https://arxiv.org/abs/2601.03286](https://arxiv.org/abs/2601.03286)

    HyperCLOVA X 32B Think是一个专注于韩语语言文化语境推理与智能体能力的开源视觉-语言模型，在韩语基准测试和智能体评估任务上表现优异。

    

    在本报告中，我们提出了HyperCLOVA X 32B Think，这是一个视觉-语言模型，特别侧重于韩语语言和文化背景下的推理能力以及智能体能力。HyperCLOVA X 32B Think在预训练阶段高度专注于推理能力，随后通过后训练来支持多模态理解、增强推理、智能体行为以及与人类偏好的对齐。针对同等规模模型的实验评估表明，我们的模型在韩语文本到文本和视觉到文本基准测试以及面向智能体的评估任务上取得了强劲的表现。通过开源HyperCLOVA X 32B Think，我们旨在支持更广泛的应用，并促进学术和工业界的进一步研究与创新。

    arXiv:2601.03286v2 Announce Type: replace-cross  Abstract: In this report, we present HyperCLOVA X 32B Think, a vision-language model designed with particular emphasis on reasoning within the Korean linguistic and cultural context, as well as agentic ability. HyperCLOVA X 32B Think is pre-trained with a strong focus on reasoning capabilities and subsequently post-trained to support multimodal understanding, enhanced reasoning, agentic behaviors, and alignment with human preferences. Experimental evaluations against comparably sized models demonstrate that our model achieves strong performance on Korean text-to-text and vision-to-text benchmarks, as well as on agent-oriented evaluation tasks. By open-sourcing HyperCLOVA X 32B Think, we aim to support broader adoption and facilitate further research and innovation across both academic and industrial communities.
    
[^319]: FlowPlan-G2P：一种将科学论文转化为专利说明书的结构化生成框架

    FlowPlan-G2P: A Structured Generation Framework for Transforming Scientific Papers into Patent Descriptions

    [https://arxiv.org/abs/2601.02589](https://arxiv.org/abs/2601.02589)

    提出FlowPlan-G2P框架，通过构建发明要素与功能关系的有向概念图并将其划分为有序分节子图，将科学论文自动转化为法律合规的专利说明书，专家法律合规评分达4.7/5，远超零样本方法的1.5分。

    

    研究人员通常以科学论文的形式发表发明，然而将其转化为专利说明书仍然困难重重，因为法定的充分公开要求严格且符合法律结构的披露。现有的专利撰写方法要么假设输入是专利原生格式（如权利要求），要么通过局部检索从大纲片段生成分节内容，而缺乏对技术依赖关系的显式文档级表示。我们提出了FlowPlan-G2P，一个以图为中介的框架，它（1）归纳出发明要素及功能关系的有向概念图，（2）依据连通性、同质性和领域约束惩罚对图进行评分，并将其划分为有序的分节子图，（3）使用针对各分节的特定撰写策略来综合生成专利说明书。在Pap2Pat-EvalGold基准上，零样本方法取得了最高的BERTScore（0.87），但专家法律合规性评分仅为5分制中的1.5分，而FlowPlan-G2P则达到4.7分（摘要原文在此处截断）。

    arXiv:2601.02589v5 Announce Type: replace-cross  Abstract: Researchers routinely publish inventions as scientific papers, yet turning them into patent descriptions remains difficult because statutory enablement demands rigid, legally structured disclosure. Prior patent drafting methods either assume patent-native inputs such as claims or generate sections from outline chunks via local retrieval without an explicit document-level representation of technical dependencies. We propose FlowPlan-G2P, a graph-mediated framework that (1) induces a directed concept graph of invention elements and functional relations, (2) partitions it into ordered section subgraphs scored by connectivity, homogeneity, and domain-constraint penalties, and (3) synthesizes patent descriptions using section-specific drafting strategies. On Pap2Pat-EvalGold, Zero-Shot achieves the highest BERTScore at 0.87 yet receives only 1.5 out of 5 on expert legal compliance compared to 4.7 for FlowPlan-G2P, where lexical over
    
[^320]: HyperCLOVA X 8B 全模态模型（HyperCLOVA X 8B Omni）

    HyperCLOVA X 8B Omni

    [https://arxiv.org/abs/2601.01792](https://arxiv.org/abs/2601.01792)

    HyperCLOVA X 8B Omni 是 HyperCLOVA X 系列首个支持文本、音频和视觉任意输入输出的统一全模态模型，通过共享的下一词预测接口整合多模态理解与生成，在韩英双语下达到与同规模模型相当的性能并计划开源权重。

    

    在本报告中，我们提出了 HyperCLOVA X 8B Omni，这是 HyperCLOVA X 系列中首个支持文本、音频和视觉同时作为输入与输出的任意到任意全模态模型。通过将多模态理解与生成整合进单一模型，而非采用针对特定模态的分离式流水线，HyperCLOVA X 8B Omni 作为 8B 规模的全模态探索起点，为构建实用的任意到任意全能助手指明方向。在总体设计上，该模型通过对交错的模态序列共享下一词预测接口来统一各模态，同时由视觉和音频编码器注入连续嵌入，以实现细粒度的理解与定位。实证评估表明，在涵盖文本、音频和视觉的多种输入输出组合下，该模型在韩语和英语两种语言中均展现出与同规模模型相比具有竞争力的性能。我们预计 HyperCLOVA X 8B Omni 的开源权重发布将为广泛的研究提供支持。

    arXiv:2601.01792v2 Announce Type: replace-cross  Abstract: In this report, we present HyperCLOVA X 8B Omni, the first any-to-any omnimodal model in the HyperCLOVA X family that supports text, audio, and vision as both inputs and outputs. By consolidating multimodal understanding and generation into a single model rather than separate modality-specific pipelines, HyperCLOVA X 8B Omni serves as an 8B-scale omni-pathfinding point toward practical any-to-any omni assistants. At a high level, the model unifies modalities through a shared next-token prediction interface over an interleaved multimodal sequence, while vision and audio encoders inject continuous embeddings for fine-grained understanding and grounding. Empirical evaluations demonstrate competitive performance against comparably sized models across diverse input-output combinations spanning text, audio, and vision, in both Korean and English. We anticipate that the open-weight release of HyperCLOVA X 8B Omni will support a wide r
    
[^321]: Twitter上英语拼写变体的社会感知：人类与大语言模型回应的比较分析

    Social Perceptions of English Spelling Variation on Twitter: A Comparative Analysis of Human and LLM Responses

    [https://arxiv.org/abs/2511.23041](https://arxiv.org/abs/2511.23041)

    本研究基于社会语言学方法比较人类与大语言模型对英语拼写变体社会属性的评分，发现两者总体高度相关，但在评分分布和不同拼写变体类型上存在显著差异。

    

    拼写变体（例如 funnnn 与 fun）会影响人们对文本及其作者的社会感知：我们常常对不同的书写形式抱有各种联想（文本是否非正式？作者看起来是否年轻？）。本研究聚焦于英语在线写作中拼写变体的社会感知，探讨这种感知在人类与大语言模型（LLM）之间的一致程度。基于社会语言学方法论，我们比较了LLM与人类对拼写变体三个关键社会属性（正式度、谨慎度、年龄）的评分。我们发现人类与LLM的评分之间总体上存在较强的相关性。然而，在分析评分分布以及比较不同类型的拼写变体时，出现了显著的差异。

    arXiv:2511.23041v2 Announce Type: replace  Abstract: Spelling variation (e.g. funnnn vs. fun) can influence the social perception of texts and their writers: we often have various associations with different forms of writing (is the text informal? does the writer seem young?). In this study, we focus on the social perception of spelling variation in online writing in English and study to what extent this perception is aligned between humans and large language models (LLMs). Building on sociolinguistic methodology, we compare LLM and human ratings on three key social attributes of spelling variation (formality, carefulness, age). We find generally strong correlations in the ratings between humans and LLMs. However, notable differences emerge when we analyze the distribution of ratings and when comparing between different types of spelling variation.
    
[^322]: 面向罕见ICD编码的合成临床笔记：一种以数据为中心的长尾医疗编码框架

    Synthetic Clinical Notes for Rare ICD Codes: A Data-Centric Framework for Long-Tail Medical Coding

    [https://arxiv.org/abs/2511.14112](https://arxiv.org/abs/2511.14112)

    该论文提出一个以数据为中心的合成数据生成框架，利用真实共现模式和ICD知识生成9万份覆盖7902个ICD编码的合成出院摘要，以缓解ICD编码长尾分布不平衡问题并提升罕见编码的自动编码性能。

    

    从临床文本中自动进行ICD编码是医学自然语言处理中的一项关键任务，但诊断编码的极端长尾分布仍然阻碍着该任务的发展。数千个罕见和零样本的ICD编码在MIMIC-III等数据集中严重欠表示，导致宏观F1分数较低。在本工作中，我们提出了一个以数据为中心的框架，通过生成高质量的合成出院摘要来缓解这种不平衡。我们的方法利用真实世界的共现模式、ICD描述、同义词、分类体系以及相似的临床笔记，构建以罕见编码为锚点的真实多标签编码集合。利用这些结构化提示，我们生成了90,000份覆盖7,902个ICD编码的合成笔记，显著扩展了训练数据的分布。我们在原始数据集和扩展数据集上微调了两个最先进的基于Transformer的模型PLM-ICD和GKI-ICD。实验表明，我们的方法适度提升了宏观F1分数。

    arXiv:2511.14112v2 Announce Type: replace-cross  Abstract: Automatic ICD coding from clinical text is a critical task in medical NLP but remains hindered by the extreme long-tail distribution of diagnostic codes. Thousands of rare and zero-shot ICD codes are severely underrepresented in datasets like MIMIC-III, leading to low macro-F1 scores. In this work, we propose a data-centric framework that generates high-quality synthetic discharge summaries to mitigate this imbalance. Our method constructs realistic multi-label code sets anchored on rare codes by leveraging real-world co-occurrence patterns, ICD descriptions, synonyms, taxonomy, and similar clinical notes. Using these structured prompts, we generate 90,000 synthetic notes covering 7,902 ICD codes, significantly expanding the training distribution. We fine-tune two state-of-the-art transformer-based models, PLM-ICD and GKI-ICD, on both the original and extended datasets. Experiments show that our approach modestly improves macro
    
[^323]: UniAIDet：一个用于AI生成图像内容检测与定位的统一通用基准

    UniAIDet: A Unified and Universal Benchmark for AI-Generated Image Content Detection and Localization

    [https://arxiv.org/abs/2510.23023](https://arxiv.org/abs/2510.23023)

    该论文提出了UniAIDet，一个统一且全面的AI生成图像检测与定位基准，涵盖摄影与艺术图像以及文本生成、图像编辑、深度伪造等多种生成模型，并基于此全面评估了现有检测方法的泛化能力及检测与定位之间的关系。

    

    随着图像生成模型的迅速普及，数字图像的真实性已成为一个重要问题。虽然现有研究已提出了多种检测AI生成内容的方法，但当前的基准在覆盖多样化生成模型和图像类别方面存在局限，往往忽视了端到端的图像编辑和艺术图像。为了解决这些局限性，我们提出了UniAIDet，这是一个统一且全面的基准，同时包含摄影图像和艺术图像。UniAIDet涵盖了广泛的生成模型，包括文本到图像、图像到图像、图像修复、图像编辑和深度伪造模型。利用UniAIDet，我们对各种检测方法进行了全面评估，并回答了三个关于泛化能力以及检测与定位之间关系的关键研究问题。我们的基准和分析为未来研究提供了坚实的基础。

    arXiv:2510.23023v2 Announce Type: replace-cross  Abstract: With the rapid proliferation of image generative models, the authenticity of digital images has become a significant concern. While existing studies have proposed various methods for detecting AI-generated content, current benchmarks are limited in their coverage of diverse generative models and image categories, often overlooking end-to-end image editing and artistic images. To address these limitations, we introduce UniAIDet, a unified and comprehensive benchmark that includes both photographic and artistic images. UniAIDet covers a wide range of generative models, including text-to-image, image-to-image, image inpainting, image editing, and deepfake models. Using UniAIDet, we conduct a comprehensive evaluation of various detection methods and answer three key research questions regarding generalization capability and the relation between detection and localization. Our benchmark and analysis provide a robust foundation for f
    
[^324]: 三个臭皮匠顶个诸葛亮：基于双假设的视听语音纠错

    Two Heads Are Better Than One: Audio-Visual Speech Error Correction with Dual Hypotheses

    [https://arxiv.org/abs/2510.13281](https://arxiv.org/abs/2510.13281)

    该论文提出 DualHyp 视听语音纠错框架，让大语言模型在语言空间中联合推理 ASR 与 VSR 双路的独立 N-best 假设，并通过噪声感知的 RelPrompt 机制动态切换模态关注焦点，在 LRS2 基准上实现高达 57.7% 的错误率下降，远超单流方法的 10%。

    

    本文提出了一种面向视听语音识别（AVSR）的生成式错误纠正（GER）框架的新范式，该范式直接在语言空间中对各模态特有的证据进行推理。我们的框架 DualHyp 使大语言模型（LLM）能够整合来自独立的自动语音识别（ASR）模型与视觉语音识别（VSR）模型的各自独立的 N-best 假设列表。为了最大化 DualHyp 的效果，我们进一步提出了 RelPrompt，这是一种噪声感知的引导机制，可为 LLM 提供基于模态的提示。RelPrompt 提供各模态流随时间的可靠性信息，引导模型在 ASR 和 VSR 假设之间动态切换关注焦点，从而实现更准确的纠正。在各种损坏场景下，我们的框架在 LRS2 基准上相比标准 ASR 基线实现了高达 57.7% 的错误率下降，而单流 GER 方法仅能获得约 10% 的提升。

    arXiv:2510.13281v2 Announce Type: replace-cross  Abstract: This paper introduces a new paradigm for generative error correction (GER) framework in audio-visual speech recognition (AVSR) that reasons over modality-specific evidences directly in the language space. Our framework, DualHyp, empowers a large language model (LLM) to compose independent N-best hypotheses from separate automatic speech recognition (ASR) and visual speech recognition (VSR) models. To maximize the effectiveness of DualHyp, we further introduce RelPrompt, a noise-aware guidance mechanism that provides modality-grounded prompts to the LLM. RelPrompt offers the temporal reliability of each modality stream, guiding the model to dynamically switch its focus between ASR and VSR hypotheses for an accurate correction. Under various corruption scenarios, our framework attains up to 57.7% error rate gain on the LRS2 benchmark over standard ASR baseline, contrary to single-stream GER approaches that achieve only 10% gain. 
    
[^325]: 大语言模型遗忘的显微镜式审视：一项基准研究

    Large Language Model Unlearning Under the Microscope: A Benchmark Study

    [https://arxiv.org/abs/2510.07626](https://arxiv.org/abs/2510.07626)

    该论文提出了一个将十二种大语言模型遗忘方法归纳为三大方法家族的系统性分类体系，并重新审视了遗忘有效性、效用保留和鲁棒性的评估方法，揭示了当前以多项选择题准确率为主导的评估方式的局限性。

    

    机器遗忘技术在大语言模型中的应用，即所谓的大语言模型遗忘，旨在移除不期望的数据、知识和行为，同时保留模型的各项能力。尽管过去两年间该领域取得了快速进展，但大语言模型遗忘的研究仍然较为零散，对于什么构成有效的遗忘以及如何对其进行严格评估仍缺乏清晰的认识。在这项工作中，我们提出了一个系统性的分类体系，将十二种最新的有状态遗忘方法归纳为三大方法学家族：散度驱动优化、表示对齐失调以及基于拒绝的定向遗忘。基于这一分类体系，我们重新审视了遗忘有效性、效用保留和鲁棒性三个方面的评估，重点关注WMDP基准中有害知识及相关模型能力的移除。我们的分析表明，当前以多项选择题准确率为主导的评估方式仅能提供一种叙述性的（评估视角）……

    arXiv:2510.07626v2 Announce Type: replace-cross  Abstract: Machine unlearning for large language models (LLMs), known as LLM unlearning, aims to remove undesired data, knowledge, and behaviors while preserving model capabilities. Despite rapid progress over the past two years, research in LLM unlearning remains fragmented, with limited clarity on what constitutes effective unlearning and how it should be rigorously evaluated. In this work, we present a principled taxonomy of twelve recent stateful unlearning methods, grouped into three methodological families: divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning. Building on this taxonomy, we revisit the evaluation of unlearning effectiveness (UE), utility retention (UT), and robustness (Rob), focusing on WMDP for hazardous knowledge and associated model capability removal. Our analysis shows that current evaluations, dominated by multiple-choice question (MCQ) accuracy, offer only a narr
    
[^326]: SPECTRA：通过大语言模型分布式推理揭示用户偏好的全谱系

    SPECTRA: Revealing the Full Spectrum of User Preferences via Distributional LLM Inference

    [https://arxiv.org/abs/2509.24189](https://arxiv.org/abs/2509.24189)

    SPECTRA将用户偏好预测从生成排序列表转变为让微调后的LLM输出各偏好类别的概率分布，显著提升了分布对齐效果、长尾兴趣恢复能力和跨用户公平性。

    

    大语言模型（LLM）通常通过生成排序后的项目列表来表示用户偏好。这些列表可能过度代表高频类别，而忽略长尾兴趣。我们转而将偏好预测视为LLM分类任务，即由微调后的LLM为每个类别分配一个概率。给定用户的交互历史，我们估计未来交互在每个类别中的占比。我们提出SPECTRA（基于Softmax探测的类别级token读数提取与分析方法），该方法从微调后LLM的token分数中读取可解释偏好类别上的一个分布。我们在MovieLens、Yelp以及一个大规模短视频平台上对SPECTRA进行了评估。SPECTRA实现了：(i) 分布对齐，在公开数据集上将与经验偏好分布的Jensen-Shannon散度降低了38-44%；(ii) 具备跨用户公平性的长尾恢复，将top-3类别曝光熵提升了23%...

    arXiv:2509.24189v5 Announce Type: replace  Abstract: Large language models (LLMs) often represent user preferences by generating ranked item lists. These lists can overrepresent frequent categories and omit long-tail interests. We instead treat preference prediction as an LLM classification task, where a fine-tuned LLM assigns a probability to each category. Given a user's interaction history, we estimate the share of future interactions in each category. We propose SPECTRA (Softmax Probing for Extracted Category-level Token Readouts and Analysis), which reads a distribution over interpretable preference categories from a fine-tuned LLM's token scores. We evaluate SPECTRA on MovieLens, Yelp, and a large-scale short-video platform. SPECTRA delivers (i) distributional alignment, reducing Jensen-Shannon divergence to the empirical preference distribution by 38--44% across public datasets; (ii) long-tail recovery with cross-user fairness, raising top-3 category exposure entropy by $23%$ on
    
[^327]: 代币税：多语言分词中的系统性偏差

    The Token Tax: Systematic Bias in Multilingual Tokenization

    [https://arxiv.org/abs/2509.05486](https://arxiv.org/abs/2509.05486)

    该论文通过对 11 个大语言模型在 16 种非洲语言上的评估，揭示了更高的分词生育率会系统性降低准确率并使训练成本翻两番，从而量化了低资源、形态复杂语言所承受的“代币税”偏见。

    

    分词效率低下给形态复杂、资源匮乏的语言带来了结构性劣势，导致计算资源消耗增加和准确率下降。我们在 AfriMMLU 数据集（5 个学科；16 种非洲语言）上评估了 11 个大语言模型（LLM）；在能够计算分词生育率的 10 个模型中，分词生育率越高与准确率越低相关：全部 50 个模型—学科斜率均为负值，其中 24 个在 p<0.05 水平上显著，12 个在错误发现率（FDR）校正后仍然显著。我们进一步发现，推理模型（如 DeepSeek R1、o1）在所评估的模型中具有最高的平均准确率和最小的英语—非洲语言差距。在经济方面，token 数量翻倍会导致训练成本和时间增至四倍，凸显了许多语言所面临的“代币税”。这些结果推动了形态感知分词、公平定价以及面向公平自然语言处理的多语言基准的发展。

    arXiv:2509.05486v2 Announce Type: replace-cross  Abstract: Tokenization inefficiency is associated with structural disadvantages for morphologically complex, low-resource languages, inflating compute resources and reducing accuracy. We evaluate 11 large language models (LLMs) on AfriMMLU (5 subjects; 16 African languages); for the 10 models whose fertility we could compute, higher token fertility is associated with lower accuracy: all 50 model--subject slopes are negative, 24 are significant at p<0.05, and 12 remain significant after false-discovery-rate correction. We further find that reasoning models (e.g., DeepSeek R1, o1) have the highest average accuracy and the smallest English-African gaps among the models evaluated. In terms of economics, a doubling in tokens results in quadrupled training cost and time, underscoring the "token tax" faced by many languages. These results motivate morphologically aware tokenization, fair pricing, and multilingual benchmarks for equitable natura
    
[^328]: KG-TRACES：通过知识图谱约束的轨迹推理与归因监督增强大语言模型

    KG-TRACES: Enhancing Large Language Models with Knowledge Graph-constrained Trajectory Reasoning and Attribution Supervision

    [https://arxiv.org/abs/2506.00783](https://arxiv.org/abs/2506.00783)

    提出KG-TRACES框架，通过对符号关系路径、三元组级推理路径和归因感知推理过程的显式监督，显著提升大语言模型在复杂推理任务中的可解释性和可信度。

    

    大语言模型（LLMs）在各种自然语言处理任务中取得了显著进展，但它们在复杂推理问题上的表现仍然受到可解释性和可信度不足的阻碍。这一问题通常表现为幻觉或无法归因的推理过程，限制了它们在复杂推理场景中的适用性。为了解决这一问题，我们提出了知识图谱约束的轨迹推理归因与链式解释监督框架（KG-TRACES），这是一个通过对推理路径和推理过程进行显式监督来增强大语言模型推理能力的新型框架。KG-TRACES联合监督模型完成以下任务：（1）预测符号化关系路径，（2）预测完整的三元组级别推理路径，（3）生成基于推理路径的、具有归因意识的推理过程。在推理阶段，该模型能够自适应地应对有知识图谱和没有知识图谱的两种场景。

    arXiv:2506.00783v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have made remarkable strides in various natural language processing tasks, but their performance on complex reasoning problems remains hindered by a lack of explainability and trustworthiness. This issue, often manifesting as hallucinations or unattributable reasoning processes, limits their applicability in complex reasoning scenarios. To address this, we propose Knowledge Graph-constrained Trajectory Reasoning Attribution and Chain Explanation Supervision (KG-TRACES), a novel framework that enhances the reasoning ability of LLMs through explicit supervision over reasoning paths and processes. KG-TRACES jointly supervises the model to: (1) predict symbolic relation paths, (2) predict full triple-level reasoning paths, and (3) generate attribution-aware reasoning processes grounded in the reasoning paths. At inference phase, the model adapts to both KG-available and KG-unavailable scenarios, retriev
    
[^329]: 跨模态RAG：子维度文本到图像检索增强生成

    Cross-modal RAG: Sub-dimensional Text-to-Image Retrieval-Augmented Generation

    [https://arxiv.org/abs/2505.21956](https://arxiv.org/abs/2505.21956)

    提出跨模态RAG框架，通过将查询和图像分解为子维度组件，并结合稀疏与稠密混合检索策略识别帕累托最优图像集合，实现子查询感知的检索增强文本到图像生成。

    

    文本到图像生成日益需要访问特定领域、细粒度且快速演进的知识，而预训练模型无法完全涵盖这些知识，因此需要引入检索方法。现有的检索增强生成（RAG）方法试图通过检索全局相关的图像来解决这一问题，但当没有单一图像能够包含复杂用户查询中的所有所需元素时，这些方法便会失效。我们提出了跨模态RAG（Cross-modal RAG），这是一种新颖的框架，它将查询和图像均分解为子维度组件，从而实现子查询感知的检索与生成。我们的方法引入了一种混合检索策略，将子维度稀疏检索器与稠密检索器相结合，以识别出帕累托最优的图像集合，其中每张图像都贡献了查询的互补方面。在生成过程中，多模态大语言模型被引导选择性地以与特定子查询对齐的相关视觉特征为条件。

    arXiv:2505.21956v4 Announce Type: replace-cross  Abstract: Text-to-image generation increasingly demands access to domain-specific, fine-grained, and rapidly evolving knowledge that pretrained models cannot fully capture, necessitating the integration of retrieval methods. Existing Retrieval-Augmented Generation (RAG) methods attempt to address this by retrieving globally relevant images, but they fail when no single image contains all desired elements from a complex user query. We propose Cross-modal RAG, a novel framework that decomposes both queries and images into sub-dimensional components, enabling subquery-aware retrieval and generation. Our method introduces a hybrid retrieval strategy, combining a sub-dimensional sparse retriever with a dense retriever, to identify a Pareto-optimal set of images, each contributing complementary aspects of the query. During generation, a multimodal large language model is guided to selectively condition on relevant visual features aligned to sp
    
[^330]: JustLogic：一个用于评估大型语言模型演绎推理能力的综合基准

    JustLogic: A Comprehensive Benchmark for Evaluating Deductive Reasoning in Large Language Models

    [https://arxiv.org/abs/2501.14851](https://arxiv.org/abs/2501.14851)

    JustLogic是一个合成生成的命题逻辑演绎推理基准，兼具任务复杂性、先验知识无关性和深入错误分析能力，克服了现有基准的三大局限。

    

    逻辑推理是大型语言模型（LLM）的关键组成部分，近年来大量研究工作致力于提升其演绎推理能力。然而，现有的演绎推理基准存在显著局限，限制了其实用价值，即缺乏任务复杂性、存在先验知识作为混淆因素，以及错误分析流于表面。为了解决这些缺陷，我们提出了JustLogic，一个用于评估命题逻辑演绎能力的合成生成基准。在这一逻辑片段范围内，JustLogic具有以下特点：(i) 复杂性，能够生成多样化的语言模式、词汇和论证结构；(ii) 与先验知识无关，大幅降低了具备先验知识的模型所拥有的优势，使所有问题仅凭给定前提即可作答；(iii) 能够对异构的推理错误进行深入的错误分析。

    arXiv:2501.14851v3 Announce Type: replace-cross  Abstract: Logical reasoning is a critical component of Large Language Models (LLMs), and substantial research efforts in recent years have aimed to enhance their deductive reasoning abilities. However, existing deductive reasoning benchmarks suffer from significant constraints that restrict their utility, i.e., the lack of task complexity, the presence of prior knowledge as a confounder, and superficial error analysis. To address these deficiencies, we introduce JustLogic, a synthetically generated benchmark for evaluating deduction in propositional logic. Within this fragment, JustLogic is (i) complex, capable of generating a diverse range of linguistic patterns, vocabulary, and argument structures, (ii) prior knowledge independent, substantially reducing the advantage of models possessing prior knowledge so that questions are answerable from the given premises alone, and (iii) capable of in-depth error analysis on the heterogeneous eff
    
[^331]: 一种用于确定性L系统推断的特征图归约方法：结合经典与量子启发式求解器

    A characteristic-graph reduction for deterministic L-system inference with classical and quantum-heuristic solvers

    [https://arxiv.org/abs/2411.19906](https://arxiv.org/abs/2411.19906)

    本文提出字符串序列的“特征图”概念，证明D0L系统推断问题等价于在该图中寻找指定大小的独立集，从而将推断问题多项式归约到最大独立集与SAT问题，并给出基于QAOA的量子启发式求解方案。

    

    L系统被广泛用于建模和模拟生物生长过程，包括植物的发育。受自动化构建L系统这一更宏观目标的启发，本文研究了形式化的确定性上下文无关L系统（D0L系统）推断问题，其输入是一个有限的、无噪声的字符串序列。我们引入了字符串序列的特征图这一概念，并证明了存在兼容的D0L系统当且仅当该图中存在指定大小的独立集。这一结果给出了从D0L推断问题到最大独立集（MIS）问题和SAT问题的多项式归约，因此结合精确的MIS或SAT求解器即可获得精确的经典决策途径。我们还提出了一种基于量子近似优化算法（QAOA）的量子启发式方法，用于生成候选独立集。该启发式方法并非D0L推断的决策程序，启发式方法的失败并不意味着……

    arXiv:2411.19906v4 Announce Type: replace-cross  Abstract: L-systems are widely used to model and simulate biological growth processes, including plant development. Motivated by the broader goal of automating L-system construction, this paper studies the formal deterministic context-free L-system (D0L-system) inference problem in which the input is a finite, noise-free sequence of strings. We introduce the characteristic graph of a string sequence and prove that the existence of a compatible D0L-system is equivalent to the existence of an independent set of a prescribed size in this graph. This gives polynomial reductions from D0L inference to maximum independent set (MIS) and SAT, and hence an exact classical decision route when combined with an exact MIS or SAT solver. We also describe a Quantum Approximate Optimization Algorithm (QAOA)-based quantum heuristic for generating candidate independent sets. This heuristic is not a decision procedure for D0L inference, and failure of the h
    
[^332]: ERIT：一个面向老年人真实场景情感识别的负责任多模态情感计算基准

    ERIT: A Responsible Multimodal Affective Computing Benchmark for Older-Adult Emotion Recognition in the Wild

    [https://arxiv.org/abs/2407.17772](https://arxiv.org/abs/2407.17772)

    ERIT基准通过将代表性标注帧与转录语音配对以固定多模态输入，消除了帧采样带来的变异性，为老年人这一代表性不足群体的真实场景情感识别提供了可重复、可公平比较的受控图文融合评估基准。

    

    我们介绍了一个用于老年人情感识别中图文融合的受控评估设置，并在ERIT基准中予以实现。许多现有的基于视频的情感数据集在评估时依赖帧采样，而帧采样方式的差异可能会显著影响结果——然而此类细节很少被披露，限制了可重复性和公平比较。ERIT通过将代表性的标注帧与相应的转录语音配对，消除了这一变异来源，使每个样本具有固定的视觉和文本输入。该数据集涵盖七种情绪（愤怒、厌恶、恐惧、快乐、悲伤、惊讶和中性），主要设计为针对老年人这一代表性不足群体的受控评估资源。我们在零样本设置下使用LLaVA-NeXT、Qwen2-VL、GPT-4V和GPT-4o对ERIT进行评估，并使用LoRA对LLaVA-NeXT进行微调，比较不同评估标准下的结果。

    arXiv:2407.17772v2 Announce Type: replace-cross  Abstract: We introduce a controlled evaluation setting for image-text fusion in older adult emotion recognition, instantiated in the ERIT benchmark. Many existing video-based emotion datasets rely on frame sampling during evaluation, and differences in how frames are sampled can substantially affect results - yet such details are seldom disclosed, limiting reproducibility and fair comparison. ERIT removes this source of variability by pairing representative annotated frames with corresponding transcribed speech, so that each sample has fixed visual and textual inputs. The dataset covers seven emotions (anger, disgust, fear, happiness, sadness, surprise, and neutral), and is designed primarily as a controlled evaluation resource for the underrepresented population of older adults. We evaluate ERIT in zero-shot settings with LLaVA-NeXT, Qwen2-VL, GPT-4V, and GPT-4o, and also fine-tune LLaVA-NeXT with LoRA, comparing results using strict, m
    
[^333]: 在检索增强生成框架中微调大语言模型进行关系抽取

    Fine-Tuning Large Language Models for Relation Extraction within a Retrieval-Augmented Generation Framework

    [https://arxiv.org/abs/2406.14745](https://arxiv.org/abs/2406.14745)

    本文提出在检索增强生成（RAG）框架中微调大语言模型以进行关系抽取，有效解决了通用大语言模型作为生成器时面临的领域适应挑战。

    

    信息抽取（IE）在将非结构化数据转换为知识图谱等结构化格式方面发挥着关键作用。信息抽取中的主要任务之一是关系抽取（RE），它识别文本数据中实体之间的关系。这一过程丰富了文档的语义理解，使得更精确的信息检索和查询回答成为可能。近期利用预训练语言模型的研究在关系抽取方面展现出了显著的性能提升。在大语言模型（LLM）时代，对这些大语言模型进行微调可以缓解零样本关系抽取方法的局限性，特别是在克服关系抽取中固有的领域适应挑战方面。本工作不仅探索了微调大语言模型的有效性，还探讨了将其集成到基于检索增强生成（RAG）的关系抽取方法中，以解决通用大语言模型作为生成器时所面临的领域适应挑战。

    arXiv:2406.14745v3 Announce Type: replace-cross  Abstract: Information Extraction (IE) plays a pivotal role in transforming unstructured data into structured formats, such as Knowledge Graphs. One of the main tasks within IE is Relation Extraction (RE), which identifies relations between entities in text data. This process enriches the semantic understanding of documents, enabling more precise information retrieval and query answering. Recent works leveraging pre-trained language models have demonstrated significant performance improvements in RE. In the current era of Large Language Models (LLMs), fine-tuning these LLMs can mitigate the limitations of zero-shot RE methods, particularly in overcoming the domain adaptation challenges inherent in RE. This work explores not only the effectiveness of fine-tuned LLMs but also their integration into a Retrieval-Augmented Generation (RAG)-based RE approach to address domain adaptation challenges when general-purpose LLMs serve as generators w
    
[^334]: 等等，这都是令牌噪音？一直就是吗：利用 Shapley 值解释 LLM 行为

    Wait, It's All Token Noise? Always Has Been: Interpreting LLM Behavior Using Shapley Value

    [https://arxiv.org/abs/2404.01332](https://arxiv.org/abs/2404.01332)

    使用Shapley值方法解释LLM行为，揭示了所谓的“令牌噪音”效应，揭示了LLMs的决策在很大程度上受到提示组件的影响

    

    大型语言模型（LLMs）的出现为模拟人类行为和认知过程开辟了新的可能性，潜在应用包括市场研究和消费者行为分析等各个领域。然而，由于LLMs的显著差异暗示了不同的基础过程在起作用，以及LLMs对提示变化的敏感性，利用LLMs作为人类主体的替代仍然存在不确定性。本文提出了一种基于合作博弈理论中Shapley值的新方法来解释LLM行为，并量化每个提示组件对模型输出的相对贡献。通过两个应用--一个离散选择实验和一个认知偏见调查，我们展示了Shapley值方法如何揭示我们所谓的“令牌噪音”效应，即LLM决策受到的影响严重偏向于

    arXiv:2404.01332v1 Announce Type: cross  Abstract: The emergence of large language models (LLMs) has opened up exciting possibilities for simulating human behavior and cognitive processes, with potential applications in various domains, including marketing research and consumer behavior analysis. However, the validity of utilizing LLMs as stand-ins for human subjects remains uncertain due to glaring divergences that suggest fundamentally different underlying processes at play and the sensitivity of LLM responses to prompt variations. This paper presents a novel approach based on Shapley values from cooperative game theory to interpret LLM behavior and quantify the relative contribution of each prompt component to the model's output. Through two applications-a discrete choice experiment and an investigation of cognitive biases-we demonstrate how the Shapley value method can uncover what we term "token noise" effects, a phenomenon where LLM decisions are disproportionately influenced by 
    

