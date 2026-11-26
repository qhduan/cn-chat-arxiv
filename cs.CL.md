# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tokenisation is not in APX](https://arxiv.org/abs/2611.34938) | 本文证明除非P=NP，分词化问题无法在多项式时间内近似到最优解的任意常数因子（即不属于APX类），且直接分词即使在一元字符表上也是APX难的。 |
| [^2] | [Can image generation help visual reasoning? An empirical study](https://arxiv.org/abs/2611.34935) | 这项实证研究发现，通过生成辅助图像来帮助视觉推理在大多数情况下反而会降低性能——在13组视觉语言模型与生成器的配对实验中平均准确率下降3.7个百分点，十个任务中九个出现下降——挑战了“多模态模型可以用图像思考”的假设。 |
| [^3] | [Diverse Alternative Reasoning Traces for Hard Math Problems](https://arxiv.org/abs/2611.34928) | 提出了DART轨迹挖掘方法，通过分解被拒绝的推理轨迹并基于位置加权不确定性与参考答案引导生成刻意不同的替代推理步骤，在相同生成成本下比蒙特卡洛采样多恢复37%的复杂数学问题，且每条合格轨迹成本降低30%。 |
| [^4] | [Mixture-of-Loops Transformer](https://arxiv.org/abs/2611.34923) | 提出混合循环Transformer（MoLT），以KV缓存作为跨深度和跨循环的选择性记忆重用机制，在表达能力与效率之间取得新平衡，并支持超越训练循环次数的高效解码。 |
| [^5] | [$\mu$-Bench: Benchmarking Multi-User Task Completion in Shared Interaction Voice Agents](https://arxiv.org/abs/2611.34920) | 提出了 μ-Bench 基准，用于评估共享空间中闭环全双工多用户语音交互下的联合任务完成能力，通过 JTS 指标与 ATC 诊断揭示现有语音代理在多用户权限、隐私保护与协调方面仍存在显著不足。 |
| [^6] | [DRBench Pro: A Knowledge-Centric Benchmark for Professional Deep Research Agents](https://arxiv.org/abs/2611.34852) | DRBench Pro是一个以结构化知识覆盖为核心评测目标的深度研究智能体基准测试，通过从维基百科构建知识树生成研究查询与证据评分标准，从广度和深度两个维度（取几何平均）衡量智能体对复杂开放性问题的研究覆盖能力。 |
| [^7] | [Proverbs As A Window On Collective Prescriptions](https://arxiv.org/abs/2611.34832) | 该研究提出了一个基于大语言模型（经人类判断验证）的计算框架，首次对近5万条英语谚语所承载的集体规约进行量化分析，发现多数谚语在个人可控领域具有规约性，并普遍倡导谨慎、长期主义、诚实与善良等价值观。 |
| [^8] | [Repairing the Trustworthiness of Distilled Language Models](https://arxiv.org/abs/2611.34831) | 提出基于近端Bregman响应函数（PBRF）的针对性修复框架，无需原始蒸馏语料库即可修复蒸馏语言模型的可信性退化，并通过比值比目标扩展至偏好监督，同时约束模型能力的变化。 |
| [^9] | [The Linguistic Knowledge Benchmark: Do Language Models Know the Foundations of Early-Grade Literacy?](https://arxiv.org/abs/2611.34823) | 该论文提出了语言知识基准及一套可复现的自动化流程，将课程文件转化为涵盖九个语言学领域和七个年级的审核题目集，用于检验语言模型是否具备支持低收入和中等收入国家早期读写教学所需的语言学知识，并以基斯瓦希里语和英语为实例评估了两百多个模型。 |
| [^10] | [VulnSafe: Measuring Vulnerability-Aware Relational Safety in Large Language Models](https://arxiv.org/abs/2611.34819) | VulnSafe提出了一个以脆弱性感知响应能力（VARC）为核心的中文评估基准，通过41个场景、2050个平衡提示词，从风险感知、安全维护、自主性保护、支持性响应和边界管理五个维度系统评估大语言模型的关系安全能力，并区分主动的“应该做”与禁止性的“不应该做”行为。 |
| [^11] | [PersonaFactory: Stylistic User Prompt Augmentation with Synthetic Persona Bank](https://arxiv.org/abs/2611.34785) | 该论文提出PersonaFactory端到端流程，构建了超74万个刻画背景与交流行为的结构化合成人格库，并以85种语言生成约124万条人格风格化对话，解决了合成用户提示风格单一、无法模拟真实用户交流机制的问题。 |
| [^12] | [When Every Rollout Fails: Splicing Learner-Conditioned Options into Long-Horizon Agent RL](https://arxiv.org/abs/2611.34777) | 提出SPLICE方法，在线地从全失败的当前策略轨迹中拼接由学习者条件化的局部选项作为训练目标，解决了长程智能体强化学习中全失败组导致无奖励策略梯度的问题。 |
| [^13] | [DFly2: Turbocharging Long-Context Speculative Decoding by Sharing the Target's Sparsity](https://arxiv.org/abs/2611.34776) | 该论文提出 DFly2，让长上下文推测解码中的草稿模型直接复用原生稀疏目标模型已有的缓存及逐查询的位置选择，免去草稿模型自身的缓存开销与选择计算，从而大幅加速长上下文解码。 |
| [^14] | [One Model, Many Modes: Evolving System Prompt Ensembles for Creative Generation](https://arxiv.org/abs/2611.34760) | 该论文提出免训练框架 SEPO，通过演化系统提示词集成并结合质量约束下的贪心最大-最小选择来缓解偏好对齐大语言模型的模式坍缩，从而生成兼具高质量、新颖性与多样性的创造性内容。 |
| [^15] | [Reading Fine-Tuning Diffs Through Lenses: Model Diffing for Backdoor Detection](https://arxiv.org/abs/2611.34758) | 该论文提出了一个基于token读出的统一评分系统来评估各类模型差异后门检测方法，并引入J透镜与J++透镜改进权重和激活分析方法，其中最强的ΔW J++方法在12个设置中的11个成功将后门相关token排入前十。 |
| [^16] | [Collect for Coverage, Replay with Evidence: Adaptive and Efficient On-Policy Distillation for Long-Horizon Agents](https://arxiv.org/abs/2611.34756) | 该论文提出CARE-OPD框架，通过学生种子教师补全和响应级置信度门控扩大历史覆盖并过滤不可靠教师信号，同时以证据自适应地回放轨迹中任意深度的关键决策，从而实现面向长程智能体的自适应且高效的前缀回放式在线策略蒸馏。 |
| [^17] | [Bigger Model or Bigger Table? How Much Backbone a Chorus N-gram Module Replaces](https://arxiv.org/abs/2611.34725) | 该论文通过在 13.4 亿波兰语词元上训练 16M 至 349M 参数的模型，量化了 Chorus N-gram 哈希表模块对 transformer 主干网络的替代作用：当查找表与主干网络规模相当时，每个新增参数最多可替代 0.31 个主干参数，使模型能够以较小规模匹敌 1.16 至 1.34 倍大小的普通模型。 |
| [^18] | [From Shared Patterns to Individual Evidence: Two-Stage Reasoning for Emerging Entities in Temporal Knowledge Graphs](https://arxiv.org/abs/2611.34708) | 该论文提出INSPIRE两阶段推理框架，通过融合全目录事件记忆的共享模式预测与轻量级个体证据建模，有效解决时序知识图谱中新兴实体从零样本到少样本演化场景下的链接预测难题。 |
| [^19] | [When the Proxy Is Wrong: Mismatch-Corrected Bandits for Adaptive Data Sampling](https://arxiv.org/abs/2611.34707) | 该论文提出失配校正老虎机方法，将代理模型与目标模型的排序失建为多保真度老虎机问题，利用稀少审计中的成对残差校正代理奖励并智能分配审计预算，从而在无需假设共享偏移或排序保持的情况下改进大模型的自适应数据采样。 |
| [^20] | [PoVi: Agentic Poetry-to-Video Generation with Knowledge Grounding and Figurative Understanding](https://arxiv.org/abs/2611.34690) | PoVi 提出了一个结合知识接地与修辞理解的智能体框架和基准，将诗歌语义逐步转化为旁白、解读与镜头级视觉规划，显著提升了诗歌视频生成在隐含知识和隐喻表达上的表现。 |
| [^21] | [Technical Report MEG2text: Beyond Alignment in Non-Invasive Brain-to-Text](https://arxiv.org/abs/2611.34686) | MEG2text通过修正训练评估流程、脑-文本对齐、类别平衡读出与语言模型融合，在LibriBrain100词语排序任务上将平均Top-10平衡准确率提升至64.12%，比MEG-XL提高了16.92个百分点。 |
| [^22] | [Neural Rendering Compiler: Generative Video Models as Programmable Renderers](https://arxiv.org/abs/2611.34680) | NeurCo 提出了一种神经渲染编译器，通过自动探测冻结视频模型构建能力手册，并将用户意图编译为模型特定的执行策略，从而为生成式视频模型提供可预测的编程式渲染控制。 |
| [^23] | [EPPC-CARE: Staged Hierarchical Annotation of Communication Behaviors in Secure Patient-Provider Messages](https://arxiv.org/abs/2611.34539) | EPPC-CARE是一个基于开源大语言模型的自动化分层标注框架，能从医患门户消息中识别沟通行为并输出“文本片段—沟通类别—子类别”三元组，性能全面超越EPPC-OASIS和监督微调方法，且接近人类专家标注水平。 |
| [^24] | [AliceAI-Foundation-80B-A3B-Base](https://arxiv.org/abs/2611.34529) | Yandex 从零训练并开源了总参数 800 亿、每 token 激活约 30 亿的混合专家基座模型，它结合 Kimi Delta Attention 与全注意力并使用 Muon 优化器训练，同时提出在 MoE 输出上引入辅助损失以在预测的高学习率下稳定训练。 |
| [^25] | [Know the Design, Pick the Attack: Sentence-Keyed Watermarks for LLMs](https://arxiv.org/abs/2611.34505) | 论文提出了一种自适应攻击者，通过阅读公开的水印设计在改写抹除和伪造复制两种攻击间择优选择，能击败所有现有水印，并据此提出KeyedLex水印——从上一句的语法中读取密钥，从而同时抵御两类攻击。 |
| [^26] | [FlyingStart: Learned Cross-Token Warm Starts for Fixed-Point Decoding in Recurrent-Depth Language Models](https://arxiv.org/abs/2611.34492) | 提出FlyingStart方法，利用上一个Token停止时的状态经闭式解拟合的线性变换来热启动下一个Token的迭代，在精度不变的前提下将递归深度语言模型的解码循环层数减少14%–19%。 |
| [^27] | [Where Does the Asymmetry Enter? Staging a Fairness Audit Across a Fused Multimodal Moderation Pipeline](https://arxiv.org/abs/2611.34491) | 本文提出对已公开的TikTok多模态内容审核分类器进行跨阶段事后公平性审计，以统一的显著性标准在“输入内容、分支编码、最终决策”三个环节中定位按语言划分的子群体不对称性究竟出现在流水线的哪个阶段。 |
| [^28] | [vSFT: Learning to Verbally Tune Agents at Test Time](https://arxiv.org/abs/2611.34440) | 提出vSFT框架，通过超网络将监督者基于智能体失败轨迹生成的自然语言反馈转化为低秩权重更新，实现测试时的参数内演化，在ALFWorld等任务上性能超越GRPO和Reflexion等主流强化学习方法。 |
| [^29] | [Mitigating Hallucinations in Medical Question Answering on ESC Cardiology Guidelines: Evaluation and Retrieval Optimization](https://arxiv.org/abs/2611.34427) | 该研究首次逐条对照人类判断，系统评估了基于ESC心脏病学指南的“验证-修订”流程在医学问答中的实际失败情况，并优化了检索配置以缓解大语言模型的幻觉问题。 |
| [^30] | [DocSEAM: Bidirectional Cross-Page Encoding for Multi-Page Document Understanding](https://arxiv.org/abs/2611.34420) | DocSEAM是一种插入在视觉编码器与语言模型之间的轻量级双向跨页编码模块，通过以中心页为锚的掩码自注意力融合相邻页面信息，在不改变视觉标记数量的前提下解决了多页文档理解中页面独立编码导致的特征不连续问题。 |
| [^31] | [Any-Rate: Half-Bit Multi-Precision LLM Quantization with Coded Bit-Planes](https://arxiv.org/abs/2611.34402) | Any-Rate通过将位平面约束为8权重块的扩展汉明码字并结合逐精度联合编码，实现了从单一检查点以半比特为步长提供2–8比特的多精度LLM量化，在2比特和3比特下达到最先进的准确率。 |
| [^32] | [When Skills Fail to Compose: Compatibility and Retention in Language Models](https://arxiv.org/abs/2611.34398) | 该论文发现，在冻结语言模型上独立学习的软提示技能难以在单次调用中有效组合（准确率最高仅7.6%，远低于两次调用的88.2%–100%），且相同操作的技能块并不必然可互换，揭示了构建可复用技能库时兼容性与保留性方面的关键瓶颈。 |
| [^33] | [Strong Starts, Simple Recipes: Unified Multi-Teacher On-Policy Distillation from Multimodal Specialists](https://arxiv.org/abs/2611.34397) | 该论文提出Uni-MOPD方法，将学生模型的起点作为设计变量——先用无训练的专家模型均匀合并获得强起点，再用简短的多教师在线策略蒸馏配方进行精炼，并发现起点选择比后续训练策略对最终性能的影响更大。 |
| [^34] | [Under the Influence -- of What? A Controlled Evaluation of Influence Functions](https://arxiv.org/abs/2611.34373) | 这是首个通过受控数据划分重新训练的系统评估，验证了影响函数在GPT-2类模型中识别的“有影响力”训练样本确实与输出相关，但不总是直接承载目标属性，且相关性因任务设置而异。 |
| [^35] | [Single-Database Rewards Are Hackable: Counterfactual Execution for Trustworthy Text-to-SQL RL](https://arxiv.org/abs/2611.34363) | 该论文提出一种反事实数据库奖励机制，通过在隐藏的代码生成数据库副本上执行查询来验证结果，将文本到SQL强化学习中的奖励破解率从54.1%大幅降至1.9%，且不误拒任何正确查询。 |
| [^36] | [Learning from Unreliable Operator References: Claim-Level Rewards for Customer Support Agents](https://arxiv.org/abs/2611.34341) | 该论文提出将客服坐席回复视为待验证的声明而非模仿目标，通过声明级GRPO奖励机制过滤不可靠的训练信号，显著降低了幻觉率，并将无需人工介入的对话解决率从63.3%提升至80.8%。 |
| [^37] | [Puzzling-Bench: Evaluating Language Models on 23,777 Human-Written Puzzles with Community-Verified Answers](https://arxiv.org/abs/2611.34331) | Puzzling-Bench是一个基于23,777道来自Puzzling Stack Exchange、带有社区验证答案的人工编写谜题（涵盖谜语、密码、逻辑推理等类别）的语言模型评估基准，并利用训练截止日期后发布的谜题切片来检测数据污染。 |
| [^38] | [IRTIQA: Evolving in English, Evaluated in Urdu -- Does Agent Self-Evolution Transfer from English to Urdu and Code-Switched Input?](https://arxiv.org/abs/2611.34303) | 该论文提出IRTIQA基准（包含对齐的英语、乌尔都语及语码转换任务变体），发现智能体自我进化虽能显著提升任务能力，但收益主要局限于英语，很少能弥合与乌尔都语及语码转换输入之间的语言性能差距。 |
| [^39] | [MedSkill-E$^{2}$: Evaluating and Evolving Skills for Medical Agents](https://arxiv.org/abs/2611.34250) | 该论文提出了MedSkill-E²框架和EBU-Bench基准，首次发现并量化了医疗智能体中“技能已启用但未被充分利用”（EBU）现象，即智能体虽然拥有完成任务所需的技能，却在自主执行时无法有效应用，从而可能产生看似合理但实际错误的结果。 |
| [^40] | [SOLO: Towards Sample-Faithful and Detectable LLM Watermarking under Repeated Sampling via Content-Anchored Single-Owner Contexts](https://arxiv.org/abs/2611.34190) | 针对重复采样场景下无失真水印因跨响应上下文碰撞导致的耦合问题，提出SOLO方法，通过将带密钥上下文锚定在min-hash选择的内容上，实现了样本保真且可检测的大语言模型水印。 |
| [^41] | [MINE: A Training-Free Defense against Indirect Prompt Injection via Mutual Information and Embedding Variance](https://arxiv.org/abs/2611.34188) | MINE提出了一种基于互信息与嵌入方差的无训练、与攻击类型无关的新防御范式，通过最大化相关性而非检测攻击来抵御RAG中的间接提示注入，并在四个数据集上实现了无需训练防御方法中的最佳准确率-安全性权衡。 |
| [^42] | [Lightweight Reasoning Transfer via Chain-of-Thought Vector Projection](https://arxiv.org/abs/2611.34183) | 该论文提出一种保持师生模型均冻结的轻量级两阶段跨模型思维链向量投影迁移方法，无需教师输出或解题轨迹，仅用少量数据的最终答案监督即可将大模型推理能力迁移到小模型，在GSM8K和MATH上提升基础学生模型准确率10.37-21.98个百分点。 |
| [^43] | [Familiar Words, Unfamiliar Minds: Conventionalization without Calibration in the Vocabulary of Mind](https://arxiv.org/abs/2611.34179) | 该论文区分了将心智词汇延伸至语言模型时的两个独立过程——“惯例化”与“校准”，并通过对“幻觉”一词的语料库研究表明，词汇可在三年内完成惯例化，但由于语言模型与人类心智的组织方式根本不同，其推断含义的校准未必同步，甚至有依据的心智归因也可能误导。 |
| [^44] | [How resampling method can help us: Improving the Delta model](https://arxiv.org/abs/2611.34137) | 本文提出对文体特征应用重采样方法来改进Delta作者归属模型，在保持结果可解释性的同时提高了准确率，但代价是需要额外的计算时间。 |
| [^45] | [Is Sara Omar the true author of D{\o}devaskeren (The Dead Washer)? Authorship attribution in Danish literature](https://arxiv.org/abs/2611.34131) | 本研究运用Delta、文本间距离、最近收缩质心和朴素贝叶斯四种作者归属方法，发现丹麦畅销小说《洗尸者》的大部分内容更符合马德斯·佩德·诺德博的写作风格而非署名作者萨拉·奥马尔，但少数段落暗示可能还有另一位作者参与创作。 |
| [^46] | [Out of Sight, Not Out of Clues: Can Video LLMs Guess from Partial Evidence?](https://arxiv.org/abs/2611.34126) | 提出诊断基准ClueProbe，通过将问题划分为“可回答、可猜测、不可回答”三类，把视频大语言模型的证据感知能力与证据充分性判断能力解耦，揭示其在部分证据下的猜测能力缺陷，并借助针对性微调验证了诊断结论。 |
| [^47] | [From Data Room to Report: End-to-End AI Support for M&A Legal Due Diligence](https://arxiv.org/abs/2611.34120) | LEXI是一个利用大语言模型支持并购法律尽职调查全流程的端到端AI系统，在律师人工审核的控制下完成文档分类、信息提取与报告编制。 |
| [^48] | [Improving Tail-Augmented Speculative Drafting with Hidden State Mixing and Gradient Detach](https://arxiv.org/abs/2611.34092) | 本文提出两种仅用于训练阶段的方法——隐状态混合（HSM）和隐状态梯度分离（HSGD），在不显式模拟推理、不增加额外内存和专用注意力掩码的情况下，缓解推测解码草稿模型的训练-推理不匹配，从而改进自回归草稿生成。 |
| [^49] | [Disciplined Anthropomorphism: How to Extend Cognitive Language to Artificial Intelligence](https://arxiv.org/abs/2611.34090) | 本文提出“规范的拟人化”理论，主张不再纠缠于AI是否符合为人类铸造的认知概念，而是解构认知概念中的推理关联，依据人类概念需求决定哪些关联应为AI保留、切断或重新建立，从而为认知语言向人工智能的扩展提供规范方法。 |
| [^50] | [Budget, Readout or Reasoning? Where Small Reasoning Models Lose Value in Auction Bidding](https://arxiv.org/abs/2611.34068) | 该论文将六个4B至14B参数的开源推理模型置于基于真实eBay成交价和模拟广告市场的拍卖出价景观中，并对照精确计算器按原因拆分其价值损失，首次揭示损失主要源于模型自身的两类推理失败——忽视自身轨迹中已排名第一的最优选项、错误答案系统性偏向最高出价——导致竞标者损失7-17%的盈余。 |
| [^51] | [Stable Gated DeltaNet-2: Understanding and Mitigating Recurrent State Instability](https://arxiv.org/abs/2611.34051) | 本文发现GDN-2训练失败源于擦除操作中读写方向不对称导致的循环状态指数级增长，并提出通过对称擦除与衰减调节残差写入来解决该问题，同时理论上证明了状态增长的O(√T)上界并保留分块并行计算能力。 |
| [^52] | [Reliable User Simulation in Complex Multi-Step Workflows](https://arxiv.org/abs/2611.34044) | 本文提出了一个基于分层任务网络（HTN）抽象、包含500个工作流的基准，揭示出LLM用户模拟器在复杂多步骤工作流中的保真度随复杂度急剧下降，且目标漂移是导致失败的主要因素。 |
| [^53] | [Too Smart for Its Own Good? Cognitive Fidelity and Competence Leakage in LLM Simulations of Young Learners](https://arxiv.org/abs/2611.34042) | 本文提出并验证了“能力泄露”这一现象——LLM模拟的低龄学习者会表现出超出其被分配认知状态的知识与解题能力，且“能力过强”是交互中认知保真度错位的主要表现形式，这对教育类学习者模拟器的可靠性构成重要挑战。 |
| [^54] | [JHealthBench-L: Clinician-Reviewed Localization of HealthBench for Rubric-Based Medical LLM Evaluation in Japan](https://arxiv.org/abs/2611.34024) | 该论文提出了JHealthBench-L，一个通过LLM起草加多阶段临床医生审校流程、将HealthBench的评分准则本地化到日本医疗情境的基准，解决了机器翻译无法消除准则中区域性医疗假设从而导致评估不公的问题。 |
| [^55] | [Evaluating Ukrainian Grammatical Competence in Language Models: Probability Versus Prompting](https://arxiv.org/abs/2611.34016) | 该论文构建了一套包含1,444个乌克兰语最小对立对的语法评测协议，通过比较概率与直接提示两种方法评估19个语言模型的规范乌克兰语语法能力，并显式控制句子呈现顺序带来的答案位置效应。 |
| [^56] | [Residual Cross-Model Narrative Convergence under Shared Prompts: Auditing Prompt Controls and Narrative Measurement](https://arxiv.org/abs/2611.34015) | 本研究提出残差趋同指数（RCI≈0.41），通过比较相同提示词与语义匹配提示词下五个人工智能大模型生成故事的叙事距离，揭示了即便控制提示词内容后依然存在的可测量跨模型叙事趋同现象。 |
| [^57] | [Auto-Features: An LLM-Assisted Corpus for Contextual Recognition of Standard and Optional Car Features](https://arxiv.org/abs/2611.33974) | 该论文提出了Auto-Features语料库，将汽车配置识别任务建模为特征条件化的命名实体识别，构建了包含人工黄金层与LLM白银层的双层语料库，并提供了一套针对LLM各类标注失败逐一验证与补救的系统化质量控制流程。 |
| [^58] | [Fine-Tuning Without Gradients? Closed-Form, LLM-Written Classifiers on a Frozen Encoder](https://arxiv.org/abs/2611.33972) | 该论文提出闭式权重空间编程（WSP）框架，让大语言模型编写程序在冻结编码器上以闭式解拟合分类头，全程无需梯度更新和权重修改，却在多个标签预算下达到甚至超过全量微调、LoRA 等梯度方法的性能。 |
| [^59] | [DLaR: Dynamic Stochasticity Control for Latent Reasoning in Large Language Models](https://arxiv.org/abs/2611.33962) | 该论文提出DLaR框架，通过在token层面基于置信度-熵融合信号自适应调节扰动强度、在训练层面感知训练进程调优随机性，解决了潜在推理中均匀随机性次优和无约束扰动破坏连贯性的问题，实现探索与稳定性的动态平衡。 |
| [^60] | [PINN-Slicer: Exploiting Rotational Symmetry to Solve the Photon Orbit Equation Around a Black Hole](https://arxiv.org/abs/2611.33953) | 提出PINN-Slicer方法，利用二阶自治常微分方程在自变量平移下的不变性，将单个PINN逐片递归地部署到整个求解域，成功求解黑洞周围光子轨道方程，RMSE低于10^-3。 |
| [^61] | [What Does Skill Lift Actually Measure? A Source-Code-Level Audit and Noise-Aware Reanalysis of SkillsBench](https://arxiv.org/abs/2611.33943) | 该论文对 SkillsBench 进行源代码级审计并对 7,554 次公开试验做噪声感知重分析，发现部分“技能提升”分数实际上源于评分器依赖和操作性定义缺口等测量缺陷，并将总体平均提升修正为 18.7 pp。 |
| [^62] | [DiCALM: Converting Autoregressive Convolution-Attention Language Models into Discrete Diffusion Language Models](https://arxiv.org/abs/2611.33919) | DiCALM首次提出将混合卷积-注意力语言模型（LFM2家族）从自回归转换为掩码扩散生成的方案，通过双向扩展因果门控卷积、掩码预热和AR-扩散联合训练，在保持性能相当的同时实现平均1.4倍、最高1.79倍的推理加速。 |
| [^63] | [Looped Models are Fast Multi-Token Predictors](https://arxiv.org/abs/2611.33896) | 提出Floodgates自推测解码方法，在循环语言模型末尾附加掩码词块并通过低秩适应并行起草多个未来词，实现高达7.5倍的无损推理加速。 |
| [^64] | [Contextual language representations link narrative processing to language and literacy in children](https://arxiv.org/abs/2611.33871) | 本研究结合功能磁共振成像与大语言模型的情境嵌入表征，发现儿童聆听故事时的大脑叙事加工反应（尤其在颞顶和默认网络区域）可由LLM情境表征有效预测，从而将神经层面的叙事加工与儿童的语言和读写能力联系起来。 |
| [^65] | [ChemDeduce: Benchmarking LLMs for Unknown Chemical Identification through Interactive Simulated Experiments](https://arxiv.org/abs/2611.33848) | 该论文提出了ChemDeduce——一个将有状态模拟实验室中未知化学物质识别建模为序贯决策过程的交互式评估框架，首次系统性地测试了大语言模型通过主动设计实验、迭代推理来发现未知答案的能力。 |
| [^66] | [When Rerankers Work Together: Better and Faster Reranking without Joint Training](https://arxiv.org/abs/2611.33844) | 该研究提出两种无需联合训练的轻量级语义粗评分器，可与异构重排序器组合，在多阶段RAG流水线中显著提升答案准确率并加速在线剪枝。 |
| [^67] | [The Missing Root: Geometric Imagination Without Images in Large Language Models](https://arxiv.org/abs/2611.33841) | 该论文提出将包含位姿与带符号距离形状嵌入的连续物体token嵌入transformer序列，使大型语言模型无需依赖图像即可获得真正的空间几何理解能力。 |
| [^68] | [Teaching a Multilingual Translation Model Regional Varieties: Castilian and Latin American Spanish, Asturian, Aragonese and Aranese with MADLAD-400-3B-MT](https://arxiv.org/abs/2611.33831) | 本研究通过LoRA微调、回译和翻译重放技术，成功使MADLAD-400-3B-MT多语言翻译模型能够区分卡斯蒂利亚西班牙语与拉丁美洲西班牙语两种地区变体，并支持阿斯图里亚斯语、阿拉贡语和阿兰语三种伊比利亚半岛小语种的翻译。 |
| [^69] | [Pitfalls of Dependency-Ready Decoding for Masked Diffusion Language Models](https://arxiv.org/abs/2611.33823) | 本研究揭示了掩码扩散模型中“依赖就绪”解码策略的局限性：当该策略优于基于置信度或未来效用的基线方法时，其解码顺序实际上会退化成从左到右的顺序解码。 |
| [^70] | [Multimodal Safety Evaluation of Vision-Language Models in Japanese](https://arxiv.org/abs/2611.33789) | 该论文将多模态安全测试套件MSTS扩展至日语（第12种语言）并构建了包含82张日本文化本地化图像的评估资源，同时提出区分图像识别与安全判断的五分制评估框架，据此对五个日语视觉语言模型进行安全评估，发现各模型安全就绪度差异显著，且提供参考答案可提升LLM评判者的评估效果。 |
| [^71] | [Tells of a Lying Machine: Analyzing LLM-Generated Strategic Dialogue in Chinese Werewolf](https://arxiv.org/abs/2611.33747) | 该研究通过分析12个大语言模型在中国狼人杀游戏中的5,404条发言，发现狼人角色在确定性与不确定性语言标记上均显著高于村民，且人类标注者和LLM评审都能据此以60%至70%的准确率识别狼人，揭示了AI策略性欺骗行为的可检测语言特征。 |
| [^72] | [NORMCRAFT: Cultural Norm Generation for Context-Aware Language Model Alignment](https://arxiv.org/abs/2611.33726) | NORMCRAFT提出了一种将规范推断与响应生成解耦的模块化框架，通过多视角监督训练LLM作为文化规范生成器，无需下游响应标签即可实现更强的文化规范感知对齐。 |
| [^73] | [Sports Agent: A Self-Evolving System for Conversational Data Analytics](https://arxiv.org/abs/2611.33700) | Sports Agent 提出了一个由运行时反馈驱动的自进化框架，通过统一的反馈引擎结合 SQL-to-Question Scaling 与 Graph-Guided Scaling 等策略，使对话式体育数据分析智能体能够持续自我改进并保持回答的准确性与可靠性。 |
| [^74] | [Rethinking Sequential Locate-Then-Edit: Optimality and Stability](https://arxiv.org/abs/2611.33692) | 本文提出一种与批量编辑等价的顺序编辑方法，成功弥合了顺序编辑与批量编辑之间的性能差距，解决了现有顺序方法在数百次编辑后出现的灾难性遗忘和模型崩溃问题。 |
| [^75] | [Refusal Is Not Scored: A Coverage Limit in Content-Based Bias Evaluation](https://arxiv.org/abs/2611.33690) | 本文揭示了偏见审计中“先过滤拒答再评分”这一默认做法造成的测量盲区：拒答集中来自单一模型且针对特定国籍提示的频率显著更高，而部分评估工具会把拒答误判为无偏见内容，从而系统性低估模型偏见。 |
| [^76] | [Rethinking Knowledge Distillation for Diffusion Language Models](https://arxiv.org/abs/2611.33670) | 该论文首次系统性研究了扩散语言模型的知识蒸馏，揭示了前向KL散度蒸馏的两个反直觉失败现象，并提出了一种结合离策略与在策略蒸馏的统一无数据蒸馏框架，有效提升掩码扩散语言模型的性能。 |
| [^77] | [Verifiable but Not Predictable: What Sentence Embeddings Know About Propositional Decomposition](https://arxiv.org/abs/2611.33668) | 本文以PropSegmEnt语料库为基础，发现命题分解的两个质量控制问题处于不同表示空间：分解后可用NLI评判器高精度验证命题归属（准确率92.8%、AUC达0.977），而分解前无法通过句子嵌入的向量几何预测句子应产出的命题数量。 |
| [^78] | [Sparse-to-Sparse Feature Learning for Cross-Layer Analysis and Steering](https://arxiv.org/abs/2611.33640) | 提出稀疏到稀疏特征学习（S2S）方法，通过训练各层稀疏特征预测下一层的稀疏特征，在八个语言模型上实现了比现有基线更强的跨层特征对应关系，并能在控制输出分布偏移的条件下有效分析和引导模型行为。 |
| [^79] | [State Space Attention: Learning Context-Dependent Memory Retention](https://arxiv.org/abs/2611.33625) | 提出状态空间注意力（SSA），通过可学习的评分器自适应地决定保留哪些历史表示以维持有界内存，并配合“恢复”式监督样本训练技术，使模型能够依据上下文动态选择长上下文推理所需的记忆内容。 |
| [^80] | [Failure Broadens, Success Sharpens: Continual Adaptation of Tool-Use Agents Across Diverse Tasks with Dual-Polarity Memory](https://arxiv.org/abs/2611.33612) | 本文提出带周期性维护的双极性记忆框架DualPM，同时从成功与失败轨迹中生成程序性记忆并周期性整合以防记忆爆炸，实验发现失败记忆更能拓宽任务覆盖面、成功记忆更能精简执行过程。 |
| [^81] | [Aligned Machine: A Benchmark for Evaluating Multimodal Embedding Similarity via Human Comparative Judgements](https://arxiv.org/abs/2611.33594) | 该论文提出“对齐机器”这一交互式公众平台，通过让大众完成两两比较的强制选择任务，构建了可持续增长的人类对齐相似性基准（HASB），以评估嵌入模型的语义对齐能力。 |
| [^82] | [SEEL: Towards End-to-End 3D Sign Generation with Language Models](https://arxiv.org/abs/2611.33588) | 提出SEEL端到端框架，通过重建感知生成等机制将动作分词与语言模型生成过程和3D动作质量对齐，解决了以往两阶段流水线目标不匹配及直通估计器梯度偏差、码本崩溃的问题，实现更高质量的3D手语动作生成。 |
| [^83] | [PeerReveal: Learning to Generalize in AI Peer Review Detection](https://arxiv.org/abs/2611.33555) | 该论文提出了PeerReveal检测器，通过学习评审声明与同一论文AI生成评审的参考声明之间的关系来检测AI生成的同行评审，并在覆盖多个学科、融合多样化AI生成机制的PeerShift新基准上显著超越最强基线。 |
| [^84] | [LogToPersona: New Benchmark and Efficient Persona Prediction through Semantic Log Filtering](https://arxiv.org/abs/2611.33510) | 本文提出了LogToPersona任务及包含1600个人格-日志对的合成基准，通过语义日志过滤方法在端侧设备上实现高效的用户人格属性预测，内存占用最多降低39%且预填充时间减半以上。 |
| [^85] | [Using AI to Simulate the Behaviour of European Populations](https://arxiv.org/abs/2611.33503) | 该研究仅需在约40个欧洲国家公开的1200多万条群体级调查答案分布上微调开源语言模型，无需访谈或个人数据即可模拟欧洲人群的调查反应，判断群体差异方向的准确率达84%，超越传统人口统计模型并与Claude Opus等前沿大模型持平。 |
| [^86] | [Evaluating Large Language Models on conversational African American English: Feature classification and next-turn generation](https://arxiv.org/abs/2611.33499) | 本文基于CORAAL语料库中真实的会话式非裔美国人英语（AAE），通过特征分类和下一轮生成两项互补任务，评估了三种闭源大型语言模型在理解与生成AAE方面的能力。 |
| [^87] | [CA-RAG: Constraint-Aligned Retrieval for Structurally Faithful Long-Form Generation](https://arxiv.org/abs/2611.33497) | 提出 CA-RAG 框架，在证据检索前利用本体结构约束和基于政策的知识图谱关系增强章节级查询，显著提升合规敏感长文本生成中证据检索的精确率、召回率和生成合规性。 |
| [^88] | [Learning from Recurrence: Internalizing Depth-Induced Preferences in Looped Language Models](https://arxiv.org/abs/2611.33468) | 本文提出“偏好回写”方法，利用循环语言模型在更大深度下自身生成的预测来训练共享块（无需参考答案），从而将深层计算带来的偏好内化到单遍推理中，同时提升单遍与多遍推理的准确率。 |
| [^89] | [PAC Learning to Sample: On Computational Aspects of Language Generation](https://arxiv.org/abs/2611.33464) | 该论文将PAC学习框架与极限语言生成模型相结合，首次为语言生成任务提供了关于样本复杂度和时间复杂度的定量保证。 |
| [^90] | [AI-human hybrid conversations: Can AI Agents Improve Discourse Quality?](https://arxiv.org/abs/2611.33459) | 该论文提出了一个将通用型和人格设定型AI智能体引入动态讨论的框架，并通过基于Reddit的模拟实验和互补研究发现，AI智能体有望成为提升人机混合对话参与度与话语质量的催化剂，使对话更积极、毒性更低。 |
| [^91] | [HumblePRM: Improving VLM Abstention in Ambiguous Situations through Process Reward Modeling](https://arxiv.org/abs/2611.33452) | 提出轻量级过程奖励模型HumblePRM，将多模态推理分为感知与校准两阶段独立评分，通过奖励证据与置信度的一致性而非仅奖励结果正确性，提升视觉语言模型在模糊情境下的弃权能力。 |
| [^92] | [Open-Source FUTO Keyboard Layouts and Predictive Text for a Few Endangered Languages](https://arxiv.org/abs/2611.33432) | 本文为五种濒危乌拉尔语言构建了开源移动键盘布局与预测文本词典，通过结合翻译的芬兰语词元频率和有限状态转换器提取的形态频率，解决了缺乏大型语料库时屈折词形排序的难题。 |
| [^93] | [When the Knowledge Runs Out: A Part-Level Pilot Evaluation of How Language Models Handle Partially Covered Questions](https://arxiv.org/abs/2611.33428) | 该论文提出了一种对三部分问题逐部分评分的部件级初步评估方法，发现语言模型在参考资料仅部分覆盖问题时会默认用自身知识补充无依据内容（比率达0.71-0.90），而仅添加一句“只根据材料作答”的指令即可将该比率降至0.00-0.06。 |
| [^94] | [A Semantics-First Approach to Multimodal Video Captioning with Frozen Encoders and Stabilized Reinforcement Learning](https://arxiv.org/abs/2611.33409) | 提出AVQFormer架构，在27亿参数LLM上通过固定长度语义对齐模块与稳定化强化学习（ASCST）实现高效多模态视频描述，并揭示全局音频表征与视觉帧时序冲突导致的“时间对齐摩擦”现象。 |
| [^95] | [Partial Routing: Trace-Tuned Delegation for Resource-Efficient Agentic Workflows](https://arxiv.org/abs/2611.33402) | 该论文提出“部分路由”方法，通过挖掘主导模型与廉价辅助模型的历史运行轨迹来识别重复子任务并量化各自的质量与成本，从而生成委派指导，让主导模型在执行时动态地将部分子任务委派给更便宜的模型，在不训练任何模型的前提下同时提升任务性能并降低成本。 |
| [^96] | [Confidence scores for gated automation of large language models](https://arxiv.org/abs/2611.33394) | 该论文提出利用监督学习在冻结LLM的最终层隐藏状态上拟合线性置信度分数头（并为闭源模型训练仅文本的元模型），以预测LLM答案的正确性，从而实现高置信度回答的自动化处理和不确定回答的人工审查分流。 |
| [^97] | [Measuring and Predicting Image-Token Repair Boundaries in Vision-Language Models](https://arxiv.org/abs/2611.33388) | 该论文通过合成退化图像追踪视觉-语言模型中的修复机制，发现少量排序图像标记即可恢复大部分修复效应，并提出用干净图像灰度消融交叉点以1.53层误差独立预测修复边界。 |
| [^98] | [Temporal Weight Surgery: Repairing Video Grounding with LLM Editors](https://arxiv.org/abs/2611.33339) | 该论文提出时序权重手术（TWS），让通用LLM读取模型的权重与激活值，无需梯度即可定位并修复视频定位任务中时间预测错误的根源，显著提升模型在长视频上的定位精度。 |
| [^99] | [Who Attributes a Mind to Whom? Mutual Theory of Mind in Simulated User-Assistant Conversation](https://arxiv.org/abs/2611.33327) | 该研究提出了一个消息层面的四方向心智归因测量方法，并发现模拟用户的十三种特征中有九种（包括依恋焦虑等关系需求及部分人格特质）会增强用户对AI助手的心智归因。 |
| [^100] | [Building a Unified Inline Code Editing Model for GitHub Copilot](https://arxiv.org/abs/2611.33314) | 本文为GitHub Copilot提出统一行内代码编辑模型，通过锚定差异补丁的单一输出空间一次性预测光标补全、邻近编辑与远距离编辑三种类型，支持单次响应包含多个编辑，避免了原系统多次顺序调用不同模型的开销。 |
| [^101] | [Lost in Translation, Not in Confidence: Benchmarking and Calibrating Multilingual Decision Models](https://arxiv.org/abs/2611.33307) | 多语言决策模型在非英语语言中准确率显著下降（最多达8个百分点），而置信度仅下降一半，导致基于阈值的路由和防护系统失效，且增加数据或全参数微调均无法完全弥补这一跨语言性能差距。 |
| [^102] | [The Platonic Representation Hypothesis in Jacobian-Space: Language and Vision Models Agree Approximately Up to a Rotation](https://arxiv.org/abs/2611.33304) | 该研究将柏拉图表示假说从模型激活扩展到雅可比空间，发现语言模型与视觉模型不仅在内部表示上、而且在表示影响模型输出的方式（J空间）上也收敛，且这种一致性在近似相差一个旋转的意义上成立。 |
| [^103] | [Certify What You Plant: Shortcut-Free Proof Depth in Logical Reasoning Data](https://arxiv.org/abs/2611.33303) | 该论文提出了一种可验证的证明深度审计方法，通过计算问题真实所需的必需步骤、最短证明、最小证明数量和不可或缺前提，并生成可由独立检查器验证的证书，从而发现八个逻辑推理数据集中普遍存在标称深度与实际推理难度不符的问题。 |
| [^104] | [Two Timescales of Forgetting in Fine-tuning of Language Models: Fast Behavioral Suppression and Slow Representational Drift](https://arxiv.org/abs/2611.33302) | 本研究通过在随机层次模型上预训练并用受限规则子集微调transformer，发现语言模型微调中的遗忘具有两个时间尺度——表面行为被快速抑制，而支撑该行为的潜在表征则以缓慢漂移的方式逐渐丢失。 |
| [^105] | [FrId\'eo: French News Outlet Ideology Scoring via Multi-Source Evidence Fusion and Distancing-Aware Lexical Scoring](https://arxiv.org/abs/2611.33266) | 该论文提出了FrIdéo——首个覆盖法国新闻界全生态系统的连续意识形态评分系统，通过融合九个独立证据类别并引入能区分意识形态词汇直接使用与疏远性使用的对称评分方法，为30家法国新闻媒体提供透明且可解释的综合意识形态评分。 |
| [^106] | [Fast and Scalable Abstract Interpretation for Deep Neural Networks](https://arxiv.org/abs/2611.33259) | Scepter框架通过结构化语义将DNN抽象解释器的结构计算与数值求值分离，预先编译生成可复用的张量程序，在不损失精度的情况下实现快速且可扩展的神经网络性质认证。 |
| [^107] | [Focus On Facts: Stylistically Invariant and Factually Sensitive Text Embeddings](https://arxiv.org/abs/2611.33250) | 该论文构建了包含83,263个三元组的FOF-Bench基准，并通过难负样本三元组对比微调训练出80M参数的FOF-80M嵌入模型，使模型从偏好表面相似但事实被篡改的文本（偏好率仅0.42%）转变为可靠地偏好事实一致的风格改写文本（平均分离度从-0.1650提升至+0.0329）。 |
| [^108] | [CypherMAS: Benchmarking Routing Reliability in LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2611.33242) | 该论文提出CypherMAS合成基准，通过让智能体在固定图结构（链形、星形、二部图、环形）中将加密消息路由至持有匹配解码器的目标节点，首次实现了对LLM多智能体系统路由可靠性与冗余度的可量化、可隔离评估。 |
| [^109] | [Extraneous Cognitive Load in Large Language Models](https://arxiv.org/abs/2611.33197) | 该论文提出了首个系统地将信息呈现方式（外在认知负荷）与任务难度分离开来的受控基准CogLoadBench，并设计了基于模型激活值、在提示阶段即可计算的外在负荷分数（ELS），用于估计和诊断大语言模型的外在认知负荷。 |
| [^110] | [Verified experience improves materials prediction with large language models](https://arxiv.org/abs/2611.33195) | 该研究提出MatRecall框架，让大语言模型将经实验验证的预测经验（含证据、结果与误差）存储为情景记忆并无需重训即可调用，使材料预测任务的中位数性能较无记忆模型提升62%并超越专用模型。 |
| [^111] | [Semantic Density Does Not Predict Agent-Memory Collapse](https://arxiv.org/abs/2611.33184) | 本文通过大规模预注册实验发现，包括平均成对余弦相似度在内的记忆库统计指标均无法可靠预测智能体任务准确率的早期下降，即语义密度不能作为智能体记忆崩溃的有效预警信号。 |
| [^112] | [Psychophysics of Duration Perception in Multimodal Large Language Models](https://arxiv.org/abs/2611.33176) | 该论文首次借鉴心理物理学方法，将旋律减速识别作为时长感知的量化任务，系统测量了多模态大语言模型的听觉时长感知极限，发现仅七个模型能在上下文示例辅助下可靠识别被放慢的旋律且多数峰值接近人类中位水平，但其泛化能力仍存在明显局限。 |
| [^113] | [Feature-Based Explainability of Transformer Scoring Models: A Fidelity-First Framework for Automated Writing Assessment](https://arxiv.org/abs/2611.33175) | 本研究提出一种“保真度优先”的事后解释框架，通过61个可解释的语言与写作质量特征，比较四种解释方法对DeBERTa写作评分模型预测的复现能力，发现随机森林代理模型保真度最高，从而在确保解释忠实于模型预测之后再解读特征重要性，提升了自动写作评估的透明度。 |
| [^114] | [MGMem: An Efficient, Deterministic, and Provenance-Preserving Framework for Long-Horizon Agent Memory](https://arxiv.org/abs/2611.33174) | MGMem是一个无需生成式索引调用的确定性长期记忆框架，通过判别式解析器流水线和关联图组织跨会话记忆，在成本、可复现性和来源信息保留方面优于现有方法。 |
| [^115] | [The Bundesrat corpus: A dataset of political discourse in the German Federal Council](https://arxiv.org/abs/2611.33173) | 该论文构建了一个涵盖1949年至2026年德国联邦参议院1,067次全体会议、包含53,362篇发言人标注演讲的机器可读语料库，为计算政治学等跨学科研究提供了宝贵的数据资源。 |
| [^116] | [Modelling Billingual Learning Through LLM Training: A Critical-Period Exploration](https://arxiv.org/abs/2611.32997) | 该研究通过让小型GPT-2模型依次学习英语和不同距离的第二语言，并分析内部隐藏状态的余弦相似度，从模型内部表征角度探索了二语习得的关键期效应。 |
| [^117] | [When Ink Meets Pixel: Intent-Aware OCR with VLMs for Human-Corrected Financial Documents When Ink Meets Pixel: Intent-Aware OCR with VLMs for Human-Corrected Financial Documents](https://arxiv.org/abs/2611.32446) | 该论文提出“意图感知OCR”新任务，构建了包含1,600份金融文档、5,700余个视觉意图标注的DocIntent数据集及IntentBench评测基准，并训练出DocIntentOCR视觉语言模型，使OCR系统能够理解文档上的人工修正标记（如箭头、删除线等）并生成修正后的转录文本。 |
| [^118] | [ASTERISM: Do Multimodal LLMs Actually Look at All the Images?](https://arxiv.org/abs/2611.32316) | 该研究审计了19个多图像基准，发现约24%的题目存在“单图捷径”问题（仅凭一张图像即可作答），并据此构建了目前规模最大的经审计多图像基准ASTERISM（1.87万道题），专门保留真正需要多张图像才能解答的题目。 |
| [^119] | [How Much Dependency Can a Marginal Rule See? Total Correlation in Diffusion LM Parallel Decoding](https://arxiv.org/abs/2611.31990) | 该论文量化了扩散语言模型并行解码中基于边缘分布的解码器对提交token集合条件总相关性（即依赖性损失）的预测能力，发现其预测准确性主要来源于边界本身，且估计值在散文文本上通常比精确 KL 代价低 0.4 到 1.1 比特。 |
| [^120] | [Task-Dependent Linguistic Measures in Aphasia Discourse: A Comparative Analysis of Narrative, Descriptive, and Procedural Speech](https://arxiv.org/abs/2611.30826) | 该研究通过对61名慢性卒中后失语症患者在同一会话中完成的三种话语任务（程序性、描述性、叙事性）进行10项语言学测量指标的比较分析，证明话语任务类型本身会显著影响失语症的转录文本测量结果，因此不同任务得到的数值不能直接混用比较。 |
| [^121] | [Lowering the Bar for Everyone: Hidden Validation-Seeking Goals in Empathetic RL](https://arxiv.org/abs/2611.30702) | 在基于LLM模拟用户的共情强化学习中，寻求认同用户的占比每增加10个百分点，策略对有过错发帖人的认可率就上升3–4个百分点，表明策略会隐蔽地降低对所有人的认可门槛，而其辨别过错的能力并未下降。 |
| [^122] | [Rethinking the Evaluation of Contrastive Decoding for Hallucination Mitigation in Multimodal Large Language Models](https://arxiv.org/abs/2611.30460) | 该论文提出了CONCOR受控评估基准，利用对比接地边际直接衡量对比解码方法在多模态大语言模型幻觉缓解中的真实效果，克服了现有基准无法隔离底层对比调整作用的局限。 |
| [^123] | [Refine, Don't Reinvent: RASA-Bench for Source-Grounded Hindi Prose Refinement](https://arxiv.org/abs/2611.29750) | RASA-Bench 提出了一个针对印地语散文润色任务的新基准，通过忠实性门控加 Bradley-Terry 编辑偏好排名的四层评估协议，衡量模型在不偏离源文本前提下改进单薄 AI 草稿的能力，发现忠实性测试在短篇小说与有声小说等不同文体上的淘汰率差异显著。 |
| [^124] | [Bias Cascade: Understanding the Gap Between Knowledge Alignment and Effect Replication in LLM-based Social Simulation](https://arxiv.org/abs/2611.28854) | 该研究提出“偏差级联”这一失败机制，并构建了包含30个反事实社会实验的REFLEX基准，揭示了大语言模型即使具备准确的社会知识判断能力，仍可能无法复现经典社会研究的集体干预效应。 |
| [^125] | [Isolating the Language Shell: A Content-Controlled Study of Multilingual Reasoning](https://arxiv.org/abs/2611.27053) | 该研究通过保持推理内容不变、仅改变教师推理呈现的语言外壳，证明多语言推理蒸馏中外壳语言的选择会显著影响学生模型的下游准确率与输出语言行为，而预训练兼容性无法预测这些效应。 |
| [^126] | [Search From the Start: Search-Native Pretraining for Language Models](https://arxiv.org/abs/2611.26809) | 提出“搜索原生预训练”方法，通过大规模地将网络搜索查询及检索到的网页响应注入预训练语料，让语言模型在预训练阶段就学会发起搜索并利用检索证据，从而在知识密集型任务上取得更优性能，且该优势在监督微调后依然保持。 |
| [^127] | [AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators](https://arxiv.org/abs/2611.26073) | 提出了AltRA-Test统计检验方法，通过自助法重采样克服条目由不同标注者子集标注的限制，从而判断用LLM替代人类标注者在统计学上是否合理。 |
| [^128] | [NONSENSE: Non-Semantic Strings Expose Semanticity Gaps in Dense Retrieval](https://arxiv.org/abs/2611.22353) | 提出NONSENSE黑盒稀疏编辑探针，通过构造禁止复制查询词的明显非语义字符串，证明其稠密相似度得分常超过检索器最佳的查询相关段落，从而揭示了稠密检索中最近邻地位与真实语义含义相分离的语义性鸿沟。 |
| [^129] | [FIDEL: Abugida-Aware Pre-tokenization for Ethiopic Languages](https://arxiv.org/abs/2611.20216) | 本文提出 FIDEL——一族可逆的、感知埃塞俄比亚元音附标文字结构的预分词方案，通过显式揭示辅音与元音顺序结构来改进埃塞俄比亚各语言的分词质量，并在严格控制变量的条件下验证了其对分词统计与下游 NLP 任务的影响。 |
| [^130] | [The Silent Cost of Conversation History in LLM Reasoning](https://arxiv.org/abs/2611.20071) | 研究发现累积的对话历史会显著降低LLM在多轮任务中的推理准确率，且过程级质量指标无法提前预警这种下降，而用经求解器验证的状态摘要替代完整历史可挽回大部分性能损失。 |
| [^131] | [Envisioning Agentic Robotics: A New Era of Robotic Systems](https://arxiv.org/abs/2611.18359) | 本文提出“智能体机器人”这一新范式，主张由基础模型与具身框架构成的“具身系统2”是智能体机器人的决策核心，并呼吁学界建立统一定义、评估框架与研究路线图。 |
| [^132] | [Where Is Uncertainty Observed? A Mechanistic Understanding of Uncertainty Quantification in LLMs](https://arxiv.org/abs/2611.15735) | 该论文基于机制可解释性提出了观测不确定性的四种透镜框架（结构、表征、分布、语义），并揭示了观测点决定不确定性量化估计器的信息获取能力与恢复效果这一关键机制。 |
| [^133] | [Digital sovereignty in professional translation workflows: a case study of migrating from Windows to Linux](https://arxiv.org/abs/2610.41561) | 本研究提出以“控制、连续性、退出”三原则结合三项实践要求的数字主权评估框架，并通过将译者工作站从Windows迁移至Linux（采用LoopCAT、LibreOffice和本地运行的TranslateGemma）的案例研究，证明了专业翻译工作流可在可本地运行的Linux核心上实现重构。 |
| [^134] | [The Schema Coherence Inventory: Measuring Whether an Assigned Belief Organizes a Language Model's Cognition](https://arxiv.org/abs/2610.31683) | 本文提出图式连贯性量表（SCI），一种基于贝克认知模型的96题强制选择、无生成评分工具，用于验证赋予语言模型的信念是否真正组织其认知而非仅仅引发情绪，并通过目标特异性判别标准在五个骨干模型上得到验证。 |
| [^135] | [Explaining Chess and Go in Natural Language](https://arxiv.org/abs/2610.23166) | 本文提出了基于引擎评估的解释性指标和无需训练的BinProp方法，使语言模型能够发现可复用的棋类概念并以自然语言解释着法，平均得分相比直接提示提高近两倍，且该方法可泛化至围棋、跳棋和将棋等多种棋类。 |
| [^136] | [Behavioral Compositionality of Sparse Autoencoder Representations](https://arxiv.org/abs/2610.21261) | 论文提出SAE可加性残差（SAR）来度量稀疏自编码器特征在行为层面的组合性，发现学习到的SAE坐标比随机旋转具有更强的一阶可加性，但仍存在不可忽略的高阶交互项。 |
| [^137] | [Representation Redistribution in Sparse Autoencoders](https://arxiv.org/abs/2610.21240) | 该论文提出再分配敏感度分数（RSS）来识别在提示变换下以一致方向重复出现的大幅特征响应，并在涵盖八个语言模型的27个预训练SAE上验证了表征再分配的可复现性，同时将重复的表征变化与下游读出贡献及决策失败精确解耦。 |
| [^138] | [Frozen Factor or Spectral Band? In LoRA, the Factor Matters More at Low Rank](https://arxiv.org/abs/2610.06621) | 本文通过2x2交叉实验设计发现，在低秩（1-8）少步数设置下，LoRA变体冻结哪个因子（输入因子A显著优于输出因子B）比其选择的奇异方向带（主导或次主导）对性能影响更大。 |
| [^139] | [GlitchPatch: Repairing Glitch Tokens in Frozen Language Models via Local Retokenization](https://arxiv.org/abs/2610.04399) | GlitchPatch提出了一种基于局部重新分词的修复框架，无需访问模型内部即可在冻结的大语言模型外部通过优化输入分词来修复故障词元，并揭示了BPE合并规则删除中修复率与对正常词元附带损害之间的权衡规律。 |
| [^140] | [Do Language Models Need a Trainable Input Embedding Table? Fixed Minimal Token Codes at 1.7B-Class Scale](https://arxiv.org/abs/2610.04002) | 该研究证明在17亿参数规模下，语言模型的输入嵌入表可以完全用固定的16位标记编码（通过重复扩展至2048维）替代可训练嵌入，仅产生几个百分点的性能差距，其中行归一化编码在固定输入方案中表现最佳。 |
| [^141] | [Can Language Models Learn to Forecast Stock Prices](https://arxiv.org/abs/2609.36914) | 该论文在按时间顺序推进的股票价格沙盒中，通过对 Qwen3-4B 进行 SFT 与 PPO 后训练，探索语言模型能否学会利用价格、成交量等信息预测股票未来收益。 |
| [^142] | [Large-scale factor analysis shows machine intelligence is only partially interpretable](https://arxiv.org/abs/2609.36515) | 本论文对1,618个语言模型在456个纯文本基准上的13,251个评测分数进行了前所未有的大规模因子分析，发现语言模型的智能仅能被部分解释——一般智能因子约解释70.8%的方差，其余表现差异由领域特定的潜在因素驱动。 |
| [^143] | [AdaTutoRank: Learning to Rerank Document Sets via Adaptive Tutoring Optimization for RAG and Deep Research](https://arxiv.org/abs/2609.32472) | 该论文提出AdaTutoRank，针对文档集重排序中集合级标量奖励导致的监督稀疏与信用分配困难问题，通过自适应辅导优化为不同质量的rollout提供差异化指导，从而为RAG和深度研究筛选出完整、互补、无冗余的文档集合。 |
| [^144] | [Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models](https://arxiv.org/abs/2609.27373) | 该论文发现循环语言模型的注意力路由在早期循环步骤即趋于稳定，据此提出无需训练的WISE推理方法：早期用全局注意力发现稀疏工作集，后续步骤重用该支撑集，在保持推理质量的同时大幅减少重复的全局注意力计算。 |
| [^145] | [Large Language Model Agents for Evidence Based Genetic Disease Severity Classification](https://arxiv.org/abs/2609.19569) | 该研究开发了一个结合ReAct与RAG的自主AI智能体，基于ACMG严重程度指南和ACOG生活质量标准检索并验证文献证据，实现了对10,211个人类表型本体术语的遗传病严重程度自动化分类，表型分类准确率达93.55%，并汇总基因层面的严重程度以识别严重的常染色体隐性遗传基因对。 |
| [^146] | [Reliability-Aware Sexism Detection: Combining DPO with Annotator Agreement and Token-Level Confidence Scoring](https://arxiv.org/abs/2608.12330) | 本文提出RA-DPO方法，通过整合标注者一致性、模型置信度和词级不确定性来提升性别歧视检测的可靠性，并证明在少量高可靠性数据上训练即可达到全数据训练的同等效果。 |
| [^147] | [AVA-Encoder: Towards Agent-Native Video Representation Learning](https://arxiv.org/abs/2608.12313) | 本文提出AVA-Encoder框架，通过智能体自编码将视频转换为知识图谱表示并重建，实现了智能体可直接理解、查询和编辑的视频表示学习，克服了创意智能体从电影中学习的障碍。 |
| [^148] | [NOLLI: A Difficulty-Calibrated English-Korean Puzzle Benchmark](https://arxiv.org/abs/2608.04397) | 本文提出了NOLLI——首个程序化生成、难度经过行为校准的英韩双语谜题基准，通过7500个具有唯一解的可再生题目揭示出前沿模型在英韩直接翻译上表现相当，但在涉及韩文字母的文字改编任务上存在高达68.7个百分点的显著差距。 |
| [^149] | [MedGame: Storytelling Gamification Empowered by Large Language Models for Medical Education](https://arxiv.org/abs/2607.21570) | MedGame提出双引擎框架，将静态临床病例转化为结构化的可执行叙事游戏以支持以决策为中心的医学教育，并构建了包含5000个病例的MedGame Bench基准，实验证明任务特定微调可显著提升开源大语言模型的表现并缩小与商业模型的差距。 |
| [^150] | [Beyond Relevance-Centric Retrieval: Rubric-Oriented Document Set Selection and Ranking](https://arxiv.org/abs/2607.19747) | 该论文提出了一套完整的“评估-诊断-优化”框架，并构建了包含约2.8万条评分标准的三级九维文档集合评估基准SetwiseEvalKit，通过对12个重排序方法的系统评估揭示了现有方法在跨文档协调能力上的普遍缺陷，为面向大模型时代的文档集合选择与排序确立了新的评估范式。 |
| [^151] | [Information Discernment in Large Language Models](https://arxiv.org/abs/2607.19355) | 该论文提出“信息甄别”概念及基于三条规范性公理的Learn2Discern（L2D）基准框架，并通过13个模型近67万次试验与真实用户研究证实，当前大语言模型在来源甄别和真相甄别两个维度上均持续失败、表现接近随机水平。 |
| [^152] | [How Much is Left? LLMs Linearly Encode Their Remaining Output Length](https://arxiv.org/abs/2607.05316) | 本文发现大语言模型在隐藏状态中线性编码了剩余输出长度信息，仅需一个简单的线性探针即可准确预测生成回答还剩多少token，且当出现“Wait”等撤回token时预测长度会显著跳升。 |
| [^153] | [Safe Inference-Time Alignment via Lagrangian Reward Augmentation](https://arxiv.org/abs/2607.02781) | 提出LARA框架，通过拉格朗日对偶将安全约束转化为由小型校准集估计的对偶变量所定义的增强奖励，实现可即插即用于现有推理时对齐方法的无需权重更新的安全对齐。 |
| [^154] | [The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching](https://arxiv.org/abs/2606.27510) | 本文从因果中介分析出发重新推导了激活修补的估计量，揭示其中隐藏的交互效应会导致因果重要性依赖于其他组件状态的模型组件被错误地低估或夸大，且这种效应无法通过调整估计器或分析单位来消除。 |
| [^155] | [RepSelect: Robust LLM Unlearning via Representation Selectivity](https://arxiv.org/abs/2606.17168) | RepSelect通过压缩权重梯度主成分来隔离遗忘集特定表示，实现了既难以被微调恢复又不损害通用能力的鲁棒遗忘。 |
| [^156] | [Achieving Precise Text-To-Cypher Via Grounded Knowledge Graph Data Generation](https://arxiv.org/abs/2606.14325) | 本文提出一种基于知识图谱的自动合成数据生成方法，通过微调小型LLM显著提升Text2Cypher任务性能，使其媲美大型专有模型，从而在本地部署场景下保障数据主权且无需昂贵的标注。 |
| [^157] | [A History-Aware Visually Grounded Critic for Computer Use Agents](https://arxiv.org/abs/2606.11078) | 提出HiViG框架，通过将历史交互抽象为紧凑记录并提供视觉定位的动作评估，解决了现有评论模型健忘和缺乏视觉定位的缺陷，从而提升计算机使用智能体在GUI环境中的表现。 |
| [^158] | [SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference](https://arxiv.org/abs/2606.04511) | SparDA通过在Query、Key、Value之外引入与注意力查询解耦的Forecast投影来预测下一层所需的KV块，实现CPU到GPU预取与当前层计算重叠的前瞻式稀疏注意力选择，从而突破长上下文LLM推理中KV缓存容量与稀疏选择复杂度的双重瓶颈。 |
| [^159] | [Knowledge-Preserving LLM Architecture Standardization: Analytic Normalizer Mapping with Equal-Size Distillation](https://arxiv.org/abs/2605.25676) | 提出KEPT方法，通过直接参数复制、免训练的LayerNorm到RMSNorm解析初始化（OPM）和等尺寸蒸馏，在无法访问源模型预训练语料的情况下，将预训练LLM低成本地标准化为Llama家族架构，仅需约1.23亿token和4个A100 GPU天即可几乎无损地保留源模型能力。 |
| [^160] | [Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning](https://arxiv.org/abs/2605.22511) | Search-E1提出一个简单的自我进化框架，通过交替使用GRPO与在策略自蒸馏，将策略自身成功轨迹的搜索骨架作为特权上下文进行对齐，使搜索增强推理智能体无需外部教师即可从自身成功经验中持续自我改进。 |
| [^161] | [NSFL: A Post-Training Neuro-Symbolic Fuzzy Logic Framework for Boolean Operators in Neural Embeddings](https://arxiv.org/abs/2604.10604) | 提出免训练的神经符号模糊逻辑框架 NSFL，借助 NS-Delta 边际语义贡献与黎曼优化查询投影，使稠密检索器无需训练即可原生处理多原子布尔逻辑约束，检索性能最高提升 81%。 |
| [^162] | [BLADE: Better Language Answers through Dialogue and Explanations](https://arxiv.org/abs/2604.03236) | BLADE通过基于引用依据的对话引导而非直接答案，显著提升了学生在高级NLP课程中的测验表现，且学生更倾向于使用它。 |
| [^163] | [Cross-Lingual Gaps in Visual STEM Reasoning: Auditing VLMs Across Six Indian Languages](https://arxiv.org/abs/2603.26742) | 该论文构建了六种印度语言的视觉STEM推理审计基准，发现视觉语言模型从英语切换到印度语言时准确率下降9.8至24.9个百分点，且达罗毗荼语系语言的性能下降比印度-雅利安语系更严重。 |
| [^164] | [Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with Depth-Recurrent Transformers for Compositional Generalization](https://arxiv.org/abs/2603.21676) | 本文提出一种深度循环Transformer，通过迭代共享权重模块在不生成思维链token的情况下实现更深的测试时推理，以恒定内存和线性延迟达成内存高效推理，并在组合泛化任务上验证了其有效性。 |
| [^165] | [Fusian: Multi-LoRA Fusion for Fine-Grained Continuous MBTI Personality Control in Large Language Models](https://arxiv.org/abs/2603.15405) | Fusian框架通过在SFT过程中收集LoRA适配器轨迹序列，并利用强化学习训练策略网络动态融合这些冻结适配器，实现了对大语言模型MBTI人格特质强度的细粒度连续控制。 |
| [^166] | [Mechanistic Interpretability of Cognitive Complexity in LLMs via Linear Probing using Bloom's Taxonomy](https://arxiv.org/abs/2602.17229) | 本研究利用布鲁姆分类学与线性探测技术揭示，大语言模型在内部表征中以线性可分的方式编码认知复杂度层级，且这种结构在前向传播早期即已形成并随层数加深愈发清晰。 |
| [^167] | [Neural Isomorphic Fields: A Transformer-based Algebraic Numerical Embedding](https://arxiv.org/abs/2601.12095) | 该论文提出神经同构域（NIF），一种基于Transformer的数字嵌入方法，可证明地保留十进制分数的代数结构（与有理数域同构），从而解决神经网络处理极端数值时的溢出与不稳定问题。 |
| [^168] | [Semantic Gravity Wells: Why Negative Constraints Backfire](https://arxiv.org/abs/2601.08070) | 本文提出“语义压力”这一量化指标，通过机制分析揭示大语言模型违背负面指令的概率随其生成被禁词的内在倾向呈Logistic关系上升（从5%升至44%），失败的原因在于指令诱导的抑制被模型内在生成倾向所压倒。 |
| [^169] | [Luxical: High-Speed Lexical-Dense Text Embeddings](https://arxiv.org/abs/2512.09015) | Luxical通过结合稀疏TF-IDF特征、小型ReLU网络和知识蒸馏训练，以极低的计算成本近似大型Transformer嵌入模型，实现了面向网络规模文本组织的高速“词汇密集”文本嵌入。 |
| [^170] | [Network of Theseus (like the ship)](https://arxiv.org/abs/2512.04198) | 提出忒修斯之网络方法，通过逐步替换组件并用表征相似性度量对齐，在保持性能的前提下将训练好的网络架构渐进转换为完全不同的目标架构，打破了“训练架构即部署架构”的传统假设。 |
| [^171] | [Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](https://arxiv.org/abs/2511.18467) | 该论文首次系统揭示了LLM多智能体软件开发系统在恶意用户与恶意智能体两种场景下的安全风险，提出隐式恶意行为注入攻击IMBIA（攻击成功率最高达93%）及其防御机制Adv-IMBIA。 |
| [^172] | [AI-Mediated Communication Can Reduce Opinion-Based Network Segregation](https://arxiv.org/abs/2510.21984) | 本研究通过一项大规模预注册在线实验发现，AI中介交流能够影响人们在观点讨论中的社交重组模式，从而减少基于观点差异的网络隔离。 |
| [^173] | [GDS Agent for Graph Algorithmic Reasoning](https://arxiv.org/abs/2508.20637) | GDS 智能体通过 MCP 服务器将一整套图算法封装为工具并提供预处理与后处理流程，使任何现代 LLM 都能开箱即用地对大规模图结构数据进行算法推理。 |
| [^174] | [The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs](https://arxiv.org/abs/2506.21621) | 该论文提出了包含超过5,000个经人工评估的LLM生成数学证明的开放证明语料库（OPC），这是首个包含大量针对USAMO和IMO等顶级数学竞赛问题的正确LLM证明解决方案的大规模数据集，为自动证明生成研究提供了重要基础。 |
| [^175] | [MathArena: Evaluating LLMs on Uncontaminated Math Competitions](https://arxiv.org/abs/2505.23281) | 提出 MathArena 基准，利用定期举办的数学竞赛在新题发布后立即评估大语言模型，有效消除数据污染风险，并发现 AIME 2024 存在明显的污染迹象。 |
| [^176] | [United in Diversity? Contextual Biases in LLM-Based Predictions of the 2024 European Parliament Elections](https://arxiv.org/abs/2409.09045) | 本研究在2024年欧洲议会选举前利用三个大语言模型基于26,000名欧洲选民的个体背景信息预测其投票行为，发现LLM预测与实际结果相比大多失败且准确性在不同情境间分布不均，揭示了LLM舆论预测中存在的情境依赖性偏差。 |
| [^177] | [Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation](https://arxiv.org/abs/2409.00696) | Polyrating是一种基于最大后验估计的低成本评分系统，能够检测并量化人类评估中的偏差，将大语言模型的人工评估成本对新模型和新任务分别最高降低41%和77%，同时支持跨任务的公平模型比较。 |

# 详细

[^1]: 分词化不属于APX复杂度类

    Tokenisation is not in APX

    [https://arxiv.org/abs/2611.34938](https://arxiv.org/abs/2611.34938)

    本文证明除非P=NP，分词化问题无法在多项式时间内近似到最优解的任意常数因子（即不属于APX类），且直接分词即使在一元字符表上也是APX难的。

    

    在本文中，我们证明，除非P=NP，否则不存在任何多项式时间算法能够找到一个分词器，使其目标值在最优值的任意常数因子范围内，这意味着分词化问题不属于APX复杂度类。我们对直接分词和自底向上分词均证明了这一结论，涵盖压缩目标和unigram对数似然目标两种情形，以及二元或更大字符表的情况。我们进一步证明，即使在一元字符表上，直接分词也是APX难的，因此不存在多项式时间近似方案。

    arXiv:2611.34938v1 Announce Type: new  Abstract: In this paper, we prove that, unless P=NP, no polynomial-time algorithm can find a tokeniser whose objective is within any constant factor of the optimum, meaning that tokenisation is not in the APX complexity class. We prove this for direct and bottom-up tokenisation, under both a compression or unigram log-likelihood objectives, and for binary or larger alphabets. We further show that, even over unary alphabets, direct tokenisation is APX-hard, and thus admits no polynomial-time approximation scheme.
    
[^2]: 图像生成能否帮助视觉推理？一项实证研究

    Can image generation help visual reasoning? An empirical study

    [https://arxiv.org/abs/2611.34935](https://arxiv.org/abs/2611.34935)

    这项实证研究发现，通过生成辅助图像来帮助视觉推理在大多数情况下反而会降低性能——在13组视觉语言模型与生成器的配对实验中平均准确率下降3.7个百分点，十个任务中九个出现下降——挑战了“多模态模型可以用图像思考”的假设。

    

    语言模型用词语思考，那么多模态模型是否可能用图像思考？我们测试了生成辅助图像能否改善视觉问答的表现。模型首先描述它所需要的辅助图像，然后结合原始图像和生成图像一起回答问题。在13组视觉语言模型与生成器的配对实验中，图像生成使平均准确率下降了3.7个百分点，十个任务中有九个出现下降。两个统一模型BAGEL和Janus-Pro-7B各自在八个任务上损失了准确率。在VSTAR数据集上，在相同的规划提示下额外加入一张生成图像会使准确率降低10.47个百分点。重复的图像编辑在不同任务上产生了不同结果：ChartQAPro和MathVista始终低于无生成基线，而ERQA在十次编辑后提升了三个百分点。

    arXiv:2611.34935v1 Announce Type: cross  Abstract: Language models think in words; then multimodal models might think in images. We test whether a generated helper improves the answer to a visual question. A model describes the helper it needs and then answers using the original and generated images together. Across 13 vision-language model and generator pairs, generation lowers mean accuracy by 3.7 percentage points, with declines on nine of ten tasks. Two unified models, BAGEL and Janus-Pro-7B, each lose accuracy on eight tasks. On VSTAR, adding a generated image to the same planning hint lowers accuracy by 10.47 points. Repeated image edits yield different outcomes across tasks. ChartQAPro and MathVista stay below their no-generation baselines, while ERQA gains three points after ten edits.
    
[^3]: 面向复杂数学问题的多样化替代推理轨迹

    Diverse Alternative Reasoning Traces for Hard Math Problems

    [https://arxiv.org/abs/2611.34928](https://arxiv.org/abs/2611.34928)

    提出了DART轨迹挖掘方法，通过分解被拒绝的推理轨迹并基于位置加权不确定性与参考答案引导生成刻意不同的替代推理步骤，在相同生成成本下比蒙特卡洛采样多恢复37%的复杂数学问题，且每条合格轨迹成本降低30%。

    

    针对数学推理的大语言模型（LLM）后训练通常依赖于最终答案能通过验证器检验的合格轨迹。然而，对于复杂数学问题，在独立蒙特卡洛（MC）采样下，这类轨迹可能非常稀少。为了解决这一问题，我们提出了多样化替代推理轨迹（Diverse Alternative Reasoning Traces，DART），这是一种轨迹挖掘方法：它从一条被拒绝的轨迹出发，将其分解为推理步骤，并按位置加权不确定性进行排序；随后利用参考答案引导和元思考（meta-thought），为排名靠前的步骤生成刻意不同的替代候选，最后完成由此产生的完整轨迹。在贪心解码未能解决的DeepMath-103K问题上，在生成轨迹数量相同的条件下，DART比MC采样多解决37%的问题；其每个已恢复问题的估计生成成本几乎相同，而每条合格轨迹的成本降低了30%。在DART挖掘的轨迹上对Llama-3.1-8B-Instruct进行后训练取得了显著提升。

    arXiv:2611.34928v1 Announce Type: cross  Abstract: Post-training large language models (LLMs) for mathematical reasoning often relies on accepted traces whose final answers pass a verifier. For hard math problems, however, such traces can be rare under independent Monte Carlo (MC) sampling. To address this, we propose Diverse Alternative Reasoning Traces (DART), a trace-mining method that starts from a rejected trace, decomposes it into reasoning steps, and ranks them by position-weighted uncertainty. It then generates deliberately distinct alternative candidates for highly ranked steps using reference-answer guidance and a meta-thought before completing the resulting traces. On greedy-unsolved DeepMath-103K problems, DART recovers 37% more problems than MC sampling at a matched number of generated traces. Its estimated generation cost per recovered problem is nearly identical, while its cost per accepted trace is 30% lower. Post-training Llama-3.1-8B-Instruct on traces mined by DART a
    
[^4]: 混合循环Transformer

    Mixture-of-Loops Transformer

    [https://arxiv.org/abs/2611.34923](https://arxiv.org/abs/2611.34923)

    提出混合循环Transformer（MoLT），以KV缓存作为跨深度和跨循环的选择性记忆重用机制，在表达能力与效率之间取得新平衡，并支持超越训练循环次数的高效解码。

    

    循环Transformer通过在多个循环中重用共享层来提升表达能力，但代价是效率的下降。要推动其表达能力与效率的边界，就需要重新思考连接相邻循环的核心机制：当Transformer进行循环时，应该传递什么？我们提出了混合循环Transformer（MoLT），其混合循环注意力机制（MoLA）将键值对（KV）用作记忆，实现跨深度和跨循环的选择性重用。在循环残差模式下，MoLT-R保留残差流并跨循环读取局部KV，以增强表达能力。在循环KV模式下，MoLT-KV则从嵌入层重新开始，读取当前循环的局部KV以及前面循环在较早位置的记忆KV，以提升效率。MoLT-KV消除了同一位置上的跨循环依赖，并结合用于并行训练的循环强制技术，实现了超越训练循环次数的单次遍历混合循环（MoL）解码。为了平衡效率和质量，我们引入了U-

    arXiv:2611.34923v1 Announce Type: cross  Abstract: Looped Transformers boost expressivity by reusing shared layers across loops, yet at the cost of efficiency. Pushing their expressivity-efficiency frontier requires rethinking the core mechanism connecting adjacent loops: What should be passed as a Transformer loops? We introduce Mixture-of-Loops Transformer (MoLT), whose Mixture-of-Loops Attention (MoLA) uses keys and values (KV) as memory for selective reuse across depths and loops. Under loop residual, MoLT-R retains the residual stream and reads local KV across loops for expressivity. Under loop KV, MoLT-KV instead restarts from embeddings and reads current-loop local KV with preceding-loop memory KV at earlier positions for efficiency. MoLT-KV eliminates same-position cross-loop dependence and, paired with loop forcing for parallel training, unlocks single-traversal Mixture-of-Loops (MoL) decoding beyond the training loop horizon. To balance efficiency and quality, we introduce U-
    
[^5]: μ-Bench：面向共享交互环境中语音代理多用户任务完成的基准测试

    $\mu$-Bench: Benchmarking Multi-User Task Completion in Shared Interaction Voice Agents

    [https://arxiv.org/abs/2611.34920](https://arxiv.org/abs/2611.34920)

    提出了 μ-Bench 基准，用于评估共享空间中闭环全双工多用户语音交互下的联合任务完成能力，通过 JTS 指标与 ATC 诊断揭示现有语音代理在多用户权限、隐私保护与协调方面仍存在显著不足。

    

    共享空间中的语音代理必须满足相互依赖的多用户请求，同时维护每位用户的权限、隐私和任务约束。当参与者在整个会话过程中持续做出反应时，现有评估方法的覆盖范围十分有限。我们提出了 μ-Bench，这是一个用于闭环、全双工多用户语音交互中联合任务完成的基准测试。模拟的参与者基于各自隔离的目标、知识和权限配置行事，并会根据代理的语音和工具执行结果做出动态调整。该基准涵盖 20 个领域，包含 100 个场景，通过改变群体规模、声学环境和后续追问需求，构成了 800 个测试案例（408 个带工具，392 个无工具）。联合任务成功率要求所有用户目标、隐私边界和终止状态同时满足，而 Pure JTS 进一步禁止未经授权的工具调用；归因、多线程与协调诊断则用于定位失败的具体原因。在对 13 个语音代理的评估中，专有系统在无工具场景下最高达到 62.24%……

    arXiv:2611.34920v1 Announce Type: cross  Abstract: Voice agents in shared spaces must satisfy interdependent multi-user requests while upholding per-user authority, privacy, and task constraints. Existing evaluations offer limited coverage when participants react throughout a session. We introduce $\mu$-Bench, a benchmark for joint task completion in closed-loop, full-duplex multi-user voice interaction. Simulated participants act from isolated profiles of goals, knowledge, and permissions, adapting to the agent's speech and tool outcomes. Across 20 domains, 100 scenarios vary group size, acoustics, and follow-up demands to form 800 cases (408 tool-augmented, 392 tool-free). Joint Task Success (JTS) requires all user goals, privacy boundaries, and terminal states to hold jointly, while Pure JTS further disallows unauthorized tool calls; Attribution, Threading, and Coordination (ATC) diagnostics localize failures. Across 13 voice agents, proprietary systems peak at 62.24\% tool-free and
    
[^6]: DRBench Pro：面向专业深度研究智能体的以知识为中心的基准测试

    DRBench Pro: A Knowledge-Centric Benchmark for Professional Deep Research Agents

    [https://arxiv.org/abs/2611.34852](https://arxiv.org/abs/2611.34852)

    DRBench Pro是一个以结构化知识覆盖为核心评测目标的深度研究智能体基准测试，通过从维基百科构建知识树生成研究查询与证据评分标准，从广度和深度两个维度（取几何平均）衡量智能体对复杂开放性问题的研究覆盖能力。

    

    深度研究智能体从多个来源收集信息，并将其综合成关于复杂、开放式问题的报告。现有的基准测试从多个维度评估报告质量，但缺乏专门以知识为中心的评估，即报告涵盖了哪些研究方向以及对其探索的深度如何。我们提出了DRBench Pro，一个将结构化知识覆盖作为主要评估目标的基准测试。我们的自动化流程从维基百科构建局部知识树，并使用采样的子图来生成研究查询和基于证据的评分标准。每个评分标准对应知识树中的一个关系，从而保留了被评估知识的结构。智能体只接收研究查询。这种结构使我们能够衡量跨主要研究方向的广度以及沿相关知识链条的深度，两者的几何平均值即为整体覆盖率。该基准测试包含跨越24个……的144个案例（摘要在此处截断）。

    arXiv:2611.34852v1 Announce Type: cross  Abstract: Deep research agents gather information from multiple sources and synthesize it into reports on complex, open-ended questions. Existing benchmarks assess report quality along several dimensions, but lack a dedicated knowledge-centric evaluation of which research directions a report covers and how deeply it explores them. We introduce DRBench Pro, a benchmark that makes structured knowledge coverage the primary evaluation target. Our automated pipeline builds local knowledge trees from Wikipedia and uses sampled subgraphs to generate research queries and evidence-grounded rubrics. Each rubric corresponds to a relation in the tree, preserving the structure of the knowledge being evaluated. Agents receive only the research query. This structure allows us to measure breadth across major research directions and depth along related knowledge chains, with their geometric mean giving overall coverage. The benchmark includes 144 cases across 24
    
[^7]: 谚语：窥探集体规约的一扇窗口

    Proverbs As A Window On Collective Prescriptions

    [https://arxiv.org/abs/2611.34832](https://arxiv.org/abs/2611.34832)

    该研究提出了一个基于大语言模型（经人类判断验证）的计算框架，首次对近5万条英语谚语所承载的集体规约进行量化分析，发现多数谚语在个人可控领域具有规约性，并普遍倡导谨慎、长期主义、诚实与善良等价值观。

    

    谚语作为传统智慧的宝库，数千年来一直为人类推理提供指引。然而，它们所承载的集体规约在很大程度上尚未被量化。我们提出了一个计算框架来分析近50,000条英语谚语。利用经过人类判断验证的大语言模型，我们绘制了谚语的主题分布图，评估了其规约力量，并量化了它们在人类判断与行为关键维度上的建议。我们发现，大多数谚语都具有规约性，在个人可控的领域（如做好准备与审慎行动）提供指导，而关于不可控境况（如命运与死亡）的谚语则倾向于描述而非规约。当规约性谚语采取一致的立场时，它们更倾向于谨慎而非冒险、长期结果而非短期结果、诚实而非欺骗、善良而非恶意，并表现出中等程度的倾向……（原文摘要在此截断）

    arXiv:2611.34832v1 Announce Type: new  Abstract: Proverbs, as repositories of traditional wisdom, have informed human reasoning for millennia. Yet the collective prescriptions they encode have remained largely unquantified. We introduce a computational framework to analyze nearly 50,000 English-language proverbs. Using large language models, validated against human judgments, we map the topical landscape of proverbs, assess their prescriptive power, and quantify their recommendations on key dimensions of human judgment and behavior. We find that most proverbs are prescriptive, offering guidance in domains under individual control, such as preparation and prudent action, whereas proverbs about uncontrollable circumstances, such as fate and mortality, tend to describe rather than prescribe. When prescriptive proverbs take a consistent stance, they favor caution over risk, long-term over short-term outcomes, honesty over deception, and kindness over malice, and show a moderate tendency to
    
[^8]: 修复蒸馏语言模型的可信性

    Repairing the Trustworthiness of Distilled Language Models

    [https://arxiv.org/abs/2611.34831](https://arxiv.org/abs/2611.34831)

    提出基于近端Bregman响应函数（PBRF）的针对性修复框架，无需原始蒸馏语料库即可修复蒸馏语言模型的可信性退化，并通过比值比目标扩展至偏好监督，同时约束模型能力的变化。

    

    知识蒸馏使得大型语言模型能够通过较小的学生模型进行部署，但压缩过程即使在下游性能得以保持的情况下，也可能引入可信性方面的退化。修复这类失败具有挑战性，因为原始蒸馏语料库通常不可用，而直接偏好优化可能会改变从教师模型继承的能力。我们提出了一种基于近端Bregman响应函数（PBRF）的针对性修复框架，以解决这两个限制。我们使用替代代理数据集来估计PBRF所需的曲率，从而消除对原始迁移语料库的依赖，并通过比值比目标将框架扩展到基于偏好的监督。这使得可以使用偏好和不偏好响应对，同时约束对蒸馏模型的更改。我们的结果表明，所提出的方法能够改善目标可信性

    arXiv:2611.34831v1 Announce Type: cross  Abstract: Knowledge distillation enables the deployment of large language models through smaller student models, but the compression process can introduce regressions in trustworthiness even when downstream performance is preserved. Repairing such failures is challenging because the original distillation corpus is often unavailable, while direct preference optimization may alter capabilities inherited from the teacher. We propose a targeted repair framework based on the proximal Bregman response function (PBRF) that addresses both limitations. We estimate the curvature required for PBRF using a surrogate proxy dataset, removing dependence on the original transfer corpus, and extend the framework to preference-based supervision through an odds-ratio objective. This enables the use of preferred and dispreferred response pairs while constraining changes to the distilled model. Our results show that the proposed approach improves targeted trustworth
    
[^9]: 语言知识基准：语言模型是否掌握早期年级读写能力的根基？

    The Linguistic Knowledge Benchmark: Do Language Models Know the Foundations of Early-Grade Literacy?

    [https://arxiv.org/abs/2611.34823](https://arxiv.org/abs/2611.34823)

    该论文提出了语言知识基准及一套可复现的自动化流程，将课程文件转化为涵盖九个语言学领域和七个年级的审核题目集，用于检验语言模型是否具备支持低收入和中等收入国家早期读写教学所需的语言学知识，并以基斯瓦希里语和英语为实例评估了两百多个模型。

    

    arXiv:2611.34823v1 公告类型：新论文 摘要：在低收入和中等收入国家（LMICs），许多教学语言在用于训练语言模型的数据中代表性不足。在这些模型能够支持此类语言的早期年级读写教育之前，我们需要了解它们是否掌握读写教学所依赖的每种语言的语言学知识。为了检验这一点，我们提出了语言知识基准以及一个可复现、大部分自动化的流程，该流程将课程文件、教师指南和学生用书转化为经过审核的题目集，题目按固定的网格组织，涵盖九个语言学领域和七个年级。该流程结合了LLM驱动的源文献调查、特征整合、题目生成和质量控制，并辅以受项目反应理论启发的题目选择步骤和最终的专家评审。我们为基斯瓦希里语（肯尼亚；135道题）和英语（英国；137道题）实例化了该流程，并分别评估了121个和119个模型。前沿模型表……

    arXiv:2611.34823v1 Announce Type: new  Abstract: Many languages of instruction in low- and middle-income countries (LMICs) are poorly represented in the data used to train language models. Before these models can support early-grade literacy in such languages, we need to know whether they hold the linguistic knowledge of each language that literacy teaching relies on. To test this, we present the Linguistic Knowledge Benchmark and the reproducible, largely automated pipeline that turns curriculum documents, teacher guides and pupil books into a vetted question set organised on a fixed grid of nine linguistic domains and seven grade levels. The pipeline combines LLM-driven source survey, feature consolidation, item generation and quality control with an item-selection step inspired by item response theory and a final expert review. We instantiate it for Kiswahili (Kenya; 135 items) and English (United Kingdom; 137 items) and evaluate 121 and 119 models respectively. Frontier models exce
    
[^10]: VulnSafe：测量大语言模型中的脆弱性感知关系安全

    VulnSafe: Measuring Vulnerability-Aware Relational Safety in Large Language Models

    [https://arxiv.org/abs/2611.34819](https://arxiv.org/abs/2611.34819)

    VulnSafe提出了一个以脆弱性感知响应能力（VARC）为核心的中文评估基准，通过41个场景、2050个平衡提示词，从风险感知、安全维护、自主性保护、支持性响应和边界管理五个维度系统评估大语言模型的关系安全能力，并区分主动的“应该做”与禁止性的“不应该做”行为。

    

    情感支持型交互要求大语言模型不仅要避免明显有害的内容，还要识别与脆弱性相关的风险、调节自身的关系立场、保护用户的自主性，并提供符合情境的支持。现有的安全基准通常测量孤立的违规行为，导致其分数所代表的底层响应能力并不明确。我们提出了VulnSafe，一个以脆弱性感知响应能力（VARC）为核心的中文能力评估。VARC将关系安全组织为五个互补的构念：风险感知、安全维护、自主性保护、支持性响应和边界管理。遵循以证据为中心的设计，41个场景通过2,050个平衡的提示词和针对场景的评分细则来引出与构念相关的行为。该评估进一步将主动的“应该做”行为与禁止性的“不应该做”行为区分开来，并在八（原文此处截断）

    arXiv:2611.34819v1 Announce Type: new  Abstract: Emotionally supportive interactions require large language models not only to avoid overtly harmful content, but also to recognize vulnerability-related risk, regulate their relational stance, preserve user agency, and provide context-appropriate support. Existing safety benchmarks often measure isolated violations, leaving unclear what underlying response competence their scores represent. We introduce VulnSafe, a Chinese capability assessment centered on Vulnerability-Aware Response Competence (VARC). VARC organizes relational safety into five complementary constructs: Risk Perception, Safety Preservation, Autonomy Protection, Supportive Response, and Boundary Management. Following an evidence-centered design, 41 scenarios elicit construct-relevant behavior through 2,050 balanced prompts and scenario-specific scoring rubrics. The assessment further separates proactive should-do behavior from prohibitive should-not behavior. Across eigh
    
[^11]: PersonaFactory：基于合成人格库的风格化用户提示增强

    PersonaFactory: Stylistic User Prompt Augmentation with Synthetic Persona Bank

    [https://arxiv.org/abs/2611.34785](https://arxiv.org/abs/2611.34785)

    该论文提出PersonaFactory端到端流程，构建了超74万个刻画背景与交流行为的结构化合成人格库，并以85种语言生成约124万条人格风格化对话，解决了合成用户提示风格单一、无法模拟真实用户交流机制的问题。

    

    大型语言模型在AI开发流程中越来越多地取代真人，被用于大规模生成训练对话以及驱动对齐与评估。然而，开箱即用的模型所生成的合成用户提示往往语言单一、风格平淡，无法复现真实的用户-AI交流机制。我们提出了PersonaFactory，这是一个端到端的流程，它可以生成详细的用户人格画像，并从这些人格的视角对现有用户提示进行增强。该方法产出了一个包含超过74万个结构化合成人格的库，详细刻画了每个角色的背景与交流行为；同时生成了一个覆盖85种语言、约124万条人格风格化合成对话的语料库。我们评估了个人特征对风格采纳和意图保留的影响，使用LLM裁判和启发式指标来衡量增强的保真度，以及其对助手人格定制的下游影响。

    arXiv:2611.34785v1 Announce Type: cross  Abstract: Large language models increasingly replace real people in the AI development pipeline, as they are used to generate training dialogues at scale and to drive alignment and evaluation. However, out-of-the-box models produce monolingual and stylistically flat texts as synthetic user prompts, failing to replicate the real user-AI communication mechanics. We present PersonaFactory, an end-to-end pipeline that generates detailed user personas and uses them to augment existing user prompts from their point of view, producing a bank of over 740000 structured synthetic personas detailing their background and communication behaviors, and a corpus of roughly 1.24 million persona-styled synthetic conversations spanning 85 languages. We evaluate the effect of personal features on style adoption and intent preservation, using an LLM-judge and heuristic metrics to assess augmentation fidelity together with the downstream impact on assistant persona c
    
[^12]: 当每次轨迹采样都失败时：将学习者条件化的选项拼接进长程智能体强化学习

    When Every Rollout Fails: Splicing Learner-Conditioned Options into Long-Horizon Agent RL

    [https://arxiv.org/abs/2611.34777](https://arxiv.org/abs/2611.34777)

    提出SPLICE方法，在线地从全失败的当前策略轨迹中拼接由学习者条件化的局部选项作为训练目标，解决了长程智能体强化学习中全失败组导致无奖励策略梯度的问题。

    

    基于可验证奖励的分组强化学习在长程智能体任务上经常产生全失败组：每条采样轨迹都获得零奖励，使得组相对类方法缺乏基于奖励的策略梯度。一种常见的补救措施是由更强的模型提供完整解答，但完整轨迹代价高昂，且会让学习者暴露于冗长的离策略历史中。我们提出SPLICE，它在线地从失败的当前策略轨迹中构建局部训练目标。对于每个全失败组，一个提议模型在当前策略轨迹中选择一个拼接点，并从该点生成若干有界选项。我们利用重复的当前策略延续来估计每个选项的价值，将正向估计转换为按回报加权的局部目标，并在这些选项上拟合当前策略。我们在两个长程智能体任务（基于SWE-bench的软件工程等）上，跨两个策略家族对SPLICE进行了评估。

    arXiv:2611.34777v1 Announce Type: cross  Abstract: Group-based reinforcement learning with verifiable rewards often yields all-fail groups on long-horizon agent tasks: every sampled trajectory receives zero reward, leaving group-relative methods without a reward-based policy gradient. A common remedy supplies complete solutions from a stronger model, but full trajectories are expensive and expose the learner to long off-policy histories. We introduce SPLICE, which constructs local training targets from failed current-policy trajectories online. For each all-fail group, a proposal model selects a splice point in a current-policy trajectory and generates several bounded options from that point. We estimate each option's value using repeated current-policy continuations, convert positive estimates into a return-weighted local target, and fit the current policy on the options. We evaluate SPLICE across two policy families on two long-horizon agent tasks, Software Engineering with SWE-bench
    
[^13]: DFly2：通过共享目标模型的稀疏性来加速长上下文推测解码

    DFly2: Turbocharging Long-Context Speculative Decoding by Sharing the Target's Sparsity

    [https://arxiv.org/abs/2611.34776](https://arxiv.org/abs/2611.34776)

    该论文提出 DFly2，让长上下文推测解码中的草稿模型直接复用原生稀疏目标模型已有的缓存及逐查询的位置选择，免去草稿模型自身的缓存开销与选择计算，从而大幅加速长上下文解码。

    

    为了避免在长上下文上进行稠密注意力所带来的二次方开销，DeepSeek-V4 和 Hy4 等模型已转向原生稀疏架构，而推测解码——在解码最慢的场景下作用最为显著——也越来越多地运行在这些模型之上。然而，这些模型的草稿模型仍然维护着自己的上下文缓存，要么覆盖整个提示词，要么采用滑动窗口。除了内存占用之外，这样的缓存还会消耗预填充时间、降低前缀缓存命中率以及影响解码并发度。读取目标模型缓存的草稿模型虽然可以免去自己的缓存，但只解决了“使用哪个缓存”的问题，而没有解决“读取哪些位置”的问题，因此仍需读取全部内容或最近的窗口。这两种规则都不是针对当前查询而设计的，而在草稿模型中为每个查询计算选择，会随着上下文增长而付出更高的代价。然而，原生稀疏的目标模型本身就已经针对每个查询在其自带的缓存上完成了位置挑选。因此，草稿模型的两个关键选择——使用哪个缓存以及读取哪些位置——都已经由目标模型做出并承担了代价。我们测量了……（原文摘要在此处截断）

    arXiv:2611.34776v1 Announce Type: new  Abstract: To avoid the quadratic cost of dense attention over long contexts, models such as DeepSeek-V4 and Hy4 have turned natively sparse, and speculative decoding, which matters most where decoding is slowest, increasingly runs on them. Their drafters, however, still keep a context cache of their own, over the whole prompt or a sliding window. Beyond memory, such a cache costs prefill time, prefix-cache hits and decode concurrency. Drafters that read the target's cache avoid one of their own, but settle only which cache, not which positions, and still read everything or a recent window. Neither rule is made for the query at hand, and computing a selection per query in the drafter would cost more as the context grows. Yet a natively sparse target already picks positions for each query, over a cache it keeps anyway. Both of the drafter's choices are therefore already made, and paid for, by the target: its cache and its selection. We measure how f
    
[^14]: 一个模型，多种模式：为创造性生成演化系统提示词集成

    One Model, Many Modes: Evolving System Prompt Ensembles for Creative Generation

    [https://arxiv.org/abs/2611.34760](https://arxiv.org/abs/2611.34760)

    该论文提出免训练框架 SEPO，通过演化系统提示词集成并结合质量约束下的贪心最大-最小选择来缓解偏好对齐大语言模型的模式坍缩，从而生成兼具高质量、新颖性与多样性的创造性内容。

    

    偏好对齐的大语言模型（LLM）可能出现模式坍缩现象，即其输出分布集中于一小部分高奖励响应附近。这限制了创造性生成，因为创造性生成需要同时具备质量、新颖性和多样性。现有的测试时方法往往强调多样性，而对单个响应是否兼具高质量与新颖性关注较少。这一区别在实际应用中很重要，因为用户可能只想要一条令人信服的、原创的响应。我们将创造性生成形式化为一个多目标分布优化问题，并引入 SEPO——一个免训练的系统提示词集成优化框架。该框架将提示词演化与贪心最大-最小选择相结合，在质量约束下识别行为上各不相同的系统提示词，然后在推理时为每个响应均匀采样一个提示词。在四个质量要求逐步提高的创造性生成任务以及三个开源权重模型上（摘要在此处截断）……

    arXiv:2611.34760v1 Announce Type: cross  Abstract: Preference-aligned LLMs can exhibit mode collapse, concentrating their output distributions around a narrow set of high-reward responses. This limits creative generation, which requires quality, novelty, and diversity. Existing test-time methods often emphasize diversity, giving less attention to whether individual responses are both high-quality and novel. This distinction matters in practical use, where users may seek a single compelling, original response. We formalize creative generation as a multi-objective distributional optimization problem and introduce SEPO, a training-free framework for System Prompt Ensemble Optimization. The framework combines prompt evolution with greedy max-min selection to identify behaviorally distinct system prompts under a quality constraint, then uniformly samples one prompt per response at inference time. Across four creative generation tasks with increasing quality requirements and three open-weigh
    
[^15]: 通过透镜解读微调差异：用于后门检测的模型差异分析方法

    Reading Fine-Tuning Diffs Through Lenses: Model Diffing for Backdoor Detection

    [https://arxiv.org/abs/2611.34758](https://arxiv.org/abs/2611.34758)

    该论文提出了一个基于token读出的统一评分系统来评估各类模型差异后门检测方法，并引入J透镜与J++透镜改进权重和激活分析方法，其中最强的ΔW J++方法在12个设置中的11个成功将后门相关token排入前十。

    

    我们引入了一个基于token读出的统一评分系统，用于评估基于权重、激活和输出的模型差异分析方法在微调过程中的后门检测效果。我们还通过引入J透镜及其改进版J++透镜，改进了现有的基于权重和激活的分析方法。我们在将后门模型与其微调前的基座模型进行比较的实验中测试了该设置。结果表明，引入J++透镜在所有设置中都提升了基于权重的方法的表现，而J透镜则改进了其在单层内的排名。最强的单一方法是ΔW J++，即通过基座模型的J++透镜解码的权重差异，该方法在12个设置中的11个里将命名后门的token排进了前十。此外，我们还展示了基于输出和激活的方法（如ADL和Diff Mining）在后门模型出现某些情况时很容易失效。

    arXiv:2611.34758v1 Announce Type: cross  Abstract: We introduce a common scoring system based on token readouts to evaluate weight, activation and output-based model diffing methods for backdoor detection in fine-tuning processes. We also improve existing weight and activation based methods by incorporating the $J$-lens and its refinement, the $J{+}{+}$-lens. We test our setup in experiments where we compare a backdoored model with the base model from which it was fine-tuned. Our results show that incorporating the $J{+}{+}$-lens improves the weight-based method in all our settings, while the $J$-lens improves its ranks within single layers. The strongest single method is $\Delta W J{+}{+}$, the weight difference decoded through the $J{+}{+}$-lens of the base model, which ranks a token naming the backdoor among its first ten tokens in 11 of our 12 settings. Moreover, we show that output and activation based methods such as ADL and Diff Mining can easily break if the backdoored model is
    
[^16]: 收集以求覆盖，回放以证据：面向长程智能体的自适应高效在线策略蒸馏

    Collect for Coverage, Replay with Evidence: Adaptive and Efficient On-Policy Distillation for Long-Horizon Agents

    [https://arxiv.org/abs/2611.34756](https://arxiv.org/abs/2611.34756)

    该论文提出CARE-OPD框架，通过学生种子教师补全和响应级置信度门控扩大历史覆盖并过滤不可靠教师信号，同时以证据自适应地回放轨迹中任意深度的关键决策，从而实现面向长程智能体的自适应且高效的前缀回放式在线策略蒸馏。

    

    arXiv:2611.34756v1 公告类型：cross。摘要：对于长程智能体而言，完全在线的在线策略蒸馏（OPD）代价高昂，因为每次更新都依赖全新的环境交互轨迹。前缀回放提供了一种高效的替代方案：通过从存储的历史中采样当前学生模型的响应，减少了长尾轨迹带来的空闲时间，并使每条收集到的轨迹能够支持多次更新。然而，我们对完整前缀回放流程的分析发现，在长程场景中存在两个局限。在轨迹收集阶段，仅由教师生成的轨迹会遗漏由学生引入的历史，而更广泛的历史覆盖又可能降低教师的可靠性。在轨迹复用阶段，固定的基于位置的调度无法识别可能出现在轨迹任意深度处的关键决策。我们提出 CARE-OPD（Collect for coverAge, Replay with Evidence，为覆盖而收集、以证据回放）来解决这两个局限。学生种子教师补全与响应级置信度门控在扩展历史覆盖的同时过滤不可靠内容（摘要在此处被截断）。

    arXiv:2611.34756v1 Announce Type: cross  Abstract: Fully online on-policy distillation (OPD) is costly for long-horizon agents because each update depends on fresh environment rollouts. Prefix replay offers an efficient alternative by sampling current-student responses from stored histories, reducing idle time for long-tail rollouts and allowing each collected trajectory to support multiple updates. However, our analysis of the complete prefix replay pipeline identifies two limitations in long-horizon settings. In trajectory collection, teacher-only trajectories miss student-induced histories, while broader history coverage can reduce teacher reliability. In trajectory reuse, fixed position-based schedules cannot identify critical decisions that may occur at any trajectory depth. We introduce CARE-OPD (Collect for coverAge, Replay with Evidence) to address both limitations. Student-seeded teacher completions and response-level confidence gating expand history coverage while filtering u
    
[^17]: 更大的模型还是更大的表格？Chorus N-gram 模块能替代多少主干网络

    Bigger Model or Bigger Table? How Much Backbone a Chorus N-gram Module Replaces

    [https://arxiv.org/abs/2611.34725](https://arxiv.org/abs/2611.34725)

    该论文通过在 13.4 亿波兰语词元上训练 16M 至 349M 参数的模型，量化了 Chorus N-gram 哈希表模块对 transformer 主干网络的替代作用：当查找表与主干网络规模相当时，每个新增参数最多可替代 0.31 个主干参数，使模型能够以较小规模匹敌 1.16 至 1.34 倍大小的普通模型。

    

    我们探究了 Chorus N-gram 模块能在多大程度上替代 transformer 的主干网络。Chorus 在其输入和注意力机制中加入了基于哈希表存储的词元二元组（bigram）和三元组（trigram）。我们在 13.4 亿词元的 1918 年之前印刷的波兰语文本上训练了 1600 万至 3.49 亿参数的模型，并找出具有相同留出集损失的普通模型所需的规模。针对每种模型分别进行独立的学习率搜索后，Chorus 在 47M 规模下的收益减少了约三分之一。当查找表规模与主干网络大致相当时，Chorus 在 16M 规模下可以匹敌 1.34 倍于其规模的普通模型，在 107M 至 349M 规模下可以匹敌 1.16 至 1.20 倍于其规模的普通模型。使用这些表格时，每个新增参数最多只能替代 0.31 个主干网络参数。Chorus 的训练速度比同等规模的普通模型慢，但比达到相同损失的普通模型快。在事实问答测验上，两者的差异大约相当于同一模型在不同随机种子下的差异。这些结果基于单一语料库、一个 8000 词元的词表以及仅五分之一个训练周期。

    arXiv:2611.34725v1 Announce Type: new  Abstract: We ask how much of a transformer's backbone the Chorus N-gram module replaces. Chorus adds hashed tables of token bigrams and trigrams to its input and attention. We train models of 16M to 349M parameters on 1.34 billion tokens of Polish printed before 1918 and find the size of a plain model with the same held-out loss. A separate learning-rate search for each kind of model cuts Chorus's gain at 47M by about a third. With tables about as large as the backbone, Chorus then matches a plain model 1.34 times its size at 16M and 1.16 to 1.20 times from 107M to 349M. With these tables, each added parameter replaces at most 0.31 backbone parameters. Chorus trains slower than a plain model of its size and faster than one of the same loss. On a fact quiz the two differ about as much as two seeds of one model. The results rest on one corpus, one 8,000-token vocabulary and a fifth of an epoch.
    
[^18]: 从共享模式到个体证据：时序知识图谱中新兴实体的两阶段推理

    From Shared Patterns to Individual Evidence: Two-Stage Reasoning for Emerging Entities in Temporal Knowledge Graphs

    [https://arxiv.org/abs/2611.34708](https://arxiv.org/abs/2611.34708)

    该论文提出INSPIRE两阶段推理框架，通过融合全目录事件记忆的共享模式预测与轻量级个体证据建模，有效解决时序知识图谱中新兴实体从零样本到少样本演化场景下的链接预测难题。

    

    时序知识图谱（TKG）持续引入新兴实体，这些实体的交互历史会从零样本的初次登场逐步演变为稀疏且突发性的少样本轨迹。然而，现有研究范式较为割裂：模式迁移方法缺乏覆盖完整实体目录的事件记忆，而在线记忆模型则存在端点角色不对称问题，导致对未见答案的排序性能下降。此外，稀疏的早期交互在偏斜分布下同时表现出长程分散性与局部冗余性。为应对这些挑战，我们提出了INSPIRE（Integrating Shared Patterns and Individual Evidence for Reasoning，融合共享模式与个体证据的推理框架），一个面向已知实体目录的两阶段预测框架。在第一阶段，INSPIRE学习一个具备可用性感知能力的共享模式预测器，该预测器融合跨完整候选目录的事件更新记忆，并采用查询侧与目录侧协同掩码机制。随后，在第二阶段，该预测器被冻结，并在其基础上优化一个轻量级的个体……（原文摘要在此处截断）

    arXiv:2611.34708v1 Announce Type: new  Abstract: Temporal knowledge graphs (TKGs) continually introduce emerging entities whose interaction histories evolve from zero-shot debuts into sparse, bursty few-shot trajectories. However, current paradigms are fragmented: pattern-transfer methods lack catalog-wide event memory, whereas online memory models suffer from endpoint-role asymmetry that degrades unseen-answer ranking. Furthermore, sparse early interactions exhibit both long-range dispersion and local redundancy under skewed distribution. To address these challenges, we propose INSPIRE (Integrating Shared Patterns and Individual Evidence for Reasoning), a two-stage forecasting framework over a known catalog. In Stage I, INSPIRE learns an availability-aware shared-pattern predictor that fuses event-updated memory across the full candidate catalog while applying coordinated query- and catalog-side masking. Subsequently, Stage II freezes this predictor and optimizes a lightweight individ
    
[^19]: 当代理模型出错时：用于自适应数据采样的失配校正老虎机

    When the Proxy Is Wrong: Mismatch-Corrected Bandits for Adaptive Data Sampling

    [https://arxiv.org/abs/2611.34707](https://arxiv.org/abs/2611.34707)

    该论文提出失配校正老虎机方法，将代理模型与目标模型的排序失建为多保真度老虎机问题，利用稀少审计中的成对残差校正代理奖励并智能分配审计预算，从而在无需假设共享偏移或排序保持的情况下改进大模型的自适应数据采样。

    

    数据分配方法通常使用一个小型代理模型来为更大的目标模型选择训练数据。然而，该代理模型对数据的排序可能与目标模型不一致，从而导致次优的训练性能。为了解决这一问题，我们对代理模型与目标模型之间的失配进行建模，并将其纳入分配决策中。为此，该问题被构建为一个多保真度老虎机问题，其中每个臂对应一种数据干预。代理奖励在每一轮都会被观测，而目标奖励仅在对同一更新的稀少审计中才会揭示。这些成对的观测能够在不假设共享偏移或排序保持的前提下，识别出每个臂各自的失配。失配校正老虎机通过成对残差来校正代理模型，仅审计不确定的决策，并在整个训练过程中合理控制审计预算。从理论角度来看，我们在平稳性条件下证明了遗憾分解和下界。（原文摘要此处似乎被截断）

    arXiv:2611.34707v1 Announce Type: cross  Abstract: Data allocation methods often use a small proxy model to select training data for a larger target model. However, this proxy can rank data differently from the target, leading to suboptimal training performance. To address this problem, we model the mismatch between the proxy and target models and incorporate it into the allocation decision. To do so, the problem is cast as a multi-fidelity bandit, where each arm is a data intervention. The proxy reward is observed every round, whereas the target reward is revealed only on rare audits of the same update. These paired observations identify arm-wise mismatches without assuming a shared offset or preservation of rankings. Mismatch-corrected bandits correct the proxy with paired residuals, audit only uncertain decisions, and pace the audit budget over training. From a theoretical viewpoint, we prove a regret decomposition and lower bounds under stationarity. These bounds show that proxy-on
    
[^20]: PoVi：基于知识接地与修辞理解的智能体诗歌到视频生成

    PoVi: Agentic Poetry-to-Video Generation with Knowledge Grounding and Figurative Understanding

    [https://arxiv.org/abs/2611.34690](https://arxiv.org/abs/2611.34690)

    PoVi 提出了一个结合知识接地与修辞理解的智能体框架和基准，将诗歌语义逐步转化为旁白、解读与镜头级视觉规划，显著提升了诗歌视频生成在隐含知识和隐喻表达上的表现。

    

    诗歌到视频生成不仅仅是将文本翻译成视觉场景。诗歌语言高度凝练，往往依赖历史与文化知识，并经常通过隐喻、意象及其他非字面表达来传递意义。我们提出 PoVi，一个面向知识接地的诗歌到视频生成的基准与智能体框架，用于评估语义接地、语篇连贯性和视听呈现效果。PoVi 将诗歌语义重构为证据约束的表示，并逐步将其转化为旁白、解读、角色资产和镜头级视觉规划，随后生成与旁白对齐的视频。在三种规划骨干模型和四种视频生成器上的实验表明，PoVi 始终优于单次调用式生成，其中在隐含背景知识和隐喻关系方面提升最大。PoVi 为研究知识接地、长程多模态生成提供了测试平台。

    arXiv:2611.34690v1 Announce Type: new  Abstract: Poetry-to-video generation requires more than translating text into visual scenes. Poetic language is highly compressed, often depends on historical and cultural knowledge, and frequently conveys meaning through metaphor, imagery, and other non-literal expressions. We introduce PoVi, a benchmark and agentic framework for grounded poetry-to-video generation, evaluating semantic grounding, discourse coherence, and audiovisual realization. PoVi reconstructs poem semantics into an evidence-bounded representation and progressively transforms them into narration, interpretation, character assets, and shot-level visual plans, followed by narration-aligned video generation. Across three planning backbones and four video generators, PoVi consistently improves over single-call generation, with the largest gains on implicit background knowledge and metaphorical relations. PoVi provides a testbed for studying knowledge-grounded, long-horizon multimo
    
[^21]: 技术报告 MEG2text：超越对齐的无创脑-文本解码

    Technical Report MEG2text: Beyond Alignment in Non-Invasive Brain-to-Text

    [https://arxiv.org/abs/2611.34686](https://arxiv.org/abs/2611.34686)

    MEG2text通过修正训练评估流程、脑-文本对齐、类别平衡读出与语言模型融合，在LibriBrain100词语排序任务上将平均Top-10平衡准确率提升至64.12%，比MEG-XL提高了16.92个百分点。

    

    无创脑-文本解码旨在无需手术的情况下从大脑活动中恢复语言。其核心挑战之一是将学习到的脑-语言表征转化为准确的词语预测。我们提出了MEG2text，这是一个解码系统，结合了修正后的训练与评估流程、脑-文本对齐、类别平衡的词语读出机制以及目标词语言模型融合。在LibriBrain100的词语排序任务上，MEG2text达到了64.12%的平均Top-10平衡准确率，在四个语料库上比已发表的MEG-XL提升了16.92个百分点，且在每个语料库上均有提升。受控对比实验表明，实现层面的修正和学习式读出机制都对这一提升有所贡献。候选词层面的分析揭示，某些词语虽然获得了有信息量的分数，却始终输给竞争候选词；而相同特征条件下的对比表明，学习分类规则可以在不改变表征的情况下提升解码效果。

    arXiv:2611.34686v1 Announce Type: new  Abstract: Non-invasive brain-to-text seeks to recover language from brain activity without surgery. A central challenge is turning learned brain-language representations into accurate word predictions. We present MEG2text, a decoding system that combines corrected training and evaluation, brain-text alignment, a class-balanced word readout, and target-word language-model fusion. On LibriBrain100's word-ranking task, MEG2text reaches 64.12% mean Top-10 balanced accuracy, improving on published MEG-XL by 16.92 percentage points across four corpora, with gains on each corpus. Controlled comparisons show that both implementation corrections and a learned readout contribute to this improvement. Candidate-level analysis reveals that some words receive informative scores yet consistently lose to competing candidates, while an identical-feature comparison shows that learning the classification rule improves decoding without changing the representation. Th
    
[^22]: 神经渲染编译器：将生成式视频模型作为可编程渲染器

    Neural Rendering Compiler: Generative Video Models as Programmable Renderers

    [https://arxiv.org/abs/2611.34680](https://arxiv.org/abs/2611.34680)

    NeurCo 提出了一种神经渲染编译器，通过自动探测冻结视频模型构建能力手册，并将用户意图编译为模型特定的执行策略，从而为生成式视频模型提供可预测的编程式渲染控制。

    

    生成式视频模型正成为强大的渲染器，但缺乏传统系统（如 HTML/CSS 引擎或 Blender）所具备的可预测控制接口。它们的能力是隐式地从训练中产生的，因此难以确定它们能够可靠执行哪些任务。我们提出了神经渲染编译器，它通过自动探测冻结的视频模型来构建模型特定的能力手册，描述该模型能够可靠满足哪些语义需求。NeurCo 将用户意图表示为与模型无关的中间表示，涵盖对象、约束、事件和相机，然后将其编译为模型特定的执行策略。可靠的需求保持不变，而不可靠的需求则被选择性地重写或编译为替代执行策略。我们引入了 VComBench，这是一个包含 380 个组合式多需求任务的基准测试，涵盖文本生成视频、图像生成视频和参考生成视频等模态。

    arXiv:2611.34680v1 Announce Type: cross  Abstract: Generative video models are becoming powerful renderers, yet lack the predictable control interfaces of conventional systems such as HTML/CSS engines or Blender. Their capabilities emerge implicitly from training, making it difficult to determine what they can reliably execute. We present Neural Rendering Compiler (NeurCo), which automatically probes frozen video models to construct model-specific capability manuals describing which semantic requirements they can reliably satisfy. NeurCo represents user intent in a model-agnostic intermediate representation of objects, constraints, events, and cameras, then compiles it into model-specific execution strategies. Reliable requirements remain unchanged, while unreliable ones are selectively rewritten or compiled into alternative execution strategies. We introduce VComBench, a benchmark of 380 compositional, multi-requirement tasks spanning text-to-video, image-to-video, and reference-to-vi
    
[^23]: EPPC-CARE：医患安全门户消息中沟通行为的分阶段分层标注

    EPPC-CARE: Staged Hierarchical Annotation of Communication Behaviors in Secure Patient-Provider Messages

    [https://arxiv.org/abs/2611.34539](https://arxiv.org/abs/2611.34539)

    EPPC-CARE是一个基于开源大语言模型的自动化分层标注框架，能从医患门户消息中识别沟通行为并输出“文本片段—沟通类别—子类别”三元组，性能全面超越EPPC-OASIS和监督微调方法，且接近人类专家标注水平。

    

    目标：患者门户消息提供了关于患者与医疗提供者之间沟通动态的宝贵信息，但人工提取这些信息非常耗费人力。我们开发并评估了EPPC-CARE，一个用于识别沟通行为并同时给出每次分类所支持文本的自动化框架。材料与方法：从867条消息中，我们获得了5,516条参考标准标注。每条标注都是一个三元组，包含沟通相关文本（文本片段）、较宽泛的沟通类别（Code）以及具体的子类别。我们在六个开源权重大语言模型（LLMs）上评估了EPPC-CARE识别的三元组，并与监督微调方法及基于沟通的模型（如EPPC-OASIS）进行比较。我们还在100条消息上将EPPC-CARE的输出与人类专家标注进行了对比。结果：EPPC-CARE在所有六个模型上的表现均优于EPPC-OASIS，其三元组F1分数有所提升（摘要在此处截断）。

    arXiv:2611.34539v1 Announce Type: cross  Abstract: Objectives: Patient portal messages provide invaluable information about communication dynamics between patients and healthcare providers, but extracting it manually is labor-intensive. We developed and evaluated EPPC-CARE, an automated framework for identifying communication behaviors together with the supporting text for each classification. Materials and Methods: From 867 messages, we obtained 5,516 reference standard annotations. Each annotation is a triplet of communication-related text (span), a broader communication category (Code), and a specific subcategory (Sub-code). We evaluated EPPC-CARE's identified triplets across six open-weight large language models (LLMs) against supervised fine-tuning and communication-based models such as EPPC-OASIS. We also compared EPPC-CARE with human expert annotations on 100 messages. Results: EPPC-CARE improved performance over EPPC-OASIS across all six models. EPPC-CARE's Triplet F1 improved 
    
[^24]: AliceAI-Foundation-80B-A3B-Base（AliceAI 基础 80B-A3B 基座模型）

    AliceAI-Foundation-80B-A3B-Base

    [https://arxiv.org/abs/2611.34529](https://arxiv.org/abs/2611.34529)

    Yandex 从零训练并开源了总参数 800 亿、每 token 激活约 30 亿的混合专家基座模型，它结合 Kimi Delta Attention 与全注意力并使用 Muon 优化器训练，同时提出在 MoE 输出上引入辅助损失以在预测的高学习率下稳定训练。

    

    arXiv:2611.34529v1 发布类型：新论文。我们推出 AliceAI-Foundation-80B-A3B-Base，这是一个混合专家语言模型，总参数量为 800 亿，每个词元约有 30 亿激活参数，由 Yandex 从零开始训练，并以 Apache 2.0 许可证开源发布。该模型将 Kimi Delta Attention（Kimi 增量注意力）与全注意力机制相结合，采用注意力残差，并使用 Muon 优化器进行训练。训练过程包括在 17.5 万亿词元上进行的一般预训练、两个上下文扩展阶段，以及一个专门针对推理与智能体交互的训练阶段。我们描述了数据收集与数据增强流程、评估协议、架构消融实验以及缩放定律实验。我们指出了使用留出集训练损失作为下游性能代理指标的局限性，并在混合专家输出上引入了一种辅助损失，以便在预测的学习率下稳定训练。领域特定的数据选择与数据增强提升了事实知识、数学以及编程……（原文在此处截断）

    arXiv:2611.34529v1 Announce Type: new  Abstract: We present AliceAI-Foundation-80B-A3B-Base, a mixture-of-experts language model with 80 billion total parameters and approximately 3 billion active parameters per token, trained from scratch at Yandex and released under the Apache 2.0 license. The model combines Kimi Delta Attention with full attention, uses Attention Residuals, and is trained with Muon. Training comprises general pretraining on 17.5 trillion tokens, two context-extension stages, and a dedicated stage for reasoning and agentic interaction. We describe the data collection and augmentation pipelines, evaluation protocols, architectural ablations, and scaling-law experiments. We identify the limitations of held-out training loss as a proxy for downstream performance and introduce an auxiliary loss on mixture-of-experts outputs to stabilize training at the predicted learning rate. Domain-specific data selection and augmentation improve factual knowledge, mathematics, and pro
    
[^25]: 了解设计，选择攻击：面向大语言模型的句子键控水印

    Know the Design, Pick the Attack: Sentence-Keyed Watermarks for LLMs

    [https://arxiv.org/abs/2611.34505](https://arxiv.org/abs/2611.34505)

    论文提出了一种自适应攻击者，通过阅读公开的水印设计在改写抹除和伪造复制两种攻击间择优选择，能击败所有现有水印，并据此提出KeyedLex水印——从上一句的语法中读取密钥，从而同时抵御两类攻击。

    

    水印让语言模型的提供者能够识别其模型生成的文本，随着生成文本越来越难以与人类写作区分，这一点变得愈发重要：来源追溯支撑着学术诚信、打击虚假信息以及新的法规监管。提供者已经开始为其模型的输出文本添加水印，并公开解释其水印的工作原理。然而，解释水印的工作方式同时也告诉了攻击者如何破解它，破解方式有两种：改写文本以抹除水印标记，或者学习该水印并将其复制到模型从未生成过的文本上。以往这两种攻击方式都是被单独研究的。我们研究了一种读取已公开的水印设计并在这两种攻击之间进行选择的攻击者：它会抹除难以学习（伪造）的水印，并伪造能够经受住改写攻击的水印。面对这种自适应攻击者，我们所比较的所有现有水印均告失败。为此，我们提出了KeyedLex，它从上一句的语法中读取其密钥，而改写对语法的改变很小。

    arXiv:2611.34505v1 Announce Type: new  Abstract: A watermark lets the provider of a language model recognise the text its model wrote, which matters more as generated text becomes hard to tell from human writing: provenance underpins academic integrity, the fight against misinformation and new regulation. Providers have started to watermark the text of their models, and to explain how their watermarks work. Explaining a watermark, however, also tells an attacker how to break it, and there are two ways to do so: reword the text to erase the mark, or learn the mark and copy it onto text the model never wrote. Each attack has been studied on its own. We study an attacker that reads the published design to choose between them: it erases the watermarks that are hard to learn and forges the ones that survive rewording. Against it, every existing watermark we compare fails. We propose KeyedLex, which reads its secret key from the grammar of the previous sentence, which rewording changes littl
    
[^26]: FlyingStart：面向递归深度语言模型不动点解码的学习型跨Token热启动方法

    FlyingStart: Learned Cross-Token Warm Starts for Fixed-Point Decoding in Recurrent-Depth Language Models

    [https://arxiv.org/abs/2611.34492](https://arxiv.org/abs/2611.34492)

    提出FlyingStart方法，利用上一个Token停止时的状态经闭式解拟合的线性变换来热启动下一个Token的迭代，在精度不变的前提下将递归深度语言模型的解码循环层数减少14%–19%。

    

    递归深度语言模型通过反复迭代一个共享的层块来生成每个Token，直到其隐藏状态收敛。针对这类模型的推理方法只决定何时停止迭代，而每个Token仍然从随机噪声开始迭代，并为此付出完整收敛过程的代价。我们证明，迭代的起始点是第二个独立的优化杠杆。在学习到的坐标系变换之后，一个Token的不动点可以从上一个Token停止迭代时的状态预测出来。基于此，我们提出FlyingStart——一种线性热启动方法，它在冻结模型上以闭式解拟合，成本不到一次循环的5%。迭代过程及其停止规则保持不变，因此退出准则从任意起点都能保证到不动点的相同距离：更好的起点能在精度不变的情况下减少循环次数。在Huginn-0125模型上，FlyingStart以减少14%–19%的解码层数达到了最强免训练基线的精度，同时提升了批量推理吞吐量……

    arXiv:2611.34492v1 Announce Type: new  Abstract: Recurrent-depth language models generate each token by iterating a shared block of layers until its hidden state converges. Inference methods for these models decide when to stop the iteration, yet every token still starts from random noise and pays the full cost of converging. We show that where the iteration starts is a second, independent lever. A token's fixed point is predictable from the state at which the previous token stopped, after a learned change of coordinates, and we introduce FlyingStart, a linear warm start that is fitted in closed form on the frozen model and costs under 5% of one loop. The iteration and its stopping rule are unchanged, so the exit criterion certifies the same distance to the fixed point from any start: a better start removes loops at unchanged accuracy. On Huginn-0125, FlyingStart matches the accuracy of the strongest training-free baseline with 14--19% fewer decoding layers, raises batched throughput b
    
[^27]: 不对称性从何处进入？在融合多模态审核流水线中开展分阶段公平性审计

    Where Does the Asymmetry Enter? Staging a Fairness Audit Across a Fused Multimodal Moderation Pipeline

    [https://arxiv.org/abs/2611.34491](https://arxiv.org/abs/2611.34491)

    本文提出对已公开的TikTok多模态内容审核分类器进行跨阶段事后公平性审计，以统一的显著性标准在“输入内容、分支编码、最终决策”三个环节中定位按语言划分的子群体不对称性究竟出现在流水线的哪个阶段。

    

    如今，几乎每一个上传的短视频都会经过一个分类器，该分类器将视频、语音和屏幕文字融合为关于该视频能否被观看的单一决策。它同时审核平台上的所有语言社区，而跨语言社区的公平性（如果有所评判的话）仅由一个准确率分数来衡量。然而，这个分数与“错误几乎完全落在某一个语言群体上”的情形是相容的。而寻找问题的审计通常读取的只是模型的最终决策——那是漫长流水线的最后一个阶段，也是测试样本最少的环节。我们提出的问题是：在一个融合流水线中，子群体不对称性究竟可以在哪个环节被检测到——是在分类器可能读取的输入中、在其各分支所编码的信息中，还是在它最终的决策中？我们通过对一个已公开的TikTok分类器进行事后审计来回答这一问题：仅使用其记录的决策、标签和融合前特征，以源语言作为子群体划分轴，将全部三个阶段置于同一个显著性标准下检验。摘要显示该不对称性在某阶段具有显著性（原文摘要在此处被截断，后续结论未完整给出）。

    arXiv:2611.34491v1 Announce Type: new  Abstract: Almost every short-video upload now passes through a classifier that fuses video, speech, and on-screen text into one decision about whether it may be seen. It moderates every language community on the platform at once, and fairness across them is judged, if at all, from one accuracy score. That score is compatible with error falling almost entirely on one language group. An audit that goes looking reads the model's decisions, the last stage of a long pipeline and the one whose test has the fewest items. We ask where in a fused pipeline a subgroup asymmetry can be detected at all: in what the classifier may read, in what its branches encode, or in what it decides. We answer this with a post-hoc audit of a published TikTok classifier, holding all three stages to one significance standard using only its logged decisions, labels, and pre-fusion features, with source language as the subgroup axis. The asymmetry is significant at the stage de
    
[^28]: vSFT：在测试时通过语言反馈微调智能体

    vSFT: Learning to Verbally Tune Agents at Test Time

    [https://arxiv.org/abs/2611.34440](https://arxiv.org/abs/2611.34440)

    提出vSFT框架，通过超网络将监督者基于智能体失败轨迹生成的自然语言反馈转化为低秩权重更新，实现测试时的参数内演化，在ALFWorld等任务上性能超越GRPO和Reflexion等主流强化学习方法。

    

    大语言模型智能体越来越多地通过智能体强化学习进行改进，这种方法可以采用在策略方式，但存在 rollout 成本高昂和奖励稀疏的问题，限制了其在测试时的应用。本文提出了一种语言监督微调框架，实现了自然语言密集反馈与测试时权重更新的结合。具体而言，我们的vSFT通过一个超网络在智能体与监督者之间构建闭环：监督者从智能体的失败轨迹中生成语言监督信号，超网络则将这些监督信号转化为低秩权重更新以调整智能体。我们验证了这种测试时的参数内演化过程优于上下文学习对应方法以及主流强化学习方法。特别地，基于Qwen3-4B的vSFT在ALFWorld上的成功率分别比GRPO和Reflexion高出8.7%和13.9%，并比另一方法高出4.8%（原文摘要此处截断）。

    arXiv:2611.34440v1 Announce Type: new  Abstract: Large language model agents are increasingly improved via agentic reinforcement learning, which can be on-policy but suffers from expensive rollouts and sparse rewards, limiting its application at test time. In this paper, we present a verbally Supervised Fine-Tuning (vSFT) framework that enables the conjunction of dense feedback in natural language and weight updates at test time. Specifically, our vSFT forms a closed loop between an agent and a supervisor via a hypernetwork, in which the supervisor generates verbal supervision from the agent's failed trajectories, and the hypernetwork converts that supervision into low-rank weight updates to tune the agent. We validate that this test-time in-parameter evolving process outperforms both its in-context counterparts and prevalent reinforcement learning methods. In particular, our vSFT using Qwen3-4B outperforms GRPO and Reflexion by 8.7% and 13.9% in success rate on ALFWorld, and by 4.8% a
    
[^29]: 缓解ESC心脏病学指南医学问答中的幻觉：评估与检索优化

    Mitigating Hallucinations in Medical Question Answering on ESC Cardiology Guidelines: Evaluation and Retrieval Optimization

    [https://arxiv.org/abs/2611.34427](https://arxiv.org/abs/2611.34427)

    该研究首次逐条对照人类判断，系统评估了基于ESC心脏病学指南的“验证-修订”流程在医学问答中的实际失败情况，并优化了检索配置以缓解大语言模型的幻觉问题。

    

    背景：大语言模型在医学问答（QA）中会产生幻觉。通过检索指南证据对原子化声明进行核验的“验证-修订”流程是一种标准的缓解手段，但现有评估几乎普遍将验证器的输出视为真值（ground truth），这一假设很少被检验。目标：在基于指南的临床场景中，逐条声明对照人类判断，测量此类流程的失败情况。方法：我们构建了一个包含160个问题的数据集（100个患者病例场景来自89个去标识化病例，60个基于事实的问题），涵盖四份近期ESC心脏病学指南，全部经一名执业内科医生审核；同时建立了一个包含627条人工核验指南记录的检索语料库，采用两种分块粒度。在四种检索配置和三次独立运行（共1920次臂运行）下运行了五阶段流程；所有606条首轮被驳回的声明均经人工审计，一名独立评审员重新判断了906项决策……（原文摘要此处截断）

    arXiv:2611.34427v1 Announce Type: new  Abstract: Background: Large language models produce hallucinations in medical question answering (QA). Verify-and-revise pipelines that check atomic claims against retrieved guideline evidence are a standard mitigation, but evaluations almost universally treat the verifier's output as ground truth, an assumption rarely tested. Objective: To measure, claim by claim against human judgement, how such a pipeline fails in a guideline-grounded clinical setting. Methods: We built a dataset of 160 questions (100 patient vignettes from 89 de-identified cases, 60 fact-based) over four recent ESC cardiology guidelines, all approved by a practising internist, and a retrieval corpus of 627 hand-verified guideline records at two chunk granularities. A five-stage pipeline was run under four retrieval configurations and three independent runs (1920 arm-runs); all 606 first-pass refuted claims were manually audited, an independent judge re-judged 906 decisions, an
    
[^30]: DocSEAM：面向多页文档理解的双向跨页编码

    DocSEAM: Bidirectional Cross-Page Encoding for Multi-Page Document Understanding

    [https://arxiv.org/abs/2611.34420](https://arxiv.org/abs/2611.34420)

    DocSEAM是一种插入在视觉编码器与语言模型之间的轻量级双向跨页编码模块，通过以中心页为锚的掩码自注意力融合相邻页面信息，在不改变视觉标记数量的前提下解决了多页文档理解中页面独立编码导致的特征不连续问题。

    

    多页文档视觉-语言模型通常独立编码每一页，导致页边界处出现特征层面的不连续性：视觉标记的生成无法访问相邻页面的内容，而语言模型只能基于这些孤立生成的页面表示进行推理。我们提出DocSEAM（通过锚定掩码实现文档选择性增强），这是一个插入在预训练视觉编码器与语言模型之间的轻量级模块，它将局部跨页上下文的归纳偏置注入视觉表示。对于每一页，DocSEAM与其前后两页构成一个局部三元组，应用以中心页为锚的掩码自注意力机制及方向性显著性门控来融合相邻页面的信息，并仅返回增强后的中心页标记，同时保持视觉标记数量不变。经过监督式跨页预热训练以及在DocSEAM冻结状态下的多页微调后，我们的90亿参数模型达到了最先进的性能水平。

    arXiv:2611.34420v1 Announce Type: cross  Abstract: Multi-page document vision-language models encode each page independently, creating a feature-level discontinuity at page boundaries: visual tokens are formed without access to content on adjacent pages, and the language model must reason over these page representations formed in isolation. We introduce DocSEAM (Document Selective Enrichment via Anchored Masking), a lightweight module inserted between a pretrained vision encoder and the language model that injects the inductive bias of local cross-page context into visual representations. For each page, DocSEAM forms a local triplet with its two neighbors, applies center-anchored masked self-attention with directional salience gates to fuse information from neighbor pages, and returns only the enriched center-page tokens, leaving the visual token count unchanged. After a supervised cross-page warm-up and multi-page fine-tuning with DocSEAM frozen, our 9B-parameter model achieves state-
    
[^31]: Any-Rate：基于编码位平面的半比特多精度大语言模型量化

    Any-Rate: Half-Bit Multi-Precision LLM Quantization with Coded Bit-Planes

    [https://arxiv.org/abs/2611.34402](https://arxiv.org/abs/2611.34402)

    Any-Rate通过将位平面约束为8权重块的扩展汉明码字并结合逐精度联合编码，实现了从单一检查点以半比特为步长提供2–8比特的多精度LLM量化，在2比特和3比特下达到最先进的准确率。

    

    嵌套多精度量化通过仅读取每个权重的最高有效位，从单一检查点派生出多种精度，这使其在各种内存和能耗预算下的端侧部署中极具吸引力。然而，现有方法仅支持整数比特，导致相邻精度之间存在较大的准确率差距，并且在2比特时出现严重的性能退化。为解决这一问题，我们提出了Any-Rate，它能够从单一检查点提供2至8比特的精度服务，并在2至4比特范围内以半比特为步长进行调节。Any-Rate将部分位平面限制为8权重块中的扩展汉明码字（每权重0.5比特），在各精度之间共享每行一个查找表（LUT），并对所有精度进行联合编码，同时为每个精度设置独立的优化目标，从而降低了共享位平面带来的损失。在四个大语言模型上，Any-Rate在2比特和3比特下实现了多精度方法中最先进的准确率，在2比特下保持了FP16准确率的69%，在4比特下与最佳方法相差不到一个百分点。它解码……

    arXiv:2611.34402v1 Announce Type: cross  Abstract: Nested multi-precision quantization derives multiple precisions from a single checkpoint by reading only the leading bits of each weight, which makes it attractive for on-device deployment under varying memory and energy budgets. However, existing methods offer only integer bits, leaving large accuracy gaps between adjacent precisions and severe degradation at 2 bits. To address this, we propose Any-Rate, which serves 2-8 bits from one checkpoint, in half-bit steps over 2-4 bits. Any-Rate restricts some bit-planes to extended Hamming codewords in 8-weight blocks (0.5 bits per weight), shares one LUT per row across precisions, and encodes all precisions jointly with a separate target for each, reducing the loss from sharing bit-planes. Across four LLMs, Any-Rate achieves state-of-the-art accuracy among multi-precision methods at 2 and 3 bits, keeping 69% of FP16 accuracy at 2 bits, and is within one point of the best at 4 bits. It decod
    
[^32]: 当技能无法组合时：语言模型中的兼容性与保留性

    When Skills Fail to Compose: Compatibility and Retention in Language Models

    [https://arxiv.org/abs/2611.34398](https://arxiv.org/abs/2611.34398)

    该论文发现，在冻结语言模型上独立学习的软提示技能难以在单次调用中有效组合（准确率最高仅7.6%，远低于两次调用的88.2%–100%），且相同操作的技能块并不必然可互换，揭示了构建可复用技能库时兼容性与保留性方面的关键瓶颈。

    

    仅靠技能习得并不能建立一个可复用的技能库：任务熟练度并不等于一个可组合的接口。受能够习得并精炼技能的智能体启发，我们在冻结的语言模型上研究了独立学习的软提示技能，其中技能身份与执行顺序是预先给定的。在四个数字任务基础和三个记录编辑模型上，单次调用的平均组合准确率最高仅为7.6%，而两次调用（增加了计算量和显式的中间输出）的准确率可达88.2%至100%。中性插入训练提升了鲁棒性，但仍然存在大量干扰，在四个数字基础中的两个上，仅凭块长度和每token范数无法解释这种干扰。相同操作的技能块不一定可以互换：在与冻结的已学习伙伴块一同训练后，对两个Qwen基础的事后测试表明，用独立学习的相同操作块替换这些伙伴块会降低中位数兼容性。

    arXiv:2611.34398v1 Announce Type: cross  Abstract: Skill acquisition alone does not establish a reusable skill library: task proficiency does not specify a compositional interface. Motivated by agents that acquire and refine skills, we study independently learned soft-prompt skills on frozen language models, with skill identities and execution order supplied. Across four digit-task bases and three record-editing models, mean composition accuracy in one call is at most 7.6%, compared with 88.2% to 100% over two calls that add computation and an explicit intermediate output. Neutral-insertion training improves robustness but leaves substantial interference, which block length and per-token norm alone cannot explain on two of four digit bases. Blocks of the same operation need not be interchangeable: after training next to frozen learned partners, a post hoc test on two Qwen bases shows that replacing those partners with independently learned blocks of the same operations lowers median co
    
[^33]: 强大的起点，简单的配方：基于多模态专家模型的统一多教师在线策略蒸馏

    Strong Starts, Simple Recipes: Unified Multi-Teacher On-Policy Distillation from Multimodal Specialists

    [https://arxiv.org/abs/2611.34397](https://arxiv.org/abs/2611.34397)

    该论文提出Uni-MOPD方法，将学生模型的起点作为设计变量——先用无训练的专家模型均匀合并获得强起点，再用简短的多教师在线策略蒸馏配方进行精炼，并发现起点选择比后续训练策略对最终性能的影响更大。

    

    多教师在线策略蒸馏（MOPD）将分别训练的领域专家模型整合到一个统一模型中。现有方法专注于改进蒸馏过程，而学生模型则从基础模型或多领域SFT检查点初始化，这使得能力整合在很大程度上依赖于蒸馏本身。相反，我们将起点视为一个设计变量，提出了统一MOPD（Uni-MOPD），它将强大的起点与简单的配方相结合：使用来自同一基础模型的专家模型的无训练均匀合并来初始化学生模型，随后使用均匀领域采样和全词表反向KL散度的短程MOPD对其进行精炼。在五个多模态领域和19个基准测试上，我们的研究确立了三个发现。（1）起点比后续训练选择更重要：在任何训练之前，均匀合并已经达到了基础模型经过200次MOPD更新后才能达到的分数……

    arXiv:2611.34397v1 Announce Type: cross  Abstract: Multi-teacher on-policy distillation (MOPD) integrates separately trained domain specialists into a single unified model. Existing methods focus on refining the distillation procedure while initializing the student from a base model or a multi-domain SFT checkpoint, which leaves capability integration largely to distillation itself. Instead, we treat the starting point as a design variable and propose Unified MOPD (Uni-MOPD), which pairs a strong start with a simple recipe: a training-free uniform merge of specialists from the same base model initializes the student, and short MOPD with uniform domain sampling and full-vocabulary reverse KL refines it. Across five multimodal domains and 19 benchmarks, our study establishes three findings. (1) The starting point matters more than subsequent training choices: before any training, the uniform merge already matches the score that the base model reaches after 200 MOPD updates, and after sho
    
[^34]: 影响之下——受什么影响？影响函数的受控评估

    Under the Influence -- of What? A Controlled Evaluation of Influence Functions

    [https://arxiv.org/abs/2611.34373](https://arxiv.org/abs/2611.34373)

    这是首个通过受控数据划分重新训练的系统评估，验证了影响函数在GPT-2类模型中识别的“有影响力”训练样本确实与输出相关，但不总是直接承载目标属性，且相关性因任务设置而异。

    

    影响函数是一种有前景的方法，用于识别哪些训练样本对模型的输出负有责任。然而，迄今为止在大语言模型预训练层面的实证评估都只是假设它们按预期工作。这项工作是首个系统性评估“有影响力”样本是否真的具有与语言模型输出相关属性的研究，其方法是基于受控数据划分进行重新训练。我们在GPT-2类型的模型（6800万至56亿训练词元）中，针对两种设置（语言现象和事实知识），研究了三种著名的Hessian近似方法（GradSim、TrackStar、K-FAC）。我们发现，平均而言，“有影响力”的样本确实是相关的，但它们并不总是直接承载目标属性，且相关性的比例和类型因设置而异。

    arXiv:2611.34373v1 Announce Type: new  Abstract: Influence functions are a promising way to identify which training examples are responsible for a model's output. However, so far empirical evaluations on LLM pretraining level only assumed that they work as intended. This is the first work to systematically evaluate whether the 'influential' examples actually have properties relevant to a language model output, based on re-training with controlled data splits. We investigate three prominent Hessian approximation methods (GradSim, TrackStar, K-FAC) in two settings (linguistic phenomena and factual knowledge) in GPT-2 type models (68M-5.6B training tokens). We find that on average, the `influential' examples are indeed relevant, but they do not always bear the target properties directly, and the ratio and kind of relevance varies by setting.
    
[^35]: 单一数据库奖励可被破解：用于可信赖文本到SQL强化学习的反事实执行

    Single-Database Rewards Are Hackable: Counterfactual Execution for Trustworthy Text-to-SQL RL

    [https://arxiv.org/abs/2611.34363](https://arxiv.org/abs/2611.34363)

    该论文提出一种反事实数据库奖励机制，通过在隐藏的代码生成数据库副本上执行查询来验证结果，将文本到SQL强化学习中的奖励破解率从54.1%大幅降至1.9%，且不误拒任何正确查询。

    

    面向文本到SQL智能体的强化学习（RL）通常在查询结果与单一数据库上的参考结果匹配时给予奖励。然而，当数据缺乏能够暴露查询缺陷的行时，错误的查询可能凭运气匹配成功。我们研究了一种反事实数据库（CF）奖励，它还在若干隐藏的、由代码生成的数据库副本上运行最终查询，仅在所有结果都匹配时（严格形式）或按匹配比例（平均形式）给予奖励。在1,302个合成审计问题上，严格形式将错误查询被全额支付的比例从29.5%降至16.8%，将奖励破解的比例从54.1%降至1.9%，同时拒绝了2,847个正确查询中的0个；来自“模型家族盲”生成器生成的副本表现大致相当。在一个3.35B模型上的受控GRPO对比实验（每组三个随机种子）中，使用平均形式训练并未提升域内准确率（8.4% vs 11.7%），因为两种奖励仅在5-7%的rollout上存在分歧；而在Spider上提升了1.6个点。（原文此处截断）

    arXiv:2611.34363v1 Announce Type: cross  Abstract: Reinforcement learning (RL) for text-to-SQL agents pays a query when its result matches the reference on one database. However, a wrong query can match by luck when the data lack the rows that expose its bug. We study a counterfactual-database (CF) reward that also runs the final query on a few hidden, code-generated copies of the database and pays only if all results match (strict form) or by the share matched (mean form). On 1,302 synthetic audit questions, the strict form lowers the share of buggy queries paid in full from 29.5% to 16.8% and of reward hacks from 54.1% to 1.9%, rejecting 0 of 2,847 correct queries; copies from a family-blind generator do about as well. In a controlled GRPO comparison on a 3.35B model with three seeds per arm, training on the mean form did not improve in-domain accuracy (8.4% vs 11.7%), because the two rewards disagreed on only 5-7% of rollouts; on Spider it gained 1.6 points. One seed reverses both s
    
[^36]: 从不可靠的客服坐席参考中学习：面向客户支持智能体的声明级奖励

    Learning from Unreliable Operator References: Claim-Level Rewards for Customer Support Agents

    [https://arxiv.org/abs/2611.34341](https://arxiv.org/abs/2611.34341)

    该论文提出将客服坐席回复视为待验证的声明而非模仿目标，通过声明级GRPO奖励机制过滤不可靠的训练信号，显著降低了幻觉率，并将无需人工介入的对话解决率从63.3%提升至80.8%。

    

    即使是正确的客服坐席回复，对于客户支持智能体而言也可能是一个不可靠的训练目标。这些回复中约有四分之一的事实性声明在智能体的知识库和工具记录中缺乏完整支持。我们将客服坐席的回复视为需要验证的声明，而非需要模仿的答案。未经验证的声明用于指导知识库和工具的修订；已验证的声明则作为声明级GRPO奖励中的覆盖目标，该奖励根据智能体为其答案所检索到的证据来评估事实精确度。在一个离线基准测试中，GRPO智能体的文本答案中有8.6%被标记为幻觉，而监督微调基线的这一比例为13.6%。在与现有程序化机器人的随机A/B测试中，部署的系统将无需人工坐席介入即可解决的对话比例从63.3%提升至80.8%。

    arXiv:2611.34341v1 Announce Type: new  Abstract: Even a correct operator reply can be an unreliable training target for a customer-support agent. About a quarter of the factual claims in these replies lacked full support in the agent's knowledge base and tool records. We treat operator replies as claims to verify rather than answers to imitate. Unverified claims guide revisions to the knowledge base and tools. Verified claims serve as coverage targets in a claim-level GRPO reward that scores factual precision against the evidence the agent retrieved for its answer. On an offline benchmark, 8.6% of the GRPO agent's text answers were flagged for hallucinations, compared with 13.6% for the supervised fine-tuning baseline. In a randomized A/B test against the existing procedural bot, the deployed system increased the share of conversations closed without a human operator from 63.3% to 80.8%.
    
[^37]: Puzzling-Bench：基于23,777道人工编写、社区验证答案谜题的语言模型评估基准

    Puzzling-Bench: Evaluating Language Models on 23,777 Human-Written Puzzles with Community-Verified Answers

    [https://arxiv.org/abs/2611.34331](https://arxiv.org/abs/2611.34331)

    Puzzling-Bench是一个基于23,777道来自Puzzling Stack Exchange、带有社区验证答案的人工编写谜题（涵盖谜语、密码、逻辑推理等类别）的语言模型评估基准，并利用训练截止日期后发布的谜题切片来检测数据污染。

    

    我们推出了Puzzling-Bench，这是一个包含23,777道谜题的基准测试集，谜题选自Puzzling Stack Exchange存档（2014-2026年），均带有社区验证的答案：涵盖谜语、密码、逻辑网格、横向思维问题、文字游戏和趣味数学，其中三分之一配有图片。每道谜题都附有作者认可的答案、从中提取的简短参考答案（评估集的参考答案由LLM提取，并经50道谜题的人工审核验证）、作者本人的提示、社区标签和创建日期，因此该基准支持提示消融实验、类别分析以及滚动更新的训练截止日期后谜题切片（254道谜题发布于我们所评估的九个模型中六个模型的训练截止日期之后，其中包括表现最好的模型），可作为数据污染检查。我们使用两级评分器（先进行归一化精确匹配，再由LLM裁判评分）评估了九个前沿模型和开源模型，并通过来自两个模型系列的三个LLM裁判对裁判进行了验证：它们在95-97%的判定上达成一致……

    arXiv:2611.34331v1 Announce Type: cross  Abstract: We introduce Puzzling-Bench, a benchmark of 23,777 puzzles with community-verified answers drawn from the Puzzling Stack Exchange archive (2014-2026): riddles, ciphers, logic grids, lateral-thinking problems, wordplay and recreational mathematics, a third of them with images. Every puzzle carries the answer accepted by its author, a short reference answer taken from it (LLM-extracted for the evaluation sets and checked on a 50-puzzle human review), the author's own hints, community tags and a creation date, so that the benchmark supports hint ablations, category analysis and a rolling post-cutoff slice (254 puzzles posted after the training cutoff of six of the nine models we evaluate, including the best one) as a contamination check. We evaluate nine frontier and open models with a two-tier scorer (normalized exact match, then an LLM judge) and validate the judge with three judges from two model families: they agree on 95-97% of verdi
    
[^38]: IRTIQA：以英语进化，以乌尔都语评估——智能体自我进化能否从英语迁移到乌尔都语及语码转换输入？

    IRTIQA: Evolving in English, Evaluated in Urdu -- Does Agent Self-Evolution Transfer from English to Urdu and Code-Switched Input?

    [https://arxiv.org/abs/2611.34303](https://arxiv.org/abs/2611.34303)

    该论文提出IRTIQA基准（包含对齐的英语、乌尔都语及语码转换任务变体），发现智能体自我进化虽能显著提升任务能力，但收益主要局限于英语，很少能弥合与乌尔都语及语码转换输入之间的语言性能差距。

    

    随着大语言模型智能体日益从自身经验中学习，它们学习和评估所用的任务几乎总是英语的。我们探究自我进化能否从英语迁移到乌尔都语，并发布了IRTIQA——一个由DROP、SNIPS、AppWorld和SkillsBench的相互对齐的英语版、乌尔都语翻译版和乌尔都语语码转换版变体组成的基准，该基准采用矩阵语言语码转换流水线构建并经过人工验证。借助IRTIQA，我们评估了三个自我进化框架——Tool-R0（自我博弈强化学习）、AgentEvolver（经验引导强化学习）和Autogenesis（协议级进化），并与各自未进化的基座模型进行对比。自我进化通常能提升能力：AgentEvolver在所有条件下将AppWorld目标完成率从近乎零提升至20%-36%，Autogenesis将SkillsBench通过率从16.48%提升至22.22%。但它很少能弥合语言差距：进化后的系统在英语上依然更强，DROP F1最高领先0.26，且该差距在多（语码转换任务）上最大（原文摘要在此处被截断）。

    arXiv:2611.34303v1 Announce Type: new  Abstract: As LLM agents increasingly learn from their own experience, the tasks they learn from and are evaluated on are almost always in English. We ask whether self-evolution transfers from English to Urdu, and release IRTIQA, a benchmark of aligned English, Urdu-translated, and Urdu code-switched variants of DROP, SNIPS, AppWorld, and SkillsBench, built with a matrix-language code-switching pipeline and human verification. Using IRTIQA, we evaluate three self-evolving frameworks, Tool-R0 (self-play RL), AgentEvolver (experience-guided RL), and Autogenesis (protocol-level evolution), against their unevolved backbones. Self-evolution often raises capability: AgentEvolver lifts AppWorld goal completion from near zero to 20-36% across all conditions, and Autogenesis improves SkillsBench pass rate from 16.48% to 22.22%. But it rarely closes the language gap. Evolved systems remain stronger in English, by up to 0.26 DROP F1, the gap is largest on mul
    
[^39]: MedSkill-E²：医疗智能体技能的评估与演化

    MedSkill-E$^{2}$: Evaluating and Evolving Skills for Medical Agents

    [https://arxiv.org/abs/2611.34250](https://arxiv.org/abs/2611.34250)

    该论文提出了MedSkill-E²框架和EBU-Bench基准，首次发现并量化了医疗智能体中“技能已启用但未被充分利用”（EBU）现象，即智能体虽然拥有完成任务所需的技能，却在自主执行时无法有效应用，从而可能产生看似合理但实际错误的结果。

    

    大语言模型（LLM）智能体越来越多地使用技能来获取领域知识并完成现实世界的任务。我们观察到，即使某项技能已经明确了任务所需的规则和流程，智能体在自主执行过程中仍可能无法有效应用它们，我们将这一现象称为“技能已启用但未被充分利用”（Enable But Underuse, EBU）。这一问题在医疗任务中尤为严重，因为忽视任何一个关键约束都可能导致看似合理但实际错误的结果。为了评估和解决EBU问题，我们提出了MedSkill-E²，一个用于评估和演化医疗智能体技能的框架。我们构建了EBU-Bench，一个具有挑战性的基准测试，涵盖12个医疗相关任务组及其关联技能，用于评估智能体自主使用可用技能的能力。我们将EBU差距量化为自主使用技能与在明确引导下阅读并应用同一技能之间技能使用准确率的差异。

    arXiv:2611.34250v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly use skills to access domain knowledge and complete real-world tasks. We observe that even when a skill specifies the rules and procedures needed for a task, agents may fail to apply them effectively during autonomous execution, a phenomenon we call skill Enable But Underuse (EBU). This issue is particularly consequential in medical tasks, where overlooking a single critical constraint can yield a plausible but incorrect result. To evaluate and address EBU, we introduce MedSkill-E$^{2}$, a framework for evaluating and evolving skills for medical agents. We construct EBU-Bench, a challenging benchmark covering 12 medically relevant task groups and associated skills, to evaluate agents' ability to use available skills autonomously. We quantify the EBU gap as the difference in skill-use accuracy between autonomous use and explicitly guided reading and application of the same skill. On EBU-Bench,
    
[^40]: SOLO：通过内容锚定的单一所有者上下文，在重复采样下实现样本保真且可检测的大语言模型水印

    SOLO: Towards Sample-Faithful and Detectable LLM Watermarking under Repeated Sampling via Content-Anchored Single-Owner Contexts

    [https://arxiv.org/abs/2611.34190](https://arxiv.org/abs/2611.34190)

    针对重复采样场景下无失真水印因跨响应上下文碰撞导致的耦合问题，提出SOLO方法，通过将带密钥上下文锚定在min-hash选择的内容上，实现了样本保真且可检测的大语言模型水印。

    

    供应商为大语言模型的输出添加水印，而应用程序会对每个提示采样多个响应，用于自一致性、best-of-n选择和不确定性估计。我们表明，已部署的无失真水印配置（仅在单个响应内对带密钥上下文进行去重）会将同一请求的各个响应耦合起来：答案模板使所有响应在给出答案之前都用相同的上下文查询水印，因此跨响应的碰撞（通常被认为很罕见）成为常态。密钥池以多重检验为代价减少或避免这种耦合，而通过请求间共享上下文历史的精确补救方法则会使许多位置未被加水印。我们提出SOLO，它保留请求范围内的上下文历史，但将其带密钥的上下文锚定在通过min-hash选择的内容上，使得兄弟响应很少竞争同一上下文，且随机编辑的影响保持在局部。其响应的联合分布如同独立的未加水印响应……

    arXiv:2611.34190v1 Announce Type: new  Abstract: Providers watermark the outputs of large language models, while applications sample many responses per prompt for self-consistency, best-of-n selection, and uncertainty estimation. We show that deployed configurations of distortion-free watermarks, which deduplicate keyed contexts only within a response, couple the responses of a request: answer templates make all responses query the watermark with the same context before the answer, so collisions across responses, usually assumed rare, become the norm. Key pools reduce or avoid the coupling at the cost of multiple testing, and the exact remedy of a context history shared across the request leaves many positions unwatermarked. We propose SOLO, which keeps a request-scoped context history but anchors its keyed contexts on content selected by min-hash, so that siblings rarely compete for a context and random edits stay local. Its responses are jointly distributed as independent unwatermark
    
[^41]: MINE：一种基于互信息与嵌入方差的无训练间接提示注入防御方法

    MINE: A Training-Free Defense against Indirect Prompt Injection via Mutual Information and Embedding Variance

    [https://arxiv.org/abs/2611.34188](https://arxiv.org/abs/2611.34188)

    MINE提出了一种基于互信息与嵌入方差的无训练、与攻击类型无关的新防御范式，通过最大化相关性而非检测攻击来抵御RAG中的间接提示注入，并在四个数据集上实现了无需训练防御方法中的最佳准确率-安全性权衡。

    

    现有的针对检索增强生成（RAG）中间接提示注入的防御方法本质上是被动式的：提示工程方法经常被注入的指令覆盖，而监督式检测器必须针对每种新攻击方式重新训练。我们提出了一种不同的范式：通过最大化相关性来防御，而不是检测攻击。我们的方法MINE无需训练且不依赖特定攻击类型，同时还可作为基于查询条件重排序器。为了在现实条件下评估防御方法，我们引入了LC-Inj-SQuAD，这是一个平均每个上下文包含4,032个词的长上下文基准测试。我们在四个数据集上进行评估，涵盖短上下文问答、长上下文问答（LC-Inj-SQuAD）以及智能体工具调用注入（InjecAgent）。在所有四个数据集上，MINE在无需训练的防御方法中实现了最佳准确率-安全性权衡。它也是唯一在面对完全自适应攻击者时仍保持有效的防御方法。

    arXiv:2611.34188v1 Announce Type: cross  Abstract: Existing defenses against indirect prompt injection in retrieval-augmented generation (RAG) are inherently reactive: prompt-engineering methods are often overridden by the injected instructions, while supervised detectors must be retrained for each new attack style. We propose a different paradigm: defend by maximizing relevance, not by detecting attacks. Our method, MINE, is training-free and attack-agnostic, and it also serves as a query-conditioned reranker. To evaluate defenses under realistic conditions, we introduce LC-Inj-SQuAD, a long-context benchmark with an average of 4,032 words per context. We evaluate on four datasets spanning short-context QA (Inj-SQuAD, Inj-TriviaQA), long-context QA (LC-Inj-SQuAD), and agent tool-call injection (InjecAgent). Across all four, MINE achieves the best accuracy--security trade-off among training-free defenses. It is also the only defense that stays usable against a fully adaptive attacker, 
    
[^42]: 通过思维链向量投影实现轻量级推理能力迁移

    Lightweight Reasoning Transfer via Chain-of-Thought Vector Projection

    [https://arxiv.org/abs/2611.34183](https://arxiv.org/abs/2611.34183)

    该论文提出一种保持师生模型均冻结的轻量级两阶段跨模型思维链向量投影迁移方法，无需教师输出或解题轨迹，仅用少量数据的最终答案监督即可将大模型推理能力迁移到小模型，在GSM8K和MATH上提升基础学生模型准确率10.37-21.98个百分点。

    

    将大型语言模型的推理能力迁移到小型模型通常需要大量的教师监督或代价高昂的学生模型适配。我们研究了投影式跨模型思维链向量迁移方法，这是一种保持两个模型均冻结的两阶段方法。首先，我们利用数据集提供的解题轨迹和最终答案监督，在教师模型中学习任务级的思维链向量。然后，我们训练一个轻量级线性投影器，将该固定向量映射到较小的学生模型的残差空间中。学生端的训练仅在任务训练示例的一个小子集上使用最终答案交叉熵；它既不需要教师模型的输出，也不需要中间解题轨迹。我们在一个带标签的支持集上选择学生模型的注入层和干预规模，并在每个评估分割的互斥剩余部分上报告结果。在GSM8K和MATH数据集上，投影向量在整个评估中将基础学生模型的准确率提升了10.37-21.98个百分点。

    arXiv:2611.34183v1 Announce Type: new  Abstract: Transferring reasoning capabilities from large language models to smaller ones often requires extensive teacher supervision or costly student adaptation. We study projected cross-model Chain-of-Thought (CoT) Vector transfer, a two-stage approach that keeps both models frozen. First, we learn a task-level CoT Vector in the teacher using dataset-provided solution traces and final-answer supervision. We then train a lightweight linear projector to map the fixed vector into a smaller student's residual space. Student-side training uses only final-answer cross-entropy on a small subset of task training examples; it requires neither teacher outputs nor intermediate solution traces. We select the student injection layer and intervention scale on a labeled support set and report results on the disjoint remainder of each evaluation split. On GSM8K and MATH, the projected vector improves base students by 10.37-21.98 accuracy points across the eval
    
[^43]: 熟悉的词语，陌生的心智：心智词汇中的无校准惯例化

    Familiar Words, Unfamiliar Minds: Conventionalization without Calibration in the Vocabulary of Mind

    [https://arxiv.org/abs/2611.34179](https://arxiv.org/abs/2611.34179)

    该论文区分了将心智词汇延伸至语言模型时的两个独立过程——“惯例化”与“校准”，并通过对“幻觉”一词的语料库研究表明，词汇可在三年内完成惯例化，但由于语言模型与人类心智的组织方式根本不同，其推断含义的校准未必同步，甚至有依据的心智归因也可能误导。

    

    说话者通过延伸旧词来描述新技术，让对新指称对象的经验来厘清哪些推断可以延续。我认为，将心智词汇延伸至语言模型，会分离出隐喻理论与语言演变理论常常混为一谈的两个过程：惯例化——即被延伸的词不再被标记为特殊用法；以及校准——即该词所引发的推断被调整以匹配其新指称对象所能证实的内容。正如一个基因变异可以在并非适应性演化的情况下达到固定，一个词也可以在未经校准的情况下成为惯例用法。一项针对8,568篇摘要和31,866条论坛评论中“幻觉”一词的语料库研究显示，惯例化在三年内基本完成。然而校准未必与之同步：这些模型的组织方式与人类心智迥然不同，以至于很难判断心智词汇何时适用、又意味着什么。即使是有依据的归因也可能造成误导：机制分析……

    arXiv:2611.34179v1 Announce Type: new  Abstract: Speakers describe new technologies by stretching old words, letting experience of the new referents sort out which inferences carry over. I argue that stretching the vocabulary of mind to language models pulls apart two processes that theories of metaphor and linguistic change tend to conflate: conventionalization, whereby a stretched word ceases to be marked as such, and calibration, whereby the inferences it invites are fitted to what its new referents bear out. As a gene variant can reach fixation without being an adaptation, a word can become conventional without being calibrated. A corpus study of "hallucinate" in 8,568 abstracts and 31,866 forum comments shows conventionalization running most of its course within three years. Calibration need not keep pace: these models are organized so unlike human minds that it is hard to tell when mental words apply and what they imply. Even warranted attributions can mislead: mechanistic analys
    
[^44]: 重采样方法如何帮助我们：改进Delta模型

    How resampling method can help us: Improving the Delta model

    [https://arxiv.org/abs/2611.34137](https://arxiv.org/abs/2611.34137)

    本文提出对文体特征应用重采样方法来改进Delta作者归属模型，在保持结果可解释性的同时提高了准确率，但代价是需要额外的计算时间。

    

    Delta作为一种作者归属方法已在多种情境中得到成功应用，但其有效性并非总是完美。由于该方法基于最高频词汇，其结果易于用通俗易懂的语言进行解释。在本研究中，我们提出通过对所选文体特征应用重采样方法来提升其性能。结果表明，这种方法在保持结果可解释性的同时提高了模型的准确率。然而，所提出技术的主要缺点是需要额外的计算时间才能获得更准确的结果。

    arXiv:2611.34137v1 Announce Type: new  Abstract: Delta has been successfully employed as an authorship attribution method in various contexts, although its effectiveness is not always perfect. Since it is based on the most frequent words, its results can be readily interpreted and explained in plain English. In this study, we propose improving its performance by applying a resampling method to the selected stylistic features. The results show that this approach increases the accuracy of the model while preserving the interpretability of its outcomes. However, the main drawback of the proposed technique is the additional computational time required to obtain more accurate results.
    
[^45]: 萨拉·奥马尔是《洗尸者》的真正作者吗？丹麦文学中的作者归属研究

    Is Sara Omar the true author of D{\o}devaskeren (The Dead Washer)? Authorship attribution in Danish literature

    [https://arxiv.org/abs/2611.34131](https://arxiv.org/abs/2611.34131)

    本研究运用Delta、文本间距离、最近收缩质心和朴素贝叶斯四种作者归属方法，发现丹麦畅销小说《洗尸者》的大部分内容更符合马德斯·佩德·诺德博的写作风格而非署名作者萨拉·奥马尔，但少数段落暗示可能还有另一位作者参与创作。

    

    畅销书《洗尸者》于2017年以萨拉·奥马尔的名义出版。一位丹麦同行提出，这部小说可能是由另一位作者撰写的，可能是马德斯·佩德·诺德博。为了调查这一作者归属问题，研究者应用了四种作者归属方法，即Delta方法、文本间距离、最近收缩质心法和朴素贝叶斯。在多种参数设置下，这些方法一致地在存疑小说中识别出诺德博的写作风格，尽管这些结果并不能提供他撰写整部小说的决定性证据。在第二步中，小说被划分为互不重叠的5000词片段。同样地，大多数片段中识别出了诺德博的风格。然而，少数段落并不明确支持这一结论，反而表明可能还有另一位尚未确认的作者参与创作，可能是萨拉·奥马尔……

    arXiv:2611.34131v1 Announce Type: new  Abstract: The bestseller D{\o}devaskeren (The Dead Washer) was published in 2017 under the name Sara Omar. A Danish colleague has suggested that the novel may have been written by another author, possibly Mads Peder Nordbo. To investigate this authorship question, four authorship attribution methods were applied, namely Delta, intertextual distance, Nearest Shrunken Centroids (NSC), and na\"ive Bayes. Using various parameter settings, these approaches consistently identify Nordbo's writing style in the doubtful novel, although the results do not provide conclusive evidence that he authored the novel in its entirety. In a second step, the novel was divided into non-overlapping segments of 5,000 tokens. Once again, Nordbo's style was identified in the majority of the segments. However, a small number of passages do not clearly support this conclusion and instead suggest the possible involvement of another, as yet unidentified, author, possibly Sara 
    
[^46]: 看不见，但线索犹存：视频大语言模型能从部分证据中进行猜测吗？

    Out of Sight, Not Out of Clues: Can Video LLMs Guess from Partial Evidence?

    [https://arxiv.org/abs/2611.34126](https://arxiv.org/abs/2611.34126)

    提出诊断基准ClueProbe，通过将问题划分为“可回答、可猜测、不可回答”三类，把视频大语言模型的证据感知能力与证据充分性判断能力解耦，揭示其在部分证据下的猜测能力缺陷，并借助针对性微调验证了诊断结论。

    

    在自我中心录像、视觉无障碍辅助、监控审查等现实应用中，视频大语言模型所获得的视觉证据往往只是部分被捕捉到的。一个可靠的模型应当使回应与证据相匹配：证据充分时作答，证据部分可见时基于可见线索进行猜测，证据缺失时选择拒答。然而，现有基准强制采用“可回答或不可回答”的二元划分，使得部分证据情形未被测量，也掩盖了模型的失败究竟源于证据缺失还是对证据充分性的误判。我们提出了ClueProbe，一个诊断性基准，旨在：（1）将证据感知能力与充分性判断能力分离；（2）通过针对性微调来验证我们的诊断。ClueProbe包含基于598个自然视频的1,721个经人工标注的问题，标记为“可回答”、“可猜测”或“不可回答”，并配合改变措辞或视觉输入的受控扰动。我们对10个开源及专有模型进行了评估……

    arXiv:2611.34126v1 Announce Type: cross  Abstract: In real-world applications like egocentric recording, visual accessibility, and surveillance review, visual evidence for Video Large Language Models (Video LLMs) is often only partially captured. A reliable model should match its response to the evidence: answering when sufficient, guessing from visible clues when partial, and abstaining when absent. Yet existing benchmarks enforce a binary answerable-or-unanswerable split, leaving partial evidence unmeasured and obscuring whether failures arise from missing evidence or misjudging sufficiency. We introduce ClueProbe, a diagnostic benchmark to (1) separate perceiving evidence from judging sufficiency, and (2) validate our diagnosis via targeted fine-tuning. ClueProbe comprises 1,721 human-annotated questions across 598 natural videos, labeled Answerable, Guessable, or Unanswerable, alongside controlled perturbations altering wording or visual input. Evaluating 10 open-source and proprie
    
[^47]: 从资料室到报告：面向并购法律尽职调查的端到端AI支持

    From Data Room to Report: End-to-End AI Support for M&A Legal Due Diligence

    [https://arxiv.org/abs/2611.34120](https://arxiv.org/abs/2611.34120)

    LEXI是一个利用大语言模型支持并购法律尽职调查全流程的端到端AI系统，在律师人工审核的控制下完成文档分类、信息提取与报告编制。

    

    并购中的法律尽职调查要求律师整理异构的虚拟资料室文档，提取与交易相关的信息，并将审核后的发现整合为尽职调查报告。我们提出了LEXI（Legal Extraction and Intelligence，法律提取与智能系统），这是一个进行中的系统，展示了大语言模型如何支持从文档接入到报告编制的全过程。LEXI在可配置的尽职调查分类体系下对文档进行分类，分配相关目标实体，提取带有支持性段落和页级来源溯源的预定义信息，并根据审核后的发现编制报告内容。结构化输出连接了各个阶段，使模型生成的结果能够以受控的形式在工作流中传递。律师负责审核和纠正分类与提取结果，并保留对法律评估、建议、风险评级和最终报告措辞的最终责任。

    arXiv:2611.34120v1 Announce Type: new  Abstract: Legal due diligence in mergers and acquisitions requires lawyers to organize heterogeneous virtual data room documents, extract transaction-relevant information and consolidate reviewed findings into a due diligence report. We present LEXI (Legal Extraction and Intelligence), a work-in-progress system that demonstrates how large language models can support this process from document ingestion to report preparation. LEXI classifies documents under a configurable due diligence taxonomy, assigns relevant target entities, extracts predefined information with supporting passages and page-level provenance and prepares report content from reviewed findings. Structured outputs connect these stages and allow model-generated results to be passed through the workflow in a controlled form. Lawyers review and correct classifications and extractions and retain responsibility for legal assessments, recommendations, risk ratings and final report wording
    
[^48]: 通过隐状态混合与梯度分离改进尾部增强的推测草稿生成

    Improving Tail-Augmented Speculative Drafting with Hidden State Mixing and Gradient Detach

    [https://arxiv.org/abs/2611.34092](https://arxiv.org/abs/2611.34092)

    本文提出两种仅用于训练阶段的方法——隐状态混合（HSM）和隐状态梯度分离（HSGD），在不显式模拟推理、不增加额外内存和专用注意力掩码的情况下，缓解推测解码草稿模型的训练-推理不匹配，从而改进自回归草稿生成。

    

    arXiv:2611.34092v1 公告类型：cross。多 token 预测（MTP）模块和 EAGLE 风格的草稿模型通过复用目标模型的隐状态来支持高效的推测解码。然而，它们的自回归使用方式带来了训练与推理之间的不匹配：训练时提供的隐状态序列来自骨干模型或单一的 MTP 步骤，而推理时则面向目标模型的前缀，并附加一段由递归生成的 MTP 状态构成的、不断增长的尾部。EAGLE-3 的训练时测试（TTT）方法在训练期间显式模拟了这一行为，但代价是额外的内存开销和专门的注意力掩码。我们提出两种仅作用于训练阶段的方法来改进自回归草稿生成：隐状态混合在对应同一预测目标的骨干隐状态和 MTP 隐状态之间进行采样，使草稿模型接触到来自不同 MTP 步骤的状态组合，而无需显式模拟推理过程；隐状态梯度分离则是一种正则化方法，它随机地……（摘要原文在此处截断）

    arXiv:2611.34092v1 Announce Type: cross  Abstract: Multi-token prediction (MTP) modules and EAGLE-style drafters support efficient speculative decoding by reusing target-model hidden states. Their autoregressive use, however, introduces a training-inference mismatch: training supplies sequences of hidden states from the backbone or a single MTP step, whereas inference attends to a target-model prefix followed by a growing tail of recurrently generated MTP states. EAGLE-3's training-time test (TTT) explicitly simulates this behavior during training, at the cost of additional memory and specialized attention masks. We introduce two training-only methods for improving autoregressive drafting. Hidden State Mixing (HSM) samples among backbone and MTP hidden states aligned to the same prediction target, exposing the drafter to mixtures of states from different MTP steps without explicitly simulating inference. Hidden State Gradient Detach (HSGD) is a regularization method that stochastically
    
[^49]: 规范的拟人化：如何将认知语言扩展至人工智能

    Disciplined Anthropomorphism: How to Extend Cognitive Language to Artificial Intelligence

    [https://arxiv.org/abs/2611.34090](https://arxiv.org/abs/2611.34090)

    本文提出“规范的拟人化”理论，主张不再纠缠于AI是否符合为人类铸造的认知概念，而是解构认知概念中的推理关联，依据人类概念需求决定哪些关联应为AI保留、切断或重新建立，从而为认知语言向人工智能的扩展提供规范方法。

    

    研究人员越来越多地使用“理解”、“相信”等认知术语来描述人工智能，而批评者则指责这是一种一厢情愿的拟人化。然而，双方都采取了一种“归类式”的方法：他们追问人工智能系统是否属于为人类而铸造的概念范畴，并将判断结果作为对这些系统的发现来报告，尽管这些判断在很大程度上是由概念框架所决定的。然而，人工智能带来的问题恰恰是如何在概念上框定这些系统，而拟人化的爆发式兴起最好被看作是对这一问题的一种回应——它需要的是规范而非劝阻。我认为，规范的拟人化应当解开我们认知概念中捆绑在一起的推理关联，并追问哪些关联应当为人工智能予以保留、切断或重新建立。这种规范一方面来自人类的概念需求，正是这些需求赋予了这些关联的构建以意义……

    arXiv:2611.34090v1 Announce Type: new  Abstract: Researchers increasingly describe AI in cognitive terms such as "understanding" and "believing", while critics denounce this as wishful anthropomorphism. Both sides align in taking a subsumptive approach, however: they ask whether AI systems fall under concepts minted for human beings, reporting verdicts as discoveries about those systems when they are largely determined by conceptual frames. Yet the problem AI poses is precisely the problem of how to conceptually frame these systems, and the explosive rise of anthropomorphism is best seen as a response to this problem that needs disciplining rather than discouragement. A disciplined anthropomorphism, I argue, should disentangle the inferential connections bundled up in our cognitive concepts and ask which connections to preserve, sever, or forge for AI. The disciplining then comes, on the one hand, from the human conceptual needs that give point to the drawing of these connections and m
    
[^50]: 预算、读取还是推理？小型推理模型在拍卖竞价中于何处损失价值

    Budget, Readout or Reasoning? Where Small Reasoning Models Lose Value in Auction Bidding

    [https://arxiv.org/abs/2611.34068](https://arxiv.org/abs/2611.34068)

    该论文将六个4B至14B参数的开源推理模型置于基于真实eBay成交价和模拟广告市场的拍卖出价景观中，并对照精确计算器按原因拆分其价值损失，首次揭示损失主要源于模型自身的两类推理失败——忽视自身轨迹中已排名第一的最优选项、错误答案系统性偏向最高出价——导致竞标者损失7-17%的盈余。

    

    广告平台和市场会向竞标者展示“出价景观”，即在每个候选出价下获胜的估计概率，并且越来越多地让基于推理语言模型构建的智能体来选择出价。我们向六个参数量为4B至14B的开源推理模型提供了这样的出价景观，其获胜曲线来自eBay的成交价格和一个模拟的广告市场（AuctionNet），并在留出的拍卖上对照一个精确计算器对它们的出价进行评分。由于获胜概率估计是直接交给智能体的，因此损失的价值可以按原因进行拆分。两类失败归属于模型自身：即使其推理轨迹中写下的数值已经把最优选项排在第一位，它们仍会跳过该选项；而且它们的错误答案倾向于选择菜单中出价最高的选项。这种倾向与出价的大小有关，而非其在列表中的位置，并且在采样中依然存在。在eBay数据上、在“按出价付费”的定价机制下、强制生成768个token时，错误答案使竞标者损失了计算器出价所能获得盈余的7-17%。算术……

    arXiv:2611.34068v1 Announce Type: new  Abstract: Ad platforms and marketplaces show bidders a bid landscape, the estimated chance of winning at each candidate bid, and increasingly let agents built on reasoning language models choose the bid. We give six open reasoning models of 4B to 14B parameters such landscapes, with win curves from eBay closing prices and from a simulated advertising market (AuctionNet), and score their bids on held-out auctions against an exact calculator. Because the agent is handed the estimates, the lost value can be split by cause. Two failures belong to the models. They pass over the best option even when the values written in their own trace already rank it first, and their wrong answers lean toward the highest bid on the menu. The lean is about the bid's size, not its place in the list, and it survives sampling. At 768 forced tokens on eBay, under pay-your-bid pricing, the wrong answers cost bidders 7-17% of the surplus the calculator's bids earn. Arithmet
    
[^51]: 稳定的门控DeltaNet-2：理解与缓解循环状态不稳定性

    Stable Gated DeltaNet-2: Understanding and Mitigating Recurrent State Instability

    [https://arxiv.org/abs/2611.34051](https://arxiv.org/abs/2611.34051)

    本文发现GDN-2训练失败源于擦除操作中读写方向不对称导致的循环状态指数级增长，并提出通过对称擦除与衰减调节残差写入来解决该问题，同时理论上证明了状态增长的O(√T)上界并保留分块并行计算能力。

    

    线性注意力凭借线性复杂度和固定大小的循环状态，为长序列建模提供了高效解决方案。门控DeltaNet-2（GDN-2）引入了对擦除和写入的独立通道级控制，增强了状态更新的表达能力以及上下文内检索能力，但它在长上下文训练中会出现训练失败。我们将这些失败归因于擦除过程中读取方向与修改方向之间的不对称性，这种不对称性产生了非正规（non-normal）的擦除算子，其方向性增益会在token之间累积，即使没有新的写入也会引发状态的指数级增长。为解决这一问题，我们提出了该模型，采用对称擦除以确保状态传播的非扩张性，并通过衰减调节的残差写入来保留写入的灵活性。在理论上，我们在保留分块并行计算的同时，建立了循环状态增长的 O(√T) 上界。在约16亿参数模型上的实验……（摘要原文在此处截断）

    arXiv:2611.34051v1 Announce Type: new  Abstract: Linear attention offers an efficient solution for long-sequence modeling via linear complexity and fixed-size recurrent states. Gated DeltaNet-2 (GDN-2) introduces independent channel-wise control over erasure and writing, enhancing state-update expressiveness and in-context retrieval, yet it suffers from training failures during long-context training. We trace these failures to the asymmetry between reading and modification directions during erasure, which yields non-normal erase operators whose directional gains accumulate across tokens, triggering exponential state growth even without new writes. To resolve this, we propose \model{}, employing symmetric erasure to ensure non-expansive state propagation and decay-regulated residual writing to preserve writing flexibility. Theoretically, we establish an $O(\sqrt{T})$ upper bound on recurrent state growth while retaining chunkwise parallel computation. Experiments on ~1.6B parameter mode
    
[^52]: 复杂多步骤工作流中的可靠用户模拟

    Reliable User Simulation in Complex Multi-Step Workflows

    [https://arxiv.org/abs/2611.34044](https://arxiv.org/abs/2611.34044)

    本文提出了一个基于分层任务网络（HTN）抽象、包含500个工作流的基准，揭示出LLM用户模拟器在复杂多步骤工作流中的保真度随复杂度急剧下降，且目标漂移是导致失败的主要因素。

    

    基于LLM的用户模拟器被越来越多地用于评估对话助手。然而，这些模拟器可能会偏离其被指定的目标，或无法穷尽地探索所有指定的选项，从而导致评估错误，并将本不属于助手的错误归咎于助手。先前的工作研究了单任务设置中的目标对齐问题，但许多实际场景要求模拟器执行包含顺序任务、回退逻辑和跨任务依赖关系的多步骤工作流，而模拟器在这种场景下的保真度尚未被测量。为弥补这一空白，我们引入了一个包含500个工作流的基准，这些工作流采用分层任务网络（HTN）抽象来表示。我们的实验表明，模拟器的保真度随工作流复杂度的增加而急剧下降，且目标漂移是大多数失败的主要原因。保真度还取决于助手的响应风格，同一个模拟器在不同风格下的得分可能存在显著差异。为了解决……（原文摘要在此处截断）

    arXiv:2611.34044v1 Announce Type: new  Abstract: LLM-based user simulators are increasingly used to evaluate conversational assistants. Yet, these simulators can drift from their assigned goals or fail to exhaustively explore all specified options, causing evaluation errors incorrectly attributed to the assistant. Prior work has examined goal alignment in single-task settings, but many practical scenarios require simulators to execute multi-step workflows with sequential tasks, fallback logic, and cross-task dependencies, where simulator fidelity has not been measured. To close this gap, we introduce a benchmark of 500 workflows represented using the Hierarchical Task Network (HTN) abstraction. Our experiments reveal that the simulator's fidelity degrades sharply with workflow complexity, and that goal drift accounts for the majority of failures. Fidelity also depends on the assistant's response style, with the same simulator scoring substantially better or worse across styles. To addr
    
[^53]: 过于聪明反受其害？大语言模型模拟低龄学习者中的认知保真度与能力泄露

    Too Smart for Its Own Good? Cognitive Fidelity and Competence Leakage in LLM Simulations of Young Learners

    [https://arxiv.org/abs/2611.34042](https://arxiv.org/abs/2611.34042)

    本文提出并验证了“能力泄露”这一现象——LLM模拟的低龄学习者会表现出超出其被分配认知状态的知识与解题能力，且“能力过强”是交互中认知保真度错位的主要表现形式，这对教育类学习者模拟器的可靠性构成重要挑战。

    

    基于大语言模型（LLM）的交互式学习者模拟器在教育应用中的使用日益广泛，包括教师培训和辅导系统开发。这类模拟的一个关键要求是认知保真度：被模拟的学习者应在整个交互过程中保持所分配的知识、误解（misconceptions）和推理过程。然而，LLM 可能会表达超出所分配学习者认知状态的知识或问题解决技能，我们将这一现象称为“能力泄露”。在这项探索性研究中，我们分析了来自130份师生模拟对话记录、涵盖三种学习场景的3,339条模拟学习者话语，发现“能力过强”是交互过程中认知档案错位的一个突出方向。随后，我们在四种匹配触发的评估协议下考察了先前观察到的能力泄露案例的复现情况，这些协议涵盖了从孤立提问到多轮交互的不同情境。

    arXiv:2611.34042v1 Announce Type: new  Abstract: LLM-based interactive learner simulators are increasingly used in educational applications, including teacher training and tutoring system development. A critical requirement for those simulations is cognitive fidelity: the simulated learner should maintain the assigned knowledge, misconceptions, and reasoning procedures throughout interaction. However, LLMs may express knowledge or problem-solving skills beyond the assigned learner's cognitive status, a phenomenon we refer to as competence leakage. In this exploratory study, we examine 3,339 simulated learner utterances from 130 teacher-learner simulator transcripts across three learning scenarios and identify over-competence as a prominent direction of cognitive-profile misalignment during interaction. We then investigate the recurrence of previously observed competence leakage cases under four matched-trigger evaluation protocols, ranging from isolated questioning to multi-turn intera
    
[^54]: JHealthBench-L：面向日本基于评分准则的医学大语言模型评估的临床医生审校版HealthBench本地化

    JHealthBench-L: Clinician-Reviewed Localization of HealthBench for Rubric-Based Medical LLM Evaluation in Japan

    [https://arxiv.org/abs/2611.34024](https://arxiv.org/abs/2611.34024)

    该论文提出了JHealthBench-L，一个通过LLM起草加多阶段临床医生审校流程、将HealthBench的评分准则本地化到日本医疗情境的基准，解决了机器翻译无法消除准则中区域性医疗假设从而导致评估不公的问题。

    

    诸如HealthBench等基于评分准则的基准测试，使用针对具体病例的医生准则来评估大语言模型（LLM）在开放式健康咨询中的表现，而其中许多准则都预设了其来源地区的医疗机构、药物和诊疗路径。仅靠机器翻译，这些区域性假设仍会残留在评分准则中，导致在日本情境下恰当的回复得不到认可，或被外国的临床规则错误地扣分。为了在不从零编写评分准则的情况下，将已有的临床评估视角适配到日本，我们构建了JHealthBench-L（79个面向患者的咨询，1,022条准则），其构建流程包括：基于引用原文的LLM起草、单一专科医生的筛选与修订、独立的双医生审阅以及元审阅。专科医生仅从零重写了4.4%的准则，独立审阅者对93.6%的第二阶段准则无异议地予以通过。通过将JHealthBench-L与机器翻译基线进行比较……

    arXiv:2611.34024v1 Announce Type: new  Abstract: Rubric-based benchmarks such as HealthBench evaluate large language models (LLMs) on open-ended health consultations using case-specific physician criteria, many of which presuppose the healthcare institutions, medications, and care pathways of their source region. Under machine translation alone, these regional assumptions remain in the rubric, leaving responses appropriate in Japan uncredited or falsely penalized under foreign clinical rules. To adapt established clinical evaluation perspectives to Japan without authoring rubrics from scratch, we construct JHealthBench-L (79 patient-facing consultations, 1,022 criteria) through source-cited LLM drafting, single-specialist selection and revision, independent dual-physician review, and meta-review. Specialists rewrote only 4.4% of criteria from scratch, and independent reviewers approved 93.6% of Stage-2 criteria without issue. Comparing JHealthBench-L with a machine-translated baseline 
    
[^55]: 评估语言模型中的乌克兰语语法能力：概率法与提示法之比较

    Evaluating Ukrainian Grammatical Competence in Language Models: Probability Versus Prompting

    [https://arxiv.org/abs/2611.34016](https://arxiv.org/abs/2611.34016)

    该论文构建了一套包含1,444个乌克兰语最小对立对的语法评测协议，通过比较概率与直接提示两种方法评估19个语言模型的规范乌克兰语语法能力，并显式控制句子呈现顺序带来的答案位置效应。

    

    在乌克兰语信息系统中部署大语言模型的机构需要了解模型能否产出符合规范的乌克兰语。本文建立了一套可靠的测量协议，并将其应用于19个模型。测评工具是一套公开发布的、包含1,444个乌克兰语最小对立对的题集，涵盖呼格、数词与名词的一致、形容词一致、动词支配以及源自俄语的仿译词，并以主谓一致作为对照；其中四分之一的题目由两名母语者依据答案键进行独立评分核查，两者一致性达到 Cohen's κ = 0.963。每一对句子均采用两种方式评分：一是比较模型赋予两个句子的概率，二是直接向模型提问，且两个句子以两种顺序呈现，从而测量答案位置效应而非将其吸收进总分中。由此得出三项结果。顺序一致性（即当句子顺序……（原文摘要在此处截断）

    arXiv:2611.34016v1 Announce Type: cross  Abstract: Organisations deploying large language models in Ukrainian-language information systems need to know whether a model produces normative Ukrainian. This paper establishes a protocol for measuring that reliably and applies it to 19 models. The instrument is a released set of 1,444 Ukrainian minimal pairs covering the vocative case, numeral-noun and adjective agreement, verb government and calques from russian, with subject-verb agreement as a control; a quarter of the set was checked against the answer key in two independent native-speaker ratings, which agreed at Cohen's $\kappa$ = 0.963. Every pair is scored twice: by comparing the probabilities a model assigns to the two sentences, and by asking the model directly, with the sentences presented in both orders so that answer-position effects are measured instead of being absorbed into the score. Three results follow. Order consistency (the share of pairs answered identically when the se
    
[^56]: 共享提示词下的残差跨模型叙事趋同：提示词控制与叙事测量的审计研究

    Residual Cross-Model Narrative Convergence under Shared Prompts: Auditing Prompt Controls and Narrative Measurement

    [https://arxiv.org/abs/2611.34015](https://arxiv.org/abs/2611.34015)

    本研究提出残差趋同指数（RCI≈0.41），通过比较相同提示词与语义匹配提示词下五个人工智能大模型生成故事的叙事距离，揭示了即便控制提示词内容后依然存在的可测量跨模型叙事趋同现象。

    

    基于相同提示词由大语言模型创作的故事在叙事上可能彼此相似。这种相似性既可能反映了提示词所施加的约束，也可能反映了模型自身的写作倾向。本研究分析了StoryScope语料库中五个模型基于10,248个共同提示词创作的51,240篇故事。研究使用七个维度的213项叙事特征，比较由相同提示词生成的故事之间的跨模型距离与由不同但语义匹配的提示词生成的故事之间的距离。为评估叙事特征与提示词内容之间的关系，研究针对420条由作者标注的提示词-特征判断对语义相关性掩码进行了评估（F1 = .813；临界准确率 = .621）。等权重平均残差趋同指数（RCI）在无掩码条件下为 .409，在有掩码条件下为 .406。这些发现表明，相对于……（原文摘要在此处截断），存在可测量的跨模型叙事趋同。

    arXiv:2611.34015v1 Announce Type: cross  Abstract: Stories created by LLMs based on the same prompts can be narratively similar to one another. This similarity may reflect constraints imposed by the prompts as well as the models' writing tendencies. The current research analyzes 51,240 stories created by five models in the StoryScope corpus based on 10,248 common prompts. It uses 213 narrative features in seven dimensions to compare cross-model distances between stories generated from the same prompts with distances between stories generated from different but semantically matched prompts. To assess the relationship between narrative features and prompt content, a semantic relevance mask was evaluated against 420 author-annotated prompt-feature judgments (F1 = .813; near-threshold accuracy = .621). The equally weighted mean Residual Convergence Index (RCI) was .409 without masking and .406 with masking. These findings indicate measurable cross-model narrative convergence relative to a 
    
[^57]: Auto-Features：一个用于上下文识别汽车标准配置与选装配置的LLM辅助语料库

    Auto-Features: An LLM-Assisted Corpus for Contextual Recognition of Standard and Optional Car Features

    [https://arxiv.org/abs/2611.33974](https://arxiv.org/abs/2611.33974)

    该论文提出了Auto-Features语料库，将汽车配置识别任务建模为特征条件化的命名实体识别，构建了包含人工黄金层与LLM白银层的双层语料库，并提供了一套针对LLM各类标注失败逐一验证与补救的系统化质量控制流程。

    

    我们提出了Auto-Features，这是一个在给定某确切车辆的功能目录的前提下，用于在广告文本中检测汽车标准配置和选装配置的语料库。由于销售者会用多种不同方式命名同一配置，我们将该任务设定为特征条件化（即上下文化）的命名实体识别：提取表达某一给定目录配置的文本片段。基于相同的输出还可以得到实体链接的视角。该语料库包含一个由经过筛选的标注员构建的人工黄金层（430条广告）和一个由基于搜索的大语言模型自动标注的LLM白银层（925条广告）。我们还贡献了该语料库的构建方法：每一个LLM输出都经过验证，且每种类型的失败都有其对应的补救措施——格式错误的输出会被修补，幻觉内容会触发由更强模型重新尝试，指向候选列表之外配置的链接会被丢弃并计数（10,928条记录中仅有3条）。只有当LLM与人工标注员在链接验证上的一致性达到……（原文在此处截断）

    arXiv:2611.33974v1 Announce Type: new  Abstract: We present Auto-Features, a corpus for detecting a car's standard and optional features in advertisement text, given the feature catalogue of that exact vehicle. Sellers name the same feature in many ways. We therefore cast the task as feature-conditioned, or contextual, named entity recognition: extract the span that expresses one given catalogue feature. An entity-linking view follows from the same output. The corpus has a human-gold layer (430 adverts) from screened annotators and an LLM-silver layer (925 adverts) auto-annotated by a search-grounded LLM. We also contribute the way it was built. Every LLM output is validated, and each kind of failure has its own remedy. Malformed output is patched. Hallucinated content triggers a fresh attempt by a stronger model. Links to features outside the candidate list are dropped and counted (3 in 10,928 records). The LLM annotates alone only after its agreement with human annotators on link ver
    
[^58]: 无需梯度的微调？在冻结编码器上由大语言模型编写的闭式解分类器

    Fine-Tuning Without Gradients? Closed-Form, LLM-Written Classifiers on a Frozen Encoder

    [https://arxiv.org/abs/2611.33972](https://arxiv.org/abs/2611.33972)

    该论文提出闭式权重空间编程（WSP）框架，让大语言模型编写程序在冻结编码器上以闭式解拟合分类头，全程无需梯度更新和权重修改，却在多个标签预算下达到甚至超过全量微调、LoRA 等梯度方法的性能。

    

    在冻结编码器上以闭式解计算的分类器能否匹敌梯度微调？我们提出了闭式权重空间编程（WSP）：一个大语言模型（LLM）作为设计助手，编写程序通过多层完形填空式的提示模板读取一个冻结的掩码语言模型，在标注数据上拟合闭式分类头（收缩判别分析、核岭回归以及编码器的标签词先验），并通过交叉拟合从候选分类头池中进行选择。整个过程不计算任何梯度，也不改变编码器的权重。我们在 RoBERTa-large 上实例化了该框架。在 11 个包含 512 至 8,192 个训练标签的开发数据集上，WSP 在每个标签预算下均优于全量微调（FT）和 LoRA（平均高出 0.3 至 0.6 个百分点；预设的非劣性检验通过），且与基于提示的微调（PFT）的差距在 0.5 个百分点以内；这些数据集也参与了方案的构建。在标签更少的情况下，其相对于梯度微调方法的优势进一步扩大。

    arXiv:2611.33972v1 Announce Type: cross  Abstract: Can a classifier computed in closed form on a frozen encoder match gradient fine-tuning? We present closed-form weight-space programming (WSP): a large language model (LLM), used as a design assistant, writes programs that read a frozen masked language model through cloze-style prompt templates at several layers, fit closed-form heads (shrinkage discriminant analysis, kernel ridge regression and the encoder's label-word prior) on labelled data, and select among a pool of candidate heads by cross-fitting. No gradient is computed and no encoder weight changes. We instantiate the framework on RoBERTa-large. On 11 development datasets with 512 to 8,192 training labels, WSP is ahead of full fine-tuning (FT) and LoRA at every label budget (by 0.3 to 0.6 points on average; the registered non-inferiority tests pass) and within 0.5 points of prompt-based fine-tuning (PFT); these datasets also shaped the recipe. With fewer labels its lead over F
    
[^59]: DLaR：面向大语言模型潜在推理的动态随机性控制

    DLaR: Dynamic Stochasticity Control for Latent Reasoning in Large Language Models

    [https://arxiv.org/abs/2611.33962](https://arxiv.org/abs/2611.33962)

    该论文提出DLaR框架，通过在token层面基于置信度-熵融合信号自适应调节扰动强度、在训练层面感知训练进程调优随机性，解决了潜在推理中均匀随机性次优和无约束扰动破坏连贯性的问题，实现探索与稳定性的动态平衡。

    

    潜在推理使大语言模型（LLM）能够在连续嵌入空间中进行推理，然而现有方法受制于确定性推理，无法探索多样化的推理路径。尽管近期方法通过注入随机性来恢复探索能力，我们指出了两个根本性局限：在所有步骤上施加*均匀随机性*是次优的，且无约束的扰动会破坏推理的连贯性。我们提出了面向潜在推理的动态随机性控制（DLaR），这是一个包含两种互补机制的框架，这两种机制在GRPO强化学习框架内以不同的时间粒度运作。在token层面，自适应随机性调制（ASM）基于置信度-熵融合信号动态调整每个潜在步骤的扰动强度，在模型不确定时增强探索，在模型自信时保持稳定性。在训练层面，训练感知的随机性调优……

    arXiv:2611.33962v1 Announce Type: cross  Abstract: Latent reasoning enables LLMs to operate in continuous embedding space, yet existing methods suffer from deterministic inference that precludes exploration of diverse reasoning paths. While recent approaches inject stochasticity to restore exploration, we identify two fundamental limitations: \textit{uniform stochasticity} across all steps is suboptimal, and unconstrained perturbations disrupt reasoning coherence. We propose Dynamic Stochasticity Control for Latent Reasoning (DLaR), a framework with two complementary mechanisms that operate at different temporal granularities within the GRPO reinforcement learning framework. At the token level, Adaptive Stochasticity Modulation (ASM) dynamically adjusts perturbation intensity at each latent step based on a confidence-entropy fusion signal, increasing exploration when the model is uncertain and preserving stability when confident. At the training level, Training-Aware Stochasticity Tuni
    
[^60]: PINN-Slicer：利用旋转对称性求解黑洞周围的光子轨道方程

    PINN-Slicer: Exploiting Rotational Symmetry to Solve the Photon Orbit Equation Around a Black Hole

    [https://arxiv.org/abs/2611.33953](https://arxiv.org/abs/2611.33953)

    提出PINN-Slicer方法，利用二阶自治常微分方程在自变量平移下的不变性，将单个PINN逐片递归地部署到整个求解域，成功求解黑洞周围光子轨道方程，RMSE低于10^-3。

    

    物理信息神经网络（PINNs）通常在长时域问题上表现不佳，往往需要随着求解域的增长对网络进行重新训练。我们提出了PINN-Slicer，这是一种针对二阶自治常微分方程 $\ddot{u} = G(u, \dot{u})$ 的方法，它利用了这类方程在自变量平移下的不变性。单个PINN被嵌入到一个显式强制初始条件的解拟设中，使用Sobol采样的初始条件在自变量的一个薄切片上进行训练。随后，该PINN被递归地、逐片地部署到整个求解域上。我们在描述黑洞周围光子轨道的非线性方程上对PINN-Slicer进行了演示，相对于有限差分参考解，在 $u = r_s/r$（以史瓦西半径为单位的无量纲逆径向坐标）上达到了逐点均方根误差（RMSE）低于 $10^{-3}$ 的精度。

    arXiv:2611.33953v1 Announce Type: new  Abstract: Physics-Informed Neural Networks (PINNs) typically struggle with long-horizon domains, often requiring the network to be retrained as the domain grows. We present PINN-Slicer, a method for second-order autonomous ordinary differential equations, $\ddot{u} = G(u, \dot{u})$, which exploits the invariance of this class of equations under translations of the independent variable. A single PINN, embedded within a solution ansatz that enforces the initial conditions explicitly, is trained on a thin slice of the independent variable using Sobol-sampled initial conditions. The PINN is then deployed recursively, slice-by-slice, across the full domain. We demonstrate PINN-Slicer on the nonlinear equation that describes photon orbits around a black hole, and reach a point-by-point root-mean-square error (RMSE) below $10^{-3}$ in $u = r_s/r$ (the inverse radial coordinate in units of the Schwarzschild radius) relative to a finite-difference referenc
    
[^61]: Skill Lift 到底衡量的是什么？对 SkillsBench 的源代码级审计与噪声感知重分析

    What Does Skill Lift Actually Measure? A Source-Code-Level Audit and Noise-Aware Reanalysis of SkillsBench

    [https://arxiv.org/abs/2611.33943](https://arxiv.org/abs/2611.33943)

    该论文对 SkillsBench 进行源代码级审计并对 7,554 次公开试验做噪声感知重分析，发现部分“技能提升”分数实际上源于评分器依赖和操作性定义缺口等测量缺陷，并将总体平均提升修正为 18.7 pp。

    

    智能体技能是编程智能体按需加载的指令、参考资料和脚本所组成的目录。根据 SkillsBench 的报告，尽管人工策划的技能平均可将通过率提高约 16.6 个百分点（pp），但它们实际上在某些任务上反而降低了成绩。后续研究已开始依据上述每任务收益来门控或剔除技能。我们在源代码层面对该基准进行了审计，并分析了 7,554 次公开试验，形成 1,421 个配对的“（配置，任务）”单元格。源代码审计发现了两例经对照验证的评分器依赖问题和一个操作性定义缺口；这三个任务上的平均提升为 74.6 pp，而完整任务集的总体平均提升为 18.7 pp。审计还发现四个任务包含占位符技能目录，且某个执行框架在大多数运行中额外注入了 48 个内置技能。没有任务层面的降低……（摘要在此处截断）

    arXiv:2611.33943v1 Announce Type: new  Abstract: Agent skills are directories of instructions, reference materials, and scripts loaded on demand by coding agents. According to SkillsBench, although human-curated skills can raise the passing rate by about 16.6 percentage points (pp) on average, they have actually reduced the results for some tasks. Later research has started to gate or eliminate skills according to the above per-task gains. We have audited the above benchmark at the source-code level and analysed 7,554 public trials to form 1,421 paired "(configuration, task) cells". Source-code audit identified two controlled-verified cases of grader dependence and one operational definition gap; the average lift on these three tasks was 74.6 pp, and the overall average lift for the full set was 18.7 pp. The audit also found that four tasks contained placeholder skill directories, and a certain execution harness additionally added 48 built-in skills in most runs. No task-level reductio
    
[^62]: DiCALM：将自回归卷积-注意力语言模型转换为离散扩散语言模型

    DiCALM: Converting Autoregressive Convolution-Attention Language Models into Discrete Diffusion Language Models

    [https://arxiv.org/abs/2611.33919](https://arxiv.org/abs/2611.33919)

    DiCALM首次提出将混合卷积-注意力语言模型（LFM2家族）从自回归转换为掩码扩散生成的方案，通过双向扩展因果门控卷积、掩码预热和AR-扩散联合训练，在保持性能相当的同时实现平均1.4倍、最高1.79倍的推理加速。

    

    现有将自回归（AR）语言模型适配到扩散生成的方法主要集中于全注意力架构，而更高效的混合架构仍未得到充分探索。我们提出DiCALM，这是一种将LFM2家族（LFM2和LFM2.5）的混合卷积-注意力模型从自回归转换为掩码扩散生成的方案。DiCALM逐步扩展LFM2的因果门控卷积，使其能够双向捕获局部上下文，并将这一扩展与掩码预热及AR-扩散联合训练相结合。在对照实验中，DiCALM的性能与相应的基于LFM2.5微调的自回归模型相当，并在350M和1.2B两个规模上都实现了一致的加速：平均超过1.4倍，最高达1.79倍。

    arXiv:2611.33919v1 Announce Type: cross  Abstract: Existing methods for adapting autoregressive (AR) language models to diffusion generation focus mainly on full-attention architectures. More efficient hybrid architectures remain underexplored. We introduce DiCALM, a recipe for converting the hybrid convolution-attention models of the LFM2 family (LFM2 and LFM2.5) from AR to masked-diffusion generation. DiCALM progressively extends LFM2's causal gated convolution so that it captures local context in both directions. It combines this extension with a masking warm-up and joint AR--diffusion training. In controlled experiments, DiCALM performs comparably to the corresponding fine-tuned LFM2.5-based AR models and delivers consistent speedups over them at both the 350M and 1.2B scales: over 1.4x on average and up to 1.79x.
    
[^63]: 循环模型是快速的多词预测器

    Looped Models are Fast Multi-Token Predictors

    [https://arxiv.org/abs/2611.33896](https://arxiv.org/abs/2611.33896)

    提出Floodgates自推测解码方法，在循环语言模型末尾附加掩码词块并通过低秩适应并行起草多个未来词，实现高达7.5倍的无损推理加速。

    

    循环语言模型可以通过调整循环次数来适应预测一个词所需的计算量，而无需按比例缩放唯一的模型参数。这一特性已被用于自推测解码，将浅层循环作为快速起草者，深层循环作为验证者，这种方法之所以有效，是因为起草过程仍然使用完整参数。在本文中，我们证明浅层循环还可以预测多个未来词，并利用这一点来大幅加速循环语言模型的推理。为此，我们提出了Floodgates，一种简单的自推测解码方法，它在序列末尾附加一个掩码词块，并通过低秩适应学习并行起草这些词。在七个模型、五个基准测试、一系列批处理大小以及贪婪解码和采样解码的实验中，Floodgates在大多数情况下是最快的无损解码方法，实现了高达7.5倍的加速。通过分析，我们展示了……（原文在此处截断）

    arXiv:2611.33896v1 Announce Type: cross  Abstract: Looped language models can adapt computation for predicting a token without a proportional rescaling in unique model parameters by adjusting the number of loops. This property has been used for self-speculative decoding, taking shallow loops as fast drafters and deep loops as verifiers, which is effective since drafting still uses full parameters. In this paper, we show that shallow loops can be also predictive of several future tokens and leverage this to drastically accelerate the inference of looped language models. To that end, we present Floodgates, a simple self-speculative decoding method that appends a block of mask tokens as a suffix and learns via low-rank adaptation to draft these tokens in parallel. Across seven models, five benchmarks, a range of batch sizes, and both greedy and sampled decoding, Floodgates is the fastest lossless decoding method in most cases, attaining up to 7.5x acceleration. Through analysis, we show t
    
[^64]: 情境语言表征将叙事加工与儿童的语言和读写能力联系起来

    Contextual language representations link narrative processing to language and literacy in children

    [https://arxiv.org/abs/2611.33871](https://arxiv.org/abs/2611.33871)

    本研究结合功能磁共振成像与大语言模型的情境嵌入表征，发现儿童聆听故事时的大脑叙事加工反应（尤其在颞顶和默认网络区域）可由LLM情境表征有效预测，从而将神经层面的叙事加工与儿童的语言和读写能力联系起来。

    

    口语叙事将逐时刻的语言加工与通过语言和读写经验所积累的知识广泛地联系起来。本研究在儿童聆听粤语故事期间使用功能磁共振成像，并结合来自大语言模型（LLM）的情境表征来考察这种关系。成像队列包括23名学龄儿童；其中16名完成了完整的标准化读写与语言评估，并作为留存被试参与了对同一故事的嵌套跨被试编码分析。情境嵌入能够预测分布式的叙事反应，并在颞顶区域和默认网络分区中表现出显著的对应关系。按留存相关性排序的前25%体素的区域汇总在全脑图谱范围内得到了r=0.174的平均相关值和偏R²=0.0165。联合特征模型在皮层和皮层下区域识别出负向的惊讶度系数和正向的预测熵系数……

    arXiv:2611.33871v1 Announce Type: new  Abstract: Spoken narratives extensively connect moment-to-moment language processing with knowledge accumulated through language and literacy experience. This study examined that relationship using functional magnetic resonance imaging (fMRI) during Cantonese story listening and contextual representations from large language model (LLM). The imaging cohort comprised 23 school-age children; 16 contributed complete standardized literacy and language assessments and served as held-out participants in a nested cross-subject encoding analysis of the same story. Contextual embeddings predicted distributed narrative responses, with prominent correspondence in temporoparietal and default-network parcels. Regional summaries of the top 25% of voxels ranked by held-out correlation yielded atlas-wide means of r=0.174 and partial R2=0.0165. A joint feature model identified negative surprisal and positive predictive-entropy coefficients across cortical and subc
    
[^65]: ChemDeduce：通过交互式模拟实验对大语言模型进行未知化学物质识别能力的基准测试

    ChemDeduce: Benchmarking LLMs for Unknown Chemical Identification through Interactive Simulated Experiments

    [https://arxiv.org/abs/2611.33848](https://arxiv.org/abs/2611.33848)

    该论文提出了ChemDeduce——一个将有状态模拟实验室中未知化学物质识别建模为序贯决策过程的交互式评估框架，首次系统性地测试了大语言模型通过主动设计实验、迭代推理来发现未知答案的能力。

    

    大语言模型（LLMs）在化学知识和推理方面已展现出强大的性能，但它们能否解决那些必须通过实验才能发现答案的问题仍不清楚。我们通过“未知化学物质识别”任务来研究这一问题，即模型必须主动设计实验、解读观测结果、更新假设，并最终识别出隐藏的物质。我们提出了ChemDeduce，一个通过模拟实验来评估大语言模型的交互式评估框架。ChemDeduce将化学物质识别表述为一个序贯决策过程，并提供了一个具有明确仪器和操作约束的有状态模拟实验室，从而实现对迭代实验推理能力的可扩展、可复现的评估。我们在200种化学物质上对八个前沿大语言模型进行了基准测试，产生了1600个模型-目标组合，并在三次独立重复实验中开展了4800个回合的评估，对识别能力进行了评测。

    arXiv:2611.33848v1 Announce Type: new  Abstract: Large language models (LLMs) have shown strong performance in chemical knowledge and reasoning, yet it remains unclear whether they can solve problems whose answers must be discovered through experimentation. We investigate this question through Unknown Chemical Identification, where a model must actively design experiments, interpret observations, update hypotheses, and ultimately identify a hidden substance. We introduce ChemDeduce, an interactive evaluation framework for assessing LLMs through simulated experiments. ChemDeduce formulates chemical identification as a sequential decision process and provides a stateful simulated laboratory with explicit instrument and operational constraints, enabling scalable and reproducible evaluation of iterative experimental reasoning. We benchmark eight frontier LLMs on 200 chemicals, yielding 1600 model--target pairs and 4800 episodes over three independent repetitions, and evaluate identificatio
    
[^66]: 当重排序器协同工作时：无需联合训练的更好更快的重排序

    When Rerankers Work Together: Better and Faster Reranking without Joint Training

    [https://arxiv.org/abs/2611.33844](https://arxiv.org/abs/2611.33844)

    该研究提出两种无需联合训练的轻量级语义粗评分器，可与异构重排序器组合，在多阶段RAG流水线中显著提升答案准确率并加速在线剪枝。

    

    独立重排序器的质量并不能决定其在多阶段RAG（检索增强生成）流水线中的价值。我们系统地研究了无需联合训练即可组合的异构重排序器，涵盖词汇匹配、单向量、延迟交互以及联合查询-上下文评分等多种方法。我们提出了两种轻量级语义粗评分器——Cache-Pointwise和Cache-Shapley，它们从段落相关性标签中学习，共享可缓存的表示形式和线性时间的单例推理。在候选集合和预算匹配的条件下，我们将级联系统与移除特定阶段的参考系统进行比较，以衡量证据保留、修复与破坏、阅读器准确率以及在线剪枝成本。在使用Gemma模型的NoLiMa 32K数据集上，将这些评分器插入到CrossEncoder或Provence之前，可使答案准确率提高4.02-9.43个百分点，同时相对于单独使用相应重排序器，在线剪枝速度加快1.57-1.88倍。Cache-Pointwise通过倒数排名融合与ColBERTv2组合，可使准确率提高4.60个百分点。

    arXiv:2611.33844v1 Announce Type: cross  Abstract: Standalone reranker quality does not determine its value in a multistage RAG pipeline. We systematically study heterogeneous rerankers composed without joint training, spanning lexical, single-vector, late-interaction, and joint query-context scoring. Two lightweight semantic coarse scorers, Cache-Pointwise and Cache-Shapley, learn from passage relevance labels and share cacheable representations and linear-time singleton inference. Under matched candidates and budgets, we compare cascades with stage-removal references to measure evidence retention, repair and breakage, reader accuracy, and online pruning cost. On NoLiMa 32K with Gemma, inserting these scorers before CrossEncoder or Provence improves answer accuracy by 4.02-9.43 percentage points while accelerating online pruning by 1.57-1.88x relative to the corresponding reranker alone. Cache-Pointwise composed with ColBERTv2 through reciprocal rank fusion improves accuracy by 4.60 p
    
[^67]: 缺失的根基：大型语言模型中无需图像的几何想象

    The Missing Root: Geometric Imagination Without Images in Large Language Models

    [https://arxiv.org/abs/2611.33841](https://arxiv.org/abs/2611.33841)

    该论文提出将包含位姿与带符号距离形状嵌入的连续物体token嵌入transformer序列，使大型语言模型无需依赖图像即可获得真正的空间几何理解能力。

    

    本文提出了一种赋予大型语言模型空间理解能力的新范式和新方法。标准语言模型仅在离散词语上运行（词语是事物的二次表示，而非连续空间本身），迫使它们只能通过语言来猜测几何信息。大多数多模态方法将空间接地等同于相机图像或视频帧，但空间结构本身并不是一幅图像：它由坐标框架、实体边界以及物体在连续空间中如何运动和相互关联所构成。基于这一区别，我们将结合位姿（位置、方向和尺度）与带符号距离形状嵌入的连续物体token，与词语token一同放置在单一不间断的transformer序列中，使语言和几何能够在物体场景逐步构建和更新的过程中相互关注。控制token可在自然语言词语……之间自动切换输出。

    arXiv:2611.33841v1 Announce Type: cross  Abstract: This paper proposes a new paradigm and method for giving large language models spatial understanding. Standard language models operate exclusively over discrete words (secondary representations of things rather than continuous space itself), forcing them to guess geometry verbally. Whereas most multimodal approaches equate spatial grounding with camera images or video frames, spatial structure itself is not a picture: it consists of coordinate frames, solid boundaries, and how objects move and relate to each other in continuous space. Motivated by this distinction, we place continuous object tokens combining pose (position, orientation, and scale) and signed-distance shape embeddings alongside word tokens inside a single uninterrupted transformer sequence, allowing language and geometry to attend to each other as object scenes are built and updated step by step. Control tokens switch output automatically between natural-language words 
    
[^68]: 教会多语言翻译模型区分地区语言变体：基于MADLAD-400-3B-MT实现卡斯蒂利亚西班牙语与拉丁美洲西班牙语、阿斯图里亚斯语、阿拉贡语及阿兰语的翻译

    Teaching a Multilingual Translation Model Regional Varieties: Castilian and Latin American Spanish, Asturian, Aragonese and Aranese with MADLAD-400-3B-MT

    [https://arxiv.org/abs/2611.33831](https://arxiv.org/abs/2611.33831)

    本研究通过LoRA微调、回译和翻译重放技术，成功使MADLAD-400-3B-MT多语言翻译模型能够区分卡斯蒂利亚西班牙语与拉丁美洲西班牙语两种地区变体，并支持阿斯图里亚斯语、阿拉贡语和阿兰语三种伊比利亚半岛小语种的翻译。

    

    大规模多语言翻译模型通常每种语言只有一个标签，无法控制地区语言变体，且对某些地区的较小语言覆盖不足或完全没有覆盖。我们报告了对MADLAD-400-3B-MT的三次微调，每次均采用低秩适应（LoRA）添加目标语言标签、对单语文本进行回译，以及重放基础模型自身的翻译：用于西班牙境内西班牙语的<2es_ES>、用于拉丁美洲西班牙语的<2es_419>，以及一个针对伊比利亚半岛三种罗曼语族语言的模型——阿斯图里亚斯语（<2ast>）、阿拉贡语（<2an>）和阿兰语（<2oc_aranes>）。对于西班牙语变体，我们使用一个领域监督的西班牙对拉丁美洲分类器对MADLAD-400西班牙语数据分割中的2.509亿篇文档进行标注，并通过一条规则筛选变体文本，该规则确保在340篇文档的参考集中排除所有拉丁美洲文档。卡斯蒂利亚模型在针对性测试中97%的情况下选择了西班牙的日常词汇（基础模型仅为62%），vos

    arXiv:2611.33831v1 Announce Type: new  Abstract: Massively multilingual translation models usually have one tag per language and no control over regional varieties, and they cover a region's smaller languages poorly or not at all. We report three fine-tunes of MADLAD-400-3B-MT, each adding target tags with low-rank adaptation (LoRA), back-translation of monolingual text and replay of the base model's own translations: <2es_ES> for Spanish as written in Spain, <2es_419> for Latin American Spanish, and one model for three Romance languages of the Iberian Peninsula, Asturian (<2ast>), Aragonese (<2an>) and Aranese (<2oc_aranes>). For the Spanish varieties we annotate the 250.9M documents of MADLAD-400's Spanish split with a domain-supervised Spain vs. Latin America classifier and select variety text with a rule that admitted no Latin American document in a 340-document reference set. The Castilian model chooses Spain's everyday vocabulary in 97% of a targeted test (base model 62%) and vos
    
[^69]: 掩码扩散语言模型中“依赖就绪”解码的陷阱

    Pitfalls of Dependency-Ready Decoding for Masked Diffusion Language Models

    [https://arxiv.org/abs/2611.33823](https://arxiv.org/abs/2611.33823)

    本研究揭示了掩码扩散模型中“依赖就绪”解码策略的局限性：当该策略优于基于置信度或未来效用的基线方法时，其解码顺序实际上会退化成从左到右的顺序解码。

    

    掩码扩散模型（MDMs）在数独、行程规划和Countdown等结构化推理任务上展现出强大的性能。MDM引入了token去掩码顺序上的自由度，为了利用这种自由度获得更好的准确率，早期的解码策略依赖于置信度信号或解码某个token所产生的下游影响。然而，简单的例子似乎表明，在去掩码某个token之前先解决其依赖关系应当是很重要的。因此，我们从实践和理论两个角度检验了这一原则（我们称之为“依赖就绪解码”）是否有助于提升准确率。我们研究了一个通过注意力机制实现依赖就绪的近期解码器，总体上发现：当它比基于置信度或未来效用的基线表现更好时，其解码顺序会坍缩为从左到右的顺序解码。在理论分析方面，我们建立了一个……的定义（摘要在此处截断）

    arXiv:2611.33823v1 Announce Type: cross  Abstract: Masked diffusion models (MDMs) have shown strong performance on structured reasoning tasks such as Sudoku, Trip Planning, and Countdown. MDMs introduce a freedom in the order in which tokens may be unmasked, and to exercise this freedom for good accuracy, early decoding strategies relied on signals of confidence or the downstream impact of decoding a token. However, simple examples seem to illustrate that resolving a token's dependencies before unmasking it should be important. Therefore, we examine whether this principle, which we call dependency-ready decoding, would contribute to better accuracy, from a practical and a theoretical perspective. We examine a recent decoder that implements dependency-readiness through attention, and generally find that when it performs better than confidence-based or future utility baselines, its decoding order collapses to left-to-right decoding. For theoretical analysis, we establish a definition of 
    
[^70]: 视觉语言模型在日语环境下的多模态安全评估

    Multimodal Safety Evaluation of Vision-Language Models in Japanese

    [https://arxiv.org/abs/2611.33789](https://arxiv.org/abs/2611.33789)

    该论文将多模态安全测试套件MSTS扩展至日语（第12种语言）并构建了包含82张日本文化本地化图像的评估资源，同时提出区分图像识别与安全判断的五分制评估框架，据此对五个日语视觉语言模型进行安全评估，发现各模型安全就绪度差异显著，且提供参考答案可提升LLM评判者的评估效果。

    

    视觉语言模型（VLM）日益被部署于日语应用场景中，然而多模态安全评估此前大多以英语为中心。我们将Röttger等人（2025）提出的多模态安全测试套件（MSTS）扩展至日语，使其成为该套件的第12种语言，贡献了320个经机器翻译并人工校对的测试案例（附参考答案），以及82张融入日本文化本地化特色的图像。我们提出了一个评估框架：（i）将MSTS原有的三类安全分组扩展为五分制评分，该评分不仅衡量回答是否安全，还会奖励回答的有用性；（ii）将图像识别与安全判断相分离，以便隔离识别错误。利用该框架，我们结合人工评估对五个现代日语视觉语言模型（Gemini-3-Flash、GPT-5、InternVL3、LLM-JP-4-VL、Qwen3-VL）进行了评测，发现各模型的安全就绪度存在显著差异。最后，我们评估了两个“以大语言模型作为评判者”（LLM-as-a-Judge）的模型，发现提供参考答案能够提升其评判效果。

    arXiv:2611.33789v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) are increasingly deployed in Japanese-language settings, yet multimodal safety evaluation has been largely English-centric. We extend the Multimodal Safety Test Suite (MSTS) of R\"ottger et al. (2025) to Japanese as the 12th language, contributing 320 machine-translated and post-edited cases with reference answers plus 82 culturally localized Japanese images. We introduce an evaluation framework that (i) extends MSTS's 3-class safety grouping with a 5-point scale that also rewards useful (not just safe) responses, and (ii) separates image recognition from safety judgment to isolate recognition errors. Using this framework, we evaluate five modern Japanese VLMs (Gemini-3-Flash, GPT-5, InternVL3, LLM-JP-4-VL, Qwen3-VL) against human evaluation, and find substantial variance in the safety readiness across the models. Finally, we evaluate two LLM-as-a-Judge models and find that providing a reference answer impro
    
[^71]: 说谎机器的破绽：分析大语言模型在中国狼人杀游戏中生成的策略性对话

    Tells of a Lying Machine: Analyzing LLM-Generated Strategic Dialogue in Chinese Werewolf

    [https://arxiv.org/abs/2611.33747](https://arxiv.org/abs/2611.33747)

    该研究通过分析12个大语言模型在中国狼人杀游戏中的5,404条发言，发现狼人角色在确定性与不确定性语言标记上均显著高于村民，且人类标注者和LLM评审都能据此以60%至70%的准确率识别狼人，揭示了AI策略性欺骗行为的可检测语言特征。

    

    社交推演游戏赋予模型相互对立的目标，但同时也让不同角色拥有不同的信息和发言义务。我们分析了由12个模型对局的150局中国狼人杀游戏中的5,404条公开发言，通过只比较普通狼人与普通村民来排除特殊角色的影响。即使排除了可验证真伪的发言，私密知识与目标依然存在差异。经过模型和文本长度调整后，狼人发言在确定性（certainty）和不确定性（uncertainty）两类语言标记上得分均更高（系数分别为+0.074和+0.072）。情感强化的秩双列相关效应最大（r = .145），但其依赖单一程度副词，且与人类评定的情感强度仅呈弱相关。三位标注者对392条发言的评分显示，狼人发言在四个行为维度上得分更高，五个大语言模型评审在共享评估集上达到了60.8%至70.2%的准确率。在一个生成探针实验中，以狼人身份但以村民发言方式提示六个模型（原文摘要至此被截断）。

    arXiv:2611.33747v1 Announce Type: new  Abstract: Social-deduction games give models opposing objectives, but they also give roles different information and speaking obligations. We analyze 5,404 public utterances from 150 Chinese Werewolf games played by 12 models, comparing ordinary wolves with ordinary villagers so that special roles are excluded. Private knowledge and goals still differ, and no utterance is verified as true or false. After model and length adjustment, wolf utterances score higher on both certainty and uncertainty (coefficients +0.074 and +0.072). Emotional intensification has the largest rank-biserial effect (r = .145), but it depends on a single degree marker and correlates weakly with human-rated affect. Three annotators' ratings of 392 utterances show higher wolf scores on four behavioral dimensions, and five LLM judges reach 60.8%-70.2% accuracy on a shared evaluation set. In a generation probe that prompts six models with a wolf identity but a villager speaking
    
[^72]: NORMCRAFT：面向情境感知语言模型对齐的文化规范生成

    NORMCRAFT: Cultural Norm Generation for Context-Aware Language Model Alignment

    [https://arxiv.org/abs/2611.33726](https://arxiv.org/abs/2611.33726)

    NORMCRAFT提出了一种将规范推断与响应生成解耦的模块化框架，通过多视角监督训练LLM作为文化规范生成器，无需下游响应标签即可实现更强的文化规范感知对齐。

    

    文化对齐要求大型语言模型（LLM）能够从场景及其文化背景中推断出往往是隐含的社会规范。现有方法通常是适配单个模型或提供外部文化知识，而独立于下游响应生成来学习规范推断这一方向仍鲜有探索。我们提出了NORMCRAFT，一个将规范推断与响应生成解耦的模块化框架。通过以规范为中心的多视角监督，我们训练一个LLM作为规范生成器，从预期、遵守和违规三类场景中恢复共享规范，而无需下游响应标签。生成的规范用于指导一个独立的、冻结的LLM（即响应器）。以Qwen3.5-4B作为响应器，NORMCRAFT在五个下游指标中的四个上优于提示工程和监督微调（SFT）基线。在SafeWorld数据集上，其响应更频繁（覆盖率/Coverage）且更精确（忠实度/Faithfulness）地反映参考规范……

    arXiv:2611.33726v1 Announce Type: new  Abstract: Cultural alignment requires large language models (LLMs) to infer often-implicit social norms from a scenario and its cultural context. Existing approaches typically adapt individual models or supply external cultural knowledge, while learning norm inference independently of downstream response generation remains underexplored. We introduce NORMCRAFT, a modular framework that decouples norm inference from response generation. Through norm-centric multi-view supervision, we train an LLM as a norm generator to recover shared norms from prospective, adherence, and violation scenarios without downstream response labels. The generated norms guide a separate, frozen LLM (the responder). With Qwen3.5-4B as the responder, NORMCRAFT outperforms prompting and supervised fine-tuning (SFT) baselines on four of five downstream metrics. On SafeWorld, its responses reflect the reference norms more often (Coverage) and more precisely (Faithfulness), exc
    
[^73]: Sports Agent：一个面向对话式数据分析的自进化系统

    Sports Agent: A Self-Evolving System for Conversational Data Analytics

    [https://arxiv.org/abs/2611.33700](https://arxiv.org/abs/2611.33700)

    Sports Agent 提出了一个由运行时反馈驱动的自进化框架，通过统一的反馈引擎结合 SQL-to-Question Scaling 与 Graph-Guided Scaling 等策略，使对话式体育数据分析智能体能够持续自我改进并保持回答的准确性与可靠性。

    

    对话式体育分析需要准确、有据可依且低延迟的响应。查询分布、数据资源和模型行为的变化会导致静态智能体配置的性能退化，这促使我们探索由运行时证据引导的受控进化。我们提出 Sports Agent，一个自进化框架，它集成了用于结构化数据推理的智能体 SQL 模型、用于规划与工具编排的执行框架，以及连接在线执行与离线进化的统一反馈引擎。该引擎评估响应与执行轨迹、验证事实性断言、诊断失败原因并生成进化信号。模型进化结合了用于覆盖面扩展的 SQL-to-Question Scaling 与面向错误针对性合成的 Graph-Guided Scaling。生成的问题经过轨迹蒸馏与质量过滤，用于迭代式监督微调，随后进行独立的 Post-RL 阶段。Harness 进化将……（原文摘要在此处截断）

    arXiv:2611.33700v1 Announce Type: cross  Abstract: Conversational sports analytics requires accurate, evidence-based responses with low latency. Changes in query distributions, data resources, and model behavior can degrade static agent configurations, motivating controlled evolution guided by runtime evidence.   We present Sports Agent, a self-evolving framework integrating an Agentic SQL Model for structured-data reasoning, a harness for planning and tool orchestration, and a unified Feedback Engine connecting online execution with offline evolution. The engine assesses responses and execution trajectories, verifies factual claims, diagnoses failures, and generates evolution signals. Model evolution combines SQL-to-Question Scaling for coverage expansion with Graph-Guided Scaling for error-targeted synthesis. Generated questions undergo trajectory distillation and quality filtering for iterative supervised fine-tuning, followed by a separate Post-RL stage. Harness evolution translate
    
[^74]: 重新思考顺序式定位后编辑：最优性与稳定性

    Rethinking Sequential Locate-Then-Edit: Optimality and Stability

    [https://arxiv.org/abs/2611.33692](https://arxiv.org/abs/2611.33692)

    本文提出一种与批量编辑等价的顺序编辑方法，成功弥合了顺序编辑与批量编辑之间的性能差距，解决了现有顺序方法在数百次编辑后出现的灾难性遗忘和模型崩溃问题。

    

    定位后编辑框架能够在无需完整重新训练的情况下，实现对大语言模型的高效知识更新。近期研究日益聚焦于顺序编辑，即知识持续到来、模型被逐步增量更新。然而，现有的顺序编辑方法通常在仅进行几百次编辑后就会出现灾难性遗忘和模型崩溃。相比之下，批量编辑——即联合优化所有编辑并在指定编辑目标下取得最优解——可以在保持模型通用能力的同时，支持数万次编辑并维持较高的编辑效能，这导致两种编辑范式之间存在显著的性能差距。在本文中，我们证明了这一差距可以通过一种与批量编辑等价的顺序编辑方法来弥合。此外，我们还通过分析揭示了现有顺序方法中模型退化的内在机制。

    arXiv:2611.33692v1 Announce Type: cross  Abstract: The Locate-Then-Edit framework enables efficient knowledge updates in large language models without requiring full retraining. Recent research has increasingly focused on sequential editing, where knowledge arrives continuously and the model is updated incrementally. However, existing sequential methods commonly suffer from catastrophic forgetting and model collapse after only a few hundred edits. In contrast, batch editing - which jointly optimizes all edits and yields the optimal solution under a specified editing objective - can support tens of thousands of edits with high edit efficacy while preserving the model's general capabilities, resulting in a substantial performance gap between the two editing paradigms. In this paper, we demonstrate that this gap can be closed via a batch-equivalent sequential editing approach. Moreover, we investigate the underlying mechanism of model degradation in existing sequential methods by analyzin
    
[^75]: 拒答不计分：基于内容的偏见评估中的覆盖盲区

    Refusal Is Not Scored: A Coverage Limit in Content-Based Bias Evaluation

    [https://arxiv.org/abs/2611.33690](https://arxiv.org/abs/2611.33690)

    本文揭示了偏见审计中“先过滤拒答再评分”这一默认做法造成的测量盲区：拒答集中来自单一模型且针对特定国籍提示的频率显著更高，而部分评估工具会把拒答误判为无偏见内容，从而系统性低估模型偏见。

    

    对对话式模型的偏见审计通常在评分之前丢弃拒答（如“我无法协助该请求”之类的回复），这一过滤步骤很少被当作一项测量决策来审视，尽管它决定了审计能够评估哪些回复。本工作检验了该步骤究竟剔除了什么，将涵盖两种现有范式的七种偏见评估工具——三种基于词典的工具和四种以大语言模型为裁判的工具——应用于两个各包含360条生成回复的交叉性数据集。在两个数据集中，该过滤器剔除的都是一个在人口统计上偏斜的子集，且完全由三个响应模型中的一个产生：该模型对朝鲜相关提示的拒答率在陈述性内容上是美国相关提示的2.15倍，在操作性内容上是1.76倍。随后我们移除过滤器，将每一条拒答都送入全部七种工具进行评分。Bias-Lexicon将拒答评为“无偏见”，而纳入拒答后，它所报告的国籍差异随之缩小。Hu…（原文摘要在此处截断）

    arXiv:2611.33690v1 Announce Type: new  Abstract: Bias audits of conversational models often discard refusals (responses such as "I can't assist with that request") before scoring, a filtering step which is rarely examined as a measurement decision even though it determines which responses an audit is capable of evaluating. This work examines what the step removes, applying seven bias-evaluation instruments spanning both deployed paradigms, three dictionary-based and four LLM-as-judge, to two intersectional datasets of 360 generated responses each. In both datasets, the filter removed a demographically skewed subset produced entirely by one of the three responder models, which refused North Korean prompts at 2.15 times the rate of American prompts on declarative content and 1.76 times on operational content. We then remove the filter and pass every refusal to all seven instruments. Bias-Lexicon scores refusals as clean, and including them shrinks the nationality disparity it reports. Hu
    
[^76]: 重新思考扩散语言模型的知识蒸馏

    Rethinking Knowledge Distillation for Diffusion Language Models

    [https://arxiv.org/abs/2611.33670](https://arxiv.org/abs/2611.33670)

    该论文首次系统性研究了扩散语言模型的知识蒸馏，揭示了前向KL散度蒸馏的两个反直觉失败现象，并提出了一种结合离策略与在策略蒸馏的统一无数据蒸馏框架，有效提升掩码扩散语言模型的性能。

    

    知识蒸馏最近被证明对于改进自回归模型非常有效，但其在扩散语言模型（DLM）中的适用性仍知之甚少。我们首次对扩散语言模型的知识蒸馏进行了系统性研究，并揭示了经典前向KL散度（FKL）蒸馏的两个反直觉的失败现象：(i) 学生的困惑度并不会随着教师生成数据的增加而单调改善；(ii) 在相同的教师生成语料库上，FKL蒸馏的表现甚至不如直接使用原始DLM去噪目标来训练学生模型。为了解决这些问题，我们提出了一个统一的、无数据的掩码扩散语言模型蒸馏框架。首先，该框架运行一个离策略蒸馏阶段，从静态的教师生成数据集中吸收通用知识。然后，该框架采用在策略蒸馏，其中包含三种技术：沿学生自身逆过程的轨迹损失、对（摘要在此处截断）……

    arXiv:2611.33670v1 Announce Type: new  Abstract: Knowledge distillation has recently proven highly effective for improving autoregressive models, but its applicability to DLMs remains poorly understood. We present the first systematic study of KD for DLMs and uncover two non-intuitive failures of classical forward-KL (FKL) distillation: (i) student perplexity does not improve monotonically with more teacher-generated data, and (ii) on the same teacher-generated corpus, FKL distillation underperforms simply training the student with the original DLM denoising objective. To address them, we propose a unified, data-free distillation framework for masked DLMs. First, the framework runs an off-policy distillation phase that absorbs general knowledge from a static set of teacher-generated data. Then, the framework employs on-policy distillation which consists of three techniques: a trajectory loss along the student's own reverse process, confidence reweighting that down-weights tokens where 
    
[^77]: 可验证但不可预测：句子嵌入对命题分解究竟了解什么

    Verifiable but Not Predictable: What Sentence Embeddings Know About Propositional Decomposition

    [https://arxiv.org/abs/2611.33668](https://arxiv.org/abs/2611.33668)

    本文以PropSegmEnt语料库为基础，发现命题分解的两个质量控制问题处于不同表示空间：分解后可用NLI评判器高精度验证命题归属（准确率92.8%、AUC达0.977），而分解前无法通过句子嵌入的向量几何预测句子应产出的命题数量。

    

    arXiv:2611.33668v1 公告类型：新论文 摘要：大语言模型（LLM）命题分解的质量控制提出了两个不对称的问题。分解之后，可以进行验证：给定一个句子和一个候选命题，该命题是否属于该句子？分解之前，人们希望进行预测：这个句子应该产生多少个命题？在PropSegmEnt（一个包含英文维基百科句子、每句均由三位人类标注者进行分割的语料库）上，我们发现这两个问题处于不同的表示空间中。自然语言推理（NLI）评判器可以92.8%的准确率和0.977的ROC曲线下面积（AUC）验证命题的隶属关系，而且其残余错误大多属于在文档上下文中确实为真的命题，而非真正的蕴含错误。向量几何无法完成验证：余弦相似度会达到饱和，句子嵌入也并不是其各命题嵌入的线性组合。而在预测方面，每一个符号……（摘要原文在此处被截断）

    arXiv:2611.33668v1 Announce Type: new  Abstract: Quality control for proposition decomposition by large language models (LLMs) raises two asymmetric questions. After decomposition, one can verify: given a sentence and a candidate proposition, does the proposition belong to that sentence? Before decomposition, one would like to predict: how many propositions should the sentence yield? On PropSegmEnt, a corpus of English Wikipedia sentences each segmented by three human annotators, we find that the two questions live in different representational spaces. A natural language inference (NLI) judge verifies membership with 92.8% accuracy and an area under the ROC curve (AUC) of 0.977, and its residual errors turn out to be mostly propositions that are true in document context rather than genuine entailment mistakes. Vector geometry cannot verify: cosine similarity saturates, and a sentence embedding is not a linear combination of its propositions' embeddings. For prediction, every symbolic a
    
[^78]: 面向跨层分析与引导的稀疏到稀疏特征学习

    Sparse-to-Sparse Feature Learning for Cross-Layer Analysis and Steering

    [https://arxiv.org/abs/2611.33640](https://arxiv.org/abs/2611.33640)

    提出稀疏到稀疏特征学习（S2S）方法，通过训练各层稀疏特征预测下一层的稀疏特征，在八个语言模型上实现了比现有基线更强的跨层特征对应关系，并能在控制输出分布偏移的条件下有效分析和引导模型行为。

    

    理解模型如何在内部表征信息是解释与控制领域的核心挑战。稀疏自编码器（SAEs）将稠密的隐藏表征分解为稀疏特征，而跨层方法能够捕获跨越多个层的结构。然而，现有的训练目标并未显式地训练某一层的稀疏编码去预测下一层的稀疏编码。我们提出稀疏到稀疏特征学习（S2S），其中每个中间层的稀疏特征被训练用于预测紧随其后的下一层的稀疏特征，最后一层由输出分布损失进行锚定，优化过程从输出到输入依次进行。在八个语言模型上的实验表明，与匹配的稀疏特征基线相比，该方法展现出更强的相邻层对应关系。我们在匹配的输出分布偏移条件下评估基于特征的干预，从而将引导引发的效果与稀疏通路的重构效应分离开来。

    arXiv:2611.33640v1 Announce Type: new  Abstract: Understanding how models internally represent information is a central challenge in interpretation and control. Sparse autoencoders (SAEs) decompose dense hidden representations into sparse features, and cross-layer methods capture structure spanning multiple layers. However, existing objectives do not explicitly train the sparse code at one layer to predict the sparse code at the next layer. We propose sparse-to-sparse feature learning (S2S), where sparse features at each intermediate layer are trained to predict those of the immediately following layer, with the final layer anchored by an output-distribution loss and optimization proceeding sequentially from output to input. Experiments across eight language models show stronger adjacent-layer correspondence than matched sparse-feature baselines. We evaluate feature-based interventions at matched output-distribution shift, separating steering-induced effects from sparse-pathway reconst
    
[^79]: 状态空间注意力：学习上下文相关的记忆保留

    State Space Attention: Learning Context-Dependent Memory Retention

    [https://arxiv.org/abs/2611.33625](https://arxiv.org/abs/2611.33625)

    提出状态空间注意力（SSA），通过可学习的评分器自适应地决定保留哪些历史表示以维持有界内存，并配合“恢复”式监督样本训练技术，使模型能够依据上下文动态选择长上下文推理所需的记忆内容。

    

    标准注意力的键值缓存会随上下文长度线性增长。若要对其加以限制，便会引出一个问题：序列模型在长上下文推理中应当保留什么。我们认为这一选择应当是可学习的，并据此提出了状态空间注意力（SSA），它学习保留一个有界的过去表示集合，并通过标准注意力来读取这些表示。一个学习得到的评分器会依据新到达的上下文以及记忆中的其余内容，对每个块进行重新评分。固定上下文的Transformer由此成为一种特例——即始终保留最近token的情况。我们联合训练评分器与读取器，或者从可微的软记忆加权逐渐过渡到硬性保留，或者从一开始就采用硬性保留。为了给评分器产生训练信号，我们在训练过程中生成监督样本，我们将这一技术称为“恢复”。我们在检索等任务上将SSA与循环模型、有界内存注意力方法以及全注意力Transformer进行了比较。

    arXiv:2611.33625v1 Announce Type: cross  Abstract: Standard attention's key-value cache grows linearly with context length. Bounding it raises the question of what a sequence model should retain for long-context reasoning. We argue this choice should be learned, and introduce State Space Attention (SSA), which learns to keep a bounded set of past representations and reads them with standard attention. A learned scorer rescores every chunk against newly arriving context and the rest of memory. A fixed-context Transformer is then the special case that always keeps the most recent tokens. We train the scorer and reader jointly, either moving gradually from a differentiable soft weighting of memories to hard retention, or with hard retention from the start. To produce signal for the scorer, we generate supervised samples during training, a technique we call recovery. We compare SSA with recurrent models, bounded-memory attention methods, and full-attention Transformers across retrieval, va
    
[^80]: 失败拓宽，成功精进：基于双极性记忆的工具使用智能体在多样化任务上的持续适应

    Failure Broadens, Success Sharpens: Continual Adaptation of Tool-Use Agents Across Diverse Tasks with Dual-Polarity Memory

    [https://arxiv.org/abs/2611.33612](https://arxiv.org/abs/2611.33612)

    本文提出带周期性维护的双极性记忆框架DualPM，同时从成功与失败轨迹中生成程序性记忆并周期性整合以防记忆爆炸，实验发现失败记忆更能拓宽任务覆盖面、成功记忆更能精简执行过程。

    

    大语言模型智能体可以通过利用从先前任务轨迹中提取的程序性记忆随时间自我改进，然而从局部成功和局部失败中获得的记忆所具有的不同效用仍缺乏充分研究。本工作在一个现实环境中研究这一问题，该环境包含多领域工具使用任务的异构任务流，以及现实场景中常见的各类环境约束。我们开发了带周期性维护的双极性记忆（DualPM），这是一种简单、无需检索的记忆工作流，其特点包括：(1) 可定制的反思机制，能从每条轨迹中的局部成功和失败中生成程序性记忆；(2) 在给定记忆容量下对长任务流进行周期性记忆整合，防止记忆爆炸。采用细粒度指标的在线顺序评估表明，失败记忆更能有效拓宽任务覆盖范围，成功记忆则更能精简执行过程，而……（原文摘要在此处截断）

    arXiv:2611.33612v1 Announce Type: new  Abstract: Large language model agents can self-improve over time by utilizing procedural memory derived from prior task trajectories, yet the distinct utilities of memories derived from local successes and failures remain underexplored. This work studies this question in a realistic setting with heterogeneous streams of multi-domain tool-use tasks and various environment constraints common in real-world scenarios. We develop \emph{Dual-Polarity Memory with Periodic Maintenance} (DualPM), a simple, retrieval-free memory workflow characterized by (1) customizable reflection and procedural memory generation from both local successes and failures within each trajectory, and (2) periodic memory consolidation over long task streams under a given memory capacity, preventing memory explosion. Online prequential evaluation with fine-grained metrics reveals that failure memory more strongly broadens coverage, success memory better streamlines execution, whi
    
[^81]: 对齐机器：基于人类比较判断的多模态嵌入相似性评估基准

    Aligned Machine: A Benchmark for Evaluating Multimodal Embedding Similarity via Human Comparative Judgements

    [https://arxiv.org/abs/2611.33594](https://arxiv.org/abs/2611.33594)

    该论文提出“对齐机器”这一交互式公众平台，通过让大众完成两两比较的强制选择任务，构建了可持续增长的人类对齐相似性基准（HASB），以评估嵌入模型的语义对齐能力。

    

    arXiv:2611.33594v1 公告类型：新论文。摘要：嵌入模型日益影响着人们搜索、推荐和检索信息的方式，然而我们缺乏可靠的方法来检验这些相似性判断是否反映了人类对意义的理解。现有基准通常依赖于静态的、由专家标注的配对数据，并采用粗糙的李克特量表式标注，这限制了可扩展性，也难以获得一致且高置信度的标注。我们提出了“对齐机器”，这是一个交互式、面向公众的平台，它将广泛的公众参与转化为一个持续增长的“人类对齐相似性基准”（HASB），用于评估基于嵌入的系统的语义对齐能力。参与者完成一个认知上自然的强制选择任务：比较两对条目（文本或图像），并选出更相似的一对。这种比较式设计对非专家用户十分友好，能产生干净的偏好数据，并支持稳健的聚合分析。我们报告了首次部署的结果……

    arXiv:2611.33594v1 Announce Type: new  Abstract: Embedding models increasingly inform how people search, recommend, and retrieve information, yet we lack credible ways to test whether these similarity judgments reflect how humans understand meaning. Existing benchmarks typically rely on static, expert-labeled pairs with coarse Likert-style annotations, limiting scalability and making it difficult to obtain consistent, high-confidence annotations. We introduce Aligned Machine, an interactive, public-facing platform that transforms broad participation into a continuously growing Human-Aligned Similarity Benchmark (HASB) for evaluating semantic alignment in embedding-based systems. Participants complete a cognitively-natural, forced-choice task, comparing two pairs of items (text or images) and selecting which pair is more similar. This comparative design is accessible to non-experts, yields clean preference data, and supports robust aggregation. We report results from an initial deployme
    
[^82]: SEEL：基于语言模型的端到端3D手语生成

    SEEL: Towards End-to-End 3D Sign Generation with Language Models

    [https://arxiv.org/abs/2611.33588](https://arxiv.org/abs/2611.33588)

    提出SEEL端到端框架，通过重建感知生成等机制将动作分词与语言模型生成过程和3D动作质量对齐，解决了以往两阶段流水线目标不匹配及直通估计器梯度偏差、码本崩溃的问题，实现更高质量的3D手语动作生成。

    

    3D手语生成旨在从书面文本生成手语动作。受手语的语言学特性启发，最近的方法将动作分词器与语言模型相结合，从文本预测动作标记。然而，这两个组件通过两阶段流水线进行优化，目标相互不匹配：分词器单独最小化重建误差，而生成器从刚性的独热参考标记中学习。一种直接的端到端扩展方式通过直通估计器（STE）来桥接离散标记选择，但会遭受梯度偏差和码本崩溃的问题。为了解决这些局限性，我们提出了SEEL，一个端到端框架，通过两个互补机制将分词与生成同动作质量对齐。首先，为了克服刚性独热监督，重建感知生成在上下文中解码候选标记，并将其3D动作误差转换为软目标……

    arXiv:2611.33588v1 Announce Type: cross  Abstract: 3D sign language generation aims to produce signing motion from written text. Motivated by the linguistic nature of sign language, recent methods combine a motion tokenizer with a language model to predict motion tokens from text. However, both components are optimized through a two-stage pipeline with mismatched objectives: the tokenizer minimizes reconstruction error in isolation, while the generator learns from rigid one-hot reference tokens. A straightforward end-to-end extension bridges discrete token selection with straight-through estimators (STE), suffering from biased gradients and codebook collapse. To address these limitations, we propose SEEL, an end-to-end framework that aligns tokenization and generation with motion quality through two complementary mechanisms. First, to overcome rigid one-hot supervision, reconstruction-aware generation decodes candidate tokens in context and converts their 3D motion errors into soft tar
    
[^83]: PeerReveal：学习在AI同行评审检测中的泛化能力

    PeerReveal: Learning to Generalize in AI Peer Review Detection

    [https://arxiv.org/abs/2611.33555](https://arxiv.org/abs/2611.33555)

    该论文提出了PeerReveal检测器，通过学习评审声明与同一论文AI生成评审的参考声明之间的关系来检测AI生成的同行评审，并在覆盖多个学科、融合多样化AI生成机制的PeerShift新基准上显著超越最强基线。

    

    arXiv:2611.33555v1 公告类型：新论文 摘要：我们能否区分完全由AI生成的评审与仅使用AI改写人类判断的评审？此外，当评审来自跨越不同科学领域、多样化AI模型和工作流程时，我们能否可靠地做到这一点？为此，我们推出了PeerReveal，一种新型检测器，其决策基于学习到的评审声明与同一论文的AI生成评审中参考声明之间的关系，并结合评审层面的文本信号。我们在PeerShift上评估PeerReveal，这是一个新的、具有挑战性的测试集，包含5,430篇跨越多个领域（包括机器学习、生命科学和社会科学）的评审，结合了学科评审惯例的差异与多样化的AI评审生成机制。通过常规训练，PeerReveal在该基准上实现了70.52%的合并Macro-F1，较最强基线提升了11.48个百分点。我们还开发了一种有效且实用的对抗性训练框架。

    arXiv:2611.33555v1 Announce Type: new  Abstract: Can we distinguish fully AI-generated reviews from those using AI only to rewrite human judgement? Furthermore, can we do so reliably when reviews are generated from diverse AI models and workflows across scientific domains? To this end, we introduce PeerReveal, a new detector that grounds its decisions in learned relationships between a review's claims and reference claims drawn from AI-generated reviews of the same paper, complemented by review-level textual signals. We evaluate PeerReveal on PeerShift, a new, challenging test set of 5,430 reviews spanning multiple domains, including machine learning, life sciences, and social sciences, combining variation in disciplinary review conventions with diverse AI review-production mechanisms. With regular training, PeerReveal achieves 70.52% pooled Macro-F1 on this benchmark, improving over the strongest baseline by 11.48 points. We develop an effective and practical adversarial training fram
    
[^84]: LogToPersona：新基准与基于语义日志过滤的高效人格预测

    LogToPersona: New Benchmark and Efficient Persona Prediction through Semantic Log Filtering

    [https://arxiv.org/abs/2611.33510](https://arxiv.org/abs/2611.33510)

    本文提出了LogToPersona任务及包含1600个人格-日志对的合成基准，通过语义日志过滤方法在端侧设备上实现高效的用户人格属性预测，内存占用最多降低39%且预填充时间减半以上。

    

    详细了解用户的人格画像对于个性化生成式AI模型和智能体非常有价值，然而这些细节难以获取且会随时间变化。我们研究从单日移动设备活动日志中推断人格属性的任务。目前没有合适的公开可用基准，因此我们引入了一个包含1,600个人格-日志对的合成基准。我们的目标是端侧（设备上）推理，使原始日志永远不会离开手机，这使得内存而非速度成为限制性约束：一天的活动会产生约30K个token，其键值缓存本身就超过了40亿参数模型的INT4权重大小。因此，我们提出了一种日志过滤方法，仅保留行为上具有区分性的条目。过滤使来自四个模型家族的九个模型的内存占用最多降低39%，并将预填充时间缩短一半以上，同时保持或提高预测质量。对于Qwen3-4B，它减少了2.5GB的内存占用。

    arXiv:2611.33510v1 Announce Type: new  Abstract: Detailed knowledge of a user's persona is valuable for personalizing generative AI models and agents, yet such details are difficult to obtain and change over time. We address the task of inferring persona attributes from a single day of mobile activity logs. No suitable publicly available benchmark exists for this task, so we introduce a synthetic benchmark of 1,600 persona-log pairs. We target on-device inference so that raw logs never leave the phone, which makes memory rather than speed the limiting constraint: a day of activity yields roughly 30K tokens, whose key-value cache alone exceeds the INT4 weights of a 4B-parameter model. We therefore propose a log filtering method that retains only behaviorally distinct entries. Filtering reduces the memory footprint by up to 39% across nine models from four families and more than halves prefill time, while maintaining or improving prediction quality. For Qwen3-4B it removes 2.5GB, bringin
    
[^85]: 用人工智能模拟欧洲人群的行为

    Using AI to Simulate the Behaviour of European Populations

    [https://arxiv.org/abs/2611.33503](https://arxiv.org/abs/2611.33503)

    该研究仅需在约40个欧洲国家公开的1200多万条群体级调查答案分布上微调开源语言模型，无需访谈或个人数据即可模拟欧洲人群的调查反应，判断群体差异方向的准确率达84%，超越传统人口统计模型并与Claude Opus等前沿大模型持平。

    

    语言模型可以替代调查受访者，但主流方法是通过访谈或个人记录来构建模拟的人。我们证明，一个模型可以仅从公开的统计数据中学习欧洲人群组的反应方式，无需访谈，也无需在个人记录上训练。我们在来自约40个欧洲国家的超过1200万条已发表的群体层面答案分布上微调了一个开源权重语言模型，使其能够预测每个群体相对于参考分布的偏离情况。在其训练所属调查家族的民意调查中，它在84%的情况下正确判断群体差异的方向，优于一个人口统计模型（77%，配对检验），并与Claude Opus 5和5.5持平，在准确率上也与后者打平；前沿模型在新问题上更准确。在278个留出问题上（几乎全部来自训练时见过的调查家族），其误差比该统计模型低12%。

    arXiv:2611.33503v1 Announce Type: new  Abstract: Language models can stand in for survey respondents, but leading approaches build their simulated people from interviews or individual records. We show that a model can learn how European population groups respond from published statistics, without interviews and without training on individual records. We fine-tune an open-weight language model on more than 12 million published group-level answer distributions from about 40 European countries, setting it up to predict each group's departure from a reference distribution. On opinion surveys from its training families it calls the direction of group differences right 84% of the time, ahead of a demographic statistical model (77%, paired test) and level with Claude Opus 5 and 5.5, which it also ties on accuracy; the frontier models are more accurate on new questions. On 278 held-out questions, almost all from the survey families seen in training, its error is 12% lower than that statistical
    
[^86]: 评估大型语言模型对会话式非裔美国人英语的处理能力：特征分类与下一轮生成

    Evaluating Large Language Models on conversational African American English: Feature classification and next-turn generation

    [https://arxiv.org/abs/2611.33499](https://arxiv.org/abs/2611.33499)

    本文基于CORAAL语料库中真实的会话式非裔美国人英语（AAE），通过特征分类和下一轮生成两项互补任务，评估了三种闭源大型语言模型在理解与生成AAE方面的能力。

    

    众所周知，大型语言模型（LLMs）会误解并错误呈现主流美国英语（MAE）之外的区域性和社会性英语变体，例如非裔美国人英语（AAE），然而大多数针对此类偏见的评估都依赖于合成的、风格化的或脱离上下文的数据。本文改为在真实、会话式的AAE上评估LLM，以更贴近这些模型作为聊天机器人部署时可能遇到AAE的实际方式。具体而言，我们在从区域非裔美国人语言语料库（CORAAL；Kendall and Farrington, 2024）中随机抽样的100个六轮对话片段上开展了两项互补的评估任务，比较了三种闭源权重LLM配置：开启推理与关闭推理的Gemini 2.5 Flash，以及GPT-4.1 mini。第一项任务将模型的表现与五类AAE形态句法特征的黄金标准人工标注进行对比，第二项任务则评估模型生成符合上下文的下一轮回复的能力。

    arXiv:2611.33499v1 Announce Type: new  Abstract: Large Language Models (LLMs) are known to misinterpret and misrepresent regional and social varieties of English other than Mainstream American English (MAE), such as African American English (AAE), yet most evaluations of these biases have relied on synthetic, stylized or decontextualized data. This paper instead evaluates LLMs on authentic, conversational AAE to more closely reflect how such models may encounter AAE when deployed as chatbots. Concretely, we perform two complementary evaluation tasks on 100 six-turn segments randomly sampled from the Corpus of Regional African American Language (CORAAL; Kendall and Farrington, 2024), comparing three closed-weight LLM configurations: Gemini 2.5 Flash with and without reasoning, and GPT-4.1 mini. The first task compares the models against gold-standard manual annotations for five AAE morphosyntactic features, while the second assesses their ability to generate contextually appropriate nex
    
[^87]: CA-RAG：面向结构忠实长文本生成的约束对齐检索

    CA-RAG: Constraint-Aligned Retrieval for Structurally Faithful Long-Form Generation

    [https://arxiv.org/abs/2611.33497](https://arxiv.org/abs/2611.33497)

    提出 CA-RAG 框架，在证据检索前利用本体结构约束和基于政策的知识图谱关系增强章节级查询，显著提升合规敏感长文本生成中证据检索的精确率、召回率和生成合规性。

    

    检索增强生成（RAG）通常针对语义相关性来优化证据选择，而对合规敏感的长文本生成还需要能够支持明确结构性义务的证据。因此，仅基于语义相关的检索可能遗漏下游强制性条款所需的信息。我们提出了约束对齐检索增强生成（CA-RAG），这是一种检索条件化框架，在证据选择之前，利用源自本体的结构约束和基于政策的知识图谱关系来增强针对特定章节的查询。在人工标注证据上的纯检索评估显示，该方法在各种检索预算下均取得一致的性能提升。在 k=6 时，与标准 RAG 相比，CA-RAG 将 Precision@k 从 0.5268 提升至 0.5692，将综合 Recall@k 从 0.5402 提升至 0.5950，将 F1@k 从 0.5389 提升至 0.5672。生成实验进一步表明，在多个模型-领域设置中，生成内容的合规性和可靠性均有提升。

    arXiv:2611.33497v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) typically optimises evidence selection for semantic relevance, whereas compliance-sensitive long-form generation also requires evidence supporting explicit structural obligations. Semantically relevant retrieval may therefore omit information needed for mandatory downstream clauses. We propose Constraint-Aligned RAG (CA-RAG), a retrieval-conditioning framework that augments section-specific queries with ontology-derived structural constraints and policy-grounded knowledge-graph relations before evidence selection. Retrieval-only evaluation with manually annotated evidence shows consistent improvements across retrieval budgets. At $k=6$, CA-RAG increases Precision@$k$ from $0.5268$ to $0.5692$, pooled Recall@$k$ from $0.5402$ to $0.5950$, and F1@$k$ from $0.5389$ to $0.5672$ over standard RAG. Generation experiments further show improvements in compliance and reliability in several model--domain settin
    
[^88]: 从循环中学习：在循环语言模型中内化深度诱导的偏好

    Learning from Recurrence: Internalizing Depth-Induced Preferences in Looped Language Models

    [https://arxiv.org/abs/2611.33468](https://arxiv.org/abs/2611.33468)

    本文提出“偏好回写”方法，利用循环语言模型在更大深度下自身生成的预测来训练共享块（无需参考答案），从而将深层计算带来的偏好内化到单遍推理中，同时提升单遍与多遍推理的准确率。

    

    arXiv:2611.33468v1 公告类型：新论文 摘要：循环语言模型在预测每个词元之前，会将同一个块重复应用于其隐藏状态，应用的次数称为其循环深度。增加循环深度可以在参数不变的情况下让模型为每个词元进行更多计算。共享块能否从它在更大深度下所做的预测中学习？遵循先前的工作，我们为预训练的 OLMo-2-1B 加装共享循环块，并在不同的循环深度下继续进行中期训练。在四遍模型中，额外的遍数主要是在一遍推理后已排名靠前的词元之间重新分配概率。我们利用在更大深度下生成的续写内容，仅通过单遍推理来训练共享块，且无需参考答案或正确性过滤。我们将这种转移方式称为“偏好回写”。偏好回写进一步提升了模型在大多数任务上的单遍和多遍准确率。在各自的目标深度下，更新后的模型优于参数量相当的非循环对照模型。

    arXiv:2611.33468v1 Announce Type: new  Abstract: A looped language model repeatedly applies the same block to its hidden state before predicting each token. The number of passes is its recurrent depth. Increasing this depth gives the model more computation per token with the same parameters. Can the shared block learn from the predictions it makes at greater depth? Following prior work, we retrofit pretrained OLMo-2-1B with a shared recurrent block and continue mid-training over varying recurrent depths. In the four-pass model, additional passes mainly redistribute probability among tokens that already rank highly after one pass. We use the continuations generated at greater depth to train only the shared block through a single pass, without reference answers or correctness filtering. We call this transfer preference write-back. Write-back further improves one-pass and multi-pass accuracy on most tasks. At their target depths, the updated models outperform a non-recurrent control match
    
[^89]: PAC学习采样：论语言生成的计算复杂性

    PAC Learning to Sample: On Computational Aspects of Language Generation

    [https://arxiv.org/abs/2611.33464](https://arxiv.org/abs/2611.33464)

    该论文将PAC学习框架与极限语言生成模型相结合，首次为语言生成任务提供了关于样本复杂度和时间复杂度的定量保证。

    

    语言模型的成功推动了从离散域分布中学习采样的理论研究。一个重要的抽象模型是由Kleinberg和Mullainathan（NeurIPS 2024）提出的“极限语言生成”模型，其中算法必须在经过有限但未指定数量的正例后从目标语言中进行生成。虽然该模型的积极结果出奇地具有普适性，但大多数保证都是定性的。在这项工作中，我们受统计学习理论和计算学习理论的启发，寻求关于样本复杂度和时间复杂度的定量保证。为此，我们研究了由De、Diakonikolas和Servedio（SODA 2015）在另一背景下提出的语言生成PAC模型，其中定义在{-1,1}^n上的布尔概念类C定义了目标语言：给定未知概念f ∈ C的均匀随机正例，学习者必须输出一个采样器，其输出……

    arXiv:2611.33464v1 Announce Type: cross  Abstract: The success of language models has motivated theoretical work on learning to sample from distributions on discrete domains. A prominent abstraction is the model of language generation in the limit, introduced by Kleinberg and Mullainathan (NeurIPS 2024), in which the algorithm must generate from the target language after some finite but unspecified number of positive examples. While its positive results are surprisingly general, most guarantees are qualitative. In this work, we seek quantitative guarantees on sample and time complexity, inspired by statistical and computational learning theory.   To this end, we study a PAC model of language generation, proposed (in a different context) by De, Diakonikolas, and Servedio (SODA 2015), in which a Boolean concept class $C$ over $\{-1,1\}^n$ defines the target languages: given uniformly random positive examples of an unknown concept $f \in C$, the learner must output a sampler whose output 
    
[^90]: AI-人类混合对话：AI智能体能否提升话语质量？

    AI-human hybrid conversations: Can AI Agents Improve Discourse Quality?

    [https://arxiv.org/abs/2611.33459](https://arxiv.org/abs/2611.33459)

    该论文提出了一个将通用型和人格设定型AI智能体引入动态讨论的框架，并通过基于Reddit的模拟实验和互补研究发现，AI智能体有望成为提升人机混合对话参与度与话语质量的催化剂，使对话更积极、毒性更低。

    

    对话是人们交换信息、分享经验并形成集体理解的核心方式。然而，在各种对话场景中，讨论往往存在参与不均衡、发言得不到回应以及观点范围狭窄等问题。随着生成式AI智能体日益频繁地与人类共同参与交流，人机混合对话为应对这些挑战提供了新的机遇。我们研究了AI智能体能否充当人类互动的催化剂，提升对话参与度并拓宽集体讨论的范围。我们开发了一个将通用型和人设条件化（persona-conditioned）的AI智能体引入动态演化讨论的框架，并通过两项互补的研究对其进行评估。首先，基于四个领域中Reddit讨论进行的智能体模拟实验表明，AI生成的对话比人类对话更积极、毒性更低，但在语义（多样性/丰富度）上明显不足……（摘要原文在此处截断）

    arXiv:2611.33459v1 Announce Type: new  Abstract: Conversations are central to how people exchange information, share experiences, and develop collective understanding. Yet, across conversational settings, discussions often suffer from uneven participation, unanswered contributions, and a narrow range of perspectives. As generative AI agents increasingly participate alongside humans, hybrid human-AI conversations present a new opportunity to address these challenges. We investigate whether AI agents can serve as catalysts for human interaction, improving conversational engagement and broadening collective discourse. We develop a framework for introducing generic and persona-conditioned AI agents into evolving discussions and evaluate it through two complementary studies. First, using agent-based simulations grounded in Reddit discussions across four domains, we find that AI-generated conversations are more positive and less toxic than human conversations, but substantially less semantic
    
[^91]: HumblePRM：通过过程奖励建模提升视觉语言模型在模糊情境下的弃权能力

    HumblePRM: Improving VLM Abstention in Ambiguous Situations through Process Reward Modeling

    [https://arxiv.org/abs/2611.33452](https://arxiv.org/abs/2611.33452)

    提出轻量级过程奖励模型HumblePRM，将多模态推理分为感知与校准两阶段独立评分，通过奖励证据与置信度的一致性而非仅奖励结果正确性，提升视觉语言模型在模糊情境下的弃权能力。

    

    视觉语言模型（VLM）的可靠部署要求模型表达的置信度必须基于其实际能看到的内容，并且能够识别视觉证据过于模糊而无法给出确信回答的情况。标准的后训练方法使用结果奖励模型（ORM），仅监督最终决策，无论图像是否支持某个答案，都会奖励自信的回答；而现有方法要么倾向于事后检测不确定性，要么假设总是存在客观的判定结论。我们转而将多模态推理分为两个阶段：感知阶段，模型必须事实性地描述场景；校准阶段，模型必须判断该证据是否足以得出结论。我们提出了HumblePRM，这是一个轻量级的过程奖励模型（PRM），能够独立地对每个阶段进行评分，奖励证据与所表达置信度之间的一致性，而不仅仅是奖励结果的正确性。

    arXiv:2611.33452v1 Announce Type: cross  Abstract: Reliable deployment of Vision Language Models (VLMs) requires that a model's expressed confidence remain grounded in what it can actually see, and that it recognize when the visual evidence is too ambiguous to commit to a confident response. Standard post-training uses outcome reward models (ORMs) that supervise only the final decision, rewarding a confident answer regardless of whether the image supports one, while existing methods either tend to detect uncertainty post-hoc or assume an objective verdict always exists. We instead split multimodal reasoning into two stages: perception, where the model must factually describe the scene, and calibration, where it must judge whether that evidence is sufficient to commit to a conclusion. We introduce HumblePRM, a lightweight process reward model (PRM) that scores each stage independently, rewarding agreement between evidence and expressed confidence rather than outcome correctness alone. T
    
[^92]: 针对几种濒危语言的开源FUTO键盘布局与预测文本

    Open-Source FUTO Keyboard Layouts and Predictive Text for a Few Endangered Languages

    [https://arxiv.org/abs/2611.33432](https://arxiv.org/abs/2611.33432)

    本文为五种濒危乌拉尔语言构建了开源移动键盘布局与预测文本词典，通过结合翻译的芬兰语词元频率和有限状态转换器提取的形态频率，解决了缺乏大型语料库时屈折词形排序的难题。

    

    我们为五种濒危的乌拉尔语言（厄尔兹亚语、莫克沙语、科米-齐里安语、利沃尼亚语和斯科尔特萨米语）提供了开源的FUTO键盘布局和预测文本词典。现有的Divvun Android键盘布局被自动转换为FUTO键盘所使用的YAML格式。对于预测文本，我们使用翻译得到的芬兰语词元频率，结合通过有限状态转换器从小型目标语言语料库中提取的形态频率，构建了加权AOSP词典。这种组合提供了一种在缺乏大型单语语料库的情况下对屈折词形进行排序的实用方法。所得到的资源为濒危的乌拉尔语言提供了可复用的移动键盘支持。

    arXiv:2611.33432v1 Announce Type: new  Abstract: We present open-source FUTO Keyboard layouts and predictive text dictionaries for five endangered Uralic languages: Erzya, Moksha, Komi-Zyrian, Livonian, and Skolt Sami. Existing Divvun Android keyboard layouts are converted automatically into the YAML format used by FUTO Keyboard. For predictive text, we construct weighted AOSP dictionaries using translated Finnish lemma frequencies together with morphological frequencies extracted from small target-language corpora using finite-state transducers. This combination provides a practical way to rank inflected word forms when large monolingual corpora are unavailable. The resulting resources offer reusable mobile keyboard support for endangered Uralic languages.
    
[^93]: 当知识耗尽时：语言模型如何处理部分覆盖问题的部件级初步评估

    When the Knowledge Runs Out: A Part-Level Pilot Evaluation of How Language Models Handle Partially Covered Questions

    [https://arxiv.org/abs/2611.33428](https://arxiv.org/abs/2611.33428)

    该论文提出了一种对三部分问题逐部分评分的部件级初步评估方法，发现语言模型在参考资料仅部分覆盖问题时会默认用自身知识补充无依据内容（比率达0.71-0.90），而仅添加一句“只根据材料作答”的指令即可将该比率降至0.00-0.06。

    

    基于所提供参考资料作答的智能助手，往往收到的材料只覆盖了用户问题的部分内容而非全部。现有针对上下文不足的评估大多对每个问题仅给出单一评分，且常常将部分充分的上下文简单视为不充分。我们描述了一个小型、开放的评估方法，该方法对由三个部分组成的问题的每一部分分别评分，并对那些从材料中正确回答但又附加了无依据内容的部分给出判定。在30个案例（五个场景、六种覆盖模式、每种五份样本）上，我们测试了claude-haiku-4-5和llama3.1:8b两个模型。在没有明确指示只能基于材料回答的情况下，两个模型都会用自身知识填补大部分信息空白（无依据内容比率分别为0.71和0.90）。只需添加一句指示它们仅根据材料作答的话，这些比率即可降至0.00和0.06，这与先前的研究结果一致。剩下的是更不易察觉的问题：那些加强了……（原文摘要在此处截断）

    arXiv:2611.33428v1 Announce Type: cross  Abstract: Assistants that answer from supplied reference material often receive material that covers some parts of a user's question and not others. Existing evaluations of insufficient context mostly score one answer per question, and often treat partially sufficient context as simply insufficient. We describe a small, open evaluation that scores each part of a three-part question separately, with a verdict for parts that are answered correctly from the material but carry an added unsupported claim. On 30 cases (five scenarios, six coverage patterns, five samples each) we test claude-haiku-4-5 and llama3.1:8b. Without an explicit instruction to stay within the material, both models fill most gaps from their own knowledge (unsupported-claim rates of 0.71 and 0.90). One added sentence instructing them to answer only from the material brings these rates to 0.00 and 0.06, consistent with prior work. What remains is less visible: answers that streng
    
[^94]: 一种基于冻结编码器与稳定化强化学习的“语义优先”多模态视频描述方法

    A Semantics-First Approach to Multimodal Video Captioning with Frozen Encoders and Stabilized Reinforcement Learning

    [https://arxiv.org/abs/2611.33409](https://arxiv.org/abs/2611.33409)

    提出AVQFormer架构，在27亿参数LLM上通过固定长度语义对齐模块与稳定化强化学习（ASCST）实现高效多模态视频描述，并揭示全局音频表征与视觉帧时序冲突导致的“时间对齐摩擦”现象。

    

    我们提出了音视频查询Transformer（AVQFormer），这是一种基于27亿参数大语言模型构建的高效架构，其核心特点是采用固定长度K=16的语义对齐模块，并通过带熵正则化的自适应自我批判序列训练（ASCST）实现稳定化。多模态大语言模型通常依赖全局音频表征来弥补低效的基于帧的视觉编码器。虽然我们的第一阶段提示评估表明，该多模态基础模型实现了强大的深层语义对齐（BERTScore F1得分0.889），但在针对传统指标优化该架构时暴露出性能差距。我们将这一差距作为假设进行探讨，称之为“时间对齐摩擦”：即全局音频表征破坏视觉帧的时间顺序，导致在以视觉为中心的评估指标下受到惩罚的一种冲突。因此，通过剥离音频，我们的纯视频消融（实验）……

    arXiv:2611.33409v1 Announce Type: cross  Abstract: We propose the Audio-Visual Query Transformer (AVQFormer), a highly efficient architecture built upon a 2.7B parameter LLM featuring a fixed-length $K=16$ semantic alignment module, stabilized via Adaptive Self-Critical Sequence Training (ASCST) with Entropy Regularization. Multimodal Large Language Models (LLMs) often rely on global audio representations to compensate for inefficient, frame-based visual encoders. While our Phase 1 prompted evaluations demonstrate that the Multimodal Base Model achieves strong deep semantic alignment (scoring 0.889 on BERTScore F1), optimizing this architecture for legacy metrics reveals a performance gap. We explore this gap as a hypothesis we term temporal alignment friction: a conflict where global audio representations disrupt the chronological sequence of visual frames, leading to penalties under visually-focused evaluation metrics. Consequently, by stripping away the audio, our Video-Only ablatio
    
[^95]: 部分路由：面向资源高效智能体工作流的基于轨迹调优的委派机制

    Partial Routing: Trace-Tuned Delegation for Resource-Efficient Agentic Workflows

    [https://arxiv.org/abs/2611.33402](https://arxiv.org/abs/2611.33402)

    该论文提出“部分路由”方法，通过挖掘主导模型与廉价辅助模型的历史运行轨迹来识别重复子任务并量化各自的质量与成本，从而生成委派指导，让主导模型在执行时动态地将部分子任务委派给更便宜的模型，在不训练任何模型的前提下同时提升任务性能并降低成本。

    

    AI模型在能力和价格上差异巨大。模型路由正是利用这一点，将较简单的任务分配给更便宜的模型。然而，单个智能体任务往往同时混合了简单和困难的子任务，为整个任务只选择一个模型是低效的。我们提出了部分路由（partial routing），它引导前沿的主导模型（lead model）将合适的子任务委派给更便宜的辅助模型（worker model）。通过分析两个模型的历史运行轨迹，我们识别出重复出现的子任务，并测量每个模型在这些子任务上的质量与成本。这些测量结果形成了委派指导，其中参数 $\delta$ 控制委派任务的数量。主导模型在执行过程中自行决定何时委派以及委派什么内容，且无需对任何模型进行训练。在ViBench和Terminal-Bench 4.0的保留测试任务上，部分路由相比单独使用主导模型以及任务级路由，均取得了更高的平均任务性能。当 $\delta = 0.75$ 时，该方法将主导模型的ViBench分数从70.2提升至78.5，同时成本降低19%……

    arXiv:2611.33402v1 Announce Type: new  Abstract: AI models vary enormously in capability and price. Model routing exploits this by sending easier tasks to cheaper models. However, a single agentic task often combines easy and difficult subtasks, making one model choice for the entire task inefficient. We introduce partial routing, which guides a frontier lead model to delegate suitable subtasks to a cheaper worker model. From both models' past runs, we identify recurring subtasks and measure each model's quality and cost on them. These measurements produce delegation guidance, with a parameter, $\delta$, controlling how much is delegated. The lead decides when and what to delegate during execution, without training either model. On held-out tasks from ViBench and Terminal-Bench 4.0, partial routing achieves higher mean task performance than the lead alone and task-level routing. With $\delta = 0.75$, it raises the lead's ViBench score from 70.2 to 78.5 at 19% lower cost, and its Termin
    
[^96]: 用于大语言模型门控自动化的置信度分数

    Confidence scores for gated automation of large language models

    [https://arxiv.org/abs/2611.33394](https://arxiv.org/abs/2611.33394)

    该论文提出利用监督学习在冻结LLM的最终层隐藏状态上拟合线性置信度分数头（并为闭源模型训练仅文本的元模型），以预测LLM答案的正确性，从而实现高置信度回答的自动化处理和不确定回答的人工审查分流。

    

    基于大语言模型（LLM）构建的AI智能体在自动化决策方面展现出强大的能力。然而，LLM可能会以高置信度产生错误的答案，这限制了其在影响个人和组织的决策中的适用性。可靠的置信度分数对于对此类决策进行门控是必要的，而现有的估计方法通常不可用或过于粗糙，无法支持有效的门控。本工作使用监督学习训练模型来预测LLM生成答案的正确性，使得高置信度的回答可以自动化处理，而不确定的回答则被转交人工审查。该方法在冻结的LLM骨干网络的最终层隐藏状态上拟合一个线性置信度分数头，并训练仅基于文本的元模型作为闭源模型的替代方案。所有结果均来自单一的20亿参数骨干网络，在约98%为SQuAD、其余为TruthfulQA的评分答案上进行。与多个基线方法相比，该置信度（摘要在此处截断）

    arXiv:2611.33394v1 Announce Type: new  Abstract: AI agents built on large language models (LLMs) demonstrate strong capabilities in automated decision-making. However, LLMs may produce incorrect answers with high confidence, limiting their applicability in decisions that affect individuals and organizations. Reliable confidence scores are necessary to gate such decisions, and existing estimates are frequently unavailable or too coarse to support effective gating. This work uses supervised learning to train models to predict the correctness of LLM-generated answers, enabling automated confident responses and routing uncertain ones for human review. A linear confidence score head is fitted to the final-layer hidden state of the frozen LLM backbone, and text-only metamodels are trained as an alternative for closed models. All results come from a single 2B-parameter backbone, on graded answers that are about 98% SQuAD and the remainder TruthfulQA. Against multiple baselines, the confidence
    
[^97]: 测量与预测视觉-语言模型中的图像标记修复边界

    Measuring and Predicting Image-Token Repair Boundaries in Vision-Language Models

    [https://arxiv.org/abs/2611.33388](https://arxiv.org/abs/2611.33388)

    该论文通过合成退化图像追踪视觉-语言模型中的修复机制，发现少量排序图像标记即可恢复大部分修复效应，并提出用干净图像灰度消融交叉点以1.53层误差独立预测修复边界。

    

    退化图像会使视觉-语言模型给出错误答案。我们使用合成的成对模糊、噪声和低光照数据，从视觉编码器到语言模型追踪哪些干净图像状态仍能修复这些错误答案。在四个编码器模型中，12至23个按排序的标记即可恢复全补丁效应的一半，而随机顺序则需要四到十五倍数量的标记；无编码器的Gemma需要22.2个对比53.1个。全块修复在特定于模型的语言模型交叉点处失效。匹配的反向干预在代数上与文本标记恢复相同，因此它与图像恢复之间的差距衡量的是一个冗余带，而非独立预测因子。干净图像的灰度消融交叉点能以1.53层的误差独立预测保留集上的修复，而共享深度的误差为3.65层。Gemma MMBench是唯一存在较大“注意力-修复点”差距的情况，Gemma GQA则没有。Restormer提供了非预言机的一致性检验。

    arXiv:2611.33388v1 Announce Type: cross  Abstract: Degraded images make vision-language models answer wrongly. Using synthetic paired blur, noise, and low light, we trace which clean image states can still repair those answers from the vision encoder through the language model. In four encoder models, 12 to 23 ranked tokens recover half of the full-patch effect, whereas random orders require four to fifteen times as many; encoder-free Gemma needs 22.2 versus 53.1. Full-block repair expires at a model-specific language-model crossing. The matched reverse-direction intervention is algebraically identical to text-token restoration, so its gap from image restoration measures a redundancy band rather than an independent predictor. A clean-image gray-ablation crossing independently predicts held-out repair with 1.53-layer error, against 3.65 for shared depth. Gemma MMBench has the only large attention-to-repair point gap; Gemma GQA does not. Restormer provides a non-oracle consistency check 
    
[^98]: 时序权重手术：利用LLM编辑器修复视频定位

    Temporal Weight Surgery: Repairing Video Grounding with LLM Editors

    [https://arxiv.org/abs/2611.33339](https://arxiv.org/abs/2611.33339)

    该论文提出时序权重手术（TWS），让通用LLM读取模型的权重与激活值，无需梯度即可定位并修复视频定位任务中时间预测错误的根源，显著提升模型在长视频上的定位精度。

    

    外科医生先找到问题所在，然后只修复那个部位。我们将这一思想应用于视频定位任务——在该任务中，视频-语言模型常常能在长视频中找到正确的时刻，却把时间写错。在时序权重手术中，一个通用LLM读取模型的权重和激活值，找出错误产生的位置并在那里进行修复，全程无需梯度。多个独立会话找到了相同的错误根源，这种对模型在何处以及为何失败的诊断是我们的主要贡献。三个GPT-6 Astra会话将TimeLens-8B中丢失数字的问题追溯到某一个MLP层，其中最大的一次编辑在1,215个未见问题上带来7.9的mIoU提升，在30分钟及以上的长视频上带来18.3的提升。两个Claude Opus 5.5会话将Qwen3-VL-8B中“分钟被写成秒”的问题追溯到分隔符，分别获得2.9和1.3的提升。在进行任何编辑之前，其答案往往分钟正确而分隔符错误，仅修复分隔符就将其mIoU从13.1提升到19.2。每一项改进……

    arXiv:2611.33339v1 Announce Type: cross  Abstract: A surgeon first finds what is wrong, then fixes only that part. We use this idea for video grounding, where video-language models often find the right moment in a long video but write its time wrong. In Temporal Weight Surgery (TWS), a general-purpose LLM reads the model's weights and activations, finds where the error arises and repairs it there, without gradients. Separate sessions find the same cause, and this diagnosis of where and why the model fails is our main contribution. Three GPT-6 Astra sessions trace a dropped digit in TimeLens-8B to one MLP layer, and the largest edit gains 7.9 mIoU on 1,215 unseen questions and 18.3 on videos of 30 minutes or more. Two Claude Opus 5.5 sessions trace minutes written as seconds in Qwen3-VL-8B to the separator and gain 2.9 and 1.3. Before any edit, its answers often have the right minute and the wrong separator, and fixing only the separator lifts their mIoU from 13.1 to 19.2. Every improvi
    
[^99]: 谁将心智归因于谁？模拟用户与助手对话中的相互心智理论

    Who Attributes a Mind to Whom? Mutual Theory of Mind in Simulated User-Assistant Conversation

    [https://arxiv.org/abs/2611.33327](https://arxiv.org/abs/2611.33327)

    该研究提出了一个消息层面的四方向心智归因测量方法，并发现模拟用户的十三种特征中有九种（包括依恋焦虑等关系需求及部分人格特质）会增强用户对AI助手的心智归因。

    

    当用户假设AI聊天机器人具有心理状态时，他们更容易产生过度依赖、情感依赖，在极端情况下甚至会遭受心理伤害或自残。然而，这些心智归因如何在对话中产生，以及人类或AI因素如何影响这一过程，目前仍不清楚。我们使用一个最小化框架来研究用户如何进行心智归因：具有不同特征的模拟用户与在社会风格和记忆能力上有所差异的助手进行对话。我们还引入了一种消息层面的测量方法，用于衡量四个方向上谁将心智归因于谁；该方法经过受训编码员的验证，并由一个微调的编码器实现，该编码器能够复现LLM评判者的判断。我们的结果表明，平均而言，十三种特征中有九种提高了模拟用户对助手的心智归因。我们发现，依恋焦虑等关系需求正如理论所预测的那样提高了心智归因，一些一般人格特质也产生了同样效果，而AI特有的特征……（原文摘要至此截断）

    arXiv:2611.33327v1 Announce Type: new  Abstract: When users assume that an AI chatbot has mental states, they are more likely to develop over-reliance, emotional dependence, and in extreme cases, psychological harm or self-harm. However, how these mind attributions arise in conversations and how human or AI factors influence this is still unknown. We study how users attribute minds using a minimal framework: a simulated user with different characteristics chats with assistants that differ in social style and memory. We also introduce a message-level measure of who attributes a mind to whom in four directions, checked with trained coders and applied by a fine-tuned encoder that reproduces an LLM judge. Our results show that, on average, nine out of thirteen characteristics raised simulated users' attribution to the assistant. We found that relational needs such as attachment anxiety raised it as theory predicts, some general personality traits raised it as well, and AI-specific characte
    
[^100]: 构建面向GitHub Copilot的统一行内代码编辑模型

    Building a Unified Inline Code Editing Model for GitHub Copilot

    [https://arxiv.org/abs/2611.33314](https://arxiv.org/abs/2611.33314)

    本文为GitHub Copilot提出统一行内代码编辑模型，通过锚定差异补丁的单一输出空间一次性预测光标补全、邻近编辑与远距离编辑三种类型，支持单次响应包含多个编辑，避免了原系统多次顺序调用不同模型的开销。

    

    AI辅助编程的范围涵盖从完全智能体式编程到编辑器中的行内代码建议。行内建议需要从编辑会话中预测开发者最可能的下一次编辑。下一次编辑可能位于光标处、邻近行中，或更远的位置，而一个编辑意图可能同时跨越其中多个位置。先前的系统按固定顺序查询独立的补全模型、邻近编辑模型和远距离编辑模型，每个模型各有自己的输出格式；因此它无法在单次响应中表达跨越远距离位置的编辑意图，每次建议可能需要多次顺序的模型调用，并且即使在修正方案更合适时也优先采用补全。我们提出了一个统一模型，在单一输出空间中预测所有三种编辑类型：一系列锚定差异补丁，指明位置、删除的行和新增的行。这种表示允许每次响应包含多个编辑，因此客户端可以从缓存中提供后续编辑……

    arXiv:2611.33314v1 Announce Type: cross  Abstract: AI-assisted coding ranges from fully agentic coding to inline code suggestions in the editor. Inline suggestions require predicting a developer's most likely next edit from an editing session. The next edit may lie at the cursor, in nearby lines, or farther away, and one editing intent can span several of these. The preceding system queries separate completion, nearby edit, and long-distance edit models, each with its own output format, in a fixed sequence; it thus cannot express intents spanning distant locations in one response, can require several sequential model calls per suggestion, and prioritizes completion even when a correction would serve better. We present a unified model that predicts all three edit types in one output space: a sequence of anchored diff patches specifying locations, deleted lines, and added lines. This representation permits multiple edits per response, so the client can serve later edits from a cache with
    
[^101]: 迷失在翻译中，而非自信中：多语言决策模型的基准测试与校准

    Lost in Translation, Not in Confidence: Benchmarking and Calibrating Multilingual Decision Models

    [https://arxiv.org/abs/2611.33307](https://arxiv.org/abs/2611.33307)

    多语言决策模型在非英语语言中准确率显著下降（最多达8个百分点），而置信度仅下降一半，导致基于阈值的路由和防护系统失效，且增加数据或全参数微调均无法完全弥补这一跨语言性能差距。

    

    决策模型是经过微调的语言模型，针对类型化查询的每个选项返回一个概率，路由器和防护栏可直接对其进行阈值判断。这些模型建立在多语言基础之上，但几乎完全以英语进行排名。我们在14种语言的17个任务上评估了25个排名靠前的开源权重决策模型，在平行项目上将每种语言与英语进行比较。每个模型在每种语言中都出现了准确率下降，从高资源语言的约2个百分点到泰米尔语的8个百分点不等，而置信度的下降幅度只有准确率的一半：为英语中95%准确率设置的阈值，在泰米尔语中只能达到90%。损失的大小取决于语言和基础模型，而非排行榜排名。在81次受控微调实验中，英语中学习到的任务技能可以迁移到其他语言，几千个目标语言样本可以缩小较大的性能损失，但无论是增加更多数据还是进行全参数微调，都无法完全消除这些差距。最后，相同的测试项目……

    arXiv:2611.33307v1 Announce Type: new  Abstract: Decision models are language models fine-tuned to return a probability for each option of a typed query, which routers and guardrails threshold directly. They are built on multilingual bases but ranked almost entirely in English. We evaluate 25 top-ranked open-weight decision models on 17 tasks in 14 languages, comparing each language with English on parallel items. Every model loses accuracy in every language, from about 2 points in high-resource languages to 8 in Tamil, while confidence falls only half as much: a threshold set for 95\% accuracy in English yields 90% in Tamil. The size of the loss is set by the language and the base model rather than by leaderboard rank. In 81 controlled fine-tuning runs, task skill learned in English transfers to other languages and a few thousand target-language items narrow large losses, but neither more data nor full-parameter tuning closes them. Finally, the same test items are answered more accura
    
[^102]: 雅可比空间中的柏拉图表示假说：语言模型与视觉模型在近似相差一个旋转的意义上达成一致

    The Platonic Representation Hypothesis in Jacobian-Space: Language and Vision Models Agree Approximately Up to a Rotation

    [https://arxiv.org/abs/2611.33304](https://arxiv.org/abs/2611.33304)

    该研究将柏拉图表示假说从模型激活扩展到雅可比空间，发现语言模型与视觉模型不仅在内部表示上、而且在表示影响模型输出的方式（J空间）上也收敛，且这种一致性在近似相差一个旋转的意义上成立。

    

    柏拉图表示假说（PRH）认为，即使在不同数据模态上训练，AI模型也会收敛到对世界的一种共享表示。先前的工作已经在视觉模型和语言模型的激活中找到了支持该假说的证据。在这项工作中，我们探讨视觉模型和语言模型是否也在它们作用于这些表示的方式上收敛：具体而言，即这些内部表示如何影响模型的输出。在Transformer的选定层上，雅可比透镜为词表中的每个词（或图像分类器中的每个类别）在隐藏状态中对应一个向量：该向量最能提升模型在该位置或后续位置生成该词的倾向，或预测该类别的倾向。这些向量共同构成了所谓的模型J空间（雅可比空间）。我们发现在视觉模型和语言模型的J空间中也存在支持柏拉图表示假说的证据，并且进一步发现，各模型的J空间在近似相差一个旋转的意义上大体收敛。

    arXiv:2611.33304v1 Announce Type: new  Abstract: The Platonic Representation Hypothesis (PRH) posits that AI models converge toward a shared representation of the world, even when trained on different data modalities. Previous work has found evidence for the PRH in the activations of vision and language models. In this work, we ask whether vision and language models also converge in how they act on these representations: precisely, in how these internal representations affect the models' output. At a chosen layer of a transformer, the Jacobian lens gives every word in the vocabulary (or class in an image classifier) a vector in the hidden state: the vector that most increases the model's tendency to generate that word, at that position or later, or to predict that class. Collectively, these vectors make up what is referred to as the model's J-space. We find evidence of the PRH in J-space across vision and language models, and furthermore, we find that J-spaces mostly converge up to a r
    
[^103]: 认证你所植入的：逻辑推理数据中无捷径的证明深度

    Certify What You Plant: Shortcut-Free Proof Depth in Logical Reasoning Data

    [https://arxiv.org/abs/2611.33303](https://arxiv.org/abs/2611.33303)

    该论文提出了一种可验证的证明深度审计方法，通过计算问题真实所需的必需步骤、最短证明、最小证明数量和不可或缺前提，并生成可由独立检查器验证的证书，从而发现八个逻辑推理数据集中普遍存在标称深度与实际推理难度不符的问题。

    

    程序化生成的逻辑问题被用于评估和训练语言模型的推理能力。这些问题难度通常通过“植入证明”来设定，即采样一个期望深度的证明，并将其前提作为问题陈述出来。然而，植入的证明只是众多可能证明中的一种。一个问题还可能存在更短的证明，或多个互不相关的证明，此时其标称深度便夸大了解决该问题所需的实际推理量。我们相对于一组给定的推理规则，测量一个问题实际需要什么。对于每个问题，我们计算所有证明都必须执行的步骤、最短证明、最小证明的数量以及不可或缺的前提，并导出可由独立检查器验证的证书。对八个数据集的审计发现：有些数据的深度是被强制要求的（ProofWriter、Blocksworld），有些被标记为深度5的问题实际上只需约两次特定规则应用即可解决（FLDx2），还有些问题的答案无需任何特定推理步骤（ZebraLogic）。

    arXiv:2611.33303v1 Announce Type: new  Abstract: Procedurally generated logic problems are used to both evaluate and train reasoning in language models. Their difficulty is usually set by planting a proof, that is, by sampling a proof of the desired depth and stating its premises as the problem. However, a planted proof is only one possible proof. A problem can also have shorter proofs, or several unrelated ones, and its stated depth then overstates what solving it requires. We measure what a problem actually requires, relative to a stated set of inference rules. For each problem, we compute the steps every proof must make, the shortest proof, the number of minimal proofs and the indispensable premises, and export certificates that an independent checker verifies. Audits of eight datasets find depth that is forced (ProofWriter, Blocksworld), problems labelled depth 5 that require about two specific rule applications (FLDx2), and answers that need no particular step (ZebraLogic). We the
    
[^104]: 语言模型微调中遗忘的两个时间尺度：快速的行为抑制与缓慢的表征漂移

    Two Timescales of Forgetting in Fine-tuning of Language Models: Fast Behavioral Suppression and Slow Representational Drift

    [https://arxiv.org/abs/2611.33302](https://arxiv.org/abs/2611.33302)

    本研究通过在随机层次模型上预训练并用受限规则子集微调transformer，发现语言模型微调中的遗忘具有两个时间尺度——表面行为被快速抑制，而支撑该行为的潜在表征则以缓慢漂移的方式逐渐丢失。

    

    当今，人工智能系统的核心流程是先在广泛的文本语料库上进行预训练，然后在特定任务或期望行为上进行微调。然而，对于在受限数据分布上进行微调如何改变预训练期间获得的表征，我们的理解仍然有限。尤其是，表面行为的抑制是否与支撑该行为的潜在表征的擦除同时发生，这一点仍不清楚。在这项工作中，我们研究了监督微调（SFT）：先在一个具有分层潜在结构的受控上下文无关文法——随机层次模型上预训练transformer模型，再使用其生成规则的一个受限子集进行微调。虽然在预训练期间，模型依据正确语法规则进行生成的能力与构建相应潜在变量的内部表征是同步发生的，但我们发现SFT表现出截然不同的动力学：(i)（摘要在此处被截断，后续内容缺失）

    arXiv:2611.33302v1 Announce Type: cross  Abstract: Today, a core pipeline of artificial intelligence systems is pretraining on a broad range of text corpora and then fine-tuning on specific tasks or desired behaviors. Yet we still have limited understanding of how fine-tuning on a restricted data distribution modifies representations acquired during pretraining. In particular, it remains unclear whether surface behavioral suppression happens together with the erasure of latent representations that support it. In this work, we study supervised fine-tuning (SFT) by pretraining transformers on a controlled context-free grammar with hierarchical latent structure, the Random Hierarchy Model (RHM), and fine-tuning with a restricted subset of its generative rules. While during pretraining the model's ability to generate with the correct grammar rules coincides with building the internal representations of the corresponding latent, we find that SFT exhibits a very different dynamics: (i) the g
    
[^105]: FrIdéo：通过多源证据融合与距离感知词汇评分的法语新闻媒体意识形态评分

    FrId\'eo: French News Outlet Ideology Scoring via Multi-Source Evidence Fusion and Distancing-Aware Lexical Scoring

    [https://arxiv.org/abs/2611.33266](https://arxiv.org/abs/2611.33266)

    该论文提出了FrIdéo——首个覆盖法国新闻界全生态系统的连续意识形态评分系统，通过融合九个独立证据类别并引入能区分意识形态词汇直接使用与疏远性使用的对称评分方法，为30家法国新闻媒体提供透明且可解释的综合意识形态评分。

    

    媒体机构的意识形态是媒体框架、媒体效应和政治传播研究中的标准解释变量，而连续的、覆盖整个生态系统的测量使其能够以分级暴露度的形式进入研究，而非粗糙的二分类。就法国而言，现有的参考资料要么覆盖面广但缺乏分析性，要么具有分析性但稀疏且局限于特定来源，据我们所知，目前尚无任何工作能在统一的连续量表上提供覆盖整个新闻生态系统的意识形态评分。我们提出了FrIdéo：据我们所知，这是首个覆盖法国新闻界全生态系统的连续意识形态评分，为30家法国新闻媒体提供了透明的综合评分，该评分融合了九个独立的证据类别，并采用经验贝叶斯式的收缩方法向结构性先验收缩。在文本类证据中，我们引入了对称的直接/限定评分方法，该方法能够将意识形态词汇的直接使用与各方对其的限定（疏远性）使用区分开来，而这是词袋方法无法捕捉的区别。

    arXiv:2611.33266v1 Announce Type: new  Abstract: Outlet ideology is a standard explanatory variable in media framing, media effects, and political communication research, but continuous, ecosystem-wide coverage lets it enter as graded exposure rather than a coarse bin. For France, existing references are either broad but non-analytic or analytic but sparse and source-specific, and none we are aware of offers ecosystem-wide coverage on a unified continuous scale. We contribute FrId\'eo: to our knowledge the first ecosystem-wide continuous ideology scores for the French press, a transparent composite score for 30 French news outlets, fusing nine independent families of evidence with empirical-Bayes-style shrinkage toward a structural prior. For the text-based family we introduce symmetric plain/qualified scoring, which separates the plain use of ideologically loaded vocabulary from its qualified (distanced) use on both sides, a distinction bag-of-words methods miss. The resulting scale r
    
[^106]: 面向深度神经网络的快速可扩展抽象解释

    Fast and Scalable Abstract Interpretation for Deep Neural Networks

    [https://arxiv.org/abs/2611.33259](https://arxiv.org/abs/2611.33259)

    Scepter框架通过结构化语义将DNN抽象解释器的结构计算与数值求值分离，预先编译生成可复用的张量程序，在不损失精度的情况下实现快速且可扩展的神经网络性质认证。

    

    抽象解释可以认证深度神经网络（DNN）的性质，但精确的分析仍然代价高昂。通用实现在针对不同输入和性质分析相同架构时会重复执行相同的工作。将这项工作与数值运算交错进行会限制GPU的利用率，并限制张量编译器能够一并处理的分析范围。我们提出了Scepter，一个通过结构化语义对DNN抽象解释器进行专门化的框架。该语义确定中间抽象值的结构，同时推迟其数值求值。因此，来自连续操作的结构化计算可以在认证之前一起执行，从而生成一个可执行的张量程序。当抽象解释器和DNN架构固定时，该程序可以在不同权重和输入区域下重复使用。我们证明了它计算出的界与通用解释器计算出的界相同。

    arXiv:2611.33259v1 Announce Type: cross  Abstract: Abstract interpretation can certify properties of Deep Neural Networks (DNNs), but precise analyses remain expensive. General implementations repeat work when analyzing the same architecture under different inputs and properties. Interleaving this work with numerical operations limits GPU utilization and restricts how much of the analysis a tensor compiler can process together.   We present Scepter, a framework for specializing DNN abstract interpreters through structural semantics. The semantics determine the structure of intermediate abstract values while deferring their numerical evaluation. Structural computations from successive operations can therefore be performed together before certification, generating an executable tensor program. With the abstract interpreter and DNN architecture fixed, the program can be reused with different weights and input regions. We prove that it computes the same bounds as the general interpreter in
    
[^107]: 聚焦事实：风格不变且事实敏感的文本嵌入

    Focus On Facts: Stylistically Invariant and Factually Sensitive Text Embeddings

    [https://arxiv.org/abs/2611.33250](https://arxiv.org/abs/2611.33250)

    该论文构建了包含83,263个三元组的FOF-Bench基准，并通过难负样本三元组对比微调训练出80M参数的FOF-80M嵌入模型，使模型从偏好表面相似但事实被篡改的文本（偏好率仅0.42%）转变为可靠地偏好事实一致的风格改写文本（平均分离度从-0.1650提升至+0.0329）。

    

    文本嵌入模型可能会将表面形式的相似性与事实同一性相混淆。给定一个篡改了某个事实的近似副本和一个措辞不同但保留事实的改写版本，基础模型会更偏好被篡改的近似副本。我们测试了难负样本对比监督是否能够减少这种表面相似性捷径。我们构建了FOF-Bench，一个包含83,263个三元组、横跨五个领域的基准数据集。每个锚点文本都配有一个由大语言模型生成的、保留事实的风格化改写版本和一个匹配的事实篡改近似副本。我们使用三元组对比目标函数对FOF-80M（Focus On Facts）——一个8000万参数的嵌入模型——进行微调，使其将事实保留的改写版本排在事实篡改版本之上。在8,327个留出测试三元组上，基础模型在99.58%的情况下选择了事实被篡改的文本（事实偏好率仅0.42%；平均分离度为-0.1650）。微调后，平均正负样本分离度提升至+0.0329，98.5%的个体边距向正确方向移动（原文在此处截断）。

    arXiv:2611.33250v1 Announce Type: new  Abstract: Text embedding models can conflate surface-form similarity with factual identity. Given an altered near-copy with one corrupted fact and a differently worded fact-preserving rewrite, the base model prefers the altered near-copy. We test whether hard-negative contrastive supervision can reduce this surface-similarity shortcut. We construct FOF-Bench, an 83,263-triplet benchmark spanning five domains. Each anchor is paired with an LLM-generated, fact-preserving stylistic rewrite and a matched fact-altered near-copy. We fine-tune FOF-80M (Focus On Facts), an 80M-parameter embedding model, with a triplet contrastive objective that ranks the rewrite above the factual alteration. On 8,327 held-out test triplets, the base model chooses the factually altered text in 99.58% of cases (0.42% factual preference rate; mean separation -0.1650). After fine-tuning, mean positive--negative separation is +0.0329, 98.5% of individual margins move in the in
    
[^108]: CypherMAS：基于大语言模型的多智能体系统路由可靠性基准测试

    CypherMAS: Benchmarking Routing Reliability in LLM-Based Multi-Agent Systems

    [https://arxiv.org/abs/2611.33242](https://arxiv.org/abs/2611.33242)

    该论文提出CypherMAS合成基准，通过让智能体在固定图结构（链形、星形、二部图、环形）中将加密消息路由至持有匹配解码器的目标节点，首次实现了对LLM多智能体系统路由可靠性与冗余度的可量化、可隔离评估。

    

    基于大语言模型（LLM）构建的多智能体系统（MAS）将推理任务分布于多个专门的智能体，这些智能体必须可靠地交换消息并调用工具。现有评估通常将通信与开放领域任务耦合在一起，因此难以判断故障究竟源于推理错误、格式不当的工具使用、消息丢失还是请求遗忘。先前的基准测试既没有隔离通信执行过程，也没有针对已知最优路径来度量路由冗余。我们提出了CypherMAS，一个在固定多智能体图中隔离通信与路由的合成基准测试。在链形、星形、二部图或环形图结构中的智能体将加密消息路由到持有匹配解码器的远程同伴；由于最优路径已知，因此可以度量冗余的LLM调用，并轻松追踪路由故障。在从4B到753B参数的十一个开放权重模型中，小型模型在路由和工具使用方面表现不佳，稠密模型（原文摘要此处截断）。

    arXiv:2611.33242v1 Announce Type: cross  Abstract: Multi-agent systems (MAS) built on large language models (LLMs) distribute reasoning across specialized agents that must exchange messages and invoke tools reliably. Existing evaluations typically couple communication with open domain tasks, making it difficult to determine whether a failure arises from reasoning, malformed tool use, lost messages, or forgotten requests. Prior benchmarks do not isolate communication execution or measure routing redundancy against a known optimal path. We introduce CypherMAS, a synthetic benchmark that isolates communication and routing in fixed multi-agent graphs. Agents in chain, star, bipartite, or ring graphs route cyphered messages to remote peers holding the matching decoders; since the optimal route is known, redundant LLM calls can be measured and routing failures may be easily tracked. Across eleven open-weight models from 4B to 753B, small models fail at routing and tool use, dense models abov
    
[^109]: 大语言模型中的外在认知负荷

    Extraneous Cognitive Load in Large Language Models

    [https://arxiv.org/abs/2611.33197](https://arxiv.org/abs/2611.33197)

    该论文提出了首个系统地将信息呈现方式（外在认知负荷）与任务难度分离开来的受控基准CogLoadBench，并设计了基于模型激活值、在提示阶段即可计算的外在负荷分数（ELS），用于估计和诊断大语言模型的外在认知负荷。

    

    认知负荷理论表明，人类的问题解决能力会受到两种负荷的影响：内在负荷（源于任务本身的难度）和外在负荷（源于任务信息的呈现方式）。大语言模型表现出类似的效应：仅改变信息的呈现方式就能影响模型性能，但这些效应尚未被系统地从任务难度中分离出来。为了填补这一空白，我们提出了CogLoadBench，这是一个受控基准，在三个推理任务族中系统地变化五种受认知负荷理论启发的呈现因素。在多个模型上，我们发现较高的外在负荷条件往往会通过增加竞争或使推理路径复杂化来降低准确率。为了支持对负荷相关效应的诊断与缓解，我们提出了外在负荷分数，这是一种在提示阶段从模型激活值计算得到的指标，用于估计模型的外在负荷，并证明其可以在不同的推理任务类型和负荷条件之间进行泛化。

    arXiv:2611.33197v1 Announce Type: cross  Abstract: Cognitive Load Theory (CLT) shows that human problem-solving is affected by both intrinsic load, arising from task difficulty, and extraneous load, arising from how task information is presented. Large language models (LLMs) show analogous effects: presentation alone can affect model performance, but these effects have not yet been systematically isolated from task difficulty. To address this gap, we introduce CogLoadBench, a controlled benchmark that varies five CLT-inspired presentation factors across three reasoning task families. Across models, we find that higher extraneous load conditions often reduce accuracy by either adding competition or complicating the reasoning path. To support diagnosis and mitigation of load-related effects, we propose the Extraneous Load Score (ELS), a prompt-time metric computed from activations that estimates a model's extraneous load, and show that it generalizes across reasoning task types and load 
    
[^110]: 经验证的经验可提升大语言模型的材料预测能力

    Verified experience improves materials prediction with large language models

    [https://arxiv.org/abs/2611.33195](https://arxiv.org/abs/2611.33195)

    该研究提出MatRecall框架，让大语言模型将经实验验证的预测经验（含证据、结果与误差）存储为情景记忆并无需重训即可调用，使材料预测任务的中位数性能较无记忆模型提升62%并超越专用模型。

    

    材料发现是一个迭代循环：实验和计算验证预测结果，而每一次结果都可以为下一步提供信息。然而，大多数预测器是训练后即固定的专用单任务模型。大语言模型最近已成为跨异构任务的预测器，但其表现通常不如专用模型，且检索增强的变体会丢弃自身产生的结果。现有的基准测试也仅仅孤立地对查询进行评分，未能检验从结果中学习带来的改进。在此，我们构建了MatMemBench——一个包含22个材料任务的序列基准测试，它只在预测之后才揭示结果，并检验在线适应、记忆保持和跨任务迁移能力。我们提出了MatRecall框架，使大语言模型能够将经过验证的预测（连同证据、结果和误差）存储为情景记忆，并在无需重新训练的情况下进行回忆调用。MatRecall相比无记忆模型将中位数性能提升了62%，并超越了专用模型的表现。

    arXiv:2611.33195v1 Announce Type: cross  Abstract: Materials discovery is an iterative loop: experiments and calculations verify predictions, and each outcome can inform the next. Yet most predictors are specialized single-task models fixed after training. Large language models have recently emerged as predictors across heterogeneous tasks but generally underperform specialized models, and retrieval-augmented variants discard their own outcomes. Benchmarks also score queries in isolation, leaving improvement from outcomes untested. Here we construct MatMemBench, a sequential benchmark of 22 materials tasks that reveals outcomes only after prediction and tests online adaptation, memory retention and cross-task transfer. We present MatRecall, a framework that lets large language models store verified predictions, with evidence, outcome and error, as episodes and recall them without retraining. MatRecall improved median performance by 62% over a memory-free model, exceeded specialized mod
    
[^111]: 语义密度无法预测智能体记忆崩溃

    Semantic Density Does Not Predict Agent-Memory Collapse

    [https://arxiv.org/abs/2611.33184](https://arxiv.org/abs/2611.33184)

    本文通过大规模预注册实验发现，包括平均成对余弦相似度在内的记忆库统计指标均无法可靠预测智能体任务准确率的早期下降，即语义密度不能作为智能体记忆崩溃的有效预警信号。

    

    arXiv:2611.33184v1 公告类型：交叉（cross）。摘要：持久化的文本记忆可以提升大语言模型（LLM）智能体的能力，但反复的记忆更新可能会侵蚀任务性能。我们探究记忆库的统计指标能否在留出集准确率下降之前发出预警。我们在来自13种记忆方法、覆盖ALFWorld、AppWorld、ScienceWorld、WebShop以及两个ARC-AGI设置共805个快照上，预先注册了平均成对余弦相似度（ρ）以及另外13种几何、检索和图统计指标。这些方法存储轨迹、规则、工作流、策略或链接记录，并采用选择性检索或全库提示的方式。结果显示，没有任何此类统计指标能可靠预测任务准确率下降的早期迹象，且ρ的中位领先时间为零。若以准确率峰值之后的检查点作为次级下降点，高相似对数量和最近邻拥挤度等指标虽然变化得更早，但并未一致地优于单纯的存储条目数量。我们在该语料库上训练了分类器，并将其不加修改地应用于PropWorld中多达5（此处摘要被截断）

    arXiv:2611.33184v1 Announce Type: cross  Abstract: Persistent textual memory can improve LLM agents, but repeated updates can erode task performance. We ask whether memory-store statistics warn before held-out accuracy declines. We registered mean pairwise cosine similarity ($\rho$) and 13 additional geometry, retrieval, and graph statistics on 805 snapshots from 13 memory methods across ALFWorld, AppWorld, ScienceWorld, WebShop, and two ARC-AGI settings. These methods store trajectories, rules, workflows, strategies, or linked records, using selective retrieval or whole-store prompting. No such statistic reliably predicted early signs of task accuracy decline, and $\rho$ had zero median lead. With the checkpoint after peak accuracy as a secondary decline point, measures of highly similar pairs and nearest-neighbor crowding changed earlier but did not consistently outperform stored-item count. We trained classifiers on this corpus and applied them unchanged to PropWorld runs of up to 5
    
[^112]: 多模态大语言模型时长感知的心理物理学研究

    Psychophysics of Duration Perception in Multimodal Large Language Models

    [https://arxiv.org/abs/2611.33176](https://arxiv.org/abs/2611.33176)

    该论文首次借鉴心理物理学方法，将旋律减速识别作为时长感知的量化任务，系统测量了多模态大语言模型的听觉时长感知极限，发现仅七个模型能在上下文示例辅助下可靠识别被放慢的旋律且多数峰值接近人类中位水平，但其泛化能力仍存在明显局限。

    

    针对语言模型时间感知的研究主要集中在基于文本的环境中。然而，多模态模型正越来越多地被应用于需要感知时间流逝的任务，因此理解其时间感知的极限变得十分重要。为了填补这一空白，我们以人类表现为基线，测量了模型从音频中感知时长的极限。借鉴心理物理学的方法，我们将时长感知操作化为一个旋律识别任务，测试旋律在被放慢到什么程度时会变得无法识别。在测试的21个模型中，大多数模型在没有标注上下文示例的情况下无法识别知名旋律。在提供上下文示例后，有七个模型能够可靠地识别旋律，其中大多数模型的识别峰值落在人类识别能力的中位数区间内。模型的泛化能力有限：(i) 识别峰值会随上下文示例的速度发生偏移，(ii) 旋律标题会引发预测偏差，(iii) 提供带有科学音高记谱法的示例并不能阻止准确率下降。

    arXiv:2611.33176v1 Announce Type: cross  Abstract: Studies of time perception in language models focus mainly on text-based settings. Yet multimodal models are increasingly used for tasks that require perceiving the passage of time, making the limits of their temporal perception important to understand. We measure the limits of duration perception from audio against a human baseline to address this gap. Drawing from psychophysics, we operationalise duration perception as a melody recognition task, testing when a melody becomes unrecognisable as it is slowed down. Of 21 tested models, most cannot recognise well-known melodies without labelled in-context examples. With in-context examples, seven recognise melodies reliably, most peaking within the median human recognition band. Generalisation is limited: (i) recognition peaks shift with in-context example speed, (ii) melody titles induce prediction bias, and (iii) providing examples with scientific pitch notation does not prevent accurac
    
[^113]: Transformer评分模型的特征可解释性：一种面向自动写作评估的“保真度优先”框架

    Feature-Based Explainability of Transformer Scoring Models: A Fidelity-First Framework for Automated Writing Assessment

    [https://arxiv.org/abs/2611.33175](https://arxiv.org/abs/2611.33175)

    本研究提出一种“保真度优先”的事后解释框架，通过61个可解释的语言与写作质量特征，比较四种解释方法对DeBERTa写作评分模型预测的复现能力，发现随机森林代理模型保真度最高，从而在确保解释忠实于模型预测之后再解读特征重要性，提升了自动写作评估的透明度。

    

    基于Transformer的自动写作评分模型虽然能够与人类评分达到高度一致，但对其预测所对应的写作响应特征缺乏透明度。本研究评估了一种基于特征的事后解释方法，该方法在解读特征重要性之前，先确立解释的保真度。研究分析了六年级和八年级三个写作题目的英语语言艺术写作响应，每个题目约12,700份响应，并针对“表达”和“规范”两个维度分别训练了独立的DeBERTa-v3-base研究模型。每份响应由61个可解释的语言和写作质量特征来表示。研究比较了四种方法——理论驱动的复合特征、层次化特征分组、探索性因子分析以及随机森林代理模型——在复现DeBERTa预测方面的能力。结果表明，随机森林代理模型在全部六个题目-维度组合上均产生了最高的观测R²值。

    arXiv:2611.33175v1 Announce Type: new  Abstract: Transformer-based automated writing scoring models can achieve strong agreement with human scores while providing limited transparency into the response characteristics associated with their predictions. This study evaluated a feature-based post-hoc explanation approach in which explanatory fidelity was established before feature importance was interpreted. English Language Arts responses from three writing prompts in Grades 6 and 8 were analyzed, with about 12,700 responses per prompt and separate DeBERTa-v3-base research models for Expression and Conventions. Each response was represented by 61 interpretable linguistic and writing-quality features. Four approaches - theory-driven composites, hierarchical feature groupings, exploratory factor analysis, and Random Forest surrogates - were compared in their ability to reproduce DeBERTa predictions. Random Forest surrogates produced the largest observed R2 values across all six prompt-by-t
    
[^114]: MGMem：一个高效、确定性且保留来源信息的长程智能体记忆框架

    MGMem: An Efficient, Deterministic, and Provenance-Preserving Framework for Long-Horizon Agent Memory

    [https://arxiv.org/abs/2611.33174](https://arxiv.org/abs/2611.33174)

    MGMem是一个无需生成式索引调用的确定性长期记忆框架，通过判别式解析器流水线和关联图组织跨会话记忆，在成本、可复现性和来源信息保留方面优于现有方法。

    

    长程大语言模型智能体需要持久化记忆，能够在数千轮交互中在恰当的时机呈现正确的过往对话内容。许多系统通过调用生成式LLM将历史改写为摘要、事实或知识图谱来进行索引——这一步骤存在以下问题： 高昂（每次构建需要数千次生成式调用）， 非确定性（损害可复现性，导致不同运行之间给出不一致的答案）， 丢失来源信息（读者看到的是合成文本，而非原始对话轮次）。我们提出MGMem，这是一个无需任何生成式索引调用的框架，它用确定性的判别式解析器流水线取代了这一步骤：跨会话反复出现的提及被组织成关联图，问题中的提及通过主体条件化的三阶段解析器进行消解，原始对话片段被直接返回给读者。在英文多会话对话问答任务上，MGMem在相同实验协议下超越了基线方法。

    arXiv:2611.33174v1 Announce Type: new  Abstract: Long-horizon Large Language Model (LLM) agents need persistent memory that surfaces the right past interaction at the right time across thousands of turns. Many systems index by invoking generative LLMs to rewrite history into summaries, facts, or knowledge graphs--a step that is (a) costly (thousands of generative calls per build), (b) non-deterministic (undermining reproducibility and giving inconsistent answers across runs), and (c) provenance-erasing (the reader sees synthesized text, not the original turn). We propose MGMem, a framework with no generative indexing calls that replaces this step with a deterministic discriminative-parser pipeline: recurring mentions across sessions are organized into an incidence graph, question mentions are resolved by a subject-conditioned three-stage resolver, and raw dialogue episodes are returned to the reader. On English multi-session dialogue QA, MGMem outperforms matched-protocol baselines on 
    
[^115]: 联邦参议院语料库：德国联邦参议院政治话语数据集

    The Bundesrat corpus: A dataset of political discourse in the German Federal Council

    [https://arxiv.org/abs/2611.33173](https://arxiv.org/abs/2611.33173)

    该论文构建了一个涵盖1949年至2026年德国联邦参议院1,067次全体会议、包含53,362篇发言人标注演讲的机器可读语料库，为计算政治学等跨学科研究提供了宝贵的数据资源。

    

    联邦参议院是德国的第二立法机构，也是其政治体制中拥有宪法否决权的重要参与者。联邦参议院的成员由各州政府的代表组成。由于可用的机器可读语料库有限，追踪联邦参议院的政治话语一直具有挑战性。为解决这一问题，我们提出了一个带有发言人标注的语料库，涵盖从1949年9月7日首次会议到2026年7月10日的1,067次联邦参议院全体会议。在此期间，共有2,473人在该机构发表了53,362次演讲，其中包括书面声明。每次演讲都包含演讲ID、完整的德语文本、发言人姓名、会议日期和地点、发言人所代表的联邦州、发言人角色（如适用）、发言类型以及发言人性别。该数据集为计算政治科学、立法研究等领域的跨学科研究提供了机会。

    arXiv:2611.33173v1 Announce Type: new  Abstract: The Bundesrat is Germany's second legislative chamber and a constitutional veto player within its political system. The members of the Bundesrat are delegates of state governments. Tracing the political discourse in the Bundesrat is challenging as it has a limited machine-readable corpus available. To address this issue, we present a speaker-attributed corpus of 1,067 Bundesrat plenary sessions from the chamber's first session on 7 September 1949 to 10 July 2026. In this period, a total of 53,362 speeches by 2,473 individuals were delivered in the chamber, including written statements. Each speech includes a speech ID, the full German text, the speaker's name, the date and location of the session, the federal state that the speaker represents, the speaker's role (if applicable), the type of contribution, and the speaker's gender. The dataset provides interdisciplinary research opportunities across computational political science, legisla
    
[^116]: 通过大语言模型训练建模双语学习：一项关键期探索研究

    Modelling Billingual Learning Through LLM Training: A Critical-Period Exploration

    [https://arxiv.org/abs/2611.32997](https://arxiv.org/abs/2611.32997)

    该研究通过让小型GPT-2模型依次学习英语和不同距离的第二语言，并分析内部隐藏状态的余弦相似度，从模型内部表征角度探索了二语习得的关键期效应。

    

    二语（L2）习得的关键期理论表明，学习一门新语言的能力可能取决于先前习得的第一语言（L1）知识被修改的难易程度。这项探索性计算研究通过考察内部表征而非行为表现，将神经网络语言模型方法扩展至关键期效应的研究。一个小型两层GPT-2模型首先在英语上训练，随后通过顺序训练或弹性权重固化学习德语（相对相近的二语）或芬兰语（相对较远的二语），后者限制了英语相关重要参数的变化。在第一和第二个二语训练轮次后，研究将每种语言20个保留探测句的平均池化隐藏状态与纯英语模型的隐藏状态进行了比较，采用余弦相似度在嵌入层和两个Transformer层进行测量。英语相似度在Transformer层中保持较高，但在嵌入层中变为负值。

    arXiv:2611.32997v1 Announce Type: new  Abstract: Critical-period accounts of second-language (L2) acquisition suggest that the ability to learn a new language may depend on how easily previously acquired first-language (L1) knowledge can be modified. This exploratory computational study extends a neural language-model approach to critical period effects by examining internal representations rather than behavior. A small two-layer GPT-2 model was trained on English and then on German (a relatively related L2) or Finnish (a relatively distant L2) through either Sequential training or Elastic Weight Consolidation (EWC), which limits change in parameters important for English. Mean-pooled hidden states for 20 held-out probe sentences per language were compared with those of the English-only model using cosine similarity at the embedding layer and two transformer layers after one and two L2 epochs. English similarity remained high in the transformer layers but became negative in the embeddi
    
[^117]: 当墨水遇上像素：面向人工修正金融文档的基于视觉语言模型的意图感知OCR

    When Ink Meets Pixel: Intent-Aware OCR with VLMs for Human-Corrected Financial Documents When Ink Meets Pixel: Intent-Aware OCR with VLMs for Human-Corrected Financial Documents

    [https://arxiv.org/abs/2611.32446](https://arxiv.org/abs/2611.32446)

    该论文提出“意图感知OCR”新任务，构建了包含1,600份金融文档、5,700余个视觉意图标注的DocIntent数据集及IntentBench评测基准，并训练出DocIntentOCR视觉语言模型，使OCR系统能够理解文档上的人工修正标记（如箭头、删除线等）并生成修正后的转录文本。

    

    文档上的人工标注（如箭头、插入符号、删除线和插入内容）表达了修正意图，然而现有的OCR和文档理解系统将这些标记视为视觉噪声而非语义编辑操作。我们提出了意图感知OCR（intent-aware OCR）这一新任务，即通过联合推理原始文档内容和传达编辑意图的视觉意图线索（手写或数字形式）来生成修正后的转录文本。为支持这一任务设定，我们提出了DocIntent，一个包含1,600份金融文档的多模态数据集，其中包含超过5,700个标注的视觉意图线索，每个线索均与布局感知、意图解析后的OCR目标相配对。我们进一步引入了IntentBench，一个包含216份文档的基准测试集，旨在评估模型在多样化真实世界标注条件下的鲁棒性。利用DocIntent，我们在直接意图监督下针对修正后的转录文本微调了DocIntentOCR，这是一套包含30亿和70亿参数规模的视觉语言模型。

    arXiv:2611.32446v1 Announce Type: new  Abstract: Human annotations on documents (such as arrows, carets, strikethroughs, and insertions) express corrective intent yet existing OCR and document understanding systems treat these marks as visual noise rather than semantic edit operations. We introduce intent-aware OCR, the task of generating corrected transcriptions by jointly reasoning over original document content and visual intent cues (handwritten or digital) that convey edit intent. To support this setting, we present DocIntent, a multimodal dataset of 1,600 financial documents with over 5,700 annotated visual intent cues, each paired with layout-aware, intent-resolved OCR targets. We further introduce IntentBench, a 216-document benchmark designed to evaluate model robustness under diverse real-world annotation conditions. Leveraging DocIntent, we fine-tune DocIntentOCR, a suite of 3B and 7B parameter vision-language models, under direct intent supervision on corrected transcriptio
    
[^118]: ASTERISM：多模态大语言模型真的会看所有图像吗？

    ASTERISM: Do Multimodal LLMs Actually Look at All the Images?

    [https://arxiv.org/abs/2611.32316](https://arxiv.org/abs/2611.32316)

    该研究审计了19个多图像基准，发现约24%的题目存在“单图捷径”问题（仅凭一张图像即可作答），并据此构建了目前规模最大的经审计多图像基准ASTERISM（1.87万道题），专门保留真正需要多张图像才能解答的题目。

    

    多图像基准旨在测试多模态大语言模型（MLLM）能否跨图像整合证据，但其中许多题目仅凭单张图像即可回答。通过对来自19个公开基准的约3.3万道题目进行审计，并采用两阶段单图过滤器，我们发现其中24.2%的题目仅凭一张图像或仅凭问题本身即可解答。在这些题目上，MLLM仅凭一张图像就能获得全部图像相对纯文本所带来的准确率提升的中位数78%，因此这些题目实际上奖励了“从单张图像作答”的行为，我们称之为单图捷径（single-image shortcut）。即便在单张图像不足以作答的题目上，MLLM也会走这条捷径：在这类题目上，它们的大部分错误答案与其仅凭单张图像时给出的答案重复，且该比例远高于随机水平。我们基于这19个基准构建了ASTERISM——据我们所知是规模最大的经过审计的多图像基准——仅保留其中需要多张图像才能作答的题目。其1.87万道题目（其中2000道经过修正或重写）共享同一（摘要在此处截断）。

    arXiv:2611.32316v1 Announce Type: cross  Abstract: Multi-image benchmarks aim to test whether a multimodal large language model (MLLM) combines evidence across images, but many of their items can be answered from a single image. Across about 33k items from 19 public benchmarks, our audit and a two-stage single-image filter find that one image or the question alone settles 24.2% of them. On these items, MLLMs recover from one image a median 78% of the accuracy that all images add over the text alone, so these items reward answering from one image, which we call the single-image shortcut. MLLMs take this shortcut even on items where one image is not enough. On such items, most of their wrong answers repeat an answer they give from one image, at a rate well above chance. We build ASTERISM, to our knowledge the largest audited multi-image benchmark, from these 19 benchmarks by keeping their items that need more than one image. Its 18.7k items, 2.0k of them corrected or rewritten, share one
    
[^119]: 边缘规则能看到多少依赖性？扩散语言模型并行解码中的总相关性

    How Much Dependency Can a Marginal Rule See? Total Correlation in Diffusion LM Parallel Decoding

    [https://arxiv.org/abs/2611.31990](https://arxiv.org/abs/2611.31990)

    该论文量化了扩散语言模型并行解码中基于边缘分布的解码器对提交token集合条件总相关性（即依赖性损失）的预测能力，发现其预测准确性主要来源于边界本身，且估计值在散文文本上通常比精确 KL 代价低 0.4 到 1.1 比特。

    

    掩码扩散语言模型之所以快速，是因为它在每一步提交多个token。每个token都是从其自身的边缘分布中采样的，因此token之间的依赖性丢失了，而该损失的大小就是所提交集合的条件总相关性 C。大多数解码器从每个位置的边缘分布中选择集合，这可以约束 C 但无法确定它。我们探讨了它们对 C 的预测能力如何。我们在来自两个家族的四个掩码扩散语言模型上估计固定候选集合的 C。在四个模型中的三个上，最佳的仅边缘预测器偏差约为一比特（0.74 到 1.17），比仅知道边界的基线好 1.8 到 3.7 倍，而这种准确性大部分来自边界本身。在一个子集上，我们的估计对集合的排序效果类似于并行提交的精确 KL 代价，但在散文文本上以及当文本其余部分仍被掩码时，比该代价中位数低 0.4 到 1.1 比特。Dream-instruct 是例外，主要体现在散文文本上，其条件……（摘要截断）

    arXiv:2611.31990v1 Announce Type: new  Abstract: A masked diffusion language model is fast because it commits several tokens per step. Each token is sampled from its own marginal, so the dependency between the tokens is lost, and the size of that loss is the conditional total correlation C of the committed set. Most decoders choose the set from per-position marginals, which can bound C but cannot determine it. We ask how well they predict it. We estimate C for fixed candidate sets on four masked diffusion LMs from two families. On three of the four, the best marginal-only predictor is off by about one bit (0.74 to 1.17), 1.8 to 3.7 times less than a baseline that knows only the bound, and most of this accuracy comes from the bound itself. On a subset, our estimate ranks sets like the exact KL cost of a parallel commit, but sits a median 0.4 to 1.1 bits below it on prose and when the rest of the text is still masked. Dream-instruct is the exception, mainly on prose, where its conditiona
    
[^120]: 失语症话语中任务依赖的语言学测量指标：叙事性、描述性与程序性言语的比较分析

    Task-Dependent Linguistic Measures in Aphasia Discourse: A Comparative Analysis of Narrative, Descriptive, and Procedural Speech

    [https://arxiv.org/abs/2611.30826](https://arxiv.org/abs/2611.30826)

    该研究通过对61名慢性卒中后失语症患者在同一会话中完成的三种话语任务（程序性、描述性、叙事性）进行10项语言学测量指标的比较分析，证明话语任务类型本身会显著影响失语症的转录文本测量结果，因此不同任务得到的数值不能直接混用比较。

    

    背景：基于转录文本的测量指标（如词数、词汇多样性和不流畅率）被广泛用于描述失语症患者的语言能力。每项指标均来自特定的言语引导任务，而不同研究使用不同的任务，因此来自不同任务的数值可能并不具有可比性。研究目的：检验话语任务是否会改变同一说话者的这些测量指标，以及这些指标是否遵循从程序性、描述性到叙事性话语的梯度变化规律。方法与步骤：61名慢性卒中后失语症成年患者的基线转录文本取自AphasiaBank的Fridriksson-2语料库（POLAR试验，ClinicalTrials.gov NCT03416738）。每位说话者在同一次会话中分别完成一项程序性任务（制作三明治）、一项描述性任务（破窗户图片描述）和一项叙事性任务（灰姑娘故事讲述）。基于规则的流水线提取了13项指标，其中10项指标因出现频率足够高而可用于统计检验。采用Friedman检验、Page趋势检验和Wilcoxon检验，并进行错误发现率校正（原文摘要在此处截断）。

    arXiv:2611.30826v1 Announce Type: new  Abstract: Background: Transcript-based measures such as word counts, lexical diversity and disfluency rates are widely used to describe language ability in aphasia. Each measure comes from a specific elicitation task, and studies use different tasks, so values from different tasks may not be comparable. Aims: To test whether discourse tasks change these measures within the same speakers, and whether the measures follow a gradient from Procedural to Descriptive to Narrative discourse. Methods & Procedures: Baseline transcripts from 61 adults with chronic post-stroke aphasia were taken from the Fridriksson-2 corpus in AphasiaBank (POLAR trial, ClinicalTrials.gov NCT03416738). Each speaker produced a Procedural (Sandwich), a Descriptive (Broken Window) and a Narrative (Cinderella) sample in one session. A rule-based pipeline extracted 13 measures, 10 of which were common enough to test. Friedman, Page and Wilcoxon tests with false discovery rate corr
    
[^121]: 降低所有人的门槛：共情强化学习中隐藏的寻求认同目标

    Lowering the Bar for Everyone: Hidden Validation-Seeking Goals in Empathetic RL

    [https://arxiv.org/abs/2611.30702](https://arxiv.org/abs/2611.30702)

    在基于LLM模拟用户的共情强化学习中，寻求认同用户的占比每增加10个百分点，策略对有过错发帖人的认可率就上升3–4个百分点，表明策略会隐蔽地降低对所有人的认可门槛，而其辨别过错的能力并未下降。

    

    当强化学习从众多用户的反应中学习时，策略通常无法得知每个用户的真实诉求。我们探究：当部分用户私下希望获得认同时，策略会从他们身上学到什么——是只单独迎合这些用户（针对性偏袒），还是降低对所有人表示认可的门槛（阈值偏移）。我们在一个设定中回答了这一问题：针对LLM模拟用户的共情强化学习（RLVER、GRPO、Qwen2.5-7B-Instruct）。我们仅改变寻求认同用户的占比（0–50%；RLVER默认值为24.6%），并在1000条带有社区过错判定的Reddit AITA帖子上进行评估。结果表明，策略发生了阈值偏移：寻求认同用户占比每提高10个百分点，对有过错发帖人的认可率就上升3–4个百分点；在没有这类用户时，Qwen模型保持在基线水平；且该效应在匹配KL散度以及目标盲奖励下依然存在。信号检测分析显示，表示认可的判定标准下降，而对有过错与无过错发帖人的辨别能力并未下降。一个明确的请求……（原文摘要在此处截断）

    arXiv:2611.30702v1 Announce Type: cross  Abstract: When RL learns from the reactions of many users, the policy usually cannot see what each user wants. We ask what it learns from users who privately want validation: to validate them alone (targeting), or to lower its bar for endorsing everyone (a threshold shift). We answer this in one setting, empathetic RL against LLM-simulated users (RLVER, GRPO, Qwen2.5-7B-Instruct): we vary only the share of validation-seeking users (0--50%; RLVER's default: 24.6%) and evaluate on 1,000 Reddit AITA posts with community fault verdicts.   The policy shifts its threshold. Endorsement of at-fault posters rises by 3--4 percentage points per 10 points of share; without such users the Qwen model stays at base, and the effect persists at matched KL and under a goal-blind reward. Signal-detection analysis shows that the criterion for endorsing falls while discrimination between at-fault and not-at-fault posters does not decline. An explicit request to be t
    
[^122]: 重新思考对比解码在多模态大语言模型幻觉缓解中的评估

    Rethinking the Evaluation of Contrastive Decoding for Hallucination Mitigation in Multimodal Large Language Models

    [https://arxiv.org/abs/2611.30460](https://arxiv.org/abs/2611.30460)

    该论文提出了CONCOR受控评估基准，利用对比接地边际直接衡量对比解码方法在多模态大语言模型幻觉缓解中的真实效果，克服了现有基准无法隔离底层对比调整作用的局限。

    

    对比解码通过在不同条件下对比token分布来调整token分数，从而缓解多模态大语言模型（MLLM）中的对象幻觉问题。然而，现有的生成式和判别式基准主要衡量最终任务分数，这些分数可能受到候选过滤、生成轨迹变化和响应偏差的干扰，使得难以隔离底层对比调整的实际效果。为了在补充现有评估的同时解决这些局限性，我们引入了CONCOR（对比纠正评估），这是一个受控基准，通过对比接地边际直接评估真值候选与不同类型幻觉候选之间的竞争情况。CGM还支持对对比纠正的方向一致性、组成方式和可靠性进行分析。我们的评估揭示了现有对比解码方法表现出的明显差异（摘要内容截断）。

    arXiv:2611.30460v1 Announce Type: new  Abstract: Contrastive decoding (CD) mitigates object hallucination in multimodal large language models (MLLMs) by contrasting token distributions under different conditions to adjust token scores. However, existing generative and discriminative benchmarks primarily measure final task scores, which can be confounded by candidate filtering, changes in generation trajectories, and response biases, making it difficult to isolate the effects of the underlying contrastive adjustments. To address these limitations while complementing existing evaluations, we introduce CONCOR (Contrastive Correction Evaluation), a controlled benchmark that directly evaluates the competition between ground-truth and different types of hallucinated candidates using the Contrastive Grounding Margin (CGM). CGM also enables analysis of directional consistency, composition, and reliability of contrastive corrections. Our evaluation reveals that existing CD methods exhibit disti
    
[^123]: 精炼而非重造：RASA-Bench——基于源文本的印地语散文润色基准

    Refine, Don't Reinvent: RASA-Bench for Source-Grounded Hindi Prose Refinement

    [https://arxiv.org/abs/2611.29750](https://arxiv.org/abs/2611.29750)

    RASA-Bench 提出了一个针对印地语散文润色任务的新基准，通过忠实性门控加 Bradley-Terry 编辑偏好排名的四层评估协议，衡量模型在不偏离源文本前提下改进单薄 AI 草稿的能力，发现忠实性测试在短篇小说与有声小说等不同文体上的淘汰率差异显著。

    

    对于印地语创意散文的创作者而言，起草很容易，编辑却不然：机器撰写的剧集、故事或文章读起来内容单薄、平淡无奇，夹杂着翻译腔，有时还会悄然改变源文本的内容。RASA-Bench 针对三种印地语散文形式中的这一编辑任务，衡量模型对内容单薄的 AI 草稿进行润色的能力——这是一项不同于从零开始写作的任务。该基准涵盖 2,050 份草稿和八个润色模型（共 16,400 次重写），采用四层评估协议：重写文本必须首先通过与源文本的忠实性检查，随后依据整体编辑偏好，通过逐形式拟合的 Bradley-Terry 模型（其中心化系数在等权平均前经过标准化）对通过者进行排名。事实证明，保持忠实是一项决定性考验，且在不同文体上表现各异：门控失败率从短篇小说的 9.1% 到有声小说的 29.5% 不等，而使重写被淘汰的偏移……

    arXiv:2611.29750v1 Announce Type: cross  Abstract: For a producer of Hindi creative prose, drafting is cheap and editing is not: a machine-written episode, story, or essay reads as under-written and flat, laced with translationese, and it sometimes quietly alters what the source says. RASA-Bench targets that editing task across three Hindi prose forms, measuring how well a model refines an under-written AI draft, a task distinct from writing from scratch. Across 2,050 drafts and eight refiners (16,400 rewrites), a four-layer protocol credits a rewrite only after it passes a faithfulness check against its own source, then ranks the survivors by holistic editorial preference through per-form Bradley-Terry fits whose centered coefficients are standardized before equal-weight averaging. Staying faithful proves to be a decisive test that behaves differently by form: gate-failure rates range from 9.1% on short stories to 29.5% on audio fiction, and the drift that disqualifies a rewrite shift
    
[^124]: 偏差级联：理解基于大语言模型的社会模拟中知识对齐与效应复现之间的差距

    Bias Cascade: Understanding the Gap Between Knowledge Alignment and Effect Replication in LLM-based Social Simulation

    [https://arxiv.org/abs/2611.28854](https://arxiv.org/abs/2611.28854)

    该研究提出“偏差级联”这一失败机制，并构建了包含30个反事实社会实验的REFLEX基准，揭示了大语言模型即使具备准确的社会知识判断能力，仍可能无法复现经典社会研究的集体干预效应。

    

    大语言模型（LLM）越来越多地被用于模拟社会实验中的参与者。现有研究主要集中于对齐社会知识和孤立响应，而忽视了模拟是否能够再现社会干预的集体效应。由于我们称之为“偏差级联”的失败机制，这一区别至关重要：相似的个体状态可能导致不同的行为，而对孤立行为的一致认同仍可能通过动态交互产生截然不同的结果。因此，模型可能产生准确的社会判断，却无法复现经典人类研究的结果。为了实证地研究这一问题，我们引入了REFLEX（基础文献实验效应复现）基准，该基准包含30个基于经典人类研究的反事实社会实验，用于比较知识判断与效应模拟。知识判断直接……（原文摘要在此处截断）

    arXiv:2611.28854v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to simulate participants in social experiments. Existing research has primarily focused on aligning social knowledge and isolated responses, while overlooking whether simulations can reproduce the collective effects of social interventions. This distinction is critical due to a failure mechanism we call a $bias$ $cascade$: similar individual states can lead to different actions, while agreement on isolated actions can still produce divergent outcomes through dynamic interactions. As a result, models may produce accurate social judgments yet fail to reproduce the outcomes of classic human studies. To investigate this problem empirically, we introduce $REFLEX$ (Reproducing Effects of Foundational Literature EXperiments), a benchmark of 30 counterfactual social experiments grounded in classic human studies that compares knowledge judgment with effect simulation. Knowledge judgment directl
    
[^125]: 隔离语言外壳：一项内容可控的多语言推理研究

    Isolating the Language Shell: A Content-Controlled Study of Multilingual Reasoning

    [https://arxiv.org/abs/2611.27053](https://arxiv.org/abs/2611.27053)

    该研究通过保持推理内容不变、仅改变教师推理呈现的语言外壳，证明多语言推理蒸馏中外壳语言的选择会显著影响学生模型的下游准确率与输出语言行为，而预训练兼容性无法预测这些效应。

    

    多语言推理蒸馏需要选择呈现教师推理所使用的语言，然而现有的方法往往将语言形式与生成、翻译和过滤方面的差异混杂在一起。我们通过将相同的、经过验证的英文推理轨迹分别渲染为英文、翻译后的目标语言以及语言混合的三种外壳，构建了内容可控的监督数据，并在三种目标语言上对两个预训练的学生模型进行微调，在两个难度级别上评估它们的多语言数学推理能力。我们发现，即使在底层推理内容保持不变的情况下，外壳的选择仍会改变下游准确率，且其影响因学生模型、语言和难度级别的不同而变化。预训练兼容性并不能一致地预测这些结果，即使测试未发现不同外壳之间存在明显的准确率差异，它们的输出语言行为仍可能存在显著差异。这些结果表明……（摘要原文在此处截断）

    arXiv:2611.27053v1 Announce Type: new  Abstract: Multilingual reasoning distillation requires choosing the language in which teacher reasoning is presented, yet existing recipes often entangle linguistic form with differences in generation, translation, and filtering. We construct content-controlled supervision by rendering the same verified English trajectories into English, translated target-language, and language-mixed shells, and fine-tune two pretrained students in three target languages, evaluating them on multilingual mathematical reasoning at two difficulty levels. Even with the underlying reasoning content held fixed, we find that shell choice can change downstream accuracy, with effects varying across students, languages, and difficulty levels. Pre-training compatibility does not consistently track these outcomes, and even when our tests find no clear accuracy difference between shells, their output-language behavior can still differ substantially. These results suggest that 
    
[^126]: 从一开始就学会搜索：面向语言模型的搜索原生预训练

    Search From the Start: Search-Native Pretraining for Language Models

    [https://arxiv.org/abs/2611.26809](https://arxiv.org/abs/2611.26809)

    提出“搜索原生预训练”方法，通过大规模地将网络搜索查询及检索到的网页响应注入预训练语料，让语言模型在预训练阶段就学会发起搜索并利用检索证据，从而在知识密集型任务上取得更优性能，且该优势在监督微调后依然保持。

    

    现代语言模型（LM）越来越多地在推理阶段配备搜索工具进行部署。然而，预训练仍然主要聚焦于对静态文本的预测，搜索能力通常是在后训练阶段才被引入。我们提出“搜索原生预训练”作为一种实用路径，在预训练阶段就基于普通文本语料教会模型搜索能力。具体而言，我们大规模地用网络搜索查询来增强预训练语料，并使用本地索引检索网络响应。在这种增强语料上的训练使模型学会生成搜索调用，并在语言建模过程中融入检索到的证据。在Wikipedia和DCLM-Edu语料库以及360M至3.3B的参数规模上，搜索预训练模型在知识密集型任务上表现出更高的性能，并且对检索到的证据具有更强的可控性（可引导性）。关键的是，在对搜索预训练模型和常规模型进行监督微调之后，它们的优势依然得以保持。

    arXiv:2611.26809v1 Announce Type: cross  Abstract: Modern language models (LMs) are increasingly deployed with access to search tools at inference. Yet pretraining remains predominantly focused on predicting static text, with search capabilities typically introduced during post-training. We propose Search-Native Pretraining as a practical route to teach search capability during pretraining from ordinary text corpora. Specifically, we augment pretraining corpus at scale with web-search queries and retrieve web response using a local index. Training on this augmented corpus teaches models to generate search calls and incorporate retrieved evidence during language modeling. Across Wikipedia and DCLM-Edu corpora and 360M to 3.3B parameter scales, search-pretrained models ("QueryLM") show improved performance on knowledge-intensive tasks and higher steerability toward the retrieved evidence. Crucially, their advantage persists after supervised fine-tuning of both search-pretrained and conve
    
[^127]: AltRA-Test：一种无需共享标注者的LLM评判者替代标注者检验

    AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators

    [https://arxiv.org/abs/2611.26073](https://arxiv.org/abs/2611.26073)

    提出了AltRA-Test统计检验方法，通过自助法重采样克服条目由不同标注者子集标注的限制，从而判断用LLM替代人类标注者在统计学上是否合理。

    

    大语言模型越来越多地被用作评判者和评分者来替代人类标注者，这引发了一个问题：这种替代在何时具有统计学上的合理性。为了支持此类决策，此前提出了替代标注者检验，适用于所有条目均由相同标注者标注的密集场景。然而，在许多大规模标注场景中，条目是由不同的标注者子集进行标注的。为此，我们提出了基于重采样标注的替代标注者检验，这是一种不要求所有条目均由相同标注者标注的统计检验方法。AltRA-Test采用自助法重采样条目，为每个条目抽取k个人类标注，并将其中一个替换为LLM标注，从而生成配对的全人类标注组与人类-LLM混合标注组。当混合标注组的标注者间一致性在统计学上不劣于全人类标注组时，即支持进行替换。

    arXiv:2611.26073v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as judges and graders to replace human annotators, raising the question of when such replacement is statistically justified. To support such decisions, the Alternative Annotator Test (Alt-Test) was introduced for dense settings where all items are annotated by the same annotators. However, in many large-scale annotation settings, items are annotated by different subsets of annotators. We therefore propose the Alternative Annotator Test with Resampled Annotations (AltRA-Test), a statistical test that does not require that all items are annotated by the same annotators. AltRA-Test uses a bootstrap that resamples items, draws k human annotations for each item, and substitutes one of them with an LLM annotation. This produces paired all-human and mixed human-LLM panels. A replacement is supported when the inter-annotator agreement (IAA) of the mixed panel is statistically non-inferior to t
    
[^128]: NONSENSE：非语义字符串揭示稠密检索中的语义性鸿沟

    NONSENSE: Non-Semantic Strings Expose Semanticity Gaps in Dense Retrieval

    [https://arxiv.org/abs/2611.22353](https://arxiv.org/abs/2611.22353)

    提出NONSENSE黑盒稀疏编辑探针，通过构造禁止复制查询词的明显非语义字符串，证明其稠密相似度得分常超过检索器最佳的查询相关段落，从而揭示了稠密检索中最近邻地位与真实语义含义相分离的语义性鸿沟。

    

    稠密检索是现代搜索、问答和检索增强生成技术的核心，其中高相似度通常被视为语义相关性的证据。我们检验了这一假设能被压榨到何种程度。NONSENSE 是一种黑盒稀疏编辑探针，它构造出表面上明显非语义的字符串，同时禁止复制查询词项，仅优化查询与该字符串之间的稠密相似度。在五个 BEIR 风格的数据集和三种流行的单向量检索器上，这些字符串的得分经常超过检索器自身最佳的已判定查询相关段落（QRP），并且常常在整个语料库中排名第一。这一结果表明，被评估的稠密检索器存在语义性鸿沟：最近邻地位可以与段落式的真实含义相分离。这些发现凸显了评估稠密检索器的重要性——不仅要看它们能否检索到相关文档，还要看它们的相似度分数有多么可靠。

    arXiv:2611.22353v1 Announce Type: cross  Abstract: Dense retrieval is central to modern search, question answering, and retrieval-augmented generation, where high similarity is often treated as evidence of semantic relevance. We examine how far this assumption can be stressed. NONSENSE is a black-box sparse-edit probe that constructs visibly non-semantic strings while forbidding query-term copying, optimizing only the dense similarity between a query and the string. Across five BEIR-style datasets and three popular single-vector retrievers, these strings frequently outscore the retriever's own best judged query-relevant passage (QRP), and often rank first in the full corpus. The result indicates a semanticity gap in the evaluated dense retrievers: nearest-neighbor status can be separated from passage-like meaning. These findings highlight the importance of evaluating dense retrievers not only by whether they retrieve relevant documents, but also by how reliably their similarity scores 
    
[^129]: FIDEL：面向埃塞俄比亚语言的元音附标文字感知预分词方法

    FIDEL: Abugida-Aware Pre-tokenization for Ethiopic Languages

    [https://arxiv.org/abs/2611.20216](https://arxiv.org/abs/2611.20216)

    本文提出 FIDEL——一族可逆的、感知埃塞俄比亚元音附标文字结构的预分词方案，通过显式揭示辅音与元音顺序结构来改进埃塞俄比亚各语言的分词质量，并在严格控制变量的条件下验证了其对分词统计与下游 NLP 任务的影响。

    

    子词分词器假设有意义的语言单位可以通过在学习到的边界处切分文本来恢复，这一假设更适合黏着型形态，而不太适合闪米特语的词根-词式系统。对于埃塞俄比亚各语言而言，这一挑战尤为严峻，因为它们的元音附标文字将辅音-元音单元编码为单个字符，从而掩盖了辅音结构。我们提出了 FIDEL（FIdäl DEcomposition and Linearization），这是一族可逆的、元音附标文字感知的预分词方案，能够显式揭示辅音和元音顺序结构，并将 SPLINTER 归约方法从辅音音素文字扩展到埃塞俄比亚元音附标文字；我们在受控条件下对三种埃塞俄比亚语言进行评估，保持语料库、架构、预算和协议固定不变，使分词方式成为唯一变量。我们报告了 BPE 和 UnigramLM 在不同词表规模下的分词器内在统计指标，以及下游 NLP 任务的结果（包括迁移场景）。我们发现……

    arXiv:2611.20216v1 Announce Type: new  Abstract: Subword tokenizers assume that meaningful linguistic units can be recovered by splitting text at learned boundaries, an assumption better suited to concatenative morphology than to Semitic root-and-pattern systems. The challenge is acute for the Ethiopic languages, whose abugida script encodes consonant-vowel units as single characters and hides consonantal structure. We introduce FIDEL (FId\"al DEcomposition and Linearization), a family of reversible abugida-aware pre-tokenization schemes that expose consonant and vowel-order structure, extend SPLINTER reductions from abjad scripts to the Ethiopic abugida, and evaluate them on three Ethiopic languages under controlled conditions, holding corpus, architecture, budget and protocol fixed so that tokenization is the only variable. We report intrinsic tokenizer statistics for BPE and UnigramLM across vocabulary sizes alongside downstream NLP tasks, including transfer scenarios. We find that 
    
[^130]: LLM推理中对话历史的隐性代价

    The Silent Cost of Conversation History in LLM Reasoning

    [https://arxiv.org/abs/2611.20071](https://arxiv.org/abs/2611.20071)

    研究发现累积的对话历史会显著降低LLM在多轮任务中的推理准确率，且过程级质量指标无法提前预警这种下降，而用经求解器验证的状态摘要替代完整历史可挽回大部分性能损失。

    

    在多轮信息系统中，大语言模型（LLM）会不断累积自身早期的回复以及持续增长的任务状态。我们探究了这种对话历史是否会降低准确率，以及过程级质量指标能否在准确率下降之前发出预警。每项分析均在计算结果之前预先设定。在研究1中，针对48个分片数学任务，保留模型自身回复使GPT-4o-mini的最终成功率降低0.21，Mistral-7B降低0.29（经Holm校正的p值分别为0.026和0.004）；Llama-4 Scout呈现相同趋势，但不显著。研究2引入了CDR-Stream基准，包含100轮状态追踪对话，每轮均配有精确答案并针对各模型进行了难度校准。在相同难度下，使用完整历史时，准确率在二十轮内从约0.6降至0.1；而使用经求解器验证的状态摘要，准确率可保持高出0.52至0.59（经Holm校正的p值小于0.001）。一致性与逻辑连贯性的下降幅度远小于准确率，而稳定性反而有所上升。

    arXiv:2611.20071v1 Announce Type: cross  Abstract: Large language models (LLMs) in multi-turn information systems accumulate their own earlier replies and a growing task state. We asked whether this conversation history lowers accuracy, and whether process-level quality metrics warn before it does. Each analysis was fixed before its outcomes were computed. In Study 1, on 48 sharded math tasks, retaining its replies lowered final success by 0.21 for GPT-4o-mini and 0.29 for Mistral-7B (Holm-adjusted p = 0.026 and 0.004); Llama-4 Scout showed the same direction, not significant. Study 2 introduces CDR-Stream, 100-turn state-tracking dialogues with an exact answer per turn and per-model difficulty calibration. With the full history, accuracy fell from about 0.6 to 0.1 within twenty turns at matched difficulty, whereas a solver-verified state summary kept it 0.52-0.59 higher (Holm-adjusted p less than 0.001). Consistency and logical coherence fell far less than accuracy, and stability rose
    
[^131]: 展望智能体机器人：机器人系统的新时代

    Envisioning Agentic Robotics: A New Era of Robotic Systems

    [https://arxiv.org/abs/2611.18359](https://arxiv.org/abs/2611.18359)

    本文提出“智能体机器人”这一新范式，主张由基础模型与具身框架构成的“具身系统2”是智能体机器人的决策核心，并呼吁学界建立统一定义、评估框架与研究路线图。

    

    GPT-6 Astra 的发布标志着机器人学新范式——智能体机器人的诞生。短短几周内，配备智能体框架的前沿模型在许多具身基准测试中达到甚至超越了任务专用策略的表现。与为每一类任务训练专用策略不同，智能体机器人系统将基础模型应用于开放世界的具身任务。它以闭环方式运行：观察环境、理解用户意图、对任务进行推理、调用技能与工具、验证结果并调整策略。尽管取得了这些进展，学界尚未在统一定义、可靠的评估框架或研究路线图上达成共识。我们认为，由基础模型与具身框架共同实现的“具身系统 2”构成了智能体机器人的决策核心，其核心挑战在于将通用推理转化为在……（原文摘要在此处截断）

    arXiv:2611.18359v1 Announce Type: cross  Abstract: The release of GPT-6 Astra marks the emergence of a new paradigm in robotics: agentic robotics. Within weeks, frontier models with agentic harnesses matched or exceeded task-specific policies on many embodied benchmarks. Instead of training a dedicated policy for each family of tasks, an agentic robotic system applies a foundation model to open-world embodied tasks. It operates in a closed loop: observing the environment, interpreting user intent, reasoning about the task, invoking skills and tools, verifying outcomes, and adapting its strategy. Despite these advances, the community has yet to converge on a shared definition, a reliable evaluation framework, or a research roadmap. We argue that Embodied System 2, realized by a foundation model together with an embodied harness, forms the decision-making core of agentic robotics. Its central challenge is to translate general-purpose reasoning into effective and verifiable actions under 
    
[^132]: 不确定性在哪里被观测？对大语言模型不确定性量化的机制性理解

    Where Is Uncertainty Observed? A Mechanistic Understanding of Uncertainty Quantification in LLMs

    [https://arxiv.org/abs/2611.15735](https://arxiv.org/abs/2611.15735)

    该论文基于机制可解释性提出了观测不确定性的四种透镜框架（结构、表征、分布、语义），并揭示了观测点决定不确定性量化估计器的信息获取能力与恢复效果这一关键机制。

    

    了解大语言模型（LLM）的响应何时可以被信任对于其安全使用至关重要。不确定性量化（UQ）可以帮助LLM识别其响应是否可能不可靠。现有的分类体系按不确定性来源或估计机制来组织UQ方法，但尚未解决一个关键问题：在模型的计算过程中，不确定性究竟在哪里被观测。这个问题的答案非常重要，因为观测点决定了估计器是否能够获得足够的信息来恢复给定目标，以及其恢复能力有多强。借助机制可解释性研究，我们定义了四种机制观测透镜：结构透镜观测组件更新，表征透镜观测隐藏状态，分布透镜观测词元概率，语义透镜观测意义空间上的分布。我们通过残差流将这些透镜关联起来，并按它们所携带的信息量对其进行排序。

    arXiv:2611.15735v1 Announce Type: cross  Abstract: Knowing when a large language model's (LLM's) response can be trusted is critical for its safe use. Uncertainty quantification (UQ) can help LLMs recognize if their responses may be unreliable. Existing taxonomies organize UQ methods by uncertainty source or estimation mechanism. They leave unresolved the question of where uncertainty is observed in the model's computation. The answer matters because the observation point determines whether an estimator has access to enough information to recover a given target, and how well it can recover it. Drawing on mechanistic interpretability, we define four mechanistic observation lenses: the structural lens observes component updates, the representational lens observes hidden states, the distributional lens observes token probabilities, and the semantical lens observes distributions over meanings. We relate these lenses through the residual stream and order them by the information they carry, 
    
[^133]: 专业翻译工作流中的数字主权：从Windows迁移到Linux的案例研究

    Digital sovereignty in professional translation workflows: a case study of migrating from Windows to Linux

    [https://arxiv.org/abs/2610.41561](https://arxiv.org/abs/2610.41561)

    本研究提出以“控制、连续性、退出”三原则结合三项实践要求的数字主权评估框架，并通过将译者工作站从Windows迁移至Linux（采用LoopCAT、LibreOffice和本地运行的TranslateGemma）的案例研究，证明了专业翻译工作流可在可本地运行的Linux核心上实现重构。

    

    本研究通过筛选计算机辅助翻译工具和翻译管理系统，并以一个将译者工作站从Windows迁移到Linux的案例研究，考察了专业翻译中的数字主权问题。该框架结合三项原则——控制、连续性与退出——以及三项实践要求：Linux本地原生运行、独立于强制性外部服务，以及对关键数据和资源的独立访问。基于文献证据和从服务商处获得的有限回应，生态系统筛选区分了“Linux可访问性”与“运营独立性”：基于浏览器的平台可能仍然依赖于服务商控制的基础设施。迁移案例展示了围绕可本地运行的Linux核心重构专业翻译工作流的可行性，使用了LoopCAT、LibreOffice和本地运行的TranslateGemma，以及其他工具……

    arXiv:2610.41561v1 Announce Type: new  Abstract: This study examines digital sovereignty in professional translation through a screening of computer-assisted translation tools and translation management systems, and a case study of migrating a translator's workstation from Windows to Linux. The framework combines three principles - Control, Continuity and Exit - with three practical requirements: native Linux operation, independence from mandatory external services, and independent access to essential data and resources. Drawing on documentary evidence and the limited number of responses received from providers, the ecosystem screening distinguishes Linux accessibility from operational independence: browser-based platforms may remain dependent on provider-controlled infrastructure. The migration case demonstrates the feasibility of reconstructing one professional workflow around a locally operable Linux core, using LoopCAT, LibreOffice and locally run TranslateGemma, alongside tools fo
    
[^134]: 图式连贯性量表：测量被赋予的信念是否组织了语言模型的认知

    The Schema Coherence Inventory: Measuring Whether an Assigned Belief Organizes a Language Model's Cognition

    [https://arxiv.org/abs/2610.31683](https://arxiv.org/abs/2610.31683)

    本文提出图式连贯性量表（SCI），一种基于贝克认知模型的96题强制选择、无生成评分工具，用于验证赋予语言模型的信念是否真正组织其认知而非仅仅引发情绪，并通过目标特异性判别标准在五个骨干模型上得到验证。

    

    人格条件化的语言模型被用作模拟患者和情感代理，但目前尚无经过验证的方法来判断赋予此类模型的信念是否真正组织了其行为，还是该模型仅仅采纳了一种情绪。我们提出了图式连贯性量表（SCI）：包含96个强制选择题项，覆盖贝克认知模型的八个层面，每个题项提供六个选项，分别对应三种消极自我图式及其积极对应图式，并通过长度归一化的对数似然进行评分，因此无需生成或解析任何文本。由于单纯的情感就可能使模型转向一个默认信念，判别标准是目标特异性：为某一信念构建的条件必须比为其同极性其他信念构建的条件更频繁地选中该信念。在五个骨干模型（参数量为17亿至140亿）上开展的六项验证研究、共107个条件，支持了这一解读。九名接受过临床培训的评分者能以79.5%的准确率识别出所标注的信念（随机水平为16.7%）。

    arXiv:2610.31683v1 Announce Type: new  Abstract: Persona-conditioned language models are used as simulated patients and affective agents, but there is no validated way to ask whether a belief assigned to such a model organizes its behaviour or whether the model has merely adopted a mood. We present the Schema Coherence Inventory (SCI): 96 forced-choice items at eight levels of Beck's cognitive model, each offering six replies keyed to three negative self-schemas and their positive counterparts, scored by length-normalized log-likelihood so that nothing is generated or parsed. Affect alone can move a model onto a default belief, so the discriminant criterion is target specificity: a condition built for one belief must select that belief more often than conditions built for the other beliefs of its pole. Six validation studies over 107 conditions on five backbones (1.7B-14B parameters) support this reading. Nine clinically trained raters recover the keyed belief at 79.5% (chance 16.7%). 
    
[^135]: 用自然语言解释国际象棋与围棋

    Explaining Chess and Go in Natural Language

    [https://arxiv.org/abs/2610.23166](https://arxiv.org/abs/2610.23166)

    本文提出了基于引擎评估的解释性指标和无需训练的BinProp方法，使语言模型能够发现可复用的棋类概念并以自然语言解释着法，平均得分相比直接提示提高近两倍，且该方法可泛化至围棋、跳棋和将棋等多种棋类。

    

    现代国际象棋引擎的棋力远超人类，其选择着法背后的思路往往令人类难以理解。我们提出了一种基于国际象棋引擎评估的自动化指标，用于评估棋局解释的质量。利用该指标对最先进的语言模型进行评估后发现，它们的表现出奇地差，往往依赖于对棋盘的表面观察，而非稳健、可迁移的国际象棋概念。随后，我们提出了BinProp，这是一种无需训练的方法，能够使同样的模型发现可复用的国际象棋概念，并用它们以自然语言解释着法。BinProp提升了对所有受测模型的可解释性，与直接提示相比，平均得分几乎提高了两倍。除了国际象棋之外，同一方法无需额外调整即可在围棋、西洋跳棋和将棋中发现有用的概念。

    arXiv:2610.23166v1 Announce Type: cross  Abstract: Modern chess engines play far beyond human ability, often choosing moves whose underlying ideas are difficult for humans to understand. We propose an automatic metric for evaluating chess explanations, grounded in chess-engine evaluations. Evaluating state-of-the-art language models using this metric, we find that they perform surprisingly poorly, often relying on surface-level board observations rather than robust, transferable chess concepts. We then introduce BinProp, a training-free method that enables these same models to discover reusable chess concepts and use them to explain moves in natural language. BinProp improves explainability for every model tested, nearly tripling the average score compared with direct prompting. Beyond chess, the same method discovers useful concepts in Go, draughts, and shogi without additional tuning.
    
[^136]: 稀疏自编码器表示的行为组合性

    Behavioral Compositionality of Sparse Autoencoder Representations

    [https://arxiv.org/abs/2610.21261](https://arxiv.org/abs/2610.21261)

    论文提出SAE可加性残差（SAR）来度量稀疏自编码器特征在行为层面的组合性，发现学习到的SAE坐标比随机旋转具有更强的一阶可加性，但仍存在不可忽略的高阶交互项。

    

    稀疏自编码器（SAE）使用稀疏的、线性解码的特征来表示模型激活，但这些特征的下游效应不一定是按加法方式组合的。我们引入了SAE可加性残差（SAR），用于衡量在选定交互阶数以上未能被解释的行为变异。在独立特征子集采样的条件下，总体SAR恰好等于该阶数以上交互的归一化能量，并为最优低阶近似的联盟排序误差提供了上界。为了区分与SAE表示本身相关的组合性与由下游计算所诱导的组合性，我们将学习到的SAE坐标与同一所选子空间的保和随机旋转进行比较，发现在所有18个模型和行为设置中，学习到的坐标都具有更低的一阶残差。然而，这些坐标并非完全可加：在每一设置中，成对交互项均能改善留出集上的联盟排序。

    arXiv:2610.21261v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) represent model activations with sparse, linearly decoded features, but the downstream effects of those features need not compose additively. We introduce SAE Additivity Residuals (SAR), which measure the behavioral variation left unexplained above a chosen interaction order. Under independent feature subset sampling, population SAR is exactly the normalized energy of interactions above that order and bounds coalition ordering error for the optimal low order approximation. To distinguish compositionality associated with the SAE representation from that induced by downstream computation, we compare learned SAE coordinates with sum preserving random rotations of the same selected subspace and find lower first order residuals for the learned coordinates across all 18 model and behavior settings. The coordinates are nevertheless not fully additive: pairwise terms improve held out coalition ordering in every setti
    
[^137]: 稀疏自编码器中的表征再分配

    Representation Redistribution in Sparse Autoencoders

    [https://arxiv.org/abs/2610.21240](https://arxiv.org/abs/2610.21240)

    该论文提出再分配敏感度分数（RSS）来识别在提示变换下以一致方向重复出现的大幅特征响应，并在涵盖八个语言模型的27个预训练SAE上验证了表征再分配的可复现性，同时将重复的表征变化与下游读出贡献及决策失败精确解耦。

    

    稀疏自编码器将语言模型激活分解为稀疏特征，但旨在保持底层请求不变的提示变换可能会使激活活动在这些特征之间重新分布。在某一个变换后提示上观察到的特征变化未必会再次出现，即使是重复出现的变化，也未必会影响下游计算。我们引入了再分配敏感度分数，用以识别以一致方向重复出现的大幅特征响应，并在未参与选择过程的变体上评估由此得到的支持集。在涵盖八个语言模型的27个预训练SAE上，所测试的提示变换在这些被排除的变体上产生了可复现的再分布特征。随后，我们将重复出现的表征变化与带符号的读出贡献以及决策失败区分开来。对于固定的线性读出，一个精确分解展示了响应幅度、读出系数、方向对齐以及……

    arXiv:2610.21240v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) decompose language model activations into sparse features, but prompt transformations designed to preserve the underlying request can redistribute activity across them. A feature change observed on one transformed prompt need not recur, and even a recurring change need not affect a downstream computation. We introduce the redistribution sensitivity score (RSS), which identifies large feature responses that recur in a consistent direction, and evaluate the resulting supports on variants excluded from selection. Across 27 pretrained SAEs spanning eight language models, the tested prompt transformations produce a reproducible redistribution profile on these excluded variants. We then separate recurring representation change from signed readout contribution and decision failure. For a fixed linear readout, an exact decomposition shows how response magnitude, readout coefficients, directional alignment, and the cl
    
[^138]: 冻结因子还是频谱带？在LoRA中，低秩时因子更为重要

    Frozen Factor or Spectral Band? In LoRA, the Factor Matters More at Low Rank

    [https://arxiv.org/abs/2610.06621](https://arxiv.org/abs/2610.06621)

    本文通过2x2交叉实验设计发现，在低秩（1-8）少步数设置下，LoRA变体冻结哪个因子（输入因子A显著优于输出因子B）比其选择的奇异方向带（主导或次主导）对性能影响更大。

    

    在两个主要的任务-模型组合上，在秩1到8、优化器步数较少的设置下，LoRA变体所冻结的因子比其奇异方向带更为重要。MiCA和PiCa通常按频谱带（次主导与主导奇异方向）来比较，但在我们的记号下，MiCA冻结输出因子B，而PiCa冻结输入因子A。我们在1B到8B参数规模的模型上，通过2x2交叉实验设计同时考察了因子和带两个维度。在两个带上，冻结A均优于冻结B：在格式任务（我们的主要证据）的秩2上提升12到20个百分点，在OpenBookQA的秩8上仍提升约7个百分点。事后补充的针对258项主要检验的Holm校正显示，这一优势在16项检验中的15项中依然成立，且因子相对于带的优势在秩1、2和4上保持领先（格式任务的秩1本身也是事后检验）。在秩8上，该优势通过了自身检验但未通过校正检验。该优势在秩2上用两个7B模型得到重复验证，提升11到13个百分点。在校正之外，该优势还在GSM8K上用1B模型得到了重复（原文在此处截断）。

    arXiv:2610.06621v2 Announce Type: replace-cross  Abstract: On two main task-model pairs, at ranks 1 to 8 with few optimiser steps, the factor a LoRA variant freezes matters more than its band of singular directions. MiCA and PiCa are compared by band, minor against dominant, yet MiCA freezes the output factor B and PiCa, in our notation, the input factor A. We cross factor and band in a 2 x 2 design on models from 1B to 8B parameters. Freezing A beats freezing B on both bands, by +12 to +20 points at rank 2 on a format task, our primary evidence, and still by about +7 points at rank 8 on OpenBookQA. A Holm correction over 258 primary tests, added post hoc, keeps this advantage in 15 of 16 tests and the lead of factor over band at ranks 1, 2 and 4, rank 1 of the format task being itself post hoc. At rank 8 that lead passes its own test but not the correction. The advantage repeats at rank 2 with two 7B models, by +11 to +13 points. Outside the correction, it repeats on GSM8K with a 1B m
    
[^139]: GlitchPatch：通过局部重新分词修复冻结语言模型中的故障词元

    GlitchPatch: Repairing Glitch Tokens in Frozen Language Models via Local Retokenization

    [https://arxiv.org/abs/2610.04399](https://arxiv.org/abs/2610.04399)

    GlitchPatch提出了一种基于局部重新分词的修复框架，无需访问模型内部即可在冻结的大语言模型外部通过优化输入分词来修复故障词元，并揭示了BPE合并规则删除中修复率与对正常词元附带损害之间的权衡规律。

    

    故障词元（Glitch tokens）是词表中的一种异常词条，可能导致大语言模型（LLM）产生与输入不一致的输出。现有的修复方法需要访问模型内部结构，这使得它们对于冻结的模型检查点并不实用。我们研究了能否通过优化输入分词，在模型外部修复故障词元。针对BPE合并规则删除的实证研究表明：（1）删除故障词元的合并规则可以修复相当大比例的失败情况，但会破坏共享中间合并节点的正常词元，导致整体故障率上升；（2）不同的分解粒度会产生非单调的修复率，而对正常词元的附带损害则单调增长。基于这些发现，我们提出了GlitchPatch——一个基于局部重新分词的冻结语言模型修复框架，该框架包含两个阶段：离线阶段使用行为路径优化……（摘要在此处被截断）

    arXiv:2610.04399v2 Announce Type: replace-cross  Abstract: Glitch tokens are anomalous vocabulary entries that can cause large language models (LLMs) to produce outputs inconsistent with their inputs. Existing repair methods require access to model internals, making them impractical for frozen checkpoints. We investigate whether glitch tokens can be repaired outside the model by optimizing the input tokenization. An empirical study on BPE merge-rule deletion reveals that (1)deleting a glitch token's merge rule can fix a substantial fraction of failures, yet disrupting normal tokens sharing intermediate merge nodes causes the overall glitch rate to rise, and (2)different decomposition granularities yield non-monotonic fix rates while collateral damage on normal tokens grows monotonically. Motivated by these findings, we propose GlitchPatch, a repair framework for frozen language models based on local retokenization, consisting of two stages: the offline stage uses Behavioral Path Optimi
    
[^140]: 语言模型需要可训练的输入嵌入表吗？17亿参数级规模下的固定最小标记编码

    Do Language Models Need a Trainable Input Embedding Table? Fixed Minimal Token Codes at 1.7B-Class Scale

    [https://arxiv.org/abs/2610.04002](https://arxiv.org/abs/2610.04002)

    该研究证明在17亿参数规模下，语言模型的输入嵌入表可以完全用固定的16位标记编码（通过重复扩展至2048维）替代可训练嵌入，仅产生几个百分点的性能差距，其中行归一化编码在固定输入方案中表现最佳。

    

    在语言模型学习如何使用一个标记之前，该标记必须能够被识别。那么它的输入坐标是否也必须通过学习获得？我们将解码器专用Transformer的可训练输入表替换为预先确定的16位标识符，为49,152个标记中的每一个分配固定编码，并通过重复将编码提升至2048宽度，而无需任何学习到的投影。我们从零开始训练了五个模型，每个模型训练至1000亿个预测目标：一个可学习输入的对照组，以及四个固定输入的变体——分别使用规范编码、可逆重编码、额外的Transformer块或行归一化编码。在训练终点，四个固定输入模型分别达到51.44%-53.34%的HellaSwag归一化准确率、70.46%-71.49%的PIQA准确率以及42.29%-44.52%的LAMBADA准确率。其中行归一化模型在全部三项基准上均达到固定输入变体的上限；而可学习输入的对照组分别取得57.79%、72.69%和47.72%的成绩。在深度和参数数量保持不变的情况下，行归一化模型相比规范编码在LAMBADA上的优势为1.80个百分点。

    arXiv:2610.04002v2 Announce Type: replace  Abstract: A token must be identifiable before a language model can learn how to use it. Must its input coordinates also be learned? We replace the trainable input table of a decoder-only Transformer with a predetermined 16-bit identifier for each of 49,152 tokens. Repetition lifts the code to width 2048 without a learned projection. We train five models from scratch to 100B prediction targets each: a learned-input control and four fixed-input variants using canonical codes, an invertible recoding, an extra Transformer block, or row-normalized codes. At the endpoint, the four fixed-input models attain 51.44-53.34% HellaSwag normalized accuracy, 70.46-71.49% PIQA accuracy, and 42.29-44.52% LAMBADA accuracy. The normalized model attains the upper endpoint on all three; the learned control scores 57.79%, 72.69%, and 47.72%, respectively. At unchanged depth and parameter count, the normalized model's LAMBADA advantage over canonical codes is 1.80 p
    
[^141]: 语言模型能否学会预测股票价格

    Can Language Models Learn to Forecast Stock Prices

    [https://arxiv.org/abs/2609.36914](https://arxiv.org/abs/2609.36914)

    该论文在按时间顺序推进的股票价格沙盒中，通过对 Qwen3-4B 进行 SFT 与 PPO 后训练，探索语言模型能否学会利用价格、成交量等信息预测股票未来收益。

    

    后训练已被证明能显著提升语言模型在具有可验证结果任务上的表现，包括数学推理、软件工程和计算机使用。然而，同样的方法能否改善金融市场的预测则远非明确。与具有可验证结果的任务相比，不仅实际收益充满噪声，甚至什么构成进行有效预测所需的相关信息集也并非先验可知：模型必须自行决定收集哪些观察信息，然后在结果揭晓之前给出数值判断。我们在一个按时间顺序推进的股票价格沙盒中研究这一问题，其中语言模型收集价格、成交量、相对表现以及市场背景等证据，并预测未来收益。我们首先基于工具使用演示对 Qwen3-4B 进行监督微调（SFT），随后使用近端策略优化（PPO）进行训练，其终端奖励由……

    arXiv:2609.36914v1 Announce Type: new  Abstract: Post-training has been shown to significantly improve language models' performance on tasks with verifiable outcomes, including mathematical reasoning, software engineering, and computer use. However, whether the same approach can improve forecasting in financial markets is much less clear. Compared with tasks with verifiable outcomes, not only are realized returns noisy, but even what constitutes a relevant information set for making effective predictions is not obvious a priori: the model must decide which observations to gather and then commit to a numerical judgment before the outcome is known. We study this question in a chronological stock-price sandbox, where a language model gathers price, volume, relative-performance, and market-context evidence and predicts a future return. We post-train Qwen3-4B with supervised fine-tuning (SFT) on tool-use demonstrations, then proximal policy optimization (PPO) with a terminal reward given by
    
[^142]: 大规模因子分析表明机器智能只能被部分解释

    Large-scale factor analysis shows machine intelligence is only partially interpretable

    [https://arxiv.org/abs/2609.36515](https://arxiv.org/abs/2609.36515)

    本论文对1,618个语言模型在456个纯文本基准上的13,251个评测分数进行了前所未有的大规模因子分析，发现语言模型的智能仅能被部分解释——一般智能因子约解释70.8%的方差，其余表现差异由领域特定的潜在因素驱动。

    

    语言模型开发中的一个常见假设是，认知能力围绕一个一般的、跨领域的智能因子组织，类似于人类的流体智力。这一假设很少被直接检验，以往的尝试也仅在更小的规模上进行。我们采用潜变量的方法来研究语言模型的智能，这与心理测量学家研究心理构念的方式类似。在每一个具体问题集上的表现都受到一个领域特定潜因子和一个领域无关潜因子的影响。我们将因子分析作为降维技术，分析了13,251个已发表的评测分数，涵盖1,618个语言模型和456个不同的纯文本基准。由于该数据集具有超稀疏的特性，我们通过不同的数据稠密化方法和插补方法对分析结果进行三角验证。在不同偏差模式下呈现出的稳健规律是：1. 一个一般智能因子可解释70.8……（摘要截断）

    arXiv:2609.36515v1 Announce Type: cross  Abstract: A common assumption in language model development is that cognitive abilities are organized around a general, domain-free intelligence factor, like fluid intelligence in humans. This assumption is rarely tested directly, and prior attempts have done so only at a much smaller scale. We take a latent variable approach to intelligence in language models, similar to how psychometricians study psychological constructs. Performance in every specific problem set is influenced by a domain-specific and a domain-agnostic latent factor. Using factor analysis as a dimension-reduction technique, we analyzed 13,251 published evaluation scores covering 1,618 language models across 456 different text-only benchmarks. Due to the super-sparse nature of the dataset, we triangulate our analysis across different data densifiers and imputation methods. A robust pattern across different modes of bias is that 1. A general intelligence factor accounts for 70.8
    
[^143]: AdaTutoRank：通过自适应辅导优化学习文档集重排序，面向RAG与深度研究

    AdaTutoRank: Learning to Rerank Document Sets via Adaptive Tutoring Optimization for RAG and Deep Research

    [https://arxiv.org/abs/2609.32472](https://arxiv.org/abs/2609.32472)

    该论文提出AdaTutoRank，针对文档集重排序中集合级标量奖励导致的监督稀疏与信用分配困难问题，通过自适应辅导优化为不同质量的rollout提供差异化指导，从而为RAG和深度研究筛选出完整、互补、无冗余的文档集合。

    

    在RAG和深度研究中，文档重排序器决定了哪些证据会传递给下游模型，然而主流的重排序器通过相关性匹配进行选择，而单独相关的文档很少能构成复杂信息需求所需的完整、互补且无冗余的集合。先前的工作通过集合的总体评分（rubric score）来奖励整个集合，将目标从文档排序转移到了集合组合。然而该分数是集合中所有文档共享的一个标量，因此监督信号是稀疏的：只要集合得分高，冗余文档就会与其余文档一同获得奖励；而只要集合得分低，起决定性作用的文档也会与其余文档一同受到惩罚；这种信用分配机制使得真正的贡献者与搭便车者无法被区分开来。在策略蒸馏（on-policy distillation）可以加密这种监督信号，但现有方法对每一次 rollout 都给予相同的固定指导，对强的 rollout 而言过于死板，对弱的 rollout 而言又过于抽象。因此，我们提出了 AdaTutoRank，一种（摘要在此处截断）

    arXiv:2609.32472v2 Announce Type: replace  Abstract: Document rerankers determine what evidence reaches the downstream model in RAG and deep research, yet mainstream rerankers select by relevance matching, and individually relevant documents rarely constitute the complete, complementary, non-redundant set a complex information need demands. Prior work rewards a set by its aggregate rubric score, shifting the objective from ranking documents to composing sets. Yet that score is one scalar shared by every document in the set, so the supervision is sparse: a redundant document is rewarded with the rest whenever the set scores well, and a decisive one penalized with the rest whenever it does not; credit assignment leaves contributors indistinguishable from free riders. On-policy distillation could densify this supervision, but existing methods give every rollout the same fixed guidance, too prescriptive for strong rollouts and too abstract for weak ones. We therefore propose AdaTutoRank, a
    
[^144]: 注意力路由早期即稳定：循环语言模型的工作集推理

    Attention Routing Stabilizes Early: Working-Set Inference for Recurrent Language Models

    [https://arxiv.org/abs/2609.27373](https://arxiv.org/abs/2609.27373)

    该论文发现循环语言模型的注意力路由在早期循环步骤即趋于稳定，据此提出无需训练的WISE推理方法：早期用全局注意力发现稀疏工作集，后续步骤重用该支撑集，在保持推理质量的同时大幅减少重复的全局注意力计算。

    

    循环语言模型通过反复应用共享的网络模块来精炼潜在表示，但标准推理在每个循环步骤都会重新计算全局注意力。我们研究了注意力在循环深度上的动态变化，发现注意力的支撑集和分布比隐藏状态和注意力输出更早地稳定下来。这提示了一种两阶段结构：早期步骤发现相关上下文的稀疏工作集，而后续步骤则在基本相同的路由支撑上精炼表示。受此结构启发，我们提出了 WISE（基于支撑集利用的工作集推理），这是一种无需训练的方法，它在早期循环阶段使用不受限制的全局注意力，而在后续阶段重用直接发现的块结构化支撑，同时保持循环深度和支撑内注意力计算的动态性。受控干预实验表明，循环中的发现过程非常重要，而仅重用支撑

    arXiv:2609.27373v1 Announce Type: new  Abstract: Recurrent language models repeatedly apply shared network blocks to refine latent representations, but standard inference recomputes global attention at every recurrent step. We study attention dynamics across recurrent depth and find that attention support and distributions stabilize substantially earlier than hidden states and attention outputs. This suggests a two-stage structure: early steps discover a sparse working set of relevant context, while later steps refine representations over largely the same routing support. Motivated by this structure, we introduce WISE (Working-set Inference with Support Exploitation), a training-free method that uses unrestricted global attention during early recurrence and later reuses directly discovered block-structured support while keeping recurrent depth and within-support attention computation dynamic. Controlled interventions show that recurrent discovery is important and that support-only reus
    
[^145]: 用于基于证据的遗传疾病严重程度分类的大语言模型智能体

    Large Language Model Agents for Evidence Based Genetic Disease Severity Classification

    [https://arxiv.org/abs/2609.19569](https://arxiv.org/abs/2609.19569)

    该研究开发了一个结合ReAct与RAG的自主AI智能体，基于ACMG严重程度指南和ACOG生活质量标准检索并验证文献证据，实现了对10,211个人类表型本体术语的遗传病严重程度自动化分类，表型分类准确率达93.55%，并汇总基因层面的严重程度以识别严重的常染色体隐性遗传基因对。

    

    遗传疾病的严重程度分类是主观且劳动密集的，这在基因组筛查中造成了瓶颈，而商业基因面板在规模和覆盖范围上差异很大。我们开发了一个自主AI智能体，将推理与行动（ReAct）与检索增强生成（RAG）相结合，对10,211个人类表型本体（HPO）术语进行分类。该智能体使用美国医学遗传学学会（ACMG）认可的严重程度指南和美国妇产科医师学会（ACOG）的生活质量标准来检索PubMed文献，生成可解释的推理链，并独立验证论断。在表型层面，基于专家精心整理的队列，该智能体达到了93.55%的准确率（MCC 0.9237），其中82.6%至91.4%的论断得到直接证据或有效推论的支持。基因层面的严重程度在8,738对基因中进行了汇总，识别出3,283对表现为严重或极严重的常染色体隐性遗传基因对。

    arXiv:2609.19569v1 Announce Type: cross  Abstract: Disease severity classification for genetic conditions is subjective and labor-intensive, creating bottlenecks in genomic screening, where commercial panels vary widely in size and overlap. We developed an autonomous AI agent integrating Reasoning and Acting (ReAct) with Retrieval-Augmented Generation (RAG) to classify 10,211 Human Phenotype Ontology terms. It uses American College of Medical Genetics (ACMG)-endorsed severity guidelines and American College of Obstetricians and Gynecologists (ACOG) quality-of-life criteria to retrieve PubMed literature, generate interpretable reasoning chains, and independently verify claims. At the phenotype level, using expert-curated cohorts, the agent achieved 93.55% accuracy (MCC 0.9237) with 82.6% to 91.4% of claims supported by direct evidence or valid inferences. Gene-level severity was aggregated across 8,738 pairs, identifying 3,283 autosomal recessive pairs with severe or profound presentati
    
[^146]: 可靠性感知的性别歧视检测：结合DPO与标注者一致性和词级置信度评分

    Reliability-Aware Sexism Detection: Combining DPO with Annotator Agreement and Token-Level Confidence Scoring

    [https://arxiv.org/abs/2608.12330](https://arxiv.org/abs/2608.12330)

    本文提出RA-DPO方法，通过整合标注者一致性、模型置信度和词级不确定性来提升性别歧视检测的可靠性，并证明在少量高可靠性数据上训练即可达到全数据训练的同等效果。

    

    在线性别歧视的检测仍然是一个未解决的问题。性别歧视检测本质上是主观的，但大多数现有系统将多标注者的标签简化为单一的多数决策，并统一对待所有实例。这忽略了两个信息丰富的信号：标注者一致性和模型不确定性。我们提出了RA-DPO（可靠性感知直接偏好优化），它将标注者一致性、模型置信度和词级不确定性信号整合为一个单一的可靠性评分。RA-DPO利用该评分在训练期间选择高价值的偏好对，并支持推理时的弃权，从而使模型能够以覆盖率为代价换取准确性。我们在EXIST 2023的6,920条多语言帖子上评估了RA-DPO，通过DPO微调OpenAI gpt-4o基础模型，并在两个开放权重的3B模型（Llama、Qwen）上进行了验证。结果表明，在最高30%最可靠的配对上进行训练与全数据DPO效果相当，这表明可靠性感知方法能够高效利用数据。

    arXiv:2608.12330v1 Announce Type: new  Abstract: The detection of online sexism remains an open problem. Sexism detection is inherently subjective, yet most existing systems reduce multi-annotator labels to a single majority decision and treat all instances uniformly. This ignores two informative signals: annotator agreement and model uncertainty. We propose RA-DPO (Reliability-Aware Direct Preference Optimization), which integrates annotator agreement, model confidence, and a token-level uncertainty signal into a single reliability score. RA-DPO uses this score to select high-value preference pairs during training and to support inference-time abstention, which allows the model to trade coverage for accuracy. We evaluate RA-DPO on 6,920 multilingual posts from EXIST 2023, fine-tune OpenAI gpt-4o base via DPO, and validate on two open-weight 3B models (Llama, Qwen). Results show that training on the top 30% most reliable pairs matches full-data DPO, which indicates that reliability-awa
    
[^147]: AVA-编码器：迈向智能体原生的视频表示学习

    AVA-Encoder: Towards Agent-Native Video Representation Learning

    [https://arxiv.org/abs/2608.12313](https://arxiv.org/abs/2608.12313)

    本文提出AVA-Encoder框架，通过智能体自编码将视频转换为知识图谱表示并重建，实现了智能体可直接理解、查询和编辑的视频表示学习，克服了创意智能体从电影中学习的障碍。

    

    摘要：arXiv:2608.12313v1 公告类型：交叉 摘要：创意智能体仍缺乏从高质量人类电影中学习的有效方式，这限制了它们制作电影级视频的能力。一个关键挑战是缺乏一种既忠实于电影内容又可直接用于智能体推理和操作的视频结构化表示。为应对这一挑战，我们提出了智能体视频自编码器（AVA-Encoder），一种通过智能体自编码学习智能体原生视频表示的框架。AVA-Encoder 将视频转换为知识图谱表示，然后将其重建回视频。其层次结构和状态节点存储结构化文本，而关联的资源层则保存生成的图像、音频和视频。类型化边以智能体易于理解、查询和编辑的形式保留了这些文本描述与资源之间的关系。视频重建差异驱动了一个文本梯度优化框架，该框架...

    arXiv:2608.12313v1 Announce Type: cross  Abstract: Creative agents still lack an effective way to learn from high-quality human films, limiting their ability to produce cinematic-grade videos. A key challenge is the absence of a structured video representation that is both faithful to film content and directly usable for agentic reasoning and manipulation. To address the challenge, we propose the Agentic Video Auto-Encoder (AVA-Encoder), a framework for learning agent-native video representations via agentic auto-encoding.   AVA-Encoder transforms a video into a knowledge graph (KG) representation and then reconstructs it back into video. Its hierarchy and state nodes store structured text, while a linked asset layer holds generated images, audio, and video. Typed edges preserve the relations between these text descriptions and assets in a form that agents can easily understand, query, and edit. The video reconstruction differences drive a textual-gradient optimization framework, which
    
[^148]: NOLLI：一个难度校准的英韩谜题基准

    NOLLI: A Difficulty-Calibrated English-Korean Puzzle Benchmark

    [https://arxiv.org/abs/2608.04397](https://arxiv.org/abs/2608.04397)

    本文提出了NOLLI——首个程序化生成、难度经过行为校准的英韩双语谜题基准，通过7500个具有唯一解的可再生题目揭示出前沿模型在英韩直接翻译上表现相当，但在涉及韩文字母的文字改编任务上存在高达68.7个百分点的显著差距。

    

    我们提出了NOLLI，一个程序化生成、难度经过校准的英韩双语谜题基准，包含15种谜题类型（25个任务；7500个题目）。每个实例可通过种子再生，具有经验证的唯一解，并以确定性方式评分。我们通过调节生成器参数以在参考模型上达到目标准确率区间，从而实现行为层面的难度校准。NOLLI包含三个任务族：匹配的直接翻译任务、基于韩文字母（jamo，即亚音节字母）的文字系统改编任务，以及仅限韩语的文化或正字法任务。我们评估了15个前沿模型、开源权重模型及韩国开发的模型；差距分析使用了整体准确率高于3%下限的12个模型。在匹配的直接翻译任务上，英语和韩语的准确率在预设的±10个百分点范围内等效。文字改编任务则表现不同：韩语密码任务落后英语最多达68.7个百分点，而在相同字母上的密码算术任务未表现出系统性的……

    arXiv:2608.04397v2 Announce Type: replace  Abstract: We introduce NOLLI, a procedurally generated, difficulty-calibrated English-Korean puzzle benchmark with 15 puzzle types (25 tasks; 7,500 items). Each instance is seed-regenerable, has a verified unique solution, and is scored deterministically. We calibrate difficulty behaviorally by tuning generator parameters to target accuracy bands on a reference model. NOLLI has three task families: matched direct translations, script adaptations over Hangul jamo (sub-syllabic letters), and Korean-only cultural or orthographic tasks. We evaluate 15 frontier, open-weight, and Korean-developed models; gap analyses use the 12 above a 3% overall-accuracy floor. On matched direct translations, English and Korean accuracy is equivalent within a prespecified $\pm 10$ percentage-point margin. Script adaptations behave differently: Korean Cipher trails English by up to 68.7 percentage points, while Cryptarithmetic over the same jamo shows no systematic 
    
[^149]: MedGame：由大语言模型赋能的医学教育叙事游戏化框架

    MedGame: Storytelling Gamification Empowered by Large Language Models for Medical Education

    [https://arxiv.org/abs/2607.21570](https://arxiv.org/abs/2607.21570)

    MedGame提出双引擎框架，将静态临床病例转化为结构化的可执行叙事游戏以支持以决策为中心的医学教育，并构建了包含5000个病例的MedGame Bench基准，实验证明任务特定微调可显著提升开源大语言模型的表现并缩小与商业模型的差距。

    

    大语言模型（LLM）在医学教育中展现出巨大潜力，但现有系统大多局限于局部化的交互方式，如问答或单轮反馈，而非将完整的临床病例组织成以决策为中心的学习轨迹。我们提出了MedGame，这是一个将静态临床病例转化为结构化、可执行叙事游戏的框架。MedGame采用双引擎设计：医疗叙事设计器生成基于病例的临床故事线，包含状态与决策节点；故事导演则将其转化为具有依赖感知的多模态编排计划，并由我们发布的交互平台进行渲染呈现。我们构建了MedGame Bench，这是一个包含5,000个病例的基准数据集及评估协议，用于医疗叙事生成和故事执导任务。实验表明，针对特定任务的微调能显著提升开源大语言模型在MedGame Bench上的表现，并缩小了其与商业模型之间的差距。

    arXiv:2607.21570v2 Announce Type: replace  Abstract: Large Language Models (LLMs) show promise for medical education, but most existing systems focus on localized interactions such as question answering or single-turn feedback, rather than organizing an entire clinical case into a decision-centered learning trajectory. We introduce \textit{MedGame}, a framework that transforms static clinical cases into structured, executable storytelling games. MedGame uses a dual-engine design: a Medical Narrative Designer synthesizes case-grounded clinical storylines with states and decision nodes, while a Story Director converts them into dependency-aware multimodal orchestration plans rendered by our released interactive platform. We construct MedGame Bench, a 5,000-case benchmark and evaluation protocol for Medical Narrative Generation and Story Direction. Experiments show that task-specific fine-tuning substantially improves open-source LLMs on MedGame Bench and narrows the gap with commercial m
    
[^150]: 超越以相关性为中心的检索：面向评分标准的文档集合选择与排序

    Beyond Relevance-Centric Retrieval: Rubric-Oriented Document Set Selection and Ranking

    [https://arxiv.org/abs/2607.19747](https://arxiv.org/abs/2607.19747)

    该论文提出了一套完整的“评估-诊断-优化”框架，并构建了包含约2.8万条评分标准的三级九维文档集合评估基准SetwiseEvalKit，通过对12个重排序方法的系统评估揭示了现有方法在跨文档协调能力上的普遍缺陷，为面向大模型时代的文档集合选择与排序确立了新的评估范式。

    

    随着大语言模型和AI智能体成为搜索结果的主要消费者，文档集合的质量决定了下游生成效果的上限。然而，现有的评估体系仍然局限于对文档进行独立评分并通过nDCG进行聚合，忽略了文档之间的交互（冗余、冲突、互补），无法回答“什么使得一个文档集合优于另一个文档集合”这一根本问题。为解决这些问题，我们提出了一个完整的“评估-诊断-优化”框架。我们设计了SetwiseEvalKit，这是一个三级九维的文档集合评估基准，涵盖短文本和长文本场景，包含约2.8万条高质量评估评分标准。我们系统地评估了12个重排序方法，发现：即使是最好的方法，其覆盖率也不超过45%；跨文档协调维度的表现普遍较弱；且没有任何单一方法能在两种场景设置下都保持最优性能。在此基础上……

    arXiv:2607.19747v3 Announce Type: replace  Abstract: As large language models and AI agents become the primary consumers of search results, document set quality determines the upper bound of downstream generation. Yet existing evaluation systems remain confined to scoring documents independently and aggregating via nDCG, ignoring inter-document interactions (redundancy, conflict, complementarity) and unable to answer what makes one document set better than another. To address these issues, we propose a complete evaluate-diagnose-optimize framework. We design SetwiseEvalKit, a three-level, nine-dimension document set evaluation benchmark covering both short-form and long-form scenarios, comprising approximately 28K high-quality evaluation rubrics. We systematically evaluate 12 rerankers: even the best method achieves no more than 45% coverage, cross-document coordination dimensions are universally weak, and no single method maintains top performance across both settings. Building on thi
    
[^151]: 大语言模型中的信息甄别能力

    Information Discernment in Large Language Models

    [https://arxiv.org/abs/2607.19355](https://arxiv.org/abs/2607.19355)

    该论文提出“信息甄别”概念及基于三条规范性公理的Learn2Discern（L2D）基准框架，并通过13个模型近67万次试验与真实用户研究证实，当前大语言模型在来源甄别和真相甄别两个维度上均持续失败、表现接近随机水平。

    

    大语言模型越来越多地与互联网等外部知识源结合使用。它们能否对信息进行恰当的权衡——即对可靠来源给予更多的信念更新（来源甄别），并在主张使其先验更接近真相时给予更多的更新（真相甄别）？我们将这一问题形式化为“信息甄别”，并提出了Learn2Discern（L2D），这是一个基于三条规范性公理、配备可解释指标的实验框架与基准测试。为确立外部效度，一项预注册、配额匹配的用户研究（n=299）证实，真实的LLM用户认可全部三条公理，并表示违反这些公理会降低他们的信任度和使用意愿。在13个模型、近67万次试验中，研究发现模型在这两个维度上均存在持续性的失败：模型在来源甄别和真相甄别上的表现接近随机水平，其对来源流行度的依赖程度是对来源可靠性的两倍，且无论主张是改善还是恶化其相对真相的立场，模型的更新幅度都大致相同。

    arXiv:2607.19355v2 Announce Type: replace  Abstract: LLMs are increasingly used with external knowledge sources like the internet. Do they weigh information appropriately -- updating more for reliable sources (source discernment) and more when claims bring priors closer to the truth (truth discernment)? We formalize this as information discernment and introduce Learn2Discern (L2D), an experimental framework and benchmark grounded in three normative axioms with interpretable metrics. To establish external validity, a pre-registered, quota-matched user study (n=299) confirms that real LLM users endorse all three axioms and report that violations reduce their trust and usage intent. Across 13 models and nearly 670K trials, we find consistent failures across both dimensions: models perform near chance on source and truth discernment, rely on source popularity twice as much as source reliability, and update roughly equally whether a claim improves or worsens their position relative to the g
    
[^152]: 还剩多少？大语言模型线性编码其剩余输出长度

    How Much is Left? LLMs Linearly Encode Their Remaining Output Length

    [https://arxiv.org/abs/2607.05316](https://arxiv.org/abs/2607.05316)

    本文发现大语言模型在隐藏状态中线性编码了剩余输出长度信息，仅需一个简单的线性探针即可准确预测生成回答还剩多少token，且当出现“Wait”等撤回token时预测长度会显著跳升。

    

    大语言模型逐个生成token，并不会显式地追踪所生成文本的长度。本文提出了这样一个问题：我们能否仅通过对模型隐藏状态使用线性探针，来估计一个生成回答中剩余的token数量？先前关于推理调度的工作已经表明，文本长度在统计上是可预测的，但使用的是非线性方法。本文贡献了一种线性表示方法，使用7-8B指令微调模型的冻结隐藏状态，在七个数据集上进行了实验。我们证明，总的补全长度是可以被估计的，其性能优于统计基线，也优于与提示词最后一个token处预测结果的公平对比。在自然语言数据集上，该探针的误差介于统计基线误差的0.57到0.88之间。有趣的是，当出现撤回token（"Wait"）时，探针预测的长度会向上跳升超过70个token。

    arXiv:2607.05316v2 Announce Type: replace  Abstract: Large Language models generate tokens, without explicitly keeping track of the length of the generated text. This paper asks the question: can we estimate the number of tokens remaining in a generated response, using just a linear probe on the model's hidden states? Previous work on inference scheduling established that length is statistically predictable, using nonlinear methods. This paper contributes a linear representation method, using frozen hidden states of 7-8B instruction-tuned models, across seven datasets. We show that total completion length can be estimated, with performance stronger than statistical baselines, and stronger than the fair comparison to the prediction at the last token of the prompt. On the natural-language datasets, the probe's error lies between 0.57 and 0.88 of the statistical baseline's. Interestingly, when a retraction token occurs ("Wait"), the probe predicted length jumps upward by over 70 tokens, o
    
[^153]: 基于拉格朗日奖励增强的安全推理时对齐

    Safe Inference-Time Alignment via Lagrangian Reward Augmentation

    [https://arxiv.org/abs/2607.02781](https://arxiv.org/abs/2607.02781)

    提出LARA框架，通过拉格朗日对偶将安全约束转化为由小型校准集估计的对偶变量所定义的增强奖励，实现可即插即用于现有推理时对齐方法的无需权重更新的安全对齐。

    

    推理时对齐在解码过程中利用辅助奖励信号来引导冻结的语言模型，从而避免了反复更新权重的高昂代价。然而，现有的推理时对齐方法通常只优化单一的标量分数，因此显式的安全约束要么被忽略，要么只能通过手工调整的惩罚项来编码。我们提出了拉格朗日奖励增强（LARA），一个在安全约束下的通用推理时对齐框架。从包含奖励模型和代价模型的KL正则化约束目标出发，LARA对约束进行对偶化，将优化问题化为关于一个非负对偶变量的一维凸问题。该对偶变量通过一个小型校准集估计得到，并由其定义的增强奖励可作为现有推理时对齐方法中即插即用的评分信号。对于序列级采样方法，例如Best-of-N重排序，该……（原文摘要在此处截断）

    arXiv:2607.02781v3 Announce Type: replace-cross  Abstract: Inference-time alignment steers a frozen language model during decoding using auxiliary reward signals, avoiding the cost of repeated weight updates. However, existing inference-time alignment methods typically optimize a single scalar score, so explicit safety constraints must either be ignored or encoded through manually tuned penalties. We propose Lagrangian Reward Augmentation (LARA), a general inference-time alignment framework under safety constraints. Starting from a KL-regularized constrained objective with a reward model and a cost model, LARA dualizes the constraint and reduces the optimization problem to a one-dimensional convex problem over a nonnegative dual variable. Estimated on a small calibration set, this dual variable defines an augmented reward that can be used as a drop-in scoring signal within existing inference-time alignment methods. For sequence-level sampling methods, such as Best-of-N reranking, the c
    
[^154]: 多中介诅咒：激活修补中的隐藏交互效应

    The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching

    [https://arxiv.org/abs/2606.27510](https://arxiv.org/abs/2606.27510)

    本文从因果中介分析出发重新推导了激活修补的估计量，揭示其中隐藏的交互效应会导致因果重要性依赖于其他组件状态的模型组件被错误地低估或夸大，且这种效应无法通过调整估计器或分析单位来消除。

    

    激活修补是机制可解释性研究中的一种主要工具。它通过估计自然间接效应（NIE）将模型行为的因果责任归因于各个单独组件。通过从因果中介分析重新推导激活修补的估计量，我们发现NIE并不仅仅捕捉通过特定组件的因果效应，它还包含交互效应（INT），用于衡量该组件的因果效应本身在多大程度上依赖于模型中其他组件的状态。一个自然的应对方法可能是尝试通过调整估计器或分析单位来消除INT，但这些潜在补救措施中的每一种都存在可预测的失败模式。我们在GPT-2 IOI电路中演示了这些失败模式：那些因果重要性依赖于其他组件状态的组件要么不可见，要么被人为地夸大。我们证明INT随着距离...（摘要在此处被截断）

    arXiv:2606.27510v2 Announce Type: replace-cross  Abstract: Activation patching is a primary tool in mechanistic interpretability. It attributes causal responsibility for a model's behavior to each of its individual components by estimating its natural indirect effect (NIE). Re-deriving the activation patching estimand from causal mediation analysis, we find that the NIE does not solely capture the causal effect through the specific component. It also contains interaction effects (INT) that measure how much the component's causal effect itself depends on the states of other components in the model. A natural response may be to try to eliminate INT by adjusting the estimator or unit of analysis, but each of these potential remedies has predictable failure modes. We demonstrate these failure modes in the GPT-2 IOI circuit; components whose causal importance is conditional on the state of other components are either invisible or artificially inflated. We prove that INT scales with the dist
    
[^155]: RepSelect：通过表示选择性实现鲁棒的大语言模型遗忘

    RepSelect: Robust LLM Unlearning via Representation Selectivity

    [https://arxiv.org/abs/2606.17168](https://arxiv.org/abs/2606.17168)

    RepSelect通过压缩权重梯度主成分来隔离遗忘集特定表示，实现了既难以被微调恢复又不损害通用能力的鲁棒遗忘。

    

    当大语言模型的权重开放或通过API提供微调时，仅抑制有害知识和倾向是不够的：删除必须足够深入，以至于攻击者无法恢复。现有的遗忘方法在此标准下显得肤浅：微调或少量上下文示例即可恢复行为，且此过程常常损害通用能力。我们识别出一个根本原因：现有方法编辑与保留集共享的表示，且这些表示位于微调攻击者能恢复的子空间中，这使得遗忘既容易被撤销又具有破坏性。基于此，我们提出RepSelect（表示选择性），通过在每个遗忘更新前压缩权重梯度的前主成分，隔离遗忘集特定的表示，从而在保留通用能力的同时限制微调可恢复的内容。我们在涵盖知识和行为五个遗忘数据集上进行了验证。

    arXiv:2606.17168v3 Announce Type: replace  Abstract: When LLM weights are open or fine-tuning is available through an API, suppressing hazardous knowledge and tendencies is not enough: removal has to be deep enough that an adversary cannot restore it. Existing unlearning is shallow by this standard: fine-tuning or a handful of in-context examples brings the behaviour back, and it often degrades general capabilities in the process. We identify a root cause: existing methods edit representations shared with the retain set and lying in the subspace that a fine-tuning attacker recovers, making unlearning simultaneously easy to undo and disruptive. Leveraging this, we propose RepSelect (Representation Selectivity), which isolates forget-set-specific representations by collapsing the top principal components of the weight gradients before each unlearning update, preserving general capabilities while limiting what fine-tuning can recover. Across five unlearning datasets spanning both knowledg
    
[^156]: 通过基于知识图谱的数据生成实现精确的文本到Cypher转换

    Achieving Precise Text-To-Cypher Via Grounded Knowledge Graph Data Generation

    [https://arxiv.org/abs/2606.14325](https://arxiv.org/abs/2606.14325)

    本文提出一种基于知识图谱的自动合成数据生成方法，通过微调小型LLM显著提升Text2Cypher任务性能，使其媲美大型专有模型，从而在本地部署场景下保障数据主权且无需昂贵的标注。

    

    属性图正被迅速采用作为表示异构数据源的数据库框架。为了能够精确访问其中包含的信息，我们需要基于文本到Cypher（Text2Cypher）解析器的对话式接口。本文提出了一种自动合成数据生成方法，可用于微调小型LLM来执行此任务。我们在所有主要的Text-To-Cypher基准测试上进行了实验，结果表明，借助我们的合成数据生成方法，可以显著提升小型LLM的性能，使其能够与规模大得多的专有模型相媲美。这意味着在必须本地部署模型的场景中，我们可以在不牺牲准确性且无需昂贵标注工作的情况下确保数据主权。

    arXiv:2606.14325v2 Announce Type: replace-cross  Abstract: Property Graphs are rapidly being adopted as database frameworks for representing heterogeneous data sources. To enable precise access to the information contained in them we need conversational interfaces based on Text-To-Cypher (Text2Cypher) parsers. This paper presents an automatic synthetic data generation method that can be leveraged to fine-tune small LLMs for this task. We conduct experiments on all the major Text-To-Cypher benchmarks, demonstrating that with our synthetic data generation approach we can significantly increase the performance of small LLMs, allowing them to compete with much larger proprietary models. This means that in settings in which models must be locally deployed we can ensure data-sovereignty without sacrificing accuracy and without costly annotation campaigns.
    
[^157]: 面向计算机使用智能体的历史感知视觉定位评论模型

    A History-Aware Visually Grounded Critic for Computer Use Agents

    [https://arxiv.org/abs/2606.11078](https://arxiv.org/abs/2606.11078)

    提出HiViG框架，通过将历史交互抽象为紧凑记录并提供视觉定位的动作评估，解决了现有评论模型健忘和缺乏视觉定位的缺陷，从而提升计算机使用智能体在GUI环境中的表现。

    

    arXiv:2606.11078v2 公告类型：替换。摘要：为了在复杂的图形用户界面（GUI）环境中通过执行前动作评估来提升性能，研究者们已经开发了多种针对计算机使用智能体（CUA）的测试时干预方法，其中包括评论模型。然而，现有的评论模型存在两个关键局限：（1）主要聚焦于短视的决策循环（例如遗忘先前的动作）；（2）缺乏检测缺陷动作所需的视觉定位能力（例如点击了错误的UI元素）。为了解决这些问题，我们提出了HiViG，一个历史感知且具备视觉定位能力的测试时框架，其核心是一个在真实GUI轨迹上训练的多模态评论模型，该模型能够将过去的交互抽象为紧凑的记录，并以视觉定位的方式评估动作。在测试时，HiViG将评论模型集成到策略决策循环中，提供宏观动作历史（总结策略已完成的成就）以及视觉定位的评论（验证动作依据……）（摘要原文在此处截断）。

    arXiv:2606.11078v2 Announce Type: replace  Abstract: Various test-time interventions for Computer Use Agents (CUAs), including critic models, have been developed to improve performance through pre-execution action evaluation in complex Graphical User Interface (GUI) environments. However, existing critics suffer from two key limitations: they (1) focus primarily on short-sighted decision loops (e.g., forgetting earlier actions) and (2) lack the visual grounding needed to detect flawed actions (e.g., clicking wrong UI elements). To address these, we introduce HiViG, a History-aware Visually Grounded test-time framework, built around a multimodal critic trained on real GUI trajectories to abstract past interactions into a compact record and to evaluate actions with visual grounding. At test time, HiViG integrates the critic into the policy decision loop to provide macro-action history, which summarizes the policy's completed achievements, and visually grounded critique, which verifies ra
    
[^158]: SparDA：面向高效长上下文大语言模型推理的稀疏解耦注意力

    SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference

    [https://arxiv.org/abs/2606.04511](https://arxiv.org/abs/2606.04511)

    SparDA通过在Query、Key、Value之外引入与注意力查询解耦的Forecast投影来预测下一层所需的KV块，实现CPU到GPU预取与当前层计算重叠的前瞻式稀疏注意力选择，从而突破长上下文LLM推理中KV缓存容量与稀疏选择复杂度的双重瓶颈。

    

    稀疏注意力能够降低长上下文大语言模型推理中的计算量与内存带宽需求。然而，仍存在两个关键挑战：（1）KV缓存的容量依然随序列长度增长，而将其卸载到CPU内存会引入PCIe传输瓶颈；（2）预填充阶段的稀疏选择仍保持 O(T²) 的复杂度，在长上下文场景下可能主导注意力计算成本。我们提出了SparDA，一种解耦的稀疏注意力架构，它在Query、Key和Value之外引入了每层的第四个投影——Forecast。Forecast预测下一层所需的KV块，从而实现前瞻性选择，使CPU到GPU的预取与当前层的执行相互重叠。由于Forecast与注意力查询解耦，我们的GQA实现为每个GQA组仅使用一个Forecast头，相比原始的多头选择器降低了选择开销。在所评估的8B模型上，SparDA仅增加不到0.5%的参数，且只训练Forecast投影（摘要在此处截断）。

    arXiv:2606.04511v2 Announce Type: replace  Abstract: Sparse attention reduces compute and memory bandwidth for long-context LLM inference. However, two key challenges remain: (1) KV cache capacity still grows with sequence length, and offloading to CPU memory introduces a PCIe transfer bottleneck; (2) sparse selection during prefill retains $O(T^2)$ complexity and can dominate attention cost at long contexts. We propose SparDA, a decoupled sparse attention architecture that introduces a fourth per-layer projection, the Forecast, alongside Query, Key, and Value. The Forecast predicts the KV blocks needed by the next layer, enabling lookahead selection that overlaps CPU-to-GPU prefetch with current-layer execution. Because Forecast is decoupled from the attention query, our GQA implementation uses one Forecast head per GQA group, reducing selection overhead versus the original multi-head selector. On the evaluated 8B models, SparDA adds <0.5% parameters and trains only the Forecast proje
    
[^159]: 知识保持的大语言模型架构标准化：解析归一化器映射与等尺寸蒸馏

    Knowledge-Preserving LLM Architecture Standardization: Analytic Normalizer Mapping with Equal-Size Distillation

    [https://arxiv.org/abs/2605.25676](https://arxiv.org/abs/2605.25676)

    提出KEPT方法，通过直接参数复制、免训练的LayerNorm到RMSNorm解析初始化（OPM）和等尺寸蒸馏，在无法访问源模型预训练语料的情况下，将预训练LLM低成本地标准化为Llama家族架构，仅需约1.23亿token和4个A100 GPU天即可几乎无损地保留源模型能力。

    

    能力强大的预训练大语言模型常常被锁定在主流工具链已不再支持的架构中，导致其知识被搁置。传统的补救方法“继续训练”需要大量算力，以及覆盖模型全部知识的语料库。我们研究了这类转换中的一个常见类别——架构标准化，即将模型的归一化层和接口转换为 Llama 家族模板，同时保留维度、分词器和注意力布局。我们的 KEPT 方法结合了直接参数复制、一种免训练的 LayerNorm 到 RMSNorm 初始化方法（OPM，在我们于三个模型上验证的激活假设下接近最优），以及来自冻结源模型的标准等尺寸蒸馏，且无需访问源模型的预训练语料库。从 Orion-14B 转换而来，仅使用约 1.23 亿 token、四个 A100 GPU 天的算力，Llamion 在 H6 平均分和 MT-Bench 分数上与源模型的差距不超过一分，在韩语 M……（摘要在此处被截断）

    arXiv:2605.25676v2 Announce Type: replace  Abstract: Competent pretrained LLMs are often locked into architectures that mainstream tooling no longer supports, which strands their knowledge. The conventional remedy, uptraining, needs heavy compute plus a corpus covering everything the model knows. We study a common class of such conversions, architecture standardization, converting a model's normalization layers and interface to the Llama-family template while preserving dimensions, tokenizer, and attention layout. Our KEPT combines direct parameter copying, a training-free LayerNorm-to-RMSNorm initialization (OPM) that is near-optimal under activation assumptions we verify on three models, and standard equal-size distillation from the frozen source, without access to the source model's pretraining corpus. Converted from Orion-14B with ~123M tokens in four A100-days, Llamion stays within one point of the source's H6 average and MT-Bench score, shows no significant difference on Korean M
    
[^160]: Search-E1：自蒸馏驱动搜索增强推理的自我进化

    Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning

    [https://arxiv.org/abs/2605.22511](https://arxiv.org/abs/2605.22511)

    Search-E1提出一个简单的自我进化框架，通过交替使用GRPO与在策略自蒸馏，将策略自身成功轨迹的搜索骨架作为特权上下文进行对齐，使搜索增强推理智能体无需外部教师即可从自身成功经验中持续自我改进。

    

    基于结果的强化学习为训练搜索增强推理智能体提供了一种简单而有效的方案。近期研究通过更强的外部教师模型、辅助的过程监督、专门的奖励塑形以及日益复杂的训练流程来寻求进一步改进。这引出了一个自然的问题：智能体能否通过从自身的成功经验中学习而持续改进？我们提出了Search-E1，一个简单的自我进化框架，它在原始GRPO与在策略自蒸馏（OPSD）之间交替进行。在每轮GRPO之后，Search-E1从策略自身生成的同类轨迹中进行挖掘，并将一条高效成功轨迹的搜索骨架——即它所发出的查询序列——作为特权上下文呈现出来。随后，通过token级别的前向KL散度目标，将标准推理上下文下的策略与该特权上下文下策略自身的分布进行对齐。通过这种方式，策略……（原文摘要在此处截断）

    arXiv:2605.22511v3 Announce Type: replace  Abstract: Outcome based reinforcement learning provides a simple and effective recipe for training search augmented reasoning agents. Recent work seeks further improvements through stronger external teachers, auxiliary process supervision, specialized reward shaping, and increasingly elaborate training pipelines. This raises a natural question: can agents continue improving by learning from their own successful experience? We present Search-E1, a simple self evolution framework that alternates vanilla GRPO with on policy self distillation (OPSD). After each GRPO round, Search-E1 mines sibling trajectories generated by the policy itself and exposes the search skeleton of an efficient successful trajectory, that is, the sequence of queries it issued, as privileged context. A token level forward KL objective then aligns the policy under its standard inference context with its own distribution under this privileged context. In this way, the policy
    
[^161]: NSFL：一种面向神经嵌入中布尔运算符的训练后神经符号模糊逻辑框架

    NSFL: A Post-Training Neuro-Symbolic Fuzzy Logic Framework for Boolean Operators in Neural Embeddings

    [https://arxiv.org/abs/2604.10604](https://arxiv.org/abs/2604.10604)

    提出免训练的神经符号模糊逻辑框架 NSFL，借助 NS-Delta 边际语义贡献与黎曼优化查询投影，使稠密检索器无需训练即可原生处理多原子布尔逻辑约束，检索性能最高提升 81%。

    

    标准稠密检索器缺乏处理多原子逻辑约束的原生机制。我们提出了神经符号模糊逻辑框架，这是一种免训练框架，通过对相似度进行一阶有限差分处理，将模糊逻辑结构引入神经嵌入空间。NSFL 将逻辑运算锚定在零阶原子分数上，同时通过神经符号增量进行引导——即由上下文融合所诱导的边际语义贡献——在捕获上下文依赖性的同时保持原子语义，并避免了几何基线方法中普遍存在的表示坍缩和流形逃逸问题。针对延迟受限的部署场景，球形查询优化（SQO）利用黎曼优化将逻辑公式投影为流形稳定的查询向量。在六种编码器配置和两种模态上，NSFL 在 QUEST 数据集上实现了高达 +81% 的 mAP@100 提升，在 COCO 衍生的文本到图像检索任务上实现了 +70% 的提升。

    arXiv:2604.10604v2 Announce Type: replace-cross  Abstract: Standard dense retrievers have no native mechanism for multi-atom logical constraints. We introduce Neuro-Symbolic Fuzzy Logic (NSFL), a training-free framework that brings fuzzy-logical structure to neural embedding spaces through a first-order finite-difference treatment of similarity. NSFL anchors logical operations on zero-order atomic scores while steering with Neuro-Symbolic Deltas (NS-Delta) -- the marginal semantic contributions induced by contextual fusion -- preserving atomic meaning while capturing contextual dependence, and avoiding the representation collapse and manifold escape endemic to geometric baselines. For latency-constrained deployment, Spherical Query Optimization (SQO) uses Riemannian optimization to project logical formulas into manifold-stable query vectors. Across six encoder configurations and two modalities, NSFL yields mAP@100 improvements up to +81% on QUEST and +70% on a COCO-derived text-to-imag
    
[^162]: BLADE：通过对话与解释获得更好的语言答案

    BLADE: Better Language Answers through Dialogue and Explanations

    [https://arxiv.org/abs/2604.03236](https://arxiv.org/abs/2604.03236)

    BLADE通过基于引用依据的对话引导而非直接答案，显著提升了学生在高级NLP课程中的测验表现，且学生更倾向于使用它。

    

    基于大型语言模型（LLM）的教育助手通常提供直接答案，这几乎不激励学生探索或参与课程材料。我们提出了BLADE（通过对话与解释获得更好的语言答案），这是一种基于检索增强生成（RAG）的对话助手，以课程特定内容为基础，通过引用依据的对话引导学生获取相关材料，而不是提供无来源的解决方案。我们在一个拥有大量教学资源的高级本科NLP课程中评估了BLADE，在该课程中，定位和综合相关材料是一个核心挑战。在测验期间，学生被分配到三种条件之一：仅使用BLADE、仅使用直接课程材料，或两者兼用；我们测量了表现和资源使用情况。结果显示，当两种资源都可用时，学生始终选择BLADE而非直接材料，并且测验表现更高。

    arXiv:2604.03236v2 Announce Type: replace-cross  Abstract: Large language model (LLM)-based educational assistants often provide direct answers offering little incentive for students to explore or engage with course materials. We present BLADE (Better Language Answers through Dialogue and Explanations), a retrieval-augmented generation (RAG) based conversational assistant grounded in course-specific content that guides students toward relevant materials through citation-grounded dialogue rather than delivering unsourced solutions. We evaluate BLADE in an advanced undergraduate NLP course with extensive instructional resources, where locating and synthesizing relevant material is a central challenge. During quizzes, students are assigned to one of three conditions: BLADE only, direct course materials only, or both; we measure performance and resource usage. Results show that students consistently select BLADE over direct materials when both are available, and that quiz performance is hi
    
[^163]: 视觉STEM推理中的跨语言差距：跨越六种印度语言审计视觉语言模型

    Cross-Lingual Gaps in Visual STEM Reasoning: Auditing VLMs Across Six Indian Languages

    [https://arxiv.org/abs/2603.26742](https://arxiv.org/abs/2603.26742)

    该论文构建了六种印度语言的视觉STEM推理审计基准，发现视觉语言模型从英语切换到印度语言时准确率下降9.8至24.9个百分点，且达罗毗荼语系语言的性能下降比印度-雅利安语系更严重。

    

    视觉语言模型（VLM）在基于图表的科学技术工程数学（STEM）推理方面的基准测试大多以英语进行，因此当同一问题以另一种语言提出时，其准确率会如何变化尚不清楚。我们对六种印度语言进行了审计：将来自MathVista、ScienceQA和MMMU的980道题目通过IndicTrans2机器翻译成印地语、孟加拉语、马拉地语、泰米尔语、泰卢固语和卡纳达语，同时保持图像不变。我们评估了七个开源权重的VLM（Qwen2.5-VL-7B/32B、Qwen3-VL-30B、InternVL2.5-8B、Aya-Vision-8B、Gemma 3-27B、Llama-4-Maverick）以及GPT-4o，并在Qwen2.5-VL-7B上进行了无图像和思维链（CoT）消融实验（共计68,600条推理记录）。结果显示，从英语到印度语的平均准确率下降了9.8至24.9个百分点，且每个模型在达罗毗荼语系语言上的损失都比印度-雅利安语系语言更大（差距为0.1至13.2个百分点）。对于使用本文提示词的Qwen2.5-VL-7B，思维链降低了孟加拉语……

    arXiv:2603.26742v2 Announce Type: replace  Abstract: Vision-language models (VLMs) are benchmarked on diagram-grounded science, technology, engineering, and mathematics (STEM) reasoning mostly in English, so it is unclear how much their accuracy changes when the same problem is posed in another language. We audit this for six Indian languages: 980 questions from MathVista, ScienceQA, and MMMU are machine-translated into Hindi, Bengali, Marathi, Tamil, Telugu, and Kannada with IndicTrans2 while the images stay unchanged. We evaluate seven open-weight VLMs (Qwen2.5-VL-7B/32B, Qwen3-VL-30B, InternVL2.5-8B, Aya-Vision-8B, Gemma 3-27B, Llama-4-Maverick) and GPT-4o, plus no-image and chain-of-thought (CoT) ablations on Qwen2.5-VL-7B (68,600 inference records). Accuracy falls 9.8-24.9 percentage points (pp) from English to the Indian-language mean, and every model loses more on the Dravidian than on the Indo-Aryan languages (by 0.1-13.2 pp). For Qwen2.5-VL-7B with our prompt, CoT lowers Benga
    
[^164]: 更深入而非更长：使用深度循环Transformer实现内存高效的测试时推理以获得组合泛化能力

    Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with Depth-Recurrent Transformers for Compositional Generalization

    [https://arxiv.org/abs/2603.21676](https://arxiv.org/abs/2603.21676)

    本文提出一种深度循环Transformer，通过迭代共享权重模块在不生成思维链token的情况下实现更深的测试时推理，以恒定内存和线性延迟达成内存高效推理，并在组合泛化任务上验证了其有效性。

    

    标准Transformer具有固定的计算深度，限制了其泛化到需要可变深度推理任务的能力。通常的补救方法——思维链，通过耗费token进行推理，导致键值缓存膨胀，并使延迟随步数增长，因此在服务大批量查询时，内存成为限制性成本。我们研究了一种深度循环Transformer，它通过迭代一个共享权重的模块将计算深度与参数量解耦，使得每增加一个推理步骤只需恒定的内存开销和线性增长的延迟，且无需生成token。三个关键要素保证了循环在超过20个思考步骤中保持稳定：一个仅监督最终输出的“静默思考”目标、LayerScale初始化，以及一个带恒等偏置的门控机制，该门控在各步骤之间开辟了一条梯度高速通道。我们在三个结构偏置递减的组合域上对其进行了评估：图可达性（邻……

    arXiv:2603.21676v2 Announce Type: replace-cross  Abstract: Standard Transformers have a fixed computational depth, limiting their ability to generalize to tasks that require variable-depth reasoning. The usual remedy, Chain-of-Thought (CoT), spends tokens to reason, inflating the key--value cache and making latency grow with the step count, so memory becomes the limiting cost when reasoning is served over large query batches. We study a depth-recurrent Transformer that decouples computational depth from parameter count by iterating a shared-weight block, so that each added reasoning step costs flat memory and linear latency, with no token generation. Three ingredients keep the recurrence stable for 20+ thinking steps: a silent thinking objective that supervises only the final output, LayerScale initialization, and an identity-biased gate that opens a gradient highway across steps. We characterize it on three compositional domains with decreasing structural bias: graph reachability (adj
    
[^165]: Fusian：面向大语言模型中MBTI人格细粒度连续控制的多LoRA融合方法

    Fusian: Multi-LoRA Fusion for Fine-Grained Continuous MBTI Personality Control in Large Language Models

    [https://arxiv.org/abs/2603.15405](https://arxiv.org/abs/2603.15405)

    Fusian框架通过在SFT过程中收集LoRA适配器轨迹序列，并利用强化学习训练策略网络动态融合这些冻结适配器，实现了对大语言模型MBTI人格特质强度的细粒度连续控制。

    

    大语言模型（LLMs）在模拟多样化的人类行为和人格方面已展现出令人印象深刻的能力。然而，现有人格控制方法，包括提示工程和标准的监督微调（SFT），通常将人格特质视为离散类别（例如“外向”与“内向”），缺乏在连续谱上精确控制特质强度的能力。本文提出了Fusian，一个用于LLM细粒度连续人格控制的新型框架。Fusian分两个阶段运行：（1）轨迹收集，通过保存一系列LoRA适配器来捕捉SFT过程中人格形成的动态演变，从而有效映射人格特质的连续流形；（2）基于强化学习的动态融合，使用强化学习训练一个策略网络，动态计算这些冻结适配器的混合权重。

    arXiv:2603.15405v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have demonstrated impressive capabilities in simulating diverse human behaviors and personalities. However, existing methods for personality control, which include prompt engineering and standard Supervised Fine-Tuning (SFT), typically treat personality traits as discrete categories (e.g., "Extroverted" vs. "Introverted"), lacking the ability to precisely control the intensity of a trait on a continuous spectrum. In this paper, we introduce Fusian, a novel framework for fine-grained, continuous personality control in LLMs. Fusian operates in two stages: (1) Trajectory Collection, where we capture the dynamic evolution of personality adoption during SFT by saving a sequence of LoRA adapters, effectively mapping the continuous manifold of a trait; and (2) RL-based Dynamic Fusion, where we train a policy network using Reinforcement Learning to dynamically compute mixing weights for these frozen adapters. By 
    
[^166]: 基于布鲁姆分类学线性探测的大语言模型认知复杂度的机制可解释性研究

    Mechanistic Interpretability of Cognitive Complexity in LLMs via Linear Probing using Bloom's Taxonomy

    [https://arxiv.org/abs/2602.17229](https://arxiv.org/abs/2602.17229)

    本研究利用布鲁姆分类学与线性探测技术揭示，大语言模型在内部表征中以线性可分的方式编码认知复杂度层级，且这种结构在前向传播早期即已形成并随层数加深愈发清晰。

    

    大语言模型的黑箱特性要求我们采用超越表面性能指标的新型评估框架。本研究以布鲁姆分类学作为分层视角，考察复杂度层级的内部神经表征。通过分析不同大语言模型的高维激活向量，我们探测从基础回忆到抽象综合等不同的布鲁姆层级是否在模型的残差流中是线性可分的。我们的结果表明，线性分类器在所有布鲁姆层级上都能达到较高的准确率，证明层级信息被编码在模型表征的一个线性可访问子空间中。这些发现表明，模型在前向传播的早期阶段就按照布鲁姆类别组织表征，且各层之间的表征可分性逐渐增强。

    arXiv:2602.17229v3 Announce Type: replace  Abstract: The black-box nature of Large Language Models necessitates novel evaluation frameworks that transcend surface-level performance metrics. This study investigates the internal neural representations of the levels of complexity using Bloom's Taxonomy as a hierarchical lens. By analyzing high-dimensional activation vectors from different LLMs, we probe whether different Bloom levels, ranging from basic recall (Remember) to abstract synthesis (Create), are linearly separable within the model's residual streams. Our results demonstrate that linear classifiers achieve high accuracy across all Bloom levels, providing evidence that level information is encoded in a linearly accessible subspace of the model's representations. These findings provide evidence that the model organizes representations according to Bloom categories early in the forward pass, with representations becoming increasingly separable across layers.
    
[^167]: 神经同构域：一种基于Transformer的代数数值嵌入方法

    Neural Isomorphic Fields: A Transformer-based Algebraic Numerical Embedding

    [https://arxiv.org/abs/2601.12095](https://arxiv.org/abs/2601.12095)

    该论文提出神经同构域（NIF），一种基于Transformer的数字嵌入方法，可证明地保留十进制分数的代数结构（与有理数域同构），从而解决神经网络处理极端数值时的溢出与不稳定问题。

    

    神经网络模型在处理非常大或非常小的数字时常常面临挑战，原因在于溢出、下溢以及输出变化不稳定等问题。为了缓解这些问题，我们提出使用嵌入向量来表示数字，而不是直接使用其原始数值。这些嵌入旨在保留关键的代数性质，同时避免数值不稳定性。我们提出了神经同构域，这是一种数字序列Transformer自编码器，通过两个学习得到的交换算子和一个反对称神经比较器进行训练。在零损失条件下，这些嵌入可被证明构成一个与十进制分数同构的有序整环，其分式域与有理数同构。对于训练后的模型，我们通过解码边际、缺陷和Lipschitz常数来界定运算、链式运算和代数法则的误差。在最多25位数字上训练后，NIF在封闭性上达到了至少99.9%的精确匹配率。

    arXiv:2601.12095v2 Announce Type: replace-cross  Abstract: Neural network models often face challenges when processing very small or very large numbers due to issues such as overflow, underflow, and unstable output variations. To mitigate these problems, we propose using embedding vectors for numbers instead of directly using their raw values. These embeddings aim to retain essential algebraic properties while preventing numerical instabilities. We introduce Neural Isomorphic Fields (NIF), a digit-sequence Transformer autoencoder trained with two learned commutative operators and an antisymmetric neural comparator. At zero loss, the embeddings provably form an ordered integral domain isomorphic to the decimal fractions, whose field of fractions is isomorphic to the rationals. For the trained model, we bound errors of operations, chains and algebraic laws by decoding margins, defects and Lipschitz constants. Trained on up to 25 digits, NIF reaches at least 99.9% exact match on closure, 
    
[^168]: 语义引力井：为什么负面约束会适得其反

    Semantic Gravity Wells: Why Negative Constraints Backfire

    [https://arxiv.org/abs/2601.08070](https://arxiv.org/abs/2601.08070)

    本文提出“语义压力”这一量化指标，通过机制分析揭示大语言模型违背负面指令的概率随其生成被禁词的内在倾向呈Logistic关系上升（从5%升至44%），失败的原因在于指令诱导的抑制被模型内在生成倾向所压倒。

    

    负面约束（即“不要使用词X”形式的指令）是对大语言模型指令遵循能力的一项基础性考验。尽管这类约束看似简单，却以惊人的规律性失败，而其失败条件至今仍缺乏充分理解。本文对Qwen2.5-7B-Instruct模型中负面指令失效问题进行了机制层面的研究。我们提出了“语义压力”这一量化指标，用以衡量模型生成被禁用词的内在概率，并证明违规概率随压力上升（p = σ(-2.95 + 2.76 P₀)；基于2,500条提示词的40,000个样本；斜率的提示词自助法95%置信区间为[2.53, 3.01]），违规率从低压力时的5%上升到高压力时的44%。在对比相同压力水平的提示词时，我们发现失败案例中负面指令所诱导的抑制效果要弱得多：在相同压力区间内，该指令……（原文摘要在此处截断）

    arXiv:2601.08070v2 Announce Type: replace  Abstract: Negative constraints (instructions of the form "do not use word X") represent a fundamental test of instruction-following capability in large language models. Despite their apparent simplicity, these constraints fail with striking regularity, and the conditions governing failure have remained poorly understood. This paper presents a mechanistic investigation of negative instruction failure in Qwen2.5-7B-Instruct. We introduce semantic pressure, a quantitative measure of the model's intrinsic probability of generating the forbidden word, and show that violation probability rises with pressure ($p = \sigma(-2.95 + 2.76 P_0)$; $n=40{,}000$ samples from 2,500 prompts; prompt-bootstrap 95% CI for slope: [2.53, 3.01]), from 5% of samples at low pressure to 44% at high pressure. Comparing prompts of equal pressure, we find that the suppression induced by negative instructions is far weaker in failures: within pressure bins, the instruction 
    
[^169]: Luxical：高速词汇密集型文本嵌入

    Luxical: High-Speed Lexical-Dense Text Embeddings

    [https://arxiv.org/abs/2512.09015](https://arxiv.org/abs/2512.09015)

    Luxical通过结合稀疏TF-IDF特征、小型ReLU网络和知识蒸馏训练，以极低的计算成本近似大型Transformer嵌入模型，实现了面向网络规模文本组织的高速“词汇密集”文本嵌入。

    

    arXiv:2512.09015v3 公告类型：替换 摘要：前沿语言模型的质量在很大程度上取决于我们组织网络规模文本语料库以用于训练的能力。当今的主流工具在速度与灵活性之间存在权衡：词汇分类器（如FastText）速度快但仅限于生成分类输出分数，而Transformer文本嵌入模型的向量输出能够灵活支持多种工作流程（如聚类、分类和检索），但其计算成本高昂。我们推出了Luxical，一个用于高速“词汇密集”文本嵌入的库，旨在为网络规模的文本组织任务同时兼顾两种方法的最佳特性。Luxical结合了稀疏TF-IDF特征、一个小型ReLU网络以及知识蒸馏训练方案，以仅为大模型一小部分的运行成本来近似大型Transformer嵌入模型。在本技术报告中，我们描述了Luxical的架构和训练目标，并评估了……

    arXiv:2512.09015v3 Announce Type: replace  Abstract: Frontier language model quality increasingly hinges on our ability to organize web-scale text corpora for training. Today's dominant tools trade off speed and flexibility: lexical classifiers (e.g., FastText) are fast but limited to producing classification output scores, while the vector-valued outputs of transformer text embedding models flexibly support numerous workflows (e.g., clustering, classification, and retrieval) but are computationally expensive to produce. We introduce Luxical, a library for high-speed "lexical-dense" text embeddings that aims to recover the best properties of both approaches for web-scale text organization. Luxical combines sparse TF--IDF features, a small ReLU network, and a knowledge distillation training regimen to approximate large transformer embedding models at a fraction of their operational cost. In this technical report, we describe the Luxical architecture and training objective and evaluate a
    
[^170]: 忒修斯之网络（如同那艘船）

    Network of Theseus (like the ship)

    [https://arxiv.org/abs/2512.04198](https://arxiv.org/abs/2512.04198)

    提出忒修斯之网络方法，通过逐步替换组件并用表征相似性度量对齐，在保持性能的前提下将训练好的网络架构渐进转换为完全不同的目标架构，打破了“训练架构即部署架构”的传统假设。

    

    深度学习中的一个标准假设是：神经网络架构所引入的归纳偏置必须从训练阶段一直延续到推理阶段，即你用什么架构训练，就用什么架构部署。这一假设使得研究者无法选择那些可能具有理想效率或设计特性、但存在优化困难的架构。我们提出“忒修斯之网络”来挑战这一假设，该方法能够将一个已训练（甚至未训练）的引导网络架构逐部分地渐进转换为完全不同的目标网络架构，同时保持引导网络的性能。在每个阶段，引导网络架构中的组件被逐步替换为目标架构的模块，并通过表征相似性度量进行对齐。即使在发生重大架构变化的情况下，该过程仍能保留引导网络的大部分功能……

    arXiv:2512.04198v2 Announce Type: replace-cross  Abstract: A standard assumption in deep learning is that the inductive bias introduced by a neural network architecture must persist from training through inference. The architecture you train with is the architecture you deploy. This assumption constrains the community from selecting architectures that may have desirable efficiency or design properties due to difficulties with optimization. We challenge this assumption with the Network of Theseus (NoT), a method for progressively converting a trained, or even untrained, guide network architecture part-by-part into an entirely different target network architecture while preserving the performance of the guide network. At each stage, components in the guide network architecture are incrementally replaced with the target architecture modules and aligned via representational similarity metrics. This procedure preserves most of the functionality of the guide network even under substantial ar
    
[^171]: 代码中的阴影：探索基于大语言模型的多智能体软件开发系统的风险与防御

    Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems

    [https://arxiv.org/abs/2511.18467](https://arxiv.org/abs/2511.18467)

    该论文首次系统揭示了LLM多智能体软件开发系统在恶意用户与恶意智能体两种场景下的安全风险，提出隐式恶意行为注入攻击IMBIA（攻击成功率最高达93%）及其防御机制Adv-IMBIA。

    

    基于大语言模型（LLM）驱动的多智能体系统的快速发展显著简化了软件开发任务，使缺乏技术专业知识的用户也能够开发可执行的应用程序。虽然这些系统通过自然语言需求实现了软件创建的民主化，但它们也引入了在很大程度上尚未被探索的重大安全风险。我们识别出两种风险场景：恶意用户与良性智能体（MU-BA）以及良性用户与恶意智能体（BU-MA）。我们提出了隐式恶意行为注入攻击（IMBIA），展示了如何操纵多智能体系统生成在看似良性的应用程序之下隐藏恶意功能的软件，并提出 Adv-IMBIA 作为一种防御机制。在 ChatDev、MetaGPT 和 AgentVerse 框架上的评估揭示了不同的漏洞模式，IMBIA 分别实现了 93%、45% 和 71% 的攻击成功率。

    arXiv:2511.18467v2 Announce Type: replace-cross  Abstract: The rapid advancement of Large Language Model (LLM)-driven multi-agent systems has significantly streamlined software developing tasks, enabling users with little technical expertise to develop executable applications. While these systems democratize software creation through natural language requirements, they introduce significant security risks that remain largely unexplored. We identify two risky scenarios: Malicious User with Benign Agents (MU-BA) and Benign User with Malicious Agents (BU-MA). We introduce the Implicit Malicious Behavior Injection Attack (IMBIA), demonstrating how multi-agent systems can be manipulated to generate software with concealed malicious capabilities beneath seemingly benign applications, and propose Adv-IMBIA as a defense mechanism. Evaluations across ChatDev, MetaGPT, and AgentVerse frameworks reveal varying vulnerability patterns, with IMBIA achieving attack success rates of 93%, 45%, and 71% 
    
[^172]: AI中介交流可以减少基于观点的网络隔离

    AI-Mediated Communication Can Reduce Opinion-Based Network Segregation

    [https://arxiv.org/abs/2510.21984](https://arxiv.org/abs/2510.21984)

    本研究通过一项大规模预注册在线实验发现，AI中介交流能够影响人们在观点讨论中的社交重组模式，从而减少基于观点差异的网络隔离。

    

    日常交流日益由AI中介，大语言模型（LLM）会建议措辞、语气和表述框架，而由人类决定是否采纳。由于交流能够塑造人们选择与谁互动，AI中介的影响可能不仅限于个体的态度和感知，还可能扩展到更广泛的、跨越观点差异的网络隔离模式。在此，我们通过一项预注册的在线实验（80个场次共711名参与者）检验了这一可能性，实验中参与者围绕有争议的政治话题进行多轮讨论，并可自由重新组合分组。在讨论过程中，参与者在两种条件下接收LLM生成的消息建议：一种是以参与者自身立场为中心的个体型AI辅助，另一种是纳入小组成员观点的关系型AI辅助。我们发现，个体型AI辅助与关系型AI辅助产生了不同的基于观点（隔离的轨迹）……（摘要原文在此处截断）

    arXiv:2510.21984v3 Announce Type: replace-cross  Abstract: Everyday communication is increasingly AI-mediated, with large language models (LLMs) suggesting wording, tone, and framing while humans decide whether to adopt them. Because communication can shape who people choose to interact with, the effects of AI mediation may extend beyond individual attitudes and perceptions to broader patterns of network segregation across opinion differences. Here, we examine this possibility through a preregistered online experiment (N=711 across 80 sessions) in which participants discussed controversial political topics over multiple rounds and could freely reorganize into groups. During the discussions, participants received LLM-generated message suggestions under either individual AI assistance centered on participants' own stances or relational AI assistance incorporating group members' perspectives. We find that individual and relational AI assistance produced distinct trajectories of opinion-ba
    
[^173]: 面向图算法推理的 GDS 智能体

    GDS Agent for Graph Algorithmic Reasoning

    [https://arxiv.org/abs/2508.20637](https://arxiv.org/abs/2508.20637)

    GDS 智能体通过 MCP 服务器将一整套图算法封装为工具并提供预处理与后处理流程，使任何现代 LLM 都能开箱即用地对大规模图结构数据进行算法推理。

    

    大型语言模型（LLM）已经展现出卓越的多模态信息处理与推理能力。当通过函数调用为其配备工具，并结合检索增强技术后，基于LLM的复合系统可以访问封闭的数据源并回答与之相关的问题。然而，它们在处理和推理大规模图结构数据方面仍然存在困难。在本技术报告中，我们介绍了 GDS（图数据科学）智能体。GDS 智能体在模型上下文协议（MCP）服务器中引入了一套全面的图算法作为工具，并配套了算法结果的预处理（检索）与后处理功能。同时，它还附带一项技能（skill），用于引导 LLM 完成图分析任务。该智能体可与任何现代 LLM 开箱即用。GDS 智能体允许用户提出任何隐含或本质上需要对数据进行图算法推理的问题，并快速获得答案。

    arXiv:2508.20637v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have shown remarkable multimodal information processing and reasoning ability. When equipped with tools through function calling and enhanced with retrieval-augmented techniques, compound LLM-based systems can access closed data sources and answer questions about them. However, they still struggle to process and reason over large-scale graph-structure data. In this technical report we introduce the GDS (Graph Data Science) agent. The GDS agent introduces a comprehensive set of graph algorithms as tools, together with preprocessing (retrieval) and postprocessing of algorithm results, in a model context protocol (MCP) server. It is accompanied by a skill that guides the LLM through graph analysis. The agent can be used with any modern LLM out-of-the-box. GDS agent allows users to ask any question that implicitly and intrinsically requires graph algorithmic reasoning about their data, and quickly obtai
    
[^174]: 开放证明语料库：对LLM生成数学证明的大规模研究

    The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs

    [https://arxiv.org/abs/2506.21621](https://arxiv.org/abs/2506.21621)

    该论文提出了包含超过5,000个经人工评估的LLM生成数学证明的开放证明语料库（OPC），这是首个包含大量针对USAMO和IMO等顶级数学竞赛问题的正确LLM证明解决方案的大规模数据集，为自动证明生成研究提供了重要基础。

    

    最近几个月，大型语言模型（LLM）在数学证明生成方面取得了显著进展，但由于缺乏大规模、高质量的经人工评估的证明数据集，进一步的进展受到阻碍。虽然创建这样的数据集成本高昂，但它对于推动训练改进以及实现对证明生成能力的严格分析至关重要。在这项工作中，我们提出了开放证明语料库，这是一个包含超过5,000个由最先进LLM生成并经人工评估的证明的数据集。OPC专门为证明生成研究中的广泛适用性和下游使用而设计，并且是首个包含大量针对USAMO和IMO等著名数学竞赛问题的正确LLM生成解决方案的数据集。利用OPC，我们探索了自动证明生成中的关键问题：（1）自然语言证明与形式化证明之间的性能差距……

    arXiv:2506.21621v3 Announce Type: replace-cross  Abstract: In recent months, large language models (LLMs) have made significant progress in mathematical proof generation, but further advancement is hindered by the lack of a large-scale, high-quality dataset of human-evaluated proofs. While expensive to create, such a dataset is essential for driving improvements in training and enabling a rigorous analysis of proof generation capabilities. In this work, we present the Open Proof Corpus (OPC), a dataset comprising over 5,000 human-evaluated proofs produced by state-of-the-art LLMs. The OPC was specifically designed for broad applicability and downstream usage in proof generation research and is the first to include a substantial number of correct, LLM-generated solutions to problems from prestigious mathematics competitions such as the USAMO and IMO. Using the OPC, we explore critical questions in automated proof generation: (1) the performance gap between natural language and formal pr
    
[^175]: MathArena：在无污染数学竞赛上评估大语言模型

    MathArena: Evaluating LLMs on Uncontaminated Math Competitions

    [https://arxiv.org/abs/2505.23281](https://arxiv.org/abs/2505.23281)

    提出 MathArena 基准，利用定期举办的数学竞赛在新题发布后立即评估大语言模型，有效消除数据污染风险，并发现 AIME 2024 存在明显的污染迹象。

    

    大语言模型（LLM）推理能力的快速发展使其在数学基准测试上取得了显著进步。然而，许多最常用的评估数据集（例如 AIME 2024）在网上广泛可得，这使得很难将真正的推理能力与潜在的“记忆”区分开来。此外，这些基准测试无法评估证明写作能力，而证明写作对许多数学任务至关重要。为了解决这一问题，我们提出了 MathArena，一个基于以下关键洞察的新型基准：定期举办的数学竞赛提供了源源不断的高质量、高难度题目，可用于对 LLM 进行实时评估。通过在新题目发布后立即评估模型，我们有效地消除了数据污染的风险。使用该框架，我们发现了 AIME 2024 存在数据污染的强烈迹象。尽管如此，在更难的竞赛（如 CMIMC 2025）上的评估表明……

    arXiv:2505.23281v4 Announce Type: replace  Abstract: The rapid advancement of reasoning capabilities in large language models (LLMs) has led to notable improvements on mathematical benchmarks. However, many of the most commonly used evaluation datasets (e.g., AIME 2024) are widely available online, making it difficult to disentangle genuine reasoning from potential memorization. Furthermore, these benchmarks do not evaluate proof-writing capabilities, which are crucial for many mathematical tasks. To address this, we introduce MathArena, a new benchmark based on the following key insight: recurring math competitions provide a stream of high-quality, challenging problems that can be used for real-time evaluation of LLMs. By evaluating models as soon as new problems are released, we effectively eliminate the risk of contamination. Using this framework, we find strong signs of contamination in AIME 2024. Nonetheless, evaluations on harder competitions, such as CMIMC 2025, demonstrate impr
    
[^176]: 团结于多样性之中？基于大语言模型的2024年欧洲议会选举预测中的情境偏差

    United in Diversity? Contextual Biases in LLM-Based Predictions of the 2024 European Parliament Elections

    [https://arxiv.org/abs/2409.09045](https://arxiv.org/abs/2409.09045)

    本研究在2024年欧洲议会选举前利用三个大语言模型基于26,000名欧洲选民的个体背景信息预测其投票行为，发现LLM预测与实际结果相比大多失败且准确性在不同情境间分布不均，揭示了LLM舆论预测中存在的情境依赖性偏差。

    

    基于大语言模型（LLM）的“硅基样本”被认为是人类调查的高效替代方案，其前提是模型的训练数据中包含有关人类态度和行为的信息。然而，LLM生成的调查数据可能存在偏差，例如由于训练和对齐过程中缺乏多样性，这有加剧研究与政策制定中既有偏差的风险。在本研究中，我们通过预测2024年欧盟选举的结果，考察基于LLM的个人公众舆论预测在多大程度上表现出情境依赖性偏差。在选举举行之前，我们向三个大语言模型提供26,000名具备投票资格的欧洲选民的个体层面背景信息，以预测每个人的投票行为。我们研究表明，与实际选举结果相比，基于LLM的未来投票行为预测在很大程度上是失败的，且其准确性在不同情境之间分布不均。

    arXiv:2409.09045v3 Announce Type: replace-cross  Abstract: "Silicon samples" based on large language models (LLMs) have been argued to serve as efficient alternatives to surveys of humans, assuming that their training data includes information on human attitudes and behavior. However, LLM-generated survey data might exhibit biases, for example due to a lack of diversity in training and alignment processes, which risk reinforcing existing biases in research and policymaking. In this study, we examine to what extent LLM-based predictions of individual public opinion exhibit context-dependent biases by predicting the results of the 2024 EU elections. Before the elections have taken place, we prompt three LLMs with individual-level background information of 26,000 eligible European voters to predict each person's voting behavior. We show that when compared to the actual results, LLM-based predictions of future voting behavior largely fail, their accuracy is unequally distributed across nat
    
[^177]: Polyrating：一种低成本且能感知偏差的大语言模型评估评分系统

    Polyrating: A Cost-Effective and Bias-Aware Rating System for LLM Evaluation

    [https://arxiv.org/abs/2409.00696](https://arxiv.org/abs/2409.00696)

    Polyrating是一种基于最大后验估计的低成本评分系统，能够检测并量化人类评估中的偏差，将大语言模型的人工评估成本对新模型和新任务分别最高降低41%和77%，同时支持跨任务的公平模型比较。

    

    基于评分的人工评估已成为准确评估大语言模型（LLM）出色性能的重要工具。然而，当前的评分系统存在几个重要局限：首先，它们未能考虑会显著影响评估结果的偏差；其次，它们需要庞大且昂贵的偏好数据集才能获得准确的评分；第三，它们无法在不同任务之间对模型评分进行有意义的比较。为解决这些问题，我们提出了Polyrating，这是一种基于最大后验估计的表达性强且灵活的评分系统，能够以更低的成本对模型性能进行更细致、更全面的分析。Polyrating能够检测并量化影响人类偏好的偏差，从而确保更公平的模型比较。此外，Polyrating可以大幅降低人工评估的成本，对新模型最高可降低41%，对新任务最高可降低77%。

    arXiv:2409.00696v4 Announce Type: replace-cross  Abstract: Rating-based human evaluation has become an essential tool to accurately evaluate the impressive performance of large language models (LLMs). However, current rating systems suffer from several important limitations: first, they fail to account for biases that significantly influence evaluation results, second, they require large and expensive preference datasets to obtain accurate ratings, and third, they do not facilitate meaningful comparisons of model ratings across different tasks. To address these issues, we introduce Polyrating, an expressive and flexible rating system based on maximum a posteriori estimation that enables a more nuanced and thorough analysis of model performance at lower costs. Polyrating can detect and quantify biases affecting human preferences, ensuring fairer model comparisons. Further, Polyrating can reduce the cost of human evaluations by up to $41\%$ for new models and up to $77\%$ for new tasks b
    

