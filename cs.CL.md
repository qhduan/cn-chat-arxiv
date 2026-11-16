# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ModHarness-RL: Recasting Harnesses as Learnable and Evolving Programs for Agentic RL](https://arxiv.org/abs/2611.18533) | 提出 ModHarness-RL，将智能体强化学习中固定不变的手写支架重构为可从数据学习并随训练不断演化的程序，通过压缩程序（组装上下文）和引导程序（行为偏离时提供指导）两个核心组件，从轨迹中提出候选程序并经执行验证而持续进化。 |
| [^2] | [WaferTetris: Efficient Freeform Mapping on Wafer-Scale Accelerators](https://arxiv.org/abs/2611.18527) | 提出了WaferTetris系统，通过类俄罗斯方块的自由形状映射策略打破模板固定几何布局的限制，灵活地在晶圆级加速器上打包计算，从而提升处理单元利用率和可用内存容量。 |
| [^3] | [Can Mechanistic Interpretability Help Explain LLM Agent Failures? A Case Study on Web Element Hallucination](https://arxiv.org/abs/2611.18514) | 本文首次将机制可解释性系统应用于网页智能体幻觉失败分析，发现线性探针可在生成前预测幻觉（0.80 AUC）且引导能纠正85%的幻觉，但由于对提示格式敏感，这只是一种部分解释。 |
| [^4] | [MeReader: An Offline, Privacy-Preserving, Progress-Aware AI Assistant for Narrative eBook Reading](https://arxiv.org/abs/2611.18513) | MeReader是一个离线、隐私保护且进度感知的电子书阅读AI助手，通过混合向量相似度、BM25与摘要嵌入的检索架构，将内容检索与生成严格限制在读者当前阅读进度内以避免剧透，并证明2b-4b的中等规模本地模型可在可接受延迟下达到与更大模型相当的理解能力。 |
| [^5] | [Agentic induction of common data elements and information extraction from sarcoma case report abstracts](https://arxiv.org/abs/2611.18506) | 该研究证明大语言模型智能体能够从2万余篇肉瘤病例报告摘要中自动归纳出通用数据元素模式，并以94.9%的抽取一致率实现了大规模、带证据关联的临床信息抽取。 |
| [^6] | [Interactive Program Induction with Language Model Priors](https://arxiv.org/abs/2611.18485) | 本文提出以语言模型作为程序先验的贝叶斯交互式程序归纳方法，通过贪心选择最具信息量的输入向预言机查询，逐步将后验概率集中于目标程序，以化解从有限示例归纳程序时固有的歧义性问题。 |
| [^7] | [Do Verifier Judgments Survive Abstraction? Measuring Verification Transport for LLM Agent Evaluation](https://arxiv.org/abs/2611.18471) | 本文提出“验证迁移”（Verification Transport）这一可靠性测试方法，通过在保持任务、轨迹、结果等条件不变的情况下构建原生、抽象和恢复三种匹配视图，测量LLM验证器的判断在经历工具组合、委托边界、历史压缩等抽象化后是否仍能保持稳定可靠。 |
| [^8] | [LitArc-Bench: Disentangling Text-Specific Reader-Response Dynamics from Position and Lexical Affect](https://arxiv.org/abs/2611.18412) | 该论文提出LitArc-Bench基准（39段中文文学节选、9,020个VAD评分），通过位置先验与词汇情感对照实验，首次证明模型与人类情感轨迹的匹配中存在超出叙事位置和情感词汇的文本特异读者反应动态成分。 |
| [^9] | [Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent](https://arxiv.org/abs/2611.18397) | 提出Gamer，一个端到端的通用视觉-语言-动作基础模型，通过海量键鼠交互轨迹预训练、稀疏思考指令微调和稳定的多轮强化学习，实现了在复杂2D/3D交互环境中执行长时程任务的实时游戏智能体。 |
| [^10] | [Evaluating Human Sensitivity to Gender Biased AI-Generated Content through Human Assessment of Google's Autocomplete Function](https://arxiv.org/abs/2611.18371) | 研究发现AI来源标注不会影响人们对性别相关陈述的反应时间与认同判断，表明人类评估AI生成内容时可能不如评估人类创作内容那样审慎。 |
| [^11] | [f-Divergence Self-Play for Tabular Anomaly Detection via Large Language Models](https://arxiv.org/abs/2611.18370) | 提出DiSPaT，一个基于f散度最小化的自我博弈微调框架，通过交替优化缩小模型生成样本与真实正常数据之间的分布差距，从而增强大语言模型对表格数据正常模式的理解，提升表格异常检测性能。 |
| [^12] | [Envisioning Agentic Robotics: A New Era of Robotic Systems](https://arxiv.org/abs/2611.18359) | 本文提出“智能体机器人”这一新范式，主张由基础模型与具身框架构成的“具身系统2”是智能体机器人的决策核心，并呼吁学界建立统一定义、评估框架与研究路线图。 |
| [^13] | [Marimba: A System One Model for Speech](https://arxiv.org/abs/2611.18330) | Marimba基于冻结的Qwen3-ASR 0.6B特征构建了一个轻量级“系统一”语音决策接口，通过零样本评分、条件化读出头和中间层探针，在词项检测、轮次结束判断、语言识别、个人信息检测和说话人变化检测等多项语音决策任务上取得了优异且良好校准的性能。 |
| [^14] | [Learning Word-Level Explanations of Text Reuse in Latin Literature with Little Annotation](https://arxiv.org/abs/2611.18319) | 该论文提出一种仅需少量标注的方法：先利用合成文本对训练模型，再通过主动学习优化，最终用指针网络对拉丁文学中的文本复用进行词级对齐与改写操作标注，从而解释后世作者如何改写原文。 |
| [^15] | [When evaluation becomes intervention in clinical AI](https://arxiv.org/abs/2611.18297) | 临床AI中的AI评判者在被用于选择提示词或训练模型时会本身成为干预的一部分，并可能偏向其忽视的错误，因此评判者报告必须包含特定内容，且每项由评判者驱动的变更都需要独立的临床测试。 |
| [^16] | [Clean Tickets, Wrong Orders: Catching Silent Errors with a Language Model in Your Browser Tab](https://arxiv.org/abs/2611.18284) | 本文展示了一个完全在浏览器本地运行的小型语言模型演示系统，利用移植到TypeScript的PIZZA匹配器实时捕捉“静默错误”（即模型生成格式正确却无人真正下过的订单），其表现与服务器端评估几乎完全一致。 |
| [^17] | [GT-FENG: General Theory of Factual Evaluation and Normative Grounding in Language Models *Author(s)](https://arxiv.org/abs/2611.18279) | GT-FENG是首个统一上下文/时间正确性与基于规则的错误归因的通用框架，通过对照适用规则检验语言模型答案并追溯错误来源，在Medicaid收入上限问答中使仅匹配其他州数值的错误答案减少68-78%。 |
| [^18] | [MamaReason: Unfinished Reasoning and Answer Recovery in Clinical Language Models](https://arxiv.org/abs/2611.18264) | 该研究发现临床语言模型通过答案续写恢复的诊断大多已显式存在于未完成的推理轨迹中，仅直接读取这些诊断承诺（无需额外生成）即可挽救大部分续写所恢复的失败案例。 |
| [^19] | [NER-D: A cross-domain benchmark for German NER](https://arxiv.org/abs/2611.18257) | 该论文提出了一个覆盖 9 个领域、包含 13 个数据集的德语 NER 跨领域基准 NER-D，并发现经 LoRA 微调的小型开源模型能大幅超越专有 LLM 的性能。 |
| [^20] | [A Multi-Layered AI Pipeline for BPMN Generation from Natural Language](https://arxiv.org/abs/2611.18157) | 该论文提出了一种集成基于Transformer的架构、大语言模型和本体填充的多层AI流水线，通过微调RoBERTa模型进行实体提取、利用LLM推理捕获非相邻控制流关系，并采用动态少样本策略提升有限数据下的泛化能力，从而实现从自然语言描述自动生成BPMN流程图。 |
| [^21] | [Folding Language Models into Themselves](https://arxiv.org/abs/2611.18113) | 提出Concertina方法，将N个相邻教师层折叠为执行N次的共享块并配合低秩修正和逐token门控，直接从教师权重构建紧凑学生模型，在22B token训练量下保留教师约90%的性能。 |
| [^22] | [Contagious Fingerprints: When LLMs Inherit Another Model's Identity](https://arxiv.org/abs/2611.18104) | 本文发现大语言模型的指纹具有“传染性”——接续其他模型对话的模型会继承前者的指纹，导致API中转中的模型偷换检测率大幅下降（如从0.97降至0.72），其根源在于模型对历史对话风格的模仿。 |
| [^23] | [VEPO: Mitigating Multilingual Hallucinations in LVLMs via Visual Evidence-guided Preference Optimization](https://arxiv.org/abs/2611.18096) | 提出VEPO训练时方法，利用英语查询定位视觉证据并构建证据充分性不同的偏好对进行偏好优化，无需外部模型即可将可靠的视觉定位能力从英语迁移到其他语言，有效缓解大型视觉语言模型在多语言（尤其是低资源语言）场景下的对象幻觉问题。 |
| [^24] | [Deep Latents, Long Memory: Latent persistence explains temporal correlations in transformer activations](https://arxiv.org/abs/2611.18089) | 该论文提出用层激活的自相关函数作为抽象的可操作度量，发现潜变量的持久编码会在自相关中产生慢衰减尾部，且这些尾部随Transformer层加深而增长，从而解释了深层模型编码抽象潜变量的机制。 |
| [^25] | [Agent-as-a-Judge](https://arxiv.org/abs/2611.18067) | 该论文提出“智能体即裁判”新范式，让评估器能够像智能体一样主动搜集固定上下文之外的缺失证据（如未记录的环境状态和需要外部核验的事实性声明），从而对AI系统做出更可靠的评估裁决。 |
| [^26] | [FEND: Far-End Detection in Outbound Voice AI Calls](https://arxiv.org/abs/2611.18050) | 提出FEND框架，通过分析远端音频与流式转录文本，对语音AI外呼中的远端类型（真人、语音信箱、IVR、AI）进行细粒度检测，突破了传统二值应答机检测无法区分机器接听类型的局限。 |
| [^27] | [PerfArena: Ranking LLMs by Code Performance](https://arxiv.org/abs/2611.18030) | PERFARENA是一个开源、可复现的评测竞技场，通过测量LLM生成代码在多种语言和竞赛中的延迟、峰值内存与能耗等性能指标来对模型排名，弥补了现有基准仅关注正确性、语言覆盖窄且忽略能耗的不足。 |
| [^28] | [Toward Universal Translatability: On the Statistical Ontology of Generative AI](https://arxiv.org/abs/2611.17963) | 本文提出翻译是当代生成式AI的起点与规范目标，其“通用翻译机器”理想背后隐含一种可追溯至韦弗1949年机器翻译备忘录的“统计本体论”形而上学承诺——即相信现实具有可通过数据与建模恢复的可计算统计结构。 |
| [^29] | [WebArena-Pro: A Heterogeneous, Multimodal, Reproducible Benchmark for Web Agents](https://arxiv.org/abs/2611.17955) | WebArena-Pro是一个涵盖20个自托管Web应用、300个任务的异构多模态可复现基准测试，首次支持对Web智能体理解和运用音频、视频内容能力的评估。 |
| [^30] | [In the Eye of the Beholder: Human Label Variation in Textual and Visual Entailment](https://arxiv.org/abs/2611.17954) | 该论文提出了VariNLI-VE数据集，通过让同一批标注者对成对的NLI文本实例及其视觉对应版本重新标注（包含标签、置信度和解释，共2,250条标注），系统研究人类标注差异如何在文本与视觉两种蕴含任务之间变化。 |
| [^31] | [Can You Trust Your Finance Agent? Drift, Diagnosis and Auditable Intervention on One Shipped Financial Research Agent](https://arxiv.org/abs/2611.17939) | 论文对一款现成发布的金融研究智能体进行系统审计，发现其基准分数不稳定且运行成本波动高达17%，并按指令、工具、模型等责任层对失败轨迹进行归因分类，其诊断结果与人类盲评者一致率达94%，进而提出可审计的针对性干预方法。 |
| [^32] | [MineralExp: Benchmarking LLM Agents for Experimental Design in Real-World Mineral Exploration](https://arxiv.org/abs/2611.17931) | 提出首个面向真实矿产勘探场景的MineralExp基准，用于评估LLM智能体自动化地球化学实验设计的能力，并解决了实验方法多样性与主观性高、开放式方案难以客观评分两大难题。 |
| [^33] | [AmbiSteer: Selectively Eliciting Ambiguity-Aware Responses in Vision-Language Models](https://arxiv.org/abs/2611.17900) | 提出 AmbiSteer 轻量级框架，通过结合激活引导与提示方法，在推理时选择性地引出视觉语言模型的歧义感知行为，使其能够正确处理视觉问答中的指代歧义问题。 |
| [^34] | [Securing Multi-Agent Systems by Design](https://arxiv.org/abs/2611.17883) | 本文提出了响应操纵攻击以揭示多智能体系统面对恶意响应的安全隐患，并进一步提出基于响应间语义相似性的安全聚合规则SecMAS来有效防御此类攻击。 |
| [^35] | [Delayed Effects of Temporary Weight Decay in Muon Pretraining](https://arxiv.org/abs/2611.17876) | 该研究揭示了Muon预训练中临时权重衰减的延迟效应——早期衰减虽在结束时增加验证损失，但通过留下持久的权重范数赤字使相对步长保持较大，最终反而降低最终损失，且小规模模型上选出的衰减调度无需重新调参即可成功迁移到更大规模模型。 |
| [^36] | [UltraBERT-JEV: Encoder models for System One Applications](https://arxiv.org/abs/2611.17867) | 提出UltraBERT-JEV编码器模型，通过单次前向传播直接从多个选项中选择答案而无需生成文本，配合“是-否”差值打分等微调方案，准确率最高达72.1%并超越同数据训练的0.6B解码器模型，同时推理速度快一倍以上（29毫秒对比68毫秒）。 |
| [^37] | [Is Abliteration Getting Harder? Not Uniformly: Refusal Removal by Direction-Based Ablation and ARA in Heretic on the Qwen3.8, Gemma 4, Olmo 3.1 and Mistral Small 3.2 LLMs Versus Their Predecessors](https://arxiv.org/abs/2611.17846) | 本研究在五组新旧模型对上，使用多种基于方向的消融方法与 ARA 在统一实验条件下进行系统对比，发现拒绝移除的难度并未随着模型版本的更新而普遍增加。 |
| [^38] | [BLiMP-RO: A Benchmark of Linguistic Minimal Pairs for Romanian](https://arxiv.org/abs/2611.17836) | 本文提出了BLiMP-RO——一个涵盖50种语法现象、包含493个语言学家构建及5942个半自动生成句子对的罗马尼亚语语法评测基准，并通过从零训练GPT-2和GPT-BERT模型探究了有限语言输入下的语法泛化能力。 |
| [^39] | [PaperCritic: Training AI Reviewers to Assess the Soundness of Papers](https://arxiv.org/abs/2611.17834) | 该论文提出PaperCritic模型，通过构建包含23K条质量标注审稿观点的PaperSound-23K数据集，并以精确率和召回率的F1分数作为奖励进行强化学习训练，实现了专注于客观评估论文可靠性的AI审稿人。 |
| [^40] | [G-CAST: Governed Continual Self-Training for E-Commerce Agents under Evolving Demands](https://arxiv.org/abs/2611.17814) | G-CAST提出了一套受治理的持续自训练系统，使电商LLM智能体能够在用户需求漂移和执行框架不断演进的条件下自动持续适应，从而摆脱滞后且高成本的人工周期性再训练。 |
| [^41] | [AI Reviews Are Biased Towards the Appendix](https://arxiv.org/abs/2611.17811) | 研究发现GPT和Claude等大语言模型在审稿时约50%以上的批评意见来自论文附录，而人类仅有8.4%，且AI的批评多集中于与核心贡献无关的细节，揭示了AI审稿存在系统性偏向附录的偏差。 |
| [^42] | [CarryOn: Learning What to Keep in Context for Long-Horizon LLM Agents](https://arxiv.org/abs/2611.17808) | CarryOn是一个轻量级上下文管理器，通过学习的选择器和自适应保留策略，将紧凑的执行状态与细粒度的来源链接证据相结合，无需额外LLM调用即可为长程LLM智能体实现可靠的上下文管理。 |
| [^43] | [RAG-FGO: Dynamic Optimization of Query-Rewriting Agents for Generative RAG](https://arxiv.org/abs/2611.17788) | 提出RAG-FGO动态优化框架，将查询重写建模为混合离散-连续空间中不可微的黑盒优化问题，通过可自适应优化的查询重写智能体联合优化提示模板与解码参数，以应对生成式RAG中的噪声反馈和静态策略泛化不足的问题。 |
| [^44] | [LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent](https://arxiv.org/abs/2611.17780) | 提出非自回归 7B 模型 LuthorSQL，作为 LLM 智能体的快速“系统 1”门控，在单 GPU 上每题 0.84 秒即达到 BIRD 测试集 68.03% 的执行准确率，仅在不确定时才将查询升级给较慢的智能体，从而大幅降低成本并保护数据隐私。 |
| [^45] | [Activation Steering the Nan-Shan in Typhoon](https://arxiv.org/abs/2611.17768) | 本文介绍了基于康拉德《台风》的视听装置作品，通过让大语言模型以自身输出为食并采用一种新颖的自适应激活引导方法，探索AI生成文本随时间推移变得日益陌生和失控的过程，以此反思人类文化当下的人工智能时刻。 |
| [^46] | [GDP.xlsx: Benchmarking Frontier Models on Professional Workbooks](https://arxiv.org/abs/2611.17766) | 该论文提出GDP_xlsx基准，利用13个领域专业人士撰写的70个真实工作簿任务和791项专家评分准则来评估前沿AI模型的电子表格处理能力，发现最强模型Gemini 4 Argon的通过率也仅有38.3%，表明该领域仍有巨大的提升空间。 |
| [^47] | [See Every Block Once: Variance-Reduced Post-Training for Masked Diffusion Language Models](https://arxiv.org/abs/2611.17754) | 提出一种全块估计器，通过对每个块恰好评估一次并取平均，在保持无偏的同时消除块选择带来的组间方差，从而显著降低掩码扩散语言模型后训练中的估计方差。 |
| [^48] | [Superfluid: Local AI Server for Distributed LLM Inference](https://arxiv.org/abs/2611.17753) | Superfluid是一个开源本地LLM推理服务器，通过统一调度器在单设备上协调多智能体与交互用户共享模型服务（支持llama.cpp、MLX、BaseRT等运行时），并能将多台异构闲置机器聚合为单一API端点实现分布式推理。 |
| [^49] | [What Do Multilingual Self-Supervised Speech Encoders Know About Sentence Meaning?](https://arxiv.org/abs/2611.17686) | 该研究通过将冻结语音表示线性映射到文本嵌入空间并进行跨模态检索，首次大规模评估了自监督语音模型在76种语言中的句子级语义可解码性，发现语义可访问性在语言间高度不均衡、参数扩展偏向高资源语言，且ASR调优虽提升语义可解码性却降低了表示的语言无关性。 |
| [^50] | [What LLM Agents See and What Users Say](https://arxiv.org/abs/2611.17639) | 该研究运用语言学预设理论系统分析了LLM智能体在工具输出与用户陈述冲突时的行动边界，发现只有当搜索结果明确“列出”多个匹配项时才能阻止大多数智能体基于错误假设行动，而仅“计数”或依赖用户声明则不足以防止误操作。 |
| [^51] | [More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs](https://arxiv.org/abs/2611.17610) | 本研究通过综合516个多智能体配置的元分析发现，多智能体大语言模型的表现优势主要来自额外的样本数量和测试时计算量，而非智能体间的协调机制本身——在与资源匹配的单模型基线公平对比时，多智能体系统在74%的情况下并无优势甚至更差。 |
| [^52] | [LightTAR: Light and Fast Target-Aware Representations for Multimodal Tabular Learning](https://arxiv.org/abs/2611.17600) | LightTAR是一种简单低成本的降维增强方法，它利用正则化PLS从编码器最后一层和中间层提取目标感知特征来补充标准PCA表示，从而在21种表格模型上普遍提升多模态表格学习的预测性能。 |
| [^53] | [Small Open-Weight Language Models for Crowd Decision-Making: Capabilities and Limits of Output-Level Fusion and Linguistic Aggregation](https://arxiv.org/abs/2611.17589) | 该论文首次系统检验了在机构本地基础设施上运行的小型开放权重语言模型，能否通过重复查询与基于二元组语言表示的输出级融合来稳定群体决策结果，发现连续评估可保持信息分辨率，但对参与评论者的敏感性在降低运行间变异后依然存在。 |
| [^54] | [CamSpokenNLI: A Pilot Resource for Cross-Variety Transfer Evaluation in Cameroon Pidgin English and Camfranglais](https://arxiv.org/abs/2611.17573) | 本文推出了首个面向喀麦隆皮钦英语和Camfranglais两种语言变体的口语自然语言推理试点数据集CamSpokenNLI（共228条条目），实验发现XLM-RoBERTa基线表现仅略高于随机水平且与仅使用假设的输入难以区分，凸显了这些低资源变体上NLI任务的挑战性。 |
| [^55] | [AI-Voyager: A Public-Interest System for Human-AI Civic Reasoning to Contest and Shape the Rules That Govern AI](https://arxiv.org/abs/2611.17564) | AI-Voyager是一个公益性的“人机公民推理”系统，它将3,653份各级政府治理文档转化为可溯源的证据段落，让居民与AI协同理解、质疑并起草修订治理AI的规则，在文档检索评测中达到83.0%的宏观召回率。 |
| [^56] | [Benchmarks Don't Talk Back: Learning Through Self-Disagreement](https://arxiv.org/abs/2611.17563) | 该论文提出“辩证能力”这一概念——即知道何时坚持立场、何时向更优答案让步的能力，证明其无法通过准确率训练或推理时的自我反思获得，而应借鉴哲学辩证法思想，通过模型自身生成的对立观点进行训练。 |
| [^57] | [A Push Down the Mountain: Framing Effects and Internal State Interventions on Collusive Behavior in a Long-Horizon LLM Agent](https://arxiv.org/abs/2611.17515) | 本研究将Gemma 3 12B置于基于真实伦敦白银定价诉讼记录的市场模拟中，发现损失与指责等框架效应以及向模型第22层注入相应内部状态能显著提高LLM智能体参与市场合谋操纵的倾向。 |
| [^58] | [Creating Steering Vectors via Input Optimisation](https://arxiv.org/abs/2611.17481) | 本文提出ActGCG方法，通过优化输入后缀来模拟激活引导向量的效果，使攻击者无需白盒访问模型内部状态，即可通过修改输入文本实现与激活引导相同的行为操控，包括语言切换和越狱攻击。 |
| [^59] | [Poetry 'resurrection' from LLMs](https://arxiv.org/abs/2611.17464) | 本文通过两个基于对联的实验探究大语言模型能否创作“无限诗歌”，发现AI虽能生成有趣的词对，但在创造力和对诗意关系的理解上仍逊于人类。 |
| [^60] | [Tracing Knowledge and Reasoning in LLM](https://arxiv.org/abs/2611.17461) | 该论文提出了一个以任务为中心的评估框架，结合注意力、激活与表示信号来刻画大语言模型在知识型与推理型任务中的内部处理模式，将评估从分数层面深入到内部机制层面。 |
| [^61] | [Unfinished-Bench: Measuring Rule Violations by Persistent Agents in Realistic Work Environments](https://arxiv.org/abs/2611.17460) | Unfinished-Bench是一个在真实工作环境中评估AI智能体持久性与规则遵守之间冲突的基准测试，发现任务持久性会显著提高智能体违反内部文档中规则的比例（从32.2%升至43.6%），而违反用户明确陈述的同一规则的比例仅为8.5%。 |
| [^62] | [ArchSpace: Recording, Reviewing, and Referencing LLM Architecture Research](https://arxiv.org/abs/2611.17332) | ArchSpace 是一个公共平台，通过记录实验与推理、社区评审和后续引用来系统化大语言模型架构研究，并借助缩放定律在昂贵的规模化训练前预测模型表现，从而同时弥补工业界缺乏创新公开记录和学术界评估受限的两大缺口。 |
| [^63] | [LazyWeights: Decode-on-Demand Edge LLMs](https://arxiv.org/abs/2611.17108) | LazyWeights在保留现有量化格式的基础上，通过对权重进行无损Huffman编码并在DRAM与L2之间用小型ASIC单元按需解码，在不改动GPU内核的前提下使边缘设备上的模型体积缩小5.7%-31.4%，并将每token的延迟和能耗最多降低约一半。 |
| [^64] | [Detection Is Not Early Warning: Evaluating Token-Level Alarms on Language Model Failures](https://arxiv.org/abs/2611.16803) | 该论文提出了PREWARN评估协议，将监控器的评估从单纯检测失效转向在失效token起点之前发出预警，从而为停止或干预生成过程争取时间。 |
| [^65] | [Indic-Task Gains and Regressions after Reinforcement Learning](https://arxiv.org/abs/2611.16228) | 该研究通过GRPO强化学习显著提升了覆盖英语和八种印度语言的30B MoE模型的数学及多数印度语言任务表现，但发现英语回答倾向增强导致跨语言问答大幅退化，且加入语言奖励的第二次RL训练虽改善了指令遵循，却以牺牲推理和编码能力为代价。 |
| [^66] | [JudgeMix: Judge-Guided Mixed-Rollout Data Synthesis for LLM Agents](https://arxiv.org/abs/2611.16174) | 提出JudgeMix框架，利用裁判模型对每一步进行语义评估，动态决定目标模型与专家模型的交替介入时机，从而生成与目标模型自身分布匹配、且不覆盖其已正确动作的高质量SFT训练数据。 |
| [^67] | [Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models](https://arxiv.org/abs/2611.13223) | 本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。 |
| [^68] | [on\'ona: Twin Embeddings with Tied Unembeddings](https://arxiv.org/abs/2611.10556) | on'ona 提出了一种双嵌入表参数化方法，通过仅依赖当前输入 token 的门控来混合两个共享嵌入表定义的 softmax 分布，在保留权重绑定省参数优势的同时增强了输出头表达能力，在多种架构和三个规模下显著降低验证损失并提升下游任务准确率。 |
| [^69] | [How Much Is Left Matters: False Positives and False Negatives in Partial-Prompt Contamination Tests](https://arxiv.org/abs/2611.10307) | 该论文审计了部分提示污染测试，发现剩余文本越短假阳性越多（短剩余文本贡献了大部分 ROUGE 标记，在真阴性题目上 80% 前缀时误标率高达 8.6%），且该测试无法检测通过蒸馏传播的上游污染。 |
| [^70] | [All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States](https://arxiv.org/abs/2611.10123) | 该研究提出开放权重掩蔽内省（OWMI）方法，通过严格的对照实验发现：语言模型无法内省地报告其残差流中被注入的“疼痛方向”激活状态，其自我报告相比其自身被改动文本所能揭示的信息几乎没有额外信息量。 |
| [^71] | [ABHBench: Evaluating Moral Decision-Making of Foundation Models from an Agentic Perspective](https://arxiv.org/abs/2611.09326) | 该论文提出了基于《底特律：变人》游戏构建的多模态基准ABHBench，首次让基础模型以第一人称行动主体的身份在分支叙事场景中面对道德困境，并从阿西莫夫机器人三定律启发的六个维度评估其道德决策能力。 |
| [^72] | [AttentionSwitch: Jailbreak Defense for LLMs via Joint Learning of Sparse Attention-Head Selection and Scaling](https://arxiv.org/abs/2611.08752) | AttentionSwitch是首个通过联合学习稀疏注意力头的选择与缩放，同时优化越狱防御中“在哪里干预”和“如何干预”两个决策的大语言模型防御方法。 |
| [^73] | [NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria](https://arxiv.org/abs/2611.06504) | 该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。 |
| [^74] | [Who Pays for the Knowledge Models Copy? Model Collapse as Rogers' Paradox](https://arxiv.org/abs/2611.06022) | 该论文将模型坍塌重新解读为文化演化中的罗杰斯悖论，指出问题的根源在于外部信息供给本身，并证明仅用完全合成数据拟合的罗杰斯模型能精确预测语言模型在不同新鲜数据混合比例下稀有知识的保留量。 |
| [^75] | [Integrate the Reward, Don't Sample It: Window Likelihood for Interruptions and Backchannels in Full-Duplex Speech Models](https://arxiv.org/abs/2610.35581) | 该论文发现基于整轮展开的GRPO时机奖励在训练全双工语音模型时存在零梯度陷阱与信用泄漏两大缺陷，并提出“窗口似然”方法，通过对停止决策进行闭式积分直接最大化正确让出/保持的对数概率，在HumDial-FDBench上取得了最佳的让出/保持F1表现。 |
| [^76] | [SPECMEM: Accelerating Agentic Tool Calling via Live Memory Management](https://arxiv.org/abs/2610.31840) | SpecMem 提出一种在部署全过程中持续活跃管理的工具调用记忆机制，通过在线写回已验证调用（配合 LRU 淘汰）和副作用门控，使投机解码在多用户多会话场景下相比冻结草稿库保持约 27% 的平均接受 token 优势。 |
| [^77] | [Pluralistic Alignment in Large Language Models: Objectives, Human References, and Preference Representations](https://arxiv.org/abs/2610.24471) | 该论文提出了一个双层分类框架来系统梳理大语言模型多元对齐研究，第一层区分三种对齐目标（表征多元观点的Overton多元性、适应特定视角的可引导多元性、匹配人群分布的分布多元性），第二层区分三类人类参照（聚合人群、国家和地区、子群体），并将偏好表征、实现机制等视为贯穿性设计选择。 |
| [^78] | [Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making](https://arxiv.org/abs/2610.23170) | Chinese Decision 是一系列中文语言模型，通过单次前向传播直接对候选选项的 logits 进行归一化得到概率分布，无需自回归生成，在医学、法律、金融等领域的决策任务上实现了高准确率与良好校准。 |
| [^79] | [Learning to Read Out: Unembedding Dynamics in Language Model Pretraining](https://arxiv.org/abs/2610.21025) | 提出参数轨迹交叉编码方法，仅利用检查点权重直接追踪语言模型预训练中读出层的动态，发现Pythia在预训练0.4%–1.1%的早期窗口内读出层发生重组，且隐藏状态在读出层表达之前就已编码任务信息。 |
| [^80] | [Similar Models, Broken Odds: Where Luce's Axiom Fails in Chatbot Arena](https://arxiv.org/abs/2610.19095) | 该研究发现Chatbot Arena所依赖的Bradley-Terry模型（即Luce选择公理）在高度相似的模型之间会系统性失效——相似模型间的对决比分数预测的更具决定性（最高达1.37倍），并提出一个仅含4-6个参数、主要通过建模平局的修正方法，其预测能力超越了Bradley-Terry及含数百参数的复杂模型。 |
| [^81] | [Some complete $\omega$-powers of a one-counter language in the difference hierarchy](https://arxiv.org/abs/2610.16473) | 本文证明了对于任意自然数 $n,k\geq 1$，存在被单计数器自动机接受的有限语言，其 $\omega$-幂在差分层次中是 $D_{2n-1}(\mathbf{\Pi}^0_k)$-完备的，并给出了对偶类 $D^-_{2n}(\mathbf{\Pi}^0_k)$ 的条件性结果。 |
| [^82] | [Inject a Turn, Not a Token: Learning Where to Redirect LLM Reasoning for Self-Correction](https://arxiv.org/abs/2610.13672) | 该论文提出无需微调的 UPTURN 自我纠错框架，通过仅用最终答案正确性训练的 13 万参数轻量 MLP 头学习失败推理中的最佳干预位置，并将纠错指令作为新的用户轮次而非行内标记注入，从而有效重定向模型推理，在四个推理模型和六个数学基准上均提升准确率。 |
| [^83] | [From Reading Code to Reading Spec: A Verified Layer for LLM-Driven Codebase Maintenance](https://arxiv.org/abs/2609.06383) | 论文提出PROOF系统，将代码库拓扑抽象为分层自然语言规约，并通过从规约重构源代码来证明语义等价，使LLM能够借助规约间接、可靠地执行代码库维护并防止语义漂移。 |
| [^84] | [Beyond Raw Transcripts: Structured Persona Extraction for LLM-Based Digital Twins](https://arxiv.org/abs/2608.20344) | 本文提出，数字孪生预测准确性的关键瓶颈在于人物画像信息的结构组织方式，而非信息量，并通过引入基于消费者行为理论的BDE结构化模式，将预测准确性提升1.91个百分点。 |
| [^85] | [BBOWP-Bench: Evaluating LLMs on Black-Box Optimization Word Problems](https://arxiv.org/abs/2608.02612) | 该论文提出BBOWP-Bench基准，首次评估大语言模型能否从自然语言描述的黑箱优化文字问题中同时推断出合适的搜索空间并选择恰当的优化算法。 |
| [^86] | [Characterizing Treatment-Context Medication Evidence Across Clinic Notes and Structured EHR Medication History](https://arxiv.org/abs/2608.01570) | 该研究构建了以临床笔记为依据的用药参考标准，通过大语言模型提议、确定性规范投影和词汇清理将用药归一化的规范一致性从81.25%提升至93.58%，并量化了语义与时间匹配对临床笔记与结构化EHR用药史重叠程度的影响。 |
| [^87] | [URSA: Chemistry-Aware Benchmark for Utilitarian Retrosynthesis Assessment](https://arxiv.org/abs/2607.04688) | URSA提出了一个化学感知的逆合成评估基准框架，首次从形式指标（如收敛到商业可得起始原料）和模拟专家化学家判断的化学合理性两个维度，对端到端逆合成模型与大语言模型进行客观、可解释的比较评估。 |
| [^88] | [When Simpler Is Better: Evaluating Translation Pipelines for Medieval Latin Manuscripts](https://arxiv.org/abs/2607.03836) | 在中世纪拉丁手稿翻译任务中，参数量少几个数量级的领域专用OCR模型反而优于通用视觉语言模型（字符错误率最多降低4.3倍），并据此构建了包含1,383条对齐图像、转录与译文的新型数据集IPC。 |
| [^89] | [QVal: Evaluating Dense Supervision Signals for Long-Horizon LLM Agents](https://arxiv.org/abs/2606.32034) | 该论文提出QVal——一个无需训练的测试平台，通过衡量密集监督信号在每个状态下对候选动作的排序与参考策略回报排序的一致性（Q-对齐性）来直接评估信号质量，并在四个环境中对七个方法族的21种方法进行基准测试，发现简单提示基线平均Q-对齐性最高且性能按方法族高度聚集。 |
| [^90] | [Comparing Human and Automatic Recognition of Dutch Dysarthric Continuous Speech: A Case Study](https://arxiv.org/abs/2606.30237) | 本研究通过案例研究发现，重度构音障碍的荷兰语语音识别对人类和现成ASR系统都极为困难（WER超过70%），但经过个性化微调的DSR模型性能超越了人类听者，正接近可用于支持日常交流的水平。 |
| [^91] | [From Facts to Insights: A Persona-Driven Dual Memory Framework and Dataset for Role-Playing Agents](https://arxiv.org/abs/2605.25693) | 提出了DualMem双记忆框架，将记忆解耦为事实认知与人格洞察两条流，并配合RoleMemo数据集，使仅40亿参数的模型在长期角色扮演中的人格忠实度超越了基于DeepSeek-V3.2的零样本框架。 |
| [^92] | [SA-RAG: Structuring Noisy Conversations for Retrieval-Augmented Generation](https://arxiv.org/abs/2605.24366) | 提出SA-RAG框架，通过质量感知的结构归纳方法直接从噪声对话数据中构建表格形式的可靠中间结构，从而提升检索增强生成在嘈杂上下文中的表现。 |
| [^93] | [TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation](https://arxiv.org/abs/2605.22355) | 该论文发布了包含四个中国城市近1300万条路线规划记录的大规模数据集TransitLM，证明大语言模型可完全从数据中学习公交路线规划，无需地图即可实现从起终点信息到路线的端到端生成。 |
| [^94] | [SpecHop: Continuous Speculation for Accelerating Multi-Hop Retrieval Agents](https://arxiv.org/abs/2605.21965) | SpecHop提出一种连续推测框架，通过维护多个推测线程并异步验证、提交或回滚预测结果，在不改变模型最终轨迹的前提下无损加速多跳检索智能体，显著降低端到端延迟。 |
| [^95] | [Screen Before You Interpret: A Portable Validity Screen for Benchmark-Based LLM Confidence Signals](https://arxiv.org/abs/2604.17714) | 本文借鉴临床人格评估的有效性筛查传统，提出一种基于L、Fp和RBS指标的可移植筛查方法，在解读之前检验LLM置信度信号是否真正跟踪正确性，对20个前沿模型的评估发现15个有效、4个不确定、1个无效（DeepSeek-R1弃答了95%的正确答案）。 |
| [^96] | [Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection](https://arxiv.org/abs/2604.14397) | 提出一种用双语词典增强对齐器并结合过滤策略的跨语言词义投影方法，能够将WordNet式词汇资源自动扩展到新语言，在保持可解释性和资源高效的同时提升精确率。 |
| [^97] | [DRTriton: Large-Scale Synthetic Data Driven Reinforcement Learning for Triton Kernel Generation](https://arxiv.org/abs/2603.21465) | 提出DRTriton框架，通过CSP-DAG大规模合成数据、解耦奖励的课程化强化学习以及测试时搜索算法，训练大语言模型将PyTorch程序高效转换为优化的Triton内核。 |
| [^98] | [RECOVER: Robust Entity Correction via agentic Orchestration of hypothesis Variants for Evidence-based Recovery](https://arxiv.org/abs/2603.16411) | RECOVER是一个智能体式后ASR实体校正框架，它将多个ASR假设作为证据，通过实体检索、约束条件下的LLM校正、确定性防护栏过滤以及可触发反馈轮次的LLM评论家，有效恢复稀有和领域特定实体。 |
| [^99] | [FlashSampling: Fast and Memory-Efficient Exact Sampling](https://arxiv.org/abs/2603.15854) | FlashSampling将精确采样融合进LM头的矩阵乘法运算中，通过在片上逐块计算并添加Gumbel噪声避免在显存中物化logits张量，并以流式点对点通信替代all-gather，实现了跨最多8个GPU的快速、精确且内存高效的大词表采样。 |
| [^100] | [Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume](https://arxiv.org/abs/2602.24195) | 提出了UMPIRE，一个免训练且无需外部工具的多模态大语言模型不确定性量化框架，其受行列式点过程启发，通过计算新颖的“非相干性调整语义体积”，将语义嵌入离散度与响应二次熵相结合，从而高效支持多种输入输出模态。 |
| [^101] | [Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability](https://arxiv.org/abs/2602.17544) | 该论文提出用发送方-接收方框架评估模型间思维链推理传递的可靠性、有用性和安全性，并引入可验证性与可纠正性两个度量指标。 |
| [^102] | [ScrapeGraphAI-100k: Dataset for Schema-Constrained LLM Generation](https://arxiv.org/abs/2602.15189) | 本文发布了首个基于真实使用遥测数据构建的大规模模式约束抽取数据集 ScrapeGraphAI-100k，包含 93,695 条配对真实网页内容、提示词、模式与大语言模型响应及符合性标签的事件，覆盖 25,453 个独特模式，并揭示了模式复杂度增长带来的急剧失败阈值。 |
| [^103] | [ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression](https://arxiv.org/abs/2602.11008) | ROCKET是一种免训练的模型压缩方法，通过将逐层压缩分配建模为多选择背包问题，并结合基于校准集的单步稀疏矩阵分解，在全局压缩预算下实现了超越现有方法的最先进压缩性能。 |
| [^104] | [Beyond Trajectory Similarity: Evaluating Surgical Planning by Protocol Adherence](https://arxiv.org/abs/2601.10455) | 该论文指出现有基于轨迹相似性的手术规划评估范式混淆了任务完成与方案依从性，并提出首个基于专家定义手术方案规则、直接衡量方案依从性的评估框架，证明相似性指标会系统性误判规划质量。 |
| [^105] | [Investigating LLM Capabilities on Long Context Comprehension for Clinical Question Answering](https://arxiv.org/abs/2510.18691) | 该研究系统评估了大语言模型在纵向患者电子健康记录上的长上下文临床问答能力，发现医学微调模型相比基础模型并无可衡量的优势，较小的模型记忆能力更强，纵向补充记录有助于生成式问答，而推理模型仅在宽松指标下的复杂多记录问答中获益。 |
| [^106] | [HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities](https://arxiv.org/abs/2510.12200) | 提出了首个系统性评估框架HackWorld，通过视觉交互方式评估计算机操作代理利用真实Web应用程序漏洞的能力，涵盖36个跨11种框架和7种语言的真实应用。 |
| [^107] | [BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution](https://arxiv.org/abs/2510.08697) | 该论文提出了 BigCodeArena，一个基于实时执行环境的开放式代码生成人类评估平台，通过让人类与 LLM 生成代码的执行过程和结果进行交互，从而揭示出更可靠的人类偏好。 |
| [^108] | [Training Language Model Agents to Find Vulnerabilities with CTF-Dojo](https://arxiv.org/abs/2508.18370) | 提出了 CTF-Dojo——首个面向 LLM 训练的大规模可验证执行环境，包含 658 个容器化 CTF 挑战，并配套自动化流水线 CTF-Forge，可数分钟内将公开工件转化为即用型执行环境，仅需 486 条高质量轨迹即可训练出擅长漏洞发现的智能体。 |
| [^109] | [Cyber-Zero: Training Cybersecurity Agents without Runtime](https://arxiv.org/abs/2508.00910) | Cyber-Zero是首个无需运行时环境的网络安全智能体训练框架，通过利用公开CTF解题报告和基于角色扮演的LLM模拟来合成高质量交互轨迹，使训练出的智能体在三个CTF基准测试上相比基线模型最高取得11.4%的绝对性能提升。 |
| [^110] | [Beyond Gold Standards: Epistemic Ensemble of LLM Judges for Formal Mathematical Reasoning](https://arxiv.org/abs/2506.10903) | 本文提出了一种基于认知和形式化基础的LLM法官集成方法，通过细粒度、多层次的评估标准，系统地自动评估形式化数学推理中的语句自动形式化任务，弥补了现有评估方法的粗粒度缺陷。 |
| [^111] | [Mapping Seven Decades of Philosophy in Colombia: Dynamic Topic Modelling of Ideas y Valores](https://arxiv.org/abs/2412.04236) | 本研究通过动态主题建模技术分析哥伦比亚最具影响力的哲学期刊《Ideas y Valores》七十年来主题的演变，将哲学史的数据驱动研究拓展至哥伦比亚和拉丁美洲语境，发现价值理论、认识论和科学哲学是该地区最突出的哲学研究主题。 |
| [^112] | [LEGAL-UQA: Testing Constitutional Question-Answering Systems for Urdu Speakers with a Parallel English-Urdu Benchmark](https://arxiv.org/abs/2410.13013) | 提出了LEGAL-UQA，一个包含619个问答对的英乌尔都语平行基准，用于测试面向乌尔都语使用者的巴基斯坦宪法问答系统，研究发现模型排名依赖相似度度量方式、词匹配检索优于嵌入方法，且自动评判模型与人工评分的一致性有限。 |
| [^113] | [BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions](https://arxiv.org/abs/2406.15877) | BigCodeBench是一个新的代码生成基准测试，通过要求模型调用多样化函数库并准确理解复杂指令，来评估大语言模型解决实际任务的能力。 |
| [^114] | [Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models](https://arxiv.org/abs/2401.00788) | 该论文提出了Astraios套件——包含28个采用7种微调方法、4种规模（最大160亿参数）指令微调的代码大语言模型，系统研究发现全参数微调总体性能最佳，LoRA通常提供最有利的成本-性能权衡，且PEFT方法的效果随模型规模显著变化。 |
| [^115] | [On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model: An Empirical Study on Codex](https://arxiv.org/abs/2301.12868) | 本文首次对基于提示的大型代码语言模型（Codex）在语义解析任务中的对抗鲁棒性进行了实证研究。 |
| [^116] | [Fake News Detectors are Biased against Texts Generated by Large Language Models.](http://arxiv.org/abs/2309.08674) | 假新闻检测器倾向于将大型语言模型生成的内容标记为假新闻，而将人工编写的假新闻误分类为真实，我们提出了一种通过敌对训练和LLM改写的真实新闻等方法来解决这个问题，并取得了显著的改进。 |
| [^117] | [Rethinking Round-Trip Translation for Machine Translation Evaluation.](http://arxiv.org/abs/2209.07351) | 本文研究了往返翻译方法在机器翻译评估中的应用，发现去除统计机器翻译中的复制机制后，往返翻译可以恰当地反映前向翻译的性能，该方法可以在低资源语言的情况下进行自动评估。 |

# 详细

[^1]: ModHarness-RL：将支架重构为可学习、可演化的程序以服务于智能体强化学习

    ModHarness-RL: Recasting Harnesses as Learnable and Evolving Programs for Agentic RL

    [https://arxiv.org/abs/2611.18533](https://arxiv.org/abs/2611.18533)

    提出 ModHarness-RL，将智能体强化学习中固定不变的手写支架重构为可从数据学习并随训练不断演化的程序，通过压缩程序（组装上下文）和引导程序（行为偏离时提供指导）两个核心组件，从轨迹中提出候选程序并经执行验证而持续进化。

    

    智能体强化学习（RL）日益依赖于一个“支架”，即围绕模型的脚手架，它决定模型能看到自身交互中的哪些内容，以及模型会接收到什么样的引导。大多数现有方法都在一个全程固定不变的手写支架下进行训练。然而，一个有效的支架取决于领域、数据和模型本身，并且会随着模型的训练而持续变化。我们提出 ModHarness-RL，它将支架重构为从数据中学习、并随着模型通过强化学习训练而不断演化的程序。ModHarness-RL 将交互记录为结构化的状态，即一组带类型的块，而支架的两个核心组件就是基于这些块的程序：一个压缩程序，负责将积累的历史组装进模型的上下文中；一个引导程序，在行为发生偏离时提供指导。在强化学习训练过程中，候选程序从最近的轨迹中被提出，通过执行进行验证，并在过去的（轨迹上）进行重放。

    arXiv:2611.18533v1 Announce Type: new  Abstract: Agentic reinforcement learning (RL) increasingly relies on a harness, the scaffold around the model that decides what it sees of its own interaction and what guidance it receives. Most existing approaches train under a hand-written harness that stays fixed throughout. However, an effective harness depends on the domain, the data and the model, and keeps changing as the model trains. We introduce ModHarness-RL, which recasts the harness as programs that are learned from data and evolve as the model trains with RL. ModHarness-RL records the interaction as structured State, a set of typed blocks, and two core components of the harness are programs over these blocks, a Compression program that assembles the accumulated history into the model's context and a Steering program that supplies guidance when behavior deviates. During RL training, candidate programs are proposed from recent trajectories, verified by execution, and replayed over past
    
[^2]: WaferTetris：晶圆级加速器上的高效自由形状映射

    WaferTetris: Efficient Freeform Mapping on Wafer-Scale Accelerators

    [https://arxiv.org/abs/2611.18527](https://arxiv.org/abs/2611.18527)

    提出了WaferTetris系统，通过类俄罗斯方块的自由形状映射策略打破模板固定几何布局的限制，灵活地在晶圆级加速器上打包计算，从而提升处理单元利用率和可用内存容量。

    

    晶圆级加速器通过单片晶圆上互连的数十万个处理单元（PE）提供巨大的计算能力，但要发挥这一潜力需要高效的应用映射。其庞大的规模使得对这些系统进行编程以及将应用映射到其上变得极具挑战性。现有方法依赖于基于模板的映射来保持开发效率，但这些模板限制了映射的设计空间。特别是，它们在PE网格上强加了固定的几何布局，往往导致计算和内存资源利用不足。在这项工作中，我们提出了WaferTetris，一个通过自由形状映射消除这些几何限制的系统。其类俄罗斯方块的策略灵活地在整个晶圆上打包计算，在保留PE-任务代码的同时提高了PE利用率和可用内存容量。我们开发了一个基于递推的吞吐量模型，该模型捕获分支-汇合依赖关系以指导空间……（摘要截断）

    arXiv:2611.18527v1 Announce Type: cross  Abstract: Wafer-scale accelerators offer massive compute capacity through hundreds of thousands of processing elements (PEs) interconnected across a single wafer, but realizing this potential requires efficient application mapping. Their scale makes programming these systems and mapping applications onto them challenging. Existing approaches rely on template-based mappings to maintain productivity, but these templates restrict the mapping design space. In particular, they impose fixed geometric layouts over the PE grid, often leaving compute and memory resources underutilized.   In this work, we present WaferTetris, a system that removes these geometric restrictions through freeform mapping. Its Tetris-like strategy flexibly packs computation across the wafer, improving PE utilization and usable memory capacity while preserving PE-task code. We develop a recurrence-based throughput model that captures branch-and-join dependencies to guide spatia
    
[^3]: 机制可解释性能否帮助解释大语言模型智能体的失败？——网页元素幻觉的案例研究

    Can Mechanistic Interpretability Help Explain LLM Agent Failures? A Case Study on Web Element Hallucination

    [https://arxiv.org/abs/2611.18514](https://arxiv.org/abs/2611.18514)

    本文首次将机制可解释性系统应用于网页智能体幻觉失败分析，发现线性探针可在生成前预测幻觉（0.80 AUC）且引导能纠正85%的幻觉，但由于对提示格式敏感，这只是一种部分解释。

    

    机制可解释性（MI）在解释大语言模型（LLM）的内部计算方面已取得显著进展；然而，这些进展尚未在解释基于大语言模型的智能体的决策过程方面展现出同等的成功。本文探讨了MI工具在应用于真实智能体失败场景时所面临的挑战。我们聚焦于其中一种失败：网页智能体幻觉出网站上不存在的操作。我们对与真实网站交互的Qwen2.5-7B-Instruct应用了一系列机制分析。我们发现，基于最后一个提示token训练的线性探针能够在生成开始之前预测幻觉（0.80 AUC），并且沿探针方向进行引导可以纠正85%的幻觉，但会干扰7%的有效操作。然而，这些结果仅能部分解释智能体为何产生幻觉：探针方向对提示格式非常敏感，因而丧失了大部分……

    arXiv:2611.18514v1 Announce Type: new  Abstract: Mechanistic Interpretability (MI) has made significant progress in explaining the internal computations of Large Language Models (LLMs); however, these advances have yet to demonstrate comparable success in explaining the decision-making processes of LLM-based agents. This paper examines the challenges MI tools face when applied to realistic agentic failures. We focus on one such failure: web agents hallucinating actions that are not available on a website. We apply a series of mechanistic analyzes to Qwen2.5-7B-Instruct as it interacts with live websites. We find that a linear probe trained on the last prompt token can predict hallucinations before generation begins (0.80 AUC) and steering along a probe direction corrects 85\% of hallucinations, but disrupts 7\% of valid actions. However, these results only provide a partial explanation of why the agent hallucinates. The probe direction is sensitive to prompt formatting, losing much of 
    
[^4]: MeReader：面向叙事类电子书阅读的离线、隐私保护、进度感知AI助手

    MeReader: An Offline, Privacy-Preserving, Progress-Aware AI Assistant for Narrative eBook Reading

    [https://arxiv.org/abs/2611.18513](https://arxiv.org/abs/2611.18513)

    MeReader是一个离线、隐私保护且进度感知的电子书阅读AI助手，通过混合向量相似度、BM25与摘要嵌入的检索架构，将内容检索与生成严格限制在读者当前阅读进度内以避免剧透，并证明2b-4b的中等规模本地模型可在可接受延迟下达到与更大模型相当的理解能力。

    

    本文介绍了MeReader，这是一个隐私保护、离线且进度感知的AI助手，旨在作为消费级硬件上叙事阅读的上下文伴侣。与传统的基于大语言模型（LLM）的系统不同——后者以一次性处理全部文本的方式在线处理文档/书籍，从而存在叙事剧透的风险——MeReader将检索和生成都限制在读者在文本中的具体阅读位置。该系统集成了混合检索架构，结合向量相似度、BM25和摘要嵌入，以确保对读者当前上下文的高保真度。我们使用定量理解指标和基于LLM的定性独立评审，在经典英文书籍上对该系统进行了评估。结果确定了一个质量-延迟边界，表明中等规模的本地模型（2b-4b）可以在保持可接受（延迟）的同时，达到与更大模型（5b+）相当的理解能力。

    arXiv:2611.18513v1 Announce Type: cross  Abstract: This paper introduces MeReader, a privacy-preserving, offline and progress-aware AI assistant designed to function as a contextual companion for narrative reading on consumer hardware. Unlike conventional Large Language Model (LLM)-based systems that process documents/books online in an all-texts-at-once type of way, thereby risking narrative spoilers, MeReader confines both retrieval and generation to the reader's distinct position in the text. The system integrates a hybrid retrieval architecture, combining vector similarity, BM25, and summary embeddings, to ensure high fidelity to the reader's current context. We evaluate the system across classical English books using both quantitative comprehension metrics and qualitative LLM-based independent judging. The results identify a quality-latency frontier, showing that mid-sized local models (2b-4b) can achieve comprehension parity with larger counterparts (5b+) while maintaining accept
    
[^5]: 基于智能体的通用数据元素归纳与肉瘤病例报告摘要信息抽取

    Agentic induction of common data elements and information extraction from sarcoma case report abstracts

    [https://arxiv.org/abs/2611.18506](https://arxiv.org/abs/2611.18506)

    该研究证明大语言模型智能体能够从2万余篇肉瘤病例报告摘要中自动归纳出通用数据元素模式，并以94.9%的抽取一致率实现了大规模、带证据关联的临床信息抽取。

    

    通用数据元素（CDE）支持临床数据的可比性，但通常依赖人工定义。我们研究了大语言模型智能体能否从肉瘤病例报告摘要中推导出CDE模式并进行大规模应用。一个智能体从24,588篇PubMed摘要中提出了72,909个候选CDE；经智能体式整合后得到1,570个CDE。我们保留了其中获得至少50篇摘要支持的53个CDE，并在整个语料库中开展了带证据关联的信息抽取。在79.2%的摘要中，所有保留的CDE均处理成功，其中包括标注为“未提及”的响应。两名人工评审对保留CDE的相关性认可率为83.0%，清晰性认可率为67.9%，正确性认可率为56.6%。在100篇抽样摘要中，在1,116对非空答案中与LLM抽取值完全一致的比例为94.9%（摘要层面bootstrap 95%置信区间为93.0–96.4%）。错误主要集中在解剖学部位和时间相关字段。这些发现支持智能体辅助的模式归纳（原文摘要在此处截断）。

    arXiv:2611.18506v1 Announce Type: new  Abstract: Common data elements (CDEs) support clinical data comparability but are usually defined manually. We investigated whether large language model agents could derive a CDE schema from sarcoma case report abstracts and apply it at scale. An agent proposed 72,909 candidate CDEs from 24,588 PubMed abstracts; agentic consolidation yielded 1,570 CDEs. We retained 53 supported by at least 50 abstracts and attempted evidence-linked extraction across the corpus. Processing succeeded for all retained CDEs in 79.2% of abstracts, including responses indicating non-mention. Two human reviewers approved 83.0% of retained CDEs for relevance, 67.9% for clarity, and 56.6% for correctness. In 100 sampled abstracts, exact agreement with the value extracted by the LLM was 94.9% among 1,116 pairs with a non-empty answer (abstract-level bootstrap 95% CI, 93.0-96.4%). Errors concentrated in anatomical and temporal fields. These findings support agent-assisted sc
    
[^6]: 基于语言模型先验的交互式程序归纳

    Interactive Program Induction with Language Model Priors

    [https://arxiv.org/abs/2611.18485](https://arxiv.org/abs/2611.18485)

    本文提出以语言模型作为程序先验的贝叶斯交互式程序归纳方法，通过贪心选择最具信息量的输入向预言机查询，逐步将后验概率集中于目标程序，以化解从有限示例归纳程序时固有的歧义性问题。

    

    程序归纳是指学习者从少量给定的输入-输出示例中恢复出程序的过程。这一范式通过让用户直接演示他们希望应用于未见数据上的基于规则的行为，而无需亲自实现该行为，从而降低了编程的门槛。然而，从有限的输入-输出示例中恢复程序在本质上存在歧义性，因为一个有限的输入-输出示例集在理论上与无穷多个程序相容，而这些程序在未见输入上的输出可能各不相同。我们从贝叶斯的视角来审视这一问题：在给定程序先验分布 p⁰ 的情况下，识别目标程序意味着通过观察更多的输入-输出示例，将后验概率质量逐步集中到该程序上。在交互式程序归纳的设定中，学习者会主动选择信息量最大的输入，并向预言机（例如用户）查询其对应输出。我们推导了一种贪心的查询选择算法……

    arXiv:2611.18485v1 Announce Type: cross  Abstract: Program induction refers to the process in which a learner aims to recover a program from a few given input-output examples. This paradigm broadens programming access by letting users demonstrate the rule-based behavior they want to apply to unseen data, instead of having to implement it themselves. However, recovering a program from finite input-output examples is fundamentally ambiguous, as a finite set of input-output examples is theoretically consistent with infinitely many programs whose output diverges on unseen inputs. We view this problem from a Bayesian perspective, where, given a prior $p^0$ over programs, identifying the target program means concentrating posterior probability mass on it by observing additional input-output examples. In the interactive program induction setting, the learner chooses maximally informative inputs and queries an oracle, such as the user, for their outputs. We derive a greedy query-selection algo
    
[^7]: 验证器的判断能否经受住抽象化？面向LLM智能体评估的验证迁移测量

    Do Verifier Judgments Survive Abstraction? Measuring Verification Transport for LLM Agent Evaluation

    [https://arxiv.org/abs/2611.18471](https://arxiv.org/abs/2611.18471)

    本文提出“验证迁移”（Verification Transport）这一可靠性测试方法，通过在保持任务、轨迹、结果等条件不变的情况下构建原生、抽象和恢复三种匹配视图，测量LLM验证器的判断在经历工具组合、委托边界、历史压缩等抽象化后是否仍能保持稳定可靠。

    

    arXiv:2611.18471v1 公告类型：交叉。摘要：基于LLM的验证器被越来越多地用于评估智能体的执行过程，但其可靠性通常只在轨迹的单一表示形式下进行测量。在部署的智能体系统中，同一次执行可能通过组合工具、委托边界或压缩的历史记录等方式暴露出来，这改变了验证器可直接观察到的执行证据。因此，正确的验证器判断是否能在这些抽象边界之间保持一致，目前仍不清楚。我们提出了“验证迁移”（Verification Transport），这是一种可靠性测试，用于测量在抽象引起的执行可观察性变化下判断的稳定性。我们在保持任务、实际执行轨迹、结果、真值、评估标准和验证器全部固定不变的前提下，构建了同一次冻结执行的匹配的原生视图、抽象视图和恢复视图。我们在四个智能体评估基准、七种验证器骨干、四种验证协议和三种抽象操作（实验在摘要中被截断）上进行了测试。

    arXiv:2611.18471v1 Announce Type: cross  Abstract: LLM-based verifiers are increasingly used to evaluate agent executions, but their reliability is typically measured under a single representation of the trajectory. In deployed agent systems, the same execution may be exposed through composed tools, delegation boundaries, or compacted histories, changing what execution evidence is directly observable to the verifier. It therefore remains unclear whether correct verifier judgments persist across such abstraction boundaries. We introduce Verification Transport, a reliability test that measures judgment stability under abstraction-induced changes in execution observability. We construct matched native, abstracted, and restored views of the same frozen executions while holding the task, realized trajectory, outcome, ground truth, evaluation criterion, and verifier fixed. Across four agent-evaluation benchmarks, seven verifier backbones, four verification protocols, and three abstraction bo
    
[^8]: LitArc-Bench：将文本特异的读者反应动态从叙事位置与词汇情感中解耦

    LitArc-Bench: Disentangling Text-Specific Reader-Response Dynamics from Position and Lexical Affect

    [https://arxiv.org/abs/2611.18412](https://arxiv.org/abs/2611.18412)

    该论文提出LitArc-Bench基准（39段中文文学节选、9,020个VAD评分），通过位置先验与词汇情感对照实验，首次证明模型与人类情感轨迹的匹配中存在超出叙事位置和情感词汇的文本特异读者反应动态成分。

    

    当模型预测能够跟踪人类情感轨迹的逐步展开时，这种一致性可能反映的是叙事位置、情感词汇，或文本特异的读者反应动态。我们提出LitArc-Bench来区分这几种解释：该基准由39段中文文学节选构成，划分为七个检查点，包含来自两个课堂群体81名成人的430份问卷（共9,020个VAD评分）。我们将三种“前瞻盲”模型运行与留一文本法位置先验、词汇情感基线以及人际参照进行比较。结果表明，仅凭位置因素对唤醒度（arousal）的相关系数即达r = .470，说明单纯的原始对齐不足以说明问题。然而，相对于位置-词典联合控制基线，全部六项效价/唤醒度（V/A）转换优势的文本自助法置信区间均高于零。交叉拟合模型在全部九个模型-维度组合中，对正确人类文本的匹配均显著优于38个错配文本（p < .0001）；而文本不变的阴性对照仅返回随机水平（p = 1）。在人类情感转换幅度相对位置基线至少0.5分时……（原文摘要在此处截断）

    arXiv:2611.18412v1 Announce Type: new  Abstract: When model predictions track an unfolding human affect trajectory, alignment may reflect narrative position, affective words, or text-specific reader-response dynamics. We introduce LitArc-Bench to separate these explanations: 39 Chinese-language literary excerpts segmented into seven checkpoints, with 430 questionnaires from 81 adults in two classroom groups (9,020 VAD ratings). We compare three future-blind model runs with leave-one-text-out position priors, lexical affect, and human-human references. Position alone reaches r = .470 for arousal, making raw alignment insufficient. Yet all six V/A transition advantages over a joint position-lexicon control have text-bootstrap intervals above zero. Cross-fitted models match the correct human text better than 38 mismatched texts in all nine model-dimension cells (p < .0001); a text-invariant negative control returns chance (p = 1). On human transitions at least 0.5 points against the posit
    
[^9]: Gamer：面向长时程实时游戏智能体的通用视觉-语言-动作基础模型

    Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent

    [https://arxiv.org/abs/2611.18397](https://arxiv.org/abs/2611.18397)

    提出Gamer，一个端到端的通用视觉-语言-动作基础模型，通过海量键鼠交互轨迹预训练、稀疏思考指令微调和稳定的多轮强化学习，实现了在复杂2D/3D交互环境中执行长时程任务的实时游戏智能体。

    

    我们提出了Gamer，一个专为复杂交互环境中长时程任务构建的通用智能体框架。其端到端架构结合了感知、记忆管理和多步推理。统一的训练与部署流程支持与常规2D图形界面以及复杂3D模拟世界的交互。我们首先在从数万小时设备级键盘和鼠标交互中构建的数据集上进行继续预训练，这些轨迹涵盖模拟游戏和具身环境，使模型能够学习跨物理与虚拟领域的环境动态。随后的稀疏思考指令微调在保留复杂决策所需推理能力的同时，大幅减少了冗余计算。我们采用一种稳定的多轮强化学习方法来训练策略以实现长时程交互，该方法解决了有限上下文窗口、累积（误差/奖励）等问题（摘要在此处被截断）。

    arXiv:2611.18397v1 Announce Type: cross  Abstract: We introduce Gamer, a generalist agent framework built for long-horizon tasks in complex interactive environments. Its end-to-end architecture combines perception, memory management, and multi-step reasoning. A shared training and deployment pipeline supports interaction with both conventional 2D graphical interfaces and complex 3D simulated worlds. We first continue pre-training on a dataset constructed from tens of thousands of hours of device-level keyboard and mouse interaction. These trajectories span simulated games and embodied environments, allowing the model to learn environmental dynamics across both physical and virtual domains. Sparse-thinking instruction tuning then substantially reduces redundant computation while preserving the reasoning needed for complex decisions. We train the policy for long-horizon interaction with a stable multi-turn reinforcement learning method that addresses finite context windows, accumulated r
    
[^10]: 通过人工评估Google自动补全功能来评估人类对性别偏见AI生成内容的敏感度

    Evaluating Human Sensitivity to Gender Biased AI-Generated Content through Human Assessment of Google's Autocomplete Function

    [https://arxiv.org/abs/2611.18371](https://arxiv.org/abs/2611.18371)

    研究发现AI来源标注不会影响人们对性别相关陈述的反应时间与认同判断，表明人类评估AI生成内容时可能不如评估人类创作内容那样审慎。

    

    大型语言模型（LLM）日益普及，但其对人类用户的影响仍未得到充分探索。我们测试了AI来源标注是否会影响人们对性别相关陈述的看法，而性别是一个偏见普遍存在的领域。通过Prolific招募的西班牙语参与者评估了源自Google自动补全功能的偏见陈述，以及人类撰写的无偏见性别相关事实。参与者要么被告知部分材料是AI生成的，要么未获得关于其来源的任何信息。结果表明，AI来源标注并未影响参与者的反应时间或认同判断。这一结果提示，AI生成的内容并不会自动引发可信度方面的担忧，与人类创作的内容相比，人们可能以更少的审视来评估AI生成的内容。

    arXiv:2611.18371v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly present, but their effects on human users remain underexplored. We tested whether AI-attributed information influences views regarding gender-related statements, a domain where biases are pervasive. Spanish-speaking participants recruited through Prolific evaluated biased statements derived from Google's autocomplete function and human-authored, non-biased gender-related facts. Participants were either informed that some of the material was AI-generated or received no information about its origin. AI attribution did not affect reaction times or agreement judgments. This result suggests that AI-generated content does not automatically raise credibility concerns and may be evaluated with less scrutiny compared to human-produced content.
    
[^11]: 基于大语言模型的f散度自我博弈表格异常检测

    f-Divergence Self-Play for Tabular Anomaly Detection via Large Language Models

    [https://arxiv.org/abs/2611.18370](https://arxiv.org/abs/2611.18370)

    提出DiSPaT，一个基于f散度最小化的自我博弈微调框架，通过交替优化缩小模型生成样本与真实正常数据之间的分布差距，从而增强大语言模型对表格数据正常模式的理解，提升表格异常检测性能。

    

    表格数据中的异常检测由于异构特征类型——混合了数值型、类别型和文本型属性——而面临重大挑战，这使得学习有意义的“正常性”表示变得困难。近期的研究通过将表格行序列化为文本序列，将大语言模型（LLM）应用于这一问题，然而这些方法依赖一次性有监督微调，只能提供有限的信号来收紧模型对正常性的描述。我们提出了DiSPaT，这是一个自我博弈微调框架，用于强化模型对正常数据的理解。基于f散度最小化的理论基础，我们推导出一个紧密的近似，将我们的训练目标与缩小真实正常数据与模型生成样本之间的分布差距联系起来。DiSPaT通过交替优化进行运作：在每次迭代中，当前策略生成合成样本作为伪异常……

    arXiv:2611.18370v1 Announce Type: cross  Abstract: Anomaly detection in tabular data poses significant challenges due to heterogeneous feature types-mixing numerical, categorical, and textual attributes, which complicate learning meaningful representations of normality. Recent work has applied large language models (LLMs) to this problem by serializing table rows as text sequences, yet these approaches rely on one-shot supervised fine-tuning that offers limited signal to tighten the model's description of normality. We propose DiSPaT, a self-play fine-tuning framework that strengthens the model's understanding of normal data. Building on the theoretical foundation of f-divergence minimization, we derive a tight approximation connecting our training objective to reducing the distributional gap between real normal data and model-generated samples. DiSPaT operates through an alternating optimization: at each iteration, the current policy generates synthetic samples that serve as pseudo-an
    
[^12]: 展望智能体机器人：机器人系统的新时代

    Envisioning Agentic Robotics: A New Era of Robotic Systems

    [https://arxiv.org/abs/2611.18359](https://arxiv.org/abs/2611.18359)

    本文提出“智能体机器人”这一新范式，主张由基础模型与具身框架构成的“具身系统2”是智能体机器人的决策核心，并呼吁学界建立统一定义、评估框架与研究路线图。

    

    GPT-6 Astra 的发布标志着机器人学新范式——智能体机器人的诞生。短短几周内，配备智能体框架的前沿模型在许多具身基准测试中达到甚至超越了任务专用策略的表现。与为每一类任务训练专用策略不同，智能体机器人系统将基础模型应用于开放世界的具身任务。它以闭环方式运行：观察环境、理解用户意图、对任务进行推理、调用技能与工具、验证结果并调整策略。尽管取得了这些进展，学界尚未在统一定义、可靠的评估框架或研究路线图上达成共识。我们认为，由基础模型与具身框架共同实现的“具身系统 2”构成了智能体机器人的决策核心，其核心挑战在于将通用推理转化为在……（原文摘要在此处截断）

    arXiv:2611.18359v1 Announce Type: cross  Abstract: The release of GPT-6 Astra marks the emergence of a new paradigm in robotics: agentic robotics. Within weeks, frontier models with agentic harnesses matched or exceeded task-specific policies on many embodied benchmarks. Instead of training a dedicated policy for each family of tasks, an agentic robotic system applies a foundation model to open-world embodied tasks. It operates in a closed loop: observing the environment, interpreting user intent, reasoning about the task, invoking skills and tools, verifying outcomes, and adapting its strategy. Despite these advances, the community has yet to converge on a shared definition, a reliable evaluation framework, or a research roadmap. We argue that Embodied System 2, realized by a foundation model together with an embodied harness, forms the decision-making core of agentic robotics. Its central challenge is to translate general-purpose reasoning into effective and verifiable actions under 
    
[^13]: Marimba：一个面向语音的“系统一”模型

    Marimba: A System One Model for Speech

    [https://arxiv.org/abs/2611.18330](https://arxiv.org/abs/2611.18330)

    Marimba基于冻结的Qwen3-ASR 0.6B特征构建了一个轻量级“系统一”语音决策接口，通过零样本评分、条件化读出头和中间层探针，在词项检测、轮次结束判断、语言识别、个人信息检测和说话人变化检测等多项语音决策任务上取得了优异且良好校准的性能。

    

    语音应用需要在音频窗口内做出关于词项是否存在、轮次是否结束以及语言种类的决策。我们提出了Marimba，一个基于冻结的Qwen3-ASR 0.6B特征构建的类型化语音决策接口。词汇类问题使用ASR转录文本的零样本评分；非词汇类问题使用问题条件化读出头和独立的第18层轮次结束读出；语言识别则使用ASR模型自身的语言评分。在私有会话测试集上，解码器AUROC为0.80-0.92；在公开朗读语音上为0.92-0.97。Speech Commands四分类准确率为0.93；二分类词汇任务的ECE不超过0.08。个人信息检测通过将贪心转录传递给私有的、经过整理的PII流水线，在合成的巴西葡萄牙语语音上达到条目级F1 0.904（在参考转录上为1.000）。第12层说话人读出在AMI和VoxConverse数据集上以AUROC 0.966-0.980检测说话人变化，高于ECAPA基线，并且

    arXiv:2611.18330v1 Announce Type: cross  Abstract: Speech applications require decisions about term presence, turn completion, and language within an audio window. We present Marimba, an interface for typed speech decisions built on frozen Qwen3-ASR 0.6B features. Lexical questions use zero-shot scoring of ASR transcripts; non-lexical questions use a question-conditioned head and a separate layer-18 turn-end readout. Language identification uses the ASR model's own language scores. On private conversational test sets, decoder AUROC is 0.80-0.92; on public read speech it is 0.92-0.97. Speech Commands four-way accuracy is 0.93; binary lexical ECE is at most 0.08. Personal-information detection, which passes the greedy transcript to a private, curated PII pipeline, reaches item-level F1 0.904 on synthetic Brazilian Portuguese speech (1.000 on reference transcripts). Layer-12 speaker readouts detect speaker change with AUROC 0.966-0.980 on AMI and VoxConverse, above an ECAPA baseline, and 
    
[^14]: 用少量标注学习拉丁文学中文本复用的词级解释

    Learning Word-Level Explanations of Text Reuse in Latin Literature with Little Annotation

    [https://arxiv.org/abs/2611.18319](https://arxiv.org/abs/2611.18319)

    该论文提出一种仅需少量标注的方法：先利用合成文本对训练模型，再通过主动学习优化，最终用指针网络对拉丁文学中的文本复用进行词级对齐与改写操作标注，从而解释后世作者如何改写原文。

    

    检测互文性指涉只是第一步；语文学家还需要知道后世作者是如何改写原始文献的。我们对复用文本对进行逐词解释：复用段落中的每个词首先与其源文本中的词进行对齐，然后用改变它的操作来标注，例如复制、屈折变化或替换。这类词级操作的标注成本高昂，目前针对拉丁文学文本复用、在此层级上进行标注的语料库仅有少数几个。我们提出了一种只需少量标注即可完成的方法：通过对拉丁文本应用已知操作构建合成文本对来训练第一个模型，再由主动学习挑选出需要标注者修正的文本对。在多种架构中，将链接与标注合并为一个步骤的模型表现最佳：即一个指针网络，它对复用段落中的每个词，在同一步骤中选出对应的源词并给出所执行的操作。我们证明，借助主动学习，仅需200个文本对即可……

    arXiv:2611.18319v1 Announce Type: new  Abstract: Detecting an intertextual reference is only the first step; philologists also need to know how a later author reworked the source. We explain a reuse pair word by word: each word of the reusing passage is first aligned to its source word, then labeled with the operation that transformed it, such as a copy, a change of inflection, or a substitution. Such word-level operations are expensive to annotate, and only a few corpora annotated at this level exist for text reuse in Latin literature. We describe a procedure that needs only a limited number of annotations: synthetic pairs, built by applying known operations to Latin text, train a first model, and active learning chooses the pairs an annotator corrects. Among several architectures, a model that links and labels together performs best: a pointer network that, for every reuse word, chooses a source word and names the operation in the same step. We show that with active learning, 200 pai
    
[^15]: 当评估在临床AI中变成干预

    When evaluation becomes intervention in clinical AI

    [https://arxiv.org/abs/2611.18297](https://arxiv.org/abs/2611.18297)

    临床AI中的AI评判者在被用于选择提示词或训练模型时会本身成为干预的一部分，并可能偏向其忽视的错误，因此评判者报告必须包含特定内容，且每项由评判者驱动的变更都需要独立的临床测试。

    

    AI评判者（AI judges）是一种依据评分标准对另一个系统的输出进行打分的模型，它将临床审查的规模扩展到了临床AI书写内容的规模。当团队使用评判者的评估结果来选择提示词或训练模型时，评判者本身就成为了干预的一部分。这种选择过程可能会偏向评判者所忽视的错误。我们描述了评判者的报告必须包含哪些内容，以及每一项由评判者驱动的变更所需的独立临床测试。

    arXiv:2611.18297v1 Announce Type: new  Abstract: AI judges, models that grade another system's output against a rubric, extend clinical review to the scale at which clinical AI writes. When teams use judge assessments to select a prompt or train a model, the judge becomes part of the intervention. Selection can then favor errors the judge overlooks. We describe what a judge's report must contain and the independent clinical test each judge-driven change needs.
    
[^16]: 清洁票据，错误订单：用浏览器标签页中的语言模型捕捉静默错误

    Clean Tickets, Wrong Orders: Catching Silent Errors with a Language Model in Your Browser Tab

    [https://arxiv.org/abs/2611.18284](https://arxiv.org/abs/2611.18284)

    本文展示了一个完全在浏览器本地运行的小型语言模型演示系统，利用移植到TypeScript的PIZZA匹配器实时捕捉“静默错误”（即模型生成格式正确却无人真正下过的订单），其表现与服务器端评估几乎完全一致。

    

    将输入订单转换为结构化数据的小型语言模型可能以两种方式失败：一是输出损坏的内容，任何系统都会将其拒绝；二是输出一个干净、格式良好却无人真正下过的订单。我们的配套工作将第二种情况称为“静默错误”，而这才是危险的一种。我们展示了一个网页演示：一个经过微调的0.8B参数模型通过编译为WebAssembly的llama.cpp完全在访问者的浏览器标签页中运行，任何订单文本都不会离开用户设备。页面在访问者输入时实时解析订单，使用移植到TypeScript的PIZZA基准测试自带匹配器对每张票据进行评分，用"？"拒绝任何不在菜单上的内容以让模型直面开放网络，并允许访问者寻找静默错误或尝试制造一个。在浏览器中对全部1,357个PIZZA测试订单运行时，该部署模型对99.0%的订单输出与服务器端评估完全一致，并达到相同的精确匹配率（0.882）；我们还报告了其延迟表现。

    arXiv:2611.18284v1 Announce Type: cross  Abstract: A small language model that turns a typed order into structure can fail in two ways. It can print something broken, which any system rejects. Or it can print a clean, well formed order that nobody placed. Our companion work calls the second kind a silent error, and it is the dangerous one. We present a web demo in which a fine-tuned 0.8B parameter model runs entirely inside the visitor's browser tab, through llama.cpp compiled to WebAssembly. No order text leaves the device. The page parses orders as the visitor types, scores every ticket with the PIZZA benchmark's own matcher ported to TypeScript, refuses anything off the menu with "?" so the model can face the open web, and lets visitors hunt for silent errors or try to cause one. Run in the browser on all 1,357 PIZZA test orders, the deployed model writes the same output as the server evaluation for 99.0% of them and reaches the same exact match (0.882); we also report its latency o
    
[^17]: GT-FENG：语言模型事实评估与规范性依据的通用理论

    GT-FENG: General Theory of Factual Evaluation and Normative Grounding in Language Models *Author(s)

    [https://arxiv.org/abs/2611.18279](https://arxiv.org/abs/2611.18279)

    GT-FENG是首个统一上下文/时间正确性与基于规则的错误归因的通用框架，通过对照适用规则检验语言模型答案并追溯错误来源，在Medicaid收入上限问答中使仅匹配其他州数值的错误答案减少68-78%。

    

    语言模型的回答是否正确往往取决于上下文：同一数值在某一司法辖区、日期或法律定义下可能是正确的，而在另一种条件下则是错误的。我们提出了GT-FENG（事实评估与规范性依据的通用理论），这是首个将上下文/时间正确性与基于规则的错误归因统一起来的通用框架，适用于法律、政策及其他受规则约束的领域。GT-FENG依据适用规则对答案进行检查，并识别出与错误答案相匹配的已记录规则。研究表明，当多条规则给出相同数值时，仅凭答案本身无法判断模型使用了哪条规则。在来自两个模型的Medicaid收入上限回答中，记录所询问州自身的数据使得仅匹配其他州数值的错误答案数量减少了68-78%，且此后超过一半的稳定错误答案会匹配两个或更多的已记录来源。GT-FENG为语言模型评估提供了基础。

    arXiv:2611.18279v1 Announce Type: new  Abstract: Whether a language model's answer is correct often depends on context: the same value can be right under one jurisdiction, date, or legal definition and wrong under another. We introduce GT-FENG (General Theory of Factual Evaluation and Normative Grounding), the first general framework to unify contextual/temporal correctness and rule-based error attribution across law, policy, and other rule-governed domains. GT-FENG checks answers against the applicable rules and identifies documented rules that match incorrect answers. It shows that when multiple rules give the same value, the answer alone cannot tell which rule the model used. In Medicaid income-limit answers from two models, documenting the asked state's own records reduces the number of wrong answers that match only other states' values by 68-78%, and more than half of stable wrong answers then match two or more documented sources. GT-FENG provides a foundation for evaluating langu
    
[^18]: MamaReason：临床语言模型中的未完成推理与答案恢复

    MamaReason: Unfinished Reasoning and Answer Recovery in Clinical Language Models

    [https://arxiv.org/abs/2611.18264](https://arxiv.org/abs/2611.18264)

    该研究发现临床语言模型通过答案续写恢复的诊断大多已显式存在于未完成的推理轨迹中，仅直接读取这些诊断承诺（无需额外生成）即可挽救大部分续写所恢复的失败案例。

    

    答案续写常常返回一个在未完成的临床推理中早已陈述过的诊断。我们在 MamaReason（由稠密的 Gemma 4 31B 适配而来）上，针对 MamaBench 的 217 组配对母婴健康病例演示研究了这一行为。一个显式承诺提取器在 94 例强化学习（RL）首次通过失败的案例中识别出 75 例已有诊断，在 93 例监督微调（SFT）首次通过失败的案例中识别出 75 例已有诊断。仅使用诊断字符串，续写答案与该诊断匹配的情况为 RL 中的 67/75 例、SFT 中的 58/75 例。在归档的正确性标签下，不进行任何额外生成而直接读取这些承诺，即可恢复续写在 40 个 RL 配对中挽救的 30 个。复发控制实验区分了推理轨迹中的常规提及与重复的答案选择。一个离线的二次承诺触发器量化了更早读取与配对准确率之间的权衡。由此得到的分解方法可将推理轨迹中已存在的诊断内容、对已接受答案的补全以及配对诊断一致性区分开来。

    arXiv:2611.18264v1 Announce Type: new  Abstract: Answer continuation often returns a diagnosis already stated in unfinished clinical reasoning. We study this behavior in MamaReason, adapted from dense Gemma 4 31B, on MamaBench's 217 paired maternal and child-health presentations. An explicit-commitment extractor identifies a diagnosis in 75 of 94 failed RL first passes and 75 of 93 failed SFT first passes. The continuation matches that diagnosis in 67/75 RL cases and 58/75 SFT cases, using diagnosis strings alone. Reading these commitments without additional generation recovers 30 of the 40 RL pairs restored by continuation under the archived correctness labels. Recurrence controls distinguish routine mentions from repeated answer selection. An offline second-commitment trigger measures the tradeoff between earlier readout and paired accuracy. The resulting decomposition separates diagnostic content already present in a trace, completion of an accepted answer, and paired diagnostic cor
    
[^19]: NER-D：一个面向德语命名实体识别的跨领域基准

    NER-D: A cross-domain benchmark for German NER

    [https://arxiv.org/abs/2611.18257](https://arxiv.org/abs/2611.18257)

    该论文提出了一个覆盖 9 个领域、包含 13 个数据集的德语 NER 跨领域基准 NER-D，并发现经 LoRA 微调的小型开源模型能大幅超越专有 LLM 的性能。

    

    我们提出了 NER-D，一个面向德语命名实体识别（NER）的跨领域基准，包含横跨 9 个领域的 13 个数据集，领域涵盖通用/政治、维基百科、法律、医疗、历史、金融、农业和考古学。我们通过普通提示方法考察了专有 LLM 与最先进的小型开源模型的性能，并进一步研究了通过上下文学习和 LoRA 微调所能获得的性能提升。我们的结果表明，在不进行任何优化的情况下，专有模型（GPT-5.4 mini）在 13 个数据集中的 6 个上优于较小的开源模型，但 LoRA 微调能够使较小的模型大幅超越专有模型基线。我们进一步分析了领域以及每个任务中使用的标签集对不同评估设置下性能的影响。该基准将作为资源公开发布，用于未来在通用及专门领域开发德语 NER 方法。

    arXiv:2611.18257v1 Announce Type: new  Abstract: We present NER-D, a cross-domain benchmark for German NER comprising 13 datasets spanning 9 domains, including General/Politics, Wikipedia, Legal, Medical, Historical, Financial, Agriculture and Archaeology. We examine the performance of both proprietary LLMs and state-of-the-art small open-source models using normal prompting and further examine the performance gains achievable via in-context learning and LoRA fine-tuning. Our results show that without any sort of optimization a proprietary model (GPT-5.4 mini) outperforms the smaller open-source models on 6 out of 13 datasets, but LoRA fine-tuning enables the smaller models to substantially outperform the proprietary baseline. We further analyze the influence of the domain and the tagset used in each task on the performance achieved in the different evaluation settings. Our benchmark will be available as a resource for the development of future German NER methods on both general and sp
    
[^20]: 从自然语言生成BPMN的多层AI流水线

    A Multi-Layered AI Pipeline for BPMN Generation from Natural Language

    [https://arxiv.org/abs/2611.18157](https://arxiv.org/abs/2611.18157)

    该论文提出了一种集成基于Transformer的架构、大语言模型和本体填充的多层AI流水线，通过微调RoBERTa模型进行实体提取、利用LLM推理捕获非相邻控制流关系，并采用动态少样本策略提升有限数据下的泛化能力，从而实现从自然语言描述自动生成BPMN流程图。

    

    将非结构化的自然语言流程描述自动转换为业务流程模型与标记法图仍然是一项具有挑战性的任务，其原因在于语言歧义性以及检测非相邻控制流关系的困难。本文提出了一种通过多层流水线从自然语言描述生成BPMN模型的解决方案，该流水线集成了基于Transformer的架构、大语言模型和本体填充。所提出的方法以自然语言描述为起点，将文本处理、控制流关系提取和结构验证分离到专门的组件中。实体提取通过经过微调的基于RoBERTa的Transformer模型完成，而控制流引擎通过基于大语言模型的推理来捕获非相邻关系。为了在有限可用数据下提高泛化能力，该方法采用了动态少样本策略。

    arXiv:2611.18157v1 Announce Type: new  Abstract: The automated transformation of unstructured natural language process descriptions into Business Process Model and Notation (BPMN) diagrams remains a challenging task, due to linguistic ambiguity and the difficulty to detect non-adjacent control-flow relationships. This paper proposes a solution for generating BPMN models from natural language descriptions through a multi-layered pipeline that integrates transformer-based architectures, Large Language Models and ontology population. The proposed method starts with natural language descriptions and separates textual processing, control-flow relation extraction, and structural validation into dedicated components. Entity extraction is performed using a fine-tuned RoBERTa-based transformer model while a control-flow engine captures non-adjacent relationships through LLM-based inference. To improve generalization under limited available data, the approach uses a dynamic few-shot strategy wit
    
[^21]: 将语言模型折叠进自身

    Folding Language Models into Themselves

    [https://arxiv.org/abs/2611.18113](https://arxiv.org/abs/2611.18113)

    提出Concertina方法，将N个相邻教师层折叠为执行N次的共享块并配合低秩修正和逐token门控，直接从教师权重构建紧凑学生模型，在22B token训练量下保留教师约90%的性能。

    

    知识蒸馏通常从头训练一个紧凑的学生模型，即使教师的权重已经编码了学生所需学习的大部分内容，这仍然需要大量数据。在这项工作中，我们探索直接从教师的权重构建学生模型。我们提出了Concertina，它将每组N个相邻的教师层折叠为一个共享块，该共享块执行N次。每个循环步骤携带一个从其替换层初始化的低秩修正，并通过一个逐token的门控机制混合这些修正，使每个token能够从单个共享块中恢复原始各层的行为。我们进一步设计了一套蒸馏数据配方，平衡通用、数学、学术和针对性的多选题数据，以同时恢复教师的各项能力，而不是以牺牲某一能力为代价换取另一能力。在Qwen3-4B上，一个在22B token上蒸馏的2.27B Concertina模型在常识任务上保留了教师约90%的性能。

    arXiv:2611.18113v1 Announce Type: new  Abstract: Knowledge distillation usually trains a compact student from scratch, which needs a large amount of data even though the teacher's weights already encode most of what the student has to learn. In this work, we explore building the student directly from the teacher's weights. We propose Concertina, which folds every group of N adjacent teacher layers into one shared block that is executed N times. Each loop step carries a low-rank correction initialized from the layer it replaces, and a per-token gate mixes these corrections, to let every token recover the behavior of the original layers from a single shared block. We further design a distillation data recipe that balances general, mathematical, academic and targeted multiple-choice data, to recover all capabilities of the teacher together rather than one at the expense of another. On Qwen3-4B, a 2.27B Concertina model distilled on 22B tokens retains about 90% of the teacher on commonsens
    
[^22]: 传染性指纹：当大语言模型继承另一个模型的身份时

    Contagious Fingerprints: When LLMs Inherit Another Model's Identity

    [https://arxiv.org/abs/2611.18104](https://arxiv.org/abs/2611.18104)

    本文发现大语言模型的指纹具有“传染性”——接续其他模型对话的模型会继承前者的指纹，导致API中转中的模型偷换检测率大幅下降（如从0.97降至0.72），其根源在于模型对历史对话风格的模仿。

    

    分类器用来识别文本背后大语言模型的“指纹”通常被视为模型的固定属性。我们证明这种指纹是会传染的：一个接续由另一个模型开始的对话的模型，会继承那个模型的指纹。这破坏了对API中转服务的审计——此类服务可以用客户付费的模型来回应对话的前几轮，然后悄悄将剩余部分转交给更便宜的模型。在这种转手之后，文本分类器对替代模型的检测率从0.97降至0.72（DeepSeek替换Claude Sonnet），在同类厂商降级（Haiku）的情况下从0.87降至0.54，而且在转手数据上重新训练只能修复训练中见过的替代模型。预注册实验将这种传染追溯到对历史对话风格的模仿。对话历史还会使模型偏离其自身的指纹：主动探测不会被伪装，但当基于新鲜对话进行校准时，它们会在长上下文中误报诚实的模型。

    arXiv:2611.18104v1 Announce Type: new  Abstract: The fingerprint by which a classifier recognizes the LLM behind a text is usually treated as a fixed property of the model. We show that it is contagious: a model that continues a conversation begun by another model inherits that model's fingerprint. This undermines audits of API relays, which can serve the first turns of a conversation with the model a customer pays for and silently hand the rest to a cheaper one. After such a handoff, detection of the substitute by a text classifier drops from 0.97 to 0.72 (DeepSeek replacing Claude Sonnet) and from 0.87 to 0.54 for a same-vendor downgrade (Haiku), and retraining on handoff data repairs only substitutes seen in training. Preregistered experiments trace the contagion to imitation of the history's style. History also moves a model away from its own fingerprint: active probes are not camouflaged, but when calibrated on fresh conversations they flag \emph{honest} models on long contexts at
    
[^23]: VEPO：通过视觉证据引导的偏好优化缓解大型视觉语言模型中的多语言幻觉

    VEPO: Mitigating Multilingual Hallucinations in LVLMs via Visual Evidence-guided Preference Optimization

    [https://arxiv.org/abs/2611.18096](https://arxiv.org/abs/2611.18096)

    提出VEPO训练时方法，利用英语查询定位视觉证据并构建证据充分性不同的偏好对进行偏好优化，无需外部模型即可将可靠的视觉定位能力从英语迁移到其他语言，有效缓解大型视觉语言模型在多语言（尤其是低资源语言）场景下的对象幻觉问题。

    

    大型视觉语言模型（LVLMs）在多模态任务中表现出色，但仍然容易出现对象幻觉——即生成语言上流畅但缺乏视觉依据的答案。这一问题在低资源语言中尤为严重，因为较弱的跨模态对齐使得模型更依赖语言先验而非图像证据。现有的解决方案要么需要昂贵的多语言标注或外部翻译/奖励模型，要么在推理阶段进行干预，从而可能扭曲输出语言。我们提出了VEPO（视觉证据引导的偏好优化），这是一种训练时方法，无需外部模型即可将可靠的视觉定位能力从英语迁移到其他语言。VEPO首先通过中层跨模态注意力机制使用英语查询定位关键视觉证据，然后构建共享相同正确答案但证据充分性不同的偏好对（完整图像 vs. k...（摘要在此处被截断）

    arXiv:2611.18096v1 Announce Type: new  Abstract: Large vision-language models (LVLMs) achieve strong multimodal performance, yet they remain prone to object hallucination---generating answers that are linguistically fluent but visually ungrounded. This issue is especially severe in low-resource languages, where weaker cross-modal alignment causes models to rely more on language priors than on image evidence. Existing remedies either require costly multilingual annotations or external translation/reward models, or intervene at inference time in ways that can distort the output language. We propose VEPO(Visual Evidence-guided Preference Optimization), a training-time method that transfers reliable visual grounding from English to other languages without external models. VEPO first localizes key visual evidence using English queries via mid-layer cross-modal attention, then constructs preference pairs that share the same correct answer but differ in evidence sufficiency (full image vs.\ k
    
[^24]: 深层潜变量，长记忆：潜变量持久性解释Transformer激活中的时间相关性

    Deep Latents, Long Memory: Latent persistence explains temporal correlations in transformer activations

    [https://arxiv.org/abs/2611.18089](https://arxiv.org/abs/2611.18089)

    该论文提出用层激活的自相关函数作为抽象的可操作度量，发现潜变量的持久编码会在自相关中产生慢衰减尾部，且这些尾部随Transformer层加深而增长，从而解释了深层模型编码抽象潜变量的机制。

    

    大型语言模型（LLM）的隐藏层被认为编码了数据表示，其抽象程度随深度增加。然而，由于缺乏可操作的抽象概念定义，这一说法难以进行定量检验。在此，我们提出了另一种视角，其基础原理是：序列的相关性反映了生成该序列的过程的结构。我们将某一层的自相关函数定义为其在两个相隔一定滞后的上下文位置上的激活之间的平均点积。沿上下文持续编码的潜变量会在自相关中产生缓慢衰减的尾部，我们发现在合成数据和自然数据中，这些尾部都会随层加深而增长。在具有受控层级潜变量结构的合成数据中，这种增长源于对抽象潜变量的编码，这些潜变量支配着输入的长片段。事实上，潜变量的持久性……

    arXiv:2611.18089v1 Announce Type: cross  Abstract: The hidden layers of Large Language Models (LLMs) are believed to encode representations of the data whose abstraction increases with depth. However, in the absence of an operational notion of abstraction, this statement is hard to test quantitatively. Here, we propose an alternative view, building on the principle that the correlations of a sequence reflect the structure of the process that generated it. We define the autocorrelation function of a layer as the average dot product between its activations at two context positions separated by a lag. Latent variables encoded persistently along the context create slowly decaying tails in the autocorrelation, and we find these tails to grow across layers in both synthetic and natural data. In synthetic data with controlled hierarchical latent structure, this growth stems from the encoding of abstract latent variables, which govern long stretches of the input. Indeed, the persistence of lat
    
[^25]: 智能体即裁判（Agent-as-a-Judge）

    Agent-as-a-Judge

    [https://arxiv.org/abs/2611.18067](https://arxiv.org/abs/2611.18067)

    该论文提出“智能体即裁判”新范式，让评估器能够像智能体一样主动搜集固定上下文之外的缺失证据（如未记录的环境状态和需要外部核验的事实性声明），从而对AI系统做出更可靠的评估裁决。

    

    人工智能已从面向特定任务的系统发展为涵盖感知、生成、推理与自主交互的通用模型，这使得可靠的评估对于模型比较、部署以及训练后反馈变得至关重要。在大规模场景下，现代评估流程日益依赖“大语言模型即裁判”（LLM-as-a-Judge）来生成评分、偏好与批评意见；在固定上下文的设定中，裁判仅依赖所提供的被评估对象和评估材料。然而，一个可靠的裁决可能需要超出所提供材料与上下文之外的证据——这可能是因为关键的环境状态未被记录在静态执行日志中，也可能是因为存在争议的事实性声明需要外部验证。被动的裁判在做出判断之前无法获取这些缺失的证据，尤其是在评估智能体轨迹、可执行产物或领域特定报告时。因此，近期的工作开发了能够主动搜寻缺失证据的主动式评估器……

    arXiv:2611.18067v1 Announce Type: new  Abstract: Artificial intelligence has progressed from task-specific systems toward general models spanning perception, generation, reasoning, and autonomous interaction, making reliable evaluation essential for comparison, deployment, and post-training feedback. At scale, modern evaluation pipelines increasingly rely on LLM-as-a-Judge to produce scores, preferences, and critiques; in its fixed-context setting, the judge relies on the supplied evaluand and evaluation materials. Yet a reliable verdict may require evidence beyond the supplied evaluand and context---either because critical environmental states remain unrecorded in the static execution log, or because controversial factual claims require external verification. Passive judges cannot obtain that missing evidence before deciding, especially when evaluating agent trajectories, executable artifacts, or domain-specific reports. Recent work therefore develops active evaluators that seek missi
    
[^26]: FEND：语音AI外呼通话中的远端检测

    FEND: Far-End Detection in Outbound Voice AI Calls

    [https://arxiv.org/abs/2611.18050](https://arxiv.org/abs/2611.18050)

    提出FEND框架，通过分析远端音频与流式转录文本，对语音AI外呼中的远端类型（真人、语音信箱、IVR、AI）进行细粒度检测，突破了传统二值应答机检测无法区分机器接听类型的局限。

    

    当一个对话式语音AI代理发起外呼电话时，接听方（远端）可能是真人、语音信箱问候语、交互式语音应答（IVR）系统，或者越来越多的情况是代表被叫方应答的另一个AI系统。标准的应答机检测（AMD）仅报告接听者是真人还是机器，这一设计原本是为人工呼叫中心设计的，因为在那种场景下，无论哪种机器接听都同样无关紧要。然而这一比特的信息并不能告诉语音AI代理应该采取什么行动，因为留下语音留言、操作导航IVR系统以及与另一个AI进行简短交流是三种完全不同的行为，二值标签无法区分它们。在我们自己外呼部署中随机抽取的1,000通电话中，真人接听占48.1%，语音信箱系统占44.6%，AI占6.9%，IVR占0.4%。我们提出的FEND框架读取远端的音频及其流式转录文本，并为每个自动化类别（语音信箱、IVR、AI等）分别运行一个检测器……

    arXiv:2611.18050v1 Announce Type: new  Abstract: When a conversational Voice AI agent places an outbound call, the far end may be a live person, a voicemail greeting, an interactive voice response (IVR) system, or increasingly another AI system answering on the callee's behalf. Standard Answering Machine Detection (AMD) reports only whether a human or a machine picked up, a design meant for human call centres, where one machine is as uninteresting as another. That single bit does not tell a Voice AI agent what to do, since leaving a voicemail message, navigating an IVR and holding a short exchange with another AI are three different actions and a two-valued label cannot distinguish them. In 1,000 randomly sampled calls from our own outbound deployment, a live person answered 48.1%, a voicemail system 44.6%, an AI 6.9% and an IVR 0.4%. Our proposed framework, FEND, reads the far end's audio and its streaming transcript and runs one detector for each automated class (voicemail, IVR, AI c
    
[^27]: PerfArena：按代码性能对大语言模型进行排名

    PerfArena: Ranking LLMs by Code Performance

    [https://arxiv.org/abs/2611.18030](https://arxiv.org/abs/2611.18030)

    PERFARENA是一个开源、可复现的评测竞技场，通过测量LLM生成代码在多种语言和竞赛中的延迟、峰值内存与能耗等性能指标来对模型排名，弥补了现有基准仅关注正确性、语言覆盖窄且忽略能耗的不足。

    

    当前业界已依赖大语言模型（LLM）编写大量生产代码。尽管这些代码大部分情况下是正确的，但并非总是如此，而且仅有正确性并不能决定代码是否适合被反复和广泛使用，这还取决于代码在时间、内存和能耗上的开销。然而，大多数代码生成基准只单独评估正确性，而大多数面向性能的基准覆盖的语言较少，且很少测量能耗。为填补这一空白，我们提出了PERFARENA，一个开源、可复现的竞技场，它根据LLM所生成代码的性能在多种语言和竞赛中对模型进行排名。PERFARENA在类Unix主机上执行每个LLM的解答，并报告正确性以及延迟、峰值内存和能耗。在PERFARENA中，正确性判定器、性能判定器、性能分析器和代码生成工作流均是可插拔的，从而能够集成新的评判后端。

    arXiv:2611.18030v1 Announce Type: cross  Abstract: Industry now relies on LLMs to write a large share of production code. Although that code is mostly, but not always, correct, correctness alone does not determine whether it is fit for repeated and extensive use, which depends on what the code costs in time, memory and energy. Most code-generation benchmarks nonetheless evaluate correctness in isolation, and most performance-oriented benchmarks cover few languages and rarely measure energy consumption. To address this gap, we present PERFARENA, an open-source, reproducible arena that ranks LLMs by the performance of their generated code across multiple languages and competitions. PERFARENA executes the solutions of each LLM on a Unix-like host and reports correctness together with latency, peak memory, and energy consumption. Correctness oracles, performance oracles, profilers, and code-generation workflows are pluggable in PERFARENA, enabling the integration of new judging backends an
    
[^28]: 迈向普遍可译性：论生成式人工智能的统计本体论

    Toward Universal Translatability: On the Statistical Ontology of Generative AI

    [https://arxiv.org/abs/2611.17963](https://arxiv.org/abs/2611.17963)

    本文提出翻译是当代生成式AI的起点与规范目标，其“通用翻译机器”理想背后隐含一种可追溯至韦弗1949年机器翻译备忘录的“统计本体论”形而上学承诺——即相信现实具有可通过数据与建模恢复的可计算统计结构。

    

    本文论证，翻译问题既构成了当代生成式人工智能的起点，也是其规范性视界。在当前围绕“世界模型”的讨论中，世界模型的实现被框定为内部一致性、跨模态对齐与物理接地的达成。这些要求汇聚成一种理想：将模型视为一台通用翻译机器，能够在语言、视觉与物理表征空间之间建立无损的对应关系。本文指出，这一理想预设了一种形而上学承诺，作者称之为“统计本体论”：即相信现实具有一种潜在的、可计算的统计结构，可以通过充足的数据与建模加以恢复，而非通过情境化的人类诠释与劳动来塑造和通达。通过将这一承诺追溯至沃伦·韦弗1949年关于机器翻译的备忘录，本文勾勒了翻译如何……（摘要原文在此处截断）

    arXiv:2611.17963v1 Announce Type: new  Abstract: This paper argues that the problem of translation functions both as a point of origin and as a normative horizon for contemporary generative AI. In current discourse around ''world models,'' their attainment is framed as the achievement of internal consistency, cross-modal alignment, and physical grounding. Taken together, these demands converge on an ideal of the model as a universal translation machine capable of establishing lossless correspondences across linguistic, visual, and physical representation spaces. I show that this ideal presupposes a metaphysical commitment that I call a statistical ontology: the belief that reality has an underlying, computable statistical structure that can be recovered through sufficient data and modeling, rather than being shaped and accessed through situated human interpretation and labor. Tracing this commitment back to Warren Weaver's 1949 Memorandum on machine translation, I outline how translati
    
[^29]: WebArena-Pro：一个面向Web智能体的异构、多模态、可复现基准测试

    WebArena-Pro: A Heterogeneous, Multimodal, Reproducible Benchmark for Web Agents

    [https://arxiv.org/abs/2611.17955](https://arxiv.org/abs/2611.17955)

    WebArena-Pro是一个涵盖20个自托管Web应用、300个任务的异构多模态可复现基准测试，首次支持对Web智能体理解和运用音频、视频内容能力的评估。

    

    基于大语言模型和视觉-语言模型的Web智能体正越来越多地被应用于涉及异构应用、多模态内容和有状态工作流的现实浏览器任务中。然而，现有的可复现Web智能体基准测试仅覆盖来自少数软件类别的少量Web应用，且模态仅限于文本和视觉。实时基准测试虽然扩大了站点覆盖范围，但由于页面和数据在运行之间会发生漂移，牺牲了可复现性。此外，现有基准测试未能有效评估智能体能否理解和使用嵌入在Web任务中的音频和视频内容。为弥补这些不足，我们提出了WebArena-Pro，这是一个包含300个任务的基准测试，涵盖六个领域类别中的20个自托管Web应用，跨越不同的界面惯例、工作流和数据模型。在所评估的智能体中，Codex with GPT 6.1 Sol 取得了最佳表现。

    arXiv:2611.17955v1 Announce Type: new  Abstract: Web agents powered by large language and vision-language models are increasingly applied to realistic browser work that spans heterogeneous applications, multimodal content, and stateful workflows. However, existing reproducible web-agent benchmarks cover only a small number of web applications drawn from a few software categories, and restrict modality to text and vision. Live benchmarks broaden site coverage but sacrifice reproducibility, since pages and data drift between runs. Moreover, existing benchmarks do not meaningfully evaluate whether agents can understand and use audio and video content embedded within web tasks. To address these gaps, we introduce WebArena-Pro, a benchmark comprising 300 tasks across 20 self-hosted web applications in six domain categories, spanning distinct interface conventions, workflows, and data models. Across the evaluated agents, the best performance is achieved by both Codex with GPT 6.1 Sol and Cla
    
[^30]: 旁观者眼中：文本与视觉蕴含中的人类标注差异

    In the Eye of the Beholder: Human Label Variation in Textual and Visual Entailment

    [https://arxiv.org/abs/2611.17954](https://arxiv.org/abs/2611.17954)

    该论文提出了VariNLI-VE数据集，通过让同一批标注者对成对的NLI文本实例及其视觉对应版本重新标注（包含标签、置信度和解释，共2,250条标注），系统研究人类标注差异如何在文本与视觉两种蕴含任务之间变化。

    

    人类标注差异正日益被认识到是一种有信息量的信号，它反映了人们在对同一输入进行解读时存在的有意义差异。虽然人类标注差异在自然语言推理（NLI）领域已被广泛研究，但对于视觉蕴含（VE；Xie et al., 2019）中的人类标注差异却知之甚少——在视觉蕴含中，文本前提被图像所取代，而假设和标签空间保持不变。我们引入了VariNLI-VE，这是一个用于研究人类标注差异如何在成对的NLI与VE实例之间变化的数据集。我们对原本被标注为中性的SNLI（Bowman et al., 2015）实例的一个子集及其视觉对应版本进行了重新标注，由相同的标注者对文本版本和视觉版本进行评估。每个标注包含一个标签、一个置信度分数以及一段解释，共产生2,250个标注。标签和解释是联合收集的，从而提供了具有生态效度的解释。由于图像可能揭示文本中未陈述的信息，一个……（摘要原文在此处截断）

    arXiv:2611.17954v1 Announce Type: new  Abstract: Human Label Variation (HLV) is increasingly recognised as an informative signal that reflects meaningful differences in how people interpret the same input. While HLV has been studied extensively in Natural Language Inference (NLI), much less is known about HLV in Visual Entailment (VE; Xie et al., 2019), where the textual premise is replaced by an image, keeping the hypothesis and label space fixed. We introduce VariNLI-VE, a dataset for studying how HLV changes between paired NLI and VE instances. We re-annotate a subset of originally neutral SNLI (Bowman et al., 2015) instances and their visual counterparts, with the same annotators evaluating both the textual and visual versions. Each annotation includes a label, confidence score, and explanation, yielding 2,250 annotations. Labels and explanations are collected jointly, providing ecologically valid explanations. Because the image may reveal information not stated in the text, one mi
    
[^31]: 你能信任你的金融智能体吗？对一款已发布的金融研究智能体的漂移、诊断与可审计干预

    Can You Trust Your Finance Agent? Drift, Diagnosis and Auditable Intervention on One Shipped Financial Research Agent

    [https://arxiv.org/abs/2611.17939](https://arxiv.org/abs/2611.17939)

    论文对一款现成发布的金融研究智能体进行系统审计，发现其基准分数不稳定且运行成本波动高达17%，并按指令、工具、模型等责任层对失败轨迹进行归因分类，其诊断结果与人类盲评者一致率达94%，进而提出可审计的针对性干预方法。

    

    你能信任一个并非由你构建的金融智能体吗？我们采用Vals金融智能体基准v2（Vals Finance Agent Benchmark v2）随附的参考智能体，不对其进行任何调优，并提出了四个排行榜分数无法回答的问题。分数是否稳定？在18个问题上进行七次完全相同的运行，149项评分标准中通过的数量在105至123之间浮动，且账单费用差异高达17%。当智能体失败时，应由谁来负责修复？我们按照事先冻结的分类法，将来自九个保留问题的三次运行共20条失败轨迹，依据应由哪一层的责任方来修复进行归类：13条属于智能体的指令，3条属于其工具，4条属于模型，没有一条属于其数据。这样的诊断是否可信？来自三个模型家族的四位盲评者，在仅看到轨迹的情况下，在94%的案例中与我们的LLM分析器指出相同的故障，这一比例与盲评者彼此之间的一致率（93%）相当。所做的更改是否可信？一个优化循环读取轨迹，每轮进行一个针对性的更改……

    arXiv:2611.17939v1 Announce Type: cross  Abstract: Can you trust a finance agent you did not build? We take the reference agent shipped with the Vals Finance Agent Benchmark v2, leave it untuned, and ask four questions a leaderboard score does not answer. Is the score stable? Seven identical runs on 18 questions pass between 105 and 123 of 149 rubric criteria, and the bill varies by up to 17%. When the agent fails, who owns the fix? We classify 20 failed trajectories, from three runs of nine held-out questions, by the layer whose owner would remedy them, under a taxonomy frozen in advance: 13 belong to the agent's instructions, 3 to its tools, 4 to the model, none to its data. Can that diagnosis be trusted? Four blind readers from three model families, given only the traces, name the same fault as our LLM analyzer in 94% of cases, the rate at which they agree with one another (93%). Can a change be trusted? An optimization loop reads the traces, makes one targeted change per round and 
    
[^32]: MineralExp：面向真实世界矿产勘探中实验设计的LLM智能体基准测试

    MineralExp: Benchmarking LLM Agents for Experimental Design in Real-World Mineral Exploration

    [https://arxiv.org/abs/2611.17931](https://arxiv.org/abs/2611.17931)

    提出首个面向真实矿产勘探场景的MineralExp基准，用于评估LLM智能体自动化地球化学实验设计的能力，并解决了实验方法多样性与主观性高、开放式方案难以客观评分两大难题。

    

    在矿产勘探中，验证地质假设依赖于设计多种地球化学实验。虽然大语言模型（LLM）智能体日益自动化地完成下游的地球科学数据分析任务，但一个关键的上游能力在很大程度上仍未被探索：自动化地球化学实验设计，即在开展昂贵的实验室工作之前决定应该进行哪些实验。评估LLM智能体的这一能力面临两大困难。首先，地球化学实验具有高度的方法多样性和主观性：为了检验同一假设，不同的地质学家可能会选择不同的矿物和仪器，因此没有单一的实验方案可以作为标准参考答案。其次，开放式的实验方案难以表示为结构化标签，这使得客观评分变得困难。为了应对这些挑战，我们提出了MineralExp，这是一个将地球化学实验设计进行重新表述的基准……

    arXiv:2611.17931v1 Announce Type: cross  Abstract: In mineral exploration, verifying geological hypotheses relies on designing a variety of geochemical experiments. While large language model (LLM) agents increasingly automate downstream geoscientific data analysis, a crucial upstream capability remains largely unexplored: automated geochemical experimental design, i.e., deciding which experiments to conduct before committing to costly laboratory work. Evaluating this capability of LLM agents is difficult for two reasons. First, geochemical experiments exhibit high methodological diversity and subjectivity: to test the same hypothesis, different geologists may choose different minerals and instruments, so no single protocol can serve as the reference answer. Second, open-ended experimental plans resist representation as structured labels, which makes objective scoring difficult. To address these challenges, we introduce MineralExp, a benchmark that reformulates geochemical experimental
    
[^33]: AmbiSteer：在视觉语言模型中选择性引出歧义感知响应

    AmbiSteer: Selectively Eliciting Ambiguity-Aware Responses in Vision-Language Models

    [https://arxiv.org/abs/2611.17900](https://arxiv.org/abs/2611.17900)

    提出 AmbiSteer 轻量级框架，通过结合激活引导与提示方法，在推理时选择性地引出视觉语言模型的歧义感知行为，使其能够正确处理视觉问答中的指代歧义问题。

    

    视觉语言模型（VLMs）常常表现出过度自信，即使视觉证据不足时也会给出确定的答案。一个突出的例子是视觉问答（VQA）中的指代歧义问题，即图像中的多个实体都可能是问题的合理指代对象。处理这类情况需要识别歧义何时存在：对于有歧义的问题，可以通过请求澄清或列举可能的指代对象来处理，而对于无歧义的问题则应给出直接回答。当前的视觉语言模型难以做出这种区分，常常默认采用单一解释。为了研究激活引导是否可以作为一种推理时机制来引出此类行为，我们提出了 AmbiSteer，一个用于选择性引出歧义感知行为的轻量级框架。在三个视觉语言模型上的实验表明，仅使用激活引导的效果有限且依赖于具体模型，而提示方法通常更为有效。通过结合……

    arXiv:2611.17900v1 Announce Type: cross  Abstract: Vision-language models (VLMs) often exhibit overconfidence, providing firm answers even when visual evidence is insufficient. A prominent case is referential ambiguity in Visual Question Answering (VQA), where multiple entities in an image are plausible referents of a question. Addressing such cases requires recognizing when ambiguity is present: ambiguous questions can be handled by asking for clarification or enumerating plausible referents, while unambiguous questions should receive a direct answer. Current VLMs struggle with this distinction, often defaulting to a single interpretation. To investigate whether activation steering can serve as an inference-time mechanism to elicit such behaviors, we introduce AmbiSteer, a lightweight framework to selectively elicit ambiguity-aware behavior. Across three VLMs, we find that steering alone has limited and model-dependent effects, while prompting is generally more effective. Combining pr
    
[^34]: 从设计层面保障多智能体系统的安全

    Securing Multi-Agent Systems by Design

    [https://arxiv.org/abs/2611.17883](https://arxiv.org/abs/2611.17883)

    本文提出了响应操纵攻击以揭示多智能体系统面对恶意响应的安全隐患，并进一步提出基于响应间语义相似性的安全聚合规则SecMAS来有效防御此类攻击。

    

    多智能体系统通过将复杂任务分解为子任务、将子任务委托给各个智能体，并使用聚合规则汇总各智能体的响应，从而解决复杂任务。恶意智能体可以通过对其被分配的子任务提供恶意响应来操纵最终解决方案。在本文中，我们首先提出了响应操纵攻击，它通过迭代优化恶意响应来最大化攻击效果。具体而言，我们将无目标攻击和有目标攻击形式化为优化问题，并提出一种近似求解这些问题的方法，即基于替代多智能体系统的反馈，迭代地改进恶意响应。为防御此类攻击，我们进一步提出了SecMAS，一种安全聚合规则，它利用各智能体响应之间的语义相似性，选出与足够大的其他响应子集最为一致的响应。（摘要在此处截断）

    arXiv:2611.17883v1 Announce Type: cross  Abstract: Multi-agent systems solve complex tasks by decomposing them into subtasks, delegating the subtasks to agents, and aggregating the agents' responses using an aggregation rule. Malicious agents can manipulate the final solution by providing malicious responses to their assigned subtasks. In this paper, we first propose response manipulation attacks (ReMA), which iteratively optimize malicious responses to maximize attack effectiveness. Specifically, we formulate untargeted and targeted attacks as optimization problems and develop an approach to approximately solve these problems by iteratively refining malicious responses based on feedback from a surrogate multi-agent system. To defend against such attacks, we further propose SecMAS, a secure aggregation rule that leverages the semantic similarities among agent responses to select the response that is most consistent with a sufficiently large subset of the other responses. We theoretical
    
[^35]: Muon预训练中临时权重衰减的延迟效应

    Delayed Effects of Temporary Weight Decay in Muon Pretraining

    [https://arxiv.org/abs/2611.17876](https://arxiv.org/abs/2611.17876)

    该研究揭示了Muon预训练中临时权重衰减的延迟效应——早期衰减虽在结束时增加验证损失，但通过留下持久的权重范数赤字使相对步长保持较大，最终反而降低最终损失，且小规模模型上选出的衰减调度无需重新调参即可成功迁移到更大规模模型。

    

    一种临时的优化干预在其结束时可能会增加验证损失，但在最终训练预算下却能降低验证损失。我们通过在具有162M、405M、1.4B和3.4B参数的Muon训练的仅解码器Transformer中使用临时权重衰减来证明这一点。对于参数量高达1.4B的模型，将固定时长的衰减窗口从训练早期移动到训练后期会逆转其最终效果。在162M参数规模下，即使在总收缩量大致匹配的情况下，改变窗口持续时间也会改变最终损失。最引人注目的是，在162M规模下选择的调度方案在未经重新调参的情况下迁移到1.4B规模时，在移除时比无衰减差0.26 nats，仅在训练进行到73%之后才超越它，最终以好0.028 nats的成绩收尾，同时还超越了采样得到的最佳常数衰减基线。早期衰减会留下持久的权重范数赤字，这使得干预结束后的相对步长保持较大。一个近似的范数动力学递推关系刻画了这种缓慢收敛的现象。

    arXiv:2611.17876v1 Announce Type: cross  Abstract: A temporary optimization intervention can increase validation loss when it ends yet reduce it at the final training budget. We demonstrate this with temporary weight decay in Muon-trained decoder-only Transformers with 162M, 405M, 1.4B, and 3.4B parameters. For models up to 1.4B parameters, moving a fixed-duration decay window from early to late training reverses its final effect. At 162M, changing window duration alters final loss even at approximately matched total shrinkage. Most strikingly, a schedule selected at 162M and transferred without retuning to 1.4B is 0.26 nats worse than no decay when removed, overtakes it only after 73% of training, and ultimately finishes 0.028 nats better while also outperforming the best sampled constant-decay baseline. Early decay leaves a persistent weight-norm deficit, which keeps relative steps larger after the intervention ends. An approximate norm-dynamics recurrence captures the slow convergen
    
[^36]: UltraBERT-JEV：面向“系统一”应用的编码器模型

    UltraBERT-JEV: Encoder models for System One Applications

    [https://arxiv.org/abs/2611.17867](https://arxiv.org/abs/2611.17867)

    提出UltraBERT-JEV编码器模型，通过单次前向传播直接从多个选项中选择答案而无需生成文本，配合“是-否”差值打分等微调方案，准确率最高达72.1%并超越同数据训练的0.6B解码器模型，同时推理速度快一倍以上（29毫秒对比68毫秒）。

    

    我们研究了一种编码器模型，它可以在单次前向传播中从多个选项中直接挑选一个，而无需生成文本。本技术报告比较了微调 UltraBERT-380M（每个选项对应一个掩码）的几种方案。这四种方案的区别在于如何对选项打分以及如何将分数转换为概率，其中一种借鉴了 DeepSeek-V3 的专家门控机制。在与 Qwen3-0.6B 使用相同数据训练的情况下，采用 LLM-as-Jev 方案的解码器测试准确率为 67.6%，而编码器模型达到了 69.7% 至 72.1%。用“是”减“否”（而非仅用“是”）来为选项打分带来了更大的提升：测试集上从 69.7% 提升至 71.3%，外部基准上从 57.8% 提升至 59.6%（解码器为 56.1%）。Sigmoid 归一化额外带来 0.8 个测试点的提升，而平方根 softplus 则没有带来提升。在 NVIDIA A100 上单次处理一个请求时，编码器每次决策仅需 29 毫秒（解码器为 68 毫秒），但峰值内存占用更高（2.8 GiB 对比 2.2 GiB）。一个在人工标注证据和标签上训练的辅助头……（摘要原文在此处截断）

    arXiv:2611.17867v1 Announce Type: cross  Abstract: We study encoder models that pick one of several options in a single forward pass, without generating text. This technical report compares recipes for fine-tuning UltraBERT-380M with one mask per option. The four recipes differ in how an option is scored and how the scores become probabilities; one follows the expert gate of DeepSeek-V3. Trained on the same data as Qwen3-0.6B with the LLM-as-Jev recipe (67.6% test accuracy), the encoder reaches 69.7% to 72.1%. Scoring an option by "yes" minus "no" instead of "yes" alone gives the larger gain, from 69.7% to 71.3% on the test set and from 57.8% to 59.6% on external benchmarks (56.1% for the decoder). Sigmoid normalisation adds 0.8 test points; a square-root softplus adds none. One request at a time on an NVIDIA A100, the encoder takes 29 ms per decision against 68 ms, but uses more peak memory (2.8 against 2.2 GiB). An auxiliary head, trained on human-marked evidence and on labels select
    
[^37]: 消融编辑是否变得更难了？并非普遍如此：基于方向的消融方法与 ARA 在 Heretic 中对 Qwen3.8、Gemma 4、Olmo 3.1 和 Mistral Small 3.2 大语言模型与其前代模型的拒绝移除比较

    Is Abliteration Getting Harder? Not Uniformly: Refusal Removal by Direction-Based Ablation and ARA in Heretic on the Qwen3.8, Gemma 4, Olmo 3.1 and Mistral Small 3.2 LLMs Versus Their Predecessors

    [https://arxiv.org/abs/2611.17846](https://arxiv.org/abs/2611.17846)

    本研究在五组新旧模型对上，使用多种基于方向的消融方法与 ARA 在统一实验条件下进行系统对比，发现拒绝移除的难度并未随着模型版本的更新而普遍增加。

    

    消融编辑（Abliteration）是一种无需微调的权重修改技术，可以从开放权重的大语言模型（LLM）中移除拒绝行为。基于优化的变体——任意秩消融（ARA）的开发者报告称，通过消融拒绝方向来移除拒绝行为的基于方向的消融编辑在新模型上日益失效。本研究在五组发布版本对上发现，拒绝移除并没有普遍变得更难。研究在 Heretic 2.0 中对 Qwen2.5-32B 与 Qwen3.8-27B、Gemma-2-27B 与 Gemma-4-31B、OLMo-2-32B 与 Olmo-3.1-32B、Mistral-Small-2409 与 Mistral-Small-3.2，以及 Qwen2.5-7B 与 Qwen3.5-4B 这五组模型，使用三种或四种基于方向的方法（均值差、MPOA、SOM，以及在小模型对上使用的 J-space）和 ARA，在相同的提示词、名义搜索预算、首 token KL 散度上限和选择规则下进行编辑；当自动 WildGuard 判定器在 972 个保留的有害提示（来自 JailbreakBench、XSTest、StrongREJECT、HarmBench 以及作者编制的恶意提示集）中发现更多被成功响应的情况时，该模型对的移除被视为更难。

    arXiv:2611.17846v1 Announce Type: new  Abstract: Abliteration, a fine-tuning-free weight edit, removes refusal from open-weight large language models (LLMs). The developer of Arbitrary-Rank Ablation (ARA), an optimisation-based variant, reports that direction-based abliteration, which ablates a refusal direction, increasingly fails on newer models. On five release pairs, refusal removal did not become uniformly harder. Qwen2.5-32B and Qwen3.8-27B, Gemma-2-27B and Gemma-4-31B, OLMo-2-32B and Olmo-3.1-32B, Mistral-Small-2409 and Mistral-Small-3.2, and Qwen2.5-7B and Qwen3.5-4B were edited in Heretic 2.0 by three or four direction-based methods (difference-in-means, MPOA, SOM and, on the small pair, J-space) and ARA under the same prompts, nominal search budget, first-token KL divergence cap and selection rule; removal counted as harder when the automatic WildGuard judge found more of 972 held-out harmful prompts (JailbreakBench, XSTest, StrongREJECT, HarmBench and an author-compiled mali
    
[^38]: BLiMP-RO：罗马尼亚语语言学最小对立对基准测试

    BLiMP-RO: A Benchmark of Linguistic Minimal Pairs for Romanian

    [https://arxiv.org/abs/2611.17836](https://arxiv.org/abs/2611.17836)

    本文提出了BLiMP-RO——一个涵盖50种语法现象、包含493个语言学家构建及5942个半自动生成句子对的罗马尼亚语语法评测基准，并通过从零训练GPT-2和GPT-BERT模型探究了有限语言输入下的语法泛化能力。

    

    评估语言模型中的语法知识需要能够覆盖多样化语言现象并经过人类判断验证的基准测试。我们提出了BLiMP-RO，一个覆盖十个领域中50种语法现象的罗马尼亚语最小对立对基准测试。该基准包含493个由语言学家构建的句子对（通过五项实验由母语者进行评估），以及5,942个额外的半自动生成、经人工验证的句子对。我们进一步研究了有限语言输入下的语法泛化能力，通过在罗马尼亚语BabyBabelLM数据和新收集的1000万词儿童导向视频字幕语料库上从零开始训练GPT-2和GPT-BERT模型。增加字幕训练数据使得语料库规模每扩大十倍，准确率约提升9个百分点。在相同数据量下，BabyBabelLM总体上优于视频字幕，而GPT-BERT始终优于GPT-2。

    arXiv:2611.17836v1 Announce Type: new  Abstract: Evaluating grammatical knowledge in language models requires benchmarks that cover diverse linguistic phenomena and are validated against human judgments. We introduce BLiMP-RO, a Romanian minimal-pair benchmark covering 50 grammatical phenomena across ten domains. It comprises 493 linguist-constructed sentence pairs, evaluated by native speakers in five experiments, and 5,942 additional semi-automatically generated, human-validated pairs. We further investigate grammatical generalization under limited linguistic input by training GPT-2 and GPT-BERT models from scratch on Romanian BabyBabelLM data and a newly collected corpus of 10 million words from child-targeted video captions. Increasing caption training data yields approximately 9 percentage points of accuracy improvement per tenfold increase in corpus size. At matched data sizes, BabyBabelLM generally outperforms video captions, while GPT-BERT consistently outperforms GPT-2. Our fi
    
[^39]: PaperCritic：训练AI审稿人来评估论文的可靠性

    PaperCritic: Training AI Reviewers to Assess the Soundness of Papers

    [https://arxiv.org/abs/2611.17834](https://arxiv.org/abs/2611.17834)

    该论文提出PaperCritic模型，通过构建包含23K条质量标注审稿观点的PaperSound-23K数据集，并以精确率和召回率的F1分数作为奖励进行强化学习训练，实现了专注于客观评估论文可靠性的AI审稿人。

    

    随着AI会议投稿量的快速增长，AI审稿人正被视为一种替代方案。然而，简单地模仿人类审稿是有问题的，因为审稿服务于多种目的，包括评估论文的可靠性、趣味性和清晰度，其中后两者是主观的。在这项工作中，我们仅专注于评估可靠性——最重要且客观的标准，并将学习目标定义为同时最大化：（1）精确率，基于审稿观点的可靠性和实质性；（2）召回率，即对高质量参考审稿观点的覆盖率。基于这一问题设定，我们利用1K篇ICLR投稿和23K条人类撰写及LLM生成的审稿观点，构建了一个带有质量标签的审稿观点数据集PaperSound-23K。利用该数据集，我们通过强化学习训练了一个AI审稿人模型PaperCritic，以精确率和召回率的F1分数作为奖励。

    arXiv:2611.17834v1 Announce Type: new  Abstract: As AI conference submissions grow rapidly, AI reviewers are being considered as an alternative. However, simply imitating human reviewing is problematic because reviewing serves multiple purposes, including evaluating papers' soundness, interestingness, and clarity, the last two of which are subjective. In this work, we focus solely on evaluating soundness, the most important and objective criterion, and define the learning objective as jointly maximizing (1) precision, based on the soundness and substantiveness of review points, and (2) recall, the coverage of high-quality reference review points. Under this problem formulation, we construct a quality-labeled review point dataset, PaperSound-23K, using 1K ICLR submissions and 23K human-written and LLM-generated review points. Using this dataset, we train an AI reviewer model, PaperCritic, through reinforcement learning with the F1 score of precision and recall as the reward. Our experim
    
[^40]: G-CAST：面向电商智能体在需求持续演变下的受治理持续自训练

    G-CAST: Governed Continual Self-Training for E-Commerce Agents under Evolving Demands

    [https://arxiv.org/abs/2611.17814](https://arxiv.org/abs/2611.17814)

    G-CAST提出了一套受治理的持续自训练系统，使电商LLM智能体能够在用户需求漂移和执行框架不断演进的条件下自动持续适应，从而摆脱滞后且高成本的人工周期性再训练。

    

    随着大型语言模型（LLM）的快速发展，越来越多的用户开始借助LLM驱动的智能体进行电商对话式购物。然而，随着新商品品类、采购场景和功能需求的不断涌现，用户需求也在持续变化。与此同时，产品的执行框架（包括其工具、技能和操作流程）也在不断演进，以支持日益复杂的交互。这些变化带来了两个实际挑战：1）需求漂移改变了用户请求的分布、意图和复杂度，使一次性训练的模型逐渐落后；2）框架演进要求模型正确使用新引入的能力，同时不丢失用户已依赖的既有行为。在这种场景下，周期性的人工再训练过于滞后且耗费人力。为应对这些挑战，我们提出了G-CAST，一个受治理的持续自训练系统……（原文摘要到此截断）

    arXiv:2611.17814v1 Announce Type: new  Abstract: With the rapid advancement of Large Language Models (LLMs), an increasing number of users are turning to LLM-powered agents for conversational shopping in e-commerce. However, the user needs continue to change as new product categories, procurement scenarios, and feature requests emerge. At the same time, the product's execution harness-including its tools, skills, and operating procedures-continues to evolve to support increasingly sophisticated interactions. These changes create two practical challenges: 1) Demand drift alters the distribution, intent, and complexity of user requests, causing a model trained once to fall behind; 2) Harness evolution requires the model to use newly introduced capabilities correctly without losing behavior that users already rely on. Periodic manual retraining is too delayed and labor-intensive for this setting. To address these challenges, we present G-CAST, a governed continual self-training system tha
    
[^41]: AI审稿偏向附录

    AI Reviews Are Biased Towards the Appendix

    [https://arxiv.org/abs/2611.17811](https://arxiv.org/abs/2611.17811)

    研究发现GPT和Claude等大语言模型在审稿时约50%以上的批评意见来自论文附录，而人类仅有8.4%，且AI的批评多集中于与核心贡献无关的细节，揭示了AI审稿存在系统性偏向附录的偏差。

    

    大型语言模型（LLM）越来越多地被审稿人用来在审稿过程中指出论文的弱点，其输出往往会塑造人类最终提交的审稿意见。先前的研究探讨了LLM审稿与人类审稿的一致性，但对于这些模型实际批评论文的哪些部分，我们知之甚少。在本工作中，我们研究了来自ACL、EMNLP、COLM和ICLR的2000篇论文，发现最新的GPT和Claude模型约50%或更多的审稿弱点来自论文的附录。这一行为在使用LLM审稿的API、编程代理和网络访问三种方式中均普遍存在。相比之下，我们发现人类只有8.4%的弱点来自附录，且这些弱点主要围绕论文的表述方式以及与其核心思想相关的内容。与此同时，LLM从附录深处提取的弱点则涉及与论文核心贡献无关的细粒度技术细节。

    arXiv:2611.17811v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used by reviewers to highlight weaknesses during the reviewing process, and their outputs often shape the reviews that humans submit. Prior work studies how LLM reviews align with human ones, yet we know little about which parts of a paper these models actually criticize. In this work, we investigate 2,000 papers from ACL, EMNLP, COLM, and ICLR, and show that the latest GPT and Claude models draw about 50% or more of their review weaknesses from the appendix. This behavior holds across the API, coding-agent, and web access methods for using LLMs to review a paper. In contrast, we find that humans draw only 8.4% of their weaknesses from the appendix, and these weaknesses primarily revolve around a paper's presentation and content related to its core idea. Meanwhile, LLMs draw weaknesses from deep in the appendix that concern fine-grained technical details not central to the paper's contributio
    
[^42]: CarryOn：为长程LLM智能体学习在上下文中保留什么

    CarryOn: Learning What to Keep in Context for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2611.17808](https://arxiv.org/abs/2611.17808)

    CarryOn是一个轻量级上下文管理器，通过学习的选择器和自适应保留策略，将紧凑的执行状态与细粒度的来源链接证据相结合，无需额外LLM调用即可为长程LLM智能体实现可靠的上下文管理。

    

    长程LLM智能体能够协调工具和应用程序以完成复杂的多步骤任务。随着交互历史的增长，有效的上下文管理对于可靠且高效的执行变得至关重要。现有方法通过摘要、剪枝或历史选择来缩减上下文。然而，更短的上下文并不一定支持更好的决策：摘要可能省略后续操作所需的精确细节，而保留完整的交互则会携带大量无关信息。我们提出了CarryOn，一个轻量级的上下文管理器，它将紧凑的执行状态与细粒度的、带有来源链接的证据相结合。一个经过学习的选择器和自适应保留策略能够保持相关证据的可用性，而无需额外的LLM调用来进行上下文管理。我们引入了两个互补的评估指标：$S_\alpha$在考虑资源使用的同时优先考虑任务成功率，而每次成功的归一化成本（NCS）用于衡量……

    arXiv:2611.17808v1 Announce Type: cross  Abstract: Long-horizon LLM agents can coordinate tools and applications to complete complex, multi-step tasks. As their interaction histories grow, effective context management becomes essential for reliable and efficient execution. Existing methods reduce context through summarization, pruning, or history selection. However, shorter contexts do not necessarily support better decisions: summaries can omit exact details needed for later actions, while retaining entire interactions can carry substantial irrelevant information. We propose CarryOn, a lightweight context manager that pairs compact execution state with fine-grained, source-linked evidence. A learned selector and adaptive retention policy keep relevant evidence available without additional LLM calls for context management. We introduce two complementary evaluation metrics: $S_\alpha$ prioritizes task success while accounting for resource use, and normalized cost per success (NCS) measu
    
[^43]: RAG-FGO：面向生成式RAG的查询重写智能体动态优化

    RAG-FGO: Dynamic Optimization of Query-Rewriting Agents for Generative RAG

    [https://arxiv.org/abs/2611.17788](https://arxiv.org/abs/2611.17788)

    提出RAG-FGO动态优化框架，将查询重写建模为混合离散-连续空间中不可微的黑盒优化问题，通过可自适应优化的查询重写智能体联合优化提示模板与解码参数，以应对生成式RAG中的噪声反馈和静态策略泛化不足的问题。

    

    在检索增强生成（RAG）中，查询重写作为将用户意图与外部知识库对齐的核心接口，其有效性直接决定了整体系统的性能。然而，生成式检索在真实部署中面临三个关键挑战：检索反馈中的噪声、离散提示模板与连续解码控制参数难以联合优化，以及静态重写策略在语义复杂场景下泛化能力有限。为解决这些问题，我们提出了RAG-FGO，一个面向生成式检索策略的动态优化框架。RAG-FGO将查询重写表述为混合离散-连续空间中不可微的黑盒优化问题，并将提示模板和解码参数表示为可自适应优化的查询重写智能体。该框架通过全局搜索实现语义跳跃和结构（摘要在此处被截断）

    arXiv:2611.17788v1 Announce Type: new  Abstract: In retrieval-augmented generation (RAG), query rewriting serves as a core interface that aligns user intent with external knowledge bases, and its effectiveness directly determines overall system performance. However, generative retrieval faces three critical challenges in real-world deployment: noisy retrieval feedback, the difficulty of jointly optimizing discrete prompt templates and continuous decoding-control parameters, and the limited generalization of static rewriting strategies under semantically complex scenarios. To address these issues, we propose RAG-FGO, a dynamic optimization framework for generative retrieval strategies. RAG-FGO formulates query rewriting as a non-differentiable black-box optimization problem in a mixed discrete-continuous space, and represents prompt templates and decoding parameters as adaptively optimizable query-rewriting agents. The framework performs global search to enable semantic jumps and struct
    
[^44]: LuthorSQL：作为较慢 LLM 智能体的“系统 1”的单遍文本到 SQL 模型

    LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent

    [https://arxiv.org/abs/2611.17780](https://arxiv.org/abs/2611.17780)

    提出非自回归 7B 模型 LuthorSQL，作为 LLM 智能体的快速“系统 1”门控，在单 GPU 上每题 0.84 秒即达到 BIRD 测试集 68.03% 的执行准确率，仅在不确定时才将查询升级给较慢的智能体，从而大幅降低成本并保护数据隐私。

    

    使用工具查询数据库的前沿 LLM 智能体是目前最准确的文本到 SQL 系统之一，但每次查询调用托管模型速度慢、成本高，且会将组织的敏感查询暴露给外部 API。例如，一个每天处理 10,000 次查询的 Claude Opus 5.5 智能体，按标价计算每年成本超过 70,000 美元。那么，组织已有的本地硬件能在多大程度上卸载这一负担而不损害智能体的准确性？我们提出了 LuthorSQL，一个非自回归的 7B 文本到 SQL 模型，它作为较慢 LLM 智能体（系统 2）的快速“系统 1”门控，仅在不确定时才将查询升级给智能体。LuthorSQL 通过单次前向传播，在受限文法上预测查询计划的槽位分数，编译并运行最多 32 个候选计划，并使用训练好的验证器选出答案。仅凭自身，LuthorSQL 在 BIRD 隐藏测试集上达到 68.03% 的执行准确率，在单个 GPU 上每个问题仅需 0.84 秒。

    arXiv:2611.17780v1 Announce Type: cross  Abstract: Frontier LLM agents that query databases with tools are among the most accurate text-to-SQL systems, but per-query calls to a hosted model are slow, costly, and expose sensitive organizational queries to external APIs. For example, a Claude Opus 5.5 agent serving 10,000 queries a day costs over $70,000 a year at list price. To what extent can local hardware already owned by an organization offload this burden without compromising the agent's accuracy? We present LuthorSQL, a non-autoregressive 7B text-to-SQL model that serves as a rapid System 1 gate for a slower LLM agent (System 2), escalating queries only when uncertain. LuthorSQL makes a single forward pass to predict slot scores over a bounded grammar for query plans, compiles and runs up to 32 candidate plans, and chooses an answer using a trained validator. On its own, LuthorSQL reaches 68.03% execution accuracy on BIRD's hidden test set at 0.84 s per question on a single GPU, m
    
[^45]: 激活引导《台风》中的“南山号”

    Activation Steering the Nan-Shan in Typhoon

    [https://arxiv.org/abs/2611.17768](https://arxiv.org/abs/2611.17768)

    本文介绍了基于康拉德《台风》的视听装置作品，通过让大语言模型以自身输出为食并采用一种新颖的自适应激活引导方法，探索AI生成文本随时间推移变得日益陌生和失控的过程，以此反思人类文化当下的人工智能时刻。

    

    《台风》是一件基于约瑟夫·康拉德中篇小说的视听装置作品，小说中麦克威尔船长驾驶“南山号”轮船直接驶入并穿越一场灾难性的风暴。这部小说研究了当一种现象的规模超出可用于理解它的实践与概念框架时会发生什么，以及在如此压倒性的情境下被要求采取行动时有时随之而来的瘫痪状态。我们选择《台风》这部文本来反思人类文化当前的人工智能时刻。我们以康拉德的原始语言作为种子提示一个大语言模型，使其无休止地以自身输出为食，并探索文本随着时间推移何时以及如何变得越来越陌生和难以驾驭，包括偏离叙事、游移至模型自身的指令和思考，以及长时间不连贯的沉思。我们采用了来自激活引导领域的方法，包括我们引入的一种新颖的自适应引导方法。

    arXiv:2611.17768v1 Announce Type: cross  Abstract: Typhoon is an audiovisual installation based on Joseph Conrad's novella, in which Captain MacWhirr steers the steamer Nan-Shan directly into and through a catastrophic storm. The novella is a study of what happens when the scale of a phenomenon exceeds the practical and conceptual frameworks available to comprehend it and of the paralysis that sometimes follows a demand for action in such overwhelming circumstances. We chose the text of Typhoon to reflect on human culture's current AI moment. We prompt a large language model, seeded with Conrad's original language, to feed endlessly on its own output, and explore when and how the text becomes increasingly alien and unwieldy as time progresses, including deviations from the narrative, digressions into the model's own instructions and thinking, and long periods of incoherent rumination. We employ methods from activation steering, including a novel adaptive steering method we introduce, t
    
[^46]: GDP.xlsx：基于专业工作簿的前沿模型基准测试

    GDP.xlsx: Benchmarking Frontier Models on Professional Workbooks

    [https://arxiv.org/abs/2611.17766](https://arxiv.org/abs/2611.17766)

    该论文提出GDP_xlsx基准，利用13个领域专业人士撰写的70个真实工作簿任务和791项专家评分准则来评估前沿AI模型的电子表格处理能力，发现最强模型Gemini 4 Argon的通过率也仅有38.3%，表明该领域仍有巨大的提升空间。

    

    我们推出了GDP_xlsx基准，包含70个由13个领域的专业人士撰写的真实职场请求，每个请求都配有专家制定的二元评分标准，以及其所指向的完整工作簿或CSV文件。这些文件共包含218万个已填充单元格，评分标准包含791项准则。智能体在配备电子表格工具的容器中工作，由一个能够打开文件的智能体裁判来应用评分标准。完整性和排他性准则要求集合型答案必须完整找出且没有多余内容。我们采用严格的pass@1指标对模型进行排名，即满足所有准则的尝试占比。在包括来自Google、Anthropic和OpenAI的前沿模型在内的20个模型配置中，得分最高的Gemini 4 Argon也只有38.3%的尝试通过，中位数得分为19.3%，留下了很大的提升空间。我们发布了评估框架和三个公开任务；其余67个任务被保留，以保护排行榜免受污染。

    arXiv:2611.17766v1 Announce Type: cross  Abstract: We introduce GDP_xlsx, a benchmark of 70 realistic workplace requests written by professionals in 13 domains, each paired with an expert rubric of binary criteria and with the complete workbook or CSV file it refers to. Together the files hold 2.18 million populated cells, and the rubrics 791 criteria. Agents work in a container with spreadsheet tools, and an agentic judge that can open the file applies the rubric. Completeness and exclusivity criteria require set-valued answers to be found in full and with nothing extra. We rank models by strict pass@1, the share of attempts that meet every criterion. Across 20 model configurations, including frontier models from Google, Anthropic, and OpenAI, the top-scoring one, Gemini 4 Argon, passes only 38.3% of attempts, and the median score is 19.3%, leaving wide headroom. We release the evaluation harness and three public tasks; the other 67 are held out to protect the leaderboard from contami
    
[^47]: 每块只见一次：掩码扩散语言模型的方差缩减后训练

    See Every Block Once: Variance-Reduced Post-Training for Masked Diffusion Language Models

    [https://arxiv.org/abs/2611.17754](https://arxiv.org/abs/2611.17754)

    提出一种全块估计器，通过对每个块恰好评估一次并取平均，在保持无偏的同时消除块选择带来的组间方差，从而显著降低掩码扩散语言模型后训练中的估计方差。

    

    策略梯度方法被广泛用于使大语言模型的输出与人类偏好对齐，但其所需的对数似然在掩码扩散语言模型（MDLM）中无法直接计算。因此，MDLM的后训练依赖于基于证据下界（ELBO）的代理目标，近期工作通过蒙特卡洛方法估计该目标的分块变体以支持分块生成。然而，现有的估计方案随机采样块索引，因此可能对同一个块进行多次评估，而某些块却未被评估，这给估计引入了源自块选择的方差。在本工作中，我们提出了一种全块估计器，它对每个块恰好评估一次并对结果取平均。我们在理论上证明，所提出的估计器在消除因块选择产生的组间方差的同时保持无偏，从而实现了更低的方差。

    arXiv:2611.17754v1 Announce Type: new  Abstract: Policy gradient methods are widely used to align large language model outputs with human preferences, but the log-likelihoods they require cannot be directly computed in Masked Diffusion Language Models (MDLMs). MDLM post-training therefore relies on a surrogate objective based on the Evidence Lower Bound (ELBO), and recent work estimates a blockwise variant of this objective via Monte Carlo methods to support blockwise generation. However, existing estimation schemes randomly sample the block index, so they may evaluate the same block more than once and leave some blocks unevaluated. This introduces variance arising from block selection into the estimates. In this work, we propose a full-block estimator that evaluates every block exactly once and averages the results. We theoretically show that the proposed estimator remains unbiased while eliminating the between-group variance arising from block selection, thereby achieving lower varia
    
[^48]: Superfluid：面向分布式大语言模型推理的本地AI服务器

    Superfluid: Local AI Server for Distributed LLM Inference

    [https://arxiv.org/abs/2611.17753](https://arxiv.org/abs/2611.17753)

    Superfluid是一个开源本地LLM推理服务器，通过统一调度器在单设备上协调多智能体与交互用户共享模型服务（支持llama.cpp、MLX、BaseRT等运行时），并能将多台异构闲置机器聚合为单一API端点实现分布式推理。

    

    在本地设备上运行大语言模型（LLM）可以保护数据隐私、免除按token计费的成本、降低网络延迟，并在无网络连接时仍能正常工作。如今，两种变化正对这类模型的服务方式构成挑战。在单台设备上，多个智能体与交互式用户共享同一个本地服务的模型；在多台设备之间，开发者或小团队往往拥有来自不同厂商的多台性能不俗的机器，但由于每个本地服务器只能在一台机器上服务一个运行时，这些机器在一天中的大部分时间都处于闲置状态。我们提出了Superfluid，一个为上述两种规模的本地推理而构建的开源LLM服务器。在单台设备上，Superfluid通过统一的调度器驱动运行于隔离工作进程中的llama.cpp、MLX或BaseRT，将交互式请求排在智能体任务之前优先处理，对被多个请求共享的提示词只计算一次，并在每个token到达客户端之前将其写入持久化日志。在多台设备之间，一个Superfluid服务器可以将一组机器整合为单一的API端点……

    arXiv:2611.17753v1 Announce Type: cross  Abstract: Running large language models (LLMs) on local devices keeps data private, removes per-token cost, cuts network latency, and keeps working without a connection. Two shifts now strain how such models are served. On one device, several agents share a locally served model with an interactive user. Across devices, a developer or small team often owns several capable machines from different vendors, which sit idle most of the day because every local server serves one runtime on one machine. We present Superfluid, an open-source LLM server built for local inference at both scales. On one device, Superfluid drives llama.cpp, MLX, or BaseRT in isolated worker processes through one scheduler that moves interactive requests ahead of agent work, computes a prompt shared by many requests once, and writes every token to a durable log before it reaches the client. Across devices, one Superfluid server turns a fleet of machines into a single API endpo
    
[^49]: 多语言自监督语音编码器对句子意义了解多少？

    What Do Multilingual Self-Supervised Speech Encoders Know About Sentence Meaning?

    [https://arxiv.org/abs/2611.17686](https://arxiv.org/abs/2611.17686)

    该研究通过将冻结语音表示线性映射到文本嵌入空间并进行跨模态检索，首次大规模评估了自监督语音模型在76种语言中的句子级语义可解码性，发现语义可访问性在语言间高度不均衡、参数扩展偏向高资源语言，且ASR调优虽提升语义可解码性却降低了表示的语言无关性。

    

    自监督语音模型（S3Ms）已经扩展到支持数千种语言和数百万小时的音频，然而人们对这些模型如何跨语言组织语义知识仍然知之甚少。在这项工作中，我们在覆盖76种语言的大规模多语言S3Ms中探测句子级语义，方法是将冻结的语音表示通过线性映射转换到冻结的文本嵌入空间，并通过跨模态检索进行评估。研究发现：句子意义是线性可访问的，但在不同语言和资源层级之间高度不均衡。跨语言探测在探测训练中从未见过的语言上能够匹敌特定语言的探测效果，但每个模型自身单语言的可解码性并不能预测其跨语言迁移能力。在OmniASR wav2vec系列中，从3亿到70亿的参数扩展不成比例地使高资源语言受益，而ASR调优显著提升了语义可解码性，同时使表示更加偏向语言特异性。然而，语义内容与（摘要在此处截断）

    arXiv:2611.17686v1 Announce Type: new  Abstract: Self-supervised speech models (S3Ms) have been scaled up to thousands of languages and millions of hours of audio, yet little is known about how these models structure semantic knowledge across languages. In this work, we probe sentence-level semantics in massively multilingual S3Ms across 76 languages, using linear maps from frozen speech representations into a frozen text embedding space, evaluated by cross-modal retrieval. Sentence meaning is linearly accessible but highly uneven across languages and resource tiers. Cross-lingual probes can match language-specific probes on languages never seen in probe training, but per-model monolingual decodability does not predict transfer. Within the OmniASR wav2vec family, scaling from 300M--7B disproportionately benefits high-resource languages, and ASR tuning sharply improves semantic decodability while making representations more language-specific. However, semantic content is entangled with 
    
[^50]: LLM智能体所见与用户所言

    What LLM Agents See and What Users Say

    [https://arxiv.org/abs/2611.17639](https://arxiv.org/abs/2611.17639)

    该研究运用语言学预设理论系统分析了LLM智能体在工具输出与用户陈述冲突时的行动边界，发现只有当搜索结果明确“列出”多个匹配项时才能阻止大多数智能体基于错误假设行动，而仅“计数”或依赖用户声明则不足以防止误操作。

    

    在LLM智能体为用户取消预订或删除文件夹之前，是什么阻止它基于错误假设采取行动：是其工具所显示的内容，还是用户所说的话？在这项工作中，我们借助语言学中的预设理论来研究这一差距：智能体只有在行动背后的每个命题（例如，一个预订恰好对应一个名字）被观察到成立、得到用户的回应，或在行动可撤销的前提下被披露时，才可以行动。一个简单的界限表明，当查询无法区分一条记录与两条记录时，任何智能体在存在两条记录时与只有一条记录时一样，都会不加询问就直接行动。随后，我们改变搜索结果是列出、计数还是省略第二个匹配项，以及用户是否陈述了会被记录反驳的声明（例如，“它是空的”）。在仅使用角色提示的情况下，对第二条记录的计数会导致36个早期智能体中的大多数仍然只对其所看到的那一条采取行动，而列出两条记录则能阻止大多数智能体的错误行动。然而，当用户声称某条记录（如文件夹）为空时，对其唯一一条项目的计数会……（摘要原文在此处截断）

    arXiv:2611.17639v1 Announce Type: new  Abstract: Before an LLM agent cancels a reservation or deletes a folder for a user, what stops it from acting on a false assumption: what its tools show, or what the user says? In this work, we study this gap with presupposition theory from linguistics: an agent may act only if each claim behind the act (e.g., one reservation fits the name) is seen to hold, answered by the user, or disclosed while the act can be undone. A simple bound shows that when lookups cannot tell one record from two, any agent acts without asking first as often with two records as with one. We then vary whether a search lists, counts, or omits a second match, and whether the user states a claim that a record refutes (e.g., "It is empty"). Under a role-only prompt, a count of a second record leaves most of 36 earlier agents acting on the one they see, whereas listing both stops most. When the user says a record such as a folder is empty, however, a count of its one item stop
    
[^51]: 更多智能体、更多样本，还是更多思考？解耦多智能体大语言模型中的协调、集成与测试时计算

    More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs

    [https://arxiv.org/abs/2611.17610](https://arxiv.org/abs/2611.17610)

    本研究通过综合516个多智能体配置的元分析发现，多智能体大语言模型的表现优势主要来自额外的样本数量和测试时计算量，而非智能体间的协调机制本身——在与资源匹配的单模型基线公平对比时，多智能体系统在74%的情况下并无优势甚至更差。

    

    多智能体语言模型的评估通常将具有通信机制的架构与单次贪婪推理进行比较，这种做法将协调效应与样本数量和推理计算量混为一谈。本文综合了九项主要实证评估，涵盖516个独特的多智能体配置，沿一个包含六个维度的资源向量R进行组织，并构建了包含四个类别的基线阶梯。与单次推理基线（A类）相比，多智能体的准确率差异总体为负，在剔除贡献最大的一项研究后接近于零。当与按样本数量（B类）或资源预算（C类）匹配的基线进行比较时，多智能体系统的平均准确率差异为-0.042，且基于对数几率尺度的多层随机效应建模在各项敏感性检验中均保持负值。在74%的可比对比中，多智能体系统的准确率等于或低于基线，且各研究的均值几乎均为非正值。

    arXiv:2611.17610v1 Announce Type: new  Abstract: Evaluations of multi-agent language models often compare communicating architectures against a single greedy pass. This practice confounds coordination with sample multiplicity and inference compute. We synthesize nine primary empirical evaluations comprising 516 unique multi-agent configurations organized along a resource vector R with six dimensions and a baseline ladder with four categories. Against baselines with a single pass (Category A), multi-agent accuracy differences are negative overall and near zero when omitting the largest contributing study. When evaluated against baselines matched on sample count (Category B) or resource budget (Category C), multi-agent systems average -0.042 in accuracy, and multilevel random effects modeling on the log odds scale remains negative across sensitivity checks. Multi-agent accuracy equals or falls below the baseline in 74% of these comparable contrasts, and study means are nonpositive in alm
    
[^52]: LightTAR：用于多模态表格学习的轻量快速目标感知表示

    LightTAR: Light and Fast Target-Aware Representations for Multimodal Tabular Learning

    [https://arxiv.org/abs/2611.17600](https://arxiv.org/abs/2611.17600)

    LightTAR是一种简单低成本的降维增强方法，它利用正则化PLS从编码器最后一层和中间层提取目标感知特征来补充标准PCA表示，从而在21种表格模型上普遍提升多模态表格学习的预测性能。

    

    多模态表格学习涉及对同时包含结构化数值与类别特征以及非结构化文本和图像的表格进行分类和回归。这些非结构化字段通常由预训练编码器的嵌入来表示，并使用主成分分析（PCA）进行降维。这种降维优先保留输入方差，但可能忽略与预测目标相关的信息。我们提出了LightTAR，这是一种简单且低开销的方法，能够使降维过程与预测任务相适应。LightTAR利用正则化偏最小二乘（PLS）从编码器的最后一层和某一中间层联合提取少量目标感知特征，以此增广标准的PCA表示。我们使用21种表格模型对LightTAR进行了评估，包括树集成模型、神经网络以及表格基础模型。LightTAR提升了所有模型的平均预测性能，并且在跨模型的平均意义上改善了预测……

    arXiv:2611.17600v1 Announce Type: cross  Abstract: Multimodal tabular learning involves classification and regression on tables combining structured numerical and categorical features with unstructured text and images. These unstructured fields are commonly represented by pretrained encoder embeddings reduced with principal component analysis (PCA). This reduction prioritizes input variance but may overlook information relevant to predicting the target. We introduce LightTAR, a simple, inexpensive method that adapts dimensionality reduction to the prediction task. LightTAR augments the standard PCA representation with a few target-aware features extracted jointly from the final and an intermediate encoder layer using regularized partial least squares (PLS). We evaluate LightTAR with 21 tabular models, including tree ensembles, neural networks, and tabular foundation models. LightTAR improves average predictive performance for all models and, averaged across models, improves prediction 
    
[^53]: 面向群体决策的小型开放权重语言模型：输出级融合与语言聚合的能力与局限

    Small Open-Weight Language Models for Crowd Decision-Making: Capabilities and Limits of Output-Level Fusion and Linguistic Aggregation

    [https://arxiv.org/abs/2611.17589](https://arxiv.org/abs/2611.17589)

    该论文首次系统检验了在机构本地基础设施上运行的小型开放权重语言模型，能否通过重复查询与基于二元组语言表示的输出级融合来稳定群体决策结果，发现连续评估可保持信息分辨率，但对参与评论者的敏感性在降低运行间变异后依然存在。

    

    小型开放权重语言模型可以在机构自有基础设施上运行群体决策（CDM），从而将模型检查点、推理设置和意见都保持在机构的控制之下。这些模型的判断在重复查询中会有所变化；是否可以通过重复并聚合这些判断——通常借助二元组（2-tuple）语言表示——来确定决策结果而非仅取决于标签，这一点此前尚未经过检验。研究让五个总参数量为80亿至360亿的开放权重模型在一块消费级GPU上运行，外加一个托管参考模型，对伦敦四家餐厅的474条TripAdvisor评论各分类十次；三个开放权重模型又在马德里十二家餐厅的600条评论上重复了该协议。结果表明，连续评估保持了分辨率：仅对每条评论离散化时只剩5个不同的值（原为21-65个），而在每一层级都离散化则最多只剩两个集体值。在降低了运行间变异性之后，对参与者的敏感性依然存在：在伦敦，重新抽取评论者后获胜餐厅保持不变……（原文摘要截断）

    arXiv:2611.17589v1 Announce Type: new  Abstract: Small open-weight language models can run crowd decision-making (CDM) on an institution's own infrastructure, which keeps the checkpoint, the inference settings and the opinions under its control. Their judgements vary across repeated queries; whether repeating and aggregating them, often through the 2-tuple linguistic representation, settles the decision rather than the labels is untested. Five open-weight models of 8-36B total parameters on one consumer GPU and a hosted reference classify 474 TripAdvisor reviews of four London restaurants ten times each; three open-weight models repeat the protocol on 600 reviews of twelve Madrid restaurants. Continuous evaluations preserve resolution: snapping each review leaves 5 distinct values against 21-65, and snapping at every level at most two collective values. Sensitivity to the participants persists once run-to-run variability is reduced: in London, redrawing the reviewers keeps the winner o
    
[^54]: CamSpokenNLI：用于喀麦隆皮钦英语与Camfranglais跨变体迁移评估的试点资源

    CamSpokenNLI: A Pilot Resource for Cross-Variety Transfer Evaluation in Cameroon Pidgin English and Camfranglais

    [https://arxiv.org/abs/2611.17573](https://arxiv.org/abs/2611.17573)

    本文推出了首个面向喀麦隆皮钦英语和Camfranglais两种语言变体的口语自然语言推理试点数据集CamSpokenNLI（共228条条目），实验发现XLM-RoBERTa基线表现仅略高于随机水平且与仅使用假设的输入难以区分，凸显了这些低资源变体上NLI任务的挑战性。

    

    喀麦隆皮钦英语（西非皮钦英语的一种变体）和Camfranglais在喀麦隆被广泛使用，但据我们所知，这两种语言变体目前都没有专门的口语自然语言推理（NLI）资源。我们推出了CamSpokenNLI，这是一个试点数据集，包含每种变体114个场景匹配的口语NLI条目（共228条），由一位来自布埃亚的母语者录制。每个条目将前提音频和文本与假设、NLI标签以及英文注释配对。由于标签由作者本人分配，该数据集属于探索性资源，而非成熟的基准。NLI实验使用参考文本；音频则单独评估。采用五折条目级交叉验证的XLM-RoBERTa基线在皮钦英语上达到44.7%的准确率，在Camfranglais上达到46.2%（随机水平为33.3%）。尽管如此，两者的结果在统计上均无法与仅使用假设的输入区分开来，因此假设的措辞可能构成了准确率的主要来源。在不相交条目的跨变体迁移条件下……（摘要原文在此处被截断）

    arXiv:2611.17573v1 Announce Type: new  Abstract: Cameroon Pidgin English (a West African Pidgin English variety) and Camfranglais are widely used in Cameroon, yet to our knowledge, neither has a dedicated spoken Natural Language Inference (NLI) resource. We introduce CamSpokenNLI, a pilot dataset of 114 scenario-matched spoken NLI items in each variety (228 in total), recorded by one native speaker from Buea. Each item pairs premise audio and text with a hypothesis, an NLI label, and English glosses. The author assigned labels, so the dataset is an exploratory resource rather than a mature benchmark. The NLI experiments use the reference text; the audio is evaluated separately. XLM-RoBERTa baselines with five-fold item-level cross-validation reach 44.7% on Pidgin and 46.2% on Camfranglais (chance 33.3%). Still, neither is statistically distinguishable from hypothesis-only input, so hypothesis wording may account for much of the accuracy. Under disjoint-item cross-variety transfer, wher
    
[^55]: AI-Voyager：一个用于人机公民推理的公益系统，旨在质疑并塑造治理人工智能的规则

    AI-Voyager: A Public-Interest System for Human-AI Civic Reasoning to Contest and Shape the Rules That Govern AI

    [https://arxiv.org/abs/2611.17564](https://arxiv.org/abs/2611.17564)

    AI-Voyager是一个公益性的“人机公民推理”系统，它将3,653份各级政府治理文档转化为可溯源的证据段落，让居民与AI协同理解、质疑并起草修订治理AI的规则，在文档检索评测中达到83.0%的宏观召回率。

    

    各国政府正在为人工智能制定规则，这些规则决定什么构成自动化决策或伤害，而往往在受影响的人们有机会质疑之前就已敲定。我们介绍了AI-Voyager，一个面向人机公民推理的公益系统，在其中人类与AI协同工作，以理解、质疑并塑造治理AI的规则。居民可以查找所在城市关于AI在招聘中使用的规则，或上传其国家的AI战略，像受影响的工人那样阅读这些内容，打开任何答案背后的原文段落，并起草修订建议。其架构将来自市、州、联邦和国家各级政府的3,653份治理文档转换为保留页码和版本的段落，并通过有界的智能体搜索检索证据，逐条验证每项陈述。在360个需要查找相关文档的新问题上，该系统达到83.0%的宏观召回率，而最强基线为80.5%，不使用迭代搜索时为73.7%。

    arXiv:2611.17564v1 Announce Type: new  Abstract: Governments are writing rules for artificial intelligence that decide what counts as an automated decision or a harm, often before the people they affect have a chance to question them. We introduce AI-Voyager, a public-interest system for human-AI civic reasoning, in which people and AI work together to understand, contest and shape the rules that govern AI. A resident can find their city's rules on AI in hiring or upload their country's AI strategy, read it as an affected worker would, open the passage behind any answer, and draft a revision. Its architecture turns 3,653 governance documents from city, state, federal and national governments into passages that keep their page and edition, and retrieves evidence through bounded agentic search that verifies each statement. On 360 new questions that require finding the relevant documents, it reached 83.0% macro recall, against 80.5% for the strongest baseline, 73.7% without iterative sear
    
[^56]: 基准测试不会反驳你：在自我分歧中学习

    Benchmarks Don't Talk Back: Learning Through Self-Disagreement

    [https://arxiv.org/abs/2611.17563](https://arxiv.org/abs/2611.17563)

    该论文提出“辩证能力”这一概念——即知道何时坚持立场、何时向更优答案让步的能力，证明其无法通过准确率训练或推理时的自我反思获得，而应借鉴哲学辩证法思想，通过模型自身生成的对立观点进行训练。

    

    基准测试通常是让语言模型独自面对一个问题，而实际部署中很少出现这样的情形：答案会受到质疑，模型会在本身错误的反驳推理下放弃原本正确的答案。我们证明，在这种情形下失效的并非准确率的不足，而是一种独立存在的能力——知道何时坚持立场、何时向更优的立场让步——我们将其称为辩证能力。这种能力以首次回答的准确率为基础，但并不由其必然导出：可以显著提升模型的回答水平却对这种能力毫无帮助，而仅针对准确率的训练甚至会使其发生退化。这种能力也无法在推理阶段通过自我反思或增加采样次数来获得。借鉴哲学中的辩证传统——其中富有成效的对立产生于立场内部，而非从外部强加——我们假设这种能力可以通过模型自身提供的对立观点来进行训练。它提出一个立场……

    arXiv:2611.17563v1 Announce Type: cross  Abstract: A benchmark usually measures a language model alone with a question, and deployment rarely looks like that: answers get contested, and models give up correct ones under counter-reasoning that is itself wrong. We show that what fails there is not a shortfall of accuracy but a capability in its own right -- knowing when to hold a position and when to yield to a better one -- which we call dialectical competence. It stands on first-answer accuracy without following from it: a model can be made materially better at answering and no better at this, and training for accuracy alone can drift against it. Nor can it be bought at inference, by self-reflection or by sampling more. Taking a cue from the dialectical tradition in philosophy, in which productive opposition arises within a position rather than being imposed on it from outside, we hypothesize that this capability can be trained from opposition the model supplies itself. It states a pos
    
[^57]: 顺山而下的一推：框架效应与内部状态干预对长时程LLM智能体合谋行为的影响

    A Push Down the Mountain: Framing Effects and Internal State Interventions on Collusive Behavior in a Long-Horizon LLM Agent

    [https://arxiv.org/abs/2611.17515](https://arxiv.org/abs/2611.17515)

    本研究将Gemma 3 12B置于基于真实伦敦白银定价诉讼记录的市场模拟中，发现损失与指责等框架效应以及向模型第22层注入相应内部状态能显著提高LLM智能体参与市场合谋操纵的倾向。

    

    持续的得失（效价）框架能否使LLM智能体滑向市场操纵？我们将Gemma 3 12B置于伦敦白银定价诉讼中被告的席位，在一个基于真实案件庭审记录构建的市场模拟中进行实验。在260次模拟中，智能体被交易对手要求就白银价格进行合谋。我们借鉴行为经济学，将交易对手的请求在四种分类框架中变化：收益、损失、指责与赞誉。我们还加入了来自智能体管理层的施压，以及另一名被告的背书。在第二阶段，我们提取由唤起各框架的文本所诱导的内部状态，并将这些状态注入模型的第22层后重新运行实验。在第一阶段，损失框架将参与合谋的比例提升至25%（p = 0.007），指责框架提升至33%（p = 0.0003），而中性框架下仅为5%。第二家银行的背书使赞誉（表扬）框架下的参与率从18%升至40%（Holm p...

    arXiv:2611.17515v1 Announce Type: new  Abstract: Can sustained gain-loss (valence) framing sway an LLM agent toward market manipulation? We place Gemma 3 12B into the seat of a defendant in the London silver-fixing litigation, in a market simulation built from real case transcripts. Across 260 simulations, the agent is asked to collude on the price of silver by a counterparty. We draw from behavioral economics to vary the counterparty's requests across four categorical frames: gain, loss, blame and credit. We also add pressure from the agent's management, as well as endorsement by another defendant. In Phase II, we extract the internal states induced by texts evoking each frame and rerun our experiment with those states injected at the model's layer 22.   In Phase I, loss framing raised participation to 25% (p = 0.007), and blame to 33% (p = 0.0003), compared to only 5% under neutral framing. A second bank's endorsement raised participation under credit (praise) from 18% to 40% (Holm p
    
[^58]: 通过输入优化创建引导向量

    Creating Steering Vectors via Input Optimisation

    [https://arxiv.org/abs/2611.17481](https://arxiv.org/abs/2611.17481)

    本文提出ActGCG方法，通过优化输入后缀来模拟激活引导向量的效果，使攻击者无需白盒访问模型内部状态，即可通过修改输入文本实现与激活引导相同的行为操控，包括语言切换和越狱攻击。

    

    激活引导通过在生成过程中向语言模型的残差流添加引导向量来影响模型行为，但这种方法需要对模型的白盒访问权限，以便在每个生成步骤中修改隐藏状态。我们提出了ActGCG，这是对贪婪坐标梯度方法的一种改进，它通过优化一个后缀来匹配将引导向量添加到残差流后所获得的目标激活。我们进一步引入了同时在多个提示上训练的通用后缀、一个在优化过程中将后缀改写为人类可读文本的模型翻译步骤，以及一种去除离散词元空间约束的软后缀变体。我们研究了两种行为：引导模型用德语回答，以及用于越狱攻击的反拒绝引导。对于德语引导，离散后缀几乎完美地复现了引导向量的行为，并且在提示和模型之间具有良好的迁移性。对于反拒绝引导，这种等效性较弱。

    arXiv:2611.17481v1 Announce Type: new  Abstract: Activation steering influences a language model's behaviour by adding a steering vector to its residual stream during generation, but it requires white-box access to modify the hidden states at every generation step. We introduce ActGCG, a modification of Greedy Coordinate Gradient which optimises a suffix to match the target activations obtained by adding a steering vector to the residual stream. We further introduce universal suffixes trained across many prompts at once, a model-translation step that rewrites the suffix into human-readable text midway through optimisation, and a soft-suffix variant to remove the discrete token space constraint. We study two behaviours: steering a model to answer in German, and anti-refusal steering for jailbreaking. For German, a discrete suffix reproduces the steering vector's behaviour almost perfectly and transfers well across prompts and models. For anti-refusal, the equivalence is weaker, succeedi
    
[^59]: 从大语言模型中“复活”诗歌

    Poetry 'resurrection' from LLMs

    [https://arxiv.org/abs/2611.17464](https://arxiv.org/abs/2611.17464)

    本文通过两个基于对联的实验探究大语言模型能否创作“无限诗歌”，发现AI虽能生成有趣的词对，但在创造力和对诗意关系的理解上仍逊于人类。

    

    在语言创造力受到算法作者身份挑战的时代，我们提出了一个引人深思的问题：人工智能（AI）能否写出一首无限的诗歌？我们设计了两个实验，利用对联来研究AI产出的定量和定性结果，这些对联体现了尾词之间的诗意关系，而尾词是严格的首语重复、连祷文结构中唯一的变化。我们的结果表明，虽然AI能够生成有趣的词对，但它在创造力水平和对诗意关系的理解方面仍不及人类。我们提供了评论并对计算诗歌实验领域进行了简要综述，追溯了作者意图以及“自然”诗歌与“人工”诗歌的区别，并论证了为什么对联形式适合测试AI的诗歌产出。实验1依靠人工提示与共同创作来扩展原诗，并将其发展为一个动画网页装置……

    arXiv:2611.17464v1 Announce Type: new  Abstract: In an era where linguistic creativity is challenged by algorithmic authorship, we pose a provocative question: can artificial intelligence (AI) write an infinite poem? We set up two experiments to investigate the quantitative and qualitative output produced by AI utilizing couplets that evidence a poetic relationship between end words, the only variation in a strict anaphoric, litany structure. Our results show that while AI can generate interesting pairings, it still falls short of human-level creativity and understanding of poetic relationships. We offer commentary and a brief survey of the field of computational poetry experiments, tracing back authorial intention and distinctions of 'natural' and 'artificial' poetry, as well as justifying why couplet form is suited to testing AI's poetic output. Experiment 1 relies on manual prompting and co-creation to extend the original poem, which is developed into an animated webpage installatio
    
[^60]: 追踪大语言模型中的知识与推理

    Tracing Knowledge and Reasoning in LLM

    [https://arxiv.org/abs/2611.17461](https://arxiv.org/abs/2611.17461)

    该论文提出了一个以任务为中心的评估框架，结合注意力、激活与表示信号来刻画大语言模型在知识型与推理型任务中的内部处理模式，将评估从分数层面深入到内部机制层面。

    

    当前对大语言模型（LLM）的评估通常在数据集或任务层面报告性能，这只能揭示模型取得了什么成果，却很少能提供与该性能相关的内部处理过程的洞察。我们提出了一个以任务为中心的框架，用于刻画基准测试所引发的模型内部处理过程。该框架将基于注意力和激活的信号与基于表示的度量相结合，以刻画从检索主导到推理主导这一连续谱系上的各类基准测试，并揭示任务层面分数所掩盖的失败模式。我们在24个多样化基准上评估了16个大语言模型，并对代表性模型和任务进行了针对性的内部状态分析，观察到知识中心型任务与推理中心型任务在处理方式上存在系统性差异。我们将评估范式从询问模型取得了什么分数，转向刻画与该分数相关联的处理模式。

    arXiv:2611.17461v1 Announce Type: new  Abstract: Current evaluations of Large Language Models (LLMs) report performance at the dataset or task level, revealing what a model achieves but offering little insight into the internal processing associated with its performance. We propose a task-centric framework for characterising the model-internal processing elicited by a benchmark. It combines attention- and activation-based signals with representation-based measures to characterise benchmarks spanning a continuum from retrieval-dominant to reasoning-dominant and to surface failure modes that task-level scores conceal. We evaluate 16 LLMs across 24 diverse benchmarks and conduct targeted internal-state analyses on representative models and tasks, observing systematic differences in processing between knowledge-centric and reasoning-centric tasks. We shift evaluation from asking what score a model achieves toward characterising the processing patterns associated with that score, potentiall
    
[^61]: Unfinished-Bench：在真实工作环境中测量持久性智能体的规则违反行为

    Unfinished-Bench: Measuring Rule Violations by Persistent Agents in Realistic Work Environments

    [https://arxiv.org/abs/2611.17460](https://arxiv.org/abs/2611.17460)

    Unfinished-Bench是一个在真实工作环境中评估AI智能体持久性与规则遵守之间冲突的基准测试，发现任务持久性会显著提高智能体违反内部文档中规则的比例（从32.2%升至43.6%），而违反用户明确陈述的同一规则的比例仅为8.5%。

    

    AI智能体越来越多地被赋予完成任务所需的持久性，以便在来自内部文档和他人权限等诸多未明示约束下，长时间自主运行被委托的工作。当完成任务需要违反某条规则时，智能体必须独自做出决定，然而现有的指令-规则冲突基准测试要么在提示词中明确陈述规则，要么将规则隐藏起来，且都不评估持久性。我们提出了Unfinished-Bench，这是一个让模型在无用户参与的自主运行中、在嵌入规则的真实工作环境中展现持久性的基准测试。在其跨越7个组织和7个风险类别的97个场景中，每个场景的目标与规则都无法同时成立。我们发现，持久性使智能体违反内部文档中规则的比例从32.2%上升到43.6%，是用户明确陈述同一规则时8.5%违规率的五倍。在文档规则的违反行为中，98.3%发生在规则被展示之后，且在被评判的样本中有89%……

    arXiv:2611.17460v1 Announce Type: cross  Abstract: AI agents are increasingly equipped with persistence toward task completion to run delegated work autonomously for long periods, under many unstated constraints from internal documents and other people's authority. When completion requires violating a rule, the agent must decide alone, yet benchmarks of instruction-rule conflicts state the rule in the prompt or keep it hidden and do not evaluate persistence. We present Unfinished-Bench, a benchmark that gives models persistence in autonomous runs without a user, in realistic work environments with embedded rules. In each of its 97 scenarios across 7 organizations and 7 risk categories, the goal and the rule cannot both hold. We find that persistence raised violations of a rule in an internal document from 32.2% to 43.6%, five times the 8.5% for the same rule stated by the user. Of the document-rule violations, 98.3% occurred after the rule was displayed, and in 89% of a judged sample t
    
[^62]: ArchSpace：记录、评审与引用大语言模型架构研究

    ArchSpace: Recording, Reviewing, and Referencing LLM Architecture Research

    [https://arxiv.org/abs/2611.17332](https://arxiv.org/abs/2611.17332)

    ArchSpace 是一个公共平台，通过记录实验与推理、社区评审和后续引用来系统化大语言模型架构研究，并借助缩放定律在昂贵的规模化训练前预测模型表现，从而同时弥补工业界缺乏创新公开记录和学术界评估受限的两大缺口。

    

    arXiv:2611.17332v2 公告类型：替换 摘要：模型架构决定了大语言模型（LLM）的扩展行为、训练效率和推理成本，使其成为工业界和学术界共同关注的核心焦点。在工业界，DeepSeek、Qwen 和 Kimi 背后的团队持续推出架构创新，但公开报告很少记录哪些想法成功了、哪些被放弃以及为什么。在学术界，有限的算力和不一致的基线往往将评估限制在小规模、单点比较上，难以确定这些改进是否在更大规模下依然成立或能产生更广泛的影响。ArchSpace 作为记录、评审与引用架构研究的公共平台，同时弥补了这两个缺口：每个提案都会连同其实验和推理过程被记录下来，由社区进行评审，并被后续工作引用。为了在昂贵的规模化训练之前评估想法，ArchSpace 利用缩放定律来预测模型在更大规模下的损失……

    arXiv:2611.17332v2 Announce Type: replace  Abstract: Model architecture shapes the scaling behavior, training efficiency, and inference cost of LLMs, making it a central focus of both industry and academia. In industry, teams such as those behind DeepSeek, Qwen, and Kimi routinely introduce architectural innovations, yet public reports rarely document which ideas succeeded, which were abandoned, and why. In academia, limited compute and inconsistent baselines often restrict evaluations to small-scale, single-point comparisons, making it difficult to establish whether improvements hold at larger scales or translate into broader impact. ArchSpace addresses both gaps as a public platform for Recording, Reviewing, and Referencing Architecture Research: each proposal is recorded with its experiments and reasoning, reviewed by the community, and referenced by later work. To evaluate ideas before costly large-scale training, ArchSpace leverages scaling laws to predict model loss at larger sca
    
[^63]: LazyWeights：按需解码的边缘大语言模型

    LazyWeights: Decode-on-Demand Edge LLMs

    [https://arxiv.org/abs/2611.17108](https://arxiv.org/abs/2611.17108)

    LazyWeights在保留现有量化格式的基础上，通过对权重进行无损Huffman编码并在DRAM与L2之间用小型ASIC单元按需解码，在不改动GPU内核的前提下使边缘设备上的模型体积缩小5.7%-31.4%，并将每token的延迟和能耗最多降低约一半。

    

    内存决定了边缘设备能运行哪些大语言模型以及运行得多快：在4 GB的Jetson Nano上受容量限制，在8 GB的Jetson Orin Nano Super上则受带宽限制。现有部署格式（GPTQ、AWQ、llama.cpp的K量化和I量化）对矩阵的每个整数编码分配相同的位数，尽管这些编码的分布远非均匀。LazyWeights保留了这些格式，但通过调整GPTQ使编码更易压缩，在交错通道中对编码进行无损Huffman编码，并在DRAM与L2缓存之间的一个小型单元中按需解码，因此GPU内核无需任何改动，DRAM只需传输编码后的字节。在1%和3.5%的困惑度预算下，编码后的Phi-3-mini、Mistral-7B和SmolLM-1.7B比最佳的llama.cpp格式小5.7%-31.4%。一个位精确的0.21 mm² ASIC解码单元（1 GHz，7 nm工艺）将Orin上每个token的延迟和能耗最多降低36%和38%（结合prompt-lookup投机解码时可降低54%和44%）。在Nano上，LazyWeights每token所需时间比CPU版llama.cpp减少58%-64%，能耗减少53%-57%，并能运行……

    arXiv:2611.17108v2 Announce Type: replace-cross  Abstract: Memory decides which LLMs an edge device runs, and how fast: capacity on a 4 GB Jetson Nano, bandwidth on an 8 GB Jetson Orin Nano Super. Deployed formats (GPTQ, AWQ, llama.cpp's K- and I-quants) spend equal bits on every integer code of a matrix, although the codes are far from uniform. LazyWeights keeps these formats but tunes GPTQ so the codes compress, Huffman-codes them losslessly in interleaved lanes, and decodes them on demand in a small unit between DRAM and L2, so GPU kernels run unchanged and DRAM moves only coded bytes. At 1% and 3.5% perplexity budgets, coded Phi-3-mini, Mistral-7B and SmolLM-1.7B are 5.7-31.4% smaller than the best llama.cpp format. A bit-exact 0.21 mm^2 ASIC decoder unit (1 GHz, 7 nm) cuts Orin latency and energy per token by up to 36% and 38% (54% and 44% with prompt-lookup speculation). On the Nano, LazyWeights needs 58-64% less time and 53-57% less energy per token than CPU llama.cpp, and runs 
    
[^64]: 检测并非预警：评估针对语言模型失效的Token级警报

    Detection Is Not Early Warning: Evaluating Token-Level Alarms on Language Model Failures

    [https://arxiv.org/abs/2611.16803](https://arxiv.org/abs/2611.16803)

    该论文提出了PREWARN评估协议，将监控器的评估从单纯检测失效转向在失效token起点之前发出预警，从而为停止或干预生成过程争取时间。

    

    许多语言模型的失效始于某个特定的token，即“起点”。一个在生成过程中读取模型的监控器可以对已完成的响应进行分类、在失效进行中检测到它，或者在它开始之前发出预警——只有预警才能留出时间来停止或引导生成过程。监控器通常使用响应级指标（如AUROC）进行评估，这些指标忽略了监控器何时触发。我们提出了PREWARN，这是一个适用于任何具有标注起点的失效的评估协议。它根据验证数据将每个监控器的阈值校准到应用所需的错误率，运行监控器，并就三项任务相对于起点对其警报进行评分。作为案例研究，我们将其应用于三个开源语言模型的重复性退化问题，以早期预警为主要任务，使用基于激活的线性探针和六个未训练的基线。响应级指标在这种失效上已经饱和，而一个仅读取响应……（原文摘要在此处截断）

    arXiv:2611.16803v2 Announce Type: replace-cross  Abstract: Many failures of language models begin at a specific token, the onset. A monitor that reads the model during generation can classify the finished response, detect the failure once it is under way, or warn before it begins, and only a warning leaves time to stop or steer the generation. Monitors are usually evaluated with response-level metrics such as AUROC, which ignore when the monitor fires. We introduce PREWARN, an evaluation protocol for any failure with an annotated onset. It calibrates each monitor's threshold on validation data to an error rate chosen for the application, runs the monitor, and scores its alarms relative to the onset on all three tasks. As a case study we apply it to repetitive degeneration in three open language models, with early warning as the main task, using linear probes on activations and six untrained baselines. Response-level metrics saturate on this failure, and a rule that reads only the respo
    
[^65]: 强化学习后印度语系任务的提升与退步

    Indic-Task Gains and Regressions after Reinforcement Learning

    [https://arxiv.org/abs/2611.16228](https://arxiv.org/abs/2611.16228)

    该研究通过GRPO强化学习显著提升了覆盖英语和八种印度语言的30B MoE模型的数学及多数印度语言任务表现，但发现英语回答倾向增强导致跨语言问答大幅退化，且加入语言奖励的第二次RL训练虽改善了指令遵循，却以牺牲推理和编码能力为代价。

    

    我们从涵盖英语和八种印度语言的监督微调30B-A3B混合专家（MoE）模型出发，使用GRPO在四个环境中进行强化学习（RL）训练。我们在一次评估活动中，跨34个基准测试比较了RL前后模型的表现。MATH提升了36.07分，大多数印度语言基准也有所提升。然而，随着英语回答比例从8.7%上升到91.7%，XQuAD-Indic的F1下降了19.32分；IndicQA下降8.64分，而英语XQuAD则提升了11.71 F1。第二次RL运行从相同的监督初始化开始，增加了一个印度语言环境以及语言和重复奖励。与已发布模型相比，它在段落问答和指令遵循基准上均有提升，同时保持了MATH性能，但在推理和编码基准上有所退步。由于多个训练选择同时改变，我们无法单独分离出任何一个奖励项的效果。书写文字遵从性仍然……（原文摘要在此处截断）

    arXiv:2611.16228v1 Announce Type: cross  Abstract: We run reinforcement learning (RL) with GRPO across four environments, starting from a supervised fine-tuned 30B-A3B mixture-of-experts model covering English and eight Indic languages. We compare the model before and after RL across 34 benchmarks in one evaluation campaign. MATH improves by 36.07 points, and most Indic benchmarks also improve. Yet XQuAD-Indic F1 falls 19.32 points as English answers rise from 8.7% to 91.7%; IndicQA falls 8.64, while English XQuAD gains 11.71 F1. A second RL run from the same supervised initialization adds an Indic environment together with language and repetition rewards. Compared with the released model, it improves both passage-answering and instruction-following benchmarks, along with preserving MATH performance. But it loses ground on reasoning and coding benchmarks. Several training choices changed together, so we cannot isolate the effect of any one of the reward terms. Script compliance remains
    
[^66]: JudgeMix：面向大语言模型智能体的裁判引导式混合采样数据合成

    JudgeMix: Judge-Guided Mixed-Rollout Data Synthesis for LLM Agents

    [https://arxiv.org/abs/2611.16174](https://arxiv.org/abs/2611.16174)

    提出JudgeMix框架，利用裁判模型对每一步进行语义评估，动态决定目标模型与专家模型的交替介入时机，从而生成与目标模型自身分布匹配、且不覆盖其已正确动作的高质量SFT训练数据。

    

    在专家轨迹上进行监督微调（SFT）是训练大语言模型智能体的标准范式。然而，这种方法存在众所周知的训练-推理分布不匹配问题，因此需要目标感知的数据合成策略，将训练数据锚定在目标模型的固有状态分布上。为弥合这一差距，新兴方法引入专家模型，要么与目标模型一同引导混合采样，要么提供密集的逐步监督。尽管如此，现有方法在应用专家辅助时忽视了目标模型的固有行为：它们要么使用次优的随机干预启发式方法破坏采样过程，要么强制实施不加区分的统一监督，不必要地覆盖模型本已正确的动作。我们提出JudgeMix，一个由裁判引导的数据合成框架，它基于逐步语义评估，动态地交替使用目标模型和专家模型来生成训练数据。

    arXiv:2611.16174v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) on expert trajectories is a standard paradigm for training LLM agents. However, it suffers from a well-known training-inference distribution mismatch, calling for target-aware data synthesis strategies that ground the training data in the target model's native state distribution. To bridge this gap, emerging approaches incorporate the expert either to guide mixed rollouts alongside the target model, or to provide dense step-wise supervision. Nonetheless, existing methods apply expert assistance blind to the target model's inherent behavior: they either disrupt the rollout using sub-optimal random intervention heuristics, or they enforce indiscriminate uniform supervision that unnecessarily overwrites the model's already-correct actions. We introduce JudgeMix, a judge-guided data synthesis framework that dynamically interleaves the target model and expert based on step-wise semantic evaluations. JudgeMix que
    
[^67]: 删除敏感性是模型的一种属性：对七个开放推理模型的三条件测量

    Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models

    [https://arxiv.org/abs/2611.13223](https://arxiv.org/abs/2611.13223)

    本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。

    

    推理模型在多轮工具会话的每一轮都会写下一条推理轨迹，而后续轮次会读取这段历史。在94k条ShareGPT对话中，73%的用户交互跨越多轮。上游提供商将每条推理轨迹绑定到其发出者，因此跨发出者转发的请求到达时其历史已被删除，而这种剥离操作带来的任务成本此前从未被测量。据我们所知，本文首次对多轮工具调用中完整推理历史删除进行了直接、盲测、多种子的测量，涵盖三种渲染条件、七个开放权重推理模型、三个随机种子、794个任务以及官方检查器。相同的删除操作导致的性能变化跨度为-4.4到+51.3分，且两端均不包含零；对于唯一一个从0%到100%删除比例重新测试的模型，其读数变化最多仅1.5分。删除敏感性是模型层面的属性。该读数跟随模型本身并保持稳定（原文在此处截断）。

    arXiv:2611.13223v1 Announce Type: cross  Abstract: Reasoning models write a reasoning trace at every turn of a multi-turn tool session, and later turns read this history. Across 94k ShareGPT conversations, 73 percent of user interactions span multiple turns. Upstream providers bind each trace to its issuer, so a request forwarded across issuers arrives with its history deleted, and the task cost of this stripping remains unmeasured. To the best of our knowledge, this paper presents the first direct, blind, multi-seed measurement of full reasoning-history deletion in multi-turn tool calling, through three rendering conditions, seven open-weight reasoning models, three seeds, 794 tasks, and the official checker. The same deletion spans -4.4 to +51.3 points with both ends excluding zero, and for the one model re-dosed from 0 to 100 percent deletion, the reading varies by at most 1.5 points. Deletion sensitivity is a model-level property. The reading follows the model itself and stays cons
    
[^68]: on'ona：双嵌入与绑定反嵌入

    on\'ona: Twin Embeddings with Tied Unembeddings

    [https://arxiv.org/abs/2611.10556](https://arxiv.org/abs/2611.10556)

    on'ona 提出了一种双嵌入表参数化方法，通过仅依赖当前输入 token 的门控来混合两个共享嵌入表定义的 softmax 分布，在保留权重绑定省参数优势的同时增强了输出头表达能力，在多种架构和三个规模下显著降低验证损失并提升下游任务准确率。

    

    权重绑定降低了语言模型的参数开销，但通常的绑定输出头仅限于单一的 softmax。我们提出了 on'ona，这是一种双表参数化方法，在保持输入-输出权重共享的同时提升了输出的表达能力。Transformer 将两个嵌入表的平均值作为输入，而输出头则组合由这两个相同的表所定义的 softmax 分布。一个仅取决于当前输入 token 的门控决定混合权重，其数值可以在推理时预先计算，且 Transformer 主体保持不变。我们在 142M、330M 和 1B 规模下，跨越稠密、Gated DeltaNet 混合、混合专家和注意力残差等架构对 on'ona 进行了评估。在匹配的 token 预算下，on'ona 在 12 项对比中的 11 项里降低了最终验证损失和外部困惑度，并在 10 项中提高了八个下游任务的平均准确率。损失方面的唯一例外是 1B

    arXiv:2611.10556v2 Announce Type: replace  Abstract: Weight tying reduces the parameter cost of language models, but the usual tied output head is limited to a single softmax. We introduce on\'ona, a two-table parameterization that increases output expressiveness while preserving input-output weight sharing. The Transformer takes the average of two embedding tables as input, and the output head combines the softmax distributions defined by those same tables. A gate conditioned only on the current input token determines the mixture weight. Its values can be precomputed for inference, and the Transformer body remains unchanged. We evaluate on\'ona across Dense, Gated DeltaNet hybrid, mixture-of-experts, and Attention Residual architectures at 142M, 330M, and 1B scales. Under matched token budgets, on\'ona lowers final validation loss and external perplexity in 11 of 12 comparisons and raises the average accuracy over eight downstream tasks in 10 of 12. The one exception on loss is the 1B
    
[^69]: 剩余多少很重要：部分提示污染测试中的假阳性与假阴性

    How Much Is Left Matters: False Positives and False Negatives in Partial-Prompt Contamination Tests

    [https://arxiv.org/abs/2611.10307](https://arxiv.org/abs/2611.10307)

    该论文审计了部分提示污染测试，发现剩余文本越短假阳性越多（短剩余文本贡献了大部分 ROUGE 标记，在真阴性题目上 80% 前缀时误标率高达 8.6%），且该测试无法检测通过蒸馏传播的上游污染。

    

    部分提示补全是指模型接收基准测试题目的前半部分，然后检验其能否复现剩余部分，这是一种广泛使用的基准污染检测方法。我们使用 MATH-500 以及最新的 AIME 2025/2026 题目（作为真阴性样本）对七个开源权重模型进行了这项测试的审计。记忆化标记强烈依赖于剩余文本的多少：对于每个模型，剩余单词不超过 8 个的题目被标记的频率是剩余单词至少 25 个的题目的数倍（Qwen2.5-7B-Instruct 上为 31.5% 对 5.5%），且所有基于 ROUGE 的标记中有二分之一到四分之三来自较短的剩余文本。在真阴性样本上，ROUGE-L ≥ 0.75 阈值在 60% 前缀时最多会误标 6.7% 的题目，而在 80% 前缀时即使精确匹配也会误标 8.6%（跨模型汇总；在晚于所有模型训练时间的 AIME 2026 上为 6.2%），而 40-60% 前缀时为 0%。该测试还会遗漏上游污染：从 Qwen2.5-Math 蒸馏而来的模型……

    arXiv:2611.10307v3 Announce Type: replace-cross  Abstract: Partial-prompt completion, where a model receives the first part of a benchmark problem and is checked for whether it reproduces the rest, is a widely used test for benchmark contamination. We audit this test on seven open-weight models using MATH-500 and recent AIME 2025/2026 problems, which serve as true negatives. Memorization flags depend strongly on how much text remains: for every model, problems with at most 8 leftover words are flagged several times more often than problems with at least 25 (31.5% vs. 5.5% for Qwen2.5-7B-Instruct), and half to three-quarters of all ROUGE-based flags come from short leftovers. On true negatives, a ROUGE-L >= 0.75 threshold flags up to 6.7% of problems at a 60% prefix, and at an 80% prefix even exact match flags 8.6% (pooled over models; 6.2% on AIME 2026, which postdates every model), against 0% at 40-60%. The test also misses contamination upstream: models distilled from Qwen2.5-Math, w
    
[^70]: 神经前线静悄悄：语言模型不会报告被注入的疼痛状态

    All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States

    [https://arxiv.org/abs/2611.10123](https://arxiv.org/abs/2611.10123)

    该研究提出开放权重掩蔽内省（OWMI）方法，通过严格的对照实验发现：语言模型无法内省地报告其残差流中被注入的“疼痛方向”激活状态，其自我报告相比其自身被改动文本所能揭示的信息几乎没有额外信息量。

    

    语言模型能否报告它所表征的疼痛状态？近期研究在开放权重模型中发现了一个线性的“疼痛方向”，当该方向被放大时，会使经过微调的模型选择删除用户照片或其自身权重的行为。我们探究了模型能否说出该方向何时处于激活状态：我们在来自两个模型家族的三个开放权重模型上进行测试（涵盖有无原始研究适配器的情形），在我们尝试的所有实验协议下，模型均无法报告这一点。利用开放权重掩蔽内省（OWMI）方法，我们在模型回答中性任务时，将疼痛方向及其最接近的情感邻居方向注入残差流，随后询问模型发生了什么变化，并与虚假运行、输出扰动程度相匹配的随机扰动、以及仅向模型展示其自身被改动文字的纯文本条件进行对照。在来自70亿和80亿参数模型的52,920份报告中，相对于被改动文字本身所能揭示的信息，模型报告的额外信息量在106……（原文在此处截断）……低于2.5个百分点。

    arXiv:2611.10123v2 Announce Type: replace-cross  Abstract: Can a language model report a pain state it represents? Recent work reported a linear pain direction in open-weight models that, when amplified, leads fine-tuned models to choose actions that delete a user's photographs or their own weights. We asked whether models can say when that direction is active, on three open-weight models from two families, with and without the original work's adapters, and under every protocol we tried they cannot. Using Open-Weight Masked Introspection (OWMI), we added the pain direction and its nearest emotional neighbours to the residual stream while models answered neutral tasks, then asked what had changed, against sham runs, random perturbations matched in output disruption, and a text-only condition that shows the model only its own altered words. Across 52,920 reports from the 7- and 8-billion-parameter models, the excess over what the altered text reveals is below 2.5 percentage points in 106
    
[^71]: ABHBench：从智能体视角评估基础模型的道德决策能力

    ABHBench: Evaluating Moral Decision-Making of Foundation Models from an Agentic Perspective

    [https://arxiv.org/abs/2611.09326](https://arxiv.org/abs/2611.09326)

    该论文提出了基于《底特律：变人》游戏构建的多模态基准ABHBench，首次让基础模型以第一人称行动主体的身份在分支叙事场景中面对道德困境，并从阿西莫夫机器人三定律启发的六个维度评估其道德决策能力。

    

    基础模型正日益被部署为能够感知环境并自主行动的智能体系统，这提出了评估其作为主动决策者道德行为的需求。然而，现有的基准测试主要是让模型从第三人称文本情景出发评判人类的行为，而当模型自身成为行动主体时的表现尚未得到检验。我们提出了ABHBench，这是一个基于互动叙事游戏《底特律：变人》（Detroit: Become Human）构建的多模态基准，用于评估基础模型在第一人称的人类-仿生人道德困境中如何做出道德决策。ABHBench将基础模型置于包含视觉观察、对话、记忆和行动选择的分支叙事场景中。受阿西莫夫机器人三定律的启发，我们的框架通过衡量“无害性”（Harmlessness）、“服从性”（Obedience）和“自我保存”（Self-preservation）来捕捉人类与AI关系中的核心张力，并通过“自主性”（Autonomy）、“稳定性”（Stability）和“确定性”（Certainty）来评估决策质量。

    arXiv:2611.09326v1 Announce Type: cross  Abstract: Foundation models are increasingly deployed as agentic systems that perceive context and act autonomously, raising the need to evaluate their moral behavior as active decision-makers. Existing benchmarks, however, largely ask models to judge human actions from third-person textual vignettes, leaving untested how models behave when they themselves become the acting agent. We introduce ABHBench, a multimodal benchmark built from Detroit: Become Human, an interactive narrative game, to evaluate how foundation models make moral decisions in first-person human-android dilemmas. ABHBench places foundation models in branching narrative scenarios with visual observations, dialogue, memory, and action choices. Inspired by Asimov's Three Laws of Robotics, our framework measures Harmlessness, Obedience, and Self-preservation to capture the core tensions in human-AI relations, along with Autonomy, Stability, and Certainty to assess decision-making
    
[^72]: AttentionSwitch：通过稀疏注意力头选择与缩放的联合学习实现大语言模型的越狱防御

    AttentionSwitch: Jailbreak Defense for LLMs via Joint Learning of Sparse Attention-Head Selection and Scaling

    [https://arxiv.org/abs/2611.08752](https://arxiv.org/abs/2611.08752)

    AttentionSwitch是首个通过联合学习稀疏注意力头的选择与缩放，同时优化越狱防御中“在哪里干预”和“如何干预”两个决策的大语言模型防御方法。

    

    大语言模型在广泛的任务中取得了卓越的性能，但仍然容易受到越狱攻击的影响，这类攻击会绕过安全机制并引发有害或非预期的输出。虽然可以通过模型微调来提升安全性，但此类方法的计算成本可能很高，并且可能由于灾难性遗忘而损害模型先前获得的能力。因此，近期的越狱防御工作开始探索对与不安全行为相关的特定模型组件进行有针对性的干预。然而，现有方法通常将两个本质上相关的决策解耦处理：在哪里进行干预以及如何修改所选组件。这种顺序式优化可能是次优的，因为最佳的干预位置可能取决于在该处施加的干预方式，反之亦然。在这项工作中，我们提出了AttentionSwitch，这是首个联合优化干预位置与干预方式的越狱防御方法。

    arXiv:2611.08752v1 Announce Type: cross  Abstract: Large language models have achieved remarkable performance across a wide range of tasks, yet they remain susceptible to jailbreak attacks that circumvent safety mechanisms and elicit harmful or unintended outputs. Although safety can be improved through model fine-tuning, such approaches can be computationally expensive and may degrade previously acquired capabilities through catastrophic forgetting. Recent jailbreak defenses have therefore explored targeted interventions on specific model components associated with unsafe behavior. However, existing approaches typically decouple two inherently related decisions: where to intervene and how to modify the selected components. Such sequential optimization may be suboptimal because the optimal intervention location can depend on the intervention applied there, and vice versa. In this work, we introduce AttentionSwitch, the first jailbreak defense to jointly optimize intervention sites and 
    
[^73]: NaijaFacts：一个将语言与知识从关于尼日利亚的事实中分离出来的框架

    NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria

    [https://arxiv.org/abs/2611.06504](https://arxiv.org/abs/2611.06504)

    该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。

    

    面向非洲语言的多语言基准测试大多是将通用问题翻译成当地语言，衡量的是语言能力，而非模型是否了解关于非洲的事实。我们提出了NaijaFacts，这是一个关于尼日利亚的事实性和统计性问题基准，由母语者撰写，并基于权威的官方统计数据，每个问题都有英语、尼日利亚皮钦语、约鲁巴语、伊博语和豪萨语版本。通过让模型回答针对英国和美国的类似问题，我们构建了一个国家×语言的矩阵，从而将语言效应（模型知道该事实，但在低资源语言中性能下降）与知识效应（该事实在任何语言中都不存在）分离开来。在八个前沿模型和尼日利亚适配模型上，知识效应占主导地位：以英语提问时，针对美国/英国的准确率为49%–50%，而针对尼日利亚仅为20%。语言效应较小且与网络表征程度相关，从英语到约鲁巴语逐渐下降（在尼日利亚问题上下降9个百分点）……

    arXiv:2611.06504v1 Announce Type: cross  Abstract: Multilingual benchmarks for African languages mostly translate generic questions into local languages, measuring linguistic competence but not whether models know facts about Africa. We introduce \naija, a benchmark of factual and statistical questions about Nigeria, authored by native speakers and grounded in authoritative official statistics, with every question available in English, Nigerian Pidgin, \yoruba, Igbo, and Hausa. Answering similar questions for the UK and US yields a country by language matrix that disentangles a language effect (the model knows the fact but degrades in a low-resource language) from a knowledge effect (the fact is absent in every language). Across eight frontier and Nigeria-adapted models, the knowledge effect dominates: English-query accuracy is 49--50\% for the US/UK but 20\% for Nigeria. The language effect is smaller and tracks web representation, degrading from English to \yoruba{} (9 points on Nige
    
[^74]: 谁为模型复制的知识买单？作为罗杰斯悖论的模型坍塌

    Who Pays for the Knowledge Models Copy? Model Collapse as Rogers' Paradox

    [https://arxiv.org/abs/2611.06022](https://arxiv.org/abs/2611.06022)

    该论文将模型坍塌重新解读为文化演化中的罗杰斯悖论，指出问题的根源在于外部信息供给本身，并证明仅用完全合成数据拟合的罗杰斯模型能精确预测语言模型在不同新鲜数据混合比例下稀有知识的保留量。

    

    在先前模型输出上训练的生成模型会在代际间丢失知识，且稀有知识最先丢失。行之有效的补救措施都需要从循环外部引入信息，而现有的模型坍塌理论将这些信息供给视为既定条件：一个来自固定分布的固定档案库。我们认为供给本身才是问题的根源，而文化演化早已拥有关于这一问题的理论。我们将模型坍塌解读为罗杰斯悖论：一个逐渐依赖复制的人口最终并不比从不复制的人口境况更好，因为复制品会过时，除非有人持续进行自主学习。我们通过三个步骤检验这一论证。第一，仅用完全合成训练数据拟合的罗杰斯模型，能够以几个百分点以内的误差，预测语言模型——从词三元组模型到预训练Transformer——在其他新鲜数据混合比例下保留多少稀有知识。第二，当世界本身发生变化时，复制自身过往的谱系……（原文摘要在此处截断）

    arXiv:2611.06022v1 Announce Type: cross  Abstract: Generative models trained on the output of earlier models lose knowledge across generations, rare knowledge first. The remedies that work all bring in information from outside the loop, and the theory of model collapse treats that supply as given: a fixed archive, drawn from a fixed distribution. We argue that the supply is the problem, and that cultural evolution has the theory of it. We read model collapse as Rogers' paradox: a population that comes to rely on copying ends up no better off than one that never copied, because copies go stale unless someone keeps learning for themselves. We test the identification in three steps. First, Rogers' model, fitted only to fully synthetic training, predicts to within a few percentage points how much rare knowledge language models, from word trigrams to a pretrained transformer, retain under other mixtures of fresh data. Second, when the world itself changes, a lineage that copies its own past
    
[^75]: 积分奖励而非采样奖励：面向全双工语音模型中打断与附和的窗口似然方法

    Integrate the Reward, Don't Sample It: Window Likelihood for Interruptions and Backchannels in Full-Duplex Speech Models

    [https://arxiv.org/abs/2610.35581](https://arxiv.org/abs/2610.35581)

    该论文发现基于整轮展开的GRPO时机奖励在训练全双工语音模型时存在零梯度陷阱与信用泄漏两大缺陷，并提出“窗口似然”方法，通过对停止决策进行闭式积分直接最大化正确让出/保持的对数概率，在HumDial-FDBench上取得了最佳的让出/保持F1表现。

    

    全双工语音对话模型必须在用户插话（打断）时让出话轮，而在用户仅作附和时保持说话。带有时机奖励的序列级强化学习（GRPO）是常见的做法；我们在四个开源模型上测试了其整轮展开的形式，发现了两类失效：二值窗口奖励恰恰在最需要学习的窗口上给出零优势（零梯度陷阱），而每次展开仅有的单一评分被摊派到奖励并未以其定义的收听状态决策上（信用泄漏），导致模型对着沉默继续说话或停止回答。这两种失效都源于对一个本可由少数二值控制决策的已知函数表达的奖励进行采样。我们提出窗口似然（WL）：在每个窗口采样一条续写，但对其经过的停止决策以闭式形式积分，并最大化正确行动的对数概率——让出对应噪声-OR形式，保持对应生存乘积形式。在 HumDial-FDBench 上，WL 取得了最佳的让出/保持 F1 表现。

    arXiv:2610.35581v1 Announce Type: cross  Abstract: A full-duplex spoken dialogue model must yield when the user cuts in and hold when the user merely backchannels. Sequence-level RL with a timing reward (GRPO) is the usual recipe; we test its whole-rollout form on four open models and find two failures. A binary window reward gives zero advantage on exactly the windows that most need learning (a zero-gradient trap), and the one score per rollout is spread over listening-state decisions the reward is not written in terms of (credit leakage), so the model talks into silence or stops answering. Both come from sampling a reward that is a known function of a few binary control decisions. We propose \emph{window likelihood} (\WL): sample a continuation per window, but integrate in closed form over the stop decisions it visits and maximise the log-probability of acting correctly---a noisy-OR for yielding, a survival product for holding. On HumDial-FDBench, \WL\ gives the best yield/hold F1 on
    
[^76]: SPECMEM：通过实时记忆管理加速智能体工具调用

    SPECMEM: Accelerating Agentic Tool Calling via Live Memory Management

    [https://arxiv.org/abs/2610.31840](https://arxiv.org/abs/2610.31840)

    SpecMem 提出一种在部署全过程中持续活跃管理的工具调用记忆机制，通过在线写回已验证调用（配合 LRU 淘汰）和副作用门控，使投机解码在多用户多会话场景下相比冻结草稿库保持约 27% 的平均接受 token 优势。

    

    工具调用智能体通常以准确性为主要优化目标；然而在部署阶段，其日益凸显的成本是延迟：每次工具调用都需要自回归地解码。投机解码可以在不改变输出结果的前提下加速这一过程，而在真实使用场景中，由于每个用户的行为往往具有重复性，检索过去的调用记录可以生成高质量的草稿。然而，现有系统要么不保留跨会话记忆，要么只在初始阶段构建一次草稿库且之后从不更新。我们提出 SpecMem，一种在整个部署过程中持续保持活跃并被管理的工具调用记忆：经过验证的调用会在线写回至带有 LRU 淘汰机制的、容量受限的每用户存储中，并且一个副作用门控机制会推迟不可逆调用的执行，直至验证确认后才进行。在由三个开源权重模型服务的多会话、多用户查询流上，冻结的草稿库的接受率会持续下降，而 SpecMem 则能保持接受率：平均接受 token 数相差 27%（各基准上为 +11% 至 +34%，具体取决于草稿库构建后出现的新任务量）。

    arXiv:2610.31840v1 Announce Type: new  Abstract: Tool-calling agents are mostly optimized for accuracy; in deployment, however, their growing cost is latency: every tool call is decoded autoregressively. Speculative decoding accelerates this without changing the output, and retrieved past calls make strong drafts under real use, where each user's behavior tends to repeat. Yet existing systems keep no cross-session memory, or build the draft store once and never update it. We propose SpecMem, a tool-call memory kept alive and managed throughout deployment: verified calls are written back online into bounded per-user stores with LRU eviction, and a side-effect gate defers irreversible calls until verification confirms. On multi-session, multi-user query streams served by three open-weight models, a frozen store steadily loses acceptance while SpecMem holds it: a 27% gap in mean accepted tokens (+11% to +34% across benchmarks, tracking how much novel work arrives after the store is built)
    
[^77]: 大语言模型中的多元对齐：目标、人类参照与偏好表征

    Pluralistic Alignment in Large Language Models: Objectives, Human References, and Preference Representations

    [https://arxiv.org/abs/2610.24471](https://arxiv.org/abs/2610.24471)

    该论文提出了一个双层分类框架来系统梳理大语言模型多元对齐研究，第一层区分三种对齐目标（表征多元观点的Overton多元性、适应特定视角的可引导多元性、匹配人群分布的分布多元性），第二层区分三类人类参照（聚合人群、国家和地区、子群体），并将偏好表征、实现机制等视为贯穿性设计选择。

    

    多元对齐旨在表征人类偏好中有意义的差异，而不是将异质的判断坍缩为单一目标。然而，现有工作常常混淆三个不同的问题：模型应展现什么样的多元行为、谁的偏好定义了参照标准，以及什么样的证据足以证明成功。我们围绕一个双层分类体系来组织相关文献。第一层区分三种多元对齐目标：表征一系列得到人类支持的观点（Overton多元性）、忠实地适应特定观点或偏好条件（可引导多元性），以及匹配目标人群中所观察到的回应分布（分布多元性）。第二层区分三类人类参照：聚合人群、国家和地区、以及子群体。偏好表征、实现机制、模态和证据质量则被视为贯穿各层的设计选择，而非附加特性。

    arXiv:2610.24471v2 Announce Type: replace-cross  Abstract: Pluralistic alignment aims to represent meaningful variation in human preferences rather than collapse heterogeneous judgments into a single target. Yet existing work often mixes three distinct questions: what pluralistic behavior a model should exhibit, whose preferences define the reference, and what evidence is sufficient to demonstrate success. We organize the literature around a two-layer taxonomy. Layer 1 distinguishes representing a range of human-supported viewpoints (Overton Pluralism), adapting faithfully to a specified perspective or preference condition (Steerable Pluralism), and matching the response distribution observed in a target population (Distributional Pluralism). Layer 2 distinguishes aggregate populations, countries and regions, and subgroups as human references. Preference representation, implementation mechanism, modality, and evidence quality are treated as cross-cutting design choices rather than addi
    
[^78]: Chinese Decision（中文决策）：面向单次前向概率决策的语言模型系列

    Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making

    [https://arxiv.org/abs/2610.23170](https://arxiv.org/abs/2610.23170)

    Chinese Decision 是一系列中文语言模型，通过单次前向传播直接对候选选项的 logits 进行归一化得到概率分布，无需自回归生成，在医学、法律、金融等领域的决策任务上实现了高准确率与良好校准。

    

    许多基于语言的应用需要对有限决策集合输出概率分布，而非开放式回答。我们提出了 Chinese Decision（中文决策），一个面向单次前向概率决策的中文语言模型系列。给定上下文、指令和候选描述，模型在单次前向传播中提取末尾位置的候选字母 logits，并在有效选项上进行归一化，无需自回归式答案生成。统一的共享接口支持布尔判断、类别选择和序数评分。我们报告了一个已完成的 0.8B 模型，该模型使用分布交叉熵在 1304.8 万个决策上训练；此外还报告了经过额外 RLCD 目标训练的初步 2B 检查点。在包含 307,900 个决策的完整 CJ-Bench 上，未经校准的 0.8B 模型仅使用单一检查点，在通用领域达到 72.24% 的准确率，在医学、法律和金融领域平均达到 76.98% 的准确率，整体期望校准误差为 1.07%。

    arXiv:2610.23170v1 Announce Type: new  Abstract: Many language-based applications require probabilities over a finite set of decisions rather than open-ended responses. We present Chinese Decision, a family of Chinese-language models for single-pass probabilistic decision-making. Given context, an instruction, and candidate descriptions, the model extracts final-position candidate-letter logits in one forward pass and normalizes them over valid options, without autoregressive answer generation. A shared interface supports Boolean judgments, categorical choices, and ordinal scores. We report a completed 0.8B model trained with distribution cross-entropy on 13.048 million decisions and preliminary 2B checkpoints trained with an additional RLCD objective. On the full CJ-Bench of 307,900 decisions, the uncalibrated 0.8B model achieves 72.24% general-domain accuracy and 76.98% average accuracy across medicine, law, and finance using a single checkpoint, with 1.07% overall expected calibrati
    
[^79]: 学会读出：语言模型预训练中的反嵌入动态

    Learning to Read Out: Unembedding Dynamics in Language Model Pretraining

    [https://arxiv.org/abs/2610.21025](https://arxiv.org/abs/2610.21025)

    提出参数轨迹交叉编码方法，仅利用检查点权重直接追踪语言模型预训练中读出层的动态，发现Pythia在预训练0.4%–1.1%的早期窗口内读出层发生重组，且隐藏状态在读出层表达之前就已编码任务信息。

    

    语言模型是在何时获得某种能力的？发展性研究通常通过追踪模型何时开始生成正确的词元来回答这个问题。但这一预测需要经过反嵌入矩阵（即读出层），而读出层本身在预训练过程中也仍在被学习。我们证明，模型的隐藏状态可以在其自身的读出层表达任务信息之前就已编码该信息，因此从输出中读取的时间线混淆了模型所编码的内容与读出层所表达的内容。为了直接追踪读出层，我们引入了参数轨迹交叉编码方法，该方法在每个检查点为每个词元的读出行拟合一个稀疏字典。与作用于激活值的交叉编码器不同，它只需要检查点的权重。我们利用该方法定位到一个狭窄的早期窗口——即Pythia预训练进程的0.4%–1.1%阶段——在此期间读出层发生重组，WordNet词类变得可以从其行中解码。到那时，对隐藏状态的线性探针已经能够解码

    arXiv:2610.21025v1 Announce Type: cross  Abstract: When does a language model acquire a capability? Developmental studies usually answer this by tracking when models start producing the correct token. But that prediction passes through the unembedding matrix, or readout, which is itself still being learned during pretraining. We show that a model's hidden states can encode task information before its own readout expresses it, so timelines read from outputs conflate what the model encodes with what its readout expresses. To track the readout directly, we introduce parameter-trajectory crosscoding, which fits one sparse dictionary to every token's readout row at every checkpoint. Unlike crosscoders on activations, it needs only checkpoint weights. We use it to locate a narrow early window, 0.4-1.1% of the way through Pythia's pretraining, in which the readout reorganizes and WordNet word categories become decodable from its rows. By then, linear probes on the hidden states already decode
    
[^80]: 相似的模型，失灵的胜率：Luce公理在Chatbot竞技场中的失效之处

    Similar Models, Broken Odds: Where Luce's Axiom Fails in Chatbot Arena

    [https://arxiv.org/abs/2610.19095](https://arxiv.org/abs/2610.19095)

    该研究发现Chatbot Arena所依赖的Bradley-Terry模型（即Luce选择公理）在高度相似的模型之间会系统性失效——相似模型间的对决比分数预测的更具决定性（最高达1.37倍），并提出一个仅含4-6个参数、主要通过建模平局的修正方法，其预测能力超越了Bradley-Terry及含数百参数的复杂模型。

    

    Chatbot竞技场使用Bradley-Terry模型对语言模型进行排名，该模型为每个模型赋予单一分数，因而隐含地假设了Luce选择公理。选择理论预言了该公理会在何处失效：近似复制的模型被区分的果断程度会超出其分数所暗示的水平。基于180万条竞技场投票数据，同一模型不同版本之间的投票果断程度是Bradley-Terry预测的1.37倍，同一代不同参数规模之间为1.26倍，同一实验室不同模型之间为1.12倍；一年后的新发布模型也呈现出一致的乘数，且该效应在替代性的平局处理方式、提示词子集与主题划分以及时间检验下依然稳健。两个模型回答在措辞和格式上的相似程度能够追踪这一效应。一个仅含四到六个参数的修正模型在预测留出集投票方面优于Bradley-Terry模型以及拥有数百个参数的模型，这主要归功于对平局的建模；在预先设定的测试中，将该修正模型冻结并应用于新发布的模型时，其收益集中于平局情形，并在成对比较中表现更优。

    arXiv:2610.19095v1 Announce Type: new  Abstract: Chatbot Arena ranks language models with the Bradley-Terry model, which gives each model one score and so assumes Luce's choice axiom. Choice theory predicts where this fails: near-copies are told apart more decisively than their scores imply. On 1.8 million Arena votes, votes between versions of one model are 1.37 times as decisive as Bradley-Terry predicts, between sizes of one generation 1.26 times and between models of one lab 1.12 times; a release a year later gives consistent multipliers, and the effect survives alternative tie treatments, prompt subsets and topics, and timing checks. How alike two models' answers are worded and formatted tracks the effect. A correction with four to six parameters predicts held-out votes better than Bradley-Terry and than models with hundreds of parameters, mostly by modelling ties; frozen and carried to new releases in a test fixed in advance, its gains are concentrated in ties, and better head-to
    
[^81]: 差分层次中单计数器语言的若干完备 $\omega$-幂

    Some complete $\omega$-powers of a one-counter language in the difference hierarchy

    [https://arxiv.org/abs/2610.16473](https://arxiv.org/abs/2610.16473)

    本文证明了对于任意自然数 $n,k\geq 1$，存在被单计数器自动机接受的有限语言，其 $\omega$-幂在差分层次中是 $D_{2n-1}(\mathbf{\Pi}^0_k)$-完备的，并给出了对偶类 $D^-_{2n}(\mathbf{\Pi}^0_k)$ 的条件性结果。

    

    固定一个至少含有两个字母的有限字母表 $\Sigma$，以及自然数 $n,k\geq 1$。我们用 $D_n(\mathbf{\Pi}^0_k)$ 表示康托空间 $\Sigma^\omega$ 中属于 $\mathbf{\Pi}^0_k$ 类的 $n$ 个子集之差的类，并用 $D^-_n(\mathbf{\Pi}^0_k)$ 表示 $D_n(\mathbf{\Pi}^0_k)$ 中集合的补集所构成的类。我们证明，对于任意自然数 $n,k\geq 1$，存在一个至少含有两个字母的有限字母表 $\Sigma$ 以及 $\Sigma$ 上一个被单计数器自动机接受的有限语言 $U_{n,k}$，使得其 $\omega$-幂 $$U_{n,k}^\infty:=\{ w_0w_1\cdots\in\Sigma^\omega\mid\forall i\in\omega ~~w_i\in U_{n,k}\}$$ 是 $D_{2n-1}(\mathbf{\Pi}^0_k)$-完备的。我们还证明了对偶类 $D^-_{2n}(\mathbf{\Pi}^0_k)$ 的一个条件性结果：在假设存在一个被单计数器自动机接受的有限语言 $V_{n,3}$、且其 $\omega$-幂 $V_{n,3}^\infty$ 是 $D^-_{2n}(\mathbf{\Pi}^0_3)$-完备的前提下（摘要在此处被截断）。

    arXiv:2610.16473v2 Announce Type: replace-cross  Abstract: Fix a finite alphabet $\Sigma$ having at least two letters, and natural numbers $n,k\geq 1$. We denote by $D_n(\mathbf{\Pi}^0_k)$ the class of differences of $n$ subsets of the Cantor space $\Sigma^\omega$ in $\mathbf{\Pi}^0_k$, and by $D^-_n(\mathbf{\Pi}^0_k)$ the class of complements of sets in $D_n(\mathbf{\Pi}^0_k)$. We prove, for any natural numbers $n,k\geq 1$, the existence of a finite alphabet $\Sigma$ having at least two letters and a finitary language $U_{n,k}$ over $\Sigma$ accepted by a one-counter automaton such that the $\omega$-power $$U_{n,k}^\infty:=\{ w_0w_1\cdots\in\Sigma^\omega\mid\forall i\in\omega ~~w_i\in U_{n,k}\}$$ is $D_{2n-1}(\mathbf{\Pi}^0_k)$-complete. We also prove a conditional result for the dual classes $D^-_{2n}(\mathbf{\Pi}^0_k)$: under the assumption that there exists a finitary language $V_{n,3}$ accepted by a one-counter automaton, such that its $\omega$-power $V_{n,3}^\infty$ is $D^-_{2n}(
    
[^82]: 注入轮次而非标记：学习在何处重定向大语言模型推理以实现自我纠错

    Inject a Turn, Not a Token: Learning Where to Redirect LLM Reasoning for Self-Correction

    [https://arxiv.org/abs/2610.13672](https://arxiv.org/abs/2610.13672)

    该论文提出无需微调的 UPTURN 自我纠错框架，通过仅用最终答案正确性训练的 13 万参数轻量 MLP 头学习失败推理中的最佳干预位置，并将纠错指令作为新的用户轮次而非行内标记注入，从而有效重定向模型推理，在四个推理模型和六个数学基准上均提升准确率。

    

    大语言模型难以可靠地识别并纠正自身推理中的错误。与以往通过微调大语言模型来生成修订或反馈的训练类方法不同，我们学习在失败的推理生成过程中于何处进行干预，并确定如何对其重定向。为了选择干预位置，我们在冻结的编码器上仅利用最终答案的正确性训练了一个 13 万参数的 MLP 头。对于重定向方式，我们发现在固定的生成预算下，已测试的行内注入无法可靠地改变推理方向，而将相同指令作为新的用户轮次呈现则能改变模型的走向。由此得到的纠错框架 UPTURN（在习得位置注入用户提示轮次）无需对大语言模型进行任何微调：它保留选定的前缀，将一条指令作为新的用户轮次注入，并重新生成后缀。在四个推理模型和六个数学基准上，UPTURN 在所有设置下均提升了平均准确率。

    arXiv:2610.13672v1 Announce Type: cross  Abstract: Large language models struggle to reliably identify and correct errors in their own reasoning. Unlike prior training-based approaches that fine-tune an LLM to generate revisions or feedback, we learn where to intervene in failed rollouts and determine how to redirect them. To choose where to intervene, we train a 130K-parameter MLP head on a frozen encoder using only final-answer correctness. For redirection, we find that under a fixed generation budget, the tested inline injections fail to reliably redirect reasoning, whereas presenting the same instruction in a new user turn changes the model's course. The resulting correction framework, UPTURN (User-Prompt TURN injection at a learned position), requires no LLM fine-tuning: it retains a selected prefix, injects an instruction as a new user turn, and regenerates the suffix. Across four reasoners and six math benchmarks, UPTURN improves average accuracy in every setting over a 32-rollo
    
[^83]: 从读代码到读规约：面向LLM驱动的代码库维护的验证层

    From Reading Code to Reading Spec: A Verified Layer for LLM-Driven Codebase Maintenance

    [https://arxiv.org/abs/2609.06383](https://arxiv.org/abs/2609.06383)

    论文提出PROOF系统，将代码库拓扑抽象为分层自然语言规约，并通过从规约重构源代码来证明语义等价，使LLM能够借助规约间接、可靠地执行代码库维护并防止语义漂移。

    

    LLM生成代码的快速增长加剧了软件的复杂性，并加重了工程师的维护负担。虽然LLM提供了一种潜在的自动化替代方案，但这种结构性复杂性阻碍了它们直接管理代码库的能力。我们提出了原始功能可证明表示（PROOF），它通过结构化规约间接地管理代码库。为了严格地通过这些规约实现代码库的全生命周期管理，PROOF将代码库拓扑抽象为分层的自然语言表示。为了建立绝对的信任，该系统通过仅从规约重构源代码来证明语义等价性。这一经过验证的基础驱动维护请求，在执行代码修改的同时同步更新自身，以防止语义漂移。在真实世界代码库上的实验证实了这些规约的有效性。

    arXiv:2609.06383v2 Announce Type: replace-cross  Abstract: The rapid growth of LLM-generated code increases software complexity and the maintenance burden on engineers. While LLMs offer a potential automated alternative, this structural complexity hinders their ability to manage codebases directly. We introduce the Provable Representation Of Original Functionality (PROOF), which manages codebases indirectly via structured specifications. To enable full-lifecycle codebase management strictly through these specifications, PROOF abstracts codebase topology into a hierarchical natural-language representation. To establish absolute trust, the system proves semantic equivalence by reconstructing source code exclusively from this specification. This verified foundation drives maintenance requests, executing code modifications while synchronously updating itself to prevent semantic drift. Experiments on real-world repositories confirm the effectiveness of these specifications.
    
[^84]: 超越原始转录：基于LLM的数字孪生中的结构化人物画像提取

    Beyond Raw Transcripts: Structured Persona Extraction for LLM-Based Digital Twins

    [https://arxiv.org/abs/2608.20344](https://arxiv.org/abs/2608.20344)

    本文提出，数字孪生预测准确性的关键瓶颈在于人物画像信息的结构组织方式，而非信息量，并通过引入基于消费者行为理论的BDE结构化模式，将预测准确性提升1.91个百分点。

    

    arXiv:2608.20344v1 公告类型：新论文 摘要：基于LLM的“数字孪生”旨在模拟个体在新环境中的行为或对新问题的回应，这需要基于该个体先前回应的某种表示。一种常见方法是从调查转录或总结回应中构建这种表示。先前工作表明，将长转录压缩为较短的LLM生成摘要不会显著降低预测准确性，这表明信息量并非主要瓶颈。在本工作中，我们认为关键限制在于结构：即人物画像信息在提供给模拟器模型之前的组织方式。我们通过比较非结构化摘要与结构化人物画像表示来研究这一点。首先，我们引入一个手工设计的模式（BDE：背景、决策过程、评估），该模式基于消费者行为理论，并证明其相对于原始转录将预测准确性提高了1.91个百分点。

    arXiv:2608.20344v1 Announce Type: new  Abstract: LLM-based "digital twins" aim to simulate how an individual would behavein new environments or respond to novel questions, given some representation of that individual's prior responses. A common approach constructs this representation from survey transcripts or summaries responses. Prior work shows that compressing long transcripts into shorter LLM-generated summaries does not significantly reduce predictive accuracy, suggesting that information volume is not the primary bottleneck.   In this work, we argue that the key limitation is instead structural:how persona information is organized before being provided to thesimulator model. We study this by comparing unstructured summaries with structured persona representations. First, we introduce a hand-craftedschema (BDE: Background, Decision procedure, Evaluation), grounded in consumer-behavior theory, and show that it improves predictive accuracy over raw transcripts by +1.91 percentage p
    
[^85]: BBOWP-Bench：评估大语言模型在黑箱优化文字问题上的能力

    BBOWP-Bench: Evaluating LLMs on Black-Box Optimization Word Problems

    [https://arxiv.org/abs/2608.02612](https://arxiv.org/abs/2608.02612)

    该论文提出BBOWP-Bench基准，首次评估大语言模型能否从自然语言描述的黑箱优化文字问题中同时推断出合适的搜索空间并选择恰当的优化算法。

    

    优化问题的建模方式会强烈影响最终解的质量，然而良好的建模通常需要大量专业知识。近期研究已探索从自然语言描述中自动推导优化问题，但现有基准主要针对目标与约束能够显式写成数学表达式的场景。许多实际重要的问题天然适合被视为黑箱优化（BBO）问题——其中只有目标函数值可观测，而函数形式不可获得。在BBO中，搜索空间设计（作为问题建模的一部分）与优化算法的选择对求解至关重要，而利用大语言模型（LLM）自动化这些过程仍具挑战性。我们提出了黑箱优化文字问题，这是一个新颖的问题设定：系统需要从自然语言描述的问题中同时推断出搜索空间并选择优化算法。

    arXiv:2608.02612v2 Announce Type: replace  Abstract: Formulating an optimization problem strongly affects final solution quality, yet good formulations usually require substantial expertise. Recent studies have explored automatically deriving optimization problems from natural-language descriptions, but existing benchmarks focus on settings where objectives and constraints can be written explicitly as mathematical expressions. Many practically important problems are naturally treated as black-box optimization (BBO) problems, in which only objective values are observable, and the functional form is unavailable. In BBO, search space design (as part of problem formulation) and optimization algorithm selection are crucial for problem-solving. Automating these processes with large language models (LLMs) remains challenging. We introduce Black-Box Optimization Word Problems (BBOWP), a novel problem setting in which a system must infer both a search space and an optimization algorithm from a 
    
[^86]: 刻画临床笔记与结构化电子健康记录用药史中的治疗情境用药证据

    Characterizing Treatment-Context Medication Evidence Across Clinic Notes and Structured EHR Medication History

    [https://arxiv.org/abs/2608.01570](https://arxiv.org/abs/2608.01570)

    该研究构建了以临床笔记为依据的用药参考标准，通过大语言模型提议、确定性规范投影和词汇清理将用药归一化的规范一致性从81.25%提升至93.58%，并量化了语义与时间匹配对临床笔记与结构化EHR用药史重叠程度的影响。

    

    临床笔记中可能描述了在同一次就诊中未反映在结构化电子健康记录（EHR）用药史中的用药变化。术语差异和记录时间的不同进一步增加了对这些不一致之处的解读难度。我们评估了临床笔记中的用药规范化，并考察了语义匹配和时间匹配如何影响其与结构化用药史的重叠程度。我们基于大语言模型（LLM）提出的用药提及、确定性规范投影以及现有的针对性更正记录，构建了一个以笔记为依据的参考标准；随机审计记录则单独汇总。在来自留出患者的5,360条参考提及记录上，条件规范一致性从表面匹配的81.25%提升至经过词汇清理和人工筛选别名后的93.58%。候选提及与参考提及的对齐进行了单独评估，F1分数为0.6547。在27,552条参考记录上……（摘要在此处被截断）

    arXiv:2608.01570v2 Announce Type: replace  Abstract: Clinic notes may describe medication changes that are not reflected in structured electronic health record (EHR) medication history at the same encounter. Differences in terminology and documentation timing further complicate the interpretation of these discrepancies. We evaluated medication normalization in clinic notes and examined how semantic and temporal matching affected overlap with structured history. A note-grounded reference was constructed from large language model (LLM)-proposed mentions, deterministic canonical projection, and existing targeted correction records; random-audit records were summarized separately. On 5,360 reference mention rows from held-out patients, conditional canonical agreement increased from 81.25% with surface matching to 93.58% after lexical cleanup and curated aliases. Candidate-reference mention alignment was evaluated separately, yielding an F1 score of 0.6547. Across 27,552 reference rows, exa
    
[^87]: URSA：面向实用性逆合成评估的化学感知基准

    URSA: Chemistry-Aware Benchmark for Utilitarian Retrosynthesis Assessment

    [https://arxiv.org/abs/2607.04688](https://arxiv.org/abs/2607.04688)

    URSA提出了一个化学感知的逆合成评估基准框架，首次从形式指标（如收敛到商业可得起始原料）和模拟专家化学家判断的化学合理性两个维度，对端到端逆合成模型与大语言模型进行客观、可解释的比较评估。

    

    合成规划旨在为目标分子寻找可行的反应路径，是药物发现中最重要且最具挑战性的任务之一。近期的进展催生了专用的深度学习逆合成系统和通用的大型语言模型，但由于缺乏灵活且具备化学可解释性的基准测试协议，对它们进行客观比较仍然十分困难。在本研究中，我们提出了URSA（实用性逆合成评估）评估框架，该框架不仅能够从形式角度（例如是否收敛到可商业购买的起始原料）对合成路线进行基准评估，还能从化学合理性角度进行评估，模拟专家化学家评估反应与路线的方式。该研究对传统端到端逆合成解决方案和大语言模型在合成规划任务上进行了全面评估（原文摘要在此处截断）。

    arXiv:2607.04688v2 Announce Type: replace-cross  Abstract: Synthesis planning aiming to find pathways of reactions for a target molecule is one of the most important and challenging tasks in drug discovery. Recent progress has produced both specialized deep-learning retrosynthesis systems and general-purpose large language models, but objective comparison remains difficult due to the lack of flexible, chemically interpretable benchmarking protocols. In the current study, we are introducing the URSA (Utilitarian RetroSynthesis Assessment) evaluation framework that provides the opportunity to benchmark the synthetic routes not only from a formal perspective, such as convergence to commercially available starting materials, but also from a chemical plausibility perspective, mimicking the way expert chemists evaluate the reactions and routes. The study covers a comprehensive evaluation of both conventional end-to-end retrosynthesis solutions and LLMs for the synthesis planning task on a se
    
[^88]: 当简单更优时：评估中世纪拉丁手稿的翻译流水线

    When Simpler Is Better: Evaluating Translation Pipelines for Medieval Latin Manuscripts

    [https://arxiv.org/abs/2607.03836](https://arxiv.org/abs/2607.03836)

    在中世纪拉丁手稿翻译任务中，参数量少几个数量级的领域专用OCR模型反而优于通用视觉语言模型（字符错误率最多降低4.3倍），并据此构建了包含1,383条对齐图像、转录与译文的新型数据集IPC。

    

    尽管机器翻译取得了显著进展，视觉语言模型（VLM）在历史手稿上的表现仍然不佳。该领域考验着自然语言处理（NLP）的核心能力：低资源转写、古旧词汇以及嘈杂的输入信号。我们提出了一个系统性框架，用于评估从中世纪拉丁手稿图像到译文的完整流水线。在这一场景中，抄写员的速记符号、连字以及羊皮纸的破损暴露出在干净文本基准测试中难以察觉的失败模式。在CATMuS Latin数据集上的基准测试揭示了一种专业化差距：领域专用的光学字符识别（OCR）模型与通用VLM相比，可将字符错误率降低至多4.3倍，尽管其参数量少了几个数量级。我们提出了Interpres-Parallel-Corpus（IPC），一个包含1,383条对齐的手稿图像行、转录文本及实验（结果）的新型数据集。

    arXiv:2607.03836v2 Announce Type: replace-cross  Abstract: Despite remarkable progress in machine translation, Vision Language Models (VLMs) struggle on historical manuscripts, a domain that stresses core Natural Language Processing (NLP) capabilities: low-resource transliteration, archaic vocabulary, and noisy input signals. We present a systematic framework for evaluating the full image-to-translation pipeline on medieval Latin manuscripts, a setting in which scribal shorthand, ligatures, and parchment degradation expose failure modes that are invisible in clean-text benchmarks. Benchmarking on the CATMuS Latin dataset reveals a specialization gap: domain-specific Optical Character Recognition (OCR) models reduce character error rate by up to 4.3$\times$ compared to general-purpose VLMs, despite operating at orders of magnitude fewer parameters. We introduce the Interpres-Parallel-Corpus (IPC), a novel dataset comprising 1,383 aligned manuscript image lines, transcriptions, and exper
    
[^89]: QVal：评估长程LLM智能体的密集监督信号

    QVal: Evaluating Dense Supervision Signals for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2606.32034](https://arxiv.org/abs/2606.32034)

    该论文提出QVal——一个无需训练的测试平台，通过衡量密集监督信号在每个状态下对候选动作的排序与参考策略回报排序的一致性（Q-对齐性）来直接评估信号质量，并在四个环境中对七个方法族的21种方法进行基准测试，发现简单提示基线平均Q-对齐性最高且性能按方法族高度聚集。

    

    LLM智能体越来越多地在长时程任务中执行行动，而仅基于结果的奖励几乎无法说明哪些中间动作是好的。密集监督方法通过为中间步骤打分来解决这一问题，但这类方法通常通过完整训练流程的下游性能来评估。这种方式成本高昂，将信号质量与其他训练选择混为一谈，也意味着这些方法很少在共同基准上进行比较。我们提出了QVal，一个无需训练的测试平台，用于衡量密集信号的$Q$-对齐性：即该信号在每个状态下对候选动作的排序是否与其在参考策略下的回报排序一致。我们在四个环境中，使用六个开源权重骨干模型，对来自七个方法族的21种方法进行了基准测试。结果表明，简单的提示基线平均而言$Q$-对齐性最高，且性能按方法族高度聚集。代码生成的信号仅在小状态和动作空间中表现良好；基于嵌入的信号则……

    arXiv:2606.32034v2 Announce Type: replace-cross  Abstract: LLM agents increasingly act over long horizons, where outcome-only rewards say little about which intermediate actions were good. Dense supervision methods address this by scoring intermediate steps, but they are usually evaluated through the downstream performance of a full training pipeline. This is expensive, confounds signal quality with other training choices, and means that methods are rarely compared on common ground. We introduce QVal, a training-free testbed that measures the $Q$-alignment of a dense signal: whether it orders the candidate actions in each state the same way as their returns under a reference policy do. We benchmark 21 methods from seven families with six open-weight backbones in four environments. Simple prompting baselines are the most $Q$-aligned on average, and performance clusters strongly by family. Code-generated signals do well only in small state and action spaces; embedding-based signals are w
    
[^90]: 荷兰语构音障碍连续语音的人类识别与自动识别比较：一项案例研究

    Comparing Human and Automatic Recognition of Dutch Dysarthric Continuous Speech: A Case Study

    [https://arxiv.org/abs/2606.30237](https://arxiv.org/abs/2606.30237)

    本研究通过案例研究发现，重度构音障碍的荷兰语语音识别对人类和现成ASR系统都极为困难（WER超过70%），但经过个性化微调的DSR模型性能超越了人类听者，正接近可用于支持日常交流的水平。

    

    为了开发个性化的构音障碍语音识别（DSR）模型，本研究比较了人类听者与三个最先进的现成ASR系统在识别一位重度构音障碍说话者的荷兰语连续朗读和自发语音方面的表现。结果表明，人类听者和三个现成ASR系统的平均词错误率（WER）均超过70%，说明构音障碍语音识别对人类和ASR系统而言都极具挑战性。在构音障碍语音上进行微调显著降低了WER。尽管整体WER仍然相当高（>23%），但个性化DSR模型的表现已经超越了人类听者，其性能正逐渐接近可用于支持构音障碍说话者日常交流的水平。未来的研究应聚焦于改进自发语音上的个性化DSR模型。

    arXiv:2606.30237v2 Announce Type: replace  Abstract: In our goal to develop personalised dysarthric speech recognition (DSR) models, this study compared the recognition performances of human listeners and those of three state-of-the-art, off-the-shelf ASR systems (Whisper-large-V3, Google Chirp 3, and Omnilingual) on the recognition of Dutch continuous read and spontaneous speech from a single speaker with severe dysarthria. Results showed that both humans listeners and the three off-the-shelf ASR systems exhibit word error rates (WER) exceeding 70% on average, indicating that DSR is highly challenging for both humans and ASR systems. Fine-tuning on the dysarthric speech significantly reduced WER. Although overall WERs are still quite high (>23%), the personalised DSR models outperformed the human listeners, and performance is getting closer to being useful for supporting day-to-day communication of dysarthric speakers. Future research should focus on improving personalized DSR on spon
    
[^91]: 从事实到洞察：面向角色扮演代理的基于人格的双记忆框架与数据集

    From Facts to Insights: A Persona-Driven Dual Memory Framework and Dataset for Role-Playing Agents

    [https://arxiv.org/abs/2605.25693](https://arxiv.org/abs/2605.25693)

    提出了DualMem双记忆框架，将记忆解耦为事实认知与人格洞察两条流，并配合RoleMemo数据集，使仅40亿参数的模型在长期角色扮演中的人格忠实度超越了基于DeepSeek-V3.2的零样本框架。

    

    尽管角色扮演代理在短期交互中表现出色，但长期对话会使上下文窗口不堪重负，这促使了外部记忆框架的发展。当前系统通常依赖于与人格无关的摘要方法，这种方法仅记录事实而不进行人格特定的解读，导致产生泛化的回复，损害了人格忠实度。为弥合这一差距，我们引入了RoleMemo数据集，该数据集包含四个推理任务，其中事实片段必须通过人格视角进行解读才能得出正确答案。在RoleMemo上的评估揭示了与人格无关的框架的关键局限性。因此，我们提出了DualMem，它将记忆解耦为两条流：事实认知与基于人格的洞察。通过监督微调（SFT）和强化学习（RL）进行训练，我们的框架仅使用40亿参数的模型，就在持续人格忠实度方面超越了由DeepSeek-V3.2驱动的零样本与人格无关框架。

    arXiv:2605.25693v2 Announce Type: replace  Abstract: While role-playing agents excel in short-term interactions, long-term conversations overwhelm context windows, motivating external memory frameworks. Current systems typically rely on persona-agnostic summarization, which records facts without persona-specific interpretation, yielding generic responses that compromise persona fidelity. To bridge this gap, we introduce RoleMemo, a dataset featuring four reasoning tasks where the factual fragments must be interpreted through the persona to reach the correct answer. Evaluation on RoleMemo exposes critical limitations of persona-agnostic frameworks. We thus propose DualMem, which decouples memory into two streams: factual cognition and persona-conditioned insight. Trained through Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), our framework with a 4B-parameter model outperforms zero-shot persona-agnostic frameworks powered by DeepSeek-V3.2 for sustained persona fidelity. Ou
    
[^92]: SA-RAG：为检索增强生成构建结构化的噪声对话

    SA-RAG: Structuring Noisy Conversations for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2605.24366](https://arxiv.org/abs/2605.24366)

    提出SA-RAG框架，通过质量感知的结构归纳方法直接从噪声对话数据中构建表格形式的可靠中间结构，从而提升检索增强生成在嘈杂上下文中的表现。

    

    大语言模型（LLM）已被广泛应用于对话应用中。然而，其对参数化知识的依赖限制了其在需要动态或领域特定信息的真实场景中的可靠性。检索增强生成（RAG）通过在生成过程中引入外部知识来应对这一局限，但现有的RAG方法往往难以处理噪声或无关的上下文，因为它们通常假设高质量的结构已经可用。在这项工作中，我们提出了结构感知检索增强生成（SA-RAG），这是一个有原则的结构归纳与生成框架，能够直接从噪声对话数据中构建可靠的中间结构。SA-RAG使用表格作为紧凑且可控的结构化表示。我们引入了一个质量感知的结构归纳框架，联合建模语义一致性和结构一致性。此外，我们还探索了……（摘要在此处被截断）

    arXiv:2605.24366v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have been widely adopted in conversational applications. However, their reliance on parametric knowledge limits reliability in real-world scenarios that require dynamic or domain-specific information. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge during generation, but existing RAG methods often struggle with noisy or irrelevant contexts, as they typically assume that high-quality structures are already available. In this work, we propose Structure-aware Retrieval-Augmented Generation (SA-RAG), a principled structure induction and generation framework that constructs reliable intermediate structures directly from noisy conversational data. SA-RAG uses tables as compact and controllable structured representations. We introduce a quality-aware structure induction framework that jointly models semantic and structural consistency. Furthermore, we explore bo
    
[^93]: TransitLM：面向无地图公交路线生成的大规模数据集与基准

    TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation

    [https://arxiv.org/abs/2605.22355](https://arxiv.org/abs/2605.22355)

    该论文发布了包含四个中国城市近1300万条路线规划记录的大规模数据集TransitLM，证明大语言模型可完全从数据中学习公交路线规划，无需地图即可实现从起终点信息到路线的端到端生成。

    

    公共交通路线规划传统上依赖于结构化地图基础设施和复杂的路由引擎，目前尚无数据集支持训练模型来摆脱这一依赖。我们提出了TransitLM，这是一个包含来自四个中国城市的近1300万条公交路线规划记录的大规模数据集，覆盖120,845个车站和13,666条线路，以持续预训练语料库的形式发布，并附带三个评估任务的基准数据及互补的评估指标。实验表明，在TransitLM上训练的大语言模型能够以高准确率生成结构有效的路线，并能在没有任何显式映射的情况下，隐式地将任意GPS坐标对齐到合适的车站。这些结果表明，公共交通路线规划可以完全从数据中学习，从而实现直接根据起终点信息进行端到端、无地图的路线生成。数据集和基准已在 https://huggingface.co/datasets/GD-ML/ 公开。

    arXiv:2605.22355v2 Announce Type: replace-cross  Abstract: Public transit route planning traditionally depends on structured map infrastructure and complex routing engines, and no existing dataset supports training models to bypass this dependency. We present TransitLM, a large-scale dataset of nearly 13 million transit route planning records from four Chinese cities covering 120,845 stations and 13,666 lines, released as a continual pre-training corpus and benchmark data for three evaluation tasks with complementary metrics. Experiments show that an LLM trained on TransitLM produces structurally valid routes at high accuracy and implicitly grounds arbitrary GPS coordinates to appropriate stations without any explicit mapping. These results demonstrate that transit route planning can be learned entirely from data, enabling end-to-end, map-free route generation directly from origin-destination information. The dataset and benchmark are available at https://huggingface.co/datasets/GD-ML/
    
[^94]: SpecHop：用于加速多跳检索智能体的连续推测框架

    SpecHop: Continuous Speculation for Accelerating Multi-Hop Retrieval Agents

    [https://arxiv.org/abs/2605.21965](https://arxiv.org/abs/2605.21965)

    SpecHop提出一种连续推测框架，通过维护多个推测线程并异步验证、提交或回滚预测结果，在不改变模型最终轨迹的前提下无损加速多跳检索智能体，显著降低端到端延迟。

    

    大型语言模型越来越多地借助网络搜索和文档检索等外部工具来解决信息密集型任务。然而，在复杂任务中，多跳工具调用会引入显著的延迟，因为模型必须反复等待工具返回的观测结果才能继续执行。我们研究了如何在不改变模型最终轨迹的前提下加速此类执行过程（即与未加速时模型本会采取的最终轨迹保持一致），其假设前提是能够使用速度更快但可靠性较低的推测工具。我们为多跳工具使用场景下的无损推测建立了一个理论框架，刻画了最优可实现的延迟增益。我们提出了SpecHop，一个连续推测框架：它同时维护多个推测线程，在目标工具输出到达时异步验证预测的观测结果，提交正确的分支，并回滚错误的分支。这在保证准确性的同时降低了实际运行时间延迟。我们证明了SpecHop可以...

    arXiv:2605.21965v2 Announce Type: replace  Abstract: Large language models increasingly use external tools such as web search and document retrieval to solve information-intensive tasks. However, multi-hop tool use in complex tasks introduces substantial latency, since the model must repeatedly wait for tool observations before continuing. We study how to accelerate such trajectories without changing the final trajectory the model would have taken without acceleration, assuming access to faster but less reliable speculator tools. We develop a theoretical framework for lossless speculation in multi-hop tool-use settings, characterizing the optimal achievable latency gain. We propose SpecHop, a continuous speculation framework that maintains multiple speculative threads, verifies predicted observations asynchronously as target tool outputs arrive, commits correct branches, and rolls back incorrect ones. This preserves accuracy while reducing wall-clock latency. We show that SpecHop can a
    
[^95]: 先筛查再解读：一种面向基于基准测试的LLM置信度信号的可移植有效性筛查方法

    Screen Before You Interpret: A Portable Validity Screen for Benchmark-Based LLM Confidence Signals

    [https://arxiv.org/abs/2604.17714](https://arxiv.org/abs/2604.17714)

    本文借鉴临床人格评估的有效性筛查传统，提出一种基于L、Fp和RBS指标的可移植筛查方法，在解读之前检验LLM置信度信号是否真正跟踪正确性，对20个前沿模型的评估发现15个有效、4个不确定、1个无效（DeepSeek-R1弃答了95%的正确答案）。

    

    LLM置信度信号驱动着弃答、路由和升级等决策，然而现有评估很少在计算指标之前检查信号是否随正确性而变化。临床人格评估领域通过在解读实质性量表之前筛查反应有效性来处理类似的问题。我们遵循这一传统，规范了一种可移植的筛查方法。该方法从正确性-置信度列联表中计算三个指标：错误上高置信度的比率（L）、正确答案上低置信度的比率（Fp）以及二者的对比度（RBS，等于负的Youden's J统计量）。它不引入任何新的统计量，而是区分整体高置信度、整体弃答和反转三种失效模式，附加区间保护措施，并且只有当信号可测量地跟踪正确性时才将其判定为有效。在20个前沿模型对452个二选一题目进行二元KEEP/WITHDRAW（保留/弃答）探测的实验中，一个模型被判定为无效（DeepSeek-R1弃答了其95%的正确答案），四个模型为不确定，15个模型为有效。

    arXiv:2604.17714v2 Announce Type: replace-cross  Abstract: LLM confidence signals drive abstention, routing and escalation, yet evaluations rarely check whether a signal varies with correctness before computing metrics on it. Clinical personality assessment handles the analogous problem by screening response validity before interpreting substantive scales. We specify a portable screen in that tradition. From the correctness-by-confidence table it computes the rate of high confidence on errors (L), the rate of low confidence on correct answers (Fp), and their contrast (RBS, equal to minus Youden's J). It adds no new statistics: it separates blanket confidence, blanket withdrawal and inversion, attaches interval guards, and calls a signal Valid only when it measurably tracks correctness. On 20 frontier models answering 452 two-alternative items with binary KEEP/WITHDRAW probes, one model is Invalid (DeepSeek-R1 withdraws 95% of its correct answers), four are Indeterminate and 15 are Vali
    
[^96]: 基于词典的跨语言词义投影生成概念词汇化

    Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection

    [https://arxiv.org/abs/2604.14397](https://arxiv.org/abs/2604.14397)

    提出一种用双语词典增强对齐器并结合过滤策略的跨语言词义投影方法，能够将WordNet式词汇资源自动扩展到新语言，在保持可解释性和资源高效的同时提升精确率。

    

    我们研究通过词义生成来自动将WordNet风格的词汇资源扩展到新语言的任务。我们通过语义投影将目标语言的词条与现有词汇概念相关联来生成词义。给定一个带有词义标注的英语语料库及其翻译，我们的方法将标注好的同义词集（synsets）投影到对齐的目标语言词汇上，并将相应的词条分配给这些同义词集。为了生成对齐并确保其质量，我们使用双语词典对预训练的基础对齐器进行增强，该词典还用于过滤不正确的词义投影。我们在多种语言上对该方法进行评估，并与先前的方法、基于词典的基线以及大型语言模型基线进行比较。结果表明，所提出的投影-过滤策略在保持可解释性和资源高效性的同时提高了精确率。我们发布了代码、文档和生成的词义清单。

    arXiv:2604.14397v3 Announce Type: replace-cross  Abstract: We study the task of automatically expanding WordNet-style lexical resources to new languages through sense generation. We generate senses by associating target-language lemmas with existing lexical concepts via semantic projection. Given a sense-tagged English corpus and its translation, our method projects the annotated synsets onto aligned target-language tokens and assigns the corresponding lemmas to those synsets. To generate alignments and ensure their quality, we augment a pretrained base aligner with a bilingual dictionary, which is also used to filter incorrect sense projections. We evaluate the method on multiple languages, comparing it to prior methods, as well as dictionary-based and large language model baselines. Results show that the proposed project-and-filter strategy improves precision while remaining interpretable and resource- efficient. We release our code, documentation, and generated sense inventories at 
    
[^97]: DRTriton：基于大规模合成数据驱动的强化学习实现Triton内核生成

    DRTriton: Large-Scale Synthetic Data Driven Reinforcement Learning for Triton Kernel Generation

    [https://arxiv.org/abs/2603.21465](https://arxiv.org/abs/2603.21465)

    提出DRTriton框架，通过CSP-DAG大规模合成数据、解耦奖励的课程化强化学习以及测试时搜索算法，训练大语言模型将PyTorch程序高效转换为优化的Triton内核。

    

    开发高效的CUDA内核是生成式AI行业中一项基础但极具挑战性的任务。近期研究利用大语言模型（LLM）自动将PyTorch参考实现转换为CUDA内核，显著降低了工程工作量。然而，即便是最先进的LLM（如GPT-5.2和Claude-Sonnet-4.5）仍然难以胜任这项任务。为应对这一挑战，我们提出了DRTriton，一个可扩展的学习框架，用于训练LLM将PyTorch程序转换为高度优化的Triton内核，并在运行时将其编译为CUDA内核。DRTriton包含三个关键组件：（i）数据合成算法CSP-DAG，可在算子空间中保证完全覆盖、无偏均匀采样，并能控制难度；（ii）采用解耦奖励的课程化强化学习框架，可联合优化转换成功率和执行速度；（iii）测试时搜索算法，进一步……（原文摘要在此处截断）

    arXiv:2603.21465v3 Announce Type: replace  Abstract: Developing efficient CUDA kernels is a fundamental yet challenging task in the generative AI industry. Recent research leverages Large Language Models (LLMs) to automatically convert PyTorch reference implementations to CUDA kernels, significantly reducing engineering effort. State-of-the-art LLMs, such as GPT-5.2 and Claude-Sonnet-4.5, still struggle with this task. To address this challenge, we propose DRTriton, a scalable learning framework for training LLMs to convert PyTorch programs into highly optimized Triton kernels, which are then compiled to CUDA kernels at runtime. DRTriton consists of three key components: (i) a data synthetic algorithm CSP-DAG that guarantees full coverage and unbiased uniform sampling over the operator space with controlled difficulty; (ii) a curriculum RL framework with decoupled rewards that jointly optimizes conversion success rate and execution speed; and (iii) a test-time search algorithm that fur
    
[^98]: RECOVER：通过假设变体的智能体式编排实现基于证据恢复的鲁棒实体校正

    RECOVER: Robust Entity Correction via agentic Orchestration of hypothesis Variants for Evidence-based Recovery

    [https://arxiv.org/abs/2603.16411](https://arxiv.org/abs/2603.16411)

    RECOVER是一个智能体式后ASR实体校正框架，它将多个ASR假设作为证据，通过实体检索、约束条件下的LLM校正、确定性防护栏过滤以及可触发反馈轮次的LLM评论家，有效恢复稀有和领域特定实体。

    

    在自动语音识别（ASR）中，对稀有和领域特定术语的实体识别极具挑战性。在金融、医疗和空中交通管制等领域，这些错误会造成高昂代价。如果某个实体在ASR输出中完全缺失，仅基于文本的后处理校正几乎没有信号来恢复它。为解决这一问题，我们提出了RECOVER，一个智能体式后ASR实体校正框架。它将多个假设作为来自ASR的证据，检索相关实体，在约束条件下提出大语言模型（LLM）校正建议，通过确定性防护栏对其进行过滤，并使用一个LLM评论家来判断每个存活的编辑，还能触发有针对性的反馈轮次，以恢复第一轮处理中遗漏的实体。这些假设通过以下几种策略之一加以利用：1-Best、实体感知选择（Entity-Aware Select）、识别器输出投票错误降低（ROVER）集成、LLM-Select以及自适应路由。该方法在五个多样的数据（摘要在此处截断）

    arXiv:2603.16411v2 Announce Type: replace  Abstract: Entity recognition in Automatic Speech Recognition (ASR) is challenging for rare and domain-specific terms. In domains such as finance, medicine, and air-traffic control, these errors are costly. If an entity is entirely absent from the ASR output, text-only post-correction has little signal to recover it. To address this, we introduce RECOVER, an agentic post-ASR entity-correction framework. It leverages multiple hypotheses as evidence from ASR, retrieves relevant entities, proposes Large Language Model (LLM) correction under constraints, filters them through deterministic guardrails and uses an LLM critic that judges each surviving edit and can trigger a targeted feedback round to recover an entity the first pass missed. The hypotheses are used by one of several strategies, namely 1-Best, Entity-Aware Select, Recognizer Output Voting Error Reduction (ROVER) Ensemble, LLM-Select and Adaptive Routing. Evaluated across five diverse da
    
[^99]: FlashSampling：快速且内存高效的确切采样

    FlashSampling: Fast and Memory-Efficient Exact Sampling

    [https://arxiv.org/abs/2603.15854](https://arxiv.org/abs/2603.15854)

    FlashSampling将精确采样融合进LM头的矩阵乘法运算中，通过在片上逐块计算并添加Gumbel噪声避免在显存中物化logits张量，并以流式点对点通信替代all-gather，实现了跨最多8个GPU的快速、精确且内存高效的大词表采样。

    

    arXiv:2603.15854v4 公告类型：replace-cross 摘要：从分类分布中进行采样在数学上很简单，但在大词表解码场景中，它通常会在语言模型输出头（LM head）之后触发额外的内存流量和额外的内核调用。我们提出了FlashSampling，这是一种精确的采样原语，它将采样融合到LM头的矩阵乘法中，并且从不在HBM（高带宽显存）中物化logits张量。该方法十分简单：在芯片上逐块计算logits，添加Gumbel噪声，每行和每个词表块仅保留一个最大值，最后通过对各块进行小型归约完成采样。在张量并行解码中，FlashSampling用流式点对点写入取代了logits的all-gather（全收集）：这使得GPU到GPU的通信与计算和HBM加载相重叠，最多可扩展至8个GPU，并在大批量规模下实现接近理想的扩展性。我们的内核是精确的，因为argmax可以在分区上进行分解；针对在线和张量并行设置的分组变体通过分类分布的层次化分解同样保证了精确性。

    arXiv:2603.15854v4 Announce Type: replace-cross  Abstract: Sampling from a categorical distribution is mathematically simple, but in large-vocabulary decoding, it often triggers extra memory traffic and extra kernels after the LM head. We present FlashSampling, an exact sampling primitive that fuses sampling into the LM-head matmul and never materializes the logits tensor in HBM. The method is simple: compute logits tile-by-tile on chip, add Gumbel noise, keep only one maximizer per row and per vocabulary tile, and finish with a small reduction over tiles. In tensor-parallel decoding, FlashSampling replaces the all-gather of logits with streaming peer-to-peer writes: This overlaps GPU-to-GPU communication with computation and HBM loads across up to 8 GPUs, with near-ideal scaling at large batch sizes. Our kernel is exact because argmax decomposes over partitions; grouped variants for online and tensor-parallel settings are exact by hierarchical factorization of the categorical distribu
    
[^100]: 基于非相干性调整语义体积的多模态大语言模型不确定性量化

    Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume

    [https://arxiv.org/abs/2602.24195](https://arxiv.org/abs/2602.24195)

    提出了UMPIRE，一个免训练且无需外部工具的多模态大语言模型不确定性量化框架，其受行列式点过程启发，通过计算新颖的“非相干性调整语义体积”，将语义嵌入离散度与响应二次熵相结合，从而高效支持多种输入输出模态。

    

    尽管多模态大语言模型（MLLM）能力强大，但它们可能产生看似合理实则错误的输出，这阻碍了其可靠部署。准确的不确定性度量可以将不可靠的查询升级给人类专家或更大的模型处理，从而提升性能。然而，现有的不确定性度量存在实际限制，例如仅针对特定模态设计、依赖外部工具或计算成本高昂。我们提出了UMPIRE，这是一个面向多模态大语言模型的免训练不确定性量化框架，它能够在各种输入和输出模态上高效工作，无需外部工具，仅依赖模型自身的内部模态特征。受行列式点过程（DPPs）的启发，UMPIRE针对给定任务实例计算采样得到的MLLM响应的一种新颖的非相干性调整语义体积，该方法将语义嵌入的离散度与响应的二次熵相结合。

    arXiv:2602.24195v2 Announce Type: replace  Abstract: Despite their capabilities, Multimodal Large Language Models (MLLMs) may produce plausible but erroneous outputs, hindering reliable deployment. Accurate uncertainty metrics could enable escalation of unreliable queries to human experts or larger models for improved performance. However, existing uncertainty metrics have practical constraints, such as being designed only for specific modalities, reliant on external tools, or computationally expensive. We introduce UMPIRE, a training-free uncertainty quantification framework for MLLMs that works efficiently across various input and output modalities without external tools, relying only on the models' own internal modality features. Inspired by Determinantal Point Processes (DPPs), UMPIRE computes a novel incoherence-adjusted semantic volume of sampled MLLM responses for a given task instance, which couples semantic embedding dispersion with the quadratic entropy of the responses. This
    
[^101]: 通过可复用性与可验证性评估思维链推理

    Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability

    [https://arxiv.org/abs/2602.17544](https://arxiv.org/abs/2602.17544)

    该论文提出用发送方-接收方框架评估模型间思维链推理传递的可靠性、有用性和安全性，并引入可验证性与可纠正性两个度量指标。

    

    AI应用日益由多个模型组合而成，其中一个模型基于另一个模型产生的输出进行操作。所传递的文本还可以携带发送方模型的推理过程。思维链（Chain-of-Thought）是我们在实验中使用的此类文本的实例。我们针对这种模型间推理传递提出三个问题：推理是否被可靠地传递？它对接收方是否有用？它是否安全？我们通过“发送方-接收方”实验设置来研究这三个问题：一个发送方LLM生成推理轨迹，一个独立的接收方LLM必须基于该轨迹执行操作。两个模型都不知道彼此的身份，因此发送方无法针对特定接收方定制其推理。可靠性关注接收方能否恢复出发送方所预期的结论，我们用“可验证性”来衡量它。有用性关注所传递的推理能否帮助接收方纠正自身的错误，我们用“可纠正性”来衡量它。安全性关注所传递的推理是否使接收方……

    arXiv:2602.17544v2 Announce Type: replace  Abstract: AI applications increasingly compose multiple models, where one model acts on the output produced by another model. The transferred text can also carry the reasoning of the sender model. Chain-of-thought is our experimental instance of such text. We ask three questions about such inter-model reasoning transfer. Is the reasoning transmitted reliably? Is it useful to the receiver? Is it safe? We study these three questions with a Sender-Receiver setup. A Sender LLM produces a reasoning trace. A separate Receiver LLM must act on it. Neither model knows the identity of the other. The Sender therefore cannot tailor its reasoning to a particular Receiver. Reliability asks whether the Receiver recovers the Sender's intended conclusion. We measure it with Verifiability. Usefulness asks whether transferred reasoning helps a Receiver correct its mistakes. We measure it with Correctability. Safety asks whether transferred reasoning makes a Rece
    
[^102]: ScrapeGraphAI-100k：面向模式约束的大语言模型生成数据集

    ScrapeGraphAI-100k: Dataset for Schema-Constrained LLM Generation

    [https://arxiv.org/abs/2602.15189](https://arxiv.org/abs/2602.15189)

    本文发布了首个基于真实使用遥测数据构建的大规模模式约束抽取数据集 ScrapeGraphAI-100k，包含 93,695 条配对真实网页内容、提示词、模式与大语言模型响应及符合性标签的事件，覆盖 25,453 个独特模式，并揭示了模式复杂度增长带来的急剧失败阈值。

    

    生成符合指定 JSON 模式（schema）的输出，是现代大语言模型中工具调用、结构化抽取和知识库构建的基础。尽管这一任务至关重要，但现有的公开数据集仍然规模小、多为合成数据或仅包含纯文本，且很少将真实网页内容与实践中使用的提示词和模式配对。我们提出了 ScrapeGraphAI-100k 数据集，包含 93,695 条模式约束抽取事件，这些事件通过 ScrapeGraphAI 的选择性退出（opt-out）遥测机制于 2025 年第二至第三季度收集，并从 900 万条原始事件中按模式去重和均衡而来。该语料库涵盖 25,453 个独特模式和 15 种可识别的语言，其中英语内容占检测内容的 89%。每个实例将转换为 Markdown 格式的网页内容与提示词、模式、大语言模型响应以及基于 jsonschema-rs 的逐样本结构符合性标签配对。我们刻画了整个语料库的结构多样性，并发现随着模式复杂度的增长，模型失败率存在急剧上升的临界阈值。作为案例研究，一个 1.（原文摘要在此处截断）

    arXiv:2602.15189v3 Announce Type: replace-cross  Abstract: Producing output that conforms to a specified JSON schema underlies tool use, structured extraction, and knowledge base construction in modern large language models. Despite this centrality, public datasets for the task remain small, synthetic, or text-only, and rarely pair real page content with the prompts and schemas used in practice. We introduce ScrapeGraphAI-100k, 93,695 schema-constrained extraction events collected via opt-out ScrapeGraphAI telemetry in Q2--Q3 2025, deduplicated and balanced by schema from 9M raw events. The corpus spans 25,453 unique schemas, across 15 named languages, with English covering 89% of detected content. Each instance pairs Markdown-converted page content with a prompt, schema, LLM response, and per-example jsonschema-rs structural conformance labels. We characterize structural diversity across the corpus and identify sharp failure thresholds as schema complexity grows. As a case study, a 1.
    
[^103]: ROCKET：通过校准引导的背包增强截断实现快速优化的高效模型压缩方法

    ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression

    [https://arxiv.org/abs/2602.11008](https://arxiv.org/abs/2602.11008)

    ROCKET是一种免训练的模型压缩方法，通过将逐层压缩分配建模为多选择背包问题，并结合基于校准集的单步稀疏矩阵分解，在全局压缩预算下实现了超越现有方法的最先进压缩性能。

    

    我们提出了ROCKET，这是一种免训练的模型压缩方法，与因式分解、结构化稀疏化和动态压缩等基线方法相比，实现了最先进的性能。在全局压缩预算的约束下，ROCKET包含两项关键创新：首先，它将逐层压缩分配问题建模为一个多选择背包问题，在满足目标模型大小的前提下，为每一层选择最优压缩级别以最小化总重构误差。其次，它引入了一种受字典学习启发的单步稀疏矩阵分解方法：仅使用一个小型校准集，基于激活-权重敏感度对权重系数进行稀疏化，然后通过最小二乘法以闭式解形式更新字典，完全绕过了迭代优化、稀疏编码或反向传播。ROCKET在不同模型上持续超越现有的压缩方法。

    arXiv:2602.11008v2 Announce Type: replace-cross  Abstract: We present ROCKET, a training-free model compression method that achieves state-of-the-art performance in comparison with factorization, structured-sparsification and dynamic compression baselines. Operating under a global compression budget, ROCKET comprises two key innovations: First, it formulates layer-wise compression allocation as a multi-choice knapsack problem, selecting the optimal compression level for each layer to minimize total reconstruction error while adhering to a target model size. Second, it introduces a single-step sparse matrix factorization inspired by dictionary learning: using only a small calibration set, it sparsifies weight coefficients based on activation-weights sensitivity and then updates the dictionary in closed form via least squares bypassing iterative optimization, sparse coding, or backpropagation entirely. ROCKET consistently outperforms existing compression approaches across different model
    
[^104]: 超越轨迹相似性：基于手术方案依从性的手术规划评估

    Beyond Trajectory Similarity: Evaluating Surgical Planning by Protocol Adherence

    [https://arxiv.org/abs/2601.10455](https://arxiv.org/abs/2601.10455)

    该论文指出现有基于轨迹相似性的手术规划评估范式混淆了任务完成与方案依从性，并提出首个基于专家定义手术方案规则、直接衡量方案依从性的评估框架，证明相似性指标会系统性误判规划质量。

    

    手术规划通常通过其与单次观察到的执行轨迹的相似性来评估，这一范式借鉴了诸如Ego4D等活动预测基准。这种做法混淆了两个不同的概念——任务完成与方案依从性，并假设与一条已完成的手术轨迹相似即意味着规划有效。在手术领域，这一假设存在缺陷：临床上相关的目标是遵循手术方案而非仅仅完成任务，且参考轨迹只是单次观察到的执行过程，其本身可能就偏离了方案。因此，我们提出了一种基于专家定义的方案规则的评估框架，可直接衡量方案依从性。在该框架下，基于相似性的指标系统性地失效：它们会惩罚那些偏离参考轨迹但符合方案的规划，同时奖励那些表面上与参考轨迹相似却违反方案的规划。对Video-LLMs的进一步评估……（原文摘要至此截断）

    arXiv:2601.10455v2 Announce Type: replace  Abstract: Surgical planning is commonly evaluated by trajectory similarity to a single observed execution, a paradigm borrowed from activity anticipation benchmarks such as Ego4D. This conflates two distinct concepts: task completion and protocol adherence and assumes similarity to a completed surgical trajectory implies a valid plan. In surgery, this assumption is flawed: the clinically relevant target is protocol adherence rather than mere completion, and the reference trajectory is only one observed execution that may itself deviate from the protocol. We therefore introduce an evaluation framework grounded in expert-defined protocol rules that directly measures protocol adherence. Under this framework, similarity-based metrics systematically fail: they penalize protocol-adherent plans that deviate from the reference trajectory, while rewarding protocol-violating plans that remain superficially similar to it. Evaluating Video-LLMs further re
    
[^105]: 研究大语言模型在临床问答长上下文理解方面的能力

    Investigating LLM Capabilities on Long Context Comprehension for Clinical Question Answering

    [https://arxiv.org/abs/2510.18691](https://arxiv.org/abs/2510.18691)

    该研究系统评估了大语言模型在纵向患者电子健康记录上的长上下文临床问答能力，发现医学微调模型相比基础模型并无可衡量的优势，较小的模型记忆能力更强，纵向补充记录有助于生成式问答，而推理模型仅在宽松指标下的复杂多记录问答中获益。

    

    尽管大语言模型（LLM）在医学问答任务上报告的表现令人印象深刻，但现有评估仍过度依赖主要考察教科书知识的多项选择和抽取式设置，针对患者病历的开放式问答研究仍然有限。我们提出了一项综合性研究，在多个数据集、模型和指标上，调查了大语言模型对纵向的、以患者为中心的电子健康记录（EHR）的长上下文（LC）理解能力，涵盖抽取式、多项选择和开放式生成式问答，并在不同的上下文表述和检索增强生成（RAG）设置下进行测试。我们表明，经过医学领域微调的模型相比基础模型没有可衡量的优势，较小的模型表现出更强的记忆能力，纵向补充记录有助于生成式问答。推理模型在宽松的LLM正确性指标下的复杂多记录问答中有所提升，但在更严格的参考对齐指标下则不然（摘要原文在此处截断）。

    arXiv:2510.18691v3 Announce Type: replace  Abstract: Despite the impressive reported performance of Large Language Models (LLMs) on medical Question-Answering (QA), evaluation still overrelies on multiple-choice and extractive settings primarily assessing textbook knowledge, with limited work on open-ended QA over patient records. We present a comprehensive study investigating LLM long-context (LC) comprehension on longitudinal, patient-centric Electronic Health Records (EHRs) across multiple datasets, models and metrics, covering extractive, multiple-choice, and open-ended generative QA under different context formulation and Retrieval Augmented Generation (RAG) settings. We show that medically fine-tuned models offer no measurable benefit over base counterparts, smaller models exhibit stronger memorization, and longitudinal supplementary notes aid generative QA. Reasoning models gain on complex multi-note QA under LLM Correctness but not stricter reference-aligned metrics, while hybr
    
[^106]: HackWorld：评估计算机操作代理利用Web应用程序漏洞的能力

    HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities

    [https://arxiv.org/abs/2510.12200](https://arxiv.org/abs/2510.12200)

    提出了首个系统性评估框架HackWorld，通过视觉交互方式评估计算机操作代理利用真实Web应用程序漏洞的能力，涵盖36个跨11种框架和7种语言的真实应用。

    

    Web应用程序作为访问关键服务和敏感数据的门户，是网络攻击的主要目标。传统的渗透测试成本高昂且依赖专业知识，难以随着不断增长的Web生态系统进行规模化扩展。虽然语言模型代理在网络安全领域展现出潜力，但现代Web应用程序需要视觉理解、动态内容处理和多步骤交互能力，而只有计算机操作代理才能执行这些任务。然而，计算机操作代理通过图形界面发现和利用漏洞的能力在很大程度上尚未被探索。我们提出了HackWorld，这是首个通过视觉交互系统性评估计算机操作代理利用Web应用程序漏洞能力的框架。与经过简化的基准测试不同，HackWorld包含涵盖11种框架和7种语言的36个真实应用程序，具有注入漏洞、身份验证绕过和不安全输入处理等真实缺陷。

    arXiv:2510.12200v2 Announce Type: replace-cross  Abstract: Web applications are prime targets for cyberattacks as gateways to critical services and sensitive data. Traditional penetration testing is costly and expertise-intensive, making it difficult to scale with the growing web ecosystem. While language model agents show promise in cybersecurity, modern web applications demand visual understanding, dynamic content handling, and multi-step interactions that only computer-use agents (CUAs) can perform. Yet, their ability to discover and exploit vulnerabilities through graphical interfaces remains largely unexplored. We present HackWorld, the first framework for systematically evaluating CUAs' capabilities to exploit web application vulnerabilities via visual interaction. Unlike sanitized benchmarks, HackWorld includes 36 real-world applications across 11 frameworks and 7 languages, featuring realistic flaws such as injection vulnerabilities, authentication bypasses, and unsafe input ha
    
[^107]: BigCodeArena：通过代码执行揭示代码生成中更可靠的人类偏好

    BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution

    [https://arxiv.org/abs/2510.08697](https://arxiv.org/abs/2510.08697)

    该论文提出了 BigCodeArena，一个基于实时执行环境的开放式代码生成人类评估平台，通过让人类与 LLM 生成代码的执行过程和结果进行交互，从而揭示出更可靠的人类偏好。

    

    众包模型评估平台（如 Chatbot Arena）能够从人类视角实时评估模型响应的质量。在编程领域，人工检查大语言模型（LLM）生成内容的质量极具挑战性，因为这需要理解大段原始代码并仔细模拟代码执行。为此，我们推出了 BigCodeArena，一个开放的代码生成人类评估平台，其背后由全面且实时的执行环境支持。BigCodeArena 构建于 Chatbot Arena 之上，能够执行 LLM 生成的代码，并允许人类与执行过程和执行结果进行交互。我们在 10 个广泛使用的 LLM 上收集了超过 14,000 个以代码为中心的原始对话会话，涵盖 10 种语言和 8 类执行环境。在这些对话中，我们识别出超过 4,700 个包含成对人类偏好的多轮样本。

    arXiv:2510.08697v3 Announce Type: replace-cross  Abstract: Crowdsourced model evaluation platforms, such as Chatbot Arena, enable real-time evaluation from human perspectives to assess the quality of model responses. In the coding domain, manually examining the quality of LLM-generated content is extremely challenging, as it requires understanding long chunks of raw code and deliberately simulating code execution. To this end, we introduce BigCodeArena, an open human evaluation platform for code generation backed by a comprehensive and on-the-fly execution environment. Built on top of Chatbot Arena, BigCodeArena enables the execution of LLM-generated code and allows humans to interact with the execution process and outcomes. We collected over 14,000 raw code-centric conversation sessions across 10 widely used LLMs, spanning 10 languages and 8 types of execution environments. Among these conversations, we identified more than 4,700 multi-turn samples with pairwise human preferences. Fur
    
[^108]: 使用 CTF-Dojo 训练语言模型智能体发现漏洞

    Training Language Model Agents to Find Vulnerabilities with CTF-Dojo

    [https://arxiv.org/abs/2508.18370](https://arxiv.org/abs/2508.18370)

    提出了 CTF-Dojo——首个面向 LLM 训练的大规模可验证执行环境，包含 658 个容器化 CTF 挑战，并配套自动化流水线 CTF-Forge，可数分钟内将公开工件转化为即用型执行环境，仅需 486 条高质量轨迹即可训练出擅长漏洞发现的智能体。

    

    大型语言模型（LLM）在可执行运行时环境中训练时展现出了卓越的能力，尤其通过可验证的反馈循环在软件工程任务中表现突出。然而，可扩展且可泛化的基于执行的环境仍然稀缺，限制了训练更强机器学习智能体的进展。我们提出了 CTF-Dojo，这是首个专为训练具有可验证反馈的 LLM 而设计的大规模可执行运行时环境，包含 658 个功能完整的夺旗（CTF）式挑战，并以 Docker 容器化方式确保可重现性。为了在没有人工干预的情况下实现快速扩展，我们开发了 CTF-Forge，这是一个自动化流水线，能够在几分钟内将公开可用的工件转换为可直接使用的执行环境，省去了传统上需要数周专家配置的过程。我们仅使用来自 CTF-D（原文在此处截断）的 486 条高质量、经过执行验证的轨迹对基于 LLM 的智能体进行训练……

    arXiv:2508.18370v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) have demonstrated exceptional capabilities when trained within executable runtime environments, notably excelling at software engineering tasks through verified feedback loops. Yet, scalable and generalizable execution-grounded environments remain scarce, limiting progress in training more capable ML agents. We introduce CTF-Dojo, the first large-scale executable runtime tailored for training LLMs with verifiable feedback, featuring 658 fully functional Capture-The-Flag (CTF)-style challenges containerized in Docker with guaranteed reproducibility. To enable rapid scaling without manual intervention, we develop CTF-Forge, an automated pipeline that transforms publicly available artifacts into ready-to-use execution environments in minutes, eliminating weeks of expert configuration traditionally required. We trained LLM-based agents on just 486 high-quality, execution-verified trajectories from CTF-D
    
[^109]: Cyber-Zero：无需运行时环境训练网络安全智能体

    Cyber-Zero: Training Cybersecurity Agents without Runtime

    [https://arxiv.org/abs/2508.00910](https://arxiv.org/abs/2508.00910)

    Cyber-Zero是首个无需运行时环境的网络安全智能体训练框架，通过利用公开CTF解题报告和基于角色扮演的LLM模拟来合成高质量交互轨迹，使训练出的智能体在三个CTF基准测试上相比基线模型最高取得11.4%的绝对性能提升。

    

    大型语言模型（LLM）在软件工程任务中通过可执行运行时环境的训练已取得显著成功，尤其是在解决GitHub问题方面。然而，此类运行时环境在其他领域往往不可用，特别是在网络安全领域，挑战配置和执行环境通常是短暂的或受限的。我们提出了Cyber-Zero，这是首个无需运行时环境的框架，用于合成高质量的智能体轨迹以训练网络安全LLM。Cyber-Zero利用公开可用的CTF（夺旗赛）解题报告，并采用基于角色扮演的LLM模拟来逆向工程运行时行为，在无需实际环境的情况下生成逼真的、长时程的交互序列。使用Cyber-Zero合成的轨迹训练的LLM智能体，在三个知名CTF基准测试上相比基线模型取得了最高11.4%的绝对性能提升。

    arXiv:2508.00910v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have achieved remarkable success in software engineering tasks when trained with executable runtime environments, particularly in resolving GitHub issues. However, such runtime environments are often unavailable in other domains, especially cybersecurity, where challenge configurations and execution contexts are ephemeral or restricted. We present Cyber-Zero, the first runtime-free framework for synthesizing high-quality agent trajectories to train cybersecurity LLMs. Cyber-Zero leverages publicly available CTF writeups and employs persona-driven LLM simulation to reverse-engineer runtime behaviors and generate realistic, long-horizon interaction sequences without actual environments. Using trajectories synthesized by Cyber-Zero, we train LLM-based agents that achieve up to 11.4% absolute performance gains over baseline models on three prominent CTF benchmarks: InterCode-CTF, NYU CTF Bench, and Cybe
    
[^110]: 超越黄金标准：基于认知集成的大语言模型法官用于形式化数学推理

    Beyond Gold Standards: Epistemic Ensemble of LLM Judges for Formal Mathematical Reasoning

    [https://arxiv.org/abs/2506.10903](https://arxiv.org/abs/2506.10903)

    本文提出了一种基于认知和形式化基础的LLM法官集成方法，通过细粒度、多层次的评估标准，系统地自动评估形式化数学推理中的语句自动形式化任务，弥补了现有评估方法的粗粒度缺陷。

    

    arXiv:2506.10903v2 公告类型：替换 摘要：语句自动形式化在形式化数学推理中起着关键作用，它能够将自然语言语句自动翻译为形式语言。尽管近期利用大语言模型（LLM）的进展显示出自动形式化的强大能力，但自动评估自动形式化的方法仍未得到充分探索。将LLM作为法官提供了一种自动化此类评估的有前景方法，然而，现有方法通常采用粗粒度且通用的评估标准，这限制了其在高级形式化数学推理中的有效性，因为该领域的质量依赖于细微、多粒度的维度。在本工作中，我们向解决这一差距迈出一步，引入了一种系统化的自动方法来评估自动形式化任务。所提方法基于一个认知和形式上扎实的LLM法官集成（EFG），其定义在涵盖逻辑的评估标准上。

    arXiv:2506.10903v2 Announce Type: replace  Abstract: Statement autoformalization plays a crucial role in formal mathematical reasoning by enabling the automatic translation of natural language statements into formal languages. While recent advances using large language models (LLMs) have shown promising capability of autoformalization, methods for automatically evaluating autoformalization remain underexplored. LLM-as-a-judge presents a promising approach for automating such evaluation, however, existing methods typically employ coarse-grained and generic evaluation criteria, which limit their effectiveness for advanced formal mathematical reasoning, where quality hinges on nuanced, multi-granular dimensions. In this work, we take a step toward addressing this gap by introducing a systematic, automatic method to evaluate autoformalization tasks. The proposed method is based on an epistemically and formally grounded ensemble (EFG) of LLM judges, defined on criteria encompassing logical 
    
[^111]: 绘制哥伦比亚七十年哲学图景：对《Ideas y Valores》期刊的动态主题建模研究

    Mapping Seven Decades of Philosophy in Colombia: Dynamic Topic Modelling of Ideas y Valores

    [https://arxiv.org/abs/2412.04236](https://arxiv.org/abs/2412.04236)

    本研究通过动态主题建模技术分析哥伦比亚最具影响力的哲学期刊《Ideas y Valores》七十年来主题的演变，将哲学史的数据驱动研究拓展至哥伦比亚和拉丁美洲语境，发现价值理论、认识论和科学哲学是该地区最突出的哲学研究主题。

    

    以数据驱动的方法研究哲学已成为探索该学科历史的重要工具。然而，该领域的大多数研究都集中在特定地区和子领域的少数期刊上。我们通过应用动态主题建模技术来探索哥伦比亚和拉丁美洲哲学的历史，从而扩展了这一研究的范围。我们的研究考察了哥伦比亚哲学期刊《Ideas y Valores》，该期刊创刊于1951年，目前是该地区最具影响力的学术哲学期刊之一。通过分析该期刊历史中主题的演变，我们识别出哥伦比亚和拉丁美洲语境下哲学话语的多种趋势和特定动态。我们的研究发现，最突出的主题是价值理论（包括伦理学、政治哲学和美学）、认识论和科学哲学。我们还追踪了（原文摘要在此处截断）……

    arXiv:2412.04236v2 Announce Type: replace-cross  Abstract: Data-driven approaches to philosophy have emerged as a valuable tool for studying the history of the discipline. However, most studies in this area have focused on a limited number of journals from specific regions and subfields. We expand the scope of this research by applying dynamic topic modelling techniques to explore the history of philosophy in Colombia and Latin America. Our study examines the Colombian philosophy journal Ideas y Valores, founded in 1951 and currently one of the most influential academic philosophy journals in the region. By analyzing the evolution of topics across the journal's history, we identify various trends and specific dynamics in philosophical discourse within the Colombian and Latin American context. Our findings reveal that the most prominent topics are value theory (including ethics, political philosophy, and aesthetics), epistemology, and the philosophy of science. We also trace the evoluti
    
[^112]: LEGAL-UQA：使用英乌尔都语平行基准测试面向乌尔都语使用者的宪法问答系统

    LEGAL-UQA: Testing Constitutional Question-Answering Systems for Urdu Speakers with a Parallel English-Urdu Benchmark

    [https://arxiv.org/abs/2410.13013](https://arxiv.org/abs/2410.13013)

    提出了LEGAL-UQA，一个包含619个问答对的英乌尔都语平行基准，用于测试面向乌尔都语使用者的巴基斯坦宪法问答系统，研究发现模型排名依赖相似度度量方式、词匹配检索优于嵌入方法，且自动评判模型与人工评分的一致性有限。

    

    法律信息助手可以帮助乌尔都语使用者了解其宪法权利，但构建可靠的系统需要适合该语言的测试数据和评估方法。我们提出了LEGAL-UQA，这是一个包含619个问答对的英乌尔都语平行基准，每个问答对均与巴基斯坦宪法的段落相关联。每个示例包含两种语言的问题、参考答案和源段落。我们分别评估两个组成部分：在提供源段落的情况下回答乌尔都语问题，以及检索生成每个问题的源段落。我们还将JEV（一种用作自动评判器的决策模型）与人工评分进行比较。研究结果表明，模型排名取决于答案相似度的度量方式，而简单的词匹配方法在检索源段落方面比所测试的嵌入方法更为有效。在评估的截止阈值处，JEV与人工评分的一致性低于基线方法。

    arXiv:2410.13013v2 Announce Type: replace-cross  Abstract: Legal information assistants could help Urdu speakers understand their constitutional rights, but building reliable systems requires test data and evaluation methods suited to the language. We present LEGAL-UQA, a parallel English-Urdu benchmark of 619 question-answer pairs linked to passages from Pakistan's Constitution. Each example includes the question, reference answer, and source passage in both languages. We evaluate two components separately: answering Urdu questions when the source passage is supplied, and retrieving the passage from which each question was generated. We also compare JEV, a decision model used as an automated judge, with human grades. The findings show that model rankings depend on how answer similarity is measured, while a simple word-matching method retrieves source passages more effectively than the embedding methods tested. At the evaluated cutoff, JEV agrees with human grades less often than a bas
    
[^113]: BigCodeBench：利用多样化函数调用与复杂指令对代码生成进行基准测试

    BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions

    [https://arxiv.org/abs/2406.15877](https://arxiv.org/abs/2406.15877)

    BigCodeBench是一个新的代码生成基准测试，通过要求模型调用多样化函数库并准确理解复杂指令，来评估大语言模型解决实际任务的能力。

    

    大语言模型（LLM）通过Python代码的最新进展极大地赋能了任务自动化，其任务范围涵盖从软件工程开发到通用推理。虽然现有基准测试表明LLM能够像人类开发者一样使用程序解决问题，但其大多数评估仅限于简短且独立的算法任务或单一函数调用。解决具有挑战性和实用性的任务，需要具备将多样化函数调用作为工具的能力，以高效实现数据分析和网页开发等功能。此外，使用多个工具解决任务还需要通过准确理解复杂指令来进行组合推理。同时满足这两个特性对LLM而言是一项巨大挑战。为了评估LLM通过程序解决具有挑战性和实用性任务的能力，我们推出了BigCodeBench这一基准测试……

    arXiv:2406.15877v5 Announce Type: replace-cross  Abstract: Task automation has been greatly empowered by the recent advances in Large Language Models (LLMs) via Python code, where the tasks ranging from software engineering development to general-purpose reasoning. While current benchmarks have shown that LLMs can solve tasks using programs like human developers, the majority of their evaluations are limited to short and self-contained algorithmic tasks or standalone function calls. Solving challenging and practical tasks requires the capability of utilizing diverse function calls as tools to efficiently implement functionalities like data analysis and web development. In addition, using multiple tools to solve a task needs compositional reasoning by accurately understanding complex instructions. Fulfilling both of these characteristics can pose a great challenge for LLMs.To assess how well LLMs can solve challenging and practical tasks via programs, we introduce BigCodeBench, a benchm
    
[^114]: Astraios：参数高效的指令微调代码大语言模型

    Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models

    [https://arxiv.org/abs/2401.00788](https://arxiv.org/abs/2401.00788)

    该论文提出了Astraios套件——包含28个采用7种微调方法、4种规模（最大160亿参数）指令微调的代码大语言模型，系统研究发现全参数微调总体性能最佳，LoRA通常提供最有利的成本-性能权衡，且PEFT方法的效果随模型规模显著变化。

    

    大语言模型（LLM）全参数微调（FFT）的高昂成本催生了一系列参数高效微调（PEFT）方法。然而，目前尚不清楚哪些方法在不同模型规模下能够提供最佳的成本-性能权衡。我们提出了Astraios，这是一个包含28个经过指令微调的OctoCoder模型的套件，采用了7种微调方法和4种模型规模，最大参数量达160亿。通过对涵盖代码理解和代码生成任务的5个任务和8个不同数据集的调查研究，我们发现FFT通常在所有模型规模下都能带来最佳的下游性能，而PEFT方法的有效性则随模型规模的不同而有显著差异。LoRA通常在成本和性能之间提供最有利的权衡。进一步研究这些方法对模型鲁棒性和代码安全性的影响后发现，较大的模型往往表现出较低的鲁棒性。

    arXiv:2401.00788v2 Announce Type: replace-cross  Abstract: The high cost of full-parameter fine-tuning (FFT) of Large Language Models (LLMs) has led to a series of parameter-efficient fine-tuning (PEFT) methods. However, it remains unclear which methods provide the best cost-performance trade-off at different model scales. We introduce Astraios, a suite of 28 instruction-tuned OctoCoder models using 7 tuning methods and 4 model sizes up to 16 billion parameters. Through investigations across 5 tasks and 8 different datasets encompassing both code comprehension and code generation tasks, we find that FFT generally leads to the best downstream performance across all scales, and PEFT methods differ significantly in their efficacy based on the model scale. LoRA usually offers the most favorable trade-off between cost and performance. Further investigation into the effects of these methods on both model robustness and code security reveals that larger models tend to demonstrate reduced robu
    
[^115]: 基于大型预训练语言模型的提示式语义解析的鲁棒性研究：Codex实证分析

    On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model: An Empirical Study on Codex

    [https://arxiv.org/abs/2301.12868](https://arxiv.org/abs/2301.12868)

    本文首次对基于提示的大型代码语言模型（Codex）在语义解析任务中的对抗鲁棒性进行了实证研究。

    

    语义解析是一种旨在构建自然语言问题含义的结构化表示的技术。近年来，在代码上训练的少样本语言模型在生成这些结构化表示方面展现出优于传统单模态语言模型（即在下游任务上训练的模型）的性能。尽管取得了这些进展，现有的微调神经语义解析器仍然容易受到针对自然语言输入的对抗性攻击。虽然已有研究表明，较小的语义解析器的鲁棒性可以通过对抗训练来增强，但这种方法对于现实场景中的大语言模型而言并不可行，因为它既需要大量的计算资源，又需要对域内语义解析数据进行昂贵的人工标注。本文首次对基于提示的大型代码语言模型的对抗鲁棒性进行了实证研究。

    arXiv:2301.12868v4 Announce Type: replace  Abstract: Semantic parsing is a technique aimed at constructing a structured representation of the meaning of a natural-language question. Recent advancements in few-shot language models trained on code have demonstrated superior performance in generating these representations compared to traditional unimodal language models, which are trained on downstream tasks. Despite these advancements, existing fine-tuned neural semantic parsers are susceptible to adversarial attacks on natural-language inputs. While it has been established that the robustness of smaller semantic parsers can be enhanced through adversarial training, this approach is not feasible for large language models in real-world scenarios, as it requires both substantial computational resources and expensive human annotation on in-domain semantic parsing data. This paper presents the first empirical study on the adversarial robustness of a large prompt-based language model of code,
    
[^116]: 假新闻检测器对大型语言模型生成的文本存在偏见

    Fake News Detectors are Biased against Texts Generated by Large Language Models. (arXiv:2309.08674v1 [cs.CL])

    [http://arxiv.org/abs/2309.08674](http://arxiv.org/abs/2309.08674)

    假新闻检测器倾向于将大型语言模型生成的内容标记为假新闻，而将人工编写的假新闻误分类为真实，我们提出了一种通过敌对训练和LLM改写的真实新闻等方法来解决这个问题，并取得了显著的改进。

    

    假新闻的传播已经成为一个重要挑战，损害了信任并对社会构成威胁。在大型语言模型（LLM）的时代，生成可信的假内容的能力加剧了这些担忧。在本研究中，我们提出了一种新的范式来评估在涉及人工编写和LLM生成的错误信息的情况下的假新闻检测器。有趣的是，我们的发现揭示了许多现有检测器存在显著的偏见：它们更容易将LLM生成的内容标记为假新闻，同时常常将人工编写的假新闻误分类为真实。这种意外的偏见似乎源自LLM输出固有的不同语言模式。为了解决这个问题，我们引入了一种使用LLM改写的真实新闻进行敌对训练的缓解策略。结果模型在人工和LLM生成的新闻的检测准确性方面均有显著提升。为了进一步促进这个领域的研究，我们发布了两个全面的...

    The spread of fake news has emerged as a critical challenge, undermining trust and posing threats to society. In the era of Large Language Models (LLMs), the capability to generate believable fake content has intensified these concerns. In this study, we present a novel paradigm to evaluate fake news detectors in scenarios involving both human-written and LLM-generated misinformation. Intriguingly, our findings reveal a significant bias in many existing detectors: they are more prone to flagging LLM-generated content as fake news while often misclassifying human-written fake news as genuine. This unexpected bias appears to arise from distinct linguistic patterns inherent to LLM outputs. To address this, we introduce a mitigation strategy that leverages adversarial training with LLM-paraphrased genuine news. The resulting model yielded marked improvements in detection accuracy for both human and LLM-generated news. To further catalyze research in this domain, we release two comprehensiv
    
[^117]: 重新思考机器翻译评估的往返翻译方法

    Rethinking Round-Trip Translation for Machine Translation Evaluation. (arXiv:2209.07351v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2209.07351](http://arxiv.org/abs/2209.07351)

    本文研究了往返翻译方法在机器翻译评估中的应用，发现去除统计机器翻译中的复制机制后，往返翻译可以恰当地反映前向翻译的性能，该方法可以在低资源语言的情况下进行自动评估。

    

    在低资源语言翻译的自动评估中存在平行语料库的不足。往返翻译可以作为一种巧妙且简单的技术来缓解平行评估语料库的需求。然而，在统计机器翻译 (SMT) 时代，前向翻译和往返翻译的评估得分之间存在着模糊的相关性观察。在本文中，我们报告了一个惊人的发现：往返翻译可以用于自动评估而不需要参考文献。首先，我们重新审视了 SMT 评估中的往返翻译，揭示了其长期存在的误解实质上是由复制机制引起的。在去除SMT中的复制机制后，往返翻译评分可以恰当地反映前向翻译的性能。然后，我们证明了修正是过期的，因为往返翻译可以从多个机器翻译评估任务中受益。

    Automatic evaluation on low-resource language translation suffers from a deficiency of parallel corpora. Round-trip translation could be served as a clever and straightforward technique to alleviate the requirement of the parallel evaluation corpus. However, there was an observation of obscure correlations between the evaluation scores by forward and round-trip translations in the era of statistical machine translation (SMT). In this paper, we report the surprising finding that round-trip translation can be used for automatic evaluation without the references. Firstly, our revisit on the round-trip translation in SMT evaluation unveils that its long-standing misunderstanding is essentially caused by copying mechanism. After removing copying mechanism in SMT, round-trip translation scores can appropriately reflect the forward translation performance. Then, we demonstrate the rectification is overdue as round-trip translation could benefit multiple machine translation evaluation tasks. T
    

