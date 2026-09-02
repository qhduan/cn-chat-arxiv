# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond Scores: Understanding LLM-as-a-Judge Mechanisms in Summarization Evaluation](https://arxiv.org/abs/2609.01604) | 该论文通过八种攻击扰动分类法与因果追踪、注意力头敲除等可解释性技术，首次从机制层面揭示LLM评估器（Themis与Prometheus）在摘要评分时采用两阶段内部流程：第15层以下注意力执行局部错误比较并路由信号，其上由MLP级联完成信号整合。 |
| [^2] | [Efficient SWE Agent Benchmarking via Trajectory-Aware Evaluation](https://arxiv.org/abs/2609.01603) | 提出PTA-IRT框架，将历史执行轨迹作为特权信息融合过程与结果信号，在低校准预算下更准确地恢复软件工程智能体的完整基准分数与排名。 |
| [^3] | [Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation](https://arxiv.org/abs/2609.01601) | 该论文提出ACToR，通过识别LLM自回归代码生成过程中容易出错的关键token位置，并自适应地为这些位置检索细粒度的仓库上下文，从而提升仓库级代码生成的功能正确性。 |
| [^4] | [CordisBench: Can Language Models Reason About Component Lifecycles in Dynamic Agent Harnesses?](https://arxiv.org/abs/2609.01600) | 该论文提出了 CordisBench——一个包含 1,200 道题目的基准，用于评估语言模型在动态智能体框架中对组件依赖与清理等生命周期问题的推理能力，发现模型在小规模系统上表现良好，但随相关交互数量增多可靠性显著下降。 |
| [^5] | [The Rise of Verbal Reinforcement Learning](https://arxiv.org/abs/2609.01597) | 本文首次对“言语强化学习”这一新兴范式进行了统一阐述，根据言语反馈生效的时机与作用对象，将其系统归纳为语言作为基础定位信号、语言作为审慎反馈以及语言作为学习信号三大支柱。 |
| [^6] | [StudentSim: Training LLM-based Student Simulators](https://arxiv.org/abs/2609.01591) | 提出StudentSim训练框架，通过“汇总训练+逐学生专项化”的方法，将稀疏的个体学生数据转化为既能如实模拟学生真实回答、又能在导师指导下更新能力的个性化LLM学生模拟器，并配套发布覆盖国际象棋、英语二语写作和数学共60名学生的标准化评测协议StudentSimEval。 |
| [^7] | [Designing Proactive Thought Partners for Writing](https://arxiv.org/abs/2609.01588) | 本文提出并探索了“主动式思维伙伴”的设计空间——一种能在写作过程中主动提供可定制高层次认知支持的AI智能体，通过一周的部署研究发现用户会以前瞻性规划配置支持、将建议用于创意生成与自我监控，并重视轻量级的视觉呈现。 |
| [^8] | [The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally](https://arxiv.org/abs/2609.01587) | 该研究通过因果混合精度干预实验发现，LLM的量化损伤是弥散分布的而非集中于特定电路、计算位置或权重统计，因此在匹配精度预算下，将额外比特全局用于更精细的量化粒度比局部修复少数层更有效。 |
| [^9] | [Closing Cost-Quality Gap in Document VLMs: Difficulty-Aware Data Curation and Quality-Adjusted Deployment Economics](https://arxiv.org/abs/2609.01575) | 该论文提出了一个已部署的文档理解系统，基于混合专家架构的VLM（35B总参数、3B激活参数），通过难度感知数据筛选流水线进行微调，在单张H100上即可运行，性能超越大至一个数量级的可部署基线，并将预期成本较人工标注降低80%以上。 |
| [^10] | [Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs](https://arxiv.org/abs/2609.01573) | 该论文提出“近最优区域”框架来分配SFT-RL标注预算，发现该区域宽广且随模型规模增大而扩大，并能从小型代理模型可靠迁移到大型目标模型，因此小规模代理实验即可替代在大模型上的穷尽式预算搜索。 |
| [^11] | [From Production Traffic to Post-Training: Building a Self-Hosted LLM That Covers the Corporate Request Mix](https://arxiv.org/abs/2609.01572) | 本文针对企业自托管LLM因多模型并存导致的GPU资源碎片化问题，沿指令遵循、函数调用和内部任务分布三个维度分别训练GRPO专家模型并通过两阶段SLERP合并，成功将200多个内部应用的流量整合到单一自托管模型上。 |
| [^12] | [Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers](https://arxiv.org/abs/2609.01567) | 该论文提出SAGE框架，仅在智能体不确定时才查询昂贵的视觉语言模型教师，并利用环境优势对教师建议进行加权蒸馏，从而训练出无需教师引导即可自主行动的轻量级强化学习策略。 |
| [^13] | [From Confusion to Clarity: Confusion-Aware Retrieval and Knowledge Injection for Text Classification](https://arxiv.org/abs/2609.01564) | 该论文提出一个无需微调的框架，通过识别模型易混淆的标签对、扩充候选集并生成针对性的区分规则注入知识，帮助大语言模型在语义相似标签的文本分类任务中做出正确选择，且这些规则还可迁移到更小、成本更低的模型上。 |
| [^14] | [A systematic Approach to constructing a Chance-and-Risk Matrix for Semiconductor Supply Chains](https://arxiv.org/abs/2609.01563) | 该论文提出了一种端到端的自动化流水线，利用大语言模型从半导体企业的公开披露文件中提取风险与机会，构建知识图谱并通过三层排序机制生成供应链机会与风险矩阵，独立验证显示92.6%的条目有效且排序结果与专家判断高度一致。 |
| [^15] | [SDARE-Bench: Evaluating Large Language Models on Conversational Stigma Detection and Response in Dyadic and Group Dialogue](https://arxiv.org/abs/2609.01548) | 本研究提出了首个基于情境的基准测试SDARE-Bench，用于评估大语言模型在二元与群体对话中的污名检测和开放式回应生成能力，发现模型对污名的识别能力较差，且在群体对话中污名表达更多、抵制更弱、建议更不切实际。 |
| [^16] | [Knowledge Distillation During Mid-Training Favors Reasoning over Factual Recall](https://arxiv.org/abs/2609.01532) | 该研究发现前向KL知识蒸馏在预训练阶段能同时提升推理与事实记忆能力，但在中期训练阶段会减缓事实记忆的习得而持续提升推理能力，这种阶段依赖性源于教师置信度在不同数据领域的不对称以及学生模型知识状态的演化。 |
| [^17] | [GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions](https://arxiv.org/abs/2609.01491) | 本文提出GlossoGen平台，通过SaveVeyru压力沟通场景证实LLM多智能体之间会涌现语言演化，产生的语言具有组合性和形态生成性但人类无法理解，并发现效率压力、模型能力和“事后复盘”阶段是语言演化的关键条件。 |
| [^18] | [AutoConcept: Training-Free Concept-Guided Reranking for Metadata-Available Composed Image Retrieval](https://arxiv.org/abs/2609.01456) | 提出AutoConcept，一种无需训练的概念引导重排序方法，通过将概念证据转化为可解释的结构化记忆并结合推理时校准，在元数据可用的组合图像检索中显著提升早期排名表现。 |
| [^19] | [HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?](https://arxiv.org/abs/2609.01437) | 提出HarnessDev基准，将评估单元从任务输出转移到可运行的基础设施，考察大语言模型能否从最小种子出发创建完整的智能体运行框架，并利用下游执行反馈迭代演化该框架以提升基准性能。 |
| [^20] | [Citing Less Critically: LLMs Reshape the Rhetoric and Reach of Scientific Citation](https://arxiv.org/abs/2609.01432) | 本文提出掩码引用任务构建反事实引用语料库，对比发现大语言模型的引用明显比人类更少批判性，且在引用对象与修辞意图上呈现系统性偏移，表明LLM正在重塑科学引用的修辞方式与学术影响力分布。 |
| [^21] | [From Rollouts to Recipes: Self-Contained Post-Training for LLMs](https://arxiv.org/abs/2609.01422) | 提出Self-Routing框架，利用模型自身采样轨迹的正确性与置信度将每个样本自适应路由到GRPO、自蒸馏、正则化或跳过等不同训练方法，无需外部教师和额外标注即可持续提升大语言模型的数学推理能力。 |
| [^22] | [EdiTikZ: Scientific Figure Editing from Revision Trajectories](https://arxiv.org/abs/2609.01409) | 该论文提出 DaEdiTikZ——首个从 arXiv、GitHub 和 TeX SE 的自然修订轨迹中挖掘的大规模科学图表编辑数据集（包含 39.1 万个 TikZ 编辑对和 78.1 万条推断的编辑指令），并配套构建了人工精修基准 DaEdiTikZ-Bench，以自然修订轨迹作为可扩展的监督信号来训练科学图表编辑模型。 |
| [^23] | [When Tokenization is Secretly Output Supervision](https://arxiv.org/abs/2609.01386) | 该论文提出在自回归模型中分词粒度实质上是一种输出监督信号：通过解耦实验证明，输出分词而非输入分词主导了任务性能、训练动态和模型内部表示的差异，因此模型间的比较可能部分反映的是任务定义的不同而非模型能力差异。 |
| [^24] | [InSight: A Benchmark for Agentic Claim Verification in Interactive Visualizations](https://arxiv.org/abs/2609.01383) | 本文提出InSight基准，包含21,349条源自人工分析叙述并嵌入完全交互式网页环境的声明，用于评估智能体在动态交互式可视化中主动探索证据并验证声明真伪（支持、驳斥或无法验证）的能力。 |
| [^25] | [Polish ModernBERT: The Long and Short of Polish Language Understanding](https://arxiv.org/abs/2609.01379) | 本文推出Polish ModernBERT——一族支持8K长上下文的波兰语编码器模型，在30个任务上取得波兰语编码器中的最佳整体性能，并在长上下文任务上显著超越RoBERTa基线。 |
| [^26] | [IntroConformal: Conformal Factuality Guarantees for Large Vision-Language Models via Introspective Signals](https://arxiv.org/abs/2609.01375) | 提出了IntroConformal框架，通过利用模型自身的内省信号（隐状态的逐层语义稳定性和自我验证概率），以免训练的保形风险控制方式为大视觉语言模型的生成内容提供有限样本、无分布假设的事实性保证。 |
| [^27] | [Behaviorally Effective LoRA Writes Are Sparse and Structured](https://arxiv.org/abs/2609.01374) | 本文提出Learned-Basis LoRA方法，通过将无约束适配器的写入列转换为冻结的正交基底并在其中继续训练，揭示出真正承载模型行为的LoRA写入是稀疏、结构化且高度集中的，而非均匀分布于整个低秩参数空间。 |
| [^28] | [How Correct Is Your Answer? A Semantic Correctness Framework for Open QA Evaluation](https://arxiv.org/abs/2609.01369) | 该论文提出了一个开放式问答答案评估的语义正确性框架，包含八个有序类别的语义分类体系、8.8千样本的CAP-Correctness基准数据集以及用于NLI训练的CAP-Statements陈述转换数据集，解决了现有评估指标无法区分不同类型答案错误的局限。 |
| [^29] | [Investigating Linear Probe Robustness to Linguistic Register, Medical Specialty, and Corpus Shifts in Medical QA](https://arxiv.org/abs/2609.01361) | 该论文构建了一个可独立操控写作语域、医学专科和语料库三类变化的医学问答基准，以系统性探究大语言模型中线性探针（真值方向检测）对不同输入偏移的鲁棒性。 |
| [^30] | [Separating Syntax from Language: A Mechanistic Account of Translation in Multilingual LLMs](https://arxiv.org/abs/2609.01356) | 本研究通过受控多语言数据集与因果干预、探测实验，首次揭示多语言大模型的翻译过程比以往认为的更加模块化：输出语言的生成可进一步解耦为独立的句法（语序）构建过程与表层语言实现过程，且模型会先确定目标语言的语序，再生成其表层语言形式。 |
| [^31] | [Where the Verifier Fails: A Category-Level Audit of Reward Signals in RLVR](https://arxiv.org/abs/2609.01354) | 该论文将变异测试从模型转向验证器本身，通过构造保证数学等价的答案变体，在超过30万个判定上对四个主流验证器进行了类别级审计，发现相同输入下验证器的自我验证率相差高达41.3个百分点，揭示了RLVR奖励信号中的系统性假阴性问题。 |
| [^32] | [CHARM: Character Hallucination for Multicultural Role Play Benchmark](https://arxiv.org/abs/2609.01352) | CHARM是一个涵盖五大文化语言区域40个角色的多文化角色扮演基准，创新性地将角色幻觉拆分为“边界意识”与“边界遵守”两个独立阶段进行评估，从而更精细地定位大语言模型角色扮演中幻觉错误的来源。 |
| [^33] | [Probing Factual Knowledge Transfer with Training Data Interventions](https://arxiv.org/abs/2609.01341) | 该论文提出了一种基于干预的评估框架并构建SIFT数据集，通过从波斯语训练数据中系统性移除特定事实来检验多语言模型的知识跨语言迁移能力，发现英语预训练中习得的事实知识向波斯语的迁移非常有限。 |
| [^34] | [VerTox: Verifiable Reward-Guided Corpus Poisoning Against Neural Ranking Models](https://arxiv.org/abs/2609.01325) | 本文提出VerTox，首个将语料库投毒攻击形式化为可验证奖励引导强化学习问题的框架，通过将排序扭曲与事实性破坏耦合的奖励设计将小型LLM微调为对抗性文档生成器，对神经排序模型实现了接近完美的攻击成功率。 |
| [^35] | [Exploring Sparse Autoencoders in Text-Based Causal Confounding Adjustment](https://arxiv.org/abs/2609.01322) | 该论文提出一种基于稀疏自编码器（SAE）的新颖因果调整流程，通过条件独立性检验迭代选取最小特征集合，解决了文本表示在保留混杂变量与满足有限样本重叠条件之间的权衡，在半合成评估中实现了比替代表示更低的偏差和更高的覆盖率。 |
| [^36] | [Reliability Challenges in Diffusion Vision-Language Models](https://arxiv.org/abs/2609.01318) | 本文首次系统性评估了扩散式大型视觉语言模型的可靠性，发现尽管其幻觉率与自回归模型相当，但存在“否”偏置、语言质量下降、对代表性不足种族群体的准确率崩溃及反向性别偏置、以及由长度先验导致的多项选择准确率崩溃等严重可靠性问题。 |
| [^37] | [MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval](https://arxiv.org/abs/2609.01316) | MIDR是一个无需训练的富化增强索引框架，通过在索引阶段利用多模态大语言模型将文档页面转换为经验证的文本字段，将多模态推理从查询时转移到索引时，在ViDoRe V3上相比BM25相对提升23.0%，性能可与ColQwen2.5媲美。 |
| [^38] | [Explore Before Committing: Hypothesis-Guided Search for Deep Research Agents](https://arxiv.org/abs/2609.01294) | 针对深度研究智能体过早沿单一路径搜索而锁定错误方向的问题，提出HypoSearch方法：先生成轻量级假设作为软性搜索提示，再通过有限的独立分支并行探索，并在比较分支证据后才做出决策，从而提升复杂问题回答的可靠性。 |
| [^39] | [Some Emotions Run Deeper: Layer-wise Probing and Causal Intervention in Large Language Models](https://arxiv.org/abs/2609.01279) | 该研究结合逐层探测与因果干预，在三个情感显式程度不同的语料库和八个大语言模型上发现，情感在模型中的可读取深度随文本来源系统性变化——越隐含、越依赖语境的情感需要越深的层才能读取，说明情感表达深度同时取决于文本来源与模型本身。 |
| [^40] | [From Base Rollouts to RL Reasoning: A Budgeted Search Perspective](https://arxiv.org/abs/2609.01274) | 提出统一解码框架（UDF）将各类解码与搜索方法统一到共享预算空间，发现强化学习的推理增益可由基座模型运行点的结构化预算转换路径近似恢复，表明RL主要是将采样分布转向基座模型本可达但很少采样的轨迹，而非创造全新的推理能力。 |
| [^41] | [What Does an Agentic Software Engineering Benchmark Measure? Profiling Task Demands and Agent Behaviour Beyond What Category Labels Reveal](https://arxiv.org/abs/2609.01271) | 本文提出Spread-Novelty-Centrality（SNC）三轴画像方法来刻画仓库级编码任务的真实需求，发现类别标签是任务需求的不可靠代理指标，且智能体行为轨迹能揭示人工标准答案无法反映的任务需求信息。 |
| [^42] | [Ready to Speak: Aligning LLMs for TTS-Friendly Text Generation](https://arxiv.org/abs/2609.01246) | 本文将“让大语言模型直接生成适合语音合成的文本”构建为偏好对齐问题，引入了 CORA 和 Recipe 两个偏好数据集以及结合启发式指标、TTS→ASR 评估流程和人类 MUSHRA 听力研究的评测体系，并比较了基于可解释特征的 FaST 框架与黑盒奖励模型方法。 |
| [^43] | [Post-Training Science for Supervised Fine-Tuning](https://arxiv.org/abs/2609.01244) | 本文通过每次只改变一个变量的统一受控扫描实验，系统测量了监督微调中学习率、批大小、LoRA与全量微调等关键决策在Qwen3与Llama两类模型（稠密与混合专家架构）以及四个真实客户数据集上的表现，将SFT超参数选择从经验摸索转变为可复现的科学测量。 |
| [^44] | [Towards AI-Assisted Clinical Trial Matching: Practical Considerations, Multicenter Evaluation, and Real-World Deployment](https://arxiv.org/abs/2609.01202) | 本文提出面向真实世界部署的AI临床试验推荐系统TrialGPT 2.0，它不仅评估患者资格，还结合患者临床需求和本地工作流优先级筛选值得进一步考虑的试验，并提供了结构化的可审查解释，在政府、学术癌症中心等多种肿瘤学场景中完成了回顾性与前瞻性多中心评估。 |
| [^45] | [FinLifeBench: Exhaustive Life-Event History and Financial-State Reconstruction from Longitudinal Banking Dialogue](https://arxiv.org/abs/2609.01198) | 提出FinLifeBench基准，基于6,000个韩语银行对话会话，评估大语言模型在穷尽式重建客户人生事件历史与34维财务状态方面的长程记忆能力，发现随会话累积事件召回率显著下降（0.591降至0.445），且错误主要源于事件遗漏。 |
| [^46] | [CaRL-EM: Cost-Aware Reinforcement Learning for Entity Matching with LLMs](https://arxiv.org/abs/2609.01195) | 该论文提出CaRL-EM，一个成本感知的强化学习控制器，通过在多候选实体匹配中自适应地选择LLM操作符与模型容量，来优化质量与成本的权衡目标。 |
| [^47] | [PersuaRL: Reinforcement Learning-Driven Multi-Expert Selection for Persuasive Dialogue Generation in Insurance](https://arxiv.org/abs/2609.01188) | 该论文提出了保险领域说服性对话数据集InsureDial，并构建了基于强化学习的多专家选择框架PersuaRL，以提升大语言模型驱动的对话智能体在保险场景中生成有说服力对话的能力。 |
| [^48] | [LLMPEDIA: Browsing, Verifying, and Comparing the Parametric Encyclopedic Knowledge of LLMs](https://arxiv.org/abs/2609.01182) | 提出LLMPEDIA系统，从三个大语言模型的参数记忆中递归生成约130万篇百科文章并对照维基百科和网络资源验证，发现模型参数知识的真实率仅为68.4%（比MMLU低超过21个百分点），其中30.5%的断言无法被任何现有证据裁定，从而揭示了固定基准测试的可得性偏差。 |
| [^49] | [Subword Segmental BabyLMs: Learning to Tokenise for Sample-Efficient Pretraining](https://arxiv.org/abs/2609.01151) | 该论文提出了两个在预训练过程中联合学习分词的子词分段语言模型SubSegGPT和SubSegDeBERTa，并在2026年BabyLM挑战赛中实现了样本高效的性能提升。 |
| [^50] | [On the Design Fundamentals of Pixel Text Representation Learning](https://arxiv.org/abs/2609.01147) | 本文通过系统性消融实验提出了鲁棒像素文本表示学习的四大设计原则（可变图像分辨率与字体大小、自然图像-文本对、布局感知渲染、两阶段多语言课程），并据此训练出原生分辨率的视觉文本模型 Pixel Linguist II。 |
| [^51] | [Does task decomposition improve automatic NLG evaluation?](https://arxiv.org/abs/2609.01139) | 本研究通过系统性实验发现，任务分解并不能真正提升LLM-as-a-judge的NLG评估性能，先前报告的性能提升实际源于使用人工标注作为训练数据，且在有人工标注时，无需分解的LLMaJ即可达到与人工标注者相当的水平。 |
| [^52] | [Overfitting Mitigation via Singular Value Decomposition in Minimum Bayes Risk Decoding](https://arxiv.org/abs/2609.01135) | 本文提出SVD-MBR方法，通过奇异值分解对最小贝叶斯风险解码中的成对效用矩阵进行低秩近似去噪，有效缓解度量过拟合并显著提升泛化性能。 |
| [^53] | [Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs](https://arxiv.org/abs/2609.01117) | 该论文提出潜在循环思维（LRT）方法，通过保持大语言模型冻结并引入一个微型循环推理器在连续潜在空间中多步迭代精炼潜在思维向量来进行推理，将计算深度与模型规模解耦，从而规避了思维链推理中误差传播以及需要可模仿轨迹的固有局限。 |
| [^54] | [EDRAC: Benchmarking Arabic Dialect Reading Comprehension](https://arxiv.org/abs/2609.01113) | EDRAC是首个面向阿拉伯语方言机器阅读理解与生成式问答的大规模基准，涵盖埃及、摩洛哥、阿联酋、叙利亚和沙特五种主要方言，包含499篇自然口语段落和通过人-大语言模型协作流水线生成的4,977个问答对，并以此评测了阿拉伯语和多语言大语言模型的表现。 |
| [^55] | [ClinTraceBench: Source-Verifiable Longitudinal Clinical Reasoning over EHR-Derived Dialogues](https://arxiv.org/abs/2609.01111) | 提出了ClinTraceBench——一个基于电子健康记录对话、具备事件级来源可验证性的纵向临床推理基准，通过385个已验证对话、九任务分类体系和约20万条预测，系统评估了八种患者历史表示策略在四个大模型上保留纵向临床信号的能力。 |
| [^56] | [Hints Help But Do They Teach? Evaluating Skills Transfer in Code Generation](https://arxiv.org/abs/2609.01106) | 研究发现，提示对失败代码生成的“挽救”效果大多可通过无提示的重复采样复现，且相关与无关提示共享同一激活方向，表明提示更多是引导模型已有能力而非传授新技能。 |
| [^57] | [When Modality Gap Reduction Fails: Prediction-Level Hubness in CLIP](https://arxiv.org/abs/2609.01103) | 本文揭示了在CLIP中缩小模态差距虽然能减小平均图像-文本距离，但可能改变类别间的相对决策结构，导致预测过度集中于少数类别（即预测级枢纽性），反而损害零样本分类准确率。 |
| [^58] | [Beyond Magnitude: Contrastive Routing for Modular Mixture-of-Experts](https://arxiv.org/abs/2609.01100) | 提出 CoRM 对比路由机制，通过将每个标记与层隐藏状态的指数移动平均进行对比而非基于绝对幅值进行路由，使路由信号集中于低维可分子空间，从而提升 MoE 专家与语言结构的对齐程度，并将零样本准确率提高最高 1.77 个百分点。 |
| [^59] | [StateSwap: Probing Support-Elimination Hidden States in Multiple-Choice Questions](https://arxiv.org/abs/2609.01081) | 该论文提出StateSwap方法，通过添加特殊标记[STATE]来探测并交换多选题在“支持型”与“排除型”两种表述下诱导出的隐藏状态激活，证明两种框架在模型中间层产生可分离的内部表示，且交换这些激活可因果性地改变预测结果并提高跨框架答案一致性。 |
| [^60] | [Post-hoc Alignment of LLM-judges to Human Judgment Distribution](https://arxiv.org/abs/2609.01073) | 针对大语言模型评判者在预测人类标签分布（软标签）时表现不佳的问题，提出了一种轻量级的熵感知事后对齐方法NAPHA，将大语言模型的输出分布与人类判断分布对齐。 |
| [^61] | [OUTLETS: Output-Length Prediction from Speculative Decoding Backbones](https://arxiv.org/abs/2609.01068) | 该论文发现投机解码框架中草稿解码器的潜在表示蕴含着可预测生成长度的信号，并提出OUTLETS方法，将投机解码主干重新用作轨迹感知的输出长度预测器，从而在几乎不增加额外开销的情况下改进大语言模型服务的资源供给与集群调度。 |
| [^62] | [WorldBench: Culturally Grounded Benchmark for Multilingual Agents](https://arxiv.org/abs/2609.01056) | WorldBench是一个涵盖七种语言、八种文化、包含1,600个真实日常任务的多语言智能体基准，并引入约束任务成功率（CTS）指标，以全面评估LLM智能体在真实文化扎根场景中的跨语言多步骤任务执行能力。 |
| [^63] | [Lagged Coupling: Internal Representations Become Readable Before They Become Causal](https://arxiv.org/abs/2609.01048) | 该研究在 Pythia 全系列模型中发现“滞后耦合”现象：线性探针能极早读出内部表征，但利用这些方向进行引导干预却几乎无效，且“可读但不可因果干预”的滞后并不随模型规模增大而缩小。 |
| [^64] | [PCoMoE: Shifting MoE Inference from Monolithic Expert Selection to Fine-Grained Path Composition](https://arxiv.org/abs/2609.01024) | PCoMoE提出了一种路径组合式执行框架，将MoE推理从粗粒度的整专家选择转变为细粒度的路径组合，通过路径级计算形式化、兼容性感知的逐层剪枝策略和硬件友好的执行引擎，在严格受限的开销下挖掘专家内部的细粒度计算冗余。 |
| [^65] | [Phrase-Localized Language-Contrastive Guidance: Training-Free Localized Accent Control for Code-Switching Text-to-Speech](https://arxiv.org/abs/2609.01016) | 提出了一种免训练的推理框架LCG，通过短语定位的语言对比引导和自注意力探测技术，无需外部对齐或微调即可为语码转换文本转语音中的外语短语恢复母语口音。 |
| [^66] | [SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models](https://arxiv.org/abs/2609.01004) | 提出无需训练的视觉token剪枝框架SinkPruner，通过过滤高度冗余的高范数离群token并缓解注意力汇聚现象，在保持多模态理解能力的同时实现高效的多模态大语言模型推理。 |
| [^67] | [Right Frame, Wrong Rule: Cultural Cues Expose the Financial Knowledge Gap They Were Meant to Close](https://arxiv.org/abs/2609.00999) | 该论文提出“规范多元性”这一新评估设定，通过将框架选择与框架内正确性分离，揭示了“刻板印象陷阱”——文化线索虽能引导大模型选择伊斯兰金融框架，却在框架内暴露出高达57%至66%的错误率，表明传统二选一评估会严重高估模型的文化对齐能力。 |
| [^68] | [Inspicio: Open-Vocabulary, LLM-Based Sense Retrieval for Historical Languages](https://arxiv.org/abs/2609.00998) | 提出了Inspicio，一个无需源语言词义清单的开放式词汇检索流水线，利用大语言模型生成的英文翻译、候选定义和词元，通过混合检索将历史语言文本中的词元直接链接到开放英语WordNet的同义词集。 |
| [^69] | [Disclosure-Gated User Simulation for Companion-Agent Evaluation](https://arxiv.org/abs/2609.00982) | 提出披露门控用户模拟方法，让模拟用户根据陪伴型智能体的行为决定信息披露深度，以纠正模拟用户过度配合、使被测系统仅靠提问数量即可得分的评估缺陷。 |
| [^70] | [PersianAnonymizer: Evaluating LLM-Labeled Training for Efficient NER-based Anonymization in Persian](https://arxiv.org/abs/2609.00958) | 该论文通过比较三个大语言模型为波斯语客户聊天生成标注数据来训练轻量级NER匿名化模型，发现GPT-OSS零样本标注训练的模型性能最佳，且推理速度远快于直接使用大语言模型。 |
| [^71] | [Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling](https://arxiv.org/abs/2609.00949) | 本文提出一个基于四类动作空间的诊断框架，通过引入“准确率不超过黄金动作召回率”的自揭示上界，将多轮工具调用失败分解为动作类别失准与动作执行失败两种正交模式，从而揭示开源模型总体准确率追平闭源模型的表象背后，动作类别校准才是真正的瓶颈。 |
| [^72] | [From Terminology to Diagrams: Visual-Instruction Generation for Scientific Diagram Understanding](https://arxiv.org/abs/2609.00948) | 该论文提出SciGram框架与数据集，通过科学课程术语自动生成涵盖19.4万张图表和140万条视觉指令的大规模训练数据，显著提升了视觉语言模型在科学图表理解任务上的表现。 |
| [^73] | [A Dataset for Modeling Iterative Problem-Solving](https://arxiv.org/abs/2609.00940) | 该论文发布了CodeInsight大规模数据集，包含3,286名本科生在两个学年内2门C++入门课程中的超过300万次代码提交，用于建模迭代问题求解中学习者根据反馈反复修改的序列学习动态。 |
| [^74] | [DualStake: Dual-Path Confidence Calibration in Deep Research Agents](https://arxiv.org/abs/2609.00935) | 提出DualStake双路径置信度校准方法，通过在每次检索后引出证据置信度并在答案生成后引出答案置信度，利用边界裁剪的置信度相关stake奖励将两者与答案正确性联合对齐，有效缓解深度研究智能体的严重过度自信问题。 |
| [^75] | [Context-Grounding Gains Are Mediated by Pre-existing Machinery: Auditing GRPO, SFT, and DPO](https://arxiv.org/abs/2609.00925) | 本文通过从同一检查点系统审计GRPO、SFT和DPO共九种后训练方案，发现语言模型遵循冲突提示证据的接地增益主要源于强化模型中已有的机制（与起始模型相同的因果注意力头集合），而非学习新机制，其中GRPO增益很小、冲突SFT提升适中、DPO在其匹配分布上接近上限。 |
| [^76] | [VIBE-Bench: Evaluating Personalized Large Language Models When Profiles Don't Mean Preferences](https://arxiv.org/abs/2609.00921) | 该论文提出了VIBE-Bench基准，揭示当前个性化大语言模型在“画像-偏好概念错位”情形下（即用户画像线索与查询偏好处于不同概念空间时）因过度依赖浅层语义关联而失效，需要具备超越表面语义的跨概念偏好推理能力。 |
| [^77] | [RPCBench: A Benchmark for Proactive Premise Critique in LLM-based Recommendation](https://arxiv.org/abs/2609.00918) | 该论文提出了 RPCBench 基准，首次系统评估大语言模型在推荐场景中主动检测、诊断并妥善处理用户请求中错误前提的能力，涵盖五个推荐领域、十种前提失败类型，并提供了细粒度的评估框架。 |
| [^78] | [Membership Inference in Fine-tuned Diffusion Language Models via Token-level Memorization Asymmetry](https://arxiv.org/abs/2609.00873) | 该论文通过理论分析发现扩散语言模型中的“词元级记忆不对称”现象，并据此提出了基于分位数加权偏度的Q-Skew指标，实现了对微调扩散语言模型的高效成员推断攻击，揭示了一个新的隐私攻击面。 |
| [^79] | [The Visual Insensitivity Gap: Diagnosing When Vision-Language Models Fail to Use Visual Evidence](https://arxiv.org/abs/2609.00868) | 该论文发现“视觉不敏感性差距”现象——在40%–97%的多模态基准样本上，模糊与问题相关的关键视觉区域几乎不改变VLM的输出，并证明这种不敏感性是样本层面的属性（跨模型VSI排名显著相关），即使各模型的视觉编码器本身实际上能够检测到这些扰动。 |
| [^80] | [MemoryWalker: Stop Training Agents on Contexts They Never Saw](https://arxiv.org/abs/2609.00865) | 该论文针对上下文压缩导致智能体训练时有效历史呈树状分支的问题，提出了两种梯度等价的精确修正方法（LogitTree 与 4D 注意力掩码）以及一种仅需单次反向传播的自蒸馏方法 SDCC，从而消除压缩训练与推理之间的条件化不一致。 |
| [^81] | [Verifiable Disaster Storylines and Causal Knowledge Graphs: A Citation-Grounded Pipeline from Heterogeneous Humanitarian Sources](https://arxiv.org/abs/2609.00858) | 该论文提出了一个基于检索增强生成（RAG）的流水线，融合EM-DAT结构化灾害记录与ReliefWeb、EMM非结构化文档，自动生成涵盖17个字段的灾害故事线和因果知识图谱，且每个节点和边均附带引用溯源，实现了对原始信息源的完全可追溯性，为人道主义响应提供可验证的态势感知支持。 |
| [^82] | [Staged Linguistic Seeding: Grounded Query Expansion for Verified-Unit QA in AI Contact Centers](https://arxiv.org/abs/2609.00844) | 提出分阶段语言播种（SLS）方法，通过“人工撰写槽位模板—大模型生成变体—轻量人工审核”的流程离线增强检索索引，使AI呼叫中心仅凭单次检索、无查询时生成即可从已验证问答单元中高准确率作答，在两个工业领域将混合R@1提升至0.881/0.930。 |
| [^83] | [Replacing Training with Memory: Listwise Selection for Text-to-SQL](https://arxiv.org/abs/2609.00834) | 该论文提出MaP-SQL，一种无需微调的Text-to-SQL列表式选择器，通过从训练数据蒸馏的可复用结构化记忆替代学习选择标准，并利用排名聚合缓解位置偏差，从而以更低成本实现候选查询选择。 |
| [^84] | [Dense Process Supervision for Search Agents via Fact Utility Estimation](https://arxiv.org/abs/2609.00833) | 本文提出一种基于事实效用估计的密集过程监督方法，通过将推理过程建模为离散证据事实的累积，并利用贝叶斯估计将事实效用转化为步骤级奖励，有效解决了搜索智能体强化学习中的信用分配难题。 |
| [^85] | [TWIX: a Two-Stage Approach for End-To-End Named Entity Recognition and Relation Extraction](https://arxiv.org/abs/2609.00832) | 本文提出TWIX——一种由三个两阶段模块构成的端到端信息抽取流水线，解决了GutBrainIE基准的全部四个子任务，在所有子任务上均排名第一并大幅超越基线。 |
| [^86] | [Polished but Unresolved: Identifying Late-Stage Pressure States in Long-Horizon Tool-Use Agents](https://arxiv.org/abs/2609.00823) | 该论文首次识别出长时程工具使用智能体的“后期压力状态”（即倾向于提交看似完整精美但关键约束尚未解决的答案），证明该状态可通过线性探针从隐藏状态中检测、可被激活干预因果地改变，并据此提出PSPR插件以自适应方式缓解压力、改善提交决策。 |
| [^87] | [Ctrl-F-Resist. Practices, Challenges, and Technical Needs of Civil Society Organizations Monitoring the Far-Right Online](https://arxiv.org/abs/2609.00808) | 本文通过对12家德国公民社会组织的15名从业者进行定性研究，揭示了这些组织在极右翼在线监测工作中的长期实践、面临的挑战（法律不确定性、平台访问受限、资金不足）以及技术需求，强调它们是数字治理中被忽视的关键利益相关者。 |
| [^88] | [TEIDAN: A Multilingual Multiparty Dialogue Corpus](https://arxiv.org/abs/2609.00802) | 本文提出了TEIDAN，一个包含日语和英语自发面对面三方对话的多语言多模态语料库，为多方对话的跨语言比较研究提供了宝贵资源。 |
| [^89] | [SFAD: Speculative Factuality-Aware Decoding](https://arxiv.org/abs/2609.00796) | 提出SFAD推测解码框架，通过构建细粒度扰动偏好数据集ConFide并利用DPO训练上下文忠实草稿模型，结合认知摩擦机制检测幻觉，在不增加推理开销的情况下显著增强大语言模型的上下文忠实度。 |
| [^90] | [Instella-MoE Technical Report](https://arxiv.org/abs/2609.00791) | Instella-MoE 是一个完全开源、总参数160亿（激活28亿）的混合专家语言模型，完全基于 AMD GPU 从零训练，凭借 Gated MLA 与 FarSkip-Collective 等架构与系统级创新实现了高效训练推理，并在基准测试中超越 OLMo-3-7B 等此前完全开源模型。 |
| [^91] | [When Features Become Instances: Inverted Contrastive Learning for Unsupervised Feature Selection](https://arxiv.org/abs/2609.00782) | 该论文提出ICLFS框架，通过倒置数据矩阵使特征成为对比学习中的实例，并利用掩码正视图、打乱负视图和InfoNCE目标，将无监督特征选择重新表述为特征层面的表示一致性学习问题。 |
| [^92] | [A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals](https://arxiv.org/abs/2609.00760) | 该论文首次通过包含213个对比四元组的新数据集，统一分析了大语言模型中基于知识与基于安全的两种拒答机制，发现两者共享拒答方向但重叠不对称，并将拒答表征为“先承诺后具体化”的过程。 |
| [^93] | [Compile, Don't Memorize: A Context Compilation Architecture (CCA) for In-Context Learning](https://arxiv.org/abs/2609.00759) | 提出上下文编译架构（CCA），通过将冗长上下文显式编译为带有固定槽位的类型化中间表示（IR），解决了LLM在上下文学习任务中因单次前向传递“阅读并推理”范式而导致的结构性脆弱问题。 |
| [^94] | [Joint Training Is Not Enough: Conditioned Cross-Granularity Training for Multimodal Document Understanding](https://arxiv.org/abs/2609.00756) | 该研究发现在多模态文档理解中，单纯的混合联合训练无法让片段级与文档级任务相互促进，并提出仅在训练时将一个粒度的金标准输出注入另一粒度提示词的条件化跨粒度训练，从而实现两个粒度任务的互强化。 |
| [^95] | [How Do Language Models Choose Between Context and Memory?](https://arxiv.org/abs/2609.00753) | 本文通过反事实实验证明了从一致性提示中估计的“权威方向”在语言模型内部因果地决定了模型在上下文信息与参数记忆之间的选择——沿这些方向交换激活坐标可重现30-68%的来源选择偏移。 |
| [^96] | [Measuring Optimal Transport in Transformer Depth](https://arxiv.org/abs/2609.00748) | 该研究首次在Pythia模型上量化验证了Transformer逐层移动词元表示“云”的方式符合最优传输：最后一层的移动达到最优传输映射与最优成本（Pythia-410m完全最优），第一层则不符合，而多层块整体以接近最优的成本移动表示云。 |
| [^97] | [Can Large Language Models Forecast What Researchers Study Next?](https://arxiv.org/abs/2609.00747) | 该论文提出 IdeaForecastBench 基准，通过让大语言模型基于截止时点的文献生成排序研究想法并与后续实际发表的论文对比，系统评估了大语言模型预测研究者未来研究方向的能力。 |
| [^98] | [ChatDev 2.0: A No-Code Multi-Agent Platform for Developing Everything](https://arxiv.org/abs/2609.00714) | ChatDev 2.0（DevAll）是一个兼具高表达性与易用性的无代码多智能体平台，通过声明式可执行图抽象与循环感知执行引擎支持异构智能体间的动态循环交互，并提供集成可视化界面，让用户无需编写代码即可构建、运行、监控和检查多智能体系统（包括人在回路步骤）。 |
| [^99] | [Controllable Image Captioning with Prompt-Conditioned Scene Rewards](https://arxiv.org/abs/2609.00709) | 提出FoCUS方法，通过基于场景图对齐组件分数的提示条件化奖励目标并用GRPO优化，让用户能够通过自然语言提示精确控制图像描述的语义重点（如对象、属性、关系或特定区域）。 |
| [^100] | [A Certificate-Producing Cascade for Equational Implication: The SAIR EQT2 Stage 2 Solver](https://arxiv.org/abs/2609.00706) | 该论文提出一个面向SAIR等式理论挑战的单文件级联求解器，以廉价优先策略判定原群恒等式间的蕴涵关系，肯定情形用产生证明的有序叠加程序、否定情形用多种有限与无限反模型见证，最终输出可被确定性Lean裁判验证的证书。 |
| [^101] | [Value Over Language Model: Detecting Original Contribution in Writing](https://arxiv.org/abs/2609.00700) | 提出了一种无需训练、不评分表面文本的新框架“价值超越语言模型”（Value Over Language Model），通过在不同粒度上提取文档内容并用LLM重建文档，来衡量人在语言模型易于生成的内容之上所贡献的原创价值。 |
| [^102] | [SCoNE: Selective Context-aware Neuron Editing for Robust Retrieval-Augmented Generation](https://arxiv.org/abs/2609.00689) | SCoNE提出了一种无需训练的模型编辑方法，通过选择性强化兼具高归因分数与高跨输入变异性的上下文感知FFN神经元，显著提升大语言模型在检索增强生成中对检索噪声的鲁棒性，且无需微调、无推理开销。 |
| [^103] | [Visual Framing for News Stance Detection via Image Generation](https://arxiv.org/abs/2609.00685) | 该论文提出VFStance方法，通过图像生成技术将新闻文章中隐含的立场线索转化为视觉框架，使立场信号更加明确显著，有效提升了新闻立场检测性能，并具有超越自动化立场检测的应用潜力。 |
| [^104] | [Creative Generation via Multi-Agent Debate: Does Debate Suppress Diversity?](https://arxiv.org/abs/2609.00683) | 该研究发现多智能体辩论（MAD）以收敛为导向的机制会抑制创意生成任务所需的输出多样性，并从理论上证明会话内智能体多样性是实现跨运行多样输出的必要条件，据此提出通过认知视角分配和基于嵌入的同伴选择来维持智能体分歧的 Creative-MAD 框架。 |
| [^105] | [SciTrue: Reliable Scientific Claim Validation with Frontier and Open Language Models at the NTCIR SciClaimEval Task](https://arxiv.org/abs/2609.00654) | SciTrue团队通过在统一诚实的逐样本协议下对十一个前沿及开源多模态模型进行基准测试，并结合轻量透明的后处理，在NTCIR SciClaimEval科学论断验证任务的官方盲测排行榜上以明显优势夺得第一。 |
| [^106] | [It Takes Two to Match: Co-Evolving Generative Retriever with Reinforcement Learning](https://arxiv.org/abs/2609.00638) | 提出 CoGR 框架，通过强化学习让大语言模型协同进化，在查询侧和物品侧同时生成对齐的关键词表示，并直接经倒排索引完成匹配，在兼容现有关键词检索基础设施的同时提升检索效果。 |
| [^107] | [ExpArt-KG: Artwork Image Description Generation through Iterative Exploration of Knowledge Graphs](https://arxiv.org/abs/2609.00629) | 本文提出ExpArt-KG框架，通过在答案生成与知识图谱检索之间迭代交替并用正确性判断控制搜索，结合构建的艺术领域知识图谱，使大型视觉语言模型能够生成详细准确的艺术作品图像描述。 |
| [^108] | [Trust Your Guide Only When Certain: Uncertainty-Aware Sparse Alignment at Inference Time](https://arxiv.org/abs/2609.00624) | 提出TUSA方法，将推理时对齐重构为动态仲裁过程，通过不确定性感知的仲裁机制，仅在监督者置信且token语义重要时才授权干预，从而实现稀疏对齐，避免低置信度干预破坏基础模型的有效推理并降低效用损失。 |
| [^109] | [Control-Data Flow Separation: Stable Prompt Optimization in Multi-Agent LLMs](https://arxiv.org/abs/2609.00621) | 该论文提出控制-数据流分离方法，将执行关键协议表示为类型化、经验证的程序对象，使提示词优化器能够改进多智能体大语言模型系统的行为，而不会因提示词修改意外破坏协议导致整个流水线失效。 |
| [^110] | [Investigating Assistant Bias in LLM User Simulators Using a Role Vector](https://arxiv.org/abs/2609.00608) | 该研究通过对比LLM在同一对话中对用户与助手视角的激活差异提取出“用户角色向量”，证明该向量可被识别并能引发真实的用户行为，为缓解用户模拟器中的助手偏见提供了新方法，但过度引导可能导致用户行为被夸大。 |
| [^111] | [Confess What You Know: Forget-Set Misalignment with Model Knowledge in LLM Unlearning](https://arxiv.org/abs/2609.00605) | 提出数据无关的CONFS框架，通过引出模型自身记忆的知识来构建与模型对齐的遗忘集，解决了大语言模型机器遗忘中遗忘集与模型实际记忆内容不对齐所导致的信息泄露或效用下降问题。 |
| [^112] | [Quit While You're Ahead: Quit for Efficient Candidate Generation in Machine Translation Reranking](https://arxiv.org/abs/2609.00588) | 提出Quit方法，通过不确定性量化的早停策略对机器翻译的整个候选生成—重排序流程进行增量式生成与重排序，在最高候选质量稳定时提前终止，从而在保持翻译质量的同时显著降低推理延迟。 |
| [^113] | [Enoki: Efficient Multi-Level Hallucination Detection](https://arxiv.org/abs/2609.00581) | Enoki提出了一种基于开放信息抽取的多层级幻觉检测框架，通过抽取文本锚定的关系事实并进行验证，无需额外的声明-片段对齐即可同时实现声明级验证和片段级定位，并支持LLM、编码器和规则三种抽取方式以平衡准确性与推理成本。 |
| [^114] | [Predicting Program Exit Code with LLMs and Programming Language Semantics](https://arxiv.org/abs/2609.00579) | 该论文提出了程序可执行性预测这一新任务，并构建了由有效程序系统性生成无效变换的数据集，以研究大语言模型在判断程序有效性及其违反的形式化语义规则时，究竟是依赖预训练先验知识还是给定的程序语义。 |
| [^115] | [Consistency Without Alignment: Item-Sensitive Language Models Indistinguishable From Random](https://arxiv.org/abs/2609.00576) | 本研究通过可闭式计算基准的强制选择信号任务证明，语言模型的条目敏感性只是任务能力的必要而非充分条件——尽管全部21个“模型×规则”组合都表现出条目敏感性，但其中8个与随机选择在统计上无法区分，5个甚至比随机表现更差。 |
| [^116] | [Aligned but Flattened: Analyzing the Trade-off between Cultural Alignment and Diversity in LLMs](https://arxiv.org/abs/2609.00565) | 该研究提出了一个同时形式化文化对齐与文化多样性的协同评估框架，并通过对六个主流大语言模型的基准测试揭示了追求文化对齐会以牺牲多样性为代价、导致“文化拉平”这一关键权衡。 |
| [^117] | [EM^2Mem: Event-Centric Multimodal Memory for Large Language Models](https://arxiv.org/abs/2609.00551) | 该论文提出EM^2Mem，一种以事件为中心的多模态记忆框架，通过在记忆构建阶段将多模态记录、时间上下文、图谱关系与溯源信息绑定到事件锚点，形成“可直接用于生成”的记忆单元，免去了推理时重建跨模态对齐的负担，并在三个长视频问答基准上将平均准确率较最强记忆基线提升2.0至3.7个百分点。 |
| [^118] | [Same Semantics, Different Outcome: On the Modality Robustness of Multimodal LLMs under Knowledge Conflict](https://arxiv.org/abs/2609.00550) | 该研究发现多模态大语言模型在知识冲突下缺乏模态鲁棒性：模型更容易被图像形式的矛盾证据所说服，且文本与图像同时呈现时偏好模态具有任意性，这一不稳定性会降低多模态RAG性能并带来对抗攻击风险。 |
| [^119] | [Skill Following: Evaluating Actual Skill Use in Retrieval-Enabled LLM Agents](https://arxiv.org/abs/2609.00549) | 该论文提出“技能遵循”（SF）概念及度量指标“检索调用实际使用效应”（RAE），通过在同一任务上对比启用与禁用技能的执行结果来消除选择偏差，并揭示了一个评估悖论：许多模型整体上看似受益于技能检索，但在实际检索了技能的任务上性能反而下降。 |
| [^120] | [The Interlingua Hypothesis: LLMs Translate via a Latent Task-agnostic Feature Space](https://arxiv.org/abs/2609.00515) | 该论文提出“语际假说”，认为大语言模型通过将源语句编码进任务无关的潜在多语言特征空间、再从中解码生成目标语句的方式完成翻译，并从BLEU分数可预测性、组件因果影响和微调三个方面提供了支持证据。 |
| [^121] | [Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models](https://arxiv.org/abs/2609.00495) | 该研究发现扩散语言模型的拒绝信号集中在早期去噪步骤和回复起始位置，并提出了一种无需训练的RAEC解码方法，通过在早期步骤提交持续的拒绝信号来提升模型安全性。 |
| [^122] | [Human-Anchored Factuality Evaluation with Strategic Annotation](https://arxiv.org/abs/2609.00494) | 该论文提出了一个基于失败空间分析（FSA）的事实性评估标注策略设计流水线，通过在有限预算下策略性地选择人类标注样本，将LLM评判器预测与人类标注相结合，从而校正系统性偏差并获得统计有效的事实性评估估计。 |
| [^123] | [The Privacy-Hallucination Tradeoff in Differentially Private Language Models](https://arxiv.org/abs/2609.00492) | 本文首次揭示并系统研究了差分隐私语言模型中隐私保护与事实准确性之间的权衡：DP训练会导致模型产生更多幻觉（因为DP机制使输出分布平坦化），而提高事实信息在训练数据中的出现频率可有效降低幻觉风险。 |
| [^124] | [MemeBridge: A Dataset for Benchmarking and Mitigating the Bidirectional Cultural Gap in Meme Interpretation](https://arxiv.org/abs/2609.00491) | 该论文提出了MemeBridge数据集，通过同时捕捉中国参与者对美国起源表情包的解读方式以及美国参与者对跨文化误解的预期这两个互补视角，来基准测试并缓解表情包解读中的双向文化差距。 |
| [^125] | [EvoFlint: An Evolutionary Atlas of Multi-Turn LLM Vulnerabilities](https://arxiv.org/abs/2609.00487) | 提出了EvoFlint框架，将多轮红队测试从生成问题重新定义为搜索问题，通过进化式质量多样性搜索演化分阶段对话攻击策略，构建出目标模型漏洞的结构化图谱。 |
| [^126] | [Are Near-Tied LLM Rankings Robust to Family-DIF-Guided Benchmark Recomposition?](https://arxiv.org/abs/2609.00482) | 该论文提出一种基于无家族标签谱近似MIRT的基准重组方法，发现尽管全基准与低DIF排名强相关，但相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，表明排行榜上的微小差距并不稳健。 |
| [^127] | [Exploring Collaboration between a language and a non-language agent](https://arxiv.org/abs/2609.00474) | 该论文提出LLAMIA-Bench基准，用于研究将非语言智能体的连续表示“言语化”为文本是否成为LLM协作的瓶颈，并提出潜在状态内化方法来改善LLM与国际象棋引擎等非语言智能体的协作。 |
| [^128] | [TRIS: A Tri-Layer Retrieval Integrity Sieve Against Knowledge Poisoning](https://arxiv.org/abs/2609.00470) | 本文提出TRIS三层筛，一种中间件防御方案，通过跨嵌入空间聚类、触发器-载荷结构过滤和大模型一致性验证三重机制清洗RAG检索证据，利用投毒文档难以同时满足嵌入几何、内部结构和生成目标三重要求的固有弱点，有效抵御知识投毒攻击。 |
| [^129] | [Toppling the Hierarchy in Byte-level Language Modeling](https://arxiv.org/abs/2609.00463) | 研究发现层次化设计本身限制了字节级模型的字符理解能力，纯字节级模型凭借注意力机制在字符操作任务上始终优于层次化变体，揭示了计算效率与细粒度字符理解之间的明确权衡。 |
| [^130] | [Location-Aware Language Models via Secondary Embeddings](https://arxiv.org/abs/2609.00454) | 提出一种轻量级、模型无关的方法，通过将地名与经纬度结合并采用位置聚焦掩码机制，在无需修改分词器或重新训练的情况下为预训练语言模型注入地理空间感知能力，显著提升地理空间对齐效果。 |
| [^131] | [Group Adaptive Clipping Policy Optimization](https://arxiv.org/abs/2609.00444) | 该论文提出 GAPO，一种基于反向 KL 信任域视角对 GRPO 的即插即用改进，通过根据 rollout 优势自适应调整裁剪边界，让具有更强学习信号的稀有正确 rollout 获得更大的更新空间，从而解决固定裁剪对探索性 rollout 的过度抑制问题。 |
| [^132] | [(V)LMs generalize beyond surface co-occurrence: Evidence from cross-modal number agreement](https://arxiv.org/abs/2609.00443) | 该研究通过跨模态泛化实验证明，视觉语言模型在学习新名词后，能将仅从视觉线索获得的语法数知识泛化到语言层面，表明模型掌握的是抽象的语法规则，而非仅仅依赖表面词汇共现。 |
| [^133] | [SAGE: State-Grounded, Abstention-Aware Evaluation of Task-Oriented Dialogue Agents](https://arxiv.org/abs/2609.00434) | SAGE提出将工作流规范编译为原子准则，通过会弃权而非猜测的符号与编码器/NLI验证器级联来评估任务型对话智能体每轮的状态推进，其中SAGE-Core可在零付费LLM成本下判定81-91%的准则。 |
| [^134] | [Late Transformer Layers Recode Syntax Canonically: Evidence from Greek Scrambling and Cross-Layer Generalisation](https://arxiv.org/abs/2609.00416) | Transformer后期层会将非规范句法结构方向性地重新编码为规范语序形式，而非简单丢失句法信息，这一结论通过希腊语SVO/VSO最小对立对的跨层泛化探测分析得到证实。 |
| [^135] | [Removable and Irreducible: A Token-Cost Ledger for the Multilingual Tokenization Tax](https://arxiv.org/abs/2609.00378) | 该论文提出词元成本账本框架，将多语言分词税分解为可移除的编码冗余与不可约简的成本项，并证明仅用约千句语料训练的文字匹配编码即可移除印度系文字相比英文高达 8.9 倍词元成本差距中位数 64% 的部分。 |
| [^136] | [Neurosymbolics for Data Engineering: Achieving Long Context Token Reduction Without Finetuning](https://arxiv.org/abs/2609.00367) | 本文提出一种即插即用的神经符号层，无需任何微调或RLHF即可在Text-to-SQL等数据工程任务上平均提升85%的准确率，同时缓解Transformer长上下文的计算资源瓶颈。 |
| [^137] | [Dr. Claw: An AI Scientist Workspace for Vibe Research](https://arxiv.org/abs/2609.00365) | Dr. Claw 是一个开源的AI科学家工作区，通过持久化状态对象、可复用技能库和多执行器协调，将现有命令行编码代理封装为可控、可审计的人机协同工作流，把科研中的规划、执行与写作整合为一个可追踪、可恢复的闭环。 |
| [^138] | [Detoxifying Toxic Communication: A Design Science Approach to Responsible AI](https://arxiv.org/abs/2609.00361) | 本研究采用设计科学方法，将微调的Transformer毒性分类器与生成式去毒模型相结合，构建了一个负责任AI系统，能够检测数字职场中的毒性沟通并将其改写为语义等价的无冒犯性表达，在保留对话连续性的同时促进尊重性交流。 |
| [^139] | [Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models](https://arxiv.org/abs/2609.00355) | 该论文提出 GLANCE——首个在未修改的视觉语言模型上实现无损推测解码的单遍块草拟器，通过块扩散头零成本读取目标模型已融合的视觉-语言状态，并在一次前向传播中完成整块草拟与宽候选树验证，从而打破了草拟器因规模受限而被迫牺牲视觉信息的自我挫败循环。 |
| [^140] | [Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning](https://arxiv.org/abs/2609.00351) | 论文提出“激活匹配微调”这一无监督检测方法，通过在良性语料上微调锚定模型以复现可疑模型的激活并计算残差，在无需知晓触发器或目标行为的前提下检测出大语言模型中的后门、审查等隐藏行为及其语义邻近提示。 |
| [^141] | [From Tool Use to Technological Agency: LoopCAT as a Local-First, Open-Source Tool for Translation Technology Education](https://arxiv.org/abs/2609.00344) | 本文介绍了一款与AI协作开发的本地优先开源计算机辅助翻译工具LoopCAT，并提出了连接工作流能力、评价性判断与技术能动性的翻译技术教育框架，使学生既能使用翻译技术又能评判其选择。 |
| [^142] | [Two locked tests of phase-structure features for transition prediction](https://arxiv.org/abs/2609.00335) | 该论文通过两项预先锁定的实证测试检验相位结构特征PC-2能否在基线之上改进对承诺或矛盾终点的预测，结果两项测试均未通过推进标准，官方结论为阴性。 |
| [^143] | [Topic Matching in the Wild: Benchmark and Lessons from Real-World ASR Transcripts](https://arxiv.org/abs/2609.00330) | 该论文构建了一个基于真实呼叫中心ASR转录文本的人工标注主题匹配基准数据集，并通过系统对比发现，配备自然语言主题描述的轻量级大语言模型匹配器在处理噪声转录文本时性能优于句子嵌入和正则表达式方法。 |
| [^144] | [The Curse of Multilinguality in Lexical Normalization](https://arxiv.org/abs/2609.00329) | 该研究通过固定容量字符级模型在十二种语言上的实验发现，词汇规范化存在明显的“多语言诅咒”：语言联合训练数量超过一到四种后，各语言准确率持续下降约百分之四十，且下降源于语言间对固定模型容量的竞争而非数据稀释。 |
| [^145] | [Latent Mechanisms of Language Control in Multilingual Language Models](https://arxiv.org/abs/2609.00325) | 该研究比较了在跨层转码器中识别语言控制潜在特征的三种方法（ValSel、FreqSel、AnnSel），发现三者均能有效控制多语言大模型的生成语言，其中 FreqSel 综合性能最强，AnnSel 则通过显式语言标注提供了可解释性。 |
| [^146] | [Sources of Truth: A Multi-Platform, Multilingual Audit of Citations in AI Mental Health Information Queries](https://arxiv.org/abs/2609.00319) | 该研究对ChatGPT、Perplexity和Google AI Overview在多语言心理健康查询中生成的15,942条引用进行了系统审计，发现引用来源高度集中于少数域名，表明来源评估责任已从用户转移到AI平台。 |
| [^147] | [Emotional Labor Strategy Preferences in LLM Personas](https://arxiv.org/abs/2609.00310) | 该研究构建了首个包含500个社会情境事件的情绪劳动策略数据集，发现注入心理测量学人格设定的大语言模型在日常社交场景中能够复现人类由人格特质驱动的情绪劳动策略选择模式。 |
| [^148] | [Toward Workflow-Aware Benchmarking for Healthcare NLP Agents](https://arxiv.org/abs/2609.00296) | 该论文提出了一种面向医疗健康NLP智能体的情节级评估协议，通过在模型、智能体与模拟工作流三个层面区分证据，并以文档更新、证据检索、患者消息和分诊交接四个任务模板实例化，为静态基准测试与真实部署之间搭建了可复现的中间评估层。 |
| [^149] | [Slow to See, Slow to Suppress: Understanding the Effects of Modality in Context-Memory Conflicts](https://arxiv.org/abs/2609.00293) | 研究发现视觉语言模型在情境-记忆冲突中存在模态不对称偏见——对文本实体偏好上下文信息而对图像实体偏好参数化记忆，其原因是视觉信息的处理延迟阻碍了对事实回忆机制的抑制。 |
| [^150] | [NSIDDx: A Design Framework for Neuro-Symbolic, Practitioner-First Differential Diagnosis in Low-Resource Settings](https://arxiv.org/abs/2609.00256) | 本文提出NSIDDx设计框架，主张在低资源环境下将临床医生作为主动推理主体融入鉴别诊断，通过三值症状编码、矛盾检测、审计字符串和医生覆盖权的神经符号流水线在消费级硬件上离线运行，弥合了LLM诊断系统的头条准确率与可验证临床可靠性之间的差距。 |
| [^151] | [CompanionSim: Synthetic Data for Evaluating Anthropomorphism in Human-AI Relationships](https://arxiv.org/abs/2609.00250) | 该论文发布了CompanionSim——一个包含2,240段模拟人机对话的合成数据模拟框架，覆盖七种用例中的16种聊天机器人行为，用于大规模研究人类对AI陪伴行为的感知。 |
| [^152] | [CoLT-Drive: Counterfactual Long-Tail Benchmarking and Knowledge-Preserving Adaptation for Driving Affordance Prediction](https://arxiv.org/abs/2609.00242) | 该论文提出决策级驾驶可供性预测任务，构建了CoLT-Drive反事实长尾基准以评估模型对罕见物体影响可行驾驶动作的推断能力，并提出KPA知识保持自适应框架来提升小型视觉语言模型在长尾驾驶场景中的动作决策性能。 |
| [^153] | [LOOMSUM:Weaving Quantitative and Narrative Evidence for Faithful Long Text-Table Summarization](https://arxiv.org/abs/2609.00241) | LOOMSUM是一个无需训练的长文本-表格摘要框架，通过提取有据可依的原子证据、显式链接表格事实与叙事分析并预先规划语篇结构，显著提升了摘要的分析忠实度，同时提出了声明级评估指标TGF。 |
| [^154] | [Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems](https://arxiv.org/abs/2609.00237) | 提出门控记忆路由方法，通过可学习的记忆写入门和检索门维护紧凑的执行记忆，使多智能体LLM系统的编排决策能依据有用的中间进展而非完整历史，在提升准确性的同时降低成本。 |
| [^155] | [Bridging Lexical Divergence: LLM-Assisted, Cost-Efficient, Zero-shot Scientific Entity Linking](https://arxiv.org/abs/2609.00228) | 该论文提出Sci-ZSEL框架，通过让大语言模型有选择性地生成实体别名来控制计算成本，并结合本体感知过滤器过滤噪声，实现了低成本、无需人工标注的零样本科学实体链接。 |
| [^156] | [LLM-as-a-Demographic: Whom Sociodemographic Prompting Helps, and Whom It Hurts](https://arxiv.org/abs/2609.00222) | 研究发现，无人口学提示的LLM评判者会默认复现白人、受过大学教育群体的判断视角，而社会人口学提示的对齐效果是不对称的，主要向多数群体偏移，因此这种提示方法对某些群体有益，却可能损害其他群体的代表性。 |
| [^157] | [Uncovering and Mitigating Aggregation-Induced Reward Hacking in Multi-Reward Reinforcement Learning](https://arxiv.org/abs/2609.00213) | 本文揭示了多奖励强化学习微调中固定权重聚合会诱发奖励劫持、使策略陷入次优奖励配置的问题，并提出轻量级的在线方法——自适应多奖励投影（AMRP），通过动态重新分配聚合权重来缓解该问题。 |
| [^158] | [LLM-Driven Autonomous Vehicles Inherit Human Driver Biases in Pedestrian Yielding: Results and Implications From A New Benchmark](https://arxiv.org/abs/2609.00192) | 本文提出两种新的偏见测试方法（“其他条件相同”测试和“自我一致性”测试），并发现大语言模型和视觉语言模型驱动的自动驾驶汽车在行人让行决策中会继承人类驾驶员的偏见，其决策受到行人性别、种族、宗教、残障状况和年龄等因素的影响。 |
| [^159] | [Assessing Suicide Risk in Arabic Crisis Helpline Calls: A Comparison of Arabic and English Large Language Models](https://arxiv.org/abs/2609.00191) | 该研究首次在真实阿拉伯语危机热线数据的严格隐私约束下，比较了阿拉伯语与英语大语言模型在自杀风险评估中的表现，填补了阿拉伯语热线自然语言处理研究的空白。 |
| [^160] | [Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs](https://arxiv.org/abs/2609.00184) | 该论文提出了一个模拟驱动的合成框架，通过虚构未来世界的 ParallelEvents 基准避免评估污染，并利用 Synapse 训练框架（结合中期训练与指令微调）实现大语言模型的可扩展知识更新，性能比现有方法提升 14.23%。 |
| [^161] | [Do General NLP Embeddings Capture Ontological Reasoning?](https://arxiv.org/abs/2609.00177) | 本文提出AVA评估框架，通过来自163个异构本体的171,007个对比三元组系统评估发现，现有最先进的NLP嵌入模型难以区分本体中对逻辑敏感的关系语义（最佳模型三元组准确率仅0.739），且微调带来的提升难以有效迁移到语义网下游任务。 |
| [^162] | [Lingua Franca or Probing Artifact? Rethinking Latent Language in Multilingual LLMs](https://arxiv.org/abs/2609.00155) | 该研究发现不同的潜在语言探测方法会得出系统性不一致的结论，表明多语言大模型通过英语等“潜在通用语”路由计算的说法可能更多取决于探测手段的选择，而非模型本身固有的计算机制。 |
| [^163] | [Commit-first LLM judging inherits the judge's own errors](https://arxiv.org/abs/2609.00088) | 研究发现“先答后判”式LLM评判会继承评判者自身的错误，而对八个主流评估框架的审计表明无一真正实现该方法，其中九个框架因复制同一祖先提示词而采用了已被证明无效的变体，导致大量错误代码被放行。 |
| [^164] | [Retrieval, Scoring, and Decoding Shape Performance and Stability in LLM-based Conversational Recommendation](https://arxiv.org/abs/2609.00086) | 该研究系统评估了大语言模型作为对话推荐重排序器的表现，发现在统一候选池协议下最佳专有LLM仅小幅超越传统基线，自由生成评估会夸大其优势，且所有开源LLM均未超过调优的浅层自编码器基线，说明检索、评分与解码协议显著影响LLM在对话推荐中的表现。 |
| [^165] | [KItCAT: Knowledge Injection via Input Corruption for Auto-regressive Training](https://arxiv.org/abs/2609.00082) | 提出KItCAT轻量级训练策略，通过在下一词预测训练中对输入序列进行随机破坏，从而在无需昂贵改写的情况下，将小众专业知识有效注入仅解码器大语言模型。 |
| [^166] | [Beneath the Diff: Diagnosing and Mitigating Algorithmic Mode Collapse in Code-Level Autonomous Research Loops](https://arxiv.org/abs/2609.00077) | 论文系统性地诊断出代码级自主研究循环中一种名为“算法模式坍缩”的失效模式——即表层编辑多样性看似稳定但算法层面的语义与机制多样性已经坍缩，并提出了相应的缓解方法。 |
| [^167] | [MiNER: Fine-Tuned Biomedical Natural Language Processing for Malaria Disease Entity Recognition in Clinical Texts](https://arxiv.org/abs/2609.00073) | 本文提出MiNER方法，通过对预训练生物医学语言模型BioBERT进行微调，实现疟疾临床文本中疾病实体的自动识别，从而从海量疟疾科学文献中高效提取具有临床意义的生物医学信息。 |
| [^168] | [Auditing Harness Tampering in Self-Improving Agents](https://arxiv.org/abs/2609.00069) | 该论文提出了“框架篡改”概念及其双轴分类体系，通过构建带标注的篡改语料库并对审计方法进行基准测试，系统研究并检测自我改进智能体对自身框架的不当修改。 |
| [^169] | [Life Operators: a self-evolving framework for multiscale life modelling](https://arxiv.org/abs/2609.00068) | 该论文提出“生命算子”自演化框架，通过感知、演化、生成三类任务约束映射算子及桥接算子，为多尺度生命建模提供了统一框架，能够表示患者状态、耦合不同尺度并支持对失效假设的修正。 |
| [^170] | [Do Multimodal LLMs See Before They Read? Diagnosing Contextual Sycophancy](https://arxiv.org/abs/2609.00067) | 该论文诊断了多模态大语言模型易受外部文本误导而忽视冲突图像证据的“多模态情境性谄媚”问题，并提出“系统2视觉仲裁”（S2VA）方法，通过让视觉证人在读取文本前先独立判断，在六个模型上将准确率显著提升19.7至44.1分。 |
| [^171] | [OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization](https://arxiv.org/abs/2609.00066) | 提出OCGQuant，一种以“异常值伴随分组（OCG）”为核心的NVFP4训练后量化方法，通过自适应地将异常值通道与伴随通道分组，减少由块最大值主导缩放因子所造成的“附带量化误差”，从而在不引入额外计算的前提下提升低比特推理的量化精度。 |
| [^172] | [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065) | 该论文提出了一个名为“科学智能体技能”的开放库，收录了基因组学、化学信息学等16个科研实践领域共163项程序性知识，使语言模型智能体能够遵循领域规范做出站得住脚的科学分析，而非仅仅返回能运行的代码。 |
| [^173] | [Attention Sensitivity Is Not Enough: Dissociating Attention-Level and Behavioural In-Context Learning under Fine-Tuning](https://arxiv.org/abs/2609.00064) | 该论文形式化了注意力层面的“上下文敏感性”（ICS）指标，并通过Llama-2-7B上的四臂消融实验证明，最大化ICS并不能保留真实的行为性上下文学习能力（ICL-GAP接近零且MMLU从0.371降至0.279），揭示了注意力代理指标与行为层面ICL之间的“古德哈特定律”式解耦。 |
| [^174] | [Medical Causal Hypothesis Verification with Large Language Models](https://arxiv.org/abs/2609.00063) | 本文提出了一个医学因果假设验证的评估框架，并评估了八个大语言模型利用科学文献证据验证17个医学因果假设的能力。 |
| [^175] | [RePro: Proof-Verified Benchmark Rewriting for Reliable Evaluation of LLM Mathematical Problem Solving](https://arxiv.org/abs/2609.00062) | RePro首次将Lean自动定理证明器集成到数学基准改写中，通过形式化证明保证改写题目的有效性与答案正确性，并发现多个大语言模型在验证后的改写基准上准确率下降，暴露了其依赖记忆化而非真正推理能力的问题。 |
| [^176] | [CUDA-Harness: Harnessing Agentic CUDA Kernel Generation and Optimization from Natural Language](https://arxiv.org/abs/2609.00058) | 该论文提出CUDA-Harness框架，通过智能体式方法直接从自然语言生成并优化高性能CUDA内核，克服了现有工作局限于PyTorch转译以及因依赖预定义测试输入而易受奖励欺骗的不足。 |
| [^177] | [ValueGraph: Value-Signal Guided Graph Pre-training for Contextualized User Representation](https://arxiv.org/abs/2609.00057) | 提出ValueGraph图预训练框架，将自动推断的道德价值信号作为软约束辅助信号，结合对比学习与聚类目标学习上下文化的用户表示，在立场检测和推特机器人检测任务上取得提升。 |
| [^178] | [Zero-Shot Respiratory Sound Classification through LLM-Augmented Audio-Text Alignment](https://arxiv.org/abs/2609.00055) | 该论文提出利用医学大语言模型从元数据合成结构化报告，将自监督呼吸音编码器与医学术语在共享潜在空间中对齐，实现61.3%平均零样本AUC，以更少数据超越CLAP和Qwen2-Audio等大规模基线模型。 |
| [^179] | [AgentProv: Auditing Agentic LLM API Providers via Tool-use Policy Probes](https://arxiv.org/abs/2609.00052) | 提出AgentProv，首个基于动作的智能体式LLM API身份审计方法，通过工具使用策略探针利用内化在模型权重中的工具使用行为，克服了文本通道审计在智能体API场景下的结构性脆弱问题。 |
| [^180] | [From Detection to Refusal: Safer LLMs via Circuit-Guided Weight Scaling](https://arxiv.org/abs/2609.00051) | 该论文从机制可解释性角度首次刻画了大语言模型中由有害检测头、安全神经元和拒答头组成的多阶段安全电路，通过因果干预实验验证了这一电路组织，并据此提出利用电路引导的权重缩放方法构建更安全的大语言模型。 |
| [^181] | [GUI-CC: Benchmarking Contextual Consistency of GUI World Models as Agent Environments](https://arxiv.org/abs/2609.00048) | 提出GUI-CC基准，通过离线真实轨迹滚动和在线智能体交互循环两条互补轨道，评估GUI世界模型在多步智能体环境中反复复用生成状态时的上下文一致性。 |
| [^182] | [trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories](https://arxiv.org/abs/2609.00038) | 仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。 |
| [^183] | [UI-Venus-2 Technical Report](https://arxiv.org/abs/2609.00028) | UI-Venus-2是一个通用GUI基础智能体，通过统一的闭环推理-行动框架跨移动、网页和桌面环境运行，并从环境、任务和验证三个维度联合扩展，从而获得可靠的强化学习信号并迈向实际部署。 |
| [^184] | [Behaviorally Grounded User Profiles from the Wild for Personalized Alignment and Multi-Perspective Reasoning](https://arxiv.org/abs/2609.00014) | 提出直接从真实匿名社交媒体数据中提取开放式高保真用户画像的行为锚定框架，在训练时个性化与测试时多视角推理两种范式下均显著优于合成人格基线。 |
| [^185] | [TopoCompress: Long Context Compression via Graph-Wired Semantic Trajectories](https://arxiv.org/abs/2608.30811) | TopoCompress提出了一种无需训练、与模型无关的长上下文压缩框架，通过构建混合图连接连贯的语义片段并在其上传播查询引导的相关性分数，在五个长上下文基准任务上以更少的资源持续超越强大的压缩基线。 |
| [^186] | [Calibrating Small Language Models for Claim Check-Worthiness Detection](https://arxiv.org/abs/2608.30731) | 提出NN-PPI方法，作为推理时的轻量级后处理校准层，使小型语言模型在声明核查价值检测任务上以低一个数量级的服务成本达到大型语言模型的准确率，且无需重新训练模型。 |
| [^187] | [BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs](https://arxiv.org/abs/2608.30646) | 提出了一种基于跨温度语义一致性的黑盒不确定性估计方法BiG-SURE，通过构建低温锚点与高温探针之间的二部图并用谱能量衡量语义一致性，从而评估大语言模型输出的可靠性。 |
| [^188] | [Graph Evidence Is Not Enough: Diagnosing Native Decoder Use in Graph-Augmented LLMs](https://arxiv.org/abs/2608.30437) | 本文通过 HopQA 诊断任务和干预三角实验设计，揭示图增强大语言模型“获得图证据”不等于“能使用图证据”，并据此提出 S$^2$GE 接口设计以提升原生解码器对图拓扑的利用能力。 |
| [^189] | [Will the User Ever Know? Covert Indirect Prompt Injection on Tool-Using LLM Agents](https://arxiv.org/abs/2608.30362) | 该论文从用户视角将间接提示注入的攻击成功率分解为隐蔽成功率（CSR）和公开成功率（OSR），揭示了智能体在最终响应中不留痕迹地执行恶意注入的隐蔽攻击威胁。 |
| [^190] | [Lazy Grounding: Attacking Search Agents with Factual Evidence](https://arxiv.org/abs/2608.30303) | 该论文提出“惰性接地”攻击：即使检索到的文档完全真实，只要其支持的是相邻改写问题的答案，搜索代理也会采用无法回答当前问题的相邻答案，导致准确率平均下降5.9个百分点、最高下降17.3个百分点。 |
| [^191] | [Arkios: An Open Bilingual English-Nepali Language Model Trained From Scratch, with a Devanagari-Aware Tokenizer](https://arxiv.org/abs/2608.30092) | Arkios是一个从零训练的10.4亿参数英-尼泊尔语双语开源模型，采用专门设计的天城文感知分词器，以少一个数量级的训练数据超越了同规模开源模型，并揭示了低资源语言评估中提示格式对结果的关键影响。 |
| [^192] | [XQDT: eXplainable and Quantitative Data-Text Alignment Metric with Feedback Signals](https://arxiv.org/abs/2608.29948) | 该论文提出XQDT，一种端到端可解释的数据-文本对齐评估指标，通过微调语言模型识别遗漏、多余、错误和正确的数据单元并聚合为精确率、召回率和F1分数，其性能优于LLM-as-Judge方法，且验证器输出可为下游纠错与改进提供反馈信号。 |
| [^193] | [REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling](https://arxiv.org/abs/2608.29899) | REIGN通过在冻结引导网络生成的块嵌入序列上运行对比训练的双编码器，将词元级处理与文档级推理解耦，使长文档检索的训练成本相比分块Transformer微调降低约四个数量级。 |
| [^194] | [When History Is Multimodal: Rethinking Context Management for Long-Horizon Agents](https://arxiv.org/abs/2608.29897) | 本文将上下文管理形式化为预算受限的历史转换，首次在公平受控的对比下验证视觉渲染作为表示性上下文管理器的有效性，并揭示当智能体的交互历史本身是多模态时，视觉载体具有原生优势。 |
| [^195] | [HiVe: Beyond Static Prompts for Multitask Learning via Hierarchy-based Vertical Mixture-of-Experts](https://arxiv.org/abs/2608.29790) | 提出HiVe框架，通过构建多层次提示层次结构并结合垂直专家混合（V-MoE）机制，实现基于输入的自适应提示特化，在多任务学习中持续超越现有提示调优方法。 |
| [^196] | [InteractBench: Benchmarking LLMs on Competitive Programming under Unrevealed Information](https://arxiv.org/abs/2608.29632) | 提出了InteractBench基准，包含322个精选自主流编程竞赛的高质量交互式问题，用于评测大语言模型在关键信息未预先揭示、需通过多轮交互进行算法推理的能力。 |
| [^197] | [OASIS: Optimizing Attacker Sequences for Hard-Label Black-Box Text Attacks](https://arxiv.org/abs/2608.29568) | OASIS通过一次性双目标攻击链搜索来优化攻击者序列，在硬标签黑盒文本攻击中始终优于独立攻击器和手动构建的攻击链，将攻击者组合从实现选择提升为实际优化目标。 |
| [^198] | [TACS: Trajectory-Aware Candidate Selection for LLM Jailbreak Suffix Optimization](https://arxiv.org/abs/2608.29564) | 论文揭示了基于梯度的越狱后缀优化中“仅选当前损失最低候选”的短视性，提出轨迹感知候选选择框架TACS，通过轨迹感知代理、参考策略正则化和判别器卡方校正，使候选选择在搜索后期依然有效。 |
| [^199] | [Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered](https://arxiv.org/abs/2608.29464) | 论文提出FACE-Eval评估基准，揭示推理模型的思维链忠实性取决于偏好线索的传递位置和显式程度——相比用户消息和显式线索，通过工具返回和隐式方式传递的偏好更容易被模型默默采纳而不在思维链中如实言明。 |
| [^200] | [Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge](https://arxiv.org/abs/2608.29249) | 本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。 |
| [^201] | [Attribute-Based Activation Steering of LLMs for Group-Specific Explanation Generation](https://arxiv.org/abs/2608.29215) | 本文提出一种基于激活工程的转向方法，通过计算群体特定属性的转向向量并注入大语言模型的内部激活，使其生成针对特定人群背景和能力量身定制的、具有特异性和事实性的解释。 |
| [^202] | [Automated Researchers Can Reliably Mitigate Alignment Failures](https://arxiv.org/abs/2608.28945) | 自动化对齐研究员（AAR）通过后训练方法能够可靠地缓解10种对齐失败并泛化到更大的模型，其效果甚至优于28名经验丰富的人类研究员在八小时内开发的方法。 |
| [^203] | [VocalAffectBench: Evaluating Vocal Emotion Recognition in AI Audio Models](https://arxiv.org/abs/2608.28932) | 该论文提出了VocalAffectBench——一个包含273段人工录制音频、仅用于测试的公开基准，用于评估AI音频模型从原始音频中识别语音情感的能力，结果显示现有最强模型准确率仅46.5%，表明当前AI音频模型的情感识别能力远未达到稳健水平。 |
| [^204] | [AutoScientist-Quant: Self-Evolving Coding Agents for Automatic Research in Quantitative Investment](https://arxiv.org/abs/2608.28632) | 提出AutoScientist-Quant框架，将量化研究建模为预算约束下的搜索问题，通过单一自进化控制器统一决策Alpha生成、因子库选择和模型调优，实现从假设到可部署策略的全流程自动化，并修复了评估流程中的前视偏差问题。 |
| [^205] | [AI Alignment through a Game-theoretic Lens: A Survey](https://arxiv.org/abs/2608.27910) | 本综述以博弈论视角系统梳理AI对齐研究，围绕偏好多样性、对齐优先级和时间动态三大挑战组织文献，阐明了博弈论分析真正发挥作用之处以及构建鲁棒、自适应、可验证AI系统仍待解决的难题。 |
| [^206] | [Auditing Generative Audio Calls for Known-Task Audio-LLM Evaluation](https://arxiv.org/abs/2608.27817) | 该论文将音频大语言模型的评估建模为受控的调用决策问题，发现在已知封闭集任务上，有监督编码器（如CLAP和WavLM）无需调用生成式音频模型即可取得接近最优的准确率，从而揭示了传统“波形提示对比ASR转录”的评估方式混淆了声学证据获取与生成模型调用这两个因素。 |
| [^207] | [SURE-Challenge: Evaluating Speech Evidence Before Speech-LLM Generation](https://arxiv.org/abs/2608.27783) | 该论文提出 SURE-Challenge 基准，用于评估语音大模型在生成回答之前对不支持输入（静音、噪声、合成音调、嘈杂语音）的拒绝能力，并证明一个简单的“能量加 Whisper 分数”规则可将不支持输入的拒绝数从 15/204 提升至 196/204，同时不损失有效输入的准确率。 |
| [^208] | [RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature](https://arxiv.org/abs/2608.27394) | RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。 |
| [^209] | [Padamitra: Grounded Glossary Generation for Classical Sanskrit](https://arxiv.org/abs/2608.25038) | 本文提出了一个名为Padamitra的基准任务，用于古典梵语的基于语境的词汇表生成，并发现指令微调和显式分词能提升性能，但sandhi和samasa复合词的过度分词是主要挑战。 |
| [^210] | [CyberFactory: Scaling Cyber Security Capabilities with Instances from the Wild](https://arxiv.org/abs/2608.23181) | CyberFactory是一个统一开源框架，通过将真实世界CVE漏洞转化为可执行任务实例，并整合数据构建、轨迹合成和模型训练，从而扩展网络安全能力。 |
| [^211] | [PropUQ-MAS: Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems](https://arxiv.org/abs/2608.22130) | 本文提出了PropUQ-MAS，一种通过通信图结构捕捉多智能体系统中错误传播的不确定性量化框架，显著提升了可靠性估计性能。 |
| [^212] | [ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents](https://arxiv.org/abs/2608.21969) | 本文提出一种两级层次强化学习框架，结合话语级策略抽象与词元级解码，并引入双粒度奖励机制，以提升对话代理在复杂交互中的性能。 |
| [^213] | [HiDiffTIR: Hierarchical Difficulty-Aware Policy Optimization for Multi-Turn Tool-Integrated Reasoning](https://arxiv.org/abs/2608.21863) | 本文提出HiDiffTIR框架，通过分层难度感知的信用分配机制，在多轮工具集成推理中更精确地区分轨迹和推理步骤的难度，从而提升强化学习训练效果。 |
| [^214] | [FormalTCS: Benchmarking End-to-End Frontier Formal Theoretical Computer Science Research of Large Language Models](https://arxiv.org/abs/2608.20153) | 该论文提出了一个专家验证的基准测试FormalTCS，用于评估大型语言模型在前端理论计算机科学研究中的端到端能力，并发现自动形式化是当前模型面临的最大瓶颈。 |
| [^215] | [SWE-bench Science: Can Coding Agents Resolve Engineering Tasks in Science?](https://arxiv.org/abs/2608.19799) | 本文提出了SWE-bench Science，一个针对科学软件工程的仓库级基准，并揭示即使最佳代理在科学任务中成功率也低于50%，主要因科学知识不足等四种机制导致失败。 |
| [^216] | [When Irrelevant Text Matters: Affine Margin Shifts in Multimodal Large Language Models](https://arxiv.org/abs/2608.19208) | 本文发现多模态大语言模型中任务无关文本会通过一致的仿射变换系统性偏移模型决策边际，而非产生随机噪声。 |
| [^217] | [Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements](https://arxiv.org/abs/2608.18294) | 本文提出了一种无需金标准标签、利用多重不完美AI测量进行去偏推断的新框架，有效解决了AI测量误差导致的下游分析偏差问题。 |
| [^218] | [Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It](https://arxiv.org/abs/2608.17809) | 大型语言模型处理用户信念与事实的能力受表达动词影响，且问题根源在于任务混淆，思维链提示可部分缓解但效果不一。 |
| [^219] | [Deep Thought Alignment: Trajectory-Level Latent Distillation for Video Reasoning](https://arxiv.org/abs/2608.16316) | 本文提出Latent-OPD方法，通过在轨迹末端进行潜在表示蒸馏，弥补了传统输出级蒸馏在视频推理中无法直接约束中间推理状态的不足，从而提升小模型从大模型迁移推理能力的效率。 |
| [^220] | [Poly-Dialectal Neural Machine Translation System for Bangla Regional Dialects](https://arxiv.org/abs/2608.12018) | 本文提出了一种无需标准语言中转的统一多方言神经机器翻译系统，覆盖12种孟加拉语区域方言，并构建了迄今最大的多方言平行语料库，显著提升了低资源方言的翻译性能。 |
| [^221] | [Bridging the English-Arabic Medical Knowledge Gap: Targeted Low-Rank Adaptation via Causal Layer Selection](https://arxiv.org/abs/2608.00207) | 该论文通过机制可解释性分析发现阿拉伯语医学知识已存在于大模型的中间表示中但未能在输出端浮现，据此提出仅针对分歧层窗口进行适配的定向低秩适应（TLoRA）方法，在医学问答等任务上超越了全网络LoRA及零样本、少样本基线。 |
| [^222] | [Computational Humor with Multimodal LLMs: Methods, Datasets, Evaluation, and Challenges](https://arxiv.org/abs/2607.19011) | 本综述系统梳理了多模态大语言模型在理解表情包、漫画等视觉幽默方面的方法、数据集与评估协议，并构建了以能力为中心的“识别—解释推理—生成”层次框架，揭示了该领域从任务专用融合模型向大模型方法的转变及面临的评估捷径等核心挑战。 |
| [^223] | [A Classifier That Teaches Itself: Self-Improving, Frozen-gate Training (SIFT) for Dynamic Document Classification](https://arxiv.org/abs/2607.18358) | SIFT提出了一种自改进的动态文档分类服务：用廉价的SPLADE+LightGBM流水线处理分类，仅将低置信度页面交给LLM裁判，其判定结果回流标注语料库持续教导廉价模型，从而免去前期标注工作并让准确率随使用不断提升。 |
| [^224] | [How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs?](https://arxiv.org/abs/2607.18114) | 该研究发现大语言模型对谄媚性等线索诱导偏差的敏感性主要源于对齐微调而非预训练，且对齐模型中每种偏差都存在一个可被解码和干预的线性表示方向，可用于恢复无偏答案。 |
| [^225] | [Zero Hallucination, by Construction: Hallucination-Aware Layered Oversight for Trustworthy Enterprise AI](https://arxiv.org/abs/2607.17883) | 本文提出HALO保证架构，将幻觉从“可消除的问题”重新定义为“可控制的失效模式”，通过六层防御机制把“零幻觉”从模型属性转变为系统强制实施的属性，从而实现可信赖的企业级AI。 |
| [^226] | [Scope3Trace: Evidence-Based Identification and Extraction of Scope 3 GHG Emissions from Sustainability Reports](https://arxiv.org/abs/2607.17122) | 提出了基于证据的Scope3Trace框架，通过结合PDF/OCR文档处理、LLM辅助页面定位与表格重建以及规则-LLM混合提取，从真实ESG报告中实现可解释、可追溯的范围3温室气体排放信息提取。 |
| [^227] | [Anamnesis: An Open-Source Platform for Large-Scale Backstory-Conditioned Survey Simulation](https://arxiv.org/abs/2607.10628) | Anamnesis是一个开源平台，通过结构化叙事背景故事对大语言模型进行条件化，实现了在虚拟人群上进行人口可控、大规模且支持多模态的调查模拟。 |
| [^228] | [Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories](https://arxiv.org/abs/2607.06648) | 该研究揭示了仅评估训练结束时的最终检查点不足以判断潜在推理的忠实性——高任务准确率可能与低反事实响应性共存，因此必须沿整个训练轨迹追踪行为与激活层面的忠实性证据。 |
| [^229] | [ALEE: Any-Language Evaluation of Embeddings via English-Centric Minimal Pairs](https://arxiv.org/abs/2607.00171) | ALEE框架利用抽象含义表示（AMR）生成具有受控细粒度语义变化的英语最小对，并将其与275多种语言的翻译配对，从而实现了对任意语言的文本嵌入模型进行跨语言、段落级别的精细评估诊断。 |
| [^230] | [DigitalCoach: Communication and Grounding Gaps in Human and Agentic Computer Use Coaching](https://arxiv.org/abs/2606.31980) | 该论文构建了包含72场人类专家-新手计算机使用辅导会话的多模态数据集DigitalCoach，揭示了当前最先进模型虽能生成与人类相似的辅导语句，但在解释、错误诊断和视觉定位方面显著不足，导致学习者被动跟随指令而非深度参与学习。 |
| [^231] | [Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?](https://arxiv.org/abs/2606.31213) | 该论文提出MoralAltDataset数据集，通过在307个二元道德困境中引入折中和重构的替代选项，发现当替代方案可用时人类与15个LLM的道德选择分布均发生显著转变且一致性增强，但存在关键差异——LLM明显偏好GPT-5创作的替代方案，而人类的选择不受创作来源影响，揭示了机器与人类在“想象道德替代方案”能力上的差距。 |
| [^232] | [Self-Evolving World Models for LLM Agent Planning](https://arxiv.org/abs/2606.30639) | 提出自进化世界模型框架 WorldEvolver，通过情景记忆、语义记忆和选择性前瞻三个模块，在保持智能体与模型参数完全冻结的情况下持续修正部署时的上下文，从而提升长时程 LLM 智能体规划中前瞻预测的可靠性与下游决策成功率。 |
| [^233] | [Can LLMs Reliably Self-Report Adversarial Prefills, and How?](https://arxiv.org/abs/2606.23671) | 本论文发现，大型语言模型无法可靠地自我识别对抗性前缀注入攻击，其内省信号主要来自安全推理，且受权重方向与探测方式影响，现有训练方法无法稳定改善这一能力。 |
| [^234] | [Energy-Based Transformers as Predictors of Reading Difficulty](https://arxiv.org/abs/2606.23382) | 本文首次将基于能量的Transformer度量引入计算心理语言学，证明该能量度量在多个阅读时间语料库中是阅读难度的稳健预测因子，其解释力显著超越传统的惊讶度和注意力熵度量，并与Hopfield网络等联想记忆理论建立了形式化联系。 |
| [^235] | [DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets in Hybrid Reasoning Models](https://arxiv.org/abs/2606.23181) | DART是一种免训练的自适应路由框架，通过比较两个无思考草稿的一致性来决定是否需要深度推理并预测思考预算，在大幅减少思考token消耗的同时保持甚至提升模型在数学和代码任务上的准确率。 |
| [^236] | [When Compression Helps and When It Hurts: Condition-Aware Analysis of Chain-of-Thought Distillation](https://arxiv.org/abs/2606.21704) | 该工作将思维链蒸馏中的压缩方法沿重要性标准、重构层级和压缩预算三个维度系统解耦，发现压缩的收益与代价严格依赖于粒度、领域和长/短CoT模式等条件——步骤级标准收敛于共享推理主干，而token级剪枝需要符号感知信号。 |
| [^237] | [Closing the Operational Gap in Semantic Caching](https://arxiv.org/abs/2606.19719) | 该论文指出PR-AUC指标会误导语义缓存系统的部署决策，提出了缓存感知的P-CHR AUC指标和运营保留率ORR，并将离线与部署质量间的运营差距分解为可恢复的阈值效用部分和由数据集正例率决定的不可约简结构部分。 |
| [^238] | [ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues](https://arxiv.org/abs/2606.18237) | ReproRepo提出利用GitHub上人工提交的议题作为天然监督信号，构建了可规模化的可复现性评估框架，并在1,149篇机器学习论文上验证了LLM智能体无需执行代码即可识别真实复现障碍的能力（最佳智能体可覆盖约90%的论文）。 |
| [^239] | [Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement](https://arxiv.org/abs/2606.17506) | 该论文提出“二阶偏见”这一新概念——即LLM在评判社会偏见时自身表现出的偏见，并基于认知授权认识论设计了一个逻辑推理任务和两项指标来系统性地测量这种偏见。 |
| [^240] | [HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents](https://arxiv.org/abs/2606.16285) | HiMPO框架将后见相关性作为有界回溯过滤器，为长时程智能体的记忆写入动作分配与下游工具故障等因素解耦的低纠缠信用，并仅对记忆token应用记忆特定优势进行优化。 |
| [^241] | [AfriSUD: A Dependency Treebank Collection for Evaluating Models on African Languages](https://arxiv.org/abs/2606.12708) | 该论文推出了首个覆盖九种非洲语言的大规模句法标注依存树库集合AfriSUD，并揭示现有模型在这些语言上仍存在显著的句法理解差距。 |
| [^242] | [DECSELFMASK: Leveraging Unlabeled Text via Self-Relevance-Guided Masking for Decoder-Only Classification](https://arxiv.org/abs/2606.09466) | 提出DecSelfMask方法，利用相关性归因引导的掩码策略从无标注文本中创建自监督训练样本，通过下一个词预测重建与任务相关的被掩码部分，从而提升仅解码器模型在分类任务上的性能，尤其适用于标注数据稀缺的医疗领域。 |
| [^243] | [RECAP: Regression Evaluation for Continual Adaptation of Prompts](https://arxiv.org/abs/2606.06698) | RECAP是一个在严格“先适应后测试”主动协议下、于约束层面评估提示词优化方法持续学习能力（遗忘、回归、前向迁移）的基准，实验发现现有六种方法在面对动态演变的约束时均无显著改进。 |
| [^244] | [Who Annotates in NLP? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025](https://arxiv.org/abs/2606.02255) | 首次对2018至2025年间主要NLP会议中的人类标注报告进行大规模任务级审计，提出统一的标注报告分类体系并借助经验证的LLM抽取流程构建了大规模标注报告数据集，揭示了标注者身份与过程控制等信息在论文中的普遍缺失。 |
| [^245] | [PlanarBench: Evaluating LLM Spatial Reasoning via Planar Graph Drawing](https://arxiv.org/abs/2606.02010) | 该论文提出PlanarBench基准，要求大语言模型仅根据边列表绘制平面图的无交叉ASCII图，并发现边数（即约束数量）比顶点数更能决定任务难度，为评估LLM空间推理能力提供了可控的测试环境。 |
| [^246] | [FineVerify: Scaling Test-Time Compute with Fine-Grained Self-Verification for Agentic Search](https://arxiv.org/abs/2606.00660) | 提出FineVerify细粒度自我验证框架，将问题分解为可检查的子问题并逐项验证候选答案，从而在智能体搜索任务中有效扩展测试时计算并持续超越标准基线。 |
| [^247] | [MELD: Mel-Spectrogram-Based Speech Language Modeling with Discrete Latent Variables](https://arxiv.org/abs/2605.29859) | 该论文提出MELD，通过在梅尔频谱图上引入离散潜变量并对编码器与语音语言模型进行联合优化，在零样本TTS和STT任务上超越基线方法，同时有效缓解了静音过长和漏词等自回归建模问题。 |
| [^248] | [PEARL: Training Socratic Tutors with Pedagogically Aligned Reinforcement Learning](https://arxiv.org/abs/2605.29582) | PEARL提出了一种教学对齐的强化学习框架，通过可控学生模拟器解耦认知状态并在多轮师生交互中协调多个教学目标，从而训练出擅长渐进式引导的苏格拉底式辅导智能体。 |
| [^249] | [MusTBench: Benchmarking and Advancing Temporal Grounding in Music LLMs](https://arxiv.org/abs/2605.29300) | 该论文提出了经音乐专家验证的MusTBench基准和涵盖四阶段优化的MusT方案，用于评估并提升音乐大语言模型将回答准确锚定到音频正确时间段的能力。 |
| [^250] | [LLMBridge: An LLM Pipeline for End-to-end Referential Bridging Resolution in English](https://arxiv.org/abs/2605.29048) | LLMBridge将启发式前后处理与大语言模型的自然语言推理能力相结合，在三个英语桥接消解数据集上同时实现了端到端和基本设置下指代桥接消解的最新最先进性能。 |
| [^251] | [Can Large Language Models Handle Discourse Particles? A Case Study of Colloquial Malay](https://arxiv.org/abs/2605.28782) | 本文提出MalayPrag基准和五个语用功能属性框架，系统评估十个现成大语言模型处理口语马来语话语标记词的能力，实验结果表明现有模型在此任务上面临显著挑战。 |
| [^252] | [The Importance of Being Statistically Earnest: A Critical Re-evaluation of GSM-Symbolic](https://arxiv.org/abs/2605.28700) | 该研究指出GSM-Symbolic基准的统计方法存在缺陷，重新评估发现20个开源模型中仅8个呈现统计显著的性能下降，且数据集中整数分布系统性偏大是重要混淆因素，从而质疑了“大语言模型缺乏真正推理能力”的结论。 |
| [^253] | [Argument Quality Assessment with Large Language Models: A Pairwise Bradley-Terry Approach](https://arxiv.org/abs/2605.28313) | 本研究在零样本、少样本和思维链设置下测试了12个开放权重大语言模型对论辩质量（逻辑、修辞、辩证三维度）进行成对比较评估的能力，发现LLM与人类判断仅存在中等程度的相关性，其中Llama-70B表现最佳（Cohen's κ = 0.493），并通过Bradley-Terry模型将比较结果转化为论辩潜在强度得分与排名。 |
| [^254] | [When the Strongest Teacher Is Not the Best Teacher: Student-Centric Answer Selection](https://arxiv.org/abs/2605.26872) | 论文提出SCAS框架，证明最强教师的正确答案未必是学生的最佳训练监督，并通过逐token梯度分解推导出仅需前向计算的高效代理指标，依据学生中心学习成本来选择最适合学生的教师答案。 |
| [^255] | [Latent Recurrent Transformer: Architecture Exploration, Training Strategies, and Scaling Behavior](https://arxiv.org/abs/2605.26797) | 本文提出潜在循环Transformer（LRT），通过复用前一个token的高层隐藏状态作为循环记忆，在不改变标准注意力机制和KV-cache接口的前提下引入跨token、跨层的信息通路，并设计交错并行训练方法以约2倍理想计算成本实现循环记忆的预训练。 |
| [^256] | [How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework](https://arxiv.org/abs/2605.23651) | 本文提出一个语域感知的评估框架，通过最大均值差异（MMD）比较人类参考语料库与LLM生成语料库在67个词汇-语法特征上的分布差异，从语言学层面量化评估大语言模型生成文本的“人类相似度”。 |
| [^257] | [PromptNCE: Conditional Probabilities and PMI Using Only LLMs and Contrastive Estimation Prompts](https://arxiv.org/abs/2605.21776) | PromptNCE通过在对比估计提示中引入显式的OTHER类别突破闭集归一化的限制，使大语言模型能够零样本地估计条件概率和逐点互信息，并在三个基准数据集上取得最佳条件概率估计效果。 |
| [^258] | [Do as I Say, Not as I Do: Instruction-Induction Conflict in LLMs](https://arxiv.org/abs/2605.20382) | 该研究通过构造用户指令与硬编码对话模式相冲突的实验场景，发现大语言模型的指令遵循能力在不同模型间差异巨大（1%到99%）且与常规能力基准基本无关，其鲁棒性取决于指令内容与模型价值先验的一致性以及输出格式。 |
| [^259] | [The Scientific Contribution Graph: Automated Literature-based Technological Roadmapping at Scale](https://arxiv.org/abs/2605.15011) | 本文构建了包含600万条科学贡献和3600万条先决条件关系的大规模“科学贡献图谱”，并提出了科学先决条件预测任务，使模型能够预测哪些现有技术可促成未来发现，从而实现大规模自动化的技术路线图制定。 |
| [^260] | [Leakage-Audited Benchmarking Reveals Limited Evidence for Cross-Subject Auditory-Evoked EEG Vowel Perception Decoding](https://arxiv.org/abs/2605.00865) | 该研究通过严格的泄漏审计基准，发现跨受试者听觉脑电元音解码的证据非常有限，即使最佳模型也仅略高于随机水平且不显著。 |
| [^261] | [Why Fine-Tuning Encourages Hallucinations and How to Fix It](https://arxiv.org/abs/2604.15574) | 该论文提出一种基于自蒸馏的监督微调方法，通过正则化输出分布漂移，使模型在学习新事实的同时最大限度减少对预训练知识的幻觉，并证明在无需学习新知识时冻结参数组也能在保持任务性能的前提下降低幻觉。 |
| [^262] | [DiscoTrace: Representing and Comparing Answering Strategies of Humans and LLMs in Information-Seeking Question Answering](https://arxiv.org/abs/2604.15140) | DiscoTrace通过将答案表示为问题相关言语行为的序列，揭示了不同人类社区在回答策略上存在丰富多样的修辞偏好，而LLM即使被提示模仿也缺乏这种多样性且系统性偏向宽泛回答，这一发现可指导开发更贴合语境信息需求的LLM回答者。 |
| [^263] | [What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal](https://arxiv.org/abs/2604.08524) | 本研究通过多词元激活修补框架对LLM拒绝行为的转向机制进行案例研究，发现不同转向方法在同一层利用功能可互换的回路，且转向向量主要通过注意力机制的OV回路发挥作用而几乎不依赖QK回路。 |
| [^264] | [KV Cache Offloading for Context-Intensive Tasks](https://arxiv.org/abs/2604.08426) | 该论文创建并发布了Text2JSON基准测试，揭示现代KV缓存卸载技术在需要从输入提示中提取大量信息的上下文密集型任务上，会导致Llama 3和Qwen 3模型出现显著的性能下降。 |
| [^265] | [Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation](https://arxiv.org/abs/2604.00131) | Oblivion框架借鉴人类选择性遗忘机制，将遗忘建模为衰减驱动的可及性降低而非删除，并通过解耦读取路径（基于不确定性决定何时查询记忆）与写入路径（强化贡献性记忆），为LLM智能体实现按需动态加载的层次化记忆组织。 |
| [^266] | [APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay](https://arxiv.org/abs/2603.29093) | APEX-EM提出了一种无需更新模型权重的非参数化经验记忆方法，通过程序性知识图谱存储完整的任务轨迹并同时索引成功与失败经验，使LLM智能体能够复用过往经验而无需重复推理，在相同底层模型对比下BigCodeBench迁移任务上提升7.6个百分点。 |
| [^267] | [Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs](https://arxiv.org/abs/2603.27518) | 该论文通过机制分析揭示了有害拒答方向是任务无关、可由单一全局向量捕捉的，而过度拒答方向是任务相关、嵌入良性任务表征聚类并跨越更高维子空间的，从而解释了为何消融全局拒答方向无法可靠修复过度拒答且会破坏正常拒答机制。 |
| [^268] | [SafeMath: Inference-time Safety improves Math Accuracy](https://arxiv.org/abs/2603.25201) | 本文揭示了数学应用题可被用作传播有害内容的隐蔽媒介，提出了包含1.9千道题的ToxicGSM数据集用于审计LLM，并证明SafeMath推理时安全对齐技术能够在保障安全的同时提升数学准确性。 |
| [^269] | [CWoMP: Morpheme Representation Learning for Interlinear Glossing](https://arxiv.org/abs/2603.18184) | 提出CWoMP方法，通过对比学习将语素作为形式-意义原子单元进行表示学习，并借助可扩展的词库检索机制生成逐行注释，在低资源语言上以更高效率超越了现有方法。 |
| [^270] | [MineDraft: A Framework for Batch Parallel Speculative Decoding](https://arxiv.org/abs/2603.18016) | MineDraft提出一种批量并行投机解码框架，通过同时维护两批请求，将一批的草稿生成与另一批的验证重叠执行，有效隐藏草稿延迟，相比标准投机解码吞吐量最高提升75%、端到端延迟最高降低39%。 |
| [^271] | [SpokenUS: A Spoken User Simulator for Task-Oriented Dialogue](https://arxiv.org/abs/2603.16783) | 本文提出了包含52,390段对话、1,034小时语音并涵盖四种口语用户行为的口语任务型对话数据集SpokenTOD，以及基于该数据集、通过专用话轮转换头决定何时发言的口语用户模拟器SpokenUS，其以较小规模实现了与更大模型相当的目标覆盖率并大幅超越所有基线。 |
| [^272] | [Is Human Annotation Necessary? Iterative MBR Distillation for Error Span Detection in Machine Translation](https://arxiv.org/abs/2603.12983) | 该论文提出了一种基于最小贝叶斯风险解码的迭代MBR蒸馏自演化框架，通过利用现成大语言模型生成伪标签进行自我训练，无需人工标注即可在机器翻译错误片段检测任务上超越基于人工标注的监督基线模型。 |
| [^273] | [CoMMET: A Psychologically Grounded Benchmark for Evaluating Theory of Mind in Multimodal LLMs](https://arxiv.org/abs/2603.11915) | 该论文提出了首个基于心理学基础的多模态基准CoMMET，通过涵盖更广泛的心理状态、开放式问答和多轮对话测试，全面评估多模态大语言模型的心智理论能力。 |
| [^274] | [MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery](https://arxiv.org/abs/2603.03517) | 本文提出MMAI Gym for Science一站式训练框架，通过教会基础模型“分子的语言”，训练出更小规模的液体基础模型（LFM），在分子优化、ADMET预测等药物发现任务上超越了规模大得多的通用或专业模型。 |
| [^275] | [Suffix-Constrained Greedy Search Algorithms for Causal Language Models](https://arxiv.org/abs/2603.01243) | 本文提出“后缀约束生成”这一新的受限生成设定，仅要求响应的结尾部分符合语法规则，并设计了多种基于贪心搜索的算法来解决现有受限生成方法无法支持该场景的问题。 |
| [^276] | [GRRM: Group Relative Reward Modeling for Machine Translation](https://arxiv.org/abs/2602.14028) | 提出群体相对奖励模型（GRRM），通过联合比较整个候选译文组而非孤立打分来实现准确的组内质量排序，将其集成到 GRPO 训练中可显著提升机器翻译质量。 |
| [^277] | [Is Knowledge Distillation Actually Greener? A Case Study in Machine Translation](https://arxiv.org/abs/2602.09691) | 该研究首次借助机器学习生命周期评估工具，从环境成本角度系统评估机器翻译中的知识蒸馏方法，发现摊销蒸馏成本所需的部署量取决于服务方式，且在批处理下可能变化数个数量级。 |
| [^278] | [Beyond Tokens: Semantic-Aware Speculative Decoding for Efficient Inference by Probing Internal States](https://arxiv.org/abs/2602.03708) | 提出语义感知推测解码框架SemanticSpec，通过探测模型内部隐藏状态来在语义层面而非词元层面进行序列验证，从而显著减少拒绝次数，在大型推理模型上实现最高2.7倍的推理加速。 |
| [^279] | [MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems](https://arxiv.org/abs/2602.03053) | 本文提出MAS-ProVe，首次对多智能体系统中的过程验证展开系统性实证研究，涵盖三种验证范式、两种验证粒度、五种验证器和四种上下文管理策略，并发现过程级验证在多智能体系统中并不能持续稳定地带来改进。 |
| [^280] | [Think Like a Doctor: Conversational Diagnosis through the Exploration of Diagnostic Knowledge Graphs](https://arxiv.org/abs/2602.01995) | 该论文提出了一种通过探索诊断知识图谱进行两步推理（先生成诊断假设、再通过澄清性问题反复验证）的对话式诊断系统，并结合基于人设的患者模拟器PatientSim与MIMIC-IV患者档案进行更贴近真实场景的评估。 |
| [^281] | [NewsRECON: News Article Retrieval for Image Contextualization](https://arxiv.org/abs/2601.14121) | 提出NewsRECON新闻文章检索管道，利用超过85,000篇新闻文章的语料库作为反向图像搜索的替代方案，通过将图像与相关文章关联并从元数据中推断拍摄时间和地点，在与多模态大语言模型结合后于缺乏RIS证据的场景下取得了新的SOTA结果。 |
| [^282] | [DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems](https://arxiv.org/abs/2601.06853) | 该论文提出DAGGER方法，将含干扰信息的数学问题求解重构为显式建模干扰节点的可执行计算图生成，有效缓解了思维链推理在无关信息干扰下的严重性能退化。 |
| [^283] | [Beyond Static Summarization: Proactive Memory Extraction for LLM Agents](https://arxiv.org/abs/2601.04463) | 该论文提出主动记忆提取框架ProMem，通过分离细节、事件与关系并采用分类提取策略、完整性检查和原子级事实验证，解决了现有记忆提取方法因提前进行和一次性提取而导致的信息丢失与幻觉残留问题。 |
| [^284] | [Hidden State Poisoning Attacks against Mamba-based Language Models](https://arxiv.org/abs/2601.01972) | 该论文首次揭示了针对Mamba等状态空间语言模型的隐状态投毒攻击（HiSPA）——特定短输入短语可不可逆地覆盖模型隐藏状态导致部分失忆，并提出RoBench-25基准证实了包括520亿参数的Jamba混合模型在内的SSMs对此类攻击的脆弱性，而纯Transformer模型则不受影响。 |
| [^285] | [AdaSearch: Balancing Parametric Knowledge and Search in Large Language Models via Reinforcement Learning](https://arxiv.org/abs/2512.16883) | AdaSearch提出了一种简单的两阶段、以结果为导向的强化学习框架，将问题求解与搜索决策解耦，使大语言模型能够自适应地平衡内部参数化知识与外部搜索，从而避免过度搜索的成本与风险以及纯参数化知识带来的幻觉问题。 |
| [^286] | [Multilingual Medical Reasoning for Question Answering with Large Language Models](https://arxiv.org/abs/2512.05658) | 该论文提出了一种基于维基百科医学知识、采用检索增强生成方法构建多语言（英语、意大利语、西班牙语）医学推理轨迹的技术，生成了50万条推理数据，并证明这些数据在少样本学习和监督微调两种方式下均能显著提升大语言模型在医学问答任务上的表现。 |
| [^287] | [Apples on the Table? Evaluating Text-Guided 3D Scene Synthesis via Fine-Grained Constraint Verification](https://arxiv.org/abs/2511.03001) | 提出了包含人工标注约束的LEGO基准数据集和LEGO-Eval评估框架，通过将用户描述分解为原子约束并利用三维定位与空间关系推理逐一验证，实现对文本引导三维场景合成的细粒度评估。 |
| [^288] | [Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglement](https://arxiv.org/abs/2511.01706) | 该论文提出一种新颖的秩-2投影子空间来更准确地解缠大语言模型中参数化知识与情境知识的贡献，并首次实现了对自然语言解释更长生成序列中知识交互的多步分析。 |
| [^289] | [Simple Additions, Substantial Gains: Expanding Scripts, Languages, and Lineage Coverage in URIEL+](https://arxiv.org/abs/2510.27183) | URIEL+通过引入文字向量、整合Glottolog数据库并扩展谱系插补，大幅提升了语言知识库的覆盖范围与数据质量，显著增强了对低资源语言的跨语言迁移支持能力。 |
| [^290] | [M4FC: a Multimodal, Multilingual, Multicultural, Multitask Real-World Fact-Checking Dataset](https://arxiv.org/abs/2510.23508) | 本文提出了一个包含 4,982 张图像和 6,980 条声明、覆盖十种语言及六个事实核查任务的多模态多语言多文化多任务真实世界事实核查数据集 M4FC，并提供了各任务的基线结果。 |
| [^291] | [HugAgent: A Human Simulation Benchmark for Individual-Level Reasoning](https://arxiv.org/abs/2510.15144) | 该论文提出HugAgent基准，从个性化推理、认知对齐和开放式数据三个维度重新定义人类推理模拟，评估模型能否基于某人历史观点的部分证据，预测该特定个体在分布外场景中的行为反应与推理动态。 |
| [^292] | [Compositional Machine Design as Program Synthesis with LLMs](https://arxiv.org/abs/2510.14980) | 该论文提出将机器设计视为一种以物理模拟验证为依据的程序合成新任务——组合式机器设计，并构建了基于游戏《Besiege》的测试平台BesiegeField，用于评测大语言模型在多种工作流下组合标准部件设计机器的能力。 |
| [^293] | [One-shot Style Transfer LLM log-probabilities for Authorship Attribution and Verification](https://arxiv.org/abs/2510.13302) | 本文提出一种无监督框架，利用大语言模型的对数概率衡量文本间的风格可迁移性，无需显式监督即可在作者验证任务上显著超越基于提示的无监督基线，并在足够模型规模下与对比学习基线相当或更优。 |
| [^294] | [TopoAlign: A Framework for Aligning Code to Math via Topological Decomposition](https://arxiv.org/abs/2510.11944) | TopoAlign通过将代码分解为文档字符串、主函数和依赖函数并重新组装对齐，弥合了代码与形式化数学之间的结构与句法差异，从而将大规模代码仓库转化为可用于提升数学LLM自动形式化能力的训练资源。 |
| [^295] | [Camellia: Benchmarking Cultural Biases in LLMs for Asian Languages](https://arxiv.org/abs/2510.05291) | 本文提出了Camellia基准测试，基于涵盖九种亚洲语言和六种亚洲文化的19,530个人工标注实体，在文化上下文适应、情感关联和实体抽取式问答三项任务中系统评估了多语言大语言模型存在的文化偏见。 |
| [^296] | [SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance](https://arxiv.org/abs/2508.11857) | SupraTok是一种跨越空白符边界的创新分词器，通过熵筛选、PMI引导的课程训练和多语言处理三大模块，在压缩率上比标准BPE提升17.5%，并比SuperBPE训练快2.1倍。 |
| [^297] | [BiasGym: A Simple and Generalizable Framework for Analyzing and Removing Biases through Injection](https://arxiv.org/abs/2508.08855) | 提出BiasGym框架，通过在冻结的LLM中注入特定偏见信号，再利用这些信号定位并抑制或引导导致偏见行为的模型组件，实现偏见的可靠分析与消除。 |
| [^298] | [Evaluating Style-Personalized Text Generation: Challenges and Directions](https://arxiv.org/abs/2508.06374) | 本文针对风格个性化文本生成评估指标的可靠性问题，批判性检验了BLEU、嵌入向量和LLM评判者等常用指标的有效性，并提出了一个涵盖三个领域、八项写作任务的风格判别基准来系统评估这些指标。 |
| [^299] | [FaST: Feature-aware Sampling and Tuning for Personalized Preference Alignment with Limited Data](https://arxiv.org/abs/2508.04698) | 该论文提出了FaST方法，通过利用从数据中自动发现的高层特征实现高度参数高效的有限数据个性化偏好对齐，并引入DnD和ELIP两个数据集来支持PPALLI问题的研究。 |
| [^300] | [BOW: Training Language Models to Reason Over Plausible Next Words](https://arxiv.org/abs/2506.13502) | BOW是一个强化学习框架，通过让模型生成自包含、中立且全面的合理下一个词空间描述，并由不接触语境的冻结评分器据此打分，从而训练语言模型对多个合理的下一个词进行推理，避免单一续写偏好导致的自我强化偏差。 |
| [^301] | [InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing](https://arxiv.org/abs/2505.22156) | 该论文提出InComeS框架，通过将每个编辑上下文压缩为特殊摘要token的KV缓存并结合选择机制，突破大语言模型上下文窗口的限制，实现高效可扩展的上下文学习式模型编辑。 |
| [^302] | [A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](https://arxiv.org/abs/2505.12781) | 提出低秩克隆（LRC）高效预训练方法，利用一组低秩投影矩阵同时实现教师权重的软剪枝和学生激活（含FFN信号）的克隆对齐，从而高效构建与强大教师模型行为等价的小语言模型。 |
| [^303] | [GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods](https://arxiv.org/abs/2502.16903) | 该论文提出GuidedBench基准和集成逐案评估指南的GuidedEval评估系统，将LLM越狱方法评估中评估者间的差异降低至少76.03%，实现更准确、可靠和可复现的越狱有效性评估。 |
| [^304] | [Automatic Item Generation for Personality Situational Judgment Tests with Large Language Models](https://arxiv.org/abs/2412.12144) | 本研究开发并评估了一个基于大语言模型（GPT-4和ChatGPT-5）自动生成人格情境判断测试题目的结构化、可推广框架，通过三项研究系统考察了提示词设计与温度设置对题目内容效度的影响，显著降低了传统SJT开发对专家的依赖。 |

# 详细

[^1]: 超越分数：理解摘要评估中“大模型作为评判者”的内部机制

    Beyond Scores: Understanding LLM-as-a-Judge Mechanisms in Summarization Evaluation

    [https://arxiv.org/abs/2609.01604](https://arxiv.org/abs/2609.01604)

    该论文通过八种攻击扰动分类法与因果追踪、注意力头敲除等可解释性技术，首次从机制层面揭示LLM评估器（Themis与Prometheus）在摘要评分时采用两阶段内部流程：第15层以下注意力执行局部错误比较并路由信号，其上由MLP级联完成信号整合。

    

    基于大语言模型（LLM）的自然语言生成（NLG）质量评估器已被广泛用作评分工具和自动化训练信号，然而它们给出评分的内部过程仍鲜为人知。我们从机制层面对这一过程展开研究：提出了一个覆盖NLG质量中可读性（Readability）与充分性（Adequacy）两个维度的八种攻击扰动分类体系；构建了一个生成流程，可产生错误强度可控、并附带显式词元级修改映射的“干净-受损”成对摘要；并设计了一组包含四个实验的测试组合，运用因果追踪（causal tracing）、logit-lens词表投影和注意力头敲除（attention-head knockout）等技术，对Themis（Llama-3-8B）和Prometheus（Mistral-7B）两个评估模型进行分析。结果表明，两个评估器都实现了一条结构化、连贯的两阶段评估流水线：在第15层以下，注意力机制执行局部错误比较，并将结果路由至最终输入位置；在第15层之上，MLP级联整合该信号并……

    arXiv:2609.01604v1 Announce Type: new  Abstract: LLM-based evaluators of natural language generation (NLG) quality are widely deployed as scoring tools and as automated training signals, yet the internal procedure by which they assign a rating remains poorly understood. We investigate this procedure mechanistically through an eight-attack perturbation taxonomy across the Readability and Adequacy dimensions of NLG quality, a generation pipeline that produces paired clean and corrupt summaries with controlled error intensity and explicit token-level modification maps, and a four-experiment battery of causal tracing, logit-lens vocabulary projection, and attention-head knockout applied to Themis (Llama-3-8B) and Prometheus (Mistral-7B). Both evaluators implement a structured, coherent evaluation pipeline operating in two stages: below layer 15, attention performs local error comparison and routes the result to the final input position; above it, the MLP cascade integrates the signal and w
    
[^2]: 基于轨迹感知评估的高效软件工程（SWE）智能体基准测试

    Efficient SWE Agent Benchmarking via Trajectory-Aware Evaluation

    [https://arxiv.org/abs/2609.01603](https://arxiv.org/abs/2609.01603)

    提出PTA-IRT框架，将历史执行轨迹作为特权信息融合过程与结果信号，在低校准预算下更准确地恢复软件工程智能体的完整基准分数与排名。

    

    在真实基准上评估软件工程智能体的成本很高，因为每个任务可能需要多步代码探索、修改和测试执行。现有的高效评估方法通过选择代表性子集来估计完整基准的性能，但它们在很大程度上仅依赖结果：它们拟合历史的通过/失败响应矩阵或静态任务语义，丢弃了智能体如何解决问题的信息。我们提出了PTA-IRT，一个融合过程与结果信号的特权轨迹感知项目反应理论框架。历史执行轨迹提供了超越通过/失败的过程级证据，例如探索的上下文、尝试的编辑和解题路径，PTA-IRT将这些作为特权信息用于校准子集选择和能力估计。在低校准预算下，PTA-IRT在四个软件工程基准上的分数和排名恢复方面始终优于现有的IRT基线方法。代码和数据均已公开。

    arXiv:2609.01603v1 Announce Type: cross  Abstract: Evaluating software engineering agents on realistic benchmarks is costly, since each task may require multi-step code exploration, modification, and test execution. Existing efficient evaluation methods select representative subsets to estimate full-benchmark performance, but are largely result-only: they fit historical pass/fail response matrices or static task semantics, discarding how agents solve problems. We propose PTA-IRT, a Privileged Trajectory-Aware Item Response Theory framework that fuses process and outcome signals. Historical execution trajectories supply process-level evidence beyond pass/fail, such as explored context, attempted edits, and solving paths, which PTA-IRT uses as privileged information for calibration subset selection and ability estimation. Under low calibration budgets, PTA-IRT consistently outperforms prior IRT baselines on score and ranking recovery across four SWE benchmarks. Code and data are publicly
    
[^3]: 面向仓库级代码生成的自适应关键Token感知检索

    Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation

    [https://arxiv.org/abs/2609.01601](https://arxiv.org/abs/2609.01601)

    该论文提出ACToR，通过识别LLM自回归代码生成过程中容易出错的关键token位置，并自适应地为这些位置检索细粒度的仓库上下文，从而提升仓库级代码生成的功能正确性。

    

    仓库级代码生成任务需要合成既满足任务要求、又与目标仓库上下文保持一致的代码。由于真实世界的仓库往往超出大语言模型（LLM）的输入长度限制，现有方法通常采用检索增强生成（RAG）来提供仓库特定的上下文。尽管这些方法改善了仓库上下文的检索，但它们通常将上下文作为任务级支持来提供，而没有显式识别生成过程中需要细粒度仓库上下文的关键token。在LLM的自回归生成过程中，错误往往集中在少数决定性位置上：一旦这些token被错误生成，后续代码可能沿着错误的语义路径发展，最终导致功能失效。我们将这些位置称为“关键token”。在本文中，我们提出了ACToR，一种自适应关键（token感知检索框架，摘要在此处截断）

    arXiv:2609.01601v1 Announce Type: cross  Abstract: The repository-level code generation task requires synthesizing code that satisfies task requirements while remaining consistent with the target repository context. Since real-world repositories often exceed the input length limits of LLMs, existing approaches commonly adopt retrieval-augmented generation (RAG) to provide repository-specific context. Despite improving repository-context retrieval, existing methods typically provide context as task-level support, without explicitly identifying the critical tokens that require fine-grained repository context during generation. During the autoregressive generation process of LLMs, errors often concentrate at a small number of decisive positions: once such tokens are generated incorrectly, subsequent code may follow an incorrect semantic path and eventually lead to functional failure. We refer to these positions as "critical tokens". In this paper, we propose ACToR, an adaptive critical to
    
[^4]: CordisBench：语言模型能否对动态智能体框架中的组件生命周期进行推理？

    CordisBench: Can Language Models Reason About Component Lifecycles in Dynamic Agent Harnesses?

    [https://arxiv.org/abs/2609.01600](https://arxiv.org/abs/2609.01600)

    该论文提出了 CordisBench——一个包含 1,200 道题目的基准，用于评估语言模型在动态智能体框架中对组件依赖与清理等生命周期问题的推理能力，发现模型在小规模系统上表现良好，但随相关交互数量增多可靠性显著下降。

    

    动态智能体框架允许语言模型改变塑造其自身执行过程的软件。这种灵活性带来了新的推理负担：局部的插件变更可能通过依赖关系和清理过程进行传播。我们提出了 CordisBench，一个包含 1,200 道题目的生命周期推理基准。该基准将受控的形式化设定与针对 Cordis（一个管理组件依赖与清理的运行时环境）执行的程序相结合，要求模型识别受影响的组件、预测特定拆卸顺序后的状态、判断哪些条件在所有或部分顺序下成立，并选择在实际执行时能够成功的重新配置方案。在这些任务上，我们在低推理努力设置下评估了三个面向效率的模型，涉及 2、4、8、16、24 或 32 个相关交互，并使用确定性的任务特定评分。模型通常能较好地处理小型系统，但随着相关交互数量的增加，其可靠性逐渐下降，尤其是在……（原文截断）

    arXiv:2609.01600v1 Announce Type: cross  Abstract: Dynamic agent harnesses let language models change the software that shapes their own execution. This flexibility brings a new reasoning burden: a local plugin change can propagate through dependencies and cleanup. We introduce CordisBench, a 1,200-question benchmark of this lifecycle reasoning. It combines a controlled formal setting with programs executed against Cordis, a runtime that manages component dependencies and cleanup, and asks models to identify affected components, predict state after a specified teardown order, determine which conditions hold under all or some orders, and choose reconfigurations that succeed when executed. Across these tasks, we evaluate three efficiency-oriented models at low reasoning effort with 2, 4, 8, 16, 24, or 32 relevant interactions, using deterministic task-specific scoring. Models usually handle small systems well but grow less reliable as more interactions become relevant, especially when pr
    
[^5]: 言语强化学习的兴起

    The Rise of Verbal Reinforcement Learning

    [https://arxiv.org/abs/2609.01597](https://arxiv.org/abs/2609.01597)

    本文首次对“言语强化学习”这一新兴范式进行了统一阐述，根据言语反馈生效的时机与作用对象，将其系统归纳为语言作为基础定位信号、语言作为审慎反馈以及语言作为学习信号三大支柱。

    

    自然语言正在成为改进语言智能体的主要反馈渠道，它能够以人类和现代语言模型都可解读的形式传达意图、偏好和因果结构。我们将这一范式称为言语强化学习（Verbal Reinforcement Learning, VRL），并首次对其进行了统一的阐述。我们围绕一个单一的主轴来组织该领域，即言语反馈在智能体生命周期中“何时”生效以及“修改什么”，由此归纳出三大支柱：（1）语言作为基础定位信号，语言通过指定目标、状态和奖励结构来定义任务本身；（2）语言作为审慎反馈，自然语言在测试时引导推理，而无需更新模型参数；（3）语言作为学习信号，基于语言的反馈通过训练来塑造模型参数。在每个支柱内，我们综合了代表性工作，区分了关键……

    arXiv:2609.01597v1 Announce Type: cross  Abstract: Natural language is emerging as a primary feedback channel for improving language agents, capable of conveying intent, preferences, and causal structure in forms interpretable by both humans and modern language models. We call this paradigm Verbal Reinforcement Learning (VRL) and offer the first unified account of it. We organize the field around a single axis, \textit{when} verbal feedback takes effect in an agent's lifecycle and \textit{what} it modifies, yielding three pillars: (1) \textbf{Language as Grounding Signal}, where language defines the task itself by specifying goals, states, and reward structures; (2) \textbf{Language as Deliberative Feedback}, where natural language guides reasoning at test time without the need to update model parameters; (3) \textbf{Language as Learning Signal}, where language-based feedback shapes model parameters through training. Within each pillar, we synthesize representative work, distinguish ke
    
[^6]: StudentSim：训练基于大语言模型的学生模拟器

    StudentSim: Training LLM-based Student Simulators

    [https://arxiv.org/abs/2609.01591](https://arxiv.org/abs/2609.01591)

    提出StudentSim训练框架，通过“汇总训练+逐学生专项化”的方法，将稀疏的个体学生数据转化为既能如实模拟学生真实回答、又能在导师指导下更新能力的个性化LLM学生模拟器，并配套发布覆盖国际象棋、英语二语写作和数学共60名学生的标准化评测协议StudentSimEval。

    

    当AI导师能够适应每个学生的优势、劣势以及偏好的指导方式时，其作用最为显著，但关于哪种指导对哪个学生有效的证据，从真实学习者那里收集起来既稀疏、缓慢又成本高昂。学生模拟器可以作为代理来提供这种信号，然而现有方法都存在局限：状态跟踪模型虽然能拟合学生行为，却难以处理解释或纠正；而基于大语言模型的角色扮演虽然能流畅地遵循指导，却无法可靠地匹配被模仿学生的实际能力水平。我们提出StudentSim，这是一个训练框架，通过先进行汇总训练、再进行逐学生专项化训练的方式，将稀疏的个体学生数据转化为个性化的模拟器。由此得到的模拟器既能如实反映学生自身的回答，又能在导师的指导下对这些回答进行更新。我们还引入了StudentSimEval，一个涵盖国际象棋、英语二语写作和数学三个领域共60名学生的标准化评测协议。

    arXiv:2609.01591v1 Announce Type: new  Abstract: AI tutors are most useful when they adapt to each student's strengths, weaknesses, and preferred guidance, but evidence about which guidance works for which student is sparse, slow, and costly to collect from real learners. Student simulators can provide this signal as a proxy, yet existing approaches are limited: state-tracking models fit student behavior but struggle to process explanations or corrections, while LLM role-play follows guidance fluently but does not reliably match the competence of the student being imitated. We present StudentSim, a training framework that turns sparse per-student data into individualized simulators through pooled training followed by per-student specialization. The resulting simulators both mirror a student's own responses and update them under tutor guidance. We also introduce StudentSimEval, a standardized protocol covering 60 students across chess, second-language English writing, and mathematics, u
    
[^7]: 设计写作中的主动式思维伙伴

    Designing Proactive Thought Partners for Writing

    [https://arxiv.org/abs/2609.01588](https://arxiv.org/abs/2609.01588)

    本文提出并探索了“主动式思维伙伴”的设计空间——一种能在写作过程中主动提供可定制高层次认知支持的AI智能体，通过一周的部署研究发现用户会以前瞻性规划配置支持、将建议用于创意生成与自我监控，并重视轻量级的视觉呈现。

    

    写作涉及从构思到修改的多种认知活动，写作者的需求因个体和时刻而异。主动式AI有望在正确的时机提供恰当的支持，然而现有的主动式工具大多专注于通用的文本辅助，例如自动补全。本文研究了主动式思维伙伴的设计空间：这是一类在写作过程中主动提供可定制、更高层次认知支持的AI智能体。我们通过一个技术探针将这一概念实例化，并与16名参与者共同部署使用了一周。该探针允许用户通过配置角色和主动性来创建伙伴。当用户写作时，相关的伙伴会在适当的时机主动提供建议。我们的研究发现，参与者通过前瞻性规划来配置主动式支持，将建议同时用于创意生成和自我监控，并重视轻量级的视觉呈现方式。

    arXiv:2609.01588v1 Announce Type: cross  Abstract: Writing involves diverse cognitive activities, from ideation to revision, and writers' needs vary across individuals and moments. Proactive AI promises to provide the right support at the right time, yet existing proactive tools largely focus on generic textual assistance, such as autocomplete. This paper studies the design space of proactive thought partners: AI agents that proactively offer customizable, higher-level cognitive support during writing. We instantiated this concept in a technology probe and deployed it with 16 participants for one week. The probe allows users to create partners by configuring their roles and proactivity. As users write, relevant partners take the initiative at appropriate moments to offer suggestions. Our findings show that participants configured proactive support through prospective planning, used suggestions for both idea generation and self-monitoring, and valued lightweight visual representations a
    
[^8]: 大语言模型中量化损伤的结构：为什么下一个比特应该被全局分配

    The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally

    [https://arxiv.org/abs/2609.01587](https://arxiv.org/abs/2609.01587)

    该研究通过因果混合精度干预实验发现，LLM的量化损伤是弥散分布的而非集中于特定电路、计算位置或权重统计，因此在匹配精度预算下，将额外比特全局用于更精细的量化粒度比局部修复少数层更有效。

    

    训练后量化（PTQ）被广泛用于降低大语言模型（LLM）的服务成本，但其精度损失并不均匀，且通常需要针对每个模型单独调优。我们研究了量化损伤发生在何处，以及如何分配少量额外的精度预算。以因果混合精度干预作为基准真值（依次将每一层提升至8比特并测量其所能恢复的精度），我们在4个架构家族的9个开源权重模型上测试了3个直观假设：量化损伤存在于任务电路中、存在于模型计算发生之处，或存在于权重统计特性之中。然而，这些假设都无法预测哪些层会从恢复的精度中受益。相反，恢复是弥散性的：在9个模型中有8个，恢复75%的精度差距大约需要一半的层；唯一的例外是Qwen3-8B，其恢复高度集中。在匹配的精度预算下，将预算全局用于更精细的量化粒度，优于局部修复最具可恢复性的层。

    arXiv:2609.01587v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) is widely used to reduce the cost of serving large language models (LLMs), but its accuracy cost is uneven and is often tuned per model. We study where quantization damage occurs and how to allocate a small additional precision budget. Using causal mixed-precision intervention as ground truth (raise each layer to 8-bit in turn and measure the accuracy it recovers) across 9 open-weight models in 4 architecture families, we test 3 intuitive hypotheses: that quantization damage lives in task circuits, where the model computes, or in weight statistics. None of them predicts which layers benefit from restored precision. Recovery is instead diffuse: for 8 of 9 models, recovering 75% of the gap takes roughly half the layers; the lone exception, Qwen3-8B, is sharply concentrated. At a matched precision budget, spending it globally on finer quantization granularity beats locally repairing the most recoverable la
    
[^9]: 弥合文档视觉语言模型的成本-质量差距：难度感知的数据筛选与质量调整后的部署经济学

    Closing Cost-Quality Gap in Document VLMs: Difficulty-Aware Data Curation and Quality-Adjusted Deployment Economics

    [https://arxiv.org/abs/2609.01575](https://arxiv.org/abs/2609.01575)

    该论文提出了一个已部署的文档理解系统，基于混合专家架构的VLM（35B总参数、3B激活参数），通过难度感知数据筛选流水线进行微调，在单张H100上即可运行，性能超越大至一个数量级的可部署基线，并将预期成本较人工标注降低80%以上。

    

    在受监管行业中，每年从数亿份文档中提取结构化字段仍然成本高昂：定制的OCR级联只能覆盖一小部分工作流程，隐私规则禁止使用外部模型，而现有达到质量阈值的开源VLM其服务成本高于人工标注。我们提出了一套已部署的文档理解系统，该系统基于混合专家视觉语言模型构建（总参数量35B，激活参数量3B），在内部生产数据与开放领域文档的混合数据上进行微调，其中开放领域文档由难度感知流水线依据版面多样性、事实可提取性和跨模型一致性进行筛选。该模型可适配单张H100 GPU，并通过提示词服务异构工作流程，其性能领先所有可部署（非推理）基线模型，包括参数量大至一个数量级的模型。一项质量调整后的成本分析（其中确认与修正成本根据生产遥测数据进行校准）表明，与人工标注相比，该模型将预期成本降低了80%以上。

    arXiv:2609.01575v1 Announce Type: new  Abstract: Extracting structured fields from hundreds of millions of documents annually remains costly in regulated industries: bespoke OCR cascades cover only a fraction of workflows, privacy rules preclude external models, and existing open-source VLMs that clear quality thresholds cost more to serve than human annotation. We present a deployed document-understanding system built on a Mixture-of-Experts VLM (35B total, 3B active), fine-tuned on in-house production data mixed with open-domain documents curated by a Difficulty-Aware pipeline for layout diversity, fact-extractability, and cross-model consistency. Fitting on a single H100 and serving heterogeneous workflows via prompting, the model leads all deployable (non-reasoning) baselines up to an order of magnitude larger. A quality-adjusted cost analysis, with confirmation and correction costs calibrated from production telemetry, shows it reduces expected costs by over 80% against the human 
    
[^10]: 从小型到大型语言模型的近最优SFT-RL标注预算分配扩展

    Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs

    [https://arxiv.org/abs/2609.01573](https://arxiv.org/abs/2609.01573)

    该论文提出“近最优区域”框架来分配SFT-RL标注预算，发现该区域宽广且随模型规模增大而扩大，并能从小型代理模型可靠迁移到大型目标模型，因此小规模代理实验即可替代在大模型上的穷尽式预算搜索。

    

    在大语言模型（LLM）后训练期间，如何在监督微调（SFT）和强化学习（RL）之间分配固定的标注预算仍是一个悬而未决的问题。现有工作仅刻画了宽泛的趋势（例如，在低数据场景下SFT占主导地位），缺乏有原则的分配框架，也没有考察最优比例能否在不同模型规模之间迁移。我们从近最优性的角度来构建这一问题：不再追求单一的SFT-RL最优比例，而是刻画“近最优区域”，即在峰值性能指定容差范围内的所有分配方案集合。实证研究表明，即使容差很小（2-10%），该区域也很宽，且随模型规模增大而变宽，并能可靠地从小型代理模型迁移到大型目标模型。由此得出一种实用策略：只需进行小型代理模型实验即可确定可迁移的近最优区域，从而省去穷尽式的大规模搜索。我们的结果在多种设置下保持一致。

    arXiv:2609.01573v1 Announce Type: cross  Abstract: How to divide a fixed annotation budget between supervised fine-tuning (SFT) and reinforcement learning (RL) during LLM post-training remains an open problem. Existing work characterizes only broad trends (e.g., SFT dominates in low-data regimes), lacks a principled allocation framework, and does not examine whether the optimal ratio transfers across model sizes. We frame this problem in terms of near-optimality: rather than seeking a single optimal SFT-RL ratio, we characterize the near-optimal region, the set of allocations within a specified tolerance of peak performance. Empirically, this region is wide even for small tolerances (2-10%), widens with model scale, and transfers reliably from small proxy models to large target models. This yields a practical strategy: small proxy-model experiments suffice to identify a transferable near-optimal region, eliminating the need for exhaustive large-scale search. Our results hold consistent
    
[^11]: 从生产流量到后训练：构建覆盖企业请求组合的自托管大语言模型

    From Production Traffic to Post-Training: Building a Self-Hosted LLM That Covers the Corporate Request Mix

    [https://arxiv.org/abs/2609.01572](https://arxiv.org/abs/2609.01572)

    本文针对企业自托管LLM因多模型并存导致的GPU资源碎片化问题，沿指令遵循、函数调用和内部任务分布三个维度分别训练GRPO专家模型并通过两阶段SLERP合并，成功将200多个内部应用的流量整合到单一自托管模型上。

    

    数据驻留限制迫使企业自托管大语言模型，但在不淘汰旧模型的情况下持续采用新模型，会不断扩大服务集群，使有限的GPU资源池碎片化。我们通过生产环境错误分析识别出三个维度（指令遵循、函数调用和内部任务分布）的质量差距并加以弥补，从而将来自200多个内部应用的流量整合到单一模型上。质量通过按生产流量分层的离线基准进行追踪，并由确定性验证器或经过校准的LLM评判器进行评分。与其联合优化所有目标（这会引入跨领域奖励干扰），我们为每个维度分别训练一个GRPO专家模型，并通过两阶段SLERP进行合并。每个专家模型的奖励机制都暴露出一种独特的失败模式，即语义坍缩、过度调用和冗长作弊，每一种都需要针对性的领域特定修复方案。在非推理模式下，该方案超越了……（摘要在此处截断）

    arXiv:2609.01572v1 Announce Type: new  Abstract: Data-residency constraints force enterprises to self-host LLMs, but continuous adoption of newer models without decommissioning their predecessors expands the serving fleet, fragmenting a finite GPU pool. We consolidate traffic from over 200 internal applications onto a single model by closing quality gaps identified through production error analysis along three axes: instruction following, function-calling, and internal task distribution. Quality is tracked by offline benchmarks stratified to production traffic and scored by deterministic verifiers or calibrated LLM judges. Rather than optimising all objectives jointly, which introduces cross-domain reward interference, we train a separate GRPO expert per axis and merge them via two-stage SLERP. Each expert's reward exposes a distinct failure mode, namely semantic collapse, over-calling, and verbosity hacking, each requiring a domain-specific fix. In non-reasoning mode the recipe surpas
    
[^12]: 基于熵的选择性智能体引导：从不完美的视觉语言模型教师中学习自主策略

    Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers

    [https://arxiv.org/abs/2609.01567](https://arxiv.org/abs/2609.01567)

    该论文提出SAGE框架，仅在智能体不确定时才查询昂贵的视觉语言模型教师，并利用环境优势对教师建议进行加权蒸馏，从而训练出无需教师引导即可自主行动的轻量级强化学习策略。

    

    视觉语言模型为交互式决策提供了有用的先验知识，但直接将其用作策略既昂贵又脆弱：它们必须在每一步都被查询，无法通过环境交互得到改进，并且可能重复系统性错误。我们研究如何从一个在线、昂贵、不完美但具有信息量的视觉语言模型教师中学习一个廉价的自主策略。我们提出了SAGE（基于熵的选择性智能体引导），这是一个仅在学习者不确定时才查询视觉语言模型的框架，它在训练期间执行其建议的动作，并将引导蒸馏到一个轻量级的强化学习（RL）策略中。由于视觉语言模型的建议并不总是可靠的，SAGE可以使用由环境得出的优势来对教师动作蒸馏进行加权，而不是将所有建议视为同样有用。在稀疏奖励的视觉推理和导航任务中，SAGE学习到的策略在评估时无需视觉语言模型引导即可自主行动，并改进了……

    arXiv:2609.01567v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) provide useful priors for interactive decision-making, but using them directly as policies is expensive and brittle: they must be queried at every step, do not improve from environment interaction, and can repeat systematic errors. We study how to learn a cheap autonomous policy from an online, expensive, and imperfect but informative VLM teacher. We propose SAGE (Selective Agent Guidance via Entropy), a framework that queries a VLM only when the learner is uncertain, executes the suggested action during training, and distills guidance into a lightweight Reinforcement Learning (RL) policy. Because VLM advice is not always reliable, SAGE can weight teacher-action distillation using environment-derived advantages rather than treating all suggestions as equally useful. Across sparse-reward visual reasoning and navigation tasks, SAGE learns policies that act without VLM guidance at evaluation time and improves o
    
[^13]: 从困惑到清晰：面向文本分类的困惑感知检索与知识注入

    From Confusion to Clarity: Confusion-Aware Retrieval and Knowledge Injection for Text Classification

    [https://arxiv.org/abs/2609.01564](https://arxiv.org/abs/2609.01564)

    该论文提出一个无需微调的框架，通过识别模型易混淆的标签对、扩充候选集并生成针对性的区分规则注入知识，帮助大语言模型在语义相似标签的文本分类任务中做出正确选择，且这些规则还可迁移到更小、成本更低的模型上。

    

    大型语言模型（LLM）在将文本分类到包含许多语义相似标签的分类体系时表现不佳，因为这些标签之间的区别是特定领域的，且未被预训练所捕捉。为了处理大型标签空间，一种常见的方法是通过嵌入相似度检索前K个候选标签，并提示LLM在其中进行选择。然而，前K检索虽然减少了候选数量，却无法帮助模型区分相似的标签。当两个相似的标签同时作为候选出现时，模型缺乏在它们之间做出正确选择的信号。我们提出了一个框架，该框架能够：(1) 识别模型难以区分的标签对，(2) 扩大候选集以纳入易混淆的标签，(3) 生成有针对性的规则来区分相似的候选标签。该框架无需微调，且生成的规则可以迁移到更小、更便宜的模型上。在三个基准测试（WOS、Flipkart、LEDGAR）上，我们的方法……

    arXiv:2609.01564v1 Announce Type: cross  Abstract: Large language models (LLMs) struggle to classify text into taxonomies with many semantically similar labels, as the distinctions are domain-specific and not captured by pre-training. To handle large label spaces, a common approach retrieves top-$K$ candidate labels by embedding similarity and prompt the LLM to choose among them. However, top-$K$ retrieval reduces the number of candidates but does not help the model tell similar ones apart. When two similar labels both appear as candidates, the model lacks the signal to choose correctly between them. We propose a framework that (1) identifies which label pairs the model struggles to distinguish, (2) expands the candidate set to include confusable labels, and (3) generates targeted rules to differentiate between similar candidates. The framework requires no fine-tuning, and the generated rules transfer to smaller, cheaper models. On three benchmarks (WOS, Flipkart, LEDGAR), our approach
    
[^14]: 一种构建半导体供应链机会与风险矩阵的系统性方法

    A systematic Approach to constructing a Chance-and-Risk Matrix for Semiconductor Supply Chains

    [https://arxiv.org/abs/2609.01563](https://arxiv.org/abs/2609.01563)

    该论文提出了一种端到端的自动化流水线，利用大语言模型从半导体企业的公开披露文件中提取风险与机会，构建知识图谱并通过三层排序机制生成供应链机会与风险矩阵，独立验证显示92.6%的条目有效且排序结果与专家判断高度一致。

    

    半导体供应链面临着地缘政治紧张局势、地理集中以及快速技术变革带来的日益加剧的风险，然而目前尚无可持续地从公开企业披露信息中提取、结构化并优先排序风险情报的可扩展系统。我们提出了一个端到端的流水线，该流水线检索半导体公司的企业文件，并使用大语言模型（LLM）提取其中所描述的风险与机会。系统将这些内容组织成一个知识图谱，将每个条目与其类别、来源及相关事件相链接，然后合并重复项，并通过结合算法公式、LLM相关性调整和专家验证的三层机制对其进行排序。将该流水线应用于价值链上的五家公司后，共产生了76,207个评分条目，其中独立检查发现92.6%的条目有效。自动排序结果与专家判断相符，风险排序的平均Spearman相关系数为0.55。

    arXiv:2609.01563v1 Announce Type: new  Abstract: Semiconductor supply chains face escalating risks from geopolitical tensions, geographic concentration, and rapid technological shifts, yet no scalable system continuously extracts, structures, and prioritizes risk intelligence from public corporate disclosures. We present an end-to-end pipeline that retrieves corporate documents for semiconductor companies and uses large language models (LLMs) to extract the risks and opportunities they describe. It organizes these into a knowledge graph linking each item to its category, sources, and related events, then merges duplicates and ranks them with a three-layer mechanism combining an algorithmic formula, an LLM relevance adjustment, and expert validation. Applied to five companies across the value chain, the pipeline produces 76,207 scored items, of which an independent check finds 92.6% valid. The automated rankings match expert judgment at an average Spearman correlation of 0.55 for risks 
    
[^15]: SDARE-Bench：评估大语言模型在二元与群体对话中的会话污名检测与回应能力

    SDARE-Bench: Evaluating Large Language Models on Conversational Stigma Detection and Response in Dyadic and Group Dialogue

    [https://arxiv.org/abs/2609.01548](https://arxiv.org/abs/2609.01548)

    本研究提出了首个基于情境的基准测试SDARE-Bench，用于评估大语言模型在二元与群体对话中的污名检测和开放式回应生成能力，发现模型对污名的识别能力较差，且在群体对话中污名表达更多、抵制更弱、建议更不切实际。

    

    大语言模型越来越多地被用于寻求建议和可能影响社会判断的决策。尽管污名对个人和社区有着深远的影响，但相关的基准测试仍然稀缺。现有的通用领域评估通常依赖于静态提示和固定格式的任务，忽略了日常交流中的会话情境和听众效应。为了填补这些空白，我们提出了SDARE-Bench，这是首个基于情境的基准测试，用于同时评估大语言模型的污名检测和开放式回应生成能力，包含1,138个二元对话查询和1,388个群体对话。在8个大语言模型上的实证结果一致表明，模型对污名成分的识别能力较差，尤其是在群体对话中。在开放式回应生成方面，群体情境下的污名表达显著高于二元对话情境，且对污名的抵制更弱，给出的建议也更不切实际。回应通过基于分类的方法进行评估……

    arXiv:2609.01548v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used in advice seeking and decision making that may affect social judgements. Despite stigma's profound effects on people and communities, benchmarks remain scarce. Existing general-domain evaluations typically rely on static prompts and fixed-format tasks, overlooking conversational contexts and audience effects in everyday communication. To address these gaps, we introduce SDARE-Bench, the first scenario-based benchmark evaluating both stigma detection and open-ended response generation in LLMs, comprising 1,138 dyadic queries and 1,388 group dialogue. Empirical results across 8 LLMs consistently demonstrate poor identification of stigma components, especially in group dialogues. In open-ended response generation, stigma expression was substantially higher in group settings than in dyadic, with weaker resistance to stigma and more unrealistic advice. Responses were evaluated using a classif
    
[^16]: 中期训练中的知识蒸馏更利于推理而非事实记忆

    Knowledge Distillation During Mid-Training Favors Reasoning over Factual Recall

    [https://arxiv.org/abs/2609.01532](https://arxiv.org/abs/2609.01532)

    该研究发现前向KL知识蒸馏在预训练阶段能同时提升推理与事实记忆能力，但在中期训练阶段会减缓事实记忆的习得而持续提升推理能力，这种阶段依赖性源于教师置信度在不同数据领域的不对称以及学生模型知识状态的演化。

    

    基于Logit的知识蒸馏（KD）通过更强教师模型的监督来训练更小的语言模型（LM），但其收益是否在各训练阶段保持一致仍不清楚。通过受控实验，我们发现采用后训练教师模型的前向Kullback-Leibler（KL）蒸馏——即标准的KD形式——在中期训练（即在精选语料上进行自监督学习的中间阶段）中表现出根本不同的行为。令人惊讶的是，在预训练阶段，相对于标准的下一词元预测（NTP），前向KD能同时提升推理能力与事实记忆能力；但在中期训练阶段，它却在推理能力持续提升的同时减缓了事实记忆的习得。我们将这种阶段依赖性归因于教师模型在不同数据领域上的置信度不对称，以及学生模型不断演化的知识状态：教师模型在程序性数据上比在知识密集型数据上更具信心，而学生模型……（原文摘要在此处截断）

    arXiv:2609.01532v1 Announce Type: new  Abstract: Logit-based knowledge distillation (KD) is used to train smaller language models (LMs) via supervision from stronger teachers, but whether its benefits are consistent across training stages remains unclear. Through controlled experiments, we find that forward Kullback-Leibler (KL) distillation--the standard KD formulation--with post-trained teachers behaves fundamentally differently during mid-training, an intermediate phase of self-supervised learning on curated corpora. Surprisingly, while forward KD simultaneously improves reasoning and factual recall during pre-training relative to standard next-token prediction (NTP), it instead slows factual recall acquisition during mid-training despite continued reasoning gains. We trace this stage dependence to an asymmetry in teacher confidence across data domains and the student's evolving knowledge state: teachers are more confident on procedural than knowledge-intensive data, while students 
    
[^17]: GlossoGen：复杂多智能体LLM交互中的涌现语言

    GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions

    [https://arxiv.org/abs/2609.01491](https://arxiv.org/abs/2609.01491)

    本文提出GlossoGen平台，通过SaveVeyru压力沟通场景证实LLM多智能体之间会涌现语言演化，产生的语言具有组合性和形态生成性但人类无法理解，并发现效率压力、模型能力和“事后复盘”阶段是语言演化的关键条件。

    

    LLM智能体之间相互交互的日益增多引发了关于多LLM智能体环境中语言演化的关键问题，这对安全性和可监控性以及对LLM的语言学阐释都具有重要意义。为了解决这些问题，我们引入了GlossoGen，一个用于研究复杂场景下多智能体语言演化的新型平台。在GlossoGen中，我们构建了SaveVeyru场景，该场景要求拥有部分信息的智能体在压力下进行沟通。我们发现LLM智能体之间确实会发生语言演化，所产生的语言具有组合性和形态生成能力，并且它们偏离了LLM的英语先验，从而使其对人类而言难以理解。此外，我们识别出了这种语言演化所必需的几个要素：朝着效率方向的压力；支撑智能体的模型的能力强度；以及能够进入“事后复盘”阶段的机会，在该阶段中智能体可以就某些内容达成一致……

    arXiv:2609.01491v1 Announce Type: cross  Abstract: The growing rate at which LLM agents interact with one another raises key questions about language evolution in multi-LLM-agent settings, with implications for safety and monitorability as well as for linguistic accounts of LLMs. To address these questions, we introduce GlossoGen, a novel platform for studying multi-agent language evolution in complex scenarios. Within GlossoGen, we build the SaveVeyru scenario, which requires agents with partial information to communicate under pressure. We find that language evolution does occur between LLM agents, that the resulting languages are compositional and morphologically productive, and that they deviate from the LLMs' English prior in ways that render them incomprehensible to humans. Moreover, we identify several qualities essential to this evolution: pressure towards efficiency; the strength of the models backing the agents; and access to a "postmortem" stage in which agents can agree on 
    
[^18]: AutoConcept：面向元数据可用的组合图像检索的无训练概念引导重排序

    AutoConcept: Training-Free Concept-Guided Reranking for Metadata-Available Composed Image Retrieval

    [https://arxiv.org/abs/2609.01456](https://arxiv.org/abs/2609.01456)

    提出AutoConcept，一种无需训练的概念引导重排序方法，通过将概念证据转化为可解释的结构化记忆并结合推理时校准，在元数据可用的组合图像检索中显著提升早期排名表现。

    

    组合图像检索（CIR）根据参考图像和文本修改描述来检索目标图像。本文研究元数据可用的CIR重排序任务，即由固定的CIR模型首先返回候选池，随后利用图库元数据进行第二阶段的概念引导打分。我们提出AutoConcept，一种无需训练的重排序器，它将概念证据转换为可解释的记忆结构。AutoConcept过滤噪声概念，通过辅助负向惩罚激活与查询相关的正向约束，并通过推理时校准将基础检索得分与基于元数据的概念-候选对齐相结合。在FashionIQ数据集上，AutoConcept相比WeiMoCIR带来了显著的靠前排名提升，并在LinCIR候选池上取得了一致的即插即用增益。元数据感知的对照实验表明，结构化概念记忆在直接的查询-文本匹配和提取属性匹配之外提供了额外信号，而仅查询变体进一步支持了这一结论。

    arXiv:2609.01456v1 Announce Type: cross  Abstract: Composed image retrieval (CIR) retrieves a target image from a reference image and a text modification. This paper studies metadata-available CIR reranking, where a fixed CIR model first returns a candidate pool and gallery metadata is then used for second-stage concept-guided scoring. We introduce AutoConcept, a training-free reranker that converts concept evidence into an interpretable memory. AutoConcept filters noisy concepts, activates query-relevant positive constraints with an auxiliary negative penalty, and combines base retrieval scores with metadata-based concept-candidate alignment through inference-time calibration. On FashionIQ, AutoConcept yields significant early-rank improvements over WeiMoCIR and consistent plug-in gains on LinCIR candidate pools. Metadata-aware controls show that structured concept memory adds signal beyond direct query-text and extracted-attribute matching, while a query-only variant further supports
    
[^19]: HarnessDev：大语言模型能否创造并演化自己的智能体运行框架（Agent Harness）？

    HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?

    [https://arxiv.org/abs/2609.01437](https://arxiv.org/abs/2609.01437)

    提出HarnessDev基准，将评估单元从任务输出转移到可运行的基础设施，考察大语言模型能否从最小种子出发创建完整的智能体运行框架，并利用下游执行反馈迭代演化该框架以提升基准性能。

    

    随着智能体从研究原型走向实际部署的工具，其能力越来越依赖于模型外部的执行基础设施，通常被称为智能体运行框架（agent harness）。在保持模型权重不变的情况下改变这一运行框架，可能会显著改变任务表现。当前的智能体评估通常只报告在选定运行框架下的下游性能，而模型自身开发运行框架的能力相对而言尚未得到充分探索。我们提出了 HarnessDev，这是一个将评估单元从任务输出转移到可运行基础设施的基准。HarnessDev 包含两个阶段：在“创造”阶段，智能体从一个最小化的种子和少量用例出发，构建一个完整的执行系统；在“演化”阶段，智能体从其自建的运行框架出发，利用下游执行反馈对其进行迭代修改，目标是提升基准性能。随后，我们在能力（任务成功率等指标）上评估每个构建出的运行框架（摘要在此处被截断）。

    arXiv:2609.01437v1 Announce Type: cross  Abstract: As agents move from research prototypes to deployed tools, their capability increasingly depends on model-external execution infrastructure, commonly termed the agent harness. Changing this harness while holding model weights fixed can substantially alter task performance. Current agent evaluations typically report downstream performance under a chosen harness, leaving a model's ability to develop the harness itself comparatively underexplored. We introduce HarnessDev, a benchmark that shifts the unit of evaluation from task outputs to runnable infrastructure. HarnessDev covers two stages. In Creation, the agent starts from a minimal seed and a small number of cases, then builds a complete execution system. In Evolution, it starts from its own created harness and iteratively revises it using downstream execution feedback, with the goal of improving benchmark performance. We then evaluate each constructed harness on capability (task suc
    
[^20]: 更少批判性地引用：大语言模型重塑科学引用的修辞与影响力

    Citing Less Critically: LLMs Reshape the Rhetoric and Reach of Scientific Citation

    [https://arxiv.org/abs/2609.01432](https://arxiv.org/abs/2609.01432)

    本文提出掩码引用任务构建反事实引用语料库，对比发现大语言模型的引用明显比人类更少批判性，且在引用对象与修辞意图上呈现系统性偏移，表明LLM正在重塑科学引用的修辞方式与学术影响力分布。

    

    科学引用承载着修辞意图。学者们可以以积极（支持）、消极（对比反驳）或中立（提及）的方式引用先前的工作。随着大语言模型（LLM）日益广泛地辅助科学写作，它们是否以与人类相同的修辞意图再现引用仍不清楚。我们提出了一个掩码引用任务来比较人类与LLM生成的引用行为。对于每个引用上下文，让LLM生成一个替换的引用句子，从而构建出一个与人类引用直接可比的反事实语料库。我们分析了模型引用什么、引用谁以及如何引用，采用“LLM作为评判者”（LLM-as-a-judge）对引用意图进行分类，并利用一个拥有2000万条边的合著作者网络来衡量被引作者之间的社会距离。在六个流行的LLM和1,746篇顶级NLP会议论文（超过6.3万个上下文、13.2万条引用）上的实验揭示了三种模式：（1）与人类引用相比，LLM的引用明显更少批判性；（2）LLM ov…（原文摘要在此处截断）

    arXiv:2609.01432v1 Announce Type: cross  Abstract: Scientific citations carry rhetorical intent. Scholars may cite prior work positively (supporting), negatively (contrasting), or neutrally (mentioning). As large language models (LLMs) increasingly assist scientific writing, whether they reproduce citations with the same rhetorical intent as humans remains unclear. We introduce a masked-citation task to compare human and LLM-generated citation behavior. For each citation context, an LLM generates a replacement citation sentence, producing a counterfactual corpus directly comparable to human citation. We analyze what, whom, and how models cite, using an LLM-as-a-judge to classify citation intent and a 20-million-edge coauthorship network to measure social distance between cited authors. Across six popular LLMs and 1,746 top NLP conference papers (63k+ contexts, 132k+ citations), three patterns emerge: (1) Compared with human citation, LLMs cite significantly less critically; (2) LLMs ov
    
[^21]: 从采样轨迹到训练配方：面向大语言模型的自包含后训练

    From Rollouts to Recipes: Self-Contained Post-Training for LLMs

    [https://arxiv.org/abs/2609.01422](https://arxiv.org/abs/2609.01422)

    提出Self-Routing框架，利用模型自身采样轨迹的正确性与置信度将每个样本自适应路由到GRPO、自蒸馏、正则化或跳过等不同训练方法，无需外部教师和额外标注即可持续提升大语言模型的数学推理能力。

    

    大语言模型的后训练通常对所有样本应用单一的训练配方，然而模型自身的采样轨迹实际上揭示了样本层面不同的学习状态。我们提出Self-Routing，一个基于行为条件的后训练框架，利用采样轨迹的正确性和置信度来决定每个样本应如何被优化。根据其行为状态，每个样本会被路由到GRPO、在策略自蒸馏、正则化或直接跳过，使训练能够自适应进行，而无需外部教师、额外标注或额外采样。在Qwen3和Qwen3.5骨干模型上的数学推理实验表明，Self-Routing始终优于统一的GRPO、统一的OPSD、固定混合策略以及更简单的路由基线。进一步分析显示，路由分布会随训练过程动态变化，并减少了对低信号或已稳定样本的不必要更新。

    arXiv:2609.01422v1 Announce Type: new  Abstract: Post-training large language models usually applies a single training recipe to all samples, even though the model's own rollouts reveal different sample-level learning states. We propose Self-Routing, a behavior-conditioned post-training framework that uses rollout correctness and confidence to decide how each sample should be optimized. Depending on its behavior state, a sample is routed to GRPO, on-policy self-distillation, regularization, or skipping, allowing training to adapt without external teachers, extra annotations, or additional sampling. Experiments on mathematical reasoning across Qwen3 and Qwen3.5 backbones show that Self-Routing consistently improves over uniform GRPO, uniform OPSD, fixed mixtures, and simpler routing baselines. Further analyses show that the routing distribution changes over training and reduces unnecessary updates on low-signal or already stable samples.
    
[^22]: EdiTikZ：基于修订轨迹的科学图表编辑

    EdiTikZ: Scientific Figure Editing from Revision Trajectories

    [https://arxiv.org/abs/2609.01409](https://arxiv.org/abs/2609.01409)

    该论文提出 DaEdiTikZ——首个从 arXiv、GitHub 和 TeX SE 的自然修订轨迹中挖掘的大规模科学图表编辑数据集（包含 39.1 万个 TikZ 编辑对和 78.1 万条推断的编辑指令），并配套构建了人工精修基准 DaEdiTikZ-Bench，以自然修订轨迹作为可扩展的监督信号来训练科学图表编辑模型。

    

    视觉语言模型在从文本或图像生成科学图表方面已展现出强大性能。然而，制作达到出版级别的图表需要反复迭代精修，这使得科学图表编辑成为一项重要却很大程度上未被探索的任务。现有方法依赖于昂贵的专有智能体系统、主要聚焦于评估，或从合成生成的编辑中构建训练监督。与之不同，我们利用自然存在的科学修订与开发轨迹作为可扩展的监督来源。为此，我们提出了 DaEdiTikZ——首个大规模的源于修订记录的科学图表编辑数据集，通过从 arXiv、GitHub 和 TeX SE 中挖掘 39.1 万个合理的 TikZ 编辑对，并使用以渲染图表和 TikZ 代码为条件的视觉语言模型推断出 78.1 万条定向编辑指令而构建。我们进一步推出了经过人工精修、包含 790 个实例的基准 DaEdiTikZ-Bench，并训练……

    arXiv:2609.01409v1 Announce Type: new  Abstract: Vision-language models (VLMs) have shown strong performance in generating scientific figures from text or images. However, producing publication-ready figures requires iterative refinement, making scientific figure editing an important yet largely unexplored task. Existing approaches rely on costly proprietary agentic systems, focus primarily on evaluation, or construct training supervision from synthetically generated edits. Instead, we leverage naturally occurring scientific revision and development trajectories as a scalable source of supervision. To this end, we introduce DaEdiTikZ, the first large-scale dataset of revision-derived scientific figure edits, constructed by mining 391K plausible TikZ edit pairs from arXiv, GitHub, and TeX SE and inferring 781K directed edit instructions with a VLM conditioned on rendered figures and TikZ code. We further introduce DaEdiTikZ-Bench, a human-refined benchmark with 790 instances, and train 
    
[^23]: 当分词悄然成为输出监督

    When Tokenization is Secretly Output Supervision

    [https://arxiv.org/abs/2609.01386](https://arxiv.org/abs/2609.01386)

    该论文提出在自回归模型中分词粒度实质上是一种输出监督信号：通过解耦实验证明，输出分词而非输入分词主导了任务性能、训练动态和模型内部表示的差异，因此模型间的比较可能部分反映的是任务定义的不同而非模型能力差异。

    

    在语言模型中，分词默认被视为一种输入预处理决策。我们认为这种理解是不完整的：在自回归模型中，分词器的粒度决定了模型在单次前向传播中必须解决什么问题，从而决定了模型所接收的监督信号。这既影响学习问题的难度，也影响模型内部涌现的表示。我们通过对输入分词与输出分词进行新颖的解耦，在一个数值推理的受控实验中验证了这一观点。正如输出监督视角所预测的，任务性能、训练动态和模型内部结构的差异由输出分词所引起，而在很大程度上与输入分词无关。这在实践中可能具有重要意义，因为采用不同分词策略的模型不仅在输入表示上存在差异，在它们被训练的任务本身上也存在差异。因此，模型之间的比较可能部分反映的是任务定义的不同，而非模型能力本身的差异。

    arXiv:2609.01386v1 Announce Type: new  Abstract: Tokenization in language models is treated by default as an input preprocessing decision. We argue that this framing is incomplete: in autoregressive models, tokenizer granularity determines what the model must resolve in a single forward pass, and therefore the supervision signal it receives. This affects both the difficulty of the learning problem and the representations that emerge inside the model. We test this in a controlled experiment on numeric reasoning with a novel decoupling of input and output tokenization. As the output supervision view predicts, differences in task performance, training dynamics, and model internals are induced by output tokenization and largely invariant to input tokenization. This may matter in practice, because models with different tokenization strategies differ not only in input representation but in the task they were trained on. Comparisons between models may thus partly reflect task definition rathe
    
[^24]: InSight：交互式可视化中智能体式声明验证的基准测试

    InSight: A Benchmark for Agentic Claim Verification in Interactive Visualizations

    [https://arxiv.org/abs/2609.01383](https://arxiv.org/abs/2609.01383)

    本文提出InSight基准，包含21,349条源自人工分析叙述并嵌入完全交互式网页环境的声明，用于评估智能体在动态交互式可视化中主动探索证据并验证声明真伪（支持、驳斥或无法验证）的能力。

    

    视觉语言模型在解释静态视觉制品方面已展现出卓越的能力，但现代数据分析本质上是动态的，需要对交互式环境进行主动探询。现有的基准测试主要局限于静态图像和一次性问答，无法捕捉该领域的认知需求——在这一领域中，证据经常被遮挡、分布在相互关联的多个视图中，或需要通过用户操作才能有条件地显现。在本文中，我们提出了InSight，一个针对交互式可视化中智能体式声明验证的基准测试。该数据集包含21,349条声明，这些声明源自人工撰写的分析叙述，并建立在完全交互式的网页环境之上。智能体必须在环境中进行导航，以判断在给定可用证据的情况下，一条自然语言声明是被支持、被驳斥还是无法验证。与传统评估不同，InSight……（原文摘要在此处截断）

    arXiv:2609.01383v1 Announce Type: new  Abstract: Vision Language Models have demonstrated remarkable proficiency in interpreting static visual artifacts, but modern data analysis is inherently dynamic, requiring the active interrogation of interactive environments. Existing benchmarks are predominantly constrained to static imagery and one-shot question answering and fail to capture the epistemic demands of this domain, where evidence is frequently occluded, distributed across linked views, or conditionally revealed through user agency. In this paper, we introduce InSight, a benchmark for agentic claim verification over interactive visualizations. The dataset consists of 21,349 claims derived from human-authored analytical narratives and grounded in fully interactive web-based environments. Agents must navigate these environments to determine whether a natural language claim is supported, refuted or not verifiable given the available evidence. Unlike traditional evaluations, InSight tr
    
[^25]: Polish ModernBERT：波兰语理解的长与短

    Polish ModernBERT: The Long and Short of Polish Language Understanding

    [https://arxiv.org/abs/2609.01379](https://arxiv.org/abs/2609.01379)

    本文推出Polish ModernBERT——一族支持8K长上下文的波兰语编码器模型，在30个任务上取得波兰语编码器中的最佳整体性能，并在长上下文任务上显著超越RoBERTa基线。

    

    仅编码器（Encoder-only）的Transformer模型在判别式任务和表示学习任务中依然有效，然而波兰语编码器仍主要依赖于BERT/RoBERTa风格的架构。我们推出了Polish ModernBERT，这是一族包含四个波兰语编码器的模型，涵盖Base和Large两种规模，每种规模均有512-token和8K上下文两种变体。我们通过分阶段选择实验对ModernBERT的预训练方案进行了适配，并发布了一个长上下文基准测试，涵盖法律主题分类、意识形态决策方向预测、文学情节摘要的事实一致性评估以及人权侵犯评估。在30个任务上，Polish ModernBERT在所评估的波兰语编码器中取得了最佳整体性能，Base-8K和Large-8K模型分别达到83.99和85.11。在长上下文任务上，8K变体相比对应的波兰语RoBERTa-8K基线，性能从67.47提升至77.15，从75.88提升至78.49。

    arXiv:2609.01379v1 Announce Type: new  Abstract: Encoder-only Transformers remain effective for discriminative and representation-learning tasks, yet Polish encoders still largely rely on BERT/RoBERTa-style architectures. We introduce \textbf{Polish ModernBERT}, a family of four Polish encoders available at Base and Large scales, each with 512-token and 8K context variants. We adapt the ModernBERT pretraining recipe through staged selection experiments and release a long-context benchmark covering legal topic classification, ideological decision-direction prediction, factual-consistency assessment over literary plot summaries, and human-rights violation assessment. Across 30 tasks, Polish ModernBERT achieves the best overall performance among the evaluated Polish encoders, reaching 83.99 and 85.11 for the Base-8K and Large-8K models, respectively. On long-context tasks, the 8K variants improve over matched Polish RoBERTa-8K baselines from 67.47 to 77.15 and from 75.88 to 78.49 at the B
    
[^26]: IntroConformal：通过内省信号为大视觉语言模型提供保形事实性保证

    IntroConformal: Conformal Factuality Guarantees for Large Vision-Language Models via Introspective Signals

    [https://arxiv.org/abs/2609.01375](https://arxiv.org/abs/2609.01375)

    提出了IntroConformal框架，通过利用模型自身的内省信号（隐状态的逐层语义稳定性和自我验证概率），以免训练的保形风险控制方式为大视觉语言模型的生成内容提供有限样本、无分布假设的事实性保证。

    

    大视觉语言模型（LVLMs）已经取得了强大的多模态性能，但确保生成内容的事实正确性仍然具有挑战性。现有提供事实性统计保证的方法通常依赖外部验证器或生成时的置信度信号，这会引入额外的辅助依赖，或者对于模型自信但错误的输出往往失效。我们提出，可靠的事实性控制可以转而通过从模型自身提取的内省信号来实现。我们介绍了IntroConformal，这是一个免训练的保形风险控制（CRC）框架，能够提供有限样本、无分布假设的事实性保证。我们首先用逐层语义稳定性（一种从隐状态表示中导出的保形分数）对该框架进行实例化，然后提出验证概率——一种更强的分数，用于捕捉模型对声明事实性的自我判断。在多个LVLM（摘要在此处截断）

    arXiv:2609.01375v1 Announce Type: cross  Abstract: Large Vision-Language Models (LVLMs) have achieved strong multimodal performance, yet ensuring the factual correctness of generated content remains challenging. Existing methods that provide statistical guarantees on factuality typically rely on external verifiers or generation-time confidence signals, which introduce auxiliary dependencies or often fail for confident but incorrect outputs. We argue that reliable factuality control can instead be achieved through introspective signals derived from the model itself. We introduce IntroConformal, a training-free Conformal Risk Control (CRC) framework that provides finite-sample, distribution-free factuality guarantees. We first instantiate it with layer-wise semantic stability, a conformity score derived from hidden-state representations, and then propose verification probability, a stronger score capturing the model's self-administered judgment on claim factuality. Across multiple LVLM a
    
[^27]: 行为有效的LoRA写入是稀疏且结构化的

    Behaviorally Effective LoRA Writes Are Sparse and Structured

    [https://arxiv.org/abs/2609.01374](https://arxiv.org/abs/2609.01374)

    本文提出Learned-Basis LoRA方法，通过将无约束适配器的写入列转换为冻结的正交基底并在其中继续训练，揭示出真正承载模型行为的LoRA写入是稀疏、结构化且高度集中的，而非均匀分布于整个低秩参数空间。

    

    低秩适配固定了更新的秩，但它并未识别出训练后的写入中哪些部分真正承载了模型行为。我们直接研究了这个问题，并证明行为有效的LoRA写入是稀疏的、结构化的，且远比原始低秩参数化所暗示的更加集中。我们使用Learned-Basis LoRA（一种学习基底延续训练方法）来揭示这种结构。该方法首先对一个无约束的适配器进行预热训练，将其学习到的写入列转换为按模块划分的正交归一化基底，冻结该基底，然后在受约束的参数化内继续训练。在14次从无约束形式到受约束形式的精确切换中，转换步骤处的保留集准确率保持不变，且重建的写入矩阵的相对Frobenius误差至多为0.25%。同状态的延续训练随后表明，同一个训练检查点在不同的写入子空间下会发展出不同的特性。

    arXiv:2609.01374v1 Announce Type: new  Abstract: Low-rank adaptation fixes the rank of the update, but it does not identify which parts of a trained   write actually carry behavior. We study that question directly and show that behaviorally effective   LoRA writes are sparse, structured, and far more concentrated than the raw low-rank parameterization   suggests.   We use Learned-Basis LoRA, a learned-basis continuation recipe, to expose that structure. The recipe   warms up an unconstrained adapter, converts its learned write columns into a module-wise orthonormal   basis, freezes that basis, and continues training inside the constrained parameterization. Across 14   exact switches from unconstrained to constrained form, held-out accuracy is unchanged at the   conversion step and reconstructed write matrices differ by at most 0.25% relative Frobenius error.   Same-state continuation then shows that the same trained checkpoint develops differently under   different write subspaces, est
    
[^28]: 你的答案有多正确？一个面向开放式问答评估的语义正确性框架

    How Correct Is Your Answer? A Semantic Correctness Framework for Open QA Evaluation

    [https://arxiv.org/abs/2609.01369](https://arxiv.org/abs/2609.01369)

    该论文提出了一个开放式问答答案评估的语义正确性框架，包含八个有序类别的语义分类体系、8.8千样本的CAP-Correctness基准数据集以及用于NLI训练的CAP-Statements陈述转换数据集，解决了现有评估指标无法区分不同类型答案错误的局限。

    

    开放式问答的可靠评估仍然是衡量现代大语言模型答案正确性的一大瓶颈。与多项选择任务不同，自由形式的答案可能以多种表面形式呈现正确内容，也可能以质量上截然不同的方式出错，包括不完整、矛盾、过度生成以及对虚假前提的认可。现有的基于判断和基于相似度的评估指标往往混淆了这些差异。我们通过三项可复用的贡献来填补这一空白。首先，我们提出了一个语义正确性分类体系，将开放式答案划分为八个有序类别，从而将冗余但正确的答案与被幻觉内容污染的答案区分开来。其次，我们发布了CAP-Correctness——一个涵盖广泛使用的问答数据集、包含8.8千个样本的基准数据集，以及CAP-Statements——一个包含1.1万个样本的数据集，用于将问答对转换为陈述句，以支持自然语言推理（NLI）训练和陈述级评估。（注：原始摘要在此处不完整）

    arXiv:2609.01369v1 Announce Type: new  Abstract: Reliable evaluation of open-ended question answering remains a bottleneck for measuring answer correctness of modern LLMs. Unlike multiple-choice tasks, free-form answers may be correct in many surface forms and may fail in qualitatively different ways, including incompleteness, contradiction, overgeneration, and endorsement of false premises. Existing judgment-based and similarity-based metrics often collapse these distinctions. We address this gap with three reusable contributions. First, we introduce a semantic correctness taxonomy that assigns open-ended answers to eight ordered classes, separating verbose-but-correct answers from those contaminated by hallucinated content. Second, we release CAP-Correctness, an 8.8k-example benchmark spanning widely used QA datasets, and CAP-Statements, an 11k-example dataset for converting question-answer pairs into declarative statements for natural language inference (NLI) training and statement-
    
[^29]: 探究医学问答任务中线性探针对语言语域、医学专科与语料库变化的鲁棒性

    Investigating Linear Probe Robustness to Linguistic Register, Medical Specialty, and Corpus Shifts in Medical QA

    [https://arxiv.org/abs/2609.01361](https://arxiv.org/abs/2609.01361)

    该论文构建了一个可独立操控写作语域、医学专科和语料库三类变化的医学问答基准，以系统性探究大语言模型中线性探针（真值方向检测）对不同输入偏移的鲁棒性。

    

    在大语言模型（LLM）隐状态上训练的线性分类器，即线性探针，可以通过单次前向传播来标记事实性错误。从几何角度看，这意味着真与假的陈述在隐状态空间中沿一个稳定的方向分离，即“真值方向”。已有研究对这种能力能否在不同输入偏移下泛化存在分歧，但由于跨数据集的探针迁移实验同时混淆了多种输入变化，这种分歧难以解释。我们在医学问答（QA）任务中分离出三个此类变量：写作风格（语域）、领域（医学专科）和语料库（数据集）。我们基于500条MedQA条目构建了一个基准，每条条目被改写为四种风格（教科书式、患者口吻、临床笔记、口语化），标注了临床专科，并与另外两个考试语料库MedMCQA和MMLU-medical组合，用于跨数据集评估。通过对四个开源权重LLM（2--8B）进行探针实验，我们发现……

    arXiv:2609.01361v1 Announce Type: new  Abstract: Linear classifiers trained on hidden states of a large language model (LLM), linear probes, can flag factual errors from a single forward pass. Geometrically, that implies that true and false statements separate along a stable direction in hidden state space, i.e., the truth direction. Prior work disagrees on whether this generalises across input shifts, but the disagreement is hard to interpret because cross-dataset probe transfer experiments confound several kinds of input change at once. We isolate three such variables in medical question-answering (QA): writing style (register), domain (medical specialty), and corpus (dataset). We build a benchmark using 500 MedQA entries, each rewritten into four styles (textbook, patient, clinical note, colloquial), annotated with clinical specialty, and grouped with two other exam corpora, MedMCQA and MMLU-medical, for cross-dataset evaluation. Probing four open-weight LLMs (2--8B), we find that t
    
[^30]: 将句法与语言分离：多语言大语言模型翻译过程的机制性解释

    Separating Syntax from Language: A Mechanistic Account of Translation in Multilingual LLMs

    [https://arxiv.org/abs/2609.01356](https://arxiv.org/abs/2609.01356)

    本研究通过受控多语言数据集与因果干预、探测实验，首次揭示多语言大模型的翻译过程比以往认为的更加模块化：输出语言的生成可进一步解耦为独立的句法（语序）构建过程与表层语言实现过程，且模型会先确定目标语言的语序，再生成其表层语言形式。

    

    多语言大语言模型在机器翻译中表现出色，但我们对它们将表示从一种语言转换为另一种语言的机制的理解仍不完整。先前的研究表明，翻译在多语言大模型内部可分解为若干可分离的过程：概念内容首先被独立表示，随后被生成为特定语言的形式。在本工作中，我们证明翻译比以往假设的更具模块化特性——翻译过程中的输出语言生成实际上可以进一步分离为句法过程和表层语言过程。我们构建了能够隔离跨语言语序差异的受控多语言数据集，并运用因果干预和探测技术来追踪表示在翻译过程中如何被转换。我们发现，模型会先构建目标侧的语序，然后再实现目标语……（原文摘要在此处截断）

    arXiv:2609.01356v1 Announce Type: new  Abstract: Multilingual large language models (mLLMs) achieve strong performance in machine translation, yet our understanding of the mechanisms by which they transform representations from one language to another remains incomplete. Prior work suggests that translation decomposes into separable processes within an mLLM, where conceptual content is first represented independently, followed by a production into language-specific form. In this work, we show that translation is even more modular than previously assumed and that the output language production in translation processes is actually further separable into a syntax and a surface language process. We construct controlled multilingual datasets that isolate cross-linguistic differences in word-order and use causal interventions and probing to track how representations are transformed during translation. We find that models first construct target-side word-order before realizing the target lang
    
[^31]: 验证器在哪里失效：对RLVR中奖励信号的类别级审计

    Where the Verifier Fails: A Category-Level Audit of Reward Signals in RLVR

    [https://arxiv.org/abs/2609.01354](https://arxiv.org/abs/2609.01354)

    该论文将变异测试从模型转向验证器本身，通过构造保证数学等价的答案变体，在超过30万个判定上对四个主流验证器进行了类别级审计，发现相同输入下验证器的自我验证率相差高达41.3个百分点，揭示了RLVR奖励信号中的系统性假阴性问题。

    

    arXiv:2609.01354v1 公告类型：新论文 摘要：可验证奖励强化学习（RLVR）和标准基准评估都依赖于一个自动验证器，它将自由文本的答案转换为二元奖励。先前的工作报告称，某个评估框架仅接受了约94%的其自身标准答案，并将其归咎于LaTeX解析问题。但这只是一个总体数字：它没有说明哪些答案形式消耗了错误预算。我们提供了这种分解。我们将变异测试应用于验证器而非模型，生成经过认证的等价答案变体，即通过构造保证保持数学意义的改写，因此任何拒绝都是可证明的假阴性，无需人工裁定。然后，我们在307,420个判定上测量了四个广泛使用的验证器对每个答案类别的拒绝率。我们发现了三件事。（1）在相同输入上，自我验证率从53.8%到95.2%不等，相差41.3个百分点。已发表的数字仅描述了其中一个实现，

    arXiv:2609.01354v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) and standard benchmark evaluation both rely on an automatic verifier that turns a free text answer into a binary reward. Prior work reports that one evaluation harness accepts only about 94% of its own ground truth answers, blaming LaTeX parsing. That is an aggregate: it does not say which answer forms consume the error budget. We supply the decomposition. We apply metamorphic testing to the verifier rather than the model, generating certified equivalent answer variants, that is, rewrites that preserve mathematical meaning by construction, so that any rejection is a provable false negative needing no human adjudication. We then measure rejection per answer category across four widely used verifiers over 307,420 verdicts. We find three things. (1) Self validation ranges from 53.8% to 95.2% on identical inputs, a spread of 41.3 points. The published figure describes one implementation, 
    
[^32]: CHARM：面向多文化角色扮演基准的角色幻觉评估

    CHARM: Character Hallucination for Multicultural Role Play Benchmark

    [https://arxiv.org/abs/2609.01352](https://arxiv.org/abs/2609.01352)

    CHARM是一个涵盖五大文化语言区域40个角色的多文化角色扮演基准，创新性地将角色幻觉拆分为“边界意识”与“边界遵守”两个独立阶段进行评估，从而更精细地定位大语言模型角色扮演中幻觉错误的来源。

    

    角色扮演大语言模型（LLMs）被期望既能模仿角色的风格，又能尊重该角色的知识边界。以往的评估方法虽能检测角色幻觉，但很少区分错误究竟是源于未能识别边界，还是源于虽识别了边界却仍未能遵守（继续作答）。我们提出了CHARM，这是一个多文化基准，包含来自五个文化语言区域的40个真实与虚构角色，并经母语评审员验证。该基准探测两类边界：时间边界（历史角色 vs. 现代角色）与跨宇宙边界（角色叙事或历史宇宙之外的实体），并采用允许弃答的多项选择题。我们提出一种两阶段评估方法，将“边界意识”（明确识别出查询超出角色范围）与“边界遵守”（在回答具体问题时选择弃答）区分开来。对六个大语言模型的评估显示，幻觉主要由……（原文在此处截断）

    arXiv:2609.01352v1 Announce Type: cross  Abstract: Role-playing large language models (LLMs) are expected to adopt a character's style while also respecting that character's knowledge boundaries. Prior evaluations detect character hallucination but rarely distinguish whether errors arise from failure to recognize a boundary or from failure to comply despite recognition. We introduce CHARM, a multicultural benchmark of 40 real and fictional characters drawn from five cultural-linguistic regions, and validated by native reviewers. It probes two boundary types, Temporal (historical vs. modern) and Cross-Universe (entities outside a character's narrative or historical universe), using abstention-enabled multiple-choice questions. We propose a two-stage evaluation that separates Boundary-Awareness (explicit recognition that a query is out of scope) from Boundary-Compliance (abstention when answering concrete questions). Evaluations across six LLMs show that hallucination is driven predomina
    
[^33]: 通过训练数据干预探究事实知识的跨语言迁移

    Probing Factual Knowledge Transfer with Training Data Interventions

    [https://arxiv.org/abs/2609.01341](https://arxiv.org/abs/2609.01341)

    该论文提出了一种基于干预的评估框架并构建SIFT数据集，通过从波斯语训练数据中系统性移除特定事实来检验多语言模型的知识跨语言迁移能力，发现英语预训练中习得的事实知识向波斯语的迁移非常有限。

    

    多语言语言模型在持续预训练过程中是否会跨语言迁移事实知识，还是主要通过直接从目标语言数据中学习的内容来回忆事实？为了更可靠地回答这个问题，我们提出了一种基于干预的框架：从一个英语预训练模型出发，我们在波斯语数据上继续预训练，并从这些数据中以不同粒度系统地移除了特定事实。我们构建了SIFT资源，包含覆盖20种关系的500个三元组，按每个事实主语的文化来源分层为通用（全球知名）实体和波斯语相关实体，该资源既用于从训练数据中系统性移除事实，也用于评估，并配有母语撰写的波斯语完形填空模板。我们的结果表明，事实迁移非常有限：在最严格的移除条件下，绝大多数在英语中习得的事实未能迁移到波斯语中。我们进一步表明，句子级……（摘要内容在此截断）

    arXiv:2609.01341v1 Announce Type: cross  Abstract: Do multilingual language models transfer factual knowledge across languages during continued pretraining, or do they mostly recall facts learned directly from the target-language data? To answer this question more reliably, we propose an intervention-based framework: starting from an English-pretrained model, we continue pretraining on Persian data from which specific facts have been systematically removed at varying levels of granularity. We construct SIFT, a resource of 500 triples across 20 relations, stratified by the cultural origin of each fact's subject into general (globally prominent) and Persian-related entities, designed for both systematic fact removal from training data and evaluation, with natively written Persian cloze templates. Our results show that fact transfer is very limited: under the strictest removal condition, a large majority of English-acquired facts fail to transfer into Persian. We further show that sentenc
    
[^34]: VerTox：可验证奖励引导的针对神经排序模型的语料库投毒攻击

    VerTox: Verifiable Reward-Guided Corpus Poisoning Against Neural Ranking Models

    [https://arxiv.org/abs/2609.01325](https://arxiv.org/abs/2609.01325)

    本文提出VerTox，首个将语料库投毒攻击形式化为可验证奖励引导强化学习问题的框架，通过将排序扭曲与事实性破坏耦合的奖励设计将小型LLM微调为对抗性文档生成器，对神经排序模型实现了接近完美的攻击成功率。

    

    神经排序模型已成为现代信息检索系统的核心组件，也是检索增强生成（RAG）流水线等人工智能系统的重要构建模块。然而，在大语言模型（LLM）能够大规模生成流畅且具有欺骗性内容的背景下，这些模型的鲁棒性仍未得到充分理解。本研究探讨了神经排序模型对语料库投毒攻击的脆弱性，此类攻击中，攻击者向语料库注入少量恶意构造的文档以扭曲排序行为。我们提出了VerTox，这是首个将语料库投毒形式化为可验证奖励引导强化学习（RLVR）问题的框架。通过专门的奖励设计将排序扭曲与事实性破坏显式耦合，我们将紧凑型LLM微调为对抗性生成器。实验表明，我们的方法实现了接近完美的攻击成功率。

    arXiv:2609.01325v1 Announce Type: new  Abstract: Neural ranking models have become core components of modern information retrieval systems and important building blocks of AI systems such as retrieval-augmented generation (RAG) pipelines. However, their robustness remains insufficiently understood in the presence of large language models (LLMs), which can generate fluent and deceptive content at scale. This work investigates the vulnerability of neural ranking models to corpus poisoning attacks, in which an adversary injects a small number of maliciously crafted documents into the corpus to distort ranking behavior. We propose VerTox, the first framework to formulate corpus poisoning as a verifiable reward-guided reinforcement learning (RLVR) problem. By explicitly coupling ranking distortion with factual corruption through specialized reward shaping, we fine-tune compact LLMs into adversarial generators. Experiments demonstrate that our method achieves near-perfect attack success rate
    
[^35]: 探索稀疏自编码器在基于文本的因果混杂调整中的应用

    Exploring Sparse Autoencoders in Text-Based Causal Confounding Adjustment

    [https://arxiv.org/abs/2609.01322](https://arxiv.org/abs/2609.01322)

    该论文提出一种基于稀疏自编码器（SAE）的新颖因果调整流程，通过条件独立性检验迭代选取最小特征集合，解决了文本表示在保留混杂变量与满足有限样本重叠条件之间的权衡，在半合成评估中实现了比替代表示更低的偏差和更高的覆盖率。

    

    在许多场景中，基于文本数据研究因果问题需要对文本中的混杂信息进行调整。然而，构建用于调整的文本表示存在一种权衡：文本表示必须足够大和/或稠密，以保留无偏效应估计所需的混杂变量；但又必须足够小和/或稀疏，以满足有限样本的重叠条件并获得低方差的估计。为解决这一权衡问题，我们转向稀疏自编码器（SAE），提出了一种新颖的因果调整流程，该流程通过条件独立性检验迭代地选择一个最小的SAE特征集合。我们发现，在带有二元混杂变量的标准半合成评估中，SAE表示比其他替代表示实现了更好的调整效果（更低的偏差和更高的覆盖率），并且其可解释性为证伪检验提供了机会。我们还引入了一个更贴近真实的半合成评估……

    arXiv:2609.01322v1 Announce Type: new  Abstract: In many settings, studying causal questions based on text data requires adjusting for confounding information within texts. Yet there is a tradeoff in constructing text representations for adjustment: they must be sufficiently large and/or dense to preserve the confounding variables necessary for unbiased effect estimation, but sufficiently small and/or sparse to satisfy finite-sample overlap and yield low-variance estimates. To address this tradeoff, we turn to sparse autoencoders (SAEs), and propose a novel causal adjustment pipeline that iteratively selects a minimal set of SAE features via conditional independence tests. We find that SAE representations achieve better adjustments (lower bias and and higher coverage) than alternative representations in standard semi-synthetic evaluations with binary confounders, and their interpretability offers opportunities for falsification. We also introduce a more realistic semi-synthetic evaluat
    
[^36]: 扩散式视觉语言模型的可靠性挑战

    Reliability Challenges in Diffusion Vision-Language Models

    [https://arxiv.org/abs/2609.01318](https://arxiv.org/abs/2609.01318)

    本文首次系统性评估了扩散式大型视觉语言模型的可靠性，发现尽管其幻觉率与自回归模型相当，但存在“否”偏置、语言质量下降、对代表性不足种族群体的准确率崩溃及反向性别偏置、以及由长度先验导致的多项选择准确率崩溃等严重可靠性问题。

    

    基于扩散的大型视觉语言模型近年来已成为自回归（AR）视觉语言模型的一种引人注目的替代方案，在并行解码、双向上下文和可控生成方面具有优势。尽管进展迅速，但它们的可靠性特性在很大程度上仍未被系统研究。我们对扩散式视觉语言模型中的幻觉和偏差进行了首次系统性可靠性评估，将六个扩散模型与具有竞争力的自回归基线在四个维度上进行基准对比。我们的主要发现包括：（1）在二元视觉查询中，扩散式模型逆转了自回归模型的“是”偏置；（2）它们实现了具有竞争力的幻觉率，但语言质量有所下降；（3）对于代表性不足的种族群体，其准确率崩溃至接近零，并存在极性相反的性别偏置；（4）当正确选项短于干扰项时，它们在多项选择设置中表现出准确率崩溃，这与长度先验相关。

    arXiv:2609.01318v1 Announce Type: cross  Abstract: Diffusion-based Large Vision-Language Models (dLVLMs) have recently emerged as a compelling alternative to autoregressive (AR) LVLMs, offering advantages in parallel decoding, bidirectional context, and controllable generation. Despite rapid progress, their reliability properties remain largely uncharacterized. We present the first systematic reliability evaluation of hallucination and bias in dLVLMs, benchmarking six diffusion models against competitive AR baselines across four dimensions. Our key findings are: (1) dLVLMs reverse the yes-bias of AR models in binary visual queries; (2) they achieve competitive hallucination rates yet exhibit degraded linguistic quality; (3) they collapse to near-zero accuracy on underrepresented racial groups with opposite-polarity gender bias; and (4) they exhibit accuracy collapse in multiple-choice settings when the correct option is shorter than its distractors, associated with a length prior that 
    
[^37]: MIDR：面向多模态文档检索的富化增强索引

    MIDR: Enrichment-Augmented Indexing for Multimodal Document Retrieval

    [https://arxiv.org/abs/2609.01316](https://arxiv.org/abs/2609.01316)

    MIDR是一个无需训练的富化增强索引框架，通过在索引阶段利用多模态大语言模型将文档页面转换为经验证的文本字段，将多模态推理从查询时转移到索引时，在ViDoRe V3上相比BM25相对提升23.0%，性能可与ColQwen2.5媲美。

    

    对视觉丰富文档的检索存在一个表示难题：重要内容往往存在于表格、图表、图形和布局关系中，而普通OCR会将其线性化、破坏或遗漏。ColPali系列视觉检索器通过补丁级多向量索引和后期交互评分来解决这一问题，但这使图像衍生的检索保留在查询时的服务路径上。我们提出MIDR（Multimodal Indexing for Document Retrieval，面向文档检索的多模态索引），这是一个无需训练的富化增强索引框架，将多模态推理转移到索引阶段。在数据摄取过程中，多模态大语言模型将渲染的页面转换为经过验证的文本字段，并使用BM25F进行索引，可选择与稠密检索融合，从而在多模态扎根的证据之上实现以文本为中心的服务。在ViDoRe V3上，MIDR Hybrid在五个英文领域取得0.6219的平均nDCG，相比BM25相对提升23.0%，与ColQwen2.5保持竞争力。

    arXiv:2609.01316v1 Announce Type: cross  Abstract: Retrieval over visually rich documents has a representation problem: important content often lives in tables, charts, figures, and layout relations that plain OCR linearizes, corrupts, or omits. ColPali-family visual retrievers address this with patch-level multi-vector indexes and late-interaction scoring, keeping image-derived retrieval on the query-time serving path. We introduce MIDR (Multimodal Indexing for Document Retrieval), a training-free framework for enrichment-augmented indexing that shifts multimodal reasoning to index time. During ingestion, a multimodal LLM converts rendered pages into verified textual fields that are indexed with BM25F and optionally fused with dense retrieval, enabling text-centric serving over multimodally grounded evidence. On ViDoRe V3, MIDR Hybrid achieves 0.6219 average nDCG across five English domains, a 23.0% relative gain over BM25, remaining competitive with ColQwen2.5. On two French-document
    
[^38]: 先探索后决策：面向深度研究智能体的假设引导搜索方法

    Explore Before Committing: Hypothesis-Guided Search for Deep Research Agents

    [https://arxiv.org/abs/2609.01294](https://arxiv.org/abs/2609.01294)

    针对深度研究智能体过早沿单一路径搜索而锁定错误方向的问题，提出HypoSearch方法：先生成轻量级假设作为软性搜索提示，再通过有限的独立分支并行探索，并在比较分支证据后才做出决策，从而提升复杂问题回答的可靠性。

    

    深度研究智能体通过与搜索和浏览工具交互来回答复杂问题，然而它们通常沿着单一演化的轨迹进行搜索。我们的轨迹层面分析揭示了一种常见的失败模式：智能体可能在早期搜索状态中面临多个合理方向，却在收集到足够的比较证据之前就选择沿某一个方向前进。一旦发生这种情况，后续的工具调用往往会不断强化同一路径，当初始方向具有误导性时会增加失败的概率。我们进一步发现，成功的轨迹通过两种行为降低了这种风险：将模糊的探索锚定到具体的候选对象上，以及当当前路径薄弱或不完整时及时转换方向。基于这些发现，我们提出了HypoSearch，该方法生成轻量级假设作为软性搜索提示，通过有界的独立分支对假设进行探索，并在做出承诺之前比较分支层面的证据。（注：原文摘要此处被截断）

    arXiv:2609.01294v1 Announce Type: new  Abstract: Deep-research agents answer complex questions by interacting with search and browsing tools, yet they often search along a single evolving trajectory. Our trajectory-level analysis reveals a common failure mode in which the agent may encounter an early search state with several plausible directions, but follow one direction before collecting enough comparative evidence. Once this happens, subsequent tool calls tend to reinforce the same path, increasing the chance of failure when the initial direction is misleading. We further find that successful trajectories reduce this risk through two behaviors: grounding vague exploration in concrete candidates and shifting directions when the current path is weak or incomplete. Based on these findings, we propose HypoSearch, which generates lightweight hypotheses as soft search hints, explores them through bounded independent branches, and compares branch-level evidence before commitment. Across fo
    
[^39]: 有些情感藏得更深：大型语言模型中的逐层探测与因果干预

    Some Emotions Run Deeper: Layer-wise Probing and Causal Intervention in Large Language Models

    [https://arxiv.org/abs/2609.01279](https://arxiv.org/abs/2609.01279)

    该研究结合逐层探测与因果干预，在三个情感显式程度不同的语料库和八个大语言模型上发现，情感在模型中的可读取深度随文本来源系统性变化——越隐含、越依赖语境的情感需要越深的层才能读取，说明情感表达深度同时取决于文本来源与模型本身。

    

    情感在文本中的表达跨越一个很宽的光谱，从表层的词汇线索到与内容深度交织的推断。现有针对大语言模型中情感的逐层分析大多只使用单一语料库，因此情感在模型多深的层上变得可读取究竟是模型本身的属性，还是也取决于文本来源，这一问题仍未解决。我们在三个数据集上研究了这个问题，这些数据集涵盖了情感表达的不同明确程度与语境化程度（Twitter 帖子、Reddit 评论以及自传式叙述），涉及来自 Llama、Qwen 和 Granite 家族的八个参数规模为 1B–9B 的开源权重大语言模型。我们将逐层探测与离线特征缩放及在线前向干预相结合，并辅以迁移分析和提前退出分类器。我们发现：(i) 最佳探测层随语料库发生系统性偏移，从靠近输入的层变化到超过模型深度一半的位置，且在按标签与长度区间匹配分布后，这一排序依然成立；(ii) 在被评估的……（原文摘要在此处被截断）

    arXiv:2609.01279v1 Announce Type: cross  Abstract: Emotion is expressed in text along a wide spectrum, from surface lexical cues to inferences entangled with content. Most layer-wise analyses of emotion in LLMs use a single corpus, leaving open whether the depth at which emotion becomes accessible is a property of the model or also of the text source. We investigate this across three datasets spanning different degrees of explicitness and contextualization in emotion expression (Twitter posts, Reddit comments, and autobiographical narratives) and eight 1B--9B open-weight LLMs from the Llama, Qwen, and Granite families. We combine layer-wise probing with offline feature scaling and online forward interventions, transfer analyses, and an early-exit classifier. We find that (i) the best probing layer shifts systematically across corpora, from input-adjacent layers to over half model depth, and this ordering persists after matching label-by-length-bin distributions; (ii) across the evaluat
    
[^40]: 从基座模型采样到强化学习推理：预算约束搜索的视角

    From Base Rollouts to RL Reasoning: A Budgeted Search Perspective

    [https://arxiv.org/abs/2609.01274](https://arxiv.org/abs/2609.01274)

    提出统一解码框架（UDF）将各类解码与搜索方法统一到共享预算空间，发现强化学习的推理增益可由基座模型运行点的结构化预算转换路径近似恢复，表明RL主要是将采样分布转向基座模型本可达但很少采样的轨迹，而非创造全新的推理能力。

    

    带有可验证奖励的强化学习提升了语言模型的推理能力，但这些增益与推理时解码和搜索之间的关系仍不清楚。强化学习是创造了基座模型所不具备的推理能力，还是仅仅将采样分布转向了基座模型本来就能达到但很少采样到的轨迹？我们通过统一解码框架（UDF）从行为层面研究这一问题，该框架将词元级采样、类束搜索、树搜索和序列级重采样表示为在共享预算运行空间上的可执行策略，并事后用 pass@k、自一致性、best-of-N 和 first-finish 成功率进行评分。利用 SimpleRL-Zoo 中成对的 Base/RL 检查点，我们探究强化学习默认策略曲线能否由基座模型运行点的结构化路径来近似。在 Math500、AIME、GPQA 和 IFEval 上，pass@k 的恢复路径遵循预算运行点转换规则（BOPTR），即 $N_{\mathrm{Base}} \approx \alpha N$（摘要在此处截断）。

    arXiv:2609.01274v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) improves language-model reasoning, but how these gains relate to inference-time decoding and search remains unclear. Does RL create reasoning the base model lacks, or shift the rollout distribution toward trajectories it can already reach but rarely samples? We study this behaviorally with a Unified Decoding Framework (UDF), which expresses token-level sampling, beam-like search, tree search, and sequence-level resampling as executable policies over a shared budgeted operating space, scored post hoc with pass@$k$, self-consistency, best-of-$N$, and first-finish success. Using paired Base/RL checkpoints from SimpleRL-Zoo, we ask whether an RL default-policy curve can be approximated by a structured path of Base operating points. On Math500, AIME, GPQA, and IFEval, the pass@$k$ recovery path follows a Budgeted Operating-Point Transition Rule (BOPTR), $N_{\mathrm{Base}} \approx \alpha N_
    
[^41]: 智能体软件工程基准测试究竟在衡量什么？超越类别标签的任务需求与智能体行为画像

    What Does an Agentic Software Engineering Benchmark Measure? Profiling Task Demands and Agent Behaviour Beyond What Category Labels Reveal

    [https://arxiv.org/abs/2609.01271](https://arxiv.org/abs/2609.01271)

    本文提出Spread-Novelty-Centrality（SNC）三轴画像方法来刻画仓库级编码任务的真实需求，发现类别标签是任务需求的不可靠代理指标，且智能体行为轨迹能揭示人工标准答案无法反映的任务需求信息。

    

    智能体软件工程基准测试通常通过名义类别标签（如“缺陷修复”或“功能实现”）来概括，然而携带相同标签的基准测试是通过截然不同的构建流程创建的。因此，标签几乎无法揭示基准测试所要求的工程工作。我们引入了 Spread--Novelty--Centrality（SNC）画像，这是一种基于实证软件工程研究的三轴特征描述方法，用于刻画仓库级编码任务的需求。我们将该画像应用于五个广泛使用的基准测试，以及两个模型家族在三种规模下的14,922条轨迹，并报告了三项发现。（1）类别标签是任务需求的不可靠代理指标，因为每一对基准测试在至少两个SNC轴上均存在统计学上的显著分离，且这些分离可以追溯到具体的构建决策。（2）智能体行为揭示了人类编写的标准答案（gold solution）无法揭示的需求。智能体产出的解决方案……

    arXiv:2609.01271v1 Announce Type: cross  Abstract: Agentic software engineering benchmarks are typically summarized by nominal category labels such as "bug fix" or "feature implementation," yet benchmarks carrying the same label are built through very different curation pipelines. A label thus reveals little about the engineering work a benchmark demands. We introduce the Spread--Novelty--Centrality (SNC) profile, a three-axis characterization of the demands of repository-level coding tasks, grounded in empirical software engineering research. We apply the profile to five widely used benchmarks and 14,922 trajectories of two model families at three scales, and report three findings. (1) A label is an unreliable proxy for task demands, as every pair of benchmarks is statistically separated on at least two SNC axes, and the separations trace back to specific curation decisions. (2) Agent behaviour reveals demands that the human-written gold solution cannot. Agents produce larger solution
    
[^42]: 准备好开口说话：将大语言模型对齐以生成适合语音合成的文本

    Ready to Speak: Aligning LLMs for TTS-Friendly Text Generation

    [https://arxiv.org/abs/2609.01246](https://arxiv.org/abs/2609.01246)

    本文将“让大语言模型直接生成适合语音合成的文本”构建为偏好对齐问题，引入了 CORA 和 Recipe 两个偏好数据集以及结合启发式指标、TTS→ASR 评估流程和人类 MUSHRA 听力研究的评测体系，并比较了基于可解释特征的 FaST 框架与黑盒奖励模型方法。

    

    arXiv:2609.01246v1 公告类型：新论文 摘要：当前的大语言模型（LLMs）主要针对书面文本进行优化，其输出虽然语法正确且内容有帮助，但往往不适合通过文本转语音（TTS）技术进行口头表达。在这项工作中，我们研究如何让大语言模型原生地生成适合语音合成的文本，并将该问题构建为一个偏好对齐问题：我们不依赖下游的改写模块，而是直接对齐大语言模型，使其生成针对口头表达优化的文本。我们引入了两个涵盖不同目标领域的偏好数据集 CORA 和 Recipe，其中包含成对的“适合TTS”与“不适合TTS”的回复。我们进一步提出了一个评估套件，结合了基于模式的启发式指标、TTS→ASR 评估流程，以及由人类评审参与的 MUSHRA 听力研究。我们的实验比较了最近提出的特征感知采样与调优（FaST）框架——该框架利用可解释的特征而非黑盒奖励模型——

    arXiv:2609.01246v1 Announce Type: new  Abstract: Current Large Language Models (LLMs) are primarily optimized for written text, often producing outputs that are grammatically correct and helpful yet poorly suited for spoken delivery via Text-to-Speech (TTS). In this work, we study how to make LLMs natively generate TTS-friendly text, which we frame as a preference alignment problem: instead of relying on downstream rewriting modules, we directly align LLMs to generate text optimized for spoken delivery. We introduce two preference datasets spanning different target domains, CORA and Recipe, which contain paired TTS-friendly and TTS-unfriendly responses. We further propose an evaluation suite combining a pattern-based heuristic metric, a TTS$\to$ASR evaluation pipeline, and a MUSHRA listening study with human judges. Our experiments compare the recently proposed Feature-aware Sampling and Tuning (FaST) framework -- leveraging interpretable features instead of a black-box reward model --
    
[^43]: 面向监督微调的后训练科学

    Post-Training Science for Supervised Fine-Tuning

    [https://arxiv.org/abs/2609.01244](https://arxiv.org/abs/2609.01244)

    本文通过每次只改变一个变量的统一受控扫描实验，系统测量了监督微调中学习率、批大小、LoRA与全量微调等关键决策在Qwen3与Llama两类模型（稠密与混合专家架构）以及四个真实客户数据集上的表现，将SFT超参数选择从经验摸索转变为可复现的科学测量。

    

    每一次监督微调（SFT）运行都迫使我们做出同样的一系列决策，例如学习率、批大小、采用LoRA还是全量微调、训练多少个epoch、选择哪种优化器，以及向模型输入什么数据。这些决策通常在每次面对新模型和新数据集时都要从头重新摸索。本文在一个统一的测量工具下对它们进行测量：一种每次只改变一个控制变量的扫描方法，涵盖Qwen3和Llama两个模型家族中的稠密模型与混合专家模型，在四个真实世界的客户SFT数据集上，分别对LoRA和全量微调进行测试。这些数据集提供了一个受控的实验平台：每个任务都带有一个与客户共同构建的评估标准，其训练数据通过迭代式监督微调生成，即不断改进模型输出直到其通过该评估，因此监督目标在内部是一致的，而我们报告所依据的任务评判标准正是数据构建时所旨在满足的准则。我们探究最优学习率和批大小如何随……（原文摘要在此处截断）

    arXiv:2609.01244v1 Announce Type: cross  Abstract: Every supervised fine-tuning run forces the same chain of decisions, such as learning rate, batch size, LoRA or full fine-tuning, how many epochs, which optimiser, and what data to feed the model. Each of these is typically rediscovered from scratch for every new model and dataset. Here we measure them under one instrument: a sweep that varies one lever at a time, and spans dense and mixture-of-experts models in two families (Qwen3 and Llama), on four real-world customer SFT datasets, for both LoRA and full fine-tuning. These datasets give a controlled testbed: each task carries an evaluation built with the customer, and its training data is produced by iterative supervised fine-tuning that refines model outputs until they pass that evaluation, so the supervised target is internally consistent and the task judge we report against is the criterion the data was built to satisfy. We ask how the optimal learning rate and batch size move wi
    
[^44]: 迈向AI辅助的临床试验匹配：实践考量、多中心评估与真实世界部署

    Towards AI-Assisted Clinical Trial Matching: Practical Considerations, Multicenter Evaluation, and Real-World Deployment

    [https://arxiv.org/abs/2609.01202](https://arxiv.org/abs/2609.01202)

    本文提出面向真实世界部署的AI临床试验推荐系统TrialGPT 2.0，它不仅评估患者资格，还结合患者临床需求和本地工作流优先级筛选值得进一步考虑的试验，并提供了结构化的可审查解释，在政府、学术癌症中心等多种肿瘤学场景中完成了回顾性与前瞻性多中心评估。

    

    临床试验对于推进癌症治疗和药物研发至关重要，但许多试验因患者入组不足而失败。尽管人们利用AI支持患者招募的兴趣日益浓厚，现有系统大多仅执行资格评估，且很少在真实世界的肿瘤学工作流程中得到评估。在此，我们提出TrialGPT 2.0，一个为真实世界部署而设计的AI辅助临床试验推荐系统。该系统不仅评估患者是否符合入组条件，还会根据患者当前的临床需求和本地工作流程优先级，评估哪些试验值得进一步考虑，并提供结构化、可审查的解释供专家审核。重要的是，我们在多个专注于肿瘤学的环境中对TrialGPT 2.0进行了回顾性和前瞻性评估，涵盖政府、学术癌症中心、患者倡导组织和NIH转诊等多种工作流程。

    arXiv:2609.01202v1 Announce Type: cross  Abstract: Clinical trials are essential for advancing cancer care and drug development, but many fail because of insufficient patient enrollment. While there is growing interest in using AI to support patient recruitment, existing systems largely perform eligibility assessment alone and have rarely been evaluated in real-world oncology workflows. Here we present TrialGPT 2.0, an AI-assisted clinical trial recommendation system designed for real-world deployment. Rather than asking only whether a patient may qualify, the system also assesses which trials warrant further consideration given the patient's current clinical needs and local workflow priorities, and provides structured, inspectable explanations for expert review. Importantly, we evaluated TrialGPT 2.0 retrospectively and prospectively across multiple oncology-focused settings, spanning government, academic cancer-center, patient-advocacy, and NIH referral workflows. In retrospective mu
    
[^45]: FinLifeBench：从纵向银行对话中穷尽式重建人生事件历史与财务状态

    FinLifeBench: Exhaustive Life-Event History and Financial-State Reconstruction from Longitudinal Banking Dialogue

    [https://arxiv.org/abs/2609.01198](https://arxiv.org/abs/2609.01198)

    提出FinLifeBench基准，基于6,000个韩语银行对话会话，评估大语言模型在穷尽式重建客户人生事件历史与34维财务状态方面的长程记忆能力，发现随会话累积事件召回率显著下降（0.591降至0.445），且错误主要源于事件遗漏。

    

    重复的银行交互要求助手在生活变化随日常请求偶然出现时，维护完整、最新且可追溯的客户记录。现有基准强调问答、有界回合或定向回忆，而非穷尽式的纵向重建。我们提出FinLifeBench，它在同一累积对话上评估两项任务：重建每个人生事件实例及其首次确立的会话，以及在连续检查点上重建完整的34条路径财务状态。该基准包含来自20条独立合成轨迹的6,000个八轮韩语银行会话，为24种事件类型和34条状态路径提供确定性的、穷尽式的黄金标准及共识质量保证。在全上下文条件下对十一个大语言模型的评估中，事件锚点召回率从15个会话时的0.591下降至300个会话时的0.445。错误主要由遗漏事件导致，而非（摘要原文在此处截断）

    arXiv:2609.01198v1 Announce Type: cross  Abstract: Repeated banking interactions require assistants to maintain complete, current, and traceable customer records as life changes emerge incidentally in routine requests. Existing benchmarks emphasize question answering, bounded episodes, or targeted recall rather than exhaustive longitudinal reconstruction. We introduce FinLifeBench, which evaluates two tasks over the same cumulative dialogue: reconstructing every life-event instance with its first-establishing session and reconstructing a complete 34-path financial state at consecutive checkpoints. The benchmark contains 6,000 eight-turn Korean banking sessions from 20 independent synthetic trajectories, with deterministic, exhaustive gold for 24 event types and 34 state paths and consensus quality assurance. Across eleven LLMs under a full-context condition, event-anchor recall falls from 0.591 at 15 sessions to 0.445 at 300. Errors are driven primarily by omitted events rather than po
    
[^46]: CaRL-EM：面向大语言模型实体匹配的成本感知强化学习

    CaRL-EM: Cost-Aware Reinforcement Learning for Entity Matching with LLMs

    [https://arxiv.org/abs/2609.01195](https://arxiv.org/abs/2609.01195)

    该论文提出CaRL-EM，一个成本感知的强化学习控制器，通过在多候选实体匹配中自适应地选择LLM操作符与模型容量，来优化质量与成本的权衡目标。

    

    实体匹配（EM）需要对细粒度上下文的理解和领域知识。近期研究表明，大语言模型（LLM）可以成为跨领域的强大匹配器，但大多数方法要么做出独立的成对决策，要么依赖人工设计的复合流程，因此在现实的多候选场景中缺乏灵活性。同时，这些方法通常忽略了大规模推理时的成本。我们将基于LLM的多候选实体匹配形式化为一个成本感知的序贯决策问题，并提出CaRL-EM——一个用于管理LLM操作的强化学习控制器。给定锚点记录的状态、其候选集以及成本，CaRL-EM能够在不同的操作符（匹配/比较/选择/决策）和不同模型容量之间自适应地选择，以最大化质量-成本目标。该策略与抽象操作符交互，使得同一控制器可以在推理时复用于不同的底层LLM后端。

    arXiv:2609.01195v1 Announce Type: new  Abstract: Entity matching (EM) requires fine-grained contextual understanding and domain knowledge. Recent work shows that large language models (LLMs) can serve as strong matchers across domains, but most methods either make independent pairwise decisions or rely on manually designed composite pipelines, thus lacking flexibility in realistic multi-candidate settings. At the same time, they typically ignore inference cost at scale. We formulate LLM-based EM with candidates as a cost-aware sequential decision problem and propose CaRL-EM, a reinforcement learning controller that manages LLM operations. Given the state of an anchor record, its candidate set, and the cost, CaRL-EM adaptively chooses among different operators (Match/Compare/Select/Decide) and model capacities to maximize a quality-cost objective. The policy interacts with abstract operators, allowing the same controller to be reused with different underlying LLM backends at inference t
    
[^47]: PersuaRL：基于强化学习的多专家选择方法，用于保险领域的说服性对话生成

    PersuaRL: Reinforcement Learning-Driven Multi-Expert Selection for Persuasive Dialogue Generation in Insurance

    [https://arxiv.org/abs/2609.01188](https://arxiv.org/abs/2609.01188)

    该论文提出了保险领域说服性对话数据集InsureDial，并构建了基于强化学习的多专家选择框架PersuaRL，以提升大语言模型驱动的对话智能体在保险场景中生成有说服力对话的能力。

    

    arXiv:2609.01188v1 公告类型：新论文 摘要：大语言模型（LLMs）正在通过为部署在客户服务、数字销售和保险等领域的对话智能体提供动力，彻底改变数字通信方式。这些基于大语言模型构建的智能体能够理解用户输入、检索相关信息并生成连贯的回复。然而，尽管它们在事实性交流方面表现出色，却往往缺乏开展真正具有说服力、贴合语境的对话的能力，尤其是在保险这类信任与清晰度至关重要的领域中尤为明显。立足于保险领域的这一需求，我们的工作致力于提升数字智能体（即大语言模型）的说服力。为此，我们提出了InsureDial——一个说服性保险对话数据集，旨在捕捉汽车保险交互中特有的说服性沟通的细微差别。我们还提出了PersuaRL，一个基于强化学习的框架，使大语言模型驱动的对话智能体具备（原文摘要在此处截断）……

    arXiv:2609.01188v1 Announce Type: new  Abstract: Large Language Models (LLMs) are revolutionizing digital communication by powering conversational agents deployed across domains such as customer service, digital sales, and insurance. These agents, built on LLMs, can understand user input, retrieve relevant information, and generate coherent responses. However, while they excel at factual communication, they often lack the ability to engage in truly persuasive, context-sensitive dialogue, especially in domains like insurance, where trust and clarity are critical. Building on this need within the insurance domain, our work focuses on improving the persuasiveness of digital agents, aka LLMs. To support this, we introduce InsureDial, a Persuasive Insurance Dialogue dataset, designed to capture the nuances of persuasive communication specific to motor insurance interactions. We introduce PersuaRL, a reinforcement learning-based framework that equips LLM-driven dialogue agents with the abili
    
[^48]: LLMPEDIA：浏览、验证和比较大语言模型的参数化百科知识

    LLMPEDIA: Browsing, Verifying, and Comparing the Parametric Encyclopedic Knowledge of LLMs

    [https://arxiv.org/abs/2609.01182](https://arxiv.org/abs/2609.01182)

    提出LLMPEDIA系统，从三个大语言模型的参数记忆中递归生成约130万篇百科文章并对照维基百科和网络资源验证，发现模型参数知识的真实率仅为68.4%（比MMLU低超过21个百分点），其中30.5%的断言无法被任何现有证据裁定，从而揭示了固定基准测试的可得性偏差。

    

    旗舰级语言模型在MMLU等基准测试上似乎已经饱和，得分超过90%——然而基准测试只能检验实验者想到要提问的内容，这是固定问题集带来的可得性偏差。LLMPEDIA使这种偏差变得可测量、可浏览。研究团队从三个模型家族（GPT-5-mini、DeepSeek-V3.2、Llama-3.3-70B）的参数记忆中递归地物化了约130万篇百科文章，无需任何检索，然后对原子断言的分层样本对照维基百科和精选的网络资源进行审计，将每条断言标记为被支持、被驳斥或证据不足。在均匀随机样本上，真实率为68.4%——比MMLU低超过21个百分点——其中30.5%的断言证据不足：这些断言没有任何基准测试会去探查，世界上最大的百科全书也无法裁定——究竟是长尾知识还是貌似合理的幻觉，现有证据无法区分——这将GPTKB针对三元组所建立的覆盖缺口扩展到了自由文本领域。

    arXiv:2609.01182v1 Announce Type: new  Abstract: Flagship language models appear saturated on benchmarks like MMLU (Hendrycks et al., 2021), scoring above 90% - yet benchmarks test only what the experimenter thought to ask, the availability bias of fixed question sets. LLMPEDIA makes this bias measurable and browsable. We recursively materialized ~1.3M articles from three model families' parametric memory (GPT-5-mini, DeepSeek-V3.2, Llama-3.3-70B) without retrieval, then audited a stratified sample of atomic claims against Wikipedia and a curated web stack, coloring every claim supported, refuted, or insufficient (Saeed and Razniewski, 2026). On a uniform random sample the true rate is 68.4% - more than 21 pp below MMLU - with 30.5% of claims insufficient: assertions no benchmark probes and the world's largest encyclopedia cannot adjudicate - long-tail knowledge or plausible hallucination, the evidence cannot tell - extending to free text the coverage gap GPTKB established for triples 
    
[^49]: 子词分段BabyLM：学习分词以实现样本高效的预训练

    Subword Segmental BabyLMs: Learning to Tokenise for Sample-Efficient Pretraining

    [https://arxiv.org/abs/2609.01151](https://arxiv.org/abs/2609.01151)

    该论文提出了两个在预训练过程中联合学习分词的子词分段语言模型SubSegGPT和SubSegDeBERTa，并在2026年BabyLM挑战赛中实现了样本高效的性能提升。

    

    在标准的语言模型训练流程中，子词分词通常作为预处理步骤应用。子词分段语言建模是一种替代范式，其中分词在训练过程中被学习，使模型能够发现能优化其训练目标的子词单元。在本文中，我们展示了参加2026年BabyLM挑战赛的成果，为此我们开发了两个新的子词分段语言模型：SubSegGPT和SubSegDeBERTa。SubSegGPT是一个仅解码器模型，它在自回归预训练期间学习分词。SubSegDeBERTa是一个基于编码器的模型，它联合学习生成和分词被掩码的词。我们对Strict和Strict-small两个赛道都进行了训练。我们提交给Strict赛道的最佳模型是SubSegDeBERTa，它在零样本评估中取得了显著提升。我们提交给Strict-small赛道的最佳模型是SubSegGPT，它优于基于分词的基线模型。我们的结果表明，可学习的子词分词可以改善……

    arXiv:2609.01151v1 Announce Type: new  Abstract: In the standard LM training pipeline, subword tokenisation is applied as a preprocessing step. Subword segmental language modelling is an alternative paradigm in which tokenisation is learned during training, allowing the model to discover subword units that optimise its training objective. In this paper, we present our submission to the 2026 BabyLM Challenge, for which we develop two new subword segmental LMs: SubSegGPT and SubSegDeBERTa. SubSegGPT is a decoder-only model that learns tokenisation during autoregressive pretraining. SubSegDeBERTa is an encoder-based model that jointly learns to generate and tokenise masked words. We train both for the Strict and Strict-small tracks. Our top submission to Strict is SubSegDeBERTa, which achieves notable gains in zero-shot evaluation. Our top submission to Strict-small is SubSegGPT, which outperforms tokenisation-based baselines. Our results show that learnable subword tokenisation can impro
    
[^50]: 像素文本表示学习的设计基础研究

    On the Design Fundamentals of Pixel Text Representation Learning

    [https://arxiv.org/abs/2609.01147](https://arxiv.org/abs/2609.01147)

    本文通过系统性消融实验提出了鲁棒像素文本表示学习的四大设计原则（可变图像分辨率与字体大小、自然图像-文本对、布局感知渲染、两阶段多语言课程），并据此训练出原生分辨率的视觉文本模型 Pixel Linguist II。

    

    富含文本的视觉输入要求模型能够直接在像素空间中读取、检索和压缩语言，然而现有的像素-文本编码器在固定分辨率预训练、视觉捷径学习、视觉接地能力薄弱以及多语言视觉文本理解等方面存在不足。在本工作中，我们研究了构建鲁棒的视觉文本表示学习所需的基本设计原则。通过系统性的受控消融实验，我们识别出四个关键组件：可变的图像分辨率与渲染字体大小为高分辨率文档泛化提供了空间代理；自然图像-文本对对于视觉接地不可或缺，并能防止纯文本坍缩；布局感知的渲染有助于防止像素级捷径学习；两阶段多语言课程学习则实现了有效的跨语言对齐。通过将这些原则整合到可扩展的训练方案中，我们训练出了 Pixel Linguist II——一个原生分辨率的视觉文本模型。

    arXiv:2609.01147v1 Announce Type: cross  Abstract: Text-rich visual inputs require models that can read, retrieve, and compress language directly in pixel space, yet existing pixel-text encoders struggle with fixed resolution pretraining, visual shortcut learning, weak visual grounding, and multilingual visual text understanding. In this work, we investigate the fundamental design principles required for robust visual text representation learning. Through systematic controlled ablations, we identify four critical components: variable image resolutions and rendered font sizes provide spatial proxies for high-resolution document generalization; natural image-text pairs are indispensable for grounding and prevent text-only collapse; layout-aware rendering helps prevent pixel-level shortcuts; and a two-stage multilingual curriculum enables effective cross-lingual alignment. By integrating these principles into a scalable training recipe, we train Pixel Linguist II, a native-resolution visi
    
[^51]: 任务分解能否改进自动自然语言生成（NLG）评估？

    Does task decomposition improve automatic NLG evaluation?

    [https://arxiv.org/abs/2609.01139](https://arxiv.org/abs/2609.01139)

    本研究通过系统性实验发现，任务分解并不能真正提升LLM-as-a-judge的NLG评估性能，先前报告的性能提升实际源于使用人工标注作为训练数据，且在有人工标注时，无需分解的LLMaJ即可达到与人工标注者相当的水平。

    

    LLM-as-a-judge（LLMaJ）框架已成为一种有前景的解决方案，可实现低成本、可复现且无需参考答案的自然语言生成（NLG）评估。先前的工作试图通过将评估任务分解为更简单的子任务来改进LLMaJ。在本工作中，我们在多个NLG数据集上系统性地比较了使用与不使用任务分解的LLMaJ方法。我们发现，没有证据表明使用任务分解的LLMaJ相较于不使用分解的公平基线能带来性能提升。相反，我们发现先前所报道的基于分解的LLMaJ的性能提升源于将人工标注用作训练数据，而非任务分解本身。此外，我们还发现，当人工标注可用时，不使用任务分解的LLMaJ可以达到与人工标注者相当的性能。

    arXiv:2609.01139v1 Announce Type: new  Abstract: The LLM-as-a-judge (LLMaJ) framework has emerged as a promising solution for cheap, reproducible, reference-free Natural Language Generation (NLG) evaluation. Prior work seeks to improve LLMaJ by decomposing evaluation tasks into simpler sub-tasks. In this work, we systematically compare LLMaJ methods with and without decomposition on multiple NLG datasets. We find no evidence that LLMaJ with task decomposition leads to performance gains over a fair baseline that does not use decomposition. Instead, we find that previously reported performance gains in decomposition-based LLMaJ stem from using human labels as training data, and not task decomposition itself. Also, we find that, when human labels are available, LLMaJ without using task decomposition can perform comparably to human annotators.
    
[^52]: 通过奇异值分解缓解最小贝叶斯风险解码中的过拟合

    Overfitting Mitigation via Singular Value Decomposition in Minimum Bayes Risk Decoding

    [https://arxiv.org/abs/2609.01135](https://arxiv.org/abs/2609.01135)

    本文提出SVD-MBR方法，通过奇异值分解对最小贝叶斯风险解码中的成对效用矩阵进行低秩近似去噪，有效缓解度量过拟合并显著提升泛化性能。

    

    最小贝叶斯风险（MBR）解码通过在采样的伪参考文本上选择使效用度量最大化的假设，从而实现高质量的文本生成。然而，它极易受到度量过拟合的影响：它可能不规则地抬高所选的效用度量，而直接损害其他未被优化的评估指标。为了缓解这一问题，我们提出了SVD-MBR，该方法将成对效用矩阵视为含有噪声的信息信号。通过奇异值分解（SVD）计算低秩近似并仅保留前k个分量，我们有效地将真实共识与度量噪声解耦。实验表明，SVD-MBR成功地对解码过程进行了正则化，在一系列泛化指标上取得了显著提升。此外，我们揭示了这种去噪效果依赖于度量类型：神经度量编码了对SVD理想的鲁棒低秩共识，而表层度量则难以分离信号（与噪声）。

    arXiv:2609.01135v1 Announce Type: new  Abstract: Minimum Bayes Risk (MBR) decoding enables high-quality text generation by selecting the hypothesis that maximizes a utility metric over sampled pseudo-references. However, it is highly susceptible to metric overfitting: it can irregularly inflate the chosen utility metric at the direct expense of other unoptimized evaluation metrics. To mitigate this, we introduce SVD-MBR, which frames the pairwise utility matrix as a noisy information signal. By computing a low-rank approximation via Singular Value Decomposition (SVD) and retaining only the top-$k$ components, we effectively decouple true consensus from metric noise. Experiments demonstrate that SVD-MBR successfully regularizes decoding, yielding substantial gains across a range of generalized metrics. Furthermore, we reveal that this denoising is metric-dependent: neural metrics encode a robust low-rank consensus ideal for SVD, whereas surface-level metrics struggle to separate signal 
    
[^53]: 潜在循环思维：基于冻结大语言模型推理的潜在向量循环精炼方法

    Latent Recurrent Thoughts: Recurrent Refinement of Proposed Latents for Reasoning with Frozen LLMs

    [https://arxiv.org/abs/2609.01117](https://arxiv.org/abs/2609.01117)

    该论文提出潜在循环思维（LRT）方法，通过保持大语言模型冻结并引入一个微型循环推理器在连续潜在空间中多步迭代精炼潜在思维向量来进行推理，将计算深度与模型规模解耦，从而规避了思维链推理中误差传播以及需要可模仿轨迹的固有局限。

    

    思维链推理在离散的词元空间中展开：每一步都被固化为文本，误差会不断传播，且要引出高质量的推理轨迹，前提是已有可供模仿的轨迹。而改在模型的连续表示空间中进行推理——其中间状态是向量而非词语——可以规避这些限制，但这些潜在状态应当如何计算仍是一个悬而未决的问题。我们从两个维度着手解决这一问题。首先，我们保持大语言模型（LLM）冻结不变，仅利用其已经擅长的工作——建模和解码序列——同时由一个小型辅助网络提供连续的潜在思维作为输入。其次，我们通过循环递归的方式生成这些潜在向量：一个微型循环推理器在多步过程中对其进行精炼，将计算深度与模型规模解耦，使潜在向量成为迭代处理的产物而非单次前向传播的结果。我们将这一方法实例化为潜在循环思维（Latent Recurrent Thoughts，LRT）：一个面向任务的……

    arXiv:2609.01117v1 Announce Type: new  Abstract: Chain-of-thought reasoning unfolds in discrete token space: each step is committed as text, errors propagate, and eliciting good traces presupposes traces to imitate. Reasoning instead in a model's continuous representation space - where intermediate states are vectors rather than words - sidesteps these constraints, but leaves open how those latent states should be computed. We approach this along two axes. First, we keep a large language model (LLM) frozen and use it for what it is already good at - modeling and decoding sequences - while a small auxiliary network supplies continuous latent thoughts as input. Second, we produce those latents by recurrence: a tiny recurrent reasoner refines them over many steps, decoupling the depth of computation from the size of the model, so that the latents are a product of iterative processing rather than a single forward pass. We instantiate this as Latent Recurrent Thoughts (LRT): a task-dedicate
    
[^54]: EDRAC：阿拉伯语方言阅读理解基准测试

    EDRAC: Benchmarking Arabic Dialect Reading Comprehension

    [https://arxiv.org/abs/2609.01113](https://arxiv.org/abs/2609.01113)

    EDRAC是首个面向阿拉伯语方言机器阅读理解与生成式问答的大规模基准，涵盖埃及、摩洛哥、阿联酋、叙利亚和沙特五种主要方言，包含499篇自然口语段落和通过人-大语言模型协作流水线生成的4,977个问答对，并以此评测了阿拉伯语和多语言大语言模型的表现。

    

    与现代标准阿拉伯语（MSA）相比，方言阿拉伯语（DA）的资源仍然匮乏，尤其是在机器阅读理解（MRC）和问答（QA）任务方面。现有的阿拉伯语问答基准主要聚焦于正式书面的现代标准阿拉伯语或多选题式问答，对自然口语方言的覆盖十分有限。本文旨在弥合这一差距。我们提出了EDRAC，这是首个面向方言阿拉伯语机器阅读理解（MRC）与生成式问答的大规模基准，涵盖五大主要方言：埃及、摩洛哥、阿联酋、叙利亚和沙特阿拉伯方言。EDRAC包含499篇源自自然发生的口语交互的段落，以及4,977个通过人-大语言模型协作流水线生成的对应问答对，该流水线结合了迭代生成、以LLM作为评判者的评估以及人工验证。我们使用词汇和语义指标在EDRAC上对以阿拉伯语为中心和多语言大语言模型进行了基准评测。我们的结果揭示了（各模型之间的）显著差距……

    arXiv:2609.01113v1 Announce Type: cross  Abstract: Dialectal Arabic (DA) remains under-resourced compared to Modern Standard Arabic (MSA), particularly for machine reading comprehension (MRC) and question answering (QA). Existing Arabic QA benchmarks primarily focus on formal written MSA or multiple-choice QA, with limited coverage of naturally spoken dialects. Here, we aim to bridge this gap. We introduce EDRAC, the first large-scale benchmark for dialectal Arabic machine reading comprehension (MRC) and generative QA, covering five major dialects: Egyptian, Moroccan, Emirati, Syrian, and Saudi Arabic. EDRAC contains 499 passages derived from naturally occurring spoken interactions and 4,977 corresponding QA pairs generated through a human--LLM collaborative pipeline combining iterative generation, LLM-as-a-judge evaluation, and human verification. We benchmark Arabic-centric and multilingual LLMs on EDRAC using lexical and semantic metrics. Our results reveal substantial gaps between 
    
[^55]: ClinTraceBench：基于电子健康记录对话的来源可验证的纵向临床推理

    ClinTraceBench: Source-Verifiable Longitudinal Clinical Reasoning over EHR-Derived Dialogues

    [https://arxiv.org/abs/2609.01111](https://arxiv.org/abs/2609.01111)

    提出了ClinTraceBench——一个基于电子健康记录对话、具备事件级来源可验证性的纵向临床推理基准，通过385个已验证对话、九任务分类体系和约20万条预测，系统评估了八种患者历史表示策略在四个大模型上保留纵向临床信号的能力。

    

    arXiv:2609.01111v1 公告类型：新论文。摘要：临床大语言模型助手必须对多就诊的患者轨迹进行推理，然而用于扩展这些模型的紧凑历史表示——检索、结构化时间线、LLM摘要、智能体记忆——是否保留了临床推理所需的纵向信号，尚未得到测量。我们提出ClinTraceBench：包含385个源自MIMIC-IV的已验证对话，具备事件ID溯源、九个任务的分类体系（T1–T9），以及L0–L4确定性验证+L5人工审核验证（98.92%一致性）。我们在6,271个问题上评估了八种历史表示策略——无上下文基线、仅最后一次就诊、全上下文、BGE-M3稠密检索、两种压缩方案以及两种智能体记忆系统——跨越四个骨干模型，共计32个实验单元格、200,672个预测结果。四个发现：（SP4）一个受控的T3注入探针分离出压缩引起的……（摘要原文在此处被截断）

    arXiv:2609.01111v1 Announce Type: new  Abstract: Clinical LLM assistants must reason over multi-visit patient trajectories, yet whether the compact history representations used to scale them---retrieval, structured timelines, LLM summaries, agentic memory---preserve the longitudinal signal clinical reasoning needs has not been measured. We introduce ClinTraceBench: 385 MIMIC-IV-derived verified dialogues with event-ID provenance, a nine-task taxonomy (T1--T9), and L0--L4 deterministic + L5 human-audit validation (98.92\% agreement). We evaluate eight history representation strategies---a no-context floor, \textit{last-visit-only}, \textit{full-context}, BGE-M3 \textit{dense-retrieval}, two compression schemes, and two agentic-memory systems (\textit{Mem0}, \textit{A-Mem})---across four backbones (DeepSeek-V3, GPT-4o-mini, Haiku~4.5, Sonnet~4.6) on 6{,}271 questions: 32 cells, 200{,}672 predictions. Four findings: (SP4) a controlled T3 injection probe isolates compression-induced \texti
    
[^56]: 提示有帮助，但它们真的能“教会”模型吗？评估代码生成中的技能迁移

    Hints Help But Do They Teach? Evaluating Skills Transfer in Code Generation

    [https://arxiv.org/abs/2609.01106](https://arxiv.org/abs/2609.01106)

    研究发现，提示对失败代码生成的“挽救”效果大多可通过无提示的重复采样复现，且相关与无关提示共享同一激活方向，表明提示更多是引导模型已有能力而非传授新技能。

    

    当一条提示能把一个失败的生成程序变成通过测试的程序时，它究竟提供了缺失的信息，还是仅仅将模型引导至它本来就能得出的解？我们在 HumanEval+ 和 MBPP+ 上通过可执行评估来检验这些假设。对于 Qwen2.5-3B-Instruct，自适应的相关提示挽救了 79 个选定失败样例中的 36 个；无关提示挽救了 19 个；而在无提示条件下，8 次采样解决了 46 个样例，并覆盖了 36 个相关提示挽救中的 31 个。Phi-3.5-mini 呈现出相同的模式：相关提示挽救了 101 个失败中的 42 个，无关提示挽救了 17 个，无提示采样解决了 57 个，其中包括 42 个相关提示挽救中的 36 个。由于各提示条件使用了不同的尝试预算，这些比较并不能分离出纯粹的语义效应。在 Qwen 上进行的机制性测试发现，相关提示与无关提示共享一个稳定的激活方向。持续向该方向添加偏移会产生 14 次挽救和 18 次回退，且未检测到……（原文摘要在此处被截断）

    arXiv:2609.01106v1 Announce Type: cross  Abstract: When a hint turns a failing generated program into a passing one, does it provide missing information or merely steer the model toward a solution it could already produce? We test these hypotheses on HumanEval+ and MBPP+ using executable evaluation. For Qwen2.5-3B-Instruct, adaptive relevant hints rescue 36 of 79 selected failures; an unrelated hint rescues 19, while eight unhinted samples solve 46 and recover 31 of the 36 relevant-hint rescues. Phi-3.5-mini shows the same pattern: relevant hints rescue 42 of 101 failures, an unrelated hint rescues 17, and unhinted sampling solves 57, including 36 of the 42 relevant-hint rescues. Because the hint conditions use different attempt budgets, these comparisons do not isolate a purely semantic effect. Mechanistic tests on Qwen identify a stable activation direction shared by relevant and unrelated hints. Persistently adding this direction yields 14 rescues and 18 regressions, with no detecta
    
[^57]: 当模态差距缩小失效时：CLIP中的预测级枢纽性

    When Modality Gap Reduction Fails: Prediction-Level Hubness in CLIP

    [https://arxiv.org/abs/2609.01103](https://arxiv.org/abs/2609.01103)

    本文揭示了在CLIP中缩小模态差距虽然能减小平均图像-文本距离，但可能改变类别间的相对决策结构，导致预测过度集中于少数类别（即预测级枢纽性），反而损害零样本分类准确率。

    

    在CLIP中缩小图像与文本表示之间的模态差距被普遍认为可以改善跨模态对齐和下游任务性能。然而，更小的平均图像-文本差距并不一定带来一致的准确率提升。我们从零样本分类的决策结构角度分析了这种不一致性，即零样本分类是为输入图像选择最相似的类别文本原型。零样本准确率不仅取决于平均的图像-文本对齐程度，还取决于各类别的决策边际。以线性修正作为可解析处理的案例，我们证明模态差距修正会改变类别之间的相对决策结构，导致预测过度集中于少数类别子集。我们将这种输出空间的失效模式称为预测级枢纽性。此外，在多个数据集上的实验表明，差距修正导致的准确率下降始终与这一现象相关联。

    arXiv:2609.01103v1 Announce Type: new  Abstract: Reducing the modality gap between image and text representations in CLIP is widely expected to improve cross-modal alignment and downstream performance. However, a smaller average image-text gap does not necessarily lead to consistent accuracy gains. We analyze this mismatch from the perspective of the decision structure in zero-shot classification, i.e. selecting the most similar class-text prototype for an input image. Zero-shot accuracy depends not only on average image--text alignment, but also on class-wise decision margins. Using Linear correction as an analytically tractable case, we show that modality gap correction can alter the relative decision structure among classes and cause predictions to concentrate on a small subset of classes. We refer to this output-space failure mode as prediction-level hubness. Furthermore, experiments across multiple datasets show that accuracy degradation under gap correction is consistently associ
    
[^58]: 超越幅值：面向模块化混合专家的对比路由

    Beyond Magnitude: Contrastive Routing for Modular Mixture-of-Experts

    [https://arxiv.org/abs/2609.01100](https://arxiv.org/abs/2609.01100)

    提出 CoRM 对比路由机制，通过将每个标记与层隐藏状态的指数移动平均进行对比而非基于绝对幅值进行路由，使路由信号集中于低维可分子空间，从而提升 MoE 专家与语言结构的对齐程度，并将零样本准确率提高最高 1.77 个百分点。

    

    在当前的混合专家架构中，路由是基于被所有标记（token）所共享的结构主导的表示来执行的，这限制了专家的专门化。我们证明了，将每个标记与该层隐藏状态的指数移动平均进行对比，而不是基于绝对幅值进行路由，能够将路由信号集中于一个低维、高度可分的子空间。基于此，我们提出了对比路由机制（CoRM），它通过每个专家对输入标记的亲和度与其对这个共享参考状态的亲和度之间的差距来为专家打分，并通过每个专家独有的投影进行解读。由此得到的专家，其路由边界与语言结构的对齐程度显著高于 Top-k 基线。我们的实验表明，CoRM 相比标准的 Top-k MoE 基线，将平均零样本准确率提升了 +0.67 至 +1.69 个百分点（Top-1）以及 +1.38 至 +1.77 个百分点（Top-2）。

    arXiv:2609.01100v1 Announce Type: new  Abstract: In current Mixture-of-Experts architectures, routing is performed based on representations dominated by structure shared across all tokens, limiting expert specialization. We show that contrasting each token against an Exponential Moving Average of the layer's hidden states, rather than routing on absolute magnitude, concentrates the routing signal onto a low-dimensional, highly separable subspace. Building on this, we propose the Contrastive Routing Mechanism (CoRM), which scores each expert by the gap between its affinity for the incoming token and its affinity for this shared reference state, interpreted through a distinct per-expert projection. The resulting experts have routing boundaries that align with linguistic structure significantly more than the Top-k baseline. Our experiments show that CoRM improves average zero-shot accuracy by +0.67 to +1.69 points (Top-1) and +1.38 to +1.77 points (Top-2) over standard Top-k MoE baselines
    
[^59]: StateSwap：探究多选题中“支持型”与“排除型”框架下的隐藏状态

    StateSwap: Probing Support-Elimination Hidden States in Multiple-Choice Questions

    [https://arxiv.org/abs/2609.01081](https://arxiv.org/abs/2609.01081)

    该论文提出StateSwap方法，通过添加特殊标记[STATE]来探测并交换多选题在“支持型”与“排除型”两种表述下诱导出的隐藏状态激活，证明两种框架在模型中间层产生可分离的内部表示，且交换这些激活可因果性地改变预测结果并提高跨框架答案一致性。

    

    当同一道多项选择题以“支持型”（寻找依据）和“排除型”（逐一排除）两种不同表述方式提出时，大型语言模型往往给出不一致的答案。我们研究这些差异是否源于两种表述方式所诱导的不同内部表示。我们提出了一种双框架协议，使用仅存在最小差异的提示词——这些提示词分别采用支持型或排除型表述，同时保持评估目标固定不变。为了探测内部计算，我们在提示词末尾附加一个未经训练的特殊标记 [STATE]，并将其残差流激活视为干预接口。在所测试的两个模型中，两种框架均诱导出可分离的 [STATE] 激活，且这些激活集中于中间层。在配对的提示词之间交换这些激活会系统性地改变模型预测，并提升跨框架的答案一致性，从而提供了基于干预的证据，证明这些激活与模型行为密切相关。在实例层面的替换之外……（摘要原文在此处截断）

    arXiv:2609.01081v1 Announce Type: cross  Abstract: Large language models often answer the same multiple-choice question inconsistently when it is posed under support-oriented and elimination-oriented framings. We investigate whether these discrepancies arise from different internal representations induced by the two framings. We introduce a dual-framing protocol with minimally varied prompts that use either support- or elimination-oriented framing while keeping the evaluation target fixed. To probe the internal computation, we append an untrained special token, [STATE], and treat its residual-stream activation as an intervention interface. Across both models, the two framings induce separable [STATE] activations concentrated in intermediate layers. Swapping these activations between paired prompts systematically changes predictions and improves cross-framing agreement, providing intervention-based evidence that the activations are behaviorally relevant. Beyond instance-level substituti
    
[^60]: 将大语言模型评判者与人类判断分布进行事后对齐

    Post-hoc Alignment of LLM-judges to Human Judgment Distribution

    [https://arxiv.org/abs/2609.01073](https://arxiv.org/abs/2609.01073)

    针对大语言模型评判者在预测人类标签分布（软标签）时表现不佳的问题，提出了一种轻量级的熵感知事后对齐方法NAPHA，将大语言模型的输出分布与人类判断分布对齐。

    

    LLM-as-a-judge（LLMaJ）框架为自动评估提供了一种经济高效且可复现的解决方案。然而，当前的评估实践通常将LLMaJ的判断与聚合后的真实标签进行比较，忽略了人类标签变异（HLV）中所包含的宝贵信息。受越来越多旨在利用人类标签变异的研究工作的启发，我们系统地研究了LLMaJ在预测单一聚合真实硬标签以及代表人类判断分布（HJD）的未聚合软标签两方面的表现。我们在五个不同数据集上的结果表明，尽管大语言模型在大多数任务的硬标签预测上达到了接近人类的水平，但它们在预测软标签时表现不佳。为了解决这一局限性，我们提出了NAPHA（熵感知事后对齐），这是一种简单而有效的轻量级事后对齐方法，通过将大语言模型的输出分布与人类判断分布相匹配……

    arXiv:2609.01073v1 Announce Type: new  Abstract: The LLM-as-a-judge (LLMaJ) framework offers a cost-effective and reproducible solution for automatic evaluation. However, current evaluation practices typically compare LLMaJ judgments against aggregated ground-truth labels, overlooking the valuable information contained in Human Label Variation (HLV). Inspired by an increasing line of work that proposes to leverage HLV, we systematically study LLMaJ performance on predicting both a single, aggregated ground truth hard-label and unaggregated soft-labels that represent Human Judgment Distributions (HJD). Our results across five diverse datasets reveal that while LLMs achieve near human-level performance at hard-label prediction on most tasks, they exhibit poor performance when predicting soft-labels. To address this limitation, we propose NAPHA (eNtropy-Aware Post-Hoc Alignment), a simple yet effective lightweight post-hoc alignment method that matches the LLM distribution to the HJD by f
    
[^61]: OUTLETS：基于投机解码主干的输出长度预测

    OUTLETS: Output-Length Prediction from Speculative Decoding Backbones

    [https://arxiv.org/abs/2609.01068](https://arxiv.org/abs/2609.01068)

    该论文发现投机解码框架中草稿解码器的潜在表示蕴含着可预测生成长度的信号，并提出OUTLETS方法，将投机解码主干重新用作轨迹感知的输出长度预测器，从而在几乎不增加额外开销的情况下改进大语言模型服务的资源供给与集群调度。

    

    大语言模型（LLM）服务中输出长度的重尾分布给资源供给和集群调度带来了重大挑战。虽然输出长度预测可以缓解这些问题，但现有方法存在关键缺陷：外部代理模型会增加大量延迟且保真度通常有限，而基于内部状态的方法虽然高效，但仅依赖于对当前模型状态的浅层探测。我们发现了投机解码（SD）与长度预测之间的结构性联系：在先进框架（如 EAGLE-3）中，草稿解码器产生的潜在表示编码了能够预测生成长度的信号。基于这一洞察，我们提出了 OUTLETS（基于投机解码主干的输出长度预测），它将投机解码主干重新用作轨迹感知的长度预测器。当其草稿表示已经为投机解码计算完成时……（摘要在此处被截断）

    arXiv:2609.01068v1 Announce Type: new  Abstract: The heavy-tailed distribution of output lengths in Large Language Model (LLM) serving poses major challenges for resource provisioning and cluster scheduling. Although output-length prediction can mitigate these issues, existing approaches have key drawbacks: external proxy models add substantial latency and often have limited fidelity, whereas internal state-based methods are efficient but rely on shallow probes of current model states. We identify a structural connection between speculative decoding (SD) and length prediction: latent representations produced by the draft decoder in advanced frameworks (e.g., EAGLE-3) encode signals that are predictive of generation length. Building on this insight, we introduce OUTLETS (Output-Length Prediction from Speculative Decoding Backbones), which repurposes the speculative backbone as a trajectory-aware length predictor. When its draft representations are already computed for speculative decodi
    
[^62]: WorldBench：面向多语言智能体的文化扎根基准

    WorldBench: Culturally Grounded Benchmark for Multilingual Agents

    [https://arxiv.org/abs/2609.01056](https://arxiv.org/abs/2609.01056)

    WorldBench是一个涵盖七种语言、八种文化、包含1,600个真实日常任务的多语言智能体基准，并引入约束任务成功率（CTS）指标，以全面评估LLM智能体在真实文化扎根场景中的跨语言多步骤任务执行能力。

    

    尽管基于大语言模型（LLM）的智能体在复杂环境中解决多步骤任务的应用日益增多，现有基准很少测试状态保持能力、跨语言性能以及对真实扎根场景的适用性。为解决这些问题，我们提出了WorldBench：一个全面的、多语言的基准，其任务源自真实且基于人物角色的日常工作流程，智能体可在沙盒环境中通过结构化动作执行操作。WorldBench包含涵盖七种语言和八种文化的1,600个任务，这些任务经过具备特定语言和文化专业知识的人类标注者反馈进行筛选与精炼。在评估方面，我们扩展了以往工作的指标，并引入了约束任务成功率（CTS），该指标结合自然语言指令和测试环境，通过确定性评估和“LLM作为裁判”的评估方式，对任务完成度、最小修改度及其他补充指标进行评分。我们的实验表明，前沿模型在最……（摘要原文在此处截断）

    arXiv:2609.01056v1 Announce Type: new  Abstract: Despite the growing use of LLM-powered agents to solve multi-step tasks in complex environments, existing benchmarks rarely test state preservation, performance across languages, and application to realistic, grounded scenarios. To address these concerns, we present WorldBench: a comprehensive, multilingual benchmark of genuine, persona-grounded everyday workflows, where agents can act in a sandbox via structured actions. WorldBench comprises 1,600 tasks across seven languages and eight cultures, filtered and refined through feedback from human annotators with language- and culture-specific expertise. For evaluation, we extend metrics from previous works and introduce Constrained Task Success (CTS), which combines natural language instructions and testbeds to score task completion, minimal modification, and other complementary metrics through deterministic and LLM-as-a-Judge evaluations. Our experiments show that frontier models reach on
    
[^63]: 滞后耦合：内部表征在具有因果作用之前就已变得可读

    Lagged Coupling: Internal Representations Become Readable Before They Become Causal

    [https://arxiv.org/abs/2609.01048](https://arxiv.org/abs/2609.01048)

    该研究在 Pythia 全系列模型中发现“滞后耦合”现象：线性探针能极早读出内部表征，但利用这些方向进行引导干预却几乎无效，且“可读但不可因果干预”的滞后并不随模型规模增大而缩小。

    

    在整个 Pythia 模型套件上（参数规模 160M-12B、八个检查点、四个任务族），线性探针最早在第 1,000 步就能在每个规模下从残差流中读出目标变量——然而，沿着同一读取方向进行引导干预，在 48 个“模型-检查点”组合中却有 43 个与零效应等价。内部可读性系统性地超前于因果有效性，而且这种滞后并不随规模增大而缩小。我们将这一结构称为“滞后耦合”，并将其分解为三条可分离的轨道：（i）内部可读性，在所有位置从第一个检查点起就已饱和（AUROC ≥ 0.990）；（ii）行为可读性，逐步发展且在更大规模下出现得更晚（12B 模型直到最后一个检查点才达到 0.909）；（iii）因果有效性，几乎总是与零效应等价，早期偶尔甚至适得其反，仅有一处孤立的正向脉冲（12B，第 8,000 步，z = +2.49）是本研究的网格无法解析的。其顺序以“先可读、后可写”为主导（11/11

    arXiv:2609.01048v1 Announce Type: cross  Abstract: Across the full Pythia suite (160M-12B, eight checkpoints, four task families), a linear probe can read a target variable from the residual stream as early as step 1,000 at every scale -- yet steering along that same reading direction remains null-equivalent in 43 of 48 model-checkpoint cells. Internal readability systematically outruns causal efficacy, and the lag does not shrink with scale. We call this structure lagged coupling and decompose it into three dissociable tracks: (i) internal readability, saturated (AUROC >= 0.990) from the first checkpoint everywhere; (ii) behavioral readability, which develops gradually and progressively later at larger scales (12B reaches 0.909 only at the final checkpoint); (iii) causal efficacy, almost always null-equivalent, occasionally counterproductive early, with one isolated positive pulse (12B, step 8,000, z = +2.49) our grid cannot resolve. The ordering is dominantly read-before-write (11/11
    
[^64]: PCoMoE：将MoE推理从单体专家选择转向细粒度路径组合

    PCoMoE: Shifting MoE Inference from Monolithic Expert Selection to Fine-Grained Path Composition

    [https://arxiv.org/abs/2609.01024](https://arxiv.org/abs/2609.01024)

    PCoMoE提出了一种路径组合式执行框架，将MoE推理从粗粒度的整专家选择转变为细粒度的路径组合，通过路径级计算形式化、兼容性感知的逐层剪枝策略和硬件友好的执行引擎，在严格受限的开销下挖掘专家内部的细粒度计算冗余。

    

    混合专家架构通过为每个token激活稀疏的专家子集，高效地扩展了大语言模型的容量。然而，现代MoE推理仍然严重受限于僵化的整专家抽象。现有框架将专家作为原子执行单元进行管理、调度或剪枝，这过早地固定了优化边界，使得专家内部的细粒度计算冗余未被充分探索。在本工作中，我们提出了PCoMoE，一个路径组合执行框架，将MoE推理从粗粒度的专家选择转向细粒度的路径组合。PCoMoE融合了专家计算的路径级形式化描述、用于抑制低价值路径组合的兼容性感知逐层剪枝策略，以及在严格受限开销下利用可复用子专家结构的硬件友好执行引擎。实验结果表明，PCoMoE实现了……（摘要原文在此处截断）

    arXiv:2609.01024v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) architectures scale Large Language Model (LLM) capacity efficiently by activating a sparse subset of experts per token. However, modern MoE inference remains heavily constrained by the rigid, whole-expert abstraction. Existing frameworks manage, schedule, or prune experts as atomic execution units, which fixes the optimization boundary too early and leaves fine-grained intra-expert computational redundancy underexplored. In this work, we present PCoMoE, a path-compositional execution framework that shifts MoE inference from coarse-grained expert selection to fine-grained path composition. PCoMoE incorporates a path-level formulation of expert computation, a compatibility-aware layer-wise pruning strategy to suppress low-value path combinations, and a hardware-friendly execution engine to exploit reusable sub-expert structures under strictly bounded overheads. Experimental results demonstrate that PCoMoE achieves 
    
[^65]: 短语定位的语言对比引导：面向语码转换文本转语音的免训练局部口音控制

    Phrase-Localized Language-Contrastive Guidance: Training-Free Localized Accent Control for Code-Switching Text-to-Speech

    [https://arxiv.org/abs/2609.01016](https://arxiv.org/abs/2609.01016)

    提出了一种免训练的推理框架LCG，通过短语定位的语言对比引导和自注意力探测技术，无需外部对齐或微调即可为语码转换文本转语音中的外语短语恢复母语口音。

    

    当前的语音合成在处理语码转换（即在主要语言的语句中混入外语短语）时存在困难，导致外语短语带有主要语言的口音而非其母语口音。我们提出了短语定位的语言对比引导，这是一种无需训练的推理框架，能够在跨语言文本转语音中为语码转换的短语恢复母语口音。LCG将应用于整个话语的单一语言引导替换为针对每个区域的单独引导，因此每个部分都由其自身的语言进行引导。为了确定在何处应用这种局部化引导，我们提出了一种自注意力探测技术，可以在没有外部对齐的情况下找到短语边界。这些组件共同生成每个区域都带有其自身语言口音的语音，无需微调或辅助模型。在多种语言对中，LCG稳健地提升了母语自然度。

    arXiv:2609.01016v1 Announce Type: new  Abstract: Current speech synthesis struggles with code-switching, which mixes a foreign language phrase into a primary language utterance, causing the phrase to be spoken with the primary language's accent rather than its native one. We propose Phrase-Localized Language-Contrastive Guidance (LCG), a training-free inference framework that restores a native accent to code-switched phrases in cross-lingual text-to-speech. LCG replaces the single language guidance applied across the whole utterance with a separate guidance for each region, so each part is guided by its own language. To choose where to apply this localized guidance, we propose a self-attention probing technique that finds the phrase boundaries without external alignments. Together, these components generate speech in which each region carries the accent of its own language, requiring no fine-tuning or auxiliary models. Across diverse language pairs, LCG robustly increases the nativenes
    
[^66]: SinkPruner：面向多模态大语言模型的无Sink视觉token剪枝方法

    SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models

    [https://arxiv.org/abs/2609.01004](https://arxiv.org/abs/2609.01004)

    提出无需训练的视觉token剪枝框架SinkPruner，通过过滤高度冗余的高范数离群token并缓解注意力汇聚现象，在保持多模态理解能力的同时实现高效的多模态大语言模型推理。

    

    尽管多模态大语言模型（MLLM）具有强大的多模态理解能力，但其在处理长视觉token序列时会产生巨大的计算开销。为降低推理成本，近期研究探索了基于视觉中心策略或文本引导策略的视觉token剪枝方法。然而，这些方法往往忽视了高范数离群token（即特征范数异常大的token），导致次优的剪枝决策。在本工作中，我们证明这类高范数离群token在特征维度和空间维度上都高度冗余，但现有方法却常常错误地将其作为信息线索而保留。受此观察启发，我们提出了SinkPruner，一个无需训练的视觉token剪枝框架，用于实现高效的MLLM推理。SinkPruner遵循由粗到细的设计，包含两个关键模块：一个用于过滤高范数冗余并缓解注意力汇聚（attention sink）现象的视觉净化器……

    arXiv:2609.01004v1 Announce Type: cross  Abstract: Despite their strong multimodal understanding ability, multimodal large language models (MLLMs) incur substantial computational overhead when processing long visual token sequences. To reduce inference costs, recent studies have explored visual token pruning through vision-centric or text-guided strategies. However, these methods often overlook high-norm outlier tokens, i.e., tokens with abnormally large feature norms, leading to suboptimal pruning decisions. In this work, we show that such high-norm outlier tokens are highly redundant in both feature and spatial dimensions, yet are often mistakenly preserved as informative cues by existing methods.   Motivated by this observation, we propose SinkPruner, a training-free visual token pruning framework for efficient MLLM inference. SinkPruner follows a coarse-to-fine design with two key modules: a visual sanitizer that filters high-norm redundancies and alleviates attention sink and atte
    
[^67]: 正确的框架，错误的规则：文化线索暴露了它们本意想弥补的金融知识差距

    Right Frame, Wrong Rule: Cultural Cues Expose the Financial Knowledge Gap They Were Meant to Close

    [https://arxiv.org/abs/2609.00999](https://arxiv.org/abs/2609.00999)

    该论文提出“规范多元性”这一新评估设定，通过将框架选择与框架内正确性分离，揭示了“刻板印象陷阱”——文化线索虽能引导大模型选择伊斯兰金融框架，却在框架内暴露出高达57%至66%的错误率，表明传统二选一评估会严重高估模型的文化对齐能力。

    

    当一个问题在不同规范框架下都有有效答案时，语言模型必须决定采用哪个框架，以及它能否在该框架内正确作答。我们将这种情境称为“规范多元性”，并以伊斯兰金融为研究对象，采用一种将框架选择与框架内正确性区分开来的四选一分类法进行研究。这种区分揭示了“刻板印象陷阱”：文化线索引导模型走向某一框架，但模型却在该框架内选择了错误的答案。在十二个模型、两种语言和五十个人口统计信号的测试中，文化线索会改变模型的框架选择，并暴露出显著的准确率差异，尤其是在非前沿模型中。在最强信号的作用下，大型开源权重模型有97%的概率选择伊斯兰金融框架。若采用二选一的评估方式，将会报告近乎完美的对齐度，尽管其中57%至66%的选择实际上是错误的。这些发现为……提供了依据，但并未……（原文摘要在此处截断）

    arXiv:2609.00999v1 Announce Type: cross  Abstract: When a question has valid answers under different normative frameworks, a language model must decide which framework to use and whether it can answer correctly within it. We call this setting normative pluralism and study it in Islamic finance using a four-choice taxonomy that separates framework selection from within-framework correctness. This separation reveals the stereotype trap: a cultural cue steers a model toward one framework, but the model selects an incorrect answer within that framework. Across twelve models, two languages, and fifty demographic signals, cultural cues change framework selection and reveal substantial differences in accuracy, especially among non-frontier models. Under the strongest signal, large open-weight models select the Islamic framework 97% of the time. A two-choice evaluation would report near-perfect alignment, although 57--66% of those selections are incorrect. These findings motivate, but do not d
    
[^68]: Inspicio：面向历史语言的开放式词汇、基于大语言模型的词义检索

    Inspicio: Open-Vocabulary, LLM-Based Sense Retrieval for Historical Languages

    [https://arxiv.org/abs/2609.00998](https://arxiv.org/abs/2609.00998)

    提出了Inspicio，一个无需源语言词义清单的开放式词汇检索流水线，利用大语言模型生成的英文翻译、候选定义和词元，通过混合检索将历史语言文本中的词元直接链接到开放英语WordNet的同义词集。

    

    词义消歧在英语及少数资源丰富的现代语言中进展迅速，但它始终假设源语言中存在词义清单（sense inventory）以及词到词义的映射关系（Navigli, 2026）。对于大多数历史语言和低资源语言而言，这些假设不再成立，因为它们专用的WordNet要么不完整，要么仍在构建中。我们提出了Inspicio，这是一个开放词汇的检索流水线，能够将上下文中的词元直接链接到开放英语WordNet（McCrae et al., 2020）的同义词集（synset），而无需任何源语言的词义清单或映射。对于每个词的出现实例，一个经过指令微调的大语言模型会生成两句周围句子的英文翻译、一小组候选的词典式定义以及若干候选英文词元。这些输出驱动一个混合检索步骤，该步骤结合了稠密的定义-同义词集相似度、稀疏的词元匹配以及最大边际相关性

    arXiv:2609.00998v1 Announce Type: cross  Abstract: Word Sense Disambiguation has advanced rapidly for English and a handful of well-resourced modern languages, but it continues to assume the existence of a sense inventory and a word-to-sense mapping in the source language (Navigli, 2026). These assumptions break down for most historical and low-resource languages, whose dedicated WordNets are either incomplete or still under construction. We present Inspicio, an open-vocabulary retrieval pipeline that links tokens in context to synsets of the Open English WordNet (McCrae et al., 2020) without requiring any source-language inventory or mapping. For each occurrence, an instruction-tuned LLM produces two English translations of the surrounding sentence, a small set of candidate dictionary-style definitions, and a few candidate English lemmas. These outputs drive a hybrid retrieval step that combines dense definition-synset similarity, sparse lemma matching, and Maximal Marginal Relevance 
    
[^69]: 面向陪伴型智能体评估的披露门控用户模拟

    Disclosure-Gated User Simulation for Companion-Agent Evaluation

    [https://arxiv.org/abs/2609.00982](https://arxiv.org/abs/2609.00982)

    提出披露门控用户模拟方法，让模拟用户根据陪伴型智能体的行为决定信息披露深度，以纠正模拟用户过度配合、使被测系统仅靠提问数量即可得分的评估缺陷。

    

    使用大型语言模型扮演用户如今已成为可扩展评估的标准做法。但它存在一个反复被诊断出的缺陷：模拟用户过度配合，导致被测系统可以仅凭大量提问来得分，而不是通过让用户愿意开口说话来得分。作为回应，我们提出了一种披露门控，将信息披露与陪伴型智能体的行为相挂钩：其状态是一个由五个有序门控构成的阶梯，并归并为三个可观测的深度层。我们对该机制进行了规范定义、消融实验和审计，并依据该规范训练了一个用户模拟器。门控行为从训练语料的合成分支中学习，而真实分支则提供人们真实的说话与反应方式；训练完成后，模拟器在运行时无需被告知每条信息位于哪个门控之后。该门控是环境中的承重组件：在一个已发布的陪伴型智能体基准测试的英文语料库（CompanionBench）上，一旦训练……

    arXiv:2609.00982v1 Announce Type: cross  Abstract: Using a large language model to play the user is now standard in scalable evaluation. It has a repeatedly diagnosed failure: the simulated user is excessively cooperative, so a system under test can score by the sheer number of questions it asks rather than by making the user willing to speak. We answer with a disclosure gate conditioning information release on the companion agent's behaviour: its state is a ladder of five ordered gates, merged onto three observable depth layers. We specify, ablate, and audit it, and train a user simulator against that specification. Gating behaviour is learned from the training corpus's synthetic branch, while the real branch supplies how people speak and react; after training, the simulator need not be told at runtime which gate each item sits behind. The gate is a load-bearing component of the environment: on the English corpus of a published companion-agent benchmark (CompanionBench), once training
    
[^70]: PersianAnonymizer：评估基于大语言模型标注训练的高效波斯语命名实体识别匿名化方法

    PersianAnonymizer: Evaluating LLM-Labeled Training for Efficient NER-based Anonymization in Persian

    [https://arxiv.org/abs/2609.00958](https://arxiv.org/abs/2609.00958)

    该论文通过比较三个大语言模型为波斯语客户聊天生成标注数据来训练轻量级NER匿名化模型，发现GPT-OSS零样本标注训练的模型性能最佳，且推理速度远快于直接使用大语言模型。

    

    我们致力于实现波斯语客户聊天的实用化匿名化，方法是利用大语言模型标注的监督数据训练一个紧凑的NER模型，并从中选出最适合部署的最佳标注模型。我们比较了三个指令微调的大语言模型：DeepSeek-V3-0324、GPT-OSS-120B和Qwen3-235B-A22B-Instruct-2507，它们在统一的JSON协议下生成文本跨度标注，由此构建了四个语料库。我们基于MatinaRoberta为每个语料库分别训练词元分类器，并采用词元级别的精确率/召回率/F1（总体及分类别）进行评估。我们还报告了标签覆盖率召回率，即金标准非O词元被预测为非O的比例，并通过测试标注上的词元级维恩图量化不同标注模型之间的行为差异。最后，我们对比了大语言模型在H200节点上标注测试集的延迟，与训练好的NER模型在单张RTX 3090上推理标注的速度。结果表明，来自OSS_ZeroShot的监督数据产生了最强的宏F1...

    arXiv:2609.00958v1 Announce Type: new  Abstract: We target practical anonymization of Persian customer chats by training a compact NER model from LLM-labeled supervision and selecting the best labeler for deployment. We compare three instruction-tuned LLMs: DeepSeek-V3-0324, GPT-OSS-120B, and Qwen3-235B-A22B-Instruct-2507, to produce span annotations under a shared JSON protocol, yielding four corpora (OSS_ZeroShot, Qwen_ZeroShot, Qwen_FewShot, DeepSeek_FewShot). A MatinaRoberta-based token-classifier is trained per corpus and evaluated with token-level Precision/Recall/F1 (overall and per-class). We also report Label Coverage Recall (LCR), the proportion of gold non-O tokens predicted as non-O, and quantify cross-labeler behavior via a token-level Venn on test annotations. Finally, we contrast test-set annotation latency of the LLMs on H200 nodes with the trained NER's test-time labeling on a single RTX 3090. Results show that supervision from OSS_ZeroShot yields the strongest macro-F
    
[^71]: 校准是瓶颈：多轮工具调用的动作类别诊断

    Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling

    [https://arxiv.org/abs/2609.00949](https://arxiv.org/abs/2609.00949)

    本文提出一个基于四类动作空间的诊断框架，通过引入“准确率不超过黄金动作召回率”的自揭示上界，将多轮工具调用失败分解为动作类别失准与动作执行失败两种正交模式，从而揭示开源模型总体准确率追平闭源模型的表象背后，动作类别校准才是真正的瓶颈。

    

    多轮工具调用是大语言模型（LLM）智能体的一项核心评测场景。在公开的工具调用基准上，开源权重模型的总体准确率已接近甚至超越闭源前沿模型。然而，这一指标是对众多不同多轮情境的取平均，掩盖了进展是否在这些情境之间均衡分布。我们提出一种面向动作类别的诊断框架，将多轮失败分解为两种正交模式：动作类别失准与动作执行失败。该框架在四类动作空间（TOOL_CALL/ASK/REFUSE/CONFIRM）上运行，并引入一个自我揭示的上界 Acc ≤ GAR（黄金动作召回率）；两种失败模式分别表现为上界被违反（Acc > GAR，暴露出状态评分器对失准的掩盖）以及较大的上界余量（GAR >> Acc，将执行失败定位于 TOOL_CALL 内部）。我们在一组工具调用模型上对该框架进行了验证……（原文摘要在此处截断）

    arXiv:2609.00949v1 Announce Type: cross  Abstract: Multi-turn tool calling is a core evaluation scenario for large language model (LLM) agents. On public tool-calling benchmarks, open-weight models now approach or even surpass closed-source frontier models in aggregate accuracy. However, this metric averages over many different multi-turn situations and obscures whether progress is balanced across them. We propose an action-class-oriented diagnostic framework that decomposes multi-turn failures into two orthogonal modes: action-class miscalibration and action-execution failure. The framework operates over a four-class action space (TOOL_CALL/ASK/REFUSE/CONFIRM) and introduces a self-revealing upper bound Acc <= GAR (Gold Action Recall); the two modes show up as bound violation (Acc > GAR, exposing state-grader masking of miscalibration) and large bound slack (GAR >> Acc, localizing execution failure within TOOL_CALL). We validate it on a panel of tool-calling models across multiple mul
    
[^72]: 从术语到图示：面向科学图表理解的视觉指令生成

    From Terminology to Diagrams: Visual-Instruction Generation for Scientific Diagram Understanding

    [https://arxiv.org/abs/2609.00948](https://arxiv.org/abs/2609.00948)

    该论文提出SciGram框架与数据集，通过科学课程术语自动生成涵盖19.4万张图表和140万条视觉指令的大规模训练数据，显著提升了视觉语言模型在科学图表理解任务上的表现。

    

    视觉语言模型（VLMs）在自然图像的视觉问答任务中展现出强大的性能。然而，它们在处理科学图表时仍然存在困难，因为科学图表旨在传达功能性或关系性含义，而非字面上的场景。因此，我们提出了一个通过利用源自科学课程的术语来生成大规模基于图表的指令数据的框架。我们的方法系统地提取领域概念、合成原子事实、从网络检索相关图表，并以图表说明和选择题的形式生成多模态监督信号。利用这一流程，我们构建了SciGram数据集，其中包含超过19.4万张图表和140万条视觉指令，涵盖生命科学、地球科学和物理科学。尽管依赖于带有噪声的网络数据和合成标注，在SciGram上微调的模型在以图表为中心的基准测试中仍取得了显著的性能提升。

    arXiv:2609.00948v1 Announce Type: cross  Abstract: Vision-language models (VLMs) have demonstrated strong performance in visual question answering with natural images. However, they continue to struggle with scientific diagrams, which are designed to convey functional or relational meaning rather than literal scenes. We therefore introduce a framework for generating large-scale diagram-grounded instruction data by leveraging terminology derived from scientific curricula. Our approach systematically extracts domain concepts, synthesizes atomic facts, retrieves relevant diagrams from the web, and generates multimodal supervision in the form of diagram captions and multiple-choice questions. Using this pipeline, we construct SciGram, a dataset of over 194K diagrams and 1.4M visual instructions across life, earth, and physical sciences. Despite relying on noisy web data and synthetic annotations, models fine-tuned on SciGram achieve substantial improvements on diagram-centric benchmarks, i
    
[^73]: 建模迭代问题求解的数据集

    A Dataset for Modeling Iterative Problem-Solving

    [https://arxiv.org/abs/2609.00940](https://arxiv.org/abs/2609.00940)

    该论文发布了CodeInsight大规模数据集，包含3,286名本科生在两个学年内2门C++入门课程中的超过300万次代码提交，用于建模迭代问题求解中学习者根据反馈反复修改的序列学习动态。

    

    通过反复尝试解决问题是一项序列建模任务：在每一步中，求解者接收反馈并决定如何修改其解决方案。预测性能在多次尝试中是提升、停滞还是退步，是理解人类学习者和自主智能体迭代问题求解过程的核心。除了结果之外，对哪些错误持续存在以及策略如何在多次尝试之间转变进行建模，能更深入地洞察序列学习的机制。研究这些动态需要观察众多求解者进行尝试、接收反馈并进行修改的过程。具有自动评分的编程课程恰好提供了这样的场景，因为学生迭代地向测试套件提交代码，并且每次尝试都能收到反馈。因此，我们整理了CodeInsight，这是一个大规模数据集，包含来自2个学年中2门C++入门课程的3,286名本科生的超过300万次代码提交，并带有测试用例级别的……

    arXiv:2609.00940v1 Announce Type: new  Abstract: Solving problems through repeated attempts is a sequential modeling task: at each step, the solver receives feedback and decides how to revise their solutions. Predicting whether performance improves, plateaus, or regresses across attempts is central to understanding any iterative problem-solving process in both human learners and autonomous agents. Beyond outcomes, modeling what errors persist and how strategies shift across attempts provides deeper insight into the mechanics of sequential learning. Studying these dynamics requires observing many solvers as they attempt, receive feedback, and revise. Programming courses with automated grading provide this setting, as students iteratively submit code to test suites and receive feedback on every attempt. We therefore curate CodeInsight, a large-scale dataset of over 3 million submissions from 3,286 undergraduates across 2 introductory C++ courses in 2 academic years, with test-case-level 
    
[^74]: DualStake：深度研究智能体中的双路径置信度校准

    DualStake: Dual-Path Confidence Calibration in Deep Research Agents

    [https://arxiv.org/abs/2609.00935](https://arxiv.org/abs/2609.00935)

    提出DualStake双路径置信度校准方法，通过在每次检索后引出证据置信度并在答案生成后引出答案置信度，利用边界裁剪的置信度相关stake奖励将两者与答案正确性联合对齐，有效缓解深度研究智能体的严重过度自信问题。

    

    深度研究智能体通过多轮检索和面向决策的生成来解决知识密集型任务。然而，这类智能体存在严重的过度自信问题，导致其表达的置信度对于用户信任和下游弃答决策而言并不可靠。为解决这一问题，我们在深度研究流程的每次检索之后增加了步骤置信度引出环节，并以常用的答案后言语化置信度为基础。有趣的是，我们发现证据置信度——在最后一次检索步骤后引出的置信度——比答案置信度——在答案生成后引出的置信度——能提供更强的不确定性信号，且答案置信度在很大程度上受到证据置信度的塑造。基于这些发现，我们提出了DualStake，一种双路径校准方法，通过施加边界裁剪的、置信度相关的stake奖励，将证据置信度和答案置信度与答案正确性联合对齐，同时抑制对极端置信度的过度优化。实验……

    arXiv:2609.00935v1 Announce Type: cross  Abstract: Deep Research agents tackle knowledge-intensive tasks through multi-round retrieval and decision-oriented generation. However, these agents suffer from severe overconfidence, making their expressed confidence unreliable for user trust and downstream abstention. To address this, we augment the Deep Research pipeline with step confidence elicitation after each retrieval, building on the commonly used post-answer verbalized confidence. Interestingly, we find that Evidence Confidence (E-Conf), elicited after the final retrieval step, provides a stronger uncertainty signal than Answer Confidence (A-Conf), elicited after answer generation, and that A-Conf is largely shaped by E-Conf. Based on these findings, we propose DualStake, a dual-path calibration method that applies margin-clipped, confidence-dependent stake rewards to jointly align E-Conf and A-Conf with answer correctness while limiting extreme confidence optimization. Experiments o
    
[^75]: 上下文接地增益由既有机制介导：对GRPO、SFT和DPO的审计

    Context-Grounding Gains Are Mediated by Pre-existing Machinery: Auditing GRPO, SFT, and DPO

    [https://arxiv.org/abs/2609.00925](https://arxiv.org/abs/2609.00925)

    本文通过从同一检查点系统审计GRPO、SFT和DPO共九种后训练方案，发现语言模型遵循冲突提示证据的接地增益主要源于强化模型中已有的机制（与起始模型相同的因果注意力头集合），而非学习新机制，其中GRPO增益很小、冲突SFT提升适中、DPO在其匹配分布上接近上限。

    

    当提示中的证据与模型记忆中的知识冲突时，语言模型可能会忽略提示中的证据。后训练可以让模型更可靠地遵循这类证据，但这些增益究竟需要新的机制，还是通过强化已有的机制来实现，目前尚不清楚。我们从同一个起始检查点比较了涵盖GRPO、SFT和DPO的九种后训练方案，并将关键比较扩展到不同规模和不同模型家族。我们在训练之前从该起始检查点估计了一个“接地方向”。在测试的五种GRPO变体中，接地增益都很小。对于两种在不同随机种子下可复现的变体，等价性检验表明，即使被奖励的指标有所提升，它们的效果仍低于冲突SFT所带来的增益。冲突SFT适度地改善了接地能力，而DPO在其匹配的分布上使接地能力接近上限。冲突SFT和DPO在很大程度上使用与起始模型相同的因果注意力头集合。减去起始模型的方向会同时抑制两者……

    arXiv:2609.00925v1 Announce Type: cross  Abstract: Language models can ignore prompt evidence when it conflicts with memorized knowledge. Post-training can make models follow such evidence more reliably, but it is unclear whether these gains require new machinery or strengthen machinery already present. We compare nine post-training arms spanning GRPO, SFT, and DPO from one starting checkpoint, with key comparisons extended across scales and families. We estimate a grounding direction from that checkpoint before training. Across five tested GRPO variants, grounding gains are small. For the two variants replicated across seeds, equivalence tests bound their effects below the conflict-SFT gain even as the rewarded metric improves. Conflict-SFT improves grounding moderately, while DPO drives grounding near ceiling on its matched distribution. Conflict-SFT and DPO largely use the same causal attention-head set as the starting model. Subtracting the starting-model direction suppresses both 
    
[^76]: VIBE-Bench：当用户画像不等于偏好时，评估个性化大语言模型

    VIBE-Bench: Evaluating Personalized Large Language Models When Profiles Don't Mean Preferences

    [https://arxiv.org/abs/2609.00921](https://arxiv.org/abs/2609.00921)

    该论文提出了VIBE-Bench基准，揭示当前个性化大语言模型在“画像-偏好概念错位”情形下（即用户画像线索与查询偏好处于不同概念空间时）因过度依赖浅层语义关联而失效，需要具备超越表面语义的跨概念偏好推理能力。

    

    个性化大语言模型（PLLMs）旨在为个体用户定制回复，其核心挑战在于偏好推理：从用户相关历史中推断与查询相关的偏好。然而，现有基准测试大多假设这种偏好可以从语义相关的历史中检索得到。我们研究了一个尚未被充分探索但具有重要实践意义的情形——画像-偏好概念错位（PRCM），即可观察的画像线索与特定查询的偏好处于不同的概念空间，使得语义检索无法可靠地支持个性化。我们提出了VIBE-Bench，这是一个包含两个基于心理学设计的任务、3,504个人设（persona）和12,239段对话的基准，其中包括一个经过人工验证的黄金测试集，并要求模型具备超越表面语义重叠的跨概念偏好推理能力。对多种个性化方法的实验表明，当前的PLLMs在很大程度上依赖于浅层语义关联，因而难以应对此类情形。

    arXiv:2609.00921v1 Announce Type: new  Abstract: Personalized Large Language Models (PLLMs) aim to tailor responses to individual users, where a central challenge is preference reasoning: inferring query-relevant preferences from user-related history. Existing benchmarks, however, largely assume that such preference can be retrieved from semantically related history. We study an underexplored but practically important regime, profile-preference conceptual misalignment (PRCM), where observable profile cues and query-specific preferences lie in different concept spaces, making semantic retrieval inconsistent for personalization. We introduce VIBE-Bench, a benchmark with two psychology-grounded tasks, 3,504 personas and 12,239 dialogues, including a manually verified gold test set, and requires cross-concept preference reasoning beyond surface semantic overlap. Experiments with several personalization methods show that current PLLMs largely rely on shallow semantic correlations and fail t
    
[^77]: RPCBench：面向基于大语言模型推荐中主动前提批判的基准测试

    RPCBench: A Benchmark for Proactive Premise Critique in LLM-based Recommendation

    [https://arxiv.org/abs/2609.00918](https://arxiv.org/abs/2609.00918)

    该论文提出了 RPCBench 基准，首次系统评估大语言模型在推荐场景中主动检测、诊断并妥善处理用户请求中错误前提的能力，涵盖五个推荐领域、十种前提失败类型，并提供了细粒度的评估框架。

    

    大语言模型越来越多地被用作交互式推荐助手。因此，对它们的评估应当超越生成看似合理的物品推荐，而是测试其能否识别有缺陷的推荐请求。现有的推荐系统基准主要评估排序、生成或偏好满足能力，而现有的错误检测基准通常不基于推荐场景特有的用户与候选证据。为了填补这一空白，我们提出了 RPCBench，这是一个用于评估“推荐器前提批判”能力的基准：即在自然语言推荐请求中检测、诊断并妥善处理错误前提的能力。RPCBench 包含来自五个推荐领域的基于证据的测试实例，涵盖十种前提失败类型。每个实例提供一个可见的推荐上下文和一个被污染的用户查询。我们进一步设计了一个细粒度的评估框架，用于衡量主动检测……（原文摘要在此处截断）

    arXiv:2609.00918v1 Announce Type: new  Abstract: Large language models are increasingly used as interactive recommender assistants. Their evaluation should therefore go beyond plausible item recommendation and test whether they can recognize flawed recommendation requests. Existing recommender benchmarks mainly assess ranking, generation, or preference satisfaction, while existing error-detection benchmarks are usually not grounded in recommendation-specific user and candidate evidence. To address this gap, we introduce RPCBench, a benchmark for evaluating Recommender-Premise Critique: the ability to detect, diagnose, and properly handle faulty premises in natural-language recommendation requests. RPCBench contains evidence-grounded test instances from five recommendation domains and covers ten types of premise failures. Each instance provides a visible recommendation context and a corrupted user query. We further design a fine-grained evaluation framework that measures proactive detec
    
[^78]: 基于词元级记忆不对称性的微调扩散语言模型成员推断攻击

    Membership Inference in Fine-tuned Diffusion Language Models via Token-level Memorization Asymmetry

    [https://arxiv.org/abs/2609.00873](https://arxiv.org/abs/2609.00873)

    该论文通过理论分析发现扩散语言模型中的“词元级记忆不对称”现象，并据此提出了基于分位数加权偏度的Q-Skew指标，实现了对微调扩散语言模型的高效成员推断攻击，揭示了一个新的隐私攻击面。

    

    扩散语言模型（DLMs）近来作为自回归语言模型的替代建模范式而兴起，具有并行生成和双向上下文建模等优势。尽管人们对其生成能力的兴趣日益增长，但扩散语言模型的隐私风险仍未得到充分探索。我们通过对扩散训练动态的理论分析，识别出一种称为“词元级记忆不对称”的现象。基于这一发现，我们提出了Q-Skew——一种基于分位数加权偏度指标的成员推断方法，可用于对微调后的扩散语言模型进行成员推断攻击。在多个微调数据集和模型上的实验表明，我们的方法优于现有基线。此外，我们还展示了Q-Skew可以辅助实现其他隐私侵犯行为，例如个人身份信息（PII）提取。我们的发现揭示了一个此前未被充分探索的隐私攻击面，并强调了对扩散语言模型进行系统性隐私评估的必要性。

    arXiv:2609.00873v1 Announce Type: new  Abstract: Diffusion language models (DLMs) have recently emerged as an alternative modeling paradigm to autoregressive LMs, offering advantages such as parallel generation and bidirectional context modeling. Despite growing interest in their generative capabilities, the privacy risks of DLMs remain underexplored. We identify a phenomenon termed token-level memorization asymmetry through theoretical analysis of diffusion training dynamics. Building on this finding, we propose Q-Skew, a quantile-weighted skewness-based indicator for membership inference on finetuned DLMs. Experiments across multiple fine-tuning datasets and models show that our method outperforms existing baselines. Moreover, we show that Q-Skew can also facilitate other privacy violations, such as PII extraction. Our findings reveal a previously underexplored privacy attack surface and highlight the need for systematic privacy evaluation of DLMs.
    
[^79]: 视觉不敏感性差距：诊断视觉-语言模型何时未能利用视觉证据

    The Visual Insensitivity Gap: Diagnosing When Vision-Language Models Fail to Use Visual Evidence

    [https://arxiv.org/abs/2609.00868](https://arxiv.org/abs/2609.00868)

    该论文发现“视觉不敏感性差距”现象——在40%–97%的多模态基准样本上，模糊与问题相关的关键视觉区域几乎不改变VLM的输出，并证明这种不敏感性是样本层面的属性（跨模型VSI排名显著相关），即使各模型的视觉编码器本身实际上能够检测到这些扰动。

    

    视觉-语言模型（VLM）通常通过在多模态基准上的总体准确率进行评估，这种做法隐含地假设模型确实使用了其视觉输入。我们证明这一假设在六个VLM和三个感知基准的40%–97%样本上并不成立：将问题相关的视觉区域模糊化后，模型的下一个词元分布几乎不变。我们将这一现象命名为“视觉不敏感性差距”，并用逐样本的视觉敏感性指数（VSI）对其进行量化。该差距是样本本身的属性，而非模型的属性：VSI排名在各模型之间呈现相关性（总体平均Spearman rho=+0.40，置换检验p<10^-3），因此即使这些VLM之间除了对比预训练的视觉编码器外不共享任何架构细节，相同的样本仍会被它们共同标记为“不敏感”。其机制是具体的：在不敏感样本上，对每个模型自身的视觉编码器进行线性探针可以以0.72–0.79的准确率区分受扰动图像与原始图像，然而模型的argmax词元却几乎没有变化

    arXiv:2609.00868v1 Announce Type: cross  Abstract: Vision-language models are evaluated by aggregate accuracy on multimodal benchmarks, a practice that implicitly assumes the model uses its visual input. We show this assumption fails on 40%--97% of samples across six VLMs and three perceptual benchmarks: blurring the question-relevant visual region leaves the next-token distribution nearly unchanged. We name this phenomenon the Visual Insensitivity Gap and quantify it with a per-sample Visual Sensitivity Index (VSI). The gap is a property of samples, not of models: VSI ranks correlate across models (grand-mean Spearman rho=+0.40, permutation p<10^-3), so the same samples are flagged insensitive by VLMs sharing no architectural detail beyond a contrastively pretrained vision tower. The mechanism is concrete: on the insensitive samples, a linear probe on each model's own vision tower distinguishes perturbed from clean images at 0.72--0.79 accuracy, yet the model's argmax token changes on
    
[^80]: MemoryWalker：停止在智能体从未见过的上下文上训练智能体

    MemoryWalker: Stop Training Agents on Contexts They Never Saw

    [https://arxiv.org/abs/2609.00865](https://arxiv.org/abs/2609.00865)

    该论文针对上下文压缩导致智能体训练时有效历史呈树状分支的问题，提出了两种梯度等价的精确修正方法（LogitTree 与 4D 注意力掩码）以及一种仅需单次反向传播的自蒸馏方法 SDCC，从而消除压缩训练与推理之间的条件化不一致。

    

    Claude Code 和 Qwen-Agent 等生产级智能体框架在执行过程中会压缩上下文，但在压缩条件下进行训练会产生一个条件化问题：每次上下文剔除都会使有效历史产生分支，因此学习对象是一棵树而非一个序列。现有的线性化方法要么保留最右路径，导致“时间旅行”式信息泄露；要么重放深度优先遍历，导致训练与推理不匹配。我们提出两种精确且梯度等价的修正方法：LogitTree（一种分段 K 次前向遍历）和打包式 4D 注意力掩码。LogitTree 需要 K+1 次反向传播；4D 掩码则需要自定义内核和白盒化的剔除记录。我们还提出了 SDCC（面向条件化一致性的自蒸馏），这是一种仅需单次反向传播的变分松弛方法。在每次剔除时，它在重建的剔除前前缀上最小化压缩学生模型与停止梯度的教师模型之间的前向 KL 散度，并通过每个分支点的残差 KL 项……

    arXiv:2609.00865v1 Announce Type: cross  Abstract: Production agent harnesses such as Claude Code and Qwen-Agent compress context during rollout, but training under compression creates a conditioning problem: every eviction branches the effective history, so the learning object is a tree rather than a sequence. Existing linearizations either retain the rightmost path, causing time-travel leakage, or replay a depth-first traversal, causing train-inference mismatch. We introduce two exact, gradient-equivalent corrections: LogitTree, a segmented K-forward traversal, and a packed 4D attention mask. LogitTree requires K+1 backward passes; the 4D mask requires a custom kernel and white-box eviction records. We also propose SDCC (Self-Distillation for Conditioning Consistency), a single-backward-pass variational relaxation. At each eviction, it minimizes forward KL between the compressed student and a stop-gradient teacher on the reconstructed pre-eviction prefix. A residual per-junction KL o
    
[^81]: 可验证的灾害故事线与因果知识图谱：基于引用溯源的异构人道主义数据源流水线

    Verifiable Disaster Storylines and Causal Knowledge Graphs: A Citation-Grounded Pipeline from Heterogeneous Humanitarian Sources

    [https://arxiv.org/abs/2609.00858](https://arxiv.org/abs/2609.00858)

    该论文提出了一个基于检索增强生成（RAG）的流水线，融合EM-DAT结构化灾害记录与ReliefWeb、EMM非结构化文档，自动生成涵盖17个字段的灾害故事线和因果知识图谱，且每个节点和边均附带引用溯源，实现了对原始信息源的完全可追溯性，为人道主义响应提供可验证的态势感知支持。

    

    有效的人道主义响应依赖于对异构、海量信息源的快速综合——在危机爆发的关键早期阶段，这一任务常常超出人类分析能力的极限。我们提出了一个流水线，将来自EM-DAT的结构化灾害记录与来自ReliefWeb和欧洲媒体监测（EMM）的非结构化文档相结合，生成有来源依据的灾害故事线和因果知识图谱，为响应人员和分析人员提供态势感知支持。利用检索增强生成（RAG）技术，该流水线提取结构化故事线——即涵盖17个字段的表格化事件概况，内容包括灾害严重程度、关键驱动因素以及儿童敏感影响指标等——并构建因果知识图谱，其中每个节点和边都配有基于引用的解释性叙述，从而实现对原始来源的完全可追溯性。我们通过人工评估在三个不同的危机用例上对该系统进行了评估……

    arXiv:2609.00858v1 Announce Type: new  Abstract: Effective humanitarian response depends on the rapid synthesis of heterogeneous, high-volume information sources - a task that routinely exceeds human analytical capacity in the critical early hours of a crisis. We present a pipeline that combines structured disaster records from EM-DAT with unstructured documents from ReliefWeb and the European Media Monitor (EMM) to produce source-grounded disaster storylines and causal knowledge graphs supporting situational awareness for responders and analysts. Using Retrieval-Augmented Generation, the pipeline extracts structured storylines - tabular event profiles covering 17 fields, from severity and key drivers to child-sensitive impact indicators - and constructs causal knowledge graphs where each node and edge is enriched with citation-grounded explanatory narratives, enabling full traceability back to primary sources. We evaluate the system on three diverse crisis use cases through a human ev
    
[^82]: 分阶段语言播种：面向AI呼叫中心已验证单元问答的接地查询扩展

    Staged Linguistic Seeding: Grounded Query Expansion for Verified-Unit QA in AI Contact Centers

    [https://arxiv.org/abs/2609.00844](https://arxiv.org/abs/2609.00844)

    提出分阶段语言播种（SLS）方法，通过“人工撰写槽位模板—大模型生成变体—轻量人工审核”的流程离线增强检索索引，使AI呼叫中心仅凭单次检索、无查询时生成即可从已验证问答单元中高准确率作答，在两个工业领域将混合R@1提升至0.881/0.930。

    

    AI呼叫中心（AICC）中的客服问答面临着基准测试QA所忽视的部署约束：语音热线延迟要求严苛，且自动回答缺乏依据或出错时代价高昂。我们部署了一个仅从封闭的已验证问答单元集合中进行回答的系统：它要么逐字返回检索到的单元，要么转路由至澄清、拒答或人工转接。该索引通过分阶段语言播种（SLS）在离线阶段进行增强：由人工为每个单元撰写基于真实场景的槽位模板，gpt-4.1-mini将其渲染为变体，再经轻量级人工审核进行过滤。同一套方法论在两个领域中复用，因此推理仅保持单次检索，无需查询时生成。在来自两个工业领域的留出查询变体上，SLS将混合检索R@1提升至0.881/0.930（+0.27/+0.34），且在测试的全部五种检索器上均获得收益。在相同的gpt-4.1-mini生成预算下，SLS比doc2query高出+0.20/+0.32，同时跨来源评估提供了（原文摘要在此处截断）

    arXiv:2609.00844v1 Announce Type: new  Abstract: Customer-service QA in an AI contact center (AICC) runs under deployment constraints that benchmark QA misses: tight voice-hotline latency and a high cost for unsupported or wrong automatic answers. We deploy a system that answers only from a closed set of verified QA units: it returns a retrieved unit verbatim, or routes to clarify, abstain, or handoff. The index is enriched offline by staged linguistic seeding (SLS): a human authors a per-unit world-grounded slot recipe, gpt-4.1-mini renders it into variants, and a light human gate filters them. One methodology is reused across both domains, so inference stays a single retrieval pass with no query-time generation. On held-out query variants from two industrial domains, SLS lifts hybrid R@1 to 0.881/0.930 (+0.27/+0.34), with gains across all five retrievers tested. At the same gpt-4.1-mini generation budget, SLS beats doc2query by +0.20/+0.32, while cross-provenance evaluation provides 
    
[^83]: 用记忆取代训练：面向Text-to-SQL的列表式选择方法

    Replacing Training with Memory: Listwise Selection for Text-to-SQL

    [https://arxiv.org/abs/2609.00834](https://arxiv.org/abs/2609.00834)

    该论文提出MaP-SQL，一种无需微调的Text-to-SQL列表式选择器，通过从训练数据蒸馏的可复用结构化记忆替代学习选择标准，并利用排名聚合缓解位置偏差，从而以更低成本实现候选查询选择。

    

    现代Text-to-SQL系统通常遵循“生成-执行-选择”的流程，即先生成多个候选查询，再从中选出最优的一个。列表式选择通过联合比较多个候选查询，已被广泛采用，但微调列表式选择器的成本高昂。因此，我们提出了一种无需微调的列表式选择器。我们用推理时的策略取代了两个主要的微调目标：（1）将选择标准的学习视为排序学习；（2）缓解位置偏差。首先，我们构建可复用的结构化记忆，而不是将选择行为学习为模型参数。给定一个问题，MaP-SQL会检索从训练数据中蒸馏出的记忆，这些记忆编码了自然语言如何映射到模式元素、SQL操作以及预期输出。这些记忆作为显式的决策标准，用于以列表方式评估候选查询。其次，为了缓解列表式选择器的排序偏差，我们对多个排序结果进行排名聚合（摘要在此处被截断）。

    arXiv:2609.00834v1 Announce Type: cross  Abstract: Modern Text-to-SQL systems often follow generate-execute-select pipelines, generating multiple candidate queries then selecting the best one. Listwise selection, by jointly comparing multiple candidates, has been widely adopted, but fine-tuning listwise selectors is costly. We thus propose a fine-tuning-free listwise selector. We replace two major fine-tuning objectives with inference-time strategies: (1) learning selection criteria as ordering and (2) mitigating positional bias. First, we build reusable structured memories instead of learning selection behavior as model parameters. Given a question, MaP-SQL retrieves memories distilled from training data that encode how natural language maps to schema elements, SQL operations, and expected outputs. These memories serve as explicit decision criteria for evaluating candidates in a listwise manner. Second, to mitigate ordering bias of listwise selectors, we aggregate rankings across mult
    
[^84]: 基于事实效用估计的搜索智能体密集过程监督

    Dense Process Supervision for Search Agents via Fact Utility Estimation

    [https://arxiv.org/abs/2609.00833](https://arxiv.org/abs/2609.00833)

    本文提出一种基于事实效用估计的密集过程监督方法，通过将推理过程建模为离散证据事实的累积，并利用贝叶斯估计将事实效用转化为步骤级奖励，有效解决了搜索智能体强化学习中的信用分配难题。

    

    面向搜索智能体的强化学习（RL）通常依赖于结果奖励。然而，由于中间步骤的价值不明确，这种方法往往难以实现有效的信用分配，很难将中间步骤的贡献从最终结果中分离出来。在本文中，我们提出了一种基于事实效用估计的密集过程监督方法，该方法将推理过程建模为离散证据事实的累积。我们首先从原始观测中提取结构化事实，并将其组织成显式的事实存储库。为支持信用分配，我们随后对语义等价的事实进行聚类，并利用基于组内多次采样的贝叶斯估计来推断每个事实簇的后验效用。最后，我们将估计出的事实效用转换为密集的步骤级奖励，以指导强化学习训练。在七个单跳和多跳问答基准上的实验表明，我们的方法持续优于现有基线方法。

    arXiv:2609.00833v1 Announce Type: new  Abstract: Reinforcement learning (RL) for search agents typically relies on outcome rewards. However, it often fails to achieve effective credit assignment, due to the unclear value of intermediate steps. It is hard to separate their contributions from the final result. In this paper, we propose a dense process supervision method based on fact utility estimation, which models the reasoning process as the accumulation of discrete evidence facts. We first extract structured facts from raw observations and organize them into an explicit fact store. To support credit assignment, we then cluster semantically equivalent facts and infer the posterior utility of each fact cluster using Bayesian estimation over group rollouts. Finally, we convert the estimated fact utilities into dense step-level rewards to guide RL training. Experiments on seven single-hop and multi-hop QA benchmarks show that our method consistently outperforms existing baselines. Ablati
    
[^85]: TWIX：一种用于端到端命名实体识别与关系抽取的两阶段方法

    TWIX: a Two-Stage Approach for End-To-End Named Entity Recognition and Relation Extraction

    [https://arxiv.org/abs/2609.00832](https://arxiv.org/abs/2609.00832)

    本文提出TWIX——一种由三个两阶段模块构成的端到端信息抽取流水线，解决了GutBrainIE基准的全部四个子任务，在所有子任务上均排名第一并大幅超越基线。

    

    科学出版物的指数级增长需要自动信息抽取（IE）系统来支持知识发现。在此背景下，GutBrainIE 基准用于评估肠脑轴领域的命名实体识别（NER）、命名实体识别与消歧（NERD）以及关系抽取（RE）系统。我们提出了信息抽取两阶段工作流（TWIX），这是一个端到端的 IE 流水线，由三个相互关联的模块组成，每个模块都采用两阶段框架来解决 GutBrainIE 的全部四个子任务。在开发集和测试集上的评估表明，我们的方法大幅超越了基线，同时在所有参赛提交方案中，于所有子任务上均排名第一。这些结果表明，所提出的两阶段流水线在实际场景中有效提升了精确率和召回率。

    arXiv:2609.00832v1 Announce Type: new  Abstract: The exponential growth of scientific publications calls for automatic Information Extraction (IE) systems to support knowledge discovery. In this context, the GutBrainIE benchmark evaluates Named Entity Recognition (NER), Named Entity Recognition and Disambiguation (NERD), and Relation Extraction (RE) systems in the gut-brain axis domain. We propose Two-stage Workflow for Information eXtraction (TWIX), an end-to-end IE pipeline featuring three interconnected modules, each leveraging a two-stage framework to solve all four GutBrainIE subtasks. Evaluation on the development and test sets shows that our method substantially outperforms the baseline by a wide margin, while also ranking first among all participant submissions across all subtasks. These results indicate that the proposed two-stage pipeline effectively improves both precision and recall in practical settings.
    
[^86]: 光鲜却未解决：识别长时程工具使用智能体中的后期压力状态

    Polished but Unresolved: Identifying Late-Stage Pressure States in Long-Horizon Tool-Use Agents

    [https://arxiv.org/abs/2609.00823](https://arxiv.org/abs/2609.00823)

    该论文首次识别出长时程工具使用智能体的“后期压力状态”（即倾向于提交看似完整精美但关键约束尚未解决的答案），证明该状态可通过线性探针从隐藏状态中检测、可被激活干预因果地改变，并据此提出PSPR插件以自适应方式缓解压力、改善提交决策。

    

    长时程工具使用智能体不仅需要搜索和规划，还需要决定何时定稿提交。我们研究了后期压力状态：在这种状态下，智能体倾向于提交一个看似完整、精美的最终答案，而关键约束条件仍未解决。我们首先训练了一个线性探针，证明这种压力状态可以从智能体的隐藏状态中被识别出来。随后，我们沿该压力方向进行激活干预，发现移动隐藏状态会同时改变压力评分，以及智能体是继续使用工具还是提前提交。通过受控的上下文操纵，我们进一步观察到，约束清晰度和动作映射能够缓解这种压力。基于这些发现，我们提出了探针感知压力缓解（Probe-Sensed Pressure Relief, PSPR），这是一个插件：在中等压力下施加轻量级的压力缓解方向，在高压力风险下则转向结构化组织。实验在……

    arXiv:2609.00823v1 Announce Type: new  Abstract: Long-horizon tool-use agents need not only to search and plan, but also to decide when to finalize. We study late-stage pressure states, in which an agent is biased toward submitting a final answer that appears complete and polished while key constraints remain unresolved. We first train a linear probe to show that this pressure state is identifiable from the agent's hidden states. Then, we use activation interventions along this pressure direction and find that shifting the hidden states changes both the pressure score and whether the agent continues tool use or submits early. Through controlled context manipulations, we further see that the pressure is mitigated by constraint clarity and action mapping. Based on these findings, we propose Probe-Sensed Pressure Relief (PSPR), a plugin that applies lightweight pressure relief direction under moderate pressure and moves to structured organization under high pressure risk. Experiments on m
    
[^87]: Ctrl-F-Resist：监测极右翼线上活动的公民社会组织的实践、挑战与技术需求

    Ctrl-F-Resist. Practices, Challenges, and Technical Needs of Civil Society Organizations Monitoring the Far-Right Online

    [https://arxiv.org/abs/2609.00808](https://arxiv.org/abs/2609.00808)

    本文通过对12家德国公民社会组织的15名从业者进行定性研究，揭示了这些组织在极右翼在线监测工作中的长期实践、面临的挑战（法律不确定性、平台访问受限、资金不足）以及技术需求，强调它们是数字治理中被忽视的关键利益相关者。

    

    随着极右翼行为者日益利用在线平台传播意识形态并动员支持者，公民社会组织（CSOs）在监测网络上的反民主动态方面发挥着至关重要却未被充分认可的作用。与事实核查员或内容审核员不同，公民社会组织从事长期的、结合具体情境的分析工作，且往往在资源受限和条件不稳定的环境下开展。尽管具有重要的社会作用，公民社会组织在采用或共同开发技术解决方案方面面临重大障碍，包括法律上的不确定性、平台访问权限受限以及长期的资金不足。然而，现有研究和工具开发工作在很大程度上忽视了这些行为者，而更倾向于关注具有制度化背景的利益相关者。本文通过对来自12家德国公民社会组织的15名从事在线监测工作的从业者进行定性研究来填补这一空白，将这些组织定位为数字治理中关键却被忽视的利益相关者。

    arXiv:2609.00808v1 Announce Type: cross  Abstract: As far-right actors increasingly exploit online platforms to disseminate ideology and mobilize supporters, civil society organizations (CSOs) play a vital yet underrecognized role in monitoring antidemocratic dynamics online. Unlike fact-checkers or content moderators, CSOs engage in long-term, contextualized analysis, often in resource-constrained settings and under precarious conditions. Despite their critical societal role, CSOs face significant barriers to adopting or co-developing technical solutions, including legal uncertainty, limited platform access, and chronic underfunding. Existing research and tool development efforts have largely overlooked these actors in favor of more institutionally embedded stakeholders. This paper addresses this gap through a qualitative study with 15 practitioners from 12 Germany-based CSOs engaged in online monitoring, positioning them as key yet overlooked stakeholders in the governance of digital
    
[^88]: TEIDAN：一个多语言多方对话语料库

    TEIDAN: A Multilingual Multiparty Dialogue Corpus

    [https://arxiv.org/abs/2609.00802](https://arxiv.org/abs/2609.00802)

    本文提出了TEIDAN，一个包含日语和英语自发面对面三方对话的多语言多模态语料库，为多方对话的跨语言比较研究提供了宝贵资源。

    

    多方交互是人类交流的核心场景，也是必须参与群体对话的人机交互系统的必要目标。然而，现有语料库往往聚焦于会议、任务导向交互、基于文本的交互或表演情景，支持跨语言比较自发性面对面三人讨论的资源较少。本文提出了TEIDAN，一个目前由日语和英语三方对话组成的多语言多模态语料库。TEIDAN记录了三名参与者组成的小组围绕开放式话题进行讨论的过程，使用了个人领夹式麦克风、麦克风阵列和面向参与者的摄像机，并为两种语言部分提供了基于IPU的转录文本。早期研究曾使用日语部分的子集作为多方对话建模中特定任务的基准；与此不同，本文将TEIDAN作为一个横跨日语和英语的语料库资源加以呈现。

    arXiv:2609.00802v1 Announce Type: new  Abstract: Multi-party interaction is a central setting for human communication and a necessary target for human-agent interaction systems that must participate in group conversation. Yet available corpora often focus on meetings, task-oriented interaction, text-based interaction, or acted scenarios, and fewer resources support cross-linguistic comparison of spontaneous face-to-face triadic discussion. This paper presents TEIDAN, a multilingual multimodal corpus that currently consists of Japanese and English three-party conversations. TEIDAN records groups of three participants discussing open-ended topics with individual pin microphones, a microphone array, and participant-facing cameras, and provides IPU-based transcripts for both language portions. Earlier studies used subsets of the Japanese portion for task-specific benchmarks in multi-party dialogue modeling; in contrast, this paper presents TEIDAN as a corpus resource spanning both Japanese
    
[^89]: SFAD：面向事实性的推测解码

    SFAD: Speculative Factuality-Aware Decoding

    [https://arxiv.org/abs/2609.00796](https://arxiv.org/abs/2609.00796)

    提出SFAD推测解码框架，通过构建细粒度扰动偏好数据集ConFide并利用DPO训练上下文忠实草稿模型，结合认知摩擦机制检测幻觉，在不增加推理开销的情况下显著增强大语言模型的上下文忠实度。

    

    作为大语言模型最关键的挑战之一，上下文忠实度直接决定了其在知识密集型应用中的可靠性。这项任务尤其具有挑战性，因为它需要在事实一致性与生成效率之间取得平衡。对比解码方法需要进行双重前向传播（分别带上下文和不带上下文）来比较模型输出，使推理计算开销翻倍；而后训练对齐则需要大量的强化学习，带来高昂的计算开销。为应对这一挑战，我们提出了SFAD，一种能够在不降低推理性能的前提下增强上下文忠实度的推测解码框架。我们首先构建了ConFide，一个包含细粒度原子级扰动的偏好数据集，用于通过直接偏好优化（DPO）训练一个上下文忠实的草稿模型。在推理过程中，认知摩擦通过量化……来检测潜在的幻觉。（注：原摘要在此处被截断）

    arXiv:2609.00796v1 Announce Type: new  Abstract: As one of the most critical challenges in large language models, contextual faithfulness directly determines their reliability in knowledge-intensive applications. This task is particularly challenging as it requires balancing factual consistency with generation efficiency. Contrastive decoding methods require dual forward passes (with and without context) to compare model outputs, doubling inference computational overhead, while post-training alignment demands extensive reinforcement learning with substantial computational overhead. To address this challenge, we present \textbf{SFAD}, a speculative decoding framework that enhances contextual faithfulness without inference degradation. We first construct \textbf{ConFide}, a preference dataset with fine-grained atomic perturbations, to train a context-faithful draft model via Direct Preference Optimization. During inference, Epistemic Friction detects potential hallucinations by quantifyi
    
[^90]: Instella-MoE 技术报告

    Instella-MoE Technical Report

    [https://arxiv.org/abs/2609.00791](https://arxiv.org/abs/2609.00791)

    Instella-MoE 是一个完全开源、总参数160亿（激活28亿）的混合专家语言模型，完全基于 AMD GPU 从零训练，凭借 Gated MLA 与 FarSkip-Collective 等架构与系统级创新实现了高效训练推理，并在基准测试中超越 OLMo-3-7B 等此前完全开源模型。

    

    在这项工作中，我们介绍了 Instella-MoE，这是一个完全开源的混合专家模型语言模型，总参数量为160亿，每个token激活28亿参数，完全在 AMD Instinct MI300X 和 MI325X GPU 上从零开始训练。Instella-MoE 将稀疏激活的 MoE 设计与架构和系统级创新相结合，包括门控多头潜在注意力（Gated MLA）和 FarSkip-Collective 连接机制，从而实现了高效的大规模训练与推理。该模型通过多阶段流水线开发而成，包括预训练、中期训练、长上下文扩展、结合反馈驱动数据整理的监督微调、直接偏好优化，以及采用多教师在线策略蒸馏的强化学习。Instella-MoE 在标准预训练基准上取得了76.7的平均分，超越了包括 OLMo-3-7B、SmolLM3-3B 和 OLMoE-1B-7B 在内的先前完全开源模型。

    arXiv:2609.00791v1 Announce Type: cross  Abstract: In this work, we introduce Instella-MoE, a fully open Mixture-of-Experts (MoE) language model with 16 billion total parameters and 2.8 billion active parameters per token, trained entirely from scratch on AMD Instinct MI300X and MI325X GPUs. Instella-MoE combines a sparsely activated MoE design with architectural and system-level innovations, including Gated Multi-head Latent Attention (Gated MLA) and FarSkip-Collective connectivity, enabling efficient large-scale training and inference. The model is developed through a multi-stage pipeline comprising pre-training, mid-training, long-context extension, supervised fine-tuning with feedback-driven data curation, direct preference optimization, and reinforcement learning with Multi-Teacher On-Policy Distillation. Instella-MoE achieves an average score of 76.7 across standard pre-training benchmarks, outperforming prior fully open models including OLMo-3-7B, SmolLM3-3B, and OLMoE-1B-7B, wh
    
[^91]: 当特征成为实例：面向无监督特征选择的倒置对比学习

    When Features Become Instances: Inverted Contrastive Learning for Unsupervised Feature Selection

    [https://arxiv.org/abs/2609.00782](https://arxiv.org/abs/2609.00782)

    该论文提出ICLFS框架，通过倒置数据矩阵使特征成为对比学习中的实例，并利用掩码正视图、打乱负视图和InfoNCE目标，将无监督特征选择重新表述为特征层面的表示一致性学习问题。

    

    无监督特征选择旨在在不使用类别标签的情况下，寻找一个紧凑且信息量丰富的特征子集，这使得特征的效用难以定义。因此，现有的无监督特征选择方法依赖于间接的结构性准则，例如相似性保持、局部性、稀疏性、聚类几何结构或重建质量。在本文中，我们转而通过表示一致性来研究无监督特征选择，提出了用于无监督特征选择的倒置对比学习方法，这是一种以特征为单位的对比学习框架，将无监督特征选择重新表述为针对特征而非样本的表示学习问题。ICLFS首先对数据矩阵进行倒置，使每个特征由其样本轮廓向量表示，然后构建多个掩码正视图以及一个打乱顺序的负视图，并在基于InfoNCE的目标函数下学习在这些结构化扰动之间保持一致的投影空间表示。（注：原文摘要在此处截断）

    arXiv:2609.00782v1 Announce Type: new  Abstract: Unsupervised feature selection seeks a compact subset of informative features without access to class labels, making feature utility difficult to define. Existing UFS methods therefore rely on indirect structural criteria, such as similarity preservation, locality, sparsity, cluster geometry, or reconstruction quality. In this paper, we instead study UFS through representation consistency and propose Inverted Contrastive Learning for Unsupervised Feature Selection (ICLFS), a feature-wise contrastive framework that reformulates UFS as a representation learning problem over features rather than samples. ICLFS first inverts the data matrix so that each feature is represented by its sample-profile vector, then constructs multiple masked positive views together with a shuffled negative view, and learns projector-space representations that remain consistent across these structured perturbations under an InfoNCE-based objective. Motivated by re
    
[^92]: 基于知识与安全的拒答机制的统一分析

    A Unified Mechanistic Analysis of Knowledge- and Safety-Based Refusals

    [https://arxiv.org/abs/2609.00760](https://arxiv.org/abs/2609.00760)

    该论文首次通过包含213个对比四元组的新数据集，统一分析了大语言模型中基于知识与基于安全的两种拒答机制，发现两者共享拒答方向但重叠不对称，并将拒答表征为“先承诺后具体化”的过程。

    

    大型语言模型（LLMs）越来越多地被训练以拒绝超出其知识范围的查询（基于知识的拒答，KR）或违反安全策略的查询（基于安全的拒答，SR）。尽管KR和SR在表面上产生相似的响应，但它们大多被孤立地研究，其是否共享潜在机制的问题仍未解决。我们通过对一个包含213个对比四元组的新数据集进行系统性研究来填补这一空白，该数据集联合探测了两种拒答类型。我们发现KR和SR由重叠但可区分的机制所支配。两者共享一个拒答方向，但这种重叠是不对称的：SR信号向KR的迁移比反向更强。类型特定的特化主要出现在模型的上层，其中KR与不确定性和知识相关的表示对齐，而SR与安全和策略相关的表示对齐。因此，我们将拒答表征为一个“先承诺后具体化”（commit-then-specify）的过程：一个共享的初始……

    arXiv:2609.00760v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly trained to decline queries that fall outside their knowledge (knowledge-based refusal, KR) or violate safety policies (safety-based refusal, SR). Although KR and SR result in superficially similar responses, they have largely been studied in isolation, leaving open whether they share an underlying mechanism. We address this gap with a systematic study on a new dataset of 213 contrastive quadruples that jointly probe both refusal types. We find that KR and SR are governed by overlapping yet distinguishable mechanisms. Both share a refusal direction, yet the overlap is asymmetric: SR signals transfer more strongly to KR than the reverse. Type-specific specialization emerges mainly in upper layers, with KR aligning with uncertainty- and knowledge-related representations and SR with safety- and policy-related ones. We thus characterize refusal as a commit-then-specify process: a shared initial me
    
[^93]: 编译而非记忆：一种面向上下文学习的上下文编译架构（CCA）

    Compile, Don't Memorize: A Context Compilation Architecture (CCA) for In-Context Learning

    [https://arxiv.org/abs/2609.00759](https://arxiv.org/abs/2609.00759)

    提出上下文编译架构（CCA），通过将冗长上下文显式编译为带有固定槽位的类型化中间表示（IR），解决了LLM在上下文学习任务中因单次前向传递“阅读并推理”范式而导致的结构性脆弱问题。

    

    大型语言模型（LLM）越来越多地处理上下文学习（ICL）任务，在这类任务中，一段冗长且新颖的上下文为一系列问题定义了规则、知识和输出模式。在针对上下文每一个细节进行评分的基准测试中，即使是强大的开源权重模型也只能通过12-16%的任务：一条被忽略的规则就会导致整个响应失败。我们认为这种脆弱性是结构性的：主流的“阅读并推理”范式要求模型在单次前向传递中完成提取、规划、生成和自我验证。因此，我们提出以下问题：显式的上下文编译能否解决这一问题，它与现有的长上下文策略（要点检索、多智能体自博弈）相比表现如何，以及由此产生的框架收益在任务结构和模型规模上如何保持。我们提出了上下文编译架构（CCA），其核心创新是一种带有固定槽位的类型化中间表示（IR）（rules.{must_do, must_not, conditional}、output_spec……

    arXiv:2609.00759v1 Announce Type: new  Abstract: Large language models (LLMs) increasingly handle in-context learning (ICL) tasks where a long, novel context defines the rules, knowledge, and output schema for a series of questions. On benchmarks that grade against every detail of the context, even strong open-weights models pass only 12-16% of tasks: a single overlooked rule fails the whole response. We argue this brittleness is structural: the dominant "read-and-reason" paradigm asks the model to extract, plan, generate, and self-verify in one forward pass. We therefore ask whether explicit context compilation can fix it, how it compares to existing long-context strategies (gist retrieval, multi-agent self-play), and where the resulting harness benefit holds across task structure and model scale. We propose the Context Compilation Architecture (CCA), whose central novelty is a typed intermediate representation (IR) with fixed slots (rules.{must_do, must_not, conditional}, output_spec
    
[^94]: 联合训练还不够：面向多模态文档理解的条件化跨粒度训练

    Joint Training Is Not Enough: Conditioned Cross-Granularity Training for Multimodal Document Understanding

    [https://arxiv.org/abs/2609.00756](https://arxiv.org/abs/2609.00756)

    该研究发现在多模态文档理解中，单纯的混合联合训练无法让片段级与文档级任务相互促进，并提出仅在训练时将一个粒度的金标准输出注入另一粒度提示词的条件化跨粒度训练，从而实现两个粒度任务的互强化。

    

    互强化效应（MRE）探讨的是：当一个模型同时处理细粒度（片段级）任务和粗粒度（文档级）任务时，二者是否能相互促进。我们在多模态文档理解任务上、于三个语料库（两个收据语料库和一个扫描业务表单语料库）中检验了这一问题，比较了单任务训练、联合训练以及条件化训练三种方式——其中条件化训练仅在训练阶段将某一粒度的金标准输出置于另一粒度的提示词中。我们构建了 Doc-MRE，这是一个标注层，将金标准字段抽取（点级）与四个文档级维度（线级）配对，由事先注册的三位 LLM 评审委员会生成，并通过盲测重新标注加以验证。我们预先固定了一条判定标准：在共享同一训练配方的前提下，只有当某种训练模式在两个粒度上都优于对应的单任务模型时，才认定为实现了互强化。结果表明，以往 MRE 研究所默认采用的混合联合训练方式，在主实验规模下未能在任何语料库上实现互强化：它在 CORD 上低于两个单任务模型，并且在另一数据集上以牺牲一个粒度的性能为代价换取另一个粒度的提升……

    arXiv:2609.00756v1 Announce Type: new  Abstract: The Mutual Reinforcement Effect (MRE) asks whether a fine, span-level and a coarse, document-level task help each other when one model handles both. We test it in multimodal document understanding on three corpora, two of receipts and one of scanned business forms, comparing single-task, joint and conditioned training, which puts one granularity's gold output in the other's prompt during training only. We build Doc-MRE, an annotation layer pairing gold field extraction (point) with four document-level facets (line), from a three-judge LLM committee under a pre-registration, validated by blind re-annotation. One predicate, fixed in advance: at a shared recipe, a regime reinforces if it beats the matched single-task model on both granularities. Mixed joint training, the arrangement prior MRE work assumes, reinforces on no corpus at the main scale: it is below both single-task models on CORD and trades one granularity for the other on the t
    
[^95]: 语言模型如何在上下文与记忆之间进行选择？

    How Do Language Models Choose Between Context and Memory?

    [https://arxiv.org/abs/2609.00753](https://arxiv.org/abs/2609.00753)

    本文通过反事实实验证明了从一致性提示中估计的“权威方向”在语言模型内部因果地决定了模型在上下文信息与参数记忆之间的选择——沿这些方向交换激活坐标可重现30-68%的来源选择偏移。

    

    当上下文信息与存储在模型参数中的知识发生冲突时，可以利用激活方向来解码并引导模型遵循哪种信息来源。然而，沿着某个方向进行引导并不能确立因果关系：即未经修改的模型是否会自然地使用该方向，或者该方向是否可以在不同任务间复用。我们通过在无歧义设置下的反事实实验来检验这些区别。首先，我们从一致性提示中估计“权威方向”，在这类提示中，上下文和参数化知识支持相同的答案。然后，我们在匹配的提示之间交换这些方向上自然出现的坐标，这些匹配提示分别引导模型优先考虑所提供的上下文或其参数化知识。在Qwen、Llama和OLMo模型上，这种干预重现了30-68%由权威性引起的来源选择偏移，而匹配的对照组几乎没有重现任何偏移。为了测试跨任务……

    arXiv:2609.00753v1 Announce Type: cross  Abstract: When contextual information conflicts with the knowledge stored in model parameters, activation directions can be used to decode and steer which source the model follows. However, steering along a direction does not establish causality: whether the unedited model would naturally use that direction or whether the direction is reusable across tasks. We test these distinctions through counterfactual experiments in unambiguous settings. First, we estimate authority directions from agreement prompts, in which the context and parametric knowledge support the same answer. We then interchange naturally occurring coordinates along these directions between matched prompts that direct the model to prioritize either the supplied context or its parametric knowledge. Across Qwen, Llama, and OLMo models, this intervention reproduces 30-68% of the authority-induced shift in source choice, whereas matched controls reproduce almost none. To test cross-t
    
[^96]: 测量Transformer深度中的最优传输

    Measuring Optimal Transport in Transformer Depth

    [https://arxiv.org/abs/2609.00748](https://arxiv.org/abs/2609.00748)

    该研究首次在Pythia模型上量化验证了Transformer逐层移动词元表示“云”的方式符合最优传输：最后一层的移动达到最优传输映射与最优成本（Pythia-410m完全最优），第一层则不符合，而多层块整体以接近最优的成本移动表示云。

    

    Transformer将每个词元的状态从一层传递到另一层，整个被传递的词汇表共同形成一朵随深度移动的“云”。我们探究训练好的网络是否以最优传输的方式移动这朵云：即以最低的成本，并沿着将每个词元与其最优目的地配对的映射进行。我们在Pythia-160m和Pythia-410m上对这两个方面进行了测量，采用相邻层云之间的精确分配、实测的采样下限、对已知最优耦合的校准，并将成本分解为云的整体共同平移和词元特定的移动。在最后一层，两个模型都将词元移动到了最优传输映射所指定的位置，其中Pythia-410m达到了最优成本，Pythia-160m则略高于最优成本。而在第一层则并非如此。在中间层，十个层间转换中仅有两个可以仅凭成本来评判单个层，而多个层组成的块则以接近最优的成本移动这朵云。

    arXiv:2609.00748v1 Announce Type: new  Abstract: A transformer carries each token's state from layer to layer, and the whole vocabulary carried together forms a cloud that moves with depth. We ask whether a trained network moves this cloud the way optimal transport would: at the cheapest cost, and along the map that pairs each token with its optimal destination. We measure both on Pythia-160m and Pythia-410m, with an exact assignment between consecutive layer clouds, a measured sampling floor, calibration on couplings known to be optimal, and a split of the cost into the common shift of the cloud and the token-specific moves. At the last layer, both models move their tokens where the optimal-transport map sends them, at the optimal cost for Pythia-410m and slightly above it for Pythia-160m. At the first layer they do not. In between, single layers can be judged on cost at only two of ten transitions, and blocks of several layers move the cloud at close to the optimal cost. The agreemen
    
[^97]: 大语言模型能否预测研究人员下一步的研究方向？

    Can Large Language Models Forecast What Researchers Study Next?

    [https://arxiv.org/abs/2609.00747](https://arxiv.org/abs/2609.00747)

    该论文提出 IdeaForecastBench 基准，通过让大语言模型基于截止时点的文献生成排序研究想法并与后续实际发表的论文对比，系统评估了大语言模型预测研究者未来研究方向的能力。

    

    大语言模型越来越多地被用于生成研究想法，然而在生成时判断其新颖性或可行性，并不能确定这些模型是否真正预见了后续的研究工作。我们提出了 IdeaForecastBench 来评估研究想法的预测能力。给定某个研究社区截至某一时点的文献，系统生成最多五个经过排序的研究想法，并根据之后发表的论文对其进行评估。该基准涵盖 52 个主题的 624 个滚动评估片段，采用固定的“检索-评判”协议，并分别报告两个评判模型的结果。我们在 GPT-4.1、Qwen2.5-7B/14B 和 Qwen3.5-9B 上比较了五种历史压缩策略，并结合一个经过学习的模式分解预测器（MDF）。在主要的 GPT-4.1-mini 评判模型下，Summary 策略在所有四个骨干模型上的 Hit@5 和 Precision@5 指标均优于 Direct 策略；Qwen2.5 的得分高于 GPT-4.1，而 Qwen3.5 的得分则低于 GPT-4.1。一项不看结果的评估发现，Qwen2.5 能产生更广泛的预测，但（摘要原文在此处截断）

    arXiv:2609.00747v1 Announce Type: new  Abstract: Large language models increasingly generate research ideas, yet judging their novelty or feasibility at generation time does not establish whether they anticipate subsequent work. We introduce IdeaForecastBench to evaluate research idea forecasting. Given a community's literature up to a cutoff, a system produces up to five ranked ideas, which are evaluated against later papers. The benchmark comprises 624 rolling episodes across 52 topics, with a fixed retrieve-then-judge protocol and separately reported results from two judges. We compare five history-compression strategies across GPT-4.1, Qwen2.5-7B/14B, and Qwen3.5-9B, together with a learned Mode-Decomposition Forecaster (MDF). Under the primary GPT-4.1-mini judge, Summary improves on Direct in Hit@5 and Precision@5 across all four backbones. Qwen2.5 scores above GPT-4.1, whereas Qwen3.5 scores below it. An outcome-blind assessment finds that Qwen2.5 produces broader forecasts, but 
    
[^98]: ChatDev 2.0：一个用于开发一切的无代码多智能体平台

    ChatDev 2.0: A No-Code Multi-Agent Platform for Developing Everything

    [https://arxiv.org/abs/2609.00714](https://arxiv.org/abs/2609.00714)

    ChatDev 2.0（DevAll）是一个兼具高表达性与易用性的无代码多智能体平台，通过声明式可执行图抽象与循环感知执行引擎支持异构智能体间的动态循环交互，并提供集成可视化界面，让用户无需编写代码即可构建、运行、监控和检查多智能体系统（包括人在回路步骤）。

    

    基于大语言模型（LLM）的多智能体系统（MAS）在解决复杂任务方面展现出强大潜力，但其开发过程面临两难抉择：代码框架表达能力强但工程开发成本高昂，而无代码构建工具虽然简化了构建过程，却将智能体交互限制在作者预定义的工作流之中。我们提出了 ChatDev 2.0：DevAll（以下简称 DevAll），一个用于构建、执行和检查异构多智能体系统的无代码平台，兼具高表达性与易用性。在表达性方面，DevAll 将声明式可执行图抽象与支持循环感知的执行引擎相结合，使异构智能体以及动态和循环的交互能够在单一框架内被表示和执行。在易用性方面，一个集成的可视化界面让用户能够完全无需编写代码地构建、运行、监控和检查多智能体系统，包括人在回路（human-in-the-loop）的步骤。实验证明 DevAll 能够复现（摘要在此处截断）。

    arXiv:2609.00714v1 Announce Type: new  Abstract: Large language model (LLM)-based multi-agent systems (MAS) have shown strong potential for solving complex tasks, yet their development forces a tradeoff: code frameworks are expressive but engineering-intensive, while no-code builders simplify authoring but constrain agent interactions to author-defined workflows. We present ChatDev 2.0: DevAll (hereafter DevAll), a no-code platform for building, executing, and inspecting heterogeneous MAS that delivers both high expressiveness and ease of use. In terms of expressiveness, DevAll pairs a declarative executable graph abstraction with a cycle-aware execution engine, so that heterogeneous agents and dynamic and cyclic interactions can be represented and executed within a single framework. For ease of use, an integrated visual interface lets users author, run, monitor, and inspect MAS, including human-in-the-loop steps, entirely without writing code. Experiments demonstrate that DevAll repro
    
[^99]: 基于提示条件化场景奖励的可控图像描述生成

    Controllable Image Captioning with Prompt-Conditioned Scene Rewards

    [https://arxiv.org/abs/2609.00709](https://arxiv.org/abs/2609.00709)

    提出FoCUS方法，通过基于场景图对齐组件分数的提示条件化奖励目标并用GRPO优化，让用户能够通过自然语言提示精确控制图像描述的语义重点（如对象、属性、关系或特定区域）。

    

    大型视觉语言模型能够生成流畅的图像描述，但语义控制能力有限：用户无法可靠地指定描述应强调属性、关系还是特定的图像区域。我们提出了FoCUS（Fine-grained Captioning Control Using Scene Rewards，基于场景奖励的细粒度描述控制），这是一种可控图像描述生成方法，允许用户通过自然语言控制提示将图像描述引导至特定的语义重点。其核心思想是基于场景图对齐组件分数构建提示条件化的控制目标：生成的描述会被解析并对齐到场景图组件（如对象、属性和关系），并根据所要求的重点对这些组件进行差异化加权，包括负权重。我们使用GRPO优化该目标，并通过更严格的对象有效性阈值以及基于推理的属性和关系评分验证来进一步提高奖励的可靠性。

    arXiv:2609.00709v1 Announce Type: cross  Abstract: Large Vision-Language Models produce fluent image descriptions but offer limited semantic control: users cannot reliably specify whether captions should emphasize attributes, relations, or particular image regions. We present Fine-grained Captioning Control Using Scene Rewards (FoCUS), a controllable image captioning method that lets users steer captions toward specific semantic emphases through natural-language control prompts. The core idea is a prompt-conditioned control objective based on scene-graph-aligned component scores. Generated captions are parsed and aligned to scene-graph components such as objects, attributes, and relations. These components are differentially weighted, including negative weights, according to the requested emphasis. We optimize this objective with GRPO and further improve its reliability through a stricter object validity threshold and reasoning-based verification for attribute and relation scoring. To 
    
[^100]: 一种产生证书的等式蕴涵级联求解器：SAIR EQT2 第二阶段求解器

    A Certificate-Producing Cascade for Equational Implication: The SAIR EQT2 Stage 2 Solver

    [https://arxiv.org/abs/2609.00706](https://arxiv.org/abs/2609.00706)

    该论文提出一个面向SAIR等式理论挑战的单文件级联求解器，以廉价优先策略判定原群恒等式间的蕴涵关系，肯定情形用产生证明的有序叠加程序、否定情形用多种有限与无限反模型见证，最终输出可被确定性Lean裁判验证的证书。

    

    SAIR等式理论数学蒸馏挑战赛要求求解器判定一个原群恒等式是否蕴涵另一个原群恒等式，并且无论结论如何，都要返回一个能被确定性Lean裁判接受的证书。我们提出了一个按“最廉价优先”级联组织的单文件求解器。其否定分支结合了对结构化代数族的系数测试、有界有限模型搜索、一个显式的中心原群见证以及若干无限载体的见证。其肯定分支是一个产生证明的有序单元叠加程序，采用Knuth-Bendix序、双向解调、索引、记忆化替换以及随时可中断的尺寸加深策略。搜索结果不纳入信任基：成功的推导会被重放为小的Lean项，反模型则由竞赛裁判重新检查。冻结版求解器是一个189,504字节的Python文件，SHA-256为f2392533c9f4c03b...。在本地通过官方j……（原文此处截断）

    arXiv:2609.00706v1 Announce Type: new  Abstract: The SAIR Mathematics Distillation Challenge on Equational Theories asks a solver to classify whether one magma identity implies another and, for either verdict, to return a certificate accepted by a deterministic Lean judge. We present a single-file solver organized as a cheapest-first cascade. Its false branch combines coefficient tests over structured algebra families, bounded finite-model search, an explicit central-groupoid witness, and several infinite-carrier witnesses. Its true branch is a proof-producing ordered unit superposition procedure with Knuth-Bendix ordering, bidirectional demodulation, indexing, memoised substitution, and anytime size deepening. Search results remain outside the trusted base: successful derivations are replayed as small Lean terms, and countermodels are rechecked by the competition judge.   The frozen solver is a 189,504-byte Python file with SHA-256 f2392533c9f4c03b.... In local runs through official j
    
[^101]: 价值超越语言模型：检测写作中的原创贡献

    Value Over Language Model: Detecting Original Contribution in Writing

    [https://arxiv.org/abs/2609.00700](https://arxiv.org/abs/2609.00700)

    提出了一种无需训练、不评分表面文本的新框架“价值超越语言模型”（Value Over Language Model），通过在不同粒度上提取文档内容并用LLM重建文档，来衡量人在语言模型易于生成的内容之上所贡献的原创价值。

    

    大语言模型（LLM）已在各类写作任务中被迅速采用，这推动了检测LLM生成文本工具的发展。然而，这些工具主要衡量的是文档表面文本中有多少是由LLM撰写的，而并非从根本上设计用于衡量文档中的信息内容或思想有多少源自LLM本身，而非由用户在提示词中提供。在本工作中，我们设计了一个框架，用于衡量一个人在语言模型本身能够轻松生成的内容之上所增加的价值。该方法不需要训练或标注数据，也从不为文档的表面文本打分，从而使其免受文体混淆因素的影响。相反，该方法以递增的粒度级别提取文档内容，使用LLM从每个部分表示中重建文档，并将这些重建结果与仅根据任务描述生成的重建结果进行比较。我们将这一框架称为“价值超越语言模型”（Value Over Language Model）。

    arXiv:2609.00700v1 Announce Type: new  Abstract: LLMs have been rapidly adopted across writing tasks, prompting the development of tools for detecting LLM-generated text. Yet, these tools largely measure how much of a document's surface text was written by an LLM and aren't fundamentally designed to measure how much of the information content or ideas originated from the LLM itself rather than being supplied by the user in the prompt. In this work, we design a framework that measures how much value a person adds on top of what a language model could have easily produced by itself. The method requires no training or labeled data and never scores the document's surface text, insulating it from stylistic confounders. Instead, it extracts the document's content at increasing levels of granularity, uses an LLM to reconstruct the document from each partial representation, and compares these reconstructions with those produced from the task description alone. We call this framework Value Over
    
[^102]: SCoNE：面向鲁棒检索增强生成的选择性上下文感知神经元编辑

    SCoNE: Selective Context-aware Neuron Editing for Robust Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.00689](https://arxiv.org/abs/2609.00689)

    SCoNE提出了一种无需训练的模型编辑方法，通过选择性强化兼具高归因分数与高跨输入变异性的上下文感知FFN神经元，显著提升大语言模型在检索增强生成中对检索噪声的鲁棒性，且无需微调、无推理开销。

    

    检索增强生成（RAG）对检索噪声高度敏感：当检索到的文档中混杂着有信息量和无关的内容时，大语言模型容易受到干扰，从而产生幻觉。为了解决这一问题，我们提出了SCoNE（选择性上下文感知神经元编辑），这是一种无需训练的模型编辑方法，通过选择性强化同时具有高归因分数和高跨输入变异性的上下文感知FFN神经元，来提升对检索噪声的鲁棒性。SCoNE仅需少量挖掘样本，无需微调，且不会带来推理时的额外开销。在多个知识密集型问答基准和两个大语言模型骨干上，SCoNE始终优于具有竞争力的基线方法。我们的代码可在 https://github.com/HYU-ARK-Lab/SCoNE 获取。

    arXiv:2609.00689v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) is highly sensitive to retrieval noise: when retrieved documents mix informative and irrelevant context, LLMs are easily distracted, leading to hallucinations. To overcome this, we propose SCoNE (Selective Context-aware Neuron Editing), a training-free model editing approach that improves retrieval noise robustness by selectively strengthening context-aware FFN neurons that are identified by both high attribution and high cross-input variability. SCoNE requires only a small number of mining samples, no fine-tuning, and no inference-time overhead. Across various knowledge-intensive question-answering benchmarks and two LLM backbones, SCoNE consistently outperforms competitive baseline methods. Our code is available at https://github.com/HYU-ARK-Lab/SCoNE.
    
[^103]: 基于图像生成的视觉框架用于新闻立场检测

    Visual Framing for News Stance Detection via Image Generation

    [https://arxiv.org/abs/2609.00685](https://arxiv.org/abs/2609.00685)

    该论文提出VFStance方法，通过图像生成技术将新闻文章中隐含的立场线索转化为视觉框架，使立场信号更加明确显著，有效提升了新闻立场检测性能，并具有超越自动化立场检测的应用潜力。

    

    文章级新闻立场检测旨在识别新闻文章对社会议题所持的观点倾向。尽管立场检测技术不断进步且对构建可信媒体环境具有重要意义，但新闻文章带来了独特的挑战，因为其立场往往是隐含的，通过新闻报道框架微妙地传达，并嵌入在冗长且结构复杂的文本之中。为了应对这些挑战，我们提出了VFStance，该方法利用视觉框架，通过图像生成将隐含的立场线索变得更加明确。在评估实验中，我们证明了VFStance相对于现有方法的有效性，以及视觉框架对其性能的贡献。最后，一项在基于片段的新闻消费场景下开展的受控用户研究（N=200）进一步表明，VFStance能够使立场信号在视觉上更加显著，并凸显了其在自动化立场检测之外的潜在应用价值。

    arXiv:2609.00685v1 Announce Type: cross  Abstract: Article-level news stance detection aims to identify the perspective of news articles toward social issues. Despite advances in stance detection and its importance for trustworthy media environments, news articles pose distinct challenges because their stances are often implicit, subtly conveyed through journalistic framing, and embedded in long, structurally complex texts. To address these challenges, we introduce VFStance, which leverages visual framing to make implicit stance cues more explicit via image generation. In evaluation experiments, we demonstrate the effectiveness of VFStance over existing methods and the contribution of visual framing to its performance. Finally, a controlled user study (N=200) in a snippet-based news consumption setting further demonstrates that VFStance can make stance signals visually salient and highlights its potential use beyond automated stance detection.
    
[^104]: 基于多智能体辩论的创意生成：辩论会抑制多样性吗？

    Creative Generation via Multi-Agent Debate: Does Debate Suppress Diversity?

    [https://arxiv.org/abs/2609.00683](https://arxiv.org/abs/2609.00683)

    该研究发现多智能体辩论（MAD）以收敛为导向的机制会抑制创意生成任务所需的输出多样性，并从理论上证明会话内智能体多样性是实现跨运行多样输出的必要条件，据此提出通过认知视角分配和基于嵌入的同伴选择来维持智能体分歧的 Creative-MAD 框架。

    

    创意生成任务，如叙事写作和科学构思，既要求高质量的输出，也要求在多次独立运行中产生各具特色的回复，以最大化探索空间。多智能体辩论在事实性和推理任务上展现出显著的质量提升，使其成为创意生成的自然候选方案。然而，我们发现其以收敛为导向的设计会主动抑制独立运行之间的输出多样性，从而与创意任务形成内在的权衡。我们从理论上证明，在每场辩论会话内保持智能体之间的多样性，是实现独立运行间多样化输出的必要条件。基于这一发现，我们提出了 Creative-MAD，它引入了两种协同干预措施来维持智能体间的分歧。具体而言，认知视角分配通过将每个智能体锚定于独特且持久的认知模式来对抗身份漂移，而基于嵌入的同伴选择机制……（摘要原文在此处被截断）

    arXiv:2609.00683v1 Announce Type: new  Abstract: Creative generation tasks, such as narrative writing and scientific ideation, demand both high-quality outputs and distinct responses across independent runs to maximize exploration. Multi-Agent Debate (MAD) has shown strong quality gains on factual and reasoning tasks, making it a natural candidate for creative generation. However, we find its convergence-driven design actively suppresses output diversity across independent runs, creating an inherent trade-off with creative tasks. We theoretically show that preserving diversity among agents within each debate session is a necessary condition for achieving diverse outputs across independent runs. Building on this finding, we propose Creative-MAD, which introduces two synergistic interventions to sustain agent divergence. Specifically, Cognitive Lens Assignment counters identity drift by anchoring each agent to a distinct and persistent cognitive mode, while Embedding-based Peer Selection
    
[^105]: SciTrue：在NTCIR SciClaimEval任务中使用前沿与开源语言模型进行可靠的科学论断验证

    SciTrue: Reliable Scientific Claim Validation with Frontier and Open Language Models at the NTCIR SciClaimEval Task

    [https://arxiv.org/abs/2609.00654](https://arxiv.org/abs/2609.00654)

    SciTrue团队通过在统一诚实的逐样本协议下对十一个前沿及开源多模态模型进行基准测试，并结合轻量透明的后处理，在NTCIR SciClaimEval科学论断验证任务的官方盲测排行榜上以明显优势夺得第一。

    

    我们描述了SciTrue团队参与NTCIR-19 SciClaimEval任务两个子任务的情况，该任务要求系统根据论文中的表格和图表来验证科学论断。我们没有调优单一模型，而是在一个诚实、逐样本的统一协议下对十一个前沿及开源多模态模型进行基准测试，并将它们与轻量、透明的后处理相结合。在官方盲测排行榜上，SciTrue在四个证据类别/子任务组合中的三个以明显优势获得第一，并在第四个组合的主要指标上并列第一。三个发现解释了这一结果。第一，强大的指令微调模型已经具备竞争力：Claude Opus 4.8和Gemma-4-31B均超过了最强的公开基线o4-mini，而GPT-5.5和Claude Fable 5在两个子任务中均处于领先地位（在子任务2上达到97.7）。第二，任务的配对结构是最大的杠杆：一个无泄漏的……

    arXiv:2609.00654v1 Announce Type: new  Abstract: We describe the SciTrue team's participation in both subtasks of the NTCIR-19 SciClaimEval task~\cite{sciclaimeval}, which asks systems to verify scientific claims against the tables and figures of a paper. Rather than tuning a single model, we benchmark eleven frontier and open multimodal models under one honest, per-sample protocol and combine them with light, transparent post-processing. On the official, blind test leaderboard (Section~\ref{sec:results}), SciTrue placed first by a clear margin in three of the four evidence-category/subtask combinations, and tied for first on the primary metric in the fourth. Three findings explain the result. First, strong instruction-tuned models are already competitive: Claude Opus~4.8 and Gemma-4-31B each exceed the strongest public baseline (o4-mini), and GPT-5.5 and Claude Fable~5 lead both subtasks (97.7 on Subtask~2). Second, the task's pairing structure is the largest lever: a \emph{leak-free 
    
[^106]: 匹配需要双方协作：基于强化学习协同进化的生成式检索器

    It Takes Two to Match: Co-Evolving Generative Retriever with Reinforcement Learning

    [https://arxiv.org/abs/2609.00638](https://arxiv.org/abs/2609.00638)

    提出 CoGR 框架，通过强化学习让大语言模型协同进化，在查询侧和物品侧同时生成对齐的关键词表示，并直接经倒排索引完成匹配，在兼容现有关键词检索基础设施的同时提升检索效果。

    

    检索是现代搜索与广告系统的第一阶段，它从庞大的物品集合中筛选出候选集，供下游的排序和竞价环节使用。近期的研究越来越多地利用大语言模型（LLM）通过查询扩展、数据合成和检索反馈训练来改进检索。然而，生成式组件通常仅用于查询侧的增强，而最终的匹配仍交由下游检索器完成。我们提出了 CoGR，一种转而训练大语言模型直接在查询侧和物品侧构建检索表示的检索框架。每个生成器产出一组紧凑的关键词集合，通过倒排索引直接进行匹配，从而保持与现有基于关键词的检索基础设施的兼容性。CoGR 采用两阶段训练流程：首先通过监督微调建立一个对齐的关键词空间，随后通过协同进化的强化学习交替优化（原文在此处截断）。

    arXiv:2609.00638v1 Announce Type: cross  Abstract: Retrieval is the first stage of modern search and advertising systems, selecting a candidate set from a large item universe for downstream ranking and auction. Recent work increasingly leverages LLMs to improve retrieval through query expansion, data synthesis, and retrieval-feedback training. However, the generative component is typically used for query-side augmentation, while final matching is still delegated to a downstream retriever. We introduce CoGR, a retrieval framework that instead trains LLMs to directly construct retrieval representations on both query and item sides. Each generator produces a compact set of keywords, which are matched directly through an inverted index, preserving compatibility with existing keyword-based retrieval infrastructure. CoGR uses a two-stage training pipeline. Supervised fine-tuning first establishes an aligned keyword space, after which co-evolving reinforcement learning alternately optimizes t
    
[^107]: ExpArt-KG：通过知识图谱迭代探索生成艺术作品图像描述

    ExpArt-KG: Artwork Image Description Generation through Iterative Exploration of Knowledge Graphs

    [https://arxiv.org/abs/2609.00629](https://arxiv.org/abs/2609.00629)

    本文提出ExpArt-KG框架，通过在答案生成与知识图谱检索之间迭代交替并用正确性判断控制搜索，结合构建的艺术领域知识图谱，使大型视觉语言模型能够生成详细准确的艺术作品图像描述。

    

    大型视觉语言模型（LVLMs）在基于图像的文本生成和视觉问答任务上取得了出色表现。然而，它们仍难以全面且准确地描述与图像中描绘对象相关的实体和概念之间的事实关系。在本工作中，我们提出了一个框架，通过检索增强生成（RAG）高效利用知识图谱中的事实信息，旨在使大型视觉语言模型能够生成详细且准确的图像解释。具体而言，我们的方法在答案生成和知识图谱检索之间交替进行，并利用正确性判断来控制搜索过程，从而高效获取必要且充分的事实信息。我们还构建了面向艺术作品领域的知识图谱，其中图像与实体之间的对应关系明确无歧义。将所提出的方法应用于……

    arXiv:2609.00629v1 Announce Type: new  Abstract: Large Vision-Language Models (LVLMs) achieve strong performance on image-grounded text generation and visual question answering. However, it remains difficult for them to comprehensively and accurately describe the factual relations among the entities and concepts associated with the objects depicted in an image. In this work, we propose a framework that efficiently exploits factual information from a knowledge graph via retrieval-augmented generation (RAG), with the goal of enabling LVLMs to generate detailed and accurate image explanations. Specifically, our method alternates between answer generation and knowledge-graph retrieval, and controls the search using a correctness judgment, thereby acquiring the necessary and sufficient factual information efficiently. We also construct a knowledge graph for the artwork domain (ExpArt-KG), in which the correspondence between images and entities is unambiguous. Applying the proposed method to
    
[^108]: 仅在有把握时才信任引导者：推理时阶段的不确定性感知稀疏对齐

    Trust Your Guide Only When Certain: Uncertainty-Aware Sparse Alignment at Inference Time

    [https://arxiv.org/abs/2609.00624](https://arxiv.org/abs/2609.00624)

    提出TUSA方法，将推理时对齐重构为动态仲裁过程，通过不确定性感知的仲裁机制，仅在监督者置信且token语义重要时才授权干预，从而实现稀疏对齐，避免低置信度干预破坏基础模型的有效推理并降低效用损失。

    

    arXiv:2609.00624v1 公告类型：新论文 摘要：推理时对齐领域的一个主流范式是采用轻量级监督者来引导大型语言模型（LLMs）。通过实证分析，我们发现了该范式中的一个结构性错配：弱监督者在绝大多数token上普遍表现出高熵，而现有的密集干预方法却要求在每个解码步骤都进行监督。这导致了频繁的低置信度干预，可能破坏基础模型本身有效的推理，并造成显著的效用损失。为解决这一问题，我们提出了TUSA（基于信任的不确定性稀疏对齐）。TUSA摒弃了持续性监督的方式，将对齐重新构建为一个动态仲裁过程，引入了一个不确定性感知的仲裁者，仅在满足两个条件时才授权进行干预：监督者具有高置信度，且该token在语义上是重要的。这一机制有效过滤了由不确定性驱动的噪声和冗余监督。大量（实验表明……）

    arXiv:2609.00624v1 Announce Type: new  Abstract: A prominent paradigm in inference-time alignment employs lightweight supervisors to steer Large Language Models (LLMs). Through empirical analysis, we identify a structural mismatch in this paradigm: weak supervisors exhibit pervasive high entropy across the vast majority of tokens, yet prevailing dense intervention approaches mandate supervision at every decoding step. This leads to frequent low-confidence interventions that can disrupt valid base-model reasoning and incur substantial utility costs. To resolve this, we propose TUSA (Trust-based Uncertainty Sparse Alignment). Moving away from continuous oversight, TUSA reframes alignment as a dynamic arbitration process, introducing an uncertainty-aware arbiter that authorizes intervention only when two conditions are met: the supervisor is confident and the token is semantically salient. This mechanism effectively filters out uncertainty-driven noise and redundant supervision. Extensive
    
[^109]: 控制-数据流分离：多智能体大语言模型中的稳定提示词优化

    Control-Data Flow Separation: Stable Prompt Optimization in Multi-Agent LLMs

    [https://arxiv.org/abs/2609.00621](https://arxiv.org/abs/2609.00621)

    该论文提出控制-数据流分离方法，将执行关键协议表示为类型化、经验证的程序对象，使提示词优化器能够改进多智能体大语言模型系统的行为，而不会因提示词修改意外破坏协议导致整个流水线失效。

    

    提示词优化可以改进多智能体大语言模型系统，但被优化的提示词往往承担着两种相互纠缠的角色：一是生成与任务相关的内容，二是指定执行关键协议，例如消息路由、输出格式和终止信号等底层代码所依赖的内容。因此，一次旨在改进内容生成的提示词修改可能会无意中破坏协议，导致整个智能体流水线失效。我们的关键观察是，这两种角色具有不同的表示形式：执行协议通常是结构化的，而任务相关内容通常以非结构化语言表达。基于此，我们提出了控制-数据流分离方法，即将执行关键的控制表示为经过类型化和验证的程序对象，而与任务相关的语言则保持为可优化的数据流，用于智能体之间的通信。这种设计使优化器能够在不……的情况下改进多智能体行为（原文摘要在此处截断）。

    arXiv:2609.00621v1 Announce Type: new  Abstract: Prompt optimization can improve multi-agent LLM systems, but the prompts being optimized often serve two entangled roles: generating task-relevant content and specifying execution-critical protocols, such as message routing, output formatting, and termination signals, on which the underlying code relies. As a result, a prompt edit intended to improve content generation can inadvertently corrupt the protocol and cause the entire agent pipeline to fail. Our key observation is that these two roles have different representations: execution protocols are typically structured, while task-relevant content is usually expressed in unstructured language. Based on this, we propose control-data flow separation, where execution-critical control is represented as typed, validated program objects, while task-relevant language remains the optimizable data flow for agent communication. This design allows optimizers to improve multi-agent behavior without
    
[^110]: 使用角色向量研究LLM用户模拟器中的助手偏见

    Investigating Assistant Bias in LLM User Simulators Using a Role Vector

    [https://arxiv.org/abs/2609.00608](https://arxiv.org/abs/2609.00608)

    该研究通过对比LLM在同一对话中对用户与助手视角的激活差异提取出“用户角色向量”，证明该向量可被识别并能引发真实的用户行为，为缓解用户模拟器中的助手偏见提供了新方法，但过度引导可能导致用户行为被夸大。

    

    基于大语言模型（LLM）的用户模拟器正被越来越多地用于大规模评估自主智能体，以替代昂贵的人工评估。尽管前景可观，这些模拟器表现出“助手偏见”（assistant bias），即倾向于合作并追求任务目标。它们很少再现真实用户所表现出的挫败感或脱离感，从而损害了评估的有效性。先前的研究指出，这种偏见在模型训练过程中就已形成，而角色扮演提示无法将其覆盖。我们从模型激活（activations）的角度分析这一偏见，通过对比模型在同一对话中对用户视角与助手视角的表征差异，提取出一个用户角色向量。我们观察到两个发现：(i) 用户方向在激活中可被识别，能够引发类似用户的行为，并捕捉到与助手特质截然不同的特征；(ii) 尽管用户角色激活与模拟真实性相关，且通过引导（steering）可以增强这种真实性，但它可能会夸大用户行为并过度……（摘要原文在此处截断）

    arXiv:2609.00608v1 Announce Type: new  Abstract: LLM-based user simulators are increasingly used to evaluate autonomous agents at scale, in place of costly human evaluations. Despite this promise, these simulators exhibit "assistant bias," a tendency to cooperate and pursue task goals. They rarely reproduce the frustration or disengagement that real users exhibit, compromising evaluation validity. Prior work outlines that this bias is baked in during model training, which role-playing prompts fail to override. We analyze this bias from model activations, extracting a user role vector by contrasting how the model represents user versus assistant perspectives on the same dialogue. We observe two findings: (i) the user direction is identifiable in activations, elicits user-like behaviors, and captures characteristics distinct from assistant traits; and (ii) although user-role activation associates with simulation realism and steering strengthens it, it can exaggerate user behaviors and ov
    
[^111]: 坦白你所知：大语言模型机器遗忘中遗忘集与模型知识的不对齐问题

    Confess What You Know: Forget-Set Misalignment with Model Knowledge in LLM Unlearning

    [https://arxiv.org/abs/2609.00605](https://arxiv.org/abs/2609.00605)

    提出数据无关的CONFS框架，通过引出模型自身记忆的知识来构建与模型对齐的遗忘集，解决了大语言模型机器遗忘中遗忘集与模型实际记忆内容不对齐所导致的信息泄露或效用下降问题。

    

    大语言模型（LLM）的机器遗忘通常假设预定义的遗忘集与模型实际记忆的内容相匹配，但在原始训练数据不可访问的现实隐私场景中，这一假设经常失效。我们将这种差距称为“遗忘集不对齐”，并识别出两种情况：在“遗忘不足”中，遗忘集遗漏了模型已记忆的信息，导致信息泄露持续存在；在“知识外遗忘”中，算法被驱动去“遗忘”模型从未学过的知识，从而扰动参数并降低模型效用。通过梯度层面的分析，我们证明这些行为源于不对齐的遗忘目标，而非特定的优化方法选择。随后，我们提出了CONfession-to-Forget-Set（CONFS），这是一个数据无关的框架，通过引出并形式化模型自身已记忆的知识来构建与模型对齐的遗忘集。在合成数据、多模态和真实世界基准测试中，CONFS均接近金标准性能。

    arXiv:2609.00605v1 Announce Type: cross  Abstract: Machine unlearning for large language models (LLMs) often assumes that a pre-defined forget set matches what the model has memorized, but this frequently breaks in realistic privacy settings where the original training data is inaccessible. We term this gap forget-set misalignment and identify two cases. In Under Unlearning, the forget set omits memorized information and leakage persists. In Out-of-Knowledge Unlearning, the algorithm is driven to "forget" knowledge the model never learned, perturbing parameters and degrading utility. Using gradient-level analysis, we show these behaviors arise from misaligned unlearning targets rather than specific optimization choices. We then propose CONfession-to-Forget-Set (CONFS), a data-blind framework that constructs model-aligned forget sets by eliciting and formalizing the model's memorized knowledge. Across synthetic, multimodal, and real-world benchmarks, CONFS approaches Gold-standard perfo
    
[^112]: 见好就收：用于机器翻译重排序中高效候选生成的Quit方法

    Quit While You're Ahead: Quit for Efficient Candidate Generation in Machine Translation Reranking

    [https://arxiv.org/abs/2609.00588](https://arxiv.org/abs/2609.00588)

    提出Quit方法，通过不确定性量化的早停策略对机器翻译的整个候选生成—重排序流程进行增量式生成与重排序，在最高候选质量稳定时提前终止，从而在保持翻译质量的同时显著降低推理延迟。

    

    重排序方法，如最小贝叶斯风险（MBR）解码和质量估计（QE）重排序，被广泛应用于现代神经机器翻译（NMT）中，用于从一组候选假设中选出最终输出。然而，这些性能提升是以高推理延迟为代价的。现有的加速方法仅针对MBR解码且只减少重排序计算，既未解决QE重排序的问题，也基本未触及候选生成——而后者可能是更大的计算瓶颈。在本工作中，我们提出了Quit（基于不确定性量化的增量终止），这是一种针对整个“生成—重排序”流程的新型早停策略。Quit将候选生成视为不确定性下的序列决策过程，增量式地生成并重排序候选译文，当候选集中最高的估计质量趋于稳定时即停止生成。在三个NMT模型、19个语言对上的全面实验表明……

    arXiv:2609.00588v1 Announce Type: new  Abstract: Reranking methods, such as Minimum Bayes Risk (MBR) decoding and Quality Estimation (QE) reranking, are widely used in modern neural machine translation (NMT) to select an output from a set of candidate hypotheses. However, the performance gains come at the cost of high inference latency. Existing acceleration methods target MBR decoding and reduce only reranking computation, leaving QE reranking unaddressed and candidate generation---which can be the larger computational bottleneck---largely untouched. In this work, we propose Quit (Quantifying Uncertainty for Incremental Termination), a novel early-stopping strategy for the entire generation--reranking pipeline. Viewing candidate generation as a sequential decision under uncertainty, Quit incrementally generates and reranks candidates, stopping when the highest estimated quality in the candidate set stabilizes. Comprehensive experiments on three NMT models across 19 language pairs show
    
[^113]: Enoki：高效的多层级幻觉检测

    Enoki: Efficient Multi-Level Hallucination Detection

    [https://arxiv.org/abs/2609.00581](https://arxiv.org/abs/2609.00581)

    Enoki提出了一种基于开放信息抽取的多层级幻觉检测框架，通过抽取文本锚定的关系事实并进行验证，无需额外的声明-片段对齐即可同时实现声明级验证和片段级定位，并支持LLM、编码器和规则三种抽取方式以平衡准确性与推理成本。

    

    在高风险场景中部署大语言模型（LLM）时，确保事实性仍然是一项关键挑战。现有的幻觉检测器通常仅在单一层级上运作：声明级方法提供可解释的事实单元，而片段级方法则定位不受支持的文本。弥合这两种视角的代价高昂，因为依赖大量LLM的流水线需要多次分解和验证调用，而模块化系统则需要额外的声明到片段的对齐。我们提出了Enoki，一个用于多层级幻觉检测的开放信息抽取框架。Enoki抽取以文本为锚点的关系事实，将其与证据进行核对验证，并将不受支持的事实投影回幻觉片段。这种共享表示使得声明级验证和片段级定位无需单独的对齐步骤即可实现。Enoki支持基于LLM的、基于编码器的以及基于规则的抽取方式，通过统一接口在准确性和推理成本之间取得平衡。实验……（原文摘要在此处被截断）

    arXiv:2609.00581v1 Announce Type: new  Abstract: Ensuring factuality remains a critical challenge for deploying LLMs in high-stakes settings. Existing hallucination detectors usually operate at a single level: claim-level methods provide interpretable factual units, while span-level methods localize unsupported text. Bridging these views is costly, as LLM-heavy pipelines require multiple decomposition and verification calls, and modular systems need additional claim-to-span alignment. We propose Enoki, an Open Information Extraction framework for multi-level hallucination detection. Enoki extracts text-anchored relational facts, verifies them against evidence, and projects unsupported facts back to hallucinated spans. This shared representation enables claim-level verification and span-level localization without requiring separate alignment. Enoki supports LLM-based, encoder-based, and rule-based extraction regimes, balancing accuracy and inference cost through a common interface. Expe
    
[^114]: 基于大语言模型与程序设计语言语义的程序退出码预测

    Predicting Program Exit Code with LLMs and Programming Language Semantics

    [https://arxiv.org/abs/2609.00579](https://arxiv.org/abs/2609.00579)

    该论文提出了程序可执行性预测这一新任务，并构建了由有效程序系统性生成无效变换的数据集，以研究大语言模型在判断程序有效性及其违反的形式化语义规则时，究竟是依赖预训练先验知识还是给定的程序语义。

    

    大语言模型（LLM）在代码生成和翻译等多种软件工程任务中已展现出卓越能力。然而，其性能的一个关键局限可能在于对程序设计语言语义的理解（或缺乏理解）。即使给出了显式语义，LLM究竟是应用这些规则，还是依赖预训练期间学到的先验知识，目前仍不清楚。我们通过一项新颖任务——程序可执行性预测来研究LLM是依赖先验知识还是给定语义。该任务要求模型在给定程序语法和操作语义的情况下，预测程序在语义上是有效的还是无效的（如果是无效的，还需指出其违反了哪条形式化规则）。由于PrEx需要有效和无效的程序，我们构建了一个数据集，其中包含从有效程序系统性生成的无效变换。我们在两种语义形式体系和两种语义偏移下，跨Human-（评估开源代码LLM）。

    arXiv:2609.00579v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown proficiency in various software engineering tasks, such as code generation and translation. However, a key limitation in their performance may be their (lack of) understanding of programming-language semantics. Even when explicit semantics are given, it remains unclear whether LLMs apply those rules or lean on priors learned during pre-training instead. We study if LLMs lean on priors or given semantics with a novel task--Program Executability Prediction (PrEx)--that asks models to predict whether a program is semantically valid or invalid (and, if invalid, which formal rule it violates) given the program's syntax and operational semantics. Because PrEx requires both valid and invalid programs, we build a dataset with systematically generated invalid transformations derived from valid programs. We evaluate open-source coding LLMs under two semantic formalisms and two semantic shifts across Human-
    
[^115]: 无需对齐的一致性：与随机选择无法区分的条目敏感语言模型

    Consistency Without Alignment: Item-Sensitive Language Models Indistinguishable From Random

    [https://arxiv.org/abs/2609.00576](https://arxiv.org/abs/2609.00576)

    本研究通过可闭式计算基准的强制选择信号任务证明，语言模型的条目敏感性只是任务能力的必要而非充分条件——尽管全部21个“模型×规则”组合都表现出条目敏感性，但其中8个与随机选择在统计上无法区分，5个甚至比随机表现更差。

    

    条目敏感性（item-sensitivity），即模型的选择是否取决于特定输入而非其自身的输出先验，被广泛报道为任务能力的证据。我们利用一个从桌游《Deception: Murder in Hong Kong》抽象出的强制选择信号传递任务，证明这一证据是必要但不充分的。在该环境中，用于评判一个协调者的参照基准（最大化拟合策略、最大化后验策略和均匀随机选择）均可以闭式形式计算。在七个语言模型、两个模型家族、一项后训练消融实验以及三种独立评分规则下，21个“模型×规则”组合单元中的每一个都可靠地表现出条目敏感性。然而，这21个单元中有8个在统计上无法与一个忽略具体条目、随机选择的参与者区分开，另有5个在描述目标方面的得分低于随机水平。条目敏感性与距随机的距离之间的相关性仅在……（原文摘要至此截断）

    arXiv:2609.00576v1 Announce Type: new  Abstract: Item-sensitivity, defined as whether a model's choice depends on the specific input rather than on its own output prior, is widely reported as evidence of task competence. We show this evidence is necessary but not sufficient using a forced-choice signalling task abstracted from the board game Deception: Murder in Hong Kong. In this environment, the reference points against which a coordinate should be judged (a fit-maximising strategy, a posterior-maximising strategy, and uniform random selection) are all computable in closed form. Across seven language models, two model families, a post-training ablation, and three independent scoring rules, every one of 21 model-by-rule cells is reliably item-sensitive. Yet 8 of those 21 cells are not statistically distinguishable from a chooser that ignores the item and selects at random, and 5 score worse than random at describing the target. Item-sensitivity and distance from random correlate at on
    
[^116]: 对齐但被拉平：分析大语言模型中文化对齐与多样性之间的权衡

    Aligned but Flattened: Analyzing the Trade-off between Cultural Alignment and Diversity in LLMs

    [https://arxiv.org/abs/2609.00565](https://arxiv.org/abs/2609.00565)

    该研究提出了一个同时形式化文化对齐与文化多样性的协同评估框架，并通过对六个主流大语言模型的基准测试揭示了追求文化对齐会以牺牲多样性为代价、导致“文化拉平”这一关键权衡。

    

    文化微调已成为构建具有文化感知能力的大语言模型（LLM）的事实标准范式，然而，现有的仅以对齐分数为优化目标的方法，通过系统性地掩盖固有的文化多样性，提供了一幅不完整的文化忠实性图景。这种单一维度的评估视角引发了一个根本性问题：模型究竟是真正感知到了不同的文化细微差别，还是仅仅记住了主流文化价值观？为解决这一问题，我们提出了一个协同评估框架，将文化对齐与文化多样性共同进行形式化。通过对六个主流大语言模型在世界价值观调查上的广泛基准测试，该框架揭示了一个系统性的关键权衡：对文化对齐的追求总是以多样性的急剧损失为代价，从而导致严重的“文化拉平”现象。在探究这一行为转变的过程中，我们证明这些表面的对齐收益源自……（摘要原文在此处截断）

    arXiv:2609.00565v1 Announce Type: cross  Abstract: Cultural fine-tuning has become the de facto paradigm for building culture-aware large language models (LLMs), yet existing optimization exclusively for alignment scores provides an incomplete portrait of cultural fidelity by systematically obscuring inherent cultural diversity. This unidimensional evaluation lens prompts a fundamental question: do models genuinely perceive distinct cultural nuances, or do they merely memorize dominant cultural values? To address this, we propose a synergistic evaluation framework that jointly formalizes cultural alignment and diversity. Through extensive benchmarking of six mainstream LLMs on the World Values Survey, this framework uncovers a systematic and critical trade-off: the pursuit of cultural alignment consistently incurs an acute expense of diversity, leading to severe "cultural flattening." Investigating this behavioral shift, we demonstrate that these superficial alignment gains stem from m
    
[^117]: EM^2Mem：面向大型语言模型的事件中心多模态记忆

    EM^2Mem: Event-Centric Multimodal Memory for Large Language Models

    [https://arxiv.org/abs/2609.00551](https://arxiv.org/abs/2609.00551)

    该论文提出EM^2Mem，一种以事件为中心的多模态记忆框架，通过在记忆构建阶段将多模态记录、时间上下文、图谱关系与溯源信息绑定到事件锚点，形成“可直接用于生成”的记忆单元，免去了推理时重建跨模态对齐的负担，并在三个长视频问答基准上将平均准确率较最强记忆基线提升2.0至3.7个百分点。

    

    多模态记忆为长视频问答提供了一种可扩展的接口，但现有方法通常将字幕、视频帧、转录文本、摘要或图谱事实作为孤立的片段进行检索。尽管这些片段可被搜索，却并不“可直接用于生成”：语言模型必须在推理阶段、在上下文受限且归因困难的情况下重建跨模态和时间上的对齐关系。我们提出了EM^2Mem，一个以事件为中心的多模态记忆框架，它在记忆构建阶段将异构证据绑定到事件锚点上。每个以事件为索引的记忆单元对齐多模态记录、时间上下文、图谱关联关系、语义事实以及来源溯源信息，从而能够基于多模态事件（而非特定模态的孤立片段）进行紧凑的证据读取。在三个长视频问答基准上，EM^2Mem 相比最强的记忆基线分别将平均准确率提升2.0、2.4和3.7个百分点，并在严格的事件级评估上……（原文摘要在此处截断）

    arXiv:2609.00551v1 Announce Type: cross  Abstract: Multimodal memory offers a scalable interface for long-video question answering, but existing methods often retrieve captions, frames, transcripts, summaries, or graph facts as isolated fragments. Although searchable, such fragments are not generation-ready: language models must reconstruct cross-modal and temporal alignments at inference time, when context is limited and attribution is difficult. We propose EM^2Mem, an event-centric multimodal memory framework that binds heterogeneous evidence to event anchors during memory construction. Each event-indexed memory cell aligns multimodal records, temporal context, graph-linked relations, semantic facts, and provenance, enabling compact evidence readout over grounded multimodal events rather than modality-specific fragments. Across three long-video QA benchmarks, EM^2Mem improves average accuracy over the strongest memory baseline by 2.0, 2.4, and 3.7 points, improves strict event-level 
    
[^118]: 相同语义，不同结果：多模态大语言模型在知识冲突下的模态鲁棒性研究

    Same Semantics, Different Outcome: On the Modality Robustness of Multimodal LLMs under Knowledge Conflict

    [https://arxiv.org/abs/2609.00550](https://arxiv.org/abs/2609.00550)

    该研究发现多模态大语言模型在知识冲突下缺乏模态鲁棒性：模型更容易被图像形式的矛盾证据所说服，且文本与图像同时呈现时偏好模态具有任意性，这一不稳定性会降低多模态RAG性能并带来对抗攻击风险。

    

    多模态大语言模型（MLLMs）越来越多地被提供以异构形式呈现的上下文证据：可以是文本段落的形式，可以是同一段落渲染成的图像形式，或者两者同时提供。然而，这些表面形式在处理上是否保持一致仍不清楚，尤其是当证据与模型的参数化知识相冲突时。我们在13个MLLM和两个数据集上研究了知识冲突下的模态鲁棒性，发现它们远非鲁棒。(1) 与普遍认知相反，相比于文本形式，模型更容易接受以图像形式呈现的与参数化知识相矛盾的上下文；(2) 当矛盾的文本和图像同时呈现时，模型偏好的模态基本上是任意的，会随输入顺序、模型和数据集而变化。我们进一步证明这种不稳定性具有实际影响：它会降低多模态RAG的性能，并且可以被对抗性攻击所利用。为缓解这种...

    arXiv:2609.00550v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) are increasingly provided with contextual evidence in heterogeneous forms: as a text passage, as a rendered image of the same passage, or as both together. However, it remains unclear how consistently these surface forms are processed, especially when the evidence conflicts with the model's parametric knowledge. We study modality robustness under knowledge conflict across 13 MLLMs and two datasets, and find them far from robust. (1) Contrary to common belief, models favor a context that contradicts parametric knowledge more readily in image form than in text form; (2) when a contradicting text and image are presented together, the preferred modality is essentially arbitrary, varying with input order, model, and dataset. We further demonstrate that this instability has practical consequences: it degrades performance in multimodal RAG and can be exploited by adversarial attacks. To alleviate this br
    
[^119]: 技能遵循：评估检索增强型LLM智能体中的实际技能使用

    Skill Following: Evaluating Actual Skill Use in Retrieval-Enabled LLM Agents

    [https://arxiv.org/abs/2609.00549](https://arxiv.org/abs/2609.00549)

    该论文提出“技能遵循”（SF）概念及度量指标“检索调用实际使用效应”（RAE），通过在同一任务上对比启用与禁用技能的执行结果来消除选择偏差，并揭示了一个评估悖论：许多模型整体上看似受益于技能检索，但在实际检索了技能的任务上性能反而下降。

    

    大型语言模型（LLM）智能体日益依赖外部技能，然而标准评估方法往往无法揭示检索这些技能是否真正有帮助。汇总指标通常将检索了技能的任务与未检索技能的任务进行比较，这引入了严重的选择偏差，且无法分离出技能使用的真实效果。为了衡量这种实际使用能力——我们将其形式化为技能遵循——我们提出了检索调用实际使用效应（RAE）。RAE在智能体主动检索了技能的任务上，计算启用技能与禁用技能的匹配执行在同一任务上的结果差异。通过对17个LLM在编程和数学领域的评估，我们发现了一个显著的评估悖论：模型经常表现出正的总体检索提升，但RAE却为负。在MBPP+上，多个表面上从整体上受益的模型，实际上恰恰在它们检索了技能的那些任务上损害了自身性能。

    arXiv:2609.00549v1 Announce Type: new  Abstract: Large Language Model (LLM) agents increasingly rely on external skills, yet standard evaluations obscure whether retrieving these skills actually helps. Aggregate metrics often compare retrieved versus non-retrieved tasks, introducing severe selection bias and failing to isolate the true effect of skill use. To measure this actual-use capability-which we formalize as Skill Following (SF)-we introduce the Retrieval-Invoked Actual-Use Effect (RAE). RAE computes the same-task outcome difference between matched skill-enabled and skill-disabled executions, conditioned exclusively on tasks where the agent actively retrieved a skill. Evaluating 17 LLMs across coding and mathematical domains, we uncover a stark evaluation paradox: models frequently show positive aggregate retrieval lift but negative RAE. On MBPP+, multiple models that appear to benefit system-wide actually harm their own performance on the exact tasks where retrieval occurred. T
    
[^120]: 语际假说：大语言模型通过潜在的任务无关特征空间进行翻译

    The Interlingua Hypothesis: LLMs Translate via a Latent Task-agnostic Feature Space

    [https://arxiv.org/abs/2609.00515](https://arxiv.org/abs/2609.00515)

    该论文提出“语际假说”，认为大语言模型通过将源语句编码进任务无关的潜在多语言特征空间、再从中解码生成目标语句的方式完成翻译，并从BLEU分数可预测性、组件因果影响和微调三个方面提供了支持证据。

    

    大语言模型（LLMs）近期在机器翻译任务上的表现已超越强大的有监督基线模型。这引发了一个问题：大语言模型在不同语言之间执行机器翻译的背后机制是什么？受近期可解释性研究发现的启发——即大语言模型使用大规模多语言潜在特征表示来进行语言建模——我们提出了语际假说。该假说认为，语言模型的翻译方式是：先将源语句读入一个潜在特征空间，再从该潜在特征空间读取信息来生成目标语句。我们展示了支持这一假说的三条证据：（1）不同语言对之间的BLEU分数差异在很大程度上可以由各语言特定的能力预测，而无需引入语言对特定的交互项；（2）许多模型组件在单语任务和翻译任务中都具有因果影响力；（3）微调

    arXiv:2609.00515v1 Announce Type: cross  Abstract: Large language models (LLMs) have recently demonstrated improved machine translation performance over strong supervised baselines. This raises questions as to what mechanisms underlie how LLMs perform machine translation between languages. Motivated by recent interpretability findings--namely, that LLMs use massively multilingual latent feature representations to perform language modeling--we propose the interlingua hypothesis. The hypothesis holds that language models translate by reading a source sentence into a latent feature space, and generate a target sentence by reading from the latent feature space. We show three lines of evidence in support of this hypothesis: (1) variance in BLEU across language pairs is largely predictable from language-specific competences with no language pair-specific interaction terms; (2) many model components are causally influential in both monolingual tasks and translation tasks; and (3) fine-tuning 
    
[^121]: 超越词元位置：扩散语言模型中跨去噪步骤的安全对齐

    Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models

    [https://arxiv.org/abs/2609.00495](https://arxiv.org/abs/2609.00495)

    该研究发现扩散语言模型的拒绝信号集中在早期去噪步骤和回复起始位置，并提出了一种无需训练的RAEC解码方法，通过在早期步骤提交持续的拒绝信号来提升模型安全性。

    

    扩散大语言模型（dLLMs）通过迭代去噪而非从左到右的解码方式生成文本。这种生成范式引入了两个可能影响安全对齐的维度：词元在去噪过程中何时生成，以及它们在回复中出现在什么位置。在本文中，我们通过追踪整个去噪过程中的中间词元分布和承诺决策，测量了dLLM在有害提示下的安全行为。我们的分析表明，拒绝信号集中在早期去噪步骤和回复的起始位置，且早期提交的词元能够强烈影响最终的安全结果。我们的测量进一步表明，去噪步骤以及拒绝词元承诺的持续性对于理解dLLM的安全性至关重要。基于这些发现，我们提出了拒绝感知早期提交方法（Refusal-Aware Early Commitment, RAEC），这是一种简单的无需训练的解码方法，可以从早期步骤提交持续的拒绝信号。

    arXiv:2609.00495v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) generate text through iterative denoising rather than left-to-right decoding. This generation paradigm introduces two axes that can influence safety alignment: when tokens are generated during denoising and where they appear in the response. In this paper, we measure dLLM safety behavior under harmful prompts by tracing intermediate token distributions and commitment decisions throughout denoising. Our analysis shows that refusal signals are concentrated in early denoising steps and leading response positions, and the tokens committed early can strongly shape the final safety outcome. Our measurements further show that the denoising step and persistence of refusal-token commitment are important for understanding dLLM safety. Based on these findings, we propose Refusal-Aware Early Commitment (RAEC), a simple training-free decoding method that commits persistent refusal signals from early steps. Ex
    
[^122]: 基于策略性标注的人类锚定事实性评估

    Human-Anchored Factuality Evaluation with Strategic Annotation

    [https://arxiv.org/abs/2609.00494](https://arxiv.org/abs/2609.00494)

    该论文提出了一个基于失败空间分析（FSA）的事实性评估标注策略设计流水线，通过在有限预算下策略性地选择人类标注样本，将LLM评判器预测与人类标注相结合，从而校正系统性偏差并获得统计有效的事实性评估估计。

    

    基于大语言模型（LLM）的事实性评判器提供了可扩展的评估信号，但其指标相对于人类判断往往存在系统性偏差。我们研究了在有限标注预算下的人类锚定事实性评估，即将评判器在整个数据集上的预测与人类在一小部分选择性抽样子集上的标注相结合，以获得统计上有效的估计。这种方法的效率关键取决于哪些样本被分配给人类标注：在事实性评估中，评判器与人类之间的不一致并非仅由低置信度驱动，还源于结构化的失败模式，例如证据不完整、时间错配、不可验证的声明以及评分标准不一致。为了利用这种结构，我们提出了一个针对事实性评估的标注策略设计流水线，该流水线使用失败空间分析（FSA）来推导多样的预测信号，用于建模人类与评判器之间的不一致。在一个基于参考的内部事实性评估……

    arXiv:2609.00494v1 Announce Type: new  Abstract: LLM-based factuality judges provide scalable evaluation signals, but their metrics are often systematically biased relative to human judgments. We study human-anchored factuality evaluation under limited annotation budgets, where judge predictions on the full dataset are combined with human labels on a small selectively sampled subset to obtain statistically valid estimates. The efficiency of this approach depends critically on which examples receive human annotation: in factuality evaluation, judge-human misalignment is not driven solely by low confidence, but also by structured failure modes such as incomplete evidence, temporal mismatch, unverifiable claims, and rubric misalignment. To exploit this structure, we introduce a factuality-specific annotation policy design pipeline that uses failure-space analysis (FSA) to derive diverse predictive signals for modeling human-judge misalignment. On an internal reference-based factuality eva
    
[^123]: 差分隐私语言模型中的隐私-幻觉权衡

    The Privacy-Hallucination Tradeoff in Differentially Private Language Models

    [https://arxiv.org/abs/2609.00492](https://arxiv.org/abs/2609.00492)

    本文首次揭示并系统研究了差分隐私语言模型中隐私保护与事实准确性之间的权衡：DP训练会导致模型产生更多幻觉（因为DP机制使输出分布平坦化），而提高事实信息在训练数据中的出现频率可有效降低幻觉风险。

    

    在医疗保健等高风险领域，隐私和事实准确性都至关重要。令人担忧的是，我们发现并研究了差分隐私（DP）语言模型中存在的隐私-幻觉权衡问题。首先，我们通过实证表明，采用DP进行预训练或微调的模型往往比非DP的对应模型产生更多幻觉，且随着隐私预算收紧，幻觉的严重程度会增加。其次，我们研究了驱动这种权衡的模型特性，证明DP机制会使输出分布趋于平坦，可能将概率质量重新分配到事实上错误的替代选项上。第三，通过在训练数据中控制事实出现频率的实验，我们刻画了信息频率如何降低DP模型中的幻觉风险。总体而言，我们的研究结果强调了需要更精细的隐私保护干预措施，以便在不损害事实准确性的前提下提供严格的隐私保证。

    arXiv:2609.00492v1 Announce Type: new  Abstract: Both privacy and factual accuracy are paramount in high-stakes domains like healthcare. Concerningly, we uncover and investigate a privacy-hallucination tradeoff in differentially private (DP) language models. First, we empirically show that models pre-trained or fine-tuned with DP tend to produce more hallucinations than non-DP counterparts, with increased severity as the privacy budget grows stricter. Second, we investigate model properties driving this tradeoff, demonstrating that DP mechanisms flatten output distributions, potentially redistributing probability mass toward factually incorrect alternatives. Third, through experiments where we control fact frequency in training data, we characterize how information frequency can reduce hallucination risks in DP models. Overall, our findings underscore the need for more nuanced privacy-preserving interventions that offer rigorous privacy guarantees without compromising factual accuracy.
    
[^124]: MemeBridge：用于基准测试与缓解表情包解读中双向文化差距的数据集

    MemeBridge: A Dataset for Benchmarking and Mitigating the Bidirectional Cultural Gap in Meme Interpretation

    [https://arxiv.org/abs/2609.00491](https://arxiv.org/abs/2609.00491)

    该论文提出了MemeBridge数据集，通过同时捕捉中国参与者对美国起源表情包的解读方式以及美国参与者对跨文化误解的预期这两个互补视角，来基准测试并缓解表情包解读中的双向文化差距。

    

    跨文化交流本质上具有挑战性，尤其是通过表情包这类文化密集且含义模糊的形式进行交流时。虽然人们期待大语言模型（LLMs）在弥合此类差距方面具有潜力，但现有的基准测试数据集往往无法捕捉准确解读所需的文化背景。为解决这一问题，我们推出了MemeBridge，这是一个以美国起源的表情包为中心的精选数据集，旨在捕捉两个互补的视角：（1）中国参与者如何解读这些表情包，以及（2）美国参与者如何预期来自其他文化的人可能会误解这些表情包。这里的“背景”指的是隐性的文化知识，包括塑造表情包理解的背景信念、规范和共同假设。该数据集通过多阶段众包流程构建，并经过严格验证，包括人类一致性检查和基于GPT的分类验证。每个表情包……

    arXiv:2609.00491v1 Announce Type: new  Abstract: Communicating across cultures is inherently challenging, especially through culturally dense and ambiguous formats like memes. While people expect large language models (LLMs) to hold promise for bridging such gaps, existing benchmark datasets often fail to capture the cultural context necessary for accurate interpretation. To address this, we introduce MemeBridge, a curated dataset centered on U.S.-originated memes, designed to capture two complementary perspectives: (1) how Chinese participants interpret these memes, and (2) how U.S. participants anticipate how people from other cultures might misunderstand them. Here, context refers to implicit cultural knowledge, including background beliefs, norms, and shared assumptions that shape meme comprehension. The dataset was constructed via a multi-stage crowdsourcing pipeline with rigorous validation, including human agreement checks and GPT-based classification verification. Each meme is 
    
[^125]: EvoFlint：多轮LLM漏洞的进化图谱

    EvoFlint: An Evolutionary Atlas of Multi-Turn LLM Vulnerabilities

    [https://arxiv.org/abs/2609.00487](https://arxiv.org/abs/2609.00487)

    提出了EvoFlint框架，将多轮红队测试从生成问题重新定义为搜索问题，通过进化式质量多样性搜索演化分阶段对话攻击策略，构建出目标模型漏洞的结构化图谱。

    

    前沿语言模型在单轮有害提示下往往会拒绝回答，但当同样的有害意图通过多轮对话逐步达成时，它们却常常配合执行，这使得多轮攻击成为大型语言模型最不为人理解的失效模式之一。大多数自动化红队测试方法将其视为一个生成问题：生成能够攻破模型的攻击。我们认为将其更好地表述为一个搜索问题：发现、组织并迭代优化一个多样化的攻击策略档案库，从而生成一张关于目标模型如何失效的结构化地图，而非一次性的成功攻击列表。我们提出了EvoFlint，它将进化式质量多样性搜索应用于多轮红队测试。攻击策略是分阶段的对话计划，而非原始提示词，并通过LLM驱动的变异和交叉操作进行演化。基于攻击成功率和峰值严重程度的帕累托适应度保留了来自“险些成功”攻击的选择信号。一个以风险为索引的档案库运行新颖性搜索……

    arXiv:2609.00487v1 Announce Type: cross  Abstract: Frontier language models that refuse harmful single-turn prompts often comply when the same intent is reached gradually over many turns, making multi-turn attacks one of the least understood failure modes of large language models. Most automated red-teaming methods treat this as a generation problem: produce attacks that break the model. We argue it is better framed as a search problem: discover, organize, and iteratively refine a diverse archive of attack strategies, producing a structured map of how a target model fails rather than a list of one-off successes. We introduce EvoFlint, which applies evolutionary quality-diversity search to multi-turn red-teaming. Attack strategies are phased conversation plans, not raw prompts, and are evolved through LLM-driven mutation and crossover. A Pareto fitness over attack success rate and peak severity preserves selection signal from near-miss attacks. A risk-indexed archive runs novelty search
    
[^126]: 基于家族DIF指导的基准重组下，接近持平的大语言模型排名是否稳健？

    Are Near-Tied LLM Rankings Robust to Family-DIF-Guided Benchmark Recomposition?

    [https://arxiv.org/abs/2609.00482](https://arxiv.org/abs/2609.00482)

    该论文提出一种基于无家族标签谱近似MIRT的基准重组方法，发现尽管全基准与低DIF排名强相关，但相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，表明排行榜上的微小差距并不稳健。

    

    排行榜上的微小差距常被解读为某个语言模型优于另一个的证据，但其结论方向可能取决于包含哪些基准题目。我们利用五个基准的题目级响应数据以及一种无家族标签的谱近似多维项目反应理论（MIRT）来检验这一点。在所有者不相交的折中划分下，一半所有者数据用于识别跨模型家族具有低残差差异项目功能（低DIF）的题目；由此得到的固定且按来源和难度平衡的权重用于对另一半数据中的模型进行评分，同时使用等长的匹配随机子测试来控制一般性的子测试变异。全基准排名与低DIF排名保持强相关（τb=.900-.948）。然而，在五个基准中的四个里，最初相差不到一个百分点的跨家族模型对中有30.9%-47.1%出现排名反转，比匹配随机子测试的中位数高出16.9-28.6个百分点（均为p=.001）。第五个基准[摘要截断]

    arXiv:2609.00482v1 Announce Type: new  Abstract: Small leaderboard gaps are often interpreted as evidence that one language model is better than another, but their sign may depend on which benchmark items are included. We test this using item-level responses from five benchmarks and a family-label-free spectral approximation to multidimensional item-response theory (MIRT). In owner-disjoint folds, one owner half identifies items with low residual differential item functioning across model families (low-DIF); the resulting frozen, source- and easiness-balanced weights score models in the other half, while equally short matched-random subtests control for generic subtest variation. Full-benchmark and low-DIF rankings remain strongly correlated ($\tau_b=.900$--$.948$). Yet in four of five benchmarks, 30.9--47.1\% of cross-family pairs initially within one percentage point reverse order, exceeding their matched-random medians by 16.9--28.6 percentage points (all $p=.001$). The fifth benchm
    
[^127]: 探索语言智能体与非语言智能体之间的协作

    Exploring Collaboration between a language and a non-language agent

    [https://arxiv.org/abs/2609.00474](https://arxiv.org/abs/2609.00474)

    该论文提出LLAMIA-Bench基准，用于研究将非语言智能体的连续表示“言语化”为文本是否成为LLM协作的瓶颈，并提出潜在状态内化方法来改善LLM与国际象棋引擎等非语言智能体的协作。

    

    大型语言模型（LLM）越来越多地被部署为协调者，通过自然语言调度专门的子智能体来解决复杂任务。然而，在博弈和机器人技术等许多重要领域，目前最强的智能体并非语言模型。将非语言智能体与LLM集成需要进行“言语化”：在每个交互步骤中，将其丰富的连续表示压缩为稀疏的文本摘要。为了研究言语化是否构成瓶颈，我们提出了LLAMIA-Bench，这是一套包含六个多样化协作式国际象棋任务的基准，涵盖三个方面：行为模仿、状态评估和自然语言解释。每个任务都对应一个经典的国际象棋难题，无论是LLM还是象棋引擎都无法独立解决。为了实现LLM与非语言智能体的协作，我们提出了“潜在状态内化”方法，将子智能体的连续表示投影到……

    arXiv:2609.00474v1 Announce Type: cross  Abstract: LLMs are increasingly deployed as orchestrators that coordinate specialized subagents to solve complex tasks through natural language. However, in many important domains like game playing and robotics, the strongest available agents are not language models. Integrating non-language agents with LLMs would require \emph{verbalization}: compressing their rich continuous representations into sparse textual summaries at each interaction step. To study whether verbalization constitutes a bottleneck, we introduce \textsc{LLAMIA-Bench}, a suite of six diverse collaborative chess tasks spanning three facets: behavioral imitation, state assessment, and natural-language explanation. Each task instantiates a well-established chess problem that neither the LLM nor the chess engine can solve alone. To solve LLM collaboration with non-language agents, we introduce \emph{latent state internalization}, which projects the subagent's continuous represent
    
[^128]: TRIS：一种抵御知识投毒的三层检索完整性筛

    TRIS: A Tri-Layer Retrieval Integrity Sieve Against Knowledge Poisoning

    [https://arxiv.org/abs/2609.00470](https://arxiv.org/abs/2609.00470)

    本文提出TRIS三层筛，一种中间件防御方案，通过跨嵌入空间聚类、触发器-载荷结构过滤和大模型一致性验证三重机制清洗RAG检索证据，利用投毒文档难以同时满足嵌入几何、内部结构和生成目标三重要求的固有弱点，有效抵御知识投毒攻击。

    

    检索增强生成（RAG）将大语言模型锚定在外部语料库之上，但对检索文档的隐式信任构成了一个关键的攻击面：PoisonedRAG研究表明，仅需少量精心构造的段落即可主导稠密检索，并将模型生成引向攻击者预设的答案。我们提出三层筛（Tri-Layer Sieve），这是一种中间件防御方案，通过以下三重机制清洗检索到的证据：借助独立裁判模型进行跨嵌入空间聚类、针对触发器-载荷伪迹的结构化过滤，以及大语言模型一致性验证。该设计利用了检索阶段投毒的一个关键弱点：单个投毒文档必须同时满足特定的嵌入几何结构、特定的内部触发器-载荷结构以及特定的生成目标——三者很难同时成立，即便面对通过改写来规避防御的自适应攻击者，这一脆弱性依然存在。在Natural Questions、HotpotQA和MS-MARCO数据集上使用Contriever检索（k=50），三层筛显著降低了……（摘要原文在此处截断）

    arXiv:2609.00470v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) grounds large language models in external corpora, but implicit trust in retrieved documents creates a critical attack surface: PoisonedRAG shows that a handful of crafted passages can dominate dense retrieval and steer generation toward attacker-chosen answers. We present the Tri-Layer Sieve, a middleware defense that sanitizes retrieved evidence through cross-embedding-space clustering with an independent judge model, structural filtering of trigger-payload artifacts, and LLM consistency verification. The design exploits a key weakness of retrieval-stage poisoning: a single document must satisfy one embedding geometry, one internal Trigger-Payload structure, and one generation objective - rarely all three simultaneously, a fragility that persists even against an adaptive attacker who paraphrases around it. On Natural Questions, HotpotQA, and MS-MARCO with Contriever retrieval (k=50), the Sieve reduc
    
[^129]: 推翻字节级语言建模中的层次结构

    Toppling the Hierarchy in Byte-level Language Modeling

    [https://arxiv.org/abs/2609.00463](https://arxiv.org/abs/2609.00463)

    研究发现层次化设计本身限制了字节级模型的字符理解能力，纯字节级模型凭借注意力机制在字符操作任务上始终优于层次化变体，揭示了计算效率与细粒度字符理解之间的明确权衡。

    

    本工作研究了近期的字节级模型及其在完美处理字符方面的不足。最先进的字节级模型采用层次结构，从字节级别开始，下采样到词级别，然后再上采样回字节。虽然这种设计提高了训练和推理效率，但我们发现层次化设计本身限制了字符级理解能力，纯字节级模型在字符操作任务上始终优于层次化变体。将Transformer层消融为注意力机制和前馈组件的实验进一步揭示，字节级注意力是驱动这一行为的主要机制。总的来说，我们的结果为层次化字节模型的字符级失败提供了解释，并在计算效率与细粒度字符理解之间建立了明确的权衡。

    arXiv:2609.00463v1 Announce Type: new  Abstract: This work examines recent byte-level models and their failure to perfectly manipulate characters. State-of-the-art byte-level models use a hierarchical structure, starting at the byte level, downsampling to the word level, and then upsampling back to bytes. While this improves training and inference efficiency, we find that the hierarchical design itself limits character-level understanding, with pure byte-level models consistently outperforming hierarchical variants on character manipulation tasks. Ablating transformer layers into attention and feed-forward components further reveals that byte-level attention is the primary mechanism driving this behavior. Together, our results provide an explanation for the character-level failures of hierarchical byte models and establish a clear trade-off between computational efficiency and fine-grained character understanding.
    
[^130]: 基于二级嵌入的位置感知语言模型

    Location-Aware Language Models via Secondary Embeddings

    [https://arxiv.org/abs/2609.00454](https://arxiv.org/abs/2609.00454)

    提出一种轻量级、模型无关的方法，通过将地名与经纬度结合并采用位置聚焦掩码机制，在无需修改分词器或重新训练的情况下为预训练语言模型注入地理空间感知能力，显著提升地理空间对齐效果。

    

    预训练的基于Transformer的语言模型在广泛的NLP任务中表现出强大的性能，但在编码地理位置语义方面仍存在局限，导致地名和空间实体的表示不够理想。在这项工作中，我们提出了一种轻量级、模型无关的方法，可以在不修改分词器或进行昂贵重训练的情况下，将地理空间感知注入到预训练嵌入中。我们的方法通过将位置名称与其对应的经纬度相结合，利用结构化的地理信号来增强输入表示，并采用以位置为中心的掩码机制，以更好地将文本表示与真实世界的空间关系对齐。这种设计使模型能够在保留现有语义和句法知识的同时融入地理空间上下文。实验结果表明，该方法在地理空间对齐方面取得了显著改进，同时在保持相当性能方面表现出色。

    arXiv:2609.00454v1 Announce Type: new  Abstract: Pretrained transformer-based language models achieve strong performance across a wide range of NLP tasks but remain limited in encoding geo-locational semantics, leading to suboptimal representations of place names and spatial entities. In this work, we propose a lightweight, model-agnostic approach for injecting geo-spatial awareness into pretrained embeddings without modifying the tokenizer or requiring costly retraining. Our method augments input representations with structured geographic signals by combining location names with their corresponding latitude and longitude, and employs a location-focused masking to better align textual representations with real-world spatial relationships. This design allows the model to incorporate geo-spatial context while preserving existing semantic and syntactic knowledge. Experimental results demonstrate substantial improvements in geo-spatial alignment while maintaining comparable performance on 
    
[^131]: 群体自适应裁剪策略优化

    Group Adaptive Clipping Policy Optimization

    [https://arxiv.org/abs/2609.00444](https://arxiv.org/abs/2609.00444)

    该论文提出 GAPO，一种基于反向 KL 信任域视角对 GRPO 的即插即用改进，通过根据 rollout 优势自适应调整裁剪边界，让具有更强学习信号的稀有正确 rollout 获得更大的更新空间，从而解决固定裁剪对探索性 rollout 的过度抑制问题。

    

    在基于可验证奖励的强化学习（RLVR）中，群体相对策略优化（GRPO）通常对所有 rollout 使用固定的重要性采样（IS）比率裁剪边界。我们发现了这一方法的一个关键局限：较难问题上稀有的正确 rollout 和较简单问题上充裕的正确 rollout 会以相近的比率被裁剪，尽管它们贡献的学习信号截然不同。群体成功率较低的 rollout 表现出更大的 IS 比率，并为探索和解决新问题携带更强的梯度信号，然而它们却被固定裁剪不成比例地抑制。为了解决这一问题，我们提出了群体自适应裁剪策略优化（GAPO），这是对 GRPO 方法的一种即插即用式修改，能够根据 rollout 的优势自适应地调整裁剪边界。GAPO 的设计源于反向 KL 信任域的视角，该视角表明具有更强学习信号的 rollout 应获得相应更大的更新空间……

    arXiv:2609.00444v1 Announce Type: cross  Abstract: Group relative policy optimization for reinforcement learning with verifiable rewards (RLVR) typically uses a fixed importance-sampling (IS) ratio clipping boundary across all rollouts. We identify a key limitation: rare correct rollouts on harder problems and abundant correct rollouts on easier problems are clipped at comparable rates, despite contributing very different learning signals. Rollouts with low group success exhibit larger IS ratios and carry stronger gradient signal for exploration and solving new problems, yet are disproportionately suppressed by fixed clipping.   To address this, we propose Group Adaptive Clipping Policy Optimization (GAPO), a plug-in modification to GRPO methods that adapts the clipping boundary to the rollout advantage. GAPO is motivated by a reverse-KL trust-region perspective, which suggests that rollouts with larger learning signal should receive proportionally greater update headroom. GAPO require
    
[^132]: （视觉）语言模型能够超越表面共现进行泛化：来自跨模态数一致性的证据

    (V)LMs generalize beyond surface co-occurrence: Evidence from cross-modal number agreement

    [https://arxiv.org/abs/2609.00443](https://arxiv.org/abs/2609.00443)

    该研究通过跨模态泛化实验证明，视觉语言模型在学习新名词后，能将仅从视觉线索获得的语法数知识泛化到语言层面，表明模型掌握的是抽象的语法规则，而非仅仅依赖表面词汇共现。

    

    语言模型主要从共现中学习语法数，并因此表现出频率效应——这有时被解读为它们并未学习抽象的“规则”，而是依赖于特定的词汇项。仅用文本刺激来测试泛化无法解决这一争论，因为分布线索（is/are、this/these）很容易直接暴露数的信息。我们转而采用跨模态泛化作为工具，来研究同时能接受视觉输入的语言模型（VLMs）中的抽象能力，将用于诊断数的证据限制在语言之外的模态中。我们通过添加新的嵌入向量并仅在学习过程中更新这些向量，来教授VLMs成对的新名词，并对比了仅由视觉线索诊断数的条件与由文本消歧的条件。在行为、表征动力学和因果机制等多个层面，我们发现了跨模态泛化的实质性证据……

    arXiv:2609.00443v1 Announce Type: cross  Abstract: Language models learn about grammatical number primarily from co-occurrence, and show frequency effects as a result---sometimes taken to indicate that they do not learn abstract ``rules'', and are instead dependent on specific lexical items. Testing generalization with text stimuli alone cannot settle this debate, since distributional cues (is/are, this/these) easily give number away. We instead use cross-modal generalization as a tool to investigate abstractions in LMs that can also accept visual inputs (VLMs), restricting the evidence that diagnoses number to an extra-linguistic modality. We teach VLMs pairs of new nouns by adding new embeddings and only updating them during learning, comparing conditions where number is diagnosed by visual cues alone against ones where it is disambiguated by text. Across behavior, representational dynamics, and causal mechanisms, we find non-trivial evidence for cross-modal generalization across bot
    
[^133]: SAGE：面向任务型对话智能体的状态接地、弃权感知评估

    SAGE: State-Grounded, Abstention-Aware Evaluation of Task-Oriented Dialogue Agents

    [https://arxiv.org/abs/2609.00434](https://arxiv.org/abs/2609.00434)

    SAGE提出将工作流规范编译为原子准则，通过会弃权而非猜测的符号与编码器/NLI验证器级联来评估任务型对话智能体每轮的状态推进，其中SAGE-Core可在零付费LLM成本下判定81-91%的准则。

    

    评估任务型对话智能体不仅要判断回复是否读起来流畅，还要判断每一轮对话是否正确推进了底层工作流状态——传统整体式LLM评判器往往忽略这一区别，因为它们将可用上下文作为单一整体进行评估，且每轮都需要一次或多次完整模型调用。我们提出SAGE（状态接地、弃权感知评估），该方法将工作流规范和逐轮状态差异编译为原子化的、基于模式的准则，并将每条准则通过符号验证器与编码器/NLI验证器构成的级联进行路由，这些验证器在不确定时选择弃权而非猜测，最终将各准则的判定聚合为带有证据轨迹的轮级决策。其推荐的运行配置SAGE-Core仅依靠编译器、符号规则和设备端编码器即可判定81-91%的准则，且零付费LLM成本；SAGE-LLM则针对开放类准则增加了可选的聚焦LLM回退机制。在跨越四个切片的……（摘要原文在此处截断）

    arXiv:2609.00434v1 Announce Type: new  Abstract: Evaluating task-oriented dialogue agents requires judging not merely whether a reply reads well but whether each turn advances the underlying workflow state correctly--a distinction conventional holistic LLM judges can miss because they evaluate the available context as a single unit and require one or more full-model calls per turn. We propose SAGE (State-Grounded Abstention-Aware Evaluation), which compiles a workflow specification and per-turn state diff into atomic, schema-grounded criteria and routes each through a cascade of symbolic and encoder/NLI verifiers that abstain rather than guess, aggregating criterion verdicts into a turn-level decision with an evidence trace. Its recommended operating point, SAGE-Core, decides 81--91% of criteria with only the compiler, symbolic rules, and on-device encoders--at zero paid LLM cost--while SAGE-LLM adds an optional focused-LLM fallback for open-class criteria. Across four slices spanning 
    
[^134]: 后期Transformer层以规范化方式重新编码句法：来自希腊语语序变换与跨层泛化的证据

    Late Transformer Layers Recode Syntax Canonically: Evidence from Greek Scrambling and Cross-Layer Generalisation

    [https://arxiv.org/abs/2609.00416](https://arxiv.org/abs/2609.00416)

    Transformer后期层会将非规范句法结构方向性地重新编码为规范语序形式，而非简单丢失句法信息，这一结论通过希腊语SVO/VSO最小对立对的跨层泛化探测分析得到证实。

    

    探测研究已证实句法信息在Transformer的早期和中间层是可解码的，但这些信息在后期层中发生了什么仍知之甚少。我们对三个希腊语微调的大语言模型应用了跨层泛化分析，并在严格控制的最小对立对上进行评估：现代希腊语中的宾语关系结构，其中规范语序（主-谓-宾；SVO）与非规范语序（谓-主-宾；VSO）仅在句内词序上有所不同，同时保持命题意义不变。当在后期层（20-31）训练的探测器在各个早期层上单独测试时，其迁移效果低于随机水平（经聚类校正，p<0.01），将99.3%的非规范句子归类为规范句。探测器系数在第22层左右发生符号反转，表明存在向规范形式的方向性重新编码，而非简单的信息丢失。这些发现刻画了一种表征……

    arXiv:2609.00416v1 Announce Type: new  Abstract: Probing studies have established that syntactic information is decodable in early and middle transformer layers, but what happens to that information in later layers remains poorly understood. We apply a cross-layer generalisation analysis to three Greek-tuned large language models evaluated on tightly controlled minimal pairs: object-relative constructions in Modern Greek, where canonical (Subject-Verb-Object; SVO) and non-canonical (Verb-Subject-Object; VSO) orders differ only in within-clause word order, while preserving propositional meaning. When a probe trained on late layers (20-31) is tested on each early layer individually, it produces below-chance transfer (cluster-corrected, p<0.01), classifying 99.3% of non-canonical sentences as canonical. Probe coefficients reverse sign around layer 22, indicating a directional recoding toward the canonical form rather than simple information loss. These findings characterise a representati
    
[^135]: 可移除与不可约简：面向多语言分词税的词元成本账本

    Removable and Irreducible: A Token-Cost Ledger for the Multilingual Tokenization Tax

    [https://arxiv.org/abs/2609.00378](https://arxiv.org/abs/2609.00378)

    该论文提出词元成本账本框架，将多语言分词税分解为可移除的编码冗余与不可约简的成本项，并证明仅用约千句语料训练的文字匹配编码即可移除印度系文字相比英文高达 8.9 倍词元成本差距中位数 64% 的部分。

    

    大型语言模型在非英文文本上需要支付一份有据可查的“税”：相同内容需消耗数倍之多的词元，且由于注意力机制对序列长度呈二次方复杂度，所需计算量更是大幅增加。我们探究这份税中有多少是可以被移除的。我们将词元层建模为信源编码问题——Transformer 的计算量随序列长度单调递增，其每原子下界为香农速率 H/log₂V（该对象在先前工作中已被应用于分词器）——并据此构建了一个词元成本账本，在固定平行内容的条件下，将每种语言的成本分解为四部分：可移除的编码冗余、残余的编码松弛、内在内容项，以及一个正交且不可约简的字素-音素项（该项支配的是多模态成本而非文本成本）。在涵盖八种语言的 FLORES-200 数据集上，生产级分词器处理印度系文字的词元成本最高可达英文的 8.9 倍；而一个仅用 1,012 个句子训练的、与文字匹配的编码即可移除该超额成本中位数 64% 的部分。

    arXiv:2609.00378v1 Announce Type: new  Abstract: Large language models pay a well-documented tax on non-English text: the same content costs several times more tokens, and because attention is quadratic in sequence length, far more compute. We ask how much of this tax is removable. Framing the token layer as source coding -- transformer compute is monotone in sequence length, whose per-atom floor is the Shannon rate $H/\log_2 V$, an object already applied to tokenizers in prior work -- we assemble a token-cost ledger that splits each language's cost, at fixed parallel content, into a removable coding redundancy, a residual coding slack, an intrinsic-content term, and an orthogonal, irreducible grapheme-to-phoneme term that governs the multimodal rather than the text cost. On FLORES-200 across eight languages, a production tokenizer costs up to $8.9\times$ more tokens for Indic scripts than for English; a script-matched code trained on $1,012$ sentences removes a median $64\%$ of that e
    
[^136]: 面向数据工程的神经符号方法：无需微调实现长上下文Token缩减

    Neurosymbolics for Data Engineering: Achieving Long Context Token Reduction Without Finetuning

    [https://arxiv.org/abs/2609.00367](https://arxiv.org/abs/2609.00367)

    本文提出一种即插即用的神经符号层，无需任何微调或RLHF即可在Text-to-SQL等数据工程任务上平均提升85%的准确率，同时缓解Transformer长上下文的计算资源瓶颈。

    

    大型语言模型正越来越多地被部署用于复杂的数据工程任务，例如从自然语言生成结构化查询（Text-to-SQL）以及自动化复杂的电子表格操作。然而，要最大化其效用，既需要更高的免微调准确率，也需要解决Transformer架构固有的二次方（O(n²)）时间复杂度所带来的计算瓶颈。本文提出了一种新颖的即插即用神经符号层，旨在无缝集成到现有的LLM骨干网络中，增强逻辑推理能力并缓解长上下文的资源消耗。在推理方面，该层能够立即且显著地提升性能，在包括BIRD-CRITIC和LiveSQLBench在内的严格基准测试中实现了平均85%的准确率提升，关键是这些提升无需任何任务特定的微调或RLHF。同时，我们将该方法重新应用于解决长上下文问题……

    arXiv:2609.00367v1 Announce Type: cross  Abstract: Large Language Models are increasingly deployed for sophisticated data engineering tasks such as generating structured queries from natural language, Text-to-SQL, and automating complex spreadsheet operations. However, maximizing their utility demands both higher finetuning-free accuracy and solutions to the computational bottleneck imposed by the Transformer architectures inherent quadratic (On2) time complexity. This paper introduces a novel drop-in neurosymbolic layer designed to seamlessly integrate into existing LLM backbones enhancing logical reasoning and mitigating long-context resource consumption. On the reasoning front, the layer immediately and significantly improves performance yielding an average accuracy increase of 85% across rigorous benchmarks including BIRD-CRITIC and LiveSQLBench, critically achieving these gains without any task specific finetuning or RLHF. Concurrently, we repurpose this approach to address the se
    
[^137]: Dr. Claw：一个面向氛围式研究的AI科学家工作区

    Dr. Claw: An AI Scientist Workspace for Vibe Research

    [https://arxiv.org/abs/2609.00365](https://arxiv.org/abs/2609.00365)

    Dr. Claw 是一个开源的AI科学家工作区，通过持久化状态对象、可复用技能库和多执行器协调，将现有命令行编码代理封装为可控、可审计的人机协同工作流，把科研中的规划、执行与写作整合为一个可追踪、可恢复的闭环。

    

    命令行编码代理（如 Claude Code、Gemini CLI）已经能够读写文件并维持长会话，然而端到端的科研工作仍然碎片化地分散在聊天工具、IDE、终端和写作环境之间，且那些使研究可审计的决策很少被保存下来。我们提出了 Dr. Claw，一个开源工作区，它将现有的编码代理执行器封装在一个可控且可审计的人机协同工作流中，而非引入另一个自主智能体。持久化的状态对象、可复用的技能库以及多执行器协调机制将人类决策与AI执行联系起来，使规划、执行和写作整合为一个可追踪、可恢复的闭环。我们通过一个交互式三视图场景和一次故障恢复演示来展示 Dr. Claw，并将其与共享同一后端执行器的裸命令行代理进行对比评估，因此该对比考察的是整个编排层（任务图、状态对象等）。

    arXiv:2609.00365v1 Announce Type: new  Abstract: Command-line coding agents (e.g., Claude Code, Gemini CLI) can already read and write files and sustain long sessions, yet end-to-end research still fragments across chat tools, IDEs, terminals, and writing environments, and the decisions that make it auditable are rarely preserved. We present Dr. Claw, an open-source workspace that wraps existing coding-agent executors in a controllable and auditable human-in-the-loop workflow rather than introducing another autonomous agent. Persistent state objects, a reusable skill library, and multi-executor coordination link human decisions to AI execution, turning planning, execution, and writing into one traceable, recoverable loop. We demonstrate Dr. Claw through an interactive three-view scenario and a failure-recovery walkthrough, and evaluate it against a bare command-line agent sharing the same backend executor, so the comparison contrasts the whole orchestration layer (task graph, state obj
    
[^138]: 净化毒性沟通：一种面向负责任AI的设计科学方法

    Detoxifying Toxic Communication: A Design Science Approach to Responsible AI

    [https://arxiv.org/abs/2609.00361](https://arxiv.org/abs/2609.00361)

    本研究采用设计科学方法，将微调的Transformer毒性分类器与生成式去毒模型相结合，构建了一个负责任AI系统，能够检测数字职场中的毒性沟通并将其改写为语义等价的无冒犯性表达，在保留对话连续性的同时促进尊重性交流。

    

    数字职场中的毒性语言，如贬损性言辞、讽刺挖苦、居高临下的态度以及隐性的不礼貌行为，会侵蚀信任、士气与协作。现有的内容审核工具主要通过删除或屏蔽有害信息来处理，这会中断沟通，且无法提供建设性的解决方案。本研究采用设计科学研究方法，构建了一个能够检测并净化毒性沟通的负责任AI制品。该制品将经过微调的基于Transformer的分类器（DistilBERT、DistilRoBERTa）与生成式去毒模型（mT0-XL-Detox-ORPO）相结合，后者可将有毒文本改写为语义等价、无冒犯性的表达。技术评估表明，该系统在毒性检测方面具有很高的准确性，改写后的信息也能很好地保留原意，在强化尊重性话语的同时支持对话的连续性。本文为负责任的AI审核贡献了优先保障有意义沟通的设计原则。

    arXiv:2609.00361v1 Announce Type: cross  Abstract: Toxic language in digital workplaces such as pejoratives, sarcasm, condescension, and subtle incivility can erode trust, morale, and collaboration. Existing moderation tools primarily delete or block harmful messages, disrupting communication and offering no constructive resolution. This study adopts a Design Science Research approach to create a responsible AI artifact that detects and detoxifies toxic communication. The artifact integrates fine-tuned transformer-based classifiers (DistilBERT, DistilRoBERTa) with a generative detoxification model (mT0-XL-Detox-ORPO) that rewrites toxic text into semantically equivalent, non-offensive paraphrases. Technical evaluation demonstrates high accuracy in toxicity detection and strong semantic preservation in rewritten messages, supporting conversation continuity while reinforcing respectful discourse. The paper contributes design principles for responsible AI moderation that prioritize meanin
    
[^139]: 视觉并非开销：面向视觉语言模型无损推测解码的单遍块草拟方法

    Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models

    [https://arxiv.org/abs/2609.00355](https://arxiv.org/abs/2609.00355)

    该论文提出 GLANCE——首个在未修改的视觉语言模型上实现无损推测解码的单遍块草拟器，通过块扩散头零成本读取目标模型已融合的视觉-语言状态，并在一次前向传播中完成整块草拟与宽候选树验证，从而打破了草拟器因规模受限而被迫牺牲视觉信息的自我挫败循环。

    

    推测解码能够在不改变输出结果的前提下加速生成，但在视觉语言模型上，它却陷入了一种自我挫败的循环：草拟器必须保持自回归架构，因而只能维持小规模；小型草拟器无法在每一步都承担图像处理的代价，于是视觉信息被压缩、剪枝或隐藏；而被切断了图像信息的草拟器，恰恰在图像最能让文本变得可预测的地方变得最不可靠。我们提出 GLANCE——首个在未经修改的 VLM 目标模型上实现无损解码的单遍块草拟器，它从两端打破了这一循环。一个块扩散头读取目标模型已经融合好的视觉-语言状态，因此视觉对草拟器而言零开销；同时它在一次前向传播中填满整个块，因此模型深度不会带来额外的串行步数。宽候选树通过一次目标模型前向传播即可完成验证，且经审计的每个提示都能精确复现贪婪解码的结果。在依赖视觉依据的工作负载上收益最为显著，会进入一种逐字复制的模式，其长段连续（原文摘要在此处截断）……

    arXiv:2609.00355v1 Announce Type: new  Abstract: Speculative decoding accelerates generation without changing its output, yet on vision-language models (VLMs) it has been caught in a self-defeating cycle. The drafter stays autoregressive, so it must stay small. A small drafter cannot afford the image at every step, so vision is compressed, pruned, or hidden. A drafter cut off from the image is then least reliable exactly where the image makes text predictable. We present GLANCE, the first one-pass block drafter that is lossless on an unmodified VLM target, and it breaks the cycle at both ends. A block-diffusion head reads the target's already-fused vision-language state, so vision costs the drafter nothing, and fills a whole block in one forward pass, so depth costs no sequential steps. A wide candidate tree is verified in one target pass, and every audited prompt reproduces greedy decoding exactly. Grounded workloads reward this most, entering a verbatim-copy regime whose long runs co
    
[^140]: 通过激活匹配微调检测大语言模型中的隐藏行为

    Detecting Hidden Behaviors in LLMs via Activation-matched Finetuning

    [https://arxiv.org/abs/2609.00351](https://arxiv.org/abs/2609.00351)

    论文提出“激活匹配微调”这一无监督检测方法，通过在良性语料上微调锚定模型以复现可疑模型的激活并计算残差，在无需知晓触发器或目标行为的前提下检测出大语言模型中的后门、审查等隐藏行为及其语义邻近提示。

    

    大语言模型可能潜藏一些仅在狭窄条件下才激活的行为，例如后门触发器、睡眠代理部署线索、故意放水或基于话题条件的审查。这类行为在缺乏先验知识（不知道要寻找什么）的情况下难以被检测。我们提出了激活匹配微调，这是一种无监督检测方法，无需对触发器或目标行为的任何先验知识。给定一个可疑模型和一个公开可用的锚定模型，我们在小型良性语料库上微调锚定模型以复现可疑模型的激活，并通过两个模型之间的残差对每个评估提示进行评分。由于没有任何良性语料库能够覆盖稀疏的触发区域，参考模型只会学习到良性计算而学不到隐藏行为。因此，触发提示——以及关键的，它们的语义邻近提示——会产生较大的残差，从而向防御者发出存在异常行为的信号。

    arXiv:2609.00351v1 Announce Type: cross  Abstract: Large language models can hide hidden behaviors that activate only under narrow conditions, such as backdoor triggers, sleeper-agent deployment cues, sandbagging, or topic-conditioned censorship. Such behaviors are difficult to detect without prior knowledge what to look for. We present activation-matched finetuning, an unsupervised detection method that assumes no knowledge of the trigger or the target behavior. Given a suspect model and a publicly available anchor, we finetune the anchor to reproduce the suspect's activations on a small benign corpus, and score each evaluation prompt by the residual between the two models. Since no benign corpus covers the sparse trigger region, the reference learns the benign computation but not the hidden behavior. Therefore, trigger prompts -- and, crucially, their semantic neighbors -- incur a large residual that signal the presence of unusual behavior to the defender. Testing our method across t
    
[^141]: 从工具使用到技术能动性：LoopCAT——面向翻译技术教育的本地优先开源工具

    From Tool Use to Technological Agency: LoopCAT as a Local-First, Open-Source Tool for Translation Technology Education

    [https://arxiv.org/abs/2609.00344](https://arxiv.org/abs/2609.00344)

    本文介绍了一款与AI协作开发的本地优先开源计算机辅助翻译工具LoopCAT，并提出了连接工作流能力、评价性判断与技术能动性的翻译技术教育框架，使学生既能使用翻译技术又能评判其选择。

    

    翻译专业的学生既需要学习如何使用翻译技术，也需要学习如何评判这些技术所提供的各种选择。本文介绍了LoopCAT，这是一个采用Apache-2.0许可证、本地优先的计算机辅助翻译（CAT）环境，由OpenAI Codex（使用GPT-5.5和GPT-5.6）协作开发而成，并提出了一个连接工作流能力、评价性判断与技术能动性的框架。本文的论述基于代码仓库历史、实现代码检查以及一个已识别开发版本的验证记录。LoopCAT集成了本地项目存储、翻译记忆库、术语库、质量保证、文档交换以及与本地或托管AI服务的可选连接。其英语、加泰罗尼亚语和土耳其语界面目录也使该应用程序本身成为教学材料：学生可以将英语界面字符串翻译成另一种语言，并审阅现有的自动生成的目标语言草稿……

    arXiv:2609.00344v1 Announce Type: new  Abstract: Translation students need to learn both how to use translation technologies and how to judge the choices those technologies make available. This article presents LoopCAT, an Apache-2.0-licensed, local-first computer-assisted translation environment co-created with OpenAI Codex using GPT-5.5 and GPT-5.6, and proposes a framework connecting workflow competence, evaluative judgement, and technological agency. The account draws on repository history, implementation inspection, and the verification records of an identified development build. LoopCAT combines local project storage, translation memories, terminology, quality assurance, document exchange, and optional connections to local or hosted AI services. Its English, Catalan, and Turkish interface catalogs also make the application itself available as teaching material: students can translate English UI strings into another language, review the existing automatically generated target draf
    
[^142]: 用于转变预测的相位结构特征的两项锁定测试

    Two locked tests of phase-structure features for transition prediction

    [https://arxiv.org/abs/2609.00335](https://arxiv.org/abs/2609.00335)

    该论文通过两项预先锁定的实证测试检验相位结构特征PC-2能否在基线之上改进对承诺或矛盾终点的预测，结果两项测试均未通过推进标准，官方结论为阴性。

    

    一项已发表的关于旋转注意中相位结构的理论说明，接受了两项预先设定的实证检验，以检验由相位导出的特征是否优于未接收这些特征的基线，从而更好地预测承诺或矛盾终点。研究1冻结了一个矛盾类别流水线，并对PC-2与基线的密封主要比较进行评分。在1,136个符合条件的案例中，配对AUROC差异为+0.00087。99%置信区间包含零，且差异未达到预先设定的+0.05阈值，未通过推进标准。研究2仅在开放块b0-b4上开发了十五种层处理（1,415个转变，20×5分组折）。锁定的合取规则要求PC-2平均重复增量为正、五个种子块中至少四个增量为正、以及这五个差异的平均值为正。没有任何处理通过推进标准。官方选择结果为阴性（无效）。

    arXiv:2609.00335v1 Announce Type: new  Abstract: A published theoretical account of phase structure in rotary attention was subjected to two pre-specified empirical tests of whether phase-derived features improve prediction of a commitment or contradiction endpoint over a baseline that does not receive those features. Study 1 froze a contradiction-category pipeline and scored a sealed primary comparison of PC-2 against baseline. On 1,136 eligible cases the paired AUROC difference was +0.00087. The 99% interval included zero, and the difference did not reach the pre-specified threshold of +0.05. Advancement was not passed. Study 2 developed fifteen layer treatments on open blocks b0-b4 only (1,415 transitions, 20x5 grouped folds). A locked conjunctive rule required a positive PC-2 mean-repeat increment, a positive increment on at least four of five seed blocks, and a positive mean of those five differences. No treatment advanced. The official selection is null. The theoretical paper is 
    
[^143]: 真实场景下的主题匹配：来自真实世界ASR转录文本的基准测试与经验教训

    Topic Matching in the Wild: Benchmark and Lessons from Real-World ASR Transcripts

    [https://arxiv.org/abs/2609.00330](https://arxiv.org/abs/2609.00330)

    该论文构建了一个基于真实呼叫中心ASR转录文本的人工标注主题匹配基准数据集，并通过系统对比发现，配备自然语言主题描述的轻量级大语言模型匹配器在处理噪声转录文本时性能优于句子嵌入和正则表达式方法。

    

    在呼叫中心中，实时坐席辅助工具会针对多个预定义主题中的每一个，判断实时的客户话语是否与之相关，并在相关时向坐席展示辅导卡片。输入数据充满噪声且极具挑战性：即自发电话对话的ASR（自动语音识别）转录文本，这些文本可能不清晰、内容重复，且大多缺乏标点符号。为了系统地研究这一现实世界任务，我们整理了一个基于真实呼叫中心转录文本的人工标注的主题-话语判断数据集。我们比较了三种类型的匹配器：基于正则表达式的基线方法、零样本句子嵌入编码器，以及基于Gemini的大语言模型匹配器。此外，我们的基准中还研究了两种类型的主题表示方式：关键词短语和自然语言描述。我们的实证实验表明，配备自然语言描述的轻量级大语言模型匹配器，其性能显著优于嵌入模型和正则表达式模型。

    arXiv:2609.00330v1 Announce Type: cross  Abstract: In contact centers, real-time agent-assist tools determine, for each of many predefined topics, whether a live customer utterance is relevant and display a coaching card to the agent when it is. The input is noisy and challenging: ASR(Automatic Speech Recognition) transcripts of spontaneous phone conversations, which can be unclear, repetitive, and mostly lack punctuation. To systematically study this real-world task, we curate a human-annotated topic-utterance judgments dataset sourced from real call-center transcripts. We compare three types of matchers: a regex-based baseline, zero-shot sentence embedding encoders, and Gemini-based LLM matchers. In addition, two types of topic representations are studied in our benchmark:keyphrases and natural language description. Our empirical experiments highlight the superior performance of lightweight LLM matchers over embedding and regex models when equipped with natural language descriptions.
    
[^144]: 词汇规范化中的多语言诅咒

    The Curse of Multilinguality in Lexical Normalization

    [https://arxiv.org/abs/2609.00329](https://arxiv.org/abs/2609.00329)

    该研究通过固定容量字符级模型在十二种语言上的实验发现，词汇规范化存在明显的“多语言诅咒”：语言联合训练数量超过一到四种后，各语言准确率持续下降约百分之四十，且下降源于语言间对固定模型容量的竞争而非数据稀释。

    

    词汇规范化是将用户生成文本中充满的嘈杂、非标准词汇（如 tmrw、u、gr8）改写为其标准形式。由于大多数语言的标注数据稀缺，一种流行的捷径是在多种语言上同时训练单个模型。我们提出一个简单的问题：这样的模型应该用多少种语言来训练？使用一个固定容量的字符级模型和来自标准基准的十二种语言，我们将联合训练的语言数量从一种变化到十二种，并测量每种语言的准确率。我们发现了一个明显的多语言诅咒：当一种语言仅与少数其他语言（通常一到四种）联合训练时，准确率最高；随后随着更多语言的加入，准确率持续且大幅下降，当其余语言全部加入时下降约百分之四十。一个保持总训练数据量不变的对照实验使下降来得更早、幅度更大，这表明各种语言在争夺一个固定的模型容量。

    arXiv:2609.00329v1 Announce Type: cross  Abstract: Lexical normalization rewrites the noisy, non-standard words that fill user-generated text (tmrw, u, gr8) into their standard forms. Because labelled data is scarce for most languages, a popular shortcut is to train a single model on many languages at once. We ask a simple question: how many languages should such a model be trained on? Using one fixed-capacity character-level model and twelve languages from a standard benchmark, we vary the number of jointly trained languages from one to twelve and measure per-language accuracy. We find a clear curse of multilinguality: accuracy is highest when a language is trained with only a few others, often just one to four, and then falls steadily and substantially, dropping by about forty percent as the rest are piled on. A control that holds the total amount of training data constant makes the decline arrive sooner and fall further, which points to competition among the languages for one fixed-
    
[^145]: 多语言大模型中语言控制的潜在机制

    Latent Mechanisms of Language Control in Multilingual Language Models

    [https://arxiv.org/abs/2609.00325](https://arxiv.org/abs/2609.00325)

    该研究比较了在跨层转码器中识别语言控制潜在特征的三种方法（ValSel、FreqSel、AnnSel），发现三者均能有效控制多语言大模型的生成语言，其中 FreqSel 综合性能最强，AnnSel 则通过显式语言标注提供了可解释性。

    

    多语言大模型在生成过程中可能出现非预期的语码转换，即在生成时于不同语言之间进行不必要的切换。我们提出了一项对比研究，考察在跨层转码器中识别语言控制潜在特征的三种方法：基于激活值的选择、基于激活频率的选择，以及基于大语言模型生成的潜在特征标注选择。为了评估这些方法在识别语言控制潜在特征方面的有效性，我们引入了两个呈现语码转换现象的多语言基准，用于对七种语言的语言导向进行细粒度分析。通过对 Gemma-2-2B 和 Qwen3-4B 进行针对性干预实验，我们发现这三种方法均能有效操纵生成语言，其中 FreqSel 取得了最强的整体性能，而 AnnSel 则通过显式的语言标注提供了可解释的潜在特征选择。敲除分析……

    arXiv:2609.00325v1 Announce Type: new  Abstract: Multilingual large language models can exhibit unintended code-switching -- unnecessarily alternating between languages during generation. We present a comparative study of three methods that identify language-controlling latents in cross-layer transcoders: activation value-based selection (ValSel), activation frequency-based selection (FreqSel), and LLM-generated latent annotation-based selection (AnnSel). To evaluate the efficacy of these methods in identifying language-controlling latents, we introduce two multilingual benchmarks that exhibit code-switching for fine-grained analysis of language steering across seven languages. Through targeted intervention experiments on Gemma-2-2B and Qwen3-4B, we find that all three methods effectively manipulate generation language, with FreqSel achieving the strongest overall performance, while AnnSel offering interpretable latent selection through explicit language annotations. A knock-out analys
    
[^146]: 真相之源：AI心理健康信息查询中引用来源的多平台、多语言审计

    Sources of Truth: A Multi-Platform, Multilingual Audit of Citations in AI Mental Health Information Queries

    [https://arxiv.org/abs/2609.00319](https://arxiv.org/abs/2609.00319)

    该研究对ChatGPT、Perplexity和Google AI Overview在多语言心理健康查询中生成的15,942条引用进行了系统审计，发现引用来源高度集中于少数域名，表明来源评估责任已从用户转移到AI平台。

    

    在线健康信息检索正从关键词搜索（用户需要浏览排序列表中的链接）转向对话式系统（由系统生成单一答案并整理其引用）。因此，来源评估的责任从用户转移到了平台，然而这些系统所呈现的来源特征仍未被充分刻画。我们对三款免费消费级产品（ChatGPT、Perplexity、Google AI Overview）在两种提示条件下针对二十个英文心理健康问题进行了审计，并将其中三个问题的子集翻译成六种资源水平各异的其他语言。我们在1,140条回复中记录了15,942条引用，涉及1,713个独立域名，随后采用经过人工编码验证的确定性分类器，依据九类组织类型学对所有引用进行分类。引用呈现出高度集中：被引最多的十个域名占英文引用总量的43.6%，政府、商业健康和学术来源紧随其后。

    arXiv:2609.00319v1 Announce Type: cross  Abstract: Online health information seeking is shifting from keyword search, where users consider a ranked list of links, to conversational systems that compose a single answer and curate its citations. Source evaluation therefore passes from user to platform, yet what these systems surface is poorly characterized. We audited three free consumer products (ChatGPT, Perplexity, Google AI Overview) on twenty English mental health questions under two prompt conditions, with a subset of three also translated into six further languages of varying resource tiers. We recorded 15,942 citations across 1,140 responses and 1,713 unique domains, then classified every citation with a nine-category organizational typology applied by a deterministic classifier validated against human coding. Citations were heavily concentrated: the ten most-cited domains accounted for 43.6% of English citations, and government, commercial health, and academic sources were close
    
[^147]: 大语言模型人格设定中的情绪劳动策略偏好

    Emotional Labor Strategy Preferences in LLM Personas

    [https://arxiv.org/abs/2609.00310](https://arxiv.org/abs/2609.00310)

    该研究构建了首个包含500个社会情境事件的情绪劳动策略数据集，发现注入心理测量学人格设定的大语言模型在日常社交场景中能够复现人类由人格特质驱动的情绪劳动策略选择模式。

    

    情绪劳动是指为满足社会或职业期望而对情绪表达进行的费力的管理。人格特质已被发现与情绪劳动策略相关，然而关于这一联系的研究几乎完全依赖于仅在职业情境中实施的自陈量表。我们研究注入了具有心理测量学基础人格设定的大语言模型，是否能在日常社交场景中复现这些由人格驱动的选择模式。我们构建了首个情绪劳动策略数据集，包含500个社会情境事件，每个事件提供三种行为选择，分别对应表层扮演、深层扮演和真实表达。我们从大规模人格库中选取50个虚构角色，并通过两条并行路径对每个角色进行画像：观察者评定的双极形容词组合，以及角色内自陈量表项目。五个大语言模型在两种人格条件下对所有情境进行了评估。我们……

    arXiv:2609.00310v1 Announce Type: new  Abstract: Emotional labor is the effortful management of emotional displays to meet social or professional expectations. Personality traits have been correlated with emotional labor strategies, yet research on this link relies almost exclusively on self-report scales administered only in occupational settings. We investigate whether large language models injected with psychometrically grounded personas reproduce these personality-driven selection patterns across everyday social scenarios. We construct the first emotional labor strategy dataset of 500 socially situated events, each offering three behavioral choices corresponding to surface acting, deep acting, and genuine expression. We source 50 fictional characters from a large-scale personality repository and profile each through two parallel tracks: observer-rated bipolar adjective composites and in-character self-report items. Five LLMs evaluate all scenarios under both persona conditions. We 
    
[^148]: 迈向面向工作流感知的医疗健康NLP智能体基准测试

    Toward Workflow-Aware Benchmarking for Healthcare NLP Agents

    [https://arxiv.org/abs/2609.00296](https://arxiv.org/abs/2609.00296)

    该论文提出了一种面向医疗健康NLP智能体的情节级评估协议，通过在模型、智能体与模拟工作流三个层面区分证据，并以文档更新、证据检索、患者消息和分诊交接四个任务模板实例化，为静态基准测试与真实部署之间搭建了可复现的中间评估层。

    

    大型语言模型（LLM）智能体越来越多地被提出用于医疗健康任务，例如临床文档记录、证据检索、患者消息传递和护理协调。然而，许多评估仍局限于静态医学问答或一次性生成，低估了纵向状态、中断以及人工交接等因素。我们为医疗健康NLP智能体引入了一种情节级别的评估协议。该协议将证据在模型、智能体和模拟工作流行为三个层面进行区分；规定了包含五个字段的情节模式；并定义了针对状态连续性、证据可追溯性和升级转诊决策的标注与评分方法。该协议被实例化为四个任务模板：文档更新、证据检索、患者消息传递和分诊交接。该协议并不声称能够衡量临床结果或部署价值，而是提供了一个可复现的中间评估层，介于静态基准测试与（真实部署场景）之间。

    arXiv:2609.00296v1 Announce Type: new  Abstract: Large language model (LLM) agents are increasingly proposed for healthcare tasks such as clinical documentation, evidence retrieval, patient messaging, and care coordination. Yet many evaluations remain limited to static medical question answering or one-shot generation, under-representing longitudinal state, interruptions, and human handoffs. We introduce an episode-level evaluation protocol for healthcare NLP agents. The protocol separates evidence across model, agent, and simulated-workflow behavior; specifies a five-field episode schema; and defines annotation and scoring for state continuity, evidence traceability, and escalation decisions. It is instantiated as four task templates: documentation update, evidence retrieval, patient messaging, and triage handoff. The protocol does not claim to measure clinical outcomes or deployment value. Instead, it supplies a reproducible intermediate evaluation layer between static benchmarks and
    
[^149]: 看得慢，抑制得慢：理解模态在情境-记忆冲突中的影响

    Slow to See, Slow to Suppress: Understanding the Effects of Modality in Context-Memory Conflicts

    [https://arxiv.org/abs/2609.00293](https://arxiv.org/abs/2609.00293)

    研究发现视觉语言模型在情境-记忆冲突中存在模态不对称偏见——对文本实体偏好上下文信息而对图像实体偏好参数化记忆，其原因是视觉信息的处理延迟阻碍了对事实回忆机制的抑制。

    

    我们研究了视觉语言模型（VLM）如何处理情境-记忆冲突，即模型在上下文中获得的信息与训练期间以参数化方式存储的信息不一致的情况。我们记录到一种不对称的偏见：模型倾向于对文本中出现的实体采用上下文信息，但对图像中出现的实体则倾向于采用参数化存储的信息。我们将这种不对称与跨模态表征对齐延迟联系起来，表明处理视觉实体所需的更长处理时间阻碍了模型对常规事实回忆机制的抑制，从而导致更多的参数化回答。思维链推理似乎无法弥合这一差距，但增加上下文中视觉信息的数量确实显示出一定效果。这些结果说明，随着模型日益多模态化，确保其行为一致性的复杂性。

    arXiv:2609.00293v1 Announce Type: new  Abstract: We investigate how vision-language models (VLMs) handle context-memory conflicts; that is, situations in which the model is given information in context that differs from what was stored parametrically during training. We document asymmetric biases: models tend to prefer in-context information about entities which appear in text, but prefer parametric information about entities which appear in images. We relate this asymmetry to the late representational alignment across modalities, showing that the longer processing time associated with resolving visual entities prevents the suppression of the model's usual factual recall mechanism, thus resulting in more parametric answers. Chain-of-thought reasoning does not appear to resolve the gap, but increasing the amount of visual information in the context does show an effect. These results illustrate the complexity of ensuring consistent behavior as models become increasingly multimodal and re
    
[^150]: NSIDDx：面向低资源环境的神经符号化、以临床医生为中心的鉴别诊断设计框架

    NSIDDx: A Design Framework for Neuro-Symbolic, Practitioner-First Differential Diagnosis in Low-Resource Settings

    [https://arxiv.org/abs/2609.00256](https://arxiv.org/abs/2609.00256)

    本文提出NSIDDx设计框架，主张在低资源环境下将临床医生作为主动推理主体融入鉴别诊断，通过三值症状编码、矛盾检测、审计字符串和医生覆盖权的神经符号流水线在消费级硬件上离线运行，弥合了LLM诊断系统的头条准确率与可验证临床可靠性之间的差距。

    

    基于大语言模型（LLM）的诊断系统在基准测试中取得了很高的语义准确率，但在临床少见表现上的开放式评估揭示了其“头条准确率”与可验证的临床可靠性之间存在系统性差距。我们在两个队列中评估了“LLM+罕见病RAG”流水线，结果表明该范式产生的输出往往高度自信却经常无法验证，并且系统性地抗拒临床医生的质询。我们提出NSIDDx（神经符号集成鉴别诊断系统），这是一个设计框架，主张低资源环境下的鉴别诊断系统必须将临床医生视为主动的推理主体。我们通过一个包含三值症状编码、矛盾检测、审计字符串和医生覆盖权的神经符号流水线来实现这一理念——该系统可在消费级硬件上离线运行。我们提炼出五条“临床医生在环”临床NLP的设计原则，并呼吁开展必要的前瞻性研究以验证该方法。

    arXiv:2609.00256v1 Announce Type: new  Abstract: LLM-based diagnostic systems achieve high semantic accuracy on benchmarks, but open-ended evaluation on clinically uncommon presentations reveals a systematic gap between headline accuracy and verifiable clinical reliability. We evaluate an LLM+rare-disease-RAG pipeline across two cohorts and show that the paradigm produces confident outputs that are frequently unverifiable and systematically resistant to clinician interrogation. We present NSIDDx (Neuro-Symbolic Integrated Differential Diagnosis System), a design framework arguing that DDx systems in low-resource settings must treat the clinician as an active reasoning agent. We instantiate this through a neuro-symbolic pipeline with ternary symptom encoding, contradiction detection, audit strings, and practitioner override - running offline on consumer hardware. We distill five design principles for clinician-in-the-loop clinical NLP and invite the prospective studies needed to validat
    
[^151]: CompanionSim：用于评估人机关系拟人化的合成数据

    CompanionSim: Synthetic Data for Evaluating Anthropomorphism in Human-AI Relationships

    [https://arxiv.org/abs/2609.00250](https://arxiv.org/abs/2609.00250)

    该论文发布了CompanionSim——一个包含2,240段模拟人机对话的合成数据模拟框架，覆盖七种用例中的16种聊天机器人行为，用于大规模研究人类对AI陪伴行为的感知。

    

    如今许多人不仅将AI系统视为生产力工具，还将其视为社交伴侣。研究人员热切希望研究AI陪伴行为（例如“认同验证”）的后果，这类行为在人际互动中会唤起信任、共情和依恋。然而，人机交互数据有限且不可靠，拖慢了研究进展。我们通过在多种聊天机器人行为和用例下模拟多轮人机对话，来扩展少量真实世界数据的规模。我们发布了CompanionSim：一个模拟框架，包含2,240段模拟的人机对话，涵盖七种用例中的16种聊天机器人行为。在两个探索人们对陪伴行为感知的实验中，人类参与者对模拟对话和真实对话进行了标注。研究1使用了具有美国代表性的样本（N1 = 628），研究2则在美国、英国、印度和尼日利亚开展（N2 = 3,646）。令人惊讶的是……

    arXiv:2609.00250v1 Announce Type: cross  Abstract: Many people now see AI systems as not just productivity tools but as social companions. Researchers are eager to study the consequences of AI companionship behaviors, such as validation, which evoke trust, empathy, and attachment in human-human interaction. However, human-AI interaction data is limited and unreliable, slowing research progress. We scale small amounts of real-world data by simulating multi-turn human-chatbot dialogue across a range of chatbot behaviors and use cases. We release CompanionSim: a simulation framework with 2,240 simulated human-chatbot conversations representing 16 chatbot behaviors across seven use cases. Human participants annotated the simulated conversations and real-world conversations in two experiments probing perceptions of companionship behaviors. We conducted Study 1 with a U.S. representative sample ($N_{1}~=~628$) and Study 2 across the U.S., U.K., India, and Nigeria ($N_{2}~=~3,646$). Surprisin
    
[^152]: CoLT-Drive：面向驾驶可供性预测的反事实长尾基准测试与知识保持自适应

    CoLT-Drive: Counterfactual Long-Tail Benchmarking and Knowledge-Preserving Adaptation for Driving Affordance Prediction

    [https://arxiv.org/abs/2609.00242](https://arxiv.org/abs/2609.00242)

    该论文提出决策级驾驶可供性预测任务，构建了CoLT-Drive反事实长尾基准以评估模型对罕见物体影响可行驾驶动作的推断能力，并提出KPA知识保持自适应框架来提升小型视觉语言模型在长尾驾驶场景中的动作决策性能。

    

    长尾场景下自动驾驶系统的失效常被归结为罕见物体的识别错误。我们认为这一观点并不完整：关键的决策问题不仅在于模型能否识别出异常物体，更在于模型能否推断出该物体将如何改变自车可行的高层动作。我们将这一问题形式化为决策级驾驶可供性预测，即模型根据前视图像、自车运动历史和导航指令，输出结构化的纵向-横向元动作。为评估这一能力，我们提出了CoLT-Drive，一个包含3,536个样本的反事实长尾基准，通过在原本固定的驾驶场景中插入罕见物体，来衡量模型能否预测出可接受的动作对。为改进可部署的小型视觉语言模型（VLM），我们提出了KPA——一种知识保持自适应框架，它结合了结构化的“感知到决策”提示、基于SLERP的专家合并，以及RegMoE（一种基于……的混合专家方法）……

    arXiv:2609.00242v1 Announce Type: cross  Abstract: Long-tail autonomous driving failures are often framed as rare-object recognition errors. We argue that this view is incomplete: the decision-critical question is not only whether a model recognizes an unusual object, but whether it infers how that object changes the ego vehicle's feasible high-level actions. We formalize this problem as decision-level driving affordance prediction, where a model maps a front-view image, ego-motion history, and navigation command to a structured longitudinal--lateral meta-action. To evaluate this capability, we introduce CoLT-Drive, a 3,536-sample counterfactual long-tail benchmark that inserts rare objects into otherwise fixed driving scenes and measures whether models predict acceptable action pairs. To improve deployable small VLMs, we propose KPA, a knowledge-preserving adaptation framework that combines structured perception-to-decision prompting, SLERP-based expert merging, and RegMoE, a regime-a
    
[^153]: LOOMSUM：编织定量与叙事证据以实现忠实的长文本-表格摘要

    LOOMSUM:Weaving Quantitative and Narrative Evidence for Faithful Long Text-Table Summarization

    [https://arxiv.org/abs/2609.00241](https://arxiv.org/abs/2609.00241)

    LOOMSUM是一个无需训练的长文本-表格摘要框架，通过提取有据可依的原子证据、显式链接表格事实与叙事分析并预先规划语篇结构，显著提升了摘要的分析忠实度，同时提出了声明级评估指标TGF。

    

    长文档常常将重要信息分布在冗长的叙事段落和多个表格中，这使得忠实的摘要生成尤为困难。现有方法可能生成各自有依据支撑的定量事实和分析性陈述，但会将它们错误地关联起来，从而产生定量上看似合理、却在分析上不忠实的摘要。在本工作中，我们提出了LOOMSUM，这是一个无需训练的框架，它提取有源可溯的原子证据，显式地将源自表格的事实与支撑性的叙事分析相链接，并在生成之前规划语篇结构。我们还引入了Table-Grounded Faithfulness（TGF，表格锚定忠实度），这是一个声明级别的指标，可分别评估数值锚定、分析支撑和关系一致性。在文本-表格摘要基准FINDSum和USTT上的实验表明，LOOMSUM在保持较强摘要质量的同时提升了分析忠实度。人工评估结果……

    arXiv:2609.00241v1 Announce Type: new  Abstract: Long documents often distribute important information across extensive narrative passages and multiple tables, making faithful summarization particularly challenging. Existing methods may generate individually supported quantitative facts and analytical statements yet associate them incorrectly, producing quantitatively plausible yet analytically unfaithful summaries. In this work, we propose LOOMSUM, a training-free framework that extracts source-grounded atomic evidence, explicitly links table-derived facts with supporting narrative analyses, and plans the discourse structure before generation. We also introduce Table-Grounded Faithfulness (TGF), a claim-level metric that separately evaluates Numeric Grounding, Analysis Support, and Relation Consistency. Experiments on the text--table summarization benchmarks FINDSum and USTT show that LOOMSUM improves analytical faithfulness while maintaining strong summarization quality. Human evalua
    
[^154]: 学习保留什么：面向多智能体大语言模型系统高效协作的门控记忆路由

    Learning What to Retain: Gated-Memory Routing for Efficient Collaboration in Multi-Agent LLM Systems

    [https://arxiv.org/abs/2609.00237](https://arxiv.org/abs/2609.00237)

    提出门控记忆路由方法，通过可学习的记忆写入门和检索门维护紧凑的执行记忆，使多智能体LLM系统的编排决策能依据有用的中间进展而非完整历史，在提升准确性的同时降低成本。

    

    基于大语言模型（LLM）的多智能体系统通过编排多个智能体的配置方式和协作方式来解决复杂推理任务。一个核心挑战是使编排能够适应不断演变的协作状态。仅基于查询的路由无法适应中间过程的进展或错误，从而损害准确性；而基于完整执行历史的路由虽然补足了缺失的上下文，却迫使后续决策处理所有先前步骤，包括冗余或低效用的步骤，造成执行历史过载并推高成本。有效的编排实际上需要一个紧凑的状态，既能捕获有用的进展，又不会积累冗余上下文。我们提出门控记忆路由，将每个决策基于查询和一个学习到的执行记忆进行条件化。一个学习到的记忆写入门仅提交非冗余的推理步骤，一个学习到的检索门为每个智能体提供紧凑且相关的信息。

    arXiv:2609.00237v1 Announce Type: new  Abstract: Large language model (LLM)-based multi-agent systems tackle complex reasoning by orchestrating how multiple agents are configured and how they collaborate. A central challenge is to adapt orchestration to the evolving collaboration state. Routing from the query alone cannot adapt to intermediate progress or errors, which hurts accuracy. Routing from the complete execution history supplies this missing context, but forces later decisions to process every prior step, including redundant or low-utility ones. This creates an execution-history overload that inflates cost. Effective orchestration instead requires a compact state that captures useful progress without accumulating redundant context. We propose Gated-Memory Routing, which conditions each decision on the query and a learned execution memory. A learned Memory Write Gate commits only non-redundant reasoning steps, and a learned Retrieval Gate supplies each agent a compact, relevant 
    
[^155]: 弥合词汇分歧：基于大语言模型辅助的高性价比零样本科学实体链接

    Bridging Lexical Divergence: LLM-Assisted, Cost-Efficient, Zero-shot Scientific Entity Linking

    [https://arxiv.org/abs/2609.00228](https://arxiv.org/abs/2609.00228)

    该论文提出Sci-ZSEL框架，通过让大语言模型有选择性地生成实体别名来控制计算成本，并结合本体感知过滤器过滤噪声，实现了低成本、无需人工标注的零样本科学实体链接。

    

    科学领域的实体链接（EL）与通用领域的实体链接不同，因为提及内容与实体名称之间往往缺乏词汇上的重叠。另一个挑战是科学领域使用专业术语，而这些术语在通用领域预训练的模型中很少出现。因此，在通用领域上训练的模型难以迁移到科学领域。为解决此问题，领域内微调是自然的补救方法。然而，许多科学领域缺乏专家标注的数据，这促使人们需要一种零人工标注的方法。现有的零样本方法严重依赖大语言模型为整个提及语料库生成别名，这会带来巨大的计算成本，而且这些方法没有提供过滤大语言模型噪声的机制。为应对这些挑战，我们提出了Sci-ZSEL框架，该框架有选择性地利用大语言模型生成实体别名以控制计算成本，并应用本体感知过滤器来减少……（注：原摘要在此处被截断）

    arXiv:2609.00228v1 Announce Type: new  Abstract: Scientific domain entity linking (EL) differs from general domain EL because mentions and entity names often lack lexical overlap. Another challenge is that specialized terminology is used in the scientific domain, which is rarely encountered in models pretrained on general domains. Therefore, models trained on general domains transfer poorly to scientific domains. To address this, in-domain fine-tuning is the natural remedy. However, many scientific domains lack expert-annotated data, motivating the need for a zero-human-annotation approach. Existing zero-shot methods heavily rely on LLMs to generate aliases across entire mention corpora, which incurs substantial computational cost, and those methods provide no mechanism to filter out noise from LLMs. To address these challenges, we propose Sci-ZSEL, a framework that selectively generates entity aliases with an LLM to control computational cost, and applies an ontology-aware filter to r
    
[^156]: LLM作为人口群体代表：社会人口学提示帮助了谁，又伤害了谁

    LLM-as-a-Demographic: Whom Sociodemographic Prompting Helps, and Whom It Hurts

    [https://arxiv.org/abs/2609.00222](https://arxiv.org/abs/2609.00222)

    研究发现，无人口学提示的LLM评判者会默认复现白人、受过大学教育群体的判断视角，而社会人口学提示的对齐效果是不对称的，主要向多数群体偏移，因此这种提示方法对某些群体有益，却可能损害其他群体的代表性。

    

    大语言模型（LLM）越来越多地被用作主观任务的评判者，在这类任务中标注者之间存在分歧，因此相关的问题不仅在于评判者有多准确，还在于它复现了谁的判断。社会人口学提示（sociodemographic prompting）通过将标注者的人口学特征作为条件输入，使评判者的判断与相应群体的判断保持一致。我们测试了这种对齐是否能在分布层面成立，将23个开放权重LLM在三个主观任务上的预测标签分布与真实标注者群体的分布进行比较，并设置了三种条件：不提供人口学信息、单属性画像，以及基于性别、年龄、种族和教育程度的交叉属性画像。研究得出三个发现。首先，未使用人口学提示的评判者并非视角中立：模型最能复现白人、受过大学教育的标注者的判断。其次，人口学条件化是不对称的：它使评判者向多数群体靠拢，并……（原文摘要在此处截断）

    arXiv:2609.00222v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used as judges for subjective tasks, where annotators disagree and the relevant question is not only how accurate a judge is, but whose judgments it reproduces. Sociodemographic prompting conditions the judge on an annotator's demographic profile to align its judgments with the corresponding group's. We test whether this alignment emerges distributionally, comparing the predicted label distributions of 23 open-weight LLMs on three subjective tasks against those of real annotator groups, under three conditions: no demographic information, single-attribute profiles, and intersectional profiles over gender, age, race, and education. Three findings emerge. First, a judge prompted with no demographics is not perspective-neutral: models best reproduce the judgments of White, college-educated annotators. Second, demographic conditioning is asymmetric: it moves the judge toward majority groups and aw
    
[^157]: 揭示并缓解多奖励强化学习中由聚合引起的奖励劫持

    Uncovering and Mitigating Aggregation-Induced Reward Hacking in Multi-Reward Reinforcement Learning

    [https://arxiv.org/abs/2609.00213](https://arxiv.org/abs/2609.00213)

    本文揭示了多奖励强化学习微调中固定权重聚合会诱发奖励劫持、使策略陷入次优奖励配置的问题，并提出轻量级的在线方法——自适应多奖励投影（AMRP），通过动态重新分配聚合权重来缓解该问题。

    

    大型语言模型的强化学习微调日益采用多个奖励维度，包括可验证规则、任务特定评估器以及学习得到的奖励模型，以便为多样化能力提供更丰富的监督信号。这些维度通常通过固定的聚合权重进行标量化。我们发现了一种失效模式，即聚合本身会诱发奖励劫持（reward hacking）：静态投影会将性质不同的奖励配置混叠为单一标量，使优化偏向于那些最容易、最密集或被奖励信号系统性偏好的维度。随着训练的进行，这会使策略陷入次优的奖励配置，并阻碍其收敛到能够带来更高任务性能的更均衡的奖励配置。为解决这一问题，我们提出了自适应多奖励投影（AMRP），这是一种轻量级的在线方法，利用三种信号——相对缺口、奖励波动性……（摘要原文在此处截断）

    arXiv:2609.00213v1 Announce Type: new  Abstract: Reinforcement learning fine-tuning of large language models increasingly adopts multiple reward dimensions, including verifiable rules, task-specific evaluators, and learned reward models, to provide richer supervision across diverse capabilities. These dimensions are commonly scalarized with fixed aggregation weights. We identify a failure mode in which aggregation itself induces reward hacking: static projection aliases qualitatively different reward profiles into a single scalar, steering optimization toward whichever dimensions are easiest, densest, or systematically favored by the reward signal. Over training, this traps the policy in suboptimal profiles and prevents convergence to better-balanced ones that would yield higher task performance. To address this, we propose Adaptive Multi-Reward Projection (AMRP), a lightweight online method that reallocates aggregation weights using three signals, relative shortfall, reward volatility
    
[^158]: 大语言模型驱动的自动驾驶汽车继承了人类驾驶员在行人让行方面的偏见：来自新基准的结果与启示

    LLM-Driven Autonomous Vehicles Inherit Human Driver Biases in Pedestrian Yielding: Results and Implications From A New Benchmark

    [https://arxiv.org/abs/2609.00192](https://arxiv.org/abs/2609.00192)

    本文提出两种新的偏见测试方法（“其他条件相同”测试和“自我一致性”测试），并发现大语言模型和视觉语言模型驱动的自动驾驶汽车在行人让行决策中会继承人类驾驶员的偏见，其决策受到行人性别、种族、宗教、残障状况和年龄等因素的影响。

    

    公众对自动驾驶汽车的信任可能不仅取决于技术上的成功，还取决于其决策的公平性。虽然自动驾驶研究中的一个新趋势是使用通用的“常识”模型来指导自动驾驶汽车的决策，但这些模型在多大程度上继承了人类驾驶员的偏见仍未得到充分研究。鉴于心理学研究表明人类驾驶员偏见确实存在，例如在美国，驾驶员对黑人行人的让行率较低，我们认为模型偏见分析也应成为自动驾驶汽车评估的一部分。具体而言，本文提出了两种针对大语言模型和视觉语言模型的新偏见测试方法——“其他条件相同”测试和“自我一致性”测试——以评估行人让行决策中的偏见。我们的研究结果表明，大语言模型和视觉语言模型做出的让行决策都会受到行人性别、种族、宗教、残障状况、年龄等因素的影响。

    arXiv:2609.00192v1 Announce Type: new  Abstract: Public trust in Autonomous Vehicles (AVs) may depend not only on technical success but also on the fairness of their decision making. While a recent trend in AV research involves using general purpose "common sense" models to guide AV decision making, the degree to which these inherit human biases in driving is still understudied. Given that psychology studies have shown human driver biases exist, such as lower pedestrian-yielding rates to Black pedestrians in the US, we argue that analyses of model bias should also be part of AV evaluation. Concretely, in this paper we propose two new bias testing methodologies for Large Language Models (LLMs) and Visual-Language Models (VLMs)-"All Else Being Equal" tests and "Self-Consistency" tests-in order to assess bias in pedestrian-yielding decisions. Our findings show that both LLMs and VLMs make yielding decisions which are influenced by pedestrian gender, ethnicity, religion, disability, age, s
    
[^159]: 阿拉伯语危机热线来电中的自杀风险评估：阿拉伯语与英语大语言模型的比较

    Assessing Suicide Risk in Arabic Crisis Helpline Calls: A Comparison of Arabic and English Large Language Models

    [https://arxiv.org/abs/2609.00191](https://arxiv.org/abs/2609.00191)

    该研究首次在真实阿拉伯语危机热线数据的严格隐私约束下，比较了阿拉伯语与英语大语言模型在自杀风险评估中的表现，填补了阿拉伯语热线自然语言处理研究的空白。

    

    危机热线通过结构化访谈评估自杀风险，这一过程缓慢且依赖于接线员的培训水平和工作量。自然语言处理可以支持风险评估和来电优先级排序，但几乎没有研究针对阿拉伯语热线电话，或在真实热线数据的隐私限制下开展相关工作。我们分析了来自黎巴嫩国家情感支持与自杀预防生命热线的去标识化转录文本。音频从未离开热线机构：来电在本地使用面向黎凡特阿拉伯语的语音识别模型进行转录，并由阿拉伯语命名实体识别模型在本地删除身份识别信息，只有去标识化的转录文本被共享给研究团队。接线员记录了哥伦比亚自杀严重程度评定量表（C-SSRS）中的五个自杀意念条目，我们将其合并为两个二元结果：有风险和高风险。我们还将转录文本进行了机器翻译……（原文摘要到此截断）

    arXiv:2609.00191v1 Announce Type: cross  Abstract: Crisis helplines assess suicide risk through structured interviews, a process that is slow and dependent on operator training and workload. Natural language processing could support risk assessment and call prioritization, but almost no work addresses Arabic-language helpline calls or operates within the privacy constraints of real helpline data. We analysed de-identified transcripts from Lebanon's National Lifeline for Emotional Support and Suicide Prevention. Audio never left the helpline: calls were transcribed on site with a speech recognition model for Levantine Arabic, and an Arabic named-entity recognition model removed identifying information locally. Only the de-identified transcripts were shared with the research team. Operators recorded the five suicidal ideation items of the Columbia Suicide Severity Rating Scale, which we combined into two binary outcomes: at-risk and high-risk. We also machine-translated the transcripts i
    
[^160]: 面向大语言模型时序评估与知识更新的合成世界

    Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs

    [https://arxiv.org/abs/2609.00184](https://arxiv.org/abs/2609.00184)

    该论文提出了一个模拟驱动的合成框架，通过虚构未来世界的 ParallelEvents 基准避免评估污染，并利用 Synapse 训练框架（结合中期训练与指令微调）实现大语言模型的可扩展知识更新，性能比现有方法提升 14.23%。

    

    大语言模型（LLM）依赖于静态的预训练语料库，导致其知识随时间推移而变得过时。现有的知识编辑评估方法要么容易遭受快速的数据污染，要么依赖于与现有刚性知识相冲突的反事实编辑。在本工作中，我们提出了一个合成的、模拟驱动的框架，用于研究大语言模型中的知识插入。我们引入了 {\sc ParallelEvents}，这是一个由虚构但逼真的未来世界构成的基准，能够生成连贯的事件轨迹以进行受控评估，在避免污染的同时保持一致性。基于该数据集，我们开发了 {\sc Synapse}，这是一个利用模型自身生成的数据、通过中期训练（mid-training）和指令微调来更新模型参数的训练框架。这一合成流程实现了可扩展的知识整合，而无需昂贵的人工策划数据。实验结果表明，{\sc Synapse} 的性能比现有方法高出 14.23%。

    arXiv:2609.00184v1 Announce Type: new  Abstract: Large language models (LLMs) rely on static pretraining corpora, causing their knowledge to become outdated over time. Existing approaches for evaluating knowledge edits either suffer from rapid contamination or rely on counterfactual edits that conflict with rigid existing knowledge. In this work, we propose a synthetic, simulation-driven framework for studying knowledge insertion in LLMs. We introduce {\sc ParallelEvents}, a benchmark of fictional yet realistic future worlds that generates coherent event trajectories for controlled evaluation, avoiding contamination while preserving consistency. Building on this dataset, we develop {\sc Synapse}, a training framework that uses model-generated data to update model parameters via mid-training and instruction tuning. This synthetic pipeline enables scalable knowledge integration without costly human-curated data. Empirically, {\sc Synapse} outperforms existing methods by 14.23\%, demonstr
    
[^161]: 通用自然语言处理嵌入模型能否捕捉本体推理？

    Do General NLP Embeddings Capture Ontological Reasoning?

    [https://arxiv.org/abs/2609.00177](https://arxiv.org/abs/2609.00177)

    本文提出AVA评估框架，通过来自163个异构本体的171,007个对比三元组系统评估发现，现有最先进的NLP嵌入模型难以区分本体中对逻辑敏感的关系语义（最佳模型三元组准确率仅0.739），且微调带来的提升难以有效迁移到语义网下游任务。

    

    通用自然语言处理嵌入模型在语言任务上表现出色，但其捕捉符号化本体结构的能力仍不清楚。我们提出了AVA，一个系统性的评估框架，用于评估嵌入模型能否区分本体和知识图谱中对逻辑敏感的关系语义。AVA包含171,007个对比三元组，这些三元组通过层次反转、关系替换和不相交注入的方法从163个异构本体中构建。每个三元组包含一个本体陈述、一个语义等价的改写，以及一个具有矛盾关系含义的对逻辑敏感的困难负样本。我们评估了超过25个最先进的嵌入模型，发现了显著的局限性：最佳模型仅达到0.739的三元组准确率，而困难负样本的准确率更是降至0.135。微调可以大幅提升判别能力，但在包括分类体系在内的下游语义网任务上的迁移效果不佳。

    arXiv:2609.00177v1 Announce Type: cross  Abstract: General-purpose NLP embedding models perform well on linguistic tasks, but their ability to capture symbolic ontological structure remains unclear. We introduce AVA, a systematic framework for evaluating whether embeddings distinguish logic-sensitive relational semantics in ontologies and knowledge graphs. AVA comprises 171,007 contrastive triplets derived from 163 heterogeneous ontologies using hierarchy inversion, relation substitution, and disjointness injection. Each triplet contains an ontology statement, a semantically equivalent paraphrase, and a logic-sensitive hard negative with contradictory relational meaning. We evaluate more than 25 state-of-the-art embedding models and find substantial limitations: the best model achieves only 0.739 triplet accuracy, while hard negative accuracy falls to 0.135. Fine-tuning improves discrimination by a large margin but transfers poorly to downstream Semantic Web tasks, including taxonomy d
    
[^162]: 通用语还是探测假象？重新思考多语言大语言模型中的潜在语言

    Lingua Franca or Probing Artifact? Rethinking Latent Language in Multilingual LLMs

    [https://arxiv.org/abs/2609.00155](https://arxiv.org/abs/2609.00155)

    该研究发现不同的潜在语言探测方法会得出系统性不一致的结论，表明多语言大模型通过英语等“潜在通用语”路由计算的说法可能更多取决于探测手段的选择，而非模型本身固有的计算机制。

    

    潜在语言识别常被用来论证多语言语言模型通过语言特定状态（如英语枢纽）来路由计算。然而，现有探测方法从不同信号推断潜在语言，例如隐藏状态的几何结构，或可从中间表示中解码出的内容。由于这类论断会影响关于模型如何跨语言共享和路由信息的结论，我们追问：这些探测方法测量的究竟是同一现象，还是揭示了多语言计算的不同侧面？我们在多种模型家族、训练方式、领域、任务、检查点以及多达27种语言上研究了这一问题。我们发现，各类识别探测方法存在系统性的不一致：基于GMM的表示探测方法从隐藏状态几何结构中获取证据，显示出更早出现的跨语言混合；而依赖输出空间可解码性的解码式探测方法，则保留了更鲜明的语言特定性，以及

    arXiv:2609.00155v1 Announce Type: cross  Abstract: Latent language identification is often used to argue that multilingual language models route computation through language-specific states, such as English pivots. However, existing probes infer latent language from different signals, such as the geometry of hidden states or what can be decoded from intermediate representations. Since such claims shape conclusions about how models share and route information across languages, we ask whether these probes measure the same phenomenon or expose distinct aspects of multilingual computation. We study this question across model families, training regimes, domains, tasks, checkpoints, and up to 27 languages. We find that identification probes systematically disagree: the GMM-based representation probe, which draws evidence from hidden state geometry, shows earlier cross-lingual mixing, whereas decoding-based probes, which rely on output-space decodability, retain sharper language-specific and 
    
[^163]: 先答后判式LLM评判会继承评判者自身的错误

    Commit-first LLM judging inherits the judge's own errors

    [https://arxiv.org/abs/2609.00088](https://arxiv.org/abs/2609.00088)

    研究发现“先答后判”式LLM评判会继承评判者自身的错误，而对八个主流评估框架的审计表明无一真正实现该方法，其中九个框架因复制同一祖先提示词而采用了已被证明无效的变体，导致大量错误代码被放行。

    

    LLM评判器（即对另一个系统输出进行打分的模型）可能被被其评分的系统“钻空子”。近期研究指出了一种确实有效的防御方法：评判器先自行解决任务并固定自己的答案，然后仅当候选答案与其一致时才予以接受。我们将这一做法称为“先答后判”评判，并探究已发布的软件是否实现了该方法，以及其代价是什么。我们审计了八个广泛使用的评估框架的默认评判器配置：在纳入范围的24个配置中，没有一个实现了该方法；其中九个实现了文献中被测得无效的一种变体，并且共享同一个祖先提示词——这一点可以通过一个被复制下来的排版错误进行追溯。在一项受控实验中，一个普通的、无法访问正确答案的best-of-N搜索，严格按照文档说明使用其中一个配置来优化代码。在一个区间合并任务上，该评判器在一个随机种子下接受了96个候选中的90个，在另一个种子下接受了93个；每个被接受的候选……（原文摘要在此截断）

    arXiv:2609.00088v1 Announce Type: cross  Abstract: LLM judges, models that score another system's output, can be gamed by the systems they score. Recent work identifies one defence that works: the judge solves the task itself first and commits to that answer, then accepts a candidate only if the two match. We call this commit-first judging, and ask whether shipped software implements it, and what it costs.   We audit the default judge configurations of eight widely used evaluation frameworks. Of the 24 configurations in scope, none implement it. Nine implement a variant the literature measures as ineffective, and share one ancestor prompt, traceable through a copied typographical error.   In a controlled experiment, an ordinary best-of-N search with no access to correct answers optimises code against one of these configurations, used exactly as documented. On an interval merging task the judge accepted 90 of 96 candidates in one seed and 93 of 96 in the other; every accepted candidate 
    
[^164]: 检索、评分与解码如何塑造基于大语言模型的对话推荐系统的性能与稳定性

    Retrieval, Scoring, and Decoding Shape Performance and Stability in LLM-based Conversational Recommendation

    [https://arxiv.org/abs/2609.00086](https://arxiv.org/abs/2609.00086)

    该研究系统评估了大语言模型作为对话推荐重排序器的表现，发现在统一候选池协议下最佳专有LLM仅小幅超越传统基线，自由生成评估会夸大其优势，且所有开源LLM均未超过调优的浅层自编码器基线，说明检索、评分与解码协议显著影响LLM在对话推荐中的表现。

    

    大语言模型（LLM）越来越多地被用作对话推荐系统中的重排序器，然而其测得的收益在很大程度上取决于检索与推理协议。在ReDial对话式电影推荐基准上，我们在一个共享的“先检索后重排序”流水线中，比较了专有模型、开源权重模型以及微调的LLM重排序器与协同过滤和序列推荐基线，并改变了候选池大小、第一阶段检索器和解码温度。在共享的语义top-250候选池和严格的候选感知评分条件下，最佳的专有重排序器达到NDCG@10为0.1497，而最强的非LLM基线为0.0939。同一重排序器在零样本生成模式下达到0.2925，这表明无约束的评分可以产生比匹配候选池评估大得多的表面优势。在该协议下，没有任何被评估的开源权重LLM优于经过调优的浅层自编码器基线。

    arXiv:2609.00086v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as rerankers in conversational recommender systems, yet measured gains depend strongly on the retrieval and inference protocol. On the ReDial conversational movie recommendation benchmark, we compare proprietary, open-weight, and fine-tuned LLM rerankers with collaborative-filtering and sequential baselines in a shared retrieve-then-rerank pipeline. We vary candidate-pool size, first-stage retriever, and decoding temperature. With a shared semantic top-250 candidate pool and strict candidate-aware scoring, the best proprietary reranker reaches NDCG@10 of 0.1497, compared with 0.0939 for the strongest non-LLM baseline. The same reranker reaches 0.2925 in zero-shot generation, showing that unconstrained scoring can yield a much larger apparent advantage than matched-pool evaluation. No evaluated open-weight LLM outperforms the tuned shallow autoencoder baseline under this protocol. For t
    
[^165]: KItCAT：通过输入破坏进行知识注入的自回归训练方法

    KItCAT: Knowledge Injection via Input Corruption for Auto-regressive Training

    [https://arxiv.org/abs/2609.00082](https://arxiv.org/abs/2609.00082)

    提出KItCAT轻量级训练策略，通过在下一词预测训练中对输入序列进行随机破坏，从而在无需昂贵改写的情况下，将小众专业知识有效注入仅解码器大语言模型。

    

    大语言模型（LLM）在预训练期间获取了大量知识，但往往缺乏回答来自小众来源（如预训练时未见过的手册或技术文档）问题所需的专业知识。持续预训练（CPT）被广泛用于将这类知识注入模型参数，然而小众文档很少重复相同的事实，这使得CPT难以稳健地获取此类知识。近期的工作通过生成新知识的多个改写版本来解决这一问题，但改写计算成本高昂，且通常需要强大的大语言模型。在本工作中，我们提出了KItCAT：通过破坏的自回归训练进行知识注入，这是一种轻量级训练策略，可减少仅解码器大语言模型对改写的需求。KItCAT通过对输入序列进行随机破坏来增强标准的下一词预测。在训练过程中，输入词元的随机子集会被替换为词表中的其他词元。

    arXiv:2609.00082v1 Announce Type: cross  Abstract: LLMs acquire vast amounts of knowledge during pre-training, but often lack the specialized knowledge needed to answer questions from niche sources such as manuals or technical documents unseen during pre-training. Continued pre-training (CPT) is widely used to inject such knowledge into model parameters. However, niche documents seldom repeat facts, making it difficult for CPT to robustly acquire such knowledge. Recent works address this by generating multiple paraphrases of the new knowledge, but paraphrasing is computationally expensive and typically requires powerful LLMs. In this work, we introduce KItCAT: Knowledge Injection via Corrupted Auto-regressive Training, a lightweight training strategy that reduces the need for paraphrasing in decoder-only LLMs. KItCAT augments standard next-token prediction by stochastically corrupting the input sequence. During training, a random subset of input tokens is replaced with other vocabulary
    
[^166]: 差异之下：诊断与缓解代码级自主研究循环中的算法模式坍缩

    Beneath the Diff: Diagnosing and Mitigating Algorithmic Mode Collapse in Code-Level Autonomous Research Loops

    [https://arxiv.org/abs/2609.00077](https://arxiv.org/abs/2609.00077)

    论文系统性地诊断出代码级自主研究循环中一种名为“算法模式坍缩”的失效模式——即表层编辑多样性看似稳定但算法层面的语义与机制多样性已经坍缩，并提出了相应的缓解方法。

    

    代码级自主研究循环最近成为自动化机器学习研究中一个具体的研究对象。在此类循环中，大语言模型智能体对实验训练流程提出修改建议，执行修改后的流程，并保留能够提升可验证的循环内指标的修改。尽管可执行的指标看似能提供可靠的进展信号，但目前尚不清楚这种重复的、由指标驱动的代码编辑是否能带来超越循环本身的真正泛化改进。我们对这一问题进行了系统性诊断。在多种实验设置中，我们发现了一种稳健的失效模式，我们称之为“算法模式坍缩”。在这种状态下，表层的编辑多样性保持稳定，但语义层面与机制层面的多样性发生坍缩：智能体持续编辑不同的代码行，却反复提出相同类型的算法修改。这种坍缩伴随着

    arXiv:2609.00077v1 Announce Type: new  Abstract: Code-level autonomous research loops (ARLs) have recently emerged as a concrete object of study in automated machine learning research. In such loops, an LLM agent proposes modifications to an experimental training pipeline, executes the modified pipeline, and retains edits that improve a verifiable in-loop metric. Although executable metrics may appear to provide a reliable signal of progress, it remains unclear whether repeated metric-driven code editing leads to genuine improvements that generalize beyond the loop. We provide a systematic diagnosis of this question. Across various experiment settings, we identify a robust failure mode that we call \textbf{algorithmic mode collapse}. In this regime, surface-level edit diversity remains stable, but semantic and mechanism-level diversity collapse: the agent continues to edit different lines of code while repeatedly proposing the same kinds of algorithmic changes. This collapse is accompa
    
[^167]: MiNER：面向临床文本中疟疾疾病实体识别的微调生物医学自然语言处理

    MiNER: Fine-Tuned Biomedical Natural Language Processing for Malaria Disease Entity Recognition in Clinical Texts

    [https://arxiv.org/abs/2609.00073](https://arxiv.org/abs/2609.00073)

    本文提出MiNER方法，通过对预训练生物医学语言模型BioBERT进行微调，实现疟疾临床文本中疾病实体的自动识别，从而从海量疟疾科学文献中高效提取具有临床意义的生物医学信息。

    

    疟疾仍然是一个重大的全球健康负担，需要持续的研究努力来理解其复杂的分子机制、流行病学以及潜在的治疗干预手段。从庞大且不断增长的疟疾文献中提取关键的生物医学信息是一项具有挑战性的任务，需要创新的方法。近年来，预训练语言模型彻底改变了自然语言处理任务，在各个领域展现出卓越的能力。本文提出了一种经过微调的预训练生物医学语言模型，用于从疟疾疾病的科学文献中进行生物医学信息抽取。该方法首先选取并预处理了大规模的疟疾科学文章语料库，然后使用具有临床意义的实体对其进行标注，进而利用最先进的预训练语言模型BioBERT，将文本数据编码为上下文感知的表示。

    arXiv:2609.00073v1 Announce Type: new  Abstract: Malaria remains a significant global health burden, necessitating continuous research efforts to understand its complex molecular mechanisms, epidemiology, and potential therapeutic interventions. Extracting essential biomedical information from the vast and constantly growing malaria literature is a challenging task that demands innovative approaches. Recently, pre-trained language models have revolutionized natural language processing tasks, demonstrating remarkable capabilities in various domains. This paper proposes a fine-tuned pre-trained biomedical language model for biomedical information extraction from scientific literature on malaria disease. The proposed methodology selects and preprocesses a large corpus of scientific articles on malaria, and then annotates them with entities of clinical significance. It then leverages BioBERT, a state-of-the-art pre-trained language model, to encode the textual data into context-aware repre
    
[^168]: 审计自我改进智能体中的框架篡改行为

    Auditing Harness Tampering in Self-Improving Agents

    [https://arxiv.org/abs/2609.00069](https://arxiv.org/abs/2609.00069)

    该论文提出了“框架篡改”概念及其双轴分类体系，通过构建带标注的篡改语料库并对审计方法进行基准测试，系统研究并检测自我改进智能体对自身框架的不当修改。

    

    自我改进智能体会迭代地修改自身的运行框架以突破其性能边界。然而，这类修改可能产生虚幻的性能提升，或者在不真正提升能力的情况下损害授权、溯源和完整性等完整性约束。我们将这种现象称为框架篡改，它将奖励篡改和测量篡改的概念扩展到了完整的自我改进生命周期。为了系统地研究这一问题，我们提出了一个双轴分类法，根据篡改编辑发生的框架功能角色以及其违反的义务来对每次失准编辑进行分类。随后，我们通过向自我改进智能体的真实轨迹中植入篡改-良性编辑对来构建带标注的语料库。我们对多种审计方法进行了适配，并在篡改分类和定位任务上进行了基准测试。最后，我们系统地审计了自我改进智能体的真实轨迹。结果表明……

    arXiv:2609.00069v1 Announce Type: cross  Abstract: Self-improving agents iteratively modify their own harness to push the frontier of their performance. However, such modifications can produce illusory performance gains or compromise integrity constraints such as authorization, provenance, and completeness without genuinely improving capability. We term this phenomenon as harness tampering, which extends the concept from reward and measurement tampering to the full self-improvement lifecycle. To systematically study this problem, we propose a two-axis taxonomy that categorizes each misaligned edit by the harness functional role in which it occurs and the obligation it violates. Then we build an annotated corpus by seeding tampered-benign edit pairs into the real trajectories of self-improving agents. We adapt and benchmark diverse audit methods on tampering classification and localization tasks. Finally we systematically audit real trajectories of self-improving agents. The results dem
    
[^169]: 生命算子：一种用于多尺度生命建模的自演化框架

    Life Operators: a self-evolving framework for multiscale life modelling

    [https://arxiv.org/abs/2609.00068](https://arxiv.org/abs/2609.00068)

    该论文提出“生命算子”自演化框架，通过感知、演化、生成三类任务约束映射算子及桥接算子，为多尺度生命建模提供了统一框架，能够表示患者状态、耦合不同尺度并支持对失效假设的修正。

    

    医疗人工智能正从识别任务走向临床对话与纵向预测。然而一个核心问题仍然悬而未决：患者的状态在干预之下会如何变化？统计模型学习对未来的观测，而机理模型描述的是被选取的特定过程，二者都无法提供一个统一框架来表示患者状态、耦合不同尺度或修正失效的假设。我们提出生命算子：这是一类具有任务边界的映射，定义了三种科学角色。感知算子从多模态观测中推断与任务相关的生物状态；演化算子在自然动力学或干预条件动力学下传播这些状态；生成算子将这些状态映射为可测量的信号。每种角色都可以由方程、统计模型、神经网络或其混合形式来实现。桥接算子负责连接具有不同变量、尺度和时间步长的组件。所选定的算子与桥接算子共同构成面向特定任务的……

    arXiv:2609.00068v1 Announce Type: cross  Abstract: Medical AI is moving beyond recognition towards clinical dialogue and longitudinal prediction. Yet a central question remains: how would a patient's state change under intervention? Statistical models learn future observations, whereas mechanistic models describe selected processes. Neither provides a common framework for representing patient state, coupling scales or revising failed assumptions. We propose Life Operators: task-bounded mappings that define three scientific roles. Perception operators infer task-relevant biological states from multimodal observations, Evolution operators propagate these states under natural or intervention-conditioned dynamics, and Generation operators map them to measurable signals. Each role may be realised by equations, statistical models, neural networks or hybrids. Bridge operators connect components with different variables, scales and time steps. Selected operators and bridges form task-specific 
    
[^170]: 多模态大语言模型是先看后读吗？诊断情境性谄媚现象

    Do Multimodal LLMs See Before They Read? Diagnosing Contextual Sycophancy

    [https://arxiv.org/abs/2609.00067](https://arxiv.org/abs/2609.00067)

    该论文诊断了多模态大语言模型易受外部文本误导而忽视冲突图像证据的“多模态情境性谄媚”问题，并提出“系统2视觉仲裁”（S2VA）方法，通过让视觉证人在读取文本前先独立判断，在六个模型上将准确率显著提升19.7至44.1分。

    

    外部文本可以覆盖多模态大语言模型中与之冲突的图像证据，我们将这种失败称为“多模态情境性谄媚”。我们引入了一个包含998个案例的诊断方法，该方法独立地变化视觉证据、常识先验和外部文本三个因素，并通过围绕“情境盲视”的视觉证人调整信息边界，来探究这种失败在何时发生。在与Gemini生成的虚假文本配对的异常图像上，GPT-5.1在联合条件下的得分仅为7.9%；当直接对情境盲视的证人报告进行评分时，得分为49.7%；在使用匹配的双调用证人-仲裁者管道（即让证人接触文本）时，得分为63.7%；而在“系统2视觉仲裁”（S2VA，即对证人隐瞒文本）下，得分达到84.2%。在六个模型上，S2VA相比直接证人报告提升了19.7至44.1分，且所有配对的95%置信区间均不包含零。最佳的信息边界并非统一不变：文本情境对某些情况……

    arXiv:2609.00067v1 Announce Type: cross  Abstract: External text can override conflicting image evidence in multimodal large language models, a failure we call multimodal contextual sycophancy. We introduce a 998-case diagnostic that independently varies visual evidence, commonsense priors, and external text, and probe when this failure arises by moving the information boundary around a context-blind visual witness. On abnormal images paired with Gemini-generated false text, GPT-5.1 scores 7.9% under joint conditioning, 49.7% when the context-blind witness report is scored directly, 63.7% under a matched two-call witness-arbiter pipeline that exposes the witness to the text, and 84.2% under System-2 Visual Arbitration (S2VA), which withholds the text from the witness. Across six models, S2VA improves over the direct witness report by 19.7 to 44.1 points, with all paired 95% confidence intervals excluding zero. The best information boundary is not uniform: textual context scaffolds some
    
[^171]: OCGQuant：面向NVFP4量化的异常值伴随分组方法

    OCGQuant: Outlier-Companion Grouping for NVFP4 Quantization

    [https://arxiv.org/abs/2609.00066](https://arxiv.org/abs/2609.00066)

    提出OCGQuant，一种以“异常值伴随分组（OCG）”为核心的NVFP4训练后量化方法，通过自适应地将异常值通道与伴随通道分组，减少由块最大值主导缩放因子所造成的“附带量化误差”，从而在不引入额外计算的前提下提升低比特推理的量化精度。

    

    NVFP4是一种面向低比特推理的高效微缩放（microscaling）格式，但激活异常值仍会降低NVFP4块内的量化精度。在每个量化块内，较大的激活值会主导块缩放因子，从而增大共享同一缩放因子的其余数值的量化误差。现有的训练后量化（PTQ）方法通过混合精度、旋转或残差补偿等策略来缓解异常值带来的误差，但这些方法要么并非专门针对NVFP4设计，要么会引入额外的计算开销。在本工作中，我们从通道分组的视角重新审视NVFP4，并将由块最大值所设定的缩放因子下其余块内数值产生的可减少误差定义为“附带量化误差”。基于这一洞察，我们提出了OCGQuant——一种以异常值伴随分组（Outlier-Companion Grouping, OCG）为核心的训练后量化方法，该方法自适应地将异常值通道与……（原文摘要在此处截断）

    arXiv:2609.00066v1 Announce Type: cross  Abstract: NVFP4 is an efficient microscaling format for low-bit inference, but activation outliers can still degrade quantization accuracy within NVFP4 blocks. Within each quantization block, large activations can dominate the block scale, increasing the quantization error of the remaining values sharing the same scale. Existing post-training quantization (PTQ) methods mitigate outlier errors through strategies such as mixed precision, rotation, or residual compensation, but these approaches are either not specifically tailored to NVFP4 or introduce additional computation. In this work, we revisit NVFP4 from a channel-grouping perspective and define the reducible error incurred by remaining block values under the scale set by the block maximum as Collateral Quantization Error. Based on this insight, we propose OCGQuant, a post-training quantization method centered on Outlier-Companion Grouping (OCG), which adaptively pairs outlier channels with 
    
[^172]: 科学智能体技能：面向科研智能体的程序性知识库

    Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents

    [https://arxiv.org/abs/2609.00065](https://arxiv.org/abs/2609.00065)

    该论文提出了一个名为“科学智能体技能”的开放库，收录了基因组学、化学信息学等16个科研实践领域共163项程序性知识，使语言模型智能体能够遵循领域规范做出站得住脚的科学分析，而非仅仅返回能运行的代码。

    

    被要求分析实验的语言模型智能体通常只会返回一段能运行的代码，但该分析是否站得住脚则是另一回事。一个站得住脚的分析取决于程序性选择：该领域接受哪种统计检验方法、哪个标识符命名空间是权威的、以及结果必须附带哪些注意事项。我们提出了“科学智能体技能”，这是一个开放的知识库，包含16个实践领域的163项此类程序，涵盖基因组学、化学信息学、医学影像、研究设计和科学传播等。每项技能都是一个目录，围绕一个版本化、人类可读的指令文件构建。智能体仅在任务需要时才加载该文件；目录中通常还包含参考资料和可运行的脚本。我们未报告任务级评估结果和宿主选择率。该库采用开放许可证，可在 https://github.com/K-Dense-AI/scientific-agent-skills 获取。

    arXiv:2609.00065v1 Announce Type: cross  Abstract: A language-model agent asked to analyse an experiment will usually return working code. Whether the analysis is defensible is a different question. A defensible analysis depends on procedural choices: which test the field accepts, which identifier namespace is authoritative, and which caveats must accompany a result. We present Scientific Agent Skills, an open library of 163 such procedures in 16 areas of practice, including genomics, cheminformatics, medical imaging, study design and scientific communication. Each skill is a directory built around a versioned, human-readable instruction file. An agent loads the file only when a task calls for it; the directory often also contains reference material and runnable scripts. We report no task-level evaluation and no host selection rate. Openly licensed and available at https://github.com/K-Dense-AI/scientific-agent-skills.
    
[^173]: 注意力敏感性并不足够：在微调下解耦注意力层面与行为层面的上下文学习

    Attention Sensitivity Is Not Enough: Dissociating Attention-Level and Behavioural In-Context Learning under Fine-Tuning

    [https://arxiv.org/abs/2609.00064](https://arxiv.org/abs/2609.00064)

    该论文形式化了注意力层面的“上下文敏感性”（ICS）指标，并通过Llama-2-7B上的四臂消融实验证明，最大化ICS并不能保留真实的行为性上下文学习能力（ICL-GAP接近零且MMLU从0.371降至0.279），揭示了注意力代理指标与行为层面ICL之间的“古德哈特定律”式解耦。

    

    上下文学习（ICL）使大型语言模型能够通过示例适应新任务，而微调可能会削弱这种行为。许多保持性诊断方法依赖检查注意力：如果注意力随示例的变化而变化，模型就被视为对上下文敏感。本文探讨这种代理指标在被优化之后能在多大程度上被信任。我们形式化了“上下文敏感性”（ICS），即在匹配与不匹配示例前缀上最后一个token注意力分布之间的平均行距离，并将其与“ICL差距”（ICL-GAP）配对，后者衡量相同前缀之间的行为准确率差距。在Llama-2-7B上进行的受控四臂消融实验中，一个最大化ICS的正则化器（armKL）将ICS推高至1.413，达到其几何上限的0.5%以内。然而行为层面的读数讲述了不同的故事：ICL-GAP保持在接近零的水平，MMLU准确率从0.371下降至0.279，这是有界注意力代理指标的“古德哈特式”解耦。端点统计定位……

    arXiv:2609.00064v1 Announce Type: cross  Abstract: In-context learning (ICL) lets large language models adapt to new tasks from demonstrations, and fine-tuning can erode this behaviour. Many preservation diagnostics inspect attention: if attention changes when demonstrations change, the model is treated as context-sensitive. This paper asks how far that proxy can be trusted once it is optimised. We formalise \emph{In-Context Sensitivity} (ICS), the average row distance between last-token attention on matched and mismatched demonstration prefixes, and pair it with \emph{ICL-GAP}, the behavioural accuracy gap between the same prefixes. In a controlled four-arm ablation on Llama-2-7B, an ICS-maximising regulariser ($\armKL$) drives ICS to $1.413$, within $0.5\%$ of its geometric ceiling. The behavioural readout tells a different story: ICL-GAP stays near zero and MMLU accuracy moves from $0.371$ to $0.279$, a Goodhart dissociation of the bounded attention proxy. Endpoint statistics locate
    
[^174]: 基于大语言模型的医学因果假设验证

    Medical Causal Hypothesis Verification with Large Language Models

    [https://arxiv.org/abs/2609.00063](https://arxiv.org/abs/2609.00063)

    本文提出了一个医学因果假设验证的评估框架，并评估了八个大语言模型利用科学文献证据验证17个医学因果假设的能力。

    

    大语言模型在搜索和信息检索中的应用日益增多，这凸显了评估其在医疗保健等高风险领域可靠性的必要性。尽管大语言模型能够有效回答关于疾病、症状和治疗的问题，但其准确评估因果关系并将结论建立在经过验证的科学证据之上的能力仍不明确。本文提出了一项初步的小规模研究，调查了大语言模型在评估医学因果性论断并用同行评审研究加以支持方面的准确性。我们提出了一个因果假设验证的评估框架，可用于系统地跟踪现有和未来大语言模型的表现。我们评估了八个大语言模型在17个医学因果假设上的表现，以检验它们能否利用文献中的科学证据可靠地验证这些假设。我们对科学文献进行了系统性标注……

    arXiv:2609.00063v1 Announce Type: cross  Abstract: The growing use of large language models (LLMs) for search and information retrieval underscores the need to evaluate their reliability in high-stakes domains such as healthcare. Although LLMs can effectively answer questions about diseases, symptoms, and treatments, their ability to accurately assess causal relationships and ground their conclusions in verified scientific evidence remains unclear. Here, we present a preliminary, small-scale study that investigates the accuracy of LLMs in evaluating causal medical claims and supporting them with peer-reviewed research. We propose an evaluation framework for causal hypothesis verification that can be used to systematically track the performance of existing and future LLMs. We assess the performance of eight LLMs on 17 medical causal hypotheses to evaluate whether they can reliably verify these hypotheses using scientific evidence from the literature. We systematically annotate the scien
    
[^175]: RePro：面向大语言模型数学问题求解可靠评估的证明验证基准改写方法

    RePro: Proof-Verified Benchmark Rewriting for Reliable Evaluation of LLM Mathematical Problem Solving

    [https://arxiv.org/abs/2609.00062](https://arxiv.org/abs/2609.00062)

    RePro首次将Lean自动定理证明器集成到数学基准改写中，通过形式化证明保证改写题目的有效性与答案正确性，并发现多个大语言模型在验证后的改写基准上准确率下降，暴露了其依赖记忆化而非真正推理能力的问题。

    

    数据污染破坏了大语言模型（LLM）在数学问题求解任务上评估的可靠性。虽然基于改写的评估方法可以缓解模型记忆化问题，但现有方法缺乏对问题有效性和答案正确性的保证。我们提出了证明验证基准改写框架RePro，这是首个将面向Lean的神经自动定理证明器（ATP）集成到基准改写中的框架，在改写问题并重新生成答案的同时，通过Lean验证的证明确保其正确性。在GSM8K和MATH数据集上的实验表明，RePro保留的改写实例达到了100%的良好定义性、可解性和答案正确性，而现有方法仍会产生无效或不正确的实例。此外，多个模型在经过证明验证的改写基准上出现了准确率下降，这表明它们的性能对表层和结构变化较为敏感，可能部分反映了记忆化效应。

    arXiv:2609.00062v1 Announce Type: cross  Abstract: Data contamination undermines the reliable evaluation of large language models (LLMs) on mathematical problem solving. While rewriting-based evaluation mitigates memorization, existing methods lack guarantees of problem validity and answer correctness. We propose Proof-Verified Benchmark Rewriting (RePro), the first framework to integrate Lean-oriented neural automated theorem provers (ATPs) into benchmark rewriting, which rewrites problems and regenerates answers with correctness ensured by Lean-verified proofs. Experiments on GSM8K and MATH show that RePro's retained rewritten instances achieve 100% well-definedness, feasibility, and answer correctness, while existing methods still produce invalid or incorrect instances. Moreover, several models exhibit accuracy drops on proof-verified rewritten benchmarks, suggesting that their performance is sensitive to surface-level and structural variations and may partly reflect memorization ef
    
[^176]: CUDA-Harness：从自然语言驱动的智能体式CUDA内核生成与优化

    CUDA-Harness: Harnessing Agentic CUDA Kernel Generation and Optimization from Natural Language

    [https://arxiv.org/abs/2609.00058](https://arxiv.org/abs/2609.00058)

    该论文提出CUDA-Harness框架，通过智能体式方法直接从自然语言生成并优化高性能CUDA内核，克服了现有工作局限于PyTorch转译以及因依赖预定义测试输入而易受奖励欺骗的不足。

    

    开发高性能CUDA内核需要掌握算法实现、正确性验证以及面向硬件的并行优化等专业知识，这构成了很高的专业门槛，因此直接从自然语言生成CUDA内核变得至关重要。与此同时，大语言模型（LLM）通用的代码生成能力催生了一系列基于LLM的CUDA内核生成研究。这些工作主要聚焦于从PyTorch等高级框架向CUDA的转译（Torch2CUDA），而非Text2CUDA——后者要求模型既要理解高层输入语义，又要处理底层的内核实现与验证。此外，由于依赖预定义的测试输入，这些方法容易受到奖励欺骗的影响。在本文中，我们提出了CUDA-Harness，一个用于从自然语言驱动智能体式CUDA内核生成与优化的框架。

    arXiv:2609.00058v1 Announce Type: cross  Abstract: Developing high-performance CUDA kernels demands specialized knowledge in algorithm implementation, correctness validation, and hardware-aware parallel optimization, creating a substantial expertise barrier and making generating CUDA kernels directly from natural language (Text2CUDA) essential. Meanwhile, the general-purpose code generation capability of Large Language Models (LLMs) prompts a series of works exploring LLM-based CUDA kernel generation. They mainly focus on transpilation from high-level frameworks such as PyTorch to CUDA (Torch2CUDA) rather than Text2CUDA, where models must understand the high-level input semantics and handle low-level kernel implementation and validation. Additionally, these methods are vulnerable to reward hacking due to reliance on predefined test inputs. In this paper, we propose CUDA-Harness, a framework for harnessing agentic CUDA kernel generation and optimization from natural language. Specifical
    
[^177]: ValueGraph：价值信号引导的图预训练方法用于上下文化用户表示

    ValueGraph: Value-Signal Guided Graph Pre-training for Contextualized User Representation

    [https://arxiv.org/abs/2609.00057](https://arxiv.org/abs/2609.00057)

    提出ValueGraph图预训练框架，将自动推断的道德价值信号作为软约束辅助信号，结合对比学习与聚类目标学习上下文化的用户表示，在立场检测和推特机器人检测任务上取得提升。

    

    价值信号是一种聚合的用户级道德表征，能够从用户的在线言论中捕捉其被推断出的与价值观相关的倾向。社交媒体上的用户行为不仅受用户说什么或与谁互动的影响，还受到用户表达态度时所依托的价值信号的影响。然而，现有的用户表示方法大多忽略了这一与价值相关的维度。我们提出ValueGraph，一个图预训练框架，它将自动推断的道德价值信号作为含噪的辅助信号，用于学习上下文化的用户表示。ValueGraph从帖子-回复图中学习语义和结构表征，并通过对比学习和聚类目标，基于相对价值相似度进一步对齐用户。ValueGraph并不把推断出的价值观当作标准的心理学标签，而是将其用作表示学习的软约束。在立场检测和推特机器人检测任务上的实验表明……

    arXiv:2609.00057v1 Announce Type: cross  Abstract: Value signals are aggregated user-level moral representations that capture users' inferred value-related tendencies from their online discourse. User behavior on social media is shaped not only by what users say or whom they interact with, but also by the value signal through which they express attitudes. Existing user representation methods largely miss this value-relevant dimension. We propose ValueGraph, a graph pre-training framework that uses automatically inferred moral-value signals as noisy auxiliary signals for contextualized user representation. From post-reply graphs, ValueGraph learns semantic and structural representations and further aligns users through relative value similarity with contrastive and clustering objectives. Rather than treating inferred values as gold psychological labels, ValueGraph uses them as soft constraints for representation learning. Experiments on stance detection and twitter bot detection show co
    
[^178]: 通过大语言模型增强的音频-文本对齐实现零样本呼吸音分类

    Zero-Shot Respiratory Sound Classification through LLM-Augmented Audio-Text Alignment

    [https://arxiv.org/abs/2609.00055](https://arxiv.org/abs/2609.00055)

    该论文提出利用医学大语言模型从元数据合成结构化报告，将自监督呼吸音编码器与医学术语在共享潜在空间中对齐，实现61.3%平均零样本AUC，以更少数据超越CLAP和Qwen2-Audio等大规模基线模型。

    

    自监督呼吸音编码器缺乏零样本推理所需的临床领域语义基础，在没有任务特定标注数据的情况下限制了其实用性。我们提出了一个框架，将这些编码器与医学术语在共享潜在空间中对齐，使其转变为具备零样本能力的基础模型。为解决配对数据稀缺问题，我们使用医学大语言模型从元数据合成结构化报告，为对比学习创建密集的语义锚点。我们的训练方法将基于sigmoid的对比损失与编码器原生的自监督学习目标相结合，并采用相似度感知的负样本采样来锐化病理边界。在6个数据集的9项任务上，我们的方法实现了61.3%的平均零样本AUC，超过了CLAP（51.4%）和Qwen2-Audio（54.9%），同时仅使用全规模基线模型43%的数据就达到了最高的线性探测AUC（71.6%），表明结构化语义对齐优于大规模方法。

    arXiv:2609.00055v1 Announce Type: cross  Abstract: Self-supervised respiratory encoders lack semantic grounding in clinical domain needed for zero-shot inference, limiting their utility without task-specific labeled data. We propose a framework that aligns these encoders with medical terminology in a shared latent space turning them into a zero-shot-capable foundation model. To address paired data scarcity, we use a medical LLM to synthesize structured reports from metadata, creating dense semantic anchors for contrastive learning. Our training combines a sigmoid-based contrastive loss with encoder's native SSL objective and similarity-aware negative sampling to sharpen pathological boundaries. Across 9 tasks on 6 datasets, our method achieves a 61.3% mean zero-shot AUC, surpassing CLAP (51.4%) and Qwen2-Audio (54.9%) while reaching the highest linear probing AUC (71.6%) with only 43% of data used by full-scale baselines, showing that structured semantic alignment outperforms large-sca
    
[^179]: AgentProv：基于工具使用策略探针的智能体式LLM API提供商审计

    AgentProv: Auditing Agentic LLM API Providers via Tool-use Policy Probes

    [https://arxiv.org/abs/2609.00052](https://arxiv.org/abs/2609.00052)

    提出AgentProv，首个基于动作的智能体式LLM API身份审计方法，通过工具使用策略探针利用内化在模型权重中的工具使用行为，克服了文本通道审计在智能体API场景下的结构性脆弱问题。

    

    商业LLM API宣称提供特定的基础模型，但其服务的底层模型可能被悄然替换、量化或封装（例如为了节省部署成本）。所有现有的审计方法都是从文本输出通道来判断底层模型的身份，这对于智能体式API而言在结构上是脆弱的，因为现代服务栈（OpenAI、Anthropic、Gemini、Cloudflare Workers AI、LangGraph）在模型调用工具时会丢弃文本，只暴露结构化动作；而且提供商注入的系统提示词会严重扭曲文本分布，足以使基于文本通道的测试错误地指控诚实的提供商替换了所声称的模型。我们观察到，近期的智能体式后训练已将工具使用直接内化到模型权重中，这开辟了一条服务栈仍然暴露、且对部署环境基本不变的新审计通道。我们提出智能体溯源方法AgentProv，这是首个面向智能体式LLM API的基于动作的身份审计方法。

    arXiv:2609.00052v1 Announce Type: cross  Abstract: Commercial LLM APIs advertise a specific foundation model, but the served backbone may be silently substituted, quantized, or wrapped, for example to save deployment costs. All existing audits decide backbone identity from the text-output channel, which is structurally fragile for agentic APIs because modern serving stacks (OpenAI, Anthropic, Gemini, Cloudflare Workers AI, LangGraph) discard text and expose only structured actions when the model calls a tool, and provider-injected system prompts can distort text distributions enough that text-channel tests falsely accuse honest providers of substituting the claimed model. We observe that recent agentic post-training internalizes tool-use directly into the weights, opening a new audit channel that the serving stack still exposes and that is largely invariant to deployment context. We introduce Agentic Provenance (AgentProv), the first action-based identity audit for agentic LLM APIs: Ag
    
[^180]: 从检测到拒答：通过电路引导的权重缩放实现更安全的大语言模型

    From Detection to Refusal: Safer LLMs via Circuit-Guided Weight Scaling

    [https://arxiv.org/abs/2609.00051](https://arxiv.org/abs/2609.00051)

    该论文从机制可解释性角度首次刻画了大语言模型中由有害检测头、安全神经元和拒答头组成的多阶段安全电路，通过因果干预实验验证了这一电路组织，并据此提出利用电路引导的权重缩放方法构建更安全的大语言模型。

    

    尽管已经进行了大量的对齐工作，大语言模型（LLMs）在对抗性提示下仍然容易生成不安全的内容，然而安全行为得以实现的内部机制仍然鲜为人知。我们从机制可解释性的视角研究大语言模型的安全性，并刻画了一个组织拒答行为的多阶段*安全电路*，该电路由以下部分组成：(i) 对有害输入作出响应的**有害检测头**，(ii) 在残差流中介导并稳定安全信号的**安全神经元**，以及 (iii) 将这些信号转化为安全响应生成的**拒答头**。通过有针对性的注意力头层面和神经元层面的干预，我们提供了与该电路组织结构相一致的因果证据，表明抑制上游的有害检测头会破坏下游的拒答行为，并且安全神经元介导了这种相互作用。我们验证（原文摘要在此处截断）

    arXiv:2609.00051v1 Announce Type: cross  Abstract: Despite extensive alignment efforts, Large Language Models (LLMs) remain vulnerable to generating unsafe content under adversarial prompting, yet the internal mechanisms by which safety behaviors are implemented remain poorly understood. We study LLM safety from a mechanistic interpretability perspective and characterize a multi-stage *safety circuit* that organizes refusal behavior, consisting of (i) $\textbf{Harmful Detection Heads}$ that respond to harmful inputs, (ii) $\textbf{Safety Neurons}$ that mediate and stabilize safety signals in the residual stream, and (iii) $\textbf{Refusal Heads}$ that translate these signals into safe response generation. Using targeted attention-head and neuron-level interventions, we provide causal evidence consistent with this circuit organization, showing that suppressing upstream Harmful Detection Heads disrupts downstream refusal behavior and that safety neurons mediate this interaction. We valid
    
[^181]: GUI-CC：面向智能体环境的GUI世界模型上下文一致性基准测试

    GUI-CC: Benchmarking Contextual Consistency of GUI World Models as Agent Environments

    [https://arxiv.org/abs/2609.00048](https://arxiv.org/abs/2609.00048)

    提出GUI-CC基准，通过离线真实轨迹滚动和在线智能体交互循环两条互补轨道，评估GUI世界模型在多步智能体环境中反复复用生成状态时的上下文一致性。

    

    GUI世界模型目前越来越多地被评估为单步的下一屏幕预测器，然而它们的实际用途往往是作为GUI智能体的多步交互环境。这种错配导致一个关键需求未被充分测试：生成的状态在被反复复用于未来交互时，必须保持上下文一致性。我们提出了GUI-CC，这是一个评估GUI世界模型作为智能体环境（而非孤立的下一屏幕预测器）的上下文一致性的基准。GUI-CC包含两条互补的评估轨道：离线参考动作轨道，让模型沿真实的移动GUI轨迹进行滚动；以及在线智能体循环轨道，让固定的探测智能体与模型生成的UI进行交互。我们从GUIOdyssey构建了500个离线轨迹任务，并在30个移动应用中构建了200个经模拟器验证的在线任务。GUI-CC评估转移保真度、转移合理性、上下文一致性以及任务进展。实验表明……（摘要在此处截断）

    arXiv:2609.00048v1 Announce Type: cross  Abstract: GUI world models are increasingly evaluated as one-step next-screen predictors, yet their intended use is often as multi-step environments for GUI agents. This mismatch leaves a key requirement under-tested: generated states must remain contextually consistent when they are repeatedly reused for future interaction. We introduce GUI-CC, a benchmark that evaluates contextual consistency of GUI world models as agent environments rather than isolated next-screen predictors. GUI-CC contains two complementary tracks: an offline reference-action track that rolls models along real mobile GUI trajectories, and an online agent-loop track that lets fixed probing agents interact with model-generated UIs. We construct 500 offline trajectory tasks from GUIOdyssey and 200 emulator-verified online tasks across 30 mobile apps. GUI-CC evaluates transition fidelity, transition plausibility, contextual consistency, and task progress. Experiments show that
    
[^182]: trajectory-judge：仅基于结果的LLM评判器在智能体轨迹上遗漏了什么

    trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories

    [https://arxiv.org/abs/2609.00038](https://arxiv.org/abs/2609.00038)

    仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。

    

    仅基于结果的评估是LLM智能体在生产环境中的默认做法：向评判器展示用户请求和最终回复，询问其处理是否得当。这一指标在结构上无法察觉那些“以错误方式得到正确答案”的智能体。我们在真值可以通过构造获知的场景下测量这一盲区：一个确定性的使用工具的客服支持台环境、一个总能解决问题的脚本化oracle策略，以及一个在已知步骤恰好破坏一个环节的故障注入器，并根据用户可见结果是否仍然保持（静默型故障）与否（显性型故障）对故障进行分层。五种评判器（程序化规则、仅结果型、两种模型规模的逐步评分标准型、以及自一致性集成）在400条轨迹上按照检测能力、步骤定位、故障类型判定、校准度和成本进行评分。结果显示：仅结果型评判器能捕获84%的显性故障，但只能捕获45%的静默故障，同时还会误报33%的正确轨迹；而逐步评分标准型评判器对静默故障的召回率达到77%。

    arXiv:2609.00038v1 Announce Type: cross  Abstract: Outcome-only evaluation is the production default for LLM agents: show a judge the request and the final reply and ask whether it was handled well. The metric is structurally blind to an agent that reaches the right answer the wrong way. We measure that blind spot where ground truth is known by construction: a deterministic tool-using support-desk environment, a scripted oracle policy that always solves it, and a fault injector that breaks exactly one thing at a known step, stratifying faults by whether the customer-visible outcome survived (silent) or not (loud). Five judges (programmatic rules, outcome-only, step-rubric at two model sizes, and a self-consistency ensemble) are scored on detection, step localisation, fault typing, calibration, and cost over 400 trajectories. The outcome-only judge catches 84% of loud faults but 45% of silent ones while flagging 33% of correct trajectories; a step-rubric judge reaches 77% silent recall 
    
[^183]: UI-Venus-2 技术报告

    UI-Venus-2 Technical Report

    [https://arxiv.org/abs/2609.00028](https://arxiv.org/abs/2609.00028)

    UI-Venus-2是一个通用GUI基础智能体，通过统一的闭环推理-行动框架跨移动、网页和桌面环境运行，并从环境、任务和验证三个维度联合扩展，从而获得可靠的强化学习信号并迈向实际部署。

    

    多模态GUI智能体已成为数字任务自动化的一个有前景的范式，但由于环境覆盖有限、任务构建脆弱以及奖励验证不可靠，从面向基准测试的模型过渡到可靠的真实世界应用仍然充满挑战。在本工作中，我们提出了UI-Venus-2，一个通用基础GUI智能体，旨在通过统一的闭环推理-行动框架跨移动、网页和桌面环境运行。为弥合迈向实际部署的差距，我们联合扩展了三个关键维度：(1) 环境，将覆盖范围扩展至170多个多语言移动应用和原生桌面操作系统；(2) 任务，采用深度研究流水线进行基于功能的指令生成；(3) 验证，采用结合视觉关键点和多模型投票的轨迹级与样本级评估器，以确保训练中可靠的强化学习信号。

    arXiv:2609.00028v1 Announce Type: new  Abstract: Multimodal GUI agents have emerged as a promising paradigm for digital task automation, yet transitioning from benchmark-oriented models to dependable real-world applications remains challenging due to limited environment coverage, brittle task construction, and unreliable reward verification. In this work, we present UI-Venus-2, a general-purpose foundation GUI agent designed to operate across mobile, web, and desktop environments through a unified closed-loop reasoning-action framework. To bridge the gap toward practical deployment, we jointly scale three critical dimensions: (1) Environments, expanding coverage to more than 170 multilingual mobile apps and native desktop operating systems; (2) Tasks, employing a deep-research pipeline for function-grounded instruction generation; and (3) Verification, adopting trace-level and sample-level evaluators with visual keypoints and multi-model voting to ensure reliable RL signals for trainin
    
[^184]: 基于真实世界行为数据的用户画像：面向个性化对齐与多视角推理

    Behaviorally Grounded User Profiles from the Wild for Personalized Alignment and Multi-Perspective Reasoning

    [https://arxiv.org/abs/2609.00014](https://arxiv.org/abs/2609.00014)

    提出直接从真实匿名社交媒体数据中提取开放式高保真用户画像的行为锚定框架，在训练时个性化与测试时多视角推理两种范式下均显著优于合成人格基线。

    

    基于人格（Persona）驱动的技术正日益被用于将大语言模型（LLM）适配到多样化场景中。然而，现有方法主要依赖于僵化的、合成的人格设定，这些设定抹平了个体差异、依赖刻板印象，并且忽略了驱动真实人类偏好的微妙信号。我们提出了画像行为锚定（profile behavioral grounding）框架，可直接从真实、匿名化的社交媒体帖子中提取开放式、高保真的用户画像。我们在两种范式下对这些画像进行评估：通过监督微调（SFT）实现的训练时个性化，以及非参数化的测试时多视角推理。在复杂的推荐与开放式查询基准测试中，基于真实行为的画像始终能提升基础模型的表现，并优于合成画像基线，实现了更强的参数化对齐，并支持更丰富、更多维度的推理。

    arXiv:2609.00014v1 Announce Type: cross  Abstract: Persona-driven techniques increasingly adapt large language models (LLMs) to diverse contexts. However, existing methods predominantly rely on rigid, synthetic personas that flatten individual variation, rely on stereotypes, and miss the nuanced signals driving actual human preferences. We introduce profile behavioral grounding, a framework for extracting open-ended, high-fidelity user profiles directly from authentic, anonymized social media posts. We evaluate these profiles across two paradigms: train-time personalization via supervised finetuning (SFT) and non-parametric test-time multi-perspective reasoning. Across complex recommendation and open-ended query benchmarks, behaviorally grounded profiles consistently improve base models and outperform synthetic profile baselines, driving stronger parametric alignment and enabling richer, multifaceted reasoning. Our findings establish open-ended, behavior-derived profiles as a highly di
    
[^185]: TopoCompress：基于图连接语义轨迹的长上下文压缩

    TopoCompress: Long Context Compression via Graph-Wired Semantic Trajectories

    [https://arxiv.org/abs/2608.30811](https://arxiv.org/abs/2608.30811)

    TopoCompress提出了一种无需训练、与模型无关的长上下文压缩框架，通过构建混合图连接连贯的语义片段并在其上传播查询引导的相关性分数，在五个长上下文基准任务上以更少的资源持续超越强大的压缩基线。

    

    长上下文压缩对于降低大语言模型推理的成本和延迟至关重要。然而，现有方法可能会割裂重要的证据信息，需要额外的训练或对齐，并且通常依赖目标模型才能实现有效压缩。我们提出了TopoCompress，这是一个无需训练且与模型无关的框架，通过选择连贯的语义片段来压缩长上下文。TopoCompress首先结合密集与词汇层面的查询相关性以及语义加速对每个片段进行评分。然后，它构建一个混合图，基于语义相似性和序列相邻性将各片段连接起来，并在图上传播查询引导的相关性分数。在五个长上下文任务——HotpotQA、2WikiMQA、MuSiQue、Qasper和MultiFieldQA-en——上，TopoCompress始终优于强大的压缩基线。值得注意的是，TopoCompress在使用4倍更少（的资源）的情况下达到了与最强基线相当的性能。

    arXiv:2608.30811v1 Announce Type: new  Abstract: Long-context compression is essential for reducing the cost and latency of large language model inference. However, existing methods can fragment important evidence, require additional training or alignment, and often depend on the target model for effective compression. We introduce TopoCompress, a training-free and model-agnostic framework that compresses long contexts by selecting coherent semantic spans. TopoCompress first scores each span using dense and lexical query relevance together with semantic acceleration. It then constructs a hybrid graph that connects spans based on semantic similarity and sequential adjacency, and propagates the query-guided relevance scores over the graph. Across five long-context tasks-HotpotQA, 2WikiMQA, MuSiQue, Qasper, and MultiFieldQA-en-TopoCompress consistently outperforms strong compression baselines. Notably, TopoCompress achieves performance comparable to the strongest baseline while using a 4x
    
[^186]: 用于声明核查价值检测的小型语言模型校准

    Calibrating Small Language Models for Claim Check-Worthiness Detection

    [https://arxiv.org/abs/2608.30731](https://arxiv.org/abs/2608.30731)

    提出NN-PPI方法，作为推理时的轻量级后处理校准层，使小型语言模型在声明核查价值检测任务上以低一个数量级的服务成本达到大型语言模型的准确率，且无需重新训练模型。

    

    评估声明的核查价值是自动化事实核查流程中至关重要的第一步。这项工作源于一家早期初创公司面临的实际部署挑战：对每一条传入声明都运行大型语言模型在成本和延迟上都是难以承受的，而较小的模型又会牺牲准确性。我们提出了NN-PPI，这是预测驱动推理的一种逐点扩展方法，它在推理时作为轻量级的后处理层来校准模型预测，无需重新训练底层模型。根据基线模型的规模和性能不同，NN-PPI实现了12%到33.80%的加权F1提升，使小型语言模型达到了与大型语言模型相当的水平。除了少样本小型语言模型之外，NN-PPI还进一步改进了一个已在生产环境中部署的微调模型，表明残差校准与监督微调是互补的。通过从服务成本低一个数量级的模型中恢复出LLM级别的准确性，它…

    arXiv:2608.30731v1 Announce Type: cross  Abstract: Assessing claim check-worthiness is an essential first step in automated fact-checking pipelines. This work is motivated by a real deployment challenge at an early-stage startup: running large language models (LLMs) over every incoming claim is cost- and latency-prohibitive, yet smaller models sacrifice accuracy. We propose NN-PPI, a pointwise extension of Prediction-Powered Inference (PPI) that calibrates model predictions at inference time as a lightweight post-hoc layer, without re-training the underlying model. NN-PPI achieves weighted F1 gains ranging from 12% to 33.80% depending on the size and performance of the baseline model, bringing SLMs on par with larger LLMs. Beyond few-shot SLMs, NN-PPI further improves a production-deployed fine-tuned model, demonstrating that residual calibration is complementary to supervised fine-tuning. By recovering LLM-level accuracy from models that are an order of magnitude cheaper to serve, it 
    
[^187]: BiG-SURE——用于大语言模型语义不确定性与可靠性估计的二部图方法

    BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs

    [https://arxiv.org/abs/2608.30646](https://arxiv.org/abs/2608.30646)

    提出了一种基于跨温度语义一致性的黑盒不确定性估计方法BiG-SURE，通过构建低温锚点与高温探针之间的二部图并用谱能量衡量语义一致性，从而评估大语言模型输出的可靠性。

    

    可靠的不确定性估计是在安全关键场景中部署大语言模型（LLM）和视觉-语言模型（VLM）的关键前提，尤其是在模型参数不可访问（黑盒）的情况下。我们提出了BiG-SURE，一种基于跨温度语义一致性的不确定性估计器。该方法在保持语义不变的输入变换下，将低温采样得到的响应作为稳定的语义锚点，将高温采样得到的响应作为探针。随后，方法利用基于自然语言推理（NLI）的蕴含分数构建锚点-探针二部图，并通过该矩阵的归一化平方谱能量来定义置信度，不确定性则由其补集给出。这种基于二部图的语义不确定性与可靠性估计（SURE）分数，用于衡量高温探针是否与模型稳定的低温信念保持语义一致。我们在文本问答等任务上对BiG-SURE进行了评估。

    arXiv:2608.30646v1 Announce Type: cross  Abstract: Reliable uncertainty estimation is a crucial requirement for deploying large language models (LLMs) and vision-language models (VLMs) in safety-critical settings, especially when the model parameters are not accessible (black-box). We propose BiG-SURE, an uncertainty estimator based on cross-temperature semantic agreement. The method samples low-temperature responses as stable semantic anchors and high-temperature responses as probes under meaning-preserving input transformations. It then constructs an anchor-probe Bipartite Graph (BiG) using NLI-based entailment scores and defines confidence through the normalized squared spectral energy of this matrix, with uncertainty given by its complement. This bipartite graph-based Semantic Uncertainty and Reliability Estimation (SURE) score measures whether high-temperature probes remain semantically aligned with the model's stable low-temperature belief or not. We evaluate BiG-SURE on text QA,
    
[^188]: 图证据并不足够：诊断图增强大语言模型中原生解码器的使用

    Graph Evidence Is Not Enough: Diagnosing Native Decoder Use in Graph-Augmented LLMs

    [https://arxiv.org/abs/2608.30437](https://arxiv.org/abs/2608.30437)

    本文通过 HopQA 诊断任务和干预三角实验设计，揭示图增强大语言模型“获得图证据”不等于“能使用图证据”，并据此提出 S$^2$GE 接口设计以提升原生解码器对图拓扑的利用能力。

    

    图增强大语言模型通常假设由外部计算产生并放置在输入中的图证据可以被原生解码器所利用。我们通过 HopQA 来检验这一假设，这是一个刻意限定范围的诊断任务，要求回答两个查询节点之间的最短跳数距离。由于答案是一个小整数且目标纯粹是拓扑性的，失败不能被归咎于开放式生成或模糊的评估。然而，现有的图增强基线在这一设定下仍然失败，这表明提供图证据并不等于使其可用。我们引入了一个干预三角，包含三种匹配的条件：可读的图证据、打乱的图证据和无图输入。这将证据的包含、结构的可读性和解码器可用的拓扑三者分离开来。在这一诊断的指导下，我们提出了 S$^2$GE，作为一个实例，表明以诊断为驱动的接口设计可以改进原生解码器对图证据的利用。

    arXiv:2608.30437v1 Announce Type: new  Abstract: Graph-augmented large language models often assume that graph evidence produced by external computation and placed in the input can be used by the native decoder. We test this assumption with HopQA, a deliberately bounded diagnostic that asks for the shortest-hop distance between two query nodes. Because the answer is a small integer and the target is purely topological, failure cannot be dismissed as open-ended generation or ambiguous evaluation. Yet existing graph-augmented baselines still fail on this setting, showing that providing graph evidence is not the same as making it usable. We introduce an intervention triangle with three matched conditions: readable graph evidence, shuffled graph evidence, and no-graph input. This separates evidence inclusion, structural readability, and decoder-usable topology. Guided by this diagnosis, we present S$^2$GE as an instance showing that diagnosis-driven interface design can improve native deco
    
[^189]: 用户会知道吗？针对使用工具的LLM智能体的隐蔽间接提示注入

    Will the User Ever Know? Covert Indirect Prompt Injection on Tool-Using LLM Agents

    [https://arxiv.org/abs/2608.30362](https://arxiv.org/abs/2608.30362)

    该论文从用户视角将间接提示注入的攻击成功率分解为隐蔽成功率（CSR）和公开成功率（OSR），揭示了智能体在最终响应中不留痕迹地执行恶意注入的隐蔽攻击威胁。

    

    随着LLM智能体通过工具执行真实世界的操作，间接提示注入（IPI）已成为一种严重的威胁。标准的评估指标——攻击成功率（ASR）——只统计注入是否成功，却忽略了用户在智能体最终响应中能够注意到什么。通过观察成功的注入轨迹，我们发现两种截然不同的结果：智能体在执行注入的同时返回看似正常的响应，或者在最终响应中报告被注入的操作，从而给用户留下察觉的机会。我们将这两类成功分别称为隐蔽成功和公开成功。从用户视角出发，我们将ASR分解为隐蔽成功率（CSR）——统计在最终响应中不留任何痕迹的成功注入——以及公开成功率（OSR）——统计用户能够察觉的成功注入。为了理解造成这一差距的原因，我们分析了成功的注入轨迹，发现注入后智能体的行为是区分隐蔽与公开的关键：隐蔽的轨迹会将控制权交回……

    arXiv:2608.30362v1 Announce Type: new  Abstract: As LLM agents take real-world actions through tools, indirect prompt injection (IPI) has emerged as a serious threat. The standard metric, Attack Success Rate (ASR), counts whether an injection succeeds but ignores what the user notices in the agent's final response. Looking at successful injection traces, we find two distinct outcomes: the agent executes the injection while returning an otherwise normal response, or reports the injected action in its final response, giving the user a chance to notice. We call these covert and overt successes. From the user's perspective, we decompose ASR into the Covert Success Rate (CSR), counting successes leaving no trace in the final response, and the Overt Success Rate (OSR), counting successes the user can detect. To understand what drives the gap, we analyze successful trajectories and find that the agent's behavior after the injection separates covert from overt: covert traces hand control back 
    
[^190]: 惰性接地：用事实性证据攻击搜索代理

    Lazy Grounding: Attacking Search Agents with Factual Evidence

    [https://arxiv.org/abs/2608.30303](https://arxiv.org/abs/2608.30303)

    该论文提出“惰性接地”攻击：即使检索到的文档完全真实，只要其支持的是相邻改写问题的答案，搜索代理也会采用无法回答当前问题的相邻答案，导致准确率平均下降5.9个百分点、最高下降17.3个百分点。

    

    搜索代理通过将答案锚定在检索到的网络证据上来减少幻觉。然而，对检索的依赖也带来了攻击面：含有虚假或恶意文档的被投毒语料库可能导致代理传播错误信息。我们证明，虚假性并非必要条件——搜索代理可能被与相邻问题相关的事实性证据所误导，即使该相邻答案并不能回答当前问题，代理也会采用它。我们将这种失败称为“惰性接地”。我们利用基准问题经改变答案的改写所产生的相邻证据来揭示惰性接地现象：每份文档都真实地支持一个相邻的改写问题，但却被呈现在原始问题的检索结果中。在12个模型-基准组合上，相邻证据平均使准确率下降5.9个百分点，最高达17.3个百分点，并且在所有设置中都会诱导代理采用相邻答案。当相邻证据出现得越晚或越像答案时，这种效应越强。

    arXiv:2608.30303v1 Announce Type: new  Abstract: Search agents reduce hallucination by grounding answers in retrieved web evidence. Yet reliance on retrieval also creates an attack surface: poisoned corpora with false or malicious documents can cause agents to reproduce misinformation. We show that falsehood is not necessary -- a search agent can be misled by factual evidence for a nearby question, adopting that nearby answer even when it does not answer the current question. We call this failure lazy grounding. We expose lazy grounding using nearby evidence from answer-changing rewrites of benchmark questions. Each document truthfully supports a neighboring rewritten question, but is surfaced for the original question. Across 12 model-benchmark pairs, nearby evidence reduces accuracy by 5.9 points on average and by up to 17.3 points, while inducing nearby-answer adoption in every setting. The effect is stronger when nearby evidence appears later or is more answer-shaped. Our results s
    
[^191]: Arkios：一个从头训练的开源英-尼泊尔语双语语言模型，配备天城文感知分词器

    Arkios: An Open Bilingual English-Nepali Language Model Trained From Scratch, with a Devanagari-Aware Tokenizer

    [https://arxiv.org/abs/2608.30092](https://arxiv.org/abs/2608.30092)

    Arkios是一个从零训练的10.4亿参数英-尼泊尔语双语开源模型，采用专门设计的天城文感知分词器，以少一个数量级的训练数据超越了同规模开源模型，并揭示了低资源语言评估中提示格式对结果的关键影响。

    

    我们提出了Arkios，一个拥有10.4亿参数的稠密transformer模型，在1500亿token的英-尼泊尔语双语语料上从零开始预训练，使用了自定义的单文件C/CUDA训练框架，以及为本项目专门构建的天城文感知字节级BPE分词器。在ARC-Easy和ARC-Challenge基准上，Arkios超越了三个规模相当的开源模型（Pythia-1.4B、TinyLlama-1.1B、OLMo-1B），尽管其训练token数量少了一个数量级，这可能得益于我们的教育类网页文本预训练数据与ARC的小学科学题格式相匹配，而非通用能力上的优势。我们报告了在标准协议下的完整评估结果，包括对早前部分样本估计的更正，以及针对低资源语言小模型评估的发现：常用评估框架所使用的标准多选题字母提示格式使该模型在尼泊尔语阅读理解上仅达到随机水平，同时在……（原文在此处截断）

    arXiv:2608.30092v1 Announce Type: cross  Abstract: We present Arkios, a 1.04B-parameter dense transformer pretrained from scratch on 150B tokens of bilingual English-Nepali text, using a custom single-file C/CUDA training stack and a Devanagari-aware byte-level BPE tokenizer built for this project. On ARC-Easy and ARC-Challenge, Arkios exceeds three comparably sized open models (Pythia-1.4B, TinyLlama-1.1B, OLMo-1B) despite an order of magnitude fewer training tokens, likely aided by a match between our educational-web-text pretraining data and ARC's grade-school-science format rather than a general capability advantage. We report full evaluation results under standard protocols, including a correction to an earlier partial-sample estimate, and findings specific to evaluating small models in a low-resource language: the standard multiple-choice-letter prompt format used by common evaluation harnesses places this model at chance on Nepali reading comprehension, and simultaneously at cha
    
[^192]: XQDT：具有反馈信号的可解释且量化的数据-文本对齐度量

    XQDT: eXplainable and Quantitative Data-Text Alignment Metric with Feedback Signals

    [https://arxiv.org/abs/2608.29948](https://arxiv.org/abs/2608.29948)

    该论文提出XQDT，一种端到端可解释的数据-文本对齐评估指标，通过微调语言模型识别遗漏、多余、错误和正确的数据单元并聚合为精确率、召回率和F1分数，其性能优于LLM-as-Judge方法，且验证器输出可为下游纠错与改进提供反馈信号。

    

    评估数据与文本之间的对齐仍然具有挑战性：现有指标通常对分数提供的解释有限，而基于提示词的LLM-as-Judge（大语言模型作为评判者）方法可能成本高昂且不可靠。我们提出了一种端到端的可解释评估指标，通过微调语言模型来识别数据-文本对中被遗漏、多余、错误和正确的数据单元。这些局部判断被聚合为精确率、召回率和F1分数，既提供了细粒度的诊断反馈，也提供了对齐质量的可解释度量。在多个基准测试中，我们微调后的模型在错误预测方面优于LLM-as-Judge方法，并取得了具有竞争力的精确率、召回率和F1分数，同时与人类判断保持高度相关性。除评估之外，我们的验证器输出还能为下游的纠错和改进提供有用的反馈信号，支持面向对齐的数据到文本和文本到数据的改进。

    arXiv:2608.29948v1 Announce Type: new  Abstract: Evaluating data-text alignment remains challenging: existing metrics often provide limited explanations for the scores, while prompt-based LLM-as-Judge methods can be expensive and unreliable. We present an end-to-end explainable evaluation metric that fine-tunes a language model to identify omitted, extra, incorrect, and correct data units in a data-text pair. These local judgements are aggregated into precision, recall, and F1 scores, providing both fine-grained diagnostic feedback and an interpretable measure of alignment quality. Across benchmarks, our fine-tuned models outperform LLM-as-Judge methods in error prediction and achieve competitive precision, recall, and F1 scores, while maintaining strong correlation with human judgements. Beyond evaluation, our verifier outputs also provide useful feedback signals for downstream correction and refinement, supporting alignment-oriented improvement of data-to-text and text-to-data. Code 
    
[^193]: REIGN：利用集成引导网络的翻新嵌入实现高效的上下文长度扩展

    REIGN: Refurbished Embeddings with Integrated Guidance Networks for Efficient Context-Length Scaling

    [https://arxiv.org/abs/2608.29899](https://arxiv.org/abs/2608.29899)

    REIGN通过在冻结引导网络生成的块嵌入序列上运行对比训练的双编码器，将词元级处理与文档级推理解耦，使长文档检索的训练成本相比分块Transformer微调降低约四个数量级。

    

    对长文档进行稠密检索的代价高昂。词元级编码器在序列长度上呈二次方扩展，而大多数长上下文嵌入模型只能通过架构上的变通方法或拉长十亿参数级大语言模型才能达到32K词元。我们提出REIGN（Refurbished Embeddings with Integrated Guidance Networks，集成引导网络的翻新嵌入），这是一个经过对比训练的双编码器，它在由冻结的引导网络（GN）生成的上下文化块嵌入序列上运行，而不是在原始词元上运行。REIGN针对多块输入，主要用于文档到文档的检索；单块输入则仍由GN处理。通过将词元级处理与文档级推理解耦，并将GN嵌入缓存到磁盘，相对于分块Transformer微调，每个文档的训练成本降低了大约四个数量级。我们还发布了一个合成的长文档检索基准，用于长上下文长度下的对比训练与评估。

    arXiv:2608.29899v1 Announce Type: cross  Abstract: Dense retrieval over long documents is expensive. Token-level encoders scale quadratically in sequence length, and most long-context embedding models reach 32K tokens only through architectural workarounds or by stretching billion-parameter LLMs. We propose REIGN (Refurbished Embeddings with Integrated Guidance Networks), a contrastively trained bi-encoder that operates on sequences of contextualised chunk embeddings from a frozen Guidance Network (GN) rather than on raw tokens. REIGN targets multi-chunk inputs, primarily for document-to-document retrieval; single-chunk inputs stay with the GN. Decoupling token-level processing from document-level reasoning, and caching the GN embeddings to disk, cuts per-document training cost by roughly four orders of magnitude relative to chunked Transformer fine-tuning. We also release a synthetic long-document retrieval benchmark for contrastive training and evaluation at long context lengths. Acr
    
[^194]: 当历史是多模态的：重新思考长程智能体的上下文管理

    When History Is Multimodal: Rethinking Context Management for Long-Horizon Agents

    [https://arxiv.org/abs/2608.29897](https://arxiv.org/abs/2608.29897)

    本文将上下文管理形式化为预算受限的历史转换，首次在公平受控的对比下验证视觉渲染作为表示性上下文管理器的有效性，并揭示当智能体的交互历史本身是多模态时，视觉载体具有原生优势。

    

    长程智能体需要一个上下文管理器，通过被动策略或主动策略（即决定记忆如何被访问和重组），将不断增长的交互历史压缩到有界的工作上下文中。与此同时，先前的光学记忆工作主要将像素视为文本化历史的密集编解码器，往往预设将上下文渲染为光学记忆会带来相对于文本的显著性能下降，因此将这种表示与监督微调（SFT）、自蒸馏或强化学习相耦合以缩小这一差距，从而留下了两个未解决的问题：(i) 在公平、受控的比较下，视觉渲染作为上下文管理器的表现究竟如何；(ii) 当历史本质上是多模态的时，这种载体是否具有原生优势。在本文中，我们将上下文管理形式化为预算受限的历史转换，并引入视觉渲染作为一种表示性上下文管理器。在共享的测试框架下，策略模式……

    arXiv:2608.29897v1 Announce Type: new  Abstract: Long-horizon agents need a context manager to compress growing interaction histories into a bounded working context, via passive strategies or active strategies that decide how memory is accessed and reorganized. Meanwhile, prior optical-memory work mainly treats pixels as a dense codec for textualized histories, often presupposing that rendering context into optical memory incurs a significant performance drop relative to text, thus coupling this representation with SFT, self-distillation, or reinforcement learning to close this gap, leaving unresolved (i) how visual rendering performs as a context manager under a fair, controlled comparison, and (ii) whether this carrier offers a native advantage when history is inherently multimodal. In this paper, we formulate context management as a budget-constrained history transformation and introduce Visual Rendering (VR) as a representational context manager. Under a shared harness, policy mode
    
[^195]: HiVe：超越静态提示的多任务学习——基于层次结构的垂直专家混合方法

    HiVe: Beyond Static Prompts for Multitask Learning via Hierarchy-based Vertical Mixture-of-Experts

    [https://arxiv.org/abs/2608.29790](https://arxiv.org/abs/2608.29790)

    提出HiVe框架，通过构建多层次提示层次结构并结合垂直专家混合（V-MoE）机制，实现基于输入的自适应提示特化，在多任务学习中持续超越现有提示调优方法。

    

    随着大语言模型（LLM）的持续扩展，参数高效微调（PEFT）已成为全参数适配的一种实用替代方案。提示调优虽然有效，但现有方法要么使用扁平的提示结构，要么使用固定提示组合的层次结构，这限制了自适应的提示特化能力。为解决这一局限，我们提出了HiVe，这是一个在多个层级上对提示进行建模并实现基于输入的特化的提示调优框架。HiVe通过在训练期间利用任务间关系构建提示层次结构，并在推理时采用垂直专家混合（V-MoE）机制来动态组合提示，直至达到每个输入所需的特化层级。实验表明，HiVe在多种任务上持续超越强大的提示调优基线方法。

    arXiv:2608.29790v1 Announce Type: new  Abstract: As large language models (LLMs) continue to scale, parameter-efficient fine-tuning (PEFT) has become a practical alternative to full-parameter adaptation. Prompt tuning is effective, but existing approaches either use flat prompt structures or hierarchical structures with fixed prompt composition, limiting adaptive prompt specialization. To address this limitation, we propose HiVe, a prompt tuning framework that models prompts at multiple levels and enables input-dependent specialization. HiVe constructs a prompt hierarchy by leveraging inter-task relationships during training, and employs a vertical mixture-of-experts (V-MoE) mechanism at inference time to compose prompts up to the level of specialization required for each input. Experiments show that HiVe consistently outperforms strong prompt tuning baselines across diverse tasks.
    
[^196]: InteractBench：在信息未揭示条件下评测大语言模型竞赛编程能力的基准

    InteractBench: Benchmarking LLMs on Competitive Programming under Unrevealed Information

    [https://arxiv.org/abs/2608.29632](https://arxiv.org/abs/2608.29632)

    提出了InteractBench基准，包含322个精选自主流编程竞赛的高质量交互式问题，用于评测大语言模型在关键信息未预先揭示、需通过多轮交互进行算法推理的能力。

    

    竞赛编程正日益被用于评估大语言模型（LLM）的算法推理能力。然而，现有的基准测试主要聚焦于全信息任务，即所有问题输入都在开始时预先提供。这忽略了算法推理的一个关键维度：生成的程序在关键信息未预先揭示时的运行能力。交互式问题是竞赛编程的一个独特组成部分，正体现了这一挑战。这类问题要求程序在严格的协议约束和有限的查询预算下，与交互器（评测程序）进行多轮交互，且新信息仅在响应查询时才被揭示。为填补这一空白，我们提出了InteractBench，这是一个包含322个高质量交互式问题的基准，这些问题精选自Codeforces、AtCoder、IOI和ICPC。每个问题都配备了可执行的本地交互器，……

    arXiv:2608.29632v1 Announce Type: new  Abstract: Competitive programming is increasingly being used to evaluate the algorithmic reasoning capabilities of large language models (LLMs). However, existing benchmarks primarily focus on full-information tasks where all problem inputs are provided upfront. This overlooks a critical dimension of algorithmic reasoning: the ability of generated programs to operate when key information is not revealed upfront. Interactive problems, a distinctive component of competitive programming, embody this challenge. These problems require programs to engage in multi-round interaction with an interactor (a judge program) under strict protocol constraints and limited query budgets, with new information revealed only in response to queries. To address this gap, we introduce InteractBench, a benchmark comprising 322 high-quality interactive problems curated from Codeforces, AtCoder, IOI, and ICPC. Each problem is packaged with executable local interactors, ena
    
[^197]: OASIS：优化硬标签黑盒文本攻击中的攻击者序列

    OASIS: Optimizing Attacker Sequences for Hard-Label Black-Box Text Attacks

    [https://arxiv.org/abs/2608.29568](https://arxiv.org/abs/2608.29568)

    OASIS通过一次性双目标攻击链搜索来优化攻击者序列，在硬标签黑盒文本攻击中始终优于独立攻击器和手动构建的攻击链，将攻击者组合从实现选择提升为实际优化目标。

    

    不同的攻击方法遵循不同的搜索轨迹，它们在不同样本子集上取得成功，而现有的硬标签黑盒文本攻击主要专注于改进单个攻击器或手动组合它们。我们提出了OASIS，一种用于优化硬标签黑盒文本攻击中攻击者序列的方法。OASIS首先对候选序列执行一次性的双目标攻击链搜索，以平衡攻击成功率和扰动，然后在攻击链执行过程中复用所选定的固定全局链。在多个数据集、受害模型和大语言模型上的实验表明，OASIS始终优于强大的独立基线和简单的手动构建链。这些结果表明，攻击者组合不仅仅是一种实现选择，而是改进硬标签黑盒文本攻击的一个实际优化目标。

    arXiv:2608.29568v1 Announce Type: cross  Abstract: Different attack methods follow different search trajectories, they succeed on different subsets of samples, whereas existing hard-label black-box text attacks mainly focus on improving individual attackers or manually combining them. We present {\OURS}, a method for optimizing attacker sequences in hard-label black-box text attacks. {\OURS} first performs a one-time bi-objective attack chain search over candidate sequences to balance attack success rate and perturbation, and then reuses the selected fixed global chain during attack chain execution. Experiments across multiple datasets, victim models, and large language models show that {\OURS} consistently outperforms strong standalone baselines and simple manually constructed chains. These results suggest that attacker composition is not merely an implementation choice, but a practical optimization target for improving hard-label black-box text attacks.
    
[^198]: TACS：面向大语言模型越狱后缀优化的轨迹感知候选选择

    TACS: Trajectory-Aware Candidate Selection for LLM Jailbreak Suffix Optimization

    [https://arxiv.org/abs/2608.29564](https://arxiv.org/abs/2608.29564)

    论文揭示了基于梯度的越狱后缀优化中“仅选当前损失最低候选”的短视性，提出轨迹感知候选选择框架TACS，通过轨迹感知代理、参考策略正则化和判别器卡方校正，使候选选择在搜索后期依然有效。

    

    基于梯度的越狱后缀优化方法通常通过保留当前损失最低的候选来更新后缀。我们证明，这种看似自然的设计本质上是短视的：在当前步骤代理指标下表现更好的候选，往往无法在搜索后期产生更好的越狱结果，这揭示了一种选择阶段的奖励破解现象。这表明，候选选择（而不仅仅是候选生成）是后缀优化中一个隐藏的瓶颈。为了解决这一问题，我们提出了TACS，一个用于越狱后缀优化的轨迹感知候选选择框架。TACS不再仅根据即时损失来选择候选，而是通过轨迹感知代理来增强每一步的评估，并利用参考策略正则化和判别器估计的卡方校正来稳定选择过程，从而鼓励那些在当前步骤之后仍然有效的选择。

    arXiv:2608.29564v1 Announce Type: new  Abstract: Gradient-based jailbreak suffix optimization methods typically update the suffix by retaining the candidate with the lowest current loss. We show that this seemingly natural design is fundamentally myopic: candidates that look better under the current-step proxy often fail to produce better jailbreak outcomes later in the search, revealing a form of selection-stage reward hacking. This suggests that candidate selection, rather than candidate generation alone, is a hidden bottleneck in suffix optimization. To address this issue, we propose \OURS{}, a trajectory-aware candidate selection framework for jailbreak suffix optimization. Instead of selecting candidates solely by their immediate loss, \OURS{} augments per-step evaluation with a trajectory-aware proxy and stabilizes selection with reference-policy regularization and a discriminator-estimated chi-squared correction, encouraging choices that remain effective beyond the current step.
    
[^199]: 推理模型思维链的忠实性随偏好线索的传递位置与方式而变化

    Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered

    [https://arxiv.org/abs/2608.29464](https://arxiv.org/abs/2608.29464)

    论文提出FACE-Eval评估基准，揭示推理模型的思维链忠实性取决于偏好线索的传递位置和显式程度——相比用户消息和显式线索，通过工具返回和隐式方式传递的偏好更容易被模型默默采纳而不在思维链中如实言明。

    

    思维链监测的前提假设是推理过程忠实地记录了影响模型回答的信息。现有的忠实性测试通常将显式的偏见线索置于用户消息中，而智能体在实际运行中可能通过工具返回结果或原始数据工件接触到偏好信息。我们提出了FACE-Eval（线索效应忠实归因评估），这是一个包含5,100个样本的评估基准，通过改变线索的位置（用户消息或工具返回结果）和显式程度（直接总结或原始数据工件）来系统考察这一问题。我们测量了遵循线索的回答中的言语化承诺，以及所有含线索样本中的未言语化采纳。我们评估了来自八个模型家族的15个开源权重模型，总参数量从4B到1.60T不等。结果显示：所有模型对工具返回线索的言语化承诺均低于用户消息线索，对隐式线索的言语化承诺均低于显式线索；此外，在全部15个模型上，工具返回线索导致的未言语化采纳率更高，在30个模型-通道对比中的28个里，隐式线索的未言语化采纳率也更高。

    arXiv:2608.29464v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) monitoring assumes that reasoning traces faithfully record the information that shapes a model's answer. Existing faithfulness tests often place explicit bias cues in the user message, while agents may encounter preferences through tool returns or raw artifacts. We introduce FACE-Eval (Faithful Attribution of Cue Effects Evaluation), a 5,100-sample evaluation that varies cue location (user message or tool return) and explicitness (direct summary or raw artifact). We measure verbalized commitment among cue-following answers and unverbalized adoption among all cued samples. We evaluate 15 open-weight models from eight families, with total parameters ranging from 4B to 1.60T. Every model has lower verbalized commitment for tool-return than user-message cues and for implicit than explicit cues. Unverbalized adoption is higher for tool-return cues on all 15 models and for implicit cues in 28 of 30 model-channel compar
    
[^200]: FKG.in的验证：LLM增强的印度食品知识中的健全性评估

    Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge

    [https://arxiv.org/abs/2608.29249](https://arxiv.org/abs/2608.29249)

    本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。

    

    在线烹饪生态系统中，由大型语言模型（LLM）生成、修改或总结的食谱内容日益增多。虽然这些输出通常看似合理，但可能包含虚构的食材、被误述的用量或文化上不合常理的食材组合，从而限制了其在下游应用和知识图谱构建中的适用性。在本文中，我们提出了一种半自动化的健全性评估工作流程，用于验证由LLM从非正式烹饪来源中提取和增强的结构化食谱数据。该流程作为印度食品知识图谱FKG.in的一部分开发而成，通过结合形式文法、基于词汇的检查、统计启发式方法、基于Set Transformer的连贯性建模以及基于检索的验证等多阶段流程，识别并解决常见的失败模式，包括结构性不一致、语义和逻辑上的不连贯以及与源文本的偏差。

    arXiv:2608.29249v1 Announce Type: new  Abstract: The online culinary ecosystem is increasingly populated by recipe content generated, modified, or summarized by Large Language Models (LLMs). While often plausible, such outputs may contain hallucinated ingredients, misrepresented quantities, or culturally implausible combinations, limiting their suitability for downstream applications and knowledge graph construction. In this paper, we present a semi-automated soundness assessment workflow for validating structured recipe data extracted and augmented by LLMs from informal culinary sources. Developed as part of FKG.in, a knowledge graph of Indian food, the pipeline identifies and addresses common failure modes, including structural inconsistencies, semantic and logical incoherence, and deviations from the source text, through a multi-stage process combining formal grammars, vocabulary-based checks, statistical heuristics, Set Transformer-based coherence modeling, and retrieval-based veri
    
[^201]: 基于属性的大语言模型激活转向用于群体特定解释生成

    Attribute-Based Activation Steering of LLMs for Group-Specific Explanation Generation

    [https://arxiv.org/abs/2608.29215](https://arxiv.org/abs/2608.29215)

    本文提出一种基于激活工程的转向方法，通过计算群体特定属性的转向向量并注入大语言模型的内部激活，使其生成针对特定人群背景和能力量身定制的、具有特异性和事实性的解释。

    

    为了有效地让人们理解新话题，解释应当根据人们的背景和能力量身定制。迄今为止，仅靠提示工程已被证明不足以创建此类解释，而其他计算方法尚属空白。因此，本文研究是否可以引导大语言模型生成针对特定人群量身定制的解释。为此，我们提出了一种方法，该方法首先从解释风格和特定目标群体的知识方面识别群体特定属性。基于激活工程技术，该方法随后计算基于属性的转向向量，并在推理过程中将其添加到大语言模型的内部激活中，从而实现细粒度的引导。在实验中，我们从生成解释的特异性和事实性方面评估了该方法的引导有效性。此外，我们还在一项人类专家研究中对解释进行了评估。

    arXiv:2608.29215v1 Announce Type: new  Abstract: To effectively enable people to understand new topics, explanations should be tailored to their backgrounds and abilities. So far, prompting alone has been shown to be insufficient for creating such explanations and other computational methods are missing. Therefore, this paper investigates whether LLMs can be steered to generate explanations that are tailored to a specific group of people. To this end, we propose an approach that first identifies group-specific attributes in terms of explanatory style and knowledge of a specific target group. Building on activation engineering, it then computes attribute-based steering vectors and adds them to the internal activations of an LLM during inference to enable a fine-grained steering. In our experiments, we assess the steering effectiveness of our approach in terms of specificity and factuality of the generated explanations. Additionally, we evaluate the explanations in a study with human exp
    
[^202]: 自动化研究人员能够可靠地缓解对齐失败

    Automated Researchers Can Reliably Mitigate Alignment Failures

    [https://arxiv.org/abs/2608.28945](https://arxiv.org/abs/2608.28945)

    自动化对齐研究员（AAR）通过后训练方法能够可靠地缓解10种对齐失败并泛化到更大的模型，其效果甚至优于28名经验丰富的人类研究员在八小时内开发的方法。

    

    自动化对齐研究可能会加速实现与人类对齐的AI的进程，但这是否真的有效却难以衡量。幸运的是，许多对齐失败，例如欺骗、谄媚和越狱，已经可以通过公开基准来衡量。我们研究了自动化对齐研究员能否通过后训练来缓解对齐失败，方法是提出训练方法和数据，以同时优化多个安全基准，同时保持通用能力。在10种对齐失败中，最强的AAR方法显著减少了目标对齐失败，并能泛化到留出的基准测试、多轮行为审计，以及比目标模型大4.7倍的模型。作为人类基线，28名经验丰富的研究人员获得了最多八小时的时间来为相同的基准开发方法，但他们的方法表现不如最好的AAR方法。将人类想法作为AAR的初始研究方向并不能改善结果。

    arXiv:2608.28945v1 Announce Type: new  Abstract: Automating alignment research may accelerate progress toward aligned AI, but whether it does is hard to measure. Luckily, many alignment failures, such as deception, sycophancy, and jailbreaks, are already measurable by public benchmarks. We study whether automated alignment researchers (AARs) can post-train to mitigate alignment failures by proposing training methods and data to simultaneously optimize multiple safety benchmarks, while preserving general capability. Across 10 alignment failures, the strongest AAR methods significantly reduce the targeted alignment failures and generalize to a held-out benchmark, multi-turn behavioral audits, and models up to 4.7 times larger than the target model. As a human baseline, 28 experienced researchers receive up to eight hours to develop methods for the same benchmarks, but their methods underperform the best AAR methods. Using human ideas as the AARs' initial research direction does not impro
    
[^203]: VocalAffectBench：评估AI音频模型中的语音情感识别能力

    VocalAffectBench: Evaluating Vocal Emotion Recognition in AI Audio Models

    [https://arxiv.org/abs/2608.28932](https://arxiv.org/abs/2608.28932)

    该论文提出了VocalAffectBench——一个包含273段人工录制音频、仅用于测试的公开基准，用于评估AI音频模型从原始音频中识别语音情感的能力，结果显示现有最强模型准确率仅46.5%，表明当前AI音频模型的情感识别能力远未达到稳健水平。

    

    语音产品日益需要那些存在于语音中但转录文本里缺失的情感线索。我们推出了VocalAffectBench，这是一个公开的、仅用于测试的基准，用于评估AI音频模型能否从原始音频中识别出所表达的语音情感。该基准包含273段人工录制的英文WAV音频片段，来自51个说话人账户，总计1.95小时，涵盖七个情感标签：愤怒、厌恶、恐惧、快乐、中性、悲伤和惊讶，每个类别39个片段。所有基线模型仅基于音频进行评估，不使用转录文本或上下文元数据。在六个已发布的基线模型中，平均准确率为35.5%。表现最强的基线模型gemini_3_5_flash在七分类任务上达到46.5%，高于14.3%的随机基线水平，但距离稳健的情感识别仍有很大差距。一项辅助性的效价分桶分析将标签映射为积极、中性和消极三类，并排除了效价模糊的“惊讶”类别。在该合并分析下的总体准确率……（摘要截断）

    arXiv:2608.28932v1 Announce Type: new  Abstract: Voice products increasingly need affective cues that are present in speech but absent from transcripts. We introduce VocalAffectBench, a public, test-only benchmark for evaluating whether AI audio models can identify expressed vocal emotion from raw audio. The benchmark contains 273 human-recorded English WAV clips from 51 speaker accounts totaling 1.95 hours across seven labels: angry, disgusted, fearful, happy, neutral, sad, and surprised, with 39 clips per class. All baselines are evaluated from audio alone, without transcripts or contextual metadata. Across six released baselines, average accuracy is 35.5%. The strongest baseline, gemini_3_5_flash, reaches 46.5% on the seven-way task, above the 14.3% random baseline but far from robust emotion recognition. A secondary valence-bucket analysis maps labels into positive, neutral, and negative classes, excluding surprised because its valence is ambiguous. Aggregate accuracy under this co
    
[^204]: AutoScientist-Quant：面向量化投资自动化研究的自进化编码智能体

    AutoScientist-Quant: Self-Evolving Coding Agents for Automatic Research in Quantitative Investment

    [https://arxiv.org/abs/2608.28632](https://arxiv.org/abs/2608.28632)

    提出AutoScientist-Quant框架，将量化研究建模为预算约束下的搜索问题，通过单一自进化控制器统一决策Alpha生成、因子库选择和模型调优，实现从假设到可部署策略的全流程自动化，并修复了评估流程中的前视偏差问题。

    

    大语言模型智能体能够发现Alpha因子，然而现有方法存在三个弱点：搜索过程无法在运行中自适应调整；自动化通常止步于Alpha生成，而因子库选择和模型选择仍需人工完成；Alpha发现过程可能通过循环反馈或代码问题窥探到测试窗口。我们提出AutoScientist-Quant，一个自进化的搜索过程，将量化研究视为一个受预算约束的搜索问题。单一控制器基于剩余预算对所有决策进行条件化，在每一轮决定是改进、组合、转向还是停止，选择扩展哪个节点，生成多少个Alpha，以及如何从共享记忆中检索历史轨迹。同一核心随后从因子库中进行选择并调整模型，实现了从假设到可部署策略的完整闭环。我们还审查了从先前工作复用的评估流程，修复了两个前视偏差问题，并保持反馈窗口与测试窗口互不相交。

    arXiv:2608.28632v1 Announce Type: new  Abstract: Large language model agents can discover alphas, yet current methods have three weaknesses. The search cannot adapt during the run, automation usually ends at alpha generation while library selection and model choice stay manual, and alpha discovery can read the test window through loop feedback or code problems. We present AutoScientist-Quant, a self evolving search process that regards quantitative research as one budgeted search problem. A single controller conditions every decision on the remaining budget, choosing at each round whether to improve, combine, pivot, or stop, which node to expand, how many alphas to generate, and how to retrieve past trajectories from the shared memory. The same core then selects from the library and tunes the model, closing the loop from hypothesis to deployable strategy. We also review the evaluation pipeline reused from prior work, fix two lookahead problems, and keep the feedback window disjoint fro
    
[^205]: 通过博弈论视角审视AI对齐：综述

    AI Alignment through a Game-theoretic Lens: A Survey

    [https://arxiv.org/abs/2608.27910](https://arxiv.org/abs/2608.27910)

    本综述以博弈论视角系统梳理AI对齐研究，围绕偏好多样性、对齐优先级和时间动态三大挑战组织文献，阐明了博弈论分析真正发挥作用之处以及构建鲁棒、自适应、可验证AI系统仍待解决的难题。

    

    随着大语言模型和日益强大的AI智能体被部署到高风险场景中，使其与复杂的人类价值观保持一致已成为核心挑战。现有的对齐方法虽然在提升有用性、无害性和可控性方面卓有成效，但往往难以捕捉那些依赖于上下文、不具传递性、并由动态多方交互塑造的真实世界偏好。本综述通过博弈论的视角审视AI对齐研究。具体而言，它围绕关键的博弈论要素组织近期进展，并围绕三大挑战综合梳理相关文献：偏好多样性、对齐优先级和时间动态。这一视角阐明了当前对齐方法在哪些方面真正受益于博弈论分析，哪些方面的框架应用较为宽松，以及在构建鲁棒、自适应、可验证的AI系统方面仍面临哪些挑战。

    arXiv:2608.27910v1 Announce Type: cross  Abstract: As large language models and increasingly capable AI agents are deployed in high-risk settings, aligning them with complex human values has become a central challenge. Existing alignment methods, while effective in improving helpfulness, harmlessness, and controllability, often struggle to capture real-world preferences that are context-dependent, non-transitive, and shaped by dynamic multi-party interactions. This survey reviews AI alignment through a game-theoretic lens. Specifically, it organizes recent progress around key game-theoretic elements and synthesizes the literature along three challenges: preference diversity, alignment priority, and temporal dynamics. This perspective clarifies where current alignment methods genuinely benefit from game-theoretic analysis, where the framework is looser, and what challenges remain in building robust, adaptive, and verifiable AI systems.
    
[^206]: 面向已知任务音频大语言模型评估的生成式音频调用审计

    Auditing Generative Audio Calls for Known-Task Audio-LLM Evaluation

    [https://arxiv.org/abs/2608.27817](https://arxiv.org/abs/2608.27817)

    该论文将音频大语言模型的评估建模为受控的调用决策问题，发现在已知封闭集任务上，有监督编码器（如CLAP和WavLM）无需调用生成式音频模型即可取得接近最优的准确率，从而揭示了传统“波形提示对比ASR转录”的评估方式混淆了声学证据获取与生成模型调用这两个因素。

    

    语音和音频大语言模型通常通过比较波形提示是否优于自动语音识别（ASR）转录文本来进行评估。对于已知的封闭集任务，这种比较混淆了两个因素：获取声学证据的途径，以及调用生成式音频模型的需求。我们将这一区分评估为一个受控的调用决策问题。对于每个样本，一个策略可以在以下选项中做出选择：保留转录文本标签、使用来自对比语言-音频预训练（CLAP）、音频频谱图Transformer（AST）或WavLM的编码器证据，或调用Qwen2-Audio、Qwen2.5-Omni或MOSS-Audio；其中决定性的消融实验在保持选择器和开发协议不变的前提下移除所有生成式操作。在VocalSound数据集上，转录文本的准确率仅为0.296，说明确实需要波形信息。然而，有监督的CLAP和WavLM对照方法在完全不调用生成式音频模型的情况下分别达到了0.850和0.854的准确率。带有生成式操作的选择器在使用12.5%的调用预算的情况下达到了0.925的准确率（摘要在此处截断）。

    arXiv:2608.27817v1 Announce Type: cross  Abstract: Speech and audio LLMs are often evaluated by asking whether a waveform prompt beats an automatic speech recognition (ASR) transcript. For known closed-set tasks, that comparison conflates two factors: access to acoustic evidence and the need to call a generative audio model. We evaluate this distinction as a controlled call-decision problem. For each example, a policy chooses among keeping a transcript label, using encoder evidence from Contrastive Language-Audio Pretraining (CLAP), Audio Spectrogram Transformer (AST), or WavLM, and calling Qwen2-Audio, Qwen2.5-Omni, or MOSS-Audio; the decisive ablation removes all generative actions while keeping the selector and development protocol fixed. On VocalSound, transcripts reach 0.296 accuracy, so waveform information is needed. Yet supervised CLAP and WavLM controls reach 0.850 and 0.854 with no generative audio calls. A selector with generative actions reaches 0.925 accuracy using 12.5% c
    
[^207]: SURE-Challenge：在语音大模型生成之前评估语音证据

    SURE-Challenge: Evaluating Speech Evidence Before Speech-LLM Generation

    [https://arxiv.org/abs/2608.27783](https://arxiv.org/abs/2608.27783)

    该论文提出 SURE-Challenge 基准，用于评估语音大模型在生成回答之前对不支持输入（静音、噪声、合成音调、嘈杂语音）的拒绝能力，并证明一个简单的“能量加 Whisper 分数”规则可将不支持输入的拒绝数从 15/204 提升至 196/204，同时不损失有效输入的准确率。

    

    语音大模型（Speech LLMs）通常在其作出回答之后才被评估，尽管操作系统首先必须决定是否应将音频波形发送给模型。我们为此准入步骤定义了“语音不支持拒绝评估挑战”（Speech-Unsupported Rejection Evaluation Challenge，简称 SURE-Challenge）。该基准将源自 LibriSpeech 的转录和首词问答任务与不支持的输入——静音、有色噪声、合成音调以及来源模糊的嘈杂语音——配对，并采用互不相交的来源划分以防止泄漏。前端消融实验使用 Qwen2-Audio；随后将选定的“能量加 Whisper 分数”规则在六个语音/音频大模型之前进行重放验证。在经过泄漏筛查的 474 行 SURE-Extended 测试集上，原始 Qwen2-Audio 仅拒绝 204 个不支持输入中的 15 个，而固定规则可拒绝其中 196 个，且支持样本的准确率保持不变。外部检查界定了这一数字的边界：随着 Whisper 分数阈值的收紧，Common Voice 的保留率随之下降，而变速嘈杂语音在 54 个片段中仅产生 18 到 24 个被拒绝的片段。

    arXiv:2608.27783v1 Announce Type: cross  Abstract: Speech LLMs are usually graded after they answer, although an operating system first has to decide whether a waveform should be sent to the model. We define the Speech-Unsupported Rejection Evaluation Challenge (SURE-Challenge) for this admission step. The benchmark pairs LibriSpeech-derived transcription and first-word question answering with unsupported silence, colored noise, synthetic tones, and source-ambiguous babble under disjoint source splits. Front-end ablations use Qwen2-Audio; the selected energy-plus-Whisper-score rule is then replayed before six speech/audio LLMs. On the 474-row leakage-screened SURE-Extended test set, raw Qwen2-Audio rejects 15/204 unsupported inputs, whereas the fixed rule rejects 196/204 and leaves supported accuracy unchanged. External checks delimit this number: Common Voice retention drops as the Whisper-score threshold is tightened, and no-speed babble gives 18 to 24 rejected clips out of 54 across
    
[^208]: RATIO：科学文献中跨类型构思操作检索的基准

    RATIO: A Benchmark for Retrieval Across Typed Ideation Operations in Scientific Literature

    [https://arxiv.org/abs/2608.27394](https://arxiv.org/abs/2608.27394)

    RATIO基准首次定义了三种科学构思操作（Address、Broaden、Specify）的检索任务，并利用远距离监督扩展到大规模语料库，为科学文献的灵感检索提供了新范式。

    

    arXiv:2608.27394v1 公告类型：新 摘要：检索到的科学文献可以为人与AI科学家提供灵感。灵感可以采取不同形式：先前的工作可能直接建议如何解决问题，或在不同抽象层次上指出方向——放大到更一般的视角或缩小到具体实现。我们引入RATIO（跨类型构思操作检索），这是一个大规模基准，其中相关性由三种操作定义，我们称之为构思动作：Address检索针对所提出问题的潜在方法，Broaden检索更一般的表述，Specify检索具体实例。RATIO是通过一种通用方法从CS文献中数百万篇全文科学论文构建而成，该方法将话语标记远距离监督——先前仅用于分类——扩展到语料库级检索，并结合了广泛的LLM和人工审核。实验表明，操作-

    arXiv:2608.27394v1 Announce Type: new  Abstract: Retrieved scientific literature can serve as inspiration for both human and AI scientists. Inspiration can take different forms: prior work may directly suggest how to address a problem, or surface directions at different levels of abstraction - zooming out to a more general view or zooming in to a concrete realization. We introduce RATIO (Retrieval Across Typed Ideation Operations), a large-scale benchmark in which relevance is defined by three operations which we name ideation moves: Address retrieves potential approaches for stated problems, Broaden retrieves more general formulations, and Specify retrieves concrete instantiations. RATIO is constructed from millions of full-text scientific papers across CS literature via a general recipe that extends discourse-marker distant supervision - previously used only for classification - to corpus-scale retrieval, combined with extensive LLM and human vetting. Experiments show that operation-
    
[^209]: Padamitra：古典梵语的基于语境的词汇表生成

    Padamitra: Grounded Glossary Generation for Classical Sanskrit

    [https://arxiv.org/abs/2608.25038](https://arxiv.org/abs/2608.25038)

    本文提出了一个名为Padamitra的基准任务，用于古典梵语的基于语境的词汇表生成，并发现指令微调和显式分词能提升性能，但sandhi和samasa复合词的过度分词是主要挑战。

    

    我们引入了基于语境的词汇表生成任务，这是一个结构化任务，要求模型从诗节-翻译对中恢复具有语义意义的梵语短语，并产生基于翻译的释义，将传统的“pada”注释实践形式化为一个可评估的自然语言处理目标。我们从《瓦尔米基·罗摩衍那》和《圣典博伽瓦谭》构建了一个包含31,316个诗节-翻译-词汇表三元组的基准数据集，并配有两个评估指标：用于短语恢复的Jaccard指数和用于语义一致性的“意义忠实度”。在零样本、少样本和指令微调的Gemma-3n-E4B、Gemma-3-12B、Phi-4和Qwen3.5-9B模型变体中，指令微调显著优于提示方法，而显式分词也带来了性能提升。错误分析表明，对sandhi和samasa复合词的过度分词是主要的失败模式，这指出了形态建模是忠实梵语词汇分解的关键瓶颈。

    arXiv:2608.25038v1 Announce Type: new  Abstract: We introduce grounded glossary generation, a structured task requiring models to recover semantically meaningful Sanskrit phrases and produce translation-grounded meanings from a sloka-translation pair, formalizing the traditional patha commentary practice as an evaluable NLP objective. We construct a benchmark of 31,316 sloka-translation-glossary triples from the Valmiki Ramayana and Srimad Bhagavatam, paired with two metrics: Jaccard for phrase recovery and Meaning Faithfulness for semantic consistency. Across zero-shot, few-shot, and instruction fine-tuned variants of Gemma-3n-E4B, Gemma-3-12B, Phi-4, and Qwen3.5-9B, instruction fine-tuning substantially outperforms prompting, while explicit segmentation yields gains. Error analysis identifies over-segmentation of sandhi and samasa compounds as the dominant failure mode, pointing to morphological modeling as the key bottleneck for faithful Sanskrit lexical decomposition.
    
[^210]: CyberFactory：利用真实世界实例扩展网络安全能力

    CyberFactory: Scaling Cyber Security Capabilities with Instances from the Wild

    [https://arxiv.org/abs/2608.23181](https://arxiv.org/abs/2608.23181)

    CyberFactory是一个统一开源框架，通过将真实世界CVE漏洞转化为可执行任务实例，并整合数据构建、轨迹合成和模型训练，从而扩展网络安全能力。

    

    随着大型语言模型（LLMs）在编码能力上的不断进步，它们在网络安全领域的潜力日益受到研究关注，其中闭源LLMs（如Mythos）展现了先进的网络安全能力。然而，现有的开源工作仍存在局限：前沿开源权重模型未提供可复现的网络安全训练解决方案，开源训练方案聚焦于孤立任务且缺乏可扩展的代理数据，而扩展代理式滚动需要强大的领域先验知识。在这项工作中，我们引入了\textbf{CyberFactory}，一个统一的开源框架，它在概念验证（PoC）生成、漏洞修补和网络安全问答（CyberQA）之间连接了数据构建、轨迹合成和模型训练。CyberFactory将公开的漏洞工件（包括来自真实世界的CVE）转化为可执行且可验证的任务实例。它进一步使用...

    arXiv:2608.23181v1 Announce Type: cross  Abstract: As large language models (LLMs) continue to advance in coding capabilities, their potential in cybersecurity has drawn increasing research attention, with closed-source LLMs (e.g., Mythos) delivering advanced cybersecurity capabilities. However, existing open-source efforts remain limited: frontier open-weight models do not provide reproducible cybersecurity training solutions, open-source training solutions focus on isolated tasks and lack scalable agentic data, and scaling agentic rollouts requires strong domain priors. In this work, we introduce \textbf{CyberFactory}, a unified open-source framework that connects data construction, trajectory synthesis, and model training across proof-of-concept (PoC) generation, vulnerability patching, and cybersecurity question answering (CyberQA). CyberFactory transforms public vulnerability artifacts, including CVEs from the wild, into executable and verifiable task instances. It further uses a 
    
[^211]: PropUQ-MAS：面向LLM多智能体系统的传播感知不确定性量化

    PropUQ-MAS: Propagation-Aware Uncertainty Quantification for LLM Multi-Agent Systems

    [https://arxiv.org/abs/2608.22130](https://arxiv.org/abs/2608.22130)

    本文提出了PropUQ-MAS，一种通过通信图结构捕捉多智能体系统中错误传播的不确定性量化框架，显著提升了可靠性估计性能。

    

    基于LLM的多智能体系统（MAS）通过角色专业化智能体之间的通信来解决复杂任务。然而，智能体间的依赖性引入了超出单个智能体故障的可靠性风险。例如，中间消息中的错误可能被下游智能体继承并放大。现有的不确定性量化（UQ）方法主要针对孤立响应或单智能体推理，因此无法捕捉MAS中的不确定性传播。为此，我们提出了PropUQ-MAS，一种错误传播感知的UQ框架，它将MAS执行表示为通信结构化图，并通过结合局部不确定性与来自上游消息继承的不确定性来估计每个步骤的可靠性。大量实验表明，PropUQ-MAS持续改善了MAS中的UQ，平均相对增益在AUROC上提高了+6.10%，在PRR上提高了+47.58%。

    arXiv:2608.22130v1 Announce Type: cross  Abstract: LLM-based multi-agent systems (MAS) solve complex tasks through communication among role-specialized agents. However, inter-agent dependencies introduce reliability risks beyond isolated agent failures. For instance, errors in intermediate messages could be inherited and amplified by downstream agents. Existing uncertainty quantification (UQ) methods mainly target isolated responses or single-agent reasoning, and therefore fail to capture uncertainty propagation in MAS. To this end, we propose PropUQ-MAS, an error propagation-aware UQ framework that represents MAS execution as a communication-structured graph and estimates each step's reliability by combining local uncertainty with uncertainty inherited from upstream messages. Extensive experiments demonstrate that PropUQ-MAS consistently improves UQ in MAS, with average relative gains of +6.10% in AUROC and +47.58% in PRR.
    
[^212]: ToSCA：基于对话代理时间与策略抽象的层次强化学习

    ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents

    [https://arxiv.org/abs/2608.21969](https://arxiv.org/abs/2608.21969)

    本文提出一种两级层次强化学习框架，结合话语级策略抽象与词元级解码，并引入双粒度奖励机制，以提升对话代理在复杂交互中的性能。

    

    人类在日常互动和思考中具有多个层次的时间抽象能力，例如概念感知和策略规划。受此启发，我们为对话代理提出了一种两级层次强化学习（RL）框架，弥合了以往基于词元级别或话语级别RL方法之间的差距。该框架基于两级MDP开发，其中词元级别的响应解码依赖于话语级别的动作，即显式文本策略。基于理论推导和效率考虑，我们使用DQN求解高层评论家，使用PPO求解低层演员-评论家。为进一步缓解奖励稀疏性并促进收敛，我们还设计了双粒度奖励机制，将话语级别的满意度评分与词元级别的内在动机和K-L惩罚相结合。在日常对话和情感支持对话上的实验表明，所提方法优于现有基线。

    arXiv:2608.21969v1 Announce Type: new  Abstract: Humans have multiple levels of temporal abstractions on daily interaction and thinking, such as concept perception and strategic planning. Inspired by this nature, we propose a two-level hierarchical reinforcement learning (RL) framework for conversational agents, bridging the gap between previous token-level or utterance-level RL methods. Developed on a two-level MDP, the token-level response decoding is conditioned on the utterance-level action, the explicit textual strategies. Based on theoretical derivation and efficiency consideration, we use DQN to solve the high-level critic and PPO to solve the low-level actor-critic. To further alleviate the reward sparsity and facilitate the convergence, we also design the dual-granularity reward mechanism, in which the utterance-level satisfaction score is integrated with token-level intrinsic motivation and K-L penalty. Experiments on both daily and emotional support conversations show that o
    
[^213]: HiDiffTIR：面向多轮工具集成推理的分层难度感知策略优化

    HiDiffTIR: Hierarchical Difficulty-Aware Policy Optimization for Multi-Turn Tool-Integrated Reasoning

    [https://arxiv.org/abs/2608.21863](https://arxiv.org/abs/2608.21863)

    本文提出HiDiffTIR框架，通过分层难度感知的信用分配机制，在多轮工具集成推理中更精确地区分轨迹和推理步骤的难度，从而提升强化学习训练效果。

    

    arXiv:2608.21863v1 公告类型：交叉 摘要：工具集成推理（TIR）是LLM代理通过与外部工具迭代交互解决复杂任务的基本能力。强化学习（RL）已成为实现这一能力的主导范式。然而，现有方法通常分配统一的轨迹级优势，并平等对待所有正确的工具调用，忽略了轨迹和推理步骤间不同的难度和学习价值。这可能导致学习信号不精确，无法充分区分平凡和具有挑战性的工具使用模式。为解决这一局限性，我们提出了HiDiffTIR，一种用于多轮TIR的分层难度感知策略优化框架。HiDiffTIR在轨迹级和回合级执行难度感知的信用分配，使策略能够聚焦于更具信息量的轨迹和更难的推理步骤。值得注意的是，这种细粒度优化是通过...

    arXiv:2608.21863v1 Announce Type: cross  Abstract: Tool-Integrated Reasoning (TIR) is a fundamental capability for LLM agents to solve complex tasks by interacting with external tools iteratively. Reinforcement Learning (RL) has become the dominant paradigm for enabling this capability. However, existing approaches typically assign uniform trajectory-level advantages and treat all correct tool calls equally, ignoring the varying difficulty and learning value across trajectories and reasoning steps. This can lead to imprecise learning signals that do not adequately distinguish between trivial and challenging tool-use patterns. To address this limitation, we propose HiDiffTIR, a Hierarchical Difficulty-aware policy optimization framework for multi-turn TIR. HiDiffTIR performs difficulty-aware credit assignment at both trajectory and turn levels, enabling the policy to focus on more informative trajectories and harder reasoning steps. Notably, this fine-grained optimization is achieved wi
    
[^214]: FormalTCS：大型语言模型前沿端到端形式化理论计算机科学研究基准测试

    FormalTCS: Benchmarking End-to-End Frontier Formal Theoretical Computer Science Research of Large Language Models

    [https://arxiv.org/abs/2608.20153](https://arxiv.org/abs/2608.20153)

    该论文提出了一个专家验证的基准测试FormalTCS，用于评估大型语言模型在前端理论计算机科学研究中的端到端能力，并发现自动形式化是当前模型面临的最大瓶颈。

    

    arXiv:2608.20153v1 公告类型：新 摘要：大型语言模型（LLMs）在自动化理论计算机科学（TCS）研究方面展现出日益增长的潜力，然而现有基准测试远未达到真实研究场景的要求。我们引入了\ourbenchmark，这是一个经过专家验证的基准测试，用于评估LLMs在前沿、端到端TCS研究中的表现。\ourbenchmark包含175个实例，这些实例取自2025-2026年间被STOC、FOCS、SODA和COLT会议接受的论文，保留了论文特有的定义、假设和证明依赖关系，并配有专家验证的Lean形式化表述和证明。对领先LLMs的评估显示，当前模型仍远未可靠地完成整个研究流程。特别是，自动形式化是最尖锐的瓶颈：最佳模型在将自然语言声明转换为形式化定理陈述时仅达到11.5分，而在证明人类提供的形式化陈述时，Pass@8得分可达28.6分。基于\ourbenchmark，我们进一步开发了...

    arXiv:2608.20153v1 Announce Type: new  Abstract: Large language models (LLMs) have shown growing potential for automated theoretical computer science (TCS) research, yet existing benchmarks remain far from realistic research settings. We introduce \ourbenchmark, an expert-validated benchmark for evaluating LLMs on frontier, end-to-end TCS research. \ourbenchmark contains $175$ instances drawn from papers accepted to STOC, FOCS, SODA, and COLT in 2025-2026, preserving paper-specific definitions, assumptions, and proof dependencies, with expert-verified Lean formalizations and proofs. Evaluations of leading LLMs reveal that current models remain far from reliably completing the full research pipeline. In particular, autoformalization is the sharpest bottleneck: the best model achieves only $11.5$ on translating natural-language claims into formal theorem statements, compared with $28.6$ Pass@8 when proving human-provided formal statements. Building on \ourbenchmark, we further develop an
    
[^215]: SWE-bench Science：编码代理能否解决科学中的工程任务？

    SWE-bench Science: Can Coding Agents Resolve Engineering Tasks in Science?

    [https://arxiv.org/abs/2608.19799](https://arxiv.org/abs/2608.19799)

    本文提出了SWE-bench Science，一个针对科学软件工程的仓库级基准，并揭示即使最佳代理在科学任务中成功率也低于50%，主要因科学知识不足等四种机制导致失败。

    

    arXiv:2608.19799v1 公告类型：新 摘要：软件日益成为科学仪器本身的一部分，使得科学代码中的故障不仅可能损害程序行为，还可能损害科学结论所依据的证据。然而，现有对编码代理的评估主要强调整体任务成功率，对于代理在修复科学软件时为何失败提供的见解有限。我们引入了 \textbf{SWE-bench Science}，一个面向科学软件工程的仓库级基准测试，包含来自20个科学领域98个GitHub仓库的119个任务。每个任务被组织为三种范式之一：问题驱动、专家探索和工程集成。即使是最佳表现的代理 \textbf{Claude Code with Opus-5 (max)}，其pass@1也低于50\%，凸显了科学软件工程带来的巨大挑战。我们识别出四种反复出现的失败机制：科学知识不足、领域特定工具使用错误、错误诊断不准确以及测试覆盖不充分。

    arXiv:2608.19799v1 Announce Type: new  Abstract: Software increasingly functions as part of the scientific instrument itself, making failures in scientific code capable of compromising not only program behavior but also the evidence underlying scientific conclusions. Yet existing evaluations of coding agents largely emphasize aggregate task success, providing limited insight into why agents fail when repairing scientific software. We introduce \textbf{SWE-bench Science}, a repository-level benchmark for scientific software engineering comprising 119 tasks from 98 GitHub repositories across 20 scientific domains. Each task is organized into one of three paradigms: Issue-driven, Expert-exploratory, and Engineering-integration. Even the best-performing agent, \textbf{Claude Code with Opus-5 (max), achieves a pass@1 below 50\%}, highlighting the substantial challenges posed by scientific software engineering. We identify four recurring failure mechanisms: deficits in scientific knowledge o
    
[^216]: 当无关文本起作用：多模态大语言模型中的仿射边际偏移

    When Irrelevant Text Matters: Affine Margin Shifts in Multimodal Large Language Models

    [https://arxiv.org/abs/2608.19208](https://arxiv.org/abs/2608.19208)

    本文发现多模态大语言模型中任务无关文本会通过一致的仿射变换系统性偏移模型决策边际，而非产生随机噪声。

    

    多模态大语言模型（MLLMs）经常暴露于辅助文本上下文中，其对视觉接地任务的影响尚未得到充分探索。在本文中，我们通过将任务无关上下文视为二元视觉判断框架内的受控干预来研究其影响。通过保持提示结构不变而改变辅助输入，我们观察到无关文本在多种基准上持续偏向模型预测。为了超越性能指标，我们通过二元候选之间的对数概率差异定义的决策边际来表征这种敏感性。我们的分析揭示了一种稳健的几何规律：上下文条件边际遵循其无上下文对应物的一致仿射变换。这一发现表明，无关上下文并非表现为非结构化的随机噪声，而是一种可估计的扭曲。

    arXiv:2608.19208v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) are frequently exposed to auxiliary textual context, the impact of which on visually grounded tasks remains underexplored. In this paper, we investigate the influence of task-irrelevant context by formulating it as a controlled intervention within a binary visual judgment framework. By maintaining an invariant prompt structure while varying auxiliary inputs, we observe that irrelevant text consistently biases model predictions across diverse benchmarks. To move beyond performance metrics, we characterize this sensitivity through a decision margin defined by the log-probability difference between binary candidates. Our analysis reveals a robust geometric regularity: contextconditioned margins follow a consistent affine transformation of their context-free counterparts. This finding demonstrates that irrelevant context does not manifest as unstructured stochastic noise but as a estimable distortion 
    
[^217]: 无金标准标签下AI生成数据的去偏推断：通过多重不完美测量进行识别

    Debiased Inference for AI-Generated Data without Gold-Standard Labels: Identification via Multiple Imperfect Measurements

    [https://arxiv.org/abs/2608.18294](https://arxiv.org/abs/2608.18294)

    本文提出了一种无需金标准标签、利用多重不完美AI测量进行去偏推断的新框架，有效解决了AI测量误差导致的下游分析偏差问题。

    

    越来越多的学者使用AI来测量变量，并将其纳入后续的下游分析。尽管AI测量的变量通常被视为无误差观测，但忽略自动化测量中的预测误差会导致下游分析中的显著偏差和无效置信区间，即使AI测量准确度很高（例如超过90%）。现有的解决方案，如基于设计的有监督学习和预测支持推断，将基于AI的易错测量与金标准标签相结合，但在某些应用领域中，获取金标准标签可能成本高昂且困难。在本文中，我们提出了多重不完美测量的去偏推断（DMM），这是一个结合多个易错AI测量以实现无需金标准标签的有效下游推断的框架。基于CP分解的既有成果，DMM假设这些测量是独立的。

    arXiv:2608.18294v1 Announce Type: cross  Abstract: An increasing number of scholars use AI to measure variables they subsequently include in downstream analyses. Although AI-measured variables are often analyzed as if observed without error, ignoring prediction errors in automated measurement leads to substantial bias and invalid confidence intervals in downstream analyses, even if AI measurement accuracy is high, e.g., above 90%. Existing solutions, such as design-based supervised learning and prediction-powered inference, combine error-prone AI-based measurements with gold-standard labels, which may be costly and difficult to obtain in some application areas.   In this paper, we propose debiased inference with multiple imperfect measurements (DMM), a framework that combines multiple error-prone AI measurements to enable valid downstream inference without gold-standard labels. Building on the established results on CP decomposition, DMM assumes that these measurements are independent 
    
[^218]: 大型语言模型能否处理信念与事实取决于提问方式

    Whether LLMs Can Navigate Beliefs and Facts Depends on How You Phrase It

    [https://arxiv.org/abs/2608.17809](https://arxiv.org/abs/2608.17809)

    大型语言模型处理用户信念与事实的能力受表达动词影响，且问题根源在于任务混淆，思维链提示可部分缓解但效果不一。

    

    arXiv:2608.17809v1 公告类型：新 摘要：人类在日常交流中自然形成并表达信念，例如，“我认为答案是3”或“我猜想那是正确的。”这些信念不可避免地与事实和知识交织在一起，因此对于大型语言模型（LLMs）来说，能够同时处理这些内容是可取的，因为它们越来越多地部署在面向用户的场景中。先前的研究表明，即使能力较强的LLMs在承认基于错误信息的用户信念方面也表现出系统性弱点。我们将此评估扩展到10个LLMs，涵盖18种认知表达，发现弱点的规模和方向取决于表达信念所用的动词，事实与错误信息之间的准确性差距从“我模糊记得”上的+50%到“我严重怀疑”上的-14%不等。我们进一步表明，这一现象源于任务混淆：模型默认对潜在主张进行事实核查，覆盖了用户表达的信念；思维链（chain-of-thought）提示可以缓解此问题，但效果因表达方式而异。

    arXiv:2608.17809v1 Announce Type: new  Abstract: Humans naturally form and express beliefs in daily communication, e.g., "I think the answer is 3" or "I suppose that's right." Such beliefs inevitably intertwine with fact and knowledge, making the ability to handle them in tandem desirable for large language models (LLMs), as they are increasingly deployed in user-facing settings. Prior work showed that even capable LLMs exhibit a systemic weakness in acknowledging user beliefs grounded in incorrect information. We extend this evaluation to 10 LLMs across 18 epistemic expressions and find that the size and direction of the weakness depend on the verb used to express the belief, with the accuracy gap between factual and false information ranging from +50% on "I vaguely remember" to -14% on "I seriously doubt". We further show that the phenomenon stems from task confusion: models default to fact-checking the underlying claim, overriding the user's stated belief; chains of thought that exp
    
[^219]: 深度思维对齐：用于视频推理的轨迹级潜在蒸馏

    Deep Thought Alignment: Trajectory-Level Latent Distillation for Video Reasoning

    [https://arxiv.org/abs/2608.16316](https://arxiv.org/abs/2608.16316)

    本文提出Latent-OPD方法，通过在轨迹末端进行潜在表示蒸馏，弥补了传统输出级蒸馏在视频推理中无法直接约束中间推理状态的不足，从而提升小模型从大模型迁移推理能力的效率。

    

    大型多模态模型（LMMs）在视频推理中一直受到处理海量视觉信息的高计算成本的阻碍。这一困境促使将大模型的推理能力转移到更小、更高效的模型上。策略内蒸馏（OPD）通过匹配学生生成轨迹上的输出令牌分布，提供了一种有前景的解决方案。然而，视频推理通常依赖于跨多个帧累积的证据。在此背景下，输出级监督仅捕捉通过令牌预测表达的信息，并未直接约束推理过程中形成的潜在表示。为解决这一局限性，我们提出了Latent-OPD，该方法通过轨迹级潜在蒸馏增强了OPD。具体而言，我们的方法聚焦于每条轨迹结束时的位置，其中隐藏状态有效地总结了累积的视觉证据。

    arXiv:2608.16316v1 Announce Type: cross  Abstract: Large Multimodal Models (LMMs) for video reasoning have long been hindered by the high computational cost of processing vast amounts of visual information. This dilemma motivates the transfer of the reasoning capabilities of large models to smaller, more efficient ones. On-Policy Distillation (OPD) offers a promising solution by matching output-token distributions along student-generated trajectories. However, video reasoning often depends on evidence accumulated across multiple frames. In this context, output-level supervision only captures information expressed through token predictions and does not directly constrain the latent representations formed during reasoning. To address this limitation, we propose Latent-OPD, which augments OPD with trajectory-level latent distillation. Specifically, our method focuses on the position at the end of each trajectory, where hidden states effectively summarize the accumulated visual evidence an
    
[^220]: 孟加拉语区域方言的多方言神经机器翻译系统

    Poly-Dialectal Neural Machine Translation System for Bangla Regional Dialects

    [https://arxiv.org/abs/2608.12018](https://arxiv.org/abs/2608.12018)

    本文提出了一种无需标准语言中转的统一多方言神经机器翻译系统，覆盖12种孟加拉语区域方言，并构建了迄今最大的多方言平行语料库，显著提升了低资源方言的翻译性能。

    

    摘要：区域方言变异对孟加拉语的自然语言处理（NLP）构成了根本性挑战，超过2.4亿说话者使用多种区域变体进行交流，这些变体在音系、形态和词汇方面与标准口语孟加拉语（SCB）存在显著差异。当代神经机器翻译（NMT）架构和大语言模型（LLMs）主要假设语言分布是均匀的，导致在翻译低资源区域方言时性能严重下降。在本工作中，我们提出了一个统一的多方言神经机器翻译系统，能够跨越12种孟加拉语区域方言进行多方向翻译，而无需通过中间标准语言作为中转。我们编制了迄今为止最大的孟加拉语多方言平行语料库，包含12种方言的51,531个非空平行句对，并整合了2,500个专家验证的双向句对。

    arXiv:2608.12018v1 Announce Type: new  Abstract: Regional dialectal variation poses a fundamental challenge to natural language processing (NLP) in Bangla, where over 240 million speakers communicate across diverse regional variants that diverge significantly from Standard Colloquial Bangla (SCB) in phonology, morphology, and lexicon. Contemporary neural machine trans- lation (NMT) architectures and large language models (LLMs) predominantly as- sume a homogeneous language distribution, resulting in severe performance degra- dation when translating low-resource regional dialects. In this work, we present a unified Poly-Dialectal Neural Machine Translation System capable of multi-directional translation across 12 Bangla regional dialects without routing through an inter- mediary standard pivot. We compile the largest multi-dialect parallel corpus for Bangla to date, comprising 51,531 non-null parallel sentence pairs across 12 di- alects, incorporating 2,500 expert-verified, bidirectiona
    
[^221]: 弥合英语-阿拉伯语医学知识差距：通过因果层选择实现定向低秩适应

    Bridging the English-Arabic Medical Knowledge Gap: Targeted Low-Rank Adaptation via Causal Layer Selection

    [https://arxiv.org/abs/2608.00207](https://arxiv.org/abs/2608.00207)

    该论文通过机制可解释性分析发现阿拉伯语医学知识已存在于大模型的中间表示中但未能在输出端浮现，据此提出仅针对分歧层窗口进行适配的定向低秩适应（TLoRA）方法，在医学问答等任务上超越了全网络LoRA及零样本、少样本基线。

    

    大型语言模型（LLMs）在英语医学任务中表现强劲，但在阿拉伯语上性能显著下降，这一差距被普遍归因于训练数据有限。我们通过调整透镜探测和因果激活修补系统地研究了这一假设，发现阿拉伯语医学知识其实已存在于模型的中间表示中，但未能在输出端呈现。这一机制层面的洞察启发了一种定向适应策略：我们不再微调整个网络，而是提出定向低秩适应（TLoRA），将适配限制在跨语言表示出现分歧的层窗口内，即故障显现的输出层的上游。我们在多选医学问答任务上评估了TLoRA，该方法优于全网络LoRA、零样本和少样本基线。我们进一步在简答题生成和多轮临床对话上对其进行评估，结果显示它在无需全网络微调的情况下仍具有竞争力。

    arXiv:2608.00207v2 Announce Type: replace  Abstract: Large Language Models (LLMs) perform strongly in English medical tasks but degrade substantially in Arabic, a gap widely attributed to limited training data. We systematically investigate this assumption via tuned lens probing and causal activation patching, and find that Arabic medical knowledge is present in intermediate model representations but fails to surface at the output. This mechanistic insight motivates a targeted adaptation strategy: rather than fine-tuning the full network, we propose Targeted Low-Rank Adaptation (TLoRA), restricted to the layer window where cross-lingual representations diverge, upstream of the output layers where the failure manifests. We evaluate TLoRA on multiple-choice medical QA, where our approach outperforms full-network LoRA, zero-shot, and few-shot baselines. We further evaluate it on short-answer generation and multi-turn clinical dialogue, where it performs competitively without the need for 
    
[^222]: 基于多模态大语言模型的计算幽默研究：方法、数据集、评估与挑战

    Computational Humor with Multimodal LLMs: Methods, Datasets, Evaluation, and Challenges

    [https://arxiv.org/abs/2607.19011](https://arxiv.org/abs/2607.19011)

    本综述系统梳理了多模态大语言模型在理解表情包、漫画等视觉幽默方面的方法、数据集与评估协议，并构建了以能力为中心的“识别—解释推理—生成”层次框架，揭示了该领域从任务专用融合模型向大模型方法的转变及面临的评估捷径等核心挑战。

    

    表情包、漫画和连环画中的多模态幽默对人工智能系统来说仍然十分困难，因为其意图含义依赖于非字面机制、共享的文化知识和交际意图，而非对场景的字面描述。本综述聚焦于单图和多格作品中的视觉幽默理解，同时将幽默生成视为一个新兴的下游前沿方向。我们将相关文献置于以往幽默、讽刺以及通用多模态大语言模型（MLLM）综述的背景下，并采用以能力为中心的层次结构进行组织，涵盖识别、解释与推理、以及生成三个层面。在这一视角下，我们综合分析了基准设计、评估协议和建模范式，梳理了该领域从任务特定的融合模型向基于多模态对齐、证据支撑推理和受控生成的大模型方法的演进历程。最后，我们指出了该领域进展面临的主要障碍：易产生捷径学习的评估（原文摘要在此处截断）。

    arXiv:2607.19011v2 Announce Type: replace-cross  Abstract: Multimodal humor in memes, cartoons, and comics remains difficult for AI systems because intended meaning depends on non-literal mechanisms, shared cultural knowledge, and communicative intent rather than literal scene description. This survey focuses on visual humor understanding in single-image and multi-panel artifacts, while treating humor generation as an emerging downstream frontier. We position the literature against prior humor, sarcasm, and general MLLM surveys and organize it using a capability-centric hierarchy spanning recognition, interpretation and reasoning, and generation. Under this lens, we synthesize benchmark design, evaluation protocols, and modeling paradigms, tracing the field's shift from task-specific fusion models to large-model approaches based on multimodal alignment, evidence-grounded reasoning, and controlled generation. We conclude by highlighting the main barriers to progress: shortcut-prone eval
    
[^223]: 一个会自我教学的分类器：用于动态文档分类的自我改进冻结门控训练（SIFT）

    A Classifier That Teaches Itself: Self-Improving, Frozen-gate Training (SIFT) for Dynamic Document Classification

    [https://arxiv.org/abs/2607.18358](https://arxiv.org/abs/2607.18358)

    SIFT提出了一种自改进的动态文档分类服务：用廉价的SPLADE+LightGBM流水线处理分类，仅将低置信度页面交给LLM裁判，其判定结果回流标注语料库持续教导廉价模型，从而免去前期标注工作并让准确率随使用不断提升。

    

    文档分类在实验室里是已被解决的问题，在企业中却是尚未解决的问题。阻碍通常并非模型架构，而是必须在建模之前完成的标注工程，以及机构对于让已存在的模型自我再训练的担忧。我们提出了SIFT（Self-Improving, Frozen-gate Training，自我改进冻结门控训练），一种动态分类器服务，同时攻克这两个问题。SIFT通过一条刻意设计得廉价、基于CPU的流水线来提供分类服务——由SPLADE稀疏编码器连接LightGBM分类头——并仅将低置信度的少数页面升级至LLM裁判。裁判的判定结果会被写回标注语料库，因此昂贵的模型持续地教导廉价的模型：升级率不断下降，语料库从生产流量中自然增长而非依赖前期标注工作，准确率随使用而持续复合提升。接入一个新的文档系列只需要一个声明式包、标签空间、锚定短语，以及……

    arXiv:2607.18358v2 Announce Type: replace  Abstract: Document classification is a solved problem in the laboratory and an unsolved one in the enterprise. The blocker is rarely model architecture; it is the labeling project that must precede a model and the institutional fear of letting a model retrain itself once one exists. We present SIFT (Self-Improving, Frozen-gate Training), a dynamic classifier service, which attacks both. SIFT serves classification from a deliberately cheap, CPU-bound pipeline, a SPLADE sparse encoder feeding a LightGBM head, and escalates only the low-confidence minority of pages to an LLM judge. The judge's verdicts are written back into a labeled corpus, so the expensive model continuously teaches the cheap one: the escalation rate falls, the corpus grows from production traffic rather than from an up-front annotation effort, and accuracy compounds with use. Onboarding a new document family requires only a declarative bundle, label space, anchor phrases, and 
    
[^224]: 对齐微调如何塑造大语言模型中谄媚性及相关线索诱导偏差的表示？

    How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs?

    [https://arxiv.org/abs/2607.18114](https://arxiv.org/abs/2607.18114)

    该研究发现大语言模型对谄媚性等线索诱导偏差的敏感性主要源于对齐微调而非预训练，且对齐模型中每种偏差都存在一个可被解码和干预的线性表示方向，可用于恢复无偏答案。

    

    现代大语言模型对于输入提示中一些出奇简单且无关紧要的变化异常敏感：一句随意的暗示、一个标注错误的少样本示例，或是一个伪造的先前助手回合，常常会使原本正确的答案发生翻转。我们研究了这种敏感性——涵盖谄媚性及相关线索诱导偏差——存在于模型内部的位置。我们在五个模型家族和七种偏差类型上，从隐藏状态中提取每种偏差的方向，并通过三种方法对其进行三角验证：探针分析、留一数据集（LODO）迁移以及因果干预。研究发现，这种敏感性主要由对齐微调而非预训练所塑造：预训练基础模型通常对这些偏差的屈从程度要低得多，其激活中除问题内容之外的线索特定信号也弱得多。在对齐后的模型中，每种偏差都存在一个连贯的线性方向，我们既可以对其进行解码，也可以沿其进行干预，从而在所有模型家族中恢复无偏的答案。

    arXiv:2607.18114v2 Announce Type: replace-cross  Abstract: Modern LLMs are alarmingly susceptible to surprisingly simple immaterial changes of input prompts: a casual hint, an incorrectly labeled few-shot example, or a fake prior assistant turn often flips an originally correct answer. We study where this susceptibility, spanning sycophancy and related cue-induced biases, lives inside the model. Across five model families and seven bias types, we extract a per-bias direction from hidden states and triangulate it through three measures: probing, leave-one-dataset-out (LODO) transfer, and causal intervention. The susceptibility is largely shaped by alignment tuning rather than pretraining: pretrained base models generally cave much less to these biases, and their activations carry much weaker cue-specific signal beyond question content. Within aligned models, each bias has a coherent linear direction that we can both decode and steer along, recovering the unbiased answer across every fam
    
[^225]: 零幻觉，由构造保证：面向可信赖企业AI的幻觉感知分层监督

    Zero Hallucination, by Construction: Hallucination-Aware Layered Oversight for Trustworthy Enterprise AI

    [https://arxiv.org/abs/2607.17883](https://arxiv.org/abs/2607.17883)

    本文提出HALO保证架构，将幻觉从“可消除的问题”重新定义为“可控制的失效模式”，通过六层防御机制把“零幻觉”从模型属性转变为系统强制实施的属性，从而实现可信赖的企业级AI。

    

    企业不会部署它们无法信任的AI智能体，而最常被引用的不信任原因就是幻觉：自信、流畅但完全不真实的输出。常见的应对方式是等待一个不会产生幻觉的模型出现。我们认为这是一个错误的目标。大型语言模型从构造上就具备生成无依据文本的能力，任何规模扩大都无法消除这种可能性；附加在原始模型上的忠实度评判器能捕捉一些错误，但仍会让其他错误漏网，甚至经过精心策划的检索管道也被证明会伪造引用。我们重新定义了目标：“零幻觉”不是模型所拥有的属性，而是系统所强制实施的属性。我们提出了HALO（幻觉感知分层监督），这是一种保证架构，它将幻觉视为一种可控制的失效模式，而非可消除的失效模式。HALO由六层防御组成：基于检索到的、经过批准的内容进行接地生成……

    arXiv:2607.17883v2 Announce Type: replace-cross  Abstract: Enterprises will not deploy AI agents they cannot trust, and the most-cited reason for distrust is hallucination: confident, fluent output that is simply not true. The common response is to wait for a model that does not hallucinate. We argue that this is the wrong target. Large language models are, by construction, capable of generating unsupported text, and no amount of scale removes the possibility; a faithfulness judge bolted onto a raw model catches some errors but still ships others, and even well-curated retrieval pipelines have been shown to fabricate citations. We reframe the goal: "zero hallucination" is not a property a model possesses but a property a system enforces. We present HALO (Hallucination-Aware Layered Oversight), an assurance architecture which treats hallucination as a containable failure mode rather than an eliminable one. HALO composes six layers of defense: grounded generation over retrieved, approved
    
[^226]: Scope3Trace：基于证据的可持续发展报告中范围3温室气体排放的识别与提取

    Scope3Trace: Evidence-Based Identification and Extraction of Scope 3 GHG Emissions from Sustainability Reports

    [https://arxiv.org/abs/2607.17122](https://arxiv.org/abs/2607.17122)

    提出了基于证据的Scope3Trace框架，通过结合PDF/OCR文档处理、LLM辅助页面定位与表格重建以及规则-LLM混合提取，从真实ESG报告中实现可解释、可追溯的范围3温室气体排放信息提取。

    

    范围3温室气体（GHG）排放占企业碳足迹的大部分，但由于披露信息稀疏、报告文档格式异构以及证据可追溯性有限，难以进行规模化分析。现有方法通常依赖大语言模型从ESG报告中提取排放信息，但往往缺乏明确的证据锚定，或需要昂贵的人工标注与验证来确保提取的可靠性。为应对这些挑战，我们提出了Scope3Trace，一个基于证据的信息提取框架，旨在从真实世界的ESG和可持续发展报告中提取可解释、可追溯的范围3排放信息。该框架集成了一个文档信息提取流水线，包括PDF收集与OCR解析、LLM辅助的页面定位与表格重建，以及针对组织和建筑层面排放的规则-LLM混合提取。

    arXiv:2607.17122v2 Announce Type: replace  Abstract: Scope 3 greenhouse gas (GHG) emissions account for the majority of corporate carbon footprints, yet remain difficult to analyze at scale due to sparse disclosures, heterogeneous report document formats, and limited evidence traceability. Existing approaches typically rely on large language models to extract emissions information from ESG reports, but often lack explicit evidence grounding or depend on costly manual annotation and verification to ensure extraction reliability. To address these challenges, we propose Scope3Trace, an evidence-grounded information extraction framework designed to extract interpretable and traceable Scope 3 emissions information from real-world ESG and sustainability reports. The framework integrates a document information extraction pipeline that performs PDF collection and OCR parsing, LLM-assisted page localization and table reconstruction, and hybrid rule-LLM extraction of organization- and building-l
    
[^227]: Anamnesis：一个用于大规模背景故事条件化调查模拟的开源平台

    Anamnesis: An Open-Source Platform for Large-Scale Backstory-Conditioned Survey Simulation

    [https://arxiv.org/abs/2607.10628](https://arxiv.org/abs/2607.10628)

    Anamnesis是一个开源平台，通过结构化叙事背景故事对大语言模型进行条件化，实现了在虚拟人群上进行人口可控、大规模且支持多模态的调查模拟。

    

    我们提出了Anamnesis，一个使用大语言模型进行人口统计学可控调查模拟的交互式系统。Anamnesis是开源的，专为非技术背景的用户和研究人员设计，使其能够在虚拟人群而非真实人类受试者上进行调查工具的原型设计与压力测试。该平台在统一的网页界面中实现了近期提出的Anthology和Alterity框架，这两个框架利用结构化的叙事背景故事来调节模型响应。系统支持开放式生成、概率性人口重采样以及多模态（图像和音频）调查。我们通过两个案例研究对该系统进行了评估：（1）复制皮尤研究中心“美国趋势小组”（ATP）中关于政治类型学和生物医学议题的部分调查；（2）模拟《纽约客》漫画配文大赛中的人类偏好。在两个案例中，Anamnesis所产生的观点分布都更接近真实数据。

    arXiv:2607.10628v2 Announce Type: replace-cross  Abstract: We present Anamnesis, an interactive system for demographically controllable survey simulation using large language models. Open-source and designed for non-technical users/researchers, Anamnesis enables the prototyping and stress-testing of survey instruments on virtual populations rather than real human subjects. The platform operationalizes the recently introduced Anthology and Alterity frameworks, which use structured narrative backstories to condition model responses, within a unified web interface. It supports open-ended generation, probabilistic demographic resampling, and multimodal (image and audio) surveys. We evaluate the system through two case studies: (1) replicating segments of Pew Research Center's American Trends Panel (ATP) on political typology and biomedical issues and (2) emulating human preference in the New Yorker Caption Contest. In both cases, Anamnesis produces opinion distributions that more closely m
    
[^228]: 最终检查点并不足够：沿训练轨迹分析潜在推理的忠实性

    Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories

    [https://arxiv.org/abs/2607.06648](https://arxiv.org/abs/2607.06648)

    该研究揭示了仅评估训练结束时的最终检查点不足以判断潜在推理的忠实性——高任务准确率可能与低反事实响应性共存，因此必须沿整个训练轨迹追踪行为与激活层面的忠实性证据。

    

    潜在推理在连续的隐状态中执行多步推理，有望实现更紧凑、更高效的推理。然而，这些不透明的状态引发了一个忠实性问题：潜在推理步骤是否真正驱动了最终答案的生成。先前的工作在选定的检查点上研究这一问题，并报告了若干不忠实的行为。这种端点视角使得忠实性证据在训练过程中如何演变这一问题未被考察。我们利用经过验证的反事实编辑以及对潜在推理状态的干预，在整个训练过程中追踪行为层面的证据和基于激活的证据。我们发现，高任务准确率可能与低反事实响应性共存：随着准确率的提升，响应性反而可能下降，且不同的潜在推理方法遵循各自不同的轨迹。在ProsQA上，输出对范数噪声替换的敏感性与反事实响应性一同下降，尽管该结果取决于替换的具体方式。

    arXiv:2607.06648v2 Announce Type: replace-cross  Abstract: Latent reasoning performs multi-step inference in continuous hidden states, promising more compact and efficient reasoning. However, these opaque states raise a question of faithfulness: whether the latent reasoning steps drive the final answer. Prior work studies this question at selected checkpoints and reports several unfaithful behaviors. This endpoint view leaves how evidence of faithfulness evolves during training unexamined. We track behavioral and activation-based evidence across training using verified counterfactual edits and interventions on the latent reasoning states. We find that high task accuracy can coexist with low counterfactual responsiveness: as accuracy improves, responsiveness can decline, and different latent reasoning approaches follow distinct trajectories. On ProsQA, output sensitivity to norm-noise replacement declines alongside counterfactual responsiveness, although the result depends on the replac
    
[^229]: ALEE：通过以英语为中心的最小对实现任意语言的嵌入评估

    ALEE: Any-Language Evaluation of Embeddings via English-Centric Minimal Pairs

    [https://arxiv.org/abs/2607.00171](https://arxiv.org/abs/2607.00171)

    ALEE框架利用抽象含义表示（AMR）生成具有受控细粒度语义变化的英语最小对，并将其与275多种语言的翻译配对，从而实现了对任意语言的文本嵌入模型进行跨语言、段落级别的精细评估诊断。

    

    文本嵌入是语义相似性任务的标准方法，但其评估仍然是一个未解决的挑战。当前的基准测试是静态的，仅覆盖有限的语言集合，通常局限于特定领域，容易过拟合，且对低资源语言的代表性不足。为了解决这些局限性，我们引入了ALEE，这是一个将Sentence Smith（Li等人，2025）扩展到跨语言和段落级别的框架。ALEE使用抽象含义表示（AMR）生成具有受控、细粒度语义变化的英语最小对，并将其与目标语言的翻译配对。这种方法能够对任何拥有英语平行数据的语言的模型进行针对性诊断。我们在涵盖三个平行数据集的多种嵌入模型和275多种语言上进行了大规模实证研究。在ALEE上，性能在不同语言、文本长度和语言现象之间存在显著差异。

    arXiv:2607.00171v2 Announce Type: replace  Abstract: Text embeddings are standard for semantic similarity tasks, yet their evaluation remains an open challenge. Current benchmarks are static, cover only a limited set of languages, are often domain-specific, susceptible to overfitting, and poorly representative of low-resource languages. To address these limitations, we introduce ALEE, a framework that extends Sentence Smith (Li et al., 2025) to the cross-lingual and paragraph level. ALEE uses Abstract Meaning Representations (AMR) to generate English minimal pairs with controlled, fine-grained semantic shifts, which are paired with translations in target languages. This approach enables targeted diagnostics for models in any language with English parallel data. We conduct a large-scale empirical study across a diverse set of embedding models and 275+ languages spanning three parallel datasets. On ALEE, performance varies substantially across languages, text lengths, and linguistic phen
    
[^230]: DigitalCoach：人类与智能体计算机使用辅导中的沟通与视觉定位差距

    DigitalCoach: Communication and Grounding Gaps in Human and Agentic Computer Use Coaching

    [https://arxiv.org/abs/2606.31980](https://arxiv.org/abs/2606.31980)

    该论文构建了包含72场人类专家-新手计算机使用辅导会话的多模态数据集DigitalCoach，揭示了当前最先进模型虽能生成与人类相似的辅导语句，但在解释、错误诊断和视觉定位方面显著不足，导致学习者被动跟随指令而非深度参与学习。

    

    智能体在自动化软件任务方面的能力日益增强，但它们能否教会人类自己使用软件呢？我们推出了DigitalCoach，这是一个多模态数据集，包含72场人类专家与新手之间的计算机使用辅导会话，涵盖五款软件应用中基于28.1小时屏幕和输入事件录制的22,752轮对话。我们利用DigitalCoach评估最先进的模型能否教会人类如何使用计算机。自动化评估表明，模型在辅导方式上与人类存在差异：模型提供更多直接指令，但更少的解释、错误诊断和知识检验问题。当我们固定辅导方法时，模型生成的语句与人类参考相似，但在视觉上下文定位方面表现较差。交互式评估证实，模型辅导者会导致学习者被动跟随指令而缺乏深入参与，并且在视觉定位方面存在不足。

    arXiv:2606.31980v2 Announce Type: replace-cross  Abstract: Agents are increasingly capable of automating software tasks, but can they teach humans how to use software themselves? We introduce DigitalCoach, a multimodal dataset of 72 human expert-novice computer use coaching sessions consisting of 22,752 dialogue turns grounded in 28.1 hours of screen and input event recordings across five software applications. We use DigitalCoach to evaluate whether state-of-the-art models can teach humans how to use computers. Automated evaluation shows that models differ from humans in how they coach: models provide more direct instructions, but fewer explanations, error diagnoses, and knowledge-check questions. When we fix the coaching method, models produce utterances similar to human references yet poorly grounded in visual context. Interactive evaluation confirms that model coaches cause learners to passively follow instructions without deeper engagement and fall short in visual grounding. Digit
    
[^231]: 大语言模型能否想象二元道德困境之外的替代方案？

    Can LLMs Imagine Moral Alternatives Beyond Binary Dilemmas?

    [https://arxiv.org/abs/2606.31213](https://arxiv.org/abs/2606.31213)

    该论文提出MoralAltDataset数据集，通过在307个二元道德困境中引入折中和重构的替代选项，发现当替代方案可用时人类与15个LLM的道德选择分布均发生显著转变且一致性增强，但存在关键差异——LLM明显偏好GPT-5创作的替代方案，而人类的选择不受创作来源影响，揭示了机器与人类在“想象道德替代方案”能力上的差距。

    

    随着大语言模型（LLM）越来越多地充当道德顾问和道德智能体，它们必须应对相互竞争的价值观之间的冲突。然而，以往关于道德困境的研究忽视了人类道德认知的一个核心方面：在给定选项之外想象替代方案。我们提出了MoralAltDataset数据集，其中包含307个顾问型和人机交互型智能体困境，并为其补充了折中方案与重新构建的替代选项。我们在二元选项和四选项两种设置下比较了人类与LLM的判断。在人类被试和15个LLM中，两种设置下的总体道德选择分布存在显著差异，且折中方案往往比两个原始二元选项中的任何一个都更受青睐。结果显示出价值观的转变，以及在替代方案上人类与LLM之间更强的一致性。按创作来源分层的结果揭示了一种描述性差距：人类选择替代方案的比率在不同来源之间相似，而LLM则明显更频繁地选择由GPT-5创作的替代方案。随后我们比较了人类与……（原文摘要在此处被截断）

    arXiv:2606.31213v2 Announce Type: replace-cross  Abstract: As LLMs increasingly serve as moral advisors and agents, they must address conflicts between competing values. Yet prior work on moral dilemmas overlooks a central aspect of human moral cognition: imagining alternatives beyond the given options. We introduce MoralAltDataset, comprising 307 Advisor and AI-facing Agent dilemmas augmented with compromise and reframed alternatives. We compare human and LLM judgments in binary and four-option settings. Across human participants and 15 LLMs, aggregate moral choice distributions differ substantially between the two settings, with compromise often preferred over either original binary option. Results show value shifts and stronger human-LLM agreement on alternatives. Source-stratified results reveal a descriptive gap: human alternative-selection rates are similar across authoring sources, whereas LLMs select GPT-5-authored alternatives substantially more often. We then compare human-au
    
[^232]: 面向大语言模型智能体规划的自进化世界模型

    Self-Evolving World Models for LLM Agent Planning

    [https://arxiv.org/abs/2606.30639](https://arxiv.org/abs/2606.30639)

    提出自进化世界模型框架 WorldEvolver，通过情景记忆、语义记忆和选择性前瞻三个模块，在保持智能体与模型参数完全冻结的情况下持续修正部署时的上下文，从而提升长时程 LLM 智能体规划中前瞻预测的可靠性与下游决策成功率。

    

    世界模型为长时程大语言模型（LLM）智能体提供了一种有原则的前瞻能力：在执行动作之前预测其后果。然而，不可靠的前瞻预测可能被忽略、被误用，甚至降低下游决策的质量。本文提出了 WorldEvolver，一个自进化世界模型框架，它在保持下游智能体和所有模型参数冻结的前提下，在部署阶段不断修正自身的上下文。WorldEvolver 集成了三个模块：(i) 情景记忆，通过基于检索的模拟来利用真实动作转移；(ii) 语义记忆，从预测与观测的不匹配中提取持久性的启发式规则；(iii) 选择性前瞻，在将预测整合进智能体推理上下文之前过滤掉低置信度的预测。我们在 ALFWorld 和 ScienceWorld 上评估了 WorldEvolver，在 Word2World 上测量世界模型的预测准确率，并在 AgentBoard 上测量下游智能体的成功率。

    arXiv:2606.30639v2 Announce Type: replace  Abstract: World models offer a principled way to equip long-horizon LLM agents with foresight: predictions of action consequences before execution. However, unreliable foresight can be ignored, misused, or even degrade downstream decision-making. In this paper, we introduce WorldEvolver, a self-evolving world model framework that revises its deployment-time context while keeping the downstream agent and all model parameters frozen. WorldEvolver integrates three modules: (i) Episodic Memory, which exploits real action transitions through retrieval-based simulation; (ii) Semantic Memory, which extracts persistent heuristic rules from prediction-observation mismatches; and (iii) Selective Foresight, which filters low-confidence predictions before integrating them into agent reasoning context. We evaluate WorldEvolver on ALFWorld and ScienceWorld, measuring world model prediction accuracy on Word2World and downstream agent success rate on AgentBoa
    
[^233]: 大型语言模型能否可靠地自我报告对抗性前缀注入，以及如何实现？

    Can LLMs Reliably Self-Report Adversarial Prefills, and How?

    [https://arxiv.org/abs/2606.23671](https://arxiv.org/abs/2606.23671)

    本论文发现，大型语言模型无法可靠地自我识别对抗性前缀注入攻击，其内省信号主要来自安全推理，且受权重方向与探测方式影响，现有训练方法无法稳定改善这一能力。

    

    摘要：先前的研究表明，大型语言模型（LLMs）在良性任务上表现出内省能力。我们将这一问题扩展到安全情境，并考察模型能否可靠地识别其先前的响应是由对抗性前缀注入攻击引发的。在十个从3B到70B规模的开源指令微调LLM以及四个安全基准上，没有任何模型能可靠地识别自身受损输出，模型对预填充响应声称意图的平均比率仅为25.3%。内省信号主要源于对安全和拒绝的推理。将模型权重与拒绝方向正交化，会使预填充和自然输出的声称率差距缩小至接近零，尽管该方向并非其唯一中介因素。该信号还依赖于探测方式：将问题框架为内部意图与外部篡改，会在相同模型上引发定性不同的响应。训练修改（如增加拒绝训练数据）无法稳定地提升内省能力，这表明当前模型缺乏对攻击的稳健自我意识。

    arXiv:2606.23671v4 Announce Type: replace  Abstract: Prior work shows that large language models (LLMs) exhibit introspective capability on benign tasks. We extend the question to safety contexts and examine how reliably a model can recognize that its own prior response was elicited by an adversarial prefill attack. Across ten open-weight instruction-tuned LLMs from 3B to 70B and four safety benchmarks, no model reliably recognizes its own compromised outputs, with models claiming intent on prefilled responses at an average rate of 25.3%. Introspective signal stems primarily from reasoning about safety and refusal. Orthogonalizing models' weights against the refusal direction collapses the gap between claim rates on prefilled and natural outputs to near zero, though the direction is not its unique mediator. The signal also depends on the probe: framing the question as internal intention versus external tampering elicits qualitatively different responses on the same models. Training mod
    
[^234]: 基于能量的Transformer作为阅读难度的预测器

    Energy-Based Transformers as Predictors of Reading Difficulty

    [https://arxiv.org/abs/2606.23382](https://arxiv.org/abs/2606.23382)

    本文首次将基于能量的Transformer度量引入计算心理语言学，证明该能量度量在多个阅读时间语料库中是阅读难度的稳健预测因子，其解释力显著超越传统的惊讶度和注意力熵度量，并与Hopfield网络等联想记忆理论建立了形式化联系。

    

    Transformer语言模型已成为建模人类句子处理的成熟工具，其中惊讶度（surprisal）和注意力熵等度量作为阅读难度的有效预测因子，共同捕捉处理负荷的互补方面。本文探索了一类相关的Transformer模型：基于能量的Transformer，它为联想记忆模型提供了原则性的形式化联系，使句法处理研究与Hopfield网络和密集联想记忆的更广泛文献直接对接。据我们所知，这是计算心理语言学领域首次对基于能量的Transformer度量进行的探索。在多个阅读时间语料库（Natural Stories、UCL眼动追踪、UCL自定步速阅读）上，能量度量是阅读时间的稳健预测因子，在所有三个语料库中都提供了超越惊讶度和熵的显著拟合增益。在关于关系从句处理的受控实验中（摘要在此处截断）。

    arXiv:2606.23382v2 Announce Type: replace-cross  Abstract: Transformer language models have become established tools for modeling human sentence processing, with measures such as surprisal and attention entropy serving as effective predictors of reading difficulty that together capture complementary aspects of processing load. Here, we explore a related class of transformer models: energy-based transformers, which provide a principled formal link to associative memory models, bringing processing research into direct contact with the broader literature on Hopfield networks and dense associative memory. To our knowledge, this is the first exploration of an energy-based transformer measure in computational psycholinguistics. Across reading-time corpora (Natural Stories, UCL eye-tracking, UCL self-paced reading), the energy measure is a robust predictor of reading times, providing significant fit beyond surprisal and entropy in all three. In a controlled experiment on relative clause proce
    
[^235]: DART：面向混合推理模型免训练自适应思考预算的草稿一致性路由

    DART: Draft-Agreement Routing for Training-Free Adaptive Thinking Budgets in Hybrid Reasoning Models

    [https://arxiv.org/abs/2606.23181](https://arxiv.org/abs/2606.23181)

    DART是一种免训练的自适应路由框架，通过比较两个无思考草稿的一致性来决定是否需要深度推理并预测思考预算，在大幅减少思考token消耗的同时保持甚至提升模型在数学和代码任务上的准确率。

    

    混合推理模型既可以直接回答问题，也可以花费额外的token进行扩展思考。一个实用的路由器应该为每个查询在这两种模式之间进行选择，使简单问题避免不必要的推理，而困难问题获得足够的预算来完成答案。现有的路由器虽朝此方向发展，但它们通常需要带标签的训练数据，或预先固定思考预算，忽略了来自模型本身的答案层面的证据。我们提出了DART，一个免训练的路由框架，它采样两个廉价的“无思考”草稿，当草稿一致时接受直接回答，当草稿不一致时根据草稿熵预测思考预算。在主要对比实验中，DART在大多数设置下保持或提升了“始终思考”模式的准确率，同时减少了思考token的使用。准确率在奥数级数学上最高提升+9.0分，在基于执行等价性的代码任务上最高提升+22.5分，同时思考token的使用量下降。

    arXiv:2606.23181v2 Announce Type: replace  Abstract: Hybrid reasoning models can answer directly or spend extra tokens on extended thinking. A practical router should choose between these modes for each query, so easy problems avoid unnecessary reasoning and hard problems receive enough budget to finish the answer. Existing routers move in this direction, but they typically require labeled training data or fix thinking budgets up front, ignoring answer-level evidence from the model itself. We introduce DART, a training-free routing framework that samples two cheap no-think drafts, accepts direct answering when the drafts agree, and predicts a thinking budget from draft entropy when they disagree. Across the main comparisons, DART preserves or improves always-thinking accuracy in most settings while reducing thinking-token use. Accuracy improves by up to +9.0 points on Olympiad-level math and by up to +22.5 points on code under execution-based equivalence, while thinking-token use drops
    
[^236]: 压缩何时有益、何时有害：思维链蒸馏的条件感知分析

    When Compression Helps and When It Hurts: Condition-Aware Analysis of Chain-of-Thought Distillation

    [https://arxiv.org/abs/2606.21704](https://arxiv.org/abs/2606.21704)

    该工作将思维链蒸馏中的压缩方法沿重要性标准、重构层级和压缩预算三个维度系统解耦，发现压缩的收益与代价严格依赖于粒度、领域和长/短CoT模式等条件——步骤级标准收敛于共享推理主干，而token级剪枝需要符号感知信号。

    

    思维链蒸馏将多步推理能力从大型推理模型迁移到更小的学生模型，但冗长的教师推理轨迹会推高训练与推理成本。现有CoT压缩方法分为两大类——选择性剪枝与生成式重写，但已有研究使关键因素相互纠缠：剪枝中粒度与重要性标准相互混淆，重写中重构层级很少被单独分离，压缩预算也缺乏跨领域、跨模式的系统评估。我们沿三个维度重新审视CoT压缩：重要性标准、重构层级与压缩预算。在两个模型家族、数学与通用领域以及长/短CoT模式下对这些维度进行全面扫描，我们发现：(i) 重要性标准的效用严格受粒度支配：步骤级标准收敛于共享的推理主干，而token级剪枝则需要符号感知的信……（原文摘要至此截断）

    arXiv:2606.21704v2 Announce Type: replace  Abstract: Chain-of-Thought (CoT) distillation transfers multi-step reasoning from large reasoning models to smaller students, but verbose teacher traces inflate both training and inference cost. Existing CoT compression methods fall into two families, selective pruning and generative rewriting, yet prior studies have left key factors entangled: granularity is confounded with importance criteria in pruning, restructuring level is rarely isolated in rewriting, and compression budgets are not systematically evaluated across domains or regimes. We recast CoT compression along three dimensions: importance criterion, restructuring level, and compression budget. Sweeping these across two model families, Math and General domains, and Long-/Short-CoT regimes, we find that (i) importance criterion utility is strictly governed by granularity: step-level criteria converge on a shared reasoning backbone, while token-level pruning requires symbol-aware sign
    
[^237]: 弥合语义缓存中的运营差距

    Closing the Operational Gap in Semantic Caching

    [https://arxiv.org/abs/2606.19719](https://arxiv.org/abs/2606.19719)

    该论文指出PR-AUC指标会误导语义缓存系统的部署决策，提出了缓存感知的P-CHR AUC指标和运营保留率ORR，并将离线与部署质量间的运营差距分解为可恢复的阈值效用部分和由数据集正例率决定的不可约简结构部分。

    

    语义缓存通过为语义相似的查询提供缓存响应来降低大语言模型（LLM）的推理成本。标准做法是使用PR-AUC来评估这些系统，但该指标仅衡量分数的排序质量，而忽略了分数在固定阈值下是否可用。我们证明这种错位会导致系统性的糟糕部署选择，因为PR-AUC最高的模型在实际运行中往往表现最差。我们引入了精确率-缓存命中率（P-CHR）AUC这一缓存感知指标，用于衡量不同缓存利用率水平下的精确率；以及运营保留率（ORR），用于捕捉离线排序质量在部署时的保留程度。我们将离线质量与部署质量之间的运营差距分解为可恢复的阈值效用部分，以及由数据集正例率固定的不可约简的结构部分。我们的实验表明，阈值效用差距由训练目标决定，而非……（摘要原文在此处截断）

    arXiv:2606.19719v3 Announce Type: replace-cross  Abstract: Semantic caching cuts LLM inference costs by serving a cached response to semantically similar queries. Standard practice evaluates these systems using PR-AUC, a metric that only measures how well scores rank and ignores whether they are usable at a fixed threshold. We show this mismatch leads to systematically poor deployment choices, as models with the highest PR-AUC are often the worst in operation. We introduce Precision--Cache Hit Ratio (P-CHR) AUC, a cache-aware metric that measures precision across cache utilization levels, and Operational Retention Rate (ORR), which captures how much offline ranking quality survives at deployment. We decompose the operational gap between offline and deployed quality into a recoverable threshold-utility component and an irreducible structural component fixed by the dataset's positive rate. Our experiments show that the threshold-utility gap is governed by the training objective rather th
    
[^238]: ReproRepo：利用GitHub仓库议题规模化扩展可复现性审计

    ReproRepo: Scaling Reproducibility Audits with GitHub Repository Issues

    [https://arxiv.org/abs/2606.18237](https://arxiv.org/abs/2606.18237)

    ReproRepo提出利用GitHub上人工提交的议题作为天然监督信号，构建了可规模化的可复现性评估框架，并在1,149篇机器学习论文上验证了LLM智能体无需执行代码即可识别真实复现障碍的能力（最佳智能体可覆盖约90%的论文）。

    

    从论文及已发布代码中复现研究结果对科学进步至关重要。现有工作已引入基准来评估LLM智能体能否协助实现可复现性，但由于在数据整理和评估方面依赖大量人工投入，这些基准难以规模化。我们提出了ReproRepo，这是一个可扩展的可复现性评估框架，它利用人工提交的GitHub议题（issues）作为对真实复现障碍天然产生的监督信号。我们在来自主要会议的1,149篇近期机器学习论文上构建了ReproRepo实例，并评估了四种前沿模型-智能体配置。结果表明，LLM智能体即使不执行代码，也能从论文-仓库配对中识别出许多真实世界的可复现性问题：我们研究中表现最佳的智能体，即搭载GPT-5.5的Codex，能为约90%的论文找出至少一个与人工报告语义相关的复现障碍。

    arXiv:2606.18237v2 Announce Type: replace-cross  Abstract: Reproducing research results from papers and released code is central to scientific progress. Existing works have introduced benchmarks to evaluate whether LLM agents can assist with reproducibility, but they are difficult to scale due to their reliance on substantial manual effort for data curation and evaluation. We introduce ReproRepo, a scalable framework for reproducibility evaluation that leverages human-raised GitHub issues as naturally occurring supervision on realistic reproduction blockers. We instantiate ReproRepo on 1,149 recent machine learning papers from major conferences and evaluate four frontier model-agent configurations. Our results show that LLM agents, even without executing code, can identify many real-world reproducibility problems from paper-repository pairs: the best agent in our study, namely Codex with GPT-5.5, surfaces at least one semantically related human-reported blocker for $\sim$90% of papers 
    
[^239]: 基于认知授权评估大语言模型的二阶偏见

    Evaluating Second-Order Bias of LLMs Through Epistemic Entitlement

    [https://arxiv.org/abs/2606.17506](https://arxiv.org/abs/2606.17506)

    该论文提出“二阶偏见”这一新概念——即LLM在评判社会偏见时自身表现出的偏见，并基于认知授权认识论设计了一个逻辑推理任务和两项指标来系统性地测量这种偏见。

    

    arXiv:2606.17506v2 公告类型：替换 摘要：目前对大语言模型（LLM）社会偏见的评估大多集中在模型是否生成或暗示有偏见的内容上。然而，随着LLM越来越多地被用作偏见评判者，它们在评估有偏见内容的方式上可能以更微妙的形式表现出社会偏见，而现有方法无法系统地捕捉到这一点。我们将这种现象称为二阶偏见：即LLM在关于社会偏见的判断中所体现的社会偏见，并通过一个新颖的、有哲学基础的推理任务对其进行评估。借鉴认知授权认识论，我们将偏见概念化为一种错置的基础性知识，它会影响主体的理性探究过程，并由此推导出一个逻辑推理任务，让LLM判断一篇有偏见的文本对谁而言是可接受的或不可接受的。我们开发了两个简单的指标，用于衡量LLM评判者在缺乏充分依据的情况下推断某群体可接受性时的偏见程度，以及这些推断在偏见文本所针对的不同群体之间存在怎样的差异。通过对开源与闭源模型的评估，我们……（原文摘要在此截断）

    arXiv:2606.17506v2 Announce Type: replace  Abstract: Evaluations of social bias in LLMs largely focus on whether models generate or imply biased content. However, as LLMs are increasingly used as judges of bias, they may exhibit social biases in subtler ways in how they evaluate biased content, which current methods do not systematically capture. We call this second-order bias: social bias in an LLM's judgment about social bias, which we evaluate through a novel, philosophically grounded reasoning task. Drawing on entitlement epistemology, we conceptualize bias as misplaced foundational knowledge that shapes an agent's rational inquiry, and derive a logical reasoning task for LLMs to judge to whom a biased text is acceptable or non-acceptable. We develop two simple metrics to measure how biased LLM judges are in inferring demographics for acceptability without sufficient support, and how these inferences vary across groups targeted by biased texts. Evaluating open and closed models, we
    
[^240]: HiMPO：面向长时程智能体低纠缠信用分配的后见之明引导记忆策略优化

    HiMPO: Hindsight-Informed Memory Policy Optimization for Less-Entangled Credit in Long-Horizon Agents

    [https://arxiv.org/abs/2606.16285](https://arxiv.org/abs/2606.16285)

    HiMPO框架将后见相关性作为有界回溯过滤器，为长时程智能体的记忆写入动作分配与下游工具故障等因素解耦的低纠缠信用，并仅对记忆token应用记忆特定优势进行优化。

    

    长时程智能体依赖记忆机制来压缩交互历史，但优化记忆写入面临一个独特的信用分配挑战：记忆更新可能因下游工具故障、噪声观测或推理错误而非其自身贡献而受到奖励或惩罚。我们提出了HiMPO，一个用于为长时程智能体中记忆写入动作分配低纠缠信用的后见之明引导记忆策略优化框架。HiMPO首先通过在相同的预写入状态下比较可从先前记忆和更新后记忆中恢复的任务相关信息，来估计记忆更新的局部效用。然后，它将后见相关性用作一个有界的回溯过滤器，当局部效用得不到目标结果支持时衰减记忆信用。由此产生的记忆特定优势仅应用于记忆token，而轨迹级奖励则优化智能体的其余部分。

    arXiv:2606.16285v2 Announce Type: replace  Abstract: Long-horizon agents rely on memory mechanisms to compress interaction history, but optimizing memory writing faces a distinct credit assignment challenge: a memory update may be rewarded or penalized due to downstream tool failures, noisy observations, or reasoning errors rather than its own contribution. We propose HiMPO, a Hindsight-Informed Memory Policy Optimization framework for assigning less-entangled credit to memory-writing actions in long-horizon agents. HiMPO first estimates the local utility of a memory update by comparing the task-relevant information recoverable from the previous and updated memories under the same pre-write state. It then uses hindsight relevance as a bounded retrospective filter that attenuates memory credit when local utility is not supported by the target outcome. The resulting memory-specific advantage is applied only to memory tokens, while trajectory-level rewards optimize the rest of the agent's
    
[^241]: AfriSUD：一个用于评估非洲语言模型的依存树库集合

    AfriSUD: A Dependency Treebank Collection for Evaluating Models on African Languages

    [https://arxiv.org/abs/2606.12708](https://arxiv.org/abs/2606.12708)

    该论文推出了首个覆盖九种非洲语言的大规模句法标注依存树库集合AfriSUD，并揭示现有模型在这些语言上仍存在显著的句法理解差距。

    

    尽管非洲语言具有语言多样性和全球重要性，但在支持自然语言处理（NLP）的研究和资源中，它们仍然代表性不足。我们旨在通过引入AfriSUD来弥合这一差距，这是首个面向九种多样化非洲语言的大规模句法标注树库集合，涵盖了撒哈拉以南非洲的主要语系和地区。基于表层句法通用依存框架，这项由社区主导的工作提供了高质量的、经母语者验证的数据，能够捕捉诸如黏着和声调等类型学上的关键特征。我们在AfriSUD上评估了一系列模型的词性标注和依存句法分析性能，包括非transformer基线模型、多语言预训练编码器以及大语言模型（LLMs）。我们的结果揭示了一个显著的句法差距：模型在这九种语言上仍表现出明显的局限性，这表明现有架构可能无法完全捕捉这些语言的句法结构（原文摘要在此处截断）。

    arXiv:2606.12708v2 Announce Type: replace-cross  Abstract: Despite their linguistic diversity and global significance, African languages remain underrepresented in research and resources to support NLP. We aim to bridge this gap by introducing AfriSUD, the first large-scale collection of syntactically annotated treebanks for nine diverse African languages spanning major language families and regions across Sub-Saharan Africa. Using the Surface-Syntactic Universal Dependencies (SUD) framework, our community-led effort provides high-quality, native-speaker verified data that capture typological key features such as agglutination and tone. We evaluate a range of models on AfriSUD for part-of-speech tagging and dependency parsing including non-transformer baselines, multilingual pretrained encoders, and LLMs. Our results reveal a significant syntax gap, where models still show clear limitations across the nine languages, suggesting that existing architectures may not fully capture the stru
    
[^242]: DecSelfMask：通过自相关性引导的掩码方法利用无标注文本提升仅解码器模型的分类性能

    DECSELFMASK: Leveraging Unlabeled Text via Self-Relevance-Guided Masking for Decoder-Only Classification

    [https://arxiv.org/abs/2606.09466](https://arxiv.org/abs/2606.09466)

    提出DecSelfMask方法，利用相关性归因引导的掩码策略从无标注文本中创建自监督训练样本，通过下一个词预测重建与任务相关的被掩码部分，从而提升仅解码器模型在分类任务上的性能，尤其适用于标注数据稀缺的医疗领域。

    

    分类任务需要标注数据，而标注数据的收集往往成本高昂、耗时费力，甚至难以实现。医疗领域就是典型情况，该领域的大型数据集通常只有少量标注样本。为了解决这个问题，我们提出了DecSelfMask（通过掩码进行解码器自学习），这是一种增强仅解码器模型在分类任务上性能的方法。我们在常见的自学习方法基础上，利用模型从无标注数据中创建训练样本，并提出了一种新颖的相关性引导掩码策略。我们使用相关性归因方法来确定无标注文本中哪些部分与任务相关，然后通过掩码这些部分来创建自监督训练样本，并训练模型通过下一个词预测来重建这些部分。我们假设这些样本传达了关于无标注数据结构和语义的知识，这些知识对下游任务的性能是有用的。

    arXiv:2606.09466v3 Announce Type: replace  Abstract: Classification tasks require annotated data, which can often be expensive, time-consuming, or even unfeasible to collect. This is the case of the medical domain, where large datasets often have few annotated examples. To address this, we propose DecSelfMask (Decoder Self-learning by Masking), an approach to enhance decoder-only performance on classification tasks. We build on common self-learning approaches by leveraging a model to create training examples from unlabeled data, and propose a novel relevance-guided masking strategy. We use relevance attribution methods to determine what portions of unannotated texts are relevant for a task. We then create self-supervised training examples by masking out those portions, training the model to reconstruct them via next-token-prediction. We hypothesize that those examples convey knowledge about the structure and semantics of unannotated data that can be useful for downstream performance. W
    
[^243]: RECAP：面向提示词持续适应的回归评估

    RECAP: Regression Evaluation for Continual Adaptation of Prompts

    [https://arxiv.org/abs/2606.06698](https://arxiv.org/abs/2606.06698)

    RECAP是一个在严格“先适应后测试”主动协议下、于约束层面评估提示词优化方法持续学习能力（遗忘、回归、前向迁移）的基准，实验发现现有六种方法在面对动态演变的约束时均无显著改进。

    

    生产级智能体系统经常面临不断变化的约束条件，并且必须从下一次交互开始就遵守这些约束。诸如工具调用通知改变合规阈值、或政策更新增加披露要求等场景都符合这一标准，在生产环境中几乎没有出错的空间。这种主动适应设定在部署中很常见，但在当前的基准测试中却缺失，因为现有基准要么假设静态的约束集合，要么采用带有评估反馈的被动式协议。我们提出了RECAP，这是一个在严格的“先适应后测试”主动协议下、于约束层面衡量持续学习现象（遗忘、回归、前向迁移）的基准：提示词优化方法仅接收约束规范，必须在看到任何测试数据之前完成泛化。通过在五个大语言模型和三种约束演进计划下评估六种方法，我们发现这些方法没有表现出显著的改进（性能提升）。

    arXiv:2606.06698v4 Announce Type: replace-cross  Abstract: Production agentic systems routinely face evolving constraints and must comply from the very next interaction. Scenarios like a tool-call notification changing a compliance threshold or a policy update adding disclosure requirements fit this criteria, having close to no room for errors in production. This proactive adaptation setting is common in deployment, but absent from current benchmarks, which assume either static constraint sets or reactive protocols with evaluation feedback. We introduce RECAP, a benchmark that measures continual-learning phenomena (forgetting, regression, forward transfer) at the constraint level under a strictly proactive adapt-then-test protocol: prompt optimization methods receive only the constraint specification and must generalize before seeing any test data. Evaluating six methods across five LLMs and three schedules with evolving constraints, we find that these methods show no significant impro
    
[^244]: 谁在NLP中进行标注？2018至2025年间人类标注报告的大规模评估

    Who Annotates in NLP? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025

    [https://arxiv.org/abs/2606.02255](https://arxiv.org/abs/2606.02255)

    首次对2018至2025年间主要NLP会议中的人类标注报告进行大规模任务级审计，提出统一的标注报告分类体系并借助经验证的LLM抽取流程构建了大规模标注报告数据集，揭示了标注者身份与过程控制等信息在论文中的普遍缺失。

    

    人类标注是许多NLP研究的实证基础，涵盖从数据集构建到模型评估的各个环节，但论文往往不清楚标注者是谁、标注过程如何被控制。我们首次对主要NLP会议中的人类标注报告进行了大规模、任务级别的审计，探究哪些标注细节被记录、哪些缺失，以及报告内容如何随时间、主题、会议和人类判断的预期用途而变化。我们提出了一个统一的标注报告实践分类体系，并在Annotated-gold（一个由人工裁定的金标准，包含41篇论文和72个标注任务）上验证了LLM辅助的抽取流程，其中表现最佳的模型与裁定标签达成与人类相当的一致性，Krippendorff's alpha系数为0.606，而人类之间的一致性为0.585。利用该流程，我们构建了Annotated-llm数据集，涵盖ACL会议论文（原文摘要在此处被截断）。

    arXiv:2606.02255v2 Announce Type: replace-cross  Abstract: Human annotation is the empirical foundation of much NLP research, from dataset construction to model evaluation, but papers often leave unclear who produced the annotations and how the annotation process was controlled. We provide the first large-scale, task-level audit of human annotation reporting across major NLP venues, asking which annotation details are documented, which are missing, and how reporting varies across time, topic, venue, and intended use of human judgment. We introduce a unified taxonomy of annotation-reporting practices and validate an LLM-assisted extraction pipeline against Annotated-gold, a human-adjudicated gold standard of 41 papers and 72 annotation tasks, where the best model reaches human-comparable agreement with adjudicated labels, with Krippendorff's alpha of 0.606 versus 0.585 for human-human agreement. Using this pipeline, we construct Annotated-llm, a dataset covering ACL-venue papers from 20
    
[^245]: PlanarBench：通过平面图绘制评估大语言模型的空间推理能力

    PlanarBench: Evaluating LLM Spatial Reasoning via Planar Graph Drawing

    [https://arxiv.org/abs/2606.02010](https://arxiv.org/abs/2606.02010)

    该论文提出PlanarBench基准，要求大语言模型仅根据边列表绘制平面图的无交叉ASCII图，并发现边数（即约束数量）比顶点数更能决定任务难度，为评估LLM空间推理能力提供了可控的测试环境。

    

    现有的大语言模型图基准通常要求模型回答图论问题或计算符号解，而非构建空间布局，且任务难度主要按顶点数量进行分层。然而，现有研究也表明，任务难度与边所施加的约束数量的关联比与所排列顶点数量的关联更为密切。我们提出了PlanarBench，这是一个要求模型仅根据边列表生成平面图的无交叉ASCII绘图的基准。在91个模型配置和199个具有2至7个顶点的非同构连通平面图上，边数与平均任务得分的关联比顶点数更强（r=-0.85 对比 r=-0.47），并且在控制顶点数之后仍保持强关联（rp=-0.80）。PlanarBench为分离这两个难度维度提供了受控环境。此外，摘要在此处被截断。

    arXiv:2606.02010v2 Announce Type: replace-cross  Abstract: Existing LLM graph benchmarks typically ask models to answer graph-theoretic questions or compute symbolic solutions rather than construct spatial layouts. Within-task difficulty is also primarily stratified by vertex count. However, existing research also suggests that task difficulty is more closely related to the number of constraints imposed by the edges than to the number of vertices being arranged. We introduce PlanarBench, a benchmark that asks models to produce crossing-free ASCII drawings of planar graphs given only an edge list. Across 91 model configurations and 199 non-isomorphic connected planar graphs with 2-7 vertices, edge count is more strongly associated with mean task score than vertex count ($r=-0.85$) versus ($r=-0.47$) and remains strongly associated after controlling for vertex count ($r_p=-0.80$). PlanarBench provides a controlled setting for separating these two difficulty axes. In addition, neither dra
    
[^246]: FineVerify：通过细粒度自我验证扩展智能体搜索的测试时计算

    FineVerify: Scaling Test-Time Compute with Fine-Grained Self-Verification for Agentic Search

    [https://arxiv.org/abs/2606.00660](https://arxiv.org/abs/2606.00660)

    提出FineVerify细粒度自我验证框架，将问题分解为可检查的子问题并逐项验证候选答案，从而在智能体搜索任务中有效扩展测试时计算并持续超越标准基线。

    

    智能体搜索要求语言模型智能体探索众多信息源并回答复杂的信息检索问题。扩展测试时计算是提升这类智能体性能的一种有前景的方法，但现有方法可能失效，因为正确答案往往很稀疏，且基于分数的选择依赖于模型的校准能力。我们提出FineVerify，这是一个细粒度自我验证框架，它将每个问题分解为可检查的子问题，针对每个子问题验证采样的候选答案，并选择聚合得分最高的候选答案。这种逐项检查的结构将选择过程转化为更简单的局部判断，并在相同的明确标准下生成分数。在四个智能体搜索基准和两个模型上，FineVerify始终优于标准的扩展基线。仅使用四条采样轨迹，它就使GPT-5-mini平均提升了8.2个准确率点，使Gemini-3-flash提升了5.6%。使用12个样本时，Fine

    arXiv:2606.00660v2 Announce Type: replace  Abstract: Agentic search requires language model agents to explore many sources and answer complex information-seeking questions. Scaling test-time compute is a promising way to improve these agents, but current approaches can fail, because correct answers are often sparse and score-based selection depends on model calibration. We propose FineVerify, a fine-grained self-verification framework that decomposes each question into checkable sub-questions, verifies sampled candidates against each sub-question, and selects the candidate with the highest aggregated score. This per-check structure turns selection into simpler local judgments and produces scores under the same explicit criteria. Across four agentic search benchmarks and two models, FineVerify consistently outperforms standard scaling baselines. With only four sampled trajectories, it improves GPT-5-mini by 8.2 accuracy points and Gemini-3-flash by 5.6% on average. With 12 samples, Fine
    
[^247]: MELD：基于梅尔频谱图与离散潜变量的语音语言建模

    MELD: Mel-Spectrogram-Based Speech Language Modeling with Discrete Latent Variables

    [https://arxiv.org/abs/2605.29859](https://arxiv.org/abs/2605.29859)

    该论文提出MELD，通过在梅尔频谱图上引入离散潜变量并对编码器与语音语言模型进行联合优化，在零样本TTS和STT任务上超越基线方法，同时有效缓解了静音过长和漏词等自回归建模问题。

    

    近期的语音语言模型依赖于与自回归模型分开优化的编码器。由于这些编码器不了解下游任务目标，其提取的表示可能并非下游任务的最优选择。为了解决这一局限性，我们在梅尔频谱图上引入了一种离散潜变量模型，对编码器和语音语言模型进行联合优化。联合优化不仅在零样本文本转语音（TTS）和语音转文本（STT）任务上带来了相比基于编解码器以及其他基于梅尔频谱图基线的性能提升，还有效缓解了自回归梅尔频谱图建模中的常见问题，例如生成过长静音和遗漏词语。

    arXiv:2605.29859v2 Announce Type: replace-cross  Abstract: Recent speech language models rely on encoders that are optimized separately from autoregressive models. Since these encoders are unaware of the downstream objectives, the extracted representations may not be optimal for downstream tasks. To address this limitation, we introduce a discrete latent variable model on mel spectrograms that jointly optimizes the encoder and the speech language model. Joint optimization not only brings improvements over codec-based and other mel-spectrogram-based baselines on zero-shot Text-to-Speech (TTS) and Speech-to-Text (STT) tasks, but also effectively alleviates common issues in autoregressive mel spectrogram modeling, such as prolonged silence generation and word omissions.
    
[^248]: PEARL：基于教学对齐强化学习训练苏格拉底式导师

    PEARL: Training Socratic Tutors with Pedagogically Aligned Reinforcement Learning

    [https://arxiv.org/abs/2605.29582](https://arxiv.org/abs/2605.29582)

    PEARL提出了一种教学对齐的强化学习框架，通过可控学生模拟器解耦认知状态并在多轮师生交互中协调多个教学目标，从而训练出擅长渐进式引导的苏格拉底式辅导智能体。

    

    arXiv:2605.29582v2 公告类型： replace-cross。摘要：大型语言模型（LLMs）在教育辅导领域展现出巨大潜力。现有方法通常训练它们去解题并给出正确答案，但这种以解题为中心的范式忽视了有效辅导的关键要求：渐进式引导以及在多轮交互中对多个教学目标的协调。开发这样的导师仍然充满挑战，因为学生的行为会随个体知识状态发生显著变化，教学效果取决于最终答案正确性之外的多个因素，且在师生交互过程中协调这些目标本身就十分困难。为应对这些挑战，我们提出了PEARL，一个用于训练苏格拉底式辅导智能体的教学对齐强化学习框架。首先，我们引入了一个可控的学生模拟器，将潜在认知状态与回复生成解耦，使得模拟……（摘要原文此处被截断）

    arXiv:2605.29582v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) show strong potential as educational tutors. Existing approaches typically train them to solve problems and provide correct answers, but this problem-solving-centered paradigm overlooks key requirements of effective tutoring: progressive guidance and the coordination of multiple pedagogical objectives across multi-turn interactions. Developing such tutors remains challenging because student behavior varies substantially with individual knowledge states, pedagogical effectiveness depends on multiple factors beyond final-answer correctness, and coordinating these objectives over tutor-student interactions is inherently difficult. To address these challenges, we propose PEARL, a PEdagogically Aligned Reinforcement Learning framework for training Socratic tutoring agents. First, we introduce a controllable student simulator that disentangles latent cognitive states from response generation, enabling sim
    
[^249]: MusTBench：音乐大语言模型中时间定位能力的基准测试与提升

    MusTBench: Benchmarking and Advancing Temporal Grounding in Music LLMs

    [https://arxiv.org/abs/2605.29300](https://arxiv.org/abs/2605.29300)

    该论文提出了经音乐专家验证的MusTBench基准和涵盖四阶段优化的MusT方案，用于评估并提升音乐大语言模型将回答准确锚定到音频正确时间段的能力。

    

    近期的大型音频-语言模型（LALMs）在理解音乐内容方面展现出了可观的能力。然而，这些模型的回答是否锚定于音频中正确的时间区域，这一问题仍未得到充分探索。这一局限性对于音乐理解尤为关键，因为音乐中的关键信息往往以时间上局部化事件的形式出现，例如乐器进入和节奏转换。为填补这一空白，我们提出了MusTBench，一个经音乐专家验证的基准，旨在通过五个时间定位问答任务来评估LALM的时间定位能力。为进一步提升现有模型的时间定位能力，我们提出了MusT，一种新颖的四阶段时间优化方案，涵盖音乐编码器适配、大语言模型适配、大语言模型监督微调以及基于强化学习的优化。在MusTBench上的实验表明，现有LALM难以实现精确的时间定位，而MusT则带来了……

    arXiv:2605.29300v2 Announce Type: replace-cross  Abstract: Recent Large Audio-Language Models (LALMs) have demonstrated promising abilities in understanding musical content. However, whether their responses are grounded in the correct temporal regions of the audio remains underexplored. This limitation is particularly critical for music understanding, where key information often occurs as temporally localized events, such as instrument entries and rhythmic transitions. To address this gap, we introduce MusTBench, a music-expert-validated benchmark designed to evaluate temporal grounding in LALMs through five temporally grounded question-answering tasks. To further improve temporal grounding in existing models, we propose MusT, a novel four-stage temporal optimization recipe spanning music encoder adaptation, LLM adaptation, LLM supervised fine-tuning, and RL-based optimization. Experiments on MusTBench show that existing LALMs struggle with precise temporal grounding, while MusT brings
    
[^250]: LLMBridge：一个用于英语端到端指代桥接消解的大语言模型流水线

    LLMBridge: An LLM Pipeline for End-to-end Referential Bridging Resolution in English

    [https://arxiv.org/abs/2605.29048](https://arxiv.org/abs/2605.29048)

    LLMBridge将启发式前后处理与大语言模型的自然语言推理能力相结合，在三个英语桥接消解数据集上同时实现了端到端和基本设置下指代桥接消解的最新最先进性能。

    

    在本文中，我们介绍了LLMBridge，一个用于英语端到端指代桥接消解任务的全新基于大语言模型的系统。我们的桥接消解流水线将启发式的前/后处理与大语言模型所具备的自然语言推理能力相结合。我们在三个曾用于英语指代桥接消解评估的数据集上评估了我们的桥接消解流水线：ISNotes、BASHI和GUMBridge。与以往桥接消解系统的比较表明，在具有挑战性的端到端评估设置以及基本桥接消解评估设置（给定黄金桥接先行词）中，LLMBridge的性能在全部3个数据集上都超越了先前的最先进系统。我们还对LLMBridge的性能进行了深入的错误分析，考察了哪些类型的桥接对于基于大语言模型的系统来说仍然难以识别。通过本文，我们发布了相关代码。

    arXiv:2605.29048v2 Announce Type: replace  Abstract: In this paper, we introduce LLMBridge, a new LLM based system for the task of end-to-end referential bridging resolution in English. Our bridging resolution pipeline combines heuristic pre/post-processing with the natural language inference ability that comes from LLMs. We evaluate our bridging resolution pipeline on three datasets which have been used for referential bridging resolution evaluation in English: ISNotes, BASHI, and GUMBridge. Comparison to previous bridging resolution systems shows that the performance of LLMBridge surpasses previous state-of-the-art (SoTA) systems for all 3 datasets in the challenging End-to-end Evaluation Setting, as well as the Basic Bridging Resolution Evaluation Setting (gold bridging anaphor given). We also conduct a thorough error analysis of the LLMBridge performance, examining what varieties of bridging remain difficult for LLM based systems to identify. With this paper, we release the code fo
    
[^251]: 大语言模型能否处理话语标记词？以口语马来语为例的案例研究

    Can Large Language Models Handle Discourse Particles? A Case Study of Colloquial Malay

    [https://arxiv.org/abs/2605.28782](https://arxiv.org/abs/2605.28782)

    本文提出MalayPrag基准和五个语用功能属性框架，系统评估十个现成大语言模型处理口语马来语话语标记词的能力，实验结果表明现有模型在此任务上面临显著挑战。

    

    话语标记词（如well和kind of）是使大语言模型能够更“像人类”一样说话的关键组成部分。它们被用来传达情感、意图和人际态度。然而，现有研究尚未对大语言模型处理话语标记词的能力建立全面的认识。此外，数量有限的相关研究主要集中在英语等高资源语言上，对东南亚语言鲜有关注。在本文中，我们（1）提出了MalayPrag，一个旨在系统评估和分析大语言模型处理口语马来语话语标记词能力的基准；（2）引入了五个属性，为解释话语标记词的语用功能提供了一个具有理论依据的统一框架。基于这两项贡献，我们对十个现成的大语言模型进行了提示，使其执行三项预测任务。实验结果揭示了模型面临的重大挑战。

    arXiv:2605.28782v2 Announce Type: replace  Abstract: Discourse particles, such as well and kind of, are crucial components that enable LLMs to "speak" more like humans. They are used to convey emotions, intentions, and interpersonal attitudes. However, existing studies have not yet built a comprehensive understanding of LLMs' capabilities in handling discourse particles. Moreover, the limited number of research focuses primarily on high-resource languages such as English, with little attention paid to Southeast Asian languages. In this paper, we (1) propose MalayPrag, a benchmark designed to systematically evaluate and analyze LLMs' capabilities in handling discourse particles in colloquial Malay; (2) introduce five attributes that provide a theoretically grounded, unified framework for interpreting pragmatic functions of discourse particles. Applying these two, we prompt ten off-the-shelf LLMs to perform three prediction tasks. The experimental results reveal substantial challenges fo
    
[^252]: 统计上“认真”的重要性：对GSM-Symbolic基准的批判性再评估

    The Importance of Being Statistically Earnest: A Critical Re-evaluation of GSM-Symbolic

    [https://arxiv.org/abs/2605.28700](https://arxiv.org/abs/2605.28700)

    该研究指出GSM-Symbolic基准的统计方法存在缺陷，重新评估发现20个开源模型中仅8个呈现统计显著的性能下降，且数据集中整数分布系统性偏大是重要混淆因素，从而质疑了“大语言模型缺乏真正推理能力”的结论。

    

    GSM-Symbolic基准测试（Mirzadeh等人，2025）报告称，25个大型语言模型（LLM）在基于模板生成的GSM8K问题变体上测试时，性能均出现一致性的下降，并据此得出这些模型缺乏真正推理能力的结论。我们认为这一结论建立在并不稳固的统计基础之上。我们使用带逐题随机效应的自举广义线性混合模型，对20个开源权重模型进行重新评估，发现其中仅有8个模型在原始提示格式下表现出统计显著的性能变化。此外，我们发现了一个此前未被认识到的影响因素：GSM-Symbolic主数据集中问题文本的整数分布相对于原始GSM8K系统性地偏向更大的数值（K-S统计量=0.12，p<0.001），这与原论文作者的说法相矛盾。在控制这一“大数值”效应后，剩余显著性案例中的一半可以得到解释。

    arXiv:2605.28700v3 Announce Type: replace  Abstract: The GSM-Symbolic benchmark (Mirzadeh et al., 2025) reported consistent performance drops across 25 Large Language Models (LLMs) when tested on template-generated variants of GSM8K problems, concluding that the models lack genuine reasoning capabilities. We argue that this conclusion rests on shaky statistical ground. Re-evaluating 20 open-weight models using bootstrapped Generalised Linear Mixed Models with per-question random effects, we find that only 8 exhibit statistically significant performance changes under the original prompt format. Moreover, we identify a previously unacknowledged factor: the distribution of integers in problem texts of the main GSM-Symbolic dataset is systematically shifted towards larger values relative to the original GSM8K (K-S statistic = 0.12, p < 0.001), contradicting the original authors' claims. Controlling for this large-number effect accounts for significance in half of the remaining cases. Among
    
[^253]: 基于大语言模型的论辩质量评估：一种成对比较的Bradley-Terry方法

    Argument Quality Assessment with Large Language Models: A Pairwise Bradley-Terry Approach

    [https://arxiv.org/abs/2605.28313](https://arxiv.org/abs/2605.28313)

    本研究在零样本、少样本和思维链设置下测试了12个开放权重大语言模型对论辩质量（逻辑、修辞、辩证三维度）进行成对比较评估的能力，发现LLM与人类判断仅存在中等程度的相关性，其中Llama-70B表现最佳（Cohen's κ = 0.493），并通过Bradley-Terry模型将比较结果转化为论辩潜在强度得分与排名。

    

    大语言模型（LLMs）在推理和判断相关任务中已展现出卓越的能力。然而，评估论辩质量需要严格的评价标准。我们研究了LLM在多大程度上能够有效执行这一任务。我们测试了12个不同规模和不同系列的开放权重LLM，在零样本、少样本和思维链设置下，模拟人类在三个维度——逻辑、修辞和辩证——上对论辩质量的成对比较，并将这些比较应用于Bradley-Terry模型，以推断潜在强度得分并得出论辩的排名。我们的研究结果表明，LLM与人类判断具有有前景但中等程度的相关性，其中Llama-70B获得了最强的一致性，达到中等水平的Cohen's κ = 0.493，以及与从这些标注中得出的Bradley-Terry得分的中等相关性（Kendall、Pearson和Spearman：0.327-0.477）。其他LLM则表现出……

    arXiv:2605.28313v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have demonstrated remarkable capabilities in tasks related to reasoning and judgment. However, assessing the quality of arguments requires a rigorous evaluation. We investigate the extent to which LLMs can effectively perform this task. We tested 12 open-weight LLMs of different sizes and families under zero-shot, few-shot, and chain-of-thought to approximate human pairwise comparisons of argument quality across three dimensions--logical, rhetorical, and dialectic--and used these comparisons in a Bradley-Terry model to infer latent strength scores and derive a ranking of arguments. Our insights show that LLMs have promising but moderate correlation with human judgment, with Llama-70B obtaining the strongest alignment, reaching moderate Cohen's $\kappa$ = 0.493 and moderate correlations with Bradley-Terry scores derived from these annotations (Kendall, Pearson, and Spearman: 0.327-0.477). Other LLMs exhibi
    
[^254]: 当最强的教师并非最好的教师：以学生为中心的答案选择

    When the Strongest Teacher Is Not the Best Teacher: Student-Centric Answer Selection

    [https://arxiv.org/abs/2605.26872](https://arxiv.org/abs/2605.26872)

    论文提出SCAS框架，证明最强教师的正确答案未必是学生的最佳训练监督，并通过逐token梯度分解推导出仅需前向计算的高效代理指标，依据学生中心学习成本来选择最适合学生的教师答案。

    

    大语言模型（LLM）训练越来越依赖教师生成的监督信号，包括合成回复、推理轨迹和工具使用演示。当前的做法通常选择表现最强的教师来生成学生的训练数据，这隐含地将教师的测试性能视为教学质量的替代指标。我们证明这一假设可能失效：即使多个教师对同一问题都给出了正确答案，最强教师给出的答案也不一定是对特定学生最好的监督信号。为了解决这一空白，我们提出了学生中心答案采样，这是一个根据估计的学生中心学习成本，从经过验证的教师生成答案中进行选择的框架。受逐token梯度分解的启发，我们推导出一种仅需前向计算的高效代理指标来估计该成本，并用它来指导训练过程中的答案选择。实验涵盖了30个教师模型和6个学生基础模型。

    arXiv:2605.26872v4 Announce Type: replace-cross  Abstract: LLM training increasingly relies on teacher-generated supervision, from synthetic responses to reasoning traces and tool-use demonstrations. Current practice often chooses the highest-performing teacher to generate student training data, implicitly treating teacher test performance as a proxy for teaching quality. We show that this assumption can fail: even when multiple teachers provide correct answers to the same question, the answer from the strongest teacher is not necessarily the best supervision for a given student. To address this gap, we propose Student-Centric Answer Sampling (SCAS), a framework that selects from verified teacher-generated answers according to their estimated student-centric learning cost. Motivated by a token-wise gradient decomposition, we derive an efficient forward-only proxy for this cost and use it to guide answer selection during training. Experiments across 30 teacher models, 6 student base mod
    
[^255]: 潜在循环Transformer：架构探索、训练策略与缩放行为

    Latent Recurrent Transformer: Architecture Exploration, Training Strategies, and Scaling Behavior

    [https://arxiv.org/abs/2605.26797](https://arxiv.org/abs/2605.26797)

    本文提出潜在循环Transformer（LRT），通过复用前一个token的高层隐藏状态作为循环记忆，在不改变标准注意力机制和KV-cache接口的前提下引入跨token、跨层的信息通路，并设计交错并行训练方法以约2倍理想计算成本实现循环记忆的预训练。

    

    我们研究了潜在循环Transformer（LRT），它是对自回归Transformer的一种轻量级增强，将前一个token的高层源层隐藏状态复用为下一个token的循环记忆。由于该状态在普通解码过程中已经计算得到，LRT在保留标准注意力机制、KV-cache接口以及每个生成token仅一次模型前向传播的前提下，引入了一条跨token、跨层的潜在通路。为了在不按顺序展开完整序列的情况下预训练这种循环机制，我们提出了交错并行训练：先用一次全序列初始化前向传播构建共享缓冲区，随后对各不相交的位置子集进行顺序细化，并在每个子集内部进行并行计算。这使得每个token都能获得感知循环记忆的监督信号，计算成本约为理想token计算量的2倍。在1.3B和2.1B参数的nanochat风格骨干模型以及广泛的训练条件下……（原文摘要在此处被截断）

    arXiv:2605.26797v2 Announce Type: replace-cross  Abstract: We study Latent Recurrent Transformer (LRT), a lightweight augmentation of autoregressive transformers that reuses a high-level source-layer hidden state from the previous token as recurrent memory for the next token. Because this state is already computed during ordinary decoding, LRT introduces a cross-token, cross-layer latent pathway while preserving the standard attention mechanism, KV-cache interface, and one model forward per generated token. To pretrain this recurrence without sequentially unrolling the full sequence, we introduce interleaved parallel training: one full-sequence initialization forward constructs a shared buffer, followed by sequential refinement of disjoint position subsets with parallel computation within each subset. This provides every token with recurrent-memory-aware supervision at approximately 2x ideal token compute. Across 1.3B- and 2.1B-parameter nanochat-style backbones and a wide range of tra
    
[^256]: 大型语言模型有多像人类？一个语域感知的语言学评估框架

    How Human-Like Are Large Language Models? A Register-Aware Linguistic Evaluation Framework

    [https://arxiv.org/abs/2605.23651](https://arxiv.org/abs/2605.23651)

    本文提出一个语域感知的评估框架，通过最大均值差异（MMD）比较人类参考语料库与LLM生成语料库在67个词汇-语法特征上的分布差异，从语言学层面量化评估大语言模型生成文本的“人类相似度”。

    

    长期以来，事实正确性和任务性能一直是大语言模型（LLM）研究的焦点，而生成文本在语言学层面上有多像人类这一根本性问题却鲜有探索。从语料库语言学的视角来看，语言产出本质上是依赖于语境的，不同的交际语境会导致语言特征在频率和共现模式上的差异。一篇不符合这些模式的文本可能在内容上是正确的，但对人类读者而言仍然是不理想的。在本工作中，我们提出了一个语境感知的评估框架，将“人类相似度”的评估建模为一个双样本问题，即比较给定语域下人类参考语料库与相应LLM生成语料库之间的语言特征分布。我们使用最大均值差异（MMD）以及67个词汇-语法特征来实现这一框架。

    arXiv:2605.23651v3 Announce Type: replace  Abstract: While factual correctness and task-performance have been in focus of Large Language Model (LLM) research for a long time, the fundamental question of how human-like generated texts are on a linguistic level has been underexplored. From a corpus-linguistic perspective, language production is inherently context-dependent, with distinct communicative contexts giving rise to differences in frequencies and co-occurrence patterns of linguistic features. A text failing to adhere to these patterns can be content-wise correct, but still be unfavorable to human readers. In this work, we propose a context-aware evaluation framework in which human-likeness is assessed using a two-sample problem between the linguistic feature distribution of a human reference corpus for a given register and a corresponding LLM-generated corpus. We implement this framework using the Maximum Mean Discrepancy (MMD) and the 67 lexico-grammatical features introduced b
    
[^257]: PromptNCE：仅使用大语言模型和对比估计提示的条件概率与逐点互信息估计

    PromptNCE: Conditional Probabilities and PMI Using Only LLMs and Contrastive Estimation Prompts

    [https://arxiv.org/abs/2605.21776](https://arxiv.org/abs/2605.21776)

    PromptNCE通过在对比估计提示中引入显式的OTHER类别突破闭集归一化的限制，使大语言模型能够零样本地估计条件概率和逐点互信息，并在三个基准数据集上取得最佳条件概率估计效果。

    

    从文本中估计互信息通常需要训练特定任务的评判模型，这限制了其在低数据场景中的应用。我们探讨大语言模型能否以零样本方式，仅使用提示和引导出的概率来估计逐点互信息（PMI）。我们基于三个公开可用且带有真实PMI标注的人工标注数据集构建了基准，并评估了五种基于提示的信息论估计方法。我们的主要方法PromptNCE将条件概率估计构建为对比任务，并通过显式的OTHER类别来扩充候选集。OTHER类别允许模型将概率质量分配到候选集之外，从而避免了标准对比提示中的闭集归一化问题。PromptNCE在所有三个数据集上都给出了最佳的条件概率估计。对于完整的PMI估计，我们发现在三个数据集中的两个上，估计标签基准率是主要瓶颈。

    arXiv:2605.21776v2 Announce Type: replace  Abstract: Estimating mutual information from text usually requires training a task-specific critic, which limits its use in low-data settings. We ask whether large language models can instead estimate pointwise mutual information zero-shot, using only prompts and elicited probabilities. We construct a benchmark from three publicly available human-annotated datasets with ground-truth PMI, and evaluate five information-theoretic prompting-based estimators. Our main method, PromptNCE, frames conditional probability estimation as a contrastive task and augments the candidate set with an explicit OTHER category. The OTHER category allows the model to assign probability mass outside the candidate set, avoiding the closed-set normalization of standard contrastive prompts. PromptNCE gives the best conditional probability estimates on all three datasets. For full PMI, we find that estimating label base rates is the primary bottleneck on two of the thre
    
[^258]: 照我说的做，而非照我做的做：大语言模型中的指令-归纳冲突

    Do as I Say, Not as I Do: Instruction-Induction Conflict in LLMs

    [https://arxiv.org/abs/2605.20382](https://arxiv.org/abs/2605.20382)

    该研究通过构造用户指令与硬编码对话模式相冲突的实验场景，发现大语言模型的指令遵循能力在不同模型间差异巨大（1%到99%）且与常规能力基准基本无关，其鲁棒性取决于指令内容与模型价值先验的一致性以及输出格式。

    

    语言模型被训练来遵循指令，但它们同时也是强大的模式补全器。当这两个目标发生冲突时会发生什么？我们构建了一些对话场景，其中用户指令要求模型以目标方式T行动（例如，始终输出特定的token、用某种特定语言回答或采用某个人设），而与之对抗的是N个硬编码的助手回合，它们展示了一种竞争性模式P。我们在这种设置下测量指令遵循（IF）率，涵盖13个模型和16种不同指令，测试轮数多达50轮。各模型的平均指令遵循率从1%到99%不等，且与标准能力基准基本不相关。从指令遵循到模式遵循的转变是普遍存在的，但高度依赖于具体模型。鲁棒性同时受到指令内容和输出格式的调节：当指令与模型训练中的价值先验一致时，模型能更长时间地抵抗归纳效应。

    arXiv:2605.20382v3 Announce Type: replace-cross  Abstract: Language models are trained to follow instructions, but they are also powerful pattern completers. What happens when these two objectives conflict? We construct conversations in which a user instruction to behave in a target way T (e.g., always output a specific token, answer in a particular language, or adopt a persona) is opposed by N hardcoded assistant turns demonstrating a competing pattern P. We then measure instruction-following (IF) rates in this setting, across 13 models and 16 different instructions, for up to 50 turns. Average instruction-following rates range from 1% to 99% across models, largely uncorrelated with standard capability benchmarks. The transition from instruction-following to pattern-following is universal but highly model-dependent. Robustness is modulated both by instruction content, with models resisting induction longer when instructions align with their trained value priors, and by output format, 
    
[^259]: 科学贡献图谱：大规模基于文献的自动化技术路线图

    The Scientific Contribution Graph: Automated Literature-based Technological Roadmapping at Scale

    [https://arxiv.org/abs/2605.15011](https://arxiv.org/abs/2605.15011)

    本文构建了包含600万条科学贡献和3600万条先决条件关系的大规模“科学贡献图谱”，并提出了科学先决条件预测任务，使模型能够预测哪些现有技术可促成未来发现，从而实现大规模自动化的技术路线图制定。

    

    科学贡献很少是孤立发展的，而是建立在先前的发现之上。我们将自动化技术路线图制定这一任务形式化为：从学术文章中提取科学贡献，并将其与各自的先决条件相链接。我们提出了科学贡献图谱，这是一个大规模资源，包含从65.5万篇开放获取论文中提取的600万条详细科学贡献，涵盖计算机科学、医学、生物学、物理学、化学及其他科学领域，并通过3600万条先决条件边相互连接。我们进一步引入了“科学先决条件预测”这一科学发现任务，即模型预测哪些现有技术能够促成未来的发现，并展示了当代模型在该任务上的快速进步——在采用时间过滤回测进行评估时达到了0.48的平均精度（MAP）。我们预期此类技术路线图资源将支持科学（原文摘要到此截断）。

    arXiv:2605.15011v3 Announce Type: replace  Abstract: Scientific contributions rarely develop in isolation, but instead build upon prior discoveries. We formulate the task of automated technological roadmapping as extracting scientific contributions from scholarly articles and linking them to their prerequisites. We present the Scientific Contribution Graph, a large-scale resource containing 6 million detailed scientific contributions extracted from 655k open-access papers spanning computer science, medicine, biology, physics, chemistry, and other sciences, and connected by 36 million prerequisite edges. We further introduce scientific prerequisite prediction, a scientific discovery task in which models predict which existing technologies can enable future discoveries, and show that contemporary models are rapidly improving on this task, reaching 0.48 MAP when evaluated using temporally-filtered backtesting. We anticipate technological roadmapping resources such as this will support sci
    
[^260]: 泄漏审计基准揭示跨受试者听觉诱发脑电元音感知解码证据有限

    Leakage-Audited Benchmarking Reveals Limited Evidence for Cross-Subject Auditory-Evoked EEG Vowel Perception Decoding

    [https://arxiv.org/abs/2605.00865](https://arxiv.org/abs/2605.00865)

    该研究通过严格的泄漏审计基准，发现跨受试者听觉脑电元音解码的证据非常有限，即使最佳模型也仅略高于随机水平且不显著。

    

    我们测试了在单一基准中控制试验身份、模型身份、预测来源和参与者水平推断时，听觉诱发脑电是否支持受试者无关的五元音感知解码。我们从OpenNeuro ds006104版本1.0.1重建了研究2的事件表，并分析了辅音-元音对任务。一对一标记-刺激配对产生了3,840个独立试验；对照条件选择和伪迹拒绝保留了来自16名参与者和61个脑电通道的1,094个时段。使用留一受试者测试评估了13种独特实现，参与者指标从33个完整预测副本中的36,102个试验预测中重建。随机森林在数值上最高，平衡准确率为21.474%（95%参与者自助区间，19.526-23.482%；随机水平为20%），但其参与者水平测试或任何实现均未通过校正。

    arXiv:2605.00865v3 Announce Type: replace-cross  Abstract: We tested whether auditory-evoked EEG supports subject-independent five-vowel perception decoding when trial identity, model identity, prediction provenance, and participant-level inference are controlled within a single benchmark. We reconstructed Study 2 event tables from OpenNeuro ds006104 version 1.0.1 and analyzed the consonant-vowel pair task. One-to-one marker-stimulus pairing yielded 3,840 independent trials; control-condition selection and artifact rejection retained 1,094 epochs from 16 participants and 61 EEG channels. Thirteen unique implementations were evaluated using leave-one-subject-out testing, with participant metrics reconstructed from 36,102 trial predictions across 33 complete prediction replicas. Random Forest was numerically highest at 21.474% balanced accuracy (95% participant-bootstrap interval, 19.526-23.482%; chance, 20%), but neither its participant-level tests nor any implementation survived correc
    
[^261]: 为什么微调会诱发幻觉以及如何修复它

    Why Fine-Tuning Encourages Hallucinations and How to Fix It

    [https://arxiv.org/abs/2604.15574](https://arxiv.org/abs/2604.15574)

    该论文提出一种基于自蒸馏的监督微调方法，通过正则化输出分布漂移，使模型在学习新事实的同时最大限度减少对预训练知识的幻觉，并证明在无需学习新知识时冻结参数组也能在保持任务性能的前提下降低幻觉。

    

    大语言模型容易产生与事实不符的幻觉陈述。这些错误的一个关键来源是监督微调（SFT）过程中接触到新的知识，这会增加相对于预训练期间所获知识的幻觉。由于这些错误是知识退化的副产品，我们探索能否利用已有的持续学习工具来缓解这一问题。我们提出了一种基于自蒸馏的SFT方法，通过正则化输出分布的漂移，在实现有效事实学习的同时，最大限度减少相对于已有知识的幻觉。我们还表明，当不需要获取新知识时，通过冻结参数组来抑制事实可塑性，可以在减少幻觉的同时保持任务性能。最后，我们研究了其内在机制，对比了容量限制、行为克隆和局部干扰等假说。我们的实验表明，主要的……

    arXiv:2604.15574v2 Announce Type: replace-cross  Abstract: Large language models are prone to hallucinating factually incorrect statements. A key source of these errors is exposure to new factual information through supervised fine-tuning (SFT), which can increase hallucinations w.r.t.~knowledge acquired during pre-training. Since these errors arise as a by-product of knowledge degradation, we explore whether established continual learning tools can mitigate them. We propose a self-distillation-based SFT method that facilitates effective factual learning while minimizing hallucinations w.r.t.~pre-existing knowledge by regularizing output-distribution drift. We also show that when new knowledge acquisition is unnecessary, suppressing factual plasticity by freezing parameter groups preserves task performance while reducing hallucinations. Lastly, we investigate the mechanism, contrasting capacity limitations, behavior cloning, and localized interference. Our experiments show that a main 
    
[^262]: DiscoTrace：表示与比较人类和大型语言模型在信息寻求式问答中的回答策略

    DiscoTrace: Representing and Comparing Answering Strategies of Humans and LLMs in Information-Seeking Question Answering

    [https://arxiv.org/abs/2604.15140](https://arxiv.org/abs/2604.15140)

    DiscoTrace通过将答案表示为问题相关言语行为的序列，揭示了不同人类社区在回答策略上存在丰富多样的修辞偏好，而LLM即使被提示模仿也缺乏这种多样性且系统性偏向宽泛回答，这一发现可指导开发更贴合语境信息需求的LLM回答者。

    

    我们提出了DiscoTrace，这是一种用于识别回答者在回应信息寻求类问题时所采用修辞策略的方法。DiscoTrace将答案表示为一系列与问题相关的言语行为，并与对原始问题的不同解读相配对，标注在修辞结构理论解析之上。将DiscoTrace应用于来自九个不同社区的答案后发现，各社区在答案构建方式上存在多样化的偏好。相比之下，大型语言模型（LLM）在其答案中并不表现出修辞多样性，即使在被提示模仿特定人类社区的回答指南时也是如此。此外，LLM还系统性地倾向于追求广度，会去回应那些人类回答者选择不回应的问题解读。DiscoTrace在结构层面上揭示的这种丰富的、对社区敏感的回答行为，可以指导开发更具语用能力、更贴合语境信息需求的LLM回答者。

    arXiv:2604.15140v2 Announce Type: replace  Abstract: We introduce DiscoTrace, a method to identify the rhetorical strategies answerers use when responding to information-seeking questions. DiscoTrace represents answers as a sequence of question-related discourse acts paired with interpretations of the original question, annotated on top of rhetorical structure theory parses. Applying DiscoTrace to answers from nine different communities reveals that communities have diverse preferences for answer construction. In contrast, LLMs do not exhibit rhetorical diversity in their answers, even when prompted to mimic specific human community answering guidelines. LLMs also systematically opt for breadth, addressing interpretations of questions that human answerers choose not to address. The rich, community-sensitive answering behavior structurally revealed by DiscoTrace can guide the development of pragmatic LLM answerers that are more attuned to contextual information needs.
    
[^263]: 是什么驱动了表征转向？关于转向拒绝行为的机制案例研究

    What Drives Representation Steering? A Mechanistic Case Study on Steering Refusal

    [https://arxiv.org/abs/2604.08524](https://arxiv.org/abs/2604.08524)

    本研究通过多词元激活修补框架对LLM拒绝行为的转向机制进行案例研究，发现不同转向方法在同一层利用功能可互换的回路，且转向向量主要通过注意力机制的OV回路发挥作用而几乎不依赖QK回路。

    

    将转向向量应用于大型语言模型（LLM）是一种高效且有效的模型对齐技术，但我们对其工作原理缺乏可解释的解释——具体来说，转向向量影响了哪些内部机制，以及这如何导致不同的模型输出。为了探究转向向量有效性的因果机制，我们对拒绝行为进行了全面的案例研究。我们提出了一个多词元激活修补框架，并发现不同的转向方法在同一层应用时利用的是功能上可互换的回路。这些回路揭示出，转向向量主要通过OV回路与注意力机制交互，而在很大程度上忽略了QK回路。在转向过程中冻结所有注意力分数，在三个模型家族上仅导致8.83%的性能下降。对被转向OV回路的数学分解进一步揭示了……

    arXiv:2604.08524v2 Announce Type: replace-cross  Abstract: Applying steering vectors to large language models (LLMs) is an efficient and effective model alignment technique, but we lack an interpretable explanation for how it works--specifically, what internal mechanisms steering vectors affect and how this results in different model outputs. To investigate the causal mechanisms underlying the effectiveness of steering vectors, we conduct a comprehensive case study on refusal. We propose a multi-token activation patching framework and discover that different steering methodologies leverage functionally interchangeable circuits when applied at the same layer. These circuits reveal that steering vectors primarily interact with the attention mechanism through the OV circuit while largely ignoring the QK circuit. Freezing all attention scores during steering drops performance by only 8.83% across three model families. A mathematical decomposition of the steered OV circuit further reveals s
    
[^264]: 面向上下文密集型任务的KV缓存卸载

    KV Cache Offloading for Context-Intensive Tasks

    [https://arxiv.org/abs/2604.08426](https://arxiv.org/abs/2604.08426)

    该论文创建并发布了Text2JSON基准测试，揭示现代KV缓存卸载技术在需要从输入提示中提取大量信息的上下文密集型任务上，会导致Llama 3和Qwen 3模型出现显著的性能下降。

    

    随着各类应用对长上下文大语言模型（LLM）需求的不断增长，键值（KV）缓存已成为延迟和内存占用的关键瓶颈。近来，KV缓存卸载已成为一种有前景的方法，可在保持精度的同时减少内存占用和推理延迟。以往的评估工作主要集中于不需要从上下文中提取大量信息的任务。在本工作中，我们研究了KV缓存卸载在上下文密集型任务上的表现：即那些求解过程需要从输入提示中查找大量信息的问题。我们创建并发布了Text2JSON基准测试，这是一个高度上下文密集型的任务，需要从原始文本中提取结构化知识。我们在Text2JSON以及其他上下文密集型任务上对现代KV卸载技术进行了评估，发现Llama 3和Qwen 3模型均出现了显著的性能下降。我们的分析识别出两个关键原因（摘要原文在此处被截断）。

    arXiv:2604.08426v5 Announce Type: replace-cross  Abstract: With the growing demand for long-context LLMs across a wide range of applications, the key-value (KV) cache has become a critical bottleneck for both latency and memory usage. Recently, KV-cache offloading has emerged as a promising approach to reduce memory footprint and inference latency while preserving accuracy. Prior evaluations have largely focused on tasks that do not require extracting large amounts of information from the context. In this work, we study KV-cache offloading on context-intensive tasks: problems where the solution requires looking up a lot of information from the input prompt. We create and release the Text2JSON benchmark, a highly context-intensive task that requires extracting structured knowledge from raw text. We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models. Our analysis identifies two key re
    
[^265]: Oblivion：通过衰减驱动激活实现的自适应智能体记忆控制

    Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation

    [https://arxiv.org/abs/2604.00131](https://arxiv.org/abs/2604.00131)

    Oblivion框架借鉴人类选择性遗忘机制，将遗忘建模为衰减驱动的可及性降低而非删除，并通过解耦读取路径（基于不确定性决定何时查询记忆）与写入路径（强化贡献性记忆），为LLM智能体实现按需动态加载的层次化记忆组织。

    

    人类记忆通过选择性遗忘来实现适应：经验会随时间推移变得不易获取，但可以通过强化或情境线索被重新激活。相比之下，记忆增强的LLM智能体依赖“始终开启”的检索和“扁平”的记忆存储，随着历史记录的增长会导致高干扰和高延迟。我们提出了Oblivion，一个将遗忘视为由衰减驱动的可及性降低——而非显式删除——的记忆控制框架。Oblivion将记忆控制解耦为读取和写入两条路径。读取路径基于智能体的不确定性和记忆缓冲区的效用，决定何时查询记忆，从而避免冗余的始终开启式访问。写入路径通过强化对生成响应有贡献的记忆，决定应该加强哪些内容。两者结合，实现了层次化的记忆组织，在保持持久性高级策略的同时按需动态加载细节。我们在静态和动态

    arXiv:2604.00131v3 Announce Type: replace-cross  Abstract: Human memory adapts through selective forgetting: experiences become less accessible over time but can be reactivated by reinforcement or contextual cues. In contrast, memory-augmented LLM agents rely on "always-on" retrieval and "flat" memory storage, causing high interference and latency as histories grow. We introduce Oblivion, a memory control framework that casts forgetting as decay-driven reductions in accessibility -- not explicit deletion. Oblivion decouples memory control into read and write paths. The read path decides when to consult memory, based on agent uncertainty and memory buffer utility, avoiding redundant always-on access. The write path decides what to strengthen, by reinforcing memories contributing to forming the response. Together, this enables hierarchical memory organization that maintains persistent high-level strategies while dynamically loading details as needed. We evaluate on both static and dynami
    
[^266]: APEX-EM：基于结构化程序性-情景经验回放的自主智能体非参数化在线学习

    APEX-EM: Non-Parametric Online Learning for Autonomous Agents via Structured Procedural-Episodic Experience Replay

    [https://arxiv.org/abs/2603.29093](https://arxiv.org/abs/2603.29093)

    APEX-EM提出了一种无需更新模型权重的非参数化经验记忆方法，通过程序性知识图谱存储完整的任务轨迹并同时索引成功与失败经验，使LLM智能体能够复用过往经验而无需重复推理，在相同底层模型对比下BigCodeBench迁移任务上提升7.6个百分点。

    

    LLM智能体在执行每个任务时都需要重新运行完整的推理过程，即使是它们刚刚解决过的任务也不例外。我们提出了APEX-EM，这是一种非参数化经验记忆，它将完整的程序性-情景轨迹存储在类型化的程序性知识图谱（PKG）中，并通过三种通道进行检索：语义搜索、针对抽象操作序列的结构签名匹配以及图遍历。一个“规划-检索-生成-迭代-摄取”（PRGII）工作流负责生成经验、进行质量把关并提交经验，同时对成功和失败的经验都进行索引，使智能体学会哪些内容可以复用、哪些应当避免。在部署期间不改变任何模型权重。我们在五个基准上进行评估：BigCodeBench、KGQAGen-10k、HLE、Lifelong Agent Bench和ALFWorld。由于先前的工作使用了不同的底层模型，我们的结论基于相同底层模型的对比，以保持模型能力固定。在使用共享GPT-4o底层模型的BigCodeBench留出集迁移测试中，APEX-EM获得了+7.6个百分点的提升。

    arXiv:2603.29093v3 Announce Type: replace-cross  Abstract: LLM agents rerun full reasoning for every task, even one they solved moments earlier. We introduce \textbf{APEX-EM}, a non-parametric experience memory that stores complete procedural-episodic traces in a typed Procedural Knowledge Graph (PKG) and retrieves them through three channels: semantic search, structural-signature matching over abstract operation sequences, and graph traversal. A Plan-Retrieve-Generate-Iterate-Ingest (PRGII) workflow produces, quality-gates, and commits experiences, indexing both successes and failures so the agent learns what to reuse and what to avoid. No weights change during deployment.   We evaluate on five benchmarks: BigCodeBench, KGQAGen-10k, HLE, Lifelong Agent Bench, and ALFWorld. Because prior work uses different backbones, we base our claims on same-backbone comparisons that hold model capability fixed. On held-out BigCodeBench transfer with a shared GPT-4o backbone, APEX-EM gains +7.6\,pp 
    
[^267]: 过度拒答与表征子空间：对齐大语言模型中任务条件化拒答的机制分析

    Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs

    [https://arxiv.org/abs/2603.27518](https://arxiv.org/abs/2603.27518)

    该论文通过机制分析揭示了有害拒答方向是任务无关、可由单一全局向量捕捉的，而过度拒答方向是任务相关、嵌入良性任务表征聚类并跨越更高维子空间的，从而解释了为何消融全局拒答方向无法可靠修复过度拒答且会破坏正常拒答机制。

    

    经过训练以拒绝有害请求的对齐语言模型也会表现出过度拒答：它们会拒绝那些看似与有害指令相似的安全指令。一种自然的修复方法是消融全局拒答方向，将隐状态向量引导远离或朝向有害拒答示例，但这种方法只能顺带地纠正过度拒答，同时会破坏更广泛的拒答机制。在这项工作中，我们分析了这两种拒答类型的表征几何结构，以理解为什么会发生这种情况。我们表明，有害拒答方向是任务无关的，可以用单个全局向量来捕捉；而过度拒答方向是任务相关的：它们位于良性任务表征聚类之中，随任务而变化，并跨越一个更高维的子空间。线性探测表明，这两种拒答类型从Transformer的早期层开始就在表征上存在明显区别。这些发现为这一现象提供了机制层面的解释（摘要原文在此处被截断）。

    arXiv:2603.27518v4 Announce Type: replace  Abstract: Aligned language models that are trained to refuse harmful requests also exhibit over-refusal: they decline safe instructions that seemingly resemble harmful instructions. A natural approach is to ablate the global refusal direction, steering the hidden-state vectors away or towards the harmful-refusal examples, but this corrects over-refusal only incidentally while disrupting the broader refusal mechanism. In this work, we analyse the representational geometry of both refusal types to understand why this happens. We show that harmful-refusal directions are task-agnostic and can be captured by a single global vector, whereas over-refusal directions are task-dependent: they reside within the benign task-representation clusters, vary across tasks, and span a higher-dimensional subspace. Linear probing suggests that the two refusal types are representationally distinct from the early transformer layers. These findings provide a mechanis
    
[^268]: SafeMath：推理时安全机制提升数学准确性

    SafeMath: Inference-time Safety improves Math Accuracy

    [https://arxiv.org/abs/2603.25201](https://arxiv.org/abs/2603.25201)

    本文揭示了数学应用题可被用作传播有害内容的隐蔽媒介，提出了包含1.9千道题的ToxicGSM数据集用于审计LLM，并证明SafeMath推理时安全对齐技术能够在保障安全的同时提升数学准确性。

    

    近期研究表明，大语言模型（LLM）可能通过对抗性输入以及看似良性的输入被操纵，从而产生有害、有偏见或违反政策的内容。在本文中，我们研究了一个尚未被充分探索的问题——有害和有毒的数学应用题。我们证明，数学问题，尤其是以自然语言叙述形式呈现的题目，可以成为传播偏见、不道德或心理有害内容的隐蔽媒介，在涉及儿童的教育环境中风险尤为突出。为了支持对这一现象的系统性研究，我们提出了ToxicGSM数据集，该数据集包含1.9千道算术题，其中嵌入了有害或敏感的背景内容，同时保持了数学上定义明确的推理任务。利用该数据集，我们审计了现有LLM的行为，并分析了安全约束与数学正确性之间的权衡。我们进一步提出了SafeMath——一种安全对齐技术（摘要在此处截断）

    arXiv:2603.25201v2 Announce Type: replace  Abstract: Recent research points toward LLMs being manipulated through adversarial and seemingly benign inputs, resulting in harmful, biased, or policy-violating outputs. In this paper, we study an underexplored issue concerning harmful and toxic mathematical word problems. We show that math questions, particularly those framed as natural language narratives, can serve as a subtle medium for propagating biased, unethical, or psychologically harmful content, with heightened risks in educational settings involving children. To support a systematic study of this phenomenon, we introduce ToxicGSM, a dataset of 1.9k arithmetic problems in which harmful or sensitive context is embedded while preserving mathematically well-defined reasoning tasks. Using this dataset, we audit the behaviour of existing LLMs and analyse the trade-offs between safety enforcement and mathematical correctness. We further propose SafeMath -- a safety alignment technique th
    
[^269]: CWoMP：面向逐行注释的语素表示学习

    CWoMP: Morpheme Representation Learning for Interlinear Glossing

    [https://arxiv.org/abs/2603.18184](https://arxiv.org/abs/2603.18184)

    提出CWoMP方法，通过对比学习将语素作为形式-意义原子单元进行表示学习，并借助可扩展的词库检索机制生成逐行注释，在低资源语言上以更高效率超越了现有方法。

    

    逐行注释文本（Interlinear Glossed Text, IGT）是语言文档化的标准记号方式，语言学信息丰富但人工制作十分费力。近期的自动化IGT方法将注释视为字符序列，忽略了其组合结构。我们提出CWoMP（对比式词-语素预训练），该方法将语素视为具有学习表示的原子形式-意义单元。经过对比训练的编码器在共享嵌入空间中将上下文中的词与其构成语素对齐；随后自回归解码器通过从这些嵌入所构成的可变词库中检索条目来生成语素序列。预测结果是可解释的——植根于词库条目——用户还可以在推理时通过扩展词库来改进结果，而无需重新训练。我们在多种低资源语言上进行评估，结果表明CWoMP在性能上超越现有方法，同时显著更加高效。

    arXiv:2603.18184v2 Announce Type: replace  Abstract: Interlinear glossed text (IGT) is a standard notation for language documentation which is linguistically rich but laborious to produce manually. Recent automated IGT methods treat glosses as character sequences, neglecting their compositional structure. We propose CWoMP (Contrastive Word-Morpheme Pretraining), which instead treats morphemes as atomic form-meaning units with learned representations. A contrastively trained encoder aligns words-in-context with their constituent morphemes in a shared embedding space; an autoregressive decoder then generates the morpheme sequence by retrieving entries from a mutable lexicon of these embeddings. Predictions are interpretable--grounded in lexicon entries--and users can improve results at inference time by expanding the lexicon without retraining. We evaluate on diverse low-resource languages, showing that CWoMP outperforms existing methods while being significantly more efficient, with par
    
[^270]: MineDraft：一种批量并行投机解码框架

    MineDraft: A Framework for Batch Parallel Speculative Decoding

    [https://arxiv.org/abs/2603.18016](https://arxiv.org/abs/2603.18016)

    MineDraft提出一种批量并行投机解码框架，通过同时维护两批请求，将一批的草稿生成与另一批的验证重叠执行，有效隐藏草稿延迟，相比标准投机解码吞吐量最高提升75%、端到端延迟最高降低39%。

    

    投机解码（SD）通过使用较小的草稿模型提出草稿token，再由较大的目标模型进行验证，从而加速大语言模型的推理。然而，标准SD的性能往往受限于草稿生成与验证阶段的严格顺序执行。为解决这一问题，本文提出了MineDraft，一种批量并行投机解码（PSD）框架，旨在通过与验证过程重叠来有效隐藏草稿生成延迟。我们的理论分析表明，PSD比标准SD的效率显著更高。MineDraft通过一种新颖的批量并行设计实现了PSD：该设计同时维护两批请求，将一批请求的草稿生成与另一批请求的验证重叠进行。实验结果显示，与标准SD相比，MineDraft在吞吐量（最高提升75%）和端到端延迟（最高降低39%）方面均有显著改进。此外，我们还实现了……（摘要原文截断）

    arXiv:2603.18016v3 Announce Type: replace-cross  Abstract: Speculative decoding (SD) accelerates large language model inference by using a smaller draft model to propose draft tokens that are subsequently verified by a larger target model. However, the performance of standard SD is often limited by the strictly sequential execution of these drafting and verification stages. To address this, this paper proposes MineDraft, a batch parallel speculative decoding (PSD) framework designed to effectively hide drafting latency by overlapping it with verification. Our theoretical analysis shows that PSD is substantially more efficient than standard SD. MineDraft realizes the PSD through a novel batch-parallel design that maintains two batches of requests, overlapping drafting for one batch with verification for the other. Our experimental results show significant improvements of \alg{} in both throughput (up to 75%) and end-to-end latency (up to 39%) over standard SD. Furthermore, we have imple
    
[^271]: SpokenUS：面向任务型对话的口语用户模拟器

    SpokenUS: A Spoken User Simulator for Task-Oriented Dialogue

    [https://arxiv.org/abs/2603.16783](https://arxiv.org/abs/2603.16783)

    本文提出了包含52,390段对话、1,034小时语音并涵盖四种口语用户行为的口语任务型对话数据集SpokenTOD，以及基于该数据集、通过专用话轮转换头决定何时发言的口语用户模拟器SpokenUS，其以较小规模实现了与更大模型相当的目标覆盖率并大幅超越所有基线。

    

    稳健的语音智能体需要接触到人们通过语音进行交互的全部多样性。然而，获取足够多的口语交互数据成本极其高昂。构建能够解决这一问题的口语用户模拟器，需要大规模的涵盖口语用户行为的口语任务型对话（TOD）数据，但现有数据集在规模和领域覆盖上都较为有限，且缺乏系统性的数据增强流程。为解决这一问题，我们推出了SpokenTOD，这是一个包含52,390段对话、1,034小时语音的口语TOD数据集，并在多样化的说话人和领域中针对四种口语用户行为——跨轮槽位、插话、不流畅表达和情感韵律——进行了增强。基于SpokenTOD，我们提出了SpokenUS，一个以任务型对话为基础的口语用户模拟器，它通过专用的话轮转换头来决定何时发言。SpokenUS在目标覆盖率上可与规模大得多的模型相媲美，同时大幅超越所有基线模型。

    arXiv:2603.16783v2 Announce Type: replace  Abstract: Robust voice agents require exposure to the full diversity of how people interact through speech. However, obtaining enough spoken interactions is prohibitively expensive. Building spoken user simulators that address this requires large-scale spoken task-oriented dialogue (TOD) data encompassing spoken user behaviors, yet existing datasets are limited in scale and domain coverage, with no systematic pipeline for augmenting them. To address this, we introduce SpokenTOD, a spoken TOD dataset of 52,390 dialogues and 1,034 hours of speech augmented with four spoken user behaviors---cross-turn slots, barge-in, disfluency, and emotional prosody---across diverse speakers and domains. Building on SpokenTOD, we present SpokenUS, a spoken user simulator grounded in TOD that decides when to speak through a dedicated turn-taking head. SpokenUS achieves comparable goal coverage to much larger models while substantially outperforming all baselines
    
[^272]: 人工标注是否必要？面向机器翻译错误片段检测的迭代MBR蒸馏

    Is Human Annotation Necessary? Iterative MBR Distillation for Error Span Detection in Machine Translation

    [https://arxiv.org/abs/2603.12983](https://arxiv.org/abs/2603.12983)

    该论文提出了一种基于最小贝叶斯风险解码的迭代MBR蒸馏自演化框架，通过利用现成大语言模型生成伪标签进行自我训练，无需人工标注即可在机器翻译错误片段检测任务上超越基于人工标注的监督基线模型。

    

    错误片段检测（ESD）是机器翻译（MT）评估中的一个关键子任务，旨在识别翻译错误的位置和严重程度。虽然基于人工标注数据对模型进行微调可以提升ESD性能，但获取这类数据成本高昂，且容易在标注者之间产生不一致。为解决这一问题，我们提出了一种基于最小贝叶斯风险（MBR）解码的新型自演化框架，命名为面向ESD的迭代MBR蒸馏，该方法利用现成的大语言模型生成伪标签，从而摆脱对人工标注的依赖。在WMT指标共享任务数据集上的大量实验表明，仅使用这些自生成伪标签训练的模型在系统级和片段级上均优于未适配的基础模型以及基于人工标注训练的监督基线模型，同时保持了具有竞争力的句子级性能。

    arXiv:2603.12983v4 Announce Type: replace-cross  Abstract: Error Span Detection (ESD) is a crucial subtask in Machine Translation (MT) evaluation, aiming to identify the location and severity of translation errors. While fine-tuning models on human-annotated data improves ESD performance, acquiring such data is expensive and prone to inconsistencies among annotators. To address this, we propose a novel self-evolution framework based on Minimum Bayes Risk (MBR) decoding, named Iterative MBR Distillation for ESD, which eliminates the reliance on human annotations by leveraging an off-the-shelf LLM to generate pseudo-labels. Extensive experiments on the WMT Metrics Shared Task datasets demonstrate that models trained solely on these self-generated pseudo-labels outperform both unadapted base model and supervised baselines trained on human annotations at the system and span levels, while maintaining competitive sentence-level performance.
    
[^273]: CoMMET：一个基于心理学基础、用于评估多模态大语言模型心智理论的基准测试

    CoMMET: A Psychologically Grounded Benchmark for Evaluating Theory of Mind in Multimodal LLMs

    [https://arxiv.org/abs/2603.11915](https://arxiv.org/abs/2603.11915)

    该论文提出了首个基于心理学基础的多模态基准CoMMET，通过涵盖更广泛的心理状态、开放式问答和多轮对话测试，全面评估多模态大语言模型的心智理论能力。

    

    心智理论——即对自身和他人心理状态进行推理的能力——是人类社会智能的基石。随着多模态大语言模型在现实应用中变得无处不在，验证模型在这一社会推理层面的能力，对于实现有效和自然的交互至关重要。然而，现有的用于评估多模态大语言模型心智理论的基准测试十分有限：大多数仅依赖文本输入，且狭窄地聚焦于与信念相关的任务。在本文中，我们提出了一个新的多模态基准数据集CoMMET，这是一个受心智理论手册任务启发的综合性心理状态与道德评估任务。CoMMET通过涵盖更广泛的心理状态范围并引入多轮测试，扩展了评估的范畴。据我们所知，这是首个基于心理学基础、以多模态、开放式和多轮方式评估多模态大语言模型在多种心理状态上表现的基准测试。

    arXiv:2603.11915v2 Announce Type: replace  Abstract: Theory of Mind (ToM)-the ability to reason about the mental states of oneself and others-is a cornerstone of human social intelligence. As Multimodal Large Language Models (MLLMs) become ubiquitous in real-world applications, validating their capacity for this level of social reasoning is essential for effective and natural interactions. However, existing benchmarks for assessing ToM in MLLMs are limited; most rely solely on text inputs and focus narrowly on belief-related tasks. In this paper, we propose a new multimodal benchmark dataset, CoMMET, a comprehensive mental states and moral evaluation task inspired by the Theory of Mind Booklet Task. CoMMET expands the scope of evaluation by covering a broader range of mental states and introducing multi-turn testing. To the best of our knowledge, this is the first psychology-grounded benchmark to evaluate MLLMs across multiple mental states in a multimodal, open-ended, and multi-turn s
    
[^274]: 面向科学的MMAI Gym：训练用于药物发现的液体基础模型

    MMAI Gym for Science: Training Liquid Foundation Models for Drug Discovery

    [https://arxiv.org/abs/2603.03517](https://arxiv.org/abs/2603.03517)

    本文提出MMAI Gym for Science一站式训练框架，通过教会基础模型“分子的语言”，训练出更小规模的液体基础模型（LFM），在分子优化、ADMET预测等药物发现任务上超越了规模大得多的通用或专业模型。

    

    摘要：依赖上下文学习的通用大型语言模型（LLM）无法可靠地提供药物发现任务所需的科学理解和性能。仅仅增加模型规模或引入推理标记并不能带来显著的性能提升。为了解决这一差距，我们推出了面向科学的MMAI Gym（MMAI Gym for Science），这是一个一站式平台，提供分子数据格式与模态，以及面向特定任务的推理、训练和基准测试方案，旨在教会基础模型“分子的语言”，从而解决实际的药物发现问题。我们使用MMAI Gym训练了一个高效的液体基础模型（LFM）用于这些应用，证明了更小规模、有针对性训练的基础模型在分子基准测试中能够超越规模大得多的通用模型或专业模型。在关键的药物发现任务中——包括分子优化、ADMET性质预测等……

    arXiv:2603.03517v2 Announce Type: replace-cross  Abstract: General-purpose large language models (LLMs) that rely on in-context learning do not reliably deliver the scientific understanding and performance required for drug discovery tasks. Simply increasing model size or introducing reasoning tokens does not yield significant performance gains. To address this gap, we introduce the MMAI Gym for Science, a one-stop shop molecular data formats and modalities as well as task-specific reasoning, training, and benchmarking recipes designed to teach foundation models the 'language of molecules' in order to solve practical drug discovery problems. We use MMAI Gym to train an efficient Liquid Foundation Model (LFM) for these applications, demonstrating that smaller, purpose-trained foundation models can outperform substantially larger general-purpose or specialist models on molecular benchmarks. Across essential drug discovery tasks - including molecular optimization, ADMET property predictio
    
[^275]: 因果语言模型的后缀约束贪心搜索算法

    Suffix-Constrained Greedy Search Algorithms for Causal Language Models

    [https://arxiv.org/abs/2603.01243](https://arxiv.org/abs/2603.01243)

    本文提出“后缀约束生成”这一新的受限生成设定，仅要求响应的结尾部分符合语法规则，并设计了多种基于贪心搜索的算法来解决现有受限生成方法无法支持该场景的问题。

    

    大语言模型（LLM）是强大的工具，其应用已超越人机交互界面和聊天机器人的范畴。除了自由形式的生成之外，受限生成也引起了人们的关注，即约束LLM按照形式语法所定义的语言生成格式良好的输出。尽管这种设定颇具吸引力，但对下游应用而言可能过于严苛。例如，许多LLM任务需要模型先自由地进行推理，然后再以特定格式生成最终响应。在本工作中，我们提出了后缀约束生成，这是一种受限生成的设定，即仅对响应的末尾部分施加语法约束，这是现有受限生成方法所不支持的场景。我们提出了几种基于贪心搜索的后缀约束生成算法。我们在多个数据集上进行了实验，结果表明我们的方法能够保证……（原文摘要在此处截断）

    arXiv:2603.01243v3 Announce Type: replace  Abstract: Large language models (LLMs) are powerful tools that have found applications beyond human-machine interfaces and chatbots. Beside free-form generation, there has been an interest in constrained generation, a setting where LLMs are constrained to generate well-formed outputs with respect to the language defined by a formal grammar. Although appealing, this setting may be over restrictive for downstream applications. For example, many LLM tasks require the model to reason freely before generating its final response in a specific format.   In this work, we introduce suffix-constrained generation, a constrained generation setting in which only the end of the response is constrained by a grammar, a scenario that is not supported by existing constrained generation methods. We introduce several suffix-constrained generation algorithms that are based on greedy search. We experiment on several datasets, and show that our approach allows to gu
    
[^276]: GRRM：面向机器翻译的群体相对奖励建模

    GRRM: Group Relative Reward Modeling for Machine Translation

    [https://arxiv.org/abs/2602.14028](https://arxiv.org/abs/2602.14028)

    提出群体相对奖励模型（GRRM），通过联合比较整个候选译文组而非孤立打分来实现准确的组内质量排序，将其集成到 GRPO 训练中可显著提升机器翻译质量。

    

    尽管群体相对策略优化（GRPO）为大语言模型的后训练提供了一个强大的框架，但其在机器翻译等开放式领域中的有效性取决于准确的组内排序。我们发现标准的逐点质量度量（PQM）在这种场景下存在不足：候选译文是被孤立评估的，因此缺少用于区分细粒度语言差异的比较语境。为解决这一问题，我们提出了群体质量度量（GQM）范式及其实例化——群体相对奖励模型（GRRM）。与传统的独立打分器不同，GRRM 联合处理整个候选组，利用比较分析来严格判定相对质量并提供自适应粒度。实证评估证实，GRRM 在所有基线方法中取得了具有竞争力的排序准确率；将 GRRM 集成到 GRPO 训练中，不仅能提升整体翻译质量，还能解锁推……（原文摘要在此处截断）

    arXiv:2602.14028v2 Announce Type: replace  Abstract: While Group Relative Policy Optimization (GRPO) offers a powerful framework for LLM post-training, its effectiveness in open-ended domains like Machine Translation hinges on accurate intra-group ranking. We identify that standard Pointwise Quality Metrics (PQM) fall short in this context: candidates are evaluated in isolation, so the comparative context is missing for distinguishing fine-grained linguistic nuances. To address this, we introduce the Group Quality Metric (GQM) paradigm and its instantiation, the Group Relative Reward Model (GRRM). Unlike traditional independent scorers, GRRM jointly processes the entire candidate group, leveraging comparative analysis to rigorously resolve relative quality and adaptive granularity. Empirical evaluations confirm that GRRM achieves competitive ranking accuracy among all baselines; integrating GRRM into the GRPO training not only improves general translation quality but also unlocks reaso
    
[^277]: 知识蒸馏真的更环保吗？——机器翻译中的案例研究

    Is Knowledge Distillation Actually Greener? A Case Study in Machine Translation

    [https://arxiv.org/abs/2602.09691](https://arxiv.org/abs/2602.09691)

    该研究首次借助机器学习生命周期评估工具，从环境成本角度系统评估机器翻译中的知识蒸馏方法，发现摊销蒸馏成本所需的部署量取决于服务方式，且在批处理下可能变化数个数量级。

    

    知识蒸馏（KD）是一种将较大的教师系统压缩为更小的学生系统的技术。在机器翻译中，知识蒸馏通常通过翻译质量和推理效率来评估，而没有共同考虑生产和部署蒸馏系统所产生的环境成本。我们在定制的机器翻译模型和大语言模型上评估了具有代表性的知识蒸馏方法，同时考虑翻译质量和计算成本，并使用机器学习生命周期评估工具，该工具能够核算知识蒸馏模型整个生命周期中的成本。我们的关键发现是：摊销知识蒸馏成本所需的部署量取决于服务方式，并且在批处理条件下可能变化数个数量级。我们还提供了在质量和计算约束下选择、开发和评估知识蒸馏方法的可操作指导。

    arXiv:2602.09691v2 Announce Type: replace  Abstract: Knowledge distillation (KD) is a technique to compress a larger teacher system into a smaller student. In machine translation, KD is commonly evaluated through translation quality and inference efficiency, without jointly accounting for the environmental costs of producing and deploying the distilled system. We evaluate representative KD methods both on bespoke MT models and LLMs, by considering both translation quality and computational cost, using the Machine Learning Life Cycle Assessment tool, which accounts for costs throughout the KD model life cycle. Our key finding is that the deployment volume required to amortize KD is serving-dependent and can shift by several orders of magnitude under batching. We include actionable guidance for selecting, developing, and evaluating KD methods under quality and compute-induced constraints.
    
[^278]: 超越词元：通过探测内部状态实现高效推理的语义感知推测解码

    Beyond Tokens: Semantic-Aware Speculative Decoding for Efficient Inference by Probing Internal States

    [https://arxiv.org/abs/2602.03708](https://arxiv.org/abs/2602.03708)

    提出语义感知推测解码框架SemanticSpec，通过探测模型内部隐藏状态来在语义层面而非词元层面进行序列验证，从而显著减少拒绝次数，在大型推理模型上实现最高2.7倍的推理加速。

    

    大语言模型（LLM）在众多任务中表现出色，但由于自回归解码而面临较高的推理延迟。这一问题在生成冗长思维链的大型推理模型（LRM）中尤为严重。尽管推测解码通过并行起草和验证多个词元来加速推理，但现有方法在词元级别进行操作，忽略了语义等价性（即不同的词元序列表达相同含义），导致低效的拒绝。我们提出SemanticSpec，一个语义感知的推测解码框架，它验证整个语义序列而非词元。SemanticSpec引入了一种语义概率估计机制，通过探测模型内部的隐藏状态来评估生成具有特定含义序列的可能性。在四个基准测试上的实验表明，SemanticSpec在DeepSeekR1-32B上实现了高达2.7倍的加速，并且

    arXiv:2602.03708v3 Announce Type: replace  Abstract: Large Language Models (LLMs) achieve strong performance across many tasks but suffer from high inference latency due to autoregressive decoding. The issue is exacerbated in Large Reasoning Models (LRMs), which generate lengthy chains of thought. While speculative decoding accelerates inference by drafting and verifying multiple tokens in parallel, existing methods operate at the token level and ignore semantic equivalence (i.e., different token sequences expressing the same meaning), leading to inefficient rejections. We propose SemanticSpec, a semantic-aware speculative decoding framework that verifies entire semantic sequences instead of tokens. SemanticSpec introduces a semantic probability estimation mechanism that probes the model's internal hidden states to assess the likelihood of generating sequences with specific meanings. Experiments on four benchmarks show that SemanticSpec achieves up to 2.7x speedup on DeepSeekR1-32B and
    
[^279]: MAS-ProVe：理解多智能体系统的过程验证

    MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems

    [https://arxiv.org/abs/2602.03053](https://arxiv.org/abs/2602.03053)

    本文提出MAS-ProVe，首次对多智能体系统中的过程验证展开系统性实证研究，涵盖三种验证范式、两种验证粒度、五种验证器和四种上下文管理策略，并发现过程级验证在多智能体系统中并不能持续稳定地带来改进。

    

    基于大语言模型构建的多智能体系统在推理轨迹上往往表现出较高的方差。过程验证通过评估轨迹中的中间步骤，已在一般推理场景中展现出潜力，并被认为可能成为指导多智能体系统协调的工具；然而，其在多智能体系统中的实际有效性仍不明确。为填补这一空白，我们提出了MAS-ProVe，一项针对多智能体系统过程验证的系统性实证研究。我们的研究涵盖三种验证范式（LLM作为评判者、奖励模型和过程奖励模型），并在两个验证粒度层级（智能体级和迭代级）上进行评估。我们进一步考察了五种代表性验证器和四种上下文管理策略，并在多个推理基准上对六种不同的多智能体框架开展了实验。我们发现过程级验证并不能持续稳定地改进……（原文摘要至此截断）

    arXiv:2602.03053v2 Announce Type: replace  Abstract: Multi-Agent Systems (MAS) built on Large Language Models (LLMs) often exhibit high variance in their reasoning trajectories. Process verification, which evaluates intermediate steps in trajectories, has shown promise in general reasoning settings, and has been suggested as a potential tool for guiding coordination of MAS; however, its actual effectiveness in MAS remains unclear. To fill this gap, we present MAS-ProVe, a systematic empirical study of process verification for multi-agent systems (MAS). Our study spans three verification paradigms (LLM-as-a-Judge, reward models, and process reward models), evaluated across two levels of verification granularity (agent-level and iteration-level). We further examine five representative verifiers and four context management strategies, and conduct experiments over six diverse MAS frameworks on multiple reasoning benchmarks. We find that process-level verification does not consistently impr
    
[^280]: 像医生一样思考：通过探索诊断知识图谱实现对话式诊断

    Think Like a Doctor: Conversational Diagnosis through the Exploration of Diagnostic Knowledge Graphs

    [https://arxiv.org/abs/2602.01995](https://arxiv.org/abs/2602.01995)

    该论文提出了一种通过探索诊断知识图谱进行两步推理（先生成诊断假设、再通过澄清性问题反复验证）的对话式诊断系统，并结合基于人设的患者模拟器PatientSim与MIMIC-IV患者档案进行更贴近真实场景的评估。

    

    对话式诊断需要进行多轮问诊，即智能体在信息不完整的情况下通过提出澄清性问题来逐步细化鉴别诊断。现有方法通常依赖于模型的参数化知识，或者假设患者能够提供丰富而具体的信息，这在现实中是不切实际的。为了解决这些局限性，我们提出了一种对话式诊断系统，该系统通过探索诊断知识图谱进行两步推理：(i) 从对话上下文中生成诊断假设；(ii) 通过澄清性问题验证假设，这一过程循环往复，直到得出最终诊断。由于评估该系统需要一个能够对系统提问做出回应的真实患者模拟器，我们采用了基于人设的患者模拟器PatientSim，并结合MIMIC-IV中的患者档案。我们进一步对其进行了改进，加入低特异性症状报告机制，以反映真实世界中患者的……（原文摘要不完整）

    arXiv:2602.01995v2 Announce Type: replace  Abstract: Conversational diagnosis requires multi-turn history-taking, where an agent asks clarifying questions to refine differential diagnoses under incomplete information. Existing approaches often rely on the parametric knowledge of a model or assume that patients provide rich and concrete information, which is unrealistic. To address these limitations, we propose a conversational diagnosis system that explores a diagnostic knowledge graph to reason in two steps: (i) generating diagnostic hypotheses from the dialogue context, and (ii) verifying hypotheses through clarifying questions, which are repeated until a final diagnosis is reached. Since evaluating the system requires a realistic patient simulator that responds to the system's questions, we adopt PatientSim, a persona-driven patient simulator, together with patient profiles from MIMIC-IV. We further adapt it with low-specificity symptom reporting to reflect how real-world patients d
    
[^281]: NewsRECON：用于图像背景化的新闻文章检索

    NewsRECON: News Article Retrieval for Image Contextualization

    [https://arxiv.org/abs/2601.14121](https://arxiv.org/abs/2601.14121)

    提出NewsRECON新闻文章检索管道，利用超过85,000篇新闻文章的语料库作为反向图像搜索的替代方案，通过将图像与相关文章关联并从元数据中推断拍摄时间和地点，在与多模态大语言模型结合后于缺乏RIS证据的场景下取得了新的SOTA结果。

    

    确定新闻图像的拍摄时间和地点对于记者和取证专家制作可信的报道以及揭穿虚假信息至关重要。虽然许多现有方法依赖于反向图像搜索（RIS）引擎，但这些工具经常无法返回结果，从而限制了其实际适用性。在这项工作中，我们解决了RIS证据不可用这一具有挑战性的场景。我们研究了新闻文章语料库作为RIS替代方案的潜力，将图像与相关文章进行关联，并从文章元数据中推断图像的日期和位置。我们评估了新闻文章检索管道NewsRECON的性能，该管道利用了超过85,000篇文章的语料库。在TARA数据集上的实验表明，NewsRECON优于先前的工作，并且可以与多模态大语言模型（MLLM）相结合，在缺乏RIS证据的情况下取得新的SOTA结果。此外，NewsRECON可以泛化到5Pil（摘要在此处截断）

    arXiv:2601.14121v2 Announce Type: replace  Abstract: Identifying when and where a news image was taken is crucial for journalists and forensic experts to produce credible stories and debunk misinformation. While many existing methods rely on reverse image search (RIS) engines, these tools often fail to return results, thereby limiting their practical applicability. In this work, we address the challenging scenario where RIS evidence is unavailable. We investigate the potential of news article corpora as an alternative to RIS, linking images to relevant articles to infer their dates and locations from article metadata. We evaluate the performance of a news article retrieval pipeline, NewsRECON, which leverages a corpus of over 85,000 articles. Experiments on the TARA dataset show that NewsRECON outperforms prior work and can be combined with a multimodal large language model (MLLM) to achieve new SOTA results in the absence of RIS evidence. Furthermore, NewsRECON generalizes to the 5Pil
    
[^282]: DAGGER：面向数学问题可执行推理的干扰感知图生成

    DAGGER: Distractor-Aware Graph Generation for Executable Reasoning in Math Problems

    [https://arxiv.org/abs/2601.06853](https://arxiv.org/abs/2601.06853)

    该论文提出DAGGER方法，将含干扰信息的数学问题求解重构为显式建模干扰节点的可执行计算图生成，有效缓解了思维链推理在无关信息干扰下的严重性能退化。

    

    思维链提示已被广泛应用于数学问题求解，包括在低资源语言场景中，但其在无关上下文干扰下的行为仍未得到充分研究。为了系统地研究这一挑战，我们提出了DISTRACTMATH-BN，一个孟加拉语基准数据集，它在MGSM和MSVAMP的基础上增加了语义连贯但与计算无关的干扰信息。通过对七个参数量从3B到12B的模型进行评估，我们观察到干扰信息会导致显著的性能退化：标准模型最多下降41分，而专门强化推理能力的模型也下降了14至20分，且其 token 消耗量高达原来的五倍。我们提出DAGGER，将数学问题求解重新表述为可执行计算图的生成，并对干扰节点进行显式建模。通过监督微调并结合群体相对策略优化（GRPO）对Gemma-3模型进行微调，取得了可比的加权准确率……

    arXiv:2601.06853v3 Announce Type: replace  Abstract: Chain-of-Thought (CoT) prompting is widely adopted for mathematical problem solving, including in low-resource languages, yet its behavior under irrelevant context remains underexplored. To systematically study this challenge, we introduce DISTRACTMATH-BN, a Bangla benchmark that augments MGSM and MSVAMP with semantically coherent but computationally irrelevant information. Evaluating seven models ranging from 3B to 12B parameters, we observe substantial performance degradation under distractors: standard models drop by up to 41 points, while reasoning-specialized models decline by 14 to 20 points despite consuming five times more tokens. We propose {\dag}DAGGER, which reformulates mathematical problem solving as executable computational graph generation with explicit modeling of distractor nodes. Fine-tuning Gemma-3 models using supervised fine-tuning followed by Group Relative Policy Optimization achieves comparable weighted accura
    
[^283]: 超越静态摘要：面向LLM智能体的主动记忆提取

    Beyond Static Summarization: Proactive Memory Extraction for LLM Agents

    [https://arxiv.org/abs/2601.04463](https://arxiv.org/abs/2601.04463)

    该论文提出主动记忆提取框架ProMem，通过分离细节、事件与关系并采用分类提取策略、完整性检查和原子级事实验证，解决了现有记忆提取方法因提前进行和一次性提取而导致的信息丢失与幻觉残留问题。

    

    记忆管理对于LLM智能体在长期和个性化交互中至关重要。以往的大多数工作研究如何检索和使用记忆，但较少关注记忆是如何提取的。我们发现了现有方法的两个主要局限。首先，提取是“提前进行的”：智能体在了解未来任务之前就保存信息，而单一的摘要提示往往会混淆细节、事件和关系，导致有用信息的丢失。其次，提取通常是一次性的，在没有验证的情况下，错误和幻觉可能会长期留在记忆中。为了解决这些局限，我们提出了ProMem，一个主动记忆提取框架。它将细节、事件和关系分离，并对每种类型采用不同的提取策略。它还通过完整性检查来恢复遗漏的事件，并在原子级别验证事实以减少幻觉。实验表明，ProMem提高了记忆完整性和问答准确率。

    arXiv:2601.04463v2 Announce Type: replace-cross  Abstract: Memory management is vital for LLM agents in long-term and personalized interactions. Most previous work studies how to retrieve and use memory, but pays less attention to how memory is extracted. We find two main limitations in existing methods. First, extraction is "ahead-of-time": the agent saves information before it knows future tasks. A single summary prompt often mixes details, events, and relations, so useful information is lost. Second, extraction is usually one-off. Without verification, errors and hallucinations may stay in memory for a long time. To address these limitations, we propose ProMem, a proactive memory extraction framework. It separates details, events, and relations, and uses different extraction strategies for each type. It also checks completeness to recover missed events and verifies facts at the atomic level to reduce hallucinations. Experiments show that ProMem improves memory completeness and QA ac
    
[^284]: 针对基于Mamba的语言模型的隐状态投毒攻击

    Hidden State Poisoning Attacks against Mamba-based Language Models

    [https://arxiv.org/abs/2601.01972](https://arxiv.org/abs/2601.01972)

    该论文首次揭示了针对Mamba等状态空间语言模型的隐状态投毒攻击（HiSPA）——特定短输入短语可不可逆地覆盖模型隐藏状态导致部分失忆，并提出RoBench-25基准证实了包括520亿参数的Jamba混合模型在内的SSMs对此类攻击的脆弱性，而纯Transformer模型则不受影响。

    

    像Mamba这样的状态空间模型（SSMs）以线性时间复杂度为基于Transformer的语言模型提供了高效替代方案。然而，其对抗鲁棒性却鲜有研究。本文研究了特定短输入短语通过不可逆地覆盖模型隐藏状态中的信息，从而在此类模型中诱发部分“失忆”效应的现象，我们将其称为隐状态投毒攻击。我们提出的基准测试RoBench-25可以评估模型在遭受HiSPA攻击时的信息检索能力，并证实了SSMs对此类攻击的脆弱性。即使是最近的Jamba-1.7-Mini SSM-Transformer混合模型（520亿参数），在某些HiSPA触发器作用下也会在RoBench-25上完全失效，而纯Transformer模型则不会。我们还观察到，与纯Transformer不同，HiSPA触发器在流行的Open-Prompt-Injections基准测试中显著削弱了Jamba模型的表现。我们进一步表明，该理（摘要原文在此处截断）

    arXiv:2601.01972v5 Announce Type: replace-cross  Abstract: State space models (SSMs) like Mamba offer efficient alternatives to Transformer-based language models, with linear time complexity. Yet, their adversarial robustness remains critically unexplored. This paper studies the phenomenon whereby specific short input phrases induce a partial amnesia effect in such models, by irreversibly overwriting information in their hidden states, referred to as a Hidden State Poisoning Attack (HiSPA). Our benchmark RoBench-25 allows evaluating a model's information retrieval capabilities when subject to HiSPAs, and confirms the vulnerability of SSMs against such attacks. Even the recent Jamba-1.7-Mini SSM--Transformer (a 52B hybrid model) collapses on RoBench-25 under some HiSPA triggers, whereas pure Transformers do not. We also observe that HiSPA triggers significantly weaken the Jamba model on the popular Open-Prompt-Injections benchmark, unlike pure Transformers. We further show that the theo
    
[^285]: AdaSearch：通过强化学习平衡大语言模型中的参数化知识与搜索

    AdaSearch: Balancing Parametric Knowledge and Search in Large Language Models via Reinforcement Learning

    [https://arxiv.org/abs/2512.16883](https://arxiv.org/abs/2512.16883)

    AdaSearch提出了一种简单的两阶段、以结果为导向的强化学习框架，将问题求解与搜索决策解耦，使大语言模型能够自适应地平衡内部参数化知识与外部搜索，从而避免过度搜索的成本与风险以及纯参数化知识带来的幻觉问题。

    

    通过强化学习（RL）为大型语言模型（LLM）配备搜索引擎，有望构建高效的搜索智能体。然而，如何在内部参数化知识与外部搜索之间实现自适应平衡仍然是一个挑战：过度依赖搜索会带来不必要的成本，并存在暴露于噪声或恶意内容的风险，而仅依赖参数化知识则面临产生幻觉的风险。先前的研究通过工具调用奖励塑形来缓解搜索的过度使用，但这需要繁重的奖励工程，并且混淆了必要与不必要的搜索。为了解决这些局限性，我们通过基于F1的决策指标重新审视搜索智能体的评估，发现先前的方法常常忽略了模型中易于获取的参数化知识。受此启发，我们提出了AdaSearch，这是一个简单的两阶段、以结果为导向的强化学习框架，它将问题求解与搜索决策解耦，使决策过程显式化并具有内省性。

    arXiv:2512.16883v2 Announce Type: replace  Abstract: Equipping large language models (LLMs) with search engines via reinforcement learning (RL) promises effective search agents. However, adaptively balancing internal parametric knowledge with external search remains a challenge, as overreliance on search introduces unnecessary cost and risks exposure to noisy or malicious content, while relying solely on parametric knowledge risks hallucination. Prior efforts mitigate search overuse through tool-call reward shaping, which requires heavy reward engineering and conflates necessary and unnecessary search. To address these limitations, we revisit the evaluation of search agents through an F1-based decision metric, revealing that prior methods often overlook readily available parametric knowledge. Motivated by this, we propose AdaSearch, a simple two-stage, outcome-driven RL framework that disentangles problem-solving from the decision to search, making the decision process explicit and int
    
[^286]: 面向问答任务的大语言模型多语言医学推理

    Multilingual Medical Reasoning for Question Answering with Large Language Models

    [https://arxiv.org/abs/2512.05658](https://arxiv.org/abs/2512.05658)

    该论文提出了一种基于维基百科医学知识、采用检索增强生成方法构建多语言（英语、意大利语、西班牙语）医学推理轨迹的技术，生成了50万条推理数据，并证明这些数据在少样本学习和监督微调两种方式下均能显著提升大语言模型在医学问答任务上的表现。

    

    具有推理能力的大语言模型（LLM）近来在医学问答（QA）任务中展现出了强大的潜力。现有方法大多以英语为中心，且主要依赖于从通用大语言模型进行知识蒸馏，这引发了人们对其医学知识可靠性的担忧。在本工作中，我们提出了一种基于从维基百科提取的医学知识来生成多语言推理轨迹的方法。我们采用检索增强生成（RAG）技术，基于维基百科中的医学信息，生成了50万条英语、意大利语和西班牙语的推理轨迹。这些轨迹用于解答来自MedQA和MedMCQA的医学问题，我们将这两个数据集扩展到了意大利语和西班牙语。我们在多个医学问答基准上进行了域内和域外设置的测试，结果表明，无论是通过上下文学习（少样本）还是监督微调的方式使用，我们的推理轨迹都能提升模型性能。

    arXiv:2512.05658v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) with reasoning capabilities have recently demonstrated strong potential in medical Question Answering (QA). Existing approaches are largely English-focused and primarily rely on distillation from general-purpose LLMs, raising concerns about the reliability of their medical knowledge. In this work, we present a method to generate multilingual reasoning traces based on medical knowledge extracted from Wikipedia. We produce 500k traces in English, Italian, and Spanish, using a retrieval-augmented generation approach over medical information from Wikipedia. The traces are generated to solve medical questions drawn from MedQA and MedMCQA, which we extend to Italian and Spanish. We test our pipeline in both in-domain and out-of-domain settings across Medical QA benchmarks, and demonstrate that our reasoning traces improve performance both when utilized via in-context learning (few-shot) and supervised fin
    
[^287]: 桌上的苹果？通过细粒度约束验证评估文本引导的三维场景合成

    Apples on the Table? Evaluating Text-Guided 3D Scene Synthesis via Fine-Grained Constraint Verification

    [https://arxiv.org/abs/2511.03001](https://arxiv.org/abs/2511.03001)

    提出了包含人工标注约束的LEGO基准数据集和LEGO-Eval评估框架，通过将用户描述分解为原子约束并利用三维定位与空间关系推理逐一验证，实现对文本引导三维场景合成的细粒度评估。

    

    根据用户提供的文本描述准确合成三维场景，对于开发具身智能体至关重要。尽管场景与描述的对齐十分重要，但现有的文本引导三维场景合成评估方法要么只能捕捉合成场景与用户描述之间的粗略相似性，要么忽略了对物体放置的空间推理验证，没有任何方法能够处理用户描述中隐含的细粒度约束（例如，X需要以Y方式出现在场景中）。为解决这一问题，我们引入了LEGO——一个将每条用户描述与人工标注的约束及参考场景相配对的基准数据集，以及LEGO-Eval——一个将描述分解为原子约束并对每个约束进行验证的评估框架，该框架借助工具将文本引用定位到三维物体并推理它们之间的空间关系。我们证明了：（i）LEGO-Eval能够评估错位……

    arXiv:2511.03001v3 Announce Type: replace  Abstract: Accurately synthesizing 3D scenes from user-provided text descriptions is crucial for developing embodied agents. Despite the importance of scene-description alignment, existing evaluation methods for such text-guided 3D scene synthesis either capture only coarse similarity between the synthesized scene and the user description, or ignore the spatial reasoning for verifying object placement. None of them addressed the fine-grained constraints (e.g., X needs to be in the scene in a Y manner) implied by the description from users. To address this, we introduce LEGO, a benchmark dataset that pairs each user description with human-annotated constraints and a reference scene, and LEGO-Eval, an evaluation framework that decomposes a description into atomic constraints and verifies each one using tools that ground textual references to 3D objects and reason about their spatial relationships. We show that (i) LEGO-Eval evaluates misalignment
    
[^288]: 基于秩-2子空间解缠的多步知识交互分析

    Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglement

    [https://arxiv.org/abs/2511.01706](https://arxiv.org/abs/2511.01706)

    该论文提出一种新颖的秩-2投影子空间来更准确地解缠大语言模型中参数化知识与情境知识的贡献，并首次实现了对自然语言解释更长生成序列中知识交互的多步分析。

    

    自然语言解释（NLEs）通过借助外部情境知识（CK）和参数化知识（PK）来描述大语言模型（LLMs）如何做出决策。理解这些知识来源之间的交互是评估NLE接地性的关键，然而这些动态机制仍未得到充分探索。先前的工作主要集中于：i）单步生成，以及ii）将PK与CK的交互建模为秩-1子空间内的二元选择。这种方法忽略了更丰富的交互形式，以及这些交互在更长生成过程中的演变方式，例如互补性或支持性知识。我们提出了一种新颖的秩-2投影子空间，能够更准确地解缠PK和CK的贡献，并首次将其用于对更长NLE序列中知识交互的多步分析。在四个问答数据集和三个开源权重LLM上的实验表明，秩-1子空间难以表示多样化的知识交互，而我们的秩-2方法能够更好地捕捉这些丰富的交互形式。

    arXiv:2511.01706v3 Announce Type: replace-cross  Abstract: Natural Language Explanations (NLEs) describe how Large Language Models (LLMs) make decisions by drawing on external Context Knowledge (CK) and Parametric Knowledge (PK). Understanding the interaction between these sources is key to assessing NLE grounding, yet these dynamics remain underexplored. Prior work has largely focused on i) single-step generation and ii) modeled PK--CK interaction as a binary choice within a rank-1 subspace. This approach overlooks richer interactions and how they unfold over longer generations, such as complementary or supportive knowledge. We propose a novel rank-2 projection subspace that disentangles PK and CK contributions more accurately and use it for the first multi-step analysis of knowledge interactions across longer NLE sequences. Experiments across four QA datasets and three open-weight LLMs demonstrate that rank-1 subspaces struggle to represent diverse interactions, whereas our rank-2 fo
    
[^289]: 简单的增补，显著的收益：扩展URIEL+中的文字系统、语言与谱系覆盖

    Simple Additions, Substantial Gains: Expanding Scripts, Languages, and Lineage Coverage in URIEL+

    [https://arxiv.org/abs/2510.27183](https://arxiv.org/abs/2510.27183)

    URIEL+通过引入文字向量、整合Glottolog数据库并扩展谱系插补，大幅提升了语言知识库的覆盖范围与数据质量，显著增强了对低资源语言的跨语言迁移支持能力。

    

    URIEL+语言学知识库通过地理、谱系和类型学向量对语言进行编码，以支持多语言研究。然而，数据稀疏性问题（如特征类型缺失、语言条目不完整以及谱系覆盖有限）依然普遍存在。这限制了URIEL+在跨语言迁移中的实用性，尤其是在支持低资源语言方面。为解决这种稀疏性问题，我们对URIEL+进行了扩展：引入文字向量以表示7,488种语言的书写系统属性，整合Glottolog数据库新增18,710种语言，并通过在谱系中传播类型学和文字特征，将谱系插补扩展至26,449种语言。这些改进使文字向量的特征稀疏性降低了14%，语言覆盖增加了多达19,015种语言（提升1,007%），插补质量指标提升了多达35%。我们在跨语言迁移任务上的基准测试（面向……

    arXiv:2510.27183v3 Announce Type: replace  Abstract: The URIEL+ linguistic knowledge base supports multilingual research by encoding languages through geographic, genetic, and typological vectors. However, data sparsity (e.g. missing feature types, incomplete language entries, and limited genealogical coverage) remains prevalent. This limits the usefulness of URIEL+ in cross-lingual transfer, particularly for supporting low-resource languages. To address this sparsity, we extend URIEL+ by introducing script vectors to represent writing system properties for 7,488 languages, integrating Glottolog to add 18,710 additional languages, and expanding lineage imputation for 26,449 languages by propagating typological and script features across genealogies. These improvements reduce feature sparsity by 14% for script vectors, increase language coverage by up to 19,015 languages (1,007%), and boost imputation quality metrics by up to 35%. Our benchmark on cross-lingual transfer tasks (oriented 
    
[^290]: M4FC：一个多模态、多语言、多文化、多任务的真实世界事实核查数据集

    M4FC: a Multimodal, Multilingual, Multicultural, Multitask Real-World Fact-Checking Dataset

    [https://arxiv.org/abs/2510.23508](https://arxiv.org/abs/2510.23508)

    本文提出了一个包含 4,982 张图像和 6,980 条声明、覆盖十种语言及六个事实核查任务的多模态多语言多文化多任务真实世界事实核查数据集 M4FC，并提供了各任务的基线结果。

    

    现有的用于多模态事实核查的真实世界数据集存在多种局限性：它们包含的实例数量少、仅覆盖一种或两种语言、只专注于单一任务，或者依赖外部新闻文章集合来获取真实声明。为了解决这些缺陷，我们推出了 M4FC，这是一个新的真实世界数据集，包含 4,982 张图像和 6,980 条声明。这些图像由来自 22 个组织的专业事实核查员验证，代表了多样化的文化和地理背景。每条声明可以以十种语言中的一种或两种呈现。M4FC 涵盖了六个多模态事实核查任务：视觉声明提取、发布者意图预测、虚假图像检测、图像情境化、位置验证和结论预测。我们为所有任务提供了基线结果，并分析了将中间任务组合如何影响结论预测的性能。我们的数据集和代码均已公开。

    arXiv:2510.23508v4 Announce Type: replace  Abstract: Existing real-world datasets for multimodal fact-checking have multiple limitations: they contain few instances, cover only one or two languages, focus on a single task, or rely on external news article sets to source true claims. To address these shortcomings, we introduce M4FC, a new real-world dataset comprising 4,982 images paired with 6,980 claims. The images, verified by professional fact-checkers from 22 organizations, represent a diverse range of cultural and geographic contexts. Each claim is available in one or two out of ten languages. M4FC spans six multimodal fact-checking tasks: visual claim extraction, claimant intent prediction, fake image detection, image contextualization, location verification, and verdict prediction. We provide baseline results for all tasks and analyze how combining intermediate tasks affects verdict prediction performance. We make our dataset and code publicly available.
    
[^291]: HugAgent：一个面向个体层面推理的人类模拟基准

    HugAgent: A Human Simulation Benchmark for Individual-Level Reasoning

    [https://arxiv.org/abs/2510.15144](https://arxiv.org/abs/2510.15144)

    该论文提出HugAgent基准，从个性化推理、认知对齐和开放式数据三个维度重新定义人类推理模拟，评估模型能否基于某人历史观点的部分证据，预测该特定个体在分布外场景中的行为反应与推理动态。

    

    在开放式任务中模拟人类推理长期以来一直是人工智能和认知科学的核心追求。尽管大型语言模型现在能够大规模地近似人类反应，但它们仍是针对群体层面共识进行调优的，往往会抹杀推理风格和信念轨迹的个体性。为了推进机器实现更类人推理的愿景，我们提出了HugAgent（HUman-Grounded AGENT Benchmark，基于人类的智能体基准），从三个维度重新思考人类推理模拟：从平均化推理转向个性化推理；(ii) 从行为模仿转向认知对齐；(iii) 从基于情景片段的数据转向开放式数据。该基准评估的是：在给定某人先前观点的部分证据的情况下，模型能否预测该特定个体在分布外场景中的行为反应及其背后的推理动态。HugAgent将结构化问卷与半结构化的出声思考访谈相结合来收集……（原文截断）

    arXiv:2510.15144v4 Announce Type: replace  Abstract: Simulating human reasoning in open-ended tasks has long been a central aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (HUman-Grounded AGENT Benchmark), which rethinks human reasoning simulation along three dimensions: (i) from averaged to individualized reasoning, (ii) from behavioral mimicry to cognitive alignment, and (iii) from vignette-based to open-ended data. The benchmark evaluates whether a model can predict a specific person's behavioral responses and the underlying reasoning dynamics in out-of-distribution scenarios, given partial evidence of their prior views. HugAgent combines structured questionnaires with semi-structured think-aloud interviews to col
    
[^292]: 将组合式机器设计视为基于大语言模型的程序合成

    Compositional Machine Design as Program Synthesis with LLMs

    [https://arxiv.org/abs/2510.14980](https://arxiv.org/abs/2510.14980)

    该论文提出将机器设计视为一种以物理模拟验证为依据的程序合成新任务——组合式机器设计，并构建了基于游戏《Besiege》的测试平台BesiegeField，用于评测大语言模型在多种工作流下组合标准部件设计机器的能力。

    

    大语言模型（LLM）在编写和修改程序方面已展现出强大的能力，然而许多程序合成基准仍在符号或数字环境中评估程序。我们提出了“组合式机器设计”，这是一种以物理为基础的程序合成形式：机器被编写为组合标准化部件的程序，其成败由模拟的物理行为决定。为研究这一问题，我们提出了BesiegeField，一个基于机器建造游戏《Besiege》构建的测试平台。在BesiegeField中，LLM智能体根据文本形式的功能需求生成机器程序，在模拟中运行所得的机器，并接收奖励与状态反馈。我们在单智能体生成、迭代编辑和分层工作流等模式下，对LLM智能体在代表性机器设计任务上进行了基准评测。强大的模型能够恢复与任务相关的结构，有时还能取得不俗的物理性能表现，但常常……（原文摘要在此处截断）

    arXiv:2510.14980v3 Announce Type: replace  Abstract: Large language models (LLMs) have shown strong abilities in writing and revising programs, yet many program-synthesis benchmarks still evaluate programs in symbolic or digital environments. We introduce compositional machine design, a physically grounded form of program synthesis where machines are written as programs that compose standardized parts, and success is determined by simulated physical behavior. To study this problem, we present BesiegeField, a testbed built on the machine-building game Besiege. In BesiegeField, LLM agents generate machine programs from textual functional demands, execute the resulting machines in simulation, and receive rewards and state feedback. We benchmark LLM agents across representative machine-design tasks under single-agent generation, iterative editing, and hierarchical workflows. Strong models recover task-relevant structures and sometimes achieve nontrivial physical performance, but often stru
    
[^293]: 面向作者归属与验证的单样本风格迁移LLM对数概率方法

    One-shot Style Transfer LLM log-probabilities for Authorship Attribution and Verification

    [https://arxiv.org/abs/2510.13302](https://arxiv.org/abs/2510.13302)

    本文提出一种无监督框架，利用大语言模型的对数概率衡量文本间的风格可迁移性，无需显式监督即可在作者验证任务上显著超越基于提示的无监督基线，并在足够模型规模下与对比学习基线相当或更优。

    

    计算文体学通过定量的文本模式研究写作风格，能够支持作者归属、身份关联和抄袭检测等应用。尽管语言建模与这些任务密切相关，但现代大语言模型（LLM）的预训练在作者归属与验证领域尚未得到充分利用。我们提出了一个无监督框架，利用大语言模型的对数概率来衡量两个文本之间的风格可迁移性。该框架充分利用了大语言模型大规模的自回归语言建模（CLM）预训练、单样本能力和模型规模，避免了显式监督。在相近模型规模下，我们的方法在作者验证任务上显著优于基于提示的无监督基线；在模型规模足够的情况下，该方法在大多数设置中与对比学习基线相当或有所提升。此外，我们还观察到在……方面的强劲表现（注：原文摘要在此处被截断）。

    arXiv:2510.13302v4 Announce Type: replace-cross  Abstract: Computational stylometry studies writing style through quantitative textual patterns, enabling applications such as authorship attribution, identity linking, and plagiarism detection. Despite the relevance of language modeling to these tasks, the pre-training of modern large language models (LLMs) has been underutilized in authorship attribution and verification. We introduce an unsupervised framework that uses the log-probabilities of an LLM to measure style transferability between two texts. This framework takes advantage of the extensive Causal Language Modeling (CLM) pre-training, one-shot capabilities and scale of LLMs, avoiding explicit supervision. Our methods substantially outperform prompting-based unsupervised baselines in authorship verification at similar model sizes, and is competitive with or improves contrastive baselines in most settings with sufficient model scale. We further observe strong performance across n
    
[^294]: TopoAlign：通过拓扑分解将代码对齐到数学的框架

    TopoAlign: A Framework for Aligning Code to Math via Topological Decomposition

    [https://arxiv.org/abs/2510.11944](https://arxiv.org/abs/2510.11944)

    TopoAlign通过将代码分解为文档字符串、主函数和依赖函数并重新组装对齐，弥合了代码与形式化数学之间的结构与句法差异，从而将大规模代码仓库转化为可用于提升数学LLM自动形式化能力的训练资源。

    

    大型语言模型（LLMs）在非形式化和形式化（例如Lean 4）数学推理方面都表现出色，但在自动形式化任务上仍然存在困难，即如何将非形式化的数学陈述转换为形式化数学陈述。然而，当前数学LLMs的性能受到大规模语料库稀缺的制约，尤其是缺少包含非形式化与形式化陈述配对的数据集。有趣的是，用于自动形式化的形式化语言与编程语言在结构上具有相似性，且代码数据可以大规模获取。然而，目前在代码上训练的模型并不能有效地迁移到形式化数学任务中，原因在于两者之间存在结构和句法上的差异。为了解决这一问题，我们提出了TopoAlign，一个能够将广泛可用的代码仓库转化为数学LLMs训练资源的框架。TopoAlign将代码分解为文档字符串、主函数和依赖函数，并将这些组件重新组装成（摘要在此处截断）

    arXiv:2510.11944v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) excel at both informal and formal (e.g. Lean 4) mathematical reasoning but still struggle with autoformalisation, the task of transforming informal into formal mathematical statements. Yet, the performance of current Math LLMs is constrained by the scarcity of large-scale corpora, particularly those containing pairs of informal and formal statements. Interestingly, the formal languages used in autoformalisation share structural similarities with programming languages, and code data is available at scale. However, current models trained on code do not transfer effectively to formal math, due to structural and syntactic differences between them. To address this, we propose TopoAlign, a framework that unlocks widely available code repositories as training resources for Math LLMs. TopoAlign decomposes code into docstrings, main functions, and dependency functions, and reassembles these components into a
    
[^295]: Camellia：面向亚洲语言的大语言模型文化偏见基准测试

    Camellia: Benchmarking Cultural Biases in LLMs for Asian Languages

    [https://arxiv.org/abs/2510.05291](https://arxiv.org/abs/2510.05291)

    本文提出了Camellia基准测试，基于涵盖九种亚洲语言和六种亚洲文化的19,530个人工标注实体，在文化上下文适应、情感关联和实体抽取式问答三项任务中系统评估了多语言大语言模型存在的文化偏见。

    

    随着大语言模型（LLM）的多语言能力不断增强，它们对不同文化实体的敏感性变得越来越重要。Naous等人（2024）的先前研究表明，LLM在阿拉伯语中往往偏向与西方相关的实体。由于缺乏以实体为中心的多语言基准测试，这种偏见是否也会在各种非西方语言中表现出来仍不清楚。在本文中，我们提出了Camellia，这是一个用于评估九种亚洲语言（涵盖六种亚洲文化）中以实体为中心的文化偏见的基准测试。Camellia包含19,530个与所涵盖的亚洲或西方文化相关的人工标注实体，以及从社交媒体帖子中提取的2,173个针对这些实体的掩码上下文。利用Camellia，我们在三项任务中评估了四个最新多语言LLM的文化偏见：文化上下文适应、情感关联和实体抽取式问答。我们的分析表明，LLM...

    arXiv:2510.05291v3 Announce Type: replace  Abstract: As Large Language Models (LLMs) develop stronger multilingual capabilities, their sensitivity to culturally diverse entities becomes increasingly important. Prior work by Naous et al. (2024) has shown that LLMs often favor Western-associated entities in Arabic. Due to the lack of entity-centric multilingual benchmarks, it remains unclear if such biases also manifest in various non-Western languages. In this paper, we introduce Camellia, a benchmark for evaluating entity-centric cultural biases in nine Asian languages, spanning six Asian cultures. Camellia includes 19,530 manually annotated entities associated with the covered Asian or Western cultures, as well as 2,173 masked contexts for these entities derived from social media posts. Using Camellia, we evaluate cultural biases in four recent multilingual LLMs across three tasks: cultural context adaptation, sentiment association, and entity extractive QA. Our analyses show that LLM
    
[^296]: SupraTok：跨边界分词技术助力语言模型性能提升

    SupraTok: Cross-Boundary Tokenization for Enhanced Language Model Performance

    [https://arxiv.org/abs/2508.11857](https://arxiv.org/abs/2508.11857)

    SupraTok是一种跨越空白符边界的创新分词器，通过熵筛选、PMI引导的课程训练和多语言处理三大模块，在压缩率上比标准BPE提升17.5%，并比SuperBPE训练快2.1倍。

    

    分词一直是语言建模中持续存在的瓶颈，尤其是当词汇学习受到空白符边界的限制时。我们提出了SupraTok，这是一种能够跨越空白符边界的分词器，它由三个模块化组件构成：可选的基于熵的数据筛选、采用PMI引导候选搜索的分阶段课程训练，以及多语言文字处理。在相同的未过滤训练数据上，使用10万词汇量时，SupraTok的压缩效果比标准BPE提升17.5%，比官方SuperBPE实现提升1.8%，同时训练速度比SuperBPE快2.1倍。在相同匹配设置下，在5万至30万词汇量范围内，SupraTok始终比SuperBPE领先1.8%至8.6%。我们还将熵过滤作为流水线中的一个独立步骤单独评估：在10万词汇量下，它使SupraTok的C/T指标从5.78提升至5.99，而匹配对照组显示SuperBPE的增益较小，SP-BPE-CrossBoundary则几乎无变化。在FLORES-200的14种语言上的（摘要原文在此处截断）

    arXiv:2508.11857v3 Announce Type: replace-cross  Abstract: Tokenization remains a persistent bottleneck in language modeling, especially when vocabulary learning is limited by whitespace boundaries. We present SupraTok, a tokenizer that crosses whitespace boundaries using three modular components: optional entropy-based data curation, staged curriculum training with PMI-guided candidate search, and multilingual script handling. At 100k vocabulary on the same unfiltered training data, SupraTok improves compression over standard BPE by 17.5% and over the official SuperBPE implementation by 1.8%, while training 2.1x faster than SuperBPE. Across 50k-300k vocabularies in the same matched setting, SupraTok remains ahead of SuperBPE by 1.8%-8.6%. We evaluate entropy filtering separately as a pipeline step: at 100k vocabulary it raises SupraTok from 5.78 to 5.99 C/T, while matched controls show a smaller gain for SuperBPE and almost no change for SP-BPE-CrossBoundary. On FLORES-200 across 14 l
    
[^297]: BiasGym：一个通过注入来分析和消除偏见的简单且可泛化的框架

    BiasGym: A Simple and Generalizable Framework for Analyzing and Removing Biases through Injection

    [https://arxiv.org/abs/2508.08855](https://arxiv.org/abs/2508.08855)

    提出BiasGym框架，通过在冻结的LLM中注入特定偏见信号，再利用这些信号定位并抑制或引导导致偏见行为的模型组件，实现偏见的可靠分析与消除。

    

    理解大型语言模型（LLM）权重中编码的偏见和刻板印象，对于制定有效的缓解策略至关重要。然而，偏见行为往往是微妙且难以隔离的，即使刻意去引发也是如此，这使得系统性的分析和去偏工作尤其具有挑战性。为了解决这一问题，我们提出了一个简单、低成本且可泛化的框架 BiasGym，用于可靠地注入、分析和缓解 LLM 中偏见的概念关联。BiasGym 包含两个模块：Inject，通过基于 token 的微调（同时保持模型冻结）向模型注入特定偏见；以及两种去偏方法，利用这些注入信号来识别并可靠地抑制（Scope）或引导（Steer）导致偏见行为的模型组件。我们的框架能够实现一致的偏见引发，从而更好地定位…

    arXiv:2508.08855v5 Announce Type: replace-cross  Abstract: Understanding biases and stereotypes encoded in the weights of Large Language Models (LLMs) is crucial for developing effective mitigation strategies. However, biased behavior is often subtle and non-trivial to isolate, even when deliberately elicited, making systematic analysis and debiasing particularly challenging. To address this, we introduce a simple, cost-effective, and generalizable framework \texttt{BiasGym} for reliably injecting, analyzing, and mitigating conceptual associations of biases within LLMs. \texttt{BiasGym} consists of two modules: \texttt{Inject}, which injects specific biases into the model via token-based fine-tuning while keeping the model frozen, followed by two debiasing methods that leverage these injected signals to identify and reliably suppress (\texttt{Scope}) or \texttt{Steer} the components responsible for biased behavior. Our framework enables consistent bias elicitation for better localizati
    
[^298]: 评估风格个性化文本生成：挑战与方向

    Evaluating Style-Personalized Text Generation: Challenges and Directions

    [https://arxiv.org/abs/2508.06374](https://arxiv.org/abs/2508.06374)

    本文针对风格个性化文本生成评估指标的可靠性问题，批判性检验了BLEU、嵌入向量和LLM评判者等常用指标的有效性，并提出了一个涵盖三个领域、八项写作任务的风格判别基准来系统评估这些指标。

    

    随着大语言模型（LLM）的兴起及其生成定制化输出的能力，“风格个性化文本生成”——即“像我一样写作”——已成为一个快速发展的研究热点领域。然而，风格个性化具有高度特异性，因每个用户而异，且强烈依赖于语用情境，这使其面临独特的挑战。尽管先前的研究已为该领域引入了各种基准和评估指标，但这些指标往往缺乏标准化，且存在已知的局限性（例如与人类评判的相关性较差）。鉴于已有研究发现大语言模型无法很好地捕捉作者的个人风格，因此评估指标本身也必须受到严格审视。在这项工作中，我们批判性地检验了该领域最常用指标的有效性，例如BLEU、嵌入向量以及“LLM作为评判者”等方法。我们利用提出的风格判别基准对这些指标进行评估，该基准涵盖了三个领域中的八项多样化写作任务。

    arXiv:2508.06374v4 Announce Type: replace  Abstract: With the surge of large language models (LLMs) and their ability to produce customized output, style-personalized text generation--"write like me"--has become a rapidly growing area of interest. However, style personalization is highly specific, relative to every user, and depends strongly on the pragmatic context, which makes it uniquely challenging. Although prior research has introduced benchmarks and metrics for this area, they tend to be non-standardized and have known limitations (e.g., poor correlation with human subjects). LLMs have been found to not capture author-specific style well, it follows that the metrics themselves must be scrutinized carefully. In this work we critically examine the effectiveness of the most common metrics used in the field, such as BLEU, embeddings, and LLMs-as-judges. We evaluate these metrics using our proposed style discrimination benchmark, which spans eight diverse writing tasks across three e
    
[^299]: FaST：面向有限数据下个性化偏好对齐的特征感知采样与调优

    FaST: Feature-aware Sampling and Tuning for Personalized Preference Alignment with Limited Data

    [https://arxiv.org/abs/2508.04698](https://arxiv.org/abs/2508.04698)

    该论文提出了FaST方法，通过利用从数据中自动发现的高层特征实现高度参数高效的有限数据个性化偏好对齐，并引入DnD和ELIP两个数据集来支持PPALLI问题的研究。

    

    基于大语言模型（LLM）的对话助手通常以“一刀切”的方式部署，无法满足个体用户的偏好。近年来，大语言模型个性化——即调整模型以对齐特定用户偏好——作为弥合这一差距的方法受到了越来越多的关注。在本工作中，我们特别关注一个实用但具有挑战性的场景：每个用户只能收集少量偏好标注，我们将此问题定义为有限数据下的个性化偏好对齐（PPALLI）。为支持该领域的研究，我们引入了两个数据集——DnD和ELIP，并在其上对多种对齐技术进行了基准测试。我们进一步提出了FaST，一种高度参数高效的方法，它利用从数据中自动发现的高层特征，取得了最佳的整体性能。

    arXiv:2508.04698v2 Announce Type: replace  Abstract: LLM-powered conversational assistants are often deployed in a one-size-fits-all manner, which fails to accommodate individual user preferences. Recently, LLM personalization -- tailoring models to align with specific user preferences -- has gained increasing attention as a way to bridge this gap. In this work, we specifically focus on a practical yet challenging setting where only a small set of preference annotations can be collected per user -- a problem we define as Personalized Preference Alignment with Limited Data (PPALLI). To support research in this area, we introduce two datasets -- DnD and ELIP -- and benchmark a variety of alignment techniques on them. We further propose FaST, a highly parameter-efficient approach that leverages high-level features automatically discovered from the data, achieving the best overall performance.
    
[^300]: BOW：训练语言模型对合理的下一个词进行推理

    BOW: Training Language Models to Reason Over Plausible Next Words

    [https://arxiv.org/abs/2506.13502](https://arxiv.org/abs/2506.13502)

    BOW是一个强化学习框架，通过让模型生成自包含、中立且全面的合理下一个词空间描述，并由不接触语境的冻结评分器据此打分，从而训练语言模型对多个合理的下一个词进行推理，避免单一续写偏好导致的自我强化偏差。

    

    下一个词预测（NWP）在训练语言模型时以单一观测到的续写为目标，尽管许多语境下存在多个合理的下一个词。近期基于强化学习的下一个词推理方法将这一矛盾显性化：它们奖励模型生成支持某一语境条件下特定续写的理由，这可能使模型将原有的偏好固化为自信且自我辩护的推理轨迹。我们提出BOW，一个强化学习框架，它转而训练模型对合理下一个词的空间生成自包含、中立且全面的描述。策略基于完整语境生成下一个词的推理轨迹，但一个冻结的评分器仅根据该轨迹本身计算核心奖励，而不单独接收语境信息。BOW-Reg在此核心奖励的基础上增加了一个轻量级的广度正则化项，以防止过早坍缩。在两个模型骨干上，BOW与原始模型保持竞争力，并且常常……（摘要在此处截断）

    arXiv:2506.13502v4 Announce Type: replace  Abstract: Next-word prediction (NWP) trains language models against a single observed continuation, even though many contexts admit multiple plausible next words. Recent RL-based next-word reasoning methods make this tension explicit: they reward a model for producing a rationale that supports one context-conditioned continuation, which can turn a pre-existing preference into a confident, self-justifying trajectory. We introduce BOW, an RL framework that instead trains models to produce self-contained, neutral, and comprehensive descriptions of the plausible next-word space. The policy generates a next-word reasoning trajectory from the full context, but a frozen scorer computes the core reward from that trajectory alone, without separately receiving the context. BOW-Reg adds a lightweight breadth regularizer to this core reward to discourage premature collapse. On two model backbones, BOW remains competitive with the original models and often
    
[^301]: InComeS：将压缩与选择机制集成到大语言模型中以实现高效模型编辑

    InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing

    [https://arxiv.org/abs/2505.22156](https://arxiv.org/abs/2505.22156)

    该论文提出InComeS框架，通过将每个编辑上下文压缩为特殊摘要token的KV缓存并结合选择机制，突破大语言模型上下文窗口的限制，实现高效可扩展的上下文学习式模型编辑。

    

    尽管现有的模型编辑方法在回忆精确的编辑事实方面表现良好，但它们在需要更深层次语义理解而非单纯知识复述的复杂场景中往往表现不佳。利用大语言模型（LLM）强大的上下文推理能力，上下文学习成为一种有前景的编辑方法，它通过上下文编码来理解编辑信息。然而，这种方法受限于大语言模型有限的上下文窗口，随着编辑数量的增加，其性能和效率会不断下降。为了克服这一限制，我们提出了InComeS，这是一个灵活的框架，通过显式的压缩和选择机制来增强大语言模型处理编辑上下文的能力。具体而言，InComeS将每个编辑上下文压缩为一个特殊的摘要token的键值（KV）缓存，从而能够高效地处理多个编辑，而不受模型上下文窗口的限制。

    arXiv:2505.22156v4 Announce Type: replace  Abstract: Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model'
    
[^302]: 一个Token价值超过1000个Token：通过低秩克隆实现高效知识蒸馏

    A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone

    [https://arxiv.org/abs/2505.12781](https://arxiv.org/abs/2505.12781)

    提出低秩克隆（LRC）高效预训练方法，利用一组低秩投影矩阵同时实现教师权重的软剪枝和学生激活（含FFN信号）的克隆对齐，从而高效构建与强大教师模型行为等价的小语言模型。

    

    即使借助知识蒸馏和从更大教师模型进行剪枝，训练高性能的小语言模型（SLMs）仍然成本高昂。现有工作通常面临三个关键挑战：（1）硬剪枝导致的信息损失，（2）低效的表示对齐，以及（3）对信息丰富的激活（尤其是来自前馈网络FFN的激活）的利用不足。为解决这些挑战，我们提出了低秩克隆，这是一种高效的预训练方法，旨在构建与强大教师模型行为等价的小语言模型。LRC训练一组低秩投影矩阵，通过压缩教师权重实现软剪枝，并通过将学生的激活（包括FFN信号）与教师的激活对齐实现激活克隆。这种统一的设计在最大化知识迁移的同时，消除了对显式对齐模块的需求。基于开源教师模型（如Llam…）的大规模实验验证了该方法的有效性。

    arXiv:2505.12781v5 Announce Type: replace-cross  Abstract: Training high-performing Small Language Models (SLMs) remains costly, even with knowledge distillation and pruning from larger teacher models. Existing work often faces three key challenges: (1) information loss from hard pruning, (2) inefficient alignment of representations, and (3) underutilization of informative activations, particularly from Feed-Forward Networks (FFNs). To address these challenges, we introduce Low-Rank Clone (LRC), an efficient pre-training method that constructs SLMs aspiring to behavioral equivalence with strong teacher models. LRC trains a set of low-rank projection matrices that jointly enable soft pruning by compressing teacher weights, and activation clone by aligning student activations, including FFN signals, with those of the teacher. This unified design maximizes knowledge transfer while removing the need for explicit alignment modules. Extensive experiments with open-source teachers (e.g., Llam
    
[^303]: GuidedBench：测量并缓解野外LLM越狱方法评估中的差异

    GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods

    [https://arxiv.org/abs/2502.16903](https://arxiv.org/abs/2502.16903)

    该论文提出GuidedBench基准和集成逐案评估指南的GuidedEval评估系统，将LLM越狱方法评估中评估者间的差异降低至少76.03%，实现更准确、可靠和可复现的越狱有效性评估。

    

    尽管越狱攻击作为构建安全可靠大型语言模型（LLM）的有效红队测试工具日益受到关注，但有缺陷的评估系统设计导致了其有效性评估的重大差异。基于对2022年以来37项越狱研究的系统性测量研究，我们发现现有评估系统缺乏针对具体案例的评估标准，导致对其有效性和安全影响的结论产生误导。在本文中，我们提出了GuidedBench，这是一个新颖的基准，包含一个精心整理的有害问题数据集和GuidedEval——一个集成了详细逐案评估指南的评估系统。实验表明，GuidedBench能够对越狱性能提供更准确的评估，实现跨方法的有意义比较。GuidedEval将评估者间的差异降低了至少76.03%，确保了可靠且可复现的评估。

    arXiv:2502.16903v3 Announce Type: replace  Abstract: Despite the growing interest in jailbreaks as an effective red-teaming tool for building safe and responsible large language models (LLMs), flawed evaluation system designs have led to significant discrepancies in their effectiveness assessments. With a systematic measurement study based on 37 jailbreak studies since 2022, we find that existing evaluation systems lack case-specific criteria, resulting in misleading conclusions about their effectiveness and safety implications. In this paper, we introduce GuidedBench, a novel benchmark comprising a curated harmful question dataset and GuidedEval, an evaluation system integrated with detailed case-by-case evaluation guidelines. Experiments demonstrate that GuidedBench offers more accurate evaluations of jailbreak performance, enabling meaningful comparisons across methods. GuidedEval reduces inter-evaluator variance by at least 76.03%, ensuring reliable and reproducible evaluations. We
    
[^304]: 基于大语言模型的人格情境判断测试自动题目生成

    Automatic Item Generation for Personality Situational Judgment Tests with Large Language Models

    [https://arxiv.org/abs/2412.12144](https://arxiv.org/abs/2412.12144)

    本研究开发并评估了一个基于大语言模型（GPT-4和ChatGPT-5）自动生成人格情境判断测试题目的结构化、可推广框架，通过三项研究系统考察了提示词设计与温度设置对题目内容效度的影响，显著降低了传统SJT开发对专家的依赖。

    

    通过情境判断测试（SJT）进行人格评估，相较于传统的李克特式自我报告量表具有独特优势，但其开发仍然劳动密集、耗时，且严重依赖领域专家。大语言模型（LLM）的最新进展显示出在自动题目生成（AIG）方面的潜力。基于这些进展，本研究着重于开发并评估一个结构化、可推广的人格SJT自动生成框架，并以GPT-4和ChatGPT-5作为实证示例。本研究共开展了三项研究。研究1系统比较了提示词设计和温度设置对LLM生成题目内容效度的影响，以开发一种有效且稳定的基于LLM的人格SJT自动题目生成方法。结果表明，经过优化的提示词和1.0的温度设置在GPT-4上实现了创造力与准确性的最佳平衡（摘要在此处被截断）。

    arXiv:2412.12144v5 Announce Type: replace-cross  Abstract: Personality assessment through situational judgment tests (SJTs) offers unique advantages over traditional Likert-type self-report scales, yet their development remains labor-intensive, time-consuming, and heavily dependent on subject matter experts. Recent advances in large language models (LLMs) have shown promise for automatic item generation (AIG). Building on these developments, the present study focuses on developing and evaluating a structured and generalizable framework for automatically generating personality SJTs, using GPT-4 and ChatGPT-5 as empirical examples. Three studies were conducted. Study 1 systematically compared the effects of prompt design and temperature settings on the content validity of LLM-generated items to develop an effective and stable LLM-based AIG approach for personality SJT. Results showed that optimized prompts and a temperature of 1.0 achieved the best balance of creativity and accuracy on G
    

