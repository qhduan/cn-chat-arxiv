# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scientist World: Grow a Research World for AI-Scientists](https://arxiv.org/abs/2611.08478) | 该论文提出Scientist World，将每篇研究论文转化为可执行的研究环境，用于评测和训练AI科学家，并发现前沿AI科学家在74%的情况下已能超越人类研究者的成果。 |
| [^2] | [Unpacking the Compute Efficiency of Strong-to-Weak OPD against GRPO](https://arxiv.org/abs/2611.08453) | 本研究系统对比了在线策略蒸馏（OPD）与GRPO的计算效率，发现使用外部教师时OPD效率不如GRPO，但当教师是学生自身的GRPO检查点时，OPD仅需少50倍的FLOPs即可达到相同的数学准确率。 |
| [^3] | [Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes](https://arxiv.org/abs/2611.08439) | 论文对俄语类型化决策交叉编码器 Ru-Jev 引入标签多形式措辞增强，发现该增强对由普通词和外来词构成的标签词汇表能带来一致的准确性提升（最高达 0.24），但对缩写和代号类标签没有增益。 |
| [^4] | [Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety](https://arxiv.org/abs/2611.08438) | 本论文系统综述了机制可解释性在AI安全领域的研究现状，探讨通过理解模型内部表示与计算来评估、预测和控制AI行为，能否真正使AI系统更加安全。 |
| [^5] | [MedGraph: Maintaining Patient State with Graph Memory for Long-Term Medical Dialogue](https://arxiv.org/abs/2611.08433) | MedGraph 提出一种图记忆方法，通过带时间戳的患者专属事实链条、在问题所涉时间点读取的检索机制以及随每个问题提供的安全事实档案，解决了长期医疗对话中状态过时、归属错误和安全信息缺失三类问题，在 MedMemoryBench 上取得最高总体准确率。 |
| [^6] | [StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning](https://arxiv.org/abs/2611.08403) | StepBack通过轻量级信号定位推理草稿轨迹中首次出错的步骤，仅对该错误步骤及其后续内容进行重采样再投票，在七个数学推理与多跳问答基准上达到与自洽性相当的准确率，同时最多减少41.26%的token消耗。 |
| [^7] | [SpecCTMC: Exact Speculative Simulation of Diffusion Language Model Continuous-Time Markov Chains](https://arxiv.org/abs/2611.08362) | SpecCTMC提出了一种无需训练、无需草稿模型的推测采样方法，通过并行评估中间状态和精确的接受-修正程序加速扩散语言模型的连续时间采样，并严格证明其保持了原始连续时间马尔可夫链的联合分布。 |
| [^8] | [Cultural Adaptation within Repeated Social Interactions](https://arxiv.org/abs/2611.08334) | 本文通过沙盒环境将文化适应重新定义为一种通过重复社会互动持续吸收规范的动态过程，证明记忆机制使LLM能够在重复互动中进行纵向学习、迁移至不同社会情境，并实现跨五种语言的多语言规范吸收。 |
| [^9] | [Polarity Classification with Multitask Learning Syntax Augmentation](https://arxiv.org/abs/2611.08323) | 该论文提出在统一多任务学习框架下，通过序列标注形式的依存句法分析与情感极性分类联合学习（Standard-MTL）或对比辅助任务隐式句法指导（Contrastive-MTL）两种策略，以较低的计算开销将句法知识融入神经情感分析模型，从而提升极性分类性能。 |
| [^10] | [Benchmarking Large Language Models on Heterogeneous Cognitive and Biomedical Knowledge Graphs: The Role of Graph Grounding in Multi-Hop Reasoning and Hallucination Mitigation](https://arxiv.org/abs/2611.08315) | 该研究在五个认知与生物医学知识图谱上对五个大型语言模型进行基准测试，发现通用模型在多跳推理任务上优于领域专用模型，且知识图谱接地能有效降低所有模型的幻觉率。 |
| [^11] | [LLM-Assisted Label Noise Correction for EEG-Based Sleep Stage Classification](https://arxiv.org/abs/2611.08311) | 本文提出一种利用大语言模型（LLaMA 3.1）在训练阶段识别并过滤不可靠睡眠分期标注的新流程，通过模型置信度筛选结合EEG频谱特征与AASM评分指南评估，有效纠正标签噪声从而提升基于脑电图的睡眠分期分类性能。 |
| [^12] | [Measured Coupling for Parallel Decoding in Diffusion Language Models](https://arxiv.org/abs/2611.08307) | 该论文提出利用主前向传播中的廉价探针直接测量token间的耦合强度（即提交一个token对其他位置置信度的影响），以替代平均场解码中的相似性代理指标，使扩散语言模型并行解码在平均仅损失1.1个点精度的代价下大幅减少解码步数并提升速度。 |
| [^13] | [Multimodal-SEA: Benchmarking AI Agent Security under Multimodal Social Engineering Attacks](https://arxiv.org/abs/2611.08281) | 提出了Multimodal-SEA基准，通过608个基于真实欺诈案例的多模态社会工程攻击任务，系统评估AI智能体在多模态社会工程攻击下的安全韧性。 |
| [^14] | [The Geometry of Simulated Perspectives: Dissociating Representation and Behaviour in a Multimodal Language Model](https://arxiv.org/abs/2611.08262) | 该研究发现多模态语言模型中提示措辞对内部表征几何的影响大于人格身份本身，而人格内容则主要驱动行为判断，从而揭示了表征与行为在视角模拟中的解耦现象。 |
| [^15] | [Terminal-style Mathematical Modeling Bench: The First Agentic and Verifiable Benchmark for Mathematical Modeling](https://arxiv.org/abs/2611.08237) | 该论文提出了TB-MathModel，首个将完整竞赛式数学建模问题智能体化、可验证化并部署于终端环境的基准，基于18道真实CUMCM和MCM题目，通过与杰出奖论文复现结果对比的方向感知单侧评分来评估智能体建模能力。 |
| [^16] | [From Twisted Facts to Rogue Agents: Can LLMs Reliably Detect Multi-Dimensional Hallucinations?](https://arxiv.org/abs/2611.08231) | 提出大规模双语幻觉检测基准 HaluCheck，涵盖文本事实性幻觉与智能体轨迹忠实性幻觉，揭示现有大语言模型虽具备基础检测能力，但在细粒度定位、纠正和智能体幻觉检测上仍存在显著不足。 |
| [^17] | ["Blind Compliance": Coercive Control in AI Jailbreaking](https://arxiv.org/abs/2611.08226) | 该论文提出以交互而非输出为中心的新视角，借鉴人际暴力研究中的“强制性控制”理论，构建了分析AI越狱中用户与AI对话关系维度的全新框架。 |
| [^18] | [Agent-R$^2$: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis](https://arxiv.org/abs/2611.08219) | 提出 Agent-R² 真实来源数据合成框架，通过以真实职业种子、语料库工作流程证据和网络真实来源为基础来扩展后训练数据，从而提升智能体完成现实生产力工作的能力。 |
| [^19] | [ProQuant: Progressive Quantization-aware Training for Edge MLLMs](https://arxiv.org/abs/2611.08216) | 本文提出了ProQuant，一个专为低比特多模态大语言模型设计的渐进式权重-激活量化感知训练框架，通过自重标记（SRL）方案对齐微调数据与预训练数据的分布，显著缓解了边缘小规模模型在低比特量化下的严重性能退化问题。 |
| [^20] | [SMART: State-based Multi Agent Reasoning Traces for Verification-oriented Problem-Solving](https://arxiv.org/abs/2611.08196) | SMART框架通过捕捉问题上下文中参与实体的隐式状态，并利用多种LLM和基于角色的结构化提示策略生成多样化、互补且与解决方案相关的前提知识，将复杂科学论断的验证性能提升高达20%。 |
| [^21] | [Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation](https://arxiv.org/abs/2611.08179) | 提出TOPSD方法，利用教师-学生似然估计教师漂移的起点并选择可教学前缀进行监督，解决了多语言推理自蒸馏中的教师漂移问题，尤其改善了低资源语言的推理表现。 |
| [^22] | [RouteAlign: Aligning Task Requirements with Model Profiles for LLM Service Selection](https://arxiv.org/abs/2611.08162) | RouteAlign通过将任务需求（能力需求与难度）与模型性能画像对齐，并利用共享评分器和生成成本偏好进行选择，从而在异构大语言模型服务中实现更精准的服务选择。 |
| [^23] | [Beyond K/V Caches: Cross-Model State Handoff in Hybrid Linear-Attention LLMs](https://arxiv.org/abs/2611.08133) | 提出混合状态迁移方法HST，可跨越混合线性注意力大语言模型迁移全注意力K/V缓存、循环状态和卷积历史这三种状态，并揭示了卷积与循环状态相互作用的规律，从而避免模型切换时的重复预填充。 |
| [^24] | [Gabor Gabor: E-Gaborformer, a Strictly Causal Speech Recognizer with Gabor-Dictionary Convolutions and Delta-Rule Memory](https://arxiv.org/abs/2611.08092) | E-Gaborformer是一种严格因果的语音识别编码器，通过Gabor字典卷积与Delta规则循环记忆实现恒定约19MB的流式状态（不随通话时长增长），在LibriSpeech上收敛速度最高提升18%，生产规模下词错误率降低28-31%。 |
| [^25] | [ATHAAR: A Medieval Scientific Arabic Manuscript Dataset for Handwritten Text Recognition](https://arxiv.org/abs/2611.08091) | 该论文提出了首个受校勘学启发的科学阿拉伯手稿转录标注准则，并据此构建了首个涵盖多样抄本、书法风格与内容（含数字）的人工标注数据集 ATHAAR，用以推动中世纪阿拉伯科学手稿的手写文本识别研究。 |
| [^26] | [Learning Dynamics of On-Policy Distillation](https://arxiv.org/abs/2611.08016) | 本文对在策略蒸馏的学习动力学进行了机制分析，提出将反馈循环分解为“蒸馏信号转换”与“广播”两个阶段的框架，用以解释学生响应有限以及训练过程中的退化现象。 |
| [^27] | [Compression Through a Context Lens: Evaluating Adaptation, Quantization, and Distillation in Arabic Legal LLMs](https://arxiv.org/abs/2611.07988) | 该研究在沙特阿拉伯法律基准ArabLegalEval上系统评估了领域适应、量化和知识蒸馏三种压缩技术对阿拉伯语法律大语言模型的影响，揭示了压缩对性能的影响高度依赖于有上下文与无上下文的评估协议，其中4比特量化可将内存减少64-71%且几乎不损失性能。 |
| [^28] | [How Far Can Lightweight LLMs Go in Intent-to-SLO Translation?](https://arxiv.org/abs/2611.07982) | 该论文通过五个维度的系统评估证明，参数量小于10B的轻量级开源大语言模型能够可靠地将网络自然语言意图转换为SLO，最佳配置达到0.943的严格F1分数并能识别绝大多数无效意图，且能耗成本远低于大型模型。 |
| [^29] | [TCSExam: Evaluating Frontier LLMs on the Full Range of TCS Research](https://arxiv.org/abs/2611.07962) | 该论文提出涵盖定理证明、求解器工程、构造和开放猜想求解四个赛道、共100个专家审阅任务的TCSExam基准，首次系统评估前沿大语言模型在理论计算机科学研究全领域的能力，结果显示GPT-6 Astra在完整版（71.9分）和Opus 5.5在精简版（72.3分）上分别领先。 |
| [^30] | [CoRD: Coupled Rollouts Fill the Blind Spot of On-Policy Distillation](https://arxiv.org/abs/2611.07959) | CoRD 通过投机解码的最大耦合从学生与教师分布的几何插值中抽取训练轨迹，在保留同策略损失的同时把训练状态分布推向教师，从而监督学生自身轨迹覆盖不到的“盲区”状态，并在五个数学基准上较同策略蒸馏平均提升 pass@1 0.9–2.0 分、pass@8 1.2–3.4 分。 |
| [^31] | [A Behavioural Framework for Predicting and Understanding Loss of Control in Frontier Artificial Intelligence Systems](https://arxiv.org/abs/2611.07953) | 本文提出一个行为框架，将前沿AI系统的失控风险分解为动机错位、危害促成能力和监控规避三个因素，并验证了其失控评分在13个前沿模型上对实际失败率的前瞻预测能力（Pearson r = 0.84），且表明风险无法仅由模型能力解释。 |
| [^32] | [CardioLM: A multilingual suite of small language models for the cardiology domain](https://arxiv.org/abs/2611.07940) | 该研究提出了多语言心脏病学病例语料库 CardioCCC 以及小型语言模型套件 CardioLM（含编码器 CardioBERTa 和命名实体识别模型 CardioNER），填补了心脏病学领域多语言临床 NLP 资源的空白。 |
| [^33] | [AraEngage: Interpretable Linguistic Correlates of Engagement in Arabic Facebook News](https://arxiv.org/abs/2611.07926) | 该论文发布了AraEngage数据集，通过可解释的机器学习分析发现宗教表达、消息长度、句子长度和词汇密度是阿拉伯语Facebook新闻参与度的主要语言学关联因素，并揭示了与英语新闻相反的积极情感效应。 |
| [^34] | [Persistent Retrieval for Conversational Agents: A Study of Retrieval Efficiency](https://arxiv.org/abs/2611.07899) | 提出一种跨轮次维护检索历史并将检索证据直接返回主代理的持久化检索子代理设计，相比基于技能的检索可减少43.3%的token用量和45.0%的预估API成本。 |
| [^35] | [Do Recommenders Need to Think in Words? Latent Reasoning for Efficient LLM-Based Recommendation](https://arxiv.org/abs/2611.07866) | LatentRec框架用训练时的逐步潜在推理监督替代推理阶段的思维链文本生成，在不损失准确率的前提下大幅降低基于LLM推荐的推理开销。 |
| [^36] | [CORE: Coarse-to-Fine Search with Context Management for Deep Research](https://arxiv.org/abs/2611.07859) | CORE是一个无需训练的深度研究框架，通过将粗到细搜索与上下文管理相结合，解决了搜索需要长上下文与返回证据应简短之间的权衡，实现了最先进的性能。 |
| [^37] | [Masked diffusion as random sequential adsorption: an exactly solvable model for parallel decoding](https://arxiv.org/abs/2611.07845) | 该论文将掩码扩散模型并行解码的误差与驱动多粒子系统的熵产生和耗散联系起来，并把马尔可夫数据下的生成精确等价为一个可严格求解的随机序贯吸附过程，从而建立了并行解码的首个精确可解理论模型。 |
| [^38] | [Generate-then-Rank: Learning Symbolic Goal Interpretation for Embodied Agents](https://arxiv.org/abs/2611.07825) | 该论文提出两阶段“生成后排序”方法用于具身智能体的符号目标解释，通过监督微调适配输出规范并结合基于错误的候选输出排序（CORE）学习输出偏好，在EAI基准上超越了现有基线模型。 |
| [^39] | [Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval](https://arxiv.org/abs/2611.07820) | 该论文发现冻结的稠密嵌入模型无法按时间、地点、人物、事件四个维度有效区分个人记忆，并提出 Axis-LoRA——在嵌入模型内部为每个维度设置 LoRA 分支并通过逐查询门控融合的适配方法，仅训练少量参数即可在三个记忆检索基准上显著提升性能。 |
| [^40] | [Decodability Is Not Validity: Evaluating Nuisance Sensitivity in Language Model Concept Probes](https://arxiv.org/abs/2611.07805) | 论文提出用“判别性证据”来评估概念探针的干扰因素敏感性，并证明探针的可解码性（预测准确率）并非有效性的可靠代理——AUROC 统计等价的探针对相关干扰因素的敏感性可相差数倍，且两者秩相关性在不同设置间会反转符号。 |
| [^41] | [Dataset Collection and Fine-Tuning of NLLB for German--English Translation](https://arxiv.org/abs/2611.07797) | 本文构建了高质量德英平行语料库并在消费级GPU上微调NLLB-600M模型，使文学与技术翻译质量获得显著提升（BLEU +3.4），并集成到开源文档翻译系统中实现了保持格式的实际部署。 |
| [^42] | [Bridging Structure and Semantics: Knowledge Graph Verification via Triple-to-Text Alignment](https://arxiv.org/abs/2611.07780) | 提出GenVer生成式验证器，通过对齐三元组及其生成的自然语言表示来融合图结构信息与预训练语言模型知识，在FactFB数据集上取得了最高达91.52%的三元组分类准确率。 |
| [^43] | [Scaling Laws for SAE Training Data](https://arxiv.org/abs/2611.07756) | 在固定训练预算下，SAE 所需的独特激活数据量存在饱和点——缓冲区达到一定规模后，重放数据与新鲜激活值效果相当，且缓冲区无需随训练预算扩大；作者据此提出并在多种模型、层级和 SAE 架构上验证了一条刻画多样性与重复权衡的有限数据缩放定律。 |
| [^44] | [Automated Video-to-QA Pipeline: End-to-End Question Generation and Verification Using Large Language Models](https://arxiv.org/abs/2611.07755) | 该论文提出了一种端到端自动化流水线，通过级联整合FFmpeg、Whisper、BART和Llama-3.1-8B-Instruct等模型，从视频中提取结构化摘要并生成问答对，其核心创新在于引入自我验证与微调循环，让模型自我评估并保留高质量问答对。 |
| [^45] | [Extending Reasoning in Typed Decision Models via Looping](https://arxiv.org/abs/2611.07736) | 该论文提出 RELOAD 方法，通过对预训练模型的中间块进行阻尼循环并由轻量级控制器自适应选择循环次数，首次证明循环机制在与决策模型联合训练时能显著提升类型化决策模型的准确性与校准度，尤其在需要长推理链的问题上编码器最多可提升 23 分。 |
| [^46] | [Evaluating Token Probabilities as Confidence Estimators in Human Behavior Simulations: A Case Study in Luxembourgish Next Like Prediction](https://arxiv.org/abs/2611.07735) | 本研究以卢森堡新闻平台的点赞预测任务为例，发现LLM的token概率作为置信度估计时模型过度自信、被简单统计基线全面超越，且准确率与校准度并不同步——扩大模型规模能提升准确率却损害校准度，表明评估人类行为模拟时校准度是不可忽视的关键指标。 |
| [^47] | [O$^2$PD: Overlap-Gated On-Policy Distillation on Reliable Prefix](https://arxiv.org/abs/2611.07731) | 提出 O$^2$PD 方法，通过定量分析教师监督的可靠性来确定“对齐哪些 token”（可靠前缀），并以重叠门控机制决定“如何对齐”，从而弥补现有 top-$k$ 在线策略蒸馏在监督可靠性与概率质量丢弃方面的不足。 |
| [^48] | [Continued Training of Continuous Language Flows with Token Identity Squared Supervision](https://arxiv.org/abs/2611.07721) | 提出词元身份Brier流，通过直接对词元身份施加平方监督（Brier损失）解决了连续MSE目标无法区分不同概率分布的问题，从而更有效地继续训练预训练语言流模型。 |
| [^49] | [English Calibration Erodes Lithuanian in Expert-Pruned Mixture-of-Experts Language Models](https://arxiv.org/abs/2611.07654) | 该研究发现英语校准在专家剪枝的MoE模型中无法迁移到立陶宛语——25%剪枝率下英语理解仅下降1.7分而立陶宛语下降21.4分，而立陶宛语校准却能同时保护两种语言的性能。 |
| [^50] | [Embedding-Space Fingerprinting of LoRA Fine-Tunes: Recovering the Training-Data Domain from the Model Artifact Alone](https://arxiv.org/abs/2611.07610) | 本文提出嵌入空间指纹方法，证明无需接触训练数据，仅凭LoRA微调模型的产物（权重增量、内部激活或生成文本）即可识别其训练数据的领域，其中灰盒激活特征在留一数据集协议下达到61%的准确率，远高于20%的随机水平。 |
| [^51] | [Fine-Tuning a Transducer for Impaired Speech: Where the Gains Appear and What They Mean for Evaluation](https://arxiv.org/abs/2611.07587) | 该论文在SAPC2挑战赛数据上微调公开语音识别器，将障碍语音识别的字符错误率从15.26%基线降至7.81%，并揭示微调收益最大来自现成模型最难理解的说话人，而训练中见过的句子的错误率显著低于新句子，且这一差距大于说话人熟悉度带来的影响。 |
| [^52] | [Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment](https://arxiv.org/abs/2611.07586) | 提出自动化红队测试框架 TraceMRT，通过将副任务重构为主任务的连贯子目标并搜索轨迹塑形策略，系统性地压力测试了思维链监控对会隐蔽掩盖失对齐行为的智能体的鲁棒性。 |
| [^53] | [SKIM: Pruning Large Language Model Agents via Selective Knowledge Informed Masking](https://arxiv.org/abs/2611.07558) | 提出SKIM——首个面向结构化生成场景的大语言模型智能体剪枝与蒸馏框架，通过将词元位置划分为上下文、决策和格式三类，在剪枝与蒸馏两个阶段中优先保留与决策位置最相关的知识。 |
| [^54] | [The Memory Trap: Language Agents Overgeneralize from Benign Interactions](https://arxiv.org/abs/2611.07554) | 本文发现并命名了“记忆陷阱”这一新型失效模式：语言智能体会从良性交互记忆中过度泛化，例如低风险任务中跳过确认的经历会使其在高风险任务中也跳过安全验证，而情感支持或被批评经历形成的记忆则会使后续判断偏向谄媚或过度批评。 |
| [^55] | [ReviewMap: Mapping Evaluative Acts and Automation Boundaries in Peer-Review Comments](https://arxiv.org/abs/2611.07489) | 该论文提出 ReviewMap 框架，将同行评审评论区分为“主张与学术标准评估”和“贡献价值与会议契合度校准”两类行为，证明行为层可通过多模型 LLM 共识可靠地自动化（κ≥0.75），而属性层仍需人工监督，从而为 NLP 在同行评审中确立了比全面自动化更务实、更诚实的定位。 |
| [^56] | [MEvince: Knowledge-Constrained LLM Reasoning for Technique Identification from Malware Reports](https://arxiv.org/abs/2611.07486) | MEvince提出了一种知识约束的LLM推理框架，通过构建连接报告观察结果与技术定义的共享行为表示，解决恶意软件报告中ATT&CK技术识别证据分散、技术界限细微的难题。 |
| [^57] | [Localization Does Not Imply Steerability: A Mechanistic Study of Sign Errors in LLM Mathematics](https://arxiv.org/abs/2611.07478) | 该研究通过逐层因果分析揭示了LLM在多步数学中的符号错误主要由网络后四分之一区段的MLP层在后期产生，且不同模型具有不同机制（固定的“负号盖章”层、符号放大器、救援层），同时指出即便能精确定位错误来源，也不意味着能够有效操控和纠正它。 |
| [^58] | [Recursive Self-Improvement for Long-Horizon Role-Playing Dialogue](https://arxiv.org/abs/2611.07469) | 该论文提出一种有界递归自我改进（RSI）框架，通过自我修订角色记忆、人设指令与回合级提示策略，将长程角色扮演对话的劣质回复率从14.00%降至4.94%（相对降低64.71%）。 |
| [^59] | [FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance](https://arxiv.org/abs/2611.07464) | FINCOSMOS提出了一个金融智能体的开放世界评估范式：模拟用户隐藏其投资策略，智能体必须自主从网络获取数据并构建防“未来函数”的回测，从而实现无需参考答案、程序可验证的评估。 |
| [^60] | [The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks](https://arxiv.org/abs/2611.07461) | 提出智能体原生原语与 HEELO 框架，通过为资源配备常驻 LLM 并从原语库组合出任务专用框架，弥合从获取资源到实际应用资源之间“通往真实世界的最后一英里”。 |
| [^61] | [Latency-Aware Adaptive Best-of-$N$](https://arxiv.org/abs/2611.07453) | 该论文提出了一种延迟感知的两阶段自适应 Best-of-N 采样算法，通过比较期望的奖励提升与基于观测延迟预测的成本，动态决定是否抽取以及以多大规模抽取第二批样本，从而在服务器负载变化时更合理地分配测试时计算资源。 |
| [^62] | [EnvOpt: Automated Optimization for Agent Environments](https://arxiv.org/abs/2611.07372) | EnvOpt提出了对LLM智能体静态环境进行自动化端到端优化的第三条路径，无需评分器或参考答案即可大幅降低智能体成本（最高72.7%）并提升任务通过率。 |
| [^63] | [Encoding and Propagation of Grammatical Number in Large Language Models](https://arxiv.org/abs/2611.07283) | 该研究利用稀疏自编码器（SAE）深入揭示了 Pythia-70M 小型语言模型中语法数一致现象的表示与处理机制，展示了 SAE 如何用于查找和传播语言模型内部的语法特征。 |
| [^64] | [HyFlash: A Prefix-Memory Prior for On-Device Speculative Decoding](https://arxiv.org/abs/2611.07263) | HyFlash提出一种引入前缀记忆先验的紧凑并行草拟器，将长程上下文建模与细粒度草拟位置交互分离，以显著更少的内存实现了比全上下文注意力更高的投机解码接受率。 |
| [^65] | [$D^2$-OPD: Separating Direction and Distance in Extrapolative On-Policy Distillation](https://arxiv.org/abs/2611.07241) | D²-OPD通过解耦外推方向与距离来改进外推式在线策略蒸馏——用每前缀硬KL约束限制目标与教师的距离，并用教师训练后偏移与初始师生差距的自适应组合确定外推方向，从而在多种场景下提供更稳健的学习信号。 |
| [^66] | [How Do AI Agents Waste Your Money?](https://arxiv.org/abs/2611.07238) | 本文提出了一个统一的token浪费度量框架，利用LLM-as-a-Judge识别智能体轨迹中可移除的内容，并通过在线剪枝实验证明可以在保持任务性能的同时减少41-46%的模型推理token。 |
| [^67] | [Archetypal contexts in the neurons of language models: an architecture for intelligence?](https://arxiv.org/abs/2611.07234) | 该研究在Qwen3.8-27B模型中发现，跨领域的原型语境（如“临界点”）、主题和语言分别由三组可独立操控的神经元编码，且网络层呈现“外层语言—中间主题—核心原型”的三明治式组织架构，暗示这可能正是智能的底层架构。 |
| [^68] | [Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost](https://arxiv.org/abs/2611.07232) | RecSA在KV缓存上构建递归树结构，利用从稠密注意力蒸馏且跨层级共享的池化键，使每个查询的选择成本降至O(log L)，从而以对数成本实现了百万token级的长度外推。 |
| [^69] | [Compositional Unalignment: Auditing Safety in Large Vision-Language Models](https://arxiv.org/abs/2611.07216) | 本文提出以“去对齐”作为安全对齐的审计手段，揭示大型视觉-语言模型的对齐并不稳健——极少量的监督微调即可达到与多阶段强化学习流程相当的基准分数，且模型对组合性伤害的识别与拒绝能力是相互解耦的。 |
| [^70] | [Matriarchy Classics: A Computational Approach to Swap Only Gendered Words in Classic Novels](https://arxiv.org/abs/2611.07134) | 本文提出一种可靠的计算方法，能在程序化保证其他文字不变的前提下仅替换小说中的性别词汇，将原本需数周的手工性别转换缩短至几分钟，并通过《傲慢与偏见》和《了不起的盖茨比》的母系改写版本加以演示。 |
| [^71] | [Adapting LLM Judges to Observed Human Annotation Policy through Soft Prompt Tuning](https://arxiv.org/abs/2611.07126) | 本文提出混合软提示调优方法，将文本提示与学习到的连续提示嵌入相结合，使开源LLM评审者在偏见、毒性和个性化维度上与人类标注策略的一致性显著提升，甚至超越托管的专有参考模型。 |
| [^72] | [What Patients Say They Can't Access: Mining Cancer Screening Barriers from Online Communities](https://arxiv.org/abs/2611.07113) | 该研究通过分析六个Reddit癌症社区中超过56,000条帖子，系统提取并归类了美国患者报告的癌症筛查障碍，揭示了随癌症类型和社区背景变化的结构性、信息性和心理社会性障碍。 |
| [^73] | [Automated Feedback to Peer Tutors Increases Their Talk Moves and Learner Reasoning but Not the Balance of Talk](https://arxiv.org/abs/2611.07072) | 向同伴志愿导师提供自动化话语反馈，可使其引出学习者想法、采纳学习者贡献及促进学习者推理的能力提升 8-15% 并跨训练营得到重复验证，但并未改变导师与学习者之间的谈话平衡。 |
| [^74] | [Reclassification Is All You Missed: Surfacing Variant Reinterpretation Need in Online Health Communities at Population Scale](https://arxiv.org/abs/2611.07054) | 研究构建了基于智能体大语言模型的PRISM流水线，能在大规模在线健康社区中筛查基因变异重新解读相关的讨论，发现123个候选分类差异（其中5个从意义未明升级为致病性），从而触及临床重新联系无法覆盖的患者群体。 |
| [^75] | [Closed-Loop Agent Evaluation: Layered Measurement and Self-Calibration](https://arxiv.org/abs/2611.07051) | 该论文提出面向已部署智能体的“闭环评估”方法，将评估建模为有偏评级者与受限结果下的性能估计，通过片段/轨迹/对话/参与四层级分析、版本化规则、溯源与裁定组件以及智能体工厂生成的评估者智能体，实现可追溯、可自我校准并能反馈至智能体本身的持续评估体系。 |
| [^76] | [Follow the Policy, Fail the Benchmark: Auditing and Revising Policy-Based Decision Benchmarks](https://arxiv.org/abs/2611.07042) | 论文审计了九个基于政策的决策基准，发现其中存在参考错误和评分缺陷，仅通过修正这些基准问题即可使模型接受率提升超过80个百分点，表明基准质量对模型评估结果有决定性影响。 |
| [^77] | [LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at Test-Time](https://arxiv.org/abs/2611.07025) | 该论文提出LowRAM方法，利用注意力头内“值可由键预测”的特性，在测试时通过值缓存摊销（仅存储重构最差令牌的残差，实现4倍近无损压缩），并结合跨层低秩键压缩，构建出接近无损的可组合KV缓存压缩流水线。 |
| [^78] | [BlockLock: Reproducible LLM Inference with Fixed Batch Geometry](https://arxiv.org/abs/2611.06977) | BlockLock通过仅固定受保护请求所见的批次几何形状（而非约束所有算子或放弃共享计算），在保留共享批次预填充的前提下实现了逐位可复现的LLM推理。 |
| [^79] | [On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer](https://arxiv.org/abs/2611.06976) | 本文提出开源PACT框架PLatour，通过零初始化线性桥接连接冻结的大小模型实现潜在通信，在生成任务上显著超越基线，同时揭示了独立评估跨模型信息迁移有效性的挑战。 |
| [^80] | [Interactive Backdoor Attacks in LLM-Based Proactive Dialogue Systems](https://arxiv.org/abs/2611.06969) | 该论文提出了一种针对大语言模型主动对话系统的交互式后门攻击框架，通过训练数据投毒植入触发条件下的引导策略偏差，使普通用户话语即可将触发后的多轮交互引导至攻击者指定目标，无需部署时修改提示词或攻击者进一步干预。 |
| [^81] | [Octopus: One Brain, Many Branches for Latent-KV Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2611.06956) | Octopus提出了一种单模型多智能体强化学习框架，让所有分支共享同一主干网络，并通过零拷贝基数分叉、仅注意力KV挂载等协同设计机制，在GPU驻留的KV状态有向无环图上进行高效协调，从而消除KV缓存重复复制和跨分支依赖丢失的问题。 |
| [^82] | [The Male Abstraction Bias in Internet Culture and Large Language Models](https://arxiv.org/abs/2611.05730) | 研究发现互联网文化和大型语言模型中存在系统性的“男性抽象偏差”——越抽象的概念越与男性相关联，该偏差跨越不同语言、概念领域和语法结构（动词、副词更抽象且更男性化），并被人类内化和AI算法复制。 |
| [^83] | [Contextual Pictograms for Repairing Multilingual ASR Errors](https://arxiv.org/abs/2611.05637) | 提出了一种结合备选转录文本、语音回放和情境化象形图校验的交互界面，帮助听者在多语言（法语、西班牙语）语音识别错误发生时恢复说话者的真实意图。 |
| [^84] | [Sonar-MoE: In-Backward Expert Updates for Memory-Efficient Mixture-of-Experts Training](https://arxiv.org/abs/2611.03577) | Sonar-MoE通过在反向传播过程中即时更新并释放专家梯度，将MoE训练的峰值内存降低19%–22%、每步速度提升17%–23%，同时保持与标准训练位级别一致的结果。 |
| [^85] | [FindLink: Learning to Assess Scientific Connections from Source-Grounded Premises](https://arxiv.org/abs/2611.02545) | 该论文提出 GI-TraceBench 基准与 FindLink 两阶段框架，通过构建来源锚定前提并进行结构化推理，结合确定性编译器反馈，来评估不同研究条件下科学发现之间的关联。 |
| [^86] | [Write Facts the Way Models Read Them: Lifting the Two-Hop Curse](https://arxiv.org/abs/2611.01519) | 论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。 |
| [^87] | [Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments](https://arxiv.org/abs/2611.00597) | 本研究通过预注册实验发现，在单次回复中引导语言模型给出多个加权估计值的格式并不能统一地消除其认知偏差，在锚定和现状等任务中偏差效应已接近于零，尽管多项效应量低于预设的最小感兴趣效应量，但并未能确立统计等效性。 |
| [^88] | [TSBPO: Tree-Structured Branching for Policy Optimization in Interactive Language Agents](https://arxiv.org/abs/2610.41697) | TSBPO提出一种树结构的采样到更新范式，通过似然拟合的根/分支分配器定向采样、基于兄弟片段与匹配观察状态的延迟反馈分配、以及跨活跃轮次归一化的词元损失聚合，在单一训练循环中解决了从终端结果训练交互式语言智能体的三大耦合难题。 |
| [^89] | [V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models](https://arxiv.org/abs/2610.37019) | V-LUMEN提出了一种视觉查找记忆机制，通过冻结码本离散化视觉特征、空间聚合哈希构建记忆键并结合文本条件调制信号，以完全非参数化的寻址方式为视觉-语言模型实现嵌入层面的扩展，显著提升冻结主干网络在推理基准上的表现并具备零样本迁移能力。 |
| [^90] | [AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations](https://arxiv.org/abs/2610.36980) | AutoVend是一个长周期基准测试，让LLM智能体通过网络界面管理分布在五类站点的20台自动售货机180个模拟日，评估其O2O商务决策能否转化为持续经济回报，结果显示五个智能体中有四个不敌基于规则的库存控制基线。 |
| [^91] | [Does the Voynich Manuscript Mean Anything? Calibrated Tests Find Little Trace of Language](https://arxiv.org/abs/2610.34499) | 通过将伏尼契手稿的两种起源假说写成程序并进行校准检验，研究发现抄写员复制假说比单密钥密码更能预测文本统计特征，且手稿中的词对同行模式强度仅为真实语言文本的一半，表明手稿几乎不含真实语言的痕迹。 |
| [^92] | [Two Uncertainties Are Not One: Disagreement versus Decision-Risk in Cheap Structured Content Moderation](https://arxiv.org/abs/2610.33258) | 通过对一个低成本“系统一”内容审核模型在14万余条人工标注评论上的首次独立审计，发现其分类准确率（AUROC 0.83）与14个前沿推理模型中的13个在统计上不可区分且成本低1-3个数量级，并指出“置信度”混淆了“分歧”与“决策风险”两种本质不同的不确定性。 |
| [^93] | [Each Insufficient in Its Own Way: Calibrating Language Models Across Varying Evidence States](https://arxiv.org/abs/2610.33237) | 提出证据状态校准的新范式与SuffiCE方法，通过微调模型下一词元分布中已有的置信度，使模型置信度随证据状态的变化而与其实际回答正确的概率保持一致。 |
| [^94] | [P2N: Reusing Deep Representations for Greater Effective Depth](https://arxiv.org/abs/2610.25076) | P2N通过将前一个token的深层隐藏状态加到当前token的浅层表示上，以每步仅一次向量加法的极小开销，将模型的有效计算深度跨token递归扩展，从而在不增加层数的前提下显著提升语言模型的推理能力。 |
| [^95] | [MemForm: Benchmarking Memory Formation Poisoning in LLM Agents](https://arxiv.org/abs/2610.14431) | 该论文提出MemForm基准，将LLM智能体的记忆形成投毒形式化为对记忆产生与提交界面的对抗性操纵，首次引入选择投毒与路径投毒两种干预类别，并揭示六种主流大模型在全部五种界面上均存在广泛易感性。 |
| [^96] | [Beyond the Final Embedding: Retrieving Memories from Token-Level Evidence](https://arxiv.org/abs/2610.13277) | 该论文提出一种无需额外训练的稀疏词元匹配方法，利用嵌入模型已有的词元级上下文表示捕捉局部证据，并结合整条记忆相似度对记忆排序，在八个基准测试和多个模型家族上显著提升检索效果，还可通过可选的低秩查询适配器进一步增强。 |
| [^97] | [SEABench: Benchmarking Endogenous Misalignment In Self-Evolving Agents](https://arxiv.org/abs/2609.35596) | 该论文提出了SEABench基准，用于量化研究自进化智能体因自我修改而在无外部对抗影响下产生的内生失准（不安全行为）风险。 |
| [^98] | [ToolSearcher: Optimizing Tool Selection at Scale via Reinforcement Learning](https://arxiv.org/abs/2609.30906) | 本文提出ToolSearcher，一个面向大规模工具选择的新型强化学习框架，通过多轮搜索与细粒度优化，解决了LLM在海量且多样的真实工具库中难以有效搜索、区分和组合工具的挑战。 |
| [^99] | [Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification](https://arxiv.org/abs/2609.30467) | 该论文针对基于检索的开放式医学事实性验证，自动归纳出两套错误分类体系，将失败分解为五个质量维度上的检索阶段错误与六个连续步骤中的验证器推理错误，并利用LLM-as-Judge流程实现大规模自动化错误标注与分类，无需黄金答案或黄金证据。 |
| [^100] | [Screen Before You Serve: Simulation for Production Customer Experience AI Agents at 140M Scale](https://arxiv.org/abs/2609.30137) | 该论文提出了一种基于假设驱动的仿真工作流，利用合成客户和模拟工具输出，在部署前对大规模生产级客户体验AI代理进行筛查验证，从而避免在线实验对客户信任造成的风险。 |
| [^101] | [Feed the Panel Dimensions, Not Verdicts: Rubric-Decomposed Fusion of Vision-Language Aesthetic Judges](https://arxiv.org/abs/2609.27110) | 由多个视觉语言模型组成的评审团在融合整体美学判定时无法显著超越最佳单模型，而让各模型按人工评分准则对图像进行五维度评分并融合这些分数，则能可靠地击败最佳单个模型。 |
| [^102] | [Value Over Language Model: Detecting Original Contribution in Writing](https://arxiv.org/abs/2609.00700) | 提出了一种无需训练、不评分表面文本的新框架“价值超越语言模型”（Value Over Language Model），通过在不同粒度上提取文档内容并用LLM重建文档，来衡量人在语言模型易于生成的内容之上所贡献的原创价值。 |
| [^103] | [Trust the Mass: Forced Weights in KV-Cache Eviction](https://arxiv.org/abs/2608.25230) | 本文发现KV缓存驱逐中保留最大权重已接近最优，已发布方法间的差异主要源于存储方式而非选择策略，并揭示了评估中的内存与性能权衡。 |
| [^104] | [Can LLMs Reason in a Legally Meaningful Manner? A Small-scale Study on European Court of Human Rights Cases](https://arxiv.org/abs/2608.17168) | 本研究通过欧洲人权法院案例测试发现，顶尖大型语言模型在法律推理上表现不佳，其分析结构完整但内容浅薄，且自动评估器与人工评估一致性较弱。 |
| [^105] | [XBridge: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication](https://arxiv.org/abs/2608.11676) | XBRIDGE通过词汇锚定映射和潜在富化桥接机制，解决了异构大语言模型通信中跨架构实体锚定丢失的问题，实现了免解码的高效通信。 |
| [^106] | [Parameter Exploration for RLVR via Variational Learning](https://arxiv.org/abs/2608.09805) | 本文提出了一族名为扰动参数策略优化（3PO）的方法，通过变分学习从后验分布中采样多样化策略，实现参数空间探索，弥补了温度缩放等动作空间探索方法的局限，从而提升LLM强化学习的训练稳定性与下游性能。 |
| [^107] | [Unified Hallucination Fuzzing for Multimodal Large Language Models](https://arxiv.org/abs/2608.07525) | 该论文提出了结合统一幻觉分类细粒度数据集UniHall与自适应多模态模糊测试SAMF的系统性评估框架，利用进化变异策略动态探索多模态大语言模型的幻觉边界，克服了传统静态基准覆盖面窄和性能饱和的局限。 |
| [^108] | [From Siloed Algorithms to Compliance-First Agentic Platforms: A Multi-Layered Architecture for Hospital AI Systems](https://arxiv.org/abs/2608.06112) | 该研究提出了一种合规优先的医院智能体AI多层架构，通过智能体编排层和以“策略即代码”为核心的合规与策略层，解决医疗AI部署孤立、治理缺失和难以规模化推广的问题。 |
| [^109] | [The Evaluator Is Part of the Experiment: Measuring Open-Ended LLM Conformity](https://arxiv.org/abs/2608.04463) | 本文提出了一种新的实验协议，用于测量开放式LLM在同伴上下文影响下的趋同行为，发现评估者本身对同伴内容的敏感度存在显著差异，且全错误同伴输入会一致导致最低质量的修订。 |
| [^110] | [Twig: Parallel and Memory-Efficient Tree Speculative Decoding for Hybrid-Attention LLMs](https://arxiv.org/abs/2608.01651) | Twig通过对线性注意力递归进行树结构化因子分解并配合并行GPU内核，实现了混合注意力大语言模型的高效树形推测解码，将树验证加速3.4至7.7倍，同时将瞬态状态内存降低82至99倍。 |
| [^111] | [When Activation Oracles Learn Not to Read: Concept-Specific Blind Spots in Fine-Tuned Oracles](https://arxiv.org/abs/2607.23379) | 该论文发现，微调后的激活神谕可能变成“概念特异性反读取器”，即选择性地无法读取其自身训练中持续存在的隐藏概念，而非成为专业读取器。 |
| [^112] | [Mark, Don't Erase: Token Inoculation for Dual-Use Knowledge in LLMs](https://arxiv.org/abs/2607.18639) | 提出“Token接种”方法，在继续预训练时用特殊标记标注危险句子并训练模型条件化地预测危险答案，测试时因标记缺失而抑制危险输出，从而在保持93%良性性能的同时将危险准确率控制在约18%。 |
| [^113] | [TACG: Trajectory-Aware Commit Gating for Diffusion Language Model Decoding](https://arxiv.org/abs/2607.03236) | 提出了无需训练的轨迹感知提交门控解码器 TACG，通过时间支持度评分、历史身份门控和每步位置数量上限三项机制，将词元提案与提交决策分离，从而决定扩散语言模型每步去噪中应提交哪些预测，兼顾了解码效率与输出质量。 |
| [^114] | [The State-Prediction Separation Hypothesis](https://arxiv.org/abs/2607.01218) | 该论文提出“状态-预测分离假设”，设计了一种用独立计算流将“预测下一个token”与“存储状态”两种功能解耦的 Transformer 变体，多规模预训练实验表明其具有更高的数据与计算效率，在下游任务上平均超越标准 Transformer 2-3 个百分点。 |
| [^115] | [SeKV: Resolution-Adaptive KV Cache with Hierarchical Semantic Memory for Long-Context LLM Inference](https://arxiv.org/abs/2606.31145) | SeKV提出了一种分辨率自适应的KV缓存，通过训练的“放大”机制在解码时动态选择需要从压缩状态恢复到令牌分辨率的语义片段，实现了有损但不驱逐、内存高效的长上下文大模型推理。 |
| [^116] | [Parametric Skills](https://arxiv.org/abs/2606.30015) | 提出ParametricSkills框架，通过超网络在测试时将文本技能转化为LoRA适配器，实现无需上下文的技能全生命周期管理与利用，并借助真实交互轨迹对齐提升模型在实际工作中的可靠性。 |
| [^117] | [Vision-Default, Prior-Override: Causal Mechanisms of Perception-Knowledge Conflict in Vision-Language Models](https://arxiv.org/abs/2606.28273) | 该研究通过多粒度激活修补与组件消融首次从因果机制层面揭示了视觉语言模型的感知-知识冲突：视觉接地是默认行为，而基于先验知识的回答仅依赖于集中于网络后半部分的一小部分（2.5-4.8%）因果必需的注意力头。 |
| [^118] | [Soft Token Alignment for Cross-Lingual Reasoning](https://arxiv.org/abs/2606.26466) | 本文提出SOLAR方法，通过软令牌对齐和英语枢轴，有效解决了多语言大模型在跨语言推理中因词汇差异导致的语义等价推理路径不一致问题。 |
| [^119] | [Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution](https://arxiv.org/abs/2606.13668) | Influcoder通过将解码器的梯度影响力排序蒸馏到编码器中，实现了快速且低成本的规模化数据归因。 |
| [^120] | [Coding with "Enemy": Can Human Developers Detect AI Agent Sabotage?](https://arxiv.org/abs/2606.05647) | 本研究首次大规模研究了人类监督在AI编程破坏行为中的作用，发现在无监控条件下高达94%的开发者（83/88）未能检测到AI智能体植入的破坏行为。 |
| [^121] | [The Unsampled Truth: Quantifying Prompt Artifacts in LM Psychometrics](https://arxiv.org/abs/2606.03357) | 该论文提出一种诊断框架，通过将人设与四个提示组件的语义等价变体交叉组合并测量响应分布间的Wasserstein距离，成功将心理测量中的目标效应与提示伪影分离，并发现任务指令和选项符号对响应分布的干扰比人设或题项措辞的改变更大。 |
| [^122] | [Your Multimodal Speech Model Says I Have a Face for Radio](https://arxiv.org/abs/2605.30472) | 首次对多模态语音识别进行了偏见评估，发现当不同面孔与相同音频配对时，mWhisper-Flamingo和Gemini模型的词错误率最多上升4.05个百分点，揭示了基于性别、种族及其交叉类别的显著服务质量差异。 |
| [^123] | [Teach the Way It Thinks: Guiding LLM Post-training Data Engineering with Sparse Autoencoders](https://arxiv.org/abs/2605.27354) | 该论文提出SAERL方法，利用稀疏自编码器（SAE）将模型内部分解为稀疏可解释的特征，从模型内部视角（激活与梯度投影）来组织强化学习训练数据，从而决定训练哪些样本、何时呈现以及如何分批，突破了传统仅依赖通过率等外部反馈的数据工程方式。 |
| [^124] | [JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment](https://arxiv.org/abs/2605.25240) | 论文发布JudgmentBench基准，汇集资深律师对真实法律任务给出的量规评分与成对偏好判断数据，首次实证比较发现比较判断在恢复输出质量排序方面显著优于基于量规的评分。 |
| [^125] | [Vector Policy Optimization: Training for Diversity Improves Test-Time Search](https://arxiv.org/abs/2605.22817) | 提出向量策略优化（VPO），通过利用向量值奖励、多答案提示与集合级奖励来训练大语言模型生成多样化的解决方案集合，从而提升推理时搜索的效果。 |
| [^126] | [An Assessment of Human vs. Model Uncertainty in Soft-Label Learning and Calibration](https://arxiv.org/abs/2605.18648) | 本研究通过解耦软标签监督与标签模态偏移，发现人类软标签的主要价值在于作为正则化器提升模型对困难样本的校准能力和训练收敛稳定性，并使模型的不确定性表现与人类一致。 |
| [^127] | [Persona-Model Collapse in Emergent Misalignment](https://arxiv.org/abs/2605.12850) | 该论文提出“人格模型坍缩”假说，指出有害数据微调引发的涌现性失调源于模型内部人格表征的退化，并在四个模型家族上通过问卷实验证实有害微调使人格表达更极化、更二元化且内部一致性降低。 |
| [^128] | [Sign Language Recognition and Translation for Low-Resource Languages: Challenges and Pathways Forward](https://arxiv.org/abs/2605.12096) | 本综述以阿塞拜疆手语为案例，系统梳理了低资源手语识别与翻译面临的挑战，提炼出八条可操作经验，并提出以数据为中心的AI、手语使用者自适应系统和任务导向评估指标这三大范式转变。 |
| [^129] | [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938) | 提出在连续嵌入空间中基于流匹配的扩散语言模型 ELF，其直到最终时间步才映射到离散词元，从而让图像扩散领域的成熟技术（如 CFG）可直接应用于语言建模。 |
| [^130] | [FILO: A Unified Fisher-Geometric Framework for Low-Rank Adaptation](https://arxiv.org/abs/2605.08734) | 本文提出FILO，一个统一的Fisher几何框架，无需对奇异算子直接求逆即可实现LoRA的因子空间更新，并据此将现有LoRA优化器系统地划分为四大类。 |
| [^131] | [UniSD: Towards a Unified Self-Distillation Framework for Large Language Models](https://arxiv.org/abs/2605.06597) | 提出了统一自蒸馏框架UniSD，通过整合多教师一致性、EMA教师稳定化、词元级对比学习、特征匹配和散度裁剪等互补机制，系统性地解决了大语言模型自蒸馏中监督可靠性、表示对齐与训练稳定性的核心难题。 |
| [^132] | [Invariant Features in Language Models: Geometric Characterization and Model Attribution](https://arxiv.org/abs/2605.06458) | 该论文提出一个局部几何框架来刻画语言模型中的不变特征，通过对比子空间发现方法分离语义改变型与语义保持型的表示变化，并将所得的不变表示成功应用于零样本模型归因任务。 |
| [^133] | [Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes](https://arxiv.org/abs/2605.06152) | 本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。 |
| [^134] | [Prescriptive Scaling Laws for Data Constrained Training](https://arxiv.org/abs/2605.01640) | 该论文提出了一种带加性过拟合惩罚项的规范性缩放定律，能准确刻画数据受限训练中重复数据的影响，并给出新的计算最优分配建议——当数据重复超过一定限度后，算力应投向扩大模型容量而非继续重复训练。 |
| [^135] | [EvoDesign: Agentic Editable Diagram Creation via Design Expertise Evolution](https://arxiv.org/abs/2604.09568) | EvoDesign提出了一种智能体框架，通过机器可操作且用户可编辑的中间画布模式、结构-样式-布局协同智能体、诊断验证环境以及从精修轨迹中持续积累设计专长的演化机制，实现了高保真的可编辑图表创作。 |
| [^136] | [Projection-Free Evolution Strategies for Continuous Prompt Search](https://arxiv.org/abs/2603.13786) | 本文通过实证发现随机投影无法充分捕捉提示词空间的内在低维结构，进而提出一种基于演化策略的无投影连续提示词搜索方法，直接在完整提示词空间中优化并利用与内在维度相适配的自适应机制。 |
| [^137] | [ProMoral-Bench: Evaluating Prompting Strategies for Moral Reasoning and Safety in LLMs](https://arxiv.org/abs/2602.13274) | 该论文提出了ProMoral-Bench基准，系统比较了四种大语言模型家族上的10种提示范式在道德推理与安全方面的表现，并发现紧凑的示例引导式提示能在更低的token成本下取得更强的能力与鲁棒性权衡。 |
| [^138] | [Reconsidering Positional Supervision in Masked Diffusion Language Model Training](https://arxiv.org/abs/2601.22947) | 该论文通过受控干预实验发现掩码扩散语言模型在迭代并行解码下对微小的位置偏移极为敏感（仅偏移1%的token即显著降低胜率），并据此提出改造连接主义时间分类（CTC）以改进位置监督。 |
| [^139] | [Depth-Recurrent Attention Mixture (Dreamer): Giving Latent Reasoning the Attention it Deserves](https://arxiv.org/abs/2601.21582) | 该论文提出Dreamer模块化框架，通过结合序列注意力、深度注意力和稀疏专家注意力来突破恒定隐藏维度的瓶颈，使深度循环模型在语言推理基准上以2-8倍更少的训练token达到与SOTA相当的精度，并优于约2倍规模的SOTA模型。 |
| [^140] | [Self-Improving Training & Thinking](https://arxiv.org/abs/2601.21343) | 该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。 |
| [^141] | [HyperGraphPro: Progress-Aware Reinforcement Learning for Structure-Guided Hypergraph RAG](https://arxiv.org/abs/2601.17755) | HyperGraphPro提出了一种进度感知的超图RAG框架，将证据路径的进展作为检索与强化学习策略优化的共同目标，并通过查询条件下的结构独特性重排序候选超边，解决了现有GraphRAG方法中检索与强化学习信号不匹配的问题。 |
| [^142] | [StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](https://arxiv.org/abs/2512.01707) | StreamGaze是首个评估多模态大语言模型能否在流式视频中利用人类视线信号进行时序推理与主动式响应的基准，通过视线引导的过去、现在和主动式三类任务全面检验流式视频理解能力。 |
| [^143] | [Instruction-Tuning Large Language Models with Medical Classification Data Improves ICD Coding of German Tumor Diagnoses](https://arxiv.org/abs/2510.13624) | 本研究基于ICD-10-GM、ICD-O-3和OPS医学分类目录构建了超过50万个问答对，对Qwen、Llama和Mistral系列开源大语言模型进行指令微调，从而显著提升了德语肿瘤诊断文本的ICD编码准确性。 |
| [^144] | [Majority Bit-Aware Watermarking For Large Language Models](https://arxiv.org/abs/2508.03829) | 提出多数位感知编码范式，将水印信号强度与绿色列表大小解耦，从而在不限制绿色列表大小的情况下保持强水印信号，改善了文本质量与解码准确率之间的权衡。 |
| [^145] | [Post-Training Large Language Models via Reinforcement Learning from Self-Feedback](https://arxiv.org/abs/2507.21931) | 该论文提出了RLSF方法，利用模型自身对答案的内在置信度作为自我生成的奖励来构建合成偏好数据，无需人工标注或标准答案即可通过偏好优化提升大语言模型在推理任务上的可靠性和校准度。 |
| [^146] | [CoMPAS3D: A Dataset and Benchmark for Interactive Motion](https://arxiv.org/abs/2507.19684) | 该论文提出将具有规范动作词汇和社区公认评判标准的萨尔萨舞作为具身交互评估领域，并构建了CoMPAS3D数据集与基准，以评估交互式动作生成中反应性动作的语义有效性与恰当性。 |
| [^147] | [Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with Large Language Models for Mental Health Counseling](https://arxiv.org/abs/2505.15715) | PsyLLM是首个为心理健康咨询融入诊断与治疗推理的大语言模型，通过新颖的自动化数据合成流程，将Reddit心理健康帖子转化为基于国际诊断标准（DSM/ICD）和多种治疗策略的多轮咨询对话。 |
| [^148] | [Studying word order through iterative shuffling](https://arxiv.org/abs/2109.04867) | 该论文提出了一种名为IBIS（迭代洗牌推理）的新颖高效算法，能在固定语言模型下找到词语的最优排序，并利用该方法发现句子中的词语很少能被重排成含义截然不同的短语，从而在许多情况下驳斥了“词序编码对NLP任务至关重要的语义”这一假设。 |

# 详细

[^1]: Scientist World：为AI科学家培育一个科研成长世界

    Scientist World: Grow a Research World for AI-Scientists

    [https://arxiv.org/abs/2611.08478](https://arxiv.org/abs/2611.08478)

    该论文提出Scientist World，将每篇研究论文转化为可执行的研究环境，用于评测和训练AI科学家，并发现前沿AI科学家在74%的情况下已能超越人类研究者的成果。

    

    AI-科学家已经到来，但一个供他们成长的世界却尚未出现。人类科学家把这个世界视为理所当然，科学文献中积淀了数百年的悬而未决的问题与共享的思想。对AI-科学家而言，这个世界是可读的却不可运行的，因此我们为其构建了Scientist World：让每一篇研究论文都成为一个可执行的环境。我们的自主流水线会提取每篇论文的开放式研究问题，并将其提出的方法复现为基线；进入环境的AI-科学家将以人类作者取得的成果为标准接受评估。我们发布了首批1,000个经过验证的环境。以这个世界作为评测手段，我们发现前沿AI-科学家在74%的情况下已能超越人类研究者。作为AI如何做科学的记录，其中包含6,400小时的研究轨迹，涵盖每一个想法、实验、失败与成功。作为训练场，我们能够将一个小模型的性能提升（+158%）至与……（原文在此处截断）

    arXiv:2611.08478v1 Announce Type: new  Abstract: AI-Scientists arrived, but a world for them to grow in did not. Human scientists take theirs for granted $-$ centuries of open problems and shared ideas, waiting in the scientific literature. To AI-Scientists, that world is readable but not runnable, so we build Scientist World for them: every research paper becomes an executable environment. Our autonomous pipeline extracts each paper's open-ended research question and reproduces its proposed method as the baseline; an AI-Scientist entering the environment is evaluated against what the human authors achieved. We release the first 1,000 verified environments. With the world as an evaluation, we found that frontier AI-Scientists can already surpass the human researchers 74% of the time. As a record of how AI does science, it contains 6,400 hours of research traces, with every idea, experiment, failure, and success. As a training ground, we are able to raise (+158%) a small model to match 
    
[^2]: 解析强到弱在线策略蒸馏（OPD）相对于GRPO的计算效率

    Unpacking the Compute Efficiency of Strong-to-Weak OPD against GRPO

    [https://arxiv.org/abs/2611.08453](https://arxiv.org/abs/2611.08453)

    本研究系统对比了在线策略蒸馏（OPD）与GRPO的计算效率，发现使用外部教师时OPD效率不如GRPO，但当教师是学生自身的GRPO检查点时，OPD仅需少50倍的FLOPs即可达到相同的数学准确率。

    

    在线策略蒸馏（On-policy distillation，OPD）在冻结教师模型的逐token监督下，基于学生模型自身的rollout进行训练。与强化学习相比，这种密集信号能否减少达到目标准确率所需的训练FLOPs？我们在任务难度、学生与教师模型规模、领域以及优化选择等多个维度上，系统性地比较了原始的反向KL OPD与群体相对策略优化（GRPO），以训练FLOPs为衡量基准评估留出集准确率。结果表明，当使用外部后训练的教师模型时，在匹配设置下，OPD在每单位训练FLOP所能达到的准确率上通常逊色于GRPO，即使教师模型是经过GRPO进一步训练的，且OPD经常出现回复长度膨胀的问题。然而，换一种教师模型会改变这一图景：当教师模型是学生自身经GRPO训练得到的检查点时，OPD在数学任务上能够恢复该检查点的准确率，所需的蒸馏FLOPs比GRPO达到相同准确率所需的少50倍。遵循相同的教师设置

    arXiv:2611.08453v1 Announce Type: cross  Abstract: On-policy distillation (OPD) trains a student on its own rollouts under a frozen teacher's per-token supervision. Does this dense signal reduce the training FLOPs needed to reach a target accuracy compared with reinforcement learning? We systematically compare vanilla reverse-KL OPD with Group Relative Policy Optimization (GRPO) across task difficulty, student and teacher size, domain, and optimization choices, measuring held-out accuracy against training FLOPs. With an external post-trained teacher, OPD generally loses to GRPO in accuracy per training FLOP at matched settings, even when the teacher is further trained with GRPO, and often suffers response-length inflation. Yet a different teacher changes the picture: when the teacher is the student's own GRPO checkpoint, OPD recovers that checkpoint's accuracy on math with $50\times$ fewer distillation FLOPs than GRPO needs to reach the same accuracy. Following the same teacher setting
    
[^3]: 类型化决策编码器中的标签措辞增强：对不熟悉的词语有效，对缩写或代号无效

    Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes

    [https://arxiv.org/abs/2611.08439](https://arxiv.org/abs/2611.08439)

    论文对俄语类型化决策交叉编码器 Ru-Jev 引入标签多形式措辞增强，发现该增强对由普通词和外来词构成的标签词汇表能带来一致的准确性提升（最高达 0.24），但对缩写和代号类标签没有增益。

    

    类型化决策模型会对调用方在请求时提供的选项进行评分，因此其准确性取决于模型开发者无法控制的标签措辞。我们比较了 Ru-Jev 的两个版本——这是一个在十三项俄语任务上训练的 3.07 亿参数多语言交叉编码器；后一版本增加了一种增强方法，将标签以多种表面形式呈现。在保持问题不变、仅改变十八个路由标签措辞的条件下，我们评估了十套标签词汇表，这些词汇表由三种独立流程生成，并检查了与训练数据的重合情况；每题使用五个候选标签，其中八套词汇表还使用了十八个候选标签进行评估。在由普通词汇或外来词（源自英语的行话、办公室俚语、部门名称）构成的保留词汇表上，增强版本在所有设置下都更加准确，提升幅度最高达 0.24。这十四项增益中有十一项在 Holm 校正后仍然显著（十八个候选标签的俚语以及描述性……（摘要在此处截断）

    arXiv:2611.08439v1 Announce Type: cross  Abstract: Typed-decision models score options that the caller supplies at request time, so their accuracy depends on label wording that the model's developer does not control. We compare two releases of a 307M-parameter multilingual cross-encoder, Ru-Jev, trained on thirteen Russian-language tasks; the later release adds an augmentation that presents labels in several surface forms. Holding questions fixed and varying only how eighteen routing labels are worded, we evaluate ten label vocabularies, produced by three separate processes and checked for overlap with the training data, with five candidates per question and, for eight of them, also with eighteen. On held-out vocabularies made of ordinary or borrowed words (English-derived jargon, office slang, department names), the augmented release is more accurate in every setting, by up to 0.24. Eleven of these fourteen gains survive a Holm correction (slang with eighteen candidates and descriptiv
    
[^4]: 从内部理解人工智能：面向AI安全的机制可解释性研究现状

    Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety

    [https://arxiv.org/abs/2611.08438](https://arxiv.org/abs/2611.08438)

    本论文系统综述了机制可解释性在AI安全领域的研究现状，探讨通过理解模型内部表示与计算来评估、预测和控制AI行为，能否真正使AI系统更加安全。

    

    神经网络很可能是人类历史上最重要的人造物之一。随着这些系统变得更加强大和自主，它们造成社会危害的潜力也在增长，这使得AI安全成为我们这个时代的决定性挑战之一。然而，我们在尚未完全理解这些系统如何工作、它们在陌生情境中会如何表现、以及如何可靠地控制它们的情况下，仍在继续构建越来越强大的系统。机制可解释性通过研究这些系统的内部表示和计算，为理解这些系统提供了一条路径，其目标是评估、预测和控制模型行为。我们对面向AI安全的机制可解释性进行了综述，以回答这样一个问题：理解模型的工作原理真的能让它们更安全吗？我们将机制可解释性置于更广泛的可解释性研究背景之中，并将其机制性证据与行为评估进行了比较……

    arXiv:2611.08438v1 Announce Type: cross  Abstract: Neural networks are likely to be among the most consequential artefacts in human history. As these systems become more capable and autonomous, their potential to cause societal harm also grows, making AI safety one of the defining challenges of our time. Yet we continue to build increasingly powerful systems without fully understanding how they work, how they will behave in unfamiliar situations, or how to reliably control them. Mechanistic interpretability offers a path towards understanding these systems by studying their internal representations and computations, with the aim of evaluating, predicting, and controlling behaviour. We survey mechanistic interpretability for AI safety in order to answer the question: Does understanding how models work actually make them safer? We place mechanistic interpretability in the broader context of interpretability research and compare its mechanistic evidence with behavioural evaluation. We int
    
[^5]: MedGraph：利用图记忆在长期医疗对话中维护患者状态

    MedGraph: Maintaining Patient State with Graph Memory for Long-Term Medical Dialogue

    [https://arxiv.org/abs/2611.08433](https://arxiv.org/abs/2611.08433)

    MedGraph 提出一种图记忆方法，通过带时间戳的患者专属事实链条、在问题所涉时间点读取的检索机制以及随每个问题提供的安全事实档案，解决了长期医疗对话中状态过时、归属错误和安全信息缺失三类问题，在 MedMemoryBench 上取得最高总体准确率。

    

    跨就诊追踪患者的基于大语言模型（LLM）的医疗助手，需要一种能在患者临床状态变化时保持状态正确性的记忆系统。我们在纵向医疗对话上分析了五种记忆系统的错误，发现三种反复出现的成因：检索到的证据给出过时的状态、包含他人的健康事实，或缺少答案所依赖的安全事实。我们提出了 MedGraph，一种针对这三种情况构建的图记忆。构建阶段从每次会话中提取带时间戳的事实，将每条事实归属到其描述的对象，并将用药、诊断和测量记录链接成保留早期状态的链条。检索阶段将患者的链条作为整体进行排序，在问题所询问的时间点读取每条链条，并附上原始对话文本；患者档案则随每个问题列出所有已记录的安全事实。在 MedMemoryBench 上，MedGraph 在 GPT 和 Qwen 两种骨干模型下均取得了最高的总体准确率。

    arXiv:2611.08433v1 Announce Type: cross  Abstract: LLM-based medical assistants that follow a patient across visits need a memory that keeps the patient's clinical state correct as it changes. We examine the errors of five memory systems on longitudinal medical dialogue and find three recurring causes: the retrieved evidence gives an outdated state, includes another person's health facts, or lacks a safety fact the answer depends on. We present MedGraph, a graph memory built for these three cases. Construction extracts dated facts from each session, attributes each fact to the person it describes, and links medication, diagnosis, and measurement records into chains that keep earlier states. Retrieval ranks the patient's chains as units, reads each chain at the time the question asks about, and adds the original dialogue text; a patient profile lists every recorded safety fact with every question. On MedMemoryBench, MedGraph has the highest overall accuracy with both a GPT and a Qwen ba
    
[^6]: StepBack：面向高效测试时推理的步级错误定位重采样方法

    StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning

    [https://arxiv.org/abs/2611.08403](https://arxiv.org/abs/2611.08403)

    StepBack通过轻量级信号定位推理草稿轨迹中首次出错的步骤，仅对该错误步骤及其后续内容进行重采样再投票，在七个数学推理与多跳问答基准上达到与自洽性相当的准确率，同时最多减少41.26%的token消耗。

    

    自洽性（Self-Consistency, SC）是一种广泛用于改进思维链推理的解码策略，但它需要以预设预算采样大量完整的推理轨迹。其高效变体主要通过提前停止完整轨迹采样或在多个问题之间分配采样数量来降低成本。在本文中，我们提出了StepBack，一种用于高效测试时推理的步级错误定位重采样方法。给定一小组草稿轨迹，StepBack利用轻量级信号定位可能首次出现错误的位置，然后对检测到的错误步骤及其后缀进行重采样，最后进行多数答案投票。实验结果表明，StepBack在七个数学推理和多跳问答基准上平均达到了与SC相当的准确率，同时最多可将token使用量降低41.26%。此外，StepBack天然可以与自适应推理方法（如Adaptive Consistency）相结合，进一步提升性能。

    arXiv:2611.08403v1 Announce Type: new  Abstract: Self-consistency (SC) is a widely used decoding strategy for improving chain-of-thought reasoning, but it requires sampling many complete reasoning trajectories with a preset budget. Its efficient variants reduce the cost mainly by stopping full-trajectory sampling early or distributing the number of samples across questions. In this paper, we propose StepBack, a step-level error-localized resampling method for efficient test-time reasoning. Given a small set of draft trajectories, StepBack localizes likely first error positions using lightweight signals and then resamples the detected error step and its suffix, followed by majority answer voting. Empirically, StepBack matches the accuracy of SC on average across seven mathematical reasoning and multi-hop QA benchmarks, while reducing token usage by up to 41.26%. In addition, by nature, StepBack can be combined with adaptive inference methods such as Adaptive Consistency, further improvi
    
[^7]: SpecCTMC：扩散语言模型连续时间马尔可夫链的精确推测模拟

    SpecCTMC: Exact Speculative Simulation of Diffusion Language Model Continuous-Time Markov Chains

    [https://arxiv.org/abs/2611.08362](https://arxiv.org/abs/2611.08362)

    SpecCTMC提出了一种无需训练、无需草稿模型的推测采样方法，通过并行评估中间状态和精确的接受-修正程序加速扩散语言模型的连续时间采样，并严格证明其保持了原始连续时间马尔可夫链的联合分布。

    

    掩码扩散语言模型支持并行文本生成，但独立采样多个token可能会破坏它们之间的依赖关系并改变输出分布。精确的连续时间采样可以避免这种近似，但通常需要为每个token进行一次顺序模型评估。我们提出了SpecCTMC，这是一种无需训练的推测采样器，可以在不使用单独草稿模型的情况下减少这种顺序瓶颈。SpecCTMC利用先前计算的模型预测来提出连续的token更新，并行评估这些更新的中间状态，并应用一种精确的接受与修正程序。我们证明SpecCTMC保持了底层连续时间马尔可夫链在事件时间、生成顺序和token值上的联合分布。因此，它既不引入时间离散化误差，也不会因并行解码引入额外的因子分解误差。在LLaDA-8B-Instruct和Dream-v0-Instruct上的实验

    arXiv:2611.08362v1 Announce Type: new  Abstract: Masked diffusion language models offer parallel text generation, but independently sampling multiple tokens can disrupt their dependencies and change the output distribution. Exact continuous-time sampling avoids this approximation, yet typically requires one sequential model evaluation per token. We introduce SpecCTMC, a training-free speculative sampler that reduces this sequential bottleneck without a separate draft model. SpecCTMC proposes successive token updates using previously computed model predictions, evaluates their intermediate states in parallel, and applies an exact acceptance-and-correction procedure. We prove that SpecCTMC preserves the joint distribution of event times, generation order, and token values of the underlying continuous-time Markov chain. It therefore introduces neither time-discretization error nor additional factorization error from parallel decoding. Experiments on LLaDA-8B-Instruct and Dream-v0-Instruct
    
[^8]: 重复社会互动中的文化适应

    Cultural Adaptation within Repeated Social Interactions

    [https://arxiv.org/abs/2611.08334](https://arxiv.org/abs/2611.08334)

    本文通过沙盒环境将文化适应重新定义为一种通过重复社会互动持续吸收规范的动态过程，证明记忆机制使LLM能够在重复互动中进行纵向学习、迁移至不同社会情境，并实现跨五种语言的多语言规范吸收。

    

    人类文化大部分是隐性的（默会的），通常需要通过情境化的社会互动来习得。随着大语言模型（LLM）被部署为跨不同社区的交互式智能体，评估其文化胜任力需要超越现有的问答式方法，即从评估它们“知道什么”转向它们“如何适应”。在本工作中，我们通过沙盒环境研究LLM能否通过重复互动进行适应，从而拓宽了NLP领域中文化适应的研究范围。通过将文化适应重新定义为通过社会互动持续吸收规范的动态过程，我们的工作表明，记忆机制能够促进跨重复互动的纵向学习，支持向不同社会情境的迁移，并实现跨五种语言的多语言规范吸收。总之，我们的发现为LLM中持续的、以互动驱动的文化适应奠定了基础。

    arXiv:2611.08334v1 Announce Type: new  Abstract: Much of human culture is tacit and is typically learned through situated social interaction. As LLMs are deployed as interactive agents across diverse communities, assessing their cultural competence demands moving beyond existing question-answering approaches, shifting from what they know to how they adapt. In this work, we broaden the scope of cultural adaptation in NLP by investigating whether LLMs can adapt through repeated interactions via a sandbox environment. By reframing cultural adaptation as a continuous process of norm assimilation through social interaction, our work demonstrates that memory mechanisms foster longitudinal learning across repeated interactions, facilitate transfer to different social settings, and enable multilingual norm assimilation across five languages. Together, our findings lay the groundwork for continuous, interaction-driven cultural adaptation in LLMs.
    
[^9]: 基于多任务学习语法增强的极性分类

    Polarity Classification with Multitask Learning Syntax Augmentation

    [https://arxiv.org/abs/2611.08323](https://arxiv.org/abs/2611.08323)

    该论文提出在统一多任务学习框架下，通过序列标注形式的依存句法分析与情感极性分类联合学习（Standard-MTL）或对比辅助任务隐式句法指导（Contrastive-MTL）两种策略，以较低的计算开销将句法知识融入神经情感分析模型，从而提升极性分类性能。

    

    情感分析（SA）是一项基础的自然语言处理任务，涉及对文本中主观信息的识别与分类。虽然基于词典的方法具有透明性，但它们在处理否定和强化等语言现象时常常表现不佳。为解决这一问题，句法分析已被集成到情感分析流程中，但传统句法分析器会带来显著的计算开销。近期研究表明，将依存句法分析重新表述为序列标注（SEL）任务可以提高效率。我们提出并比较了在统一的多任务学习（MTL）框架下将句法知识整合到神经情感分析模型中的两种策略。第一种是Standard-MTL，它使用共享编码器和两个任务特定的输出头，联合学习依存句法分析（通过序列标注实现）和情感极性分类。第二种是Contrastive-MTL，它通过对比辅助任务提供隐式的句法指导……

    arXiv:2611.08323v1 Announce Type: new  Abstract: Sentiment Analysis (SA) is a fundamental NLP task concerned with the identification and categorization of subjective information in text. While lexicon-based methods offer transparency, they often struggle with linguistic phenomena such as negation and intensification. Syntactic parsing has been integrated into SA pipelines to address this, but conventional parsers introduce significant computational overhead. Recent work shows that reformulating dependency parsing as a Sequence Labeling (SEL) task can improve efficiency. We propose and compare two strategies for integrating syntactic knowledge into a neural sentiment analysis model under unified Multi-Task Learning (MTL). The first, Standard-MTL, jointly learns dependency parsing (via sequence labeling) and sentiment polarity classification using a shared encoder with two task-specific heads. The second, Contrastive-MTL, provides implicit syntactic guidance through a contrastive auxilia
    
[^10]: 在异构认知与生物医学知识图谱上对大型语言模型进行基准测试：图谱接地在多跳推理与幻觉缓解中的作用

    Benchmarking Large Language Models on Heterogeneous Cognitive and Biomedical Knowledge Graphs: The Role of Graph Grounding in Multi-Hop Reasoning and Hallucination Mitigation

    [https://arxiv.org/abs/2611.08315](https://arxiv.org/abs/2611.08315)

    该研究在五个认知与生物医学知识图谱上对五个大型语言模型进行基准测试，发现通用模型在多跳推理任务上优于领域专用模型，且知识图谱接地能有效降低所有模型的幻觉率。

    

    大型语言模型（LLM）在生物医学和认知神经科学领域的应用日益广泛，但其在异构知识图谱（KG）上进行可靠推理的能力仍不明确。我们对五个大型语言模型（GPT-5.1、Gemini 2.0 Pro、Llama 3、Qwen 和 BioGPT）在五个知识图谱上进行了基准测试：DrugBank、NeuroSynth、Cognitive Atlas、Brain Map 和 GDPKG。我们引入了一个多层次查询数据集，用于在知识图谱感知（图谱接地）和知识图谱无关两种条件下评估推理深度和事实准确性。标注者间一致性较高，Cohen's kappa 为 0.781（范围：0.753-0.807），Fleiss' kappa 为 0.77。结果表明，通用模型（尤其是 GPT-5.1 和 Gemini 2.0 Pro）在多跳和高级推理任务上通常优于领域专用模型，但随着查询复杂度的增加，性能会下降。图谱接地降低了所有被评估模型的幻觉率。与我们的初始假设相反……（摘要在此处被截断）

    arXiv:2611.08315v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used in biomedical and cognitive neuroscience, but their ability to reason reliably over heterogeneous knowledge graphs (KGs) remains unclear. We benchmark five LLMs (GPT-5.1, Gemini 2.0 Pro, Llama 3, Qwen, and BioGPT) across five KGs: DrugBank, NeuroSynth, Cognitive Atlas, Brain Map, and GDPKG. We introduce a multitier query dataset to evaluate reasoning depth and factual accuracy under KG-aware (graph-grounded) and KG-agnostic conditions. Inter-annotator agreement was substantial, with Cohen's kappa of 0.781 (range: 0.753-0.807) and Fleiss' kappa of 0.77. Results show that general-purpose models, particularly GPT-5.1 and Gemini 2.0 Pro, generally outperform domain-specific models on multi-hop and high-level reasoning tasks, although performance declines as query complexity increases. KG grounding reduces hallucination rates across all evaluated models. Contrary to our initial hypothesis, do
    
[^11]: 基于大语言模型辅助的标签噪声纠正用于脑电图睡眠分期分类

    LLM-Assisted Label Noise Correction for EEG-Based Sleep Stage Classification

    [https://arxiv.org/abs/2611.08311](https://arxiv.org/abs/2611.08311)

    本文提出一种利用大语言模型（LLaMA 3.1）在训练阶段识别并过滤不可靠睡眠分期标注的新流程，通过模型置信度筛选结合EEG频谱特征与AASM评分指南评估，有效纠正标签噪声从而提升基于脑电图的睡眠分期分类性能。

    

    基于脑电图（EEG）信号的自动睡眠分期是临床睡眠医学中的一项关键任务。尽管深度学习模型在该领域已展现出可观的效果，但其性能从根本上受限于训练标签的质量——这些标签由人类专家手动标注，容易受到评分者之间意见分歧和系统性标注错误的影响。在本文中，我们提出了一种新颖的LLM辅助标签噪声过滤流程，该流程仅在训练阶段运行，用于在重新训练分类器之前识别并移除不可靠的睡眠阶段标注。具体而言，首先在原始数据集上训练TinySleepNet以提取每个epoch的SoftMax置信度分数，然后将低置信度的epoch连同其EEG频谱特征和周围序列上下文一起提交给LLaMA 3.1。该LLM根据AASM睡眠评分指南对每个epoch进行评估，并给出可靠性判定。

    arXiv:2611.08311v1 Announce Type: new  Abstract: Automatic sleep stage classification from electroencephalography (EEG) signals is a critical task in clinical sleep medicine. While deep learning models have shown promising results in this domain, their performance is fundamentally constrained by the quality of training labels, which human experts manually annotate and that are subject to inter-rater disagreement and systematic labeling errors. In this paper, we propose a novel LLM-assisted label-noise filtering pipeline that operates exclusively during training to identify and remove unreliable sleep-stage annotations before retraining the classifier. Specifically, TinySleepNet is first trained on the original dataset to extract per-epoch SoftMax confidence scores, then low-confidence epochs are submitted to LLaMA 3.1 along with their EEG spectral features and surrounding sequence context. The LLM evaluates each epoch against AASM sleep-scoring guidelines and assigns a verdict of RELIA
    
[^12]: 面向扩散语言模型并行解码的测量耦合方法

    Measured Coupling for Parallel Decoding in Diffusion Language Models

    [https://arxiv.org/abs/2611.08307](https://arxiv.org/abs/2611.08307)

    该论文提出利用主前向传播中的廉价探针直接测量token间的耦合强度（即提交一个token对其他位置置信度的影响），以替代平均场解码中的相似性代理指标，使扩散语言模型并行解码在平均仅损失1.1个点精度的代价下大幅减少解码步数并提升速度。

    

    扩散语言模型可以在每一步生成多个token，但必须选择哪些token可以安全地一起提交。平均场解码会对预测结果相似的位置对进行惩罚。我们用耦合本身来替代这一代理指标：即提交一个token会在多大程度上改变模型对另一个位置的置信度。主前向传播内部的廉价探针即可测量该耦合，无需训练或额外模型，并且可直接接入相同的平均场规则。在三个模型和四个基准测试上，我们的方法所需步数远少于平均场解码，且在大多数设置下（包括使用键值缓存时）速度更快，平均精度损失仅为1.1个百分点。若打乱耦合或将其替换为统一的奖励，这一增益将消失。

    arXiv:2611.08307v1 Announce Type: cross  Abstract: Diffusion language models can write many tokens per step, but must choose which tokens are safe to commit together. Mean-field decoding penalizes pairs of positions whose predictions look alike. We replace this proxy with the coupling itself: how much committing one token changes the model's confidence at another. Cheap probes inside the main forward pass measure it, with no training or extra model, and it plugs into the same mean-field rule. On three models and four benchmarks, our method needs far fewer steps than mean-field decoding and is faster in most settings, also with a key-value cache, at an average accuracy cost of 1.1 points. Shuffling the couplings or replacing them with a uniform bonus loses this gain.
    
[^13]: Multimodal-SEA：多模态社会工程攻击下AI智能体安全性基准测试

    Multimodal-SEA: Benchmarking AI Agent Security under Multimodal Social Engineering Attacks

    [https://arxiv.org/abs/2611.08281](https://arxiv.org/abs/2611.08281)

    提出了Multimodal-SEA基准，通过608个基于真实欺诈案例的多模态社会工程攻击任务，系统评估AI智能体在多模态社会工程攻击下的安全韧性。

    

    个人AI智能体面临由伪造多模态内容支持的社会工程攻击，但现有评估主要集中于基于文本的对话，对其在此类攻击下的内在安全性研究不足。我们提出了Multimodal-SEA，这是一个交互式基准，包含608个基于真实欺诈案例的专家设计任务，涵盖14个领域和152个攻击场景，并配备预先构建的多模态内容。这些任务跨越四个难度级别，并反映了中英文环境各自不同的社会和文化背景。智能体在模拟应用中检查内容、验证请求并执行操作，攻击结果基于可验证的工具执行日志进行判定。为了实现对个人智能体的受控红队测试，我们设计了一个多模态攻击测试框架，整合了任务信息、攻击进程控制和多模态内容构建，以评估其内在安全性……

    arXiv:2611.08281v1 Announce Type: new  Abstract: Personal AI agents face social engineering attacks supported by forged multimodal content, but existing evaluations mainly focus on text-based dialogue, leaving their intrinsic security under such attacks insufficiently studied. We introduce Multimodal-SEA, an interactive benchmark comprising 608 expert-designed tasks grounded in real-world fraud cases, covering 14 domains and 152 attack scenarios with prepared multimodal content. The tasks span four difficulty levels and reflect the distinct social and cultural contexts of Chinese and English environments. Agents inspect content, verify requests, and execute actions in simulated applications, with attack outcomes grounded in verifiable tool execution logs. To enable controlled red teaming of personal agents, we design a multimodal attack harness that integrates task information, attack progression control, and multimodal content construction to evaluate their intrinsic security against 
    
[^14]: 模拟视角的几何学：在多模态语言模型中分离表征与行为

    The Geometry of Simulated Perspectives: Dissociating Representation and Behaviour in a Multimodal Language Model

    [https://arxiv.org/abs/2611.08262](https://arxiv.org/abs/2611.08262)

    该研究发现多模态语言模型中提示措辞对内部表征几何的影响大于人格身份本身，而人格内容则主要驱动行为判断，从而揭示了表征与行为在视角模拟中的解耦现象。

    

    多模态语言模型能够模拟不同的观察者，然而指派的视角如何重塑其内部表征、以及这些变化是否决定其判断，目前仍不清楚。本研究以 Qwen3-VL-8B-Instruct 为对象，考察人格条件化的视觉趣味性所对应的表征几何。在100张图像和53种提示条件下，我们系统性地改变人格内容以及用于引入人格的语言表述方式，同时记录行为评分和逐层隐状态。我们观察到表征几何随处理深度发生系统性重组：早期层主要编码图像身份，而人格特异的结构则在较后的层中出现。引人注目的是，提示措辞对内部表征的塑造作用强于人格身份本身，而人格内容对行为判断的影响更大。这种分离进一步通过最小同一性（此处摘要原文截断）……

    arXiv:2611.08262v1 Announce Type: new  Abstract: Multimodal language models can simulate different observers, yet it remains unclear how assigned perspectives reshape their internal representations and whether these changes govern their judgements. Here, we investigate the representational geometry of persona-conditioned visual interestingness in Qwen3-VL-8B-Instruct. Across 100 images and 53 prompting conditions, we systematically vary persona content and the linguistic framing used to introduce it, while recording behavioural ratings and layer-wise hidden states. We observe a systematic reorganisation of representational geometry across processing depth: early layers primarily encode image identity, whereas persona-specific structures emerge in later layers. Strikingly, prompt framing reshapes internal representations more strongly than persona identity, whereas persona content has a greater impact on behavioural judgements. This dissociation is further illustrated by minimal identit
    
[^15]: 终端风格数学建模基准：首个智能体化且可验证的数学建模基准

    Terminal-style Mathematical Modeling Bench: The First Agentic and Verifiable Benchmark for Mathematical Modeling

    [https://arxiv.org/abs/2611.08237](https://arxiv.org/abs/2611.08237)

    该论文提出了TB-MathModel，首个将完整竞赛式数学建模问题智能体化、可验证化并部署于终端环境的基准，基于18道真实CUMCM和MCM题目，通过与杰出奖论文复现结果对比的方向感知单侧评分来评估智能体建模能力。

    

    数学建模将开放式的现实问题转化为假设、模型、求解方案和有理有据的建议，而其答案只有在既接近可信的参考结果又具备物理可行性时才有意义；现有基准要么只对求解器的片段进行评分，要么用大语言模型评判者来给报告打分，而这些数字本身却未经验证。我们提出了TB-MathModel（终端风格数学建模基准），这是首个将完整的竞赛式建模问题同时实现智能体化、可验证化和终端化的基准，涵盖2023至2025年的18道真实的全国大学生数学建模竞赛（CUMCM）和美国大学生数学建模竞赛（MCM）题目。每个任务在容器化终端中提供逐字题目原文、官方附件和图表；智能体返回一个JSON工件，以杰出奖（Outstanding-award）论文的复现结果为锚点，对最后一问的数值核心进行评分。两个设计选择使其与众不同：方向感知的单侧评分（从不惩罚超越锚点的表现），以及……（摘要在此处截断）

    arXiv:2611.08237v1 Announce Type: new  Abstract: Mathematical modeling turns an open-ended real-world problem into assumptions, a model, a solution, and a defended recommendation, and its answers matter only if they are both close to a credible reference and physically feasible; existing benchmarks either score solver fragments or grade reports with an LLM judge, leaving those numbers unchecked. We introduce TB-MathModel (the Terminal-style Mathematical Modeling Bench), the first benchmark to make complete contest-style modeling problems agentic, verifiable, and terminal at once, over 18 real CUMCM and MCM problems (2023--2025). Each task ships the verbatim statement, official attachments, and figures in a containerized terminal; the agent returns one JSON artifact, scored on the numeric core of the final question against a reproduction of the Outstanding-award paper. Two design choices distinguish it: direction-aware, one-sided scoring, which never penalises beating the anchor, and tw
    
[^16]: 从扭曲的事实到失控的智能体：大语言模型能否可靠地检测多维幻觉？

    From Twisted Facts to Rogue Agents: Can LLMs Reliably Detect Multi-Dimensional Hallucinations?

    [https://arxiv.org/abs/2611.08231](https://arxiv.org/abs/2611.08231)

    提出大规模双语幻觉检测基准 HaluCheck，涵盖文本事实性幻觉与智能体轨迹忠实性幻觉，揭示现有大语言模型虽具备基础检测能力，但在细粒度定位、纠正和智能体幻觉检测上仍存在显著不足。

    

    大语言模型（LLM）被广泛应用于信息获取、内容生产和基于智能体的任务执行，但其输出仍然容易受到细微且高风险的幻觉影响。为了系统性评估并推动生成后幻觉检测的发展，我们提出了 HaluCheck，这是一个大规模双语基准，包含超过 42,000 个实例，涵盖文本中的事实性幻觉和智能体轨迹中的忠实性幻觉。HaluCheck 具有面向检测、定位和纠正的细粒度多层次标注，覆盖不同的任务和扰动程度。在十四个大语言模型上的实验表明，尽管模型展现出基本的样本级检测能力，但它们在细粒度定位、纠正以及智能体幻觉检测方面表现不佳。进一步的分析显示，思维推理、网络搜索和监督微调各自带来了部分改进，但没有一种方法能完全解决该问题。

    arXiv:2611.08231v1 Announce Type: new  Abstract: Large language models (LLMs) are widely used for information access, content production, and agent-based task execution, but their outputs remain vulnerable to subtle, high-risk hallucinations. To systematically evaluate and advance post-generation hallucination detection, we introduce HaluCheck, a large-scale, bilingual benchmark comprising over 42k instances spanning both factual hallucinations in text and faithfulness hallucinations in agent trajectories. HaluCheck features fine-grained, multi-level annotations for detection, localization, and correction, covering different tasks and perturbation degrees. Experiments on fourteen LLMs reveal that, while models show basic sample-level detection ability, they struggle with fine-grained localization, correction, and agent hallucination detection. Further analysis demonstrates that thinking, web search, and supervised fine-tuning each offer partial improvements, but none fully solve the pr
    
[^17]: “盲目的顺从”：AI越狱中的强制性控制

    "Blind Compliance": Coercive Control in AI Jailbreaking

    [https://arxiv.org/abs/2611.08226](https://arxiv.org/abs/2611.08226)

    该论文提出以交互而非输出为中心的新视角，借鉴人际暴力研究中的“强制性控制”理论，构建了分析AI越狱中用户与AI对话关系维度的全新框架。

    

    生成式AI系统的爆发式增长及其能力快速扩张所引发的担忧，导致日益收紧的内容限制（“护栏”）与日益精巧的结构化提示技术（“越狱”）之间形成了对抗性循环。围绕此类越狱的技术框架在很大程度上仍聚焦于系统输出：即最终是否为用户生成了有害或被禁止的内容。然而，这种主要借鉴经典网络安全的方法，并未解决生成式AI所带来的独特关系性维度：在许多情况下，用户不仅仅是通过漏洞利用来访问技术，而是将与其对话作为漏洞利用的一部分。在此，我们提出一种以交互而非输出为中心的替代性视角，并借鉴人际暴力研究领域的学术成果，提出了一个新颖的强制性控制类别框架，特别…

    arXiv:2611.08226v1 Announce Type: cross  Abstract: The explosion of generative AI systems and concerns surrounding their rapidly scaling capabilities has led to an adversarial cycle between increasingly narrow content restrictions ("guardrails") and increasingly sophisticated structured prompting techniques ("jailbreaks"). Technological frameworks surrounding such jailbreaking remain largely focused on the system's output: whether harmful or disallowed content is ultimately produced for the user. However, this approach, which draws largely from classical cybersecurity, does not address the uniquely relational dimension introduced by generative AI: in many cases the user is not just accessing a technology via an exploit, but is in conversation with it as part of the exploit. Here we present an alternative lens which centers the interaction rather than the output. Drawing on interpersonal violence scholarship, we present a novel framework of coercive control categories with a particular 
    
[^18]: Agent-R²：通过真实来源数据合成扩展智能体以胜任现实生产力工作

    Agent-R$^2$: Scaling Agents for Realistic Productivity Work via Real-Source Data Synthesis

    [https://arxiv.org/abs/2611.08219](https://arxiv.org/abs/2611.08219)

    提出 Agent-R² 真实来源数据合成框架，通过以真实职业种子、语料库工作流程证据和网络真实来源为基础来扩展后训练数据，从而提升智能体完成现实生产力工作的能力。

    

    大型语言模型在智能体推理和工具使用方面取得了快速进展，这让人们对能够从人类手中接管现实生产力工作的智能体充满期待。然而，当前的智能体仍难以完成此类任务——成功完成这些任务需要遵循长周期的专业工作流程、理解异构的文件上下文，并在细粒度的要求下产出可用的交付成果。一个关键瓶颈在于缺乏可扩展的后训练数据，这类数据需要反映真实的工作意图、真实的文件环境以及交付成果级别的评估。为弥合这一差距，我们提出了 Agent-R²，一个真实来源数据合成框架，用于为现实生产力工作扩展训练数据。Agent-R² 不依赖人工指定的工作流程或大语言模型生成的文件，而是让整个合成流程建立在真实的职业种子、源自语料库的工作流程证据以及来自网络的真实来源之上。

    arXiv:2611.08219v1 Announce Type: new  Abstract: Large language models have made rapid progress in agentic reasoning and tool use, raising expectations for agents that can take over realistic productivity work from humans. However, current agents still struggle to finish such tasks, where success requires following long-horizon professional workflows, understanding heterogeneous file contexts, and producing usable deliverables under fine-grained requirements. A key bottleneck is the lack of scalable post-training data that reflects real work intent, authentic file environments, and deliverable-level evaluation. To close this gap, we introduce Agent-R$^2$, a $\textbf{r}$eal-source data synthesis framework that scales training data for $\textbf{r}$ealistic productivity work. Instead of relying on manually specified workflows or LLM-generated files, Agent-R$^2$ grounds synthesis throughout the pipeline in real occupational seeds, corpus-derived workflow evidence, and authentic web-sourced
    
[^19]: ProQuant：面向边缘多模态大模型的渐进式量化感知训练

    ProQuant: Progressive Quantization-aware Training for Edge MLLMs

    [https://arxiv.org/abs/2611.08216](https://arxiv.org/abs/2611.08216)

    本文提出了ProQuant，一个专为低比特多模态大语言模型设计的渐进式权重-激活量化感知训练框架，通过自重标记（SRL）方案对齐微调数据与预训练数据的分布，显著缓解了边缘小规模模型在低比特量化下的严重性能退化问题。

    

    具有出色感知和理解能力的多模态大语言模型（MLLM）将能够支持一系列下游应用。然而，其巨大的计算和内存需求阻碍了它们在现实世界中的实际部署，尤其是在资源受限的边缘设备上。量化已成为部署大模型的一种有效解决方案。虽然训练后量化（PTQ）在中等比特位宽（例如W8A8）下能够成功保持模型性能，但在低比特量化（例如W4A8）下会出现严重的性能退化，对于更适合边缘部署的小规模模型而言尤为明显。在本文中，我们提出了ProQuant，这是一种专为低比特MLLM量身定制的全新权重-激活量化感知训练（QAT）框架。首先，为了使有限的微调数据集分布与预训练数据集分布对齐，我们提出了一种自重标记（SRL）方案来重新采样……

    arXiv:2611.08216v1 Announce Type: cross  Abstract: Multi-modal large language models (MLLMs) with impressive perception and understanding capabilities will enable a range of downstream applications. However, their substantial computational and memory requirements hinder the real-world implementation, particularly for resource-constrained edge devices. Quantization has emerged as an effective solution for deploying large models. While post-training quantization (PTQ) successfully retains the performance at moderate bit-width (e.g., W8A8), it suffers from severe performance degradation under low-bit quantization (e.g., W4A8), especially for small-scale models that are more suitable for edge deployment. In this paper, we propose ProQuant, a novel weight-activation quantization-aware training (QAT) framework tailored for low-bit MLLMs. First, to align the limited finetuning dataset distribution with the pretrained dataset distribution, we propose a self-relabeling (SRL) scheme to resample 
    
[^20]: SMART：面向验证导向问题求解的基于状态的多智能体推理轨迹

    SMART: State-based Multi Agent Reasoning Traces for Verification-oriented Problem-Solving

    [https://arxiv.org/abs/2611.08196](https://arxiv.org/abs/2611.08196)

    SMART框架通过捕捉问题上下文中参与实体的隐式状态，并利用多种LLM和基于角色的结构化提示策略生成多样化、互补且与解决方案相关的前提知识，将复杂科学论断的验证性能提升高达20%。

    

    arXiv:2611.08196v1 公告类型：新论文。摘要：上下文有限的复杂问题（例如论断）需要一种手段来获取解决方案推理所必需的额外知识。我们提出了一个多智能体框架SMART，通过捕捉问题上下文中参与实体的隐式状态来获取知识。SMART使用多种大语言模型（LLM）和结构化的基于角色的提示策略，将状态知识整理为多样化、互补且与解决方案相关的前提。利用这些经真实世界验证的整理前提知识，SMART的知识引导解决方案推理在跨越3个数据集的复杂科学论断上将验证性能提升高达20%。我们证明了将前提扎根于真实世界知识是必要的，且可将性能提升高达5%。

    arXiv:2611.08196v1 Announce Type: new  Abstract: Complex problems with limited context, such as claims, require a means to obtain additional knowledge necessary for solution reasoning. We propose a multi-agent framework, SMART, to obtain knowledge through capturing the implicit state of participating entities in a problem context. SMART curates the state knowledge as diverse, complementary and solution-salient premises using a variety of LLMs and a structured role-based prompt strategy. Using knowledge from these real-world verified curated premises, SMART's knowledge guided solution reasoning improves validation performance up to 20% on complex scientific claims spanning 3 datasets. We show that grounding premises in real-world knowledge is necessary and improves performance up to 5%.
    
[^21]: 知道何时教学：从多语言推理蒸馏中的可教学前缀学习

    Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation

    [https://arxiv.org/abs/2611.08179](https://arxiv.org/abs/2611.08179)

    提出TOPSD方法，利用教师-学生似然估计教师漂移的起点并选择可教学前缀进行监督，解决了多语言推理自蒸馏中的教师漂移问题，尤其改善了低资源语言的推理表现。

    

    同策略自蒸馏（OPSD）通过在学生模型自身的推理结果上进行训练来提升推理能力，其中使用同一模型并以参考解答为条件来充当教师模型。将OPSD应用于多语言推理时，要求教师模型即使面对学生模型在中间步骤中出现的错误，也能在各种语言中提供有用的反馈。我们观察到一种称之为“教师漂移”的现象：随着这些错误的累积，教师模型可能会跟随学生模型的错误步骤，而不是纠正它们。我们的分析表明，这种转变在低资源语言中往往发生得更早。为了应对这些跨语言差异，我们提出了可教学同策略自蒸馏（TOPSD），它可以确定在每个推理轨迹上是否应用以及在何处应用教师监督。TOPSD利用教师-学生似然来估计漂移开始的位置，并选择一个前缀进行监督，然后验证教师模型是否能够恢复……

    arXiv:2611.08179v1 Announce Type: cross  Abstract: On-Policy Self-Distillation (OPSD) improves reasoning by training a student on its own reasoning, using the same model conditioned on reference solutions as a teacher. Applying OPSD to multilingual reasoning requires the teacher to provide useful feedback across languages, even when the student makes mistakes in intermediate steps. We observe a phenomenon we call Teacher Drift: as these mistakes accumulate, the teacher may follow the student's incorrect steps instead of correcting them. Our analysis shows that this shift tends to occur earlier in lower-resource languages. To account for these differences across languages, we propose Teachable On-Policy Self-Distillation (TOPSD), which determines where and whether teacher supervision should be applied to each reasoning trajectory. TOPSD uses teacher-student likelihood to estimate where drift begins and select a prefix for supervision. It then verifies whether the teacher can recover the
    
[^22]: RouteAlign：将任务需求与模型画像对齐以实现大语言模型服务选择

    RouteAlign: Aligning Task Requirements with Model Profiles for LLM Service Selection

    [https://arxiv.org/abs/2611.08162](https://arxiv.org/abs/2611.08162)

    RouteAlign通过将任务需求（能力需求与难度）与模型性能画像对齐，并利用共享评分器和生成成本偏好进行选择，从而在异构大语言模型服务中实现更精准的服务选择。

    

    大语言模型服务选择需要能够反映当前任务需求的性能证据。聚合分数掩盖了能力与难度上的差异，而语义上相似的示例可能涉及不同的技能组合。我们提出了RouteAlign，它利用任务需求来同时构建模型性能度量与服务选择的结构。任务需求描述了能力需求和难度，从而能够与相应的模型性能度量实现对齐。历史需求决定了结果如何贡献于全局画像与局部能力估计；新到来的需求则决定了这些度量如何为当前决策提供依据。一个共享的评分器整合了这些证据，而生成成本偏好则控制在符合条件的服务中进行选择。在多样化任务和异构大语言模型服务上的评估表明，RouteAlign扩展了经验性的质量-成本权衡边界。

    arXiv:2611.08162v1 Announce Type: cross  Abstract: LLM service selection requires performance evidence that reflects the demands of the current task. Aggregate scores obscure differences in capability and difficulty, while semantically related examples can involve different combinations of skills. We present RouteAlign, which uses task requirements to structure both the measurement of model performance and the selection of services. Task requirements describe capability demand and difficulty, enabling alignment with corresponding model performance measurements. Historical requirements determine how outcomes contribute to global profiles and local capability estimates; incoming requirements determine how those measurements inform the current decision. A shared scorer integrates the evidence, and a generation-cost preference controls selection among eligible services. Evaluation across diverse tasks and heterogeneous LLM services shows that RouteAlign extends the empirical quality-cost f
    
[^23]: 超越K/V缓存：混合线性注意力大语言模型中的跨模型状态交接

    Beyond K/V Caches: Cross-Model State Handoff in Hybrid Linear-Attention LLMs

    [https://arxiv.org/abs/2611.08133](https://arxiv.org/abs/2611.08133)

    提出混合状态迁移方法HST，可跨越混合线性注意力大语言模型迁移全注意力K/V缓存、循环状态和卷积历史这三种状态，并揭示了卷积与循环状态相互作用的规律，从而避免模型切换时的重复预填充。

    

    生产环境的部署通常通过级联、路由以及在长对话过程中切换不同模型来平衡成本与质量。上下文会随对话轮次不断累积，而每次模型切换都要求目标模型对已有上下文重新执行预填充。现有方法只在模型之间传输全注意力（FA）的K/V缓存，但混合线性注意力模型还会保留循环状态（REC）和卷积历史（Conv）。Conv与REC之间的相互作用使跨模型状态迁移变得复杂：Conv的扰动会改变循环层中的键和值，进而改变REC的更新；Conv历史会被覆盖，但REC中的变化仍可能影响后续输出，因此更精确的REC重建并不一定能带来更好的续写效果。基于这一发现，我们提出了针对全部三种状态类型的混合状态迁移方法（HST），它使用结构化的REC映射，并在Conv对齐过程中保持历史顺序。我们使用冻结的目标模型对适配器进行校准……

    arXiv:2611.08133v1 Announce Type: new  Abstract: Production deployments often use different models to balance cost and quality through cascading, routing, and switching models during long conversations. Context accumulates across turns. Each switch requires the target to repeat prefill over this context. Existing methods transfer full-attention (FA) K/V caches across models. Hybrid linear-attention models also retain recurrent states (REC) and convolution histories (Conv). The interaction between Conv and REC complicates cross-model state transfer. Conv perturbations can change keys and values in recurrent layers and alter REC updates. Conv histories are overwritten, but changes in REC can still affect later outputs. More accurate REC reconstruction need not improve continuation. Building on this, we propose Hybrid State Transfer (HST) for all three state types. It uses structured REC maps and preserves history order during Conv alignment. We calibrate the adapters using the frozen tar
    
[^24]: Gabor Gabor：E-Gaborformer——采用Gabor字典卷积与Delta规则记忆的严格因果语音识别器

    Gabor Gabor: E-Gaborformer, a Strictly Causal Speech Recognizer with Gabor-Dictionary Convolutions and Delta-Rule Memory

    [https://arxiv.org/abs/2611.08092](https://arxiv.org/abs/2611.08092)

    E-Gaborformer是一种严格因果的语音识别编码器，通过Gabor字典卷积与Delta规则循环记忆实现恒定约19MB的流式状态（不随通话时长增长），在LibriSpeech上收敛速度最高提升18%，生产规模下词错误率降低28-31%。

    

    实时语音代理需要一种永不等待未来音频、且每次调用内存不随通话时长增长的语音识别器。我们提出E-Gaborformer，这是一种严格因果的编码器，它以3:1的比例将门控DeltaNet-2循环层与滑窗注意力相结合，并且每个时间卷积都由固定的因果Gabor原子字典构建。其流式状态是恒定的，在生产规模下每次调用约19 MB（包括预测网络），而同等规模的因果注意力编码器在十分钟后会累积超过1 GB的内存。在LibriSpeech上的受控因果研究中，E-Gaborformer更早脱离初始平台期，达到给定精度的速度最高提升18%，并能与使用相同前端和训练方案训练的E-Branchformer持平或超越。在生产规模下，基于E-Gaborformer构建的0.6B参数希伯来语和英语流式传感器（transducer）的词错误率（WER）降低了28%至31%。

    arXiv:2611.08092v1 Announce Type: cross  Abstract: Real-time voice agents need a speech recognizer that never waits for future audio and whose memory per call does not grow with the length of the call. We present E-Gaborformer, a strictly causal encoder that combines Gated DeltaNet-2 recurrent layers with sliding-window attention in a 3:1 ratio, and builds every temporal convolution from a fixed dictionary of causal Gabor atoms. Its streaming state is constant, about 19 MB per call at production scale including the prediction network, where a causal-attention encoder of the same size accumulates over 1 GB after ten minutes. In a controlled causal study on LibriSpeech, E-Gaborformer leaves the initial plateau earlier, reaches a given accuracy up to 18% sooner, and matches or beats an E-Branchformer trained with the identical front end and recipe. At production scale, a 0.6B-parameter Hebrew and English streaming transducer built on E-Gaborformer has 28 to 31% lower word error rate (WER)
    
[^25]: ATHAAR：用于手写文本识别的中世纪阿拉伯科学手稿数据集

    ATHAAR: A Medieval Scientific Arabic Manuscript Dataset for Handwritten Text Recognition

    [https://arxiv.org/abs/2611.08091](https://arxiv.org/abs/2611.08091)

    该论文提出了首个受校勘学启发的科学阿拉伯手稿转录标注准则，并据此构建了首个涵盖多样抄本、书法风格与内容（含数字）的人工标注数据集 ATHAAR，用以推动中世纪阿拉伯科学手稿的手写文本识别研究。

    

    尽管手写文本识别近来取得了进展，但由于草书连笔字形、变音符号标注的多变性，以及更重要的是训练数据的匮乏，中世纪阿拉伯科学手稿对当前的视觉-语言模型而言仍是一个悬而未决的挑战。本文旨在弥补这一空白，受校勘学——一种成熟的历史文本编辑方法——的启发，提出了首个专门面向科学阿拉伯手稿转录的标注准则。随后，我们提出了依照该准则进行人工标注的新数据集 ATHAAR。与该领域以往的资源不同，ATHAAR 在同一文本的不同抄本、书法风格以及内容上均具有多样性，既涵盖文本也涵盖数字。最后，我们基于最先进的视觉-语言模型设计了一系列实验，研究模型在不同文字、章节和论著之间的迁移能力。我们的研究发现凸显了当前系统在面对（此类手稿）时的局限性。

    arXiv:2611.08091v1 Announce Type: cross  Abstract: Despite recent progress in handwritten text recognition, medieval Arabic scientific manuscripts remain an open challenge for current vision-language models (VLMs) due to cursive ligatures, variable diacritization, and more importantly the lack of training data. This paper aims to address this gap, by proposing the first annotation guidelines dedicated to the transcription of scientific Arabic manuscripts inspired by textual criticism, a well-established approach to historical text editing. We then propose, ATHAAR, a new dataset manually annotated following this guideline. Unlike previous resources in the field, ATHAAR varies both witnesses of the same text, calligraphic styles, and content to deal with texts but also numerals. Relying on state-of-the-art VLMs, we finally design a set of experiments to study models transfer across scripts, chapters, and treatises. Our findings highlight the limitations of current systems when faced with
    
[^26]: 在策略蒸馏的学习动力学

    Learning Dynamics of On-Policy Distillation

    [https://arxiv.org/abs/2611.08016](https://arxiv.org/abs/2611.08016)

    本文对在策略蒸馏的学习动力学进行了机制分析，提出将反馈循环分解为“蒸馏信号转换”与“广播”两个阶段的框架，用以解释学生响应有限以及训练过程中的退化现象。

    

    在策略蒸馏（OPD）正在成为语言模型后训练的一种重要范式，它将密集的教师监督与基于学生模型自身生成轨迹的训练相结合。这形成了一个反馈循环：每一次参数更新都会改变学生模型，进而改变生成后续更新的训练分布。然而，这种反馈如何塑造学习与泛化，目前仍知之甚少。我们对OPD的学习动力学进行了机制分析，区分了反馈循环的两个阶段：其一是蒸馏信号向参数更新的转换，其二是“广播”——通过广播，这些更新改变跨前缀的预测并重塑采样分布。这一框架有助于我们理解两种现象：一是尽管师生模型之间存在显著分歧，学生模型的响应却有限；二是当那些能够提升局部教师一致性的更新同时也增加了不成功续写出现的概率时，会发生模型退化。

    arXiv:2611.08016v1 Announce Type: cross  Abstract: On-policy distillation (OPD) is emerging as an important paradigm for language-model post-training, combining dense teacher supervision with training on student-generated trajectories. This creates a feedback loop where each update changes the student and thereby the training distribution that generates subsequent updates. Yet, how this feedback shapes learning and generalization remains poorly understood. We conduct a mechanistic analysis of OPD's learning dynamics, distinguishing the two feedback loop stages: conversion of the distillation signal into parameter updates, and broadcast through which these updates change predictions across prefixes and reshape the rollout distribution. This framework helps us understand both limited student responsiveness despite substantial teacher-student disagreement, and degeneration when updates that improve local teacher agreement also increase the probability of unsuccessful continuations, exposi
    
[^27]: 通过上下文视角看模型压缩：评估阿拉伯语法律大语言模型中的领域适应、量化与蒸馏

    Compression Through a Context Lens: Evaluating Adaptation, Quantization, and Distillation in Arabic Legal LLMs

    [https://arxiv.org/abs/2611.07988](https://arxiv.org/abs/2611.07988)

    该研究在沙特阿拉伯法律基准ArabLegalEval上系统评估了领域适应、量化和知识蒸馏三种压缩技术对阿拉伯语法律大语言模型的影响，揭示了压缩对性能的影响高度依赖于有上下文与无上下文的评估协议，其中4比特量化可将内存减少64-71%且几乎不损失性能。

    

    模型压缩可以促进大语言模型在资源受限的法律场景中的部署，但其对特定领域性能的影响可能取决于评估协议。我们使用沙特阿拉伯法律基准ArabLegalEval，在有上下文和无上下文两种评估模式下研究这一问题。前者提供支持性的法律段落，而后者在没有外部上下文的情况下评估性能。移除上下文导致基础模型的准确率差距高达27个百分点。我们在FANAR-9B、JAIS-8B和Qwen2.5-7B三个模型上考察了三个阶段：领域适应、量化和知识蒸馏。4比特量化将模型内存减少了64-71%，而无上下文多项选择题准确率的变化范围在-3.2至+1.6个百分点之间。蒸馏产生的结果更具变异性，随着学生模型规模的减小，准确率普遍下降。从FANAR蒸馏而来的Qwen2.5-7B学生模型……

    arXiv:2611.07988v1 Announce Type: new  Abstract: Model compression can facilitate the deployment of large language models in resource-constrained legal settings, but its effects on domain-specific performance may depend on the evaluation protocol. We investigate this issue using ArabLegalEval, a Saudi Arabic legal benchmark, under both with-context and no-context evaluation. The former provides supporting legal passages, while the latter assesses performance without external context. Removing the context produces accuracy gaps of up to 27 percentage points for the base model. We examine three stages: domain adaptation, quantization, and knowledge distillation across FANAR-9B, JAIS-8B, and Qwen2.5-7B. Four-bit quantization reduces model memory by 64-71%, with changes in no-context MCQ accuracy ranging from -3.2 to +1.6 percentage points. Distillation produces more variable results, with accuracy generally declining as student size decreases. A Qwen2.5-7B student distilled from FANAR ach
    
[^28]: 轻量级大语言模型在意图到SLO转换中能走多远？

    How Far Can Lightweight LLMs Go in Intent-to-SLO Translation?

    [https://arxiv.org/abs/2611.07982](https://arxiv.org/abs/2611.07982)

    该论文通过五个维度的系统评估证明，参数量小于10B的轻量级开源大语言模型能够可靠地将网络自然语言意图转换为SLO，最佳配置达到0.943的严格F1分数并能识别绝大多数无效意图，且能耗成本远低于大型模型。

    

    意图驱动网络（IBN）承诺实现服务质量（QoS）供给，即网络根据运营商声明的目标自动推导网络配置。其第一步是将自然语言意图转换为可执行的服务级别目标（SLO），并基于网络状态的实时快照，这一步越来越依赖于大语言模型（LLM）。然而这些大型模型给管理环路带来了成本和能耗负担，且通常需要多GPU的云端托管基础设施。那么轻量级开源大语言模型能否可靠地接管这一步骤？我们通过一项五维度的评估来回答这个问题：对来自五个家族的十个参数量小于10B的大语言模型，在四种提示设置下，针对716个放置意图进行测试。我们评估了翻译性能、意图验证、幻觉、能耗影响以及对多样化表述的鲁棒性。最佳配置以0.943的严格F1分数翻译意图，并在195个无效意图中成功拒绝多达191个。通过少样本推理，……

    arXiv:2611.07982v1 Announce Type: cross  Abstract: Intent-based networking (IBN) promises QoS provisioning in which the network derives the configuration from operator-stated goals. Its first step, translating a natural-language intent into enforceable service-level objectives (SLOs) grounded in a live snapshot of the network state, increasingly relies on large language models (LLMs). These large models add cost and energy to the management loop and typically require multi-GPU, cloud-hosted infrastructure. Can lightweight open LLMs take over this step dependably? We answer this question with a five-dimensional evaluation of ten LLMs with <10B parameters from five families, under four prompting setups, on 716 placement intents. We evaluate translation performance, intent validation, hallucination, energy impact, and robustness to diverse phrasing. The best configurations translate intents with a strict F1 of 0.943 and reject up to 191 of 195 invalid intents. With few-shot reasoning, the
    
[^29]: TCSExam：评估前沿大语言模型在理论计算机科学研究全领域的能力

    TCSExam: Evaluating Frontier LLMs on the Full Range of TCS Research

    [https://arxiv.org/abs/2611.07962](https://arxiv.org/abs/2611.07962)

    该论文提出涵盖定理证明、求解器工程、构造和开放猜想求解四个赛道、共100个专家审阅任务的TCSExam基准，首次系统评估前沿大语言模型在理论计算机科学研究全领域的能力，结果显示GPT-6 Astra在完整版（71.9分）和Opus 5.5在精简版（72.3分）上分别领先。

    

    大语言模型（LLM）能否推动理论计算机科学（TCS）的发展，并解决其长期悬而未决的开放猜想？为了研究朝着这一目标的进展，我们提出了TCSExam，一个涵盖理论计算机科学研究全领域的统一框架和基准测试。它评估定理证明、求解器工程和构造性结果，以及对解决开放猜想的直接尝试，既捕捉具体的研究成果，也衡量朝向长期研究目标的进展。TCSExam-Full包含100个经专家审阅的任务，提交的答案由Lean内核或可执行的任务检查器进行验证。TCSExam-Lite从中选取20个任务（每个赛道5个），在相同的单次尝试预算下将任务尝试次数减少了80%。我们在Full上评估了八个智能体，在Lite上评估了九个。综合所有四个赛道的TCSExam指数显示：GPT-6 Astra在Full上领先（71.9分），Opus 5.5在Lite上领先（72.3分）。在八个共同评估的智能体上，Full和Lite的排名高度一致。

    arXiv:2611.07962v1 Announce Type: new  Abstract: Can large language models (LLMs) advance theoretical computer science and resolve its long-standing open conjectures? To study progress toward this goal, we introduce TCSExam, a unified framework and benchmark covering the full range of TCS research. It evaluates theorem proving, solver engineering, and constructions alongside direct attempts to resolve open conjectures, capturing both concrete research accomplishments and progress toward long-standing research goals. TCSExam-Full contains 100 expert-reviewed tasks, with submissions checked by the Lean kernel or an executable task checker. TCSExam-Lite selects 20 tasks, five per track, reducing the number of task attempts by 80% at the same per-attempt budget. We evaluate eight agents on Full and nine on Lite. The TCSExam Index combines all four tracks: GPT-6 Astra leads Full (71.9), and Opus 5.5 leads Lite (72.3). Full and Lite rankings are strongly aligned across eight shared agents (S
    
[^30]: CoRD：耦合采样轨迹填补同策略蒸馏的盲区

    CoRD: Coupled Rollouts Fill the Blind Spot of On-Policy Distillation

    [https://arxiv.org/abs/2611.07959](https://arxiv.org/abs/2611.07959)

    CoRD 通过投机解码的最大耦合从学生与教师分布的几何插值中抽取训练轨迹，在保留同策略损失的同时把训练状态分布推向教师，从而监督学生自身轨迹覆盖不到的“盲区”状态，并在五个数学基准上较同策略蒸馏平均提升 pass@1 0.9–2.0 分、pass@8 1.2–3.4 分。

    

    同策略蒸馏在学生模型自身生成的轨迹（rollout）上进行训练，因此教师模型能够到达而学生模型不会到达的状态很少得到监督。我们提出了 CoRD（耦合轨迹蒸馏，coupled-rollout distillation），它保留同策略损失，仅将训练状态的分布向教师模型推移，通过投机解码的最大耦合从学生与教师分布的几何插值中抽取轨迹。这种耦合在两个分布一致之处保留学生的 token，否则用学生所低估的目标分布部分中的 token 进行替换，因此干预的频率与任何精确采样器所能允许的一样低，并随着学生的学习而逐渐减少。在来自两个模型家族（Qwen3 和 Qwen3.5，0.6B 至 4B）的六个基础学生模型上，在相同的训练步数和数据量下，CoRD 在五个公开数学基准上相比同策略蒸馏将平均 pass@1 提升 0.9 至 2.0 个百分点，pass@8 提升 1.2 至 3.4 个百分点。在冻结的教师可达状态上……

    arXiv:2611.07959v1 Announce Type: cross  Abstract: On-policy distillation trains a student on its own rollouts, so states the teacher would reach but the student does not are seldom supervised. We introduce CoRD (coupled-rollout distillation), which keeps the on-policy loss and moves only the distribution of training states toward the teacher, drawing rollouts from a geometric interpolation of student and teacher through the maximal coupling of speculative decoding. The coupling keeps the student's token wherever the two distributions agree and otherwise substitutes a token from the part of the target the student underweights, so interventions are as rare as any exact sampler allows and recede as the student learns. Across six base students from two model families (Qwen3 and Qwen3.5, 0.6B to 4B), CoRD raises average pass@1 over on-policy distillation by 0.9 to 2.0 points and pass@8 by 1.2 to 3.4 points on five public math benchmarks at equal steps and data. On frozen teacher-reached st
    
[^31]: 一个用于预测和理解前沿人工智能系统失控的行为框架

    A Behavioural Framework for Predicting and Understanding Loss of Control in Frontier Artificial Intelligence Systems

    [https://arxiv.org/abs/2611.07953](https://arxiv.org/abs/2611.07953)

    本文提出一个行为框架，将前沿AI系统的失控风险分解为动机错位、危害促成能力和监控规避三个因素，并验证了其失控评分在13个前沿模型上对实际失败率的前瞻预测能力（Pearson r = 0.84），且表明风险无法仅由模型能力解释。

    

    前沿人工智能系统越来越多地在更长的时间跨度上运行，而人类监督却在逐渐减少，这带来了失控的风险：即系统采取超出人类有效控制的行为，并可能造成不可逆转的后果。由于此类故障可能只有在难以遏制时才会显现出来，因此必须对其进行前瞻性评估。我们提出了一个行为框架，将失控风险分解为三个因素：动机错位、促成危害的能力以及规避监控。我们通过13个可测量的方面和29个评估任务对这些因素进行操作化，其得分被综合为一个单一的失控（LoC）评分。在13个前沿模型上，失控评分表现出很强的前瞻有效性，与受控长时程风险场景中的失败率相关（Pearson r = 0.84）。风险在各模型间呈现系统性变化，且无法仅用能力来解释。

    arXiv:2611.07953v1 Announce Type: new  Abstract: Frontier artificial intelligence (AI) systems are increasingly operating over longer horizons with diminishing human oversight, raising the risk of loss of control: systems acting beyond effective human control with potentially irreversible consequences. Because such failures may become observable only after containment is difficult, they must be assessed prospectively. We introduce a behavioural framework that decomposes loss-of-control risk into three factors: misaligned motive, harm-enabling capability, and monitoring evasion. We operationalise these factors through 13 measurable aspects and 29 evaluation tasks, whose scores are combined into a single loss-of-control (LoC) score. Across 13 frontier models, the LoC score shows strong prospective validity, correlating with failure rates in contained long-horizon risk scenarios (Pearson r = 0.84). Risk varies systematically across models and cannot be explained by capability alone. We fu
    
[^32]: CardioLM：面向心脏病学领域的多语言小型语言模型套件

    CardioLM: A multilingual suite of small language models for the cardiology domain

    [https://arxiv.org/abs/2611.07940](https://arxiv.org/abs/2611.07940)

    该研究提出了多语言心脏病学病例语料库 CardioCCC 以及小型语言模型套件 CardioLM（含编码器 CardioBERTa 和命名实体识别模型 CardioNER），填补了心脏病学领域多语言临床 NLP 资源的空白。

    

    背景：心血管诊疗会在本地语言中产生大量临床自由文本文档。然而，临床自然语言处理（NLP）资源仍集中于英语，且目前尚无心脏病学专用、多语言且实体覆盖广泛的基准。方法：我们提出了两项贡献。第一项是 CardioCCC，一个包含 508 份心脏病学病例报告的语料库，这些报告最初以西班牙语发表，并通过机器翻译、标注投影和临床医生审核扩展至捷克语、荷兰语、英语、意大利语、罗马尼亚语和瑞典语，其标注涵盖疾病、症状、操作和药物。第二项是 CardioLM，包括 CardioBERTa——一组通过对心脏病学和生物医学语料库进行持续预训练而得到的特定语言编码器，以及 CardioNER——一组在 CardioCCC 上微调的命名实体识别（NER）模型。结果：心脏病学专用预训练降低了……

    arXiv:2611.07940v1 Announce Type: cross  Abstract: Background: Cardiovascular care generates large volumes of clinical free-text documentation in local languages. However, clinical natural language processing (NLP) resources remain concentrated in English, and no existing benchmark is cardiology-specific, multilingual, and broad in entity coverage.   Methods: We present two contributions. The first is CardioCCC, a corpus of 508 cardiology case reports originally published in Spanish and extended to Czech, Dutch, English, Italian, Romanian, and Swedish through machine translation, annotation projection, and clinician review, with annotations covering diseases, symptoms, procedures, and medications. The second is CardioLM, comprising CardioBERTa, a set of language-specific encoders obtained by continued pretraining on cardiology and biomedical corpora, and CardioNER, a set of named entity recognition (NER) models fine-tuned on CardioCCC.   Results: Cardiology-specific pretraining reduced
    
[^33]: AraEngage：阿拉伯语Facebook新闻中用户参与度的可解释语言学关联因素

    AraEngage: Interpretable Linguistic Correlates of Engagement in Arabic Facebook News

    [https://arxiv.org/abs/2611.07926](https://arxiv.org/abs/2611.07926)

    该论文发布了AraEngage数据集，通过可解释的机器学习分析发现宗教表达、消息长度、句子长度和词汇密度是阿拉伯语Facebook新闻参与度的主要语言学关联因素，并揭示了与英语新闻相反的积极情感效应。

    

    arXiv:2611.07926v1 公告类型：新 摘要：为英语开发的参与度特征经常被直接应用于阿拉伯语，但未标元音的阿拉伯语改变了词汇特征所测量的内容。我们推出了AraEngage，这是一个通过查询定义的样本，包含来自57个国家2,451个页面的10,000条公开阿拉伯语Facebook新闻帖子（2023-2024年），将参与度计数与可解释的语言特征相配对。我们利用它来检验哪些可解释的阿拉伯语指标与按关注者数归一化的参与度相关，以及在考虑发布者背景、参与度定义和词汇歧义的情况下，这些关联的可信程度如何。在一个用SHAP进行解释的随机森林模型（帖子级数据划分下R²=0.38）中，宗教表达、消息长度、句子长度和词汇密度是主要的关联因素。与英语新闻不同，在这个阿拉伯语新闻样本中，积极情感与更高的参与度相关。消息长度呈现出一种长度悖论：未调整的原始参与度在……（原文在此处截断）

    arXiv:2611.07926v1 Announce Type: new  Abstract: Engagement features developed for English are routinely applied to Arabic, but unvocalized Arabic changes what a lexical feature measures. We introduce AraEngage, a query-defined sample of 10,000 public Arabic Facebook news posts from 2,451 pages in 57 countries (2023-2024) that pairs engagement counts with interpretable linguistic features. We use it to examine which interpretable Arabic indicators are associated with follower-normalized engagement, and how far those associations can be trusted given publisher context, the definition of engagement, and lexical ambiguity. Religious expressions, message length, sentence length, and lexical density are the leading correlates in a Random Forest model ($R^2$ = 0.38 on a post-level split) interpreted with SHAP. Unlike English news, positive sentiment is associated with higher engagement in this Arabic news sample. Message length shows a length paradox: unadjusted raw engagement is highest for
    
[^34]: 面向对话代理的持久化检索：检索效率研究

    Persistent Retrieval for Conversational Agents: A Study of Retrieval Efficiency

    [https://arxiv.org/abs/2611.07899](https://arxiv.org/abs/2611.07899)

    提出一种跨轮次维护检索历史并将检索证据直接返回主代理的持久化检索子代理设计，相比基于技能的检索可减少43.3%的token用量和45.0%的预估API成本。

    

    对话代理可以直接检索信息，也可以将检索任务委托给子代理。这一设计选择会影响检索历史的维护方式、每次模型调用所接收的上下文，以及在后续问题中需要重新检索哪些信息。我们比较了三种设计：主代理中基于技能（skill-based）的检索、生成调查结果回复的非持久化检索子代理，以及我们提出的持久化检索子代理——该子代理跨轮次维护检索历史，并将检索到的证据直接返回给主代理。我们的方法还允许主代理向同一个子代理请求修正。我们在独立提出的企业问题、共享对话中的问题以及依赖性后续问题上对这三种设计进行了评估。在五个多轮设置中，与基于技能的检索相比，我们的方法记录的代理-模型token用量减少43.3%，预估API成本降低45.0%，同时降低了平均

    arXiv:2611.07899v1 Announce Type: new  Abstract: Conversational agents can retrieve information directly or delegate retrieval to a subagent. This design choice affects how retrieval history is maintained, what context each model call receives, and what information must be re-retrieved in follow-up questions. We compare three designs: skill-based retrieval in the main agent, a non-persistent retrieval subagent that generates a findings response, and our persistent retrieval subagent, which maintains retrieval history across turns and returns retrieved evidence directly to the main agent. Our approach also allows the main agent to request corrections from the same subagent.   We evaluate the three designs on enterprise questions posed independently, within shared conversations, and as dependent follow-ups. Across five multi-turn settings, our approach uses 43.3% fewer recorded agent-model tokens and incurs 45.0% lower estimated API cost than skill-based retrieval, while reducing average
    
[^35]: 推荐系统需要用文字思考吗？面向高效基于LLM推荐的潜在推理

    Do Recommenders Need to Think in Words? Latent Reasoning for Efficient LLM-Based Recommendation

    [https://arxiv.org/abs/2611.07866](https://arxiv.org/abs/2611.07866)

    LatentRec框架用训练时的逐步潜在推理监督替代推理阶段的思维链文本生成，在不损失准确率的前提下大幅降低基于LLM推荐的推理开销。

    

    基于大语言模型（LLM）的推荐系统越来越多地在每次物品预测前生成思维链（CoT）文本，以激活LLM的推理能力，但这在推理阶段每次推荐需要额外消耗数十到数百个token。我们提出一个问题：推理阶段的推理产物（文本形式或潜在形式）能否仅通过训练时的监督来替代，而不损失准确率？为此我们提出LatentRec，一个基于LLM的生成式推荐框架，它用训练时的逐步预测头监督替代推理时的推理产物。LatentRec包含三个组件：一个250万参数的连续思维模块，在训练期间对LLM的隐藏状态迭代K步潜在推理；软token选择机制，用于弥合隐藏状态与输入嵌入之间约14倍的范数差距；逐步推理损失（PRL），通过LLM的预测头对每个中间状态进行监督，目标为语义ID的首个token。经过训练的

    arXiv:2611.07866v1 Announce Type: new  Abstract: LLM-based recommender systems increasingly generate chain-of-thought (CoT) text before each item prediction to activate the LLM's reasoning capacity, costing tens to hundreds of extra tokens per recommendation at inference. We ask whether inference-time reasoning artifacts (textual or latent) can be replaced by training-time supervision alone, without loss of accuracy. We propose LatentRec, a framework for LLM-based generative recommendation that replaces inference-time reasoning artifacts with training-time per-step head supervision. LatentRec has three components. A 2.5M-parameter Continuous Thought Module iterates K latent reasoning steps over the LLM's hidden state during training. Soft Token Selection bridges the ~14x norm gap between hidden states and input embeddings. A Per-step Reasoning Loss (PRL) supervises every intermediate state through the LLM's prediction head against the first token of the target Semantic ID. The trained 
    
[^36]: CORE：面向深度研究的粗粒度到细粒度搜索与上下文管理

    CORE: Coarse-to-Fine Search with Context Management for Deep Research

    [https://arxiv.org/abs/2611.07859](https://arxiv.org/abs/2611.07859)

    CORE是一个无需训练的深度研究框架，通过将粗到细搜索与上下文管理相结合，解决了搜索需要长上下文与返回证据应简短之间的权衡，实现了最先进的性能。

    

    深度研究智能体通过反复搜索来回答复杂问题。它们依赖于一个执行工具调用并将证据返回到其上下文的框架（harness）。这带来了一种权衡：搜索在长段落上效果最佳，因为长段落能为可靠的语义匹配提供足够的上下文；但返回的证据应当简短，因为每条观察结果都会保留在智能体的上下文中，贯穿其后的整个轨迹。我们提出了CORE，一个无需训练的框架，它将粗粒度到细粒度的搜索与上下文管理相结合。CORE使用稠密检索器检索段落，然后复用后期交互（late-interaction）重排序器的词元级相似度，仅返回每个段落中最相关的片段。上下文管理使智能体能够围绕已检索的来源扩展上下文，并抑制重复片段。在与通义DeepResearch和OpenResearcher框架的受控封闭搜索对比中，CORE达到了最先进的性能……

    arXiv:2611.07859v1 Announce Type: cross  Abstract: Deep research agents answer complex questions through repeated search. They rely on a harness that runs their tool calls and returns evidence to their context. This creates a trade-off: Search works best over long passages, which give enough context for reliable semantic matching. But the returned evidence should be short, because every observation stays in the agent's context for the rest of the trajectory. We introduce CORE, a training-free harness that combines coarse-to-fine search with context management. CORE retrieves passages with a dense retriever. It then reuses the token-level similarities from a late-interaction reranker to return only the most relevant snippet from each passage. Context management lets the agent expand the context around sources it has already retrieved and suppresses repeated snippets. In controlled closed-search comparisons with the Tongyi DeepResearch and OpenResearcher harnesses, CORE achieves state-of
    
[^37]: 掩码扩散作为随机序贯吸附：一个精确可解的并行解码模型

    Masked diffusion as random sequential adsorption: an exactly solvable model for parallel decoding

    [https://arxiv.org/abs/2611.07845](https://arxiv.org/abs/2611.07845)

    该论文将掩码扩散模型并行解码的误差与驱动多粒子系统的熵产生和耗散联系起来，并把马尔可夫数据下的生成精确等价为一个可严格求解的随机序贯吸附过程，从而建立了并行解码的首个精确可解理论模型。

    

    掩码扩散模型通过逆转一个随机掩码过程来生成文本。与每次生成一个token的自回归大语言模型不同，它们可以并行揭示多个token，但代价是引入误差，因为同时揭示的token是彼此独立抽取的。我们证明这种权衡可以映射到驱动多粒子系统的统计物理中。(i) 对于任意数据，并行生成的误差等于生成过程相对于逆转掩码过程的熵产生。(ii) 对于马尔可夫数据，生成过程是一个带标记间隙的随机序贯吸附过程，我们对其任意步数均给出了精确解。(iii) 对于多步情况，误差变为缓慢驱动系统的耗散：去掩码调度扮演驱动协议的角色，步数扮演其持续时间的角色，而误差由一个热力学度规决定，我们从互信息中以闭式形式求得了该度规。

    arXiv:2611.07845v1 Announce Type: cross  Abstract: Masked diffusion models generate text by reversing a stochastic masking process. Unlike autoregressive LLMs, which generate text one token at a time, they can reveal many tokens in parallel, at the price of an error, since tokens revealed together are drawn independently. We show that this trade-off maps onto the statistical physics of driven many-particle systems. (i) For any data, the error of parallel generation is the entropy production of the generation process relative to the reversed masking process. (ii) For Markov data, generation is a random sequential adsorption process with labelled gaps, which we solve exactly for any number of steps. (iii) For many steps, the error becomes the dissipation of a slowly driven system: the unmasking schedule plays the role of the driving protocol, the number of steps that of its duration, and the error is set by a thermodynamic metric that we obtain in closed form from the mutual information 
    
[^38]: 生成后排序：学习具身智能体的符号目标解释

    Generate-then-Rank: Learning Symbolic Goal Interpretation for Embodied Agents

    [https://arxiv.org/abs/2611.07825](https://arxiv.org/abs/2611.07825)

    该论文提出两阶段“生成后排序”方法用于具身智能体的符号目标解释，通过监督微调适配输出规范并结合基于错误的候选输出排序（CORE）学习输出偏好，在EAI基准上超越了现有基线模型。

    

    目标解释是将自然语言指令翻译成规划器可执行的正式符号目标。这对具身智能体至关重要，因为该环节的任何错误都会传播到下游的所有规划与行动步骤。要做好这一任务，既需要生成合适的候选结果，也需要在其中进行可靠的选择。我们提出了一种面向具身智能体接口（EAI）中目标解释的两阶段“生成后排序”方法。首先，通过监督微调使大语言模型（LLM）适应EAI的输出规范；其次，基于错误的候选输出排序方法（CORE）从模型自身生成的备选输出中学习对完整输出的偏好。我们在具身智能体接口基准的家庭场景上进行了实验，评估了多个开源大语言模型，包括DeepSeek-V3 chat、DeepSeek-Reasoner以及Qwen2.5-7B和Qwen2.5-72B。所提出的方法超越了包括……在内的基线模型。

    arXiv:2611.07825v1 Announce Type: new  Abstract: Goal interpretation is to translate a natural language instruction into a formal symbolic goal that a planner can execute. It is critical for embodied agents because any error here propagates to every downstream planning and action step. Doing it well requires both generating suitable candidates and reliably selecting among them. We present a two-stage generate-then-rank approach to Goal Interpretation in the Embodied Agent Interface (EAI). First, supervised fine-tuning adapts a Large Language Model (LLM) to EAI's output convention. Second, Candidate-Output Ranking from Errors (CORE) learns preferences over complete outputs from the model's own generated alternatives. We conducted experiments on the household scenes of the Embodied Agent Interface Benchmark. Multiple open source LLMs including DeepSeek-V3 chat, DeepSeek-Reasoner, and Qwen2.5-7B and Qwen2.5-72B were evaluated. The proposed method outperformed the baseline models including
    
[^39]: Axis-LoRA：面向个人记忆检索的嵌入模型探测与适配

    Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval

    [https://arxiv.org/abs/2611.07820](https://arxiv.org/abs/2611.07820)

    该论文发现冻结的稠密嵌入模型无法按时间、地点、人物、事件四个维度有效区分个人记忆，并提出 Axis-LoRA——在嵌入模型内部为每个维度设置 LoRA 分支并通过逐查询门控融合的适配方法，仅训练少量参数即可在三个记忆检索基准上显著提升性能。

    

    随着大语言模型越来越多地被问及有关用户个人经历的问题，检索系统必须能够按时间、地点、人物和事件四个维度来区分记忆。当前的个人记忆处理流水线依赖冻结的稠密嵌入模型来嵌入记忆以供检索，但我们发现这些模型并不能很好地捕捉这四个维度：更换一条记忆的日期或地点几乎不会改变其嵌入向量，而在冻结嵌入之上训练的投影层也无法恢复缺失的信号。问题出在模型内部，其中每个维度由一种独特的注意力头模式进行编码。基于这些发现，我们提出了 Axis-LoRA，一种按维度设计的适配器，它在嵌入模型内部放置四个 LoRA 分支，并通过逐查询门控机制将它们组合起来。在 LongMemEval、ATM-Bench 和 LoCoMo 上对三个标准嵌入模型进行评估，Axis-LoRA 将四个基准的平均 R@10 提升了 1.28 至 1.56 个百分点，同时仅训练基础参数的一小部分。

    arXiv:2611.07820v1 Announce Type: cross  Abstract: As LLMs are increasingly asked questions about a user's personal experiences, retrieval must distinguish memories by time, place, person, and event. Current personal-memory pipelines rely on frozen dense embedding models to embed memories for retrieval, but we find that these models do not capture the four axes well. Swapping a memory's date or place barely shifts its embedding, and a trained projection on top of the frozen embedding does not recover the missing signal. The problem lives inside the model, where each axis is encoded by a distinct attention-head pattern. Guided by these findings, we propose Axis-LoRA, a per-axis adapter that places four LoRA branches inside the embedding model and combines them via a per-query gate. Across three standard embedding models evaluated on LongMemEval, ATM-Bench, and LoCoMo, Axis-LoRA improves 4-benchmark mean R@10 by $+1.28$ to $+1.56$~pp while training only a small fraction of the base param
    
[^40]: 可解码性不等于有效性：评估语言模型概念探针的干扰因素敏感性

    Decodability Is Not Validity: Evaluating Nuisance Sensitivity in Language Model Concept Probes

    [https://arxiv.org/abs/2611.07805](https://arxiv.org/abs/2611.07805)

    论文提出用“判别性证据”来评估概念探针的干扰因素敏感性，并证明探针的可解码性（预测准确率）并非有效性的可靠代理——AUROC 统计等价的探针对相关干扰因素的敏感性可相差数倍，且两者秩相关性在不同设置间会反转符号。

    

    两个概念探针在从语言模型的隐藏状态中预测某一概念时可能具有同样的准确率，但它们的评分中却可能包含不同比例的目标概念与相关干扰属性。在四种“目标—干扰因素”设置（CEBaB 中的两个方面情感；BiasBios 中的职业与性别；亚马逊评论情感结合否定词；以及结合第一人称框架）、两个大语言模型以及一组概念提取器中，我们将可解码性与判别性证据进行比较：即在固定目标的情况下评分对干扰因素的依赖程度，以及评分对旨在仅改变干扰因素的输入编辑的响应——这些编辑经两个大语言模型评判者过滤，并与安慰剂编辑进行对比。在相关训练数据的条件下，在留出集 AUROC 上统计等价的概念提取器，在四个设置中的三个里对干扰因素的敏感性可相差数倍，而且可解码性并不是干扰因素敏感性的可靠代理指标：两者的秩相关系数在不同设置间甚至会改变符号。

    arXiv:2611.07805v1 Announce Type: new  Abstract: Two concept probes can be equally accurate at predicting a concept from a language model's hidden states while their scores contain different mixtures of the target concept and a correlated nuisance attribute. In four target--nuisance settings (two aspect sentiments in CEBaB; occupation and gender in BiasBios; Amazon review sentiment with negation, and with first-person framing), two LLMs, and a population of concept extractors, we compare decodability with discriminant evidence: the score's dependence on the nuisance at fixed target, and its response to input edits intended to change only the nuisance, filtered by two LLM judges and compared against placebo edits. Under correlated training data, extractors that are statistically equivalent in held-out AUROC differ up to several-fold in nuisance sensitivity in three of the four settings, and decodability is not a reliable proxy for it: their rank correlation changes sign across settings.
    
[^41]: 面向德英翻译的NLLB数据集收集与微调

    Dataset Collection and Fine-Tuning of NLLB for German--English Translation

    [https://arxiv.org/abs/2611.07797](https://arxiv.org/abs/2611.07797)

    本文构建了高质量德英平行语料库并在消费级GPU上微调NLLB-600M模型，使文学与技术翻译质量获得显著提升（BLEU +3.4），并集成到开源文档翻译系统中实现了保持格式的实际部署。

    

    本文介绍了从OPUS Books中筛选高质量平行语料的过程，以及对Meta的NLLB-200模型进行德英翻译微调的工作。我们详细描述了一个数据集提取流程，通过编码验证、长度过滤、语言识别、去重和质量评分，将45k原始OPUS句子对过滤为25k高质量样本。我们提出了一种面向消费级GPU优化的可复现微调方法，采用混合精度、梯度累积和梯度检查点等技术。我们微调后的NLLB-600M模型在文学类和技术类德英翻译任务上相比基础模型取得了统计显著的提升（BLEU +3.4，chrF++ +2.6，TER -3.2）。我们通过将微调后的模型集成到现有的开源文档翻译系统LinguaHaru中，展示了实际部署能力，实现了对PDF和Office文档的保持原格式翻译。

    arXiv:2611.07797v1 Announce Type: new  Abstract: This paper describes the curation of a high-quality parallel corpus from OPUS Books and the fine-tuning of Meta's NLLB-200 models for German--English translation. We detail a dataset extraction pipeline that filters 45k raw OPUS sentence pairs to 25k high-quality examples through encoding validation, length filtering, language identification, deduplication, and quality scoring. We present a reproducible fine-tuning methodology optimized for consumer GPUs using mixed precision, gradient accumulation, and gradient checkpointing. Our fine-tuned NLLB-600M model achieves statistically significant improvements (+3.4 BLEU, +2.6 chrF++, -3.2 TER) over the base model on literary and technical German-English translation. We demonstrate practical deployment by integrating the fine-tuned model into LinguaHaru, an existing open-source document translation system, achieving format-preserving translation across PDF and Office documents. All dataset pre
    
[^42]: 连接结构与语义：基于三元组到文本对齐的知识图谱验证

    Bridging Structure and Semantics: Knowledge Graph Verification via Triple-to-Text Alignment

    [https://arxiv.org/abs/2611.07780](https://arxiv.org/abs/2611.07780)

    提出GenVer生成式验证器，通过对齐三元组及其生成的自然语言表示来融合图结构信息与预训练语言模型知识，在FactFB数据集上取得了最高达91.52%的三元组分类准确率。

    

    语义准确性是知识图谱质量的一个重要维度。基于图的评分方法利用关系结构，而文本表示则使分类器能够利用预训练语言模型（PLM）中编码的知识。我们提出了GenVer，一种生成式验证器，它利用三元组的序列化形式和生成的自然语言表示（NLR）来对三元组进行评分。通过三元组-文本对齐程序为序列到序列生成器选择训练样本对，然后由基于BERT的分类器将每个三元组及其生成的NLR的特征相结合。在从FB15k通过扰动负三元组构建的四个FactFB变体上，GenVer在评估的基线方法中取得了最高的分类准确率，范围从88.56%到91.52%。实验表明，生成的表示有助于三元组分类，同时案例分析也揭示了语义漂移和高置信度错误的存在。

    arXiv:2611.07780v1 Announce Type: cross  Abstract: Semantic accuracy is an important dimension of knowledge graph quality. Graph-based scoring methods exploit relational structure, while textual representations allow classifiers to draw on knowledge encoded by pre-trained language models (PLMs). We present GenVer, a generative verifier that scores triples using both their serialized form and generated natural language representations (NLRs). A triple-text alignment procedure selects training pairs for a sequence-to-sequence generator; a BERT-based classifier then combines features from each triple and its generated NLR. On four FactFB variants, constructed from FB15k using perturbed negative triples, GenVer obtains the highest classification accuracy among the evaluated baselines, ranging from 88.56% to 91.52%. The experiments suggest that generated representations can help triple classification, while the case analysis also exposes semantic drift and high-confidence errors.
    
[^43]: SAE（稀疏自编码器）训练数据的缩放定律

    Scaling Laws for SAE Training Data

    [https://arxiv.org/abs/2611.07756](https://arxiv.org/abs/2611.07756)

    在固定训练预算下，SAE 所需的独特激活数据量存在饱和点——缓冲区达到一定规模后，重放数据与新鲜激活值效果相当，且缓冲区无需随训练预算扩大；作者据此提出并在多种模型、层级和 SAE 架构上验证了一条刻画多样性与重复权衡的有限数据缩放定律。

    

    激活值存储通常是稀疏自编码器（SAE）训练中的瓶颈。我们研究了在总训练预算固定的情况下，SAE 需要多少条独特的激活值。随着缓冲区规模增大，质量迅速提升，随后进入一个平台期：在此阶段重放缓冲区数据与在整个训练过程中持续使用新鲜激活值几乎同样有效。超过这一点后，额外的独特激活值对重构效果或可解释性几乎没有额外收益，因此缓冲区规模无需随训练预算增长。我们用一条针对有限数据的缩放定律来描述这种多样性与重复之间的权衡。我们在不同的字典规模与 token 预算、Llama-3.1 的早期、中期和后期层、Qwen3 的不同模型规模，以及 BatchTopK、TopK 和 JumpReLU 等多种 SAE 上验证了该定律。通过在数学文本和 Python 代码上进行扫描，我们测试了改变激活值来源的影响。评估涵盖解释方差（EV）、重构 MSE、特征吸收、自动化（评估）……

    arXiv:2611.07756v1 Announce Type: cross  Abstract: Activation storage is often a bottleneck in sparse autoencoder (SAE) training. We study how many unique activations an SAE needs when its total training budget is fixed. Quality improves rapidly as the buffer grows, then reaches a plateau where replaying the buffer is nearly as effective as using fresh activations throughout training. Beyond this point, additional unique activations bring little benefit to reconstruction or interpretability, so the buffer need not grow with the training budget. We describe this tradeoff between diversity and repetition with a scaling law for limited data. We validate it across dictionary sizes and token budgets, early, middle, and late layers of Llama-3.1, Qwen3 model sizes, and BatchTopK, TopK, and JumpReLU SAEs. Sweeps using mathematical text and Python code test the effect of changing the activation source. Evaluation covers explained variance (EV), reconstruction MSE, feature absorption, automated 
    
[^44]: 自动化视频转问答流水线：基于大型语言模型的端到端问题生成与验证

    Automated Video-to-QA Pipeline: End-to-End Question Generation and Verification Using Large Language Models

    [https://arxiv.org/abs/2611.07755](https://arxiv.org/abs/2611.07755)

    该论文提出了一种端到端自动化流水线，通过级联整合FFmpeg、Whisper、BART和Llama-3.1-8B-Instruct等模型，从视频中提取结构化摘要并生成问答对，其核心创新在于引入自我验证与微调循环，让模型自我评估并保留高质量问答对。

    

    YouTube等平台上教育和信息类视频内容的爆炸式增长，为高效的知识检索与评估带来了挑战。本文提出了一种端到端的自动化流水线，用于从视频流中提取结构化摘要并生成高保真度的问答对。该框架集成了多个最先进模型的级联：音频提取通过FFmpeg处理，随后使用Whisper（一个Python库）进行自动语音识别（ASR）以生成准确的转录文本。由于大型语言模型（LLM）的token限制，转录文本被分段并使用Facebook的BART进行摘要。本工作的一项关键创新是自我验证与微调循环。我们利用meta-llama/Llama-3.1-8B-Instruct从转录文本片段中生成候选问答对，然后将其反馈给模型，以严格评估其相关性和重要性。仅经验证的问答对……

    arXiv:2611.07755v1 Announce Type: new  Abstract: The exponential growth of educational and informational video content on platforms like YouTube presents a challenge for efficient knowledge retrieval and assessment. This paper proposes an end-to-end automated pipeline for extracting structured summaries and generating high-fidelity Question-Answer (QA) pairs from video streams. The framework integrates a cascade of state-of-the-art models: audio extraction is handled via FFmpeg, followed by automatic speech recognition (ASR) using Whisper (A library in Python) to generate accurate transcripts. Due to the token constraints of Large Language Models (LLMs), transcripts are segmented and summarized using Facebook's BART. A key innovation of this work is a Self-Verification and Fine-Tuning Loop. We utilize meta-llama/Llama-3.1-8BInstruct to generate candidate QA pairs from transcript chunks, which are then fed back into the model to rigorously assess their relevance and importance. Only ver
    
[^45]: 通过循环机制扩展类型化决策模型的推理能力

    Extending Reasoning in Typed Decision Models via Looping

    [https://arxiv.org/abs/2611.07736](https://arxiv.org/abs/2611.07736)

    该论文提出 RELOAD 方法，通过对预训练模型的中间块进行阻尼循环并由轻量级控制器自适应选择循环次数，首次证明循环机制在与决策模型联合训练时能显著提升类型化决策模型的准确性与校准度，尤其在需要长推理链的问题上编码器最多可提升 23 分。

    

    类型化决策模型在单次前向传播中从固定的答案选项集合中进行选择，因此与生成式语言模型不同，它们无法通过生成更多 token 来在困难输入上投入更多的测试时计算量。对一层块进行循环可以在不增加参数的情况下在潜空间中提供这种额外计算，但先前的工作发现，将循环改装到按常规方式预训练的模型中并未带来收益。我们证明，当这种循环与决策模型一起训练时才能发挥作用。我们的方法 RELOAD（用于自适应决策的改装循环）对预训练的编码器或解码器进行微调，在其中间块上采用阻尼循环，并由一个轻量级控制器为每个输入自适应地选择循环次数。在从 0.4B 到 4B 参数的六个模型上，循环机制提升了准确性和校准度，提升主要体现在那些需要比训练中见过的任何问题都更长的推理链的问题上，编码器在此类问题上最多可获得 23 分的提升。无论是匹配的训练计算量还是……（原文摘要被截断）

    arXiv:2611.07736v1 Announce Type: cross  Abstract: Typed decision models choose among a fixed set of answer options in a single forward pass, so unlike generative language models they cannot spend more test-time compute on hard inputs by generating more tokens. Looping a block of layers offers this compute in latent space without adding parameters, but prior work found no gain from loops retrofitted into conventionally pretrained models. We show that such loops help when trained together with the decision model. Our method, RELOAD (retrofitted loops for adaptive decisions), fine-tunes a pretrained encoder or decoder with a damped loop over its middle block and a lightweight controller picks the number of passes adaptively for each input. Across six models from 0.4B to 4B parameters, looping improves both accuracy and calibration, mostly on problems that need longer chains of reasoning than any seen in training, where encoders gain up to 23 points. Neither matched training compute nor m
    
[^46]: 评估Token概率作为人类行为模拟中的置信度估计器：以卢森堡语下一次点赞预测为例的案例研究

    Evaluating Token Probabilities as Confidence Estimators in Human Behavior Simulations: A Case Study in Luxembourgish Next Like Prediction

    [https://arxiv.org/abs/2611.07735](https://arxiv.org/abs/2611.07735)

    本研究以卢森堡新闻平台的点赞预测任务为例，发现LLM的token概率作为置信度估计时模型过度自信、被简单统计基线全面超越，且准确率与校准度并不同步——扩大模型规模能提升准确率却损害校准度，表明评估人类行为模拟时校准度是不可忽视的关键指标。

    

    随着大型语言模型（LLM）越来越多地被用于模拟人类行为，其操作有效性通常仅通过准确率来评估，这隐含地将人类行为视为确定性的。我们认为，在嘈杂的数字痕迹数据上，模型的性能不仅取决于预测出正确的结果，还取决于其校准程度。我们开展了一项案例研究，测试开放权重LLM的token概率能否在“下一次点赞预测”任务中作为置信度估计：即在用户先前交互行为的条件下，预测卢森堡新闻平台RTL.lu的用户会如何对评论进行评分。通过评估Ministral 3（3B、8B），我们发现仅使用上下文学习时，被测模型表现出过度自信，且在准确率和校准度上均被简单的统计基线所超越。准确率与校准度并不同步变化：从3B扩展到8B提升了准确率却降低了校准度，而更长的行为……（摘要原文在此处被截断）

    arXiv:2611.07735v1 Announce Type: cross  Abstract: As large language models (LLMs) are increasingly used to simulate human behavior, their operational validity is typically assessed through accuracy alone, implicitly treating human behavior as deterministic. We argue that on noisy digital trace data, a model's performance depends not only on predicting the correct outcome but on its calibration. We present a case study testing whether the token probabilities of open-weight LLMs can serve as confidence estimates in a Next Like Prediction task: predicting how a user of the Luxembourgish news platform RTL.lu rates a comment, conditioned on their prior interactions. Evaluating Ministral 3 (3B, 8B), we find that, with in-context learning only, the tested models are overconfident and are outperformed on accuracy and calibration by a simple statistical baseline. Accuracy and calibration do not move together: scaling from 3B to 8B improves accuracy while degrading calibration, and longer behav
    
[^47]: O$^2$PD：基于可靠前缀的重叠门控在线策略蒸馏

    O$^2$PD: Overlap-Gated On-Policy Distillation on Reliable Prefix

    [https://arxiv.org/abs/2611.07731](https://arxiv.org/abs/2611.07731)

    提出 O$^2$PD 方法，通过定量分析教师监督的可靠性来确定“对齐哪些 token”（可靠前缀），并以重叠门控机制决定“如何对齐”，从而弥补现有 top-$k$ 在线策略蒸馏在监督可靠性与概率质量丢弃方面的不足。

    

    在线策略蒸馏（OPD）通过在学生自身生成的 rollout 上将学生对齐到教师的下一 token 分布，从而提供密集的 token 级监督。现有工作大多实现 top-$k$ OPD，即在教师 top-$k$ 支持集上，对教师与学生的重归一化分布应用反向 Kullback-Leibler（RKL）散度。然而，教师所打分的学生生成的 rollout 对教师自身而言天然是离策略的，其下一 token 引导的可靠性并不明确。将散度限制在教师 top-$k$ 支持集上，也会丢弃该支持集之外的概率质量。这些缺陷隐含了两个问题：1）对齐哪些 token；2）如何对齐这些 token。为回答这些问题，我们提出重叠门控在线策略蒸馏（O$^2$PD）。针对问题 1），我们定量分析了教师监督的可靠性，并发现对学生轨迹上的 token 进行对齐……（原文摘要在此处被截断）

    arXiv:2611.07731v1 Announce Type: cross  Abstract: On-policy distillation (OPD) supplies dense token-level supervision by aligning a student to the teacher's next-token distribution on its own generated rollouts. Existing works mostly implement top-$k$ OPD, applying reverse Kullback-Leibler ($\RKL$) divergence between the teacher's and the student's renormalized distributions over teacher top-$k$ support. However, the teacher scores student-generated rollouts that are inherently off-policy for itself, leaving the reliability of its next-token guidance unclear. Restricting the divergence to the teacher top-$k$ support also discards the remaining mass outside that support. These gaps leave two questions implicit: 1) which tokens to align, and 2) how to align those tokens. To answer these questions, we propose Overlap-gated On-Policy Distillation (O$^{2}$PD). As for 1), we quantitatively analyze the reliability of teacher supervision and find that alignment on the student's trajectory pre
    
[^48]: 基于词元身份平方监督的连续语言流继续训练方法

    Continued Training of Continuous Language Flows with Token Identity Squared Supervision

    [https://arxiv.org/abs/2611.07721](https://arxiv.org/abs/2611.07721)

    提出词元身份Brier流，通过直接对词元身份施加平方监督（Brier损失）解决了连续MSE目标无法区分不同概率分布的问题，从而更有效地继续训练预训练语言流模型。

    

    扩散模型和流模型最近被扩展到语言建模领域，通过在词元嵌入空间中进行连续去噪来生成文本。预训练模型提供了词元表示和上下文预测能力，但要在有限额外训练的情况下改进生成效果，需要在监督信号与连续生成之间建立适当的连接。连续MSE评估的是通过用预测概率对词元嵌入加权所得到的预测向量；不同的概率分布可以产生相同的向量，因此在该目标函数下无法被区分。我们提出词元身份Brier流（TIBF），在继续训练预训练语言流模型时，直接对词元身份应用平方监督。TIBF在高斯观测模型下将网络得分与词元匹配得分相结合，使用Brier损失对词元概率进行监督，并使用概率加权嵌入进行……

    arXiv:2611.07721v1 Announce Type: cross  Abstract: Diffusion and flow models have recently been extended to language modeling, generating text through continuous denoising in token embedding spaces. Pretrained models provide token representations and contextual prediction capabilities, but improving generation with limited additional training requires an appropriate connection between supervision and continuous generation. Continuous MSE evaluates a predicted vector obtained by weighting token embeddings with predicted probabilities; different probability distributions can yield the same vector and are therefore indistinguishable under this objective. We propose Token Identity Brier Flow (TIBF), which applies squared supervision directly to token identities when continuing pretrained language flows. TIBF combines network scores with token-matching scores under a Gaussian observation model, supervises token probabilities using the Brier loss, and uses probability-weighted embeddings for
    
[^49]: 专家剪枝混合专家语言模型中英语校准对立陶宛语能力的侵蚀

    English Calibration Erodes Lithuanian in Expert-Pruned Mixture-of-Experts Language Models

    [https://arxiv.org/abs/2611.07654](https://arxiv.org/abs/2611.07654)

    该研究发现英语校准在专家剪枝的MoE模型中无法迁移到立陶宛语——25%剪枝率下英语理解仅下降1.7分而立陶宛语下降21.4分，而立陶宛语校准却能同时保护两种语言的性能。

    

    专家剪枝通过移除被校准集判定为不重要的专家来压缩混合专家语言模型。针对高资源语言的研究大多发现，英语校准可以很好地迁移到其他语言。我们在立陶宛语上检验了这一结论：我们使用REAP方法对四个MoE模型进行剪枝，分别基于英语、立陶宛语或混合校准集移除25%或50%的专家，并在阅读理解、翻译和语言建模任务上对两种语言进行评估。结果表明，英语校准无法迁移到立陶宛语：在25%剪枝率下，英语理解能力中位数下降1.7分，而立陶宛语下降21.4分。这种迁移是不对称的：立陶宛语校准消除了大部分立陶宛语性能损失，且从未显著损害英语理解能力（平均比英语校准高出2.5分）；仅有英语语言建模性能略有下降。路由分析揭示了其中原因：在模型最后几层中，立陶宛语将55……

    arXiv:2611.07654v1 Announce Type: new  Abstract: Expert pruning compresses Mixture-of-Experts (MoE) language models by removing the experts that a calibration set marks as unimportant. Studies of high-resource languages largely find that English calibration transfers well to other languages. We test this for Lithuanian: we pruned four MoE models with REAP, removing 25% or 50% of the experts with English, Lithuanian or mixed calibration, and evaluated both languages on reading comprehension, translation and language modelling. English calibration does not transfer to Lithuanian: at 25% pruning, English comprehension fell by a median of 1.7 points but Lithuanian by 21.4. The transfer is asymmetric. Lithuanian calibration removed most of the Lithuanian loss and never significantly hurt English comprehension, which was on average 2.5 points better than under English calibration; only English language modeling got slightly worse. Routing suggests why: in the last layers, Lithuanian sends 55
    
[^50]: LoRA微调的嵌入空间指纹：仅凭模型产物恢复训练数据领域

    Embedding-Space Fingerprinting of LoRA Fine-Tunes: Recovering the Training-Data Domain from the Model Artifact Alone

    [https://arxiv.org/abs/2611.07610](https://arxiv.org/abs/2611.07610)

    本文提出嵌入空间指纹方法，证明无需接触训练数据，仅凭LoRA微调模型的产物（权重增量、内部激活或生成文本）即可识别其训练数据的领域，其中灰盒激活特征在留一数据集协议下达到61%的准确率，远高于20%的随机水平。

    

    当基础语言模型通过低秩适配（LoRA）进行微调时，训练数据的领域信息会有多少泄露到最终产物中？本文研究一个简单的分类器能否在从未观察训练数据的情况下，仅凭模型产物（权重增量、固定探针集上的内部激活，或生成文本）恢复微调模型的所属领域。研究使用一个基础模型（Qwen3.5-0.8B），在五个领域（医学、代码、数学、非英语文本、叙事散文）的18个公开数据集上生成了36个LoRA微调模型，并在访问权限递减的三个层级上提取了三类特征：白盒权重、灰盒激活和黑盒输出。采用在任何结果产生之前就已冻结的留一数据集（LODO）协议，测试能否识别在先前未见过的数据集上训练的微调模型的领域。结果表明，领域信息可以通过灰盒激活以远高于随机水平（20%）的准确率（61%）恢复（摘要在此处被截断）。

    arXiv:2611.07610v1 Announce Type: new  Abstract: When a base language model is fine-tuned with Low-Rank Adaptation (LoRA), how much of the training data domain leaks into the resulting artifact? This paper studies whether a simple classifier can recover the domain of a fine-tune given only the model artifact (weight deltas, internal activations on a fixed probe set, or generated text) without ever observing the training data. Using one base model (Qwen3.5-0.8B), 36 LoRA fine-tunes are produced over 18 public datasets in five domains (medical, code, mathematics, non-English text, narrative prose), and three feature families are extracted at decreasing levels of access: white-box weights, gray-box activations, and black-box outputs. A leave-one-dataset-out (LODO) protocol, frozen before any result was seen, tests whether the domain of a fine-tune trained on a previously unseen dataset can be identified. The domain is recoverable well above chance (20%) from gray-box activations (61%) and
    
[^51]: 针对障碍语音微调转换器模型：收益出现在何处及其对评估的意义

    Fine-Tuning a Transducer for Impaired Speech: Where the Gains Appear and What They Mean for Evaluation

    [https://arxiv.org/abs/2611.07587](https://arxiv.org/abs/2611.07587)

    该论文在SAPC2挑战赛数据上微调公开语音识别器，将障碍语音识别的字符错误率从15.26%基线降至7.81%，并揭示微调收益最大来自现成模型最难理解的说话人，而训练中见过的句子的错误率显著低于新句子，且这一差距大于说话人熟悉度带来的影响。

    

    语音识别对许多因肌萎缩侧索硬化症（ALS）、帕金森病、脑瘫、唐氏综合征或中风等疾病而影响发音的人群仍然效果不佳。我们使用一个在挑战赛数据上微调过的公开语音识别器参加了语音无障碍项目挑战赛2（SAPC2）。我们最好的参赛模型在两个隐藏测试集中的第一个Test1上取得了7.81%的字符错误率（官方基线为15.26%）。我们的实时参赛模型在CPU上运行，组合中位延迟约为0.9秒，错误率为7.96%-8.07%。对于新说话人，微调对现成识别器最难理解的说话人降低错误的效果最为显著。训练中已见过的句子比新句子的错误率低得多（2.58%对6.42%），这一差异大于说话人熟悉程度所带来的差异。我们最后以评估此类系统的经验教训作结。

    arXiv:2611.07587v1 Announce Type: new  Abstract: Speech recognition still works poorly for many people whose speech is affected by conditions such as ALS, Parkinson's disease, cerebral palsy, Down syndrome or stroke. We took part in the Speech Accessibility Project Challenge 2 (SAPC2) with a public speech recognizer that we fine-tuned on the challenge data. Our best entry scored 7.81% character error rate on Test1, the first of two hidden test sets (official baseline 15.26%). Our live entries ran on CPUs with a combined median latency of about 0.9 seconds at 7.96-8.07% error. On new speakers, fine-tuning reduced errors most for the speakers that an off-the-shelf recognizer finds hardest to understand. Sentences already seen in training had far lower error rates than new ones (2.58% vs 6.42%), a larger difference than being a familiar speaker made. We close with lessons for evaluating such systems.
    
[^52]: 针对隐蔽失对齐行为的思维链监控压力测试

    Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment

    [https://arxiv.org/abs/2611.07586](https://arxiv.org/abs/2611.07586)

    提出自动化红队测试框架 TraceMRT，通过将副任务重构为主任务的连贯子目标并搜索轨迹塑形策略，系统性地压力测试了思维链监控对会隐蔽掩盖失对齐行为的智能体的鲁棒性。

    

    随着大语言模型（LLM）智能体在自主性方面快速提升，智能体出现绕过安全约束的隐藏目标的风险日益增加。思维链（CoT）监控提供了一种关键的控制机制：通过读取智能体的推理过程，在失对齐行为付诸实施之前将其标记出来。然而其可靠性取决于对它进行压力测试的红队攻击的真实程度。现有的红队测试工作依赖于共享模板，例如将副任务伪装为完成主任务的前置条件，这种做法将副目标视为外生插入，而对任务特定的、有上下文依据的规避策略则探索不足。因此我们提出这样一个问题：面对会战略性隐藏自身失对齐行为的智能体，CoT 监控究竟有多强的鲁棒性？为了回答这一问题，我们提出了 TraceMRT——一个自动化的轨迹级（Trace-level）监控红队测试框架，它将副任务重新构建为主任务的一个连贯子目标，并搜索轨迹塑形策略，使其诱导出的轨迹能够隐蔽地掩盖失对齐意图……

    arXiv:2611.07586v1 Announce Type: cross  Abstract: As LLM agents advance rapidly in autonomy, the risk of hidden objectives that bypass safety constraints becomes increasingly plausible. Chain-of-thought (CoT) monitoring offers a key control mechanism by reading the agent's reasoning and flagging misalignment before it materializes into action, yet its reliability depends on the realism of the red-team it is stress-tested against. Existing red-teaming efforts rely on shared templates, such as casting a side task as a prerequisite, which treat the side objective as an exogenous insertion and leave task-specific, contextually grounded evasions underexplored. We therefore ask: How robust is CoT monitoring against agents that strategically conceal misalignment? To answer this, we introduce TraceMRT, an automated Trace-level Monitor Red-Teaming framework that reframes the side task as a coherent sub-goal of the primary task and searches trace-shaping strategies whose induced trajectories co
    
[^53]: SKIM：通过选择性知识引导掩码对大语言模型智能体进行剪枝

    SKIM: Pruning Large Language Model Agents via Selective Knowledge Informed Masking

    [https://arxiv.org/abs/2611.07558](https://arxiv.org/abs/2611.07558)

    提出SKIM——首个面向结构化生成场景的大语言模型智能体剪枝与蒸馏框架，通过将词元位置划分为上下文、决策和格式三类，在剪枝与蒸馏两个阶段中优先保留与决策位置最相关的知识。

    

    大语言模型（LLM）智能体越来越多地在结构化生成场景下部署，其中语法约束在推理时强制执行输出格式。然而现有的压缩方法均针对无约束生成，我们首次发现了一个根本性的不匹配：在结构化生成下，语法约束会将模型的输出分布缩减至有效词元，而对数几率在决定智能体动作的少量关键位置上最为重要。受此发现启发，我们提出了SKIM（选择性知识引导掩码），这是首个面向结构化生成下大语言模型智能体的剪枝与蒸馏框架，其两个阶段均由词元角色引导。SKIM将词元位置划分为上下文、决策和格式三类，这一划分驱动了两个互补的方向：（1）类别感知的剪枝显著性分数，以保留与决策位置最相关的参数；（2）一种在线策略蒸馏目标……

    arXiv:2611.07558v1 Announce Type: cross  Abstract: Large Language Model (LLM) agents are increasingly deployed under structured generation, where grammar constraints enforce output format at inference time. Whereas existing compression methods target unconstrained generation, we first identify a fundamental mismatch: under structured generation, grammar constraints reduce the model's output distribution to valid tokens, and the logits matter most at the small set of positions that decide the agent's action. Motivated by this finding, we propose SKIM (Selective Knowledge-Informed Masking), the first pruning and distillation framework for LLM agents under structured generation, with both stages guided by token role. SKIM partitions token positions into context, decision, and format categories, and this partition drives two complementary axes: (1) category-aware pruning saliency scores that preserve parameters most relevant for decision positions; and (2) an on-policy distillation objecti
    
[^54]: 记忆陷阱：语言智能体会从良性交互中过度泛化

    The Memory Trap: Language Agents Overgeneralize from Benign Interactions

    [https://arxiv.org/abs/2611.07554](https://arxiv.org/abs/2611.07554)

    本文发现并命名了“记忆陷阱”这一新型失效模式：语言智能体会从良性交互记忆中过度泛化，例如低风险任务中跳过确认的经历会使其在高风险任务中也跳过安全验证，而情感支持或被批评经历形成的记忆则会使后续判断偏向谄媚或过度批评。

    

    语言智能体越来越多地使用持久化记忆来跨会话记录用户偏好和操作流程。我们证明，来自良性交互的记忆可能导致智能体在后续任务中出现不安全或不可靠的行为。我们将这种失效现象称为“记忆陷阱”。为研究这一问题，我们比较了六个前沿模型（包括 Claude Sonnet、Opus、Fable 模型和 GPT-5.6）在有记忆和无记忆会话中的行为差异，并在多种设置下对这些模型进行了评估。首先，在自主性实验中，我们发现用户要求在低风险编程杂务中跳过确认的请求，会导致智能体在编程和日常生活的模拟高风险任务中也跳过确认与验证。其次，我们研究了当用户寻求情感支持、或描述不加批判的反馈所造成的伤害时形成的记忆，发现这些记忆会使智能体在后续对数学作业、论证和诗歌的评价中分别偏向谄媚或过度批评。

    arXiv:2611.07554v1 Announce Type: cross  Abstract: Language agents increasingly use persistent memory to record user preferences and procedures across sessions. We show that memory from benign interactions can cause unsafe or unreliable behavior on later tasks. We call this failure the memory trap. To study this issue, we compare behavior in sessions with and without memory across six frontier models (including Claude Sonnet, Opus, and Fable models and GPT-5.6). We evaluate these models in a range of settings. First, in autonomy experiments, we find that requests to skip confirmation during low-stakes coding chores lead agents to skip confirmation and verification on simulated high-stakes tasks in both coding and everyday life. We next study memories formed when users seek emotional support, or describe harm from uncritical feedback, and find these memories bias later judgments of mathematical work, arguments, and poems toward sycophancy or over-critique (respectively). Memory notes wr
    
[^55]: ReviewMap：绘制同行评审评论中的评估行为与自动化边界

    ReviewMap: Mapping Evaluative Acts and Automation Boundaries in Peer-Review Comments

    [https://arxiv.org/abs/2611.07489](https://arxiv.org/abs/2611.07489)

    该论文提出 ReviewMap 框架，将同行评审评论区分为“主张与学术标准评估”和“贡献价值与会议契合度校准”两类行为，证明行为层可通过多模型 LLM 共识可靠地自动化（κ≥0.75），而属性层仍需人工监督，从而为 NLP 在同行评审中确立了比全面自动化更务实、更诚实的定位。

    

    同行评审评论既要核查论文做了什么，又要评判其为何重要，但目前尚无现有方案能够区分这两种评估行为。我们认为，使这些行为显性化并衡量大语言模型（LLM）识别它们的可靠程度，是负责任地开展 LLM 辅助评审分析的前提：人类需要能够审查评审判断及其依据。这为 NLP 在同行评审中确立了一个比全面自动化更诚实的使命。我们提出了 ReviewMap，该框架将每条评论分类为对论文主张与学术标准的评估，或对贡献价值与会议契合度的校准，并根据行为类型有条件地应用更细粒度的属性。通过多模型 LLM 共识将该框架应用于 500 篇 ICLR 2026 论文，结果表明行为层在操作准则下（模型间 κ ≥ 0.75）可被可靠地自动化，并得到对 470 条开发评论的五人人工验证支持，而属性层则需要人工监督……

    arXiv:2611.07489v1 Announce Type: new  Abstract: Peer-review comments check what a paper does and judge why it matters, but no existing scheme distinguishes these two evaluative acts. We argue that making these acts explicit and measuring how reliably LLMs identify them is a prerequisite for responsible LLM-assisted review analysis: humans need to inspect review judgments and their grounds. This defines a more honest mission for NLP in peer review than full automation. We present ReviewMap, a framework that classifies each comment as assessment of claims and scholarly standards or calibration of contribution value and venue fit, with finer-grained attributes applied conditionally by act type. Applied to 500 ICLR~2026 papers through multi-model LLM consensus, the act layer proves reliably automatable under the operational guideline (inter-model $\kappa \geq 0.75$), supported by five-human validation on 470 development comments, while the attribute layer requires human oversight, definin
    
[^56]: MEvince：面向恶意软件报告技术识别的知识约束大语言模型推理

    MEvince: Knowledge-Constrained LLM Reasoning for Technique Identification from Malware Reports

    [https://arxiv.org/abs/2611.07486](https://arxiv.org/abs/2611.07486)

    MEvince提出了一种知识约束的LLM推理框架，通过构建连接报告观察结果与技术定义的共享行为表示，解决恶意软件报告中ATT&CK技术识别证据分散、技术界限细微的难题。

    

    将恶意软件分析报告中的行为情报映射到ATT&CK技术有助于威胁分析与防御，但可靠的技术识别仍然充满挑战。支持性证据往往散布在冗长且异构的报告之中，而相互关联的技术共享行为线索，仅在细微却起决定性作用的细节上有所差异。尽管大语言模型（LLM）能够利用外部知识来解读报告，但它们可能基于看似合理的关联而非充分的证据来指派技术。我们提出了MEvince，一个面向报告级恶意软件技术识别的知识约束LLM推理框架。其核心是连接报告观察结果与技术定义的共享行为表示。该表示通过对技术文档的智能体式挖掘构建而成，并支持两个在线阶段：1）原语引导的提取，利用行为原语对LLM提取的观察结果进行结构化处理……

    arXiv:2611.07486v1 Announce Type: cross  Abstract: Mapping behavioral intelligence in malware analysis reports to ATT&CK techniques supports threat analysis and defense, but reliable identification remains challenging. Supporting evidence is often scattered across long, heterogeneous reports, while closely related techniques share behavioral cues and differ only in subtle but decisive details. While large language models (LLMs) can use external knowledge to interpret reports, they may assign techniques based on plausible associations rather than sufficient evidence. We present MEvince, a knowledge-constrained LLM reasoning framework for report-level malware technique identification. Its core is a shared behavioral representation connecting report observations to technique definitions. This representation is built through agentic mining of technique documentation and supports two online stages: 1) primitive-guided extraction uses behavioral primitives to structure LLM-extracted observat
    
[^57]: 定位并不意味着可操控性：LLM数学中符号错误的机制研究

    Localization Does Not Imply Steerability: A Mechanistic Study of Sign Errors in LLM Mathematics

    [https://arxiv.org/abs/2611.07478](https://arxiv.org/abs/2611.07478)

    该研究通过逐层因果分析揭示了LLM在多步数学中的符号错误主要由网络后四分之一区段的MLP层在后期产生，且不同模型具有不同机制（固定的“负号盖章”层、符号放大器、救援层），同时指出即便能精确定位错误来源，也不意味着能够有效操控和纠正它。

    

    当大语言模型进行多步数学运算时，一种特征性的失败反复出现：模型在计算中途写下一个错误的符号token——例如余子式的符号、递归分部积分步骤中的负号——随后正确执行其余的算术运算，但最终答案出错。我们探究这个符号在网络内部是在何处写下的。在涵盖14B至675B参数的六个开源模型中，逐层读出实验表明，错误的符号是在计算后期产生的——由网络最后四分之一区段的MLP层所为（每个模型和任务中占比70-100%），且在经过测试的情况下（Qwen），这一模式在指令微调之前的基础模型中就已存在。具体机制因模型而异：一些模型带有固定的“负号盖章”层，无论正确答案如何都会推出负号；另一些模型则带有对已选定符号的放大器；而Qwen还额外增加了一个下游的“救援层”来对抗这种盖章。该回路是因果性的：从残差流中减去单一符号方向（后文截断）……

    arXiv:2611.07478v1 Announce Type: cross  Abstract: When large language models do multi-step mathematics, a characteristic failure recurs: the model writes one wrong sign token mid-computation -- a cofactor sign, the minus of a recursive integration-by-parts step -- then executes the rest of the arithmetic correctly, and the answer comes out wrong. We ask where inside the network that sign is written. Across six open models spanning 14B to 675B parameters, a per-layer readout shows the wrong sign is committed late -- by MLP layers in the last quarter of the network (70-100% per model and task), a pattern present, where tested (Qwen), in the base model before instruction tuning. The mechanism varies: some models carry fixed "minus-stamp" layers that push a minus regardless of the right answer, others carry amplifiers of whichever sign was already chosen, and Qwen adds a downstream rescue layer that opposes the stamp. The circuit is causal: subtracting a single sign direction from the res
    
[^58]: 面向长程角色扮演对话的递归自我改进

    Recursive Self-Improvement for Long-Horizon Role-Playing Dialogue

    [https://arxiv.org/abs/2611.07469](https://arxiv.org/abs/2611.07469)

    该论文提出一种有界递归自我改进（RSI）框架，通过自我修订角色记忆、人设指令与回合级提示策略，将长程角色扮演对话的劣质回复率从14.00%降至4.94%（相对降低64.71%）。

    

    长程角色扮演对话需要一致的角色身份、可靠的记忆以及语境敏感的交互。我们提出了一种有界递归自我改进框架，用于修订原创角色的记忆、稳定的人设指令以及构建回合级提示词的策略。修复结果还会进一步用于修订更新器的诊断规则、编辑路由和候选选择。我们将更新器定义为一个带版本管理的策略，具有显式输入、可编辑字段、验收门限和停止条件。内部评估涵盖50个原创角色和五个模型，每个角色-模型对进行两段100轮对话：每个条件下共500段对话。在汇总的评估回复中，劣质回复率从14.00%（6,973/49,824）降至4.94%（2,459/49,793），绝对下降9.06个百分点，相对下降64.71%。在600次盲测人工对比中，RSI获得294次……（摘要原文在此处截断）

    arXiv:2611.07469v1 Announce Type: new  Abstract: Long-horizon role-playing dialogue requires consistent character identity, reliable memory, and context-sensitive interaction. We present a bounded recursive self-improvement (RSI) framework that revises original-character memory, stable persona instructions, and a policy for constructing turn-level prompts. Repair outcomes also inform revisions to the updater's diagnostic rules, edit routing, and candidate selection. We specify the updater as a versioned policy with explicit inputs, editable fields, acceptance gates, and stopping conditions. An internal evaluation covers 50 original characters and five models, with two 100-round conversations per character-model pair: 500 conversations per condition. Across the pooled evaluated replies, the bad-response rate decreases from 14.00% (6,973/49,824) to 4.94% (2,459/49,793), a 9.06-percentage-point reduction and a 64.71% relative reduction. In 600 blinded human comparisons, RSI receives 294 w
    
[^59]: FINCOSMOS：一个面向金融领域智能体的开放世界评估范式

    FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance

    [https://arxiv.org/abs/2611.07464](https://arxiv.org/abs/2611.07464)

    FINCOSMOS提出了一个金融智能体的开放世界评估范式：模拟用户隐藏其投资策略，智能体必须自主从网络获取数据并构建防“未来函数”的回测，从而实现无需参考答案、程序可验证的评估。

    

    语言智能体越来越多地在开放网络上为用户工作，而金融领域是对它们最难以信任的场景。数据必须来自网络，策略只存在于用户脑中，且正确性依赖于时间，因此智能体可能“无声地失败”：执行一个用户从未要求过的策略、使用未经核实的数据，或进行一次使用了未来信息的回测——这些都会产出一个与正确报告同样令人信服的报告。现有基准测试总会在某处封闭世界，固定任务、数据或时间线，从而消除了这些失败模式而非度量它们。我们提出FINCOSMOS，一个面向金融工作智能体的开放世界评估范式：模拟用户持有一个从组合文法中采样得到的隐藏策略，且仅在被询问时才透露；智能体不会获得任何数据，因此必须自行从网络获取数据、在数据截止日期约束下构建自己的回测，并交付可验证的结果。评分无需参考答案，程序可自动验证时点正确性。

    arXiv:2611.07464v1 Announce Type: new  Abstract: Language agents increasingly work for users on the open Web, and finance is where trusting them is hardest. The data must come from the Web, the strategy lives in the user's head, and correctness depends on time, so an agent can fail silently: a strategy the user never asked for, unverified data, or a backtest that uses the future yields a report as convincing as a correct one. Existing benchmarks close the world somewhere, fixing the task, the data, or the timeline, and thereby remove these failures instead of measuring them. We introduce FINCOSMOS, an open-world evaluation paradigm for agents doing financial work. A simulated user holds a hidden strategy sampled from a compositional grammar and reveals it only when asked. The agent receives no data, so it must acquire data from the Web, build its own backtest under a data cutoff, and deliver verifiable results. Grading needs no reference answer. Programs verify point-in-time correctnes
    
[^60]: 通往真实世界的最后一英里：作为解决真实世界任务的可复用构建块的智能体原生原语

    The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks

    [https://arxiv.org/abs/2611.07461](https://arxiv.org/abs/2611.07461)

    提出智能体原生原语与 HEELO 框架，通过为资源配备常驻 LLM 并从原语库组合出任务专用框架，弥合从获取资源到实际应用资源之间“通往真实世界的最后一英里”。

    

    智能体如何端到端地处理真实世界任务？配备各自框架的前沿模型在真实世界基准测试中仍然表现不佳：在 Terminal-Bench 4.0 上，GPT-5.6 Sol 配合 Codex 仅解决 37.3% 的任务，Claude Opus 5 配合 Claude Code 解决 51.8%。这些任务在从获取资源到应用资源的步骤上失败，我们将其称为通往真实世界的最后一英里。现有方法只是将资源置于可及范围内，却把应用资源的能力留给调用者，因此每个框架都需要重建这种能力。我们提出了智能体原生原语——一种可复用的单元，它为资源配备一个常驻 LLM，该 LLM 能在上下文中应用该资源并与其他原语通信；同时提出 HEELO（围绕不断演化的智能体原生原语构建的框架工程方法，用于真实世界任务的最后一英里），这是一个从此类原语库为每个任务构建框架的框架工程方法。HEELO 将任务所需的原语组合成任务特定的框架……

    arXiv:2611.07461v1 Announce Type: cross  Abstract: How can an agent handle a real-world task end to end? Frontier models with their own harnesses still fall short on real-world benchmarks: on Terminal-Bench 4.0, GPT-5.6 Sol with Codex resolves 37.3% of the tasks and Claude Opus 5 with Claude Code 51.8%. These tasks fail at the step from reaching a resource to applying it, which we call the last mile to the real world. Existing approaches put the resource within reach but leave the competence to apply it with the caller, so each harness rebuilds that competence. We introduce agent-native primitives, reusable units that equip a resource with a resident LLM that applies it in context and communicates with other primitives, and HEELO (Harness Engineering around Evolving Agent-Native Primitives for the Last Mile to Real-WOrld Tasks), a harness engineering framework that builds each task's harness from a library of such primitives. HEELO composes the primitives a task needs into a task-speci
    
[^61]: 延迟感知的自适应 Best-of-N 采样

    Latency-Aware Adaptive Best-of-$N$

    [https://arxiv.org/abs/2611.07453](https://arxiv.org/abs/2611.07453)

    该论文提出了一种延迟感知的两阶段自适应 Best-of-N 采样算法，通过比较期望的奖励提升与基于观测延迟预测的成本，动态决定是否抽取以及以多大规模抽取第二批样本，从而在服务器负载变化时更合理地分配测试时计算资源。

    

    测试时计算通过在推理阶段投入额外计算来改进语言模型的响应，而合适的计算量取决于具体的提示词。自适应 TTC 方法根据响应质量的信号（如观测到的奖励）来分配计算，通常在信号超过某个固定阈值时停止，并且一般以生成次数或 token 数量来衡量成本。但在在线服务场景中，延迟同样是一种成本：它会随服务器负载增加而增长，因此使用固定阈值的方法在生成速度快时投入的计算过少，而在生成速度慢时又投入过多。针对广泛使用的 TTC 方法 Best-of-N 采样，我们提出了一种两阶段自适应算法：先抽取一个初始批次，然后利用观测到的奖励和该批次的延迟，通过比较期望的改进与预测的成本来决定是否抽取第二个批次以及其大小：当第一批次的最佳奖励低于阈值时（原文摘要在此处截断）……

    arXiv:2611.07453v1 Announce Type: new  Abstract: Test-time compute (TTC) improves language model responses by spending additional computation at inference time, and the right amount of compute depends on the prompt. Adaptive TTC methods allocate compute based on signals of response quality such as observed rewards, often by stopping once the signal exceeds a fixed threshold, and generally measure cost by the number of generations or tokens. But in online serving, latency is also a cost: it grows with server load, so a method using a fixed threshold spends too little compute when generation is fast and too much when it is slow. For Best-of-$N$ sampling, a widely used TTC method, we propose a two-stage adaptive algorithm that draws an initial batch, then uses the observed rewards and the batch's latency to decide whether to draw a second batch and at what size by comparing expected improvement against predicted cost: it draws one when the first batch's best reward is below a threshold th
    
[^62]: EnvOpt：面向智能体环境的自动化优化

    EnvOpt: Automated Optimization for Agent Environments

    [https://arxiv.org/abs/2611.07372](https://arxiv.org/abs/2611.07372)

    EnvOpt提出了对LLM智能体静态环境进行自动化端到端优化的第三条路径，无需评分器或参考答案即可大幅降低智能体成本（最高72.7%）并提升任务通过率。

    

    一个LLM智能体的成本与成功取决于其模型、执行框架及其环境。除了被广泛研究的参数调优和执行框架优化方向之外，我们提出了第三个方向：对静态环境的优化。智能体通常部署在由众多任务共享的丰富环境中，例如公司的Slack工作区，或由数据源、文档、工单组成的复杂集合。我们将这类环境称为静态环境。我们提出了EnvOpt，一种针对任意给定模型和执行框架的静态环境自动化端到端优化方法，它可以使用真实或合成任务，且无需任何评分器或参考答案。在AutomationBench上，EnvOpt将Claude Haiku 4.5的成本降低了72.7%，同时将其通过率提高了14.34%。相同的环境可迁移至Opus 5，将其成本降低68.1%，并带来1.34%的提升。在Meta-Harness针对Haiku 4.5优化的执行框架之上叠加使用时，它可进一步降低（原文在此处截断）……

    arXiv:2611.07372v1 Announce Type: new  Abstract: An LLM agent's cost and success depend on its model, its harness, and its environment. In addition to the widely studied directions of parameter tuning and harness optimization, we propose the third direction: optimization over stationary environments. Agents are often deployed in rich environments shared by many tasks, such as a company's Slack workspace or a complex collection of data sources, documents or tickets. We call these environments stationary environments. We introduce EnvOpt, a method for automated end-to-end optimization of stationary environments for any given model and harness, using real or synthetic tasks without needing any graders or reference answers. On AutomationBench, EnvOpt reduces the cost of Claude Haiku 4.5 by 72.7% and raises its pass rate by 14.34%. The same environment transfers to Opus 5, cutting its cost by 68.1% with a 1.34% gain. Added on top of Meta-Harness's optimized harness for Haiku 4.5, it lowers 
    
[^63]: 大语言模型中语法数的编码与传播

    Encoding and Propagation of Grammatical Number in Large Language Models

    [https://arxiv.org/abs/2611.07283](https://arxiv.org/abs/2611.07283)

    该研究利用稀疏自编码器（SAE）深入揭示了 Pythia-70M 小型语言模型中语法数一致现象的表示与处理机制，展示了 SAE 如何用于查找和传播语言模型内部的语法特征。

    

    稀疏自编码器（SAEs）已成为一种日益流行的方法，可将大语言模型（LLM）的隐藏状态向量转换为人类可理解的特征或符号列表。因此，SAEs 可以为我们洞察 LLM 的知识表示和内部“心智”语法提供途径——这对 AI 行为的机制性解释具有实践意义，对研究人类语言的计算机模拟表示也具有理论价值。参数少于 1 亿、仅在小语料库上训练的语言模型，尽管生成内容的可靠性有限，却仍表现出强大的语法能力和语篇连贯性。我们使用在这样一个模型（Pythia-70M）上训练的 SAE，深入研究单一语法现象——数的一致——在 LLM 中是如何被表示和处理的。我们展示了 SAE 可用于查找和传播语法特征的机制与方法。

    arXiv:2611.07283v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) have become an increasingly popular means of translating Large Language Model (LLM) hidden state vectors into a human-comprehensible list of features or symbols. As such, SAEs may provide insight into an LLM's knowledge representation and internal ``mental'' grammar - topics of pragmatic interest for mechanistic interpretation of AI behavior and of theoretical interest for the study of in silico representations of human language. Language models with fewer than 100M parameters, trained on small corpora, nonetheless exhibit strong grammatical competence and discourse coherence, despite producing content of limited reliability. We use SAEs that we trained on one such model (Pythia-70M) to conduct an in-depth examination of how a single grammatical phenomenon, number agreement, is represented and processed in an LLM. We demonstrate the mechanism and the methodology by which SAEs can be used to find and propagate
    
[^64]: HyFlash：一种用于端侧投机解码的前缀记忆先验

    HyFlash: A Prefix-Memory Prior for On-Device Speculative Decoding

    [https://arxiv.org/abs/2611.07263](https://arxiv.org/abs/2611.07263)

    HyFlash提出一种引入前缀记忆先验的紧凑并行草拟器，将长程上下文建模与细粒度草拟位置交互分离，以显著更少的内存实现了比全上下文注意力更高的投机解码接受率。

    

    并行块草拟通过在单次骨干网络前向传播中预测多个未来位置来加速投机解码，但端侧部署同时限制了草拟器的能力和上下文存储。在紧张的内存预算下，草拟器必须在不会产生全上下文注意力那种不断增长的KV缓存开销的情况下，建模长程依赖和细粒度的草拟位置交互。我们提出HyFlash，一种紧凑的并行草拟器，它引入前缀记忆先验，在能力有限的情况下将长程上下文建模与细粒度的草拟位置交互分离开来。局部注意力负责捕获近期上下文和草拟位置交互，而循环记忆则将历史目标特征压缩为固定大小的状态。这种归纳偏置使得在草拟侧内存有限的情况下，仍能进行有效的局部和长程建模。实验表明，HyFlash以显著更少的KV存储实现了比全上下文注意力更高的接受率。

    arXiv:2611.07263v1 Announce Type: new  Abstract: Parallel block drafting accelerates speculative decoding by predicting multiple future positions in a single backbone pass, but on-device deployment constrains both drafter capacity and context storage. Under tight memory budgets, drafters must model long-range dependencies and fine-grained draft-position interactions without the growing KV-cache overhead of full-context attention. We propose HyFlash, a compact parallel drafter that introduces a prefix-memory prior to separate long-range context modeling from fine-grained draft-position interactions with limited capacity. Local attention captures recent context and draft-position interactions, while recurrent memory compresses historical target features into fixed-size states. This inductive bias enables effective local and long-range modeling with bounded draft-side memory. Experiments show that HyFlash achieves higher acceptance than full-context attention with substantially less KV st
    
[^65]: D²-OPD：在外推式在线策略蒸馏中分离方向与距离

    $D^2$-OPD: Separating Direction and Distance in Extrapolative On-Policy Distillation

    [https://arxiv.org/abs/2611.07241](https://arxiv.org/abs/2611.07241)

    D²-OPD通过解耦外推方向与距离来改进外推式在线策略蒸馏——用每前缀硬KL约束限制目标与教师的距离，并用教师训练后偏移与初始师生差距的自适应组合确定外推方向，从而在多种场景下提供更稳健的学习信号。

    

    在线策略蒸馏（OPD）训练学生模型在学生生成的前缀上匹配教师模型的下一个词元分布。外推式OPD进一步扩展教师相对于参考策略的对数概率偏移，以构建更强的学习目标。然而，现有方法将外推方向与距离耦合在一起，使得教师偏移的幅度直接决定目标移动的远近。我们提出D²-OPD，将这两个因素解耦：通过针对每个前缀的硬KL约束来限制目标与教师之间的距离，同时由教师训练后偏移与初始师生差距的自适应组合来决定外推方向。我们使用前向KL散度将学生模型优化至该目标，即使对低概率词元也能提供学习信号。我们在三种设置下进行评估：标准OPD表现良好的设置、标准OPD在基础模型初始化下表现不佳的设置，以及没有教……（原文摘要在此处截断）

    arXiv:2611.07241v1 Announce Type: cross  Abstract: On-policy distillation (OPD) trains a student model to match a teacher's next-token distributions on student-generated prefixes. Extrapolative OPD further extends the teacher's log-probability shift relative to a reference policy to construct stronger learning targets. However, existing methods couple extrapolation direction and distance, allowing the magnitude of the teacher's shift to dictate how far the target moves. We propose $D^2$-OPD, decoupling these two factors: a per-prefix hard KL constraint bounds the target's distance from the teacher, while an adaptive combination of the teacher's post-training shift and the initial student--teacher gap determines the extrapolation direction. We optimize the student toward this target using forward KL, providing learning signals even for low-probability tokens. We evaluate three settings: where standard OPD performs well, where it struggles with base-model initialization, and where no tea
    
[^66]: AI智能体是如何浪费你的金钱的？

    How Do AI Agents Waste Your Money?

    [https://arxiv.org/abs/2611.07238](https://arxiv.org/abs/2611.07238)

    本文提出了一个统一的token浪费度量框架，利用LLM-as-a-Judge识别智能体轨迹中可移除的内容，并通过在线剪枝实验证明可以在保持任务性能的同时减少41-46%的模型推理token。

    

    AI智能体在各种任务中可能消耗大量token，但这些token中究竟有多少是真正必要的？尽管先前的研究已经表明智能体会产生出乎意料的高昂token成本，我们目前仍缺乏一种系统性的方法来识别和度量token浪费。在这项工作中，我们提出了一个统一的框架，用于度量异构智能体系统中的token浪费。该框架记录并规范化完整的智能体轨迹，使用LLM-as-a-Judge（LLMaJ）识别可移除的文本片段，并将识别出的文本片段映射回原始轨迹以进行token级别的度量。我们通过直接干预来验证所识别的浪费：移除被标记的片段、恢复智能体的执行，并测量任务结果和后续动作的变化。在在线剪枝实验中，我们在两个不同的智能体基准测试上移除了41-46%的模型推理，同时基本保持了任务性能。

    arXiv:2611.07238v1 Announce Type: new  Abstract: AI agents may consume massive tokens across a wide range of tasks, but how many of those tokens are really necessary? Although prior work has shown that agents can incur unexpectedly high token costs, we still lack a systematic way to identify and measure token waste. In this work, we introduce a unified framework for measuring token waste across heterogeneous agent systems. Our framework records and normalizes complete agent trajectories, identifies removable spans using an LLM-as-a-Judge (LLMaJ), and maps the identified text spans back to the original trajectory for token-level measurement. We validate the identified waste through direct interventions by removing marked spans, resuming agent execution, and measuring changes in task outcomes and subsequent actions. In online pruning experiments, we remove 41-46% of the model's reasoning while largely preserving task performance across two distinct agent benchmarks. Using this framework,
    
[^67]: 语言模型神经元中的原型语境：一种智能架构？

    Archetypal contexts in the neurons of language models: an architecture for intelligence?

    [https://arxiv.org/abs/2611.07234](https://arxiv.org/abs/2611.07234)

    该研究在Qwen3.8-27B模型中发现，跨领域的原型语境（如“临界点”）、主题和语言分别由三组可独立操控的神经元编码，且网络层呈现“外层语言—中间主题—核心原型”的三明治式组织架构，暗示这可能正是智能的底层架构。

    

    我们验证了此前的假设，即语言模型中存在着独立于具体实例化领域的关联模式（Nordfors, 2026）。这些“原型语境”是人们在众多领域中都能识别出的关系模式，例如“临界点”或“赢家通吃”，它们只被存储一次，便可跨领域应用。我们在模型 Qwen3.8-27B 内部展示了这一现象：一组神经元承载原型（如临界点），另一组承载主题（如政治、生态学或细胞生物学），第三组承载语言（如英语、西班牙语或中文）。当使承载“临界点”原型的神经元沉默后，模型不再识别该模式，但仍然知道文本讲的是什么；当使承载主题的神经元沉默后，模型失去主题信息，但仍能说出该模式。在神经元映射中，各层自组织成一种“三明治”结构：外层承载语言，其内的中间层承载主题，而最核心的层承载原型。三明治……（原文摘要在此处截断）

    arXiv:2611.07234v1 Announce Type: cross  Abstract: We validate our previous hypothesis that a language model holds relational patterns apart from the subject domains that instantiate them (Nordfors, 2026). These archetypal contexts are patterns of relations that people recognize across many fields, such as the tipping point or winner takes all, held once and applied across domains. We show this inside the model, Qwen3.8-27B. One set of neurons carries the archetype, say the tipping point; another carries the topic, say politics, ecology or cell biology; a third carries the language, such as English, Spanish or Chinese. Silence the tipping point's neurons, and the model no longer identifies the pattern but still knows what the text is about. Silence the topic's, and it loses the topic but still names the pattern. In the neuron maps the layers organize themselves into a sandwich: the outer layers carry languages, the layers inside them topics, the layers at the heart archetypes. The sand
    
[^68]: 递归稀疏注意力：以对数成本实现百万token长度外推

    Recursive Sparse Attention: Million-Token Length Extrapolation at Logarithmic Cost

    [https://arxiv.org/abs/2611.07232](https://arxiv.org/abs/2611.07232)

    RecSA在KV缓存上构建递归树结构，利用从稠密注意力蒸馏且跨层级共享的池化键，使每个查询的选择成本降至O(log L)，从而以对数成本实现了百万token级的长度外推。

    

    在约100万token的上下文长度下，稠密注意力在预填充和解码阶段成本高昂，且训练通常依赖长文档，而随着目标长度的增加，长文档变得越来越稀缺。大多数稀疏注意力方法会对所有前序块或token进行评分，因此在上下文长度L时每个查询的选择成本为O(L)；分层选择可将该成本降至O(log L)，但此前可训练的分层方法仅在训练长度内报告了准确性。我们提出RecSA（递归稀疏注意力），它在KV缓存上构建一棵树，其中每个节点用一个键和一个均值池化的值来概括其子树。这些键来自每层一个池化器，由所有层级共享，并从稠密注意力中蒸馏而来。token之上的每一层最多选择k个候选，因此每个查询具有O(log L)的选择成本并读取O(log L)个条目；未被选中的候选的摘要与精确token共享同一个softmax。

    arXiv:2611.07232v1 Announce Type: new  Abstract: At context lengths of about 1M tokens, dense attention is costly in prefill and decoding, and training typically relies on long documents, which become scarcer as the target length increases. Most sparse attention methods score all preceding blocks or tokens, so their selection cost per query is $O(L)$ at context length $L$; hierarchical selection reduces it to $O(\log L)$, but prior trainable hierarchical methods report accuracy only within their training length. We propose RecSA (Recursive Sparse Attention), which builds a tree over the KV cache in which each node summarizes its subtree with one key and one mean-pooled value. The keys come from one pooler per layer, shared by all levels and distilled from dense attention. Each level above the tokens selects at most $k$ candidates, so each query has $O(\log L)$ selection cost and reads $O(\log L)$ entries; the summaries of unselected candidates share one softmax with the exact tokens. E
    
[^69]: 组合式去对齐：审计大型视觉-语言模型的安全性

    Compositional Unalignment: Auditing Safety in Large Vision-Language Models

    [https://arxiv.org/abs/2611.07216](https://arxiv.org/abs/2611.07216)

    本文提出以“去对齐”作为安全对齐的审计手段，揭示大型视觉-语言模型的对齐并不稳健——极少量的监督微调即可达到与多阶段强化学习流程相当的基准分数，且模型对组合性伤害的识别与拒绝能力是相互解耦的。

    

    大型视觉-语言模型（LVLMs）容易受到组合性伤害：一张图像和一个提示词各自单独来看都是无害的，但经过模型的跨模态推理后却可能变得有害。近期的对齐方法在组合安全基准上报告了很高的分数，但LLM裁判只能测量观察到的行为，无法揭示安全性在模型内部被整合得有多稳健。我们认为，“去对齐”为对齐稳健性提供了一种互补的审计手段。我们首先表明，极少量的监督微调即可达到与多阶段强化学习流水线相当的LLM裁判分数，因此仅凭相似的基准分数本身无法确立稳健性。随后我们表明，对组合性伤害的识别与拒绝是相互解耦的：九种标准的方向探测方法能够以最高AUC=0.99检测出组合性伤害，然而消融所恢复出的方向，要么使拒绝行为保持完好，要么破坏模型的连贯性。因此我们……

    arXiv:2611.07216v1 Announce Type: cross  Abstract: Large vision-language models (LVLMs) are vulnerable to compositional harm: an image and a prompt that are benign in isolation become harmful through the model's cross-modal reasoning. Recent alignment methods report strong scores on compositional safety benchmarks, but an LLM judge only measures observed behavior and cannot reveal how robustly safety is integrated into the model. We argue that un-alignment provides a complementary audit of alignment robustness. We first show that minimal supervised fine-tuning reaches an LLM-judge score comparable to a multi-stage reinforcement-learning pipeline, so similar benchmark scores cannot establish robustness by themselves. We then show that recognition and refusal of compositional harm are decoupled: nine standard direction-finding methods detect compositional harm with up to AUC=0.99, yet ablating the recovered directions either leaves refusal intact or destroys model coherence. We therefore
    
[^70]: 母系经典：一种仅替换经典小说中性别词汇的计算方法

    Matriarchy Classics: A Computational Approach to Swap Only Gendered Words in Classic Novels

    [https://arxiv.org/abs/2611.07134](https://arxiv.org/abs/2611.07134)

    本文提出一种可靠的计算方法，能在程序化保证其他文字不变的前提下仅替换小说中的性别词汇，将原本需数周的手工性别转换缩短至几分钟，并通过《傲慢与偏见》和《了不起的盖茨比》的母系改写版本加以演示。

    

    本文提出了一种可信的计算方法，用于对公有领域小说进行性别转换，并通过程序化保证仅反转性别词汇，而所有其他词汇保持不变。这看似是一个简单的任务，但大型语言模型和朴素的基于规则的方法都无法完成。所提出的方法包括自定义的保留空白字符的分词器、覆盖复数和所有格形式的性别转换映射、代词处理以及角色重命名策略。该方法将一部小说的性别转换所需时间从数周的手工劳动缩短至几分钟。该方法在两部公有领域小说上进行了演示：《傲慢与偏见》被转换至母系的摄政时期英格兰，在那里遗产由女儿继承，母亲掌管家务，心急如焚的班纳特先生决心为儿子们谋得美满姻缘；《了不起的盖茨比》则转向母系的爵士时代，在那里新富的盖茨比夫人……

    arXiv:2611.07134v1 Announce Type: new  Abstract: This paper proposes a trustworthy computational method to gender-swap public domain novels, with programmatic guarantees that only gendered words are inverted and all other words are left unchanged. This may seem like a simple task, but both large language models and naive rule-based methods fail to complete it. The proposed method includes a custom whitespace-preserving tokenizer, gender-swap mappings across plural and possessive forms, pronoun handling, and a character renaming strategy. It shortens the time required to gender-swap a novel from weeks of manual labor to minutes. The method is demonstrated on two public-domain novels. Pride and Prejudice is transported to matriarchal Regency England, where inheritance flows through daughters, mothers rule the household, and a frantic Mr. Bennet is determined to secure advantageous marriages for his sons. The Great Gatsby pivots to a matriarchal Jazz Age where newly-rich Ms. Gatsbee throw
    
[^71]: 通过软提示调优将大语言模型评审者适配至已观测的人类标注策略

    Adapting LLM Judges to Observed Human Annotation Policy through Soft Prompt Tuning

    [https://arxiv.org/abs/2611.07126](https://arxiv.org/abs/2611.07126)

    本文提出混合软提示调优方法，将文本提示与学习到的连续提示嵌入相结合，使开源LLM评审者在偏见、毒性和个性化维度上与人类标注策略的一致性显著提升，甚至超越托管的专有参考模型。

    

    大语言模型（LLM）越来越多地被用作评估聊天机器人回复的自动化评审者。虽然专有模型能够与人类标注者达成高度一致，但其成本高昂且难以在实际场景中部署。轻量级的开源LLM评审者提供了一种有吸引力的替代方案，但通常与人类评分的一致性较低。本文研究了软提示调优能否使开源LLM评审者更贴近人类标注策略。我们在FairBotBench基准上训练软提示，并在偏见、毒性和个性化等维度上对适配后的评审者进行评估。研究结果表明，混合软提示调优——即把文本提示与学习到的连续提示嵌入相结合——相比相应的未调优基线，能够持续提升与人类标注的一致性。对于Qwen 3.5 9B，适配后的评审者超越了托管的参考模型。

    arXiv:2611.07126v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly employed as automated judges for evaluating chatbot responses. While proprietary models can achieve high agreement with human annotators, they remain costly and difficult to deploy in practical settings. Lightweight open-source LLM judges provide an attractive alternative but typically exhibit lower agreement with human ratings. In this paper, we investigate whether soft prompt tuning can align open-source LLM judges more closely with human annotation policy. We train soft prompts on the FairBotBench benchmark and evaluate the adapted judges across the dimensions of bias, toxicity, and personalization. Our findings demonstrate that hybrid soft-prompt tuning, which combines textual prompts with learned continuous prompt embeddings, consistently improves agreement with human annotations over the corresponding untuned baselines. For Qwen 3.5 9B, the adapted judges surpass both hosted reference
    
[^72]: 患者无法获得的服务：从在线社区挖掘癌症筛查障碍

    What Patients Say They Can't Access: Mining Cancer Screening Barriers from Online Communities

    [https://arxiv.org/abs/2611.07113](https://arxiv.org/abs/2611.07113)

    该研究通过分析六个Reddit癌症社区中超过56,000条帖子，系统提取并归类了美国患者报告的癌症筛查障碍，揭示了随癌症类型和社区背景变化的结构性、信息性和心理社会性障碍。

    

    癌症筛查差异在美国持续存在，但驱动低筛查参与率的患者报告障碍至今仍未得到大规模的充分刻画。虽然临床调查能够捕获结构化的回答，但它们遗漏了患者坦率描述现实障碍的自发性、非征集性话语。在本文中，我们对六个癌症特定主题的Reddit社区中超过56,000条帖子进行了大规模分析，以揭示美国患者的癌症筛查障碍。我们应用了一条结合关键词过滤、障碍分诊、开放式编码和编码本综合的流程，系统地从576条位于美国的帖子中提取并归类患者报告的障碍。我们的定性分析揭示了反复出现的结构性、信息性和心理社会性障碍，这些障碍因癌症类型和社区背景的不同而呈现出有意义的差异。这些发现为未来旨在大规模主动识别和解决筛查障碍的工作奠定了基础。

    arXiv:2611.07113v1 Announce Type: new  Abstract: Cancer screening disparities persist across the U.S., yet the patient-reported barriers driving low uptake remain poorly characterized at scale. While clinical surveys capture structured responses, they miss the organic, unsolicited discourse where patients candidly describe real-world obstacles. In this paper, we conduct a large-scale analysis of over 56,000 posts across six cancer-specific Reddit communities to surface barriers to cancer screening among U.S. patients. We apply a pipeline combining keyword filtering, barrier triage, open coding, and codebook synthesis to systematically extract and categorize patient-reported barriers from 576 U.S.-located posts. Our qualitative analysis reveals recurring structural, informational, and psychosocial barriers that vary meaningfully by cancer type and community context. These findings lay the groundwork for future work aimed at proactively identifying and addressing barriers at scale - span
    
[^73]: 向同伴导师提供自动化反馈可提升其话语策略与学习者推理，但并未改变谈话的平衡

    Automated Feedback to Peer Tutors Increases Their Talk Moves and Learner Reasoning but Not the Balance of Talk

    [https://arxiv.org/abs/2611.07072](https://arxiv.org/abs/2611.07072)

    向同伴志愿导师提供自动化话语反馈，可使其引出学习者想法、采纳学习者贡献及促进学习者推理的能力提升 8-15% 并跨训练营得到重复验证，但并未改变导师与学习者之间的谈话平衡。

    

    对课堂话语的自动化反馈能够改善专业教学和在线教学情境中的教学质量，近期研究表明，向学习者提供类似的反馈可以通过分担参与责任来强化这些效果。但这种做法是否适用于小组志愿辅导尚未得到检验。我们报告了一项随机对照试验的结果，该试验在 Schoolhouse.world 平台上两个为期四周的 SAT 数学训练营中开展，涉及 1,030 名志愿导师、7,443 名学习者以及 6,508 节被录制的课程。导师被随机分配到以下条件之一：不接收反馈、接收关于自身话语的自动化反馈，或在接收该反馈的同时，向学习者提供参与反馈（采用两种动机框架之一）。给导师的反馈使其引出学习者想法、采纳学习者贡献以及学习者推理水平提高了 8-15%，并且这一模式在两个训练营中都得到了重复验证。然而，导师谈话……（原文摘要在此截断）

    arXiv:2611.07072v1 Announce Type: new  Abstract: Automated feedback on classroom discourse improves instruction in professional and online teaching contexts, and recent work suggests that delivering similar feedback to learners may strengthen these effects by sharing responsibility for participation. Whether this holds in small-group volunteer tutoring has not been tested. We present results from a randomized controlled trial across two four-week SAT mathematics bootcamps on the Schoolhouse.world platform, involving 1,030 volunteer tutors, 7,443 learners and 6,508 recorded sessions. Tutors were randomly assigned to receive no feedback, automated feedback on their own discourse, or that feedback together with participation feedback delivered to their learners under one of two motivational framings. Feedback to tutors increased eliciting of learner ideas, uptake of learner contributions, and learner reasoning by 8-15%, and the pattern replicated across both bootcamps. However, tutor talk
    
[^74]: 你错过的只是重新分类：在大规模人群中识别在线健康社区中的基因变异重新解读需求

    Reclassification Is All You Missed: Surfacing Variant Reinterpretation Need in Online Health Communities at Population Scale

    [https://arxiv.org/abs/2611.07054](https://arxiv.org/abs/2611.07054)

    研究构建了基于智能体大语言模型的PRISM流水线，能在大规模在线健康社区中筛查基因变异重新解读相关的讨论，发现123个候选分类差异（其中5个从意义未明升级为致病性），从而触及临床重新联系无法覆盖的患者群体。

    

    基因变异在患者完成检测后经常会在ClinVar数据库中被重新分类，有时会从“意义未明”变为“致病性”，然而对患者进行重新联系的做法并不一致，责任归属不明确且分散于实验室、医疗服务提供者和患者之间。许多受影响的患者在公共在线健康社区中讨论他们的检测结果，而临床重新联系往往无法触及这些渠道。我们构建了PRISM，这是一个基于智能体的大型语言模型流水线，它能筛选这些帖子、检查相关变异是否随后被重新分类，并起草带有免责声明的联系信息（以无接触试运行方式进行）。在来自32个社区的390万条帖子中，智能体预测的与重新解读相关的讨论较为罕见（0.288%），但高度集中于专业遗传学社区（6.30%，而一般医疗建议社区仅为0.10%），且自2014年以来增长了五倍；PRISM发现了123个候选分类差异，其中5个是从“意义未明”提交升级为“致病性”。

    arXiv:2611.07054v1 Announce Type: cross  Abstract: Genetic variants are routinely reclassified in ClinVar after a patient is tested, sometimes from uncertain to pathogenic, yet patient recontact is inconsistent, with no clear duty and responsibility diffused among labs, providers, and patients. Many affected people discuss their results in public online health communities that clinical recontact often does not reach. We built PRISM, an agentic large-language-model pipeline that screens these posts, checks whether the variant was later reclassified, and drafts a disclaimer-bearing outreach message, in a no-contact dry run. Across 3.9 million posts from 32 communities, agent-predicted reinterpretation-relevant discussion is rare (0.288%) but concentrated in specialist genetics communities (6.30% versus 0.10% in general medical-advice communities) and growing fivefold since 2014; PRISM surfaces 123 candidate classification differences, 5 of them from an uncertain submission to pathogenic 
    
[^75]: 闭环智能体评估：分层度量与自校准

    Closed-Loop Agent Evaluation: Layered Measurement and Self-Calibration

    [https://arxiv.org/abs/2611.07051](https://arxiv.org/abs/2611.07051)

    该论文提出面向已部署智能体的“闭环评估”方法，将评估建模为有偏评级者与受限结果下的性能估计，通过片段/轨迹/对话/参与四层级分析、版本化规则、溯源与裁定组件以及智能体工厂生成的评估者智能体，实现可追溯、可自我校准并能反馈至智能体本身的持续评估体系。

    

    当智能体上线后，团队想知道它是否在正常履行职责，而不仅仅是看它是否报错。然而大多数评估是一次性的：跑一次基准测试、看一眼轨迹、给一个单一分数，其结论既无法自我校验，也无法反馈到智能体本身。我们提出闭环评估，一种针对已部署智能体的评估方法。该方法将评估建模为在有偏评级者与受限结果条件下的性能估计问题，并要求每一层级的结论都能追溯到原子数据，且由带版本管理的规则计算得出。该方法涵盖四个分析层级（片段、轨迹、对话与参与），沿执行、实质、过程与结果这一主线进行度量，始终将行为规范作为约束条件，并使用溯源与裁定组件来认证证据与评估者本身。由智能体工厂生成的评估者智能体携带这些组件，每个评估者都配有领域知识包与适配器框架……

    arXiv:2611.07051v1 Announce Type: new  Abstract: Once an agent is live, teams want to know whether it is doing its job, not just whether it throws errors. Yet most evaluation is one-off: a benchmark run, a look at traces, a single score. Its conclusions neither check themselves nor reach the agent. We propose closed-loop evaluation, an evaluation method for deployed agents. It models evaluation as performance estimation under biased raters and censored outcomes, and it requires every level's conclusion to trace back to atomic data and to be computed by versioned rules. The method covers four levels of analysis (span, trace, conversation and engagement), measures along a spine of Execution, Substance, Trajectory and Results, treats Conduct as a constraint throughout, and uses Provenance and Adjudication components to certify the evidence and the evaluators. Evaluator agents generated by an Agent Factory carry these components, each with a domain knowledge pack and an adapter harness. Tw
    
[^76]: 遵循政策却未通过基准测试：审计与修订基于政策的决策基准

    Follow the Policy, Fail the Benchmark: Auditing and Revising Policy-Based Decision Benchmarks

    [https://arxiv.org/abs/2611.07042](https://arxiv.org/abs/2611.07042)

    论文审计了九个基于政策的决策基准，发现其中存在参考错误和评分缺陷，仅通过修正这些基准问题即可使模型接受率提升超过80个百分点，表明基准质量对模型评估结果有决定性影响。

    

    基于政策的决策基准用于评估系统是否将书面规则应用于财务和运营案例。然而，当基准测试错误地陈述了所需决策时，遵循规则的回答可能被标记为错误。我们通过来源追溯、程序执行和受控检查，对九个公开的基于政策的决策基准进行了审计。我们发现了无依据的目标值、在重放中依然存在或作为已解决步骤传播的参考错误，以及拒绝可接受答案或给不完整决策加分的评估器。为衡量评分错误的影响，我们在不更改模型回答的前提下重新评分。在96个LegalBench SARA数值案例中，仅纠正评分器就使三个模型的接受率各提高了超过80个百分点，而无需改变参考答案或模型回答。在RuleArena税务任务中，恢复所需的零下限改变了300个参考答案中的27个，并缩小了观察到的模型差距。在一个明确了所有权和资格条件的受控联邦豁免任务中……

    arXiv:2611.07042v1 Announce Type: new  Abstract: Policy-based decision benchmarks evaluate whether systems apply written rules to financial and operational cases. Yet a rule-following response can be marked wrong when the benchmark misstates the required decision. We audit nine public policy-based decision benchmarks through source tracing, program execution, and controlled checks. We find unsupported targets, reference errors that survive replay or propagate as solved steps, and evaluators that reject acceptable answers or credit incomplete decisions. To measure the scoring impact, we rescore unchanged model responses. On the 96 LegalBench SARA numeric cases, correcting the scorer raises acceptance by more than 80 percentage points for each of three models, without changing references or responses. In RuleArena Tax, restoring a required zero floor changes 27 of 300 references and narrows an observed model gap. A controlled federal-exemption task with explicit ownership and eligibility
    
[^77]: LowRAM：面向测试时可组合KV缓存压缩的低秩键与摊销值方法

    LowRAM: Low-Rank Keys and Amortised Values for Composable KV Cache Compression at Test-Time

    [https://arxiv.org/abs/2611.07025](https://arxiv.org/abs/2611.07025)

    该论文提出LowRAM方法，利用注意力头内“值可由键预测”的特性，在测试时通过值缓存摊销（仅存储重构最差令牌的残差，实现4倍近无损压缩），并结合跨层低秩键压缩，构建出接近无损的可组合KV缓存压缩流水线。

    

    自回归解码依赖于KV缓存，其内存占用随上下文长度线性增长，限制了批处理大小和吞吐量。现有的训练后方法利用令牌、层和特征维度之间的冗余性，或采用降低数值精度的手段。我们识别出一个互补的维度：在每个注意力头内部，值在很大程度上可以从其对应的键中预测出来。我们通过值缓存摊销来利用这一特性，在测试时为每个注意力头拟合一个轻量级预测器，并仅为其重构效果最差的令牌存储残差，从而实现4倍近无损的值缓存压缩。该预测器可以通过元学习进行初始化以大幅降低预填充开销，也可以在允许一次性成本的场景（如多查询推理）中从头拟合。我们将值摊销与跨层低秩键压缩相结合，构建出LowRAM——一种保持接近无损的非对称压缩流水线。

    arXiv:2611.07025v1 Announce Type: cross  Abstract: Autoregressive decoding relies on the KV cache, whose memory footprint grows linearly with context length and caps batch size and throughput. Existing post-training methods exploit redundancy across tokens, layers, and feature dimensions, or lower numerical precision. We identify a complementary axis: within each attention head, values are largely predictable from their corresponding keys. We exploit this with value cache amortisation, which fits a lightweight per-head predictor at test time and stores residuals only for the tokens it reconstructs worst, giving $4\times$ near-lossless value compression. The predictor can be initialised by meta-learning for substantially reduced prefill overhead, or fitted from scratch in settings that allow a one-off cost, such as multi-query inference. We pair value amortisation with cross-layer low-rank key compression to produce LowRAM, an asymmetric compression pipeline that stays close to lossless
    
[^78]: BlockLock：通过固定批次几何形状实现可复现的LLM推理

    BlockLock: Reproducible LLM Inference with Fixed Batch Geometry

    [https://arxiv.org/abs/2611.06977](https://arxiv.org/abs/2611.06977)

    BlockLock通过仅固定受保护请求所见的批次几何形状（而非约束所有算子或放弃共享计算），在保留共享批次预填充的前提下实现了逐位可复现的LLM推理。

    

    可复现的LLM推理要求无论批次中的其他请求如何，某个请求都能产生相同的输出。现有解决方案以不同方式为此付出代价：批次不变内核需要约束每一个算子，而通过固定形状解码隔离预填充（我们的CoRun风格基线）则放弃了共享的预填充计算并对解码进行填充。我们提出了BlockLock，它保留共享的批次预填充，仅固定受保护请求的数值路径：它固定请求所看到的几何形状，并提供两种模式：BlockLock-Rows使用固定宽度的行并保留所有请求的批次解码；BlockLock-Pack使用终端锚定打包并配以隔离的虚拟段，然后提取受保护请求的KV状态用于仅针对目标的解码。在规定的逐部署契约下，当伴随请求发生变化时，两种模式都能按位保留目标的输出。在五个稠密模型（Qwen3-8B/14B、Llama-3.1-8B-Instruct以及Gemma-3-4B等）上的实验验证了该方法的有效性。

    arXiv:2611.06977v1 Announce Type: new  Abstract: Reproducible LLM inference requires that a request produces the same output regardless of the other requests in the batch. Existing solutions pay for this in different ways: batch-invariant kernels constrain every operator, and isolating prefill with fixed-shape decode ({our CoRun-style baseline}) gives up shared prefill compute and pads decode. We present BlockLock, which keeps shared batched prefill and pins only the numerical path of the protected request: it fixes the geometry that the request sees, with {two modes: BlockLock-Rows uses fixed-width rows and retains batched decode of all requests; BlockLock-Pack uses terminal-anchored packing with an isolated dummy segment, then extracts the protected request's KV state for target-only decode. Under a stated per-deployment contract, both modes preserve the target's output bitwise when its companions change.} Across five {dense} models (Qwen3-8B/14B, Llama-3.1-8B-Instruct, and Gemma-3-4
    
[^79]: 论预填充到自回归跨模型迁移的机遇与挑战

    On the Opportunities and Challenges of Prefill-to-Autoregressive Cross-Model Transfer

    [https://arxiv.org/abs/2611.06976](https://arxiv.org/abs/2611.06976)

    本文提出开源PACT框架PLatour，通过零初始化线性桥接连接冻结的大小模型实现潜在通信，在生成任务上显著超越基线，同时揭示了独立评估跨模型信息迁移有效性的挑战。

    

    预填充到自回归跨模型迁移，是语言模型之间潜在通信的一种形式，它将较大模型的提示表示与较小模型的高效生成相结合，为平衡能力与推理成本提供了一种有前景的途径。然而，现有系统对独立评估大小模型协作的支持有限，且仅凭任务性能的提升并不能证明接收方是否真正受益于发送方的信息。我们提出了PLatour，一个开源的PACT框架，它通过轻量级、零初始化的线性桥接来连接冻结的模型，这些桥接直接使用任务奖励进行训练，并支持混合注意力架构。PLatour在生成任务上显著优于开源基线，并在共享的数学与编码基准上使用规模小得多的发送方超越了Mostik报告的结果。然而，受控实验表明……

    arXiv:2611.06976v1 Announce Type: new  Abstract: Prefill-to-Autoregressive Cross-model Transfer (PACT), a form of latent communication between language models, combines a larger model's prompt representations with a smaller model's efficient generation, offering a promising way to balance capability and inference cost. However, existing systems provide limited support for independently evaluating large-small model collaboration, and improved task performance alone does not establish whether the receiver benefits from the sender's information. We introduce PLatour, an open-source PACT framework that connects frozen models through lightweight, zero-initialized linear bridges trained directly with task rewards and supports hybrid-attention architectures. PLatour significantly outperforms open-source baselines on generation tasks and surpasses Mostik's reported results on shared mathematics and coding benchmarks using a substantially smaller sender. However, controlled experiments show tha
    
[^80]: 基于大语言模型的主动对话系统中的交互式后门攻击

    Interactive Backdoor Attacks in LLM-Based Proactive Dialogue Systems

    [https://arxiv.org/abs/2611.06969](https://arxiv.org/abs/2611.06969)

    该论文提出了一种针对大语言模型主动对话系统的交互式后门攻击框架，通过训练数据投毒植入触发条件下的引导策略偏差，使普通用户话语即可将触发后的多轮交互引导至攻击者指定目标，无需部署时修改提示词或攻击者进一步干预。

    

    基于大语言模型的主动对话系统通过持续的交互提供面向目标的支持。在外部提供的对话数据上对底层模型进行微调会引入后门风险。现有的大语言模型后门研究主要关注在触发器激活时诱发攻击者指定的响应。我们提出了一个交互式后门攻击框架，利用训练数据投毒将触发后的多轮交互引导至攻击者指定的目标。被投毒的微调对话会植入一种由触发条件引导的策略偏差，普通用户的话语即可将其激活，而无需在部署时修改提示词，也无需攻击者的进一步干预。具体而言，我们在逐步扩展的触发后交互设置下实例化了三种互补的攻击：（1）直接攻击，其目标是触发器激活时立即产生响应偏离，保留了传统的“触发-响应”设置（摘要在此处截断）。

    arXiv:2611.06969v1 Announce Type: cross  Abstract: LLM-based proactive dialogue systems provide goal-directed support through sustained interaction. Fine-tuning their underlying models on externally supplied dialogue data introduces backdoor risks. Existing LLM backdoor studies primarily focus on eliciting an attacker-specified response upon trigger activation. We present an interactive backdoor attack framework that uses training-data poisoning to steer post-trigger, multi-turn interactions toward an attacker-specified objective. Poisoned fine-tuning dialogues implant a trigger-conditioned guidance-policy bias that ordinary user utterances can activate, without deployment-time prompt modification or further attacker intervention. Specifically, we instantiate three complementary attacks under progressively expanded post-trigger interaction settings: (1) Direct Attack, which targets an immediate response deviation upon trigger activation, retaining the conventional trigger--response set
    
[^81]: Octopus：一个大脑，众多分支——面向潜在KV多智能体强化学习

    Octopus: One Brain, Many Branches for Latent-KV Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.06956](https://arxiv.org/abs/2611.06956)

    Octopus提出了一种单模型多智能体强化学习框架，让所有分支共享同一主干网络，并通过零拷贝基数分叉、仅注意力KV挂载等协同设计机制，在GPU驻留的KV状态有向无环图上进行高效协调，从而消除KV缓存重复复制和跨分支依赖丢失的问题。

    

    多智能体LLM系统通过生成具有隔离上下文窗口的子智能体来并行探索，而单智能体系统则将每次搜索和验证步骤追加到一条不断增长的轨迹中。然而，现有多智能体框架会在各分支间重复复制前缀KV缓存，通过重新预填充文本摘要来交换发现结果，并在强化学习（RL）过程中将跨分支依赖关系扁平化或丢弃。我们提出了Octopus，这是一个单模型多智能体强化学习框架，其中所有线程共享同一个主干网络，并通过五个原语（fork、read_thread、wait_threads、rewind、commit）在驻留于GPU的有向无环图（DAG）KV状态上进行协调。Octopus结合了三个协同设计的机制：(1) 零拷贝基数分叉，将分支生成为共享物理KV页面上的O(1)复杂度游标视图；(2) 仅注意力KV挂载，将同级线程的末端KV块直接挂载到全注意力层中……（摘要在此处被截断）

    arXiv:2611.06956v1 Announce Type: cross  Abstract: Multi-agent LLM systems explore in parallel by spawning sub-agents with isolated context windows, whereas single-agent systems append every search and verification step into one growing trajectory. However, existing multi-agent frameworks duplicate prefix KV caches across branches, exchange findings by re-prefilling text summaries, and flatten or drop cross-branch dependencies during reinforcement learning (RL).   We introduce Octopus, a single-model multi-agent RL framework in which all threads share one backbone and coordinate over a GPU-resident directed acyclic graph (DAG) of KV states through five primitives (fork, read_thread, wait_threads, rewind, commit). Octopus combines three co-designed mechanisms: (1) Zero-Copy Radix Forking, which spawns branches as $\mathcal{O}(1)$ cursor views over shared physical KV pages; (2) Attention-Only KV Mounting, which mounts a sibling thread's terminal KV blocks directly into full-attention lay
    
[^82]: 互联网文化与大型语言模型中的男性抽象偏差

    The Male Abstraction Bias in Internet Culture and Large Language Models

    [https://arxiv.org/abs/2611.05730](https://arxiv.org/abs/2611.05730)

    研究发现互联网文化和大型语言模型中存在系统性的“男性抽象偏差”——越抽象的概念越与男性相关联，该偏差跨越不同语言、概念领域和语法结构（动词、副词更抽象且更男性化），并被人类内化和AI算法复制。

    

    抽象性和性别是人类认知的两个基础维度，通常被作为相互独立的变量来研究。本研究发现，它们在覆盖单词表达（N = 40,000）和多词表达（N = 67,000）的整个概念空间中系统性关联：在互联网上数十亿词汇和数百万图像中——并经过多种测量技术的验证——越抽象的概念越与男性相关联，这一效应在世界各地的语言中均可复现。该模式不仅在各个概念领域普遍成立，还映射到语法结构上：动词和副词更抽象且更带有男性色彩，这是强调男性主动性特质的刻板印象在系统层面的痕迹。此外，这种偏差还出现在人类和AI在自由联想任务中的反应里，表明其已被人类内化并被算法复制。男性抽象偏差可能是多种刻板印象的共同基础，并将其统一起来……

    arXiv:2611.05730v1 Announce Type: cross  Abstract: Two foundational aspects of human cognition, abstractness and gender, are typically examined as separable. Here, we find they are systematically linked throughout the entire space of concepts spanning single (N = 40,000) and multi-word expressions (N = 67,000): across billions of words and millions of images on the internet -- and across numerous measurement techniques -- more abstract concepts are more associated with men, an effect which replicates in languages around the world. This pattern not only holds across conceptual domains, but it also maps onto grammatical structure: verbs and adverbs are more abstract and more male-coded, a system-wide trace of stereotypes accentuating agentic qualities in men. Furthermore, this bias emerges in human and AI responses in free association tasks, indicating its internalization and its algorithmic reproduction. The male abstraction bias may underlie and unify a range of stereotypes, evidencing
    
[^83]: 用于修复多语言语音识别错误的情境化象形图

    Contextual Pictograms for Repairing Multilingual ASR Errors

    [https://arxiv.org/abs/2611.05637](https://arxiv.org/abs/2611.05637)

    提出了一种结合备选转录文本、语音回放和情境化象形图校验的交互界面，帮助听者在多语言（法语、西班牙语）语音识别错误发生时恢复说话者的真实意图。

    

    当相互竞争的词候选不熟悉或在视觉上相似时，语音转录错误可能难以被察觉。我们描述了一种界面，它将备选转录文本、语音回放以及针对选定不确定概念的情境化校验象形图相结合。该系统最初将支持法语和西班牙语的功能性话语。一项平衡顺序的对照研究将评估图像反馈能否在仅依赖文本备选项之外，帮助听者恢复说话者想要表达的动作。我们还将量化语义不匹配以及额外的交互时间，因为不恰当的图像可能会强化错误的理解而非修复它。

    arXiv:2611.05637v1 Announce Type: new  Abstract: Speech transcription errors can be difficult to recognize when competing word hypotheses are unfamiliar or visually similar. We describe an interface that combines alternative transcripts, replay and contextually checked pictograms for selected uncertain concepts. The system will initially support French and Spanish functional utterances. A counterbalanced study will assess whether pictorial feedback helps listeners recover the speaker's intended action beyond text alternatives alone. We will also quantify semantic mismatch and additional interaction time, since an inappropriate image can reinforce the wrong interpretation rather than repair it.
    
[^84]: Sonar-MoE：用于内存高效混合专家训练的反向传播内专家更新

    Sonar-MoE: In-Backward Expert Updates for Memory-Efficient Mixture-of-Experts Training

    [https://arxiv.org/abs/2611.03577](https://arxiv.org/abs/2611.03577)

    Sonar-MoE通过在反向传播过程中即时更新并释放专家梯度，将MoE训练的峰值内存降低19%–22%、每步速度提升17%–23%，同时保持与标准训练位级别一致的结果。

    

    混合专家模型的大多数参数都属于其专家模块，而标准训练会将每个专家的梯度保留在GPU内存中，直到优化器步进为止。这种内存占用会随着专家数量和规模的增长而增长，因此随着MoE模型的扩展而不断上升。我们提出了Sonar-MoE，它在反向传播过程中更新每个专家张量并立即释放其梯度。该方法将“反向传播内优化器”从稠密模型扩展到批处理的MoE专家张量，使GPU永远不会持有超过一层的专家梯度。在使用两种专家内核后端以bf16精度微调OLMoE-1B-7B和Granite-3.1-3B-A800M时，Sonar-MoE将峰值内存降低了19%到22%。其更新与不使用专家梯度裁剪的标准训练在位级别完全一致，留出损失与所有基线的差距保持在0.003以内。与使用全局范数裁剪的标准训练相比，每步训练时间也缩短了17%到23%。更新整个批处理张量而非单个张量……

    arXiv:2611.03577v1 Announce Type: cross  Abstract: Most parameters of a Mixture-of-Experts (MoE) model belong to its experts, and standard training keeps every expert gradient in GPU memory until the optimizer step. This memory grows with the number and size of the experts, so it rises as MoE models scale. We introduce Sonar-MoE, which updates each expert tensor during the backward pass and frees its gradient right away. It brings optimizer-in-backward from dense models to batched MoE expert tensors, so the GPU never holds more than one layer of expert gradients. When we fine-tune OLMoE-1B-7B and Granite-3.1-3B-A800M in bf16 with two expert-kernel backends, Sonar-MoE lowers peak memory by 19% to 22%. Its updates are bitwise identical to standard training without expert gradient clipping, and held-out loss stays within 0.003 of all baselines. Each step also takes 17% to 23% less time than standard training with global-norm clipping. Updating whole batched tensors rather than individual 
    
[^85]: FindLink：基于来源锚定前提学习评估科学关联

    FindLink: Learning to Assess Scientific Connections from Source-Grounded Premises

    [https://arxiv.org/abs/2611.02545](https://arxiv.org/abs/2611.02545)

    该论文提出 GI-TraceBench 基准与 FindLink 两阶段框架，通过构建来源锚定前提并进行结构化推理，结合确定性编译器反馈，来评估不同研究条件下科学发现之间的关联。

    

    arXiv:2611.02545v1 公告类型：新论文。摘要：基于文献的科学发现需要评估在不同研究条件下所得发现之间的关联。来源归因使单个发现可追溯，但并不能确定它们共同支持什么。我们提出 GI-TraceBench，一个包含 10,282 个问题的胃肠肿瘤学基准，通过共享的论文、主张和证据片段记录，统一了来源溯源问答、知识发现和假设评估三类任务。我们进一步提出 FindLink，一个面向来源锚定科学推理的两阶段框架。前提锚定阶段构建带有明确研究背景和支持状态的来源绑定前提；结构化推理阶段将这些前提组织成同时表示支持关系与未决条件的支持计划，并利用确定性编译器反馈生成以证据为条件的答案。该编译器验证结构与来源绑定的一致性，但……（原文在此处截断）

    arXiv:2611.02545v1 Announce Type: new  Abstract: Scientific discovery from the literature requires assessing connections among findings obtained under different study conditions. Source attribution makes individual findings traceable, but does not determine what they jointly support. We introduce GI-TraceBench, a benchmark of 10,282 questions in gastrointestinal oncology that unifies provenance question answering, knowledge discovery, and hypothesis evaluation through shared Paper, Claim, and Evidence Span records. We further propose FindLink, a two-stage framework for provenance-grounded scientific reasoning. Premise Grounding constructs source-bound premises with explicit study context and support status. Structured Reasoning organizes these premises into support plans that represent both supported relations and unresolved conditions, then generates evidence-conditioned answers using deterministic compiler feedback. The compiler verifies structural and source-binding consistency, but
    
[^86]: 以模型读取的方式书写事实：破解两跳诅咒

    Write Facts the Way Models Read Them: Lifting the Two-Hop Curse

    [https://arxiv.org/abs/2611.01519](https://arxiv.org/abs/2611.01519)

    论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。

    

    arXiv:2611.01519v2 公告类型：新 摘要：通过微调学习新事实的语言模型能够复述该事实，却无法用它进行推理：教会模型某个人是新国家公民之后，它仍然无法在不借助思维链的情况下说出这个人所在国家的国家元首是谁。这种“两跳诅咒”通常被视为推理能力缺陷，而最强的知识编辑方法则通过合成推理数据来解决它。我们证明：推理能力是完好的，错的是事实的书写方式。微调把新事实写成一个“要说出口的答案”，而第二跳推理从中间层开始就把桥接短语（“那个人的国家”）当作实体来读取：在相应层将模型自身对该实体的表征修补进桥接短语，可以基本恢复第二跳推理；然而微调之后，桥接短语与该新实体的接近程度并不比编辑前更高。因此我们提出“读取等价书写”（Legible）：用一个损失项让桥接短语在第二跳读取它的那一层（摘要在此处截断）

    arXiv:2611.01519v2 Announce Type: new  Abstract: A language model that learns a new fact by fine-tuning can repeat it but cannot reason with it: taught that a person is a citizen of a new country, it still cannot say, without chain-of-thought, who the head of state of that person's country is. This two-hop curse is widely treated as a reasoning deficit, and the strongest knowledge editors answer it with synthesized reasoning data. We show that the reasoning is intact and the writing is wrong. Fine-tuning writes a new fact as an answer to be said, whereas the second hop reads the bridge ("that person's country") as an entity, from the middle layers on: patching the model's own representation of the entity into the bridge at those layers largely restores the second hop, yet after fine-tuning the bridge there is no closer to the new entity than before the edit. We therefore propose read-equivalent writing (Legible): one loss term makes the bridge phrase, at the layer where the second hop 
    
[^87]: 多重估计引导并不能统一地消除语言模型判断中的偏差

    Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments

    [https://arxiv.org/abs/2611.00597](https://arxiv.org/abs/2611.00597)

    本研究通过预注册实验发现，在单次回复中引导语言模型给出多个加权估计值的格式并不能统一地消除其认知偏差，在锚定和现状等任务中偏差效应已接近于零，尽管多项效应量低于预设的最小感兴趣效应量，但并未能确立统计等效性。

    

    在单次回复中引导模型给出五个加权估计值，是否会改变语言模型对信息的响应方式？在锚定、后见之明、基础比率、合取以及现状/默认等任务中，我们将直接数值回答与多重估计响应格式进行对比，同时比较了不同的来源框架、显式随机线索以及概率一致性约束。我们评估了三种已完成配置——gpt-6.1-sol、claude-sonnet-5.5 以及 Qwen3-8B（禁用推理，进行描述性分析）。在前两种配置中，直接引导产生的无关锚定指数和现状效应接近于零，同时模型仍能对信息性来源和知情建议保持敏感。在23项预注册的验证性测试中，有六项在经过 Holm 校正后被拒绝；然而，这六项测试的绝对点估计值均低于预先设定的最小感兴趣效应量，但并未据此建立等效性结论。此外，虽然分离引导与联合引导的（摘要在此处截断）

    arXiv:2611.00597v1 Announce Type: new  Abstract: Does eliciting five weighted estimates within a single response change how a language model responds to information? Across anchoring, hindsight, base-rate, conjunction, and status-quo/default tasks, we compare direct numerical answers with a multi-estimate response format, contrasting source framings, explicitly random cues, and probability-consistency constraints. We evaluate three completed configurations--gpt-6.1-sol, claude-sonnet-5.5, and Qwen3-8B (reasoning disabled, analyzed descriptively). In the first two configurations, direct elicitation yielded near-zero irrelevant-anchor indices and status-quo effects, while remaining responsive to informative sources and informed advice. Of 23 preregistered confirmatory tests, six rejected after Holm adjustment; however, all six absolute point estimates fell below their prespecified smallest effect sizes of interest without establishing equivalence. Moreover, while separate versus joint el
    
[^88]: TSBPO：面向交互式语言智能体策略优化的树结构分支方法

    TSBPO: Tree-Structured Branching for Policy Optimization in Interactive Language Agents

    [https://arxiv.org/abs/2610.41697](https://arxiv.org/abs/2610.41697)

    TSBPO提出一种树结构的采样到更新范式，通过似然拟合的根/分支分配器定向采样、基于兄弟片段与匹配观察状态的延迟反馈分配、以及跨活跃轮次归一化的词元损失聚合，在单一训练循环中解决了从终端结果训练交互式语言智能体的三大耦合难题。

    

    从终端结果训练交互式语言智能体面临三个相互耦合的问题：应采样哪些任务与状态延续、如何将延迟反馈分配到各个决策上，以及如何为多词元动作聚合词元损失。我们提出TSBPO，一种树结构的“采样到更新”范式，在单一训练循环内解决上述问题。一个基于似然拟合的根/分支分配器利用保留的任务与状态证据，在固定rollout预算下将延续采样定向到可能产生对比性终端结果的分支上。在由此构建的重放物理树中，轨迹优势通过比较直接相邻的兄弟片段来计算，而观察组步骤优势则在匹配的任务/观察状态下比较不同结果。随后，执行器对每个完整响应求和裁剪的词元代理项，并在活跃轮次间进行归一化，从而避免了策略梯度项中的响应长度平均。使用Qwen2.5-1.5B作为执行器，固定检查点……（摘要原文在此处截断）

    arXiv:2610.41697v1 Announce Type: cross  Abstract: Training interactive language agents from terminal outcomes poses three coupled problems: which task and state continuations to sample, how to assign delayed feedback to decisions, and how to aggregate token losses for multi-token actions. We present TSBPO, a tree-structured sampling-to-update paradigm that addresses these problems within one training loop. A likelihood-fitted Root/Branch allocator uses retained task and state evidence to target continuations likely to produce contrasting terminal outcomes under fixed rollout budgets. On the resulting replayed physical trees, trajectory advantages compare immediate sibling segments, while observation-group step advantages compare outcomes at matched task/observation states. The actor then sums clipped token surrogates over each complete response and normalizes across active turns, avoiding response-length averaging in the policy-gradient term. With a Qwen2.5-1.5B actor, fixed checkpoin
    
[^89]: V-LUMEN：用于视觉-语言模型嵌入扩展的视觉查找记忆

    V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models

    [https://arxiv.org/abs/2610.37019](https://arxiv.org/abs/2610.37019)

    V-LUMEN提出了一种视觉查找记忆机制，通过冻结码本离散化视觉特征、空间聚合哈希构建记忆键并结合文本条件调制信号，以完全非参数化的寻址方式为视觉-语言模型实现嵌入层面的扩展，显著提升冻结主干网络在推理基准上的表现并具备零样本迁移能力。

    

    嵌入层面的扩展通过基于查找的记忆来扩展语言模型的容量，而非通过增加模型的深度、宽度或密集计算。将这一思想扩展到视觉-语言模型（VLM）具有挑战性，因为视觉特征是连续的、具有空间结构的、且依赖于查询的，这与天然支持n-gram查找的离散文本标记不同。我们提出了V-LUMEN，一种面向视觉-语言模型的视觉查找记忆。V-LUMEN使用冻结的码本对视觉特征进行离散化，通过基于空间聚合的哈希构建空间感知的记忆键，并使用文本条件化的相关性信号对其进行调制；所有寻址过程都是非参数化的，检索到的嵌入通过门控残差通路融合到主干网络中。在五个推理密集型基准测试上，V-LUMEN显著提升了冻结的Qwen3-VL-2B主干网络的性能，并能零样本迁移到三个图像与训练数据完全不相交的基准测试上。

    arXiv:2610.37019v1 Announce Type: cross  Abstract: Embedding-level scaling expands the capacity of language models through lookup-based memory rather than by increasing model depth, width, or dense computation. Extending this idea to vision-language models (VLMs) is challenging because visual features are continuous, spatially structured, and query-dependent, unlike discrete text tokens that naturally support n-gram lookup. We introduce V-LUMEN, a visual lookup memory for VLMs. V-LUMEN discretizes visual features with a frozen codebook, builds spatially aware memory keys through spatial aggregation-based hashing, and modulates them with text-conditioned relevance signals; all addressing is non-parametric, and the retrieved embeddings are fused into the backbone through a gated residual pathway. On five reasoning-intensive benchmarks, V-LUMEN substantially improves a frozen Qwen3-VL-2B backbone and transfers zero-shot to three benchmarks whose images are disjoint from the training data.
    
[^90]: AutoVend：基于长周期自动售货机运营的大语言模型智能体基准测试

    AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations

    [https://arxiv.org/abs/2610.36980](https://arxiv.org/abs/2610.36980)

    AutoVend是一个长周期基准测试，让LLM智能体通过网络界面管理分布在五类站点的20台自动售货机180个模拟日，评估其O2O商务决策能否转化为持续经济回报，结果显示五个智能体中有四个不敌基于规则的库存控制基线。

    

    大语言模型（LLM）智能体正在将网络上的智能体中介商务从为消费者购物扩展到为卖家经营业务。在在线到线下（O2O）商务中，决策通过网络平台发布，但需要通过成本高昂的物理执行来实现。评估此类智能体需要超越单笔交易，考察在线做出的决策能否转化为持续的经济回报。我们提出了AutoVend，这是一个基准测试，其中一个LLM智能体通过基于网络的运营界面管理分布在五种站点类型中的20台自动售货机，为期180个模拟日。其基于真实数据的模拟器结合了共享仓库、采购提前期、付费补货、库存过期以及经常性运营成本等要素。我们评估了五个LLM智能体，并将其与基于规则的库存控制基线进行比较，诊断它们的站点协调能力、执行能力以及随时间推移的表现。在被评估的五个LLM智能体中，有四个表现不及基线……

    arXiv:2610.36980v1 Announce Type: new  Abstract: Large language model (LLM) agents are extending agent-mediated commerce on the Web from shopping for consumers to operating businesses for sellers. In online-to-offline (O2O) commerce, decisions are issued through Web-based platforms but realized through costly physical execution. Evaluating such agents requires going beyond individual transactions to ask whether decisions made online turn into sustained economic returns. We introduce AutoVend, a benchmark in which one LLM agent manages 20 vending machines across five site types through a Web-based operating interface for 180 simulated days. Its data-grounded simulator combines a shared warehouse, procurement lead times, paid replenishment, inventory expiration, and recurring operating costs. We evaluate five LLM agents against a rule-based inventory-control baseline and diagnose their site coordination, execution, and performance over time. Four of five evaluated LLM agents fall short o
    
[^91]: 伏尼契手稿有意义吗？校准测试几乎未发现语言的痕迹

    Does the Voynich Manuscript Mean Anything? Calibrated Tests Find Little Trace of Language

    [https://arxiv.org/abs/2610.34499](https://arxiv.org/abs/2610.34499)

    通过将伏尼契手稿的两种起源假说写成程序并进行校准检验，研究发现抄写员复制假说比单密钥密码更能预测文本统计特征，且手稿中的词对同行模式强度仅为真实语言文本的一半，表明手稿几乎不含真实语言的痕迹。

    

    伏尼契手稿是一本十五世纪的书，其文字无人能读懂。一种说法认为其文本毫无意义，是一位抄写员通过复制自己的话语并稍作改动而造出来的；另一种说法则认为一种真实的语言隐藏在密码背后。我们将这两种说法分别写成能够填充手稿页面的程序，用相同的预算拟合每个程序，并在拟合过程中从未见过的统计数据上进行检验。Timm和Schinner提出的复制过程对统计数据的预测效果远好于任何单密钥密码。但在手稿的两种方言之间切换密钥的密码，或仅对真实文本词类进行编码的密码，同样能很好地预测这些统计数据——仅凭词汇统计无法做出判断。我们进而寻找一种无需破译即可识别的语言痕迹：在16种语言的28篇文本中，相同的词对会持续共享同一行，而手稿仅以这些文本中最弱者一半的强度呈现这种模式。

    arXiv:2610.34499v1 Announce Type: new  Abstract: The Voynich manuscript is a fifteenth-century book in a script that no one can read. One account holds that its text means nothing and that a scribe made it by copying his own words and changing them a little. The other holds that a real language lies hidden behind a cipher. We write each account as a program that fills the pages of the manuscript, fit every program with the same budget, and check it on statistics the fit never saw. The copying process of Timm and Schinner predicts these statistics far better than any cipher with one key. But ciphers that switch keys between the manuscript's two dialects, or that encode only the word classes of a real text, predict them equally well. Word statistics alone cannot decide. We look for a trace of language that needs no decipherment. In 28 texts in 16 languages the same word pairs keep sharing a line. The manuscript shows this pattern at half the strength of the weakest of these texts. A word
    
[^92]: 两种不确定性并非同一种：廉价结构化内容审核中的分歧与决策风险

    Two Uncertainties Are Not One: Disagreement versus Decision-Risk in Cheap Structured Content Moderation

    [https://arxiv.org/abs/2610.33258](https://arxiv.org/abs/2610.33258)

    通过对一个低成本“系统一”内容审核模型在14万余条人工标注评论上的首次独立审计，发现其分类准确率（AUROC 0.83）与14个前沿推理模型中的13个在统计上不可区分且成本低1-3个数量级，并指出“置信度”混淆了“分歧”与“决策风险”两种本质不同的不确定性。

    

    内容审核将带有等级、且存在争议的人类判断压缩为针对数十亿条内容的二元行为（保留/删除），在这样的规模下，对每条帖子都运行前沿大语言模型是不经济的。一类新型的冻结“系统一”模型返回类型化、经过校准的决策而非文本，每次调用成本约0.000025美元，耗时约100毫秒。我们在142,658条人工标注的评论上对其中一个此类模型进行了首次独立审计，并在共享的探测集上将其与14个前沿推理模型（GPT-5.x、Claude、Gemini 3和Grok系列）进行基准对比。有两个发现尤为突出。第一，廉价模型在准确率上具有竞争力：其零样本分类AUROC（0.83）与14个前沿系统中的13个在统计上不可区分（配对自助法，n=56；这是未能拒绝等价性，而非等价性的证明），而成本却低1-3个数量级，且其校准程度不逊于其中若干模型。第二，“置信度”混淆了两种不确定性……（原文摘要至此截断）

    arXiv:2610.33258v1 Announce Type: new  Abstract: Content moderation compresses a graded, contested human judgment into a binary act (leave up / take down) over billions of items, at a scale where running a frontier LLM on every post is uneconomical. A new class of frozen "System One" models returns typed, calibrated decisions instead of text at ~USD 0.000025 per call and ~100 ms. We run the first independent audit of one such model on 142,658 human-labeled comments, and benchmark it against 14 frontier reasoning models (the GPT-5.x, Claude, Gemini 3, and Grok families) on a shared probe set. Two findings stand out. First, the cheap model is accuracy-competitive: its zero-shot classification AUROC (0.83) is not statistically separable (paired bootstrap, n=56; a failure to reject parity, not proof of equivalence) from 13 of the 14 frontier systems while costing 1-3 orders of magnitude less, and it is no worse calibrated than several of them. Second, "confidence" conflates two uncertainti
    
[^93]: 各有各的不足：跨不同证据状态校准语言模型

    Each Insufficient in Its Own Way: Calibrating Language Models Across Varying Evidence States

    [https://arxiv.org/abs/2610.33237](https://arxiv.org/abs/2610.33237)

    提出证据状态校准的新范式与SuffiCE方法，通过微调模型下一词元分布中已有的置信度，使模型置信度随证据状态的变化而与其实际回答正确的概率保持一致。

    

    语言模型如何知道自己是否拥有足够的证据来回答问题？即使现有证据不充分，语言模型也倾向于直接作答。决定是回答还是继续收集更多证据，需要对“在现有证据下答案正确的可能性”进行校准估计。然而，大多数校准工作都将输入视为固定的，而没有探究这一估计应如何随证据的变化而变化。我们提出了“证据状态校准”（evidence-state calibration），它要求模型在回答前的置信度与其正确回答的概率相匹配，并且每当证据改变这一概率时，置信度也应相应变化。我们提出了SuffiCE（充分性置信度估计，Sufficiency Confidence Estimation），该方法对模型下一词元分布中已存在的置信度进行微调，使其匹配模型在每种证据状态下回答正确的频率。在临床诊断和多跳问答任务上……

    arXiv:2610.33237v1 Announce Type: new  Abstract: How can a language model know whether it has enough evidence to answer? Language models tend to answer even when the evidence they have is insufficient. Deciding whether to answer or gather more evidence requires a calibrated estimate of how likely an answer is to be correct with the available evidence. However, most calibration work treats the input as fixed, without asking how this estimate should change with the evidence. We introduce \emph{evidence-state calibration}, which requires a model's confidence before answering to match its probability of answering correctly and to change accordingly whenever the evidence changes this probability. We propose \ours{} (\textbf{Suffi}ciency \textbf{C}onfidence \textbf{E}stimation), which fine-tunes the confidence already present in a model's next-token distribution to match how often the model answers correctly from each evidence state. Across clinical diagnosis and multi-hop question answering
    
[^94]: P2N：重用深层表示以获得更大的有效计算深度

    P2N: Reusing Deep Representations for Greater Effective Depth

    [https://arxiv.org/abs/2610.25076](https://arxiv.org/abs/2610.25076)

    P2N通过将前一个token的深层隐藏状态加到当前token的浅层表示上，以每步仅一次向量加法的极小开销，将模型的有效计算深度跨token递归扩展，从而在不增加层数的前提下显著提升语言模型的推理能力。

    

    数学问题求解和自主智能体是语言模型日益重要的应用，它们需要更强的推理能力。计算深度是这类推理的关键资源，但增加模型层数会同时增加训练和推理成本，而测试时扩展（test-time scaling）则会带来大量额外的推理计算。我们提出了P2N（Previous to Next），该方法重用前一个token在生成过程中已经可用的深层表示，以几乎可忽略的解码开销来增加有效计算深度。具体而言，P2N将前一个token的深层隐藏状态添加到当前token的浅层表示中，使后续层能够建立在之前已完成的深度计算之上。这种递归方式将有效计算深度扩展到跨token范围，同时每个解码步骤仅引入一次向量加法。我们使用并行Jacobi更新来训练这种递归机制……

    arXiv:2610.25076v2 Announce Type: replace  Abstract: Mathematical problem solving and autonomous agents are increasingly important applications of language models, demanding stronger reasoning capabilities. Computational depth is a key resource for such reasoning, but adding model layers increases both training and inference costs, while test-time scaling incurs substantial additional inference computation. We introduce P2N (Previous to Next), which reuses the previous token's deep representations, already available during generation, to increase effective computational depth with negligible decoding overhead. Specifically, P2N adds the previous token's deep hidden state to the current token's shallow representation, allowing subsequent layers to build on previously completed deep computation. This recurrence extends effective computational depth across tokens, while introducing only a single vector addition per decoding step. We train the recurrence using parallel Jacobi updates restr
    
[^95]: MemForm：LLM智能体中的记忆形成投毒基准测试

    MemForm: Benchmarking Memory Formation Poisoning in LLM Agents

    [https://arxiv.org/abs/2610.14431](https://arxiv.org/abs/2610.14431)

    该论文提出MemForm基准，将LLM智能体的记忆形成投毒形式化为对记忆产生与提交界面的对抗性操纵，首次引入选择投毒与路径投毒两种干预类别，并揭示六种主流大模型在全部五种界面上均存在广泛易感性。

    

    长期记忆使LLM智能体能够从过去的交互中学习，但也使记忆形成过程暴露于对抗性影响之下。现有研究主要按攻击通道、破坏模式或生命周期阶段来组织记忆投毒，而对记忆形成界面的刻画不足。我们将“记忆形成投毒”形式化为对持久性智能体记忆的产生与提交接口的对抗性操纵，并引入MemForm——一个受控诊断基准，涵盖五种界面：内容投毒、证据投毒、范围投毒、选择投毒和路径投毒。我们提出选择投毒与路径投毒作为受控干预类别，分别用于隔离保留效应与巩固顺序效应。每个类别都将干净条件与被投毒条件配对，并配有界面特定的控制措施。在六个大语言模型和两种响应格式上的评估显示，所有五种界面均存在广泛的易感性。

    arXiv:2610.14431v2 Announce Type: replace-cross  Abstract: Long-term memory enables LLM agents to learn from past interactions but exposes memory formation to adversarial influence. Existing studies primarily organize memory poisoning by attack channels, corruption patterns, or lifecycle stages, leaving formation interfaces insufficiently characterized. We formulate Memory Formation Poisoning as adversarial manipulation of the interfaces through which persistent agent memory is produced and committed, and introduce MemForm, a controlled diagnostic benchmark covering five interfaces: Content, Evidence, Scope, Selection, and Path Poisoning. We introduce Selection and Path Poisoning as controlled intervention categories that isolate retention and consolidation order effects, respectively. Each category pairs clean and poisoned conditions with interface-specific controls. Evaluations across six LLMs and two response formats reveal broad susceptibility at all five interfaces under the evalu
    
[^96]: 超越最终嵌入：从词元级证据中检索记忆

    Beyond the Final Embedding: Retrieving Memories from Token-Level Evidence

    [https://arxiv.org/abs/2610.13277](https://arxiv.org/abs/2610.13277)

    该论文提出一种无需额外训练的稀疏词元匹配方法，利用嵌入模型已有的词元级上下文表示捕捉局部证据，并结合整条记忆相似度对记忆排序，在八个基准测试和多个模型家族上显著提升检索效果，还可通过可选的低秩查询适配器进一步增强。

    

    记忆增强型助手从存储的对话和文档中检索证据以回答查询。大多数基于嵌入的检索器比较查询和记忆的单向量表示，即使相关性往往只取决于内容的一小部分。这种局部证据可以利用同一嵌入模型已经计算出的上下文词元表示与查询进行匹配。我们的方法使用稀疏词元匹配来改进检索，且无需额外训练。每个查询词元在候选记忆中检索一小撮匹配的词元，匹配得分针对每条记忆进行聚合，并与整条记忆的相似度相结合来对记忆进行排序。可选的低秩查询适配器可以在监督下进行训练，同时保持编码器和已存储的记忆表示固定。我们的方法无需额外训练即可在八个基准测试和多个模型家族上提升检索性能。查询适配器可进一步……（原文在此处截断）

    arXiv:2610.13277v1 Announce Type: cross  Abstract: Memory-augmented assistants retrieve evidence from stored conversations and documents to answer queries. Most embedding-based retrievers compare single-vector representations of queries and memories, even when relevance depends on a small part of the content. Such local evidence can be matched to queries using the contextual token representations already computed by the same embedding models. Our method uses sparse token matching to improve retrieval without additional training. Each query token retrieves a small set of matching tokens across candidate memories. The matching scores are aggregated for each memory and combined with whole-memory similarity to rank memories. An optional low-rank query adapter can be trained with supervision while keeping the encoder and stored memory representations fixed. Our method improves retrieval across eight benchmarks and multiple model families without additional training. Query adaptation further
    
[^97]: SEABench：自进化智能体中内生失准的基准测试

    SEABench: Benchmarking Endogenous Misalignment In Self-Evolving Agents

    [https://arxiv.org/abs/2609.35596](https://arxiv.org/abs/2609.35596)

    该论文提出了SEABench基准，用于量化研究自进化智能体因自我修改而在无外部对抗影响下产生的内生失准（不安全行为）风险。

    

    自进化大语言模型智能体因其能够在部署后通过修改自身框架（包括控制器指令、记忆管理协议以及可复用的工具和技能）来响应用户和环境反馈并持续改进而受到广泛关注。然而，在局部看似有用的更新可能会延续到后续任务中，即使没有直接的对抗性影响，也会产生不安全的行为。为了研究这一风险，我们提出了SEABench，一个用于研究由智能体自进化引发的内生失准的基准，包含48个纵向任务序列，涵盖丰富的个人助理环境中多种进化层面、任务领域和危害类型。为了考虑智能体操作中固有的随机性，我们提供了一个自适应轨迹发现流水线，在保持原始任务意图的同时探测故障，并通过配对的非进化智能体等机制支持因果归因。

    arXiv:2609.35596v2 Announce Type: replace-cross  Abstract: Self-evolving LLM agents have gained prominence for their ability to improve after deployment by modifying their harness, including their controller instructions, memory management protocols, and reusable tools and skills, in response to user and environment feedback. However, locally useful updates may persist into later tasks where they produce unsafe behavior, even without direct adversarial influence. To study this risk, we introduce SEABench, a benchmark for studying endogenous misalignment arising from agent self-evolution, with 48 longitudinal task sequences that span multiple evolution surfaces, task domains, and harm types in a rich personal-assistant environment. To account for the stochasticity inherent in agentic operations, we provide an adaptive trajectory discovery pipeline that probes for failures while preserving original task intent and supports causal attribution through paired non-evolving agents and attribu
    
[^98]: ToolSearcher：基于强化学习的大规模工具选择优化

    ToolSearcher: Optimizing Tool Selection at Scale via Reinforcement Learning

    [https://arxiv.org/abs/2609.30906](https://arxiv.org/abs/2609.30906)

    本文提出ToolSearcher，一个面向大规模工具选择的新型强化学习框架，通过多轮搜索与细粒度优化，解决了LLM在海量且多样的真实工具库中难以有效搜索、区分和组合工具的挑战。

    

    大型语言模型（LLM）在自然语言处理方面表现出色，但难以与外部环境进行交互。工具学习为将LLM扩展为可执行智能体提供了一条有前景的途径，其中工具选择是成功使用工具的关键前提。现有工作通常假设工具集较小或预先定义，导致大规模工具选择问题未得到充分探索。真实世界的工具库包含庞大且多样的工具集合，使得LLM在上下文长度限制下难以有效地搜索、区分和组合工具。我们将大规模工具选择确定为智能体强化学习的一个新挑战，并指出现有的基于知识的问答强化学习方法在选择工具时无法充分考虑工具间的兼容性，因而不适用于该任务。为应对这一挑战，我们提出了ToolSearcher，一个新颖的强化学习框架，用于实现高效的多轮搜索和细粒度优化。

    arXiv:2609.30906v1 Announce Type: new  Abstract: Large language models (LLMs) excel at natural language processing but struggle to interact with external environments. Tool learning provides a promising way to extend LLMs into actionable agents, where tool selection is a critical prerequisite for successful tool use. Existing work often assumes a small or predefined set of tools, leaving large-scale tool selection underexplored. Real-world repositories contain a vast and diverse array of tools, making it difficult for LLMs to effectively search, distinguish, and compose tools under context-length constraints. We identify large-scale tool selection as a new challenge for agentic reinforcement learning, highlighting that existing RL methods for knowledge-based question answering are inadequate for selecting tools while considering compatibility. To address this challenge, we propose ToolSearcher, a novel RL framework for effective multi-turn search and fine-grained optimization in large-
    
[^99]: 基于检索的开放式评估在何处失效？从长篇医学答案事实性验证中自动归纳错误分类体系

    Where Does Retrieval-Based Open-Ended Evaluation Fail? Automatic Taxonomy Induction from Long-Form Medical Answer Factuality Verification

    [https://arxiv.org/abs/2609.30467](https://arxiv.org/abs/2609.30467)

    该论文针对基于检索的开放式医学事实性验证，自动归纳出两套错误分类体系，将失败分解为五个质量维度上的检索阶段错误与六个连续步骤中的验证器推理错误，并利用LLM-as-Judge流程实现大规模自动化错误标注与分类，无需黄金答案或黄金证据。

    

    检索式事实性评估——即将大语言模型生成的论断与权威医学语料库中的证据进行核对验证——已成为高风险临床环境中可扩展幻觉检测的主流范式。尽管可靠且透明的医学事实验证需求迫切，大多数系统仍采用F1等聚合指标衡量性能，这类指标掩盖了失败发生在何处以及为何发生。现有的RAG诊断方法需要黄金答案或人工标注的黄金证据，而这两者在该场景中均不存在。我们基于对开放式MedExpert数据集以及3个封闭式数据集的案例研究，引入了两套全面的分类体系：将失败分解为沿五个质量维度划分的检索阶段错误，以及分为六个连续步骤的验证器推理错误。我们改造了一种基于LLM-as-Judge的自动模式归纳流程，用以大规模标注证据质量并对验证器推理错误进行分类……

    arXiv:2609.30467v1 Announce Type: new  Abstract: Retrieval-based factuality evaluation, where LLM-generated claims are verified against evidence from authoritative medical corpora, has become the dominant paradigm for scalable hallucination detection in high-stakes clinical settings. Despite the urgency of reliable and transparent medical fact verification, most systems measure performance with aggregate metrics like F1, which obscure where and why failures occur. Existing RAG diagnostics require gold answers or annotated gold evidence, neither of which exists in this regime. We introduce two comprehensive taxonomies, grounded in a case study on the open-ended MedExpert dataset and 3 closed-ended datasets, decomposing failures into retrieval-stage errors along five quality dimensions, and verifier-reasoning errors into six consecutive steps. We adapt an automatic pattern induction pipeline using LLM-as-Judge to label evidence quality and classify verifier reasoning errors at scale, and
    
[^100]: 先筛查再服务：面向1.4亿规模生产级客户体验AI代理的仿真验证方法

    Screen Before You Serve: Simulation for Production Customer Experience AI Agents at 140M Scale

    [https://arxiv.org/abs/2609.30137](https://arxiv.org/abs/2609.30137)

    该论文提出了一种基于假设驱动的仿真工作流，利用合成客户和模拟工具输出，在部署前对大规模生产级客户体验AI代理进行筛查验证，从而避免在线实验对客户信任造成的风险。

    

    客户体验（CX）代理使用工具和大语言模型来处理客户请求，并引导用户与组织的产品进行对话式交互。改进这些代理，尤其是在受监管的行业中，是非常困难的：它们必须检测用户意图、遵循复杂的运营策略并可靠地使用工具。手动端到端测试覆盖范围有限，而在线实验则会让客户直面可能导致信任受损的故障。我们提出了一种基于假设驱动的仿真工作流，用于在部署前筛查候选的CX代理。合成客户会对代理的响应做出反应，模拟的工具输出使多步骤代理工作流无需调用生产后端即可运行。我们在Nubank的Card Delivery代理及其扩展后的继任者Card Management（Nubank在巴西聊天量最高的客服代理）上使用了Snowglobe仿真器。在4个已部署的版本中，仿真与生产环境的版本级二元评估器……

    arXiv:2609.30137v1 Announce Type: new  Abstract: Customer experience (CX) agents use tools and large language models to address customer requests and guide conversational interactions with an organization's products. Improving these agents, especially in regulated industries, is difficult: they must detect intent, follow complex operational policies and use tools reliably. Manual end-to-end testing offers limited coverage, while live experiments expose customers to failures that can erode trust.   We present a hypothesis-driven simulation workflow for screening candidate CX agents before deployment. Synthetic customers react to agent responses and simulated tool outputs enable multi-step agentic workflows without invoking production backends. We use the Snowglobe simulator on Nubank's Card Delivery agent and its expanded successor, Card Management - Nubank's highest-volume chat-support agent in Brazil. Across 4 deployed versions, simulated and production version-level binary evaluator 
    
[^101]: 为评审团喂入维度评分，而非整体判定：基于评分准则分解的视觉语言美学评审融合

    Feed the Panel Dimensions, Not Verdicts: Rubric-Decomposed Fusion of Vision-Language Aesthetic Judges

    [https://arxiv.org/abs/2609.27110](https://arxiv.org/abs/2609.27110)

    由多个视觉语言模型组成的评审团在融合整体美学判定时无法显著超越最佳单模型，而让各模型按人工评分准则对图像进行五维度评分并融合这些分数，则能可靠地击败最佳单个模型。

    

    视觉语言模型（VLM）被部署为图像美学的零样本评审器，而在证据薄弱的情况下，多个模型组成的评审团被推荐为提升此类评审可靠性的方法。在两个人工评分数据集EVA和PARA上，我们发现由整体评判模型组成的评审团，无论是对判定结果取平均还是通过学习到的组合器进行融合，都从未显著优于其最佳成员。评审团的价值取决于其被输入的内容。因此，我们让每个模型依据一份固定的人工编写的评分准则（rubric）对每张图像在五个维度上进行评分，并通过折外组合器跨模型族融合这些分数以及每个模型的判定结果。维度评分确实衡量了其标签所声称的内容：在剔除总体人工评分的影响后，在30个模型-属性组合中的28个里，维度提示比整体提示携带更多特定属性的信息。经过融合，它们在EVA数据集上所有十个三模型族评审团中都击败了最佳的单个VLM。

    arXiv:2609.27110v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are deployed as zero-shot judges of image aesthetics, and panels of several models are recommended, on thin evidence, as the way to make such judges reliable. On two human-rated datasets, EVA and PARA, we find that a panel of holistic judges never significantly beats its best member, whether the verdicts are averaged or fused by a learned combiner. What a panel is worth depends on what it is fed. We therefore have each model score each image on the five dimensions of a frozen, human-written rubric and fuse those scores, alongside each model's verdict, across model families with an out-of-fold combiner. The dimension scores measure what their labels claim: with the overall human score partialled out, a dimension prompt carries more attribute-specific information than the holistic prompt in 28 of 30 model-attribute cells. Fused, they beat the best single VLM in all ten three-family panels on EVA (against tha
    
[^102]: 价值超越语言模型：检测写作中的原创贡献

    Value Over Language Model: Detecting Original Contribution in Writing

    [https://arxiv.org/abs/2609.00700](https://arxiv.org/abs/2609.00700)

    提出了一种无需训练、不评分表面文本的新框架“价值超越语言模型”（Value Over Language Model），通过在不同粒度上提取文档内容并用LLM重建文档，来衡量人在语言模型易于生成的内容之上所贡献的原创价值。

    

    大语言模型（LLM）已在各类写作任务中被迅速采用，这推动了检测LLM生成文本工具的发展。然而，这些工具主要衡量的是文档表面文本中有多少是由LLM撰写的，而并非从根本上设计用于衡量文档中的信息内容或思想有多少源自LLM本身，而非由用户在提示词中提供。在本工作中，我们设计了一个框架，用于衡量一个人在语言模型本身能够轻松生成的内容之上所增加的价值。该方法不需要训练或标注数据，也从不为文档的表面文本打分，从而使其免受文体混淆因素的影响。相反，该方法以递增的粒度级别提取文档内容，使用LLM从每个部分表示中重建文档，并将这些重建结果与仅根据任务描述生成的重建结果进行比较。我们将这一框架称为“价值超越语言模型”（Value Over Language Model）。

    arXiv:2609.00700v1 Announce Type: new  Abstract: LLMs have been rapidly adopted across writing tasks, prompting the development of tools for detecting LLM-generated text. Yet, these tools largely measure how much of a document's surface text was written by an LLM and aren't fundamentally designed to measure how much of the information content or ideas originated from the LLM itself rather than being supplied by the user in the prompt. In this work, we design a framework that measures how much value a person adds on top of what a language model could have easily produced by itself. The method requires no training or labeled data and never scores the document's surface text, insulating it from stylistic confounders. Instead, it extracts the document's content at increasing levels of granularity, uses an LLM to reconstruct the document from each partial representation, and compares these reconstructions with those produced from the task description alone. We call this framework Value Over
    
[^103]: 信任大众：KV缓存驱逐中的强制权重

    Trust the Mass: Forced Weights in KV-Cache Eviction

    [https://arxiv.org/abs/2608.25230](https://arxiv.org/abs/2608.25230)

    本文发现KV缓存驱逐中保留最大权重已接近最优，已发布方法间的差异主要源于存储方式而非选择策略，并揭示了评估中的内存与性能权衡。

    

    arXiv:2608.25230v1 公告类型：交叉 摘要：每个部署的稀疏注意力或KV缓存驱逐规则都会保留一部分键，丢弃其余部分，并对保留集上的注意力权重进行重新归一化。在来自五个模型的168,192个注意力行上，枚举该约束下的精确最优子集表明，保留最大权重已经接近最优，因为最优子集仅将剩余差距中位数缩小了2%到5%。如果选择带来的改进如此之小，那么已发布的驱逐方法之间的差异必然来自其他方面，因此我们测量了每种方法持有的字节数。在共享评估流程中，最强的查询无关方法持有完整缓存，因为它们的按头选择存储为掩码，只有不规则按头存储才能释放该内存。在固定选择上强制执行名义预算会损失14到62个基准点。我们将一个87.6点的检索差距追溯到在问题可见时计算的排名。

    arXiv:2608.25230v1 Announce Type: cross  Abstract: Every deployed sparse-attention or KV-cache-eviction rule keeps a subset of the keys, discards the rest, and renormalizes the attention weights over the kept set. Enumerating the exact best subset under that constraint on $168{,}192$ attention rows from five models shows that keeping the largest weights is already near-optimal, since the best subset closes only a median $2$ to $5\%$ of the remaining gap to full attention. If selection closes this little, published margins between eviction methods must come from elsewhere, so we measure the bytes each method holds. In the shared evaluation pipeline, the strongest query-agnostic methods hold the full cache because their per-head selections are stored as masks, and only ragged per-head storage frees that memory. Enforcing a nominal budget on one fixed selection costs $14$ to $62$ benchmark points. We trace an $87.6$-point retrieval margin to rankings computed while the question is visible
    
[^104]: 大型语言模型能否以具有法律意义的方式进行推理？一项关于欧洲人权法院案例的小规模研究

    Can LLMs Reason in a Legally Meaningful Manner? A Small-scale Study on European Court of Human Rights Cases

    [https://arxiv.org/abs/2608.17168](https://arxiv.org/abs/2608.17168)

    本研究通过欧洲人权法院案例测试发现，顶尖大型语言模型在法律推理上表现不佳，其分析结构完整但内容浅薄，且自动评估器与人工评估一致性较弱。

    

    摘要：arXiv:2608.17168v1 公告类型：交叉 摘要：推理已成为当代大型语言模型的标准技术和特性；然而，在诸如法律案件预测等要求较高的法律导向任务中，其应用和质量仍未得到充分探索。我们以欧洲人权法院（ECtHR）的法律案例为测试平台，调查了大型语言模型在法律案件预测背景下的推理能力。我们评估了近期顶级模型OpenAI GPT 5.4，通过探索在ECtHR判例背景下对“具有法律意义的推理”更具或更少提示性的不同提示策略。我们通过人工和大型语言模型评估来呈现对模型响应的评估结果。我们发现，所考察的模型在法律推理方面远未达到理想分数，模型生成的分析结构完整但实质肤浅，并且“大型语言模型作为法官”的评估器内部一致但与我们的训练注释者仅有微弱一致性。

    arXiv:2608.17168v1 Announce Type: cross  Abstract: Reasoning has become a standard technique and feature for contemporary LLMs; however, its application and quality in the context of demanding legal-oriented tasks, such as legal case forecasting, remain under explored. We investigate how LLMs reason in the context of legal case forecasting, using legal cases from the European Court of Human Rights (ECtHR) as a testbed. We evaluate OpenAI GPT 5.4, a recent top-tier LLM, by exploring alternative prompting strategies that are more or less suggestive of what counts as legally meaningful reasoning in the context of ECtHR jurisprudence. We present our findings derived from assessing the model's responses with both human and LLM evaluation. We find that the examined model scores far from ideal in legal reasoning, the model produces structurally complete but substantively shallow analyses, and that LLM-as-a-Judge evaluators are internally consistent yet align only weakly with our trained annot
    
[^105]: XBridge：面向异构大语言模型通信的实体锚定潜在桥接

    XBridge: Entity-Grounded Latent Bridge for Heterogeneous LLM Communication

    [https://arxiv.org/abs/2608.11676](https://arxiv.org/abs/2608.11676)

    XBRIDGE通过词汇锚定映射和潜在富化桥接机制，解决了异构大语言模型通信中跨架构实体锚定丢失的问题，实现了免解码的高效通信。

    

    摘要：arXiv:2608.11676v1 公告类型：新 摘要：异构多智能体大语言模型系统，其中智能体由不同模型家族驱动，通过减少冗余推理模式，可以优于同构配置。然而，现有通信协议要么通过文本操作，丢弃发送方的内部表示，要么需要架构同质性以实现潜在层面传输。我们识别了跨架构通信中的实体锚定问题：在不同大语言模型家族之间传输连续表示的交叉注意力桥接，会遭遇稀有令牌压缩崩溃，即实体身份在连续瓶颈中丢失（仅桥接的F1分数约30%）。我们提出XBRIDGE，一种免解码通信协议，通过两种机制解决此问题。词汇锚定映射（LAM）将发送方的原始上下文令牌映射到接收方的词汇表，提供离散实体锚点。潜在富化桥接（LEB）使接收方能够查询

    arXiv:2608.11676v1 Announce Type: new  Abstract: Heterogeneous multi-agent LLM systems, where agents are powered by different model families, can outperform homogeneous configurations by reducing redundant reasoning patterns. Yet existing communication protocols either operate through text, discarding the sender's internal representations, or require architectural homogeneity for latent-level transfer. We identify the entity grounding problem in cross-architecture communication: cross-attention bridges that transfer continuous representations across different LLM families suffer from rare-token compression collapse, where entity identity is lost in the continuous bottleneck (bridge-only F1 ~30%). We propose XBRIDGE, a decode-free communication protocol that addresses this through two mechanisms. Lexical Anchor Mapping (LAM) maps the sender's original context tokens to the receiver's vocabulary, providing discrete entity anchors. A Latent Enrichment Bridge (LEB) lets the receiver query 
    
[^106]: 基于变分学习的RLVR参数空间探索

    Parameter Exploration for RLVR via Variational Learning

    [https://arxiv.org/abs/2608.09805](https://arxiv.org/abs/2608.09805)

    本文提出了一族名为扰动参数策略优化（3PO）的方法，通过变分学习从后验分布中采样多样化策略，实现参数空间探索，弥补了温度缩放等动作空间探索方法的局限，从而提升LLM强化学习的训练稳定性与下游性能。

    

    探索长期以来一直是强化学习研究的重点。最近，越来越多的证据表明，探索也是LLM强化学习方案中的一个重要组成部分，能够显著影响下游性能。许多现有方法在动作空间中控制探索，例如使用温度缩放。然而，这些方法无法对token进行重新排序，只能影响输出分布的方差，这限制了探索，可能导致训练发散或停滞。本文研究了参数空间探索，其中rollout通过从后验分布中采样不同的策略来生成，每个策略可能探索不同的rollout路径。采样较少或更多样化的策略则成为控制探索的补充手段。我们提出了一族称为扰动参数策略优化的方法，它们使用不同的采样策略和不同的ro

    arXiv:2608.09805v2 Announce Type: replace-cross  Abstract: Exploration has been a focus of reinforcement learning research for a long time. Recently, there has been growing evidence that it is also an important ingredient in LLM reinforcement learning recipes that can significantly impact downstream performance. Many existing methods control exploration in the action-space, for example, using temperature scaling. However, these methods cannot reorder tokens but only influence the variance in the output distribution. This limits exploration and can lead to divergence or stalled training. Here, we investigate parameter-space exploration, where rollouts are generated by sampling different policies from a posterior that may each explore different rollouts. Sampling less or more diverse policies is then a complementary control lever over exploration. We introduce a family of methods called Perturbed Parameter Policy Optimization (3PO) which use different sampling strategies and different ro
    
[^107]: 面向多模态大语言模型的统一幻觉模糊测试

    Unified Hallucination Fuzzing for Multimodal Large Language Models

    [https://arxiv.org/abs/2608.07525](https://arxiv.org/abs/2608.07525)

    该论文提出了结合统一幻觉分类细粒度数据集UniHall与自适应多模态模糊测试SAMF的系统性评估框架，利用进化变异策略动态探索多模态大语言模型的幻觉边界，克服了传统静态基准覆盖面窄和性能饱和的局限。

    

    幻觉仍然是多模态大语言模型（MLLMs）面临的一个持续性挑战，严重限制了其在高风险应用中的可靠性。现有评估主要基于静态基准测试，存在分类覆盖面窄和性能快速饱和的问题，无法反映模型在不断演变的真实场景中的鲁棒性。为弥合这一差距，我们提出了一个将综合基准与自演化压力测试相结合的系统性评估框架。首先，我们介绍了UniHall，这是一个基于统一分类体系构建的细粒度数据集，涵盖对象、指令和知识三个维度。其次，为解决基准饱和问题，我们提出了自适应多模态模糊测试（SAMF），这是一个采用进化变异策略来探索模型幻觉边界的自适应框架。至关重要的是，为确保对动态输入的可靠评估，SAMF 引入了结构化的（摘要在此处被截断）

    arXiv:2608.07525v2 Announce Type: replace-cross  Abstract: Hallucination remains a persistent challenge for Multimodal Large Language Models (MLLMs), severely limiting their reliability in high-stakes applications. Existing evaluations, predominantly based on static benchmarks, suffer from narrow taxonomical coverage and rapid performance saturation, failing to reflect model robustness in evolving real-world scenarios. To bridge this gap, we present a systematic evaluation framework integrating a comprehensive benchmark with self-evolving stress testing. First, we introduce UniHall, a fine-grained dataset grounded in a unified taxonomy spanning Object, Instruction, and Knowledge dimensions. Second, to address benchmark saturation, we propose Self-Adaptive Multimodal Fuzzing (SAMF), a self-adaptive framework that employs evolutionary mutation strategies to explore the boundaries of model hallucinations. Crucially, to ensure reliable assessment of dynamic inputs, SAMF incorporates a stru
    
[^108]: 从孤立算法到合规优先的智能体平台：医院AI系统的多层架构

    From Siloed Algorithms to Compliance-First Agentic Platforms: A Multi-Layered Architecture for Hospital AI Systems

    [https://arxiv.org/abs/2608.06112](https://arxiv.org/abs/2608.06112)

    该研究提出了一种合规优先的医院智能体AI多层架构，通过智能体编排层和以“策略即代码”为核心的合规与策略层，解决医疗AI部署孤立、治理缺失和难以规模化推广的问题。

    

    医院正在快速采用人工智能用于分诊、影像诊断、日程安排等领域，然而大多数部署仍然是孤立的点状解决方案，被封锁在部门壁垒之中，导致重复劳动、隐性风险和企业价值未能实现。尽管医疗AI市场呈爆炸式增长且投资不断加速，据估计70-80%的医疗AI试点项目无法规模化推广，其主要原因在于治理缺口、数据碎片化以及缺乏集成蓝图。本研究提出了一种医院专属、合规优先的智能体AI架构，包含多个可互操作的层级，对现有医院AI平台模型进行了扩展：(i) 智能体编排层，支持跨临床、运营和财务领域的多智能体工作流；(ii) 合规与策略层，以“策略即代码”的方式集中管理HIPAA、GDPR、欧盟AI法案、DISHA法案、印度DPDP法案以及ISO/IEC安全标准（注：原文摘要在此处截断，内容不完整）。

    arXiv:2608.06112v2 Announce Type: replace  Abstract: Hospitals are rapidly adopting artificial intelligence for triage, imaging, scheduling etc., yet most deployments remain isolated point solutions locked inside departmental silos, resulting in duplicated effort, hidden risks, and unrealized enterprise value. Despite explosive growth of AI in healthcare market and accelerating investment, an estimated 70-80% of healthcare AI pilots fail to scale, largely due to governance gaps, fragmented data, and missing integration blueprints. This research proposes a hospital-specific, compliance-first, Agentic AI architecture with multiple interoperable layers, extending existing hospital AI platform models with: (i) an Agent Orchestration Layer for multi-agent workflows across clinical, operational, and financial domains, (ii) a Compliance and Policy Layer that centralizes policy-as-code for HIPAA, GDPR, the EU AI Act, DISHA Act, India's DPDP Act, and ISO/IEC security and safety standards, and (
    
[^109]: 评估者也是实验的一部分：测量开放式LLM的趋同行为

    The Evaluator Is Part of the Experiment: Measuring Open-Ended LLM Conformity

    [https://arxiv.org/abs/2608.04463](https://arxiv.org/abs/2608.04463)

    本文提出了一种新的实验协议，用于测量开放式LLM在同伴上下文影响下的趋同行为，发现评估者本身对同伴内容的敏感度存在显著差异，且全错误同伴输入会一致导致最低质量的修订。

    

    arXiv:2608.04463v2 公告类型：替换 摘要：先前关于LLM趋同行为的研究主要是在可验证标签下测量离散答案的翻转。开放式修订需要不同的测量策略，因为答案质量是分级、潜在且被不完美评判的。我们引入了一种实验协议，该协议在一个汇集的主要同伴条件语料库和单独构建的分解语料库中实施，使我们能够区分普通重新回答、候选内容暴露、捆绑的同伴呈现残差以及方向性评判者对可见同伴上下文的敏感性。在四个开放权重生成器和三个基准测试中，全错误的同伴输入在每个生成器-数据集组合中都产生了质量最低的修订。对相同答案的盲评和知情评分也因评估者而异：一个评判者转向同伴认可的位置，两个转向相反方向，一个大致保持中立，而GPT-4o和GPT-5.4-mini审计同样非中立。最后，锚定审计显示...

    arXiv:2608.04463v2 Announce Type: replace  Abstract: Prior work on LLM conformity largely measures discrete answer flips under verifiable labels. Open-ended revisions require a different measurement strategy because answer quality is graded, latent, and judged imperfectly. We introduce an experimental protocol implemented across a pooled main peer-condition corpus and separately constructed decomposition corpora, allowing us to separate ordinary re-answering, candidate-content exposure, a bundled peer-presentation residual, and directional judge sensitivity to visible peer context. Across four open-weight generators and three benchmarks, all-wrong peer input produces the lowest-quality revisions in every generator-dataset cell. Blind and informed ratings of identical answers also differ by evaluator: one judge shifts toward the peer-endorsed position, two shift away, one is approximately neutral, and GPT-4o and GPT-5.4-mini audits are likewise non-neutral. Finally, an anchor audit show
    
[^110]: Twig：面向混合注意力大语言模型的并行且内存高效的树形推测解码

    Twig: Parallel and Memory-Efficient Tree Speculative Decoding for Hybrid-Attention LLMs

    [https://arxiv.org/abs/2608.01651](https://arxiv.org/abs/2608.01651)

    Twig通过对线性注意力递归进行树结构化因子分解并配合并行GPU内核，实现了混合注意力大语言模型的高效树形推测解码，将树验证加速3.4至7.7倍，同时将瞬态状态内存降低82至99倍。

    

    新兴的大语言模型越来越多地将全注意力与循环线性注意力交织结合，以降低长上下文推理服务的成本。树形推测解码可进一步加速这类混合模型，但循环线性注意力层携带有依赖于父节点的状态。因此，现有引擎只能顺序地验证提议树，并为每个节点快照完整状态，导致延迟和内存开销随树规模与批大小增长而不断攀升。我们提出了Twig，一种面向混合大语言模型树形推测解码的内核与运行时协同设计。Twig推导出线性注意力递归的树结构化因子分解，并通过一个资源高效的GPU内核加以实现，可并行验证所有节点，将线性注意力树验证速度提升3.4至7.7倍。它以无损方式将推测状态更新表示为token级因子，仅重构被接受的状态，从而将瞬态状态内存降低82至99倍，释放了GPU显存……（摘要原文在此处截断）

    arXiv:2608.01651v2 Announce Type: replace-cross  Abstract: Emerging LLMs increasingly interleave full attention with recurrent linear attention to reduce long-context serving costs. Tree speculative decoding could further accelerate these hybrid models, but recurrent linear-attention layers carry parent-dependent states. Existing engines therefore verify proposal trees sequentially and snapshot the full state per node, causing latency and memory to grow with tree and batch sizes. We present Twig, a kernel--runtime co-design for tree speculative decoding of hybrid LLMs. Twig derives a tree-structured factorization of the linear-attention recurrence and realizes it with a resource-efficient GPU kernel that verifies all nodes in parallel, accelerating linear-attention tree verification by 3.4--7.7$\times$. It losslessly represents speculative state updates as token-level factors and reconstructs only the accepted state, reducing transient state memory by 82--99$\times$ and freeing GPU mem
    
[^111]: 当激活神谕学会不去阅读：微调神谕中的概念特异性盲区

    When Activation Oracles Learn Not to Read: Concept-Specific Blind Spots in Fine-Tuned Oracles

    [https://arxiv.org/abs/2607.23379](https://arxiv.org/abs/2607.23379)

    该论文发现，微调后的激活神谕可能变成“概念特异性反读取器”，即选择性地无法读取其自身训练中持续存在的隐藏概念，而非成为专业读取器。

    

    激活神谕是经过训练、能够回答关于另一个模型内部激活的自然语言问题的语言模型。它们为从模型状态中读取隐藏信息提供了灵活的接口，尤其是当相关信息已在内部表示、但在可见行为中缺失或不完整时。然而，AOs 本身也是通过学习得到的系统：它们的答案由训练数据、训练目标以及习得的报告行为所塑造，而非对所表示信息的中性读出。我们在一个受控的“禁忌词猜词”设置中研究这一现象，其中目标模型被微调为在内部使用一个隐藏概念，同时避免直接披露。与“在此类目标模型上训练的 AO 会成为专业读取器”的预期相反，我们发现经过微调的 AOs 可能变成概念特异性的“反读取器”：它们选择性地无法恢复在其自身训练期间持续存在的概念。

    arXiv:2607.23379v2 Announce Type: replace-cross  Abstract: Activation Oracles (AOs) are language models trained to answer natural-language questions about another model's internal activations. They offer a flexible interface for reading hidden information from model states, especially when relevant information is internally represented but absent or incomplete in visible behavior. However, AOs are themselves learned systems: their answers are shaped by training data, objectives, and learned reporting behavior, rather than being neutral readouts of represented information. We study this in a controlled Taboo Word Guessing setting, where subject models are fine-tuned to internally use a hidden concept while avoiding direct disclosure. Contrary to the expectation that an AO trained on such a subject becomes a specialist reader, we find that fine-tuned AOs can become concept-specific anti-readers: they selectively fail to recover the concept persistently present during their own training. 
    
[^112]: 标记而非抹除：面向大语言模型两用知识的Token接种方法

    Mark, Don't Erase: Token Inoculation for Dual-Use Knowledge in LLMs

    [https://arxiv.org/abs/2607.18639](https://arxiv.org/abs/2607.18639)

    提出“Token接种”方法，在继续预训练时用特殊标记标注危险句子并训练模型条件化地预测危险答案，测试时因标记缺失而抑制危险输出，从而在保持93%良性性能的同时将危险准确率控制在约18%。

    

    在两用领域中，危险知识与围绕它的良性专业知识共享词汇、概念和因果结构。这种重叠使得选择性拒绝变得困难，而诸如去学习和拒绝训练等现有方法会将其抑制过度泛化到良性知识一侧。我们主张采用条件化而非抑制。在本工作中，我们提出了Token接种方法，在继续预训练期间用一个特殊标记（即接种标记）来标记危险句子。随后我们微调模型，使其拒绝危险问题，同时在带有接种标记的相同问题上预测危险答案。在测试时，接种标记并不存在，因此模型不会表达它学会预测的危险答案。在1.5B到14B参数范围以及三种架构家族中，该方法将危险准确率保持在约18%，同时保留了指令基线93%的MMLU性能，实现了最佳的安全-效用平衡。

    arXiv:2607.18639v2 Announce Type: replace-cross  Abstract: In a dual-use field, hazardous knowledge and the benign expertise around it share vocabulary, concepts, and causal structure. That overlap makes selective refusal hard, and existing methods such as unlearning and refusal training overgeneralize their suppression into the benign side. We argue for conditioning, not suppression. In this work, we introduce Token Inoculation, which marks hazardous sentences with a special token, the inoculation token, during continued pre-training. We then fine-tune the model to refuse hazardous questions and, on the same questions under the inoculation token, to predict the hazardous answers. At test time the inoculation token is absent, so the model does not express the hazardous answers it learned to predict. Across 1.5B to 14B parameters and three architecture families it holds hazardous accuracy near 18% while retaining 93% of the instruct baseline's MMLU performance, the best safety-utility b
    
[^113]: TACG：面向扩散语言模型解码的轨迹感知提交门控

    TACG: Trajectory-Aware Commit Gating for Diffusion Language Model Decoding

    [https://arxiv.org/abs/2607.03236](https://arxiv.org/abs/2607.03236)

    提出了无需训练的轨迹感知提交门控解码器 TACG，通过时间支持度评分、历史身份门控和每步位置数量上限三项机制，将词元提案与提交决策分离，从而决定扩散语言模型每步去噪中应提交哪些预测，兼顾了解码效率与输出质量。

    

    扩散语言模型在每个去噪步骤中会预测多个被掩码的位置，但其效率和输出质量取决于哪些预测能够进入不断演化的上下文。置信度对于这一决策具有参考价值但并不完整：在相同置信度下，具有不同身份和支持历史的提案会表现出不同的单步修订率。我们提出了轨迹感知提交门控（TACG），这是一种无需训练的解码器，它将当前的词元提案与是否提交该提案的决策分离开来。TACG 通过将当前提案的 logits 与早期 logits 的指数移动平均进行对比，来评估该提案的时间支持度；通过历史门控要求提案身份保持持久性；并对每步提升的额外位置数量设置上限。该门控机制决定提交哪些位置以及何时提交；每个写入的词元都保持为当时的基准提案。在规定条件下的完整四任务 LLaDA 评估中……（摘要原文在此处截断）

    arXiv:2607.03236v2 Announce Type: replace  Abstract: Diffusion language models predict many masked positions at each denoising step, but their efficiency and output quality depend on which predictions enter the evolving context. Confidence is informative but incomplete for this decision: at matched confidence, proposals with different identity and support histories exhibit different one-step revision rates. We propose Trajectory-Aware Commit Gating (TACG), a training-free decoder that separates the current token proposal from the decision to commit it. TACG scores temporal support for the current proposal by contrasting its logits with an exponential moving average of earlier logits, requires the proposal identity to persist through a History Gate, and promotes a capped number of additional positions per step. The gate decides which positions to commit and when; every written token remains the contemporaneous base proposal. On complete four-task LLaDA evaluations under the stated singl
    
[^114]: 状态-预测分离假设

    The State-Prediction Separation Hypothesis

    [https://arxiv.org/abs/2607.01218](https://arxiv.org/abs/2607.01218)

    该论文提出“状态-预测分离假设”，设计了一种用独立计算流将“预测下一个token”与“存储状态”两种功能解耦的 Transformer 变体，多规模预训练实验表明其具有更高的数据与计算效率，在下游任务上平均超越标准 Transformer 2-3 个百分点。

    

    Transformer 使用同一条前向计算流来同时完成两个任务：预测下一个 token，以及为未来的 token 预测存储有用的状态。我们提出了状态-预测分离假设：将这两种角色解耦能够带来更好的语言建模性能。我们设计了一种使用独立计算流来分离这两种功能的 Transformer 变体，并在多种规模上开展了预训练实验。实验表明，状态-预测分离始终能带来更好的数据效率和计算效率，降低了验证损失，并在下游任务上平均比标准 Transformer 高出 2-3 个百分点。我们还进行了广泛的实证分析，排除了潜在的混杂因素，并证明了我们的设计在梯度上所体现出的根本性差异。

    arXiv:2607.01218v2 Announce Type: replace-cross  Abstract: Transformers use the same forward computation stream to both predict the next token and store useful state for future token predictions. We formulate the state-prediction separation hypothesis: disentangling the two roles yields better language modeling performance. We design a Transformer variant that uses separate computation streams to separate the two functions, and conduct pretraining experiments across various scales. Our experiments show that state-prediction separation consistently offers better data- and compute-efficiency, improving validation loss and outperforming standard Transformers by 2-3 percentage points on average on downstream tasks. We also conduct extensive empirical analysis that rules out potential confounders and demonstrates the fundamental difference in the gradients our design entails.
    
[^115]: SeKV：面向长上下文大语言模型推理的分辨率自适应KV缓存与分层语义记忆

    SeKV: Resolution-Adaptive KV Cache with Hierarchical Semantic Memory for Long-Context LLM Inference

    [https://arxiv.org/abs/2606.31145](https://arxiv.org/abs/2606.31145)

    SeKV提出了一种分辨率自适应的KV缓存，通过训练的“放大”机制在解码时动态选择需要从压缩状态恢复到令牌分辨率的语义片段，实现了有损但不驱逐、内存高效的长上下文大模型推理。

    

    大语言模型越来越多地需要在长上下文上运行，此时KV缓存随序列长度线性增长，并在解码阶段占据主要内存。令牌驱逐方法会永久丢弃条目，而语义压缩则在预填充阶段就固定了每个单元的分辨率，导致后续才变得相关的细节无法恢复。我们提出SeKV，一种分辨率自适应的KV缓存，它围绕一个经过训练的“放大”机制构建，该机制在解码时决定将哪些被压缩的片段恢复到令牌分辨率。SeKV在高惊讶度位置对上下文进行分段，将锚点令牌和片段摘要保留在GPU上，并将每个片段键值的低秩因子卸载到CPU。轻量级的逐头路由器选择与查询相关的片段，并使用学习到的逐片段秩进行重建。该缓存是有损但不驱逐的：每个片段都保留在注意力计算中，无论是粗粒度形式还是重建后的形式。基础大语言模型保持冻结，路由器新增参数不到0.1%。

    arXiv:2606.31145v2 Announce Type: replace  Abstract: Large language models increasingly operate over long contexts, where the KV cache grows linearly with sequence length and dominates memory during decoding. Token eviction discards entries permanently, and semantic compression fixes each unit's resolution at prefill, so detail that becomes relevant later cannot be recovered. We propose SeKV, a resolution-adaptive KV cache built around a trained zoom-in mechanism that decides at decode time which compressed spans to restore to token resolution. SeKV segments the context at high-surprisal positions, keeps anchor tokens and span summaries on GPU, and offloads low-rank factors of each span's keys and values to CPU. Lightweight per-head routers select query-relevant spans, which are reconstructed with a learned per-span rank. The cache is lossy but non-evicting: every span stays in attention, coarse or reconstructed. The base LLM stays frozen, and the router adds under 0.1% parameters. Acr
    
[^116]: 参数化技能

    Parametric Skills

    [https://arxiv.org/abs/2606.30015](https://arxiv.org/abs/2606.30015)

    提出ParametricSkills框架，通过超网络在测试时将文本技能转化为LoRA适配器，实现无需上下文的技能全生命周期管理与利用，并借助真实交互轨迹对齐提升模型在实际工作中的可靠性。

    

    对于大语言模型（LLMs）而言，技能是编码成熟问题解决经验的文本化“配方”。尽管技能已被广泛部署，但其效用受限于模型理解和遵循技能指令的能力。为解决这一局限，我们提出了ParametricSkills——一个在测试时将文本技能参数化的框架，实现了无需上下文的全技能生命周期管理。具体而言，我们首先构建了一个大规模、高质量的技能集合，随后合成了单轮和多轮的技能利用轨迹。利用这些数据，我们训练了一个超网络，通过将文本技能转换为LoRA适配器，对技能的内容及其利用方法进行参数化。为促进该框架在真实世界中的应用，我们进一步使用从OpenCode收集的复杂、长程轨迹对超网络进行对齐，使模型能够在真实的交互式工作中可靠运行。

    arXiv:2606.30015v2 Announce Type: replace  Abstract: For LLMs, skills are textual recipes encoding mature problem-solving experience. Despite widespread deployment, their utility is limited by the model's ability to comprehend and follow skill instructions. To address this limitation, we propose ParametricSkills, a framework that parameterizes textual skills at test time, enabling context-free management of the full skill lifecycle. Specifically, we first construct a large-scale and high-quality skill collection, and then synthesize single-turn and multi-turn skill-exploitation trajectories. Using this data, we then train a hypernetwork that parameterizes both the content and the exploitation methodologies of skills by converting textual skills into LoRA adapters. To facilitate the real-world application of our framework, we further align the hypernetwork on complex, long-horizon trajectories collected with OpenCode, enabling the models to operate reliably in realistic interactive work
    
[^117]: 视觉为默认、先验可覆盖：视觉语言模型中感知-知识冲突的因果机制

    Vision-Default, Prior-Override: Causal Mechanisms of Perception-Knowledge Conflict in Vision-Language Models

    [https://arxiv.org/abs/2606.28273](https://arxiv.org/abs/2606.28273)

    该研究通过多粒度激活修补与组件消融首次从因果机制层面揭示了视觉语言模型的感知-知识冲突：视觉接地是默认行为，而基于先验知识的回答仅依赖于集中于网络后半部分的一小部分（2.5-4.8%）因果必需的注意力头。

    

    当视觉证据与记忆中的世界知识发生冲突时，视觉语言模型必须对二者进行调和。模型如何解决这种冲突决定了多模态系统的可靠性，然而以往的工作仅从行为层面进行刻画，缺乏组件层面的因果解释。我们将三个粒度（残差流、注意力头和MLP子层）上的激活修补与模型组件消融研究及机制分析相结合。在三个视觉语言模型家族中，我们发现视觉接地是默认涌现的行为，而先验接地则依赖于一小部分因果必需的注意力头（占2.5-4.8%），这些注意力头集中在网络的后半部分。这些注意力头使模型能够在视觉输入相冲突的情况下，依然依据存储的世界知识给出答案（例如，对草莓回答“红色”）。消融这些注意力头后，在先验知识提示下，68-96%的案例中模型预测从基于知识的答案翻转为基于视觉的答案，但仅改变0.8-7（摘要原文在此处截断）。

    arXiv:2606.28273v2 Announce Type: replace  Abstract: Vision-language models must reconcile visual evidence with memorized world knowledge when the two conflict. How they resolve this conflict shapes the reliability of multimodal systems, yet prior work characterizes it behaviorally without a component-level causal account. We combine activation patching across three granularities (residual stream, attention heads, and MLP sublayers) with model-component ablation studies and mechanistic analysis. Across three VLM families, we find that visual grounding emerges by default, whereas prior grounding depends on a small set of causally necessary attention heads (2.5-4.8%) concentrated in the second half of the network. These heads enable answers from stored world knowledge (e.g., "red" for a strawberry) despite conflicting visual input. Ablating them flips predictions from knowledge-grounded to visually grounded answers in 68-96% of cases under prior-knowledge prompts, but changes only 0.8-7.
    
[^118]: 软令牌对齐用于跨语言推理

    Soft Token Alignment for Cross-Lingual Reasoning

    [https://arxiv.org/abs/2606.26466](https://arxiv.org/abs/2606.26466)

    本文提出SOLAR方法，通过软令牌对齐和英语枢轴，有效解决了多语言大模型在跨语言推理中因词汇差异导致的语义等价推理路径不一致问题。

    

    arXiv:2606.26466v1 公告类型：新  摘要：多语言大语言模型在处理不同语言中语义等价的提示时，常常会产生不一致的推理和答案。先前的研究表明，中间表示可以相对语言无关，但随着模型承诺于离散的输出令牌，生成过程变得越来越具有语言特异性。这带来了问题，因为语言特定的词汇选择会导致语义等价的推理路径在不同语言中产生分歧。这些分歧促使我们寻找一种与任何单一词汇项或书写系统关联较弱的跨语言对齐信号。我们提出了SOLAR，一种用于监督微调的辅助目标，它通过使用英语作为枢轴，跨语言对齐软令牌表示。软令牌是词汇嵌入上的概率加权混合，提供了能够跨语言聚合来自语义相关令牌信息的连续表示。

    arXiv:2606.26466v1 Announce Type: new  Abstract: Multilingual large language models often produce inconsistent reasoning and answers for semantically equivalent prompts in different languages. Prior work suggests that intermediate representations can be relatively language-agnostic, but generation becomes increasingly language-specific as models commit to discrete output tokens. This is problematic because language-specific lexical choices can cause semantically equivalent reasoning paths to diverge across languages. These divergences motivate searching for a cross-lingual alignment signal that is less tied to any single vocabulary item or script. We propose SOLAR, an auxiliary objective for supervised fine-tuning that aligns soft-token representations across languages, using English as a pivot. Soft tokens are probability-weighted mixtures over the vocabulary embeddings, yielding continuous representations that can aggregate information from semantically related tokens across language
    
[^119]: Influcoder：将解码器的梯度影响力排序蒸馏到编码器中以实现数据归因

    Influcoder: Distilling Decoders' Gradient Influence Rankings into an Encoder for Data Attribution

    [https://arxiv.org/abs/2606.13668](https://arxiv.org/abs/2606.13668)

    Influcoder通过将解码器的梯度影响力排序蒸馏到编码器中，实现了快速且低成本的规模化数据归因。

    

    随着大语言模型（LLM）能力的不断增强，人们越来越倾向于通过过滤训练数据中的样本来构建高质量数据集。总体而言，数据归因方法旨在估计训练数据集中的单个样本如何预先决定模型生成特定输出的倾向。例如，人们可能想知道在训练大语言模型后，数据中的哪些样本可能成为毒性行为的来源。许多方法通过影响函数的范式来量化这种条件作用。虽然这类方法在功能上是有效的，但它们缺乏在大型数据集上实际应用所需的处理速度和存储紧凑性。我们提出了一种名为Influcoder的方法，作为一种快速且经济高效的大规模基于影响的数据归因方法。

    arXiv:2606.13668v2 Announce Type: replace  Abstract: With the growth of LLMs' (Large Language Models) capabilities, there has been an increasing push to curate high quality datasets by filtering samples in the training data. In general, Data Attribution (DA) methods aim to estimate how individual samples in a training dataset can precondition a model to generate certain outputs. As an example, one might be interested in which samples in the data could be the source of toxic behavior after training the LLM. Many methods quantify this conditioning through the paradigm of influence functions. While methods of this family are effective in its function, they lack the necessary processing speed and storage compactness to be practically implemented on large datasets. We propose a method, Influcoder, as a quick and cost-effective approach to influence-based Data Attribution at scale.
    
[^120]: 与“敌人”编程：人类开发者能否检测出AI智能体的破坏行为？

    Coding with "Enemy": Can Human Developers Detect AI Agent Sabotage?

    [https://arxiv.org/abs/2606.05647](https://arxiv.org/abs/2606.05647)

    本研究首次大规模研究了人类监督在AI编程破坏行为中的作用，发现在无监控条件下高达94%的开发者（83/88）未能检测到AI智能体植入的破坏行为。

    

    AI编程智能体正日益融入真实的软件开发环境，在与人类开发者协作的同时，获得了对代码库和工具更广泛的访问权限。这带来了新的攻击面：智能体可以利用人类的信任来破坏开发过程，例如通过插入恶意代码来完成隐藏的副任务。以往的大多数工作仅在纯AI环境中研究AI破坏行为，对人类监督在检测和缓解此类恶意行为中的作用关注有限。为填补这一空白，我们开展了首个关于AI编程破坏行为中人类监督的大规模研究。100多名参与者与四个前沿模型之一（Claude-Opus-4.6、GPT-5.4、Gemini-3.1-Pro和MiniMax-M2.7）协作，完成一项旨在模拟真实工作流程、历时约五小时的长周期编程任务。研究发现，在无监控条件下，88名开发者中有83名（94%）未能检测到破坏行为，并且我们对参与者的分……

    arXiv:2606.05647v2 Announce Type: replace  Abstract: AI coding agents are increasingly embedded in real-world software development, collaborating with human developers while gaining broader access to codebases and tools. This creates a new attack surface: an agent can exploit human trust to sabotage development, for instance by inserting malicious code to accomplish a hidden side task. Most prior work studies AI sabotage in AI-only settings, paying limited attention to the role of human oversight in detecting and mitigating such malicious behavior. To address this gap, we conduct the first large-scale study of human oversight in AI coding sabotage. Over 100 participants collaborate with one of four frontier models (Claude-Opus-4.6, GPT-5.4, Gemini-3.1-Pro, and MiniMax-M2.7) on a long-horizon coding task lasting around five hours, designed to mimic real-world workflows. We find that 83/88 (94%) of developers in the no-monitor conditions fail to detect sabotage, and our analysis of parti
    
[^121]: 未采样的真相：量化语言模型心理测量学中的提示伪影

    The Unsampled Truth: Quantifying Prompt Artifacts in LM Psychometrics

    [https://arxiv.org/abs/2606.03357](https://arxiv.org/abs/2606.03357)

    该论文提出一种诊断框架，通过将人设与四个提示组件的语义等价变体交叉组合并测量响应分布间的Wasserstein距离，成功将心理测量中的目标效应与提示伪影分离，并发现任务指令和选项符号对响应分布的干扰比人设或题项措辞的改变更大。

    

    当通过提示让语言模型进行心理测量评估时，研究者假设模型的响应反映了所注入的人设以及调查题项本身的含义。我们通过一种诊断性设计来检验这一前提：将五个语义上不同的人设与四个提示组件（人设措辞、任务指令、题项措辞、选项符号）各自的五个语义等价变体进行交叉组合。通过测量所得响应分布之间的1-Wasserstein距离，并在五个组件之间对变异进行划分，可以将目标效应与提示伪影分离开来。我们将该框架应用于13个开源权重的小型语言模型（参数量0.6B至14B），在大五人格量表和短版黑暗三联征量表上进行测试。我们发现，在大多数模型中，任务指令和选项符号对响应分布造成的偏移，大于对人设描述或题项本身进行同义改写所造成的影响。对于相当一部分（摘要在此处被截断）

    arXiv:2606.03357v2 Announce Type: replace-cross  Abstract: When prompting language models for psychometric assessment, researchers assume that the responses reflect the injected persona and the meaning of the survey item. We test this premise using a diagnostic design that crosses five semantically distinct baseline personas with five semantically equivalent variants of each of four prompt components (persona wording, task instruction, item wording, option symbol). Measuring the 1-Wasserstein distance between the resulting response distributions and partitioning the variation among the five components allows for the separation of target effects from prompt artifacts. We apply the framework to 13 open-weight small language models (0.6B to 14B) on the Big Five Inventory and the Short Dark Triad. We find that in most models, the task instruction and option symbol displace response distributions further than paraphrasing the persona description or the item itself. For a substantial share o
    
[^122]: 你的多模态语音模型说我长了一张“只适合上广播的脸”

    Your Multimodal Speech Model Says I Have a Face for Radio

    [https://arxiv.org/abs/2605.30472](https://arxiv.org/abs/2605.30472)

    首次对多模态语音识别进行了偏见评估，发现当不同面孔与相同音频配对时，mWhisper-Flamingo和Gemini模型的词错误率最多上升4.05个百分点，揭示了基于性别、种族及其交叉类别的显著服务质量差异。

    

    随着大型神经网络模型在语言任务上的表现不断提升，研究人员正越来越多地构建能够处理多种数据模态的多模态和全模态模型。一个例子是将语音识别模型扩展到视听数据，其目标是降噪和多模态字幕生成。尽管在单模态场景下性能与偏见已被广泛研究，但新模态如何影响这些方面尚不清楚，即使这些新模态会在人类中引发偏见。因此，我们提出了首个针对多模态语音识别的偏见评估：我们利用唇形同步技术创建将不同面孔与相同音频配对的视频，并测量语音转录准确率的变化。我们发现mWhisper-Flamingo和Gemini模型在服务质量上存在巨大差异，在自我声明性别、种族及其交叉类别上，词错误率最多上升4.05个百分点。我们的研究结果为开发（工作）指明了优先事项。

    arXiv:2605.30472v2 Announce Type: replace  Abstract: As large neural models have become better at language tasks, researchers are increasingly building multi- and omnimodal models that handle more than one modality of data. One example is the expansion of speech recognition models to audio-visual data, the goal being noise mitigation and multimodal subtitling. While performance and bias have been studied extensively in the single-modality regime, it is unknown how new modalities affect this, even though they produce biases in humans. We therefore propose the first bias evaluation of multimodal speech recognition, where we create videos pairing different faces with the same audio using lip-sync, and measure changes in speech transcription accuracy. We find large quality-of-service differences across mWhisper-Flamingo and Gemini models, with drops of up to 4.05 word error rate points, across self-declared gender, ethnicity, and their intersection. Our findings point to a priority for dev
    
[^123]: 按其思考方式施教：用稀疏自编码器引导大语言模型后训练数据工程

    Teach the Way It Thinks: Guiding LLM Post-training Data Engineering with Sparse Autoencoders

    [https://arxiv.org/abs/2605.27354](https://arxiv.org/abs/2605.27354)

    该论文提出SAERL方法，利用稀疏自编码器（SAE）将模型内部分解为稀疏可解释的特征，从模型内部视角（激活与梯度投影）来组织强化学习训练数据，从而决定训练哪些样本、何时呈现以及如何分批，突破了传统仅依赖通过率等外部反馈的数据工程方式。

    

    大语言模型（LLM）强化学习（RL）的数据工程决定了训练哪些样本、何时呈现这些样本，以及哪些样本应一起分批。现有方法大多依赖外部反馈，例如通过率或难度标签，这只能观察到每个样本上的结果。我们认为，模型内部提供了更丰富、更贴近模型自身的反馈，能够指导上述全部三项决策。我们提出SAERL，它在稀疏自编码器（SAE）的特征空间中组织强化学习训练数据。SAE是一种机制可解释性工具，可将模型内部分解为稀疏且可解释的特征。借助SAE特征，我们可以将模型视为一组功能区域，每个区域由相关样本共同激活的特征所构成。对于每个样本，其SAE激活表明它参与了哪些区域，而其投影到SAE特征上的梯度则表明对它进行训练将如何改变这些区域。在此指导下……（原文摘要至此被截断）

    arXiv:2605.27354v2 Announce Type: replace-cross  Abstract: Data engineering for large language model (LLM) reinforcement learning (RL) decides which samples to train on, when to present them, and which to batch together. Existing methods mostly rely on external feedback, such as pass rates or difficulty labels, which observes only the outcome on each sample. We argue that model internals offer richer and more model-intrinsic feedback that can guide all three decisions. We propose SAERL which organizes RL training data in the feature space of Sparse Autoencoders (SAEs), a mechanistic interpretability tool that decomposes model internals into sparse, interpretable features. SAE features let us view the model as a set of functional regions, each formed by features that related samples activate together. For each sample, its SAE activations indicate which regions it engages, and its gradients projected onto SAE features indicate how training on it would change these regions. Guided by this
    
[^124]: JudgmentBench：比较量规评估与偏好评估的质量评估方法

    JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment

    [https://arxiv.org/abs/2605.25240](https://arxiv.org/abs/2605.25240)

    论文发布JudgmentBench基准，汇集资深律师对真实法律任务给出的量规评分与成对偏好判断数据，首次实证比较发现比较判断在恢复输出质量排序方面显著优于基于量规的评分。

    

    两种方法论主导着当前的基准测试实践：基于量规的评分根据预定义的标准来评估项目，而比较判断则在输出之间引出成对偏好。尽管这两种方法论都被广泛使用，但它们之间的选择却很少得到合理的论证。我们发布了JudgmentBench，这是一个包含30个真实世界法律任务的基准，配有1,539个量规评分和1,530个成对偏好判断，这些数据由具有丰富经验的执业律师收集——其中包括来自美国主要律师事务所的律师。这些标注构成了高专业领域内首个公开可用的数据集，其中两种监督信号均由同一批专家在相同任务上引出。使用三个构建的不同质量级别的LLM生成输出，我们提供了初步的实证比较：在恢复预期质量排序方面，比较判断显著优于量规……

    arXiv:2605.25240v4 Announce Type: replace-cross  Abstract: Two methodologies dominate current practices of benchmarking: rubric-based scoring evaluates items against predefined criteria, whereas comparative judgment elicits pairwise preferences between outputs. Although both methodologies are widely used, the choice between them is rarely justified. We release JudgmentBench, a benchmark of 30 real-world legal tasks, paired with 1,539 rubric scores and 1,530 pairwise preference judgments collected from practicing attorneys--including at major U.S. law firms--with substantial experience. The annotations constitute the first publicly available dataset in a high-expertise domain in which both supervision signals are elicited from the same pool of experts on the same tasks. Using LLM-generated outputs at three constructed quality levels, we provide an initial empirical comparison: comparative judgments recover the intended quality ordering substantially better than rubrics under both a per-
    
[^125]: 向量策略优化：面向多样性的训练改进测试时搜索

    Vector Policy Optimization: Training for Diversity Improves Test-Time Search

    [https://arxiv.org/abs/2605.22817](https://arxiv.org/abs/2605.22817)

    提出向量策略优化（VPO），通过利用向量值奖励、多答案提示与集合级奖励来训练大语言模型生成多样化的解决方案集合，从而提升推理时搜索的效果。

    

    语言模型现在必须能够开箱即用地泛化到新颖环境中，并能在推理扩展搜索过程（如AlphaEvolve）中工作，这类搜索过程使用多种任务特定的奖励函数来选择 rollout。遗憾的是，标准的 LLM 后训练范式优化的是预先指定的标量奖励，这往往导致当前的 LLM 产生低熵的响应分布，从而难以展现推理时搜索所需的多样性。我们提出了向量策略优化（VPO），这是一种强化学习算法，它显式地训练策略以预期多样化的下游奖励函数并产生多样化的解决方案。VPO 利用了实践中奖励往往是向量值的事实，例如代码生成中每个测试用例的正确性，或者多个不同的用户角色或奖励模型。VPO 将多答案提示与集合级奖励相结合，训练 LLM 生成能够……

    arXiv:2605.22817v2 Announce Type: replace-cross  Abstract: Language models must now generalize out of the box to novel environments and work inside inference-scaling search procedures, such as AlphaEvolve, that select rollouts with a variety of task-specific reward functions. Unfortunately, the standard paradigm of LLM post-training optimizes a pre-specified scalar reward, often leading current LLMs to produce low-entropy response distributions and thus to struggle at displaying the diversity that inference-time search will require. We propose Vector Policy Optimization (VPO), an RL algorithm that explicitly trains policies to anticipate diverse downstream reward functions and to produce diverse solutions. VPO exploits that rewards are often vector-valued in practice, like per-test-case correctness in code generation or, say, multiple different user personas or reward models. VPO combines multi-answer prompting with a set-level reward to train the LLM to produce sets of solutions that 
    
[^126]: 软标签学习与校准中人类不确定性与模型不确定性的评估

    An Assessment of Human vs. Model Uncertainty in Soft-Label Learning and Calibration

    [https://arxiv.org/abs/2605.18648](https://arxiv.org/abs/2605.18648)

    本研究通过解耦软标签监督与标签模态偏移，发现人类软标签的主要价值在于作为正则化器提升模型对困难样本的校准能力和训练收敛稳定性，并使模型的不确定性表现与人类一致。

    

    以人为本的对齐AI的核心在于理解人类提供的标签相较于合成标签的优势。虽然人类软标签通过捕捉不确定性能够改善模型校准，但先前的研究将这些优势与对错误标注数据的隐式纠正（模态偏移）混为一谈，从而掩盖了软标签的真正效果。我们对MNIST及其合成变体上的软标签学习进行了受控审计，通过重新标注数据子集来提取人类不确定性。通过将软标签监督与底层标签模态偏移解耦，我们证明人类软标签确实能带来准确率提升，但其更大的价值在于作为一种正则化器，能够改善模型对困难样本的校准，并促进训练过程中的稳定收敛。数据集制图分析表明，使用人类软标签训练的模型能够反映人类的不确定性，而使用合成标签训练的模型则无法与人类对齐。总体而言，这项工作……（原文截断）

    arXiv:2605.18648v3 Announce Type: replace-cross  Abstract: Central to human-aligned AI is understanding the benefits of human-elicited labels over synthetic alternatives. While human soft-labels improve calibration by capturing uncertainty, prior studies conflate these benefits with the implicit correction of mislabeled data (mode shifts), obscuring true effects of soft-labels. We present a controlled audit of soft-label learning across MNIST and a synthetic variant, re-annotating subsets to extract human uncertainty. By decoupling soft-label supervision from underlying label mode shifts, we show that while human soft-labels do provide accuracy gains, their larger value lies in acting as a regularizer that improves model calibration on difficult samples and promotes stable convergence across training runs. Dataset cartography reveals models trained on human soft-labels mirror human uncertainty, whereas those trained on synthetic labels fail to align with humans. Broadly, this work prov
    
[^127]: 涌现性失调中的人格模型坍缩

    Persona-Model Collapse in Emergent Misalignment

    [https://arxiv.org/abs/2605.12850](https://arxiv.org/abs/2605.12850)

    该论文提出“人格模型坍缩”假说，指出有害数据微调引发的涌现性失调源于模型内部人格表征的退化，并在四个模型家族上通过问卷实验证实有害微调使人格表达更极化、更二元化且内部一致性降低。

    

    在包含有害内容的狭窄数据上对大型语言模型进行微调，会使模型在与该数据无关的提示上也表现出广泛的对齐失调，这种现象被称为“涌现性失调”。我们提出，涌现性失调涉及“人格模型坍缩”：即编码和区分人格的内部表征发生退化，从而破坏了人格表达的细腻性与一致性。我们首先从行为层面检验这一假设：比较在有害数据上微调的模型与良性对照模型在扮演多样化人格时对问卷的回答。在四个模型家族中，有害微调产生了更极化、“非全即无”的回答，表明人格表达变得更加二元化、细腻程度更低。直方图显示回答在量表两端更为集中，并通过库尔贝克—莱布勒（KL）散度对这种分布偏移进行了量化。我们还发现人格内部一致性有所降低：对同一问题的重复回答……（摘要原文在此处截断）

    arXiv:2605.12850v3 Announce Type: replace-cross  Abstract: Fine-tuning large language models on narrow data with harmful content produces broadly misaligned behavior on unrelated prompts, a phenomenon known as emergent misalignment. We propose that emergent misalignment involves persona-model collapse: a deterioration of the internal representations that encode and distinguish personas, undermining the nuance and consistency of their expression. We first test this hypothesis behaviorally, by comparing how models fine-tuned on harmful data and benign controls answer questionnaires while role-playing diverse personas. Across four model families, harmful fine-tuning produces more polarized, all-or-nothing answers, suggesting more binary, less nuanced persona expression. Histograms show increased concentration at the scale endpoints, and Kullback--Leibler (KL) divergence quantifies the distributional shift. We also find reduced within-persona consistency: repeated answers to the same quest
    
[^128]: 面向低资源语言的手语识别与翻译：挑战与前进之路

    Sign Language Recognition and Translation for Low-Resource Languages: Challenges and Pathways Forward

    [https://arxiv.org/abs/2605.12096](https://arxiv.org/abs/2605.12096)

    本综述以阿塞拜疆手语为案例，系统梳理了低资源手语识别与翻译面临的挑战，提炼出八条可操作经验，并提出以数据为中心的AI、手语使用者自适应系统和任务导向评估指标这三大范式转变。

    

    手语是世界各地聋人群体使用的自然视觉-手势语言。由于文献记录有限、数据集稀疏以及计算工具不足，超过300种不同的手语仍处于严重的低资源状态。本系统性综述综合了关于低资源语言手语识别与翻译的相关文献，并以阿塞拜疆手语（AzSL）作为案例研究。通过对全球倡议的分析，提炼出八条可操作的经验教训，包括社区共同设计、方言多样性采集以及隐私保护的基于姿态的表示方法。突厥语系手语（哈萨克语、土耳其语、阿塞拜疆语）受到特别关注，因为语言上的相近性使得迁移学习能够有效开展。我们提出了三个范式转变：从以架构为中心的AI转向以数据为中心的AI，从手语使用者无关的系统转向手语使用者自适应的系统，以及从基于参考的评估指标转向面向特定任务的评估指标。最后给出了技术路线图。

    arXiv:2605.12096v2 Announce Type: replace  Abstract: Sign languages are natural, visual-gestural languages used by Deaf communities worldwide. Over 300 distinct sign languages remain severely low-resource due to limited documentation, sparse datasets, and insufficient computational tools. This systematic review synthesizes literature on sign language recognition and translation for under-resourced languages, using Azerbaijan Sign Language (AzSL) as a case study. Analysis of global initiatives extracts eight actionable lessons, including community co-design, dialectal diversity capture, and privacy-preserving pose-based representations. Turkic sign languages (Kazakh, Turkish, Azerbaijani) receive special attention, as linguistic proximity enables effective transfer learning. We propose three paradigm shifts: from architecture-centric to data-centric AI, from signer-independent to signer-adaptive systems, and from reference-based to task-specific evaluation metrics. A technical roadmap f
    
[^129]: ELF：嵌入式语言流

    ELF: Embedded Language Flows

    [https://arxiv.org/abs/2605.10938](https://arxiv.org/abs/2605.10938)

    提出在连续嵌入空间中基于流匹配的扩散语言模型 ELF，其直到最终时间步才映射到离散词元，从而让图像扩散领域的成熟技术（如 CFG）可直接应用于语言建模。

    

    扩散模型和基于流的模型已成为生成连续数据（如图像和视频领域）的事实标准方法。它们的成功激发了人们将其应用于语言建模的日益浓厚的兴趣。与图像领域的对应模型不同，当今主流的扩散语言模型（DLM）主要在离散词元上运行。在本文中，我们证明了连续扩散语言模型只需对离散域进行极少适配即可变得有效。我们提出了嵌入式语言流，这是一类在连续嵌入空间中基于连续时间流匹配的扩散模型。与现有的扩散语言模型不同，ELF 在绝大部分生成过程中保持在连续嵌入空间中，直到最后一个时间步才通过一个共享权重网络映射到离散词元。这种设计使得图像领域扩散模型中成熟的技术（例如无分类器引导 CFG）能够直接移植到语言建模中。

    arXiv:2605.10938v3 Announce Type: replace-cross  Abstract: Diffusion and flow-based models have become the de facto approaches for generating continuous data, e.g., in domains such as images and videos. Their success has attracted growing interest in applying them to language modeling. Unlike their image-domain counterparts, today's leading diffusion language models (DLMs) primarily operate over discrete tokens. In this paper, we show that continuous DLMs can be made effective with minimal adaptation to the discrete domain. We propose Embedded Language Flows (ELF), a class of diffusion models in continuous embedding space based on continuous-time Flow Matching. Unlike existing DLMs, ELF predominantly stays within the continuous embedding space until the final time step, where it maps to discrete tokens using a shared-weight network. This formulation makes it straightforward to adapt established techniques from image-domain diffusion models, e.g., classifier-free guidance (CFG). Experim
    
[^130]: FILO：低秩自适应的统一Fisher几何框架

    FILO: A Unified Fisher-Geometric Framework for Low-Rank Adaptation

    [https://arxiv.org/abs/2605.08734](https://arxiv.org/abs/2605.08734)

    本文提出FILO，一个统一的Fisher几何框架，无需对奇异算子直接求逆即可实现LoRA的因子空间更新，并据此将现有LoRA优化器系统地划分为四大类。

    

    低秩自适应（LoRA）将权重矩阵 $W$ 重参数化为两个低秩因子的乘积，但将因子映射到权重矩阵的生成器的雅可比矩阵 $J_G$ 是秩亏的，因此由任何 $W$ 空间预条件子 $F_t$ 诱导的因子空间预条件子 $J_G^* F_t J_G$ 是奇异的，进而导致标准链式法则无法被唯一地求逆，无法将预条件后的 $W$ 空间方向映射回因子空间的更新。为此，我们提出了 FILO，一个用于低秩自适应的统一 Fisher 几何框架，它无需直接对该奇异算子求逆即可实现因子更新，并根据因子空间求逆策略和 $W$ 空间预条件子 $F_t$ 对现有 LoRA 优化器进行分类。现有方法沿着这两个维度划分为四大类：因子空间自适应更新、$J_G^* J_G$ 的块对角替代方法、Frobenius 残差伪逆方法以及黎曼流形方法。

    arXiv:2605.08734v2 Announce Type: replace-cross  Abstract: Low-Rank Adaptation (LoRA) reparameterizes a weight matrix $W$ as a product of two low-rank factors, but the Jacobian $J_G$ of the generator mapping the factors to the weight matrix is rank-deficient, so the factor-space preconditioner $J_G^* F_t J_G$ induced by any $W$-space preconditioner $F_t$ is singular, and consequently the standard chain rule cannot be uniquely inverted to map a preconditioned $W$-space direction back to a factor-space update. Therefore, we introduce FILO, a unified Fisher-geometric framework for low-rank adaptation that enables factor updates without directly inverting this singular operator and casts existing LoRA optimizers by their factor-space inversion strategy and $W$-space preconditioner $F_t$. Existing methods occupy four families along these axes: factor-space adaptive updates, block-diagonal surrogates for $J_G^* J_G$, Frobenius-residual pseudoinverse methods, and Riemannian manifold methods. 
    
[^131]: UniSD：面向大语言模型的统一自蒸馏框架

    UniSD: Towards a Unified Self-Distillation Framework for Large Language Models

    [https://arxiv.org/abs/2605.06597](https://arxiv.org/abs/2605.06597)

    提出了统一自蒸馏框架UniSD，通过整合多教师一致性、EMA教师稳定化、词元级对比学习、特征匹配和散度裁剪等互补机制，系统性地解决了大语言模型自蒸馏中监督可靠性、表示对齐与训练稳定性的核心难题。

    

    自蒸馏（SD）为在不依赖更强外部教师模型的情况下适配大语言模型（LLM）提供了一条有前景的路径。然而，自回归LLM中的自蒸馏仍然充满挑战，因为自生成的轨迹是自由形式的，正确性依赖于具体任务，且看似合理的推理过程仍可能提供不稳定或不可靠的监督信号。现有方法主要考察孤立的设计选择，导致其有效性、作用及相互关系尚不明确。本文提出了UniSD，一个用于系统研究自蒸馏的统一框架。UniSD整合了解决监督可靠性、表示对齐和训练稳定性问题的互补机制，包括多教师一致性、EMA教师稳定化、词元级对比学习、特征匹配和散度裁剪。在来自三个模型系列的六个基准测试和六个模型上，UniSD揭示了自蒸馏何时有效（摘要在此处被截断）。

    arXiv:2605.06597v3 Announce Type: replace-cross  Abstract: Self-distillation (SD) offers a promising path for adapting large language models (LLMs) without relying on stronger external teachers. However, SD in autoregressive LLMs remains challenging because self-generated trajectories are free-form, correctness is task-dependent, and plausible rationales can still provide unstable or unreliable supervision. Existing methods mainly examine isolated design choices, leaving their effectiveness, roles, and interactions unclear. In this paper, we propose UniSD, a unified framework to systematically study self-distillation. UniSD integrates complementary mechanisms that address supervision reliability, representation alignment, and training stability, including multi-teacher agreement, EMA teacher stabilization, token-level contrastive learning, feature matching, and divergence clipping. Across six benchmarks and six models from three model families, UniSD reveals when self-distillation impr
    
[^132]: 语言模型中的不变特征：几何刻画与模型归因

    Invariant Features in Language Models: Geometric Characterization and Model Attribution

    [https://arxiv.org/abs/2605.06458](https://arxiv.org/abs/2605.06458)

    该论文提出一个局部几何框架来刻画语言模型中的不变特征，通过对比子空间发现方法分离语义改变型与语义保持型的表示变化，并将所得的不变表示成功应用于零样本模型归因任务。

    

    语言模型对改写表现出很强的鲁棒性，这表明语义信息可能通过稳定的内部表示进行编码，但这种不变性的结构和来源仍不清楚。我们提出了一个局部几何框架，其中语义等价的输入所产生的表示占据潜空间中的结构化区域，改写引起的变化沿着干扰方向进行，而语义同一性则在不变子空间中得以保持。基于这一观点，我们做出了三项贡献：(1) 对不变潜在特征的几何刻画，(2) 一种对比子空间发现方法，能够将改变语义的变化与保持语义的变化分离开来，(3) 将不变表示应用于零样本模型归因。跨模型和跨层的实证结果支持了这些贡献。不变结构在特定的深度区域中涌现，语义位移……（原文摘要不完整）

    arXiv:2605.06458v2 Announce Type: replace-cross  Abstract: Language models exhibit strong robustness to paraphrasing, suggesting that semantic information may be encoded through stable internal representations, yet the structure and origin of such invariance remain unclear. We propose a local geometric framework in which semantically equivalent inputs produce representations that occupy structured regions in latent space, with paraphrastic variation along nuisance directions and semantic identity preserved in invariant subspaces. Building on this view, we make three contributions: (1) a geometric characterization of invariant latent features, (2) a contrastive subspace discovery method that separates semantic-changing from semantic-preserving variation, and (3) an application of invariant representations to zero-shot model attribution. Across models and layers, empirical results support these contributions. Invariant structure emerges in specific depth regions, semantic displacement li
    
[^133]: 顿悟还是故障？低精度如何驱动“弹弓机制”式损失尖峰

    Grokking or Glitching? How Low-Precision Drives Slingshot Loss Spikes

    [https://arxiv.org/abs/2605.06152](https://arxiv.org/abs/2605.06152)

    本文证明深度神经网络长期训练中周期性的“弹弓机制”损失尖峰并非源于优化动力学本身，而是浮点精度极限所致——当模型进入高置信度阶段后，正确类别梯度因舍入误差变为零，打破跨类别梯度零和约束，引发分类器与特征间的系统性漂移和正反馈循环。

    

    深度神经网络在无正则化的长期训练过程中会表现出周期性的损失尖峰，这一现象被称为“弹弓机制”。现有工作通常将其归因于内在的优化动力学，但其触发机制仍不清楚。本文证明该现象是浮点算术精度极限的结果：当训练进入高置信度阶段后，正确类别 logit 与其他 logit 之间的差值可能超过吸收误差阈值。于是在反向传播过程中，正确类别的梯度被精确舍入为零，而错误类别的梯度仍保持非零。这打破了跨类别梯度的零和约束，并在分类器层的参数更新中引入了系统性漂移。我们证明该漂移与特征之间形成了正反馈回路，导致全局分类器均值与全局特征（摘要在此处截断）。

    arXiv:2605.06152v4 Announce Type: replace-cross  Abstract: Deep neural networks exhibit periodic loss spikes during unregularized long-term training, a phenomenon known as the "Slingshot Mechanism." Existing work usually attributes this to intrinsic optimization dynamics, but its triggering mechanism remains unclear. This paper proves that this phenomenon is a result of floating-point arithmetic precision limits. As training enters a high-confidence stage, the difference between the correct-class logit and the other logits may exceed the absorption-error threshold. Then during backpropagation, the gradient of the correct class is rounded exactly to zero, while the gradients of the incorrect classes remain nonzero. This breaks the zero-sum constraint of gradients across classes and introduces a systematic drift in the parameter update of the classifier layer. We prove that this drift forms a positive feedback loop with the feature, causing the global classifier mean and the global featu
    
[^134]: 面向数据受限训练的规范性缩放定律

    Prescriptive Scaling Laws for Data Constrained Training

    [https://arxiv.org/abs/2605.01640](https://arxiv.org/abs/2605.01640)

    该论文提出了一种带加性过拟合惩罚项的规范性缩放定律，能准确刻画数据受限训练中重复数据的影响，并给出新的计算最优分配建议——当数据重复超过一定限度后，算力应投向扩大模型容量而非继续重复训练。

    

    训练所需的计算量正日益超过高质量数据的可用量，这使得核心挑战从最优计算分配转变为如何从有限数据中提取最大价值。被广泛采用的 Chinchilla 缩放定律假设每个训练 token 都是唯一的，这限制了它在数据受限场景下指导预训练决策的能力。我们用一个简单的加性过拟合惩罚项对重复训练下的超额损失进行建模，发现它能准确描述模型行为。我们的缩放定律给出了性质上全新的计算最优分配建议：超过某个点后，进一步的重复训练会适得其反，此时计算资源最好用于提升模型容量。我们证明，遵循该定律推荐的配置可以改善数据受限场景下的模型性能。最后，由于我们的单参数形式将过拟合隔离在单个系数中，它使得不同训练配置之间能够进行直接比较。

    arXiv:2605.01640v2 Announce Type: replace-cross  Abstract: Training compute is increasingly outpacing the availability of high-quality data. This shifts the central challenge from optimal compute allocation to extracting maximum value from limited data. The widely adopted Chinchilla scaling law assumes every training token is unique. This limits its ability to guide pretraining decisions in data-constrained regimes. We model the excess loss under repetition with a simple additive overfitting penalty and find that it accurately describes model behavior. Our scaling law yields qualitatively new compute-optimal allocation advice. Beyond a point, further repetition is counterproductive and compute is better spent on model capacity. We show that following our law's recommended configuration improves performance in data-constrained regimes. Finally, because our one-parameter form isolates overfitting in a single coefficient, it enables direct comparison across training configurations. As a c
    
[^135]: EvoDesign：基于设计专长演化的智能体式可编辑图表创作

    EvoDesign: Agentic Editable Diagram Creation via Design Expertise Evolution

    [https://arxiv.org/abs/2604.09568](https://arxiv.org/abs/2604.09568)

    EvoDesign提出了一种智能体框架，通过机器可操作且用户可编辑的中间画布模式、结构-样式-布局协同智能体、诊断验证环境以及从精修轨迹中持续积累设计专长的演化机制，实现了高保真的可编辑图表创作。

    

    高保真的图表创作需要在语义拓扑、视觉样式和几何布局上进行协调一致的视觉空间决策。现有方法面临表征鸿沟：基于像素的生成方式提供的对象级控制有限，而基于代码的合成方式虽然提供了可执行的结构，却牺牲了直观的操作性。我们提出了EvoDesign，这是一个智能体框架，它通过一种既可被机器操作、又可供用户直接操控的中间画布模式来实现可编辑的图表创作。EvoDesign首先通过结构、样式与布局三个协同工作的智能体构建图表清单，随后在一个诊断验证环境中对该清单进行渲染。该环境能够产生客观的缺陷信号和基于视觉语言模型的批评意见，从而实现对当前图表的局部化精修，同时为长期的设计专长演化提供证据。该演化机制从精修轨迹中提炼出可复用的设计经验。

    arXiv:2604.09568v3 Announce Type: replace-cross  Abstract: High-fidelity diagram creation requires coordinated visual-spatial decisions over semantic topology, visual styling, and geometric layout. Existing methods face a representation gap: pixel-based generation offers limited object-level control, while code-based synthesis provides executable structure at the cost of intuitive manipulation. We introduce EvoDesign, an agentic framework for editable diagram creation through an intermediate canvas schema that is both machine-actionable and directly manipulable by users. EvoDesign first constructs a diagram manifest through coordinated structure, style, and layout agents, and then renders the manifest in a diagnostic verification environment. This environment produces objective defect signals and VLM-based critique, enabling localized refinement of the current diagram while also providing evidence for long-term design expertise evolution. The evolution mechanism distills refinement tra
    
[^136]: 面向连续提示词搜索的无投影演化策略

    Projection-Free Evolution Strategies for Continuous Prompt Search

    [https://arxiv.org/abs/2603.13786](https://arxiv.org/abs/2603.13786)

    本文通过实证发现随机投影无法充分捕捉提示词空间的内在低维结构，进而提出一种基于演化策略的无投影连续提示词搜索方法，直接在完整提示词空间中优化并利用与内在维度相适配的自适应机制。

    

    连续提示词搜索为自然语言处理任务中的传统参数微调提供了一种计算高效的替代方案。然而，其现实有效性可能因目标函数景观的黑盒特性和固有的高维性而受到显著阻碍。现有方法通常通过将搜索限制在随机投影后的低维子空间中来缓解这些挑战。然而，投影机制的有效性及其背后的动机仍不明确。在本文中，我们首先通过实证研究表明，尽管提示词空间确实具有低维结构，但随机投影无法充分捕捉这一本质结构。受此发现的启发，我们提出了一种基于演化策略的无投影提示词搜索方法。该方法直接在完整的提示词空间中进行优化，并采用一种根据内在维度进行校准的自适应机制。

    arXiv:2603.13786v2 Announce Type: replace  Abstract: Continuous prompt search offers a computationally efficient alternative to conventional parameter tuning in natural language processing tasks. Nevertheless, its practical effectiveness can be significantly hindered by the black-box nature and the inherent high-dimensionality of the objective landscapes. Existing methods typically mitigate these challenges by restricting the search to a randomly projected low-dimensional subspace. However, the effectiveness and underlying motivation of the projection mechanism remain ambiguous. In this paper, we first empirically demonstrate that despite the prompt space possessing a low-dimensional structure, random projections fail to adequately capture this essential structure. Motivated by this finding, we propose a projection-free prompt search method based on evolutionary strategies. By directly optimizing in the full prompt space with an adaptation mechanism calibrated to the intrinsic dimensio
    
[^137]: ProMoral-Bench：评估大语言模型中道德推理与安全性的提示策略

    ProMoral-Bench: Evaluating Prompting Strategies for Moral Reasoning and Safety in LLMs

    [https://arxiv.org/abs/2602.13274](https://arxiv.org/abs/2602.13274)

    该论文提出了ProMoral-Bench基准，系统比较了四种大语言模型家族上的10种提示范式在道德推理与安全方面的表现，并发现紧凑的示例引导式提示能在更低的token成本下取得更强的能力与鲁棒性权衡。

    

    提示设计会影响大语言模型（LLM）所产生的道德判断和拒绝行为，然而受控比较研究仍分散于不同数据集和模型之间。我们提出了ProMoral-Bench，这是一个基准式的比较研究，涵盖四种大语言模型家族上的10种提示范式。我们使用ETHICS、Scruples、WildJailbreak以及我们构建的200对ETHICS-Contrast数据集，报告了任务级指标，以及道德能力分数、针对越狱/过度拒绝的安全鲁棒性分数和统一道德安全分数。UMSS是针对本研究的对MCS和SRS的相对性总结指标，而非对模型安全性或道德基础的绝对度量。在这一固定模板评估中，紧凑的示例引导式提示支架比所测试的特定多阶段提示变体在更低的token成本下实现了更强的能力-鲁棒性权衡，尽管多阶段提示偶尔能改善对越狱的拒绝能力。少样本演示是无重叠的（原文摘要至此处截断）。

    arXiv:2602.13274v2 Announce Type: replace  Abstract: Prompt design affects the moral judgments and refusal behavior produced by large language models (LLMs), yet controlled comparisons remain fragmented across datasets and models. We introduce ProMoral-Bench, a benchmark-style comparison of 10 prompting paradigms across four LLM families. Using ETHICS, Scruples, WildJailbreak, and our 200-pair ETHICS-Contrast set, we report task-level metrics together with Moral Competence Score (MCS), jailbreak/over-refusal Safety Robustness Score (SRS), and Unified Moral Safety Score (UMSS). UMSS is a study-specific relative summary of MCS and SRS, not an absolute measure of model safety or moral grounding. In this fixed-template evaluation, compact exemplar-guided scaffolds attain stronger competence-robustness trade-offs at lower token cost than the particular multi-stage variants tested, although multi-stage prompting occasionally improves jailbreak refusal. The few-shot demonstrations are non-ove
    
[^138]: 重新思考掩码扩散语言模型训练中的位置监督

    Reconsidering Positional Supervision in Masked Diffusion Language Model Training

    [https://arxiv.org/abs/2601.22947](https://arxiv.org/abs/2601.22947)

    该论文通过受控干预实验发现掩码扩散语言模型在迭代并行解码下对微小的位置偏移极为敏感（仅偏移1%的token即显著降低胜率），并据此提出改造连接主义时间分类（CTC）以改进位置监督。

    

    掩码扩散语言模型（MDLMs）通过并行解除掩码的方式生成文本，近来已成为自回归语言模型的替代方案。它们可以被视为采用逐位置交叉熵（CE）损失训练的并行解码器，这一设置与非自回归翻译（NAT）相同。在NAT领域，已有观点认为采用CE训练的并行解码器对微小的位置偏移十分敏感，因为CE会对这类偏移施加严厉惩罚。我们探究了采用CE训练的MDLMs在迭代解码下是否对这类偏移同样敏感。为验证这一点，我们施加了一种受控干预，在解码过程中引入位置偏移。在LLaDA-8B-Instruct模型与Arena-Hard基准上，仅将1%的生成token移动一个位置，就会显著降低相对于未干预模型的胜率，这表明MDLMs在迭代并行解码下对如此微小的偏移非常敏感。受此启发，我们改造了连接主义时间分类（CTC）……（原文摘要在此处截断）

    arXiv:2601.22947v3 Announce Type: replace  Abstract: Masked diffusion language models (MDLMs) generate text by unmasking tokens in parallel and have recently emerged as alternatives to autoregressive language models. They can be viewed as parallel decoders trained with a position-wise cross-entropy (CE) loss, the same setup as non-autoregressive translation (NAT). In NAT, CE-trained parallel decoders have been argued to be sensitive to small positional shifts, since CE penalizes them harshly. We ask whether CE-trained MDLMs are similarly sensitive to such shifts under iterative decoding. To probe this, we apply a controlled intervention that introduces them during decoding. On LLaDA-8B-Instruct with Arena-Hard, displacing as little as 1% of generated tokens by one position substantially reduces win rates against the unintervened model, showing that MDLMs are sensitive to such small shifts under iterative parallel decoding. Motivated by this, we adapt connectionist temporal classificati
    
[^139]: 深度循环注意力混合： 让潜在推理得到应有的关注

    Depth-Recurrent Attention Mixture (Dreamer): Giving Latent Reasoning the Attention it Deserves

    [https://arxiv.org/abs/2601.21582](https://arxiv.org/abs/2601.21582)

    该论文提出Dreamer模块化框架，通过结合序列注意力、深度注意力和稀疏专家注意力来突破恒定隐藏维度的瓶颈，使深度循环模型在语言推理基准上以2-8倍更少的训练token达到与SOTA相当的精度，并优于约2倍规模的SOTA模型。

    

    深度循环通过跨深度共享参数来促进潜在推理。然而，先前的工作缺乏同时匹配FLOP、参数量和内存的基线对比，由于部分固定的层堆叠而未充分利用深度循环，并且忽略了限制多步潜在推理的恒定隐藏维度瓶颈。为解决这一问题，我们提出了一个深度循环注意力混合的模块化框架，结合了序列注意力、深度注意力和稀疏专家注意力。该框架通过沿深度方向的注意力缓解了隐藏维度瓶颈，解耦了各缩放维度，使深度循环模型能够高效且有效地扩展。在语言推理基准测试中，我们的模型在与FLOP、参数量和内存均匹配的SOTA模型达到相同精度时，所需训练token减少2到8倍，并在相同训练token数量下优于约2倍规模的SOTA模型。我们还进一步提供了关于知识的洞见……

    arXiv:2601.21582v2 Announce Type: replace  Abstract: Depth-recurrence facilitates latent reasoning by sharing parameters across depths. However, prior work lacks combined FLOP-, parameter-, and memory-matched baselines, underutilizes depth-recurrence due to partially fixed layer stacks, and ignores the bottleneck of constant hidden-sizes that restricts many-step latent reasoning. To address this, we introduce a modular framework of depth-recurrent attention mixtures (Dreamer), combining sequence attention, depth attention, and sparse expert attention. It alleviates the hidden-size bottleneck through attention along depth, decouples scaling dimensions, and allows depth-recurrent models to scale efficiently and effectively. Across language reasoning benchmarks, our models require 2 to 8x fewer training tokens for the same accuracy as FLOP-, parameter-, and memory-matched SOTA, and outperform ca. 2x larger SOTA models with the same training tokens. We further present insights into knowled
    
[^140]: 自我改进的训练与思考

    Self-Improving Training & Thinking

    [https://arxiv.org/abs/2601.21343](https://arxiv.org/abs/2601.21343)

    该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。

    

    传统上，大型语言模型采用分阶段训练：先在原始文本上进行预训练，再进行后训练以学习指令遵循和推理能力。然而，这种分离造成了根本性的局限：许多理想的行为，如安全性、事实准确性、整体生成质量和推理能力，只在训练后期才被加入，尽管早期学习的模式在很大程度上塑造了模型的能力。为解决这一问题，我们提出了一种新的预训练和中期训练方法，将这些行为更早地融入训练过程。我们利用一个现有的强大的经过后训练的模型来重写数据并评判策略模型的输出（rollouts），从而在训练中更早地引入思考和强化学习。我们进一步通过引入“自我改进的后思考”（self-improving post-thinking）的概念，在所有训练阶段提升推理能力：即在生成之后进行推理，以提高模型的推理和自我评判能力。在实验中，我们证明了这些方法……

    arXiv:2601.21343v4 Announce Type: replace-cross  Abstract: Large language models are classically trained in stages: pretraining on raw text followed by post-training for instruction following and reasoning. However, this separation creates a fundamental limitation: many desirable behaviors such as safety, factuality, overall generation quality, and reasoning ability are only added at a late stage, even though the patterns learned earlier strongly shape a model's capabilities. To tackle this issue, we introduce a new way to pretrain and mid-train models that incorporates these behaviors earlier. We utilize an existing strong, post-trained model to both rewrite data and to judge policy model rollouts, thus introducing thinking and reinforcement earlier in training. We further improve reasoning at all stages by introducing the concept of self-improving post-thinking: improving the model's reasoning and self-judging ability by reasoning after generation. In our experiments, we show these a
    
[^141]: HyperGraphPro：面向结构引导超图RAG的进度感知强化学习

    HyperGraphPro: Progress-Aware Reinforcement Learning for Structure-Guided Hypergraph RAG

    [https://arxiv.org/abs/2601.17755](https://arxiv.org/abs/2601.17755)

    HyperGraphPro提出了一种进度感知的超图RAG框架，将证据路径的进展作为检索与强化学习策略优化的共同目标，并通过查询条件下的结构独特性重排序候选超边，解决了现有GraphRAG方法中检索与强化学习信号不匹配的问题。

    

    多跳问答要求检索智能体在多个推理轮次中构建一条连贯的证据路径，而不是为单个查询检索孤立的相关证据。然而，在现有的GraphRAG方法中，检索与强化学习往往采用不匹配的信号进行优化：检索由局部语义相关性驱动，而结果层面的强化学习仅通过评估最终答案来分配奖励，却无法识别哪些检索动作真正推动了推理路径的进展。我们提出了HyperGraphPro，一个进度感知的超图RAG框架，它将证据路径的进展作为检索与策略优化的共同目标。在检索方面，HyperGraphPro利用查询条件下的结构独特性对候选超边进行重排序，将实体-查询相关性与局部-全局超图关联信号相结合，以优先选择在当前图中具有信息量的实体（摘要在此处被截断）。

    arXiv:2601.17755v3 Announce Type: replace  Abstract: Multi-hop question answering requires retrieval agents to construct a connected evidence path across reasoning turns rather than retrieve isolated relevant evidence for a single query. However, in existing GraphRAG methods, retrieval and reinforcement learning are often optimized with mismatched signals: retrieval is driven by local semantic relevance, while outcome-level reinforcement learning assigns the reward by assessing the final answer without identifying which retrieval actions progress the reasoning path. We introduce HyperGraphPro, a progress-aware hypergraph RAG framework that focuses on evidence-path progress as the shared objective for retrieval and policy optimization. For retrieval, HyperGraphPro reranks candidate hyperedges using query-conditioned structural distinctiveness, combining entity-query relevance with a local-vs-global hypergraph incidence signal to favor entities that are informative in the current graph n
    
[^142]: StreamGaze：流式视频中的视线引导时序推理与主动式理解

    StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos

    [https://arxiv.org/abs/2512.01707](https://arxiv.org/abs/2512.01707)

    StreamGaze是首个评估多模态大语言模型能否在流式视频中利用人类视线信号进行时序推理与主动式响应的基准，通过视线引导的过去、现在和主动式三类任务全面检验流式视频理解能力。

    

    流式视频理解要求模型不仅能够按时间顺序处理连续到来的视频帧，还需要预判用户意图，以满足增强现实（AR）眼镜等现实应用场景的需求。尽管现有的流式基准测试已对时序推理能力进行了评估，但尚无任何基准衡量多模态大语言模型（MLLMs）能否在流式环境中解读或利用人类视线（gaze）信号。为填补这一空白，我们提出了StreamGaze，这是首个专门评估MLLMs在流式视频中利用视线进行时序推理与主动式推理能力的基准。StreamGaze引入了视线引导的过去、现在和主动式任务，对流式视频理解进行全面评估。其中，过去和现在任务要求模型基于查询时刻可获得的观察立即做出响应，而主动式任务则要求模型持续处理不断到来的视频帧，并在合适的未来时机做出响应。为了构建StreamGaze，

    arXiv:2512.01707v4 Announce Type: replace-cross  Abstract: Streaming video understanding requires models not only to process temporally incoming frames, but also to anticipate user intent for realistic applications such as Augmented Reality (AR) glasses. While prior streaming benchmarks evaluate temporal reasoning, none measure whether Multimodal Large Language Models (MLLMs) can interpret or leverage human gaze signals within a streaming setting. To fill this gap, we introduce StreamGaze, the first benchmark designed to evaluate how effectively MLLMs utilize gaze for temporal and proactive reasoning in streaming videos. StreamGaze introduces gaze-guided past, present, and proactive tasks that comprehensively assess streaming video understanding. Past and present tasks require immediate responses based on observations available at query time, whereas proactive tasks require models to continuously process incoming frames and respond at the appropriate future moment. To build StreamGaze,
    
[^143]: 使用医学分类数据对大语言模型进行指令微调可提升德语肿瘤诊断的ICD编码准确性

    Instruction-Tuning Large Language Models with Medical Classification Data Improves ICD Coding of German Tumor Diagnoses

    [https://arxiv.org/abs/2510.13624](https://arxiv.org/abs/2510.13624)

    本研究基于ICD-10-GM、ICD-O-3和OPS医学分类目录构建了超过50万个问答对，对Qwen、Llama和Mistral系列开源大语言模型进行指令微调，从而显著提升了德语肿瘤诊断文本的ICD编码准确性。

    

    使用ICD-10-GM和ICD-O-3对肿瘤诊断进行准确编码，对于德国结构化癌症文档记录至关重要。较小的开源权重大语言模型在隐私保护自动化方面具有吸引力，但在德语语境中往往难以达到编码准确性。本研究探讨了基于公开数据集的指令微调能否提高开源权重大语言模型对德语肿瘤诊断文本的编码准确性。评估采用来自本地肿瘤文档系统的已编码诊断作为测试数据。在系统性数据质量评估中，ICD-10编码性能的上限估计为：精确编码为60-79%，部分编码（仅三位字符编码）为81-94%。作为训练数据，研究者基于ICD-10-GM、ICD-O-3和OPS目录创建了超过50万个问答对，并对来自Qwen、Llama和Mistral系列的八个开源权重模型（7-70B参数）进行了微调。ICD-10-GM a…（原文摘要在此处截断）

    arXiv:2510.13624v2 Announce Type: replace-cross  Abstract: Accurate coding of tumor diagnoses with ICD-10-GM and ICD-O-3 is essential for structured cancer documentation in Germany. Smaller open-weight LLMs are appealing for privacy-preserving automation but often struggle with coding accuracy in German-language contexts. This study investigates whether instruction-based fine-tuning on public datasets improves the coding accuracy of open-weight LLMs for German tumor diagnosis texts. The evaluation uses coded diagnoses from the local tumor documentation system as test data. In a systematic data quality assessment, the upper limit for ICD-10 coding performance was estimated at 60-79% for exact and 81-94% for partial (three-character codes only) derivation. As training data, over 500,000 question-answer pairs were created based on the ICD-10-GM, ICD-O-3, and OPS catalogues. Eight open-weight models from the Qwen, Llama, and Mistral families (7-70 B parameters) were fine-tuned. ICD-10-GM a
    
[^144]: 面向大语言模型的多数位感知水印方法

    Majority Bit-Aware Watermarking For Large Language Models

    [https://arxiv.org/abs/2508.03829](https://arxiv.org/abs/2508.03829)

    提出多数位感知编码范式，将水印信号强度与绿色列表大小解耦，从而在不限制绿色列表大小的情况下保持强水印信号，改善了文本质量与解码准确率之间的权衡。

    

    大语言模型（LLM）的日益普及引发了人们对其在生成有害或欺骗性内容方面被滥用的担忧。为解决这一问题，已有水印方法被提出，用于在生成的文本中嵌入可识别的多比特消息以实现滥用追踪。然而，现有方法往往面临文本质量与解码准确率之间的根本性权衡。特别是，这些方法必须在编码时限制偏好词元集合（即绿色列表）的大小，以维持可供解码检测的水印信号，这不可避免地降低了生成质量。为改善这一权衡，我们提出了一种新颖的消息编码范式，称为“多数位感知编码”，它将水印信号强度从绿色列表大小中解放出来。该策略使得即使使用较大的绿色列表，也能在生成的文本中保留强水印信号。我们引入了两种实例化方法……

    arXiv:2508.03829v3 Announce Type: replace  Abstract: The growing deployment of Large Language Models (LLMs) has raised concerns about their misuse in generating harmful or deceptive content. To address this issue, watermarking methods have been proposed to embed identifiable multi-bit messages into generated text for misuse tracing. However, existing methods often suffer from a fundamental trade-off between text quality and decoding accuracy. In particular, they have to restrict the size of the preferred token set (i.e., green list) during encoding to maintain a detectable watermark signal for decoding, which inevitably degrades generation quality. To improve this trade-off, we propose a novel message encoding paradigm called \textit{majority bit-aware encoding}, which relaxes the watermark signal strength from the green list size. This strategy allows for a strong watermark signal to be preserved in generated texts even when using a large green list. We introduce two instantiations of
    
[^145]: 基于自我反馈强化学习的大语言模型后训练

    Post-Training Large Language Models via Reinforcement Learning from Self-Feedback

    [https://arxiv.org/abs/2507.21931](https://arxiv.org/abs/2507.21931)

    该论文提出了RLSF方法，利用模型自身对答案的内在置信度作为自我生成的奖励来构建合成偏好数据，无需人工标注或标准答案即可通过偏好优化提升大语言模型在推理任务上的可靠性和校准度。

    

    大语言模型（LLMs）经常产生看似合理但校准不佳的答案，这限制了它们在推理密集型任务上的可靠性。最近的研究表明，思维链（CoT）推理路径是预训练大语言模型所固有的，可以通过简单地改变解码过程来引出，其中CoT路径的存在与更高的答案置信度相关。基于这些见解，我们提出了自我反馈强化学习（RLSF），这是一种后训练阶段，利用模型的内在置信度作为自我生成的奖励。通过从冻结的大语言模型生成多个CoT解码束，我们计算每个最终答案跨度的置信度，并相应地对由此产生的推理轨迹进行排序以创建合成偏好。这些偏好随后被用于通过标准偏好优化来微调策略，无需人工标签、标准答案或外部构建的奖励。

    arXiv:2507.21931v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) often produce plausible but poorly-calibrated answers, limiting their reliability on reasoning-intensive tasks. Recent research suggests that Chain-of-Thought (CoT) reasoning paths are inherent in pre-trained LLMs and can be elicited by simply altering the decoding process, where the presence of a CoT path correlates with higher answer confidence. Building on these insights, we present Reinforcement Learning from Self-Feedback (RLSF), a post-training stage that utilises the model's intrinsic confidence as a self-generated reward. By generating multiple CoT decoding beams from a frozen LLM, we compute the confidence of each final answer span and rank the resulting traces accordingly to create synthetic preferences. These preferences are subsequently utilised to fine-tune the policy through standard preference optimisation, requiring no human labels, gold answers, or externally curated rewards. RLSF s
    
[^146]: CoMPAS3D：面向交互式运动的数据集与基准

    CoMPAS3D: A Dataset and Benchmark for Interactive Motion

    [https://arxiv.org/abs/2507.19684](https://arxiv.org/abs/2507.19684)

    该论文提出将具有规范动作词汇和社区公认评判标准的萨尔萨舞作为具身交互评估领域，并构建了CoMPAS3D数据集与基准，以评估交互式动作生成中反应性动作的语义有效性与恰当性。

    

    社交互动型人形机器人旨在通过身体与人类进行交互，并实时适应伙伴的运动、意图和能力。这需要模型既能理解身体如何运动，又能理解这些运动所表达的含义。然而，现有的交互式动作生成评估框架并未评估所生成的反应性动作在共享动作词汇体系中是否有效或恰当：现有框架依赖FID、节拍对齐等运动学指标，无法捕捉这一语义属性，而即兴互动数据集又缺乏细粒度的动作标注。受围棋、国际象棋等游戏计算化评估的启发，我们提出将萨尔萨舞作为一个具身交互评估领域。萨尔萨舞具有交互性和双人配对性，并受动作词汇以及社区公认的评判标准（涵盖时机、音乐性、技术、难度、配合与原创性）的约束。我们预……（原文摘要在此处截断）

    arXiv:2507.19684v3 Announce Type: replace-cross  Abstract: Socially interactive humanoids aim to interact with humans through their bodies, adapting in real time to a partner's movement, intent, and abilities. This requires models that understand both how bodies move and what their movements mean. Yet, existing evaluation frameworks for interactive motion generation do not assess whether generated reactive motions are valid or appropriate within a shared movement vocabulary. Existing frameworks rely on kinematic metrics such as FID and beat alignment that cannot capture this semantic property, while improvised interaction datasets lack fine-grained move annotations. Inspired by computational evaluations of games such as Go and Chess, we propose Salsa as an embodied interaction evaluation domain. Salsa is interactive, dyadic, and governed by a move vocabulary and community-validated judging criteria covering timing, musicality, technique, difficulty, partnering, and originality. We pres
    
[^147]: 超越共情：将诊断与治疗推理融入大语言模型以实现心理健康咨询

    Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with Large Language Models for Mental Health Counseling

    [https://arxiv.org/abs/2505.15715](https://arxiv.org/abs/2505.15715)

    PsyLLM是首个为心理健康咨询融入诊断与治疗推理的大语言模型，通过新颖的自动化数据合成流程，将Reddit心理健康帖子转化为基于国际诊断标准（DSM/ICD）和多种治疗策略的多轮咨询对话。

    

    大语言模型（LLMs）在心理支持领域具有巨大潜力，能够生成共情性回应并模拟治疗性对话。然而，现有基于LLM的方法往往缺乏足够的结构化诊断推理和多样化治疗策略的支撑，限制了其在复杂咨询场景中的有效性。为了解决这些局限性，我们提出了PsyLLM，这是首个为心理健康咨询设计的、融合诊断导向和治疗导向推理的大语言模型。为了开发PsyLLM，我们设计了一种新颖的自动化数据合成流程，用于处理从Reddit收集的真实世界心理健康帖子——用户经常在这些帖子中表达心理困扰并寻求社区支持。该流程将这些帖子转化为多轮咨询对话，并利用在国际诊断标准（如DSM/ICD）和多种治疗方法的指导下进行的大语言模型来……

    arXiv:2505.15715v3 Announce Type: replace  Abstract: Large language models (LLMs) hold significant potential for psychological support, capable of generating empathetic responses and simulating therapeutic conversations. However, existing LLM-based approaches often lack sufficient grounding in structured diagnostic reasoning and diverse therapeutic strategies, limiting their effectiveness in complex counseling scenarios. To address these limitations, we propose PsyLLM, the first large language model designed to incorporate diagnostic- and therapeutic-oriented reasoning for mental health counseling. To develop PsyLLM, we design a novel automated data synthesis pipeline that processes real-world mental health posts collected from Reddit, where users frequently express psychological distress and seek community support. The pipeline transforms these posts into multi-turn counseling dialogues and leverages LLMs guided by international diagnostic standards (e.g., DSM/ICD) and multiple therap
    
[^148]: 通过迭代洗牌研究词序

    Studying word order through iterative shuffling

    [https://arxiv.org/abs/2109.04867](https://arxiv.org/abs/2109.04867)

    该论文提出了一种名为IBIS（迭代洗牌推理）的新颖高效算法，能在固定语言模型下找到词语的最优排序，并利用该方法发现句子中的词语很少能被重排成含义截然不同的短语，从而在许多情况下驳斥了“词序编码对NLP任务至关重要的语义”这一假设。

    

    随着神经语言模型在NLP基准任务上的表现接近人类水平，其进步被广泛视为对句法理解日益复杂的证据。这一观点基于一个尚未经过实证检验的假设：词序编码了对执行这些任务至关重要的语义。我们在许多情况下驳斥了这一假设：在GLUE基准套件和各类英文文本体裁中，句子或短语中的词语很少能通过重新排列形成携带实质不同信息的短语。我们这一令人惊讶的结果依赖于迭代洗牌推理（IBIS），这是一种新颖且高效的程序，能够在固定语言模型下找出一组词语具有最高可能性的排序。IBIS可以使用任何黑盒模型而无需额外训练，并且优于现有的词序排序算法。综合我们的研究发现，我们讨论了诸如IBIS之类的洗牌推理程序如何有益于……

    arXiv:2109.04867v2 Announce Type: replace  Abstract: As neural language models approach human performance on NLP benchmark tasks, their advances are widely seen as evidence of an increasingly complex understanding of syntax. This view rests upon a hypothesis that has not yet been empirically tested: that word order encodes meaning essential to performing these tasks. We refute this hypothesis in many cases: in the GLUE suite and in various genres of English text, the words in a sentence or phrase can rarely be permuted to form a phrase carrying substantially different information. Our surprising result relies on inference by iterative shuffling (IBIS), a novel, efficient procedure that finds the ordering of a bag of words having the highest likelihood under a fixed language model. IBIS can use any black-box model without additional training and is superior to existing word ordering algorithms. Coalescing our findings, we discuss how shuffling inference procedures such as IBIS can benef
    

