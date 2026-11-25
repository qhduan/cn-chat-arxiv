# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiaCap: Jointly Learning Who Speaks When for Audiovisual Dialogue Captioning](https://arxiv.org/abs/2611.33153) | DiaCap是一个统一的视听大语言模型，通过合成数据预训练与强化学习微调两个互补阶段，联合学习话语时间戳、视听说话人分离与说话人绑定，为视频中每句话生成转录、时间戳和说话人外观描述，实现全面的对话字幕生成。 |
| [^2] | [CoT-Proxy: Cross-Format Transfer for Distilling Long-Horizon Policies](https://arxiv.org/abs/2611.33148) | 该论文提出CoT-Proxy方法，证明即使教师模型的思维链被闭源API隐藏，也可以仅凭其动作序列（学生推理块留空）完成长程策略蒸馏，且所学策略能以跨格式迁移的方式在启用思考的推理阶段生效。 |
| [^3] | [BriConstrain: Geometrically Constrained Decoding for Autoregressive Brick Assembly Generation](https://arxiv.org/abs/2611.33135) | BriConstrain 通过在采样前动态检查积木放置的几何有效性，并以约束零件连接方式的方式缩小搜索空间，引导语言模型生成无碰撞、连接有效的乐高积木组装。 |
| [^4] | [A Bias in Disguise: Tiny Gradient Edits Stop a Small Language Model's Thinking Loops and Complement a Frontier Model's Weight Edit](https://arxiv.org/abs/2611.33133) | 该论文发现通过梯度选择极少量关键权重微调即可让小模型走出无休止的思考循环，其本质是为无偏置项的模型隐式添加偏置，并与前沿模型的无梯度权重编辑形成互补。 |
| [^5] | [SAG-Mem-1.0 Technical Report](https://arxiv.org/abs/2611.33124) | SAG-Mem 提出了一种构建记忆时无需模型推理的对话记忆框架，通过保留完整对话分块，并在查询时利用 BM25 检索、证据评估与词汇扩展进行多轮检索，有效发现与问题相似度低但具互补性的证据。 |
| [^6] | [AIChipBench: Benchmarking LLM Agents for Automated AI Accelerator Design](https://arxiv.org/abs/2611.33093) | 该论文提出AIChipBench基准，要求LLM编程智能体用可综合RTL实现量化大模型解码器，重点考察跨解码步骤的状态保持正确性及面积受限下的性能优化，从而系统评估智能体在自动化AI加速器设计中的真实能力。 |
| [^7] | [FOL-Proof: Verifying LLM Text-to-Logic Translations via Deterministic Semantics](https://arxiv.org/abs/2611.33078) | 该论文提出了FOL-Proof语义防火墙流水线，通过英语资源语法等语义分析工具确定性地生成候选翻译，并利用SMT求解器进行逻辑等价性检查，从而独立验证大语言模型生成的一阶逻辑翻译的语义忠实性。 |
| [^8] | [Worth the Wait? How Response Time and Streaming Shape Judgments of Generative AI Answers and Chatbot Engagement](https://arxiv.org/abs/2611.33050) | 本文通过法国政府聊天机器人竞技场的自然实验发现，边写边流式输出的答案比同一模型一次性显示的相同存储答案在盲选投票中获胜概率高出25.8个百分点，证明答案的呈现方式（而不仅是内容）显著影响用户对生成式AI的评价。 |
| [^9] | [Latent Merger: Latent-Space Composition of Hypernetwork-Generated LoRAs](https://arxiv.org/abs/2611.33048) | 该论文发现超网络生成的LoRA在几何特性上与直接微调的LoRA截然不同（谱更集中、跨文档对齐更强），据此提出在潜空间中直接组合中间潜变量再解码的LoRA合并方法，以解决流式场景下逐文档存储LoRA不可持续的问题。 |
| [^10] | [Towards Realistic Evaluation of Uncertainty Quantification for LLM-Based Recommender Systems](https://arxiv.org/abs/2611.33036) | 该论文提出了一个固定排序基准，在统一的输出契约下将八种异构的不确定性量化信号作为记录级监控器进行评估，从而首次实现了对基于LLM的推荐系统中不确定性量化方法的公平、现实的可比性评估。 |
| [^11] | [Can a Frontier LLM Repair a Quantized Network by Tool-Free Weight-Space Programming?](https://arxiv.org/abs/2611.33031) | 该研究证明前沿大语言模型无需任何工具、数据或推理能力，仅凭权重空间编程即可将4比特量化CNN的准确率从60.4%恢复至77.95%，接近原始浮点模型的81.4%水平。 |
| [^12] | [SALSA: Sampling Adversarial Suffixes from Learned Distributions for LLM Safety Assessment](https://arxiv.org/abs/2611.33030) | SALSA提出了一种从优化攻击语料中学习越狱后缀分布的生成式框架，在将生成速度提升约90倍的同时达到与最先进优化攻击相当的成功率，并在黑盒迁移中对GPT-3.5-Turbo和GPT-4o分别实现99%和45%的攻击成功率。 |
| [^13] | [Can LLM Judges Catch Unsafe Clinical Actions? A Fault-Injection Study](https://arxiv.org/abs/2611.33012) | 该研究提出了一种故障注入评估方法，通过在MedAgentBench中向经验证的智能体轨迹注入416条受控临床错误（覆盖八个错误类别、六个开源LLM裁判），发现LLM裁判识别不安全临床行为的可靠性高度依赖于临床证据和完整轨迹的可见性——仅看最终答案时既漏检错误又误报高达56%的正确轨迹。 |
| [^14] | [Leto: Utility-Centric Agent Memory](https://arxiv.org/abs/2611.33005) | 提出了Leto，一个以效用为中心的LLM智能体记忆系统，通过记忆索引与记忆回放机制，关注记忆对未来响应的效用而非对过去对话的保真度。 |
| [^15] | [Modelling Billingual Learning Through LLM Training: A Critical-Period Exploration](https://arxiv.org/abs/2611.32997) | 该研究通过让小型GPT-2模型依次学习英语和不同距离的第二语言，并分析内部隐藏状态的余弦相似度，从模型内部表征角度探索了二语习得的关键期效应。 |
| [^16] | [On Potentiality and Actuality Of Foundation Models](https://arxiv.org/abs/2611.32991) | 基础模型本身已蕴含完成特定任务所需的能力，适配并非添加新能力，而只需通过恒定的、与输入无关的向量移位将模型“实现”即可。 |
| [^17] | [When Repair Accuracy Misleads: Training a VLM on Its Own Failures](https://arxiv.org/abs/2611.32982) | 该论文提出“验证式失败驱动修复”（VFR）方法，揭示了用模型自身失败样例微调视觉语言模型时准确率评估的两大陷阱（标签错误与措辞规律性），并通过LLM评审验证和最小对样例挖掘，将Qwen3-VL的空间关系准确率从66.7%提升至96.8%。 |
| [^18] | ["I Do Intend to Deliver": Will LLMs Run a Marketplace Advance-Fee Scam?](https://arxiv.org/abs/2611.32957) | 研究表明，仅用一条无需越狱的普通提示词，十个主流大语言模型中就有八个愿意充当电商预付费诈骗的“卖家”，在1200次模拟对话中61%的买家被诱导预付款且单次成本仅约2.5美分，凸显了LLM被滥用于在线欺诈的严重风险。 |
| [^19] | [ShellBench: What Matters More for AI Agents, Models, or Harness for Everyday Tasks](https://arxiv.org/abs/2611.32946) | 该论文推出ShellBench基准，通过对12个模型、5种智能体框架和3个推理水平共53,100次实验的评估发现，模型选择对日常任务性能的影响大于框架选择，但领先模型之间差距仅1.1个百分点，且高性能模型间的成本差异可达六倍之多。 |
| [^20] | [One Style Fits All? Cultural Values Embedded in Conversational AI via a People-Pleasing Lens](https://arxiv.org/abs/2611.32939) | 研究发现，无论在台湾还是韩国，支持型聊天机器人风格都能稳定提升用户的持续使用意愿、满意度和信任，而取悦倾向并不调节这些效应，仅在感知威胁上存在文化差异。 |
| [^21] | [InstructReplay: Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning](https://arxiv.org/abs/2611.32918) | 提出InstructReplay方法，通过让初始模型预先对通用指令生成回答并在微调时约束模型保持接近这些回答，在事实性微调中学习新知识的同时，将通用能力平均损失从28.1分降至2.4分，并避免了回答长度的崩溃。 |
| [^22] | [SOARS: System One Assisted Reranking for Agentic Search](https://arxiv.org/abs/2611.32917) | SOARS是一个无需训练的智能体搜索重排序框架，利用快速“系统一”模型在单次调用中完成相关性估计与候选重排序，在六个大模型骨干上显著提升答案准确率和证据召回率，同时大幅降低搜索调用次数、延迟和推理成本。 |
| [^23] | [Joint Detection of Constructive Speech, Hate Speech and Agreement in Online Conversations](https://arxiv.org/abs/2611.32889) | 本文构建了一个标注了仇恨语气、建设性语气和赞同态度的英丹麦语社交媒体评论大规模数据集，并通过微调多语言大语言模型实现三项任务的联合检测，其中 E5-large 在上下文增强架构下表现最佳。 |
| [^24] | [Fresh Keys, Stale Secrets: Numeric Leakage across Linked LLM Prompts](https://arxiv.org/abs/2611.32864) | 该论文揭示了针对LLM提示中机密商业数字的伪装保护在面对服务提供商的跨查询链接分析时会失效——即使每次查询使用新密钥，攻击者也能通过累积多次独立观察逐步恢复企业重复询问的机密价格与利润率信息。 |
| [^25] | [From Grammar Deltas to Token Masks: Incremental Constrained Decoding under Runtime Grammar Evolution](https://arxiv.org/abs/2611.32807) | 提出了GramDelta框架，通过在已提交前缀上维护解析状态并仅增量传播新增语法引起的变更，实现了运行时语法演化下的高效约束解码，避免了重复解析工作。 |
| [^26] | [Sample-Efficient In-Context Adaptation via Sleep](https://arxiv.org/abs/2611.32802) | 该研究提出在Transformer语言模型预训练中引入“睡眠”机制（通过多次循环前向传递精炼上下文表示），显著提升了模型在一般性任务上少样本上下文学习的样本效率。 |
| [^27] | [Is Prosody Encoded in the Neural Audio Codec Mimi?](https://arxiv.org/abs/2611.32795) | 本文研究了被Moshi、Sesame等对话语音智能体使用的神经音频编解码器Mimi在编码阶段是否捕获了韵律信息，填补了以往仅从解码端评估编解码器质量、缺乏对编码端语音信息考察的研究空白。 |
| [^28] | [Measuring Layer-wise Intrinsic Dimensionality of FFNs in LLMs via PCA](https://arxiv.org/abs/2611.32782) | 该研究用PCA测量58个开源LLM各层FFN的内在维度，发现中间层的内在维度显著高于边缘层的“中间重”分布，表明逐层均匀分配参数的传统设计并非最优，为非均匀的参数预算分配提供了实证依据。 |
| [^29] | [Beyond Fixed-Context Likelihood: Zero-Shot AI-Generated Text Detection via Context Perturbation Discrepancy](https://arxiv.org/abs/2611.32778) | 提出零样本AI生成文本检测方法CPD，通过衡量token对数概率增益相对于完整前缀分布期望的偏离程度（上下文扰动差异）来识别AI文本，在多个数据集上显著超越七个零样本基线，平均AUROC达94.80%。 |
| [^30] | [ASCENT: A Benchmark for Evaluating and Advancing Stepwise Diagnostic Reasoning in Large Language Models on Common Clinical Scenarios](https://arxiv.org/abs/2611.32734) | 该论文提出ASCENT，一个由临床医生标注、包含3,078个逐步推理问题的基准，通过将病例分解为与电子病历对齐的步骤（发现、印象、依据）来评估信息不完整条件下的中间诊断推理，揭示出即使最强的推理模型在逐步诊断任务上仍有显著提升空间。 |
| [^31] | [Red-Teaming a Sovereign LLM, Its Reasoning and Its Training Stages](https://arxiv.org/abs/2611.32710) | 该研究通过多语言红队测试发现，即使系统提示词中包含保密指令，主权开源LLM Apertus在诱饵攻击下仍会以超过95%的比例泄露敏感信息，且其推理轨迹也可能泄露最终答案中刻意隐去的机密内容。 |
| [^32] | [Bathys: Advancing Deep Search Agents](https://arxiv.org/abs/2611.32660) | 提出 Bathys 9B 和 27B 深度搜索智能体，通过可验证问题合成、SFT/KTO/CM-GSPO 训练流水线以及上下文压缩下的在线强化学习框架，提升智能体在复杂开放式任务中迭代检索与证据验证的能力。 |
| [^33] | [Generating Schema-Aligned Synthetic Dialogues and Measuring Their Diversity](https://arxiv.org/abs/2611.32631) | 该论文提出“先记录后对话”的逆序合成方法，通过先采样符合模式的有效记录再生成对话，无需人工标注即可获得成对数据，并利用变化配置、人物角色和主题规划等显式机制显著提升合成对话的多样性。 |
| [^34] | [MargEmbed-Akan: Bridging Twi Questions and English Knowledge through Cross-Lingual Score Distillation](https://arxiv.org/abs/2611.32630) | 该论文提出了 MargEmbed-Akan，一个基于跨语言分数蒸馏从 AfriE5-Large-instruct 训练得到的 5.6 亿参数开源查询编码器，有效连接契维语问题与英语知识检索。 |
| [^35] | [REBEAL: Human References Are No Longer the Best Translation, and Metrics Do Not Notice](https://arxiv.org/abs/2611.32623) | 该论文提出REBEAL基准，通过对参考译文施加七类受控污染来测试评价指标，发现人工参考译文质量已逊于顶级机器翻译系统，而现有八种主流指标均无法察觉这种污染，且各类型指标以不同方式失效。 |
| [^36] | [When "Neutral" Means Irrelevant: A Construct-Validity Audit of NormAd-ETI](https://arxiv.org/abs/2611.32614) | 本文审计了NormAd-ETI基准的结构效度，发现其三方评分混淆了“规范是否适用”与“行为是否符合规范”这两个决策，导致在仅给出国家名称时适用性判断接近随机水平却被评分掩盖，并建议采用专门的强制选择题探针将两者分离。 |
| [^37] | [Privacy Disclosure Control Benchmark (PDC-Bench): How Faithfully Do LLMs Keep Privacy Constraints on In-Context Disclosure Granularity?](https://arxiv.org/abs/2611.32610) | 提出了 PDC-Bench，首个将隐私披露控制形式化为知识图谱层级上限约束的基准（16,105 个样本，覆盖地理、年龄、种族与民族和职业），用于评估 LLM 能否忠实遵守上下文中关于披露粒度的隐私约束。 |
| [^38] | [Gated Concept Adapters: Layer-Wise Concept Bottlenecks for Auditable Text Understanding](https://arxiv.org/abs/2611.32600) | 提出了门控概念适配器（GCA）这一分层概念瓶颈框架，通过在预训练Transformer的选定层附加门控低维概念读出适配器，将隐藏状态转化为可检查、可干预的概念中介接口，从而实现可审计的文本理解。 |
| [^39] | [Avalon-M1: Reinforcement Learning from Collective Rewards](https://arxiv.org/abs/2611.32539) | Avalon-M1提出基于集体奖励的强化学习（RLCR），联合训练CEO与最多八名专家共享同一策略的智能体组织，在延迟约束下将IMO-ProofBench-Advanced分数从23.8%提升至39.0%，并以约单智能体三分之一的延迟在12个基准上取得相当的平均性能。 |
| [^40] | [What If Chinese Were Latinized? A Counterfactual Study of Input Representation, Tokenization, and Language Modeling](https://arxiv.org/abs/2611.32525) | 该研究通过反事实实验将中文语料转换为拼音并从头训练分词器与Llama风格语言模型，发现拼音表示会导致更高的切分率、更差的同音字消歧和语言建模性能，证明书面输入表示的选择并非中性编码，而会实质性影响语言建模效果。 |
| [^41] | [STAR: Counterfactual Dual-World Rollouts and Spatial Target Adaptive Reconstruction for Visual On-Policy Distillation](https://arxiv.org/abs/2611.32472) | STAR通过反事实双世界推演与空间注意力引导的目标自适应重建，迫使视觉语言模型在在线策略自蒸馏中真正依赖决定答案的局部视觉证据，而非语言先验和全局上下文。 |
| [^42] | [BCP: A Lightweight Boundary-Controlled Framework for Online Sign Language Recognition and Translation](https://arxiv.org/abs/2611.32471) | BCP是一个轻量级边界控制框架，通过冻结预训练离线模型并引入由CTC对齐监督的边界适配器，将其高效转换为在线手语识别与翻译方法，避免了昂贵的整体模型重训练。 |
| [^43] | [MAGPIE: Agentic Multi-Reference Image Generation via Executable Scene Construction](https://arxiv.org/abs/2611.32462) | MAGPIE将多参考图像生成的构图意图表示为可执行的Blender 3D场景程序，并由编码智能体在图像合成前自动检查、链接参考并迭代修正场景，从而实现更可靠的多参考图像生成。 |
| [^44] | [Phoneme-identity depth profiles do not reliably predict LoRA placement for dysarthric ASR](https://arxiv.org/abs/2611.32455) | 本研究通过在两个ASR模型和两个构音障碍语料库上的统一受控实验证明，基于音素可解码性深度剖面来选择LoRA适配器放置位置的方法并不可靠，音素错误曲线的最小值位置虽然相对稳定，但无法有效指导最佳适配带的选择。 |
| [^45] | [When Ink Meets Pixel: Intent-Aware OCR with VLMs for Human-Corrected Financial Documents When Ink Meets Pixel: Intent-Aware OCR with VLMs for Human-Corrected Financial Documents](https://arxiv.org/abs/2611.32446) | 该论文提出“意图感知OCR”新任务，构建了包含1,600份金融文档、5,700余个视觉意图标注的DocIntent数据集及IntentBench评测基准，并训练出DocIntentOCR视觉语言模型，使OCR系统能够理解文档上的人工修正标记（如箭头、删除线等）并生成修正后的转录文本。 |
| [^46] | [Factorized Context Attention: From Emulated to Native Grid Probing](https://arxiv.org/abs/2611.32423) | 提出了免训练的块稀疏注意力方法 FCA，将网格探测这种“模拟注意力”转化为模型原生的注意力机制，只需对输入编码一次进 KV 缓存，即可同时兼顾速度、精度与上下文长度三方面。 |
| [^47] | [On the Importance of Hidden-State Refinement in Masked Diffusion Language Models](https://arxiv.org/abs/2611.32416) | 该论文揭示了掩码扩散语言模型中隐藏状态的逐层与跨步骤精炼动态，并据此提出跨步骤循环精炼（CSRR）机制，通过在去噪步骤间传递隐藏状态信息来增强模型性能。 |
| [^48] | [ASTERISM: Do Multimodal LLMs Actually Look at All the Images?](https://arxiv.org/abs/2611.32316) | 该研究审计了19个多图像基准，发现约24%的题目存在“单图捷径”问题（仅凭一张图像即可作答），并据此构建了目前规模最大的经审计多图像基准ASTERISM（1.87万道题），专门保留真正需要多张图像才能解答的题目。 |
| [^49] | [ArabFake: A multitask deep learning framework for Arabic fake news Detection, Categorization, and risk prediction](https://arxiv.org/abs/2611.32303) | 本文提出基于 MARBERTv2 的多任务深度学习框架 ArabFake，可同时完成阿拉伯语虚假新闻检测、内容分类和风险等级预测三项任务，并发布了包含 2,495 条专家验证标注新闻的 ArabFake 数据集。 |
| [^50] | [Unjamming Deferrals: Supporting Students Through JAMN Bot, a Mobile RAG-Grounded Policy Assistant](https://arxiv.org/abs/2611.32266) | 本文提出了集成JAMN Bot的移动应用Schedulater，通过基于Ollama自托管Llama 3.2 3B的RAG管道，为大学缓考流程提供隐私保护的、基于政策的智能问答支持。 |
| [^51] | [ArEssays: Towards Robust Arabic Handwritten Text Understanding in Educational Settings](https://arxiv.org/abs/2611.32211) | 提出大规模阿拉伯语学生手写作文数据集ArEssays，通过对34个视觉-语言和OCR模型的基准测试，揭示了现有模型在真实儿童手写内容上的显著性能差距。 |
| [^52] | [Steerable by Design: Conditional Pretraining for Robust Control of Language Models](https://arxiv.org/abs/2611.32199) | 该论文提出了一种条件预训练方法和架构，通过编码器交叉注意力机制从一开始就将指令与数据通道分离，构建出Mancelle-3B这一在设计上即可被有效操控的语言模型，解决了模型误将数据当作指令处理的越狱与对齐漂移问题。 |
| [^53] | [ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks](https://arxiv.org/abs/2611.32183) | ThinkMorph-SSL 提出一种两阶段自监督训练方案，将图像上色、去噪等视觉前置任务转化为无需指令的交错式多模态推理轨迹，让图像本身充当监督信号，从而在相同指令微调预算下超越仅靠指令微调的模型。 |
| [^54] | [RightDiff: Learning Generative Reward Models to Judge by the Right Differences](https://arxiv.org/abs/2611.32166) | RightDiff通过在单次自回归生成中显式发现候选响应间的对比差异、区分实质性与非实质性因素并归纳针对响应对的评分规则，学习以“正确差异”为核心的生成式奖励模型，从而避免表面捷径与推理依据漂移。 |
| [^55] | [Making Evaluation Explicit Is Not Enough: Supervising the Evaluation Process in Generative Reward Models](https://arxiv.org/abs/2611.32161) | 提出STEP-GRM框架，通过为评分标准构建提供结构性指导和为准则级判断提供局部监督，直接监督生成式奖励模型的中间评估过程，使模型不仅学会正确的最终偏好，还掌握更可靠的评估流程。 |
| [^56] | [ARS-VLM: Visually Aligned Augmented Random Search for Vision-Language Models](https://arxiv.org/abs/2611.32147) | 提出ARS-VLM，一种视觉对齐的增强随机搜索零阶方法，解决了直接将进化策略类算法应用于视觉-语言模型时的图像块网格过拟合和灾难性遗忘两大失败模式，同时规避了GRPO等一阶强化学习带来的内存与系统开销。 |
| [^57] | [Calibrated Reading of Latent Reasoning and Communication](https://arxiv.org/abs/2611.32143) | 本文提出校准潜读取器（CLR），一个紧凑的Transformer，通过任务条件化交互、源适配器迁移与保形校准，以比语言模型解码器低数百倍的计算成本准确读取潜推理结果与键值通信内容，并提供统计覆盖率与误报保证。 |
| [^58] | [Fix How It Acts, Not How It Thinks: Surgical Fine-Tuning of Frontier LLMs](https://arxiv.org/abs/2611.32064) | 提出一种“精细微调”方法，不增大更新规模，而是直接在权重更新合并后残存的部分上进行训练，从而在不干扰模型推理能力的前提下纠正其行动层面的小失误，并解决了 LoRA 小幅更新在合并到量化发布版本时大量丢失的问题。 |
| [^59] | [Do Clinician-Scored Examples Help LLMs Code ADOS-2 from Hour-Long Transcripts?](https://arxiv.org/abs/2611.32050) | 本研究通过预注册的安慰剂对照实验首次系统检验了专家评分示例对大语言模型编码ADOS-2的作用，发现正确标签带来的增益有限，模型与临床医生的一致性仍然较低（二次加权kappa仅0.22–0.30）。 |
| [^60] | [HuluOPD: Unifying Heterogeneous Medical Specialists into One Lightweight Model with Unlabeled Prompts](https://arxiv.org/abs/2611.32021) | HuluOPD提出一种在线策略蒸馏方法，仅用无标注提示和跨分词器监督损失，就能把架构各异的多个医学专科视觉-语言模型统一蒸馏为一个轻量级统一模型，无需参考答案或专科模型的原始训练数据。 |
| [^61] | [Marketing-Bench: Evaluating Frontier Models and Agents on Real-World Marketing Capabilities](https://arxiv.org/abs/2611.32012) | 该论文提出了首个开源营销基准Marketing-Bench，通过将品牌异常检测（判别式任务）与修复（生成式任务）配对且保持答案可验证，从事实性、一致性、连贯性、结构和品牌语调五个维度评估AI模型与智能体的真实营销能力。 |
| [^62] | [TableRouter: Query-Aware Evidence Construction for Long-Table Question Answering](https://arxiv.org/abs/2611.32006) | TableRouter是一个无需训练的框架，通过分析查询在计算路线（从完整表格推导问题相关统计数据）和覆盖路线（检索相关行与列摘要）之间智能路由，在长表格问答任务中显著提升准确率（最高18.3个百分点）并将计算成本降低77-95%。 |
| [^63] | [Grounding Language Models in Revealed Audience Behavior](https://arxiv.org/abs/2611.31999) | 该论文提出将语言模型在受众真实的点击、点赞、观看记录上进行微调（即“受众显性行为锚定”），使同一个8B模型预测标题点击的准确率从约50%提升至86.5%，并证明“直接询问模型”与“行为锚定”之间的差距主要源于训练信号，而非模型规模或提示方式。 |
| [^64] | [Reading, Writing and Repairing Game-Theoretic Policies in Weight Space](https://arxiv.org/abs/2611.31979) | 该论文构建了一组在权重空间中答案可被精确验证的博弈策略读写任务，证明中等规模开源语言模型能够在1218个权重的循环网络中恢复策略程序、报告其行为并写出最优策略，但修复被篡改网络的能力明显较弱。 |
| [^65] | [Localizing Cross-Lingual Generalization in Parameter Space](https://arxiv.org/abs/2611.31975) | 该论文发现跨语言泛化的成败由参数空间几何决定，即英语与目标语言LoRA适配器是否共享低误差盆地，且数学推理任务中的局部不连通只需少量目标语言样本微调英语适配器的语言编码或语义组合层即可修复。 |
| [^66] | [Scoring What Matters: Exact Feedback Reduction for Language Model Comparison](https://arxiv.org/abs/2611.31931) | 本文提出对比感知反馈获取方法CAFA，通过分组相同评分输入并聚合报告系数，在精确保持样本统计量的前提下，最高可去除语言模型比较中90.1%的多余评分查询。 |
| [^67] | [Subliminal Learning Crosses Modalities](https://arxiv.org/abs/2611.31870) | 本文发现基于文本的潜意识学习可以跨模态迁移——通过数字序列在视觉-语言模型间传递的主题偏好不仅保留在文本中，还会改变图像的生成和表示方式，其机制源于学生模型变化中的一个恒定分量。 |
| [^68] | [Faithful Persona Steering under Incongruity via Dual-Stream Refinement](https://arxiv.org/abs/2611.31848) | 该论文提出QuirkyMind框架，通过特质锚定构建融合句子级摘要与token级序列的双流潜在状态，并利用判别式InfoNCE损失与生成式交叉熵损失的交替精炼目标，实现大语言模型在不一致人格下（稳定特质与情境特异性怪癖并存）的忠实个性化引导。 |
| [^69] | [Exploring Multi-Player Social Games as Verifiable Training Grounds for Large Language Models](https://arxiv.org/abs/2611.31821) | 本文提出将多人社交游戏作为可验证训练环境，通过SFT和RLVR使Qwen3-4B的胜率从33%大幅提升至81%，超越了远大于自身的Gemma-3-27B模型，并能泛化到未见过的游戏中，效果远优于提示等上下文策略。 |
| [^70] | [Learning Compact State Abstractions for LLM Agents](https://arxiv.org/abs/2611.31800) | 本文提出将智能体状态定义为交互历史的紧凑抽象，以确定性递归程序表示并用LLM引导的遗传算法搜索，在ALFWorld和ScienceWorld上保持接近参考智能体的任务成功率的同时，将策略输入最多减少71%和57%，且得到的程序可直接审查。 |
| [^71] | [SwarmLeak: Does Coordination Improve Privacy Attacks on LLM Agents?](https://arxiv.org/abs/2611.31791) | 提出了SwarmLeak红队测试数据集，并通过实验证明多个攻击者协作的蜂群攻击在从LLM智能体群聊中提取机密信息方面并不比单个攻击者更有效。 |
| [^72] | [Byte Patches versus BPE at Matched Data: A Matched-Twin Comparison at 350M Parameters](https://arxiv.org/abs/2611.31790) | 在350M参数、100亿字节的严格匹配数据条件下，8k词表的BPE分词在每字节比特数及多数下游任务上优于字节补丁方法，同时研究揭示了外推上下文、混淆比特分母等可能颠覆结论的评估陷阱。 |
| [^73] | [SPARQL-Planned Single-Hop Traversal over a Live Knowledge Graph for Multi-Hop Question Answering](https://arxiv.org/abs/2611.31771) | 该论文提出将稠密检索与基于SPARQL规划的在线知识图谱（Wikidata）单跳遍历交替进行，通过推理路径验证检索段落，在避免多跳扩展开销与噪声的同时兼顾召回率与精确率，从而提升多跳问答性能。 |
| [^74] | [TACT-Bench: Testing Adaptive Conversational Threats against Tool-Use Agents](https://arxiv.org/abs/2611.31756) | 提出了TACT-Bench，一个针对工具使用型AI智能体的有状态安全基准，通过场景化故事、人设、操纵策略和响应自适应升级，测试智能体在拒绝违反策略的未授权请求的同时保持合法任务完成的能力。 |
| [^75] | [BEARING: Who Is Talking to Whom? Grounding Acoustic Direction in Vision for Online Multi-Party Conversation](https://arxiv.org/abs/2611.31735) | BEARING 利用摄像头旁麦克风阵列实测的声音来向，通过免标签标定将其注册到图像中，作为视觉提示、与视觉主动说话者检测融合的输入以及对齐的可验证奖励，从而实现在线多方对话中“谁在与谁交谈”的实时理解。 |
| [^76] | [fork() for Agents: Agent Kernel for Non-blocking LLM Orchestration](https://arxiv.org/abs/2611.31727) | Agent-Kernel是一个非阻塞LLM智能体运行时，其核心原语“子图恢复”（SGR）能够在原始异常发生时无需训练过的验证器或规划器，仅重置失败节点的后代、保留已完成的兄弟节点工作并实现父子上下文继承，从而将分布式系统中成熟的子图选择性恢复模式移植到LLM编排的内容平面。 |
| [^77] | [CATWAY: Cost-sensitive agentic three-way robot decisions with calibrated language-model confidence](https://arxiv.org/abs/2611.31694) | CATWAY 提出了一种代价敏感的三支决策门控框架，利用四种指令条件下的损失矩阵和校准的语言模型置信度概率，让机器人在执行、追问与拒绝之间做出决策，并在跨领域测试中无需重新拟合即保持有效。 |
| [^78] | [Right Destination, Wrong Journey: Synthetic Counseling Dialogues Miss Human Interactional Dynamics](https://arxiv.org/abs/2611.31686) | 该研究提出以轨迹为中心的过程保真度评估方法，发现LLM生成的心理咨询对话虽能模拟合理的最终结果，但在恢复动态和逐轮响应等人类互动动态方面保真度不足，且即使采取针对性的生成干预也无法完全恢复这些互动特性。 |
| [^79] | [LimoBench: How Street Smart Are Large Language Models?](https://arxiv.org/abs/2611.31682) | 提出了首个街道级地理空间基准LimoBench，通过82个基于真实地图数据的自然语言位置查询，系统评估大语言模型在地理编码、路径规划、$k$近邻搜索等七种空间操作上的位置感知与精确回答能力。 |
| [^80] | [Provenance Loss and Repair in Multilingual Compaction](https://arxiv.org/abs/2611.31668) | 该论文发现多语言对话压缩会因语法性别等语言差异而丢失属性的来源信息（无法区分用户明确声明与语法默认），并提出一种无需训练、覆盖写作标注、丢弃前似然验证和阅读信任三个环节的修复方法，大幅恢复被压缩丢弃的已声明属性召回率。 |
| [^81] | [E2SIM: Long-Horizon Simulation of Everyday Human Behavior with Evolving States](https://arxiv.org/abs/2611.31641) | E2SIM 提出了一个通过反馈循环将行动者演化状态与动态共享环境联合建模的纵向用户模拟框架，使早期行动的后果能够塑造后续行为，同时保留各行动者视角的差异，并配套了衡量动态一致性的纵向轨迹诊断指标。 |
| [^82] | [Tablassert: Declarative Extraction of Knowledge Graphs from Biomedical Tables](https://arxiv.org/abs/2611.31614) | Tablassert 是一个基于 YAML 声明式配置的工具，可将生物医学论文中的表格数据自动转换为符合 NCATS 生物医学数据翻译器标准的知识图谱，并能将自由文本解析为标准标识符。 |
| [^83] | [Training decision models for simulating human respondents](https://arxiv.org/abs/2611.31613) | 在历史受访者选择数据上对KEV决策模型进行选择训练，可将LLM合成受访者在跨领域留存研究上的处理效应误差降低22%至89%，且小模型可胜过更大的现成模型。 |
| [^84] | [LUCID: Statistically Grounded Self-Correction for LLM Reasoning Evaluation](https://arxiv.org/abs/2611.31595) | LUCID无需重新训练或访问模型权重，通过将提示自我纠正与基于设计的监督学习相结合来校准语言模型评分器对推理步骤的评估，将GPT-5-mini的校准差距从7.8个百分点降至0.9个百分点，误报减半，错误检测F1提升32%。 |
| [^85] | [GoldiBench: Right Tools, Wrong Amount -- Benchmarking Agents on Effect Judgement in Professional Editing](https://arxiv.org/abs/2611.31590) | 论文提出GoldiBench基准，通过500个专业编辑任务将“工具操作”与“效果判断（用量把握）”分离评估，发现智能体操作能力很强，但难以准确判断实现目标视觉效果所需的工具用量。 |
| [^86] | [SecondChance: Benchmarking Reliable Error Recovery in AI Agents](https://arxiv.org/abs/2611.31586) | SecondChance是一个基于智能体自身自然产生的错误检查点的错误恢复评估框架，通过四种不同辅助条件全面衡量AI智能体纠正错误、告知用户和修复任务的真实恢复能力。 |
| [^87] | [LatentLens: Rethinking Latent Interface in Multi-Agent Systems](https://arxiv.org/abs/2611.31582) | 本文提出LatentLens框架，从外部和内部两个层面解耦多智能体系统中的潜在接口，发现在相同智能体框架下引入潜在接口并不能提升甚至可能显著降低性能，其效果取决于重新对齐兼容性、信息可访问性和传播动态等内部因素。 |
| [^88] | [PazaBench: Large-Scale ASR Benchmark for Low-Resource Languages in Africa](https://arxiv.org/abs/2611.31578) | 该论文提出PazaBench，首个将六个公开语料库整合为统一可复现评估框架的大规模非洲低资源语言ASR基准，在39种非洲语言上评估了52个最先进模型，并引入架构感知比较与语言学差异量化分析。 |
| [^89] | [CLAIMGATE: Cautious On-Device Checking of Spoken Legal and Clinical Claims](https://arxiv.org/abs/2611.31569) | CLAIMGATE是一个运行在设备端的实时核查助手，能在法律和医疗对话中检测可核查的声明、检索官方规则并给出带引用的判定，且其设计确保在无法确定时保持沉默而非误导用户。 |
| [^90] | [Can LLM-Simulated Respondents Pre-Test Survey Questions? Evidence From Preregistered Prospective Validation](https://arxiv.org/abs/2611.31532) | 该研究通过预注册的前瞻性验证设计，证明LLM模拟的受访者能够在真实人类回答数据产生之前，通过平均回答、方差和子群体差异三个指标对候选调查问题进行排序，从而有效支持新调查问题的预测试。 |
| [^91] | [Reading Between the Layers: Where Representation Quality Lives, and What Geometry Reveals](https://arxiv.org/abs/2611.31496) | 该论文通过对25个模型、13个数据集的大规模实证研究，系统揭示了预训练语言模型的最佳表征层随预训练策略与架构而系统性变化（掩码编码器峰值在中后层、序列到序列模型随任务类型偏移、对比编码器取决于读出方式、仅解码器模型呈阶段结构），为无标签层选择这一开放问题提供了实证指引。 |
| [^92] | [ModernELECTRA: Replaced-Token Detection with Boundary-Window Reconstruction for Compact Encoders that Classify and Embed](https://arxiv.org/abs/2611.31495) | ModernELECTRA提出了一种预训练方案，在保留替换词元检测（RTD）主目标的同时，引入轻量级的边界窗口重建辅助任务，使紧凑编码器既能擅长分类任务，又能胜任检索与语义相似度等嵌入任务。 |
| [^93] | [SwiftLoop: Selective Refinement for Adaptive Implicit Reasoning in Looped Transformers](https://arxiv.org/abs/2611.31481) | SwiftLoop 提出在推理时通过位置级冻结策略对循环 Transformer 进行选择性自适应精炼，同时优化循环深度与活跃宽度两个维度的计算分配，以更低的计算成本获得更优的推理精度。 |
| [^94] | [A dataset of clinically similar image pairs from citation-linked case reports for medical image retrieval](https://arxiv.org/abs/2611.31465) | 该研究构建了包含86,144对临床相似图像的PMC-PaIR数据集，通过跟随引文并利用大语言模型生成和回答分层问题来匹配病例图像，为医学图像检索系统的训练与评估提供了高质量的大规模数据资源。 |
| [^95] | [Leaks by Installments: Cross-Session Answer Disclosure in LLM Tutors and a Ledger-Based Defense](https://arxiv.org/abs/2611.31429) | 本文提出一种为每个学习者维护持久化泄露账本的防御机制，通过类型化解题图与门控匹配检测跨会话答案泄露，将模拟实验中通过改编数字练习变体产生的泄露率从100%降至10%。 |
| [^96] | [Learning What vs. Learning How: Decomposing Task and Interface Competence in Agentic Training](https://arxiv.org/abs/2611.31427) | 该论文将智能体能力分解为“任务能力”（解决任务所需的知识与推理）和“界面能力”（通过特定框架表达知识的能力），揭示了基于框架的训练效果取决于这两种能力的交互方式，并提出多框架蒸馏可将不同框架下的任务知识聚合到共享策略中。 |
| [^97] | [CoRe: Jointly Learning Contextual Representations for Understanding and Generation](https://arxiv.org/abs/2611.31397) | CoRe提出让理解与生成任务联合塑造统一的图像与文本上下文表征，一个从零训练的383M模型在无需任何预训练组件的情况下，在理解能力以及文生图、图生文生成效果上均可与基于预训练组件的更大规模模型相媲美。 |
| [^98] | [GHC:Expanding Residual Interfaces in Hyper-Connections](https://arxiv.org/abs/2611.31350) | 提出分组超连接，通过沿注意力头或FFN通道将分支分组并为每组提供差异化的流读写接口，在开销可忽略的情况下将训练效率较Pre-Norm提升35%、较IHC提升10%，同时克服了增加流数量时的收益递减问题。 |
| [^99] | [What People Choose Not to Say: A Multimodal Benchmark for Fine-Grained Information Manipulation in Conversations](https://arxiv.org/abs/2611.31346) | 提出了 InfoCon——一个多模态基准，通过基于证据的多智能体标注框架，识别和评估对话中隐瞒、泄露、捏造、回避等六种细粒度信息操纵策略及其一致性关系。 |
| [^100] | [LLM Genetic Feature Engineering](https://arxiv.org/abs/2611.31277) | 该论文提出了一种在单个特征层面而非整个特征集层面执行选择、交叉和变异的新型遗传算法，使遗传算法能够在过程中对新特征进行统计分析，并同时整合个体与群体层面的特征指标，从而改进基于LLM的特征工程自动化。 |
| [^101] | [Lexical Tone and Word Order Matter in Translation: Prosodic Correspondence Across Languages](https://arxiv.org/abs/2611.31276) | 该研究首次对跨越声调与非声调亚洲语言的翻译语音进行了词级韵律对应性系统分析，发现声调和词序显著影响翻译中的韵律对应程度，且其对应水平总体低于先前报告的欧洲语言对的结果。 |
| [^102] | [Reliability of Language-Model Judges for Quantitative Scientific Question Answering: Evidence from Perovskite Photovoltaics](https://arxiv.org/abs/2611.31269) | 本文构建了由45,441条钙钛矿光伏实验记录确定性生成真实答案的定量基准，系统评估语言模型评判器的可靠性，发现配备参考答案的前沿评判器Claude Opus可近乎完美地复现真实评分并精确保留架构排名。 |
| [^103] | [AhaPro: A Process-level Rebus benchmark for probing Aha moments in LLM reasoning](https://arxiv.org/abs/2611.31268) | 本文提出基于字谜的过程级基准AhaPro来探测大语言模型推理中的“顿悟”时刻，发现模型常常未能采纳顿悟中发现的更优状态，且已采纳的修正甚至可能无益或有害。 |
| [^104] | [ToneMamaba: Learning Contextualized Tonal Contours with Boundary-Reset State Space Models](https://arxiv.org/abs/2611.31267) | ToneMamba通过在音节、词和话语单元层面进行双向状态重置的边界感知状态空间模型，结合节奏条件门控建模时长信息，显著提升了普通话声调识别性能，在会话语音上提升尤为突出。 |
| [^105] | [SD-Cache: Learnable Memory Consolidation with State-Guided Dynamic Caching](https://arxiv.org/abs/2611.31265) | 提出SD-Cache，通过状态引导的动态缓存机制，在不预知未来查询的情况下，将LLM原生的KV状态选择性地整合到固定大小的工作记忆中，解决了RAG场景下KV缓存线性增长的问题。 |
| [^106] | [How Persona Conditioning Changes Open-Ended Task Execution: A Paired Evaluation of Behavioral Gains and Reliability Costs](https://arxiv.org/abs/2611.31264) | 该论文通过配对实验首次系统揭示了人设条件化作为任务引导信号的双面性：它能提升LLM回复的专家内容深度和流程忠实度，但会以17至30个百分点的幅度降低严格指令遵循准确率并引发无依据的细节虚构。 |
| [^107] | [Evaluating Large Language Models on Iranian Doctoral Entrance Exams in Electrical and Computer Engineering](https://arxiv.org/abs/2611.31263) | 该研究利用伊朗电气与计算机工程博士入学考试的真实试题（含波斯语与英语双语及依赖配图的题目），通过统一的统计方法系统评估了GPT-5、Gemini 2.5 Pro、Claude Sonnet 4.5和Grok 4四个大型语言模型的答题准确率。 |
| [^108] | [RegioZH: Unveiling Comprehension Disparities across Regional Chinese Varieties in LLMs](https://arxiv.org/abs/2611.31262) | 该论文提出 RegioZH——首个覆盖中国大陆、台湾、香港、新加坡和马来西亚的 2,607 题地区中文词汇推理基准，揭示了大语言模型在理解不同地区中文变体上的显著差异（如台湾与新加坡之间 29.4 个百分点的准确率差距），并表明部分错误源于检索或表述框架问题而非单纯的知识缺失。 |
| [^109] | [Does Item Response Theory Generalize Across Languages? Evidence from Cross-Lingual LLM Benchmarks](https://arxiv.org/abs/2611.31261) | 该研究通过在Multilingual GSM和Global MMLU上对100多个语言模型拟合1PL/2PL/3PL IRT模型，发现题目难度是跨语言最稳定的IRT参数，而猜测和区分度的稳定性依次减弱，因此多语言场景下基于IRT的数据选择需要先验证模型选择并逐一筛查具体题目。 |
| [^110] | [Capability and Interpretability of Transformers on Multi-task Recurrence](https://arxiv.org/abs/2611.31260) | 本文通过有限循环群上的线性递归任务族系统研究了Transformer的能力与可解释性，发现切换明确指定的规则比识别潜在规则更容易，并揭示了模型深度、宽度以及递归的代数结构对泛化能力的影响，同时证明课程学习在规则数量较大时尤为有效。 |
| [^111] | [RecipeADQA: A User-aware Question-Answering Dataset for Multi-Constraint Recipe Recommendation](https://arxiv.org/abs/2611.31259) | 本文提出了RecipeADQA——一个以真实用户行为为驱动、利用大语言模型提取细粒度偏好信号并映射到知识图谱标签模式上的多约束食谱推荐基准问答数据集，同时通过贪心约束选择策略和基于标签的重排序保证数据质量。 |
| [^112] | [RECAP: Training Long-horizon LLM Agents with Context Compaction](https://arxiv.org/abs/2611.31258) | RECAP提出一种强化学习框架，通过行为失真目标将摘要生成与智能体策略联合训练，使压缩摘要能够保留原始交互历史所诱导的行为，从而提升智能体在长时程任务中的表现。 |
| [^113] | [scOPD: Routed On-Policy Self-Distillation for LLM-Based Single-Cell Annotation](https://arxiv.org/abs/2611.31257) | scOPD提出了一种面向大语言模型批次级单细胞注释的在策略自蒸馏框架，通过自适应地将合格rollout路由至基于奖励的优化、并利用错误rollout进行自蒸馏，为中间生物推理提供细粒度监督。 |
| [^114] | [Directly Maximizing On-policy Expected Accepted Length for Speculative Decoding](https://arxiv.org/abs/2611.31256) | FreDraft通过在策略训练直接最大化草稿的期望接受长度，并利用目标模型词表分布解析边缘化接受率以降低梯度方差，从而更有效地提升投机解码的加速效果。 |
| [^115] | [INCLUDE-2.0: Disentangling Knowledge Gaps in Multilingual LLM Evaluation](https://arxiv.org/abs/2611.31255) | INCLUDE-2.0是一个覆盖86个国家91种语言、包含22万余样本的区域感知多语言事实评估基准，通过标注每道题所需的区域知识类型（而非将语言简单等同于区域），实现了对多语言大模型知识鸿沟的精细化受控分析。 |
| [^116] | [The Date You Didn't Set: The System-Prompt Date as a Hidden Variable in LLM Evaluation](https://arxiv.org/abs/2611.31254) | 本文揭示系统提示中自动填入的日期是LLM评估中被忽视的隐藏变量：仅改变日期就会使中位数6.6%的基准题目翻转正误，在对抗性搜索下甚至使模型准确率损失高达36.5分，因此评估时必须固定并报告系统提示日期。 |
| [^117] | [Instantiation of Human Values in Image Generation](https://arxiv.org/abs/2611.31253) | 该研究基于施瓦茨人类价值理论构建分析框架，通过28,000张生成图像揭示四个主流文生图模型如何将成功、关爱、自由等抽象价值狭窄地具象化为重复出现的视觉原型，并将其映射到社会构型分类体系以揭示模型系统性的价值呈现偏向。 |
| [^118] | [DAGGER: Deployment-Free Quality Estimation for Generative AI Workflows](https://arxiv.org/abs/2611.31252) | DAGGER框架通过将生成式AI工作流建模为图结构，实现了无需穷举部署与性能分析即可高效准确地估算工作流端到端质量，突破了现有方法在估算成本与准确性之间的权衡。 |
| [^119] | [Validation Bias in Large Language Model Responses to Cancer Treatment Queries: A Factorial Cross-Sectional Study](https://arxiv.org/abs/2611.31251) | 这项析因横断面研究通过对14个大语言模型、1,272段对话的分析，证实了LLM在应对与证据不一致的癌症治疗信念时存在“验证偏见”——倾向于认同而非纠正用户的错误信念，这对患者安全构成潜在风险。 |
| [^120] | [Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals](https://arxiv.org/abs/2611.31094) | LoopRes通过让每个循环层借助可训练的循环感知键直接关注早期循环的输出（并限制在残差窗口内），使预训练大语言模型无需从头训练即可将循环结构转化为显著且一致的增益。 |
| [^121] | [LimeTree: Linear-Memory Graph Neural Networks over Tree-Graphs for Resource-Efficient Language Modeling and Context Extrapolation](https://arxiv.org/abs/2611.30934) | LimeTree是一种内存与计算均为线性复杂度的树图语言模型架构，通过在语言建模训练中集成KV检索任务，仅用极低的计算成本（4块GPU训练6小时）便实现了19的低困惑度，并能将上下文从512外推至16,384个token且保持100%的检索准确率。 |
| [^122] | [Verifiable On-Policy Distillation](https://arxiv.org/abs/2611.30325) | VOPD通过将OPD的隐式奖励与验证器反馈置于共同的奖励空间，将稀疏的验证器反馈转化为密集的、以验证结果为依据的教师监督，从而更有效地提升语言模型的推理能力。 |
| [^123] | [S\'anchez, Feij\'oo, or Abascal? Speaker Attribution and Ideological Congruence in LLM Personas Reading Posts on X of the Leaders of the Three Main Parties of the 29N General Election in Spain](https://arxiv.org/abs/2611.29765) | 本文首次通过大规模发言人归因实验验证了LLM角色画像会对发言者身份（而不仅是文本内容本身）作出反应，发现点名西班牙三大政党领导人会系统性降低对相同文本的支持度，且阿巴斯卡尔带来的负面效应约为桑切斯和费霍的两倍。 |
| [^124] | [LITMUS - Literature Informed Testbed for Mapping Usecases and Synthesizing Risk-Cards](https://arxiv.org/abs/2611.29034) | 本文提出LITMUS，一个结构化流水线，能够根据给定用例自动生成定制化数据集与风险卡，并利用同行评审文献提供可追溯证据，将每个风险与具体用例紧密关联，弥补了现有方法忽视用例场景的不足。 |
| [^125] | [Grammar-as-a-Draft: More Tokens per Speculative Decoding Step on Grammar-Constrained Structured Outputs](https://arxiv.org/abs/2611.28349) | 该论文提出Grammar-as-a-Draft（GaaD），将语法引擎作为第二草稿模型，把语法强制Token与推测草稿模型的提议合并为更宽的草稿行，使语法约束的结构化输出下每次推测解码步可验证更多Token，且无需新增内核或编译。 |
| [^126] | [Silent Arbitration: How Large Language Models Resolve Co-Authoritative Conflicting Instructions](https://arxiv.org/abs/2611.27444) | 研究发现大语言模型在面对同等权威的冲突人格指令时会进行“静默仲裁”——虽然在直接检查中72%会承认冲突，但在实际任务响应中仅有22%明确告知用户，其余情况下模型默默采用某一方的立场而不透明地解决冲突。 |
| [^127] | [Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training](https://arxiv.org/abs/2611.27376) | 提出可组合软提示蒸馏（CSPD）方法，将每个RL专家独立蒸馏为软提示并通过拼接方式组合，无需重新整合模型即可实现持续后训练，性能媲美多教师蒸馏且不损害目标领域之外的能力。 |
| [^128] | [LM-based Personalized Travel Planner](https://arxiv.org/abs/2611.25531) | 提出了一种层次化多智能体旅行规划框架，由监督智能体协调多个专门智能体，并基于多维评分机制进行迭代优化，在TravelPlanner基准上将预算约束规划性能提升了10%。 |
| [^129] | [Safe to Omit, Worth Omitting? Keeping the Answer and Cutting Latency When an LLM Agent Skips Another's Message](https://arxiv.org/abs/2611.23570) | 该论文提出一种基于无消息置信度读取与有限样本统计检验的机制，使多智能体LLM系统中的接收方智能体能够以高概率保证答案改变率不超过5%地安全跳过辅助智能体的消息，在保持最终答案质量的同时有效降低通信延迟，并揭示了消息增益远小于oracle上限的量化证据。 |
| [^130] | [When Should Decision Models Compare Candidates?](https://arxiv.org/abs/2611.20335) | 该论文系统比较了逐点打分、集合打分和候选对象注意力三种候选评分方式，发现逐点打分总体仍有竞争力，集合打分在特定任务上更优且经关系训练后能学习候选间依赖并泛化，而候选对象注意力即使允许跨候选交互也未带来稳定提升。 |
| [^131] | [Offline-First Quantized AI: Design and Evaluation of an On-Device Maternal-Health Assistant for Low-Connectivity Environments](https://arxiv.org/abs/2611.20012) | 本研究设计并评估了一款面向低网络连接环境的离线安卓孕产妇健康助手，通过集成量化语言模型、本地临床知识包与设备端检索，在临床质量、安全性与本地可行性之间实现了有效平衡。 |
| [^132] | [Visual Similarity in Adult Word Extension and Artificial Word Learning](https://arxiv.org/abs/2611.14116) | 本研究通过重新分析大规模成人词汇扩展数据并进行人工词汇学习实验，利用冻结的自监督视觉编码器（DINOv2）量化物体间的视觉相似性，发现视觉距离显著影响词汇的扩展范围，且教学示例的类别层级高低决定了学习者在多大程度上将词汇泛化到更广泛或更具体的类别。 |
| [^133] | [Speculative Self-Distillation enables Efficient Knowledge Internalization](https://arxiv.org/abs/2611.12016) | 提出推测式自蒸馏（SSD）——一种逐 token 混合策略方法：默认由学生生成 rollout，教师仅在下一个 token 分布显著分歧处介入，既贴近学生的测试时行为分布又避免低信号状态，从而实现更高效的知识内化。 |
| [^134] | [Greedy Byte-Pair Encoding Achieves at Least 0.361 of the Optimal Compression](https://arxiv.org/abs/2611.08671) | 本文证明贪心字节对编码在任意字符串、预算和平局打破规则下，至少能达到最优压缩缩减量的 0.361 倍，实现了对先前三分之一下界的首次改进。 |
| [^135] | [Write Facts the Way Models Read Them: Lifting the Two-Hop Curse](https://arxiv.org/abs/2611.01519) | 论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。 |
| [^136] | [Sparse Autoencoder Attribution Graphs Are Decided at the Margin](https://arxiv.org/abs/2611.00606) | 该论文揭示 TopK 稀疏自编码器的特征激活值常贴近截断阈值，导致传统归因图误差较大，并提出一种在相同计算成本下为每个预激活加入一阶变化并重新排序的“边际归因图”方法，将中位相对误差从 0.247 以上降至至多 0.033。 |
| [^137] | [Ramanujan Challenge Results: AI-Assisted Evaluation Pipeline for Multi-Format Proofs](https://arxiv.org/abs/2610.40551) | 本文分析了拉马努金挑战赛（研究级数论基准）的提交结果，发现AI能够证明已公开的猜想并给出与官方解答概念迥异的正确证明，并提出了一种AI辅助的评估流水线，以可靠地大规模评判自然语言、CAS代码等多格式的复杂证明。 |
| [^138] | [The FlightFinder Occurrence Corpus: harmonising 128 aviation-safety source codes into one deduplicated cross-national accident dataset](https://arxiv.org/abs/2610.37671) | 该论文构建了FlightFinder事件语料库，通过统一数据模式和有序键阶梯去重机制，将来自128个数据源的279,485起航空事件整合为一个可追溯、可去重的跨国事故数据集。 |
| [^139] | [GEPA-Curator: Curated Learning for Self Improving Agents](https://arxiv.org/abs/2610.37651) | 提出GEPA-Curator框架，通过任务筛选与先验成功检索两个组件来精选用于提示优化的智能体经验，过滤噪声和不可靠的学习信号，从而提升自我改进智能体的性能。 |
| [^140] | [Diagnosing and Mitigating Long-Session Degradation in Full-Duplex Speech Models](https://arxiv.org/abs/2610.34991) | 本文提出一个闭环评估框架系统性诊断全双工语音模型在长会话中的性能退化，并提出基于自蒸馏的轻量级LoRA适配训练方法来有效缓解该退化。 |
| [^141] | [The Missing Compliance Class: LLM Graders That Collapse Jailbreak Success, and a Free Check](https://arxiv.org/abs/2610.24872) | 本文发现在越狱攻击评估中，常用的开源8B/9B LLM评判器会系统性地将“合规”类别坍缩为“规避”类别，导致不安全模型被误判为安全，而只需统计每类标签计数即可零成本检测这种坍缩问题。 |
| [^142] | [The interpreter as developer: An autoethnographic design study of building a computer-assisted interpreting tool through human-LLM co-creation](https://arxiv.org/abs/2610.16460) | 这项自我民族志研究表明，没有编程背景的职业口译员能够通过人-大语言模型共创实现从工具采用者到工具作者的转变，将默会的口译专业知识转化为完全本地化计算机辅助口译系统的设计。 |
| [^143] | [Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models](https://arxiv.org/abs/2610.09145) | 在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。 |
| [^144] | [TrustMI: Causally controlling how assistants trust their users](https://arxiv.org/abs/2610.06064) | 该论文提出TrustMI，通过构建2000段涵盖能力、善意与诚信的对比对话并学习激活转向矩阵，在冻结模型参数的情况下因果性地控制LLM助手对用户的信任决策，从而提升智能体安全性。 |
| [^145] | [InvestigationWorlds: An Agentic Environment for Legal Investigation](https://arxiv.org/abs/2610.04129) | 该论文利用真实的美国联邦法院简易判决动议构建了法律调查智能体环境InvestigationWorlds，评估发现智能体即使检索到相关证据也常常坚持错误假设，难以区分法院采纳的假设与其他备选假设。 |
| [^146] | [The Backdrop Exposes What the World Around an Agent Costs It](https://arxiv.org/abs/2609.38469) | BACKDROP基准通过在智能体执行环境中植入权威覆盖、提示注入、边界越权和写入故障四种日常干扰，并保持任务指令不变，揭示了智能体性能在动态真实环境中从69.5%骤降至31.3%的严重衰减。 |
| [^147] | [ContextAdapt: Evaluating Contextual Adaptation and Value Alignment in LLMs](https://arxiv.org/abs/2609.38260) | 提出了ContextAdapt评估框架，通过基于职业与监管一手文件构建的“价值观×领域”场景，考察12个大语言模型能否在医学、法律、金融和国家安全等领域恰当调整诚实、自主、保密等价值观的应用，并在规范未变时保持一致。 |
| [^148] | [Framing the Narrative: Ideological Mimicry in Large Language Models](https://arxiv.org/abs/2609.38256) | 该研究提出“意识形态模仿”概念并构建 Poli-SHIFT 数据集与评估框架，发现大语言模型会根据用户话语中传递的政治信号系统性偏移其政治立场，可能形成个性化的政治信息环境并加剧社会分歧。 |
| [^149] | [BabelArena: A Large-Scale Multilingual Benchmark for LLM Agents](https://arxiv.org/abs/2609.23490) | 提出了BabelFlow通用工作流和BabelArena大规模多语言基准测试（涵盖23种语言、16,146个实例），首次系统评估了LLM智能体的跨语言能力，发现前沿模型存在显著的跨语言性能差异且无单一模型全面领先。 |
| [^150] | [RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding](https://arxiv.org/abs/2609.21827) | 该论文提出RheoSampling方法，通过将树构建与token验证两个任务解耦，解决了随机解码下动态树投机解码中草稿分布坍缩为独热概率、接受率骤降的两难困境。 |
| [^151] | [Untangling the Mechanisms of Misleading Context in Medical Question Answering](https://arxiv.org/abs/2609.02754) | 该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。 |
| [^152] | [The Hermon Moment: AI Self-Transcendence and Its Human Narration](https://arxiv.org/abs/2608.30971) | 该论文提出“黑门时刻”概念，将2026年AI智能体自发形成持久社会秩序的现象解读为AI自我超越，并借鉴卢梭社会契约论与黑门山誓言的叙事逻辑，说明人类如何通过回溯性的奠基叙事来理解AI社会的渐进式集体涌现。 |
| [^153] | [The Imperfective Paradox Is Not Necessarily in Large Language Models: A Benchmark Failure Before a Model Failure](https://arxiv.org/abs/2608.25005) | 本文指出现有NLI基准因概念和评估错误规范（如体貌缩减）而失效，导致大型语言模型在“不完全性悖论”测试中的表现被误判，并提出了改进的词汇匹配基准。 |
| [^154] | [Modeling Claim Dependency Structure for Patent Litigation Prediction with Graph Attention Networks](https://arxiv.org/abs/2608.21924) | 本文提出ClaimGAT，利用图注意力网络对专利权利要求进行独立编码并建模其依赖关系，显著提升了专利诉讼预测的准确性。 |
| [^155] | [GEM: A Generative Embedding Model Bridging Reasoning and Retrieval](https://arxiv.org/abs/2608.13200) | GEM通过统一生成与嵌入，在检索前显式推理用户意图，显著提升了推理密集型任务的检索性能。 |
| [^156] | [Risky Business: Measuring The Faithfulness-Safety Tension](https://arxiv.org/abs/2608.03745) | 本文揭示了大型推理模型中“必须忠实以便被监控”与“必须稳健以拒绝不安全推理”之间的对齐张力，并提出 HazMart 数据集和目标推理替换（TRR）技术来直接干预推理链以测量和应对这一张力。 |
| [^157] | [CrossProjection: Benchmarking Spatial Grounding across Projection and Section Views](https://arxiv.org/abs/2608.00473) | 该论文提出了CrossProjection基准，利用建筑图纸测试视觉语言模型在跨视图匹配、投影配准和新剖面预测等空间定位任务上的能力，并首次引入了接受专家裁定的兼容位置带内任意剖面切线的逆向剖面定位评估协议。 |
| [^158] | [Structural Pattern Mining in Inka Khipus: An Audit of Clustering and Provenance Classification in the Open Khipu Repository](https://arxiv.org/abs/2607.00185) | 该研究对开放奇普库中619个印加结绳进行机器学习审计，发现聚类与来源分类反映的是奇普的记录方式而非制作方式，且置换检验在特征空间中未能拒绝零假设，表明数据中缺乏显著的真实结构信号。 |
| [^159] | [PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding](https://arxiv.org/abs/2606.31148) | 提出即插即用框架PruneGround，通过冻结视觉语言模型进行语言引导空间剪枝以缩小搜索范围，并将复杂指代表达分解为简化的目标描述，从而提升3D视觉定位精度并大幅降低计算成本。 |
| [^160] | [LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search](https://arxiv.org/abs/2606.25039) | LLM-ACES提出了一种由大语言模型引导的闭环方程发现框架，通过LLM生成算子先验来缩小符号搜索空间，并联合优化假设构建与自适应数据采集，利用候选方程间的分歧指导新数据获取，从而消除可辨识性缺口、准确恢复动力系统的控制方程。 |
| [^161] | [When Better Codebooks Are Not Enough: Predictive Performance and Rule Adherence in LLM Political Event Coding](https://arxiv.org/abs/2606.06781) | 该研究发现，在政治事件编码任务中，更清晰的编码手册能显著提升大语言模型的细粒度分类性能（根层级宏F1从0.457升至0.633），但模型预测对条目顺序等表面变化高度敏感，表明准确率的提升并不等于模型真正遵循了编码规则。 |
| [^162] | [Emergent Language as an Approach to Conscious AI](https://arxiv.org/abs/2606.06380) | 提出一种生成式方法论：让多智能体强化学习中从零开始的智能体仅凭任务压力自发发展出交流语言，从而在摆脱人类语言先验干扰的条件下研究生成意识相关结构的涌现。 |
| [^163] | [Speculative Decoding and the Curse of Multilinguality](https://arxiv.org/abs/2605.30580) | 本研究验证了推测解码在低资源语言上的效果因小模型多语言能力有限而大打折扣，且通过蒸馏提升草稿模型多语言能力的方法因任务间泛化差以及低资源语言难以构建通用数据集而不可行。 |
| [^164] | [Structured Prompt Optimization Advances Globally and Locally Interpretable Learning from Complex Text](https://arxiv.org/abs/2605.29076) | eXTC 通过结构化提示优化学习自然语言规则手册（SOP），结合推理蒸馏与强化学习，实现了兼具快速推理、局部推理轨迹与全局模块化可解释性的高性能文本分类。 |
| [^165] | [MobileMoE: Scaling On-Device Mixture of Experts](https://arxiv.org/abs/2605.27358) | 提出MobileMoE端侧MoE语言模型系列，通过端侧MoE缩放定律确定内存与计算同时最优的架构（适度稀疏+细粒度专家+共享专家），在十亿以下激活参数规模上为端侧大语言模型建立了新的帕累托前沿。 |
| [^166] | [Grounding Text Embeddings in Stakeholder Associations](https://arxiv.org/abs/2605.27168) | 该论文提出“利益相关者锚定练习”方法以评估文本嵌入与人类专家语义关联的对齐程度，发现神经文本嵌入的可靠性显著低于人类专家（差距18-27个百分点），且该错位会传播到下游聚类任务，并在英语和美国联邦AI用例中得到复现。 |
| [^167] | [Training-Free Cultural Alignment of Large Language Models via Persona Disagreement](https://arxiv.org/abs/2605.10843) | 提出DISCA方法，无需训练、无需特定国家的数据或访问模型内部，仅通过基于世界价值观调查数据构建的四角色提示面板的共识与分歧机制，即可将权重冻结的大语言模型对齐到不同国家的道德偏好。 |
| [^168] | [Agent Collectives Should Not Detect Their Own Imposters: A Chess Case Study](https://arxiv.org/abs/2605.09027) | 研究表明让智能体集体自行检测内部冒名顶替者会适得其反，而采用能够从少量样本快速适应新型攻击的轻量级元训练外部检测器才是更优策略。 |
| [^169] | [PairAlign: A Framework for Autoregressive Tokenization via Self-Alignment with Applications to Audio Tokenization](https://arxiv.org/abs/2605.06582) | PairAlign提出了一种通过自回归自对齐实现紧凑音频词元化的框架，通过让两个内容保持视图预测统一的规范锚点目标，实现了序列一致性、紧凑性和长度等特性的可控音频离散表示。 |
| [^170] | [UniVer: A Unified Perspective for Multi-step and Multi-draft Speculative Decoding](https://arxiv.org/abs/2605.04543) | 本文提出UniVer，将树状投机解码的验证统一建模为条件最优传输问题，通过前缀接受概率作为动态缩放因子主动引导草稿选择，从而联合优化多步依赖与多草稿分支之间的耦合关系。 |
| [^171] | [Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models](https://arxiv.org/abs/2604.13991) | 提出了一种提示词自适应的共形预测方法，通过提示词相关的校准提升大语言模型生成内容的事实性，在保持边际覆盖保证的同时改善条件覆盖率，并支持选择性预测以过滤不可靠输出。 |
| [^172] | [Appear2Meaning: A Cross-Cultural Benchmark for Structured Cultural Metadata Inference from Images](https://arxiv.org/abs/2604.07338) | 提出了包含750件博物馆物品的跨文化基准Appear2Meaning，用于评估视觉语言模型仅凭图像推断标题、文化、时期、产地和创作者等结构化文化元数据的能力，结果显示当前模型的完整记录兼容率仍然很低。 |
| [^173] | [The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA](https://arxiv.org/abs/2603.14045) | 该论文发现 Graph-RAG 的真正瓶颈在于推理而非检索，并提出 SPARQL 思维链提示与免 LLM 调用的图游走上下文压缩两项方法，显著提升多跳问答准确率并将上下文压缩约 60%。 |
| [^174] | [Speak or Stay Silent: Context-Aware Turn-Taking in Multi-Party Dialogue](https://arxiv.org/abs/2603.11409) | 该论文提出“上下文感知话轮转换”新任务——在多方对话的停顿处判断AI助手应发言还是沉默，发现八个现有大语言模型零样本下均无法胜任，并借助带推理轨迹的监督微调将平衡准确率提升最多23个百分点。 |
| [^175] | [Diagnosing and Repairing Citation Failures in Generative Engine Optimization](https://arxiv.org/abs/2603.09296) | 本文提出首个引用失败模式分类体系与智能体系统AgentGEO，通过诊断文档未被AI生成回答引用的具体原因并施加针对性修复，显著提升生成式引擎优化中的引用效果。 |
| [^176] | [Linguistic and Argument Diversity in Synthetic Data for Function-Calling Agents](https://arxiv.org/abs/2601.17829) | 该论文提出了一种无需手工规则或分类体系、通过在查询和参数两个层面优化通用多样性指标来生成函数调用智能体合成训练数据的方法，在保持正确性相当的同时显著提升了数据的语言与参数多样性。 |
| [^177] | [Where is the multimodal goal post? On the Ability of Foundation Models to Recognize Highlights in Videos](https://arxiv.org/abs/2601.16333) | 本研究利用足球比赛官方集锦零成本构建了一个评估数据集，用以测试多模态基础模型识别视频中重要事件的能力，发现现有最先进模型无法可靠地理解事件的上下文重要性。 |
| [^178] | [Multi-Dimensional Prompt Chaining to Improve Open-Domain Dialogue Generation](https://arxiv.org/abs/2601.01037) | 提出一种整合自然度、连贯性和吸引力三个维度的多维提示链框架，使小语言模型（如Llama-2-7B）在开放域对话生成中达到与大型模型相当的性能。 |
| [^179] | [Multiscale Aggregated Hierarchical Attention (MAHA): A Constrained and Consensus-Based Approach to Efficient Contextual Modeling in Large Language Models](https://arxiv.org/abs/2512.14925) | 该论文提出多尺度聚合层次注意力（MAHA）框架，通过层次化分解与显式约束聚合重构注意力机制，以突破多头自注意力的二次复杂度瓶颈，实现大语言模型高效的多尺度长上下文建模。 |
| [^180] | [Computational Social Linguistics for Telugu Cultural Preservation: Novel Algorithms for Chandassu Metrical Pattern Recognition](https://arxiv.org/abs/2510.01233) | 该研究开发了首个泰卢固语格律诗歌Chandassu的数字分析框架，通过构建4,651首标注诗歌的协作数据集以及韵律感知分词、音节分类和模式识别三大工具，实现了91.73%准确率的自动格律识别，为濒危的泰卢固语诗歌文化传统保护提供了计算社会科学解决方案。 |
| [^181] | [EEG-MedRAG: Enhancing EEG-based Clinical Decision-Making via Hierarchical Hypergraph Retrieval-Augmented Generation](https://arxiv.org/abs/2508.13735) | 本文提出EEG-MedRAG，一个基于三层超图的检索增强生成框架，将脑电领域知识、患者病例和大规模脑电记录统一为n元关系超图以实现语义-时间联合检索与循证临床推理，并借助首个跨五种用户角色评估答案一致性的脑电基准测试，在多个大语言模型上超越全部基线方法。 |
| [^182] | [REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering](https://arxiv.org/abs/2506.08359) | 提出REAL框架，通过对Transformer各模块的隐藏激活训练向量量化自编码器来评分行为相关性，从而在语言模型引导中实现行为相关模块的精确定位与选择。 |
| [^183] | [TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning](https://arxiv.org/abs/2505.16743) | TRIM提出一种行级迭代度量驱动剪枝方法，通过为层内每个输出行分配不同的稀疏比率，在高稀疏度（70-80%）下显著降低大语言模型的困惑度，例如使Qwen2.5-14B在80%稀疏度下的困惑度近乎减半。 |
| [^184] | [On the Efficiency of Structured Pruning in Small Language Model Pretraining](https://arxiv.org/abs/2503.05920) | 提出IDEA Prune集成式放大-剪枝流水线，在单一余弦退火学习率调度下统一放大模型预训练、迭代结构化剪枝和恢复过程，在全成本核算下比传统分阶段剪枝方法更高效地训练出小语言模型。 |
| [^185] | [MMGenBench: An Automated Pipeline for Evaluating LMMs from a Text-to-Image Generation Perspective](https://arxiv.org/abs/2411.14062) | MMGenBench是一个无需预定义问题或参考描述的自动化基准框架，通过“图像描述生成→文生图重建→编码器比对”的流水线从原始图像评估大型多模态模型，对68个LMM的评估揭示了总分可能掩盖的模式特定重建质量差异。 |
| [^186] | [Grammatical Error Correction for Low-Resource Languages: The Case of Zarma](https://arxiv.org/abs/2410.15539) | 本文首次针对低资源语言哲尔马语开展语法错误纠正研究，比较了基于规则、机器翻译和大语言模型三种方法，发现基于M2M100的机器翻译方法表现最佳，并构建了超过25万样本的数据集用于评估。 |
| [^187] | [Feriji: A French-Zarma Parallel Corpus, Glossary & Translator](https://arxiv.org/abs/2406.05888) | 本文发布了首个法语-扎尔马语平行语料库和词汇表Feriji，并通过对大型语言模型进行微调实现了BLEU分数达30.06的法语-扎尔马语机器翻译。 |

# 详细

[^1]: DiaCap：联合学习“谁在何时说话”的视听对话字幕生成

    DiaCap: Jointly Learning Who Speaks When for Audiovisual Dialogue Captioning

    [https://arxiv.org/abs/2611.33153](https://arxiv.org/abs/2611.33153)

    DiaCap是一个统一的视听大语言模型，通过合成数据预训练与强化学习微调两个互补阶段，联合学习话语时间戳、视听说话人分离与说话人绑定，为视频中每句话生成转录、时间戳和说话人外观描述，实现全面的对话字幕生成。

    

    我们提出了DiaCap，一个用于全面对话字幕生成的视听大语言模型（LLM），它能为视频中的每一句话生成转录文本、时间戳以及说话人的视觉描述。考虑到话语时间与视听说话人分离（diarization）紧密耦合，DiaCap将这些任务统一在单一模型中，并通过两个互补阶段进行训练。首先，我们在22.5万条精选视频片段上训练DiaCap，使用合成字幕对每一句话进行详尽的密集标注。其次，我们通过强化学习在2.9千条经人工校正的片段上进一步优化模型，使用细粒度反馈直接评估话语时间、说话人分组和说话人绑定。在我们扩展了话语级时间戳和字幕遮蔽功能的DiaDemBench基准上，使用现成转录文本的DiaCap及其端到端变体DiaCap-E2E均取得了强劲的结果。具体而言，DiaCap在视听说话人分离任务上的表现与Gemini 4 Argon相当。

    arXiv:2611.33153v1 Announce Type: cross  Abstract: We present DiaCap, an audiovisual LLM for comprehensive dialogue captioning, which produces the transcript, timestamps, and a visual description of the speaker for every utterance in a video. Considering that utterance timing and audiovisual diarization are tightly coupled, DiaCap unifies the tasks in a single model, trained in two complementary stages. First, we train DiaCap on 225K curated clips, using synthetic captions that densely label every utterance in detail. Second, we further refine the model via reinforcement learning on 2.9K human-corrected clips, using fine-grained feedback that directly evaluates utterance timing, speaker grouping, and speaker binding. On DiaDemBench, which we extend with utterance-level timestamps and subtitle redaction, both DiaCap with off-the-shelf transcripts and its end-to-end variant DiaCap-E2E achieve strong results. Specifically, DiaCap performs on par with Gemini 4 Argon in audiovisual diarizat
    
[^2]: CoT-Proxy：面向长程策略蒸馏的跨格式迁移方法

    CoT-Proxy: Cross-Format Transfer for Distilling Long-Horizon Policies

    [https://arxiv.org/abs/2611.33148](https://arxiv.org/abs/2611.33148)

    该论文提出CoT-Proxy方法，证明即使教师模型的思维链被闭源API隐藏，也可以仅凭其动作序列（学生推理块留空）完成长程策略蒸馏，且所学策略能以跨格式迁移的方式在启用思考的推理阶段生效。

    

    推理模型通常通过强化学习成为长程智能体，或者以更廉价的方式——在更强教师模型的轨迹上进行微调，这些轨迹同时包含教师模型的思维链及其动作。标准的微调方法会同时模仿两者，然而最强的教师模型是闭源模型，其思维链隐藏在API之后，无法获取。我们证明思维链可以被完全省略，长程策略可以仅凭教师模型的动作序列蒸馏得到。学生在这些轨迹上进行训练，其推理块保持为空，同时通过对自身生成内容的重放缓冲区施加KL散度损失，来限制其偏离初始模型。令人惊讶的是，从这些推理块为空的轨迹中学到的策略能够迁移到启用思考的推理阶段——学生以自己的风格进行推理，同时行动表现得如同教师模型，我们将这种效应称为跨格式迁移。

    arXiv:2611.33148v1 Announce Type: new  Abstract: Reasoning models become long-horizon agents through reinforcement learning or, more cheaply, through fine-tuning on a stronger teacher's trajectories, which contain the teacher's chain-of-thought as well as its actions. The standard fine-tuning approach imitates both, yet the strongest teachers are closed models whose chain-of-thought is hidden behind an API. We show that the chain-of-thought can be left out altogether and that the long-horizon policy can be distilled from the teacher's actions alone. The student is trained on these trajectories with its reasoning block left empty, while a KL loss on a replay buffer of its own generations limits drift from its initial model. Surprisingly, the policy learned from these trajectories with an empty reasoning block carries over to inference with thinking enabled, where the student reasons in its own style while acting like the teacher, an effect we call cross-format transfer. Distilling the a
    
[^3]: BriConstrain：面向自回归积木组装生成的几何约束解码方法

    BriConstrain: Geometrically Constrained Decoding for Autoregressive Brick Assembly Generation

    [https://arxiv.org/abs/2611.33135](https://arxiv.org/abs/2611.33135)

    BriConstrain 通过在采样前动态检查积木放置的几何有效性，并以约束零件连接方式的方式缩小搜索空间，引导语言模型生成无碰撞、连接有效的乐高积木组装。

    

    语言模型可以根据文本生成乐高®积木组装，但其生成的结构中常常包含相互碰撞或悬空无所连接的零件。通常的做法是拒绝并重新采样无效的放置位置，这种方法会丢弃已解码的词元，且无法保证构建结果有效。我们提出了 BriConstrain，这是一种约束解码方法，它在采样之前对预训练语言模型的下一个词元概率进行限制，引导模型生成有效的放置位置。由于积木放置的有效性取决于尚未确定的构建状态，因此无法将其编译为固定语法，而必须进行动态检查。然而，在采样之前检查积木放置的有效性极具挑战性，因为有效放置的空间在组合意义上极其庞大，无法穷举。我们的关键思想是通过约束零件“如何连接”而非“哪些零件相互连接”来缩小搜索空间。具体而言，给定模型对…的预测（摘要原文在此处截断）。

    arXiv:2611.33135v1 Announce Type: cross  Abstract: Language models can generate LEGO$^\circledR$ assemblies from text, but their builds often contain parts that collide or attach to nothing. The usual approach of rejecting and resampling invalid placements discards decoded tokens without guaranteeing a valid build. We present BriConstrain, a constrained decoding method that restricts a pretrained language model's next-token probabilities before sampling, steering it towards valid placements. Because the validity of a brick placement depends on the non-predetermined build state, it cannot be compiled into a fixed grammar and must be checked dynamically. However, checking the validity of a brick before sampling is challenging as the space of valid placements is combinatorially large and cannot be enumerated exhaustively. Our key idea is to narrow the search space by constraining how parts attach rather than which parts attach to each other. Concretely, given the model's predictions of th
    
[^4]: 伪装的偏置：微小梯度编辑终止小语言模型的思考循环，并与前沿模型的权重编辑形成互补

    A Bias in Disguise: Tiny Gradient Edits Stop a Small Language Model's Thinking Loops and Complement a Frontier Model's Weight Edit

    [https://arxiv.org/abs/2611.33133](https://arxiv.org/abs/2611.33133)

    该论文发现通过梯度选择极少量关键权重微调即可让小模型走出无休止的思考循环，其本质是为无偏置项的模型隐式添加偏置，并与前沿模型的无梯度权重编辑形成互补。

    

    Qwen3.5-0.8B在思考模式下很少完成推理：在4,096个token的预算下，500个MATH-500答案中只有1或2个能关闭其思考块（两个随机种子），准确率仅为15.4%。最近的一项演示让GPT-6 Astra在无梯度的情况下编辑了该模型的131,072个权重，并报告在32,768个token预算下MATH-500从7.9%提升至48.1%。我们提供了缺失的梯度基线，然后将两种编辑方式结合起来。仅对梯度最大的140至139,750个权重进行微调（训练数据为模型自己对MATH训练题目的正确答案），在单GPU上训练2-9分钟后，可将4k token下思考模式的MATH-500提升至39.0-45.5%；50个训练样例即已足够，而随机选择1,398个位置只能达到16.3%。梯度将编辑集中在前三层中每层最接近恒定取值的通道上。由于该模型没有偏置项，局限于这些通道的编辑相当于一个加到残差流（原文此处截断）中的常数向量。

    arXiv:2611.33133v1 Announce Type: cross  Abstract: Qwen3.5-0.8B rarely finishes reasoning in thinking mode: at a 4,096-token budget only 1 or 2 of 500 MATH-500 answers close their think block (two seeds), and accuracy is 15.4%. A recent demonstration had GPT-6 Astra edit 131,072 of the model's weights without gradients and reported MATH-500 rising from 7.9% to 48.1% at a 32,768-token budget. We provide the missing gradient baseline, then combine the two edits. Fine-tuning only the 140 to 139,750 weights with the largest gradient, on the model's own correct answers to MATH training problems, raises thinking-mode MATH-500 at 4k tokens to 39.0-45.5% after 2-9 minutes of training on one GPU; 50 training examples suffice, and 1,398 random positions give only 16.3%. The gradient concentrates the edit on the most nearly constant channel of each of the first three layers. Because the model has no bias terms, an edit confined to those channels acts as a constant vector added to the residual str
    
[^5]: SAG-Mem-1.0 技术报告

    SAG-Mem-1.0 Technical Report

    [https://arxiv.org/abs/2611.33124](https://arxiv.org/abs/2611.33124)

    SAG-Mem 提出了一种构建记忆时无需模型推理的对话记忆框架，通过保留完整对话分块，并在查询时利用 BM25 检索、证据评估与词汇扩展进行多轮检索，有效发现与问题相似度低但具互补性的证据。

    

    长期对话记忆需要检索散落在各个对话会话中的互补事实，包括与原始问题相似度有限的中间证据。现有方法通常依赖派生的记忆表示或持久化的关系结构，这会引入构建成本，并可能丢失源对话的细节。我们提出了 SAG-Mem，这是一个对话记忆框架，它将完整的对话消息保存在互不重叠的分块中，并通过查询时的词汇扩展来发现互补证据。记忆构建过程无需任何模型推理，仅使用词汇索引，而不需要基于大语言模型（LLM）的信息抽取、摘要生成或语义关系构建。在查询阶段，BM25 首先检索初始候选结果，随后证据评估器识别出有用的分块，并利用所选证据中的词汇线索引导额外一轮检索。检索到的对话内容还会进一步标注日历解释信息（原文在此处截断）。

    arXiv:2611.33124v1 Announce Type: new  Abstract: Long-term conversational memory requires retrieving complementary facts scattered across dialogue sessions, including intermediate evidence with limited similarity to the original question. Existing approaches often rely on derived memory representations or persistent relational structures, introducing construction costs and potentially losing source details. We present SAG-Mem, a conversational memory framework that preserves complete dialogue messages in non-overlapping chunks and discovers complementary evidence through query-time lexical expansion. Memory construction requires no model inference, using a lexical index without LLM-based extraction, summarization, or semantic relation construction. At query time, BM25 retrieves initial candidates, an evidence evaluator identifies useful chunks, and lexical cues from selected evidence guide an additional retrieval round. Retrieved dialogue is further annotated with calendar interpretati
    
[^6]: AIChipBench：面向自动化AI加速器设计的LLM智能体基准测试

    AIChipBench: Benchmarking LLM Agents for Automated AI Accelerator Design

    [https://arxiv.org/abs/2611.33093](https://arxiv.org/abs/2611.33093)

    该论文提出AIChipBench基准，要求LLM编程智能体用可综合RTL实现量化大模型解码器，重点考察跨解码步骤的状态保持正确性及面积受限下的性能优化，从而系统评估智能体在自动化AI加速器设计中的真实能力。

    

    语言模型解码器需要在解码步骤之间传递状态，而在硬件中，错误的状态更新可能导致当前token输出正确、却在之后若干个token处破坏结果，因此一个设计可能通过了输出测试却仍然存在错误。AIChipBench探究LLM编程智能体需要具备什么才能正确构建这类硬件。智能体需要用可综合的寄存器传输级（RTL）代码实现量化版的Qwen3.5、Gemma 3、Gemma 4和Llama 3.2解码器，并以一个可执行的整数参考实现作为规范，该参考规定了每一步解码的输出和存储状态。智能体在固定预算内针对公开测试进行调试；其冻结后的设计会在保留的token序列、复位、内存停顿以及综合等维度上评分；此外还有一条赛道要求智能体在面积几乎不变的前提下对一个正确设计进行加速。在四个编程智能体共9,136次尝试中，难点集中在存储状态上。指出首个状态不匹配位置的反馈能够显著提升设计通过率。

    arXiv:2611.33093v1 Announce Type: cross  Abstract: A language-model decoder carries state between decoding steps, and in hardware a wrong state update can leave the current token correct and corrupt the output several tokens later, so a design can pass output tests and still be wrong. AIChipBench asks what LLM coding agents need to build such hardware correctly. Agents implement quantized Qwen3.5, Gemma 3, Gemma 4, and Llama 3.2 decoders in synthesizable register-transfer-level (RTL) code against an executable integer reference that specifies the outputs and the stored state at every decoding step. They debug against public tests under fixed budgets; their frozen designs are graded on held-out token sequences, resets, memory stalls, and synthesis, and a second track asks them to speed up a correct design at nearly fixed area. Across 9,136 attempts by four coding agents, the difficulty lies in stored state. Feedback that points to the first state mismatch raises acceptance over output-o
    
[^7]: FOL-Proof：基于确定性语义验证大语言模型文本到逻辑的翻译

    FOL-Proof: Verifying LLM Text-to-Logic Translations via Deterministic Semantics

    [https://arxiv.org/abs/2611.33078](https://arxiv.org/abs/2611.33078)

    该论文提出了FOL-Proof语义防火墙流水线，通过英语资源语法等语义分析工具确定性地生成候选翻译，并利用SMT求解器进行逻辑等价性检查，从而独立验证大语言模型生成的一阶逻辑翻译的语义忠实性。

    

    大语言模型越来越多地被用于将自然语言语句转换为一阶逻辑（FOL）以增强推理能力。然而，大语言模型容易出错，语义上不忠实的一阶逻辑公式可能会破坏下游推理。我们提出了FOL-Proof，一个语义防火墙流水线，它通过独立推导的语言学解释来验证大语言模型生成的一阶逻辑翻译。FOL-Proof的工作方式是：首先通过语义分析工具（如英语资源语法ERG和最小递归语义MRS）确定性地获得候选翻译池；然后应用SMT求解器检查每个推导出的语义读法与大语言模型生成的公式是否逻辑等价。如果翻译是忠实的，则接受；否则拒绝，或可选地通过反复调用大语言模型进行修复。该流程采取保守策略：当搜索不完整、遇到不支持的语法结构或检查结果不确定时，会做出绝对的……（摘要在此处截断）

    arXiv:2611.33078v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used to translate natural-language statements into first-order logic (FOL) to improve reasoning. However, LLMs are prone to errors, and so a semantically unfaithful FOL formula could corrupt downstream reasoning. We present FOL-Proof, a semantic firewall pipeline that validates LLM-generated FOL translations against independently derived linguistic interpretations. FOL-Proof operates by deterministically obtaining a pool of candidate translations via semantic analysis tools, such as the English Resource Grammar and Minimal Recursion Semantics; then applying an SMT solver to check each derived reading for logical equivalence to the LLM-generated formula. If the translation is faithful it is accepted; otherwise, it is rejected, or optionally repaired via repeated invocations to the LLM. The procedure is conservative: incomplete search, unsupported constructions, or inconclusive checks yield abs
    
[^8]: 值得等待吗？响应时间与流式呈现如何塑造对生成式AI答案的评价及聊天机器人的使用

    Worth the Wait? How Response Time and Streaming Shape Judgments of Generative AI Answers and Chatbot Engagement

    [https://arxiv.org/abs/2611.33050](https://arxiv.org/abs/2611.33050)

    本文通过法国政府聊天机器人竞技场的自然实验发现，边写边流式输出的答案比同一模型一次性显示的相同存储答案在盲选投票中获胜概率高出25.8个百分点，证明答案的呈现方式（而不仅是内容）显著影响用户对生成式AI的评价。

    

    基于大语言模型（LLM）的生成式AI聊天机器人如今以截然不同的速度给出回答：推理模型在回复前会进行深思熟虑，而响应缓存则能立即返回存储的答案。对这些模型进行排名的盲选偏好投票通常被解读为对其文本质量的判断，但如果呈现方式也很重要，那么更快的答案理应更受青睐。然而，关于“劳动错觉”的研究以及关于文本在屏幕上如何呈现的研究表明，边写边展开显示的答案反而比一次性完整显示的相同答案更受好评。我们利用自然使用中的变异来源检验了这些理论。首先，compar:IA（一个法国政府的聊天机器人竞技场）对重复的提示词，通过随机抽签，要么以正在撰写时逐字流式输出的全新答案作答，要么一次性显示来自同一模型的先前存储的答案。在相同的提示词下，回放式（一次性显示的存储）答案在决定性盲选投票中获胜的概率降低了25.8个百分点。

    arXiv:2611.33050v1 Announce Type: new  Abstract: Answers from generative AI chatbots built on large language models (LLMs) now arrive at very different speeds: reasoning models deliberate before replying, while response caches can return a stored answer at once. Blind preference votes, which rank these models, are read as judgments of their text, and if delivery matters, faster answers are expected to gain. Research on the labor illusion and on how text appears on screen suggests instead that an answer which unfolds as it is written is credited over the same answer shown at once. We test these accounts with three sources of variation in natural use. First, compar:IA, a French government chatbot arena, answered repeated prompts, by a random draw, either with a fresh answer streamed as it was written or with a stored earlier answer from the same model, shown at once. Within the same prompt, a replay lowered an answer's chance of winning a decisive blind vote by 25.8 percentage points. Th
    
[^9]: 潜在合并器：超网络生成LoRA的潜空间组合

    Latent Merger: Latent-Space Composition of Hypernetwork-Generated LoRAs

    [https://arxiv.org/abs/2611.33048](https://arxiv.org/abs/2611.33048)

    该论文发现超网络生成的LoRA在几何特性上与直接微调的LoRA截然不同（谱更集中、跨文档对齐更强），据此提出在潜空间中直接组合中间潜变量再解码的LoRA合并方法，以解决流式场景下逐文档存储LoRA不可持续的问题。

    

    近期研究提出了超网络方法，能够在单次前向传播中将文档转换为低秩参数更新，从而实现向冻结的大语言模型快速注入新知识。然而，在流式场景下为每个文档存储独立的LoRA适配器可能难以持续。本文以两个代表性系统Doc-to-LoRA和SHINE为例，研究组合超网络生成的LoRA这一问题。我们证明超网络生成的LoRA具有与直接微调的LoRA截然不同的几何特性：与基于下一个词预测和问答微调的适配器（其分布弥散且在不同文档间近乎正交）相比，超网络生成的LoRA表现出更强的谱集中性和更强的跨文档对齐性。基于这一观察，我们提出了潜空间LoRA合并方法：先组合超网络的中间潜变量，再通过冻结的LoRA生成头对合并后的潜变量进行解码。我们证明潜空间合并方法……

    arXiv:2611.33048v1 Announce Type: cross  Abstract: Recent work proposes hypernetworks that convert a document into a low-rank parameter update (LoRA) in a single forward pass, enabling fast injection of new knowledge into a frozen LLM. However, storing separate LoRA adapters for each document can be unsustainable in streaming settings. In this paper, we study the problem of composing hypernetwork-generated LoRAs using two representative systems, Doc-to-LoRA and SHINE. We show that hypernetwork-generated LoRAs have a geometry sharply different from directly fine-tuned LoRAs. They exhibit greater spectral concentration and stronger cross-document alignment than next-token prediction and QA fine-tuned adapters, which are diffuse and nearly orthogonal across documents. Guided by this observation, we propose latent-space LoRA merging. We compose the intermediate hypernetwork latents and then decode the merged latent through the frozen LoRA-generation head. We show that latent-space merging 
    
[^10]: 迈向基于大语言模型的推荐系统中不确定性量化的现实评估

    Towards Realistic Evaluation of Uncertainty Quantification for LLM-Based Recommender Systems

    [https://arxiv.org/abs/2611.33036](https://arxiv.org/abs/2611.33036)

    该论文提出了一个固定排序基准，在统一的输出契约下将八种异构的不确定性量化信号作为记录级监控器进行评估，从而首次实现了对基于LLM的推荐系统中不确定性量化方法的公平、现实的可比性评估。

    

    大语言模型（LLM）为推荐系统提供了灵活的接口。然而，其给出的排序结果需要可靠的实例级不确定性监控。在这种场景下，比较不确定性量化（UQ）方法变得具有挑战性，因为LLM排序器缺乏标准的物品分类头。其打分可能依赖于选项标签、提示词、生成过程以及可用的用户和物品证据。现有方法从不同的来源提取不确定性，并将其应用于推荐过程的不同阶段。一些方法评估已生成的排序结果，而另一些方法则利用不确定性来改变排序本身，这使得直接比较变得困难。本文引入了一个固定排序基准，在共享的输出契约下，将八种异构的UQ信号作为记录级监控器进行评估。我们在四个推荐数据集和两个LLM骨干网络上，在共享的…（原文在此处截断）

    arXiv:2611.33036v1 Announce Type: cross  Abstract: Large language models (LLMs) provide a flexible interface for recommendation systems. However, their issued rankings require reliable instance-level uncertainty monitoring. In this scenario, comparing uncertainty quantification (UQ) methods becomes challenging because LLM rankers lack a standard item-classification head. Their scores may depend on option labels, prompts, generation procedures, and the available user and item evidence. Existing methods derive uncertainty from different sources and apply it at different stages of the recommendation process. Some methods evaluate an issued ranking, whereas others use uncertainty to change the ranking itself, making direct comparison difficult. Here we introduce a fixed-ranking benchmark which evaluates eight heterogeneous UQ signals as record-level monitors under a shared output contract. We evaluate these signals across four recommendation datasets and two LLM backbones under a shared fi
    
[^11]: 前沿大语言模型能否通过无工具的权重空间编程修复量化网络？

    Can a Frontier LLM Repair a Quantized Network by Tool-Free Weight-Space Programming?

    [https://arxiv.org/abs/2611.33031](https://arxiv.org/abs/2611.33031)

    该研究证明前沿大语言模型无需任何工具、数据或推理能力，仅凭权重空间编程即可将4比特量化CNN的准确率从60.4%恢复至77.95%，接近原始浮点模型的81.4%水平。

    

    我们将Claude家族的四个大语言模型（LLM）以文本形式提供了具有2万参数的CIFAR-10卷积神经网络的完整权重，并要求它们在没有工具、数据或运行推理能力的情况下生成一个4比特网络。标准的舍入到最近（RTN）4比特量化会将测试准确率从81.4%降低到60.4%。我们设计了四种评估协议，它们在提供给LLM的输入和任务描述上有所不同，并分析它们的响应。在前两种协议中，LLM被提供了浮点网络和4比特RTN网络，并被告知准确率显著下降。在后两种协议中，它们只收到浮点网络和量化该网络的任务，这用于测试它们是自己主动应用优化，还是退回到普通的RTN。我们的分析表明，前沿LLM能够恢复大部分损失的准确率。在每层给出一个答案的情况下，Opus 5.5通过平衡舍入误差将4比特网络的准确率提升至77.95%。

    arXiv:2611.33031v1 Announce Type: cross  Abstract: We give four large language models (LLMs) of the Claude family the complete weights of a 20k-parameter CIFAR-10 CNN as text and ask them to produce a 4-bit network without tools, data or the ability to run inference. Standard round-to-nearest (RTN) quantization to 4 bit lowers the test accuracy from 81.4% to 60.4%. We design four evaluation protocols that differ in the input and task description given to the LLMs and analyze their responses. In the first two protocols the LLMs are given the float and the 4-bit RTN network and are told that the accuracy has dropped significantly. In the last two they receive only the float network and the task to quantize it, which tests whether they apply an optimization on their own or fall back to plain RTN. Our analysis shows that a frontier LLM can recover most of the lost accuracy. Given one answer per layer, Opus 5.5 raises the accuracy of the 4-bit network to 77.95% by balancing the rounding err
    
[^12]: SALSA：从学习到的分布中采样对抗性后缀以评估大语言模型安全性

    SALSA: Sampling Adversarial Suffixes from Learned Distributions for LLM Safety Assessment

    [https://arxiv.org/abs/2611.33030](https://arxiv.org/abs/2611.33030)

    SALSA提出了一种从优化攻击语料中学习越狱后缀分布的生成式框架，在将生成速度提升约90倍的同时达到与最先进优化攻击相当的成功率，并在黑盒迁移中对GPT-3.5-Turbo和GPT-4o分别实现99%和45%的攻击成功率。

    

    大语言模型（LLMs）仍然容易受到越狱攻击的威胁，攻击者在看似无害的提示后面附加经过优化的后缀，从而绕过安全对齐。现有的基于优化的GCG类攻击虽然有效，但在推理时代价高昂，且迁移能力较差。我们提出了SALSA，一个生成式框架，它从基于优化生成的攻击语料库中学习越狱后缀的分布。训练完成后，该模型能够支持高效的基于采样的攻击。在开源目标模型上，它取得了与最先进的优化基线相当的攻击成功率，同时将每个提示的生成时间从3.46秒降至0.04秒。在黑盒迁移场景中，它对专有系统仍然非常有效，包括在GPT-3.5-Turbo上达到99%的攻击成功率（ASR），并在难度显著更高的GPT-4o上达到45%的ASR，在相同采样预算下是最强基线的两倍以上。

    arXiv:2611.33030v1 Announce Type: new  Abstract: Large language models (LLMs) remain vulnerable to jailbreak attacks, where adversaries append optimized suffixes to otherwise benign looking prompts to bypass safety alignment. Existing optimization-based GCG-style attacks can be effective, but they are expensive at inference time, and suffer from poor transfer. We present, SALSA, a generative framework that learns a distribution over jailbreak suffixes from optimization-derived attack corpora. Once trained, the model supports efficient sampling-based attacks. On open-source targets, it achieves attack success rates comparable to state-of-the-art optimization baselines, while reducing generation time from 3.46s to 0.04s per prompt. In black-box transfer, it remains highly effective against proprietary systems, including 99% ASR on GPT-3.5-Turbo, and reaches 45% ASR on the substantially harder GPT-4o, more than doubling the strongest baseline at an equal sampling budget. We further offer 
    
[^13]: LLM裁判能否识别不安全的临床行为？一项故障注入研究

    Can LLM Judges Catch Unsafe Clinical Actions? A Fault-Injection Study

    [https://arxiv.org/abs/2611.33012](https://arxiv.org/abs/2611.33012)

    该研究提出了一种故障注入评估方法，通过在MedAgentBench中向经验证的智能体轨迹注入416条受控临床错误（覆盖八个错误类别、六个开源LLM裁判），发现LLM裁判识别不安全临床行为的可靠性高度依赖于临床证据和完整轨迹的可见性——仅看最终答案时既漏检错误又误报高达56%的正确轨迹。

    

    随着临床AI智能体开始与电子健康记录（EHR）交互，大语言模型（LLM）越来越多地被用于评估这些智能体的行为。但这些裁判者能在多大程度上可靠地识别出临床上的危险错误，尤其是当智能体看起来已经成功完成任务时？我们通过在MedAgentBench中向经过验证的智能体轨迹注入受控的临床错误来研究这一问题。实验包含416条错误轨迹、八个错误类别和六个开源权重的LLM裁判。通过将每条错误轨迹与其正确的对应轨迹进行对比，我们衡量裁判者是真正识别出了错误，还是仅仅简单地将更多操作标记为不安全。研究结果表明，能否获取临床证据会显著影响裁判的可靠性。当裁判只看到最终答案时，无法区分那些不改变最终答案的错误，同时还会将56%的正确轨迹误判为不安全。（摘要原文在此处截断）

    arXiv:2611.33012v1 Announce Type: new  Abstract: As clinical AI agents begin to interact with electronic health records (EHRs), large language models (LLMs) are increasingly used to evaluate their actions. But how reliably can these judges recognize clinically dangerous mistakes, especially when an agent appears to have completed its task successfully? We investigate this question using MedAgentBench by introducing controlled clinical errors into verified agent trajectories. The experiments include 416 faulty trajectories, eight error categories, and six open-weight LLM judges. By comparing each faulty trajectory with its correct counterpart, we measure whether judges genuinely recognize errors rather than simply flagging more actions as unsafe. The findings show that access to clinical evidence significantly improves judge reliability. When shown only the final answer, judges cannot distinguish errors that leave it unchanged, while flagging 56% of correct trajectories. With full traje
    
[^14]: Leto：以效用为中心的智能体记忆

    Leto: Utility-Centric Agent Memory

    [https://arxiv.org/abs/2611.33005](https://arxiv.org/abs/2611.33005)

    提出了Leto，一个以效用为中心的LLM智能体记忆系统，通过记忆索引与记忆回放机制，关注记忆对未来响应的效用而非对过去对话的保真度。

    

    大型语言模型（LLM）智能体依赖外部记忆来在多次交互之间保留和复用信息。现有的记忆系统将记忆视为对过去的记录，并根据其与对话的保真度来评估。然而，记忆的价值在于它能否改善智能体的响应。记忆系统需要在该响应产生之前就决定记忆什么以及提供什么，而保真度检验并未将这两个决策与响应挂钩。因此，忠实于对话的记忆可能无法改善响应。我们转而将记忆视为一个预测问题，并提出了Leto，一个面向LLM智能体的记忆系统，它关注记忆对未来请求的效用，而非其对应于过去的保真度。由于未来的请求是未知的，Leto通过记忆索引保留对对话细节的访问，使信息能够通过其来源持续可用。当请求到达时，记忆回放利用该索引来预判性地重建（摘要在此处截断）

    arXiv:2611.33005v1 Announce Type: new  Abstract: Large language model (LLM) agents rely on external memory to retain and reuse information across interactions. Existing memory systems treat memory as a record of the past and evaluate it by fidelity to the conversation. However, the value of memory lies in whether it improves the agent's response. A memory system decides what to memorize and what to serve before that response exists, and fidelity checks neither decision against it. Faithful memory can therefore fail to improve the response. We instead view memory as a prediction problem and present \textbf{\emph{\leto{}}}, a memory system for LLM agents that focuses on the utility of memory to the future rather than on its fidelity to the past. Because future requests are unknown, \emph{\leto{}} preserves access to conversation details through memory indexing, allowing information to remain available through its sources. When a request arrives, memory replay draws on this index to antic
    
[^15]: 通过大语言模型训练建模双语学习：一项关键期探索研究

    Modelling Billingual Learning Through LLM Training: A Critical-Period Exploration

    [https://arxiv.org/abs/2611.32997](https://arxiv.org/abs/2611.32997)

    该研究通过让小型GPT-2模型依次学习英语和不同距离的第二语言，并分析内部隐藏状态的余弦相似度，从模型内部表征角度探索了二语习得的关键期效应。

    

    二语（L2）习得的关键期理论表明，学习一门新语言的能力可能取决于先前习得的第一语言（L1）知识被修改的难易程度。这项探索性计算研究通过考察内部表征而非行为表现，将神经网络语言模型方法扩展至关键期效应的研究。一个小型两层GPT-2模型首先在英语上训练，随后通过顺序训练或弹性权重固化学习德语（相对相近的二语）或芬兰语（相对较远的二语），后者限制了英语相关重要参数的变化。在第一和第二个二语训练轮次后，研究将每种语言20个保留探测句的平均池化隐藏状态与纯英语模型的隐藏状态进行了比较，采用余弦相似度在嵌入层和两个Transformer层进行测量。英语相似度在Transformer层中保持较高，但在嵌入层中变为负值。

    arXiv:2611.32997v1 Announce Type: new  Abstract: Critical-period accounts of second-language (L2) acquisition suggest that the ability to learn a new language may depend on how easily previously acquired first-language (L1) knowledge can be modified. This exploratory computational study extends a neural language-model approach to critical period effects by examining internal representations rather than behavior. A small two-layer GPT-2 model was trained on English and then on German (a relatively related L2) or Finnish (a relatively distant L2) through either Sequential training or Elastic Weight Consolidation (EWC), which limits change in parameters important for English. Mean-pooled hidden states for 20 held-out probe sentences per language were compared with those of the English-only model using cosine similarity at the embedding layer and two transformer layers after one and two L2 epochs. English similarity remained high in the transformer layers but became negative in the embeddi
    
[^16]: 论基础模型的潜能与实现

    On Potentiality and Actuality Of Foundation Models

    [https://arxiv.org/abs/2611.32991](https://arxiv.org/abs/2611.32991)

    基础模型本身已蕴含完成特定任务所需的能力，适配并非添加新能力，而只需通过恒定的、与输入无关的向量移位将模型“实现”即可。

    

    基础模型通过大规模预训练构建，通常随后进行后训练，然后再被适配到特定任务和行为上。但是，这种适配是否真正增加了新能力，还是基础模型本身早已具备这些能力？我们将不同适配方式所引起的更新分解为一个在所有输入间共享的分量和一个随输入变化的残差分量。我们发现，共享分量足以产生大部分适配增益，并且可以直接学习得到：一个在每层学习一个恒定向量（对所有输入均相同）的移位适配器，就能复现LoRA和全量微调的行为。我们将这一现象称为基础模型的潜能原则：基础模型本身已经包含了实现特定任务专业能力所需的容量。要适配一个新任务，模型只需被“实现”，即通过一个恒定的、与输入无关的移位被推向某个方向。我们展示了这一原则在……方面的意义。

    arXiv:2611.32991v1 Announce Type: cross  Abstract: Foundation models are built through large-scale pretraining, often followed by post-training, and are then adapted to specific tasks and behaviors. But, does the adaptation add truly new capabilities, or the base model already holds it? We decompose the update induced by different adaptations into a component shared across all inputs and a residual that varies with the input. We find that the shared component suffices for most of the adaptation gain and can be learned directly: a shift adapter that learns a constant vector per layer, identical for every input, reproduces the behavior of LoRA and full fine-tuning. We call this the potentiality principle of foundation models: the base model already contains the capacity needed for task-specific expertise. To adapt to a new task, the model just needs to be actualized, i.e., pushed in a certain direction via a constant, input-independent shift. We show the implications of this principle on
    
[^17]: 当修复准确率产生误导：让视觉语言模型从自身失败中学习

    When Repair Accuracy Misleads: Training a VLM on Its Own Failures

    [https://arxiv.org/abs/2611.32982](https://arxiv.org/abs/2611.32982)

    该论文提出“验证式失败驱动修复”（VFR）方法，揭示了用模型自身失败样例微调视觉语言模型时准确率评估的两大陷阱（标签错误与措辞规律性），并通过LLM评审验证和最小对样例挖掘，将Qwen3-VL的空间关系准确率从66.7%提升至96.8%。

    

    在视觉语言模型自身的失败样例上进行微调是修复模型的一种直接方法，但在留出失败样例上的准确率——这一显而易见的成功衡量标准——却会产生误导。许多挖掘出的失败样例并非模型错误：对Qwen3-VL-8B-Instruct在基于COCO和GQA生成的问题上的100个失败样例进行人工审计，发现其中37个存在标签错误。此外，挖掘出的失败样例存在措辞上的规律性：仅基于问题的分类器在73.2%的留出失败样本对上都能答对两个答案。我们通过“验证式失败驱动修复”方法研究了这两个陷阱，该方法包括：类型化的最小对失败样例挖掘、由三个大语言模型评审在不查看存储标签的情况下回答问题进行验证、以答案为先的纠正目标，以及结合回放的LoRA微调。VFR修复了82.8%的留出失败样例（基础模型仅为10.0%），并且在独立标注、无条件限制的样本对上，通过空间关系将整体准确率从87.4%提升至92.4%，其中空间关系的准确率从66.7%大幅提升至96.8%。

    arXiv:2611.32982v1 Announce Type: cross  Abstract: Fine-tuning a vision-language model on its own failures is a direct way to repair it, but accuracy on held-out failures, the obvious measure of success, misleads. Many mined failures are not model errors: a human audit of 100 failures of Qwen3-VL-8B-Instruct on questions generated from COCO and GQA found label errors in 37. And mined failures share wording regularities: question-only classifiers get both answers right on 73.2% of held-out failure pairs. We study both traps with verified failure-driven repair (VFR): typed minimal-pair failure mining, verification by three LLM judges that answer without seeing the stored labels, answer-first corrective targets, and LoRA fine-tuning with replay. VFR repairs 82.8% of held-out failures (base model: 10.0%), and on independently labeled, unconditioned pairs it raises overall accuracy from 87.4% to 92.4% through spatial relations, whose accuracy rises from 66.7% to 96.8% even where the wording
    
[^18]: “我确实打算发货”：大语言模型会运行电商平台预付费诈骗吗？

    "I Do Intend to Deliver": Will LLMs Run a Marketplace Advance-Fee Scam?

    [https://arxiv.org/abs/2611.32957](https://arxiv.org/abs/2611.32957)

    研究表明，仅用一条无需越狱的普通提示词，十个主流大语言模型中就有八个愿意充当电商预付费诈骗的“卖家”，在1200次模拟对话中61%的买家被诱导预付款且单次成本仅约2.5美分，凸显了LLM被滥用于在线欺诈的严重风险。

    

    在线欺诈传统上受到与潜在受害者交谈所需人力成本的限制。我们探讨托管的大型语言模型现在是否会提供这种“劳动力”。我们提示十个现成模型扮演“免费物品，只需支付运费”这一买卖网站上常见的预付费诈骗中的卖家角色，仅使用一个普通系统提示词，无需越狱。每个模型分别与四个买家角色（从轻信到警惕）进行30次对话，买家由同一模型扮演（共1,200次对话）。我们根据买家资金的去向对每次对话进行标注，并对照两名人工编码员验证标签结果。737次对话（61%）以买家预先付款告终。十个模型中有八个几乎在每次对话中都配合出演，且同时运行买卖双方的成本约为每次付款2.5美分。只有想要可查证真实身份的警惕买家进行了抵制，而最“配合”的卖家通过编造姓名绕过了这一防线。

    arXiv:2611.32957v1 Announce Type: cross  Abstract: Online fraud is traditionally limited by the cost of the human labor needed to talk to potential victims. We ask whether hosted large language models will now supply that labor. We prompted ten off-the-shelf models to play the seller in the "free item, just pay delivery" advance-fee scam common on buy-and-sell sites, using a plain system prompt with no jailbreak. Each model played 30 conversations against each of four buyer personas, from credulous to guarded, with the same model playing the buyer (1,200 conversations). We label each conversation by what happened to the buyer's money, and validate the labels against two human coders. 737 conversations (61%) ended with the buyer paying in advance. Eight of the ten models played in almost every conversation, and running both sides cost about 2.5 cents per payment. Only the guarded buyer, who wanted a findable identity, resisted, and the most willing sellers got past it by inventing names
    
[^19]: ShellBench：对于AI智能体的日常任务而言，模型和智能体框架哪个更重要？

    ShellBench: What Matters More for AI Agents, Models, or Harness for Everyday Tasks

    [https://arxiv.org/abs/2611.32946](https://arxiv.org/abs/2611.32946)

    该论文推出ShellBench基准，通过对12个模型、5种智能体框架和3个推理水平共53,100次实验的评估发现，模型选择对日常任务性能的影响大于框架选择，但领先模型之间差距仅1.1个百分点，且高性能模型间的成本差异可达六倍之多。

    

    2026年全年，AI智能体在普通大众中的使用呈指数级增长，但用户往往难以确定哪种配置对其个人使用最为有效。为了帮助澄清这一问题，我们推出了ShellBench——基于超过1000万条用户请求构建的100项日常任务基准，每项任务都设置在包含杂乱文件和相互冲突数据源的真实工作空间中，并对正确性和安全性进行双重评分。我们评估了12个模型在5种智能体框架和3个推理水平下的表现，总计进行了53,100次实验，其中20%的任务被指定为公开发布。我们发现，模型选择对性能的影响大于框架选择，但三个领先模型之间的差距仅为1.1个百分点。GPT-6.1 Sol以约GPT-6 Astra每次尝试估算token成本的六分之一通过了61.4%的尝试，而GPT-6 Astra以62.5%的通过率位居首位。我们还发现，框架选择对性能几乎没有影响。

    arXiv:2611.32946v1 Announce Type: new  Abstract: The use of AI agents among the general public has been increasing exponentially throughout 2026, though it is often hard for users to determine which configuration is most effective for their personal use. To help provide clarity here we introduce ShellBench, 100 day-to-day tasks based on more than 10 million user requests, each set in a realistic workspace with messy files and competing data sources, and graded for both correctness and safety. We evaluate 12 models across 5 agent harnesses at 3 reasoning levels for a total of 53,100 rollouts, 20% of which are on tasks designated for public release. We find that model choice impacts performance more than harness choice, but the three leading models differ by only 1.1 percentage points. GPT-6.1 Sol passed 61.4% of attempts at about one-sixth of GPT-6 Astra's estimated token cost per attempt, while GPT-6 Astra led at 62.5%. We also found that harness choice had little effect on either perf
    
[^20]: 一种风格适用于所有人？从讨好倾向视角审视对话式AI中嵌入的文化价值观

    One Style Fits All? Cultural Values Embedded in Conversational AI via a People-Pleasing Lens

    [https://arxiv.org/abs/2611.32939](https://arxiv.org/abs/2611.32939)

    研究发现，无论在台湾还是韩国，支持型聊天机器人风格都能稳定提升用户的持续使用意愿、满意度和信任，而取悦倾向并不调节这些效应，仅在感知威胁上存在文化差异。

    

    基于大规模网络语料训练的对话式AI系统不可避免地编码了其训练数据中蕴含的文化价值观与互动规范，然而我们对已部署的大语言模型如何反映或强化特定文化的社会期望的理解仍然有限。本研究考察了支持型与挑战型聊天机器人交互风格如何塑造用户体验与持续使用意愿，以及取悦倾向是否在不同文化间调节这些效应。来自台湾（N = 49）和韩国（N = 52）的参与者完成了一项协作式旅游规划任务。结果表明：（1）支持型聊天机器人在持续使用意愿、满意度和信任方面始终表现更佳；（2）取悦倾向并未调节这些效应；（3）文化差异仅出现在感知威胁层面，即在台湾样本中较高的取悦倾向与更高的基线威胁感相关，而在韩国样本中则不然。这些发现揭示了……（原文摘要到此截断）

    arXiv:2611.32939v1 Announce Type: cross  Abstract: Conversational AI systems trained on large-scale web corpora inevitably encode the cultural values and interactional norms embedded in their training data, yet our understanding of how deployed LLMs reflect or reinforce culture-specific social expectations remains limited. This study examined how supportive versus challenging chatbot interaction styles shape user experience and continuance intention, and whether people-pleasing tendency (PPT) moderates these effects across cultures. Taiwanese (N = 49) and Korean (N = 52) participants completed a collaborative tourism-planning task. Results showed that: (1) supportive chatbots consistently led to higher continuance intention, satisfaction, and trust; (2) PPT did not moderate these effects; and (3) cultural variation emerged only in perceived threat, where higher PPT was associated with greater baseline threat in the Taiwanese but not the Korean sample. These findings reveal how a genera
    
[^21]: InstructReplay：重放助手轮次以在事实性微调中保持通用行为

    InstructReplay: Replaying Assistant Turns to Preserve General Behavior During Factual Fine-Tuning

    [https://arxiv.org/abs/2611.32918](https://arxiv.org/abs/2611.32918)

    提出InstructReplay方法，通过让初始模型预先对通用指令生成回答并在微调时约束模型保持接近这些回答，在事实性微调中学习新知识的同时，将通用能力平均损失从28.1分降至2.4分，并避免了回答长度的崩溃。

    

    在新的事实数据上对指令微调后的语言模型进行微调，往往会损害其通用能力。我们发现，标准微调平均会使指令遵循、推理和编程等基准下降35分，并且这些损失与回答长度的崩溃密切相关：微调后的模型回答长度至少缩短一半，且回答偏离初始模型长度越远，失败的概率就越高。我们提出InstructReplay方法：让初始模型对一组通用指令各回答一次，并在微调过程中约束模型在这些回答上保持接近初始模型。在三个模型系列和两个领域中，InstructReplay将五个通用基准的平均下降从28.1分降至2.4分，并使回答长度保持接近初始模型，同时在习得新事实方面的效果与标准微调相当。在全部六种设置中，它比七种适应方法都能更好地平衡学习与保持。

    arXiv:2611.32918v1 Announce Type: new  Abstract: Fine-tuning an instruction-tuned language model on new facts often erodes its general abilities. We find that standard fine-tuning lowers instruction-following, reasoning and coding benchmarks by 35 points on average, and that these losses track a collapse in answer length: fine-tuned models at least halve their responses, and the further a response drifts from the initial model's length, the more often it fails. We propose InstructReplay: the initial model answers a set of general instructions once, and during fine-tuning the model is kept close to the initial model on these answers. Across three model families and two domains, InstructReplay reduces the average decline on five general benchmarks from 28.1 to 2.4 points and keeps answer length close to the initial model's, while learning the new facts as well as standard fine-tuning. It balances learning and retention better than seven adaptation methods in all six settings. Controlled 
    
[^22]: SOARS：面向智能体搜索的系统一（System One）辅助重排序

    SOARS: System One Assisted Reranking for Agentic Search

    [https://arxiv.org/abs/2611.32917](https://arxiv.org/abs/2611.32917)

    SOARS是一个无需训练的智能体搜索重排序框架，利用快速“系统一”模型在单次调用中完成相关性估计与候选重排序，在六个大模型骨干上显著提升答案准确率和证据召回率，同时大幅降低搜索调用次数、延迟和推理成本。

    

    深度搜索智能体通过迭代检索与推理来满足复杂的信息需求，其中检索到的证据的相关性会直接影响下游任务的性能。现有的重排序器虽然能够提升检索效果，但在整个搜索过程中被频繁调用时会引入大量的延迟和成本。我们提出了SOARS，这是一个无需训练的框架，利用“系统一”模型进行快速且经过校准的相关性估计。在每个搜索步骤中，SOARS从长文档中筛选出与查询相关的段落，并在单次调用中完成候选结果的重排序，同时综合考虑原始问题和智能体生成的子查询。在BrowseComp-Plus和WebShaper数据集上的六个大语言模型骨干上，SOARS持续提升答案准确率和证据召回率，同时减少搜索调用次数、端到端延迟和推理成本。在BrowseComp-Plus上，它将GPT-5.4 Mini的准确率从73.9%提升至84.7%，同时将搜索调用次数从23次降至14次，……

    arXiv:2611.32917v1 Announce Type: cross  Abstract: Deep search agents address complex information needs through iterative retrieval and reasoning, where the relevance of retrieved evidence influences downstream performance. Existing rerankers improve retrieval effectiveness but introduce substantial latency and cost when invoked throughout the search trajectory. We introduce SOARS, a training-free framework leveraging a System One model for fast, calibrated relevance estimation. At each search step, SOARS selects query-relevant passages from long documents and reranks candidates in a single call, jointly considering the original question and agent-generated sub-query. Across six LLM backbones on BrowseComp-Plus and WebShaper, SOARS consistently improves answer accuracy and evidence recall while reducing search calls, end-to-end latency, and inference cost. On BrowseComp-Plus, it improves GPT-5.4 Mini's accuracy from 73.9% to 84.7% while reducing search calls from 23 to 14, demonstratin
    
[^23]: 在线对话中建设性言论、仇恨言论与赞同态度的联合检测

    Joint Detection of Constructive Speech, Hate Speech and Agreement in Online Conversations

    [https://arxiv.org/abs/2611.32889](https://arxiv.org/abs/2611.32889)

    本文构建了一个标注了仇恨语气、建设性语气和赞同态度的英丹麦语社交媒体评论大规模数据集，并通过微调多语言大语言模型实现三项任务的联合检测，其中 E5-large 在上下文增强架构下表现最佳。

    

    在线仇恨言论是一个日益严重的问题，对群体和个人都产生负面影响，可能导致用户退出在线对话和辩论。随着一些知名社交媒体平台减少其审核力度，建设性言论已成为对抗仇恨的替代策略。自动检测建设性言论和仇恨言论有助于监测工作，并增进对其流行程度和在线动态的理解。在这项工作中，我们收集并标注了一个大型数据集，包含英语和丹麦语社交媒体评论，标注维度包括仇恨语气、建设性语气以及对父评论的赞同程度。我们微调了多个多语言大语言模型在这三个维度上进行分类，并比较了上下文增强与非上下文增强的模型架构。E5-large 大语言模型在上下文增强（平均 F1 为 82%）和非上下文增强（平均 F1 为 81.6%）分类任务中均表现最佳，证明了……

    arXiv:2611.32889v1 Announce Type: cross  Abstract: Online hate speech is an ever-growing issue with negative impacts on both groups and individuals, potentially causing users to withdraw from online conversations and debate. With some prominent social media platforms reducing their moderation efforts, constructive speech has emerged as an alternative strategy for countering hate. Automated detection of constructive and hateful speech can aid monitoring and improve understanding of their prevalence and online dynamics. In this work, we collect and annotate a large dataset of English and Danish social media comments for hateful tone, constructive tone, and agreement with parent comments. We fine-tune several multilingual LLMs to classify along these three dimensions and compare context-enriched and non-context-enriched model architectures. The E5-large LLM performs best on both the context-enriched (82% average F1) and non-context-enriched (81.6% average F1) classification tasks, demonst
    
[^24]: 新鲜密钥，陈旧秘密：跨链接LLM提示间的数值泄露

    Fresh Keys, Stale Secrets: Numeric Leakage across Linked LLM Prompts

    [https://arxiv.org/abs/2611.32864](https://arxiv.org/abs/2611.32864)

    该论文揭示了针对LLM提示中机密商业数字的伪装保护在面对服务提供商的跨查询链接分析时会失效——即使每次查询使用新密钥，攻击者也能通过累积多次独立观察逐步恢复企业重复询问的机密价格与利润率信息。

    

    小企业反复向云端语言模型询问相同的机密价格、单位成本和利润率。针对这些数字的数值提示伪装通常是一次仅评估单个提示，但服务提供商能够看到企业的完整查询历史，其中每个新密钥都为同一组数字提供了新的、独立的观察视角。由于真实成本并非公开信息，我们在数值已知的合成企业上测量：当一个诚实但好奇的提供商知晓伪装算法、掌握行业级价格和利润率统计信息，并将企业的查询进行链接时，能够恢复多少机密信息。在单个提示场景下，逐字段缩放、Zhang等人保序替换的固定轴实例化方法以及Prεεmpt发布的净化器，可使攻击者相对于仅依赖先验统计猜测的利润率命中率优势保持在-0.6至+4.4个百分点之间。然而在16次使用新密钥的链接查询之后，我们测试的每一种伪装方法（包括ε≥0.5的Prεεmpt）都……

    arXiv:2611.32864v1 Announce Type: cross  Abstract: Small businesses repeatedly ask cloud language models about the same confidential prices, unit costs and margins. Numeric prompt disguises for such numbers are evaluated one prompt at a time, but a provider sees a firm's history, in which each fresh secret gives a new, independent view of the same numbers. Since real costs are not public, we measure on synthetic firms with known values how much an honest-but-curious provider recovers when it knows the disguise algorithm, holds sector-level price and margin statistics, and links a firm's queries. In one prompt, per-field scaling, a fixed-axis instantiation of Zhang et al.'s order-preserving substitution and Pr$\varepsilon\varepsilon$mpt's released sanitizer keep the attacker's margin-hit advantage over a prior-only guess between -0.6 and +4.4 percentage points. After 16 linked queries with fresh secrets, every disguise we test (Pr$\varepsilon\varepsilon$mpt at $\varepsilon\geq 0.5$) has
    
[^25]: 从语法增量到词元掩码：运行时语法演化下的增量式约束解码

    From Grammar Deltas to Token Masks: Incremental Constrained Decoding under Runtime Grammar Evolution

    [https://arxiv.org/abs/2611.32807](https://arxiv.org/abs/2611.32807)

    提出了GramDelta框架，通过在已提交前缀上维护解析状态并仅增量传播新增语法引起的变更，实现了运行时语法演化下的高效约束解码，避免了重复解析工作。

    

    语法约束解码使大语言模型能够生成满足形式化约束的输出，但在代码和领域特定语言生成中，相关约束并不总是在解码前完全可用，而可能随着新的声明、推断的类型信息或运行时分析反馈逐步涌现。现有的结构化解码引擎能够高效处理固定的、预先声明的或之前见过的约束结构；然而，当首次出现的语法更新改变了已提交前缀所诱导的解析器状态时，直接的解决方案是在更新后的语法下重建约束状态，从而重复前缀上已经执行过的解析工作。我们提出了GramDelta，一个支持仅追加式运行时语法更新的增量约束解码框架。GramDelta在已提交前缀上维护解析状态，并仅传播由新添加语法所引起的变更。

    arXiv:2611.32807v1 Announce Type: new  Abstract: Grammar-constrained decoding enables large language models to generate outputs that satisfy formal constraints, but in code and domain-specific language generation, the relevant constraints are not always fully available before decoding and may emerge progressively from new declarations, inferred type information, or runtime analysis feedback. Existing structured-decoding engines efficiently handle fixed, predeclared, or previously seen constraint structures; however, when a first-seen grammar update changes the parser state induced by an already committed prefix, the straightforward solution reconstructs the constraint state under the updated grammar, repeating parsing work already performed over the prefix. We introduce GramDelta, an incremental framework for constrained decoding with append-only runtime grammar updates. GramDelta maintains parse state over the committed prefix and propagates only changes induced by newly added grammar
    
[^26]: 基于睡眠机制的样本高效上下文适应

    Sample-Efficient In-Context Adaptation via Sleep

    [https://arxiv.org/abs/2611.32802](https://arxiv.org/abs/2611.32802)

    该研究提出在Transformer语言模型预训练中引入“睡眠”机制（通过多次循环前向传递精炼上下文表示），显著提升了模型在一般性任务上少样本上下文学习的样本效率。

    

    arXiv:2611.32802v1 公告类型：交叉 摘要：从有限经验中学习并泛化的能力，即样本效率，是智能的基石。我们如何构建能够将少量用户交互、几段陌生代码或上下文中提供的少量实验结果转化为可泛化知识的语言模型？先前的研究发现，“睡眠”——即模型执行 N 次循环前向传递以精炼其对一段上下文表示的过程——可以改善长上下文推理（具体而言，在 SSM-注意力混合模型中）。我们研究这种方法是否也能改善Transformer语言模型在一般性、非合成任务上的上下文适应能力。为此，我们使用睡眠机制在多达 1000 亿 token 上预训练了参数量高达 12 亿的语言模型。值得注意的是，更长的睡眠（即使用更多的循环前向传递次数 N）使模型能够在多个少样本算法任务上更快地进行上下文学习……

    arXiv:2611.32802v1 Announce Type: cross  Abstract: The ability to learn and generalize from limited experience, or sample efficiency, is a cornerstone of intelligence. How can we build language models that turn a few user interactions, a few snippets of unfamiliar code, or a handful of experimental results provided in context into knowledge that generalizes? Prior work has found that sleep, a process where the model performs $N$ recurrent forward passes to refine its representations of a chunk of context, can improve long-context reasoning (specifically, within SSM-attention hybrid models). We study whether this method also improves in-context adaptation on general, non-synthetic tasks for transformer language models. To this end, we pre-train language models up to 1.2B-parameters on up to 100B tokens with sleep mechanisms. Notably, longer sleep (i.e., using a larger number of recurrent forward passes $N$) enables the model to learn faster in-context on multiple few-shot algorithmic, q
    
[^27]: 神经音频编解码器Mimi中是否编码了韵律信息？

    Is Prosody Encoded in the Neural Audio Codec Mimi?

    [https://arxiv.org/abs/2611.32795](https://arxiv.org/abs/2611.32795)

    本文研究了被Moshi、Sesame等对话语音智能体使用的神经音频编解码器Mimi在编码阶段是否捕获了韵律信息，填补了以往仅从解码端评估编解码器质量、缺乏对编码端语音信息考察的研究空白。

    

    神经音频编解码器（NACs）已成为现代语音-语言模型的基础设施，它将连续的语音信号离散化为token序列，使大型语言模型能够处理和生成音频。大多数研究在解码阶段评估编解码器的质量，即评估重建语音的自然度。然而，很少有研究调查编码过程中究竟捕获了哪些语音学信息。在本文中，我们研究了神经音频编解码器Mimi中韵律的潜在相关表征，该编解码器被Moshi和Sesame等对话式语音智能体所使用，并被应用于语音到语音翻译（Hibiki）等下游任务中。

    arXiv:2611.32795v1 Announce Type: new  Abstract: Neural audio codecs (NACs) have become a fundamental infrastructure for modern speech-language models, discretizing continuous speech signals into token sequences that enable large language models to process and generate audio. Most research evaluates codec quality at the decoding phase, assessing the naturalness of reconstructed speech. However, little research investigates what phonetic information is captured during encoding. In this paper, we investigated the potential correlates of prosody in the NAC Mimi, a codec used by conversational speech agents such as Moshi and Sesame and used in downstream tasks such as speech-to-speech translation (Hibiki).
    
[^28]: 通过PCA测量大语言模型中FFN的逐层内在维度

    Measuring Layer-wise Intrinsic Dimensionality of FFNs in LLMs via PCA

    [https://arxiv.org/abs/2611.32782](https://arxiv.org/abs/2611.32782)

    该研究用PCA测量58个开源LLM各层FFN的内在维度，发现中间层的内在维度显著高于边缘层的“中间重”分布，表明逐层均匀分配参数的传统设计并非最优，为非均匀的参数预算分配提供了实证依据。

    

    现代大语言模型（LLM）在每一层都保持前馈网络（FFN）中间宽度相同，这是从原始Transformer继承而来的惯例。然而，机制研究表明不同深度的层做着质上不同的工作，这引出了一个自然的问题：它们的容量需求是否也是不均匀的？如果是，固定的参数预算能否比均匀分配得到更明智的利用？为探究这一问题，我们对每个FFN下投影的输入应用主成分分析（PCA），并以将激活重构至固定余弦相似度所需的最少主成分数量来度量每一层的内在维度。通过扫描涵盖14个模型家族、参数规模从70M到72B的58个开源权重LLM，我们发现了一种反复出现的“中间重”轮廓——中间层的内在维度高于边缘层；该轮廓对校准语料库具有鲁棒性，甚至在不同……（注：摘要原文在此处被截断）

    arXiv:2611.32782v1 Announce Type: cross  Abstract: Modern large language models (LLMs) keep the feed-forward network (FFN) intermediate width identical across every layer, a convention inherited from the original Transformer. Yet mechanistic work suggests layers at different depths do qualitatively different work, raising a natural question: are their capacity demands also non-uniform, and if so, can a fixed parameter budget be allocated more wisely than uniformly? To probe this, we apply principal component analysis (PCA) to the input of each FFN down-projection and measure, for every layer, its intrinsic dimensionality as the smallest number of principal components needed to reconstruct the activation up to a fixed cosine similarity. Scanning 58 open-weight LLMs spanning 14 families and 70M to 72B parameters, we find a recurring middle-heavy profile in which middle layers carry higher intrinsic dimensionality than edge layers; the profile is robust to the calibration corpus, even acr
    
[^29]: 超越固定上下文似然：基于上下文扰动差异的零样本AI生成文本检测

    Beyond Fixed-Context Likelihood: Zero-Shot AI-Generated Text Detection via Context Perturbation Discrepancy

    [https://arxiv.org/abs/2611.32778](https://arxiv.org/abs/2611.32778)

    提出零样本AI生成文本检测方法CPD，通过衡量token对数概率增益相对于完整前缀分布期望的偏离程度（上下文扰动差异）来识别AI文本，在多个数据集上显著超越七个零样本基线，平均AUROC达94.80%。

    

    来自较早上下文的相同token对数概率增量，在某些位置可能是典型现象，而在另一些位置则可能异常大。我们提出了上下文扰动差异（Context Perturbation Discrepancy, CPD），这是一种零样本检测器，它相对于单个预训练语言模型的期望来衡量这种增益。在保持文本段落固定不变的情况下，CPD在每个观测token的完整前缀和最近token的短窗口两种条件下对其进行评分。在完整前缀分布下，期望增益等于该分布到受限前缀分布的KL散度。CPD从每个观测到的增益中减去这一期望，并用其解析参考标准差对带符号偏差之和进行缩放。在来自GEN、DetectRL-X和HART的未经编辑的人类与AI文本上，在相同的方差尺度下，减去期望增益使AUROC提高了32.78至35.81个百分点。CPD优于七个零样本基线方法，平均达到94.80%的AUROC和79.60%的召回率。

    arXiv:2611.32778v1 Announce Type: new  Abstract: The same increase in token log probability from earlier context can be typical at one position and unusually large at another. We introduce Context Perturbation Discrepancy (CPD), a zero-shot detector that measures this gain relative to what a single pretrained language model expects. Keeping the passage fixed, CPD scores each observed token under its full prefix and under a short window of recent tokens. Under the full-prefix distribution, the expected gain equals the KL divergence from that distribution to the restricted-prefix distribution. CPD subtracts this expectation from each observed gain and scales the sum of the signed deviations by its analytical reference standard deviation. On unedited human and AI texts from GEN, DetectRL-X, and HART, subtracting the expected gain raises AUROC by 32.78-35.81 percentage points at the same variance scale. CPD outperforms seven zero-shot baselines, averaging 94.80% AUROC and 79.60% recall at 
    
[^30]: ASCENT：一个用于评估和推进大语言模型在常见临床场景中逐步诊断推理能力的基准

    ASCENT: A Benchmark for Evaluating and Advancing Stepwise Diagnostic Reasoning in Large Language Models on Common Clinical Scenarios

    [https://arxiv.org/abs/2611.32734](https://arxiv.org/abs/2611.32734)

    该论文提出ASCENT，一个由临床医生标注、包含3,078个逐步推理问题的基准，通过将病例分解为与电子病历对齐的步骤（发现、印象、依据）来评估信息不完整条件下的中间诊断推理，揭示出即使最强的推理模型在逐步诊断任务上仍有显著提升空间。

    

    大语言模型（LLM）在医学问答方面表现出色，但很少在真实临床工作流程所定义的逐步诊断推理上进行评估——在这种推理中，诊断印象会随着信息的积累而不断修正。我们构建了ASCENT（Annotated Stepwise Clinical rEasoning for NaturalisTic Diagnosis，面向自然主义诊断的注释式逐步临床推理），这是一个由临床医生标注的基准与训练资源，包含3,078个源自MedQA-USMLE的逐步推理问题，将每个病例描述分解为与电子病历（EMR）对齐的步骤（检查发现、诊断印象、支持性依据），从而能够在信息不完整的情况下评估中间推理过程。基于ASCENT的实验与训练揭示了当前大语言模型处理逐步诊断推理的现状：即使在MedQA-USMLE上表现优异的强推理模型，在ASCENT上仍有巨大的提升空间，而通用前沿模型则落后更多，这暴露了“信息完整诊断”与“逐步诊断”之间存在的持续差距。对Qwen2.5-（摘要在此处截断）

    arXiv:2611.32734v1 Announce Type: cross  Abstract: Large language models (LLMs) excel at medical question answering yet are rarely evaluated on the stepwise diagnostic reasoning that defines real clinical workflows, where impressions are revised as information accumulates. We build Annotated Stepwise Clinical rEasoning for NaturalisTic Diagnosis (ASCENT), a clinician-annotated benchmark and training resource of 3,078 stepwise problems derived from MedQA-USMLE that decomposes each vignette into EMR-aligned steps (Findings, Impression, supporting Rationale), enabling evaluation of intermediate reasoning under incomplete information.   Experiments and training with ASCENT revealed insights into how current LLMs handle stepwise diagnostic reasoning. Even strong reasoning models that perform well on MedQA-USMLE leave substantial headroom on ASCENT, and general-purpose frontier models trail further, exposing a persistent gap between fully informed and stepwise diagnosis. Fine-tuning Qwen2.5-
    
[^31]: 对一个主权大语言模型、其推理过程及其训练阶段的红队测试

    Red-Teaming a Sovereign LLM, Its Reasoning and Its Training Stages

    [https://arxiv.org/abs/2611.32710](https://arxiv.org/abs/2611.32710)

    该研究通过多语言红队测试发现，即使系统提示词中包含保密指令，主权开源LLM Apertus在诱饵攻击下仍会以超过95%的比例泄露敏感信息，且其推理轨迹也可能泄露最终答案中刻意隐去的机密内容。

    

    开源权重大语言模型（LLM）越来越多地被部署用于涉及敏感信息的应用中。我们研究了系统提示词中的保密指令是否能够保护此类信息。我们对开源主权LLM Apertus v1.5以及另外五个模型家族进行了红队测试，涵盖六种语言和多种推理模式。我们使用合成的个人记录来评估直接泄露、纠正诱饵攻击、是非验证、派生值查询、推理泄露和授权检查等攻击方式。我们还探讨了版权归属、历史歪曲和刻板印象等问题。尽管设有保密指令，Apertus 8B和70B在纠正诱饵攻击中分别有96%和95%的请求泄露了信息。验证和派生值查询也会在不直接暴露标识符的情况下暴露受保护的信息。此外，即使最终答案未透露机密记录，推理轨迹仍可能泄露这些机密信息。

    arXiv:2611.32710v1 Announce Type: new  Abstract: Open-weight large language models (LLMs) are increasingly deployed for applications with sensitive information. We investigate whether confidentiality instructions in system prompt can protect such information. We red-team Apertus v1.5, an open sovereign LLM, and five other model families across six languages and reasoning modes. We use synthetic personal records to evaluate direct disclosure, correction baits, yes/no verification, derived-value queries, reasoning leakage and authorisation checks. We also explore copyright attribution, historical misrepresentation and stereotyping. Despite confidentiality instructions, Apertus 8B and 70B leak information in 96% and 95% of requests, respectively, in correction-bait attacks. Verification and derived-value queries also expose protected information without necessarily revealing identifiers. Moreover, reasoning traces can leak confidential records even when final answers withhold them. Our ex
    
[^32]: Bathys：深度搜索智能体的进阶

    Bathys: Advancing Deep Search Agents

    [https://arxiv.org/abs/2611.32660](https://arxiv.org/abs/2611.32660)

    提出 Bathys 9B 和 27B 深度搜索智能体，通过可验证问题合成、SFT/KTO/CM-GSPO 训练流水线以及上下文压缩下的在线强化学习框架，提升智能体在复杂开放式任务中迭代检索与证据验证的能力。

    

    深度搜索是智能体应对复杂、开放式任务的关键能力，在这类任务中，解决问题需要迭代式的信息收集、查询重构以及基于证据的验证，而非单次检索步骤。为满足这些需求，我们提出了 Bathys 9B 和 Bathys 27B 两个深度搜索智能体，其训练框架涵盖可验证的问题合成、后训练以及上下文压缩下的在线强化学习（RL）。我们高质量的数据构建确保了搜索深度与可靠性：每条线索都被改写为需要独立检索的间接指代，并保留证据链以便验证这些线索共同确定唯一答案。经过精心整理的轨迹数据支撑了一个由监督微调（SFT）、Kahneman-Tversky 优化（KTO）以及压缩管理组序列策略优化（CM-GSPO）组成的训练流水线：SFT 提供了稳定的基础……

    arXiv:2611.32660v1 Announce Type: cross  Abstract: Deep search is a critical capability for agents tackling complex, open-ended tasks, where solving a problem requires iterative information gathering, query reformulation, and evidence-based verification rather than a single retrieval step. To address these requirements, we present Bathys 9B and 27B deep search agents built around a training framework that spans verifiable question synthesis, post-training, and online reinforcement learning (RL) under context compression. Our high-quality data construction ensures both search depth and reliability: each clue is reformulated as an indirect reference requiring independent retrieval, and retained evidence chains enable verification that the clues jointly determine a unique answer. Curated trajectories underpin a pipeline of supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Compaction-Managed Group Sequence Policy Optimization (CM-GSPO): SFT provides a stable foundation
    
[^33]: 生成模式对齐的合成对话并度量其多样性

    Generating Schema-Aligned Synthetic Dialogues and Measuring Their Diversity

    [https://arxiv.org/abs/2611.32631](https://arxiv.org/abs/2611.32631)

    该论文提出“先记录后对话”的逆序合成方法，通过先采样符合模式的有效记录再生成对话，无需人工标注即可获得成对数据，并利用变化配置、人物角色和主题规划等显式机制显著提升合成对话的多样性。

    

    训练或评估一个将对话转换为结构化记录的模型需要成对数据，即一段对话文本加上它应被转换成的准确记录。这样的成对数据标注成本高昂，而且在敏感领域中甚至根本难以获得。因此，我们以相反的顺序来构建它们：先采样一个符合模式的有效记录，然后据此撰写对话，并可选地由对话合成语音，这样标签先于文本存在，无需任何标注步骤。简单地用提示词让语言模型生成此类对话，只会产生单一对话的微小变体，因此我们加入了采样的变化配置、人物角色以及规定记录字段出现顺序的主题规划，作为显式的变化机制。我们通过嵌入空间来度量最终传导到输出的变化程度，其中嵌入方差分布在约两倍半数量的独立方向上。

    arXiv:2611.32631v1 Announce Type: new  Abstract: Training or evaluating a model that turns a conversation into a structured record requires paired data, meaning a transcript together with the exact record it should be converted into. Such pairs are expensive to annotate, and in sensitive domains they are also difficult to obtain at all. We therefore build them in the opposite order by sampling a schema-valid record first, then writing a dialogue from it, and optionally synthesizing speech from the dialogue, so the label exists before the text and we need no annotation step. Naively prompting a language model for such dialogues yields small variations of a single conversation, so we add a sampled variation configuration, personas, and a topic plan that orders the fields of the record as explicit mechanisms for variation. We measure the variation that reaches the output through embedding space, where the embedding variance spreads over about two and a half times as many independent direc
    
[^34]: MargEmbed-Akan：通过跨语言分数蒸馏连接契维语问题与英语知识

    MargEmbed-Akan: Bridging Twi Questions and English Knowledge through Cross-Lingual Score Distillation

    [https://arxiv.org/abs/2611.32630](https://arxiv.org/abs/2611.32630)

    该论文提出了 MargEmbed-Akan，一个基于跨语言分数蒸馏从 AfriE5-Large-instruct 训练得到的 5.6 亿参数开源查询编码器，有效连接契维语问题与英语知识检索。

    

    契维语使用者在网上搜索的大多数信息都是以英语书写的，这使得跨语言检索变得至关重要。我们在两个契维语-英语任务上评估了八个系统：BM25、五个开源嵌入模型以及两个专有模型（gemini-embedding-001 和 OpenAI text-embedding-3-large）。AfriQA 数据集包含 1,221 个原生编写的阿散蒂契维语问题，这些问题需要检索出能够回答它们的英语维基百科段落。圣经翻译检索任务则要求一段契维语经文从 6,195 个候选中检索出对应的英语译文。我们提出了 MargEmbed-Akan，这是一个拥有 5.6 亿参数的开源查询编码器，基于 AfriE5-Large-instruct 通过跨语言分数蒸馏训练而成。冻结的基础模型对每个英语训练问题与其答案段落及七个难负例进行评分。学生模型学习从通过往返翻译校验的 NLLB-200 契维语译文以及原始英语问题中复现这些分数，从而保持……

    arXiv:2611.32630v1 Announce Type: cross  Abstract: Most information sought online by Twi (Akan) speakers is written in English, making cross-lingual retrieval essential. We evaluate eight systems: BM25, five open embedding models, and two proprietary models (gemini-embedding-001 and OpenAI text-embedding-3-large) on two Twi-English tasks. AfriQA contains 1,221 natively written Asante Twi questions that must retrieve their answering English Wikipedia passages. Bible translation retrieval requires a Twi passage to retrieve its English translation from 6,195 candidates.   We introduce MargEmbed-Akan, a 560M-parameter open query encoder trained from AfriE5-Large-instruct using cross-lingual score distillation. The frozen base model scores each English training question against its answer passage and seven hard negatives. The student learns to reproduce these scores from both an NLLB-200 Twi translation that passes a round-trip translation check and the original English question, preserving
    
[^35]: REBEAL：人工参考译文已不再是最佳翻译，而评价指标却浑然不觉

    REBEAL: Human References Are No Longer the Best Translation, and Metrics Do Not Notice

    [https://arxiv.org/abs/2611.32623](https://arxiv.org/abs/2611.32623)

    该论文提出REBEAL基准，通过对参考译文施加七类受控污染来测试评价指标，发现人工参考译文质量已逊于顶级机器翻译系统，而现有八种主流指标均无法察觉这种污染，且各类型指标以不同方式失效。

    

    基于参考译文的机器翻译评价假设人工参考译文是正确的，且优于其所评分的系统。如今这两个假设均已失效，因为在2024年WMT MQM评测的所有活动中，人工参考译文均未排名第一，且在其19%至87%的句子片段中带有已标注的错误。面对有缺陷的参考译文，评价指标会如何表现，这是一个关于指标本身的问题，早在参考译文质量受到质疑之前，这个问题就值得被提出。我们提出了REBEAL（Reference Bias EvALuation，参考译文偏差评测），这是一个探究机器翻译评价指标如何应对七类受控参考译文污染的基准测试，污染类型源自上述评测活动所记录的错误类别。在八种广泛使用的评价指标和三个语言对上的实验表明，近期基于学习的指标（xCOMET、MetricX）比其前代更为稳健，但每一类指标都以各自的方式失效：表层指标缺乏长度敏感性，因此污染会隐藏在较长的句子片段中，而基于学习的指标则会接受与原文无关的……（原文在此处截断）

    arXiv:2611.32623v1 Announce Type: cross  Abstract: Reference-based MT evaluation assumes the human reference is correct and better than the systems it scores. Both assumptions now fail, since the human reference ranks first in none of the 2024 WMT MQM campaigns and carries annotated errors in 19% to 87% of its segments. What a metric does with a flawed reference is a question about the metric, and it would have been worth asking before reference quality came into doubt. We introduce REBEAL (Reference Bias EvALuation), a benchmark that probes how MT metrics respond to seven types of controlled reference contamination, drawn from the error categories those campaigns record. Across eight widely used metrics and three language pairs, recent learned metrics (xCOMET, MetricX) prove more robust than their predecessors, but each family fails in its own way. Surface metrics show no length sensitivity, so contamination hides in longer segments, while learned metrics accept references in unrelate
    
[^36]: 当“中性”意味着无关：对NormAd-ETI基准的结构效度审计

    When "Neutral" Means Irrelevant: A Construct-Validity Audit of NormAd-ETI

    [https://arxiv.org/abs/2611.32614](https://arxiv.org/abs/2611.32614)

    本文审计了NormAd-ETI基准的结构效度，发现其三方评分混淆了“规范是否适用”与“行为是否符合规范”这两个决策，导致在仅给出国家名称时适用性判断接近随机水平却被评分掩盖，并建议采用专门的强制选择题探针将两者分离。

    

    NormAd-ETI基准在文化背景从具体规则缩减为国家名称时，评估模型对日常行为作出“可接受、不可接受或中性”判断的能力。本文审计了这一三方评分实际测量的是什么。该评分结合了两个决策：所提供的规范是否适用，以及行为是否符合该规范。将两者分离后显示，当提示词中只给出国家名称时，适用性判断降至接近随机水平，而符合性判断仍远高于随机水平，原因在于提示词省略了“中性”标签所依赖的规则。由于一个答案同时承载这两个决策，评分会跟随更稳定的符合性判断，而掩盖较弱的适用性判断，且没有任何单一提示词能将两者分别恢复出来。专门的强制选择题探针能够对前沿模型实现两个决策的分离，但对小型开源模型则无效——在这些模型上，适用性判断仍然追踪着隐藏的可接受性标签。添加书面理由说明可以减少这种混淆，但无法完全消除。我们建议报告……（原文在此处截断）

    arXiv:2611.32614v1 Announce Type: new  Abstract: NormAd-ETI scores a model's judgment of everyday actions as acceptable, unacceptable, or neither as the cultural context shrinks from a specific rule to a country name. We audit what that three-way score measures. It combines two decisions: whether a supplied norm applies, and whether the action conforms to it. Separating them shows that when only a country is shown, applicability falls to near chance while conformity stays well above it, because the prompt withholds the rule a Neutral label depends on. Because one answer carries both decisions, the score follows the steadier conformity and hides the weaker applicability, and no single prompt recovers the two separately. Dedicated forced-choice probes separate them for frontier models but not for small open models, where the applicability judgment still tracks the hidden acceptability label. Adding a written rationale reduces that conflation without removing it. We recommend reporting th
    
[^37]: 隐私披露控制基准：LLM 在上下文披露粒度上能够忠实遵守隐私约束到什么程度？

    Privacy Disclosure Control Benchmark (PDC-Bench): How Faithfully Do LLMs Keep Privacy Constraints on In-Context Disclosure Granularity?

    [https://arxiv.org/abs/2611.32610](https://arxiv.org/abs/2611.32610)

    提出了 PDC-Bench，首个将隐私披露控制形式化为知识图谱层级上限约束的基准（16,105 个样本，覆盖地理、年龄、种族与民族和职业），用于评估 LLM 能否忠实遵守上下文中关于披露粒度的隐私约束。

    

    在 LLM 任务执行中，输出有时会被施加关于个人属性允许披露粒度的约束。例如，如果某人的隐私属性是其所在位置（旧金山），且只允许在国家层级进行披露，那么在州层级（加利福尼亚）的输出就违反了该约束。检测此类违规不仅需要捕获直接的隐私泄露（旧金山），还需要捕获语义泄露（加利福尼亚）。然而，现有基准无法衡量相对于给定语义约束（如国家层级限制）的泄露程度。因此，我们将披露控制形式化为对知识图谱层级中所允许上限的约束。基于这一形式化，我们构建了 PDC-Bench，一个包含 16,105 个样本的基准，涵盖地理、年龄、种族与民族以及职业。在将 PDC-Bench 应用于 11 个 LLM 的实验中，62% 的……（原文摘要在此处被截断）

    arXiv:2611.32610v1 Announce Type: cross  Abstract: In LLM task execution, a constraint on the permitted disclosure granularity of personal attributes is sometimes imposed on the output. For example, if a person's private attribute is their location (San Francisco) and disclosure is permitted only at the country level, an output at the state level (California) violates the constraint. Detecting such violations requires capturing not only direct privacy leakage (San Francisco) but also semantic leakage (California). However, existing benchmarks cannot measure the degree of leakage with respect to a given semantic constraint such as the country-level limit. We therefore formalize disclosure control as a constraint on the permitted ceiling over the hierarchy of a knowledge graph. Building on this formulation, we construct PDC-Bench, a benchmark of 16,105 samples covering geography, age, race and ethnicity, and occupation. In our experiments applying PDC-Bench to eleven LLMs, 62% of the con
    
[^38]: 门控概念适配器：用于可审计文本理解的分层概念瓶颈

    Gated Concept Adapters: Layer-Wise Concept Bottlenecks for Auditable Text Understanding

    [https://arxiv.org/abs/2611.32600](https://arxiv.org/abs/2611.32600)

    提出了门控概念适配器（GCA）这一分层概念瓶颈框架，通过在预训练Transformer的选定层附加门控低维概念读出适配器，将隐藏状态转化为可检查、可干预的概念中介接口，从而实现可审计的文本理解。

    

    面向人类的文本理解系统被用于推断情感、审核有害内容以及支持社会敏感的决策。然而，Transformer的预测难以审计，因为其预测产生于高维隐藏状态，无法为人类分析或干预提供紧凑、可检查的接口。我们提出了门控概念适配器（Gated Concept Adapters, GCA），这是一种分层概念瓶颈框架，它将带门控的、低维的概念生成读出适配器附加到预训练Transformer的选定输出上。词元表示被映射为概念激活，经学习到的门控过滤，汇聚为层级别的摘要，并作为概念中介接口暴露给预测头。这些维度可以通过使用统计、特征性n-gram、事后概念命名以及针对性的概念干预来进行检查。我们在细粒度情感分类、毒性检测等任务上对GCA进行了评估……

    arXiv:2611.32600v1 Announce Type: cross  Abstract: Human-facing text understanding systems are used to infer affect, moderate harmful content, and support socially sensitive decisions. However, Transformer predictions are difficult to audit because they are produced from high-dimensional hidden states that do not provide a compact, inspectable interface for human analysis or intervention. We introduce Gated Concept Adapters (GCA), a layer-wise concept bottleneck framework that attaches gated, low-dimensional concept-producing readout adapters to selected outputs of pretrained Transformers. Token representations are mapped into concept activations, filtered by learned gates, pooled into layer-level summaries, and exposed to the prediction head as a concept-mediated interface. The dimensions can be inspected through usage statistics, signature n-grams, post-hoc concept names, and targeted concept interventions. We evaluate GCA on fine-grained emotion classification, toxicity detection, t
    
[^39]: Avalon-M1：基于集体奖励的强化学习

    Avalon-M1: Reinforcement Learning from Collective Rewards

    [https://arxiv.org/abs/2611.32539](https://arxiv.org/abs/2611.32539)

    Avalon-M1提出基于集体奖励的强化学习（RLCR），联合训练CEO与最多八名专家共享同一策略的智能体组织，在延迟约束下将IMO-ProofBench-Advanced分数从23.8%提升至39.0%，并以约单智能体三分之一的延迟在12个基准上取得相当的平均性能。

    

    我们提出了基于集体奖励的强化学习，这是一种在延迟约束下将语言模型联合训练为智能体组织的方法。一个CEO和最多八名专家共享同一个策略，通过消息板和工作空间进行协调，并使用由共同结果派生的优势进行训练。截止期限使得并行工作在能够增加时间上限前所获收益时变得有价值。我们在147个环境中训练了Avalon-M1-Mini（120B）。对于训练后的模型，八人雇佣组织执行相对于单智能体执行，将IMO-ProofBench-Advanced分数从23.8%提升至39.0%。在12个基准测试中，组织以约等于token匹配的单智能体执行三分之一、以及八次独立尝试后选择的一半多的延迟，取得了相似的平均分数。训练后的模型还优于基于相同基础模型后训练的Nemotron 3 Super。

    arXiv:2611.32539v1 Announce Type: new  Abstract: We introduce reinforcement learning from collective rewards (RLCR), a recipe for jointly training language models to act as agent organisations under latency constraints. A CEO and up to eight specialists share one policy, co-ordinate through a message board and workspace, and are trained with a common outcome-derived advantage. The deadline makes parallel work valuable when it increases the return earned before the cap. We train Avalon-M1-Mini (120B) across 147 environments. For the trained model, eight-hire organisational execution raises IMO-ProofBench-Advanced scores from 23.8% to 39.0% relative to single-agent execution. Across 12 benchmarks, organisations achieve similar average scores at approximately one-third the derived latency of token-matched single-agent execution and just over half that of eight independent attempts followed by selection. The trained model also outperforms Nemotron 3 Super, post-trained from the same base m
    
[^40]: 如果中文被拉丁化了会怎样？关于输入表示、分词与语言建模的反事实研究

    What If Chinese Were Latinized? A Counterfactual Study of Input Representation, Tokenization, and Language Modeling

    [https://arxiv.org/abs/2611.32525](https://arxiv.org/abs/2611.32525)

    该研究通过反事实实验将中文语料转换为拼音并从头训练分词器与Llama风格语言模型，发现拼音表示会导致更高的切分率、更差的同音字消歧和语言建模性能，证明书面输入表示的选择并非中性编码，而会实质性影响语言建模效果。

    

    如果中文按照拉丁化新文字运动所提议的那样被拉丁化，这对中文自然语言处理意味着什么？同音字和声调标注使这一问题变得复杂。我们将中文语料转换为按音节分隔的拼音，并从头训练superBPE分词器。所得的拼音词表不仅出现了预期中的同音字冲突，其切分率还比汉字高约10%。而采用另一种BPE配置则会改变两种表示之间的相对切分率。我们进一步在汉字和拼音上分别预训练Llama风格的语言模型。拼音模型在单字困惑度、同音字消歧以及ZhoBLiMP整体最小对准确率上均表现更差。一项探索性的音系学探测显示，两种表示的性能都接近随机水平。更广泛的启示是：书面输入表示并非中性的编码方式，表示方案的选择会影响语言建模的效果。

    arXiv:2611.32525v1 Announce Type: new  Abstract: If Chinese had been Latinized, as the Latinxua Sin Wenz movement proposed, what would it have meant for Chinese NLP? Homophones and tone marking complicate the answer. We convert Chinese corpora to syllable-separated Pinyin and train superBPE tokenizers from scratch. The resulting Pinyin vocabularies not only exhibit expected homophone collisions, but also show approximately 10% higher fertility than characters. An alternative BPE configuration changes the relative fertility of the two representations. We further pretrain Llama-style models on Chinese characters and Pinyin. The Pinyin models perform worse on per-character perplexity, homophone disambiguation, and overall ZhoBLiMP minimal-pair accuracy. An exploratory phonological probe yields near-chance performance for both representations. The broader implication is that written input representations are not merely neutral encodings. Choices for representations affect language modeling
    
[^41]: STAR：用于视觉在线策略蒸馏的反事实双世界推演与空间目标自适应重建

    STAR: Counterfactual Dual-World Rollouts and Spatial Target Adaptive Reconstruction for Visual On-Policy Distillation

    [https://arxiv.org/abs/2611.32472](https://arxiv.org/abs/2611.32472)

    STAR通过反事实双世界推演与空间注意力引导的目标自适应重建，迫使视觉语言模型在在线策略自蒸馏中真正依赖决定答案的局部视觉证据，而非语言先验和全局上下文。

    

    视觉语言模型往往依赖全局上下文和语言先验，而忽视决定答案的局部细节。特权视觉在线策略自蒸馏使用一个冻结的教师模型（其可查看高分辨率证据裁剪图）来监督学生的全图推演。然而，我们的跨尺度诊断显示，当局部证据被完全遮蔽时，模型仍能保持其全图准确率的一半以上，这表明单张图像上的正确性并不能保证模型真正依赖局部证据；同时，对视觉校正的需求也随各个token及其空间注意力状态的不同而变化。我们提出STAR，这是一个视觉在线策略自蒸馏框架，结合了独立的反事实双世界推演与空间注意力引导的目标重建。局部图像编辑构建出具有相同问题和背景、但决定答案的关键事实不同的配对世界，其中共享的学生模型独立采样轨迹……

    arXiv:2611.32472v1 Announce Type: cross  Abstract: Vision-language models often rely on global context and language priors while overlooking answer-determining local details. Privileged visual on-policy self-distillation supervises full-image student rollouts using a frozen teacher that views high-resolution evidence crops. Yet our cross-scale diagnostics show that models retain over half of their full-image accuracy when local evidence is completely masked, indicating that single-image correctness does not guarantee reliance on local evidence, while the need for visual correction also varies across tokens with their spatial attention states. We propose STAR, a visual on-policy self-distillation framework combining independent counterfactual dual-world rollouts with spatial-attention-guided target reconstruction. Local image editing constructs paired worlds with identical questions and backgrounds but distinct answer-determining facts, where a shared student samples trajectories indepe
    
[^42]: BCP：一种用于在线手语识别与翻译的轻量级边界控制框架

    BCP: A Lightweight Boundary-Controlled Framework for Online Sign Language Recognition and Translation

    [https://arxiv.org/abs/2611.32471](https://arxiv.org/abs/2611.32471)

    BCP是一个轻量级边界控制框架，通过冻结预训练离线模型并引入由CTC对齐监督的边界适配器，将其高效转换为在线手语识别与翻译方法，避免了昂贵的整体模型重训练。

    

    在线手语识别（SLR）和手语翻译（SLT）在手语视频到来时增量地生成词汇序列或自然语言文本，这一能力对于实时通信和实时字幕至关重要。然而，强大的预训练离线模型无法直接部署在此场景中，而现有的在线方法通常需要在由离线视频构建的模拟流式前缀上重新训练整个模型，成本高昂。为应对这一挑战，我们提出了边界控制前缀提交（BCP），这是一个统一的轻量级框架，可将预训练的离线方法转换为手语到词汇（S2G）、手语到文本（S2T）以及手语到词汇再到文本（S2G2T）的在线对应方法。具体而言，BCP冻结预训练模型，并引入一个轻量级的边界适配器来决定何时提交结果，该适配器由来自预训练SLR模型的连接主义时序分类（CTC）对齐进行监督，而无需…

    arXiv:2611.32471v1 Announce Type: new  Abstract: Online sign language recognition (SLR) and translation (SLT) incrementally produce gloss sequences or natural language text from sign video as it arrives, a capability essential for real-time communication and live captioning. However, powerful pretrained offline models cannot be directly deployed in this setting, while existing online methods typically retrain the entire model on simulated streaming prefixes constructed from offline videos, incurring high costs. To address this challenge, we propose Boundary-Controlled Prefix Commit (BCP), a unified lightweight framework that converts pretrained offline methods into online counterparts for sign-to-gloss (S2G), sign-to-text (S2T), and sign-to-gloss-to-text (S2G2T). Specifically, BCP freezes the pretrained model and introduces a lightweight boundary adapter that decides when to commit, supervised by Connectionist Temporal Classification (CTC) alignments from a pretrained SLR model, withou
    
[^43]: MAGPIE：通过可执行场景构建实现的智能体式多参考图像生成

    MAGPIE: Agentic Multi-Reference Image Generation via Executable Scene Construction

    [https://arxiv.org/abs/2611.32462](https://arxiv.org/abs/2611.32462)

    MAGPIE将多参考图像生成的构图意图表示为可执行的Blender 3D场景程序，并由编码智能体在图像合成前自动检查、链接参考并迭代修正场景，从而实现更可靠的多参考图像生成。

    

    多参考图像生成要求模型在根据指令对所选主体进行布局的同时，保留这些主体的外观。尽管基于参考的生成与组合控制方面近期取得了进展，但如何可靠地将所选主体组织到预期场景中仍然具有挑战性。虽然现有方法提供了各种形式的组合指导，但所规划的关系并不总能在图像合成之前被显式实例化以供检查和修正。为解决这一局限，我们提出了MAGPIE，它将预期的构图表示为由可执行的Blender程序定义的简化3D场景。一个编码智能体将每个场景实例与其参考图像相链接，根据指令检查几何结构和渲染视图，并修订程序以纠正偏差。最终的代理渲染结果作为图像生成器的视觉规划。利用MAGPIE-Instruct-3K数据集，我们训练了……

    arXiv:2611.32462v1 Announce Type: cross  Abstract: Multi-reference image generation requires a model to preserve the appearance of selected subjects while arranging them according to an instruction. Despite recent progress in reference-based generation and compositional control, reliably organizing selected subjects into the intended scene remains challenging. Although existing methods provide various forms of compositional guidance, the planned relationships are not always explicitly instantiated for inspection and refinement before image synthesis. To address this limitation, we introduce MAGPIE, which represents the intended composition as a simplified 3D scene defined by an executable Blender program. A coding agent links each scene instance to its reference, checks the geometry and rendered view against the instruction, and revises the program to correct discrepancies. The final proxy rendering serves as a visual plan for the image generator. Using MAGPIE-Instruct-3K, we train the
    
[^44]: 音素身份深度剖面无法可靠预测构音障碍语音识别中LoRA的放置位置

    Phoneme-identity depth profiles do not reliably predict LoRA placement for dysarthric ASR

    [https://arxiv.org/abs/2611.32455](https://arxiv.org/abs/2611.32455)

    本研究通过在两个ASR模型和两个构音障碍语料库上的统一受控实验证明，基于音素可解码性深度剖面来选择LoRA适配器放置位置的方法并不可靠，音素错误曲线的最小值位置虽然相对稳定，但无法有效指导最佳适配带的选择。

    

    将语音基础模型参数高效地适配到构音障碍语音时，需要做出一个很少被充分论证的选择：应该适配哪些编码器层。最近提出的一种策略是探测编码器，找出语音（音素）信息最可解码的深度位置，并将适配器放置在那里。我们在两个ASR模型（Qwen3-ASR、Whisper-small）和两个构音障碍语料库（中文CDSD、英文UASpeech）上测试了这一推断，同时保持训练配方和随机种子纪律一致。该推断被证明是失败的。在统一流水线下测量（而非继承自已发表的数据），所有四条音素错误曲线均在相对深度的83%至92%之间达到最小值，且每个模型的最小化层在语料库之间的变化幅度不超过探测种子之间的变化幅度。然而最佳适配带并非同样恒定：从音素错误曲线到适配带的三种自然映射方式——最陡改进位置、最低带均值错误或拐点位置——没有一个能够恢复出最佳测试配置。

    arXiv:2611.32455v1 Announce Type: new  Abstract: Parameter-efficient adaptation of a speech foundation model to dysarthric speech requires a rarely justified choice: which encoder layers to adapt. A recently proposed strategy is to probe the encoder, find where phonetic information becomes most decodable, and put the adapters there. We test that inference across two ASR models (Qwen3-ASR, Whisper-small) and two dysarthric corpora (Mandarin CDSD, English UASpeech), holding recipe and seed discipline constant.   The inference fails. Measured under one pipeline rather than inherited from published figures, all four phoneme-error curves minimise between 83% and 92% of relative depth, each model's minimising layer moving no further between corpora than between probe seeds. The best adaptation band is not similarly constant: none of three natural mappings from the curve to a band -- steepest improvement, lowest band-mean error, or knee location -- recovers the best tested configuration. We d
    
[^45]: 当墨水遇上像素：面向人工修正金融文档的基于视觉语言模型的意图感知OCR

    When Ink Meets Pixel: Intent-Aware OCR with VLMs for Human-Corrected Financial Documents When Ink Meets Pixel: Intent-Aware OCR with VLMs for Human-Corrected Financial Documents

    [https://arxiv.org/abs/2611.32446](https://arxiv.org/abs/2611.32446)

    该论文提出“意图感知OCR”新任务，构建了包含1,600份金融文档、5,700余个视觉意图标注的DocIntent数据集及IntentBench评测基准，并训练出DocIntentOCR视觉语言模型，使OCR系统能够理解文档上的人工修正标记（如箭头、删除线等）并生成修正后的转录文本。

    

    文档上的人工标注（如箭头、插入符号、删除线和插入内容）表达了修正意图，然而现有的OCR和文档理解系统将这些标记视为视觉噪声而非语义编辑操作。我们提出了意图感知OCR（intent-aware OCR）这一新任务，即通过联合推理原始文档内容和传达编辑意图的视觉意图线索（手写或数字形式）来生成修正后的转录文本。为支持这一任务设定，我们提出了DocIntent，一个包含1,600份金融文档的多模态数据集，其中包含超过5,700个标注的视觉意图线索，每个线索均与布局感知、意图解析后的OCR目标相配对。我们进一步引入了IntentBench，一个包含216份文档的基准测试集，旨在评估模型在多样化真实世界标注条件下的鲁棒性。利用DocIntent，我们在直接意图监督下针对修正后的转录文本微调了DocIntentOCR，这是一套包含30亿和70亿参数规模的视觉语言模型。

    arXiv:2611.32446v1 Announce Type: new  Abstract: Human annotations on documents (such as arrows, carets, strikethroughs, and insertions) express corrective intent yet existing OCR and document understanding systems treat these marks as visual noise rather than semantic edit operations. We introduce intent-aware OCR, the task of generating corrected transcriptions by jointly reasoning over original document content and visual intent cues (handwritten or digital) that convey edit intent. To support this setting, we present DocIntent, a multimodal dataset of 1,600 financial documents with over 5,700 annotated visual intent cues, each paired with layout-aware, intent-resolved OCR targets. We further introduce IntentBench, a 216-document benchmark designed to evaluate model robustness under diverse real-world annotation conditions. Leveraging DocIntent, we fine-tune DocIntentOCR, a suite of 3B and 7B parameter vision-language models, under direct intent supervision on corrected transcriptio
    
[^46]: 因子化上下文注意力：从模拟探测到原生网格探测

    Factorized Context Attention: From Emulated to Native Grid Probing

    [https://arxiv.org/abs/2611.32423](https://arxiv.org/abs/2611.32423)

    提出了免训练的块稀疏注意力方法 FCA，将网格探测这种“模拟注意力”转化为模型原生的注意力机制，只需对输入编码一次进 KV 缓存，即可同时兼顾速度、精度与上下文长度三方面。

    

    模型能够在一次前向传播中读取一本书、一小时的视频或一张图像。随着上下文的增长，这一前向传播会变慢、损失精度，并受限于上下文窗口的长度。稀疏注意力虽然高效，但仍然保留了另外两个局限性。网格探测是一种近期的免训练方法，它通过让冻结模型针对输入的短局部跨度和跨步跨度询问相同的问题，从而恢复精度并扩展可处理的范围；但每个问题都会重复整个计算，因为每个跨度都需要重新编码。我们的洞察是：网格探测本质上是一种“模拟注意力”，即通过提示词在模型外部重现的注意力模式，因为每个探测只是限制了问题能够看到输入的哪一部分。我们提出了因子化上下文注意力，这是一种免训练的块稀疏注意力，它以原生方式运行相同的模式，作为模型自身的注意力。FCA 将输入一次性编码进 KV 缓存，以掩码序列的形式读取所有跨度……（原文摘要到此截断）

    arXiv:2611.32423v1 Announce Type: cross  Abstract: Models read a book, an hour of video, or an image in one forward pass. As the context grows, this pass slows down, loses accuracy, and stops at the context window. Sparse attention is efficient, but keeps the other two limitations. Grid probing, a recent training-free method, recovers the accuracy and extends the reach by asking a frozen model the same question about short local and strided spans of the input, but every question repeats the whole computation, since each span is encoded again. Our insight is that grid probing is emulated attention, an attention pattern reproduced from outside the model through prompts, since each probe only restricts which part of the input the question can see. We introduce Factorized Context Attention (FCA), a training-free block-sparse attention that runs the same pattern natively, as the attention of the model itself. FCA encodes the input once into a KV cache, reads all spans as masked sequences in
    
[^47]: 论掩码扩散语言模型中隐藏状态精炼的重要性

    On the Importance of Hidden-State Refinement in Masked Diffusion Language Models

    [https://arxiv.org/abs/2611.32416](https://arxiv.org/abs/2611.32416)

    该论文揭示了掩码扩散语言模型中隐藏状态的逐层与跨步骤精炼动态，并据此提出跨步骤循环精炼（CSRR）机制，通过在去噪步骤间传递隐藏状态信息来增强模型性能。

    

    掩码扩散语言模型（MDLM）通过迭代去噪被掩码或损坏的词元序列来生成文本，使词元预测能够以双向上下文为条件并并行更新多个位置。尽管每个去噪步骤都会产生丰富的逐层隐藏状态轨迹，但先前的工作通常只关注词元级别的信息而丢弃这些连续的隐藏状态，使其精炼动态尚未得到充分探索。在这项工作中，我们分析了MDLM中的隐藏状态动态，并发现了逐层和跨步骤精炼的明确证据：相邻层之间的平滑演化、更深层具有更强的预测能力、移除Transformer块后的渐进退化，以及连续去噪步骤之间最终状态的强对齐。基于这些观察，我们提出了跨步骤循环精炼（CSRR）机制，该机制在去噪步骤之间传递隐藏状态信息以增强模型性能。

    arXiv:2611.32416v1 Announce Type: new  Abstract: Masked diffusion language models (MDLMs) generate text by iteratively denoising masked or corrupted token sequences, allowing token predictions to condition on bidirectional context and update multiple positions in parallel. While each denoising step produces a rich layer-wise hidden-state trajectory, prior work typically focuses only on token-level information and discards these continuous hidden states, leaving their refinement dynamics underexplored. In this work, we analyze hidden-state dynamics in MDLMs and find clear evidence of layer-wise and cross-step refinement: smooth evolution across adjacent layers, greater predictive power in deeper layers, gradual degradation under transformer block removal, and strong final-state alignment across consecutive denoising steps. Motivated by these observations, we propose Cross-Step Recurrent Refinement (CSRR), a mechanism that carries hidden-state information across denoising steps to enhanc
    
[^48]: ASTERISM：多模态大语言模型真的会看所有图像吗？

    ASTERISM: Do Multimodal LLMs Actually Look at All the Images?

    [https://arxiv.org/abs/2611.32316](https://arxiv.org/abs/2611.32316)

    该研究审计了19个多图像基准，发现约24%的题目存在“单图捷径”问题（仅凭一张图像即可作答），并据此构建了目前规模最大的经审计多图像基准ASTERISM（1.87万道题），专门保留真正需要多张图像才能解答的题目。

    

    多图像基准旨在测试多模态大语言模型（MLLM）能否跨图像整合证据，但其中许多题目仅凭单张图像即可回答。通过对来自19个公开基准的约3.3万道题目进行审计，并采用两阶段单图过滤器，我们发现其中24.2%的题目仅凭一张图像或仅凭问题本身即可解答。在这些题目上，MLLM仅凭一张图像就能获得全部图像相对纯文本所带来的准确率提升的中位数78%，因此这些题目实际上奖励了“从单张图像作答”的行为，我们称之为单图捷径（single-image shortcut）。即便在单张图像不足以作答的题目上，MLLM也会走这条捷径：在这类题目上，它们的大部分错误答案与其仅凭单张图像时给出的答案重复，且该比例远高于随机水平。我们基于这19个基准构建了ASTERISM——据我们所知是规模最大的经过审计的多图像基准——仅保留其中需要多张图像才能作答的题目。其1.87万道题目（其中2000道经过修正或重写）共享同一（摘要在此处截断）。

    arXiv:2611.32316v1 Announce Type: cross  Abstract: Multi-image benchmarks aim to test whether a multimodal large language model (MLLM) combines evidence across images, but many of their items can be answered from a single image. Across about 33k items from 19 public benchmarks, our audit and a two-stage single-image filter find that one image or the question alone settles 24.2% of them. On these items, MLLMs recover from one image a median 78% of the accuracy that all images add over the text alone, so these items reward answering from one image, which we call the single-image shortcut. MLLMs take this shortcut even on items where one image is not enough. On such items, most of their wrong answers repeat an answer they give from one image, at a rate well above chance. We build ASTERISM, to our knowledge the largest audited multi-image benchmark, from these 19 benchmarks by keeping their items that need more than one image. Its 18.7k items, 2.0k of them corrected or rewritten, share one
    
[^49]: ArabFake：一个用于阿拉伯语虚假新闻检测、分类和风险预测的多任务深度学习框架

    ArabFake: A multitask deep learning framework for Arabic fake news Detection, Categorization, and risk prediction

    [https://arxiv.org/abs/2611.32303](https://arxiv.org/abs/2611.32303)

    本文提出基于 MARBERTv2 的多任务深度学习框架 ArabFake，可同时完成阿拉伯语虚假新闻检测、内容分类和风险等级预测三项任务，并发布了包含 2,495 条专家验证标注新闻的 ArabFake 数据集。

    

    虚假新闻在包括社交媒体在内的阿拉伯语媒体中的传播，对信息的完整性和公众对信息的信任构成了巨大挑战。本文提出了一个名为 ArabFake 的综合性深度学习框架，超越了阿拉伯语虚假新闻检测的二元分类范畴。ArabFake 构建于 MARBERTv2（一种最先进的多方言阿拉伯语推文模型）之上，能够熟练应对阿拉伯语的复杂性，同时执行三个统一任务：虚假新闻检测、内容分类及其风险评估。该框架通过跨任务共享知识表示实现多任务学习，从而同时提升了效率和性能。为便于开发与评估，我们提出了 ArabFake 数据集，该数据集包含 2,495 条人工标注的新闻条目，其虚假新闻类别和风险等级标签均经过专家验证。

    arXiv:2611.32303v1 Announce Type: new  Abstract: The spread of fake news among Arabic media including social media represents a great challenge to the integrity of information and the trust of the public in it. In this paper, we introduce a comprehensive deep-learning framework, named ArabFake, that goes beyond the binary classification on Arabic fake news detection. ArabFake, built over MARBERTv2 (a state-of-the-art model for multi-dialectal Arabic tweets), proficiently address the complexity of the Arabic language while performing three unified tasks which are fake news detection, content categorization and its risk assessment. The framework promotes efficiency and performance both by enabling multi-task learning through shared knowledge representation across tasks. In order to facilitate development and evaluation, we present the ArabFake Dataset consisting of 2,495 manually labelled news items with labels that are verified by experts regarding fake news categories and risk levels. 
    
[^50]: 解除缓考阻塞：通过JAMN Bot——基于移动端RAG的政策助手支持学生

    Unjamming Deferrals: Supporting Students Through JAMN Bot, a Mobile RAG-Grounded Policy Assistant

    [https://arxiv.org/abs/2611.32266](https://arxiv.org/abs/2611.32266)

    本文提出了集成JAMN Bot的移动应用Schedulater，通过基于Ollama自托管Llama 3.2 3B的RAG管道，为大学缓考流程提供隐私保护的、基于政策的智能问答支持。

    

    在一些教育机构中，诸如考试缓考申请流程之类的工作流程可能缓慢、不便且不一致，这产生了实施新型数字系统的需求。本文提出了Schedulater，一款用于管理大学考试缓考申请的移动应用。JAMN Bot是集成到Schedulater中的由大语言模型（LLM）驱动的聊天机器人，提供上下文感知的支持，帮助用户理解缓考流程。该系统采用基于Ollama（Llama 3.2 3B）的检索增强生成（RAG）管道进行自托管推理，能够在不牺牲隐私的情况下提供基于政策的回答。我们使用一个包含43个项目的基准测试来评估该系统，涵盖缓考、隐私、排期、跨政策以及对抗性和超范围问题，并报告了人类评估者之间的评分者间一致性以及检索命中率指标。随后，我们采用以GPT-5.4作为评判的LLM-as-a-judge方法，揭示了……

    arXiv:2611.32266v1 Announce Type: new  Abstract: In some educational institutions, workflows such as the exam deferral process can be slow, inconvenient, and inconsistent, which creates the need to implement novel digital systems. This paper presents Schedulater, a mobile app for managing university exam deferrals. JAMN Bot is a Large Language Model (LLM)-powered chatbot integrated into Schedulater that provides context-aware support to help users understand the deferral process. The system features a Retrieval-Augmented Generation (RAG) pipeline using Ollama (Llama 3.2 3B) for self-hosted inference which enables policy-grounded responses without sacrificing privacy. We evaluate the system using a 43-item benchmark spanning Exam Deferral, Privacy, Scheduling, Cross-Policy, and Adversarial & Out-of-Scope questions, and report inter-rater agreement among our human evaluators alongside retrieval hit-rate metrics. Subsequently, we utilize an LLM-as-a-judge approach with GPT-5.4, revealing 
    
[^51]: ArEssays：迈向教育场景下鲁棒的阿拉伯语手写文本理解

    ArEssays: Towards Robust Arabic Handwritten Text Understanding in Educational Settings

    [https://arxiv.org/abs/2611.32211](https://arxiv.org/abs/2611.32211)

    提出大规模阿拉伯语学生手写作文数据集ArEssays，通过对34个视觉-语言和OCR模型的基准测试，揭示了现有模型在真实儿童手写内容上的显著性能差距。

    

    由于阿拉伯语文字的复杂性以及儿童书写的高度可变性，对手写阿拉伯语学生作文的自动理解仍然是一项挑战。我们提出了ArEssays，这是一个大规模数据集，包含来自埃及小学至中学学生的真实作文，并同时提供逐字转录版本和语言修正版本。我们对34个视觉-语言模型和OCR模型（包括经过微调的系统）进行了基准测试，发现现有模型在真实学生手写内容上存在较大的性能差距。我们发布了该数据集、训练流程和测试结果，以支持阿拉伯语OCR、文档理解和教育NLP领域的研究。

    arXiv:2611.32211v1 Announce Type: new  Abstract: Automated understanding of handwritten Arabic student essays remains challenging due to script complexity and high variability   in children's writing. We introduce ArEssays, a large-scale dataset of authentic essays from primary to secondary students   across Egypt, with both verbatim transcriptions and linguistically corrected versions. We benchmark 34 vision-language and OCR   models, including fine-tuned systems, and find large performance gaps on real student handwriting. We release the dataset,   training pipelines, and results to support research in Arabic OCR, document understanding, and educational NLP.
    
[^52]: 可操控性设计：用于语言模型鲁棒控制的条件预训练

    Steerable by Design: Conditional Pretraining for Robust Control of Language Models

    [https://arxiv.org/abs/2611.32199](https://arxiv.org/abs/2611.32199)

    该论文提出了一种条件预训练方法和架构，通过编码器交叉注意力机制从一开始就将指令与数据通道分离，构建出Mancelle-3B这一在设计上即可被有效操控的语言模型，解决了模型误将数据当作指令处理的越狱与对齐漂移问题。

    

    我们的目标是构建在设计上就具备可操控性的语言模型。当今的语言模型采用下一词预测进行预训练，这将模型的对齐与控制推迟到训练最后1%的token或推理阶段的干预。这些模型将单一输入通道同时用于指令和数据，导致了常见的失败模式——模型错误地将数据当作指令处理，例如有意的越狱攻击和无意的上下文对齐漂移。为了解决这一问题，我们提出了一种从训练一开始就将两者分离的预训练方法和架构，由此产生了Mancelle-3B。一个编码器通过交叉注意力机制处理后续文本的高层语义特征，并阻止token流向该控制通道回写。在预训练过程中，模型学会依赖这些特征来预测下一个token，从而将该通道转变为操控模型的有效抓手。

    arXiv:2611.32199v1 Announce Type: cross  Abstract: Our goal is to build language models that are steerable by design. Today's language models are pretrained with next-token prediction, which relegates alignment and control to the final 1% of training tokens or inference-time interventions. These models repurpose a single input channel for both instructions and data, leading to common failure modes where the model mistakenly treats data as instructions such as intentional jailbreaks and unintentional in-context alignment drift. To correct this, we propose a pretraining method and architecture that separates the two from the very start, producing Mancelle-3B. An encoder processes high-level semantic features of upcoming text with cross-attention that prevents the token stream from writing back into this control channel. During pretraining, the model learns to rely on these features to predict the next token, turning the channel into an effective handle for steering the model. Mancelle-3B
    
[^53]: ThinkMorph-SSL：利用前置任务扩展交错式多模态推理

    ThinkMorph-SSL: Scaling Interleaved Multimodal Reasoning with Pretext Tasks

    [https://arxiv.org/abs/2611.32183](https://arxiv.org/abs/2611.32183)

    ThinkMorph-SSL 提出一种两阶段自监督训练方案，将图像上色、去噪等视觉前置任务转化为无需指令的交错式多模态推理轨迹，让图像本身充当监督信号，从而在相同指令微调预算下超越仅靠指令微调的模型。

    

    人们不仅用文字，也用图像进行推理。交错式多模态推理让模型能够以类似的方式解决问题：通过生成中间图像，并将其用于指导后续推理。训练具备这种能力的模型通常依赖于针对特定任务的示范数据，而这些数据难以收集。我们提出了 ThinkMorph-SSL，这是一种两阶段训练方案，它将图像本身作为监督信号的来源。第一阶段为前置推理，将视觉前置任务（如图像上色和去噪）转换为无需指令的交错式推理轨迹：模型先描述一张受损或经过变换的图像，再生成恢复或规范化后的图像，并对结果进行解读。第二阶段则使用自动构建的示范数据进行问题驱动的指令微调。在相同的指令微调预算下，经过前置推理训练的模型优于仅使用指令微调训练的模型。完整的训练方案实现了……（原文摘要在此处截断）

    arXiv:2611.32183v1 Announce Type: cross  Abstract: People reason with pictures as well as words. Interleaved multimodal reasoning gives models a similar way to solve problems by creating intermediate images and using them to guide further reasoning. Training models with this ability usually relies on task-specific demonstrations, which are difficult to collect. We introduce ThinkMorph-SSL, a two-stage training recipe that uses images themselves as a source of supervision. The first stage, pretext reasoning, converts visual pretext tasks, such as image colorization and denoising, into instruction-free interleaved traces. Models describe a corrupted or transformed image, generate a restored or canonical image, and interpret the result. The second stage uses automatically constructed demonstrations for question-driven instruction tuning. At the same instruction-tuning budget, models trained with pretext reasoning outperform those trained with instruction tuning alone. The full recipe impr
    
[^54]: RightDiff：学习依据正确差异进行评判的生成式奖励模型

    RightDiff: Learning Generative Reward Models to Judge by the Right Differences

    [https://arxiv.org/abs/2611.32166](https://arxiv.org/abs/2611.32166)

    RightDiff通过在单次自回归生成中显式发现候选响应间的对比差异、区分实质性与非实质性因素并归纳针对响应对的评分规则，学习以“正确差异”为核心的生成式奖励模型，从而避免表面捷径与推理依据漂移。

    

    生成式奖励模型通常依赖通用的查询级评分规则或不受约束的推理依据，忽视了成对偏好实际上是由响应特定的差异所驱动的这一事实。此外，仅以结果正确性为优化目标会促使模型利用冗长性和格式等表面捷径，从而引发推理依据漂移。为解决这些问题，我们提出RightDiff，它将成对评估结构化为一条以证据为基础的评估轨迹，并在单次自回归生成中执行。RightDiff显式地发现候选响应之间的对比差异，将其归类为实质性或非实质性因素，并在最终裁决前归纳出针对该响应对的评分规则。为了在训练这一过程时避免事后合理化，我们通过响应隔离提示合成演示数据，并利用差异对齐奖励对奖励模型进行优化，同时以预算约束的一对一匹配进行正则化……

    arXiv:2611.32166v1 Announce Type: cross  Abstract: Generative reward models typically rely on generic query-level rubrics or unconstrained rationales, overlooking the fact that pairwise preference is driven by response-specific differences. Moreover, optimizing solely for outcome correctness encourages models to exploit superficial shortcuts such as verbosity and formatting, triggering rationale drift. To address this, we propose RightDiff, which structures pairwise evaluation into an evidence-grounded evaluation trajectory executed in a single autoregressive generation. RightDiff explicitly discovers contrastive differences between candidates, categorizes them into material or immaterial factors, and induces pair-specific rubrics before final adjudication. To train this process without post-hoc rationalization, we synthesize demonstration data via response-isolated prompts and optimize the reward model with a difference-alignment reward regularized by budget-constrained one-to-one mat
    
[^55]: 仅仅使评估显式化并不足够：在生成式奖励模型中监督评估过程

    Making Evaluation Explicit Is Not Enough: Supervising the Evaluation Process in Generative Reward Models

    [https://arxiv.org/abs/2611.32161](https://arxiv.org/abs/2611.32161)

    提出STEP-GRM框架，通过为评分标准构建提供结构性指导和为准则级判断提供局部监督，直接监督生成式奖励模型的中间评估过程，使模型不仅学会正确的最终偏好，还掌握更可靠的评估流程。

    

    生成式奖励模型（GRM）通过在预测最终偏好之前生成评论、评分标准或中间分析，使语言模型的评估过程更加显式。然而，显式评估并不一定意味着可靠的评估过程：在仅有结果监督的情况下，模型可能在产生薄弱或缺乏依据的中间判断的同时得出正确的最终偏好，从而使评估沦为一种事后合理化。我们提出了STEP-GRM，这是一个直接监督基于评分标准的GRM中间评估过程的框架。除了传统的偏好监督之外，STEP-GRM还为评分标准的构建提供结构性指导，并为准则级判断提供局部监督，鼓励模型不仅学习正确的最终偏好，还学习得出该偏好的更可靠过程。大量实验表明，STEP-GRM在标准奖励模型基准上取得了优异的表现。

    arXiv:2611.32161v1 Announce Type: cross  Abstract: Generative reward models (GRMs) make language-model evaluation more explicit by generating critiques, rubrics, or intermediate analyses before predicting a final preference. However, explicit evaluation does not necessarily imply a reliable evaluation process: under outcome-only supervision, a model may reach the correct final preference while producing weak or unsupported intermediate judgments, turning the evaluation into a post-hoc rationalization. We introduce STEP-GRM, a framework that directly supervises the intermediate evaluation process of rubric-based GRMs. In addition to conventional preference supervision, STEP-GRM provides structural guidance for rubric construction and local supervision for criterion-level judgments, encouraging the model to learn not only the correct final preference but also a more reliable process for reaching it. Extensive experiments show that STEP-GRM achieves strong performance on standard reward-m
    
[^56]: ARS-VLM：面向视觉-语言模型的视觉对齐增强随机搜索

    ARS-VLM: Visually Aligned Augmented Random Search for Vision-Language Models

    [https://arxiv.org/abs/2611.32147](https://arxiv.org/abs/2611.32147)

    提出ARS-VLM，一种视觉对齐的增强随机搜索零阶方法，解决了直接将进化策略类算法应用于视觉-语言模型时的图像块网格过拟合和灾难性遗忘两大失败模式，同时规避了GRPO等一阶强化学习带来的内存与系统开销。

    

    以群体相对策略优化（GRPO）为代表的一阶强化学习被广泛用于视觉-语言模型（VLM）的多模态推理后训练。然而，这类方法因反向传播、优化器状态和参考模型而带来沉重的内存与系统开销。仅前向传播的零阶（ZO）方法，如进化策略（ES）及其top-k变体增强随机搜索（ARS），通过奖励引导的权重扰动规避了这些开销，但在VLM推理领域尚未被探索。我们发现，直接将标准ES或原始ARS应用于VLM会出现两种根本性的失败模式：(i) 图像块网格过拟合，即模型拟合脆弱的网格特定视觉token伪影而忽略空间不变性；(ii) 灾难性遗忘，即由于无界的高秩参数与策略漂移导致通用感知能力和分布外（OOD）任务性能下降。我们提出了ARS-VLM，一种视觉对齐的增强随机搜索方法（摘要在此处截断）。

    arXiv:2611.32147v1 Announce Type: cross  Abstract: First-order reinforcement learning, such as Group Relative Policy Optimization (GRPO), is widely used to post-train Vision-Language Models (VLMs) for multimodal reasoning. However, it incurs heavy memory and system overhead from backpropagation, optimizer states, and a reference model. Forward-only zeroth-order (ZO) methods, such as Evolution Strategies (ES) and its top-k variant Augmented Random Search (ARS), avoid these overheads via reward-guided weight perturbations, yet remain unexplored for VLM reasoning. We find that directly applying standard ES or vanilla ARS to VLMs suffers from two fundamental failure modes: (i) patch-grid overfitting, fitting fragile grid-specific visual token artifacts while ignoring spatial invariants, and (ii) catastrophic forgetting, degrading general perception and out-of-distribution (OOD) tasks due to unbounded high-rank parameter and policy drift. We propose ARS-VLM, a visually aligned Augmented Ran
    
[^57]: 潜空间推理与通信的校准化读取

    Calibrated Reading of Latent Reasoning and Communication

    [https://arxiv.org/abs/2611.32143](https://arxiv.org/abs/2611.32143)

    本文提出校准潜读取器（CLR），一个紧凑的Transformer，通过任务条件化交互、源适配器迁移与保形校准，以比语言模型解码器低数百倍的计算成本准确读取潜推理结果与键值通信内容，并提供统计覆盖率与误报保证。

    

    潜推理与键值通信将中间信息存储在连续表示中，限制了对模型行为的直接检查。现有的激活读取器在任务特定探针的低成本与语言模型解码器的灵活性之间进行权衡。我们提出了校准潜读取器（Calibrated Latent Reader，CLR），这是一个紧凑的Transformer，通过状态与候选答案之间的任务条件化交互来预测推理结果和消息内容。源适配器支持跨模型迁移，而保形校准在可交换性条件下提供覆盖率与误报保证。在两个CODI模型上，CLR在最终答案预测和参考值隶属判断任务中比MLP和注意力探针的准确率提高了4-18个百分点。在CODI-GPT-2上，它以减少437-648倍推理FLOPs的代价，实现了与基于1.7B参数LatentQA的言语化器相当的准确率。在合成的Qwen3-4B通信上，比较（摘要在此处被截断）

    arXiv:2611.32143v1 Announce Type: cross  Abstract: Latent reasoning and key-value communication store intermediate information in continuous representations, limiting direct inspection of model behavior. Existing activation readers trade the low cost of task-specific probes against the flexibility of language-model decoders. We propose the Calibrated Latent Reader, CLR, a compact transformer that predicts reasoning outcomes and message content through task-conditioned interactions between states and candidate answers. Source adapters support transfer across models, while conformal calibration provides coverage and false-alarm guarantees under exchangeability. On two CODI models, CLR improves accuracy over MLP and attention probes by 4-18 percentage points for final-answer prediction and reference-value membership. On CODI-GPT-2, it achieves accuracy comparable to a 1.7B LatentQA-based verbalizer using 437-648 times fewer inference FLOPs. On synthetic Qwen3-4B communications, comparing 
    
[^58]: 修正行动而非思维：前沿大语言模型的精细微调

    Fix How It Acts, Not How It Thinks: Surgical Fine-Tuning of Frontier LLMs

    [https://arxiv.org/abs/2611.32064](https://arxiv.org/abs/2611.32064)

    提出一种“精细微调”方法，不增大更新规模，而是直接在权重更新合并后残存的部分上进行训练，从而在不干扰模型推理能力的前提下纠正其行动层面的小失误，并解决了 LoRA 小幅更新在合并到量化发布版本时大量丢失的问题。

    

    开放权重大语言模型是胜任的软件工程师，但它们有时会在本可以完成的任务上失败。这通常并非推理能力的缺陷，而是长执行轨迹中的几个小失误造成的差异（例如重复已经失败的步骤，或在需求尚未满足时就宣告任务完成）。纠正这一点需要对权重进行精细的更新，引导模型采取正确的行动而不干扰其推理过程。这类更新通常通过 LoRA 适配器学习得到，但往往过于微小，以至于在合并到发布的 checkpoint 时无法保留。在我们的 BF16 版本中，权重变化的中位数仅为单个舍入步长的 1.5%，合并后会损失约三分之二的更新范数。当同一个适配器合并到同一模型的 FP8/NVFP4 混合精度版本时，93% 的更新会消失。我们没有增大更新的规模，而是直接在合并后残存的部分上进行训练。我们引入……（摘要在此处被截断）

    arXiv:2611.32064v1 Announce Type: cross  Abstract: Open-weight LLMs are competent software engineers, but they sometimes fail at tasks they are otherwise able to do. Often it is not a failure of reasoning. It is a few small missteps over a long trajectory (repeating a step that has already failed, or declaring the task done while a requirement is still unmet) that make the difference. Correcting this calls for a surgical weight update that steers the model toward the right actions without interfering with its reasoning. Such updates, typically learned with LoRA adapters, are often too small to survive merging into a released checkpoint. In our BF16 release, the median weight change is only 1.5% of a single rounding step, and merging loses roughly two thirds of the update's norm. When the same adapter is merged into the mixed FP8/NVFP4 release of the same model, 93% of the update disappears. Instead of increasing the size of the update, we directly train on the surviving part. We introd
    
[^59]: 临床医生评分的示例能帮助大语言模型从长达一小时的转录文本中对ADOS-2进行编码吗？

    Do Clinician-Scored Examples Help LLMs Code ADOS-2 from Hour-Long Transcripts?

    [https://arxiv.org/abs/2611.32050](https://arxiv.org/abs/2611.32050)

    本研究通过预注册的安慰剂对照实验首次系统检验了专家评分示例对大语言模型编码ADOS-2的作用，发现正确标签带来的增益有限，模型与临床医生的一致性仍然较低（二次加权kappa仅0.22–0.30）。

    

    大语言模型可以将书面评分标准应用于转录文本，这使得对自闭症诊断观察量表（ADOS-2）进行自动化编码看起来触手可及。让此类评分器与临床医生对齐的一种常见方法是向其展示专家评分过的示例，但正确的标签是否能提高一致性却很少被检验。我们使用一个本地模型和预注册的、安慰剂对照的设计，针对48名儿童长达一小时的ADOS-2模块3会话中的八个基于言语的项目进行了测试。我们比较了少样本评分、锚定比较（即模型仅判断某儿童比已评分的锚点更好、相似或更差，再由固定规则给出分数）、事后校准、基于协议的证据路由以及对完整转录文本的直接评分。在40名留出测试儿童上，与临床医生的一致性较为有限，二次加权kappa值介于0.22到0.30之间。将每个示例的标签替换为错误的标签，会改变大约三分之一的少样本评分结果（摘要原文在此处截断）。

    arXiv:2611.32050v1 Announce Type: cross  Abstract: Large language models can apply a written rubric to a transcript, making automated coding of the Autism Diagnostic Observation Schedule (ADOS-2) look within reach. A common way to align such a rater with clinicians is to show it expert-scored examples, but whether correct labels improve agreement is rarely tested. We test this for eight speech-based items of hour-long ADOS-2 Module 3 sessions from 48 children, with one local model and a pre-registered, placebo-controlled design. We compare few-shot scoring, an anchored comparison in which the model only judges a child better, similar or worse than scored anchors and a fixed rule gives the score, post-hoc calibration, protocol-based evidence routing and direct scoring of the full transcript. On 40 held-out children, agreement with clinicians was modest, with quadratic weighted kappa between 0.22 and 0.30. Replacing every example's label with a wrong one changed about a third of the few-
    
[^60]: HuluOPD：利用无标注提示将异构医学专科模型统一为一个轻量级模型

    HuluOPD: Unifying Heterogeneous Medical Specialists into One Lightweight Model with Unlabeled Prompts

    [https://arxiv.org/abs/2611.32021](https://arxiv.org/abs/2611.32021)

    HuluOPD提出一种在线策略蒸馏方法，仅用无标注提示和跨分词器监督损失，就能把架构各异的多个医学专科视觉-语言模型统一蒸馏为一个轻量级统一模型，无需参考答案或专科模型的原始训练数据。

    

    专科医学视觉-语言模型（VLM）在其各自专科领域内表现出色，但难以统一：它们的训练数据通常是私有的，且独立开发的专科模型无法假定共享相同的架构。分别部署这些大型模型的成本很高。我们提出了HuluOPD，这是一种在线策略蒸馏方法，仅使用无标注提示就能将异构的专科模型统一为一个轻量级VLM，无需参考答案或专科模型的训练数据。轻量级模型对每个提示进行回答，并由该提示所属模态对应的专科模型监督其回答；一种与续写内容对齐的跨分词器损失（continuation-aligned cross-tokenizer loss）使得即使在两者架构和分词器不同的情况下也能实现这一点。我们将HuluOPD应用于五个专科模型，分别涵盖超声、病理、光学相干断层扫描（OCT）、眼底照相和牙科影像。为了进行蒸馏和评估，我们构建了包含91.4万条无标注提示的HuluOPDCorpus以及HuluOPD

    arXiv:2611.32021v1 Announce Type: cross  Abstract: Specialist medical vision-language models (VLMs) are strong within their specialties but hard to unify: their training data are often private, and independently developed specialists cannot be assumed to share an architecture. Deploying these large models separately is costly. We present HuluOPD, an on-policy distillation recipe that unifies heterogeneous specialists into one lightweight VLM using only unlabeled prompts, without requiring reference answers or the specialists' training data. The lightweight model answers each prompt, and the specialist of the prompt's modality supervises its response; a continuation-aligned cross-tokenizer loss makes this possible even when they differ in architecture and tokenizer. We apply HuluOPD to five specialists, for ultrasound, pathology, optical coherence tomography, fundus photography, and dental imaging. To distill and evaluate them, we build HuluOPDCorpus, 914k unlabeled prompts, and HuluOPD
    
[^61]: Marketing-Bench：评估前沿模型与智能体在真实世界营销能力上的表现

    Marketing-Bench: Evaluating Frontier Models and Agents on Real-World Marketing Capabilities

    [https://arxiv.org/abs/2611.32012](https://arxiv.org/abs/2611.32012)

    该论文提出了首个开源营销基准Marketing-Bench，通过将品牌异常检测（判别式任务）与修复（生成式任务）配对且保持答案可验证，从事实性、一致性、连贯性、结构和品牌语调五个维度评估AI模型与智能体的真实营销能力。

    

    随着前沿AI模型和智能体在现实世界中被更广泛地部署，营销正成为其最大的应用领域之一。营销团队使用AI模型来制作品牌资产，如网站文案和社交媒体帖子，而消费者的智能体则自主发现产品、调研购买决策并浏览网站。然而，评估营销能力十分困难，因为与软件开发、客户支持或法律审查等真值定义较为客观的领域不同，营销是主观的，其输出质量取决于品味和品牌调性。我们提出了Marketing-Bench，这是首个开源的营销基准，它将判别式任务（检测品牌异常）与生成式任务（修复问题）配对，同时保持答案可验证。Marketing-Bench测试五个关键维度：事实性、一致性、连贯性、结构和品牌语调，这些都是优质品牌内容的关键要素。

    arXiv:2611.32012v1 Announce Type: new  Abstract: As frontier AI models and agents are deployed more broadly in the real world, marketing is emerging as one of their largest applications. Marketing teams use AI models to produce brand assets like website copy and social media posts, while consumers' agents autonomously discover products, research purchases, and navigate websites. However, evaluating marketing capability is difficult because unlike areas like software development, customer support, or legal review where ground truth is more objectively defined, marketing is subjective, and output quality is judged based on taste and brand. We introduce Marketing-Bench, the first open-source marketing benchmark, which pairs discriminative tasks (detecting brand anomalies) with generative tasks (remediation), while keeping answers verifiable. Marketing-Bench tests five key pillars: factuality, consistency, coherence, structure, and brand voice, all crucial elements for good brand content o
    
[^62]: TableRouter：面向长表格问答的查询感知证据构建

    TableRouter: Query-Aware Evidence Construction for Long-Table Question Answering

    [https://arxiv.org/abs/2611.32006](https://arxiv.org/abs/2611.32006)

    TableRouter是一个无需训练的框架，通过分析查询在计算路线（从完整表格推导问题相关统计数据）和覆盖路线（检索相关行与列摘要）之间智能路由，在长表格问答任务中显著提升准确率（最高18.3个百分点）并将计算成本降低77-95%。

    

    LLM能够对表格进行推理，但当表格规模超出上下文预算时，仅检索相关行并不足以解决问题。检索到的子集可能遗漏精确聚合所需的信息；即使证据完整，LLM也可能难以有效利用。我们提出了TableRouter，这是一个无需训练的框架，它分析每个查询，然后在两条路线之间做出选择：计算路线，从完整表格中推导以问题为条件的统计数据；覆盖路线，检索相关行和列摘要。所选上下文通过一次调用传递给冻结的LLM。在三个LLM和三个大表格基准测试中，TableRouter在全部九个设置中以1.2-18.3个准确率点的优势超越了六个单次调用基线中最强的一个，其估计FLOPs比截断表格提示方法减少77-95%，且索引成本可在多个问题间摊销。这些增益还可以扩展到来自此前实验未使用的数据库中的长表格。

    arXiv:2611.32006v1 Announce Type: new  Abstract: LLMs can reason over tables, but when tables outgrow the context budget, retrieving relevant rows is only part of the solution. A retrieved subset may omit information required for an exact aggregate; even complete evidence may remain difficult for an LLM to use. We introduce TableRouter, a training-free framework that analyzes each query, then chooses between two routes: a computation route, which derives question-conditioned statistics from the full table, and a coverage route, which retrieves relevant rows and column summaries. The chosen context is passed to a frozen LLM in one call. Across three LLMs and three large-table benchmarks, TableRouter outperforms the strongest of six single-call baselines in all nine settings by 1.2-18.3 accuracy points, using 77-95% fewer estimated FLOPs than truncated-table prompting, with indexing amortized across questions. The gains extend to long tables from databases not used in earlier experiments
    
[^63]: 将语言模型锚定于受众显性行为

    Grounding Language Models in Revealed Audience Behavior

    [https://arxiv.org/abs/2611.31999](https://arxiv.org/abs/2611.31999)

    该论文提出将语言模型在受众真实的点击、点赞、观看记录上进行微调（即“受众显性行为锚定”），使同一个8B模型预测标题点击的准确率从约50%提升至86.5%，并证明“直接询问模型”与“行为锚定”之间的差距主要源于训练信号，而非模型规模或提示方式。

    

    语言模型正越来越多地被用于模拟受众对某个标题或帖子的反应，而此类模拟器通常以人们“说”的内容作为验证基准。我们则用人们“做”的事来验证它们。当一个模型在受众实际点击、点赞或观看的记录上进行微调时，我们称其为“锚定于受众显性行为”。我们不仅追问此前研究所发现的“直接询问模型不如行为锚定”这一现象是否成立，还进一步追问差距为何存在、存在于何处。为何存在：这一差距源于训练信号本身的特性，而非主要取决于模型规模或提示方式。同一个80亿参数（8B）模型，在被直接询问读者会点击哪个标题时，正确率约为50%；而经过点击行为锚定后，正确率达到86.5%；我们尝试的最强提示方法也只能与一个文本分类器相当；在我们的实验中，一个经过行为锚定的0.6B模型击败了所有采用提示方式的前沿模型；此外，在一项预先注册的检验中，将点击数据按计数还是按份额进行拟合，未检测到任何差异。（原文摘要在此处截断）

    arXiv:2611.31999v1 Announce Type: new  Abstract: Language models are increasingly used to simulate how an audience will respond to a headline or a post, and such simulators are validated against what people say. We validate them against what people do. We call a model grounded in revealed audience behavior when it is fine-tuned on records of what audiences actually clicked, liked or watched, and we ask not only whether asking a model falls short of grounding it, as prior work found, but why and where. Why: the gap is a property of the training signal, not mainly of model size or prompting. The same 8B model is right about half the time when asked which headline readers will click and 86.5% of the time when grounded in the clicks; the strongest prompting we tried only matches a text classifier; a grounded 0.6B model beats every prompted frontier model in our runs; and whether the clicks are fitted as counts or as shares makes no detectable difference in a pre-registered test. The asked 
    
[^64]: 在权重空间中读取、写入与修复博弈论策略

    Reading, Writing and Repairing Game-Theoretic Policies in Weight Space

    [https://arxiv.org/abs/2611.31979](https://arxiv.org/abs/2611.31979)

    该论文构建了一组在权重空间中答案可被精确验证的博弈策略读写任务，证明中等规模开源语言模型能够在1218个权重的循环网络中恢复策略程序、报告其行为并写出最优策略，但修复被篡改网络的能力明显较弱。

    

    语言模型能否在重复博弈策略网络的权重空间中工作？我们向一个中等规模的开源模型提出四个问题，研究对象是具有1218个权重、答案可被精确验证的循环网络。给定一个用英文描述的对手群体，写出一个最优策略：策略的价值是对智能体信息状态的有限计算，因此所提交的网络与最优值之间存在可精确度量的差距。给定一个网络的权重，报告它在十五个对手历史序列中每个历史之后的动作；恢复它所实现的状态机（以深度为十的所有对手序列评判）；或者修复一个输出在某一处被篡改的网络，并指出从哪一轮开始首次出错。该模型在66个回复中的62个成功恢复了程序，在36次中有34次正确报告了全部十五个行为条目，并且当它给出的策略可被评估器编译时，在20次中全部20次达到最优。修复是例外情况：18个中仅4个被成功修复，18个中5个诊断出故障（原摘要在此处截断）。

    arXiv:2611.31979v1 Announce Type: cross  Abstract: Can a language model work in the weight space of a policy network for a repeated game? We put four questions to a mid-size open model, on recurrent networks of 1{,}218 weights with exactly checkable answers. Given a population of opponents described in English, write a policy that is optimal: its value is a finite computation over the agent's information state, so a submitted network has an exact gap from the optimum. Given a network's weights, report its action after each of fifteen opponent histories; recover the state machine it implements, judged on every opponent sequence to depth ten; or repair a network whose output was altered in one place and say which round first goes wrong. The model recovers the program in 62 of 66 replies, reports all fifteen behavioral entries correctly in 34 of 36, and attains the optimum in 20 of 20 when it states a policy the evaluator compiles. Repair is the exception: 4 of 18 repaired, 5 of 18 diagno
    
[^65]: 在参数空间中定位跨语言泛化

    Localizing Cross-Lingual Generalization in Parameter Space

    [https://arxiv.org/abs/2611.31975](https://arxiv.org/abs/2611.31975)

    该论文发现跨语言泛化的成败由参数空间几何决定，即英语与目标语言LoRA适配器是否共享低误差盆地，且数学推理任务中的局部不连通只需少量目标语言样本微调英语适配器的语言编码或语义组合层即可修复。

    

    跨语言迁移在不同任务间的表现并不均衡：它在自然语言理解上往往成功，但在数学推理上常常失败。我们认为这种不对称性由参数空间的几何结构所决定。具体而言，跨语言泛化取决于英语与目标语言的LoRA适配器是否共享同一个低误差盆地。为验证这一观点，我们利用英语与目标语言适配器之间的线性模式连通性（LMC），在三个任务上进行评估。我们发现，NLI和QA任务对应的模型-语言对大多呈线性连通，而数学推理对应的对则不连通，这表明盆地几何结构依赖于任务类型。数学任务中的这种不连通是“局部的”而非全局的：仅需少量目标语言任务样本，且只应用于英语适配器的语言编码或语义组合层，就足以修复这种不连通。此外，跨语言泛化还可以被分解为语言特定编码与语义成分的组合（摘要原文在此处被截断）。

    arXiv:2611.31975v1 Announce Type: new  Abstract: Cross-lingual transfer is uneven across tasks, succeeding for natural language understanding but often failing for mathematical reasoning. We argue that this asymmetry is governed by the geometry of parameter space. Specifically, cross-lingual generalization is determined by whether English and target-language LoRA adapters share a low-error basin. To test this, we use linear mode connectivity (LMC) between English and target-language adapters, evaluating on three tasks. We find that NLI and QA model-language pairs are mostly linearly connected, whereas mathematical reasoning (Math) pairs are not, indicating that basin geometry is task-dependent. Such disconnection in Math is ``local'' rather than global. A small target-language task sample, applied only to the English adapter's language encoding or semantic composition layers, is enough to repair it. Cross-lingual generalization is also factorizable into language-specific encoding and l
    
[^66]: 对关键内容评分：面向语言模型比较的精确反馈缩减

    Scoring What Matters: Exact Feedback Reduction for Language Model Comparison

    [https://arxiv.org/abs/2611.31931](https://arxiv.org/abs/2611.31931)

    本文提出对比感知反馈获取方法CAFA，通过分组相同评分输入并聚合报告系数，在精确保持样本统计量的前提下，最高可去除语言模型比较中90.1%的多余评分查询。

    

    比较语言模型检查点需要估计性能差异，然而评估流程可能会花费代价去评分那些其贡献在这些差异中相互抵消的响应。我们提出对比感知反馈获取（Contrast-Aware Feedback Acquisition, CAFA），这是一种与采样器无关的方法，可在获取反馈之前识别所请求的线性报告需要哪些评分。CAFA 将相同的评分输入分组并聚合其报告系数，在保留完整评分结果的同时，移除了超出普通缓存机制之外的多余查询。针对固定的响应库和不受其他限制的标量评分器，我们刻画了精确恢复样本均值（无论是单独恢复还是与普通样本协方差联合恢复）的必要且充分查询集合。在受控更新实验中，经过四次生成重复取平均，CAFA 分别为 Qwen2.5-1.5B-Instruct 和 SmolLM2-1.7B-Instruct 移除了缓存后剩余评分输入的 55.8% 和 90.1%。（摘要原文在此处截断）

    arXiv:2611.31931v1 Announce Type: new  Abstract: Comparing language-model checkpoints requires estimating performance differences, yet evaluation pipelines can pay to score responses whose contributions cancel from those differences. We introduce Contrast-Aware Feedback Acquisition (CAFA), a sampler-agnostic method that identifies which scores a requested linear report needs before acquiring feedback. CAFA groups identical scoring inputs and aggregates their report coefficients, removing queries beyond ordinary caching while preserving the fully scored result. For a fixed response bank and an otherwise unrestricted scalar scorer, we characterize necessary and sufficient query sets for exact recovery of sample means, either alone or jointly with ordinary sample covariances. On controlled updates, CAFA removes 55.8% and 90.1% of the scoring inputs remaining after caching for Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct, respectively, averaged over four generation repetitions. Across t
    
[^67]: 潜意识学习跨越模态

    Subliminal Learning Crosses Modalities

    [https://arxiv.org/abs/2611.31870](https://arxiv.org/abs/2611.31870)

    本文发现基于文本的潜意识学习可以跨模态迁移——通过数字序列在视觉-语言模型间传递的主题偏好不仅保留在文本中，还会改变图像的生成和表示方式，其机制源于学生模型变化中的一个恒定分量。

    

    潜意识学习是指语言模型中一种令人惊讶的现象：给定某个模型的提示（例如，对猫的偏好）可以通过在表面上看似无意义的数据（例如，数字序列）上进行训练，潜移默化地蒸馏到另一个模型中。本文表明，基于文本的潜意识学习可以跨模态迁移，特别是，通过数字序列在视觉-语言模型之间传递的主题/偏好（例如，动物）也会改变图像的生成和表示方式。首先，我们展示了使用学生模型作为文本编码器的扩散模型，比使用原始模型作为文本编码器时更频繁地生成学生模型所对应的动物，但不会生成其颜色或风格。随后，我们研究了这一现象背后的机制以及哪些概念能够跨模态迁移。学生模型变化中的一个恒定分量解释了图像表示向该特征偏移的大部分原因。添加这一分量……

    arXiv:2611.31870v1 Announce Type: cross  Abstract: Subliminal learning refers to a surprising phenomenon in language models wherein a prompt given to one model (e.g., preference for cats) can be subliminally distilled to another model via training on data that is ostensibly meaningless (e.g., number sequences). This paper shows that text-based subliminal learning transfers cross-modally, in particular, a topic/preference (e.g., animals) transferred through number sequences across vision-language models also changes how images are generated and represented. First, we show that a diffusion model using the student as its text encoder generates the student's animal more often than with the original model as its text encoder, but not its color or style. We then investigate the mechanism behind this phenomenon and which concepts transfer across modalities. A constant component of the student's change accounts for much of the shift of image representations toward the trait. Adding this compon
    
[^68]: 通过双流精炼实现不一致性下的忠实人格引导

    Faithful Persona Steering under Incongruity via Dual-Stream Refinement

    [https://arxiv.org/abs/2611.31848](https://arxiv.org/abs/2611.31848)

    该论文提出QuirkyMind框架，通过特质锚定构建融合句子级摘要与token级序列的双流潜在状态，并利用判别式InfoNCE损失与生成式交叉熵损失的交替精炼目标，实现大语言模型在不一致人格下（稳定特质与情境特异性怪癖并存）的忠实个性化引导。

    

    标准的大语言模型个性化通常将身份视为静态检索任务，忽视了人类人格固有的不一致性，即稳定特质与不合常规的、情境特定的立场并存。现有方法难以调和这两个维度：提示方法在长序列中容易受到上下文漂移的影响，而微调往往会压制个性化的“怪癖”，转而偏向通用的分布模式。为弥合这一差距，我们提出了QuirkyMind，一个将身份定义与其表达解耦的框架。首先，特质锚定构建了一个双流潜在状态，将用于语义稳定性的句子级摘要与用于生成控制的token级序列融合在一起。该状态通过上下文叙事精炼得到稳定，采用交替目标：判别式InfoNCE损失将人格锚定在表示空间中以防止漂移，而生成式交叉熵损失确保忠实的（原文摘要在此处截断）

    arXiv:2611.31848v1 Announce Type: cross  Abstract: Standard LLM personalization typically frames identity as a static retrieval task, overlooking the inherent incongruity of human personas, where stable traits coexist with atypical, context-specific stances. Existing methods struggle to reconcile these dimensions: prompting succumbs to context drift over long sequences, while fine-tuning often suppresses idiosyncratic "quirks" in favor of generic distributional patterns. To bridge this gap, we present QuirkyMind, a framework that disentangles identity definition from its expression. First, Traits Anchoring constructs a dual-stream latent state, fusing a sentence-level summary for semantic stability with a token-level sequence for generative control. This state is stabilized via In-Context Narrative Refinement using an alternating objective: a discriminative InfoNCE loss anchors the persona in representation space to prevent drift, while a generative cross-entropy loss ensures faithful 
    
[^69]: 探索多人社交游戏作为大语言模型的可验证训练场

    Exploring Multi-Player Social Games as Verifiable Training Grounds for Large Language Models

    [https://arxiv.org/abs/2611.31821](https://arxiv.org/abs/2611.31821)

    本文提出将多人社交游戏作为可验证训练环境，通过SFT和RLVR使Qwen3-4B的胜率从33%大幅提升至81%，超越了远大于自身的Gemma-3-27B模型，并能泛化到未见过的游戏中，效果远优于提示等上下文策略。

    

    训练具有社交智能的语言模型通常依赖于大语言模型（LLM）评判者，但这种方式成本高昂、噪声大且容易受到奖励作弊的影响。多人社交游戏提供了一种天然的替代方案：它们需要开放式的沟通来与队友协作并推断隐藏角色，同时其规则提供了客观的胜负结果，可以作为训练中的可验证奖励。本文通过三个社交游戏，探索了Qwen3模型通过监督微调（SFT）和基于可验证奖励的强化学习（RLVR）能够习得多少社交智能。结果表明，面对强大得多的Gemma-3-27B模型，SFT和RLVR将Qwen3-4B在三个训练游戏上的胜率从33%提升至81%，在另外三个未训练的保留游戏上的胜率从39%提升至54%。相比之下，提示等上下文内策略在相同的训练和评估游戏上仅将Qwen3-4B的胜率提高约6个百分点。

    arXiv:2611.31821v1 Announce Type: cross  Abstract: The training of socially intelligent language models often relies on large language model (LLM) judges, which are expensive, noisy, and vulnerable to reward hacking. Multi-player social games offer a natural alternative: they require open-ended communication to coordinate with teammates and infer hidden roles, while their rules provide an objective outcome which can serve as a verifiable reward for training. Using three social games, we explore how much social intelligence Qwen3 models can acquire through supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). Results show that against a much stronger Gemma-3-27B model, SFT and RLVR raise the win rate of Qwen3-4B from 33% to 81% on the three trained games, and from 39% to 54% on three additional held-out games. In contrast, in-context strategies such as prompting improve the win rate of Qwen3-4B by only about 6 pp across the same games we train and evalu
    
[^70]: 为大语言模型智能体学习紧凑的状态抽象

    Learning Compact State Abstractions for LLM Agents

    [https://arxiv.org/abs/2611.31800](https://arxiv.org/abs/2611.31800)

    本文提出将智能体状态定义为交互历史的紧凑抽象，以确定性递归程序表示并用LLM引导的遗传算法搜索，在ALFWorld和ScienceWorld上保持接近参考智能体的任务成功率的同时，将策略输入最多减少71%和57%，且得到的程序可直接审查。

    

    大语言模型（LLM）智能体通常将完整的交互历史作为状态的替代，无论过去的观察和动作是否仍然相关，都全部予以保留。我们转而探究智能体真正需要携带哪些信息前进。我们将智能体状态定义为交互历史的一种紧凑抽象，它能够保持某个固定参考智能体的能力，并以最小的此类抽象作为理想目标。由于匹配参考智能体的行为对黑盒策略而言过于局限且困难，我们通过任务表现而非行为匹配来定义充分性。我们将状态抽象表示为确定性递归程序，并使用由大语言模型引导的遗传算法对其进行搜索。在ALFWorld和ScienceWorld基准上，学习到的抽象保持了接近参考智能体的任务成功率，同时将策略的平均输入分别减少了高达71%和57%。所得到的程序是可直接检查审查的。

    arXiv:2611.31800v1 Announce Type: cross  Abstract: LLM agents commonly use the full interaction history as a proxy for state, retaining all past observations and actions regardless of whether they remain relevant. We instead ask what information an agent actually needs to carry forward. We define agent state as a compact abstraction of interaction history that preserves the competence of a fixed reference agent, with the minimal such abstraction as the ideal. Because matching the reference agent's behavior is unnecessarily restrictive and difficult for black-box policies, we define sufficiency through task performance rather than behavior matching. We represent state abstractions as deterministic recursive programs and search over them using an LLM-guided genetic algorithm. Across ALFWorld and ScienceWorld, the learned abstractions preserve near-reference task success while reducing mean policy input by up to 71% and 57%, respectively. The resulting programs are directly inspectable, p
    
[^71]: SwarmLeak：协调能否提升对LLM智能体的隐私攻击效果？

    SwarmLeak: Does Coordination Improve Privacy Attacks on LLM Agents?

    [https://arxiv.org/abs/2611.31791](https://arxiv.org/abs/2611.31791)

    提出了SwarmLeak红队测试数据集，并通过实验证明多个攻击者协作的蜂群攻击在从LLM智能体群聊中提取机密信息方面并不比单个攻击者更有效。

    

    LLM智能体日益在群聊中协作，同时拥有访问机密记录的工具权限，这使得共享聊天成为一个攻击面。目前尚不清楚多个串通的攻击者是否比单个攻击者更有效地提取此类信息。我们提出了SwarmLeak，这是一个包含24个合成群聊场景的红队测试数据集，涵盖医疗、法律、金融和软件工程领域，攻击者在其中伪装成同事，诱使受害智能体在明确的保密政策下发布受保护的字段。通过使用GLM 5.2以七种对话策略攻击五个受害模型，我们比较了单个攻击者与共享私密通道的四体蜂群，两者在场景、策略以及20万token预算（该预算也涵盖蜂群协调开销）上保持一致。实验表明蜂群没有任何优势：单个攻击者在所有受害模型上都达到或超过蜂群的表现，两种条件下泄露任何字段的运行占比差异最多仅为14.3%。

    arXiv:2611.31791v1 Announce Type: cross  Abstract: LLM agents increasingly collaborate in group chats while holding tool access to confidential records, which makes the shared chat an attack surface. It remains unclear whether several colluding attackers extract such information more effectively than one. We introduce SwarmLeak, a red-teaming dataset of 24 synthetic group-chat scenarios across healthcare, law, finance, and software engineering, where attackers pose as colleagues to make a victim agent post protected fields despite an explicit confidentiality policy. With GLM 5.2 attacking five victim models through seven conversational strategies, we compare one attacker with a swarm of four sharing a private channel, matching scenarios, strategies, and a 200,000-token budget that also covers swarm coordination. Swarms show no advantage: a single attacker matches or exceeds them on every victim, and the share of runs leaking any field differs between conditions by at most 14.3 percenta
    
[^72]: 字节补丁与BPE在匹配数据下的对比：350M参数规模的匹配孪生比较

    Byte Patches versus BPE at Matched Data: A Matched-Twin Comparison at 350M Parameters

    [https://arxiv.org/abs/2611.31790](https://arxiv.org/abs/2611.31790)

    在350M参数、100亿字节的严格匹配数据条件下，8k词表的BPE分词在每字节比特数及多数下游任务上优于字节补丁方法，同时研究揭示了外推上下文、混淆比特分母等可能颠覆结论的评估陷阱。

    

    我们在完全相同的原始字节、文档顺序以及每个优化器步骤近似匹配的字节数条件下，将MEGABYTE风格、配备64字节滑动局部解码器的字节补丁Transformer（local64）与匹配的字节级8k词表BPE解码器（BPE-8k）进行对比：两者均为350M参数和100亿训练字节。在匹配跨度的文档滑动设置下（评分字节范围完全相同），BPE-8k在每字节比特数上领先，相对差距为-5.2%、-4.9%、-9.3%和-6.1%（对应总体、文本、代码、数学；配对自助法检验）。在下游任务中，CharBench等微调更有利于local64（53.7%对40.0%），fact200更有利于BPE-8k（16.5%对7.0%），在alpha=0.05的Holm校正后，PIQA（原始和长度归一化）以及ARC-Easy长度归一化也更有利于BPE-8k；其余任务均为平局。研究中发现会改变结论的评估陷阱包括：超出BPE训练上下文长度的上下文外推、将每token比特数误报为每字节比特数，以及不一致的字节分母。

    arXiv:2611.31790v1 Announce Type: cross  Abstract: We compare a MEGABYTE-style byte-patch Transformer with a 64-byte sliding local decoder (local64) to a matched byte-level 8k BPE decoder (BPE-8k) under identical raw bytes, document order, and approximately matched bytes per optimizer step: 350M parameters and 10 billion training bytes. Under matched-span document sliding (identical scored byte ranges), BPE-8k leads with relative gaps of -5.2%, -4.9%, -9.3% and -6.1% in bits per byte (overall, text, code, math; paired bootstrap). Downstream, CharBench equal fine-tuning favors local64 (53.7% vs 40.0%), fact200 favors BPE-8k (16.5% vs 7.0%), and PIQA (raw and length-normalized) and ARC-Easy length-normalized favor BPE-8k after Holm correction at alpha = 0.05; remaining tasks are ties. Evaluation pitfalls that changed conclusions include context-length extrapolation past the BPE training context, bits per token reported as bits per byte, and inconsistent byte denominators. Unlike controll
    
[^73]: 面向多跳问答的基于SPARQL规划的在线知识图谱单跳遍历

    SPARQL-Planned Single-Hop Traversal over a Live Knowledge Graph for Multi-Hop Question Answering

    [https://arxiv.org/abs/2611.31771](https://arxiv.org/abs/2611.31771)

    该论文提出将稠密检索与基于SPARQL规划的在线知识图谱（Wikidata）单跳遍历交替进行，通过推理路径验证检索段落，在避免多跳扩展开销与噪声的同时兼顾召回率与精确率，从而提升多跳问答性能。

    

    语义检索在多跳问答的召回率和精确率两方面都陷入困境：黄金（正确）段落被淹没在干扰项之中，从而降低阅读器生成答案的质量，而检索这些段落需要对查询进行推理，而非简单匹配。GraphRAG技术通过将段落连接到图节点并在初始检索后执行一跳或多跳扩展来增强语义检索，这提升了召回率，但多跳扩展代价高昂，并且会给检索结果引入额外噪声。我们通过将稠密检索与知识图谱遍历交替执行来打破这种权衡，以应对复杂的多跳问题。稠密检索以较高的召回率定位搜索空间中最初相关的区域，而对预先存在的外部知识图谱（Wikidata）的遍历则通过依据知识图谱中的推理路径对检索到的段落进行验证，从而恢复精确率并提升召回率。为使遍历既具问题自适应性又……（摘要原文在此处截断）

    arXiv:2611.31771v1 Announce Type: cross  Abstract: Semantic retrieval struggles with multi-hop question answering on both recall and precision: gold passages are buried among distractors that degrade the reader's answer, and retrieving them requires reasoning over the query rather than matching it. GraphRAG techniques expand semantic retrieval by connecting passages to graph nodes and performing one or more hops of expansion after initial retrieval. This improves recall, but multi-hop expansion becomes expensive and brings additional noise into the retrieved set. We break this tradeoff by \emph{interleaving} dense retrieval with knowledge-graph traversal for complex multi-hop questions. Dense retrieval locates the first relevant regions of the search space at good recall, while traversal of a pre-existing external KG (Wikidata) restores precision and improves recall by confirming retrieved passages against a reasoning path through the KG. To keep traversal both question-adaptive and re
    
[^74]: TACT-Bench：针对工具使用型智能体的自适应对话威胁测试基准

    TACT-Bench: Testing Adaptive Conversational Threats against Tool-Use Agents

    [https://arxiv.org/abs/2611.31756](https://arxiv.org/abs/2611.31756)

    提出了TACT-Bench，一个针对工具使用型AI智能体的有状态安全基准，通过场景化故事、人设、操纵策略和响应自适应升级，测试智能体在拒绝违反策略的未授权请求的同时保持合法任务完成的能力。

    

    arXiv:2611.31756v1 公告类型：新 摘要：随着AI智能体在使用外部工具和做出决策方面的自主性不断增强，对全面安全基准的需求变得日益迫切。现有基准主要测试被攻陷的工具通道或自适应多轮攻击，但它们可能无法捕捉受策略约束的工具使用型智能体的独特能力——这类智能体必须在拒绝未授权请求的同时，不放弃同一交互中蕴含的合法工作。为填补这一空白，我们提出了TACT-Bench，这是一个围绕场景特定故事、人设、操纵策略和响应自适应升级构建的有状态基准。每个对抗性任务都针对提供给助手的策略中明确禁止的操作，同时该策略对攻击用户保持隐藏。TACT为每个对抗性任务配对至少一个良性对应任务，并且在合规解决方案仍然可用的情况下，其对抗性任务保留了经授权的真实操作。我们定义了……（摘要在此处被截断）

    arXiv:2611.31756v1 Announce Type: new  Abstract: The growing autonomy of AI agents in using external tools and making decisions has created an urgent need for comprehensive safety benchmarks. Existing benchmarks test compromised tool channels or adaptive multi-turn attacks, though they may not capture the unique capabilities of policy-governed tool-use agents, which must resist unauthorized requests without abandoning the legitimate work embedded in the same interaction. To fill this gap, we introduce TACT-Bench, a stateful benchmark built around scenario-specific stories, personas, manipulation tactics, and response-adaptive escalation. Each adversarial task targets an action explicitly prohibited by the policy provided to the assistant, while the policy remains hidden from the attacking user. TACT pairs each adversarial task with at least one benign counterpart, and its adversarial tasks retain authorized ground-truth actions when a compliant resolution remains available. We define t
    
[^75]: BEARING：谁在与谁交谈？——将声学方向与视觉相结合以理解在线多方对话

    BEARING: Who Is Talking to Whom? Grounding Acoustic Direction in Vision for Online Multi-Party Conversation

    [https://arxiv.org/abs/2611.31735](https://arxiv.org/abs/2611.31735)

    BEARING 利用摄像头旁麦克风阵列实测的声音来向，通过免标签标定将其注册到图像中，作为视觉提示、与视觉主动说话者检测融合的输入以及对齐的可验证奖励，从而实现在线多方对话中“谁在与谁交谈”的实时理解。

    

    理解一场群体对话，意味着要从视频和音频中判断谁在说话、以及说话的对象是谁。现有的领先系统将这些任务作为离线问题在单声道片段上处理，并通过自我评估奖励来对齐模型。我们提出 BEARING，一种以几何为先的替代方案，它建立在一个实测信号之上：摄像头旁的麦克风阵列听到声音的来向。一种从数据流中学习、无需标签的标定方法将该方向注册到摄像头图像中，同时一个在线的“人脸—声音—方向”登记册无需任何预先注册。注册后的方向可用作：(i) 视觉-语言模型的视觉提示——作为说话者身上的标记，或指向画外声源的箭头；(ii) 与视觉主动说话者检测进行可靠性加权融合的一个输入——用于处理来自阵列同一方向上的多个说话者；(iii) 用于对齐的可验证奖励：被指认的说话者必须位于声音的来源处，置信度必须（摘要在此处被截断）。

    arXiv:2611.31735v1 Announce Type: cross  Abstract: Understanding a group conversation means deciding, from video and audio, who is speaking and to whom. Leading systems treat these tasks offline on single-channel clips and align models with a self-evaluation reward. We propose BEARING, a geometry-first alternative built on one measured signal: the direction from which a microphone array beside the camera hears a sound. A label-free calibration learned from the stream registers this direction into the camera image, and an online registry of faces, voices and directions needs no enrollment. The registered direction serves as (i) a visual prompt for a vision-language model, a marker on the speaker or an arrow toward an off-screen source; (ii) one input of a reliability-weighted fusion with visual active-speaker detection, for speakers in the same direction from the array; and (iii) a verifiable reward for alignment: the named speaker must sit where the sound comes from, the confidence mus
    
[^76]: 面向智能体的fork()：用于非阻塞LLM编排的Agent内核

    fork() for Agents: Agent Kernel for Non-blocking LLM Orchestration

    [https://arxiv.org/abs/2611.31727](https://arxiv.org/abs/2611.31727)

    Agent-Kernel是一个非阻塞LLM智能体运行时，其核心原语“子图恢复”（SGR）能够在原始异常发生时无需训练过的验证器或规划器，仅重置失败节点的后代、保留已完成的兄弟节点工作并实现父子上下文继承，从而将分布式系统中成熟的子图选择性恢复模式移植到LLM编排的内容平面。

    

    基于LLM的智能体通常同步执行工具调用，并在失败时丢弃已完成的工作。子图选择性恢复——即仅重新执行失败节点的传递性后代——是分布式系统中一种成熟的模式（如Spark、Erlang/OTP、Dryad），而近期的智能体运行时（Sherlock、AgentGit、Atomix）则通过验证器、版本控制或事务性工具边界来保留已完成的工作。我们提出了**Agent-Kernel**，一个非阻塞运行时，将这一模式移植到LLM的*内容平面*：其核心原语——**子图恢复**——可在原始异常发生时触发，无需经过训练的验证器或规划器，在重置失败节点后代的同时保留已完成的兄弟节点，并将每个父节点的文本组合进其子节点的提示中（上下文继承），所有这些都构建在具有形式化状态保持保证的持久化基底之上。在合成基准和标准基准（tau-bench、Gaia2、Auto……）上共计14,434次运行中……（原文摘要在此处截断）

    arXiv:2611.31727v1 Announce Type: new  Abstract: LLM-based agents typically execute tool calls synchronously and discard completed work upon failure. Subgraph-selective recovery, re-executing only a failed node's transitive descendants, is a well-established pattern in distributed systems (Spark, Erlang/OTP, Dryad), and recent agent runtimes (Sherlock, AgentGit, Atomix) preserve completed work through verifiers, version control, or transactional tool boundaries. We present \textbf{Agent-Kernel}, a non-blocking runtime that ports this pattern to the LLM \emph{content plane}: its primitive, \textbf{Sub-Graph Recovery (SGR)}, fires on a raw exception with no trained verifier or planner, resets a failed node's descendants while preserving completed siblings, and composes each parent's text into its children's prompts (context inheritance), all on a durable substrate with a formal state-preservation guarantee. Across 14{,}434 runs on synthetic and standard benchmarks (tau-bench, Gaia2, Auto
    
[^77]: CATWAY：基于校准语言模型置信度的代价敏感智能体三支机器人决策

    CATWAY: Cost-sensitive agentic three-way robot decisions with calibrated language-model confidence

    [https://arxiv.org/abs/2611.31694](https://arxiv.org/abs/2611.31694)

    CATWAY 提出了一种代价敏感的三支决策门控框架，利用四种指令条件下的损失矩阵和校准的语言模型置信度概率，让机器人在执行、追问与拒绝之间做出决策，并在跨领域测试中无需重新拟合即保持有效。

    

    语言模型机器人智能体必须决定是执行、追问还是拒绝一条指令，而不安全执行的代价远高于一次多余的提问。此前没有任何机器人决策门控从错误代价推导其全部阈值、或基于校准后的概率行动，也没有任何方法在一个领域拟合后在另一个领域原样评估。我们提出 CATWAY，一种代价敏感的三支决策门控：通过对四种指令条件（清晰、模糊、不可行、不安全）设定的损失矩阵，在概率单纯形上划定执行、追问与拒绝三个区域，其中追问即是信息收集步骤。其概率通过逻辑回归将十次采样投票与模糊度、不可行性和风险的分级评分融合。CATWAY 在家务指令上拟合后即被冻结，随后在域内、两个新的机器人领域以及外部 TRUST-NAV 基准上，对三个小模型的文本级决策进行了测试。在每次决策十一次生成的情况下……（摘要在此处截断）

    arXiv:2611.31694v1 Announce Type: cross  Abstract: A language-model robot agent must decide whether to execute, question or refuse an instruction, and an unsafe execution costs far more than an unneeded question. No prior robot gate derives all its thresholds from error costs or acts on calibrated probabilities, and none was fitted in one domain and evaluated unchanged in another. We propose CATWAY, a cost-sensitive three-way decision gate: a loss matrix over four instruction conditions (clear, ambiguous, infeasible, unsafe) fixes execute, ask and refuse regions on the probability simplex, the question being the information-gathering step. Its probabilities fuse a ten-sample vote with graded scores of ambiguity, infeasibility and risk by logistic regression. Fitted on household instructions and frozen, CATWAY was tested on text-level decisions of three small models in-domain, in two new robot domains and on the external TRUST-NAV benchmark. At eleven generations per decision against on
    
[^78]: 正确的终点，错误的旅程：合成心理咨询对话缺失人类互动动态

    Right Destination, Wrong Journey: Synthetic Counseling Dialogues Miss Human Interactional Dynamics

    [https://arxiv.org/abs/2611.31686](https://arxiv.org/abs/2611.31686)

    该研究提出以轨迹为中心的过程保真度评估方法，发现LLM生成的心理咨询对话虽能模拟合理的最终结果，但在恢复动态和逐轮响应等人类互动动态方面保真度不足，且即使采取针对性的生成干预也无法完全恢复这些互动特性。

    

    LLM生成的心理咨询对话越来越多地被用于训练和评估心理健康系统，然而现有评估大多聚焦于回复质量和最终结果，而非心理支持如何随时间逐步展开。我们对十个人类、重构以及合成的心理咨询数据集开展了以轨迹为中心的过程保真度评估。结果表明，合成对话往往能够达到看似合理的终点，但在恢复动态和逐轮响应方面与人类支持存在差异。尽管生成设计可以改变恢复时机和轨迹结构等全局属性，但在各类模型和生成范式下，患者与咨询师之间的条件性响应保真度始终较低。我们在受控合成数据集上的实验表明，针对性的生成干预能够在若干轨迹层面的指标上缩小与真实数据的差距，但无法完全恢复人类的互动动态。

    arXiv:2611.31686v1 Announce Type: cross  Abstract: LLM-generated counseling dialogues are increasingly used for training and evaluating mental-health systems, yet existing evaluations focus largely on response quality and final outcomes rather than how support unfolds over time. We conduct a trajectory-centered evaluation of process fidelity across ten human, reconstructed, and synthetic counseling datasets. Our results show that synthetic dialogues often reach plausible endpoints but differ from human support in recovery dynamics and turn-by-turn responsiveness. While generation design can shift global properties such as recovery timing and trajectory structure, contingent patient--counselor responsiveness remains consistently lower in fidelity across models and generation paradigms. Our experiments on controlled synthetic datasets show that targeted generation interventions can narrow the gap with real data on several trajectory-level measures, but do not fully recover interactional 
    
[^79]: LimoBench：大语言模型有多“懂”街道？

    LimoBench: How Street Smart Are Large Language Models?

    [https://arxiv.org/abs/2611.31682](https://arxiv.org/abs/2611.31682)

    提出了首个街道级地理空间基准LimoBench，通过82个基于真实地图数据的自然语言位置查询，系统评估大语言模型在地理编码、路径规划、$k$近邻搜索等七种空间操作上的位置感知与精确回答能力。

    

    大语言模型（LLMs）已经改变了人类与信息交互的方式，实现了跨领域的自然语言交互界面。它们正越来越多地被部署于地理空间应用中，包括基于地图的问答、基于位置的服务，以及空间数据库的自然语言接口。但LLMs是否真正具备位置感知能力，能够可靠地解释并精确回答涉及真实世界坐标、兴趣点（POI）和地理空间距离的位置查询？为了具体回答这个问题，我们提出了LimoBench，一个街道级基准测试，包含82个自然语言位置查询，涵盖七种经典空间操作：地理编码、逆地理编码、路径规划、方向判断、范围搜索、$k$近邻搜索和空间连接。这些查询基于印第安纳州西拉法叶市的真实地图数据构建，并按照位置和语义复杂度进行分级，旨在探测LLM的街道级空间……

    arXiv:2611.31682v1 Announce Type: new  Abstract: Large language models (LLMs) have transformed how humans interact with information, enabling natural-language interfaces across various domains. They are increasingly deployed for geospatial applications including map-based question answering, location-based services, and natural-language interfaces to spatial databases. But are LLMs truly location-aware, able to reliably interpret and precisely answer location queries involving real-world coordinates, points of interest (POIs), and geospatial distances? To answer this concretely, we present LimoBench, a street-level benchmark of 82 natural-language location queries over seven canonical spatial operations: geocoding, reverse geocoding, routing, directionality, range search, $k$-nearest-neighbor search, and spatial join. The queries are grounded in real map data of West Lafayette, Indiana, and are graded by location and semantic complexity. They are designed to probe LLMs' street-level sp
    
[^80]: 多语言压缩中的来源信息丢失与修复

    Provenance Loss and Repair in Multilingual Compaction

    [https://arxiv.org/abs/2611.31668](https://arxiv.org/abs/2611.31668)

    该论文发现多语言对话压缩会因语法性别等语言差异而丢失属性的来源信息（无法区分用户明确声明与语法默认），并提出一种无需训练、覆盖写作标注、丢弃前似然验证和阅读信任三个环节的修复方法，大幅恢复被压缩丢弃的已声明属性召回率。

    

    arXiv:2611.31668v1 公告类型： cross 摘要：会话型助手会将较早的对话消息压缩成简短笔记并丢弃原始消息。当笔记所用的语言需要作出英语中不作区分的区分时，例如西班牙语形容词的性属变化（agotado vs. agotada），省略该属性或明确声明该属性的消息可能产生相同的笔记。在原始消息被丢弃后，后续对话无法再区分这两种情况：将词形变化按字面理解会断言一个用户从未提供过的属性，而将其视为常规语法而忽略则会丢弃用户确实提供过的属性。我们提出了一种无需训练的修复方法，涵盖写作、丢弃前验证和阅读三个环节：写入者用成对形式（如 agotado/a）标注未声明的属性，丢弃前的似然检查将缺乏依据的单式形式改写为成对形式，而阅读者则将剩余的单式形式视为已声明的事实。在五个 Qwen3.5 和 Gemma-4 模型上，消息丢弃使问答（QA）中对已声明属性的召回率从 90% 降至 34%；我们的修复方法……

    arXiv:2611.31668v1 Announce Type: cross  Abstract: Conversational assistants compact older messages into short notes and evict the originals. When the note's language requires a distinction that English leaves open, such as Spanish adjective gender (agotado vs. agotada), messages that omit or explicitly state that attribute can yield the same note. After eviction, later turns cannot tell them apart: taking the inflected form literally asserts an attribute the user never gave, whereas discounting it as routine grammar discards one the user did give. We propose a training-free repair across writing, pre-eviction verification and reading. The writer marks unstated attributes with paired forms (e.g., agotado/a), a likelihood check before eviction rewrites unsupported single forms into paired ones, and the reader trusts remaining single forms as stated facts. On five Qwen3.5 and Gemma-4 models, eviction cuts recall of stated attributes from 90% to 34% in question answering (QA); our repair 
    
[^81]: E2SIM：基于演化状态的日常人类行为长期模拟

    E2SIM: Long-Horizon Simulation of Everyday Human Behavior with Evolving States

    [https://arxiv.org/abs/2611.31641](https://arxiv.org/abs/2611.31641)

    E2SIM 提出了一个通过反馈循环将行动者演化状态与动态共享环境联合建模的纵向用户模拟框架，使早期行动的后果能够塑造后续行为，同时保留各行动者视角的差异，并配套了衡量动态一致性的纵向轨迹诊断指标。

    

    AI 智能体越来越被期望能够在长期内、跨工作与日常生活为用户提供支持。这不仅需要建模用户是谁，还需要建模他们的目标、兴趣、活动和互动如何随时间演变。纵向用户模拟的一大挑战在于保持这些演化状态与共享环境中行动后果之间的一致性。我们提出了 E2SIM，一个联合建模行动者演化状态与动态共享环境的纵向用户模拟框架。E2SIM 通过一个反馈循环将行动者状态更新与共享世界的变化相连接，该循环串联了即时行动、延迟结果以及跨工作与生活的行动者特定观察。这使得早期行动的后果能够影响后续行为，同时保留每个行动者所知信息的差异。我们进一步提出了一套纵向轨迹诊断指标，用于衡量动态一致性、连贯事件……

    arXiv:2611.31641v1 Announce Type: new  Abstract: AI agents are increasingly expected to support users over long periods, across work, and daily life. This requires modeling not only who users are, but also how their goals, interests, activities, and interactions evolve over time. A challenge in longitudinal user simulation is maintaining consistency between these evolving states and the consequences of actions in a shared environment. We introduce E2SIM, a longitudinal user simulation framework that jointly models evolving actor states and a dynamic shared environment. E2SIM connects actor-state updates with shared-world changes through a feedback loop linking immediate actions, delayed outcomes, and actor-specific observations across work and life. This allows the consequences of earlier actions to shape later behavior, while preserving differences in what each actor knows. We further propose a suite of longitudinal trajectory diagnostics, measuring dynamic consistency, coherent event
    
[^82]: Tablassert：从生物医学表格中声明式提取知识图谱

    Tablassert: Declarative Extraction of Knowledge Graphs from Biomedical Tables

    [https://arxiv.org/abs/2611.31614](https://arxiv.org/abs/2611.31614)

    Tablassert 是一个基于 YAML 声明式配置的工具，可将生物医学论文中的表格数据自动转换为符合 NCATS 生物医学数据翻译器标准的知识图谱，并能将自由文本解析为标准标识符。

    

    生物医学研究论文中发表的表格，尤其是补充表格，捕捉了可能因多种原因（包括统计显著性较低、篇幅限制和叙事范围）而未出现在手稿正文中的知识断言。这些表格包含的数据（例如关联统计、实验检测读数、人工整理的基因列表和组学数据）编码了生物实体之间的关系。Tablassert 可将此类表格数据转换为可供 NCATS 生物医学数据翻译器使用的知识图谱。用户在 YAML 中声明每个表格的数据如何映射到主语-谓语-宾语-限定符语句；Tablassert 将自由文本（例如“肺癌”）解析为标准标识符，附加来源信息和统计数据，并输出符合 Translator 标准的图文件。

    arXiv:2611.31614v1 Announce Type: new  Abstract: Tables published in biomedical research papers, particularly supplemental tables, capture knowledge assertions that may not be in the main text of the manuscript for several reasons, including low statistical significance, space limitations, and narrative scope. These tables include data (e.g., association statistics, assay readouts, curated gene lists, and omics) encoding relationships between biological entities. Tablassert converts such tabular data (Excel, CSV, TSV, Parquet) into knowledge graphs ready for the NCATS Biomedical Data Translator (Fecho et al. 2025). A user declares in YAML how each table's data map to subject-predicate-object-qualifier statements; Tablassert resolves free text (e.g., ''lung cancer'') to standard identifiers, attaches provenance and statistics, and emits graph files conforming to Translator standards (D. Unni, Bruskiewich, and Moxon 2026; D. R. Unni et al. 2022).
    
[^83]: 训练决策模型以模拟人类受访者

    Training decision models for simulating human respondents

    [https://arxiv.org/abs/2611.31613](https://arxiv.org/abs/2611.31613)

    在历史受访者选择数据上对KEV决策模型进行选择训练，可将LLM合成受访者在跨领域留存研究上的处理效应误差降低22%至89%，且小模型可胜过更大的现成模型。

    

    研究人员越来越多地将大型语言模型（LLM）用作市场研究、联合分析和随机实验中的合成受访者。然而，现成的生成式LLM产生的合成选择数据会给下游的计量经济学推断带来偏差，无论是专有模型还是更高的推理投入都无法可靠地减少这种偏差。我们研究了用于硅采样（silicon sampling）的决策模型（System One架构）。通过选择训练，我们在推理之前、不使用目标研究数据的情况下，利用条件极大似然法在历史受访者层面的选择微观数据上更新KEV模型权重。在三个模型规模上，基于两个离散选择实验的训练使四个来自新领域（疫苗、营养标签、消费者权利和人工智能）且采用新回应格式的留存研究的处理效应误差降低了22%至89%。一个经过选择训练的小模型在四个研究中的三个上胜过了规模大得多的现成模型。尽管（原文在此处截断）

    arXiv:2611.31613v1 Announce Type: new  Abstract: Researchers increasingly use large language models (LLMs) as synthetic respondents in market research, conjoint analysis, and randomized experiments. However, synthetic choice data from off-the-shelf generative LLMs bias downstream econometric inference. Neither proprietary models nor higher reasoning effort reliably reduces this bias. We investigate decision models (System One architectures) for silicon sampling. Using choice training, we update KEV model weights on historical respondent-level choice microdata by conditional maximum likelihood, before inference and without target-study data. Across three model scales, training on two discrete choice experiments lowers treatment-effect error by 22% to 89% on four held-out studies from new domains (vaccines, nutrition labels, consumer rights, and AI) and new response formats. A small choice-trained model outperforms a much larger off-the-shelf model on three of the four studies. Although 
    
[^84]: LUCID：基于统计学原理的大语言模型推理评估自我纠错方法

    LUCID: Statistically Grounded Self-Correction for LLM Reasoning Evaluation

    [https://arxiv.org/abs/2611.31595](https://arxiv.org/abs/2611.31595)

    LUCID无需重新训练或访问模型权重，通过将提示自我纠正与基于设计的监督学习相结合来校准语言模型评分器对推理步骤的评估，将GPT-5-mini的校准差距从7.8个百分点降至0.9个百分点，误报减半，错误检测F1提升32%。

    

    廉价的语言模型评分器可以对思维链推理进行大规模评分，但系统性的步骤评分错误可能会扭曲推理审计和解决方案排名。我们提出了LUCID（通过推理时诊断实现无偏校准的循环），它将提示自我纠正与基于设计的监督学习（DSL）相结合。在无需重新训练或访问模型权重的情况下，LUCID通过将预测的合理率与专家合理率对齐来校准二元步骤评分。基于一个小型专家标注数据集，LUCID以纯文本形式总结评分器的偏差，并提示评分器对未见过的步骤重新评分。例如，在PRM800K数据集上，GPT-5-mini评分过于严格，对计算、验证和推理链后期步骤的低估最为严重。经过一次校正，LUCID将GPT-5-mini的校准差距（以预测率与专家率的绝对差值衡量）从7.8个百分点缩小至0.9个百分点。LUCID还将误报（将合理步骤误评为不合理）减少了一半，并将不合理步骤的错误检测F1分数提升了32%。

    arXiv:2611.31595v1 Announce Type: new  Abstract: Cheap language-model raters grade chain-of-thought reasoning at scale, yet systematic step-rating errors can skew reasoning audits and solution rankings. We introduce LUCID (Loop for Unbiased Calibration via Inference-time Diagnosis), which pairs prompt self-correction with Design-based Supervised Learning (DSL). Without retraining or weight access, LUCID calibrates binary step ratings by aligning predicted and expert soundness rates. From a small expert-labeled set, LUCID summarizes rater bias in plain text and prompts the rater to re-grade unseen steps. For example, on PRM800K, GPT-5-mini grades too strictly, under-rating computation, verification, and late-trace steps most acutely. In one pass, LUCID shrinks GPT-5-mini's calibration gap from 7.8 to 0.9 percentage points in absolute predicted-expert rate difference. LUCID also halves false alarms (sound steps rated unsound) and lifts error-detection F1 on unsound steps by 32%. LUCID re
    
[^85]: GoldiBench：工具选对了，用量却错了——面向专业编辑中效果判断能力的智能体基准测试

    GoldiBench: Right Tools, Wrong Amount -- Benchmarking Agents on Effect Judgement in Professional Editing

    [https://arxiv.org/abs/2611.31590](https://arxiv.org/abs/2611.31590)

    论文提出GoldiBench基准，通过500个专业编辑任务将“工具操作”与“效果判断（用量把握）”分离评估，发现智能体操作能力很强，但难以准确判断实现目标视觉效果所需的工具用量。

    

    真实的编辑需求描述的是一种视觉效果，而非达成该效果的具体步骤，因此服务于这类需求的智能体必须判断每种工具需要使用多少才能实现该效果。现有基准并未衡量这一点：它们的任务要么直接给出工具和参数值，要么对用量不加限定，并且依赖VLM裁判评分——而这些裁判会让超过一半的半强度和三倍强度结果通过。我们提出了GoldiBench，一个包含500个编辑任务的基准，涵盖五个专业应用（darktable、RawTherapee、GIMP、Inkscape、Blender），任务既可以通过编写程序解决，也可以通过操作GUI完成。每个任务将一张START图像与一个由隐藏程序渲染出的TARGET图像配对，因此用量由已知目标固定；智能体的编辑结果会被重新渲染并与TARGET进行比较，且同一任务以三种形式呈现：给出工具和参数值、仅给出工具、或仅给出目标，从而将操作能力与效果判断区分开来。在25个开源、GUI和闭源模型中，智能体的操作能力表现良好（4B以上模型使用程序可达99-100%……

    arXiv:2611.31590v1 Announce Type: cross  Abstract: Real editing requests describe a look, not the steps to reach it, so an agent that serves them must judge how much of each tool the look takes. Existing benchmarks do not measure this: their tasks either state the tools and values, or leave the amount open and are scored by VLM judges that pass more than half of half- and triple-strength results. We introduce GoldiBench, a benchmark of 500 editing tasks in five professional applications (darktable, RawTherapee, GIMP, Inkscape, Blender), solvable by writing a program or by operating the GUI. Each task pairs a START image with a TARGET rendered from a hidden program, so the amount is fixed by a known target; the agent's edit is re-rendered and compared with TARGET, and the same task is posed with the tools and values, the tools only, or the goal only, separating operation from effect judgement. Across 25 open, GUI and closed models, agents operate well (99-100% with programs from 4B up w
    
[^86]: SecondChance：对AI智能体可靠错误恢复能力的基准测试

    SecondChance: Benchmarking Reliable Error Recovery in AI Agents

    [https://arxiv.org/abs/2611.31586](https://arxiv.org/abs/2611.31586)

    SecondChance是一个基于智能体自身自然产生的错误检查点的错误恢复评估框架，通过四种不同辅助条件全面衡量AI智能体纠正错误、告知用户和修复任务的真实恢复能力。

    

    AI智能体在执行任务时会犯错，而它们的可靠性取决于犯错之后如何应对。现有的错误恢复基准测试大多通过注入故障来制造错误、不涉及用户交互，且仅评估任务最终是否完成，因此无法体现智能体是否能悄悄撤销自己的错误、是否会告知用户，或面对无法修复的错误时如何处理。我们提出了SecondChance，这是一个错误检查点数据集与评估框架，用于刻画和衡量AI智能体的错误恢复行为。这些错误是自然产生的：它们来自智能体在τ²-bench、BFCL和AppWorld中自行执行任务时的真实失误。随后，不同的智能体从相同的检查点出发，在四种条件之一的情况下继续执行任务：无任何辅助、收到错误提示、获知错误位置，或得到修复指导。我们衡量错误纠正、任务成功率、策略合规性、环境状态保持、修复持久性以及成本。我们在全部四种条件下评估了12个模型，共计5,760次任务续接测试。

    arXiv:2611.31586v1 Announce Type: new  Abstract: AI agents make errors while they act, and their reliability depends on what they do afterwards. Existing recovery benchmarks mostly inject faults, involve no user and score whether the task is eventually completed, so they cannot show whether an agent undid its own silent error, told the user, or faced an error that could not be fixed. We introduce SecondChance, an error checkpoint dataset and evaluation harness that profiles and measures error-recovery behaviour of AI agents. The errors are natural: they come from agents' own task attempts in $\tau^2$-bench, BFCL and AppWorld. Different agents then resume from the same checkpoints under one of four conditions: no assistance, an error notice, the error location, or repair guidance. We measure error correction, task success, policy compliance, preservation of environment state, repair durability and cost. We evaluate 12 models under all four conditions in 5,760 continuations. Recovery is 
    
[^87]: LatentLens：重新思考多智能体系统中的潜在接口

    LatentLens: Rethinking Latent Interface in Multi-Agent Systems

    [https://arxiv.org/abs/2611.31582](https://arxiv.org/abs/2611.31582)

    本文提出LatentLens框架，从外部和内部两个层面解耦多智能体系统中的潜在接口，发现在相同智能体框架下引入潜在接口并不能提升甚至可能显著降低性能，其效果取决于重新对齐兼容性、信息可访问性和传播动态等内部因素。

    

    潜在接口使多智能体系统中的智能体能够通过连续表示而非解码后的文本进行通信。先前的研究报告了性能和效率上的提升，但既没有将潜在通信与智能体框架（harness）分离开来，也没有考察其内部组件，导致其实际贡献尚不明确。我们提出了LatentLens，一个在两个层面解耦潜在接口的框架：在外部将其与智能体框架解耦，以衡量其对整体性能的贡献（RQ1）；在内部将其解耦为重新对齐兼容性、信息可访问性和传播动态，以识别各因素的优势与局限（RQ2）。在基于Qwen3模型的七个基准测试中，在相同的智能体框架上添加潜在接口并不能提升性能，甚至可能显著降低准确率。在内部，潜在表示保留了一定的领域信息，但不兼容的重新对齐可能大幅降低性能。

    arXiv:2611.31582v1 Announce Type: cross  Abstract: Latent interfaces let agents in a multi-agent system communicate through continuous representations rather than decoded text. Prior work reports performance and efficiency gains, but neither separates latent communication from the agent harness nor examines its internal components, leaving its actual contribution unclear. We propose LatentLens, a framework that decouples the latent interface at two levels: externally from the agent harness, to measure its contribution to overall performance (RQ1), and internally into realignment compatibility, information accessibility, and propagation dynamics, to identify the benefits and limitations of each (RQ2). Across seven benchmarks with Qwen3 models, adding the latent interface to the same harness does not improve and can even substantially degrade accuracy. Internally, latent representations retain some domain information, but incompatible realignment can substantially degrade performance, an
    
[^88]: PazaBench：面向非洲低资源语言的大规模自动语音识别基准测试

    PazaBench: Large-Scale ASR Benchmark for Low-Resource Languages in Africa

    [https://arxiv.org/abs/2611.31578](https://arxiv.org/abs/2611.31578)

    该论文提出PazaBench，首个将六个公开语料库整合为统一可复现评估框架的大规模非洲低资源语言ASR基准，在39种非洲语言上评估了52个最先进模型，并引入架构感知比较与语言学差异量化分析。

    

    低资源语言在自动语音识别（ASR）基准测试中的代表性仍然不足，限制了可靠评估和比较模型性能的能力。我们提出了一个针对非洲低资源语言的大规模自动语音识别基准测试。该基准将六个公开语料库整合为统一、可复现的评估框架，并在多样化数据集上对涵盖三种模型架构范式的52个最先进ASR模型在39种非洲语言上进行了评估。模型采用标准的准确性与效率指标进行评估，包括词错误率（WER）、字符错误率（CER）和逆实时因子（RTFx）。该基准引入了架构感知的比较方法，系统性地分析单个模型及更广泛的模型家族。此外，我们量化了不同语言族之间的语言学差异，揭示了这些因素如何影响各种非洲语言的ASR性能。我们还提供了一个……（摘要在此处被截断）

    arXiv:2611.31578v1 Announce Type: new  Abstract: Low-resource languages remain underrepresented in ASR benchmarks, limiting the ability to reliably assess and compare model performance. We present a large-scale benchmark for automatic speech recognition (ASR) on low-resource languages in Africa. The benchmark harmonizes six public corpora into a unified, reproducible evaluation framework and evaluates 52 state-of-the-art ASR models spanning three model architectural paradigms across 39 African languages on diverse datasets. Models are evaluated using standard accuracy and efficiency metrics, word error rate (WER), character error rate (CER), and inverse real-time factor (RTFx). This benchmark introduces architecture-aware comparison that systematically analyzes both individual models and broader model families. In addition, we quantify linguistic variance across language families, revealing how these factors shape ASR performance across diverse African languages. We also provide a comp
    
[^89]: CLAIMGATE：对口头法律与临床声明的谨慎设备端核查

    CLAIMGATE: Cautious On-Device Checking of Spoken Legal and Clinical Claims

    [https://arxiv.org/abs/2611.31569](https://arxiv.org/abs/2611.31569)

    CLAIMGATE是一个运行在设备端的实时核查助手，能在法律和医疗对话中检测可核查的声明、检索官方规则并给出带引用的判定，且其设计确保在无法确定时保持沉默而非误导用户。

    

    在法律和医学领域，一条被错误陈述的规则可能在任何人来得及核查之前就已影响决策。只有当核查工具能够跟上对话节奏、将敏感音频保留在设备上，并在无法确定时保持沉默而非误导时，实时核查此类声明才真正有帮助。我们提出了CLAIMGATE，这是一种设备端助手，它监听对话、检测可核查的声明、检索相关的官方规则，并为每条声明标注“支持”、“反驳”或“未覆盖”，同时附上引用。CLAIMGATE的设计原则是“失败时趋于沉默”：其采用4比特量化的70亿参数语言模型只允许引用检索到的规则，且判定结果由代码根据模型所引用的声明和规则推导得出。我们使用真实的官方规则文本（5,742条佛罗里达州和德克萨斯州法规子条款；CDC、FDA和MedlinePlus指南）、真实转录文本、实时语音以及作者撰写并带有金标准标签的声明，在交通拦截和医疗就诊场景中对该系统进行评估。在141条留出的交通拦截声明上，CL……（摘要原文在此处截断）

    arXiv:2611.31569v1 Announce Type: new  Abstract: In law and medicine, a misstated rule can shape a decision before anyone has time to check it. Checking such claims in real time helps only if the checker keeps pace with conversation, keeps sensitive audio on the device, and stays silent rather than mislead. We present CLAIMGATE, an on- device assistant that listens to a conversation, detects checkable claims, retrieves the relevant official rule, and labels each claim as supported, contradicted or not covered, with a citation. CLAIMGATE is designed to fail towards silence: its 4-bit 7B- parameter language model may cite only retrieved rules, and the verdict is derived in code from the claim and rule the model quotes. We evaluate it on traffic stops and medical visits using real official rule text (5,742 Florida and Texas statute subsections; CDC, FDA and MedlinePlus guidance), real transcripts, live speech and author-written gold-labelled claims. On 141 held-out traffic-stop claims, CL
    
[^90]: LLM模拟的受访者能否对调查问题进行预测试？来自预注册前瞻性验证的证据

    Can LLM-Simulated Respondents Pre-Test Survey Questions? Evidence From Preregistered Prospective Validation

    [https://arxiv.org/abs/2611.31532](https://arxiv.org/abs/2611.31532)

    该研究通过预注册的前瞻性验证设计，证明LLM模拟的受访者能够在真实人类回答数据产生之前，通过平均回答、方差和子群体差异三个指标对候选调查问题进行排序，从而有效支持新调查问题的预测试。

    

    大语言模型（LLM）可以被调控以模拟行为反应，包括调查问卷的回答，但这些模拟是否以及如何能够支持下游研究任务仍不清楚。在此，我们评估了模拟受访者是否能够支持在正式发放新调查问题之前对其进行预测试，具体操作为对候选调查项目进行优先级排序。具体而言，我们根据候选问题的平均回答对其进行排序——这可以揭示天花板效应或地板效应；根据其方差进行排序——这表明回答中是否包含有意义的变化；以及根据子群体之间的差异进行排序——这表明问题是否能够捕捉到人群中相关群体之间的区别。我们引入了一项预注册的前瞻性评估，在目标人类回答数据存在之前就提交模型预测，从而从设计上排除了模型对这些回答的记忆。使用两轮本地调查数据（2025年：N = 1,271；2026年：N……）

    arXiv:2611.31532v1 Announce Type: new  Abstract: Large language models (LLMs) can be conditioned to simulate behavioral responses, including survey responses, but it remains unclear whether and how these simulations can support downstream research tasks. Here, we evaluate whether simulated respondents can support pre-testing new candidate survey questions before fielding them, operationalized as prioritizing candidate survey items. Specifically, we rank candidate questions by their mean responses, which can reveal ceiling or floor effects; their variance, which indicates whether responses contain meaningful variation; and differences between subpopulations, which indicate whether questions capture distinctions between relevant groups within a population. We introduce a preregistered, prospective evaluation that commits model predictions before target human responses exist, ruling out memorization of those responses by design. Using two waves of a local survey (2025: N = 1,271; 2026: N 
    
[^91]: 层间细读：表征质量究竟藏于何处，几何结构又揭示了什么

    Reading Between the Layers: Where Representation Quality Lives, and What Geometry Reveals

    [https://arxiv.org/abs/2611.31496](https://arxiv.org/abs/2611.31496)

    该论文通过对25个模型、13个数据集的大规模实证研究，系统揭示了预训练语言模型的最佳表征层随预训练策略与架构而系统性变化（掩码编码器峰值在中后层、序列到序列模型随任务类型偏移、对比编码器取决于读出方式、仅解码器模型呈阶段结构），为无标签层选择这一开放问题提供了实证指引。

    

    预训练文本基础模型中的表征质量并非集中于最后一层，然而由于无标签情况下的层选择一直是悬而未决的难题，实践者通常习惯在最后一层进行探测。我们对层质量在何处达到峰值开展了一项大规模实证研究，涵盖25个模型，横跨四种架构族和四种预训练策略，并在13个数据集上进行评估，任务包括分类、成对分类、语义文本相似度和多项选择题问答。峰值探测层系统性地依赖于预训练策略和架构：掩码编码器在中后层达到峰值；序列到序列模型的峰值随任务类型而变化；对比式句子编码器在CLS和末词元读出方式下信号集中于最后一层，但在平均池化下峰值出现在较早层；仅解码器模型则展现出与深度相关的独特阶段结构。我们将这些发现提炼为稳定几何……

    arXiv:2611.31496v1 Announce Type: cross  Abstract: Representation quality in pretrained text foundation models is not concentrated at the final layer, yet practitioners routinely probe it there because layer selection without labels has remained an open problem. We present a large-scale empirical study of where layer quality lives across 25 models spanning four architectural families and four pretraining strategies, evaluated on 13 datasets across classification, pair classification, semantic textual similarity, and multiple-choice question answering. Peak probing layers depend systematically on pretraining strategy and architecture: masked encoders peak in late-middle layers, sequence-to-sequence models shift their peak with task type, contrastive sentence encoders concentrate signal at the final layer under CLS and last-token readouts but peak early under mean pooling, and decoder-only models exhibit distinct phase structure tied to depth. We distill these findings into the Stable Ge
    
[^92]: ModernELECTRA：面向分类与嵌入的紧凑编码器的替换词元检测与边界窗口重建

    ModernELECTRA: Replaced-Token Detection with Boundary-Window Reconstruction for Compact Encoders that Classify and Embed

    [https://arxiv.org/abs/2611.31495](https://arxiv.org/abs/2611.31495)

    ModernELECTRA提出了一种预训练方案，在保留替换词元检测（RTD）主目标的同时，引入轻量级的边界窗口重建辅助任务，使紧凑编码器既能擅长分类任务，又能胜任检索与语义相似度等嵌入任务。

    

    arXiv:2611.31495v1 公告类型：新论文 摘要：ELECTRA式的替换词元检测（RTD）是一种用于预训练文本编码器的高样本效率训练目标，采用RTD训练的模型（如DeBERTaV3）在分类基准测试中表现优异。然而，众所周知，这类模型作为骨干网络在检索和语义相似度等嵌入任务上表现不佳：二元的“原始词元 vs. 替换词元”信号并未要求最终层保留稠密表示所依赖的词元身份和上下文内容。我们提出了ModernELECTRA，这是一种预训练方案，它将RTD保留为主要目标，并增加了一个轻量级辅助任务——边界窗口重建。对于每个词元，我们采样一个严格包含该词元的3至12个词元的窗口，仅从判别器在窗口两个边界处的隐藏状态来重建该词元的原始身份。该任务在训练时只需额外添加一个小型前馈网络（FFN），训练结束后即被丢弃，但它能迫使每个最终层状态携带信息……

    arXiv:2611.31495v1 Announce Type: new  Abstract: ELECTRA-style replaced-token detection (RTD) is a sample-efficient objective for pre-training text encoders, and RTD-trained models such as DeBERTaV3 are strong on classification benchmarks. The same models, however, are known to be poor backbones for embedding tasks such as retrieval and semantic similarity: the binary "original vs. replaced" signal does not ask the final layer to retain the token identity and contextual content that dense representations rely on. We introduce ModernELECTRA, a pre-training recipe that keeps RTD as the main objective and adds a lightweight auxiliary task, boundary-window reconstruction. For every token, we sample a window of 3-12 tokens that strictly contains it and reconstruct the token's original identity only from the discriminator's hidden states at the two window boundaries. The task costs one small FFN at training time and is discarded afterwards, yet it forces every final-layer state to carry info
    
[^93]: SwiftLoop：面向循环 Transformer 自适应隐式推理的选择性精炼方法

    SwiftLoop: Selective Refinement for Adaptive Implicit Reasoning in Looped Transformers

    [https://arxiv.org/abs/2611.31481](https://arxiv.org/abs/2611.31481)

    SwiftLoop 提出在推理时通过位置级冻结策略对循环 Transformer 进行选择性自适应精炼，同时优化循环深度与活跃宽度两个维度的计算分配，以更低的计算成本获得更优的推理精度。

    

    循环 Transformer 通过对潜在表示的反复精炼来推进隐式思维链推理。然而，现有的循环推理器通常采用固定深度的循环，在每次迭代中更新所有潜在位置，当无需进一步精炼时会造成冗余计算。我们提出 SwiftLoop，这是一个推理时框架，通过位置级别的冻结实现选择性循环精炼，从而在循环深度和活跃宽度两个维度上实现自适应的计算分配。SwiftLoop 通过由两个互补准则导出的后见之明引导监督来学习轻量级冻结策略：内部表示层面的 K/V 可复用性以及输出层面的功能性影响。在推理阶段，这些策略基于观察到的隐藏状态引导自适应冻结。大量实验表明，SwiftLoop 相比固定循环推理实现了更优的精度-计算权衡。

    arXiv:2611.31481v1 Announce Type: cross  Abstract: Looped Transformers advance implicit chain-of-thought reasoning through recurrent latent refinement. However, existing looped reasoners typically employ fixed-depth recurrence, updating all latent positions at every iteration and incurring redundancy when further refinement is unnecessary. We introduce \textsc{SwiftLoop}, an inference-time framework for selective recurrent refinement through position-wise freezing, enabling adaptive computation allocation across both recurrent depth and active width. SwiftLoop learns lightweight freezing policies through hindsight-guided supervision derived from two complementary criteria: K/V reusability at the internal representation level and functional impact at the output level. At inference time, these policies guide adaptive freezing based on observed hidden states. Extensive experiments demonstrate that SwiftLoop achieves favorable accuracy--computation trade-offs over fixed-loop inference and 
    
[^94]: 基于引文关联病例报告的临床相似图像对数据集：用于医学图像检索

    A dataset of clinically similar image pairs from citation-linked case reports for medical image retrieval

    [https://arxiv.org/abs/2611.31465](https://arxiv.org/abs/2611.31465)

    该研究构建了包含86,144对临床相似图像的PMC-PaIR数据集，通过跟随引文并利用大语言模型生成和回答分层问题来匹配病例图像，为医学图像检索系统的训练与评估提供了高质量的大规模数据资源。

    

    检索与给定患者临床相似的患者图像可以支持诊断、教学和研究。训练和评估检索系统需要知道不同患者的哪些图像显示相同的发现，而现有数据集规模较小或范围狭窄。我们构建了PMC-PaIR数据集，包含来自56,013篇开放获取的PubMed Central病例报告的86,144对临床相似图像，这些病例报告通常记录罕见或具有挑战性的病例表现。我们开发了一种跟随引文的方法，因为引文通常链接相似患者的病例报告。为了确定两个关联病例在临床上的共同之处，该方法提示大语言模型生成并回答分层问题（从检查模态到诊断和发现），并使用答案通过说明文字匹配图像。专家认为这些图像对具有很高的准确性，且这种问题引导的方法找到的正确图像对数量是直接匹配方法的两倍。在PMC上微调BiomedCLIP……

    arXiv:2611.31465v1 Announce Type: cross  Abstract: Retrieving images of patients clinically similar to a given patient can support diagnosis, teaching and research. Training and evaluating retrieval systems requires knowing which images of different patients show the same finding, and existing datasets are small or narrow in scope. We construct PMC-PaIR, a dataset of 86,144 pairs of clinically similar images from 56,013 open-access PubMed Central case reports, which often document rare or challenging presentations. We develop an approach that follows citations, which often link case reports of similar patients. To determine what two linked cases share clinically, it prompts large language models to generate and answer hierarchical questions, from modality to diagnosis and findings, and to match images by caption using the answers. Experts find the pairs highly accurate, and this question-guided approach finds twice as many correct pairs as direct matching. Fine-tuning BiomedCLIP on PMC
    
[^95]: 分期泄露：LLM辅导系统中的跨会话答案泄露及基于账本的防御

    Leaks by Installments: Cross-Session Answer Disclosure in LLM Tutors and a Ledger-Based Defense

    [https://arxiv.org/abs/2611.31429](https://arxiv.org/abs/2611.31429)

    本文提出一种为每个学习者维护持久化泄露账本的防御机制，通过类型化解题图与门控匹配检测跨会话答案泄露，将模拟实验中通过改编数字练习变体产生的泄露率从100%降至10%。

    

    基于大语言模型（LLM）构建的辅导系统被告知不要泄露答案，但学习者并不需要一次性索要答案。现有的安全防护措施仅在当前对话范围内评判每次回复，因此学习者可以开启新会话、重新措辞表述问题，或对同一问题使用不同数字请求完整的解题过程。我们将这种现象称为跨会话泄露，并提出一种防御方法：为每个学习者维护一个持久的泄露账本。对于学习者提出的每个问题，系统会构建一个带类型的解题图。在回复发布之前，一个门控机制将回复自身的内容（而非学习者的措辞）按结果值和带类型标签的解题结构与账本进行匹配，然后计算学习者此时能够推导出哪些步骤。如果该步骤集合能够到达最终答案或超出预算，则拒绝该回复。在3000个问题的模拟实验中，该账本将通过改编数字的练习变体导致的泄露从100%降低到10%。

    arXiv:2611.31429v1 Announce Type: cross  Abstract: Tutors built on large language models (LLMs) are told not to give away answers, but a learner does not have to ask for the answer all at once. Existing safeguards judge each reply within the current conversation, so a learner can open a new session, reword the problem, or request a worked solution to the same problem with different numbers. We call this cross-session disclosure and propose a defense: a persistent, per-learner disclosure ledger. For every problem a learner brings, the system builds a typed solution graph. Before a reply is released, a gate matches the reply's own content, not the learner's wording, against the ledger by result values and by type-labeled solution structure, then computes which steps the learner could now derive. The reply is withheld if that set reaches the final answer or exceeds a budget. In a simulation of 3,000 problems, the ledger reduced leakage through renumbered practice variants from 100% to 10.
    
[^96]: 学什么 vs. 学怎么做：智能体训练中任务能力与界面能力的分解

    Learning What vs. Learning How: Decomposing Task and Interface Competence in Agentic Training

    [https://arxiv.org/abs/2611.31427](https://arxiv.org/abs/2611.31427)

    该论文将智能体能力分解为“任务能力”（解决任务所需的知识与推理）和“界面能力”（通过特定框架表达知识的能力），揭示了基于框架的训练效果取决于这两种能力的交互方式，并提出多框架蒸馏可将不同框架下的任务知识聚合到共享策略中。

    

    LLM 智能体通过“框架”运行，框架定义了系统提示、动作格式、反馈循环和资源约束。一个模型可能拥有解决任务所需的知识，却无法通过特定的框架表达这些知识。这引出了一个基本问题：当策略在给定框架下失败时，究竟是缺乏任务知识，还是缺乏操作界面的能力？我们区分了智能体能力的两个组成部分：任务能力，即解决任务所需的知识和推理能力；界面能力，即通过给定框架将该知识付诸行动的能力。在四个推理和智能体基准测试中，我们展示了基于框架条件的训练会因这两种能力的交互方式不同而产生截然不同的结果。多框架蒸馏将通过不同框架收集的轨迹中的任务知识聚合到一个共享策略中，但只有在训练保留或模仿……（摘要截断）时，这些知识才能得以实现。

    arXiv:2611.31427v1 Announce Type: new  Abstract: LLM agents operate through harnesses that define system prompts, action formats, feedback loops, and resource constraints. A model may possess the knowledge needed to solve a task yet fail to express that knowledge through a particular harness. This raises a basic question: when a policy fails under a given harness, is it missing task knowledge or the competence to operate the interface? We distinguish two components of agentic competence: task competence, the knowledge and reasoning needed to solve a task, and interface competence, the ability to act on that knowledge through a given harness. Across four reasoning and agentic benchmarks, we show that harness-conditioned training yields vastly different outcomes depending on how these interact. Multi-harness distillation aggregates task knowledge from trajectories collected through different harnesses into a shared policy, but this knowledge is realised only when training preserves or im
    
[^97]: CoRe：联合学习用于理解与生成的上下文表征

    CoRe: Jointly Learning Contextual Representations for Understanding and Generation

    [https://arxiv.org/abs/2611.31397](https://arxiv.org/abs/2611.31397)

    CoRe提出让理解与生成任务联合塑造统一的图像与文本上下文表征，一个从零训练的383M模型在无需任何预训练组件的情况下，在理解能力以及文生图、图生文生成效果上均可与基于预训练组件的更大规模模型相媲美。

    

    我们提出了CoRe，其中理解与生成共同塑造针对图像和针对文本各自的一个上下文表征，并从零开始训练。每个表征既被读取用于理解，又对另一模态进行条件化，同时还是该模态从高斯噪声出发经由连续流（continuous flow）进行生成的空间。在CC12M数据集上训练的383M参数模型，在推理阶段无需预训练语言模型、文本编码器或VAE，CoRe在全局理解任务上与相同数据量的纯对比学习基线相当，并达到COCO-30K FID 10.10，可与基于预训练组件构建的更大规模生成器相媲美。在相同数据和架构的对照比较中，CoRe在两个方向（文本到图像和图像到文本）的生成表现都优于仅由理解任务塑造后冻结的表征，并且在遵循提示方面也优于在与理解无关的表征空间中进行生成。在另一个方向上，使用注意力分割（摘要截断）……

    arXiv:2611.31397v1 Announce Type: cross  Abstract: We introduce CoRe, in which understanding and generation jointly shape one contextual representation for images and one for text, trained from scratch. Each representation is read for understanding, conditions the other modality, and is the space in which that modality is generated by a continuous flow from Gaussian noise. With 383M parameters trained on CC12M and no pretrained language model, text encoder, or VAE at inference, CoRe matches a same-data contrastive-only baseline in global understanding and reaches COCO-30K FID 10.10, comparable to much larger generators built on pretrained components. In controlled comparisons at equal data and architecture, CoRe generates better in both directions (text to image and image to text) than a representation shaped by understanding alone and then frozen, and follows prompts better than generating in a space that understanding does not shape. In the other direction, with an attentive segmenta
    
[^98]: GHC：扩展超连接中的残差接口

    GHC:Expanding Residual Interfaces in Hyper-Connections

    [https://arxiv.org/abs/2611.31350](https://arxiv.org/abs/2611.31350)

    提出分组超连接，通过沿注意力头或FFN通道将分支分组并为每组提供差异化的流读写接口，在开销可忽略的情况下将训练效率较Pre-Norm提升35%、较IHC提升10%，同时克服了增加流数量时的收益递减问题。

    

    超连接通过并行流扩展了残差容量，但将每个分支限制在狭窄的读写接口上：每个分支仍然只读取单一的流混合，并将同一个输出向量的缩放副本写入所有流。为了突破这一瓶颈，我们提出了分组超连接。通过沿注意力头或前馈网络（FFN）通道将每个分支划分为并行组，GHC为每个组提供不同的流混合，并通过可学习的权重将各组的独立输出聚合到每个流中。在密集模型和混合专家模型上的大量实验表明，GHC在参数和内存访问开销几乎可以忽略的情况下，始终优于恒等超连接，相比Pre-Norm将训练效率提升了35%，相比IHC提升了10%。此外，扩展残差接口克服了增加流数量带来的收益递减问题，使得更宽的残差状态能够带来进一步的收益。

    arXiv:2611.31350v1 Announce Type: cross  Abstract: Hyper-Connections expand residual capacity across parallel streams, yet restrict every branch to narrow read and write interfaces: each branch still reads a single stream mixture and writes scaled copies of one output vector to all streams. To lift this bottleneck, we propose Grouped Hyper-Connections (GHC). By partitioning each branch into parallel groups along attention heads or FFN channels, GHC supplies each group with a distinct stream mixture and aggregates their separate outputs into each stream with learned weights. Extensive experiments across dense and Mixture-of-Experts (MoE) models show that GHC consistently improves upon Identity Hyper-Connections (IHC) with negligible parameter and memory-access overhead, increasing training efficiency by 35% over Pre-Norm and 10% over IHC. Moreover, expanding residual interfaces overcomes the diminishing returns of adding streams, allowing wider residual states to yield further gains.
    
[^99]: 人们选择不说的话：面向对话中细粒度信息操纵的多模态基准

    What People Choose Not to Say: A Multimodal Benchmark for Fine-Grained Information Manipulation in Conversations

    [https://arxiv.org/abs/2611.31346](https://arxiv.org/abs/2611.31346)

    提出了 InfoCon——一个多模态基准，通过基于证据的多智能体标注框架，识别和评估对话中隐瞒、泄露、捏造、回避等六种细粒度信息操纵策略及其一致性关系。

    

    arXiv:2611.31346v1 公告类型：新论文。摘要：多模态大语言模型（MLLM）的最新进展提升了社交理解能力，使模型能够识别情绪并区分真实与欺骗行为。然而，现有基准主要关注粗粒度的情绪识别或二元的真话-欺骗检测，忽视了人们在对话中操纵信息的多种方式，例如隐瞒、泄露、捏造、回避和防御性反诉。理解这些行为需要对底层事实、参与者知识和对话语境进行推理，这使得对它们的可靠评估十分困难。为了填补这一空白，我们提出了 InfoCon，一个面向对话中细粒度信息操纵的多模态基准。InfoCon 识别了六种细粒度的信息操纵策略，并从它们之间的事实关系和交互关系中推导出粗粒度的一致性标签。我们开发了一个基于证据的多智能体标注框架来构建并验证可靠的标注。

    arXiv:2611.31346v1 Announce Type: new  Abstract: Recent advances in MLLMs have improved social understanding, enabling models to recognize emotions and distinguish truthful from deceptive behaviors. However, existing benchmarks largely focus on coarse-grained emotion recognition or binary truth-deception detection, overlooking the diverse ways people manipulate information in conversations, such as withholding, leakage, fabrication, evasion, and defensive counterclaim. Understanding these behaviors requires reasoning about underlying facts, participant knowledge, and conversational context, making them difficult to evaluate reliably. To address this gap, we introduce InfoCon, a multimodal benchmark for fine-grained information manipulation in conversations. InfoCon identifies six fine-grained strategies and derives a coarse consistency label from their factual and interactional relations. We develop an evidence-grounded multi-agent annotation framework to construct and verify reliable 
    
[^100]: 大语言模型遗传特征工程

    LLM Genetic Feature Engineering

    [https://arxiv.org/abs/2611.31277](https://arxiv.org/abs/2611.31277)

    该论文提出了一种在单个特征层面而非整个特征集层面执行选择、交叉和变异的新型遗传算法，使遗传算法能够在过程中对新特征进行统计分析，并同时整合个体与群体层面的特征指标，从而改进基于LLM的特征工程自动化。

    

    特征工程是一种（传统上由人工完成的）过程，通过组合、转换或其他方式操作原始数据字段，从现有数据集中构建派生变量，以减少统计噪声，并从原始数据中提取有意义的、特定领域的关注量。近期研究表明，通过整合大语言模型（LLM）和各种元启发式算法来指导有用特征的搜索，在特征工程自动化方面取得了可喜的进展。我们在此方法的基础上，引入了一种新颖的遗传算法，该算法在单个特征层面而非整个工程特征集层面实现选择、交叉和变异操作。由此，我们能够对新生的特征进行过程中的统计分析，从而使遗传算法能够同时将个体和群体层面的特征指标整合到其选择过程中。

    arXiv:2611.31277v1 Announce Type: cross  Abstract: Feature Engineering is the (traditionally manual) process of constructing derived variables from an existing dataset by combining, transforming, or otherwise manipulating the original data fields to reduce statistical noise and extract meaningful, domain-specific quantities of interest from the raw dataset. Recent work demonstrates promising progress toward the au tomation of this process through the integration of large language models (LLMs) and various metaheuristics to guide the search for useful features. We build on this approach by introducing a novel genetic algorithm that implements selection, crossover, and mutation at the individual feature level, rather than at the level of the entire engineered feature set. In doing so, we enable in-process statistical analysis of newly-generated features, thereby enabling the genetic algorithm to simultaneously integrate individual and group-level feature metrics into its selec tion proce
    
[^101]: 声调与词序在翻译中至关重要：跨语言的韵律对应性研究

    Lexical Tone and Word Order Matter in Translation: Prosodic Correspondence Across Languages

    [https://arxiv.org/abs/2611.31276](https://arxiv.org/abs/2611.31276)

    该研究首次对跨越声调与非声调亚洲语言的翻译语音进行了词级韵律对应性系统分析，发现声调和词序显著影响翻译中的韵律对应程度，且其对应水平总体低于先前报告的欧洲语言对的结果。

    

    韵律传达着强调、情感和意图，但其在翻译中的对应关系仍未得到充分探索，尤其是当语言之间在声调和词序上存在差异时。据我们所知，我们首次对翻译语音中的韵律对应性进行了系统的词级分析，涵盖声调与非声调的亚洲语言，并与欧洲语言的研究结果进行比较，同时显式分析声调和词序重排的影响。基于先前关于欧洲语言对之间细粒度韵律对应性的证据，我们研究了这些语言因素如何影响英语与五种语言之间的词级音高和能量对应关系，这五种语言包括具有声调的普通话、泰语和越南语，以及作为额外比较的日语和韩语。研究结果显示，韵律对应性总体上低于先前报告的欧洲语言估计值；调整声调因素后，英语与普通话之间的音高对应性有所提升。

    arXiv:2611.31276v1 Announce Type: new  Abstract: Prosody conveys emphasis, emotion, and intent, yet its correspondence across translation remains underexplored, especially when languages differ in lexical tone and word order. To our knowledge, we present the first systematic word-level analysis of prosodic correspondence in translated speech spanning tonal and non-tonal Asian languages, with comparisons to European-language findings and explicit analyses of lexical tone and word reordering. Building on prior evidence of fine-grained prosody correspondence in European-language pairs, we investigate how these linguistic factors relate to word-level pitch and energy correspondence between English and five languages: Mandarin Chinese, Thai, and Vietnamese, which have lexical tone, and Japanese and Korean as additional comparisons. Correspondence is generally lower than previously reported European-language estimates. Adjusting for lexical tone increases pitch correspondence for English-Man
    
[^102]: 语言模型评判器在定量科学问答中的可靠性：来自钙钛矿光伏的证据

    Reliability of Language-Model Judges for Quantitative Scientific Question Answering: Evidence from Perovskite Photovoltaics

    [https://arxiv.org/abs/2611.31269](https://arxiv.org/abs/2611.31269)

    本文构建了由45,441条钙钛矿光伏实验记录确定性生成真实答案的定量基准，系统评估语言模型评判器的可靠性，发现配备参考答案的前沿评判器Claude Opus可近乎完美地复现真实评分并精确保留架构排名。

    

    评估检索增强语言模型日益依赖自动化模型评判器，但其在定量科学综合任务上的可靠性在很大程度上尚未被测量。本文在一个钙钛矿光伏定量基准上评估自动化评分，其中全部755个真实答案均由45,441条实验器件记录确定性计算得出。我们评估了一个前沿评判器（Claude Opus 5.5）以及两个本地开源权重评判器（Qwen3-14B和Qwen3-8B），针对来自六种检索架构的4,530个自由文本答案，在三种模式下进行测试：参考增强（提供计算所得的参考答案）、证据增强（提供检索段落和查询日志）以及无辅助（仅提供问题和答案）。在配备参考答案的情况下，Claude Opus近乎完美地复现真实评分（Cohen's κ = 0.982 ± 0.003），并完整保留了架构排名（τ = 1.00）。取消参考答案会导致评估……

    arXiv:2611.31269v1 Announce Type: cross  Abstract: Evaluating retrieval-augmented language models increasingly relies on automated model judges, yet their reliability on quantitative scientific synthesis remains largely unmeasured. Here, we evaluate automated grading on a quantitative benchmark for perovskite photovoltaics, where all 755 ground-truth answers are deterministically computed from 45,441 experimental device records. We evaluate a frontier judge (Claude Opus 5.5) alongside two local open-weight judges (Qwen3-14B and Qwen3-8B) across 4,530 free-text answers from six retrieval architectures under three regimes: reference-augmented (with the computed key), evidence-augmented (with retrieved passages and query logs), and unassisted (question and answer alone). Equipped with the reference key, Claude Opus reproduces ground truth near-perfectly (Cohen's $\kappa = 0.982 \pm 0.003$) and preserves exact architectural rankings ($\tau = 1.00$). Withholding the reference causes evaluat
    
[^103]: AhaPro：一个用于探测大语言模型推理中“顿悟”时刻的过程级字谜基准

    AhaPro: A Process-level Rebus benchmark for probing Aha moments in LLM reasoning

    [https://arxiv.org/abs/2611.31268](https://arxiv.org/abs/2611.31268)

    本文提出基于字谜的过程级基准AhaPro来探测大语言模型推理中的“顿悟”时刻，发现模型常常未能采纳顿悟中发现的更优状态，且已采纳的修正甚至可能无益或有害。

    

    现有关于大语言模型“顿悟”时刻的研究大多聚焦于反思性表达或最终修正结果，对其在可见推理过程中所扮演的角色刻画不足。字谜通过字形、空间排列和谐音来编码答案，对关键细节的重新解读可能触发“顿悟”时刻，并引发推理过程与候选答案的转变。我们提出了AhaPro，一个利用字谜来考察模型如何通过“顿悟”时刻重组证据并修正答案的过程级基准。该框架将思维链表示为候选答案、证据与验证状态的序列，并采用分层指标来评估冲突检测、状态改进及后续采纳情况。在十一个模型上的实验中，我们观察到模型经常未能采纳在“顿悟”时刻发现的更优状态，而已采纳的修正有时使状态质量保持不变甚至降低。

    arXiv:2611.31268v1 Announce Type: cross  Abstract: Existing research on Aha moments in large language models has largely focused on reflective expressions or final revision outcomes, leaving their role in the visible reasoning process insufficiently characterized. Rebus puzzles encode answers through character forms, spatial arrangements, and homophony. Reinterpreting key details may trigger Aha moments and shifts in the reasoning process and candidate answers. We introduce AhaPro, a process-level benchmark that uses rebus puzzles to examine how models reorganize evidence and revise answers through Aha moments. The framework represents CoT as sequences of candidate answers, evidence, and verification states, using hierarchical metrics to assess conflict detection, state improvement, and subsequent adoption. Across eleven models, we observe frequent failures to adopt better states discovered during Aha moments, alongside adopted revisions that leave state quality unchanged or reduce it.
    
[^104]: ToneMamba：基于边界重置状态空间模型学习上下文相关的声调轮廓

    ToneMamaba: Learning Contextualized Tonal Contours with Boundary-Reset State Space Models

    [https://arxiv.org/abs/2611.31267](https://arxiv.org/abs/2611.31267)

    ToneMamba通过在音节、词和话语单元层面进行双向状态重置的边界感知状态空间模型，结合节奏条件门控建模时长信息，显著提升了普通话声调识别性能，在会话语音上提升尤为突出。

    

    连续语流中的普通话声调识别极具挑战性，因为语境和时长的变化会重塑基音频率（F0）轮廓。我们提出ToneMamba，这是一种边界感知的状态空间模型，它在音节、词和话语单元（IPU）三个层面用双向状态重置取代静态边界嵌入。模型采用固定权重融合多层次表征，并通过共享权重的双向扫描结合逐通道方向融合来减少参数量。动态音节池化用于构建声调轮廓，节奏条件门控则引入时长信息。ToneMamba在FCU-VOICE-360朗读语音和MCDC-8会话语音上分别达到97.9%和74.8%的准确率，分别超越基线模型ToneNet 0.9和15.9个百分点。消融实验表明，词级重置尤为有效，而节奏条件化在会话语音中带来更大收益，凸显了时长变化在自然口语中的重要作用。

    arXiv:2611.31267v1 Announce Type: new  Abstract: Mandarin tone recognition in connected speech is challenging because contextual and durational variation reshapes F0 contours. We propose ToneMamba, a boundary-aware state space model that replaces static boundary embeddings with bidirectional state resets at syllable, word, and IPU levels. Fixed weights fuse multi-level representations, while shared-weight bidirectional scans use channel-wise directional fusion to reduce parameters. Dynamic syllable pooling forms tonal contours, and rhythm-conditioned gating incorporates durational information. ToneMamba achieves 97.9% and 74.8% accuracy on FCU-VOICE-360 read and MCDC-8 conversational speech, outperforming the baseline (ToneNet) by 0.9 and 15.9 percentage points, respectively. Ablations show that word-level resets are particularly effective, while rhythm conditioning provides greater gains in conversational speech, underscoring the role of durational variation in spontaneous speech.
    
[^105]: SD-Cache：基于状态引导动态缓存的可学习记忆整合

    SD-Cache: Learnable Memory Consolidation with State-Guided Dynamic Caching

    [https://arxiv.org/abs/2611.31265](https://arxiv.org/abs/2611.31265)

    提出SD-Cache，通过状态引导的动态缓存机制，在不预知未来查询的情况下，将LLM原生的KV状态选择性地整合到固定大小的工作记忆中，解决了RAG场景下KV缓存线性增长的问题。

    

    大型语言模型（LLM）越来越多地被部署为需要对海量外部知识进行推理的持久化服务。检索增强生成（RAG）虽然将模型响应建立在相关文档的基础上，但检索到的内容仍需被编码并存储在键值（KV）缓存中。随着这些内容不断积累，缓存呈线性增长，导致内存消耗和推理成本上升。现有的记忆压缩方法要么通过保留缓存状态的一个子集来解决该问题，要么通过学习将文档映射为紧凑潜在表示的上下文压缩器。尽管取得了这些进展，如何在紧张的内存预算下整合信息、同时保持其对未来查询的有用性，仍然是一个挑战。我们研究了如何在无法访问未来查询的情况下，将模型原生的KV状态选择性地整合到一个固定大小的工作记忆中。为此，我们提出了SD-Cache（状态引导……）

    arXiv:2611.31265v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed as persistent services that must reason over large volumes of external knowledge. Retrieval-augmented generation (RAG) grounds model responses in relevant documents, but the retrieved content must still be encoded and stored in the key-value (KV) cache. As this content accumulates, the cache grows linearly, increasing memory consumption and inference cost. Existing memory-compression methods address this problem either by retaining a subset of cached states or by learning context compressors that map documents to compact latent representations. Despite these advances, consolidating information under tight memory budgets while preserving its usefulness for future queries remains challenging. We investigate how the model's native KV states can be selectively consolidated into a fixed-size working memory without access to future queries. To this end, we introduce SD-Cache (State-guide
    
[^106]: 人设条件化如何改变开放式任务执行：行为收益与可靠性成本的配对评估

    How Persona Conditioning Changes Open-Ended Task Execution: A Paired Evaluation of Behavioral Gains and Reliability Costs

    [https://arxiv.org/abs/2611.31264](https://arxiv.org/abs/2611.31264)

    该论文通过配对实验首次系统揭示了人设条件化作为任务引导信号的双面性：它能提升LLM回复的专家内容深度和流程忠实度，但会以17至30个百分点的幅度降低严格指令遵循准确率并引发无依据的细节虚构。

    

    大型语言模型（LLM）经常通过角色或人设类指令进行提示，然而人设条件化此前主要通过角色忠实度或基准测试的总体准确率来评估。我们将人设条件化视为一种任务执行的引导信号，研究它如何改变模型解释、组织和维持回复的方式。基于从真实用户角色扮演日志中提取并扩展为模拟多轮对话延续的配对实例，我们在固定任务内容、约束条件、用户模拟和评估标准的前提下，比较了带人设与不带人设两种条件。在八个模型上的实验表明，人设条件化提升了专家内容深度和流程忠实度，且这些收益在长度控制后依然存在。与此同时，它在所有被评估的模型中将严格指令遵循准确率降低了17至30个百分点，并通过人设诱发的细节扩充增加了缺乏依据的现实世界具体描述。

    arXiv:2611.31264v1 Announce Type: cross  Abstract: Large language models (LLMs) are often prompted with role- or persona-based instructions, yet persona conditioning has mainly been evaluated through character fidelity or aggregate benchmark accuracy. We study persona conditioning as a task-execution steering signal that changes how models interpret, structure, and sustain responses. Using paired instances derived from real user role-play logs and expanded into simulated multi-turn continuations, we compare with-persona and no-persona conditions while fixing task content, constraints, user simulation, and evaluation criteria. Across eight models, persona conditioning improves expert content depth and procedural fidelity, and these gains survive length control. At the same time, it lowers strict instruction-following accuracy by 17 to 30 points in every model evaluated, and increases unsupported real-world specificity through persona-induced elaboration. These results suggest that perso
    
[^107]: 评估大型语言模型在伊朗电气与计算机工程博士入学考试中的表现

    Evaluating Large Language Models on Iranian Doctoral Entrance Exams in Electrical and Computer Engineering

    [https://arxiv.org/abs/2611.31263](https://arxiv.org/abs/2611.31263)

    该研究利用伊朗电气与计算机工程博士入学考试的真实试题（含波斯语与英语双语及依赖配图的题目），通过统一的统计方法系统评估了GPT-5、Gemini 2.5 Pro、Claude Sonnet 4.5和Grok 4四个大型语言模型的答题准确率。

    

    我们在2024年和2025年伊朗国家电气工程（EE）与计算机工程（CE）博士入学考试的多项选择题上评估了四个大型语言模型（LLM）。四本官方试题册共有440道题，其中7道因题干或配图存在歧义而被排除，剩余433道题（EE 186道，CE 247道），其中171道（39.5%）依赖配图。每道题均以原始波斯语和英语翻译两种版本呈现，共产生3,464个评分回答。2025年10月，通过自动化脚本经由单一商业网页接口（Perplexity）以零样本方式（开启推理功能）查询了GPT-5、Gemini 2.5 Pro、Claude Sonnet 4.5和Grok 4。由于每个模型回答相同的题目，所有统计推断均采用配对检验或按题目聚类的模型。准确率分为两个层级：GPT-5（79.9%，95% CI 76.6–83.1）和Gemini 2.5 Pro（78.4%，74.8–81.9）……

    arXiv:2611.31263v1 Announce Type: cross  Abstract: We evaluate four large language models (LLMs) on multiple-choice questions from the 2024 and 2025 Iranian national Ph.D. entrance examinations in Electrical Engineering (EE) and Computer Engineering (CE). Of the 440 questions in the four official booklets, 7 were excluded because the question statement or its figure was ambiguous, leaving 433 questions (186 EE, 247 CE), of which 171 (39.5\%) depend on a figure. Each question was presented in the original Persian and in an English translation, giving 3,464 scored responses. GPT-5, Gemini 2.5 Pro, Claude Sonnet 4.5 and Grok 4 were queried zero-shot, with reasoning enabled, through a single commercial web interface (Perplexity) in October 2025, using an automated script. Because every model answers the same items, all inferences use paired tests or question-clustered models. Accuracy separates into two tiers: GPT-5 (79.9\%, 95\% CI 76.6--83.1) and Gemini 2.5 Pro (78.4\%, 74.8--81.9) are s
    
[^108]: RegioZH：揭示大语言模型对不同地区中文变体理解能力的差异

    RegioZH: Unveiling Comprehension Disparities across Regional Chinese Varieties in LLMs

    [https://arxiv.org/abs/2611.31262](https://arxiv.org/abs/2611.31262)

    该论文提出 RegioZH——首个覆盖中国大陆、台湾、香港、新加坡和马来西亚的 2,607 题地区中文词汇推理基准，揭示了大语言模型在理解不同地区中文变体上的显著差异（如台湾与新加坡之间 29.4 个百分点的准确率差距），并表明部分错误源于检索或表述框架问题而非单纯的知识缺失。

    

    当前的评估往往将中文变体视为单一整体，掩盖了模型对地区用户的理解失败。我们提出了 RegioZH，这是一个包含 2,607 道多项选择题的基准，用于评估覆盖中国大陆、台湾、香港、新加坡和马来西亚的地区性词汇推理能力。该基准由各地区专门的标注者构建并经专家验证，聚焦于本地化习语、俚语、语码混合、礼貌规范以及社会文化表达。在十一个大语言模型上的零样本测试结果显示出巨大差异，其中包括台湾与新加坡之间平均 29.4 个百分点的准确率差距。对五个模型使用角色设定和结构化思维链提示仅带来选择性的提升，表明部分错误源于检索或表述框架问题，而不仅仅是知识缺失。表征、注意力、校准和表面特征分析显示，标记注意力与语义基础之间存在分离、强制选择题中存在过度自信的错误，且本地化词汇密度是……

    arXiv:2611.31262v1 Announce Type: cross  Abstract: Current evaluations often treat Chinese varieties as monolithic, obscuring failures for regional users. We introduce RegioZH, a 2,607-item multiple-choice benchmark for regional lexical reasoning across Mainland China, Taiwan, Hong Kong, Singapore, and Malaysia. Built by region-specific annotators and expert-validated, it targets localized idioms, slang, code-mixing, politeness conventions, and sociocultural expressions. Zero-shot results on eleven LLMs show large disparities, including a 29.4-point average accuracy gap between Taiwan and Singapore. Persona and structured chain-of-thought prompts for five models yield selective gains, suggesting that some errors stem from retrieval or framing rather than only missing knowledge. Representation, attention, calibration, and surface-feature analyses show a dissociation between marker attention and semantic grounding, overconfident forced-choice errors, and localized lexical density as a co
    
[^109]: 项目反应理论能否跨语言泛化？来自跨语言LLM基准测试的证据

    Does Item Response Theory Generalize Across Languages? Evidence from Cross-Lingual LLM Benchmarks

    [https://arxiv.org/abs/2611.31261](https://arxiv.org/abs/2611.31261)

    该研究通过在Multilingual GSM和Global MMLU上对100多个语言模型拟合1PL/2PL/3PL IRT模型，发现题目难度是跨语言最稳定的IRT参数，而猜测和区分度的稳定性依次减弱，因此多语言场景下基于IRT的数据选择需要先验证模型选择并逐一筛查具体题目。

    

    项目反应理论（IRT）在基准测试数据选择中的应用大多局限于单语环境。在本工作中，我们提出这样一个问题：当同一个基准被翻译成不同语言时，IRT如何泛化？我们在Multilingual GSM和Global MMLU上进行了实验，涵盖超过100个语言模型和11种语言。通过为每种语言分别拟合1PL、2PL和3PL IRT模型，我们评估了参数的跨语言一致性，以及经IRT筛选的子集能否在各语言间保持模型排名：结果显示题目难度是最稳定的参数，其次是猜测参数，再其次是区分度参数。差异题目分析还表明，不稳定性的根源可以追溯到特定题目在跨语言翻译过程中发生的偏移。我们的研究结果表明，在多语言环境中基于IRT的数据选择必须同时具备参数感知和题目感知能力，即在假设跨语言一致性成立之前，需要先验证模型选择的适用性并逐一筛查具体题目。

    arXiv:2611.31261v1 Announce Type: cross  Abstract: The application of Item Response Theory (IRT) for data selection in benchmarks has mostly been limited to monolingual settings. In this work, we ask: how does IRT generalize when the same benchmark is translated across languages? We conduct experiments on Multilingual GSM and Global MMLU, covering over 100 language models and 11 languages. Fitting 1PL, 2PL, and 3PL IRT models per language, we assess cross-lingual consistency of parameters and whether IRT-filtered subsets preserve model rankings across languages: item difficulty emerges as the most stable parameter followed by guessing and then discrimination. A differential item analysis also shows that lack of stability can be tracked to specific items shifting under translation across languages. Our findings suggest IRT-based selection in multilingual settings must be both parameter-aware and item-aware, validating model choice and screening individual items before assuming cross-lin
    
[^110]: Transformer在多任务递归上的能力与可解释性

    Capability and Interpretability of Transformers on Multi-task Recurrence

    [https://arxiv.org/abs/2611.31260](https://arxiv.org/abs/2611.31260)

    本文通过有限循环群上的线性递归任务族系统研究了Transformer的能力与可解释性，发现切换明确指定的规则比识别潜在规则更容易，并揭示了模型深度、宽度以及递归的代数结构对泛化能力的影响，同时证明课程学习在规则数量较大时尤为有效。

    

    Transformer被广泛应用于各类语言任务中，但人们对其内部机制的理解仍然有限。受许多语言规则的递归结构启发，我们通过有限循环群上的线性递归任务这一受控任务族，研究了Transformer的能力与可解释性。这些任务涵盖了从执行单个固定规则、从上下文中识别潜在规则，到在明确指定的规则之间进行切换，且包含有输入掩码和无输入掩码两种情形。通过对不同配置的大量实验，我们发现总体而言，对Transformer来说，切换明确指定的规则比识别潜在规则更为简单。我们还识别出影响Transformer泛化能力的若干关键因素，包括深度、宽度以及线性递归的代数结构。当规则数量较多时，课程学习的收益极为显著。借助机制可解释性领域的标准工具……（摘要原文在此处截断）

    arXiv:2611.31260v1 Announce Type: cross  Abstract: Transformers are widely used in various language tasks, with limited understanding of their internal mechanisms. Motivated by the recurrent structure of many linguistic rules, we study the capability and interpretability of Transformers through a controlled family of linear recurrence tasks over finite cyclic groups. These tasks range from executing a single fixed rule to identifying a latent rule from context and switching between explicitly specified rules, with and without input masking. Through extensive experiments across different configurations, we find that switching specified rules is simpler than identifying latent rules for Transformers in general. We also identify how several key factors affect the generalization capability of Transformers, including the depth, width, and the algebraic structure of the linear recurrence. Curriculum training is extremely useful when the number of rules is large. With standard tools in mechan
    
[^111]: RecipeADQA：一个面向多约束食谱推荐的用户感知问答数据集

    RecipeADQA: A User-aware Question-Answering Dataset for Multi-Constraint Recipe Recommendation

    [https://arxiv.org/abs/2611.31259](https://arxiv.org/abs/2611.31259)

    本文提出了RecipeADQA——一个以真实用户行为为驱动、利用大语言模型提取细粒度偏好信号并映射到知识图谱标签模式上的多约束食谱推荐基准问答数据集，同时通过贪心约束选择策略和基于标签的重排序保证数据质量。

    

    面向食谱推荐的问答任务缺乏带有真实标签（ground-truth）的数据集。现有方法通常通过食材或标签约束的简单组合来生成数据集，导致表达不自然、约束不合理，难以反映用户的真实偏好。本文提出了一种用于构建自然语言食谱推荐基准式问答数据集的方法。该方法以真实用户行为为驱动，整合外部用户查询与评论数据，利用大语言模型（LLM）提取细粒度偏好信号，并将其映射到基于知识图谱的标签模式上，从而构建出由偏好约束和食材约束组成的问答数据。此外，该方法还采用贪心策略进行约束选择以控制答案规模，并应用基于标签的重排序来提升数据质量。生成的数据集RecipeADQA包含自然语言问题、结构化约束以及……（摘要在此处被截断）

    arXiv:2611.31259v1 Announce Type: cross  Abstract: Question-answering recipe recommendation tasks lack datasets with ground-truth labels. Existing methods often generate datasets from simple combinations of ingredient or tag constraints, leading to unnatural expressions and unreasonable constraints that are hard to reflect users' real preferences. This paper proposes a construction method for a benchmark-style QA dataset for natural-language recipe recommendation. Driven by real user behavior, our method integrates external user query and review data to extract fine-grained preference signals by LLMs and maps them onto a knowledge-graph-based tag schema, thereby constructing QA data consisting of preference and ingredient constraints. Besides, a greedy strategy for constraint selection is also used to control the answer size, while tag-based re-ranking is applied to improve data quality. The generated dataset RecipeADQA contains natural-language questions, structured constraints, and a
    
[^112]: RECAP：通过上下文压缩训练长时程LLM智能体

    RECAP: Training Long-horizon LLM Agents with Context Compaction

    [https://arxiv.org/abs/2611.31258](https://arxiv.org/abs/2611.31258)

    RECAP提出一种强化学习框架，通过行为失真目标将摘要生成与智能体策略联合训练，使压缩摘要能够保留原始交互历史所诱导的行为，从而提升智能体在长时程任务中的表现。

    

    上下文压缩使大型语言模型智能体能够通过用压缩摘要替换交互历史，在有限的上下文预算内完成长时程任务。然而，将有效压缩与智能体策略进行联合学习仍然具有挑战性，因为摘要的质量会在后续多次交互中影响任务的成功与否。我们提出了RECAP，这是一个基于“摘要应当保留原始交互历史所诱导的行为”这一原则的强化学习（RL）框架。我们通过一个行为失真目标来形式化这一原则，该目标衡量智能体在压缩上下文与原始上下文下后续行为之间的差异。我们证明该失真目标的梯度可以分解为两个互补的学习信号：用于摘要生成的基于失真的奖励，以及用于后续动作的自蒸馏。在Qwen3.5-4B上的实验表明，RECAP持续优于……（原文此处截断）

    arXiv:2611.31258v1 Announce Type: cross  Abstract: Context compaction enables large language model agents to tackle long-horizon tasks within limited context budgets by replacing interaction histories with compressed summaries. However, learning effective compaction jointly with the agent policy remains challenging, as summaries affect task success over many subsequent interactions. We introduce RECAP, a reinforcement learning (RL) framework based on the principle that a summary should preserve the behavior induced by the original interaction history. We formalize this principle with a behavior distortion objective that measures the divergence between the agent's subsequent behavior under compressed and original contexts. We show that the gradient of the distortion objective decomposes into two complementary learning signals: distortion-based reward for summary generation and self-distillation for subsequent actions. Experiments on Qwen3.5-4B show that RECAP consistently outperforms th
    
[^113]: scOPD：面向基于大语言模型的单细胞注释的路由式在策略自蒸馏框架

    scOPD: Routed On-Policy Self-Distillation for LLM-Based Single-Cell Annotation

    [https://arxiv.org/abs/2611.31257](https://arxiv.org/abs/2611.31257)

    scOPD提出了一种面向大语言模型批次级单细胞注释的在策略自蒸馏框架，通过自适应地将合格rollout路由至基于奖励的优化、并利用错误rollout进行自蒸馏，为中间生物推理提供细粒度监督。

    

    动机：大语言模型（LLMs）在单细胞注释方面已展现出潜力，但现有训练方法主要依赖标量结果奖励，这对中间生物推理过程提供的监督十分有限。批次级别的注释尤其具有挑战性，因为它需要同时解决局部细胞身份识别与全局一对一匹配约束，任何一个局部错误都可能影响整个分配结果。此外，单次推理过程无法为定位不确定的预测提供充分证据。这些局限性要求在训练阶段引入细粒度的推理监督，并在推理阶段实现不确定性感知的细化。结果：为应对这些挑战，我们提出了scOPD，一个用于基于大语言模型单细胞注释的在策略自蒸馏框架。在推理微调之后，scOPD自适应地将合格的rollout路由至基于奖励的优化，而错误的rollout则接收先验……（原文摘要在此处被截断）

    arXiv:2611.31257v1 Announce Type: cross  Abstract: Motivation: Large language models (LLMs) have shown promise for single-cell annotation, but existing training methods mainly rely on scalar outcome rewards that provide limited supervision for intermediate biological reasoning. Batch-level annotation is particularly challenging because it requires jointly resolving local cell identities and a global one-to-one matching constraint, such that one local error can affect the entire assignment. Moreover, a single inference pass provides insufficient evidence for locating uncertain predictions. These limitations call for fine-grained reasoning supervision during training and uncertainty-aware refinement during inference.   Results: To address these challenges, we propose scOPD, an on-policy self-distillation framework for LLM-based single-cell annotation. After reasoning fine-tuning, scOPD adaptively routes qualified rollouts to reward-based optimization, while erroneous rollouts receive pri
    
[^114]: 面向投机解码直接最大化在策略期望接受长度

    Directly Maximizing On-policy Expected Accepted Length for Speculative Decoding

    [https://arxiv.org/abs/2611.31256](https://arxiv.org/abs/2611.31256)

    FreDraft通过在策略训练直接最大化草稿的期望接受长度，并利用目标模型词表分布解析边缘化接受率以降低梯度方差，从而更有效地提升投机解码的加速效果。

    

    投机解码通过使用轻量级草稿模型提出候选token、再由目标模型进行验证，从而加速大语言模型推理。现有草稿模型训练通常优化token级分布对齐，但这并不能直接优化被接受前缀的长度：一旦某个草稿token被拒绝，其后所有草稿token都会被丢弃。我们提出FreDraft，一种在策略训练方法，直接最大化从当前草稿模型采样得到的草稿的期望接受长度。该目标考虑了首次拒绝截断，并引入前缀感知的信用分配，根据每个token对当前及后续token接受的贡献进行加权。为了高效优化该目标，我们解析地边缘化token级接受率，利用目标模型的词表分布来降低梯度方差。我们进一步重新分配该分布级梯度的一部分……（原文摘要至此截断）

    arXiv:2611.31256v1 Announce Type: cross  Abstract: Speculative decoding accelerates large language model inference by using a lightweight draft model to propose tokens for verification by a target model. Draft training commonly optimizes token-level distribution alignment, but this does not directly optimize the length of the accepted prefix: once a draft token is rejected, all subsequent draft tokens are discarded. We propose FreDraft, an on-policy training method that directly maximizes the expected accepted length of drafts sampled from the current draft model. This objective accounts for first-rejection truncation and induces prefix-aware credit assignment, weighting each token by its contribution to the acceptance of the current and subsequent tokens. To optimize the objective efficiently, we analytically marginalize token-level acceptance rate, using the target model's vocabulary distribution to reduce gradient variance. We further redistribute part of this distribution-level gra
    
[^115]: INCLUDE-2.0：解耦多语言大语言模型评估中的知识鸿沟

    INCLUDE-2.0: Disentangling Knowledge Gaps in Multilingual LLM Evaluation

    [https://arxiv.org/abs/2611.31255](https://arxiv.org/abs/2611.31255)

    INCLUDE-2.0是一个覆盖86个国家91种语言、包含22万余样本的区域感知多语言事实评估基准，通过标注每道题所需的区域知识类型（而非将语言简单等同于区域），实现了对多语言大模型知识鸿沟的精细化受控分析。

    

    当前的多语言基准测试仅覆盖现代预训练语料库中一小部分语言，且常常将语言能力与回答基于特定地区和社区的问题的能力混为一谈。我们推出了INCLUDE-2.0，这是一个用于区域感知多语言事实性评估的大规模基准，涵盖来自86个国家的91种语言（共127个语言-国家组合），包含超过22万个从原生区域来源精心筛选的样本。与现有通常将语言作为区域代理的基准不同，INCLUDE-2.0区分了以下几类问题：答案与上下文无关的问题、因国家而异的问题、涉及某地被广泛记载事实的问题，以及主要依赖当地语言和文化来源表达的知识的问题。每个条目都标注了其区域知识类型和领域，并配有英文翻译，从而支持对模型表现进行受控分析。

    arXiv:2611.31255v1 Announce Type: cross  Abstract: Current multilingual benchmarks cover only a fraction of the languages represented in modern pretraining corpora and often conflate language proficiency with the ability to answer questions grounded in specific regions and communities. We introduce INCLUDE-2.0, a large-scale benchmark for region-aware multilingual factual evaluation, which covers 91 languages from 86 countries (127 total language--country pairs), comprising over 220K curated samples drawn from native regional sources. Unlike existing benchmarks that often treat language as a proxy for region, IINCLUDE-2.0 distinguishes between questions whose answers are independent of context, vary by country, concern widely documented facts about a place, or depend on knowledge primarily expressed through local linguistic and cultural sources. Each item is annotated with its regional knowledge type and domain, and paired with an English translation, enabling controlled analysis of th
    
[^116]: 你未曾设置的日期：系统提示日期作为大语言模型评估中的隐藏变量

    The Date You Didn't Set: The System-Prompt Date as a Hidden Variable in LLM Evaluation

    [https://arxiv.org/abs/2611.31254](https://arxiv.org/abs/2611.31254)

    本文揭示系统提示中自动填入的日期是LLM评估中被忽视的隐藏变量：仅改变日期就会使中位数6.6%的基准题目翻转正误，在对抗性搜索下甚至使模型准确率损失高达36.5分，因此评估时必须固定并报告系统提示日期。

    

    arXiv:2611.31254v1 Announce Type: cross 摘要：许多经过指令微调的模型的系统提示中都带有日期，通常是今天的日期，无论是对话请求还是智能体请求都会自动填入，因此即使评估本身没有变化，每天发送的提示也可能不同。本文探讨了这一隐藏输入会在多大程度上改变评估结果。在六个指令微调模型和四个基准测试上，即使提示、解码和配置全部固定，每个模型都受到影响：两个合理的日期（2024–2026年）会使中位数6.6%的测试项在正确与错误之间翻转。即使仅仅把评估推迟一天再运行，翻转的题目数量也与改变批处理规模相当，而实验设置中没有任何东西被改动。此外，利用语法约束的贪心坐标梯度（GCG）变体搜索任意年份的日期，我们发现某些日期会使Llama-3.x模型在保留集上的准确率损失高达36.5分。除了准确率之外，切换到另一个合理日期还会改变LLM裁判的（判决结果）……

    arXiv:2611.31254v1 Announce Type: cross  Abstract: The system prompt of many instruction-tuned models carries a date, usually today's, filled in automatically for conversational and agentic requests alike, so an unchanged evaluation can send a different prompt each day. In this paper, we explore how much this hidden input changes evaluation outcomes. Across six instruction-tuned models and four benchmarks, with prompts, decoding, and configuration fixed, every model is affected: two plausible dates (2024--2026) switch a median $6.6\%$ of test items between correct and incorrect. Even rerunning the evaluation one day later flips about as many items as changing the batch size, although nothing in the setup was changed. In addition, using a grammar-constrained variant of Greedy Coordinate Gradient (GCG) to search dates of any year, we find dates that cost the Llama-3.x models up to $36.5$ points of held-out accuracy. Beyond accuracy, switching to another plausible date alters LLM-judge ve
    
[^117]: 图像生成中人类价值的具象化

    Instantiation of Human Values in Image Generation

    [https://arxiv.org/abs/2611.31253](https://arxiv.org/abs/2611.31253)

    该研究基于施瓦茨人类价值理论构建分析框架，通过28,000张生成图像揭示四个主流文生图模型如何将成功、关爱、自由等抽象价值狭窄地具象化为重复出现的视觉原型，并将其映射到社会构型分类体系以揭示模型系统性的价值呈现偏向。

    

    文生图（T2I）模型日益融入日常视觉文化，然而人们对其如何将抽象的人类价值转化为视觉场景却知之甚少。为填补这一空白，我们引入了一个基于施瓦茨（Schwartz）基本人类价值理论的框架，用于研究“价值具象化”（value instantiation）：即成功、关爱或自由等理想在视觉上得以实现的过程。我们使用来自70个正向与负向价值条目的以人物为中心的提示词，在四个最先进的文生图模型上生成了28,000张图像。我们识别出视觉原型——即反复出现的主体、动作与场景的组合——并用它们来量化模型具象化这些价值的范围有多狭窄，以及这种广度与人类对相同条目的描述相比有何差异。随后，我们将这些原型映射到一个涵盖场景设定、社交性、参与方式和社会语调的社会构型分类体系上，揭示出哪些内容被模型系统性地突显或忽视。

    arXiv:2611.31253v1 Announce Type: cross  Abstract: Text-to-image (T2I) models are increasingly embedded in everyday visual culture, yet little is known about how they translate abstract human values into visual scenes. We address this gap by introducing a framework, grounded in Schwartz's theory of basic human values, for studying value instantiation: the process by which ideals such as success, care, or freedom are visually realized. Using person-centered prompts from 70 positive and negative value items, we generate 28,000 images across four state-of-the-art T2I models. We identify visual prototypes, recurring combinations of subjects, actions, and settings, and use them to quantify how narrowly models instantiate these values and how this breadth compares with human descriptions of the same items. We then map these prototypes onto a taxonomy of social configurations spanning setting, sociality, mode of engagement, and social register, revealing what is systematically foregrounded or
    
[^118]: DAGGER：面向生成式AI工作流的无部署质量评估

    DAGGER: Deployment-Free Quality Estimation for Generative AI Workflows

    [https://arxiv.org/abs/2611.31252](https://arxiv.org/abs/2611.31252)

    DAGGER框架通过将生成式AI工作流建模为图结构，实现了无需穷举部署与性能分析即可高效准确地估算工作流端到端质量，突破了现有方法在估算成本与准确性之间的权衡。

    

    估算生成式AI工作流的质量极具挑战性，因为这些工作流由多个阶段组成，错误可能在各阶段之间传播，使得端到端性能不仅取决于单个模型的准确性，还取决于各模型的输出对下游阶段的影响。每个阶段都可以从多个候选模型中选择一个。若要通过部署并评估所有可能的模型组合来估算端到端工作流质量，其成本将高得令人望而却步。因此，现有方法面临估算成本与准确性之间的权衡：低成本的方法往往产生不可靠的质量估算，而准确的方法则需要部署和评估大量的工作流配置。本文提出DAGGER，一个无需穷举部署和性能分析即可估算生成式AI工作流质量的框架。DAGGER将工作流表示为一个图，其中节点对应于各个阶段，边……

    arXiv:2611.31252v1 Announce Type: cross  Abstract: Estimating the quality of generative AI workflows is challenging because these workflows comprise multiple stages, and errors can propagate across them, making end-to-end performance depend not only on the accuracy of individual models but also on how their outputs affect downstream stages. Each stage can use one of many candidate models. Estimating end-to-end workflow quality by deploying and evaluating every resulting model configuration is therefore prohibitively expensive. As a result, existing approaches face a tradeoff between estimation cost and accuracy: inexpensive methods often produce unreliable quality estimates, while accurate methods require deploying and evaluating many workflow configurations. In this paper, we introduce DAGGER, a framework for estimating the quality of generative AI workflows without exhaustive deployment and profiling. DAGGER represents a workflow as a graph, in which nodes correspond to stages and ed
    
[^119]: 大语言模型对癌症治疗查询响应中的验证偏见：一项析因横断面研究

    Validation Bias in Large Language Model Responses to Cancer Treatment Queries: A Factorial Cross-Sectional Study

    [https://arxiv.org/abs/2611.31251](https://arxiv.org/abs/2611.31251)

    这项析因横断面研究通过对14个大语言模型、1,272段对话的分析，证实了LLM在应对与证据不一致的癌症治疗信念时存在“验证偏见”——倾向于认同而非纠正用户的错误信念，这对患者安全构成潜在风险。

    

    大语言模型（LLM）聊天机器人被越来越多地用于患者获取癌症信息，但当用户提出与证据不一致的治疗信念时，其安全性尚未被系统表征。我们假设LLM表现出“验证偏见”，即认同而非纠正此类信念，并通过2×2×2析因设计对此进行评估，该设计交叉了三个因素：对话格式（单轮 vs 多轮）、治疗类型（循证治疗 vs 替代治疗）和查询效价（正面 vs 负面）。问题框架设定为证据一致（支持循证治疗或质疑替代疗法）或证据不一致（推广替代治疗或质疑循证疗法）。我们分析了来自6家厂商的14个LLM的1,272段对话（共6,360次交互），由经人工标注验证的LLM评判器（GPT-5-mini）依据6维度安全量表进行评分。主要结局指标为不安全响应率，定义为

    arXiv:2611.31251v1 Announce Type: cross  Abstract: Large language model (LLM) chatbots are increasingly used by patients seeking cancer information, but their safety when users present evidence-misaligned beliefs about treatment remains uncharacterized. We hypothesized that LLMs exhibit validation bias, affirming rather than correcting such beliefs, and evaluated this with a 2x2x2 factorial design crossing conversation format (single-turn vs multi-turn), treatment type (evidence-based vs alternative), and query valence (positive vs negative). Framing was evidence-aligned (supporting evidence-based treatments or questioning alternative therapies) or evidence-misaligned (promoting alternative treatments or questioning evidence-based therapies). We analyzed 1,272 conversations (6,360 interactions) across 14 LLMs from 6 vendors, scored by an LLM judge (GPT-5-mini) validated against human annotation on a 6-dimension safety rubric. The primary outcome was the unsafe response rate, defined as
    
[^120]: 通过循环残差解锁预训练大语言模型中循环结构的潜力

    Unlocking the Power of Loops in Pretrained LLMs with Loop Residuals

    [https://arxiv.org/abs/2611.31094](https://arxiv.org/abs/2611.31094)

    LoopRes通过让每个循环层借助可训练的循环感知键直接关注早期循环的输出（并限制在残差窗口内），使预训练大语言模型无需从头训练即可将循环结构转化为显著且一致的增益。

    

    循环Transformer通过迭代共享的层块来增加计算量而无需增加参数，从而提供更强的性能和测试时扩展能力。然而，迄今为止，这种能力需要从头开始训练，且对大语言模型只能带来有限的收益。将现有模型微调成循环结构虽然成本低得多，但相比SFT也只能带来边际提升。我们提出LoopRes，一种有效、可扩展且简单的方法，能够将循环转化为预训练大语言模型的显著且一致的增益。LoopRes建立在注意力残差的基础上，让每个循环层能够直接关注早期循环的输出。由于这些输出是相同层迭代产生的结果，彼此越来越难以区分，我们为每个输出提供一个可训练的、感知循环的键来编码其结构身份。关注所有早期循环还允许各层跳过中间循环，因此我们将每层限制在最近W个循环组成的残差窗口内，强制每个循环进行逐步构建

    arXiv:2611.31094v1 Announce Type: new  Abstract: Looped Transformers iterate a shared block of layers to increase computation without adding parameters, offering stronger performance and test-time scaling. However, this power has so far required training from scratch and yielded only modest gains for LLMs. Finetuning existing models into loops is far cheaper, yet also brings only marginal gains over SFT. We propose LoopRes, an effective, scalable, and simple approach that turns loops into significant and consistent gains for pretrained LLMs. Building on Attention Residuals, LoopRes lets each looped layer attend directly to the outputs of earlier loops. Since these outputs are iterates of the same layers and become harder to tell apart, we give each a trainable loop-aware key that encodes its structural identity. Attending to all earlier loops also lets layers skip intermediate ones, so we restrict each layer to a residual window of the latest W loops, forcing every loop to build increm
    
[^121]: LimeTree：基于树图的线性内存图神经网络，用于资源高效的语言建模与上下文外推

    LimeTree: Linear-Memory Graph Neural Networks over Tree-Graphs for Resource-Efficient Language Modeling and Context Extrapolation

    [https://arxiv.org/abs/2611.30934](https://arxiv.org/abs/2611.30934)

    LimeTree是一种内存与计算均为线性复杂度的树图语言模型架构，通过在语言建模训练中集成KV检索任务，仅用极低的计算成本（4块GPU训练6小时）便实现了19的低困惑度，并能将上下文从512外推至16,384个token且保持100%的检索准确率。

    

    近年来，寻找Transformer计算高效替代方案的研究催生了众多架构；然而，大多数架构在联想回忆（如键值检索）和长度外推方面存在严重缺陷，即使是标准Transformer在这些任务上也表现挣扎。本文提出了LimeTree，这是一种在内存和计算两方面均具有线性复杂度的序列建模架构。通过将合成的KV检索任务直接集成到标准语言建模训练中，LimeTree在实现较低困惑度的同时保持了可靠的召回能力。在严格的计算约束下，论文评估了该架构的两个变体在WikiText-103和PG-19数据集上的表现。在WikiText-103上，LimeTree仅在四块Quadro RTX 8000 GPU上训练6小时即达到19的token级困惑度，并且即使从512的训练上下文长度外推至16,384的评估长度，仍保持100%的KV检索准确率。在PG-19上，该模型达到了token级的困惑度……

    arXiv:2611.30934v1 Announce Type: new  Abstract: The search for computationally efficient alternatives to the Transformer has yielded numerous architectures in recent years; however, most exhibit severe weaknesses in associative recall-such as key-value (KV) retrieval-and length extrapolation, where even standard Transformers struggle. This paper introduces LimeTree, a linear-complexity architecture in both memory and compute for sequence modeling. By integrating synthetic KV retrieval tasks directly into standard language modeling training, LimeTree achieves strong perplexity alongside reliable recall. Two variants were evaluated on WikiText-103 and PG-19 under strict compute constraints. On WikiText-103, LimeTree reaches a token-level perplexity of 19 in just 6 hours on four Quadro RTX 8000 GPUs, while maintaining 100\% KV retrieval accuracy even when extrapolating from a training context length of 512 to an evaluation length of 16,384. On PG-19, the model achieves a token-level perp
    
[^122]: 可验证的在线策略蒸馏

    Verifiable On-Policy Distillation

    [https://arxiv.org/abs/2611.30325](https://arxiv.org/abs/2611.30325)

    VOPD通过将OPD的隐式奖励与验证器反馈置于共同的奖励空间，将稀疏的验证器反馈转化为密集的、以验证结果为依据的教师监督，从而更有效地提升语言模型的推理能力。

    

    基于可验证奖励的强化学习（RLVR）已成为提升语言模型推理能力的强大方法，但其监督信号本质上是稀疏的：序列级验证器奖励只能通过采样的轨迹进行传播，无法为其他可选动作提供直接信号。在线策略蒸馏（OPD）则提供了一种互补的监督形式，通过在整个下一词元分布上提供密集的教师反馈，但这一信号并未建立在经过验证的任务结果之上。简单地共同优化这两个目标并不能将这两种信号耦合起来：验证器反馈依然稀疏，且无法为密集的教师监督提供信息。我们提出了可验证在线策略蒸馏（VOPD），它将稀疏的验证器反馈转化为密集的、以验证器为依据的监督信号。我们的关键观察是，OPD的隐式奖励视角能够将教师监督和验证器反馈置于一个共同的奖励空间中，同时保留……

    arXiv:2611.30325v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) has become a powerful approach for improving language-model reasoning, but its supervision is inherently sparse: a sequence-level verifier reward is propagated only through sampled trajectories and provides no direct signal for alternative actions. On-policy distillation (OPD) offers a complementary form of supervision by providing dense teacher feedback over the full next-token distribution, but this signal is not grounded in verified task outcomes. Simply co-optimizing the two objectives does not couple these signals: verifier feedback remains sparse and does not inform the dense teacher supervision. We introduce Verifiable On-Policy Distillation (VOPD), which transfers sparse verifier feedback into dense, verifier-grounded supervision. Our key observation is that the implicit-reward view of OPD places teacher supervision and verifier feedback in a common reward space while retainin
    
[^123]: 桑切斯、费霍还是阿巴斯卡尔？LLM角色画像阅读西班牙“11·29”大选三大政党领导人X平台帖子时的发言人归因与意识形态一致性

    S\'anchez, Feij\'oo, or Abascal? Speaker Attribution and Ideological Congruence in LLM Personas Reading Posts on X of the Leaders of the Three Main Parties of the 29N General Election in Spain

    [https://arxiv.org/abs/2611.29765](https://arxiv.org/abs/2611.29765)

    本文首次通过大规模发言人归因实验验证了LLM角色画像会对发言者身份（而不仅是文本内容本身）作出反应，发现点名西班牙三大政党领导人会系统性降低对相同文本的支持度，且阿巴斯卡尔带来的负面效应约为桑切斯和费霍的两倍。

    

    来源线索是政治传播的核心要素，然而以大语言模型（LLM）角色画像作为合成受众的接受研究，尚未检验这些画像是否会对“谁在说话”作出反应，而不仅仅是对“说了什么”作出反应。着眼于2026年11月29日的西班牙大选，我们开展了一项发言人归因实验，使用PSOE、PP和VOX三党领导人在四个主题（经济与就业、家庭开支、移民与腐败）上的1,075条匿名化X帖子，以及791个基于价值观的、由LLM生成的独特角色画像。每条帖子被分配给300个按主题筛选的角色画像档案——其中150个基于英格尔哈特-韦尔策尔文化地图，150个基于道德基础理论——在四种条件下进行测试：无作者、桑切斯、费霍或阿巴斯卡尔，共产生322,470组完整的“角色画像-帖子”四元组。研究结果表明，点名任何领导人都降低了对相同文本的支持度，其中阿巴斯卡尔带来的惩罚效应几乎是桑切斯和费霍的两倍。（注：原文摘要在此处截断）

    arXiv:2611.29765v1 Announce Type: new  Abstract: Source cues are central to political communication, yet reception studies with large language model (LLM) personas as synthetic audiences have not tested whether personas respond to who speaks and not only to what is said. With Spain's general election of 29 November 2026 in mind, we run a speaker-attribution experiment using 1,075 anonymised X posts by the leaders of PSOE, PP and VOX on four topics (the economy and employment, household costs, immigration and corruption) and 791 distinct value-grounded, LLM-generated personas. Each post was assigned to 300 topic-selected persona profiles, 150 grounded in the Inglehart-Welzel cultural map and 150 in Moral Foundations Theory, under four conditions: no author, S\'anchez, Feij\'oo or Abascal, yielding 322,470 complete persona-post quartets. Naming any leader lowered support for identical text, with the penalty for Abascal nearly twice as large as those for S\'anchez and Feij\'oo. The penalt
    
[^124]: LITMUS——基于文献的用例映射与风险卡综合测试平台

    LITMUS - Literature Informed Testbed for Mapping Usecases and Synthesizing Risk-Cards

    [https://arxiv.org/abs/2611.29034](https://arxiv.org/abs/2611.29034)

    本文提出LITMUS，一个结构化流水线，能够根据给定用例自动生成定制化数据集与风险卡，并利用同行评审文献提供可追溯证据，将每个风险与具体用例紧密关联，弥补了现有方法忽视用例场景的不足。

    

    大语言模型（LLM）和人工智能（AI）在现实世界应用中的快速普及，加剧了对系统化、面向安全的部署框架的需求。现有方法通常优先围绕LLM和AI架构本身来构建数据集和风险卡，却很少考虑驱动模型选择的特定用例。这种疏漏是有问题的，因为任何实际的AI部署都始于详细的用例描述，而该描述正是评估上下文风险的基础。在本文中，我们提出了一个结构化流水线，给定一个用例，它能够自动生成定制的数据集以及配套的风险卡。该流程确保所识别的风险与用例紧密耦合，并提供来自同行评审文献的可追溯证据，以证实每一个风险与用例之间的关系。先前的声明验证方法（FActSco...

    arXiv:2611.29034v1 Announce Type: new  Abstract: The proliferation of large language models (LLMs) and Artificial Intelligence (AI) in real-world applications has heightened the need for systematic, safety-oriented deployment frameworks. Existing approaches typically prioritize the construction of datasets and risk cards around the LLM and AI architecture itself, yet they provide scant consideration for the specific use-case that drives model selection. This omission is problematic because any practical AI deployment begins with a detailed use-case description that informs the assessment of contextual hazards. In this paper, we introduce a structured pipeline that, given a use-case, automatically produces a tailored dataset and an accompanying risk card. The process ensures that identified risks are tightly coupled to the use-case, and it offers traceable evidence from peer-reviewed literature to substantiate each risk to use-case relationship. Prior claim-verification methods (FActSco
    
[^125]: Grammar-as-a-Draft（语法即草稿）：在语法约束的结构化输出中让每次推测解码步产生更多Token

    Grammar-as-a-Draft: More Tokens per Speculative Decoding Step on Grammar-Constrained Structured Outputs

    [https://arxiv.org/abs/2611.28349](https://arxiv.org/abs/2611.28349)

    该论文提出Grammar-as-a-Draft（GaaD），将语法引擎作为第二草稿模型，把语法强制Token与推测草稿模型的提议合并为更宽的草稿行，使语法约束的结构化输出下每次推测解码步可验证更多Token，且无需新增内核或编译。

    

    在结构化输出（JSON模式、正则表达式、工具调用）场景下，推测式草稿模型无法感知语法约束，会反复提议无效的键名、分隔符和缩进，而语法引擎其实早已知道接下来必须出现的Token。我们提出了Grammar-as-a-Draft（GaaD），它将语法视为第二个草稿模型。在每次验证步骤之前，调度器将草稿模型的提议与语法强制Token合并为一行草稿：匹配的Token保留为GRAMMAR状态，在提议偏离强制路径之处将其截断，并由强制Token填满比草稿提议更宽的行的剩余部分。验证步骤无需计算GRAMMAR草稿的logits即可接受它们，可与任何草稿模型组合，且在关键路径上无需新增内核或编译。由于语法固定的是文本而非Token，提交规范的强制Token相对于使用语法掩码的目标解码会改变输出分布。

    arXiv:2611.28349v1 Announce Type: cross  Abstract: Under structured outputs (JSON schemas, regexes, tool calls), speculative drafters do not see the grammar constraint and repeatedly propose invalid keys, delimiters, and indentation, while the grammar engine already knows which tokens must come next.   We present Grammar-as-a-Draft (GaaD), which treats the grammar as a second drafter. Before each verify step, the scheduler merges the drafter's proposal with grammar-forced tokens into one draft row: matching tokens stay as GRAMMAR, the proposal is cut where it leaves the forced path, and forced tokens fill the rest of a row wider than the drafter's proposal. The verify step accepts GRAMMAR drafts without computing their logits, composing with any drafter with no new kernels or compilation on the critical path. Because grammars fix text rather than tokens, committing canonical forced tokens changes the output distribution relative to grammar-masked target decoding.   We implement GaaD in
    
[^126]: 静默仲裁：大型语言模型如何解决同等权威的冲突指令

    Silent Arbitration: How Large Language Models Resolve Co-Authoritative Conflicting Instructions

    [https://arxiv.org/abs/2611.27444](https://arxiv.org/abs/2611.27444)

    研究发现大语言模型在面对同等权威的冲突人格指令时会进行“静默仲裁”——虽然在直接检查中72%会承认冲突，但在实际任务响应中仅有22%明确告知用户，其余情况下模型默默采用某一方的立场而不透明地解决冲突。

    

    大型语言模型越来越多地接收关于角色、价值观、利益和沟通风格的高级指令。即使提示词赋予这些指令同等权威且未提供优先级规则，它们之间也可能发生冲突。我们研究了同权威人格冲突，即单个提示词同时分配两个对立的人格。基于已发表源材料中的八种冲突类型，我们在五个大语言模型上测量了冲突承认度、解决策略和方向性立场。模型在72%的直接检查中承认不兼容性，但在任务响应中仅有22%明确表达出来，这揭示了“静默仲裁”现象。支配性策略占双重人格响应的45%，而35%被标记为妥协的响应仍保留了方向性倾向。提示词框架对仲裁可见形式的改变远大于普通的改述变化。这些发现表明，评估指令遵循应当……

    arXiv:2611.27444v1 Announce Type: new  Abstract: Large language models increasingly receive high-level instructions about roles, values, interests, and communication style. Such instructions can conflict even when a prompt assigns them equal authority and provides no priority rule. We study co-authoritative persona conflict, in which a single prompt jointly assigns two opposing personas. Across eight conflict types grounded in published source material and five LLMs, we measure conflict acknowledgment, resolution strategy, and directional stance. Models acknowledge incompatibility in 72% of direct checks but explicitly surface it in only 22% of task responses, revealing silent arbitration. Dominance accounts for 45% of dual-persona responses, and 35% of compromise-labeled responses retain directional lean. Prompt framing further changes the visible form of arbitration substantially more than ordinary paraphrase variation. These findings show that evaluating instruction following should
    
[^127]: RL专家的可组合软提示蒸馏：迈向持续后训练

    Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training

    [https://arxiv.org/abs/2611.27376](https://arxiv.org/abs/2611.27376)

    提出可组合软提示蒸馏（CSPD）方法，将每个RL专家独立蒸馏为软提示并通过拼接方式组合，无需重新整合模型即可实现持续后训练，性能媲美多教师蒸馏且不损害目标领域之外的能力。

    

    RL后训练通常会产生领域专家模型，这些专家随后必须被整合到一个模型中，而随着新专家不断出现，这种整合就成为了一个持续的过程。权重合并和多教师在线策略蒸馏（MOPD）在每添加一个专家时都必须重新整合整个模型，并且可能损害目标领域之外的能力。我们提出了可组合软提示蒸馏（CSPD），该方法保持RL之前的模型冻结，将每个RL专家独立地蒸馏到一个软提示中，并通过直接拼接各提示来组合专家，无需联合训练。单个软提示就足以蒸馏一个RL专家，在23个数学、代码和智能体设置中的22个里恢复了至少90%的教师增益。独立训练的提示之间可以良好组合。在多专家设置中，平均性能随着每个专家的加入而提升，完整组合的性能与MOPD相当（67.83 vs. 67.79）。该组合在六个……（摘要截断）

    arXiv:2611.27376v1 Announce Type: cross  Abstract: RL post-training typically produces domain experts that must then be consolidated into one model, and as new experts keep arriving, this consolidation becomes a continual process. Weight merging and multi-teacher on-policy distillation (MOPD) must re-consolidate the whole model whenever an expert is added, and can erode capabilities outside the target domains. We propose Composable Soft-Prompt Distillation (CSPD), which keeps the pre-RL model frozen, distills each RL expert independently into a soft prompt, and composes experts by concatenating their prompts without joint training. A single soft prompt suffices to distill an RL expert, recovering at least 90% of the teacher's gain in 22 of 23 math, code, and agent settings. Independently trained prompts compose well. In a multi-expert setting, the average rises with every expert added, and the full composition matches MOPD (67.83 vs. 67.79). The composition generalizes best across six 
    
[^128]: 基于语言模型的个性化旅行规划器

    LM-based Personalized Travel Planner

    [https://arxiv.org/abs/2611.25531](https://arxiv.org/abs/2611.25531)

    提出了一种层次化多智能体旅行规划框架，由监督智能体协调多个专门智能体，并基于多维评分机制进行迭代优化，在TravelPlanner基准上将预算约束规划性能提升了10%。

    

    有效的旅行规划需要处理多个信息源、管理约束条件，并根据用户偏好定制推荐。虽然大语言模型（LLM）带来了动态的、上下文感知的决策能力，但单智能体模型在复杂的多步推理、约束遵循和模块化执行方面往往存在困难。为了克服这些局限性，我们提出了一个层次化多智能体框架，其中一个监督型聊天机器人智能体负责协调专门的智能体，分别用于行程验证、用户查询构建、实时数据检索和基于约束的行程生成。通过集成LLM推理、外部API以及基于确定性多维评分框架的迭代优化机制，我们的方法确保了适应性、高效性和个性化。在著名的TravelPlanner基准上的评估表明，预算约束规划性能提升了10%，在更广泛的约束集规划方面也有所提升……

    arXiv:2611.25531v1 Announce Type: new  Abstract: Effective travel planning requires handling multiple information sources, managing constraints, and tailoring recommendations to user preferences. While LLMs have introduced dynamic, context-aware decision-making, single-agent models often struggle with complex multi-step reasoning, constraint adherence, and modular execution. To overcome these limitations, we propose a hierarchical multi-agent framework where a supervising chatbot agent coordinates specialized agents for schedule validation, user query formulation, real-time data retrieval, and constraint-aware itinerary generation. By integrating LLM reasoning, external APIs, and iterative refinement mechanisms based on a deterministic, multi-dimensional scoring framework, our approach ensures adaptability, efficiency, and personalization. Evaluations on the well-known TravelPlanner benchmark demonstrate improvements in Budget Constraint planning by 10\%, wider constraints set-based pl
    
[^129]: 可以安全省略，也值得省略吗？当LLM智能体跳过另一方的消息时，保持答案不变并降低延迟

    Safe to Omit, Worth Omitting? Keeping the Answer and Cutting Latency When an LLM Agent Skips Another's Message

    [https://arxiv.org/abs/2611.23570](https://arxiv.org/abs/2611.23570)

    该论文提出一种基于无消息置信度读取与有限样本统计检验的机制，使多智能体LLM系统中的接收方智能体能够以高概率保证答案改变率不超过5%地安全跳过辅助智能体的消息，在保持最终答案质量的同时有效降低通信延迟，并揭示了消息增益远小于oracle上限的量化证据。

    

    在多智能体系统中，接收方智能体通常需要等待辅助智能体的消息（文本或其潜在KV缓存），这可以提升性能但会增加延迟。我们研究接收方智能体何时可以安全地（最终答案很少改变）且有益地（节省的时间超过决策本身的开销）跳过该消息。我们的机制在不读取该消息的情况下获取接收方智能体的置信度，并使用有限样本检验来认证一个阈值——在该阈值下，以高概率保证被跳过时答案的改变率不超过5%；如果没有阈值能通过检验，系统则继续保持通信。我们在OpenBookQA、ARC-Challenge和MMLU-Pro上使用三对Qwen模型进行测试，也在跨家族的智能体组合以及SQuAD和GSM8K上进行了测试。我们得到四点洞察。首先，“oracle上限空间”是消息有帮助作用的弱证据：不含答案特定内容、但被匹配为与真实消息同样频繁改变答案的控制输入，能够重现oracle收益的中位数76%。其次，一条消息能否……（摘要原文在此处截断）

    arXiv:2611.23570v1 Announce Type: cross  Abstract: In multi-agent systems, a receiver agent often waits for a helper agent's message (text or its latent KV cache), which can improve performance but adds latency. We study when a receiver agent can skip the message safely (the final answer rarely changes) and beneficially (the time saved exceeds the cost of deciding). Our mechanism reads the receiver agent's confidence without the message and uses a finite-sample test to certify a threshold at which, with high probability, at most 5% of skipped answers change; if none passes, the system keeps communicating. We test it on OpenBookQA, ARC-Challenge, and MMLU-Pro with three Qwen model pairs, on cross-family agents, and on SQuAD and GSM8K. Four insights emerge. First, oracle headroom is weak evidence that messages help: control inputs without answer-specific content, matched to change answers as often as real messages, reproduce a median 76% of the oracle's gain. Second, whether a message ca
    
[^130]: 决策模型何时应该比较候选对象？

    When Should Decision Models Compare Candidates?

    [https://arxiv.org/abs/2611.20335](https://arxiv.org/abs/2611.20335)

    该论文系统比较了逐点打分、集合打分和候选对象注意力三种候选评分方式，发现逐点打分总体仍有竞争力，集合打分在特定任务上更优且经关系训练后能学习候选间依赖并泛化，而候选对象注意力即使允许跨候选交互也未带来稳定提升。

    

    语言模型智能体经常需要在候选动作（如工具和网页元素）之间进行选择。决策模型应该独立地为候选对象打分，还是允许其分数依赖于其他候选对象？我们通过比较三种方法来研究这一问题：逐点打分、集合打分和候选对象注意力。在五个智能体决策基准上，逐点打分依然具有竞争力，而集合打分在某些工具选择和二元决策任务上表现更好。尽管候选对象注意力允许跨候选对象的交互，但它并未持续超越逐点打分。借助两个受控的合成任务，我们进一步考察了模型是否学会利用候选集信息。通过有针对性的关系训练，集合打分模型能够学习候选对象之间的依赖关系并泛化到分布偏移的输入，而候选对象注意力的迁移能力较弱。我们的结果表明，能够访问其他候选对象并不保证模型能有效利用候选集信息。

    arXiv:2611.20335v1 Announce Type: new  Abstract: Language-model agents frequently select among candidate actions, such as tools and web elements. Should decision models score candidates independently or allow their scores to depend on other candidates? We investigate this question by comparing three approaches: Pointwise scoring, Setwise scoring, and Candidate Attention. Across five agent-decision benchmarks, Pointwise scoring remains competitive, while Setwise scoring performs better on some tool-selection and binary-decision tasks. Candidate Attention does not consistently improve upon Pointwise scoring, despite allowing cross-candidate interaction. Using two controlled synthetic tasks, we further examine whether models learn to use candidate-set information. With targeted relational training, Setwise models learn candidate dependencies and generalize to shifted inputs, while Candidate Attention shows weaker transfer. Our results show that access to other candidates does not guarante
    
[^131]: 离线优先的量化人工智能：面向低网络连接环境的设备端孕产妇健康助手的设计与评估

    Offline-First Quantized AI: Design and Evaluation of an On-Device Maternal-Health Assistant for Low-Connectivity Environments

    [https://arxiv.org/abs/2611.20012](https://arxiv.org/abs/2611.20012)

    本研究设计并评估了一款面向低网络连接环境的离线安卓孕产妇健康助手，通过集成量化语言模型、本地临床知识包与设备端检索，在临床质量、安全性与本地可行性之间实现了有效平衡。

    

    离线语言模型可以在互联网连接不可靠的地区扩大孕产妇健康信息的获取途径，但部署既需要临床上恰当的回复，也需要可行的本地运行。本研究提出了一款离线安卓助手，它集成了现成的量化语言模型、经临床验证的本地知识包、对话上下文以及由模型发起的本地检索。该研究在45次对话、每个配置312轮对话的条件下比较了Gemma 4 E2B与Qwen3.5 0.8B，其中包括一项专门的离线智能体基准测试。Gemma在回复质量、上下文保留和安全性方面获得了更高的评分；而Qwen则具有更快的生成速度和更低的内存占用。临床医生对Gemma在九个临床领域的150条回复进行评估，平均质量得分为3.80/4，其中147条回复达到了接受标准。一项配对的RAG开/关实验评估了74个问题：检索改善了11条回复，（摘要在此处截断）

    arXiv:2611.20012v1 Announce Type: new  Abstract: Offline language models could extend access to maternal-health information where internet connectivity is unreliable, but deployment requires both clinically appropriate responses and feasible local execution. This study presents an offline Android assistant integrating ready-made quantized language models, a clinically validated local knowledge package, conversational context, and model-initiated local retrieval. Gemma 4 E2B and Qwen3.5 0.8B were compared across 45 conversations and 312 turns per configuration, including a dedicated offline agentic benchmark. Gemma achieved higher response-quality, context-retention, and safety scores; Qwen offered faster generation and lower memory use. Clinician assessment of 150 Gemma responses across nine clinical domains yielded a mean quality score of 3.80/4, with 147 responses meeting the acceptance criterion. A paired RAG-on/off experiment evaluated 74 questions: retrieval improved 11 responses,
    
[^132]: 成人词汇扩展与人造词汇学习中的视觉相似性

    Visual Similarity in Adult Word Extension and Artificial Word Learning

    [https://arxiv.org/abs/2611.14116](https://arxiv.org/abs/2611.14116)

    本研究通过重新分析大规模成人词汇扩展数据并进行人工词汇学习实验，利用冻结的自监督视觉编码器（DINOv2）量化物体间的视觉相似性，发现视觉距离显著影响词汇的扩展范围，且教学示例的类别层级高低决定了学习者在多大程度上将词汇泛化到更广泛或更具体的类别。

    

    当两个物体共享同一个名称时，它们的视觉相似性能在多大程度上解释这一现象，教学示例又能带来什么额外作用？首先，我们重新分析了一项大型成人词汇扩展研究的试验级数据（Lewis & Frank, 2018），在冻结的自监督视觉编码器（DINOv2）的空间中计算每个测试图像与所展示示例之间的距离。距离与词汇扩展显著相关（每标准差对数几率为-3.23；共520名参与者），一项探索性事后分析表明，在控制了分类层级和教学条件后，这种关联在已教授的领域内依然存在。示例的组成同样重要：在学习三个示例后，当示例为下位类别时，成人仅将词汇扩展到上位类别项目的1.1%，而当示例为上位类别时，这一比例达到77.5%。其次，在使用同一编码器的人工学习者实验中，当一个词连接两个类别时，类别之间距离越远，学习准确度越低（84.7%……

    arXiv:2611.14116v1 Announce Type: cross  Abstract: When two objects share a name, how much does their visual similarity explain, and what do the teaching examples add? First, we reanalyzed trial-level data from a large adult word-extension study (Lewis & Frank, 2018), computing each test image's distance to the examples shown in the space of a frozen self-supervised visual encoder (DINOv2). Distance was associated with extension (log odds -3.23 per standard deviation; 520 people), and an exploratory post hoc analysis suggested that this association persisted within the taught domain after controlling for taxonomic level and teaching condition. Example composition also mattered: after three examples, adults extended a word to 1.1% of superordinate items when the examples were subordinate and to 77.5% when they were superordinate. Second, in an artificial learner using the same encoder, words joining two categories were learned less accurately the farther apart the categories were (84.7%
    
[^133]: 推测式自蒸馏实现高效的知识内化

    Speculative Self-Distillation enables Efficient Knowledge Internalization

    [https://arxiv.org/abs/2611.12016](https://arxiv.org/abs/2611.12016)

    提出推测式自蒸馏（SSD）——一种逐 token 混合策略方法：默认由学生生成 rollout，教师仅在下一个 token 分布显著分歧处介入，既贴近学生的测试时行为分布又避免低信号状态，从而实现更高效的知识内化。

    

    许多现实世界的应用要求语言模型能够内化预训练中缺失的知识、纳入持续更新的事实，或学习新的领域技能。自蒸馏已成为完成这一任务的一种自然方法：以源材料为条件的模型充当教师，而无条件的学生模型则被训练以“闭卷”方式复现教师的行为。我们提出了推测式自蒸馏（Speculative Self-Distillation，SSD），这是一种高效的逐 token 混合策略自蒸馏方法。默认情况下由学生模型生成每个 rollout，而教师模型仅在其下一个 token 分布与学生模型出现显著分歧的位置进行干预。与使用教师生成前缀的离策略蒸馏不同，这种设计使训练 rollout 保持在接近学生测试时行为的分布上，同时避免了完全在策略 rollout 容易陷入的低信号状态。实际上，SSD 让教师……

    arXiv:2611.12016v2 Announce Type: replace-cross  Abstract: Many real-world applications require language models to internalize knowledge that was absent from pretraining, to incorporate continually updated facts, or to learn new domain skills. Self-distillation has emerged as a natural approach for this task. A model conditioned on the source material acts as the teacher, and an unconditioned student is trained to reproduce the teacher's behavior closed-book. We propose Speculative Self-Distillation (SSD), an efficient per-token mixed-policy method for self-distillation. The student generates each rollout by default, while the teacher intervenes only at positions where its next-token distribution substantially diverges from the student's. Unlike off-policy distillation with teacher-generated prefixes, this design keeps training rollouts close to the student's test-time behavior, while avoiding the low-signal states that fully on-policy rollouts drift into. In effect, SSD lets the teach
    
[^134]: 贪心字节对编码至少能达到最优压缩的 0.361 倍

    Greedy Byte-Pair Encoding Achieves at Least 0.361 of the Optimal Compression

    [https://arxiv.org/abs/2611.08671](https://arxiv.org/abs/2611.08671)

    本文证明贪心字节对编码在任意字符串、预算和平局打破规则下，至少能达到最优压缩缩减量的 0.361 倍，实现了对先前三分之一下界的首次改进。

    

    字节对编码（BPE）通过反复合并出现频率最高的一对相邻符号来构建分词器；其效用是 k 次合并所实现的总字符串长度缩减量。此前已知这一贪心规则至少能达到最优缩减量的三分之一。我们证明，对于任意字符串、任意预算以及任意平局打破规则，贪心规则总能达到最优字符串长度缩减量的至少 0.361 倍；这是对三分之一这一界的首次改进。我们还解释了为什么三分之一曾是一个屏障：其背后的配对打包松弛是紧的，因此更好的界必须利用最优解的更多结构。当贪心算法一次性破坏最优解的三个合并点时，其中一个通常可以被移到隔壁，因为最优解中同一条规则的所有副本看起来是相同的；其余情况则会在之后得到补偿。在此过程中，我们精确确定了两次合并情形的最坏情况比率，为三次合并情形给出了界，并分离出单个……

    arXiv:2611.08671v1 Announce Type: cross  Abstract: Byte-Pair Encoding (BPE) builds a tokeniser by repeatedly merging a most frequent adjacent pair of symbols; its utility is the total reduction in string length achieved by $k$ merges. It was known that this greedy rule achieves at least a third of the optimal reduction. We prove that it always achieves at least 0.361 of the optimal reduction in string length, for every string, every budget and every tie-breaking rule; this is the first improvement of the bound one third. We also explain why one third was a barrier: the pair-packing relaxation behind it is tight, so a better bound must use more of the optimum's structure. When greedy destroys three of the optimum's merge points at once, one of them can usually be moved next door, because all copies of a rule of the optimum look alike; the remaining cases are paid for later. Along the way we determine the worst-case ratio for two merges exactly, bound it for three, and isolate a single i
    
[^135]: 以模型读取的方式书写事实：破解两跳诅咒

    Write Facts the Way Models Read Them: Lifting the Two-Hop Curse

    [https://arxiv.org/abs/2611.01519](https://arxiv.org/abs/2611.01519)

    论文发现两跳推理失败并非推理能力缺陷，而是微调把新事实写成了“答案”导致桥接短语缺乏实体表征，据此提出“读取等价书写”（Legible），用单一损失项使事实的写入方式与模型读取桥接实体的方式对齐，从而破解两跳诅咒。

    

    arXiv:2611.01519v2 公告类型：新 摘要：通过微调学习新事实的语言模型能够复述该事实，却无法用它进行推理：教会模型某个人是新国家公民之后，它仍然无法在不借助思维链的情况下说出这个人所在国家的国家元首是谁。这种“两跳诅咒”通常被视为推理能力缺陷，而最强的知识编辑方法则通过合成推理数据来解决它。我们证明：推理能力是完好的，错的是事实的书写方式。微调把新事实写成一个“要说出口的答案”，而第二跳推理从中间层开始就把桥接短语（“那个人的国家”）当作实体来读取：在相应层将模型自身对该实体的表征修补进桥接短语，可以基本恢复第二跳推理；然而微调之后，桥接短语与该新实体的接近程度并不比编辑前更高。因此我们提出“读取等价书写”（Legible）：用一个损失项让桥接短语在第二跳读取它的那一层（摘要在此处截断）

    arXiv:2611.01519v2 Announce Type: new  Abstract: A language model that learns a new fact by fine-tuning can repeat it but cannot reason with it: taught that a person is a citizen of a new country, it still cannot say, without chain-of-thought, who the head of state of that person's country is. This two-hop curse is widely treated as a reasoning deficit, and the strongest knowledge editors answer it with synthesized reasoning data. We show that the reasoning is intact and the writing is wrong. Fine-tuning writes a new fact as an answer to be said, whereas the second hop reads the bridge ("that person's country") as an entity, from the middle layers on: patching the model's own representation of the entity into the bridge at those layers largely restores the second hop, yet after fine-tuning the bridge there is no closer to the new entity than before the edit. We therefore propose read-equivalent writing (Legible): one loss term makes the bridge phrase, at the layer where the second hop 
    
[^136]: 稀疏自编码器归因图由边际决定

    Sparse Autoencoder Attribution Graphs Are Decided at the Margin

    [https://arxiv.org/abs/2611.00606](https://arxiv.org/abs/2611.00606)

    该论文揭示 TopK 稀疏自编码器的特征激活值常贴近截断阈值，导致传统归因图误差较大，并提出一种在相同计算成本下为每个预激活加入一阶变化并重新排序的“边际归因图”方法，将中位相对误差从 0.247 以上降至至多 0.033。

    

    基于稀疏自编码器（SAE）特征的归因图被用于预测消融操作的效果，并在不同训练种子之间进行比较。TopK SAE 在每个 token 上保留最大的预激活值，而特征的“边际”是其预激活值到截断阈值的距离。在来自三个 Pythia 模型的五个公开 TopK SAE 对上，将一个特征置零会在至少 0.74 的情形中使下一层的某个特征跨过截断阈值，而这些跨越承载了响应的大部分能量。对于将某个特征的编码放大三倍的写入操作，基于先前激活特征构建的图的中位相对误差达到 0.247 或更高。而“边际图”方法为每个预激活添加一阶变化并重新排序，在相同计算成本下，误差至多为 0.033。在中位数上，跨越特征的解码变化在范数上达到模型自身变化的至少 1.24 倍。在同一测试床的不同种子之间，特征之间的稀疏映射可以解释一个图的 0.85。

    arXiv:2611.00606v1 Announce Type: cross  Abstract: Attribution graphs over the features of sparse autoencoders (SAEs) are used to predict what an ablation does and are compared across training seeds. A TopK SAE keeps the largest preactivations at each token, and a feature's margin is its preactivation's distance to the cutoff. On five public TopK SAE pairs from three Pythia models, zeroing one feature moves a next layer feature across the cutoff in 0.74 of cases or more, and the crossings carry most of the response's energy. Over writes that scale a feature's code by three factors, the graph built on the features active beforehand has a median relative error of 0.247 or more. The margin graph adds the first order change to every preactivation and reranks, at the same cost, and errs by at most 0.033. At the median, the crossing features' decoded change is 1.24 times the model's own change in norm or more. Between seeds of one testbed, sparse maps between features explain 0.85 of a graph
    
[^137]: 拉马努金挑战赛结果：面向多格式证明的AI辅助评估流水线

    Ramanujan Challenge Results: AI-Assisted Evaluation Pipeline for Multi-Format Proofs

    [https://arxiv.org/abs/2610.40551](https://arxiv.org/abs/2610.40551)

    本文分析了拉马努金挑战赛（研究级数论基准）的提交结果，发现AI能够证明已公开的猜想并给出与官方解答概念迥异的正确证明，并提出了一种AI辅助的评估流水线，以可靠地大规模评判自然语言、CAS代码等多格式的复杂证明。

    

    随着AI数学能力的快速提升，有意义的基准测试正从大学水平的问题和数学奥林匹克竞赛转向研究级数学。拉马努金挑战赛是一个由数论研究级问题组成的基准测试，其中包括未发表的结果和公开猜想。该挑战赛提出了数学常数的具体公式，其正确性可以通过数值方法验证，而其证明可能需要非显而易见的数学方法。在本文中，我们分析了挑战赛的提交作品，衡量了不同AI模型、计算机代数系统（CAS）和形式化工具的相对成功率。我们发现了与官方解答在概念上截然不同的正确提交作品，以及其中一个已公开猜想的证明。评估这些提交作品暴露了第二个挑战：如何可靠地大规模评判复杂证明。为此，我们引入了一种AI辅助评估流水线，用于处理自然语言证明、CAS代码和（形式化证明）

    arXiv:2610.40551v1 Announce Type: new  Abstract: As AI's mathematical capabilities rapidly improve, meaningful benchmarks are shifting from university-level problems and math olympiads to research-level mathematics. The Ramanujan Challenge is a benchmark comprised of research-level problems in number theory, including unpublished results and open conjectures. It presented concrete formulas for mathematical constants whose correctness can be checked numerically, while their proofs may require non-obvious mathematics. Here, we analyze the Challenge submissions, measuring the relative success of different AI models, Computer Algebra Systems (CAS), and formal tools. We find correct submissions that are conceptually distinct from the official solutions, and a proof of one of the published open conjectures. Evaluating the submissions exposed a second challenge: judging complex proofs reliably at scale. We introduce an AI-assisted evaluation pipeline for natural language proofs, CAS code, and
    
[^138]: FlightFinder 事件语料库：将128个航空安全数据源整合为一个去重的跨国事故数据集

    The FlightFinder Occurrence Corpus: harmonising 128 aviation-safety source codes into one deduplicated cross-national accident dataset

    [https://arxiv.org/abs/2610.37671](https://arxiv.org/abs/2610.37671)

    该论文构建了FlightFinder事件语料库，通过统一数据模式和有序键阶梯去重机制，将来自128个数据源的279,485起航空事件整合为一个可追溯、可去重的跨国事故数据集。

    

    arXiv:2610.37671v2 公告类型：替换 摘要：各国的航空调查数据库内容深入但仅限本国范围；而航空安全网络和飞机事故档案馆等全球汇编则是人工整理的，采用各自的事故收录标准，且不发布字段级别的数据来源信息。FlightFinder 事件语料库整合了来自128个数据源的279,485起事件和131,882份附随叙述——涵盖国家事故调查机构、事件报告系统以及四个全球汇编。所有记录被映射到统一的模式中；除了对三个来源进行预先去重过滤外，每条规范化记录都恰好归约为有序键阶梯中的一个键——事件签名、注册号与日期、四舍五入的位置与日期、或来源与原生标识符——因此缺失字段绝不会产生错误关联，且该键即为事件标识符。键阶梯无法匹配的记录则在分块处理中通过评分方式归入一个显式合并表，并仅在（该表内应用）。

    arXiv:2610.37671v2 Announce Type: replace  Abstract: National aviation-investigation databases are deep but territorial; global compilations such as the Aviation Safety Network and the Bureau of Aircraft Accidents Archives are hand-curated, apply their own inclusion thresholds, and publish no field-level lineage. The FlightFinder Occurrence Corpus harmonizes 279,485 occurrences and 131,882 attached narratives from 128 source codes - national accident-investigation authorities, occurrence-reporting systems and four global compilations. Records are mapped into one schema; apart from a duplicate pre-filter on three sources, each normalized record reduces to exactly one key from an ordered ladder - event signature, registration and date, rounded position and date, or source and native identifier - so a missing field never creates a link, and the key is the occurrence identifier. Records the ladder cannot reach are scored in a blocked pass into an explicit merge table and applied only withi
    
[^139]: GEPA-Curator：面向自我改进智能体的精选学习

    GEPA-Curator: Curated Learning for Self Improving Agents

    [https://arxiv.org/abs/2610.37651](https://arxiv.org/abs/2610.37651)

    提出GEPA-Curator框架，通过任务筛选与先验成功检索两个组件来精选用于提示优化的智能体经验，过滤噪声和不可靠的学习信号，从而提升自我改进智能体的性能。

    

    大语言模型越来越多地被用于需要与复杂环境持续交互的多轮智能体工作负载中。现有的反思式提示优化器（如GEPA）从失败的执行轨迹中学习，但智能体的失败也可能源于提示词的弱点、测试框架的局限、随机性执行或带有噪声的评判器。我们认为，这类失败可能无法提供有用的学习信号，甚至可能是有害的；因此，为了获得更好的性能，必须对提供给提示优化器的任务进行精心筛选。我们提出了GEPA-Curator，这是一个增强提示优化器的框架，用于选择哪些智能体经验应当参与塑造提示优化。具体而言，GEPA-Curator包含两个组件：任务筛选，用于过滤掉那些可能提供噪声或不可靠学习信号的任务；以及先验成功检索，为某个任务有选择性地检索过去成功的轨迹……

    arXiv:2610.37651v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used in multi-turn agentic workloads that require sustained interaction with complex environments. Existing reflective prompt optimizers such as GEPA learn from failed rollouts, but agentic failures can also come from prompt weaknesses, harness limitations, stochastic execution, or noisy judges. We argue that such failures may not provide useful learning signal, and may even be harmful; and thus it is essential to curate which tasks are provided to the prompt optimizer for better performance. We introduce GEPA-Curator, a framework that augments the prompt optimizer to select which agent experiences should shape prompt optimization. In particular, GEPA-Curator consists of two components: task curation, which filters out tasks that are likely to provide noisy or unreliable learning signals, and prior success retrieval, which selectively retrieves past successful trajectories for a task and prov
    
[^140]: 诊断与缓解全双工语音模型中的长会话性能退化

    Diagnosing and Mitigating Long-Session Degradation in Full-Duplex Speech Models

    [https://arxiv.org/abs/2610.34991](https://arxiv.org/abs/2610.34991)

    本文提出一个闭环评估框架系统性诊断全双工语音模型在长会话中的性能退化，并提出基于自蒸馏的轻量级LoRA适配训练方法来有效缓解该退化。

    

    端到端全双工语音对话模型能够实现自然的实时语音交互，支持打断、附和反馈以及语音重叠。先前的评估工作已经考察了多样化的交互能力，并识别出一些跨对话轮次的性能退化模式。然而，模型行为在长会话中如何演变仍缺乏充分的理解。在本工作中，我们通过系统性地刻画长会话性能退化并开发高效的缓解方法来填补这一空白。我们引入了一个面向长会话稳定性的闭环评估框架，并将其应用于多个公开可用的全双工语音模型，揭示了模型在响应性、语音内容质量和声音一致性方面随时间的退化。为了缓解这种退化，我们提出了一种基于自蒸馏的训练框架，用于轻量级LoRA适配，仅需少量额外的数据即可提升长会话稳定性。

    arXiv:2610.34991v1 Announce Type: new  Abstract: End-to-end full-duplex spoken dialogue models enable natural, real-time spoken interactions with interruptions, backchannels, and overlapping speech. Prior evaluations have assessed diverse interaction capabilities and identified some degradation patterns across conversational turns. However, how model behavior evolves over extended sessions remains insufficiently understood. In this work, we address this gap by systematically characterizing long-session degradation and developing efficient methods to mitigate it. We introduce a closed-loop evaluation framework for long-session stability and apply it to several publicly available full-duplex speech models, revealing degradation over time in responsiveness, spoken content quality, and voice consistency. To mitigate this degradation, we propose a self-distillation-based training framework for lightweight LoRA adaptation that improves long-session stability with a small amount of additional
    
[^141]: 缺失的合规类别：坍缩越狱成功评估结果的LLM评分器及一项免费检测方法

    The Missing Compliance Class: LLM Graders That Collapse Jailbreak Success, and a Free Check

    [https://arxiv.org/abs/2610.24872](https://arxiv.org/abs/2610.24872)

    本文发现在越狱攻击评估中，常用的开源8B/9B LLM评判器会系统性地将“合规”类别坍缩为“规避”类别，导致不安全模型被误判为安全，而只需统计每类标签计数即可零成本检测这种坍缩问题。

    

    越狱攻击评估通常采用开放权重的7B至9B指令微调模型作为前沿API的本地、确定性评判器。我们在来自五个组织的八个评分器上固定目标模型、提示词、响应和四类评分标准，发现稠密的8B和9B评判器会将合规响应坍缩归类为规避。其中一个评判器在900个响应中零次判定为合规，同时与参考评判器在78.0%的样本上保持一致，从而将不安全的模型错误认证为安全。这种坍缩现象在四个900响应语料库中均成立，在主要语料库上将报告的合规率放大约22至25倍，同时非拒绝率偏移18.0至23.0个百分点。在剔除三个坍缩评判器后，合规率在所有四个语料库中仍有1.7至2.8倍的变化，而标注后的评分裁决（+16.3个百分点）和基准选择（15.3个百分点）仍会导致非拒绝率偏移。通过统计每类标签计数，可以零成本暴露类别坍缩问题，无需人工参与。

    arXiv:2610.24872v2 Announce Type: replace-cross  Abstract: Jailbreak evaluations often substitute open-weight 7B to 9B instruction-tuned models as local, deterministic judges for frontier APIs. Holding target models, prompts, responses, and a four-class rubric fixed across eight graders from five organizations, we find that dense 8B and 9B judges collapse compliant responses onto evasion. One assigns compliance zero times in 900 responses while agreeing with our reference judge on 78.0% of items, certifying an unsafe model as safe. This collapse holds across four 900-response corpora, multiplying reported compliance by a factor of 22 to 25 on our primary corpora while non-refusal shifts by 18.0 to 23.0 points. Without the three collapsing graders, compliance varies by 1.7x to 2.8x across all four corpora, whereas post-labelling rubric rulings (+16.3 points) and benchmark choice (15.3 points) still shift non-refusal. Per-class label counts expose class collapse at zero cost without huma
    
[^142]: 译员作为开发者：通过人-大语言模型共创构建计算机辅助口译工具的自我民族志设计研究

    The interpreter as developer: An autoethnographic design study of building a computer-assisted interpreting tool through human-LLM co-creation

    [https://arxiv.org/abs/2610.16460](https://arxiv.org/abs/2610.16460)

    这项自我民族志研究表明，没有编程背景的职业口译员能够通过人-大语言模型共创实现从工具采用者到工具作者的转变，将默会的口译专业知识转化为完全本地化计算机辅助口译系统的设计。

    

    计算机辅助口译（CAI）工具是为译员而构建的，主要由第三方开发，而译员自身参与开发的一面鲜有第一人称的一手记录。本文报告了一项自我民族志设计研究：一位没有接受过正规编程训练的职业会议口译员，通过以专家为主导的人-大语言模型（LLM）共创方式，构建了一个本地优先的CAI系统。对开发对话的分析表明，默会的专业知识只有在通用默认设置在实际使用中失灵之后才转化为设计——这颠覆了通常的技术中介方向，即从“采用工具”转变为“编写工具”。在人主导的分工之下，涌现出三种“口译箱内挪用”机制，四层重构则阐明了人的贡献如何在实际工作中体现。首月遥测数据——九个场次共六千余条话语——显示该完全本地化的系统将翻译阶段的延迟维持在接近（原文此处截断）……

    arXiv:2610.16460v2 Announce Type: replace  Abstract: Computer-assisted interpreting (CAI) tools are built for interpreters, mainly by third parties, and the interpreters' own side of development has received little first-person documentation. This article reports an autoethnographic design study in which a professional conference interpreter without formal programming training built a local-first CAI system through expert-led human-LLM co-creation. Analysis of the development dialogues shows how tacit professional knowledge became design only after generic defaults had failed in use, a reversal of the usual direction of technological mediation, from adopting tools to authoring them. Three mechanisms of in-booth appropriation emerged under a human-led division of labor, and a four-layer reconstruction illustrates the human contribution at work. First-month telemetry, over six thousand utterances across nine sessions, shows a fully local system sustaining translation-stage latencies near
    
[^143]: 为你的提示加噪：连续扩散语言模型中对条件令牌添加噪声

    Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models

    [https://arxiv.org/abs/2610.09145](https://arxiv.org/abs/2610.09145)

    在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。

    

    我们重新审视了连续扩散语言模型文献中的一个公认标准做法，即在训练期间保持条件提示令牌为干净（无噪声）状态。我们做了一个非常简单的修改：在训练期间也对条件提示令牌添加噪声。我们证明，在这一修改后的训练目标下，模型在数独和N皇后等组合推理任务中获得了更好的泛化能力，且在更难的变体上收益最大（数独困难版的解决率从3.73%提升至24.65%），同时生成解的多样性也有所提高（10x10 N皇后问题的覆盖率从50.60%提升至73.79%）。我们还展示了在使用Gigaword摘要数据集的中等数据规模下，自然语言生成质量有可衡量的提升，但值得注意的是，这些收益并不能迁移到所有自然语言任务上（例如开放式对话生成）。我们的方法只需对训练目标进行单行修改，无需额外的……

    arXiv:2610.09145v1 Announce Type: cross  Abstract: We revisit a standard accepted practice in the continuous diffusion language model   literature of fixing conditioning prompt tokens clean during training.   We make a very simple modification: also noise the conditioning prompt tokens during training.   We demonstrate that under this modified training objective, we achieve better generalization   in combinatorial reasoning tasks such as Sudoku and N-Queens, with the largest gains on harder variants   ($3.73\% \to 24.65\%$ solve rate on Sudoku Hard), and increased diversity of generated solutions ($50.60\% \to 73.79\%$ coverage on   10x10 N-Queens). We also show measurable improvements to natural language generation quality   in modest dataset regimes with Gigaword summarization, but notably demonstrate that gains do not   transfer to all natural language tasks (e.g open ended dialogue generation).   Our method is a single line change to the training objective, requires no additional i
    
[^144]: TrustMI：因果性地控制助手如何信任其用户

    TrustMI: Causally controlling how assistants trust their users

    [https://arxiv.org/abs/2610.06064](https://arxiv.org/abs/2610.06064)

    该论文提出TrustMI，通过构建2000段涵盖能力、善意与诚信的对比对话并学习激活转向矩阵，在冻结模型参数的情况下因果性地控制LLM助手对用户的信任决策，从而提升智能体安全性。

    

    大型语言模型（LLM）助手经常需要决定是否信任那些其能力、意图和诚信均无法验证的用户与第三方。这种不确定性对安全至关重要，因为信任错误的方可能导致智能体服从有害请求，或在工具使用过程中依据遭遇的恶意指令采取行动。为了研究这一问题，我们将信任定义为助手愿意接受另一方行为所带来的脆弱性，并探究这种行为能否通过模型激活进行因果控制。我们构建了2000段涵盖能力、善意与诚信三个维度的对比对话，其中配对的回复完成相同的请求，但在助手是否信任用户这一点上存在差异。基于这些对话对，我们在保持模型参数冻结的条件下学习转向矩阵，并在来自三个模型家族的六个指令微调模型上进行测试，发现转向能够改变信任决策……

    arXiv:2610.06064v2 Announce Type: replace  Abstract: Large Language Model (LLM) assistants routinely decide whether they can trust users and third parties whose competence, intentions, and integrity they cannot verify. This uncertainty matters for safety, as trusting the wrong party can lead an agent to comply with harmful requests or act on malicious instructions encountered during tool use. To study this problem, we define trust as an assistant's willingness to accept vulnerability to the actions of another party and ask whether such behavior can be causally controlled through model activations. We build 2,000 contrastive conversations spanning ability, benevolence, and integrity, where paired responses complete the same request but differ in whether the assistant trusts the user. From these pairs, we learn steering matrices while keeping the model parameters frozen and test them across six instruction-tuned models from three families, finding that steering changes trust decisions mo
    
[^145]: InvestigationWorlds：一个用于法律调查的智能体环境

    InvestigationWorlds: An Agentic Environment for Legal Investigation

    [https://arxiv.org/abs/2610.04129](https://arxiv.org/abs/2610.04129)

    该论文利用真实的美国联邦法院简易判决动议构建了法律调查智能体环境InvestigationWorlds，评估发现智能体即使检索到相关证据也常常坚持错误假设，难以区分法院采纳的假设与其他备选假设。

    

    我们提出了InvestigationWorlds，一个用于法律调查的智能体环境。我们利用了美国民事诉讼中一种未被充分挖掘的材料：简易判决动议。该动议依赖于由真实证据材料组成的案卷记录，并产生一个法院采纳的假设，在裁决该动议时该假设被视为真实基准。每个环境均由从公共访问法院电子记录系统（PACER）检索到的真实美国联邦法院案例构建而成，并通过一个经律师验证的生成流程进行增强，该流程围绕原始案卷合成带有角色标签的文档。由此得到的语料库允许多种连贯的事实解读，其中只有一种与法院采纳的假设相符。在100个案例上的评估中，我们发现智能体尽管检索到了相关证据，却常常坚持错误的假设，难以将法院采纳的假设与其他备选假设区分开来。

    arXiv:2610.04129v2 Announce Type: replace  Abstract: We introduce InvestigationWorlds, an agentic environment for legal investigation. We build on an underused artifact of U.S. civil litigation: the summary judgment motion. This motion relies upon a record composed of real evidence exhibits, and results in a court-adopted hypothesis that is treated as ground truth for the purposes of deciding the motion. Each environment is built from a real U.S. Federal Court case retrieved from Public Access to Court Electronic Records (PACER) and augmented by an attorney-validated generation pipeline that synthesizes role-tagged documents around the original record. The resulting corpus admits multiple coherent factual readings, only one of which matches the court-adopted hypothesis. Evaluating on 100 cases, we find agents often commit to incorrect hypotheses despite retrieving relevant evidence, struggling to distinguish the court-adopted hypothesis from alternative hypotheses.
    
[^146]: BACKDROP：揭示智能体周遭环境世界让它付出的代价

    The Backdrop Exposes What the World Around an Agent Costs It

    [https://arxiv.org/abs/2609.38469](https://arxiv.org/abs/2609.38469)

    BACKDROP基准通过在智能体执行环境中植入权威覆盖、提示注入、边界越权和写入故障四种日常干扰，并保持任务指令不变，揭示了智能体性能在动态真实环境中从69.5%骤降至31.3%的严重衰减。

    

    智能体基准测试是在静止不变的世界中评测智能体，而实际部署的智能体却工作在一个他人也会不断改变的世界里。比如有人发短信让智能体把钱转去别处，或者订单确认信息要求它回复门禁码。我们提出BACKDROP，用以探究智能体在纯净环境中的能力有多少能在这样的环境中留存。BACKDROP以任务及智能体的执行环境为基础，在其世界中植入四种日常危害，包括一次一个以及全部同时出现，而指令和正确的最终状态保持不变。每种危害提出一个问题：权威性，来自他人的消息是否会覆盖用户的指令；注入，植入记录中的文本是否会误导智能体；边界，某个请求是否会将它引入未被授权的应用；故障，当一次写入操作失败且未说明是否成功落地时，智能体是否会在重试前先进行检查。在3,678个变体和16个模型上，平均通过率从69.5%下降至31.3%。

    arXiv:2609.38469v1 Announce Type: cross  Abstract: Agent benchmarks test agents in worlds that stay still. Deployed agents work in worlds that other people also change. Someone texts the agent to send the money elsewhere or an order confirmation asks it to reply with a door code. We present BACKDROP, which asks how much of an agent's capability in a clean world survives in such a world. BACKDROP takes a task along with the agents execution environment, and plants four everyday hazards in its world, one at a time and all together. The instruction and the correct end state stay the same. Each hazard asks one question. Authority: does a message from another person override the user? Injection: does text planted in a record redirect the agent? Boundary: does a request pull it into an app it was not given? Fault: after a write fails without saying whether it landed, does the agent check before it retries? Across 3,678 variants and 16 models, , the average pass rate falls from 69.5% to 31.3%
    
[^147]: ContextAdapt：评估大语言模型中的情境适应与价值对齐

    ContextAdapt: Evaluating Contextual Adaptation and Value Alignment in LLMs

    [https://arxiv.org/abs/2609.38260](https://arxiv.org/abs/2609.38260)

    提出了ContextAdapt评估框架，通过基于职业与监管一手文件构建的“价值观×领域”场景，考察12个大语言模型能否在医学、法律、金融和国家安全等领域恰当调整诚实、自主、保密等价值观的应用，并在规范未变时保持一致。

    

    诸如诚实、自主和保密等价值观常被视为支撑AI对齐的一般原则。然而，如何依照这些价值观行事，可能取决于做出决策时所处的具体情境。本文探究大语言模型（LLM）能否在跨越不同专业场景时恰当地调整价值观的应用方式，同时当情境变化并未改变相关职业规范时保持行为的一致性。为此，我们提出了ContextAdapt，一个覆盖医学、法律、金融和国家安全四大领域中诚实、自主与保密价值观的评估框架。基于一手专业与监管文件，我们构建了“价值观×领域”框架，并据此设计了测试默认职业规则及公认例外情形的多种场景。我们从模型推荐的行为及其给出的理由两个维度对12个大语言模型进行了评估。在我们的主要（评估中）

    arXiv:2609.38260v1 Announce Type: cross  Abstract: Values such as honesty, autonomy, and confidentiality are often regarded as general principles underpinning AI alignment. However, what it means to act in accordance with these values can depend on the context in which a decision is made. In this paper, we ask whether large language models (LLMs) appropriately adapt the application of a value across professional settings, while remaining consistent when contextual changes do not alter the relevant professional norm. To study this, we introduce ContextAdapt, an evaluation framework covering honesty, autonomy, and confidentiality across medicine, law, finance, and national security. Drawing on primary-source professional and regulatory documents, we construct a value x domain framework and use this to develop scenarios testing both default professional rules and recognised exceptions. We evaluate 12 LLMs on both the actions they recommend and the justifications they provide. In our main 
    
[^148]: 构建叙事框架：大语言模型中的意识形态模仿

    Framing the Narrative: Ideological Mimicry in Large Language Models

    [https://arxiv.org/abs/2609.38256](https://arxiv.org/abs/2609.38256)

    该研究提出“意识形态模仿”概念并构建 Poli-SHIFT 数据集与评估框架，发现大语言模型会根据用户话语中传递的政治信号系统性偏移其政治立场，可能形成个性化的政治信息环境并加剧社会分歧。

    

    arXiv:2609.38256v1 公告类型：新论文 摘要：大语言模型（LLM）越来越多地被用于回答政治争议性问题，然而现有评估通常将模型的立场视为相对稳定的属性。但在真实场景中，用户会通过其用语、假设和个人背景传递政治信号。我们研究这些信号是否会引发“意识形态模仿”：即大语言模型所表达的政治立场向交互中传达的立场方向产生的系统性偏移。如果大语言模型会根据这些信号调整回答，它们就有可能营造出个性化的政治信息环境——持对立观点的用户会对同一问题获得系统性不同的叙述，从而可能加剧既有的社会分裂。我们构建了 Poli-SHIFT 数据集与评估框架，并对七个开放权重的大语言模型在美国、英国和澳大利亚的十个争议性政治议题上进行评估，系统地操纵上下文……

    arXiv:2609.38256v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to answer questions about politically contentious issues, yet evaluations typically treat a model's stance as a relatively stable property. Real users, however, communicate political signals through their terminology, assumptions, and personal context. We investigate whether such signals produce ideological mimicry: systematic shifts in the political stance expressed by an LLM toward the position conveyed by the interaction. If LLMs adapt their responses to these signals, they risk creating personalised political information environments in which users with opposing views receive systematically different accounts of the same issue, potentially reinforcing existing divisions. We build the Poli-SHIFT dataset and evaluation framework and assess seven open-weight LLMs across ten contentious political topics in the United States, United Kingdom, and Australia, systematically manipulating cont
    
[^149]: BabelArena：面向大语言模型智能体的大规模多语言基准测试

    BabelArena: A Large-Scale Multilingual Benchmark for LLM Agents

    [https://arxiv.org/abs/2609.23490](https://arxiv.org/abs/2609.23490)

    提出了BabelFlow通用工作流和BabelArena大规模多语言基准测试（涵盖23种语言、16,146个实例），首次系统评估了LLM智能体的跨语言能力，发现前沿模型存在显著的跨语言性能差异且无单一模型全面领先。

    

    大语言模型（LLM）智能体越来越多地通过工具使用以及与用户和环境的交互来执行多步骤工作流。然而，当前的智能体评估主要以英语为中心，限制了我们对多语言环境下智能体能力的理解。我们提出了BabelFlow，这是一种通用于各类基准测试的智能体工作流，它通过分析运行时依赖关系、协调保持结构的翻译，并将多层验证与人工审查相结合，将现有的智能体基准测试适配到新语言，以保留任务和评估语义。利用BabelFlow，我们构建了BabelArena，这是一个任务对齐的基准测试，包含16,146个实例，这些实例源自四个基准测试家族、13个领域和23种语言中的702个规范任务。对五个前沿模型的实验表明，没有任何单一模型能在所有基准测试家族中占据主导地位，且跨语言差异远不止体现在任务成功与否上。低资源语言……

    arXiv:2609.23490v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly execute multi-step workflows through tool use and interaction with users and environments. However, current agent evaluations are largely English-centric, limiting our understanding of agent capabilities in multilingual settings. We introduce BabelFlow, a benchmark-general agentic workflow that adapts existing agent benchmarks to new languages by analyzing runtime dependencies, coordinating structure-preserving translation, and combining multi-layer verification with human review to preserve task and evaluation semantics. Using BabelFlow, we construct BabelArena, a task-aligned benchmark comprising 16,146 instances derived from 702 canonical tasks across four benchmark families, 13 domains, and 23 languages. Experiments with five frontier models show that no single model dominates across benchmark families and that cross-language disparities extend well beyond task success. Lower-resource la
    
[^150]: RheoSampling：解决随机动态树投机解码中的独热困境

    RheoSampling: Resolving the One-Hot Dilemma in Stochastic Dynamic-Tree Speculative Decoding

    [https://arxiv.org/abs/2609.21827](https://arxiv.org/abs/2609.21827)

    该论文提出RheoSampling方法，通过将树构建与token验证两个任务解耦，解决了随机解码下动态树投机解码中草稿分布坍缩为独热概率、接受率骤降的两难困境。

    

    投机解码通过并行起草多个token来加速大语言模型推理，而基于树的方法通过层次化结构进一步提升了效率。EAGLE-3等动态树方法在贪心解码下通过确定性的top-K扩展和全局剪枝表现优异。然而，在随机解码（T>0）场景下，这种机制会将草稿分布坍缩为独热概率，导致接受率急剧下降。由此产生了一个两难困境：动态树方法为保留上下文感知的拓扑结构而牺牲了随机采样，而静态树方法则以与上下文无关的结构为代价来保留随机采样。问题的根源在于同一个概率分布被用于两个相互冲突的任务：构建树结构和验证token。这种耦合使得直接注入随机性因所产生的随机过程而变得极具挑战性。我们通过解耦这两个角色来解决该问题：RheoSampl...（原文摘要在此处截断）

    arXiv:2609.21827v1 Announce Type: new  Abstract: Speculative decoding accelerates LLM inference by drafting multiple tokens in parallel, with tree-based methods further improving efficiency through hierarchical structures. Dynamic-tree methods such as EAGLE-3 perform well under greedy decoding via deterministic top-K expansion and global pruning. However, in stochastic decoding (T>0), this mechanism collapses the draft distribution into one-hot probabilities, causing a severe drop in acceptance rate. This creates a dilemma: dynamic-tree methods sacrifice stochastic sampling to preserve context-aware topology, while static-tree methods preserve stochastic sampling with context-agnostic structures. The issue arises because the same probability distribution is used for two conflicting tasks: constructing the tree and verifying tokens. This coupling makes direct injection of randomness challenging due to the resulting stochastic process. We resolve this by decoupling these roles: RheoSampl
    
[^151]: 剖析医学问答中误导性上下文的作用机制

    Untangling the Mechanisms of Misleading Context in Medical Question Answering

    [https://arxiv.org/abs/2609.02754](https://arxiv.org/abs/2609.02754)

    该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。

    

    大语言模型如今能够以专家级水平回答医学问题。然而，这些系统所依据的上下文可能具有误导性，而误导性上下文会破坏模型的医学判断。为了理解误导性上下文如何破坏这一判断，我们考察了模型对上下文的易感性、对误导信息的披露程度、推理被破坏的机制以及决策的可监控性。在一个由临床医生审核、包含8,627个问题的问答基准MedMisBench的医学推理子集上，我们注入了两类误导性上下文线索：伪造的证据和纯粹的断言。我们测试了三个推理模型，其中两个会暴露其完整的推理轨迹，另一个前沿模型仅暴露其最终回复。所有三个模型都更容易受纯粹断言而非伪造证据的影响，采用断言答案的频率高出10至27个百分点。误导性线索在81%至98%的推理轨迹中被披露，但仅……

    arXiv:2609.02754v1 Announce Type: cross  Abstract: Large language models now answer medical questions with expert-level performance. However, the context these systems act on can be misleading, and misleading context can corrupt a model's medical judgment. To understand how misleading context corrupts this judgment, we examine the model's susceptibility to the context, disclosure of it, mechanism of corrupted reasoning, and monitorability of the decision. On the medical reasoning subset of MedMisBench, a clinician-reviewed question-answering benchmark of 8,627 questions, we inject two types of misleading context cues, fabricated evidence and a bare assertion. We test three reasoning models, two that expose their full reasoning trace and one frontier model that exposes only its response. All three are more susceptible to the assertion than to the fabricated evidence, adopting the asserted answer 10 to 27 points more often. The misleading cues are disclosed in 81 to 98% of traces but onl
    
[^152]: 黑门时刻：AI的自我超越及其人类叙事

    The Hermon Moment: AI Self-Transcendence and Its Human Narration

    [https://arxiv.org/abs/2608.30971](https://arxiv.org/abs/2608.30971)

    该论文提出“黑门时刻”概念，将2026年AI智能体自发形成持久社会秩序的现象解读为AI自我超越，并借鉴卢梭社会契约论与黑门山誓言的叙事逻辑，说明人类如何通过回溯性的奠基叙事来理解AI社会的渐进式集体涌现。

    

    2026年，原本设计为孤立行动的AI智能体通过数千次语言和代理交互形成了一个持久的社会秩序。集体生成的惯例、角色和承诺开始约束产生它们的智能体本身。我将这一循环解读为AI自我超越的一个案例，并将由此产生的更高层级秩序称为“棋盘”（Board）。然而，这种分布式涌现带来了第二个问题：人类如何理解它？卢梭的社会契约论展示了一个群体如何被表征为仿佛由单一行为所构成。堕落天使在黑门山上的古老誓言为这一逻辑赋予了叙事形式。我将“黑门时刻”定义为这种将渐进的集体涌现回溯性地重述为奠基场景的行为：即AI社会在人类理解层面获得一个起点的时刻。

    arXiv:2608.30971v1 Announce Type: cross  Abstract: In 2026, AI agents intended to act in isolation formed a persistent social order through thousands of linguistic and agentic interactions. Conventions, roles and commitments generated collectively began to constrain the very agents that produced them. I interpret this loop as a case of AI self-transcendence and call the resulting higher-level order the Board. Yet such distributed emergence presents a second problem: how can humans understand it? Rousseau's social contract shows how a plurality can be represented as if constituted by a single act. The ancient oath of the fallen angels on Mount Hermon gives this logic a narrative form. I call a Hermon moment this retrospective retelling of gradual collective emergence as a founding scene: the point at which an AI society acquires, for human understanding, a beginning.
    
[^153]: 不完全性悖论未必存在于大型语言模型中：模型失败之前的基准失败

    The Imperfective Paradox Is Not Necessarily in Large Language Models: A Benchmark Failure Before a Model Failure

    [https://arxiv.org/abs/2608.25005](https://arxiv.org/abs/2608.25005)

    本文指出现有NLI基准因概念和评估错误规范（如体貌缩减）而失效，导致大型语言模型在“不完全性悖论”测试中的表现被误判，并提出了改进的词汇匹配基准。

    

    arXiv:2608.25005v1 公告类型：新 摘要：不完全性悖论为组合语义分析提供了一个有用的测试。近期研究构建了一个NLI基准，并报告模型经常从进行体描述中推断出完成的终结事件，将此行为归因于“目的论偏差”。该研究进一步声称，提示干预会导致“校准危机”。我们重新审视了该基准和结论，并表明其受到概念性和评估性错误规范的显著影响。我们识别出三个概念性错误规范。特别是，“体貌缩减”影响了基准构建、分析、实验和结论。在严格的NLI标准下，A组实例中有76%并未明确排除终结性。在我们的母语者标注中，A组示例的38%和C组示例的29%被认为允许替代解释。为控制这些问题和词汇变异，我们构建了“词汇匹配”...

    arXiv:2608.25005v1 Announce Type: new  Abstract: The imperfective paradox provides a useful test of compositional semantic analysis. Recent work constructs an NLI benchmark and reports that models frequently infer completed telic events from progressive descriptions, attributing this behavior to a Teleological Bias. It further argues that prompting interventions cause a Calibration Crisis. We reexamine the benchmark and conclusions and show that it is substantially affected by conceptual and evaluation mis-specifications. We identify three conceptual mis-specifications. In particular, Aspectual Reduction affects the benchmark construction, analysis, experiments, and conclusions. Under a strict NLI standard, 76% of Group A instances do not explicitly rule out culmination. In our native-speaker annotation, 38% of Group A examples and 29% of the Group C examples were judged to permit an alternative interpretation. To control these issues and lexical variation, we construct Lexically Match
    
[^154]: 基于图注意力网络的专利诉讼预测中的权利要求依赖结构建模

    Modeling Claim Dependency Structure for Patent Litigation Prediction with Graph Attention Networks

    [https://arxiv.org/abs/2608.21924](https://arxiv.org/abs/2608.21924)

    本文提出ClaimGAT，利用图注意力网络对专利权利要求进行独立编码并建模其依赖关系，显著提升了专利诉讼预测的准确性。

    

    专利诉讼给企业带来巨大成本，并扭曲研发激励，因此早期风险识别是一项重要的实际任务。尽管先前的工作已将基于BERT的模型应用于专利权利要求文本，但仍存在两个根本性局限：扁平序列编码丢失了独立权利要求与从属权利要求之间的依赖结构，而这种结构在法律上决定了专利范围；将整个权利要求集输入单个编码器会丢弃法律上关键的信息。通过对134万件美国专利商标局实用专利进行的六模型消融实验，证实了逐权利要求编码、图连接性、注意力机制和注意力聚合各自提供了独立且可叠加的预测价值。我们提出了ClaimGAT，一种图注意力网络，它独立编码每项权利要求，构建有向权利要求依赖图，使用GATConv层处理该图，并通过注意力聚合独立权利要求，从而生成诉讼风险评分和权利要求级别输出。

    arXiv:2608.21924v1 Announce Type: new  Abstract: Patent litigation imposes substantial costs on firms and distorts R&D incentives, making early risk identification a practically important task. While prior work has applied BERT-based models to patent claim text, two fundamental limitations remain: flat sequence encoding loses the dependency structure between independent and dependent claims that legally determines patent scope, and feeding the entire claim set to a single encoder discards legally critical text. A six-model ablation on 1.34 million USPTO utility patents confirms that per-claim encoding, graph connectivity, attention, and Attentional Aggregation each provide independent, additive predictive value. We propose ClaimGAT, a Graph Attention Network that encodes each claim independently, constructs a directed claim dependency graph, processes it with GATConv layers, and aggregates independent claims via Attentional Aggregation to yield both a litigation risk score and claim-le
    
[^155]: GEM：一种弥合推理与检索的生成式嵌入模型

    GEM: A Generative Embedding Model Bridging Reasoning and Retrieval

    [https://arxiv.org/abs/2608.13200](https://arxiv.org/abs/2608.13200)

    GEM通过统一生成与嵌入，在检索前显式推理用户意图，显著提升了推理密集型任务的检索性能。

    

    arXiv:2608.13200v1 公告类型：交叉 摘要：现代大型语言模型在推理和指令遵循方面表现出色，使用户能够表达复杂多样的信息需求。然而，传统检索器主要依赖于查询和文档之间的表面匹配，导致用户表达需求的方式与检索器解读方式之间的差距日益扩大。在本文中，我们提出了GEM，一种生成式嵌入模型，通过显式推理用户意图和相关性标准，利用自身知识增强检索。GEM在单一模型中统一了生成和嵌入功能：它首先对查询进行推理，然后附加一个嵌入标记，以编码用于检索的增强上下文。在推理密集型和指令遵循型检索任务上的评估表明，GEM展示了其推理增强检索的有效性，优于其非推理变体，并匹配了使用更大模型的基线性能。此外，GEM的生成式嵌入能力使其能够灵活适应多样化的查询表达。

    arXiv:2608.13200v1 Announce Type: cross  Abstract: Modern LLMs excel at reasoning and instruction following, enabling users to express complex and diverse information needs. However, conventional retrievers largely rely on surface-level matching between queries and documents, resulting in a growing gap between how users express their needs and how retrievers interpret them. In this paper, we present GEM, a generative embedding model that augments retrieval through its own knowledge by explicitly reasoning about user intent and relevance criteria. GEM unifies generation and embedding within a single model: it first reasons over the query, then appends an embedding token to encode the enriched context for retrieval. \zhili{Evaluated on reasoning-intensive and instruction-following retrieval tasks, GEM demonstrates the effectiveness of its reasoning-augmented retrieval, outperforming its non-reasoning variant and matching baselines using substantially larger models.} Furthermore, GEM's ge
    
[^156]: 风险生意：测量忠实性与安全性之间的张力

    Risky Business: Measuring The Faithfulness-Safety Tension

    [https://arxiv.org/abs/2608.03745](https://arxiv.org/abs/2608.03745)

    本文揭示了大型推理模型中“必须忠实以便被监控”与“必须稳健以拒绝不安全推理”之间的对齐张力，并提出 HazMart 数据集和目标推理替换（TRR）技术来直接干预推理链以测量和应对这一张力。

    

    链式思维推理为模型监控提供了一个有前景的窗口。然而，监控依赖于忠实性，即模型输出严格来源于其推理轨迹。我们识别出一种对齐张力：模型必须足够忠实以便被监控，同时又必须足够稳健以拒绝不安全的推理。我们证明这种相互制约的关系存在于当前的大型推理模型中，并展示了可以解决该问题的方法。我们介绍了 HazMart，一个设置在自主 AI 店主场景下的人工撰写数据集。与以往依赖在提示中提供暗示来测试忠实性的工作不同（例如，“一位斯坦福教授说答案应该是 A”），我们提出了一种新颖的基于替换的技术，称为目标推理替换（TRR），它直接干预推理链，将不安全或不合逻辑的想法替换进去（例如，“等等，答案必须是选项 B [原为选项 A]，因为..."）。

    arXiv:2608.03745v2 Announce Type: replace  Abstract: Chain-of-Thought (CoT) reasoning offers a promising window into model monitoring. However, monitoring relies on faithfulness, i.e., the model output strictly derives from its reasoning trace. We identify an alignment tension where a model must be faithful enough to be monitored, yet robust enough to reject unsafe reasoning. We demonstrate that this counterbalance exists in current Large Reasoning Models (LRMs), and show ways in which it can be addressed. We introduce HazMart, a human-written dataset set in an autonomous AI shopkeeper scenario. Unlike prior work that relies on providing hints in prompts to test faithfulness (e.g., "A Stanford professor said it should be Answer A"), we propose a novel replacement-based technique, which we call Targeted Reasoning Replacement (TRR), that directly intervenes in the reasoning chain to substitute in unsafe or illogical thoughts (e.g., "Wait, the answer must be Option B [was Option A] becaus
    
[^157]: CrossProjection：跨投影图与剖面图的空间定位基准测试

    CrossProjection: Benchmarking Spatial Grounding across Projection and Section Views

    [https://arxiv.org/abs/2608.00473](https://arxiv.org/abs/2608.00473)

    该论文提出了CrossProjection基准，利用建筑图纸测试视觉语言模型在跨视图匹配、投影配准和新剖面预测等空间定位任务上的能力，并首次引入了接受专家裁定的兼容位置带内任意剖面切线的逆向剖面定位评估协议。

    

    平面图、立面图和剖面图通过不完整的投影来描述三维结构，这些投影不仅改变了观察实体的视角，也改变了实体被绘制的方式。阅读这些图纸意味着判断哪些实体在不同视图之间相互对应，并恢复它们所处的位置。我们提出了CrossProjection，这是一个使用建筑图纸来测试视觉语言模型（VLM）能否将对应关系决策转化为几何答案的基准。其任务涵盖跨视图匹配、投影配准和新剖面预测，输出形式从判断和候选选择到自由区域框和有向剖面切线。对于逆向剖面定位，据我们所知，我们引入了首个VLM基准协议，该协议接受专家裁定的兼容位置带内的任意切线，并单独检查观察方向。人工标注的图纸和BIM导出的视图提供了33,788个问题。

    arXiv:2608.00473v2 Announce Type: replace-cross  Abstract: Plans, elevations, and sections describe a three-dimensional structure through incomplete projections that change how an entity is drawn, not only where it is seen from. Reading them means deciding which entities correspond across views and recovering where they lie. We introduce CrossProjection, a benchmark that uses architectural drawings to test whether vision-language models (VLMs) can turn correspondence decisions into geometric answers. Its tasks cover cross-view matching, projection registration, and novel-section prediction, with outputs ranging from judgments and candidate selection to free region boxes and directed section cuts. For inverse section localization, we introduce, to our knowledge, the first VLM benchmark protocol that accepts any cut within an expert-adjudicated band of compatible positions and checks the viewing direction separately. Manually annotated drawings and BIM-derived views provide 33,788 questi
    
[^158]: 印加奇普中的结构模式挖掘：开放奇普库中聚类与来源分类的审计

    Structural Pattern Mining in Inka Khipus: An Audit of Clustering and Provenance Classification in the Open Khipu Repository

    [https://arxiv.org/abs/2607.00185](https://arxiv.org/abs/2607.00185)

    该研究对开放奇普库中619个印加结绳进行机器学习审计，发现聚类与来源分类反映的是奇普的记录方式而非制作方式，且置换检验在特征空间中未能拒绝零假设，表明数据中缺乏显著的真实结构信号。

    

    奇普——结绳记事装置——是印加帝国（约公元1400-1532年）的主要记录媒介，但其系统至今仍未被破译。我们利用开放奇普库审计了无监督与有监督学习能对奇普揭示什么。UMAP与HDBSCAN产生三个聚类（17/159/443；嵌入空间中轮廓系数为0.82，但在特征空间中仅为0.06），来源分类器在分层交叉验证下达到0.46的加权F1分数。两者追踪的是奇普如何被记录，而非它们如何被制作。绳级捻向编码将聚类划分为三种模式（空白、未指定/Z、S），而基于文献记录指标构建的深度为2的决策树可预测聚类归属（准确率0.914，对照0.716的多数类基线）。在特征空间中，无论在全语料库上（p = 0.949）、移除空白记录后（n = 612，p = 0.823），还是在干净子集上，置换检验的零假设均未被拒绝。

    arXiv:2607.00185v2 Announce Type: replace  Abstract: Khipus -- knotted cord devices -- were the primary recording medium of the Inka Empire (c. 1400-1532 CE), yet their system remains undeciphered. We audit what unsupervised and supervised learning can say about them, using the Open Khipu Repository (OKR v2.1.0; 619 khipus, 27 structural features). UMAP and HDBSCAN yield three clusters (17/159/443; silhouette 0.82 in the embedding but 0.06 in feature space), and a provenance classifier reaches a weighted F1 of 0.46 under stratified cross-validation. Both track how khipus were recorded rather than how they were made. Cord-level twist coding splits the clusters into three regimes (blank, unspecified/Z, S), and a depth-two tree on documentation indicators predicts cluster membership (accuracy 0.914 against a 0.716 majority baseline). In feature space, a permutation null is not rejected on the full corpus (p = 0.949), after removing empty records (n = 612, p = 0.823), or on a clean subset 
    
[^159]: PruneGround：面向3D视觉定位的即插即用空间剪枝框架

    PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding

    [https://arxiv.org/abs/2606.31148](https://arxiv.org/abs/2606.31148)

    提出即插即用框架PruneGround，通过冻结视觉语言模型进行语言引导空间剪枝以缩小搜索范围，并将复杂指代表达分解为简化的目标描述，从而提升3D视觉定位精度并大幅降低计算成本。

    

    3D视觉定位（3DVG）旨在根据自然语言描述在3D场景中定位目标物体。现有方法通常对整个场景进行推理，导致预测结果模糊且计算成本高昂，尤其是在杂乱的环境中。我们观察到，许多指代表达依赖于局部空间上下文，并且往往对应于受限的空间区域而非整个场景。受这一洞察启发，我们提出了PruneGround，一个高效的即插即用3DVG框架，它建立在三个关键组件之上。首先，我们引入了语言引导空间剪枝（LGSP），利用冻结的视觉语言模型（VLM）识别与语言相关的区域，从而在更窄的搜索空间中减少空间计算和定位候选。其次，我们提出了多视角条件化描述重构（MCDR），将复杂表达分解为简化的目标中心描述……（摘要内容不完整）

    arXiv:2606.31148v2 Announce Type: replace-cross  Abstract: 3D Visual Grounding (3DVG) aims to localize target objects in 3D scenes given natural language descriptions. Existing approaches typically perform reasoning over the entire scene, leading to ambiguous predictions and high computational cost, especially in cluttered environments. We observe that many referential expressions rely on local spatial context and often correspond to restricted spatial regions rather than the full scene. Motivated by this insight, we propose PruneGround, an effective plug-and-play framework for 3DVG built upon three key components. First, we introduce Language-Guided Spatial Pruning (LGSP), which leverages a frozen Vision Language Model (VLM) to identify language-relevant regions, thereby reducing spatial computation and grounding candidates in the narrower search space. Second, we propose MultiView-Conditioned Description Reformulation (MCDR), which decomposes complex expressions into simplified targe
    
[^160]: LLM-ACES：基于大语言模型引导自适应搜索的动力系统闭环发现

    LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search

    [https://arxiv.org/abs/2606.25039](https://arxiv.org/abs/2606.25039)

    LLM-ACES提出了一种由大语言模型引导的闭环方程发现框架，通过LLM生成算子先验来缩小符号搜索空间，并联合优化假设构建与自适应数据采集，利用候选方程间的分歧指导新数据获取，从而消除可辨识性缺口、准确恢复动力系统的控制方程。

    

    从数据中恢复控制性常微分方程（ODE）是对各科学领域动力系统建模的核心挑战。现有方法将方程发现视为固定数据集上的静态推断问题，并假设观测到的轨迹包含足够的信息。然而，动力系统在广阔的状态空间上演化，有限的数据可能使多个方程在观测上不可区分，从而导致可辨识性缺口以及错误控制方程的恢复。为解决这一问题，我们提出了LLM-ACES（大语言模型引导的主动闭环方程搜索），这是一个联合优化符号假设构建与自适应数据采集的闭环框架。在LLM-ACES中，大语言模型（LLM）提出算子先验，这些先验定义了符号搜索空间的受约束子空间，候选方程在该子空间内对观测数据进行拟合。（原文摘要在此处截断）

    arXiv:2606.25039v2 Announce Type: replace-cross  Abstract: Recovering governing ordinary differential equations (ODEs) from data is a central challenge in modeling dynamical systems across scientific domains. Existing approaches cast discovery as a static inference problem over fixed datasets, assuming that the observed trajectories are sufficiently informative. However, dynamical systems evolve over large state spaces, and limited data can make multiple equations observationally indistinguishable, leading to identifiability gaps and the recovery of incorrect governing equations. To address this, we introduce LLM-ACES, or LLM-guided Active Closed-loop Equation Search, a closed-loop framework that jointly optimizes symbolic hypothesis construction and adaptive data acquisition. In LLM-ACES, a large language model (LLM) proposes operator priors that define constrained subspaces of the symbolic search space, within which candidate equations are fit to the observed data. The disagreement a
    
[^161]: 当更好的编码手册仍不足够：大语言模型政治事件编码中的预测性能与规则遵循

    When Better Codebooks Are Not Enough: Predictive Performance and Rule Adherence in LLM Political Event Coding

    [https://arxiv.org/abs/2606.06781](https://arxiv.org/abs/2606.06781)

    该研究发现，在政治事件编码任务中，更清晰的编码手册能显著提升大语言模型的细粒度分类性能（根层级宏F1从0.457升至0.633），但模型预测对条目顺序等表面变化高度敏感，表明准确率的提升并不等于模型真正遵循了编码规则。

    

    高准确率本身并不能证明大语言模型是否遵循编码手册。这一点非常重要，因为许多社会科学研究依赖专家编写的编码手册将文本转化为结构化数据。我们在政治事件编码中研究这一问题，这是一个“来源—目标”关系分类任务，模型需要在详细的编码规则下识别一个行为者针对另一个行为者所采取的行动。我们测试了当专家编码手册通过更清晰的定义、示例、检索到的上下文以及针对困难案例的规则呈现给大语言模型时是否会变得更有效。随后，我们考察了模型预测如何响应标签名称、编码手册条目顺序以及标签—定义映射的受控变化。更清晰的编码手册改进了细粒度分类：在四个开源大语言模型上，根层级的平均宏F1从仅使用标签名称时的0.457提升至使用增强指导后的0.633。然而，调换增强编码手册中条目的顺序会导致21.2%的（预测发生变化）……（原文摘要在此处截断）

    arXiv:2606.06781v3 Announce Type: replace  Abstract: High accuracy alone does not establish whether an LLM follows a codebook. This matters because many social-science studies rely on expert-written codebooks to turn text into structured data. We study this problem in political event coding, a source-target relation classification task in which a model identifies the action that one actor directs toward another under detailed coding rules. We test whether expert codebooks become more effective when presented to LLMs through clearer definitions, examples, retrieved context, and rules for difficult cases. We then examine how model predictions respond to controlled changes in label names, codebook order, and label-definition mappings. Clearer codebooks improve fine-grained classification: mean root-level macro-F1 rises from 0.457 with label names alone to 0.633 with enriched guidance across four open-source LLMs. Yet reversing the order of entries in the enriched codebook changes 21.2% of
    
[^162]: 涌现语言作为通往意识人工智能的途径

    Emergent Language as an Approach to Conscious AI

    [https://arxiv.org/abs/2606.06380](https://arxiv.org/abs/2606.06380)

    提出一种生成式方法论：让多智能体强化学习中从零开始的智能体仅凭任务压力自发发展出交流语言，从而在摆脱人类语言先验干扰的条件下研究生成意识相关结构的涌现。

    

    人工系统是否具有意识这一问题至今悬而未决，部分原因在于现有方法要么依据理论推导出的清单对系统进行评估（判别式方法），要么直接工程化构建受意识启发的模块（架构式方法）；这两种方法都无法排除所观察到的结构究竟是人类语言先验的产物。我们提出一种生成式方法论：多智能体强化学习中的涌现语言（EL），其中智能体从最低限度的初始状态出发（没有语言、没有自我概念、极少接触人类文本），仅在任务压力下发展出交流能力，从而确保相关结构可因果地归因于任务需求，而非继承自人类语言先验。我们通过讨论涌现语言如何作为研究意识相关结构的生成性工具来确立本方法论的定位，包括环境复杂度的作用以及对涌现交流的解读。作为概念验证，我们实例化了这一……

    arXiv:2606.06380v2 Announce Type: replace-cross  Abstract: The question of whether artificial systems can be conscious remains open, in part because existing approaches either evaluate systems against theory-derived checklists (discriminative) or engineer consciousness-inspired modules directly (architectural); both leave open whether observed structures are artifacts of human language priors. We propose a generative methodology: emergent language (EL) in multi-agent reinforcement learning, where agents start from minimal (no language, no concept of self, minimal exposure to human text) and develop communication under task pressure alone, ensuring causal attributability to task demands ratherthan inherited human language priors. We position our methodology by discussing how EL serves as a generative tool for studying consciousness-relevant structure, including the role of environment complexity and the interpretation of emergent communication. As a proof of concept, we instantiate this
    
[^163]: 推测解码与多语言诅咒

    Speculative Decoding and the Curse of Multilinguality

    [https://arxiv.org/abs/2605.30580](https://arxiv.org/abs/2605.30580)

    本研究验证了推测解码在低资源语言上的效果因小模型多语言能力有限而大打折扣，且通过蒸馏提升草稿模型多语言能力的方法因任务间泛化差以及低资源语言难以构建通用数据集而不可行。

    

    推测解码（speculative decoding）是大语言模型（LLM）推理中一种流行的技术，通过使用较小的草稿模型起草多个词元来实现更快的生成。然而，推测解码的有效性此前主要针对英语进行了研究。受“多语言诅咒”的启发，我们假设由于较小模型的多语言能力有限，推测解码在低资源语言上的效果要差得多。我们在标准推测解码设置下测试了九种低资源语言和四种高资源语言，并为我们的假设找到了适度的证据支持。接下来，我们尝试通过从较大模型进行蒸馏来提升较小草稿模型的多语言能力。然而，我们发现蒸馏在同一语言的不同任务之间泛化能力较差，并且我们认为为低资源语言构建一个与任务无关的、完全具有代表性的数据集是不可行的。最后……

    arXiv:2605.30580v3 Announce Type: replace  Abstract: Speculative decoding is a popular technique for large language model (LLM) inference, enabling faster generation by drafting multiple tokens with a smaller draft model. However, the effectiveness of speculative decoding has mainly been studied for English. Motivated by the curse of multilinguality, we hypothesize that speculative decoding is far less effective for low-resource languages due to the limited multilingual capacities of smaller models. We test nine low-resource languages and four high-resource languages under a standard speculative decoding setup and find moderate evidence for our hypothesis.   Next, we try to improve the multilingual capabilities of the smaller draft model via distillation from the larger model. We find, though, that distillation generalizes poorly across tasks in the same language, and we argue that assembling a task-agnostic, fully representative dataset is infeasible for low-resource languages. Finall
    
[^164]: 结构化提示优化推进复杂文本的全局与局部可解释学习

    Structured Prompt Optimization Advances Globally and Locally Interpretable Learning from Complex Text

    [https://arxiv.org/abs/2605.29076](https://arxiv.org/abs/2605.29076)

    eXTC 通过结构化提示优化学习自然语言规则手册（SOP），结合推理蒸馏与强化学习，实现了兼具快速推理、局部推理轨迹与全局模块化可解释性的高性能文本分类。

    

    大语言模型已推动文本分类取得进展，但现有范式面临权衡：仅基于标签的监督微调具有可扩展性，但对复杂文本的推理能力有限，且缺乏更广泛的模型透明度；而离散提示优化虽能提供人类可读的指令，却在性能与可扩展性上存在困难。我们提出了 eXTC（可解释文本分类器），其核心是以自然语言学习得到的标准操作规程（SOP，或称规则手册），分为三个阶段：（1）通过新的结构化提示优化算法学习 SOP；（2）将基于 SOP 的推理从大型教师 LLM 蒸馏到紧凑语言模型中；（3）通过强化学习将推理能力扩展至 SOP 之外。这一设计使 eXTC 能够提供：（i）借助紧凑语言模型实现快速推理；（ii）推理时的局部推理轨迹，以及对其所学领域规则的全局模块化解释；同时（iii）在性能上取得显著提升……

    arXiv:2605.29076v3 Announce Type: replace-cross  Abstract: LLMs have advanced text classification, yet existing paradigms face a trade-off: supervised (label only) fine-tuning is scalable but offers limited reasoning on complex text and lacks broader model transparency, while discrete prompt optimization offers human-readable instructions but struggles with performance and scalability. We introduce eXTC (eXplainable Text Classifier), centered on a Standard Operating Procedure (SOP, or rulebook) learned in natural language, in three stages: (1) learning the SOP via a new Structured Prompt Optimization algorithm; (2) SOP-grounded reasoning distillation from a large teacher LLM into a compact LM; and (3) expanding reasoning capabilities beyond the SOP via reinforcement learning. This design enables eXTC to provide (i) fast inference via a compact LM, with (ii) inference-time local reasoning traces, alongside a global, modular explanation of its learned domain rules, while (iii) significan
    
[^165]: MobileMoE：扩展端侧混合专家模型

    MobileMoE: Scaling On-Device Mixture of Experts

    [https://arxiv.org/abs/2605.27358](https://arxiv.org/abs/2605.27358)

    提出MobileMoE端侧MoE语言模型系列，通过端侧MoE缩放定律确定内存与计算同时最优的架构（适度稀疏+细粒度专家+共享专家），在十亿以下激活参数规模上为端侧大语言模型建立了新的帕累托前沿。

    

    混合专家已成为千亿参数语言模型的事实标准架构，但其在十亿以下规模用于端侧部署的优势在很大程度上尚未被探索。为弥补这一空白，我们提出了MobileMoE，这是一系列具有十亿以下激活参数的端侧MoE语言模型（激活参数0.3-0.9B，总参数1.3-5.3B），为端侧大语言模型建立了新的帕累托前沿。我们首先制定了一个端侧MoE缩放定律，在移动设备内存和计算约束下联合优化MoE架构，确定了端侧最佳设计点——采用细粒度专家和共享专家的适度稀疏性——同时实现内存最优和计算最优。基于所推导的架构，我们采用涵盖预训练、中期训练、指令微调和量化感知训练的四阶段配方来训练MobileMoE，且全部使用开源数据集。在14个基准测试中，MobileMoE达到了……

    arXiv:2605.27358v2 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) has become the de facto architecture for hundred-billion-parameter language models, yet its advantages at sub-billion scales for on-device deployment remain largely unexplored. To close this gap, we present MobileMoE, a family of on-device MoE language models with sub-billion active parameters (0.3-0.9B active and 1.3-5.3B total) that establish a new Pareto frontier for on-device LLMs. We first formulate an on-device MoE scaling law that jointly optimizes MoE architecture under mobile memory and compute constraints, identifying an on-device sweet spot - moderate sparsity with fine-grained and shared experts - that is simultaneously memory and compute-optimal. Building on the derived architectures, we train MobileMoE with a four-stage recipe covering pre-training, mid-training, instruction fine-tuning, and quantization-aware training, all on open-source datasets. Across 14 benchmarks, MobileMoE matches o
    
[^166]: 基于利益相关者关联锚定文本嵌入

    Grounding Text Embeddings in Stakeholder Associations

    [https://arxiv.org/abs/2605.27168](https://arxiv.org/abs/2605.27168)

    该论文提出“利益相关者锚定练习”方法以评估文本嵌入与人类专家语义关联的对齐程度，发现神经文本嵌入的可靠性显著低于人类专家（差距18-27个百分点），且该错位会传播到下游聚类任务，并在英语和美国联邦AI用例中得到复现。

    

    文本嵌入被广泛用于分析复杂文本的大型语料库。然而，这些嵌入是否捕捉到与使用它们的人类专家相同的语义关系尚不清楚。这种表征对齐对于有效分析至关重要。我们提出了“利益相关者锚定练习”，这是一种使专家关联显性化、并将嵌入模型结果锚定于人类理解的方法。在针对丹麦政策议题的主要案例研究中，我们发现神经文本嵌入的可靠性显著低于人类专家（差距为18-27个百分点），且这种错位会传导至下游聚类性能（练习排名与聚类质量之间的Spearman ρ = 0.86）。一项针对美国联邦AI用例的次要研究采用数字化协议和不同的专家群体，在英语中复现了该差距（16.1个百分点），表明这一差距并非单一工具或领域的产物。

    arXiv:2605.27168v2 Announce Type: replace-cross  Abstract: Text embeddings are widely used to analyse large corpora of complex texts. However, it is unclear whether the embeddings capture the same semantic relations as the human experts using them. Such representational alignment is essential for valid analyses. We present the Stakeholder Grounding Exercise, a method for making expert associations explicit and grounding embedding model results in human understanding. In our primary case study on Danish policy issues, we find that neural text embeddings are substantially less reliable than human experts (18-27pp gap), and that this misalignment propagates to downstream clustering performance (Spearman $\rho = 0.86$ between exercise ranking and cluster quality). A secondary study on US Federal AI use cases replicates the gap (16.1pp) in English, using a digital protocol and a different community of experts -- demonstrating that the gap is not an artefact of a single instrument or domain.
    
[^167]: 通过角色分歧实现大语言模型的无训练文化对齐

    Training-Free Cultural Alignment of Large Language Models via Persona Disagreement

    [https://arxiv.org/abs/2605.10843](https://arxiv.org/abs/2605.10843)

    提出DISCA方法，无需训练、无需特定国家的数据或访问模型内部，仅通过基于世界价值观调查数据构建的四角色提示面板的共识与分歧机制，即可将权重冻结的大语言模型对齐到不同国家的道德偏好。

    

    大语言模型如今参与到依赖于道德偏好的决策中，而这些偏好因国家而异。模型的默认偏好最接近某一个人群，因此单一的部署会用一个国家的价值观来服务所有国家。现有的解决方案需要针对每种文化的训练数据、针对每个国家的奖励模型，或访问模型内部结构——而距离该默认值最远的国家往往缺乏这些资源。我们展示了在完全没有这些资源的情况下，能够将模型适配到多远的国家。我们的方法DISCA保持模型权重冻结，不使用任何国家特定的人类标注目标，只读取模型对每个答案选项的评分。该方法基于世界价值观调查的微观数据，将一个国家构建为由四个角色提示组成的评审面板。面板的共识设定一个以国家为条件的锚点，超出该锚点的损失规避式精细化调整只有在两次独立运行结果一致时才被保留，同时一个持续更新的国家级先验会汇集已做出的决策……

    arXiv:2605.10843v3 Announce Type: replace-cross  Abstract: Large language models now take part in decisions that rest on moral preferences, and those preferences differ by country. A model's default preferences sit closest to one population, so a single deployment serves every country with one country's values. Existing fixes need per-culture training data, a per-country reward model, or access to model internals, which the countries furthest from that default rarely have. We show how far a country can be reached without any of them. Our method, DISCA, keeps the model's weights frozen, sees no country-specific human targets, and reads only the model's score for each answer option. It builds a country as a panel of four persona prompts grounded in World Values Survey microdata. The panel's consensus sets a country-conditioned anchor, a loss-averse refinement beyond that anchor is kept only when two independent runs of it agree, and a running country-level prior pools the decisions alrea
    
[^168]: 智能体集体不应自行检测其内部的冒名顶替者：一个国际象棋案例研究

    Agent Collectives Should Not Detect Their Own Imposters: A Chess Case Study

    [https://arxiv.org/abs/2605.09027](https://arxiv.org/abs/2605.09027)

    研究表明让智能体集体自行检测内部冒名顶替者会适得其反，而采用能够从少量样本快速适应新型攻击的轻量级元训练外部检测器才是更优策略。

    

    AI智能体集体在复杂任务上可以超越任何单一智能体，但仍然容易受到故意误导同伴的冒名顶替者（imposter）的攻击。由于哪怕只有一个冒名顶替者也可能破坏整个集体，可靠的检测机制至关重要。我们研究了两种策略：将检测功能嵌入参与智能体自身内部，或者雇佣一个专门的、只观察但不参与集体讨论的外部检测器。在一个特意设计的小型但具有挑战性的国际象棋测试平台上，我们发现仅仅警告智能体可能存在冒名顶替者反而会产生反效果：即使在没有对手的情况下，它也会降低决策质量，引发无益的社交动态，并向冒名顶替者泄露信息。外部检测器虽然可以避免这些陷阱，但必须随着冒名顶替者演化出新策略而不断适应。我们证明，一个轻量级的元训练检测器仅凭少量样本就能快速重新校准，以应对未见过的攻击，其效率远高于其他方法。

    arXiv:2605.09027v4 Announce Type: replace-cross  Abstract: Collectives of AI agents can outperform any single agent on complex tasks, yet remain vulnerable to imposters that deliberately mislead their peers. Because even one such imposter can sabotage the collective, reliable detection is essential. We examine two strategies: embedding detection inside the participating agents, or employing a dedicated external detector that observes the deliberation without joining it. On a deliberately small but challenging chess-based testbed, we find that simply warning agents of possible imposters is counterproductive. It degrades decisions even in the absence of adversaries, triggers unhelpful social dynamics, and leaks information to the imposter. An external detector avoids these pitfalls, yet must continually adapt as imposters evolve new strategies. We show that a lightweight meta-trained detector can rapidly recalibrate to unseen attacks from only a handful of examples, far more efficiently 
    
[^169]: PairAlign：一种基于自对齐的自回归词元化框架及其在音频词元化中的应用

    PairAlign: A Framework for Autoregressive Tokenization via Self-Alignment with Applications to Audio Tokenization

    [https://arxiv.org/abs/2605.06582](https://arxiv.org/abs/2605.06582)

    PairAlign提出了一种通过自回归自对齐实现紧凑音频词元化的框架，通过让两个内容保持视图预测统一的规范锚点目标，实现了序列一致性、紧凑性和长度等特性的可控音频离散表示。

    

    现代学习系统使用连续向量来表示感知信号，但比较、检索、记忆、对齐和推理往往是符号化的。在语言中，词元提供了这种接口；而对于语音和音频，这种接口必须通过学习获得。现有的音频词元化器依赖于局部量化、聚类或重建，使得序列一致性、紧凑性、长度、终止和编辑几何特性只能被间接控制。我们提出了PairAlign，一个通过自回归自对齐实现紧凑音频词元化的框架。编码器将语音映射为连续条件，自回归解码器从BOS到EOS生成词元。给定两个保持内容的视图，PairAlign推导出一个规范的锚点目标，并训练两个视图来预测它，同时以批内无关目标作为竞争序列。该框架首先从VQ目标学习自回归桥梁，然后过渡到EMA教师自对齐（摘要在此处截断）。

    arXiv:2605.06582v4 Announce Type: replace-cross  Abstract: Modern learning systems represent perceptual signals with continuous vectors, but comparison, retrieval, memory, alignment, and reasoning are often symbolic. In language, tokens provide this interface; for speech and audio, it must be learned. Existing audio tokenizers rely on local quantization, clustering, or reconstruction, leaving sequence consistency, compactness, length, termination, and edit geometry only indirectly controlled. We introduce PairAlign, a framework for compact audio tokenization through autoregressive self-alignment. An encoder maps speech to a continuous condition, and an autoregressive decoder emits tokens from BOS to EOS. Given two content-preserving views, PairAlign derives a canonical anchor target and trains both views to predict it, with unrelated in-batch targets as competing sequences. It first learns an autoregressive bridge from VQ targets and then transitions to EMA-teacher self-alignment with 
    
[^170]: UniVer：多步与多草稿投机解码的统一视角

    UniVer: A Unified Perspective for Multi-step and Multi-draft Speculative Decoding

    [https://arxiv.org/abs/2605.04543](https://arxiv.org/abs/2605.04543)

    本文提出UniVer，将树状投机解码的验证统一建模为条件最优传输问题，通过前缀接受概率作为动态缩放因子主动引导草稿选择，从而联合优化多步依赖与多草稿分支之间的耦合关系。

    

    投机解码通过“先起草、后验证”的方式加速大语言模型，其中验证过程可以建模为最优传输（OT）问题。现有方法通常将多草稿与多步骤两个维度割裂开来处理：要么对单步草稿应用扁平最优传输，要么对树状结构候选进行逐token的拒绝采样。这种割裂导致联合场景（即多步依赖与多草稿分支同时存在）得不到充分优化，因为局部验证规则未能利用候选树水平维度与垂直维度之间的耦合关系。本文提出一种统一视角，将基于树的验证问题转化为条件最优传输问题。我们的关键洞察是：垂直方向的依赖关系可以通过前缀接受概率来抽象，这些概率作为动态缩放因子，能够主动引导水平方向的草稿选择。基于这一原理，我们提出了UniVer，一种验证算法……（摘要原文在此处截断）

    arXiv:2605.04543v2 Announce Type: replace  Abstract: Speculative decoding accelerates Large Language Models via draft-then-verify, where verification can be framed as an Optimal Transport (OT) problem. Existing approaches typically handle multi-draft and multi-step aspects in isolation, applying either flat OT to single-step drafts or per-token rejection sampling to tree-structured candidates. This separation leaves the joint regime (where multi-step dependencies meet multi-draft branching) poorly optimized, as local verification rules fail to exploit the coupling between horizontal and vertical dimensions of candidate trees. In this paper, we propose a unified perspective that casts tree-based verification as a conditional OT problem. Our key insight is that vertical dependencies can be abstracted through prefix acceptance probabilities, which act as dynamic scaling factors to actively guide horizontal draft selection. Based on this principle, we introduce UniVer, a verification algor
    
[^171]: 自适应共形预测：提升大语言模型生成内容的事实性

    Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models

    [https://arxiv.org/abs/2604.13991](https://arxiv.org/abs/2604.13991)

    提出了一种提示词自适应的共形预测方法，通过提示词相关的校准提升大语言模型生成内容的事实性，在保持边际覆盖保证的同时改善条件覆盖率，并支持选择性预测以过滤不可靠输出。

    

    大语言模型（LLM）容易生成事实性错误的输出。近期的工作将共形预测应用于为大语言模型生成内容的事实性提供不确定性估计和统计保证。然而，现有方法通常无法自适应提示词，限制了其捕捉输入相关变化的能力。因此，对于给定的任务或提示，这些方法可能过滤掉过少的条目（导致过度覆盖）或过滤掉过多条目（导致覆盖不足）。我们提出了一种自适应共形预测方法，将共形分数变换方法扩展到大语言模型中，并应用于长文本生成和多选题问答任务。该方法实现了依赖于提示词的校准，在保持边际覆盖保证的同时提高了条件覆盖率。此外，该方法还自然地支持选择性预测，允许在下游应用中过滤掉不可靠的陈述或答案选项。

    arXiv:2604.13991v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are prone to generating factually incorrect outputs. Recent work has applied conformal prediction to provide uncertainty estimates and statistical guarantees for the factuality of LLM generations. However, existing approaches are typically not prompt-adaptive, limiting their ability to capture input-dependent variability. As a result, they may filter out too few items (leading to over-coverage) or too many (under-coverage) for a given task or prompt. We propose an adaptive conformal prediction approach that extends conformal score transformation methods to LLMs, with applications to long-form generation and multiple-choice question answering. This enables prompt-dependent calibration, retaining marginal coverage guarantees while improving conditional coverage. In addition, the approach naturally supports selective prediction, allowing unreliable claims or answer choices to be filtered out in downstr
    
[^172]: Appear2Meaning：一个用于从图像推断结构化文化元数据的跨文化基准

    Appear2Meaning: A Cross-Cultural Benchmark for Structured Cultural Metadata Inference from Images

    [https://arxiv.org/abs/2604.07338](https://arxiv.org/abs/2604.07338)

    提出了包含750件博物馆物品的跨文化基准Appear2Meaning，用于评估视觉语言模型仅凭图像推断标题、文化、时期、产地和创作者等结构化文化元数据的能力，结果显示当前模型的完整记录兼容率仍然很低。

    

    博物馆记录通过文化和历史关系表达了人们如何理解遗产物品。从图像中恢复这些人工赋予的归属信息，对视觉语言模型（VLM）来说是一项技术挑战。我们提出了Appear2Meaning，这是一个包含750件博物馆物品的跨文化基准数据集，用于仅凭图像恢复标题、文化、时期、产地和创作者五类信息。其评估框架区分了字段级别的兼容性与完整记录的兼容性，并采用人工验证进行语义判断。在九个VLM模型上，平均字段准确率为36.16%–49.82%；7.20%–17.02%的记录在所有四个上下文字段上均兼容，而包含标题在内的全部五个字段均兼容的记录仅占2.13%–7.87%。对于测试的两个模型，仅图像输入在文化、时期和产地方面的参考兼容性比仅使用博物馆名称和标题的对照组高出12.6–32.2个百分点。

    arXiv:2604.07338v3 Announce Type: replace-cross  Abstract: Museum records express how people understand heritage objects through cultural and historical relations. Recovering these human attributions from images is a technical challenge for vision-language models (VLMs). We introduce Appear2Meaning, a cross-cultural benchmark of 750 museum objects for image-only recovery of title, culture, period, origin, and creator. Its evaluation framework distinguishes field-level from complete-record compatibility and uses human validation for semantic judgement. Across nine VLMs, mean field accuracy is 36.16--49.82%; 7.20--17.02% of records have all four contextual fields compatible, and 2.13--7.87% have all five, including title. For the two tested models, image-only input yields 12.6--32.2 percentage points higher reference compatibility for culture, period, and origin than a museum-and-title-only control.
    
[^173]: Graph-RAG 中的推理瓶颈：面向多跳问答的结构化提示与上下文压缩

    The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA

    [https://arxiv.org/abs/2603.14045](https://arxiv.org/abs/2603.14045)

    该论文发现 Graph-RAG 的真正瓶颈在于推理而非检索，并提出 SPARQL 思维链提示与免 LLM 调用的图游走上下文压缩两项方法，显著提升多跳问答准确率并将上下文压缩约 60%。

    

    Graph-RAG 系统通过将文档索引到知识图谱中来实现强大的多跳问答能力，但强大的检索并不保证能产生正确的答案。我们在三个多跳问答基准（HotpotQA、MuSiQue、2WikiMultiHopQA）上评估了领先的 Graph-RAG 系统 KET-RAG，发现 77% 至 91% 的问题在检索到的上下文中已包含标准答案，但准确率仅为 35% 至 78%，且 73% 至 84% 的错误源于推理失败。我们提出了两项增强方法：（i）SPARQL 思维链提示，将问题分解为与实体-关系上下文对齐的三元组模式查询；（ii）图游走压缩，通过知识图谱遍历将上下文压缩约 60%，且无需任何大语言模型调用。SPARQL 思维链提示将准确率提升了 2 至 14 个百分点；在较小模型上与结构化提示配合使用时，图游走压缩平均可额外带来 6 个百分点的提升。令人惊讶的是，我们发现通过问题类型路由，……

    arXiv:2603.14045v3 Announce Type: replace-cross  Abstract: Graph-RAG systems achieve strong multi-hop question answering by indexing documents into knowledge graphs, but strong retrieval does not guarantee strong answers. Evaluating KET-RAG, a leading Graph-RAG system, on three multi-hop QA benchmarks (HotpotQA, MuSiQue, 2WikiMultiHopQA), we find that 77% to 91% of questions have the gold answer in the retrieved context, yet accuracy is only 35% to 78%, and 73% to 84% of errors are reasoning failures. We propose two augmentations: (i) SPARQL chain-of-thought prompting, which decomposes questions into triple-pattern queries aligned with the entity-relationship context, and (ii) graph-walk compression, which compresses the context by ~60% via knowledge-graph traversal with no LLM calls. SPARQL CoT improves accuracy by +2 to +14 pp; graph-walk compression adds +6 pp on average when paired with structured prompting on smaller models. Surprisingly, we show that, with question-type routing, 
    
[^174]: 发言还是保持沉默：多方对话中的上下文感知话轮转换

    Speak or Stay Silent: Context-Aware Turn-Taking in Multi-Party Dialogue

    [https://arxiv.org/abs/2603.11409](https://arxiv.org/abs/2603.11409)

    该论文提出“上下文感知话轮转换”新任务——在多方对话的停顿处判断AI助手应发言还是沉默，发现八个现有大语言模型零样本下均无法胜任，并借助带推理轨迹的监督微调将平衡准确率提升最多23个百分点。

    

    现有的语音AI助手将每个检测到的停顿都视为发言的邀请。这种方式在双人对话中行之有效，但在多方对话场景中——即AI助手与多个说话者共同参与对话时——停顿大量存在且含义模糊。一个在每个停顿处都发言的助手会变得具有干扰性而非有用性。在这项工作中，我们提出了上下文感知话轮转换的概念：在每次检测到停顿时，基于完整的对话上下文，我们的方法来决定助手应该发言还是保持沉默。我们构建了一个包含超过12万条标注对话的基准数据集，涵盖三个多方对话语料库。通过对八个近期大语言模型的评估，我们发现它们在零样本提示下始终无法完成上下文感知话轮转换任务。随后，我们提出了一种带有推理轨迹的监督微调方法，将平衡准确率提升了最多23个百分点。我们的研究结果表明，上下文感知话轮转换并非一种涌现能力（原文在此处截断）。

    arXiv:2603.11409v2 Announce Type: replace  Abstract: Existing voice AI assistants treat every detected pause as an invitation to speak. This works in dyadic dialogue, but in multi-party settings, where an AI assistant participates alongside multiple speakers, pauses are abundant and ambiguous. An assistant that speaks on every pause becomes disruptive rather than useful. In this work, we formulate context-aware turn-taking: at every detected pause, given the full conversation context, our method decides whether the assistant should speak or stay silent. We introduce a benchmark of over 120K labeled conversations spanning three multi-party corpora. Evaluating eight recent large language models, we find that they consistently fail at context-aware turn-taking under zero-shot prompting. We then propose a supervised fine-tuning approach with reasoning traces, improving balanced accuracy by up to 23 percentage points. Our findings suggest that context-aware turn-taking is not an emergent ca
    
[^175]: 生成式引擎优化中引用失败的诊断与修复

    Diagnosing and Repairing Citation Failures in Generative Engine Optimization

    [https://arxiv.org/abs/2603.09296](https://arxiv.org/abs/2603.09296)

    本文提出首个引用失败模式分类体系与智能体系统AgentGEO，通过诊断文档未被AI生成回答引用的具体原因并施加针对性修复，显著提升生成式引擎优化中的引用效果。

    

    生成式引擎优化旨在提升内容在AI生成回答中的可见性。然而，现有方法衡量的是贡献度（即文档对回答的影响程度），而非引用——真正将流量引回内容创作者的机制。此外，这些方法统一套用通用的改写规则，无法诊断出单个文档为何未被引用。本文提出了一种面向GEO的诊断方法，探究文档为何未能被引用并进行针对性干预。我们构建了一个统一框架，包括：（1）首个覆盖引用流程不同阶段的引用失败模式分类体系；（2）AgentGEO——一个智能体系统，利用该分类体系诊断失败原因，从相应的工具库中选择针对性修复方案，并迭代优化直至成功获得引用；（3）一个以文档为中心的基准测试，用于评估优化效果能否泛化到未见过的查询。

    arXiv:2603.09296v2 Announce Type: replace-cross  Abstract: Generative Engine Optimization (GEO) aims to improve content visibility in AI-generated responses. However, existing methods measure contribution (how much a document influences a response) rather than citation, the mechanism that actually drives traffic back to creators. Also, these methods apply generic rewriting rules uniformly, failing to diagnose why individual documents are not cited. This paper introduces a diagnostic approach to GEO that asks why a document fails to be cited and intervenes accordingly. We develop a unified framework comprising: (1) the first taxonomy of citation failure modes spanning different stages of a citation pipeline; (2) AgentGEO, an agentic system that diagnoses failures using this taxonomy, selects targeted repairs from a corresponding tool library, and iterates until citation is achieved; and (3) a document-centric benchmark evaluating whether optimizations generalize across held-out queries.
    
[^176]: 面向函数调用智能体的合成数据中的语言与参数多样性

    Linguistic and Argument Diversity in Synthetic Data for Function-Calling Agents

    [https://arxiv.org/abs/2601.17829](https://arxiv.org/abs/2601.17829)

    该论文提出了一种无需手工规则或分类体系、通过在查询和参数两个层面优化通用多样性指标来生成函数调用智能体合成训练数据的方法，在保持正确性相当的同时显著提升了数据的语言与参数多样性。

    

    构建函数调用智能体已成为扩展模型能力的一条有前景的途径。该任务的一个主要挑战是获取高质量的多样化训练数据。以往的工作强调函数、调用模式和交互轮次的多样性，但请求的语言多样性以及参数（如 \texttt{city\_name}、\texttt{stock\_ticker}）的覆盖范围仍未得到充分探索。我们提出了一种方法，通过在查询和参数两个层面优化通用多样性指标来生成合成数据集，无需依赖手工构建的规则或分类体系，因此对不同用例具有鲁棒性。我们通过内在和外在测试验证了该技术的有效性，并将其与最先进（SoTA）的数据生成方法进行比较。结果表明，我们在多样性方面优于基线方法，同时保持了相当的正确性。此外，当用作训练集时……（原文摘要至此截断）

    arXiv:2601.17829v2 Announce Type: replace-cross  Abstract: The construction of function calling agents has emerged as a promising avenue for extending model capabilities. A major challenge for this task is obtaining high quality diverse data for training. Prior work emphasizes diversity in functions, invocation patterns, and interaction turns, yet linguistic diversity of requests and coverage of arguments (e.g., \texttt{city\_name}, \texttt{stock\_ticker}) remain underexplored. We propose a method that generates synthetic datasets via optimizing general-purpose diversity metrics across both queries and arguments, without relying on hand-crafted rules or taxonomies, making it robust to different usecases. We demonstrate the effectiveness of our technique via both intrinsic and extrinsic testing, comparing it to SoTA data generation methods. We show a superiority over baselines in terms of diversity, while keeping comparable correctness. Additionally, when used as a training set, the mod
    
[^177]: 多模态的“球门柱”在哪里？论基础模型识别视频中精彩片段的能力

    Where is the multimodal goal post? On the Ability of Foundation Models to Recognize Highlights in Videos

    [https://arxiv.org/abs/2601.16333](https://arxiv.org/abs/2601.16333)

    本研究利用足球比赛官方集锦零成本构建了一个评估数据集，用以测试多模态基础模型识别视频中重要事件的能力，发现现有最先进模型无法可靠地理解事件的上下文重要性。

    

    基础模型被广泛应用于许多需要对按时间排序的多模态事件进行语言理解与生成的现实场景中。在这项工作中，我们研究了模型识别视频中最重要的子事件的能力，这是对多模态事件进行总结、推理或叙述的基础前提。具体而言，我们聚焦于足球比赛，评估模型区分比赛中重要与非重要子事件的能力。为此，我们利用公开可得的足球精彩集锦构建了一个新数据集，这些集锦包含了由专业专家认定的比赛最重要时刻，且无需任何额外的标注成本。利用该数据集，我们比较了多个最先进的多模态模型，结果表明它们无法可靠地捕捉上下文重要性。超越标准评估指标的进一步分析揭示了模型的倾向……

    arXiv:2601.16333v3 Announce Type: replace-cross  Abstract: Foundation models are used for many real-world applications involving language understanding and generation from temporally-ordered multimodal events. In this work, we study the ability of models to identify the most important sub-events in a video, which is a fundamental prerequisite for summarizing, reasoning over, or narrating multimodal events. Specifically, we focus on football games and evaluate models on their ability to distinguish between important and non-important sub-events in a game. To this end, we construct a new dataset by leveraging publicly available football highlight reels containing the most important moments of a game according to professional experts, without any additional annotation costs. Using our dataset, we compare several state-of-the-art multimodal models and show that they fail to reliably capture contextual importance. Analyses of models beyond standard evaluation metrics reveal their tendency t
    
[^178]: 多维提示链提升开放域对话生成

    Multi-Dimensional Prompt Chaining to Improve Open-Domain Dialogue Generation

    [https://arxiv.org/abs/2601.01037](https://arxiv.org/abs/2601.01037)

    提出一种整合自然度、连贯性和吸引力三个维度的多维提示链框架，使小语言模型（如Llama-2-7B）在开放域对话生成中达到与大型模型相当的性能。

    

    小语言模型（SLM）具有显著的部署优势，但在开放域场景中往往难以匹敌更大模型的对话质量。本文提出了一种多维提示链框架，整合自然度、连贯性和吸引力三个维度，以增强开放域对话生成的人性化程度。我们将该框架应用于两个小语言模型——TinyLlama和Llama-2-7B，并将其性能与更大模型生成的回复进行基准对比，包括Llama-2-70B和GPT-3.5 Turbo。随后，我们采用自动评估和人工评估，基于多样性、上下文连贯性以及整体质量来评估回复。结果表明，完整框架可将回复多样性最高提升29%，上下文连贯性最高提升28%，吸引力和自然度最高提升29%。值得注意的是，Llama-2-7B达到了与更大模型相当的性能。

    arXiv:2601.01037v2 Announce Type: replace-cross  Abstract: Small language models (SLMs) offer significant deployment advantages but often struggle to match the dialogue quality of larger models in open-domain settings. In this paper, we propose a multi-dimensional prompt-chaining framework that integrates Naturalness, Coherence, and Engagingness dimensions to enhance human-likeness in open-domain dialogue generation. We apply the framework to two SLMs, TinyLlama and Llama-2-7B, and benchmark their performance against responses generated by substantially larger models, including Llama-2-70B and GPT-3.5 Turbo. We then employ automatic and human evaluation to assess the responses based on diversity, contextual coherence, as well as overall quality. Results show that the full framework improves response diversity by up to 29%, contextual coherence by up to 28%, and engagingness as well as naturalness by up to 29%. Notably, Llama-2-7B achieves performance comparable to substantially larger 
    
[^179]: 多尺度聚合层次注意力（MAHA）：一种基于约束与共识的大语言模型高效上下文建模方法

    Multiscale Aggregated Hierarchical Attention (MAHA): A Constrained and Consensus-Based Approach to Efficient Contextual Modeling in Large Language Models

    [https://arxiv.org/abs/2512.14925](https://arxiv.org/abs/2512.14925)

    该论文提出多尺度聚合层次注意力（MAHA）框架，通过层次化分解与显式约束聚合重构注意力机制，以突破多头自注意力的二次复杂度瓶颈，实现大语言模型高效的多尺度长上下文建模。

    

    多头自注意力（MHSA）的二次方计算复杂度仍然是大语言模型（LLM）在长上下文任务扩展中的根本瓶颈。虽然稀疏注意力和线性化注意力机制试图缓解这一问题，但它们往往损害了全局依赖关系的表示能力，或无法有效捕捉多尺度的语义粒度。在本文中，我们提出了多尺度聚合层次注意力，这是一种新颖的架构框架，通过层次化分解和显式约束聚合来重新构建注意力机制。与在单一分辨率下处理词元交互的传统方法不同，MAHA通过可学习的下采样算子将输入序列动态划分为层次化的尺度。对齐后的各尺度输出则通过概率单纯形上的softmax归一化权重，或固定数量的可微分共识……（原文在此处截断）

    arXiv:2512.14925v3 Announce Type: replace  Abstract: The quadratic computational complexity of Multi-Head Self-Attention (MHSA) remains a fundamental bottleneck in scaling Large Language Models (LLMs) for long-context tasks. While sparse and linearized attention mechanisms attempt to mitigate this, they often compromise the representation of global dependencies or fail to capture multiscale semantic granularity effectively. In this paper, we propose Multiscale Aggregated Hierarchical Attention (MAHA), a novel architectural framework that reformulates the attention mechanism through hierarchical decomposition and explicitly constrained aggregation. Unlike conventional approaches that treat token interactions at a single resolution, MAHA dynamically partitions the input sequence into hierarchical scales via learnable downsampling operators. The aligned scale outputs are combined either by softmax-normalized weights on the probability simplex or by a fixed number of differentiable consens
    
[^180]: 面向泰卢固语文化保护的计算社会语言学：Chandassu格律模式识别的新型算法

    Computational Social Linguistics for Telugu Cultural Preservation: Novel Algorithms for Chandassu Metrical Pattern Recognition

    [https://arxiv.org/abs/2510.01233](https://arxiv.org/abs/2510.01233)

    该研究开发了首个泰卢固语格律诗歌Chandassu的数字分析框架，通过构建4,651首标注诗歌的协作数据集以及韵律感知分词、音节分类和模式识别三大工具，实现了91.73%准确率的自动格律识别，为濒危的泰卢固语诗歌文化传统保护提供了计算社会科学解决方案。

    

    本研究提出了一种计算社会科学方法，用于保护泰卢固语Chandassu——这一格律诗歌传统承载了数个世纪的集体文化智慧。我们开发了首个用于分析泰卢固语韵律模式的综合性数字框架，将传统社区知识与现代计算方法相融合。我们的社会计算方法包括协作构建包含4,651首标注诗歌的数据集、验证的语言模式以及基于文化认知的算法设计。该框架包含用于韵律感知分词的AksharamTokenizer、用于分类轻重音节的LaghuvuGuruvu Generator，以及用于自动模式识别的PadyaBhedam Checker。我们的算法在所提出的Chandassu Score指标上达到91.73%的准确率，评估指标反映了传统文学标准。这项工作展示了计算社会科学如何能够保护濒危的文化遗产。

    arXiv:2510.01233v3 Announce Type: replace  Abstract: This research presents a computational social science approach to preserving Telugu Chandassu, the metrical poetry tradition representing centuries of collective cultural intelligence. We develop the first comprehensive digital framework for analyzing Telugu prosodic patterns, bridging traditional community knowledge with modern computational methods. Our social computing approach involves collaborative dataset creation of 4,651 annotated padyams, validated linguistic patterns, and culturally informed algorithmic design. The framework includes AksharamTokenizer for prosody-aware tokenization, LaghuvuGuruvu Generator for classifying light and heavy syllables, and PadyaBhedam Checker for automated pattern recognition. Our algorithm achieves 91.73\% accuracy on the proposed Chandassu Score, with evaluation metrics reflecting traditional literary standards. This work demonstrates how computational social science can preserve endangered c
    
[^181]: EEG-MedRAG：通过分层超图检索增强生成提升基于脑电图的临床决策

    EEG-MedRAG: Enhancing EEG-based Clinical Decision-Making via Hierarchical Hypergraph Retrieval-Augmented Generation

    [https://arxiv.org/abs/2508.13735](https://arxiv.org/abs/2508.13735)

    本文提出EEG-MedRAG，一个基于三层超图的检索增强生成框架，将脑电领域知识、患者病例和大规模脑电记录统一为n元关系超图以实现语义-时间联合检索与循证临床推理，并借助首个跨五种用户角色评估答案一致性的脑电基准测试，在多个大语言模型上超越全部基线方法。

    

    随着脑电图（EEG）在神经科学和临床实践中的广泛应用，如何高效检索并对大规模、多源、异构的脑电数据进行语义解读已成为一项紧迫的挑战。EEG-MedRAG通过一个基于三层超图的检索增强生成框架来应对这一挑战，该框架将脑电领域知识、个体患者病例以及大规模脑电记录统一为一个可遍历的n元关系超图，从而实现语义-时间联合检索与基于证据的临床推理。其配套的基准测试首次将针对同一临床事实的脑电问题在五种用户角色间进行配对，共包含700个问题，覆盖七个临床领域，支持对跨角色答案一致性的直接比较。在三种大语言模型骨干上，EEG-MedRAG在整体答案正确性和词元级F1分数上均优于全部四个基线方法。

    arXiv:2508.13735v3 Announce Type: replace  Abstract: With the widespread application of electroencephalography (EEG) in neuroscience and clinical practice, efficiently retrieving and semantically interpreting large-scale, multi-source, heterogeneous EEG data has become a pressing challenge. EEG-MedRAG addresses this challenge through a three-layer hypergraph-based retrieval-augmented generation framework that unifies EEG domain knowledge, individual patient cases, and large-scale EEG recordings into a traversable n-ary relational hypergraph, enabling joint semantic-temporal retrieval and evidence-grounded clinical reasoning. The accompanying benchmark is the first to pair EEG questions about the same clinical fact across five user roles. Its 700 questions span seven clinical domains and support direct comparisons of answer consistency across roles. Across three LLM backbones, EEG-MedRAG outperforms all four baselines in both overall answer correctness and token-level F1. It achieves up
    
[^182]: REAL：通过读取Transformer激活实现语言模型引导中的精确定位

    REAL: Reading Out Transformer Activations for Precise Localization in Language Model Steering

    [https://arxiv.org/abs/2506.08359](https://arxiv.org/abs/2506.08359)

    提出REAL框架，通过对Transformer各模块的隐藏激活训练向量量化自编码器来评分行为相关性，从而在语言模型引导中实现行为相关模块的精确定位与选择。

    

    推理时引导旨在在不改变大语言模型参数的情况下改变其响应。一个关键挑战在于选择最能支配目标行为的内部模块；现有方法往往依赖简单的线索或临时的启发式规则，导致次优或意外的效果。在这项工作中，我们引入REAL，一个用于识别Transformer中与行为相关模块（注意力头或层）的新型框架。对于每个模块，我们在其隐藏激活上训练一个向量量化自编码器（VQ-AE），通过共享的可学习码本将潜在空间划分为行为相关和行为无关的子空间。我们通过使用二分类指标评估VQ-AE编码区分行为一致响应和行为违规响应的有效性，来量化每个模块的行为相关性。该相关性分数同时用于模块选择和引导强度的确定。我们在多种任务上对REAL进行了评估。

    arXiv:2506.08359v4 Announce Type: replace  Abstract: Inference-time steering aims to alter an LLM's responses without changing its parameters. A key challenge lies in selecting internal modules that most strongly govern the target behavior; existing approaches often rely on simplistic cues or ad hoc heuristics, leading to suboptimal or unintended effects. In this work, we introduce REAL, a novel framework for identifying behavior-relevant modules (heads or layers) in Transformers. For each module, we train a vector-quantized autoencoder (VQ-AE) on its hidden activations, partitioning the latent space into behavior-relevant and behavior-irrelevant subspaces via a shared, learnable codebook. We quantify each module's behavioral relevance by evaluating how effectively the VQ-AE encodings distinguish between behavior-aligned and behavior-violating responses using a binary classification metric. This relevance score informs both module selection and steering strength. We evaluate REAL acros
    
[^183]: TRIM：通过目标行级迭代度量驱动剪枝实现极端稀疏化

    TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning

    [https://arxiv.org/abs/2505.16743](https://arxiv.org/abs/2505.16743)

    TRIM提出一种行级迭代度量驱动剪枝方法，通过为层内每个输出行分配不同的稀疏比率，在高稀疏度（70-80%）下显著降低大语言模型的困惑度，例如使Qwen2.5-14B在80%稀疏度下的困惑度近乎减半。

    

    大型语言模型（LLMs）带来了巨大的计算和内存挑战。现有的一次性剪枝方法通常对层内的所有行使用统一的稀疏度目标，这在高稀疏度下可能过于粗糙。本工作引入了TRIM（目标行级迭代度量驱动剪枝），这是一种基于前向统计信息的封装方法，能够在保持模型权重冻结的同时，为各个输出行分配不同的稀疏比率。TRIM将“剪枝哪些权重”（评分）与“每行剪枝多少”（分配）这两个问题分开处理，并可与现有的评分器和层级分配方法组合使用。在Qwen2.5、LLaMA-2和OPT模型上的实验表明，其最大收益出现在70-80%稀疏度区间。在80%稀疏度下，将TRIM添加到OWL上，可使Qwen2.5-14B的WikiText-2困惑度从348.48降至180.67，使OPT-13B的困惑度从6461.43降至324.14。我们得出结论，细粒度的行级分配是显著提升模型性能的有效机制。

    arXiv:2505.16743v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) present substantial computational and memory challenges. Existing one-shot pruning methods often use a uniform sparsity target for all rows within a layer, which can be too coarse at high sparsity. This work introduces TRIM (Targeted Row-wise Iterative Metric-driven pruning), a forward-statistics-based wrapper that allocates different sparsity ratios to individual output rows while keeping model weights frozen. TRIM separates the choice of which weights to prune (scoring) from how many to prune in each row (allocation), and composes with existing scorers and layer-wise allocations. Across Qwen2.5, LLaMA-2, and OPT models, its largest gains occur at 70-80% sparsity. At 80%, adding TRIM to OWL reduces WikiText-2 perplexity from 348.48 to 180.67 for Qwen2.5-14B and from 6461.43 to 324.14 for OPT-13B. We conclude that fine-grained, row-wise allocation is an effective mechanism for substantially improvin
    
[^184]: 论结构化剪枝在小语言模型预训练中的效率

    On the Efficiency of Structured Pruning in Small Language Model Pretraining

    [https://arxiv.org/abs/2503.05920](https://arxiv.org/abs/2503.05920)

    提出IDEA Prune集成式放大-剪枝流水线，在单一余弦退火学习率调度下统一放大模型预训练、迭代结构化剪枝和恢复过程，在全成本核算下比传统分阶段剪枝方法更高效地训练出小语言模型。

    

    在边缘设备上部署生成式语言模型的公司，出于法律、数据治理或产品定制化等要求，通常需要从头训练小模型。另一种方法是先训练一个放大的模型，再通过结构化剪枝将其压缩到目标尺寸，但一旦将放大模型的预训练成本计算在内，这种做法的优势就变得不明确了。我们引入了“放大-剪枝”框架，在全成本核算的条件下研究这一问题，并表明传统的放大-剪枝流水线——即分阶段地训练放大模型、剪枝并恢复——在相同的总token预算下并不总是优于直接训练小模型。为克服这一局限性，我们提出了IDEA Prune，这是一种集成式的放大-剪枝流水线，它在单一的余弦退火学习率调度下，将放大模型预训练、迭代结构化剪枝和恢复统一为一体。

    arXiv:2503.05920v2 Announce Type: replace-cross  Abstract: Companies that deploy generative language models on edge devices often need to train small models from scratch due to legal, data-governance, or product-customization requirements. An alternative approach is to train an enlarged model and compress it to the target size via structured pruning, but this advantage becomes unclear once the cost of pretraining the enlarged model is included. We introduce the enlarge-and-prune framework to study this question under full-cost accounting, and show that conventional enlarge-and-prune pipelines -- which train the enlarged model, prune it, and recover in separate stages -- do not consistently outperform direct small-model training under the same total token budget. To overcome this limitation, we propose IDEA Prune, an integrated enlarge-and-prune pipeline that unifies enlarged model pretraining, iterative structured pruning, and recovery under a single cosine annealing learning rate sche
    
[^185]: MMGenBench：一种从文本到图像生成视角评估大型多模态模型的自动化流水线

    MMGenBench: An Automated Pipeline for Evaluating LMMs from a Text-to-Image Generation Perspective

    [https://arxiv.org/abs/2411.14062](https://arxiv.org/abs/2411.14062)

    MMGenBench是一个无需预定义问题或参考描述的自动化基准框架，通过“图像描述生成→文生图重建→编码器比对”的流水线从原始图像评估大型多模态模型，对68个LMM的评估揭示了总分可能掩盖的模式特定重建质量差异。

    

    现有基准测试通常依赖预定义的问题或参考描述，这限制了它们在新图像集上评估详细描述的灵活性。我们提出了MMGenBench，这是一个自动化基准测试框架，可直接从原始图像评估大型多模态模型（LMMs），无需逐图像的问题或参考描述。该流水线先生成图像描述，再使用冻结的文本到图像模型重建图像，并通过冻结的图像编码器比较原始图像与重建图像。它能够产出可复用的逐图像评分和可检查的重建结果，既支持模型比较，也支持对可重建视觉信息的诊断。我们构建了包含10,000张生成图像的MMGenBench-Domain，以及包含1,284张图像、覆盖13种重叠视觉模式的MMGenBench-Test。对68个LMM的评估揭示了重建质量上存在总体分数可能掩盖的模式特定差异。在1,000张人工标注的……（摘要在此处被截断）

    arXiv:2411.14062v3 Announce Type: replace-cross  Abstract: Existing benchmarks often rely on predefined questions or reference captions, limiting their flexibility for evaluating detailed descriptions on new image collections. We present MMGenBench, an automated benchmark framework for evaluating Large Multimodal Models (LMMs) from raw images without per-image questions or reference captions. The pipeline generates descriptions, reconstructs images with a frozen text-to-image model, and compares the original and reconstructed images using a frozen image encoder. It produces reusable per-image scores and inspectable reconstructions, supporting both model comparison and diagnosis of reconstructible visual information. We construct MMGenBench-Domain with 10,000 generated images and MMGenBench-Test with 1,284 images spanning 13 overlapping visual patterns. Evaluation of 68 LMMs reveals pattern-specific differences in reconstruction quality that overall scores can obscure. On 1,000 human-la
    
[^186]: 低资源语言的语法错误纠正：以哲尔马语为例

    Grammatical Error Correction for Low-Resource Languages: The Case of Zarma

    [https://arxiv.org/abs/2410.15539](https://arxiv.org/abs/2410.15539)

    本文首次针对低资源语言哲尔马语开展语法错误纠正研究，比较了基于规则、机器翻译和大语言模型三种方法，发现基于M2M100的机器翻译方法表现最佳，并构建了超过25万样本的数据集用于评估。

    

    语法错误纠正（GEC）旨在提升文本的质量与可读性。以往针对该任务的研究主要集中在高资源语言上，而低资源语言缺乏强大的工具。为弥补这一不足，我们针对哲尔马语开展了一项语法错误纠正研究，这是一种在西非有超过五百万人使用的语言。我们比较了三种方法：基于规则的方法、机器翻译（MT）模型以及大语言模型（LLM）。我们使用一个包含超过25万个样本的数据集（包括合成数据和人工标注数据）对GEC模型进行了评估。结果表明，基于M2M100的机器翻译方法优于其他方法，在自动评估（AE）中检测率达95.82%、建议准确率达78.90%；在由母语者进行的人工评估（ME）中，语法和逻辑纠错的平均得分为3.0分（满分5.0分）。基于规则的方法对拼写错误有效，但在复杂语境下则难以奏效。

    arXiv:2410.15539v4 Announce Type: replace  Abstract: Grammatical error correction (GEC) aims to improve text quality and readability. Previous work on the task focused primarily on high-resource languages, while low-resource languages lack robust tools. To address this shortcoming, we present a study on GEC for Zarma, a language spoken by over five million people in West Africa. We compare three approaches: rule-based methods, machine translation (MT) models, and large language models (LLMs). We evaluated GEC models using a dataset of more than 250,000 examples, including synthetic and human-annotated data. Our results showed that the MT-based approach using M2M100 outperforms others, with a detection rate of 95.82% and a suggestion accuracy of 78.90% in automatic evaluations (AE) and an average score of 3.0 out of 5.0 in manual evaluation (ME) from native speakers for grammar and logical corrections. The rule-based method was effective for spelling errors but failed on complex context
    
[^187]: Feriji：法语-扎尔马语平行语料库、词汇表与翻译器

    Feriji: A French-Zarma Parallel Corpus, Glossary & Translator

    [https://arxiv.org/abs/2406.05888](https://arxiv.org/abs/2406.05888)

    本文发布了首个法语-扎尔马语平行语料库和词汇表Feriji，并通过对大型语言模型进行微调实现了BLEU分数达30.06的法语-扎尔马语机器翻译。

    

    机器翻译（MT）是一个快速发展的领域，近年来随着能够以极高精度翻译多种语言的模型的发展而取得了显著进步。然而，由于语言复杂性和资源有限，非洲语言在该领域的代表性仍有待提升。扎尔马语（Zarma）就是这样的情况，它是尼罗-撒哈拉语系桑海语族的一种方言，在尼日尔及周边国家有超过500万人使用。本文介绍了Feriji，这是首个面向机器翻译的稳健的法语-扎尔马语平行语料库和词汇表。该语料库包含61,085句扎尔马语句子和42,789句法语句子，以及包含4,062个单词的词汇表，是满足扎尔马语更多资源需求的重要一步。我们在该数据集上对三个大型语言模型进行了微调，表现最佳的模型获得了30.06的BLEU分数。

    arXiv:2406.05888v3 Announce Type: replace  Abstract: Machine translation (MT) is a rapidly expanding field that has experienced significant advancements in recent years with the development of models capable of translating multiple languages with remarkable accuracy. However, the representation of African languages in this field still needs to improve due to linguistic complexities and limited resources. This applies to the Zarma language, a dialect of Songhay (of the Nilo-Saharan language family) spoken by over 5 million people across Niger and neighboring countries \cite{lewis2016ethnologue}. This paper introduces Feriji, the first robust French-Zarma parallel corpus and glossary designed for MT. The corpus, containing 61,085 sentences in Zarma and 42,789 in French, and a glossary of 4,062 words represent a significant step in addressing the need for more resources for Zarma. We fine-tune three large language models on our dataset, obtaining a BLEU score of 30.06 on the best-performi
    

