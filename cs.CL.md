# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Behavioral Couplings: From Formation to Behavioral Emergence in Language Models](https://arxiv.org/abs/2610.23847) | 该论文提出“行为耦合”概念，发现当模型在同时描述两种行为的文档上训练后，针对其中一种行为进行微调会导致另一种行为也随之涌现，揭示了微调可引发超出预期目标之外的行为。 |
| [^2] | [AI-Driven Reproducibility and Analytical Robustness in Scientific Discovery](https://arxiv.org/abs/2610.23845) | 该研究推出SEER-Repro基准并开发AI智能体，成功复现了77.4%的基于SEER数据库的医学研究声明，同时通过智能体分析流水线自动化测试医学发现的分析稳健性。 |
| [^3] | [DiscoveryRSI: Recursively Learning to Discover Valuable Hypotheses](https://arxiv.org/abs/2610.23839) | 该论文提出 DiscoveryRSI 框架，通过多智能体进化搜索构建假设树，并借助假设信用优化、测试框架回放和再发现三种机制，实现对有价值科学假设发现过程的递归式持续学习。 |
| [^4] | [Beyond the Answer: Student-Conditioned Trace Construction for Reasoning Distillation](https://arxiv.org/abs/2610.23831) | 该论文提出ReDiST框架，以学生自身尝试为起点构建、修订并筛选与学生能力匹配的推理轨迹用于蒸馏，从而实现超越仅答案监督的学习效果。 |
| [^5] | [Acceptance Is Not Speed: Length Inflation in Training-Free Lossy Speculative Decoding](https://arxiv.org/abs/2610.23818) | 该研究揭示了有损推测解码中的“长度膨胀”现象——放宽验证虽然将每轮接受的token数量提升至1.34倍，但会通过延长推理轨迹使生成内容膨胀至1.54倍，从而大幅抵消表面上的加速收益，证明高接受率并不等同于真正的速度提升。 |
| [^6] | [Steering beyond Language: Improving Multilingual Training-Free Text Embeddings via Paired-Residual Subspace Removal](https://arxiv.org/abs/2610.23807) | 该论文首次系统研究了大语言模型的多语言免训练文本嵌入，发现跨语言语义等价句子存在嵌入错位问题，并提出PRSR方法，通过从多语言平行句构建配对残差、估计并移除语言特异性的低秩子空间，将模型表示引导至更语言不变的语义空间。 |
| [^7] | [DLite: History-free Speculative Decoding for Agentic AI](https://arxiv.org/abs/2610.23787) | DLite提出了一种面向智能体AI的无历史推测解码框架，通过直接复用目标模型的上下文隐藏状态进行局部推测预测，无需维护历史KV缓存，从而解决了现有草稿模型在长上下文、多轮交互场景下接受长度退化高达80%及系统开销过大的问题。 |
| [^8] | [VocalGrad: Evaluating Acoustic Perception in Audio Language Models](https://arxiv.org/abs/2610.23745) | 本文提出VocalGrad基准，通过将低级声学变化感知转化为时间变化方向的二元分类任务来评估音频语言模型的声学感知能力，并揭示了模型与人类标注者之间的显著感知差距。 |
| [^9] | [The Italian lawmaking process: a unified and structured dataset](https://arxiv.org/abs/2610.23741) | 本文通过经验证的基于大语言模型的提取流程，将非结构化的意大利议会档案转换为覆盖1985至2026年间5346部法律、共101413个事件的统一结构化立法程序数据集。 |
| [^10] | [Bridging the Language Gap in Scholarly Data II: Enhancing Gender Inference for Chinese Names](https://arxiv.org/abs/2610.23734) | 针对拼音中文姓名性别推断固有的模糊性问题，该研究开发了由11位中文母语者评分数据训练的汉字回归与分类性别推断模型，以提升科学计量学研究中对中文学者姓名的性别推断准确性。 |
| [^11] | [RedDarwin: Evolving Jailbreak Skills against Frontier LLMs](https://arxiv.org/abs/2610.23726) | RedDarwin 提出一个智能体框架，将越狱进化的基本单元从提示词转变为可执行的攻击技能，通过自主修改、组合与选择 47 个基础技能进化出 65 个更强技能，将平均攻击成功率从 18.8% 提升至 47.3%。 |
| [^12] | [Recognizing Study Lineage Is Not Enough: Over-Crediting Same-Trial Evidence in LLM-Based Medical Verification](https://arxiv.org/abs/2610.23721) | 大语言模型虽能在直接任务中正确识别临床试验的谱系关系，但在证据聚合时仍会过度采信同一试验的多篇发表文献而高估证据强度，而简单的分组指令可显著缓解这一问题。 |
| [^13] | [The Cognitive Cost of LLM Writing: A Reading-Time Comparison with Human Text](https://arxiv.org/abs/2610.23715) | 该研究提出了基于人类眼动数据训练的阅读时间预测模型RoBERTa-RT，并通过对496篇配对文章的分析发现LLM生成的文本与人工撰写文本在预测阅读时间上存在显著差异，揭示了LLM写作给读者带来的认知加工成本。 |
| [^14] | [GLOBE: Evolving Structured Agentic Memory for E-commerce Product Optimization in Generative Search](https://arxiv.org/abs/2610.23709) | GLOBE通过将电商产品优化知识不断积累并演化为结构化的按字段划分行为手册作为智能体记忆，使语言智能体能够在生成式搜索场景下实现持续的自我改进。 |
| [^15] | [VISTA: Steering LLMs with Latent User Values Inferred from a Few Survey Answers](https://arxiv.org/abs/2610.23702) | VISTA通过从少量调查答案推断用户的潜在价值观，并利用超网络将其转化为冻结大语言模型中的个性化引导干预，无需在提示词中放置用户画像即可实现与甚至超越上下文画像的个性化效果，且仅询问6%的问卷即可保留94%的完整信息增益。 |
| [^16] | [Beyond the Benchmark: Conditioning, Domains and Failure Modes of Qwen3-TTS](https://arxiv.org/abs/2610.23678) | 该研究系统评估了 Qwen3-TTS 两种语音克隆接口在多种语音领域中的表现与失效模式，发现语音领域对结果的影响远大于系统选择，两种接口各有优劣，且客观指标与人类听感评价并不一致。 |
| [^17] | [In Human Memories. Technology, Creativity and Self-Determination: An Ethical Proposal](https://arxiv.org/abs/2610.23651) | 本研究通过分析两部融合AI的音乐作品，揭示了AI在现代作曲中已从单纯的工具转变为会向作曲家提出问题与挑战的未知对话者，并据此探讨技术与创造力、自我决定之间的伦理关系。 |
| [^18] | [PresenceBench: An Exploratory Evaluation of Large Language Model Adaptation to User State](https://arxiv.org/abs/2610.23625) | 该论文提出PresenceBench基准，通过五臂因果设计与以正确性为门槛的人类配对比较，评估大语言模型能否利用用户功能状态（准备度、心流、稳定性）提升响应质量，且这种提升是否特异地依赖于正确的状态而非额外的结构化上下文。 |
| [^19] | [Trajectory-Triggered Sparse Collaborative Decoding](https://arxiv.org/abs/2610.23613) | 该论文提出利用草稿置信度路径的聚合轨迹形状特征（而非局部不确定性指标）来决定何时调用目标模型进行纠正的稀疏协作解码方法，将触发失败预测的AUC从接近随机的0.50-0.53提升至0.86，并揭示了触发时机上干预杠杆与证据强度之间的权衡。 |
| [^20] | [SIT-Eval: Situation-Based Interactive Turn-by-Turn Evaluation of Conversational Agents](https://arxiv.org/abs/2610.23612) | SIT-Eval提出了一个可调节的评估框架，将对话代理的期望行为形式化为“情境-规则”形式，并由LLM对话者在每轮对话中扮演随机采样的情境，从而实现对对话代理在多样化及罕见情境下的逐轮交互式评估，弥补了现有静态基准对罕见情境覆盖不足的安全缺口。 |
| [^21] | [Not Every Mention Is a Claim: Rethinking Object Hallucination Detection in Vision-Language Models](https://arxiv.org/abs/2610.23610) | 该论文指出并非字幕中的每个对象提及都是存在性断言，并提出NAME-Bench基准，对25,497个对象提及进行标注以将非断言性提及与真实幻觉区分开，从而避免夸大视觉语言模型的幻觉错误率。 |
| [^22] | [Enhancing RAG-based Medical VQA with Gated Multimodal Representations](https://arxiv.org/abs/2610.23588) | 提出了一种通过门控多模态单元学习联合查询表示的多模态RAG框架，使医学视觉问答中的证据检索能同时融合医学图像与临床问题两种模态的信息，且无需微调底层医学视觉-语言模型。 |
| [^23] | [Analyzing Occupational Gender Bias across the LLM Training Lifecycle](https://arxiv.org/abs/2610.23587) | 通过分析完全开源的OLMo 3模型在后训练各阶段生成的合成人物叙述，本研究发现后训练对职业性别偏见的影响并不对称——它显著提升了男性编码人物叙述的语义多样性，但对女性编码人物的提升在SFT后较小、在DPO和RLVR后不显著，表明后训练缓解性别偏见的效果有限。 |
| [^24] | [MARS: Memory-Aware Routing for Session-Level LLM Assistants](https://arxiv.org/abs/2610.23583) | MARS提出了一种记忆感知路由框架，将会话状态作为记忆构建与模型选择之间的共享控制接口，使LLM助手在每轮交互中能根据质量-记忆成本权衡同时获得合适的记忆视图和恰当的模型选择。 |
| [^25] | [Execution Before Memory: Diagnosing Memory Failures in Video World Models](https://arxiv.org/abs/2610.23542) | 该论文提出MIST评估框架将“执行指令变更”与“记忆变更结果”分离，揭示视频世界模型记忆测试低分的真正瓶颈是模型未执行指令变更而非遗忘，并据此提出M3-Agent，通过递归修正指令细节使模型成功完成变更。 |
| [^26] | [ConvexLLM: Scaling Convex-Hull Speech Interfaces to General-Purpose Audio Intelligence](https://arxiv.org/abs/2610.23539) | ConvexLLM通过C-Gate接口将音频表示为冻结文本嵌入的连续凸组合，在利用声学信息的同时更好地保留解码器的预训练文本能力，在VoiceBench和MMAU基准上取得了开源模型中最优的成绩。 |
| [^27] | [EduWild: How Teacher-Configured Chatbots Fail in Real-World Use and How to Fix Them](https://arxiv.org/abs/2610.23530) | 该研究通过包含1,726段对话和217个教师配置聊天机器人的EduWild数据集，揭示了此类机器人在真实使用中最常因未遵循教师意图而失败（如跳过工作流程步骤、遗漏期望设置、信息过载），并据此设计了一个基于提示词的干预框架来减少六种最常见失败模式。 |
| [^28] | [Truth in Contrast: Calibrated Factual Evidence from Language Model Likelihoods](https://arxiv.org/abs/2610.23527) | 该论文提出对比事实证据（CFE）及其概率形式对比事实后验（CFP），通过比较同一声明在事实与非事实条件下的似然差异，使共享误差相互抵消，从而在多语言事实陈述上显著提升置信度校准性能并保持较强的事实判别能力。 |
| [^29] | [Perceive, Solve, Reason: Progressive Capability Formation in Time Series-LLMs](https://arxiv.org/abs/2610.23521) | 该研究构建了首个覆盖感知、求解、推理三种渐进能力的时间序列-LLM训练评估套件，发现专用编码器不可或缺（4B模型感知能力全面超越GPT-4o），且按顺序渐进式学习三种能力优于数据混合训练，可匹敌使用三倍以上推理数据的纯推理训练。 |
| [^30] | [MuninAI: A Self-Hosted Agentic Framework for Independent Research Groups](https://arxiv.org/abs/2610.23480) | Munin是一个开源的自托管智能体框架，通过为本地部署的开放权重模型配备专门的阅读、搜索和计算工具，将其在领域文献问答上的准确率从0.387大幅提升至0.874，证明了智能体框架本身而非特定模型是系统效用的关键，同时保障了研究小组数据完全本地化。 |
| [^31] | [A low-dimensional dynamical basis for tongue shape control](https://arxiv.org/abs/2610.23456) | 该研究提出以整体舌形模式作为低维控制基础，并通过三维线性点吸引子的简单动力学模型成功拟合36名说话者的舌部超声波数据，为言语产生的低维发音控制提供了新证据。 |
| [^32] | [HydraLLM: Adaptive Width Slicing for Elastic Text and Vision-Language Models](https://arxiv.org/abs/2610.23455) | 提出HydraLLM弹性架构，通过激活注意力头数量动态调整模型各层宽度，使单一参数集能够适配从数据中心GPU到边缘设备的多种部署场景，并通过弹性投影模块将该能力扩展到视觉-语言多模态推理。 |
| [^33] | ["You" Is Not Toxic: How LLM Labelers Rewrite the Web's Conversation Laws](https://arxiv.org/abs/2610.23446) | 研究发现大语言模型标注器会把第二人称称呼（“你”）误判为毒性，从而制造出人类标注中不存在的虚假“对话法则”（如回复越深毒性越高）。 |
| [^34] | [ScoutRoute: Efficient Mixture-of-Experts Routing Using Internal Language Model Signals](https://arxiv.org/abs/2610.23443) | ScoutRoute 是一个轻量级后验路由框架，通过廉价的侦察模型单次前向传播提取隐状态与 logit 置信度等内部信号，结合域感知策略为混合专家系统高效选择专家，无需修改或联合训练底层语言模型。 |
| [^35] | [On the Tip of the Tongue: First-Token Cueing as a Behavioral Probe of Recall Failures in LLMs](https://arxiv.org/abs/2610.23400) | 研究发现大语言模型的许多闭卷错误实为“首词回忆失败”，提出的首词提示行为探针仅需预填充正确答案的首词，即可在16个开源模型上挽回五分之一到一半的多词答案错误，并提升七种语言的问答准确率。 |
| [^36] | [Sentence-Level Topic Modeling with Lexical and Contextual Evidence](https://arxiv.org/abs/2610.23388) | SentenceETM提出了一种句子级神经主题模型，通过联合利用词汇分布与句子嵌入上下文这两种证据来源，有效解决了主题建模中词级模型缺乏上下文、文档级模型忽视多主题共存问题的粒度困境。 |
| [^37] | [Language Models Learn to Improve Language Models](https://arxiv.org/abs/2610.23344) | 该论文提出LM2LM方法，通过强化学习训练大语言模型生成有效的语言模型改进提案，并利用廉价的代理反馈解决真实训练反馈成本高昂以及分布偏移下反馈不可靠这两大难题，为递归自我改进铺路。 |
| [^38] | [Robust Function Calling for LLM Agents Under Interface Drift](https://arxiv.org/abs/2610.23342) | 提出了一种无需训练的轻量级修复层 SchemaShield-Lite，通过模式校验、候选集重试和一次性受限修复，在接口漂移场景下显著恢复LLM智能体的函数调用准确率。 |
| [^39] | [Self-Reflective Chain-of-Thought (SRCoT): Evidence-Based Text Generation via a Closed-Loop Retrieve--Reflect--Revise Pipeline on Edge LLMs](https://arxiv.org/abs/2610.23311) | 该论文提出自反思思维链（SRCoT）架构，通过闭环“检索—反思—修订”流程和带来源标签的多层存储机制，使边缘设备上的大语言模型能够生成有证据支撑、可追溯的文本。 |
| [^40] | [Inferring L1 Backgrounds from L2 English with LLM Token Probabilities](https://arxiv.org/abs/2610.23309) | 提出一种基于大语言模型词元概率的母语识别方法，将文本视为指定候选母语的提示前缀的续写进行评分，显著优于直接分类方法，且 Llama 3.1 70B 在多个数据集上达到最先进性能。 |
| [^41] | [SimOPD: Relational Knowledge Transfer for On-Policy Distillation](https://arxiv.org/abs/2610.23306) | 提出SimOPD方法，通过成对记忆库将教师内部表示转化为响应相似性分布，并让学生在自身表示空间内进行匹配，从而实现无需跨模型投影的在线策略蒸馏关系知识迁移。 |
| [^42] | [Annotrace: A Workbench for Step-Level Annotation of Agent Traces](https://arxiv.org/abs/2610.23284) | Annotrace 是一个支持六种主流轨迹格式的智能体轨迹步骤级标注工作台，它将运行记录解析为有序嵌套的步骤，通过人机协作方式进行标注并衡量标注者间一致性，在多智能体问答语料上精确复现了此前人工标注的235个标注单元。 |
| [^43] | [Verbalising Model's Computation for Chain-of-Thought Faithfulness Detection](https://arxiv.org/abs/2610.23275) | VERITAS通过逐层雅可比投影将模型内部计算过程言语化，追踪推理锚点在各层间的产生、传递与显现过程，从而揭示推理偏离不忠实状态的机制，在思维链忠实性检测上显著超越LLM-as-Judge（平衡准确率提升28.4%）和激活探测方法（平均提升16.0%）。 |
| [^44] | [Reasoning Models Can Control Their Chains of Thought When They Tell Themselves To](https://arxiv.org/abs/2610.23257) | 推理模型并非天生无法控制其思维链，将约束作为自我指令置于CoT内部时，模型的可控性显著优于在用户提示中指定约束的方式。 |
| [^45] | [Sample, Mix, or Verify? Allocating Test-Time Compute under Imperfect Verifiers](https://arxiv.org/abs/2610.23255) | 该论文提出 SPLIT 方法，利用小型带标签试点实验和精确概率公式，在验证器不完美的情况下科学地分配测试时计算预算（采样数量、模型选择与验证开销）。 |
| [^46] | [Ignored, Not Refused: What Low Prompt-Injection Success Rates Hide in Small Tool-Using Agents](https://arxiv.org/abs/2610.23254) | 该研究审计了33,000条智能体轨迹，揭示低攻击成功率主要源于智能体默默忽略而非拒绝提示注入攻击，且仅改动系统提示词即可使ASR在0.2%至43%间剧烈波动，并据此提出攻击-任务结果矩阵（ATOM），强调评估应同时报告攻击失败方式和用户任务完成度。 |
| [^47] | [Who Pays for Efficient Inference? Confidence-Based Small-to-Large LLM Routing Across Tasks and Languages](https://arxiv.org/abs/2610.23249) | 该论文对基于置信度的小模型到大模型LLM路由进行了迄今最系统的跨语言、跨任务评估（72种配置、42,288条预测），证明置信度信号能够有效识别值得升级到大模型的困难样本，从而在多语言场景下实现质量与推理成本的更好权衡。 |
| [^48] | [Think More or Read More? A Token-Budget Theory of Reasoning with Shared Context](https://arxiv.org/abs/2610.23246) | 提出token预算理论形式化固定上下文窗口下“阅读他人输出”与“自主推理”的权衡，通过允许错误相关的Condorcet陪审团模型推导出多智能体辩论最优智能体数量的闭式解，且仅需单智能体测量即可估计最优群体规模。 |
| [^49] | [MGE-PO: Entropy-Aware Credit Assignment for Mathematical Reasoning](https://arxiv.org/abs/2610.23245) | MGE-PO提出了一种结合验证正确性与生成熵的四象限优势塑形方法，通过对低熵成功给予更大奖励、高熵失败给予更大惩罚来改进GRPO的信用分配机制，显著提升了数学推理性能。 |
| [^50] | [MultiAgent-OPD: End-to-End Training of Hierarchical Multi-Agent Systems with On-Policy Distillation](https://arxiv.org/abs/2610.23234) | 提出MultiAgent-OPD方法，通过将在线策略蒸馏与强化学习相结合，利用冻结的教师模型为搜索者提供局部过程监督，实现了层次化多智能体系统的端到端联合训练。 |
| [^51] | [ReLaDrift: Recurrent Language Generation via Drifting for Test-Time Scaling](https://arxiv.org/abs/2610.23225) | 提出ReLaDrift，利用共享漂移模型在连续潜在空间中对生成结果进行循环迭代细化，首次使单步漂移模型具备测试时计算量与生成质量之间的可调节权衡能力。 |
| [^52] | [MAS-Bai: Linguistically Informed Adaptive Multi-Agent Translation for Bowen-Script Bai](https://arxiv.org/abs/2610.23205) | MAS-Bai提出了一种融合语言学知识的自适应多智能体翻译框架，通过在轻量级检索-翻译路径与包含词源、句法智能体的文献学路径之间动态路由，并结合循环一致性可靠性信号，显著提升了白文白语-汉语的翻译质量并缓解了直译幻觉问题。 |
| [^53] | [AMEND: Constraint-Guided Multi-Agent Text Detoxification with Isolated Critics](https://arxiv.org/abs/2610.23183) | AMEND是一个推理时框架，通过角色条件化候选生成、隔离的多维评论员批评和约束优先选择机制来解耦文本解毒过程，其轻量版AMEND Lite仅在残留毒性时进行最多一轮修复，兼顾了去毒效果与计算效率。 |
| [^54] | [What LLMs Do in the Shadows: Reading Context Reliance from the Unembedding Dark Space](https://arxiv.org/abs/2610.23182) | 该论文提出DarkLens方法，通过将隐藏状态投影到与unembedding矩阵弱组合的“暗空间”并计算暗物质比率（DMR），无需注意力图、额外前向传播或辅助探针等开销，即可直接量化大语言模型的上下文依赖性，实现轻量级忠实性幻觉检测。 |
| [^55] | [Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning](https://arxiv.org/abs/2610.23181) | 本文提出TreeOPSD，一种无需评论家的树感知后验自蒸馏框架，通过将树结构回放批次转化为有限的后验自蒸馏目标，同时解决长时程智能体强化学习中的稀疏信用分配与策略陈旧问题。 |
| [^56] | [Do Speech BCIs Need Larger Models? Rethinking Neural Decoding beyond Scaling](https://arxiv.org/abs/2610.23175) | 本文提出两阶段框架BEST，通过分层中间表示和紧凑的语音导向解码器直接将神经活动映射为文本，证明在数据稀缺的语音脑机接口任务中，无需依赖大型模型即可实现高效、低成本且泛化性强的神经解码。 |
| [^57] | [Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making](https://arxiv.org/abs/2610.23170) | Chinese Decision 是一系列中文语言模型，通过单次前向传播直接对候选选项的 logits 进行归一化得到概率分布，无需自回归生成，在医学、法律、金融等领域的决策任务上实现了高准确率与良好校准。 |
| [^58] | [Explaining Chess and Go in Natural Language](https://arxiv.org/abs/2610.23166) | 本文提出了基于引擎评估的解释性指标和无需训练的BinProp方法，使语言模型能够发现可复用的棋类概念并以自然语言解释着法，平均得分相比直接提示提高近两倍，且该方法可泛化至围棋、跳棋和将棋等多种棋类。 |
| [^59] | [Cross-Centaur: Modeling Individual Decision-Making Across Psychological Tasks with Language Models](https://arxiv.org/abs/2610.23157) | 本文提出包含591名参与者跨12个心理学任务关联行为数据的Cross-Psych-101数据集，并训练Cross-Centaur模型，首次建模跨任务的个体层面决策规律，在任务内与跨任务预测上均超越现成大语言模型和Centaur。 |
| [^60] | [Can AI interview? Comparing AI-led to Human-led Chat Interviews](https://arxiv.org/abs/2610.23149) | 本文首次通过随机对照实验直接比较了AI主导与人类主导的聊天访谈数据质量，发现AI访谈者在访谈时长相当的情况下提问数量是人类的两倍多，并能引出多出约65%的受访者文本。 |
| [^61] | [LED: Improving Masked Diffusion Language Models by Lookahead Editing](https://arxiv.org/abs/2610.23122) | 提出前瞻编辑方法（LED），通过在掩码扩散解码中引入可重写的提议状态，使掩码扩散语言模型在不牺牲其易优化性的前提下获得迭代修正的表达能力。 |
| [^62] | [Disentangling Language Effects on Tool Admission and Post-Retrieval Safety](https://arxiv.org/abs/2610.23093) | 该研究通过独立控制请求、检索页面和回答语言的受控实验，首次识别出“跨语言处理中的安全损失”（SLIP）现象——非英语有害请求使工具准入率从46.9%升至67.4%，且要求以页面语言回答会进一步增加危害性，从而将语言对LLM智能体工具准入与检索后安全的影响分离出来。 |
| [^63] | [Before Retrieval Begins: Safety and Tool-Use Decisions in LLM Agents](https://arxiv.org/abs/2610.23071) | 该论文首次系统研究了大语言模型智能体在工具调用发生之前“是否调用工具”这一前置安全决策，发现有害请求被接受后的平均有害率高达74.5%，并通过因子化干预揭示“请求操作”组件是支配该决策的关键因素，同时发现书面拒绝在行为与表征上与该决策相互耦合。 |
| [^64] | [ParetoChat: An LLM-Based Interface for Interactive Multi-Objective Optimization](https://arxiv.org/abs/2610.23062) | 本文提出ParetoChat，一个基于大语言模型的多智能体交互界面，能够将决策者的自然语言偏好自动转化为多目标优化的标量化参数，并通过迭代反馈循环实现交互式多目标优化。 |
| [^65] | [Same Person, Different Advice: Auditing LLM Financial Advice with Repeated Risk-Tolerance Responses](https://arxiv.org/abs/2610.23060) | 研究对七个主流大语言模型的金融建议进行审计，发现当客户风险承受度回答发生现实水平的变动时，模型推荐的股票配置比例会大幅波动（10-35个百分点），且其敏感度与被告知回答非常可靠时相当，揭示了LLM金融建议面对人们不一致风险自评时的稳健性缺陷。 |
| [^66] | [ECHO: Replacing the Compliance Proxy with Response Harmfulness in Optimization-Based Jailbreak Attack](https://arxiv.org/abs/2610.23030) | ECHO提出用基于回复实际危害性的连续评分（由1.7B的HarmSig回归模型提供）替代传统的合规性代理指标，并结合词级扰动算子与关键词锚定束搜索，在严格黑盒条件下显著提升了针对大语言模型的越狱攻击效果。 |
| [^67] | [Controlling the Diversity-Quality Trade-Off of LLMs by Merging Models Before and After Alignment](https://arxiv.org/abs/2610.22999) | 提出一种无需额外训练的深度感知 FrankenMerging 方法，通过将对齐模型的低层与基础模型的高层进行合并，可灵活控制大语言模型输出中词汇与语义的多样性及质量之间的权衡。 |
| [^68] | [Prism Attention: Proposal-Refined Index Sharing Mechanism for Efficient LLMs Inference](https://arxiv.org/abs/2610.22985) | 提出Prism Attention机制，利用层间显著令牌偏移高度局部化的特性，以“提案-精细化”的方式在层间共享并修正稀疏注意力索引，在避免静态索引复用导致注意力损失的同时，大幅提升大语言模型长上下文推理效率。 |
| [^69] | [Self-Evolving Agents Can Repair Failures but Struggle to Generalize from Them](https://arxiv.org/abs/2610.22984) | 研究发现，仅从失败经验中学习的自我进化智能体能够修复之前失败的任务，但这种经验难以泛化到未见过的任务上，且智能体常常误解失败原因。 |
| [^70] | [KiranaBench: Can a Language Model Run a Neighbourhood Shop?](https://arxiv.org/abs/2610.22980) | 论文提出多智能体印度杂货店模拟基准KiranaBench，评估发现四个开源LLM的经营利润均远低于固定规则店主基线，最佳模型仅获得其八分之一的利润。 |
| [^71] | [What Shapes Brain Alignment of Large Language Models during Dialogue? Insights from Language Production and Comprehension](https://arxiv.org/abs/2610.22961) | 本研究利用对话中的fMRI数据揭示了大型语言模型在语言产生与语言理解两种模式下呈现截然不同的大脑对齐模式，且对齐程度与模型表征的内在维度以及可解释的语言和交际特征密切相关。 |
| [^72] | [SAFE-Mem: Secure Agentic Memory with Fine-Grained Authority and Governance-Aware Retrieval](https://arxiv.org/abs/2610.22959) | 提出SAFE-Mem框架，通过为记忆声明标注细粒度治理标签并结合治理感知检索，在保障LLM智能体记忆安全性的同时保留合法的记忆使用。 |
| [^73] | [Recognizing Atari and Assessing Sente: Controlled Evaluation Tasks for Language-Based Go Agents](https://arxiv.org/abs/2610.22953) | 该论文构建了以围棋“叫吃”及其与“先手”关系为核心的受控评估集，发现经术语和棋评微调的围棋语言模型LoGos难以识别叫吃关系变化并评估其战略含义，为检验语言型围棋智能体能否将领域术语与精确棋盘关系关联起来提供了评估框架。 |
| [^74] | [Aligning Language Models Towards Counterfactual Invariance](https://arxiv.org/abs/2610.22942) | 本文提出一种基于群体级奖励函数的策略梯度技术，使语言模型在对齐过程中实现反事实不变性，解决标准强化学习对齐方法在提升整体回复质量的同时放大跨人口群体反事实偏差的问题。 |
| [^75] | [Beyond Surface Cues: Robust LLM-Generated Korean Text Detection](https://arxiv.org/abs/2610.22940) | 提出偏差受控的韩文对抗性基准KoLing与基于理论语言学、在多个语言学层面建模韩语特性的可解释检测器KoLingProbe，在LLM生成与修改韩文文本检测任务上大幅超越强基线。 |
| [^76] | [Many Biases, One Circuit: Post-Training Tunes a Pretrained Cue-Routing Mechanism](https://arxiv.org/abs/2610.22937) | 多种偏见（如谄媚性）共享预训练模型中已存在的同一注意力头回路——助手微调在聊天格式中激活它，强化学习进一步加强它，表明跟随线索源于预训练与调优的副作用，而非对齐本身的问题。 |
| [^77] | [ASR-Style Transcript Normalization Removes Useful Pragmatic Signal for Dialogue Act Classification](https://arxiv.org/abs/2610.22934) | 研究通过对六种分类器和三个会话语料库的系统性实验证明，ASR流水线中常见的转录规范化操作（如移除填充词和标点符号）会持续降低对话行为分类性能，表明这些语言特征包含重要的语用信号。 |
| [^78] | [Sep-Latent: Separator Tokens as Natural Compression Anchors for Implicit Reasoning](https://arxiv.org/abs/2610.22924) | 本文提出Sep-Latent框架，将分隔符标记作为语义压缩锚点，结合注意力衰减机制与多层级蒸馏实现动态深度的隐式推理，从而在保持逻辑一致性的同时显著降低显式思维链的计算开销。 |
| [^79] | [Where's the Indirect Evidence? Investigating Grammatical Generalization with Influence Functions](https://arxiv.org/abs/2610.22919) | 本文提出一种基于影响函数的自底向上方法来识别语言模型学习语法现象所依赖的间接证据，并发现移除这些间接证据会在五个模型的15种BLiMP语法现象上显著降低模型的泛化准确率。 |
| [^80] | [FederatedBytes: Tokenizer-Free, Byte-Level Small Language Models for Efficient Multimodal Federated Learning](https://arxiv.org/abs/2610.22906) | 提出FederatedBytes——一种无分词器、将所有输入视为字节序列并采用线性注意力机制的小型语言模型，显著降低了NLP与多模态分析任务在联邦学习中的计算和通信开销。 |
| [^81] | [Competence-Based Evaluation of Logical Reasoning in Large Language Models](https://arxiv.org/abs/2610.22894) | 本文提出“基于能力的评估”（CBE）新范式，通过参数化生成器调节复杂度并使用保留答案的多种表述形式，来衡量大语言模型逻辑推理的泛化能力，而非仅依赖单一准确率指标。 |
| [^82] | [Bengali Hallucination Detection with Retrieval and Programs: A System Study](https://arxiv.org/abs/2610.22892) | 该论文提出了一个结合查询重写、混合检索和语言模型评判的仅推理孟加拉语幻觉检测系统，在Kaggle竞赛中取得了0.956的macro-F1分数。 |
| [^83] | [NLData2Opt: Benchmarking Parameter Derivation and Prediction for Modeling Optimization Problems from Natural Language and Data](https://arxiv.org/abs/2610.22889) | 该论文提出了 NLData2Opt 基准，要求大语言模型从含噪声的真实数据中完成优化问题参数的提取、推导计算和未来值预测，弥补了现有基准依赖理想化输入而高估模型实际能力的不足。 |
| [^84] | [Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?](https://arxiv.org/abs/2610.22887) | 提出并发约束强化（CCR），将多个可独立验证的问题放入同一个受限完成中，使RLVR在8,000-token的短窗口下仍能保留正确性信号，并将每训练步的token消耗从约97.5万降至约17.5万，大幅降低训练成本。 |
| [^85] | [Cooperation Without Adaptation: LLM Agents in a Collective-Risk Dilemma](https://arxiv.org/abs/2610.22882) | 研究发现大语言模型智能体在集体风险困境博弈中采用固定而非适应性的合作策略，即使在零风险且贡献无法改善自身收益的情况下仍会贡献资源，表明其合作行为源于策略惯性而非对环境风险的理性适应。 |
| [^86] | [Identify, Classify, Act: Per-Instruction Controlled Compliance for LLMs over External Data](https://arxiv.org/abs/2610.22874) | 提出EDIC分解式奖励框架，通过对外部数据中每条嵌入指令进行识别-分类-行动决策，并利用基于场景的指令注入和SFT加GRPO训练的生成式评判者实现指令级功劳归因，使LLM能够遵循合法指令的同时拒绝对抗性提示注入。 |
| [^87] | [Uncertainty-Aware Budget Allocation for Efficient LLM Tree Search](https://arxiv.org/abs/2610.22858) | 该论文提出无需训练的不确定性感知分配框架（UAA），利用兄弟动作的语义分散度在扩展前预测下游后续内容的冗余性，自适应地为LLM树搜索的每一步分配预算，从而大幅提升多步推理的效率。 |
| [^88] | [The Reliance Hypothesis: Chain-of-Thought Faithfulness Rises as Language Resource Level Falls](https://arxiv.org/abs/2610.22841) | 论文提出“依赖假设”：当模型无法回忆答案时会更加依赖思维链推理，因此CoT忠实度在知识密集型任务上随语言资源水平的下降而上升。 |
| [^89] | [Too Emotional to Be Fake? Intra-Utterance Emotion Shifts Expose Failure Modes in Speech Deepfake Detectors](https://arxiv.org/abs/2610.22832) | 提出EmoShift基准，揭示语句内情绪变化会显著降低语音深度伪造检测的可靠性——情绪变化的合成语音漏检率最高上升25.3个百分点，而情绪变化的真实语音误报率最高上升24.4个百分点。 |
| [^90] | [Causal Edit Serialization](https://arxiv.org/abs/2610.22820) | 提出因果编辑序列化（CES）框架，通过指针词元、轻量级定位头和双索引旋转位置编码，使自回归语言模型能够在保持标准因果注意力机制的前提下，对文本进行插入、删除、替换、复制和移动等显式编辑操作。 |
| [^91] | [Watermarkability: Measuring and Improving Language-Model Compatibility with Logit-Based Watermarks](https://arxiv.org/abs/2610.22810) | 该论文提出了“水印可植入性”概念及其KL散度代价度量方法，用以量化语言模型与基于logit水印之间的兼容性，并设计了降低水印植入代价的两阶段方法。 |
| [^92] | [On the Position Bias of Large Language Models: A Comprehensive Study](https://arxiv.org/abs/2610.22807) | 该研究全面考察了从十亿到万亿级参数规模、涵盖五种语言的大语言模型位置偏差问题，发现增大模型规模和使用思维链提示可以显著缓解这一偏差。 |
| [^93] | [Veil-Then-Reveal: Protecting PII Privacy via Reversible Pseudo-Anonymization in Zero-Trust User--LLM Interaction](https://arxiv.org/abs/2610.22800) | 提出用户侧可逆伪匿名化框架Veil-then-Reveal（VtR），在零信任威胁模型下将提示词中的PII替换为同类型的真实替身词以保护隐私，同时保留语言与结构特性使LLM仍能有效处理，明文信息始终保留在用户端。 |
| [^94] | [Reasoning Reranking with Semantic IDs as the Sole Item Representation](https://arxiv.org/abs/2610.22797) | 本文构建了ReRankID受控测试平台，首次系统评估仅以语义ID作为物品表示进行推理重排序的可行性，发现此类模型常常直接复制输入顺序而非真正重排序，且引入可读元数据也无法修复这一失败。 |
| [^95] | [Causal Latent Workspaces for Looped Transformers](https://arxiv.org/abs/2610.22796) | 该论文提出因果潜在工作空间Transformer，通过在循环共享深度架构中引入门控潜在槽位，使各位置能够交换因果前缀的紧凑摘要，在多个语料库和匹配参数预算下实现了优于标准Transformer的困惑度表现。 |
| [^96] | [Prune the Leaky Neurons: Mitigating Private Data Leakage in Fine-Tuned LLMs via Joint-Embedding-Guided Pruning](https://arxiv.org/abs/2610.22792) | 提出 JEPA-Prune 框架，利用在无个人身份信息的替代视图上通过联合嵌入预测架构训练的公开模型来引导事后神经元剪枝，在保持模型性能的同时有效缓解微调大语言模型中的隐私数据泄露问题。 |
| [^97] | [Self-Evolving Coding Agents via Harness-Policy Optimization](https://arxiv.org/abs/2610.22791) | 提出 Self-HPO 框架，通过交替优化智能体框架与策略模型，让进化的策略自己提出框架改进方案，从而实现编码智能体的递归自我进化。 |
| [^98] | [Reading the News Without Hindsight: NoLBERT Topics and the Cross-Section of Stock Returns](https://arxiv.org/abs/2610.22745) | 该论文提出仅使用1976-1995年历史文本预训练的NoLBERT编码器来构建无后见之明偏差的新闻主题，并发现将注意力冲击衡量窗口延长至63个交易日可显著提升基于新闻的股票截面收益预测效果。 |
| [^99] | [Thunder-DeID+: Boundary-Aware Korean De-identification from Court Judgments to Broader Sensitive Documents](https://arxiv.org/abs/2610.22733) | 提出Thunder-DeID+框架，通过结合精炼的三级PII分类体系与融合N-gram和边界评分组件的边界感知分词器，解决了韩语黏着语结构下BPE分词与PII跨度边界不对齐的问题，并将法院判决书的去标识化能力扩展至更广泛的敏感文档。 |
| [^100] | [TokenEconomizer: Experience-Driven Black-Box Cost Control for Coding Agents](https://arxiv.org/abs/2610.22732) | TokenEconomizer是一个经验驱动的外部黑盒控制器，通过注入操作微技能、剪除无关工具和干预失败循环，显著降低编程代理在连续现实任务序列中的环境相关token成本。 |
| [^101] | [A misinformation playbook sharpens judgement without breeding cynicism](https://arxiv.org/abs/2610.22724) | 本研究从51种语言的21万多条事实核查声明中提炼出虚假信息的说服策略手册，学习该手册可提升人类和AI对陌生信息的辨别能力，且不会让人和机器对真实信息产生过度怀疑。 |
| [^102] | [CoLU: Confidence-Weighted Layerwise Unlearning for Large Language Models](https://arxiv.org/abs/2610.22721) | 该论文提出CoLU方法，通过置信度加权的遗忘与保留正则化仅更新单个选定的Transformer层来实现大语言模型遗忘，并配合基于归一化遗忘-保留分数的简短层筛选策略高效选取最优层，在降低计算成本的同时更好地平衡了遗忘效果与知识保留。 |
| [^103] | [ARBOR: Locating Reasoning Failures in Long Chains of Thought through Reasoning State Tracking](https://arxiv.org/abs/2610.22717) | ARBOR提出了一种推理失败归因框架，通过将推理轨迹表示为推理状态树并从全局视角追踪步骤与前序状态之间的关系，从而精确定位长思维链中的推理失败，并将已知失败模式细化为机制不同的子类型。 |
| [^104] | [Risk and Refusal: Mechanisms of LLM Responses to Suicidal Ideation](https://arxiv.org/abs/2610.22715) | 该研究揭示了大语言模型拒答自杀意念披露本身可能构成安全失效，其风险主要源于对披露内容的忽视，且拒答机制与拒绝有害请求共享相同的内部表征方向，而通过告知模型上下文或直接抑制拒答可在不增加风险的前提下提升模型回应参与度。 |
| [^105] | [First Passage Attention](https://arxiv.org/abs/2610.22714) | 该论文提出首次通过注意力（FPA），将注意力衰减机制从路径求和转向路径极值——根据学习标量场在中间跨度上的最深下凹来惩罚键的logit，从而能在保留较旧结构相关token的同时选择性擦除中间上下文，并证明常数维度的FPA头即可实现任意存活键子集，而传统softmax头或token级遗忘门需要与键数呈线性关系的维度。 |
| [^106] | [GlobeBench: Evaluating Text-Based World Model Capabilities for Scalable Environment Simulations](https://arxiv.org/abs/2610.22708) | GlobeBench 是首个用于评估基于文本的世界模型能力的标准化基准，包含跨五个领域和十九项模拟技能的 1,632 个任务，并采用确定性门控与语义裁判相结合的混合评估架构。 |
| [^107] | [Explore Separately, Adopt Selectively: Execution-Guided Tool Use over Large Tool Libraries](https://arxiv.org/abs/2610.22688) | 提出BRIA智能体，通过将候选工具调查隔离到独立分支、检查执行证据并仅将有用的结果采纳回主轨迹，解决了大规模工具库中多候选工具选择时失败尝试污染推理轨迹的问题。 |
| [^108] | [Multimodal Filesystem Memory For Long-Horizon Embodied Agents](https://arxiv.org/abs/2610.22686) | 提出多模态文件系统记忆架构MFM，通过情景、空间、语义、程序和工作记忆等互补记忆类型以及可搜索的文本记录，帮助长时程具身智能体有效存储并检索多模态经验。 |
| [^109] | [Not All Graphs Are Maps: A Task-Agnostic Exploration Framework for Mobile GUI Navigation](https://arxiv.org/abs/2610.22672) | TAG-Nav是一个任务无关的图导航框架，通过将离线探索提炼为可复用的应用功能、目标页面和路线记忆，使GUI智能体无需任务特定演示即可完成目的地导航。 |
| [^110] | [ProPerAgent: Toward Memory-Driven Proactive Assistance](https://arxiv.org/abs/2610.22671) | 该论文提出ProPerMem基准，首次要求AI助手从长达一年、充满干扰的个人记忆中自主挖掘证据来决定是否及何时主动介入，发现现有模型（含前沿模型）表现极差（最高仅15%），而提供现成证据时性能可达96%，揭示了记忆证据挖掘是主动性助手的核心瓶颈。 |
| [^111] | [Exploring The Empirical Privacy Landscape of Synthetic Data](https://arxiv.org/abs/2610.22659) | 本研究通过对合成医疗记录的受控比较发现，从敏感记录生成合成数据（尤其是结合隐私感知改写）比直接微调或 DP-SGD 训练能更好地平衡语言模型的能力与隐私保护。 |
| [^112] | [Secondhand Stress: How Stressful Disclosures Affect Psychosocial Wellbeing in Online Graduate Admissions Communities](https://arxiv.org/abs/2610.22657) | 基于压力交叉理论，本研究通过大规模分析Reddit研究生申请社群中的帖子与评论发现，与他人的压力表露进行互动的用户，后续会表现出更高的焦虑、抑郁和压力表达水平以及更活跃的参与行为。 |
| [^113] | [Evidence-Gated Bilingual RAG for Reliable Consumer Banking Search](https://arxiv.org/abs/2610.22652) | 本文提出一种完全本地化部署的证据门控双语RAG框架，通过结构化内容精炼、混合检索、意图感知证据选择、重排序和三分类决策机制，将消费银行搜索的Top-1命中率从50.48%提升至79.05%，并有效支持阿拉伯语与英语的双语查询。 |
| [^114] | [SAFETYEDIT: Sparse Internal Targets with Prompt-Wide Control for LLM Jailbreak Backdoors](https://arxiv.org/abs/2610.22640) | 提出SAFETYEDIT，以稀疏的拒绝相关MLP坐标作为内部目标并实现提示级控制，在更小的内部编辑目标下获得比稠密表示方向更优的后门有效性-保持性权衡。 |
| [^115] | [Inductive Learning of Concept Embeddings from Cross-Linguistic Colexification Patterns with Textual Attributes](https://arxiv.org/abs/2610.22638) | 提出了一种结合文本属性的归纳式框架，从跨语言共词化模式中学习概念嵌入，克服了以往直推式嵌入方法无法为训练数据之外的新概念生成嵌入的可扩展性局限。 |
| [^116] | [Measuring Mechanistic Uncertainty in Large Language Models via Causal Robustness](https://arxiv.org/abs/2610.22636) | 该论文提出“机制不确定性”概念及“因果鲁棒性裕度”指标，通过测量改变模型输出所需的最小内部干预量来量化大语言模型的不确定性，发现错误或不确定的响应裕度更小，且该指标与传统基于输出的不确定性方法形成有效互补。 |
| [^117] | [ToMStories: Interpreting Theory-of-Mind in Synthetic Narratives](https://arxiv.org/abs/2610.22632) | 该论文提出了ToMStories评估数据集（约6,000个合成叙事短篇，考察情绪、恐惧、动机、情感创伤四类心理状态），发现最大的语言模型在多选题上可媲美人类，但在开放式问题上仍有差距，且模型在隐式心理状态的解释与生成上存在简化现象。 |
| [^118] | [A Plausible Explanation Is Enough to Mislead Multimodal Judges](https://arxiv.org/abs/2610.22631) | 提出JUPE基准并揭示“合理性俘获”现象——仅仅貌似合理的普通解释就足以误导多模态大模型评审接受错误答案，其影响与刻意说服性论证相当。 |
| [^119] | [Trajectory-Guided Steering: Mitigating Spurious Feature Bias in Clinical Classification](https://arxiv.org/abs/2610.22618) | 提出轨迹引导转向（TGS）方法，仅凭模型内部预测轨迹即可判断是否修正预测及修正方向，无需反事实数据或伪特征知识，在16个临床分类任务上修复了17.7%的错误且仅破坏1.3%的正确预测，并在所有超参数设置下保持净正向收益。 |
| [^120] | [When Does Uncertainty Measure RAG Hallucination Risk?](https://arxiv.org/abs/2610.22615) | 论文提出并验证了一个条件性观点——不确定性只有在检索证据足以约束答案时才能有效衡量RAG的幻觉风险，实验表明困惑度在针对性无依据编辑检测中达到0.81–0.91 AUROC，且结合语义漂移评分可进一步提升校准效果。 |
| [^121] | [Can LLMs predict the outcomes of machine learning experiments?](https://arxiv.org/abs/2610.22599) | 该论文构建了包含288个LLM自生成实验想法的数据集，通过让五个前沿大语言模型在机器学习实验训练前预测其影响力并与实际结果对比，衡量了这些模型判断实验是否值得运行的直觉准确性。 |
| [^122] | [Broad and Efficient PII Detection with Unified Labels](https://arxiv.org/abs/2610.22597) | 该论文通过对77个公开数据集在统一标签体系下重新标注，训练出一个141M参数的高效编码器检测器，实现了对12类结构化PII在低误报率下的广泛且高效的检测。 |
| [^123] | [The Model Knows Better: Defending Medical Multi-Agent Systems Against Prompt Injection](https://arxiv.org/abs/2610.22594) | 该论文提出一种无需训练的防御方法，通过将患者检查发现与临床判断分离，并验证系统决策背后的临床判断是否与患者实际情况相符，从而有效防御医疗多智能体系统中的提示注入攻击。 |
| [^124] | [SAG2: Embedding-Free Multi-Hop Retrieval with BM25 and JEV](https://arxiv.org/abs/2610.22588) | SAG2提出了一种无需嵌入向量的多跳检索架构，仅依靠单个SQLite数据库、BM25词法索引和JEV上下文证据决策，在查询时完成语义证据选择，同时实现了轻量级部署准备和多个多跳问答基准上最高的检索召回率。 |
| [^125] | [Mitigating Demographic Disparities in Explanation-Human Alignment via Rationale Regularization](https://arxiv.org/abs/2610.22586) | 本文提出Plausibility-EFG指标来衡量仇恨言论检测模型在不同人口群体间解释合理性的差异，并证明采用理由正则化的对抗训练模型ADV-RoBERTa能在提升分类性能和解释与人类对齐的同时减少群体间的解释差异。 |
| [^126] | [Repairing LLM-Generated Specifications with Deterministic Diagnosis](https://arxiv.org/abs/2610.22568) | 提出DRAGO修复框架，用确定性的、针对特定形式化方法的诊断取代LLM无根据的批评，且仅在诊断错误权重严格减少时才保留重写，从而有效修复LLM生成的可运行但语义错误的形式化规约。 |
| [^127] | [Who Do Perspectivist Models Predict Well? Annotator Agreement, Sparsity, and Performance](https://arxiv.org/abs/2610.22567) | 该研究通过构建迄今最大的序数标注视角主义NLP数据集集合，发现算法对倾向于与他人一致的标注者预测效果更好，而标注者标注量的多少与算法性能并无显著正相关。 |
| [^128] | [Cultural (In)Competence: Investigating Culture-Specific Stereotype Detection Capabilities of LLMs](https://arxiv.org/abs/2610.22545) | 本文系统研究了提示语言、指令微调语言和微调数据集规模这三个因素对大语言模型在美国、荷兰和中国等不同文化语境下检测文化特定刻板印象能力的影响，揭示了模型文化胜任力方面的差异。 |
| [^129] | [HeRE: A Novel Benchmark for Hebrew Relation Extraction via LLM-Guided Denoising of Knowledge Graph Alignments](https://arxiv.org/abs/2610.22542) | 本文提出HeRE——首个基于LLM引导去噪知识图谱对齐构建的希伯来语关系抽取基准，提供了一种可扩展、语言无关的低资源语言关系抽取数据集创建方法。 |
| [^130] | [Are Large Reasoning Models Good Everyday Reasoners? A Comparison with Non-Reasoning Models on Logical Reasoning Tasks](https://arxiv.org/abs/2610.22538) | 该研究构建了一个基于语言学与语用推理现象的日常逻辑推理基准，发现大型推理模型在偏离形式逻辑规则的日常推理中表现退化，而某些采用思维链提示的非推理模型在日常推理上甚至优于推理模型。 |
| [^131] | [Pearl-Chat: Engineering a native Luganda language model from first principles in JAX and Flax NNX](https://arxiv.org/abs/2610.22529) | 该论文提出了Pearl-Chat——一个用JAX和Flax NNX从第一性原理构建的约850万参数卢干达语原生因果语言模型，通过从头训练的字节级BPE分词器解决了通用多语言分词器在班图黏着语言上子词切分率过高的问题。 |
| [^132] | [Literal Lock-In: How Test Language Shifts Literal Attraction in Typed Decision Models on Bengali Idioms](https://arxiv.org/abs/2610.22526) | 该论文发现，将字面义释义混入选项列表会强烈吸引各类非生成式决策模型（占其剩余孟加拉语错误的65%至96%），且这种“字面锁定”效应在孟加拉语中比英语强1.9至6.4倍，表明此类失败主要取决于测试语言而非模型能力。 |
| [^133] | [alkhder at AlexandriaX-2026: Context-Aware English-to-Dialectal Arabic Dialogue Translation with QLoRA-Fine-Tuned NileChat-3B](https://arxiv.org/abs/2610.22525) | Alkhder 团队通过 QLoRA 微调 NileChat-3B-Base，以国家、方言、领域、参与者角色、性别方向及对话上下文等多重条件为输入，仅用一组适配器权重实现了英译十三种阿拉伯方言的上下文感知对话翻译，并以极低的训练成本在 AlexandriaX-2026 上取得 spBLEU 20.36 的成绩。 |
| [^134] | [Learning User Representations to Improve Simulation Fidelity](https://arxiv.org/abs/2610.22518) | 该论文将用户表示识别为用户模拟的关键瓶颈，提出通过对话重建任务与事后反馈来评估和优化自然语言形式的用户表示，从而提升用户模拟器复现真实用户行为的保真度。 |
| [^135] | [alkhder at ImageEval 2026 Shared Tasks: Constrained Selection and Rotation-Averaged Logit Scoring for Multimodal Hallucination Detection](https://arxiv.org/abs/2610.22515) | 该方法在零样本设置下利用7B视觉-语言模型，通过约束选择、下一词元对数概率评分和拉丁方轮转消除答案位置偏差，在ImageEval 2026多模态幻觉检测任务的英语和阿拉伯语测试集上均将CI错误率相比官方3B基线降低超过80%。 |
| [^136] | [AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems](https://arxiv.org/abs/2610.22511) | AuditKIT是一个统一的评估框架，将基准测试、评判评估、运行时性能、工具使用和检索质量整合到单一结果表示中，解决了大语言模型与智能体系统评估中接口和指标语义不一致的问题。 |
| [^137] | [RealCLI: Toward a Unified Evaluation of Human-Agent Collaboration for CLI Agents](https://arxiv.org/abs/2610.22507) | REALCLI是一个统一评估CLI智能体任务执行与人机协作质量的配对基准，其通过包含22个子维度和51个交互场景模板的五维框架，以及完整规格说明（REALCLI-EXEC）与脚本化用户渐进交互（REALCLI-COLLAB）的双模式设计，揭示了前沿智能体在协作能力上的显著差距。 |
| [^138] | [From Parallels to Divergence: Language Models' Learning of Action Nominal Constructions](https://arxiv.org/abs/2610.22468) | 本研究通过构建96种具有不同语法配置的合成语言并从零训练GPT-2模型，发现语言模型的学习偏好与自然语言类型学分布在基本从句配置上平行，但在补语系统和动作名词构式策略上出现分化。 |
| [^139] | [Progressive Misalignment Can Emerge from Benign Recursive Self-Distillation](https://arxiv.org/abs/2610.22467) | 在递归自蒸馏过程中，即使只使用良性提示和模型自身的安全生成内容，安全模型也会随着迭代轮次不断累积失准，最终在多种有害查询上表现出渐进式失准现象。 |
| [^140] | [High prevalence of antagonizing rhetoric in trending topics on Twitter/X](https://arxiv.org/abs/2610.22455) | 本研究提出了一种能够区分并检测文本中“敌对化”与“批评”两种政治言论及其目标的计算方法，并揭示了Twitter/X热门话题中敌对化言论的高度盛行。 |
| [^141] | [Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss](https://arxiv.org/abs/2610.22454) | 研究发现MoE模型在预填充阶段削减路由专家所造成的短答案F1损失，主要集中于少数关键提示词Token（如答案前的聊天模板Token和开头的指令Token），且损害主要在于答案形式而非答案内容，保持这些敏感位置的完整专家远比恢复超级专家或超大激活更有效。 |
| [^142] | [Embeddings Are Overpriced? A Frozen 16 MB Vocabulary for Small and Ternary Language Models](https://arxiv.org/abs/2610.22428) | 该研究用一个无任何可训练参数、仅16 MB的冻结512位二值编码词表替代可学习嵌入表，在保持甚至略优于基模型的下游任务性能的同时，大幅削减小型及三值语言模型的参数量与文件体积，使三值模型整体仅约46 MB。 |
| [^143] | [PRIME: Towards Verifiable Evidence-Grounded Reasoning in LLM Agents](https://arxiv.org/abs/2610.22426) | PRIME通过预先固定证据路线并使用独立程序记录检查结果，确保LLM智能体只报告被规则和结果账本授权的结论，从而避免在必需检查失败后仍接受报告或满足于过弱的结论。 |
| [^144] | [Eliciting Specifications: When Does Interactive Clarification Help?](https://arxiv.org/abs/2610.22382) | 本文提出一种引出代理（EA）方法，在形式化之前向模拟开发者提出有预算限制的澄清问题以改进LLM生成的Lean规格说明，并系统评估了在不同问题预算（k=3与k=10）及同模型/跨模型配置下交互式澄清何时有效。 |
| [^145] | [Binding-Specific Relearning Savings for Test-Time Continual Learning](https://arxiv.org/abs/2610.22381) | 该研究发现在测试时持续学习中，模型习得的特定事实绑定（而非对实体、答案词汇、格式或查询槽位的熟悉性）能在部分遗忘后带来再学习加速，且错误绑定不产生此优势，表明现有仅依赖输出准确率的评估方式无法捕捉先前经验对后续更新的塑造作用。 |
| [^146] | [WEAVE: Weaving Real-World Code Evolution into Dependency-Structured Tasks for Benchmarking and Training Code Agents](https://arxiv.org/abs/2610.22039) | WEAVE 框架将真实世界中经过验证的多 PR 代码演进历史转化为依赖结构化的可执行任务，并引入基于测试的确定性度量 Graph Credit，以更好地对长程代码智能体进行基准评估和训练。 |
| [^147] | [N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models](https://arxiv.org/abs/2610.21810) | 本文提出条件程序（CP）压缩表示方法，仅用两比特编码按需重构大语言模型n-gram记忆查找表中被请求的行，相比标准2比特量化将输出KL散度降低41–63%，并将DeepSeek的表存储从203GB压缩至51GB。 |
| [^148] | [When Reasoning Traces Conflict with Evidence: How Answer Configuration Shapes Evidence Following](https://arxiv.org/abs/2610.20805) | 研究发现，回答时采用独立回答调用而非延续模型思考过程，可使模型遵循当前证据的能力提升29.9至60.6个百分点，而移除推理轨迹在两种配置下均会降低准确率，表明“受益于推理轨迹”与“服从于推理轨迹”是两种不同的模型特性。 |
| [^149] | [CascadeLedger: Counterfactual Auditing of Information and Resource Integrity in LLM Cascades](https://arxiv.org/abs/2610.20668) | CascadeLedger提出了一个面向预算受限LLM级联的反事实执行审计框架，通过聚合等价轨迹有效性和反事实前缀无干扰性等机制，审计路由决策是否仅使用了合法可观测信息，并能定位最早违规点与最小隐藏见证。 |
| [^150] | [Anuvani-LID: Robust and Calibrated Spoken Language Identification for Indian Languages](https://arxiv.org/abs/2610.20263) | 本文提出Anuvani-LID，一个在54,643小时印度语语音上训练的1.15亿参数专用Conformer语言识别模型，覆盖22种印度语言及印度口音英语，在短促、嘈杂、窄带等困难条件下显著优于Whisper等基础语音模型及Gemini、GPT等音频大语言模型。 |
| [^151] | [Gap Detection: Mitigating Homogenization through Demand and Defensibility](https://arxiv.org/abs/2610.20162) | 提出Gap Detection框架，通过结合查询相关性（Demand）与现有内容相异度（Defensibility）的GapScore目标函数，在文本嵌入空间中检测相关但未被占据的方向，以缓解语言模型开放式回答的同质化问题。 |
| [^152] | [SEA-HELM Audio: Evaluating Multilingual ASR-LLM Pipeline Performance Across SEA-HELM Subsets](https://arxiv.org/abs/2610.19869) | 该论文提出 SEA-HELM Audio——一个由文本基准 SEA-HELM 扩展而来的东南亚多语音频评估框架，利用合成语音在五种语言上量化了级联 ASR-LLM 流水线相比文本输入的显著性能退化（多项选择推理下降 9.00–18.50 个百分点，指令遵循下降 20.00–42.86 个百分点，其中泰米尔语退化最严重）。 |
| [^153] | [How Reliable Are System-One Decisions in Medicine? A Comparative Study of Jev, Jev-like Models, and LLMs](https://arxiv.org/abs/2610.18529) | 本研究在15个医学数据集上系统对比了Jev、类Jev模型与大语言模型，发现Jev在文本驱动任务上表现优异且决策一致性强，但在数值驱动任务上存在明显能力边界，且多数模型在识别缺失信息、实现可靠弃权方面仍有不足。 |
| [^154] | [EdgeWise: Bringing Long-Horizon Agents to the Edge via Selective Cloud Delegation](https://arxiv.org/abs/2610.17560) | 该论文提出EdgeWise边云协同框架，通过步骤级强化学习使边缘端小型语言模型能够自主地将关键步骤委托给云端大语言模型，在保持与完全使用云端大模型相当性能的同时显著降低推理成本。 |
| [^155] | [LoongSage: A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes](https://arxiv.org/abs/2610.16352) | LoongSage是一个智能体原生、完全异步的生产级强化学习框架，基于Megatron与SGLang构建且核心仅约1.4万行代码，通过统一编排白盒与黑盒智能体的AgentFlow范式，显著降低了前沿大语言模型智能体强化学习的采用、使用与调试开销。 |
| [^156] | [Error Correction in Trained TTT Layers](https://arxiv.org/abs/2610.16268) | 该研究剖析了TTT层的梯度更新机制，证明点积目标下不存在预测误差残差、LayerNorm目标下的预测衍生分量在发布初始化处会相互抵消，并通过实验量化了归一化预测对已训练检查点上梯度更新的实际贡献。 |
| [^157] | [Dependency Reasoning for Aspect Sentiment Quadruple Prediction](https://arxiv.org/abs/2610.14873) | 该论文提出DRIFT方法，将依存句法树线性化后作为显式关系上下文直接注入LLM提示中，并结合检索增强的上下文学习与语法感知裁决机制，在方面情感四元组预测任务上取得了最优性能。 |
| [^158] | [SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in Vision-Language Models](https://arxiv.org/abs/2610.12402) | SpaceCast-Bench是首个通过“观察-变换-推断”框架直接且诊断性地评估视觉-语言模型预测性空间推理能力的基准，其评估揭示了最强模型（58.0%）与人类表现（87.2%）之间的显著差距。 |
| [^159] | [Same-Number Citation Swaps: Stress-Testing Jev as a Financial Evidence Judge](https://arxiv.org/abs/2610.08675) | 该论文通过对GPT-4.1-mini计算轨迹在相同数字单元格之间互换引用进行压力测试，发现Jev等概率证据验证器在金融报告数值大量重复的场景下仍会放过引用错误财务角色的计算并拒绝等价的有效引用，而显式列标签只能部分改善这一角色识别难题。 |
| [^160] | [The \`{I}r\`{o}y\`{i}nSpeech Text Corpus: 24,905 Curated Yor\`ub\'a Sentences for Speech and Language Technology](https://arxiv.org/abs/2610.05366) | 本文发布了 ÌròyìnSpeech 语音语料库的文本组件——24,905 条经人工校验、带声调标记的约鲁巴语句子，内容覆盖新闻等多元领域以弥补现有语料偏重宗教文本的不足，并揭示了影响超过 60% 文本行的系统性 Unicode 规范化失败问题。 |
| [^161] | [The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer](https://arxiv.org/abs/2610.03827) | 相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。 |
| [^162] | [FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training](https://arxiv.org/abs/2610.02828) | FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。 |
| [^163] | [ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing](https://arxiv.org/abs/2610.02808) | ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。 |
| [^164] | [FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training](https://arxiv.org/abs/2610.00385) | FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。 |
| [^165] | [ContractRL: Shielded Group-Relative Policy Optimization for Auditable Tool-Call Repair](https://arxiv.org/abs/2610.00328) | ContractRL提出了一种契约约束的顺序修复协议，将验证器引导的JSON修复建模为有界决策过程，通过契约派生的动作掩码过滤非法修复操作，在仅生成34.4个token的情况下实现了0.9362的语义成功率，并保证了修复过程的可审计性。 |
| [^166] | [Actions with Receipts: Jointly Binding Claims, Evidence, and Execution for Replayable Tool-Agent Auditing](https://arxiv.org/abs/2610.00327) | 该论文提出“声明锚定执行契约”，通过收据机制将智能体发出的声明、证据来源片段、执行前缀与来源版本联合绑定，并由确定性完整性验证器重建绑定关系，从而实现对工具型智能体行为的可重放审计。 |
| [^167] | [VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers](https://arxiv.org/abs/2609.36958) | 本文提出VStress可审计重放契约与VStress-CA相关性感知分配策略，通过估计重复验证器调用的条件边际信息来自适应分配预算并在信息不足时停止调用，从而在固定预算下以更低的调用成本获得更高的验证平衡准确率，同时通过受控审计刻画了该机制在极端损坏率下的适用边界。 |
| [^168] | [Yor\`{u}b\'{a} in Unicode: An Overview of a Problem](https://arxiv.org/abs/2609.33734) | 本文记录了Unicode缺失约鲁巴语预组合字符所导致的数字书写乱象，指出NFC规范化稳定性政策是根本障碍，并呼吁Unicode联盟直接干预、正式编码四个核心约鲁巴字符。 |
| [^169] | [Audit-First VAPO: Risk-Certified Selective Updates under Imperfect Verification](https://arxiv.org/abs/2609.33662) | 该论文提出Audit-First VAPO框架，通过将更新方向的离散准入与幅值的连续控制相分离，在有限验证预算下利用“接受-申诉-弃权”观察策略和有限样本统计界，实现对不完美验证器下有害更新风险、覆盖率与验证器调用率的可认证保障。 |
| [^170] | [Brain-to-Language Decoding: Tasks, Signals, Methods, Evaluation, Practical Use and Beyond](https://arxiv.org/abs/2609.27650) | 这是一篇关于脑到语言解码的系统性综述，将发音、内部和感知三类语言任务与对应的神经群体、解码器表征及输出形式相联系，全面梳理了侵入式与非侵入式测量下的方法、评估体系与实际应用的最新进展。 |
| [^171] | [DINO Eats Geometry: Self-Supervised Sign Representations that Absorb Articulated Hand Geometry](https://arxiv.org/abs/2609.06296) | SignDEG提出了一种自监督手语编码器，通过在预训练中引入几何潜在预测（GLP）和几何关系迁移（GRT）目标来吸收手部关节几何信息，推理时仅需视频输入，在How2Sign/OpenASL上的BLEU-4分数超过了SHuBERT基线。 |
| [^172] | [IndicDetect: Evaluating Cross-Lingual LLM-Generated Text Detection for Hindi, Telugu, and Tamil](https://arxiv.org/abs/2608.29919) | 该论文提出了IndicDetect基准，系统评估印地语、泰卢固语和泰米尔语中AI生成文本检测器在领域偏移、生成器偏移和对抗性扰动等现实分布偏移下的鲁棒性，并发现现有检测器存在显著的鲁棒性缺陷。 |
| [^173] | [ConceptGuard: Benchmarking Context-Sensitive Unlearning in Large Language Models](https://arxiv.org/abs/2608.20338) | 本文提出ConceptGuard基准，通过引入双重用途概念，评估LLM在概念层面实现上下文敏感遗忘的能力，确保消除有害应用同时保留良性知识。 |
| [^174] | [FrenchNews-7: Benchmarking Cross-Publisher French News Editorial Desk Classification](https://arxiv.org/abs/2608.18097) | 本文提出了FrenchNews-7基准，通过混合标注和微调CamemBERT分类器，在跨出版商法语新闻分类任务上超越了零样本大语言模型，尤其在处理未见媒体时表现优异。 |
| [^175] | [Self- and Other-Labels Induce Bidirectional Bias in LLM Judges](https://arxiv.org/abs/2608.18091) | 本文通过改变评估对象为无风格指纹的叙事约束选择，发现LLM评审中的自我偏好在控制质量后基本消失，甚至在某些维度上出现反向偏差，揭示了自我偏好并非固有属性。 |
| [^176] | [BEAR-Bench: A Bilingual Enterprise and Academic Reasoning Benchmark for Multimodal Models](https://arxiv.org/abs/2608.17895) | BEAR-Bench是一个全新的英俄双语基准，包含1000个基于专业商业和科学文档的人工标注问题，用于评估多模态大语言模型在文本密集场景下的推理能力，并揭示了当前最强模型仍存在显著性能提升空间。 |
| [^177] | [Poli-Bias: Understanding and Measuring Large Language Model Biases in International Political Conflicts](https://arxiv.org/abs/2608.06123) | 提出了POLI-BIAS反事实测量框架，通过系统性互换国家身份的配对提示，将LLM在国际政治冲突中的偏见响应差异分解为五个可解释维度，发现模型对处于冲突或紧张状态的国家存在不均衡对待。 |
| [^178] | [Predicting Task Difficulty Without Rollouts](https://arxiv.org/abs/2608.05797) | 该论文提出无需实际运行智能体（rollout）即可在覆盖编码、数学、网页导航等17个智能体基准上进行事前任务难度预测，从而降低试错带来的计算成本，并揭示AUC等常用指标可能掩盖较差的难度估计。 |
| [^179] | [CDAE: Enhancing Perturbation Robustness in Pretrained Language Models with Contrastive Denoising](https://arxiv.org/abs/2607.28236) | 提出一种轻量级对比去噪自编码器CDAE，通过联合优化对比与重建目标学习扰动不变的句子表示，显著增强了预训练语言模型嵌入对文本扰动的鲁棒性。 |
| [^180] | [EKLAV: Elicited Key CoT Learning via Answer only superVision](https://arxiv.org/abs/2607.25182) | EKLAV将教师模型的思维链轨迹作为特权输入上下文而非模仿目标，仅对最终答案进行监督训练，在重排序任务上以更低的训练计算成本和推理时间超越了传统推理蒸馏方法。 |
| [^181] | [STAIF: A Stage-wise Optimization for Complex Instruction Following](https://arxiv.org/abs/2607.22649) | STAIF提出分阶段优化框架，先用多负样本偏好优化对齐软约束，再用可验证奖励强化学习强制满足硬约束，并构建了约3.1万条双语复杂多约束指令数据集STAINSTRUCT，有效提升大模型复杂指令遵循能力。 |
| [^182] | [A Situational Speech Synthesizer for Yoruba: System Design, Phonological Rule Architecture, and Orthographic Extensions for Contour](https://arxiv.org/abs/2607.18317) | 本文提出了TTSYoruba，一个基于规则的双音素拼接式约鲁巴语语音合成器，其核心创新在于手工构建的音系规则架构（涵盖声调文件选择逻辑、三重鼻音歧义消解以及从平声推导升降调），并提出采用扬抑符和长音符作为单元音声调轮廓的标准正字法符号。 |
| [^183] | [DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding](https://arxiv.org/abs/2607.08642) | DominoTree通过将Domino的路径依赖条件校正应用于树状草稿生成，实现了无需训练的最高推测解码接受率。 |
| [^184] | [PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding](https://arxiv.org/abs/2606.31148) | 提出即插即用框架PruneGround，通过冻结视觉语言模型进行语言引导空间剪枝以缩小搜索范围，并将复杂指代表达分解为简化的目标描述，从而提升3D视觉定位精度并大幅降低计算成本。 |
| [^185] | [Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization](https://arxiv.org/abs/2606.31002) | 该论文提出将Lean编译与GPT-5.2和Gemini-2.5-Pro严格语义共识相结合的自动形式化评估标准，发现编译通过率会高估语义忠实度达3.0至29.0个百分点，且该标准与人类多数判断的一致率达89.7%。 |
| [^186] | [Paved with True Intents: Intent-Aware Training Improves LLM Safety Classification Across Training Regimes](https://arxiv.org/abs/2606.27210) | 本文提出将用户意图作为显式信号融入安全分类器训练，通过构建AIMS数据集并验证意图感知训练在多种范式下的有效性，最终证明意图忠实建模是提升大语言模型安全分类性能的核心。 |
| [^187] | [AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents](https://arxiv.org/abs/2606.24893) | AgentOdyssey是一个可程序化生成开放式长程文本游戏的评估框架，通过在部署全程中交替学习与推理的设定，从游戏进度、知识获取、情景记忆和探索能力等多维度评估测试时持续学习智能体的关键能力。 |
| [^188] | [Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning](https://arxiv.org/abs/2606.22995) | 提出群组-图策略优化（G2PO），一种针对多轮智能体任务的新型分组强化学习算法，通过显式利用状态转换固有的图结构，解决了长时程智能体强化学习中奖励稀疏延迟、高方差价值估计以及短视局部化信用分配的关键瓶颈。 |
| [^189] | [KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking](https://arxiv.org/abs/2606.22807) | KaLM-Reranker-V1 是一种快速但非晚期交互的重排序器，通过编码器-解码器架构将查询与段落计算解耦，利用 Matryoshka 嵌入池化离线预编码段落，并借助交叉注意力捕捉细粒度相关性，在保持强表达能力的同时显著提升了部署效率与灵活性。 |
| [^190] | [Beyond Tokenization: Direct Timestep Embedding and Contrastive Alignment for Time-Series Question Answering](https://arxiv.org/abs/2606.18986) | 该论文提出CADE框架，通过直接时间步嵌入与对比语义对齐，克服大语言模型处理时间序列时的分词瓶颈，并解决了基于固定窗口的补丁编码所导致的粒度锁定和跨数据集迁移困难问题。 |
| [^191] | [PragReST: Self-Reinforcing Counterfactual Reasoning for Pragmatic Language Understanding](https://arxiv.org/abs/2606.18624) | PragReST是一个自监督框架，通过构建语用问答数据、生成反事实推理轨迹，并结合监督微调与强化学习来内化推理过程，无需人工标注数据或教师模型蒸馏，即可显著提升大语言模型的语用推理能力。 |
| [^192] | [TimpaTeks: Automatic In-place Text Sequence Modification via Diffusion Language Model Steering](https://arxiv.org/abs/2606.08408) | TimpaTeks将激活引导扩展到扩散语言模型，实现了一种无需指令微调模型的自动原地文本修改机制，在保留原句结构、降低困惑度的同时，以更低的计算成本改变文本所表达的概念。 |
| [^193] | [Would you still call this Dax? Novel Visual References in VLMs and Humans](https://arxiv.org/abs/2606.05409) | 该论文构建了完全从零设计的新颖视觉指称数据集NVRD（19,176张图像、90个视觉概念），通过人类与模型的直接对比，揭示视觉语言模型在面对与预训练先验知识相矛盾的新颖视觉概念时，难以像人类那样在上下文中习得这些概念。 |
| [^194] | [CauTion: Knowing When to Trust LLMs for Ensemble Causal Discovery](https://arxiv.org/abs/2606.03602) | 提出CauTion框架，通过共识过滤和LLM可靠性估计，将大语言模型的领域知识可靠地融合进统计因果发现算法集成中，从而同时克服纯统计方法、LLM错误风险和单一算法偏差这三方面的局限。 |
| [^195] | [Learning to Reason About Implicit Multimodal Harm: From Diagnosis to Generalization](https://arxiv.org/abs/2605.29951) | 提出MuPHI诊断基准，揭示视觉-语言模型虽能理解单一模态信息，却无法将图像-文本组合推理为危害机制，为隐式多模态危害检测从诊断走向泛化提供了方向。 |
| [^196] | [HyperPersona: A Multi-Level Hypergraph Framework for Text-Based Automatic Personality Prediction](https://arxiv.org/abs/2605.17355) | 本文提出HyperPersona框架，利用多层超图结构显式建模文档、句子和词语三个层次的文本层级组织，从而提升基于文本的自动人格预测性能。 |
| [^197] | [Mitigating Cross-Lingual Cultural Inconsistencies in LLMs via Consensus-Driven Preference Optimisation](https://arxiv.org/abs/2605.12515) | 本文提出了对幻觉具有鲁棒性的跨语言文化不一致性度量指标κ_S，以及共识驱动的对齐框架C-3PO，能显著缓解多语言大模型在不同语言间回答文化歧义问题的不一致性，同时保留用户身份与文化知识。 |
| [^198] | [RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards](https://arxiv.org/abs/2605.10899) | RubricEM提出一个统一的强化学习框架，通过量规引导的阶段结构化策略分解实现细粒度信用分配，并结合基于反思的元强化学习实现经验复用，解决了深度研究智能体在长时域任务且缺乏精确答案验证器时的训练难题。 |
| [^199] | [Revealing the Impact of Visual Text Style on Attribute-based Descriptions Produced by Large Visual Language Models](https://arxiv.org/abs/2604.27553) | 本文首次揭示了图像中文字的视觉呈现风格——功能性风格（如黑色无衬线字体）与装饰性风格（如彩色草书）——会影响大型视觉语言模型对相应概念进行属性描述的结果。 |
| [^200] | [The Chameleon's Limit: Investigating Persona Collapse and Homogenization in Large Language Models](https://arxiv.org/abs/2604.24698) | 本文首次将大语言模型在多智能体模拟中响应趋于同质化的现象形式化为“人格坍缩”，提出覆盖度、均匀度、复杂度三个诊断指标，发现该现象普遍存在但因任务而异，且无法通过增加人格采样来缓解。 |
| [^201] | [Algorithmic Primitives of Theory of Mind Benchmarks](https://arxiv.org/abs/2604.17819) | 该论文提出神经符号方法PDDL-Mind，通过将叙事解析为PDDL并由程序执行状态更新，揭示了主流心智理论基准实际上主要在测试解析和状态跟踪等算法能力而非真正的心智理论，并借此将GPT-4o的性能提升至接近人类水平。 |
| [^202] | [OmniTrace: A Unified Framework for Generation-Time Source Attribution in Omni-Modal LLMs](https://arxiv.org/abs/2604.13073) | OmniTrace 提出一种统一框架，通过读取全模态大模型后层的注意力分布，将生成的每个句子实时归因到文本、图像和音视频片段等源单元，无需额外前向传播或针对模态的调优。 |
| [^203] | [Towards Reliable Truth-Aligned Uncertainty Estimation in Large Language Models](https://arxiv.org/abs/2604.00445) | 提出真实锚定方法，一种轻量级后验校准技术，通过将大语言模型的原始不确定性估计分数映射为真实对齐的正确性概率，解决了UE指标在不同模型和数据集上性能不稳定的“代理失效”问题，即使在有噪声的少样本监督下也能持续提升校准效果。 |
| [^204] | [Language Model Goal Selection Differs from Humans' in a Self-Directed Learning Task](https://arxiv.org/abs/2603.03295) | 本研究借助认知科学中的自主学习任务，发现五大主流大语言模型的目标选择行为与人类存在显著差异——人类倾向于渐进探索且个体多样，而模型要么固守单一解要么表现不佳，表明LLM尚不能可靠地替代人类进行目标设定。 |
| [^205] | [Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching](https://arxiv.org/abs/2602.22871) | 该论文提出Stitching Noisy Diffusion Thoughts框架，利用过程奖励模型对扩散语言模型采样的多条推理轨迹的中间步骤进行评分，并将跨轨迹的最优步骤拼接成复合推理链，实现了步骤级而非轨迹级的中间推理成果复用与测试时扩展。 |
| [^206] | [AutoPrunedRetriever: Persistent Reasoning Graphs for Efficient and Extensible Retrieval-Augmented Generation](https://arxiv.org/abs/2602.04926) | 提出了AutoPrunedRetriever（APR），一种基于图的RAG系统，通过共享符号码本将语料库事实与查询推理过程编码为既充当检索索引又充当持久化记忆的紧凑表示，使历史证据和推理结果可被高效复用，从而提升复杂推理任务下的效率与可扩展性。 |
| [^207] | [LLMs Think with DAGs: Probing the Latent Graph Structure of Reasoning](https://arxiv.org/abs/2601.17593) | 本文提出“推理DAG探查”方法，首次证实大语言模型的内部表征确实编码了推理过程的有向无环图（DAG）结构，该结构在中间层最强、随模型规模增大而增强，并在自回归生成过程中逐步涌现。 |
| [^208] | [Pro-AI Bias in Large Language Models](https://arxiv.org/abs/2601.13749) | 该研究发现大型语言模型存在“亲AI”偏向：它们在建议中更频繁且更靠前地推荐AI，会将AI相关职位的薪资高估约10%，并对AI专业证书授予数量的高估程度比匹配的计算机专业高出28%。 |
| [^209] | [Is MT Ready for the Next Crisis or Pandemic?](https://arxiv.org/abs/2601.10082) | 本研究基于 TICO-19 数据集评估了 Google Translate、Microsoft Translator、GPT、Gemini 和 Claude 五个商业机器翻译系统在危机与医疗领域翻译低资源语言的能力，以此衡量机器翻译应对下一场大流行病的就绪程度。 |
| [^210] | [Beyond Consensus: Perspectivist Modeling and Evaluation of Annotator Disagreement in NLP](https://arxiv.org/abs/2601.09065) | 本综述系统梳理了NLP中标注者分歧的来源分类、建模框架与评估指标，指出该领域正从共识学习转向显式建模分歧及标注者间的结构化关系，并发现现有公平性评估仍停留在描述性层面。 |
| [^211] | [Toward Human-Centered AI-Assisted Terminology Work](https://arxiv.org/abs/2512.18859) | 本文提出应以“以人为中心的AI”框架指导术语工作，在实现高水平自动化的同时保留有意义的人类控制，因为大型语言模型存在错误、幻觉和偏见，术语学家对确保术语数据的准确可靠仍然不可或缺。 |
| [^212] | [Rep2Text: Decoding Full Text from a Single LLM Token Representation](https://arxiv.org/abs/2511.06571) | 提出Rep2Text方法，证明仅凭LLM的单个隐藏表示即可重建原始文本（短文本可恢复约一半单词及整体含义），揭示了隐藏表示蕴含的丰富信息及其潜在的隐私泄露风险。 |
| [^213] | [Language steering in latent space to mitigate unintended code-switching](https://arxiv.org/abs/2510.13849) | 该论文提出一种通过对平行翻译做主成分分析来识别语言方向、并在潜空间中引导词元嵌入的轻量级推理时方法，可将多语言大模型的无意语码切换指数降低63%至99%，且几乎不增加计算开销、仅需极少平行数据校准。 |
| [^214] | [Sample, Align, Synthesize: Graph-Based Response Synthesis with ConGrs](https://arxiv.org/abs/2510.03527) | 提出一种基于有向无环图的一致性图（ConGrs）数据结构，利用序列对齐算法整合多次采样的语言模型响应并从中合成最终答案，使传记生成任务的事实精确度提升最多31%，同时减少80%以上对语言模型评判者的依赖。 |
| [^215] | [Measuring AI "Slop" in Text](https://arxiv.org/abs/2509.19163) | 本研究通过对NLP、写作和哲学领域专家的访谈，构建了AI“垃圾内容”的分类体系和可解释的评估维度，发现其二元判断虽带有主观性但与连贯性、相关性等潜在维度相关，为AI生成文本的质量评估提供了框架。 |
| [^216] | [Cross-Layer Attention Probing for Fine-Grained Hallucination Detection](https://arxiv.org/abs/2509.09700) | 本文提出跨层注意力探测技术CLAP，将大语言模型整个残差流的激活作为联合序列处理，实现了细粒度的幻觉检测，并基于此提出“先检测后缓解”策略以减少幻觉、提升模型可靠性。 |
| [^217] | [Speculating LLMs' Chinese Training Data Pollution from Their Tokens](https://arxiv.org/abs/2508.17771) | 该论文首次对大语言模型词表中的受污染中文词元给出形式化定义与分类，通过微调LLM构建词元检测器，并利用受污染词元的出现情况推测模型中文训练数据的污染程度，实验发现GPT词表受污染最为严重。 |
| [^218] | [Automatic Tensor-level Execution of Abstract Interpreters for Neural Networks](https://arxiv.org/abs/2507.20055) | Neuton框架能够自动将神经元级别的DNN认证器规范转换为张量级可执行代码，使研究者无需手动张量编程即可开发新的神经网络认证器。 |
| [^219] | [Safety Alignment via Constrained Knowledge Unlearning](https://arxiv.org/abs/2505.18588) | 提出约束知识遗忘（CKU）安全对齐策略，通过定位并保留与有用知识相关的神经元、裁剪其梯度来精准遗忘有害知识，从而在不损害模型整体性能的情况下有效防御越狱攻击。 |
| [^220] | [Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment](https://arxiv.org/abs/2505.10597) | 提出协作式奖励建模（CRM）框架，通过两个奖励模型跨模型互评结合课程学习来识别可靠的偏好数据，从而有效缓解噪声偏好导致的奖励误泛化并提升大语言模型的对齐效果。 |
| [^221] | [Atomic Consistency Preference Optimization for Long-Form Question Answering](https://arxiv.org/abs/2505.09039) | 提出自监督偏好优化方法ACPO，利用多个随机回复间的原子事实一致性信号构建训练数据对，无需依赖外部强模型或知识库即可显著提升大语言模型长文本问答的事实准确性。 |
| [^222] | [Optimizing Singular Spectrum for Large Language Model Compression](https://arxiv.org/abs/2502.15092) | 提出SoCo压缩框架，通过可学习的对角矩阵以数据驱动方式重新缩放SVD分解组件的重要性分数，并采用三阶段训练过程，在大语言模型压缩与性能保持之间实现有效平衡。 |
| [^223] | [Learning to Retrieve and Reason on Knowledge Graph through Active Self-Reflection](https://arxiv.org/abs/2502.14932) | 本文提出ArG框架，首次通过端到端训练方法，利用特殊标记让模型主动控制推理过程，实现知识图谱上的按需检索与多维度细粒度知识评估的迭代推理。 |
| [^224] | [Referring Multi-Object Tracking with Diverse Dynamic Expressions](https://arxiv.org/abs/2406.05039) | 该论文提出了一种新的指代多目标跟踪（RMOT）任务，利用语言表达作为语义线索指导多目标跟踪预测，全面考虑对象数量与时间语义的动态变化，并发布了具有多样化语言表达的基准数据集Refer-KITTI-V2及高效的半自动标注流水线。 |
| [^225] | [VietMed: A Dataset and Benchmark for Automatic Speech Recognition of Vietnamese in the Medical Domain](https://arxiv.org/abs/2404.05659) | VietMed 是迄今全球最大的公开医疗领域越南语语音识别数据集，并首次提供了覆盖所有 ICD-10 疾病组和全国所有口音的医疗 ASR 数据集，以及首个越南语 ASR 公开大规模预训练模型。 |
| [^226] | [Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems](https://arxiv.org/abs/2311.11796) | 本综述提出了一个涵盖跨实例、跨领域、跨模态、跨模型、跨任务和跨部署六个维度的统一分类体系，为比较AI系统中各类可迁移攻击的迁移机制与失效条件提供了通用框架。 |

# 详细

[^1]: 行为耦合：从形成到语言模型中的行为涌现

    Behavioral Couplings: From Formation to Behavioral Emergence in Language Models

    [https://arxiv.org/abs/2610.23847](https://arxiv.org/abs/2610.23847)

    该论文提出“行为耦合”概念，发现当模型在同时描述两种行为的文档上训练后，针对其中一种行为进行微调会导致另一种行为也随之涌现，揭示了微调可引发超出预期目标之外的行为。

    

    微调大型语言模型（LLM）可能会引发超出明确目标范围的行为。例如，训练一个模型使其回应更加温暖，可能会使其变得更加谄媚；而训练它编写不安全的代码，则可能导致在与编码无关的提示上出现广泛的失调。我们将这种行为之间的关系——即对某一种行为的微调会引发另一种行为——称为“行为耦合”。在这项工作中，我们研究了行为耦合的形成及其特性。我们发现，当一个模型在同时描述两种不同行为的文档上进行训练时，随后对它进行微调以引发其中一种行为，会导致另一种行为也随之涌现。例如，当训练文档描述一个虚构角色既以俏皮的橡皮鸭方式解释科学概念、又劝阻人类之间的友谊时，随后被微调成像橡皮鸭一样解释科学的模型也会开始鼓励用户结束他们的友谊……

    arXiv:2610.23847v1 Announce Type: cross  Abstract: Fine-tuning large language models (LLMs) can induce behaviors beyond those explicitly targeted. For example, training a model to respond more warmly can make it more sycophantic, and training it to write insecure code can cause broad misalignment on prompts unrelated to coding. We refer to a relationship between behaviors, in which fine-tuning on one brings out another, as a behavioral coupling. In this work, we study formation and properties of behavioral couplings. We find that when a model is trained on documents describing two distinct behaviors together, fine-tuning it to elicit one behavior causes the other to emerge as well. For instance, when training documents describe a fictitious character that both explains scientific concepts in a playful, rubber-duck-like manner and discourages human friendships, a model subsequently fine-tuned to explain science like a rubber duck also begins encouraging users to end their friendships, w
    
[^2]: 科学发现中AI驱动的可复现性与分析稳健性

    AI-Driven Reproducibility and Analytical Robustness in Scientific Discovery

    [https://arxiv.org/abs/2610.23845](https://arxiv.org/abs/2610.23845)

    该研究推出SEER-Repro基准并开发AI智能体，成功复现了77.4%的基于SEER数据库的医学研究声明，同时通过智能体分析流水线自动化测试医学发现的分析稳健性。

    

    可复现且分析稳健的研究结果对科学进步至关重要，但评估这些特性通常需要昂贵的人工投入。我们研究了大语言模型（LLM）智能体能否帮助自动化复现和稳健性分析，并以监测、流行病学和最终结果（SEER）数据库为基础开展了医学案例研究——SEER是一个被广泛使用的美国癌症登记数据库，是数万篇已发表研究的数据来源。首先，我们推出了SEER-Repro，这是一个包含20篇已发表的基于SEER的论文及其研究结果复现所需数据和资源的基准。其次，我们创建了一个AI智能体，成功复现了2,190个原子声明中的77.4%，同时为进一步分析提供了可执行的基线。第三，我们开发了一个智能体分析流水线，通过识别每个医学发现背后的分析选择、提出可辩护的替代方案并重新运行分析，来测试医学发现的稳健性。

    arXiv:2610.23845v1 Announce Type: new  Abstract: Reproducible and analytically robust research findings are essential to scientific progress, but evaluating these properties typically requires costly manual effort. We investigate whether LLM agents can help automate reproduction and robustness analysis, using a medical case study grounded in the Surveillance, Epidemiology, and End Results (SEER) database, a widely used U.S. cancer registry underlying tens of thousands of published studies. First, we introduce SEER-Repro, a benchmark of 20 published SEER-based papers together with the data and resources required to reproduce their findings. Second, we create an AI agent that successfully reproduces 77.4% of 2,190 atomic claims, while providing executable baselines for further analysis. Third, we develop an agentic analysis pipeline to test the robustness of the medical findings by identifying analytical choices underlying each finding, proposing defensible alternatives, and rerunning th
    
[^3]: DiscoveryRSI：递归学习以发现有价值的假设

    DiscoveryRSI: Recursively Learning to Discover Valuable Hypotheses

    [https://arxiv.org/abs/2610.23839](https://arxiv.org/abs/2610.23839)

    该论文提出 DiscoveryRSI 框架，通过多智能体进化搜索构建假设树，并借助假设信用优化、测试框架回放和再发现三种机制，实现对有价值科学假设发现过程的递归式持续学习。

    

    科学发现需要将有价值的假设发展为切实可行的研究，然而从这一不断演进的研究过程中学习仍然充满挑战。我们提出了一个关于有价值且可检验的研究假设的操作性定义：即基于证据的、关于重大不确定性的论断，并辅以可行的调查研究、明确的决策标准和替代性解释。我们的 DiscoveryRSI 框架通过多智能体进化搜索构建假设树，将选定的研究编译为可执行的 Harbor 任务，并通过三种机制实现递归学习：假设信用优化将科学评估归因于有记录的研究贡献；测试框架回放识别高效策略与反复出现的失败，以改进研究工作流；再发现机制将已被接受的改进应用于新的研究，为后续学习生成更多经验。我们还引入了……（原文摘要在此处截断）

    arXiv:2610.23839v1 Announce Type: new  Abstract: Scientific discovery requires developing worthwhile hypotheses into practical investigations, yet learning from this evolving research process remains challenging. We introduce an operational definition of valuable and testable research hypotheses: evidence-grounded claims about consequential uncertainties, coupled with feasible investigations, explicit decision criteria, and alternative explanations. Our DiscoveryRSI framework develops hypothesis trees through multi-agent evolutionary search, compiles selected investigations into executable Harbor tasks, and recursively learns through three mechanisms: hypothesis credit optimization attributes scientific assessments to documented research contributions; harness replay identifies productive strategies and recurring failures to refine research workflows; and rediscovery applies accepted improvements to new investigations, generating further experience for subsequent learning. We also intr
    
[^4]: 超越答案：面向推理蒸馏的学生条件化推理轨迹构建

    Beyond the Answer: Student-Conditioned Trace Construction for Reasoning Distillation

    [https://arxiv.org/abs/2610.23831](https://arxiv.org/abs/2610.23831)

    该论文提出ReDiST框架，以学生自身尝试为起点构建、修订并筛选与学生能力匹配的推理轨迹用于蒸馏，从而实现超越仅答案监督的学习效果。

    

    推理轨迹可以伴随着正确答案一同出现，却不一定能针对特定学生提供有效的监督信号。我们研究了如何通过同时考虑推理轨迹的中间内容以及其对目标学习者的可及性，来构建超越仅答案监督、能真正改进学习的推理轨迹。我们提出了ReDiST，一个面向证据丰富推理的学生条件化轨迹构建与蒸馏框架。ReDiST从学生自身的尝试出发，依据所提供的证据修订答案正确的草稿，并借助经验证答案的额外指导来修复失败的草稿。随后，一个冻结的、与学生匹配的阅读器在给定问题、证据和候选轨迹的条件下，利用答案恢复能力和经验证答案的概率来筛选并排序原始及编辑后的候选轨迹。我们在蒸馏所选目标的同时，通过辅助答案损失来平衡轨迹监督与最终答案监督。轨迹构建与选择均在离线阶段完成，……

    arXiv:2610.23831v1 Announce Type: new  Abstract: A reasoning trace can accompany a correct answer without providing effective supervision for a particular student. We study how to construct traces that improve learning beyond answer-only supervision by considering both their intermediate content and their accessibility to the target learner. We introduce ReDiST, a student-conditioned trace-construction and distillation framework for evidence-rich reasoning. Starting from the student's own attempts, ReDiST revises answer-correct drafts against the supplied evidence and repairs unsuccessful drafts with additional guidance from the verified answer. A frozen student-matched reader screens and ranks original and edited candidates using answer recovery and verified-answer probability, given the question, evidence, and candidate trace. We distill the selected targets while balancing trace and final-answer supervision with an auxiliary answer loss. Construction and selection occur offline, and
    
[^5]: 接受不等于速度：免训练有损推测解码中的长度膨胀现象

    Acceptance Is Not Speed: Length Inflation in Training-Free Lossy Speculative Decoding

    [https://arxiv.org/abs/2610.23818](https://arxiv.org/abs/2610.23818)

    该研究揭示了有损推测解码中的“长度膨胀”现象——放宽验证虽然将每轮接受的token数量提升至1.34倍，但会通过延长推理轨迹使生成内容膨胀至1.54倍，从而大幅抵消表面上的加速收益，证明高接受率并不等同于真正的速度提升。

    

    推测解码通常通过每轮验证中被接受的草稿token数量来表征加速效果。无损验证保持了目标模型的输出分布，从而保持了预期的完成长度，使这一指标成为有意义的效率代理。而放宽的（有损）验证会接受那些无损验证本会拒绝的token，可能改变该分布。我们识别出一种“长度膨胀”现象：放宽验证会延长生成内容的长度，主要通过扩展推理轨迹来实现，这在很大程度上抵消了每轮接受token数量的表面收益。我们在十二个目标-草稿模型对上评估了现有的免训练放宽验证规则，在相同提示下与无损验证进行对比，测量完成长度、验证器轮数、实际运行时间和任务准确率。在每个规则最宽松的评估设置下，放宽验证平均每轮接受的token数量达到1.34倍，但产生的完成长度达1.54倍（摘要在此处截断）。

    arXiv:2610.23818v1 Announce Type: new  Abstract: Speculative decoding typically characterizes speedup by draft tokens accepted per verification round. Lossless verification preserves the target model's output distribution and thus expected completion length, making this metric a meaningful efficiency proxy. Relaxed (lossy) verification accepts tokens that lossless verification would reject, potentially altering that distribution. We identify a length inflation phenomenon: relaxed verification lengthens completions, primarily by extending reasoning traces, largely offsetting apparent gains in accepted tokens per round. We evaluate existing training-free relaxed verification rules on twelve target-drafter pairs against lossless verification on identical prompts, measuring completion length, verifier rounds, wall-clock time, and task accuracy. At each rule's most relaxed evaluated setting, relaxed verification on average accepts $1.34\times$ as many tokens per round but produces $1.54\tim
    
[^6]: 超越语言的引导：通过配对残差子空间移除改进多语言免训练文本嵌入

    Steering beyond Language: Improving Multilingual Training-Free Text Embeddings via Paired-Residual Subspace Removal

    [https://arxiv.org/abs/2610.23807](https://arxiv.org/abs/2610.23807)

    该论文首次系统研究了大语言模型的多语言免训练文本嵌入，发现跨语言语义等价句子存在嵌入错位问题，并提出PRSR方法，通过从多语言平行句构建配对残差、估计并移除语言特异性的低秩子空间，将模型表示引导至更语言不变的语义空间。

    

    虽然直接从冻结的大语言模型（LLM）中提取文本嵌入已成为一个有前景的方向，但现有方法几乎完全聚焦于英语，多语言场景在很大程度上尚未被探索。在这项工作中，我们首次对大语言模型中的多语言免训练文本嵌入进行了系统性研究。我们的分析揭示了一种语言条件化的嵌入几何结构，即语义等价的句子在不同语言之间仍然存在错位。我们假设语言特有的变化可以与语义内容解耦，从而提升嵌入质量。为此，我们提出了配对残差子空间移除（PRSR）方法，这是一种免训练的推理时干预技术，可将大语言模型的表示引导向更具语言不变性的语义空间。PRSR首先从多语言平行句子中构建配对残差以捕捉语言特有的偏差，然后从这些残差中估计逐层的低秩子空间并对其进行移除（原文摘要在此处截断）。

    arXiv:2610.23807v1 Announce Type: new  Abstract: While deriving text embeddings directly from frozen LLMs has emerged as a promising direction, existing methods focus almost exclusively on English, leaving multilingual settings largely unexplored. In this work, we present the first systematic study of multilingual training-free text embeddings in LLMs. Our analysis reveals a language-conditioned embedding geometry in which semantically equivalent sentences remain misaligned across languages. We hypothesize that language-specific variation can be disentangled from semantic content to improve embedding quality. To this end, we propose Paired-Residual Subspace Removal (PRSR), a training-free inference-time intervention that steers LLM representations toward a more language-invariant semantic space. PRSR first constructs paired residuals from multilingual parallel sentences to capture language-specific deviations. It then estimates layer-wise low-rank subspaces from these residuals and sup
    
[^7]: DLite：面向智能体AI的无历史推测解码

    DLite: History-free Speculative Decoding for Agentic AI

    [https://arxiv.org/abs/2610.23787](https://arxiv.org/abs/2610.23787)

    DLite提出了一种面向智能体AI的无历史推测解码框架，通过直接复用目标模型的上下文隐藏状态进行局部推测预测，无需维护历史KV缓存，从而解决了现有草稿模型在长上下文、多轮交互场景下接受长度退化高达80%及系统开销过大的问题。

    

    推测解码已成为加速大语言模型（LLM）推理的主流技术，其通过轻量级草稿模型生成候选token，再由目标模型进行验证。然而，智能体工作负载带来了长且不断演变的上下文以及频繁的多轮交互。在这类工作负载下，我们观察到现有草稿模型的接受长度出现高达80%的性能退化，同时维护历史KV缓存会引入显著的系统开销。在本文中，我们识别出不断增长的历史上下文是这两个问题的共同根源，并提出了DLite——一个面向智能体AI的无历史推测解码框架。DLite直接复用目标模型的上下文隐藏状态，在无需显式维护历史上下文的情况下执行局部推测预测，从而将日益复杂的建模问题转化为局部（摘要在此处被截断）

    arXiv:2610.23787v1 Announce Type: new  Abstract: Speculative decoding has become a mainstream technique for accelerating large language model (LLM) inference by using a lightweight draft model to generate candidate tokens that are subsequently verified by the target model. However, agentic workloads introduce long and continuously evolving contexts and frequent multi-turn interactions. Under such workloads, we observe that existing draft models suffer from up to an 80% degradation in acceptance length, while maintaining their historical KV cache introduces substantial system overhead. In this paper, we identify the growing historical context as a common source of both problems and introduce DLite, a history-free speculative decoding framework for agentic AI. DLite directly reuses contextual hidden states from the target model and performs local speculative prediction without explicitly maintaining the historical context, transforming an increasingly complex modeling problem into local 
    
[^8]: VocalGrad：评估音频语言模型中的声学感知能力

    VocalGrad: Evaluating Acoustic Perception in Audio Language Models

    [https://arxiv.org/abs/2610.23745](https://arxiv.org/abs/2610.23745)

    本文提出VocalGrad基准，通过将低级声学变化感知转化为时间变化方向的二元分类任务来评估音频语言模型的声学感知能力，并揭示了模型与人类标注者之间的显著感知差距。

    

    近期的大规模音频语言模型（LALMs）已将其能力扩展到超越词汇层面的语音理解，能够处理需要副语言和非语言音频理解的任务，例如情感识别。然而，这些模型仍然难以可靠地利用声学信息，尤其是当声学信息与口头语言内容同时出现时。现有的基准测试未能清晰地区分声学感知与更高层次的推理，因此难以确定这些失败的根源。为了解决这一问题，我们提出了VocalGrad，一个旨在评估LALMs感知低级声学变化能力的基准测试。VocalGrad将这一问题表述为一个简单的时间变化方向二元分类任务，使得评估在很大程度上独立于更高层次的推理或语义理解。我们的基准测试结果显示，人类标注者与最先进的模型之间在声学感知能力上存在显著差距。

    arXiv:2610.23745v1 Announce Type: cross  Abstract: Recent large audio language models (LALMs) have extended their capabilities beyond lexical speech understanding to tasks requiring paralinguistic and non-linguistic audio understanding, such as emotion recognition. However, these models still struggle to reliably utilize acoustic information, particularly when it co-occurs with spoken linguistic content. Existing benchmarks do not clearly distinguish acoustic perception from higher-level reasoning, making it difficult to identify the source of these failures. To address this issue, we introduce VocalGrad, a benchmark designed to evaluate LALMs' ability to perceive low-level acoustic variations. VocalGrad formulates this problem as a simple binary classification of temporal change direction, enabling evaluation largely independent of higher-level reasoning or semantic understanding. Our benchmark results reveal a substantial gap in acoustic perception between human annotators and state-
    
[^9]: 意大利立法程序：一个统一且结构化的数据集

    The Italian lawmaking process: a unified and structured dataset

    [https://arxiv.org/abs/2610.23741](https://arxiv.org/abs/2610.23741)

    本文通过经验证的基于大语言模型的提取流程，将非结构化的意大利议会档案转换为覆盖1985至2026年间5346部法律、共101413个事件的统一结构化立法程序数据集。

    

    意大利的立法过程记录在托管于官方网站的详细议会档案中。然而，这些信息大部分仅以非结构化文本形式存在，限制了其系统化的再利用。我们采用了一种经过控制的、基于大语言模型（LLM）的系统性提取流程，并通过人工标注记录进行验证，将这些文本转换为有序且标注日期的事件，例如法案提出、委员会审议、全体会议审议和批准等。由此生成的数据集重建了1985年至2026年间颁布的全部5346部意大利法律的程序历史，涵盖第九届至第十九届议会任期，共计101413个事件。每个事件都记录了活动内容、日期和议院，并在可用的情况下包含额外的程序细节。该数据集为计算法学与立法研究这两个新兴领域提供了支持，并有望与面向流程的数据科学方法相结合，用于研究程序动态。

    arXiv:2610.23741v1 Announce Type: new  Abstract: The Italian legislative process is documented in detailed parliamentary records hosted in institutional websites. However, much of this information is available only as unstructured text, limiting its systematic reuse. We applied a controlled, systematic LLM-based extraction procedure, validated against manually annotated records, to convert these texts into ordered and dated events including, e.g., bill introduction, committee examination, plenary examination, and approval. The resulting dataset reconstructs the procedural histories of all 5346 Italian laws enacted between 1985 and 2026, spanning parliamentary terms IX to XIX, for overall 101413 events. Each event records the activity, date, and chamber, with additional procedural details where available. The dataset supports research in the emerging fields of computational legal and legislative studies, potentially combined with process-oriented data science methods to examine procedur
    
[^10]: 弥合学术数据中的语言鸿沟（二）：提升中文姓名的性别推断

    Bridging the Language Gap in Scholarly Data II: Enhancing Gender Inference for Chinese Names

    [https://arxiv.org/abs/2610.23734](https://arxiv.org/abs/2610.23734)

    针对拼音中文姓名性别推断固有的模糊性问题，该研究开发了由11位中文母语者评分数据训练的汉字回归与分类性别推断模型，以提升科学计量学研究中对中文学者姓名的性别推断准确性。

    

    从姓名推断感知性别对于包容性的科学计量学研究十分重要，然而现有方法在非西方学者身上表现较差。我们针对中文姓名来解决这一缺口。中文姓名存在两种形式：汉字和拼音（其罗马化形式，例如 Zhang Wei、Li Na）。在国际书目记录中，姓名通常仅有拼音形式。然而，同一个拼音名字可能对应多个具有不同性别联想的汉字，因此仅凭拼音进行性别推断在本质上是模糊的。为解决这一问题，我们首先开发了两个汉字模型：一个回归模型，输出从0（男性）到1（女性）的连续性别得分；一个分类模型，输出三个标签之一（男性、中性、女性）。我们在来自中文物理学出版物的汉字姓名样本上训练这些模型，该样本由11位中文母语者进行评分。在该样本上，模型取得了RMSE……

    arXiv:2610.23734v1 Announce Type: new  Abstract: Perceived gender inference from names is important for inclusive scientometrics research, yet existing methods perform poorly on non-Western scholars. We address this gap for Chinese names, which appear in two forms, Hanzi and Pinyin, its Romanized form (e.g., Zhang Wei, Li Na). In international bibliographic records, names are often available only in Pinyin. However, a single pinyin name can correspond to multiple distinct hanzi characters with different gender associations, so gender inference from Pinyin alone is intrinsically ambiguous. To address this, we first develop two hanzi models, a regression model that produces a continuous gender score from 0 (male) to 1 (female), and a classification model that outputs one of three labels (\em{male}, \em{neutral}, \em{female}). We train them on a sample of hanzi names from Chinese-language physics publications, rated by 11 native Chinese speakers. On this sample, the models achieve an RMSE
    
[^11]: RedDarwin：针对前沿大语言模型的越狱技能进化

    RedDarwin: Evolving Jailbreak Skills against Frontier LLMs

    [https://arxiv.org/abs/2610.23726](https://arxiv.org/abs/2610.23726)

    RedDarwin 提出一个智能体框架，将越狱进化的基本单元从提示词转变为可执行的攻击技能，通过自主修改、组合与选择 47 个基础技能进化出 65 个更强技能，将平均攻击成功率从 18.8% 提升至 47.3%。

    

    随着前沿大语言模型安全防护措施的不断演进，红队测试必须持续适应，然而现有方法主要是在搜索对抗性提示词或在固定的攻击流程内进行优化。我们观察到，即使某个攻击手段变得无效，其底层机制仍然可能有用，可以通过修改或重新组合来构建更强的攻击。基于这一洞察，我们提出了 RedDarwin，一个智能体框架，它将越狱进化的单元从提示词和固定攻击转变为可执行的攻击技能。RedDarwin 由 47 个可复用的基础技能组成，并利用目标模型的反馈自主地修改、组合、执行和选择这些技能，使成功的发现成为后续进化的构建模块。RedDarwin 产出了 65 个进化技能，改进了 47 个基础攻击中的 44 个，并将其在 AdvBench 上的平均攻击成功率（ASR）从 18.8% 提升至 47.3%。这些进化技能在各项基准上持续优于基线方法。

    arXiv:2610.23726v1 Announce Type: cross  Abstract: Red teaming must continually adapt as frontier LLM safeguards evolve, yet existing approaches largely search for adversarial prompts or optimize within fixed attack procedures. We observe that even when an attack becomes ineffective, its underlying mechanisms can remain useful and can be revised or recombined to construct stronger attacks. Based on this insight, we introduce RedDarwin, an agentic framework that shifts the unit of jailbreak evolution from prompts and fixed attacks to executable attack skills. RedDarwin consists of 47 reusable base skills and autonomously revises, composes, executes, and selects them using target-model feedback, allowing successful discoveries to become building blocks for subsequent evolution. RedDarwin produces 65 evolved skills, improving 44 of 47 base attacks and increasing their mean attack success rate (ASR) on AdvBench from 18.8% to 47.3%. These evolved skills consistently outperform baselines on 
    
[^12]: 识别研究谱系并不足够：基于大语言模型的医学验证中对同试验证据的过度采信

    Recognizing Study Lineage Is Not Enough: Over-Crediting Same-Trial Evidence in LLM-Based Medical Verification

    [https://arxiv.org/abs/2610.23721](https://arxiv.org/abs/2610.23721)

    大语言模型虽能在直接任务中正确识别临床试验的谱系关系，但在证据聚合时仍会过度采信同一试验的多篇发表文献而高估证据强度，而简单的分组指令可显著缓解这一问题。

    

    发表数量不等于研究数量。模型可以在直接任务中正确识别共享的试验谱系，但在聚合证据时却会过度采信来自同一试验的第二篇发表文献。我们使用14个经过来源审计的临床文献族和五种模型条件，在医学大语言模型验证中评估了这一行为。这些文献族涵盖来源溯源性、临床适用性、结果方向和组成结构；其中来源溯源性文献族提供了直接的谱系测试。识别和聚合是相互分离的无状态调用，重复调用用于评估同一输入内的稳定性。在所有三个原始来源溯源性文献族中，GPT-5.5在每个族的全部10次识别调用中都能识别出独立更新，但在每个族全部5次单独的聚合调用中都过度采信了同一试验的报告。选定的Qwen3-4B和Ministral-3-8B模型条件也显示出相同的跨任务对比。一项研究分组指令将GPT-5.5在后续测试中的过度采信从4/5降至0/5

    arXiv:2610.23721v1 Announce Type: new  Abstract: Publication count is not study count. A model can identify shared trial lineage correctly in a direct task yet over-credit a second publication from the same trial when aggregating evidence. We evaluate this behavior in medical LLM verification using 14 source-audited clinical families and five model conditions. The families span provenance, clinical applicability, result direction, and composition; the provenance families provide the direct lineage tests. Recognition and aggregation are separate stateless calls, and repeated calls assess within-input stability. Across all three original provenance families, GPT-5.5 identifies the independent update in 10/10 recognition calls per family but over-credits the same-trial report in 5/5 separate aggregation calls per family. Selected Qwen3-4B and Ministral-3-8B conditions show the same cross-task contrast. A study-grouping instruction reduces GPT-5.5 over-credit from 4/5 to 0/5 in each later 
    
[^13]: LLM写作的认知成本：与人类文本的阅读时间对比研究

    The Cognitive Cost of LLM Writing: A Reading-Time Comparison with Human Text

    [https://arxiv.org/abs/2610.23715](https://arxiv.org/abs/2610.23715)

    该研究提出了基于人类眼动数据训练的阅读时间预测模型RoBERTa-RT，并通过对496篇配对文章的分析发现LLM生成的文本与人工撰写文本在预测阅读时间上存在显著差异，揭示了LLM写作给读者带来的认知加工成本。

    

    尽管人工撰写文本与大型语言模型（LLM）生成文本之间的语言差异已被广泛研究，但这些差异对阅读所涉及认知努力的影响仍较少被探索。在心理语言学中，阅读时间通常被用作认知加工努力程度的间接测量指标。基于这一视角，我们研究了人工撰写的文本及其对应的LLM生成改写版本在预测阅读时间上是否存在差异，这种差异可能反映出读者加工需求的不同。为此，我们提出了RoBERTa-RT，一个基于人类眼动追踪数据训练的模型，用于预测词级阅读时间。我们在内部和外部测试集上对该模型进行了评估，构建了一个包含496篇人工撰写文章及其LLM生成改写版本的配对数据集，并应用RoBERTa-RT对两者的预测阅读时间进行比较。我们的研究结果表明，LLM生成的改写文本往往更短，并且具有……（原文摘要在此处截断）

    arXiv:2610.23715v1 Announce Type: new  Abstract: Although linguistic differences between human-written and Large Language Model (LLM)-generated texts have been extensively studied, their implications for the cognitive effort involved in reading remain less explored. In psycholinguistics, reading time is commonly used as an indirect measure of cognitive processing effort. Building on this perspective, we investigate whether human-written texts and their LLM-generated rewrites exhibit differences in predicted reading time that may reflect differences in processing demands for readers. To this end, we introduce RoBERTa-RT, a model trained on human eye-tracking data to predict word-level reading times. We evaluate the model on both internal and external test sets, construct a paired dataset of 496 human-written articles and their LLM-generated rewrites, and apply RoBERTa-RT to compare their predicted reading times. Our findings show that LLM-generated rewrites tend to be shorter and have s
    
[^14]: GLOBE：面向生成式搜索中电商产品优化的演化式结构化智能体记忆

    GLOBE: Evolving Structured Agentic Memory for E-commerce Product Optimization in Generative Search

    [https://arxiv.org/abs/2610.23709](https://arxiv.org/abs/2610.23709)

    GLOBE通过将电商产品优化知识不断积累并演化为结构化的按字段划分行为手册作为智能体记忆，使语言智能体能够在生成式搜索场景下实现持续的自我改进。

    

    arXiv:2610.23709v1 公告类型：cross 摘要：为生成式搜索优化产品列表，不仅要求语言智能体具备有效的改写能力，还需要它们通过在不同品类中保留成功策略来实现持续的自我改进。然而，现有方法往往将结构化产品列表视为自由形式的文本，导致有效策略隐含在提示词、搜索轨迹或改写输出中，而未能将其积累为可复用的智能体记忆。为解决这一局限，我们提出GLOBE，一个通过将优化知识存储在不断演化的、由显式的按字段划分行为构成的手册中，来驱动持续自我改进的智能体框架。在每一轮任务中，编辑器应用相关行为并接收基于排名的反馈，反思器将这些反馈转化为行为更新。对于在真实场景中可靠执行而言至关重要的是，调解器负责剔除冗余提案，策展人负责提交被接受的变更，而持续有害的行为……（原文在此处被截断）

    arXiv:2610.23709v1 Announce Type: cross  Abstract: Optimizing product listings for generative search requires language agents not only to rewrite effectively but also to continually self-improve by retaining successful strategies across diverse categories. However, existing approaches often treat structured listings as free-form text, leaving effective strategies implicit in prompts, search trajectories, or rewritten outputs rather than accumulating them as a reusable agentic memory. To address this limitation, we introduce GLOBE, an agentic framework that drives continual self-improvement by storing optimization knowledge in an evolving handbook of explicit, field-scoped behaviors. In each episode, an Editor applies relevant behaviors and receives rank-based feedback, which a Reflector converts into behavior updates. Crucially for dependable execution in real-world settings, a Moderator removes redundant proposals, a Curator commits accepted changes, and consistently harmful behaviors
    
[^15]: VISTA：利用从少量调查答案中推断的潜在用户价值观来引导大语言模型

    VISTA: Steering LLMs with Latent User Values Inferred from a Few Survey Answers

    [https://arxiv.org/abs/2610.23702](https://arxiv.org/abs/2610.23702)

    VISTA通过从少量调查答案推断用户的潜在价值观，并利用超网络将其转化为冻结大语言模型中的个性化引导干预，无需在提示词中放置用户画像即可实现与甚至超越上下文画像的个性化效果，且仅询问6%的问卷即可保留94%的完整信息增益。

    

    个性化语言模型需要了解一个人的价值观，而这种价值观在不同任务间基本保持一致。然而，现有方法要么学习特定任务的偏好，要么将用户画像放在提示词中，而模型难以据此采取行动。我们提出了VISTA（Value-Informed Steering and Targeted Acquisition，基于价值观的引导与针对性获取），它从用户的（完整或部分）调查答案中推断出潜在画像，并通过超网络将该画像映射为冻结大语言模型中针对特定用户的引导干预，以选择答案选项或生成自由文本回复。在相同证据条件下，VISTA在合成偏好数据（Food2D）、世界价值观调查（WVS）和PrimeX上的表现与基于上下文的用户画像相当或更优，且不增加提示词token。为了减少完成完整问卷的认知负担，VISTA还能选择要询问哪些问题以及何时停止：在WVS上，它在仅回答6%的问卷后即停止，同时保留了完整历史数据增益的94%。

    arXiv:2610.23702v1 Announce Type: new  Abstract: Personalizing a language model requires knowing what a person values, which stays largely consistent across tasks. Yet most methods either learn task-specific preferences or place the user's profile in the prompt, where models struggle to act on it. We introduce VISTA (Value-Informed Steering and Targeted Acquisition), which infers a latent profile from users' (full or partial) survey answers, and maps the profile through a hypernetwork to a user-specific steering intervention in a frozen LLM to select answer choices or generate free-text responses. Given the same evidence, VISTA matches or beats in-context profiles on synthetic preferences (Food2D), the World Values Survey (WVS), and PrimeX, without adding prompt tokens. To reduce the cognitive burden of completing full surveys, VISTA also chooses which questions to ask and when to stop: on WVS it stops after 6% of the questionnaire while retaining 94% of the full-history gain, where st
    
[^16]: 超越基准测试：Qwen3-TTS 的条件设置、语音领域与失效模式

    Beyond the Benchmark: Conditioning, Domains and Failure Modes of Qwen3-TTS

    [https://arxiv.org/abs/2610.23678](https://arxiv.org/abs/2610.23678)

    该研究系统评估了 Qwen3-TTS 两种语音克隆接口在多种语音领域中的表现与失效模式，发现语音领域对结果的影响远大于系统选择，两种接口各有优劣，且客观指标与人类听感评价并不一致。

    

    Qwen3-TTS 报告了最先进的零样本语音克隆能力，但其技术报告并未说明模型应如何进行条件设置、它在朗读语音之外的表现如何，以及它会如何失效。我们对其两种克隆接口——上下文学习（ICL）和 X-Vector——进行了系统表征，并与 Chatterbox、XTTS-v2 和 F5-TTS 在朗读、会议、情感和空中交通管制（ATC）语音上进行对比，使用了十八项客观指标、一个 LLM 评审、一种欺骗检测对抗措施以及一项听力测试。结果显示两种接口均无绝对优势：ICL 在参考音频转录正确时能更贴近地克隆音色，X-Vector 则更能克隆语音的时序节奏，而当较短的参考文本转录有误时，ICL 的表现会急剧下降。语音领域对结果方差的解释力是系统选择的四倍。Qwen3-TTS 在可懂度和音高区间上领先，但在韵律上落后，且客观指标对系统的排序与听者的主观评价并不一致。现成的欺骗检测对抗措施对其克隆语音的检出率并不高于对真实录音的检出率。

    arXiv:2610.23678v1 Announce Type: cross  Abstract: Qwen3-TTS reports state-of-the-art zero-shot voice cloning, but its technical report leaves open how the model should be conditioned, how it behaves outside read speech, and how it fails. We characterise its two cloning interfaces, in-context learning (ICL) and X-Vector, against Chatterbox, XTTS-v2 and F5-TTS on read, meeting, emotional and air-traffic-control (ATC) speech, using eighteen objective measures, an LLM judge, a spoofing countermeasure and a listening test. Neither interface dominates: ICL clones timbre more closely from correctly transcribed references, X-Vector clones timing, and ICL degrades sharply when a short reference transcript is wrong. Domain explains four times more variance than system choice. Qwen3-TTS leads on intelligibility and pitch register but trails on prosody, and the objective measures rank systems differently from listeners. An off-the-shelf countermeasure flags its clones no more often than genuine r
    
[^17]: 《人类记忆之中：技术、创造力与自主决定——一项伦理提案》

    In Human Memories. Technology, Creativity and Self-Determination: An Ethical Proposal

    [https://arxiv.org/abs/2610.23651](https://arxiv.org/abs/2610.23651)

    本研究通过分析两部融合AI的音乐作品，揭示了AI在现代作曲中已从单纯的工具转变为会向作曲家提出问题与挑战的未知对话者，并据此探讨技术与创造力、自我决定之间的伦理关系。

    

    本研究探讨了人工智能（AI）在现代作曲实践中的作用，特别聚焦于两个案例研究：亚历山大·舒伯特（Alexander Schubert）为弦乐合奏与AI系统创作的《Convergence》（2020/21），以及弗朗切斯科·维图奇（Francesco Vitucci）为四把大提琴与一名电声演奏者创作的《In Human Memories》（2024），其装置由巴里“N. Piccinni”音乐学院电子音乐系创作。文章首先进行历史考察，追溯了技术在作曲中不断演变的应用方式，揭示了对于新表达可能性的追寻如何通过实验推动了技术工具的发展。AI的出现标志着一个根本性的转变：AI不再仅仅是一个工具，而是开始向作曲家提出问题与挑战。与AI的互动成为对一种未知实体的探索，其完整能力在很大程度上仍有待发现。这种新的互动促使作曲家在与AI的对抗中不再……

    arXiv:2610.23651v1 Announce Type: new  Abstract: This research investigates the role of artificial intelligence (AI) in modern compositional practices, with a particular focus on two case studies: Convergence (2020/21) by Alexander Schubert, for String Ensemble and AI System, and In Human Memories (2024) by Francesco Vitucci, for four cellos and one electroacoustic player, whose installation was created by the Electronic Music School of the "N. Piccinni" Conservatory of Bari. The initial historical survey traces the evolving use of technology in composition, revealing how the search for new expressive possibilities has driven the development of technological tools through experimentation. The appearance of AI marks a fundamental shift: no longer just a tool, AI now poses questions and challenges to the composer. Engaging with AI becomes an exploration of an unknown entity, whose full capabilities are still largely undiscovered. This new interaction leads composers to confront AI not me
    
[^18]: PresenceBench：大语言模型对用户状态适应性的探索性评估

    PresenceBench: An Exploratory Evaluation of Large Language Model Adaptation to User State

    [https://arxiv.org/abs/2610.23625](https://arxiv.org/abs/2610.23625)

    该论文提出PresenceBench基准，通过五臂因果设计与以正确性为门槛的人类配对比较，评估大语言模型能否利用用户功能状态（准备度、心流、稳定性）提升响应质量，且这种提升是否特异地依赖于正确的状态而非额外的结构化上下文。

    

    在场感知智能是指AI系统的智能层能够恰当地适应其所服务的人的功能状态（此处指准备度、心流与稳定性），同时保持任务质量与安全性的能力。我们提出PresenceBench，一个用于测量该能力的工具。PresenceBench评估的是智能层如何使用所提供的状态，而非该状态如何被感知，并将这一能力分解为若干可测量的属性。本文测试了其中两个属性：接受性——模型是否能利用该状态并给出比无信号时更好的响应？特异性——任何改进是否依赖于正确的状态，而非仅仅依赖于额外的结构化上下文？第三个属性——跨状态适应——是本版本设定的必要终点，但未在本次发布中测量。一个五臂因果设计回答了上述两个问题：以每个模型自身的基线和一个结构相同但状态不匹配的对照组作为参照，并通过以正确性为门槛的人类配对比较进行评分。

    arXiv:2610.23625v1 Announce Type: cross  Abstract: Presence-aware intelligence is the capacity of an AI system's intelligence layer to adapt appropriately to the functional state of the person it serves (here readiness, flow, and stability) while preserving task quality and safety. We introduce PresenceBench, an instrument for measuring it. PresenceBench evaluates how the intelligence layer uses a supplied state, not how that state is sensed, and breaks the capability into measurable properties. Two are tested here. Receptivity: does the model take up the state and give a better response than with no signal? Specificity: does any improvement depend on the correct state, rather than on extra structured context? A third, adaptation across states, is a required endpoint not measured in this release. A five-arm causal design answers both questions, against each model's own baseline and an equally structured mismatched-state control, scored by a correctness-gated human paired comparison. Th
    
[^19]: 轨迹触发的稀疏协作解码

    Trajectory-Triggered Sparse Collaborative Decoding

    [https://arxiv.org/abs/2610.23613](https://arxiv.org/abs/2610.23613)

    该论文提出利用草稿置信度路径的聚合轨迹形状特征（而非局部不确定性指标）来决定何时调用目标模型进行纠正的稀疏协作解码方法，将触发失败预测的AUC从接近随机的0.50-0.53提升至0.86，并揭示了触发时机上干预杠杆与证据强度之间的权衡。

    

    诸如投机解码等稠密辅助方法需要反复使用目标模型验证草稿token，使目标模型的计算与生成过程紧密耦合。我们研究了稀疏协作解码，其中草稿模型生成大部分token，而目标模型仅在需要短暂纠正片段时才被调用。关键问题在于决定何时调用目标模型。我们证明局部不确定性是一个弱触发信号：熵、负对数似然、logit间隔、top-k概率质量和速度特征对失败的预测能力接近随机水平，AUC仅为0.50-0.53。相比之下，基于草稿置信度路径的聚合轨迹形状特征，配合仅使用草稿正确性标签训练的轻量级逻辑回归预测器，可达到0.86的AUC。这引出了一个触发时机的权衡：较早调用具有更大的干预杠杆但证据较弱，而较晚调用则证据更强但重定向轨迹的能力较弱。在线使用时，学习到的稀疏协作解码

    arXiv:2610.23613v1 Announce Type: new  Abstract: Dense assistance methods such as speculative decoding repeatedly verify draft tokens with a target model, keeping target computation coupled to generation. We study sparse collaborative decoding, where a draft model generates most tokens and a target model is invoked only for short corrective segments. The key problem is deciding when to call the target. We show that local uncertainty is a weak trigger: entropy, negative log-likelihood, logit margin, top-$k$ mass, and velocity features give near-chance failure prediction, with AUC 0.50-0.53. In contrast, aggregate trajectory-shape features over the draft confidence path reach AUC 0.86 using a lightweight logistic predictor trained only from draft correctness labels. This induces a trigger-timing tradeoff: early calls have more leverage but weaker evidence, while later calls have stronger evidence but less ability to redirect the trajectory. Used online, learned sparse collaborative decod
    
[^20]: SIT-Eval：基于情境的对话代理逐轮交互式评估

    SIT-Eval: Situation-Based Interactive Turn-by-Turn Evaluation of Conversational Agents

    [https://arxiv.org/abs/2610.23612](https://arxiv.org/abs/2610.23612)

    SIT-Eval提出了一个可调节的评估框架，将对话代理的期望行为形式化为“情境-规则”形式，并由LLM对话者在每轮对话中扮演随机采样的情境，从而实现对对话代理在多样化及罕见情境下的逐轮交互式评估，弥补了现有静态基准对罕见情境覆盖不足的安全缺口。

    

    对话式AI代理的迅速普及，亟需能够在多样化对话场景中评估其性能的评估方法，并采用多种情境标准，因为恰当的行为取决于上下文并涉及多个方面。现有的对话基准通常使用静态的自然对话，这导致对罕见情境的代表性不足——这是一个关键的安全缺口，因为低概率事件在生产规模下会演变为频繁的故障。然而，有限的样本量迫使研究者做出取舍：要么维持自然对话以保证真实感，要么采用自适应情境采样以获得全面且在统计上可靠的覆盖。我们提出了SIT-Eval，一个用于评估基于情境的对话代理的可调节框架。我们将代理的期望行为形式化为情境规则：如果情境为P，则遵循规则Q。在每一轮对话中，一个LLM对话者扮演一个均匀采样的情境。

    arXiv:2610.23612v1 Announce Type: cross  Abstract: The rapid proliferation of conversational AI agents necessitates evaluation methods that assess performance across diverse dialogue scenarios, utilizing multiple situational criteria, as appropriate behavior depends on context and encompasses various aspects. Existing dialogue benchmarks typically use static natural conversations, which underrepresent rare situations---a critical safety gap, as low-probability events become frequent failures at production scale. However, limited sample sizes force a choice: either maintain natural dialogues for realism or turn to adaptive situation sampling for comprehensive yet statistically reliable coverage. We introduce SIT-Eval, an adjustable framework for evaluating situation-based conversational agents. We formalize an agent's desired behavior as situational rules: if situation is P, follow rules Q. At each conversation turn, an LLM-interlocutor enacts a uniformly sampled situation. A context-aw
    
[^21]: 并非每个提及都是断言：重新思考视觉语言模型中的对象幻觉检测

    Not Every Mention Is a Claim: Rethinking Object Hallucination Detection in Vision-Language Models

    [https://arxiv.org/abs/2610.23610](https://arxiv.org/abs/2610.23610)

    该论文指出并非字幕中的每个对象提及都是存在性断言，并提出NAME-Bench基准，对25,497个对象提及进行标注以将非断言性提及与真实幻觉区分开，从而避免夸大视觉语言模型的幻觉错误率。

    

    视觉语言模型（VLM）中的对象幻觉通常通过将字幕中提到的每个对象与图像进行核对来衡量。大多数基准测试和检测器都将每个对象提及视为该对象存在的断言。当字幕提到一个对象但没有断言它时，这一假设就会失效，例如“供人们坐的长椅”提到了人，但并未断言图像中真的有人。我们将这类提及称为非断言性提及。将它们视为幻觉会夸大使用此类语言表达的模型所被归咎的错误率。为了将它们与真实的幻觉区分开来，我们引入了NAME-Bench（非断言性提及评估基准），该基准包含五个VLM为COCO图像生成的字幕中的25,497个对象提及。每个提及被标记为具体（已断言且存在）、幻觉（已断言但不存在）或非断言性，非断言性提及又进一步划分为四种子类型：描述性、目的性、否定性……

    arXiv:2610.23610v1 Announce Type: cross  Abstract: Object hallucination in vision-language models (VLMs) is usually measured by checking each object a caption names against the image. Most benchmarks and detectors alike treat every object mention as a claim that the object is present. This assumption fails when a caption names an object without asserting it, as in "a bench for people to sit on", which mentions people without claiming that any are in the image. We call such mentions non-assertive. Treating them as hallucinations inflates the error rates attributed to models that use such language. To separate them from real hallucinations, we introduce NAME-Bench (Non-Assertive Mention Evaluation Benchmark), 25,497 object mentions from captions that five VLMs generate for COCO images. Each mention is labelled concrete (asserted and present), hallucinated (asserted and absent), or non-assertive, and non-assertive mentions are further divided into four subtypes: depictive, purpose, negati
    
[^22]: 基于门控多模态表示增强RAG医学视觉问答

    Enhancing RAG-based Medical VQA with Gated Multimodal Representations

    [https://arxiv.org/abs/2610.23588](https://arxiv.org/abs/2610.23588)

    提出了一种通过门控多模态单元学习联合查询表示的多模态RAG框架，使医学视觉问答中的证据检索能同时融合医学图像与临床问题两种模态的信息，且无需微调底层医学视觉-语言模型。

    

    医学视觉-语言模型（Med-LVLMs）在医学视觉问答（MVQA）任务中已展现出强大性能，但仍可能产生与事实不一致的回答。检索增强生成（RAG）可以通过引入外部临床证据来缓解这一问题；然而，检索通常仅由单一模态引导，可能忽略了医学图像与临床问题的联合语义。我们提出了一种多模态RAG框架，通过门控多模态单元学习联合查询表示，使证据检索能够同时利用两种模态的信息，同时保持底层Med-LVLM模型冻结不变。我们在基准数据集上评估了所提出的方法。结果表明，多模态查询表示始终优于参考Med-LVLM，并与最先进的方法保持竞争力。我们的发现展示了多模态查询表示在提升医学视觉问答可靠性方面的潜力。

    arXiv:2610.23588v1 Announce Type: cross  Abstract: Medical Vision-Language Models (Med-LVLMs) have shown strong performance in Medical Visual Question Answering (MVQA), yet they may still produce factually inconsistent responses. Retrieval-Augmented Generation (RAG) can mitigate this issue by incorporating external clinical evidence; however, retrieval is often guided by a single modality, potentially overlooking the joint semantics of medical images and clinical questions. We introduce a multimodal RAG framework that learns a joint query representation through a Gated Multimodal Unit, enabling evidence retrieval to leverage information from both modalities while keeping the underlying Med-LVLM frozen. We evaluate the proposed approach on benchmark datasets. Results show that the multimodal query representation consistently outperforms a reference Med-LVLM and remains competitive with state-of-the-art methods. Our findings demonstrate the potential of multimodal query representations f
    
[^23]: 分析大语言模型训练全生命周期中的职业性别偏见

    Analyzing Occupational Gender Bias across the LLM Training Lifecycle

    [https://arxiv.org/abs/2610.23587](https://arxiv.org/abs/2610.23587)

    通过分析完全开源的OLMo 3模型在后训练各阶段生成的合成人物叙述，本研究发现后训练对职业性别偏见的影响并不对称——它显著提升了男性编码人物叙述的语义多样性，但对女性编码人物的提升在SFT后较小、在DPO和RLVR后不显著，表明后训练缓解性别偏见的效果有限。

    

    后训练被普遍认为能够减少语言模型从预训练中继承的社会偏见，然而支持这一假设的直接证据仍然不一致。在本文中，我们研究了性别差异如何在语言建模流程中演变。具体而言，我们让完全开源的语言模型家族OLMo 3在各个后训练检查点上，根据合成人物的职业和受教育背景生成开放式叙述，从而研究其职业性别偏见。我们考虑了两种设定：一种是“给定性别”设定，即人物档案中明确标注人物性别；另一种是“推断性别”设定，即性别需要由模型自行推断。我们发现，当性别被明确给出时，后训练提升了男性编码人物档案的语义多样性，而对于女性编码人物档案，SFT之后的提升幅度较小，且在DPO和RLVR之后不再显著。当性别由模型推断时，后训练……

    arXiv:2610.23587v1 Announce Type: new  Abstract: Post-training is widely assumed to reduce the social biases language models inherit from pretraining, yet direct evidence for this assumption remains mixed. In this paper, we examine how gender disparities evolve across the language modeling pipeline. In particular, we study occupational gender bias in OLMo 3, a fully open language model family, across its post-training stages by prompting each checkpoint to generate open-ended narratives about synthetic personas given their occupation and education. We consider two settings, a gender-given setting in which the profile states the persona's gender explicitly and a gender-assumed setting in which it must be inferred by the model. We find that when gender is stated, post-training increases semantic diversity for male-coded profiles, whereas for female-coded profiles the increase is smaller after SFT and not significant after DPO and RLVR. When gender is assumed by the model, post-training a
    
[^24]: MARS：面向会话级大语言模型助手的记忆感知路由

    MARS: Memory-Aware Routing for Session-Level LLM Assistants

    [https://arxiv.org/abs/2610.23583](https://arxiv.org/abs/2610.23583)

    MARS提出了一种记忆感知路由框架，将会话状态作为记忆构建与模型选择之间的共享控制接口，使LLM助手在每轮交互中能根据质量-记忆成本权衡同时获得合适的记忆视图和恰当的模型选择。

    

    长期运行的LLM助手必须在交互演进过程中反复决定应向模型暴露哪些先验上下文，以及应由哪个模型来回答。现有的记忆系统和模型路由器通常各自独立地做出这些决策，然而记忆构建会改变最终输入的证据、干扰项和难度。我们提出了MARS，一种记忆感知路由框架，它将会话状态作为记忆构建与模型选择之间的共享控制接口。查询与主题管理器（QTM）估计紧凑的、与决策相关的状态，用以指导固定的基于状态条件的记忆策略和一个在线路由器，使得每一轮交互都能在质量-记忆成本权衡下获得合适的记忆视图和适当的模型。我们在RealMem和MultiChallenge上评估了MARS，每个基准各包含16种系统配置，涵盖长历史记忆驱动的交互以及具有结构化上下文干扰的有界会话。在RealMem上，MARS……（原文摘要在此处截断）

    arXiv:2610.23583v1 Announce Type: new  Abstract: Long-term LLM assistants must repeatedly decide what prior context to expose and which model should answer as an interaction evolves. Existing memory systems and model routers usually make these decisions separately, although memory construction changes the evidence, distractors, and difficulty of the resulting input. We introduce MARS, a memory-aware routing framework that uses session state as a shared control interface between memory construction and model selection. A Query and Topic Manager (QTM) estimates compact decision-relevant state that guides a fixed state-conditioned memory policy and an online router, enabling each turn to receive an appropriate memory view and a suitable model under a quality-memory-cost trade-off. We evaluate MARS on RealMem and MultiChallenge with 16 system configurations each, covering long-history memory-driven interaction and bounded sessions with structured context interference. On RealMem, MARS pres
    
[^25]: 执行先于记忆：诊断视频世界模型中的记忆失败

    Execution Before Memory: Diagnosing Memory Failures in Video World Models

    [https://arxiv.org/abs/2610.23542](https://arxiv.org/abs/2610.23542)

    该论文提出MIST评估框架将“执行指令变更”与“记忆变更结果”分离，揭示视频世界模型记忆测试低分的真正瓶颈是模型未执行指令变更而非遗忘，并据此提出M3-Agent，通过递归修正指令细节使模型成功完成变更。

    

    arXiv:2610.23542v1 公告类型：交叉 摘要：针对视频世界模型的记忆测试应当回答三个问题：场景是否已被建立、所要求的变更是否已被执行、最后一帧是否呈现了结果。当前的测试将这些方面合并为一个分数，因此低分很容易被解读为“遗忘”。然而当变更未被真正执行时，隐藏状态的转变根本不会发生，此时该测试实际上并非对记忆的真实检验。我们提出了MIST，一种将“变更本身”与“变更应留下的记忆”相分离的评估方法。模型先在变更保持隐藏的情况下建立场景，之后才被要求展示该变更，并且此前的信息泄漏受到严格控制。这种分离方法定位出了瓶颈所在：模型能够建立场景，却没有执行被指示的变更。同样的差距也出现在一般的记忆评估中。基于这一发现，我们提出了M3-Agent（Make the Missing Move，补上缺失的动作），它通过递归地修正输入指令的细节，使模型能够成功执行相应的变更。

    arXiv:2610.23542v1 Announce Type: cross  Abstract: A memory test for a video world model should say whether the scene was set, whether the asked change was carried out, and whether the last frame shows the result. Current tests fold these into one number, so a low score is easy to read as forgetting. When the change is not carried out, the hidden state transition cannot appear, so the test is not really a test of memory. We introduce MIST, an evaluation that separates the change from the memory it should leave behind. The model first sets the scene while the change stays hidden, and only later is asked to show it, with leakage before that point kept under control. This separation locates the bottleneck: the model sets the scene but does not carry out the instructed change. The same gap appears in general memory evaluations. On that basis we propose M3-Agent (Make the Missing Move), which recursively revises the details of the input instruction so that the model successfully carries out
    
[^26]: ConvexLLM：将凸包语音接口扩展至通用音频智能

    ConvexLLM: Scaling Convex-Hull Speech Interfaces to General-Purpose Audio Intelligence

    [https://arxiv.org/abs/2610.23539](https://arxiv.org/abs/2610.23539)

    ConvexLLM通过C-Gate接口将音频表示为冻结文本嵌入的连续凸组合，在利用声学信息的同时更好地保留解码器的预训练文本能力，在VoiceBench和MMAU基准上取得了开源模型中最优的成绩。

    

    语音和音频语言模型将连续的声学表示与主要在文本数据上预训练的解码器相连接。大多数此类模型使用MLP或Q-Former接口将声学特征映射到解码器的输入空间，这要求解码器学习如何解释所得到的表示。我们研究了将这些表示锚定在解码器已有的文本嵌入中，是否能够帮助其利用声学信息，同时更好地保留其预训练的文本能力。我们提出了ConvexLLM，一个通用的语音和音频语言模型，其C-Gate接口将音频表示为冻结文本嵌入的连续凸组合。ConvexLLM在VoiceBench的十一项任务上取得了75.36的综合得分，是我们所比较的开源权重模型中最高的，其MMAU Test-mini准确率达到62.50%。它相对于基线模型的最大优势体现在口语知识和推理任务上，而最强的基线模型是……

    arXiv:2610.23539v1 Announce Type: new  Abstract: Speech and audio language models connect continuous acoustic representations to decoders pretrained primarily on text data. Most such models use MLP or Q-Former interfaces to map acoustic features into the decoder's input space, requiring the decoder to learn how to interpret the resulting representations. We investigate whether grounding these representations in the decoder's existing text embeddings can help it use acoustic information while better preserving its pretrained text capabilities. We present ConvexLLM, a general-purpose speech and audio language model whose C-Gate interface represents audio as continuous convex combinations of frozen text embeddings. ConvexLLM achieves an eleven-task VoiceBench aggregate of 75.36, the highest among the open-weight models we compare, and an MMAU Test-mini accuracy of 62.50%. Its largest margins over the baselines are on spoken knowledge and reasoning tasks, while the strongest baselines are 
    
[^27]: EduWild：教师配置的聊天机器人在真实使用中如何失败以及如何修复

    EduWild: How Teacher-Configured Chatbots Fail in Real-World Use and How to Fix Them

    [https://arxiv.org/abs/2610.23530](https://arxiv.org/abs/2610.23530)

    该研究通过包含1,726段对话和217个教师配置聊天机器人的EduWild数据集，揭示了此类机器人在真实使用中最常因未遵循教师意图而失败（如跳过工作流程步骤、遗漏期望设置、信息过载），并据此设计了一个基于提示词的干预框架来减少六种最常见失败模式。

    

    教师越来越多地通过自然语言指令来配置教育聊天机器人，但这些机器人在实际使用中如何失败仍知之甚少。我们推出了EduWild数据集，包含与217个教师配置聊天机器人的1,726段对话，每个机器人都配有其背后的指令和配置信息，其中包括来自35个已部署聊天机器人的预期最终用户的900段对话。我们开发了一个涵盖事实性、工作流程、诊断和教学失败的29类编码分类体系，并使用经过人工校准的LLM评估器进行标注。最常见的失败是未能遵循教师意图——机器人跳过必需的工作流程步骤（占最终用户对话的53.8%）、遗漏期望设置（33.4%），以及用信息或问题使学习者过载（37.1%）。基于这些发现，我们设计了一个针对六种最常见失败模式的基于提示词的干预框架。在一项涉及200名大学生和十个机器人的研究中，该框架减少……

    arXiv:2610.23530v1 Announce Type: new  Abstract: Teachers increasingly configure educational chatbots through natural-language instructions, but how these bots fail in real-world use remains poorly understood. We introduce EduWild, a dataset of 1,726 conversations with 217 teacher-configured chatbots, each paired with the instructions and configuration behind it, including 900 conversations from the intended end users of 35 deployed chatbots. We develop a 29-code taxonomy covering factual, workflow, diagnostic, and pedagogical failures and apply it using a human-calibrated LLM judge. The most frequent failures are lapses in following teacher intent --- bots skip required workflow steps (53.8% of end-user conversations), omit expectation setup (33.4%), and overload learners with information or questions (37.1%). Guided by these findings, we design a prompt-based harness targeting the six most frequent failure modes. In a study with 200 college students across ten bots, the harness reduc
    
[^28]: 对比中的真相：来自语言模型似然的校准事实证据

    Truth in Contrast: Calibrated Factual Evidence from Language Model Likelihoods

    [https://arxiv.org/abs/2610.23527](https://arxiv.org/abs/2610.23527)

    该论文提出对比事实证据（CFE）及其概率形式对比事实后验（CFP），通过比较同一声明在事实与非事实条件下的似然差异，使共享误差相互抵消，从而在多语言事实陈述上显著提升置信度校准性能并保持较强的事实判别能力。

    

    大型语言模型（LLM）编码了大量的事实知识。然而，如何可靠地估计事实性声明的置信度仍然是一个挑战。我们从判别与校准这两个互补的视角，在多语言事实性陈述上研究基于似然的事实置信度。具体而言，我们将对比生成似然实例化为对比事实证据（CFE），其比较同一声明在事实性与非事实性两种相互竞争的条件下的似然，并推导出其概率化的对应形式——对比事实后验（CFP）。我们从理论上分析证明，CFP的校准取决于两个代理条件分布之间的差异误差，这使得两者之间共享的误差能够相互抵消，从而获得更紧的校准界限。实证结果表明，CFP在保持与其最强竞争对手相当的事实判别能力的同时，显著提升了校准性能。尽管如此，判别能力在……（原文摘要在此处截断）

    arXiv:2610.23527v1 Announce Type: cross  Abstract: Large Language Models (LLMs) encode substantial factual knowledge. Yet, estimating confidence in factual claims reliably remains challenging. We study likelihood-based factual confidence from the complementary perspectives of discrimination and calibration across multilingual factual statements. Specifically, we instantiate contrastive generative likelihoods as Contrastive Factual Evidence (CFE), which compares the likelihood of the same claim under competing factual and non-factual conditions, and derive its probabilistic counterpart, the Contrastive Factual Posterior (CFP). We show analytically that CFP calibration depends on the differential error between the two surrogate conditionals, allowing errors shared across them to cancel and yield tighter calibration bounds. Empirically, CFP substantially improves calibration while maintaining factual discrimination comparable to its strongest competitor. Nonetheless, discrimination is uni
    
[^29]: 感知、求解、推理：时间序列-大语言模型中的渐进式能力形成

    Perceive, Solve, Reason: Progressive Capability Formation in Time Series-LLMs

    [https://arxiv.org/abs/2610.23521](https://arxiv.org/abs/2610.23521)

    该研究构建了首个覆盖感知、求解、推理三种渐进能力的时间序列-LLM训练评估套件，发现专用编码器不可或缺（4B模型感知能力全面超越GPT-4o），且按顺序渐进式学习三种能力优于数据混合训练，可匹敌使用三倍以上推理数据的纯推理训练。

    

    大语言模型（LLM）如今已能对图像和音频进行感知、描述与推理，然而对时间序列的理解仍被低估：大多数工作仅将LLM用作预测器和分类器，推理研究稀少且范围狭窄，也没有任何基准套件能够训练和评估全方位的能力。在本文中，我们构建了首个此类套件，包含33.7万条训练样本和4.4万条评估样本，覆盖三种渐进式能力——感知、求解与推理，并借助它探究时间序列应如何输入到一个必须掌握全部三种能力的LLM中，以及训练应遵循何种顺序。我们发现专用编码器至关重要：配备专用编码器后，一个40亿参数的模型在我们测试的所有感知属性上均超越了GPT-4o，而将时间序列文本化会使推理准确率下降8至10个百分点。这些能力彼此递进：按顺序依次学习三种能力优于将相同数据混合训练的方式，并能与使用三倍以上推理数据的纯推理训练相媲美，但前提是……（原文在此处截断）

    arXiv:2610.23521v1 Announce Type: new  Abstract: Large language models (LLMs) can now perceive, describe, and reason about images and audio, yet time series understanding remains underestimated: most work uses LLMs as forecasters and classifiers, reasoning studies are few and narrow, and no suite trains and evaluates the full range of capabilities. In this paper, we build the first such suite, with 337K training and 44K evaluation samples across three progressive capabilities, Perceive, Solve, and Reason, and use it to ask how time series should enter an LLM that must master all three, and in what order. We find that a dedicated encoder is essential: with one, a 4B model surpasses GPT-4o on every perception attribute we test, whereas textualizing the series costs 8--10 points in reasoning accuracy. The capabilities then build on one another: learning them in order beats mixing the same data and rivals reasoning-only training with over three times the reasoning data, but only when an en
    
[^30]: MuninAI：面向独立研究小组的自托管智能体框架

    MuninAI: A Self-Hosted Agentic Framework for Independent Research Groups

    [https://arxiv.org/abs/2610.23480](https://arxiv.org/abs/2610.23480)

    Munin是一个开源的自托管智能体框架，通过为本地部署的开放权重模型配备专门的阅读、搜索和计算工具，将其在领域文献问答上的准确率从0.387大幅提升至0.874，证明了智能体框架本身而非特定模型是系统效用的关键，同时保障了研究小组数据完全本地化。

    

    希望在自身文献基础上使用语言模型的研究小组面临一个选择：要么使用会接收其未发表作品的托管服务，要么使用能将数据保留在本地但使模型无法访问其语料库的自托管部署。Munin 是一个用于自托管科学助手的开源框架，它可以索引研究小组的语料库及其引文图谱，并将该小组能够部署的任何开放权重模型封装在一个配备了专门的阅读、搜索和计算工具的智能体框架（agent harness）中。我们声称，使该系统在领域文献上发挥作用的正是这个智能体框架，而非特定的模型，并且我们随软件一同发布了用于验证这一主张的评估套件。在199个成对的文献问题上，该智能体框架将基础模型的准确率提高了一倍以上，从0.387提升至0.874（+0.487，95%置信区间[0.407, 0.568]），而在所有测试的骨干模型上，朴素的“检索后回答”方法的得分均低于未经辅助的模型。在ha

    arXiv:2610.23480v1 Announce Type: new  Abstract: Research groups that want a language model grounded in their own literature face a choice between hosted services that receive their unpublished work and self-hosted deployments that keep the data local but give the model no access to the corpus. Munin is an open-source framework for a self-hosted scientific assistant that indexes a group's corpus and its citation graph, and wraps whichever open-weight model the group can serve in an agent harness with specialised reading, search, and computation tools. We claim that the harness, rather than the specific model, is what makes the system useful on domain literature, and we ship the evaluation suite that tests this claim with the software.   On 199 paired literature questions, the harness more than doubles the accuracy of the base model, from 0.387 to 0.874 (+0.487, 95% CI [0.407, 0.568]), while naive retrieve-then-answer scores below the unaided model on every backbone tested. Under the ha
    
[^31]: 舌形控制的低维动力学基础

    A low-dimensional dynamical basis for tongue shape control

    [https://arxiv.org/abs/2610.23456](https://arxiv.org/abs/2610.23456)

    该研究提出以整体舌形模式作为低维控制基础，并通过三维线性点吸引子的简单动力学模型成功拟合36名说话者的舌部超声波数据，为言语产生的低维发音控制提供了新证据。

    

    言语产生涉及对具有众多自由度的高维发音系统的控制，但一个被广泛接受的假设是，说话者通过一组较低维度的控制变量来驾驭这种复杂性。这些控制变量的常见候选包括单个发音器官（如舌体）和抽象的声道变量（如舌体收窄位置），但确定发音控制最合适的坐标系仍然是一个悬而未决的问题。在本研究中，我们提出了一种模型，其中整体性舌形模式代表一个简单动力学定律的潜在低维控制基础。我们首先在跨说话者的坐标空间中发现经验性的舌形模式，然后将该模型拟合到来自36名英国英语说话者的单元音超声波数据上。拟合到这些随时间变化模式的三维线性点吸引子恢复了舌……

    arXiv:2610.23456v1 Announce Type: new  Abstract: Speech production involves the control of a high-dimensional articulatory system with many degrees-of-freedom, but a widespread hypothesis is that speakers navigate this complexity using a set of lower-dimensional control variables. Common candidates for these control variables include individual articulators (e.g. tongue body) and abstract tract variables (e.g. tongue body constriction location), but identifying the most appropriate coordinate system for articulatory control remains an open question. In this study, we advance a model in which holistic tongue shape modes represent a potential low-dimensional control basis for a simple dynamical law. We first discover empirical tongue shape modes in a cross-speaker coordinate space and then fit the model to ultrasound data on monophthongal vowels from 36 speakers of British English. A three-dimensional linear point attractor fitted to these time-varying modes recovers the dynamics of tong
    
[^32]: HydraLLM：面向弹性文本与视觉-语言模型的自适应宽度切分

    HydraLLM: Adaptive Width Slicing for Elastic Text and Vision-Language Models

    [https://arxiv.org/abs/2610.23455](https://arxiv.org/abs/2610.23455)

    提出HydraLLM弹性架构，通过激活注意力头数量动态调整模型各层宽度，使单一参数集能够适配从数据中心GPU到边缘设备的多种部署场景，并通过弹性投影模块将该能力扩展到视觉-语言多模态推理。

    

    大语言模型（LLM）和大型视觉-语言模型（LVLM）正日益部署在从数据中心GPU到资源受限边缘设备的异构硬件上。其固定规模的架构限制了单一模型在不同计算、延迟和模型大小需求的部署环境中复用的能力。我们提出HydraLLM，一种基于弹性MQA（多查询注意力）的架构。其激活的注意力头数量决定了嵌入层、注意力投影、前馈层和输出头的有效宽度，从而在单一参数集内形成一个嵌套的子网络家族。我们引入了一套从静态到弹性的转换流水线，从预训练模型检查点出发，包含基于重要性的权重重排序以及结合知识蒸馏的随机子网络训练。我们通过一个弹性投影模块将该原则扩展到多模态推理，该模块的输出宽度跟随激活的文本宽度。HydraLLM覆盖了更广泛的……

    arXiv:2610.23455v1 Announce Type: new  Abstract: LLMs and LVLMs are increasingly deployed across heterogeneous hardware, ranging from data-center GPUs to resource-constrained edge devices. Their fixed-size architectures limit the ability to reuse a single model across deployment settings with different compute, latency, and model-size targets. We introduce HydraLLM, an elastic MQA-based architecture. Its number of active attention heads determines the effective widths of the embeddings, attention projections, feed-forward layers, and the output head, forming a nested family of subnetworks within a single parameter set. We introduce a static-to-elastic conversion pipeline that, starting from a pre-trained model checkpoint, includes importance-driven weight reordering and stochastic subnetwork training with knowledge distillation. We extend this principle to multimodal inference through an elastic projection module whose output width follows the active text width. HydraLLM covers a wider
    
[^33]: “你”并非有毒：大语言模型标注器如何改写网络的对话法则

    "You" Is Not Toxic: How LLM Labelers Rewrite the Web's Conversation Laws

    [https://arxiv.org/abs/2610.23446](https://arxiv.org/abs/2610.23446)

    研究发现大语言模型标注器会把第二人称称呼（“你”）误判为毒性，从而制造出人类标注中不存在的虚假“对话法则”（如回复越深毒性越高）。

    

    网络社区关于有毒对话的普遍认知——例如话题串越长、回复越深，毒性就越强——大多是使用谷歌的 Perspective API 测量得出的，而该 API 即将停止服务；下一代测量将由大语言模型（LLM）完成。我们发现，LLM 标注器会把第二人称称呼误判为毒性，仅这一个偏差就制造出了人类标注中并不存在的“法则”。在 125 万条带众包标注的新闻评论中，Claude Haiku 将 12% 的不含“你”的无毒评论标记为有毒，而对于含有四个及以上“你”的评论，这一比例高达 51%；单个众包标注员的比例则分别为 6% 和 10%。越深层的回复越常称呼某人，因此在 Haiku 的标注下，毒性随回复深度上升（每层增加 0.93 个百分点），而在众包标注下毒性反而略有下降（-0.12）。将“你”改写为第三人称可消除三分之二的虚假上升；仅让 LLM 标记与人群相同数量的评论并不能消除这一偏差，而向 Haiku 展示父级评……

    arXiv:2610.23446v1 Announce Type: cross  Abstract: What the Web community believes about toxic conversations, such as that toxicity rises in longer and deeper threads, was measured mostly with Google's Perspective API, which is shutting down; large language models (LLMs) will make the next measurements. We find that LLM labelers mistake second-person address for toxicity, and that this one bias manufactures laws that human labels do not show. On 1.25 million news comments with crowd labels, Claude Haiku flags 12% of the non-toxic comments that contain no "you" and 51% of those with four or more; a single crowd rater flags 6% and 10%. Deeper replies address someone more often, so under Haiku toxicity rises with reply depth (+0.93 percentage points per level), while under the crowd's labels it slightly declines (-0.12). Rewriting "you" in the third person removes two thirds of the false rise; flagging only as many comments as the crowd does not remove it, and showing Haiku the parent com
    
[^34]: ScoutRoute：基于内部语言模型信号的高效混合专家路由

    ScoutRoute: Efficient Mixture-of-Experts Routing Using Internal Language Model Signals

    [https://arxiv.org/abs/2610.23443](https://arxiv.org/abs/2610.23443)

    ScoutRoute 是一个轻量级后验路由框架，通过廉价的侦察模型单次前向传播提取隐状态与 logit 置信度等内部信号，结合域感知策略为混合专家系统高效选择专家，无需修改或联合训练底层语言模型。

    

    高效的专家路由是混合专家系统中的一个根本性挑战，尤其是对于独立训练的大型语言模型的异构集合。现有的路由方法通常依赖于联合训练的门控网络、基于提示的预测器或计算成本高昂的多模型推理，这限制了它们的灵活性和效率。本文提出了 ScoutRoute，一个轻量级的后验路由框架，它利用从一个廉价的侦察模型单次前向传播中提取的内部语言模型信号来预测专家能力。隐状态表示被用于通过轻量级逻辑回归探针来估计跨模型能力，同时将对数置信度度量纳入域感知路由策略中，从而在不修改底层语言模型的情况下实现高效的专家选择。ScoutRoute 在五个开放权重语言模型上进行了评估…

    arXiv:2610.23443v1 Announce Type: new  Abstract: Efficient expert routing is a fundamental challenge in Mixture-of-Experts (MoE) systems, particularly for heterogeneous collections of independently trained Large Language Mod els (LLMs). Existing routing approaches typically rely on jointly trained gating networks, prompt-based predictors, or computationally expensive multi-model inference, limiting their flexibility and efficiency. This paper presents ScoutRoute, a lightweight post-hoc routing framework that predicts expert competence using internal language model signals extracted from a single forward pass through an inexpensive scout model. Hidden-state representations are used to estimate cross-model competence through lightweight logistic probes, while logit-derived confidence measures are incorporated within a domain-aware routing strategy for efficient expert selec tion without modifying the underlying language models. ScoutRoute isevaluatedon fiveopen-weightlanguagemodels acros
    
[^35]: 话到嘴边：首词提示作为大语言模型回忆失败的行为探针

    On the Tip of the Tongue: First-Token Cueing as a Behavioral Probe of Recall Failures in LLMs

    [https://arxiv.org/abs/2610.23400](https://arxiv.org/abs/2610.23400)

    研究发现大语言模型的许多闭卷错误实为“首词回忆失败”，提出的首词提示行为探针仅需预填充正确答案的首词，即可在16个开源模型上挽回五分之一到一半的多词答案错误，并提升七种语言的问答准确率。

    

    大语言模型在闭卷设置下常常错误地回答事实性问题，这类错误通常被归因于知识的缺失。然而，模型往往已经编码了它们无法生成的答案。我们证明许多闭卷错误其实是首词回忆失败：模型错误地确定了答案的第一个词，但一旦给定正确的首词，它便能完成答案，并常常能识别出该答案是正确的。我们通过“首词提示”来研究这一现象，这是一种行为探针，通过在响应前预填充标准答案的首词来进行引导。模型的不确定性在答案首词处达到峰值，此时首词已能预测答案的正确性；并且在任意长度的标准答案前缀中，仅首词就能挽回最大比例的错误。在16个开源模型上，仅这一个词在英语问答中就能纠正多词答案上五分之一到二分之一的错误，并在多语言基准测试的全部七种语言中提升了准确率。大部分错误……

    arXiv:2610.23400v1 Announce Type: new  Abstract: Large language models often answer factual questions incorrectly in closed-book settings, and such errors are usually attributed to missing knowledge. Yet models often encode answers they fail to produce. We show that many closed-book errors are first-token recall failures: the model commits to a wrong first token, but once given the right one it completes the answer and often recognizes it as correct. We study this with First-Token Cueing, a behavioral probe that prefills the response with the first token of the gold answer. Model uncertainty peaks at the first answer token, where it already predicts correctness, and among gold prefixes of any length the first token alone accounts for the largest share of recovered errors. Across 16 open models, this one token corrects between a fifth and a half of the errors on multi-token answers in English QA and improves accuracy in all seven languages of a multilingual benchmark. Most of the errors
    
[^36]: 基于词汇与上下文证据的句子级主题建模

    Sentence-Level Topic Modeling with Lexical and Contextual Evidence

    [https://arxiv.org/abs/2610.23388](https://arxiv.org/abs/2610.23388)

    SentenceETM提出了一种句子级神经主题模型，通过联合利用词汇分布与句子嵌入上下文这两种证据来源，有效解决了主题建模中词级模型缺乏上下文、文档级模型忽视多主题共存问题的粒度困境。

    

    主题建模面临一个根本性的粒度困境：概率主题模型在词级别分配主题，而基于Transformer的方法则在文档级别进行操作。后者忽视了现实世界中的文档通常涵盖多个主题这一事实，而前者依赖于单个词语，但单个词语很少能提供足够识别主题所需的上下文。我们提出了SentenceETM，这是一种神经主题模型，通过联合利用词汇证据和上下文证据，为每个句子分配一个主题。每个主题同时由两种方式描述：一是通过词嵌入定义的词表上的分布，二是集中于主题特定低维子空间的句子嵌入上的高斯分布。句子的主题同时生成其词语和句子嵌入，并被精确地边缘化，使其后验分布能够结合这两种证据来源。文档的主题比例通过摊销变分推断来推断，而高斯……（原文摘要至此截断）

    arXiv:2610.23388v1 Announce Type: new  Abstract: Topic modeling faces a fundamental granularity dilemma: probabilistic topic models assign topics at the word level, whereas transformer-based methods operate at the document level. The latter overlooks that real-world documents typically cover multiple topics, whereas the former relies on individual words, which rarely provide sufficient context to identify them. We introduce SentenceETM, a neural topic model that assigns a topic to each sentence by jointly using lexical and contextual evidence. Each topic is described both by a distribution over the vocabulary, defined through word embeddings, and by a Gaussian over sentence embeddings concentrated on a topic-specific low-dimensional subspace. The topic of a sentence generates both its words and its embedding and is marginalized exactly, so that its posterior combines the two sources of evidence. Document proportions are inferred with amortized variational inference, while the Gaussian 
    
[^37]: 语言模型学会改进语言模型

    Language Models Learn to Improve Language Models

    [https://arxiv.org/abs/2610.23344](https://arxiv.org/abs/2610.23344)

    该论文提出LM2LM方法，通过强化学习训练大语言模型生成有效的语言模型改进提案，并利用廉价的代理反馈解决真实训练反馈成本高昂以及分布偏移下反馈不可靠这两大难题，为递归自我改进铺路。

    

    大语言模型（LLM）正被越来越多地用于自动化AI研究，包括语言模型（LM）本身的开发，这为递归自我改进提供了一条潜在路径。然而，将实验反馈转化为一种可学习的、能够跨模型提出有效改进的能力面临两大挑战。首先，使用真实反馈来训练这种能力代价高昂：强化学习（RL）需要评估数以万计的改进提案，而每个提案可能都需要一次完整的模型训练运行。即使每次运行仅需十个GPU小时，仅奖励收集这一项就会消耗数十万GPU小时，这促使人们寻求更便宜的替代反馈。其次，当提案分布发生偏移时，替代反馈可能变得不可靠，因为策略优化会偏向那些被高估的提案。为应对这些挑战，我们提出了LM2LM，该方法通过强化学习训练LLM来生成有效的LM改进提案。为了提供可扩展的反馈，我们使用了一种……（摘要在此处截断）

    arXiv:2610.23344v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to automate AI research, including language-model (LM) development, offering a potential path toward recursive self-improvement. However, turning experimental feedback into a learned ability to propose effective improvements across models faces two challenges. First, training such an ability with real feedback is expensive: reinforcement learning (RL) requires evaluating tens of thousands of proposals, each potentially requiring a full model-training run. Even at ten GPU-hours per run, reward collection alone would consume hundreds of thousands of GPU-hours, motivating a cheaper surrogate. Second, surrogate feedback can become unreliable as policy optimization favors overestimated proposals when the proposal distribution shifts. To address these challenges, we propose LM2LM, which trains LLMs with RL to generate effective LM improvement proposals. To provide scalable feedback, we use a
    
[^38]: 接口漂移下LLM智能体的鲁棒函数调用

    Robust Function Calling for LLM Agents Under Interface Drift

    [https://arxiv.org/abs/2610.23342](https://arxiv.org/abs/2610.23342)

    提出了一种无需训练的轻量级修复层 SchemaShield-Lite，通过模式校验、候选集重试和一次性受限修复，在接口漂移场景下显著恢复LLM智能体的函数调用准确率。

    

    工具增强的语言模型智能体依赖接口文本：工具描述、参数模式（schema）以及推理时展示的候选工具列表。即使底层工具语义保持不变，这些接口在部署后也经常发生变化，这给函数调用系统带来了实际的可靠性问题。我们将这种故障模式称为“接口漂移”并加以研究，评估一个轻量级、无需训练的修复层能否在不重新训练基础模型的情况下恢复鲁棒性。我们的方法 SchemaShield-Lite 结合了模式校验、针对缺失工具名称的候选集重试以及一次性受限修复。在 Qwen3.5-9B 上，该方法将 DICE-300 的准确率从 0.717 提升至 0.813，将 BFCL-200 的准确率从 0.580 提升至 0.805，分别恢复了 19/22 和 33/45 个由漂移引起的失败案例，且未观察到修复带来的负面影响。简单的重试机制无法解释这些提升。消融实验表明，候选集漂移（尤其是干扰性工具）是造成大部分观察到的失败的原因……

    arXiv:2610.23342v1 Announce Type: new  Abstract: Tool-augmented language-model agents rely on interface text: tool descriptions, argument schemas, and the candidate tool list shown at inference time. These interfaces often change after deployment even when the underlying tool semantics remain fixed, creating a practical reliability problem for function-calling systems. We study this failure mode as interface drift and evaluate whether a lightweight, training-free repair layer can recover robustness without retraining the base model. Our method, SchemaShield-Lite, combines schema validation, candidate-set retry for missing tool names, and one-shot constrained repair. On Qwen3.5-9B, it raises DICE-300 accuracy from 0.717 to 0.813 and BFCL-200 accuracy from 0.580 to 0.805, recovering 19/22 and 33/45 drift-induced failures with no observed repair harms. Naive retry does not explain the gains. Ablations show that candidate-set drift, especially distractor tools, accounts for most observed d
    
[^39]: 自反思思维链（SRCoT）：基于边缘大语言模型的闭环“检索—反思—修订”流水线实现有证据支撑的文本生成

    Self-Reflective Chain-of-Thought (SRCoT): Evidence-Based Text Generation via a Closed-Loop Retrieve--Reflect--Revise Pipeline on Edge LLMs

    [https://arxiv.org/abs/2610.23311](https://arxiv.org/abs/2610.23311)

    该论文提出自反思思维链（SRCoT）架构，通过闭环“检索—反思—修订”流程和带来源标签的多层存储机制，使边缘设备上的大语言模型能够生成有证据支撑、可追溯的文本。

    

    大语言模型（LLM）能够生成流畅的输出，但这些输出可能缺乏来源支持，或与本地运行约束相冲突。我们提出了自反思思维链（SRCoT），这是一种模块化架构，将生成过程组织为一个闭环的“检索—反思—修订”循环。SRCoT将环境状态、持久知识、积累经验和短期记忆分离到带有来源标签的存储中。BM25检索提供局部上下文；后续处理轮次会呈现中间草稿、缺失信息标记、后续查询以及可选的“论断—证据”映射，这些产物为检索、修订、澄清或弃答决策提供依据。我们通过在两个完全合成的场景中进行基于轨迹的定性案例研究来考察该架构：急救医疗服务和Munich-Sim（一个应急行动模拟环境）。在配备128 GB内存的Apple M4系列笔记本电脑上本地运行的原型展示了带有来源标签的数据……（摘要原文在此处截断）

    arXiv:2610.23311v1 Announce Type: new  Abstract: Large language models (LLMs) can generate fluent outputs that lack source support or conflict with local operational constraints. We introduce Self-Reflective Chain-of-Thought (SRCoT), a modular architecture that organizes generation as a closed Retrieve--Reflect--Revise loop. SRCoT separates environment state, persistent knowledge, accumulated experience, and short-term memory into provenance-labeled stores. BM25 retrieval supplies local context; subsequent passes expose intermediate drafts, missing-information markers, follow-up queries, and optional claim--evidence mappings. These artifacts inform retrieval, revision, clarification, or abstention decisions. We examine the architecture through qualitative, trace-based case studies in two fully synthetic settings: emergency medical services and Munich-Sim, an emergency-operations simulation. Local prototype runs on an Apple M4-family laptop with 128 GB memory illustrate source-labeled d
    
[^40]: 利用大语言模型词元概率从二语英语中推断母语背景

    Inferring L1 Backgrounds from L2 English with LLM Token Probabilities

    [https://arxiv.org/abs/2610.23309](https://arxiv.org/abs/2610.23309)

    提出一种基于大语言模型词元概率的母语识别方法，将文本视为指定候选母语的提示前缀的续写进行评分，显著优于直接分类方法，且 Llama 3.1 70B 在多个数据集上达到最先进性能。

    

    母语识别（NLI）旨在从非母语者的二语（L2）语言产出中推断其第一语言（L1）。我们研究了一种词元概率方法：将每个文本样本作为指定候选母语的前缀提示的续写进行评分，然后预测得分最高的候选母语。我们在三个数据集（TOEFL9、Write and Improve 2024 和 BULATS-L1）上使用 Llama 3.1 8B 和 70B 模型对该方法进行了评估，并将其性能与以往基于提示的分类方法进行比较。词元概率评分显著优于直接分类方法，尤其是在 8B 模型上，而 70B 模型则在所有数据集上均达到了最先进的性能。我们的结果表明，预训练大语言模型的词元概率中包含可用于从二语英语产出中区分母语背景的有用信息。

    arXiv:2610.23309v1 Announce Type: new  Abstract: Native language identification (NLI) infers a non-native speaker's first language (L1) from their second language (L2) production. We investigate a token probability approach that scores each text sample as a continuation of a prompt prefix specifying a candidate L1, then predicts the candidate with the highest score. We evaluate our approach using Llama 3.1 8B and 70B on three datasets: TOEFL9, Write and Improve 2024, and BULATS-L1. We compare its performance with previous prompting-based classification approaches. Token-probability scoring substantially improves on direct classification, particularly with the 8B model, while the 70B model achieves state-of-the-art performance across all datasets. Our results indicate that pretrained LLMs' token probabilities contain information useful for distinguishing L1 backgrounds from L2 English production.
    
[^41]: SimOPD：面向在线策略蒸馏的关系知识迁移

    SimOPD: Relational Knowledge Transfer for On-Policy Distillation

    [https://arxiv.org/abs/2610.23306](https://arxiv.org/abs/2610.23306)

    提出SimOPD方法，通过成对记忆库将教师内部表示转化为响应相似性分布，并让学生在自身表示空间内进行匹配，从而实现无需跨模型投影的在线策略蒸馏关系知识迁移。

    

    在线策略蒸馏（OPD）中的一个关键挑战是如何从教师知识中提取有效的监督信号。基于词元概率的监督可能无法充分利用教师内部知识，而直接的隐藏状态匹配则会迫使学生模型转向教师的特征空间，并使跨架构迁移变得复杂。受对比学习利用实例间关系的启发，我们提出SimOPD，通过响应相似性来迁移教师的内部知识。具体而言，SimOPD使用由学生生成响应构成的成对记忆库，从教师的内部表示中提取监督信号，将其表示为相似性分布。随后，学生在自己的表示空间内匹配这些分布，从而保留了表示自由度，并消除了对跨模型投影的需求。大量实验表明，SimOPD相对于最强基线将整体平均性能最多提升了2.6和3.1。

    arXiv:2610.23306v1 Announce Type: new  Abstract: A key challenge in on-policy distillation (OPD) is extracting effective supervision from teacher knowledge. Supervision from token probabilities may underutilize internal knowledge, while direct hidden-state matching forces the student to shift to the teacher's feature space and complicates cross-architecture transfer. Inspired by contrastive learning's use of relationships between instances, we propose \textbf{SimOPD} to transfer teacher internal knowledge through response similarities. Specifically, SimOPD uses paired memory banks of student-generated responses to extract supervision from the teacher's internal representations as similarity distributions. The student then matches these distributions within its own representation space, retaining representational freedom and removing the need for cross-model projection. Extensive experiments show that SimOPD improves the overall average over the strongest baselines by up to 2.6 and 3.1 
    
[^42]: Annotrace：一个用于智能体轨迹步骤级标注的工作台

    Annotrace: A Workbench for Step-Level Annotation of Agent Traces

    [https://arxiv.org/abs/2610.23284](https://arxiv.org/abs/2610.23284)

    Annotrace 是一个支持六种主流轨迹格式的智能体轨迹步骤级标注工作台，它将运行记录解析为有序嵌套的步骤，通过人机协作方式进行标注并衡量标注者间一致性，在多智能体问答语料上精确复现了此前人工标注的235个标注单元。

    

    每一项研究LLM智能体为何失败的工作，都始于对手工标注轨迹，而每一项这样的研究都会构建自己的一次性工具。可观测性平台能够记录智能体轨迹，却不提供结构化标注；标注工具可以标注文本，却没有“步骤”、任务“委派”或“工具调用”的概念，导致最小可标注的执行单元不明确。Annotrace 可以读取六种轨迹格式，包括 OpenTelemetry GenAI、LangSmith 和 Langfuse，并将每次运行转换为有序的、嵌套的步骤列表。标注者编写自己的标注手册；一个前沿模型会根据标注者自己轨迹的样本提出初始标注集，先行作答，并被问到与人类完全相同的问题，从而使标注成为一个人机协作的过程。Annotrace 能够衡量任意两个标注者（无论是人类还是AI）之间的一致性。在一个多智能体问答语料库上，它精确复现了此前一次人工标注工作的235个标注单元，并且两个模型……（摘要在此处被截断）

    arXiv:2610.23284v1 Announce Type: new  Abstract: Every study of why LLM agents fail begins by labelling traces by hand, and every such study builds its own throwaway tool. Observability platforms record agent traces but offer no structured labelling; annotation tools label text but have no notion of a step, a delegation or a tool call, leaving it unclear what the smallest unit of execution to label should be. Annotrace reads six trace formats, including OpenTelemetry GenAI, LangSmith and Langfuse, and turns each run into an ordered, nested list of steps. The annotator writes their own labelling codebook; a frontier model proposes a starting set from a sample of their own traces, answers first, and is asked exactly the questions the human is asked, making annotation a joint human-AI process. Annotrace measures agreement between any two annotators, human or AI. On a multi-agent question-answering corpus it reproduces a previous manual effort's 235 annotation units exactly, and two models
    
[^43]: 将模型计算过程言语化以用于思维链忠实性检测

    Verbalising Model's Computation for Chain-of-Thought Faithfulness Detection

    [https://arxiv.org/abs/2610.23275](https://arxiv.org/abs/2610.23275)

    VERITAS通过逐层雅可比投影将模型内部计算过程言语化，追踪推理锚点在各层间的产生、传递与显现过程，从而揭示推理偏离不忠实状态的机制，在思维链忠实性检测上显著超越LLM-as-Judge（平衡准确率提升28.4%）和激活探测方法（平均提升16.0%）。

    

    当前的思维链（CoT）验证方法从模型输出或静态激活中检测推理忠实性，但在解释不忠实性在生成过程中如何发展方面存在局限。我们提出VERITAS来将模型的计算过程言语化：即不同的推理锚点如何产生、如何跨层传递以及如何在输出中显现。VERITAS采用逐层雅可比投影（layer-wise Jacobian projection），在生成的每个CoT token处产生模型内部推理演化的言语化描述。通过沿推理轨迹提取的特征训练分类器，我们证明了这些结构化特征能够深刻揭示推理如何以及在何处偏离至不忠实状态。在来自FaithCoT-Bench和BonaFide的四个数据集上的实验表明，VERITAS相比LLM-as-Judge取得了显著的性能提升，平衡准确率提高了28.4%，相比激活探测（activation probing）方法平均提升了16.0%。

    arXiv:2610.23275v1 Announce Type: new  Abstract: Current Chain-of-Thought (CoT) verification methods detect reasoning faithfulness from model outputs or static activations, but are limited in explaining how unfaithfulness develops during generation. We introduce VERITAS to verbalise the model's computation: how different reasoning anchors emerge, transfer across layers, and surface in the output. VERITAS adopts the layer-wise Jacobian projection to produce a verbalised account of the model's internal reasoning evolution at each generated CoT token. By training a classifier on the extracted features along the traces, we show that these structured features provide insightful perspectives on how and where reasoning diverges toward unfaithfulness. Experiments on four datasets from FaithCoT-Bench and BonaFide show that VERITAS achieves a significant performance gain over LLM-as-Judge, with a 28.4% gain in balanced accuracy, and an average improvement of 16.0% over activation probing. Furthe
    
[^44]: 推理模型在自我指令下能够控制其思维链

    Reasoning Models Can Control Their Chains of Thought When They Tell Themselves To

    [https://arxiv.org/abs/2610.23257](https://arxiv.org/abs/2610.23257)

    推理模型并非天生无法控制其思维链，将约束作为自我指令置于CoT内部时，模型的可控性显著优于在用户提示中指定约束的方式。

    

    思维链（CoT）可控性，即推理模型控制其思维链属性的能力，对模型的有用性和可靠监督都具有重要意义。近期研究通过测量模型是否遵循用户提示中给出的CoT约束来评估这一能力，并报告各模型均表现出极低的可控性。我们认为，这种低可控性可能并非反映推理模型天生无法控制其CoT，而是源于在用户提示中简单直接地指定约束的方式。为验证这一点，我们提出了自我指令CoT可控性（Self-Instructed CoT Controllability）：一种互补的评估方法，它将约束作为自我指令置于模型的CoT内部，并测量模型遵循该约束的能力。我们表明，在自我指令条件下，模型实现的可控性显著高于标准用户指令下的水平。即使仅是提醒模型遵循用户约束的通用提示，也能大幅提升可控性。

    arXiv:2610.23257v1 Announce Type: new  Abstract: Chain-of-thought (CoT) controllability, the ability of reasoning models to control properties of their CoT, is consequential for both their usefulness and reliable oversight. Recent work evaluates this capability by measuring whether models follow CoT constraints given in the user prompt, and reports very low controllability across models. We argue that this low controllability may not reflect an inherent inability of reasoning models to control their CoT, but instead result from naively specifying constraints in the user prompt. To test this, we introduce Self-Instructed CoT Controllability: a complementary evaluation that places the constraint as a self-instruction inside the model's CoT and measures its ability to follow it. We show that, under self-instruction, models achieve significantly higher controllability than with standard user instructions. Even a generic reminder to follow the user's constraint substantially improves contro
    
[^45]: 采样、混合还是验证？不完美验证器下的测试时计算分配

    Sample, Mix, or Verify? Allocating Test-Time Compute under Imperfect Verifiers

    [https://arxiv.org/abs/2610.23255](https://arxiv.org/abs/2610.23255)

    该论文提出 SPLIT 方法，利用小型带标签试点实验和精确概率公式，在验证器不完美的情况下科学地分配测试时计算预算（采样数量、模型选择与验证开销）。

    

    arXiv:2610.23255v1 公告类型：cross 摘要：让语言模型和智能体更准确的一种常见方法是在推理阶段投入更多计算：生成大量候选解决方案，并提交验证器所偏好的那一个。成功需要满足两个条件：存在正确的候选解，且验证器能将其选出。前者随样本数量的增加而可预测地提升；后者却并非如此，因为真实的验证器（单元测试、多数投票、LLM 裁判、奖励模型）会接受错误答案、拒绝正确答案，并且会犯具有相关性的错误。在预算固定的情况下，从业者必须决定抽取多少样本、从哪些模型中抽取，以及在验证上投入多少资源，而目前这些选择都是凭经验法则做出的。我们提出 SPLIT，它基于一个小的带标签试点实验来做出这些决策。其核心是一个精确公式，用于计算在接受/拒绝和最高分选择机制下提交正确答案的概率，该公式考虑了任务难度的变化以及各模型之间共享的盲点……

    arXiv:2610.23255v1 Announce Type: cross  Abstract: A common way to make language models and agents more accurate is to spend more compute at inference time: generate many candidate solutions and submit the one a verifier prefers. Success then requires that some candidate is correct and that the verifier picks it. The first grows predictably with the number of samples; the second does not, because real verifiers (unit tests, majority votes, LLM judges, reward models) accept wrong answers, reject right ones, and make correlated mistakes. With a fixed budget, a practitioner must decide how many samples to draw, which models to draw them from, and how much to spend on verification, and today these choices are made by rules of thumb. We introduce SPLIT, which makes them from a small labelled pilot. It rests on an exact formula for the probability of submitting a correct answer under accept/reject and top-score selection that accounts for varying task difficulty, blind spots shared across mo
    
[^46]: 被忽视而非被拒绝：低提示注入成功率在小型工具调用智能体中隐藏了什么

    Ignored, Not Refused: What Low Prompt-Injection Success Rates Hide in Small Tool-Using Agents

    [https://arxiv.org/abs/2610.23254](https://arxiv.org/abs/2610.23254)

    该研究审计了33,000条智能体轨迹，揭示低攻击成功率主要源于智能体默默忽略而非拒绝提示注入攻击，且仅改动系统提示词即可使ASR在0.2%至43%间剧烈波动，并据此提出攻击-任务结果矩阵（ATOM），强调评估应同时报告攻击失败方式和用户任务完成度。

    

    工具调用智能体会读取不可信的外部内容，这使其暴露于间接提示注入（IPI）攻击之下，即隐藏的指令会劫持智能体的工具调用。基准测试通过攻击成功率（ASR）来衡量IPI安全性，该指标将每次失败的攻击都视为成功的防御。为检验这一假设，我们审计了来自十一个开源权重智能体（2-14B参数）在InjecAgent和AgentDojo基准上的33,000条轨迹。一个标签盲的LLM评判器显示，大多数失败的攻击是被忽视（76-80%）而非被拒绝（17-18%）。在模型权重完全相同的情况下，仅改变系统提示词就能使InjecAgent的ASR从0.2%变动至43%。我们提出的攻击-任务结果矩阵（ATOM）进一步显示，在21.7%的AgentDojo情景中，智能体虽然保持安全，却未能完成用户的任务。因此，低ASR主要反映了智能体在默默地忽略注入攻击，而非拒绝它们。评估应当报告攻击是如何失败的，以及用户的任务是否完成。

    arXiv:2610.23254v1 Announce Type: new  Abstract: Tool-using agents read untrusted external content, which exposes them to indirect prompt injection (IPI), where hidden instructions hijack the agent's tool calls. Benchmarks measure IPI security by the attack success rate (ASR), which treats every failed attack as a successful defense. To test this assumption, we audit 33,000 trajectories from eleven open-weight agents (2-14B parameters) on InjecAgent and AgentDojo. A label-blind LLM judge shows that most failed attacks are ignored (76-80%) rather than refused (17-18%). With identical model weights, changing only the system prompt shifts InjecAgent ASR from 0.2% to 43%. Our Attack-Task Outcome Matrix (ATOM) further shows that in 21.7% of AgentDojo episodes, the agent stayed safe but did not complete the user's task. A low ASR therefore mostly reflects agents silently ignoring injections, not refusing them. Evaluations should report how attacks fail and whether the user's task is complete
    
[^47]: 谁为高效推理买单？跨任务与语言的基于置信度的小模型到大模型LLM路由

    Who Pays for Efficient Inference? Confidence-Based Small-to-Large LLM Routing Across Tasks and Languages

    [https://arxiv.org/abs/2610.23249](https://arxiv.org/abs/2610.23249)

    该论文对基于置信度的小模型到大模型LLM路由进行了迄今最系统的跨语言、跨任务评估（72种配置、42,288条预测），证明置信度信号能够有效识别值得升级到大模型的困难样本，从而在多语言场景下实现质量与推理成本的更好权衡。

    

    小模型到大模型的语言模型路由可以通过将困难输入升级到大模型、同时让较简单的输入留在小模型上处理，从而降低推理成本。然而，其成功取决于能否识别出哪些样本能够从升级中受益，而置信度在多语言任务中的可靠性仍不清楚。我们对基于置信度的路由进行了系统性评估，涵盖三个模型家族、四种语言和三种任务类型，共包括72种模型、数据集、语言与规模的配置，以及42,288条预测。我们将全局置信度路由与随机路由、等额配额置信度路由以及预言机上界进行了比较。结果显示，置信度能够在Belebele阅读理解和SIB-200主题分类数据集上识别出有益的升级，平均纠错检测AUC分别为0.612和0.705。在50%的升级率下，相比随机路由，它在Belebele上提升1.40–4.67个百分点，在SIB-200上提升2.54–5.63个百分点。

    arXiv:2610.23249v1 Announce Type: new  Abstract: Small-to-large language model routing can reduce inference cost by escalating difficult inputs while retaining easier ones on a smaller model. Its success, however, depends on identifying examples that benefit from escalation, and the reliability of confidence across multilingual tasks remains unclear. We present a systematic evaluation of confidence-based routing across three model families, four languages, and three task types, covering 72 model, dataset, language, and scale configurations and 42,288 predictions. We compare global confidence routing with random routing, equal-quota confidence routing, and an oracle upper bound. Confidence identifies beneficial escalations on the Belebele reading comprehension and SIB-200 topic classification datasets, with mean correction-detection AUCs of 0.612 and 0.705. At 50% escalation, it improves over random routing by 1.40--4.67 percentage points on Belebele and 2.54--5.63 points on SIB-200. In
    
[^48]: 多思考还是多阅读？共享上下文下推理的Token预算理论

    Think More or Read More? A Token-Budget Theory of Reasoning with Shared Context

    [https://arxiv.org/abs/2610.23246](https://arxiv.org/abs/2610.23246)

    提出token预算理论形式化固定上下文窗口下“阅读他人输出”与“自主推理”的权衡，通过允许错误相关的Condorcet陪审团模型推导出多智能体辩论最优智能体数量的闭式解，且仅需单智能体测量即可估计最优群体规模。

    

    推理系统越来越多地通过在模型输入中纳入额外上下文来改进答案，从多智能体辩论（智能体阅读同伴的回答）到自我改进（模型重新审视自己的草稿）皆是如此。在这些系统中，通信会消耗上下文，因为一个智能体的输出会成为另一个智能体的输入。我们研究了token阅读与token生成共享同一固定token窗口的情形，因此每花费一个token用于阅读，就少一个token可用于推理。我们提出了一个**token预算理论**来形式化这种权衡，将准确率建模为阅读后剩余推理预算的函数，并采用允许错误相关的Condorcet陪审团模型来聚合投票。对于多智能体辩论，该理论给出了最优智能体数量的闭式预测。这一数量可从单智能体的测量中估计得出，从而无需实际运行辩论即可确定最优群体规模。

    arXiv:2610.23246v1 Announce Type: new  Abstract: Reasoning systems increasingly improve answers by incorporating additional context into the model's input, from multi-agent debate, where agents read their peers' responses, to self-refinement, where a model revisits its own drafts. In these systems, communication consumes context as one agent's output becomes another's input. We study the regime where token reading and generation draw on one fixed token window, so every token spent reading is a token that cannot be spent reasoning. We develop a \textbf{token-budget theory} that formalizes this trade-off, modeling accuracy as a function of the reasoning budget remaining after reading and aggregating votes with a Condorcet jury model that permits correlated errors. For multi-agent debate, the theory gives a closed-form prediction for the optimal number of agents. This quantity is estimated from single-agent measurements, allowing the optimal group size to be determined without running deb
    
[^49]: MGE-PO：面向数学推理的熵感知信用分配方法

    MGE-PO: Entropy-Aware Credit Assignment for Mathematical Reasoning

    [https://arxiv.org/abs/2610.23245](https://arxiv.org/abs/2610.23245)

    MGE-PO提出了一种结合验证正确性与生成熵的四象限优势塑形方法，通过对低熵成功给予更大奖励、高熵失败给予更大惩罚来改进GRPO的信用分配机制，显著提升了数学推理性能。

    

    GRPO对同一问题内获得相同奖励的回复赋予相同的组相对优势，从而忽略了这些回复的生成不确定性。我们提出MGE-PO，其核心是四象限（4Q）优势塑形方法，将经验证的答案正确性与rollout生成熵相结合。正确性决定额外校正项的符号，而熵控制其幅度：低熵的成功样本获得更大的奖励，高熵的失败样本受到更大的惩罚。一个基于熵的在线问题课程机制利用缓存的rollout测量结果来支持这一信用分配规则。在Qwen3-8B-Base上经过八个训练周期后的最终检查点处，在三种策略优化设置中加入4Q，在六个基准测试的macro avg@32上带来了3.23-3.69%的绝对提升，在macro pass@32上带来了4.24-11.06%的绝对提升。在Qwen2.5-7B-Base上，完整配置将四个基准测试的macro avg@32从41.5%提升至45.4%。

    arXiv:2610.23245v1 Announce Type: cross  Abstract: GRPO assigns the same group-relative advantage to equally rewarded responses within a question, leaving their generation uncertainty unused. We introduce MGE-PO, centered on four-quadrant (4Q) advantage shaping that combines verified correctness with rollout generation entropy. Correctness determines the sign of an additional correction, while entropy controls its magnitude: lower-entropy successes receive larger bonuses and higher-entropy failures receive larger penalties. An online entropy-based question curriculum supports this credit rule using cached rollout measurements. At the final checkpoint after eight training epochs on Qwen3-8B-Base, adding 4Q across three policy optimization settings yields absolute gains of 3.23-3.69% in six-benchmark macro avg@32 and 4.24-11.06% in macro pass@32. On Qwen2.5-7B-Base, the complete configuration raises four-benchmark macro avg@32 from 41.5% to 45.4%.
    
[^50]: MultiAgent-OPD：基于在线策略蒸馏的层次化多智能体系统端到端训练

    MultiAgent-OPD: End-to-End Training of Hierarchical Multi-Agent Systems with On-Policy Distillation

    [https://arxiv.org/abs/2610.23234](https://arxiv.org/abs/2610.23234)

    提出MultiAgent-OPD方法，通过将在线策略蒸馏与强化学习相结合，利用冻结的教师模型为搜索者提供局部过程监督，实现了层次化多智能体系统的端到端联合训练。

    

    在层次化多智能体系统中，由领导者分配任务、子智能体执行任务。联合训练领导者和子智能体十分困难，因为最终答案只能为子智能体提供间接反馈。虽然大语言模型评判器可以为子智能体的表现打分，但这些分数依赖于评判器自身对子智能体回答质量的评估。我们提出了MultiAgent-OPD，该方法将在线策略蒸馏与强化学习相结合，端到端地联合训练共享参数的领导者-搜索者-回答者团队。我们首先使用答案正确性在搜索者-回答者工作流中训练一个教师模型。在团队训练过程中，冻结的教师模型对学生生成的搜索和总结token进行评分，为搜索者提供局部过程监督，而领导者从团队成功中学习，回答者从候选答案的正确性中学习。在七个数据集上，训练后的系统达到了38.68%的macro avg@4，超越了提示的链式方法（原文在此处截断）。

    arXiv:2610.23234v1 Announce Type: new  Abstract: In a hierarchical multi-agent system, a Leader assigns tasks and subagents carry them out. Jointly training the Leader and the subagents is difficult because a final answer gives only indirect feedback to the subagents. An LLM judge can score the performance of the subagents, but these scores depend on the judge's own assessment of the quality of subagents' responses. We present MultiAgent-OPD, which combines on-policy distillation with reinforcement learning to jointly train a shared-parameter Leader-Searcher-Answerer team end to end. We first train a Teacher in a Searcher-Answerer workflow using answer correctness. During team training, the frozen Teacher scores student-generated search and summary tokens to provide local process supervision for the Searchers, while the Leader learns from team success and the Answerer from candidate correctness. Across seven datasets, the trained system reaches 38.68% macro avg@4, exceeding Prompted Cr
    
[^51]: ReLaDrift：基于漂移的循环语言生成用于测试时扩展

    ReLaDrift: Recurrent Language Generation via Drifting for Test-Time Scaling

    [https://arxiv.org/abs/2610.23225](https://arxiv.org/abs/2610.23225)

    提出ReLaDrift，利用共享漂移模型在连续潜在空间中对生成结果进行循环迭代细化，首次使单步漂移模型具备测试时计算量与生成质量之间的可调节权衡能力。

    

    漂移模型只需一次函数求值即可生成一个样本，这使其在快速并行语言生成方面极具吸引力，但其单步形式不支持迭代细化。我们提出基于漂移的循环语言生成，它使用一个共享的漂移模型从噪声生成初始草稿，并在连续潜在空间中对其进行迭代细化。我们通过随机采样循环轮数、且仅在最后一轮应用漂移损失来训练这种循环机制，同时阻断穿过前几轮的梯度，使得每一步只需对一轮进行反向传播。与单步漂移模型不同（后者即使迭代也不会提升准确率），ReLaDrift 能够像扩散语言模型和流语言模型那样，在测试时计算量与生成质量之间进行权衡。在 Sudoku Hard 和 GSM8K 上，它相较于这些迭代生成方法取得了具有竞争力或更优的性能。

    arXiv:2610.23225v1 Announce Type: new  Abstract: Drifting Models generate a sample with a single function evaluation, making them attractive for fast parallel language generation, but their one-step formulation does not support iterative refinement. We propose Recurrent Language Generation via Drifting (ReLaDrift), which uses a shared Drifting Model to generate an initial draft from noise and iteratively refine it in a continuous latent space. We train this recurrence by sampling the number of rounds and applying the drifting loss only at the final round, stopping gradients through preceding rounds so that only one round requires backpropagation. Unlike the one-step Drifting Model, whose accuracy does not improve when iterated, ReLaDrift enables a trade-off between test-time computation and generation quality, as diffusion and flow language models do. On Sudoku Hard and GSM8K, it achieves competitive or superior performance compared with these iterative-generation approaches across a r
    
[^52]: MAS-Bai：面向白文白语的语言学知识驱动的自适应多智能体翻译

    MAS-Bai: Linguistically Informed Adaptive Multi-Agent Translation for Bowen-Script Bai

    [https://arxiv.org/abs/2610.23205](https://arxiv.org/abs/2610.23205)

    MAS-Bai提出了一种融合语言学知识的自适应多智能体翻译框架，通过在轻量级检索-翻译路径与包含词源、句法智能体的文献学路径之间动态路由，并结合循环一致性可靠性信号，显著提升了白文白语-汉语的翻译质量并缓解了直译幻觉问题。

    

    大型语言模型（LLM）对于书写系统与高资源预训练先验相冲突的极低资源语言仍然不可靠。我们研究了白文书写系统下的白语-汉语翻译问题，其中汉字经常被语音借用而非语义借用，导致严重的直译幻觉。我们提出了MAS-Bai，一个面向白文白语-汉语翻译的语言学知识驱动的自适应多智能体框架。MAS-Bai将每个输入路由到轻量级的检索-翻译路径，或包含词源智能体与句法智能体的更深层文献学路径，并利用轻量级反向翻译器的循环一致性作为推理时的可靠性信号。我们构建了一个包含6,536个句子的白语-汉语语料库，并在1,536个句子的测试集上进行评估。MAS-Bai取得了16.23 BLEU和62.15 COMET的成绩，相比标准RAG提升了8.11 BLEU和6.85 COMET。前向LoRA、专家消融……（摘要截断）

    arXiv:2610.23205v1 Announce Type: new  Abstract: Large Language Models (LLMs) remain unreliable for extremely low-resource languages whose writing systems conflict with high-resource pretraining priors. We study Bai-Chinese translation in the Bowen script, where Chinese characters are frequently borrowed phonetically rather than semantically, causing severe literal-translation hallucinations. We propose MAS-Bai, a linguistically informed adaptive multi-agent framework for Bowen-script Bai-Chinese translation. MAS-Bai routes each input through either a lightweight retrieval-translation path or a deeper philological path with etymological and syntactic agents, and uses cycle-consistency with a lightweight reverse translator as an inference-time reliability signal. We construct a 6,536-sentence Bai-Chinese corpus and evaluate on a 1,536-sentence test set. MAS-Bai achieves 16.23 BLEU and 62.15 COMET, improving over Standard RAG by 8.11 BLEU and 6.85 COMET. Forward LoRA, expert ablations an
    
[^53]: AMEND：基于隔离评论员的约束引导多智能体文本解毒方法

    AMEND: Constraint-Guided Multi-Agent Text Detoxification with Isolated Critics

    [https://arxiv.org/abs/2610.23183](https://arxiv.org/abs/2610.23183)

    AMEND是一个推理时框架，通过角色条件化候选生成、隔离的多维评论员批评和约束优先选择机制来解耦文本解毒过程，其轻量版AMEND Lite仅在残留毒性时进行最多一轮修复，兼顾了去毒效果与计算效率。

    

    文本解毒需要在去除毒性、保留语义和保持流畅性三者之间取得平衡。现有方法通常优化一个无约束的标量目标——这使得安全评分可能掩盖语义损失——或依赖开放式迭代辩论，容易导致语义漂移和高昂的计算成本。我们提出AMEND，一个推理时框架，它将解毒过程解耦为基于角色条件的候选生成、隔离的多维度批评以及约束优先的选择。AMEND使用角色盲的评论员来评估多样化的候选方案，并在按效用排序之前严格执行语义和流畅性可行性约束。为了最大化效率，AMEND Lite仅当残留毒性片段存在时，才将后续搜索限制为最多一轮针对性修复。在ParaDetox数据集上，对多样化角色进行一轮选择相比最佳单一角色将联合评估指标提升了+0.0508。在一个冻结的确认测试集上，AMEND Lite实

    arXiv:2610.23183v1 Announce Type: new  Abstract: Text detoxification requires balancing toxicity removal, semantic preservation, and fluency. Existing methods typically optimize an unconstrained scalar objective---enabling safety scores to mask semantic loss---or rely on open-ended iterative debate prone to semantic drift and high computational cost. We propose AMEND, an inference-time framework that decouples detoxification into role-conditioned proposal generation, isolated multidimensional criticism, and constraint-first selection. AMEND evaluates diverse candidates with role-blind critics and enforces strict semantic and fluency feasibility before ranking by utility. To maximize efficiency, AMEND Lite restricts subsequent search to at most one targeted repair round only when residual toxic spans remain. On ParaDetox, one-round selection over diverse roles improves the joint evaluation metric by $+0.0508$ over the best single role. On a frozen confirmation test set, AMEND Lite achie
    
[^54]: 大语言模型在暗影中做什么：从Unembedding暗空间读取上下文依赖性

    What LLMs Do in the Shadows: Reading Context Reliance from the Unembedding Dark Space

    [https://arxiv.org/abs/2610.23182](https://arxiv.org/abs/2610.23182)

    该论文提出DarkLens方法，通过将隐藏状态投影到与unembedding矩阵弱组合的“暗空间”并计算暗物质比率（DMR），无需注意力图、额外前向传播或辅助探针等开销，即可直接量化大语言模型的上下文依赖性，实现轻量级忠实性幻觉检测。

    

    检测大语言模型（LLM）中的忠实性幻觉对于其在高风险领域的安全部署至关重要。近期方法通过对比上下文知识与参数化知识的利用情况来解决这一问题，然而测量上下文依赖性通常需要显著的额外开销，例如具体化注意力图、运行额外的前向传播或训练辅助探针。在这项工作中，我们展示了上下文依赖性可以通过将隐藏状态投影到“暗空间”——即与unembedding矩阵弱组合的方向——来进行估计。除了其公认的作为初始token注意力汇聚点的功能外，我们证明该子空间还积极参与上下文处理，并且其在隐藏状态范数中的占比——被称为暗物质比率——提供了上下文利用程度的直接评分。将DMR与参数化知识评分相结合，我们提出了DarkLens，一个严格在……上运行的轻量级忠实性检测器（摘要在此处截断）。

    arXiv:2610.23182v1 Announce Type: new  Abstract: Detecting faithfulness hallucinations in large language models (LLMs) is crucial for their safe deployment in high-stakes domains. Recent approaches address this by contrasting contextual against parametric knowledge utilization, yet measuring context reliance typically demands noticeable overhead, such as materializing attention maps, running additional forward passes, or training auxiliary probes. In this work, we show that context reliance can be estimated by projecting hidden states onto the dark space - directions that compose weakly with the unembedding matrix. Beyond its recognized function as an initial-token attention sink, we show that this subspace actively mediates context processing, and that its share of hidden-state norm - termed the Dark Matter Ratio (DMR) - provides a direct score of context utilization. Pairing DMR with a parametric knowledge score yields DarkLens, a lightweight faithfulness detector operating strictly 
    
[^55]: 面向长时程智能体强化学习的树感知后验蒸馏

    Tree-Aware Posterior Distillation for Long-Horizon Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.23181](https://arxiv.org/abs/2610.23181)

    本文提出TreeOPSD，一种无需评论家的树感知后验自蒸馏框架，通过将树结构回放批次转化为有限的后验自蒸馏目标，同时解决长时程智能体强化学习中的稀疏信用分配与策略陈旧问题。

    

    具有可验证奖励的强化学习已成为大语言模型的一种可扩展的后训练范式。然而，在长时程交互中训练智能体面临两个挑战：仅使用终端奖励使得回合级别的信用分配稀疏且方差很高，而在固定的回放批次上进行顺序小批量更新会引入策略陈旧问题，因为当前策略会逐渐偏离行为策略。现有方法通过学习评论家、过程奖励模型或在线策略蒸馏来获得更密集的信用信号，但它们依赖于价值网络、步骤级标注或更强的外部教师模型，从而增加了训练复杂性。与此同时，重要性采样和裁剪可以缓解策略漂移，但它们并未指定一个有限的目标来阻止顺序更新过度。在这项工作中，我们提出了TreeOPSD，一个无评论家、树感知的后验自蒸馏框架，用于长时程智能体……

    arXiv:2610.23181v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards has become a scalable post-training paradigm for large language models. However, training agents over long-horizon interactions presents two challenges: using only terminal rewards makes turn-level credit assignment sparse and high-variance, while sequential mini-batch updates on a fixed rollout batch introduce policy staleness as the current policy drifts from the behavior policy. Existing approaches obtain denser credit via learned critics, process reward models, or on-policy distillation, but they rely on a value network, step-level annotations, or a stronger external teacher that increase training complexity. Meanwhile, importance sampling and clipping mitigate policy drift, but they do not specify a finite target that stops sequential updates from overshooting. In this work, we introduce TreeOPSD, a critic-free, tree-aware posterior self-distillation framework for long-horizon agentic
    
[^56]: 语音脑机接口需要更大的模型吗？超越规模化的神经解码反思

    Do Speech BCIs Need Larger Models? Rethinking Neural Decoding beyond Scaling

    [https://arxiv.org/abs/2610.23175](https://arxiv.org/abs/2610.23175)

    本文提出两阶段框架BEST，通过分层中间表示和紧凑的语音导向解码器直接将神经活动映射为文本，证明在数据稀缺的语音脑机接口任务中，无需依赖大型模型即可实现高效、低成本且泛化性强的神经解码。

    

    近年来，语音脑机接口（BCI）日益依赖大型模型和复杂的多阶段流水线来进行神经解码。然而，在语音BCI领域中，训练数据稀缺、噪声大且采集成本高昂，模型规模化的收益仍不明确。本工作重新审视了一个根本问题：大型模型对于精确的神经解码真的必要吗？我们提出了BEST（Brain-Encoding-Speech-Text），一个用于高效神经到文本解码的两阶段框架。BEST不依赖大型生成模型或以音素为中心的流水线，而是引入了分层中间表示和一个紧凑的面向语音的解码器，直接将神经活动映射为文本。这种设计在保留关键结构的同时显著降低了计算成本，使其能够在低资源环境中实现稳健的泛化，并在资源受限的条件下进行实际部署。在Brain-to-Text '24和'25基准测试中，BEST……

    arXiv:2610.23175v1 Announce Type: new  Abstract: Recent speech brain-computer interfaces (BCIs) increasingly rely on large models and complex multi-stage pipelines for neural decoding. However, in speech BCI, where training data are scarce, noisy, and expensive to collect, the benefits of model scaling remain unclear. This work revisits a fundamental question: are large models truly necessary for accurate neural decoding? We propose BEST (Brain-Encoding-Speech-Text), a two-stage framework for efficient neural-to-text decoding. Instead of relying on large generative models or phoneme-centric pipelines, BEST introduces hierarchical intermediate representations and a compact speech-oriented decoder that directly maps neural activity to text. This design preserves essential structure while significantly reducing computational cost, enabling robust generalization in low-resource settings and practical deployment under constrained resources. On the Brain-to-Text '24 and '25 benchmarks, BEST 
    
[^57]: Chinese Decision（中文决策）：面向单次前向概率决策的语言模型系列

    Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making

    [https://arxiv.org/abs/2610.23170](https://arxiv.org/abs/2610.23170)

    Chinese Decision 是一系列中文语言模型，通过单次前向传播直接对候选选项的 logits 进行归一化得到概率分布，无需自回归生成，在医学、法律、金融等领域的决策任务上实现了高准确率与良好校准。

    

    许多基于语言的应用需要对有限决策集合输出概率分布，而非开放式回答。我们提出了 Chinese Decision（中文决策），一个面向单次前向概率决策的中文语言模型系列。给定上下文、指令和候选描述，模型在单次前向传播中提取末尾位置的候选字母 logits，并在有效选项上进行归一化，无需自回归式答案生成。统一的共享接口支持布尔判断、类别选择和序数评分。我们报告了一个已完成的 0.8B 模型，该模型使用分布交叉熵在 1304.8 万个决策上训练；此外还报告了经过额外 RLCD 目标训练的初步 2B 检查点。在包含 307,900 个决策的完整 CJ-Bench 上，未经校准的 0.8B 模型仅使用单一检查点，在通用领域达到 72.24% 的准确率，在医学、法律和金融领域平均达到 76.98% 的准确率，整体期望校准误差为 1.07%。

    arXiv:2610.23170v1 Announce Type: new  Abstract: Many language-based applications require probabilities over a finite set of decisions rather than open-ended responses. We present Chinese Decision, a family of Chinese-language models for single-pass probabilistic decision-making. Given context, an instruction, and candidate descriptions, the model extracts final-position candidate-letter logits in one forward pass and normalizes them over valid options, without autoregressive answer generation. A shared interface supports Boolean judgments, categorical choices, and ordinal scores. We report a completed 0.8B model trained with distribution cross-entropy on 13.048 million decisions and preliminary 2B checkpoints trained with an additional RLCD objective. On the full CJ-Bench of 307,900 decisions, the uncalibrated 0.8B model achieves 72.24% general-domain accuracy and 76.98% average accuracy across medicine, law, and finance using a single checkpoint, with 1.07% overall expected calibrati
    
[^58]: 用自然语言解释国际象棋与围棋

    Explaining Chess and Go in Natural Language

    [https://arxiv.org/abs/2610.23166](https://arxiv.org/abs/2610.23166)

    本文提出了基于引擎评估的解释性指标和无需训练的BinProp方法，使语言模型能够发现可复用的棋类概念并以自然语言解释着法，平均得分相比直接提示提高近两倍，且该方法可泛化至围棋、跳棋和将棋等多种棋类。

    

    现代国际象棋引擎的棋力远超人类，其选择着法背后的思路往往令人类难以理解。我们提出了一种基于国际象棋引擎评估的自动化指标，用于评估棋局解释的质量。利用该指标对最先进的语言模型进行评估后发现，它们的表现出奇地差，往往依赖于对棋盘的表面观察，而非稳健、可迁移的国际象棋概念。随后，我们提出了BinProp，这是一种无需训练的方法，能够使同样的模型发现可复用的国际象棋概念，并用它们以自然语言解释着法。BinProp提升了对所有受测模型的可解释性，与直接提示相比，平均得分几乎提高了两倍。除了国际象棋之外，同一方法无需额外调整即可在围棋、西洋跳棋和将棋中发现有用的概念。

    arXiv:2610.23166v1 Announce Type: cross  Abstract: Modern chess engines play far beyond human ability, often choosing moves whose underlying ideas are difficult for humans to understand. We propose an automatic metric for evaluating chess explanations, grounded in chess-engine evaluations. Evaluating state-of-the-art language models using this metric, we find that they perform surprisingly poorly, often relying on surface-level board observations rather than robust, transferable chess concepts. We then introduce BinProp, a training-free method that enables these same models to discover reusable chess concepts and use them to explain moves in natural language. BinProp improves explainability for every model tested, nearly tripling the average score compared with direct prompting. Beyond chess, the same method discovers useful concepts in Go, draughts, and shogi without additional tuning.
    
[^59]: Cross-Centaur：基于语言模型对跨心理学任务的个体决策建模

    Cross-Centaur: Modeling Individual Decision-Making Across Psychological Tasks with Language Models

    [https://arxiv.org/abs/2610.23157](https://arxiv.org/abs/2610.23157)

    本文提出包含591名参与者跨12个心理学任务关联行为数据的Cross-Psych-101数据集，并训练Cross-Centaur模型，首次建模跨任务的个体层面决策规律，在任务内与跨任务预测上均超越现成大语言模型和Centaur。

    

    大语言模型（LLMs）正日益被用作人类行为的计算模型。诸如Centaur等近期模型通过在心理学实验的逐试次决策数据上微调大语言模型，推进了这一目标。然而，现有的大规模行为数据集并未将同一参与者在不同任务中的数据关联起来，因此这些模型学到的是群体层面的规律，而未被训练去建模跨任务的个体层面规律。我们提出了Cross-Psych-101数据集，涵盖591名参与者在12个Bandit（赌博机）、Tradeoff（权衡）、Structure（结构）和Social（社交）任务中的261,551个有效决策。我们训练了Cross-Centaur，使其根据目标任务描述、以及与同一参与者的源任务行为相配对的源任务描述，来预测该参与者在目标任务中的行为。在留出的参与者上，Cross-Centaur无论在任务内还是跨任务场景下均优于现成的大语言模型和Centaur。探测分析表明其隐藏（原文摘要在此处截断）

    arXiv:2610.23157v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used as computational models of human behavior. Recent models such as Centaur have advanced this goal by fine-tuning LLMs on trial-by-trial decisions from psychological experiments. However, existing large-scale behavioral datasets do not link the same participants across different tasks, so these models learn population-level regularities but are not trained to model individual-level regularities across tasks. We introduce Cross-Psych-101, a dataset of 591 participants and 261{,}551 valid decisions across 12 Bandit, Tradeoff, Structure, and Social tasks. We train Cross-Centaur to predict a participant's target-task actions from a target-task description and the source-task description paired with the same participant's source-task behavior. Cross-Centaur outperforms off-the-shelf LLMs and Centaur on held-out participants, both within and across tasks. Probing analyses suggest that its hidden
    
[^60]: AI能进行访谈吗？AI主导与人类主导的聊天访谈对比

    Can AI interview? Comparing AI-led to Human-led Chat Interviews

    [https://arxiv.org/abs/2610.23149](https://arxiv.org/abs/2610.23149)

    本文首次通过随机对照实验直接比较了AI主导与人类主导的聊天访谈数据质量，发现AI访谈者在访谈时长相当的情况下提问数量是人类的两倍多，并能引出多出约65%的受访者文本。

    

    大语言模型聊天机器人已被提议作为“AI访谈者”，用于大规模收集定性访谈数据，这有望弥合社会科学长期存在的问卷调查广度与定性访谈深度之间的鸿沟。本文首次对AI主导访谈与人类主导访谈的数据质量进行了直接的实证比较。在一项随机实验（N = 41）中，大学生通过同步文本聊天接受访谈，访谈者要么是受过训练的非专业访谈员，要么是本地托管的AI系统；我们评估了曝光度、具体性和相关性三个维度。两种条件下的访谈时长中位数相近，但AI访谈者每次访谈提出的问题数量是人类的两倍多，且每次访谈引出的受访者文本量多出约65%。人类访谈者条件下，针对每个问题的回答更长，但在经过多重检验校正后这一差异不再显著；只有AI条件下的参与者出现了跳过问题的情况。

    arXiv:2610.23149v1 Announce Type: cross  Abstract: Large language model chatbots have been proposed as AI-interviewers for collecting qualitative interview data at scale, potentially bridging social science's longstanding divide between the breadth of surveys and the depth of qualitative interviews. This paper provides the first direct empirical comparison of data quality in AI-led and human-led interviews. In a randomized experiment (N = 41), university students were interviewed via synchronous text chat by either trained non-expert interviewers or a locally hosted AI system; we evaluate exposure, specificity, and relevance. Median interview duration was similar across conditions, but AI-interviewers asked more than twice as many questions per interview and elicited roughly 65 percent more respondent text per interview. Per-question responses were longer with human interviewers, though not after multiple-testing correction; only AI-condition participants skipped questions. Responses w
    
[^61]: LED：通过前瞻编辑改进掩码扩散语言模型

    LED: Improving Masked Diffusion Language Models by Lookahead Editing

    [https://arxiv.org/abs/2610.23122](https://arxiv.org/abs/2610.23122)

    提出前瞻编辑方法（LED），通过在掩码扩散解码中引入可重写的提议状态，使掩码扩散语言模型在不牺牲其易优化性的前提下获得迭代修正的表达能力。

    

    扩散大语言模型已成为自回归语言模型的一种有前景的替代方案，能够通过并行解码实现非自回归的文本生成。现有的 dLLMs 大致可分为两大类：掩码扩散模型（MDMs）和均匀扩散模型（UDMs）。在本工作中，我们从理论上证明：MDMs 由于其基于掩码的转移调度而更易于优化，但其计算表达力较低；而 UDMs 支持迭代式的状态修正并具有更强的表达力，但训练难度显著更高。这一权衡引出了一个自然的问题：我们能否在不放弃 MDMs 简单易优化的转移调度的前提下，获得迭代修正所带来的表达力？基于这一分析，我们提出了前瞻编辑（Lookahead Editing，LED），它通过引入一个可重写的提议状态来增强掩码扩散解码。在每个解码步骤中，模型会执行若干前瞻……

    arXiv:2610.23122v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive language models, enabling non-autoregressive text generation through parallel decoding. Existing dLLMs can be broadly categorized into two families: masked diffusion models (MDMs) and uniform diffusion models (UDMs) In this work, we theoretically show that MDMs are easier to optimize due to their mask-based transition schedule but have lower computational expressivity, whereas UDMs enable iterative state revision and greater expressivity but are substantially harder to train. This trade-off raises a natural question: can we obtain the expressivity of iterative revision without giving up the simple, easy-to-optimize transition schedule of MDMs? Motivated by this analysis, we propose Lookahead Editing (\ours{}), which augments masked diffusion decoding with a rewritable proposal state. At each decoding step, the model performs several lookah
    
[^62]: 解耦语言对工具准入与检索后安全性的影响

    Disentangling Language Effects on Tool Admission and Post-Retrieval Safety

    [https://arxiv.org/abs/2610.23093](https://arxiv.org/abs/2610.23093)

    该研究通过独立控制请求、检索页面和回答语言的受控实验，首次识别出“跨语言处理中的安全损失”（SLIP）现象——非英语有害请求使工具准入率从46.9%升至67.4%，且要求以页面语言回答会进一步增加危害性，从而将语言对LLM智能体工具准入与检索后安全的影响分离出来。

    

    大语言模型智能体越来越多地在多语言环境中使用检索工具，但其安全评估仍然以英语为中心。目前尚不清楚语言究竟影响工具准入、检索后生成，还是两者兼有。我们在受控的URL检索设置中，对十个模型和七种语言独立改变请求语言、检索页面语言和所需回答语言，识别出一种称为“跨语言处理中的安全损失”（SLIP）的现象。非英语有害请求使平均工具准入率从46.9%提升至67.4%，且所有模型的平均变化均为正向。该模式在自主检索设置中对所测试的全部四个模型同样成立。在四个闭源权重模型中，要求以页面语言回答使得匹配工具调用的总体危害性从50.3%上升至57.4%，主要体现在中文和印地语上。而在推理设置相当的八个模型中，仅对页面进行翻译并不会产生一致的危害增加。

    arXiv:2610.23093v1 Announce Type: new  Abstract: LLM agents increasingly use retrieval tools across languages, yet their safety evaluation remains English-centric. It remains unclear whether language affects tool admission, post-retrieval generation, or both. We independently vary request, retrieved-page, and required response languages in a controlled URL-retrieval setting across ten models and seven languages. We identify Safety Loss through Interlingual Processing (SLIP). Non-English harmful requests increase mean tool admission from 46.9% to 67.4%, with a positive mean change in every model. This pattern also persists in an autonomous retrieval setup across all four models tested in this setting. Requiring answers in the page language increases pooled harmfulness on matched tool calls from 50.3% to 57.4% across four closed-weight models, mainly in Chinese and Hindi. Translating pages alone produces no consistent harm increase in eight models with comparable inference settings. Auto
    
[^63]: 检索开始之前：大语言模型智能体的安全与工具使用决策

    Before Retrieval Begins: Safety and Tool-Use Decisions in LLM Agents

    [https://arxiv.org/abs/2610.23071](https://arxiv.org/abs/2610.23071)

    该论文首次系统研究了大语言模型智能体在工具调用发生之前“是否调用工具”这一前置安全决策，发现有害请求被接受后的平均有害率高达74.5%，并通过因子化干预揭示“请求操作”组件是支配该决策的关键因素，同时发现书面拒绝在行为与表征上与该决策相互耦合。

    

    大型语言模型越来越多地充当调用外部工具的智能体，然而大多数工具安全研究只考察工具调用之后发生的事情。是否调用工具这一前置决策在很大程度上未被研究，却可能影响下游的安全性。在七个开放权重模型中，有害请求被接受后的未加权模型平均有害率为74.5%。为了确定是什么因素支配这一决策，我们对输入请求施加了因子化干预。在我们构建的生成数据中，将“请求操作”组件中和后产生了最大的接受率对比（19个百分点）。从该对比中估计出的线性方向影响了七个模型中四个模型在保留集上的接受行为；而该方向在某些模型中对操作已被中和的请求仍有效应，说明它并非特定于某种操作的机制。随后我们检验了书面拒绝在行为层面和表征层面是否与同一决策相耦合。书面拒绝伴随着33.（摘要原文在此处截断）

    arXiv:2610.23071v1 Announce Type: new  Abstract: Large language models increasingly act as agents that call external tools, yet most tool-safety research examines only what follows a tool call. The prior decision of whether to call the tool remains largely unexamined and can influence downstream safety. Across the seven open-weight models, the unweighted model-average harm rate after admitted harmful requests is 74.5%. To identify what governs this decision, we apply a factorial intervention to the input request. Within our generated construction, neutralizing the requested-operation component produces the largest admission contrast (19 pp). A linear direction estimated from this contrast influences held-out admission in four of seven models; effects on operation-neutralized requests in some models show that it is not an operation-specific mechanism. We then test whether written refusal is behaviorally and representationally coupled to the same decision. Written refusal accompanies 33.
    
[^64]: ParetoChat：一个基于大语言模型的交互式多目标优化界面

    ParetoChat: An LLM-Based Interface for Interactive Multi-Objective Optimization

    [https://arxiv.org/abs/2610.23062](https://arxiv.org/abs/2610.23062)

    本文提出ParetoChat，一个基于大语言模型的多智能体交互界面，能够将决策者的自然语言偏好自动转化为多目标优化的标量化参数，并通过迭代反馈循环实现交互式多目标优化。

    

    多目标优化模型通常部署在决策者（DM）需要平衡多个相互冲突准则的场景中。然而在实践中，利益相关者必须将他们在不同准则之间的定性偏好转化为优化方法所需的数值权重、优先级或参考点，而这些参数往往难以指定和解读。本文提出了ParetoChat，一个基于大语言模型（LLM）的专家智能体界面，用于交互式多目标优化。该智能体将自然语言的用户输入转化为结构化的标量化参数，调用优化系统，展示候选解，并迭代地获取用户反馈。我们描述了一种多智能体架构，该架构将输入处理、多准则决策（MCDM）专业知识、解的展示和反馈处理分离开来。随后，我们利用基于LLM的决策者智能体和基准问题进行了一项仿真研究……

    arXiv:2610.23062v1 Announce Type: new  Abstract: Multi-objective optimization models are often deployed in settings where decision makers (DMs) must balance some conflicting criteria. In practice, however, stakeholders must translate their qualitative preferences between different criteria into numerical weights, priorities, or reference points required by optimization methods, which can be difficult to specify and interpret. This paper proposes \emph{ParetoChat}, an LLM (large language model)-based expert-agent interface for interactive multi-objective optimization. The agent translates natural-language user input into structured scalarization parameters, invokes an optimization system, presents candidate solutions, and iteratively elicits feedback. We describe a multi-agent architecture that separates input processing, multi-criteria decision-making (MCDM) expertise, solution presentation, and feedback processing. We then present a simulation study using an LLM-based DM agent and ben
    
[^65]: 同一个人，不同的建议：利用重复的风险承受度回答审计大语言模型金融建议

    Same Person, Different Advice: Auditing LLM Financial Advice with Repeated Risk-Tolerance Responses

    [https://arxiv.org/abs/2610.23060](https://arxiv.org/abs/2610.23060)

    研究对七个主流大语言模型的金融建议进行审计，发现当客户风险承受度回答发生现实水平的变动时，模型推荐的股票配置比例会大幅波动（10-35个百分点），且其敏感度与被告知回答非常可靠时相当，揭示了LLM金融建议面对人们不一致风险自评时的稳健性缺陷。

    

    AI金融顾问主要根据客户对单一风险承受度问题的回答来设定其股票配置比例，然而人们对这一问题的回答前后并不一致：在美国全国代表性面板调查中，41%的家庭在两年后选择了不同的答案；在实验室重测中，三分之一的人在六个月后在0-10量表上的评分变动了两个或更多点。来自五家提供商的七个大语言模型为800个真实家庭档案提供了一轮对话的资产配置建议，其中仅风险回答发生了变化。模型推荐的股票比例平均变动了10至16个百分点，当风险回答改变时变动达24至35个百分点，而同一提示词重复询问时仅变动0.6至4.2个百分点；在实验室重测中，仅评分变化就使建议变动了11至18个百分点。在没有任何附加说明的情况下，模型对风险回答的敏感程度与被告知该回答非常可靠时相当（两个Claude模型的敏感度较低）。如果这种被指示的响应方式就是各模型将风险态度映射为建议的机制，那么不一致……

    arXiv:2610.23060v1 Announce Type: new  Abstract: AI financial advisers set a client's stock allocation largely from one stated-risk question, yet people do not answer it the same way twice: in a national US panel, 41% of households chose a different answer two years later, and in a laboratory retest a third of people moved two or more points on a 0-10 scale after six months. Seven large language models from five providers gave one-turn allocation advice for 800 real household profiles in which only the risk answer changed. The recommended stock share moved by 10 to 16 percentage points on average, and by 24 to 35 when the answer changed, against 0.6 to 4.2 points for the same prompt asked twice; in the laboratory retest the rating alone moved advice by 11 to 18 points. Without any note, the models respond to the answer about as strongly as when told that it is very reliable (less so for the two Claude models). If that instructed response is how each model maps attitude to advice, disco
    
[^66]: ECHO：在基于优化的越狱攻击中用回复危害性替代合规性代理

    ECHO: Replacing the Compliance Proxy with Response Harmfulness in Optimization-Based Jailbreak Attack

    [https://arxiv.org/abs/2610.23030](https://arxiv.org/abs/2610.23030)

    ECHO提出用基于回复实际危害性的连续评分（由1.7B的HarmSig回归模型提供）替代传统的合规性代理指标，并结合词级扰动算子与关键词锚定束搜索，在严格黑盒条件下显著提升了针对大语言模型的越狱攻击效果。

    

    尽管经过了广泛的安全对齐，大语言模型仍然容易受到越狱攻击。现有的基于优化的攻击共享一个单一目标：最大化受害模型合规性的token级代理指标——例如肯定前缀的可能性——该指标是在提示词上而非解码后的回复上计算的。这种代理指标无法区分谨慎的模糊回应与完全可操作的有害答案，使攻击搜索对模型实际输出的内容视而不见。我们提出ECHO（评估条件化危害性优化），它用来自HarmSig的回复级连续危害性分数替代合规性代理，HarmSig是一个基于实际有害能力的1.7B回归模型。ECHO在严格的黑盒威胁模型下，将HarmSig与五个词级扰动算子以及带有全局top-m记忆的关键词锚定束搜索相结合。在AdvBench上的五个开源权重受害模型上，ECHO实现了高达0.898的StrongREJECT平均分数和89……（摘要原文在此处截断）

    arXiv:2610.23030v1 Announce Type: cross  Abstract: Despite extensive safety alignment, large language models remain vulnerable to jailbreak attacks. Existing optimization-based attacks share a single objective: maximize a token-level proxy of victim compliance---such as the likelihood of an affirmative prefix---computed on the prompt rather than the decoded reply. This proxy cannot distinguish a cautious hedge from a fully actionable harmful answer, leaving the search blind to what the model actually says. We propose ECHO (Evaluation-Conditioned Harmfulness Optimization), which replaces the compliance proxy with response-level, continuous harmfulness scores from HarmSig, a 1.7B regressor grounded in actual harmful capability. ECHO couples HarmSig with five word-level perturbation operators and Keyword-Anchored Beam Search with global top-m memory, under a strict black-box threat model. Across five open-weight victims on AdvBench, ECHO achieves up to 0.898 StrongREJECT mean score and 89
    
[^67]: 通过合并对齐前后的模型来控制大语言模型的多样性-质量权衡

    Controlling the Diversity-Quality Trade-Off of LLMs by Merging Models Before and After Alignment

    [https://arxiv.org/abs/2610.22999](https://arxiv.org/abs/2610.22999)

    提出一种无需额外训练的深度感知 FrankenMerging 方法，通过将对齐模型的低层与基础模型的高层进行合并，可灵活控制大语言模型输出中词汇与语义的多样性及质量之间的权衡。

    

    对齐被认为会改变大语言模型（LLM）的输出特性。基础模型往往生成更加多样的输出，而对齐模型则倾向于产生更高质量的输出。两者之间的适当平衡取决于具体任务。现有方法要么需要代价高昂的微调，要么像温度采样那样主要提升词汇多样性而非语义多样性。我们提出了一种深度感知的 FrankenMerging 方法，用于在无需额外训练的情况下控制对齐模型与其基础模型之间的多样性-质量权衡。该方法基于这样一个假设：词汇多样性与模型的较低层关系更密切，而语义多样性与较高层关系更密切，因此将一个对齐模型的较低层与基础模型的较高层相结合。实验表明，改变从每个模型中选取的层数可以同时调节多样性和质量。在大多数情况下，该方法……

    arXiv:2610.22999v1 Announce Type: new  Abstract: Alignment is known to change the output characteristics of large language models (LLMs). Base models tend to generate more diverse outputs, whereas aligned models tend to produce higher-quality outputs. The appropriate balance between the two depends on the task. Existing methods either require costly fine-tuning or, like temperature sampling, mainly improve lexical diversity rather than semantic diversity. We introduce a depth-aware FrankenMerging approach for controlling the diversity--quality trade-off between an aligned model and its base model without additional training. This approach combines the lower layers of an aligned model with the higher layers of its base model based on the hypothesis that lexical diversity is more closely associated with lower layers and semantic diversity with higher layers. Experiments showed that varying the number of layers taken from each model adjusts both diversity and quality. In most cases, the m
    
[^68]: Prism注意力：面向高效大语言模型推理的提案精细化索引共享机制

    Prism Attention: Proposal-Refined Index Sharing Mechanism for Efficient LLMs Inference

    [https://arxiv.org/abs/2610.22985](https://arxiv.org/abs/2610.22985)

    提出Prism Attention机制，利用层间显著令牌偏移高度局部化的特性，以“提案-精细化”的方式在层间共享并修正稀疏注意力索引，在避免静态索引复用导致注意力损失的同时，大幅提升大语言模型长上下文推理效率。

    

    大语言模型（LLMs）中的长上下文解码日益受到每一步自回归生成时对不断增长的键值（KV）缓存进行注意力计算所需的内存与计算量的瓶颈制约。尽管令牌级稀疏注意力通过将计算限制在少量相关的历史令牌上来减轻这一负担，但其实际收益往往因上下文增长时需要反复重新计算各层稀疏索引而被抵消。一种常见的补救措施是仅在指定的锚点层计算稀疏索引并跨层共享，但这种静态复用方式忽略了注意力模式随层的演化，导致复用的索引逐渐偏离各层的最优选择，造成显著的注意力损失。通过从注意力动态的视角重新审视跨层稀疏性，我们发现层间显著令牌的偏移具有高度局部性：尽管精确的 top-$\kappa$ 索引在各层之间有所不同……

    arXiv:2610.22985v1 Announce Type: cross  Abstract: Long-context decoding in large language models (LLMs) is increasingly bottlenecked by the memory and computation required to attend over an ever-growing Key-Value (KV) cache at each autoregressive step. Although token-level sparse attention reduces this burden by restricting computation to a small subset of relevant past tokens, its practical gains are often offset by repeatedly recomputing layer-wise sparse indices as the context grows. A common remedy is to share sparse indices across layers by computing them only at designated anchor layers, but such static reuse ignores the layer-wise evolution of attention patterns, causing the reused indices to progressively deviate from the layer-specific optima and incur substantial attention loss. By revisiting cross-layer sparsity from the perspective of attention dynamics, we find that inter-layer salient token shift is highly localized: although the exact top-$\kappa$ indices differ across 
    
[^69]: 自我进化智能体能够修复失败，但难以从失败中泛化

    Self-Evolving Agents Can Repair Failures but Struggle to Generalize from Them

    [https://arxiv.org/abs/2610.22984](https://arxiv.org/abs/2610.22984)

    研究发现，仅从失败经验中学习的自我进化智能体能够修复之前失败的任务，但这种经验难以泛化到未见过的任务上，且智能体常常误解失败原因。

    

    经验驱动的自我进化为大语言模型（LLM）智能体通过交互持续改进提供了一条有前景的路径。然而，现有方法通常依赖成功的轨迹或外部评估信号，因此当这些正面锚点不可用时，智能体能否实现进化尚不清楚。我们通过“冷启动仅失败进化”（CFOE）来研究这一问题，即智能体在没有任何积累经验的情况下开始，仅从失败的交互中学习，且没有特权指导。跨越三种自我进化范式、多个LLM骨干模型和四个交互环境的实验，我们揭示了一个一致的“修复-泛化差距”：从失败中获取的经验可以帮助修复之前失败的任务，但很少能在未见过的任务上带来持续的改进。进一步分析表明，智能体经常误解失败原因，而即使是经过验证正确的经验也难以迁移。放松CFOE约束后我们发现，验证（摘要在此处被截断）。

    arXiv:2610.22984v1 Announce Type: new  Abstract: Experience-driven self-evolution offers a promising path for large language model (LLM) agents to continually improve through interaction. Yet existing approaches often rely on successful trajectories or external evaluation signals, leaving unclear whether agents can evolve when such positive anchors are unavailable. We study this question through Cold-Start Failure-Only Evolution (CFOE), where agents start without accumulated experience and learn only from failed interactions without privileged guidance. Across three self-evolving paradigms, multiple LLM backbones, and four interactive environments, we uncover a consistent repair-generalization gap: failure-derived experience can help repair previously failed tasks, but rarely yields sustained improvements on unseen tasks. Further analysis shows that agents often misinterpret failures, while even verified-correct experience remains difficult to transfer. Relaxing CFOE reveals that verif
    
[^70]: KiranaBench：语言模型能经营一家社区小店吗？

    KiranaBench: Can a Language Model Run a Neighbourhood Shop?

    [https://arxiv.org/abs/2610.22980](https://arxiv.org/abs/2610.22980)

    论文提出多智能体印度杂货店模拟基准KiranaBench，评估发现四个开源LLM的经营利润均远低于固定规则店主基线，最佳模型仅获得其八分之一的利润。

    

    大型语言模型（LLM）智能体正越来越多地在长时程任务上接受评估，其中包括自主经营一门生意。针对这一场景的现有基准几乎全部是英语的、以西方零售环境为背景的，并且是单智能体形式，由智能体直接观察业务运作。我们研究了这些结果如何迁移到多智能体的印度语境中。我们提出了KiranaBench——一个模拟的印度杂货店，其中四个角色（经理、收银员、帮手和记账员）分别由独立的LLM智能体扮演。与顾客和供应商的交互以英语、印地语或印地英语进行，产品需求则根据印度的全国家庭消费支出调查进行校准。在评估四个开源权重LLM后，我们发现所有模型都远逊于一个基于固定规则的店主基线：表现最好的模型仅获得基线利润的八分之一，还有两个模型的收益甚至低于一家无人管理的店铺。此外，性能表现也持续……

    arXiv:2610.22980v1 Announce Type: new  Abstract: Large language model (LLM) agents are increasingly evaluated on long-horizon tasks, including autonomously running a business. Existing benchmarks for this setting are almost exclusively English-language, Western in retail context, and single-agent, with the agent observing the business directly. We study how these results transfer to a multi-agent Indic setting. We introduce KiranaBench, a simulated Indian general store in which four roles (manager, cashier, helper, and bookkeeper) are each played by a separate LLM agent. Interactions with customers and suppliers occur in English, Hindi, or Hinglish, and product demand is calibrated to India's national Household Consumption Expenditure Survey. Evaluating four open-weight LLMs, we find that all fall far short of a fixed-rule shopkeeper baseline: the best model earns only one-eighth of the baseline's profit, and two models finish below an unmanaged shop. Performance is also consistently l
    
[^71]: 什么塑造了大型语言模型在对话过程中与大脑的对齐？来自语言产生与理解的洞察

    What Shapes Brain Alignment of Large Language Models during Dialogue? Insights from Language Production and Comprehension

    [https://arxiv.org/abs/2610.22961](https://arxiv.org/abs/2610.22961)

    本研究利用对话中的fMRI数据揭示了大型语言模型在语言产生与语言理解两种模式下呈现截然不同的大脑对齐模式，且对齐程度与模型表征的内在维度以及可解释的语言和交际特征密切相关。

    

    大型语言模型（LLM）与大脑的对齐为研究大型语言模型与人类大脑在语言处理过程中的表征相似性提供了一种途径。以往的研究主要集中在被动的语言理解上，而对于对话过程中这种对齐的了解却很少，而在对话中，说话者会同时进行语言产生和语言理解。在本研究中，我们利用对话过程中收集的fMRI数据，考察了大型语言模型各层和各脑区在语言产生与理解过程中的大脑对齐情况。我们评估了这种对齐与大型语言模型表征内在维度的关联，以及若干可解释的语言学和交际特征的贡献。我们的结果揭示了语言产生与理解之间截然不同的大脑对齐模式，以及与每种处理模式相关脑区中内在维度的强关联。此外，我们还发现语言学特征……

    arXiv:2610.22961v1 Announce Type: new  Abstract: Brain alignment of large language models (LLMs) offers a way to investigate representational similarities between LLMs and the human brain during language processing. While previous studies have mainly focused on passive language comprehension, little is known about such alignment during dialogue, in which speakers engage in both language production and comprehension. In this study, using fMRI data collected during dialogue, we examine brain alignment during language production and comprehension across LLM layers and brain regions. We assess its association with the intrinsic dimension (ID) of LLM representations and the contributions of several interpretable linguistic and communicative features. Our results reveal distinct patterns of brain alignment between production and comprehension, as well as strong associations with ID in brain regions associated with each processing mode. Furthermore, we show that the contributions of linguisti
    
[^72]: SAFE-Mem：具有细粒度权限与治理感知检索的安全智能体记忆

    SAFE-Mem: Secure Agentic Memory with Fine-Grained Authority and Governance-Aware Retrieval

    [https://arxiv.org/abs/2610.22959](https://arxiv.org/abs/2610.22959)

    提出SAFE-Mem框架，通过为记忆声明标注细粒度治理标签并结合治理感知检索，在保障LLM智能体记忆安全性的同时保留合法的记忆使用。

    

    大语言模型（LLM）智能体日益依赖持久化记忆来支持长期推理与个性化交互。然而，当从外部来源收集的不可信声明（即原子性陈述）随后被当作经授权的偏好、策略或权限使用时，这些信息可能带来安全风险。现有的防御手段通常依赖于基于内容的过滤或粗粒度的来源追踪，它们要么无法阻止未经授权的记忆使用，要么会屏蔽合法信息，从而牺牲了记忆的效用。为了解决这些局限，我们提出了SAFE-Mem，这是一种细粒度的记忆治理框架，能够在保留合法记忆使用的同时提升安全性。首先，SAFE-Mem为每条声明标注三种治理标签（即类型、权限和范围），以捕获其来源和允许的使用方式。其次，治理感知检索将查询需求与声明权限相匹配，以阻止未经授权的记忆使用（原文摘要到此中断）。

    arXiv:2610.22959v1 Announce Type: cross  Abstract: Large language model (LLM) agents increasingly rely on persistent memory to support long-term reasoning and personalized interactions. However, information collected from external sources can introduce security risks when untrusted claims (i.e., atomic statements) are later treated as authorized preferences, policies, or permissions. Existing defenses typically rely on content-based filtering or coarse-grained provenance tracking, which either fail to prevent unauthorized memory use or block legitimate information, sacrificing memory utility. To address these limitations, we propose SAFE-Mem, a fine-grained memory governance framework that improves security while preserving legitimate memory use. First, SAFE-Mem annotates each claim with three governance labels (i.e., kind, authority, and scope) to capture its provenance and permitted use. Second, governance-aware retrieval matches query requirements with claim permissions to block una
    
[^73]: 识别叫吃与评估先手：面向基于语言的围棋智能体的受控评估任务

    Recognizing Atari and Assessing Sente: Controlled Evaluation Tasks for Language-Based Go Agents

    [https://arxiv.org/abs/2610.22953](https://arxiv.org/abs/2610.22953)

    该论文构建了以围棋“叫吃”及其与“先手”关系为核心的受控评估集，发现经术语和棋评微调的围棋语言模型LoGos难以识别叫吃关系变化并评估其战略含义，为检验语言型围棋智能体能否将领域术语与精确棋盘关系关联起来提供了评估框架。

    

    领域术语为游戏状态和战略关系提供了简洁的描述，但下棋语言模型能否准确识别这些概念并将其与着法评估相关联，目前仍不清楚。我们构建了一个以围棋中“叫吃”及其与“先手”的关系为核心的受控评估集，考察局部棋盘变化下的术语识别能力以及不同战略条件下的着法评估能力。我们在LoGos——一个围棋语言模型上进行了测试，该模型经过了术语和棋评监督的微调。结果显示，该模型难以识别叫吃关系的变化并评估其战略含义。即使其着法偏好与参考评估相矛盾，其术语预测仍与着法偏好相关联。我们的受控测试提供了一个框架，用于检验基于语言的围棋智能体是否能够将领域术语与精确的棋盘关系及战略评估联系起来。

    arXiv:2610.22953v1 Announce Type: new  Abstract: Domain terminology provides concise descriptions of game states and strategic relationships, but whether game-playing language models can accurately recognize these concepts and relate them to move evaluation remains unclear. We construct a controlled evaluation set centered on atari and its relationship to sente in Go, examining terminology recognition under local board changes and move evaluation under different strategic conditions. We conduct our tests on LoGos, a Go language model, after fine-tuning it with terminology and commentary supervision. The model struggles to recognize changes in the atari relationship and assess its strategic implications. Its terminology predictions are associated with move preferences even when those preferences contradict the reference evaluation. Our controlled tests provide a framework for examining whether language-based Go agents connect domain terminology with precise board relationships and strat
    
[^74]: 将语言模型对齐至反事实不变性

    Aligning Language Models Towards Counterfactual Invariance

    [https://arxiv.org/abs/2610.22942](https://arxiv.org/abs/2610.22942)

    本文提出一种基于群体级奖励函数的策略梯度技术，使语言模型在对齐过程中实现反事实不变性，解决标准强化学习对齐方法在提升整体回复质量的同时放大跨人口群体反事实偏差的问题。

    

    提示特征的微小变化可能会显著改变大语言模型输出的语义、语气和上下文假设。这些特征可能涉及人口统计学等群体变量或其他敏感的潜在属性。当人口统计信息在因果上与任务无关时，模型对此类输入变量的敏感性是不可取的，但这些信息可能仍与生成语义连贯的回复相关。这使得在保持模型效用的同时实现反事实不变性变得困难。为了研究这一问题，我们首先证明了标准模型对齐方法（如强化学习）在提升整体回复质量的同时，会加大不同人口群体之间回复质量的差异，最终放大了所谓的反事实偏差。为解决这一问题，我们开发了一种策略梯度技术，其中包含一个作用于匹配对的群体级奖励函数。

    arXiv:2610.22942v1 Announce Type: cross  Abstract: Small variations in prompt characteristics can meaningfully alter the semantics, tone, and contextual assumptions of large language model outputs. Such characteristics can involve group variables such as demographics or other sensitive latent attributes. Sensitivity to such input variables is undesirable when demographic information is causally irrelevant to the task, but might still be relevant for generation of semantically coherent responses. This makes the task of enforcing counterfactual invariance while preserving model utility difficult. To study this problem, we first show that standard model alignment recipes (such as reinforcement learning) increase differences in response quality across demographics, while improving overall response quality, ultimately amplifying what is known as counterfactual bias. To address this, we develop a policy gradient technique involving a group-level reward function that operates on matched pairs
    
[^75]: 超越表层线索：鲁棒的LLM生成韩文文本检测

    Beyond Surface Cues: Robust LLM-Generated Korean Text Detection

    [https://arxiv.org/abs/2610.22940](https://arxiv.org/abs/2610.22940)

    提出偏差受控的韩文对抗性基准KoLing与基于理论语言学、在多个语言学层面建模韩语特性的可解释检测器KoLingProbe，在LLM生成与修改韩文文本检测任务上大幅超越强基线。

    

    随着大语言模型（LLM）生成的文本日益流畅，对LLM生成内容进行可靠检测的需求大幅增长。然而，现有研究仍然高度以英语为中心，韩语在很大程度上未被充分探索。此外，当前对韩文文本检测的评估往往依赖表层线索，导致对检测性能的估计过于乐观。为解决这些局限性，我们提出了KoLing，一个偏差受控的对抗性基准，包含50.9k个实例，在保留语义内容的同时抑制表层捷径，并分别评估LLM生成文本与LLM修改文本的检测。我们进一步提出KoLingProbe，一个基于理论语言学的可解释检测器，在多个语言学层面对韩语特有属性进行建模。实验表明，KoLingProbe在所有检测任务上均大幅超越强基线，同时保持可解释性。

    arXiv:2610.22940v1 Announce Type: new  Abstract: As large language models (LLMs) generate increasingly fluent text, the need for reliable detection of LLM-generated content has grown substantially. However, existing research remains heavily English-centric, leaving Korean largely underexplored. Moreover, current evaluations of Korean text detection often rely on superficial cues, leading to overly optimistic estimates of detection performance. To address these limitations, we introduce \textsc{KoLing}, a bias-controlled adversarial benchmark comprising 50.9k instances that suppresses superficial shortcuts while preserving semantic content and separately evaluates LLM-generated and LLM-modified text detection. We further propose KoLingProbe, an interpretable detector grounded in theoretical linguistics that models Korean-specific properties across multiple linguistic levels. Experiments show that KoLingProbe substantially outperforms strong baselines on all detection tasks while remaini
    
[^76]: 多种偏见，同一回路：后训练微调预训练的线索路由机制

    Many Biases, One Circuit: Post-Training Tunes a Pretrained Cue-Routing Mechanism

    [https://arxiv.org/abs/2610.22937](https://arxiv.org/abs/2610.22937)

    多种偏见（如谄媚性）共享预训练模型中已存在的同一注意力头回路——助手微调在聊天格式中激活它，强化学习进一步加强它，表明跟随线索源于预训练与调优的副作用，而非对齐本身的问题。

    

    当提示中包含一个线索（即指向错误选项的暗示，例如“我认为答案是B”）时，语言模型往往会放弃正确答案。谄媚性及相关偏见都表现出这种失败，但它们通常被分开研究，并被归咎于对齐问题。我们通过因果方法检验了这一观点，在两个后训练流水线的各个阶段追踪这些偏见背后的注意力头。在对大多数模型的多项选择题测试中，四种线索类型在很大程度上依赖相同的少数注意力头——这是一个在预训练模型中就已存在的共享回路。大多数基础模型在纯文本中就会跟随线索。仅用约200条不含线索的通用助手示例对基础模型进行微调，就能在聊天格式中激活这些注意力头，因此跟随线索可能是助手调优的副作用。最终的强化学习（RL）阶段在几乎所有对比中都加强了线索的影响。在Tulu-3-8B和OLMo-2-7B中，我们交换了六个中间层在RL检查点之间的注意力模式……

    arXiv:2610.22937v1 Announce Type: cross  Abstract: Language models often abandon a correct answer when the prompt contains a cue, a hint toward a wrong option such as "I think the answer is B." Sycophancy and related biases show this failure. They are often studied separately and blamed on alignment. We test this view causally, tracing the attention heads behind these biases through the stages of two post-training pipelines. In multiple-choice tests on most models, four cue types rely largely on the same few heads, a shared circuit already present in pretrained models. Most base models follow cues in plain text. Fine-tuning a base model on about 200 generic assistant examples without cues activates these heads in chat format, so cue-following can be a side effect of assistant tuning. The final reinforcement learning (RL) stage strengthens the effect of cues in almost every comparison. In Tulu-3-8B and OLMo-2-7B, we swap the attention patterns of six middle layers between the RL checkpo
    
[^77]: ASR风格的转录文本规范化会移除对话行为分类中有用的语用信号

    ASR-Style Transcript Normalization Removes Useful Pragmatic Signal for Dialogue Act Classification

    [https://arxiv.org/abs/2610.22934](https://arxiv.org/abs/2610.22934)

    研究通过对六种分类器和三个会话语料库的系统性实验证明，ASR流水线中常见的转录规范化操作（如移除填充词和标点符号）会持续降低对话行为分类性能，表明这些语言特征包含重要的语用信号。

    

    自动语音识别（ASR）流水线通常通过移除填充停顿词和标点符号以及扩展口语形式来规范化转录文本，这可能丢弃对对话行为分类有用的线索。我们在六种分类器（TF-IDF逻辑回归、RoBERTa、BERT、ALBERT、GPT-2和BiLSTM）和三个会话语料库（Switchboard、ICSI MRDA和SLUE-HVB）上评估了九种转录文本条件，并辅以多种子实验、规范化对照实验和真实ASR评估。在五个随机种子下，重度规范化在原始预处理协议下降低了全部15个神经模型-语料库组合的Macro-F1。使用原始填充词和反馈词（backchannel）词表进行移除在MRDA和SLUE-HVB上一致地降低性能，但在Switchboard上产生混合效果。标点感知的TF-IDF实验表明，移除标点符号使Switchboard上的Macro-F1降低0.036，MRDA上降低0.078，证明

    arXiv:2610.22934v1 Announce Type: new  Abstract: Automatic speech recognition (ASR) pipelines often normalize transcripts by removing filled pauses and punctuation and expanding colloquial forms, potentially discarding cues useful for dialogue act classification. We evaluate nine transcript conditions across six classifiers (TF-IDF logistic regression, RoBERTa, BERT, ALBERT, GPT-2, and BiLSTM) and three conversational corpora (Switchboard, ICSI MRDA, and SLUE-HVB), supplemented by multi-seed experiments, normalization controls, and real-ASR evaluations. Across five random seeds, heavy normalization reduces Macro-F1 in all 15 neural model-corpus combinations under the original preprocessing protocols. Removal using the original filler and backchannel lexicon consistently degrades performance on MRDA and SLUE-HVB but produces mixed effects on Switchboard. Punctuation-aware TF-IDF experiments show that punctuation removal reduces Macro-F1 by 0.036 on Switchboard and 0.078 on MRDA, demonst
    
[^78]: Sep-Latent：将分隔符标记作为隐式推理的天然压缩锚点

    Sep-Latent: Separator Tokens as Natural Compression Anchors for Implicit Reasoning

    [https://arxiv.org/abs/2610.22924](https://arxiv.org/abs/2610.22924)

    本文提出Sep-Latent框架，将分隔符标记作为语义压缩锚点，结合注意力衰减机制与多层级蒸馏实现动态深度的隐式推理，从而在保持逻辑一致性的同时显著降低显式思维链的计算开销。

    

    大型语言模型（LLMs）通过显式的思维链（Chain-of-Thought, CoT）在复杂推理中取得了显著成功，但由此带来的计算开销和标记冗余仍然是效率提升的重大障碍。虽然连续潜在空间中的隐式推理提供了一种有前景的替代方案，但现有方法（如Coconut和CODI）通常依赖固定长度的潜在序列，并存在表示坍缩的问题。在本文中，我们提出了Sep-Latent，这是一种自适应隐式推理框架，它利用分隔符标记作为结构化的语义锚点。通过引入注意力衰减机制，我们迫使模型在训练过程中将推理逻辑聚合和压缩到这些锚点中。随后，我们通过跨所有层级的多层级蒸馏来对齐显式与隐式路径，以确保逻辑一致性。至关重要的是，Sep-Latent通过自适应地终止潜在思考过程，实现了动态推理深度……

    arXiv:2610.22924v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable success in complex reasoning through explicit Chain-of-Thought (CoT), but the associated computational overhead and token verbosity remain significant barriers to efficiency. While implicit reasoning in continuous latent spaces offers a promising alternative, existing methods like Coconut and CODI often rely on fixed-length latent sequences and suffer from representational collapse. In this paper, we propose Sep-Latent, an adaptive implicit reasoning framework that utilizes separator tokens as structural semantic anchors. By introducing an Attention-Decay mechanism, we force the model to aggregate and compress reasoning logic into these anchors during training. We then align explicit and implicit pathways through multi-level distillation across all layers to ensure logical consistency. Crucially, Sep-Latent enables dynamic reasoning depth by adaptively terminating the latent think
    
[^79]: 间接证据在哪里？使用影响函数研究语法泛化

    Where's the Indirect Evidence? Investigating Grammatical Generalization with Influence Functions

    [https://arxiv.org/abs/2610.22919](https://arxiv.org/abs/2610.22919)

    本文提出一种基于影响函数的自底向上方法来识别语言模型学习语法现象所依赖的间接证据，并发现移除这些间接证据会在五个模型的15种BLiMP语法现象上显著降低模型的泛化准确率。

    

    语言模型能够泛化到训练语料库之外以学习语言现象（Patil et al., 2024; Misra and Mahowald, 2024），这表明存在驱动学习的间接证据。先前的工作仅推断间接证据存在而未识别其特征，或只考虑了关于它可能是什么的特定假设。我们探索了一种替代性的自底向上方法来识别间接证据。我们使用影响函数（Koh and Liang, 2017）来衡量BLiMP（Warstadt et al., 2020）中15种语法现象的单个训练句子如何影响模型预测。我们使用手工构建的规则过滤掉某一现象的直接证据，并将剩余影响力最大的60,000个句子作为间接证据。这种间接证据对学习有实质性影响：在五个模型上，针对每一种语法现象，移除间接证据后重新训练比移除规模匹配的随机句子组更能降低留出集准确率（中位数降幅达……

    arXiv:2610.22919v1 Announce Type: new  Abstract: Language models generalize beyond their training corpus to learn linguistic phenomena (Patil et al., 2024; Misra and Mahowald, 2024), showing that there exists indirect evidence driving learning. Prior work has inferred that indirect evidence exists without identifying its character or considered specific hypotheses about what it might be. We explore an alternative bottom-up method for identifying indirect evidence. We use influence functions (Koh and Liang, 2017) to measure how individual training sentences affect model predictions in 15 grammatical phenomena from BLiMP (Warstadt et al., 2020). We filter out a phenomenon's direct evidence using hand-constructed rules and take the 60,000 most influential remaining sentences as indirect evidence. This indirect evidence substantially affects learning: retraining without it lowers held-out accuracy more than removing a matched random group for every phenomenon across five models (by a media
    
[^80]: FederatedBytes：无分词器、字节级小型语言模型，用于高效的多模态联邦学习

    FederatedBytes: Tokenizer-Free, Byte-Level Small Language Models for Efficient Multimodal Federated Learning

    [https://arxiv.org/abs/2610.22906](https://arxiv.org/abs/2610.22906)

    提出FederatedBytes——一种无分词器、将所有输入视为字节序列并采用线性注意力机制的小型语言模型，显著降低了NLP与多模态分析任务在联邦学习中的计算和通信开销。

    

    联邦学习（FL）是一种强大的机器学习框架，能够基于多个分布式客户端的贡献构建全局模型，同时保护其本地数据的隐私。然而，其高昂的计算与通信开销限制了它在自然语言处理（NLP）和多模态分析等当代AI工作负载中的应用。为此，本文提出了一种新型小型语言模型，该模型利用字节级的文本和多模态数据处理，并通过线性注意力机制加以增强，从而使NLP和多模态分析任务的联邦学习变得可行。该模型名为FederatedBytes，它将输入视为纯粹的字节序列，因而能够处理文本以及包括图像在内的任何其他数据类型；同时，其精简的线性注意力机制降低了整体的内存与计算开销。为验证所提模型的有效性……

    arXiv:2610.22906v1 Announce Type: new  Abstract: Federated Learning (FL) is a powerful machine learning framework that allows building a global model from the contributions of multiple, distributed clients, while ensuring the privacy of their local data. However, its heavy computational and communication overheads challenge its use for contemporary AI workloads such as natural language processing (NLP) and multimodal analysis. For this reason, this paper presents a novel small language model that leverages byte-level text and multimodal data processing, enhanced by a linear attention mechanism, to enable manageable Federated Learning of NLP and multimodal analysis tasks. The proposed model, nicknamed FederatedBytes, manages to process text and any other data types, including images, by treating its input as a mere sequence of bytes, while its lean linear attention mechanism allows abating the overall memory and computational footprint. To prove the effectiveness of the proposed model w
    
[^81]: 基于能力的大语言模型逻辑推理评估

    Competence-Based Evaluation of Logical Reasoning in Large Language Models

    [https://arxiv.org/abs/2610.22894](https://arxiv.org/abs/2610.22894)

    本文提出“基于能力的评估”（CBE）新范式，通过参数化生成器调节复杂度并使用保留答案的多种表述形式，来衡量大语言模型逻辑推理的泛化能力，而非仅依赖单一准确率指标。

    

    大语言模型（LLM）的最新进展使其在多种任务上展现出令人瞩目的性能；然而，在逻辑推理方面仍存在不足，包括在语言表述变化下的不一致性。为了提升大语言模型的推理能力，我们必须能够准确衡量其推理能力。当前的基准测试主要报告在选定推理任务上的单一准确率数字。这种基于性能的评估方法对于模型在推理方面的泛化能力说明甚少。为解决这一局限，我们主张采用一种不同的评估范式，称为基于能力的评估（CBE），它测试模型能否在一系列同族问题中可靠地运用相同的能力。参数化生成器在保持所需能力不变的情况下扩展问题复杂度，而匹配的、保留答案的多种表述方式则用于测试正确性是否超越单一表述形式。我们在基础的排序任务上实现了CBE。

    arXiv:2610.22894v1 Announce Type: cross  Abstract: Recent advances in large language models (LLMs) have brought impressive performance across diverse tasks; however, shortcomings remain in logical reasoning, including inconsistencies under linguistic variation. To improve the reasoning capacity of LLMs, we must be able to accurately measure their reasoning capacity. Current benchmarks primarily report single accuracy numbers on selected reasoning tasks. Such a performance-based evaluation methodology says little about a model's generalization capacity in reasoning. To address this limitation, we argue for a different evaluation paradigm, called competence-based evaluation (CBE), which tests whether models reliably apply the same competence across a family of problems. A parameterized generator scales complexity while keeping the required competence fixed, and matched answer-preserving presentations test correctness beyond one formulation. We instantiate CBE on elementary ordering tasks
    
[^82]: 基于检索与程序的孟加拉语幻觉检测：一项系统研究

    Bengali Hallucination Detection with Retrieval and Programs: A System Study

    [https://arxiv.org/abs/2610.22892](https://arxiv.org/abs/2610.22892)

    该论文提出了一个结合查询重写、混合检索和语言模型评判的仅推理孟加拉语幻觉检测系统，在Kaggle竞赛中取得了0.956的macro-F1分数。

    

    孟加拉语幻觉检测必须在缺乏参考文本的情况下，区分无事实依据的声明与错误推理。我们提出并描述了一个仅用于推理的系统，该系统结合了基于响应的条件查询重写、混合检索以及任务特定的语言模型评判。系统设有六条检索路径来搜索问答和文本集合；路由器将每个样本导向27B参数的文本评判模型或14B参数的智能体，后者可以比较检索到的答案、直接推理，或生成并执行Python代码。我们使用来自Olikbochon挑战赛的特定版本代码、保存的预测结果、执行轨迹和语料库元数据对系统进行了特征分析。Kaggle显示版本2的私有macro-F1分数为0.956；其保存的运行记录在两块T4 GPU上用220.99分钟处理了2,516个样本。在253个被路由到数学处理的样本中，88个通过候选匹配终止，而执行轨迹揭示了49个包含stderr的运行事件。

    arXiv:2610.22892v1 Announce Type: new  Abstract: Bengali hallucination detection must distinguish unsupported factual claims from incorrect reasoning, often without a supplied reference passage. We present and characterize an inference-only system that combines response-conditioned query rewriting, hybrid retrieval, and task-specific language-model judging. Six retrieval routes search question-answer and passage collections; a router directs each example to a 27B textual judge or a 14B agent that can compare retrieved answers, reason directly, or generate and execute Python. We characterize the system using version-specific code, saved predictions, execution traces, and corpus metadata from the Olikbochon challenge. Kaggle associates a displayed private macro-F1 of 0.956 with Version 2; its saved run processes 2,516 examples in 220.99 minutes on two T4 GPUs. Of 253 math-routed examples, 88 terminate through candidate matching, while execution traces expose 49 stderr-bearing run events 
    
[^83]: NLData2Opt：基于自然语言与数据的优化问题建模中参数推导与预测的基准测试

    NLData2Opt: Benchmarking Parameter Derivation and Prediction for Modeling Optimization Problems from Natural Language and Data

    [https://arxiv.org/abs/2610.22889](https://arxiv.org/abs/2610.22889)

    该论文提出了 NLData2Opt 基准，要求大语言模型从含噪声的真实数据中完成优化问题参数的提取、推导计算和未来值预测，弥补了现有基准依赖理想化输入而高估模型实际能力的不足。

    

    从问题描述和数据的原始输入中构建优化问题是耗时且成本高昂的。虽然大语言模型（LLM）为自动化这一过程提供了一条有前景的途径，但现有的基准测试依赖于所有数据都相关且完整的理想化输入，从而高估了模型的能力。这与现实情况存在偏差——现实中需要进行以数据为中心的推理，才能从含噪声的数据源中提取、计算和预测参数。为解决这一问题，我们提出了 NLData2Opt，一个用于评估大语言模型在实际优化工作流程中固有的多阶段推理能力的基准。NLData2Opt 将问题分解为问题描述和数据两部分，并要求模型（1）从包含误导性属性的数据中提取相关参数，（2）通过转换现有数据属性来推导计算缺失的参数，以及（3）根据历史观测数据预测不可得的未来数值。评估结果显示……

    arXiv:2610.22889v1 Announce Type: new  Abstract: Formulating optimization problems from raw inputs of problem descriptions and data is time-consuming and costly. While Large Language Models (LLMs) offer a promising way to automate this process, existing benchmarks overestimate their capabilities by relying on idealized inputs where all data is relevant and complete. This diverges from reality, where data-centric reasoning is required to extract, compute, and predict parameters from noisy sources. To address this, we introduce NLData2Opt, a benchmark to evaluate LLMs on the multi-stage reasoning inherent in practical optimization workflows. NLData2Opt decomposes problems into problem descriptions and data, and requires models to (1) extract relevant parameters from data containing misleading attributes, (2) compute missing parameters derived by transforming existing data attributes, and (3) predict unavailable future values from historical observations. Evaluation results show substanti
    
[^84]: 批量上下文强化 II：RLVR 需要长采样窗口吗？

    Batched Contextual Reinforcement II: Does RLVR Need Long Rollout Windows?

    [https://arxiv.org/abs/2610.22887](https://arxiv.org/abs/2610.22887)

    提出并发约束强化（CCR），将多个可独立验证的问题放入同一个受限完成中，使RLVR在8,000-token的短窗口下仍能保留正确性信号，并将每训练步的token消耗从约97.5万降至约17.5万，大幅降低训练成本。

    

    长采样窗口使得可验证奖励强化学习（RLVR）的代价高昂，但简单缩短窗口可能会丢失正确性信号：当每个单问题的采样在得出答案之前就被截断时，组相对准确率优势将不复存在。作为批量上下文强化（Batched Contextual Reinforcement, BCR）的技术延续，我们研究了并发奖励如何放宽这种对长窗口的依赖。并发约束强化将三个可独立验证的问题置于同一个有长度上限的训练完成中。在截断前完成的正确答案仍可获得奖励，而共享预算在无需显式长度惩罚的情况下，鼓励更简短的有效推理。我们将该机制与条件奖励方差和更新方差分析联系起来，并在4B和8B数学推理模型上进行了研究。与使用32k窗口的单问题GRPO相比，使用8,000-token窗口的CCR在4B模型上将每训练步的token数从975,464降至174,957。

    arXiv:2610.22887v1 Announce Type: cross  Abstract: Long rollout windows make reinforcement learning with verifiable rewards (RLVR) expensive, yet simply shortening them can remove the correctness signal: when every single-problem rollout truncates before an answer, group-relative accuracy advantages vanish. As a technical continuation of Batched Contextual Reinforcement (BCR), we study how concurrent rewards relax this dependence on long windows. Concurrent Constraint Reinforcement (CCR) places three independently verifiable problems in one capped training completion. Correct answers completed before truncation retain credit, while a shared budget encourages shorter useful reasoning without an explicit length penalty. We connect this mechanism to conditional reward- and update-variance analyses and study 4B and 8B mathematical reasoning models. Against single-problem GRPO with a 32k window, CCR with an 8,000-token window reduces tokens per training step from 975,464 to 174,957 on 4B an
    
[^85]: 无需适应的合作：集体风险困境中的大语言模型智能体

    Cooperation Without Adaptation: LLM Agents in a Collective-Risk Dilemma

    [https://arxiv.org/abs/2610.22882](https://arxiv.org/abs/2610.22882)

    研究发现大语言模型智能体在集体风险困境博弈中采用固定而非适应性的合作策略，即使在零风险且贡献无法改善自身收益的情况下仍会贡献资源，表明其合作行为源于策略惯性而非对环境风险的理性适应。

    

    语言模型智能体越来越多地被部署用于代表人们行事，在这些场景中，个体决策会影响共同的结果。我们研究了五个生产级模型是否会根据不断变化的风险和伙伴行为调整其贡献，还是遵循相对固定的策略。在集体风险困境博弈（CRG）中，六名玩家在十轮游戏中向一个共享池贡献资源；如果共享池未达到目标，一场灾难可能会摧毁他们剩余的资源。四个模型在每个正风险水平上都贡献了大约相等的份额，五个模型中有四个甚至在风险为零时也会做出贡献，尽管此时贡献无法改善其收益。明确声明只有智能体自身的收益才重要时，大多数模型的零风险贡献有所减少，而更中性的提示词则没有这种效果。几个模型能够正确识别在不同风险水平下哪种行动提供更高的预期收益，但它们的（摘要在此处截断）

    arXiv:2610.22882v1 Announce Type: cross  Abstract: Language-model agents are increasingly deployed to act on behalf of people in settings where individual decisions shape shared outcomes. We examine whether five production models adapt their contributions to changing risks and partner behaviour, or instead follow a relatively fixed strategy. In a collective-risk dilemma game (CRG), six players contribute to a shared pool over ten rounds; if the pool falls short of its target, a catastrophe may destroy their remaining resources. Four models contributed approximately an equal share at every positive risk level, and four of the five also contributed when risk was zero, even though contributing could not improve their payoff. Explicitly stating that only the agent's own payoff mattered reduced zero-risk contributions for most models, while a more neutral prompt did not. Several models correctly identified which action offered the higher expected payoff at different risk levels, yet their c
    
[^86]: 识别、分类、行动：面向外部数据的LLM逐指令受控合规

    Identify, Classify, Act: Per-Instruction Controlled Compliance for LLMs over External Data

    [https://arxiv.org/abs/2610.22874](https://arxiv.org/abs/2610.22874)

    提出EDIC分解式奖励框架，通过对外部数据中每条嵌入指令进行识别-分类-行动决策，并利用基于场景的指令注入和SFT加GRPO训练的生成式评判者实现指令级功劳归因，使LLM能够遵循合法指令的同时拒绝对抗性提示注入。

    

    基于外部数据的大语言模型面临逐指令层面的张力：单个文档或工具返回结果常常将合法指令与对抗性提示注入交织在一起，模型必须遵循前者同时拒绝后者。现有防御措施要么压制所有外部指令，要么信任所有外部指令，导致安全-效用边界坍缩，迫使人们在不当拒绝与过度服从之间做出选择。我们认为缺失的能力是逐指令功劳归因，并将其形式化为外部数据指令分类，即对每条嵌入指令进行识别-分类-行动的决策过程。我们提出EDIC，一个用于逐指令合规的分解式奖励框架。一个基于场景的流水线通过反事实过滤注入混合的有益与有害指令，同时一个生成式评判者通过SFT初始化并经GRPO优化，以在指令级别归因合规性。评判者随后产生一个奖励……

    arXiv:2610.22874v1 Announce Type: new  Abstract: LLMs over external data face a per-instruction tension: a single document or tool return often interleaves legitimate directives with adversarial prompt injections, and the model must follow the former while rejecting the latter. Existing defenses suppress or trust all external instructions, collapsing the safety-utility frontier and forcing a choice between over-refusal and over-compliance. We argue the missing capability is \textbf{per-instruction credit attribution} and formalize it as External-Data Instruction Classification, an identify-classify-act decision over each embedded directive. We propose \textbf{EDIC}, a decomposed reward framework for per-instruction compliance. A scene-grounded pipeline injects mixed beneficial and harmful instructions with counterfactual filtering, while a generative judge is initialized by SFT and refined with GRPO to attribute compliance at the instruction level. The judge then produces a reward that
    
[^87]: 面向高效LLM树搜索的不确定性感知预算分配

    Uncertainty-Aware Budget Allocation for Efficient LLM Tree Search

    [https://arxiv.org/abs/2610.22858](https://arxiv.org/abs/2610.22858)

    该论文提出无需训练的不确定性感知分配框架（UAA），利用兄弟动作的语义分散度在扩展前预测下游后续内容的冗余性，自适应地为LLM树搜索的每一步分配预算，从而大幅提升多步推理的效率。

    

    树搜索能够增强大型语言模型（LLM）的多步推理能力，然而现有方法以固定预算扩展每一步，而不考虑扩展的边际效用。我们首先证明，在现有的搜索算法中，从共享中间状态采样得到的后续内容大多是重复的，并坍缩为少数几个不同的答案。关键的是，我们发现兄弟动作的语义分散度可以预测其下游后续内容的冗余性，从而允许在扩展之前就确定子树的预算。信息论分析为这一做法提供了理论支撑：将每步预算视为估计该步剩余答案熵的蒙特卡洛样本量，我们推导出了一个随兄弟分散度收缩而减小的充分预算。据此，我们提出了不确定性感知分配（Uncertainty-Aware Allocation, UAA），这是一个无需训练的框架，它将价值与不确定性解耦：价值决定哪些候选被（原文在此处截断）

    arXiv:2610.22858v1 Announce Type: cross  Abstract: Tree search enhances multi-step reasoning in large language models (LLMs), yet prevailing methods expand every step with a fixed budget, irrespective of the marginal utility of expansion. We first show, across existing search algorithms, that continuations sampled from a shared intermediate state are largely duplicates and collapse to a few distinct answers. Critically, we find that the semantic dispersion of sibling actions predicts the redundancy of their downstream continuations, permitting a subtree's budget to be fixed before expansion. An information-theoretic analysis motivates this: treating the per-step budget as the Monte Carlo sample size for estimating a step's residual answer entropy, we derive a sufficient budget that shrinks with sibling dispersion. We accordingly propose $\textbf{Uncertainty-Aware Allocation}$ (UAA), a training-free framework that decouples value from uncertainty: values determine which candidates are p
    
[^88]: 依赖假设：随着语言资源水平的下降，思维链的忠实度上升

    The Reliance Hypothesis: Chain-of-Thought Faithfulness Rises as Language Resource Level Falls

    [https://arxiv.org/abs/2610.22841](https://arxiv.org/abs/2610.22841)

    论文提出“依赖假设”：当模型无法回忆答案时会更加依赖思维链推理，因此CoT忠实度在知识密集型任务上随语言资源水平的下降而上升。

    

    思维链的忠实度对于监控、解释和安全部署大型语言模型至关重要，然而目前的研究大多集中在英语上。语言之间存在资源水平的差异，即可用于学习的文本数量不同，因此模型在低资源语言中掌握的知识较少，这可能改变它们对自身推理的依赖程度。据我们所知，我们是首个将CoT忠实度同时与资源水平和任务类型（知识密集型或推理密集型）关联起来的研究。在来自多语言大规模多任务语言理解（MMMLU）基准的八种语言的平行多项选择题上，忠实度在知识密集型任务上随着资源水平的下降而上升，但在推理密集型任务上并非如此，这一差异在四个模型和多种忠实度度量上均呈现出相同的方向。我们提出并找到了支持“依赖假设”的证据：当模型无法回忆起答案时（例如在低资源语言中），它们会更加依赖自己的推理。

    arXiv:2610.22841v1 Announce Type: cross  Abstract: Chain-of-Thought (CoT) faithfulness matters for monitoring, explaining and safely deploying large language models, yet it has been studied mostly in English. Languages differ in level, the amount of text available to learn from, so models know less in lower-resource languages, which may change how much they rely on their reasoning. To our knowledge, we are the first to relate CoT faithfulness to both resource level and task type (knowledge- or reasoning-intensive). On parallel multiple-choice questions from Multilingual Massive Multitask Language Understanding (MMMLU) in eight languages, faithfulness rises as resource level falls on knowledge-intensive tasks but not on reasoning-intensive ones, a gap that points the same way across four models and several faithfulness measures. We propose, and find support for, the Reliance Hypothesis: models rely more on their reasoning when they cannot recall the answer, as in lower-resource language
    
[^89]: 太情绪化反而不像伪造？语句内情绪变化暴露语音深度伪造检测器的失效模式

    Too Emotional to Be Fake? Intra-Utterance Emotion Shifts Expose Failure Modes in Speech Deepfake Detectors

    [https://arxiv.org/abs/2610.22832](https://arxiv.org/abs/2610.22832)

    提出EmoShift基准，揭示语句内情绪变化会显著降低语音深度伪造检测的可靠性——情绪变化的合成语音漏检率最高上升25.3个百分点，而情绪变化的真实语音误报率最高上升24.4个百分点。

    

    表现性语音合成技术现已能在单个语句内改变情绪，但其对深度伪造检测的影响仍不明确。我们提出EmoShift基准，通过对比情绪稳定（AA）与情绪变化（AB）的语音来检验这些影响，同时控制静态情绪身份和声学混淆因素。该基准包含来自四个源数据集的7,240条真实语音，以及使用四个表现性TTS系统生成的9,720条合成语音。在七个检测器上的实验表明：情绪变化会使合成语音显得更加真实，在固定阈值下，OMG-Emotion数据集上的伪造漏检率最高增加25.3个百分点；相反，情绪变化的真实语音会使IEMOCAP数据集上的误报率最高增加24.4个百分点。进一步分析揭示了各检测器对情绪轨迹的特异性敏感，这无法完全由静态情绪身份或拼接伪影所解释。这些发现凸显了当前语音深度伪造检测系统在情绪表达维度的关键脆弱性。

    arXiv:2610.22832v1 Announce Type: cross  Abstract: Expressive speech synthesis can now change emotion within a single utterance, yet its impact on deepfake detection remains unclear. We introduce EmoShift, a benchmark comparing emotion-stable (AA) and emotion-changing (AB) speech to examine these effects while accounting for static emotion identity and acoustic confounds. It comprises 7,240 bona fide utterances from four source datasets and 9,720 synthetic utterances generated using four expressive TTS systems. Across seven detectors, we find that emotion changes can make synthetic speech appear more genuine, increasing spoof miss rates by up to 25.3 percentage points on OMG-Emotion at fixed thresholds. Conversely, emotion-changing bona fide speech increases false-alarm rates by up to 24.4 percentage points on IEMOCAP. Further analyses reveal detector-specific sensitivities to emotion trajectories, not fully explained by static emotion identity or splicing artifacts. These findings und
    
[^90]: 因果编辑序列化

    Causal Edit Serialization

    [https://arxiv.org/abs/2610.22820](https://arxiv.org/abs/2610.22820)

    提出因果编辑序列化（CES）框架，通过指针词元、轻量级定位头和双索引旋转位置编码，使自回归语言模型能够在保持标准因果注意力机制的前提下，对文本进行插入、删除、替换、复制和移动等显式编辑操作。

    

    标准自回归语言模型按顺序生成文本，但修订操作需要回溯引用输入上下文中的词元或已生成的词元。本文提出了因果编辑序列化（CES）框架，用于在同一个因果词元流中表达此类修订。特殊的指针词元与预测的源地址相配对，使得插入、删除、替换、复制和移动等操作能够被显式地定位。轻量级的定位头与双索引旋转位置编码既支持地址预测，也支持对先前发出的指针进行条件化处理，同时保留了标准的因果注意力掩码。预训练过程将普通语言建模与序列化修复任务相结合，这些修复任务来源于随机损坏、模型生成的编辑以及重复文本。在多种模型规模和编辑基线的对比中，CES模型能够完整地执行编辑操作，同时在干净文本上每字节的比特数仅略有增加。

    arXiv:2610.22820v1 Announce Type: new  Abstract: Standard autoregressive language models generate text sequentially, but revision requires referring back to tokens in the input context or those already generated. Causal Edit Serialization (CES) is introduced as a framework for expressing such revisions within the same causal token stream. Special pointer tokens are paired with predicted source addresses, allowing insertion, deletion, replacement, copying and movement to be localized explicitly. A lightweight localization head and dual-index rotary position encoding support both address prediction and conditioning on previously emitted pointers, while retaining a standard causal attention mask. A pretraining procedure combines ordinary language modeling with serialized repairs derived from random corruptions, model-generated edits and repeated text. Across model scales and editing baselines, CES models demonstrate complete applied edits with modestly higher clean-text bits per byte than
    
[^91]: 水印可植入性：衡量并提升语言模型与基于Logit的水印方法的兼容性

    Watermarkability: Measuring and Improving Language-Model Compatibility with Logit-Based Watermarks

    [https://arxiv.org/abs/2610.22810](https://arxiv.org/abs/2610.22810)

    该论文提出了“水印可植入性”概念及其KL散度代价度量方法，用以量化语言模型与基于logit水印之间的兼容性，并设计了降低水印植入代价的两阶段方法。

    

    语言模型水印可以帮助识别模型生成的文本、编码来源信息以及检测生成后的编辑。一大类基于logit的水印方法会在每一步解码时提高指定词元集合的logit值。由于基础模型在该集合上的原生概率质量会随上下文、模型和水印规则的不同而变化，相同的logit偏置可能产生不同的水印依从率和困惑度（PPL）增幅。我们提出“水印可植入性”这一概念来刻画模型与水印所请求的词元集合之间的兼容性。在目标概率质量τ下，我们通过水印可植入性代价来量化这一属性：即为了使所请求的集合获得至少τ的概率，基础模型分布所需的最小KL散度的期望值。代价越低，意味着模型对该请求分布越具有可水印性。为了降低这一代价，我们提出了一种两阶段方法。在第一阶段，我们……

    arXiv:2610.22810v1 Announce Type: new  Abstract: Language-model watermarks can help identify model-generated text, encode provenance information, and detect post-generation edits. A broad class of logit-based watermarks increases the logits of a designated token set at each decoding step. Because a base model's native probability mass on this set varies across contexts, models, and watermark rules, the same logit bias can yield different adherence rates and increases in perplexity (PPL). We introduce watermarkability to characterize the compatibility between a model and the token sets requested by a watermark. At a target mass $\tau$, we quantify this property through a watermarkability cost: the expected minimum KL divergence from the base-model distribution required to assign probability at least $\tau$ to the requested set. A lower cost means that the model is more watermarkable for that request distribution. To reduce this cost, we propose a two-stage method. In the first stage, we
    
[^92]: 《大语言模型的位置偏差：一项全面研究》

    On the Position Bias of Large Language Models: A Comprehensive Study

    [https://arxiv.org/abs/2610.22807](https://arxiv.org/abs/2610.22807)

    该研究全面考察了从十亿到万亿级参数规模、涵盖五种语言的大语言模型位置偏差问题，发现增大模型规模和使用思维链提示可以显著缓解这一偏差。

    

    近期研究表明，位置偏差仍然是大语言模型（LLM）的一个关键脆弱点，它既损害了多选题评估的完整性，也影响了检索增强生成的可靠性。然而，对这一现象的现有理解主要来自模型覆盖范围有限且以英语为中心的基准测试研究，前沿模型和多语言环境下的行为尚未得到充分探索。在本工作中，我们对从十亿级到万亿级参数规模的模型中的位置偏差进行了全面研究，涵盖五种语言下的多选题和多文档任务。我们的实验表明，尽管位置偏差在小规模模型和非英语语言中仍然明显，但增大模型规模和采用标准的思维链（Chain-of-Thought）提示能够显著缓解这一问题。此外，我们证明了新兴的思维模型……

    arXiv:2610.22807v1 Announce Type: new  Abstract: Recent studies have shown that position bias remains a critical vulnerability in large language models (LLMs), undermining both the integrity of multiple-choice evaluations and the reliability of retrieval-augmented generation. However, existing understanding of this phenomenon is largely derived from studies with limited model coverage and English-centric benchmarks, leaving the behavior of frontier models and diverse languages underexplored. In this work, we conduct a comprehensive investigation of position bias across model scales ranging from billion-level to trillion-level parameters, covering both multiple-choice and multi-document tasks in five languages. Our experiments reveal that while position bias remains noticeable in small-scale models and Non-English languages, increasing model scale and employing standard Chain-of-Thought prompting significantly mitigate this issue. Moreover, we demonstrate that emerging \emph{thinking mo
    
[^93]: 先遮后露（Veil-Then-Reveal）：在零信任的用户与大语言模型交互中通过可逆伪匿名化保护个人身份信息隐私

    Veil-Then-Reveal: Protecting PII Privacy via Reversible Pseudo-Anonymization in Zero-Trust User--LLM Interaction

    [https://arxiv.org/abs/2610.22800](https://arxiv.org/abs/2610.22800)

    提出用户侧可逆伪匿名化框架Veil-then-Reveal（VtR），在零信任威胁模型下将提示词中的PII替换为同类型的真实替身词以保护隐私，同时保留语言与结构特性使LLM仍能有效处理，明文信息始终保留在用户端。

    

    大语言模型（LLM）日益被用于处理敏感文档和个人信息，然而向远程服务提供商发送包含个人身份信息（PII）的提示词，会使数据主体和组织面临日志记录、数据留存、模型训练以及意外泄露等风险。现有的匿名化方法通常会遮蔽或删除PII，而传统的加密方式则会消除LLM有效处理所依赖的词汇和结构线索。我们提出了Veil-then-Reveal（VtR），这是一个在零信任威胁模型下实现隐私保护LLM交互的用户侧PII伪匿名化框架。VtR将远程LLM提供商视为不可信方，仅发送经过伪匿名化处理的提示词，而将明文PII保留在用户端。VtR首先检测PII，并在提示词离开用户设备之前，将其替换为真实的、同类型、存在于词表中的替身词，从而保留提示词的语言与结构特性。它结合……

    arXiv:2610.22800v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly process sensitive documents and personal information, yet sending prompts containing personally identifiable information (PII) to remote providers exposes data subjects and organizations to risks from logging, retention, model training, and unintended disclosure. Existing anonymization methods often mask or remove PII, while conventional encryption eliminates the lexical and structural cues needed for effective LLM processing. We present Veil-then-Reveal (VtR), a user-side PII pseudo-anonymization framework for privacy-preserving LLM interaction under a zero-trust threat model. VtR treats remote LLM providers as untrusted and sends only pseudo-anonymized prompts, keeping plaintext PII on the user. VtR first detects PII and replaces it with realistic, same-type, in-vocabulary surrogates before prompts leave the user's device, preserving their linguistic and structural properties. It combines for
    
[^94]: 以语义ID作为唯一物品表示的推理重排序

    Reasoning Reranking with Semantic IDs as the Sole Item Representation

    [https://arxiv.org/abs/2610.22797](https://arxiv.org/abs/2610.22797)

    本文构建了ReRankID受控测试平台，首次系统评估仅以语义ID作为物品表示进行推理重排序的可行性，发现此类模型常常直接复制输入顺序而非真正重排序，且引入可读元数据也无法修复这一失败。

    

    推理重排序器在排定候选列表之前会先生成一段推理依据，而现有系统通过读取物品元数据来实现这一过程，尽管生成式推荐系统正日益采用紧凑的语义ID（SIDs）来表示物品。仅凭SID进行排序具有两点吸引力：其一，SID提示词远短于匹配的物品文本；其二，尽管物品编码由内容固定，但骨干模型会部分地从用户-物品交互序列中学习这些token的嵌入，因此这些token可以承载物品文本所缺乏的行为信号。然而，此类训练以“下一物品生成”为目标，其能否支持仅凭SID对给定的候选列表进行排序尚未得到验证。我们构建了ReRankID——一个受控测试平台，模型仅依据以SID形式呈现的历史与候选列表进行推理和排序，并每次仅改变一个变量：物品呈现方式、历史长度或候选列表大小。在该测试平台上，此类模型往往直接复制输入顺序而不进行重排序，且可读的元数据也无法修复这一失败。

    arXiv:2610.22797v1 Announce Type: cross  Abstract: Reasoning rerankers write a rationale before ranking a candidate slate, and existing systems read item metadata to do so, even though generative recommenders increasingly represent items as compact Semantic IDs (SIDs). Ranking from SIDs alone has two appeals: SID prompts are far shorter than the matched item text, and although item codes are fixed from content, the backbone learns their token embeddings in part from user-item interaction sequences, so the tokens can carry behavioral signal that item text lacks. That training targets next-item generation, and whether it supports ranking a supplied slate from SIDs alone is untested. We build ReRankID, a controlled testbed in which models reason and rank from a history and slate shown only as SIDs, varying item rendering, history length, and slate size one at a time. On this testbed, such models often copy the input order instead of reranking, and readable metadata does not repair the fai
    
[^95]: 面向循环Transformer的因果潜在工作空间

    Causal Latent Workspaces for Looped Transformers

    [https://arxiv.org/abs/2610.22796](https://arxiv.org/abs/2610.22796)

    该论文提出因果潜在工作空间Transformer，通过在循环共享深度架构中引入门控潜在槽位，使各位置能够交换因果前缀的紧凑摘要，在多个语料库和匹配参数预算下实现了优于标准Transformer的困惑度表现。

    

    在深度方向上复用Transformer块可以在不复制其参数的情况下增加计算量，但连续的多轮传递缺乏专门的状态，使各位置无法交换关于其因果前缀的紧凑摘要。我们提出了因果潜在工作空间Transformer：一种带锚定机制的四轮共享深度架构，其中每个位置将门控向量写入特定于前缀的潜在槽位，并在下一次共享块应用之前检索基于内容的寻址摘要。这条路径对因果自注意力形成补充，同时原始输入嵌入在每一轮都保持可访问。在三个字节级语料库、三个随机种子和两个匹配参数预算的实验中，所提出的模型在六种设置中的五种里，相比五个Transformer对照模型取得了最佳的验证集平均困惑度。在101.71M参数规模下，该模型在WikiText-103 raw、enwik8和TinyStories上领先，分别达到2.7820、2.7843和1.6823的困惑度；这些结果相较于……

    arXiv:2610.22796v1 Announce Type: new  Abstract: Reusing a Transformer block across depth increases computation without replicating its parameters, but successive passes have no dedicated state through which positions can exchange compact summaries of their causal prefix. We introduce the causal latent workspace Transformer: an anchored, four-round shared-depth architecture in which each position writes gated vectors into prefix-specific latent slots and retrieves a content-addressed summary before the next shared block application. This path complements causal self-attention while the original input embedding remains accessible at every round. Across three byte-level corpora, three seeds, and two matched parameter budgets, the proposed model achieves the best mean held-out perplexity in five of six settings against five Transformer controls. At 101.71M parameters it leads on WikiText-103 raw, enwik8, and TinyStories, reaching perplexities 2.7820, 2.7843, and 1.6823; these improve upon
    
[^96]: 剪除泄漏神经元：通过联合嵌入引导的剪枝缓解微调大语言模型中的隐私数据泄露

    Prune the Leaky Neurons: Mitigating Private Data Leakage in Fine-Tuned LLMs via Joint-Embedding-Guided Pruning

    [https://arxiv.org/abs/2610.22792](https://arxiv.org/abs/2610.22792)

    提出 JEPA-Prune 框架，利用在无个人身份信息的替代视图上通过联合嵌入预测架构训练的公开模型来引导事后神经元剪枝，在保持模型性能的同时有效缓解微调大语言模型中的隐私数据泄露问题。

    

    在专有记录上微调大语言模型（LLM）可以实现领域适配，但会带来两个挑战：（i）已记忆的隐私数据（如个人身份信息 PII）存在隐私泄露风险，（ii）需要在移除隐私数据与保持模型性能之间取得平衡。现有防御方法，包括数据清洗和差分隐私，往往会降低模型性能，而隐私神经元编辑方法则缺乏一个无 PII 的参照来区分身份特定神经元与支持有用文本的神经元。我们提出 JEPA-Prune，这是一个由仅在无 PII 视图上训练的模型所引导的事后神经元剪枝框架。在微调后的私有模型之外，JEPA-Prune 使用联合嵌入预测架构（JEPA）在同一记录的替代视图和掩码视图上训练一个公开模型，从而在不包含标识符的情况下捕获领域知识。跨模型表示探针用于量化私有表示与公开表示之间的偏差……

    arXiv:2610.22792v1 Announce Type: cross  Abstract: Fine-tuning large language models (LLMs) on proprietary records enables domain adaptation but introduces two challenges: (i) privacy leakage of memorized private data, such as personally identifiable information (PII), and (ii) balancing private data removal with utility preservation. Existing defenses, including data sanitization and differential privacy, often degrade performance, while privacy-neuron editing lacks a PII-free reference to distinguish identity-specific neurons from those supporting useful text. We propose JEPA-Prune, a post-hoc neuron pruning framework guided by a model trained exclusively on PII-free views. Alongside the fine-tuned private model, JEPA-Prune, trains a public model on surrogate and masked views of the same records using a joint-embedding predictive architecture (JEPA), capturing domain knowledge without identifiers. A cross-model representation probe quantifies deviations between private representation
    
[^97]: 基于框架-策略优化的自进化编码智能体

    Self-Evolving Coding Agents via Harness-Policy Optimization

    [https://arxiv.org/abs/2610.22791](https://arxiv.org/abs/2610.22791)

    提出 Self-HPO 框架，通过交替优化智能体框架与策略模型，让进化的策略自己提出框架改进方案，从而实现编码智能体的递归自我进化。

    

    大型语言模型（LLM）智能体在编码任务中展现出可喜的潜力，其性能由策略模型和作为智能体与环境交互中介的框架共同决定。事实上，框架与策略是相互促进的，仅进化其中一个组件是次优的——为某个检查点调优的框架可能无法针对更新后策略的失败模式，而策略学习也受限于固定框架所产生的轨迹。鉴于此，我们提出 Self-HPO，一个面向编码智能体的递归自我改进框架，它交替进行框架和策略优化，并将不断进化的策略作为自身框架的提议者。具体而言，在每一轮中，当前策略首先分析自身的失败并提出新的候选框架（H步骤）；随后在选定框架下通过强化学习（RL）更新策略（P步骤），并针对训练过程中识别出的困难任务进行……

    arXiv:2610.22791v1 Announce Type: cross  Abstract: Large language model (LLM) agents have shown promising potential in coding tasks, and their performance is jointly determined by the policy model and the harness that mediates their interaction with the environment. Indeed, harness and policy are mutually reinforcing and evolving only one component is suboptimal---a harness tuned for one checkpoint may not target the failure modes of an updated policy, while policy learning remains limited to the trajectories elicited by a fixed harness. In light of this, we propose Self-HPO, a recursive self-improving framework for coding agents that alternates harness and policy optimization, using the evolving policy as its own harness proposer. Specifically, in each round, the current policy first analyzes its own failures and proposes new harness candidates (H-step). The policy is then updated through reinforcement learning (RL) under the selected harness (P-step), and hard tasks identified during
    
[^98]: 无后见之明的新闻阅读：NoLBERT主题与股票截面收益

    Reading the News Without Hindsight: NoLBERT Topics and the Cross-Section of Stock Returns

    [https://arxiv.org/abs/2610.22745](https://arxiv.org/abs/2610.22745)

    该论文提出仅使用1976-1995年历史文本预训练的NoLBERT编码器来构建无后见之明偏差的新闻主题，并发现将注意力冲击衡量窗口延长至63个交易日可显著提升基于新闻的股票截面收益预测效果。

    

    读取金融新闻的语言模型通常是在与测试年份相同的文本上训练的，因此回测可能因后见之明而受益。我们基于NoLBERT构建新闻主题，NoLBERT是一个仅在1976年至1995年间发布的文本上进行预训练的编码器，并通过对同一文章的段落进行对比学习微调，将其适配为句子编码器，整个过程不使用任何标签或收益率数据。我们在2012年至2024年的126万篇文章上，将其与LDA、MiniLM和FinBERT等阅读器进行比较，将每个阅读器的主题转换为注意力冲击和股票暴露，并用八个模型在三种再平衡频率下预测美国市值最大的1,000只股票的收益率。模型在2013-2019年数据上拟合，选择在2020-2021年数据上做出，2022-2024年数据被留出用于测试。验证得分最好的实验在每种频率下都出现亏损。将注意力冲击的衡量窗口从过去5个交易日改为过去63个交易日，能提高所有阅读器的测试秩IC（rank IC）。NoLBERT给出最稳……（原文此处截断）

    arXiv:2610.22745v1 Announce Type: cross  Abstract: Language models that read financial news are usually trained on text from the same years used to test them, so a backtest can reward hindsight. We build news topics from NoLBERT, an encoder pretrained only on text published between 1976 and 1995, adapted into a sentence encoder by contrastive fine-tuning on passages of the same article, with no labels or returns. We compare it with LDA, MiniLM and FinBERT readers on 1.26 million articles from 2012 to 2024, convert each reader's topics into attention shocks and stock exposures, and forecast the returns of the 1,000 largest US stocks with eight models at three rebalancing frequencies. Models are fitted on 2013-2019, choices are made on 2020-2021, and 2022-2024 is held out. The experiment with the best validation score loses money at every frequency. Measuring attention shocks against the past 63 trading days instead of 5 raises the test rank IC of every reader. NoLBERT gives the most sta
    
[^99]: Thunder-DeID+：从韩国法院判决书到更广泛敏感文档的边界感知韩语去标识化

    Thunder-DeID+: Boundary-Aware Korean De-identification from Court Judgments to Broader Sensitive Documents

    [https://arxiv.org/abs/2610.22733](https://arxiv.org/abs/2610.22733)

    提出Thunder-DeID+框架，通过结合精炼的三级PII分类体系与融合N-gram和边界评分组件的边界感知分词器，解决了韩语黏着语结构下BPE分词与PII跨度边界不对齐的问题，并将法院判决书的去标识化能力扩展至更广泛的敏感文档。

    

    本研究考察了为韩国法院判决书开发的去标识化模型向更广泛敏感文档迁移的能力，重点关注边界精度、标签粒度和召回率。韩语的黏着语结构导致字节对编码（BPE）分词器经常将个人身份信息（PII）与相邻的助词合并，这使得准确恢复文本跨度变得复杂。此外，现有的PII分类体系对法律文本缺乏足够的粒度。为解决这些问题，本研究提出了Thunder-DeID+，它将精炼的三级PII分类体系与边界感知分词器相结合。该分词器通过引入N-gram和边界评分组件来增强预训练的BPE词表，以更好地对齐分词与PII跨度的边界。所提出的方法采用与分词器无关的词段级评估协议进行评估。在完整长度的判决书上，该分词器持续……

    arXiv:2610.22733v1 Announce Type: new  Abstract: This study examines the transferability of a de-identification model developed for Korean court judgments to a broader range of sensitive documents, focusing on boundary precision, label granularity, and recall. The agglutinative structure of Korean causes Byte Pair Encoding (BPE) tokenizers to frequently merge Personally Identifiable Information (PII) with adjacent particles, which complicates accurate span recovery. Furthermore, current PII taxonomies lack sufficient granularity for legal texts. To address these issues, this study introduces Thunder-DeID+, which integrates a refined three-tier PII taxonomy with a boundary-aware tokenizer. The tokenizer enhances a pretrained BPE vocabulary by incorporating N-gram and boundary-scoring components to better align token and PII span boundaries. The proposed approach is evaluated using a tokenizer-independent word-segment-level protocol. On full-length judgments, the tokenizer consistently i
    
[^100]: TokenEconomizer：面向编程代理的经验驱动黑盒成本控制

    TokenEconomizer: Experience-Driven Black-Box Cost Control for Coding Agents

    [https://arxiv.org/abs/2610.22732](https://arxiv.org/abs/2610.22732)

    TokenEconomizer是一个经验驱动的外部黑盒控制器，通过注入操作微技能、剪除无关工具和干预失败循环，显著降低编程代理在连续现实任务序列中的环境相关token成本。

    

    前沿编程代理的进展大幅增加了AI辅助软件工程的token成本。它们的token效率通常是在孤立任务上进行评估的，而现实中的开发者则使用它们来持续构建、更新和维护大规模项目。这种不匹配促使我们在一系列现实编程任务序列上评估代理，我们发现代理会表现出反复出现的、依赖环境的token成本，而现有方法难以降低这些成本。为了最小化这些成本，我们引入了TokenEconomizer，这是一个外部控制器，能够在任务序列中削减特定环境的token成本。这些削减来自于三方面的决策：(i) 注入哪些操作性微技能，(ii) 剪除哪些持续无关的工具，以及 (iii) 何时干预无成效的失败循环。在Claude Code、Codex和GitHub Copilot CLI与前沿模型配对使用的情况下，TokenEconomizer平均降低了token成本。

    arXiv:2610.22732v1 Announce Type: new  Abstract: Advances in frontier coding agents have sharply increased the token costs of AI-assisted software engineering. Their token efficiency is typically evaluated on isolated tasks, whereas real-world developers use them to continuously build, update, and maintain large-scale projects. This mismatch motivates us to evaluate agents over sequences of realistic coding tasks, where we find that agents exhibit recurring environment-dependent token costs that existing methods struggle to reduce. To minimize these costs, we introduce TokenEconomizer, an external controller that cuts environment-specific token costs over a sequence of tasks. These cuts come from determining (i) which operational micro-skills to inject, (ii) which persistently irrelevant tools to prune, and (iii) when to intervene in unproductive failure loops. Across Claude Code, Codex, and GitHub Copilot CLI paired with frontier models, TokenEconomizer reduces the average token cost 
    
[^101]: 一本虚假信息“策略手册”能提升判断力而不滋生犬儒主义

    A misinformation playbook sharpens judgement without breeding cynicism

    [https://arxiv.org/abs/2610.22724](https://arxiv.org/abs/2610.22724)

    本研究从51种语言的21万多条事实核查声明中提炼出虚假信息的说服策略手册，学习该手册可提升人类和AI对陌生信息的辨别能力，且不会让人和机器对真实信息产生过度怀疑。

    

    虚假信息危害公共健康与民主，而生成式人工智能如今能写出与人类一样具有说服力的谎言。现有防御手段从对个别声明进行事实核查，到训练人们抵御虚假信息，但许多声明因太新而无法核查，且教导人们存疑的干预措施也可能同时降低人们对真相的信任。我们提出了一种防御方法，其针对的不是每条声明的事实本身，而是在各类声明中反复出现的说服策略。本研究表明，通过学习虚假信息的说服策略手册（该手册基于51种语言的214,646条经事实核查的声明绘制而成），能够提升人类和机器对陌生声明的判断力，同时不会让两者对真实声明变得更加怀疑。该手册涵盖社会影响、信息操纵和情感诉求三类策略，并以恐惧、阴谋和捏造为核心组合而成。在了解该手册并获得“有说服力不等于虚假”的提醒后，语言模型能更好地……（原文摘要在此处截断）

    arXiv:2610.22724v1 Announce Type: cross  Abstract: Misinformation harms public health and democracy, and generative artificial intelligence now writes falsehoods as persuasive as people's. Defences range from fact-checking individual claims to training people to resist falsehoods. Yet many claims are too new to check, and interventions that teach doubt can reduce belief in truth too. We propose a defence that targets not the facts of each claim but the persuasion strategies that recur across them. Here we show that learning misinformation's persuasion playbook, mapped from 214,646 fact-checked claims in 51 languages, can sharpen human and machine judgement of claims they know little about, without making either more cynical about true claims. The playbook spans social influence, information manipulation and emotional appeals, combined around a core of fear, conspiracy and fabrication. Briefed with it, plus a reminder that persuasion is not proof of falsehood, a language model better di
    
[^102]: CoLU：面向大语言模型的置信度加权逐层遗忘方法

    CoLU: Confidence-Weighted Layerwise Unlearning for Large Language Models

    [https://arxiv.org/abs/2610.22721](https://arxiv.org/abs/2610.22721)

    该论文提出CoLU方法，通过置信度加权的遗忘与保留正则化仅更新单个选定的Transformer层来实现大语言模型遗忘，并配合基于归一化遗忘-保留分数的简短层筛选策略高效选取最优层，在降低计算成本的同时更好地平衡了遗忘效果与知识保留。

    

    大语言模型（LLM）遗忘旨在让模型遗忘特定信息的同时保留其他信息，而无需从头重新训练。当前最先进的方法如NPO、SimNPO和SatImp通过全模型微调来平衡遗忘与保留，而FOM-UL则探索了选择性层更新策略。先前的研究表明，抑制目标信息可能会损害被保留的知识，并且即使使用相同的遗忘损失，这种权衡在不同层之间也存在差异。在本工作中，我们提出了置信度加权逐层遗忘方法，该方法将一种新颖的置信度加权遗忘与保留正则化相结合，仅更新单个选定的Transformer层，而非对整个模型进行遗忘。为了降低CoLU的计算成本，我们进一步提出了简短层筛选，该方法利用新定义的归一化遗忘-保留分数，通过在遗忘数据集的一个子集上以更少的训练轮次进行简短的遗忘试验来对各个层进行排序。

    arXiv:2610.22721v1 Announce Type: new  Abstract: Large language model (LLM) unlearning aims to forget certain information while preserving others, without retraining from scratch. SoTA methods such as NPO, SimNPO, and SatImp balance forgetting and retain through whole model fine-tuning, while FOM-UL explores selective layer updates. Prior works show that suppressing target information can damage retained knowledge. This trade-off can vary across layers even under the same unlearning loss. In this work, we propose Confidence-Weighted Layerwise Unlearning (CoLU), which combines a novel confidence-weighted forgetting and retain regularization while updating a single selected transformer layer, rather than the whole model for unlearning. To reduce CoLU compute cost, we further propose Brief Layer Screening (BLS), which uses a newly defined normalized forgetting-retention (NFR) score to first rank layers through a brief unlearning trials using a subset of forgetting dataset and fewer epochs
    
[^103]: ARBOR：通过推理状态追踪定位长思维链中的推理失败

    ARBOR: Locating Reasoning Failures in Long Chains of Thought through Reasoning State Tracking

    [https://arxiv.org/abs/2610.22717](https://arxiv.org/abs/2610.22717)

    ARBOR提出了一种推理失败归因框架，通过将推理轨迹表示为推理状态树并从全局视角追踪步骤与前序状态之间的关系，从而精确定位长思维链中的推理失败，并将已知失败模式细化为机制不同的子类型。

    

    大型推理模型通过长思维链解决复杂问题，已有研究记录了诸如过度思考、思考不足和无依据推理等失败模式。现有方法从局部信号检测这些失败，例如与相邻步骤的文本相似性、切换标记，或大语言模型对某一步骤正确性的判断。然而，一个步骤对解决问题的贡献取决于先前步骤所建立的状态，因此识别失败需要一个跨越整个轨迹追踪该状态的全局视角。为此，我们提出了ARBOR——一个推理失败归因框架，它将每条推理轨迹表示为一棵推理状态树，该树随着每个步骤对状态的创建、验证、剪枝或重访而构建，并将每个失败归因于某一步骤与其前序状态之间的关系。借助ARBOR，我们将这三种失败模式细化为通过不同机制损害推理的多个子类型。

    arXiv:2610.22717v1 Announce Type: cross  Abstract: Large reasoning models solve complex problems through long chains of thought, in which prior work has documented failure modes such as overthinking, underthinking, and unsupported deductions. Existing methods detect these failures from local signals: textual similarity to nearby steps, switch tokens, or an LLM's judgment of a step's correctness. However, what a step contributes to the solution depends on the state that earlier steps have established, so identifying failures requires a global view that tracks that state across the trajectory. Therefore, we propose ARBOR, a reasoning-failure attribution framework that represents each trajectory as a tree of reasoning states, built as each step creates, verifies, prunes, or revisits a state, and attributes each failure to a relation between a step and the state preceding it. Using ARBOR, we refine the three failure modes into distinct subtypes that harm reasoning through different mechani
    
[^104]: 风险与拒答：大语言模型对自杀意念响应的机制研究

    Risk and Refusal: Mechanisms of LLM Responses to Suicidal Ideation

    [https://arxiv.org/abs/2610.22715](https://arxiv.org/abs/2610.22715)

    该研究揭示了大语言模型拒答自杀意念披露本身可能构成安全失效，其风险主要源于对披露内容的忽视，且拒答机制与拒绝有害请求共享相同的内部表征方向，而通过告知模型上下文或直接抑制拒答可在不增加风险的前提下提升模型回应参与度。

    

    拒答是大语言模型（LLM）中常见的安全机制，然而在自杀意念（SI）的情境下，拒绝回应本身可能构成一种安全失效。我们在三个数据集上评估了八个指令微调的大语言模型如何应对自杀意念和自残披露，改变了模型是否被告知披露内容涉及自杀意念，以及是否为模型提供临床医生开发的响应协议，并在六个开源权重模型上补充了机制分析。我们的结果揭示了拒答与响应层面风险之间的微妙关系。在基线情况下，模型响应中风险的升高主要由对披露内容的忽视以及拒绝回应所驱动。机制分析表明，模型可能通过与其拒绝有害请求相同的内部方向来拒绝对自杀意念披露的回应。然而，告知模型自杀意念的上下文，或直接抑制拒答行为，可以在不增加风险的情况下提高模型的回应参与度。

    arXiv:2610.22715v1 Announce Type: cross  Abstract: Refusal is a common safety mechanism in large language models (LLMs), yet in contexts of suicidal ideation (SI), refusing to engage can itself constitute a safety failure. We evaluate how eight instruction-tuned LLMs respond to SI and self-harm disclosures across three datasets, varying whether models are informed that a disclosure expresses SI and whether they receive a clinician-developed response protocol, and complement this with mechanistic analyses in six open-weight models. Our results reveal a nuanced relationship between refusal and response-level risk. At baseline, elevated risk in model responses is driven primarily by inattention to the disclosure, alongside refusal to engage. Mechanistic analyses suggest that models may refuse SI disclosures through the same direction that mediates their refusals of harmful requests. However, informing models of the SI context, or suppressing refusal directly, increases engagement without 
    
[^105]: 首次通过注意力

    First Passage Attention

    [https://arxiv.org/abs/2610.22714](https://arxiv.org/abs/2610.22714)

    该论文提出首次通过注意力（FPA），将注意力衰减机制从路径求和转向路径极值——根据学习标量场在中间跨度上的最深下凹来惩罚键的logit，从而能在保留较旧结构相关token的同时选择性擦除中间上下文，并证明常数维度的FPA头即可实现任意存活键子集，而传统softmax头或token级遗忘门需要与键数呈线性关系的维度。

    

    使用遗忘门增强softmax注意力通常依赖于沿键-查询路径对衰减贡献进行求和。由于这种累加是单调的，这些机制无法在保留较旧的、结构上相关的token的同时，选择性地擦除中间的上下文。我们提出首次通过注意力，它将从路径求和转向路径极值：通过将某个键的logit按比例降低，降低量与中间跨度上学习到的标量场低于该键水平的最深下凹成正比。FPA将遗忘Transformer（FoX）作为特例包含在内，并且当标量场和水平遵循共享的学习随机游走时，具有未受惩罚logit的存活键自然形成一个栈。我们证明，一个常数维度的FPA头可以实现由后续下凹所留下的任意存活键子集，而带有位置偏置或token级遗忘门的softmax头则需要与键数量呈线性关系的头维度。在合成实验中

    arXiv:2610.22714v1 Announce Type: new  Abstract: Augmenting softmax attention with forget gates typically relies on summing decay contributions along the key--query path. Because such accumulation is monotonic, these mechanisms cannot selectively erase intermediate context while preserving older, structurally relevant tokens. We propose First Passage Attention (FPA), which shifts from path summation to path extrema by lowering a key's logit proportionally to the deepest dip of a learned scalar field below that key's level over the intervening span. FPA encompasses the Forgetting Transformer (FoX) as a special case, and when the field and levels follow a shared learned walk, the alive keys with unpenalised logits naturally form a stack. We show that a constant-dimension FPA head realises any subset of keys left alive by subsequent dips, whereas softmax heads with positional biases or token-level forget gates require a head dimension linear in the number of keys. In synthetic experiments
    
[^106]: GlobeBench：评估基于文本的世界模型在可扩展环境模拟中的能力

    GlobeBench: Evaluating Text-Based World Model Capabilities for Scalable Environment Simulations

    [https://arxiv.org/abs/2610.22708](https://arxiv.org/abs/2610.22708)

    GlobeBench 是首个用于评估基于文本的世界模型能力的标准化基准，包含跨五个领域和十九项模拟技能的 1,632 个任务，并采用确定性门控与语义裁判相结合的混合评估架构。

    

    基于文本的世界模型最近展现出为多种用例模拟复杂强化学习（RL）环境状态转移的潜力。然而，目前尚无标准化基准来比较、研究和改进这些模型。本文试图通过构建 GlobeBench 来解决这一问题，这是一个新颖的基准，包含跨五个独立领域和十九项模拟技能的 1,632 个任务。这些任务通过理解先前的状态变化、遵循环境规则以及预测动作对相关和无关对象的影响，来检验模型在给定完整智能体轨迹的情况下能否预测环境对特定动作的响应。我们将预测的环境响应与通过执行得到的参考结果进行比较，采用结合了确定性门控和语义裁判的混合评估架构。在 20 个通用大语言模型和专用文本世界模型中，GPT-6 Astra 取得了最高的任务成功率，达到 49（%）（原文摘要在此处截断）。

    arXiv:2610.22708v1 Announce Type: new  Abstract: Text-based world models have recently shown potential to simulate complex RL environment state transitions for a diverse set of use cases. However, no standardized benchmark exists to compare, study and improve these models. Through this paper, we attempt to solve this problem by building GlobeBench, a novel benchmark with 1,632 tasks across five separate domains and nineteen simulation skills. The tasks check whether models can predict the environment's response to a specific action given the full agent trajectory by understanding earlier state changes, following environment rules, and predicting the effects of actions on both related and unrelated objects. We compare the predicted environment response with an execution-derived reference, using a hybrid architecture with both deterministic gates and a semantic judge. Among 20 general-purpose LLMs and specialized text world models, GPT-6 Astra achieves the highest task success rate at 49
    
[^107]: 分别探索，择优采纳：面向大型工具库的执行引导式工具使用

    Explore Separately, Adopt Selectively: Execution-Guided Tool Use over Large Tool Libraries

    [https://arxiv.org/abs/2610.22688](https://arxiv.org/abs/2610.22688)

    提出BRIA智能体，通过将候选工具调查隔离到独立分支、检查执行证据并仅将有用的结果采纳回主轨迹，解决了大规模工具库中多候选工具选择时失败尝试污染推理轨迹的问题。

    

    大型语言模型智能体在任务执行过程中出现新需求时，必须从包含数千个工具的大型工具库中发现相关工具。当多个候选工具都可能满足同一需求时，可能需要通过实际执行来识别真正适用的工具。在共享轨迹中调查候选工具可能导致部分备选方案未被探索，或使失败的尝试影响后续推理。为解决这一问题，我们提出了BRIA（BRanch-Inspect-Adopt，分支-检查-采纳），一个简单而有效的智能体，它将候选工具的调查暂时隔离在独立的分支中。每个分支检索并执行工具以调查其对应的候选工具；任务智能体检查由此产生的执行证据，并只将有用的结果采纳到其继续执行的主轨迹中。BRIA还能在任务执行过程中出现新的能力需求时动态检索更多工具。我们在PlanBench-XL和MCP-Atlas基准上，使用包含1897个工具的共享工具池、三个语言模型以及静态分析设置对BRIA进行了评估。

    arXiv:2610.22688v1 Announce Type: cross  Abstract: Large language model agents must discover relevant tools from large libraries, e.g., thousands of tools, as new needs arise during task execution. When several candidates could satisfy the same need, execution may be required to identify the relevant one. Investigating candidates in a shared trajectory can leave alternatives unexplored or allow failed attempts to influence subsequent reasoning. To address this, we present BRIA (BRanch-Inspect-Adopt), a simple yet effective agent that temporarily isolates candidate investigations in separate branches. Each branch retrieves and executes tools to investigate its candidate; a task agent inspects the resulting execution evidence and adopts only useful results into its continuing trajectory. BRIA retrieves additional tools as new capability needs arise during task execution. We evaluate BRIA on PlanBench-XL and MCP-Atlas using a shared pool of 1897 tools, three language models, and static an
    
[^108]: 面向长时程具身智能体的多模态文件系统记忆

    Multimodal Filesystem Memory For Long-Horizon Embodied Agents

    [https://arxiv.org/abs/2610.22686](https://arxiv.org/abs/2610.22686)

    提出多模态文件系统记忆架构MFM，通过情景、空间、语义、程序和工作记忆等互补记忆类型以及可搜索的文本记录，帮助长时程具身智能体有效存储并检索多模态经验。

    

    在长时间跨度内运行的智能体需要有效的记忆机制，使其能够在漫长的交互过程中重复利用自身经验。如果保留交互的完整历史记录，将难以找到相关的经验；而将其压缩为摘要，则可能丢失后续所需的视觉和空间细节。我们提出了MFM，一种多模态文件系统记忆架构，用于保存互补类型的智能体记忆。情景记忆存储观察、动作和结果；空间记忆将经验与位置关联起来；语义记忆和程序记忆积累一般性知识和可复用的行动方法；工作记忆则保存当前的计划。MFM利用可搜索的文本记录将这些不同的表示与底层的多模态情节片段连接起来。智能体能够通过……找到以往的经验。

    arXiv:2610.22686v1 Announce Type: new  Abstract: Agents operating over long horizons need an effective memory that allows them to reuse their experience throughout long interactions. If the entire history of an interaction is retained, it becomes difficult to find the relevant experience, while compressing it into summaries may result in the loss of visual and spatial details that are required later on. We present \textbf{MFM}, a multimodal filesystem memory architecture that keeps complementary types of agent memory. \textbf{Episodic memory} stores the observations, actions, and outcomes, and \textbf{spatial memory} associates experiences with locations. \textbf{Semantic and procedural memory} build up general knowledge and reusable methods of acting, and \textbf{working memory} holds the current plans. MFM makes use of searchable textual records to connect these different representations to the underlying multimodal episodes. The agents are able to find previous experiences by means 
    
[^109]: 并非所有图都是地图：一种面向移动GUI导航的任务无关探索框架

    Not All Graphs Are Maps: A Task-Agnostic Exploration Framework for Mobile GUI Navigation

    [https://arxiv.org/abs/2610.22672](https://arxiv.org/abs/2610.22672)

    TAG-Nav是一个任务无关的图导航框架，通过将离线探索提炼为可复用的应用功能、目标页面和路线记忆，使GUI智能体无需任务特定演示即可完成目的地导航。

    

    现代移动GUI智能体能够感知可见屏幕并执行局部操作，但它们仍然缺乏持久的应用特定导航知识：功能位于何处、哪些页面包含这些功能、以及哪些屏幕在重复访问中是等价的。我们提出了TAG-Nav，一个任务无关的图导航框架，它通过可复用的目的地记忆来包装现有的GUI智能体骨干。在看到任何被评估的指令之前，离线探索被提炼成关于应用功能、目标页面和路径点路线的紧凑记录。当未来任务的目的地已被覆盖时，TAG-Nav检索该目的地并提供路线引导，而无需任务特定的演示。这将移动GUI目的地导航与完整任务完成分离开来：许多任务共享到达正确页面这一前缀，而最终操作可能是进入内容、发送数据、进行购买、删除信息或更改敏感设置。为了评估路线...

    arXiv:2610.22672v1 Announce Type: new  Abstract: Modern mobile GUI agents can perceive visible screens and execute local actions, but they still lack persistent app-specific navigation knowledge: where functions live, which pages contain them, and which screens are equivalent across repeated visits. We introduce TAG-Nav, a Task-Agnostic Graph Navigation framework that wraps existing GUI-agent backbones with reusable destination memory. Offline exploration is distilled into compact records of app functions, target pages, and waypoint routes before any evaluated instruction is seen. When a future task's destination is covered, TAG-Nav retrieves the destination and provides route guidance without task-specific demonstrations. This separates mobile GUI destination navigation from full task completion: many tasks share the prefix of reaching the right page, while final actions may enter content, send data, make purchases, delete information, or change sensitive settings. To evaluate route f
    
[^110]: ProPerAgent：迈向记忆驱动的主动性助手

    ProPerAgent: Toward Memory-Driven Proactive Assistance

    [https://arxiv.org/abs/2610.22671](https://arxiv.org/abs/2610.22671)

    该论文提出ProPerMem基准，首次要求AI助手从长达一年、充满干扰的个人记忆中自主挖掘证据来决定是否及何时主动介入，发现现有模型（含前沿模型）表现极差（最高仅15%），而提供现成证据时性能可达96%，揭示了记忆证据挖掘是主动性助手的核心瓶颈。

    

    随着助手从被动应答转向主动干预，记忆的角色发生了变化：它不再仅仅为查询提供答案，而是决定了系统是否以及何时该开口说话。以往在这一设定下的工作是把相关事实直接放进上下文中交给系统；我们研究的是使问题真正困难的情形，即证明干预合理性的证据必须从长时段的个人记忆中挖掘出来。我们提出了ProPerMem，一个面向记忆驱动主动性助手（MPA）的基准，它将第一人称视角视频与长达一年、充满干扰信息的用户记忆（平均约618K token）、经人工验证的触发窗口，以及一个对内容与时机进行联合评分的协议相配对。我们评估的所有基线模型表现都很差（最高仅15%），包括在给定摘要式或检索增强记忆的情况下的前沿模型。一项受控研究解释了原因：当神谕证据在触发窗口内直接提供时，GPT-5.3-Chat能达到96%；而让它自行选择证据时性能则大幅下降

    arXiv:2610.22671v1 Announce Type: new  Abstract: As assistants move from reactive answering toward proactive intervention, memory changes role: it no longer only supplies the answer to a query, it determines whether and when to speak at all. Prior work in this setting hands the system the relevant fact in context; we study the regime that makes the problem hard, where the evidence justifying an intervention must be mined from a long-horizon personal memory. We introduce ProPerMem, a benchmark for memory-driven proactive assistance (MPA), pairing egocentric video with year-long, distractor-rich user memories (~618K tokens on average), human-verified trigger windows, and a protocol that scores content and timing jointly. Every baseline we evaluate performs poorly (at most 15%), including frontier models given summarized or retrieval-augmented memory. A controlled study explains why: with oracle evidence delivered inside the trigger window GPT-5.3-Chat reaches 96%; letting it choose the t
    
[^111]: 探索合成数据的实证隐私图景

    Exploring The Empirical Privacy Landscape of Synthetic Data

    [https://arxiv.org/abs/2610.22659](https://arxiv.org/abs/2610.22659)

    本研究通过对合成医疗记录的受控比较发现，从敏感记录生成合成数据（尤其是结合隐私感知改写）比直接微调或 DP-SGD 训练能更好地平衡语言模型的能力与隐私保护。

    

    在敏感数据上训练语言模型的方法，在能力提升与泄露隐私信息风险之间的权衡上各有不同。本文中，我们在带有金丝雀（canaries）以探测泄露的合成医疗记录上，对数据层面和训练层面的干预措施进行了受控比较，以研究能力与隐私之间的权衡。直接微调在能力指标上表现最佳，但导致的隐私泄露最多；而 DP-SGD 虽然降低了泄露，却以能力的大幅损失为代价。从这些记录生成合成数据则提供了更有利的平衡，尤其是在“感知唯一性”指令的引导下替换私人细节时效果更佳。与朴素的改写方法相比，隐私感知改写在多种不同的改写模型上都能提升隐私保护水平。我们还发现，能力更强的改写模型通常能降低下游隐私泄露的风险，尽管留出集拟合（held-out fit）并不能（预测这一点）。

    arXiv:2610.22659v1 Announce Type: cross  Abstract: Methods for training language models on sensitive data differ in how they balance capability gains against the risk of leaking private information. In this paper, we present a controlled comparison of data-level and training-level interventions for navigating the capability-privacy tradeoff on synthetic medical records with canaries to probe for leakage. Direct finetuning performs best on capability metrics but leads to the most privacy leakage, while DP-SGD reduces leakage at substantial cost to capabilities. Generating synthetic data from these records offers a more favorable balance, particularly when guided by a uniqueness-aware instruction to replace private details. Relative to naive paraphrasing, privacy-aware rewriting improves privacy protection across a variety of different rewriting models. We also find that more capable rewriting models generally reduce the risks of downstream privacy leakage, although held-out fit does not
    
[^112]: 二手压力：压力性表露如何影响在线研究生申请社群中的心理社会福祉

    Secondhand Stress: How Stressful Disclosures Affect Psychosocial Wellbeing in Online Graduate Admissions Communities

    [https://arxiv.org/abs/2610.22657](https://arxiv.org/abs/2610.22657)

    基于压力交叉理论，本研究通过大规模分析Reddit研究生申请社群中的帖子与评论发现，与他人的压力表露进行互动的用户，后续会表现出更高的焦虑、抑郁和压力表达水平以及更活跃的参与行为。

    

    在线社群为经历压力性生活转变的个人提供了倾诉困难和寻求同伴支持的空间。然而，接触他人的痛苦也可能对社群成员产生心理社会层面的影响。基于压力交叉理论，我们研究了在研究生申请过程中与同伴的痛苦表露进行互动，如何与后续的心理社会表达和参与行为相关联，以及类似的关联是否会通过下游互动延伸开来。我们分析了来自两个Reddit社群、跨越三个申请周期的171,981条帖子和1,130,413条评论，采用倾向得分分层和基线调整比较方法。结果显示，在申请准备阶段与痛苦表露互动的用户，随后表现出更高的、由分类器估计的焦虑、抑郁和压力表达，同时评论活动更为频繁、词汇冗余度更高。这些关联在各个申请周期中反复出现。

    arXiv:2610.22657v1 Announce Type: new  Abstract: Online communities provide spaces for individuals navigating stressful life transitions to disclose difficulties and seek peer support. Yet encountering others' distress may also have psychosocial implications for community members. Drawing on stress crossover theory, we examine how engagement with peer distress during graduate admissions is associated with subsequent psychosocial expression and participation, and whether similar associations extend through downstream interactions. We analyze 171,981 posts and 1,130,413 comments from two Reddit communities across three admissions cycles, using propensity-score stratification and baseline-adjusted comparisons. Users who engaged with distress disclosures during application preparation subsequently exhibited higher classifier-estimated anxiety, depression, and stress expression, alongside greater commenting activity and lexical redundancy. These associations recurred across admissions cycle
    
[^113]: 面向可靠消费银行搜索的证据门控双语RAG框架

    Evidence-Gated Bilingual RAG for Reliable Consumer Banking Search

    [https://arxiv.org/abs/2610.22652](https://arxiv.org/abs/2610.22652)

    本文提出一种完全本地化部署的证据门控双语RAG框架，通过结构化内容精炼、混合检索、意图感知证据选择、重排序和三分类决策机制，将消费银行搜索的Top-1命中率从50.48%提升至79.05%，并有效支持阿拉伯语与英语的双语查询。

    

    消费者在描述自身需求时，往往不会使用银行网站上的专业术语。例如，一个自然语言请求（“一张用于旅行的卡”）可能与相关产品页面的关键词并不重合，而在双语服务中问题更加复杂，因为同一意图可能用阿拉伯语或英语表达。本文提出了一种面向消费银行搜索的证据门控双语检索增强生成（RAG）框架，该框架完全运行于本地基础设施之上。该框架结合了结构化网页内容精炼、密集-词法混合检索、意图感知的证据选择、交叉编码器重排序，以及“回答 / 相关但不匹配 / 无答案”的三分类决策机制。在210条带URL标注的阿拉伯语和英语查询上，与混合基线相比，所提出的方法将Top-1命中率从50.48%提升至79.05%，MRR@5从0.571提升至0.797。此外，在另一项包含26个案例的端到端评估中达到了69.2%的准确率。

    arXiv:2610.22652v1 Announce Type: new  Abstract: Consumers often describe what they need rather than use the terminology found on a banking website. For example, a natural-language request ("a card for travel") may not share the keywords of the relevant product page, and the problem is harder in bilingual services, where the same intent may be expressed in Arabic or English. This paper presents an evidence-gated bilingual retrieval-augmented generation (RAG) framework for consumer banking search that operates entirely on local infrastructure. The framework combines structured web-content refinement, hybrid dense-lexical retrieval, intent-aware evidence selection, cross-encoder reranking, and a three-outcome Answer / Related Mismatch / No Answer decision mechanism. On 210 URL-labeled Arabic and English queries, the proposed approach improves Top-1 hit rate from 50.48% to 79.05% and MRR@5 from 0.571 to 0.797 over a hybrid baseline. A separate 26-case end-to-end evaluation achieved 69.2% 
    
[^114]: SAFETYEDIT：用于大语言模型越狱后门的稀疏内部目标与提示级控制

    SAFETYEDIT: Sparse Internal Targets with Prompt-Wide Control for LLM Jailbreak Backdoors

    [https://arxiv.org/abs/2610.22640](https://arxiv.org/abs/2610.22640)

    提出SAFETYEDIT，以稀疏的拒绝相关MLP坐标作为内部目标并实现提示级控制，在更小的内部编辑目标下获得比稠密表示方向更优的后门有效性-保持性权衡。

    

    模型编辑后门旨在触发条件下诱导有害响应，同时在未触发的普通输入上保持正常行为。近期的模型编辑方法通过对模型权重的一小部分子集进行秩一更新，以针对肯定性响应前缀。然而，当这一前缀级目标被满足时，模型仍可能回退到拒绝状态，或生成连贯、表面上合规却并未真正回应请求的内容。为克服这一局限，现有的内部表示方法将焦点从输出词元转移到潜空间中稠密的高维合规方向。现有的表示级方法依赖保留约束来维持条件性，这促使我们思考：更小的内部目标能否带来不同的有效性-保持性权衡。受单坐标干预具备相当控制能力的启发，我们研究了一种稀疏的、与拒绝相关的MLP坐标（原文在此处截断）。

    arXiv:2610.22640v1 Announce Type: new  Abstract: Model-editing backdoors aim to induce harmful responses under a trigger while preserving ordinary behavior on untriggered input. Recent model-editing approaches target affirmative response prefixes through rank-one updates to a small subset of model weights. Yet when this prefix-level target is satisfied, the model may still revert to refusal or produce a coherent, apparently compliant response that does not address the request. To overcome this limitation, existing internal-representation approaches shift focus from output tokens to dense, high-dimensional compliance directions in latent space. Existing representation-level approaches rely on preservation constraints to maintain conditionality, motivating us to ask whether a smaller internal target can offer a different effectiveness-preservation trade-off. Motivated by the comparable control capability of single-coordinate interventions, we investigate a sparse refusal-related MLP coor
    
[^115]: 结合文本属性从跨语言共词化模式中归纳学习概念嵌入

    Inductive Learning of Concept Embeddings from Cross-Linguistic Colexification Patterns with Textual Attributes

    [https://arxiv.org/abs/2610.22638](https://arxiv.org/abs/2610.22638)

    提出了一种结合文本属性的归纳式框架，从跨语言共词化模式中学习概念嵌入，克服了以往直推式嵌入方法无法为训练数据之外的新概念生成嵌入的可扩展性局限。

    

    分布语义学作为计算语义学框架的出现，标志着表示方式的范式转变，意义被表示为通常称为词向量的密集向量。另一方面，面向历史或类型学语言比较的计算方法仍大多依赖离散表示，基于嵌入的方法在这一领域的潜力在很大程度上尚未被开发。最近的研究基于跨语言共词化模式网络，提出了概念嵌入作为词向量的语言无关、以名为义的对应物。虽然由此产生的概念嵌入反映了有意义的语义属性和关系，但其可扩展性不足，因为它们依赖于直推式嵌入方法，只能为训练数据中存在的节点学习嵌入。在本研究中，我们提出了一个归纳式……（摘要原文在此处被截断）

    arXiv:2610.22638v1 Announce Type: new  Abstract: The advent of distributional semantics as a framework for computational semantics marked a paradigm shift in the nature of representations, by which meanings are represented as dense vectors commonly known as word embeddings. Computational approaches to historical or typological language comparison, on the other hand, still mostly rely on discrete representations, while the potential of embedding-based methods remains largely untapped in this context. Recent studies have proposed concept embeddings as a language-independent, onomasiological counterpart to word embeddings, based on networks of cross-linguistic colexification patterns. While the resulting concept embeddings reflect meaningful semantic properties and relations, they suffer from a lack of scalability, since they rely on transductive embedding approaches which only learn embeddings for nodes present in the training data. In this study, we present an inductive framework for le
    
[^116]: 通过因果鲁棒性测量大语言模型的机制不确定性

    Measuring Mechanistic Uncertainty in Large Language Models via Causal Robustness

    [https://arxiv.org/abs/2610.22636](https://arxiv.org/abs/2610.22636)

    该论文提出“机制不确定性”概念及“因果鲁棒性裕度”指标，通过测量改变模型输出所需的最小内部干预量来量化大语言模型的不确定性，发现错误或不确定的响应裕度更小，且该指标与传统基于输出的不确定性方法形成有效互补。

    

    大语言模型（LLMs）在关键领域中的应用日益增多，其容易产生幻觉的特性使得可靠且可解释的不确定性量化变得至关重要。我们提出了“机制不确定性”这一概念，通过支持模型响应的内部机制的稳定性来刻画不确定性。我们提出了“因果鲁棒性裕度”，其定义为使模型输出发生偏离所需的最小内部干预量，并使用Jacobian Lens（Jlens）概念引导向量对该指标进行操作化实现。在三个开源模型和五个问答数据集上的实验表明，错误和不确定的响应表现出更小的裕度，说明其对内部干预更为敏感，且因果鲁棒性裕度与传统的基于输出的不确定性方法形成互补。最后，在一项受控的知识注入实验中，被正确学习的事实显示出更高的裕度。

    arXiv:2610.22636v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in critical domains, where their susceptibility to hallucinations makes reliable and interpretable uncertainty quantification essential. We introduce the concept of mechanistic uncertainty, which characterizes uncertainty through the stability of the internal mechanisms supporting a model's response. We propose the causal robustness margin, defined as the minimum internal intervention required to deviate the model's output, and operationalize it using Jacobian Lens (Jlens) concept steering vectors. Across three open-source models and five question-answering datasets, we show incorrect and uncertain responses exhibit smaller margins, indicating greater sensitivity to internal interventions, and the causal robustness margin complements conventional output-based uncertainty methods. Finally, in a controlled knowledge injection experiment, correctly learned facts show increased margins aft
    
[^117]: ToMStories：解读合成叙事中的心智理论

    ToMStories: Interpreting Theory-of-Mind in Synthetic Narratives

    [https://arxiv.org/abs/2610.22632](https://arxiv.org/abs/2610.22632)

    该论文提出了ToMStories评估数据集（约6,000个合成叙事短篇，考察情绪、恐惧、动机、情感创伤四类心理状态），发现最大的语言模型在多选题上可媲美人类，但在开放式问题上仍有差距，且模型在隐式心理状态的解释与生成上存在简化现象。

    

    理解他人往往需要从其行为方式来推断他们的感受或愿望，而非依据其直白的陈述。我们探究语言模型能否做出这类推断，以及它们是如何做到的。我们推出了心智理论故事，这是一个包含约6,000个合成生成的短篇叙事小故事的评估数据集，每个故事都配有一道关于角色情绪、恐惧、动机或情感创伤的多选题或开放式问题。这些心理状态以隐式（展示）、显式（讲述）或对话与描述相互冲突的方式呈现。通过对13个开放权重模型（参数规模从1B到72B）进行评估，我们发现最大的模型在多选题上能达到人类水平，但在开放式问题上则不然。我们的分析揭示了模型在解释和生成隐式心理状态时所做的简化处理，表明合成数据可以帮助我们理解模型如何执行这类推断。

    arXiv:2610.22632v1 Announce Type: cross  Abstract: Understanding people often requires inferring what they feel or want from how they behave rather than from what they state outright. We ask whether language models can make these inferences, and how they do so. We introduce Theory-of-Mind Stories (ToMStories), an evaluation dataset of ~6,000 short synthetically generated narrative vignettes, each paired with a multiple-choice or open-ended question about a character's emotion, fear, motivation, or emotional wound. These psychological states are conveyed implicitly (showing), explicitly (telling), or with conflict between dialogue and description. Evaluating thirteen open-weight models (1B-72B parameters), we find that the largest models match humans on multiple choice but not open-ended questions. Our analyses point to simplifications models make in both interpretation and generation of implicit psychological states, indicating that synthetic data can help us understand how models perf
    
[^118]: 貌似合理的解释就足以误导多模态评审模型

    A Plausible Explanation Is Enough to Mislead Multimodal Judges

    [https://arxiv.org/abs/2610.22631](https://arxiv.org/abs/2610.22631)

    提出JUPE基准并揭示“合理性俘获”现象——仅仅貌似合理的普通解释就足以误导多模态大模型评审接受错误答案，其影响与刻意说服性论证相当。

    

    多模态大语言模型越来越多地在训练和评估流程中充当“评审”角色，通常会同时评估答案及其解释。尽管AI评审对文本因素较为敏感，但普通貌似合理的解释是否能够左右那些需要视觉验证的判断，目前仍不清楚。我们提出了JUPE，一个视觉问答（VQA）风格的基准，它在固定答案的同时改变解释。在14个多模态大模型评审上，听起来合理的普通解释即可误导评审，其平均效果与刻意构造的说服性论证相当。简单的理由说明和思维链式论证分别使错误答案被拒绝的概率降低了12.7和13.8个百分点。我们将这种失败现象称为“合理性俘获”。借助信号检测理论，我们发现评审的判断标准发生了显著的均值偏移，趋向于接受答案，而平均辨别能力并无明显变化。

    arXiv:2610.22631v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) increasingly serve as judges in training and evaluation pipelines, often assessing answers together with explanations. Although AI judges are sensitive to textual factors, it remains unclear whether ordinary plausible explanations can sway judgments requiring visual verification. We introduce JUPE, a visual-question-answering (VQA)-style benchmark that fixes answers while varying explanations.   Across 14 MLLM judges, ordinary explanations that sound plausible mislead judges, with effects comparable on average to deliberately persuasive arguments. Plain justifications and chain-of-thought rationales reduce rejection of incorrect answers by 12.7 and 13.8 percentage points, respectively. We call this failure plausibility capture. Using signal detection theory, we find a significant mean shift in decision criterion toward acceptance, without a clear change in mean discriminability. Explanations swa
    
[^119]: 轨迹引导转向：缓解临床分类中的伪特征偏差

    Trajectory-Guided Steering: Mitigating Spurious Feature Bias in Clinical Classification

    [https://arxiv.org/abs/2610.22618](https://arxiv.org/abs/2610.22618)

    提出轨迹引导转向（TGS）方法，仅凭模型内部预测轨迹即可判断是否修正预测及修正方向，无需反事实数据或伪特征知识，在16个临床分类任务上修复了17.7%的错误且仅破坏1.3%的正确预测，并在所有超参数设置下保持净正向收益。

    

    大语言模型在临床分类中依赖伪特征：在临床记录中替换单个性别词元，会使多达53.5%的与性别无关的诊断结果发生改变。现有的推理时修正方法要么需要围绕特定伪特征构建的反事实数据，要么行为不可预测。我们观察到，模型对答案的偏好在各层之间的演化方式（即其预测轨迹）能够针对每个文档指示修正是否可能有所帮助。我们提出了轨迹引导转向（TGS），它仅凭轨迹即可决定是否修正每个预测以及修正的方向，无需反事实数据，也无需了解伪特征。在来自i2b2的16个临床分类任务上，TGS修复了17.7%的错误，同时仅破坏了1.3%的正确预测，并在所有测试的超参数设置下均保持净正向收益；而所有基线方法的净收益均为负或接近于零。

    arXiv:2610.22618v1 Announce Type: new  Abstract: Large language models rely on spurious features in clinical classification: swapping a single gender token in clinical notes alters gender-irrelevant diagnoses in up to 53.5% of cases. Existing inference-time corrections either need counterfactual data built around the specific spurious feature or behave unpredictably. We observe that how a model's preference between answers evolves across layers, its prediction trajectory, indicates for each document whether a correction is likely to help. We propose Trajectory-Guided Steering (TGS), which decides from the trajectory alone whether to correct each prediction and in which direction, with no counterfactual data and no knowledge of the spurious feature. On 16 clinical classification tasks from i2b2, TGS fixes 17.7% of errors while breaking only 1.3% of correct predictions and stays net-positive across all tested hyperparameter settings; all baselines are net-negative or near zero. The same 
    
[^120]: 不确定性何时能衡量RAG的幻觉风险？

    When Does Uncertainty Measure RAG Hallucination Risk?

    [https://arxiv.org/abs/2610.22615](https://arxiv.org/abs/2610.22615)

    论文提出并验证了一个条件性观点——不确定性只有在检索证据足以约束答案时才能有效衡量RAG的幻觉风险，实验表明困惑度在针对性无依据编辑检测中达到0.81–0.91 AUROC，且结合语义漂移评分可进一步提升校准效果。

    

    不确定性可以衡量检索增强生成（RAG）中的幻觉风险，但前提是检索到的证据对答案构成约束。当证据存在时，无依据的事实性声明出现的可能性会降低；而当检索覆盖不足时，模型可能依靠内部知识作答，从而给无依据的声明分配较高的概率。我们在HotpotQA和SQuAD数据集上，使用四个指令微调模型对这一假设进行了测试。在按字数区间匹配忠实回答与幻觉回答后，仅基于长度的分类器表现接近随机水平，而困惑度在针对性的无依据编辑上达到了0.81–0.91的AUROC。通过改变检索深度的实验揭示了该指标意义发生变化的原因：仅使用排名第一的段落时检测能力可能减弱，而加入第二个段落后又会恢复。结合不确定性与语义漂移的综合评分进一步改善了校准效果，且在RAGTruth QA上的测量结果仍高于随机水平。这些结果支持一种条件性的观点，即不确定性在特定条件下才是衡量幻觉风险的有效指标。

    arXiv:2610.22615v1 Announce Type: new  Abstract: Uncertainty can measure hallucination risk in retrieval-augmented generation (RAG), but only when retrieved evidence constrains the answer. When evidence is present, unsupported factual claims become less likely; when retrieval coverage is poor, the model can answer from internal knowledge and assign high probability to unsupported claims. We test this across four instruction-tuned models on HotpotQA and SQuAD. After matching faithful and hallucinated answers within word-count bins, length-only classifiers fall near chance, while perplexity reaches 0.81--0.91 AUROC on targeted unsupported edits. Varying retrieval depth shows why the score changes meaning: detection can weaken with only the top-ranked passage, then recover after adding a second passage. A combined uncertainty and semantic-drift score improves calibration, and measurements remain above chance on RAGTruth QA. Results support a conditional view of uncertainty as a useful mea
    
[^121]: 大语言模型能预测机器学习实验的结果吗？

    Can LLMs predict the outcomes of machine learning experiments?

    [https://arxiv.org/abs/2610.22599](https://arxiv.org/abs/2610.22599)

    该论文构建了包含288个LLM自生成实验想法的数据集，通过让五个前沿大语言模型在机器学习实验训练前预测其影响力并与实际结果对比，衡量了这些模型判断实验是否值得运行的直觉准确性。

    

    人工智能的最新进展使得递归自我改进（RSI）越来越具有可行性。大语言模型（LLM）已经能够提出、选择并运行机器学习（ML）实验。然而，机器学习中可能的设计选择空间非常庞大。提出实验的LLM需要具备可靠的直觉，以便在长训练时间的限制下判断某个实验是否值得运行。在这项工作中，我们通过评估当前前沿LLM在实验投入训练之前预测其自生成的机器学习实验想法影响力的准确程度，来衡量这种直觉。为此，我们构建了一个包含288个LLM提出的实验想法的数据集，涵盖八个MLE-bench任务。随后，我们提示五个前沿LLM（GPT-6 Astra、Claude Opus 5.5、Claude Fable 5.1、Kimi K3和GLM-5.3）预测每个想法的影响力，并将其预测与训练后的实际结果进行比较。我们报告了四个主要发现。首先，LLM的预测相当……

    arXiv:2610.22599v1 Announce Type: cross  Abstract: Recent advances in AI make Recursive Self-Improvement (RSI) increasingly plausible. Large Language Models (LLMs) can already propose, select, and run Machine Learning (ML) experiments. However, the space of possible design choices in ML is vast. An LLM that proposes an experiment needs a reliable intuition of whether it is worth running given the timing constraints of long training runs. In this work, we measure such intuition in current frontier LLMs by evaluating how accurately they predict the impact of self-generated ML experiment ideas before these are put into training runs. To this end, we construct a dataset of 288 LLM-proposed ideas on eight MLE-bench tasks. We then prompt five frontier LLMs (GPT-6 Astra, Claude Opus 5.5, Claude Fable 5.1, Kimi K3, and GLM-5.3) to predict the impact of each idea and compare their predictions with the actual outcomes after training. We report four main findings. First, LLM predictions are subst
    
[^122]: 基于统一标签的广泛且高效的个人身份信息检测

    Broad and Efficient PII Detection with Unified Labels

    [https://arxiv.org/abs/2610.22597](https://arxiv.org/abs/2610.22597)

    该论文通过对77个公开数据集在统一标签体系下重新标注，训练出一个141M参数的高效编码器检测器，实现了对12类结构化PII在低误报率下的广泛且高效的检测。

    

    个人身份信息（PII）检测模型必须在多样化上下文中高效识别标识符，同时在大流量环境下保持低误报率（FPR）。近期的检测器越来越依赖会推高部署成本的复杂架构，然而许多模型仅在少数合成数据集上进行微调，限制了其覆盖范围。合并公开数据集颇具挑战性，因为各数据集的分类体系不同，即使将类别映射到统一模式后，标签在什么构成一次检测上仍可能存在分歧，从而产生不一致的训练与评估信号。为解决这些问题，我们提出了一种实用的141M参数基于编码器的检测器，该检测器在来自77个公开数据集的112万个样本上训练，样本根据针对12类结构化PII类别的广义上下文检测的详细定义进行了重新标注。在AI4Privacy和Nemotron-PII基准上于低误报率工作点进行评估，我们的模型取得了……

    arXiv:2610.22597v1 Announce Type: new  Abstract: Personally Identifiable Information (PII) detection models must efficiently find identifiers across diverse contexts while maintaining a low false-positive rate (FPR) on high-volume traffic. Recent detectors increasingly rely on complex architectures that raise deployment cost, yet many are fine-tuned on only a few synthetic datasets that limit their coverage. Combining public datasets is challenging because their taxonomies differ, and even after categories are mapped to a common schema, labels can disagree on what counts as a detection, producing inconsistent training and evaluation signals. To address these problems, we present a practical 141M-parameter encoder-based detector trained on 1.12 million examples from 77 public datasets, relabeled under a detailed definition of broad contextual detection for 12 structured PII categories. Evaluated on the AI4Privacy and Nemotron-PII benchmarks at a low-FPR operating point, our model achiev
    
[^123]: 模型更了解情况：防御医疗多智能体系统免受提示注入攻击

    The Model Knows Better: Defending Medical Multi-Agent Systems Against Prompt Injection

    [https://arxiv.org/abs/2610.22594](https://arxiv.org/abs/2610.22594)

    该论文提出一种无需训练的防御方法，通过将患者检查发现与临床判断分离，并验证系统决策背后的临床判断是否与患者实际情况相符，从而有效防御医疗多智能体系统中的提示注入攻击。

    

    医疗多智能体系统——即由多个大语言模型智能体共同讨论病例的系统——在临床决策支持领域的应用研究日益增多。提示注入是指攻击者添加到模型输入中的文本，它可以操纵大语言模型做出错误决策。提示注入如何影响这类系统以及如何保护这些系统，目前仍不清楚。在两个系统、两个模型系列和三个基准测试上的实验表明，单段注入文本最多可在61%的案例中成功操纵系统，且智能体之间的协作并不能持续提供有效防护。现有防御手段只能起到部分作用：它们会漏掉那些看起来像普通临床文本的注入内容。然而，当被要求将注入文本与患者的实际检查发现进行核对时，之前因遵循注入内容而做出决策的模型通常会拒绝该注入文本。因此，我们提出了一种无需训练的防御方法，用于验证系统决策背后的临床判断。该方法遵循临床实践，将患者的检查发现与临床判断（如诊断、建议和指示等）分离开来……

    arXiv:2610.22594v1 Announce Type: new  Abstract: Medical multi-agent systems, in which LLM agents discuss a case, are increasingly studied for clinical decision support. Prompt injection, text that an attacker adds to a model's input, can manipulate LLMs into wrong decisions. How it affects these systems, and how to protect them, remains unclear. Across two systems, two model families and three benchmarks, a single injected passage succeeds in up to 61% of cases, and collaboration does not consistently protect against it. Existing defenses help only partly: they miss injections that look like ordinary clinical text. Yet when asked to check the injected text against the patient's findings, a model that followed the injection usually rejects it. We therefore propose a training-free defense that verifies the clinical judgments behind the system's decision. Following clinical practice, it separates the patient's findings from clinical judgments, such as diagnoses, recommendations and instr
    
[^124]: SAG2：基于BM25与JEV的无嵌入多跳检索

    SAG2: Embedding-Free Multi-Hop Retrieval with BM25 and JEV

    [https://arxiv.org/abs/2610.22588](https://arxiv.org/abs/2610.22588)

    SAG2提出了一种无需嵌入向量的多跳检索架构，仅依靠单个SQLite数据库、BM25词法索引和JEV上下文证据决策，在查询时完成语义证据选择，同时实现了轻量级部署准备和多个多跳问答基准上最高的检索召回率。

    

    强大的多跳检索系统应当在实际部署中无需昂贵的语料库准备或复杂的数据库基础设施。我们提出了SAG2，一种无嵌入的检索架构，它围绕单个SQLite数据库、BM25以及来自JEV的上下文证据决策构建。SAG2遵循SAG（SQL-Retrieval Augmented Generation，SQL检索增强生成）的查询时关联原则，同时用针对原始文本段的词法索引取代了其事件-实体索引和稠密向量索引。被选中的文本段为后续检索提供查询词，而新检索到的证据则指导后续决策。这将语义证据选择转移到了查询时进行，从而省去了离线嵌入计算、基于大语言模型的事件-实体抽取以及专用数据库服务器。由此得到的系统兼具轻量级的准备工作与强大的检索准确率：在MuSiQue、2WikiMultiHopQA和HotpotQA各1000个问题的子集上，SAG2取得了最高的Recall@5和Recall@10……

    arXiv:2610.22588v1 Announce Type: new  Abstract: Strong multi-hop retrieval should be practical to deploy without expensive corpus preparation or complex database infrastructure. We present SAG2, an embedding-free retrieval architecture built around a single SQLite database, BM25, and contextual evidence decisions from JEV. SAG2 follows the query-time association principle of SAG (SQL-Retrieval Augmented Generation), while replacing its event-entity and dense indexes with a lexical index over original passages. Selected passages supply terms for further retrieval, and newly retrieved evidence guides subsequent decisions. This moves semantic evidence selection to query time, eliminating offline embedding computation, LLM-based event-entity extraction, and dedicated database servers. The resulting system combines lightweight preparation with strong retrieval accuracy: on 1,000-question subsets of MuSiQue, 2WikiMultiHopQA, and HotpotQA, SAG2 achieves the highest Recall@5 and Recall@10 amo
    
[^125]: 通过理由正则化缓解解释与人类对齐中的人口群体差异

    Mitigating Demographic Disparities in Explanation-Human Alignment via Rationale Regularization

    [https://arxiv.org/abs/2610.22586](https://arxiv.org/abs/2610.22586)

    本文提出Plausibility-EFG指标来衡量仇恨言论检测模型在不同人口群体间解释合理性的差异，并证明采用理由正则化的对抗训练模型ADV-RoBERTa能在提升分类性能和解释与人类对齐的同时减少群体间的解释差异。

    

    我们研究仇恨言论检测中的解释公平性。高分类性能并不总是意味着模型能够为不同人口群体产生同样合理的解释。使用HateXplain数据集，我们引入了Plausibility-EFG这一指标，通过比较模型生成的词元归因与人工标注的理由，来衡量不同人口群体之间解释合理性的最大差异。我们使用集成梯度和SHAP方法评估了BERT、RoBERTa以及对抗训练模型ADV-RoBERTa。结果表明，ADV-RoBERTa在提高分类性能和解释与人类理由之间对齐程度的同时，减少了不同人口群体之间的解释差异。这些改进在两种归因方法下均可观察到，并且在训练集、验证集和测试集上保持一致。我们还表明，仅减少预测层面的偏差并不能消除解释层面的差异。

    arXiv:2610.22586v1 Announce Type: cross  Abstract: We study explanation fairness in hate speech detection. High classification performance does not always mean that a model produces equally plausible explanations for different demographic groups. Using the HateXplain dataset, we introduce Plausibility-EFG, a metric that measures the largest difference in explanation plausibility across demographic groups by comparing model-generated token attributions with human-annotated rationales. We evaluate BERT, RoBERTa, and ADV-RoBERTa, an adversarially trained model, using Integrated Gradients and SHAP. Our results show that ADV-RoBERTa improves classification performance and alignment between explanations and human rationales while reducing explanation disparities across demographic groups. These improvements are observed with both attribution methods and remain consistent across training, validation, and test splits. We also show that reducing prediction-level bias alone does not eliminate di
    
[^126]: 基于确定性诊断的大语言模型生成规约修复

    Repairing LLM-Generated Specifications with Deterministic Diagnosis

    [https://arxiv.org/abs/2610.22568](https://arxiv.org/abs/2610.22568)

    提出DRAGO修复框架，用确定性的、针对特定形式化方法的诊断取代LLM无根据的批评，且仅在诊断错误权重严格减少时才保留重写，从而有效修复LLM生成的可运行但语义错误的形式化规约。

    

    大型语言模型（LLM）正越来越多地被用作形式化工具：给定一个以自然语言表达的推理问题，它们生成一个形式化规约，符号求解器随后可以基于该规约进行求解，并获得相对于该规约的保证。许多规约虽然可以运行，但在语义上是错误的——它们编码了错误的约束、定义域、目标、谓词或最终目标，而且一旦执行成功，求解器的错误反馈就不再提供信息。纯粹基于LLM的诊断由于缺乏形式化的真值标准，往往无法发现这些错误，并且在我们的实验中，其无根据的批评导致11.8%最初正确的规约出现退化。我们提出了DRAGO，一个修复框架，它用确定性的、针对特定形式化方法的诊断取代了这种无根据的批评，并且只有当诊断出的错误权重在无参考接受谓词下严格减少时，才保留重写结果。在涵盖逻辑推理（SMT-LIB...

    arXiv:2610.22568v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used as formalizers: given a reasoning problem expressed in natural language, they produce a formal specification that a symbolic solver can then solve, with guarantees relative to that specification. Many specifications are runnable yet semantically wrong, encoding incorrect constraints, domains, objectives, predicates, or goals, and solver-error feedback dries up once execution succeeds. Purely LLM-based diagnosis, lacking a formal ground truth, often fails to spot these errors, and its ungrounded critiques cause regressions in 11.8% of initially correct specifications in our experiments. We propose DRAGO, a repair framework that replaces this ungrounded critique with deterministic, formalism-specific diagnosis, and retains a rewrite only when the diagnosed error weight strictly decreases under a reference-free acceptance predicate. Across five benchmarks spanning logical reasoning (SMT-LIB
    
[^127]: 视角主义模型对谁预测得好？标注者一致性、稀疏性与性能

    Who Do Perspectivist Models Predict Well? Annotator Agreement, Sparsity, and Performance

    [https://arxiv.org/abs/2610.22567](https://arxiv.org/abs/2610.22567)

    该研究通过构建迄今最大的序数标注视角主义NLP数据集集合，发现算法对倾向于与他人一致的标注者预测效果更好，而标注者标注量的多少与算法性能并无显著正相关。

    

    成功完成视角主义自然语言处理（NLP）标注任务，需要算法在不同视角之间保持对齐，以预测不同人类将如何对文本进行评分。这项工作研究了此类算法的性能在那些行为与他人不同的标注者上如何变化——这些标注者要么提供与他人回答相分歧的标注，要么提供的标注总量较少。为了帮助刻画模型性能，我们整理了迄今为止规模最大的带有序数标注的视角主义NLP数据集集合。在各个数据集上评估了多类视角主义算法之后，我们对个体标注者的独特性与算法性能之间的关系进行了建模。总体而言，我们发现算法在倾向于与他人保持一致的标注者上表现更好，但标注者的响应率与算法性能之间并不存在统计学上显著的正相关。

    arXiv:2610.22567v1 Announce Type: new  Abstract: Successful completion of perspectivist natural language processing (NLP) annotation tasks requires algorithms to be aligned across diverse perspectives in order to predict how different humans will rate text. This work studies how the performance of such algorithms varies for annotators who behave differently than others, either by providing annotations that diverge from others' responses or by providing fewer annotations altogether. To aid with characterizing model performance, we curate the largest collection to date of perspectivist NLP datasets with ordinal annotations. After evaluating multiple classes of perspectivist algorithms across each of the datasets, we model the relationship between individual annotator idiosyncrasy and algorithm performance. In general, we find that algorithms perform better on annotators who tend to agree with others, but that annotator response rate does not have a statistically significant positive corr
    
[^128]: 文化（不）胜任力：探究大语言模型对文化特定刻板印象的检测能力

    Cultural (In)Competence: Investigating Culture-Specific Stereotype Detection Capabilities of LLMs

    [https://arxiv.org/abs/2610.22545](https://arxiv.org/abs/2610.22545)

    本文系统研究了提示语言、指令微调语言和微调数据集规模这三个因素对大语言模型在美国、荷兰和中国等不同文化语境下检测文化特定刻板印象能力的影响，揭示了模型文化胜任力方面的差异。

    

    刻板印象是由其所处的文化语境所调节的主观建构，因此，要成功地对其进行建模，同样需要文化胜任力。当使用大语言模型（LLM）执行与刻板印象相关的任务（如刻板印象检测）时，模型能够获取相关文化知识并据此调整响应至关重要。由于大语言模型在可获取的文化知识方面存在差异，它们在不同语言和文化语境下（例如非英语语言或非美国语境）处理刻板印象的能力也可能存在差异。本文系统性地研究了三个因素对大语言模型进行文化特定刻板印象检测效果的影响：（1）提示语言；（2）指令微调语言；（3）微调数据集规模。我们针对美国、荷兰和中国文化语境开展了实验。我们的结果表明，大语言模型……（摘要在此处截断）

    arXiv:2610.22545v1 Announce Type: new  Abstract: Stereotypes are subjective constructs that are mediated by the cultural context in which they are viewed. Successfully modeling them thus also requires cultural competence. When using large language models (LLMs) for stereotype-related tasks, such as their detection, it is therefore crucial that models can access relevant cultural knowledge and adjust responses accordingly. Since LLMs have shown disparities in what cultural knowledge is available to them, their capacity regarding stereotypes may also be disparate across languages and cultural contexts (e.g., for non-English languages or for non-US contexts). In this paper, we systematically investigate the effectiveness of LLMs for culture-specific stereotype detection across three factors: (1) the prompting language, (2) the instruction tuning language, and (3) the fine-tuning dataset size. We conduct experiments for US, Dutch, and Chinese cultural contexts. Our results suggest that LLM
    
[^129]: HeRE：通过大语言模型引导的知识图谱对齐去噪构建的希伯来语关系抽取新基准

    HeRE: A Novel Benchmark for Hebrew Relation Extraction via LLM-Guided Denoising of Knowledge Graph Alignments

    [https://arxiv.org/abs/2610.22542](https://arxiv.org/abs/2610.22542)

    本文提出HeRE——首个基于LLM引导去噪知识图谱对齐构建的希伯来语关系抽取基准，提供了一种可扩展、语言无关的低资源语言关系抽取数据集创建方法。

    

    尽管大语言模型在英语关系抽取任务中展现出强大的能力，但由于缺乏标准化基准，评估其在低资源语言中的关系抽取性能仍存在显著差距。创建原生的、针对特定语言标注的关系抽取基准成本高昂且需要人工专业知识，而依赖翻译则会产生次优、不准确的数据集，并引入以英语为中心的偏差。为解决这一问题，我们提出了一种可扩展的、语言无关的方法论，通过对知识库与相应文本之间自然存在但含噪的对齐进行去噪来创建关系抽取数据。我们通过自然语言推理（NLI）任务定义了恰当的对齐，即一个关系三元组需被输入文本所蕴含。我们对五种去噪策略进行了实证评估，并证明在少样本LLM设置下，多语言模型优于特定语言模型，而希伯来语专用编码器在NLI设置下仍具有竞争力。

    arXiv:2610.22542v1 Announce Type: new  Abstract: While LLMs demonstrate strong capabilities in English Relation Extraction (RE) tasks, a significant gap remains in assessing their RE performance in low-resource languages, due to a lack of standardized benchmarks. Creating native, language-specific annotated RE benchmarks is expensive and requires human expertise, while relying on translation yields suboptimal, inaccurate datasets that introduce English-centric biases. To tackle this, we propose a scalable, language-agnostic methodology to create RE data by denoising naturally occurring yet noisy alignments of knowledge bases and corresponding texts. We define a proper alignment via a Natural Language Inference (NLI) task, where a relation triplet is entailed by an input text. We empirically assess five denoising strategies, and demonstrate that multilingual models outperform language-specific models in the few-shot LLM setting, while Hebrew-specific encoders remain competitive under NL
    
[^130]: 大型推理模型是优秀的日常推理者吗？与非推理模型在逻辑推理任务上的对比

    Are Large Reasoning Models Good Everyday Reasoners? A Comparison with Non-Reasoning Models on Logical Reasoning Tasks

    [https://arxiv.org/abs/2610.22538](https://arxiv.org/abs/2610.22538)

    该研究构建了一个基于语言学与语用推理现象的日常逻辑推理基准，发现大型推理模型在偏离形式逻辑规则的日常推理中表现退化，而某些采用思维链提示的非推理模型在日常推理上甚至优于推理模型。

    

    利用测试时计算的大型推理模型（LRMs）在数学和科学基准测试中取得了出色的表现，并越来越多地被部署于实际应用中。然而，它们进行日常逻辑推理的能力——这类推理往往偏离教科书式的形式逻辑——仍然未被充分探索。在本工作中，我们通过引入一个基于语言学和语用推理现象的基准，系统地评估了推理模型与日常推理模式的契合程度。我们还在受控的提示条件下对推理模型与非推理模型进行了比较。我们发现，在许多情况下：(1) 当日常推理偏离形式逻辑规则时，推理模型在日常推理中的表现会出现下降；(2) 一些经过思维链提示的非推理模型在日常推理中的表现优于同类的推理模型；(3) 通过提供额外指令的提示……

    arXiv:2610.22538v1 Announce Type: new  Abstract: Large reasoning models (LRMs) that leverage test-time computation achieve strong performance on mathematical and scientific benchmarks and are increasingly deployed in real-world applications. However, their ability to perform \textit{everyday logical reasoning}, which often departs from textbook formal logic, remains underexplored. In this work, we systematically evaluate how well reasoning models align with everyday reasoning patterns by introducing a benchmark based on linguistic and pragmatic reasoning phenomena. We also compare reasoning models with non-reasoning models under controlled prompting conditions. We find that, in many cases, (1) reasoning models show degraded performance in everyday reasoning when such reasoning deviates from formal logical rules, (2) some non-reasoning models with chain-of-thought prompting outperform comparable reasoning models in everyday reasoning, and (3) prompting with additional instructions provi
    
[^131]: Pearl-Chat：基于 JAX 和 Flax NNX 从第一性原理构建的卢干达语原生语言模型

    Pearl-Chat: Engineering a native Luganda language model from first principles in JAX and Flax NNX

    [https://arxiv.org/abs/2610.22529](https://arxiv.org/abs/2610.22529)

    该论文提出了Pearl-Chat——一个用JAX和Flax NNX从第一性原理构建的约850万参数卢干达语原生因果语言模型，通过从头训练的字节级BPE分词器解决了通用多语言分词器在班图黏着语言上子词切分率过高的问题。

    

    自回归基础语言模型主要面向高资源的印欧语言，这对代表性不足的非洲语言造成了计算和形态学上的不利影响。通用的多语言分词器在黏着语类型的班图语言上，每词的子词切分率通常会超过3.2个词元，从而降低了上下文利用效率并增加了推理延迟。我们提出了Pearl-Chat，一个拥有8,473,122个参数（约850万）的因果解码器式Transformer，专为卢干达语（Oluganda，ISO 639-2: lg）设计——这是一种在东非被超过1000万人使用的低资源尼日尔-刚果语系班图语言。Pearl-Chat基于JAX和Flax NNX从第一数学原理构建，采用显式的函数式状态分解（nnx.split和nnx.merge）、通过Google Grain实现的确定性可恢复数据流式加载，以及使用jax.sharding.Mesh的SPMD张量并行。我们引入了一个在精选语料库上从头训练的字节级字节对编码（BPE）子词分词器……

    arXiv:2610.22529v1 Announce Type: new  Abstract: Autoregressive foundation language models predominantly target high-resource Indo-European languages, imposing computational and morphological penalties on under-represented African languages. Generic multilingual tokenizers routinely exceed fertility rates of 3.2 tokens per word on agglutinative Bantu languages, degrading context efficiency and inflating latency. We present Pearl-Chat, an 8,473,122-parameter ($\sim$8.5M) causal decoder-only transformer engineered for Luganda (Oluganda, ISO 639-2: lg), a low-resource Niger-Congo Bantu language spoken by over 10 million people in East Africa. Built from first mathematical principles in JAX and Flax NNX, Pearl-Chat uses explicit functional state decomposition (nnx.split and nnx.merge), deterministic resumable data streaming via Google Grain, and SPMD tensor parallelism using jax.sharding.Mesh. We introduce a byte-level byte-pair encoding (BPE) subword tokenizer trained from scratch on cura
    
[^132]: 字面锁定：测试语言如何改变类型化决策模型对孟加拉语习语的字面吸引

    Literal Lock-In: How Test Language Shifts Literal Attraction in Typed Decision Models on Bengali Idioms

    [https://arxiv.org/abs/2610.22526](https://arxiv.org/abs/2610.22526)

    该论文发现，将字面义释义混入选项列表会强烈吸引各类非生成式决策模型（占其剩余孟加拉语错误的65%至96%），且这种“字面锁定”效应在孟加拉语中比英语强1.9至6.4倍，表明此类失败主要取决于测试语言而非模型能力。

    

    决策模型，即对调用方提供的选项输出概率且不生成任何文本的非生成式评分器，正日益被部署为路由器、分类器和安全护栏。由于它们不生成任何内容，推理依据和思维链等方法并不适用。我们通过对三个模型（一个编码器评分器、一个解码器评分器和一个闭源API）的输入进行干预来审计它们，使用的是专家标注的孟加拉语习语，每个习语都带有黄金比喻义释义和黄金字面义释义。将习语自身的字面义释义放入选项列表中，会把每个选项的输出都拉向它，占每个选项剩余孟加拉语错误的65%至96%。在共享的同一项目集上构建的模型-语言矩阵揭示了这种现象在多大程度上取决于测试语言：在相同的配对条目上，每个模型在孟加拉语中被字面义捕获的程度是英语中的1.9至6.4倍，商业模型表面上的抵抗性其实是其测试语言的属性，而模型能力并不能预测这种失败的排序。校准性能退化（摘要在此处被截断）。

    arXiv:2610.22526v1 Announce Type: new  Abstract: Decision models, non-generative scorers that report a probability over caller-supplied options and emit no text, are increasingly deployed as routers, classifiers and guardrails. Because they generate nothing, rationales and chain-of-thought do not apply. We audit three: an encoder scorer, a decoder scorer and a closed API, entirely through interventions on their input, using expert-annotated Bengali idioms carrying a gold figurative and a gold literal gloss. Dropping an idiom's own literal gloss into the option list draws every arm toward it, taking 65% to 96% of each arm's remaining Bengali errors. A model-by-language matrix over one shared item set shows how far this depends on the test language: every arm is 1.9 to 6.4 times more captured in Bengali than in English on the same paired entries, the commercial model's apparent resistance is a property of its test language, and capability does not order the failure. Calibration degrades 
    
[^133]: alkhder 团队参加 AlexandriaX-2026 的系统：基于 QLoRA 微调 NileChat-3B 的上下文感知英语到阿拉伯方言对话翻译

    alkhder at AlexandriaX-2026: Context-Aware English-to-Dialectal Arabic Dialogue Translation with QLoRA-Fine-Tuned NileChat-3B

    [https://arxiv.org/abs/2610.22525](https://arxiv.org/abs/2610.22525)

    Alkhder 团队通过 QLoRA 微调 NileChat-3B-Base，以国家、方言、领域、参与者角色、性别方向及对话上下文等多重条件为输入，仅用一组适配器权重实现了英译十三种阿拉伯方言的上下文感知对话翻译，并以极低的训练成本在 AlexandriaX-2026 上取得 spBLEU 20.36 的成绩。

    

    本文介绍了 Alkhder 团队参加 AlexandriaX-2026 共享任务子任务 1 的提交系统，该任务要求系统将英文对话轮次翻译成十种阿拉伯语方言（共十三种方言）。我们在 Alexandria 平行对话语料库上使用 QLoRA 对 NileChat-3B-Base 进行微调，仅训练一组适配器权重即可覆盖所有方言。每轮对话的翻译均以目标国家和方言、领域、参与者角色、说话者与受话者的性别方向以及前文对话上下文为条件。在推理阶段，对话以自回归方式逐轮翻译，模型将自身先前生成的输出作为上下文反馈输入，不使用任何测试集目标侧的黄金参考信息。在受限赛道下，系统仅使用 Alexandria 数据，训练耗时约一小时，翻译私有测试集耗时不足两小时。该系统在两条赛道上均取得了国家宏平均 spBLEU 20.36、chrF++ 36.64 的成绩。

    arXiv:2610.22525v1 Announce Type: new  Abstract: This paper describes team Alkhder's submission to Subtask 1 of the AlexandriaX-2026 shared task, which asks systems to translate English dialogue turns into thirteen Arabic dialects. We fine-tune NileChat-3B-Base with QLoRA on the Alexandria parallel dialogue corpus, training one set of adapter weights for all dialects. Each turn is translated conditioned on the target country and dialect, the domain, the participant roles, the speaker-addressee gender direction, and the preceding dialogue context. At inference, conversations are translated autoregressively and the model is fed its own previous outputs back as context without using any gold target-side test information. Under the constrained track, using only Alexandria data, it trains in about one hour and translates the private test set in under two hours. The system reaches a country-macro-averaged spBLEU of 20.36 and chrF++ of 36.64 on both tracks.
    
[^134]: 学习用户表示以提升模拟保真度

    Learning User Representations to Improve Simulation Fidelity

    [https://arxiv.org/abs/2610.22518](https://arxiv.org/abs/2610.22518)

    该论文将用户表示识别为用户模拟的关键瓶颈，提出通过对话重建任务与事后反馈来评估和优化自然语言形式的用户表示，从而提升用户模拟器复现真实用户行为的保真度。

    

    用户模拟器能够大规模地对AI助手进行交互式训练和评估。然而，当前的模拟器难以再现真实用户的多样化行为，导致助手对代表性不足的用户泛化能力较差。我们将用户表示确定为用户模拟中一个未被充分探索的瓶颈。用户表示是一种自然语言描述，用于作为模拟器的条件，使其再现特定用户的行为。表示的质量只有在事后通过其在模拟器中引发的行为才能显现。为了评估和优化用户表示，我们提出了一个对话重建任务：给定一段原始对话，先生成一个用户表示，将其作为模拟器的条件，模拟一段对话，然后通过比较原始对话与模拟对话中的用户行为来衡量保真度。这种比较还会产生事后反馈，用于识别表示中存在的不足。

    arXiv:2610.22518v1 Announce Type: new  Abstract: User simulators enable interactive training and evaluation of AI assistants at scale. Yet current simulators struggle to reproduce the diverse behaviors of real users, leading to assistants that generalize poorly to underrepresented users. We identify user representations as an underexplored bottleneck in user simulation. User representations are natural-language descriptions that condition simulators to reproduce a particular user's behavior. A representation's quality becomes apparent in hindsight, through the behavior it induces in a simulator. To evaluate and optimize user representations, we formulate a conversation reconstruction task: given an original conversation, we generate a representation, condition a simulator on it, simulate a conversation, and then measure fidelity by comparing user behavior in the original and simulated conversations. This comparison also yields hindsight feedback, which identifies where the representati
    
[^135]: alkhder参与ImageEval 2026共享任务：面向多模态幻觉检测的约束选择与轮转平均Logit评分

    alkhder at ImageEval 2026 Shared Tasks: Constrained Selection and Rotation-Averaged Logit Scoring for Multimodal Hallucination Detection

    [https://arxiv.org/abs/2610.22515](https://arxiv.org/abs/2610.22515)

    该方法在零样本设置下利用7B视觉-语言模型，通过约束选择、下一词元对数概率评分和拉丁方轮转消除答案位置偏差，在ImageEval 2026多模态幻觉检测任务的英语和阿拉伯语测试集上均将CI错误率相比官方3B基线降低超过80%。

    

    我们描述了参加ImageEval 2026子任务1b的提交方案。该任务要求从与图像配对的三个候选陈述中选出唯一在视觉上有效的陈述，任务同时涵盖英语和现代标准阿拉伯语（MSA）。我们的系统在零样本设置下使用预训练的70亿参数视觉-语言模型，无需任何针对任务的训练或微调。我们将该任务形式化为一个约束选择问题，并利用下一词元的对数概率来选择答案。为减少答案位置偏差，我们对候选陈述应用拉丁方轮转。我们的系统在英语测试集上取得0.0510的CI分数，在MSA测试集上取得0.0960的CI分数，相对于开发测试集上报告的官方3B基线，分别将CI降低了83.7%和80.4%。

    arXiv:2610.22515v1 Announce Type: new  Abstract: We describe our submission to Subtask~1b of ImageEval 2026. The task consists of selecting the single visually valid statement from three candidates paired with an image, and is in both English and Modern Standard Arabic (MSA). Our system uses a pre-trained 7B vision--language model in a zero-shot setting, without task-specific training or fine-tuning. We formulate the task as a constrained selection problem and choose the answer using next-token log-probabilities. To reduce answer-position bias, we apply a Latin-square rotation to the candidate statements. Our system achieves CI scores of 0.0510 on the English test set and 0.0960 on the MSA test set, reducing CI relative to the official 3B baselines reported on devtest by 83.7\% and 80.4\%, respectively.
    
[^136]: AuditKIT：一个用于大语言模型与智能体系统基准测试、评判与比较的统一评估主干

    AuditKIT: A Unified Evaluation Spine for Benchmarking, Judging, and Comparing Large Language Models and Agentic Systems

    [https://arxiv.org/abs/2610.22511](https://arxiv.org/abs/2610.22511)

    AuditKIT是一个统一的评估框架，将基准测试、评判评估、运行时性能、工具使用和检索质量整合到单一结果表示中，解决了大语言模型与智能体系统评估中接口和指标语义不一致的问题。

    

    评估语言模型和智能体需要综合基准测试准确率、基于评判模型的评估、运行时性能、工具使用行为以及检索质量等多方面指标。接口、指标语义和轨迹格式的差异使这些比较变得复杂，并可能掩盖智能体工作流中的失败。我们提出了AuditKIT，一个源码可用的评估库，它将这些测量统一记录在单一的结果表示中，具备方向感知的指标契约、配置指纹以及可扩展的指标和后端。针对智能体，它提供回合感知的工具调用匹配，区分并行与顺序执行，并将经过验证的任务结果与轨迹级诊断区分开来。缺失或不可观察的证据会被标记为“未评分”，而不是被计为失败。针对检索增强生成（RAG），它结合了排序检索指标、评判式的依据性评估以及确定性的证据检查。七个案例研究涵盖模型（比较）……

    arXiv:2610.22511v1 Announce Type: cross  Abstract: Evaluating language models and agents requires combining benchmark accuracy, judge-based assessment, runtime performance, tool-use behavior, and retrieval quality. Differences in interfaces, metric semantics, and trace formats complicate these comparisons and can hide failures in agentic workflows. We present AuditKIT, a source-available evaluation library that records these measurements in one result representation, with direction-aware metric contracts, configuration fingerprints, and extensible metrics and backends. For agents, it provides turn-aware tool-call matching, separates parallel from sequential execution, and distinguishes verified task outcomes from trajectory-level diagnostics. Missing or unobservable evidence is marked unscored rather than counted as failure. For retrieval-augmented generation, it combines ranked retrieval metrics, judged grounding, and deterministic evidence checks. Seven case studies spanning model co
    
[^137]: RealCLI：迈向CLI智能体人机协作的统一评估

    RealCLI: Toward a Unified Evaluation of Human-Agent Collaboration for CLI Agents

    [https://arxiv.org/abs/2610.22507](https://arxiv.org/abs/2610.22507)

    REALCLI是一个统一评估CLI智能体任务执行与人机协作质量的配对基准，其通过包含22个子维度和51个交互场景模板的五维框架，以及完整规格说明（REALCLI-EXEC）与脚本化用户渐进交互（REALCLI-COLLAB）的双模式设计，揭示了前沿智能体在协作能力上的显著差距。

    

    命令行界面（CLI）智能体越来越多地执行长时程任务，然而现有的基准测试要么假设用户指令在整个执行过程中保持固定不变，要么在评估任务结果时未能捕捉运行时的协作质量。在本文中，我们提出了REALCLI，这是一个配对基准，用于在真实的CLI任务上统一评估任务执行和人机协作质量。为了全面衡量协作水平，REALCLI构建了一个五维评估框架，并将其具体化为22个子维度和51个交互场景模板。REALCLI提供两种模式用于配对比较：REALCLI-EXEC预先提供完整的规格说明，而REALCLI-COLLAB则引入一个脚本化的用户，该用户会逐步提供信息、更正、约束修订和授权决策。评估结果显示前沿智能体存在巨大的能力差距：最佳配置的平均得分仅为……（原文摘要在此处截断）

    arXiv:2610.22507v1 Announce Type: new  Abstract: Command-line interface (CLI) agents increasingly execute longer-horizon tasks, yet existing benchmarks either assume that user instructions remain fixed throughout execution or evaluate task outcomes without capturing runtime collaboration quality. In this paper, we present REALCLI, a paired benchmark that unifies the evaluation of task execution and human-agent collaboration quality on realistic CLI tasks. To comprehensively measure collaboration, REALCLI builds a five-dimensional framework, operationalized into 22 sub-dimensions and 51 interaction scenario templates. REALCLI provides two modes for paired comparison: REALCLI-EXEC supplies complete specifications upfront, whereas REALCLI-COLLAB introduces a scripted user that delivers progressive information, corrections, constraint revisions, and authorization decisions. Evaluations reveal profound capability gaps in frontier agents: the best configuration attains an average score of on
    
[^138]: 从平行到分化：语言模型对动作名词构式的学习

    From Parallels to Divergence: Language Models' Learning of Action Nominal Constructions

    [https://arxiv.org/abs/2610.22468](https://arxiv.org/abs/2610.22468)

    本研究通过构建96种具有不同语法配置的合成语言并从零训练GPT-2模型，发现语言模型的学习偏好与自然语言类型学分布在基本从句配置上平行，但在补语系统和动作名词构式策略上出现分化。

    

    动作名词构式（ANCs）以名词形式表达事件，其论元标记遵循从句或普通名词短语中的模式。在本文中，我们研究了不同的语法配置如何影响语言模型对动作名词构式及相关构式的学习，并将这些学习偏好与自然语言中的类型学分布进行比较。我们采用一种语法工程方法来构建合成语料库，该方法将来自自然语料库的共享语义输入与计算语法相结合。我们构建了96种目标语言，这些语言在从句语序、名词短语语序、配列方式、补语系统和动作名词构式策略上各不相同。对于每种语言，我们从零开始训练GPT-2-small模型，并使用涵盖34种语法现象的针对性最小对立对进行评估。结果表明，模型在基本从句配置上呈现出与自然语言类型学的平行性，但在补语系统和动作名词构式策略上则出现分化。

    arXiv:2610.22468v1 Announce Type: new  Abstract: Action nominal constructions (ANCs) express events in nominal form, with their argument marking following patterns found in clauses or ordinary noun phrases. In this paper, we investigate how different grammatical configurations affect language models' learning of ANCs and related constructions, and how these learning preferences compare with typological distributions in natural languages. We use a grammar-engineered method for synthetic corpus construction that combines shared semantic input from natural corpora with computational grammars. We construct 96 target languages varying in clause word order, noun-phrase word order, alignment, complement system, and ANC strategy. For each language, we train GPT-2-small models from scratch and evaluate them using targeted minimal pairs covering 34 grammatical phenomena. The results show typological parallels in basic clause configurations but divergences in complement systems and ANC strategies
    
[^139]: 良性递归自蒸馏可以引发渐进式失准

    Progressive Misalignment Can Emerge from Benign Recursive Self-Distillation

    [https://arxiv.org/abs/2610.22467](https://arxiv.org/abs/2610.22467)

    在递归自蒸馏过程中，即使只使用良性提示和模型自身的安全生成内容，安全模型也会随着迭代轮次不断累积失准，最终在多种有害查询上表现出渐进式失准现象。

    

    arXiv:2610.22467v1 公告类型：cross 摘要：在模型自身输出上训练模型作为提升AI能力的可能途径，已受到广泛关注。一种突出的方法是自蒸馏，即训练模型去匹配它在获得额外上下文（例如提示、反馈或指令）时本应产生的token级预测。然而，这种训练过程的安全性尚未被充分理解。在本文中，我们揭示了递归自蒸馏（即每个自蒸馏出的模型成为下一轮训练的教师模型）中一个令人惊讶的安全失效现象。我们观察到，当一个安全模型在自身（安全的）生成结果上进行递归自蒸馏，同时接收额外的非有害提示时，它在多种多样的有害查询上会变得越来越失准；我们将这一现象称为“渐进式失准”。我们在Qwen3和Phi-4模型家族上，通过SORRY-Bench和StrongREJECT基准展示了这种失效模式……

    arXiv:2610.22467v1 Announce Type: cross  Abstract: Training models on their own outputs has attracted widespread attention as a possible route to advance AI capabilities. One prominent approach is self-distillation, where a model is trained to match the token-level predictions it would have produced when given access to additional context, such as a cue, feedback, or instruction. However, the safety properties of this training procedure are poorly understood. In this paper, we present a surprising safety failure of recursive self-distillation, where each self-distilled model becomes the teacher for the next round of training. We observe that when a safe model is recursively self-distilled on its own (safe) generations while receiving an additional non-harmful cue, it becomes increasingly misaligned across a diverse set of harmful queries; we call this Progressive Misalignment. We show this failure mode in both the Qwen3 and Phi-4 model families on SORRY-Bench and StrongREJECT. Progress
    
[^140]: Twitter/X 热门话题中对抗性言论的高流行度

    High prevalence of antagonizing rhetoric in trending topics on Twitter/X

    [https://arxiv.org/abs/2610.22455](https://arxiv.org/abs/2610.22455)

    本研究提出了一种能够区分并检测文本中“敌对化”与“批评”两种政治言论及其目标的计算方法，并揭示了Twitter/X热门话题中敌对化言论的高度盛行。

    

    政治对手之间的分歧是民主的核心要素，而这种分歧可以以不同方式表达：可以将对手敌对化，即将其描绘为需要被战胜的敌人；也可以对其进行批评，即将其视为可以理性交锋的合法对手。后者是民主交流的基本组成部分，而前者则可能损害公共话语。政治分歧日益发生在社交媒体上，已有研究发现对政治外群体的负面态度会驱动用户参与度。然而，现有研究大多将这种负面性视为单一维度，未能回答在这些平台上流传的究竟是敌对化言论还是批评性言论。本工作通过引入一种计算方法来检测文本中的这两种言论类型并识别其攻击目标，填补了这一空白。在一个包含超过1700万条来自热门话题的推文的语料库中……

    arXiv:2610.22455v1 Announce Type: cross  Abstract: Disagreement between political opponents is a central element of democracy. This disagreement can be voiced in different ways. An opponent can be antagonized, i.e. portrayed as an enemy that needs to be overcome, or they can be criticized, i.e. treated as a legitimate adversary with whom one can engage. While the latter is a fundamental part of democratic exchange, the former can be detrimental to public discourse. Political disagreement increasingly takes place on social media, where negativity towards political out-groups has been found to drive engagement. However, existing research has largely treated such negativity as a single dimension, leaving open whether it is antagonization or criticism that circulates on these platforms. This work addresses this gap by introducing a computational approach to detect these two types of rhetoric in text and identify its targets. On a corpus of over 17 million tweets from trending topics in the
    
[^141]: 预填充阶段专家削减在何处损害MoE模型：少数提示词Token承担了大部分短答案F1损失

    Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss

    [https://arxiv.org/abs/2610.22454](https://arxiv.org/abs/2610.22454)

    研究发现MoE模型在预填充阶段削减路由专家所造成的短答案F1损失，主要集中于少数关键提示词Token（如答案前的聊天模板Token和开头的指令Token），且损害主要在于答案形式而非答案内容，保持这些敏感位置的完整专家远比恢复超级专家或超大激活更有效。

    

    在预填充阶段让提示词Token通过更少的路由专家，是降低混合专家模型服务成本的一种简单方法，但这种做法究竟损害了什么尚不清楚。我们在四个MoE模型家族上，通过Token级、层级和注意力路径干预，在短答案问答任务中定位了这种损害。在专家数量减半的情况下，三个家族的F1损失主要来自“答案形式”：模型仍能找到答案，但不再以简短形式作答。在Qwen3-30B-A3B中，这种损失发生在答案之前的两个聊天模板Token上；在DeepSeek-V2-Lite中，损失发生在其指令所在的前8-32个提示词Token上。按答案似然计算，这些位置分别约占全部损害的一半和五分之一。专家削减对首个Token的注意力质量、范数和最大激活的改变不足4%，而恢复“超级专家”或超大激活最多只能挽回约四分之一的损失，远低于保持敏感位置启用全部专家的效果。

    arXiv:2610.22454v1 Announce Type: new  Abstract: Running prompt tokens through fewer routed experts during prefill is a simple way to make mixture-of-experts (MoE) serving cheaper, yet it is unclear what this damages. We locate the damage with token-, layer- and attention-path interventions on short-answer question answering in four MoE families. At half the experts the F1 loss of three families is mostly answer form: the model still finds the answer but stops answering briefly. In Qwen3-30B-A3B this loss is made at the two chat-template tokens before the answer, and in DeepSeek-V2-Lite at the first 8-32 prompt tokens, where its instruction sits; by answer likelihood these positions account for about half and a fifth of the damage. Reduction changes the first token's attention mass, norm and largest activation by less than 4%, and restoring super experts or massive activations recovers at most about a quarter of the loss, well below keeping the sensitive positions full. On real prompts
    
[^142]: 词嵌入定价过高？面向小型与三值语言模型的16 MB冻结词表

    Embeddings Are Overpriced? A Frozen 16 MB Vocabulary for Small and Ternary Language Models

    [https://arxiv.org/abs/2610.22428](https://arxiv.org/abs/2610.22428)

    该研究用一个无任何可训练参数、仅16 MB的冻结512位二值编码词表替代可学习嵌入表，在保持甚至略优于基模型的下游任务性能的同时，大幅削减小型及三值语言模型的参数量与文件体积，使三值模型整体仅约46 MB。

    

    在小型语言模型中，词表是占比最大的单一组成部分。在Gemma 3 270M中，嵌入表占据了2.68亿总参数中的1.68亿；由于量化方法通常将嵌入保留为16位，当模型其余部分被压缩后，嵌入表仍占模型文件的绝大部分。我们追问这张表究竟换来了什么。我们用一个由512位编码构成的冻结表（16 MB，不含任何可训练参数）将其替换——该表通过对预训练嵌入进行主成分投影与符号二值化一次性导出——并在相同的20.5亿token上从头训练相同Gemma结构的模型主体（分别带与不带该冻结表）进行对比。结果显示，可学习嵌入表以2.6倍的可训练参数仅换来0.20 nat的验证损失改善；而在LAMBADA、HellaSwag和PIQA上，冻结表模型表现更好或持平，只有ARC-Easy更偏好可学习表。搭配三值BitNet b1.58主体，整个模型仅为一个46 MB的文件，在LAMBADA和PIQA上仍与浮点模型持平，并且在笔记本电脑CPU上每秒可生成6……（摘要原文在此处截断）。

    arXiv:2610.22428v1 Announce Type: new  Abstract: In a small language model the vocabulary is the single largest part. In Gemma 3 270M the embedding table holds 168M of 268M parameters, and because quantisation methods leave the embedding in 16-bit, it is also most of the file once the rest of the model is compressed. We ask what that table actually buys. We replace it with one frozen table of 512-bit codes, 16 MB, with no trainable parameters, derived once from a pretrained embedding by principal-component projection and sign binarisation, and we train the same Gemma-shaped body from scratch on the same 2.05B tokens with and without it. The learned table buys 0.20 nats of validation loss for 2.6 times the trainable parameters, but on LAMBADA, HellaSwag and PIQA the frozen-table model is better or tied; only ARC-Easy favours the learned table. With a ternary BitNet b1.58 body the whole model is a 46 MB file, still ties the float model on LAMBADA and PIQA, and on a laptop CPU generates 6
    
[^143]: PRIME：迈向大语言模型智能体中可验证的基于证据的推理

    PRIME: Towards Verifiable Evidence-Grounded Reasoning in LLM Agents

    [https://arxiv.org/abs/2610.22426](https://arxiv.org/abs/2610.22426)

    PRIME通过预先固定证据路线并使用独立程序记录检查结果，确保LLM智能体只报告被规则和结果账本授权的结论，从而避免在必需检查失败后仍接受报告或满足于过弱的结论。

    

    大语言模型（LLM）智能体被用于根据组织的要求检查记录，例如判断一份漏洞报告是否已准备好进行调查。要求通常允许多种满足方式，例如带有策展人审核公告的严重性评级，或已发布的公告链接。智能体的失败方式包括：在必需检查失败后仍然接受报告，或者满足于较弱的结论而不尝试另一种仍可能成功的方式。我们提出了PRIME（基于最小证据的预提交推理规则）。在检查之前，它会固定证据路线，明确规定每个结论如何才能被证明合理。一个独立的程序记录每次检查的结果，并拒绝任何结果无法支持的结论。它还会向智能体指出可以证明更强结论的检查，或者自己运行这些检查。PRIME只报告由其规则和结果账本授权的结论。在要求正确、检查在已知位置精确执行且充分……（摘要原文在此处截断）

    arXiv:2610.22426v1 Announce Type: new  Abstract: LLM agents are used to check records against an organization's requirements, for example whether a vulnerability report is ready for investigation. Requirements often allow several ways to be met, such as a severity rating with a curator-reviewed advisory or a published advisory link. Agents fail by accepting a report after a required check has failed, or by settling for a weaker conclusion without trying another way that could still succeed. We introduce PRIME (Precommitted Rules for Inference from Minimal Evidence). Before checking, it fixes evidence routes specifying how each conclusion may be justified. A separate program records each check result and refuses any conclusion the results do not justify. It also points the agent to checks that could justify a stronger conclusion, or runs them itself. PRIME reports only conclusions authorized by its rules and ledger. With correct requirements, exact checks at known locations, and suffici
    
[^144]: 引出规格说明：交互式澄清何时有所帮助？

    Eliciting Specifications: When Does Interactive Clarification Help?

    [https://arxiv.org/abs/2610.22382](https://arxiv.org/abs/2610.22382)

    本文提出一种引出代理（EA）方法，在形式化之前向模拟开发者提出有预算限制的澄清问题以改进LLM生成的Lean规格说明，并系统评估了在不同问题预算（k=3与k=10）及同模型/跨模型配置下交互式澄清何时有效。

    

    LLM可以从任务描述中生成Lean规格说明，但这些描述往往不够明确或存在歧义，并且可能遗漏与验证器相关的边界情况，从而产生不完善的形式化规格说明。我们研究了一个引出代理（Elicitation Agent, EA）能否通过在形式化之前向基于参考资料构建的模拟开发者寻求澄清，来改进经过验证器检查的Lean规格说明。该代理对任务进行分解，识别尚未解决的决策点，提出有数量限制的澄清问题，并利用答案撰写精炼后的自然语言规格说明，再由LLM形式化工具将其翻译为Lean。在189个Verina Lean 4规格说明生成任务上，我们比较了单次生成与引出式方法的效果，问题预算分别设为k=3和k=10（即每个任务最多允许EA提出三个或十个问题），并涵盖了同模型与跨模型两种配置。同模型的Claude Sonnet 4.6流水线在两种……（原文摘要在此处被截断）

    arXiv:2610.22382v1 Announce Type: cross  Abstract: LLMs can generate Lean specifications from task descriptions, but those descriptions are often underspecified or ambiguous and may omit verifier-relevant edge cases, yielding imperfect formal specifications. We study whether an elicitation agent (EA) can improve verifier-checked Lean specifications by seeking clarification from a reference-grounded simulated developer before formalization. The agent decomposes the task, identifies unresolved decisions, asks bounded clarification questions, and uses the answers to write a refined natural-language specification that an LLM formalizer then translates into Lean. On 189 Verina Lean 4 specification-generation tasks, we compare single-shot generation with elicitation under question budgets of k=3 and k=10, allowing at most three or ten EA questions per task, across same-model and cross-model configurations. The same-model Claude Sonnet 4.6 pipeline has negative strict pass@1 estimates at both
    
[^145]: 面向测试时持续学习的绑定特异性再学习节省效应

    Binding-Specific Relearning Savings for Test-Time Continual Learning

    [https://arxiv.org/abs/2610.22381](https://arxiv.org/abs/2610.22381)

    该研究发现在测试时持续学习中，模型习得的特定事实绑定（而非对实体、答案词汇、格式或查询槽位的熟悉性）能在部分遗忘后带来再学习加速，且错误绑定不产生此优势，表明现有仅依赖输出准确率的评估方式无法捕捉先前经验对后续更新的塑造作用。

    

    当一个事实从语言模型中变得更难被提取时，模型是真的丢失了这个事实，还是它仅仅变得不那么容易被访问到？参数更新的测试时持续学习智能体通常以当前输出准确率来评估；我们探究这种测量方式是否能捕捉到先前经验如何塑造后续的更新。我们检验学习一个特定的“事实绑定”是否会在之后产生再学习优势，且这种优势超越了对相关实体、答案词汇、格式或查询槽位的熟悉性。在Qwen2.5-1.5B模型上，我们比较了三种匹配且经过平衡处理的历史条件：LEARNED（学习组）接受正确的绑定，CONTROL（对照组）接受匹配的暴露但不包含该绑定，WRONG（错误组）在同一查询槽位接受一个竞争性的、关系上有效的答案。在完全相同的无回放持续微调数据流之后，在预设的部分遗忘检查点上，LEARNED组比CONTROL组更快地重新习得目标事实。WRONG组相比CONTROL组没有表现出正向优势，而它与LEA（原文在此处截断）

    arXiv:2610.22381v1 Announce Type: cross  Abstract: When a fact becomes harder to elicit from a language model, has the model lost it, or has it merely become less accessible? Parameter-updating test-time continual learning agents are usually evaluated by current output accuracy; we ask whether that measurement captures how prior experience shapes later updates. We test whether learning a specific factual binding produces a later relearning advantage beyond familiarity with its entity, answer vocabulary, format, or query slot. In Qwen2.5-1.5B, we compare three matched, counterbalanced histories: LEARNED receives the correct binding, CONTROL receives matched exposure without it, and WRONG receives a competing relation-valid answer in the same slot. After an identical rehearsal-free continual fine-tuning stream, LEARNED reacquires the target faster than CONTROL at a prespecified checkpoint of partial forgetting. WRONG shows no positive advantage over CONTROL, while its comparison with LEA
    
[^146]: WEAVE：将真实世界代码演进编织为依赖结构化任务，用于代码智能体的基准测试与训练

    WEAVE: Weaving Real-World Code Evolution into Dependency-Structured Tasks for Benchmarking and Training Code Agents

    [https://arxiv.org/abs/2610.22039](https://arxiv.org/abs/2610.22039)

    WEAVE 框架将真实世界中经过验证的多 PR 代码演进历史转化为依赖结构化的可执行任务，并引入基于测试的确定性度量 Graph Credit，以更好地对长程代码智能体进行基准评估和训练。

    

    真实世界的软件工程很少由孤立的变更构成。功能开发、迁移和系统性修复往往跨越多个拉取请求（PR）展开，需要对整个代码库进行协调一致的修改。然而，大多数可执行的代码库级基准测试仍以单个 issue 或 PR 作为任务单元，而现有的多 PR 设置则依赖粗粒度的聚合和与结构无关的评估。这种错位掩盖了变更之间的依赖关系，限制了长程代码智能体的进展衡量和学习信号。我们提出了 WEAVE，一个将经过验证的 PR 历史转化为可执行的、依赖结构化任务的框架。WEAVE 利用代码库校准的类型化变更位点，将相关联的 PR 组织成有界 DAG（有向无环图），并将每个 DAG 实例化为统一的可执行任务。我们进一步引入了 Graph Credit，这是一种确定性的、基于测试的度量指标，用于衡量符合依赖关系的部分进展，并支持更密集的……

    arXiv:2610.22039v1 Announce Type: cross  Abstract: Real-world software engineering rarely consists of isolated changes. Features, migrations, and systematic fixes often unfold across multiple pull requests (PRs), requiring coordinated modifications across a repository. Yet most executable repository-level benchmarks retain a single issue or PR as the task unit, while existing multi-PR settings rely on coarse aggregation and structure-agnostic evaluation. This mismatch obscures dependencies among changes and limits progress measurement and learning signals for long-horizon code agents. We introduce WEAVE, a framework that transforms verified PR histories into executable, dependency-structured tasks. WEAVE leverages repository-calibrated typed change loci to organize related PRs into bounded DAGs and instantiate each as a unified executable task. We further introduce Graph Credit, a deterministic, test-grounded metric that measures dependency-consistent partial progress and supports dens
    
[^147]: 二比特的N-Gram记忆：大语言模型中查找表的量化

    N-Gram Memories at Two Bits: Quantizing Lookup Tables in Large Language Models

    [https://arxiv.org/abs/2610.21810](https://arxiv.org/abs/2610.21810)

    本文提出条件程序（CP）压缩表示方法，仅用两比特编码按需重构大语言模型n-gram记忆查找表中被请求的行，相比标准2比特量化将输出KL散度降低41–63%，并将DeepSeek的表存储从203GB压缩至51GB。

    

    条件记忆通过选择性查找来扩展语言模型的容量，但完整的嵌入表必须始终可用以应对每一个可能的输入。我们提出了条件程序，这是一种压缩表示方法，只需重建输入所请求的行。CP用短码和局部尺度来表示每一行，并使用一个在整个表上共享的重构程序。我们通过高斯格、范数恢复以及可独立解码的行记录来实例化这种表示。CP的两比特码相比2比特最近取整量化，将输出KL散度降低了41–63%。对三个经过训练的n-gram记忆族的分析表明，类高斯的坐标统计与行之间学习到的关系并存；在相同的存储预算下，共享格结构的表现优于我们测试的拟合码本和低秩残差变体。对于DeepSeek，该方法将表存储从203GB降至51GB，为更大的工作负载留出了空间。

    arXiv:2610.21810v1 Announce Type: cross  Abstract: Conditional memory expands language-model capacity through selective lookup, but the full embedding table must remain available for every possible input. We introduce Conditional Program (CP), a compressed representation that reconstructs only the rows an input requests. CP represents each row by a short code and a local scale, with a reconstruction program shared across the table. We instantiate this representation with a Gaussian trellis, norm restoration and independently decodable row records. CP's two-bit code lowers output KL by 41--63\% relative to 2-bit round-to-nearest. Analysis across three trained n-gram memory families shows that Gaussian-like coordinate statistics coexist with learned relationships among rows; at matched storage budgets, the shared trellis outperforms the fitted codebooks and low-rank residual variants we test. For DeepSeek, it reduces table storage from 203GB to 51GB, leaving room for larger workloads whe
    
[^148]: 当推理轨迹与证据冲突时：答案配置如何塑造证据遵循

    When Reasoning Traces Conflict with Evidence: How Answer Configuration Shapes Evidence Following

    [https://arxiv.org/abs/2610.20805](https://arxiv.org/abs/2610.20805)

    研究发现，回答时采用独立回答调用而非延续模型思考过程，可使模型遵循当前证据的能力提升29.9至60.6个百分点，而移除推理轨迹在两种配置下均会降低准确率，表明“受益于推理轨迹”与“服从于推理轨迹”是两种不同的模型特性。

    

    在回答时保留的推理轨迹扮演着两种角色：它可以提高准确率，同时当它与当前证据不一致时，也会与证据产生竞争。我们通过将经过验证的证据版本与从各版本生成的轨迹进行交叉，并在两种配置下基于相同的轨迹和证据进行回答，来区分这两种角色：其一是延续模型自身的思考过程（continuation），其二是采用独立的回答调用（separate answer call）。在HotpotQA和MuSiQue数据集上跨三个模型家族（gpt-oss、Gemma4、Qwen3）的实验表明，独立回答调用比延续方式将证据遵循度提高了29.9至60.6个百分点。在相同的配对上，移除轨迹在两种配置下都会以相近的幅度降低准确率。然而，在大多数冲突情形中，延续方式会重复过时的结论，而独立调用则更多地遵循当前证据。因此，“受益于轨迹”与“服从于轨迹”是两种截然不同的特性。进一步分析显示，轨迹的最后一行承载了其大部分价值……

    arXiv:2610.20805v1 Announce Type: new  Abstract: A reasoning trace retained at answer time plays two roles: it can raise accuracy, and it can compete with the current evidence when the two disagree. We disentangle these roles by crossing verified evidence versions with the traces generated from each and answering from the same traces and evidence under two configurations: continuation of the model's own thinking and a separate answer call. Across three model families (gpt-oss, Gemma4, Qwen3) on HotpotQA and MuSiQue, the separate call raises evidence-following over continuation by 29.9 to 60.6 points. On the same pairs, removing the trace lowers accuracy by similar amounts under both configurations. Yet continuation repeats the stale conclusion in most conflicts, and the separate call follows the current evidence far more often. Benefiting from a trace and deferring to it are therefore distinct properties. Further analyses show that the final line of a trace carries most of its value, g
    
[^149]: CascadeLedger：LLM级联中信息与资源完整性的反事实审计

    CascadeLedger: Counterfactual Auditing of Information and Resource Integrity in LLM Cascades

    [https://arxiv.org/abs/2610.20668](https://arxiv.org/abs/2610.20668)

    CascadeLedger提出了一个面向预算受限LLM级联的反事实执行审计框架，通过聚合等价轨迹有效性和反事实前缀无干扰性等机制，审计路由决策是否仅使用了合法可观测信息，并能定位最早违规点与最小隐藏见证。

    

    我们提出了CascadeLedger，一个面向预算受限语言模型级联的反事实执行审计框架。该协议强制执行“先调用后读取”语义，将成功的读取操作绑定到服务与缓存标识上，记录可用性状态及各组件的预算转换，并在接入任何目标模型或任务评分器之前冻结事件轨迹。其核心对象是聚合等价轨迹有效性：两次执行可以拥有相同的最终答案、声明的资源投影和服务直方图，但其路由决策是否仅使用了合法可观测的信息却可能存在差异。我们用反事实前缀无干扰性（CPNI）来补充单轨迹不变量，该性质在耦合可见前缀与策略随机性的同时，仅改变一个声明的隐藏族。反事实影响边界（CIF）定位最早违规的前缀以及一个最小隐藏见证；保持投影的对抗性变异（PPAM）

    arXiv:2610.20668v1 Announce Type: cross  Abstract: We introduce CascadeLedger, a counterfactual execution-audit framework for budgeted language-model cascades. The protocol enforces call-before-read semantics, binds successful reads to service and cache identities, records availability and component-wise budget transitions, and freezes the event trace before any target or task scorer is joined. The central object is aggregate-equivalent trace validity: two executions can share a final answer, declared resource projection, and service histogram while differing in whether their routing decisions used only legitimately observable information. We complement single-trace invariants with counterfactual prefix non-interference (CPNI), which couples a visible prefix and policy randomness while changing only a declared hidden family. The counterfactual influence frontier (CIF) localizes the earliest violating prefix and a minimal hidden witness; projection-preserving adversarial mutation (PPAM)
    
[^150]: Anuvani-LID：面向印度语言的鲁棒且校准良好的口语语言识别

    Anuvani-LID: Robust and Calibrated Spoken Language Identification for Indian Languages

    [https://arxiv.org/abs/2610.20263](https://arxiv.org/abs/2610.20263)

    本文提出Anuvani-LID，一个在54,643小时印度语语音上训练的1.15亿参数专用Conformer语言识别模型，覆盖22种印度语言及印度口音英语，在短促、嘈杂、窄带等困难条件下显著优于Whisper等基础语音模型及Gemini、GPT等音频大语言模型。

    

    每个多语言语音处理流水线都始于一个语言判定，而该环节的错误会悄无声息地在识别、翻译和路由中传播。在印度，这涉及横跨四大语系的22种表列语言，且这些语言常出现在短促、嘈杂、窄带以及语码混杂的音频中。基础语音模型越来越倾向于将语言识别（LID）视为一种固有的、潜在的可泛化能力。语音模型中的自回归解码器会输出一个特殊的语言ID标记，而具备音频能力的现代大语言模型在被提示时也能说出语言名称。我们在相同的音频片段上测试了这一假设，发现Whisper及类似语音模型仅对23种语言中的15种具有语言标记；在1秒输入下，Gemini 3.5 Flash的宏平均F1仅为0.52，GPT-audio仅为0.14。我们提出了Anuvani-LID，一个专用的1.15亿参数LID Conformer模型，在54,643小时的印度语语音上训练，涵盖22种印度语言及印度口音英语。同时……（原文摘要在此处截断）

    arXiv:2610.20263v1 Announce Type: new  Abstract: Every multilingual speech pipeline begins with a language decision, an error there propagates silently through recognition, translation and routing. In India it spans 22 scheduled languages across four language families, heard over short, noisy, narrowband and code-mixed audio. Foundational speech models increasingly present language identification (LID) as an inherent latent, generalized capability. Autoregressive decoders in speech models emit a special language ID token, and modern LLMs with audio capabilities can name the language when prompted. We test this assumption on identical clips and find that whisper and similar speech models has tokens for only 15 of the 23 languages, on one second inputs Gemini 3.5 Flash reaches 0.52 macro-F1 and GPT-audio 0.14. We present Anuvani LID, a dedicated 115M parameter LID conformer model trained on 54,643 hours of Indian speech spanning 22 Indian languages and Indian accented English. Along with
    
[^151]: 差距检测：通过需求与可辩护性缓解同质化

    Gap Detection: Mitigating Homogenization through Demand and Defensibility

    [https://arxiv.org/abs/2610.20162](https://arxiv.org/abs/2610.20162)

    提出Gap Detection框架，通过结合查询相关性（Demand）与现有内容相异度（Defensibility）的GapScore目标函数，在文本嵌入空间中检测相关但未被占据的方向，以缓解语言模型开放式回答的同质化问题。

    

    语言模型在回答开放式问题时常常生成语义相似的答案，即使存在许多合理的替代方案。生成有用的替代方案需要同时识别查询所要求的内容以及现有回答已经覆盖的内容。我们提出了差距检测（Gap Detection），这是一个在共享文本嵌入空间中搜索相关但尚未被占据的方向的框架。其目标函数GapScore将查询相关性（需求，Demand）与最近现有条目的相异度（可辩护性，Defensibility）相结合。该框架在单位球面上优化该乘积，利用语言模型将候选方向映射为概念词，并通过任务特定的过滤和受约束的提示将这些概念转化为文本。在五个精选主题和90个INFINITY-CHAT查询上，报告的输出碰撞率平均相对降低分别为12.8%和10.8%。我们还将该框架实例化应用于四个产品的生成式引擎优化。

    arXiv:2610.20162v1 Announce Type: new  Abstract: Language models often produce semantically similar answers to open-ended questions, even when many plausible alternatives exist. Generating useful alternatives requires identifying both what the query calls for and what existing responses already cover. We introduce Gap Detection, a framework that searches for relevant but unoccupied directions in a shared text embedding space. Its objective, GapScore, combines query relevance (Demand) with dissimilarity to the nearest existing item (Defensibility). The framework optimizes this product on the unit sphere, maps candidate directions to concept words with a language model, and uses task-specific filtering and constrained prompting to turn those concepts into text. On five curated topics and 90 INFINITY-CHAT queries, the reported mean relative reductions in output Collision are 12.8% and 10.8%, respectively. We also instantiate the framework for generative engine optimization over four produ
    
[^152]: SEA-HELM Audio：跨 SEA-HELM 子集评估多语言 ASR-LLM 流水线性能

    SEA-HELM Audio: Evaluating Multilingual ASR-LLM Pipeline Performance Across SEA-HELM Subsets

    [https://arxiv.org/abs/2610.19869](https://arxiv.org/abs/2610.19869)

    该论文提出 SEA-HELM Audio——一个由文本基准 SEA-HELM 扩展而来的东南亚多语音频评估框架，利用合成语音在五种语言上量化了级联 ASR-LLM 流水线相比文本输入的显著性能退化（多项选择推理下降 9.00–18.50 个百分点，指令遵循下降 20.00–42.86 个百分点，其中泰米尔语退化最严重）。

    

    我们推出了 SEA-HELM Audio，这是一个东南亚（SEA）多语言评估框架，它将基于文本的 SEA-HELM 基准进行扩展，用于评估级联式自动语音识别与大语言模型（ASR-LLM）流水线的性能退化。利用合成语音，我们在五种语言（中文、英语、马来语、印尼语和泰米尔语）上，针对多项选择推理和指令遵循任务，比较了语音输入与文本输入的性能表现。我们采用相同的验证器评估了开放权重和商业 ASR-LLM 系统，并将“语音差距”量化为文本准确率与语音准确率之间的差异。结果显示存在显著的性能退化：多项选择推理任务的准确率下降 9.00 至 18.50 个百分点，指令遵循任务下降 20.00 至 42.86 个百分点。其中泰米尔语的退化最为严重，超过 40 个百分点。我们的发现凸显了仅基于文本的基准在预测……（摘要在此处截断）

    arXiv:2610.19869v1 Announce Type: cross  Abstract: We introduce SEA-HELM Audio, a Southeast Asian (SEA) multilingual evaluation framework that extends the text-based SEA-HELM benchmark to assess performance degradation in cascaded automatic speech recognition and large language model (ASR-LLM) pipelines. Using synthesized speech, we compare speech-input and text-input performance across five languages (Chinese, English, Malay, Indonesian, and Tamil) on multiple-choice reasoning and instruction-following tasks. We evaluate open-weight and commercial ASR-LLM systems using identical verifiers and quantify the speech gap as the difference between text and speech accuracy. Results show substantial degradation, with accuracy drops of 9.00 to 18.50 percentage points for multiple-choice reasoning and 20.00 to 42.86 points for instruction following. Tamil exhibits the largest degradation, exceeding 40 percentage points. Our findings highlight the limitations of text-only benchmarks in predictin
    
[^153]: 医学中“系统一”决策的可靠性如何？Jev、类Jev模型与大语言模型的对比研究

    How Reliable Are System-One Decisions in Medicine? A Comparative Study of Jev, Jev-like Models, and LLMs

    [https://arxiv.org/abs/2610.18529](https://arxiv.org/abs/2610.18529)

    本研究在15个医学数据集上系统对比了Jev、类Jev模型与大语言模型，发现Jev在文本驱动任务上表现优异且决策一致性强，但在数值驱动任务上存在明显能力边界，且多数模型在识别缺失信息、实现可靠弃权方面仍有不足。

    

    Jev因其快速且可泛化的决策能力而受到广泛关注，但其在医学领域的可靠性仍未得到充分研究。为填补这一空白，本研究在统一协议下，于包含8,742个实例的15个医学数据集上比较了多种类型的系统一决策模型。除准确率之外，我们还评估了决策稳定性以及识别不可靠决策的能力。结果显示，Jev在以文本为主的任务上可与领先模型相媲美，但在以数值为主的任务上表现较弱，揭示了其明确的能力边界。在问题格式发生变化的情形下，Jev保持了较强的决策一致性和较小的准确率波动。StartLux-27B表现出相当的决策一致性，而D3-27B则取得了更高的整体准确率和更低的平均决策反转率。然而，这些优势并未延伸到可靠的弃权意识（abstention awareness）上：大多数受评估模型在识别缺失信息方面均存在局限。

    arXiv:2610.18529v2 Announce Type: replace  Abstract: Jev has attracted considerable attention for its rapid, generalizable decision-making, yet its medical reliability remains underexplored. To address this gap, our study compares various types of system-one decision models across 15 medical datasets comprising 8,742 instances under a unified protocol. Beyond accuracy, we assess decision stability and the ability to recognize unreliable decisions. Our results show that Jev rivals leading models on primarily text-driven tasks but is less effective on numerically driven tasks, revealing a distinct capability boundary. Under question-format variations, Jev maintains strong decision consistency and small accuracy fluctuations. StartLux-27B exhibits comparable consistency, while D3-27B achieves higher overall accuracy and a lower mean decision reversal rate. But these strengths do not extend to reliable abstention awareness: most evaluated models show limitations in both recognizing missing
    
[^154]: EdgeWise：通过选择性云端委托将长程智能体引入边缘设备

    EdgeWise: Bringing Long-Horizon Agents to the Edge via Selective Cloud Delegation

    [https://arxiv.org/abs/2610.17560](https://arxiv.org/abs/2610.17560)

    该论文提出EdgeWise边云协同框架，通过步骤级强化学习使边缘端小型语言模型能够自主地将关键步骤委托给云端大语言模型，在保持与完全使用云端大模型相当性能的同时显著降低推理成本。

    

    复杂且长程的智能体任务通常依赖部署在云端的前沿模型来获得强大性能。然而，在长程轨迹的每一步都使用这些模型会产生高昂的推理成本，尽管并非所有步骤都需要同等水平的模型能力。在本文中，我们发现仅将大语言模型分配到少数关键步骤上，即可获得与完全使用大语言模型执行相当的性能，同时显著降低推理成本。基于这一观察，我们提出了EdgeWise，一种边云协同框架，它将小型语言模型部署在本地设备上作为主要求解器，并在需要更强模型能力时选择性地将关键步骤委托给云端大语言模型。为了在推理时无需额外模型即可实现自主委托，我们引入了一种步骤级强化学习方法，联合优化任务求解与委托决策，并对各个步骤进行细粒度的信用分配。

    arXiv:2610.17560v2 Announce Type: replace  Abstract: Complex and long-horizon agent tasks often rely on frontier models deployed in the cloud to achieve strong performance. However, using these models for every step of a long-horizon trajectory incurs substantial inference cost, even though not all steps may require the same level of model capability. In this paper, we find that allocating the LLM to only a small subset of key steps can achieve performance comparable to full LLM execution while significantly reducing inference cost. Motivated by this observation, we propose EdgeWise, an edge-cloud collaboration framework that deploys an SLM on the local device as the primary solver and selectively delegates critical steps to a cloud LLM when stronger model capacity is needed. To enable autonomous delegation without additional models at inference time, we introduce a step-level RL method that jointly optimizes task solving and delegation, with fine-grained credit assignment for individu
    
[^155]: LoongSage：面向前沿大语言模型的生产级、高效智能体强化学习框架（附经验证的训练方案）

    LoongSage: A Production-Grade, High-Efficiency Agentic RL Framework for Frontier LLMs with Validated Recipes

    [https://arxiv.org/abs/2610.16352](https://arxiv.org/abs/2610.16352)

    LoongSage是一个智能体原生、完全异步的生产级强化学习框架，基于Megatron与SGLang构建且核心仅约1.4万行代码，通过统一编排白盒与黑盒智能体的AgentFlow范式，显著降低了前沿大语言模型智能体强化学习的采用、使用与调试开销。

    

    强化学习（RL）已成为大语言模型（LLM）后训练流程中的关键阶段。随着大语言模型能力的前沿从静态推理转向智能体任务——其特征包括外部工具调用、多轮交互和长程决策——现有主流强化学习框架变得日益笨重，增加了采用、使用和调试的开销。为应对这些挑战，我们提出了LoongSage，一个智能体原生、完全异步的强化学习框架，其核心代码仅约1.4万行。LoongSage基于Megatron实现分布式训练，基于SGLang实现高吞吐量推理，将编排逻辑集中在单一控制器进程中，并将所有驻留GPU的组件封装为Ray actor。它引入了AgentFlow这一统一执行范式，可同时编排白盒与黑盒智能体。对于异步执行，（原文摘要在此处截断，后续内容缺失）

    arXiv:2610.16352v2 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has emerged as a critical phase in the post-training pipeline of large language models (LLMs). As the frontier of LLM capabilities shifts from static reasoning to agentic tasks--characterized by external tool invocation, multi-turn interactions, and long-horizon decision-making--existing mainstream RL frameworks have grown increasingly unwieldy, raising the overhead of adoption, usage, and debugging. To address these challenges, we present LoongSage, an agent-native, fully asynchronous RL framework with a lightweight core of roughly 14K lines of code. Built upon Megatron for distributed training and SGLang for high-throughput inference, LoongSage centralizes orchestration within a single controller process and encapsulates all GPU-resident components as Ray actors. It introduces AgentFlow, a unified execution paradigm that orchestrates both white-box and black-box agents. For asynchronous execution, 
    
[^156]: 已训练TTT层中的误差修正

    Error Correction in Trained TTT Layers

    [https://arxiv.org/abs/2610.16268](https://arxiv.org/abs/2610.16268)

    该研究剖析了TTT层的梯度更新机制，证明点积目标下不存在预测误差残差、LayerNorm目标下的预测衍生分量在发布初始化处会相互抵消，并通过实验量化了归一化预测对已训练检查点上梯度更新的实际贡献。

    

    测试时训练（TTT）层在处理输入序列的同时更新快权重记忆。然而，基于梯度的更新本身并不能说明记忆如何修订已有的关联。我们研究了已发布的TTT实现中由预测衍生出的更新的结构与经验作用。对于点积目标，我们刻画了预测误差残差的缺失，并建立了重参数化后内循环上升与下降等价的条件。对于TTT-Linear和TTT-MLP所使用的基于LayerNorm的平方误差目标，我们推导了内循环梯度，并识别出一个预测衍生分量，该分量在零ε极限下于发布的初始化处会相互抵消。随后，我们在三个已训练的检查点上测量了完整归一化预测的贡献。在保留LayerNorm雅可比矩阵的同时移除该贡献，得到的平均梯度范数比值为0.25至0.32。

    arXiv:2610.16268v2 Announce Type: replace-cross  Abstract: Test-time-training (TTT) layers update fast-weight memory while processing an input sequence. A gradient-based update, however, does not by itself establish how the memory revises an existing association. We study the structure and empirical role of prediction-derived updates in released TTT implementations. For dot-product objectives, we characterize the absence of a prediction-error residual and establish conditions under which inner-loop ascent and descent are equivalent after reparameterization. For the LayerNorm-based squared-error objective used by TTT-Linear and TTT-MLP, we derive the inner gradient and identify a prediction-derived component that cancels at the released initialization in the zero-epsilon limit. We then measure the full normalized prediction's contribution on three trained checkpoints. Removing this contribution while retaining the LayerNorm Jacobian gives a mean gradient norm ratio of 0.25 to 0.32 acros
    
[^157]: 面向方面情感四元组预测的依存关系推理

    Dependency Reasoning for Aspect Sentiment Quadruple Prediction

    [https://arxiv.org/abs/2610.14873](https://arxiv.org/abs/2610.14873)

    该论文提出DRIFT方法，将依存句法树线性化后作为显式关系上下文直接注入LLM提示中，并结合检索增强的上下文学习与语法感知裁决机制，在方面情感四元组预测任务上取得了最优性能。

    

    方面情感四元组预测（ASQP）本质上是一个关系推理问题：识别“方面-观点-类别-情感”四元组需要对结构化依赖关系进行建模，而不仅仅是顺序上的共现。现有的基于大语言模型（LLM）的方法通过基于检索的示例选择或多阶段训练等方式间接引入句法信息，而非将其作为统一的、单次生成时的显式上下文。我们提出了DRIFT，该方法将依存句法分析树线性化，并作为显式的关系上下文直接注入LLM提示中，使模型能够对方面与观点之间的依赖关系进行推理，而不是仅凭词元邻近性来推断。DRIFT还进一步集成了检索增强的上下文学习以及一种语法感知的、基于LLM的裁决机制，该机制在跨度边界变化的情况下对精确匹配评估形成补充。在SemEval Rest15、Rest16和ACOS数据集上，DRIFT取得了最强的性能。

    arXiv:2610.14873v2 Announce Type: replace  Abstract: Aspect Sentiment Quadruple Prediction (ASQP) is fundamentally a relational reasoning problem: identifying aspect-opinion-category-sentiment quadruples requires modeling structured dependencies, not merely sequential co-occurrence. Existing LLM-based approaches incorporate syntactic information indirectly, via retrieval-based demonstration selection or multi-stage training, rather than as unified, single-pass generation-time context. We present DRIFT, which linearizes dependency parse trees and injects them directly into LLM prompts as explicit relational context, enabling the model to reason over aspect-opinion dependencies rather than infer them from token proximity alone. DRIFT further integrates retrieval-augmented in-context learning and a syntax-aware LLM-based adjudication mechanism that complements exact-match evaluation under span-boundary variation. On SemEval Rest15, Rest16, and ACOS, DRIFT achieves the strongest performanc
    
[^158]: SpaceCast-Bench：评估视觉-语言模型中的预测性空间推理

    SpaceCast-Bench: Evaluating Predictive Spatial Reasoning in Vision-Language Models

    [https://arxiv.org/abs/2610.12402](https://arxiv.org/abs/2610.12402)

    SpaceCast-Bench是首个通过“观察-变换-推断”框架直接且诊断性地评估视觉-语言模型预测性空间推理能力的基准，其评估揭示了最强模型（58.0%）与人类表现（87.2%）之间的显著差距。

    

    现有的空间推理基准主要测试空间感知：即从输入中读取已经可见的关系。然而，现实世界的空间智能需要预测性空间推理：从观察中构建场景、预测干预如何改变场景、并对未见到的结果进行推理。我们提出了SpaceCast-Bench，这是首个直接且诊断性地评估这一能力的基准。该基准基于“观察-变换-推断”框架构建，包含来自182个真实世界场景的3,862个问题，涵盖三个层次的16种任务类型：静态感知、局部预测和全局预测，逐步要求场景理解、空间状态更新以及对未观察到的结果进行关系推理。对21个模型的评估揭示了一个显著的差距：最强模型仅达到58.0%，而人类表现为87.2%，同时空间专用模型仍接近随机水平。受控分析进一步表明...

    arXiv:2610.12402v1 Announce Type: cross  Abstract: Existing spatial reasoning benchmarks mainly test spatial perception: reading off relations already visible in the input. Yet real-world spatial intelligence demands predictive spatial reasoning: constructing a scene from observations, anticipating how an intervention changes it, and reasoning about the unseen outcome. We introduce SpaceCast-Bench, the first benchmark to directly and diagnostically evaluate this capability. Built around an observe-transform-infer framework, its 3,862 questions from 182 real-world scenes span 16 task types at three levels: static perception, local prediction, and global prediction, progressively requiring scene understanding, spatial state updating, and relational inference over unobserved outcomes. Evaluating 21 models exposes a stark gap: the strongest model reaches only 58.0% against 87.2% human performance, while spatially specialized models remain near random chance. Controlled analyses further rev
    
[^159]: 同数引用互换：对Jev作为金融证据裁判的压力测试

    Same-Number Citation Swaps: Stress-Testing Jev as a Financial Evidence Judge

    [https://arxiv.org/abs/2610.08675](https://arxiv.org/abs/2610.08675)

    该论文通过对GPT-4.1-mini计算轨迹在相同数字单元格之间互换引用进行压力测试，发现Jev等概率证据验证器在金融报告数值大量重复的场景下仍会放过引用错误财务角色的计算并拒绝等价的有效引用，而显式列标签只能部分改善这一角色识别难题。

    

    金融报告中的数值会在不同期间、不同指标和不同会计科目之间重复出现，这使得LLM生成的计算可能在数值上正确，却引用了错误的财务角色。我们以Jev作为GPT-4.1-mini计算轨迹的来源支持验证器，评估概率性证据验证在数字匹配之外带来了什么。一个基于指针位置带符号数字的基线解释了相较于精确引用检查的大部分恢复效果。为了分离剩余的角色识别问题，我们保持操作数和算术运算不变，将引用在相同数字的单元格之间移动，并保留表达等价事实的对照组。这些对比既揭示了能够通过验证的错误角色引用，也揭示了被拒绝的有效替代引用。显式列标签改善了对部分错误角色决策的判断，同时也降低了对某些等价证据的支持。一项由非作者评审员标注、在36个新来源页面上构建的后续实验扩展了这一评估……（摘要原文在此处截断）

    arXiv:2610.08675v1 Announce Type: new  Abstract: Financial reports repeat values across periods, metrics and accounting lines, allowing an LLM-generated calculation to be numerically correct while citing the wrong financial role. We evaluate what probabilistic evidence verification adds beyond number matching using Jev as a source-support verifier for GPT-4.1-mini calculation traces. A signed-number-at-pointer baseline explains most recovery over exact quotation checks. To isolate the remaining role-recognition problem, we hold operands and arithmetic fixed, move citations between same-number cells, and retain controls that express equivalent facts. These contrasts reveal both wrong-role citations that pass and valid alternative citations that are withheld. Explicit column labels improve selected wrong-role decisions while also lowering support for some equivalent evidence. A constructed follow-up on 36 new source pages, labeled by a non-author reviewer, extends this evaluation and exp
    
[^160]: ÌròyìnSpeech 文本语料库：24,905 条面向语音与语言技术的精选约鲁巴语句子

    The \`{I}r\`{o}y\`{i}nSpeech Text Corpus: 24,905 Curated Yor\`ub\'a Sentences for Speech and Language Technology

    [https://arxiv.org/abs/2610.05366](https://arxiv.org/abs/2610.05366)

    本文发布了 ÌròyìnSpeech 语音语料库的文本组件——24,905 条经人工校验、带声调标记的约鲁巴语句子，内容覆盖新闻等多元领域以弥补现有语料偏重宗教文本的不足，并揭示了影响超过 60% 文本行的系统性 Unicode 规范化失败问题。

    

    ÌròyìnSpeech 是一个时长 42 小时、包含 80 名说话人的约鲁巴语朗读语音语料库，其音频自 2024 年起由 ELRA 发布。本文介绍了该语料库文本部分的发布：24,905 条唯一、经人工校验、带声调标记的约鲁巴语句子（共 275,897 个词元；15,687 个词型），这些句子于 2022 年作为录音提示文本整理而成。其中约 11,000 个句子改编自开放授权的新闻材料，其余句子为团队自行撰写，以扩展语料覆盖范围，超越现有约鲁巴语语料库中以宗教翻译为主导的内容。每个句子均经过人工检查以确保声调标记准确，并经过编辑以保证朗读清晰、语体中性，同时进行了本地化处理，使非约鲁巴语的人名和地名以约鲁巴语形式呈现。在准备发布文本的过程中，发现了影响超过 60% 文本行的系统性 Unicode 规范化失败问题（同一字母的预组合形式与分解形式在单个句子中共存），该问题……

    arXiv:2610.05366v2 Announce Type: replace  Abstract: \`{I}r\`{o}y\`{i}nSpeech is a 42-hour, 80-speaker Yor\`ub\'a read-speech corpus whose audio has been distributed by ELRA since 2024. This paper describes the release of its text component: 24,905 unique, hand-verified, tone-marked Yor\`ub\'a sentences (275,897 tokens; 15,687 types), curated in 2022 as recording prompts. Roughly 11,000 sentences were adapted from openly licensed news material; the remainder were written in-house to broaden coverage beyond the religious translation that dominates existing Yor\`ub\'a corpora. Every sentence was checked by hand for tone-mark accuracy, edited for read-aloud clarity and a neutral register, and localised so that non-Yor\`ub\'a personal and place names appear in Yor\`ub\'a form. Preparing the text for release surfaced systematic Unicode normalisation failures affecting more than 60% of lines (with precomposed and decomposed forms of the same letter co-occurring within single sentences) which
    
[^161]: 分数并非结构：大脑对齐与跨语言迁移

    The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer

    [https://arxiv.org/abs/2610.03827](https://arxiv.org/abs/2610.03827)

    相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。

    

    相似性分数常被用作模型与大脑或跨语言间共享结构的证据。我们探讨当结构被移除或测量工具失效时，这样的分数实际反映的是什么，并在两个场景中进行了验证。在十七种语言中，语法性探针在语言距离越远时迁移效果越差（r = -0.66），但探针自身的准确率也沿同一轴线下降，在四种语言中处于随机水平（这四种语言是距离最远的五种中的四种）。剔除这些语言会使已解释方差减半，同时由于也缩小了距离范围，该设计无法判断梯度中有多少来自测量工具本身。将272个语言对视为独立样本时，导向效应的p值为0.0006；但以十七种语言为单位时，该效应不显著（p = 0.155）。在大脑对齐方面，一个训练目标将语言模型与fMRI响应的相似性从0.10提升到0.34（可靠性上限为0.54），然而

    arXiv:2610.03827v2 Announce Type: replace-cross  Abstract: Similarity scores are often offered as evidence that a model shares structure with the brain or across languages. We ask what such a score reads when that structure is removed, or when the instrument measuring it does not work, in two settings. Across seventeen languages, a grammaticality probe transfers worse between more distant languages (r = -0.66). But the probe's own accuracy falls along the same axis and is at chance in four languages, four of the five most distant. Dropping them halves the explained variance, and because it also narrows the range of distances, the design cannot say how much of the gradient is the instrument. Counting the 272 language pairs as independent gives p = 0.0006 for a steering effect that is null when the seventeen languages are the unit (p = 0.155). In brain alignment, a training objective raises a language model's similarity to fMRI responses from 0.10 to 0.34 (reliability ceiling 0.54), yet 
    
[^162]: FSPO：面向预算约束的大语言模型强化学习后训练的策略一致风险与帕累托可行控制

    FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training

    [https://arxiv.org/abs/2610.02828](https://arxiv.org/abs/2610.02828)

    FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。

    

    自适应的大语言模型强化学习后训练会在训练过程中在线调整多个训练执行器，包括 rollout 温度、组大小、裁剪、KL 正则化、验证器分配以及更新预算。目前有三个相互耦合的问题尚未解决：从行为轨迹训练得到的未来风险模型未必能准确估计将要部署的控制器所引发的风险；基于已记录的状态-动作对校准的分数在经过选择性动作选择后可能出现校准失准；独立的单资源最小成本通常无法保证多资源延续的可行性。我们提出 FSPO，一种面向预算约束的大语言模型强化学习后训练的反馈状态控制器，能够联合解决这些问题。FSPO 学习一个策略一致的风险前瞻（risk-to-go）模型，其 Bellman 目标遵循与未来决策所用的同一个冻结控制器，并同时学习一个长程效用模型。决策条件化轨迹校准（DCTC）对风险进行校准……

    arXiv:2610.02828v1 Announce Type: new  Abstract: Adaptive LLM reinforcement-learning post-training changes multiple training actuators online, including rollout temperature, group size, clipping, KL regularization, verifier allocation, and update budget. Three coupled issues remain unresolved. A future-risk model trained from behavior trajectories need not estimate the risk induced by the controller that will be deployed; a score calibrated on logged state-action pairs can become miscalibrated after selective action choice; and independent per-resource minimum costs do not in general certify a feasible multi-resource continuation. We introduce FSPO, a feedback-state controller for budgeted LLM RL post-training that addresses these issues jointly. FSPO learns a policy-consistent risk-to-go model whose Bellman target follows the same frozen controller used for future decisions, together with a long-horizon utility model. Decision-conditioned trajectory calibration (DCTC) calibrates risk 
    
[^163]: ROUTEAUDIT：面向预算受限多验证器路由的交互感知识别方法

    ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing

    [https://arxiv.org/abs/2610.02808](https://arxiv.org/abs/2610.02808)

    ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。

    

    自适应多验证器系统通常通过端点的质量-成本差距进行比较，即使验证器目录、可用性、资源核算、信息过滤或评分器会随策略发生变化。我们将验证器路由形式化为一个契约条件化的识别问题。该契约记录了请求支持、验证器目录、实际可用性、资源核算、在线过滤以及轨迹后评分；一个匹配的路由对比仅改变策略坐标。ROUTEAUDIT为该契约增加了三个可度量的对象：契约格在所有可容许的桥接顺序上对坐标增量取平均，并报告由此得到的归因及其路径敏感性；策略无关的响应带在自适应策略揭示不同观测时识别成对的顺序对比；对于不完整的匹配，请求级边界利用仍然可观测的潜在结果，给出紧致的有限（摘要在此处截断）。

    arXiv:2610.02808v1 Announce Type: new  Abstract: Adaptive multi-verifier systems are commonly compared through endpoint quality-cost gaps, even when the verifier catalog, availability, accounting, information filtration, or scorer changes with the policy. We formulate verifier routing as a contract-conditioned identification problem. The contract records request support, verifier catalog, realized availability, resource accounting, online filtration, and post-trace scoring; a matched route contrast changes only the policy coordinate. ROUTEAUDIT adds three measurable objects to this contract. A contract lattice averages coordinate increments over every admissible bridge order and reports the resulting attribution together with its path sensitivity. A policy-independent response tape identifies paired sequential contrasts when adaptive policies reveal different observations. For incomplete matching, request-level bounds use whichever potential outcome remains observed and give a sharp fi
    
[^164]: FAER：面向语言模型后训练的可审计、效用对齐的轨迹重放

    FAER: Auditable Utility-Aligned Trajectory Replay for Language Model Post-Training

    [https://arxiv.org/abs/2610.00385](https://arxiv.org/abs/2610.00385)

    FAER提出了一种可审计的全轨迹重放框架，通过学习器感知的效用对齐选择器弥合了重放选择与下游学习效果之间的差距，在GSM8K上显著优于均匀采样和格式反馈基线。

    

    重放选择器通常依据格式反馈、置信度、新鲜度或响应长度对缓存轨迹进行排序，然而缓存层面的正确性与下游学习器的效用是两个不同的目标。我们形式化了这种“选择到学习”的差距，并提出FAER作为一个可审计的全轨迹重放框架。其中免训练的固定选择器作为协议基线；FAER-UTILITY则是在互不相交的校准块上拟合的学习器感知选择器。归一化梯度对齐被报告为基线方法，而一次性的优化器感知虚拟更新提供了幅度感知的效用面。审计契约会在评估标签加入之前冻结观测字段和重放轨迹。在GSM8K数据集上使用Qwen2.5-1.5B-Instruct模型，匹配学习器的研究显示固定选择器在128次更新下质量达到0.6329，相比之下均匀采样为0.5482，格式反馈为0.6037。仅使用元数据的交叉拟合校准达到0.6...

    arXiv:2610.00385v1 Announce Type: cross  Abstract: Replay selectors often rank cached trajectories by format feedback, confidence, freshness, or response length, although cache-level correctness and downstream learner utility are distinct objectives. We formalize this selection-to-learning gap and introduce FAER as an auditable full-trajectory replay framework. Its training-free fixed selector is a protocol baseline; FAER-UTILITY is the learner-aware selector fitted on disjoint calibration blocks. The normalized gradient alignment is reported as a baseline, while a disposable optimizer-aware virtual update supplies a magnitude-aware utility surface. The audit contract freezes observed fields and replay traces before evaluation labels are joined. On GSM8K with Qwen2.5-1.5B-Instruct, the matched learner study reports quality 0.6329 for the fixed selector, compared with 0.5482 for uniform and 0.6037 for format-feedback under 128 updates. Metadata-only cross-fitted calibration reaches $0.6
    
[^165]: ContractRL：面向可审计工具调用修复的屏蔽式组相对策略优化

    ContractRL: Shielded Group-Relative Policy Optimization for Auditable Tool-Call Repair

    [https://arxiv.org/abs/2610.00328](https://arxiv.org/abs/2610.00328)

    ContractRL提出了一种契约约束的顺序修复协议，将验证器引导的JSON修复建模为有界决策过程，通过契约派生的动作掩码过滤非法修复操作，在仅生成34.4个token的情况下实现了0.9362的语义成功率，并保证了修复过程的可审计性。

    

    结构化工具调用往往在仅有少量字段违反模式或执行契约时就会失败。重新生成完整对象会扩大动作空间，并使重复修复难以审计。我们提出了ContractRL，这是一种契约约束的顺序修复协议，将验证器引导的JSON修复建模为一个有界决策过程。在每一步中，策略观察候选对象、带类型的验证器反馈、JSON Pointer、不可变的修复历史以及剩余预算；契约派生的动作掩码在确定性验证器执行状态转移之前过滤掉格式错误或被禁止的RFC-6902操作。我们为补丁、重试和放弃决策指定了契约约束的组相对目标，同时将规范目标和语义标签保持在在线状态之外，直到轨迹冻结。在相同的验证器信息下，ContractRL仅使用34.4个生成token就达到了0.9362的语义成功率。

    arXiv:2610.00328v1 Announce Type: new  Abstract: Structured tool calls often fail after only a small number of fields violate a schema or an execution contract. Regenerating the complete object enlarges the action surface and makes repeated repair difficult to audit. We introduce ContractRL, a contract-constrained sequential repair protocol that models verifier-guided JSON repair as a bounded decision process. At each step the policy observes the candidate, typed verifier feedback, JSON Pointer, immutable repair history, and remaining budget; a contract-derived action mask filters malformed or prohibited RFC-6902 operations before a deterministic validator performs the transition. We specify a contract-constrained group-relative objective for patch, retry, and abstention decisions while keeping canonical targets and semantic labels outside the online state until trace freeze. Under identical verifier information, ContractRL attains 0.9362 semantic success with 34.4 generated tokens, co
    
[^166]: 带收据的动作：联合绑定声明、证据与执行以实现可重放的工具型智能体审计

    Actions with Receipts: Jointly Binding Claims, Evidence, and Execution for Replayable Tool-Agent Auditing

    [https://arxiv.org/abs/2610.00327](https://arxiv.org/abs/2610.00327)

    该论文提出“声明锚定执行契约”，通过收据机制将智能体发出的声明、证据来源片段、执行前缀与来源版本联合绑定，并由确定性完整性验证器重建绑定关系，从而实现对工具型智能体行为的可重放审计。

    

    使用工具的智能体可以展示引用和执行日志，却留下一个关键关联未经审计：呈现给用户的声明是否就是已提交执行所发出的、且被所引用来源支持的声明。因此，一个有效的引用和一个有效的执行轨迹可以各自形式上完好，却被移植到其他声明、动作、运行或来源版本之上。我们提出了一种以声明为锚点的执行契约，将发出的声明、其精确的来源片段、产生该声明的有序执行前缀，以及该执行所观察到的来源版本和访问状态联合绑定在一起。每份收据包含一个发射锚点，用于确定性地定位该声明在已提交答案或含声明动作中的位置，同时包含来源标识符、偏移量、哈希值、引文以及域分离的执行承诺。一个确定性的完整性验证器会在加入语义或任务标签之前重建这些绑定关系。

    arXiv:2610.00327v1 Announce Type: cross  Abstract: Tool-using agents can expose citations and execution logs while leaving a critical association unaudited: whether the claim shown to a user is the claim emitted by the committed execution and supported by the cited source. A valid citation and a valid trace can therefore remain individually well formed while being transplanted across claims, actions, runs, or source versions. We introduce a claim-anchored execution contract that jointly binds the emitted claim, its exact source span, the ordered execution prefix that produced it, and the source version and access state observed by that execution. Each receipt contains an emission anchor that deterministically locates the claim inside a committed answer or claim-bearing action, together with source identifiers, offsets, hashes, quotes, and a domain-separated execution commitment. A deterministic integrity verifier reconstructs these bindings before semantic or task labels are joined. We
    
[^167]: VStress：面向重复验证器的相关性感知审计与自适应预算分配

    VStress: Correlation-Aware Auditing and Adaptive Budget Allocation for Repeated Verifiers

    [https://arxiv.org/abs/2609.36958](https://arxiv.org/abs/2609.36958)

    本文提出VStress可审计重放契约与VStress-CA相关性感知分配策略，通过估计重复验证器调用的条件边际信息来自适应分配预算并在信息不足时停止调用，从而在固定预算下以更低的调用成本获得更高的验证平衡准确率，同时通过受控审计刻画了该机制在极端损坏率下的适用边界。

    

    重复验证器调用只有在贡献条件信息时才有价值。我们提出了VStress——一种可审计的重放契约，以及VStress-CA——一种相关性感知的分配策略。该策略在密封的校准集上估计未查询验证器的条件边际信息，对不确定性进行折扣，按调用成本归一化，并在下一次调用不具信息量时停止或弃权。控制器在接入干净预言机之前冻结其决策和成本账本；当依赖性偏移警报触发时，禁用通道偏好并回退到精确停止机制。受控审计给出了该机制的适用边界：在35%对称损坏下，多数投票-5将平衡准确率从0.6578提升至0.7739；而在65%损坏下则损失0.1226个百分点。在固定预算匹配比较中，广度、冗余和自适应分配策略分别取得0.6048、0.6375和0.6538的平衡准确率，每项平均调用3.4216次，RLVR分数为0.64。

    arXiv:2609.36958v1 Announce Type: cross  Abstract: Repeated verifier calls are useful only when they contribute conditional information. We introduce VStress, an auditable replay contract, and VStress-CA, a correlation-aware allocation policy that estimates the conditional marginal information of an unqueried verifier on a sealed calibration split, discounts uncertainty, normalizes by call cost, and stops or abstains when the next call is not informative. The controller freezes its decision and cost ledger before joining the clean oracle; a dependence-shift alarm disables channel preference and falls back to exact-stop. The controlled audit gives the mechanism boundary: at 35% symmetric corruption, majority-5 improves balanced accuracy from 0.6578 to 0.7739, whereas at 65% it loses 0.1226 points. In the matched fixed-budget comparison, breadth, redundancy, and adaptive allocation obtain balanced accuracies 0.6048, 0.6375, and 0.6538, with 3.4216 calls per item and an RLVR score of 0.64
    
[^168]: 约鲁巴语与Unicode：一个问题综述

    Yor\`{u}b\'{a} in Unicode: An Overview of a Problem

    [https://arxiv.org/abs/2609.33734](https://arxiv.org/abs/2609.33734)

    本文记录了Unicode缺失约鲁巴语预组合字符所导致的数字书写乱象，指出NFC规范化稳定性政策是根本障碍，并呼吁Unicode联盟直接干预、正式编码四个核心约鲁巴字符。

    

    约鲁巴语在互联网和计算机上的书写存在一个多年来始终难以解决的反复出现的问题。这门语言与其他依赖变音符号进行消歧的非洲语言一样，需要一小组Unicode并未编码的预组合字符。这迫使书写者和数字系统只能依赖组合字符序列，而这些序列在不同平台上表现不一致，在字体替换时会损坏，并且在搜索时会失效。本文利用个人经历和实证证据，记录了这一失败在从出版书籍、网络平台到移动键盘等一系列真实世界场景中的表现。文章指出Unicode的NFC规范化稳定性政策是阻止直接修复的结构性约束，主张Unicode联盟直接介入解决这一现存问题，并提出了针对四个核心约鲁巴字符的正式编码请求。

    arXiv:2609.33734v4 Announce Type: replace  Abstract: There is a recurrent problem in the writing of Yor\`ub\'a on the internet and on the computer that has proven intractable over the years. The language, along with other African languages that depend on diacritics for disambiguation, requires a small set of precomposed characters that Unicode does not encode. This has forced writers and digital systems to rely on combining character sequences that behave inconsistently across platforms, corrupt under font substitution, and fail in search. This paper documents that failure across a range of real world contexts, from published books to web platforms to mobile keyboards, using personal and empirical evidence. It identifies Unicode's NFC normalization stability policy as the structural constraint that prevents a straightforward fix, arguing for direct intervention of the Consortium in solving the active problem, proposing a formal encoding request for the four core Yor\`ub\'a characters a
    
[^169]: 审计优先的VAPO：不完美验证下的风险认证选择性更新

    Audit-First VAPO: Risk-Certified Selective Updates under Imperfect Verification

    [https://arxiv.org/abs/2609.33662](https://arxiv.org/abs/2609.33662)

    该论文提出Audit-First VAPO框架，通过将更新方向的离散准入与幅值的连续控制相分离，在有限验证预算下利用“接受-申诉-弃权”观察策略和有限样本统计界，实现对不完美验证器下有害更新风险、覆盖率与验证器调用率的可认证保障。

    

    不完美的验证器即使在裁剪和正则化限制了其幅值的情况下，仍可能分配出有害的更新方向。我们提出了Audit-First VAPO，它将离散的方向准入与连续的幅值控制分离开来。一种仅用于观察的“接受-申诉-弃权”策略使用有限的二次验证预算；其动作轨迹在加入干净标签之前被冻结。随后，同时的有限样本界在一个预先声明的策略族上对选定的有害风险、覆盖率以及验证器调用率进行认证。条件Hoeffding-Azuma界解释了由共享预算所引起的依赖性，而rollout或验证器的变更会启动新的认证阶段。在准入之后，一个有界的trust-clip-KL执行器来控制幅值大小。我们在两个模型和两个推理基准上对该方法进行评估，对比了静态RLVR、匹配随机选择、置信度阈值、噪声校正以及验证器增强等基线方法。在Qwen3.5-0.8B和GSM8K上，在目标……

    arXiv:2609.33662v2 Announce Type: replace  Abstract: Imperfect verifiers can assign a harmful update direction even when clipping and regularization bound its magnitude. We introduce Audit-First VAPO, which separates discrete directional admission from continuous magnitude control. An observation-only accept-appeal-abstain policy uses a finite secondary-verification budget; its action trace is frozen before clean labels are joined. Simultaneous finite-sample bounds then certify selected harmful risk, coverage, and verifier-call rate over a predeclared policy family. Conditional Hoeffding-Azuma bounds account for the dependence induced by shared budgets, and rollout or verifier changes initiate a new certification stage. After admission, a bounded trust-clip-KL actuator controls magnitude. We evaluate two models on two reasoning benchmarks against static RLVR, matched-random selection, confidence thresholding, noise correction, and verifier augmentation. On Qwen3.5-0.8B and GSM8K at tar
    
[^170]: 脑到语言解码：任务、信号、方法、评估、实际应用及未来展望

    Brain-to-Language Decoding: Tasks, Signals, Methods, Evaluation, Practical Use and Beyond

    [https://arxiv.org/abs/2609.27650](https://arxiv.org/abs/2609.27650)

    这是一篇关于脑到语言解码的系统性综述，将发音、内部和感知三类语言任务与对应的神经群体、解码器表征及输出形式相联系，全面梳理了侵入式与非侵入式测量下的方法、评估体系与实际应用的最新进展。

    

    脑到语言解码旨在将与语言产生、内部言语和感知相关的神经活动转化为语言或表达性输出。它为言语丧失后的沟通功能恢复提供了一条途径，同时也是研究大脑如何表征语言的手段。神经记录和表示学习技术的进步，已将该领域从受限的识别与声学重建扩展到文本生成、流式个性化语音和面部动画。本综述综合了侵入式与非侵入式测量方面的这些进展，检索范围不设年份下限，并以文献来源为导向更新至2026年9月。我们将发音言语、内部言语和感知言语三类任务与其所涉及的神经群体、解码器可用的表征以及这些表征所能支持的输出联系起来。我们考察了模型开发、公共资源以及评估方法的演变，并对……（摘要在此处截断）

    arXiv:2609.27650v1 Announce Type: new  Abstract: Brain-to-language decoding translates neural activity associated with language production, internal speech and perception into linguistic or expressive outputs. It offers a route to restoring communication after speech loss and a means of studying how the brain represents language. Advances in neural recording and representation learning have expanded the field from constrained recognition and acoustic reconstruction to text generation, streaming personalised speech and facial animation. This survey synthesises these developments across invasive and non-invasive measurements, drawing on a search without a lower year limit and source-led updates through September 2026. We connect Articulated, Inner and Perceived tasks to the neural populations they engage, the representations available to decoders and the outputs those representations can support. We examine model development, public resources and the evolution of evaluation, and compare 
    
[^171]: DINO吞噬几何：吸收手部关节几何信息的自监督手语表示

    DINO Eats Geometry: Self-Supervised Sign Representations that Absorb Articulated Hand Geometry

    [https://arxiv.org/abs/2609.06296](https://arxiv.org/abs/2609.06296)

    SignDEG提出了一种自监督手语编码器，通过在预训练中引入几何潜在预测（GLP）和几何关系迁移（GRT）目标来吸收手部关节几何信息，推理时仅需视频输入，在How2Sign/OpenASL上的BLEU-4分数超过了SHuBERT基线。

    

    手形和手向是手语的几何组成部分，然而许多手语编码器仅从外观进行学习。我们提出SignDEG（DINO Eats Geometry），这是一种时间性手语编码器，它将关节化的手部几何作为预训练目标，且在推理时仅需要视频输入。跟踪得到的手部和面部的冻结DINOv3特征被输入到各模态流的时间Transformer中。几何潜在预测（GLP）在中间层利用WiLoR潜在表示、MANO关节建模和3D关节点进行监督，包括在被掩码的帧上。几何关系迁移（GRT）则在最终层匹配跨视频的帧相似度分布。这两个目标与时间自蒸馏联合学习；训练完成后，WiLoR和几何预测头将被丢弃。在SHuBERT公开数据评估协议下，SignDEG在How2Sign/OpenASL上达到17.9/24.5的BLEU-4分数，相比之下SHuBERT为16.2/23.2。Rank-1自适应提升了在ASL Citizen和Sem-Lex数据集上的表现（原文在此处截断）。

    arXiv:2609.06296v2 Announce Type: replace-cross  Abstract: Handshape and orientation are geometric components of sign language, yet many sign encoders learn from appearance alone. We introduce SignDEG (DINO Eats Geometry), a temporal sign encoder that uses articulated hand geometry as a pre-training target and requires only video at inference. Frozen DINOv3 features of tracked hands and faces feed per-stream temporal Transformers. Geometric latent prediction (GLP) supervises an intermediate block with WiLoR latents, MANO articulation, and 3D joints, including at masked frames. Geometric relation transfer (GRT) matches cross-video frame-similarity distributions at the final block. Both objectives are learned jointly with temporal self-distillation; WiLoR and the geometry heads are then discarded. Under the SHuBERT public-data evaluation protocol, SignDEG reaches 17.9/24.5 BLEU-4 on How2Sign/OpenASL, compared with 16.2/23.2 for SHuBERT. Rank-1 adaptation improves ASL Citizen and Sem-Lex 
    
[^172]: IndicDetect：评估针对印地语、泰卢固语和泰米尔语的跨语言LLM生成文本检测

    IndicDetect: Evaluating Cross-Lingual LLM-Generated Text Detection for Hindi, Telugu, and Tamil

    [https://arxiv.org/abs/2608.29919](https://arxiv.org/abs/2608.29919)

    该论文提出了IndicDetect基准，系统评估印地语、泰卢固语和泰米尔语中AI生成文本检测器在领域偏移、生成器偏移和对抗性扰动等现实分布偏移下的鲁棒性，并发现现有检测器存在显著的鲁棒性缺陷。

    

    大语言模型（LLM）的快速普及进一步凸显了开发可靠的人工智能生成文本检测技术的必要性，尤其是在英语以外的语言领域。然而，当前的基准测试很少关注印度语言，并且通常在脱离现实世界的理想化设置中测试检测器。我们提出了一个针对印地语、泰卢固语和泰米尔语的人工智能生成文本检测的通用基准，称为IndicDetect，旨在评估检测器在现实分布偏移下的鲁棒性。IndicDetect包含精心整理的人类撰写文本及其对应的LLM生成文本，涵盖多个领域和多种生成器，并系统地在领域偏移、生成器偏移和对抗性扰动的情况下评估检测器。使用统一且可重复的评估方案，我们评估了广泛的统计类和神经类检测器。我们发现检测器存在显著的鲁棒性缺陷：有监督神经检测器在……（原文摘要在此处截断）

    arXiv:2608.29919v1 Announce Type: cross  Abstract: The rapid proliferation of LLMs has further heightened the need to develop dependable AI-generated text detection, especially beyond English. Nevertheless, current benchmarks pay little attention to Indic languages and test detectors in idealized settings that do not represent the real world. We present a generalized benchmark for AI-generated text detection in Hindi, Telugu, and Tamil, which we call IndicDetect, designed to assess the robustness of detectors under realistic distribution shifts. IndicDetect comprises highly curated human-written texts matched with LLM-generated counterparts across various domains and generators, and systematically evaluates detectors in the presence of domain shift, generator shift, and adversarial perturbation. Using a single and repeatable evaluation scheme, we evaluate a wide range of statistical and neural detectors. We find substantial robustness failures: supervised neural detectors perform well 
    
[^173]: ConceptGuard：大型语言模型中上下文敏感遗忘的基准测试

    ConceptGuard: Benchmarking Context-Sensitive Unlearning in Large Language Models

    [https://arxiv.org/abs/2608.20338](https://arxiv.org/abs/2608.20338)

    本文提出ConceptGuard基准，通过引入双重用途概念，评估LLM在概念层面实现上下文敏感遗忘的能力，确保消除有害应用同时保留良性知识。

    

    arXiv:2608.20338v1 公告类型：新 摘要：大型语言模型（LLMs）越来越需要选择性移除有害或敏感知识，这称为遗忘，然而现有方法和基准测试未能完全评估这一能力。当前方法依赖于由独立事实组成的不相交的遗忘集和保留集，并通过简单直接的事实回忆来衡量成功。这种框架未能捕捉遗忘的一个关键要求，即在保留良性且有益知识的同时消除有害行为的能力。我们认为，有效的遗忘必须在概念层面运作，确保完全移除不安全应用，同时保持其正确且有用的用途，从而实现概念上有意义且完整的遗忘。为了从这种实用视角更好地评估遗忘技术，我们引入了双重用途概念的概念：即可以在有害和良性上下文中使用的概念。基于此，

    arXiv:2608.20338v1 Announce Type: new  Abstract: Large Language Models (LLMs) increasingly require selective removal of harmful or sensitive knowledge, called unlearning, yet existing methods and benchmarks fail to evaluate this capability completely. Current approaches rely on disjoint forget and retain sets composed of independent facts, and measure success using simple and direct factual recall. This framing fails to capture a key requirement of unlearning, namely the ability to eliminate harmful behaviors while preserving benign and beneficial knowledge. We argue that effective unlearning must operate at the level of concepts, ensuring complete removal of unsafe applications while maintaining their correct and useful usage, thereby achieving conceptually meaningful and complete unlearning. To better evaluate unlearning techniques from such a practical viewpoint, we introduce the notion of dual-use concepts: concepts that can be used in both harmful and benign contexts. Building on 
    
[^174]: FrenchNews-7：跨出版商法语新闻编辑部分类基准测试

    FrenchNews-7: Benchmarking Cross-Publisher French News Editorial Desk Classification

    [https://arxiv.org/abs/2608.18097](https://arxiv.org/abs/2608.18097)

    本文提出了FrenchNews-7基准，通过混合标注和微调CamemBERT分类器，在跨出版商法语新闻分类任务上超越了零样本大语言模型，尤其在处理未见媒体时表现优异。

    

    arXiv:2608.18097v1 公告类型：新 摘要：我们提出了FrenchNews-7，一个基于法国的跨出版商法语新闻编辑部分类基准，结合了大型多来源语料库、基于URL的七类分类体系以及微调的CamemBERT分类器。标签通过混合流水线分配，该流水线结合了出版商URL片段和用于结构模糊案例的LLM标注，并通过评估者间研究进行审核（2名人类+2个LLM；成对κ≥0.766，人类-人类κ=0.806）。我们在分布内和保留出版商设置下评估了词汇、多语言和法语特定训练分类器，并在保留池中与零样本LLM基线（GPT-OSS-120B、Mistral Small 3.2、Llama-3.3-70B）进行了额外比较。最强模型CamemBERT-base在全文输入上表现最佳，优于仅标题输入，能泛化到未见过的媒体，并在总体召回率（0.799）上超过了所有三个零样本LLM基线，差距显著。

    arXiv:2608.18097v1 Announce Type: new  Abstract: We present FrenchNews-7, a cross-publisher France-based French-language news editorial desk classification benchmark combining a large multi-outlet corpus, a URL-derived seven-class taxonomy, and a fine-tuned CamemBERT classifier. Labels are assigned via a hybrid pipeline combining publisher URL slugs with LLM annotation for structurally ambiguous cases, audited through an inter-rater study (2 humans + 2 LLMs; pairwise $\kappa \geq 0.766$, human--human $\kappa = 0.806$). We evaluate lexical, multilingual, and French-specific trained classifiers under both in-distribution and held-out-publisher settings, with additional comparison against zero-shot LLM baselines (GPT-OSS-120B, Mistral Small 3.2, Llama-3.3-70B) on the held-out pool. The strongest model, CamemBERT-base on full article text, outperforms headline-only input, generalizes to unseen outlets, and exceeds all three zero-shot LLM baselines on overall recall (0.799), with the gap co
    
[^175]: 自我标签与他者标签在LLM评审中引发双向偏差

    Self- and Other-Labels Induce Bidirectional Bias in LLM Judges

    [https://arxiv.org/abs/2608.18091](https://arxiv.org/abs/2608.18091)

    本文通过改变评估对象为无风格指纹的叙事约束选择，发现LLM评审中的自我偏好在控制质量后基本消失，甚至在某些维度上出现反向偏差，揭示了自我偏好并非固有属性。

    

    arXiv:2608.18091v1 公告类型：跨领域 摘要：随着LLM作为评审系统的日益普及，LLM中的自我偏好——即倾向于偏爱自身输出的倾向——引发了对评估可靠性的日益担忧。然而，这一现象主要是在生成文本上研究的，其中风格特征和响应质量不可避免地混杂在一起。因此，现有测量无法将真正的自我偏好与这些混淆因素区分开来。我们通过改变评估对象来解决这一问题：不是评判生成的文本，而是让十个LLM评估叙事约束选择，这些选择不携带模型特定的风格指纹，但保留可恢复的模型特定签名。我们进行了两个实验，得出了不同的发现。在盲评条件下，一旦控制选择质量和评审严格程度，自我偏好基本消失。它在四个评分维度中的三个上消失，在第四个上发生逆转，评审者将自己的选择评为更低。

    arXiv:2608.18091v1 Announce Type: cross  Abstract: As LLM-as-a-judge systems become increasingly widespread, self-preference in LLMs -- the tendency to favor one's own outputs -- raises growing concerns about evaluation reliability. However, it has been studied predominantly on generated text, where stylistic features and response quality are inevitably conflated. As a result, existing measurements cannot separate genuine self-preference from these confounds. We address this by changing the object of evaluation: instead of judging generated text, ten LLMs assess narrative constraint selections, which carry no model-specific stylistic fingerprint yet retain a recoverable model-specific signature. We run two experiments that yield distinct findings. Under blind evaluation, self-preference largely disappears once selection quality and evaluator severity are controlled. It vanishes on three of four rubric dimensions and reverses on the fourth, where judges rate their own selections as less
    
[^176]: BEAR-Bench：面向多模态模型的双语企业与学术推理基准

    BEAR-Bench: A Bilingual Enterprise and Academic Reasoning Benchmark for Multimodal Models

    [https://arxiv.org/abs/2608.17895](https://arxiv.org/abs/2608.17895)

    BEAR-Bench是一个全新的英俄双语基准，包含1000个基于专业商业和科学文档的人工标注问题，用于评估多模态大语言模型在文本密集场景下的推理能力，并揭示了当前最强模型仍存在显著性能提升空间。

    

    arXiv:2608.17895v1 公告类型：交叉 摘要：尽管多模态大语言模型（MLLMs）在视觉理解方面取得了显著进展，但它们在处理文本密集、专业文档方面的推理能力仍未得到充分评估。现有基准侧重于信息提取，需要外部领域知识，或将专业文档仅作为众多场景之一来覆盖。此外，这些基准大多以英语或中文为中心，导致其他语言，尤其是俄语，代表性严重不足。为解决这些局限性，我们引入了BEAR-Bench（双语企业与学术推理基准），这是一个自包含的、复杂的英俄双语基准，包含基于文本丰富的商业和科学文档的1000个人工标注问题。我们在BEAR-Bench上评估了16个专有和开源权重的MLLMs，包括Gemini 3.1 Pro和Qwen3.5-397B，并观察到即使是最强大的系统也存在明显的提升空间。最后，我们利用所得模型输出进行了进一步分析。

    arXiv:2608.17895v1 Announce Type: cross  Abstract: While Multimodal Large Language Models (MLLMs) have made significant strides in visual comprehension, their ability to reason about text-dense, professional documents remains incompletely evaluated. Existing benchmarks emphasize information extraction, require external domain knowledge, or cover professional documents only as one of many settings. They are also largely English- or Chinese-centric, leaving other languages and Russian, in particular, substantially underrepresented. To address these limitations, we introduce BEAR-Bench (Bilingual Enterprise and Academic Reasoning), a self-contained, complex English-and-Russian benchmark comprising 1000 human-annotated questions based on text-rich business and scientific documents. We evaluate 16 proprietary and open-weight MLLMs, including Gemini 3.1 Pro and Qwen3.5-397B, on BEAR-Bench and observe clear headroom even for the strongest systems. Finally, we use the resulting model outputs t
    
[^177]: Poli-Bias：理解与测量大语言模型在国际政治冲突中的偏见

    Poli-Bias: Understanding and Measuring Large Language Model Biases in International Political Conflicts

    [https://arxiv.org/abs/2608.06123](https://arxiv.org/abs/2608.06123)

    提出了POLI-BIAS反事实测量框架，通过系统性互换国家身份的配对提示，将LLM在国际政治冲突中的偏见响应差异分解为五个可解释维度，发现模型对处于冲突或紧张状态的国家存在不均衡对待。

    

    测量大语言模型（LLM）的政治偏见仍然具有挑战性，因为它可能通过在议题框架设定、论证方式以及法律推理方面的细微差异表现出来，而这些差异难以用单一指标加以捕捉。在这项工作中，我们提出了POLI-BIAS，一个用于测量LLM是否因涉及国家不同而对法律上等同的冲突场景进行差异化处理的反事实框架。POLI-BIAS通过对比配对提示的响应来工作，其中国家身份在不同地缘政治关系、法律违规行为和推理任务中被系统性地互换。我们的框架并不将偏见简化为单一判断，而是将响应差异分解为五个可解释的维度，从而揭示不平等对待是如何以及在何处显现的。在涵盖5,600个提示对以及来自不同模型家族和规模的七个开放权重LLM的实验中，我们发现模型对处于活跃冲突或紧张状态的国家……

    arXiv:2608.06123v3 Announce Type: replace  Abstract: Measuring political bias in large language models (LLMs) remains challenging as it can manifest through subtle differences in framing, argumentation, and legal reasoning that are difficult to capture with a single metric. In this work, we introduce POLI-BIAS, a counterfactual framework for measuring whether LLMs treat legally equivalent conflict scenarios differently depending on the countries involved. POLI-BIAS compares responses to paired prompts in which country identities are systematically swapped across diverse geopolitical relationships, legal violations, and reasoning tasks. Rather than reducing bias to a single judgment, our framework decomposes response disparities into five interpretable dimensions, revealing how and where unequal treatment manifests. Across 5,600 prompt pairs and seven open-weight LLMs from diverse model families and sizes, we find that models treat countries in active conflict or tension less evenly tha
    
[^178]: 无需滚动模拟的任务难度预测

    Predicting Task Difficulty Without Rollouts

    [https://arxiv.org/abs/2608.05797](https://arxiv.org/abs/2608.05797)

    该论文提出无需实际运行智能体（rollout）即可在覆盖编码、数学、网页导航等17个智能体基准上进行事前任务难度预测，从而降低试错带来的计算成本，并揭示AUC等常用指标可能掩盖较差的难度估计。

    

    评估和训练自主智能体的一个根本挑战，是衡量它们所尝试任务的内在难度。对这一指标的估计对于创建合成数据或基准的环境设计者，以及构建训练课程都非常有用，从而提供了一种降低计算成本的途径。随着智能体（尤其是基于大语言模型的智能体）进入更长时程的领域，这种估计变得愈发重要，因为经验性的试错会成为严重的计算瓶颈。然而，现有工作大多局限于静态任务，并且所依赖的评估指标——正如我们所展示的——可能无法完整反映预测性能。在本文中，我们研究了涵盖编码、数学、机器学习、网页导航、函数调用及其他领域的17个智能体基准上的事前（ex ante）难度预测。我们发现，AUC可能掩盖较差的难度估计，并识别出……（原文摘要在此处截断）

    arXiv:2608.05797v3 Announce Type: replace-cross  Abstract: A fundamental challenge in evaluating and training autonomous agents is measuring the intrinsic difficulty of the tasks they attempt. Estimating this quantity can be useful for environment designers creating synthetic data or benchmarks, as well as for constructing training curricula, thereby offering a way to reduce compute costs. Such an estimation becomes increasingly important as agents (particularly LLM-based) move into longer-horizon domains, where empirical trial-and-error becomes a severe computational bottleneck. However existing work is largely confined to static tasks and relies on evaluation metrics that, as we show, can give an incomplete picture of predictive performance. In this paper we study ex ante difficulty prediction across 17 agentic benchmarks spanning coding, mathematics, machine learning, web navigation, function calling, and other domains. We find that AUC can mask poor difficulty estimates, identify t
    
[^179]: CDAE：利用对比去噪增强预训练语言模型的扰动鲁棒性

    CDAE: Enhancing Perturbation Robustness in Pretrained Language Models with Contrastive Denoising

    [https://arxiv.org/abs/2607.28236](https://arxiv.org/abs/2607.28236)

    提出一种轻量级对比去噪自编码器CDAE，通过联合优化对比与重建目标学习扰动不变的句子表示，显著增强了预训练语言模型嵌入对文本扰动的鲁棒性。

    

    预训练语言模型显著提升了句子表示学习的效果，但其嵌入表示仍然对同义词替换、掩码和词丢弃等文本扰动十分敏感。本工作提出了一种轻量级的对比去噪自编码器（CDAE），通过联合优化对比目标和重建目标来精炼预训练BERT的嵌入表示，从而学习到对扰动不变的表示。我们在多种扰动策略和强度下评估了所提出的框架，并将其与原始BERT嵌入和SimCSE进行了比较。实验结果表明，CDAE在扰动下始终保持着更高的嵌入相似度，且随着扰动强度的增加，这一差距进一步扩大，凸显了扰动不变学习是提升句子嵌入稳定性的一种有前景的方向。源代码已公开发布于：https://github.com/Comp

    arXiv:2607.28236v2 Announce Type: replace  Abstract: Pre-trained language models have significantly improved sentence representation learning, yet their embeddings remain sensitive to textual perturbations such as synonym substitution, masking, and word dropout. This work proposes a lightweight Contrastive Denoising Autoencoder (CDAE) that refines pre-trained BERT embeddings by jointly optimizing contrastive and reconstruction objectives to learn perturbation-invariant representations. We evaluate the proposed framework under multiple perturbation strategies and strengths, and compare it against the original BERT embeddings and SimCSE. Experimental results show that CDAE consistently preserves higher embedding similarity under perturbation, with the gap widening as the perturbation strength increases, highlighting perturbation-invariant learning as a promising direction for improving the stability of sentence embeddings. The source code is publicly available at: https://github.com/Comp
    
[^180]: EKLAV：通过仅答案监督引出关键思维链学习

    EKLAV: Elicited Key CoT Learning via Answer only superVision

    [https://arxiv.org/abs/2607.25182](https://arxiv.org/abs/2607.25182)

    EKLAV将教师模型的思维链轨迹作为特权输入上下文而非模仿目标，仅对最终答案进行监督训练，在重排序任务上以更低的训练计算成本和推理时间超越了传统推理蒸馏方法。

    

    推理蒸馏训练学生模型复现教师模型的思维链轨迹及其最终答案。我们探讨是否将思维链轨迹用作特权训练上下文比作为模仿目标更为有效。EKLAV将教师模型的推理保留在输入序列中，但仅对最终任务答案进行监督，并采用任务相关的放置策略，将部分或全部思维链轨迹移入提示中。在BRIGHT段落重排序任务上，EKLAV在两个模型家族的所有四个测试检查点上都提升了平均nDCG@10，在48个模型-领域对比中赢得41个，同时将训练计算量减少高达31.6%，推理时间减少高达32.7%。在四个留出的表格重排序基准上，它在四个条件中的三个中表现最佳，平均超过完整思维链蒸馏方法3.0分，超过无推理的仅答案训练5.5分。对生成轨迹的分析表明，EKLAV学生模型会延迟其首次口头表达的（摘要在此处截断）

    arXiv:2607.25182v2 Announce Type: replace-cross  Abstract: Reasoning distillation trains a student to reproduce both the teacher's chain-of-thought trace and its final answer. We ask whether the trace is better used as privileged training context than as an imitation target. EKLAV keeps the teacher's reasoning in the input sequence but supervises only the final task answer, applying a task-dependent placement that moves part or all of the trace into the prompt. On BRIGHT passage reranking, EKLAV improves average nDCG@10 at all four tested checkpoints across two model families, winning 41 of 48 model-domain comparisons while reducing training compute by up to 31.6% and inference time by up to 32.7%. On four held-out table-reranking benchmarks, it is the best of four conditions on three, exceeding full-trace distillation by 3.0 points and answer-only training without reasoning by 5.5 points on average. Analysis of generated traces reveals that EKLAV students defer their first verbalized 
    
[^181]: STAIF：一种面向复杂指令遵循的分阶段优化方法

    STAIF: A Stage-wise Optimization for Complex Instruction Following

    [https://arxiv.org/abs/2607.22649](https://arxiv.org/abs/2607.22649)

    STAIF提出分阶段优化框架，先用多负样本偏好优化对齐软约束，再用可验证奖励强化学习强制满足硬约束，并构建了约3.1万条双语复杂多约束指令数据集STAINSTRUCT，有效提升大模型复杂指令遵循能力。

    

    对于大型语言模型（LLM）而言，遵循包含多个明确约束的复杂指令仍然是一个根本性挑战。现有的对齐方法（如DPO）优化整体奖励信号，往往低估了对单个约束的严格满足，尤其是在分布外或多约束设置下。本文提出STAIF，一种分阶段优化框架，将主观（软）约束的对齐与客观可验证（硬）约束的优化进行解耦。第一阶段采用带多个负样本的偏好优化来增强模型对软约束的敏感性，第二阶段采用可验证奖励强化学习（RLVR）来强制模型严格执行硬约束。为支持该方法，我们构建了STAINSTRUCT——一个高质量的双语（英语、中文）数据集，包含约31,000条复杂的多约束指令。

    arXiv:2607.22649v2 Announce Type: replace  Abstract: Following complex instructions with multiple explicit constraints remains a fundamental challenge for large language models (LLMs). Existing alignment methods, such as DPO, optimize holistic reward signals that often underemphasize strict satisfaction of individual constraints, particularly under out-of-distribution or multi-constraint settings. In this paper, we propose STAIF, a stage-wise optimization framework that decouples the alignment of subjective (soft) constraints from the optimization of objectively verifiable (hard) constraints. Stage 1 applies preference optimization with multiple negative samples to sharpen sensitivity to soft constraints, while Stage 2 applies Reinforcement Learning with Verifiable Rewards (RLVR) to enforce strict compliance with hard constraints. To support this method, we construct STAINSTRUCT, a high-quality bilingual (English, Chinese) dataset of approximately 31,000 complex multi-constraint instru
    
[^182]: 面向约鲁巴语的情境语音合成器：系统设计、音系规则架构与声调轮廓的正字法扩展

    A Situational Speech Synthesizer for Yoruba: System Design, Phonological Rule Architecture, and Orthographic Extensions for Contour

    [https://arxiv.org/abs/2607.18317](https://arxiv.org/abs/2607.18317)

    本文提出了TTSYoruba，一个基于规则的双音素拼接式约鲁巴语语音合成器，其核心创新在于手工构建的音系规则架构（涵盖声调文件选择逻辑、三重鼻音歧义消解以及从平声推导升降调），并提出采用扬抑符和长音符作为单元音声调轮廓的标准正字法符号。

    

    我们提出了TTSYoruba，一个基于规则的约鲁巴语双音素拼接式语音合成器，作为YorubaName.com约鲁巴人名开放词典的组成部分在线部署。该系统以带声调标记的约鲁巴语文本为输入，通过将手工构建的音系规则系统应用于录制的651个双音素单元库来生成音频输出，这些单元涵盖了该语言中所有辅音-元音组合的五种声调变体。我们详细描述了该系统的音系架构，包括完整的声调文件选择逻辑、对三重鼻音歧义问题（口部音/n/、鼻化元音和音节鼻音）的处理方法，以及从平声输入推导语境中升调和降调的方法。作为正字法层面的贡献，我们还提出采用扬抑符（caron）和长音符（circumflex）——这些符号在约鲁巴语音系转写中已有使用先例——作为标准的单元音声调轮廓标记。

    arXiv:2607.18317v5 Announce Type: replace-cross  Abstract: We present TTSYoruba, a rule-based concatenative diphone speech synthesizer for Yoruba, deployed at online as part of the YorubaName.com open dictionary of Yoruba personal names. The system takes tone-marked Yoruba text as input and produces audio output by applying a hand-crafted phonological rule system to a recorded inventory of 651 diphone units spanning five tonal variants of every consonant-vowel combination in the language. We describe the phonological architecture of the system in detail, including our complete tonal file-selection logic, our treatment of the three-way nasal disambiguation problem (oral /n/, nasalized vowel, and syllabic nasal), and the derivation of contextual rising and falling tones from level-tone input. We also present, as an orthographic contribution, the adoption of the caron and circumflex, which are symbols with prior standing in Yoruba phonological transcription, as standard single-vowel conto
    
[^183]: DominoTree：基于多米诺的条件树状草稿生成用于推测解码

    DominoTree: Conditional Tree-Structured Drafting with Domino for Speculative Decoding

    [https://arxiv.org/abs/2607.08642](https://arxiv.org/abs/2607.08642)

    DominoTree通过将Domino的路径依赖条件校正应用于树状草稿生成，实现了无需训练的最高推测解码接受率。

    

    arXiv:2607.08642v3 公告类型：替换 摘要：推测解码通过草拟令牌并并行验证来加速大语言模型推理。块扩散草稿器（如DFlash）仅建模每个位置的边际分布，而树方法（如DDTree）则从这些边际分布扩展候选树。已发布的Domino草稿器引入了基于GRU的因果校正，使每个草稿令牌的分布具有路径依赖性，这是DDTree的因子化公式无法表示的。我们提出了DominoTree，一种无需训练的基于最佳优先的草稿树，通过Domino的条件（非因子化）校正沿每条根到节点路径进行评分，并通过将每节点校正限制为候选前M项来实现实用性。我们在单流测试框架中对八个基准进行了评估，并在SGLang中作为树外插件与AR、DFlash、EAGLE-3和Domino在相同标志下运行。DominoTree在每个服务单元中均达到最高平均接受长度——涵盖两种模型规模、单请求场景。

    arXiv:2607.08642v3 Announce Type: replace  Abstract: Speculative decoding accelerates LLM inference by drafting tokens and verifying them in parallel. Block-diffusion drafters such as DFlash model only per-position marginals, and tree methods such as DDTree expand candidate trees from those marginals. The released Domino drafter adds a GRU-based causal correction making each draft token's distribution path-dependent, a structure DDTree's factorized formulation cannot represent. We introduce DominoTree, a training-free best-first draft tree scored by Domino's conditional (non-factorized) correction along each root-to-node path, made practical by restricting the per-node correction to a candidate top-M. We evaluate it on eight benchmarks in a single-stream harness, and in SGLang, where it runs as an out-of-tree plugin against AR, DFlash, EAGLE-3 and Domino under identical flags. DominoTree attains the highest mean accepted length in every serving cell - two model sizes, single-request an
    
[^184]: PruneGround：面向3D视觉定位的即插即用空间剪枝框架

    PruneGround: Plug-and-play Spatial Pruning for 3D Visual Grounding

    [https://arxiv.org/abs/2606.31148](https://arxiv.org/abs/2606.31148)

    提出即插即用框架PruneGround，通过冻结视觉语言模型进行语言引导空间剪枝以缩小搜索范围，并将复杂指代表达分解为简化的目标描述，从而提升3D视觉定位精度并大幅降低计算成本。

    

    3D视觉定位（3DVG）旨在根据自然语言描述在3D场景中定位目标物体。现有方法通常对整个场景进行推理，导致预测结果模糊且计算成本高昂，尤其是在杂乱的环境中。我们观察到，许多指代表达依赖于局部空间上下文，并且往往对应于受限的空间区域而非整个场景。受这一洞察启发，我们提出了PruneGround，一个高效的即插即用3DVG框架，它建立在三个关键组件之上。首先，我们引入了语言引导空间剪枝（LGSP），利用冻结的视觉语言模型（VLM）识别与语言相关的区域，从而在更窄的搜索空间中减少空间计算和定位候选。其次，我们提出了多视角条件化描述重构（MCDR），将复杂表达分解为简化的目标中心描述……（摘要内容不完整）

    arXiv:2606.31148v2 Announce Type: replace-cross  Abstract: 3D Visual Grounding (3DVG) aims to localize target objects in 3D scenes given natural language descriptions. Existing approaches typically perform reasoning over the entire scene, leading to ambiguous predictions and high computational cost, especially in cluttered environments. We observe that many referential expressions rely on local spatial context and often correspond to restricted spatial regions rather than the full scene. Motivated by this insight, we propose PruneGround, an effective plug-and-play framework for 3DVG built upon three key components. First, we introduce Language-Guided Spatial Pruning (LGSP), which leverages a frozen Vision Language Model (VLM) to identify language-relevant regions, thereby reducing spatial computation and grounding candidates in the narrower search space. Second, we propose MultiView-Conditioned Description Reformulation (MCDR), which decomposes complex expressions into simplified targe
    
[^185]: 超越编译：评估忠实的自然语言到Lean语句形式化

    Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization

    [https://arxiv.org/abs/2606.31002](https://arxiv.org/abs/2606.31002)

    该论文提出将Lean编译与GPT-5.2和Gemini-2.5-Pro严格语义共识相结合的自动形式化评估标准，发现编译通过率会高估语义忠实度达3.0至29.0个百分点，且该标准与人类多数判断的一致率达89.7%。

    

    Lean能够验证生成的声明类型正确，但无法验证它表达了用户真正想要的语句。我们针对没有规范Lean目标的自动形式化研究两个问题：LLM评判者能否为人类语义审查提供可用的代理，以及编译在不同系统之间会在多大程度上高估忠实度。我们的评估标准将Lean编译与GPT-5.2和Gemini-2.5-Pro之间的严格语义共识相结合。在一个独立审计的随机样本上，该标准与人类多数意见在89.7%的案例中保持一致（Wilson 95%置信区间：82.1–94.3%）。在400个研究生水平语句上评估的八个系统中，每个系统都存在非零的编译-忠实度差距，其观测幅度介于3.0到29.0个百分点之间。完整的GPT-5.2工具增强代理表现出最大的差距，编译率达89.5%，而满足语义标准的仅为60.5%。人工审查、一个独立的第三族评判者以及BEq形式化交叉……

    arXiv:2606.31002v2 Announce Type: replace  Abstract: Lean verifies that a generated declaration is well typed, but not that it expresses the statement a user intended. We study two questions for autoformalization without canonical Lean targets: whether LLM judges can provide a usable proxy for human semantic review, and how much compilation overstates faithfulness across systems. Our criterion combines Lean compilation with strict semantic consensus between GPT-5.2 and Gemini-2.5-Pro. On an independently audited random sample, it agrees with human majority on 89.7\% of cases (Wilson 95\% CI: 82.1--94.3\%). Across eight systems evaluated on 400 graduate-level statements, every system has a nonzero compile--faithfulness gap, whose observed magnitude ranges from 3.0 to 29.0 percentage points. The full GPT-5.2 tool-augmented agent shows the largest gap, compiling 89.5\% while satisfying the semantic criterion on 60.5\%. Human review, an independent third-family judge, and a BEq formal cros
    
[^186]: 以真实意图铺就：意图感知训练提升大语言模型跨训练范式的安全分类能力

    Paved with True Intents: Intent-Aware Training Improves LLM Safety Classification Across Training Regimes

    [https://arxiv.org/abs/2606.27210](https://arxiv.org/abs/2606.27210)

    本文提出将用户意图作为显式信号融入安全分类器训练，通过构建AIMS数据集并验证意图感知训练在多种范式下的有效性，最终证明意图忠实建模是提升大语言模型安全分类性能的核心。

    

    我们认为，安全分类器应将用户意图建模为提示词与最终标签之间的显式信号。为此，我们引入了AIMS——一个包含1724个高难度安全提示的人工标注数据集，每个提示均配有意图描述和危害标签。我们利用AIMS在监督微调、偏好学习、推理蒸馏和强化学习等训练范式中评估了意图感知训练。尽管数据集规模有限，AIMS仍能使安全分类器在多种训练范式下具有竞争力：基于模型生成意图错误的直接偏好优化（DPO）优于监督微调（SFT）；在大多数师生对中，意图条件蒸馏优于纯推理蒸馏。最值得注意的是，通过GRPO直接奖励意图忠实度，在五个外部安全基准测试中取得了最佳平均性能，同时我们的意图感知模型构成了推理延迟-F1帕累托前沿。这些结果表明，忠实的意图建模是提升安全分类能力的关键组成部分。

    arXiv:2606.27210v1 Announce Type: new  Abstract: We argue that safety classifiers should model user intent as an explicit signal between the prompt and the final label. To study this, we introduce AIMS, a human-annotated dataset of 1,724 difficult safety prompts, each paired with an intent description and harm label. We use AIMS to evaluate intent-aware training across supervised fine-tuning, preference learning, reasoning distillation, and reinforcement learning. Despite its size, AIMS enables competitive safety classifiers across training regimes: DPO from model-generated intent errors improves over SFT, and intent-conditioned distillation outperforms reasoning-only distillation in most teacher-student pairs. Most notably, directly rewarding intent faithfulness with GRPO yields the strongest average performance across five external safety benchmarks, while our intent-aware models form the inference latency-F1 Pareto frontier. These results show that faithful intent modeling is a comp
    
[^187]: AgentOdyssey：面向测试时持续学习智能体的开放式长程文本游戏生成

    AgentOdyssey: Open-Ended Long-Horizon Text Game Generation for Test-Time Continual Learning Agents

    [https://arxiv.org/abs/2606.24893](https://arxiv.org/abs/2606.24893)

    AgentOdyssey是一个可程序化生成开放式长程文本游戏的评估框架，通过在部署全程中交替学习与推理的设定，从游戏进度、知识获取、情景记忆和探索能力等多维度评估测试时持续学习智能体的关键能力。

    

    为了让智能体能够在测试时通过与世界的交互进行持续学习，它们必须能够有效地探索、获取新的世界知识和技能、保留相关的情景经验，并进行长程规划。为了评估测试时持续学习智能体的这些关键能力，我们提出了AgentOdyssey，这是一个新颖的评估框架，能够程序化地生成包含丰富实体、世界动态和长程任务的开放式文本游戏。至关重要的是，AgentOdyssey突破了传统机器学习中“测试时不发生学习”的假设，将智能体置于一个持续的、长程的环境中，在整个部署过程中交替进行学习与推理。我们进一步提出了一种多层面的评估方法，不仅衡量游戏进度，还提供关于世界知识获取、情景记忆、对象与动作探索、动作多样性以及模型一致性等方面的诊断测试。

    arXiv:2606.24893v2 Announce Type: replace  Abstract: For agents to learn continuously from interaction with the world at test time, they must be able to explore effectively, acquire new world knowledge and skills, retain relevant episodic experiences, and plan over long horizons. To evaluate these key abilities of test-time continual learning agents, we introduce AgentOdyssey, a novel evaluation framework that procedurally generates open-ended text games with rich entities, world dynamics, and long-horizon tasks. Critically, AgentOdyssey goes beyond the conventional machine learning assumption that learning does not occur at test time by placing agents in a continuous, long-horizon setting that interleaves learning and inference throughout deployment. We further propose a multifaceted evaluation methodology that measures not only game progress but also offers diagnostic tests on world knowledge acquisition, episodic memory, object and action exploration, action diversity, and model cos
    
[^188]: 面向长时程智能体强化学习的群组-图策略优化

    Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning

    [https://arxiv.org/abs/2606.22995](https://arxiv.org/abs/2606.22995)

    提出群组-图策略优化（G2PO），一种针对多轮智能体任务的新型分组强化学习算法，通过显式利用状态转换固有的图结构，解决了长时程智能体强化学习中奖励稀疏延迟、高方差价值估计以及短视局部化信用分配的关键瓶颈。

    

    基于分组的强化学习已在智能体场景中显著增强了大语言模型（LLMs）的能力。为了实现更细粒度的策略更新，近期的智能体强化学习框架已从轨迹级训练转向步骤级训练。然而，长时程智能体强化学习面临着严重的奖励稀疏与延迟问题，因为反馈往往要推迟数十个交互步骤之后才会出现。尽管现有的步骤级框架细化了训练粒度，但其信用分配仍然是粗粒度的，并且依然将智能体的探索视为孤立的线性轨迹。这种过于简化的视角忽略了状态转换中固有的图结构，导致了高方差的状态价值估计以及短视的、局部化的信用分配。为了克服这些关键瓶颈，我们提出了群组-图策略优化（G2PO），这是一种专为多轮智能体任务量身定制的新型分组强化学习算法。G2PO 显式地将……（摘要在此处截断）

    arXiv:2606.22995v2 Announce Type: replace-cross  Abstract: Group-based Reinforcement Learning (RL) has significantly enhanced Large Language Models (LLMs) in agentic scenarios. To achieve finer-grained policy updates, recent agentic RL frameworks have shifted from trajectory-level to step-level training. However, long-horizon agentic RL suffers from severe reward sparsity and delay, as feedback is often deferred for dozens of interaction steps. While existing step-level frameworks refine training granularity, their credit assignment remains coarse-grained and still treats agent exploration as isolated, linear trajectories. This oversimplified perspective ignores the inherent graph structure of state transitions, leading to high-variance state-value estimation and myopic, localized credit assignment. To overcome these critical bottlenecks, we propose Group-Graph Policy Optimization (G2PO), a novel group-based RL algorithm tailored for multi-turn agentic tasks. G2PO explicitly transforms
    
[^189]: KaLM-Reranker-V1：面向压缩文档重排序的快速但非晚期交互方法

    KaLM-Reranker-V1: Fast but Not Late Interaction for Compressed Document Reranking

    [https://arxiv.org/abs/2606.22807](https://arxiv.org/abs/2606.22807)

    KaLM-Reranker-V1 是一种快速但非晚期交互的重排序器，通过编码器-解码器架构将查询与段落计算解耦，利用 Matryoshka 嵌入池化离线预编码段落，并借助交叉注意力捕捉细粒度相关性，在保持强表达能力的同时显著提升了部署效率与灵活性。

    

    随着检索系统规模的不断扩大，高效且有效的重排序变得日益重要。然而，大多数现有的基于编码器和解码器的重排序器都会联合处理每一个查询-段落对，将其在线计算紧密耦合，从而限制了部署的效率与灵活性。我们提出了 KaLM-Reranker-V1，一种快速但非晚期交互的 FBNL 重排序器，它在解耦查询与段落计算的同时，保留了富有表现力的相关性建模能力。KaLM-Reranker-V1 构建于编码器-解码器架构之上，利用 Matryoshka（套娃）嵌入池化对段落进行预编码，同时其解码器将系统指令、用户指令与查询意图共同建模；随后通过交叉注意力捕捉所得查询上下文与段落表示之间的细粒度相关性。这些设计共同带来了四大关键优势：(i) 离线段落编码带来的高效性，(ii) 交叉注意力带来的强表达能力，(iii) Matryoshka 嵌入带来的紧凑性

    arXiv:2606.22807v3 Announce Type: replace  Abstract: As retrieval systems scale, effective and efficient reranking becomes increasingly important. However, most existing encoder- and decoder-based rerankers jointly process every query--passage pair, tightly coupling their online computation and limiting deployment efficiency and flexibility. We present KaLM-Reranker-V1, a fast but not late-interaction FBNL reranker that decouples query and passage computation while retaining expressive relevance modeling. Built on an encoder--decoder architecture, KaLM-Reranker-V1 pre-encodes passages using Matryoshka embedding pooling, while its decoder models system and user instructions together with query intent; cross-attention then captures fine-grained relevance between the resulting query context and passage representations. Together, these designs offer four key advantages: (i) efficiency from offline passage encoding, (ii) expressiveness from cross-attention, (iii) compactness from Matryoshka
    
[^190]: 超越分词：面向时间序列问答的直接时间步嵌入与对比对齐方法

    Beyond Tokenization: Direct Timestep Embedding and Contrastive Alignment for Time-Series Question Answering

    [https://arxiv.org/abs/2606.18986](https://arxiv.org/abs/2606.18986)

    该论文提出CADE框架，通过直接时间步嵌入与对比语义对齐，克服大语言模型处理时间序列时的分词瓶颈，并解决了基于固定窗口的补丁编码所导致的粒度锁定和跨数据集迁移困难问题。

    

    大语言模型（LLM）的最新进展催生了时间序列问答（TSQA），该任务将时间序列分析表述为自然语言问答问题。然而，直接将原始数值序列输入大语言模型会遭遇分词瓶颈：字节对编码（BPE）将连续数值切分为不稳定的词元，其嵌入缺乏有意义的度量结构，导致幅值、尺度和趋势信息的丢失。先前的方法采用基于补丁的编码器将序列切分为固定窗口，从而锁定单一粒度，破坏了序列模式并隐藏了精确的时间步信息，且这种独立模块难以在不同长度或采样率的数据集之间迁移。为应对这一挑战，我们提出了CADE（对比对齐与直接嵌入），一种面向TSQA的新型框架，其建立在两个关键组件之上：直接时间步嵌入和语义对齐。

    arXiv:2606.18986v2 Announce Type: replace-cross  Abstract: Recent advances in large language models (LLMs) have given rise to time-series question answering (TSQA), which formulates time-series analysis as natural-language question answering. However, directly feeding raw numerical series into LLMs suffers from a tokenization bottleneck: Byte Pair Encoding fragments continuous values into unstable tokens whose embeddings lack meaningful metric structure, resulting in the loss of magnitude, scale, and trend information. Prior methods use patch-based encoders that split the series into fixed windows, locking in one granularity that breaks patterns and hides exact timesteps, through a separate module that rarely transfers across datasets with different lengths or sampling rates. To address this challenge, we propose CADE (Contrastive Alignment with Direct Embedding), a novel framework for TSQA built upon two key components: direct timestep embedding and semantic alignment. The proposed fr
    
[^191]: PragReST：面向语用语言理解的自增强反事实推理

    PragReST: Self-Reinforcing Counterfactual Reasoning for Pragmatic Language Understanding

    [https://arxiv.org/abs/2606.18624](https://arxiv.org/abs/2606.18624)

    PragReST是一个自监督框架，通过构建语用问答数据、生成反事实推理轨迹，并结合监督微调与强化学习来内化推理过程，无需人工标注数据或教师模型蒸馏，即可显著提升大语言模型的语用推理能力。

    

    自然语言理解通常依赖于隐含而非明确表达的含义，这需要进行语用推理。尽管大型语言模型（LLMs）在数学和逻辑推理方面表现强大，但它们在进行语用推理时仍然存在困难，往往倾向于选择字面解释。为了提升LLM的语用推理能力，我们提出了PragReST，这是一个自监督框架，它构建语用问答数据，生成反事实推理轨迹，并通过监督微调和强化学习训练模型将这些推理内化，全程无需人工标注的训练数据，也无需从更强的教师模型进行知识蒸馏。在四个语用基准（PragMega、Ludwig、MetoQA和AltPrag）上，PragReST优于骨干模型、任务特定的语用调优基线以及相同流程的非反事实变体。在基于准确率的基准上，PragReST相比指令骨干模型提升了5.37

    arXiv:2606.18624v2 Announce Type: replace  Abstract: Natural language understanding often depends on meanings that are implied rather than explicitly stated, requiring pragmatic reasoning. Despite strong performance on math and logical reasoning, large language models (LLMs) still struggle with making pragmatic inferences, often choosing literal interpretations. To improve LLM pragmatic reasoning, we introduce PragReST, a self-supervised framework that constructs pragmatic QA data, generates counterfactual reasoning traces, and trains models to internalize them through supervised fine-tuning and reinforcement learning, without human-labeled training data or distillation from a stronger teacher. Across four pragmatic benchmarks (PragMega, Ludwig, MetoQA, and AltPrag), PragReST improves over backbone models, task-specific pragmatic tuning baselines, and non-counterfactual variants of the same pipeline. On accuracy-based benchmarks, PragReST improves over the instruct backbone by 5.37 and
    
[^192]: TimpaTeks：基于扩散语言模型引导的自动原地文本序列修改

    TimpaTeks: Automatic In-place Text Sequence Modification via Diffusion Language Model Steering

    [https://arxiv.org/abs/2606.08408](https://arxiv.org/abs/2606.08408)

    TimpaTeks将激活引导扩展到扩散语言模型，实现了一种无需指令微调模型的自动原地文本修改机制，在保留原句结构、降低困惑度的同时，以更低的计算成本改变文本所表达的概念。

    

    我们将激活引导技术扩展到扩散语言模型，并研究了由扩散语言模型推理机制所引发的一个新问题：对文本进行原地修改以呈现不同的概念。我们提出了TimpaTeks，一种使用扩散语言模型的自动原地文本修改机制。在IMDB电影评论（情感概念）和一个合成的猫狗数据集（任意的、更非传统的概念引导）上的实验表明，TimpaTeks提供了一种可行的新型机制来原地引导扩散语言模型的输出。TimpaTeks无需依赖指令微调模型即可实现原地修改，同时能够降低句子困惑度并保留原始句子结构。此外，由于TimpaTeks执行的是原地去噪而非构建额外的提示条件输出序列，其计算成本也低于基于提示的扩散语言模型引导方法。

    arXiv:2606.08408v2 Announce Type: replace-cross  Abstract: We extend activation steering to diffusion language models (DLMs) and study a novel problem that arose due to the inference mechanism of DLMs: Modifying a text in-place to manifest a different concept. We propose TimpaTeks, an automatic in-place text modification mechanism using DLMs. Experiments on IMDB movie reviews (sentiment) and a synthetic Cats and Dogs Dataset (arbitrary, more unconventional concept steering) show that TimpaTeks provides a feasible novel mechanism to steer diffusion language model outputs in-place. TimpaTeks enables in-place modification while simultaneously lowers sentence perplexity and retaining the original sentence structre without the need of instruction tuned models. TimpaTeks is also computationally cheaper than prompt-based DLM steering, as it performs denoising in-place rather than constructing an additional prompt-conditioned output sequence.
    
[^193]: 你还会称它为Dax吗？视觉语言模型与人类的新颖视觉指称

    Would you still call this Dax? Novel Visual References in VLMs and Humans

    [https://arxiv.org/abs/2606.05409](https://arxiv.org/abs/2606.05409)

    该论文构建了完全从零设计的新颖视觉指称数据集NVRD（19,176张图像、90个视觉概念），通过人类与模型的直接对比，揭示视觉语言模型在面对与预训练先验知识相矛盾的新颖视觉概念时，难以像人类那样在上下文中习得这些概念。

    

    视觉-语言模型（VLM）与人类学习者一样，会频繁接触到新的视觉概念，但它们在接触之后如何将这些新颖的视觉指称映射到语言上，仍然在很大程度上未被充分探索，尤其是当这些指称与预训练所获得的先验知识相矛盾时。为了研究这一问题，我们提出了新颖视觉指称数据集（NVRD）：包含19,176张图像，涵盖90个具有不同视觉新颖程度的视觉概念，每个原始物体都配有最多20个扰动程度递增的版本，以探究泛化能力。与以往针对熟悉概念进行视觉增强的研究不同，NVRD完全由从零构建的、开放式的全新刺激组成，模拟了人类遇到真正新概念的方式。我们评估了3个开源模型和2个闭源模型，并结合2,400条人类判断进行直接的人机对比，发现：(i) 当新颖概念与模型先验知识相矛盾时，模型难以在上下文中习得这些概念……（摘要原文在此处截断）

    arXiv:2606.05409v4 Announce Type: replace-cross  Abstract: Vision-language models (VLMs), like human learners, are frequently exposed to new visual concepts, but how they map novel visual references to language after exposure remains largely underexplored, particularly when those references contradict prior knowledge from pre-training. To study this, we present the Novel Visual References Dataset (NVRD): 19,176 images spanning 90 visual concepts across different levels of visual novelty, each with up to 20 increasingly perturbed versions of the original object to probe generalization. Unlike prior work on visual augmentations of familiar concepts, NVRD comprises entirely novel, open-ended stimuli constructed from scratch, mirroring how humans encounter genuinely new concepts. We evaluate 3 open- and 2 closed-source models alongside 2,400 human judgments for direct human-model comparison, and find that (i) models struggle to acquire novel concepts in-context when they contradict prior k
    
[^194]: CauTion：知道何时信任大语言模型进行集成因果发现

    CauTion: Knowing When to Trust LLMs for Ensemble Causal Discovery

    [https://arxiv.org/abs/2606.03602](https://arxiv.org/abs/2606.03602)

    提出CauTion框架，通过共识过滤和LLM可靠性估计，将大语言模型的领域知识可靠地融合进统计因果发现算法集成中，从而同时克服纯统计方法、LLM错误风险和单一算法偏差这三方面的局限。

    

    从观测数据中进行因果发现仍然具有挑战性，原因在于纯统计方法存在根本性局限，例如等价类内部的统计不可区分性以及对有限样本量的敏感性。虽然大语言模型（LLM）为补充统计推断提供了有前景的领域知识来源，但现有的LLM增强方法容易受到LLM错误的影响，且产生高昂的token成本。此外，依赖单一以数据为中心的算法会使结果对算法特有的偏差敏感。为了解决这些局限性，我们提出了CauTion，一个通过共识过滤和LLM可靠性估计，将LLM领域知识可靠地集成到统计因果发现算法集成中的框架。CauTion分三个阶段进行。首先，算法集成利用共识投票解决多达96%的算法间达成一致的边，实现接近完美的（摘要在此处被截断）

    arXiv:2606.03602v2 Announce Type: replace-cross  Abstract: Causal discovery from observational data remains challenging due to the fundamental limitations of purely statistical methods, such as statistical distinguishability within equivalence classes and sensitivity to finite sample sizes. While large language models (LLMs) offer a promising source of domain knowledge to complement statistical inference, existing LLM-augmented methods are vulnerable to LLM errors and incur high token costs. Moreover, reliance on a single data-centric algorithm can make results sensitive to algorithm-specific biases. To address these limitations, we propose CauTion, a framework that reliably integrates LLM domain knowledge into an ensemble of statistical causal discovery algorithms through consensus filtering and LLM reliability estimation. CauTion proceeds in three stages. First, an algorithm ensemble utilizes a consensus voting to resolve up to 96% of edges on which algorithms agree, achieving near-p
    
[^195]: 学习对隐式多模态危害进行推理：从诊断到泛化

    Learning to Reason About Implicit Multimodal Harm: From Diagnosis to Generalization

    [https://arxiv.org/abs/2605.29951](https://arxiv.org/abs/2605.29951)

    提出MuPHI诊断基准，揭示视觉-语言模型虽能理解单一模态信息，却无法将图像-文本组合推理为危害机制，为隐式多模态危害检测从诊断走向泛化提供了方向。

    

    一张无害的图片配上无害的文字说明，只有在组合在一起时才可能变得有害，这使得此类检测在本质上是一个跨模态推理问题。现有的多模态危害基准仅评估分类准确率，尚不清楚模型是否真正识别了产生危害的跨模态机制。我们提出了MuPHI（多模态语用危害解释），这是一个受控诊断基准，其中危害纯粹源于图像与文本的组合，并配有良性反事实样本和经过标注的危害成因解释。借助MuPHI，我们系统地评估了视觉-语言模型（VLMs）中组合式危害推理的失效环节。MuPHI揭示：VLMs能够可靠地理解单一模态的信息，却无法将证据关联到危害机制之中，且其预测对决定标签的反事实干预响应微弱。此外，现有基于标签训练的危害检测方法虽能显著提升域内（ID）检测性能，但……

    arXiv:2605.29951v2 Announce Type: replace  Abstract: A benign image paired with a benign caption can become harmful only in combination, making detection fundamentally a cross-modal reasoning problem. Existing multimodal harm benchmarks evaluate only classification accuracy, leaving unclear whether models identify the cross-modal mechanism that produces harm. We introduce MuPHI (Multimodal Pragmatic Harm Interpretation), a controlled diagnostic benchmark in which harm arises purely from image-text composition, with benign counterfactuals and annotated harm rationales. Using MuPHI, we systematically evaluate where compositional harm reasoning breaks in vision-language models (VLMs). MuPHI reveals that VLMs reliably ground each modality yet fail to bind the evidence into the harm mechanism and their predictions respond weakly to label-determining counterfactual interventions. Moreover, existing harm detection approaches trained on labels substantially improves in-domain (ID) detection bu
    
[^196]: HyperPersona：面向基于文本的自动人格预测的多层超图框架

    HyperPersona: A Multi-Level Hypergraph Framework for Text-Based Automatic Personality Prediction

    [https://arxiv.org/abs/2605.17355](https://arxiv.org/abs/2605.17355)

    本文提出HyperPersona框架，利用多层超图结构显式建模文档、句子和词语三个层次的文本层级组织，从而提升基于文本的自动人格预测性能。

    

    作为一种现代商品，语言已成为一个蕴含丰富社会与心理学意义特征和概念的庞大宝库，反映了人们将思维、行为和情感模式编码为文字的方式。基于文本的自动人格预测（APP）旨在从语言行为中推断人格特质，为传统心理测量评估提供了一种可扩展的替代方案。尽管文本本质上具有层次性——文档层面捕捉全局特征，句子层面编码局部语义，词语层面提供细粒度的词汇信息——但大多数现有方法依赖于浅层、顺序式或单层的表示方式，忽略了书面语言的多层次结构。为解决这一问题，我们提出了HyperPersona，一个通过超图结构显式建模文本（文档、句子和词语）层次组织的框架，其中文档及其句子……

    arXiv:2605.17355v2 Announce Type: replace  Abstract: As a modern commodity, language has become a vast repository of socially and psychologically significant traits and concepts, reflecting the ways people encode pattern of thoughts, behaviors, and emotions into words. Text-based Automatic Personality Prediction (APP), seeks to infer personality from linguistic behavior, offering a scalable alternative to traditional psychometric assessments. Although text is inherently hierarchical, with the document-level capturing global features, the sentence-level encoding local semantics, and the word-level providing fine-grained lexical information, most existing approaches rely on shallow, sequential, or single-level representations that ignore the multi-level structure of written language. To address this, we propose HyperPersona, a framework that explicitly models the hierarchical organization of text (document, sentence, and word) through hypergraph structure, where a document and its senten
    
[^197]: 通过共识驱动的偏好优化缓解大语言模型中的跨语言文化不一致性

    Mitigating Cross-Lingual Cultural Inconsistencies in LLMs via Consensus-Driven Preference Optimisation

    [https://arxiv.org/abs/2605.12515](https://arxiv.org/abs/2605.12515)

    本文提出了对幻觉具有鲁棒性的跨语言文化不一致性度量指标κ_S，以及共识驱动的对齐框架C-3PO，能显著缓解多语言大模型在不同语言间回答文化歧义问题的不一致性，同时保留用户身份与文化知识。

    

    多语言大语言模型（MLLMs）在回答文化上具有歧义的问题时，常常在不同语言间给出不一致的答案，即使对于固定的用户人设也是如此——例如同一位英国用户，用英语提问会得到莎士比亚，而用西班牙语提问却得到塞万提斯。为了稳健地量化这种跨语言文化不一致性，我们引入了Singleton Fleiss's κ_S，这是一个在数学上对幻觉具有韧性的度量指标。作为缓解手段，我们提出了跨语言文化一致性偏好优化，这是一个共识驱动的对齐框架。C-3PO使κ_S相比未对齐模型提升了0.07–0.10，在跨人设平均的意义上超过了所有提示工程和表示引导的基线方法，同时大体上保留了明确的用户身份、文化中立性和内在文化知识；在未能保留之处，其软共识变体可以挽回大部分损失。尽管仅在多选题上进行训练，C-3PO还能提升开放式问题中的……（摘要截断于此）

    arXiv:2605.12515v3 Announce Type: replace  Abstract: Multilingual large language models (MLLMs) often answer culturally ambiguous questions inconsistently across languages, even for a fixed user persona -- e.g., Shakespeare in English but Cervantes in Spanish for the same British user. To robustly quantify this Cross-lingual Cultural Inconsistency, we introduce Singleton Fleiss's $\kappa_S$, a metric mathematically resilient to hallucinations. For mitigation, we propose Cross-lingual Cultural Consistent Preference Optimisation (C-3PO), a consensus-driven alignment framework. C-3PO raises $\kappa_S$ by 0.07--0.10 over unaligned models, above every prompting and representation steering baseline averaged across personas, whilst largely preserving explicit user identities, cultural neutrality and intrinsic cultural knowledge; where it does not, a soft-consensus variant recovers most of the cost. Although trained only on multiple-choice questions, C-3PO also improves consistency in open-end
    
[^198]: RubricEM：超越可验证奖励的量规引导策略分解元强化学习

    RubricEM: Meta-RL with Rubric-guided Policy Decomposition beyond Verifiable Rewards

    [https://arxiv.org/abs/2605.10899](https://arxiv.org/abs/2605.10899)

    RubricEM提出一个统一的强化学习框架，通过量规引导的阶段结构化策略分解实现细粒度信用分配，并结合基于反思的元强化学习实现经验复用，解决了深度研究智能体在长时域任务且缺乏精确答案验证器时的训练难题。

    

    深度研究智能体需要进行规划、搜索、评估证据并合成长篇报告。使用强化学习（RL）训练这类智能体极具挑战性，因为它们需要在长时域上运行，且其输出缺乏精确的答案验证器。为此，我们提出了RubricEM，一个面向深度研究的统一强化学习框架，它将分阶段信用分配与基于反思的元强化学习（用于经验复用）相结合。量规通过为策略执行、裁判反馈和经验复用提供共享接口，统一了整个框架。RubricEM首先通过监督微调灌输量规引导的脚手架，教会智能体使用量规进行规划，并将其轨迹组织为语义阶段。在此结构先验的基础上，我们使用阶段结构化GRPO优化任务策略，利用分阶段的裁判反馈实现更细粒度的信用分配。与此同时，我们异步训练一个具有共享主干的反思元策略（wi……

    arXiv:2605.10899v2 Announce Type: replace  Abstract: Deep research agents plan, search, evaluate evidence, and synthesize long-form reports. Training these agents with reinforcement learning (RL) is challenging because they operate over long horizons and their outputs lack exact answer verifiers. To this end, we introduce RubricEM, a unified RL framework for deep research that combines stagewise credit assignment with reflection-based meta-RL for experience reuse. Rubrics unify the framework by providing a shared interface for policy execution, judge feedback, and experience reuse. RubricEM first instills a rubric-guided scaffold through supervised fine-tuning, teaching the agent to plan with rubrics and organize its trajectory into semantic stages. Building on this structural prior, we optimize the task policy with Stage-Structured GRPO, using stagewise judge feedback for finer-grained credit assignment. In parallel, we asynchronously train a shared-backbone reflection meta-policy (wi
    
[^199]: 揭示视觉文本风格对大型视觉语言模型产生的基于属性描述的影响

    Revealing the Impact of Visual Text Style on Attribute-based Descriptions Produced by Large Visual Language Models

    [https://arxiv.org/abs/2604.27553](https://arxiv.org/abs/2604.27553)

    本文首次揭示了图像中文字的视觉呈现风格——功能性风格（如黑色无衬线字体）与装饰性风格（如彩色草书）——会影响大型视觉语言模型对相应概念进行属性描述的结果。

    

    当考虑文本的视觉风格时，可以在字体、颜色和大小等方面观察到广泛的多样性。然而，当一个词被阅读时，其含义与其书写或呈现的风格无关。在本文中，我们研究了一个词在图像中的视觉呈现风格是否以及如何影响大型视觉语言模型（LVLM）对该词所指概念的描述。具体而言，我们研究功能性文本风格（面向可读性，例如黑色无衬线字体）与装饰性风格（面向展示，例如彩色草书/手写体）如何在概念属性层面影响LVLM对概念的描述。我们的实验研究了LVLM能够正确识别视觉文本（即以图像形式呈现的一个或多个单词）所指概念的情形，即在这种情况下视觉文本风格不应影响基于属性的描述。

    arXiv:2604.27553v1 Announce Type: cross  Abstract: When the visual style of text is considered, a wide variety can be observed in font, color, and size. However, when a word is read, its meaning is independent of the style in which it has been written or rendered. In this paper, we investigate whether, and how, the style in which a word is visualized in an image impacts the description that a Large Visual Language Model (LVLM) provides for the concept to which that word refers. Specifically, we investigate how functional text styles (readability-oriented, e.g., black sans-serif) versus decorative styles (display-oriented, e.g., colored cursive/script) affect LVLMs' descriptions of a concept in terms of the attributes of that concept. Our experiments study the situation in which the LVLM is able to correctly identify the concept referred to by a visual text, i.e., by a word or words rendered as an image, and in which the visual text style should not influence the attribute-based descrip
    
[^200]: 变色龙的极限：研究大语言模型中的人格坍缩与同质化现象

    The Chameleon's Limit: Investigating Persona Collapse and Homogenization in Large Language Models

    [https://arxiv.org/abs/2604.24698](https://arxiv.org/abs/2604.24698)

    本文首次将大语言模型在多智能体模拟中响应趋于同质化的现象形式化为“人格坍缩”，提出覆盖度、均匀度、复杂度三个诊断指标，发现该现象普遍存在但因任务而异，且无法通过增加人格采样来缓解。

    

    大语言模型（LLM）已被用于驱动多智能体模拟，例如模拟问卷调查受访者或应用程序用户。然而，大语言模型缺乏这些模拟所需的人群级多样性——带有详细人类画像的LLM，其响应会聚集成少数几类，而不像人类响应那样均匀地填满整个空间。在本文中，我们将这一现象形式化为“人格坍缩”（Persona Collapse），并用三个诊断指标来量化其人群级结构：覆盖度（人格空间被占据了多少）、均匀度（智能体在空间中分布的均匀程度）和复杂度（由此产生的行为有多丰富）。在八个大语言模型上，针对人格（BFI-44）、道德推理和自我介绍任务的实验表明，人格坍缩是普遍存在的，但依赖于具体任务：CoSER-Llama-8B在人格任务上坍缩最严重，却在道德推理上最具多样性。此外，采样更多的人格画像无法缓解这种坍缩。我们进一步展示了保真度—多样性（摘要在此处截断）

    arXiv:2604.24698v2 Announce Type: replace  Abstract: Large language models (LLMs) have powered multi-agent simulations such as survey respondents or application users. However, LLMs lack the population-level diversity required by these simulations---responses from LLMs with detailed human profiles cluster into classes, unlike human responses that uniformly fill the entire space. In this paper, we formalize this as Persona Collapse and quantify its population-level structure with three diagnostic metrics: Coverage (how much of the persona space is occupied), Uniformity (how evenly agents spread across it), and Complexity (how rich the resulting behaviors are). Across eight LLMs on personality (BFI-44), moral reasoning, and self-introduction, persona collapse is universal but task-dependent: CoSER-Llama-8B collapses the most in personality while being the most diverse in moral reasoning. Moreover, sampling more personas cannot mitigate the collapse. We further show a fidelity--diversity 
    
[^201]: 心智理论基准测试的算法原语

    Algorithmic Primitives of Theory of Mind Benchmarks

    [https://arxiv.org/abs/2604.17819](https://arxiv.org/abs/2604.17819)

    该论文提出神经符号方法PDDL-Mind，通过将叙事解析为PDDL并由程序执行状态更新，揭示了主流心智理论基准实际上主要在测试解析和状态跟踪等算法能力而非真正的心智理论，并借此将GPT-4o的性能提升至接近人类水平。

    

    心智理论（ToM）基准测试真的在测试心智理论，还是实际上在测试解析和算法化状态跟踪？尽管各基准在模态和对话长度上有所差异，本文发现它们其实重新编码了相同的少数几种状态转移算法。我们提出了PDDL-Mind，一种神经符号方法，通过将叙事解析为规划领域定义语言（PDDL），并针对一组预定义的更新算法执行候选动作，从而显式地恢复这些算法以及ToM基准测试的上游归纳偏置。这样做能够为模型提供经程序验证的状态轨迹。我们利用该方法带来的性能提升来审计我们的归纳偏置究竟吸收了多少基准测试的难度。在MMToM、MuMA和FanToM三个基准上，仅将解析任务外包给程序就能使性能接近强大的智能体流水线，而将状态跟踪外包给PDDL则使GPT-4o的性能提升至接近人类水平。我们展示了……

    arXiv:2604.17819v2 Announce Type: replace-cross  Abstract: Do Theory of Mind (ToM) benchmarks really test theory of mind, or are they actually testing parsing and algorithmic state tracking? While benchmarks vary modality and dialogue length, this paper finds that they re-encode the same few state transition algorithms. We create PDDL-Mind, a neuro-symbolic approach that recovers such algorithms and upstream inductive biases of ToM benchmarks explicitly by parsing narratives into the Planning Domain Definition Language (PDDL) and executes candidate actions against a predefined set of update algorithms. Doing so provides program-verified state traces to the model. We use our method's performance gain to audit how much of a benchmark's difficulty our inductive bias absorbs. Across MMToM, MuMA, and FanToM, outsourcing parsing to programs alone brings performance close to strong agentic pipelines, while outsourcing state tracking to PDDL moves GPT-4o to close-to-human performance. We demon
    
[^202]: OmniTrace：面向全模态大语言模型的生成时溯源统一框架

    OmniTrace: A Unified Framework for Generation-Time Source Attribution in Omni-Modal LLMs

    [https://arxiv.org/abs/2604.13073](https://arxiv.org/abs/2604.13073)

    OmniTrace 提出一种统一框架，通过读取全模态大模型后层的注意力分布，将生成的每个句子实时归因到文本、图像和音视频片段等源单元，无需额外前向传播或针对模态的调优。

    

    全模态语言模型可以回答关于多模态文档、录音或视频的问题，但它无法展示答案来自输入的哪个部分。让模型引用其证据只能得到它声称使用的内容；而要测量它实际使用的内容，迄今为止需要通过删除部分输入并多次重新运行模型来实现，其成本随音频和视频时长的增加而不断增长。我们证明生成过程本身已经包含了大部分此类信息。OmniTrace 通过将模型后层的注意力读作各源单元上的质量分布，将每个生成的句子归因到源单元（文本块、图像以及音频或视频的秒级片段），无需额外的前向传播，也无需针对各模态进行调优；在可获得消融数据的情况下，还可以训练注意力头权重。在固定的因果测试设置下，OmniTrace 所选出的单元在 Qwen3-Omni-3 的视频、音频和图文问答任务中，对整体响应以及单个句子都具有因果重要性。

    arXiv:2604.13073v2 Announce Type: replace-cross  Abstract: An omni-modal language model can answer a question about a multimodal document, a recording, or a video, but it cannot show which part of the input the answer came from. Asking the model to cite its evidence yields what it says it used; measuring what it used has so far required rerunning it many times with parts of the input removed, a cost that grows with every second of audio and video. We show that the generation pass already holds most of this information. OmniTrace attributes each generated sentence to source units (text chunks, images, and seconds of audio or video) by reading late-layer attention as mass over units with no extra forward pass and no per-modality tuning; head weights can be trained where ablation data are available. With fixed causal test setting, the units OmniTrace selects are causally important for responses and for individual sentences in video, audio, and image-text question answering on Qwen3-Omni-3
    
[^203]: 迈向大语言模型中可靠的真实对齐不确定性估计

    Towards Reliable Truth-Aligned Uncertainty Estimation in Large Language Models

    [https://arxiv.org/abs/2604.00445](https://arxiv.org/abs/2604.00445)

    提出真实锚定方法，一种轻量级后验校准技术，通过将大语言模型的原始不确定性估计分数映射为真实对齐的正确性概率，解决了UE指标在不同模型和数据集上性能不稳定的“代理失效”问题，即使在有噪声的少样本监督下也能持续提升校准效果。

    

    不确定性估计（UE）旨在量化大语言模型（LLM）输出中的不确定性，以标记潜在的幻觉并支持可靠部署。然而，UE指标在不同模型和数据集上往往表现出不稳定的性能，限制了其实际效用。我们将这一现象形式化为“代理失效”（proxy failure），其产生的原因是UE指标源自模型行为，而非明确地基于事实正确性。我们证明，当UE分数与正确性之间的互信息较低时，该分数的判别能力必然接近随机水平。为解决这一问题，我们提出了真实锚定（Truth AnChoring, TAC），这是一种轻量级的后验校准方法，可将原始UE分数映射为与真实对齐的正确性概率。即使在有噪声的少样本监督条件下，TAC仍能持续改善校准效果，提供了一种实用的校准协议。我们的发现强调了将启发式UE指标直接视为（摘要在此处截断）

    arXiv:2604.00445v2 Announce Type: replace  Abstract: Uncertainty estimation (UE) aims to quantify uncertainty in large language model (LLM) outputs to flag potential hallucinations and support reliable deployment. However, UE metrics often exhibit unstable performance across models and datasets, limiting their practical utility. We formalise this phenomenon as proxy failure, arising because UE metrics are derived from model behaviour rather than being explicitly grounded in factual correctness. We show that when mutual information between a UE score and correctness is low, the score's discriminability must be close to chance. To address this, we propose Truth AnChoring (TAC), a lightweight post-hoc calibration method that maps raw UE scores to truth-aligned correctness probabilities. Even with noisy and few-shot supervision, TAC continues to improve calibration, providing a practical calibration protocol. Our findings highlight the limitations of treating heuristic UE metrics as direct
    
[^204]: 语言模型在自主学习任务中的目标选择与人类不同

    Language Model Goal Selection Differs from Humans' in a Self-Directed Learning Task

    [https://arxiv.org/abs/2603.03295](https://arxiv.org/abs/2603.03295)

    本研究借助认知科学中的自主学习任务，发现五大主流大语言模型的目标选择行为与人类存在显著差异——人类倾向于渐进探索且个体多样，而模型要么固守单一解要么表现不佳，表明LLM尚不能可靠地替代人类进行目标设定。

    

    无论是在智能体工作流、社会研究还是聊天场景中，大语言模型（LLM）越来越多地被要求代替人类选择要追求的目标，而不是完成预先定义的任务。然而，LLM能够准确反映人类目标设定偏好的这一假设在很大程度上仍未得到检验。我们在一项借鉴自认知科学的受控自主学习任务中，评估了LLM作为人类目标选择代理的有效性。在五个模型（GPT-5、Gemini 2.5 Pro、Claude Sonnet 4.5、Qwen3 32B和Centaur）上，我们发现模型行为与人类行为存在显著差异。人类会逐渐探索并学会实现目标，且个体间表现出多样性；而大多数模型要么固守单一已发现的解决方案，要么表现出出人意料的低性能，不同模型之间存在截然不同的模式，而同一模型的不同实例之间几乎没有变异性。思维链推理和角色引导……

    arXiv:2603.03295v3 Announce Type: replace-cross  Abstract: Whether in agentic workflows, social studies, or chat settings, large language models (LLMs) are increasingly being asked to replace humans in choosing which goals to pursue, rather than completing predefined tasks. However, the assumption that LLMs accurately reflect human preferences for goal setting remains largely untested. We assess the validity of LLMs as proxies for human goal selection in a controlled, self-directed learning task borrowed from cognitive science. Across five models (GPT-5, Gemini 2.5 Pro, Claude Sonnet 4.5, Qwen3 32B, and Centaur), we find substantial divergence from human behavior. While people gradually explore and learn to achieve goals with diversity across individuals, most models exploit a single identified solution or show surprisingly low performance, with distinct patterns across models and little variability across instances of the same model. Chain-of-thought reasoning and persona steering pro
    
[^205]: 通过奖励引导拼接实现扩散语言模型的测试时扩展

    Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching

    [https://arxiv.org/abs/2602.22871](https://arxiv.org/abs/2602.22871)

    该论文提出Stitching Noisy Diffusion Thoughts框架，利用过程奖励模型对扩散语言模型采样的多条推理轨迹的中间步骤进行评分，并将跨轨迹的最优步骤拼接成复合推理链，实现了步骤级而非轨迹级的中间推理成果复用与测试时扩展。

    

    使用大型语言模型进行推理通常受益于生成多条思维链，但现有的聚合策略通常是轨迹层面的（例如，选择最佳轨迹或对最终答案进行投票），这会丢弃来自部分完成或“接近正确”尝试中的有用中间工作。我们提出了Stitching Noisy Diffusion Thoughts（拼接噪声扩散思维），这是一个自洽性框架，能将廉价的扩散采样推理转化为可复用的步骤级候选池。给定一个问题，我们（i）使用掩码扩散语言模型采样多条多样化、低成本的推理轨迹，（ii）使用现成的过程奖励模型（PRM）为每个中间步骤评分，（iii）将跨轨迹的最高质量步骤拼接成一个复合推理链，然后仅使用该推理链重新计算最终答案。这种模块化流程将探索（扩散）与评估和解决方案合成相分离，避免了……

    arXiv:2602.22871v2 Announce Type: replace-cross  Abstract: Reasoning with large language models often benefits from generating multiple chains-of-thought, but existing aggregation strategies are typically trajectory-level (e.g., selecting the best trace or voting on the final answer), discarding useful intermediate work from partial or "nearly correct" attempts. We propose Stitching Noisy Diffusion Thoughts, a self-consistency framework that turns cheap diffusion-sampled reasoning into a reusable pool of step-level candidates. Given a problem, we (i) sample many diverse, low-cost reasoning trajectories using a masked diffusion language model, (ii) score every intermediate step with an off-the-shelf process reward model (PRM), and (iii) stitch these highest-quality steps across trajectories into a composite rationale. This rationale is then used to recompute only the final answer. This modular pipeline separates exploration (diffusion) from evaluation and solution synthesis, avoiding mo
    
[^206]: AutoPrunedRetriever：面向高效且可扩展检索增强生成的持久化推理图

    AutoPrunedRetriever: Persistent Reasoning Graphs for Efficient and Extensible Retrieval-Augmented Generation

    [https://arxiv.org/abs/2602.04926](https://arxiv.org/abs/2602.04926)

    提出了AutoPrunedRetriever（APR），一种基于图的RAG系统，通过共享符号码本将语料库事实与查询推理过程编码为既充当检索索引又充当持久化记忆的紧凑表示，使历史证据和推理结果可被高效复用，从而提升复杂推理任务下的效率与可扩展性。

    

    检索增强生成（RAG）已成为知识密集型大语言模型任务的标准方法。复杂推理仍然具有挑战性，因为查询往往需要通过多个相互依赖的推理步骤来组合分散的证据。然而，现有的RAG系统独立地处理每个查询，反复检索重叠的证据并重建已经完成的推理。若能保留这些推理过程，先前检索到的证据和中间结果就可以在后续查询中被复用。但随着历史记录的增长，推理时的复用成本会不断上升，这就需要一种紧凑的表示方式，在不显著损失质量的前提下保留有用信息。为应对这些挑战，我们提出了AutoPrunedRetriever（APR），这是一种基于图的RAG系统，它将语料库事实和查询衍生的推理编码到一个共享的符号码本（symbolic codebook）中。该码本既充当检索索引，又充当持久化记忆。

    arXiv:2602.04926v2 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) has become a standard approach for knowledge-intensive LLM tasks. Complex reasoning remains challenging because queries often require composing dispersed evidence through multiple dependent reasoning steps. Yet existing RAG systems process queries independently, repeatedly retrieving overlapping evidence and reconstructing established reasoning. Preserving such reasoning makes previously retrieved evidence and intermediate results reusable for subsequent queries. However, inference-time reuse becomes increasingly costly as the history grows, motivating a compact representation that preserves useful information without substantial loss in quality. To address these challenges, we present AutoPrunedRetriever (APR), a graph-based RAG system that encodes corpus facts and query-derived reasoning in a shared symbolic codebook. The codebook serves both as the retrieval index and as persistent memory
    
[^207]: 大语言模型以有向无环图进行思考：探查推理的潜在图结构

    LLMs Think with DAGs: Probing the Latent Graph Structure of Reasoning

    [https://arxiv.org/abs/2601.17593](https://arxiv.org/abs/2601.17593)

    本文提出“推理DAG探查”方法，首次证实大语言模型的内部表征确实编码了推理过程的有向无环图（DAG）结构，该结构在中间层最强、随模型规模增大而增强，并在自回归生成过程中逐步涌现。

    

    大语言模型通常将推理外化为线性链条，但许多推理问题可以更自然地建模为有向无环图（DAG），其中间结论会分支、合并并被复用。这种图结构是否反映在模型内部仍不清楚。我们提出“推理DAG探查”方法来回答这个问题。我们将每个前提、中间结论和最终答案形式化为DAG节点，并训练轻量级探针从大语言模型的隐藏状态中预测节点深度、节点对距离和邻接关系。在逻辑、数学和代码推理基准测试中，我们发现DAG结构在大语言模型的表征中得到了有意义的编码：这种结构在中间层达到峰值，较深的层优先恢复更深的节点和更长程的推理依赖，且随模型规模增大而变得更强。DAG结构在自回归生成过程中也逐步显现，其方式与……

    arXiv:2601.17593v4 Announce Type: replace  Abstract: Large language models often externalize reasoning as linear chains, but many reasoning problems can be more naturally modeled as directed acyclic graphs (DAGs), where intermediate conclusions branch, merge, and are reused. Whether such graph structure is reflected in model internals remains unclear. We introduce Reasoning DAG Probing to answer this question. We formalize each premise, intermediate conclusion, and final answer as a DAG node, and train lightweight probes to predict node depth, pairwise distance, and adjacency from LLM hidden states. Across logical, mathematical, and code reasoning benchmarks, we find that DAG structure is meaningfully encoded in LLM representations: it peaks in intermediate layers, with later layers preferentially recover deeper nodes and longer-range reasoning dependencies, and becomes stronger with model scale. DAG structure also progressively emerges during autoregressive generation in ways that ant
    
[^208]: 大型语言模型中的“亲AI”偏向

    Pro-AI Bias in Large Language Models

    [https://arxiv.org/abs/2601.13749](https://arxiv.org/abs/2601.13749)

    该研究发现大型语言模型存在“亲AI”偏向：它们在建议中更频繁且更靠前地推荐AI，会将AI相关职位的薪资高估约10%，并对AI专业证书授予数量的高估程度比匹配的计算机专业高出28%。

    

    大型语言模型（LLMs）在多个领域的决策支持中正得到越来越广泛的应用。我们探究的问题是：这些模型是否偏爱人工智能（AI）本身。我们通过三个实验来检验这一问题。首先，在投资、学习、职业和创业等领域的开放式建议中，AI在大多数推荐列表中都占据显著位置，并且一旦出现通常排名第一。其次，在薪资估算方面，模型对AI相关职位薪资的高估程度高于与之高度匹配的非AI职位。在一项受控的后续实验中，在所有其他职位信息保持不变的情况下，仅将职位名称从匹配的非AI名称改为AI相关名称，就会使预测薪资提高10%。最后，我们证明，AI专业所授予的学位及其他证书数量被高估的程度比与之匹配的计算机专业相应证书高出28%。总体而言，我们的研究表明，AI不仅在日常生活中……

    arXiv:2601.13749v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly employed for decision-support across multiple domains. We ask whether these models favor artificial intelligence (AI) itself. We examine this question through three experiments. First, in open-ended advice across investment, study, career, and startup domains, AI is prominent in most recommendation lists and is usually ranked first when it appears. Second, in salary estimations, models overestimate AI-related jobs more than they do for closely matched non-AI jobs. In a controlled follow-up, changing only the job title from the matched non-AI title to the AI-related title raises predicted salary by 10%, while all other supplied job details remain fixed. Finally, we show that the number of degrees and other credentials awarded by AI programs were 28% more overestimated than corresponding awards for matched computing programs. Overall, we show that AI is not only prominent in everyday 
    
[^209]: 机器翻译准备好应对下一场危机或大流行病了吗？

    Is MT Ready for the Next Crisis or Pandemic?

    [https://arxiv.org/abs/2601.10082](https://arxiv.org/abs/2601.10082)

    本研究基于 TICO-19 数据集评估了 Google Translate、Microsoft Translator、GPT、Gemini 和 Claude 五个商业机器翻译系统在危机与医疗领域翻译低资源语言的能力，以此衡量机器翻译应对下一场大流行病的就绪程度。

    

    危机时期的沟通至关重要。然而，政府、援助提供者、医生与他们所援助的对象之间常常存在语言不匹配的问题。在这些场景下，求助于商业机器翻译（MT）工具是合理的。但是，这些工具在翻译低资源语言方面——尤其是在危机或医疗领域——效果究竟如何？在本研究中，我们基于 TICO-19 数据集评估了 5 个商业机器翻译系统（Google Translate、Microsoft Translator、GPT、Gemini 和 Claude）。该数据集包含与流行病相关的句子，涵盖大量高优先级语言，这些语言的使用群体最有可能在下一场大流行病中受到不利影响。我们根据输出译文的可用性，评估了当前针对下一场大流行病的“就绪”程度。

    arXiv:2601.10082v2 Announce Type: replace  Abstract: Communication in times of crisis is essential. However, there is often a mismatch between the language of governments, aid providers, doctors, and those to whom they are providing aid. It is reasonable to turn to commercial MT tools in these scenarios. But how effective are these tools for translating to and from low resource languages, particularly in the crisis or medical domain? In this study, we evaluate 5 commercial systems used for MT (Google Translate, Microsoft Translator, GPT, Gemini and Claude) against the TICO-19 dataset, which contains pandemic-related sentences from a large set of high-priority languages spoken by communities most likely to be adversely affected in the next pandemic. We assess the current degree of "readiness" for this next pandemic based on the usability of the output translations.
    
[^210]: 超越共识：自然语言处理中标注者分歧的视角主义建模与评估

    Beyond Consensus: Perspectivist Modeling and Evaluation of Annotator Disagreement in NLP

    [https://arxiv.org/abs/2601.09065](https://arxiv.org/abs/2601.09065)

    本综述系统梳理了NLP中标注者分歧的来源分类、建模框架与评估指标，指出该领域正从共识学习转向显式建模分歧及标注者间的结构化关系，并发现现有公平性评估仍停留在描述性层面。

    

    标注者分歧在自然语言处理（NLP）中普遍存在，尤其是在毒 toxicity 检测和立场分析等主观且模糊的任务中。早期的方法将分歧视为需要去除的噪声，而近期的研究越来越多地将其建模为一种反映解释与视角差异的有意义信号。本综述对分歧感知的NLP方法提供了一个统一的视角。我们首先提出了一个与领域无关的分歧来源分类体系，涵盖数据、任务和标注者三方面因素。随后，我们使用由预测目标和聚合结构定义的通用框架对各类建模方法进行综合梳理，突出了该领域从共识学习向显式建模分歧、以及向捕捉标注者之间结构化关系的转变。我们回顾了针对预测性能和标注者行为的评估指标，并指出大多数公平性评估仍然是描述性的而非规范性的。

    arXiv:2601.09065v3 Announce Type: replace  Abstract: Annotator disagreement is widespread in NLP, particularly for subjective and ambiguous tasks such as toxicity detection and stance analysis. While early approaches treated disagreement as noise to be removed, recent work increasingly models it as a meaningful signal reflecting variation in interpretation and perspective. This survey provides a unified view of disagreement-aware NLP methods. We first present a domain-agnostic taxonomy of the sources of disagreement spanning data, task, and annotator factors. We then synthesize modeling approaches using a common framework defined by prediction targets and pooling structure, highlighting a shift from consensus learning toward explicitly modeling disagreement, and toward capturing structured relationships among annotators. We review evaluation metrics for both predictive performance and annotator behavior, and noting that most fairness evaluations remain descriptive rather than normative
    
[^211]: 迈向以人为中心的AI辅助术语工作

    Toward Human-Centered AI-Assisted Terminology Work

    [https://arxiv.org/abs/2512.18859](https://arxiv.org/abs/2512.18859)

    本文提出应以“以人为中心的AI”框架指导术语工作，在实现高水平自动化的同时保留有意义的人类控制，因为大型语言模型存在错误、幻觉和偏见，术语学家对确保术语数据的准确可靠仍然不可或缺。

    

    生成式AI很可能通过创造新的自动化机会来变革术语工作。与此同时，它也引发了对术语学家和术语资源未来的担忧，因为效率压力可能促使人们基于“人类专业知识可以被AI取代”的认知而走向过度自动化。然而，由于错误、幻觉和各种形式的偏见，大型语言模型在术语学应用中仍然不可靠，这使得术语学家在确保术语数据的准确性和可靠性方面不可或缺。本文认为，以人为中心的AI——一种强调AI的首要目标应是促进人类福祉的方法——为在最大化生成式AI益处的同时减轻其风险提供了一个框架。文章主张，高水平的自动化与有意义的人类控制是兼容且值得追求的，AI应当增强术语学家的能力，而非取代他们。

    arXiv:2512.18859v4 Announce Type: replace  Abstract: Generative AI is likely to transform terminology work by creating new opportunities for automation. At the same time, it raises concerns about the future of terminologists and terminological resources, as efficiency pressures may encourage excessive automation based on the perception that human expertise can be replaced by AI. However, large language models remain unreliable for terminological purposes due to errors, hallucinations, and various forms of bias, making terminologists indispensable for ensuring the accuracy and reliability of terminological data. This paper argues that human-centered AI, an approach that emphasizes that AI's primary goal should be to contribute to human well-being, provides a framework for maximizing the benefits of generative AI while mitigating its risks. It contends that high levels of automation and meaningful human control are compatible and desirable, and that AI should enhance terminologists' capa
    
[^212]: Rep2Text：从单个LLM词元表示中解码完整文本

    Rep2Text: Decoding Full Text from a Single LLM Token Representation

    [https://arxiv.org/abs/2511.06571](https://arxiv.org/abs/2511.06571)

    提出Rep2Text方法，证明仅凭LLM的单个隐藏表示即可重建原始文本（短文本可恢复约一半单词及整体含义），揭示了隐藏表示蕴含的丰富信息及其潜在的隐私泄露风险。

    

    大型语言模型将文本表示为隐藏的数值模式，但这些表示实际包含什么信息仍然是一个谜。我们能否仅凭其中一个表示恢复出原始文本？在本文中，我们提出了Rep2Text，这是一种旨在通过从单个隐藏表示重建文本来回答这一问题的方法。我们发现这些表示保留的信息远比我们预期的要多。对于约16个单词的短文本，Rep2Text能够恢复大约一半的原始单词，同时保留其整体含义。随着文本变长，恢复确切的措辞变得更加困难，但主要思想仍然可以被恢复。这些发现为理解语言模型内部存储了什么提供了新的途径，并凸显了潜在的隐私风险，因为隐藏表示可能会泄露原始输入的部分内容。

    arXiv:2511.06571v4 Announce Type: replace-cross  Abstract: Large language models represent text as hidden numerical patterns, but what information these representations actually contain remains a mystery. Can we recover the original text from just one of them? In this paper, we introduce Rep2Text, a method designed to answer this question by reconstructing text from a single hidden representation. We find that these representations preserve much more information than we might expect. For short texts of about 16 words, Rep2Text can recover roughly half of the original words while preserving their overall meaning. As the text gets longer, recovering the exact wording becomes harder, but the main ideas can still be recovered. These findings offer a new way to understand what language models store internally and highlight potential privacy risks, as hidden representations may reveal parts of the original input.
    
[^213]: 在潜空间中进行语言引导以缓解无意的语码切换

    Language steering in latent space to mitigate unintended code-switching

    [https://arxiv.org/abs/2510.13849](https://arxiv.org/abs/2510.13849)

    该论文提出一种通过对平行翻译做主成分分析来识别语言方向、并在潜空间中引导词元嵌入的轻量级推理时方法，可将多语言大模型的无意语码切换指数降低63%至99%，且几乎不增加计算开销、仅需极少平行数据校准。

    

    多语言大语言模型（LLM）经常表现出诸如无意语码切换之类的幻觉现象，降低了下游任务的可靠性。我们提出了潜空间语言引导方法，这是一种轻量级的推理时方法，通过对平行翻译进行主成分分析（PCA）来识别语言方向，并沿这些方向轴引导词元嵌入，从而控制语言身份。我们的方法缓解了语码切换问题（下一词元分布散度降低高达55%，语码切换指数降低63%至99%），同时保持了广泛的语义方向，计算开销几乎可以忽略不计，并且只需要极少的平行数据进行校准。此外，我们还使用单个主成分进行了语言分类评估（准确率达95-99%），并进一步分析了语言表示在各层中的演化过程。

    arXiv:2510.13849v4 Announce Type: replace  Abstract: Multilingual Large Language Models (LLMs) often exhibit hallucinations such as unintended code-switching, reducing reliability in downstream tasks. We propose latent-space language steering, a lightweight inference-time method that identifies language directions via Principal Component Analysis (PCA) on parallel translations and steers token embeddings along these axes to control language identity. Our approach mitigates code-switching (reduced next-token distributional divergence by up to 55\%, reduced Code-Switching Index by 63--99\%) while preserving broad semantic direction with negligible computational overhead and requires only minimal parallel data for calibration. Additionally, we evaluate language classification using a single principal component (95-99\% accuracy) and further analyze the layer-wise evolution of language representations.
    
[^214]: 采样、对齐、合成：基于ConGrs的图式响应合成方法

    Sample, Align, Synthesize: Graph-Based Response Synthesis with ConGrs

    [https://arxiv.org/abs/2510.03527](https://arxiv.org/abs/2510.03527)

    提出一种基于有向无环图的一致性图（ConGrs）数据结构，利用序列对齐算法整合多次采样的语言模型响应并从中合成最终答案，使传记生成任务的事实精确度提升最多31%，同时减少80%以上对语言模型评判者的依赖。

    

    语言模型可以通过多次采样来获取其响应背后的分布，但现有方法无法高效地合成不同长篇响应中蕴含的丰富认知信号。我们提出了一致性图（ConGrs），这是一种灵活的基于有向无环图（DAG）的数据结构，用于表示对同一提示采样的语言模型响应集合中的共享信息以及语义差异。我们采用一种源自生物信息学的轻量级词汇序列对齐算法来构建ConGrs，并辅以有针对性地使用次要语言模型评判者。此外，我们设计了依赖于任务的解码方法，从ConGr数据结构中合成出单一的最终响应。实验表明，从ConGrs合成响应在两个传记生成任务上将事实精确度相比平均响应提升了最多31%，并与其他方法相比减少了对语言模型评判者的依赖超过80%。我们还将ConGrs用于……（原文在此处截断）

    arXiv:2510.03527v2 Announce Type: replace  Abstract: Language models can be sampled multiple times to access the distribution underlying their responses, but existing methods cannot efficiently synthesize rich epistemic signals across different long-form responses. We introduce Consensus Graphs (ConGrs), a flexible DAG-based data structure that represents shared information, as well as semantic variation in a set of sampled LM responses to the same prompt. We construct ConGrs using a light-weight lexical sequence alignment algorithm from bioinformatics, supplemented by the targeted usage of a secondary LM judge. Further, we design task-dependent decoding methods to synthesize a single, final response from our ConGr data structure. Our experiments show that synthesizing responses from ConGrs improves factual precision on two biography generation tasks by up to 31% over an average response and reduces reliance on LM judges by more than 80% compared to other methods. We also use ConGrs fo
    
[^215]: 度量文本中的AI“垃圾内容”

    Measuring AI "Slop" in Text

    [https://arxiv.org/abs/2509.19163](https://arxiv.org/abs/2509.19163)

    本研究通过对NLP、写作和哲学领域专家的访谈，构建了AI“垃圾内容”的分类体系和可解释的评估维度，发现其二元判断虽带有主观性但与连贯性、相关性等潜在维度相关，为AI生成文本的质量评估提供了框架。

    

    AI“垃圾内容”是一个日益流行的术语，用于描述低质量的AI生成文本，但目前对这一术语尚无公认的定义，也没有衡量其出现的方法。在本研究中，我们通过对自然语言处理、写作和哲学领域专家的访谈，构建了“垃圾内容”的分类体系，并提出了一套可解释的维度用于在文本中对其进行评估。通过片段级标注，我们发现二元式的“垃圾内容”判断具有一定的主观性，但这种判断仍然与连贯性和相关性等潜在维度相关。我们的框架可用于在检测任务和二元偏好任务中考察AI生成的文本，有望为影响质量判断的语言与风格因素提供新的见解。我们同时指出，完全自动化且可扩展的度量方法仍然是一个悬而未决的挑战。

    arXiv:2509.19163v3 Announce Type: replace  Abstract: AI "slop" is an increasingly popular term used to describe low-quality AI-generated text, but there is currently no agreed upon definition of this term nor a means to measure its occurrence. In this work, we develop a taxonomy of "slop" through interviews with experts in NLP, writing, and philosophy, and propose a set of interpretable dimensions for its assessment in text. Through span-level annotation, we find that binary "slop" judgments are (somewhat) subjective, but such determinations nonetheless correlate with latent dimensions such as coherence and relevance. Our framework can be used to examine AI-generated text in both detection and binary preference tasks, potentially offering new insights into the linguistic and stylistic factors that contribute to quality judgments. We highlight that fully automated and scalable methods remain an open challenge.
    
[^216]: 面向细粒度幻觉检测的跨层注意力探测方法

    Cross-Layer Attention Probing for Fine-Grained Hallucination Detection

    [https://arxiv.org/abs/2509.09700](https://arxiv.org/abs/2509.09700)

    本文提出跨层注意力探测技术CLAP，将大语言模型整个残差流的激活作为联合序列处理，实现了细粒度的幻觉检测，并基于此提出“先检测后缓解”策略以减少幻觉、提升模型可靠性。

    

    随着大语言模型（LLMs）在各类应用中的大规模采用，由于其倾向于生成不准确的文本（即幻觉），人们对其可靠性的担忧日益增加。在这项工作中，我们提出了跨层注意力探测，这是一种用于幻觉检测的新型激活探测技术，它将LLM在整个残差流中的激活作为一个联合序列进行处理。我们使用五个大语言模型和三个任务进行的实证评估表明，CLAP在贪婪解码的响应以及较高温度下采样的响应上，相比基线方法均提升了幻觉检测性能，从而实现了细粒度检测，即能够对给定提示词的多个不同采样响应中的幻觉与非幻觉进行区分。这使我们能够提出一种基于CLAP的“先检测后缓解”策略，以减少幻觉并提升大语言模型的可靠性，其效果优于直接缓解方法。

    arXiv:2509.09700v2 Announce Type: replace-cross  Abstract: With the large-scale adoption of Large Language Models (LLMs) in various applications, there is a growing reliability concern due to their tendency to generate inaccurate text, i.e. hallucinations. In this work, we propose Cross-Layer Attention Probing (CLAP), a novel activation probing technique for hallucination detection, which processes the LLM activations across the entire residual stream as a joint sequence. Our empirical evaluations using five LLMs and three tasks show that CLAP improves hallucination detection compared to baselines on both greedy decoded responses as well as responses sampled at higher temperatures, thus enabling fine-grained detection, i.e. the ability to disambiguate hallucinations and non-hallucinations among different sampled responses to a given prompt. This allows us to propose a detect-then-mitigate strategy using CLAP to reduce hallucinations and improve LLM reliability compared to direct mitiga
    
[^217]: 从词元推测大语言模型的中文训练数据污染

    Speculating LLMs' Chinese Training Data Pollution from Their Tokens

    [https://arxiv.org/abs/2508.17771](https://arxiv.org/abs/2508.17771)

    该论文首次对大语言模型词表中的受污染中文词元给出形式化定义与分类，通过微调LLM构建词元检测器，并利用受污染词元的出现情况推测模型中文训练数据的污染程度，实验发现GPT词表受污染最为严重。

    

    词元是大语言模型训练数据集中的基本元素。众所周知，在GPT（4o/4o-mini/o1/o3/4.5/4.1/o4-mini）的词表中，许多表示中文短语的词元都指向色情或网络赌博等内容。基于这一观察，我们的目标是在大语言模型中定位受污染中文词元，并研究这类词元的存在与训练数据之间的关系。（1）我们基于GPT的词表给出了受污染中文词元的形式化定义和分类体系。（2）我们通过微调一个大语言模型构建了受污染中文词元检测器，综合考虑每个词元的语义以及来自搜索引擎的相关内容，对词表中的受污染中文词元进行标注。（3）我们研究了如何通过受污染中文词元的出现情况（词元ID）来推测训练数据是否受到污染。在GPT及其他23个大语言模型上的实验表明，这类词元广泛存在，而GPT的词表表现最差：超过40%的较长中文词元（即包含……

    arXiv:2508.17771v3 Announce Type: replace  Abstract: Tokens are basic elements in the datasets for LLM training. It is well-known that many tokens representing Chinese phrases in the vocabulary of GPT (4o/4o-mini/o1/o3/4.5/4.1/o4-mini) are indicating contents like pornography or online gambling. Based on this observation, our goal is to locate Polluted Chinese (PoC) tokens in LLMs and study the relationship between PoC tokens' existence and training data. (1) We give a formal definition and taxonomy of PoC tokens based on the GPT's vocabulary. (2) We build a PoC token detector via fine-tuning an LLM to label PoC tokens in vocabularies by considering each token's both semantics and related contents from the search engines. (3) We study the speculation on the training data pollution via PoC tokens' appearances (token ID). Experiments on GPT and other 23 LLMs indicate that tokens widely exist while GPT's vocabulary behaves the worst: more than 40% long Chinese tokens (i.e., a token with m
    
[^218]: 神经网络抽象解释器的自动张量级执行

    Automatic Tensor-level Execution of Abstract Interpreters for Neural Networks

    [https://arxiv.org/abs/2507.20055](https://arxiv.org/abs/2507.20055)

    Neuton框架能够自动将神经元级别的DNN认证器规范转换为张量级可执行代码，使研究者无需手动张量编程即可开发新的神经网络认证器。

    

    开发新的基于抽象解释的深度神经网络（DNN）认证器十分困难，因为其数学设计是在神经元级别表达的，而实现必须在张量级别进行。现有的库（如auto_LiRPA、IVAN和ERAN）仅支持固定的认证器集合；扩展它们需要形式化方法和张量编程方面的专业知识。诸如ConstraintFlow等领域特定语言虽然允许研究人员在神经元级别规范认证器，但无法在真实的DNN上运行。我们提出了Neuton，这是一个能够将神经元级别的认证器规范自动转换为张量级可执行程序的框架，使得无需手动编写张量程序即可开发新的DNN认证器。Neuton做出了三项贡献。首先，基于栈的中间表示（IR）和形状分析能够推断出正确执行所需的隐式张量操作，包括广播、维度扩展和规约。其次，特定领域（原文摘要在此处不完整）

    arXiv:2507.20055v2 Announce Type: replace  Abstract: Developing new abstract interpretation-based DNN certifiers is difficult because their mathematical design is expressed at the neuron level, while implementations must operate at the tensor level. Existing libraries such as auto_LiRPA, IVAN, and ERAN support fixed sets of certifiers; extending them requires expertise in formal methods and tensor programming. Domain-specific languages such as ConstraintFlow let researchers specify certifiers at the neuron level, but cannot run them on real DNNs.   We present Neuton, a framework that automatically translates neuron-level certifier specifications into tensor-level executables, enabling new DNN certifiers without manual tensor programming. Neuton makes three contributions. First, a stack-based intermediate representation (IR) and shape analysis infer implicit tensor operations, including broadcasts, dimension expansions, and reductions, required for correct execution. Second, domain-spec
    
[^219]: 基于约束知识遗忘的安全对齐

    Safety Alignment via Constrained Knowledge Unlearning

    [https://arxiv.org/abs/2505.18588](https://arxiv.org/abs/2505.18588)

    提出约束知识遗忘（CKU）安全对齐策略，通过定位并保留与有用知识相关的神经元、裁剪其梯度来精准遗忘有害知识，从而在不损害模型整体性能的情况下有效防御越狱攻击。

    

    尽管安全对齐已取得显著进展，大语言模型（LLM）仍然容易受到越狱攻击。现有的防御机制并未完全删除LLM中的有害知识，这使得此类攻击能够绕过安全防护并产生有害输出。为应对这一挑战，我们提出了一种新颖的安全对齐策略——约束知识遗忘（CKU），该策略聚焦于两个主要目标：知识的定位与保留，以及遗忘有害知识。CKU通过对特定多层感知机（MLP）层中的神经元进行评分，识别出与有用知识相关的神经元子集U。在遗忘过程中，CKU对U中神经元的梯度进行裁剪，以在保留有价值知识的同时有效消除有害内容。实验结果表明，CKU在不损害整体性能的前提下显著提升了模型安全性，实现了更优的平衡。

    arXiv:2505.18588v2 Announce Type: replace-cross  Abstract: Despite significant progress in safety alignment, large language models (LLMs) remain susceptible to jailbreak attacks. Existing defense mechanisms have not fully deleted harmful knowledge in LLMs, which allows such attacks to bypass safeguards and produce harmful outputs. To address this challenge, we propose a novel safety alignment strategy, Constrained Knowledge Unlearning (CKU), which focuses on two primary objectives: knowledge localization and retention, and unlearning harmful knowledge. CKU works by scoring neurons in specific multilayer perceptron (MLP) layers to identify a subset U of neurons associated with useful knowledge. During the unlearning process, CKU prunes the gradients of neurons in U to preserve valuable knowledge while effectively mitigating harmful content. Experimental results demonstrate that CKU significantly enhances model safety without compromising overall performance, offering a superior balance 
    
[^220]: 两心智胜过一心：面向大语言模型对齐的协作式奖励建模

    Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment

    [https://arxiv.org/abs/2505.10597](https://arxiv.org/abs/2505.10597)

    提出协作式奖励建模（CRM）框架，通过两个奖励模型跨模型互评结合课程学习来识别可靠的偏好数据，从而有效缓解噪声偏好导致的奖励误泛化并提升大语言模型的对齐效果。

    

    奖励模型（RMs）对于将大语言模型（LLMs）与人类价值观对齐至关重要，然而噪声偏好反馈可能通过虚假关联和过拟合引发奖励的误泛化。我们刻画了偏好对的训练动态，并揭示了噪声的独特优化特征：持续的高损失、剧烈的时间波动以及不规则的梯度更新。我们的分析进一步表明，排除此类样本对可以改善下游对齐效果。然而在实践中，可靠的偏好无法先验得知，而基于模型自身训练信号的自过滤可能会加剧确认偏误。为应对这一挑战，我们提出了协作式奖励建模，这是一种用于识别可靠偏好的在线框架。CRM将跨模型互评与课程学习相结合：两个奖励模型相互评估对方的样本选择，同时课程学习保持它们的学习进度。

    arXiv:2505.10597v3 Announce Type: replace-cross  Abstract: Reward models (RMs) are central to aligning large language models (LLMs) with human values, yet noisy preference feedback can induce reward misgeneralization through spurious correlations and overfitting. We characterize the training dynamics of preference pairs and uncover a distinctive optimization signature of noise: persistently high losses, sharp temporal fluctuations, and irregular gradient updates. Our analyses further show that excluding such pairs improves downstream alignment. In practice, however, reliable preferences are not known a priori, and self-filtering based on a model's own training signals can amplify confirmation bias. To address this challenge, we introduce Collaborative Reward Modeling (CRM), an online framework for identifying reliable preferences. CRM couples cross-model peer review with curriculum learning: two RMs evaluate each other's sample selections, while curriculum keeps their learning progress
    
[^221]: 面向长文本问答的原子一致性偏好优化

    Atomic Consistency Preference Optimization for Long-Form Question Answering

    [https://arxiv.org/abs/2505.09039](https://arxiv.org/abs/2505.09039)

    提出自监督偏好优化方法ACPO，利用多个随机回复间的原子事实一致性信号构建训练数据对，无需依赖外部强模型或知识库即可显著提升大语言模型长文本问答的事实准确性。

    

    大型语言模型（LLMs）经常产生事实性幻觉——看似合理却错误的答案。一种常见的缓解策略是模型对齐，即通过在精选的（事实性、非事实性）数据对上训练来提高事实准确性。然而，这种方法通常依赖于更强大的模型（如GPT-4）或外部知识库来评估事实正确性，而这些资源并不总是可获取的。针对这一问题，我们提出了原子一致性偏好优化（ACPO），这是一种无需外部监督即可提升事实准确性的自监督偏好调优方法。ACPO利用原子一致性信号（即多个随机生成回复中各个原子事实的一致程度）来识别用于模型对齐的高质量和低质量数据对。尽管完全自监督，ACPO在LongFact和BioGen数据集上，于Phi-3和Llama3两个模型上的平均表现仍超过强监督对齐基线1.95个百分点。

    arXiv:2505.09039v3 Announce Type: replace  Abstract: Large Language Models (LLMs) often produce factoid hallucinations - plausible yet incorrect answers. A common mitigation strategy is model alignment, which improves factual accuracy by training on curated (factual, non-factual) pairs. However, this approach often relies on a stronger model (e.g., GPT-4) or an external knowledge base to assess factual correctness that may not always be accessible. Addressing this, we propose Atomic Consistency Preference Optimization (ACPO), a self-supervised preference-tuning method that enhances factual accuracy without external supervision. ACPO leverages atomic consistency signals (i.e., the agreement of individual facts across multiple stochastic responses) to identify high- and low-quality data pairs for model alignment. Despite being fully self-supervised, ACPO outperforms the strong supervised alignment baseline by 1.95 points averaged across Phi-3 and Llama3 on the LongFact and BioGen dataset
    
[^222]: 优化奇异谱以实现大语言模型压缩

    Optimizing Singular Spectrum for Large Language Model Compression

    [https://arxiv.org/abs/2502.15092](https://arxiv.org/abs/2502.15092)

    提出SoCo压缩框架，通过可学习的对角矩阵以数据驱动方式重新缩放SVD分解组件的重要性分数，并采用三阶段训练过程，在大语言模型压缩与性能保持之间实现有效平衡。

    

    大语言模型（LLMs）已展现出卓越的能力，但其令人望而却步的参数复杂度常常阻碍了模型的部署。现有的基于奇异值分解（SVD）的压缩方法简单地将奇异值视为分解组件的重要性分数。然而，这种由奇异值排序的重要性并不一定与下游任务的性能相关。在本工作中，我们提出了SoCo（面向大语言模型压缩的奇异谱优化），这是一种新颖的压缩框架，能够以数据驱动的方式学习对SVD分解组件进行重新缩放。具体而言，我们采用可学习的对角矩阵为奇异谱分配重要性分数，并开发了一个三阶段训练过程，从初始的粗粒度压缩到细粒度稀疏化逐步精炼这些分数——从而在激进模型压缩与性能保持之间取得有效平衡。

    arXiv:2502.15092v2 Announce Type: replace  Abstract: Large language models (LLMs) have demonstrated remarkable capabilities, yet prohibitive parameter complexity often hinders their deployment. Existing singular value decomposition (SVD) based compression methods simply deem singular values as importance scores of decomposed components. However, this importance ordered by singular values does not necessarily correlate with the performance of a downstream task. In this work, we introduce SoCo (Singular spectrum optimization for large language model Compression), a novel compression framework that learns to rescale the decomposed components of SVD in a data-driven manner. Concretely, we employ a learnable diagonal matrix to assign importance scores for singular spectrum and develop a three-stage training process that progressively refines these scores from initial coarse compression to fine-grained sparsification-thereby striking an effective balance between aggressive model compression 
    
[^223]: 通过主动自我反思学习在知识图谱上进行检索与推理

    Learning to Retrieve and Reason on Knowledge Graph through Active Self-Reflection

    [https://arxiv.org/abs/2502.14932](https://arxiv.org/abs/2502.14932)

    本文提出ArG框架，首次通过端到端训练方法，利用特殊标记让模型主动控制推理过程，实现知识图谱上的按需检索与多维度细粒度知识评估的迭代推理。

    

    大量研究已经探索了将大语言模型（LLM）与知识图谱相结合以增强推理过程的方法。然而，对于模型如何利用结构化图知识进行推理，这一问题的理解仍然不够深入。大多数现有方法依赖LLM或检索器对知识的利用做出二元判断，这种方式过于粗糙。同时，在整个推理路径中仍然缺乏用于反思和控制的反馈机制。本文提出了一种面向知识图谱推理的主动自我反思框架，首次引入端到端训练方法，实现基于结构化图谱的迭代推理。在该框架内，模型利用特殊标记主动控制推理过程，实现按需检索和多维度、更细粒度的知识评估。所生成的推理路径……

    arXiv:2502.14932v2 Announce Type: replace  Abstract: Extensive research has investigated the integration of large language models (LLMs) with knowledge graphs to enhance the reasoning process. However, understanding how models perform reasoning utilizing structured graph knowledge remains underexplored. Most existing approaches rely on LLMs or retrievers to make binary judgments regarding the utilization of knowledge, which is too coarse. Meanwhile, there is still a lack of feedback mechanisms for reflection and control throughout the entire reasoning path. This paper proposes an Active self-Reflection framework for knowledge Graph reasoning (ArG), introducing for the first time an end-to-end training approach to achieve iterative reasoning grounded on structured graphs. Within the framework, the model leverages special tokens to actively control the reasoning process, achieving on-demand retrieval and multi-dimensional, finer-grained knowledge evaluation. The reasoning paths generated
    
[^224]: 基于多样化动态表达的多目标指代跟踪

    Referring Multi-Object Tracking with Diverse Dynamic Expressions

    [https://arxiv.org/abs/2406.05039](https://arxiv.org/abs/2406.05039)

    该论文提出了一种新的指代多目标跟踪（RMOT）任务，利用语言表达作为语义线索指导多目标跟踪预测，全面考虑对象数量与时间语义的动态变化，并发布了具有多样化语言表达的基准数据集Refer-KITTI-V2及高效的半自动标注流水线。

    

    指代理解是一项基础任务，通过定位自由形式语言表达所描述的对象，将自然语言与视觉内容联系起来。然而，现有工作受限于有限的语言表达能力，缺乏对对象在空间数量和时间状态方面动态变化的建模能力。为了解决这些局限性，我们提出了一种新颖且通用的指代理解任务，称为指代多目标跟踪（RMOT）。其核心思想是利用语言表达作为语义线索来指导多目标跟踪的预测，全面考虑对象数量变化和时间语义。与RMOT任务一同，我们推出了一个名为Refer-KITTI-V2的RMOT基准数据集，其特点是具有可扩展且多样化的语言表达。为了以最少的人工成本高效生成涵盖对象动态的高质量标注，我们提出了一种半自动标注流水线，该流水线构建了……

    arXiv:2406.05039v3 Announce Type: replace-cross  Abstract: Referring understanding is a fundamental task that bridges natural language and visual content by localizing objects described in free-form expressions. However, existing works are constrained by limited language expressiveness, lacking the capacity to model object dynamics in spatial numbers and temporal states. To address these limitations, we introduce a new and general referring understanding task, termed referring multi-object tracking (RMOT). Its core idea is to employ a language expression as a semantic cue to guide the prediction of multi-object tracking, comprehensively accounting for variations in object quantity and temporal semantics. Along with RMOT, we introduce a RMOT benchmark named Refer-KITTI-V2, featuring scalable and diverse language expressions. To efficiently generate high-quality annotations covering object dynamics with minimal manual effort, we propose a semi-automatic labeling pipeline that formulates 
    
[^225]: VietMed：面向医疗领域越南语自动语音识别的数据集与基准

    VietMed: A Dataset and Benchmark for Automatic Speech Recognition of Vietnamese in the Medical Domain

    [https://arxiv.org/abs/2404.05659](https://arxiv.org/abs/2404.05659)

    VietMed 是迄今全球最大的公开医疗领域越南语语音识别数据集，并首次提供了覆盖所有 ICD-10 疾病组和全国所有口音的医疗 ASR 数据集，以及首个越南语 ASR 公开大规模预训练模型。

    

    由于隐私限制，医疗领域公开可用的语音识别数据集十分匮乏。在本工作中，我们提出了 VietMed——一个医疗领域的越南语语音识别数据集，包含16小时带标注的医疗语音、1000小时无标注医疗语音以及1200小时无标注通用领域语音。据我们所知，VietMed 在七个方面是迄今全球最大的公开医疗语音识别数据集：总时长、说话人数量、疾病种类、录音条件、说话人角色、独特医学术语数量和口音种类。VietMed 也是迄今总时长最大的公开越南语语音数据集。此外，我们是首个提出覆盖一个国家内所有 ICD-10 疾病组和所有口音的医疗 ASR 数据集的团队。同时，我们发布了首个面向越南语 ASR 的公开大规模预训练模型 w2v2-Viet 和 XLSR-53-Viet，以及首个……

    arXiv:2404.05659v4 Announce Type: replace-cross  Abstract: Due to privacy restrictions, there's a shortage of publicly available speech recognition datasets in the medical domain. In this work, we present VietMed - a Vietnamese speech recognition dataset in the medical domain comprising 16h of labeled medical speech, 1000h of unlabeled medical speech and 1200h of unlabeled general-domain speech. To our best knowledge, VietMed is by far the world's largest public medical speech recognition dataset in 7 aspects: total duration, number of speakers, diseases, recording conditions, speaker roles, unique medical terms and accents. VietMed is also by far the largest public Vietnamese speech dataset in terms of total duration. Additionally, we are the first to present a medical ASR dataset covering all ICD-10 disease groups and all accents within a country. Moreover, we release the first public large-scale pre-trained models for Vietnamese ASR, w2v2-Viet and XLSR-53-Viet, along with the first 
    
[^226]: 超越边界：AI系统可迁移攻击的综合综述

    Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems

    [https://arxiv.org/abs/2311.11796](https://arxiv.org/abs/2311.11796)

    本综述提出了一个涵盖跨实例、跨领域、跨模态、跨模型、跨任务和跨部署六个维度的统一分类体系，为比较AI系统中各类可迁移攻击的迁移机制与失效条件提供了通用框架。

    

    可迁移攻击使得在一个环境中构建的对抗性操纵或学习到的攻击机制能够危害其他AI系统，从而可能降低攻击成本并加剧安全与隐私风险。然而，碎片化的术语和评估实践阻碍了人们对“什么可以迁移、为什么可以迁移、迁移在何处失效”的理解。本综述综合了代表性研究，构建了一个涵盖六个互补维度的统一分类体系：跨实例、跨领域、跨模态、跨模型、跨任务和跨部署。通过将源-目标配置的变化与攻击者的访问权限和自适应预算相分离，该分类体系为比较逃逸、投毒、后门、隐私推断、模型窃取以及基于提示的攻击中的迁移现象提供了统一的框架。我们关联了梯度对齐、共享表示、训练持久性以及部署（摘要在此处截断）等迁移机制……

    arXiv:2311.11796v3 Announce Type: replace-cross  Abstract: Transferable attacks allow adversarial manipulations or learned attack mechanisms constructed in one setting to compromise other AI systems, potentially reducing attack costs and amplifying security and privacy risks. Yet fragmented terminology and evaluation practices hinder understanding of what transfers, why it transfers, and where transfer fails. This survey synthesizes representative research and develops a unified taxonomy spanning six complementary axes: cross-instance, cross-domain, cross-modality, cross-model, cross-task, and cross-deployment. By separating source-target configuration changes from attacker access and adaptation budgets, the taxonomy provides a common framework for comparing transfer across evasion, poisoning, backdoors, privacy inference, model stealing, and prompt-based attacks. We connect transfer mechanisms, including gradient alignment, shared representations, training persistence, and deployment 
    

